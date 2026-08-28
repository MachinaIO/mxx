//! Production-boundary adapter for the operational-noise arenas.
//!
//! This module consumes the real frozen Graph IR and the occurrence-aware [`ProtocolPlan`].
//! It deliberately has no synthetic lowering graph: structural nodes are interpreted through
//! the plan's aliases and output mappings, while ordinary nodes are interned directly.

use super::{
    arena::{
        ConstantValue, DeterministicHashDefinition, DeterministicHashDescriptor, ExprId, ExprNode,
        FamilyDomain, MatrixConstantKind, MatrixLayout, MatrixOperation, ProgramInput,
        ProgramSignature, ResolvedMatrixType, ResolvedValueType, SampleEventId, SamplerOperation,
        ScalarOperation, SemanticFamilySourceIdentity, SemanticSourceIdentity, TrapdoorOperation,
        TrustedIndexRange, TypedConstant, ValueOperator, ValueTransformOperation,
    },
    facts::{
        CoefficientBound, MatrixFacts, MatrixMetadata, NumericContract, PolynomialFacts, ValueFacts,
    },
    job::{CandidateToken, CheckerJob, JobError},
    normal_form::CompactShellPlan,
    program::{
        BetaReason, FamilyValueId, ProgramDiagnosticCounters, SelectionSelector, ValueProgramId,
    },
    protocol::{PlannedNode, PlannedWire, ProgramOccurrence, ProtocolPlan},
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
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt::Write,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};
use thiserror::Error;
use tracing::{debug, info};

#[derive(Clone, Debug, Eq, PartialEq)]
struct RelationCandidate {
    preimage: ExprId,
    public: ExprId,
    trapdoor: ExprId,
    target: ExprId,
    family_operands: Option<(FamilyValueId, FamilyValueId, FamilyValueId, FamilyValueId)>,
    wire: PlannedWire,
}

/// The parent information which can affect compact authorization. Ordinary algebra parents do
/// not participate in state identity; exact product and fixed-slice use sites do. This is what
/// lets the compiler merge shared ordinary DAG subgraphs without merging token-bearing uses.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum CompactCompilerParent {
    Ordinary,
    Product { consumer: ExprId, is_right: bool },
    FixedSliceChild { slice: ExprId },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct CompactCompilerStateKey {
    expression: ExprId,
    owner: Option<ValueProgramId>,
    /// The interned call expression which supplies the current binding environment.  A body
    /// may be shared by calls with different arguments, so the body occurrence cannot merge
    /// across those environments even when its owner and local expression are identical.
    binding_environment: Option<ExprId>,
    parent: CompactCompilerParent,
    under_planned_shell: bool,
    scalar_call_context: Option<ExprId>,
    binding_context: Option<(ResolvedValueType, Option<TrustedIndexRange>)>,
    binding_subtree: bool,
}

#[derive(Clone, Debug)]
enum CompactCompilerPlanDelta {
    Gadget {
        shell: ExprId,
        input: ExprId,
        rule: GadgetRecompositionRule,
    },
    Scalar {
        expression: ExprId,
        value_type: ResolvedMatrixType,
    },
    ScalarProgramCall {
        consumer: ExprId,
        call: ExprId,
        scalar_is_right: bool,
        value_type: ResolvedMatrixType,
    },
}

struct CompactCompilerState {
    key: CompactCompilerStateKey,
    children: BTreeMap<usize, u64>,
    multiplicity: u64,
    virtual_node: bool,
    plan_delta: Option<CompactCompilerPlanDelta>,
}

fn enqueue_compact_compiler_state(
    key: CompactCompilerStateKey,
    parent: Option<usize>,
    states: &mut Vec<CompactCompilerState>,
    state_ids: &mut BTreeMap<CompactCompilerStateKey, usize>,
    active: &BTreeSet<usize>,
    work: &mut Vec<(usize, bool)>,
) -> Result<usize, String> {
    let child = if let Some(child) = state_ids.get(&key).copied() {
        if active.contains(&child) {
            return Err("cycle or recursive generated body".to_owned());
        }
        child
    } else {
        let child = states.len();
        state_ids.insert(key.clone(), child);
        states.push(CompactCompilerState {
            key,
            children: BTreeMap::new(),
            multiplicity: 0,
            virtual_node: false,
            plan_delta: None,
        });
        work.push((child, false));
        child
    };
    if let Some(parent) = parent {
        let entry = states[parent].children.entry(child).or_insert(0);
        *entry = entry
            .checked_add(1)
            .ok_or_else(|| "compact occurrence multiplicity overflow".to_owned())?;
    }
    Ok(child)
}

#[derive(Clone, Copy, Debug, Default)]
struct IndexRangeProjectionStats {
    nodes: u64,
    program_calls: u64,
}

#[derive(Clone, Debug)]
struct IndexRangeProjectionBinding {
    expression: ExprId,
    value_type: ResolvedValueType,
    range: (BigInt, BigInt),
}

struct IndexRangeProjector<'p, 'a> {
    adapter: &'p ProductionAdapter<'a>,
    environments: Vec<Box<[IndexRangeProjectionBinding]>>,
    memo: HashMap<(ExprId, usize), Option<(BigInt, BigInt)>>,
    active: BTreeSet<(ExprId, usize)>,
    stats: IndexRangeProjectionStats,
}

impl<'p, 'a> IndexRangeProjector<'p, 'a> {
    fn new(adapter: &'p ProductionAdapter<'a>) -> Self {
        Self {
            adapter,
            environments: Vec::new(),
            memo: HashMap::new(),
            active: BTreeSet::new(),
            stats: IndexRangeProjectionStats::default(),
        }
    }

    fn push_environment(&mut self, bindings: Vec<IndexRangeProjectionBinding>) -> usize {
        let id = self.environments.len();
        self.environments.push(bindings.into_boxed_slice());
        id
    }

    fn supports_integer_operator(operator: &ValueOperator) -> bool {
        matches!(
            operator,
            ValueOperator::ExtractCoefficient { .. } |
                ValueOperator::Scalar(
                    ScalarOperation::Add |
                        ScalarOperation::Subtract |
                        ScalarOperation::Multiply |
                        ScalarOperation::Divide |
                        ScalarOperation::Remainder
                )
        )
    }

    fn evaluate(&mut self, expression: ExprId, environment: usize) -> Option<(BigInt, BigInt)> {
        let key = (expression, environment);
        if let Some(result) = self.memo.get(&key) {
            return result.clone();
        }
        if !self.active.insert(key) {
            return None;
        }
        self.stats.nodes = self.stats.nodes.saturating_add(1);
        let result = self.evaluate_uncached(expression, environment);
        self.active.remove(&key);
        self.memo.insert(key, result.clone());
        result
    }

    fn evaluate_uncached(
        &mut self,
        expression: ExprId,
        environment: usize,
    ) -> Option<(BigInt, BigInt)> {
        let node = self.adapter.job.expressions().node(expression).ok()?.clone();
        match node.operator {
            ValueOperator::Constant(TypedConstant { value: ConstantValue::Int(value), .. }) => {
                Some((value.clone(), value + BigInt::from(1_u8)))
            }
            ValueOperator::Argument { position, value_type } => {
                let bindings = self.environments.get(environment)?;
                let binding = bindings
                    .get(position as usize)
                    .or_else(|| bindings.iter().find(|binding| binding.expression == expression))?;
                (&binding.value_type == &value_type).then(|| binding.range.clone())
            }
            ValueOperator::ExtractCoefficient { position, canonical_input_exclusive_upper } => {
                if node.inputs.len() != 1 {
                    return None;
                }
                let ResolvedValueType::Matrix(matrix) =
                    self.adapter.job.expressions().value_type(node.inputs[0]).ok()?.clone()
                else {
                    return None;
                };
                let element_count = matrix.rows.checked_mul(matrix.columns)?;
                if position as usize >= element_count {
                    return None;
                }
                let upper = canonical_input_exclusive_upper?;
                if upper.is_zero() || upper > matrix.modulus {
                    return None;
                }
                Some((BigInt::from(0_u8), BigInt::from(upper)))
            }
            ValueOperator::ProgramCall { program } => {
                self.stats.program_calls = self.stats.program_calls.saturating_add(1);
                let (signature, body) = {
                    let program = self.adapter.job.programs().program(program).ok()?;
                    (program.signature.clone(), program.root)
                };
                let family = self.adapter.job.programs().family_for_program(program)?;
                if !self.adapter.job.programs().family_is_reducible(family).ok()? {
                    return None;
                }
                if self.adapter.job.expressions().value_type(body).ok()? != &signature.output {
                    return None;
                }
                if node.inputs.len() != signature.inputs.len() {
                    return None;
                }
                let mut bindings = Vec::with_capacity(node.inputs.len());
                for (input, expected) in node.inputs.iter().copied().zip(signature.inputs.iter()) {
                    if self.adapter.job.expressions().value_type(input).ok()? !=
                        &expected.value_type
                    {
                        return None;
                    }
                    let range = self.evaluate(input, environment)?;
                    bindings.push(IndexRangeProjectionBinding {
                        expression: input,
                        value_type: expected.value_type.clone(),
                        range,
                    });
                }
                let body_environment = self.push_environment(bindings);
                self.evaluate(body, body_environment)
            }
            ValueOperator::Scalar(operation) => {
                let [left, right] = node.inputs.as_ref() else {
                    return None;
                };
                let left_range = self.evaluate(*left, environment)?;
                let right_range = self.evaluate(*right, environment)?;
                let one = BigInt::from(1_u8);
                match operation {
                    ScalarOperation::Add => Some((
                        &left_range.0 + &right_range.0,
                        (&left_range.1 - &one) + (&right_range.1 - &one) + &one,
                    )),
                    ScalarOperation::Subtract => Some((
                        &left_range.0 - (&right_range.1 - &one),
                        (&left_range.1 - &one) - &right_range.0 + &one,
                    )),
                    ScalarOperation::Multiply => {
                        if let Some(factor) = self.constant_integer(*right) {
                            Some(multiply_open_range(left_range.0, left_range.1, factor))
                        } else if let Some(factor) = self.constant_integer(*left) {
                            Some(multiply_open_range(right_range.0, right_range.1, factor))
                        } else {
                            None
                        }
                    }
                    ScalarOperation::Divide => {
                        let divisor = self.constant_integer(*right)?;
                        if divisor.is_zero() ||
                            divisor <= BigInt::from(0_u8) ||
                            left_range.0 < BigInt::from(0_u8)
                        {
                            return None;
                        }
                        Some((
                            &left_range.0 / &divisor,
                            (&left_range.1 - &one + &divisor) / &divisor,
                        ))
                    }
                    ScalarOperation::Remainder => {
                        let divisor = self.constant_integer(*right)?;
                        remainder_open_range(left_range.0, left_range.1, divisor)
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn constant_integer(&self, expression: ExprId) -> Option<BigInt> {
        let node = self.adapter.job.expressions().node(expression).ok()?;
        let ValueOperator::Constant(TypedConstant { value: ConstantValue::Int(value), .. }) =
            &node.operator
        else {
            return None;
        };
        Some(value.clone())
    }
}

fn format_beta_reason_snapshot(counters: &ProgramDiagnosticCounters) -> String {
    let mut snapshot = String::new();
    let mut has_reason = false;
    let mut miss_sum = 0_u64;
    let mut visit_sum = 0_u64;
    let mut allocation_sum = 0_u64;
    for reason in BetaReason::ALL {
        let index = reason as usize;
        let misses = counters.beta_reason_misses[index];
        let visits = counters.beta_reason_visits[index];
        let allocations = counters.beta_reason_expr_allocations[index];
        miss_sum = miss_sum.saturating_add(misses);
        visit_sum = visit_sum.saturating_add(visits);
        allocation_sum = allocation_sum.saturating_add(allocations);
        if misses == 0 && visits == 0 && allocations == 0 {
            continue;
        }
        if has_reason {
            snapshot.push(';');
        }
        has_reason = true;
        write!(
            &mut snapshot,
            "{}:m={},v={},expr_allocations={}",
            reason.label(),
            misses,
            visits,
            allocations
        )
        .expect("writing to a String cannot fail");
    }
    if has_reason {
        snapshot.push(';');
    }
    write!(&mut snapshot, "sum:m={},v={},expr_allocations={}", miss_sum, visit_sum, allocation_sum)
        .expect("writing to a String cannot fail");
    snapshot
}

type ScopedExprKey = (ProgramOccurrence, ExprId);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProductionRoot {
    Closed(super::arena::ClosedExprId),
    /// A closed residual whose reducible generated calls were validated by the pre-freeze
    /// compact-root preflight.  This is an internal transport marker; it is never represented in
    /// the expression or program IR.
    Compact(super::arena::ClosedExprId),
    /// A reducible indexed family whose formal body passed the compact preflight.  This private
    /// marker keeps the family authority and its owned compact plan together without creating a
    /// synthetic family call or selecting one index.
    CompactFamily(FamilyValueId),
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
    #[error("adapter-local plan wire id is foreign or out of range: token={token}, slot={slot}")]
    InvalidPlanWireId { token: u64, slot: u32 },
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
enum MatrixFactProjection {
    Found(ExprId),
    ProvenAbsent,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NodeKindClass {
    Supported,
    Structural,
    TypedUnsupported,
}

static NEXT_PLAN_WIRE_TOKEN: AtomicU64 = AtomicU64::new(1);

fn allocate_plan_wire_token(source: &AtomicU64) -> Option<u64> {
    source.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| current.checked_add(1)).ok()
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct PlanWireId {
    token: u64,
    slot: u32,
}

#[derive(Clone, Debug)]
struct CompactPlannedNode {
    planned: Arc<PlannedNode>,
    arguments: Box<[PlanWireId]>,
}

struct PlanWireTable {
    token: u64,
    wires: Box<[Arc<PlannedWire>]>,
    ids: BTreeMap<PlannedWire, PlanWireId>,
    nodes: Box<[Option<Arc<CompactPlannedNode>>]>,
    occurrence_slots: BTreeMap<(StageId, ProgramOccurrence), Box<[PlanWireId]>>,
}

impl PlanWireTable {
    fn build(plan: &ProtocolPlan) -> Result<Self, ProductionAdapterError> {
        let token = allocate_plan_wire_token(&NEXT_PLAN_WIRE_TOKEN).ok_or_else(|| {
            ProductionAdapterError::Descriptor {
                reason: "plan wire identity space exhausted".to_owned(),
            }
        })?;
        let mut wires = BTreeSet::new();
        wires.extend(plan.nodes().keys().cloned());
        wires.insert(plan.target().residual.clone());
        wires.insert(plan.target().decoder.clone());
        for alias in plan.aliases() {
            wires.insert(alias.child.clone());
            wires.insert(alias.parent.clone());
        }
        for mapping in plan.output_mappings() {
            wires.insert(mapping.parent.clone());
            wires.insert(mapping.child.clone());
        }
        for producer in plan.artifact_producers() {
            wires.insert(producer.consumer.clone());
            wires.insert(producer.producer.clone());
        }
        for (owner, node) in plan.nodes() {
            wires.extend(node.arguments.iter().copied().map(|wire| PlannedWire {
                stage: owner.stage.clone(),
                occurrence: owner.occurrence.clone(),
                wire,
            }));
        }
        let wires = wires.into_iter().map(Arc::new).collect::<Vec<_>>().into_boxed_slice();
        let mut ids = BTreeMap::new();
        for (slot, wire) in wires.iter().enumerate() {
            let slot = u32::try_from(slot).map_err(|_| ProductionAdapterError::Descriptor {
                reason: "plan wire table exceeds u32 slots".to_owned(),
            })?;
            ids.insert(wire.as_ref().clone(), PlanWireId { token, slot });
        }
        let mut nodes = vec![None; wires.len()];
        for (wire, planned) in plan.nodes() {
            let id = ids
                .get(wire)
                .copied()
                .ok_or_else(|| ProductionAdapterError::MissingWire { wire: wire.clone() })?;
            let arguments = planned
                .arguments
                .iter()
                .copied()
                .map(|argument| {
                    let argument = PlannedWire {
                        stage: wire.stage.clone(),
                        occurrence: wire.occurrence.clone(),
                        wire: argument,
                    };
                    ids.get(&argument)
                        .copied()
                        .ok_or(ProductionAdapterError::MissingWire { wire: argument })
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_boxed_slice();
            nodes[id.slot as usize] =
                Some(Arc::new(CompactPlannedNode { planned: Arc::clone(planned), arguments }));
        }
        let mut occurrence_slots = BTreeMap::<_, Vec<_>>::new();
        for (slot, wire) in wires.iter().enumerate() {
            occurrence_slots
                .entry((wire.stage.clone(), wire.occurrence.clone()))
                .or_default()
                .push(PlanWireId { token, slot: slot as u32 });
        }
        Ok(Self {
            token,
            wires,
            ids,
            nodes: nodes.into_boxed_slice(),
            occurrence_slots: occurrence_slots
                .into_iter()
                .map(|(owner, slots)| (owner, slots.into_boxed_slice()))
                .collect(),
        })
    }

    fn id(&self, wire: &PlannedWire) -> Option<PlanWireId> {
        self.ids.get(wire).copied()
    }

    fn wire(&self, id: PlanWireId) -> Option<&PlannedWire> {
        (id.token == self.token)
            .then(|| self.wires.get(id.slot as usize))
            .flatten()
            .map(Arc::as_ref)
    }

    fn wire_arc(&self, id: PlanWireId) -> Option<Arc<PlannedWire>> {
        (id.token == self.token).then(|| self.wires.get(id.slot as usize)).flatten().cloned()
    }

    fn node(&self, id: PlanWireId) -> Option<Arc<CompactPlannedNode>> {
        (id.token == self.token)
            .then(|| self.nodes.get(id.slot as usize))
            .flatten()
            .and_then(|node| node.as_ref())
            .cloned()
    }
}

type OverrideEnv = Rc<BTreeMap<PlanWireId, Value>>;

/// State carried by the non-recursive parallel-loop continuation.  The parent arguments and
/// child inputs are walked one at a time so a loop body can contain arbitrarily deep structural
/// subgraphs without growing the Rust call stack.
struct ParallelState {
    wire: PlanWireId,
    spec: mxx_ir_core::node::ParallelLoop,
    overrides: OverrideEnv,
    planned_node: Arc<CompactPlannedNode>,
    domain: FamilyDomain,
    argument: ExprId,
    child_inputs: Box<[PlanWireId]>,
    child_outputs: Box<[PlanWireId]>,
    child_occurrence: super::protocol::ProgramOccurrence,
    next_input: usize,
    child_overrides: BTreeMap<PlanWireId, Value>,
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
    wire: PlanWireId,
    spec: mxx_ir_core::node::SequentialLoop,
    overrides: OverrideEnv,
    planned_node: Arc<CompactPlannedNode>,
    child_inputs: Box<[PlanWireId]>,
    child_outputs: Box<[PlanWireId]>,
    child_occurrence: super::protocol::ProgramOccurrence,
    carried: Vec<Value>,
    invariant: Vec<Value>,
    next_outputs: Vec<Value>,
    iteration_overrides: OverrideEnv,
    iteration: usize,
    count: usize,
    saved_loop_indices: BTreeMap<u32, BigInt>,
    saved_loop_arguments: BTreeMap<u32, ExprId>,
    saved_loop_argument_ranges: BTreeMap<ScopedExprKey, TrustedIndexRange>,
}

enum ResolveFrame {
    Resolve {
        wire: PlanWireId,
        overrides: OverrideEnv,
    },
    Lower {
        wire: PlanWireId,
        planned_node: Arc<CompactPlannedNode>,
        overrides: OverrideEnv,
        next: usize,
        inputs: Vec<Value>,
    },
    Store {
        wire: PlanWireId,
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

const RESOLVER_PROGRESS_INTERVAL: u64 = 1 << 20;

#[derive(Clone, Copy, Default)]
struct ResolverProgressSnapshot {
    total_frames: u64,
    cache_hits: u64,
    ordinary_lowers: u64,
    parallel_prepare: u64,
    parallel_body: u64,
    parallel_finish: u64,
    relation_empty_buckets: u64,
    relation_nonempty_buckets: u64,
}

struct ResolverProgress {
    enabled: bool,
    started_at: Option<Instant>,
    counters: ResolverProgressSnapshot,
}

impl ResolverProgress {
    fn new() -> Self {
        let enabled = tracing::enabled!(
            target: "mxx_correctness::operational_noise",
            tracing::Level::INFO
        );
        Self { enabled, started_at: None, counters: ResolverProgressSnapshot::default() }
    }

    fn start(&mut self) {
        if self.enabled {
            self.started_at = Some(Instant::now());
        }
    }

    fn observe_frame(&mut self, frame: &ResolveFrame) -> Option<ResolverProgressSnapshot> {
        if !self.enabled {
            return None;
        }
        self.counters.total_frames = self.counters.total_frames.saturating_add(1);
        match frame {
            ResolveFrame::ParallelPrepare { .. } => {
                self.counters.parallel_prepare = self.counters.parallel_prepare.saturating_add(1);
            }
            ResolveFrame::ParallelBody { .. } => {
                self.counters.parallel_body = self.counters.parallel_body.saturating_add(1);
            }
            ResolveFrame::ParallelFinish { .. } => {
                self.counters.parallel_finish = self.counters.parallel_finish.saturating_add(1);
            }
            _ => {}
        }
        (self.counters.total_frames % RESOLVER_PROGRESS_INTERVAL == 0).then_some(self.counters)
    }

    fn record_cache_hit(&mut self) {
        if self.enabled {
            self.counters.cache_hits = self.counters.cache_hits.saturating_add(1);
        }
    }

    fn record_ordinary_lower(&mut self) {
        if self.enabled {
            self.counters.ordinary_lowers = self.counters.ordinary_lowers.saturating_add(1);
        }
    }

    fn record_relation_bucket(&mut self, nonempty: bool) {
        if !self.enabled {
            return;
        }
        if nonempty {
            self.counters.relation_nonempty_buckets =
                self.counters.relation_nonempty_buckets.saturating_add(1);
        } else {
            self.counters.relation_empty_buckets =
                self.counters.relation_empty_buckets.saturating_add(1);
        }
    }

    fn elapsed(&self) -> Option<std::time::Duration> {
        self.started_at.map(|started_at| started_at.elapsed())
    }
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
        NodeKind::MatrixMulAccumulate { .. } |
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
pub(crate) struct ProductionAdapter<'a> {
    plan: &'a ProtocolPlan,
    wire_table: PlanWireTable,
    graphs: BTreeMap<StageId, &'a Graph>,
    params: ParamEnv,
    pub(crate) job: CheckerJob,
    token: CandidateToken,
    values: Vec<Option<Value>>,
    value_count: usize,
    aliases: HashMap<PlanWireId, PlanWireId>,
    outputs: HashMap<PlanWireId, PlanWireId>,
    artifacts: HashMap<PlanWireId, PlanWireId>,
    protocol_inputs: BTreeMap<(StageId, StageInputName), ProtocolInputId>,
    input_contracts: BTreeMap<ProtocolInputId, &'a InputValueContract>,
    sample_events: BTreeMap<SampleKey, SampleEventId>,
    static_indices: HashMap<PlannedWire, ExprId>,
    active_loop_indices: BTreeMap<u32, BigInt>,
    active_loop_arguments: BTreeMap<u32, ExprId>,
    active_loop_argument_ranges: BTreeMap<ScopedExprKey, TrustedIndexRange>,
    active_parallel_depth: usize,
    generated_family_calls: u64,
    matrix_fact_projection_direct_hits: u64,
    matrix_fact_projection_closed_root_hits: u64,
    matrix_fact_projection_argument_hits: u64,
    matrix_fact_projection_sidecar_hits: u64,
    matrix_fact_projection_proven_absent: u64,
    matrix_fact_projection_fallbacks: u64,
    matrix_fact_projection_fallback_beta_nodes: u64,
    matrix_select_open_observation_skips: u64,
    matrix_select_open_observation_skipped_branches: u64,
    selector_range_compact_direct_hits: u64,
    selector_range_projected_program_call_hits: u64,
    selector_range_fallback_materializations: u64,
    selector_range_projector_nodes: u64,
    selector_range_projector_program_calls: u64,
    relation_candidates: Vec<RelationCandidate>,
    relation_candidate_indices: BTreeMap<ProgramOccurrence, Vec<usize>>,
    gadget_decompositions: BTreeMap<ExprId, (ExprId, u64, bool, u32)>,
    trapdoor_values: BTreeMap<SampleKey, ExprId>,
    occurrence_descendants: BTreeMap<(StageId, ProgramOccurrence), BTreeSet<ProgramOccurrence>>,
    diagnostic_budget: u16,
    resolver_progress: ResolverProgress,
    #[cfg(test)]
    test_sampler_fact_bound: Option<u64>,
}

impl<'a> ProductionAdapter<'a> {
    fn compact_wire(&self, wire: &PlannedWire) -> Result<PlanWireId, ProductionAdapterError> {
        self.wire_table
            .id(wire)
            .ok_or_else(|| ProductionAdapterError::MissingWire { wire: wire.clone() })
    }

    fn full_wire(&self, id: PlanWireId) -> Result<&PlannedWire, ProductionAdapterError> {
        self.wire_table
            .wire(id)
            .ok_or(ProductionAdapterError::InvalidPlanWireId { token: id.token, slot: id.slot })
    }

    fn full_wire_arc(&self, id: PlanWireId) -> Result<Arc<PlannedWire>, ProductionAdapterError> {
        self.wire_table
            .wire_arc(id)
            .ok_or(ProductionAdapterError::InvalidPlanWireId { token: id.token, slot: id.slot })
    }

    fn immediate_value(
        &self,
        id: PlanWireId,
        overrides: &OverrideEnv,
    ) -> Result<Option<Value>, ProductionAdapterError> {
        // Compact authority is checked before every lookup, including an override hit.  An
        // adapter-local override must never turn a stale/foreign numeric ID into a valid wire.
        self.full_wire(id)?;
        Ok(overrides.get(&id).copied().or(self.values[id.slot as usize]))
    }

    fn store_value(&mut self, id: PlanWireId, value: Value) -> Result<(), ProductionAdapterError> {
        self.full_wire(id)?;
        let slot = &mut self.values[id.slot as usize];
        if slot.is_none() {
            self.value_count = self.value_count.saturating_add(1);
        }
        *slot = Some(value);
        Ok(())
    }

    fn compact_child_wire(
        &self,
        stage: &StageId,
        occurrence: &ProgramOccurrence,
        wire: WireRef,
    ) -> Result<PlanWireId, ProductionAdapterError> {
        self.compact_wire(&PlannedWire {
            stage: stage.clone(),
            occurrence: occurrence.clone(),
            wire,
        })
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
        self.intern_node_operator_with_expected_output(
            wire,
            &expected_output,
            operator,
            inputs,
            check_output,
        )
    }

    fn intern_node_operator_with_expected_output(
        &mut self,
        wire: &PlannedWire,
        expected_output: &ResolvedValueType,
        operator: ValueOperator,
        inputs: Box<[ExprId]>,
        check_output: bool,
    ) -> Result<ExprId, ProductionAdapterError> {
        // `ExprArena::intern` consumes the input slice even when validation fails. Retain only
        // the cheap IDs on the error path; resolving and cloning their potentially large types
        // is deferred until an actionable diagnostic is actually needed.
        let input_ids = inputs.to_vec();
        let operator_for_error = operator.clone();
        let result = self.job.expressions_mut().intern(operator, inputs);
        let expression = match result {
            Ok(expression) => expression,
            Err(source) => {
                if !matches!(source, super::arena::ArenaError::IncompatibleMatrixTypes) {
                    return Err(ProductionAdapterError::Arena(source));
                }
                let actual_inputs = input_ids
                    .iter()
                    .map(|input| self.job.expressions().value_type(*input).cloned())
                    .collect::<Result<Vec<_>, _>>()?
                    .into_boxed_slice();
                return Err(ProductionAdapterError::ArenaContext {
                    wire: wire.clone(),
                    operation: format!("{operator_for_error:?}"),
                    expected_output: expected_output.clone(),
                    actual_inputs,
                    source,
                });
            }
        };
        if check_output && self.job.expressions().value_type(expression)? != expected_output {
            let actual_inputs = input_ids
                .iter()
                .map(|input| self.job.expressions().value_type(*input).cloned())
                .collect::<Result<Vec<_>, _>>()?
                .into_boxed_slice();
            return Err(ProductionAdapterError::ArenaContext {
                wire: wire.clone(),
                operation: format!("{operator_for_error:?}"),
                expected_output: expected_output.clone(),
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
        self.call_family_with_wire_reason(
            family,
            index,
            self.plan.target().residual.clone(),
            BetaReason::ScalarFamilyGet,
        )
    }

    fn direct_canonical_extract_range(&self, expression: ExprId) -> Option<TrustedIndexRange> {
        let node = self.job.expressions().node(expression).ok()?;
        let ValueOperator::ExtractCoefficient {
            canonical_input_exclusive_upper: Some(upper), ..
        } = &node.operator
        else {
            return None;
        };
        let maximum_exclusive = upper.to_u64()?;
        (maximum_exclusive > 0).then_some(TrustedIndexRange { minimum: 0, maximum_exclusive })
    }

    fn project_index_range_through_reducible_calls(
        &self,
        expression: ExprId,
        wire: &PlannedWire,
    ) -> (Option<TrustedIndexRange>, IndexRangeProjectionStats) {
        let mut projector = IndexRangeProjector::new(self);
        let active_ranges = self
            .active_loop_argument_ranges
            .iter()
            .filter(|((occurrence, _), _)| occurrence == &wire.occurrence)
            .map(|((_, argument), range)| (*argument, *range))
            .collect::<Vec<_>>();
        for (argument, range) in active_ranges {
            let environment = projector.push_environment(vec![IndexRangeProjectionBinding {
                expression: argument,
                value_type: ResolvedValueType::Int,
                range: (BigInt::from(range.minimum), BigInt::from(range.maximum_exclusive)),
            }]);
            let Some((minimum, maximum_exclusive)) = projector.evaluate(expression, environment)
            else {
                continue;
            };
            let Some(minimum) = minimum.to_u64() else { continue };
            let Some(maximum_exclusive) = maximum_exclusive.to_u64() else { continue };
            if minimum < maximum_exclusive {
                return (Some(TrustedIndexRange { minimum, maximum_exclusive }), projector.stats);
            }
        }
        (None, projector.stats)
    }

    fn call_family_with_wire(
        &mut self,
        family: FamilyValueId,
        index: ExprId,
        wire: PlannedWire,
    ) -> Result<ExprId, ProductionAdapterError> {
        self.call_family_with_wire_reason(family, index, wire, BetaReason::Other)
    }

    fn call_family_with_wire_reason(
        &mut self,
        family: FamilyValueId,
        index: ExprId,
        wire: PlannedWire,
        reason: BetaReason,
    ) -> Result<ExprId, ProductionAdapterError> {
        if self.job.facts().trusted_index_range(index).is_ok() {
            let range =
                self.job.facts().trusted_index_range(index).map_err(|_| {
                    ProductionAdapterError::MissingSelectorRange { wire: wire.clone() }
                })?;
            return self.call_family_with_resolved_range_reason(family, index, range, reason);
        }
        if let Ok(Some(index_range)) = self.derived_open_index_range(index, &wire) {
            self.selector_range_compact_direct_hits =
                self.selector_range_compact_direct_hits.saturating_add(1);
            return self.call_family_with_resolved_range_reason(family, index, index_range, reason);
        }
        if let Some(index_range) = self.direct_canonical_extract_range(index) {
            self.selector_range_compact_direct_hits =
                self.selector_range_compact_direct_hits.saturating_add(1);
            return self.call_family_with_resolved_range_reason(family, index, index_range, reason);
        }
        let (projected_range, projection_stats) =
            self.project_index_range_through_reducible_calls(index, &wire);
        self.selector_range_projector_nodes =
            self.selector_range_projector_nodes.saturating_add(projection_stats.nodes);
        self.selector_range_projector_program_calls = self
            .selector_range_projector_program_calls
            .saturating_add(projection_stats.program_calls);
        if let Some(index_range) = projected_range {
            if projection_stats.program_calls != 0 {
                self.selector_range_projected_program_call_hits =
                    self.selector_range_projected_program_call_hits.saturating_add(1);
            } else {
                self.selector_range_compact_direct_hits =
                    self.selector_range_compact_direct_hits.saturating_add(1);
            }
            return self.call_family_with_resolved_range_reason(family, index, index_range, reason);
        }
        // A top-level Parallel Zip now remains a compact generated-family call. Range analysis is
        // a semantic exposure boundary: inspect a fully materialized selector view while retaining
        // the original compact expression in the finalized body and family call identity.
        self.selector_range_fallback_materializations =
            self.selector_range_fallback_materializations.saturating_add(1);
        let range_index = self.job.materialize_reducible_generated_calls_with_reason(
            index,
            BetaReason::SelectorRangeExposure,
        )?;
        if let Some(index_range) = self.derived_open_index_range(range_index, &wire)? {
            return self.call_family_with_resolved_range_reason(family, index, index_range, reason);
        }
        let Some(index_range) =
            self.active_loop_argument_ranges.get(&(wire.occurrence.clone(), range_index)).copied()
        else {
            let extracted_range = self.job.expressions().node(range_index).ok().and_then(|node| {
                match &node.operator {
                    ValueOperator::ExtractCoefficient {
                        canonical_input_exclusive_upper: Some(upper),
                        ..
                    } => Some(TrustedIndexRange { minimum: 0, maximum_exclusive: upper.to_u64()? }),
                    _ => None,
                }
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
            return self.call_family_with_resolved_range_reason(family, index, index_range, reason);
        };
        self.call_family_with_resolved_range_reason(family, index, index_range, reason)
    }

    fn call_family_with_resolved_range(
        &mut self,
        family: FamilyValueId,
        index: ExprId,
        range: TrustedIndexRange,
    ) -> Result<ExprId, ProductionAdapterError> {
        self.call_family_with_resolved_range_reason(family, index, range, BetaReason::Other)
    }

    fn call_family_with_resolved_range_reason(
        &mut self,
        family: FamilyValueId,
        index: ExprId,
        range: TrustedIndexRange,
        reason: BetaReason,
    ) -> Result<ExprId, ProductionAdapterError> {
        if matches!(self.job.programs().family_element_type(family)?, ResolvedValueType::Matrix(_))
        {
            Ok(self.job.call_family_in_program_scope_deferred_generated_with_reason(
                family, index, range, reason,
            )?)
        } else {
            Ok(self.job.call_family_in_program_scope_with_reason(family, index, range, reason)?)
        }
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
    fn call_family_in_program_scope_deferred_generated(
        &mut self,
        family: FamilyValueId,
        index: ExprId,
        range: TrustedIndexRange,
    ) -> Result<ExprId, ProductionAdapterError> {
        Ok(self.job.call_family_in_program_scope_deferred_generated(family, index, range)?)
    }
    fn generated_family(
        &mut self,
        domain: FamilyDomain,
        body: ExprId,
    ) -> Result<FamilyValueId, ProductionAdapterError> {
        let family = self.job.with_arena_stores(|expressions, programs, _| {
            programs.generated_family_from_body(expressions, domain, body)
        })?;
        self.generated_family_calls = self.generated_family_calls.saturating_add(1);
        Ok(family)
    }
    fn opaque_generated_family(
        &mut self,
        domain: FamilyDomain,
        body: ExprId,
    ) -> Result<FamilyValueId, ProductionAdapterError> {
        let family = self.job.with_arena_stores(|expressions, programs, _| {
            programs.opaque_generated_family_from_body(expressions, domain, body)
        })?;
        self.generated_family_calls = self.generated_family_calls.saturating_add(1);
        Ok(family)
    }
    fn explicit_family(
        &mut self,
        domain: FamilyDomain,
        values: Box<[ExprId]>,
    ) -> Result<FamilyValueId, ProductionAdapterError> {
        Ok(self.job.with_arena_stores(|expressions, programs, facts| {
            programs.explicit_family(expressions, facts, domain, values)
        })?)
    }
    fn authoritative_matrix_observation_view(
        &mut self,
        expression: ExprId,
    ) -> Result<ExprId, ProductionAdapterError> {
        if matches!(self.job.facts().facts(expression), Ok(super::facts::ValueFacts::Matrix(_))) {
            Ok(expression)
        } else {
            Ok(self.job.materialize_reducible_generated_calls_with_reason(
                expression,
                BetaReason::MatrixFactObservation,
            )?)
        }
    }

    fn project_matrix_fact_owner(
        &mut self,
        root: ExprId,
    ) -> Result<MatrixFactProjection, ProductionAdapterError> {
        self.project_matrix_fact_owner_in(root, &[], None, &mut BTreeSet::new())
    }

    fn project_matrix_fact_owner_in(
        &mut self,
        root: ExprId,
        bindings: &[ExprId],
        signature: Option<&[ProgramInput]>,
        visiting: &mut BTreeSet<ExprId>,
    ) -> Result<MatrixFactProjection, ProductionAdapterError> {
        if !visiting.insert(root) {
            return Ok(MatrixFactProjection::Unknown);
        }
        let result = self.project_matrix_fact_owner_inner(root, bindings, signature, visiting);
        visiting.remove(&root);
        result
    }

    fn project_matrix_fact_owner_inner(
        &mut self,
        root: ExprId,
        bindings: &[ExprId],
        signature: Option<&[ProgramInput]>,
        visiting: &mut BTreeSet<ExprId>,
    ) -> Result<MatrixFactProjection, ProductionAdapterError> {
        if matches!(self.job.facts().facts(root), Ok(super::facts::ValueFacts::Matrix(_))) {
            self.matrix_fact_projection_direct_hits =
                self.matrix_fact_projection_direct_hits.saturating_add(1);
            return Ok(MatrixFactProjection::Found(root));
        }
        let node = self.job.expressions().node(root)?.clone();
        match node.operator {
            ValueOperator::Argument { position, value_type } => {
                let Some(expected) = signature.and_then(|inputs| inputs.get(position as usize))
                else {
                    return Ok(MatrixFactProjection::Unknown);
                };
                if value_type != expected.value_type {
                    return Ok(MatrixFactProjection::Unknown);
                }
                let Some(&binding) = bindings.get(position as usize) else {
                    return Ok(MatrixFactProjection::Unknown);
                };
                if self.job.expressions().value_type(binding)? != &expected.value_type {
                    return Ok(MatrixFactProjection::Unknown);
                }
                self.matrix_fact_projection_argument_hits =
                    self.matrix_fact_projection_argument_hits.saturating_add(1);
                self.project_matrix_fact_owner_in(binding, &[], None, visiting)
            }
            ValueOperator::ProgramCall { program } => {
                if program.arena != self.job.programs().token() {
                    return Ok(MatrixFactProjection::Unknown);
                }
                if let Some(reduced) = self.job.expressions().program_call_reduction(root)? {
                    self.matrix_fact_projection_sidecar_hits =
                        self.matrix_fact_projection_sidecar_hits.saturating_add(1);
                    return self.project_matrix_fact_owner_in(reduced, &[], None, visiting);
                }
                let Some(family) = self.job.programs().family_for_program(program) else {
                    return Ok(MatrixFactProjection::Unknown);
                };
                if !self.job.programs().family_is_reducible(family)? {
                    return Ok(MatrixFactProjection::Unknown);
                }
                let callee_signature = self.job.programs().program_signature(program)?.clone();
                if callee_signature.inputs.len() != node.inputs.len() {
                    return Ok(MatrixFactProjection::Unknown);
                }
                let mut resolved_inputs = Vec::with_capacity(node.inputs.len());
                for (position, input) in node.inputs.iter().copied().enumerate() {
                    let Some(resolved) =
                        self.resolve_projection_binding(input, bindings, signature)?
                    else {
                        return Ok(MatrixFactProjection::Unknown);
                    };
                    if self.job.expressions().value_type(resolved)? !=
                        &callee_signature.inputs[position].value_type
                    {
                        return Ok(MatrixFactProjection::Unknown);
                    }
                    resolved_inputs.push(resolved);
                }
                let body = self.job.programs().family_body(family)?;
                if self.job.expressions().is_closed(body)? {
                    self.matrix_fact_projection_closed_root_hits =
                        self.matrix_fact_projection_closed_root_hits.saturating_add(1);
                    self.project_matrix_fact_owner_in(body, &[], None, visiting)
                } else {
                    self.project_matrix_fact_owner_in(
                        body,
                        &resolved_inputs,
                        Some(&callee_signature.inputs),
                        visiting,
                    )
                }
            }
            ValueOperator::OpaqueFamilyElement { .. } => Ok(MatrixFactProjection::Unknown),
            _ if node.inputs.is_empty() && self.job.expressions().is_closed(root)? => {
                self.matrix_fact_projection_proven_absent =
                    self.matrix_fact_projection_proven_absent.saturating_add(1);
                Ok(MatrixFactProjection::ProvenAbsent)
            }
            _ => Ok(MatrixFactProjection::Unknown),
        }
    }

    fn resolve_projection_binding(
        &self,
        root: ExprId,
        bindings: &[ExprId],
        signature: Option<&[ProgramInput]>,
    ) -> Result<Option<ExprId>, ProductionAdapterError> {
        let mut current = root;
        let mut seen = BTreeSet::new();
        loop {
            if !seen.insert(current) {
                return Ok(Some(current));
            }
            let node = self.job.expressions().node(current)?;
            let ValueOperator::Argument { position, ref value_type } = node.operator else {
                return Ok(Some(current));
            };
            let Some(expected) = signature.and_then(|inputs| inputs.get(position as usize)) else {
                return Ok(None);
            };
            if value_type != &expected.value_type {
                return Ok(None);
            }
            let Some(&binding) = bindings.get(position as usize) else {
                return Ok(None);
            };
            if self.job.expressions().value_type(binding)? != &expected.value_type {
                return Ok(None);
            }
            current = binding;
        }
    }

    fn gadget_fact_owner(
        &mut self,
        input: ExprId,
    ) -> Result<Option<ExprId>, ProductionAdapterError> {
        match self.project_matrix_fact_owner(input)? {
            MatrixFactProjection::Found(owner) => Ok(Some(owner)),
            MatrixFactProjection::ProvenAbsent => Ok(None),
            MatrixFactProjection::Unknown => {
                self.matrix_fact_projection_fallbacks =
                    self.matrix_fact_projection_fallbacks.saturating_add(1);
                let before = self.job.programs().diagnostic_counters().beta_nodes_visited;
                let owner = self.job.materialize_reducible_generated_calls_with_reason(
                    input,
                    BetaReason::GadgetFactFallback,
                )?;
                let after = self.job.programs().diagnostic_counters().beta_nodes_visited;
                self.matrix_fact_projection_fallback_beta_nodes = self
                    .matrix_fact_projection_fallback_beta_nodes
                    .saturating_add(after.saturating_sub(before));
                Ok(Some(owner))
            }
        }
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
    pub(crate) fn new(
        protocol: &'a ProtocolDecl,
        plan: &'a ProtocolPlan,
        parameters: BTreeMap<String, BigInt>,
    ) -> Result<Self, ProductionAdapterError> {
        let wire_table = PlanWireTable::build(plan)?;
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
        let aliases = plan
            .aliases()
            .iter()
            .map(|alias| {
                Ok((
                    wire_table.id(&alias.child).ok_or_else(|| {
                        ProductionAdapterError::MissingWire { wire: alias.child.clone() }
                    })?,
                    wire_table.id(&alias.parent).ok_or_else(|| {
                        ProductionAdapterError::MissingWire { wire: alias.parent.clone() }
                    })?,
                ))
            })
            .collect::<Result<HashMap<_, _>, ProductionAdapterError>>()?;
        let outputs = plan
            .output_mappings()
            .iter()
            .map(|mapping| {
                Ok((
                    wire_table.id(&mapping.parent).ok_or_else(|| {
                        ProductionAdapterError::MissingWire { wire: mapping.parent.clone() }
                    })?,
                    wire_table.id(&mapping.child).ok_or_else(|| {
                        ProductionAdapterError::MissingWire { wire: mapping.child.clone() }
                    })?,
                ))
            })
            .collect::<Result<HashMap<_, _>, ProductionAdapterError>>()?;
        let artifacts = plan
            .artifact_producers()
            .iter()
            .map(|producer| {
                Ok((
                    wire_table.id(&producer.consumer).ok_or_else(|| {
                        ProductionAdapterError::MissingWire { wire: producer.consumer.clone() }
                    })?,
                    wire_table.id(&producer.producer).ok_or_else(|| {
                        ProductionAdapterError::MissingWire { wire: producer.producer.clone() }
                    })?,
                ))
            })
            .collect::<Result<HashMap<_, _>, ProductionAdapterError>>()?;
        let adapter = Self {
            plan,
            values: vec![None; wire_table.wires.len()],
            value_count: 0,
            aliases,
            outputs,
            artifacts,
            wire_table,
            graphs,
            params: ParamEnv { integers: parameters, ..ParamEnv::default() },
            job,
            token,
            protocol_inputs,
            input_contracts,
            sample_events: BTreeMap::new(),
            static_indices: HashMap::new(),
            active_loop_indices: BTreeMap::new(),
            active_loop_arguments: BTreeMap::new(),
            active_loop_argument_ranges: BTreeMap::new(),
            active_parallel_depth: 0,
            generated_family_calls: 0,
            matrix_fact_projection_direct_hits: 0,
            matrix_fact_projection_closed_root_hits: 0,
            matrix_fact_projection_argument_hits: 0,
            matrix_fact_projection_sidecar_hits: 0,
            matrix_fact_projection_proven_absent: 0,
            matrix_fact_projection_fallbacks: 0,
            matrix_fact_projection_fallback_beta_nodes: 0,
            matrix_select_open_observation_skips: 0,
            matrix_select_open_observation_skipped_branches: 0,
            selector_range_compact_direct_hits: 0,
            selector_range_projected_program_call_hits: 0,
            selector_range_fallback_materializations: 0,
            selector_range_projector_nodes: 0,
            selector_range_projector_program_calls: 0,
            relation_candidates: Vec::new(),
            relation_candidate_indices: BTreeMap::new(),
            gadget_decompositions: BTreeMap::new(),
            trapdoor_values: BTreeMap::new(),
            occurrence_descendants,
            diagnostic_budget: 128,
            resolver_progress: ResolverProgress::new(),
            #[cfg(test)]
            test_sampler_fact_bound: None,
        };
        let mut adapter = adapter;
        adapter.assign_sample_events()?;
        adapter.predeclare_trapdoors()?;
        adapter.selector_prepass()?;
        adapter.constant_matrix_prepass()?;
        adapter.job.finalize_facts(adapter.token)?;
        Ok(adapter)
    }

    pub(crate) fn lower(self) -> Result<(CheckerJob, ProductionRoots), ProductionAdapterError> {
        self.lower_internal(false)
    }

    #[cfg(test)]
    fn lower_force_eager(self) -> Result<(CheckerJob, ProductionRoots), ProductionAdapterError> {
        self.lower_internal(true)
    }

    #[cfg(test)]
    fn with_test_sampler_fact_bound(mut self, bound: u64) -> Self {
        self.test_sampler_fact_bound = Some(bound);
        self
    }

    fn lower_internal(
        mut self,
        force_eager: bool,
    ) -> Result<(CheckerJob, ProductionRoots), ProductionAdapterError> {
        let lowering_started = Instant::now();
        info!(
            target: "mxx_correctness::operational_noise",
            stage = "resolve_residual",
            event = "start",
            plan_nodes = self.plan.nodes().len(),
            relation_candidates = self.relation_candidates.len(),
            "operational noise lowering stage"
        );
        let stage_started = Instant::now();
        self.resolver_progress.start();
        let residual =
            self.resolve(self.plan.target().residual.clone(), Rc::new(BTreeMap::new()))?;
        info!(
            target: "mxx_correctness::operational_noise",
            stage = "resolve_residual",
            event = "complete",
            elapsed_ms = stage_started.elapsed().as_millis() as u64,
            resolved_values = self.value_count,
            generated_family_calls = self.generated_family_calls,
            relation_candidates = self.relation_candidates.len(),
            lifted_relation_candidates = self
                .relation_candidates
                .iter()
                .filter(|candidate| candidate.family_operands.is_some())
                .count(),
            matrix_fact_projection_direct_hits = self.matrix_fact_projection_direct_hits,
            matrix_fact_projection_closed_root_hits =
                self.matrix_fact_projection_closed_root_hits,
            matrix_fact_projection_argument_hits = self.matrix_fact_projection_argument_hits,
            matrix_fact_projection_sidecar_hits = self.matrix_fact_projection_sidecar_hits,
            matrix_fact_projection_proven_absent = self.matrix_fact_projection_proven_absent,
            matrix_fact_projection_fallbacks = self.matrix_fact_projection_fallbacks,
            matrix_fact_projection_fallback_beta_nodes =
                self.matrix_fact_projection_fallback_beta_nodes,
            matrix_select_open_observation_skips = self.matrix_select_open_observation_skips,
            matrix_select_open_observation_skipped_branches =
                self.matrix_select_open_observation_skipped_branches,
            selector_range_compact_direct_hits = self.selector_range_compact_direct_hits,
            selector_range_projected_program_call_hits =
                self.selector_range_projected_program_call_hits,
            selector_range_fallback_materializations = self.selector_range_fallback_materializations,
            selector_range_projector_nodes = self.selector_range_projector_nodes,
            selector_range_projector_program_calls = self.selector_range_projector_program_calls,
            "operational noise lowering stage"
        );

        let stage_started = Instant::now();
        info!(
            target: "mxx_correctness::operational_noise",
            stage = "resolve_decoder",
            event = "start",
            "operational noise lowering stage"
        );
        let decoder = self.resolve(self.plan.target().decoder.clone(), Rc::new(BTreeMap::new()))?;
        info!(
            target: "mxx_correctness::operational_noise",
            stage = "resolve_decoder",
            event = "complete",
            elapsed_ms = stage_started.elapsed().as_millis() as u64,
            resolved_values = self.value_count,
            generated_family_calls = self.generated_family_calls,
            matrix_fact_projection_direct_hits = self.matrix_fact_projection_direct_hits,
            matrix_fact_projection_closed_root_hits =
                self.matrix_fact_projection_closed_root_hits,
            matrix_fact_projection_argument_hits = self.matrix_fact_projection_argument_hits,
            matrix_fact_projection_sidecar_hits = self.matrix_fact_projection_sidecar_hits,
            matrix_fact_projection_proven_absent = self.matrix_fact_projection_proven_absent,
            matrix_fact_projection_fallbacks = self.matrix_fact_projection_fallbacks,
            matrix_fact_projection_fallback_beta_nodes =
                self.matrix_fact_projection_fallback_beta_nodes,
            matrix_select_open_observation_skips = self.matrix_select_open_observation_skips,
            matrix_select_open_observation_skipped_branches =
                self.matrix_select_open_observation_skipped_branches,
            selector_range_compact_direct_hits = self.selector_range_compact_direct_hits,
            selector_range_projected_program_call_hits =
                self.selector_range_projected_program_call_hits,
            selector_range_fallback_materializations = self.selector_range_fallback_materializations,
            selector_range_projector_nodes = self.selector_range_projector_nodes,
            selector_range_projector_program_calls = self.selector_range_projector_program_calls,
            "operational noise lowering stage"
        );

        #[cfg(test)]
        if let Some(bound) = self.test_sampler_fact_bound {
            let token = self.job.active_candidate_token();
            let expression_token = self.job.expressions().token();
            for slot in 0..self.job.expressions().node_count() {
                let expression = ExprId::new(expression_token, slot as u32);
                let Ok(node) = self.job.expressions().node(expression) else { continue };
                if !matches!(node.operator, ValueOperator::Sampler { .. }) {
                    continue;
                }
                let ResolvedValueType::Matrix(matrix) =
                    self.job.expressions().value_type(expression)?.clone()
                else {
                    continue;
                };
                let mut facts = MatrixFacts::new(
                    matrix.clone(),
                    MatrixMetadata::new(MatrixLayout::row_major(matrix.rows, matrix.columns)),
                );
                facts.coefficient_bound = NumericContract::Known(CoefficientBound::finite(bound));
                self.job.insert_matrix_facts(token, expression, facts)?;
            }
        }

        let stage_started = Instant::now();
        info!(
            target: "mxx_correctness::operational_noise",
            stage = "register_relations",
            event = "start",
            relation_candidates = self.relation_candidates.len(),
            "operational noise lowering stage"
        );
        let mut compact_residual =
            if force_eager { Some("test-forced eager residual path".to_owned()) } else { None };
        let mut compact_shell_plan = if compact_residual.is_none() {
            match self.compile_compact_root(&residual) {
                Ok(plan) => Some(plan),
                Err(reason) => {
                    compact_residual = Some(reason);
                    None
                }
            }
        } else {
            None
        };
        let compact_plan_diagnostics = compact_shell_plan.as_ref().map(|plan| {
            (
                plan.preflight_node_occurrences,
                plan.gadget_shells.len() as u64,
                plan.gadget_occurrences(),
                plan.scalar_occurrences(),
                plan.shell_allocated,
                plan.shell_new,
                plan.shell_hits,
            )
        });
        let root_kind = match residual {
            Value::Expr(_) => "closed",
            Value::Family(_) => "family",
        };
        info!(
            target: "mxx_correctness::operational_noise",
            stage = "compile_compact_root",
            root_kind,
            eligible = compact_residual.is_none(),
            reason = compact_residual.as_deref().unwrap_or("eligible"),
            preflight_node_occurrences = compact_plan_diagnostics.map_or(0, |value| value.0),
            planned_unique_gadget_shells = compact_plan_diagnostics.map_or(0, |value| value.1),
            planned_gadget_shell_occurrences = compact_plan_diagnostics.map_or(0, |value| value.2),
            planned_scalar_occurrences = compact_plan_diagnostics.map_or(0, |value| value.3),
            shell_allocated = compact_plan_diagnostics.map_or(0, |value| value.4),
            shell_new = compact_plan_diagnostics.map_or(0, |value| value.5),
            shell_hits = compact_plan_diagnostics.map_or(0, |value| value.6),
            "compact residual root selection"
        );
        self.register_reached_relations()?;
        info!(
            target: "mxx_correctness::operational_noise",
            stage = "register_relations",
            event = "complete",
            elapsed_ms = stage_started.elapsed().as_millis() as u64,
            "operational noise lowering stage"
        );

        // Resolver construction may retain validated generated-family calls for matrix-valued
        // dynamic accesses. Materialize each semantic root exactly once, in deterministic root
        // order, after relation registration has inspected its authoritative family views and
        // before resource counters are frozen.
        let residual = if compact_residual.is_none() {
            residual
        } else {
            self.materialize_root_value_with_reason(residual, BetaReason::ResidualRoot)?
        };
        let decoder = self.materialize_root_value_with_reason(decoder, BetaReason::DecoderRoot)?;
        self.emit_beta_reason_diagnostics("final");

        let stage_started = Instant::now();
        info!(
            target: "mxx_correctness::operational_noise",
            stage = "freeze_relations",
            event = "start",
            "operational noise lowering stage"
        );
        if let Some(plan) = compact_shell_plan.as_mut() {
            self.materialize_compact_shell_plan(plan)?;
            self.job.set_compact_shell_plan(plan.clone())?;
        }
        self.job.freeze_relations(self.token)?;
        info!(
            target: "mxx_correctness::operational_noise",
            stage = "freeze_relations",
            event = "complete",
            elapsed_ms = stage_started.elapsed().as_millis() as u64,
            "operational noise lowering stage"
        );

        let stage_started = Instant::now();
        info!(
            target: "mxx_correctness::operational_noise",
            stage = "close_roots",
            event = "start",
            "operational noise lowering stage"
        );
        let roots = ProductionRoots {
            residual: self.close_root(
                residual,
                &self.plan.target().residual,
                "close residual root",
                compact_residual.is_none(),
            )?,
            decoder: self.close_root(
                decoder,
                &self.plan.target().decoder,
                "close decoder root",
                false,
            )?,
            occurrences: self.plan.counters().occurrences,
            samples: self.sample_events.len() as u64,
        };
        info!(
            target: "mxx_correctness::operational_noise",
            stage = "close_roots",
            event = "complete",
            elapsed_ms = stage_started.elapsed().as_millis() as u64,
            total_elapsed_ms = lowering_started.elapsed().as_millis() as u64,
            "operational noise lowering stage"
        );
        Ok((self.job, roots))
    }

    #[cfg(test)]
    fn materialize_root_value(&mut self, value: Value) -> Result<Value, ProductionAdapterError> {
        self.materialize_root_value_with_reason(value, BetaReason::Other)
    }

    fn compact_parent_capability(
        &self,
        parent: ExprId,
        index: usize,
        child: ExprId,
        valid_fixed_slices: &BTreeSet<ExprId>,
    ) -> CompactCompilerParent {
        if index == 0 &&
            valid_fixed_slices.contains(&parent) &&
            self.job
                .expressions()
                .node(child)
                .ok()
                .is_some_and(|node| matches!(node.operator, ValueOperator::DeterministicHash(_)))
        {
            return CompactCompilerParent::FixedSliceChild { slice: parent };
        }
        let Some(parent_node) = self.job.expressions().node(parent).ok() else {
            return CompactCompilerParent::Ordinary;
        };
        if !matches!(
            parent_node.operator,
            ValueOperator::Matrix(MatrixOperation::Multiply) |
                ValueOperator::Matrix(MatrixOperation::Tensor { .. })
        ) || parent_node.inputs.len() != 2
        {
            return CompactCompilerParent::Ordinary;
        }
        let child_is_gadget = self.job.expressions().node(child).ok().is_some_and(|node| {
            matches!(
                node.operator,
                ValueOperator::Transform(ValueTransformOperation::GadgetDecompose { .. })
            )
        });
        let child_is_scalar =
            self.job.expressions().value_type(child).ok().is_some_and(|value_type| {
                matches!(
                    value_type,
                    ResolvedValueType::Matrix(matrix) if matrix.rows == 1 && matrix.columns == 1
                )
            });
        let sibling_is_nonscalar = self
            .job
            .expressions()
            .value_type(parent_node.inputs[usize::from(index == 0)])
            .ok()
            .is_some_and(|value_type| {
                matches!(
                    value_type,
                    ResolvedValueType::Matrix(matrix) if matrix.rows != 1 || matrix.columns != 1
                )
            });
        if index == 1 && (child_is_gadget || (child_is_scalar && sibling_is_nonscalar)) {
            CompactCompilerParent::Product { consumer: parent, is_right: true }
        } else if index == 0 && child_is_scalar && sibling_is_nonscalar {
            CompactCompilerParent::Product { consumer: parent, is_right: false }
        } else {
            CompactCompilerParent::Ordinary
        }
    }

    /// Compile the private compact authorization before relation registration. This is the sole
    /// occurrence-sensitive traversal: structural validation, exact gadget/scalar consumer
    /// planning, and preflight occurrence accounting happen in the same enter/exit walk. Only
    /// the final gadget-rule map lookup runs after the walk.
    fn compile_compact_root(&self, value: &Value) -> Result<CompactShellPlan, String> {
        let root_kind = match value {
            Value::Expr(_) => "closed",
            Value::Family(_) => "family",
        };
        info!(
            target: "mxx_correctness::operational_noise",
            stage = "compile_compact_root",
            event = "start",
            root_kind,
            plan_nodes = self.plan.nodes().len(),
            "compact compiler start"
        );
        if self.plan.nodes().values().any(|node| matches!(node.kind, NodeKind::SequentialLoop(_))) {
            return Err("residual contains sequential structure".to_owned());
        }
        let (root, initial_owner) = match *value {
            Value::Expr(root) => {
                let root_node = match self.job.expressions().node(root) {
                    Ok(node) => node,
                    Err(_) => return Err("invalid expression authority".to_owned()),
                };
                if self.job.expressions().value_type(root).is_err() {
                    return Err("invalid expression type".to_owned());
                }
                if !self.job.expressions().is_closed(root).unwrap_or(false) {
                    return Err("residual is open".to_owned());
                }
                let ValueOperator::ProgramCall { program: root_program } = root_node.operator
                else {
                    return Err("residual root is not a generated call".to_owned());
                };
                let Some(root_family) = self.job.programs().family_for_program(root_program) else {
                    return Err("residual root has no family authority".to_owned());
                };
                if !self.job.programs().family_is_reducible(root_family).unwrap_or(false) {
                    return Err("residual root is opaque".to_owned());
                }
                (root, None)
            }
            Value::Family(family) => {
                let domain = match self.job.programs().family_domain(family) {
                    Ok(domain) => domain,
                    Err(_) => return Err("compact family authority is missing".to_owned()),
                };
                if !self.job.programs().family_is_reducible(family).unwrap_or(false) {
                    return Err("residual family is opaque".to_owned());
                }
                let signature = match self.job.programs().program_signature(family.program()) {
                    Ok(signature) => signature,
                    Err(_) => return Err("compact family signature is missing".to_owned()),
                };
                let element_type = match self.job.programs().family_element_type(family) {
                    Ok(element_type) => element_type,
                    Err(_) => return Err("compact family element authority is missing".to_owned()),
                };
                let expected_range = TrustedIndexRange {
                    minimum: domain.minimum,
                    maximum_exclusive: domain.maximum_exclusive,
                };
                if signature.inputs.len() != 1 ||
                    signature.inputs[0].value_type != ResolvedValueType::Int ||
                    signature.inputs[0].trusted_index_range != Some(expected_range) ||
                    signature.output != element_type
                {
                    return Err("compact family signature or range mismatch".to_owned());
                }
                let body = match self.job.programs().family_body(family) {
                    Ok(body) => body,
                    Err(_) => return Err("compact family body is missing".to_owned()),
                };
                if self.job.expressions().value_type(body).ok() != Some(&signature.output) {
                    return Err("compact family output mismatch".to_owned());
                }
                (body, Some(family.program()))
            }
        };
        let mut plan = CompactShellPlan::default();
        let mut authorized_scalars = BTreeSet::<ExprId>::new();
        let mut other_scalar_parents = BTreeSet::<ExprId>::new();
        let mut index_projector = IndexRangeProjector::new(self);
        let root_key = CompactCompilerStateKey {
            expression: root,
            owner: initial_owner,
            binding_environment: None,
            parent: CompactCompilerParent::Ordinary,
            under_planned_shell: false,
            scalar_call_context: None,
            binding_context: None,
            binding_subtree: false,
        };
        let mut states = Vec::<CompactCompilerState>::new();
        let mut state_ids = BTreeMap::<CompactCompilerStateKey, usize>::new();
        let mut active = BTreeSet::<usize>::new();
        let mut work = Vec::<(usize, bool)>::new();
        enqueue_compact_compiler_state(
            root_key,
            None,
            &mut states,
            &mut state_ids,
            &active,
            &mut work,
        )?;
        let mut order = Vec::<usize>::new();
        let mut valid_fixed_slices = BTreeSet::<ExprId>::new();
        let mut virtual_nodes = 0_u64;
        let mut reducible_expansions = 0_u64;
        let mut work_peak = work.len();
        let relation_endpoints = self
            .relation_candidates
            .iter()
            .flat_map(|candidate| {
                [candidate.preimage, candidate.public, candidate.trapdoor, candidate.target]
            })
            .collect::<BTreeSet<_>>();
        while let Some((state_index, exit)) = work.pop() {
            let state_key = states[state_index].key.clone();
            if exit {
                active.remove(&state_index);
                order.push(state_index);
                continue;
            }
            if !active.insert(state_index) {
                return Err("cycle or recursive generated body".to_owned());
            }
            let expression = state_key.expression;
            let owner = state_key.owner;
            let parent = match state_key.parent {
                CompactCompilerParent::Product { consumer, is_right } => Some((consumer, is_right)),
                CompactCompilerParent::FixedSliceChild { slice } => Some((slice, false)),
                CompactCompilerParent::Ordinary => None,
            };
            let under_planned_shell = state_key.under_planned_shell;
            let scalar_call_context = state_key.scalar_call_context;
            let binding_context = state_key.binding_context.clone();
            let under_binding_subtree = state_key.binding_subtree;
            work.push((state_index, true));
            let node = match self.job.expressions().node(expression) {
                Ok(node) => node,
                Err(_) => return Err("invalid expression authority".to_owned()),
            };
            let output = match self.job.expressions().value_type(expression) {
                Ok(output) => output,
                Err(_) => return Err("invalid expression type".to_owned()),
            };
            // A binding context applies to the complete expression occurrence, including its
            // descendants. Keep this derived bit local to the current work item so an ordinary
            // sibling cannot accidentally inherit a binding marker.
            let binding_subtree = under_binding_subtree || binding_context.is_some();
            let mut state_virtual_node = false;
            let mut state_plan_delta = None;
            let mut scalar_program_call = false;
            if let Some((expected_type, expected_range)) = binding_context.as_ref() {
                if output != expected_type {
                    self.log_compact_open_binding_rejection(
                        expression,
                        &node,
                        expected_type,
                        *expected_range,
                        owner,
                    );
                    return Err("compact binding expression is open".to_owned());
                }
                if !self.job.expressions().is_closed(expression).unwrap_or(false) {
                    let Some(expected_range) = *expected_range else {
                        self.log_compact_open_binding_rejection(
                            expression,
                            &node,
                            expected_type,
                            *expected_range,
                            owner,
                        );
                        return Err("compact binding expression is open".to_owned());
                    };
                    let Some(owner) = owner else {
                        self.log_compact_open_binding_rejection(
                            expression,
                            &node,
                            expected_type,
                            Some(expected_range),
                            owner,
                        );
                        return Err("compact binding expression is open".to_owned());
                    };
                    let owner_signature = match self.job.programs().program_signature(owner) {
                        Ok(signature) => signature,
                        Err(_) => {
                            self.log_compact_open_binding_rejection(
                                expression,
                                &node,
                                expected_type,
                                Some(expected_range),
                                Some(owner),
                            );
                            return Err("compact binding expression is open".to_owned());
                        }
                    };
                    let Some(owner_input) = owner_signature.inputs.first() else {
                        self.log_compact_open_binding_rejection(
                            expression,
                            &node,
                            expected_type,
                            Some(expected_range),
                            Some(owner),
                        );
                        return Err("compact binding expression is open".to_owned());
                    };
                    let Some(owner_range) = owner_input.trusted_index_range else {
                        self.log_compact_open_binding_rejection(
                            expression,
                            &node,
                            expected_type,
                            Some(expected_range),
                            Some(owner),
                        );
                        return Err("compact binding expression is open".to_owned());
                    };
                    if owner_signature.inputs.len() != 1 ||
                        owner_input.value_type != ResolvedValueType::Int
                    {
                        self.log_compact_open_binding_rejection(
                            expression,
                            &node,
                            expected_type,
                            Some(expected_range),
                            Some(owner),
                        );
                        return Err("compact binding expression is open".to_owned());
                    }
                    let environment =
                        index_projector.push_environment(vec![IndexRangeProjectionBinding {
                            expression,
                            value_type: ResolvedValueType::Int,
                            range: (
                                BigInt::from(owner_range.minimum),
                                BigInt::from(owner_range.maximum_exclusive),
                            ),
                        }]);
                    let Some((minimum, maximum_exclusive)) =
                        index_projector.evaluate(expression, environment)
                    else {
                        self.log_compact_open_binding_rejection(
                            expression,
                            &node,
                            expected_type,
                            Some(expected_range),
                            Some(owner),
                        );
                        return Err("compact binding expression is open".to_owned());
                    };
                    if minimum < BigInt::from(expected_range.minimum) ||
                        maximum_exclusive > BigInt::from(expected_range.maximum_exclusive) ||
                        minimum >= maximum_exclusive
                    {
                        self.log_compact_open_binding_rejection(
                            expression,
                            &node,
                            expected_type,
                            Some(expected_range),
                            Some(owner),
                        );
                        return Err("compact binding expression is open".to_owned());
                    }
                }
            }
            match &node.operator {
                ValueOperator::ProgramCall { program } => {
                    let family = match self.job.programs().family_for_program(*program) {
                        Some(family) => family,
                        None => return Err("generated call has no family authority".to_owned()),
                    };
                    if under_planned_shell &&
                        self.job.programs().family_is_reducible(family).unwrap_or(false)
                    {
                        return Err(
                            "gadget decomposition input contains a reducible generated call"
                                .to_owned(),
                        );
                    }
                    if scalar_call_context.is_some() {
                        return Err(
                            "indexed scalar slice contains a nested generated call".to_owned()
                        );
                    }
                    if !self.job.programs().family_is_reducible(family).unwrap_or(false) {
                        let signature = match self.job.programs().program_signature(*program) {
                            Ok(signature) => signature,
                            Err(_) => return Err("invalid opaque program authority".to_owned()),
                        };
                        if node.inputs.len() != signature.inputs.len() ||
                            node.inputs.iter().zip(signature.inputs.iter()).any(
                                |(input, expected)| {
                                    self.job.expressions().value_type(*input).ok() !=
                                        Some(&expected.value_type)
                                },
                            )
                        {
                            return Err("opaque call signature or arity mismatch".to_owned());
                        }
                        for (index, input) in node.inputs.iter().copied().enumerate().rev() {
                            let child_key = CompactCompilerStateKey {
                                expression: input,
                                owner,
                                binding_environment: state_key.binding_environment,
                                parent: self.compact_parent_capability(
                                    expression,
                                    index,
                                    input,
                                    &valid_fixed_slices,
                                ),
                                under_planned_shell,
                                scalar_call_context,
                                binding_context: None,
                                binding_subtree,
                            };
                            enqueue_compact_compiler_state(
                                child_key,
                                Some(state_index),
                                &mut states,
                                &mut state_ids,
                                &active,
                                &mut work,
                            )?;
                        }
                        continue;
                    }
                    let signature = match self.job.programs().program_signature(*program) {
                        Ok(signature) => signature,
                        Err(_) => return Err("invalid generated program authority".to_owned()),
                    };
                    if node.inputs.len() != signature.inputs.len() ||
                        node.inputs.iter().zip(signature.inputs.iter()).any(
                            |(input, expected)| {
                                self.job.expressions().value_type(*input).ok() !=
                                    Some(&expected.value_type)
                            },
                        )
                    {
                        return Err("generated call signature or arity mismatch".to_owned());
                    }
                    // Binding expressions are part of this same occurrence-sensitive walk.  The
                    // projector-backed context below validates each input before any relation
                    // registration or compact materialization can occur.
                    for (input, expected) in
                        node.inputs.iter().copied().zip(signature.inputs.iter()).rev()
                    {
                        let child_key = CompactCompilerStateKey {
                            expression: input,
                            owner,
                            binding_environment: Some(expression),
                            parent: CompactCompilerParent::Ordinary,
                            under_planned_shell,
                            scalar_call_context: None,
                            binding_context: Some((
                                expected.value_type.clone(),
                                expected.trusted_index_range,
                            )),
                            binding_subtree: true,
                        };
                        enqueue_compact_compiler_state(
                            child_key,
                            Some(state_index),
                            &mut states,
                            &mut state_ids,
                            &active,
                            &mut work,
                        )?;
                    }
                    let body = match self.job.programs().family_body(family) {
                        Ok(body) => body,
                        Err(_) => return Err("invalid generated family body".to_owned()),
                    };
                    if self.job.expressions().value_type(body).ok() != Some(&signature.output) {
                        return Err("generated family output mismatch".to_owned());
                    }
                    let body_scalar_context = if let Some((parent_expression, is_right_child)) =
                        parent
                    {
                        let is_scalar = matches!(output, ResolvedValueType::Matrix(matrix) if matrix.rows == 1 && matrix.columns == 1);
                        let consumer = self.job.expressions().node(parent_expression).ok();
                        let parent_is_scalar_consumer = consumer.is_some_and(|parent_node| {
                            let is_product = matches!(
                                parent_node.operator,
                                ValueOperator::Matrix(MatrixOperation::Multiply) |
                                    ValueOperator::Matrix(MatrixOperation::Tensor { .. })
                            ) && parent_node.inputs.len() == 2 &&
                                parent_node.inputs[usize::from(is_right_child)] == expression
                                && self
                                    .job
                                    .expressions()
                                    .value_type(parent_node.inputs[usize::from(!is_right_child)])
                                    .ok()
                                    .is_some_and(|value_type| {
                                        matches!(value_type, ResolvedValueType::Matrix(matrix) if matrix.rows != 1 || matrix.columns != 1)
                                    });
                            is_product
                        });
                        if is_scalar && parent_is_scalar_consumer {
                            if binding_subtree {
                                return Err(
                                    "compact binding subtree contains a scalar plan occurrence"
                                        .to_owned(),
                                );
                            }
                            let parent_node = self
                                .job
                                .expressions()
                                .node(parent_expression)
                                .map_err(|_| "invalid scalar consumer parent".to_owned())?;
                            let ResolvedValueType::Matrix(other_type) = self
                                .job
                                .expressions()
                                .value_type(parent_node.inputs[usize::from(!is_right_child)])
                                .map_err(|_| {
                                    "scalar compact factor sibling type is missing".to_owned()
                                })?
                            else {
                                return Err(
                                    "scalar compact factor sibling is not a matrix".to_owned()
                                );
                            };
                            let scalar_type = match output {
                                ResolvedValueType::Matrix(matrix) => matrix,
                                _ => unreachable!(),
                            };
                            let body = self
                                .job
                                .programs()
                                .family_body(family)
                                .map_err(|_| "invalid generated family body".to_owned())?;
                            let body_is_indexed_slice =
                                self.job.expressions().node(body).ok().is_some_and(|body_node| {
                                    matches!(
                                        body_node.operator,
                                        ValueOperator::Matrix(MatrixOperation::IndexedSlice { .. })
                                    )
                                });
                            if body_is_indexed_slice {
                                self.authorize_compact_indexed_scalar_call(
                                    expression,
                                    *program,
                                    owner,
                                    is_right_child,
                                    parent_expression,
                                    scalar_type,
                                    other_type,
                                    match self
                                        .job
                                        .expressions()
                                        .value_type(parent_expression)
                                        .map_err(|_| {
                                            "scalar compact product output type is missing"
                                                .to_owned()
                                        })? {
                                        ResolvedValueType::Matrix(matrix) => matrix,
                                        _ => {
                                            return Err(
                                                "scalar compact product output is not a matrix"
                                                    .to_owned(),
                                            )
                                        }
                                    },
                                )?;
                            } else if self.job.expressions().node(body).ok().is_some_and(
                                |body_node| matches!(body_node.operator, ValueOperator::Source(_)),
                            ) {
                                self.authorize_compact_closed_scalar_program_call(
                                    expression,
                                    *program,
                                    owner,
                                    scalar_type,
                                )?;
                            } else {
                                return Err(
                                    "scalar ProgramCall body is not an authorized compact scalar body"
                                        .to_owned(),
                                );
                            }
                            scalar_program_call = true;
                            Some(expression)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    let body_key = CompactCompilerStateKey {
                        expression: body,
                        owner: Some(*program),
                        binding_environment: Some(expression),
                        parent: CompactCompilerParent::Ordinary,
                        under_planned_shell,
                        scalar_call_context: body_scalar_context,
                        binding_context: None,
                        binding_subtree,
                    };
                    enqueue_compact_compiler_state(
                        body_key,
                        Some(state_index),
                        &mut states,
                        &mut state_ids,
                        &active,
                        &mut work,
                    )?;
                    reducible_expansions = reducible_expansions.saturating_add(1);
                }
                ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
                    output: decomposition_type,
                    base,
                    small,
                    digit_count,
                }) => {
                    if binding_subtree {
                        return Err(
                            "compact binding subtree contains a gadget plan occurrence".to_owned()
                        );
                    }
                    if node.inputs.len() != 1 ||
                        !matches!(output, ResolvedValueType::Matrix(_)) ||
                        !matches!(
                            self.job.expressions().value_type(node.inputs[0]),
                            Ok(ResolvedValueType::Matrix(_))
                        )
                    {
                        return Err("gadget decomposition type or arity mismatch".to_owned());
                    }
                    let input = node.inputs[0];
                    let input_node = match self.job.expressions().node(input) {
                        Ok(node) => node,
                        Err(_) => return Err("invalid gadget input authority".to_owned()),
                    };
                    if !matches!(
                        input_node.operator,
                        ValueOperator::Constant(_) |
                            ValueOperator::Source(_) |
                            ValueOperator::Sample { .. } |
                            ValueOperator::Sampler { .. } |
                            ValueOperator::DeterministicHash(_) |
                            ValueOperator::OpaqueFamilyElement { .. } |
                            ValueOperator::ExplicitElement { .. } |
                            ValueOperator::Trapdoor(_)
                    ) {
                        return Err("gadget decomposition input is not a concrete leaf".to_owned());
                    }
                    match self.job.expressions().is_closed(input) {
                        Ok(true) => {}
                        Ok(false) => {
                            return Err("gadget decomposition input is binder-dependent".to_owned())
                        }
                        Err(_) => return Err("invalid gadget input closedness".to_owned()),
                    }
                    let Some(ResolvedValueType::Matrix(input_type)) =
                        self.job.expressions().value_type(input).ok()
                    else {
                        return Err("gadget decomposition input type mismatch".to_owned());
                    };
                    let decomposition_layout =
                        self.job.facts().facts(expression).ok().and_then(|facts| match facts {
                            ValueFacts::Matrix(facts) => Some(facts.metadata.layout.clone()),
                            _ => None,
                        });
                    let input_layout =
                        self.job.facts().facts(input).ok().and_then(|facts| match facts {
                            ValueFacts::Matrix(facts) => Some(facts.metadata.layout.clone()),
                            _ => None,
                        });
                    if !self.job.gadget_recompositions().allows_decomposition_half_unfrozen(
                        *base,
                        *small,
                        *digit_count,
                        decomposition_type,
                        &input_type,
                        decomposition_layout.as_ref(),
                        input_layout.as_ref(),
                    ) {
                        return Err("gadget decomposition contract is uncertain".to_owned());
                    }
                    let Some((parent_expression, is_right_child)) = parent else {
                        return Err(
                            "gadget decomposition has no authorized product consumer".to_owned()
                        );
                    };
                    let parent_node = self
                        .job
                        .expressions()
                        .node(parent_expression)
                        .map_err(|_| "invalid gadget consumer parent".to_owned())?;
                    if !matches!(
                        parent_node.operator,
                        ValueOperator::Matrix(MatrixOperation::Multiply)
                    ) || !is_right_child ||
                        parent_node.inputs.len() != 2 ||
                        parent_node.inputs[1] != expression ||
                        !self
                            .is_gadget_product_operand(parent_node.inputs[0])
                            .map_err(|_| "invalid gadget product operand".to_owned())?
                    {
                        return Err(
                            "gadget decomposition has no authorized product consumer".to_owned()
                        );
                    }
                    let ResolvedValueType::Matrix(decomposition_type) = self
                        .job
                        .expressions()
                        .value_type(expression)
                        .map_err(|_| "compact decomposition type is missing".to_owned())?
                    else {
                        return Err("compact decomposition is not a matrix".to_owned());
                    };
                    let gadget_type = ResolvedMatrixType::new(
                        input_type.modulus.clone(),
                        input_type.ring_dimension,
                        input_type.rows,
                        decomposition_type.rows,
                    )
                    .map_err(|_| "compact gadget type is invalid".to_owned())?;
                    let gadget_layout = MatrixLayout::row_major(
                        input_type.rows,
                        input_type.rows.saturating_mul(*digit_count as usize),
                    );
                    let Some(rule) = self.job.gadget_recompositions().matching_rule_unfrozen(
                        *base,
                        *small,
                        *digit_count,
                        &gadget_type,
                        decomposition_type,
                        &input_type,
                        &input_type,
                        Some(&gadget_layout),
                        decomposition_layout.as_ref(),
                        input_layout.as_ref(),
                    ) else {
                        return Err("compact gadget rule is missing or ambiguous".to_owned());
                    };
                    if !self
                        .gadget_operand_matches(parent_node.inputs[0], &rule)
                        .map_err(|_| "invalid gadget product consumer".to_owned())?
                    {
                        return Err("gadget decomposition consumer rule mismatch".to_owned());
                    }
                    state_plan_delta =
                        Some(CompactCompilerPlanDelta::Gadget { shell: expression, input, rule });
                    let child_key = CompactCompilerStateKey {
                        expression: input,
                        owner,
                        binding_environment: state_key.binding_environment,
                        parent: CompactCompilerParent::Ordinary,
                        under_planned_shell: true,
                        scalar_call_context,
                        binding_context,
                        binding_subtree,
                    };
                    enqueue_compact_compiler_state(
                        child_key,
                        Some(state_index),
                        &mut states,
                        &mut state_ids,
                        &active,
                        &mut work,
                    )?;
                }
                ValueOperator::Matrix(MatrixOperation::IndexedSlice {
                    output: slice_output,
                    layout,
                }) => {
                    if scalar_call_context.is_none() ||
                        node.inputs.len() != 5 ||
                        output != &ResolvedValueType::Matrix(slice_output.clone()) ||
                        *layout != MatrixLayout::row_major(1, 1) ||
                        self.job.facts().facts(expression).is_ok() ||
                        relation_endpoints.contains(&expression)
                    {
                        return Err(
                            "indexed scalar slice is not an authorized compact body".to_owned()
                        );
                    }
                    state_virtual_node = true;
                    for (index, input) in node.inputs.iter().copied().enumerate().rev() {
                        let child_key = CompactCompilerStateKey {
                            expression: input,
                            owner,
                            binding_environment: state_key.binding_environment,
                            parent: self.compact_parent_capability(
                                expression,
                                index,
                                input,
                                &valid_fixed_slices,
                            ),
                            under_planned_shell,
                            scalar_call_context,
                            binding_context: None,
                            binding_subtree,
                        };
                        enqueue_compact_compiler_state(
                            child_key,
                            Some(state_index),
                            &mut states,
                            &mut state_ids,
                            &active,
                            &mut work,
                        )?;
                    }
                }
                ValueOperator::Matrix(
                    operation @ (MatrixOperation::Add |
                    MatrixOperation::Subtract |
                    MatrixOperation::Negate |
                    MatrixOperation::Scale |
                    MatrixOperation::Multiply),
                ) => {
                    let valid = match operation {
                        MatrixOperation::Negate => {
                            matches!(output, ResolvedValueType::Matrix(_)) &&
                                node.inputs.len() == 1 &&
                                self.job.expressions().value_type(node.inputs[0]).ok() ==
                                    Some(output)
                        }
                        MatrixOperation::Add | MatrixOperation::Subtract => {
                            matches!(output, ResolvedValueType::Matrix(_)) &&
                                node.inputs.len() == 2 &&
                                node.inputs.iter().all(|input| {
                                    self.job.expressions().value_type(*input).ok() == Some(output)
                                })
                        }
                        MatrixOperation::Scale => {
                            matches!(output, ResolvedValueType::Matrix(_)) &&
                                node.inputs.len() == 2 &&
                                self.job.expressions().value_type(node.inputs[0]).ok() ==
                                    Some(output) &&
                                self.job.expressions().value_type(node.inputs[1]).ok() ==
                                    Some(&ResolvedValueType::Int) &&
                                self.closed_integer(node.inputs[1]).is_some()
                        }
                        MatrixOperation::Multiply => {
                            match (
                                (node.inputs.len() == 2).then_some(()),
                                node.inputs.first().and_then(|input| {
                                    self.job.expressions().value_type(*input).ok()
                                }),
                                node.inputs.get(1).and_then(|input| {
                                    self.job.expressions().value_type(*input).ok()
                                }),
                                output,
                            ) {
                                (
                                    Some(()),
                                    Some(ResolvedValueType::Matrix(left)),
                                    Some(ResolvedValueType::Matrix(right)),
                                    ResolvedValueType::Matrix(result),
                                ) => {
                                    let left_scalar = left.rows == 1 && left.columns == 1;
                                    let right_scalar = right.rows == 1 && right.columns == 1;
                                    let one_sided_scalar = left_scalar ^ right_scalar;
                                    let both_scalar = left_scalar && right_scalar;
                                    let scalar_leaf = if one_sided_scalar {
                                        let scalar = if left_scalar {
                                            node.inputs[0]
                                        } else {
                                            node.inputs[1]
                                        };
                                        self.job.expressions().node(scalar).ok().is_some_and(
                                            |node| {
                                                (node.inputs.is_empty() &&
                                                    matches!(
                                                        node.operator,
                                                        ValueOperator::Constant(_) |
                                                            ValueOperator::Sampler { .. } |
                                                            ValueOperator::Source(_) |
                                                            ValueOperator::Sample { .. } |
                                                            ValueOperator::ExplicitElement { .. }
                                                    )) ||
                                                    matches!(
                                                        node.operator,
                                                        ValueOperator::ProgramCall { .. }
                                                    )
                                            },
                                        )
                                    } else {
                                        true
                                    };
                                    let valid_shape = if one_sided_scalar {
                                        result.rows ==
                                            if left_scalar { right.rows } else { left.rows } &&
                                            result.columns ==
                                                if right_scalar {
                                                    left.columns
                                                } else {
                                                    right.columns
                                                } &&
                                            (result.rows != 1 || result.columns != 1)
                                    } else {
                                        left.columns == right.rows &&
                                            result.rows == left.rows &&
                                            result.columns == right.columns &&
                                            (both_scalar || (!left_scalar && !right_scalar))
                                    };
                                    scalar_leaf &&
                                        left.modulus == right.modulus &&
                                        left.modulus == result.modulus &&
                                        left.ring_dimension == right.ring_dimension &&
                                        left.ring_dimension == result.ring_dimension &&
                                        valid_shape
                                }
                                _ => false,
                            }
                        }
                        _ => false,
                    };
                    if !valid {
                        self.log_compact_matrix_rejection(expression, &node, output);
                        return Err("virtual matrix operator type or arity mismatch".to_owned());
                    }
                    if self.job.facts().facts(expression).is_ok() {
                        return Err("virtual node has facts".to_owned());
                    }
                    if relation_endpoints.contains(&expression) {
                        return Err("virtual node is a relation endpoint".to_owned());
                    }
                    state_virtual_node = true;
                    for (index, input) in node.inputs.iter().copied().enumerate().rev() {
                        let child_key = CompactCompilerStateKey {
                            expression: input,
                            owner,
                            binding_environment: state_key.binding_environment,
                            parent: self.compact_parent_capability(
                                expression,
                                index,
                                input,
                                &valid_fixed_slices,
                            ),
                            under_planned_shell,
                            scalar_call_context,
                            binding_context: None,
                            binding_subtree,
                        };
                        enqueue_compact_compiler_state(
                            child_key,
                            Some(state_index),
                            &mut states,
                            &mut state_ids,
                            &active,
                            &mut work,
                        )?;
                    }
                }
                ValueOperator::Matrix(MatrixOperation::Tensor { .. }) => {
                    let valid = if node.inputs.len() == 2 {
                        match (
                            self.job.expressions().value_type(node.inputs[0]).ok(),
                            self.job.expressions().value_type(node.inputs[1]).ok(),
                            output,
                            &node.operator,
                        ) {
                            (
                                Some(ResolvedValueType::Matrix(left)),
                                Some(ResolvedValueType::Matrix(right)),
                                ResolvedValueType::Matrix(result),
                                ValueOperator::Matrix(MatrixOperation::Tensor {
                                    output: tensor_output,
                                    left_layout,
                                    right_layout,
                                    output_layout,
                                }),
                            ) => {
                                let scalar_left = left.rows == 1 &&
                                    left.columns == 1 &&
                                    *left_layout == MatrixLayout::row_major(1, 1);
                                let scalar_right = right.rows == 1 &&
                                    right.columns == 1 &&
                                    *right_layout == MatrixLayout::row_major(1, 1);
                                let scalar_side = scalar_left ^ scalar_right;
                                let expected_rows = left.rows.checked_mul(right.rows);
                                let expected_columns = left.columns.checked_mul(right.columns);
                                scalar_side &&
                                    left.modulus == right.modulus &&
                                    left.modulus == result.modulus &&
                                    left.ring_dimension == right.ring_dimension &&
                                    left.ring_dimension == result.ring_dimension &&
                                    tensor_output == result &&
                                    expected_rows == Some(result.rows) &&
                                    expected_columns == Some(result.columns) &&
                                    *left_layout ==
                                        MatrixLayout::row_major(left.rows, left.columns) &&
                                    *right_layout ==
                                        MatrixLayout::row_major(right.rows, right.columns) &&
                                    *output_layout ==
                                        MatrixLayout::row_major(result.rows, result.columns)
                            }
                            _ => false,
                        }
                    } else {
                        false
                    };
                    if !valid {
                        self.log_compact_matrix_rejection(expression, &node, output);
                        return Err("virtual tensor scalar type or arity mismatch".to_owned());
                    }
                    if self.job.facts().facts(expression).is_ok() {
                        return Err("virtual node has facts".to_owned());
                    }
                    if relation_endpoints.contains(&expression) {
                        return Err("virtual node is a relation endpoint".to_owned());
                    }
                    state_virtual_node = true;
                    for (index, input) in node.inputs.iter().copied().enumerate().rev() {
                        let child_key = CompactCompilerStateKey {
                            expression: input,
                            owner,
                            binding_environment: state_key.binding_environment,
                            parent: self.compact_parent_capability(
                                expression,
                                index,
                                input,
                                &valid_fixed_slices,
                            ),
                            under_planned_shell,
                            scalar_call_context,
                            binding_context: None,
                            binding_subtree,
                        };
                        enqueue_compact_compiler_state(
                            child_key,
                            Some(state_index),
                            &mut states,
                            &mut state_ids,
                            &active,
                            &mut work,
                        )?;
                    }
                }
                _ => {
                    let fixed_slice = self.authorize_compact_fixed_slice(
                        expression,
                        &node,
                        output,
                        &relation_endpoints,
                    )?;
                    if fixed_slice {
                        valid_fixed_slices.insert(expression);
                        state_virtual_node = true;
                    }
                    if let ValueOperator::Argument { position, value_type } = &node.operator {
                        let Some(owner) = owner else {
                            return Err("unscoped argument in compact root".to_owned());
                        };
                        let signature = match self.job.programs().program_signature(owner) {
                            Ok(signature) => signature,
                            Err(_) => return Err("invalid argument owner authority".to_owned()),
                        };
                        if signature.inputs.get(*position as usize).map(|input| &input.value_type) !=
                            Some(value_type)
                        {
                            return Err("generated argument type or position mismatch".to_owned());
                        }
                    }
                    let basic_body_operator = matches!(
                        node.operator,
                        ValueOperator::Argument { .. } |
                            ValueOperator::Constant(_) |
                            ValueOperator::Source(_) |
                            ValueOperator::Sample { .. } |
                            ValueOperator::Sampler { .. } |
                            ValueOperator::OpaqueFamilyElement { .. } |
                            ValueOperator::ExplicitElement { .. } |
                            ValueOperator::Trapdoor(_)
                    );
                    // DeterministicHash remains an exact atom.  It is admitted here only as
                    // the single direct child of an authorized fixed Slice; its key/tag inputs
                    // continue through this same owner/binder traversal below.
                    let fixed_slice_child = parent.is_some_and(|(parent_expression, is_right)| {
                        !is_right &&
                            valid_fixed_slices.contains(&parent_expression) &&
                            matches!(node.operator, ValueOperator::DeterministicHash(_))
                    });
                    // Integer index expressions are part of the existing compact binding
                    // vocabulary.  Their exact range is proved by the projector when they are
                    // used as a call binding; the same operator check also admits a reducible
                    // Int family body for traversal under its owner scope.
                    let binding_integer_operator = matches!(output, ResolvedValueType::Int) &&
                        IndexRangeProjector::supports_integer_operator(&node.operator);
                    let scalar_coordinate_operator = scalar_call_context.is_some() &&
                        matches!(node.operator, ValueOperator::Scalar(_));
                    if owner.is_some() &&
                        !(basic_body_operator ||
                            scalar_coordinate_operator ||
                            binding_integer_operator ||
                            fixed_slice ||
                            fixed_slice_child)
                    {
                        self.log_compact_concrete_rejection(
                            root, expression, &node, output, owner, parent,
                        );
                        return Err(
                            "generated body contains unsupported concrete operator".to_owned()
                        );
                    }
                    for (index, input) in node.inputs.iter().copied().enumerate().rev() {
                        let child_key = CompactCompilerStateKey {
                            expression: input,
                            owner,
                            binding_environment: state_key.binding_environment,
                            parent: self.compact_parent_capability(
                                expression,
                                index,
                                input,
                                &valid_fixed_slices,
                            ),
                            under_planned_shell,
                            scalar_call_context,
                            binding_context: None,
                            binding_subtree,
                        };
                        enqueue_compact_compiler_state(
                            child_key,
                            Some(state_index),
                            &mut states,
                            &mut state_ids,
                            &active,
                            &mut work,
                        )?;
                    }
                }
            }
            if let Ok(ResolvedValueType::Matrix(matrix)) =
                self.job.expressions().value_type(expression)
            {
                let scalar_product_parent = parent.is_some_and(|(parent_expression, _)| {
                    let Some(parent_node) = self.job.expressions().node(parent_expression).ok()
                    else {
                        return false;
                    };
                    if !matches!(
                        parent_node.operator,
                        ValueOperator::Matrix(MatrixOperation::Multiply) |
                            ValueOperator::Matrix(MatrixOperation::Tensor { .. })
                    ) || parent_node.inputs.len() != 2
                    {
                        return false;
                    }
                    parent.and_then(|(_, is_right)| {
                        self.job
                            .expressions()
                            .value_type(parent_node.inputs[usize::from(!is_right)])
                            .ok()
                    })
                    .is_some_and(|value_type| {
                        matches!(value_type, ResolvedValueType::Matrix(matrix) if matrix.rows != 1 || matrix.columns != 1)
                    })
                });
                if matrix.rows == 1 && matrix.columns == 1 && scalar_product_parent {
                    if binding_subtree {
                        return Err(
                            "compact binding subtree contains a scalar plan occurrence".to_owned()
                        );
                    }
                    let Some((parent_expression, is_right_child)) = parent else {
                        return Err(
                            "scalar compact factor has no authorized product consumer".to_owned()
                        );
                    };
                    if !scalar_program_call &&
                        (!node.inputs.is_empty() ||
                            !matches!(
                                node.operator,
                                ValueOperator::Constant(_) |
                                    ValueOperator::Sampler { .. } |
                                    ValueOperator::Source(_) |
                                    ValueOperator::Sample { .. } |
                                    ValueOperator::ExplicitElement { .. }
                            ))
                    {
                        return Err("scalar compact factor is not a concrete leaf".to_owned());
                    }
                    let parent_node = self
                        .job
                        .expressions()
                        .node(parent_expression)
                        .map_err(|_| "invalid scalar consumer parent".to_owned())?;
                    if matches!(
                        parent_node.operator,
                        ValueOperator::Matrix(MatrixOperation::Tensor { .. })
                    ) && !scalar_program_call
                    {
                        return Err(
                            "tensor scalar consumer requires indexed scalar ProgramCall".to_owned()
                        );
                    }
                    if !matches!(
                        parent_node.operator,
                        ValueOperator::Matrix(MatrixOperation::Multiply) |
                            ValueOperator::Matrix(MatrixOperation::Tensor { .. })
                    ) || parent_node.inputs.len() != 2 ||
                        parent_node.inputs[usize::from(is_right_child)] != expression
                    {
                        return Err("scalar compact factor has a non-product parent".to_owned());
                    }
                    let other = parent_node.inputs[usize::from(!is_right_child)];
                    let ResolvedValueType::Matrix(other_type) =
                        self.job.expressions().value_type(other).map_err(|_| {
                            "scalar compact factor sibling type is missing".to_owned()
                        })?
                    else {
                        return Err("scalar compact factor sibling is not a matrix".to_owned());
                    };
                    if other_type.rows == 1 && other_type.columns == 1 {
                        return Err(
                            "scalar compact product requires one non-scalar operand".to_owned()
                        );
                    }
                    if matches!(
                        parent_node.operator,
                        ValueOperator::Matrix(MatrixOperation::Tensor { .. })
                    ) && !matches!(
                        self.job
                            .expressions()
                            .node(other)
                            .map_err(|_| "tensor scalar sibling authority is missing".to_owned())?
                            .operator,
                        ValueOperator::Matrix(MatrixOperation::Multiply)
                    ) {
                        return Err(
                            "tensor scalar consumer requires Matrix::Multiply sibling".to_owned()
                        );
                    }
                    if scalar_program_call {
                        // The weighted state finalization below inserts this exact token with its
                        // checked occurrence multiplicity.
                    } else {
                        let bound = self
                            .job
                            .facts()
                            .coefficient_bound(expression)
                            .map_err(|_| "scalar compact factor bound is missing".to_owned())?;
                        if !matches!(bound, NumericContract::Known(CoefficientBound::Finite(_))) {
                            return Err("scalar compact factor bound is not finite".to_owned());
                        }
                    }
                    if scalar_program_call {
                        state_plan_delta = Some(CompactCompilerPlanDelta::ScalarProgramCall {
                            consumer: parent_expression,
                            call: expression,
                            scalar_is_right: is_right_child,
                            value_type: matrix.clone(),
                        });
                    } else {
                        state_plan_delta = Some(CompactCompilerPlanDelta::Scalar {
                            expression,
                            value_type: matrix.clone(),
                        });
                        authorized_scalars.insert(expression);
                    }
                } else if node.inputs.is_empty() {
                    other_scalar_parents.insert(expression);
                }
            }
            states[state_index].virtual_node = state_virtual_node;
            states[state_index].plan_delta = state_plan_delta;
            if state_virtual_node {
                virtual_nodes = virtual_nodes.saturating_add(1);
            }
            work_peak = work_peak.max(work.len());
            let processed_states = order.len() + active.len();
            if processed_states != 0 &&
                (processed_states.is_power_of_two() || processed_states % 1_000_000 == 0)
            {
                info!(
                    target: "mxx_correctness::operational_noise",
                    stage = "compile_compact_root",
                    event = "progress",
                    root_kind,
                    unique_states = states.len(),
                    processed_states,
                    pending_work = work.len(),
                    active_depth = active.len(),
                    work_peak,
                    reducible_expansions,
                    range_environments = index_projector.environments.len(),
                    range_memo_entries = index_projector.memo.len(),
                    "compact compiler progress"
                );
            }
        }
        if virtual_nodes == 0 {
            return Err("no virtual matrix algebra".to_owned());
        }
        if authorized_scalars.iter().any(|expression| other_scalar_parents.contains(expression)) {
            return Err("scalar compact factor has mixed consumer parents".to_owned());
        }
        // The enter/exit walk gives a useful cycle check, but its reversed exit order is not a
        // topological order when a later sibling points at a state that was already interned by
        // an earlier sibling.  Compute a deterministic parent-before-child order over the
        // completed unique-state graph before propagating any multiplicity.
        let mut indegree = vec![0_usize; states.len()];
        for state in &states {
            for &child in state.children.keys() {
                indegree[child] = indegree[child]
                    .checked_add(1)
                    .ok_or_else(|| "compact state indegree overflow".to_owned())?;
            }
        }
        let mut ready = BTreeSet::new();
        for (state_index, degree) in indegree.iter().copied().enumerate() {
            if degree == 0 {
                ready.insert(state_index);
            }
        }
        let mut topological = Vec::with_capacity(states.len());
        while let Some(state_index) = ready.iter().next().copied() {
            ready.remove(&state_index);
            topological.push(state_index);
            let children = states[state_index].children.keys().copied().collect::<Vec<_>>();
            for child in children {
                let degree = indegree
                    .get_mut(child)
                    .ok_or_else(|| "compact state child index is invalid".to_owned())?;
                *degree = degree
                    .checked_sub(1)
                    .ok_or_else(|| "compact state indegree underflow".to_owned())?;
                if *degree == 0 {
                    ready.insert(child);
                }
            }
        }
        if topological.len() != states.len() {
            return Err("cycle or recursive generated body".to_owned());
        }
        states[0].multiplicity = 1;
        for state_index in topological {
            let count = states[state_index].multiplicity;
            if count == 0 {
                continue;
            }
            plan.preflight_node_occurrences = plan
                .preflight_node_occurrences
                .checked_add(count)
                .ok_or_else(|| "compact occurrence multiplicity overflow".to_owned())?;
            if let Some(delta) = states[state_index].plan_delta.clone() {
                match delta {
                    CompactCompilerPlanDelta::Gadget { shell, input, rule } => {
                        if !plan.insert_gadget_count(shell, input, rule, count) {
                            return Err("compact gadget rule identity changed".to_owned());
                        }
                    }
                    CompactCompilerPlanDelta::Scalar { expression, value_type } => {
                        if !plan.insert_scalar_count(expression, value_type, count) {
                            return Err("compact scalar factor type identity changed".to_owned());
                        }
                    }
                    CompactCompilerPlanDelta::ScalarProgramCall {
                        consumer,
                        call,
                        scalar_is_right,
                        value_type,
                    } => {
                        if !plan.insert_scalar_program_call_count(
                            consumer,
                            call,
                            scalar_is_right,
                            value_type,
                            count,
                        ) {
                            return Err(
                                "indexed scalar ProgramCall type identity changed".to_owned()
                            );
                        }
                    }
                }
            }
            let children = states[state_index].children.clone();
            for (child, edge_count) in children {
                let contribution = count
                    .checked_mul(edge_count)
                    .ok_or_else(|| "compact occurrence multiplicity overflow".to_owned())?;
                states[child].multiplicity =
                    states[child]
                        .multiplicity
                        .checked_add(contribution)
                        .ok_or_else(|| "compact occurrence multiplicity overflow".to_owned())?;
            }
        }
        let unique_edges = states.iter().map(|state| state.children.len()).sum::<usize>();
        let unique_gadget_consumers = states
            .iter()
            .filter(|state| {
                matches!(state.plan_delta, Some(CompactCompilerPlanDelta::Gadget { .. }))
            })
            .count();
        info!(
            target: "mxx_correctness::operational_noise",
            stage = "compile_compact_root",
            event = "complete",
            root_kind,
            unique_states = states.len(),
            unique_edges,
            work_peak,
            reducible_expansions,
            range_environments = index_projector.environments.len(),
            range_memo_entries = index_projector.memo.len(),
            unique_gadget_consumers,
            preflight_node_occurrences = plan.preflight_node_occurrences,
            "compact compiler complete"
        );
        Ok(plan)
    }

    /// Authorize the fixed Slice form used by Tall bodies. Slice coordinates are already
    /// resolved by lowering, so this is a contract check rather than a second graph traversal.
    /// A DeterministicHash child receives an additional exact descriptor refinement; every other
    /// source remains subject to the common owner-scoped compiler vocabulary below.
    fn authorize_compact_fixed_slice(
        &self,
        expression: ExprId,
        node: &ExprNode,
        output: &ResolvedValueType,
        relation_endpoints: &BTreeSet<ExprId>,
    ) -> Result<bool, String> {
        let ValueOperator::Matrix(MatrixOperation::Slice {
            row_start,
            row_end_exclusive,
            column_start,
            column_end_exclusive,
            layout,
        }) = &node.operator
        else {
            return Ok(false);
        };
        let ResolvedValueType::Matrix(output_type) = output else {
            return Err("fixed Slice output is not a matrix".to_owned());
        };
        let [input] = node.inputs.as_ref() else {
            return Err("fixed Slice must have one input".to_owned());
        };
        let ResolvedValueType::Matrix(input_type) = self
            .job
            .expressions()
            .value_type(*input)
            .map_err(|_| "fixed Slice input type is missing".to_owned())?
        else {
            return Err("fixed Slice input is not a matrix".to_owned());
        };
        let input_node = self
            .job
            .expressions()
            .node(*input)
            .map_err(|_| "fixed Slice input authority is missing".to_owned())?;
        if input_type.modulus != output_type.modulus ||
            input_type.ring_dimension != output_type.ring_dimension
        {
            return Err("fixed Slice input/output ring contract mismatch".to_owned());
        }
        if let ValueOperator::DeterministicHash(descriptor) = &input_node.operator {
            if descriptor.binary_tag_count != 0 ||
                descriptor.decimal_tag_count != 0 ||
                descriptor.u64_le_tag_count != 0 ||
                descriptor.dynamic_tag_count != 0 ||
                input_node.inputs.len() != 1
            {
                return Err("fixed Slice hash has unsupported tags".to_owned());
            }
            if descriptor.output != *input_type {
                return Err("fixed Slice input/output ring contract mismatch".to_owned());
            }
        }
        if *row_start >= *row_end_exclusive ||
            *column_start >= *column_end_exclusive ||
            *row_end_exclusive > input_type.rows ||
            *column_end_exclusive > input_type.columns
        {
            return Err("fixed Slice coordinates are empty or out of bounds".to_owned());
        }
        let rows = row_end_exclusive - row_start;
        let columns = column_end_exclusive - column_start;
        if output_type.rows != rows ||
            output_type.columns != columns ||
            *layout != MatrixLayout::row_major(rows, columns)
        {
            return Err("fixed Slice output geometry or layout mismatch".to_owned());
        }
        if self.job.facts().facts(expression).is_ok() {
            return Err("fixed Slice node has facts".to_owned());
        }
        if relation_endpoints.contains(&expression) {
            return Err("fixed Slice node is a relation endpoint".to_owned());
        }
        Ok(true)
    }

    #[cfg(test)]
    fn compact_residual_preflight(&self, value: &Value) -> Option<String> {
        self.compile_compact_root(value).err()
    }

    /// Test-only compatibility view of the unified compiler's rejection result.
    #[cfg(test)]
    fn build_compact_shell_plan(&self, value: &Value) -> Result<CompactShellPlan, String> {
        self.compile_compact_root(value)
    }

    /// Authorize a scalar family whose body is already a closed concrete Source. The ordinary
    /// compact evaluator can reduce this body directly; this check only proves the same exact
    /// owner, range, binding, and scalar identity needed by the product token.
    fn authorize_compact_closed_scalar_program_call(
        &self,
        call: ExprId,
        program: ValueProgramId,
        owner: Option<ValueProgramId>,
        scalar_type: &ResolvedMatrixType,
    ) -> Result<(), String> {
        let Some(owner) = owner else {
            return Err("scalar ProgramCall has no owner binder".to_owned());
        };
        let Some(family) = self.job.programs().family_for_program(program) else {
            return Err("scalar ProgramCall has no family authority".to_owned());
        };
        if !self.job.programs().family_is_reducible(family).unwrap_or(false) {
            return Err("scalar ProgramCall is opaque".to_owned());
        }
        let signature = self
            .job
            .programs()
            .program_signature(program)
            .map_err(|_| "scalar family signature is missing".to_owned())?;
        let [input] = signature.inputs.as_ref() else {
            return Err("scalar family must be unary".to_owned());
        };
        let Some(callee_range) = input.trusted_index_range else {
            return Err("scalar family range is missing".to_owned());
        };
        if input.value_type != ResolvedValueType::Int ||
            signature.output != ResolvedValueType::Matrix(scalar_type.clone())
        {
            return Err("scalar family signature type mismatch".to_owned());
        }
        let owner_signature = self
            .job
            .programs()
            .program_signature(owner)
            .map_err(|_| "scalar owner signature is missing".to_owned())?;
        let [owner_input] = owner_signature.inputs.as_ref() else {
            return Err("scalar owner must be unary".to_owned());
        };
        if owner_input.value_type != ResolvedValueType::Int ||
            owner_input.trusted_index_range != Some(callee_range)
        {
            return Err("scalar owner/callee range mismatch".to_owned());
        }
        let call_node = self
            .job
            .expressions()
            .node(call)
            .map_err(|_| "scalar call authority is missing".to_owned())?;
        let [binding] = call_node.inputs.as_ref() else {
            return Err("scalar call must have one binding".to_owned());
        };
        let ValueOperator::Argument { position: 0, value_type } = &self
            .job
            .expressions()
            .node(*binding)
            .map_err(|_| "scalar binding is missing".to_owned())?
            .operator
        else {
            return Err("scalar binding is not owner Argument(0)".to_owned());
        };
        if *value_type != ResolvedValueType::Int {
            return Err("scalar binding type mismatch".to_owned());
        }
        let body = self
            .job
            .programs()
            .family_body(family)
            .map_err(|_| "scalar family body is missing".to_owned())?;
        let body_node = self
            .job
            .expressions()
            .node(body)
            .map_err(|_| "scalar family body authority is missing".to_owned())?;
        if !matches!(body_node.operator, ValueOperator::Source(_)) ||
            !body_node.inputs.is_empty() ||
            self.job.expressions().value_type(body).ok() !=
                Some(&ResolvedValueType::Matrix(scalar_type.clone())) ||
            !self.job.expressions().is_closed(body).unwrap_or(false)
        {
            return Err("scalar family body is not a closed Source".to_owned());
        }
        Ok(())
    }

    /// Authorize the one actual Tall scalar-family shape.  This is deliberately narrower than
    /// the ordinary generated-call validator: the call is only a unary Int-indexed family whose
    /// body is one exact unit `IndexedSlice`.  The compact evaluator still evaluates that slice;
    /// this method only proves that its identity and coordinate contract are closed enough for
    /// the typed scalar action at the surrounding product boundary.
    fn authorize_compact_indexed_scalar_call(
        &self,
        call: ExprId,
        program: ValueProgramId,
        owner: Option<ValueProgramId>,
        scalar_is_right: bool,
        consumer: ExprId,
        scalar_type: &ResolvedMatrixType,
        other_type: &ResolvedMatrixType,
        output_type: &ResolvedMatrixType,
    ) -> Result<(), String> {
        let Some(owner) = owner else {
            return Err("indexed scalar ProgramCall has no owner binder".to_owned());
        };
        let Some(family) = self.job.programs().family_for_program(program) else {
            return Err("indexed scalar ProgramCall has no family authority".to_owned());
        };
        if !self.job.programs().family_is_reducible(family).unwrap_or(false) {
            return Err("indexed scalar ProgramCall is opaque".to_owned());
        }
        let signature = self
            .job
            .programs()
            .program_signature(program)
            .map_err(|_| "indexed scalar family signature is missing".to_owned())?;
        let [input] = signature.inputs.as_ref() else {
            return Err("indexed scalar family must be unary".to_owned());
        };
        let Some(callee_range) = input.trusted_index_range else {
            return Err("indexed scalar family range is missing".to_owned());
        };
        if input.value_type != ResolvedValueType::Int ||
            signature.output != ResolvedValueType::Matrix(scalar_type.clone())
        {
            return Err("indexed scalar family signature type mismatch".to_owned());
        }
        let owner_signature = self
            .job
            .programs()
            .program_signature(owner)
            .map_err(|_| "indexed scalar owner signature is missing".to_owned())?;
        let [owner_input] = owner_signature.inputs.as_ref() else {
            return Err("indexed scalar owner must be unary".to_owned());
        };
        if owner_input.value_type != ResolvedValueType::Int ||
            owner_input.trusted_index_range != Some(callee_range)
        {
            return Err("indexed scalar owner/callee range mismatch".to_owned());
        }
        let call_node = self
            .job
            .expressions()
            .node(call)
            .map_err(|_| "indexed scalar call authority is missing".to_owned())?;
        let [binding] = call_node.inputs.as_ref() else {
            return Err("indexed scalar call must have one binding".to_owned());
        };
        let ValueOperator::Argument { position: 0, value_type } = &self
            .job
            .expressions()
            .node(*binding)
            .map_err(|_| "indexed scalar binding is missing".to_owned())?
            .operator
        else {
            return Err("indexed scalar binding is not owner Argument(0)".to_owned());
        };
        if *value_type != ResolvedValueType::Int {
            return Err("indexed scalar binding type mismatch".to_owned());
        }
        let body = self
            .job
            .programs()
            .family_body(family)
            .map_err(|_| "indexed scalar family body is missing".to_owned())?;
        let body_node = self
            .job
            .expressions()
            .node(body)
            .map_err(|_| "indexed scalar family body authority is missing".to_owned())?;
        let ValueOperator::Matrix(MatrixOperation::IndexedSlice { output, layout }) =
            &body_node.operator
        else {
            self.log_compact_indexed_scalar_body_rejection(
                call,
                program,
                family,
                owner,
                body,
                body_node,
                consumer,
                scalar_is_right,
            );
            return Err("indexed scalar family body is not IndexedSlice".to_owned());
        };
        if output != scalar_type ||
            *layout != MatrixLayout::row_major(1, 1) ||
            body_node.inputs.len() != 5 ||
            self.job.expressions().value_type(body).ok() !=
                Some(&ResolvedValueType::Matrix(scalar_type.clone()))
        {
            return Err("indexed scalar IndexedSlice contract mismatch".to_owned());
        }
        let ResolvedValueType::Matrix(source_type) = self
            .job
            .expressions()
            .value_type(body_node.inputs[0])
            .map_err(|_| "indexed scalar source type is missing".to_owned())?
        else {
            return Err("indexed scalar source is not a matrix".to_owned());
        };
        if source_type.modulus != scalar_type.modulus ||
            source_type.ring_dimension != scalar_type.ring_dimension ||
            source_type.modulus != other_type.modulus ||
            source_type.ring_dimension != other_type.ring_dimension
        {
            return Err("indexed scalar source ring mismatch".to_owned());
        }
        let source_node = self
            .job
            .expressions()
            .node(body_node.inputs[0])
            .map_err(|_| "indexed scalar source authority is missing".to_owned())?;
        let allowed_source = matches!(
            source_node.operator,
            ValueOperator::Source(_) |
                ValueOperator::OpaqueFamilyElement { .. } |
                ValueOperator::Sampler {
                    operation: SamplerOperation::UniformResidue { .. } |
                        SamplerOperation::UniformInterval { .. },
                    ..
                }
        );
        if !allowed_source ||
            !source_node.inputs.is_empty() ||
            !self.job.expressions().is_closed(body_node.inputs[0]).unwrap_or(false)
        {
            return Err("indexed scalar source is not a closed Tall source leaf".to_owned());
        }
        let mut formal_argument = None;
        for endpoint in &body_node.inputs[1..] {
            if matches!(
                self.job.expressions().node(*endpoint).ok().map(|node| &node.operator),
                Some(ValueOperator::Argument { position: 0, value_type: ResolvedValueType::Int })
            ) {
                formal_argument = Some(*endpoint);
                break;
            }
        }
        let Some(formal_argument) = formal_argument else {
            return Err("indexed scalar slice has no formal Argument(0)".to_owned());
        };
        if *binding != formal_argument {
            return Err("indexed scalar binding is not the owner Argument(0)".to_owned());
        }
        let forms = body_node.inputs[1..]
            .iter()
            .map(|endpoint| {
                affine_form_for_argument(&self.job, *endpoint, formal_argument)
                    .ok_or_else(|| "indexed scalar coordinate is not owner-affine".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let check_axis = |start: &(BigInt, BigInt), end: &(BigInt, BigInt), extent: usize| {
            let span = exact_indexed_slice_span(
                if start.0.is_zero() { None } else { Some(0) },
                &start.0,
                &start.1,
                if end.0.is_zero() { None } else { Some(0) },
                &end.0,
                &end.1,
                extent,
                1,
            )?;
            if span != 1 {
                return Err("indexed scalar slice span is not one".to_owned());
            }
            let (minimum, maximum_exclusive) = if start.0.is_zero() {
                (start.1.clone(), &start.1 + 1_u8)
            } else {
                let (minimum, maximum) =
                    affine_range(callee_range, start.0.clone(), start.1.clone());
                (minimum, maximum)
            };
            if minimum < BigInt::from(0_u8) || maximum_exclusive > BigInt::from(extent as u64) {
                return Err("indexed scalar slice coordinate range is out of bounds".to_owned());
            }
            Ok(())
        };
        check_axis(&forms[0], &forms[1], source_type.rows)?;
        check_axis(&forms[2], &forms[3], source_type.columns)?;
        let consumer_node = self
            .job
            .expressions()
            .node(consumer)
            .map_err(|_| "indexed scalar consumer authority is missing".to_owned())?;
        let ValueOperator::Matrix(consumer_operation) = &consumer_node.operator else {
            return Err("indexed scalar consumer is not a matrix operation".to_owned());
        };
        match consumer_operation {
            MatrixOperation::Multiply => {
                if output_type != other_type {
                    return Err("indexed scalar product output type mismatch".to_owned());
                }
            }
            MatrixOperation::Tensor { output, left_layout, right_layout, output_layout } => {
                let sibling = consumer_node
                    .inputs
                    .get(usize::from(!scalar_is_right))
                    .ok_or("tensor scalar consumer requires Matrix::Multiply sibling".to_owned())?;
                if !matches!(
                    self.job
                        .expressions()
                        .node(*sibling)
                        .map_err(|_| "tensor scalar consumer requires Matrix::Multiply sibling"
                            .to_owned())?
                        .operator,
                    ValueOperator::Matrix(MatrixOperation::Multiply)
                ) {
                    return Err(
                        "tensor scalar consumer requires Matrix::Multiply sibling".to_owned()
                    );
                }
                if consumer_node.inputs.len() != 2 ||
                    consumer_node.inputs[usize::from(scalar_is_right)] != call ||
                    *left_layout !=
                        MatrixLayout::row_major(
                            if scalar_is_right { other_type.rows } else { scalar_type.rows },
                            if scalar_is_right { other_type.columns } else { scalar_type.columns },
                        ) ||
                    *right_layout !=
                        MatrixLayout::row_major(
                            if scalar_is_right { scalar_type.rows } else { other_type.rows },
                            if scalar_is_right { scalar_type.columns } else { other_type.columns },
                        ) ||
                    *output_layout != MatrixLayout::row_major(output.rows, output.columns) ||
                    *output != *output_type ||
                    output_type != other_type
                {
                    return Err("indexed scalar tensor contract mismatch".to_owned());
                }
            }
            _ => return Err("indexed scalar consumer is not Multiply or Tensor".to_owned()),
        }
        Ok(())
    }

    fn is_gadget_product_operand(&self, root: ExprId) -> Result<bool, ProductionAdapterError> {
        let mut work = vec![root];
        while let Some(expression) = work.pop() {
            let node = self.job.expressions().node(expression)?;
            if matches!(
                node.operator,
                ValueOperator::Source(super::arena::SemanticSourceIdentity {
                    matrix_constant: Some(MatrixConstantKind::Gadget { .. }),
                    ..
                })
            ) {
                continue;
            }
            if matches!(node.operator, ValueOperator::Matrix(MatrixOperation::Add)) &&
                node.inputs.len() == 2
            {
                work.extend(node.inputs.iter().copied());
                continue;
            }
            return Ok(false);
        }
        Ok(true)
    }

    fn gadget_operand_matches(
        &self,
        root: ExprId,
        rule: &GadgetRecompositionRule,
    ) -> Result<bool, ProductionAdapterError> {
        let mut work = vec![root];
        while let Some(expression) = work.pop() {
            let node = self.job.expressions().node(expression)?;
            let ValueOperator::Source(source) = &node.operator else {
                if matches!(node.operator, ValueOperator::Matrix(MatrixOperation::Add)) &&
                    node.inputs.len() == 2
                {
                    work.extend(node.inputs.iter().copied());
                    continue;
                }
                return Ok(false);
            };
            if !matches!(
                source.matrix_constant,
                Some(MatrixConstantKind::Gadget { base, small })
                    if base == rule.base && small == rule.small
            ) || self.job.expressions().value_type(expression)? !=
                &ResolvedValueType::Matrix(rule.gadget_type.clone())
            {
                return Ok(false);
            }
            let layout = self.job.facts().facts(expression).ok().and_then(|facts| match facts {
                ValueFacts::Matrix(facts) => Some(facts.metadata.layout.clone()),
                _ => None,
            });
            if layout.as_ref() != rule.gadget_layout.as_ref() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Defensively revalidate and eagerly intern only the exact closed gadget shells admitted by
    /// the compiler. This phase runs after relation registration but before the freeze barrier;
    /// it never discovers candidates or copies facts. Closedness and reducible-call absence are
    /// compiler invariants, so a violation fails closed rather than selecting eager after freeze.
    fn materialize_compact_shell_plan(
        &mut self,
        plan: &mut CompactShellPlan,
    ) -> Result<(), ProductionAdapterError> {
        let shells = plan.gadget_shells.keys().copied().collect::<Vec<_>>();
        for (shell, input) in shells {
            let rule = plan
                .gadget_shells
                .get(&(shell, input))
                .ok_or_else(|| ProductionAdapterError::Structural {
                    wire: self.plan.target().residual.clone(),
                    reason: "compact shell plan entry disappeared".to_owned(),
                })?
                .rule
                .clone();
            let node = self.job.expressions().node(shell)?.clone();
            let ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
                base,
                small,
                digit_count,
                ..
            }) = &node.operator
            else {
                return Err(ProductionAdapterError::Structural {
                    wire: self.plan.target().residual.clone(),
                    reason: "planned compact shell operator changed".to_owned(),
                });
            };
            if node.inputs.as_ref() != [input] ||
                *base != rule.base ||
                *small != rule.small ||
                *digit_count != rule.digit_count
            {
                return Err(ProductionAdapterError::Structural {
                    wire: self.plan.target().residual.clone(),
                    reason: "planned compact shell contract changed".to_owned(),
                });
            }
            let input_type = self.job.expressions().value_type(input)?;
            let output_type = self.job.expressions().value_type(shell)?;
            if input_type != &ResolvedValueType::Matrix(rule.input_type.clone()) ||
                output_type != &ResolvedValueType::Matrix(rule.decomposition_type.clone())
            {
                return Err(ProductionAdapterError::Structural {
                    wire: self.plan.target().residual.clone(),
                    reason: "planned compact shell typed contract changed".to_owned(),
                });
            }
            self.job
                .programs()
                .ensure_no_reducible_generated_calls(self.job.expressions(), shell)?;
            if !self.job.expressions().is_closed(shell)? {
                return Err(ProductionAdapterError::Structural {
                    wire: self.plan.target().residual.clone(),
                    reason: "planned compact shell is not closed".to_owned(),
                });
            }
            let before = self.job.expressions().node_count();
            let materialized = self
                .job
                .materialize_reducible_generated_calls_with_reason(shell, BetaReason::Other)?;
            self.job
                .programs()
                .ensure_no_reducible_generated_calls(self.job.expressions(), materialized)?;
            let after = self.job.expressions().node_count();
            plan.shell_allocated = plan.shell_allocated.saturating_add(1);
            if after == before {
                plan.shell_hits = plan.shell_hits.saturating_add(1);
            } else {
                plan.shell_new = plan.shell_new.saturating_add((after - before) as u64);
            }
        }
        Ok(())
    }

    /// Validate one reducible-call binding as expression data.  Binding expressions are never
    /// sent through polynomial normalization: this pass checks their exact arena identity,
    /// output type, closedness, call signatures, and descendants before the root is selected.
    fn log_compact_open_binding_rejection(
        &self,
        expression: ExprId,
        node: &super::arena::ExprNode,
        expected_type: &ResolvedValueType,
        expected_range: Option<TrustedIndexRange>,
        owner: Option<ValueProgramId>,
    ) {
        if !tracing::enabled!(
            target: "mxx_correctness::operational_noise",
            tracing::Level::INFO
        ) {
            return;
        }
        let owner_signature =
            owner.and_then(|owner| self.job.programs().program_signature(owner).ok());
        let child = node.inputs.first().copied();
        let child_node = child.and_then(|child| self.job.expressions().node(child).ok());
        let child_program = child_node.and_then(|node| match node.operator {
            ValueOperator::ProgramCall { program } => Some(program),
            _ => None,
        });
        let child_family =
            child_program.and_then(|program| self.job.programs().family_for_program(program));
        let child_reducible =
            child_family.and_then(|family| self.job.programs().family_is_reducible(family).ok());
        let child_signature =
            child_program.and_then(|program| self.job.programs().program_signature(program).ok());
        let child_body = child_program.and_then(|program| {
            self.job.programs().program(program).ok().map(|program| program.root)
        });
        let child_body_node = child_body.and_then(|body| self.job.expressions().node(body).ok());
        let child_body_type =
            child_body.and_then(|body| self.job.expressions().value_type(body).ok());
        let child_body_closed =
            child_body.and_then(|body| self.job.expressions().is_closed(body).ok());
        let child_body_shape =
            child_body_node.map(|node| compact_binding_operator_shape(&node.operator));
        let child_body_input_count = child_body_node.map(|node| node.inputs.len());
        let (extract_position, extract_upper) = match &node.operator {
            ValueOperator::ExtractCoefficient { position, canonical_input_exclusive_upper } => {
                (Some(*position), canonical_input_exclusive_upper.as_ref())
            }
            _ => (None, None),
        };
        let projected_range = self.direct_canonical_extract_range(expression);
        let projected_range_authority = if projected_range.is_some() {
            "canonical_extract_coefficient"
        } else if expected_range.is_some() {
            "caller_or_index_projector"
        } else {
            "none"
        };
        info!(
            target: "mxx_correctness::operational_noise",
            stage = "compact_binding_preflight",
            event = "rejected_open_binding",
            binding_expr = ?expression,
            operator = compact_binding_operator_shape(&node.operator),
            input_count = node.inputs.len(),
            child_expr = ?child,
            child_operator = child_node.map(|node| compact_binding_operator_shape(&node.operator)),
            child_input_count = child_node.map(|node| node.inputs.len()),
            child_program = ?child_program,
            child_family = ?child_family,
            child_reducible = ?child_reducible,
            child_signature_input_count = child_signature.map(|signature| signature.inputs.len()),
            child_signature_output = ?child_signature.map(|signature| &signature.output),
            child_signature_input_range = ?child_signature
                .and_then(|signature| signature.inputs.first().and_then(|input| input.trusted_index_range)),
            child_body = ?child_body,
            child_body_operator = child_body_shape,
            child_body_input_count,
            child_body_type = ?child_body_type,
            child_body_closed = ?child_body_closed,
            extract_position,
            extract_canonical_upper = ?extract_upper,
            expected_type = ?expected_type,
            expected_range = ?expected_range,
            projected_range = ?projected_range,
            projected_range_authority,
            projector_program_call_hits = self.selector_range_projected_program_call_hits,
            projector_nodes = self.selector_range_projector_nodes,
            projector_fallback_materializations = self.selector_range_fallback_materializations,
            owner = ?owner,
            owner_input_count = owner_signature.as_ref().map_or(0, |signature| signature.inputs.len()),
            owner_input_type = ?owner_signature
                .as_ref()
                .and_then(|signature| signature.inputs.first().map(|input| &input.value_type)),
            owner_input_range = ?owner_signature
                .as_ref()
                .and_then(|signature| signature.inputs.first().and_then(|input| input.trusted_index_range)),
            owner_output = ?owner_signature.as_ref().map(|signature| &signature.output),
            "compact family binding rejected: open expression is not the authorized formal index"
        );
    }

    /// Emit one bounded summary for a virtual matrix node rejected before relation freezing.
    /// Input types are rendered individually so malformed arity/geometry is diagnosable without
    /// traversing or formatting the residual graph.
    fn log_compact_matrix_rejection(
        &self,
        expression: ExprId,
        node: &super::arena::ExprNode,
        output: &ResolvedValueType,
    ) {
        let detailed = tracing::enabled!(
            target: "mxx_correctness::operational_noise",
            tracing::Level::INFO
        );
        if matches!(node.operator, ValueOperator::Matrix(MatrixOperation::Multiply)) && detailed {
            for (operand_index, input) in node.inputs.iter().copied().take(2).enumerate() {
                let (operator, input_count) = self
                    .job
                    .expressions()
                    .node(input)
                    .map(|child| {
                        (compact_matrix_operator_shape(&child.operator), child.inputs.len())
                    })
                    .unwrap_or(("<invalid>", 0));
                let closed = self.job.expressions().is_closed(input).ok();
                let (concrete_finite_scalar_leaf, failure_category) =
                    self.compact_scalar_leaf_diagnostic(input);
                info!(
                    target: "mxx_correctness::operational_noise",
                    stage = "compile_compact_root",
                    event = "rejected_scalar_operand",
                    expression = ?expression,
                    operand_index,
                    operator,
                    input_count,
                    closed = ?closed,
                    concrete_finite_scalar_leaf,
                    failure_category,
                    "compact multiply operand diagnostic"
                );
            }
        }
        let input_types = if detailed {
            node.inputs
                .iter()
                .map(|input| {
                    self.job
                        .expressions()
                        .value_type(*input)
                        .map(|value_type| format!("{value_type:?}"))
                        .unwrap_or_else(|_| "<invalid>".to_owned())
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        info!(
            target: "mxx_correctness::operational_noise",
            stage = "compile_compact_root",
            event = "rejected_virtual_matrix",
            expression = ?expression,
            operator = compact_matrix_operator_shape(&node.operator),
            input_count = node.inputs.len(),
            output_type = ?output,
            input_types = ?input_types,
            "compact family matrix node rejected: type or arity mismatch"
        );
    }

    /// Emit one bounded diagnostic for a generated-body concrete operator which is outside the
    /// compact vocabulary. This is deliberately INFO-gated and inspects at most two direct
    /// children; the compiler's eligibility decision remains unchanged.
    fn log_compact_concrete_rejection(
        &self,
        root: ExprId,
        expression: ExprId,
        node: &super::arena::ExprNode,
        output: &ResolvedValueType,
        owner: Option<ValueProgramId>,
        parent: Option<(ExprId, bool)>,
    ) {
        if !tracing::enabled!(
            target: "mxx_correctness::operational_noise",
            tracing::Level::INFO
        ) {
            return;
        }
        let (parent_expression, parent_is_right, parent_operator) =
            parent.map_or((None, None, None), |(parent, is_right)| {
                (
                    Some(parent),
                    Some(is_right),
                    self.job
                        .expressions()
                        .node(parent)
                        .ok()
                        .map(|node| compact_concrete_operator_shape(&node.operator)),
                )
            });
        info!(
            target: "mxx_correctness::operational_noise",
            stage = "compile_compact_root",
            event = "rejected_concrete_operator",
            root = ?root,
            expression = ?expression,
            owner = ?owner,
            operator = compact_concrete_operator_shape(&node.operator),
            input_count = node.inputs.len(),
            output_type = ?output,
            parent_expression = ?parent_expression,
            parent_is_right = ?parent_is_right,
            parent_operator,
            "generated body concrete operator is outside compact vocabulary"
        );
        for (child_index, child) in node.inputs.iter().copied().take(2).enumerate() {
            let child_node = self.job.expressions().node(child).ok();
            let child_type = self.job.expressions().value_type(child).ok();
            info!(
                target: "mxx_correctness::operational_noise",
                stage = "compile_compact_root",
                event = "rejected_concrete_operator_child",
                expression = ?expression,
                child_index,
                child_expression = ?child,
                child_operator = child_node
                    .map(|node| compact_concrete_operator_shape(&node.operator)),
                child_input_count = child_node.map_or(0, |node| node.inputs.len()),
                child_type = ?child_type,
                "direct child of rejected compact concrete operator"
            );
        }
    }

    /// Emit a bounded summary when a scalar ProgramCall reaches the exact IndexedSlice gate but
    /// its authoritative family body has another operator. This is diagnostic-only: it records
    /// the body root and at most two direct children without materializing or walking the body.
    fn log_compact_indexed_scalar_body_rejection(
        &self,
        call: ExprId,
        program: ValueProgramId,
        family: FamilyValueId,
        owner: ValueProgramId,
        body: ExprId,
        body_node: &super::arena::ExprNode,
        consumer: ExprId,
        scalar_is_right: bool,
    ) {
        if !tracing::enabled!(
            target: "mxx_correctness::operational_noise",
            tracing::Level::INFO
        ) {
            return;
        }
        let consumer_operator = self
            .job
            .expressions()
            .node(consumer)
            .ok()
            .map(|node| compact_matrix_operator_shape(&node.operator));
        info!(
            target: "mxx_correctness::operational_noise",
            stage = "compile_compact_root",
            event = "rejected_indexed_scalar_body",
            call = ?call,
            program = ?program,
            family = ?family,
            owner = ?owner,
            body = ?body,
            body_operator = compact_matrix_operator_shape(&body_node.operator),
            body_input_count = body_node.inputs.len(),
            body_type = ?self.job.expressions().value_type(body).ok(),
            body_closed = ?self.job.expressions().is_closed(body).ok(),
            formal_argument_zero = ?self.compact_body_formal_argument_zero(body),
            consumer = ?consumer,
            consumer_operator,
            scalar_is_right,
            "indexed scalar family body rejected by the exact IndexedSlice gate"
        );
        for (child_index, child) in body_node.inputs.iter().copied().take(2).enumerate() {
            let child_node = self.job.expressions().node(child).ok();
            info!(
                target: "mxx_correctness::operational_noise",
                stage = "compile_compact_root",
                event = "rejected_indexed_scalar_body_child",
                body = ?body,
                child_index,
                child = ?child,
                child_operator = child_node
                    .map(|node| compact_matrix_operator_shape(&node.operator)),
                child_input_count = child_node.map_or(0, |node| node.inputs.len()),
                child_type = ?self.job.expressions().value_type(child).ok(),
                "direct child of rejected indexed scalar family body"
            );
        }
    }

    fn compact_scalar_leaf_diagnostic(&self, expression: ExprId) -> (bool, &'static str) {
        let Ok(ResolvedValueType::Matrix(matrix)) = self.job.expressions().value_type(expression)
        else {
            return (false, "type mismatch");
        };
        if matrix.rows != 1 || matrix.columns != 1 {
            return (false, "type mismatch");
        }
        let Ok(node) = self.job.expressions().node(expression) else {
            return (false, "type mismatch");
        };
        if let ValueOperator::ProgramCall { program } = node.operator {
            self.log_compact_scalar_program_call_diagnostic(expression, program);
            return (false, "not concrete leaf");
        }
        if !node.inputs.is_empty() {
            return (false, "inputs");
        }
        if !matches!(
            node.operator,
            ValueOperator::Constant(_) |
                ValueOperator::Sampler { .. } |
                ValueOperator::Source(_) |
                ValueOperator::Sample { .. } |
                ValueOperator::ExplicitElement { .. }
        ) {
            return (false, "not concrete leaf");
        }
        if !self.job.expressions().is_closed(expression).unwrap_or(false) {
            return (false, "not concrete leaf");
        }
        let Ok(bound) = self.job.facts().coefficient_bound(expression) else {
            return (false, "missing or Large bound");
        };
        if !matches!(bound, NumericContract::Known(CoefficientBound::Finite(_))) {
            return (false, "missing or Large bound");
        }
        (true, "none")
    }

    /// Emit a bounded authority/body summary for a reducible scalar ProgramCall which cannot be
    /// admitted as a closed concrete scalar leaf.  This is diagnostic-only and deliberately
    /// avoids expanding the family body: at most 64 body nodes are inspected and queued to
    /// identify a formal `Argument(0)` occurrence, with `None` reporting that the cap was reached.
    fn log_compact_scalar_program_call_diagnostic(
        &self,
        expression: ExprId,
        program: ValueProgramId,
    ) {
        if !tracing::enabled!(
            target: "mxx_correctness::operational_noise",
            tracing::Level::INFO
        ) {
            return;
        }
        let signature = self.job.programs().program_signature(program).ok();
        let family = self.job.programs().family_for_program(program);
        let family_reducible =
            family.and_then(|family| self.job.programs().family_is_reducible(family).ok());
        let body = family.and_then(|family| self.job.programs().family_body(family).ok());
        let body_node = body.and_then(|body| self.job.expressions().node(body).ok());
        let family_bound_category = family.and_then(|family| {
            self.job
                .programs()
                .family_matrix_facts(family)
                .ok()
                .flatten()
                .map(|facts| compact_bound_category(&facts.coefficient_bound))
        });
        let body_bound_category = body.and_then(|body| {
            self.job.facts().coefficient_bound(body).ok().map(compact_bound_category)
        });
        let authoritative_bound_category =
            family_bound_category.or(body_bound_category).unwrap_or("missing");
        let formal_argument_zero =
            body.map(|body| self.compact_body_formal_argument_zero(body)).flatten();
        info!(
            target: "mxx_correctness::operational_noise",
            stage = "compile_compact_root",
            event = "rejected_scalar_program_call",
            expression = ?expression,
            program = ?program,
            program_authority_present = signature.is_some(),
            family_authority_present = family.is_some(),
            reducible = ?family_reducible,
            signature_input_count = signature.as_ref().map_or(0, |signature| signature.inputs.len()),
            signature_input_type = ?signature
                .as_ref()
                .and_then(|signature| signature.inputs.first().map(|input| &input.value_type)),
            signature_input_range = ?signature
                .as_ref()
                .and_then(|signature| signature.inputs.first().and_then(|input| input.trusted_index_range)),
            signature_output = ?signature.as_ref().map(|signature| &signature.output),
            body = ?body,
            body_operator = body_node
                .as_ref()
                .map(|node| compact_binding_operator_shape(&node.operator)),
            body_input_count = body_node.as_ref().map_or(0, |node| node.inputs.len()),
            body_type = ?body.and_then(|body| self.job.expressions().value_type(body).ok()),
            body_closed = ?body.and_then(|body| self.job.expressions().is_closed(body).ok()),
            formal_argument_zero,
            authoritative_bound_category,
            "compact scalar ProgramCall rejected as non-leaf"
        );
    }

    fn compact_body_formal_argument_zero(&self, body: ExprId) -> Option<bool> {
        const MAX_NODES: usize = 64;
        let mut seen = BTreeSet::new();
        let mut work = vec![body];
        while let Some(expression) = work.pop() {
            if !seen.insert(expression) {
                continue;
            }
            if seen.len() > MAX_NODES {
                return None;
            }
            let node = self.job.expressions().node(expression).ok()?;
            if matches!(&node.operator, ValueOperator::Argument { position: 0, .. }) {
                return Some(true);
            }
            if work.len().saturating_add(seen.len()).saturating_add(node.inputs.len()) > MAX_NODES {
                return None;
            }
            work.extend(node.inputs.iter().copied());
        }
        Some(false)
    }

    fn materialize_root_value_with_reason(
        &mut self,
        value: Value,
        reason: BetaReason,
    ) -> Result<Value, ProductionAdapterError> {
        let Value::Expr(expression) = value else { return Ok(value) };
        let materialized =
            self.job.materialize_reducible_generated_calls_with_reason(expression, reason)?;
        self.job
            .programs()
            .ensure_no_reducible_generated_calls(self.job.expressions(), materialized)?;
        Ok(Value::Expr(materialized))
    }

    fn emit_beta_reason_diagnostics(&self, stage: &'static str) {
        let counters = self.job.programs().diagnostic_counters();
        for reason in BetaReason::ALL {
            let index = reason as usize;
            let misses = counters.beta_reason_misses[index];
            let visits = counters.beta_reason_visits[index];
            let expr_allocations = counters.beta_reason_expr_allocations[index];
            if misses != 0 || visits != 0 || expr_allocations != 0 {
                info!(
                    target: "mxx_correctness::operational_noise",
                    stage,
                    event = "beta_reason",
                    reason = reason.label(),
                    misses,
                    visits,
                    expr_allocations,
                    "beta attribution diagnostics"
                );
            }
        }
    }

    fn close_root(
        &mut self,
        value: Value,
        wire: &PlannedWire,
        operation: &str,
        compact: bool,
    ) -> Result<ProductionRoot, ProductionAdapterError> {
        Ok(match value {
            Value::Family(family) if compact => ProductionRoot::CompactFamily(family),
            Value::Family(family) => ProductionRoot::Family(family),
            Value::Expr(expression) => {
                let closed = self.close_expression(wire, expression, operation)?;
                if compact {
                    ProductionRoot::Compact(closed)
                } else {
                    ProductionRoot::Closed(closed)
                }
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
    /// body's root expression. Only direct candidates from the current occurrence are reachable
    /// through compact body edges; candidates already lifted by an inner occurrence keep their
    /// existing family operands and are intentionally skipped.
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
        let candidate_indices =
            self.relation_candidate_indices.get(occurrence).cloned().unwrap_or_default();
        self.resolver_progress.record_relation_bucket(!candidate_indices.is_empty());
        if candidate_indices.is_empty() {
            return Ok((body, Vec::new()));
        }
        // This is an invocation-local read-only snapshot.  If even its diagnostic traversal
        // encounters an arena error, discard that error and retain the legacy candidate-ordered
        // reachability calls below so the authoritative first error remains unchanged.
        let reachable_snapshot = self.expression_reachable_set(body).ok();
        let mut candidates = Vec::new();
        for index in candidate_indices {
            let candidate = self.relation_candidates.get(index).ok_or_else(|| {
                ProductionAdapterError::Structural {
                    wire: wire.clone(),
                    reason: "relation candidate occurrence index is invalid".to_owned(),
                }
            })?;
            if candidate.family_operands.is_none() &&
                &candidate.wire.occurrence == occurrence &&
                self.expression_reaches_with_snapshot(
                    body,
                    candidate.preimage,
                    reachable_snapshot.as_ref(),
                )?
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
                let reachable = self.expression_reaches_with_snapshot(
                    body,
                    operand,
                    reachable_snapshot.as_ref(),
                )?;
                if must_rewrite && operand == preimage && !reachable {
                    return Err(ProductionAdapterError::Structural {
                        wire: self.relation_candidates[index].wire.clone(),
                        reason: "reached preimage relation output is not present in parallel body"
                            .to_owned(),
                    });
                }
                let family = match self.family_for_expression(operand, domain, argument, wire)? {
                    Some(family) => family,
                    None if opaque_fallback => self.opaque_generated_family(domain, operand)?,
                    None => self.generated_family(domain, operand)?,
                };
                if reachable {
                    let call = self.call_family_in_program_scope_deferred_generated(
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
        drop(reachable_snapshot);
        Ok((self.rewrite_expression_exact(body, &replacements)?, lifted))
    }

    fn push_relation_candidate(&mut self, candidate: RelationCandidate) {
        let pending_occurrence =
            candidate.family_operands.is_none().then(|| candidate.wire.occurrence.clone());
        let index = self.relation_candidates.len();
        self.relation_candidates.push(candidate);
        if let Some(occurrence) = pending_occurrence {
            self.relation_candidate_indices.entry(occurrence).or_default().push(index);
        }
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

    fn expression_reachable_set(
        &self,
        root: ExprId,
    ) -> Result<BTreeSet<ExprId>, super::arena::ArenaError> {
        let mut reachable = BTreeSet::new();
        let mut work = vec![root];
        while let Some(expression) = work.pop() {
            if !reachable.insert(expression) {
                continue;
            }
            work.extend(self.job.expressions().node(expression)?.inputs.iter().copied());
        }
        Ok(reachable)
    }

    fn expression_reaches_with_snapshot(
        &self,
        root: ExprId,
        target: ExprId,
        reachable: Option<&BTreeSet<ExprId>>,
    ) -> Result<bool, ProductionAdapterError> {
        match reachable {
            Some(reachable) => Ok(reachable.contains(&target)),
            None => self.expression_reaches(root, target),
        }
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
        let fact_owner = self.gadget_fact_owner(input)?;
        if small &&
            !fact_owner.is_some_and(|owner| {
                matches!(
                    self.job.facts().coefficient_bound(owner),
                    Ok(NumericContract::Known(CoefficientBound::ExactZero))
                )
            })
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
        let layout = |expression: Option<ExprId>| match expression
            .and_then(|expression| self.job.facts().facts(expression).ok())
        {
            Some(super::facts::ValueFacts::Matrix(facts)) => Some(facts.metadata.layout.clone()),
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
                decomposition_layout: layout(Some(decomposition)),
                input_layout: layout(fact_owner),
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
        self.relation_candidate_indices.clear();
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
            let preimage_root = self.job.materialize_reducible_generated_calls_with_reason(
                self.job.programs().family_body(preimage_family)?,
                BetaReason::RelationLiftEndpoint,
            )?;
            let public_root = self.job.materialize_reducible_generated_calls_with_reason(
                self.job.programs().family_body(public_family)?,
                BetaReason::RelationLiftEndpoint,
            )?;
            let trapdoor_root = self.job.materialize_reducible_generated_calls_with_reason(
                self.job.programs().family_body(trapdoor_family)?,
                BetaReason::RelationLiftEndpoint,
            )?;
            let target_root = self.job.materialize_reducible_generated_calls_with_reason(
                self.job.programs().family_body(target_family)?,
                BetaReason::RelationLiftEndpoint,
            )?;
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
                debug!(
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

    fn intern_index_constant(&mut self, value: BigInt) -> Result<ExprId, ProductionAdapterError> {
        Ok(self
            .job
            .expressions_mut()
            .intern(ValueOperator::Constant(TypedConstant::int(value)), Box::new([]))?)
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
        let state_wire = self.full_wire_arc(state.wire)?;
        if state.carried.len() != state.spec.carried_count ||
            state.child_inputs.len() != state.carried.len().saturating_add(state.invariant.len())
        {
            return Err(ProductionAdapterError::Structural {
                wire: state_wire.as_ref().clone(),
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
            .get(&(state_wire.stage.clone(), state.child_occurrence.clone()))
            .cloned()
            .unwrap_or_else(|| BTreeSet::from([state.child_occurrence.clone()]));
        for occurrence in &body_occurrences {
            if let Some(slots) = self
                .wire_table
                .occurrence_slots
                .get(&(state_wire.stage.clone(), occurrence.clone()))
            {
                for id in slots.iter().copied() {
                    let slot = &mut self.values[id.slot as usize];
                    if slot.take().is_some() {
                        self.value_count -= 1;
                    }
                }
            }
        }
        self.active_loop_indices = state.saved_loop_indices.clone();
        self.active_loop_arguments = state.saved_loop_arguments.clone();
        self.active_loop_indices.insert(state.spec.index_slot, BigInt::from(state.iteration));
        state.next_outputs.clear();
        let mut iteration_overrides = BTreeMap::new();
        for (input, value) in state
            .child_inputs
            .iter()
            .copied()
            .zip(state.carried.iter().copied().chain(state.invariant.iter().copied()))
        {
            iteration_overrides.insert(input, value);
        }
        state.iteration_overrides = Rc::new(iteration_overrides);
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
                wire: output,
                overrides: state.iteration_overrides.clone(),
            });
        }
        Ok(())
    }

    fn observe_resolver_frame(&mut self, frame: &ResolveFrame, pending_frames: usize) {
        let Some(snapshot) = self.resolver_progress.observe_frame(frame) else {
            return;
        };
        let program = self.job.programs().diagnostic_counters();
        let beta_reason_snapshot = format_beta_reason_snapshot(&program);
        info!(
            target: "mxx_correctness::operational_noise",
            stage = "resolve",
            event = "progress",
            total_frames = snapshot.total_frames,
            pending_frames,
            cache_hits = snapshot.cache_hits,
            ordinary_lowers = snapshot.ordinary_lowers,
            parallel_prepare = snapshot.parallel_prepare,
            parallel_body = snapshot.parallel_body,
            parallel_finish = snapshot.parallel_finish,
            values = self.value_count,
            expression_nodes = self.job.expressions().node_count(),
            programs = self.job.programs().len(),
            generated_family_calls = self.generated_family_calls,
            family_calls = program.family_calls,
            beta_program_call_sidecar_hits = program.beta_program_call_sidecar_hits,
            beta_program_call_sidecar_misses = program.beta_program_call_sidecar_misses,
            beta_program_call_sidecar_entries = program.beta_program_call_sidecar_entries,
            beta_program_call_early_source_hits = program.beta_program_call_early_source_hits,
            beta_program_call_early_source_input_skips =
                program.beta_program_call_early_source_input_skips,
            beta_identity_shortcuts = program.beta_identity_shortcuts,
            beta_nodes_visited = program.beta_nodes_visited,
            beta_closed_subtrees_reused = program.beta_closed_subtrees_reused,
            beta_reason_snapshot = %beta_reason_snapshot,
            root_validation_facts_hits = program.root_validation_facts_hits,
            root_validation_facts_misses = program.root_validation_facts_misses,
            root_validation_facts_entries = program.root_validation_facts_entries,
            root_validation_nodes_visited = program.root_validation_nodes_visited,
            root_validation_cached_subroot_skips = program.root_validation_cached_subroot_skips,
            matrix_fact_projection_direct_hits = self.matrix_fact_projection_direct_hits,
            matrix_fact_projection_closed_root_hits =
                self.matrix_fact_projection_closed_root_hits,
            matrix_fact_projection_argument_hits = self.matrix_fact_projection_argument_hits,
            matrix_fact_projection_sidecar_hits = self.matrix_fact_projection_sidecar_hits,
            matrix_fact_projection_proven_absent = self.matrix_fact_projection_proven_absent,
            matrix_fact_projection_fallbacks = self.matrix_fact_projection_fallbacks,
            matrix_fact_projection_fallback_beta_nodes =
                self.matrix_fact_projection_fallback_beta_nodes,
            matrix_select_open_observation_skips = self.matrix_select_open_observation_skips,
            matrix_select_open_observation_skipped_branches =
                self.matrix_select_open_observation_skipped_branches,
            selector_range_compact_direct_hits = self.selector_range_compact_direct_hits,
            selector_range_projected_program_call_hits =
                self.selector_range_projected_program_call_hits,
            selector_range_fallback_materializations = self.selector_range_fallback_materializations,
            selector_range_projector_nodes = self.selector_range_projector_nodes,
            selector_range_projector_program_calls = self.selector_range_projector_program_calls,
            relation_empty_buckets = snapshot.relation_empty_buckets,
            relation_nonempty_buckets = snapshot.relation_nonempty_buckets,
            elapsed_ms = self
                .resolver_progress
                .elapsed()
                .map(|elapsed| elapsed.as_millis())
                .unwrap_or_default(),
            "operational noise resolver progress"
        );
    }

    fn resolve(
        &mut self,
        wire: PlannedWire,
        overrides: OverrideEnv,
    ) -> Result<Value, ProductionAdapterError> {
        let wire = self.compact_wire(&wire)?;
        let mut frames = vec![ResolveFrame::Resolve { wire, overrides }];
        let mut result = None;
        while let Some(frame) = frames.pop() {
            self.observe_resolver_frame(&frame, frames.len());
            match frame {
                ResolveFrame::Resolve { wire, overrides } => {
                    let mut scheduled = false;
                    if let Some(value) = self.immediate_value(wire, &overrides)? {
                        self.resolver_progress.record_cache_hit();
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
                        let full_wire = self.full_wire_arc(wire)?;
                        let node = self.wire_table.node(wire).ok_or_else(|| {
                            ProductionAdapterError::MissingWire { wire: full_wire.as_ref().clone() }
                        })?;
                        match &node.planned.kind {
                            NodeKind::SubgraphCall(_) => {
                                let child = self.outputs.get(&wire).cloned().ok_or_else(|| {
                                    ProductionAdapterError::Structural {
                                        wire: full_wire.as_ref().clone(),
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
                                let (child_inputs, child_outputs) = {
                                    let child = self.child_scope(&full_wire)?;
                                    (
                                        child.inputs().to_vec().into_boxed_slice(),
                                        child.outputs().to_vec().into_boxed_slice(),
                                    )
                                };
                                if node.arguments.len() != child_inputs.len() ||
                                    node.arguments.len() != spec.input_modes.len()
                                {
                                    return Err(ProductionAdapterError::Structural {
                                        wire: full_wire.as_ref().clone(),
                                        reason: format!(
                                            "parallel input arity mismatch: parent={}, child={}, modes={}",
                                            node.arguments.len(),
                                            child_inputs.len(),
                                            spec.input_modes.len()
                                        ),
                                    });
                                }
                                let domain = FamilyDomain::new(0, self.eval_u64(&spec.count)?)?;
                                let argument = self
                                    .job
                                    .expressions_mut()
                                    .intern_argument(0, ResolvedValueType::Int)?;
                                let child_occurrence = self.child_occurrence(&full_wire)?;
                                let child_inputs = child_inputs
                                    .iter()
                                    .copied()
                                    .map(|input| {
                                        self.compact_child_wire(
                                            &full_wire.stage,
                                            &child_occurrence,
                                            input,
                                        )
                                    })
                                    .collect::<Result<Vec<_>, _>>()?
                                    .into_boxed_slice();
                                let child_outputs = child_outputs
                                    .iter()
                                    .copied()
                                    .map(|output| {
                                        self.compact_child_wire(
                                            &full_wire.stage,
                                            &child_occurrence,
                                            output,
                                        )
                                    })
                                    .collect::<Result<Vec<_>, _>>()?
                                    .into_boxed_slice();
                                frames.push(ResolveFrame::ParallelPrepare {
                                    state: ParallelState {
                                        wire,
                                        spec: spec.clone(),
                                        overrides,
                                        planned_node: Arc::clone(&node),
                                        domain,
                                        argument,
                                        child_inputs,
                                        child_outputs,
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
                                let child_occurrence =
                                    self.plan.child_occurrence(&full_wire).cloned().ok_or_else(
                                        || ProductionAdapterError::Structural {
                                            wire: full_wire.as_ref().clone(),
                                            reason: "missing planned child occurrence".to_owned(),
                                        },
                                    )?;
                                frames.push(ResolveFrame::SequentialPrepare {
                                    state: SequentialState {
                                        wire: wire.clone(),
                                        spec: spec.clone(),
                                        overrides,
                                        planned_node: Arc::clone(&node),
                                        child_inputs: Box::new([]),
                                        child_outputs: Box::new([]),
                                        child_occurrence,
                                        carried: Vec::new(),
                                        invariant: Vec::new(),
                                        next_outputs: Vec::new(),
                                        iteration_overrides: Rc::new(BTreeMap::new()),
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
                            _ => {
                                frames.push(ResolveFrame::Lower {
                                    wire,
                                    planned_node: node,
                                    overrides,
                                    next: 0,
                                    inputs: Vec::new(),
                                });
                            }
                        }
                    }
                }
                ResolveFrame::Lower { wire, planned_node, overrides, next, inputs } => {
                    if next < planned_node.arguments.len() {
                        let argument = planned_node.arguments[next];
                        frames.push(ResolveFrame::Lower {
                            wire: wire.clone(),
                            planned_node,
                            overrides: overrides.clone(),
                            next: next + 1,
                            inputs,
                        });
                        frames.push(ResolveFrame::Resolve { wire: argument, overrides });
                    } else {
                        let full_wire = self.full_wire_arc(wire)?;
                        self.resolver_progress.record_ordinary_lower();
                        let value = self.lower_node(
                            &full_wire,
                            &planned_node.planned.kind,
                            &planned_node.planned.output_type,
                            &inputs,
                        )?;
                        self.store_value(wire, value)?;
                        result = Some(value);
                    }
                }
                ResolveFrame::Store { wire } => {
                    let full_wire = self.full_wire_arc(wire)?;
                    let value = result.ok_or_else(|| ProductionAdapterError::Structural {
                        wire: full_wire.as_ref().clone(),
                        reason: "worklist completed without a child value".to_owned(),
                    })?;
                    self.store_value(wire, value)?;
                    result = Some(value);
                }
                ResolveFrame::ParallelPrepare { mut state } => {
                    if state.next_input < state.child_inputs.len() {
                        let state_wire = self.full_wire_arc(state.wire)?;
                        let position = state.next_input;
                        let parent =
                            *state.planned_node.arguments.get(position).ok_or_else(|| {
                                ProductionAdapterError::Structural {
                                    wire: state_wire.as_ref().clone(),
                                    reason: "parallel parent input arity mismatch".to_owned(),
                                }
                            })?;
                        state.next_input = position + 1;
                        let parent_overrides = state.overrides.clone();
                        frames.push(ResolveFrame::ParallelInput { state, position });
                        frames.push(ResolveFrame::Resolve {
                            wire: parent,
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
                        let state_wire = self.full_wire_arc(state.wire)?;
                        let output = *state
                            .child_outputs
                            .get(state_wire.wire.port.0 as usize)
                            .ok_or_else(|| ProductionAdapterError::Structural {
                                wire: state_wire.as_ref().clone(),
                                reason: "invalid parallel output".to_owned(),
                            })?;
                        let body_overrides = Rc::new(std::mem::take(&mut state.child_overrides));
                        frames.push(ResolveFrame::ParallelBody { state });
                        frames.push(ResolveFrame::Resolve {
                            wire: output,
                            overrides: body_overrides,
                        });
                    }
                }
                ResolveFrame::ParallelFinish { state, family } => {
                    let state_wire = self.full_wire_arc(state.wire)?;
                    let node = self.wire_table.node(state.wire).ok_or_else(|| {
                        ProductionAdapterError::MissingWire { wire: state_wire.as_ref().clone() }
                    })?;
                    let output_type = &node.planned.output_type;
                    let value = if matches!(output_type, WireType::IndexedFamily { .. }) {
                        Value::Family(family)
                    } else {
                        let domain = self.job.programs().family_domain(family)?;
                        Value::Expr(self.call_family_with_resolved_range(
                            family,
                            state.argument,
                            TrustedIndexRange {
                                minimum: domain.minimum,
                                maximum_exclusive: domain.maximum_exclusive,
                            },
                        )?)
                    };
                    self.store_value(state.wire, value)?;
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
                    let state_wire = self.full_wire_arc(state.wire)?;
                    let count = self.eval_u64(&state.spec.count)?;
                    state.count = usize::try_from(count).map_err(|_| {
                        ProductionAdapterError::IntegerExpression {
                            expression: state.spec.count.clone(),
                            reason: "sequential count does not fit usize".to_owned(),
                        }
                    })?;
                    if state.planned_node.arguments.len() < state.spec.carried_count {
                        return Err(ProductionAdapterError::Structural {
                            wire: state_wire.as_ref().clone(),
                            reason: "carried schema exceeds input arity".to_owned(),
                        });
                    }
                    let (child_inputs, child_outputs) = {
                        let child = self.child_scope(&state_wire)?;
                        (
                            child.inputs().to_vec().into_boxed_slice(),
                            child.outputs().to_vec().into_boxed_slice(),
                        )
                    };
                    if child_outputs.len() != state.spec.carried_count ||
                        child_inputs.len() != state.planned_node.arguments.len()
                    {
                        return Err(ProductionAdapterError::Structural {
                            wire: state_wire.as_ref().clone(),
                            reason: format!(
                                "sequential carried schema mismatch: parent={}, child inputs={}, child outputs={}, carried={}",
                                state.planned_node.arguments.len(),
                                child_inputs.len(),
                                child_outputs.len(),
                                state.spec.carried_count
                            ),
                        });
                    }
                    state.child_inputs = child_inputs
                        .iter()
                        .copied()
                        .map(|input| {
                            self.compact_child_wire(
                                &state_wire.stage,
                                &state.child_occurrence,
                                input,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?
                        .into_boxed_slice();
                    state.child_outputs = child_outputs
                        .iter()
                        .copied()
                        .map(|output| {
                            self.compact_child_wire(
                                &state_wire.stage,
                                &state.child_occurrence,
                                output,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?
                        .into_boxed_slice();
                    if state.spec.carried_count == 0 {
                        frames.push(ResolveFrame::SequentialInvariant { state, position: 0 });
                    } else {
                        let argument = state.planned_node.arguments[0];
                        frames.push(ResolveFrame::SequentialInit { state, position: 0 });
                        let state = match frames.last() {
                            Some(ResolveFrame::SequentialInit { state, .. }) => state,
                            _ => unreachable!(),
                        };
                        frames.push(ResolveFrame::Resolve {
                            wire: argument,
                            overrides: state.overrides.clone(),
                        });
                    }
                }
                ResolveFrame::SequentialInit { state, position } => {
                    if position >= state.spec.carried_count {
                        return Err(ProductionAdapterError::Structural {
                            wire: self.full_wire(state.wire)?.clone(),
                            reason: "sequential carried initializer is out of range".to_owned(),
                        });
                    }
                    let argument = state.planned_node.arguments[position];
                    frames.push(ResolveFrame::SequentialInit { state, position });
                    let state = match frames.last() {
                        Some(ResolveFrame::SequentialInit { state, .. }) => state,
                        _ => unreachable!(),
                    };
                    frames.push(ResolveFrame::Resolve {
                        wire: argument,
                        overrides: state.overrides.clone(),
                    });
                }
                ResolveFrame::SequentialInvariant { state, position } => {
                    let invariant_count =
                        state.planned_node.arguments.len() - state.spec.carried_count;
                    if position < invariant_count {
                        let argument =
                            state.planned_node.arguments[state.spec.carried_count + position];
                        frames.push(ResolveFrame::SequentialInvariant { state, position });
                        let state = match frames.last() {
                            Some(ResolveFrame::SequentialInvariant { state, .. }) => state,
                            _ => unreachable!(),
                        };
                        frames.push(ResolveFrame::Resolve {
                            wire: argument,
                            overrides: state.overrides.clone(),
                        });
                    } else if state.count == 0 {
                        frames.push(ResolveFrame::SequentialFinish { state });
                    } else {
                        self.schedule_sequential_iteration(state, &mut frames)?;
                    }
                }
                ResolveFrame::SequentialIterationOutput { state, next_output } => {
                    let state_wire = self.full_wire_arc(state.wire)?;
                    let output = *state.child_outputs.get(next_output).ok_or_else(|| {
                        ProductionAdapterError::Structural {
                            wire: state_wire.as_ref().clone(),
                            reason: "sequential output is out of range".to_owned(),
                        }
                    })?;
                    let overrides = state.iteration_overrides.clone();
                    frames.push(ResolveFrame::SequentialIterationOutput { state, next_output });
                    frames.push(ResolveFrame::Resolve { wire: output, overrides });
                }
                ResolveFrame::SequentialCommit { mut state, next_state } => {
                    if next_state.len() != state.spec.carried_count {
                        return Err(ProductionAdapterError::Structural {
                            wire: self.full_wire(state.wire)?.clone(),
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
                    let state_wire = self.full_wire_arc(state.wire)?;
                    let port = state_wire.wire.port.0 as usize;
                    let value = *state.carried.get(port).ok_or_else(|| {
                        ProductionAdapterError::Structural {
                            wire: state_wire.as_ref().clone(),
                            reason: format!(
                                "invalid sequential output port {port} for {} carried outputs",
                                state.carried.len()
                            ),
                        }
                    })?;
                    self.store_value(state.wire, value)?;
                    result = Some(value);
                }
            }
            while let Some(value) = result.take() {
                let Some(parent) = frames.pop() else {
                    result = Some(value);
                    break;
                };
                self.observe_resolver_frame(&parent, frames.len());
                match parent {
                    ResolveFrame::Lower { wire, planned_node, overrides, next, mut inputs } => {
                        inputs.push(value);
                        frames.push(ResolveFrame::Lower {
                            wire,
                            planned_node,
                            overrides,
                            next,
                            inputs,
                        });
                    }
                    ResolveFrame::Store { wire } => {
                        self.store_value(wire, value)?;
                        result = Some(value);
                    }
                    ResolveFrame::ParallelInput { mut state, position } => {
                        let state_wire = self.full_wire_arc(state.wire)?;
                        let mode =
                            state.spec.input_modes.get(position).copied().ok_or_else(|| {
                                ProductionAdapterError::Structural {
                                    wire: state_wire.as_ref().clone(),
                                    reason: "parallel input mode is missing".to_owned(),
                                }
                            })?;
                        let mapped = match (mode, value) {
                            // Zip consumes one element from a family at the active loop index.
                            // A scalar here is a malformed Graph IR binding and must not be
                            // silently treated as a broadcast; doing so would hide an incorrect
                            // input mode and could make the generated program unsound.
                            (LoopInputMode::Zip, Value::Family(family)) => {
                                let range = TrustedIndexRange {
                                    minimum: state.domain.minimum,
                                    maximum_exclusive: state.domain.maximum_exclusive,
                                };
                                let expression = self
                                    .call_family_in_program_scope_deferred_generated(
                                        family,
                                        state.argument,
                                        range,
                                    )?;
                                Value::Expr(expression)
                            }
                            (LoopInputMode::Zip, Value::Expr(_)) => {
                                return Err(ProductionAdapterError::Structural {
                                    wire: state_wire.as_ref().clone(),
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
                                        wire: state_wire.as_ref().clone(),
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
                                            wire: state_wire.as_ref().clone(),
                                        },
                                    )?;
                                let range =
                                    TrustedIndexRange { minimum: offset, maximum_exclusive };
                                let expression = self
                                    .call_family_in_program_scope_deferred_generated(
                                        family, mapped, range,
                                    )?;
                                Value::Expr(expression)
                            }
                            (LoopInputMode::ZipOffset { .. }, Value::Expr(_)) => {
                                return Err(ProductionAdapterError::Structural {
                                    wire: state_wire.as_ref().clone(),
                                    reason: "parallel ZipOffset input is not a family".to_owned(),
                                });
                            }
                        };
                        let input = *state.child_inputs.get(position).ok_or_else(|| {
                            ProductionAdapterError::Structural {
                                wire: state_wire.as_ref().clone(),
                                reason: "parallel child input is missing".to_owned(),
                            }
                        })?;
                        state.child_overrides.insert(input, mapped);
                        frames.push(ResolveFrame::ParallelPrepare { state });
                    }
                    ResolveFrame::ParallelBody { state } => {
                        let state_wire = self.full_wire_arc(state.wire)?;
                        let Value::Expr(body) = value else {
                            return Err(ProductionAdapterError::Structural {
                                wire: state_wire.as_ref().clone(),
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
                            &state_wire,
                        )?;
                        let family = self.generated_family(state.domain, body)?;
                        for (index, preimage, public, trapdoor, target) in lifted_operands {
                            if self.diagnostic_budget > 0 {
                                self.diagnostic_budget -= 1;
                                debug!(
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
                                wire: self.full_wire(state.wire)?.clone(),
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
                                wire: self.full_wire(state.wire)?.clone(),
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
                                wire: self.full_wire(state.wire)?.clone(),
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
                    Value::Family(family)
                } else {
                    let source = self.source_identity(wire, name, wire_type)?;
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
            NodeKind::ConstantInt(value) => Value::Expr(self.job.expressions_mut().intern(
                ValueOperator::Constant(TypedConstant::int(value.clone())),
                Box::new([]),
            )?),
            NodeKind::ConstantReal(value) => Value::Expr(self.job.expressions_mut().intern(
                ValueOperator::Constant(TypedConstant::real(real_descriptor(value)?)),
                Box::new([]),
            )?),
            NodeKind::ConstantBool(value) => Value::Expr(
                self.job
                    .expressions_mut()
                    .intern(ValueOperator::Constant(TypedConstant::bool(*value)), Box::new([]))?,
            ),
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
            NodeKind::MatrixMulAccumulate { coefficients, has_bias } => {
                let values = inputs
                    .iter()
                    .map(|value| match value {
                        Value::Expr(id) => Ok(*id),
                        Value::Family(_) => Err(ProductionAdapterError::UnsupportedNode {
                            kind: "family multi-row GEMM input".to_owned(),
                            wire: wire.clone(),
                        }),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let expected_output = self.resolved_type(output, wire)?;
                let mut terms = Vec::with_capacity(coefficients.len());
                for (product, coefficient) in coefficients.iter().enumerate() {
                    let multiplied = self.intern_node_operator_with_expected_output(
                        wire,
                        &expected_output,
                        ValueOperator::Matrix(MatrixOperation::Multiply),
                        vec![values[2 * product], values[2 * product + 1]].into_boxed_slice(),
                        true,
                    )?;
                    let coefficient = self.intern_int_expression(coefficient, wire)?;
                    terms.push(self.intern_node_operator_with_expected_output(
                        wire,
                        &expected_output,
                        ValueOperator::Matrix(MatrixOperation::Scale),
                        vec![multiplied, coefficient].into_boxed_slice(),
                        true,
                    )?);
                }
                let mut accumulated = terms[0];
                for term in terms.into_iter().skip(1) {
                    accumulated = self.intern_node_operator_with_expected_output(
                        wire,
                        &expected_output,
                        ValueOperator::Matrix(MatrixOperation::Add),
                        vec![accumulated, term].into_boxed_slice(),
                        true,
                    )?;
                }
                if *has_bias {
                    accumulated = self.intern_node_operator_with_expected_output(
                        wire,
                        &expected_output,
                        ValueOperator::Matrix(MatrixOperation::Add),
                        vec![accumulated, values[2 * coefficients.len()]].into_boxed_slice(),
                        true,
                    )?;
                }
                Value::Expr(accumulated)
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
                    let (operation, endpoints) = self.indexed_slice_operation(
                        output,
                        matrix,
                        rows.as_ref(),
                        columns.as_ref(),
                        wire,
                    )?;
                    let mut operation_inputs = Vec::with_capacity(5);
                    operation_inputs.push(matrix);
                    operation_inputs.extend(endpoints);
                    Value::Expr(self.intern_node_operator(
                        wire,
                        output,
                        ValueOperator::Matrix(operation),
                        operation_inputs.into_boxed_slice(),
                        true,
                    )?)
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
                    self.job.expressions().is_closed(expression).is_ok_and(|closed| closed)
                });
                let family_operands = if closed {
                    let domain = FamilyDomain::new(0, 1)?;
                    Some((
                        self.opaque_generated_family(domain, preimage)?,
                        self.generated_family(domain, public)?,
                        self.generated_family(domain, trapdoor)?,
                        self.generated_family(domain, target)?,
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
                self.push_relation_candidate(RelationCandidate {
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
                Value::Expr(self.call_family(family_id, index)?)
            }
            NodeKind::FamilyGetDynamic => {
                let family_id = family(inputs, 0)?;
                let Value::Expr(index) = inputs.get(1).copied().ok_or_else(|| {
                    ProductionAdapterError::MissingSelectorRange { wire: wire.clone() }
                })?
                else {
                    return Err(ProductionAdapterError::MissingSelectorRange { wire: wire.clone() });
                };
                Value::Expr(self.call_family_with_wire_reason(
                    family_id,
                    index,
                    wire.clone(),
                    BetaReason::ScalarFamilyGet,
                )?)
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
                    let selector = match self.binder_open_selector(selector, family_range, wire)? {
                        Some(selector) => selector,
                        None => SelectionSelector::Closed(self.close_expression(
                            wire,
                            selector,
                            "close closed-family selector",
                        )?),
                    };
                    Value::Family(self.select_family(selector, &families)?)
                } else if branches.iter().all(|value| matches!(value, Value::Expr(_))) {
                    let values = branches
                        .iter()
                        .map(|value| match value {
                            Value::Expr(id) => Ok(*id),
                            _ => unreachable!(),
                        })
                        .collect::<Result<Vec<_>, ProductionAdapterError>>()?;
                    let element_type = self.job.expressions().value_type(values[0])?.clone();
                    let fact_values = if matches!(element_type, ResolvedValueType::Matrix(_)) {
                        let selector_closed = self.job.expressions().is_closed(selector)?;
                        let mut branches_closed = true;
                        for value in &values {
                            branches_closed &= self.job.expressions().is_closed(*value)?;
                        }
                        if selector_closed && branches_closed {
                            Some(
                                values
                                    .iter()
                                    .map(|value| self.authoritative_matrix_observation_view(*value))
                                    .collect::<Result<Vec<_>, _>>()?,
                            )
                        } else {
                            self.matrix_select_open_observation_skips =
                                self.matrix_select_open_observation_skips.saturating_add(1);
                            self.matrix_select_open_observation_skipped_branches = self
                                .matrix_select_open_observation_skipped_branches
                                .saturating_add(values.len() as u64);
                            None
                        }
                    } else {
                        None
                    };
                    let mut body = Vec::with_capacity(values.len() + 1);
                    body.push(selector);
                    body.extend(values);
                    let branch_values = body[1..].to_vec();
                    let expression = self.job.expressions_mut().intern(
                        ValueOperator::ExplicitElement {
                            domain: FamilyDomain::new(0, body.len() as u64 - 1)?,
                            element_type,
                        },
                        body.into_boxed_slice(),
                    )?;
                    self.job.transfer_explicit_matrix_facts(
                        fact_values.as_deref().unwrap_or(&branch_values),
                        expression,
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
    ) -> Result<SemanticSourceIdentity, ProductionAdapterError> {
        let value_type = self.resolved_type(wire_type, wire)?;
        let protocol_input =
            self.protocol_inputs.get(&(wire.stage.clone(), StageInputName(name.to_owned())));
        Ok(SemanticSourceIdentity {
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
        })
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
        if self.active_loop_indices.is_empty() {
            return expression.evaluate(&self.params).map_err(|source| {
                ProductionAdapterError::IntegerExpression {
                    expression: expression.clone(),
                    reason: source.to_string(),
                }
            });
        }
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
    ) -> Result<(MatrixOperation, [ExprId; 4]), ProductionAdapterError> {
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

        let row_span = self.validate_indexed_slice_axis(
            row_start,
            row_end,
            input_type.rows,
            output_type.rows,
            wire,
        )?;
        let column_span = self.validate_indexed_slice_axis(
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
        ))
    }

    fn validate_indexed_slice_axis(
        &self,
        start: ExprId,
        end: ExprId,
        input_extent: usize,
        output_extent: usize,
        wire: &PlannedWire,
    ) -> Result<usize, ProductionAdapterError> {
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
        for (name, expression) in [("start", start), ("end", end)] {
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
        }
        Ok(span)
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

fn compact_binding_operator_shape(operator: &ValueOperator) -> &'static str {
    match operator {
        ValueOperator::Argument { .. } => "Argument",
        ValueOperator::Constant(_) => "Constant",
        ValueOperator::Source(_) => "Source",
        ValueOperator::Sample { .. } => "Sample",
        ValueOperator::Sampler { .. } => "Sampler",
        ValueOperator::DeterministicHash(_) => "DeterministicHash",
        ValueOperator::OpaqueFamilyElement { .. } => "OpaqueFamilyElement",
        ValueOperator::IndexMap { .. } => "IndexMap",
        ValueOperator::ExplicitElement { .. } => "ExplicitElement",
        ValueOperator::ProgramCall { .. } => "ProgramCall",
        ValueOperator::Transform(_) => "Transform",
        ValueOperator::ExtractCoefficient { .. } => "ExtractCoefficient",
        ValueOperator::Scalar(_) => "Scalar",
        ValueOperator::Matrix(_) => "Matrix",
        ValueOperator::Trapdoor(_) => "Trapdoor",
    }
}

fn compact_matrix_operator_shape(operator: &ValueOperator) -> &'static str {
    match operator {
        ValueOperator::Matrix(MatrixOperation::Add) => "Matrix::Add",
        ValueOperator::Matrix(MatrixOperation::Subtract) => "Matrix::Subtract",
        ValueOperator::Matrix(MatrixOperation::Negate) => "Matrix::Negate",
        ValueOperator::Matrix(MatrixOperation::Scale) => "Matrix::Scale",
        ValueOperator::Matrix(MatrixOperation::Multiply) => "Matrix::Multiply",
        ValueOperator::Matrix(MatrixOperation::Transpose) => "Matrix::Transpose",
        ValueOperator::Matrix(MatrixOperation::Slice { .. }) => "Matrix::Slice",
        ValueOperator::Matrix(MatrixOperation::IndexedSlice { .. }) => "Matrix::IndexedSlice",
        ValueOperator::Matrix(MatrixOperation::View { .. }) => "Matrix::View",
        ValueOperator::Matrix(MatrixOperation::Concat { .. }) => "Matrix::Concat",
        ValueOperator::Matrix(MatrixOperation::Tensor { .. }) => "Matrix::Tensor",
        ValueOperator::Matrix(MatrixOperation::CrtRecompose { .. }) => "Matrix::CrtRecompose",
        ValueOperator::Matrix(MatrixOperation::ExtractCoefficient { .. }) => {
            "Matrix::ExtractCoefficient"
        }
        ValueOperator::Matrix(MatrixOperation::LiftConstantPolynomial { .. }) => {
            "Matrix::LiftConstantPolynomial"
        }
        _ => compact_binding_operator_shape(operator),
    }
}

fn compact_concrete_operator_shape(operator: &ValueOperator) -> &'static str {
    match operator {
        ValueOperator::Scalar(operation) => match operation {
            ScalarOperation::Add => "Scalar::Add",
            ScalarOperation::Subtract => "Scalar::Subtract",
            ScalarOperation::Multiply => "Scalar::Multiply",
            ScalarOperation::Divide => "Scalar::Divide",
            ScalarOperation::Remainder => "Scalar::Remainder",
            ScalarOperation::Negate => "Scalar::Negate",
            ScalarOperation::Equal => "Scalar::Equal",
            ScalarOperation::Less => "Scalar::Less",
            ScalarOperation::LessEqual => "Scalar::LessEqual",
            ScalarOperation::BoolToInt => "Scalar::BoolToInt",
            ScalarOperation::IntToReal => "Scalar::IntToReal",
            ScalarOperation::RealAdd => "Scalar::RealAdd",
            ScalarOperation::RealSubtract => "Scalar::RealSubtract",
            ScalarOperation::RealMultiply => "Scalar::RealMultiply",
            ScalarOperation::RealDivide => "Scalar::RealDivide",
            ScalarOperation::RealSqrt => "Scalar::RealSqrt",
            ScalarOperation::ThresholdDecode { .. } => "Scalar::ThresholdDecode",
            ScalarOperation::Bit { .. } => "Scalar::Bit",
            ScalarOperation::Slice { .. } => "Scalar::Slice",
            ScalarOperation::Hash { .. } => "Scalar::Hash",
            ScalarOperation::ExtractCoefficient { .. } => "Scalar::ExtractCoefficient",
            ScalarOperation::LiftConstantPolynomial { .. } => "Scalar::LiftConstantPolynomial",
        },
        ValueOperator::Matrix(_) => compact_matrix_operator_shape(operator),
        _ => compact_binding_operator_shape(operator),
    }
}

fn compact_bound_category(bound: &NumericContract<CoefficientBound>) -> &'static str {
    match bound {
        NumericContract::Known(CoefficientBound::Finite(_)) => "finite",
        NumericContract::Known(CoefficientBound::Large) => "large",
        NumericContract::Known(CoefficientBound::ExactZero) => "finite",
        NumericContract::Missing => "missing",
    }
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
fn real_descriptor(value: &RealExpr) -> Result<String, ProductionAdapterError> {
    serde_json::to_string(value).map_err(|error| ProductionAdapterError::Descriptor {
        reason: format!("real descriptor serialization failed: {error}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stable_program_fingerprint(job: &CheckerJob, program: ValueProgramId) -> String {
        let record = job.programs().program(program).expect("program fingerprint");
        let family = job.programs().family_for_program(program).map(|family| {
            (
                job.programs().family_domain(family).expect("family domain fingerprint"),
                job.programs().family_element_type(family).expect("family type fingerprint"),
                job.programs().family_is_reducible(family).expect("family mode fingerprint"),
                job.programs()
                    .family_artifact(family)
                    .expect("family artifact fingerprint")
                    .cloned(),
            )
        });
        format!(
            "signature={:?};family={:?};body={}",
            record.signature,
            family,
            stable_expression_fingerprint(job, record.root),
        )
    }

    fn stable_operator_fingerprint(job: &CheckerJob, operator: &ValueOperator) -> String {
        match operator {
            ValueOperator::ProgramCall { program } => {
                format!("ProgramCall({})", stable_program_fingerprint(job, *program))
            }
            // Index function IDs are stable semantic definition IDs, not arena handles. Keep the
            // complete operator payload while spelling the ID explicitly instead of relying on a
            // Debug representation of the enclosing expression.
            ValueOperator::IndexMap { definition, parameters } => {
                format!("IndexMap(definition={},parameters={parameters:?})", definition.0)
            }
            other => format!("{other:?}"),
        }
    }

    fn stable_expression_fingerprint(job: &CheckerJob, root: ExprId) -> String {
        let mut work = vec![(root, false)];
        let mut active = BTreeSet::new();
        let mut fingerprints = BTreeMap::<ExprId, String>::new();
        while let Some((expression, exit)) = work.pop() {
            if exit {
                active.remove(&expression);
                let node = job.expressions().node(expression).expect("expression fingerprint node");
                let children = node
                    .inputs
                    .iter()
                    .map(|child| fingerprints.get(child).expect("child fingerprint"))
                    .collect::<Vec<_>>();
                let output =
                    job.expressions().value_type(expression).expect("expression fingerprint type");
                fingerprints.insert(
                    expression,
                    format!(
                        "type={output:?};operator={};children={children:?}",
                        stable_operator_fingerprint(job, &node.operator),
                    ),
                );
                continue;
            }
            if fingerprints.contains_key(&expression) {
                continue;
            }
            if !active.insert(expression) {
                panic!("cyclic expression graph in structural NF oracle: {expression:?}");
            }
            work.push((expression, true));
            let node = job.expressions().node(expression).expect("expression fingerprint node");
            work.extend(node.inputs.iter().rev().map(|child| (*child, false)));
        }
        fingerprints.remove(&root).expect("root fingerprint")
    }

    fn stable_scoped_expression_fingerprint(
        job: &CheckerJob,
        factor: super::super::arena::ScopedExprId,
    ) -> String {
        assert!(
            job.expressions().is_closed(factor.expression()).expect("factor expression closedness"),
            "full NF parity oracle requires closed factor expressions"
        );
        // Monomial semantics are carried by the factor expression itself. The scoped owner is a
        // job-local traversal context and may differ between compact and eager realizations even
        // when the value expression is identical, so it is deliberately not part of parity.
        stable_expression_fingerprint(job, factor.expression())
    }

    fn full_nf_descriptor_map(
        job: &CheckerJob,
        value: &super::super::normal_form::AnalyzedValue,
    ) -> (
        BTreeMap<(Vec<String>, Vec<String>), BigInt>,
        super::super::normal_form::BoundedSummary,
        NumericContract<CoefficientBound>,
    ) {
        let normal_form = value.exact_nf.as_ref().expect("exact normal form");
        let scope = value.semantic.program();
        let monomials = job.monomials().get(scope).expect("normal-form monomial scope");
        let mut terms = BTreeMap::new();
        for (monomial, coefficient) in &normal_form.exact_terms {
            let descriptor = monomials.descriptor(*monomial).expect("monomial descriptor");
            let operators = |factors: &[super::super::arena::ScopedExprId]| {
                factors
                    .iter()
                    .map(|factor| stable_scoped_expression_fingerprint(job, *factor))
                    .collect::<Vec<_>>()
            };
            let key =
                (operators(&descriptor.central_factors), operators(&descriptor.ordered_factors));
            *terms.entry(key).or_insert_with(BigInt::zero) += coefficient;
        }
        (terms, normal_form.bounded_summary.clone(), value.coefficient_bound.clone())
    }

    #[test]
    fn beta_reason_progress_snapshot_formats_nonzero_buckets_and_reconciles_sums() {
        let mut counters = ProgramDiagnosticCounters::default();
        let index = BetaReason::MatrixFactObservation as usize;
        counters.beta_reason_misses[index] = 3;
        counters.beta_reason_visits[index] = 5;
        counters.beta_reason_expr_allocations[index] = 7;
        counters.beta_program_call_sidecar_misses = 3;
        counters.beta_nodes_visited = 5;

        let snapshot = format_beta_reason_snapshot(&counters);

        assert!(snapshot.contains("matrix-fact-observation:m=3,v=5,expr_allocations=7"));
        assert!(snapshot.ends_with("sum:m=3,v=5,expr_allocations=7"));
        assert_eq!(
            counters.beta_reason_misses.iter().sum::<u64>(),
            counters.beta_program_call_sidecar_misses
        );
        assert_eq!(counters.beta_reason_visits.iter().sum::<u64>(), counters.beta_nodes_visited);
    }

    #[test]
    fn selector_range_projector_matches_materialized_reducible_call_chain_without_allocation() {
        let protocol = crate::toy_example::protocol();
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("toy plan");
        let mut adapter = ProductionAdapter::new(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .expect("production adapter");
        adapter.job.programs_mut().enable_diagnostic_counters();
        let wire = plan.target().residual.clone();
        let domain = FamilyDomain::new(0, 4).unwrap();
        let range = TrustedIndexRange::new(0, 4).unwrap();
        let argument =
            adapter.job.expressions_mut().intern_argument(0, ResolvedValueType::Int).unwrap();
        let one = adapter.intern_index_constant(BigInt::from(1_u8)).unwrap();
        let mut family = adapter.generated_family(domain, argument).unwrap();
        for _ in 0..8 {
            let call = adapter
                .job
                .call_family_in_program_scope_deferred_generated(family, argument, range)
                .unwrap();
            let body = adapter
                .job
                .expressions_mut()
                .intern(ValueOperator::Scalar(ScalarOperation::Add), [call, one].into())
                .unwrap();
            family = adapter.generated_family(domain, body).unwrap();
        }
        let index = adapter
            .job
            .call_family_in_program_scope_deferred_generated(family, argument, range)
            .unwrap();
        adapter.active_loop_argument_ranges.insert((wire.occurrence.clone(), argument), range);
        let nodes_before = adapter.job.expressions().node_count();
        let counters_before = adapter.job.programs().diagnostic_counters();
        let (projected, stats) = adapter.project_index_range_through_reducible_calls(index, &wire);
        let nodes_after = adapter.job.expressions().node_count();
        let counters_after = adapter.job.programs().diagnostic_counters();

        let expected = TrustedIndexRange::new(8, 12).unwrap();
        assert_eq!(projected, Some(expected));
        assert!(stats.program_calls >= 8);
        assert_eq!(nodes_after, nodes_before);
        assert_eq!(counters_after.beta_nodes_visited, counters_before.beta_nodes_visited);
        assert_eq!(
            counters_after.beta_program_call_sidecar_misses,
            counters_before.beta_program_call_sidecar_misses
        );
        let materialized = adapter.job.materialize_reducible_generated_calls(index).unwrap();
        assert_eq!(adapter.derived_open_index_range(materialized, &wire).unwrap(), Some(expected));
    }

    #[test]
    fn selector_range_projector_remainder_requires_exact_rhs_and_matches_fallback() {
        let protocol = crate::toy_example::protocol();
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("toy plan");
        let mut adapter = ProductionAdapter::new(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .expect("production adapter");
        let wire = plan.target().residual.clone();
        let range = TrustedIndexRange::new(0, 3).unwrap();
        let argument =
            adapter.job.expressions_mut().intern_argument(0, ResolvedValueType::Int).unwrap();
        adapter.active_loop_argument_ranges.insert((wire.occurrence.clone(), argument), range);

        let two = adapter.intern_index_constant(BigInt::from(2_u8)).unwrap();
        let positive = adapter
            .job
            .expressions_mut()
            .intern(ValueOperator::Scalar(ScalarOperation::Remainder), [argument, two].into())
            .unwrap();
        let (projected_positive, _) =
            adapter.project_index_range_through_reducible_calls(positive, &wire);
        assert_eq!(projected_positive, Some(TrustedIndexRange::new(0, 2).unwrap()));
        assert_eq!(adapter.derived_open_index_range(positive, &wire).unwrap(), projected_positive);

        let nonconstant_rhs = adapter
            .job
            .expressions_mut()
            .intern(ValueOperator::Scalar(ScalarOperation::Add), [two, argument].into())
            .unwrap();
        let nonconstant = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Scalar(ScalarOperation::Remainder),
                [argument, nonconstant_rhs].into(),
            )
            .unwrap();
        let (projected_nonconstant, _) =
            adapter.project_index_range_through_reducible_calls(nonconstant, &wire);
        assert_eq!(projected_nonconstant, None);
        assert_eq!(adapter.derived_open_index_range(nonconstant, &wire).unwrap(), None);

        for divisor in [BigInt::from(0_u8), BigInt::from(-2_i8)] {
            let divisor = adapter.intern_index_constant(divisor).unwrap();
            let expression = adapter
                .job
                .expressions_mut()
                .intern(
                    ValueOperator::Scalar(ScalarOperation::Remainder),
                    [argument, divisor].into(),
                )
                .unwrap();
            let (projected, _) =
                adapter.project_index_range_through_reducible_calls(expression, &wire);
            assert_eq!(projected, None);
            assert_eq!(adapter.derived_open_index_range(expression, &wire).unwrap(), None);
        }
    }

    fn repeated_named_parallel_artifact_protocol() -> crate::ProtocolDecl {
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
        let artifact_b_source = ring.input_family("artifact-b-source", count.clone(), (1, 1));
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

        let protocol = repeated_named_parallel_artifact_protocol();
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

    #[test]
    fn compact_plan_wire_table_is_total_deterministic_and_authority_local() {
        let protocol = repeated_named_parallel_artifact_protocol();
        let plan =
            ProtocolPlan::build(&protocol, "named-parallel-artifact").expect("protocol plan");
        let first = PlanWireTable::build(&plan).expect("first compact wire table");
        let second = PlanWireTable::build(&plan).expect("second compact wire table");

        assert_eq!(first.wires, second.wires);
        assert!(first.wires.windows(2).all(|pair| pair[0] < pair[1]));
        for (wire, planned) in plan.nodes() {
            let first_id = first.id(wire).expect("node wire is indexed");
            let second_id = second.id(wire).expect("node wire is indexed again");
            assert_eq!(first_id.slot, second_id.slot);
            let first_node = first.node(first_id).expect("compact node");
            let second_node = second.node(second_id).expect("compact node again");
            assert_eq!(first_node.arguments.len(), planned.arguments.len());
            assert_eq!(
                first_node.arguments.iter().map(|id| id.slot).collect::<Vec<_>>(),
                second_node.arguments.iter().map(|id| id.slot).collect::<Vec<_>>()
            );
        }

        let alias_only = plan
            .aliases()
            .iter()
            .flat_map(|alias| [&alias.child, &alias.parent])
            .find(|wire| !plan.nodes().contains_key(*wire))
            .expect("structural plan has an alias-only endpoint");
        let alias_only_id = first.id(alias_only).expect("alias-only wire is indexed");
        assert!(first.node(alias_only_id).is_none());

        let first_root = first.id(&plan.target().residual).expect("first root id");
        assert!(second.wire(first_root).is_none(), "foreign table token must fail closed");
        assert!(
            first.wire(PlanWireId { token: first.token, slot: first.wires.len() as u32 }).is_none(),
            "out-of-range slot must fail closed"
        );
    }

    #[test]
    fn compact_resolve_rejects_foreign_and_out_of_range_ids_before_override_hits() {
        let protocol = repeated_named_parallel_artifact_protocol();
        let plan =
            ProtocolPlan::build(&protocol, "named-parallel-artifact").expect("protocol plan");
        let mut adapter =
            ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).expect("adapter");
        let value =
            Value::Expr(adapter.intern_index_constant(BigInt::from(0_u8)).expect("override value"));
        let foreign_table = PlanWireTable::build(&plan).expect("foreign compact table");
        let foreign = foreign_table.id(&plan.target().residual).expect("foreign root id");
        let foreign_overrides = Rc::new(BTreeMap::from([(foreign, value)]));
        assert!(matches!(
            adapter.immediate_value(foreign, &foreign_overrides),
            Err(ProductionAdapterError::InvalidPlanWireId { .. })
        ));

        let out_of_range = PlanWireId {
            token: adapter.wire_table.token,
            slot: adapter.wire_table.wires.len() as u32,
        };
        let range_overrides = Rc::new(BTreeMap::from([(out_of_range, value)]));
        assert!(matches!(
            adapter.immediate_value(out_of_range, &range_overrides),
            Err(ProductionAdapterError::InvalidPlanWireId { .. })
        ));
    }

    #[test]
    fn compact_plan_wire_token_allocation_exhausts_without_wrapping() {
        let almost_exhausted = AtomicU64::new(u64::MAX - 1);
        assert_eq!(allocate_plan_wire_token(&almost_exhausted), Some(u64::MAX - 1));
        assert_eq!(almost_exhausted.load(Ordering::Relaxed), u64::MAX);
        assert_eq!(allocate_plan_wire_token(&almost_exhausted), None);
        assert_eq!(almost_exhausted.load(Ordering::Relaxed), u64::MAX);

        let exhausted = AtomicU64::new(u64::MAX);
        assert_eq!(allocate_plan_wire_token(&exhausted), None);
        assert_eq!(exhausted.load(Ordering::Relaxed), u64::MAX);
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
                let mut lowering = BggTallSlotLowering::new(
                    BggTallEncodingCompiler { public_key: public_compiler.clone() },
                    BggPublicKeyWire {
                        matrix: diagonal_mask_public_key.clone(),
                        reveal_plaintext: true,
                    },
                    secret_rows.clone(),
                    BggTallEncodingSampler {
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
                    BTreeMap::new(),
                    None,
                );
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

    fn singleton_preimage_protocol() -> crate::ProtocolDecl {
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

    fn top_level_serial_parallel_protocol(layers: usize, dynamic_get: bool) -> crate::ProtocolDecl {
        use crate::{
            ComparatorEndpointBinding, ComparatorSpec, EndpointAnchor, EndpointAnchors,
            EndpointSemanticBinding, EndpointSpecId, OperationalDecoderKind,
            OperationalDecoderTarget, OutputRef, StageId, operational_protocol_from_graphs,
        };
        use mxx_dsl::{Bool, DslContext, Family, IdealSpec, Mat, Ring, SemanticAnchor};
        use mxx_ir_core::IntExpr;

        let ring = Ring::new(257, 1);
        let count = IntExpr::constant(4);
        let seed = ring.input_family("serial-parallel-seed", count.clone(), (1, 1));
        let increment = ring.input_family("serial-parallel-increment", count, (1, 1));
        let mut current = seed;
        for _ in 0..layers {
            current = Family::<Mat>::parallel_zip_many_with_broadcast_values(
                vec![current, increment.clone()],
                Vec::<Family<Mat>>::new(),
                |_index, zipped, broadcast| {
                    let [left, right] = zipped.as_slice() else {
                        return Err(mxx_dsl::DslError::Schema);
                    };
                    if !broadcast.is_empty() {
                        return Err(mxx_dsl::DslError::Schema);
                    }
                    Ok(left.clone() + right.clone())
                },
            )
            .expect("serial top-level parallel layer");
        }
        let residual = if dynamic_get {
            let selector = ring
                .input("serial-parallel-selector", (1, 1))
                .extract_coefficient_with_canonical_input_exclusive_upper(
                    0,
                    Some(BigUint::from(4_u8)),
                );
            let factor = ring.input("serial-parallel-factor", (1, 1));
            (current.get(selector) + factor.clone()) * factor
        } else {
            current.get_static(0)
        };
        let decoded = residual
            .clone()
            .threshold_decode_bools(2, 1)
            .into_iter()
            .next()
            .expect("decoder output")
            .semantic_anchor("serial-parallel.decoder")
            .expect("decoder anchor");
        let consumer = DslContext::new("serial-parallel-consumer")
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
                    DslContext::new("serial-parallel-ideal")
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
                        semantic_anchor: "serial-parallel.decoder".to_owned(),
                        semantics: EndpointSemanticBinding::ThresholdDecode,
                        workflow_output: OutputRef {
                            stage: decoder_stage.clone(),
                            output: "decoded".to_owned(),
                        },
                        ideal_output: "decoded".to_owned(),
                    }],
                };
                bundle.operational_decoder_targets = vec![OperationalDecoderTarget {
                    target_id: "serial-parallel".to_owned(),
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
        .expect("serial parallel operational protocol")
    }

    fn top_level_parallel_preimage_protocol() -> crate::ProtocolDecl {
        use crate::{
            ComparatorEndpointBinding, ComparatorSpec, EndpointAnchor, EndpointAnchors,
            EndpointSemanticBinding, EndpointSpecId, OperationalDecoderKind,
            OperationalDecoderTarget, OutputRef, StageId, operational_protocol_from_graphs,
        };
        use mxx_dsl::{Bool, DslContext, IdealSpec, Parallel, Ring, SemanticAnchor};
        use mxx_ir_core::IntExpr;

        let ring = Ring::new(257, 1);
        let trapdoors = Parallel::range(2)
            .map_values(|_| ring.sample_trapdoor(1, 3, 4, 2, 8))
            .expect("trapdoor family");
        let targets =
            Parallel::range(2).map(|_| ring.uniform_residue((1, 1))).expect("target family");
        let preimages = trapdoors
            .clone()
            .parallel_zip_mat_values(targets.clone(), |_, trapdoor, target| {
                trapdoor.sample_preimage(target, (4, 1)).as_mat()
            })
            .expect("preimage family");
        let products = trapdoors
            .public_matrices()
            .parallel_zip(preimages, |_, public, preimage| public * preimage)
            .expect("product family");
        let residual = products.get_static(0) - targets.get_static(0);
        let decoded = residual
            .clone()
            .threshold_decode_bools(2, 1)
            .into_iter()
            .next()
            .expect("decoder output")
            .semantic_anchor("top-level-parallel-preimage.decoder")
            .expect("decoder anchor");
        let consumer = DslContext::new("top-level-parallel-preimage-consumer")
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
                    DslContext::new("top-level-parallel-preimage-ideal")
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
                        semantic_anchor: "top-level-parallel-preimage.decoder".to_owned(),
                        semantics: EndpointSemanticBinding::ThresholdDecode,
                        workflow_output: OutputRef {
                            stage: decoder_stage.clone(),
                            output: "decoded".to_owned(),
                        },
                        ideal_output: "decoded".to_owned(),
                    }],
                };
                bundle.operational_decoder_targets = vec![OperationalDecoderTarget {
                    target_id: "top-level-parallel-preimage".to_owned(),
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
        .expect("top-level parallel preimage operational protocol")
    }

    #[test]
    fn top_level_serial_parallel_keeps_reducible_edges_compact_until_exposure() {
        const LAYERS: usize = 16;
        let protocol = top_level_serial_parallel_protocol(LAYERS, false);
        let plan = ProtocolPlan::build(&protocol, "serial-parallel").expect("protocol plan");
        let final_parallel = plan
            .nodes()
            .iter()
            .filter(|(wire, node)| {
                matches!(node.kind, NodeKind::ParallelLoop(_)) &&
                    matches!(wire.occurrence.definition, FrozenGraphScopeId::Root)
            })
            .map(|(wire, _)| wire.clone())
            .max_by_key(|wire| wire.wire.node.0)
            .expect("top-level parallel output");
        let mut adapter =
            ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).expect("production adapter");
        adapter.job.programs_mut().enable_diagnostic_counters();
        let nodes_before = adapter.job.expressions().node_count();
        let Value::Family(family) = adapter
            .resolve(final_parallel, Rc::new(BTreeMap::new()))
            .expect("resolve compact top-level chain")
        else {
            panic!("top-level parallel output must remain a family")
        };
        let construction_nodes = adapter.job.expressions().node_count() - nodes_before;
        let counters = adapter.job.programs().diagnostic_counters();
        assert_eq!(counters.beta_nodes_visited, 0);
        assert!(construction_nodes <= 24 * LAYERS);

        let compact_body = adapter.job.programs().family_body(family).unwrap();
        assert_eq!(
            adapter
                .job
                .programs()
                .ensure_no_reducible_generated_calls(adapter.job.expressions(), compact_body),
            Err(super::super::arena::ArenaError::ProgramSignatureMismatch)
        );
        let body_node = adapter.job.expressions().node(compact_body).unwrap();
        assert!(matches!(body_node.operator, ValueOperator::Matrix(MatrixOperation::Add)));
        let compact_calls = body_node
            .inputs
            .iter()
            .copied()
            .filter(|input| {
                matches!(
                    adapter.job.expressions().node(*input).unwrap().operator,
                    ValueOperator::ProgramCall { .. }
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(compact_calls.len(), 2);
        assert!(compact_calls.iter().all(|call| adapter.job.facts().facts(*call).is_err()));
        assert!(compact_calls.iter().all(|call| matches!(
            adapter.job.expressions().value_type(*call).unwrap(),
            ResolvedValueType::Matrix(_)
        )));
        assert!(body_node.inputs.iter().any(|input| matches!(
            adapter.job.expressions().node(*input).unwrap().operator,
            ValueOperator::ProgramCall { program }
                if adapter.job.programs().family_for_program(program).is_some_and(|family| {
                    adapter.job.programs().family_body(family).unwrap() != compact_body
                })
        )));

        let argument =
            adapter.job.expressions_mut().intern_argument(0, ResolvedValueType::Int).unwrap();
        let nodes_before_exposure = adapter.job.expressions().node_count();
        let visits_before_exposure =
            adapter.job.programs().diagnostic_counters().beta_nodes_visited;
        let exposed = adapter
            .call_family_in_program_scope(
                family,
                argument,
                TrustedIndexRange { minimum: 0, maximum_exclusive: 4 },
            )
            .unwrap();
        let exposure_nodes = adapter.job.expressions().node_count() - nodes_before_exposure;
        let exposure_visits = adapter.job.programs().diagnostic_counters().beta_nodes_visited -
            visits_before_exposure;
        assert!(exposure_nodes <= 32 * LAYERS);
        assert!(exposure_visits <= 32 * LAYERS as u64);
        adapter
            .job
            .programs()
            .ensure_no_reducible_generated_calls(adapter.job.expressions(), exposed)
            .unwrap();
        assert_eq!(
            exposed,
            adapter.job.materialize_reducible_generated_calls(compact_body).unwrap()
        );
    }

    #[test]
    fn dynamic_matrix_get_chain_stays_compact_until_root_materialization() {
        const LAYERS: usize = 32;
        let protocol = top_level_serial_parallel_protocol(LAYERS, true);
        let plan = ProtocolPlan::build(&protocol, "serial-parallel").expect("protocol plan");
        assert!(plan.nodes().values().any(|node| matches!(node.kind, NodeKind::FamilyGetDynamic)));
        assert!(plan.nodes().values().any(|node| matches!(
            node.kind,
            NodeKind::ExtractCoefficient { canonical_input_exclusive_upper: Some(_), .. }
        )));

        let mut adapter =
            ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).expect("production adapter");
        adapter.job.programs_mut().enable_diagnostic_counters();
        let nodes_before = adapter.job.expressions().node_count();
        let residual = adapter
            .resolve(plan.target().residual.clone(), Rc::new(BTreeMap::new()))
            .expect("compact dynamic residual");
        let Value::Expr(compact) = residual else { panic!("matrix residual") };
        let construction_nodes = adapter.job.expressions().node_count() - nodes_before;
        let construction_visits = adapter.job.programs().diagnostic_counters().beta_nodes_visited;
        assert!(construction_nodes <= 64 * LAYERS);
        assert!(construction_visits <= 4 * LAYERS as u64);
        assert_eq!(
            adapter
                .job
                .programs()
                .ensure_no_reducible_generated_calls(adapter.job.expressions(), compact),
            Err(super::super::arena::ArenaError::ProgramSignatureMismatch)
        );

        // Build the former eager result in the same arena. The production residual is exactly
        // `(get(selector) + factor) * factor`, so replacing only that compact call gives an exact
        // structural oracle without a second arena or a hand-written evaluator.
        let compact_multiply = adapter.job.expressions().node(compact).unwrap().clone();
        let [compact_add, factor] = compact_multiply.inputs.as_ref() else {
            panic!("dynamic residual multiply shape")
        };
        let compact_add_node = adapter.job.expressions().node(*compact_add).unwrap().clone();
        let compact_call = compact_add_node.inputs[0];
        let compact_call_node = adapter.job.expressions().node(compact_call).unwrap().clone();
        let ValueOperator::ProgramCall { program } = compact_call_node.operator else {
            panic!("dynamic get must remain a compact ProgramCall")
        };
        let family = adapter.job.programs().family_for_program(program).unwrap();
        let index = compact_call_node.inputs[0];
        let domain = adapter.job.programs().family_domain(family).unwrap();
        let eager_get = adapter
            .call_family_in_program_scope(
                family,
                index,
                TrustedIndexRange {
                    minimum: domain.minimum,
                    maximum_exclusive: domain.maximum_exclusive,
                },
            )
            .unwrap();
        let eager_add = adapter
            .job
            .expressions_mut()
            .intern(compact_add_node.operator, Box::new([eager_get, *factor]))
            .unwrap();
        let eager = adapter
            .job
            .expressions_mut()
            .intern(compact_multiply.operator, Box::new([eager_add, *factor]))
            .unwrap();

        let nodes_before_materialization = adapter.job.expressions().node_count();
        let visits_before_materialization =
            adapter.job.programs().diagnostic_counters().beta_nodes_visited;
        let Value::Expr(materialized) = adapter
            .materialize_root_value(Value::Expr(compact))
            .expect("materialized dynamic residual")
        else {
            panic!("matrix residual")
        };
        let materialization_nodes =
            adapter.job.expressions().node_count() - nodes_before_materialization;
        let materialization_visits =
            adapter.job.programs().diagnostic_counters().beta_nodes_visited -
                visits_before_materialization;
        assert!(materialization_nodes <= 96 * LAYERS);
        assert!(materialization_visits <= 96 * LAYERS as u64);
        adapter
            .job
            .programs()
            .ensure_no_reducible_generated_calls(adapter.job.expressions(), materialized)
            .unwrap();
        assert_eq!(materialized, eager);
        assert_eq!(
            adapter.job.expressions().value_type(compact).unwrap(),
            adapter.job.expressions().value_type(materialized).unwrap()
        );
    }

    #[test]
    fn matrix_parallel_finish_uses_deferred_family_call_while_scalars_reduce_eagerly() {
        let protocol = crate::toy_example::protocol();
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("toy plan");
        let mut adapter = ProductionAdapter::new(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .expect("production adapter");
        adapter.job.programs_mut().enable_diagnostic_counters();

        let matrix = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap();
        let matrix_body = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(992),
                    operation: SamplerOperation::UniformResidue { output: matrix },
                },
                Box::new([]),
            )
            .unwrap();
        let domain = FamilyDomain::new(0, 1).unwrap();
        let matrix_family = adapter.generated_family(domain, matrix_body).unwrap();
        let index = adapter.intern_index_constant(BigInt::ZERO).unwrap();
        let matrix_call = adapter
            .call_family_with_resolved_range(
                matrix_family,
                index,
                TrustedIndexRange::new(0, 1).unwrap(),
            )
            .unwrap();
        assert!(matches!(
            adapter.job.expressions().node(matrix_call).unwrap().operator,
            ValueOperator::ProgramCall { .. }
        ));
        assert_eq!(adapter.job.programs().diagnostic_counters().beta_nodes_visited, 0);

        let scalar_body =
            adapter.job.expressions_mut().intern_argument(0, ResolvedValueType::Int).unwrap();
        let scalar_family = adapter.generated_family(domain, scalar_body).unwrap();
        let scalar_call = adapter
            .call_family_with_resolved_range(
                scalar_family,
                index,
                TrustedIndexRange::new(0, 1).unwrap(),
            )
            .unwrap();
        assert!(!matches!(
            adapter.job.expressions().node(scalar_call).unwrap().operator,
            ValueOperator::ProgramCall { .. }
        ));
    }

    #[test]
    fn top_level_parallel_preimage_keeps_relation_dispatch_and_rewrite() {
        let protocol = top_level_parallel_preimage_protocol();
        let plan =
            ProtocolPlan::build(&protocol, "top-level-parallel-preimage").expect("protocol plan");
        assert!(plan.nodes().iter().any(|(wire, node)| {
            matches!(wire.occurrence.definition, FrozenGraphScopeId::Root) &&
                matches!(node.kind, NodeKind::ParallelLoop(_))
        }));
        let adapter =
            ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).expect("production adapter");
        let (mut job, roots) = adapter.lower().expect("production lowering");
        assert_eq!(job.relations().has_universal_relations(), Ok(true));
        let ProductionRoot::Closed(residual) = roots.residual else { panic!("closed residual") };
        let analysis = job.normalize_closed_root(residual).expect("residual normalization");
        assert!(analysis.counters.relation_applied >= 1);
        assert!(analysis.value.exact_nf.as_ref().is_some_and(|exact| exact.exact_terms.is_empty()));

        let eager_adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("forced-eager relation adapter");
        let (mut eager_job, eager_roots) = eager_adapter.lower_force_eager().unwrap();
        let ProductionRoot::Closed(eager_root) = eager_roots.residual else { unreachable!() };
        let eager = eager_job.normalize_closed_root(eager_root).unwrap();
        assert_eq!(
            full_nf_descriptor_map(&job, &analysis.value),
            full_nf_descriptor_map(&eager_job, &eager.value)
        );
    }

    #[test]
    fn production_compact_family_formal_index_shared_relation_occurrences_rewrite() {
        // Keep the closed/static control successful while exercising the formal-index
        // CompactFamily boundary with shared relation-bearing occurrences.
        let static_protocol = top_level_parallel_preimage_protocol();
        let static_plan =
            ProtocolPlan::build(&static_protocol, "top-level-parallel-preimage").unwrap();
        let static_adapter =
            ProductionAdapter::new(&static_protocol, &static_plan, BTreeMap::new()).unwrap();
        let (mut static_job, static_roots) = static_adapter.lower().unwrap();
        let ProductionRoot::Closed(static_root) = static_roots.residual else {
            panic!("closed/static control must remain closed")
        };
        let static_analysis = static_job.normalize_closed_root(static_root).unwrap();
        assert!(static_analysis.counters.relation_applied >= 1);

        let protocol = generated_gather_protocol(7);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").unwrap();
        let mut adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
        let occurrence = ProgramOccurrence { definition: FrozenGraphScopeId::Root, path: 0 };
        let domain = FamilyDomain::new(0, 4).unwrap();
        let range = TrustedIndexRange::new(0, 4).unwrap();
        let argument =
            adapter.job.expressions_mut().intern_argument(0, ResolvedValueType::Int).unwrap();
        let public_matrix = ResolvedMatrixType::new(256_u16.into(), 1, 1, 4).unwrap();
        let preimage_matrix = ResolvedMatrixType::new(256_u16.into(), 1, 4, 2).unwrap();
        let target_matrix = ResolvedMatrixType::new(256_u16.into(), 1, 1, 2).unwrap();
        let public = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(991),
                    operation: SamplerOperation::UniformResidue { output: public_matrix },
                },
                Box::new([]),
            )
            .unwrap();
        let preimage = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(992),
                    operation: SamplerOperation::Preimage {
                        output: preimage_matrix,
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
                    descriptor: "diagnostic-formal-index-trapdoor".to_owned(),
                    parameters: Box::new([]),
                    paired_public_event: SampleEventId(991),
                    paired_public_output_role: "value".to_owned(),
                }),
                Box::new([]),
            )
            .unwrap();
        adapter
            .job
            .insert_trapdoor_facts(
                adapter.token,
                trapdoor,
                super::super::facts::TrapdoorFacts {
                    coefficient_bound: NumericContract::Missing,
                    descriptor: "diagnostic-formal-index-trapdoor".to_owned(),
                    paired_public_event: SampleEventId(991),
                    paired_public_output_role: "value".to_owned(),
                },
            )
            .unwrap();
        let target = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(993),
                    operation: SamplerOperation::UniformResidue { output: target_matrix },
                },
                Box::new([]),
            )
            .unwrap();
        let product = adapter
            .job
            .expressions_mut()
            .intern_matrix_transform(MatrixOperation::Multiply, &[public, preimage])
            .unwrap();
        let candidate_index = adapter.relation_candidates.len();
        adapter.push_relation_candidate(RelationCandidate {
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
                product,
                argument,
                &plan.target().residual,
            )
            .unwrap();
        assert_eq!(lifted.len(), 1);
        let (_, preimage_family, public_family, trapdoor_family, target_family) = lifted[0];
        adapter.relation_candidates[candidate_index].family_operands =
            Some((preimage_family, public_family, trapdoor_family, target_family));
        let target_call = adapter
            .call_family_in_program_scope_deferred_generated(target_family, argument, range)
            .unwrap();
        // Keep the relation-bearing product and target shared across two logical occurrences.
        // Compact normalization must close each occurrence independently, rather than relying
        // on a global virtual-DAG memo, while preserving the eager relation result.
        let shared_product = adapter
            .job
            .expressions_mut()
            .intern(ValueOperator::Matrix(MatrixOperation::Add), [rewritten, rewritten].into())
            .unwrap();
        let shared_target = adapter
            .job
            .expressions_mut()
            .intern(ValueOperator::Matrix(MatrixOperation::Add), [target_call, target_call].into())
            .unwrap();
        let family_body = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Matrix(MatrixOperation::Subtract),
                [shared_product, shared_target].into(),
            )
            .unwrap();
        let family = adapter.generated_family(domain, family_body).unwrap();
        let value = Value::Family(family);
        let preflight_reason = adapter.compact_residual_preflight(&value);
        assert!(preflight_reason.is_none(), "formal-index compact preflight: {preflight_reason:?}");
        let shell_plan = adapter.build_compact_shell_plan(&value).unwrap();
        adapter.register_reached_relations().unwrap();
        adapter.job.set_compact_shell_plan(shell_plan).unwrap();
        adapter.job.freeze_relations(adapter.token).unwrap();
        let compact = adapter.job.analyze_compact_family_root(family).unwrap();
        // The two logical occurrences share one canonical relation-bearing expression.  The
        // relation dispatcher closes that canonical expression once, and polynomial expansion
        // retains both occurrences; no relation-bearing term remains afterward.
        assert_eq!(compact.counters.relation_applied, 1);
        assert_eq!(compact.counters.relation_remaining, 0);
        assert_eq!(compact.exact_term_count, 0);
        let eager = adapter.job.analyze_family_root(family).unwrap();
        assert_eq!(compact.exact_term_diagnostics, eager.exact_term_diagnostics);
        assert_eq!(compact.bounded_summary, eager.bounded_summary);
        assert_eq!(
            compact.bounded_summary.coefficient_bound(),
            eager.bounded_summary.coefficient_bound()
        );
        assert_eq!(compact.counters.relation_applied, eager.counters.relation_applied);
        assert_eq!(compact.counters.relation_remaining, eager.counters.relation_remaining);
        let compact_root = super::super::report::AnalyzedRoot {
            exact_term_count: compact.exact_term_count,
            bound: compact.bounded_summary.coefficient_bound(),
            exact_terms: compact.exact_term_diagnostics.clone(),
        };
        let eager_root = super::super::report::AnalyzedRoot {
            exact_term_count: eager.exact_term_count,
            bound: eager.bounded_summary.coefficient_bound(),
            exact_terms: eager.exact_term_diagnostics.clone(),
        };
        let decoder = super::super::report::AnalyzedRoot {
            exact_term_count: 0,
            bound: NumericContract::Known(CoefficientBound::ExactZero),
            exact_terms: Box::new([]),
        };
        let target = super::super::report::ReportTarget {
            target_id: "formal-index-relation".to_owned(),
            plaintext_modulus: 2_u8.into(),
            ciphertext_modulus: 257_u16.into(),
            boolean_interval: false,
        };
        let compact_report = super::super::report::report_analyzed_roots(
            target.clone(),
            &compact_root,
            &decoder,
            super::super::report::ReportCounters {
                normalization: compact.counters,
                ..Default::default()
            },
        )
        .unwrap();
        let eager_report = super::super::report::report_analyzed_roots(
            target,
            &eager_root,
            &decoder,
            super::super::report::ReportCounters {
                normalization: eager.counters,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(compact_report.residual, eager_report.residual);
        assert_eq!(compact_report.noise_bound, eager_report.noise_bound);
        assert_eq!(compact_report.accepted, eager_report.accepted);
        assert_eq!(compact_report.acceptance, eager_report.acceptance);
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
        use mxx_dsl::{DslContext, Ring};
        let ring = Ring::new(256, 1);
        let left = ring.input_family("left-family", 5, (1, 1));
        let right = ring.input_family("right-family", 7, (1, 1));
        let early = left.get_static(0);
        let mapped = left.clone().parallel_map(|_, value| value).unwrap();
        let zipped = mapped
            .parallel_zip_offset(right.clone(), 2, |_, first, second| first + second)
            .unwrap();
        let independent = right.parallel_map(|_, value| value).unwrap();
        let residual = early + zipped.get_static(0) + independent.get_static(0);
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

    #[derive(Clone, Copy)]
    enum GeneratedSliceCase {
        Unit,
        Static,
        OutOfBounds,
        NonAffine,
    }

    fn generated_gather_protocol(source_count: usize) -> crate::ProtocolDecl {
        generated_slice_protocol(source_count, GeneratedSliceCase::Unit)
    }

    fn generated_direct_family_protocol() -> crate::ProtocolDecl {
        use crate::ProtocolInputId;
        use mxx_dsl::{DslContext, Parallel, Ring};
        let ring = Ring::new(256, 1);
        let source = Parallel::range(1)
            .map_values(|_| {
                let left = ring.uniform_residue((1, 1));
                let right = ring.uniform_residue((1, 1));
                left + right
            })
            .expect("generated sampler family");
        let residual = source.get_static(0);
        let encrypt = DslContext::new("production-direct-family")
            .int_parameter("cutoff")
            .public_output("ciphertext", residual.clone())
            .expect("ciphertext output")
            .private_output("operational-residual", residual)
            .expect("residual output")
            .build()
            .expect("direct family graph");
        let mut protocol = generated_gather_protocol(1);
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
        protocol.bundle.input_contract.inputs.retain(|input| {
            input.id != ProtocolInputId::from("slice-source") &&
                input.id != ProtocolInputId::from("gather-source")
        });
        protocol.bundle.input_bindings.retain(|binding| {
            binding.input != ProtocolInputId::from("slice-source") &&
                binding.input != ProtocolInputId::from("gather-source")
        });
        protocol
    }

    #[derive(Clone, Copy)]
    enum FixedSliceInput {
        Hash,
        DynamicHashTag,
        Sampler,
        UnsupportedTranspose,
    }

    fn generated_fixed_slice_hash_protocol(
        source_rows: usize,
        source_columns: usize,
        row_start: usize,
        row_end: usize,
        column_start: usize,
        column_end: usize,
        input_kind: FixedSliceInput,
        shared_slice: bool,
    ) -> crate::ProtocolDecl {
        use crate::{
            ComparatorEndpointBinding, ComparatorSpec, EndpointAnchor, EndpointAnchors,
            EndpointSemanticBinding, EndpointSpecId, OperationalDecoderKind,
            OperationalDecoderTarget, OutputRef, StageId, operational_protocol_from_graphs,
        };
        use mxx_dsl::{Bool, DslContext, IdealSpec, Parallel, Ring, SemanticAnchor};
        let ring = Ring::new(256, 1);
        let key = ring.bytes_input("fixed-slice-key", 32);
        let source = ring.input("fixed-slice-source", (source_rows, source_columns));
        let family = Parallel::range(1)
            .map_values(|index| {
                let input = match input_kind {
                    FixedSliceInput::Hash => ring.hash_matrix(
                        key.clone(),
                        b"fixed-slice/v1".as_slice(),
                        (source_rows, source_columns),
                    ),
                    FixedSliceInput::DynamicHashTag => ring.hash_matrix(
                        key.clone(),
                        mxx_dsl::tag!("fixed-slice/v1", index),
                        (source_rows, source_columns),
                    ),
                    FixedSliceInput::Sampler => ring.uniform_residue((source_rows, source_columns)),
                    FixedSliceInput::UnsupportedTranspose => source.clone().transpose(),
                };
                let sliced = input.slice(
                    Some(mxx_ir_core::node::IndexRange {
                        start: IntExpr::constant(row_start),
                        end: IntExpr::constant(row_end),
                    }),
                    Some(mxx_ir_core::node::IndexRange {
                        start: IntExpr::constant(column_start),
                        end: IntExpr::constant(column_end),
                    }),
                );
                if shared_slice {
                    sliced.clone() + sliced
                } else if matches!(input_kind, FixedSliceInput::Sampler) {
                    sliced + ring.uniform_residue((row_end - row_start, column_end - column_start))
                } else {
                    sliced
                }
            })
            .expect("fixed slice family");
        let selected = family.get_static(0).slice(
            Some(mxx_ir_core::node::IndexRange {
                start: IntExpr::constant(0),
                end: IntExpr::constant(row_end - row_start),
            }),
            None,
        );
        let decoded = selected
            .threshold_decode_bools(2, 1)
            .into_iter()
            .next()
            .expect("fixed slice decoder")
            .semantic_anchor("fixed-slice.decoder")
            .expect("fixed slice decoder anchor");
        let graph = DslContext::new("production-fixed-slice-hash")
            .private_family_output("operational-residual", family)
            .expect("fixed slice residual")
            .bool_output("decoded", decoded)
            .expect("fixed slice decoded output")
            .build()
            .expect("fixed slice graph");
        let decoder_node = graph.graph.outputs()["decoded"].value.node;
        let endpoint = EndpointSpecId::ToyThresholdDecode;
        let stage = StageId("fixed-slice".to_owned());
        operational_protocol_from_graphs(
            vec![(stage.0.clone(), &graph)],
            &stage.0,
            &BTreeMap::new(),
            &BTreeMap::new(),
            |bundle| {
                bundle.ideal = IdealSpec::new(
                    DslContext::new("fixed-slice-ideal")
                        .bool_output("decoded", Bool::constant(false))
                        .expect("fixed slice ideal output")
                        .build()
                        .expect("fixed slice ideal graph"),
                )
                .expect("fixed slice ideal");
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
                        stage: stage.clone(),
                        semantic_anchor: "fixed-slice.decoder".to_owned(),
                        semantics: EndpointSemanticBinding::ThresholdDecode,
                        workflow_output: OutputRef {
                            stage: stage.clone(),
                            output: "decoded".to_owned(),
                        },
                        ideal_output: "decoded".to_owned(),
                    }],
                };
                bundle.operational_decoder_targets = vec![OperationalDecoderTarget {
                    target_id: "fixed-slice".to_owned(),
                    residual_stage: stage.clone(),
                    residual_output: "operational-residual".to_owned(),
                    decoder_stage: stage.clone(),
                    decoder_node,
                    kind: OperationalDecoderKind::ThresholdDecode {
                        plaintext_modulus: IntExpr::constant(2),
                    },
                }];
                bundle.endpoint_specs = vec![endpoint];
            },
        )
        .expect("fixed slice operational protocol")
    }

    fn generated_indexed_family_protocol(
        count: u64,
        binder_dependent_gadget: bool,
        binder_dependent_explicit: bool,
        nested_depth: usize,
    ) -> crate::ProtocolDecl {
        use crate::{
            ComparatorEndpointBinding, ComparatorSpec, EndpointAnchor, EndpointAnchors,
            EndpointSemanticBinding, EndpointSpecId, OperationalDecoderKind,
            OperationalDecoderTarget, OutputRef, StageId, operational_protocol_from_graphs,
        };
        use mxx_dsl::{DslContext, IdealSpec, Ring, SemanticAnchor};
        let ring = Ring::new(256, 1);
        let source = ring.input_family("indexed-source", IntExpr::constant(count), (1, 1));
        // The map callback is lowered as the family-owned formal Argument(0), so this keeps
        // every source lane distinct instead of sampling one selected index during preflight.
        let mut family = if binder_dependent_explicit {
            source
                .parallel_map(|index, value| {
                    index
                        .as_int()
                        .select(vec![value.clone(), value])
                        .expect("binder-dependent explicit family body")
                        .decompose(4, 2)
                        .as_mat()
                })
                .expect("binder-dependent explicit family body")
        } else if binder_dependent_gadget {
            source
                .parallel_map(|_, value| value.decompose(4, 2).as_mat())
                .expect("binder-dependent family body")
        } else {
            source.parallel_map(|_, value| value + ring.zero((1, 1))).expect("indexed family body")
        };
        for _ in 1..nested_depth {
            family = family
                .parallel_map(|_, value| value + ring.zero((1, 1)))
                .expect("nested indexed family body");
        }
        let selected = family.get_static(0).slice(
            Some(mxx_ir_core::node::IndexRange {
                start: IntExpr::constant(0),
                end: IntExpr::constant(1),
            }),
            None,
        );
        let decoded = selected
            .threshold_decode_bools(2, 1)
            .into_iter()
            .next()
            .expect("indexed family decoder")
            .semantic_anchor("indexed-family.decoder")
            .expect("indexed family decoder anchor");
        let graph = DslContext::new("production-indexed-family")
            .private_family_output("operational-residual", family)
            .expect("indexed family residual")
            .bool_output("decoded", decoded)
            .expect("indexed family decoder")
            .build()
            .expect("indexed family graph");
        let decoder_node = graph.graph.outputs()["decoded"].value.node;
        let endpoint = EndpointSpecId::ToyThresholdDecode;
        let stage = StageId("indexed-family".to_owned());
        let stage_name = stage.0.clone();
        operational_protocol_from_graphs(
            vec![(stage_name.clone(), &graph)],
            &stage_name,
            &BTreeMap::from([(
                "indexed-source".to_owned(),
                crate::ExactMatrixInputMetadata {
                    canonical_coefficient_exclusive_upper_bound: Some(IntExpr::constant(2)),
                    is_constant_polynomial: false,
                },
            )]),
            &BTreeMap::new(),
            |bundle| {
                bundle.ideal = IdealSpec::new(
                    DslContext::new("indexed-family-ideal")
                        .bool_output("decoded", mxx_dsl::Bool::constant(false))
                        .expect("indexed family ideal output")
                        .build()
                        .expect("indexed family ideal graph"),
                )
                .expect("indexed family ideal");
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
                        stage: stage.clone(),
                        semantic_anchor: "indexed-family.decoder".to_owned(),
                        semantics: EndpointSemanticBinding::ThresholdDecode,
                        workflow_output: OutputRef {
                            stage: stage.clone(),
                            output: "decoded".to_owned(),
                        },
                        ideal_output: "decoded".to_owned(),
                    }],
                };
                bundle.operational_decoder_targets = vec![OperationalDecoderTarget {
                    target_id: "indexed-family-production".to_owned(),
                    residual_stage: stage.clone(),
                    residual_output: "operational-residual".to_owned(),
                    decoder_stage: stage.clone(),
                    decoder_node,
                    kind: OperationalDecoderKind::ThresholdDecode {
                        plaintext_modulus: IntExpr::constant(2),
                    },
                }];
                bundle.endpoint_specs = vec![endpoint];
            },
        )
        .expect("indexed family operational protocol")
    }

    fn generated_extract_binding_family_protocol() -> crate::ProtocolDecl {
        use mxx_dsl::{DslContext, IdealSpec, Parallel, Ring, SemanticAnchor};

        let ring = Ring::new(256, 1);
        let family = Parallel::range(4)
            .map_values(|index| {
                let matrix_family = Parallel::range(256)
                    .map_values(|_| ring.uniform_residue((1, 1)))
                    .expect("binding matrix family");
                let selected = matrix_family.get(index.as_int());
                let selector = selected.extract_coefficient_with_canonical_input_exclusive_upper(
                    0,
                    Some(BigUint::from(256_u16)),
                );
                matrix_family.get(selector) + ring.zero((1, 1))
            })
            .expect("extract-binding family");
        let decoder_input = family.get_static(0).slice(
            Some(mxx_ir_core::node::IndexRange {
                start: IntExpr::constant(0),
                end: IntExpr::constant(1),
            }),
            None,
        );
        let graph = DslContext::new("production-extract-binding-family")
            .private_family_output("operational-residual", family)
            .expect("extract-binding residual")
            .bool_output(
                "decoded",
                decoder_input
                    .threshold_decode_bools(2, 1)
                    .into_iter()
                    .next()
                    .expect("extract-binding decoder")
                    .semantic_anchor("extract-binding.decoder")
                    .expect("extract-binding decoder anchor"),
            )
            .expect("extract-binding decoder output")
            .build()
            .expect("extract-binding graph");
        let decoder_node = graph.graph.outputs()["decoded"].value.node;
        let endpoint = crate::EndpointSpecId::ToyThresholdDecode;
        let stage = crate::StageId("extract-binding".to_owned());
        let stage_name = stage.0.clone();
        crate::operational_protocol_from_graphs(
            vec![(stage_name.clone(), &graph)],
            &stage_name,
            &BTreeMap::new(),
            &BTreeMap::new(),
            |bundle| {
                bundle.ideal = IdealSpec::new(
                    DslContext::new("extract-binding-ideal")
                        .bool_output("decoded", mxx_dsl::Bool::constant(false))
                        .expect("extract-binding ideal output")
                        .build()
                        .expect("extract-binding ideal graph"),
                )
                .expect("extract-binding ideal");
                bundle.comparator = crate::ComparatorSpec::Equality {
                    endpoints: vec![crate::ComparatorEndpointBinding {
                        endpoint,
                        actual_input: "decoded".to_owned(),
                        ideal_input: "decoded".to_owned(),
                        result_output: "failure".to_owned(),
                        failure_value: true,
                    }],
                };
                bundle.endpoints = crate::EndpointAnchors {
                    entries: vec![crate::EndpointAnchor {
                        spec: endpoint,
                        stage: stage.clone(),
                        semantic_anchor: "extract-binding.decoder".to_owned(),
                        semantics: crate::EndpointSemanticBinding::ThresholdDecode,
                        workflow_output: crate::OutputRef {
                            stage: stage.clone(),
                            output: "decoded".to_owned(),
                        },
                        ideal_output: "decoded".to_owned(),
                    }],
                };
                bundle.operational_decoder_targets = vec![crate::OperationalDecoderTarget {
                    target_id: "toy-threshold".to_owned(),
                    residual_stage: stage.clone(),
                    residual_output: "operational-residual".to_owned(),
                    decoder_stage: stage.clone(),
                    decoder_node,
                    kind: crate::OperationalDecoderKind::ThresholdDecode {
                        plaintext_modulus: IntExpr::constant(2),
                    },
                }];
                bundle.endpoint_specs = vec![endpoint];
            },
        )
        .expect("extract-binding protocol")
    }

    #[derive(Clone, Copy)]
    enum GeneratedGadgetProductCase {
        Single,
        Sum,
        Shared,
        Mixed,
        Mismatch,
        Standalone,
    }

    fn generated_gadget_product_protocol(
        reverse: bool,
        case: GeneratedGadgetProductCase,
    ) -> crate::ProtocolDecl {
        use crate::ProtocolInputId;
        use mxx_dsl::SemanticAnchor;
        use mxx_ir_core::{
            Graph, IntExpr,
            artifact::ArtifactConfidentiality,
            graph::{GraphOutput, NodeHandle, SubgraphHandle, with_new_construction_scope},
            node::{ConstantMatrix, MatrixBinaryOp, NodeKind},
            types::{MatrixType, WireType},
        };
        let input_matrix = MatrixType {
            modulus: IntExpr::constant(256),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(2),
            columns: IntExpr::constant(4),
        };
        let gadget_matrix = MatrixType {
            rows: if reverse { IntExpr::constant(4) } else { IntExpr::constant(2) },
            columns: IntExpr::constant(4),
            ..input_matrix.clone()
        };
        let decomposition_matrix =
            MatrixType { rows: IntExpr::constant(4), ..input_matrix.clone() };
        let output_matrix = if reverse || matches!(case, GeneratedGadgetProductCase::Standalone) {
            MatrixType {
                rows: IntExpr::constant(4),
                columns: IntExpr::constant(4),
                ..input_matrix.clone()
            }
        } else {
            input_matrix.clone()
        };
        let (body, _) = with_new_construction_scope(|scope| {
            let input = NodeHandle::new(
                NodeKind::UniformResidueSample { matrix_type: input_matrix.clone() },
                Vec::new(),
                vec![WireType::Matrix(input_matrix.clone())],
            )
            .output(0)
            .expect("gadget input");
            let decomposition = NodeHandle::new(
                NodeKind::GadgetDecompose {
                    base: IntExpr::constant(4),
                    small: false,
                    digit_count: IntExpr::constant(2),
                },
                vec![input],
                vec![WireType::Preimage(decomposition_matrix.clone())],
            )
            .output(0)
            .expect("gadget decomposition");
            let gadget = NodeHandle::new(
                NodeKind::ConstantMatrix {
                    matrix_type: gadget_matrix.clone(),
                    value: ConstantMatrix::Gadget {
                        base: if matches!(case, GeneratedGadgetProductCase::Mismatch) {
                            IntExpr::constant(8)
                        } else {
                            IntExpr::constant(4)
                        },
                        small: false,
                    },
                },
                Vec::new(),
                vec![WireType::Matrix(gadget_matrix.clone())],
            )
            .output(0)
            .expect("gadget source");
            let gadget_sum = if matches!(case, GeneratedGadgetProductCase::Sum) {
                let gadget_two = NodeHandle::new(
                    NodeKind::ConstantMatrix {
                        matrix_type: gadget_matrix.clone(),
                        value: ConstantMatrix::Gadget { base: IntExpr::constant(4), small: false },
                    },
                    Vec::new(),
                    vec![WireType::Matrix(gadget_matrix.clone())],
                )
                .output(0)
                .expect("second gadget source");
                Some(
                    NodeHandle::new(
                        NodeKind::MatrixBinary(MatrixBinaryOp::Add),
                        vec![gadget.clone(), gadget_two],
                        vec![WireType::Matrix(gadget_matrix.clone())],
                    )
                    .output(0)
                    .expect("gadget sum"),
                )
            } else {
                None
            };
            let make_product = |left, right| {
                NodeHandle::new(
                    NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
                    vec![left, right],
                    vec![WireType::Matrix(output_matrix.clone())],
                )
                .output(0)
                .expect("gadget product")
            };
            let product = if matches!(case, GeneratedGadgetProductCase::Standalone) {
                decomposition.clone()
            } else if reverse {
                make_product(decomposition, gadget)
            } else if let Some(sum) = gadget_sum {
                make_product(sum, decomposition)
            } else if matches!(case, GeneratedGadgetProductCase::Shared) {
                let gadget_two = NodeHandle::new(
                    NodeKind::ConstantMatrix {
                        matrix_type: gadget_matrix.clone(),
                        value: ConstantMatrix::Gadget { base: IntExpr::constant(4), small: false },
                    },
                    Vec::new(),
                    vec![WireType::Matrix(gadget_matrix.clone())],
                )
                .output(0)
                .expect("second shared gadget source");
                let second_left = NodeHandle::new(
                    NodeKind::MatrixBinary(MatrixBinaryOp::Add),
                    vec![gadget.clone(), gadget_two],
                    vec![WireType::Matrix(gadget_matrix.clone())],
                )
                .output(0)
                .expect("second shared gadget sum");
                let first = make_product(gadget, decomposition.clone());
                let second = make_product(second_left, decomposition);
                NodeHandle::new(
                    NodeKind::MatrixBinary(MatrixBinaryOp::Add),
                    vec![first, second],
                    vec![WireType::Matrix(output_matrix.clone())],
                )
                .output(0)
                .expect("shared product sum")
            } else if matches!(case, GeneratedGadgetProductCase::Mixed) {
                let ordinary = NodeHandle::new(
                    NodeKind::UniformResidueSample { matrix_type: gadget_matrix.clone() },
                    Vec::new(),
                    vec![WireType::Matrix(gadget_matrix.clone())],
                )
                .output(0)
                .expect("ordinary mixed product source");
                let first = make_product(gadget, decomposition.clone());
                let second = make_product(ordinary, decomposition);
                NodeHandle::new(
                    NodeKind::MatrixBinary(MatrixBinaryOp::Add),
                    vec![first, second],
                    vec![WireType::Matrix(output_matrix.clone())],
                )
                .output(0)
                .expect("mixed product sum")
            } else {
                make_product(gadget, decomposition)
            };
            (
                SubgraphHandle::new(
                    "gadget-product-body",
                    scope,
                    Vec::new(),
                    vec![product.clone()],
                )
                .expect("gadget product body"),
                product,
            )
        });
        let family = NodeHandle::parallel_loop(
            body,
            Vec::new(),
            vec![WireType::IndexedFamily {
                element: Box::new(if matches!(case, GeneratedGadgetProductCase::Standalone) {
                    WireType::Preimage(decomposition_matrix.clone())
                } else {
                    WireType::Matrix(output_matrix.clone())
                }),
                count: IntExpr::constant(1),
            }],
            mxx_ir_core::node::ParallelLoop {
                count: IntExpr::constant(1),
                minimum_count: 0,
                index_slot: 0,
                bindings: Vec::new(),
                input_modes: Vec::new(),
            },
        )
        .output(0)
        .expect("gadget product family");
        let residual = NodeHandle::new(
            NodeKind::FamilyGetStatic { index: IntExpr::constant(0) },
            vec![family],
            vec![WireType::Matrix(output_matrix.clone())],
        )
        .output(0)
        .expect("gadget product residual");
        let graph = Graph::freeze(
            format!("production-gadget-product-{reverse}"),
            vec![mxx_ir_core::CompileParameter {
                name: "cutoff".to_owned(),
                kind: mxx_ir_core::CompileParameterKind::Integer,
            }],
            BTreeMap::from([
                (
                    "ciphertext".to_owned(),
                    GraphOutput {
                        value: residual.clone(),
                        confidentiality: Some(ArtifactConfidentiality::Public),
                    },
                ),
                (
                    "operational-residual".to_owned(),
                    GraphOutput { value: residual, confidentiality: None },
                ),
            ]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .expect("gadget product graph")
        .0;
        let mut protocol = generated_gather_protocol(1);
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
        protocol.bundle.input_contract.inputs.retain(|input| {
            input.id != ProtocolInputId::from("slice-source") &&
                input.id != ProtocolInputId::from("gather-source")
        });
        protocol.bundle.input_bindings.retain(|binding| {
            binding.input != ProtocolInputId::from("slice-source") &&
                binding.input != ProtocolInputId::from("gather-source")
        });
        let production_id = mxx_ir_core::artifact::ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([0; 32]),
            execution_nonce: [0; 32],
        };
        let ring = mxx_dsl::Ring::new(256, 1);
        let ciphertext = ring.artifact_input(
            production_id,
            "ciphertext",
            if reverse || matches!(case, GeneratedGadgetProductCase::Standalone) {
                (4, 4)
            } else {
                (2, 4)
            },
            mxx_ir_core::artifact::ArtifactConfidentiality::Public,
        );
        let decoded = ciphertext
            .threshold_decode_bools(IntExpr::constant(2), 1)
            .into_iter()
            .next()
            .expect("gadget product decoder output")
            .semantic_anchor(crate::toy_example::DECODED_ENDPOINT)
            .expect("gadget product decoder anchor");
        let decrypt = mxx_dsl::DslContext::new("production-gadget-product-decrypt")
            .int_parameter("cutoff")
            .bool_output("decoded", decoded)
            .expect("gadget product decoder output")
            .build()
            .expect("gadget product decoder graph");
        let decrypt_stage = protocol
            .bundle
            .workflow
            .stages
            .iter_mut()
            .find(|stage| stage.id == StageId("decrypt".to_owned()))
            .expect("decrypt stage");
        decrypt_stage.graph = decrypt.graph;
        decrypt_stage.semantic_anchors = decrypt.anchors;
        decrypt_stage.derivation_attachments = decrypt.derivation_attachments;
        protocol
    }

    fn generated_scale_protocol(composite_scalar: bool) -> crate::ProtocolDecl {
        use crate::ProtocolInputId;
        use mxx_ir_core::{
            Graph,
            artifact::ArtifactConfidentiality,
            graph::{GraphOutput, NodeHandle, SubgraphHandle, with_new_construction_scope},
            node::NodeKind,
            types::{MatrixType, WireType},
        };
        let matrix = MatrixType {
            modulus: IntExpr::constant(256),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let scalar = if composite_scalar {
            IntExpr::Add(Box::new(IntExpr::constant(2)), Box::new(IntExpr::constant(1)))
        } else {
            IntExpr::constant(3)
        };
        let (body, _) = with_new_construction_scope(|scope| {
            let source = NodeHandle::new(
                NodeKind::UniformResidueSample { matrix_type: matrix.clone() },
                Vec::new(),
                vec![WireType::Matrix(matrix.clone())],
            )
            .output(0)
            .expect("scale source");
            let scaled = NodeHandle::new(
                NodeKind::MatrixScale { scalar },
                vec![source],
                vec![WireType::Matrix(matrix.clone())],
            )
            .output(0)
            .expect("scale node");
            (
                SubgraphHandle::new("scale-body", scope, Vec::new(), vec![scaled.clone()])
                    .expect("scale body"),
                scaled,
            )
        });
        let family = NodeHandle::parallel_loop(
            body,
            Vec::new(),
            vec![WireType::IndexedFamily {
                element: Box::new(WireType::Matrix(matrix.clone())),
                count: IntExpr::constant(1),
            }],
            mxx_ir_core::node::ParallelLoop {
                count: IntExpr::constant(1),
                minimum_count: 0,
                index_slot: 0,
                bindings: Vec::new(),
                input_modes: Vec::new(),
            },
        )
        .output(0)
        .expect("scale family");
        let residual = NodeHandle::new(
            NodeKind::FamilyGetStatic { index: IntExpr::constant(0) },
            vec![family],
            vec![WireType::Matrix(matrix)],
        )
        .output(0)
        .expect("scale residual");
        let graph = Graph::freeze(
            format!("production-scale-{composite_scalar}"),
            vec![mxx_ir_core::CompileParameter {
                name: "cutoff".to_owned(),
                kind: mxx_ir_core::CompileParameterKind::Integer,
            }],
            BTreeMap::from([
                (
                    "ciphertext".to_owned(),
                    GraphOutput {
                        value: residual.clone(),
                        confidentiality: Some(ArtifactConfidentiality::Public),
                    },
                ),
                (
                    "operational-residual".to_owned(),
                    GraphOutput { value: residual, confidentiality: None },
                ),
            ]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .expect("scale graph")
        .0;
        let mut protocol = generated_gather_protocol(1);
        let stage = protocol
            .bundle
            .workflow
            .stages
            .iter_mut()
            .find(|stage| stage.id == StageId("encrypt".to_owned()))
            .expect("encrypt stage");
        stage.graph = graph;
        stage.semantic_anchors = Default::default();
        stage.derivation_attachments = Default::default();
        protocol.bundle.input_contract.inputs.retain(|input| {
            input.id != ProtocolInputId::from("slice-source") &&
                input.id != ProtocolInputId::from("gather-source")
        });
        protocol.bundle.input_bindings.retain(|binding| {
            binding.input != ProtocolInputId::from("slice-source") &&
                binding.input != ProtocolInputId::from("gather-source")
        });
        protocol
    }

    fn generated_product_protocol(
        reverse: bool,
        cancellation: bool,
        one_sided_scalar: bool,
        bounded_non_scalar: bool,
    ) -> crate::ProtocolDecl {
        generated_product_protocol_with_shape(
            reverse,
            cancellation,
            one_sided_scalar,
            bounded_non_scalar,
            2,
            2,
        )
    }

    fn generated_product_protocol_with_shape(
        reverse: bool,
        cancellation: bool,
        one_sided_scalar: bool,
        bounded_non_scalar: bool,
        rows: usize,
        columns: usize,
    ) -> crate::ProtocolDecl {
        use crate::ProtocolInputId;
        use mxx_dsl::SemanticAnchor;
        use mxx_ir_core::{
            Graph, RealExpr,
            artifact::ArtifactConfidentiality,
            graph::{GraphOutput, NodeHandle, SubgraphHandle, with_new_construction_scope},
            node::{ConstantMatrix, MatrixBinaryOp, NodeKind},
            types::{MatrixType, WireType},
        };
        let matrix = MatrixType {
            modulus: IntExpr::constant(256),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(rows as u64),
            columns: IntExpr::constant(columns as u64),
        };
        let (body, _) = with_new_construction_scope(|scope| {
            let sample = |event: &str, gaussian: bool, sample_matrix: &MatrixType| {
                let kind = if gaussian {
                    NodeKind::GaussianSample {
                        matrix_type: sample_matrix.clone(),
                        sigma: RealExpr::from_integer(1),
                        max_coefficient_bound: IntExpr::constant(1),
                    }
                } else {
                    NodeKind::UniformResidueSample { matrix_type: sample_matrix.clone() }
                };
                NodeHandle::new(kind, Vec::new(), vec![WireType::Matrix(sample_matrix.clone())])
                    .output(0)
                    .unwrap_or_else(|| panic!("{event} source"))
            };
            let scalar_matrix = MatrixType {
                modulus: IntExpr::constant(256),
                ring_dimension: IntExpr::constant(1),
                rows: IntExpr::constant(1),
                columns: IntExpr::constant(1),
            };
            let left = if one_sided_scalar && bounded_non_scalar {
                NodeHandle::new(
                    NodeKind::ConstantMatrix {
                        matrix_type: matrix.clone(),
                        value: ConstantMatrix::Identity,
                    },
                    Vec::new(),
                    vec![WireType::Matrix(matrix.clone())],
                )
                .output(0)
                .expect("bounded large source")
            } else {
                sample("left", false, &matrix)
            };
            let right = if one_sided_scalar {
                NodeHandle::new(
                    NodeKind::ConstantMatrix {
                        matrix_type: scalar_matrix.clone(),
                        value: mxx_ir_core::node::ConstantMatrix::Identity,
                    },
                    Vec::new(),
                    vec![WireType::Matrix(scalar_matrix)],
                )
                .output(0)
                .expect("scalar source")
            } else {
                sample("right", true, &matrix)
            };
            let product = |left, right| {
                NodeHandle::new(
                    NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
                    vec![left, right],
                    vec![WireType::Matrix(matrix.clone())],
                )
                .output(0)
                .expect("product node")
            };
            let unscaled = if reverse {
                product(right.clone(), left.clone())
            } else {
                product(left.clone(), right.clone())
            };
            let output = if cancellation {
                let scaled_left = NodeHandle::new(
                    NodeKind::MatrixScale { scalar: IntExpr::constant(1) },
                    vec![left.clone()],
                    vec![WireType::Matrix(matrix.clone())],
                )
                .output(0)
                .expect("unit scale node");
                let scaled = if reverse {
                    product(right.clone(), scaled_left)
                } else {
                    product(scaled_left, right.clone())
                };
                NodeHandle::new(
                    NodeKind::MatrixBinary(MatrixBinaryOp::Subtract),
                    vec![scaled, unscaled],
                    vec![WireType::Matrix(matrix.clone())],
                )
                .output(0)
                .expect("cancellation node")
            } else {
                unscaled
            };
            (
                SubgraphHandle::new("product-body", scope, Vec::new(), vec![output.clone()])
                    .expect("product body"),
                output,
            )
        });
        let family = NodeHandle::parallel_loop(
            body,
            Vec::new(),
            vec![WireType::IndexedFamily {
                element: Box::new(WireType::Matrix(matrix.clone())),
                count: IntExpr::constant(1),
            }],
            mxx_ir_core::node::ParallelLoop {
                count: IntExpr::constant(1),
                minimum_count: 0,
                index_slot: 0,
                bindings: Vec::new(),
                input_modes: Vec::new(),
            },
        )
        .output(0)
        .expect("product family");
        let residual = NodeHandle::new(
            NodeKind::FamilyGetStatic { index: IntExpr::constant(0) },
            vec![family],
            vec![WireType::Matrix(matrix)],
        )
        .output(0)
        .expect("product residual");
        let graph = Graph::freeze(
            format!("production-product-{reverse}-{cancellation}-{one_sided_scalar}"),
            vec![mxx_ir_core::CompileParameter {
                name: "cutoff".to_owned(),
                kind: mxx_ir_core::CompileParameterKind::Integer,
            }],
            BTreeMap::from([
                (
                    "ciphertext".to_owned(),
                    GraphOutput {
                        value: residual.clone(),
                        confidentiality: Some(ArtifactConfidentiality::Public),
                    },
                ),
                (
                    "operational-residual".to_owned(),
                    GraphOutput { value: residual, confidentiality: None },
                ),
            ]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .expect("product graph")
        .0;
        let mut protocol = generated_gather_protocol(1);
        let stage = protocol
            .bundle
            .workflow
            .stages
            .iter_mut()
            .find(|stage| stage.id == StageId("encrypt".to_owned()))
            .expect("encrypt stage");
        stage.graph = graph;
        stage.semantic_anchors = Default::default();
        stage.derivation_attachments = Default::default();
        let ring = mxx_dsl::Ring::new(256, 1);
        let ciphertext = ring.artifact_input(
            mxx_ir_core::artifact::ProductionId {
                spec_hash: mxx_ir_core::artifact::SpecHash([0; 32]),
                execution_nonce: [0; 32],
            },
            "ciphertext",
            (rows, columns),
            mxx_ir_core::artifact::ArtifactConfidentiality::Public,
        );
        let decoded = ciphertext
            .threshold_decode_bools(IntExpr::constant(2), 1)
            .into_iter()
            .next()
            .expect("product decoder output")
            .semantic_anchor(crate::toy_example::DECODED_ENDPOINT)
            .expect("product decoder anchor");
        let decrypt = mxx_dsl::DslContext::new("production-product-decrypt")
            .int_parameter("cutoff")
            .bool_output("decoded", decoded)
            .expect("product decoder graph")
            .build()
            .expect("product decoder graph");
        let decrypt_stage = protocol
            .bundle
            .workflow
            .stages
            .iter_mut()
            .find(|stage| stage.id == StageId("decrypt".to_owned()))
            .expect("decrypt stage");
        decrypt_stage.graph = decrypt.graph;
        decrypt_stage.semantic_anchors = decrypt.anchors;
        decrypt_stage.derivation_attachments = decrypt.derivation_attachments;
        protocol.bundle.input_contract.inputs.retain(|input| {
            input.id != ProtocolInputId::from("slice-source") &&
                input.id != ProtocolInputId::from("gather-source")
        });
        protocol.bundle.input_bindings.retain(|binding| {
            binding.input != ProtocolInputId::from("slice-source") &&
                binding.input != ProtocolInputId::from("gather-source")
        });
        protocol
    }

    #[derive(Clone, Copy, Debug)]
    enum GeneratedScalarNegativeCase {
        Composite,
        MixedParent,
        BothScalar,
        MissingBound,
        LargeBound,
    }

    /// Build a complete production protocol for a scalar-action candidate which must select the
    /// eager path.  Keeping these as real Graph-IR protocols makes the test exercise resolution,
    /// preflight, relation registration, and the normal production closure rather than only the
    /// private arena constructors.
    fn generated_scalar_negative_protocol(
        case: GeneratedScalarNegativeCase,
    ) -> crate::ProtocolDecl {
        use crate::ProtocolInputId;
        use mxx_dsl::SemanticAnchor;
        use mxx_ir_core::{
            Graph, IntExpr,
            artifact::ArtifactConfidentiality,
            graph::{GraphOutput, NodeHandle, SubgraphHandle, with_new_construction_scope},
            node::{ConstantMatrix, MatrixBinaryOp, NodeKind},
            types::{MatrixType, WireType},
        };
        let matrix = MatrixType {
            modulus: IntExpr::constant(256),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(2),
            columns: IntExpr::constant(2),
        };
        let scalar_matrix = MatrixType {
            modulus: IntExpr::constant(256),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let output_matrix = if matches!(case, GeneratedScalarNegativeCase::BothScalar) {
            scalar_matrix.clone()
        } else {
            matrix.clone()
        };
        let (body, _) = with_new_construction_scope(|scope| {
            let sample = |matrix_type: &MatrixType, node: &str| {
                NodeHandle::new(
                    NodeKind::UniformResidueSample { matrix_type: matrix_type.clone() },
                    Vec::new(),
                    vec![WireType::Matrix(matrix_type.clone())],
                )
                .output(0)
                .unwrap_or_else(|| panic!("{node} sample"))
            };
            let constant = |matrix_type: &MatrixType, value: ConstantMatrix| {
                NodeHandle::new(
                    NodeKind::ConstantMatrix { matrix_type: matrix_type.clone(), value },
                    Vec::new(),
                    vec![WireType::Matrix(matrix_type.clone())],
                )
                .output(0)
                .expect("scalar constant")
            };
            let product = |left, right, output: &MatrixType| {
                NodeHandle::new(
                    NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
                    vec![left, right],
                    vec![WireType::Matrix(output.clone())],
                )
                .output(0)
                .expect("scalar-action product")
            };
            let large = sample(&matrix, "large");
            let scalar = match case {
                GeneratedScalarNegativeCase::MissingBound => sample(&scalar_matrix, "scalar"),
                GeneratedScalarNegativeCase::LargeBound => constant(
                    &scalar_matrix,
                    ConstantMatrix::Gadget { base: IntExpr::constant(4), small: false },
                ),
                _ => constant(&scalar_matrix, ConstantMatrix::Identity),
            };
            let result = match case {
                GeneratedScalarNegativeCase::Composite => {
                    let zero = constant(&scalar_matrix, ConstantMatrix::Zero);
                    let composite = NodeHandle::new(
                        NodeKind::MatrixBinary(MatrixBinaryOp::Add),
                        vec![scalar.clone(), zero],
                        vec![WireType::Matrix(scalar_matrix.clone())],
                    )
                    .output(0)
                    .expect("composite scalar");
                    product(large, composite, &matrix)
                }
                GeneratedScalarNegativeCase::MixedParent => {
                    let direct = product(large, scalar.clone(), &matrix);
                    let other = sample(&matrix, "mixed");
                    let negated = NodeHandle::new(
                        NodeKind::MatrixNegate,
                        vec![scalar],
                        vec![WireType::Matrix(scalar_matrix.clone())],
                    )
                    .output(0)
                    .expect("unauthorized scalar parent");
                    let unauthorized = product(other, negated, &matrix);
                    NodeHandle::new(
                        NodeKind::MatrixBinary(MatrixBinaryOp::Add),
                        vec![direct, unauthorized],
                        vec![WireType::Matrix(matrix.clone())],
                    )
                    .output(0)
                    .expect("mixed scalar products")
                }
                GeneratedScalarNegativeCase::BothScalar => {
                    let other = constant(&scalar_matrix, ConstantMatrix::Identity);
                    product(scalar, other, &scalar_matrix)
                }
                GeneratedScalarNegativeCase::MissingBound |
                GeneratedScalarNegativeCase::LargeBound => product(scalar, large, &matrix),
            };
            (
                SubgraphHandle::new(
                    "scalar-negative-body",
                    scope,
                    Vec::new(),
                    vec![result.clone()],
                )
                .expect("scalar-negative body"),
                result,
            )
        });
        let family = NodeHandle::parallel_loop(
            body,
            Vec::new(),
            vec![WireType::IndexedFamily {
                element: Box::new(WireType::Matrix(output_matrix.clone())),
                count: IntExpr::constant(1),
            }],
            mxx_ir_core::node::ParallelLoop {
                count: IntExpr::constant(1),
                minimum_count: 0,
                index_slot: 0,
                bindings: Vec::new(),
                input_modes: Vec::new(),
            },
        )
        .output(0)
        .expect("scalar-negative family");
        let residual = NodeHandle::new(
            NodeKind::FamilyGetStatic { index: IntExpr::constant(0) },
            vec![family],
            vec![WireType::Matrix(output_matrix.clone())],
        )
        .output(0)
        .expect("scalar-negative residual");
        let graph = Graph::freeze(
            format!("production-scalar-negative-{case:?}"),
            vec![mxx_ir_core::CompileParameter {
                name: "cutoff".to_owned(),
                kind: mxx_ir_core::CompileParameterKind::Integer,
            }],
            BTreeMap::from([
                (
                    "ciphertext".to_owned(),
                    GraphOutput {
                        value: residual.clone(),
                        confidentiality: Some(ArtifactConfidentiality::Public),
                    },
                ),
                (
                    "operational-residual".to_owned(),
                    GraphOutput { value: residual, confidentiality: None },
                ),
            ]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .expect("scalar-negative graph")
        .0;
        let mut protocol = generated_gather_protocol(1);
        let encrypt_stage = protocol
            .bundle
            .workflow
            .stages
            .iter_mut()
            .find(|stage| stage.id == StageId("encrypt".to_owned()))
            .expect("scalar-negative encrypt stage");
        encrypt_stage.graph = graph;
        encrypt_stage.semantic_anchors = Default::default();
        encrypt_stage.derivation_attachments = Default::default();
        let ring = mxx_dsl::Ring::new(256, 1);
        let ciphertext = ring.artifact_input(
            mxx_ir_core::artifact::ProductionId {
                spec_hash: mxx_ir_core::artifact::SpecHash([0; 32]),
                execution_nonce: [0; 32],
            },
            "ciphertext",
            if matches!(case, GeneratedScalarNegativeCase::BothScalar) { (1, 1) } else { (2, 2) },
            ArtifactConfidentiality::Public,
        );
        let decoded = ciphertext
            .threshold_decode_bools(IntExpr::constant(2), 1)
            .into_iter()
            .next()
            .expect("scalar-negative decoder")
            .semantic_anchor(crate::toy_example::DECODED_ENDPOINT)
            .expect("scalar-negative decoder anchor");
        let decrypt = mxx_dsl::DslContext::new("production-scalar-negative-decrypt")
            .int_parameter("cutoff")
            .bool_output("decoded", decoded)
            .expect("scalar-negative decoder output")
            .build()
            .expect("scalar-negative decoder graph");
        let decrypt_stage = protocol
            .bundle
            .workflow
            .stages
            .iter_mut()
            .find(|stage| stage.id == StageId("decrypt".to_owned()))
            .expect("scalar-negative decrypt stage");
        decrypt_stage.graph = decrypt.graph;
        decrypt_stage.semantic_anchors = decrypt.anchors;
        decrypt_stage.derivation_attachments = decrypt.derivation_attachments;
        protocol.bundle.input_contract.inputs.retain(|input| {
            input.id != ProtocolInputId::from("slice-source") &&
                input.id != ProtocolInputId::from("gather-source")
        });
        protocol.bundle.input_bindings.retain(|binding| {
            binding.input != ProtocolInputId::from("slice-source") &&
                binding.input != ProtocolInputId::from("gather-source")
        });
        protocol
    }

    fn generated_composite_binding_protocol() -> crate::ProtocolDecl {
        use crate::ProtocolInputId;
        use mxx_dsl::{DslContext, Parallel, Ring};
        let ring = Ring::new(256, 1);
        let outer = Parallel::range(1)
            .map_values(|_index| {
                let inner = Parallel::range(1)
                    .map_values(|_| ring.uniform_residue((1, 1)))
                    .expect("generated inner family");
                let picked = inner.get_static(IntExpr::Add(
                    Box::new(IntExpr::constant(0)),
                    Box::new(IntExpr::constant(0)),
                ));
                picked + ring.zero((1, 1))
            })
            .expect("generated composite-binding family");
        let residual = outer.get_static(0);
        let encrypt = DslContext::new("production-composite-binding")
            .int_parameter("cutoff")
            .public_output("ciphertext", residual.clone())
            .expect("ciphertext output")
            .private_output("operational-residual", residual)
            .expect("residual output")
            .build()
            .expect("composite-binding graph");
        let mut protocol = generated_gather_protocol(1);
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
        protocol.bundle.input_contract.inputs.retain(|input| {
            input.id != ProtocolInputId::from("slice-source") &&
                input.id != ProtocolInputId::from("gather-source")
        });
        protocol.bundle.input_bindings.retain(|binding| {
            binding.input != ProtocolInputId::from("slice-source") &&
                binding.input != ProtocolInputId::from("gather-source")
        });
        protocol
    }

    fn generated_deep_compact_protocol(depth: usize) -> crate::ProtocolDecl {
        use crate::ProtocolInputId;
        use mxx_ir_core::{
            Graph,
            artifact::ArtifactConfidentiality,
            graph::{GraphOutput, NodeHandle},
            node::NodeKind,
            types::{MatrixType, WireType},
        };
        let matrix = MatrixType {
            modulus: IntExpr::constant(256),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let (body, _body_output) = mxx_ir_core::graph::with_new_construction_scope(|scope| {
            let mut value = NodeHandle::new(
                NodeKind::UniformResidueSample { matrix_type: matrix.clone() },
                Vec::new(),
                vec![WireType::Matrix(matrix.clone())],
            )
            .output(0)
            .expect("deep sampler source");
            for _ in 0..depth {
                value = NodeHandle::new(
                    NodeKind::MatrixNegate,
                    vec![value],
                    vec![WireType::Matrix(matrix.clone())],
                )
                .output(0)
                .expect("deep negate node");
            }
            (
                mxx_ir_core::graph::SubgraphHandle::new(
                    format!("deep-compact-body-{depth}"),
                    scope,
                    Vec::new(),
                    vec![value.clone()],
                )
                .expect("deep compact body"),
                value,
            )
        });
        let loop_output = NodeHandle::parallel_loop(
            body,
            Vec::new(),
            vec![WireType::IndexedFamily {
                element: Box::new(WireType::Matrix(matrix.clone())),
                count: IntExpr::constant(1),
            }],
            mxx_ir_core::node::ParallelLoop {
                count: IntExpr::constant(1),
                minimum_count: 0,
                index_slot: 0,
                bindings: Vec::new(),
                input_modes: Vec::new(),
            },
        )
        .output(0)
        .expect("deep family loop");
        let residual = NodeHandle::new(
            NodeKind::FamilyGetStatic { index: IntExpr::constant(0) },
            vec![loop_output],
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .expect("deep family output");
        let graph = Graph::freeze(
            format!("production-deep-compact-{depth}"),
            vec![mxx_ir_core::CompileParameter {
                name: "cutoff".to_owned(),
                kind: mxx_ir_core::CompileParameterKind::Integer,
            }],
            BTreeMap::from([
                (
                    "ciphertext".to_owned(),
                    GraphOutput {
                        value: residual.clone(),
                        confidentiality: Some(ArtifactConfidentiality::Public),
                    },
                ),
                (
                    "operational-residual".to_owned(),
                    GraphOutput { value: residual, confidentiality: None },
                ),
            ]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .expect("deep compact graph freeze")
        .0;
        let mut protocol = generated_gather_protocol(1);
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
        protocol.bundle.input_contract.inputs.retain(|input| {
            input.id != ProtocolInputId::from("slice-source") &&
                input.id != ProtocolInputId::from("gather-source")
        });
        protocol.bundle.input_bindings.retain(|binding| {
            binding.input != ProtocolInputId::from("slice-source") &&
                binding.input != ProtocolInputId::from("gather-source")
        });
        protocol
    }

    fn generated_shared_virtual_dag_protocol_shape(
        depth: usize,
        forward_edge: bool,
    ) -> crate::ProtocolDecl {
        use crate::ProtocolInputId;
        use mxx_dsl::{DslContext, Parallel, Ring};
        let ring = Ring::new(256, 1);
        let family = Parallel::range(1)
            .map_values(|_| {
                let sampled = ring.uniform_residue((1, 1));
                let mut shared = -sampled;
                for _ in 0..depth {
                    shared = shared.clone() + shared;
                }
                if forward_edge {
                    let branch = shared.clone() + shared.clone();
                    branch + shared
                } else {
                    shared
                }
            })
            .expect("generated shared virtual DAG family");
        let residual = family.get_static(0);
        let graph = DslContext::new("production-shared-virtual-dag")
            .int_parameter("cutoff")
            .public_output("ciphertext", residual.clone())
            .expect("ciphertext output")
            .private_output("operational-residual", residual)
            .expect("residual output")
            .build()
            .expect("shared virtual DAG graph");
        let mut protocol = generated_gather_protocol(1);
        let encrypt_stage = protocol
            .bundle
            .workflow
            .stages
            .iter_mut()
            .find(|stage| stage.id == StageId("encrypt".to_owned()))
            .expect("encrypt stage");
        encrypt_stage.graph = graph.graph;
        encrypt_stage.semantic_anchors = graph.anchors;
        encrypt_stage.derivation_attachments = graph.derivation_attachments;
        protocol.bundle.input_contract.inputs.retain(|input| {
            input.id != ProtocolInputId::from("slice-source") &&
                input.id != ProtocolInputId::from("gather-source")
        });
        protocol.bundle.input_bindings.retain(|binding| {
            binding.input != ProtocolInputId::from("slice-source") &&
                binding.input != ProtocolInputId::from("gather-source")
        });
        protocol
    }

    fn generated_shared_virtual_dag_protocol_with_depth(depth: usize) -> crate::ProtocolDecl {
        generated_shared_virtual_dag_protocol_shape(depth, false)
    }

    fn generated_forward_edge_virtual_dag_protocol() -> crate::ProtocolDecl {
        generated_shared_virtual_dag_protocol_shape(0, true)
    }

    fn generated_shared_virtual_dag_protocol() -> crate::ProtocolDecl {
        generated_shared_virtual_dag_protocol_with_depth(1)
    }

    fn generated_slice_protocol(
        source_count: usize,
        slice_case: GeneratedSliceCase,
    ) -> crate::ProtocolDecl {
        use crate::{
            InputContractEntry, InputValueContract, ProtocolInputBinding, ProtocolInputDestination,
            ProtocolInputId, StageInputName,
        };
        use mxx_dsl::{DslContext, Int, Parallel, Ring};
        let ring = Ring::new(256, 1);
        let source = ring.input_family("gather-source", source_count, (1, 1));
        let slice_source_rows =
            if matches!(slice_case, GeneratedSliceCase::OutOfBounds) { 3 } else { 8 };
        let slice_source = ring.input("slice-source", (slice_source_rows, 1));
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
        let encrypt = DslContext::new("production-generated-gather")
            .int_parameter("cutoff")
            .public_output("ciphertext", residual.clone())
            .expect("ciphertext output")
            .private_output("operational-residual", residual)
            .expect("residual output")
            .build()
            .expect("generated gather graph");

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
        crate::ProtocolDecl::new(protocol).unwrap()
    }

    #[derive(Clone, Copy, Debug)]
    enum IndexedScalarCase {
        Valid,
        NestedSource,
        UnauthorizedGaussianSource,
        WrongBinding,
        OutOfRange,
    }

    #[derive(Clone, Copy, Debug)]
    enum IndexedScalarProductKind {
        Multiply,
        Tensor,
        TensorNonProgramScalar,
        TensorNonMultiplySibling,
        Standalone,
        Mixed,
    }

    fn generated_indexed_scalar_product_protocol(
        reverse: bool,
        shared_cancellation: bool,
        bounded_inputs: bool,
        case: IndexedScalarCase,
        product_kind: IndexedScalarProductKind,
    ) -> crate::ProtocolDecl {
        use crate::{
            InputContractEntry, InputValueContract, ProtocolInputBinding, ProtocolInputDestination,
            ProtocolInputId, StageInputName,
        };
        use mxx_dsl::{DslContext, Int, Parallel, Ring, SemanticAnchor};
        let ring = Ring::new(256, 1);
        let source = if bounded_inputs {
            ring.uniform_interval((1, 80), IntExpr::constant(-1), IntExpr::constant(1))
        } else {
            ring.input("indexed-scalar-source", (1, 80))
        };
        let large = if bounded_inputs {
            ring.uniform_interval((1, 80), IntExpr::constant(-1), IntExpr::constant(1))
        } else {
            ring.input("indexed-scalar-large", (1, 80))
        };
        let tensor_inputs = matches!(
            product_kind,
            IndexedScalarProductKind::Tensor |
                IndexedScalarProductKind::TensorNonProgramScalar |
                IndexedScalarProductKind::TensorNonMultiplySibling |
                IndexedScalarProductKind::Mixed
        )
        .then(|| {
            (
                if bounded_inputs {
                    ring.uniform_interval((1, 2), IntExpr::constant(-1), IntExpr::constant(1))
                } else {
                    ring.input("indexed-tensor-left-0", (1, 2))
                },
                if bounded_inputs {
                    ring.uniform_interval((2, 80), IntExpr::constant(-1), IntExpr::constant(1))
                } else {
                    ring.input("indexed-tensor-right-0", (2, 80))
                },
                if bounded_inputs {
                    ring.uniform_interval((1, 2), IntExpr::constant(-1), IntExpr::constant(1))
                } else {
                    ring.input("indexed-tensor-left-1", (1, 2))
                },
                if bounded_inputs {
                    ring.uniform_interval((2, 80), IntExpr::constant(-1), IntExpr::constant(1))
                } else {
                    ring.input("indexed-tensor-right-1", (2, 80))
                },
            )
        });
        let products = Parallel::range(4)
            .map_values(|index| {
                let scalar_family = Parallel::range(4)
                    .map_values(|inner_index| {
                        let start = inner_index.expression();
                        let (start, end) = if matches!(case, IndexedScalarCase::OutOfRange) {
                            (
                                IntExpr::Add(
                                    Box::new(start.clone()),
                                    Box::new(IntExpr::constant(80)),
                                ),
                                IntExpr::Add(Box::new(start), Box::new(IntExpr::constant(81))),
                            )
                        } else {
                            (
                                start.clone(),
                                IntExpr::Add(Box::new(start), Box::new(IntExpr::constant(1))),
                            )
                        };
                        let scalar_source = match case {
                            IndexedScalarCase::NestedSource => {
                                let nested_source_family = Parallel::range(4)
                                    .map_values(|_| source.clone())
                                    .expect("indexed scalar nested source family");
                                nested_source_family.get_static(0)
                            }
                            IndexedScalarCase::UnauthorizedGaussianSource => {
                                ring.gaussian((1, 80), 1, 1)
                            }
                            _ => source.clone(),
                        };
                        scalar_source
                            .clone()
                            .slice(None, Some(mxx_ir_core::node::IndexRange { start, end }))
                    })
                    .expect("indexed scalar family");
                let scalar_binding = if matches!(case, IndexedScalarCase::WrongBinding) {
                    Int::constant(0)
                } else {
                    index.as_int()
                };
                let scalar =
                    if matches!(product_kind, IndexedScalarProductKind::TensorNonProgramScalar) {
                        ring.uniform_interval((1, 1), IntExpr::constant(-1), IntExpr::constant(1))
                    } else {
                        scalar_family.get(scalar_binding)
                    };
                let mut tensor_product_id = 0_u8;
                let mut product = |scalar: mxx_dsl::Mat, large: mxx_dsl::Mat| {
                    let is_tensor = matches!(
                        product_kind,
                        IndexedScalarProductKind::Tensor |
                            IndexedScalarProductKind::TensorNonProgramScalar |
                            IndexedScalarProductKind::TensorNonMultiplySibling
                    ) || matches!(product_kind, IndexedScalarProductKind::Mixed) &&
                        tensor_product_id > 0;
                    tensor_product_id = tensor_product_id.saturating_add(1);
                    if matches!(product_kind, IndexedScalarProductKind::Standalone) {
                        scalar + ring.zero((1, 1))
                    } else if !is_tensor {
                        if reverse { large * scalar } else { scalar * large }
                    } else {
                        let id = tensor_product_id.saturating_sub(1);
                        let (tensor_left, tensor_right) = if id == 0 {
                            let (left, right, _, _) =
                                tensor_inputs.as_ref().expect("tensor inputs");
                            (left.clone(), right.clone())
                        } else {
                            let (_, _, left, right) =
                                tensor_inputs.as_ref().expect("tensor inputs");
                            (left.clone(), right.clone())
                        };
                        let other = tensor_left * tensor_right;
                        let other = if matches!(
                            product_kind,
                            IndexedScalarProductKind::TensorNonMultiplySibling
                        ) {
                            large.clone()
                        } else {
                            other
                        };
                        if reverse { other.tensor(scalar) } else { scalar.tensor(other) }
                    }
                };
                if shared_cancellation {
                    let first = product(scalar.clone(), large.clone());
                    let second_large = large.clone() - ring.zero((1, 80));
                    let second = product(scalar, second_large);
                    first - second
                } else {
                    product(scalar, large.clone())
                }
            })
            .expect("indexed scalar product family");
        let residual = products.get_static(0);
        let encrypt = DslContext::new("production-indexed-scalar-product")
            .int_parameter("cutoff")
            .public_output("ciphertext", residual.clone())
            .expect("ciphertext output")
            .private_output("operational-residual", residual)
            .expect("residual output")
            .build()
            .expect("indexed scalar graph");
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
        let ciphertext_shape = if matches!(product_kind, IndexedScalarProductKind::Standalone) {
            (IntExpr::constant(1), IntExpr::constant(1))
        } else if matches!(
            product_kind,
            IndexedScalarProductKind::Tensor |
                IndexedScalarProductKind::TensorNonProgramScalar |
                IndexedScalarProductKind::TensorNonMultiplySibling
        ) {
            (
                IntExpr::Mul(Box::new(IntExpr::constant(1)), Box::new(IntExpr::constant(1))),
                if reverse {
                    IntExpr::Mul(Box::new(IntExpr::constant(80)), Box::new(IntExpr::constant(1)))
                } else {
                    IntExpr::Mul(Box::new(IntExpr::constant(1)), Box::new(IntExpr::constant(80)))
                },
            )
        } else {
            (IntExpr::constant(1), IntExpr::constant(80))
        };
        let ciphertext = ring.artifact_input(
            mxx_ir_core::artifact::ProductionId {
                spec_hash: mxx_ir_core::artifact::SpecHash([0; 32]),
                execution_nonce: [0; 32],
            },
            "ciphertext",
            ciphertext_shape,
            mxx_ir_core::artifact::ArtifactConfidentiality::Public,
        );
        let decoded = ciphertext
            .threshold_decode_bools(IntExpr::constant(2), 1)
            .into_iter()
            .next()
            .expect("indexed scalar decoder output")
            .semantic_anchor(crate::toy_example::DECODED_ENDPOINT)
            .expect("indexed scalar decoder anchor");
        let decrypt = DslContext::new("production-indexed-scalar-product-decrypt")
            .int_parameter("cutoff")
            .bool_output("decoded", decoded)
            .expect("indexed scalar decoder graph")
            .build()
            .expect("indexed scalar decoder graph");
        let decrypt_stage = protocol
            .bundle
            .workflow
            .stages
            .iter_mut()
            .find(|stage| stage.id == StageId("decrypt".to_owned()))
            .expect("decrypt stage");
        decrypt_stage.graph = decrypt.graph;
        decrypt_stage.semantic_anchors = decrypt.anchors;
        decrypt_stage.derivation_attachments = decrypt.derivation_attachments;
        for binding in &mut protocol.bundle.input_bindings {
            if binding.input == ProtocolInputId::from("message") {
                binding.destinations.retain(|destination| {
                    matches!(destination, ProtocolInputDestination::Ideal { .. })
                });
            }
        }
        let matrix_contract = |rows: usize, columns: usize| InputValueContract::MatrixLarge {
            matrix_type: mxx_ir_core::types::MatrixType {
                modulus: IntExpr::constant(256),
                ring_dimension: IntExpr::constant(1),
                rows: IntExpr::constant(rows),
                columns: IntExpr::constant(columns),
            },
        };
        let mut external_contracts = Vec::new();
        if !matches!(product_kind, IndexedScalarProductKind::TensorNonProgramScalar) {
            external_contracts.push(("indexed-scalar-source", matrix_contract(1, 80)));
        }
        if matches!(
            product_kind,
            IndexedScalarProductKind::Multiply |
                IndexedScalarProductKind::TensorNonMultiplySibling |
                IndexedScalarProductKind::Mixed
        ) {
            external_contracts.push(("indexed-scalar-large", matrix_contract(1, 80)));
        }
        if matches!(
            product_kind,
            IndexedScalarProductKind::Tensor |
                IndexedScalarProductKind::TensorNonProgramScalar |
                IndexedScalarProductKind::Mixed
        ) {
            let tensor_id =
                if matches!(product_kind, IndexedScalarProductKind::Mixed) { 1 } else { 0 };
            external_contracts.extend([
                (
                    if tensor_id == 0 { "indexed-tensor-left-0" } else { "indexed-tensor-left-1" },
                    matrix_contract(1, 2),
                ),
                (
                    if tensor_id == 0 {
                        "indexed-tensor-right-0"
                    } else {
                        "indexed-tensor-right-1"
                    },
                    matrix_contract(2, 80),
                ),
            ]);
            if shared_cancellation && !matches!(product_kind, IndexedScalarProductKind::Mixed) {
                external_contracts.extend([
                    ("indexed-tensor-left-1", matrix_contract(1, 2)),
                    ("indexed-tensor-right-1", matrix_contract(2, 80)),
                ]);
            }
        }
        for (name, value) in
            (!bounded_inputs).then_some(external_contracts).into_iter().flatten().filter(
                |(name, _)| {
                    !matches!(case, IndexedScalarCase::UnauthorizedGaussianSource) ||
                        *name != "indexed-scalar-source"
                },
            )
        {
            let id = ProtocolInputId::from(name);
            protocol.bundle.input_contract.inputs.push(InputContractEntry {
                id: id.clone(),
                name: name.to_owned(),
                value,
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
    fn matrix_mul_accumulate_rejects_family_input_before_invalid_output_type() {
        let protocol = crate::toy_example::protocol();
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("toy plan");
        let mut adapter = ProductionAdapter::new(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .expect("production adapter");
        let family = adapter
            .job
            .with_arena_stores(|expressions, programs, _| {
                let argument = expressions.intern_argument(0, ResolvedValueType::Int)?;
                programs.generated_family_from_body(expressions, FamilyDomain::new(0, 1)?, argument)
            })
            .expect("generated family");
        let wire = plan.target().residual.clone();
        let invalid_output = WireType::TypedBlob {
            type_name: "unsupported-output".to_owned(),
            schema_hash: [0; 32],
        };

        let error = adapter
            .lower_node(
                &wire,
                &NodeKind::MatrixMulAccumulate {
                    coefficients: vec![IntExpr::constant(1)],
                    has_bias: false,
                },
                &invalid_output,
                &[Value::Family(family)],
            )
            .expect_err("family input must win the dual failure");
        assert!(matches!(
            error,
            ProductionAdapterError::UnsupportedNode { ref kind, .. }
                if kind == "family multi-row GEMM input"
        ));
    }

    #[test]
    fn matrix_observation_materializes_wrapped_calls_but_preserves_fact_authority() {
        let protocol = crate::toy_example::protocol();
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("toy plan");
        let mut adapter = ProductionAdapter::new(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .expect("production adapter");
        let matrix = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap();
        let value = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(991),
                    operation: SamplerOperation::UniformResidue { output: matrix.clone() },
                },
                Box::new([]),
            )
            .unwrap();
        let mut value_facts =
            MatrixFacts::new(matrix.clone(), MatrixMetadata::new(MatrixLayout::row_major(1, 1)));
        value_facts.coefficient_bound = NumericContract::Known(CoefficientBound::finite(3_u8));
        adapter.job.insert_matrix_facts(adapter.token, value, value_facts.clone()).unwrap();
        assert_eq!(
            adapter.project_matrix_fact_owner(value).unwrap(),
            MatrixFactProjection::Found(value)
        );
        let domain = FamilyDomain::new(0, 1).unwrap();
        let family = adapter.generated_family(domain, value).unwrap();
        let index = adapter.intern_index_constant(BigInt::ZERO).unwrap();
        let compact = adapter
            .call_family_in_program_scope_deferred_generated(
                family,
                index,
                TrustedIndexRange::new(0, 1).unwrap(),
            )
            .unwrap();
        assert_eq!(
            adapter.project_matrix_fact_owner(compact).unwrap(),
            MatrixFactProjection::Found(value)
        );
        let wrapped = adapter
            .job
            .expressions_mut()
            .intern_matrix_transform(MatrixOperation::Add, &[compact, compact])
            .unwrap();
        let expected = adapter
            .job
            .expressions_mut()
            .intern_matrix_transform(MatrixOperation::Add, &[value, value])
            .unwrap();
        assert_eq!(
            adapter.project_matrix_fact_owner(wrapped).unwrap(),
            MatrixFactProjection::Unknown
        );
        assert_eq!(adapter.gadget_fact_owner(wrapped).unwrap(), Some(expected));
        assert_eq!(adapter.matrix_fact_projection_fallbacks, 1);
        assert_eq!(adapter.authoritative_matrix_observation_view(wrapped).unwrap(), expected);

        let selector = adapter.intern_index_constant(BigInt::ZERO).unwrap();
        let explicit = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::ExplicitElement {
                    domain,
                    element_type: ResolvedValueType::Matrix(matrix),
                },
                Box::new([selector, compact]),
            )
            .unwrap();
        adapter.job.insert_matrix_facts(adapter.token, explicit, value_facts).unwrap();
        assert_ne!(adapter.job.materialize_reducible_generated_calls(explicit).unwrap(), explicit);
        assert_eq!(adapter.authoritative_matrix_observation_view(explicit).unwrap(), explicit);
    }

    #[test]
    fn matrix_select_skips_observation_for_open_selector_and_materializes_closed_branches() {
        let protocol = crate::toy_example::protocol();
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("toy plan");
        let mut adapter = ProductionAdapter::new(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .expect("production adapter");
        adapter.job.programs_mut().enable_diagnostic_counters();
        let matrix = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap();
        let value = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(993),
                    operation: SamplerOperation::UniformResidue { output: matrix.clone() },
                },
                Box::new([]),
            )
            .unwrap();
        let mut value_facts =
            MatrixFacts::new(matrix.clone(), MatrixMetadata::new(MatrixLayout::row_major(1, 1)));
        value_facts.coefficient_bound = NumericContract::Known(CoefficientBound::finite(3_u8));
        adapter.job.insert_matrix_facts(adapter.token, value, value_facts.clone()).unwrap();
        let family = adapter.generated_family(FamilyDomain::new(0, 1).unwrap(), value).unwrap();
        let index = adapter.intern_index_constant(BigInt::ZERO).unwrap();
        let branch = adapter
            .call_family_in_program_scope_deferred_generated(
                family,
                index,
                TrustedIndexRange::new(0, 1).unwrap(),
            )
            .unwrap();
        let selector =
            adapter.job.expressions_mut().intern_argument(0, ResolvedValueType::Int).unwrap();
        let output = WireType::Matrix(mxx_ir_core::types::MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(4),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        });
        let wire = plan.target().residual.clone();
        let beta_before = adapter.job.programs().diagnostic_counters().beta_nodes_visited;
        let sidecars_before =
            adapter.job.programs().diagnostic_counters().beta_program_call_sidecar_entries;
        let nodes_before = adapter.job.expressions().node_count();
        let Value::Expr(open_selected) = adapter
            .lower_node(
                &wire,
                &NodeKind::Select { count: IntExpr::constant(2) },
                &output,
                &[Value::Expr(selector), Value::Expr(branch), Value::Expr(branch)],
            )
            .unwrap()
        else {
            panic!("matrix select must produce an expression")
        };
        assert_eq!(adapter.job.programs().diagnostic_counters().beta_nodes_visited, beta_before);
        assert_eq!(
            adapter.job.programs().diagnostic_counters().beta_program_call_sidecar_entries,
            sidecars_before
        );
        assert_eq!(adapter.job.expressions().node_count(), nodes_before + 1);
        assert!(adapter.job.facts().facts(open_selected).is_err());
        let open_node = adapter.job.expressions().node(open_selected).unwrap();
        assert_eq!(open_node.inputs.as_ref(), &[selector, branch, branch]);
        assert_eq!(adapter.matrix_select_open_observation_skips, 1);
        assert_eq!(adapter.matrix_select_open_observation_skipped_branches, 2);

        let eager = adapter.job.materialize_reducible_generated_calls(open_selected).unwrap();
        let eager_expected = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::ExplicitElement {
                    domain: FamilyDomain::new(0, 2).unwrap(),
                    element_type: ResolvedValueType::Matrix(matrix.clone()),
                },
                Box::new([selector, value, value]),
            )
            .unwrap();
        assert_eq!(eager, eager_expected);
        assert_eq!(
            adapter.job.expressions().value_type(eager).unwrap(),
            adapter.job.expressions().value_type(open_selected).unwrap()
        );

        let closed_selector = adapter.intern_index_constant(BigInt::ZERO).unwrap();
        let early_sidecar_hits_before_closed =
            adapter.job.programs().diagnostic_counters().beta_program_call_early_source_hits;
        let Value::Expr(closed_selected) = adapter
            .lower_node(
                &wire,
                &NodeKind::Select { count: IntExpr::constant(2) },
                &output,
                &[Value::Expr(closed_selector), Value::Expr(branch), Value::Expr(branch)],
            )
            .unwrap()
        else {
            panic!("matrix select must produce an expression")
        };
        assert!(
            adapter.job.programs().diagnostic_counters().beta_program_call_early_source_hits >
                early_sidecar_hits_before_closed
        );
        assert_eq!(
            adapter.job.facts().facts(closed_selected).unwrap(),
            &super::super::facts::ValueFacts::Matrix(value_facts)
        );
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

        let domain = FamilyDomain::new(0, 1).unwrap();
        let zero_family = adapter.generated_family(domain, zero).unwrap();
        let index = adapter.intern_index_constant(BigInt::ZERO).unwrap();
        let compact_zero = adapter
            .call_family_in_program_scope_deferred_generated(
                zero_family,
                index,
                TrustedIndexRange::new(0, 1).unwrap(),
            )
            .unwrap();
        let compact_decomposition = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
                    output: ResolvedMatrixType::new(BigUint::from(17_u8), 4, 3, 1).unwrap(),
                    base: 8,
                    small: true,
                    digit_count: 3,
                }),
                Box::new([compact_zero]),
            )
            .unwrap();
        adapter
            .register_gadget_decomposition_contract(compact_decomposition, compact_zero, &wire)
            .unwrap();
        assert_eq!(adapter.job.gadget_recompositions().rule_count(), 2);

        let opaque_family = adapter.opaque_generated_family(domain, zero).unwrap();
        let opaque_call = adapter
            .call_family_in_program_scope_deferred_generated(
                opaque_family,
                index,
                TrustedIndexRange::new(0, 1).unwrap(),
            )
            .unwrap();
        let opaque_decomposition = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
                    output: ResolvedMatrixType::new(BigUint::from(17_u8), 4, 3, 1).unwrap(),
                    base: 8,
                    small: true,
                    digit_count: 3,
                }),
                Box::new([opaque_call]),
            )
            .unwrap();
        let fallbacks_before = adapter.matrix_fact_projection_fallbacks;
        adapter
            .register_gadget_decomposition_contract(opaque_decomposition, opaque_call, &wire)
            .unwrap();
        assert_eq!(adapter.job.gadget_recompositions().rule_count(), 2);
        assert_eq!(adapter.matrix_fact_projection_fallbacks, fallbacks_before + 1);
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
    fn compact_phase_a_production_pipeline_matches_forced_legacy() {
        // Exercise the same adapter preflight, relation freeze, closed-root transport, job
        // normalizer, and report bridge used by production.  The tiny generated family is
        // deliberately an Add/Subtract/Negate-only island so no Phase-B operation is involved.
        fn build(
            protocol: &crate::ProtocolDecl,
            plan: &ProtocolPlan,
            force_legacy: bool,
        ) -> (CheckerJob, ProductionRoots, u64) {
            let mut adapter = ProductionAdapter::new(protocol, plan, BTreeMap::new())
                .expect("production adapter");
            let matrix = ResolvedMatrixType::new(257_u16.into(), 1, 1, 1).unwrap();
            let source = |adapter: &mut ProductionAdapter<'_>, event| {
                adapter
                    .job
                    .expressions_mut()
                    .intern(
                        ValueOperator::Sampler {
                            event: SampleEventId(event),
                            operation: SamplerOperation::UniformResidue { output: matrix.clone() },
                        },
                        Box::new([]),
                    )
                    .unwrap()
            };
            let left = source(&mut adapter, 70);
            let right = source(&mut adapter, 71);
            let sum = adapter
                .job
                .expressions_mut()
                .intern(ValueOperator::Matrix(MatrixOperation::Add), [left, right].into())
                .unwrap();
            let negated = adapter
                .job
                .expressions_mut()
                .intern(ValueOperator::Matrix(MatrixOperation::Negate), [sum].into())
                .unwrap();
            let reverse_sum = adapter
                .job
                .expressions_mut()
                .intern(ValueOperator::Matrix(MatrixOperation::Add), [right, left].into())
                .unwrap();
            let reverse_negated = adapter
                .job
                .expressions_mut()
                .intern(ValueOperator::Matrix(MatrixOperation::Negate), [reverse_sum].into())
                .unwrap();
            let body = adapter
                .job
                .expressions_mut()
                .intern(
                    ValueOperator::Matrix(MatrixOperation::Subtract),
                    [negated, reverse_negated].into(),
                )
                .unwrap();
            let family = adapter.generated_family(FamilyDomain::new(0, 1).unwrap(), body).unwrap();
            let index = adapter.intern_index_constant(BigInt::ZERO).unwrap();
            let range = TrustedIndexRange::new(0, 1).unwrap();
            let root = adapter
                .call_family_in_program_scope_deferred_generated(family, index, range)
                .unwrap();
            assert!(adapter.compact_residual_preflight(&Value::Expr(root)).is_none());
            let beta_before = adapter.job.programs().diagnostic_counters().beta_nodes_visited;
            adapter.register_reached_relations().unwrap();
            let root = if force_legacy {
                let Value::Expr(root) = adapter
                    .materialize_root_value_with_reason(Value::Expr(root), BetaReason::ResidualRoot)
                    .unwrap()
                else {
                    panic!("generated root must remain an expression")
                };
                root
            } else {
                root
            };
            let compact = !force_legacy;
            let beta_after_residual =
                adapter.job.programs().diagnostic_counters().beta_nodes_visited;
            let decoder = adapter
                .materialize_root_value_with_reason(Value::Expr(root), BetaReason::DecoderRoot)
                .unwrap();
            let Value::Expr(decoder) = decoder else {
                panic!("generated decoder root must remain an expression")
            };
            let closed = adapter
                .close_expression(&plan.target().residual, root, "compact phase-a test root")
                .unwrap();
            let decoder_closed = adapter
                .close_expression(&plan.target().decoder, decoder, "compact phase-a test decoder")
                .unwrap();
            adapter.job.freeze_relations(adapter.token).unwrap();
            let production_root = if compact {
                ProductionRoot::Compact(closed)
            } else {
                ProductionRoot::Closed(closed)
            };
            (
                adapter.job,
                ProductionRoots {
                    residual: production_root,
                    decoder: ProductionRoot::Closed(decoder_closed),
                    occurrences: 0,
                    samples: 2,
                },
                beta_after_residual.saturating_sub(beta_before),
            )
        }

        let protocol = generated_gather_protocol(1);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("protocol plan");
        let (mut compact_job, compact_roots, compact_beta) = build(&protocol, &plan, false);
        assert!(matches!(compact_roots.residual, ProductionRoot::Compact(_)));
        let ProductionRoot::Compact(compact_root) = compact_roots.residual else { unreachable!() };
        let compact_nodes_before = compact_job.expressions().node_count();
        let compact_analysis = compact_job
            .normalize_compact_closed_root(compact_root)
            .expect("compact root normalization");
        let compact_nodes_after = compact_job.expressions().node_count();
        assert!(compact_analysis.counters.compact_virtual_calls > 0);
        assert!(compact_analysis.counters.compact_max_frames > 0);
        assert_eq!(compact_analysis.counters.compact_live_frames, 0);
        assert_eq!(compact_analysis.counters.compact_live_values, 0);
        assert_eq!(compact_analysis.counters.compact_memo_entries, 0);
        assert_eq!(compact_analysis.counters.compact_peak_memo_entries, 0);
        assert!(compact_nodes_after.saturating_sub(compact_nodes_before) <= 32);
        assert_eq!(compact_beta, 0, "compact residual must not beta-materialize");

        let (mut legacy_job, legacy_roots, _legacy_beta) = build(&protocol, &plan, true);
        assert!(matches!(legacy_roots.residual, ProductionRoot::Closed(_)));
        let ProductionRoot::Closed(legacy_root) = legacy_roots.residual else { unreachable!() };
        let legacy_analysis =
            legacy_job.normalize_closed_root(legacy_root).expect("legacy root normalization");
        assert_eq!(
            compact_analysis.exact_term_diagnostics, legacy_analysis.exact_term_diagnostics,
            "complete PolynomialNF diagnostics must match"
        );
        assert_eq!(
            compact_analysis.value.coefficient_bound,
            legacy_analysis.value.coefficient_bound
        );
        assert_eq!(
            compact_analysis.counters.relation_applied,
            legacy_analysis.counters.relation_applied
        );
        assert_eq!(
            compact_analysis.counters.relation_remaining,
            legacy_analysis.counters.relation_remaining
        );

        let target = super::super::report::ReportTarget {
            target_id: "compact-phase-a".to_owned(),
            plaintext_modulus: 2_u8.into(),
            ciphertext_modulus: 257_u16.into(),
            boolean_interval: false,
        };
        let compact_report =
            super::super::report::analyze_roots(&mut compact_job, &compact_roots, &target)
                .expect("compact report");
        let legacy_report =
            super::super::report::analyze_roots(&mut legacy_job, &legacy_roots, &target)
                .expect("legacy report");
        assert_eq!(compact_report.accepted, legacy_report.accepted);
        assert_eq!(compact_report.residual, legacy_report.residual);
        assert_eq!(compact_report.residual.bound, legacy_report.residual.bound);
        assert_eq!(
            compact_report.counters.normalization.relation_applied,
            legacy_report.counters.normalization.relation_applied
        );
        assert_eq!(
            compact_report.counters.normalization.relation_remaining,
            legacy_report.counters.normalization.relation_remaining
        );
    }

    #[test]
    fn production_compact_family_marker_matches_ordinary_family_analysis() {
        fn build(
            protocol: &crate::ProtocolDecl,
            plan: &ProtocolPlan,
            compact: bool,
        ) -> (CheckerJob, ProductionRoot, CompactShellPlan) {
            let mut adapter = ProductionAdapter::new(protocol, plan, BTreeMap::new())
                .expect("production adapter");
            let matrix = ResolvedMatrixType::new(257_u16.into(), 1, 1, 1).unwrap();
            let source = |adapter: &mut ProductionAdapter<'_>, event| {
                adapter
                    .job
                    .expressions_mut()
                    .intern(
                        ValueOperator::Sampler {
                            event: SampleEventId(event),
                            operation: SamplerOperation::UniformResidue { output: matrix.clone() },
                        },
                        Box::new([]),
                    )
                    .unwrap()
            };
            let left = source(&mut adapter, 401);
            let right = source(&mut adapter, 402);
            let nested_body = source(&mut adapter, 403);
            let domain = FamilyDomain::new(0, 4).unwrap();
            let nested_family = adapter.generated_family(domain, nested_body).unwrap();
            let argument =
                adapter.job.expressions_mut().intern_argument(0, ResolvedValueType::Int).unwrap();
            let nested_call = adapter
                .job
                .call_family_in_program_scope_deferred_generated(
                    nested_family,
                    argument,
                    TrustedIndexRange::new(0, 4).unwrap(),
                )
                .unwrap();
            let sum = adapter
                .job
                .expressions_mut()
                .intern(ValueOperator::Matrix(MatrixOperation::Add), [left, right].into())
                .unwrap();
            let body = adapter
                .job
                .expressions_mut()
                .intern(ValueOperator::Matrix(MatrixOperation::Add), [sum, nested_call].into())
                .unwrap();
            let family = adapter.generated_family(domain, body).unwrap();
            let residual = Value::Family(family);
            assert_eq!(adapter.compact_residual_preflight(&residual), None);
            let shell_plan = adapter.build_compact_shell_plan(&residual).unwrap();
            assert!(shell_plan.preflight_node_occurrences > 0);
            adapter.register_reached_relations().unwrap();
            if compact {
                adapter.job.set_compact_shell_plan(shell_plan.clone()).unwrap();
            }
            adapter.job.freeze_relations(adapter.token).unwrap();
            let root = adapter
                .close_root(
                    residual,
                    &plan.target().residual,
                    "compact family production root",
                    compact,
                )
                .unwrap();
            (adapter.job, root, shell_plan)
        }

        let protocol = generated_gather_protocol(1);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("protocol plan");
        let (mut compact_job, compact_root, shell_plan) = build(&protocol, &plan, true);
        let ProductionRoot::CompactFamily(compact_family) = compact_root else {
            panic!("eligible indexed family must use the private compact marker")
        };
        let compact = compact_job
            .analyze_compact_family_root(compact_family)
            .expect("compact family analysis");
        assert!(compact.counters.nodes_processed > 0);
        assert_eq!(compact.counters.nodes_total, compact.counters.nodes_processed);
        assert!(shell_plan.preflight_node_occurrences > 0);

        let (mut eager_job, eager_root, _) = build(&protocol, &plan, false);
        let ProductionRoot::Family(eager_family) = eager_root else {
            panic!("ordinary family must retain the ordinary marker")
        };
        let eager = eager_job.analyze_family_root(eager_family).expect("ordinary family analysis");
        assert_eq!(compact.exact_term_diagnostics, eager.exact_term_diagnostics);
        assert_eq!(compact.bounded_summary, eager.bounded_summary);
        assert_eq!(compact.counters.relation_applied, eager.counters.relation_applied);
        assert_eq!(compact.counters.relation_remaining, eager.counters.relation_remaining);
        assert_eq!(shell_plan.scalar_occurrences(), 0);
    }

    #[test]
    fn production_lower_routes_indexed_family_through_compact_report() {
        let protocol = generated_indexed_family_protocol(4, false, false, 64);
        let plan =
            ProtocolPlan::build(&protocol, "indexed-family-production").expect("protocol plan");
        let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("indexed-family production adapter");
        let (mut job, roots) = adapter.lower().expect("indexed-family production lowering");
        assert!(matches!(roots.residual, ProductionRoot::CompactFamily(_)));
        let target = super::super::report::ReportTarget {
            target_id: "indexed-family-production".to_owned(),
            plaintext_modulus: 2_u8.into(),
            ciphertext_modulus: 257_u16.into(),
            boolean_interval: false,
        };
        let compact_report = super::super::report::analyze_roots(&mut job, &roots, &target)
            .expect("indexed-family compact report");
        assert!(compact_report.counters.normalization.nodes_processed > 0);
        assert_eq!(
            compact_report.counters.normalization.nodes_total,
            compact_report.counters.normalization.nodes_processed
        );
        let eager_adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("indexed-family eager adapter");
        let (mut eager_job, eager_roots) =
            eager_adapter.lower_force_eager().expect("indexed-family eager lowering");
        assert!(matches!(eager_roots.residual, ProductionRoot::Family(_)));
        let eager_report =
            super::super::report::analyze_roots(&mut eager_job, &eager_roots, &target)
                .expect("indexed-family eager report");
        assert_eq!(compact_report.residual, eager_report.residual);
        assert_eq!(compact_report.accepted, eager_report.accepted);
        assert_eq!(
            compact_report.counters.normalization.relation_applied,
            eager_report.counters.normalization.relation_applied
        );
        assert_eq!(
            compact_report.counters.normalization.relation_remaining,
            eager_report.counters.normalization.relation_remaining
        );
    }

    #[test]
    fn production_extract_coefficient_binding_compact_matches_eager() {
        let protocol = generated_extract_binding_family_protocol();
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("extract-binding plan");
        let mut preflight = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("extract-binding preflight adapter");
        let residual = preflight
            .resolve(plan.target().residual.clone(), Rc::new(BTreeMap::new()))
            .expect("extract-binding residual");
        let compiled =
            preflight.build_compact_shell_plan(&residual).expect("extract-binding compact plan");
        assert!(compiled.preflight_node_occurrences > 0);
        assert!(!preflight.job.relations().is_frozen());

        let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("extract-binding compact adapter");
        let (mut compact_job, compact_roots) = adapter.lower().expect("extract-binding lowering");
        let ProductionRoot::CompactFamily(compact_family) = compact_roots.residual else {
            panic!("exact ExtractCoefficient binding must retain CompactFamily")
        };
        let before = compact_job.programs().diagnostic_counters();
        let (compact_value, compact_counters) = compact_job
            .normalize_compact_family(compact_family)
            .expect("extract-binding compact analysis");
        let after = compact_job.programs().diagnostic_counters();
        assert_eq!(after.beta_nodes_visited, before.beta_nodes_visited);
        assert_eq!(after.beta_reason_misses, before.beta_reason_misses);
        assert_eq!(after.beta_reason_expr_allocations, before.beta_reason_expr_allocations);
        assert_eq!(compact_counters.compact_memo_entries, 0);
        assert_eq!(compact_counters.compact_peak_memo_entries, 0);
        assert_eq!(compact_counters.compact_scalar_holds_unmatched, 0);
        assert_eq!(compact_counters.compact_shell_holds_unmatched, 0);

        let eager_adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("extract-binding eager adapter");
        let (mut eager_job, eager_roots) =
            eager_adapter.lower_force_eager().expect("extract-binding eager lowering");
        assert!(matches!(eager_roots.residual, ProductionRoot::Family(_)));
        let ProductionRoot::Family(eager_family) = eager_roots.residual else {
            panic!("eager ExtractCoefficient binding must retain Family")
        };
        let eager_root = eager_job
            .programs()
            .root(&eager_job.expressions(), eager_family.program())
            .expect("extract-binding eager family root");
        let eager_materialized = eager_job
            .materialize_reducible_generated_calls(eager_root.expression())
            .expect("extract-binding eager materialization");
        let eager_scoped = eager_job
            .programs()
            .detached_scoped(&eager_job.expressions(), eager_family.program(), eager_materialized)
            .expect("extract-binding eager scope");
        let (eager_value, eager_counters) =
            eager_job.normalize(eager_scoped).expect("extract-binding eager analysis");
        assert_eq!(
            compact_value.exact_nf.as_ref().map(|nf| nf.term_count()),
            eager_value.exact_nf.as_ref().map(|nf| nf.term_count())
        );
        assert_eq!(
            full_nf_descriptor_map(&compact_job, &compact_value),
            full_nf_descriptor_map(&eager_job, &eager_value)
        );
        assert_eq!(compact_counters.relation_applied, eager_counters.relation_applied);
        assert_eq!(compact_counters.relation_remaining, eager_counters.relation_remaining);
    }

    #[test]
    fn production_extract_binding_nested_matrix_body_negative_paths_precede_freeze() {
        #[derive(Clone, Copy)]
        enum InvalidBody {
            Factful,
            RelationEndpoint,
            Unsupported,
            ScalarPlan,
        }

        fn build(
            adapter: &mut ProductionAdapter<'_>,
            invalid: InvalidBody,
            wire: PlannedWire,
            inject_relation_endpoint: bool,
        ) -> Value {
            let matrix = ResolvedMatrixType::new(BigUint::from(256_u16), 1, 1, 1).unwrap();
            let wide_matrix = ResolvedMatrixType::new(BigUint::from(256_u16), 1, 1, 2).unwrap();
            let source = |adapter: &mut ProductionAdapter<'_>,
                          event,
                          output: &ResolvedMatrixType| {
                adapter
                    .job
                    .expressions_mut()
                    .intern(
                        ValueOperator::Sampler {
                            event: SampleEventId(event),
                            operation: SamplerOperation::UniformResidue { output: output.clone() },
                        },
                        Box::new([]),
                    )
                    .unwrap()
            };
            let first = source(adapter, 95_001, &matrix);
            let inner_body = if matches!(invalid, InvalidBody::Unsupported) {
                adapter
                    .job
                    .expressions_mut()
                    .intern(ValueOperator::Matrix(MatrixOperation::Transpose), [first].into())
                    .unwrap()
            } else if matches!(invalid, InvalidBody::ScalarPlan) {
                let second = source(adapter, 95_002, &wide_matrix);
                adapter
                    .job
                    .expressions_mut()
                    .intern(
                        ValueOperator::Matrix(MatrixOperation::Multiply),
                        [first, second].into(),
                    )
                    .unwrap()
            } else {
                let second = source(adapter, 95_002, &matrix);
                adapter
                    .job
                    .expressions_mut()
                    .intern(ValueOperator::Matrix(MatrixOperation::Add), [first, second].into())
                    .unwrap()
            };
            if matches!(invalid, InvalidBody::Factful) {
                adapter
                    .job
                    .insert_matrix_facts(
                        adapter.token,
                        inner_body,
                        MatrixFacts::new(
                            matrix.clone(),
                            MatrixMetadata::new(MatrixLayout::row_major(1, 1)),
                        ),
                    )
                    .unwrap();
            }
            let inner_family =
                adapter.generated_family(FamilyDomain::new(0, 256).unwrap(), inner_body).unwrap();
            let argument =
                adapter.job.expressions_mut().intern_argument(0, ResolvedValueType::Int).unwrap();
            let inner_call = adapter
                .call_family_in_program_scope_deferred_generated(
                    inner_family,
                    argument,
                    TrustedIndexRange::new(0, 4).unwrap(),
                )
                .unwrap();
            if inject_relation_endpoint {
                adapter.push_relation_candidate(RelationCandidate {
                    preimage: inner_body,
                    public: inner_body,
                    trapdoor: inner_body,
                    target: inner_body,
                    family_operands: None,
                    wire,
                });
            }
            let binding = adapter
                .job
                .expressions_mut()
                .intern(
                    ValueOperator::ExtractCoefficient {
                        position: 0,
                        canonical_input_exclusive_upper: Some(BigUint::from(256_u16)),
                    },
                    [inner_call].into(),
                )
                .unwrap();
            // Keep the scalar-plan body exclusively under the ExtractCoefficient binding.  The
            // consumer call has its own closed body; otherwise the same inner family body would
            // also be visited as an ordinary product body before the binding occurrence, which
            // would test a different (unbound) compact-factor rejection.
            let consumer_body = source(
                adapter,
                95_004,
                if matches!(invalid, InvalidBody::ScalarPlan) { &wide_matrix } else { &matrix },
            );
            let consumer_family = adapter
                .generated_family(FamilyDomain::new(0, 256).unwrap(), consumer_body)
                .unwrap();
            let outer_call = adapter
                .call_family_in_program_scope_deferred_generated(
                    consumer_family,
                    binding,
                    TrustedIndexRange::new(0, 256).unwrap(),
                )
                .unwrap();
            let sibling = source(
                adapter,
                95_003,
                if matches!(invalid, InvalidBody::ScalarPlan) { &wide_matrix } else { &matrix },
            );
            let outer_body = adapter
                .job
                .expressions_mut()
                .intern(ValueOperator::Matrix(MatrixOperation::Add), [outer_call, sibling].into())
                .unwrap();
            Value::Family(
                adapter.generated_family(FamilyDomain::new(0, 4).unwrap(), outer_body).unwrap(),
            )
        }

        let protocol = generated_gather_protocol(1);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("nested binding plan");
        for (invalid, expected) in [
            (InvalidBody::Factful, "virtual node has facts"),
            (InvalidBody::RelationEndpoint, "virtual node is a relation endpoint"),
            (InvalidBody::Unsupported, "generated body contains unsupported concrete operator"),
            (InvalidBody::ScalarPlan, "compact binding subtree contains a scalar plan occurrence"),
        ] {
            let mut adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
                .expect("nested binding adapter");
            let residual = build(
                &mut adapter,
                invalid,
                plan.target().residual.clone(),
                matches!(invalid, InvalidBody::RelationEndpoint),
            );
            assert_eq!(adapter.compact_residual_preflight(&residual).as_deref(), Some(expected));
            assert!(!adapter.job.relations().is_frozen());
            let mut eager = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
                .expect("nested binding eager adapter");
            // The eager copy deliberately omits the synthetic endpoint marker. This isolates the
            // preflight rejection proof from the ordinary operator graph used for parity.
            let eager_residual = build(&mut eager, invalid, plan.target().residual.clone(), false);
            let Value::Family(eager_family) = eager_residual else {
                panic!("nested binding eager residual must remain a family")
            };
            // Eager generated-call exposure is deliberately completed before the resource
            // snapshot taken by relation freezing.
            let eager_root = eager
                .job
                .programs()
                .root(&eager.job.expressions(), eager_family.program())
                .expect("nested binding eager family root");
            let eager_materialized = eager
                .job
                .materialize_reducible_generated_calls(eager_root.expression())
                .expect("nested binding eager materialization");
            let eager_scoped = eager
                .job
                .programs()
                .detached_scoped(
                    &eager.job.expressions(),
                    eager_family.program(),
                    eager_materialized,
                )
                .expect("nested binding eager scope");
            eager.register_reached_relations().expect("nested binding eager relations");
            eager.job.freeze_relations(eager.token).expect("nested binding eager freeze");
            let (eager_value, _) =
                eager.job.normalize(eager_scoped).expect("nested binding eager normalization");
            assert!(eager_value.exact_nf.as_ref().is_some_and(|nf| nf.term_count() > 0));
            let eager_descriptors = full_nf_descriptor_map(&eager.job, &eager_value);
            assert!(matches!(eager_descriptors.2, NumericContract::Known(_)));
        }
    }

    #[test]
    fn production_extract_binding_gadget_body_rejects_plan_before_freeze() {
        fn build(adapter: &mut ProductionAdapter<'_>) -> Value {
            let residual = adapter
                .resolve(adapter.plan.target().residual.clone(), Rc::new(BTreeMap::new()))
                .expect("gadget binding source residual");
            let Value::Expr(root) = residual else { unreachable!() };
            let ValueOperator::ProgramCall { program } =
                adapter.job.expressions().node(root).unwrap().operator
            else {
                unreachable!()
            };
            let family = adapter.job.programs().family_for_program(program).unwrap();
            let body = adapter.job.programs().family_body(family).unwrap();
            let product = adapter.job.expressions().node(body).unwrap().clone();
            let [gadget, _decomposition] = product.inputs.as_ref() else { unreachable!() };
            let inner_family =
                adapter.generated_family(FamilyDomain::new(0, 256).unwrap(), body).unwrap();
            let argument =
                adapter.job.expressions_mut().intern_argument(0, ResolvedValueType::Int).unwrap();
            let inner_call = adapter
                .call_family_in_program_scope_deferred_generated(
                    inner_family,
                    argument,
                    TrustedIndexRange::new(0, 256).unwrap(),
                )
                .unwrap();
            let binding = adapter
                .job
                .expressions_mut()
                .intern(
                    ValueOperator::ExtractCoefficient {
                        position: 0,
                        canonical_input_exclusive_upper: Some(BigUint::from(256_u16)),
                    },
                    [inner_call].into(),
                )
                .unwrap();
            let consumer_family =
                adapter.generated_family(FamilyDomain::new(0, 256).unwrap(), *gadget).unwrap();
            let consumer_call = adapter
                .call_family_in_program_scope_deferred_generated(
                    consumer_family,
                    binding,
                    TrustedIndexRange::new(0, 256).unwrap(),
                )
                .unwrap();
            let outer_body = adapter
                .job
                .expressions_mut()
                .intern(
                    ValueOperator::Matrix(MatrixOperation::Add),
                    [consumer_call, *gadget].into(),
                )
                .unwrap();
            let outer_family =
                adapter.generated_family(FamilyDomain::new(0, 1).unwrap(), outer_body).unwrap();
            Value::Family(outer_family)
        }

        let protocol = generated_gadget_product_protocol(false, GeneratedGadgetProductCase::Single);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").unwrap();
        let mut adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
        let value = build(&mut adapter);
        assert_eq!(
            adapter.compact_residual_preflight(&value).as_deref(),
            Some("compact binding subtree contains a gadget plan occurrence")
        );
        assert!(!adapter.job.relations().is_frozen());

        let mut eager = ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
        let eager_value = build(&mut eager);
        let Value::Family(eager_family) = eager_value else { unreachable!() };
        eager.register_reached_relations().unwrap();
        eager.job.freeze_relations(eager.token).unwrap();
        let eager_analysis = eager.job.analyze_family_root(eager_family).unwrap();
        assert!(eager_analysis.exact_term_count > 0);
        assert!(!matches!(
            eager_analysis.bounded_summary.coefficient_bound(),
            NumericContract::Missing
        ));
    }

    #[test]
    fn production_opaque_call_input_gadget_product_remains_compact() {
        fn build(adapter: &mut ProductionAdapter<'_>) -> FamilyValueId {
            let residual = adapter
                .resolve(adapter.plan.target().residual.clone(), Rc::new(BTreeMap::new()))
                .expect("opaque input source residual");
            let Value::Expr(root) = residual else { unreachable!() };
            let ValueOperator::ProgramCall { program } =
                adapter.job.expressions().node(root).unwrap().operator
            else {
                unreachable!()
            };
            let family = adapter.job.programs().family_for_program(program).unwrap();
            let body = adapter.job.programs().family_body(family).unwrap();
            let product = adapter.job.expressions().node(body).unwrap().clone();
            let [gadget, _decomposition] = product.inputs.as_ref() else { unreachable!() };
            let argument =
                adapter.job.expressions_mut().intern_argument(0, ResolvedValueType::Int).unwrap();
            let zero = adapter.intern_index_constant(BigInt::ZERO).unwrap();
            let index = adapter
                .job
                .expressions_mut()
                .intern(ValueOperator::Scalar(ScalarOperation::Add), [argument, zero].into())
                .unwrap();
            let opaque_family = adapter
                .opaque_generated_family(FamilyDomain::new(0, 256).unwrap(), *gadget)
                .unwrap();
            let opaque_call = adapter
                .call_family_in_program_scope_deferred_generated(
                    opaque_family,
                    index,
                    TrustedIndexRange::new(0, 256).unwrap(),
                )
                .unwrap();
            let outer_body = adapter
                .job
                .expressions_mut()
                .intern(ValueOperator::Matrix(MatrixOperation::Add), [opaque_call, body].into())
                .unwrap();
            adapter.generated_family(FamilyDomain::new(0, 256).unwrap(), outer_body).unwrap()
        }

        let protocol = generated_gadget_product_protocol(false, GeneratedGadgetProductCase::Single);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").unwrap();
        let mut compact_adapter =
            ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
        let compact_family = build(&mut compact_adapter);
        let compact_plan = compact_adapter
            .compile_compact_root(&Value::Family(compact_family))
            .expect("opaque call input should retain compact gadget plan");
        compact_adapter.register_reached_relations().unwrap();
        compact_adapter.job.set_compact_shell_plan(compact_plan).unwrap();
        compact_adapter.job.freeze_relations(compact_adapter.token).unwrap();
        let compact = compact_adapter.job.analyze_compact_family_root(compact_family).unwrap();
        assert_eq!(compact.counters.compact_planned_shell_occurrences, 1);
        assert_eq!(compact.counters.compact_shell_holds_released, 1);
        assert_eq!(compact.counters.compact_shell_holds_unmatched, 0);

        let mut eager_adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
        let eager_family = build(&mut eager_adapter);
        eager_adapter.register_reached_relations().unwrap();
        eager_adapter.job.freeze_relations(eager_adapter.token).unwrap();
        let eager = eager_adapter.job.analyze_family_root(eager_family).unwrap();
        assert_eq!(compact.exact_term_count, eager.exact_term_count);
        let mut compact_diagnostics = compact
            .exact_term_diagnostics
            .iter()
            .map(|diagnostic| format!("{diagnostic:?}"))
            .collect::<Vec<_>>();
        let mut eager_diagnostics = eager
            .exact_term_diagnostics
            .iter()
            .map(|diagnostic| format!("{diagnostic:?}"))
            .collect::<Vec<_>>();
        compact_diagnostics.sort();
        eager_diagnostics.sort();
        assert_eq!(compact_diagnostics, eager_diagnostics);
        assert_eq!(compact.bounded_summary, eager.bounded_summary);
        assert_eq!(
            compact.bounded_summary.coefficient_bound(),
            eager.bounded_summary.coefficient_bound()
        );
        assert_eq!(compact.counters.relation_applied, eager.counters.relation_applied);
        assert_eq!(compact.counters.relation_remaining, eager.counters.relation_remaining);
    }

    #[test]
    fn production_compact_family_work_is_independent_of_domain_width() {
        fn analyze(count: u64) -> super::super::report::OperationalReport {
            let protocol = generated_indexed_family_protocol(count, false, false, 64);
            let plan = ProtocolPlan::build(&protocol, "indexed-family-production")
                .expect("indexed-family width protocol plan");
            let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
                .expect("indexed-family width adapter");
            let (mut job, roots) = adapter.lower().expect("indexed-family width lowering");
            assert!(matches!(roots.residual, ProductionRoot::CompactFamily(_)));
            let target = super::super::report::ReportTarget {
                target_id: "indexed-family-production".to_owned(),
                plaintext_modulus: 2_u8.into(),
                ciphertext_modulus: 257_u16.into(),
                boolean_interval: false,
            };
            let before = job.programs().diagnostic_counters();
            let report = super::super::report::analyze_roots(&mut job, &roots, &target)
                .expect("indexed-family width report");
            let after = job.programs().diagnostic_counters();
            assert_eq!(after.beta_nodes_visited, before.beta_nodes_visited);
            assert_eq!(after.beta_reason_misses, before.beta_reason_misses);
            assert_eq!(after.beta_reason_visits, before.beta_reason_visits);
            assert_eq!(after.beta_reason_expr_allocations, before.beta_reason_expr_allocations);
            report
        }

        let small = analyze(4);
        let large = analyze(1_000_000);
        assert_eq!(small.residual, large.residual);
        assert_eq!(small.accepted, large.accepted);
        assert_eq!(
            small.counters.normalization.nodes_processed,
            large.counters.normalization.nodes_processed
        );
        assert_eq!(
            small.counters.normalization.nodes_total,
            small.counters.normalization.nodes_processed
        );
        assert_eq!(
            large.counters.normalization.nodes_total,
            large.counters.normalization.nodes_processed
        );
        assert_eq!(
            small.counters.normalization.compact_peak_live_frames,
            large.counters.normalization.compact_peak_live_frames
        );
        assert_eq!(
            small.counters.normalization.compact_peak_live_values,
            large.counters.normalization.compact_peak_live_values
        );
    }

    #[test]
    fn production_binder_dependent_gadget_family_falls_back_before_freeze() {
        let protocol = generated_indexed_family_protocol(4, true, false, 1);
        let plan = ProtocolPlan::build(&protocol, "indexed-family-production")
            .expect("binder-dependent family protocol plan");
        let mut preflight = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("binder-dependent preflight adapter");
        let residual = preflight
            .resolve(plan.target().residual.clone(), Rc::new(BTreeMap::new()))
            .expect("binder-dependent family residual");
        let reason = preflight
            .compact_residual_preflight(&residual)
            .expect("binder-dependent gadget must reject compact family");
        assert_eq!(reason, "gadget decomposition input is not a concrete leaf");
        assert!(!preflight.job.relations().is_frozen());

        let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("binder-dependent eager adapter");
        let (mut job, roots) = adapter.lower().expect("binder-dependent eager fallback lowering");
        let ProductionRoot::Family(_) = roots.residual else {
            panic!("binder-dependent family must retain ordinary family marker")
        };
        let target = super::super::report::ReportTarget {
            target_id: "indexed-family-production".to_owned(),
            plaintext_modulus: 2_u8.into(),
            ciphertext_modulus: 257_u16.into(),
            boolean_interval: false,
        };
        let eager = super::super::report::analyze_roots(&mut job, &roots, &target)
            .expect("binder-dependent eager family report");
        assert_eq!(eager.residual.exact_term_count, 0);
        assert_ne!(eager.residual.bound, super::super::report::BoundClass::Missing);
    }

    #[test]
    fn production_binder_dependent_explicit_gadget_family_falls_back_before_freeze() {
        let protocol = generated_indexed_family_protocol(4, false, true, 1);
        let plan = ProtocolPlan::build(&protocol, "indexed-family-production")
            .expect("binder-dependent explicit family protocol plan");
        let mut preflight = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("binder-dependent explicit preflight adapter");
        let residual = preflight
            .resolve(plan.target().residual.clone(), Rc::new(BTreeMap::new()))
            .expect("binder-dependent explicit family residual");
        assert_eq!(
            preflight.compact_residual_preflight(&residual).as_deref(),
            Some("gadget decomposition input is binder-dependent")
        );
        assert!(!preflight.job.relations().is_frozen());

        let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("binder-dependent explicit eager adapter");
        let (mut job, roots) = adapter.lower().expect("binder-dependent explicit eager fallback");
        assert!(matches!(roots.residual, ProductionRoot::Family(_)));
        let target = super::super::report::ReportTarget {
            target_id: "indexed-family-production".to_owned(),
            plaintext_modulus: 2_u8.into(),
            ciphertext_modulus: 257_u16.into(),
            boolean_interval: false,
        };
        let eager = super::super::report::analyze_roots(&mut job, &roots, &target)
            .expect("binder-dependent explicit eager family report");
        assert_eq!(eager.residual.exact_term_count, 0);
        assert_ne!(eager.residual.bound, super::super::report::BoundClass::Missing);
    }

    #[test]
    fn production_binder_dependent_scalar_action_is_rejected_before_freeze() {
        let protocol = generated_gather_protocol(1);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("scalar plan");
        let mut adapter =
            ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).expect("scalar adapter");
        let scalar_type = ResolvedMatrixType::new(BigUint::from(256_u16), 1, 1, 1).unwrap();
        let matrix_type = ResolvedMatrixType::new(BigUint::from(256_u16), 1, 2, 2).unwrap();
        let selector = adapter
            .job
            .expressions_mut()
            .intern_argument(0, ResolvedValueType::Int)
            .expect("binder selector");
        let scalar_branch = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(94_000),
                    operation: SamplerOperation::UniformResidue { output: scalar_type.clone() },
                },
                Box::new([]),
            )
            .expect("scalar branch");
        let scalar = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::ExplicitElement {
                    domain: FamilyDomain::new(0, 2).unwrap(),
                    element_type: ResolvedValueType::Matrix(scalar_type),
                },
                Box::new([selector, scalar_branch, scalar_branch]),
            )
            .expect("binder-dependent scalar");
        let matrix = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(94_001),
                    operation: SamplerOperation::UniformResidue { output: matrix_type },
                },
                Box::new([]),
            )
            .expect("non-scalar sibling");
        let body = adapter
            .job
            .expressions_mut()
            .intern(ValueOperator::Matrix(MatrixOperation::Multiply), [scalar, matrix].into())
            .expect("binder-dependent scalar product");
        let family = adapter.generated_family(FamilyDomain::new(0, 1).unwrap(), body).unwrap();
        let index = adapter.intern_index_constant(BigInt::ZERO).unwrap();
        let call = adapter
            .call_family_in_program_scope_deferred_generated(
                family,
                index,
                TrustedIndexRange::new(0, 1).unwrap(),
            )
            .unwrap();
        assert_eq!(
            adapter.compact_residual_preflight(&Value::Expr(call)).as_deref(),
            Some("virtual matrix operator type or arity mismatch")
        );
        assert!(!adapter.job.relations().is_frozen());
    }

    #[test]
    fn production_formal_affine_family_binding_passes_compact_preflight() {
        let protocol = generated_gather_protocol(1);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("formal plan");
        let mut adapter =
            ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).expect("formal adapter");
        let domain = FamilyDomain::new(0, 4).unwrap();
        let matrix_type = ResolvedMatrixType::new(BigUint::from(256_u16), 1, 1, 1).unwrap();
        let inner_body = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(94_101),
                    operation: SamplerOperation::UniformResidue { output: matrix_type.clone() },
                },
                Box::new([]),
            )
            .unwrap();
        let inner_family = adapter.generated_family(domain, inner_body).unwrap();
        let argument =
            adapter.job.expressions_mut().intern_argument(0, ResolvedValueType::Int).unwrap();
        let zero = adapter
            .job
            .expressions_mut()
            .intern(ValueOperator::Constant(TypedConstant::int(0)), Box::new([]))
            .unwrap();
        let mapped_index = adapter
            .job
            .expressions_mut()
            .intern(ValueOperator::Scalar(ScalarOperation::Add), [argument, zero].into())
            .unwrap();
        // Mirror a gather's nested generated index family: the outer family receives an open
        // ProgramCall, whose detached unary body is itself the authorized affine index map.
        let index_family = adapter.generated_family(domain, mapped_index).unwrap();
        let nested_mapped_index = adapter
            .call_family_in_program_scope_deferred_generated(
                index_family,
                argument,
                TrustedIndexRange::new(0, 4).unwrap(),
            )
            .unwrap();
        let inner_call = adapter
            .call_family_in_program_scope_deferred_generated(
                inner_family,
                nested_mapped_index,
                TrustedIndexRange::new(0, 4).unwrap(),
            )
            .unwrap();
        // A second nested call uses a disjoint owner/range. Conflating environments would
        // incorrectly project its translated binding into the outer [0, 4) range.
        let offset_domain = FamilyDomain::new(8, 12).unwrap();
        let offset_branch = |adapter: &mut ProductionAdapter<'_>, event| {
            adapter
                .job
                .expressions_mut()
                .intern(
                    ValueOperator::Sampler {
                        event: SampleEventId(event),
                        operation: SamplerOperation::UniformResidue { output: matrix_type.clone() },
                    },
                    Box::new([]),
                )
                .unwrap()
        };
        let offset_body_argument =
            adapter.job.expressions_mut().intern_argument(0, ResolvedValueType::Int).unwrap();
        let offset_branches = [
            offset_branch(&mut adapter, 94_103),
            offset_branch(&mut adapter, 94_104),
            offset_branch(&mut adapter, 94_105),
            offset_branch(&mut adapter, 94_106),
        ];
        let offset_body = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::ExplicitElement {
                    domain: offset_domain,
                    element_type: ResolvedValueType::Matrix(matrix_type.clone()),
                },
                [offset_body_argument]
                    .into_iter()
                    .chain(offset_branches)
                    .collect::<Vec<_>>()
                    .into(),
            )
            .unwrap();
        let offset_family = adapter.generated_family(offset_domain, offset_body).unwrap();
        let eight = adapter.intern_index_constant(BigInt::from(8_u8)).unwrap();
        let offset_index = adapter
            .job
            .expressions_mut()
            .intern(ValueOperator::Scalar(ScalarOperation::Add), [argument, eight].into())
            .unwrap();
        let offset_call = adapter
            .call_family_in_program_scope_deferred_generated(
                offset_family,
                offset_index,
                TrustedIndexRange::new(8, 12).unwrap(),
            )
            .unwrap();
        let sibling = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(94_102),
                    operation: SamplerOperation::UniformResidue { output: matrix_type.clone() },
                },
                Box::new([]),
            )
            .unwrap();
        let nested_sum = adapter
            .job
            .expressions_mut()
            .intern(ValueOperator::Matrix(MatrixOperation::Add), [inner_call, offset_call].into())
            .unwrap();
        let body = adapter
            .job
            .expressions_mut()
            .intern(ValueOperator::Matrix(MatrixOperation::Add), [nested_sum, sibling].into())
            .unwrap();
        let family = adapter.generated_family(domain, body).unwrap();
        let unrelated_argument =
            adapter.job.expressions_mut().intern_argument(1, ResolvedValueType::Int).unwrap();
        let invalid_binding = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Scalar(ScalarOperation::Add),
                [nested_mapped_index, unrelated_argument].into(),
            )
            .unwrap();
        let mut projector = IndexRangeProjector::new(&adapter);
        let environment = projector.push_environment(vec![IndexRangeProjectionBinding {
            expression: argument,
            value_type: ResolvedValueType::Int,
            range: (BigInt::ZERO, BigInt::from(4_u8)),
        }]);
        assert!(projector.evaluate(invalid_binding, environment).is_none());
        let value = Value::Family(family);
        assert_eq!(adapter.compact_residual_preflight(&value), None);
        let bad_call = adapter
            .call_family_in_program_scope_deferred_generated(
                inner_family,
                offset_index,
                TrustedIndexRange::new(0, 4).unwrap(),
            )
            .unwrap();
        let bad_body = adapter
            .job
            .expressions_mut()
            .intern(ValueOperator::Matrix(MatrixOperation::Add), [bad_call, sibling].into())
            .unwrap();
        let bad_family = adapter.generated_family(domain, bad_body).unwrap();
        assert_eq!(
            adapter.compact_residual_preflight(&Value::Family(bad_family)).as_deref(),
            Some("compact binding expression is open")
        );
        assert!(!adapter.job.relations().is_frozen());
        let shell_plan = adapter.build_compact_shell_plan(&value).expect("formal compact plan");
        assert!(shell_plan.preflight_node_occurrences > 0);
        assert!(!adapter.job.relations().is_frozen());
        adapter.register_reached_relations().unwrap();
        adapter.job.set_compact_shell_plan(shell_plan).unwrap();
        adapter.job.freeze_relations(adapter.token).unwrap();
        let compact_root = adapter
            .close_root(value, &plan.target().residual, "formal nested-call compact root", true)
            .unwrap();
        let ProductionRoot::CompactFamily(compact_family) = compact_root else {
            panic!("formal nested-call family must retain its compact marker")
        };
        let analysis = adapter
            .job
            .analyze_compact_family_root(compact_family)
            .expect("formal nested-call compact analysis");
        assert!(analysis.counters.nodes_processed > 0);
        let eager = adapter
            .job
            .analyze_family_root(compact_family)
            .expect("formal nested-call eager analysis");
        assert_eq!(analysis.exact_term_count, eager.exact_term_count);
        assert_eq!(analysis.exact_term_diagnostics, eager.exact_term_diagnostics);
        assert_eq!(analysis.bounded_summary, eager.bounded_summary);
        assert_eq!(analysis.counters.relation_applied, eager.counters.relation_applied);
        assert_eq!(analysis.counters.relation_remaining, eager.counters.relation_remaining);
    }

    #[test]
    fn compact_phase_a_unsupported_or_factful_islands_select_closed_before_freeze() {
        fn root_with_body(adapter: &mut ProductionAdapter<'_>, body: ExprId) -> ExprId {
            let family = adapter.generated_family(FamilyDomain::new(0, 1).unwrap(), body).unwrap();
            let index = adapter.intern_index_constant(BigInt::ZERO).unwrap();
            adapter
                .call_family_in_program_scope_deferred_generated(
                    family,
                    index,
                    TrustedIndexRange::new(0, 1).unwrap(),
                )
                .unwrap()
        }

        let protocol = generated_gather_protocol(1);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("protocol plan");
        for unsupported in [true, false] {
            let mut adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
                .expect("production adapter");
            let matrix = ResolvedMatrixType::new(257_u16.into(), 1, 1, 1).unwrap();
            let left = adapter
                .job
                .expressions_mut()
                .intern(
                    ValueOperator::Sampler {
                        event: SampleEventId(if unsupported { 80 } else { 81 }),
                        operation: SamplerOperation::UniformResidue { output: matrix.clone() },
                    },
                    Box::new([]),
                )
                .unwrap();
            let right = adapter
                .job
                .expressions_mut()
                .intern(
                    ValueOperator::Sampler {
                        event: SampleEventId(if unsupported { 82 } else { 83 }),
                        operation: SamplerOperation::UniformResidue { output: matrix.clone() },
                    },
                    Box::new([]),
                )
                .unwrap();
            let body = if unsupported {
                adapter
                    .job
                    .expressions_mut()
                    .intern(
                        ValueOperator::Matrix(MatrixOperation::Tensor {
                            output: matrix.clone(),
                            left_layout: MatrixLayout::row_major(1, 1),
                            right_layout: MatrixLayout::row_major(1, 1),
                            output_layout: MatrixLayout::row_major(1, 1),
                        }),
                        [left, right].into(),
                    )
                    .unwrap()
            } else {
                let add = adapter
                    .job
                    .expressions_mut()
                    .intern(ValueOperator::Matrix(MatrixOperation::Add), [left, right].into())
                    .unwrap();
                let mut facts = MatrixFacts::new(
                    matrix.clone(),
                    MatrixMetadata::new(MatrixLayout::row_major(1, 1)),
                );
                facts.coefficient_bound = NumericContract::Known(CoefficientBound::finite(2_u8));
                adapter.job.insert_matrix_facts(adapter.token, add, facts).unwrap();
                add
            };
            let root = root_with_body(&mut adapter, body);
            let reason = adapter
                .compact_residual_preflight(&Value::Expr(root))
                .expect("unsupported/factful root must select eager path");
            assert!(
                (unsupported &&
                    (reason.contains("unsupported concrete operator") ||
                        reason.contains("virtual matrix operator type or arity mismatch") ||
                        reason.contains("virtual tensor scalar type or arity mismatch"))) ||
                    (!unsupported && reason.contains("virtual node has facts")),
                "unexpected fallback reason: {reason}"
            );
            assert!(!adapter.job.relations().is_frozen());
            let eager = adapter
                .materialize_root_value_with_reason(Value::Expr(root), BetaReason::ResidualRoot)
                .expect("legacy materialization");
            let closed = adapter
                .close_root(eager, &plan.target().residual, "fallback root", false)
                .expect("closed fallback root");
            assert!(matches!(closed, ProductionRoot::Closed(_)));
            adapter.job.freeze_relations(adapter.token).unwrap();
        }

        for case in [0_u64, 2] {
            let mut adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
                .expect("table fallback adapter");
            let matrix = if case < 2 {
                ResolvedMatrixType::new(257_u16.into(), 1, 2, 2).unwrap()
            } else {
                ResolvedMatrixType::new(257_u16.into(), 1, 1, 1).unwrap()
            };
            let left = adapter
                .job
                .expressions_mut()
                .intern(
                    ValueOperator::Sampler {
                        event: SampleEventId(86 + case * 2),
                        operation: SamplerOperation::UniformResidue { output: matrix.clone() },
                    },
                    Box::new([]),
                )
                .unwrap();
            let right = adapter
                .job
                .expressions_mut()
                .intern(
                    ValueOperator::Sampler {
                        event: SampleEventId(87 + case * 2),
                        operation: SamplerOperation::UniformResidue { output: matrix.clone() },
                    },
                    Box::new([]),
                )
                .unwrap();
            let multiply = adapter
                .job
                .expressions_mut()
                .intern(ValueOperator::Matrix(MatrixOperation::Multiply), [left, right].into())
                .unwrap();
            let body = match case {
                0 => {
                    let mut facts = MatrixFacts::new(
                        matrix.clone(),
                        MatrixMetadata::new(MatrixLayout::row_major(1, 1)),
                    );
                    facts.coefficient_bound =
                        NumericContract::Known(CoefficientBound::finite(2_u8));
                    adapter.job.insert_matrix_facts(adapter.token, multiply, facts).unwrap();
                    multiply
                }
                _ => adapter
                    .job
                    .expressions_mut()
                    .intern(
                        ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
                            output: matrix.clone(),
                            base: 2,
                            small: true,
                            digit_count: 1,
                        }),
                        [multiply].into(),
                    )
                    .unwrap(),
            };
            let root = root_with_body(&mut adapter, body);
            let reason = adapter
                .compact_residual_preflight(&Value::Expr(root))
                .expect("table fallback must select eager");
            assert!(!adapter.job.relations().is_frozen());
            assert!(
                (case == 0 && reason == "virtual node has facts") ||
                    (case == 2 && reason == "gadget decomposition input is not a concrete leaf"),
                "unexpected table fallback reason: {reason}"
            );
            let eager = adapter
                .materialize_root_value_with_reason(Value::Expr(root), BetaReason::ResidualRoot)
                .unwrap();
            if case == 2 {
                let closed = adapter
                    .close_root(eager, &plan.target().residual, "table fallback root", false)
                    .unwrap();
                assert!(matches!(closed, ProductionRoot::Closed(_)));
            }
        }

        let mut adapter = ProductionAdapter::new(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .expect("binding preflight adapter");
        let matrix = ResolvedMatrixType::new(257_u16.into(), 1, 1, 1).unwrap();
        let left = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(84),
                    operation: SamplerOperation::UniformResidue { output: matrix.clone() },
                },
                Box::new([]),
            )
            .unwrap();
        let right = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(85),
                    operation: SamplerOperation::UniformResidue { output: matrix.clone() },
                },
                Box::new([]),
            )
            .unwrap();
        let multiply = adapter
            .job
            .expressions_mut()
            .intern(ValueOperator::Matrix(MatrixOperation::Multiply), [left, right].into())
            .unwrap();
        let mut projector = IndexRangeProjector::new(&adapter);
        let environment = projector.push_environment(Vec::new());
        assert!(projector.evaluate(multiply, environment).is_none());
        assert!(!adapter.job.relations().is_frozen());
    }

    #[test]
    fn real_lower_selects_compact_and_test_force_eager_matches_it() {
        let protocol = generated_direct_family_protocol();
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("protocol plan");
        let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("compact production adapter");
        let (mut compact_job, compact_roots) = adapter.lower().expect("compact lowering");
        assert!(matches!(compact_roots.residual, ProductionRoot::Compact(_)));
        let ProductionRoot::Compact(compact_root) = compact_roots.residual else { unreachable!() };
        let compact =
            compact_job.normalize_compact_closed_root(compact_root).expect("compact normalization");
        assert!(compact.counters.compact_virtual_calls > 0);
        assert_eq!(
            compact.counters.relation_applied, 0,
            "an eligible Phase-A compact root must not apply relations"
        );

        let eager_adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("forced-eager production adapter");
        let (mut eager_job, eager_roots) =
            eager_adapter.lower_force_eager().expect("forced-eager lowering");
        assert!(matches!(eager_roots.residual, ProductionRoot::Closed(_)));
        let ProductionRoot::Closed(eager_root) = eager_roots.residual else { unreachable!() };
        let eager = eager_job.normalize_closed_root(eager_root).expect("eager normalization");
        assert_eq!(
            full_nf_descriptor_map(&compact_job, &compact.value),
            full_nf_descriptor_map(&eager_job, &eager.value)
        );
    }

    #[test]
    fn production_literal_scale_is_compact_and_composite_scale_is_eager_before_freeze() {
        let literal_protocol = generated_scale_protocol(false);
        let literal_plan = ProtocolPlan::build(&literal_protocol, "toy-threshold").unwrap();
        let literal_adapter =
            ProductionAdapter::new(&literal_protocol, &literal_plan, BTreeMap::new()).unwrap();
        let (mut compact_job, compact_roots) = literal_adapter.lower().unwrap();
        let ProductionRoot::Compact(compact_root) = compact_roots.residual else {
            panic!("literal scale must be compact")
        };
        let compact = compact_job.normalize_compact_closed_root(compact_root).unwrap();
        assert!(compact.counters.compact_logical_scale > 0);
        assert_eq!(compact.counters.relation_applied, 0);

        let eager_adapter =
            ProductionAdapter::new(&literal_protocol, &literal_plan, BTreeMap::new()).unwrap();
        let (mut eager_job, eager_roots) = eager_adapter.lower_force_eager().unwrap();
        let ProductionRoot::Closed(eager_root) = eager_roots.residual else { unreachable!() };
        let eager = eager_job.normalize_closed_root(eager_root).unwrap();
        assert_eq!(
            full_nf_descriptor_map(&compact_job, &compact.value),
            full_nf_descriptor_map(&eager_job, &eager.value)
        );

        let composite_protocol = generated_scale_protocol(true);
        let composite_plan = ProtocolPlan::build(&composite_protocol, "toy-threshold").unwrap();
        let mut preflight =
            ProductionAdapter::new(&composite_protocol, &composite_plan, BTreeMap::new()).unwrap();
        let residual = preflight
            .resolve(composite_plan.target().residual.clone(), Rc::new(BTreeMap::new()))
            .unwrap();
        let reason = preflight
            .compact_residual_preflight(&residual)
            .expect("composite scalar must select eager");
        assert_eq!(reason, "virtual matrix operator type or arity mismatch");
        assert!(!preflight.job.relations().is_frozen());
        let adapter =
            ProductionAdapter::new(&composite_protocol, &composite_plan, BTreeMap::new()).unwrap();
        let (job, roots) = adapter.lower().unwrap();
        assert!(matches!(roots.residual, ProductionRoot::Closed(_)));
        assert!(job.relations().is_frozen());
    }

    #[test]
    fn production_strict_products_are_compact_ordered_and_eager_parity_holds() {
        let protocol = generated_product_protocol(false, false, false, false);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").unwrap();
        let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
        let (mut compact_job, compact_roots) = adapter.lower().unwrap();
        let ProductionRoot::Compact(compact_root) = compact_roots.residual else {
            panic!("strict product must be compact")
        };
        let compact = compact_job.normalize_compact_closed_root(compact_root).unwrap();
        assert!(compact.counters.nodes_total > 0);
        assert_eq!(compact.counters.nodes_processed, compact.counters.nodes_total);
        assert!(compact.counters.compact_strict_products > 0);
        let eager_adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
        let (mut eager_job, eager_roots) = eager_adapter.lower_force_eager().unwrap();
        let ProductionRoot::Closed(eager_root) = eager_roots.residual else { unreachable!() };
        let eager = eager_job.normalize_closed_root(eager_root).unwrap();
        assert_eq!(
            full_nf_descriptor_map(&compact_job, &compact.value),
            full_nf_descriptor_map(&eager_job, &eager.value)
        );

        let reverse_protocol = generated_product_protocol(true, false, false, false);
        let reverse_plan = ProtocolPlan::build(&reverse_protocol, "toy-threshold").unwrap();
        let reverse_adapter =
            ProductionAdapter::new(&reverse_protocol, &reverse_plan, BTreeMap::new()).unwrap();
        let (mut reverse_job, reverse_roots) = reverse_adapter.lower().unwrap();
        let ProductionRoot::Compact(reverse_root) = reverse_roots.residual else {
            panic!("reversed strict product must be compact")
        };
        let reverse = reverse_job.normalize_compact_closed_root(reverse_root).unwrap();
        assert_ne!(
            full_nf_descriptor_map(&compact_job, &compact.value).0,
            full_nf_descriptor_map(&reverse_job, &reverse.value).0
        );
    }

    #[test]
    fn production_one_sided_scalar_products_are_compact_with_exact_holds_and_eager_parity() {
        for reverse in [false, true] {
            let protocol = generated_product_protocol(reverse, false, true, false);
            let plan = ProtocolPlan::build(&protocol, "toy-threshold").unwrap();
            let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
            let (mut compact_job, compact_roots) = adapter.lower().unwrap();
            let ProductionRoot::Compact(compact_root) = compact_roots.residual else {
                panic!("one-sided scalar product must be compact")
            };
            let compact = compact_job.normalize_compact_closed_root(compact_root).unwrap();
            assert_eq!(compact.counters.compact_scalar_consumers, 1);
            assert_eq!(compact.counters.compact_scalar_holds_released, 1);
            assert_eq!(compact.counters.compact_scalar_holds_unmatched, 0);
            let compact_evidence = full_nf_descriptor_map(&compact_job, &compact.value);
            assert!(!compact_evidence.0.is_empty());
            assert!(
                compact_evidence
                    .0
                    .keys()
                    .all(|(central, ordered)| central.len() == 1 && ordered.len() == 1)
            );

            let eager_adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
            let (mut eager_job, eager_roots) = eager_adapter.lower_force_eager().unwrap();
            let ProductionRoot::Closed(eager_root) = eager_roots.residual else { unreachable!() };
            let eager = eager_job.normalize_closed_root(eager_root).unwrap();
            assert_eq!(
                full_nf_descriptor_map(&compact_job, &compact.value),
                full_nf_descriptor_map(&eager_job, &eager.value)
            );
        }
    }

    #[test]
    fn production_both_scalar_products_are_compact_without_scalar_plan_and_eager_parity() {
        let protocol = generated_scalar_negative_protocol(GeneratedScalarNegativeCase::BothScalar);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("both-scalar plan");
        let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("both-scalar compact adapter");
        let (mut compact_job, compact_roots) = adapter.lower().expect("compact lowering");
        let ProductionRoot::Compact(compact_root) = compact_roots.residual else {
            panic!("both-scalar product must be compact")
        };
        let compact = compact_job
            .normalize_compact_closed_root(compact_root)
            .expect("both-scalar compact normalization");
        assert_eq!(compact.counters.compact_scalar_consumers, 0);
        assert_eq!(compact.counters.compact_scalar_holds_released, 0);
        assert_eq!(compact.counters.compact_scalar_holds_unmatched, 0);
        assert_eq!(
            compact.value.exact_nf.as_ref().expect("both-scalar NF").exact_terms.len(),
            0,
            "finite both-scalar action should retain only its bounded summary"
        );
        let compact_evidence = full_nf_descriptor_map(&compact_job, &compact.value);

        let eager_adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("both-scalar eager adapter");
        let (mut eager_job, eager_roots) =
            eager_adapter.lower_force_eager().expect("both-scalar eager lowering");
        let ProductionRoot::Closed(eager_root) = eager_roots.residual else { unreachable!() };
        let eager =
            eager_job.normalize_closed_root(eager_root).expect("both-scalar eager normalization");
        assert_eq!(compact_evidence, full_nf_descriptor_map(&eager_job, &eager.value));
    }

    #[test]
    fn production_both_scalar_program_calls_are_compact_without_holds_and_eager_parity() {
        fn build(
            protocol: &crate::ProtocolDecl,
            plan: &ProtocolPlan,
            compact: bool,
            cancellation: bool,
        ) -> (CheckerJob, ProductionRoot, CompactShellPlan) {
            let mut adapter = ProductionAdapter::new(protocol, plan, BTreeMap::new())
                .expect("both-scalar program-call adapter");
            let matrix = ResolvedMatrixType::new(257_u16.into(), 1, 1, 1).unwrap();
            let opaque_leaf = adapter
                .job
                .expressions_mut()
                .intern(
                    ValueOperator::Sampler {
                        event: SampleEventId(1201),
                        operation: SamplerOperation::UniformResidue { output: matrix.clone() },
                    },
                    Box::new([]),
                )
                .unwrap();
            let reducible_leaf = adapter
                .job
                .expressions_mut()
                .intern(
                    ValueOperator::Sampler {
                        event: SampleEventId(1202),
                        operation: SamplerOperation::UniformResidue { output: matrix.clone() },
                    },
                    Box::new([]),
                )
                .unwrap();
            let domain = FamilyDomain::new(0, 1).unwrap();
            let opaque_family =
                adapter.explicit_family(domain, Box::new([opaque_leaf])).expect("opaque family");
            let reducible_family =
                adapter.generated_family(domain, reducible_leaf).expect("reducible family");
            let argument =
                adapter.job.expressions_mut().intern_argument(0, ResolvedValueType::Int).unwrap();
            let range = TrustedIndexRange::new(0, 1).unwrap();
            let opaque_call = adapter
                .call_family_in_program_scope_deferred_generated(opaque_family, argument, range)
                .unwrap();
            let reducible_call = adapter
                .call_family_in_program_scope_deferred_generated(reducible_family, argument, range)
                .unwrap();
            let product = adapter
                .job
                .expressions_mut()
                .intern(
                    ValueOperator::Matrix(MatrixOperation::Multiply),
                    [opaque_call, reducible_call].into(),
                )
                .unwrap();
            let body = if cancellation {
                adapter
                    .job
                    .expressions_mut()
                    .intern(
                        ValueOperator::Matrix(MatrixOperation::Subtract),
                        [product, product].into(),
                    )
                    .unwrap()
            } else {
                product
            };
            let family = adapter.generated_family(domain, body).expect("outer family");
            let residual = Value::Family(family);
            let shell_plan = adapter
                .compile_compact_root(&residual)
                .expect("both scalar program calls are compact eligible");
            assert_eq!(shell_plan.scalar_occurrences(), 0);
            assert_eq!(shell_plan.scalar_program_calls.len(), 0);
            adapter.register_reached_relations().unwrap();
            if compact {
                adapter.job.set_compact_shell_plan(shell_plan.clone()).unwrap();
            }
            adapter.job.freeze_relations(adapter.token).unwrap();
            let root = adapter
                .close_root(
                    residual,
                    &plan.target().residual,
                    "both-scalar program-call root",
                    compact,
                )
                .unwrap();
            (adapter.job, root, shell_plan)
        }

        let protocol = generated_gather_protocol(1);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("both-scalar plan");
        let (mut compact_job, compact_root, compact_plan) = build(&protocol, &plan, true, false);
        let ProductionRoot::CompactFamily(compact_family) = compact_root else {
            panic!("both-scalar program-call family must be compact")
        };
        let compact = compact_job
            .analyze_compact_family_root(compact_family)
            .expect("both-scalar compact analysis");
        assert_eq!(compact.counters.compact_scalar_consumers, 0);
        assert_eq!(compact.counters.compact_scalar_holds_released, 0);
        assert_eq!(compact.counters.compact_scalar_holds_unmatched, 0);
        assert!(compact.exact_term_count > 0, "both-scalar factors must retain exact identity");
        assert_eq!(compact_plan.scalar_occurrences(), 0);

        let (mut eager_job, eager_root, _) = build(&protocol, &plan, false, false);
        let ProductionRoot::Family(eager_family) = eager_root else {
            panic!("ordinary both-scalar program-call family marker")
        };
        let eager =
            eager_job.analyze_family_root(eager_family).expect("both-scalar eager analysis");
        assert_eq!(compact.exact_term_diagnostics, eager.exact_term_diagnostics);
        assert_eq!(compact.exact_term_count, eager.exact_term_count);
        assert_eq!(compact.bounded_summary, eager.bounded_summary);
        assert_eq!(compact.counters.relation_applied, eager.counters.relation_applied);
        assert_eq!(compact.counters.relation_remaining, eager.counters.relation_remaining);

        let (mut compact_job, compact_root, _) = build(&protocol, &plan, true, true);
        let ProductionRoot::CompactFamily(compact_family) = compact_root else {
            panic!("both-scalar cancellation family must be compact")
        };
        let compact = compact_job
            .analyze_compact_family_root(compact_family)
            .expect("both-scalar cancellation compact analysis");
        assert_eq!(compact.exact_term_count, 0);
        assert!(compact.exact_term_diagnostics.is_empty());

        let (mut eager_job, eager_root, _) = build(&protocol, &plan, false, true);
        let ProductionRoot::Family(eager_family) = eager_root else {
            panic!("ordinary both-scalar cancellation family marker")
        };
        let eager = eager_job
            .analyze_family_root(eager_family)
            .expect("both-scalar cancellation eager analysis");
        assert_eq!(compact.exact_term_count, eager.exact_term_count);
        assert_eq!(compact.exact_term_diagnostics, eager.exact_term_diagnostics);
        assert_eq!(compact.bounded_summary, eager.bounded_summary);
        assert_eq!(compact.counters.relation_applied, eager.counters.relation_applied);
        assert_eq!(compact.counters.relation_remaining, eager.counters.relation_remaining);
    }

    #[test]
    fn production_both_scalar_closed_program_calls_have_full_descriptor_parity() {
        fn build(
            protocol: &crate::ProtocolDecl,
            plan: &ProtocolPlan,
            compact: bool,
        ) -> (CheckerJob, ProductionRoot) {
            let mut adapter = ProductionAdapter::new(protocol, plan, BTreeMap::new())
                .expect("closed both-scalar adapter");
            let matrix = ResolvedMatrixType::new(257_u16.into(), 1, 1, 1).unwrap();
            let opaque_leaf = adapter
                .job
                .expressions_mut()
                .intern(
                    ValueOperator::Sampler {
                        event: SampleEventId(1211),
                        operation: SamplerOperation::UniformResidue { output: matrix.clone() },
                    },
                    Box::new([]),
                )
                .unwrap();
            let reducible_leaf = adapter
                .job
                .expressions_mut()
                .intern(
                    ValueOperator::Sampler {
                        event: SampleEventId(1212),
                        operation: SamplerOperation::UniformResidue { output: matrix.clone() },
                    },
                    Box::new([]),
                )
                .unwrap();
            let domain = FamilyDomain::new(0, 1).unwrap();
            let opaque_family =
                adapter.explicit_family(domain, Box::new([opaque_leaf])).expect("opaque family");
            let reducible_family =
                adapter.generated_family(domain, reducible_leaf).expect("reducible family");
            let index = adapter.intern_index_constant(BigInt::ZERO).unwrap();
            let range = TrustedIndexRange::new(0, 1).unwrap();
            let opaque_call = adapter
                .call_family_in_program_scope_deferred_generated(opaque_family, index, range)
                .unwrap();
            let reducible_call = adapter
                .call_family_in_program_scope_deferred_generated(reducible_family, index, range)
                .unwrap();
            let expression = adapter
                .job
                .expressions_mut()
                .intern(
                    ValueOperator::Matrix(MatrixOperation::Multiply),
                    [opaque_call, reducible_call].into(),
                )
                .unwrap();
            let outer_family = adapter.generated_family(domain, expression).unwrap();
            let root = adapter
                .call_family_in_program_scope_deferred_generated(outer_family, index, range)
                .unwrap();
            let value = if compact {
                Value::Expr(root)
            } else {
                adapter
                    .materialize_root_value_with_reason(Value::Expr(root), BetaReason::ResidualRoot)
                    .unwrap()
            };
            let shell_plan =
                adapter.compile_compact_root(&Value::Expr(root)).expect("closed both-scalar plan");
            assert_eq!(shell_plan.scalar_occurrences(), 0);
            adapter.register_reached_relations().unwrap();
            if compact {
                adapter.job.set_compact_shell_plan(shell_plan).unwrap();
            }
            adapter.job.freeze_relations(adapter.token).unwrap();
            let root = adapter
                .close_root(value, &plan.target().residual, "closed both-scalar root", compact)
                .unwrap();
            (adapter.job, root)
        }

        let protocol = generated_gather_protocol(1);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("closed scalar plan");
        let (mut compact_job, compact_root) = build(&protocol, &plan, true);
        let ProductionRoot::Compact(compact_root) = compact_root else {
            panic!("closed both-scalar root must be compact")
        };
        let compact = compact_job
            .normalize_compact_closed_root(compact_root)
            .expect("closed both-scalar compact normalization");

        let (mut eager_job, eager_root) = build(&protocol, &plan, false);
        let ProductionRoot::Closed(eager_root) = eager_root else {
            panic!("closed both-scalar eager root marker")
        };
        let eager = eager_job
            .normalize_closed_root(eager_root)
            .expect("closed both-scalar eager normalization");
        assert_eq!(
            full_nf_descriptor_map(&compact_job, &compact.value),
            full_nf_descriptor_map(&eager_job, &eager.value)
        );
        let compact_evidence = full_nf_descriptor_map(&compact_job, &compact.value);
        assert_eq!(compact.value.exact_nf.as_ref().unwrap().exact_terms.len(), 1);
        assert!(
            compact_evidence.0.keys().any(|(central, ordered)| central.len() + ordered.len() == 2),
            "both scalar factors disappeared from the full descriptor: {compact_evidence:?}"
        );
        assert_eq!(compact.value.coefficient_bound, eager.value.coefficient_bound);
        assert_eq!(compact.counters.relation_applied, eager.counters.relation_applied);
        assert_eq!(compact.counters.relation_remaining, eager.counters.relation_remaining);
    }

    #[test]
    fn production_both_scalar_gadget_compact_family_boundary_and_reverse_fallback() {
        fn build(
            protocol: &crate::ProtocolDecl,
            plan: &ProtocolPlan,
            reverse: bool,
            shared: bool,
            compact: bool,
        ) -> (CheckerJob, ProductionRoot) {
            let mut adapter = ProductionAdapter::new(protocol, plan, BTreeMap::new())
                .expect("1x1 gadget adapter");
            let matrix = ResolvedMatrixType::new(257_u16.into(), 1, 1, 1).unwrap();
            let gadget = adapter
                .job
                .expressions_mut()
                .intern(
                    ValueOperator::Source(SemanticSourceIdentity {
                        stable_definition: "production-1x1-gadget".to_owned(),
                        invocation: "production-1x1-gadget".to_owned(),
                        sample_event: None,
                        output_role: "value".to_owned(),
                        sampler: None,
                        artifact: None,
                        value_type: ResolvedValueType::Matrix(matrix.clone()),
                        coordinates: Box::new([]),
                        matrix_constant: Some(MatrixConstantKind::Gadget { base: 2, small: false }),
                    }),
                    Box::new([]),
                )
                .unwrap();
            adapter
                .job
                .insert_matrix_facts(
                    adapter.token,
                    gadget,
                    MatrixFacts::new(
                        matrix.clone(),
                        MatrixMetadata::new(MatrixLayout::row_major(1, 1)),
                    ),
                )
                .unwrap();
            let input = adapter
                .job
                .expressions_mut()
                .intern(
                    ValueOperator::Sampler {
                        event: SampleEventId(12_101),
                        operation: SamplerOperation::UniformResidue { output: matrix.clone() },
                    },
                    Box::new([]),
                )
                .unwrap();
            let decomposition = adapter
                .job
                .expressions_mut()
                .intern(
                    ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
                        output: matrix.clone(),
                        base: 2,
                        small: false,
                        digit_count: 1,
                    }),
                    Box::new([input]),
                )
                .unwrap();
            adapter
                .job
                .register_gadget_recomposition(
                    adapter.token,
                    GadgetRecompositionRule {
                        base: 2,
                        small: false,
                        digit_count: 1,
                        gadget_type: matrix.clone(),
                        decomposition_type: matrix.clone(),
                        output_type: matrix.clone(),
                        gadget_layout: Some(MatrixLayout::row_major(1, 1)),
                        decomposition_layout: None,
                        input_layout: None,
                        input_type: matrix.clone(),
                    },
                )
                .unwrap();
            let inputs = if reverse { [decomposition, gadget] } else { [gadget, decomposition] };
            let product = adapter
                .job
                .expressions_mut()
                .intern(ValueOperator::Matrix(MatrixOperation::Multiply), inputs.into())
                .unwrap();
            let body = if shared {
                adapter
                    .job
                    .expressions_mut()
                    .intern(ValueOperator::Matrix(MatrixOperation::Add), [product, product].into())
                    .unwrap()
            } else {
                product
            };
            let family = adapter.generated_family(FamilyDomain::new(0, 1).unwrap(), body).unwrap();
            let residual = Value::Family(family);
            let planned = adapter.compile_compact_root(&residual);
            if compact {
                let mut planned = planned.expect("forward gadget plan");
                assert_eq!(planned.scalar_occurrences(), 0);
                adapter.register_reached_relations().unwrap();
                adapter.materialize_compact_shell_plan(&mut planned).unwrap();
                adapter.job.set_compact_shell_plan(planned).unwrap();
            } else {
                adapter.register_reached_relations().unwrap();
            }
            adapter.job.freeze_relations(adapter.token).unwrap();
            let root = adapter
                .close_root(residual, &plan.target().residual, "1x1 gadget root", compact)
                .unwrap();
            (adapter.job, root)
        }

        fn eager_analysis(
            job: &mut CheckerJob,
            family: FamilyValueId,
        ) -> (
            super::super::normal_form::AnalyzedValue,
            super::super::normal_form::NormalizationCounters,
        ) {
            let root = job.programs().root(&job.expressions(), family.program()).unwrap();
            let materialized =
                job.materialize_reducible_generated_calls(root.expression()).unwrap();
            let scoped = job
                .programs()
                .detached_scoped(&job.expressions(), family.program(), materialized)
                .unwrap();
            job.normalize(scoped).unwrap()
        }

        let protocol = generated_gather_protocol(1);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("gadget plan");
        for shared in [false, true] {
            let (mut compact_job, compact_root) = build(&protocol, &plan, false, shared, true);
            let ProductionRoot::CompactFamily(compact_family) = compact_root else {
                panic!("forward gadget product must use CompactFamily")
            };
            let (compact_value, compact_counters) =
                compact_job.normalize_compact_family(compact_family).unwrap();
            let expected_occurrences = if shared { 2 } else { 1 };
            assert!(compact_counters.compact_strict_products >= expected_occurrences);
            assert_eq!(compact_counters.compact_planned_shell_occurrences, expected_occurrences);
            assert_eq!(compact_counters.compact_shell_allocated, 1);
            assert_eq!(
                compact_counters.compact_shell_holds_released, expected_occurrences,
                "unexpected gadget counters: {compact_counters:?}"
            );
            assert_eq!(compact_counters.compact_shell_holds_unmatched, 0);
            assert_eq!(compact_counters.compact_scalar_consumers, 0);
            assert_eq!(compact_counters.compact_scalar_holds_unmatched, 0);
            let (mut eager_job, eager_root) = build(&protocol, &plan, false, shared, false);
            let ProductionRoot::Family(eager_family) = eager_root else {
                panic!("eager gadget product must use Family")
            };
            let (eager_value, eager_counters) = eager_analysis(&mut eager_job, eager_family);
            assert_eq!(
                full_nf_descriptor_map(&compact_job, &compact_value),
                full_nf_descriptor_map(&eager_job, &eager_value)
            );
            assert_eq!(compact_value.coefficient_bound, eager_value.coefficient_bound);
            assert_eq!(compact_counters.relation_applied, eager_counters.relation_applied);
            assert_eq!(compact_counters.relation_remaining, eager_counters.relation_remaining);
        }

        let (mut reverse_job, reverse_root) = build(&protocol, &plan, true, false, false);
        let ProductionRoot::Family(reverse_family) = reverse_root else {
            panic!("reverse gadget product must use ordinary Family")
        };
        let (reverse_value, reverse_counters) = eager_analysis(&mut reverse_job, reverse_family);
        let reverse_terms = full_nf_descriptor_map(&reverse_job, &reverse_value);
        assert_eq!(reverse_value.exact_nf.as_ref().unwrap().term_count(), 1);
        assert_eq!(reverse_terms.0.len(), 1);
        assert_eq!(reverse_counters.relation_remaining, 0);
        let (central, ordered) = reverse_terms.0.keys().next().unwrap();
        assert!(ordered.is_empty());
        assert_eq!(central.len(), 2, "reverse product must retain both factor identities");
        assert_ne!(central[0], central[1]);
    }

    #[test]
    fn production_tall_one_sided_scalar_1x80_products_preserve_order_and_eager_parity() {
        for reverse in [false, true] {
            let protocol =
                generated_product_protocol_with_shape(reverse, false, true, false, 1, 80);
            let plan = ProtocolPlan::build(&protocol, "toy-threshold").unwrap();
            let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
            let (mut compact_job, compact_roots) = adapter.lower().unwrap();
            let ProductionRoot::Compact(compact_root) = compact_roots.residual else {
                panic!("1x80 one-sided scalar product must be compact")
            };
            let compact = compact_job.normalize_compact_closed_root(compact_root).unwrap();
            assert_eq!(compact.counters.compact_scalar_consumers, 1);
            assert_eq!(compact.counters.compact_scalar_holds_released, 1);
            assert_eq!(compact.counters.compact_scalar_holds_unmatched, 0);
            let compact_evidence = full_nf_descriptor_map(&compact_job, &compact.value);
            assert!(!compact_evidence.0.is_empty());
            assert!(
                compact_evidence
                    .0
                    .keys()
                    .all(|(central, ordered)| { central.len() == 1 && ordered.len() == 1 })
            );

            let eager_adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
            let (mut eager_job, eager_roots) = eager_adapter.lower_force_eager().unwrap();
            let ProductionRoot::Closed(eager_root) = eager_roots.residual else { unreachable!() };
            let eager = eager_job.normalize_closed_root(eager_root).unwrap();
            assert_eq!(compact_evidence, full_nf_descriptor_map(&eager_job, &eager.value));
        }
    }

    #[test]
    fn production_indexed_slice_scalar_program_call_is_compact_and_eager_parity_holds() {
        for reverse in [false, true] {
            let protocol = generated_indexed_scalar_product_protocol(
                reverse,
                false,
                false,
                IndexedScalarCase::Valid,
                IndexedScalarProductKind::Multiply,
            );
            let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("indexed plan");
            let mut preflight = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
                .expect("indexed scalar preflight adapter");
            let residual = preflight
                .resolve(plan.target().residual.clone(), Rc::new(BTreeMap::new()))
                .expect("indexed scalar preflight residual");
            let compiled = match preflight.compile_compact_root(&residual) {
                Ok(compiled) => compiled,
                Err(reason) => panic!("indexed scalar preflight rejected: {reason}"),
            };
            assert_eq!(compiled.scalar_program_calls.len(), 1);
            assert_eq!(compiled.scalar_occurrences(), 1);
            let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
                .expect("indexed scalar adapter");
            let (mut compact_job, compact_roots) = adapter.lower().expect("compact lowering");
            let ProductionRoot::Compact(compact_root) = compact_roots.residual else {
                panic!("indexed scalar product must be compact")
            };
            let compact = compact_job
                .normalize_compact_closed_root(compact_root)
                .expect("indexed scalar compact normalization");
            assert_eq!(compact.counters.compact_scalar_consumers, 1);
            assert_eq!(compact.counters.compact_scalar_holds_unmatched, 0);

            let eager_adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
                .expect("indexed eager adapter");
            let (mut eager_job, eager_roots) =
                eager_adapter.lower_force_eager().expect("indexed eager lowering");
            let ProductionRoot::Closed(eager_root) = eager_roots.residual else { unreachable!() };
            let eager =
                eager_job.normalize_closed_root(eager_root).expect("indexed eager normalization");
            assert_eq!(
                full_nf_descriptor_map(&compact_job, &compact.value),
                full_nf_descriptor_map(&eager_job, &eager.value)
            );
        }
    }

    #[test]
    fn production_indexed_slice_scalar_tensor_subtract_is_compact_and_eager_parity_holds() {
        for reverse in [false, true] {
            let protocol = generated_indexed_scalar_product_protocol(
                reverse,
                true,
                false,
                IndexedScalarCase::Valid,
                IndexedScalarProductKind::Tensor,
            );
            let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("tensor plan");
            let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
                .expect("tensor compact adapter");
            let (mut compact_job, compact_roots) =
                adapter.lower().expect("tensor compact lowering");
            let ProductionRoot::Compact(compact_root) = compact_roots.residual else {
                panic!("Tensor scalar subtraction must remain compact")
            };
            let compact = compact_job
                .normalize_compact_closed_root(compact_root)
                .expect("tensor compact normalization");
            assert_eq!(compact.counters.compact_scalar_consumers, 2);
            assert_eq!(compact.counters.compact_scalar_holds_unmatched, 0);
            assert!(compact.value.exact_nf.as_ref().is_some_and(|nf| !nf.exact_terms.is_empty()));

            let eager_adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
                .expect("tensor eager adapter");
            let (mut eager_job, eager_roots) =
                eager_adapter.lower_force_eager().expect("tensor eager lowering");
            let ProductionRoot::Closed(eager_root) = eager_roots.residual else { unreachable!() };
            let eager = eager_job.normalize_closed_root(eager_root).expect("tensor eager NF");
            assert_eq!(
                full_nf_descriptor_map(&compact_job, &compact.value),
                full_nf_descriptor_map(&eager_job, &eager.value)
            );
            assert_eq!(compact.counters.relation_applied, eager.counters.relation_applied);
            assert_eq!(compact.counters.relation_remaining, eager.counters.relation_remaining);
        }
    }

    #[test]
    fn production_tensor_nonprogram_scalar_falls_back_before_freeze() {
        let protocol = generated_indexed_scalar_product_protocol(
            false,
            false,
            false,
            IndexedScalarCase::Valid,
            IndexedScalarProductKind::TensorNonProgramScalar,
        );
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("tensor fallback plan");
        let mut preflight = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("tensor fallback preflight adapter");
        let residual = preflight
            .resolve(plan.target().residual.clone(), Rc::new(BTreeMap::new()))
            .expect("tensor fallback residual");
        let reason = preflight
            .compact_residual_preflight(&residual)
            .expect("non-ProgramCall tensor scalar must reject compact");
        assert_eq!(reason, "tensor scalar consumer requires indexed scalar ProgramCall");
        assert!(!preflight.job.relations().is_frozen());

        let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("tensor fallback eager adapter");
        let (mut eager_job, eager_roots) =
            adapter.lower_force_eager().expect("tensor fallback eager lowering");
        let ProductionRoot::Closed(eager_root) = eager_roots.residual else {
            panic!("tensor fallback must use the ordinary closed root")
        };
        let eager = eager_job
            .normalize_closed_root(eager_root)
            .expect("tensor fallback eager normalization");
        assert!(eager.value.exact_nf.as_ref().is_some_and(|nf| !nf.exact_terms.is_empty()));
    }

    #[test]
    fn production_indexed_scalar_call_multiply_and_tensor_tokens_are_distinct() {
        let protocol = generated_indexed_scalar_product_protocol(
            false,
            true,
            false,
            IndexedScalarCase::Valid,
            IndexedScalarProductKind::Mixed,
        );
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("mixed plan");
        let mut preflight = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("mixed preflight adapter");
        let residual = preflight
            .resolve(plan.target().residual.clone(), Rc::new(BTreeMap::new()))
            .expect("mixed residual");
        let compiled = preflight.compile_compact_root(&residual).expect("mixed compact plan");
        assert_eq!(compiled.scalar_program_calls.len(), 2);
        assert_eq!(compiled.scalar_occurrences(), 2);
        assert!(compiled.scalar_program_calls.keys().all(|(consumer, call, _)| *consumer != *call));
        let mut scalar_call_ids = compiled.scalar_program_calls.keys().map(|(_, call, _)| *call);
        let shared_call = scalar_call_ids.next().expect("mixed scalar call");
        assert!(scalar_call_ids.all(|call| call == shared_call));
        let (mut multiply_consumers, mut tensor_consumers) = (0_u8, 0_u8);
        for (consumer, _, _) in compiled.scalar_program_calls.keys() {
            match preflight.job.expressions().node(*consumer).unwrap().operator {
                ValueOperator::Matrix(MatrixOperation::Multiply) => multiply_consumers += 1,
                ValueOperator::Matrix(MatrixOperation::Tensor { .. }) => tensor_consumers += 1,
                _ => panic!("indexed scalar token has an unsupported consumer operator"),
            }
        }
        assert_eq!(multiply_consumers, 1);
        assert_eq!(tensor_consumers, 1);

        let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("mixed compact adapter");
        let (mut compact_job, compact_roots) = adapter.lower().expect("mixed lowering");
        let ProductionRoot::Compact(compact_root) = compact_roots.residual else {
            panic!("mixed scalar consumers must remain compact")
        };
        let compact = compact_job
            .normalize_compact_closed_root(compact_root)
            .expect("mixed compact normalization");
        assert_eq!(compact.counters.compact_scalar_consumers, 2);
        assert_eq!(compact.counters.compact_scalar_holds_unmatched, 0);

        let eager_adapter =
            ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).expect("mixed eager adapter");
        let (mut eager_job, eager_roots) =
            eager_adapter.lower_force_eager().expect("mixed eager lowering");
        let ProductionRoot::Closed(eager_root) = eager_roots.residual else { unreachable!() };
        let eager = eager_job.normalize_closed_root(eager_root).expect("mixed eager NF");
        assert_eq!(
            full_nf_descriptor_map(&compact_job, &compact.value),
            full_nf_descriptor_map(&eager_job, &eager.value)
        );
        assert_eq!(compact.counters.relation_applied, eager.counters.relation_applied);
        assert_eq!(compact.counters.relation_remaining, eager.counters.relation_remaining);
    }

    #[test]
    fn production_closed_source_scalar_program_call_uses_tensor_token() {
        let protocol = generated_gather_protocol(1);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("source scalar plan");
        fn build<'a>(
            protocol: &'a crate::ProtocolDecl,
            plan: &'a ProtocolPlan,
            unsupported_body: bool,
        ) -> (ProductionAdapter<'a>, FamilyValueId) {
            let mut adapter = ProductionAdapter::new(protocol, plan, BTreeMap::new())
                .expect("source scalar adapter");
            let scalar_type = ResolvedMatrixType::new(256_u16.into(), 1, 1, 1).unwrap();
            let matrix_type = ResolvedMatrixType::new(256_u16.into(), 1, 1, 80).unwrap();
            let sibling_left_type = ResolvedMatrixType::new(256_u16.into(), 1, 1, 2).unwrap();
            let sibling_right_type = ResolvedMatrixType::new(256_u16.into(), 1, 2, 80).unwrap();
            let source = |adapter: &mut ProductionAdapter<'_>, name: &str, value_type| {
                adapter
                    .job
                    .expressions_mut()
                    .intern(
                        ValueOperator::Source(SemanticSourceIdentity {
                            stable_definition: name.to_owned(),
                            invocation: name.to_owned(),
                            sample_event: None,
                            output_role: "value".to_owned(),
                            sampler: None,
                            artifact: None,
                            value_type,
                            coordinates: Box::new([]),
                            matrix_constant: None,
                        }),
                        Box::new([]),
                    )
                    .unwrap()
            };
            let scalar_source = source(
                &mut adapter,
                "closed-source-scalar",
                ResolvedValueType::Matrix(scalar_type.clone()),
            );
            let scalar_body = if unsupported_body {
                let second_source = source(
                    &mut adapter,
                    "closed-source-scalar-second",
                    ResolvedValueType::Matrix(scalar_type.clone()),
                );
                adapter
                    .job
                    .expressions_mut()
                    .intern(
                        ValueOperator::Matrix(MatrixOperation::Add),
                        [scalar_source, second_source].into(),
                    )
                    .unwrap()
            } else {
                scalar_source
            };
            let domain = FamilyDomain::new(0, 1).unwrap();
            let scalar_family = adapter.generated_family(domain, scalar_body).unwrap();
            let argument =
                adapter.job.expressions_mut().intern_argument(0, ResolvedValueType::Int).unwrap();
            let scalar_call = adapter
                .job
                .call_family_in_program_scope_deferred_generated(
                    scalar_family,
                    argument,
                    TrustedIndexRange::new(0, 1).unwrap(),
                )
                .unwrap();
            let left = source(
                &mut adapter,
                "closed-source-left",
                ResolvedValueType::Matrix(sibling_left_type),
            );
            let right = source(
                &mut adapter,
                "closed-source-right",
                ResolvedValueType::Matrix(sibling_right_type),
            );
            let sibling = adapter
                .job
                .expressions_mut()
                .intern(ValueOperator::Matrix(MatrixOperation::Multiply), [left, right].into())
                .unwrap();
            let tensor = adapter
                .job
                .expressions_mut()
                .intern(
                    ValueOperator::Matrix(MatrixOperation::Tensor {
                        output: matrix_type,
                        left_layout: MatrixLayout::row_major(1, 1),
                        right_layout: MatrixLayout::row_major(1, 80),
                        output_layout: MatrixLayout::row_major(1, 80),
                    }),
                    [scalar_call, sibling].into(),
                )
                .unwrap();
            let family = adapter.generated_family(domain, tensor).unwrap();
            (adapter, family)
        }
        let (mut adapter, family) = build(&protocol, &plan, false);
        let compiled = adapter
            .compile_compact_root(&Value::Family(family))
            .expect("closed Source scalar family must compile");
        assert_eq!(compiled.scalar_program_calls.len(), 1);
        assert_eq!(compiled.scalar_occurrences(), 1);
        adapter.register_reached_relations().unwrap();
        adapter.job.set_compact_shell_plan(compiled).unwrap();
        adapter.job.freeze_relations(adapter.token).unwrap();
        let compact = adapter
            .job
            .analyze_compact_family_root(family)
            .expect("closed Source compact family normalization");
        assert_eq!(compact.counters.compact_scalar_consumers, 1);
        assert_eq!(compact.counters.compact_scalar_holds_unmatched, 0);

        let (eager_adapter, eager_family) = build(&protocol, &plan, false);
        let eager_token = eager_adapter.token;
        let mut eager_job = eager_adapter.job;
        eager_job.freeze_relations(eager_token).unwrap();
        let eager = eager_job
            .analyze_family_root(eager_family)
            .expect("closed Source eager family normalization");
        assert_eq!(compact.exact_term_diagnostics, eager.exact_term_diagnostics);
        assert_eq!(compact.bounded_summary, eager.bounded_summary);
        assert_eq!(
            compact.bounded_summary.coefficient_bound(),
            eager.bounded_summary.coefficient_bound()
        );
        assert_eq!(compact.counters.relation_applied, eager.counters.relation_applied);
        assert_eq!(compact.counters.relation_remaining, eager.counters.relation_remaining);

        let (rejected_adapter, rejected_family) = build(&protocol, &plan, true);
        let rejection = rejected_adapter
            .compile_compact_root(&Value::Family(rejected_family))
            .expect_err("generic scalar family body must not receive a compact token");
        assert_eq!(rejection, "scalar ProgramCall body is not an authorized compact scalar body");
        assert!(!rejected_adapter.job.relations().is_frozen());
    }

    #[test]
    fn production_indexed_scalar_tensor_bounded_only_has_summary_parity() {
        let protocol = generated_indexed_scalar_product_protocol(
            false,
            false,
            true,
            IndexedScalarCase::Valid,
            IndexedScalarProductKind::Tensor,
        );
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("bounded tensor plan");
        let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("bounded tensor adapter");
        let (mut compact_job, compact_roots) = adapter.lower().expect("bounded tensor lowering");
        let ProductionRoot::Compact(compact_root) = compact_roots.residual else {
            panic!("bounded Tensor scalar action must remain compact")
        };
        let compact = compact_job
            .normalize_compact_closed_root(compact_root)
            .expect("bounded Tensor compact normalization");
        let compact_evidence = full_nf_descriptor_map(&compact_job, &compact.value);
        assert!(compact_evidence.0.is_empty());
        assert!(!matches!(
            compact_evidence.1.coefficient_bound(),
            NumericContract::Known(CoefficientBound::Large)
        ));
        assert!(matches!(
            compact_evidence.1.coefficient_bound(),
            NumericContract::Known(CoefficientBound::Finite(bound))
                if !bound.maximum_absolute_coefficient.is_zero()
        ));
        assert_eq!(compact.counters.compact_scalar_holds_unmatched, 0);

        let eager_adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("bounded tensor eager adapter");
        let (mut eager_job, eager_roots) =
            eager_adapter.lower_force_eager().expect("bounded tensor eager lowering");
        let ProductionRoot::Closed(eager_root) = eager_roots.residual else { unreachable!() };
        let eager = eager_job.normalize_closed_root(eager_root).expect("bounded Tensor eager NF");
        assert_eq!(compact_evidence, full_nf_descriptor_map(&eager_job, &eager.value));
        assert_eq!(compact.counters.relation_applied, eager.counters.relation_applied);
        assert_eq!(compact.counters.relation_remaining, eager.counters.relation_remaining);
    }

    #[test]
    fn production_tensor_scalar_negative_paths_fallback_before_freeze() {
        fn tensor_expression(adapter: &ProductionAdapter<'_>, residual: &Value) -> ExprId {
            let root = match *residual {
                Value::Expr(root) => root,
                Value::Family(family) => adapter.job.programs().family_body(family).unwrap(),
            };
            let mut work = vec![root];
            while let Some(expression) = work.pop() {
                let node = adapter.job.expressions().node(expression).unwrap();
                if matches!(node.operator, ValueOperator::Matrix(MatrixOperation::Tensor { .. })) {
                    return expression;
                }
                if let ValueOperator::ProgramCall { program } = node.operator {
                    if let Some(family) = adapter.job.programs().family_for_program(program) {
                        work.push(adapter.job.programs().family_body(family).unwrap());
                        continue;
                    }
                }
                work.extend(node.inputs.iter().copied());
            }
            panic!("Tensor node missing from production residual");
        }

        for factful in [true, false] {
            let protocol = generated_indexed_scalar_product_protocol(
                false,
                false,
                false,
                IndexedScalarCase::Valid,
                // The ordinary Tensor fixture is intentionally indexed and therefore has an
                // open family body.  Use the closed non-program scalar variant for the
                // factful case so FactStore can attach a valid fact to the Tensor node; the
                // preflight rejection being tested is then specifically the virtual-node fact
                // authority, not the earlier BinderOpenExpression guard.
                if factful {
                    IndexedScalarProductKind::TensorNonProgramScalar
                } else {
                    IndexedScalarProductKind::Tensor
                },
            );
            let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("negative plan");
            let mut preflight = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
                .expect("negative adapter");
            let residual = preflight
                .resolve(plan.target().residual.clone(), Rc::new(BTreeMap::new()))
                .expect("negative residual");
            let tensor = tensor_expression(&preflight, &residual);
            if factful {
                let ResolvedValueType::Matrix(matrix) =
                    preflight.job.expressions().value_type(tensor).unwrap().clone()
                else {
                    panic!("Tensor negative fixture has non-matrix output")
                };
                preflight
                    .job
                    .insert_matrix_facts(
                        preflight.token,
                        tensor,
                        MatrixFacts::new(
                            matrix.clone(),
                            MatrixMetadata::new(MatrixLayout::row_major(
                                matrix.rows,
                                matrix.columns,
                            )),
                        ),
                    )
                    .expect("Tensor fact insertion");
            } else {
                preflight.push_relation_candidate(RelationCandidate {
                    preimage: tensor,
                    public: tensor,
                    trapdoor: tensor,
                    target: tensor,
                    family_operands: None,
                    wire: plan.target().residual.clone(),
                });
            }
            let reason = preflight
                .compact_residual_preflight(&residual)
                .expect("Tensor negative path must reject compact");
            assert_eq!(
                reason,
                if factful {
                    "virtual node has facts"
                } else {
                    "virtual node is a relation endpoint"
                }
            );
            assert!(!preflight.job.relations().is_frozen());

            let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
                .expect("Tensor eager fallback adapter");
            let (mut eager_job, eager_roots) =
                adapter.lower_force_eager().expect("Tensor eager fallback lowering");
            let ProductionRoot::Closed(eager_root) = eager_roots.residual else {
                panic!("Tensor negative path must use ordinary eager root")
            };
            let eager = eager_job
                .normalize_closed_root(eager_root)
                .expect("Tensor eager fallback normalization");
            assert!(eager.value.exact_nf.is_some());
        }
    }

    #[test]
    fn production_tensor_and_standalone_scalar_unauthorized_edges_fallback_before_freeze() {
        for (product_kind, reason_fragment) in [
            (
                IndexedScalarProductKind::TensorNonMultiplySibling,
                "tensor scalar consumer requires Matrix::Multiply sibling",
            ),
            (
                IndexedScalarProductKind::Standalone,
                "indexed scalar slice is not an authorized compact body",
            ),
        ] {
            let protocol = generated_indexed_scalar_product_protocol(
                false,
                false,
                false,
                IndexedScalarCase::Valid,
                product_kind,
            );
            let plan = ProtocolPlan::build(&protocol, "toy-threshold")
                .expect("unauthorized scalar edge plan");
            let mut preflight = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
                .expect("unauthorized scalar edge adapter");
            let residual = preflight
                .resolve(plan.target().residual.clone(), Rc::new(BTreeMap::new()))
                .expect("unauthorized scalar edge residual");
            let reason = preflight
                .compact_residual_preflight(&residual)
                .expect("unauthorized scalar edge must reject compact");
            assert!(
                reason.contains(reason_fragment),
                "unexpected unauthorized scalar edge reason: {reason}"
            );
            assert!(!preflight.job.relations().is_frozen());

            let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
                .expect("unauthorized scalar eager adapter");
            let (mut eager_job, eager_roots) =
                adapter.lower().expect("unauthorized scalar eager fallback");
            let ProductionRoot::Closed(eager_root) = eager_roots.residual else {
                panic!("unauthorized scalar edge must select the ordinary closed root")
            };
            let eager = eager_job
                .normalize_closed_root(eager_root)
                .expect("unauthorized scalar eager normalization");
            let evidence = full_nf_descriptor_map(&eager_job, &eager.value);
            assert!(!evidence.0.is_empty());
            assert!(!matches!(evidence.1.coefficient_bound(), NumericContract::Missing));
        }
    }

    #[test]
    fn production_indexed_slice_scalar_program_call_shared_cancellation_consumes_two_tokens() {
        let protocol = generated_indexed_scalar_product_protocol(
            false,
            true,
            false,
            IndexedScalarCase::Valid,
            IndexedScalarProductKind::Multiply,
        );
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("indexed plan");
        let mut preflight = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("indexed scalar preflight adapter");
        let residual = preflight
            .resolve(plan.target().residual.clone(), Rc::new(BTreeMap::new()))
            .expect("indexed scalar preflight residual");
        let compiled =
            preflight.compile_compact_root(&residual).expect("shared indexed scalar preflight");
        assert_eq!(compiled.scalar_program_calls.len(), 2);
        assert_eq!(compiled.scalar_occurrences(), 2);
        let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("indexed scalar adapter");
        let (mut compact_job, compact_roots) = adapter.lower().expect("compact lowering");
        let ProductionRoot::Compact(compact_root) = compact_roots.residual else {
            panic!("shared indexed scalar product must be compact")
        };
        let compact = compact_job
            .normalize_compact_closed_root(compact_root)
            .expect("shared indexed scalar compact normalization");
        assert_eq!(compact.counters.compact_scalar_holds_unmatched, 0);
        assert!(compact.value.exact_nf.as_ref().is_some_and(|nf| nf.is_zero()));

        let eager_adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("indexed eager adapter");
        let (mut eager_job, eager_roots) =
            eager_adapter.lower_force_eager().expect("indexed eager lowering");
        let ProductionRoot::Closed(eager_root) = eager_roots.residual else { unreachable!() };
        let eager =
            eager_job.normalize_closed_root(eager_root).expect("indexed eager normalization");
        assert_eq!(
            full_nf_descriptor_map(&compact_job, &compact.value),
            full_nf_descriptor_map(&eager_job, &eager.value)
        );
    }

    #[test]
    fn production_indexed_slice_scalar_program_call_bounded_only_has_summary_parity() {
        let protocol = generated_indexed_scalar_product_protocol(
            false,
            false,
            true,
            IndexedScalarCase::Valid,
            IndexedScalarProductKind::Multiply,
        );
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("indexed plan");
        let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("indexed bounded adapter");
        let (mut compact_job, compact_roots) = adapter.lower().expect("compact lowering");
        let ProductionRoot::Compact(compact_root) = compact_roots.residual else {
            panic!("bounded indexed scalar product must be compact")
        };
        let compact = compact_job
            .normalize_compact_closed_root(compact_root)
            .expect("bounded indexed scalar compact normalization");
        let compact_evidence = full_nf_descriptor_map(&compact_job, &compact.value);
        // A finite IndexedSlice action follows the eager bounded-only contract: its structural
        // identity is not retained as an exact Large term.
        assert!(compact_evidence.0.is_empty());
        assert!(!matches!(
            compact_evidence.1.coefficient_bound(),
            NumericContract::Known(CoefficientBound::Large)
        ));
        assert_eq!(compact.counters.compact_scalar_holds_current, 0);
        assert_eq!(compact.counters.compact_scalar_consumers, 1);
        assert_eq!(compact.counters.compact_scalar_holds_released, 0);
        assert_eq!(compact.counters.compact_scalar_holds_unmatched, 0);

        let eager_adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("indexed bounded eager adapter");
        let (mut eager_job, eager_roots) =
            eager_adapter.lower_force_eager().expect("indexed bounded eager lowering");
        let ProductionRoot::Closed(eager_root) = eager_roots.residual else { unreachable!() };
        let eager = eager_job
            .normalize_closed_root(eager_root)
            .expect("bounded indexed eager normalization");
        assert_eq!(compact.counters.relation_applied, eager.counters.relation_applied);
        assert_eq!(compact.counters.relation_remaining, eager.counters.relation_remaining);
        assert_eq!(compact_evidence, full_nf_descriptor_map(&eager_job, &eager.value));
    }

    #[test]
    fn production_indexed_slice_scalar_program_call_malformed_cases_fallback_pre_freeze() {
        for case in [
            IndexedScalarCase::NestedSource,
            IndexedScalarCase::UnauthorizedGaussianSource,
            IndexedScalarCase::WrongBinding,
            IndexedScalarCase::OutOfRange,
        ] {
            let protocol = generated_indexed_scalar_product_protocol(
                false,
                false,
                false,
                case,
                IndexedScalarProductKind::Multiply,
            );
            let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("indexed plan");
            let mut preflight = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
                .expect("indexed malformed preflight adapter");
            let resolved =
                preflight.resolve(plan.target().residual.clone(), Rc::new(BTreeMap::new()));
            if matches!(case, IndexedScalarCase::OutOfRange) {
                let error = resolved.expect_err("out-of-range slice must reject during resolve");
                assert!(format!("{error:?}").contains("escapes input extent"));
                assert!(!preflight.job.relations().is_frozen());
                continue;
            }
            let residual = resolved.expect("indexed malformed residual");
            let reason = preflight
                .compact_residual_preflight(&residual)
                .expect("malformed indexed scalar must reject compact preflight");
            assert!(!preflight.job.relations().is_frozen());
            assert!(
                reason.contains("indexed scalar") || reason.contains("virtual matrix"),
                "unexpected indexed malformed reason for {case:?}: {reason}"
            );
            let result = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
                .expect("indexed malformed eager adapter")
                .lower();
            let (mut job, roots) = result.expect("valid malformed graph must eager-fallback");
            assert!(matches!(roots.residual, ProductionRoot::Closed(_)));
            let ProductionRoot::Closed(root) = roots.residual else { unreachable!() };
            let eager = job.normalize_closed_root(root).expect("eager fallback normalization");
            let evidence = full_nf_descriptor_map(&job, &eager.value);
            assert!(!evidence.0.is_empty(), "eager fallback must retain the indexed product");
        }
    }

    #[test]
    fn production_bounded_one_sided_scalar_action_has_summary_parity() {
        let protocol = generated_product_protocol(false, false, true, true);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").unwrap();
        let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
        let (mut compact_job, compact_roots) = adapter.lower().unwrap();
        let ProductionRoot::Compact(compact_root) = compact_roots.residual else {
            panic!("bounded scalar product must be compact")
        };
        let compact = compact_job.normalize_compact_closed_root(compact_root).unwrap();
        let compact_evidence = full_nf_descriptor_map(&compact_job, &compact.value);
        assert!(compact_evidence.0.is_empty());
        assert!(!matches!(compact_evidence.1.coefficient_bound(), NumericContract::Missing));

        let eager_adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
        let (mut eager_job, eager_roots) = eager_adapter.lower_force_eager().unwrap();
        let ProductionRoot::Closed(eager_root) = eager_roots.residual else { unreachable!() };
        let eager = eager_job.normalize_closed_root(eager_root).unwrap();
        assert_eq!(compact_evidence, full_nf_descriptor_map(&eager_job, &eager.value));
    }

    #[test]
    fn production_shared_scalar_cancellation_consumes_two_exact_tokens() {
        let protocol = generated_product_protocol(false, true, true, false);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").unwrap();
        let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
        let (mut compact_job, compact_roots) = adapter.lower().unwrap();
        let ProductionRoot::Compact(compact_root) = compact_roots.residual else {
            panic!("shared scalar cancellation must be compact")
        };
        let compact = compact_job.normalize_compact_closed_root(compact_root).unwrap();
        assert_eq!(compact.counters.compact_scalar_consumers, 2);
        assert_eq!(compact.counters.compact_scalar_holds_released, 2);
        assert_eq!(compact.counters.compact_scalar_holds_unmatched, 0);
        assert!(compact.value.exact_nf.as_ref().is_some_and(|nf| nf.is_zero()));

        let eager_adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
        let (mut eager_job, eager_roots) = eager_adapter.lower_force_eager().unwrap();
        let ProductionRoot::Closed(eager_root) = eager_roots.residual else { unreachable!() };
        let eager = eager_job.normalize_closed_root(eager_root).unwrap();
        assert_eq!(
            full_nf_descriptor_map(&compact_job, &compact.value),
            full_nf_descriptor_map(&eager_job, &eager.value)
        );
    }

    #[test]
    fn production_scalar_action_negative_cases_reject_before_freeze_and_normalize_eagerly() {
        let cases = [
            (
                GeneratedScalarNegativeCase::Composite,
                Some("virtual matrix operator type or arity mismatch"),
                "virtual matrix operator type or arity mismatch",
            ),
            (
                GeneratedScalarNegativeCase::MixedParent,
                Some("virtual matrix operator type or arity mismatch"),
                "virtual matrix operator type or arity mismatch",
            ),
            (
                GeneratedScalarNegativeCase::MissingBound,
                Some("scalar compact factor bound is missing"),
                "scalar compact factor bound is missing",
            ),
            (
                GeneratedScalarNegativeCase::LargeBound,
                Some("scalar compact factor bound is not finite"),
                "scalar compact factor bound is not finite",
            ),
        ];
        for (case, preflight_reason, plan_reason) in cases {
            let protocol = generated_scalar_negative_protocol(case);
            let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("negative plan");
            let mut preflight = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
                .expect("negative preflight adapter");
            let residual = preflight
                .resolve(plan.target().residual.clone(), Rc::new(BTreeMap::new()))
                .expect("negative residual resolution");
            assert_eq!(
                preflight.compact_residual_preflight(&residual).as_deref(),
                preflight_reason,
                "unexpected preflight result for {case:?}"
            );
            assert!(!preflight.job.relations().is_frozen());
            let Value::Expr(root) = residual else {
                panic!("negative residual must be expression")
            };
            assert_eq!(
                preflight.build_compact_shell_plan(&Value::Expr(root)).unwrap_err(),
                plan_reason,
                "unexpected scalar plan result for {case:?}"
            );
            assert!(!preflight.job.relations().is_frozen());

            let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
                .expect("negative eager adapter");
            let (mut job, roots) = adapter.lower().expect("eager fallback lowering");
            assert!(matches!(roots.residual, ProductionRoot::Closed(_)));
            let ProductionRoot::Closed(root) = roots.residual else { unreachable!() };
            let eager = job.normalize_closed_root(root).expect("eager negative normalization");
            let evidence = full_nf_descriptor_map(&job, &eager.value);
            assert!(!matches!(evidence.2, NumericContract::Missing));
        }
    }

    #[test]
    fn production_scalar_action_incompatible_types_fail_at_unfrozen_arena_boundary() {
        let cases = [
            (
                "modulus",
                ResolvedMatrixType::new(BigUint::from(257_u16), 1, 1, 1).unwrap(),
                ResolvedMatrixType::new(BigUint::from(256_u16), 1, 2, 2).unwrap(),
            ),
            (
                "ring dimension",
                ResolvedMatrixType::new(BigUint::from(256_u16), 2, 1, 1).unwrap(),
                ResolvedMatrixType::new(BigUint::from(256_u16), 1, 2, 2).unwrap(),
            ),
            (
                "output geometry",
                ResolvedMatrixType::new(BigUint::from(256_u16), 1, 2, 3).unwrap(),
                ResolvedMatrixType::new(BigUint::from(256_u16), 1, 4, 2).unwrap(),
            ),
            (
                "both-scalar modulus",
                ResolvedMatrixType::new(BigUint::from(257_u16), 1, 1, 1).unwrap(),
                ResolvedMatrixType::new(BigUint::from(256_u16), 1, 1, 1).unwrap(),
            ),
            (
                "both-scalar ring dimension",
                ResolvedMatrixType::new(BigUint::from(256_u16), 2, 1, 1).unwrap(),
                ResolvedMatrixType::new(BigUint::from(256_u16), 1, 1, 1).unwrap(),
            ),
        ];
        for (kind, left_type, right_type) in cases {
            let protocol = generated_gather_protocol(1);
            let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("type plan");
            let mut adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
                .expect("type-boundary adapter");
            let left = adapter
                .job
                .expressions_mut()
                .intern(
                    ValueOperator::Sampler {
                        event: SampleEventId(91_001),
                        operation: SamplerOperation::UniformResidue { output: left_type },
                    },
                    Box::new([]),
                )
                .expect("left type-boundary leaf");
            let right = adapter
                .job
                .expressions_mut()
                .intern(
                    ValueOperator::Sampler {
                        event: SampleEventId(91_002),
                        operation: SamplerOperation::UniformResidue { output: right_type },
                    },
                    Box::new([]),
                )
                .expect("right type-boundary leaf");
            let error = adapter
                .job
                .expressions_mut()
                .intern(ValueOperator::Matrix(MatrixOperation::Multiply), Box::new([left, right]));
            // These contracts are rejected while constructing the authoritative expression, so
            // there is no valid Graph-IR root to resolve or eagerly normalize. Keep the adapter
            // unfrozen assertion below to prove the failure cannot partially register facts or
            // relations before the earlier type boundary reports it.
            assert_eq!(
                error,
                Err(super::super::arena::ArenaError::IncompatibleMatrixTypes),
                "{kind}"
            );
            assert!(!adapter.job.relations().is_frozen());
        }
    }

    #[test]
    fn production_gadget_product_uses_dedicated_compact_frame_and_releases_hold() {
        let protocol = generated_gadget_product_protocol(false, GeneratedGadgetProductCase::Single);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").unwrap();
        let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
        let (mut compact_job, compact_roots) = adapter.lower().unwrap();
        let ProductionRoot::Compact(compact_root) = compact_roots.residual else {
            panic!("authorized gadget product must be compact")
        };
        let compact = compact_job.normalize_compact_closed_root(compact_root).unwrap();
        assert!(compact.counters.compact_strict_products > 0);
        assert_eq!(compact.counters.compact_planned_unique_shells, 1);
        assert_eq!(compact.counters.compact_planned_shell_occurrences, 1);
        assert_eq!(compact.counters.compact_shell_allocated, 1);
        assert_eq!(compact.counters.compact_shell_holds_released, 1);
        assert_eq!(compact.counters.compact_shell_holds_unmatched, 0);

        let eager_adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
        let (mut eager_job, eager_roots) = eager_adapter.lower_force_eager().unwrap();
        let ProductionRoot::Closed(eager_root) = eager_roots.residual else { unreachable!() };
        let eager = eager_job.normalize_closed_root(eager_root).unwrap();
        assert_eq!(
            full_nf_descriptor_map(&compact_job, &compact.value),
            full_nf_descriptor_map(&eager_job, &eager.value)
        );

        let reverse_protocol =
            generated_gadget_product_protocol(true, GeneratedGadgetProductCase::Single);
        let reverse_plan = ProtocolPlan::build(&reverse_protocol, "toy-threshold").unwrap();
        let reverse_adapter =
            ProductionAdapter::new(&reverse_protocol, &reverse_plan, BTreeMap::new()).unwrap();
        let (_reverse_job, reverse_roots) = reverse_adapter.lower().unwrap();
        assert!(matches!(reverse_roots.residual, ProductionRoot::Closed(_)));
    }

    #[test]
    fn production_gadget_explicit_reducible_input_is_rejected_before_materialization() {
        let protocol = generated_gadget_product_protocol(false, GeneratedGadgetProductCase::Single);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").unwrap();
        let mut adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
        let residual =
            adapter.resolve(plan.target().residual.clone(), Rc::new(BTreeMap::new())).unwrap();
        let Value::Expr(root) = residual else { unreachable!() };
        let ValueOperator::ProgramCall { program } =
            adapter.job.expressions().node(root).unwrap().operator
        else {
            unreachable!()
        };
        let family = adapter.job.programs().family_for_program(program).unwrap();
        let body = adapter.job.programs().family_body(family).unwrap();
        let product = adapter.job.expressions().node(body).unwrap().clone();
        let [gadget, decomposition] = product.inputs.as_ref() else { unreachable!() };
        let decomposition_node = adapter.job.expressions().node(*decomposition).unwrap().clone();
        let input = decomposition_node.inputs[0];
        let input_type = adapter.job.expressions().value_type(input).unwrap().clone();
        let nested_family =
            adapter.generated_family(FamilyDomain::new(0, 1).unwrap(), input).unwrap();
        let selector = adapter.intern_index_constant(BigInt::ZERO).unwrap();
        let nested_call = adapter
            .call_family_in_program_scope_deferred_generated(
                nested_family,
                selector,
                TrustedIndexRange::new(0, 1).unwrap(),
            )
            .unwrap();
        let explicit = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::ExplicitElement {
                    domain: FamilyDomain::new(0, 1).unwrap(),
                    element_type: input_type,
                },
                Box::new([selector, nested_call]),
            )
            .unwrap();
        let ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
            output,
            base,
            small,
            digit_count,
        }) = decomposition_node.operator
        else {
            unreachable!()
        };
        let wrapped_decomposition = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
                    output,
                    base,
                    small,
                    digit_count,
                }),
                Box::new([explicit]),
            )
            .unwrap();
        let wrapped_product = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Matrix(MatrixOperation::Multiply),
                Box::new([*gadget, wrapped_decomposition]),
            )
            .unwrap();
        let wrapped_family =
            adapter.generated_family(FamilyDomain::new(0, 1).unwrap(), wrapped_product).unwrap();
        let wrapped_root = adapter
            .call_family_in_program_scope_deferred_generated(
                wrapped_family,
                selector,
                TrustedIndexRange::new(0, 1).unwrap(),
            )
            .unwrap();
        assert_eq!(
            adapter.compile_compact_root(&Value::Expr(wrapped_root)).unwrap_err(),
            "gadget decomposition input contains a reducible generated call"
        );
        assert!(!adapter.job.relations().is_frozen());

        adapter.register_reached_relations().unwrap();
        let eager = adapter
            .job
            .materialize_reducible_generated_calls_with_reason(
                wrapped_root,
                BetaReason::ResidualRoot,
            )
            .unwrap();
        adapter.job.freeze_relations(adapter.token).unwrap();
        let closed = adapter
            .close_root(
                Value::Expr(eager),
                &plan.target().residual,
                "explicit reducible gadget eager root",
                false,
            )
            .unwrap();
        let ProductionRoot::Closed(closed) = closed else { unreachable!() };
        let normalized = adapter.job.normalize_closed_root(closed).unwrap();
        assert!(normalized.value.exact_nf.is_some());
    }

    #[test]
    fn production_standalone_gadget_decomposition_is_rejected_before_freeze() {
        let protocol =
            generated_gadget_product_protocol(false, GeneratedGadgetProductCase::Standalone);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").unwrap();
        let mut adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
        let residual =
            adapter.resolve(plan.target().residual.clone(), Rc::new(BTreeMap::new())).unwrap();
        assert_eq!(
            adapter.compact_residual_preflight(&residual).as_deref(),
            Some("gadget decomposition has no authorized product consumer")
        );
        let Value::Expr(root) = residual else { unreachable!() };
        assert_eq!(
            adapter.build_compact_shell_plan(&Value::Expr(root)).unwrap_err(),
            "gadget decomposition has no authorized product consumer"
        );
        assert!(!adapter.job.relations().is_frozen());
    }

    #[test]
    fn production_gadget_sum_consumes_one_boundary_hold_after_all_splices() {
        let protocol = generated_gadget_product_protocol(false, GeneratedGadgetProductCase::Sum);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").unwrap();
        let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
        let (mut compact_job, compact_roots) = adapter.lower().unwrap();
        let ProductionRoot::Compact(compact_root) = compact_roots.residual else {
            panic!("gadget sum product must be compact")
        };
        let compact = compact_job.normalize_compact_closed_root(compact_root).unwrap();
        assert_eq!(compact.counters.compact_planned_shell_occurrences, 1);
        assert_eq!(compact.counters.compact_shell_holds_released, 1);
        assert_eq!(compact.counters.compact_shell_holds_unmatched, 0);

        let eager_adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
        let (mut eager_job, eager_roots) = eager_adapter.lower_force_eager().unwrap();
        let ProductionRoot::Closed(eager_root) = eager_roots.residual else { unreachable!() };
        let eager = eager_job.normalize_closed_root(eager_root).unwrap();
        assert_eq!(
            full_nf_descriptor_map(&compact_job, &compact.value),
            full_nf_descriptor_map(&eager_job, &eager.value)
        );
    }

    #[test]
    fn production_shared_gadget_shell_has_two_boundary_tokens_and_one_materialization() {
        let protocol = generated_gadget_product_protocol(false, GeneratedGadgetProductCase::Shared);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").unwrap();
        let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
        let (mut compact_job, compact_roots) = adapter.lower().unwrap();
        let ProductionRoot::Compact(compact_root) = compact_roots.residual else {
            panic!("shared gadget products must be compact")
        };
        let compact = compact_job.normalize_compact_closed_root(compact_root).unwrap();
        assert_eq!(compact.counters.compact_planned_unique_shells, 1);
        assert_eq!(compact.counters.compact_planned_shell_occurrences, 2);
        assert_eq!(compact.counters.compact_shell_allocated, 1);
        assert_eq!(compact.counters.compact_shell_holds_released, 2);
        assert_eq!(compact.counters.compact_shell_holds_unmatched, 0);

        let eager_adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
        let (mut eager_job, eager_roots) = eager_adapter.lower_force_eager().unwrap();
        let ProductionRoot::Closed(eager_root) = eager_roots.residual else { unreachable!() };
        let eager = eager_job.normalize_closed_root(eager_root).unwrap();
        assert_eq!(
            full_nf_descriptor_map(&compact_job, &compact.value),
            full_nf_descriptor_map(&eager_job, &eager.value)
        );
    }

    #[test]
    fn production_gadget_shell_with_non_gadget_consumer_is_eager_before_freeze() {
        let protocol = generated_gadget_product_protocol(false, GeneratedGadgetProductCase::Mixed);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").unwrap();
        let mut preflight_adapter =
            ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
        let residual = preflight_adapter
            .resolve(plan.target().residual.clone(), Rc::new(BTreeMap::new()))
            .unwrap();
        assert_eq!(
            preflight_adapter.compact_residual_preflight(&residual).as_deref(),
            Some("gadget decomposition has no authorized product consumer")
        );
        let Value::Expr(root) = residual else { unreachable!() };
        assert_eq!(
            preflight_adapter.build_compact_shell_plan(&Value::Expr(root)).unwrap_err(),
            "gadget decomposition has no authorized product consumer"
        );
        assert!(!preflight_adapter.job.relations().is_frozen());

        let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
        let (mut job, roots) = adapter.lower().unwrap();
        assert!(matches!(roots.residual, ProductionRoot::Closed(_)));
        let ProductionRoot::Closed(root) = roots.residual else { unreachable!() };
        let eager = job.normalize_closed_root(root).unwrap();
        let evidence = full_nf_descriptor_map(&job, &eager.value);
        assert!(!evidence.0.is_empty());
        assert!(!matches!(evidence.2, NumericContract::Missing));
    }

    #[test]
    fn production_gadget_rule_mismatch_is_eager_before_freeze() {
        let protocol =
            generated_gadget_product_protocol(false, GeneratedGadgetProductCase::Mismatch);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").unwrap();
        let mut preflight_adapter =
            ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
        let residual = preflight_adapter
            .resolve(plan.target().residual.clone(), Rc::new(BTreeMap::new()))
            .unwrap();
        assert_eq!(
            preflight_adapter.compact_residual_preflight(&residual).as_deref(),
            Some("gadget decomposition consumer rule mismatch")
        );
        let Value::Expr(root) = residual else { unreachable!() };
        assert_eq!(
            preflight_adapter.build_compact_shell_plan(&Value::Expr(root)).unwrap_err(),
            "gadget decomposition consumer rule mismatch"
        );
        assert!(!preflight_adapter.job.relations().is_frozen());

        let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
        let (mut job, roots) = adapter.lower().unwrap();
        assert!(matches!(roots.residual, ProductionRoot::Closed(_)));
        let ProductionRoot::Closed(root) = roots.residual else { unreachable!() };
        let eager = job.normalize_closed_root(root).unwrap();
        let evidence = full_nf_descriptor_map(&job, &eager.value);
        assert!(!evidence.0.is_empty());
        assert!(!matches!(evidence.2, NumericContract::Missing));
    }

    #[test]
    fn production_product_cancellation_is_exact_zero_with_eager_parity() {
        let protocol = generated_product_protocol(false, true, false, false);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").unwrap();
        let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
        let (mut compact_job, compact_roots) = adapter.lower().unwrap();
        let ProductionRoot::Compact(compact_root) = compact_roots.residual else {
            panic!("distinct cancellation products must remain compact")
        };
        let compact = compact_job.normalize_compact_closed_root(compact_root).unwrap();
        assert!(compact.counters.compact_strict_products >= 2);
        assert!(compact.counters.compact_logical_scale >= 1);
        assert!(compact.value.exact_nf.as_ref().is_some_and(|nf| nf.exact_terms.is_empty()));
        let eager_adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).unwrap();
        let (mut eager_job, eager_roots) = eager_adapter.lower_force_eager().unwrap();
        let ProductionRoot::Closed(eager_root) = eager_roots.residual else { unreachable!() };
        let eager = eager_job.normalize_closed_root(eager_root).unwrap();
        assert_eq!(
            full_nf_descriptor_map(&compact_job, &compact.value),
            full_nf_descriptor_map(&eager_job, &eager.value)
        );
    }

    #[test]
    fn production_bounded_strict_product_compresses_with_finite_summary_parity() {
        let protocol = generated_product_protocol(false, false, false, false);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").unwrap();
        let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .unwrap()
            .with_test_sampler_fact_bound(2);
        let (mut compact_job, compact_roots) = adapter.lower().unwrap();
        let ProductionRoot::Compact(compact_root) = compact_roots.residual else {
            panic!("bounded product must be compact")
        };
        let compact = compact_job.normalize_compact_closed_root(compact_root).unwrap();
        assert!(compact.value.exact_nf.as_ref().unwrap().exact_terms.is_empty());

        let eager_adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .unwrap()
            .with_test_sampler_fact_bound(2);
        let (mut eager_job, eager_roots) = eager_adapter.lower_force_eager().unwrap();
        let ProductionRoot::Closed(eager_root) = eager_roots.residual else { unreachable!() };
        let eager = eager_job.normalize_closed_root(eager_root).unwrap();
        assert_eq!(
            full_nf_descriptor_map(&compact_job, &compact.value),
            full_nf_descriptor_map(&eager_job, &eager.value)
        );
    }

    #[test]
    fn relation_endpoint_virtual_add_selects_eager_before_freeze() {
        let protocol = generated_direct_family_protocol();
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("protocol plan");
        let mut adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("relation-boundary adapter");
        let residual = adapter
            .resolve(plan.target().residual.clone(), Rc::new(BTreeMap::new()))
            .expect("residual resolution");
        let Value::Expr(root) = residual else {
            panic!("generated residual must be an expression")
        };
        let ValueOperator::ProgramCall { program } =
            adapter.job.expressions().node(root).expect("root expression").operator
        else {
            panic!("generated residual root must be a family call")
        };
        let family =
            adapter.job.programs().family_for_program(program).expect("generated root family");
        let body = adapter.job.programs().family_body(family).expect("generated family body");
        assert!(matches!(
            adapter.job.expressions().node(body).expect("generated family body node").operator,
            ValueOperator::Matrix(MatrixOperation::Add)
        ));
        adapter.push_relation_candidate(RelationCandidate {
            preimage: body,
            public: body,
            trapdoor: body,
            target: body,
            family_operands: None,
            wire: plan.target().residual.clone(),
        });
        let reason = adapter
            .compact_residual_preflight(&Value::Expr(root))
            .expect("relation endpoint must force eager selection");
        assert_eq!(reason, "virtual node is a relation endpoint");
        assert!(!adapter.job.relations().is_frozen());
        let eager = adapter
            .materialize_root_value_with_reason(Value::Expr(root), BetaReason::ResidualRoot)
            .expect("relation-boundary eager materialization");
        let closed = adapter
            .close_root(eager, &plan.target().residual, "relation-boundary eager root", false)
            .expect("relation-boundary eager root closure");
        assert!(matches!(closed, ProductionRoot::Closed(_)));
        adapter.job.freeze_relations(adapter.token).unwrap();
    }

    #[test]
    fn composite_scalar_index_binding_stays_compact_without_scalar_matrix_atom() {
        let protocol = generated_composite_binding_protocol();
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("composite plan");
        let adapter =
            ProductionAdapter::new(&protocol, &plan, BTreeMap::new()).expect("composite adapter");
        let (mut compact_job, compact_roots) = adapter.lower().expect("composite lowering");
        let ProductionRoot::Compact(compact_root) = compact_roots.residual else {
            panic!("composite scalar/index binding must remain compact")
        };
        let compact = compact_job
            .normalize_compact_closed_root(compact_root)
            .expect("composite compact normalization");
        assert!(compact.counters.compact_virtual_calls > 0);
        let (terms, _, _) = full_nf_descriptor_map(&compact_job, &compact.value);
        assert!(
            terms
                .keys()
                .flat_map(|(central, ordered)| central.iter().chain(ordered))
                .all(|fingerprint| !fingerprint.contains("operator=Scalar(")),
        );

        let eager_adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("forced-eager composite adapter");
        let (mut eager_job, eager_roots) =
            eager_adapter.lower_force_eager().expect("forced-eager composite lowering");
        let ProductionRoot::Closed(eager_root) = eager_roots.residual else { unreachable!() };
        let eager = eager_job
            .normalize_closed_root(eager_root)
            .expect("forced-eager composite normalization");
        assert_eq!(
            full_nf_descriptor_map(&compact_job, &compact.value),
            full_nf_descriptor_map(&eager_job, &eager.value)
        );
    }

    #[test]
    fn compact_binding_foreign_authority_and_wrong_type_fail_closed_without_partial_state() {
        let protocol = crate::toy_example::protocol();
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("toy plan");
        let mut adapter = ProductionAdapter::new(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .expect("binding authority adapter");
        let matrix = ResolvedMatrixType::new(BigUint::from(257_u16), 1, 1, 1).unwrap();
        let body = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(77_001),
                    operation: SamplerOperation::UniformResidue { output: matrix.clone() },
                },
                Box::new([]),
            )
            .expect("binding body");
        let family = adapter.generated_family(FamilyDomain::new(0, 2).unwrap(), body).unwrap();
        let mut foreign_expressions = super::super::arena::ExprArena::new();
        let foreign_index = foreign_expressions
            .intern(ValueOperator::Constant(TypedConstant::int(0)), Box::new([]))
            .expect("foreign index");
        let before_foreign = (
            adapter.job.expressions().node_count(),
            adapter.job.programs().len(),
            adapter.job.facts().len(),
        );
        let foreign_error = adapter
            .call_family_with_resolved_range(
                family,
                foreign_index,
                TrustedIndexRange::new(0, 2).unwrap(),
            )
            .expect_err("foreign binding authority must fail closed");
        let ProductionAdapterError::Arena(super::super::arena::ArenaError::ForeignExpression {
            expected,
            actual,
        }) = foreign_error
        else {
            panic!("unexpected foreign binding error: {foreign_error:?}")
        };
        assert_eq!(expected, adapter.job.expressions().token());
        assert_eq!(actual, foreign_expressions.token());
        assert_eq!(
            before_foreign,
            (
                adapter.job.expressions().node_count(),
                adapter.job.programs().len(),
                adapter.job.facts().len(),
            )
        );
        assert!(!adapter.job.relations().is_frozen());

        let wrong_matrix = ResolvedMatrixType::new(BigUint::from(257_u16), 1, 1, 1).unwrap();
        let wrong_type = adapter
            .job
            .expressions_mut()
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(77_002),
                    operation: SamplerOperation::UniformResidue { output: wrong_matrix.clone() },
                },
                Box::new([]),
            )
            .expect("wrong-type binding");
        let before_wrong_type = (
            adapter.job.expressions().node_count(),
            adapter.job.programs().len(),
            adapter.job.facts().len(),
        );
        let wrong_type_error = adapter
            .call_family_with_resolved_range(
                family,
                wrong_type,
                TrustedIndexRange::new(0, 2).unwrap(),
            )
            .expect_err("wrong binding type must fail closed");
        let ProductionAdapterError::Arena(super::super::arena::ArenaError::TypeMismatch {
            operator,
            position,
            expected,
            actual,
        }) = wrong_type_error
        else {
            panic!("unexpected wrong-type binding error: {wrong_type_error:?}")
        };
        assert_eq!(operator, "FamilyCall");
        assert_eq!(position, 0);
        assert_eq!(expected, ResolvedValueType::Int);
        assert_eq!(actual, ResolvedValueType::Matrix(wrong_matrix));
        assert_eq!(
            before_wrong_type,
            (
                adapter.job.expressions().node_count(),
                adapter.job.programs().len(),
                adapter.job.facts().len(),
            )
        );
        assert!(!adapter.job.relations().is_frozen());
    }

    #[test]
    fn deep_compact_sampler_chain_normalizes_iteratively_with_bounded_live_state() {
        let depth = 20_000;
        let compact_handle = std::thread::Builder::new()
            .name("compact-small-stack".to_owned())
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                let protocol = generated_deep_compact_protocol(depth);
                let plan =
                    ProtocolPlan::build(&protocol, "toy-threshold").expect("deep compact plan");
                let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
                    .expect("deep compact adapter");
                let (mut compact_job, compact_roots) =
                    adapter.lower().expect("deep compact lowering");
                let ProductionRoot::Compact(compact_root) = compact_roots.residual else {
                    panic!("deep sampler chain must remain compact")
                };
                let compact = compact_job
                    .normalize_compact_closed_root(compact_root)
                    .expect("deep compact normalization");
                let compact_nf = full_nf_descriptor_map(&compact_job, &compact.value);

                let eager_protocol = generated_deep_compact_protocol(depth);
                let eager_plan =
                    ProtocolPlan::build(&eager_protocol, "toy-threshold").expect("eager plan");
                let eager_adapter =
                    ProductionAdapter::new(&eager_protocol, &eager_plan, BTreeMap::new())
                        .expect("forced-eager deep adapter");
                let (mut eager_job, eager_roots) =
                    eager_adapter.lower_force_eager().expect("forced-eager deep lowering");
                let ProductionRoot::Closed(eager_root) = eager_roots.residual else {
                    unreachable!()
                };
                let eager = eager_job
                    .normalize_closed_root(eager_root)
                    .expect("forced-eager normalization");
                let eager_nf = full_nf_descriptor_map(&eager_job, &eager.value);
                (compact_nf, compact.counters, eager_nf)
            })
            .expect("small-stack normalization thread");
        let (compact_nf, counters, eager_nf) =
            compact_handle.join().expect("compact normalization panicked");
        assert!(counters.compact_virtual_calls > 0);
        assert!(counters.compact_max_frames <= depth as u64 + 1);
        assert!(counters.compact_peak_live_values <= 1);
        assert_eq!(counters.compact_live_frames, 0);
        assert_eq!(counters.compact_live_values, 0);
        assert_eq!(compact_nf, eager_nf);
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

        let protocol = generated_slice_protocol(7, GeneratedSliceCase::Static);
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
    fn production_fixed_slice_hash_compact_matches_eager_for_multiple_geometries() {
        for (
            source_rows,
            source_columns,
            row_start,
            row_end,
            column_start,
            column_end,
            shared_slice,
        ) in [
            (1, 8720, 0, 1, 0, 80, false),
            (4, 23, 1, 3, 7, 19, false),
            (1, 8720, 0, 1, 0, 80, true),
        ] {
            let protocol = generated_fixed_slice_hash_protocol(
                source_rows,
                source_columns,
                row_start,
                row_end,
                column_start,
                column_end,
                FixedSliceInput::Hash,
                shared_slice,
            );
            let plan = ProtocolPlan::build(&protocol, "fixed-slice")
                .expect("fixed slice hash protocol plan");
            let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
                .expect("fixed slice hash adapter");
            let (mut compact_job, compact_roots) =
                adapter.lower().expect("fixed slice hash compact lowering");
            let ProductionRoot::CompactFamily(compact_family) = compact_roots.residual else {
                panic!("fixed deterministic hash Slice must remain compact")
            };
            let (compact_value, compact_counters) = compact_job
                .normalize_compact_family(compact_family)
                .expect("fixed slice hash compact analysis");

            let eager_adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
                .expect("fixed slice hash eager adapter");
            let (mut eager_job, eager_roots) =
                eager_adapter.lower_force_eager().expect("fixed slice hash eager lowering");
            let ProductionRoot::Family(eager_family) = eager_roots.residual else {
                panic!("forced eager fixed slice hash must remain an ordinary family")
            };
            let eager_root = eager_job
                .programs()
                .root(&eager_job.expressions(), eager_family.program())
                .expect("fixed slice eager family root");
            let eager_materialized = eager_job
                .materialize_reducible_generated_calls(eager_root.expression())
                .expect("fixed slice eager family materialization");
            let eager_scoped = eager_job
                .programs()
                .detached_scoped(
                    &eager_job.expressions(),
                    eager_family.program(),
                    eager_materialized,
                )
                .expect("fixed slice eager family scope");
            let (eager_value, eager_counters) =
                eager_job.normalize(eager_scoped).expect("fixed slice hash eager analysis");
            assert_eq!(
                full_nf_descriptor_map(&compact_job, &compact_value),
                full_nf_descriptor_map(&eager_job, &eager_value)
            );
            assert_eq!(compact_counters.relation_applied, eager_counters.relation_applied);
            assert_eq!(compact_counters.relation_remaining, eager_counters.relation_remaining);
            assert!(compact_counters.nodes_processed > 0);
            assert_eq!(compact_counters.nodes_total, compact_counters.nodes_processed);
        }
    }

    #[test]
    fn production_shared_virtual_algebra_compact_matches_forced_eager() {
        let protocol = generated_shared_virtual_dag_protocol();
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("protocol plan");
        let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("shared-DAG production adapter");
        let (mut compact_job, compact_roots) = adapter.lower().expect("compact lowering");
        let (compact_value, compact_counters) = match compact_roots.residual {
            ProductionRoot::Compact(root) => {
                let analysis = compact_job
                    .normalize_compact_closed_root(root)
                    .expect("shared compact normalization");
                (analysis.value, analysis.counters)
            }
            ProductionRoot::CompactFamily(family) => compact_job
                .normalize_compact_family(family)
                .expect("shared compact family normalization"),
            ProductionRoot::Closed(_) | ProductionRoot::Family(_) => {
                panic!("shared virtual algebra must remain compact")
            }
        };
        let compact_evidence = full_nf_descriptor_map(&compact_job, &compact_value);

        let eager_adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("shared-DAG eager adapter");
        let (mut eager_job, eager_roots) =
            eager_adapter.lower_force_eager().expect("forced-eager lowering");
        let ProductionRoot::Closed(eager_root) = eager_roots.residual else {
            panic!("forced-eager shared virtual algebra must be closed")
        };
        let eager =
            eager_job.normalize_closed_root(eager_root).expect("forced-eager shared normalization");
        assert_eq!(compact_evidence, full_nf_descriptor_map(&eager_job, &eager.value));
        assert!(compact_counters.nodes_processed > 0);
        assert_eq!(compact_counters.nodes_total, compact_counters.nodes_processed);
        assert_eq!(compact_counters.relation_applied, eager.counters.relation_applied);
        assert_eq!(compact_counters.relation_remaining, eager.counters.relation_remaining);
    }

    #[test]
    fn compact_compiler_shared_diamond_reports_occurrences_from_unique_states() {
        let mut counts = Vec::new();
        for depth in 1..=6 {
            let protocol = generated_shared_virtual_dag_protocol_with_depth(depth);
            let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("protocol plan");
            let mut adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
                .expect("shared-DAG adapter");
            let residual = adapter
                .resolve(plan.target().residual.clone(), Rc::new(BTreeMap::new()))
                .expect("shared-DAG residual");
            let compiled =
                adapter.compile_compact_root(&residual).expect("shared-DAG compact compiler");
            counts.push(compiled.preflight_node_occurrences);
        }
        let expected = (1..=6).map(|depth| 3 * (1_u64 << depth) + 1).collect::<Vec<_>>();
        assert_eq!(counts, expected);
    }

    #[test]
    fn compact_compiler_shared_diamond_overflow_falls_back_before_freeze() {
        let protocol = generated_shared_virtual_dag_protocol_with_depth(128);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("protocol plan");
        let mut adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("shared-DAG overflow adapter");
        let residual = adapter
            .resolve(plan.target().residual.clone(), Rc::new(BTreeMap::new()))
            .expect("shared-DAG overflow residual");
        let error = adapter
            .compile_compact_root(&residual)
            .expect_err("exponential occurrence count must overflow");
        assert_eq!(error, "compact occurrence multiplicity overflow");
        assert!(!adapter.job.relations().is_frozen());
    }

    #[test]
    fn compact_compiler_topological_multiplicity_handles_cross_sibling_edges() {
        let protocol = generated_forward_edge_virtual_dag_protocol();
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("protocol plan");
        let mut adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("forward-edge adapter");
        let residual = adapter
            .resolve(plan.target().residual.clone(), Rc::new(BTreeMap::new()))
            .expect("forward-edge residual");
        let compiled =
            adapter.compile_compact_root(&residual).expect("forward-edge compact compiler");
        assert_eq!(compiled.preflight_node_occurrences, 10);
        assert!(!adapter.job.relations().is_frozen());

        let compact_adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("forward-edge compact runtime adapter");
        let (mut compact_job, compact_roots) = compact_adapter.lower().expect("compact lowering");
        let (compact_value, compact_counters) = match compact_roots.residual {
            ProductionRoot::Compact(root) => {
                let analysis = compact_job
                    .normalize_compact_closed_root(root)
                    .expect("forward-edge compact normalization");
                (analysis.value, analysis.counters)
            }
            ProductionRoot::CompactFamily(family) => compact_job
                .normalize_compact_family(family)
                .expect("forward-edge compact family normalization"),
            ProductionRoot::Closed(_) | ProductionRoot::Family(_) => {
                panic!("forward-edge virtual DAG must remain compact")
            }
        };
        assert_eq!(compact_counters.compact_shell_holds_unmatched, 0);
        assert_eq!(compact_counters.compact_scalar_holds_unmatched, 0);

        let eager_adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("forward-edge eager runtime adapter");
        let (mut eager_job, eager_roots) =
            eager_adapter.lower_force_eager().expect("forward-edge eager lowering");
        let ProductionRoot::Closed(eager_root) = eager_roots.residual else {
            panic!("forward-edge eager virtual DAG must be closed")
        };
        let eager = eager_job.normalize_closed_root(eager_root).expect("forward-edge eager norm");
        assert_eq!(
            full_nf_descriptor_map(&compact_job, &compact_value),
            full_nf_descriptor_map(&eager_job, &eager.value)
        );
        assert_eq!(compact_counters.relation_applied, eager.counters.relation_applied);
        assert_eq!(compact_counters.relation_remaining, eager.counters.relation_remaining);
    }

    #[test]
    fn production_fixed_slice_hash_open_tag_falls_back_before_freeze() {
        let protocol = generated_fixed_slice_hash_protocol(
            1,
            8720,
            0,
            1,
            0,
            80,
            FixedSliceInput::DynamicHashTag,
            false,
        );
        let plan = ProtocolPlan::build(&protocol, "fixed-slice").expect("fixed slice tag plan");
        let mut preflight = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("fixed slice tag adapter");
        let residual = preflight
            .resolve(plan.target().residual.clone(), Rc::new(BTreeMap::new()))
            .expect("fixed slice tag residual");
        let reason = preflight
            .compact_residual_preflight(&residual)
            .expect("open hash tag must reject compact preflight");
        assert_eq!(reason, "fixed Slice hash has unsupported tags");
        assert!(!preflight.job.relations().is_frozen());
        let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("fixed slice tag eager adapter");
        let (mut eager_job, eager_roots) =
            adapter.lower_force_eager().expect("fixed slice tag eager");
        let ProductionRoot::Family(eager_family) = eager_roots.residual else {
            panic!("fixed slice tag eager family marker")
        };
        let eager =
            eager_job.analyze_family_root(eager_family).expect("fixed slice tag eager analysis");
        assert!(eager.exact_term_count > 0);
    }

    #[test]
    fn production_fixed_slice_sampler_add_compact_matches_eager() {
        let protocol = generated_fixed_slice_hash_protocol(
            1,
            8720,
            0,
            1,
            0,
            80,
            FixedSliceInput::Sampler,
            false,
        );
        let plan = ProtocolPlan::build(&protocol, "fixed-slice").expect("sampler slice plan");
        let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("sampler slice adapter");
        let (mut compact_job, compact_roots) =
            adapter.lower().expect("sampler slice compact lowering");
        let ProductionRoot::CompactFamily(compact_family) = compact_roots.residual else {
            panic!("closed Sampler fixed Slice under Add must remain compact")
        };
        let (compact_value, compact_counters) = compact_job
            .normalize_compact_family(compact_family)
            .expect("sampler slice compact analysis");

        let eager_adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
            .expect("sampler slice eager adapter");
        let (mut eager_job, eager_roots) =
            eager_adapter.lower_force_eager().expect("sampler slice eager lowering");
        let ProductionRoot::Family(eager_family) = eager_roots.residual else {
            panic!("forced eager sampler slice must remain an ordinary family")
        };
        let eager_root = eager_job
            .programs()
            .root(&eager_job.expressions(), eager_family.program())
            .expect("sampler slice eager family root");
        let eager_materialized = eager_job
            .materialize_reducible_generated_calls(eager_root.expression())
            .expect("sampler slice eager family materialization");
        let eager_scoped = eager_job
            .programs()
            .detached_scoped(&eager_job.expressions(), eager_family.program(), eager_materialized)
            .expect("sampler slice eager family scope");
        let (eager_value, eager_counters) =
            eager_job.normalize(eager_scoped).expect("sampler slice eager analysis");

        assert_eq!(
            full_nf_descriptor_map(&compact_job, &compact_value),
            full_nf_descriptor_map(&eager_job, &eager_value),
            "source-agnostic Slice under Add must preserve exact terms and bound"
        );
        assert_eq!(compact_counters.relation_applied, eager_counters.relation_applied);
        assert_eq!(compact_counters.relation_remaining, eager_counters.relation_remaining);
        assert!(compact_counters.nodes_processed > 0);
        assert_eq!(compact_counters.nodes_total, compact_counters.nodes_processed);
    }

    #[test]
    fn production_fixed_slice_negative_paths_fallback_before_freeze() {
        #[derive(Clone, Copy)]
        enum NegativeCase {
            UnsupportedStructuralChild,
            FactfulSlice,
            RelationEndpoint,
        }

        for (case, expected_reason) in [
            (
                NegativeCase::UnsupportedStructuralChild,
                "generated body contains unsupported concrete operator",
            ),
            (NegativeCase::FactfulSlice, "fixed Slice node has facts"),
            (NegativeCase::RelationEndpoint, "fixed Slice node is a relation endpoint"),
        ] {
            let input_kind = if matches!(case, NegativeCase::UnsupportedStructuralChild) {
                FixedSliceInput::UnsupportedTranspose
            } else {
                FixedSliceInput::Hash
            };
            let protocol = generated_fixed_slice_hash_protocol(
                if matches!(case, NegativeCase::UnsupportedStructuralChild) { 80 } else { 1 },
                if matches!(case, NegativeCase::UnsupportedStructuralChild) { 1 } else { 8720 },
                0,
                1,
                0,
                if matches!(case, NegativeCase::UnsupportedStructuralChild) { 80 } else { 80 },
                input_kind,
                false,
            );
            let plan = ProtocolPlan::build(&protocol, "fixed-slice")
                .expect("fixed slice negative protocol plan");
            let mut preflight = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
                .expect("fixed slice negative preflight adapter");
            let residual = preflight
                .resolve(plan.target().residual.clone(), Rc::new(BTreeMap::new()))
                .expect("fixed slice negative residual");
            let Value::Family(family) = residual else {
                panic!("fixed slice negative residual must remain a family")
            };
            let body = preflight
                .job
                .programs()
                .family_body(family)
                .expect("fixed slice negative family body");
            if matches!(case, NegativeCase::FactfulSlice) {
                let ResolvedValueType::Matrix(matrix) = preflight
                    .job
                    .expressions()
                    .value_type(body)
                    .expect("fixed slice negative body type")
                    .clone()
                else {
                    panic!("fixed slice negative body must be a matrix")
                };
                let facts = MatrixFacts::new(
                    matrix.clone(),
                    MatrixMetadata::new(MatrixLayout::row_major(matrix.rows, matrix.columns)),
                );
                preflight
                    .job
                    .insert_matrix_facts(preflight.token, body, facts)
                    .expect("fixed slice facts insertion");
            }
            if matches!(case, NegativeCase::RelationEndpoint) {
                preflight.push_relation_candidate(RelationCandidate {
                    preimage: body,
                    public: body,
                    trapdoor: body,
                    target: body,
                    family_operands: None,
                    wire: plan.target().residual.clone(),
                });
            }
            let reason = preflight
                .compact_residual_preflight(&Value::Family(family))
                .expect("negative fixed Slice must reject compact preflight");
            assert_eq!(reason, expected_reason);
            assert!(!preflight.job.relations().is_frozen());

            let adapter = ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
                .expect("fixed slice eager fallback adapter");
            let (mut eager_job, eager_roots) =
                adapter.lower_force_eager().expect("fixed Slice eager fallback");
            match eager_roots.residual {
                ProductionRoot::Closed(eager_root) => {
                    let eager = eager_job
                        .normalize_closed_root(eager_root)
                        .expect("fixed Slice eager fallback normalization");
                    assert!(
                        eager.value.exact_nf.as_ref().is_some_and(|nf| !nf.exact_terms.is_empty())
                    );
                }
                ProductionRoot::Family(eager_family) => {
                    let eager = eager_job
                        .analyze_family_root(eager_family)
                        .expect("fixed Slice eager family analysis");
                    assert!(eager.exact_term_count > 0);
                }
                ProductionRoot::Compact(_) | ProductionRoot::CompactFamily(_) => {
                    panic!("fixed Slice negative must not remain compact")
                }
            }
        }
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
        let actual = adapter.opaque_generated_family(domain, body).unwrap();
        let synthetic = adapter.generated_family(domain, body).unwrap();
        assert_ne!(actual, synthetic);
        assert_eq!(adapter.generated_family_calls, 2);
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
        adapter.push_relation_candidate(RelationCandidate {
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

        let external_public_family = adapter.opaque_generated_family(domain, public).unwrap();
        let external_target_family = adapter.opaque_generated_family(domain, target).unwrap();
        let external_trapdoor_family = adapter.opaque_generated_family(domain, trapdoor).unwrap();
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
        adapter.push_relation_candidate(RelationCandidate {
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
        adapter.push_relation_candidate(RelationCandidate {
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
    fn relation_lift_empty_occurrence_bucket_preserves_the_valid_body_identity() {
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
        let body = adapter.intern_index_constant(BigInt::from(7)).unwrap();

        let (rewritten, lifted) = adapter
            .lift_relation_family_operands(
                &occurrence,
                domain,
                body,
                argument,
                &plan.target().residual,
            )
            .expect("an empty occurrence bucket is a valid no-op");

        assert_eq!(rewritten, body);
        assert!(lifted.is_empty());
    }

    #[test]
    fn relation_candidate_occurrence_index_preserves_global_and_bucket_push_order() {
        let protocol = generated_gather_protocol(7);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("gather plan");
        let mut adapter = ProductionAdapter::new(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .expect("generated gather adapter");
        let first = ProgramOccurrence { definition: FrozenGraphScopeId::Root, path: 10 };
        let second = ProgramOccurrence { definition: FrozenGraphScopeId::Root, path: 11 };
        let stale_bucket = ProgramOccurrence { definition: FrozenGraphScopeId::Root, path: 12 };
        let expression = adapter.intern_index_constant(BigInt::from(3)).unwrap();
        for occurrence in [first.clone(), second.clone(), first.clone()] {
            let mut wire = plan.target().residual.clone();
            wire.occurrence = occurrence;
            adapter.push_relation_candidate(RelationCandidate {
                preimage: expression,
                public: expression,
                trapdoor: expression,
                target: expression,
                family_operands: None,
                wire,
            });
        }

        assert_eq!(adapter.relation_candidate_indices.get(&first), Some(&vec![0, 2]));
        assert_eq!(adapter.relation_candidate_indices.get(&second), Some(&vec![1]));
        assert_eq!(
            adapter
                .relation_candidates
                .iter()
                .map(|candidate| candidate.wire.occurrence.clone())
                .collect::<Vec<_>>(),
            vec![
                first,
                second,
                ProgramOccurrence { definition: FrozenGraphScopeId::Root, path: 10 }
            ]
        );

        adapter.relation_candidate_indices.insert(stale_bucket.clone(), vec![0]);
        let argument =
            adapter.job.expressions_mut().intern_argument(0, ResolvedValueType::Int).unwrap();
        let (rewritten, lifted) = adapter
            .lift_relation_family_operands(
                &stale_bucket,
                FamilyDomain::new(0, 4).unwrap(),
                expression,
                argument,
                &plan.target().residual,
            )
            .expect("a stale cross-occurrence bucket entry must be ignored");
        assert_eq!(rewritten, expression);
        assert!(lifted.is_empty());
    }

    #[test]
    fn reachability_snapshot_matches_legacy_membership_and_invalid_fallback() {
        let protocol = generated_gather_protocol(7);
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("gather plan");
        let mut adapter = ProductionAdapter::new(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .expect("generated gather adapter");
        let argument =
            adapter.job.expressions_mut().intern_argument(0, ResolvedValueType::Int).unwrap();
        let constant = adapter.intern_index_constant(BigInt::from(5)).unwrap();
        let shared = adapter
            .job
            .expressions_mut()
            .intern(ValueOperator::Scalar(ScalarOperation::Add), [argument, constant].into())
            .unwrap();
        let body = adapter
            .job
            .expressions_mut()
            .intern(ValueOperator::Scalar(ScalarOperation::Add), [shared, shared].into())
            .unwrap();
        let unreachable = adapter.intern_index_constant(BigInt::from(9)).unwrap();
        let snapshot = adapter.expression_reachable_set(body).expect("valid shared DAG");
        for target in [body, shared, argument, constant, unreachable] {
            assert_eq!(
                adapter.expression_reaches_with_snapshot(body, target, Some(&snapshot)).unwrap(),
                adapter.expression_reaches_with_snapshot(body, target, None).unwrap()
            );
        }

        let invalid = ExprId::new(body.arena(), u32::MAX);
        assert!(adapter.expression_reachable_set(invalid).is_err());
        let legacy = adapter.expression_reaches(invalid, shared).unwrap_err();
        let fallback = adapter.expression_reaches_with_snapshot(invalid, shared, None).unwrap_err();
        assert_eq!(format!("{legacy:?}"), format!("{fallback:?}"));
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
        adapter
            .job
            .insert_trapdoor_facts(
                adapter.token,
                trapdoor,
                super::super::facts::TrapdoorFacts {
                    coefficient_bound: NumericContract::Missing,
                    descriptor: "production-shaped-trapdoor".to_owned(),
                    paired_public_event: SampleEventId(901),
                    paired_public_output_role: "value".to_owned(),
                },
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
        adapter.push_relation_candidate(RelationCandidate {
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
        let (candidate_index, preimage_family, public_family, trapdoor_family, target_family) =
            lifted[0];
        let range = TrustedIndexRange { minimum: 0, maximum_exclusive: 4 };
        let compact_public_call = adapter
            .call_family_in_program_scope_deferred_generated(public_family, argument, range)
            .unwrap();
        assert!(matches!(
            adapter.job.expressions().node(compact_public_call).unwrap().operator,
            ValueOperator::ProgramCall { program } if program == public_family.program()
        ));
        let compact_preimage_call = adapter
            .call_family_in_program_scope_deferred_generated(preimage_family, argument, range)
            .unwrap();
        assert!(matches!(
            adapter.job.expressions().node(compact_preimage_call).unwrap().operator,
            ValueOperator::ProgramCall { program } if program == preimage_family.program()
        ));
        let rewritten_node = adapter.job.expressions().node(rewritten).unwrap();
        assert!(matches!(
            rewritten_node.operator,
            ValueOperator::Matrix(MatrixOperation::Multiply)
        ));
        assert!(rewritten_node.inputs.contains(&compact_public_call));
        assert!(rewritten_node.inputs.contains(&compact_preimage_call));

        let materialized = adapter.job.materialize_reducible_generated_calls(rewritten).unwrap();
        let public_call =
            adapter.call_family_in_program_scope(public_family, argument, range).unwrap();
        let preimage_call =
            adapter.call_family_in_program_scope(preimage_family, argument, range).unwrap();
        assert!(matches!(
            adapter.job.expressions().node(public_call).unwrap().operator,
            ValueOperator::Sampler { event: SampleEventId(901), .. }
        ));
        let materialized_node = adapter.job.expressions().node(materialized).unwrap();
        assert!(materialized_node.inputs.contains(&public_call));
        assert!(materialized_node.inputs.contains(&preimage_call));

        adapter.relation_candidates[candidate_index].family_operands =
            Some((preimage_family, public_family, trapdoor_family, target_family));
        adapter
            .register_reached_relations()
            .expect("materialized direct sampler/trapdoor endpoints must register");

        let transitive_public = adapter
            .job
            .expressions_mut()
            .intern_matrix_transform(MatrixOperation::Transpose, &[public])
            .unwrap();
        let transitive_public_family = adapter.generated_family(domain, transitive_public).unwrap();
        adapter.push_relation_candidate(RelationCandidate {
            preimage,
            public: transitive_public,
            trapdoor,
            target,
            family_operands: Some((
                preimage_family,
                transitive_public_family,
                trapdoor_family,
                target_family,
            )),
            wire: plan.target().residual.clone(),
        });
        assert!(matches!(
            adapter.register_reached_relations(),
            Err(ProductionAdapterError::Structural { ref reason, .. })
                if reason == "preimage relation public operand has no concrete sample event"
        ));
    }

    #[test]
    fn production_adapter_rejects_invalid_dynamic_slice_geometry() {
        for case in [GeneratedSliceCase::OutOfBounds, GeneratedSliceCase::NonAffine] {
            let protocol = generated_slice_protocol(7, case);
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
}
