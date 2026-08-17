//! Iterative Graph-IR lowering state for the operational-noise checker.
//!
//! A lowering wire is a concrete graph occurrence plus its active symbolic coordinates.  The
//! single memo below is the only owner of graph-wire lowering results; integer ranges and
//! selector provenance remain exclusively in the typed scalar store.

use super::{
    OperationalCheckRequest,
    bound::{BoundClass, MatrixBound, MatrixMetadata, ResolvedMatrixConstant, gadget_digit_bound},
    error::{LowerError, SelectorOnlyConsumer},
    family::{self, FamilyCoverageStorage, FamilyLoweringValue},
    identity::{
        BinderKey, CanonicalResidueConvention, CanonicalTermIdentity, OccurrenceScope,
        ResolvedIntExpr, SamplerDescriptorId, SamplerIdentity, SequentialStateKey, SymbolTables,
        TrapdoorDescriptorId, TrapdoorIdentity, TrapdoorSourceKey, WireSourceKey,
    },
    normal_form::{
        ExpressionDag, ExpressionNode, FactorIdentity, FactorKind, FactorOwner, RelationPattern,
        RelationRegistry, SymbolicFactor, TermId, centered_residue,
    },
    normal_form_family,
    normal_form_ops::{
        AdditionalOperations, BoolBit, CoefficientPreservingView, IntegerInterval,
        PolynomialNFOperations, ViewSpec,
    },
    scalar::{
        IntegerDomain, ScalarId, ScalarNode, ScalarOperation, ScalarProvenance, ScalarSort,
        ScalarStore, direct_extract_facts, matrix_types_equal, resolved_constant, sorts_equal,
    },
};
use crate::{
    DeclaredBoundExpr, InputValueContract, ProtocolDecl, ProtocolInputDestination, StageId,
    StageInputName,
};
use mxx_ir_core::{
    IntExpr, RealExpr, WireRef, WireType,
    graph::FrozenGraphScopeId,
    node::{
        ConcatAxis, HashVariant, IntBinaryOp, IntCompareOp, LoopInputMode, MatrixBinaryOp,
        NodeKind, ParallelLoop, RealBinaryOp, SequentialLoop,
    },
    types::MatrixType,
};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Structural-family dispatch is deliberately outside ordinary expression lowering.
///
/// The resolver receives the exact occurrence and lexical environment, so it can enter a
/// child scope or select one family element without introducing a second family cache.  It must
/// return the next concrete wire to lower; the caller remains the sole memo owner.
pub trait FamilyResolver {
    fn resolve(
        &mut self,
        wire: &LoweringWire,
        environment: &LowerEnv,
        index: Option<&LoweredInt>,
    ) -> Result<LoweringWire, LowerError>;
}

/// The lowering side of the checker progress reporter. Production callers pass
/// one live implementation through [`GraphLowerer::new_with_control`]; direct
/// lowering tests may use [`GraphLowerer::new`] without it.
pub trait LoweringControl {
    fn work(
        &mut self,
        scope: &OccurrenceScope,
        node: mxx_ir_core::NodeId,
    ) -> Result<(), LowerError>;
}

fn resolved_integer(value: &ResolvedIntExpr) -> Option<BigInt> {
    resolved_constant(value)
}
fn resolved_nonnegative(value: &ResolvedIntExpr) -> Option<num_bigint::BigUint> {
    resolved_integer(value)?.to_biguint()
}
fn concrete_matrix_type(
    value: &super::identity::ResolvedMatrixType,
) -> Option<mxx_ir_core::types::ConcreteMatrixType> {
    Some(mxx_ir_core::types::ConcreteMatrixType {
        modulus: resolved_integer(&value.modulus)?,
        ring_dimension: resolved_nonnegative(&value.ring_dimension)?.to_usize()?,
        rows: resolved_nonnegative(&value.rows)?.to_usize()?,
        columns: resolved_nonnegative(&value.columns)?.to_usize()?,
    })
}

fn apply_slice_type(
    mut matrix: mxx_ir_core::types::ConcreteMatrixType,
    spec: &super::identity::SliceSpec,
) -> Result<mxx_ir_core::types::ConcreteMatrixType, LowerError> {
    let length = |range: &super::identity::ResolvedIndexRange, extent: usize| {
        let start = resolved_nonnegative(&range.start)
            .and_then(|value| value.to_usize())
            .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
        let end = resolved_nonnegative(&range.end)
            .and_then(|value| value.to_usize())
            .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
        if start >= end || end > extent {
            return Err(LowerError::UnsupportedMatrixProductExpansion);
        }
        Ok(end - start)
    };
    if let Some(range) = &spec.rows {
        matrix.rows = length(range, matrix.rows)?;
    }
    if let Some(range) = &spec.columns {
        matrix.columns = length(range, matrix.columns)?;
    }
    Ok(matrix)
}

fn concat_matrix_type(
    dag: &ExpressionDag,
    inputs: &[TermId],
    axis: super::identity::Axis,
) -> Result<mxx_ir_core::types::ConcreteMatrixType, LowerError> {
    let shapes = inputs
        .iter()
        .map(|input| match dag.node(*input) {
            Ok(ExpressionNode::Atom(factor)) => factor
                .matrix_bound
                .as_ref()
                .map(|bound| bound.matrix_type.clone())
                .ok_or(LowerError::UnsupportedMatrixProductExpansion),
            _ => Err(LowerError::UnsupportedMatrixProductExpansion),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let first = shapes.first().cloned().ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
    if shapes
        .iter()
        .any(|shape| shape.modulus != first.modulus || shape.ring_dimension != first.ring_dimension)
    {
        return Err(LowerError::UnsupportedMatrixProductExpansion);
    }
    let (rows, columns) = match axis {
        super::identity::Axis::Rows => {
            if shapes.iter().any(|shape| shape.columns != first.columns) {
                return Err(LowerError::UnsupportedMatrixProductExpansion);
            }
            (
                shapes
                    .iter()
                    .try_fold(0usize, |sum, shape| sum.checked_add(shape.rows))
                    .ok_or(LowerError::UnsupportedMatrixProductExpansion)?,
                first.columns,
            )
        }
        super::identity::Axis::Columns => {
            if shapes.iter().any(|shape| shape.rows != first.rows) {
                return Err(LowerError::UnsupportedMatrixProductExpansion);
            }
            (
                first.rows,
                shapes
                    .iter()
                    .try_fold(0usize, |sum, shape| sum.checked_add(shape.columns))
                    .ok_or(LowerError::UnsupportedMatrixProductExpansion)?,
            )
        }
        super::identity::Axis::Diagonal => (
            shapes
                .iter()
                .try_fold(0usize, |sum, shape| sum.checked_add(shape.rows))
                .ok_or(LowerError::UnsupportedMatrixProductExpansion)?,
            shapes
                .iter()
                .try_fold(0usize, |sum, shape| sum.checked_add(shape.columns))
                .ok_or(LowerError::UnsupportedMatrixProductExpansion)?,
        ),
    };
    Ok(mxx_ir_core::types::ConcreteMatrixType {
        modulus: first.modulus,
        ring_dimension: first.ring_dimension,
        rows,
        columns,
    })
}

/// Closed routing table for graph operations.  Keeping this table separate from expression
/// construction makes accidental support for a structural family node impossible: every newly
/// added `NodeKind` must be classified here before it can reach an e-node constructor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NodeDispatch {
    Ordinary,
    Source,
    Structural,
    DecoderOnly,
}

pub const fn node_dispatch(kind: &NodeKind) -> NodeDispatch {
    match kind {
        NodeKind::Input { .. } |
        NodeKind::GadgetTrapdoor { .. } |
        NodeKind::TrapdoorPublic |
        NodeKind::UniformResidueSample { .. } |
        NodeKind::UniformIntervalSample { .. } |
        NodeKind::GaussianSample { .. } |
        NodeKind::HashSample { .. } |
        NodeKind::TrapdoorSample { .. } |
        NodeKind::PreimageSample { .. } |
        NodeKind::GadgetDecompose { .. } => NodeDispatch::Source,
        NodeKind::SubgraphCall(_) |
        NodeKind::ParallelLoop(_) |
        NodeKind::SequentialLoop(_) |
        NodeKind::FamilyPack { .. } |
        NodeKind::FamilyGetStatic { .. } |
        NodeKind::FamilyGetDynamic |
        NodeKind::Select { .. } |
        NodeKind::PackPolynomialCoefficients { .. } => NodeDispatch::Structural,
        NodeKind::ThresholdDecode { .. } => NodeDispatch::DecoderOnly,
        NodeKind::ConstantInt(_) |
        NodeKind::EvaluateInt(_) |
        NodeKind::ConstantReal(_) |
        NodeKind::ConstantBool(_) |
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
        NodeKind::ExtractCoefficient { .. } |
        NodeKind::LiftIntegerToConstantPolynomial { .. } |
        NodeKind::CrtRecompose { .. } |
        NodeKind::ConstantMatrix { .. } => NodeDispatch::Ordinary,
    }
}

/// An integer term together with a stable owner-resolved expression when one exists.
///
/// Its range is deliberately absent: `ScalarFacts` is the only range owner.
#[derive(Clone, Debug)]
pub struct LoweredInt {
    pub scalar: ScalarId,
}

/// One active loop coordinate, ordered outermost to innermost.
#[derive(Clone, Debug)]
pub struct Coordinate {
    pub binder: BinderKey,
    pub index: LoweredInt,
}

/// A graph wire occurrence and the symbolic coordinates at which it is evaluated.
#[derive(Clone, Debug)]
pub struct LoweringWire {
    pub source: WireSourceKey,
    pub indices: Box<[LoweredInt]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
enum FamilyIndexKey {
    Stable(ResolvedIntExpr),
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct LoweringWireKey {
    source: WireSourceKey,
    indices: Box<[FamilyIndexKey]>,
}

#[derive(Clone, Debug)]
pub enum LoweredValue {
    /// A matrix expression in the single typed lowering DAG.
    Matrix(TermId),
    MatrixFamily(FamilyLoweringValue<TermId>),
    /// A typed scalar/domain value owned by this lowering job.
    Scalar(ScalarId),
    Family(FamilyLoweringValue<ScalarId>),
    Trapdoor(TrapdoorDescriptorId),
    TrapdoorFamily {
        representative: TrapdoorDescriptorId,
        binder: BinderKey,
        logical_count: num_bigint::BigUint,
    },
}

/// Lexical bindings for one concrete scope occurrence.
#[derive(Clone, Debug)]
pub struct LowerEnv {
    pub occurrence: OccurrenceScope,
    pub parameters: BTreeMap<String, ResolvedIntExpr>,
    pub binders: Vec<(BinderKey, LoweredInt)>,
    /// Ordinary call-input aliases keyed by their complete owning occurrence.
    /// Inherited entries remain available while lowering a nested call.
    pub inputs: BTreeMap<WireSourceKey, LoweringWire>,
    /// Loop-owned input values, including sequential carried state.  The
    /// source occurrence is part of the key so a nested subgraph can retain a
    /// parent-loop binding without confusing equal local node numbers.
    pub state_inputs: BTreeMap<WireSourceKey, LoweredValue>,
    pub active_coordinates: Vec<Coordinate>,
}

/// One continuation on the job-wide graph-lowering stack.  Structural nodes
/// schedule their child bodies here instead of re-entering lowering with a
/// nested work stack.
enum LoweringFrame {
    Enter {
        wire: LoweringWire,
        environment: LowerEnv,
    },
    Finish {
        wire: LoweringWire,
        environment: LowerEnv,
        kind: NodeKind,
        dependency_count: usize,
    },
    FinishStructural {
        wire: LoweringWire,
        environment: LowerEnv,
        kind: NodeKind,
        output_type: WireType,
        dependency_count: usize,
    },
    FinishAlias {
        wire: LoweringWire,
    },
    FinishProtocolTrapdoor {
        wire: LoweringWire,
        environment: LowerEnv,
        output_type: WireType,
    },
    FinishValue {
        wire: LoweringWire,
        value: LoweredValue,
    },
    FinishIndexedAlias {
        wire: LoweringWire,
        index: LoweredInt,
    },
    FinishPreimage {
        wire: LoweringWire,
        environment: LowerEnv,
        cutoff: IntExpr,
        output_type: WireType,
    },
    FinishHashSample {
        wire: LoweringWire,
        environment: LowerEnv,
        matrix_type: MatrixType,
        variant: HashVariant,
        base: Option<IntExpr>,
        digit_count: Option<IntExpr>,
        output_type: WireType,
        dependency_count: usize,
    },
    FinishGadgetDecompose {
        wire: LoweringWire,
        environment: LowerEnv,
        base: IntExpr,
        digit_count: IntExpr,
        small: bool,
        output_type: WireType,
    },
    FinishParallelLoop {
        wire: LoweringWire,
        body_source: WireSourceKey,
        environment: LowerEnv,
        specification: ParallelLoop,
        output_type: WireType,
        binder: BinderKey,
        logical_count: num_bigint::BigUint,
        maximum: BigInt,
    },
    FinishSequentialMatrixLoop {
        wire: LoweringWire,
        environment: LowerEnv,
        count: ResolvedIntExpr,
        initial: Vec<TermId>,
        state_factors: Vec<FactorIdentity>,
        iteration_binder: Option<BinderKey>,
        output_types: Vec<super::identity::ResolvedMatrixType>,
        output_type: WireType,
        carried_index: usize,
        dependency_count: usize,
    },
}

/// The sole mutable owner for one lowering job.
pub struct GraphLowerer<'a, 'control> {
    pub protocol: &'a ProtocolDecl,
    pub request: &'a OperationalCheckRequest,
    pub symbols: SymbolTables,
    /// The sole matrix-expression store for this lowering job.
    pub dag: ExpressionDag,
    /// The sole normal-form relation registry for this lowering job.
    pub relation_registry: RelationRegistry,
    memo: HashMap<LoweringWireKey, LoweredValue>,
    /// Job-local memo for owner-aware shared-family substitutions.  This is
    /// deliberately keyed by the source term and binder/value, not by a
    /// logical lane; no family is expanded into a Cartesian cache.
    family_substitution_memo: BTreeMap<(ScalarId, BinderKey, ResolvedIntExpr), ScalarId>,
    matrix_family_substitution_memo: BTreeMap<(TermId, BinderKey, ResolvedIntExpr), TermId>,
    scalar_store: ScalarStore,
    active: HashSet<LoweringWireKey>,
    control: Option<&'control mut dyn LoweringControl>,
}

impl<'a> GraphLowerer<'a, '_> {
    pub fn new(
        protocol: &'a ProtocolDecl,
        request: &'a OperationalCheckRequest,
        symbols: SymbolTables,
    ) -> Self {
        Self {
            protocol,
            request,
            symbols,
            dag: ExpressionDag::new(),
            relation_registry: RelationRegistry::default(),
            memo: HashMap::new(),
            family_substitution_memo: BTreeMap::new(),
            matrix_family_substitution_memo: BTreeMap::new(),
            scalar_store: ScalarStore::default(),
            active: HashSet::new(),
            control: None,
        }
    }
}

impl<'a, 'control> GraphLowerer<'a, 'control> {
    /// Constructs a production lowerer with the job-wide control bridge.
    /// This bridge is retained by nested parallel and sequential body walks,
    /// rather than being recreated per lexical scope.
    pub fn new_with_control(
        protocol: &'a ProtocolDecl,
        request: &'a OperationalCheckRequest,
        symbols: SymbolTables,
        control: &'control mut dyn LoweringControl,
    ) -> Self {
        Self {
            protocol,
            request,
            symbols,
            dag: ExpressionDag::new(),
            relation_registry: RelationRegistry::default(),
            memo: HashMap::new(),
            family_substitution_memo: BTreeMap::new(),
            matrix_family_substitution_memo: BTreeMap::new(),
            scalar_store: ScalarStore::default(),
            active: HashSet::new(),
            control: Some(control),
        }
    }

    /// Consumes the lowering-phase view and returns the same lowered state with
    /// no remaining borrow of the job control.
    pub fn into_uncontrolled(mut self) -> GraphLowerer<'a, 'static> {
        self.control = None;
        GraphLowerer {
            protocol: self.protocol,
            request: self.request,
            symbols: self.symbols,
            dag: self.dag,
            relation_registry: self.relation_registry,
            memo: self.memo,
            family_substitution_memo: self.family_substitution_memo,
            matrix_family_substitution_memo: self.matrix_family_substitution_memo,
            scalar_store: self.scalar_store,
            active: self.active,
            control: None,
        }
    }

    /// Reads canonical scalar facts without creating a second evaluator or cache.
    pub fn integer_analysis(&self, scalar: ScalarId) -> Option<(&IntegerDomain, ScalarProvenance)> {
        let data = self.scalar_store.facts(scalar)?;
        if data.sort != Ok(ScalarSort::Int) {
            return None;
        }
        Some((data.integer_domain.as_ref()?, data.scalar_provenance?))
    }

    fn canonical_term_identity(&self, scalar: ScalarId) -> Result<ResolvedIntExpr, LowerError> {
        self.scalar_store
            .identity(scalar)
            .ok_or(LowerError::MissingIntegerAnalysis { term: scalar })
    }

    fn lowering_wire_key(&self, wire: &LoweringWire) -> Result<LoweringWireKey, LowerError> {
        Ok(LoweringWireKey {
            source: wire.source.clone(),
            indices: wire
                .indices
                .iter()
                .map(|index| {
                    self.canonical_scalar_identity(index.scalar).map(FamilyIndexKey::Stable)
                })
                .collect::<Result<Box<_>, _>>()?,
        })
    }

    fn canonical_scalar_identity(&self, scalar: ScalarId) -> Result<ResolvedIntExpr, LowerError> {
        let facts = self
            .scalar_store
            .facts(scalar)
            .ok_or(LowerError::MissingIntegerAnalysis { term: scalar })?;
        if facts.sort != Ok(ScalarSort::Int) {
            return Err(LowerError::MissingIntegerAnalysis { term: scalar });
        }
        self.canonical_term_identity(scalar)
    }

    fn selector_reachable(&self, term: ScalarId, count: usize) -> Result<Box<[usize]>, LowerError> {
        let (domain, _) =
            self.integer_analysis(term).ok_or(LowerError::MissingIntegerAnalysis { term })?;
        let interval = domain.interval().map_err(|_| LowerError::FamilyAccessOutOfRange {
            index: IntExpr::constant(-1),
            count: IntExpr::constant(count),
        })?;
        let count_value = BigInt::from(count);
        if interval.minimum < BigInt::zero() || interval.maximum >= count_value {
            return Err(LowerError::FamilyAccessOutOfRange {
                index: IntExpr::constant(-1),
                count: IntExpr::constant(count),
            });
        }
        let start =
            interval.minimum.to_usize().ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
        let end =
            interval.maximum.to_usize().ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
        Ok((start..=end).collect())
    }

    /// Enforces the closed selector-only consumer table at every scalar use-site.
    pub fn validate_integer_consumer(
        &self,
        term: ScalarId,
        consumer: SelectorOnlyConsumer,
        selector_allowed: bool,
    ) -> Result<(), LowerError> {
        self.validate_scalar_consumer(
            term,
            ScalarSort::Int,
            mxx_ir_core::WireType::Int,
            consumer,
            selector_allowed,
        )
    }

    fn validate_boolean_consumer(
        &self,
        term: ScalarId,
        consumer: SelectorOnlyConsumer,
        selector_allowed: bool,
    ) -> Result<(), LowerError> {
        self.validate_scalar_consumer(
            term,
            ScalarSort::Bool,
            mxx_ir_core::WireType::Bool,
            consumer,
            selector_allowed,
        )
    }

    fn validate_scalar_consumer(
        &self,
        term: ScalarId,
        expected_sort: ScalarSort,
        expected_wire_type: mxx_ir_core::WireType,
        consumer: SelectorOnlyConsumer,
        selector_allowed: bool,
    ) -> Result<(), LowerError> {
        let data =
            self.scalar_store.facts(term).ok_or(LowerError::MissingIntegerAnalysis { term })?;
        if data.sort != Ok(expected_sort.clone()) {
            let actual = match data.sort.as_ref().ok() {
                Some(ScalarSort::Int) => mxx_ir_core::WireType::Int,
                Some(ScalarSort::Bool) => mxx_ir_core::WireType::Bool,
                Some(ScalarSort::Real) => mxx_ir_core::WireType::Real,
                _ => expected_wire_type.clone(),
            };
            return Err(LowerError::InvalidOperandSort { expected: expected_wire_type, actual });
        }
        let Some(provenance) = data.scalar_provenance else {
            return Err(LowerError::InvalidOperandSort {
                expected: expected_wire_type.clone(),
                actual: expected_wire_type,
            });
        };
        if provenance == ScalarProvenance::SelectorOnly && !selector_allowed {
            return Err(LowerError::SelectorOnlyValueUsedByForbiddenConsumer { consumer });
        }
        Ok(())
    }

    /// Begins one memoized wire lowering.  A repeated active key is a graph dependency cycle;
    /// completed keys return their one stored result without repeating graph work.
    pub fn begin_wire(&mut self, wire: &LoweringWire) -> Result<Option<LoweredValue>, LowerError> {
        let key = self.lowering_wire_key(wire)?;
        if let Some(value) = self.memo.get(&key) {
            return Ok(Some(value.clone()));
        }
        if self.active.contains(&key) {
            return Err(LowerError::CyclicGraphDependency { wire: wire.source.wire });
        }
        self.active.insert(key);
        Ok(None)
    }

    pub fn finish_wire(&mut self, wire: &LoweringWire, value: LoweredValue) {
        let key = self.lowering_wire_key(wire).expect("lowered scalar identity");
        self.active.remove(&key);
        self.memo.insert(key, value);
    }

    /// Exposes the graph-work accounting used by count-independence fixtures.
    pub fn lowered_wire_count(&self) -> usize {
        self.memo.len()
    }

    /// Returns the one matrix expression DAG owned by this lowering job.
    pub fn expression_dag(&self) -> &ExpressionDag {
        &self.dag
    }

    /// Returns the one checked normal-form relation registry owned by this job.
    pub fn normal_form_relations(&self) -> &RelationRegistry {
        &self.relation_registry
    }

    /// Starts the one job-wide, non-recursive lowering traversal at a workflow-stage wire.
    /// Structural family producers are deliberately routed through `FamilyResolver`; ordinary
    /// graph dependencies are visited in producer order without using the Rust call stack.
    pub fn lower_stage_wire(
        &mut self,
        stage: &StageId,
        wire: WireRef,
    ) -> Result<LoweredValue, LowerError> {
        if !self.protocol.stages().iter().any(|candidate| &candidate.id == stage) {
            return Err(LowerError::MissingWire { wire });
        }
        let environment = LowerEnv {
            occurrence: OccurrenceScope {
                program: super::identity::ProgramKey::WorkflowStage(stage.clone()),
                definition: FrozenGraphScopeId::Root,
                path: Box::new([]),
            },
            parameters: self
                .request
                .environment
                .iter()
                .filter_map(|(name, value)| match value {
                    super::OperationalParameterValue::Integer(value) => {
                        Some((name.clone(), ResolvedIntExpr::Const(value.clone())))
                    }
                    super::OperationalParameterValue::Rational { .. } => None,
                })
                .collect(),
            binders: Vec::new(),
            inputs: BTreeMap::new(),
            state_inputs: BTreeMap::new(),
            active_coordinates: Vec::new(),
        };
        self.lower_wire_iterative(
            LoweringWire {
                source: WireSourceKey { scope: environment.occurrence.clone(), wire },
                indices: Box::new([]),
            },
            environment,
        )
    }

    fn lower_wire_iterative(
        &mut self,
        root: LoweringWire,
        root_environment: LowerEnv,
    ) -> Result<LoweredValue, LowerError> {
        let mut work = vec![LoweringFrame::Enter { wire: root, environment: root_environment }];
        let mut values = Vec::<LoweredValue>::new();
        while let Some(frame) = work.pop() {
            match frame {
                LoweringFrame::Enter { wire, environment } => {
                    if let Some(control) = self.control.as_deref_mut() {
                        control.work(&wire.source.scope, wire.source.wire.node)?;
                    }
                    if let Some(value) = self.begin_wire(&wire)? {
                        values.push(value);
                        continue;
                    }
                    let active_graph = self.graph_for_program(&environment.occurrence.program)?;
                    let scope = active_graph
                        .scope(&wire.source.scope.definition)
                        .ok_or(LowerError::MissingWire { wire: wire.source.wire })?;
                    let node = scope
                        .node(wire.source.wire.node)
                        .ok_or(LowerError::MissingNode { node: wire.source.wire.node })?;
                    if node.output_types().get(wire.source.wire.port.0 as usize).is_none() {
                        return Err(LowerError::InvalidOutputPort {
                            wire: wire.source.wire,
                            output_count: node.output_types().len(),
                        });
                    }
                    if let Some(value) = environment.state_inputs.get(&wire.source) {
                        self.finish_wire(&wire, value.clone());
                        values.push(value.clone());
                        continue;
                    }
                    if let Some(bound) = environment.inputs.get(&wire.source) {
                        if let Some(index) = bound.indices.first() {
                            work.push(LoweringFrame::FinishIndexedAlias {
                                wire,
                                index: index.clone(),
                            });
                        } else {
                            work.push(LoweringFrame::FinishAlias { wire });
                        }
                        work.push(LoweringFrame::Enter { wire: bound.clone(), environment });
                        continue;
                    }
                    if let mxx_ir_core::node::NodeKind::Input { name, artifact: Some(_), .. } =
                        node.kind()
                    {
                        let super::identity::ProgramKey::WorkflowStage(consumer) =
                            &environment.occurrence.program
                        else {
                            return Err(LowerError::ArtifactProducerMissing {
                                consumer: StageId("<non-workflow>".to_owned()),
                                input: StageInputName(name.clone()),
                            });
                        };
                        let stage = self
                            .protocol
                            .stages()
                            .iter()
                            .find(|stage| &stage.id == consumer)
                            .ok_or(LowerError::ArtifactProducerMissing {
                                consumer: consumer.clone(),
                                input: StageInputName(name.clone()),
                            })?;
                        let bindings = stage
                            .bindings
                            .iter()
                            .filter(|binding| {
                                binding.consumer_input == StageInputName(name.clone())
                            })
                            .collect::<Vec<_>>();
                        let [binding] = bindings.as_slice() else {
                            return Err(if bindings.is_empty() {
                                LowerError::ArtifactProducerMissing {
                                    consumer: consumer.clone(),
                                    input: StageInputName(name.clone()),
                                }
                            } else {
                                LowerError::ArtifactProducerAmbiguous {
                                    consumer: consumer.clone(),
                                    input: StageInputName(name.clone()),
                                    candidates: Box::new([]),
                                }
                            });
                        };
                        let producer = self
                            .protocol
                            .stages()
                            .iter()
                            .find(|stage| stage.id == binding.producer_stage)
                            .ok_or(LowerError::ArtifactProducerMissing {
                                consumer: consumer.clone(),
                                input: StageInputName(name.clone()),
                            })?;
                        let output =
                            producer.graph.outputs().get(&binding.producer_output.0).ok_or(
                                LowerError::ArtifactProducerMissing {
                                    consumer: consumer.clone(),
                                    input: StageInputName(name.clone()),
                                },
                            )?;
                        work.push(LoweringFrame::FinishAlias { wire });
                        work.push(LoweringFrame::Enter {
                            wire: LoweringWire {
                                source: WireSourceKey {
                                    scope: OccurrenceScope {
                                        program: super::identity::ProgramKey::WorkflowStage(
                                            producer.id.clone(),
                                        ),
                                        definition: FrozenGraphScopeId::Root,
                                        path: Box::new([]),
                                    },
                                    wire: output.value,
                                },
                                indices: Box::new([]),
                            },
                            environment: LowerEnv {
                                occurrence: OccurrenceScope {
                                    program: super::identity::ProgramKey::WorkflowStage(
                                        producer.id.clone(),
                                    ),
                                    definition: FrozenGraphScopeId::Root,
                                    path: Box::new([]),
                                },
                                parameters: environment.parameters.clone(),
                                binders: Vec::new(),
                                inputs: BTreeMap::new(),
                                state_inputs: BTreeMap::new(),
                                active_coordinates: Vec::new(),
                            },
                        });
                        continue;
                    }
                    match node_dispatch(node.kind()) {
                        NodeDispatch::Source => {
                            if let NodeKind::Input { artifact: None, .. } = node.kind() {
                                let output_type =
                                    node.output_types()[wire.source.wire.port.0 as usize].clone();
                                if matches!(output_type, WireType::Trapdoor { .. }) {
                                    work.push(LoweringFrame::FinishProtocolTrapdoor {
                                        wire,
                                        environment,
                                        output_type,
                                    });
                                    continue;
                                }
                            }
                            if let NodeKind::HashSample {
                                matrix_type,
                                variant,
                                tag_prefix: _,
                                tag_expressions: _,
                                tag_decimal_expressions: _,
                                tag_u64_le_expressions: _,
                                base,
                                digit_count,
                            } = node.kind()
                            {
                                let arguments = scope
                                    .arguments(node)
                                    .ok_or(LowerError::MissingWire { wire: wire.source.wire })?;
                                if arguments.is_empty() {
                                    return Err(LowerError::InvalidOperandArity {
                                        expected: 1,
                                        actual: 0,
                                    });
                                }
                                work.push(LoweringFrame::FinishHashSample {
                                    wire: wire.clone(),
                                    environment: environment.clone(),
                                    matrix_type: matrix_type.clone(),
                                    variant: *variant,
                                    base: base.clone(),
                                    digit_count: digit_count.clone(),
                                    output_type: node.output_types()
                                        [wire.source.wire.port.0 as usize]
                                        .clone(),
                                    dependency_count: arguments.len(),
                                });
                                for argument in arguments.into_iter().rev() {
                                    work.push(LoweringFrame::Enter {
                                        wire: LoweringWire {
                                            source: WireSourceKey {
                                                scope: wire.source.scope.clone(),
                                                wire: argument,
                                            },
                                            indices: wire.indices.clone(),
                                        },
                                        environment: environment.clone(),
                                    });
                                }
                                continue;
                            }
                            if let NodeKind::GadgetDecompose { base, small, digit_count } =
                                node.kind()
                            {
                                let arguments = scope
                                    .arguments(node)
                                    .ok_or(LowerError::MissingWire { wire: wire.source.wire })?;
                                let [argument] = arguments.as_slice() else {
                                    return Err(LowerError::InvalidOperandArity {
                                        expected: 1,
                                        actual: arguments.len(),
                                    });
                                };
                                work.push(LoweringFrame::FinishGadgetDecompose {
                                    wire: wire.clone(),
                                    environment: environment.clone(),
                                    base: base.clone(),
                                    digit_count: digit_count.clone(),
                                    small: *small,
                                    output_type: node.output_types()
                                        [wire.source.wire.port.0 as usize]
                                        .clone(),
                                });
                                work.push(LoweringFrame::Enter {
                                    wire: LoweringWire {
                                        source: WireSourceKey {
                                            scope: wire.source.scope.clone(),
                                            wire: *argument,
                                        },
                                        indices: wire.indices.clone(),
                                    },
                                    environment: environment.clone(),
                                });
                                continue;
                            }
                            if let NodeKind::PreimageSample { max_coefficient_bound, .. } =
                                node.kind()
                            {
                                let arguments = scope
                                    .arguments(node)
                                    .ok_or(LowerError::MissingWire { wire: wire.source.wire })?;
                                if arguments.len() != 3 {
                                    return Err(LowerError::InvalidOperandArity {
                                        expected: 3,
                                        actual: arguments.len(),
                                    });
                                }
                                work.push(LoweringFrame::FinishPreimage {
                                    wire: wire.clone(),
                                    environment: environment.clone(),
                                    cutoff: max_coefficient_bound.clone(),
                                    output_type: node.output_types()
                                        [wire.source.wire.port.0 as usize]
                                        .clone(),
                                });
                                for argument in arguments.into_iter().rev() {
                                    work.push(LoweringFrame::Enter {
                                        wire: LoweringWire {
                                            source: WireSourceKey {
                                                scope: wire.source.scope.clone(),
                                                wire: argument,
                                            },
                                            indices: wire.indices.clone(),
                                        },
                                        environment: environment.clone(),
                                    });
                                }
                                continue;
                            }
                            if let NodeKind::TrapdoorSample {
                                matrix_type,
                                sigma,
                                gadget_base,
                                digit_count,
                                preimage_max_coefficient_bound,
                            } = node.kind() &&
                                wire.source.wire.port.0 == 1
                            {
                                let matrix_type = matrix_type.clone();
                                let sigma = sigma.clone();
                                let gadget_base = gadget_base.clone();
                                let digit_count = digit_count.clone();
                                let preimage_max_coefficient_bound =
                                    preimage_max_coefficient_bound.clone();
                                let source = TrapdoorSourceKey::GraphWire(
                                    super::identity::GraphWireSourceKey {
                                        wire: wire.source.clone(),
                                        coordinate_binders: environment
                                            .active_coordinates
                                            .iter()
                                            .map(|coordinate| coordinate.binder.clone())
                                            .collect(),
                                    },
                                );
                                let descriptor = TrapdoorIdentity {
                                    source,
                                    indices: environment
                                        .active_coordinates
                                        .iter()
                                        .map(|coordinate| {
                                            self.canonical_scalar_identity(coordinate.index.scalar)
                                        })
                                        .collect::<Result<Box<[_]>, _>>()?,
                                    matrix_type: self
                                        .resolve_matrix_type(&matrix_type, &environment)?,
                                    public: CanonicalTermIdentity::Source(
                                        super::identity::GraphWireSourceKey {
                                            wire: WireSourceKey {
                                                scope: wire.source.scope.clone(),
                                                wire: WireRef {
                                                    node: wire.source.wire.node,
                                                    port: mxx_ir_core::Port(0),
                                                },
                                            },
                                            coordinate_binders: environment
                                                .active_coordinates
                                                .iter()
                                                .map(|coordinate| coordinate.binder.clone())
                                                .collect(),
                                        },
                                    ),
                                    sigma_bits: self.resolve_real(&sigma, &environment)?.to_bits(),
                                    gadget_base: self.resolve_int(&gadget_base, &environment)?,
                                    digit_count: self.resolve_int(&digit_count, &environment)?,
                                    preimage_cutoff: self.resolve_int(
                                        &preimage_max_coefficient_bound,
                                        &environment,
                                    )?,
                                };
                                let descriptor = self.symbols.trapdoors.intern(descriptor);
                                let value =
                                    LoweredValue::Trapdoor(TrapdoorDescriptorId(descriptor));
                                self.finish_wire(&wire, value.clone());
                                values.push(value);
                                continue;
                            }
                            let role = match node.kind() {
                                NodeKind::PreimageSample { .. } => {
                                    Some(super::identity::AtomicRelationRole::Preimage)
                                }
                                NodeKind::HashSample {
                                    variant: HashVariant::Decomposed, ..
                                } => Some(super::identity::AtomicRelationRole::DecomposedHash),
                                NodeKind::HashSample {
                                    variant: HashVariant::SmallDecomposed,
                                    ..
                                } => {
                                    Some(super::identity::AtomicRelationRole::SmallDecomposedHash {
                                        // The Graph IR has no range proof on a hash sampler.
                                        // A relation consumer therefore remains fail-closed until
                                        // an explicit producer contract establishes this fact.
                                        range_proved: false,
                                    })
                                }
                                NodeKind::GadgetDecompose { small: false, .. } => {
                                    Some(super::identity::AtomicRelationRole::GadgetDecomposition)
                                }
                                NodeKind::GadgetDecompose { small: true, .. } => Some(
                                    super::identity::AtomicRelationRole::SmallGadgetDecomposition {
                                        range_proved: false,
                                    },
                                ),
                                _ => None,
                            };
                            let output_type =
                                node.output_types()[wire.source.wire.port.0 as usize].clone();
                            if let WireType::IndexedFamily { element, count } = output_type {
                                let value = if wire.indices.is_empty() {
                                    self.lower_source_family(&wire, &environment, *element, count)?
                                } else {
                                    match *element {
                                        WireType::Matrix(_) | WireType::Preimage(_) => {
                                            return Err(
                                                LowerError::UnsupportedMatrixProductExpansion,
                                            )
                                        }
                                        element => LoweredValue::Scalar(self.atom_for_wire(
                                            &wire,
                                            &environment,
                                            element,
                                            role,
                                        )?),
                                    }
                                };
                                self.finish_wire(&wire, value.clone());
                                values.push(value);
                                continue;
                            }
                            let value = match output_type {
                                WireType::Matrix(_) | WireType::Preimage(_) => {
                                    LoweredValue::Matrix(self.matrix_atom_for_wire(
                                        &wire,
                                        &environment,
                                        output_type,
                                        role,
                                    )?)
                                }
                                output_type => LoweredValue::Scalar(self.atom_for_wire(
                                    &wire,
                                    &environment,
                                    output_type,
                                    role,
                                )?),
                            };
                            self.finish_wire(&wire, value.clone());
                            values.push(value);
                        }
                        NodeDispatch::Ordinary => {
                            let arguments = scope
                                .arguments(node)
                                .ok_or(LowerError::MissingWire { wire: wire.source.wire })?;
                            work.push(LoweringFrame::Finish {
                                wire: wire.clone(),
                                environment: environment.clone(),
                                kind: node.kind().clone(),
                                dependency_count: arguments.len(),
                            });
                            for argument in arguments.into_iter().rev() {
                                work.push(LoweringFrame::Enter {
                                    wire: LoweringWire {
                                        source: WireSourceKey {
                                            scope: wire.source.scope.clone(),
                                            wire: argument,
                                        },
                                        indices: wire.indices.clone(),
                                    },
                                    environment: environment.clone(),
                                });
                            }
                        }
                        NodeDispatch::Structural | NodeDispatch::DecoderOnly => {
                            if let mxx_ir_core::node::NodeKind::SubgraphCall(call) = node.kind() {
                                let child_definition = active_graph
                                    .child_scope_id(
                                        &wire.source.scope.definition,
                                        wire.source.wire.node,
                                    )
                                    .ok_or(LowerError::MissingWire { wire: wire.source.wire })?;
                                let child_scope = active_graph
                                    .scope(&child_definition)
                                    .ok_or(LowerError::MissingWire { wire: wire.source.wire })?;
                                let arguments = scope
                                    .arguments(node)
                                    .ok_or(LowerError::MissingWire { wire: wire.source.wire })?;
                                let child_inputs = child_scope.inputs().to_vec();
                                let child_outputs = child_scope.outputs().to_vec();
                                if child_inputs.len() != arguments.len() {
                                    return Err(LowerError::InvalidOperandArity {
                                        expected: child_inputs.len(),
                                        actual: arguments.len(),
                                    });
                                }
                                let mut child = environment.clone();
                                child.occurrence.definition = child_definition.clone();
                                child.occurrence.path = environment
                                    .occurrence
                                    .path
                                    .iter()
                                    .cloned()
                                    .chain([super::identity::OccurrenceFrame::Call {
                                        parent: wire.source.scope.definition.clone(),
                                        owner: wire.source.wire.node,
                                    }])
                                    .collect();
                                child.inputs.extend(
                                    child_inputs.iter().copied().zip(arguments).map(
                                        |(input, argument)| {
                                            (
                                                WireSourceKey {
                                                    scope: child.occurrence.clone(),
                                                    wire: input,
                                                },
                                                LoweringWire {
                                                    source: WireSourceKey {
                                                        scope: wire.source.scope.clone(),
                                                        wire: argument,
                                                    },
                                                    indices: wire.indices.clone(),
                                                },
                                            )
                                        },
                                    ),
                                );
                                let parameter_bindings = call.bindings.clone();
                                for (name, expression) in &parameter_bindings {
                                    let lowered = self.lower_int_expr(expression, &environment)?;
                                    let value = self.canonical_scalar_identity(lowered.scalar)?;
                                    child.parameters.insert(name.clone(), value);
                                }
                                let output = *child_outputs
                                    .get(wire.source.wire.port.0 as usize)
                                    .ok_or(LowerError::InvalidOutputPort {
                                    wire: wire.source.wire,
                                    output_count: child_outputs.len(),
                                })?;
                                work.push(LoweringFrame::FinishAlias { wire: wire.clone() });
                                work.push(LoweringFrame::Enter {
                                    wire: LoweringWire {
                                        source: WireSourceKey {
                                            scope: child.occurrence.clone(),
                                            wire: output,
                                        },
                                        indices: wire.indices.clone(),
                                    },
                                    environment: child,
                                });
                            } else if matches!(node_dispatch(node.kind()), NodeDispatch::Structural)
                            {
                                let arguments = scope
                                    .arguments(node)
                                    .ok_or(LowerError::MissingWire { wire: wire.source.wire })?;
                                work.push(LoweringFrame::FinishStructural {
                                    wire: wire.clone(),
                                    environment: environment.clone(),
                                    kind: node.kind().clone(),
                                    output_type: node.output_types()
                                        [wire.source.wire.port.0 as usize]
                                        .clone(),
                                    dependency_count: arguments.len(),
                                });
                                for argument in arguments.into_iter().rev() {
                                    work.push(LoweringFrame::Enter {
                                        wire: LoweringWire {
                                            source: WireSourceKey {
                                                scope: wire.source.scope.clone(),
                                                wire: argument,
                                            },
                                            indices: wire.indices.clone(),
                                        },
                                        environment: environment.clone(),
                                    });
                                }
                            } else {
                                return Err(LowerError::MissingWire { wire: wire.source.wire });
                            }
                        }
                    }
                }
                LoweringFrame::Finish { wire, environment, kind, dependency_count } => {
                    if values.len() < dependency_count {
                        return Err(LowerError::InvalidOperandArity {
                            expected: dependency_count,
                            actual: values.len(),
                        });
                    }
                    let arguments = values.split_off(values.len() - dependency_count);
                    let value = match &kind {
                        NodeKind::ConstantMatrix { value, matrix_type } => {
                            if !arguments.is_empty() {
                                return Err(LowerError::InvalidOperandArity {
                                    expected: 0,
                                    actual: arguments.len(),
                                });
                            }
                            self.lower_matrix_constant_dag(&wire, value, matrix_type, &environment)?
                        }
                        NodeKind::LiftIntegerToConstantPolynomial { matrix_type }
                            if arguments.len() == 1 &&
                                matches!(arguments[0], LoweredValue::Scalar(_)) =>
                        {
                            self.lower_lift_constant_dag(
                                &wire,
                                &arguments[0],
                                matrix_type,
                                &environment,
                            )?
                        }
                        _ => self.lower_node(&kind, &arguments, &environment)?,
                    };
                    self.finish_wire(&wire, value.clone());
                    values.push(value);
                }
                LoweringFrame::FinishStructural {
                    wire,
                    environment,
                    kind,
                    output_type,
                    dependency_count,
                } => {
                    if values.len() < dependency_count {
                        return Err(LowerError::InvalidOperandArity {
                            expected: dependency_count,
                            actual: values.len(),
                        });
                    }
                    let arguments = values.split_off(values.len() - dependency_count);
                    match &kind {
                        NodeKind::ParallelLoop(specification) => {
                            self.queue_parallel_loop(
                                wire,
                                specification.clone(),
                                arguments,
                                environment,
                                output_type,
                                &mut work,
                            )?;
                            continue;
                        }
                        NodeKind::SequentialLoop(specification) => {
                            self.queue_sequential_loop(
                                wire,
                                specification.clone(),
                                arguments,
                                environment,
                                output_type,
                                &mut work,
                            )?;
                            continue;
                        }
                        _ => {}
                    }
                    let value = self.lower_structural_node(
                        &wire,
                        &kind,
                        &arguments,
                        &environment,
                        output_type,
                    )?;
                    self.finish_wire(&wire, value.clone());
                    values.push(value);
                }
                LoweringFrame::FinishParallelLoop {
                    wire,
                    body_source,
                    environment,
                    specification,
                    output_type,
                    binder,
                    logical_count,
                    maximum,
                } => {
                    let representative_value =
                        values.pop().ok_or(LowerError::MissingWire { wire: wire.source.wire })?;
                    let expected_element_type = match &output_type {
                        WireType::IndexedFamily { element, .. } => element.as_ref().clone(),
                        output_type => output_type.clone(),
                    };
                    let representative_value =
                        if let WireType::IndexedFamily { element, .. } = &output_type {
                            self.normalize_singleton_for_input(representative_value, element)?
                        } else {
                            representative_value
                        };
                    if let LoweredValue::Trapdoor(representative) = representative_value {
                        if !matches!(&output_type, WireType::IndexedFamily { element, .. } if matches!(element.as_ref(), WireType::Trapdoor { .. }))
                        {
                            return Err(LowerError::FamilyElementLoweringMismatch {
                                expected: expected_element_type,
                                actual_category: super::error::LoweredValueCategory::Trapdoor,
                                actual_sort: None,
                                producer: body_source,
                            });
                        }
                        let value =
                            LoweredValue::TrapdoorFamily { representative, binder, logical_count };
                        self.finish_wire(&wire, value.clone());
                        values.push(value);
                        continue;
                    }
                    if let LoweredValue::Matrix(representative) = representative_value {
                        let value = self.finish_parallel_loop_matrix(
                            &specification,
                            &environment,
                            output_type,
                            binder,
                            logical_count,
                            maximum,
                            representative,
                        )?;
                        self.finish_wire(&wire, value.clone());
                        values.push(value);
                        continue;
                    }
                    let LoweredValue::Scalar(representative) = representative_value else {
                        return Err(LowerError::FamilyElementLoweringMismatch {
                            expected: expected_element_type,
                            actual_category: Self::lowered_value_category(&representative_value)
                                .expect("non-term loop body"),
                            actual_sort: None,
                            producer: body_source,
                        });
                    };
                    let value = self.finish_parallel_loop(
                        &specification,
                        &environment,
                        output_type,
                        binder,
                        logical_count,
                        maximum,
                        representative,
                        body_source,
                    )?;
                    self.finish_wire(&wire, value.clone());
                    values.push(value);
                }
                LoweringFrame::FinishSequentialMatrixLoop {
                    wire,
                    environment,
                    count,
                    initial,
                    state_factors,
                    iteration_binder,
                    output_types,
                    output_type,
                    carried_index,
                    dependency_count,
                } => {
                    if values.len() < dependency_count {
                        return Err(LowerError::InvalidOperandArity {
                            expected: dependency_count,
                            actual: values.len(),
                        });
                    }
                    let transition_values = values.split_off(values.len() - dependency_count);
                    let transitions = transition_values
                        .into_iter()
                        .map(|value| match value {
                            LoweredValue::Matrix(term) => Ok(term),
                            _ => Err(LowerError::InvalidOperandArity {
                                expected: dependency_count,
                                actual: dependency_count,
                            }),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let value = self.finish_sequential_matrix_loop(
                        &wire,
                        &environment,
                        count,
                        initial,
                        state_factors,
                        iteration_binder,
                        transitions,
                        output_types,
                        output_type,
                        carried_index,
                    )?;
                    self.finish_wire(&wire, value.clone());
                    values.push(value);
                }
                LoweringFrame::FinishAlias { wire } => {
                    let value =
                        values.pop().ok_or(LowerError::MissingWire { wire: wire.source.wire })?;
                    self.finish_wire(&wire, value.clone());
                    values.push(value);
                }
                LoweringFrame::FinishProtocolTrapdoor { wire, environment, output_type } => {
                    let trapdoor =
                        self.protocol_input_trapdoor(&wire, &environment, &output_type)?.ok_or(
                            LowerError::FamilyProducerNotResolved { family: wire.source.wire },
                        )?;
                    let value = LoweredValue::Trapdoor(trapdoor);
                    self.finish_wire(&wire, value.clone());
                    values.push(value);
                }
                LoweringFrame::FinishValue { wire, value } => {
                    self.finish_wire(&wire, value.clone());
                    values.push(value);
                }
                LoweringFrame::FinishIndexedAlias { wire, index } => {
                    let value =
                        values.pop().ok_or(LowerError::MissingWire { wire: wire.source.wire })?;
                    let value = match value {
                        LoweredValue::Scalar(term) => LoweredValue::Scalar(self.scalar_term(term)?),
                        LoweredValue::Matrix(_) | LoweredValue::MatrixFamily(_) => {
                            return Err(LowerError::UnsupportedMatrixProductExpansion)
                        }
                        LoweredValue::Family(family) => self.family_element(&family, &index)?,
                        LoweredValue::Trapdoor(_) | LoweredValue::TrapdoorFamily { .. } => {
                            return Err(LowerError::FamilyProducerNotResolved {
                                family: wire.source.wire,
                            });
                        }
                    };
                    self.finish_wire(&wire, value.clone());
                    values.push(value);
                }
                LoweringFrame::FinishPreimage { wire, environment, cutoff, output_type } => {
                    let arguments = values.split_off(values.len().checked_sub(3).ok_or(
                        LowerError::InvalidOperandArity { expected: 3, actual: values.len() },
                    )?);
                    if let [
                        LoweredValue::Matrix(public),
                        LoweredValue::Trapdoor(trapdoor),
                        LoweredValue::Matrix(target),
                    ] = arguments.as_slice()
                    {
                        let preimage = self.matrix_preimage_atom(
                            &wire,
                            &environment,
                            &output_type,
                            &cutoff,
                            *trapdoor,
                        )?;
                        let public_key = match self.dag_factor_key(*public) {
                            Ok(key) => key,
                            Err(error) => {
                                return Err(error);
                            }
                        };
                        let target_key = match self.dag_factor_key(*target) {
                            Ok(key) => key,
                            Err(error) => {
                                if matches!(self.dag.node(*target), Ok(ExpressionNode::Zero)) {
                                    FactorIdentity {
                                        owner: FactorOwner::Derived {
                                            parent: Box::new(public_key.clone()),
                                            tag: b"exact-zero-target".to_vec().into_boxed_slice(),
                                        },
                                        kind: FactorKind::Signal,
                                        port: mxx_ir_core::Port(0),
                                        coordinates: Box::new([]),
                                        public: None,
                                        layout: None,
                                        selector: None,
                                        trapdoor: None,
                                        selector_mapping: Box::new([]),
                                    }
                                } else {
                                    return Err(error);
                                }
                            }
                        };
                        let preimage_key = self.dag_factor_key(preimage)?;
                        let matrix_type = match self.dag.node(preimage) {
                            Ok(ExpressionNode::Atom(factor)) => {
                                factor.matrix_bound.as_ref().map(|bound| bound.matrix_type.clone())
                            }
                            _ => None,
                        };
                        let trapdoor_key = self
                            .symbols
                            .trapdoors
                            .get(trapdoor.0)
                            .map(|descriptor| descriptor.source.clone());
                        let public_central = matches!(
                            self.dag.node(*public),
                            Ok(ExpressionNode::Atom(factor)) if factor.is_central_scalar()
                        );
                        let preimage_central = matches!(
                            self.dag.node(preimage),
                            Ok(ExpressionNode::Atom(factor)) if factor.is_central_scalar()
                        );
                        let registration = super::normal_form::RelationRegistration {
                            key: super::normal_form::FullRelationKey {
                                source: preimage_key.owner.clone(),
                                ordered_indices: public_key.coordinates.clone(),
                                public: public_key.clone(),
                                target: target_key,
                                matrix_type,
                                layout: public_key.layout.clone(),
                                trapdoor: trapdoor_key,
                                selector: public_key.selector.clone().map(|selector| {
                                    (selector, public_key.selector_mapping.clone())
                                }),
                            },
                            preimage: preimage_key.clone(),
                            target: *target,
                            pattern: RelationPattern::new(
                                [
                                    public_central.then(|| public_key.clone()),
                                    preimage_central.then(|| preimage_key.clone()),
                                ]
                                .into_iter()
                                .flatten(),
                                [
                                    (!public_central).then(|| public_key.clone()),
                                    (!preimage_central).then(|| preimage_key.clone()),
                                ]
                                .into_iter()
                                .flatten(),
                            ),
                        };
                        self.relation_registry
                            .register_pattern(registration.pattern.clone(), registration)
                            .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
                        let value = LoweredValue::Matrix(preimage);
                        self.finish_wire(&wire, value.clone());
                        values.push(value);
                        continue;
                    }
                    return Err(LowerError::UnsupportedMatrixProductExpansion);
                }
                LoweringFrame::FinishHashSample {
                    wire,
                    environment,
                    matrix_type,
                    variant,
                    base,
                    digit_count,
                    output_type,
                    dependency_count,
                } => {
                    let arguments =
                        values.split_off(values.len().checked_sub(dependency_count).ok_or(
                            LowerError::InvalidOperandArity {
                                expected: dependency_count,
                                actual: values.len(),
                            },
                        )?);
                    let arguments = arguments
                        .into_iter()
                        .map(|value| match value {
                            LoweredValue::Scalar(term) => {
                                let sort = self
                                    .scalar_store
                                    .facts(term)
                                    .and_then(|facts| facts.sort.as_ref().ok());
                                // Hash inputs are scalar/domain values that are consumed by the
                                // matrix source descriptor.  Bytes is intentionally allowed here
                                // as a non-matrix source domain; it is never emitted as a matrix
                                // carrier or passed to scalar arithmetic.
                                matches!(
                                    sort,
                                    Some(
                                        ScalarSort::Bytes(_) |
                                            ScalarSort::Int |
                                            ScalarSort::Bool |
                                            ScalarSort::Real
                                    )
                                )
                                .then_some(term)
                                .ok_or(LowerError::UnsupportedMatrixProductExpansion)
                            }
                            LoweredValue::Matrix(_) | LoweredValue::MatrixFamily(_) => {
                                Err(LowerError::UnsupportedMatrixProductExpansion)
                            }
                            LoweredValue::Family(_) |
                            LoweredValue::Trapdoor(_) |
                            LoweredValue::TrapdoorFamily { .. } => {
                                Err(LowerError::InvalidOperandArity {
                                    expected: dependency_count,
                                    actual: dependency_count,
                                })
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    if matches!(output_type, WireType::Matrix(_) | WireType::Preimage(_)) {
                        if variant == HashVariant::Plain {
                            let output_matrix =
                                self.resolve_matrix_type(&matrix_type, &environment)?;
                            let concrete = concrete_matrix_type(&output_matrix)
                                .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
                            let arguments = arguments
                                .iter()
                                .map(|term| {
                                    Ok(super::normal_form::PolynomialNF::exact_factor_typed(
                                        FactorIdentity::scalar_selector(
                                            self.canonical_scalar_identity(*term)?,
                                        ),
                                        concrete.clone(),
                                    ))
                                })
                                .collect::<Result<Vec<_>, LowerError>>()?;
                            let query = self.graph_factor_identity(
                                &wire,
                                &environment,
                                b"hash-plain-query",
                            )?;
                            let nf = <super::normal_form::PolynomialNF as AdditionalOperations>::hash_plain_nf(
                                query.clone(),
                                &arguments,
                                concrete.clone(),
                                None,
                            )
                            .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
                            let value =
                                self.push_nf_term(nf, query, concrete).map(LoweredValue::Matrix)?;
                            self.finish_wire(&wire, value.clone());
                            values.push(value);
                            continue;
                        }
                        // Keep the structural hash contract checks on the DAG path.  In
                        // particular, decomposed hashes still require an integral row/digit
                        // layout before an atom can be admitted; otherwise malformed Graph IR
                        // would be silently represented as a finite sampler.
                        if matches!(variant, HashVariant::Decomposed | HashVariant::SmallDecomposed)
                        {
                            let (Some(base), Some(digit_count)) =
                                (base.clone(), digit_count.clone())
                            else {
                                return Err(LowerError::InvalidOperandArity {
                                    expected: 2,
                                    actual: 0,
                                });
                            };
                            let resolved_matrix =
                                self.resolve_matrix_type(&matrix_type, &environment)?;
                            let base = self.resolve_int(&base, &environment)?;
                            let digit_count = self.resolve_int(&digit_count, &environment)?;
                            let Some(base_value) = resolved_integer(&base) else {
                                return Err(LowerError::NonExactIdentityIndex {
                                    expression: IntExpr::constant(0),
                                });
                            };
                            let Some(digit_count_value) = resolved_nonnegative(&digit_count)
                                .and_then(|value| value.to_usize())
                            else {
                                return Err(LowerError::NonExactIdentityIndex {
                                    expression: IntExpr::constant(0),
                                });
                            };
                            let Some(output_rows) = resolved_nonnegative(&resolved_matrix.rows)
                                .and_then(|value| value.to_usize())
                            else {
                                return Err(LowerError::NonExactIdentityIndex {
                                    expression: matrix_type.rows.clone(),
                                });
                            };
                            if base_value <= BigInt::from(1) ||
                                digit_count_value == 0 ||
                                output_rows == 0 ||
                                output_rows % digit_count_value != 0
                            {
                                return Err(LowerError::InvalidOperandArity {
                                    expected: digit_count_value,
                                    actual: output_rows,
                                });
                            }
                        }
                        let role = match variant {
                            HashVariant::Decomposed => {
                                Some(super::identity::AtomicRelationRole::DecomposedHash)
                            }
                            HashVariant::SmallDecomposed => {
                                Some(super::identity::AtomicRelationRole::SmallDecomposedHash {
                                    range_proved: false,
                                })
                            }
                            _ => None,
                        };
                        let value = LoweredValue::Matrix(self.matrix_atom_for_wire(
                            &wire,
                            &environment,
                            output_type,
                            role,
                        )?);
                        self.finish_wire(&wire, value.clone());
                        values.push(value);
                        continue;
                    }
                    return Err(LowerError::UnsupportedMatrixProductExpansion);
                }
                LoweringFrame::FinishGadgetDecompose {
                    wire,
                    environment,
                    base,
                    digit_count,
                    small,
                    output_type,
                } => {
                    let argument = values
                        .pop()
                        .ok_or(LowerError::InvalidOperandArity { expected: 1, actual: 0 })?;
                    self.validate_gadget_decompose(
                        &base,
                        &digit_count,
                        &environment,
                        &output_type,
                        &argument,
                    )?;
                    if matches!(output_type, WireType::Matrix(_) | WireType::Preimage(_)) {
                        let value = LoweredValue::Matrix(self.matrix_atom_for_wire(
                            &wire,
                            &environment,
                            output_type,
                            Some(if small {
                                super::identity::AtomicRelationRole::SmallGadgetDecomposition {
                                    range_proved: false,
                                }
                            } else {
                                super::identity::AtomicRelationRole::GadgetDecomposition
                            }),
                        )?);
                        self.finish_wire(&wire, value.clone());
                        values.push(value);
                        continue;
                    }
                    return Err(LowerError::UnsupportedMatrixProductExpansion);
                }
            }
        }
        values.pop().ok_or(LowerError::MissingWire {
            wire: WireRef { node: mxx_ir_core::NodeId(0), port: mxx_ir_core::Port(0) },
        })
    }

    fn graph_for_program(
        &self,
        program: &super::identity::ProgramKey,
    ) -> Result<&mxx_ir_core::Graph, LowerError> {
        let super::identity::ProgramKey::WorkflowStage(stage) = program else {
            return Err(LowerError::MissingWire {
                wire: WireRef { node: mxx_ir_core::NodeId(0), port: mxx_ir_core::Port(0) },
            });
        };
        self.protocol
            .stages()
            .iter()
            .find(|candidate| &candidate.id == stage)
            .map(|stage| &stage.graph)
            .ok_or(LowerError::MissingWire {
                wire: WireRef { node: mxx_ir_core::NodeId(0), port: mxx_ir_core::Port(0) },
            })
    }

    fn lower_source_family(
        &mut self,
        wire: &LoweringWire,
        environment: &LowerEnv,
        element: WireType,
        count: IntExpr,
    ) -> Result<LoweredValue, LowerError> {
        let count_value = self.lower_int_expr(&count, environment)?;
        let count_range = self
            .integer_analysis(count_value.scalar)
            .and_then(|(domain, _)| domain.interval().ok())
            .ok_or_else(|| LowerError::InvalidFamilyCount { count: count.clone() })?;
        if count_range.minimum != count_range.maximum || count_range.minimum <= BigInt::zero() {
            return Err(LowerError::InvalidFamilyCount { count });
        }
        let logical_count = count_range
            .minimum
            .to_biguint()
            .ok_or_else(|| LowerError::InvalidFamilyCount { count: count.clone() })?;
        let binder = BinderKey {
            loop_scope: environment.occurrence.clone(),
            loop_node: wire.source.wire.node,
            slot: wire.source.wire.port.0,
        };
        self.symbols.binders.intern(super::identity::BinderDescriptor {
            key: binder.clone(),
            minimum: BigInt::zero(),
            maximum: count_range.minimum - BigInt::from(1_u8),
        });
        let scalar = self
            .scalar_store
            .intern_node(ScalarNode::IntBinder(binder.clone()), &self.symbols)
            .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
        let index = LoweredInt { scalar };
        let mut indexed_wire = wire.clone();
        indexed_wire.indices = wire.indices.iter().cloned().chain([index.clone()]).collect();
        let mut indexed_environment = environment.clone();
        indexed_environment.active_coordinates.push(Coordinate { binder: binder.clone(), index });
        let element_type = self.resolve_family_element_sort(&element, &indexed_environment)?;
        if matches!(element, WireType::Matrix(_) | WireType::Preimage(_)) {
            let representative =
                self.matrix_atom_for_wire(&indexed_wire, &indexed_environment, element, None)?;
            return Ok(LoweredValue::MatrixFamily(FamilyLoweringValue {
                element_type,
                storage: FamilyCoverageStorage::SharedTemplate {
                    domain: family::LoopDomainKey { binder: binder.clone(), logical_count },
                    representative,
                    binder_domains: vec![family::CoverageBinderDomain {
                        binder,
                        minimum: BigInt::zero(),
                        maximum: count_range.maximum - BigInt::from(1_u8),
                    }]
                    .into_boxed_slice(),
                },
            }));
        }
        let representative =
            self.atom_for_wire(&indexed_wire, &indexed_environment, element, None)?;
        Ok(LoweredValue::Family(FamilyLoweringValue {
            element_type,
            storage: FamilyCoverageStorage::SharedTemplate {
                domain: family::LoopDomainKey { binder: binder.clone(), logical_count },
                representative,
                binder_domains: vec![family::CoverageBinderDomain {
                    binder,
                    minimum: BigInt::zero(),
                    maximum: count_range.maximum - BigInt::from(1_u8),
                }]
                .into_boxed_slice(),
            },
        }))
    }

    fn atom_for_wire(
        &mut self,
        wire: &LoweringWire,
        environment: &LowerEnv,
        ty: WireType,
        relation_role: Option<super::identity::AtomicRelationRole>,
    ) -> Result<ScalarId, LowerError> {
        let sort = match ty {
            WireType::Int | WireType::ConstantInt => ScalarSort::Int,
            WireType::Bool | WireType::ConstantBool => ScalarSort::Bool,
            WireType::Real | WireType::ConstantReal => ScalarSort::Real,
            WireType::Bytes { length } => {
                ScalarSort::Bytes(self.resolve_int(&length, environment)?)
            }
            WireType::TypedBlob { type_name, schema_hash } => {
                ScalarSort::TypedBlob { type_name, schema_hash }
            }
            WireType::Matrix(matrix) | WireType::Preimage(matrix) => {
                ScalarSort::Matrix(self.resolve_matrix_type(&matrix, environment)?)
            }
            WireType::Trapdoor { .. } | WireType::IndexedFamily { .. } => {
                return Err(LowerError::FamilyProducerNotResolved { family: wire.source.wire });
            }
        };
        let graph_source = super::identity::GraphWireSourceKey {
            wire: wire.source.clone(),
            coordinate_binders: environment
                .active_coordinates
                .iter()
                .map(|coordinate| coordinate.binder.clone())
                .collect(),
        };
        let (key, integer_domain, canonical_residue_convention) = if let Some(sampler) =
            self.non_relation_sampler_for_wire(wire, environment)?
        {
            (super::identity::AtomicSourceKey::Sampler(SamplerDescriptorId(sampler)), None, None)
        } else {
            match self.protocol_input_source(wire, environment, &sort)? {
                Some(protocol) => protocol,
                None => (
                    if self.is_explicit_large_source(wire)? {
                        super::identity::AtomicSourceKey::ExplicitLarge(graph_source)
                    } else {
                        super::identity::AtomicSourceKey::GraphWire(graph_source)
                    },
                    None,
                    None,
                ),
            }
        };
        let descriptor = super::identity::AtomicSourceDescriptor {
            key: key.clone(),
            sort: sort.clone(),
            integer_domain,
            canonical_residue_convention,
            relation_role,
        };
        let source = self.symbols.atomic_sources.intern(descriptor);
        self.scalar_store
            .intern_node(
                ScalarNode::Source {
                    source: super::identity::AtomicSourceId(source),
                    indices: environment
                        .active_coordinates
                        .iter()
                        .map(|coordinate| coordinate.index.scalar)
                        .collect(),
                },
                &self.symbols,
            )
            .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)
    }

    /// Lowers a matrix source directly into the job DAG.  Scalar operands are
    /// intentionally not used as an intermediate matrix representation here.
    fn matrix_atom_for_wire(
        &mut self,
        wire: &LoweringWire,
        environment: &LowerEnv,
        ty: WireType,
        relation_role: Option<super::identity::AtomicRelationRole>,
    ) -> Result<TermId, LowerError> {
        let matrix = match ty {
            WireType::Matrix(matrix) | WireType::Preimage(matrix) => {
                self.resolve_matrix_type(&matrix, environment)?
            }
            actual => {
                return Err(LowerError::InvalidOperandSort {
                    expected: WireType::Matrix(MatrixType {
                        modulus: IntExpr::constant(1),
                        ring_dimension: IntExpr::constant(1),
                        rows: IntExpr::constant(1),
                        columns: IntExpr::constant(1),
                    }),
                    actual,
                });
            }
        };
        let graph_source = super::identity::GraphWireSourceKey {
            wire: wire.source.clone(),
            coordinate_binders: environment
                .active_coordinates
                .iter()
                .map(|coordinate| coordinate.binder.clone())
                .collect(),
        };
        let key = if let Some(sampler) = self.non_relation_sampler_for_wire(wire, environment)? {
            super::identity::AtomicSourceKey::Sampler(SamplerDescriptorId(sampler))
        } else {
            match self.protocol_input_source(
                wire,
                environment,
                &ScalarSort::Matrix(matrix.clone()),
            )? {
                Some((key, _, _)) => key,
                None => {
                    if self.is_explicit_large_source(wire)? {
                        super::identity::AtomicSourceKey::ExplicitLarge(graph_source)
                    } else {
                        super::identity::AtomicSourceKey::GraphWire(graph_source)
                    }
                }
            }
        };
        let coordinates = environment
            .active_coordinates
            .iter()
            .map(|coordinate| {
                let index = self.canonical_scalar_identity(coordinate.index.scalar)?;
                Ok((coordinate.binder.clone(), index))
            })
            .collect::<Result<Box<[_]>, LowerError>>()?;
        let factor_key = FactorIdentity::atomic(key.clone(), coordinates);
        let protocol_contract = match &key {
            super::identity::AtomicSourceKey::ProtocolInput(input) => {
                self.protocol_matrix_contract_bound(input, &matrix)?
            }
            _ => None,
        };
        if matches!(key, super::identity::AtomicSourceKey::ProtocolInput(_)) &&
            protocol_contract.is_none()
        {
            return Err(LowerError::UnsupportedMatrixProductExpansion);
        }
        let coefficient_class = match protocol_contract.as_ref().map(|(class, _)| class) {
            Some(class) => class.clone(),
            None => match relation_role {
                Some(
                    super::identity::AtomicRelationRole::DecomposedHash |
                    super::identity::AtomicRelationRole::GadgetDecomposition,
                ) => {
                    let kind = self
                        .graph_for_program(&wire.source.scope.program)?
                        .scope(&wire.source.scope.definition)
                        .and_then(|scope| scope.node(wire.source.wire.node))
                        .map(|node| node.kind().clone());
                    let base = match kind {
                        Some(NodeKind::HashSample { base: Some(base), .. }) => {
                            self.resolve_int(&base, environment)?
                        }
                        Some(NodeKind::GadgetDecompose { base, .. }) => {
                            self.resolve_int(&base, environment)?
                        }
                        _ => return Err(LowerError::UnsupportedMatrixProductExpansion),
                    };
                    gadget_digit_bound(
                        &resolved_integer(&base)
                            .ok_or(LowerError::UnsupportedMatrixProductExpansion)?,
                        false,
                    )
                    .ok_or(LowerError::UnsupportedMatrixProductExpansion)?
                }
                Some(
                    super::identity::AtomicRelationRole::SmallDecomposedHash { .. } |
                    super::identity::AtomicRelationRole::SmallGadgetDecomposition { .. },
                ) => {
                    let kind = self
                        .graph_for_program(&wire.source.scope.program)?
                        .scope(&wire.source.scope.definition)
                        .and_then(|scope| scope.node(wire.source.wire.node))
                        .map(|node| node.kind().clone());
                    let base = match kind {
                        Some(NodeKind::HashSample { base: Some(base), .. }) => {
                            self.resolve_int(&base, environment)?
                        }
                        Some(NodeKind::GadgetDecompose { base, .. }) => {
                            self.resolve_int(&base, environment)?
                        }
                        _ => return Err(LowerError::UnsupportedMatrixProductExpansion),
                    };
                    gadget_digit_bound(
                        &resolved_integer(&base)
                            .ok_or(LowerError::UnsupportedMatrixProductExpansion)?,
                        true,
                    )
                    .ok_or(LowerError::UnsupportedMatrixProductExpansion)?
                }
                _ => match &key {
                    super::identity::AtomicSourceKey::Sampler(id) => {
                        let sampler = self
                            .symbols
                            .samplers
                            .get(id.0)
                            .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
                        match sampler {
                            SamplerIdentity::Gaussian { max_coefficient_bound, .. } => {
                                resolved_integer(max_coefficient_bound)
                                    .and_then(|value| value.to_biguint())
                                    .map(BoundClass::bounded)
                                    .ok_or(LowerError::UnsupportedMatrixProductExpansion)?
                            }
                            SamplerIdentity::UniformInterval { minimum, maximum, .. } => {
                                let minimum = resolved_integer(minimum)
                                    .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
                                let maximum = resolved_integer(maximum)
                                    .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
                                if minimum > maximum {
                                    return Err(LowerError::UnsupportedMatrixProductExpansion);
                                }
                                BoundClass::bounded(
                                    minimum
                                        .abs()
                                        .max(maximum.abs())
                                        .to_biguint()
                                        .expect("absolute bound"),
                                )
                            }
                            SamplerIdentity::Preimage { cutoff, .. } => resolved_integer(cutoff)
                                .and_then(|value| value.to_biguint())
                                .map(BoundClass::bounded)
                                .ok_or(LowerError::UnsupportedMatrixProductExpansion)?,
                            SamplerIdentity::DecomposedHash { base, small, .. } |
                            SamplerIdentity::GadgetDecomposition { base, small, .. } => {
                                gadget_digit_bound(
                                    &resolved_integer(base)
                                        .ok_or(LowerError::UnsupportedMatrixProductExpansion)?,
                                    *small,
                                )
                                .ok_or(LowerError::UnsupportedMatrixProductExpansion)?
                            }
                        }
                    }
                    _ => BoundClass::Large,
                },
            },
        };
        let bound = MatrixBound {
            matrix_type: concrete_matrix_type(&matrix)
                .ok_or(LowerError::UnsupportedMatrixProductExpansion)?,
            coefficient_class,
        };
        self.push_matrix_atom_with_metadata(
            factor_key,
            bound,
            matches!(relation_role, Some(super::identity::AtomicRelationRole::Preimage)),
            protocol_contract.map(|(_, metadata)| metadata).unwrap_or_else(MatrixMetadata::unknown),
        )
    }

    fn protocol_matrix_contract_bound(
        &self,
        input: &crate::ProtocolInputId,
        matrix: &super::identity::ResolvedMatrixType,
    ) -> Result<Option<(BoundClass, MatrixMetadata)>, LowerError> {
        let mut contract = self
            .protocol
            .bundle
            .input_contract
            .inputs
            .iter()
            .find(|entry| entry.id == *input)
            .map(|entry| &entry.value);
        while let Some(InputValueContract::Family { element, .. }) = contract {
            contract = Some(element);
        }
        let Some(contract) = contract else { return Ok(None) };
        match contract {
            InputValueContract::MatrixExact {
                canonical_coefficient_exclusive_upper_bound: Some(upper),
                is_constant_polynomial,
                ..
            } => {
                let mut env = mxx_ir_core::ParamEnv::default();
                env.integers.extend(self.request.environment.iter().filter_map(|(name, value)| {
                    match value {
                        super::OperationalParameterValue::Integer(value) => {
                            Some((name.clone(), value.clone()))
                        }
                        super::OperationalParameterValue::Rational { .. } => None,
                    }
                }));
                let upper = upper
                    .evaluate(&env)
                    .ok()
                    .and_then(|value| value.to_biguint())
                    .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
                let modulus = resolved_integer(&matrix.modulus)
                    .and_then(|value| value.to_biguint())
                    .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
                if upper.is_zero() || upper > modulus {
                    return Err(LowerError::UnsupportedMatrixProductExpansion);
                }
                Ok(Some((
                    BoundClass::bounded(upper.clone() - BigUint::one()),
                    MatrixMetadata {
                        canonical_coefficient_exclusive_upper: Some(upper.clone()),
                        is_constant_polynomial: *is_constant_polynomial,
                        known_zero_rows: None,
                        polynomial: None,
                    },
                )))
            }
            InputValueContract::MatrixLarge { .. } => {
                Ok(Some((BoundClass::Large, MatrixMetadata::unknown())))
            }
            InputValueContract::MatrixBounded { max_centered_coefficient, .. } => {
                let bound = self
                    .declared_bound_value(max_centered_coefficient)
                    .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
                Ok(Some((BoundClass::bounded(bound), MatrixMetadata::unknown())))
            }
            _ => Ok(None),
        }
    }

    fn declared_bound_value(&self, expression: &DeclaredBoundExpr) -> Option<BigUint> {
        match expression {
            DeclaredBoundExpr::Constant(value) => Some(value.clone()),
            DeclaredBoundExpr::Parameter(value) => {
                let mut env = mxx_ir_core::ParamEnv::default();
                env.integers.extend(self.request.environment.iter().filter_map(|(name, value)| {
                    match value {
                        super::OperationalParameterValue::Integer(value) => {
                            Some((name.clone(), value.clone()))
                        }
                        super::OperationalParameterValue::Rational { .. } => None,
                    }
                }));
                value.evaluate(&env).ok().and_then(|value| value.to_biguint())
            }
            DeclaredBoundExpr::Add(left, right) => {
                Some(self.declared_bound_value(left)? + self.declared_bound_value(right)?)
            }
            DeclaredBoundExpr::Multiply(left, right) => {
                Some(self.declared_bound_value(left)? * self.declared_bound_value(right)?)
            }
            DeclaredBoundExpr::Maximum(left, right) => {
                Some(self.declared_bound_value(left)?.max(self.declared_bound_value(right)?))
            }
            DeclaredBoundExpr::Minimum(left, right) => {
                Some(self.declared_bound_value(left)?.min(self.declared_bound_value(right)?))
            }
            DeclaredBoundExpr::Absolute(value) => {
                let mut env = mxx_ir_core::ParamEnv::default();
                env.integers.extend(self.request.environment.iter().filter_map(|(name, value)| {
                    match value {
                        super::OperationalParameterValue::Integer(value) => {
                            Some((name.clone(), value.clone()))
                        }
                        super::OperationalParameterValue::Rational { .. } => None,
                    }
                }));
                value.evaluate(&env).ok().and_then(|value| value.abs().to_biguint())
            }
            DeclaredBoundExpr::FloorDivide { value, positive_divisor } => {
                if positive_divisor.is_zero() {
                    None
                } else {
                    Some(self.declared_bound_value(value)? / positive_divisor)
                }
            }
            DeclaredBoundExpr::MatrixProduct { ring_dimension, inner_dimension, left, right } => {
                let mut env = mxx_ir_core::ParamEnv::default();
                env.integers.extend(self.request.environment.iter().filter_map(|(name, value)| {
                    match value {
                        super::OperationalParameterValue::Integer(value) => {
                            Some((name.clone(), value.clone()))
                        }
                        super::OperationalParameterValue::Rational { .. } => None,
                    }
                }));
                let ring_dimension = ring_dimension.evaluate(&env).ok()?.to_biguint()?;
                let inner_dimension = inner_dimension.evaluate(&env).ok()?.to_biguint()?;
                if ring_dimension.is_zero() || inner_dimension.is_zero() {
                    return None;
                }
                Some(
                    ring_dimension *
                        inner_dimension *
                        self.declared_bound_value(left)? *
                        self.declared_bound_value(right)?,
                )
            }
        }
    }

    fn push_matrix_atom_with_metadata(
        &mut self,
        key: FactorIdentity,
        bound: MatrixBound,
        relation_live: bool,
        metadata: MatrixMetadata,
    ) -> Result<TermId, LowerError> {
        let factor = if relation_live {
            SymbolicFactor::relation_live(key, bound)
                .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?
        } else if matches!(bound.coefficient_class, BoundClass::Large) {
            SymbolicFactor::large_with_metadata(key, bound, metadata.clone())
        } else {
            SymbolicFactor::bounded_with_metadata(key, bound, metadata.clone())
                .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?
        };
        self.dag
            .push(ExpressionNode::Atom(factor))
            .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)
    }

    fn push_matrix_atom_with_trapdoor(
        &mut self,
        key: FactorIdentity,
        bound: MatrixBound,
        relation_live: bool,
        trapdoor: TrapdoorSourceKey,
    ) -> Result<TermId, LowerError> {
        let factor = if relation_live {
            SymbolicFactor::relation_live(key, bound)
        } else if matches!(bound.coefficient_class, BoundClass::Large) {
            Ok(SymbolicFactor::large_with_metadata(key, bound, MatrixMetadata::unknown()))
        } else {
            SymbolicFactor::bounded(key, bound)
        }
        .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?
        .with_trapdoor(trapdoor);
        self.dag
            .push(ExpressionNode::Atom(factor))
            .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)
    }

    fn dag_factor_key(&self, term: TermId) -> Result<FactorIdentity, LowerError> {
        match self.dag.node(term).map_err(|_| LowerError::UnsupportedMatrixProductExpansion)? {
            ExpressionNode::Atom(factor) => Ok(factor.key.clone()),
            _ => Err(LowerError::UnsupportedMatrixProductExpansion),
        }
    }

    fn dag_matrix_type(
        &self,
        term: TermId,
    ) -> Result<mxx_ir_core::types::ConcreteMatrixType, LowerError> {
        Ok(self
            .dag
            .facts(term)
            .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?
            .concrete_type
            .clone())
    }

    fn graph_factor_identity(
        &self,
        wire: &LoweringWire,
        environment: &LowerEnv,
        tag: &[u8],
    ) -> Result<FactorIdentity, LowerError> {
        let source = super::identity::GraphWireSourceKey {
            wire: wire.source.clone(),
            coordinate_binders: environment
                .active_coordinates
                .iter()
                .map(|coordinate| coordinate.binder.clone())
                .collect(),
        };
        let coordinates = environment
            .active_coordinates
            .iter()
            .map(|coordinate| {
                Ok((
                    coordinate.binder.clone(),
                    self.canonical_scalar_identity(coordinate.index.scalar)?,
                ))
            })
            .collect::<Result<Box<_>, LowerError>>()?;
        let mut key = FactorIdentity::atomic(
            super::identity::AtomicSourceKey::GraphWire(source),
            coordinates,
        );
        key.owner = FactorOwner::Derived {
            parent: Box::new(key.clone()),
            tag: tag.to_vec().into_boxed_slice(),
        };
        Ok(key)
    }

    fn push_nf_term(
        &mut self,
        value: super::normal_form::PolynomialNF,
        fallback: FactorIdentity,
        matrix_type: mxx_ir_core::types::ConcreteMatrixType,
    ) -> Result<TermId, LowerError> {
        if value.is_exact_zero() {
            let bound = MatrixBound { matrix_type, coefficient_class: BoundClass::ExactZero };
            let factor = SymbolicFactor::bounded(fallback, bound)
                .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
            return self
                .dag
                .push(ExpressionNode::Atom(factor))
                .map_err(|_| LowerError::UnsupportedMatrixProductExpansion);
        }
        if value.exact_terms().len() == 1 && value.bounded_summary().is_exact_zero() {
            let term = value.exact_terms().values().next().expect("one exact term");
            if term.multiplicity == BigInt::from(1) && term.monomial.factors().len() == 1 {
                return self
                    .dag
                    .push(ExpressionNode::Atom(term.monomial.factors()[0].clone()))
                    .map_err(|_| LowerError::UnsupportedMatrixProductExpansion);
            }
        }
        let bound = value
            .bounded_summary()
            .as_matrix_bound()
            .cloned()
            .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
        let factor = SymbolicFactor::bounded(fallback, bound)
            .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
        self.dag
            .push(ExpressionNode::Atom(factor))
            .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)
    }

    fn lower_extract_coefficient_dag(
        &mut self,
        matrix: TermId,
        position: LoweredInt,
        explicit_upper: Option<&BigUint>,
        _environment: &LowerEnv,
    ) -> Result<LoweredValue, LowerError> {
        let concrete = self.dag_matrix_type(matrix)?;
        let modulus = concrete.modulus.to_biguint().ok_or(
            LowerError::InvalidExtractCoefficientCanonicalUpper {
                upper: BigUint::zero(),
                modulus: BigUint::zero(),
            },
        )?;
        let facts =
            self.dag.facts(matrix).map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
        let explicit_upper = explicit_upper.cloned();
        let authoritative_upper = explicit_upper
            .as_ref()
            .or(facts.metadata.canonical_coefficient_exclusive_upper.as_ref());
        if let Some(upper) = authoritative_upper {
            if upper.is_zero() || upper > &modulus {
                return Err(LowerError::InvalidExtractCoefficientCanonicalUpper {
                    upper: upper.clone(),
                    modulus,
                });
            }
        }
        let resolved_matrix = super::identity::ResolvedMatrixType {
            modulus: ResolvedIntExpr::Const(concrete.modulus.clone()),
            ring_dimension: ResolvedIntExpr::Const(concrete.ring_dimension.into()),
            rows: ResolvedIntExpr::Const(concrete.rows.into()),
            columns: ResolvedIntExpr::Const(concrete.columns.into()),
        };
        let data = direct_extract_facts(resolved_matrix, modulus, authoritative_upper)
            .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
        let scalar = self
            .scalar_store
            .intern_extract(
                self.dag
                    .resolved_identity(facts.identity)
                    .ok_or(LowerError::UnsupportedMatrixProductExpansion)?,
                position.scalar,
                data,
            )
            .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
        Ok(LoweredValue::Scalar(scalar))
    }

    fn lower_matrix_constant_dag(
        &mut self,
        wire: &LoweringWire,
        value: &mxx_ir_core::node::ConstantMatrix,
        matrix_type: &MatrixType,
        environment: &LowerEnv,
    ) -> Result<LoweredValue, LowerError> {
        let matrix_type = self.resolve_matrix_type(matrix_type, environment)?;
        let constant = match value {
            mxx_ir_core::node::ConstantMatrix::Zero => super::identity::MatrixConstantValue::Zero,
            mxx_ir_core::node::ConstantMatrix::Identity => {
                super::identity::MatrixConstantValue::Identity
            }
            mxx_ir_core::node::ConstantMatrix::UnitRow { index } => {
                super::identity::MatrixConstantValue::UnitRow {
                    index: self.resolve_int(index, environment)?,
                }
            }
            mxx_ir_core::node::ConstantMatrix::UnitColumn { index } => {
                super::identity::MatrixConstantValue::UnitColumn {
                    index: self.resolve_int(index, environment)?,
                }
            }
            mxx_ir_core::node::ConstantMatrix::Gadget { base, small } => {
                super::identity::MatrixConstantValue::Gadget {
                    base: self.resolve_int(base, environment)?,
                    small: *small,
                }
            }
            mxx_ir_core::node::ConstantMatrix::PowerOfBase { base, exponent } => {
                super::identity::MatrixConstantValue::PowerOfBase {
                    base: self.resolve_int(base, environment)?,
                    exponent: self.resolve_int(exponent, environment)?,
                }
            }
            mxx_ir_core::node::ConstantMatrix::Rotation { exponent } => {
                super::identity::MatrixConstantValue::Rotation {
                    exponent: self.resolve_int(exponent, environment)?,
                }
            }
            mxx_ir_core::node::ConstantMatrix::Polynomial { coefficients } => {
                super::identity::MatrixConstantValue::Polynomial {
                    coefficients: coefficients
                        .iter()
                        .map(|coefficient| self.resolve_int(coefficient, environment))
                        .collect::<Result<Box<_>, _>>()?,
                }
            }
        };
        let resolved = match constant {
            super::identity::MatrixConstantValue::Zero => ResolvedMatrixConstant::Zero,
            super::identity::MatrixConstantValue::Identity => ResolvedMatrixConstant::Identity,
            super::identity::MatrixConstantValue::UnitRow { index } => {
                ResolvedMatrixConstant::UnitRow {
                    index: resolved_nonnegative(&index)
                        .ok_or(LowerError::UnsupportedMatrixProductExpansion)?,
                }
            }
            super::identity::MatrixConstantValue::UnitColumn { index } => {
                ResolvedMatrixConstant::UnitColumn {
                    index: resolved_nonnegative(&index)
                        .ok_or(LowerError::UnsupportedMatrixProductExpansion)?,
                }
            }
            super::identity::MatrixConstantValue::Gadget { base, small } => {
                ResolvedMatrixConstant::Gadget {
                    base: resolved_integer(&base)
                        .ok_or(LowerError::UnsupportedMatrixProductExpansion)?,
                    small,
                }
            }
            super::identity::MatrixConstantValue::PowerOfBase { base, exponent } => {
                ResolvedMatrixConstant::PowerOfBase {
                    base: resolved_integer(&base)
                        .ok_or(LowerError::UnsupportedMatrixProductExpansion)?,
                    exponent: resolved_nonnegative(&exponent)
                        .ok_or(LowerError::UnsupportedMatrixProductExpansion)?,
                }
            }
            super::identity::MatrixConstantValue::Rotation { exponent } => {
                ResolvedMatrixConstant::Rotation {
                    exponent: resolved_integer(&exponent)
                        .ok_or(LowerError::UnsupportedMatrixProductExpansion)?,
                }
            }
            super::identity::MatrixConstantValue::Polynomial { coefficients } => {
                ResolvedMatrixConstant::Polynomial {
                    coefficients: coefficients
                        .iter()
                        .map(|coefficient| resolved_integer(coefficient))
                        .collect::<Option<Vec<_>>>()
                        .ok_or(LowerError::UnsupportedMatrixProductExpansion)?
                        .into_boxed_slice(),
                }
            }
        };
        let matrix_type = concrete_matrix_type(&matrix_type)
            .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
        let key = self.graph_factor_identity(wire, environment, b"constant-matrix")?;
        let nf = <super::normal_form::PolynomialNF as AdditionalOperations>::matrix_constant_nf(
            key.clone(),
            &resolved,
            matrix_type.clone(),
        )
        .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
        self.push_nf_term(nf, key, matrix_type).map(LoweredValue::Matrix)
    }

    fn lower_lift_constant_dag(
        &mut self,
        wire: &LoweringWire,
        input: &LoweredValue,
        matrix_type: &MatrixType,
        environment: &LowerEnv,
    ) -> Result<LoweredValue, LowerError> {
        let (interval, direct_extract_upper, selector_only) = match input {
            LoweredValue::Scalar(id) => {
                let entry = self
                    .scalar_store
                    .get(*id)
                    .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
                let interval = entry
                    .analysis
                    .integer_domain
                    .as_ref()
                    .and_then(|domain| domain.interval().ok())
                    .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
                let interval = super::scalar::IntegerInterval {
                    minimum: interval.minimum,
                    maximum: interval.maximum,
                };
                (
                    interval,
                    entry
                        .analysis
                        .direct_extract
                        .as_ref()
                        .and_then(|fact| fact.canonical_upper.clone()),
                    entry.analysis.scalar_provenance ==
                        Some(super::scalar::ScalarProvenance::SelectorOnly),
                )
            }
            _ => return Err(LowerError::UnsupportedMatrixProductExpansion),
        };
        if selector_only && direct_extract_upper.is_none() {
            return Err(LowerError::SelectorOnlyValueUsedByForbiddenConsumer {
                consumer: SelectorOnlyConsumer::LiftConstantPolynomial,
            });
        }
        let matrix_type =
            concrete_matrix_type(&self.resolve_matrix_type(matrix_type, environment)?)
                .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
        let key = self.graph_factor_identity(wire, environment, b"lift-constant-polynomial")?;
        let source = super::normal_form::PolynomialNF::bounded(MatrixBound {
            matrix_type: matrix_type.clone(),
            coefficient_class: BoundClass::bounded(
                interval.minimum.magnitude().max(interval.maximum.magnitude()).clone(),
            ),
        })
        .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
        let domain = IntegerInterval::new(interval.minimum.clone(), interval.maximum.clone())
            .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
        let domain = match direct_extract_upper {
            Some(upper) => domain.selector_only_direct_extract(upper),
            None => domain,
        };
        let nf = source
            .lift_constant_polynomial_nf(matrix_type.clone(), &domain)
            .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
        self.push_nf_term(nf, key, matrix_type).map(LoweredValue::Matrix)
    }

    fn matrix_preimage_atom(
        &mut self,
        wire: &LoweringWire,
        environment: &LowerEnv,
        output_type: &WireType,
        cutoff: &IntExpr,
        _trapdoor: TrapdoorDescriptorId,
    ) -> Result<TermId, LowerError> {
        let (WireType::Matrix(matrix) | WireType::Preimage(matrix)) = output_type else {
            return Err(LowerError::UnsupportedMatrixProductExpansion);
        };
        let matrix = self.resolve_matrix_type(matrix, environment)?;
        let graph_source = super::identity::GraphWireSourceKey {
            wire: wire.source.clone(),
            coordinate_binders: environment
                .active_coordinates
                .iter()
                .map(|coordinate| coordinate.binder.clone())
                .collect(),
        };
        let key = FactorIdentity::atomic(
            super::identity::AtomicSourceKey::GraphWire(graph_source),
            environment
                .active_coordinates
                .iter()
                .map(|coordinate| {
                    Ok((
                        coordinate.binder.clone(),
                        self.canonical_scalar_identity(coordinate.index.scalar)?,
                    ))
                })
                .collect::<Result<Box<[_]>, LowerError>>()?,
        );
        let cutoff = self.resolve_int(cutoff, environment)?;
        let coefficient = resolved_integer(&cutoff)
            .and_then(|value| value.to_biguint())
            .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
        let matrix_bound = MatrixBound {
            matrix_type: concrete_matrix_type(&matrix)
                .ok_or(LowerError::UnsupportedMatrixProductExpansion)?,
            coefficient_class: BoundClass::bounded(coefficient),
        };
        let trapdoor_source = self
            .symbols
            .trapdoors
            .get(_trapdoor.0)
            .map(|descriptor| descriptor.source.clone())
            .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
        let preimage =
            self.push_matrix_atom_with_trapdoor(key, matrix_bound, true, trapdoor_source)?;
        Ok(preimage)
    }

    fn is_explicit_large_source(&self, wire: &LoweringWire) -> Result<bool, LowerError> {
        let node = self
            .graph_for_program(&wire.source.scope.program)?
            .scope(&wire.source.scope.definition)
            .ok_or(LowerError::MissingWire { wire: wire.source.wire })?
            .node(wire.source.wire.node)
            .ok_or(LowerError::MissingNode { node: wire.source.wire.node })?;
        Ok(
            matches!(node.kind(), NodeKind::UniformResidueSample { .. } | NodeKind::TrapdoorPublic) ||
                matches!(node.kind(), NodeKind::TrapdoorSample { .. }) &&
                    wire.source.wire.port.0 == 0,
        )
    }

    fn non_relation_sampler_for_wire(
        &mut self,
        wire: &LoweringWire,
        environment: &LowerEnv,
    ) -> Result<Option<u32>, LowerError> {
        let kind = self
            .graph_for_program(&wire.source.scope.program)?
            .scope(&wire.source.scope.definition)
            .ok_or(LowerError::MissingWire { wire: wire.source.wire })?
            .node(wire.source.wire.node)
            .ok_or(LowerError::MissingNode { node: wire.source.wire.node })?
            .kind()
            .clone();
        let source = super::identity::GraphWireSourceKey {
            wire: wire.source.clone(),
            coordinate_binders: environment
                .active_coordinates
                .iter()
                .map(|coordinate| coordinate.binder.clone())
                .collect(),
        };
        let indices = environment
            .active_coordinates
            .iter()
            .map(|coordinate| self.canonical_scalar_identity(coordinate.index.scalar))
            .collect::<Result<Box<[_]>, _>>()?;
        let sampler = match kind {
            NodeKind::GaussianSample { max_coefficient_bound, .. } => {
                let max_coefficient_bound =
                    self.resolve_int(&max_coefficient_bound, environment)?;
                if let Some(cutoff) = resolved_integer(&max_coefficient_bound) &&
                    cutoff.is_negative()
                {
                    return Err(LowerError::NegativeSamplerCutoff { cutoff });
                }
                SamplerIdentity::Gaussian { source, indices, max_coefficient_bound }
            }
            NodeKind::UniformIntervalSample { range, .. } => {
                let minimum = self.resolve_int(&range.minimum, environment)?;
                let maximum = self.resolve_int(&range.maximum, environment)?;
                if let (Some(minimum_value), Some(maximum_value)) =
                    (resolved_integer(&minimum), resolved_integer(&maximum)) &&
                    minimum_value > maximum_value
                {
                    return Err(LowerError::InvalidUniformInterval {
                        minimum: minimum_value,
                        maximum: maximum_value,
                    });
                }
                SamplerIdentity::UniformInterval { source, indices, minimum, maximum }
            }
            _ => return Ok(None),
        };
        Ok(Some(self.symbols.samplers.intern(sampler)))
    }

    /// Normalizes a non-artifact root-stage input to its closed protocol input identity.  The
    /// graph's local input name is never used as an analysis identity.
    fn protocol_input_source(
        &mut self,
        wire: &LoweringWire,
        environment: &LowerEnv,
        sort: &ScalarSort,
    ) -> Result<
        Option<(
            super::identity::AtomicSourceKey,
            Option<super::identity::IntegerSourceDomain>,
            Option<CanonicalResidueConvention>,
        )>,
        LowerError,
    > {
        let super::identity::ProgramKey::WorkflowStage(stage) = &wire.source.scope.program else {
            return Ok(None);
        };
        let stage_decl = self
            .protocol
            .stages()
            .iter()
            .find(|candidate| &candidate.id == stage)
            .ok_or(LowerError::MissingWire { wire: wire.source.wire })?;
        let scope = stage_decl
            .graph
            .scope(&wire.source.scope.definition)
            .ok_or(LowerError::MissingWire { wire: wire.source.wire })?;
        let node = scope
            .node(wire.source.wire.node)
            .ok_or(LowerError::MissingNode { node: wire.source.wire.node })?;
        let mxx_ir_core::node::NodeKind::Input { name, artifact: None, .. } = node.kind() else {
            return Ok(None);
        };
        let destination = ProtocolInputDestination::WorkflowStage {
            stage: stage.clone(),
            input: StageInputName(name.clone()),
        };
        let input = self
            .protocol
            .bundle
            .input_bindings
            .iter()
            .find(|binding| binding.destinations.contains(&destination))
            .map(|binding| binding.input.clone())
            .ok_or_else(|| LowerError::MissingProtocolInputBinding {
                input: crate::ProtocolInputId(name.clone()),
            })?;
        let contract = self
            .protocol
            .bundle
            .input_contract
            .inputs
            .iter()
            .find(|entry| entry.id == input)
            .ok_or_else(|| LowerError::MissingProtocolInputBinding { input: input.clone() })?;
        let integer_range = match &contract.value {
            InputValueContract::IntegerRange { lower, upper } => Some((lower, upper)),
            InputValueContract::Family { element, .. } => match element.as_ref() {
                InputValueContract::IntegerRange { lower, upper } => Some((lower, upper)),
                _ => None,
            },
            _ => None,
        };
        let integer_domain = match (integer_range, sort) {
            (Some((lower, upper)), ScalarSort::Int) => {
                let lower = self.resolve_int(lower, environment)?;
                let upper = self.resolve_int(upper, environment)?;
                let (Some(minimum), Some(maximum)) =
                    (resolved_integer(&lower), resolved_integer(&upper))
                else {
                    return Err(LowerError::UnboundParameter { parameter: contract.name.clone() });
                };
                Some(super::identity::IntegerSourceDomain { minimum, maximum })
            }
            _ => None,
        };
        let canonical_upper_expression = match &contract.value {
            InputValueContract::MatrixExact {
                canonical_coefficient_exclusive_upper_bound, ..
            } => Some(canonical_coefficient_exclusive_upper_bound.as_ref()),
            InputValueContract::Family { element, .. } => match element.as_ref() {
                InputValueContract::MatrixExact {
                    canonical_coefficient_exclusive_upper_bound,
                    ..
                } => Some(canonical_coefficient_exclusive_upper_bound.as_ref()),
                _ => None,
            },
            _ => None,
        };
        let canonical_residue_convention = match (canonical_upper_expression, sort) {
            (Some(upper), ScalarSort::Matrix(matrix)) => {
                let modulus = resolved_nonnegative(&matrix.modulus).ok_or_else(|| {
                    LowerError::InvalidExtractCoefficientCanonicalUpper {
                        upper: num_bigint::BigUint::from(0_u8),
                        modulus: num_bigint::BigUint::from(0_u8),
                    }
                })?;
                let upper = match upper {
                    Some(upper) => match resolved_integer(&self.resolve_int(upper, environment)?) {
                        Some(upper) => upper.to_biguint().ok_or_else(|| {
                            LowerError::InvalidExtractCoefficientCanonicalUpper {
                                upper: num_bigint::BigUint::from(0_u8),
                                modulus: modulus.clone(),
                            }
                        })?,
                        None => {
                            return Err(LowerError::UnboundParameter {
                                parameter: contract.name.clone(),
                            })
                        }
                    },
                    None => modulus.clone(),
                };
                if upper.is_zero() || upper > modulus {
                    return Err(LowerError::InvalidExtractCoefficientCanonicalUpper {
                        upper,
                        modulus,
                    });
                }
                Some(CanonicalResidueConvention::Nonnegative)
            }
            _ => None,
        };
        Ok(Some((
            super::identity::AtomicSourceKey::ProtocolInput(input),
            integer_domain,
            canonical_residue_convention,
        )))
    }

    fn protocol_input_trapdoor(
        &mut self,
        wire: &LoweringWire,
        environment: &LowerEnv,
        output_type: &WireType,
    ) -> Result<Option<TrapdoorDescriptorId>, LowerError> {
        let super::identity::ProgramKey::WorkflowStage(stage) = &wire.source.scope.program else {
            return Ok(None);
        };
        let graph = self.graph_for_program(&wire.source.scope.program)?;
        let scope = graph
            .scope(&wire.source.scope.definition)
            .ok_or(LowerError::MissingWire { wire: wire.source.wire })?;
        let NodeKind::Input { name, artifact: None, .. } = scope
            .node(wire.source.wire.node)
            .ok_or(LowerError::MissingNode { node: wire.source.wire.node })?
            .kind()
        else {
            return Ok(None)
        };
        let destination = ProtocolInputDestination::WorkflowStage {
            stage: stage.clone(),
            input: StageInputName(name.clone()),
        };
        let input = self
            .protocol
            .bundle
            .input_bindings
            .iter()
            .find(|binding| binding.destinations.contains(&destination))
            .map(|binding| binding.input.clone())
            .ok_or_else(|| LowerError::MissingProtocolInputBinding {
                input: crate::ProtocolInputId(name.clone()),
            })?;
        let contract = self
            .protocol
            .bundle
            .input_contract
            .inputs
            .iter()
            .find(|entry| entry.id == input)
            .ok_or_else(|| LowerError::MissingProtocolInputBinding { input: input.clone() })?;
        let InputValueContract::Trapdoor {
            matrix_type,
            sigma,
            gadget_base,
            digit_count,
            preimage_max_coefficient_bound,
            public_input,
        } = &contract.value
        else {
            return Ok(None)
        };
        let public_contract = self
            .protocol
            .bundle
            .input_contract
            .inputs
            .iter()
            .find(|entry| entry.id == *public_input)
            .ok_or_else(|| LowerError::MissingProtocolInputBinding {
                input: public_input.clone(),
            })?;
        let InputValueContract::MatrixExact { matrix_type: public_matrix, .. } =
            &public_contract.value
        else {
            return Err(LowerError::MissingProtocolInputBinding { input: public_input.clone() });
        };
        if matrix_type != public_matrix {
            return Err(LowerError::InvalidOperandSort {
                expected: output_type.clone(),
                actual: output_type.clone(),
            });
        }
        let matrix_type = self.resolve_matrix_type(matrix_type, environment)?;
        let descriptor = TrapdoorIdentity {
            source: TrapdoorSourceKey::ProtocolInput(input),
            indices: environment
                .active_coordinates
                .iter()
                .map(|coordinate| self.canonical_scalar_identity(coordinate.index.scalar))
                .collect::<Result<Box<[_]>, _>>()?,
            matrix_type,
            public: CanonicalTermIdentity::Source(super::identity::GraphWireSourceKey {
                wire: WireSourceKey {
                    scope: wire.source.scope.clone(),
                    wire: WireRef { node: wire.source.wire.node, port: mxx_ir_core::Port(0) },
                },
                coordinate_binders: environment
                    .active_coordinates
                    .iter()
                    .map(|coordinate| coordinate.binder.clone())
                    .collect(),
            }),
            sigma_bits: self.resolve_real(sigma, environment)?.to_bits(),
            gadget_base: self.resolve_int(gadget_base, environment)?,
            digit_count: self.resolve_int(digit_count, environment)?,
            preimage_cutoff: self.resolve_int(preimage_max_coefficient_bound, environment)?,
        };
        Ok(Some(TrapdoorDescriptorId(self.symbols.trapdoors.intern(descriptor))))
    }

    /// Lowers a graph attribute expression without recursive descent.  Attribute division is
    /// the exact, compile-time operator; runtime integer division is handled by `lower_node`.
    pub fn lower_int_expr(
        &mut self,
        expression: &IntExpr,
        environment: &LowerEnv,
    ) -> Result<LoweredInt, LowerError> {
        enum Frame<'b> {
            Enter(&'b IntExpr),
            Finish(&'b IntExpr),
        }
        let mut work = vec![Frame::Enter(expression)];
        let mut values = Vec::<LoweredInt>::new();
        while let Some(frame) = work.pop() {
            match frame {
                Frame::Enter(
                    value @ (IntExpr::Const(_) | IntExpr::Var(_) | IntExpr::LoopIndex(_)),
                ) => {
                    let lowered = match value {
                        IntExpr::Const(value) => self.add_int(value.clone()),
                        IntExpr::Var(name) => {
                            let value = environment.parameters.get(name).ok_or_else(|| {
                                LowerError::UnboundParameter { parameter: name.clone() }
                            })?;
                            self.add_resolved_int(value.clone())?
                        }
                        IntExpr::LoopIndex(slot) => {
                            let (binder, value) = environment
                                .binders
                                .iter()
                                .rev()
                                .find(|(binder, _)| binder.slot == *slot)
                                .ok_or_else(|| LowerError::UnboundBinder {
                                    binder: BinderKey {
                                        loop_scope: environment.occurrence.clone(),
                                        loop_node: mxx_ir_core::NodeId(0),
                                        slot: *slot,
                                    },
                                })?;
                            self.symbols.binders.intern(super::identity::BinderDescriptor {
                                key: binder.clone(),
                                minimum: self
                                    .integer_analysis(value.scalar)
                                    .and_then(|(domain, _)| domain.interval().ok())
                                    .map_or_else(BigInt::zero, |range| range.minimum),
                                maximum: self
                                    .integer_analysis(value.scalar)
                                    .and_then(|(domain, _)| domain.interval().ok())
                                    .map_or_else(BigInt::zero, |range| range.maximum),
                            });
                            let scalar = self
                                .scalar_store
                                .intern_node(ScalarNode::IntBinder(binder.clone()), &self.symbols)
                                .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
                            LoweredInt { scalar }
                        }
                        _ => unreachable!(),
                    };
                    values.push(lowered);
                }
                Frame::Enter(value) => {
                    work.push(Frame::Finish(value));
                    match value {
                        IntExpr::Add(left, right) |
                        IntExpr::Sub(left, right) |
                        IntExpr::Mul(left, right) |
                        IntExpr::Div(left, right) |
                        IntExpr::RoundDiv(left, right) => {
                            work.push(Frame::Enter(right));
                            work.push(Frame::Enter(left));
                        }
                        IntExpr::Log2Ceil(value) => work.push(Frame::Enter(value)),
                        IntExpr::Const(_) | IntExpr::Var(_) | IntExpr::LoopIndex(_) => {
                            unreachable!()
                        }
                    }
                }
                Frame::Finish(value) => {
                    let arity = if matches!(value, IntExpr::Log2Ceil(_)) { 1 } else { 2 };
                    if values.len() < arity {
                        return Err(LowerError::IntervalOperationNotSupported {
                            expression: value.clone(),
                        });
                    }
                    let start = values.len() - arity;
                    let children = values.split_off(start);
                    let result = match value {
                        IntExpr::Add(_, _) => self.combine_int(children, ScalarOperation::Add)?,
                        IntExpr::Sub(_, _) => self.combine_int(children, ScalarOperation::Sub)?,
                        IntExpr::Mul(_, _) => self.combine_int(children, ScalarOperation::Mul)?,
                        IntExpr::Div(_, _) => {
                            self.combine_int(children, ScalarOperation::ExactDiv)?
                        }
                        IntExpr::RoundDiv(_, _) => {
                            self.combine_int(children, ScalarOperation::RoundDiv)?
                        }
                        IntExpr::Log2Ceil(_) => {
                            let child = children.into_iter().next().expect("one child");
                            let scalar = self
                                .scalar_store
                                .intern_node(ScalarNode::IntLog2Ceil([child.scalar]), &self.symbols)
                                .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
                            LoweredInt { scalar }
                        }
                        _ => unreachable!(),
                    };
                    values.push(result);
                }
            }
        }
        values.pop().ok_or_else(|| LowerError::IntervalOperationNotSupported {
            expression: expression.clone(),
        })
    }

    fn add_int(&mut self, value: BigInt) -> LoweredInt {
        let scalar = self
            .scalar_store
            .intern_node(ScalarNode::IntConst(value), &self.symbols)
            .expect("integer constants have a valid scalar transfer");
        LoweredInt { scalar }
    }

    fn add_resolved_int(&mut self, value: ResolvedIntExpr) -> Result<LoweredInt, LowerError> {
        match value {
            ResolvedIntExpr::Const(value) => Ok(self.add_int(value.clone())),
            arena @ ResolvedIntExpr::Arena(_) => resolved_constant(&arena)
                .map(|value| self.add_int(value))
                .ok_or(LowerError::UnsupportedMatrixProductExpansion),
            ResolvedIntExpr::Parameter(name) => {
                let value = self
                    .request
                    .environment
                    .iter()
                    .find_map(|(key, value)| (key == &name).then_some(value))
                    .and_then(|value| match value {
                        super::OperationalParameterValue::Integer(value) => Some(value.clone()),
                        _ => None,
                    });
                if let Some(value) = value {
                    Ok(self.add_int(value.clone()))
                } else {
                    let scalar = self
                        .scalar_store
                        .intern_node(ScalarNode::IntParameter(name), &self.symbols)
                        .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
                    Ok(LoweredInt { scalar })
                }
            }
            ResolvedIntExpr::Binder(_) |
            ResolvedIntExpr::Source { .. } |
            ResolvedIntExpr::Add(_, _) |
            ResolvedIntExpr::Sub(_, _) |
            ResolvedIntExpr::Mul(_, _) |
            ResolvedIntExpr::Div(_, _) |
            ResolvedIntExpr::EuclideanDiv(_, _) |
            ResolvedIntExpr::EuclideanRemainder(_, _) |
            ResolvedIntExpr::RoundDiv(_, _) |
            ResolvedIntExpr::Log2Ceil(_) |
            ResolvedIntExpr::ExtractMatrixCoefficient { .. } => {
                Err(LowerError::UnsupportedMatrixProductExpansion)
            }
        }
    }

    fn combine_int(
        &mut self,
        children: Vec<LoweredInt>,
        operation: ScalarOperation,
    ) -> Result<LoweredInt, LowerError> {
        let [left, right]: [LoweredInt; 2] =
            children.try_into().map_err(|values: Vec<LoweredInt>| {
                LowerError::InvalidOperandArity { expected: 2, actual: values.len() }
            })?;
        let scalar = self
            .scalar_store
            .intern_node(
                match operation {
                    ScalarOperation::Add => ScalarNode::IntAdd([left.scalar, right.scalar]),
                    ScalarOperation::Sub => ScalarNode::IntSub([left.scalar, right.scalar]),
                    ScalarOperation::Mul => ScalarNode::IntMul([left.scalar, right.scalar]),
                    ScalarOperation::ExactDiv => {
                        ScalarNode::IntExactDiv([left.scalar, right.scalar])
                    }
                    ScalarOperation::EuclideanDiv => {
                        ScalarNode::IntEuclideanDiv([left.scalar, right.scalar])
                    }
                    ScalarOperation::EuclideanRemainder => {
                        ScalarNode::IntEuclideanRemainder([left.scalar, right.scalar])
                    }
                    ScalarOperation::RoundDiv => {
                        ScalarNode::IntRoundDiv([left.scalar, right.scalar])
                    }
                    _ => return Err(LowerError::UnsupportedMatrixProductExpansion),
                },
                &self.symbols,
            )
            .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
        Ok(LoweredInt { scalar })
    }

    fn scalar_term(&self, scalar: ScalarId) -> Result<ScalarId, LowerError> {
        let facts =
            self.scalar_store.facts(scalar).ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
        matches!(
            facts.sort.as_ref().ok(),
            Some(ScalarSort::Int | ScalarSort::Bool | ScalarSort::Real)
        )
        .then_some(scalar)
        .ok_or(LowerError::UnsupportedMatrixProductExpansion)
    }

    pub(crate) fn scalar_store_len(&self) -> usize {
        self.scalar_store.len()
    }

    fn register_scalar_node_identity(
        &mut self,
        kind: &NodeKind,
        arguments: &[LoweredValue],
        term: ScalarId,
    ) -> Result<(), LowerError> {
        let _ = (kind, arguments, term);
        Ok(())
    }

    /// Constructs one closed ordinary expression after its graph arguments have already been
    /// lowered by the iterative driver.  Structural nodes are intentionally delegated to the
    /// `FamilyResolver`; decoder nodes do not enter a residual expression.
    pub fn lower_node(
        &mut self,
        kind: &NodeKind,
        arguments: &[LoweredValue],
        environment: &LowerEnv,
    ) -> Result<LoweredValue, LowerError> {
        if node_dispatch(kind) != NodeDispatch::Ordinary {
            // Source and structural lowering need the producer occurrence, output port, and
            // (for families) requested element index.  They are routed by the iterative driver,
            // never approximated as an ordinary value here.
            return Err(LowerError::MissingWire {
                wire: WireRef { node: mxx_ir_core::NodeId(0), port: mxx_ir_core::Port(0) },
            });
        }
        // Matrix values are never converted back to scalar syntax.  This is the
        // first Stage5 cutover: the core ring operations are DAG nodes, while
        // every other matrix operation remains explicitly fail-closed.
        let matrix_operands = || {
            arguments
                .iter()
                .map(|value| match value {
                    LoweredValue::Matrix(term) => Ok(*term),
                    _ => Err(LowerError::UnsupportedMatrixProductExpansion),
                })
                .collect::<Result<Vec<_>, _>>()
        };
        match kind {
            NodeKind::ExtractCoefficient { position, canonical_input_exclusive_upper }
                if arguments.iter().any(|value| matches!(value, LoweredValue::Matrix(_))) =>
            {
                let (matrix, position) = match arguments {
                    [LoweredValue::Matrix(matrix)] => {
                        (*matrix, self.lower_int_expr(position, environment)?)
                    }
                    [LoweredValue::Matrix(matrix), LoweredValue::Scalar(position)] => {
                        let position = self.scalar_term(*position)?;
                        (*matrix, LoweredInt { scalar: position })
                    }
                    _ => {
                        return Err(LowerError::InvalidOperandArity {
                            expected: 1,
                            actual: arguments.len(),
                        })
                    }
                };
                return self.lower_extract_coefficient_dag(
                    matrix,
                    position,
                    canonical_input_exclusive_upper.as_ref(),
                    environment,
                );
            }
            NodeKind::ConstantMatrix { value: mxx_ir_core::node::ConstantMatrix::Zero, .. } => {
                if !arguments.is_empty() {
                    return Err(LowerError::InvalidOperandArity {
                        expected: 0,
                        actual: arguments.len(),
                    });
                }
                return self
                    .dag
                    .push(ExpressionNode::Zero)
                    .map(LoweredValue::Matrix)
                    .map_err(|_| LowerError::UnsupportedMatrixProductExpansion);
            }
            NodeKind::Slice { rows, columns }
                if arguments.iter().any(|value| matches!(value, LoweredValue::Matrix(_))) =>
            {
                let [LoweredValue::Matrix(input)] = arguments else {
                    return Err(LowerError::InvalidOperandArity {
                        expected: 1,
                        actual: arguments.len(),
                    });
                };
                let spec = super::identity::SliceSpec {
                    rows: rows
                        .as_ref()
                        .map(|range| self.resolve_range(range, environment))
                        .transpose()?,
                    columns: columns
                        .as_ref()
                        .map(|range| self.resolve_range(range, environment))
                        .transpose()?,
                };
                if spec.rows.is_none() && spec.columns.is_none() {
                    return Ok(LoweredValue::Matrix(*input));
                }
                let input_type = self.dag_matrix_type(*input)?;
                let output_type = apply_slice_type(input_type, &spec)?;
                return self
                    .dag
                    .push(ExpressionNode::View {
                        input: *input,
                        view: ViewSpec::CoefficientPreserving {
                            view: CoefficientPreservingView::Slice(spec),
                        },
                        output_type,
                    })
                    .map(LoweredValue::Matrix)
                    .map_err(|_| LowerError::UnsupportedMatrixProductExpansion);
            }
            NodeKind::Transpose
                if arguments.iter().any(|value| matches!(value, LoweredValue::Matrix(_))) =>
            {
                let [LoweredValue::Matrix(input)] = arguments else {
                    return Err(LowerError::InvalidOperandArity {
                        expected: 1,
                        actual: arguments.len(),
                    });
                };
                return self
                    .dag
                    .push(ExpressionNode::Transpose(*input))
                    .map(LoweredValue::Matrix)
                    .map_err(|_| LowerError::UnsupportedMatrixProductExpansion);
            }
            NodeKind::Tensor
                if arguments.iter().any(|value| matches!(value, LoweredValue::Matrix(_))) =>
            {
                let [LoweredValue::Matrix(left), LoweredValue::Matrix(right)] = arguments else {
                    return Err(LowerError::InvalidOperandArity {
                        expected: 2,
                        actual: arguments.len(),
                    });
                };
                return self
                    .dag
                    .push(ExpressionNode::Tensor { left: *left, right: *right })
                    .map(LoweredValue::Matrix)
                    .map_err(|_| LowerError::UnsupportedMatrixProductExpansion);
            }
            NodeKind::Concat { axis }
                if arguments.iter().any(|value| matches!(value, LoweredValue::Matrix(_))) =>
            {
                let inputs = matrix_operands()?;
                let axis = match axis {
                    ConcatAxis::Rows => super::identity::Axis::Rows,
                    ConcatAxis::Columns => super::identity::Axis::Columns,
                    ConcatAxis::Diagonal => super::identity::Axis::Diagonal,
                };
                let output_type = concat_matrix_type(&self.dag, &inputs, axis)?;
                return self
                    .dag
                    .push(ExpressionNode::Concat {
                        inputs: inputs.into_boxed_slice(),
                        axis,
                        output_type,
                    })
                    .map(LoweredValue::Matrix)
                    .map_err(|_| LowerError::UnsupportedMatrixProductExpansion);
            }
            NodeKind::CrtRecompose { plaintext_moduli, reconstruction_coefficients }
                if arguments.iter().any(|value| matches!(value, LoweredValue::Matrix(_))) =>
            {
                let inputs = matrix_operands()?;
                let first = inputs
                    .first()
                    .copied()
                    .ok_or(LowerError::InvalidOperandArity { expected: 1, actual: 0 })?;
                let output_type = self.dag_matrix_type(first)?;
                let spec = super::identity::CrtSpec {
                    plaintext_moduli: plaintext_moduli
                        .iter()
                        .map(|value| self.resolve_int(value, environment))
                        .collect::<Result<Vec<_>, _>>()?
                        .into_boxed_slice(),
                    reconstruction_coefficients: reconstruction_coefficients
                        .iter()
                        .map(|value| self.resolve_int(value, environment))
                        .collect::<Result<Vec<_>, _>>()?
                        .into_boxed_slice(),
                };
                return self
                    .dag
                    .push(ExpressionNode::CrtRecompose {
                        inputs: inputs.into_boxed_slice(),
                        spec,
                        output_type,
                    })
                    .map(LoweredValue::Matrix)
                    .map_err(|_| LowerError::UnsupportedMatrixProductExpansion);
            }
            NodeKind::MatrixBinary(operation)
                if arguments.iter().any(|value| matches!(value, LoweredValue::Matrix(_))) =>
            {
                let values = matrix_operands()?;
                if values.len() != 2 {
                    return Err(LowerError::InvalidOperandArity {
                        expected: 2,
                        actual: values.len(),
                    });
                }
                let node = match operation {
                    MatrixBinaryOp::Add => ExpressionNode::Add(values.into_boxed_slice()),
                    MatrixBinaryOp::Subtract => {
                        let negated = self
                            .dag
                            .push(ExpressionNode::Negate(values[1]))
                            .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
                        ExpressionNode::Add(vec![values[0], negated].into_boxed_slice())
                    }
                    MatrixBinaryOp::Multiply => ExpressionNode::Product(values.into_boxed_slice()),
                };
                return self
                    .dag
                    .push(node)
                    .map(LoweredValue::Matrix)
                    .map_err(|_| LowerError::UnsupportedMatrixProductExpansion);
            }
            NodeKind::MatrixNegate
                if arguments.iter().any(|value| matches!(value, LoweredValue::Matrix(_))) =>
            {
                let values = matrix_operands()?;
                let [value] = values.as_slice() else {
                    return Err(LowerError::InvalidOperandArity {
                        expected: 1,
                        actual: arguments.len(),
                    });
                };
                return self
                    .dag
                    .push(ExpressionNode::Negate(*value))
                    .map(LoweredValue::Matrix)
                    .map_err(|_| LowerError::UnsupportedMatrixProductExpansion);
            }
            NodeKind::MatrixScale { scalar }
                if arguments.iter().any(|value| {
                    matches!(value, LoweredValue::Matrix(_) | LoweredValue::MatrixFamily(_))
                }) =>
            {
                let [matrix] = arguments else {
                    return Err(LowerError::InvalidOperandArity {
                        expected: 1,
                        actual: arguments.len(),
                    });
                };
                let LoweredValue::Matrix(matrix) = matrix else {
                    return Err(LowerError::UnsupportedMatrixProductExpansion);
                };
                let scalar = self.lower_int_expr(scalar, environment)?;
                self.validate_integer_consumer(
                    scalar.scalar,
                    SelectorOnlyConsumer::NoiseBoundArithmetic,
                    false,
                )?;
                let identity = self.canonical_scalar_identity(scalar.scalar)?;
                let value = if let Some(value) = resolved_integer(&identity) {
                    value
                } else {
                    let interval = self
                        .integer_analysis(scalar.scalar)
                        .and_then(|(domain, _)| domain.interval().ok())
                        .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
                    let input_type = self.dag_matrix_type(*matrix)?;
                    let scalar_type = mxx_ir_core::types::ConcreteMatrixType {
                        modulus: input_type.modulus.clone(),
                        ring_dimension: input_type.ring_dimension,
                        rows: 1,
                        columns: 1,
                    };
                    let maximum = interval.minimum.magnitude().max(interval.maximum.magnitude());
                    if maximum.is_zero() {
                        return self
                            .dag
                            .push(ExpressionNode::Zero)
                            .map(LoweredValue::Matrix)
                            .map_err(|_| LowerError::UnsupportedMatrixProductExpansion);
                    }
                    let scalar_factor = SymbolicFactor::bounded_with_metadata(
                        FactorIdentity::runtime_scalar(scalar_type.clone(), identity),
                        MatrixBound {
                            matrix_type: scalar_type.clone(),
                            coefficient_class: BoundClass::bounded(maximum.clone()),
                        },
                        MatrixMetadata {
                            canonical_coefficient_exclusive_upper: None,
                            is_constant_polynomial: true,
                            known_zero_rows: None,
                            polynomial: Some(
                                super::bound::PolynomialFacts::new(1, scalar_type.ring_dimension)
                                    .expect("scalar support"),
                            ),
                        },
                    )
                    .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
                    let scalar = self
                        .dag
                        .push(ExpressionNode::Atom(scalar_factor))
                        .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
                    return self
                        .dag
                        .push(ExpressionNode::Product(vec![scalar, *matrix].into_boxed_slice()))
                        .map(LoweredValue::Matrix)
                        .map_err(|_| LowerError::UnsupportedMatrixProductExpansion);
                };
                if value.is_one() {
                    return Ok(LoweredValue::Matrix(*matrix));
                }
                if value.is_zero() {
                    return self
                        .dag
                        .push(ExpressionNode::Zero)
                        .map(LoweredValue::Matrix)
                        .map_err(|_| LowerError::UnsupportedMatrixProductExpansion);
                }
                let input_type = self.dag_matrix_type(*matrix)?;
                let scalar_type = mxx_ir_core::types::ConcreteMatrixType {
                    modulus: input_type.modulus.clone(),
                    ring_dimension: input_type.ring_dimension,
                    rows: 1,
                    columns: 1,
                };
                let value = centered_residue(&value, &scalar_type.modulus);
                if value.is_zero() {
                    return self
                        .dag
                        .push(ExpressionNode::Zero)
                        .map(LoweredValue::Matrix)
                        .map_err(|_| LowerError::UnsupportedMatrixProductExpansion);
                }
                if value == BigInt::from(1_u8) {
                    return Ok(LoweredValue::Matrix(*matrix));
                }
                let scalar_factor = SymbolicFactor::bounded_with_metadata(
                    FactorIdentity::constant_polynomial(&scalar_type, &value),
                    MatrixBound {
                        matrix_type: scalar_type.clone(),
                        coefficient_class: BoundClass::bounded(value.magnitude().clone()),
                    },
                    MatrixMetadata {
                        canonical_coefficient_exclusive_upper: None,
                        is_constant_polynomial: true,
                        known_zero_rows: None,
                        polynomial: Some(
                            super::bound::PolynomialFacts::new(1, scalar_type.ring_dimension)
                                .expect("constant support"),
                        ),
                    },
                )
                .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
                let scalar = self
                    .dag
                    .push(ExpressionNode::Atom(scalar_factor))
                    .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
                return self
                    .dag
                    .push(ExpressionNode::Product(vec![scalar, *matrix].into_boxed_slice()))
                    .map(LoweredValue::Matrix)
                    .map_err(|_| LowerError::UnsupportedMatrixProductExpansion);
            }
            _ if arguments.iter().any(|value| matches!(value, LoweredValue::Matrix(_))) => {
                return Err(LowerError::UnsupportedMatrixProductExpansion);
            }
            _ => {}
        }
        let terms = |expected: usize| -> Result<Vec<ScalarId>, LowerError> {
            if arguments.len() != expected {
                return Err(LowerError::InvalidOperandArity { expected, actual: arguments.len() });
            }
            arguments
                .iter()
                .map(|value| match value {
                    LoweredValue::Scalar(term) => self.scalar_term(*term),
                    LoweredValue::Matrix(_) | LoweredValue::MatrixFamily(_) => {
                        Err(LowerError::UnsupportedMatrixProductExpansion)
                    }
                    LoweredValue::Trapdoor(_) | LoweredValue::TrapdoorFamily { .. } => {
                        Err(LowerError::InvalidOperandSort {
                            expected: WireType::Matrix(MatrixType {
                                modulus: IntExpr::constant(1),
                                ring_dimension: IntExpr::constant(1),
                                rows: IntExpr::constant(1),
                                columns: IntExpr::constant(1),
                            }),
                            actual: WireType::Trapdoor {
                                matrix: MatrixType {
                                    modulus: IntExpr::constant(1),
                                    ring_dimension: IntExpr::constant(1),
                                    rows: IntExpr::constant(1),
                                    columns: IntExpr::constant(1),
                                },
                                sigma: RealExpr::from(0_i32),
                                gadget_base: IntExpr::constant(2),
                                digit_count: IntExpr::constant(1),
                                preimage_max_coefficient_bound: IntExpr::constant(0),
                            },
                        })
                    }
                    LoweredValue::Family(_) => {
                        Err(LowerError::InvalidOperandArity { expected, actual: arguments.len() })
                    }
                })
                .collect()
        };
        let scalar = match kind {
            NodeKind::ConstantInt(value) => self
                .scalar_store
                .intern_node(ScalarNode::IntConst(value.clone()), &self.symbols)
                .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?,
            NodeKind::EvaluateInt(value) => {
                return Ok(LoweredValue::Scalar(self.lower_int_expr(value, environment)?.scalar))
            }
            NodeKind::ConstantBool(value) => self
                .scalar_store
                .intern_node(ScalarNode::BoolConst(*value), &self.symbols)
                .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?,
            NodeKind::ConstantReal(value) => self
                .scalar_store
                .intern_node(
                    ScalarNode::RealConst(self.resolve_real(value, environment)?.to_bits()),
                    &self.symbols,
                )
                .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?,
            NodeKind::IntBinary(operation) => {
                let values = terms(2)?;
                let operation = match operation {
                    IntBinaryOp::Add => ScalarOperation::Add,
                    IntBinaryOp::Subtract => ScalarOperation::Sub,
                    IntBinaryOp::Multiply => ScalarOperation::Mul,
                    IntBinaryOp::Divide => ScalarOperation::EuclideanDiv,
                    IntBinaryOp::Remainder => ScalarOperation::EuclideanRemainder,
                };
                self.scalar_store
                    .intern_node(
                        match operation {
                            ScalarOperation::Add => ScalarNode::IntAdd([values[0], values[1]]),
                            ScalarOperation::Sub => ScalarNode::IntSub([values[0], values[1]]),
                            ScalarOperation::Mul => ScalarNode::IntMul([values[0], values[1]]),
                            ScalarOperation::EuclideanDiv => {
                                ScalarNode::IntEuclideanDiv([values[0], values[1]])
                            }
                            ScalarOperation::EuclideanRemainder => {
                                ScalarNode::IntEuclideanRemainder([values[0], values[1]])
                            }
                            _ => unreachable!(),
                        },
                        &self.symbols,
                    )
                    .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?
            }
            NodeKind::IntCompare(operation) => {
                let values = terms(2)?;
                let node = match operation {
                    IntCompareOp::Equal => ScalarNode::IntEqual([values[0], values[1]]),
                    IntCompareOp::Less => ScalarNode::IntLess([values[0], values[1]]),
                    IntCompareOp::LessEqual => ScalarNode::IntLessEqual([values[0], values[1]]),
                };
                self.scalar_store
                    .intern_node(node, &self.symbols)
                    .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?
            }
            NodeKind::BitExtract { bit } => {
                let input = terms(1)?[0];
                let bit = self.resolve_int(bit, environment)?;
                self.scalar_store
                    .intern_node(ScalarNode::BitExtract { bit, input: [input] }, &self.symbols)
                    .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?
            }
            NodeKind::BoolToInt => {
                let values = terms(1)?;
                self.validate_boolean_consumer(values[0], SelectorOnlyConsumer::BoolToInt, false)?;
                self.scalar_store
                    .intern_node(ScalarNode::BoolToInt([values[0]]), &self.symbols)
                    .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?
            }
            NodeKind::IntToReal => {
                let values = terms(1)?;
                self.validate_integer_consumer(values[0], SelectorOnlyConsumer::IntToReal, false)?;
                self.scalar_store
                    .intern_node(ScalarNode::IntToReal([values[0]]), &self.symbols)
                    .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?
            }
            NodeKind::RealBinary(operation) => {
                let values = terms(2)?;
                let node = match operation {
                    RealBinaryOp::Add => ScalarNode::RealAdd([values[0], values[1]]),
                    RealBinaryOp::Subtract => ScalarNode::RealSub([values[0], values[1]]),
                    RealBinaryOp::Multiply => ScalarNode::RealMul([values[0], values[1]]),
                    RealBinaryOp::Divide => ScalarNode::RealDiv([values[0], values[1]]),
                };
                self.scalar_store
                    .intern_node(node, &self.symbols)
                    .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?
            }
            NodeKind::RealSqrt => self
                .scalar_store
                .intern_node(ScalarNode::RealSqrt([terms(1)?[0]]), &self.symbols)
                .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?,
            NodeKind::ExtractCoefficient { position, canonical_input_exclusive_upper } => {
                let matrix = terms(1)?[0];
                let _position = self.lower_int_expr(position, environment)?;
                if let Some(upper) = canonical_input_exclusive_upper {
                    // Matrix coefficient extraction is handled by the DAG branch
                    // above; a scalar reaching this branch has no matrix modulus.
                    let _ = (upper, matrix);
                }
                return Err(LowerError::UnsupportedMatrixProductExpansion)
            }
            NodeKind::LiftIntegerToConstantPolynomial { .. } | NodeKind::CrtRecompose { .. } => {
                // These operations are matrix-valued and are lowered by the
                // direct normal-form handlers above.  They must never enter
                // the scalar-store fallback.
                return Err(LowerError::UnsupportedMatrixProductExpansion);
            }
            NodeKind::Input { .. } |
            NodeKind::GadgetTrapdoor { .. } |
            NodeKind::TrapdoorPublic |
            NodeKind::UniformResidueSample { .. } |
            NodeKind::UniformIntervalSample { .. } |
            NodeKind::GaussianSample { .. } |
            NodeKind::HashSample { .. } |
            NodeKind::TrapdoorSample { .. } |
            NodeKind::PreimageSample { .. } |
            NodeKind::GadgetDecompose { .. } |
            NodeKind::PackPolynomialCoefficients { .. } |
            NodeKind::SubgraphCall(_) |
            NodeKind::ParallelLoop(_) |
            NodeKind::SequentialLoop(_) |
            NodeKind::FamilyPack { .. } |
            NodeKind::FamilyGetStatic { .. } |
            NodeKind::FamilyGetDynamic |
            NodeKind::Select { .. } |
            NodeKind::ThresholdDecode { .. } => unreachable!("closed dispatch was checked above"),
            // Matrix values are represented by `LoweredValue::Matrix` and never enter this
            // scalar constructor.  Keeping a typed rejection here makes malformed callers
            // fail closed without retaining a scalar matrix fallback.
            NodeKind::ConstantMatrix { .. } |
            NodeKind::MatrixBinary(_) |
            NodeKind::MatrixNegate |
            NodeKind::MatrixScale { .. } |
            NodeKind::Transpose |
            NodeKind::Slice { .. } |
            NodeKind::Tensor |
            NodeKind::Concat { .. } => return Err(LowerError::UnsupportedMatrixProductExpansion),
        };
        self.register_scalar_node_identity(kind, arguments, scalar)?;
        Ok(LoweredValue::Scalar(scalar))
    }

    fn lower_structural_node(
        &mut self,
        wire: &LoweringWire,
        kind: &NodeKind,
        arguments: &[LoweredValue],
        environment: &LowerEnv,
        output_type: WireType,
    ) -> Result<LoweredValue, LowerError> {
        match kind {
            NodeKind::PackPolynomialCoefficients { matrix_type, coefficient_bits }
                if matches!(arguments, [LoweredValue::Family(_)]) =>
            {
                let [LoweredValue::Family(family)] = arguments else { unreachable!() };
                if family.element_type != ScalarSort::Bool {
                    return Err(LowerError::PackRequiresExplicitBooleanFamily {
                        actual: output_type,
                    });
                }
                let FamilyCoverageStorage::ExactStored { elements } = &family.storage else {
                    return Err(LowerError::PackRequiresExplicitBooleanFamily {
                        actual: output_type,
                    });
                };
                let matrix_type = self.resolve_matrix_type(matrix_type, environment)?;
                let concrete = concrete_matrix_type(&matrix_type)
                    .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
                let bits = elements
                    .iter()
                    .enumerate()
                    .map(|(position, term)| {
                        let data = self
                            .scalar_store
                            .facts(*term)
                            .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
                        if data.sort != Ok(ScalarSort::Bool) ||
                            data.scalar_provenance != Some(ScalarProvenance::Ordinary)
                        {
                            return Err(LowerError::PackRequiresExplicitBooleanFamily {
                                actual: WireType::Bool,
                            });
                        }
                        let wire =
                            LoweringWire { source: wire.source.clone(), indices: Box::new([]) };
                        let key = self.graph_factor_identity(
                            &wire,
                            environment,
                            format!("pack-bit:{position}").as_bytes(),
                        )?;
                        Ok(BoolBit {
                            value: super::normal_form::PolynomialNF::exact_factor_typed(
                                key.clone(),
                                concrete.clone(),
                            ),
                            identity: key,
                            maximum: if data.possible_true {
                                BigUint::from(1_u8)
                            } else {
                                BigUint::zero()
                            },
                            position,
                            weight: BigUint::from(1_u8) << position,
                            is_bool: true,
                            known_zero: !data.possible_true,
                        })
                    })
                    .collect::<Result<Vec<_>, LowerError>>()?;
                let coefficient_bits =
                    resolved_integer(&self.resolve_int(coefficient_bits, environment)?)
                        .and_then(|value| value.to_usize())
                        .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
                let ring_dimension = concrete.ring_dimension;
                let key = self.graph_factor_identity(wire, environment, b"pack-polynomial")?;
                let nf = <super::normal_form::PolynomialNF as AdditionalOperations>::pack_polynomial_coefficients_nf(
                    &bits,
                    ring_dimension,
                    coefficient_bits,
                    concrete.clone(),
                )
                .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
                return self.push_nf_term(nf, key, concrete).map(LoweredValue::Matrix);
            }
            NodeKind::FamilyPack { .. } => {
                let WireType::IndexedFamily { element, .. } = output_type else {
                    return Err(LowerError::IncompatibleFamilyCoverage {
                        expected: WireType::IndexedFamily {
                            element: Box::new(WireType::Int),
                            count: IntExpr::constant(0),
                        },
                        actual: output_type,
                    });
                };
                let element_wire_type = *element;
                let element_type =
                    self.resolve_family_element_sort(&element_wire_type, environment)?;
                let argument_sources = self.structural_argument_sources(wire)?;
                if argument_sources.len() != arguments.len() {
                    return Err(LowerError::InvalidOperandArity {
                        expected: argument_sources.len(),
                        actual: arguments.len(),
                    });
                }
                if matches!(element_wire_type, WireType::Matrix(_) | WireType::Preimage(_)) {
                    let elements = arguments
                        .iter()
                        .map(|argument| match argument {
                            LoweredValue::Matrix(term) => Ok(*term),
                            LoweredValue::MatrixFamily(_) => {
                                Err(LowerError::UnsupportedMatrixProductExpansion)
                            }
                            _ => Err(LowerError::FamilyElementLoweringMismatch {
                                expected: element_wire_type.clone(),
                                actual_category: Self::lowered_value_category(argument)
                                    .expect("family value category"),
                                actual_sort: None,
                                producer: wire.source.clone(),
                            }),
                        })
                        .collect::<Result<Box<[_]>, _>>()?;
                    let value = FamilyLoweringValue {
                        element_type,
                        storage: FamilyCoverageStorage::ExactStored { elements },
                    };
                    value.validate().map_err(|_| LowerError::InvalidFamilyCount {
                        count: IntExpr::constant(0),
                    })?;
                    return Ok(LoweredValue::MatrixFamily(value));
                }
                let elements = arguments
                    .iter()
                    .zip(argument_sources)
                    .map(|(argument, producer)| match argument {
                        LoweredValue::Scalar(term)
                            if self.scalar_store.facts(*term).is_some_and(|facts| {
                                facts
                                    .sort
                                    .as_ref()
                                    .is_ok_and(|actual| sorts_equal(&element_type, actual))
                            }) =>
                        {
                            self.scalar_term(*term)
                        }
                        LoweredValue::Scalar(term) => {
                            Err(LowerError::FamilyElementLoweringMismatch {
                                expected: element_wire_type.clone(),
                                actual_category: super::error::LoweredValueCategory::Term,
                                actual_sort: self
                                    .scalar_store
                                    .facts(*term)
                                    .and_then(|facts| facts.sort.as_ref().ok().cloned()),
                                producer,
                            })
                        }
                        LoweredValue::Matrix(_) => {
                            Err(LowerError::UnsupportedMatrixProductExpansion)
                        }
                        LoweredValue::MatrixFamily(_) => {
                            Err(LowerError::UnsupportedMatrixProductExpansion)
                        }
                        LoweredValue::Family(_) |
                        LoweredValue::Trapdoor(_) |
                        LoweredValue::TrapdoorFamily { .. } => {
                            Err(LowerError::FamilyElementLoweringMismatch {
                                expected: element_wire_type.clone(),
                                actual_category: Self::lowered_value_category(argument)
                                    .expect("non-term family member"),
                                actual_sort: None,
                                producer,
                            })
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let value = FamilyLoweringValue {
                    element_type,
                    storage: FamilyCoverageStorage::ExactStored { elements: elements.into() },
                };
                value
                    .validate()
                    .map_err(|_| LowerError::InvalidFamilyCount { count: IntExpr::constant(0) })?;
                Ok(LoweredValue::Family(value))
            }
            NodeKind::FamilyGetStatic { index } => {
                if let [LoweredValue::TrapdoorFamily { representative, binder, logical_count }] =
                    arguments
                {
                    let index = self.lower_int_expr(index, environment)?;
                    return self.trapdoor_family_element(
                        *representative,
                        binder,
                        logical_count,
                        &index,
                    );
                }
                if let [LoweredValue::MatrixFamily(family)] = arguments {
                    let index = self.lower_int_expr(index, environment)?;
                    let stable = self.canonical_scalar_identity(index.scalar)?;
                    {
                        if let Some(element) =
                            normal_form_family::static_matrix_term(family, &stable).map_err(
                                |_| LowerError::FamilyAccessOutOfRange {
                                    index: IntExpr::constant(-1),
                                    count: IntExpr::constant(0),
                                },
                            )?
                        {
                            return Ok(LoweredValue::Matrix(element));
                        }
                    }
                    return self.shared_matrix_family_element(family, &index);
                }
                let [LoweredValue::Family(family)] = arguments else {
                    return Err(LowerError::InvalidOperandArity {
                        expected: 1,
                        actual: arguments.len(),
                    });
                };
                let index = self.lower_int_expr(index, environment)?;
                let index_identity = self.canonical_scalar_identity(index.scalar)?;
                if let Some(element) =
                    family::static_get(family, &index_identity).map_err(|_| {
                        LowerError::FamilyAccessOutOfRange {
                            index: self
                                .canonical_scalar_identity(index.scalar)
                                .ok()
                                .and_then(|value| resolved_integer(&value).map(IntExpr::constant))
                                .unwrap_or_else(|| IntExpr::constant(-1)),
                            count: IntExpr::constant(0),
                        }
                    })?
                {
                    return Ok(LoweredValue::Scalar(self.scalar_term(element)?));
                }
                self.shared_family_element(family, &index)
            }
            NodeKind::FamilyGetDynamic => {
                if let [LoweredValue::MatrixFamily(family), LoweredValue::Scalar(selector)] =
                    arguments
                {
                    let scalar = self.scalar_term(*selector)?;
                    let selector = LoweredInt { scalar };
                    return self.matrix_family_element(family, &selector, wire, environment);
                }
                if let [
                    LoweredValue::TrapdoorFamily { representative, binder, logical_count },
                    LoweredValue::Scalar(selector),
                ] = arguments
                {
                    let scalar = self.scalar_term(*selector)?;
                    let selector = LoweredInt { scalar };
                    return self.trapdoor_family_element(
                        *representative,
                        binder,
                        logical_count,
                        &selector,
                    );
                }
                let [LoweredValue::Family(family), LoweredValue::Scalar(selector)] = arguments
                else {
                    return Err(LowerError::InvalidOperandArity {
                        expected: 2,
                        actual: arguments.len(),
                    });
                };
                let scalar = self.scalar_term(*selector)?;
                let selector = LoweredInt { scalar };
                match &family.storage {
                    FamilyCoverageStorage::ExactStored { elements } => {
                        let term =
                            self.scalar_dynamic_get(family, selector.scalar).map_err(|_| {
                                LowerError::InvalidFamilyCount {
                                    count: IntExpr::constant(elements.len()),
                                }
                            })?;
                        Ok(LoweredValue::Scalar(term))
                    }
                    FamilyCoverageStorage::SharedTemplate { .. } => {
                        self.shared_family_element(family, &selector)
                    }
                }
            }
            NodeKind::Select { .. } => {
                let Some((LoweredValue::Scalar(selector), cases)) = arguments.split_first() else {
                    return Err(LowerError::InvalidOperandArity {
                        expected: 2,
                        actual: arguments.len(),
                    });
                };
                let selector = self.scalar_term(*selector)?;
                if cases.iter().all(|value| matches!(value, LoweredValue::Matrix(_))) {
                    let reachable = self.selector_reachable(selector, cases.len())?;
                    let selector_identity = self.scalar_selector_identity(selector)?;
                    let matrix_cases = cases
                        .iter()
                        .map(|value| match value {
                            LoweredValue::Matrix(term) => Ok(*term),
                            _ => unreachable!("matrix case check above"),
                        })
                        .collect::<Result<Box<_>, LowerError>>()?;
                    let term = self
                        .dag
                        .push(ExpressionNode::Select {
                            selector: selector_identity,
                            cases: matrix_cases,
                            reachable,
                        })
                        .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
                    return Ok(LoweredValue::Matrix(term));
                }
                if cases.iter().all(|value| matches!(value, LoweredValue::MatrixFamily(_))) {
                    let matrix_families = cases
                        .iter()
                        .map(|value| match value {
                            LoweredValue::MatrixFamily(family) => Ok(family),
                            _ => unreachable!("matrix-family case check above"),
                        })
                        .collect::<Result<Vec<_>, LowerError>>()?;
                    let first = matrix_families.first().ok_or(LowerError::InvalidOperandArity {
                        expected: 2,
                        actual: arguments.len(),
                    })?;
                    normal_form_family::validate_matrix_term_family(first).map_err(|_| {
                        LowerError::IncompatibleFamilyCoverage {
                            expected: output_type.clone(),
                            actual: output_type.clone(),
                        }
                    })?;
                    for family in &matrix_families[1..] {
                        normal_form_family::validate_matrix_term_family(family).map_err(|_| {
                            LowerError::IncompatibleFamilyCoverage {
                                expected: output_type.clone(),
                                actual: output_type.clone(),
                            }
                        })?;
                        if family.element_type != first.element_type {
                            return Err(LowerError::IncompatibleFamilyCoverage {
                                expected: output_type.clone(),
                                actual: output_type.clone(),
                            });
                        }
                    }
                    let selector_value = LoweredInt { scalar: selector };
                    let reachable = self.selector_reachable(selector, matrix_families.len())?;
                    let selector =
                        self.family_selector_identity(first, &selector_value, wire, environment)?;
                    let storage = match &first.storage {
                        FamilyCoverageStorage::ExactStored { elements } => {
                            let width = elements.len();
                            let mut selected = Vec::with_capacity(width);
                            for lane in 0..width {
                                let mut lane_cases = Vec::with_capacity(matrix_families.len());
                                for family in &matrix_families {
                                    let FamilyCoverageStorage::ExactStored { elements } =
                                        &family.storage
                                    else {
                                        return Err(LowerError::IncompatibleFamilyCoverage {
                                            expected: output_type.clone(),
                                            actual: output_type.clone(),
                                        });
                                    };
                                    if elements.len() != width {
                                        return Err(LowerError::IncompatibleFamilyCoverage {
                                            expected: output_type.clone(),
                                            actual: output_type.clone(),
                                        });
                                    }
                                    lane_cases.push(elements[lane]);
                                }
                                selected.push(
                                    self.dag
                                        .push(ExpressionNode::Select {
                                            selector: selector.clone(),
                                            cases: lane_cases.into_boxed_slice(),
                                            reachable: reachable.clone(),
                                        })
                                        .map_err(|_| {
                                            LowerError::UnsupportedMatrixProductExpansion
                                        })?,
                                );
                            }
                            FamilyCoverageStorage::ExactStored {
                                elements: selected.into_boxed_slice(),
                            }
                        }
                        FamilyCoverageStorage::SharedTemplate {
                            domain, binder_domains, ..
                        } => {
                            let mut selected = Vec::with_capacity(matrix_families.len());
                            for family in &matrix_families {
                                let FamilyCoverageStorage::SharedTemplate {
                                    domain: candidate_domain,
                                    binder_domains: candidate_binders,
                                    representative,
                                } = &family.storage
                                else {
                                    return Err(LowerError::IncompatibleFamilyCoverage {
                                        expected: output_type.clone(),
                                        actual: output_type.clone(),
                                    });
                                };
                                if candidate_domain != domain || candidate_binders != binder_domains
                                {
                                    return Err(LowerError::IncompatibleFamilyCoverage {
                                        expected: output_type.clone(),
                                        actual: output_type.clone(),
                                    });
                                }
                                selected.push(*representative);
                            }
                            let representative = self
                                .dag
                                .push(ExpressionNode::Select {
                                    selector,
                                    cases: selected.into_boxed_slice(),
                                    reachable,
                                })
                                .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
                            FamilyCoverageStorage::SharedTemplate {
                                domain: domain.clone(),
                                representative,
                                binder_domains: binder_domains.clone(),
                            }
                        }
                    };
                    return Ok(LoweredValue::MatrixFamily(FamilyLoweringValue {
                        element_type: first.element_type.clone(),
                        storage,
                    }));
                }
                let mut families = cases.iter().cloned().collect::<Vec<_>>();
                if !matches!(output_type, WireType::IndexedFamily { .. }) {
                    let zero = self.add_int(BigInt::zero());
                    for value in &mut families {
                        if let LoweredValue::Family(family) = value &&
                            matches!(&family.storage, FamilyCoverageStorage::SharedTemplate { domain, .. } if domain.logical_count == num_bigint::BigUint::from(1_u8))
                        {
                            *value = self.family_element(family, &zero)?;
                        }
                    }
                }
                if families.iter().all(|value| matches!(value, LoweredValue::Family(_))) {
                    let families = families
                        .into_iter()
                        .map(|value| match value {
                            LoweredValue::Family(family) => family,
                            _ => unreachable!(),
                        })
                        .collect::<Vec<_>>();
                    let families = self.align_selected_shared_families(families).map_err(|_| {
                        LowerError::IncompatibleFamilyCoverage {
                            expected: output_type.clone(),
                            actual: output_type.clone(),
                        }
                    })?;
                    return self
                        .scalar_select_family(selector, &families)
                        .map(LoweredValue::Family)
                        .map_err(|_| LowerError::IncompatibleFamilyCoverage {
                            expected: output_type.clone(),
                            actual: output_type,
                        });
                }
                let terms = families
                    .iter()
                    .map(|value| match value {
                        LoweredValue::Scalar(term) => self.scalar_term(*term),
                        _ => Err(LowerError::IncompatibleFamilyCoverage {
                            expected: output_type.clone(),
                            actual: output_type.clone(),
                        }),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let children = std::iter::once(selector).chain(terms).collect::<Box<_>>();
                let scalar = self
                    .scalar_store
                    .intern_node(ScalarNode::Switch(children), &self.symbols)
                    .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
                Ok(LoweredValue::Scalar(scalar))
            }
            NodeKind::ParallelLoop(_) | NodeKind::SequentialLoop(_) => {
                unreachable!("loop lowering is scheduled on the outer continuation stack")
            }
            NodeKind::SubgraphCall(_) | NodeKind::ThresholdDecode { .. } => unreachable!(),
            _ => unreachable!("only structural nodes reach structural lowering"),
        }
    }

    fn structural_argument_sources(
        &self,
        wire: &LoweringWire,
    ) -> Result<Vec<WireSourceKey>, LowerError> {
        let scope = self
            .graph_for_program(&wire.source.scope.program)?
            .scope(&wire.source.scope.definition)
            .ok_or(LowerError::MissingWire { wire: wire.source.wire })?;
        let node = scope
            .node(wire.source.wire.node)
            .ok_or(LowerError::MissingNode { node: wire.source.wire.node })?;
        scope.arguments(node).ok_or(LowerError::MissingWire { wire: wire.source.wire }).map(
            |arguments| {
                arguments
                    .into_iter()
                    .map(|argument| WireSourceKey {
                        scope: wire.source.scope.clone(),
                        wire: argument,
                    })
                    .collect()
            },
        )
    }

    fn lowered_value_category(value: &LoweredValue) -> Option<super::error::LoweredValueCategory> {
        match value {
            LoweredValue::Matrix(_) => Some(super::error::LoweredValueCategory::Term),
            LoweredValue::MatrixFamily(_) => Some(super::error::LoweredValueCategory::Family),
            LoweredValue::Scalar(_) => Some(super::error::LoweredValueCategory::Term),
            LoweredValue::Family(_) => Some(super::error::LoweredValueCategory::Family),
            LoweredValue::Trapdoor(_) => Some(super::error::LoweredValueCategory::Trapdoor),
            LoweredValue::TrapdoorFamily { .. } => {
                Some(super::error::LoweredValueCategory::TrapdoorFamily)
            }
        }
    }

    fn scalar_select_family(
        &mut self,
        selector: ScalarId,
        families: &[FamilyLoweringValue<ScalarId>],
    ) -> Result<FamilyLoweringValue<ScalarId>, LowerError> {
        let first =
            families.first().ok_or(LowerError::InvalidOperandArity { expected: 2, actual: 1 })?;
        let selector_facts = self
            .scalar_store
            .facts(selector)
            .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
        let interval = selector_facts
            .integer_domain
            .as_ref()
            .ok_or(LowerError::UnsupportedMatrixProductExpansion)?
            .interval()
            .map_err(|_| LowerError::IncompatibleFamilyCoverage {
                expected: WireType::Int,
                actual: WireType::Int,
            })?;
        if interval.minimum < BigInt::zero() || interval.maximum >= BigInt::from(families.len()) {
            return Err(LowerError::IncompatibleFamilyCoverage {
                expected: WireType::Int,
                actual: WireType::Int,
            });
        }
        for family in families {
            if family.element_type != first.element_type {
                return Err(LowerError::IncompatibleFamilyCoverage {
                    expected: WireType::Int,
                    actual: WireType::Int,
                });
            }
        }
        let make_switch = |this: &mut Self, cases: &[ScalarId]| {
            let children =
                std::iter::once(selector).chain(cases.iter().copied()).collect::<Box<_>>();
            this.scalar_store
                .intern_node(ScalarNode::Switch(children), &this.symbols)
                .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)
        };
        let storage = match &first.storage {
            FamilyCoverageStorage::ExactStored { elements } => {
                let width = elements.len();
                let mut selected = Vec::with_capacity(width);
                for lane in 0..width {
                    let mut cases = Vec::with_capacity(families.len());
                    for family in families {
                        let FamilyCoverageStorage::ExactStored { elements } = &family.storage
                        else {
                            return Err(LowerError::IncompatibleFamilyCoverage {
                                expected: WireType::Int,
                                actual: WireType::Int,
                            });
                        };
                        let value = elements.get(lane).copied().ok_or(
                            LowerError::IncompatibleFamilyCoverage {
                                expected: WireType::Int,
                                actual: WireType::Int,
                            },
                        )?;
                        cases.push(value);
                    }
                    selected.push(make_switch(self, &cases)?);
                }
                FamilyCoverageStorage::ExactStored { elements: selected.into_boxed_slice() }
            }
            FamilyCoverageStorage::SharedTemplate { domain, binder_domains, .. } => {
                let mut representatives = Vec::with_capacity(families.len());
                for family in families {
                    let FamilyCoverageStorage::SharedTemplate {
                        domain: candidate_domain,
                        binder_domains: candidate_binders,
                        representative,
                    } = &family.storage
                    else {
                        return Err(LowerError::IncompatibleFamilyCoverage {
                            expected: WireType::Int,
                            actual: WireType::Int,
                        });
                    };
                    if candidate_domain != domain || candidate_binders != binder_domains {
                        return Err(LowerError::IncompatibleFamilyCoverage {
                            expected: WireType::Int,
                            actual: WireType::Int,
                        });
                    }
                    representatives.push(*representative);
                }
                FamilyCoverageStorage::SharedTemplate {
                    domain: domain.clone(),
                    representative: make_switch(self, &representatives)?,
                    binder_domains: binder_domains.clone(),
                }
            }
        };
        Ok(FamilyLoweringValue { element_type: first.element_type.clone(), storage })
    }

    /// A Select chooses one value at one logical family index.  Branch-local
    /// parallel loops may use distinct binder identities for that same index;
    /// align those alpha-equivalent templates to the first case without
    /// enumerating their logical elements.
    fn align_selected_shared_families(
        &mut self,
        mut families: Vec<FamilyLoweringValue<ScalarId>>,
    ) -> Result<Vec<FamilyLoweringValue<ScalarId>>, ()> {
        let Some(FamilyLoweringValue {
            element_type: common_element_type,
            storage:
                FamilyCoverageStorage::SharedTemplate {
                    domain: common_domain,
                    binder_domains: common_binder_domains,
                    ..
                },
        }) = families.first()
        else {
            return Ok(families);
        };
        let common_element_type = common_element_type.clone();
        let common_domain = common_domain.clone();
        let common_binder_domains = common_binder_domains.clone();
        let common_binder_term = self
            .scalar_store
            .intern_node(ScalarNode::IntBinder(common_domain.binder.clone()), &self.symbols)
            .map_err(|_| ())?;
        for family in families.iter_mut().skip(1) {
            if family.element_type != common_element_type {
                return Err(());
            }
            let FamilyCoverageStorage::SharedTemplate { domain, representative, binder_domains } =
                &family.storage
            else {
                return Ok(families);
            };
            if domain.logical_count != common_domain.logical_count ||
                binder_domains.len() != common_binder_domains.len()
            {
                return Err(());
            }
            let mut normalized_domains = binder_domains.to_vec();
            for binder_domain in &mut normalized_domains {
                if binder_domain.binder == domain.binder {
                    binder_domain.binder = common_domain.binder.clone();
                }
            }
            if normalized_domains.as_slice() != common_binder_domains.as_ref() {
                return Err(());
            }
            if domain.binder == common_domain.binder {
                continue;
            }
            let scope = domain.binder.loop_scope.clone();
            let node = domain.binder.loop_node;
            let control = &mut self.control;
            if let Some(control) = control.as_deref_mut() {
                control.work(&scope, node).map_err(|_| ())?;
            }
            let representative = self
                .substitute_scalar_binder(*representative, &domain.binder, common_binder_term)
                .map_err(|_| ())?;
            family.storage = FamilyCoverageStorage::SharedTemplate {
                domain: common_domain.clone(),
                representative,
                binder_domains: common_binder_domains.clone(),
            };
        }
        Ok(families)
    }

    fn shared_family_element(
        &mut self,
        family: &FamilyLoweringValue<ScalarId>,
        index: &LoweredInt,
    ) -> Result<LoweredValue, LowerError> {
        let (representative, domain, _) = family::shared_element(family)
            .map_err(|_| LowerError::InvalidFamilyCount { count: IntExpr::constant(0) })?;
        let Some(index_analysis) = self.integer_analysis(index.scalar) else {
            return Err(LowerError::MissingIntegerAnalysis { term: index.scalar });
        };
        let index_domain = index_analysis.0;
        if index_domain.interval().map_or(true, |interval| {
            interval.minimum < BigInt::zero() ||
                interval.maximum >= BigInt::from(domain.logical_count.clone())
        }) {
            return Err(LowerError::FamilyAccessOutOfRange {
                index: self
                    .canonical_scalar_identity(index.scalar)
                    .ok()
                    .and_then(|value| resolved_integer(&value).map(IntExpr::constant))
                    .unwrap_or_else(|| IntExpr::constant(-1)),
                count: IntExpr::constant(domain.logical_count.clone()),
            });
        }
        if self.canonical_scalar_identity(index.scalar).ok() ==
            Some(ResolvedIntExpr::Binder(domain.binder.clone()))
        {
            return Ok(LoweredValue::Scalar(*representative));
        }
        let scope = domain.binder.loop_scope.clone();
        let node = domain.binder.loop_node;
        let control = &mut self.control;
        if let Some(control) = control.as_deref_mut() {
            control.work(&scope, node)?;
        }
        self.substitute_scalar_binder(*representative, &domain.binder, index.scalar)
            .map(LoweredValue::Scalar)
            .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)
    }

    /// Rebuilds one scalar template after replacing its owner binder.  The
    /// traversal is an explicit postorder walk over the scalar arena; no
    /// logical family lanes are materialized and no recursive call depth is
    /// proportional to expression depth.
    fn substitute_scalar_binder(
        &mut self,
        root: ScalarId,
        binder: &BinderKey,
        replacement: ScalarId,
    ) -> Result<ScalarId, LowerError> {
        enum Visit {
            Enter(ScalarId),
            Exit(ScalarId),
        }
        let mut completed = std::collections::BTreeMap::<ScalarId, ScalarId>::new();
        let mut work = vec![Visit::Enter(root)];
        while let Some(visit) = work.pop() {
            let id = match visit {
                Visit::Enter(id) => {
                    if completed.contains_key(&id) {
                        continue;
                    }
                    if matches!(self.scalar_store.node(id), Some(ScalarNode::IntBinder(key)) if key == binder)
                    {
                        completed.insert(id, replacement);
                        continue;
                    }
                    work.push(Visit::Exit(id));
                    let children = self
                        .scalar_store
                        .children(id)
                        .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
                    for child in children.iter().rev() {
                        work.push(Visit::Enter(*child));
                    }
                    continue;
                }
                Visit::Exit(id) => id,
            };
            if completed.contains_key(&id) {
                continue;
            }
            let node = self
                .scalar_store
                .node(id)
                .cloned()
                .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
            let remap = |child: ScalarId| *completed.get(&child).expect("scalar postorder child");
            let rebuilt = match node {
                ScalarNode::Source { source, indices } => ScalarNode::Source {
                    source,
                    indices: indices.iter().map(|child| remap(*child)).collect(),
                },
                ScalarNode::IntConst(value) => ScalarNode::IntConst(value),
                ScalarNode::IntParameter(value) => ScalarNode::IntParameter(value),
                ScalarNode::IntBinder(value) => ScalarNode::IntBinder(value),
                ScalarNode::IntAdd(ids) => ScalarNode::IntAdd([remap(ids[0]), remap(ids[1])]),
                ScalarNode::IntSub(ids) => ScalarNode::IntSub([remap(ids[0]), remap(ids[1])]),
                ScalarNode::IntMul(ids) => ScalarNode::IntMul([remap(ids[0]), remap(ids[1])]),
                ScalarNode::IntExactDiv(ids) => {
                    ScalarNode::IntExactDiv([remap(ids[0]), remap(ids[1])])
                }
                ScalarNode::IntEuclideanDiv(ids) => {
                    ScalarNode::IntEuclideanDiv([remap(ids[0]), remap(ids[1])])
                }
                ScalarNode::IntEuclideanRemainder(ids) => {
                    ScalarNode::IntEuclideanRemainder([remap(ids[0]), remap(ids[1])])
                }
                ScalarNode::IntRoundDiv(ids) => {
                    ScalarNode::IntRoundDiv([remap(ids[0]), remap(ids[1])])
                }
                ScalarNode::IntLog2Ceil(ids) => ScalarNode::IntLog2Ceil([remap(ids[0])]),
                ScalarNode::BoolConst(value) => ScalarNode::BoolConst(value),
                ScalarNode::IntEqual(ids) => ScalarNode::IntEqual([remap(ids[0]), remap(ids[1])]),
                ScalarNode::IntLess(ids) => ScalarNode::IntLess([remap(ids[0]), remap(ids[1])]),
                ScalarNode::IntLessEqual(ids) => {
                    ScalarNode::IntLessEqual([remap(ids[0]), remap(ids[1])])
                }
                ScalarNode::BitExtract { bit, input } => {
                    ScalarNode::BitExtract { bit, input: [remap(input[0])] }
                }
                ScalarNode::BoolToInt(ids) => ScalarNode::BoolToInt([remap(ids[0])]),
                ScalarNode::RealConst(bits) => ScalarNode::RealConst(bits),
                ScalarNode::IntToReal(ids) => ScalarNode::IntToReal([remap(ids[0])]),
                ScalarNode::RealAdd(ids) => ScalarNode::RealAdd([remap(ids[0]), remap(ids[1])]),
                ScalarNode::RealSub(ids) => ScalarNode::RealSub([remap(ids[0]), remap(ids[1])]),
                ScalarNode::RealMul(ids) => ScalarNode::RealMul([remap(ids[0]), remap(ids[1])]),
                ScalarNode::RealDiv(ids) => ScalarNode::RealDiv([remap(ids[0]), remap(ids[1])]),
                ScalarNode::RealSqrt(ids) => ScalarNode::RealSqrt([remap(ids[0])]),
                ScalarNode::Switch(ids) => {
                    ScalarNode::Switch(ids.iter().map(|child| remap(*child)).collect())
                }
                ScalarNode::ExtractCoefficient { matrix, position } => {
                    ScalarNode::ExtractCoefficient { matrix, position: remap(position) }
                }
            };
            let rebuilt = self
                .scalar_store
                .intern_node(rebuilt, &self.symbols)
                .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
            completed.insert(id, rebuilt);
        }
        completed.get(&root).copied().ok_or(LowerError::UnsupportedMatrixProductExpansion)
    }

    fn shared_matrix_family_element(
        &mut self,
        family: &FamilyLoweringValue<TermId>,
        index: &LoweredInt,
    ) -> Result<LoweredValue, LowerError> {
        let (representative, domain, _) = family::shared_element(family)
            .map_err(|_| LowerError::InvalidFamilyCount { count: IntExpr::constant(0) })?;
        let Some(index_analysis) = self.integer_analysis(index.scalar) else {
            return Err(LowerError::MissingIntegerAnalysis { term: index.scalar });
        };
        if family::validate_family_index(index_analysis.0, &domain.logical_count).is_err() {
            return Err(LowerError::FamilyAccessOutOfRange {
                index: IntExpr::constant(-1),
                count: IntExpr::constant(domain.logical_count.clone()),
            });
        }
        if self.canonical_scalar_identity(index.scalar).ok() ==
            Some(ResolvedIntExpr::Binder(domain.binder.clone()))
        {
            return Ok(LoweredValue::Matrix(*representative));
        }
        let replacement = self.canonical_scalar_identity(index.scalar)?;
        normal_form_family::validate_matrix_term_family(family).map_err(|_| {
            LowerError::InvalidFamilyCount {
                count: IntExpr::constant(domain.logical_count.clone()),
            }
        })?;
        // The DAG substitution handles arbitrary products, additions, switches,
        // and structural nodes.  It also preserves owner-resolved coordinates
        // in nested factors; no logical family lane is materialized.
        let instantiated = self
            .dag
            .substitute_binder(
                *representative,
                &domain.binder,
                &replacement,
                &mut self.matrix_family_substitution_memo,
            )
            .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
        Ok(LoweredValue::Matrix(instantiated))
    }

    /// Selects one matrix-family element while retaining only the selector's
    /// reachable interval.  Exact storage uses a compact dynamic DAG barrier;
    /// shared storage substitutes its owner binder without enumerating lanes.
    fn matrix_family_element(
        &mut self,
        family: &FamilyLoweringValue<TermId>,
        index: &LoweredInt,
        wire: &LoweringWire,
        environment: &LowerEnv,
    ) -> Result<LoweredValue, LowerError> {
        normal_form_family::validate_matrix_term_family(family)
            .map_err(|_| LowerError::InvalidFamilyCount { count: IntExpr::constant(0) })?;
        if matches!(family.storage, FamilyCoverageStorage::SharedTemplate { .. }) {
            return self.shared_matrix_family_element(family, index);
        }
        let FamilyCoverageStorage::ExactStored { elements } = &family.storage else {
            return Err(LowerError::FamilyAccessOutOfRange {
                index: IntExpr::constant(-1),
                count: IntExpr::constant(0),
            });
        };
        let Some((domain, _)) = self.integer_analysis(index.scalar) else {
            return Err(LowerError::MissingIntegerAnalysis { term: index.scalar });
        };
        let count = BigUint::from(elements.len());
        let interval = domain.interval().map_err(|_| LowerError::FamilyAccessOutOfRange {
            index: IntExpr::constant(-1),
            count: IntExpr::constant(count.clone()),
        })?;
        if interval.minimum < BigInt::zero() || interval.maximum >= BigInt::from(count.clone()) {
            return Err(LowerError::FamilyAccessOutOfRange {
                index: IntExpr::constant(-1),
                count: IntExpr::constant(count),
            });
        }
        let stable = self.canonical_scalar_identity(index.scalar)?;
        {
            if let Some(term) =
                normal_form_family::static_matrix_term(family, &stable).map_err(|_| {
                    LowerError::FamilyAccessOutOfRange {
                        index: IntExpr::constant(-1),
                        count: IntExpr::constant(BigUint::from(elements.len())),
                    }
                })?
            {
                return Ok(LoweredValue::Matrix(term));
            }
        }
        let start =
            interval.minimum.to_usize().ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
        let end =
            interval.maximum.to_usize().ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
        let selector = self.family_selector_identity(family, index, wire, environment)?;
        let term = self
            .dag
            .push(ExpressionNode::FamilyGetDynamic {
                selector,
                cases: elements[start..=end].iter().copied().collect(),
                stored_indices: (start..=end).map(BigUint::from).collect(),
                domain_upper: BigUint::from(end + 1),
            })
            .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
        Ok(LoweredValue::Matrix(term))
    }

    fn family_element(
        &mut self,
        family: &FamilyLoweringValue<ScalarId>,
        index: &LoweredInt,
    ) -> Result<LoweredValue, LowerError> {
        match &family.storage {
            FamilyCoverageStorage::ExactStored { elements } => {
                family::static_get(family, &self.canonical_scalar_identity(index.scalar)?)
                    .map_err(|_| LowerError::FamilyAccessOutOfRange {
                        index: IntExpr::constant(-1),
                        count: IntExpr::constant(elements.len()),
                    })?
                    .map(LoweredValue::Scalar)
                    .map_or_else(
                        || {
                            self.scalar_dynamic_get(family, index.scalar)
                                .map(LoweredValue::Scalar)
                                .map_err(|_| LowerError::InvalidFamilyCount {
                                    count: IntExpr::constant(elements.len()),
                                })
                        },
                        Ok,
                    )
            }
            FamilyCoverageStorage::SharedTemplate { .. } => {
                self.shared_family_element(family, index)
            }
        }
    }

    fn scalar_dynamic_get(
        &mut self,
        family: &FamilyLoweringValue<ScalarId>,
        selector: ScalarId,
    ) -> Result<ScalarId, LowerError> {
        let FamilyCoverageStorage::ExactStored { elements } = &family.storage else {
            return Err(LowerError::InvalidFamilyCount { count: IntExpr::constant(0) });
        };
        if elements.is_empty() {
            return Err(LowerError::InvalidFamilyCount { count: IntExpr::constant(0) });
        }
        let facts = self
            .scalar_store
            .facts(selector)
            .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
        let domain =
            facts.integer_domain.as_ref().ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
        let interval = domain.interval().map_err(|_| LowerError::FamilyAccessOutOfRange {
            index: IntExpr::constant(-1),
            count: IntExpr::constant(elements.len()),
        })?;
        if interval.minimum < BigInt::zero() || interval.maximum >= BigInt::from(elements.len()) {
            return Err(LowerError::FamilyAccessOutOfRange {
                index: IntExpr::constant(-1),
                count: IntExpr::constant(elements.len()),
            });
        }
        let children =
            std::iter::once(selector).chain(elements.iter().copied()).collect::<Box<_>>();
        self.scalar_store
            .intern_node(ScalarNode::Switch(children), &self.symbols)
            .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)
    }

    fn trapdoor_family_element(
        &mut self,
        representative: TrapdoorDescriptorId,
        binder: &BinderKey,
        logical_count: &num_bigint::BigUint,
        index: &LoweredInt,
    ) -> Result<LoweredValue, LowerError> {
        let index_domain = self
            .integer_analysis(index.scalar)
            .map(|(domain, _)| domain)
            .ok_or_else(|| LowerError::FamilyAccessOutOfRange {
                index: IntExpr::constant(-1),
                count: IntExpr::constant(logical_count.clone()),
            })?;
        family::validate_family_index(index_domain, logical_count).map_err(|_| {
            LowerError::FamilyAccessOutOfRange {
                index: IntExpr::constant(-1),
                count: IntExpr::constant(logical_count.clone()),
            }
        })?;
        let template = self.symbols.trapdoors.get(representative.0).cloned().ok_or(
            LowerError::FamilyProducerNotResolved {
                family: WireRef { node: binder.loop_node, port: mxx_ir_core::Port(binder.slot) },
            },
        )?;
        let replacement = self.canonical_scalar_identity(index.scalar)?;
        let indices = template
            .indices
            .iter()
            .map(|value| super::identity::substitute_resolved_int_expr(value, binder, &replacement))
            .collect::<Box<[_]>>();
        // The public operand is stable provenance, not a matrix DAG atom.
        // Its graph occurrence is shared by every family lane; only the
        // ordered coordinate expressions vary with the selected binder.
        let descriptor = TrapdoorIdentity { indices, ..template };
        let descriptor = self.symbols.trapdoors.intern(descriptor);
        Ok(LoweredValue::Trapdoor(TrapdoorDescriptorId(descriptor)))
    }

    fn normalize_singleton_for_input(
        &mut self,
        value: LoweredValue,
        input_type: &WireType,
    ) -> Result<LoweredValue, LowerError> {
        if matches!(input_type, WireType::IndexedFamily { .. }) {
            return Ok(value);
        }
        let zero = self.add_int(BigInt::zero());
        match value {
            LoweredValue::Family(family)
                if matches!(
                    &family.storage,
                    FamilyCoverageStorage::SharedTemplate { domain, .. }
                        if domain.logical_count == num_bigint::BigUint::from(1_u8)
                ) =>
            {
                self.family_element(&family, &zero)
            }
            LoweredValue::TrapdoorFamily { representative, binder, logical_count }
                if logical_count == num_bigint::BigUint::from(1_u8) =>
            {
                self.trapdoor_family_element(representative, &binder, &logical_count, &zero)
            }
            LoweredValue::MatrixFamily(family)
                if matches!(
                    &family.storage,
                    FamilyCoverageStorage::SharedTemplate { domain, .. }
                        if domain.logical_count == num_bigint::BigUint::from(1_u8)
                ) =>
            {
                self.shared_matrix_family_element(&family, &zero)
            }
            LoweredValue::MatrixFamily(family) => {
                if !matches!(
                    &family.storage,
                    FamilyCoverageStorage::ExactStored { elements } if elements.len() == 1
                ) {
                    return Err(LowerError::UnsupportedMatrixProductExpansion);
                }
                let term = normal_form_family::static_matrix_term(
                    &family,
                    &ResolvedIntExpr::Const(BigInt::zero()),
                )
                .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?
                .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
                Ok(LoweredValue::Matrix(term))
            }
            value => Ok(value),
        }
    }

    /// Enters a parallel body once with an owner-resolved binder.  Its output is retained as a
    /// shared template, so a 30,720-lane family costs one body traversal rather than one
    /// traversal per logical lane.
    fn queue_parallel_loop(
        &mut self,
        wire: LoweringWire,
        specification: ParallelLoop,
        arguments: Vec<LoweredValue>,
        environment: LowerEnv,
        output_type: WireType,
        work: &mut Vec<LoweringFrame>,
    ) -> Result<(), LowerError> {
        let count = self.lower_int_expr(&specification.count, &environment)?;
        let Some((domain, _)) = self.integer_analysis(count.scalar) else {
            return Err(LowerError::InvalidFamilyCount { count: specification.count.clone() });
        };
        let range = domain
            .interval()
            .map_err(|_| LowerError::InvalidFamilyCount { count: specification.count.clone() })?;
        if range.minimum != range.maximum || range.minimum <= BigInt::zero() {
            return Err(LowerError::InvalidFamilyCount { count: specification.count.clone() });
        }
        let logical_count = range
            .minimum
            .to_biguint()
            .ok_or_else(|| LowerError::InvalidFamilyCount { count: specification.count.clone() })?;
        if arguments.len() != specification.input_modes.len() {
            return Err(LowerError::InvalidOperandArity {
                expected: specification.input_modes.len(),
                actual: arguments.len(),
            });
        }
        let (parent_arguments, child_definition, child_inputs, child_outputs) = {
            let active_graph = self.graph_for_program(&environment.occurrence.program)?;
            let parent_scope = active_graph
                .scope(&wire.source.scope.definition)
                .ok_or(LowerError::MissingWire { wire: wire.source.wire })?;
            let parent_node = parent_scope
                .node(wire.source.wire.node)
                .ok_or(LowerError::MissingNode { node: wire.source.wire.node })?;
            let parent_arguments = parent_scope
                .arguments(parent_node)
                .ok_or(LowerError::MissingWire { wire: wire.source.wire })?;
            let child_definition = active_graph
                .child_scope_id(&wire.source.scope.definition, wire.source.wire.node)
                .ok_or(LowerError::MissingWire { wire: wire.source.wire })?;
            let child_scope = active_graph
                .scope(&child_definition)
                .ok_or(LowerError::MissingWire { wire: wire.source.wire })?;
            (
                parent_arguments,
                child_definition,
                child_scope.inputs().to_vec(),
                child_scope.outputs().to_vec(),
            )
        };
        if child_inputs.len() != parent_arguments.len() ||
            parent_arguments.len() != specification.input_modes.len()
        {
            return Err(LowerError::InvalidOperandArity {
                expected: child_inputs.len(),
                actual: parent_arguments.len(),
            });
        }
        let binder = BinderKey {
            loop_scope: environment.occurrence.clone(),
            loop_node: wire.source.wire.node,
            slot: specification.index_slot,
        };
        self.symbols.binders.intern(super::identity::BinderDescriptor {
            key: binder.clone(),
            minimum: BigInt::zero(),
            maximum: range.minimum.clone() - BigInt::from(1_u8),
        });
        let scalar = self
            .scalar_store
            .intern_node(ScalarNode::IntBinder(binder.clone()), &self.symbols)
            .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
        let index = LoweredInt { scalar };
        let mut child = environment.clone();
        child.occurrence.definition = child_definition;
        child.occurrence.path = environment
            .occurrence
            .path
            .iter()
            .cloned()
            .chain([super::identity::OccurrenceFrame::ParallelLoop {
                parent: wire.source.scope.definition.clone(),
                owner: wire.source.wire.node,
            }])
            .collect();
        child.binders.push((binder.clone(), index.clone()));
        child.active_coordinates.push(Coordinate { binder: binder.clone(), index: index.clone() });
        for ((input, argument), mode) in
            child_inputs.iter().copied().zip(arguments).zip(specification.input_modes.iter())
        {
            let value = match mode {
                LoopInputMode::Broadcast => argument,
                LoopInputMode::Zip => match argument {
                    LoweredValue::Family(family) => self.family_element(&family, &index)?,
                    LoweredValue::MatrixFamily(family) => {
                        self.matrix_family_element(&family, &index, &wire, &environment)?
                    }
                    LoweredValue::TrapdoorFamily { representative, binder, logical_count } => self
                        .trapdoor_family_element(representative, &binder, &logical_count, &index)?,
                    value => value,
                },
                LoopInputMode::ZipOffset { offset } => {
                    let offset = self.add_int(BigInt::from(*offset));
                    let offset_index =
                        self.combine_int(vec![index.clone(), offset], ScalarOperation::Add)?;
                    match argument {
                        LoweredValue::Family(family) => {
                            self.family_element(&family, &offset_index)?
                        }
                        LoweredValue::MatrixFamily(family) => {
                            self.matrix_family_element(&family, &offset_index, &wire, &environment)?
                        }
                        LoweredValue::TrapdoorFamily { representative, binder, logical_count } => {
                            self.trapdoor_family_element(
                                representative,
                                &binder,
                                &logical_count,
                                &offset_index,
                            )?
                        }
                        value => value,
                    }
                }
            };
            let input_type = self
                .graph_for_program(&child.occurrence.program)?
                .scope(&child.occurrence.definition)
                .and_then(|scope| scope.node(input.node))
                .and_then(|node| node.output_types().get(input.port.0 as usize))
                .cloned()
                .ok_or(LowerError::MissingWire { wire: input })?;
            let value = self.normalize_singleton_for_input(value, &input_type)?;
            child
                .state_inputs
                .insert(WireSourceKey { scope: child.occurrence.clone(), wire: input }, value);
        }
        for (name, expression) in &specification.bindings {
            let value = self.resolve_int(expression, &child)?;
            child.parameters.insert(name.clone(), value);
        }
        let body_output = *child_outputs.get(wire.source.wire.port.0 as usize).ok_or(
            LowerError::InvalidOutputPort {
                wire: wire.source.wire,
                output_count: child_outputs.len(),
            },
        )?;
        work.push(LoweringFrame::FinishParallelLoop {
            wire,
            body_source: WireSourceKey { scope: child.occurrence.clone(), wire: body_output },
            environment,
            specification,
            output_type,
            binder,
            logical_count,
            maximum: range.minimum - BigInt::from(1_u8),
        });
        work.push(LoweringFrame::Enter {
            wire: LoweringWire {
                source: WireSourceKey { scope: child.occurrence.clone(), wire: body_output },
                indices: Box::new([]),
            },
            environment: child,
        });
        Ok(())
    }

    fn finish_parallel_loop_matrix(
        &mut self,
        specification: &ParallelLoop,
        environment: &LowerEnv,
        output_type: WireType,
        binder: BinderKey,
        logical_count: num_bigint::BigUint,
        maximum: BigInt,
        representative: TermId,
    ) -> Result<LoweredValue, LowerError> {
        let WireType::IndexedFamily { element, .. } = output_type else {
            return Err(LowerError::IncompatibleFamilyCoverage {
                expected: WireType::IndexedFamily {
                    element: Box::new(WireType::Int),
                    count: specification.count.clone(),
                },
                actual: output_type,
            });
        };
        let element_wire_type = *element;
        let element_type = self.resolve_family_element_sort(&element_wire_type, environment)?;
        let mut binder_domains = environment
            .active_coordinates
            .iter()
            .map(|coordinate| {
                let interval = self
                    .integer_analysis(coordinate.index.scalar)
                    .and_then(|(domain, _)| domain.interval().ok())
                    .ok_or_else(|| LowerError::InvalidFamilyCount {
                        count: specification.count.clone(),
                    })?;
                Ok(family::CoverageBinderDomain {
                    binder: coordinate.binder.clone(),
                    minimum: interval.minimum,
                    maximum: interval.maximum,
                })
            })
            .collect::<Result<Vec<_>, LowerError>>()?;
        binder_domains.push(family::CoverageBinderDomain {
            binder: binder.clone(),
            minimum: BigInt::zero(),
            maximum,
        });
        let value = FamilyLoweringValue {
            element_type,
            storage: FamilyCoverageStorage::SharedTemplate {
                domain: family::LoopDomainKey { binder, logical_count },
                representative,
                binder_domains: binder_domains.into_boxed_slice(),
            },
        };
        normal_form_family::validate_matrix_term_family(&value)
            .map_err(|_| LowerError::InvalidFamilyCount { count: specification.count.clone() })?;
        Ok(LoweredValue::MatrixFamily(value))
    }

    fn finish_parallel_loop(
        &mut self,
        specification: &ParallelLoop,
        environment: &LowerEnv,
        output_type: WireType,
        binder: BinderKey,
        logical_count: num_bigint::BigUint,
        maximum: BigInt,
        representative: ScalarId,
        body_source: WireSourceKey,
    ) -> Result<LoweredValue, LowerError> {
        let WireType::IndexedFamily { element, .. } = output_type else {
            return Err(LowerError::IncompatibleFamilyCoverage {
                expected: WireType::IndexedFamily {
                    element: Box::new(WireType::Int),
                    count: specification.count.clone(),
                },
                actual: output_type,
            });
        };
        let element_wire_type = *element;
        let element_type = self.resolve_family_element_sort(&element_wire_type, environment)?;
        let actual_sort = self.scalar_store.facts(representative).map(|facts| &facts.sort);
        let sort_matches = match (&element_type, actual_sort) {
            (ScalarSort::Matrix(expected), Some(Ok(ScalarSort::Matrix(actual)))) => {
                matrix_types_equal(expected, actual)
            }
            (expected, Some(Ok(actual))) => expected == actual,
            (_, Some(Err(_)) | None) => false,
        };
        if !sort_matches {
            return Err(LowerError::FamilyElementLoweringMismatch {
                expected: element_wire_type,
                actual_category: super::error::LoweredValueCategory::Term,
                actual_sort: actual_sort.and_then(|sort| sort.as_ref().ok().cloned()),
                producer: body_source,
            });
        }
        let mut binder_domains = environment
            .active_coordinates
            .iter()
            .map(|coordinate| {
                let interval = self
                    .integer_analysis(coordinate.index.scalar)
                    .and_then(|(domain, _)| domain.interval().ok())
                    .ok_or_else(|| LowerError::InvalidFamilyCount {
                        count: specification.count.clone(),
                    })?;
                Ok(family::CoverageBinderDomain {
                    binder: coordinate.binder.clone(),
                    minimum: interval.minimum,
                    maximum: interval.maximum,
                })
            })
            .collect::<Result<Vec<_>, LowerError>>()?;
        binder_domains.push(family::CoverageBinderDomain {
            binder: binder.clone(),
            minimum: BigInt::zero(),
            maximum,
        });
        let value = FamilyLoweringValue {
            element_type,
            storage: FamilyCoverageStorage::SharedTemplate {
                domain: family::LoopDomainKey { binder: binder.clone(), logical_count },
                representative,
                binder_domains: binder_domains.into_boxed_slice(),
            },
        };
        value
            .validate()
            .map_err(|_| LowerError::InvalidFamilyCount { count: specification.count.clone() })?;
        Ok(LoweredValue::Family(value))
    }

    fn queue_sequential_loop(
        &mut self,
        wire: LoweringWire,
        specification: SequentialLoop,
        arguments: Vec<LoweredValue>,
        environment: LowerEnv,
        output_type: WireType,
        work: &mut Vec<LoweringFrame>,
    ) -> Result<(), LowerError> {
        if specification.carried_count == 0 || arguments.len() < specification.carried_count {
            return Err(LowerError::InvalidOperandArity {
                expected: specification.carried_count,
                actual: arguments.len(),
            });
        }
        let count = self.lower_int_expr(&specification.count, &environment)?;
        let Some((domain, _)) = self.integer_analysis(count.scalar) else {
            return Err(LowerError::InvalidFamilyCount { count: specification.count.clone() });
        };
        let count_range = domain
            .interval()
            .map_err(|_| LowerError::InvalidFamilyCount { count: specification.count.clone() })?;
        if count_range.minimum != count_range.maximum || count_range.minimum < BigInt::zero() {
            return Err(LowerError::InvalidFamilyCount { count: specification.count.clone() });
        }
        let carried_index = wire.source.wire.port.0 as usize;
        if carried_index >= specification.carried_count {
            return Err(LowerError::InvalidOutputPort {
                wire: wire.source.wire,
                output_count: specification.carried_count,
            });
        }
        let matrix_output = matches!(output_type, WireType::Matrix(_) | WireType::Preimage(_));
        if !matrix_output {
            return Err(LowerError::UnsupportedMatrixProductExpansion);
        }
        let matrix_arguments = arguments
            .iter()
            .take(specification.carried_count)
            .all(|value| matches!(value, LoweredValue::Matrix(_)));
        if matrix_output && !matrix_arguments {
            return Err(LowerError::UnsupportedMatrixProductExpansion);
        }
        if count_range.minimum.is_zero() {
            let value =
                arguments.get(carried_index).cloned().ok_or(LowerError::InvalidOperandArity {
                    expected: specification.carried_count,
                    actual: arguments.len(),
                })?;
            work.push(LoweringFrame::FinishValue { wire, value });
            return Ok(());
        }
        let count_identity = self.canonical_scalar_identity(count.scalar)?;
        self.queue_sequential_matrix_loop(
            wire,
            specification,
            arguments,
            environment,
            output_type,
            count_identity,
            count_range.maximum,
            work,
        )
    }

    /// Queues a matrix recurrence without creating a scalar matrix atom.
    /// Integer transfer still uses the scalar store, but every carried
    /// value and every transition output remains a `TermId` in the job DAG.
    fn queue_sequential_matrix_loop(
        &mut self,
        wire: LoweringWire,
        specification: SequentialLoop,
        arguments: Vec<LoweredValue>,
        environment: LowerEnv,
        output_type: WireType,
        count: ResolvedIntExpr,
        maximum: BigInt,
        work: &mut Vec<LoweringFrame>,
    ) -> Result<(), LowerError> {
        let (child_definition, child_inputs, child_outputs, parent_arguments) = {
            let graph = self.graph_for_program(&environment.occurrence.program)?;
            let parent = graph
                .scope(&wire.source.scope.definition)
                .ok_or(LowerError::MissingWire { wire: wire.source.wire })?;
            let node = parent
                .node(wire.source.wire.node)
                .ok_or(LowerError::MissingNode { node: wire.source.wire.node })?;
            let parent_arguments =
                parent.arguments(node).ok_or(LowerError::MissingWire { wire: wire.source.wire })?;
            let child_definition = graph
                .child_scope_id(&wire.source.scope.definition, wire.source.wire.node)
                .ok_or(LowerError::MissingWire { wire: wire.source.wire })?;
            let child = graph
                .scope(&child_definition)
                .ok_or(LowerError::MissingWire { wire: wire.source.wire })?;
            (child_definition, child.inputs().to_vec(), child.outputs().to_vec(), parent_arguments)
        };
        if child_inputs.len() != arguments.len() ||
            parent_arguments.len() != arguments.len() ||
            child_outputs.len() != specification.carried_count
        {
            return Err(LowerError::InvalidOperandArity {
                expected: child_inputs.len(),
                actual: arguments.len(),
            });
        }
        let mut child = environment.clone();
        child.occurrence.definition = child_definition;
        child.occurrence.path = environment
            .occurrence
            .path
            .iter()
            .cloned()
            .chain([super::identity::OccurrenceFrame::SequentialLoop {
                parent: wire.source.scope.definition.clone(),
                owner: wire.source.wire.node,
            }])
            .collect();
        let binder = BinderKey {
            loop_scope: environment.occurrence.clone(),
            loop_node: wire.source.wire.node,
            slot: specification.index_slot,
        };
        self.symbols.binders.intern(super::identity::BinderDescriptor {
            key: binder.clone(),
            minimum: BigInt::zero(),
            maximum: maximum - BigInt::from(1_u8),
        });
        let scalar = self
            .scalar_store
            .intern_node(ScalarNode::IntBinder(binder.clone()), &self.symbols)
            .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
        let iteration = LoweredInt { scalar };
        child.active_coordinates.push(Coordinate { binder: binder.clone(), index: iteration });

        let mut initial = Vec::with_capacity(specification.carried_count);
        let mut state_factors = Vec::with_capacity(specification.carried_count);
        let mut output_types = Vec::with_capacity(specification.carried_count);
        for position in 0..specification.carried_count {
            let LoweredValue::Matrix(initial_term) = arguments[position].clone() else {
                return Err(LowerError::UnsupportedMatrixProductExpansion);
            };
            initial.push(initial_term);
            let graph = self.graph_for_program(&child.occurrence.program)?;
            let scope = graph
                .scope(&child.occurrence.definition)
                .ok_or(LowerError::MissingWire { wire: child_inputs[position] })?;
            let input_node = scope
                .node(child_inputs[position].node)
                .ok_or(LowerError::MissingNode { node: child_inputs[position].node })?;
            let input_type =
                input_node.output_types()[child_inputs[position].port.0 as usize].clone();
            let (WireType::Matrix(matrix) | WireType::Preimage(matrix)) = input_type else {
                return Err(LowerError::InvalidOperandSort {
                    expected: output_type.clone(),
                    actual: input_type,
                });
            };
            let matrix_type = self.resolve_matrix_type(&matrix, &child)?;
            let concrete = concrete_matrix_type(&matrix_type)
                .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
            output_types.push(matrix_type);
            let state_key = SequentialStateKey {
                loop_scope: environment.occurrence.clone(),
                loop_node: wire.source.wire.node,
                carried_index: position,
            };
            let factor_key = FactorIdentity::atomic(
                super::identity::AtomicSourceKey::SequentialState(state_key),
                std::iter::empty(),
            );
            let placeholder = SymbolicFactor {
                key: factor_key.clone(),
                bound: BoundClass::Large,
                relation_live: false,
                trapdoor: None,
                matrix_bound: Some(MatrixBound {
                    matrix_type: concrete.clone(),
                    coefficient_class: BoundClass::Large,
                }),
                matrix_type: concrete.clone(),
                polynomial_facts: crate::operational_noise::bound::PolynomialFacts::conservative(
                    concrete.ring_dimension,
                ),
                matrix_value_metadata: MatrixMetadata::unknown(),
                switch: None,
            };
            let placeholder = self
                .dag
                .push(ExpressionNode::Atom(placeholder))
                .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
            state_factors.push(factor_key);
            child.state_inputs.insert(
                WireSourceKey { scope: child.occurrence.clone(), wire: child_inputs[position] },
                LoweredValue::Matrix(placeholder),
            );
        }
        for (input, argument) in child_inputs
            .iter()
            .copied()
            .skip(specification.carried_count)
            .zip(arguments.iter().skip(specification.carried_count).cloned())
        {
            child
                .state_inputs
                .insert(WireSourceKey { scope: child.occurrence.clone(), wire: input }, argument);
        }
        for (name, expression) in &specification.bindings {
            let value = self.resolve_int(expression, &child)?;
            child.parameters.insert(name.clone(), value);
        }
        let dependency_count = child_outputs.len();
        let carried_index = wire.source.wire.port.0 as usize;
        work.push(LoweringFrame::FinishSequentialMatrixLoop {
            wire,
            environment,
            count,
            initial,
            state_factors,
            iteration_binder: Some(binder),
            output_types,
            output_type,
            carried_index,
            dependency_count,
        });
        for output in child_outputs.into_iter().rev() {
            work.push(LoweringFrame::Enter {
                wire: LoweringWire {
                    source: WireSourceKey { scope: child.occurrence.clone(), wire: output },
                    indices: Box::new([]),
                },
                environment: child.clone(),
            });
        }
        Ok(())
    }

    fn finish_sequential_matrix_loop(
        &mut self,
        wire: &LoweringWire,
        environment: &LowerEnv,
        count: ResolvedIntExpr,
        initial: Vec<TermId>,
        state_factors: Vec<FactorIdentity>,
        iteration_binder: Option<BinderKey>,
        transition: Vec<TermId>,
        output_types: Vec<super::identity::ResolvedMatrixType>,
        output_type: WireType,
        carried_index: usize,
    ) -> Result<LoweredValue, LowerError> {
        let (WireType::Matrix(matrix) | WireType::Preimage(matrix)) = output_type else {
            return Err(LowerError::UnsupportedMatrixProductExpansion);
        };
        if carried_index >= transition.len() ||
            initial.len() != transition.len() ||
            initial.len() != state_factors.len() ||
            initial.len() != output_types.len()
        {
            return Err(LowerError::InvalidOperandArity {
                expected: initial.len(),
                actual: transition.len(),
            });
        }
        let output_matrix = self.resolve_matrix_type(&matrix, environment)?;
        let output_matrix = concrete_matrix_type(&output_matrix)
            .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
        for (term, expected) in transition.iter().zip(output_types.iter()) {
            let actual = self.dag_matrix_type(*term)?;
            let expected = concrete_matrix_type(expected)
                .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
            if actual != expected {
                return Err(LowerError::UnsupportedMatrixProductExpansion);
            }
        }
        if self.dag_matrix_type(transition[carried_index])? != output_matrix {
            return Err(LowerError::UnsupportedMatrixProductExpansion);
        }
        let recurrence = super::normal_form_family::TermSequentialRecurrence {
            initial: initial.into_boxed_slice(),
            transition: transition.into_boxed_slice(),
            state_factors: state_factors.into_boxed_slice(),
            iteration_binder,
            count: IntegerDomain::Exact(
                resolved_integer(&count).ok_or(LowerError::UnsupportedMatrixProductExpansion)?,
            ),
        };
        let state = recurrence
            .evaluate(&mut self.dag)
            .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
        let term = state.get(carried_index).copied().ok_or(LowerError::InvalidOutputPort {
            wire: wire.source.wire,
            output_count: state.len(),
        })?;
        Ok(LoweredValue::Matrix(term))
    }

    fn resolve_family_element_sort(
        &mut self,
        element: &WireType,
        environment: &LowerEnv,
    ) -> Result<ScalarSort, LowerError> {
        match element {
            WireType::Int => Ok(ScalarSort::Int),
            WireType::Bool => Ok(ScalarSort::Bool),
            WireType::Matrix(matrix) => {
                self.resolve_matrix_type(matrix, environment).map(ScalarSort::Matrix)
            }
            actual => Err(LowerError::FamilyElementTypeMismatch {
                expected: WireType::Int,
                actual: actual.clone(),
            }),
        }
    }

    /// Gives a selector an owner-resolved identity for a DAG barrier.  The
    /// loop owner and selector coordinate are retained; no runtime value is
    /// guessed and no logical case is expanded here.
    fn family_selector_identity(
        &self,
        family: &FamilyLoweringValue<TermId>,
        selector: &LoweredInt,
        _wire: &LoweringWire,
        _environment: &LowerEnv,
    ) -> Result<FactorIdentity, LowerError> {
        let _ = family;
        self.scalar_selector_identity(selector.scalar)
    }

    fn scalar_selector_identity(&self, selector: ScalarId) -> Result<FactorIdentity, LowerError> {
        Ok(FactorIdentity::scalar_selector(self.canonical_scalar_identity(selector)?))
    }

    fn resolve_int(
        &mut self,
        expression: &IntExpr,
        environment: &LowerEnv,
    ) -> Result<ResolvedIntExpr, LowerError> {
        let lowered = self.lower_int_expr(expression, environment)?;
        self.canonical_scalar_identity(lowered.scalar)
    }

    fn resolve_range(
        &mut self,
        range: &mxx_ir_core::node::IndexRange,
        environment: &LowerEnv,
    ) -> Result<super::identity::ResolvedIndexRange, LowerError> {
        Ok(super::identity::ResolvedIndexRange {
            start: self.resolve_int(&range.start, environment)?,
            end: self.resolve_int(&range.end, environment)?,
        })
    }

    fn resolve_matrix_type(
        &mut self,
        ty: &MatrixType,
        environment: &LowerEnv,
    ) -> Result<super::identity::ResolvedMatrixType, LowerError> {
        Ok(super::identity::ResolvedMatrixType {
            modulus: self.resolve_int(&ty.modulus, environment)?,
            ring_dimension: self.resolve_int(&ty.ring_dimension, environment)?,
            rows: self.resolve_int(&ty.rows, environment)?,
            columns: self.resolve_int(&ty.columns, environment)?,
        })
    }

    fn validate_gadget_decompose(
        &mut self,
        base: &IntExpr,
        digit_count: &IntExpr,
        environment: &LowerEnv,
        output_type: &WireType,
        argument: &LoweredValue,
    ) -> Result<(), LowerError> {
        if !matches!(argument, LoweredValue::Matrix(_)) {
            return Err(LowerError::UnsupportedMatrixProductExpansion);
        }
        let (WireType::Matrix(matrix) | WireType::Preimage(matrix)) = output_type else {
            return Err(LowerError::UnsupportedMatrixProductExpansion);
        };
        let resolved_matrix = self.resolve_matrix_type(matrix, environment)?;
        let base = self.resolve_int(base, environment)?;
        let digit_count = self.resolve_int(digit_count, environment)?;
        let Some(base_value) = resolved_integer(&base) else {
            return Err(LowerError::NonExactIdentityIndex { expression: IntExpr::constant(0) });
        };
        let Some(digit_count_value) =
            resolved_nonnegative(&digit_count).and_then(|value| value.to_usize())
        else {
            return Err(LowerError::NonExactIdentityIndex { expression: IntExpr::constant(0) });
        };
        let Some(output_rows) =
            resolved_nonnegative(&resolved_matrix.rows).and_then(|value| value.to_usize())
        else {
            return Err(LowerError::NonExactIdentityIndex { expression: matrix.rows.clone() });
        };
        if base_value <= BigInt::from(1) ||
            digit_count_value == 0 ||
            output_rows == 0 ||
            output_rows % digit_count_value != 0
        {
            return Err(LowerError::InvalidOperandArity {
                expected: digit_count_value,
                actual: output_rows,
            });
        }
        Ok(())
    }

    fn resolve_real(&self, value: &RealExpr, environment: &LowerEnv) -> Result<f64, LowerError> {
        let mut env = mxx_ir_core::ParamEnv::default();
        for (name, value) in &environment.parameters {
            if let Some(value) = resolved_integer(value) {
                env.integers.insert(name.clone(), value);
            }
        }
        for (name, parameter) in &self.request.environment {
            let rational = match parameter {
                super::OperationalParameterValue::Integer(value) => {
                    mxx_ir_core::Rational::from_integer(value.clone())
                }
                super::OperationalParameterValue::Rational { numerator, denominator } => {
                    mxx_ir_core::Rational::new(numerator.clone(), denominator.clone()).map_err(
                        |_| LowerError::InvalidRealOperation {
                            operation: NodeKind::ConstantReal(value.clone()),
                        },
                    )?
                }
            };
            env.reals.insert(name.clone(), rational);
        }
        value.evaluate_f64(&env).ok().filter(|value| value.is_finite()).ok_or_else(|| {
            LowerError::InvalidRealOperation { operation: NodeKind::ConstantReal(value.clone()) }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::normal_form::monomial_bound;
    use num_bigint::BigInt;
    use std::collections::BTreeMap;

    #[derive(Default)]
    struct RecordingLoweringControl {
        sites: Vec<(OccurrenceScope, mxx_ir_core::NodeId)>,
    }

    impl LoweringControl for RecordingLoweringControl {
        fn work(
            &mut self,
            scope: &OccurrenceScope,
            node: mxx_ir_core::NodeId,
        ) -> Result<(), LowerError> {
            self.sites.push((scope.clone(), node));
            Ok(())
        }
    }

    fn root_test_environment() -> LowerEnv {
        LowerEnv {
            occurrence: OccurrenceScope {
                program: super::super::identity::ProgramKey::WorkflowStage(StageId(
                    "encrypt".to_owned(),
                )),
                definition: FrozenGraphScopeId::Root,
                path: Box::new([]),
            },
            parameters: BTreeMap::new(),
            binders: Vec::new(),
            inputs: BTreeMap::new(),
            state_inputs: BTreeMap::new(),
            active_coordinates: Vec::new(),
        }
    }

    fn test_integer_atom(
        lowerer: &mut GraphLowerer<'_, '_>,
        name: &str,
        minimum: i64,
        maximum: i64,
    ) -> ScalarId {
        let source =
            lowerer.symbols.atomic_sources.intern(super::super::identity::AtomicSourceDescriptor {
                key: super::super::identity::AtomicSourceKey::ProtocolInput(
                    crate::ProtocolInputId::from(name),
                ),
                sort: ScalarSort::Int,
                integer_domain: Some(super::super::identity::IntegerSourceDomain {
                    minimum: minimum.into(),
                    maximum: maximum.into(),
                }),
                canonical_residue_convention: None,
                relation_role: None,
            });
        lowerer
            .scalar_store
            .intern_node(
                ScalarNode::Source {
                    source: super::super::identity::AtomicSourceId(source),
                    indices: Box::new([]),
                },
                &lowerer.symbols,
            )
            .expect("test source transfers")
    }

    fn test_int(lowerer: &mut GraphLowerer<'_, '_>, value: i64) -> ScalarId {
        lowerer
            .scalar_store
            .intern_node(ScalarNode::IntConst(value.into()), &lowerer.symbols)
            .expect("test integer transfers")
    }

    fn test_binder(lowerer: &mut GraphLowerer<'_, '_>, binder: BinderKey) -> ScalarId {
        lowerer.symbols.binders.intern(super::super::identity::BinderDescriptor {
            key: binder.clone(),
            minimum: BigInt::zero(),
            maximum: BigInt::from(511_u16),
        });
        lowerer
            .scalar_store
            .intern_node(ScalarNode::IntBinder(binder.clone()), &lowerer.symbols)
            .expect("test binder transfers")
    }

    fn test_resolved_matrix() -> super::super::identity::ResolvedMatrixType {
        super::super::identity::ResolvedMatrixType {
            modulus: ResolvedIntExpr::Const(17.into()),
            ring_dimension: ResolvedIntExpr::Const(1.into()),
            rows: ResolvedIntExpr::Const(1.into()),
            columns: ResolvedIntExpr::Const(1.into()),
        }
    }

    fn recurrence_lowerer() -> (ProtocolDecl, OperationalCheckRequest) {
        (
            crate::toy_example::protocol(),
            OperationalCheckRequest {
                environment: Vec::new(),
                layouts: Vec::new(),
                target_id: "recurrence-bound".to_owned(),
            },
        )
    }

    #[test]
    fn matrix_sequential_term_recurrence_handles_zero_one_n_and_relation_exposure() {
        let (protocol, request) = recurrence_lowerer();
        for count in [0_u64, 1, 3] {
            let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
            let environment = root_test_environment();
            let matrix = test_resolved_matrix();
            let concrete = concrete_matrix_type(&matrix).unwrap();
            let bound =
                |class| MatrixBound { matrix_type: concrete.clone(), coefficient_class: class };
            let large_matrix = |key| SymbolicFactor {
                key,
                bound: BoundClass::Large,
                relation_live: false,
                trapdoor: None,
                matrix_bound: Some(bound(BoundClass::Large)),
                matrix_type: concrete.clone(),
                polynomial_facts: super::super::bound::PolynomialFacts::conservative(
                    concrete.ring_dimension,
                ),
                matrix_value_metadata: MatrixMetadata::unknown(),
                switch: None,
            };
            let initial = lowerer
                .dag
                .push(ExpressionNode::Atom(
                    SymbolicFactor::bounded(
                        FactorIdentity::named("initial"),
                        bound(BoundClass::bounded(1_u8.into())),
                    )
                    .unwrap(),
                ))
                .unwrap();
            let state_key = FactorIdentity::named("state");
            let state =
                lowerer.dag.push(ExpressionNode::Atom(large_matrix(state_key.clone()))).unwrap();
            let public = FactorIdentity::named("B");
            let preimage = FactorIdentity::named("K");
            let target = FactorIdentity::named("P");
            let public_term =
                lowerer.dag.push(ExpressionNode::Atom(large_matrix(public.clone()))).unwrap();
            let preimage_term = lowerer
                .dag
                .push(ExpressionNode::Atom(
                    SymbolicFactor::relation_live(
                        preimage.clone(),
                        bound(BoundClass::bounded(1_u8.into())),
                    )
                    .unwrap(),
                ))
                .unwrap();
            let product = lowerer
                .dag
                .push(ExpressionNode::Product(vec![public_term, preimage_term].into()))
                .unwrap();
            let transition =
                lowerer.dag.push(ExpressionNode::Add(vec![state, product].into())).unwrap();
            let target_term = lowerer
                .dag
                .push(ExpressionNode::Atom(SymbolicFactor::large_with_metadata(
                    target.clone(),
                    bound(BoundClass::Large),
                    MatrixMetadata {
                        canonical_coefficient_exclusive_upper: None,
                        is_constant_polynomial: true,
                        known_zero_rows: None,
                        polynomial: Some(
                            super::super::bound::PolynomialFacts::new(1, concrete.ring_dimension)
                                .unwrap(),
                        ),
                    },
                )))
                .unwrap();
            lowerer
                .relation_registry
                .register(super::super::normal_form::RelationRegistration {
                    pattern: super::super::normal_form::RelationPattern::new(
                        [public.clone(), preimage.clone()],
                        [],
                    ),
                    key: super::super::normal_form::FullRelationKey {
                        source: "named".into(),
                        ordered_indices: Box::new([]),
                        public: public.clone(),
                        target: target.clone(),
                        matrix_type: Some(concrete.clone()),
                        layout: None,
                        trapdoor: None,
                        selector: None,
                    },
                    preimage,
                    target: target_term,
                })
                .unwrap();
            let wire = LoweringWire {
                source: WireSourceKey {
                    scope: environment.occurrence.clone(),
                    wire: WireRef { node: mxx_ir_core::NodeId(99), port: mxx_ir_core::Port(0) },
                },
                indices: Box::new([]),
            };
            let output_type = WireType::Matrix(MatrixType {
                modulus: IntExpr::constant(17),
                ring_dimension: IntExpr::constant(1),
                rows: IntExpr::constant(1),
                columns: IntExpr::constant(1),
            });
            let LoweredValue::Matrix(result) = lowerer
                .finish_sequential_matrix_loop(
                    &wire,
                    &environment,
                    ResolvedIntExpr::Const(count.into()),
                    vec![initial],
                    vec![state_key],
                    None,
                    vec![transition],
                    vec![matrix],
                    output_type,
                    0,
                )
                .unwrap()
            else {
                panic!("matrix recurrence must return a normal-form term")
            };
            match count {
                0 => assert_eq!(result, initial),
                1 => assert_eq!(
                    lowerer
                        .dag
                        .normalize(result, &lowerer.relation_registry)
                        .unwrap()
                        .first_large_witness()
                        .unwrap()
                        .identity,
                    target,
                ),
                _ => assert_ne!(result, initial),
            }
        }
    }

    #[test]
    fn matrix_sequential_recurrence_checks_each_carried_shape_independently() {
        let (protocol, request) = recurrence_lowerer();
        let matrix_a = test_resolved_matrix();
        let matrix_b = super::super::identity::ResolvedMatrixType {
            modulus: ResolvedIntExpr::Const(17.into()),
            ring_dimension: ResolvedIntExpr::Const(1.into()),
            rows: ResolvedIntExpr::Const(2.into()),
            columns: ResolvedIntExpr::Const(1.into()),
        };
        let output_type = |matrix: &super::super::identity::ResolvedMatrixType| {
            WireType::Matrix(MatrixType {
                modulus: IntExpr::constant(17),
                ring_dimension: IntExpr::constant(1),
                rows: IntExpr::constant(if matrix.rows == ResolvedIntExpr::Const(2.into()) {
                    2
                } else {
                    1
                }),
                columns: IntExpr::constant(1),
            })
        };
        let wire = LoweringWire {
            source: WireSourceKey {
                scope: root_test_environment().occurrence.clone(),
                wire: WireRef { node: mxx_ir_core::NodeId(199), port: mxx_ir_core::Port(0) },
            },
            indices: Box::new([]),
        };
        let build = |lowerer: &mut GraphLowerer<'_, '_>, matrix, name| {
            let concrete = concrete_matrix_type(matrix).unwrap();
            lowerer
                .dag
                .push(ExpressionNode::Atom(SymbolicFactor {
                    key: FactorIdentity::named(name),
                    bound: BoundClass::Large,
                    relation_live: false,
                    trapdoor: None,
                    matrix_bound: Some(MatrixBound {
                        matrix_type: concrete.clone(),
                        coefficient_class: BoundClass::Large,
                    }),
                    matrix_type: concrete.clone(),
                    polynomial_facts: super::super::bound::PolynomialFacts::conservative(
                        concrete.ring_dimension,
                    ),
                    matrix_value_metadata: MatrixMetadata::unknown(),
                    switch: None,
                }))
                .unwrap()
        };
        let mut valid = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let initial_a = build(&mut valid, &matrix_a, "initial-a");
        let initial_b = build(&mut valid, &matrix_b, "initial-b");
        let transition_a = build(&mut valid, &matrix_a, "transition-a");
        let transition_b = build(&mut valid, &matrix_b, "transition-b");
        assert!(matches!(
            valid
                .finish_sequential_matrix_loop(
                    &wire,
                    &root_test_environment(),
                    ResolvedIntExpr::Const(1.into()),
                    vec![initial_a, initial_b],
                    vec![FactorIdentity::named("state-a"), FactorIdentity::named("state-b")],
                    None,
                    vec![transition_a, transition_b],
                    vec![matrix_a.clone(), matrix_b.clone()],
                    output_type(&matrix_a),
                    0,
                )
                .unwrap(),
            LoweredValue::Matrix(_)
        ));

        let mut swapped = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let initial_a = build(&mut swapped, &matrix_a, "initial-a");
        let initial_b = build(&mut swapped, &matrix_b, "initial-b");
        let transition_a = build(&mut swapped, &matrix_a, "transition-a");
        let transition_b = build(&mut swapped, &matrix_b, "transition-b");
        assert!(matches!(
            swapped.finish_sequential_matrix_loop(
                &wire,
                &root_test_environment(),
                ResolvedIntExpr::Const(1.into()),
                vec![initial_a, initial_b],
                vec![FactorIdentity::named("state-a"), FactorIdentity::named("state-b")],
                None,
                vec![transition_b, transition_a],
                vec![matrix_a, matrix_b],
                output_type(&test_resolved_matrix()),
                0,
            ),
            Err(LowerError::UnsupportedMatrixProductExpansion)
        ));
    }

    #[test]
    fn runtime_switch_direct_select_matches_the_shared_family_constructor() {
        let protocol = crate::toy_example::protocol();
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "direct-select-switch".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let selector = test_integer_atom(&mut lowerer, "direct-select-range", 0, 1);
        let cases = [10, 20, 30].map(|value| test_int(&mut lowerer, value));
        let environment = root_test_environment();
        let wire = LoweringWire {
            source: WireSourceKey {
                scope: environment.occurrence.clone(),
                wire: WireRef { node: mxx_ir_core::NodeId(1), port: mxx_ir_core::Port(0) },
            },
            indices: Box::new([]),
        };
        let arguments =
            std::iter::once(selector).chain(cases).map(LoweredValue::Scalar).collect::<Vec<_>>();
        let LoweredValue::Scalar(direct) = lowerer
            .lower_structural_node(
                &wire,
                &NodeKind::Select { count: IntExpr::constant(3) },
                &arguments,
                &environment,
                WireType::Int,
            )
            .expect("direct Select lowers")
        else {
            panic!("direct Select is an integer term")
        };
        let shared = lowerer
            .scalar_dynamic_get(
                &FamilyLoweringValue {
                    element_type: ScalarSort::Int,
                    storage: FamilyCoverageStorage::ExactStored { elements: cases.into() },
                },
                selector,
            )
            .expect("typed runtime switch");
        assert_eq!(direct, shared);
    }

    #[test]
    fn shared_matrix_family_accepts_bounded_runtime_selector_without_lanes() {
        let (protocol, request) = recurrence_lowerer();
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let owner = super::super::identity::BinderKey {
            loop_scope: root_test_environment().occurrence.clone(),
            loop_node: mxx_ir_core::NodeId(301),
            slot: 0,
        };
        let matrix = test_resolved_matrix();
        let concrete = concrete_matrix_type(&matrix).unwrap();
        let mut key = FactorIdentity::named("shared-runtime");
        key.coordinates = vec![(owner.clone(), ResolvedIntExpr::Binder(owner.clone()))].into();
        let representative = lowerer
            .dag
            .push(ExpressionNode::Atom(SymbolicFactor {
                key,
                bound: BoundClass::Large,
                relation_live: false,
                trapdoor: None,
                matrix_bound: Some(MatrixBound {
                    matrix_type: concrete.clone(),
                    coefficient_class: BoundClass::Large,
                }),
                matrix_type: concrete.clone(),
                polynomial_facts: super::super::bound::PolynomialFacts::conservative(
                    concrete.ring_dimension,
                ),
                matrix_value_metadata: MatrixMetadata::unknown(),
                switch: None,
            }))
            .unwrap();
        let family = FamilyLoweringValue {
            element_type: ScalarSort::Matrix(matrix),
            storage: FamilyCoverageStorage::SharedTemplate {
                domain: family::LoopDomainKey {
                    binder: owner.clone(),
                    logical_count: 30_720_u64.into(),
                },
                representative,
                binder_domains: vec![family::CoverageBinderDomain {
                    binder: owner,
                    minimum: 0.into(),
                    maximum: 30_719.into(),
                }]
                .into(),
            },
        };
        let selector = test_integer_atom(&mut lowerer, "bounded-runtime-selector", 0, 30_719);
        let selector = LoweredInt { scalar: selector };
        let before = lowerer.dag.term_count();
        let LoweredValue::Matrix(instantiated) = lowerer
            .shared_matrix_family_element(&family, &selector)
            .expect("runtime selector binds the shared representative")
        else {
            unreachable!()
        };
        assert_ne!(instantiated, representative);
        assert!(lowerer.dag.term_count() <= before + 1);
        let one_family = FamilyLoweringValue {
            element_type: family.element_type.clone(),
            storage: FamilyCoverageStorage::SharedTemplate {
                domain: family::LoopDomainKey {
                    binder: match &family.storage {
                        FamilyCoverageStorage::SharedTemplate { domain, .. } => {
                            domain.binder.clone()
                        }
                        FamilyCoverageStorage::ExactStored { .. } => unreachable!(),
                    },
                    logical_count: 1_u8.into(),
                },
                representative,
                binder_domains: vec![family::CoverageBinderDomain {
                    binder: match &family.storage {
                        FamilyCoverageStorage::SharedTemplate { domain, .. } => {
                            domain.binder.clone()
                        }
                        FamilyCoverageStorage::ExactStored { .. } => unreachable!(),
                    },
                    minimum: 0.into(),
                    maximum: 0.into(),
                }]
                .into(),
            },
        };
        assert!(matches!(
            lowerer.normalize_singleton_for_input(
                LoweredValue::MatrixFamily(one_family),
                &WireType::Matrix(MatrixType {
                    modulus: IntExpr::constant(17),
                    ring_dimension: IntExpr::constant(1),
                    rows: IntExpr::constant(1),
                    columns: IntExpr::constant(1),
                }),
            ),
            Ok(LoweredValue::Matrix(_))
        ));
    }

    #[test]
    fn exact_matrix_family_dynamic_checks_upper_and_keeps_reachable_mapping() {
        let (protocol, request) = recurrence_lowerer();
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let matrix = test_resolved_matrix();
        let concrete = concrete_matrix_type(&matrix).unwrap();
        let term = |lowerer: &mut GraphLowerer<'_, '_>, name| {
            lowerer
                .dag
                .push(ExpressionNode::Atom(SymbolicFactor {
                    key: FactorIdentity::named(name),
                    bound: BoundClass::Large,
                    relation_live: false,
                    trapdoor: None,
                    matrix_bound: Some(MatrixBound {
                        matrix_type: concrete.clone(),
                        coefficient_class: BoundClass::Large,
                    }),
                    matrix_type: concrete.clone(),
                    polynomial_facts: super::super::bound::PolynomialFacts::conservative(
                        concrete.ring_dimension,
                    ),
                    matrix_value_metadata: MatrixMetadata::unknown(),
                    switch: None,
                }))
                .unwrap()
        };
        let stored0 = term(&mut lowerer, "stored-0");
        let stored1 = term(&mut lowerer, "stored-1");
        let family = FamilyLoweringValue {
            element_type: ScalarSort::Matrix(matrix),
            storage: FamilyCoverageStorage::ExactStored { elements: vec![stored0, stored1].into() },
        };
        let environment = root_test_environment();
        let wire = LoweringWire {
            source: WireSourceKey {
                scope: environment.occurrence.clone(),
                wire: WireRef { node: mxx_ir_core::NodeId(302), port: mxx_ir_core::Port(0) },
            },
            indices: Box::new([]),
        };
        let reachable = test_integer_atom(&mut lowerer, "reachable", 0, 0);
        let reachable = LoweredInt { scalar: reachable };
        let LoweredValue::Matrix(dynamic) =
            lowerer.matrix_family_element(&family, &reachable, &wire, &environment).unwrap()
        else {
            unreachable!()
        };
        let ExpressionNode::FamilyGetDynamic { cases, stored_indices, domain_upper, .. } =
            lowerer.dag.node(dynamic).unwrap()
        else {
            panic!("runtime exact access must remain a compact dynamic barrier")
        };
        assert_eq!(cases.len(), 1);
        assert_eq!(stored_indices.as_ref(), &[BigUint::from(0_u8)]);
        assert_eq!(*domain_upper, BigUint::from(1_u8));
        let singleton = FamilyLoweringValue {
            element_type: family.element_type.clone(),
            storage: FamilyCoverageStorage::ExactStored { elements: vec![stored0].into() },
        };
        assert!(matches!(
            lowerer.normalize_singleton_for_input(
                LoweredValue::MatrixFamily(singleton),
                &WireType::Matrix(MatrixType {
                    modulus: IntExpr::constant(17),
                    ring_dimension: IntExpr::constant(1),
                    rows: IntExpr::constant(1),
                    columns: IntExpr::constant(1),
                }),
            ),
            Ok(LoweredValue::Matrix(_))
        ));

        let invalid = test_integer_atom(&mut lowerer, "invalid-upper", 0, 2);
        let invalid = LoweredInt { scalar: invalid };
        assert!(matches!(
            lowerer.matrix_family_element(&family, &invalid, &wire, &environment),
            Err(LowerError::FamilyAccessOutOfRange { .. })
        ));
    }

    #[test]
    fn resolved_integer_requires_closed_exact_operations() {
        let constant = |value| Box::new(ResolvedIntExpr::Const(BigInt::from(value)));
        assert_eq!(
            resolved_integer(&ResolvedIntExpr::Div(constant(12), constant(3))),
            Some(BigInt::from(4))
        );
        assert_eq!(
            resolved_integer(&ResolvedIntExpr::RoundDiv(constant(-3), constant(2))),
            Some(BigInt::from(-1))
        );
        assert_eq!(
            resolved_integer(&ResolvedIntExpr::Log2Ceil(constant(9))),
            Some(BigInt::from(4))
        );
        assert!(resolved_integer(&ResolvedIntExpr::Div(constant(1), constant(0))).is_none());
        assert!(resolved_integer(&ResolvedIntExpr::Div(constant(5), constant(2))).is_none());
        assert!(resolved_integer(&ResolvedIntExpr::Log2Ceil(constant(0))).is_none());
        assert!(resolved_integer(&ResolvedIntExpr::Parameter("unresolved".to_owned())).is_none());
    }

    #[test]
    fn scalar_consumer_validation_uses_the_consumed_boolean_sort() {
        let protocol = crate::toy_example::protocol();
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "scalar-consumer".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let boolean = lowerer
            .scalar_store
            .intern_node(ScalarNode::BoolConst(true), &lowerer.symbols)
            .unwrap();
        lowerer
            .validate_boolean_consumer(boolean, SelectorOnlyConsumer::BoolToInt, false)
            .expect("an ordinary boolean is a valid BoolToInt operand");

        let integer = test_int(&mut lowerer, 1);
        assert_eq!(
            lowerer.validate_boolean_consumer(integer, SelectorOnlyConsumer::BoolToInt, false,),
            Err(LowerError::InvalidOperandSort { expected: WireType::Bool, actual: WireType::Int })
        );
    }

    #[test]
    fn matrix_operations_have_only_dag_lowered_values() {
        let protocol = crate::toy_example::protocol();
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "matrix-dag-only".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let matrix_type = mxx_ir_core::types::ConcreteMatrixType {
            modulus: 17.into(),
            ring_dimension: 1,
            rows: 1,
            columns: 1,
        };
        let matrix = lowerer
            .dag
            .push(ExpressionNode::Atom(SymbolicFactor {
                key: FactorIdentity::named("matrix-dag-only"),
                bound: BoundClass::bounded(1_u8.into()),
                relation_live: false,
                trapdoor: None,
                matrix_bound: Some(MatrixBound {
                    matrix_type: matrix_type.clone(),
                    coefficient_class: BoundClass::bounded(1_u8.into()),
                }),
                matrix_type: matrix_type.clone(),
                polynomial_facts: super::super::bound::PolynomialFacts::conservative(
                    matrix_type.ring_dimension,
                ),
                matrix_value_metadata: MatrixMetadata::unknown(),
                switch: None,
            }))
            .expect("matrix atom");
        let scalar = IntExpr::constant(1);
        for (kind, args) in [
            (
                NodeKind::MatrixBinary(MatrixBinaryOp::Add),
                vec![LoweredValue::Matrix(matrix), LoweredValue::Matrix(matrix)],
            ),
            (NodeKind::MatrixNegate, vec![LoweredValue::Matrix(matrix)]),
            (NodeKind::Transpose, vec![LoweredValue::Matrix(matrix)]),
            (NodeKind::Tensor, vec![LoweredValue::Matrix(matrix), LoweredValue::Matrix(matrix)]),
            (NodeKind::MatrixScale { scalar: scalar.clone() }, vec![LoweredValue::Matrix(matrix)]),
        ] {
            let value = lowerer
                .lower_node(&kind, &args, &root_test_environment())
                .expect("matrix operation lowers through DAG");
            assert!(matches!(value, LoweredValue::Matrix(_)));
        }
        assert!(lowerer.scalar_store.len() >= 1, "matrix DAG keeps scalar constants typed");
    }

    #[test]
    fn runtime_interval_matrix_scale_emits_a_ring_typed_scalar_factor() {
        let protocol = crate::toy_example::protocol();
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "runtime-interval-scale".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let mut environment = root_test_environment();
        let binder = BinderKey {
            loop_scope: environment.occurrence.clone(),
            loop_node: mxx_ir_core::NodeId(777),
            slot: 0,
        };
        let source = test_integer_atom(&mut lowerer, "runtime-scale", -3, 5);
        environment.binders.push((binder.clone(), LoweredInt { scalar: source }));
        let matrix = |lowerer: &mut GraphLowerer<'_, '_>, modulus: i64| {
            let matrix_type = mxx_ir_core::types::ConcreteMatrixType {
                modulus: modulus.into(),
                ring_dimension: 1,
                rows: 2,
                columns: 2,
            };
            lowerer
                .dag
                .push(ExpressionNode::Atom(SymbolicFactor::large_typed(
                    FactorIdentity::named(format!("runtime-scale-{modulus}").as_str()),
                    matrix_type,
                )))
                .unwrap()
        };
        let extract_scalar = |lowerer: &GraphLowerer<'_, '_>, term: TermId| {
            let ExpressionNode::Product(children) = lowerer.dag.node(term).unwrap() else {
                panic!("interval scale must lower to an ordinary product")
            };
            let ExpressionNode::Atom(factor) = lowerer.dag.node(children[0]).unwrap() else {
                panic!("ordinary product must expose its scalar atom first")
            };
            factor.clone()
        };
        let first_matrix = matrix(&mut lowerer, 17);
        let LoweredValue::Matrix(first_term) = lowerer
            .lower_node(
                &NodeKind::MatrixScale { scalar: IntExpr::LoopIndex(0) },
                &[LoweredValue::Matrix(first_matrix)],
                &environment,
            )
            .unwrap()
        else {
            panic!("interval scale must lower to a matrix term")
        };
        let first_scalar = extract_scalar(&lowerer, first_term);
        assert!(first_scalar.is_central_scalar());
        assert_eq!(first_scalar.matrix_type.rows, 1);
        assert_eq!(first_scalar.matrix_type.columns, 1);
        assert_eq!(first_scalar.bound, BoundClass::bounded(5_u8.into()));
        assert_eq!(first_scalar.polynomial_facts.support_upper, 1);
        assert!(first_scalar.matrix_value_metadata.is_constant_polynomial);

        let second_matrix = matrix(&mut lowerer, 19);
        let LoweredValue::Matrix(second_term) = lowerer
            .lower_node(
                &NodeKind::MatrixScale { scalar: IntExpr::LoopIndex(0) },
                &[LoweredValue::Matrix(second_matrix)],
                &environment,
            )
            .unwrap()
        else {
            panic!("interval scale must lower to a matrix term")
        };
        let second_scalar = extract_scalar(&lowerer, second_term);
        assert_ne!(first_scalar.key, second_scalar.key);
        assert!(matches!(
            first_scalar.key.owner,
            FactorOwner::RuntimeScalar { ref matrix_type, .. }
                if matrix_type.modulus == 17.into()
        ));
    }

    #[test]
    fn coefficient_extraction_bridges_matrix_dag_to_authoritative_scalar_facts() {
        let protocol = crate::toy_example::protocol();
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "matrix-coefficient-bridge".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let matrix_type = mxx_ir_core::types::ConcreteMatrixType {
            modulus: 17.into(),
            ring_dimension: 1,
            rows: 1,
            columns: 1,
        };
        let matrix = lowerer
            .dag
            .push(ExpressionNode::Atom(SymbolicFactor {
                key: FactorIdentity::atomic(
                    super::super::identity::AtomicSourceKey::ProtocolInput(crate::ProtocolInputId(
                        "coefficient-source".to_owned(),
                    )),
                    [],
                ),
                bound: BoundClass::bounded(6_u8.into()),
                relation_live: false,
                trapdoor: None,
                matrix_bound: Some(MatrixBound {
                    matrix_type: matrix_type.clone(),
                    coefficient_class: BoundClass::bounded(6_u8.into()),
                }),
                matrix_type: matrix_type.clone(),
                polynomial_facts: super::super::bound::PolynomialFacts::conservative(
                    matrix_type.ring_dimension,
                ),
                matrix_value_metadata: MatrixMetadata {
                    canonical_coefficient_exclusive_upper: Some(7_u8.into()),
                    is_constant_polynomial: false,
                    known_zero_rows: None,
                    polynomial: None,
                },
                switch: None,
            }))
            .unwrap();
        let environment = root_test_environment();
        let value = lowerer
            .lower_node(
                &NodeKind::ExtractCoefficient {
                    position: IntExpr::constant(0),
                    canonical_input_exclusive_upper: None,
                },
                &[LoweredValue::Matrix(matrix)],
                &environment,
            )
            .unwrap();
        let LoweredValue::Scalar(term) = value else { panic!("coefficient must be scalar") };
        let entry = lowerer.scalar_store.get(term).unwrap();
        assert_eq!(
            entry.analysis.integer_domain.as_ref().unwrap().interval().unwrap().maximum,
            BigInt::from(6)
        );
        assert_eq!(
            entry.analysis.scalar_provenance,
            Some(super::super::scalar::ScalarProvenance::SelectorOnly)
        );
        assert_eq!(
            entry.analysis.direct_extract,
            Some(super::super::scalar::DirectExtractFact { canonical_upper: Some(7_u8.into()) })
        );

        let fallback_matrix = lowerer
            .dag
            .push(ExpressionNode::Atom(SymbolicFactor {
                key: FactorIdentity::atomic(
                    super::super::identity::AtomicSourceKey::ProtocolInput(crate::ProtocolInputId(
                        "fallback-source".to_owned(),
                    )),
                    [],
                ),
                bound: BoundClass::Large,
                relation_live: false,
                trapdoor: None,
                matrix_bound: Some(MatrixBound {
                    matrix_type: matrix_type.clone(),
                    coefficient_class: BoundClass::Large,
                }),
                matrix_type: matrix_type.clone(),
                polynomial_facts: super::super::bound::PolynomialFacts::conservative(
                    matrix_type.ring_dimension,
                ),
                matrix_value_metadata: MatrixMetadata::unknown(),
                switch: None,
            }))
            .unwrap();
        let LoweredValue::Scalar(fallback) = lowerer
            .lower_node(
                &NodeKind::ExtractCoefficient {
                    position: IntExpr::constant(0),
                    canonical_input_exclusive_upper: Some(5_u8.into()),
                },
                &[LoweredValue::Matrix(fallback_matrix)],
                &environment,
            )
            .unwrap()
        else {
            panic!("coefficient must be scalar")
        };
        assert_eq!(
            lowerer
                .scalar_store
                .get(fallback)
                .unwrap()
                .analysis
                .integer_domain
                .as_ref()
                .unwrap()
                .interval()
                .unwrap()
                .maximum,
            BigInt::from(4)
        );

        let LoweredValue::Scalar(full_modulus) = lowerer
            .lower_node(
                &NodeKind::ExtractCoefficient {
                    position: IntExpr::constant(0),
                    canonical_input_exclusive_upper: None,
                },
                &[LoweredValue::Matrix(fallback_matrix)],
                &environment,
            )
            .unwrap()
        else {
            panic!("coefficient must be scalar")
        };
        assert_eq!(
            lowerer
                .scalar_store
                .get(full_modulus)
                .unwrap()
                .analysis
                .integer_domain
                .as_ref()
                .unwrap()
                .interval()
                .unwrap()
                .maximum,
            BigInt::from(16)
        );
        assert_eq!(fallback, full_modulus);
        assert!(matches!(
            lowerer.lower_node(
                &NodeKind::ExtractCoefficient {
                    position: IntExpr::constant(0),
                    canonical_input_exclusive_upper: Some(18_u8.into()),
                },
                &[LoweredValue::Matrix(fallback_matrix)],
                &environment,
            ),
            Err(LowerError::InvalidExtractCoefficientCanonicalUpper { .. })
        ));
    }

    #[test]
    fn coefficient_extraction_uses_shared_structural_facts_for_all_matrix_nodes() {
        let protocol = crate::toy_example::protocol();
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "matrix-coefficient-structural-facts".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let environment = root_test_environment();
        let matrix_type = mxx_ir_core::types::ConcreteMatrixType {
            modulus: 17.into(),
            ring_dimension: 1,
            rows: 2,
            columns: 2,
        };
        let source = FactorIdentity::atomic(
            super::super::identity::AtomicSourceKey::ProtocolInput(crate::ProtocolInputId(
                "shared-structural-source".to_owned(),
            )),
            [],
        );
        let atom = lowerer
            .dag
            .push(ExpressionNode::Atom(SymbolicFactor {
                key: source.clone(),
                bound: BoundClass::bounded(6_u8.into()),
                relation_live: false,
                trapdoor: None,
                matrix_bound: Some(MatrixBound {
                    matrix_type: matrix_type.clone(),
                    coefficient_class: BoundClass::bounded(6_u8.into()),
                }),
                matrix_type: matrix_type.clone(),
                polynomial_facts: super::super::bound::PolynomialFacts::conservative(
                    matrix_type.ring_dimension,
                ),
                matrix_value_metadata: MatrixMetadata {
                    canonical_coefficient_exclusive_upper: Some(7_u8.into()),
                    is_constant_polynomial: false,
                    known_zero_rows: None,
                    polynomial: None,
                },
                switch: None,
            }))
            .unwrap();
        let transpose = lowerer.dag.push(ExpressionNode::Transpose(atom)).unwrap();
        let slice = lowerer
            .dag
            .push(ExpressionNode::Slice {
                input: transpose,
                spec: super::super::identity::SliceSpec {
                    rows: Some(super::super::identity::ResolvedIndexRange {
                        start: ResolvedIntExpr::Const(0.into()),
                        end: ResolvedIntExpr::Const(1.into()),
                    }),
                    columns: None,
                },
            })
            .unwrap();
        let view = lowerer
            .dag
            .push(ExpressionNode::View {
                input: slice,
                view: ViewSpec::Identity,
                output_type: mxx_ir_core::types::ConcreteMatrixType {
                    modulus: 17.into(),
                    ring_dimension: 1,
                    rows: 1,
                    columns: 2,
                },
            })
            .unwrap();
        let add =
            lowerer.dag.push(ExpressionNode::Add(vec![atom, view].into_boxed_slice())).unwrap();
        let product =
            lowerer.dag.push(ExpressionNode::Product(vec![atom, atom].into_boxed_slice())).unwrap();
        let static_family = lowerer
            .dag
            .push(ExpressionNode::FamilyGetStatic { cases: vec![atom, view].into(), index: 1 })
            .unwrap();
        let dynamic_family = lowerer
            .dag
            .push(ExpressionNode::FamilyGetDynamic {
                selector: FactorIdentity::scalar_selector(ResolvedIntExpr::Parameter("j".into())),
                cases: vec![atom, view].into(),
                stored_indices: vec![0_u8.into(), 1_u8.into()].into(),
                domain_upper: 2_u8.into(),
            })
            .unwrap();
        let select = lowerer
            .dag
            .push(ExpressionNode::Select {
                selector: FactorIdentity::scalar_selector(ResolvedIntExpr::Parameter("s".into())),
                cases: vec![atom, view].into(),
                reachable: vec![0, 1].into(),
            })
            .unwrap();
        let position = lowerer.lower_int_expr(&IntExpr::constant(0), &environment).unwrap();
        for term in [transpose, slice, view, static_family, dynamic_family, select] {
            let LoweredValue::Scalar(id) = lowerer
                .lower_extract_coefficient_dag(term, position.clone(), None, &environment)
                .unwrap()
            else {
                panic!("derived matrix extraction must produce a typed scalar")
            };
            assert_eq!(
                lowerer
                    .scalar_store
                    .get(id)
                    .unwrap()
                    .analysis
                    .integer_domain
                    .as_ref()
                    .unwrap()
                    .interval()
                    .unwrap()
                    .maximum,
                BigInt::from(6)
            );
        }
        for term in [add, product] {
            let LoweredValue::Scalar(id) = lowerer
                .lower_extract_coefficient_dag(term, position.clone(), None, &environment)
                .unwrap()
            else {
                panic!("arithmetic extraction must fall back to a typed scalar")
            };
            assert_eq!(
                lowerer
                    .scalar_store
                    .get(id)
                    .unwrap()
                    .analysis
                    .integer_domain
                    .as_ref()
                    .unwrap()
                    .interval()
                    .unwrap()
                    .maximum,
                BigInt::from(16)
            );
        }
        let mut deep = atom;
        for _ in 0..256 {
            deep = lowerer.dag.push(ExpressionNode::Transpose(deep)).unwrap();
        }
        assert!(lowerer.lower_extract_coefficient_dag(deep, position, None, &environment).is_ok());

        let same_transpose = lowerer.dag.push(ExpressionNode::Transpose(atom)).unwrap();
        assert_eq!(
            lowerer.dag.facts(transpose).unwrap().identity,
            lowerer.dag.facts(same_transpose).unwrap().identity
        );
        assert!(lowerer.scalar_store.len() >= 1);
    }

    #[test]
    fn gadget_decompose_dag_path_validates_base_digits_rows_and_input_category() {
        let protocol = crate::toy_example::protocol();
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "gadget-validation".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let matrix = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(3),
            columns: IntExpr::constant(2),
        };
        let matrix_term = lowerer
            .dag
            .push(ExpressionNode::Atom(SymbolicFactor {
                key: FactorIdentity::named("gadget-input"),
                bound: BoundClass::Large,
                relation_live: false,
                trapdoor: None,
                matrix_bound: Some(MatrixBound {
                    matrix_type: mxx_ir_core::types::ConcreteMatrixType {
                        modulus: 17.into(),
                        ring_dimension: 1,
                        rows: 3,
                        columns: 2,
                    },
                    coefficient_class: BoundClass::Large,
                }),
                matrix_type: mxx_ir_core::types::ConcreteMatrixType {
                    modulus: 17.into(),
                    ring_dimension: 1,
                    rows: 3,
                    columns: 2,
                },
                polynomial_facts: super::super::bound::PolynomialFacts::conservative(1),
                matrix_value_metadata: MatrixMetadata::unknown(),
                switch: None,
            }))
            .unwrap();
        let valid = LoweredValue::Matrix(matrix_term);
        let environment = root_test_environment();
        assert!(
            lowerer
                .validate_gadget_decompose(
                    &IntExpr::constant(4),
                    &IntExpr::constant(3),
                    &environment,
                    &WireType::Matrix(matrix.clone()),
                    &valid,
                )
                .is_ok()
        );
        for (base, digits, output_rows) in [
            (1, 3, 3), // non-positive/unit base
            (4, 0, 3), // zero digit count
            (4, 2, 3), // non-divisible rows
        ] {
            let mut malformed = matrix.clone();
            malformed.rows = IntExpr::constant(output_rows);
            assert!(
                lowerer
                    .validate_gadget_decompose(
                        &IntExpr::constant(base),
                        &IntExpr::constant(digits),
                        &environment,
                        &WireType::Matrix(malformed),
                        &valid,
                    )
                    .is_err()
            );
        }
        let scalar = LoweredValue::Scalar(test_int(&mut lowerer, 1));
        assert_eq!(
            lowerer.validate_gadget_decompose(
                &IntExpr::constant(4),
                &IntExpr::constant(3),
                &environment,
                &WireType::Matrix(matrix),
                &scalar,
            ),
            Err(LowerError::UnsupportedMatrixProductExpansion)
        );
    }

    #[test]
    fn matrix_family_metadata_is_rejected_at_scalar_boundaries() {
        let protocol = crate::toy_example::protocol();
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "matrix-term-rejection".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let matrix = super::super::identity::ResolvedMatrixType {
            modulus: ResolvedIntExpr::Const(17.into()),
            ring_dimension: ResolvedIntExpr::Const(1.into()),
            rows: ResolvedIntExpr::Const(1.into()),
            columns: ResolvedIntExpr::Const(1.into()),
        };
        let scalar = test_int(&mut lowerer, 1);
        assert!(matches!(
            lowerer.lower_node(
                &NodeKind::IntBinary(IntBinaryOp::Add),
                &[LoweredValue::Scalar(scalar), LoweredValue::Scalar(scalar)],
                &root_test_environment(),
            ),
            Ok(LoweredValue::Scalar(_))
        ));

        let family = FamilyLoweringValue {
            element_type: ScalarSort::Matrix(matrix),
            storage: FamilyCoverageStorage::ExactStored {
                elements: vec![scalar].into_boxed_slice(),
            },
        };
        let wire = LoweringWire {
            source: WireSourceKey {
                scope: root_test_environment().occurrence,
                wire: WireRef { node: mxx_ir_core::NodeId(7), port: mxx_ir_core::Port(0) },
            },
            indices: Box::new([]),
        };
        assert!(matches!(
            lowerer.lower_structural_node(
                &wire,
                &NodeKind::PackPolynomialCoefficients {
                    matrix_type: MatrixType {
                        modulus: IntExpr::constant(17),
                        ring_dimension: IntExpr::constant(1),
                        rows: IntExpr::constant(1),
                        columns: IntExpr::constant(1),
                    },
                    coefficient_bits: IntExpr::constant(1),
                },
                &[LoweredValue::Family(family)],
                &root_test_environment(),
                WireType::Matrix(MatrixType {
                    modulus: IntExpr::constant(17),
                    ring_dimension: IntExpr::constant(1),
                    rows: IntExpr::constant(1),
                    columns: IntExpr::constant(1),
                }),
            ),
            Err(LowerError::PackRequiresExplicitBooleanFamily { .. })
        ));
    }

    #[test]
    fn matrix_normal_form_is_independent_of_unrelated_scalar_insertions() {
        let protocol = crate::toy_example::protocol();
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "matrix-nf-determinism".to_owned(),
        };
        let build = |lowerer: &mut GraphLowerer<'_, '_>| {
            let matrix = mxx_ir_core::types::ConcreteMatrixType {
                modulus: 17.into(),
                ring_dimension: 1,
                rows: 1,
                columns: 1,
            };
            let left = lowerer
                .dag
                .push(ExpressionNode::Atom(SymbolicFactor {
                    key: FactorIdentity::named("nf-left"),
                    bound: BoundClass::Large,
                    relation_live: false,
                    trapdoor: None,
                    matrix_bound: Some(MatrixBound {
                        matrix_type: matrix.clone(),
                        coefficient_class: BoundClass::Large,
                    }),
                    matrix_type: matrix.clone(),
                    polynomial_facts: super::super::bound::PolynomialFacts::conservative(
                        matrix.ring_dimension,
                    ),
                    matrix_value_metadata: MatrixMetadata::unknown(),
                    switch: None,
                }))
                .unwrap();
            let right = lowerer
                .dag
                .push(ExpressionNode::Atom(SymbolicFactor {
                    key: FactorIdentity::named("nf-right"),
                    bound: BoundClass::Large,
                    relation_live: false,
                    trapdoor: None,
                    matrix_bound: Some(MatrixBound {
                        matrix_type: matrix.clone(),
                        coefficient_class: BoundClass::Large,
                    }),
                    matrix_type: matrix,
                    polynomial_facts: super::super::bound::PolynomialFacts::conservative(1),
                    matrix_value_metadata: MatrixMetadata::unknown(),
                    switch: None,
                }))
                .unwrap();
            lowerer.dag.push(ExpressionNode::Product(vec![left, right].into_boxed_slice())).unwrap()
        };
        let mut plain = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let plain_root = build(&mut plain);
        let mut polluted = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let first = test_int(&mut polluted, 9);
        let second = test_int(&mut polluted, 11);
        let _ = polluted
            .scalar_store
            .intern_node(ScalarNode::IntAdd([first, second]), &polluted.symbols);
        let polluted_root = build(&mut polluted);
        let plain_nf = plain.dag.normalize(plain_root, &plain.relation_registry).unwrap();
        let polluted_nf =
            polluted.dag.normalize(polluted_root, &polluted.relation_registry).unwrap();
        assert_eq!(plain_nf.exact_terms(), polluted_nf.exact_terms());
        assert_eq!(plain_nf.bounded_summary(), polluted_nf.bounded_summary());
    }

    fn hash_layout() -> super::super::OperationalGadgetLayout {
        super::super::OperationalGadgetLayout {
            params_id: "hash-layout".to_owned(),
            ring_dimension: 1,
            crt_moduli: vec![17],
            crt_bits: 5,
            base_bits: 2,
            base: BigInt::from(4),
            regular_digit_count: 3,
            small_digit_count: 3,
            smallest_crt_modulus: 17,
        }
    }

    fn decomposed_hash_graph(
        variant: HashVariant,
        digit_count: i64,
    ) -> (mxx_ir_core::graph::Graph, WireRef) {
        use mxx_ir_core::graph::{GraphOutput, NodeHandle};

        let matrix = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(3),
            columns: IntExpr::constant(2),
        };
        let key = NodeHandle::new(
            NodeKind::Input {
                name: "key".to_owned(),
                wire_type: WireType::Bytes { length: IntExpr::constant(32) },
                artifact: None,
            },
            Vec::new(),
            vec![WireType::Bytes { length: IntExpr::constant(32) }],
        )
        .output(0)
        .expect("hash key output");
        let runtime_tag = NodeHandle::new(
            NodeKind::ConstantInt(BigInt::from(23)),
            Vec::new(),
            vec![WireType::Int],
        )
        .output(0)
        .expect("runtime tag output");
        let hash = NodeHandle::new(
            NodeKind::HashSample {
                matrix_type: matrix.clone(),
                variant,
                tag_prefix: b"hash-fixture".to_vec(),
                tag_expressions: vec![IntExpr::constant(7)],
                tag_decimal_expressions: vec![IntExpr::constant(8)],
                tag_u64_le_expressions: vec![IntExpr::constant(9)],
                base: Some(IntExpr::constant(4)),
                digit_count: Some(IntExpr::constant(digit_count)),
            },
            vec![key, runtime_tag],
            vec![WireType::Matrix(matrix)],
        )
        .output(0)
        .expect("hash output");
        let graph = mxx_ir_core::graph::Graph::freeze(
            "decomposed-hash-fixture",
            Vec::new(),
            BTreeMap::from([(
                "output".to_owned(),
                GraphOutput { value: hash.clone(), confidentiality: None },
            )]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .expect("freeze hash graph")
        .0;
        let output = graph.outputs()["output"].value;
        (graph, output)
    }

    fn hash_protocol(graph: mxx_ir_core::graph::Graph) -> crate::ProtocolDecl {
        let mut protocol = crate::toy_example::protocol();
        protocol.bundle.workflow.stages[0].graph = graph;
        let key = crate::ProtocolInputId("hash-key".to_owned());
        protocol.bundle.input_contract.inputs.push(crate::InputContractEntry {
            id: key.clone(),
            name: "key".to_owned(),
            value: crate::InputValueContract::Bytes { length: IntExpr::constant(32) },
        });
        protocol.bundle.input_bindings.push(crate::ProtocolInputBinding {
            input: key,
            destinations: vec![ProtocolInputDestination::WorkflowStage {
                stage: StageId("encrypt".to_owned()),
                input: StageInputName("key".to_owned()),
            }],
        });
        protocol
    }

    fn sampled_matrix_graph(kind: NodeKind) -> (mxx_ir_core::graph::Graph, WireRef) {
        use mxx_ir_core::graph::{GraphOutput, NodeHandle};

        let matrix = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let output = NodeHandle::new(kind, Vec::new(), vec![WireType::Matrix(matrix)])
            .output(0)
            .expect("sampler output");
        let graph = mxx_ir_core::graph::Graph::freeze(
            "bounded-sampler-fixture",
            Vec::new(),
            BTreeMap::from([(
                "output".to_owned(),
                GraphOutput { value: output.clone(), confidentiality: None },
            )]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .expect("freeze sampler graph")
        .0;
        let output = graph.outputs()["output"].value;
        (graph, output)
    }

    fn with_lowered_sampled_matrix(
        kind: NodeKind,
        inspect: impl FnOnce(&GraphLowerer<'_, '_>, TermId),
    ) {
        let (graph, output) = sampled_matrix_graph(kind);
        let protocol = hash_protocol(graph);
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "sampler".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let LoweredValue::Matrix(term) = lowerer
            .lower_stage_wire(&StageId("encrypt".to_owned()), output)
            .expect("lower sampler")
        else {
            panic!("sampler is a DAG matrix term")
        };
        inspect(&lowerer, term);
    }

    fn dag_atom<'a, 'b, 'c>(lowerer: &'a GraphLowerer<'b, 'c>, term: TermId) -> &'a SymbolicFactor {
        match lowerer.expression_dag().node(term).expect("DAG term") {
            ExpressionNode::Atom(factor) => factor,
            node => panic!("expected DAG atom, got {node:?}"),
        }
    }

    fn dag_bound(lowerer: &GraphLowerer<'_, '_>, term: TermId) -> BoundClass {
        match lowerer.expression_dag().node(term).expect("DAG term") {
            ExpressionNode::Zero => return BoundClass::ExactZero,
            ExpressionNode::Atom(factor) if factor.bound == BoundClass::ExactZero => {
                return BoundClass::ExactZero
            }
            _ => {}
        }
        let normalized = lowerer
            .expression_dag()
            .normalize(term, lowerer.normal_form_relations())
            .expect("finite DAG bound");
        if let Some(summary) = normalized.bounded_summary().as_matrix_bound() {
            return summary.coefficient_class.clone();
        }
        let exact = normalized.exact_terms().values().next().expect("finite exact term");
        monomial_bound(&exact.monomial)
            .expect("finite exact matrix bound")
            .coefficient_class
            .clone()
    }

    fn lower_sampled_matrix_result(kind: NodeKind) -> Result<(), LowerError> {
        let (graph, output) = sampled_matrix_graph(kind);
        let protocol = hash_protocol(graph);
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "sampler".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        lowerer.lower_stage_wire(&StageId("encrypt".to_owned()), output).map(|_| ())
    }

    fn assert_explicit_large_source(
        kind: NodeKind,
        arguments: Vec<mxx_ir_core::graph::ValueHandle>,
        output_types: Vec<WireType>,
        port: u32,
    ) {
        use mxx_ir_core::graph::{GraphOutput, NodeHandle};

        let output = NodeHandle::new(kind, arguments, output_types)
            .output(port)
            .expect("explicit-large output");
        let graph = mxx_ir_core::graph::Graph::freeze(
            "explicit-large-source",
            Vec::new(),
            BTreeMap::from([(
                "output".to_owned(),
                GraphOutput { value: output, confidentiality: None },
            )]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .expect("freeze explicit-large graph")
        .0;
        let output = graph.outputs()["output"].value;
        let mut protocol = crate::toy_example::protocol();
        protocol.bundle.workflow.stages[0].graph = graph;
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "explicit-large-source".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let LoweredValue::Matrix(term) = lowerer
            .lower_stage_wire(&StageId("encrypt".to_owned()), output)
            .expect("lower explicit-large source")
        else {
            panic!("explicit-large source is a DAG matrix term")
        };
        assert!(matches!(
            dag_atom(&lowerer, term).key.owner,
            super::super::normal_form::FactorOwner::Atomic(
                super::super::identity::AtomicSourceKey::ExplicitLarge(_)
            )
        ));
        assert_eq!(dag_atom(&lowerer, term).bound, BoundClass::Large);
    }

    #[test]
    fn only_semantically_explicit_source_operations_create_large_atoms() {
        use mxx_ir_core::graph::NodeHandle;

        let matrix = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        assert_explicit_large_source(
            NodeKind::UniformResidueSample { matrix_type: matrix.clone() },
            Vec::new(),
            vec![WireType::Matrix(matrix.clone())],
            0,
        );
        let trapdoor_type = WireType::Trapdoor {
            matrix: matrix.clone(),
            sigma: mxx_ir_core::RealExpr::from_integer(1),
            gadget_base: IntExpr::constant(2),
            digit_count: IntExpr::constant(2),
            preimage_max_coefficient_bound: IntExpr::constant(3),
        };
        assert_explicit_large_source(
            NodeKind::TrapdoorSample {
                matrix_type: matrix.clone(),
                sigma: mxx_ir_core::RealExpr::from_integer(1),
                gadget_base: IntExpr::constant(2),
                digit_count: IntExpr::constant(2),
                preimage_max_coefficient_bound: IntExpr::constant(3),
            },
            Vec::new(),
            vec![WireType::Matrix(matrix.clone()), trapdoor_type.clone()],
            0,
        );
        let trapdoor = NodeHandle::new(
            NodeKind::GadgetTrapdoor { matrix_type: matrix.clone(), base: IntExpr::constant(2) },
            Vec::new(),
            vec![trapdoor_type],
        )
        .output(0)
        .expect("gadget trapdoor");
        assert_explicit_large_source(
            NodeKind::TrapdoorPublic,
            vec![trapdoor],
            vec![WireType::Matrix(matrix)],
            0,
        );
    }

    #[test]
    fn artifact_alias_preserves_the_producer_preimage_cutoff() {
        use mxx_dsl::{DslContext, Ring};
        use mxx_ir_core::artifact::{ArtifactConfidentiality, ProductionId, SpecHash};

        let ring = Ring::new(17, 1);
        let trapdoor = ring.sample_trapdoor(1, 1, 2, 2, 9);
        let preimage = trapdoor
            .sample_preimage(
                ring.zero((1, 1)),
                (trapdoor.public_matrix().matrix_type().columns.clone(), 1),
            )
            .as_mat();
        let producer = DslContext::new("preimage-producer")
            .private_output("preimage", preimage)
            .expect("producer output")
            .build()
            .expect("producer graph");
        let placeholder = ProductionId { spec_hash: SpecHash([0; 32]), execution_nonce: [0; 32] };
        let artifact =
            ring.artifact_input(placeholder, "preimage", (4, 1), ArtifactConfidentiality::Private);
        let consumer = DslContext::new("preimage-consumer")
            .private_output("copied", artifact)
            .expect("consumer output")
            .build()
            .expect("consumer graph");
        let output = consumer.graph.outputs()["copied"].value;

        let mut protocol = crate::toy_example::protocol();
        protocol.bundle.workflow.stages[0].graph = producer.graph;
        protocol.bundle.workflow.stages[1].graph = consumer.graph;
        protocol.bundle.workflow.stages[1].bindings = vec![crate::ArtifactBinding {
            consumer_input: StageInputName("preimage".to_owned()),
            producer_stage: StageId("encrypt".to_owned()),
            producer_output: crate::ArtifactName("preimage".to_owned()),
        }];
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "artifact-preimage".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let LoweredValue::Matrix(term) = lowerer
            .lower_stage_wire(&StageId("decrypt".to_owned()), output)
            .expect("artifact aliases producer")
        else {
            panic!("preimage artifact is a DAG matrix term")
        };
        assert_eq!(dag_bound(&lowerer, term), BoundClass::bounded(9_u8.into()));
        assert!(dag_atom(&lowerer, term).relation_live);
    }

    #[test]
    fn explicit_boolean_family_pack_lowers_without_lane_inference() {
        use mxx_dsl::{Bool, DslContext, Family, Ring};

        let ring = Ring::new(17, 1);
        let bits = Family::<Bool>::pack(vec![
            Bool::constant(true),
            Bool::constant(false),
            Bool::constant(true),
            Bool::constant(false),
            Bool::constant(false),
        ])
        .expect("five explicit bits");
        let packed = ring.pack_polynomial_coefficients(bits, 5);
        let built = DslContext::new("pack-bits")
            .private_output("packed", packed)
            .expect("packed output")
            .build()
            .expect("pack graph");
        let output = built.graph.outputs()["packed"].value;
        let mut protocol = crate::toy_example::protocol();
        protocol.bundle.workflow.stages[0].graph = built.graph;
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "pack-bits".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let LoweredValue::Matrix(term) = lowerer
            .lower_stage_wire(&StageId("encrypt".to_owned()), output)
            .expect("explicit bits lower")
        else {
            panic!("packed polynomial is a matrix term")
        };
        assert_eq!(dag_bound(&lowerer, term), BoundClass::bounded(5_u8.into()));
    }

    #[test]
    fn family_pack_accepts_a_slice_with_a_semantically_equal_matrix_type() {
        use mxx_ir_core::{
            graph::{GraphOutput, NodeHandle},
            node::IndexRange,
        };

        let source_type = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(2),
            columns: IntExpr::constant(1),
        };
        let element_type = MatrixType { rows: IntExpr::constant(1), ..source_type.clone() };
        let source = NodeHandle::new(
            NodeKind::ConstantMatrix {
                matrix_type: source_type.clone(),
                value: mxx_ir_core::node::ConstantMatrix::Zero,
            },
            Vec::new(),
            vec![WireType::Matrix(source_type)],
        )
        .output(0)
        .expect("source matrix");
        let slice = NodeHandle::new(
            NodeKind::Slice {
                rows: Some(IndexRange { start: IntExpr::constant(0), end: IntExpr::constant(1) }),
                columns: None,
            },
            vec![source],
            vec![WireType::Matrix(element_type.clone())],
        )
        .output(0)
        .expect("one-row slice");
        let family_type = WireType::IndexedFamily {
            element: Box::new(WireType::Matrix(element_type.clone())),
            count: IntExpr::constant(2),
        };
        let family = NodeHandle::new(
            NodeKind::FamilyPack { count: IntExpr::constant(2) },
            vec![slice.clone(), slice],
            vec![family_type],
        )
        .output(0)
        .expect("family");
        let output = NodeHandle::new(
            NodeKind::FamilyGetStatic { index: IntExpr::constant(0) },
            vec![family],
            vec![WireType::Matrix(element_type)],
        )
        .output(0)
        .expect("family element");
        let graph = mxx_ir_core::graph::Graph::freeze(
            "family-pack-semantic-slice-type",
            Vec::new(),
            BTreeMap::from([(
                "output".to_owned(),
                GraphOutput { value: output, confidentiality: None },
            )]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .expect("freeze graph")
        .0;
        let mut protocol = crate::toy_example::protocol();
        protocol.bundle.workflow.stages[0].graph = graph;
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "family-pack-semantic-slice-type".to_owned(),
        };
        let output = protocol.bundle.workflow.stages[0].graph.outputs()["output"].value;
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        lowerer
            .lower_stage_wire(&StageId("encrypt".to_owned()), output)
            .expect("family accepts the equivalent one-row slice type");
    }

    #[test]
    fn gaussian_and_uniform_interval_are_bounded_nonrelation_samplers() {
        with_lowered_sampled_matrix(
            NodeKind::GaussianSample {
                matrix_type: MatrixType {
                    modulus: IntExpr::constant(17),
                    ring_dimension: IntExpr::constant(1),
                    rows: IntExpr::constant(1),
                    columns: IntExpr::constant(1),
                },
                sigma: mxx_ir_core::RealExpr::from_integer(1),
                max_coefficient_bound: IntExpr::constant(5),
            },
            |gaussian, gaussian_term| {
                let [SamplerIdentity::Gaussian { max_coefficient_bound, .. }] =
                    gaussian.symbols.samplers.values.as_slice()
                else {
                    panic!("expected one Gaussian sampler")
                };
                assert_eq!(resolved_constant(max_coefficient_bound), Some(BigInt::from(5)));
                assert_eq!(
                    dag_bound(gaussian, gaussian_term),
                    BoundClass::Bounded { maximum_absolute_coefficient: 5_u8.into() },
                );
                assert!(dag_atom(gaussian, gaussian_term).matrix_bound.is_some());
            },
        );

        with_lowered_sampled_matrix(
            NodeKind::UniformIntervalSample {
                matrix_type: MatrixType {
                    modulus: IntExpr::constant(17),
                    ring_dimension: IntExpr::constant(1),
                    rows: IntExpr::constant(1),
                    columns: IntExpr::constant(1),
                },
                range: mxx_ir_core::node::SampleRange {
                    minimum: IntExpr::constant(-3),
                    maximum: IntExpr::constant(2),
                },
            },
            |interval, interval_term| {
                let [SamplerIdentity::UniformInterval { minimum, maximum, .. }] =
                    interval.symbols.samplers.values.as_slice()
                else {
                    panic!("expected one uniform sampler")
                };
                assert_eq!(resolved_constant(minimum), Some(BigInt::from(-3)));
                assert_eq!(resolved_constant(maximum), Some(BigInt::from(2)));
                assert_eq!(
                    dag_bound(interval, interval_term),
                    BoundClass::Bounded { maximum_absolute_coefficient: 3_u8.into() },
                );
            },
        );
    }

    #[test]
    fn bounded_nonrelation_sampler_contracts_fail_closed() {
        let gaussian = lower_sampled_matrix_result(NodeKind::GaussianSample {
            matrix_type: MatrixType {
                modulus: IntExpr::constant(17),
                ring_dimension: IntExpr::constant(1),
                rows: IntExpr::constant(1),
                columns: IntExpr::constant(1),
            },
            sigma: mxx_ir_core::RealExpr::from_integer(1),
            max_coefficient_bound: IntExpr::constant(-1),
        });
        assert!(
            matches!(gaussian, Err(LowerError::NegativeSamplerCutoff { cutoff }) if cutoff == BigInt::from(-1))
        );

        let interval = lower_sampled_matrix_result(NodeKind::UniformIntervalSample {
            matrix_type: MatrixType {
                modulus: IntExpr::constant(17),
                ring_dimension: IntExpr::constant(1),
                rows: IntExpr::constant(1),
                columns: IntExpr::constant(1),
            },
            range: mxx_ir_core::node::SampleRange {
                minimum: IntExpr::constant(3),
                maximum: IntExpr::constant(2),
            },
        });
        assert!(
            matches!(interval, Err(LowerError::InvalidUniformInterval { minimum, maximum }) if minimum == BigInt::from(3) && maximum == BigInt::from(2))
        );
    }

    #[test]
    fn declared_matrix_product_bound_rejects_zero_dimensions() {
        let protocol = crate::toy_example::protocol();
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "declared-matrix-product-bound".to_owned(),
        };
        let lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let product = |ring_dimension, inner_dimension| DeclaredBoundExpr::MatrixProduct {
            ring_dimension: IntExpr::constant(ring_dimension),
            inner_dimension: IntExpr::constant(inner_dimension),
            left: Box::new(DeclaredBoundExpr::Constant(4_u8.into())),
            right: Box::new(DeclaredBoundExpr::Constant(5_u8.into())),
        };
        assert_eq!(lowerer.declared_bound_value(&product(0, 3)), None);
        assert_eq!(lowerer.declared_bound_value(&product(2, 0)), None);
        assert_eq!(lowerer.declared_bound_value(&product(2, 3)), Some(120_u16.into()));
    }

    #[test]
    fn decomposed_hash_lowers_exact_query_gadget_and_relation() {
        let (graph, output) = decomposed_hash_graph(HashVariant::Decomposed, 3);
        let protocol = hash_protocol(graph);
        let request = OperationalCheckRequest {
            environment: vec![(
                "cutoff".to_owned(),
                super::super::OperationalParameterValue::Integer(BigInt::from(1)),
            )],
            layouts: vec![hash_layout()],
            target_id: "hash".to_owned(),
        };
        let stage = StageId("encrypt".to_owned());
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let LoweredValue::Matrix(lowered) =
            lowerer.lower_stage_wire(&stage, output).expect("lower decomposed hash")
        else {
            panic!("decomposed hash is a DAG matrix term")
        };
        let factor = dag_atom(&lowerer, lowered);
        assert_eq!(factor.bound, BoundClass::Bounded { maximum_absolute_coefficient: 2_u8.into() });
        assert_eq!(dag_bound(&lowerer, lowered), factor.bound);
        assert!(matches!(
            factor.key.owner,
            super::super::normal_form::FactorOwner::Atomic(
                super::super::identity::AtomicSourceKey::GraphWire(_)
            )
        ));
    }

    #[test]
    fn gadget_decomposition_sampler_uses_the_regular_digit_bound() {
        let (graph, output) = decomposed_hash_graph(HashVariant::Decomposed, 3);
        let protocol = hash_protocol(graph);
        let request = OperationalCheckRequest {
            environment: vec![(
                "cutoff".to_owned(),
                super::super::OperationalParameterValue::Integer(BigInt::from(1)),
            )],
            layouts: vec![hash_layout()],
            target_id: "hash".to_owned(),
        };
        let stage = StageId("encrypt".to_owned());
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let LoweredValue::Matrix(decomposed_hash_dag) =
            lowerer.lower_stage_wire(&stage, output).expect("lower decomposed hash")
        else {
            panic!("decomposed hash is a DAG matrix term")
        };
        assert_eq!(
            dag_bound(&lowerer, decomposed_hash_dag),
            BoundClass::Bounded { maximum_absolute_coefficient: 2_u8.into() },
        );
        assert!(dag_atom(&lowerer, decomposed_hash_dag).matrix_bound.is_some());
    }

    #[test]
    fn lowering_reports_owner_resolved_work_without_static_callback_ownership() {
        let (graph, output) = decomposed_hash_graph(HashVariant::Decomposed, 3);
        let protocol = hash_protocol(graph);
        let request = OperationalCheckRequest {
            environment: vec![(
                "cutoff".to_owned(),
                super::super::OperationalParameterValue::Integer(BigInt::from(1)),
            )],
            layouts: vec![hash_layout()],
            target_id: "hash".to_owned(),
        };
        let mut control = RecordingLoweringControl::default();
        let mut lowerer = GraphLowerer::new_with_control(
            &protocol,
            &request,
            SymbolTables::default(),
            &mut control,
        );
        lowerer
            .lower_stage_wire(&StageId("encrypt".to_owned()), output)
            .expect("lower with borrowed progress control");
        let lowerer = lowerer.into_uncontrolled();
        assert!(lowerer.lowered_wire_count() >= 3);
        assert!(control.sites.len() >= 3);
        assert!(control.sites.iter().all(|(scope, _)| {
            scope.program ==
                super::super::identity::ProgramKey::WorkflowStage(StageId("encrypt".to_owned()))
        }));
    }

    #[test]
    fn decomposed_hash_rejects_nondivisible_output_rows() {
        let (graph, output) = decomposed_hash_graph(HashVariant::Decomposed, 2);
        let protocol = hash_protocol(graph);
        let request = OperationalCheckRequest {
            environment: vec![(
                "cutoff".to_owned(),
                super::super::OperationalParameterValue::Integer(BigInt::from(1)),
            )],
            layouts: vec![hash_layout()],
            target_id: "hash".to_owned(),
        };
        let stage = StageId("encrypt".to_owned());
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        assert!(matches!(
            lowerer.lower_stage_wire(&stage, output),
            Err(LowerError::InvalidOperandArity { .. })
        ));
    }

    #[test]
    fn small_decomposed_hash_has_a_digit_bound_without_a_range_proof() {
        let (graph, output) = decomposed_hash_graph(HashVariant::SmallDecomposed, 3);
        let protocol = hash_protocol(graph);
        let request = OperationalCheckRequest {
            environment: vec![(
                "cutoff".to_owned(),
                super::super::OperationalParameterValue::Integer(BigInt::from(1)),
            )],
            layouts: vec![hash_layout()],
            target_id: "hash".to_owned(),
        };
        let stage = StageId("encrypt".to_owned());
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let LoweredValue::Matrix(lowered) =
            lowerer.lower_stage_wire(&stage, output).expect("lower small decomposed hash")
        else {
            panic!("small decomposed hash is a DAG matrix term")
        };
        assert_eq!(
            dag_bound(&lowerer, lowered),
            BoundClass::Bounded { maximum_absolute_coefficient: 3_u8.into() },
        );
        assert!(dag_atom(&lowerer, lowered).matrix_bound.is_some());
    }

    #[test]
    fn root_constant_matrix_lowers_to_exact_zero_without_a_graph_wire() {
        use mxx_ir_core::graph::{GraphOutput, NodeHandle};

        let matrix = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let output = NodeHandle::new(
            NodeKind::ConstantMatrix {
                matrix_type: matrix.clone(),
                value: mxx_ir_core::node::ConstantMatrix::Zero,
            },
            Vec::new(),
            vec![WireType::Matrix(matrix)],
        )
        .output(0)
        .expect("zero constant output");
        let graph = mxx_ir_core::graph::Graph::freeze(
            "root-constant-matrix",
            Vec::new(),
            BTreeMap::from([(
                "output".to_owned(),
                GraphOutput { value: output, confidentiality: None },
            )]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .expect("freeze zero constant graph")
        .0;
        let output = graph.outputs()["output"].value;
        let mut protocol = crate::toy_example::protocol();
        protocol.bundle.workflow.stages[0].graph = graph;
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "root-constant-matrix".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let LoweredValue::Matrix(term) = lowerer
            .lower_stage_wire(&StageId("encrypt".to_owned()), output)
            .expect("lower zero constant")
        else {
            panic!("zero constant is a DAG matrix term")
        };
        let ExpressionNode::Atom(factor) =
            lowerer.expression_dag().node(term).expect("zero DAG node")
        else {
            panic!("typed zero constant is an atom in the matrix DAG")
        };
        assert_eq!(factor.bound, BoundClass::ExactZero);
        assert_eq!(
            factor.matrix_bound.as_ref().expect("typed zero matrix bound").matrix_type,
            mxx_ir_core::types::ConcreteMatrixType {
                modulus: BigInt::from(17),
                ring_dimension: 1,
                rows: 1,
                columns: 1,
            }
        );
        assert_eq!(dag_bound(&lowerer, term), BoundClass::ExactZero);
    }

    #[test]
    fn parallel_constant_matrix_polynomial_lowers_without_a_graph_wire() {
        use mxx_ir_core::graph::{
            GraphOutput, NodeHandle, SubgraphHandle, with_new_construction_scope,
        };

        let matrix = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(2),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let body = with_new_construction_scope(|scope| {
            let polynomial = NodeHandle::new(
                NodeKind::ConstantMatrix {
                    matrix_type: matrix.clone(),
                    value: mxx_ir_core::node::ConstantMatrix::Polynomial {
                        coefficients: vec![IntExpr::constant(-5), IntExpr::constant(3)],
                    },
                },
                Vec::new(),
                vec![WireType::Matrix(matrix.clone())],
            )
            .output(0)
            .expect("polynomial constant output");
            SubgraphHandle::new("constant-body", scope, Vec::new(), vec![polynomial])
                .expect("constant loop body")
        });
        let family_type = WireType::IndexedFamily {
            element: Box::new(WireType::Matrix(matrix.clone())),
            count: IntExpr::constant(2),
        };
        let family = NodeHandle::parallel_loop(
            body,
            Vec::new(),
            vec![family_type.clone()],
            ParallelLoop {
                count: IntExpr::constant(2),
                minimum_count: 0,
                index_slot: 0,
                bindings: Vec::new(),
                input_modes: Vec::new(),
            },
        )
        .output(0)
        .expect("constant family output");
        let output = NodeHandle::new(
            NodeKind::FamilyGetStatic { index: IntExpr::constant(0) },
            vec![family],
            vec![WireType::Matrix(matrix)],
        )
        .output(0)
        .expect("constant family element");
        let graph = mxx_ir_core::graph::Graph::freeze(
            "parallel-constant-matrix",
            Vec::new(),
            BTreeMap::from([(
                "output".to_owned(),
                GraphOutput { value: output, confidentiality: None },
            )]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .expect("freeze constant loop graph")
        .0;
        let output = graph.outputs()["output"].value;
        let mut protocol = crate::toy_example::protocol();
        protocol.bundle.workflow.stages[0].graph = graph;
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "parallel-constant-matrix".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let LoweredValue::Matrix(term) = lowerer
            .lower_stage_wire(&StageId("encrypt".to_owned()), output)
            .expect("lower polynomial constant through parallel loop")
        else {
            panic!("polynomial constant is a matrix term")
        };
        assert_eq!(dag_bound(&lowerer, term), BoundClass::bounded(5_u8.into()));
        assert!(lowerer.symbols.atomic_sources.values.iter().all(|descriptor| {
            !matches!(descriptor.key, super::super::identity::AtomicSourceKey::GraphWire(_))
        }));
    }

    fn deep_parallel_graph(depth: usize, logical_count: i64) -> mxx_ir_core::graph::Graph {
        use mxx_ir_core::{
            graph::{
                GraphOutput, NodeHandle, SubgraphHandle, ValueHandle, with_new_construction_scope,
            },
            node::LoopInputMode,
        };

        let matrix = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        fn nested(
            depth: usize,
            input: ValueHandle,
            matrix: &MatrixType,
            logical_count: i64,
        ) -> ValueHandle {
            if depth == 0 {
                return NodeHandle::new(
                    NodeKind::MatrixScale { scalar: IntExpr::constant(1) },
                    vec![input],
                    vec![WireType::Matrix(matrix.clone())],
                )
                .output(0)
                .expect("matrix scale output");
            }
            let body = with_new_construction_scope(|scope| {
                let body_input = NodeHandle::new(
                    NodeKind::Input {
                        name: format!("input-{depth}"),
                        wire_type: WireType::Matrix(matrix.clone()),
                        artifact: None,
                    },
                    Vec::new(),
                    vec![WireType::Matrix(matrix.clone())],
                )
                .output(0)
                .expect("body input output");
                let output = nested(depth - 1, body_input.clone(), matrix, logical_count);
                SubgraphHandle::new(
                    format!("deep-parallel-{depth}"),
                    scope,
                    vec![body_input],
                    vec![output],
                )
                .expect("parallel body")
            });
            let family = WireType::IndexedFamily {
                element: Box::new(WireType::Matrix(matrix.clone())),
                count: IntExpr::constant(logical_count),
            };
            let loop_output = NodeHandle::parallel_loop(
                body,
                vec![input],
                vec![family.clone()],
                ParallelLoop {
                    count: IntExpr::constant(logical_count),
                    minimum_count: 0,
                    index_slot: 0,
                    bindings: Vec::new(),
                    input_modes: vec![LoopInputMode::Broadcast],
                },
            )
            .output(0)
            .expect("parallel loop output");
            NodeHandle::new(
                NodeKind::FamilyGetStatic { index: IntExpr::constant(0) },
                vec![loop_output],
                vec![WireType::Matrix(matrix.clone())],
            )
            .output(0)
            .expect("family element output")
        }

        let root_source = NodeHandle::new(
            NodeKind::ConstantMatrix {
                matrix_type: matrix.clone(),
                value: mxx_ir_core::node::ConstantMatrix::Zero,
            },
            Vec::new(),
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .expect("root input output");
        // Keep a parent-owned dependency below the wire captured by every nested body.
        // Lowering that captured wire must retain the wire owner's definition instead of
        // accidentally resolving this source node in the innermost child definition.
        let input = NodeHandle::new(
            NodeKind::MatrixScale { scalar: IntExpr::constant(1) },
            vec![root_source],
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .expect("parent-owned dependency output");
        let output = nested(depth, input, &matrix, logical_count);
        mxx_ir_core::graph::Graph::freeze(
            "deep-parallel-lowering",
            Vec::new(),
            BTreeMap::from([(
                "output".to_owned(),
                GraphOutput { value: output, confidentiality: None },
            )]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .expect("freeze deep graph")
        .0
    }

    #[test]
    fn parallel_body_binding_survives_a_nested_subgraph_call() {
        use mxx_ir_core::{
            graph::{GraphOutput, NodeHandle, SubgraphHandle, with_new_construction_scope},
            node::LoopInputMode,
        };

        let matrix = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let root = NodeHandle::new(
            NodeKind::UniformIntervalSample {
                matrix_type: matrix.clone(),
                range: mxx_ir_core::node::SampleRange {
                    minimum: IntExpr::constant(0),
                    maximum: IntExpr::constant(0),
                },
            },
            Vec::new(),
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .expect("root constant");
        let body = with_new_construction_scope(|scope| {
            let body_input = NodeHandle::new(
                NodeKind::Input {
                    name: "parallel-input".to_owned(),
                    wire_type: WireType::Matrix(matrix.clone()),
                    artifact: None,
                },
                Vec::new(),
                vec![WireType::Matrix(matrix.clone())],
            )
            .output(0)
            .expect("parallel body input");
            let identity = with_new_construction_scope(|inner_scope| {
                let input = NodeHandle::new(
                    NodeKind::Input {
                        name: "subgraph-input".to_owned(),
                        wire_type: WireType::Matrix(matrix.clone()),
                        artifact: None,
                    },
                    Vec::new(),
                    vec![WireType::Matrix(matrix.clone())],
                )
                .output(0)
                .expect("subgraph input");
                SubgraphHandle::new("identity", inner_scope, vec![input.clone()], vec![input])
                    .expect("identity subgraph")
            });
            let output = NodeHandle::subgraph_call(
                identity,
                vec![body_input.clone()],
                Vec::new(),
                vec![None],
            )
            .output(0)
            .expect("subgraph output");
            SubgraphHandle::new("parallel-body", scope, vec![body_input], vec![output])
                .expect("parallel body")
        });
        let family_type = WireType::IndexedFamily {
            element: Box::new(WireType::Matrix(matrix.clone())),
            count: IntExpr::constant(2),
        };
        let family = NodeHandle::parallel_loop(
            body,
            vec![root],
            vec![family_type],
            mxx_ir_core::node::ParallelLoop {
                count: IntExpr::constant(2),
                minimum_count: 0,
                index_slot: 0,
                bindings: Vec::new(),
                input_modes: vec![LoopInputMode::Broadcast],
            },
        )
        .output(0)
        .expect("parallel output");
        let output = NodeHandle::new(
            NodeKind::FamilyGetStatic { index: IntExpr::constant(0) },
            vec![family],
            vec![WireType::Matrix(matrix)],
        )
        .output(0)
        .expect("family element");
        let graph = mxx_ir_core::graph::Graph::freeze(
            "parallel-subgraph-alias",
            Vec::new(),
            BTreeMap::from([(
                "output".to_owned(),
                GraphOutput { value: output, confidentiality: None },
            )]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .expect("freeze graph")
        .0;
        let output = graph.outputs()["output"].value;
        let mut protocol = crate::toy_example::protocol();
        protocol.bundle.workflow.stages[0].graph = graph;
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "parallel-subgraph-alias".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        lowerer
            .lower_stage_wire(&StageId("encrypt".to_owned()), output)
            .expect("bound parallel input lowers through subgraph call");
        assert!(lowerer.symbols.atomic_sources.values.iter().all(|descriptor| {
            !matches!(descriptor.key, super::super::identity::AtomicSourceKey::GraphWire(_))
        }));
    }

    fn zipped_parallel_graph(
        right_mode: LoopInputMode,
    ) -> (mxx_ir_core::graph::Graph, WireRef, Vec<(&'static str, crate::ProtocolInputId)>) {
        use mxx_ir_core::graph::{
            GraphOutput, NodeHandle, SubgraphHandle, with_new_construction_scope,
        };

        let family_type = WireType::IndexedFamily {
            element: Box::new(WireType::Int),
            count: IntExpr::constant(4),
        };
        let left = NodeHandle::new(
            NodeKind::Input {
                name: "zip-left".to_owned(),
                wire_type: family_type.clone(),
                artifact: None,
            },
            Vec::new(),
            vec![family_type.clone()],
        )
        .output(0)
        .expect("left family input");
        let right = NodeHandle::new(
            NodeKind::Input {
                name: "zip-right".to_owned(),
                wire_type: family_type.clone(),
                artifact: None,
            },
            Vec::new(),
            vec![family_type.clone()],
        )
        .output(0)
        .expect("right family input");
        let body = with_new_construction_scope(|scope| {
            let left_input = NodeHandle::new(
                NodeKind::Input {
                    name: "left".to_owned(),
                    wire_type: WireType::Int,
                    artifact: None,
                },
                Vec::new(),
                vec![WireType::Int],
            )
            .output(0)
            .expect("left body input");
            let right_input = NodeHandle::new(
                NodeKind::Input {
                    name: "right".to_owned(),
                    wire_type: WireType::Int,
                    artifact: None,
                },
                Vec::new(),
                vec![WireType::Int],
            )
            .output(0)
            .expect("right body input");
            let output = NodeHandle::new(
                NodeKind::IntBinary(IntBinaryOp::Add),
                vec![left_input.clone(), right_input.clone()],
                vec![WireType::Int],
            )
            .output(0)
            .expect("body output");
            SubgraphHandle::new("zip-body", scope, vec![left_input, right_input], vec![output])
                .expect("zip body")
        });
        let output_family_type = WireType::IndexedFamily {
            element: Box::new(WireType::Int),
            count: IntExpr::constant(3),
        };
        let family = NodeHandle::parallel_loop(
            body,
            vec![left, right],
            vec![output_family_type],
            ParallelLoop {
                count: IntExpr::constant(3),
                minimum_count: 0,
                index_slot: 0,
                bindings: Vec::new(),
                input_modes: vec![LoopInputMode::Zip, right_mode],
            },
        )
        .output(0)
        .expect("parallel family output");
        let output = NodeHandle::new(
            NodeKind::FamilyGetStatic { index: IntExpr::constant(0) },
            vec![family],
            vec![WireType::Int],
        )
        .output(0)
        .expect("selected family output");
        let graph = mxx_ir_core::graph::Graph::freeze(
            "frozen-zip-family",
            Vec::new(),
            BTreeMap::from([(
                "output".to_owned(),
                GraphOutput { value: output, confidentiality: None },
            )]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .expect("freeze zip graph")
        .0;
        let inputs = [
            ("zip-left", crate::ProtocolInputId("zip-left".to_owned())),
            ("zip-right", crate::ProtocolInputId("zip-right".to_owned())),
        ];
        (graph.clone(), graph.outputs()["output"].value, inputs.to_vec())
    }

    fn zip_protocol(
        graph: mxx_ir_core::graph::Graph,
        inputs: &[(&'static str, crate::ProtocolInputId)],
    ) -> crate::ProtocolDecl {
        let mut protocol = crate::toy_example::protocol();
        protocol.bundle.workflow.stages[0].graph = graph;
        for (name, id) in inputs {
            protocol.bundle.input_contract.inputs.push(crate::InputContractEntry {
                id: id.clone(),
                name: (*name).to_owned(),
                value: InputValueContract::Family {
                    count: IntExpr::constant(4),
                    element: Box::new(InputValueContract::IntegerRange {
                        lower: IntExpr::constant(0),
                        upper: IntExpr::constant(9),
                    }),
                },
            });
            protocol.bundle.input_bindings.push(crate::ProtocolInputBinding {
                input: id.clone(),
                destinations: vec![ProtocolInputDestination::WorkflowStage {
                    stage: StageId("encrypt".to_owned()),
                    input: StageInputName((*name).to_owned()),
                }],
            });
        }
        protocol
    }

    #[test]
    fn frozen_parallel_zip_consumes_the_matching_family_lane() {
        let (graph, output, inputs) = zipped_parallel_graph(LoopInputMode::Zip);
        let protocol = zip_protocol(graph, &inputs);
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "frozen-zip-family".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        assert!(matches!(
            lowerer.lower_stage_wire(&StageId("encrypt".to_owned()), output),
            Ok(LoweredValue::Scalar(_))
        ));
    }

    #[test]
    fn frozen_parallel_zip_offset_consumes_the_offset_family_lane() {
        let (graph, output, inputs) = zipped_parallel_graph(LoopInputMode::ZipOffset { offset: 1 });
        let protocol = zip_protocol(graph, &inputs);
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "frozen-zip-family".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        assert!(matches!(
            lowerer.lower_stage_wire(&StageId("encrypt".to_owned()), output),
            Ok(LoweredValue::Scalar(_))
        ));
    }

    fn zipped_parallel_matrix_graph(
        right_mode: LoopInputMode,
    ) -> (mxx_ir_core::graph::Graph, WireRef, Vec<(&'static str, crate::ProtocolInputId)>) {
        use mxx_ir_core::graph::{
            GraphOutput, NodeHandle, SubgraphHandle, with_new_construction_scope,
        };
        let matrix = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let family_type = WireType::IndexedFamily {
            element: Box::new(WireType::Matrix(matrix.clone())),
            count: IntExpr::constant(4),
        };
        let left = NodeHandle::new(
            NodeKind::Input {
                name: "zip-matrix-left".to_owned(),
                wire_type: family_type.clone(),
                artifact: None,
            },
            Vec::new(),
            vec![family_type.clone()],
        )
        .output(0)
        .expect("left matrix family input");
        let right = NodeHandle::new(
            NodeKind::Input {
                name: "zip-matrix-right".to_owned(),
                wire_type: family_type.clone(),
                artifact: None,
            },
            Vec::new(),
            vec![family_type],
        )
        .output(0)
        .expect("right matrix family input");
        let body = with_new_construction_scope(|scope| {
            let left_input = NodeHandle::new(
                NodeKind::Input {
                    name: "left".to_owned(),
                    wire_type: WireType::Matrix(matrix.clone()),
                    artifact: None,
                },
                Vec::new(),
                vec![WireType::Matrix(matrix.clone())],
            )
            .output(0)
            .expect("left matrix body input");
            let right_input = NodeHandle::new(
                NodeKind::Input {
                    name: "right".to_owned(),
                    wire_type: WireType::Matrix(matrix.clone()),
                    artifact: None,
                },
                Vec::new(),
                vec![WireType::Matrix(matrix.clone())],
            )
            .output(0)
            .expect("right matrix body input");
            let output = NodeHandle::new(
                NodeKind::MatrixBinary(MatrixBinaryOp::Add),
                vec![left_input.clone(), right_input.clone()],
                vec![WireType::Matrix(matrix.clone())],
            )
            .output(0)
            .expect("matrix body output");
            SubgraphHandle::new(
                "zip-matrix-body",
                scope,
                vec![left_input, right_input],
                vec![output],
            )
            .expect("matrix zip body")
        });
        let output_family_type = WireType::IndexedFamily {
            element: Box::new(WireType::Matrix(matrix.clone())),
            count: IntExpr::constant(3),
        };
        let family = NodeHandle::parallel_loop(
            body,
            vec![left, right],
            vec![output_family_type],
            ParallelLoop {
                count: IntExpr::constant(3),
                minimum_count: 0,
                index_slot: 0,
                bindings: Vec::new(),
                input_modes: vec![LoopInputMode::Zip, right_mode],
            },
        )
        .output(0)
        .expect("parallel matrix family output");
        let output = NodeHandle::new(
            NodeKind::FamilyGetStatic { index: IntExpr::constant(0) },
            vec![family],
            vec![WireType::Matrix(matrix)],
        )
        .output(0)
        .expect("selected matrix family output");
        let graph = mxx_ir_core::graph::Graph::freeze(
            "frozen-zip-matrix-family",
            Vec::new(),
            BTreeMap::from([(
                "output".to_owned(),
                GraphOutput { value: output, confidentiality: None },
            )]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .expect("freeze matrix zip graph")
        .0;
        let inputs = [
            ("zip-matrix-left", crate::ProtocolInputId("zip-matrix-left".to_owned())),
            ("zip-matrix-right", crate::ProtocolInputId("zip-matrix-right".to_owned())),
        ];
        (graph.clone(), graph.outputs()["output"].value, inputs.to_vec())
    }

    fn zip_matrix_protocol(
        graph: mxx_ir_core::graph::Graph,
        inputs: &[(&'static str, crate::ProtocolInputId)],
    ) -> crate::ProtocolDecl {
        let mut protocol = crate::toy_example::protocol();
        protocol.bundle.workflow.stages[0].graph = graph;
        let matrix = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        for (name, id) in inputs {
            protocol.bundle.input_contract.inputs.push(crate::InputContractEntry {
                id: id.clone(),
                name: (*name).to_owned(),
                value: InputValueContract::Family {
                    count: IntExpr::constant(4),
                    element: Box::new(InputValueContract::MatrixExact {
                        matrix_type: matrix.clone(),
                        canonical_coefficient_exclusive_upper_bound: Some(IntExpr::constant(17)),
                        is_constant_polynomial: true,
                    }),
                },
            });
            protocol.bundle.input_bindings.push(crate::ProtocolInputBinding {
                input: id.clone(),
                destinations: vec![ProtocolInputDestination::WorkflowStage {
                    stage: StageId("encrypt".to_owned()),
                    input: StageInputName((*name).to_owned()),
                }],
            });
        }
        protocol
    }

    #[test]
    fn frozen_parallel_matrix_zip_returns_a_dag_matrix() {
        let (graph, output, inputs) = zipped_parallel_matrix_graph(LoopInputMode::Zip);
        let protocol = zip_matrix_protocol(graph, &inputs);
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "frozen-zip-matrix-family".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        assert!(matches!(
            lowerer.lower_stage_wire(&StageId("encrypt".to_owned()), output),
            Ok(LoweredValue::Matrix(_))
        ));
    }

    #[test]
    fn frozen_parallel_matrix_zip_offset_returns_a_dag_matrix() {
        let (graph, output, inputs) =
            zipped_parallel_matrix_graph(LoopInputMode::ZipOffset { offset: 1 });
        let protocol = zip_matrix_protocol(graph, &inputs);
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "frozen-zip-matrix-family".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        assert!(matches!(
            lowerer.lower_stage_wire(&StageId("encrypt".to_owned()), output),
            Ok(LoweredValue::Matrix(_))
        ));
    }

    #[test]
    fn protocol_input_identity_and_bound_survive_two_nested_calls() {
        use mxx_ir_core::graph::{
            GraphOutput, NodeHandle, SubgraphHandle, with_new_construction_scope,
        };

        let matrix = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let root_input = NodeHandle::new(
            NodeKind::Input {
                name: "plaintext".to_owned(),
                wire_type: WireType::Matrix(matrix.clone()),
                artifact: None,
            },
            Vec::new(),
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .expect("root input");
        let inner = with_new_construction_scope(|scope| {
            let input = NodeHandle::new(
                NodeKind::Input {
                    name: "inner-input".to_owned(),
                    wire_type: WireType::Matrix(matrix.clone()),
                    artifact: None,
                },
                Vec::new(),
                vec![WireType::Matrix(matrix.clone())],
            )
            .output(0)
            .expect("inner input");
            SubgraphHandle::new("inner", scope, vec![input.clone()], vec![input])
                .expect("inner subgraph")
        });
        let outer = with_new_construction_scope(|scope| {
            let input = NodeHandle::new(
                NodeKind::Input {
                    name: "outer-input".to_owned(),
                    wire_type: WireType::Matrix(matrix.clone()),
                    artifact: None,
                },
                Vec::new(),
                vec![WireType::Matrix(matrix.clone())],
            )
            .output(0)
            .expect("outer input");
            let output =
                NodeHandle::subgraph_call(inner, vec![input.clone()], Vec::new(), vec![None])
                    .output(0)
                    .expect("inner call output");
            SubgraphHandle::new("outer", scope, vec![input], vec![output]).expect("outer subgraph")
        });
        let output = NodeHandle::subgraph_call(outer, vec![root_input], Vec::new(), vec![None])
            .output(0)
            .expect("outer call output");
        let graph = mxx_ir_core::graph::Graph::freeze(
            "nested-protocol-input",
            Vec::new(),
            BTreeMap::from([(
                "output".to_owned(),
                GraphOutput { value: output, confidentiality: None },
            )]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .expect("freeze nested graph")
        .0;
        let nested_inputs = graph
            .scopes()
            .values()
            .filter_map(|scope| scope.inputs().first().copied())
            .collect::<Vec<_>>();
        assert_eq!(nested_inputs.len(), 2);
        assert_eq!(nested_inputs[0], nested_inputs[1], "local wire IDs intentionally collide");
        let output = graph.outputs()["output"].value;
        let input_id = crate::ProtocolInputId::from("plaintext");
        let mut protocol = crate::toy_example::protocol();
        protocol.bundle.workflow.stages[0].graph = graph;
        protocol.bundle.input_contract.inputs.push(crate::InputContractEntry {
            id: input_id.clone(),
            name: "plaintext".to_owned(),
            value: InputValueContract::MatrixExact {
                matrix_type: matrix,
                canonical_coefficient_exclusive_upper_bound: Some(IntExpr::constant(7)),
                is_constant_polynomial: true,
            },
        });
        protocol.bundle.input_bindings.push(crate::ProtocolInputBinding {
            input: input_id.clone(),
            destinations: vec![ProtocolInputDestination::WorkflowStage {
                stage: StageId("encrypt".to_owned()),
                input: StageInputName("plaintext".to_owned()),
            }],
        });
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "nested-protocol-input".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let LoweredValue::Matrix(term) = lowerer
            .lower_stage_wire(&StageId("encrypt".to_owned()), output)
            .expect("nested protocol input")
        else {
            panic!("protocol matrix input is a DAG term")
        };
        assert!(matches!(
            &dag_atom(&lowerer, term).key.owner,
            super::super::normal_form::FactorOwner::Atomic(
                super::super::identity::AtomicSourceKey::ProtocolInput(found)
            ) if found == &input_id
        ));
        let factor = dag_atom(&lowerer, term);
        let bound = factor.matrix_bound.as_ref().expect("protocol matrix bound");
        assert_eq!(bound.coefficient_class, BoundClass::bounded(6_u8.into()));
        assert!(factor.matrix_value_metadata.is_constant_polynomial);
    }

    #[test]
    fn completed_shared_dependency_is_reused_but_an_active_back_edge_is_rejected() {
        let protocol = crate::toy_example::protocol();
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "memo-colors".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let wire = LoweringWire {
            source: WireSourceKey {
                scope: root_test_environment().occurrence,
                wire: WireRef { node: mxx_ir_core::NodeId(1), port: mxx_ir_core::Port(0) },
            },
            indices: Box::new([]),
        };
        assert!(lowerer.begin_wire(&wire).unwrap().is_none(), "white becomes active gray");
        assert!(matches!(
            lowerer.begin_wire(&wire),
            Err(LowerError::CyclicGraphDependency { wire: rejected }) if rejected == wire.source.wire
        ));
        let value = LoweredValue::Scalar(test_int(&mut lowerer, 1));
        lowerer.finish_wire(&wire, value.clone());
        assert!(matches!(
            (lowerer.begin_wire(&wire), &value),
            (Ok(Some(LoweredValue::Scalar(reused))), LoweredValue::Scalar(value)) if reused == *value
        ));
    }

    #[test]
    fn select_aligns_alpha_equivalent_shared_family_binders_without_lanes() {
        let protocol = crate::toy_example::protocol();
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "shared-select".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let scope = root_test_environment().occurrence;
        let first =
            BinderKey { loop_scope: scope.clone(), loop_node: mxx_ir_core::NodeId(1), slot: 0 };
        let second = BinderKey { loop_scope: scope, loop_node: mxx_ir_core::NodeId(2), slot: 0 };
        let family = |binder: BinderKey, representative| FamilyLoweringValue {
            element_type: ScalarSort::Int,
            storage: FamilyCoverageStorage::SharedTemplate {
                domain: family::LoopDomainKey {
                    binder: binder.clone(),
                    logical_count: 512_u16.into(),
                },
                representative,
                binder_domains: vec![family::CoverageBinderDomain {
                    binder,
                    minimum: BigInt::zero(),
                    maximum: BigInt::from(511_u16),
                }]
                .into_boxed_slice(),
            },
        };
        let first_family = family(first.clone(), test_binder(&mut lowerer, first.clone()));
        let second_family = family(second.clone(), test_binder(&mut lowerer, second));
        let aligned = lowerer
            .align_selected_shared_families(vec![first_family, second_family])
            .expect("equal family domains differ only by alpha-renamed owner");
        let (_, domain, _) = family::shared_element(&aligned[1]).unwrap();
        assert_eq!(domain.binder, first);

        let mut incompatible = aligned[1].clone();
        let FamilyCoverageStorage::SharedTemplate { domain, binder_domains, .. } =
            &mut incompatible.storage
        else {
            unreachable!()
        };
        domain.logical_count = 511_u16.into();
        binder_domains[0].maximum = BigInt::from(510_u16);
        assert!(
            lowerer.align_selected_shared_families(vec![aligned[0].clone(), incompatible]).is_err()
        );
    }

    fn protocol_trapdoor_input_fixture(
        public_contract: InputValueContract,
    ) -> (crate::ProtocolDecl, WireRef) {
        use mxx_ir_core::graph::{GraphOutput, NodeHandle};
        let matrix = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let sigma = RealExpr::FromInt(IntExpr::constant(2));
        let trapdoor_type = WireType::Trapdoor {
            matrix: matrix.clone(),
            sigma: sigma.clone(),
            gadget_base: IntExpr::constant(2),
            digit_count: IntExpr::constant(3),
            preimage_max_coefficient_bound: IntExpr::constant(5),
        };
        let public = NodeHandle::new(
            NodeKind::Input {
                name: "public".to_owned(),
                wire_type: WireType::Matrix(matrix.clone()),
                artifact: None,
            },
            Vec::new(),
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .unwrap();
        let trapdoor = NodeHandle::new(
            NodeKind::Input {
                name: "trapdoor".to_owned(),
                wire_type: trapdoor_type,
                artifact: None,
            },
            Vec::new(),
            vec![WireType::Trapdoor {
                matrix: matrix.clone(),
                sigma: sigma.clone(),
                gadget_base: IntExpr::constant(2),
                digit_count: IntExpr::constant(3),
                preimage_max_coefficient_bound: IntExpr::constant(5),
            }],
        )
        .output(0)
        .unwrap();
        let graph = mxx_ir_core::graph::Graph::freeze(
            "protocol-trapdoor-input",
            Vec::new(),
            BTreeMap::from([
                ("output".to_owned(), GraphOutput { value: trapdoor, confidentiality: None }),
                ("public".to_owned(), GraphOutput { value: public, confidentiality: None }),
            ]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .unwrap()
        .0;
        let output = graph.outputs()["output"].value;
        let mut protocol = crate::toy_example::protocol();
        let stage = StageId("encrypt".to_owned());
        protocol.bundle.workflow.stages[0].graph = graph;
        protocol.bundle.input_contract.inputs = vec![
            crate::InputContractEntry {
                id: crate::ProtocolInputId::from("public"),
                name: "public".to_owned(),
                value: public_contract,
            },
            crate::InputContractEntry {
                id: crate::ProtocolInputId::from("trapdoor"),
                name: "trapdoor".to_owned(),
                value: InputValueContract::Trapdoor {
                    matrix_type: matrix,
                    sigma,
                    gadget_base: IntExpr::constant(2),
                    digit_count: IntExpr::constant(3),
                    preimage_max_coefficient_bound: IntExpr::constant(5),
                    public_input: crate::ProtocolInputId::from("public"),
                },
            },
        ];
        protocol.bundle.input_bindings = vec![
            crate::ProtocolInputBinding {
                input: crate::ProtocolInputId::from("public"),
                destinations: vec![ProtocolInputDestination::WorkflowStage {
                    stage: stage.clone(),
                    input: StageInputName("public".to_owned()),
                }],
            },
            crate::ProtocolInputBinding {
                input: crate::ProtocolInputId::from("trapdoor"),
                destinations: vec![ProtocolInputDestination::WorkflowStage {
                    stage,
                    input: StageInputName("trapdoor".to_owned()),
                }],
            },
        ];
        (protocol, output)
    }

    #[test]
    fn protocol_trapdoor_input_uses_its_declared_public_contract() {
        let matrix = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let (protocol, output) = protocol_trapdoor_input_fixture(InputValueContract::MatrixExact {
            matrix_type: matrix,
            canonical_coefficient_exclusive_upper_bound: Some(IntExpr::constant(7)),
            is_constant_polynomial: true,
        });
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "trapdoor-input".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        let LoweredValue::Trapdoor(id) =
            lowerer.lower_stage_wire(&StageId("encrypt".to_owned()), output).unwrap()
        else {
            panic!("protocol trapdoor input")
        };
        let descriptor = lowerer.symbols.trapdoors.get(id.0).unwrap();
        assert_eq!(
            descriptor.source,
            TrapdoorSourceKey::ProtocolInput(crate::ProtocolInputId::from("trapdoor"))
        );
        assert_eq!(resolved_constant(&descriptor.gadget_base), Some(BigInt::from(2)));
        assert_eq!(resolved_constant(&descriptor.digit_count), Some(BigInt::from(3)));
        assert_eq!(resolved_constant(&descriptor.preimage_cutoff), Some(BigInt::from(5)));
        assert!(matches!(
            descriptor.public,
            super::super::identity::CanonicalTermIdentity::Source(_)
        ));
    }

    #[test]
    fn protocol_trapdoor_input_rejects_a_non_matrix_declared_public_contract() {
        let (protocol, output) = protocol_trapdoor_input_fixture(InputValueContract::Boolean);
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "trapdoor-input".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        assert!(matches!(
            lowerer.lower_stage_wire(&StageId("encrypt".to_owned()), output),
            Err(LowerError::MissingProtocolInputBinding { input })
                if input == crate::ProtocolInputId::from("public")
        ));
    }

    #[test]
    fn protocol_trapdoor_input_rejects_a_missing_declared_public_contract() {
        let matrix = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let (mut protocol, output) =
            protocol_trapdoor_input_fixture(InputValueContract::MatrixExact {
                matrix_type: matrix,
                canonical_coefficient_exclusive_upper_bound: None,
                is_constant_polynomial: true,
            });
        protocol
            .bundle
            .input_contract
            .inputs
            .retain(|entry| entry.id != crate::ProtocolInputId::from("public"));
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "trapdoor-input".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
        assert!(matches!(
            lowerer.lower_stage_wire(&StageId("encrypt".to_owned()), output),
            Err(LowerError::MissingProtocolInputBinding { input })
                if input == crate::ProtocolInputId::from("public")
        ));
    }

    #[test]
    fn deeply_nested_parallel_loops_use_one_compact_work_stack() {
        const DEPTH: usize = 96;
        const LOGICAL_COUNT: i64 = 1_000_000;
        let graph = deep_parallel_graph(DEPTH, LOGICAL_COUNT);
        let output = graph.outputs()["output"].value;
        let mut protocol = crate::toy_example::protocol();
        protocol.bundle.workflow.stages[0].graph = graph;
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "deep-stack".to_owned(),
        };
        let result = std::thread::Builder::new()
            .name("deep-lowering-stack".to_owned())
            .stack_size(1024 * 1024)
            .spawn(move || {
                let stage = StageId("encrypt".to_owned());
                let mut lowerer = GraphLowerer::new(&protocol, &request, SymbolTables::default());
                lowerer.lower_stage_wire(&stage, output).expect("deep lowering succeeds");
                (lowerer.scalar_store_len(), lowerer.symbols.binders.values.len())
            })
            .expect("constrained stack thread")
            .join()
            .expect("deep lowering thread");
        assert_eq!(result.1, DEPTH);
        assert!(
            result.0 < DEPTH * 8,
            "lowering must scale with structural nodes, not {LOGICAL_COUNT} logical lanes"
        );
    }
}
