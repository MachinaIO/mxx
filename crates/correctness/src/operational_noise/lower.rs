//! Iterative Graph-IR lowering state for the operational-noise checker.
//!
//! A lowering wire is a concrete graph occurrence plus its active symbolic coordinates.  The
//! single memo below is the only owner of graph-wire lowering results; integer ranges and
//! selector provenance remain exclusively in the e-graph analysis.

use super::{
    OperationalCheckRequest,
    analysis::{IntegerDomain, MxxAnalysis, MxxSort, ScalarProvenance},
    bound::{
        BoundClass, BoundEvaluationControl, BoundEvaluationError, BoundEvaluator, BoundInput,
        MatrixBound, MatrixMetadata, ResolvedMatrixConstant,
    },
    error::{LowerError, SelectorOnlyConsumer},
    family::{self, FamilyCoverageStorage, FamilyLoweringValue},
    identity::{
        BinderKey, OccurrenceScope, ResolvedIntExpr, SamplerDescriptorId, SamplerIdentity,
        SequentialRecurrenceDescriptor, SequentialStateKey, TrapdoorDescriptorId, TrapdoorIdentity,
        TrapdoorSourceKey, WireSourceKey,
    },
    language::MxxLang,
};
use crate::{InputValueContract, ProtocolDecl, ProtocolInputDestination, StageId, StageInputName};
use egg::{EGraph, Id};
use mxx_ir_core::{
    IntExpr, RealExpr, WireRef, WireType,
    graph::FrozenGraphScopeId,
    node::{
        ConcatAxis, IntBinaryOp, IntCompareOp, LoopInputMode, MatrixBinaryOp, NodeKind,
        ParallelLoop, RealBinaryOp, SequentialLoop,
    },
    types::MatrixType,
};
use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive, Zero};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::Arc,
};

/// Structural-family dispatch is deliberately outside ordinary expression lowering.
///
/// The resolver receives the exact occurrence and lexical environment, so it can enter a
/// child scope or select one family element without introducing a second family cache.  It must
/// return the next concrete wire to lower; the caller remains the sole memo/e-graph owner.
pub trait FamilyResolver {
    fn resolve(
        &mut self,
        wire: &LoweringWire,
        environment: &LowerEnv,
        index: Option<&LoweredInt>,
    ) -> Result<LoweringWire, LowerError>;
}

/// The lowering side of the single checker-job resource owner.  Production
/// callers pass one live implementation through [`GraphLowerer::new_with_control`];
/// direct lowering tests may use [`GraphLowerer::new`] without a budget.
pub trait LoweringControl: Send + Sync {
    fn check_deadline(&self) -> Result<(), LowerError>;
    fn reserve_owned_elements(&self, requested: usize) -> Result<(), LowerError>;
}

/// Read-only production bridge from a lowered e-graph to the bound evaluator.
/// It deliberately has no memo: `BoundEvaluator` owns computed bounds.
pub struct ProductionBoundInput<'a, 'protocol> {
    lowerer: &'a GraphLowerer<'protocol>,
    control: Option<&'a dyn BoundEvaluationControl>,
}

impl BoundInput for ProductionBoundInput<'_, '_> {
    fn node(&self, term: Id) -> Option<&MxxLang> {
        self.lowerer.egraph[self.lowerer.egraph.find(term)].nodes.first()
    }

    fn matrix_type(
        &self,
        term: Id,
    ) -> Result<mxx_ir_core::types::ConcreteMatrixType, BoundEvaluationError> {
        let term = self.lowerer.egraph.find(term);
        let Ok(MxxSort::Matrix(matrix)) = &self.lowerer.egraph[term].data.sort else {
            return Err(BoundEvaluationError::NonMatrixTerm { term });
        };
        concrete_matrix_type(matrix).ok_or(BoundEvaluationError::NonMatrixTerm { term })
    }

    fn atom_bound(
        &self,
        source: super::identity::AtomicSourceId,
        term: Id,
    ) -> Result<MatrixBound, BoundEvaluationError> {
        if self.lowerer.sequential_recurrence(source).is_some() {
            return self.evaluate_sequential_recurrence(source, term);
        }
        let descriptor = self
            .lowerer
            .egraph
            .analysis
            .symbols
            .atomic_sources
            .get(source.0)
            .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })?;
        let matrix_type = self.matrix_type(term)?;
        let coefficient_class = match &descriptor.key {
            super::identity::AtomicSourceKey::Sampler(id) => {
                let sampler = self
                    .lowerer
                    .egraph
                    .analysis
                    .symbols
                    .samplers
                    .get(id.0)
                    .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })?;
                let SamplerIdentity::Preimage { cutoff, .. } = sampler;
                let cutoff = resolved_integer(cutoff)
                    .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })?;
                let cutoff = cutoff
                    .to_biguint()
                    .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })?;
                BoundClass::Bounded { maximum_absolute_coefficient: cutoff }
            }
            // A carried-state placeholder is meaningful only inside the
            // descriptor-owned simultaneous transition overlay below.
            super::identity::AtomicSourceKey::SequentialRecurrence { .. } |
            super::identity::AtomicSourceKey::SequentialState(_) |
            super::identity::AtomicSourceKey::ProtocolInput(_) |
            super::identity::AtomicSourceKey::GraphWire(_) => BoundClass::Large,
        };
        Ok(MatrixBound { matrix_type, coefficient_class, metadata: MatrixMetadata::unknown() })
    }

    fn matrix_constant(
        &self,
        spec: super::identity::MatrixConstantSpecId,
        term: Id,
    ) -> Result<
        (mxx_ir_core::types::ConcreteMatrixType, ResolvedMatrixConstant),
        BoundEvaluationError,
    > {
        let descriptor = self
            .lowerer
            .egraph
            .analysis
            .symbols
            .matrix_constants
            .get(spec.0)
            .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })?;
        let value = match &descriptor.value {
            super::identity::MatrixConstantValue::Zero => ResolvedMatrixConstant::Zero,
            super::identity::MatrixConstantValue::Identity => ResolvedMatrixConstant::Identity,
            super::identity::MatrixConstantValue::UnitRow { index } => {
                ResolvedMatrixConstant::UnitRow {
                    index: resolved_nonnegative(index)
                        .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })?,
                }
            }
            super::identity::MatrixConstantValue::UnitColumn { index } => {
                ResolvedMatrixConstant::UnitColumn {
                    index: resolved_nonnegative(index)
                        .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })?,
                }
            }
            super::identity::MatrixConstantValue::Gadget { base, small } => {
                ResolvedMatrixConstant::Gadget {
                    base: resolved_integer(base)
                        .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })?,
                    small: *small,
                }
            }
            super::identity::MatrixConstantValue::PowerOfBase { base, exponent } => {
                ResolvedMatrixConstant::PowerOfBase {
                    base: resolved_integer(base)
                        .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })?,
                    exponent: resolved_nonnegative(exponent)
                        .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })?,
                }
            }
            super::identity::MatrixConstantValue::Rotation { exponent } => {
                ResolvedMatrixConstant::Rotation {
                    exponent: resolved_integer(exponent)
                        .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })?,
                }
            }
            super::identity::MatrixConstantValue::Polynomial { coefficients } => {
                ResolvedMatrixConstant::Polynomial {
                    coefficients: coefficients
                        .iter()
                        .map(resolved_integer)
                        .collect::<Option<Vec<_>>>()
                        .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })?
                        .into_boxed_slice(),
                }
            }
        };
        Ok((self.matrix_type(term)?, value))
    }

    fn scalar_maximum_absolute(
        &self,
        term: Id,
    ) -> Result<num_bigint::BigUint, BoundEvaluationError> {
        let term = self.lowerer.egraph.find(term);
        let domain = self.lowerer.egraph[term]
            .data
            .integer_domain
            .as_ref()
            .ok_or(BoundEvaluationError::InvalidMatrixScale { term })?
            .interval()
            .map_err(|_| BoundEvaluationError::InvalidMatrixScale { term })?;
        Ok(domain.minimum.abs().max(domain.maximum.abs()).to_biguint().expect("absolute integer"))
    }

    fn lift_constant_polynomial_class(
        &self,
        _: Id,
        input: Id,
    ) -> Result<BoundClass, BoundEvaluationError> {
        Ok(BoundClass::Bounded {
            maximum_absolute_coefficient: self.scalar_maximum_absolute(input)?,
        })
    }

    fn crt_coefficients(
        &self,
        spec: super::identity::CrtSpecId,
        term: Id,
    ) -> Result<Box<[BigInt]>, BoundEvaluationError> {
        self.lowerer
            .egraph
            .analysis
            .symbols
            .crts
            .get(spec.0)
            .and_then(|spec| {
                spec.reconstruction_coefficients
                    .iter()
                    .map(resolved_integer)
                    .collect::<Option<Vec<_>>>()
                    .map(Vec::into_boxed_slice)
            })
            .ok_or(BoundEvaluationError::InvalidCrtRecompose { term })
    }

    fn validate_integer(
        &self,
        value: &num_bigint::BigUint,
        operation: &'static str,
    ) -> Result<(), BoundEvaluationError> {
        if let Some(control) = &self.control {
            control.validate_integer_bits(value, operation)?;
        }
        Ok(())
    }
    fn validate_integer_bits(
        &self,
        value: &num_bigint::BigUint,
        operation: &'static str,
    ) -> Result<(), BoundEvaluationError> {
        if let Some(control) = self.control {
            control.validate_integer_bits(value, operation)?;
        }
        Ok(())
    }
    fn reserve_owned_elements(&self, requested: usize) -> Result<(), BoundEvaluationError> {
        if let Some(control) = self.control {
            control.reserve_owned_elements(requested)?;
        }
        Ok(())
    }
    fn check_deadline(&self) -> Result<(), BoundEvaluationError> {
        if let Some(control) = self.control {
            control.check_deadline()?;
        }
        Ok(())
    }
    fn validate_pack(&self, term: Id, bit_count: usize) -> Result<(), BoundEvaluationError> {
        if let Some(control) = self.control {
            control.validate_pack(term, bit_count)?;
        }
        Ok(())
    }
}

impl ProductionBoundInput<'_, '_> {
    /// Evaluates a graph-owned sequential descriptor without rebuilding its
    /// body or materializing any logical iteration/lane graph.  Each numeric
    /// iteration evaluates the one fixed transition with a read-only overlay
    /// of every *previous* carried bound, and commits the complete next vector
    /// only after all outputs succeed.
    fn evaluate_sequential_recurrence(
        &self,
        source: super::identity::AtomicSourceId,
        term: Id,
    ) -> Result<MatrixBound, BoundEvaluationError> {
        let (descriptor, carried_index) = self
            .lowerer
            .sequential_recurrence(source)
            .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })?;
        let count = resolved_nonnegative(&descriptor.count)
            .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })?;
        if descriptor.initial.len() != descriptor.transition.len() ||
            descriptor.transition.len() != descriptor.output_types.len() ||
            carried_index >= descriptor.transition.len()
        {
            return Err(BoundEvaluationError::InvalidMatrixConstant { term });
        }
        self.reserve_owned_elements(descriptor.transition.len())?;
        let mut state = descriptor
            .initial
            .iter()
            .map(|initial| BoundEvaluator::new(self).evaluate(*initial))
            .collect::<Result<Vec<_>, _>>()?;
        if count.is_zero() {
            return state
                .get(carried_index)
                .cloned()
                .ok_or(BoundEvaluationError::InvalidMatrixConstant { term });
        }
        let state_sources = self.sequential_state_sources(descriptor, term)?;
        let mut iteration = num_bigint::BigUint::zero();
        while iteration < count {
            self.check_deadline()?;
            self.reserve_owned_elements(descriptor.transition.len())?;
            let overlay =
                SequentialBoundInput { base: self, states: &state_sources, values: &state };
            let next = descriptor
                .transition
                .iter()
                .map(|transition| BoundEvaluator::new(&overlay).evaluate(*transition))
                .collect::<Result<Vec<_>, _>>()?;
            // The `next` vector is constructed from the unmodified `state`;
            // replacing it here is the sole simultaneous-commit boundary.
            state = next;
            iteration += num_bigint::BigUint::from(1_u8);
        }
        state
            .get(carried_index)
            .cloned()
            .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })
    }

    fn sequential_state_sources(
        &self,
        descriptor: &SequentialRecurrenceDescriptor,
        term: Id,
    ) -> Result<Vec<super::identity::AtomicSourceId>, BoundEvaluationError> {
        (0..descriptor.transition.len())
            .map(|carried_index| {
                self.lowerer
                    .egraph
                    .analysis
                    .symbols
                    .atomic_sources
                    .values
                    .iter()
                    .enumerate()
                    .find_map(|(source, descriptor_candidate)| {
                        matches!(
                            &descriptor_candidate.key,
                            super::identity::AtomicSourceKey::SequentialState(state)
                            if state.loop_scope == descriptor.loop_scope &&
                                state.loop_node == descriptor.loop_node &&
                                state.carried_index == carried_index
                        )
                        .then_some(super::identity::AtomicSourceId(source as u32))
                    })
                    .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })
            })
            .collect()
    }
}

/// A recurrence transition delegates every ordinary fact to the production
/// input, overriding only the descriptor's carried-state atoms.  It owns no
/// cache; each [`BoundEvaluator`] remains the sole owner of its bound memo.
struct SequentialBoundInput<'a, 'protocol> {
    base: &'a ProductionBoundInput<'a, 'protocol>,
    states: &'a [super::identity::AtomicSourceId],
    values: &'a [MatrixBound],
}

impl BoundInput for SequentialBoundInput<'_, '_> {
    fn node(&self, term: Id) -> Option<&MxxLang> {
        self.base.node(term)
    }
    fn matrix_type(
        &self,
        term: Id,
    ) -> Result<mxx_ir_core::types::ConcreteMatrixType, BoundEvaluationError> {
        self.base.matrix_type(term)
    }
    fn atom_bound(
        &self,
        source: super::identity::AtomicSourceId,
        term: Id,
    ) -> Result<MatrixBound, BoundEvaluationError> {
        self.states
            .iter()
            .position(|candidate| *candidate == source)
            .map(|index| self.values[index].clone())
            .map_or_else(|| self.base.atom_bound(source, term), Ok)
    }
    fn matrix_constant(
        &self,
        spec: super::identity::MatrixConstantSpecId,
        term: Id,
    ) -> Result<
        (mxx_ir_core::types::ConcreteMatrixType, ResolvedMatrixConstant),
        BoundEvaluationError,
    > {
        self.base.matrix_constant(spec, term)
    }
    fn scalar_maximum_absolute(
        &self,
        term: Id,
    ) -> Result<num_bigint::BigUint, BoundEvaluationError> {
        self.base.scalar_maximum_absolute(term)
    }
    fn lift_constant_polynomial_class(
        &self,
        term: Id,
        input: Id,
    ) -> Result<BoundClass, BoundEvaluationError> {
        self.base.lift_constant_polynomial_class(term, input)
    }
    fn crt_coefficients(
        &self,
        spec: super::identity::CrtSpecId,
        term: Id,
    ) -> Result<Box<[BigInt]>, BoundEvaluationError> {
        self.base.crt_coefficients(spec, term)
    }
    fn validate_integer(
        &self,
        value: &num_bigint::BigUint,
        operation: &'static str,
    ) -> Result<(), BoundEvaluationError> {
        self.base.validate_integer(value, operation)
    }
    fn validate_integer_bits(
        &self,
        value: &num_bigint::BigUint,
        operation: &'static str,
    ) -> Result<(), BoundEvaluationError> {
        self.base.validate_integer_bits(value, operation)
    }
    fn reserve_owned_elements(&self, requested: usize) -> Result<(), BoundEvaluationError> {
        self.base.reserve_owned_elements(requested)
    }
    fn check_deadline(&self) -> Result<(), BoundEvaluationError> {
        self.base.check_deadline()
    }
    fn validate_pack(&self, term: Id, bit_count: usize) -> Result<(), BoundEvaluationError> {
        self.base.validate_pack(term, bit_count)
    }
}

fn resolved_integer(value: &ResolvedIntExpr) -> Option<BigInt> {
    match value {
        ResolvedIntExpr::Const(value) => Some(value.clone()),
        _ => None,
    }
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
        NodeKind::ConstantMatrix { .. } |
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
        NodeKind::CrtRecompose { .. } => NodeDispatch::Ordinary,
    }
}

/// An integer term together with a stable owner-resolved expression when one exists.
///
/// Its range is deliberately absent: `AnalysisData` is the only range owner.
#[derive(Clone, Debug)]
pub struct LoweredInt {
    pub term: Id,
    pub stable_identity: Option<ResolvedIntExpr>,
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
    RuntimeTerm(Id),
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct LoweringWireKey {
    source: WireSourceKey,
    indices: Box<[FamilyIndexKey]>,
}

impl From<&LoweringWire> for LoweringWireKey {
    fn from(value: &LoweringWire) -> Self {
        Self {
            source: value.source.clone(),
            indices: value
                .indices
                .iter()
                .map(|index| {
                    index
                        .stable_identity
                        .clone()
                        .map_or(FamilyIndexKey::RuntimeTerm(index.term), FamilyIndexKey::Stable)
                })
                .collect(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum LoweredValue {
    Term(Id),
    Family(FamilyLoweringValue),
    Trapdoor(TrapdoorDescriptorId),
}

/// Lexical bindings for one concrete scope occurrence.
#[derive(Clone, Debug)]
pub struct LowerEnv {
    pub occurrence: OccurrenceScope,
    pub parameters: BTreeMap<String, ResolvedIntExpr>,
    pub binders: Vec<(BinderKey, LoweredInt)>,
    pub inputs: BTreeMap<WireRef, LoweringWire>,
    /// Sequential carried inputs are symbolic state terms rather than parent
    /// wires.  Keeping them separate preserves the ordinary input alias path.
    pub state_inputs: BTreeMap<WireRef, LoweredValue>,
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
    FinishParallelLoop {
        wire: LoweringWire,
        environment: LowerEnv,
        specification: ParallelLoop,
        output_type: WireType,
        binder: BinderKey,
        logical_count: num_bigint::BigUint,
        maximum: BigInt,
    },
    FinishSequentialLoop {
        wire: LoweringWire,
        environment: LowerEnv,
        count: ResolvedIntExpr,
        initial: Vec<Id>,
        output_types: Vec<super::identity::ResolvedMatrixType>,
        output_type: WireType,
        carried_index: usize,
        dependency_count: usize,
    },
}

/// The sole mutable owner for one lowering/rewrite job.
pub struct GraphLowerer<'a> {
    pub protocol: &'a ProtocolDecl,
    pub request: &'a OperationalCheckRequest,
    pub egraph: EGraph<MxxLang, MxxAnalysis>,
    memo: HashMap<LoweringWireKey, LoweredValue>,
    active: HashSet<LoweringWireKey>,
    control: Option<Arc<dyn LoweringControl>>,
}

impl<'a> GraphLowerer<'a> {
    pub fn new(
        protocol: &'a ProtocolDecl,
        request: &'a OperationalCheckRequest,
        analysis: MxxAnalysis,
    ) -> Self {
        Self {
            protocol,
            request,
            egraph: EGraph::new(analysis),
            memo: HashMap::new(),
            active: HashSet::new(),
            control: None,
        }
    }
}

impl<'a> GraphLowerer<'a> {
    /// Constructs a production lowerer with the job-wide control bridge.
    /// This bridge is retained by nested parallel and sequential body walks,
    /// rather than being recreated per lexical scope.
    pub fn new_with_control(
        protocol: &'a ProtocolDecl,
        request: &'a OperationalCheckRequest,
        analysis: MxxAnalysis,
        control: Arc<dyn LoweringControl>,
    ) -> Self {
        Self {
            protocol,
            request,
            egraph: EGraph::new(analysis),
            memo: HashMap::new(),
            active: HashSet::new(),
            control: Some(control),
        }
    }

    /// Reads the canonical e-class facts without creating a second evaluator or cache.
    pub fn integer_analysis(&self, term: Id) -> Option<(&IntegerDomain, ScalarProvenance)> {
        let data = &self.egraph[self.egraph.find(term)].data;
        if data.sort != Ok(MxxSort::Int) {
            return None;
        }
        Some((data.integer_domain.as_ref()?, data.scalar_provenance?))
    }

    /// Enforces the closed selector-only consumer table at every scalar use-site.
    pub fn validate_integer_consumer(
        &self,
        term: Id,
        consumer: SelectorOnlyConsumer,
        selector_allowed: bool,
    ) -> Result<(), LowerError> {
        let Some((_, provenance)) = self.integer_analysis(term) else {
            return Err(LowerError::InvalidOperandSort {
                expected: mxx_ir_core::WireType::Int,
                actual: mxx_ir_core::WireType::Int,
            });
        };
        if provenance == ScalarProvenance::SelectorOnly && !selector_allowed {
            return Err(LowerError::SelectorOnlyValueUsedByForbiddenConsumer { consumer });
        }
        Ok(())
    }

    /// Applies one authoritative call-boundary range annotation and immediately unions it with
    /// the matrix child.  No parallel metadata table is retained.
    pub fn attach_canonical_range(
        &mut self,
        child: Id,
        upper: num_bigint::BigUint,
    ) -> Result<Id, LowerError> {
        if upper == num_bigint::BigUint::from(0_u8) ||
            !matches!(self.egraph[self.egraph.find(child)].data.sort, Ok(MxxSort::Matrix(_)))
        {
            return Err(LowerError::InvalidInternalCanonicalRangeContract {
                upper,
                modulus: num_bigint::BigUint::from(0_u8),
            });
        }
        let annotation =
            self.egraph.add(MxxLang::MatrixCanonicalRangeContract { upper, input: [child] });
        self.egraph.union(annotation, child);
        Ok(annotation)
    }

    /// Begins one memoized wire lowering.  A repeated active key is a graph dependency cycle;
    /// completed keys return their one stored result without repeating graph work.
    pub fn begin_wire(&mut self, wire: &LoweringWire) -> Result<Option<LoweredValue>, LowerError> {
        if let Some(control) = &self.control {
            control.check_deadline()?;
            control.reserve_owned_elements(1)?;
        }
        let key = LoweringWireKey::from(wire);
        if let Some(value) = self.memo.get(&key) {
            return Ok(Some(value.clone()));
        }
        if !self.active.insert(key) {
            return Err(LowerError::CyclicGraphDependency { wire: wire.source.wire });
        }
        Ok(None)
    }

    pub fn finish_wire(&mut self, wire: &LoweringWire, value: LoweredValue) {
        let key = LoweringWireKey::from(wire);
        self.active.remove(&key);
        self.memo.insert(key, value);
    }

    /// Exposes the graph-work accounting used by count-independence fixtures.
    pub fn lowered_wire_count(&self) -> usize {
        self.memo.len()
    }

    /// Builds the complete relation registry from the exact sampler descriptors
    /// owned by this lowerer's e-graph.  No caller may reconstruct a sampler
    /// relation from a source node number or a cutoff estimate.
    pub fn relation_registrations(&self) -> Vec<super::relation::RelationRegistration> {
        self.egraph
            .analysis
            .symbols
            .samplers
            .values
            .iter()
            .enumerate()
            .map(|(id, sampler)| match sampler {
                SamplerIdentity::Preimage { public, trapdoor, target, indices, .. } => {
                    super::relation::RelationRegistration {
                        source: super::identity::AtomicSourceId(
                            self.egraph
                                .analysis
                                .symbols
                                .atomic_sources
                                .values
                                .iter()
                                .position(|source| {
                                    matches!(
                                        source.key,
                                        super::identity::AtomicSourceKey::Sampler(
                                            super::identity::SamplerDescriptorId(source_id)
                                        ) if source_id == id as u32
                                    )
                                })
                                .expect("lowered sampler has an atom source")
                                as u32,
                        ),
                        expected_public: *public,
                        target: *target,
                        trapdoor: Some(*trapdoor),
                        indices: indices.clone(),
                    }
                }
            })
            .collect()
    }

    /// Returns the one production view used by the bound evaluator.  It reads
    /// canonical e-graph analysis and exact lowering descriptors only.
    pub fn production_bound_view(&self) -> ProductionBoundInput<'_, 'a> {
        ProductionBoundInput { lowerer: self, control: None }
    }

    /// Constructs the production evaluator view with the caller's single
    /// job-wide resource owner.  Checker execution must use this entry point;
    /// the control-free view remains only for direct, deterministic unit tests.
    pub fn production_bound_view_with_control<'b>(
        &'b self,
        control: &'b dyn BoundEvaluationControl,
    ) -> ProductionBoundInput<'b, 'a> {
        ProductionBoundInput { lowerer: self, control: Some(control) }
    }

    /// Returns the complete, graph-owned recurrence for one compact sequential
    /// output.  Consumers receive the descriptor and selected carried slot
    /// together, rather than re-identifying a loop from a node number or
    /// replaying the loop body.  In particular, this API never expands the
    /// count into iterations or lanes.
    pub fn sequential_recurrence(
        &self,
        source: super::identity::AtomicSourceId,
    ) -> Option<(&SequentialRecurrenceDescriptor, usize)> {
        let descriptor = self.egraph.analysis.symbols.atomic_sources.get(source.0)?;
        let super::identity::AtomicSourceKey::SequentialRecurrence { recurrence, carried_index } =
            descriptor.key
        else {
            return None;
        };
        self.egraph
            .analysis
            .symbols
            .sequential_recurrences
            .get(recurrence.0)
            .map(|descriptor| (descriptor, carried_index))
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
            if let Some(control) = &self.control {
                control.check_deadline()?;
            }
            match frame {
                LoweringFrame::Enter { wire, environment } => {
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
                    if let Some(value) = environment.state_inputs.get(&wire.source.wire) {
                        self.finish_wire(&wire, value.clone());
                        values.push(value.clone());
                        continue;
                    }
                    if let Some(bound) = environment.inputs.get(&wire.source.wire) {
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
                                                scope: environment.occurrence.clone(),
                                                wire: argument,
                                            },
                                            indices: Box::new([]),
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
                                let public = LoweringWire {
                                    source: WireSourceKey {
                                        scope: wire.source.scope.clone(),
                                        wire: WireRef {
                                            node: wire.source.wire.node,
                                            port: mxx_ir_core::Port(0),
                                        },
                                    },
                                    indices: wire.indices.clone(),
                                };
                                let public_atom = self.atom_for_wire(
                                    &public,
                                    &environment,
                                    WireType::Matrix(matrix_type.clone()),
                                    None,
                                )?;
                                let public = self.egraph.add(public_atom);
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
                                        .map(|coordinate| coordinate.index.term)
                                        .collect(),
                                    matrix_type: self
                                        .resolve_matrix_type(&matrix_type, &environment)?,
                                    public,
                                    sigma_bits: self.resolve_real(&sigma, &environment)?.to_bits(),
                                    gadget_base: self.resolve_int(&gadget_base, &environment)?,
                                    digit_count: self.resolve_int(&digit_count, &environment)?,
                                    preimage_cutoff: self.resolve_int(
                                        &preimage_max_coefficient_bound,
                                        &environment,
                                    )?,
                                };
                                let descriptor =
                                    self.egraph.analysis.symbols.trapdoors.intern(descriptor);
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
                            let atom = self.atom_for_wire(
                                &wire,
                                &environment,
                                node.output_types()[wire.source.wire.port.0 as usize].clone(),
                                role,
                            )?;
                            let value = LoweredValue::Term(self.egraph.add(atom));
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
                                            scope: environment.occurrence.clone(),
                                            wire: argument,
                                        },
                                        indices: Box::new([]),
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
                                child.inputs = child_inputs
                                    .iter()
                                    .copied()
                                    .zip(arguments)
                                    .map(|(input, argument)| {
                                        (
                                            input,
                                            LoweringWire {
                                                source: WireSourceKey {
                                                    scope: environment.occurrence.clone(),
                                                    wire: argument,
                                                },
                                                indices: Box::new([]),
                                            },
                                        )
                                    })
                                    .collect();
                                child.state_inputs.clear();
                                let parameter_bindings = call.bindings.clone();
                                for (name, expression) in &parameter_bindings {
                                    let value = self
                                        .lower_int_expr(expression, &environment)?
                                        .stable_identity
                                        .ok_or_else(|| LowerError::NonExactIdentityIndex {
                                            expression: expression.clone(),
                                        })?;
                                    child.parameters.insert(name.clone(), value);
                                }
                                let output = *child_outputs
                                    .get(wire.source.wire.port.0 as usize)
                                    .ok_or(LowerError::InvalidOutputPort {
                                    wire: wire.source.wire,
                                    output_count: child_outputs.len(),
                                })?;
                                work.push(LoweringFrame::FinishAlias { wire });
                                work.push(LoweringFrame::Enter {
                                    wire: LoweringWire {
                                        source: WireSourceKey {
                                            scope: child.occurrence.clone(),
                                            wire: output,
                                        },
                                        indices: Box::new([]),
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
                                                scope: environment.occurrence.clone(),
                                                wire: argument,
                                            },
                                            indices: Box::new([]),
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
                    let value = self.lower_node(&kind, &arguments, &environment)?;
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
                    let value =
                        self.lower_structural_node(&kind, &arguments, &environment, output_type)?;
                    self.finish_wire(&wire, value.clone());
                    values.push(value);
                }
                LoweringFrame::FinishParallelLoop {
                    wire,
                    environment,
                    specification,
                    output_type,
                    binder,
                    logical_count,
                    maximum,
                } => {
                    let LoweredValue::Term(representative) =
                        values.pop().ok_or(LowerError::MissingWire { wire: wire.source.wire })?
                    else {
                        return Err(LowerError::NonUniformParallelMatrixType {
                            expected: output_type,
                            actual: WireType::Int,
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
                    )?;
                    self.finish_wire(&wire, value.clone());
                    values.push(value);
                }
                LoweringFrame::FinishSequentialLoop {
                    wire,
                    environment,
                    count,
                    initial,
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
                    let transitions = values
                        .split_off(values.len() - dependency_count)
                        .into_iter()
                        .map(|value| match value {
                            LoweredValue::Term(term) => Ok(term),
                            _ => Err(LowerError::InvalidOperandArity {
                                expected: dependency_count,
                                actual: dependency_count,
                            }),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let value = self.finish_sequential_loop(
                        &wire,
                        &environment,
                        count,
                        initial,
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
                LoweringFrame::FinishValue { wire, value } => {
                    self.finish_wire(&wire, value.clone());
                    values.push(value);
                }
                LoweringFrame::FinishIndexedAlias { wire, index } => {
                    let value =
                        values.pop().ok_or(LowerError::MissingWire { wire: wire.source.wire })?;
                    let LoweredValue::Family(family) = value else {
                        return Err(LowerError::FamilyProducerNotResolved {
                            family: wire.source.wire,
                        });
                    };
                    let value = self.family_element(&family, &index)?;
                    self.finish_wire(&wire, value.clone());
                    values.push(value);
                }
                LoweringFrame::FinishPreimage { wire, environment, cutoff, output_type } => {
                    let arguments = values.split_off(values.len().checked_sub(3).ok_or(
                        LowerError::InvalidOperandArity { expected: 3, actual: values.len() },
                    )?);
                    let [
                        LoweredValue::Term(public),
                        LoweredValue::Trapdoor(trapdoor),
                        LoweredValue::Term(target),
                    ] = arguments.as_slice()
                    else {
                        return Err(LowerError::InvalidOperandArity {
                            expected: 3,
                            actual: arguments.len(),
                        });
                    };
                    let (WireType::Matrix(matrix) | WireType::Preimage(matrix)) = output_type
                    else {
                        return Err(LowerError::InvalidOperandSort {
                            expected: WireType::Matrix(MatrixType {
                                modulus: IntExpr::constant(1),
                                ring_dimension: IntExpr::constant(1),
                                rows: IntExpr::constant(1),
                                columns: IntExpr::constant(1),
                            }),
                            actual: output_type,
                        });
                    };
                    let resolved_cutoff = self.resolve_int(&cutoff, &environment)?;
                    let resolved_matrix = self.resolve_matrix_type(&matrix, &environment)?;
                    let sampler =
                        self.egraph.analysis.symbols.samplers.intern(SamplerIdentity::Preimage {
                            source: super::identity::GraphWireSourceKey {
                                wire: wire.source.clone(),
                                coordinate_binders: environment
                                    .active_coordinates
                                    .iter()
                                    .map(|coordinate| coordinate.binder.clone())
                                    .collect(),
                            },
                            indices: environment
                                .active_coordinates
                                .iter()
                                .map(|coordinate| coordinate.index.term)
                                .collect(),
                            public: *public,
                            trapdoor: *trapdoor,
                            target: *target,
                            cutoff: resolved_cutoff,
                        });
                    let source = self.egraph.analysis.symbols.atomic_sources.intern(
                        super::identity::AtomicSourceDescriptor {
                            key: super::identity::AtomicSourceKey::Sampler(SamplerDescriptorId(
                                sampler,
                            )),
                            sort: MxxSort::Matrix(resolved_matrix),
                            integer_domain: None,
                            canonical_residue_convention: None,
                            relation_role: Some(super::identity::AtomicRelationRole::Preimage),
                        },
                    );
                    let value = LoweredValue::Term(
                        self.egraph.add(MxxLang::Atom {
                            source: super::identity::AtomicSourceId(source),
                            indices: environment
                                .active_coordinates
                                .iter()
                                .map(|coordinate| coordinate.index.term)
                                .collect(),
                        }),
                    );
                    self.finish_wire(&wire, value.clone());
                    values.push(value);
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

    fn atom_for_wire(
        &mut self,
        wire: &LoweringWire,
        environment: &LowerEnv,
        ty: WireType,
        relation_role: Option<super::identity::AtomicRelationRole>,
    ) -> Result<MxxLang, LowerError> {
        let sort = match ty {
            WireType::Int | WireType::ConstantInt => MxxSort::Int,
            WireType::Bool | WireType::ConstantBool => MxxSort::Bool,
            WireType::Real | WireType::ConstantReal => MxxSort::Real,
            WireType::Bytes { length } => MxxSort::Bytes(self.resolve_int(&length, environment)?),
            WireType::TypedBlob { type_name, schema_hash } => {
                MxxSort::TypedBlob { type_name, schema_hash }
            }
            WireType::Matrix(matrix) | WireType::Preimage(matrix) => {
                MxxSort::Matrix(self.resolve_matrix_type(&matrix, environment)?)
            }
            WireType::Trapdoor { .. } | WireType::IndexedFamily { .. } => {
                return Err(LowerError::FamilyProducerNotResolved { family: wire.source.wire });
            }
        };
        let (key, integer_domain) =
            self.protocol_input_source(wire, environment, &sort)?.unwrap_or_else(|| {
                (
                    super::identity::AtomicSourceKey::GraphWire(
                        super::identity::GraphWireSourceKey {
                            wire: wire.source.clone(),
                            coordinate_binders: environment
                                .active_coordinates
                                .iter()
                                .map(|coordinate| coordinate.binder.clone())
                                .collect(),
                        },
                    ),
                    None,
                )
            });
        let descriptor = super::identity::AtomicSourceDescriptor {
            key,
            sort,
            integer_domain,
            canonical_residue_convention: None,
            relation_role,
        };
        let source = self.egraph.analysis.symbols.atomic_sources.intern(descriptor);
        Ok(MxxLang::Atom {
            source: super::identity::AtomicSourceId(source),
            indices: environment
                .active_coordinates
                .iter()
                .map(|coordinate| coordinate.index.term)
                .collect(),
        })
    }

    /// Normalizes a non-artifact root-stage input to its closed protocol input identity.  The
    /// graph's local input name is never used as an analysis identity.
    fn protocol_input_source(
        &mut self,
        wire: &LoweringWire,
        environment: &LowerEnv,
        sort: &MxxSort,
    ) -> Result<
        Option<(super::identity::AtomicSourceKey, Option<super::identity::IntegerSourceDomain>)>,
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
        let integer_domain = match (&contract.value, sort) {
            (InputValueContract::IntegerRange { lower, upper }, MxxSort::Int) => {
                let lower = self.resolve_int(lower, environment)?;
                let upper = self.resolve_int(upper, environment)?;
                let (ResolvedIntExpr::Const(minimum), ResolvedIntExpr::Const(maximum)) =
                    (lower, upper)
                else {
                    return Err(LowerError::UnboundParameter { parameter: contract.name.clone() });
                };
                Some(super::identity::IntegerSourceDomain { minimum, maximum })
            }
            _ => None,
        };
        Ok(Some((super::identity::AtomicSourceKey::ProtocolInput(input), integer_domain)))
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
            if let Some(control) = &self.control {
                control.check_deadline()?;
                control.reserve_owned_elements(1)?;
            }
            match frame {
                Frame::Enter(
                    value @ (IntExpr::Const(_) | IntExpr::Var(_) | IntExpr::LoopIndex(_)),
                ) => {
                    let lowered = match value {
                        IntExpr::Const(value) => {
                            self.add_int(value.clone(), ResolvedIntExpr::Const(value.clone()))
                        }
                        IntExpr::Var(name) => {
                            let value = environment.parameters.get(name).ok_or_else(|| {
                                LowerError::UnboundParameter { parameter: name.clone() }
                            })?;
                            self.add_resolved_int(value.clone())
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
                            let binder_id = self.egraph.analysis.symbols.binders.intern(
                                super::identity::BinderDescriptor {
                                    key: binder.clone(),
                                    minimum: self
                                        .integer_analysis(value.term)
                                        .and_then(|(domain, _)| domain.interval().ok())
                                        .map_or_else(BigInt::zero, |range| range.minimum),
                                    maximum: self
                                        .integer_analysis(value.term)
                                        .and_then(|(domain, _)| domain.interval().ok())
                                        .map_or_else(BigInt::zero, |range| range.maximum),
                                },
                            );
                            LoweredInt {
                                term: self
                                    .egraph
                                    .add(MxxLang::IntBinder(super::identity::BinderId(binder_id))),
                                stable_identity: Some(ResolvedIntExpr::Binder(binder.clone())),
                            }
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
                        IntExpr::Add(_, _) => {
                            self.combine_int(children, MxxLang::IntAdd, ResolvedIntExpr::Add)?
                        }
                        IntExpr::Sub(_, _) => {
                            self.combine_int(children, MxxLang::IntSub, ResolvedIntExpr::Sub)?
                        }
                        IntExpr::Mul(_, _) => {
                            self.combine_int(children, MxxLang::IntMul, ResolvedIntExpr::Mul)?
                        }
                        IntExpr::Div(_, _) => {
                            self.combine_int(children, MxxLang::IntExactDiv, ResolvedIntExpr::Div)?
                        }
                        IntExpr::RoundDiv(_, _) => self.combine_int(
                            children,
                            MxxLang::IntRoundDiv,
                            ResolvedIntExpr::RoundDiv,
                        )?,
                        IntExpr::Log2Ceil(_) => {
                            let child = children.into_iter().next().expect("one child");
                            let term = self.egraph.add(MxxLang::IntLog2Ceil([child.term]));
                            LoweredInt {
                                term,
                                stable_identity: child
                                    .stable_identity
                                    .map(|value| ResolvedIntExpr::Log2Ceil(Box::new(value))),
                            }
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

    fn add_int(&mut self, value: BigInt, identity: ResolvedIntExpr) -> LoweredInt {
        LoweredInt {
            term: self.egraph.add(MxxLang::IntConst(value)),
            stable_identity: Some(identity),
        }
    }

    fn add_resolved_int(&mut self, value: ResolvedIntExpr) -> LoweredInt {
        match value {
            ResolvedIntExpr::Const(value) => {
                self.add_int(value.clone(), ResolvedIntExpr::Const(value))
            }
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
                value
                    .map(|value| self.add_int(value.clone(), ResolvedIntExpr::Const(value)))
                    .unwrap_or_else(|| LoweredInt {
                        term: self.egraph.add(MxxLang::IntParameter(name.clone())),
                        stable_identity: Some(ResolvedIntExpr::Parameter(name)),
                    })
            }
            other => LoweredInt {
                term: self.egraph.add(MxxLang::IntParameter(format!("{other:?}"))),
                stable_identity: Some(other),
            },
        }
    }

    fn combine_int(
        &mut self,
        children: Vec<LoweredInt>,
        node: impl FnOnce([Id; 2]) -> MxxLang,
        identity: impl FnOnce(Box<ResolvedIntExpr>, Box<ResolvedIntExpr>) -> ResolvedIntExpr,
    ) -> Result<LoweredInt, LowerError> {
        let [left, right]: [LoweredInt; 2] =
            children.try_into().map_err(|values: Vec<LoweredInt>| {
                LowerError::InvalidOperandArity { expected: 2, actual: values.len() }
            })?;
        let term = self.egraph.add(node([left.term, right.term]));
        Ok(LoweredInt {
            term,
            stable_identity: left
                .stable_identity
                .zip(right.stable_identity)
                .map(|(left, right)| identity(Box::new(left), Box::new(right))),
        })
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
        let terms = |expected: usize| -> Result<Vec<Id>, LowerError> {
            if arguments.len() != expected {
                return Err(LowerError::InvalidOperandArity { expected, actual: arguments.len() });
            }
            arguments
                .iter()
                .map(|value| match value {
                    LoweredValue::Term(term) => Ok(*term),
                    LoweredValue::Trapdoor(_) => Err(LowerError::InvalidOperandSort {
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
                    }),
                    LoweredValue::Family(_) => {
                        Err(LowerError::InvalidOperandArity { expected, actual: arguments.len() })
                    }
                })
                .collect()
        };
        let term = match kind {
            NodeKind::ConstantInt(value) => self.egraph.add(MxxLang::IntConst(value.clone())),
            NodeKind::EvaluateInt(value) => {
                return Ok(LoweredValue::Term(self.lower_int_expr(value, environment)?.term))
            }
            NodeKind::ConstantBool(value) => self.egraph.add(MxxLang::BoolConst(*value)),
            NodeKind::ConstantReal(value) => self
                .egraph
                .add(MxxLang::RealConst(self.resolve_real(value, environment)?.to_bits())),
            NodeKind::IntBinary(operation) => {
                let values = terms(2)?;
                match operation {
                    IntBinaryOp::Add => self.egraph.add(MxxLang::IntAdd([values[0], values[1]])),
                    IntBinaryOp::Subtract => {
                        self.egraph.add(MxxLang::IntSub([values[0], values[1]]))
                    }
                    IntBinaryOp::Multiply => {
                        self.egraph.add(MxxLang::IntMul([values[0], values[1]]))
                    }
                    IntBinaryOp::Divide => {
                        self.egraph.add(MxxLang::IntEuclideanDiv([values[0], values[1]]))
                    }
                    IntBinaryOp::Remainder => {
                        self.egraph.add(MxxLang::IntEuclideanRemainder([values[0], values[1]]))
                    }
                }
            }
            NodeKind::IntCompare(operation) => {
                let values = terms(2)?;
                match operation {
                    IntCompareOp::Equal => {
                        self.egraph.add(MxxLang::IntEqual([values[0], values[1]]))
                    }
                    IntCompareOp::Less => self.egraph.add(MxxLang::IntLess([values[0], values[1]])),
                    IntCompareOp::LessEqual => {
                        self.egraph.add(MxxLang::IntLessEqual([values[0], values[1]]))
                    }
                }
            }
            NodeKind::BitExtract { bit } => {
                let input = terms(1)?[0];
                let bit = self.resolve_int(bit, environment)?;
                self.egraph.add(MxxLang::BitExtract { bit, input: [input] })
            }
            NodeKind::BoolToInt => {
                let values = terms(1)?;
                self.validate_integer_consumer(values[0], SelectorOnlyConsumer::BoolToInt, false)?;
                self.egraph.add(MxxLang::BoolToInt([values[0]]))
            }
            NodeKind::IntToReal => {
                let values = terms(1)?;
                self.validate_integer_consumer(values[0], SelectorOnlyConsumer::IntToReal, false)?;
                self.egraph.add(MxxLang::IntToReal([values[0]]))
            }
            NodeKind::RealBinary(operation) => {
                let values = terms(2)?;
                match operation {
                    RealBinaryOp::Add => self.egraph.add(MxxLang::RealAdd([values[0], values[1]])),
                    RealBinaryOp::Subtract => {
                        self.egraph.add(MxxLang::RealSub([values[0], values[1]]))
                    }
                    RealBinaryOp::Multiply => {
                        self.egraph.add(MxxLang::RealMul([values[0], values[1]]))
                    }
                    RealBinaryOp::Divide => {
                        self.egraph.add(MxxLang::RealDiv([values[0], values[1]]))
                    }
                }
            }
            NodeKind::RealSqrt => self.egraph.add(MxxLang::RealSqrt([terms(1)?[0]])),
            NodeKind::MatrixBinary(operation) => {
                let values = terms(2)?;
                match operation {
                    MatrixBinaryOp::Add => {
                        self.egraph.add(MxxLang::MatrixAdd(values.into_boxed_slice()))
                    }
                    MatrixBinaryOp::Subtract => {
                        let negate = self.egraph.add(MxxLang::MatrixNegate([values[1]]));
                        self.egraph
                            .add(MxxLang::MatrixAdd(vec![values[0], negate].into_boxed_slice()))
                    }
                    MatrixBinaryOp::Multiply => {
                        self.egraph.add(MxxLang::MatrixMultiply(values.into_boxed_slice()))
                    }
                }
            }
            NodeKind::MatrixNegate => self.egraph.add(MxxLang::MatrixNegate([terms(1)?[0]])),
            NodeKind::MatrixScale { scalar } => {
                let matrix = terms(1)?[0];
                let scalar = self.lower_int_expr(scalar, environment)?;
                self.validate_integer_consumer(
                    scalar.term,
                    SelectorOnlyConsumer::MatrixScale,
                    false,
                )?;
                self.egraph.add(MxxLang::MatrixScale([matrix, scalar.term]))
            }
            NodeKind::Transpose => self.egraph.add(MxxLang::MatrixTranspose([terms(1)?[0]])),
            NodeKind::Slice { rows, columns } => {
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
                let id = self.egraph.analysis.symbols.slices.intern(spec);
                self.egraph.add(MxxLang::MatrixSlice {
                    spec: super::identity::SliceSpecId(id),
                    input: [terms(1)?[0]],
                })
            }
            NodeKind::Tensor => {
                let values = terms(2)?;
                self.egraph.add(MxxLang::MatrixTensor([values[0], values[1]]))
            }
            NodeKind::Concat { axis } => {
                let values = arguments
                    .iter()
                    .map(|value| match value {
                        LoweredValue::Term(value) => Ok(*value),
                        LoweredValue::Trapdoor(_) => {
                            Err(LowerError::InvalidOperandArity { expected: 0, actual: 1 })
                        }
                        LoweredValue::Family(_) => {
                            Err(LowerError::InvalidOperandArity { expected: 0, actual: 1 })
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let axis = match axis {
                    ConcatAxis::Rows => super::identity::Axis::Rows,
                    ConcatAxis::Columns => super::identity::Axis::Columns,
                    ConcatAxis::Diagonal => super::identity::Axis::Diagonal,
                };
                self.egraph.add(MxxLang::MatrixConcat { axis, inputs: values.into_boxed_slice() })
            }
            NodeKind::ExtractCoefficient { position } => {
                let matrix = terms(1)?[0];
                let position = self.lower_int_expr(position, environment)?;
                self.egraph.add(MxxLang::ExtractCoefficient([matrix, position.term]))
            }
            NodeKind::LiftIntegerToConstantPolynomial { matrix_type: ty } => {
                let input = terms(1)?[0];
                self.validate_integer_consumer(
                    input,
                    SelectorOnlyConsumer::LiftConstantPolynomial,
                    false,
                )?;
                let matrix_type = self.resolve_matrix_type(ty, environment)?;
                self.egraph.add(MxxLang::LiftConstantPolynomial { matrix_type, input: [input] })
            }
            NodeKind::CrtRecompose { plaintext_moduli, reconstruction_coefficients } => {
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
                let id = self.egraph.analysis.symbols.crts.intern(spec);
                self.egraph.add(MxxLang::CrtRecompose {
                    spec: super::identity::CrtSpecId(id),
                    inputs: arguments
                        .iter()
                        .map(|value| match value {
                            LoweredValue::Term(value) => Ok(*value),
                            LoweredValue::Trapdoor(_) => {
                                Err(LowerError::InvalidOperandArity { expected: 0, actual: 1 })
                            }
                            LoweredValue::Family(_) => {
                                Err(LowerError::InvalidOperandArity { expected: 0, actual: 1 })
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()?
                        .into_boxed_slice(),
                })
            }
            NodeKind::Input { .. } |
            NodeKind::ConstantMatrix { .. } |
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
        };
        Ok(LoweredValue::Term(term))
    }

    fn lower_structural_node(
        &mut self,
        kind: &NodeKind,
        arguments: &[LoweredValue],
        environment: &LowerEnv,
        output_type: WireType,
    ) -> Result<LoweredValue, LowerError> {
        match kind {
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
                let WireType::Matrix(matrix) = *element else {
                    return Err(LowerError::FamilyElementTypeMismatch {
                        expected: WireType::Matrix(MatrixType {
                            modulus: IntExpr::constant(1),
                            ring_dimension: IntExpr::constant(1),
                            rows: IntExpr::constant(1),
                            columns: IntExpr::constant(1),
                        }),
                        actual: *element,
                    });
                };
                let elements = arguments
                    .iter()
                    .map(|argument| match argument {
                        LoweredValue::Term(term) => Ok(*term),
                        LoweredValue::Family(_) | LoweredValue::Trapdoor(_) => {
                            Err(LowerError::FamilyElementTypeMismatch {
                                expected: WireType::Matrix(matrix.clone()),
                                actual: WireType::Int,
                            })
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let element_type = self.concrete_matrix_type(&matrix, environment)?;
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
                let [LoweredValue::Family(family)] = arguments else {
                    return Err(LowerError::InvalidOperandArity {
                        expected: 1,
                        actual: arguments.len(),
                    });
                };
                let index = self.lower_int_expr(index, environment)?;
                if let Some(element) = family::static_get(family, &index).map_err(|_| {
                    LowerError::FamilyAccessOutOfRange {
                        index: index
                            .clone()
                            .stable_identity
                            .and_then(|value| match value {
                                ResolvedIntExpr::Const(value) => Some(IntExpr::constant(value)),
                                _ => None,
                            })
                            .unwrap_or_else(|| IntExpr::constant(-1)),
                        count: IntExpr::constant(0),
                    }
                })? {
                    return Ok(LoweredValue::Term(element));
                }
                self.shared_family_element(family, &index)
            }
            NodeKind::FamilyGetDynamic => {
                let [LoweredValue::Family(family), LoweredValue::Term(selector)] = arguments else {
                    return Err(LowerError::InvalidOperandArity {
                        expected: 2,
                        actual: arguments.len(),
                    });
                };
                match &family.storage {
                    FamilyCoverageStorage::ExactStored { elements } => {
                        let term = family::dynamic_get(&mut self.egraph, family, *selector)
                            .map_err(|_| LowerError::InvalidFamilyCount {
                                count: IntExpr::constant(elements.len()),
                            })?;
                        Ok(LoweredValue::Term(term))
                    }
                    FamilyCoverageStorage::SharedTemplate { .. } => self.shared_family_element(
                        family,
                        &LoweredInt { term: *selector, stable_identity: None },
                    ),
                }
            }
            NodeKind::Select { .. } => {
                let Some((LoweredValue::Term(selector), cases)) = arguments.split_first() else {
                    return Err(LowerError::InvalidOperandArity {
                        expected: 2,
                        actual: arguments.len(),
                    });
                };
                let families = cases.iter().cloned().collect::<Vec<_>>();
                if families.iter().all(|value| matches!(value, LoweredValue::Family(_))) {
                    let families = families
                        .into_iter()
                        .map(|value| match value {
                            LoweredValue::Family(family) => family,
                            _ => unreachable!(),
                        })
                        .collect::<Vec<_>>();
                    return family::select_family(&mut self.egraph, *selector, &families)
                        .map(LoweredValue::Family)
                        .map_err(|_| LowerError::IncompatibleFamilyCoverage {
                            expected: output_type.clone(),
                            actual: output_type,
                        });
                }
                let terms = cases
                    .iter()
                    .map(|value| match value {
                        LoweredValue::Term(term) => Ok(*term),
                        _ => Err(LowerError::IncompatibleFamilyCoverage {
                            expected: output_type.clone(),
                            actual: output_type.clone(),
                        }),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(LoweredValue::Term(
                    self.egraph
                        .add(MxxLang::Switch(std::iter::once(*selector).chain(terms).collect())),
                ))
            }
            NodeKind::ParallelLoop(_) | NodeKind::SequentialLoop(_) => {
                unreachable!("loop lowering is scheduled on the outer continuation stack")
            }
            NodeKind::PackPolynomialCoefficients { .. } => {
                Err(LowerError::PackRequiresExplicitBooleanFamily { actual: output_type })
            }
            NodeKind::SubgraphCall(_) | NodeKind::ThresholdDecode { .. } => unreachable!(),
            _ => unreachable!("only structural nodes reach structural lowering"),
        }
    }

    fn shared_family_element(
        &self,
        family: &FamilyLoweringValue,
        _index: &LoweredInt,
    ) -> Result<LoweredValue, LowerError> {
        let (representative, _, _) = family::shared_element(family)
            .map_err(|_| LowerError::InvalidFamilyCount { count: IntExpr::constant(0) })?;
        // The parallel-loop frame installs the requested index as the body binder before
        // constructing this representative, so no logical family lane is enumerated here.
        Ok(LoweredValue::Term(representative))
    }

    fn family_element(
        &mut self,
        family: &FamilyLoweringValue,
        index: &LoweredInt,
    ) -> Result<LoweredValue, LowerError> {
        match &family.storage {
            FamilyCoverageStorage::ExactStored { elements } => family::static_get(family, index)
                .map_err(|_| LowerError::FamilyAccessOutOfRange {
                    index: IntExpr::constant(-1),
                    count: IntExpr::constant(elements.len()),
                })?
                .map(LoweredValue::Term)
                .map_or_else(
                    || {
                        family::dynamic_get(&mut self.egraph, family, index.term)
                            .map(LoweredValue::Term)
                            .map_err(|_| LowerError::InvalidFamilyCount {
                                count: IntExpr::constant(elements.len()),
                            })
                    },
                    Ok,
                ),
            FamilyCoverageStorage::SharedTemplate { .. } => {
                self.shared_family_element(family, index)
            }
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
        let Some((domain, _)) = self.integer_analysis(count.term) else {
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
        let binder_id =
            self.egraph.analysis.symbols.binders.intern(super::identity::BinderDescriptor {
                key: binder.clone(),
                minimum: BigInt::zero(),
                maximum: range.minimum.clone() - BigInt::from(1_u8),
            });
        let index = LoweredInt {
            term: self.egraph.add(MxxLang::IntBinder(super::identity::BinderId(binder_id))),
            stable_identity: Some(ResolvedIntExpr::Binder(binder.clone())),
        };
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
            child_inputs.iter().copied().zip(parent_arguments).zip(specification.input_modes.iter())
        {
            let indices: Box<[LoweredInt]> = match mode {
                LoopInputMode::Broadcast => Box::new([]),
                LoopInputMode::Zip => Box::new([index.clone()]),
                LoopInputMode::ZipOffset { offset } => {
                    let offset = self.add_int(
                        BigInt::from(*offset),
                        ResolvedIntExpr::Const(BigInt::from(*offset)),
                    );
                    Box::new([self.combine_int(
                        vec![index.clone(), offset],
                        MxxLang::IntAdd,
                        ResolvedIntExpr::Add,
                    )?])
                }
            };
            child.inputs.insert(
                input,
                LoweringWire {
                    source: WireSourceKey { scope: environment.occurrence.clone(), wire: argument },
                    indices,
                },
            );
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

    fn finish_parallel_loop(
        &mut self,
        specification: &ParallelLoop,
        environment: &LowerEnv,
        output_type: WireType,
        binder: BinderKey,
        logical_count: num_bigint::BigUint,
        maximum: BigInt,
        representative: Id,
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
        let WireType::Matrix(matrix) = *element else {
            return Err(LowerError::FamilyElementTypeMismatch {
                expected: WireType::Int,
                actual: *element,
            });
        };
        let value = FamilyLoweringValue {
            element_type: self.concrete_matrix_type(&matrix, environment)?,
            storage: FamilyCoverageStorage::SharedTemplate {
                domain: family::LoopDomainKey { binder: binder.clone(), logical_count },
                representative,
                binder_domains: Box::new([family::CoverageBinderDomain {
                    binder,
                    minimum: BigInt::zero(),
                    maximum,
                }]),
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
        let Some((domain, _)) = self.integer_analysis(count.term) else {
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
        if count_range.minimum.is_zero() {
            let value =
                arguments.get(carried_index).cloned().ok_or(LowerError::InvalidOperandArity {
                    expected: specification.carried_count,
                    actual: arguments.len(),
                })?;
            work.push(LoweringFrame::FinishValue { wire, value });
            return Ok(());
        }
        let count_identity = count.stable_identity.ok_or_else(|| {
            LowerError::NonExactIdentityIndex { expression: specification.count.clone() }
        })?;
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
        let maximum = count_range.maximum - BigInt::from(1_u8);
        let binder_id =
            self.egraph.analysis.symbols.binders.intern(super::identity::BinderDescriptor {
                key: binder.clone(),
                minimum: BigInt::zero(),
                maximum,
            });
        let iteration = LoweredInt {
            term: self.egraph.add(MxxLang::IntBinder(super::identity::BinderId(binder_id))),
            stable_identity: Some(ResolvedIntExpr::Binder(binder.clone())),
        };
        child.binders.push((binder.clone(), iteration.clone()));
        child.active_coordinates.push(Coordinate { binder, index: iteration });
        let mut initial = Vec::with_capacity(specification.carried_count);
        let mut output_types = Vec::with_capacity(specification.carried_count);
        for position in 0..specification.carried_count {
            let LoweredValue::Term(initial_term) = arguments[position] else {
                return Err(LowerError::InvalidOperandArity {
                    expected: specification.carried_count,
                    actual: arguments.len(),
                });
            };
            initial.push(initial_term);
            let graph = self.graph_for_program(&child.occurrence.program)?;
            let scope = graph
                .scope(&child.occurrence.definition)
                .ok_or(LowerError::MissingWire { wire: child_inputs[position] })?;
            let input_node = scope
                .node(child_inputs[position].node)
                .ok_or(LowerError::MissingNode { node: child_inputs[position].node })?;
            let ty = input_node.output_types()[child_inputs[position].port.0 as usize].clone();
            let (WireType::Matrix(matrix) | WireType::Preimage(matrix)) = ty else {
                return Err(LowerError::InvalidOperandSort {
                    expected: output_type.clone(),
                    actual: ty,
                });
            };
            let matrix_type = self.resolve_matrix_type(&matrix, &child)?;
            output_types.push(matrix_type.clone());
            let state = self.egraph.analysis.symbols.atomic_sources.intern(
                super::identity::AtomicSourceDescriptor {
                    key: super::identity::AtomicSourceKey::SequentialState(SequentialStateKey {
                        loop_scope: environment.occurrence.clone(),
                        loop_node: wire.source.wire.node,
                        carried_index: position,
                    }),
                    sort: MxxSort::Matrix(matrix_type),
                    integer_domain: None,
                    canonical_residue_convention: None,
                    relation_role: None,
                },
            );
            child.state_inputs.insert(
                child_inputs[position],
                LoweredValue::Term(self.egraph.add(MxxLang::Atom {
                    source: super::identity::AtomicSourceId(state),
                    indices: Box::new([]),
                })),
            );
        }
        for (input, argument) in child_inputs
            .iter()
            .copied()
            .skip(specification.carried_count)
            .zip(parent_arguments.into_iter().skip(specification.carried_count))
        {
            child.inputs.insert(
                input,
                LoweringWire {
                    source: WireSourceKey { scope: environment.occurrence.clone(), wire: argument },
                    indices: Box::new([]),
                },
            );
        }
        for (name, expression) in &specification.bindings {
            let value = self.resolve_int(expression, &child)?;
            child.parameters.insert(name.clone(), value);
        }
        let dependency_count = child_outputs.len();
        work.push(LoweringFrame::FinishSequentialLoop {
            wire,
            environment,
            count: count_identity,
            initial,
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

    fn finish_sequential_loop(
        &mut self,
        wire: &LoweringWire,
        environment: &LowerEnv,
        count: ResolvedIntExpr,
        initial: Vec<Id>,
        transition: Vec<Id>,
        output_types: Vec<super::identity::ResolvedMatrixType>,
        output_type: WireType,
        carried_index: usize,
    ) -> Result<LoweredValue, LowerError> {
        let recurrence = self.egraph.analysis.symbols.sequential_recurrences.intern(
            SequentialRecurrenceDescriptor {
                loop_scope: environment.occurrence.clone(),
                loop_node: wire.source.wire.node,
                count,
                initial: initial.into_boxed_slice(),
                transition: transition.into_boxed_slice(),
                output_types: output_types.into_boxed_slice(),
            },
        );
        let (WireType::Matrix(matrix) | WireType::Preimage(matrix)) = output_type else {
            return Err(LowerError::InvalidOperandSort {
                expected: WireType::Matrix(MatrixType {
                    modulus: IntExpr::constant(1),
                    ring_dimension: IntExpr::constant(1),
                    rows: IntExpr::constant(1),
                    columns: IntExpr::constant(1),
                }),
                actual: output_type,
            });
        };
        let resolved_matrix = self.resolve_matrix_type(&matrix, environment)?;
        let source = self.egraph.analysis.symbols.atomic_sources.intern(
            super::identity::AtomicSourceDescriptor {
                key: super::identity::AtomicSourceKey::SequentialRecurrence {
                    recurrence: super::identity::SequentialRecurrenceId(recurrence),
                    carried_index,
                },
                sort: MxxSort::Matrix(resolved_matrix),
                integer_domain: None,
                canonical_residue_convention: None,
                relation_role: None,
            },
        );
        Ok(LoweredValue::Term(self.egraph.add(MxxLang::Atom {
            source: super::identity::AtomicSourceId(source),
            indices: Box::new([]),
        })))
    }

    fn concrete_matrix_type(
        &mut self,
        matrix: &MatrixType,
        environment: &LowerEnv,
    ) -> Result<mxx_ir_core::types::ConcreteMatrixType, LowerError> {
        let integer = |value: &ResolvedIntExpr| match value {
            ResolvedIntExpr::Const(value) => Ok(value.clone()),
            _ => Err(LowerError::NonExactIdentityIndex { expression: IntExpr::constant(0) }),
        };
        let modulus = integer(&self.resolve_int(&matrix.modulus, environment)?)?;
        let ring_dimension = integer(&self.resolve_int(&matrix.ring_dimension, environment)?)?
            .to_usize()
            .ok_or_else(|| LowerError::NonExactIdentityIndex {
                expression: matrix.ring_dimension.clone(),
            })?;
        let rows = integer(&self.resolve_int(&matrix.rows, environment)?)?
            .to_usize()
            .ok_or_else(|| LowerError::NonExactIdentityIndex { expression: matrix.rows.clone() })?;
        let columns =
            integer(&self.resolve_int(&matrix.columns, environment)?)?.to_usize().ok_or_else(
                || LowerError::NonExactIdentityIndex { expression: matrix.columns.clone() },
            )?;
        Ok(mxx_ir_core::types::ConcreteMatrixType { modulus, ring_dimension, rows, columns })
    }

    fn resolve_int(
        &mut self,
        expression: &IntExpr,
        environment: &LowerEnv,
    ) -> Result<ResolvedIntExpr, LowerError> {
        Ok(self
            .lower_int_expr(expression, environment)?
            .stable_identity
            .ok_or_else(|| LowerError::NonExactIdentityIndex { expression: expression.clone() })?)
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

    fn resolve_real(&self, value: &RealExpr, environment: &LowerEnv) -> Result<f64, LowerError> {
        let mut env = mxx_ir_core::ParamEnv::default();
        for (name, value) in &environment.parameters {
            if let ResolvedIntExpr::Const(value) = value {
                env.integers.insert(name.clone(), value.clone());
            }
        }
        value.evaluate_f64(&env).ok().filter(|value| value.is_finite()).ok_or_else(|| {
            LowerError::InvalidRealOperation { operation: NodeKind::ConstantReal(value.clone()) }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use std::collections::BTreeMap;

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

        let input = NodeHandle::new(
            NodeKind::ConstantMatrix {
                matrix_type: matrix.clone(),
                value: mxx_ir_core::node::ConstantMatrix::Zero,
            },
            Vec::new(),
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .expect("root input output");
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
    fn runtime_and_stable_indices_have_distinct_memo_keys() {
        let source = WireSourceKey {
            scope: OccurrenceScope {
                program: super::super::identity::ProgramKey::Ideal,
                definition: mxx_ir_core::FrozenGraphScopeId::Root,
                path: Box::new([]),
            },
            wire: WireRef { node: mxx_ir_core::NodeId(1), port: mxx_ir_core::Port(0) },
        };
        let stable = LoweringWire {
            source: source.clone(),
            indices: Box::new([LoweredInt {
                term: Id::from(0),
                stable_identity: Some(ResolvedIntExpr::Const(BigInt::from(3))),
            }]),
        };
        let runtime = LoweringWire {
            source,
            indices: Box::new([LoweredInt { term: Id::from(3), stable_identity: None }]),
        };

        assert_ne!(LoweringWireKey::from(&stable), LoweringWireKey::from(&runtime));
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
                let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
                lowerer.lower_stage_wire(&stage, output).expect("deep lowering succeeds");
                (
                    lowerer.egraph.number_of_classes(),
                    lowerer.egraph.analysis.symbols.binders.values.len(),
                )
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
