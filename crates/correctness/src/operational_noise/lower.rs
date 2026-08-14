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
        BinderKey, CanonicalResidueConvention, OccurrenceScope, ResolvedIntExpr,
        SamplerDescriptorId, SamplerIdentity, SequentialRecurrenceDescriptor, SequentialStateKey,
        TrapdoorDescriptorId, TrapdoorIdentity, TrapdoorSourceKey, WireSourceKey,
    },
    language::MxxLang,
};
use crate::{InputValueContract, ProtocolDecl, ProtocolInputDestination, StageId, StageInputName};
use egg::{EGraph, Id, RecExpr};
use mxx_ir_core::{
    IntExpr, RealExpr, WireRef, WireType,
    graph::FrozenGraphScopeId,
    node::{
        ConcatAxis, HashVariant, IntBinaryOp, IntCompareOp, LoopInputMode, MatrixBinaryOp,
        NodeKind, ParallelLoop, RealBinaryOp, SequentialLoop,
    },
    types::MatrixType,
};
use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive, Zero};
use std::collections::{BTreeMap, HashMap, HashSet};

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

/// Read-only production bridge from a lowered e-graph to the bound evaluator.
/// It deliberately has no memo: `BoundEvaluator` owns computed bounds.
pub struct ProductionBoundInput<'a, 'protocol, 'control> {
    lowerer: &'a GraphLowerer<'protocol, 'control>,
    control: Option<&'a dyn BoundEvaluationControl>,
}

impl BoundInput for ProductionBoundInput<'_, '_, '_> {
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
                let SamplerIdentity::Preimage { cutoff, .. } = sampler else {
                    return Ok(MatrixBound {
                        matrix_type,
                        coefficient_class: BoundClass::Large,
                        metadata: MatrixMetadata::unknown(),
                    });
                };
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
        let metadata = match &descriptor.key {
            super::identity::AtomicSourceKey::ProtocolInput(input) => self
                .lowerer
                .protocol
                .bundle
                .input_contract
                .inputs
                .iter()
                .find(|entry| entry.id == *input)
                .and_then(|entry| match &entry.value {
                    InputValueContract::MatrixExact { is_constant_polynomial, .. } => {
                        Some(MatrixMetadata {
                            is_constant_polynomial: *is_constant_polynomial,
                            known_zero_rows: None,
                        })
                    }
                    InputValueContract::Family { element, .. } => match element.as_ref() {
                        InputValueContract::MatrixExact { is_constant_polynomial, .. } => {
                            Some(MatrixMetadata {
                                is_constant_polynomial: *is_constant_polynomial,
                                known_zero_rows: None,
                            })
                        }
                        _ => None,
                    },
                    _ => None,
                })
                .unwrap_or_else(MatrixMetadata::unknown),
            _ => MatrixMetadata::unknown(),
        };
        Ok(MatrixBound { matrix_type, coefficient_class, metadata })
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

    fn validate_pack(&self, term: Id, bit_count: usize) -> Result<(), BoundEvaluationError> {
        if let Some(control) = self.control {
            control.validate_pack(term, bit_count)?;
        }
        Ok(())
    }
}

impl ProductionBoundInput<'_, '_, '_> {
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
struct SequentialBoundInput<'a, 'protocol, 'control> {
    base: &'a ProductionBoundInput<'a, 'protocol, 'control>,
    states: &'a [super::identity::AtomicSourceId],
    values: &'a [MatrixBound],
}

impl BoundInput for SequentialBoundInput<'_, '_, '_> {
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
        tag_prefix: Vec<u8>,
        tag_expressions: Vec<IntExpr>,
        tag_decimal_expressions: Vec<IntExpr>,
        tag_u64_le_expressions: Vec<IntExpr>,
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
pub struct GraphLowerer<'a, 'control> {
    pub protocol: &'a ProtocolDecl,
    pub request: &'a OperationalCheckRequest,
    pub egraph: EGraph<MxxLang, MxxAnalysis>,
    memo: HashMap<LoweringWireKey, LoweredValue>,
    active: HashSet<LoweringWireKey>,
    /// Extracted shared representatives are immutable valid snapshots.  They
    /// are keyed by canonical e-class at first use and remain valid after
    /// later unions; a later canonical root simply causes a harmless miss.
    shared_templates: HashMap<Id, RecExpr<MxxLang>>,
    control: Option<&'control mut dyn LoweringControl>,
}

impl<'a> GraphLowerer<'a, '_> {
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
            shared_templates: HashMap::new(),
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
        analysis: MxxAnalysis,
        control: &'control mut dyn LoweringControl,
    ) -> Self {
        Self {
            protocol,
            request,
            egraph: EGraph::new(analysis),
            memo: HashMap::new(),
            active: HashSet::new(),
            shared_templates: HashMap::new(),
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
            egraph: self.egraph,
            memo: self.memo,
            active: self.active,
            shared_templates: self.shared_templates,
            control: None,
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
        self.validate_scalar_consumer(
            term,
            MxxSort::Int,
            mxx_ir_core::WireType::Int,
            consumer,
            selector_allowed,
        )
    }

    fn validate_boolean_consumer(
        &self,
        term: Id,
        consumer: SelectorOnlyConsumer,
        selector_allowed: bool,
    ) -> Result<(), LowerError> {
        self.validate_scalar_consumer(
            term,
            MxxSort::Bool,
            mxx_ir_core::WireType::Bool,
            consumer,
            selector_allowed,
        )
    }

    fn validate_scalar_consumer(
        &self,
        term: Id,
        expected_sort: MxxSort,
        expected_wire_type: mxx_ir_core::WireType,
        consumer: SelectorOnlyConsumer,
        selector_allowed: bool,
    ) -> Result<(), LowerError> {
        let data = &self.egraph[self.egraph.find(term)].data;
        if data.sort != Ok(expected_sort.clone()) {
            let actual = match data.sort.as_ref().ok() {
                Some(MxxSort::Int) => mxx_ir_core::WireType::Int,
                Some(MxxSort::Bool) => mxx_ir_core::WireType::Bool,
                Some(MxxSort::Real) => mxx_ir_core::WireType::Real,
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
        let key = LoweringWireKey::from(wire);
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
            .filter_map(|(id, sampler)| match sampler {
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
                .into(),
                SamplerIdentity::DecomposedHash { public, target, indices, .. } |
                SamplerIdentity::GadgetDecomposition { public, target, indices, .. } => {
                    Some(super::relation::RelationRegistration {
                        source: super::identity::AtomicSourceId(
                            self.egraph
                                .analysis
                                .symbols
                                .atomic_sources
                                .values
                                .iter()
                                .position(|source| {
                                    matches!(source.key,
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
                        trapdoor: None,
                        indices: indices.clone(),
                    })
                }
            })
            .collect()
    }

    /// Returns the one production view used by the bound evaluator.  It reads
    /// canonical e-graph analysis and exact lowering descriptors only.
    pub fn production_bound_view(&self) -> ProductionBoundInput<'_, 'a, 'control> {
        ProductionBoundInput { lowerer: self, control: None }
    }

    /// Constructs the production evaluator view with semantic pack validation.
    /// The control-free view remains available for direct, deterministic unit
    /// tests.
    pub fn production_bound_view_with_control<'b>(
        &'b self,
        control: &'b dyn BoundEvaluationControl,
    ) -> ProductionBoundInput<'b, 'a, 'control> {
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
                    if let Some(value) = environment.state_inputs.get(&wire.source.wire) {
                        self.finish_wire(&wire, value.clone());
                        values.push(value.clone());
                        continue;
                    }
                    if wire.source.scope == environment.occurrence &&
                        let Some(bound) = environment.inputs.get(&wire.source.wire)
                    {
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
                                tag_prefix,
                                tag_expressions,
                                tag_decimal_expressions,
                                tag_u64_le_expressions,
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
                                    tag_prefix: tag_prefix.clone(),
                                    tag_expressions: tag_expressions.clone(),
                                    tag_decimal_expressions: tag_decimal_expressions.clone(),
                                    tag_u64_le_expressions: tag_u64_le_expressions.clone(),
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
                                let public = self.atom_for_wire(
                                    &public,
                                    &environment,
                                    WireType::Matrix(matrix_type.clone()),
                                    None,
                                )?;
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
                                    LoweredValue::Term(self.atom_for_wire(
                                        &wire,
                                        &environment,
                                        *element,
                                        role,
                                    )?)
                                };
                                self.finish_wire(&wire, value.clone());
                                values.push(value);
                                continue;
                            }
                            let value = LoweredValue::Term(self.atom_for_wire(
                                &wire,
                                &environment,
                                output_type,
                                role,
                            )?);
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
                                child.inputs = child_inputs
                                    .iter()
                                    .copied()
                                    .zip(arguments)
                                    .map(|(input, argument)| {
                                        (
                                            input,
                                            LoweringWire {
                                                source: WireSourceKey {
                                                    scope: wire.source.scope.clone(),
                                                    wire: argument,
                                                },
                                                indices: wire.indices.clone(),
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
                    let representative_value =
                        values.pop().ok_or(LowerError::MissingWire { wire: wire.source.wire })?;
                    let representative_value =
                        if let WireType::IndexedFamily { element, .. } = &output_type {
                            self.normalize_singleton_for_input(representative_value, element)?
                        } else {
                            representative_value
                        };
                    if let LoweredValue::Trapdoor(representative) = representative_value {
                        if !matches!(&output_type, WireType::IndexedFamily { element, .. } if matches!(element.as_ref(), WireType::Trapdoor { .. }))
                        {
                            return Err(LowerError::NonUniformParallelMatrixType {
                                expected: output_type,
                                actual: WireType::Int,
                            });
                        }
                        let value =
                            LoweredValue::TrapdoorFamily { representative, binder, logical_count };
                        self.finish_wire(&wire, value.clone());
                        values.push(value);
                        continue;
                    }
                    let LoweredValue::Term(representative) = representative_value else {
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
                    let transition_values = values.split_off(values.len() - dependency_count);
                    if count == ResolvedIntExpr::Const(BigInt::from(1_u8)) {
                        let value = transition_values.get(carried_index).cloned().ok_or(
                            LowerError::InvalidOutputPort {
                                wire: wire.source.wire,
                                output_count: transition_values.len(),
                            },
                        )?;
                        let value = self.normalize_singleton_for_input(value, &output_type)?;
                        self.finish_wire(&wire, value.clone());
                        values.push(value);
                        continue;
                    }
                    let transitions = transition_values
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
                        LoweredValue::Term(term) => LoweredValue::Term(term),
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
                LoweringFrame::FinishHashSample {
                    wire,
                    environment,
                    matrix_type,
                    variant,
                    tag_prefix,
                    tag_expressions,
                    tag_decimal_expressions,
                    tag_u64_le_expressions,
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
                            LoweredValue::Term(term) => Ok(term),
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
                    let Some(key) = arguments.first() else {
                        return Err(LowerError::InvalidOperandArity { expected: 1, actual: 0 });
                    };
                    if !matches!(
                        self.egraph[self.egraph.find(*key)].data.sort,
                        Ok(MxxSort::Bytes(_))
                    ) || arguments.iter().skip(1).any(|argument| {
                        !matches!(
                            self.egraph[self.egraph.find(*argument)].data.sort,
                            Ok(MxxSort::Int)
                        )
                    }) {
                        return Err(LowerError::InvalidOperandSort {
                            expected: WireType::Int,
                            actual: WireType::Int,
                        });
                    }
                    let (WireType::Matrix(output_matrix) | WireType::Preimage(output_matrix)) =
                        output_type
                    else {
                        return Err(LowerError::InvalidOperandSort {
                            expected: WireType::Matrix(matrix_type),
                            actual: output_type,
                        });
                    };
                    if output_matrix != matrix_type {
                        return Err(LowerError::InvalidOperandSort {
                            expected: WireType::Matrix(matrix_type),
                            actual: WireType::Matrix(output_matrix),
                        });
                    }
                    let output_matrix = self.resolve_matrix_type(&output_matrix, &environment)?;
                    let mut tag_program = Vec::new();
                    if !tag_prefix.is_empty() {
                        tag_program.push(super::identity::HashTagPart::Literal(
                            tag_prefix.into_boxed_slice(),
                        ));
                    }
                    for expression in &tag_expressions {
                        tag_program.push(super::identity::HashTagPart::BinaryStatic(
                            self.resolve_int(expression, &environment)?,
                        ));
                    }
                    for expression in &tag_decimal_expressions {
                        tag_program.push(super::identity::HashTagPart::DecimalStatic(
                            self.resolve_int(expression, &environment)?,
                        ));
                    }
                    for expression in &tag_u64_le_expressions {
                        tag_program.push(super::identity::HashTagPart::U64LeStatic(
                            self.resolve_int(expression, &environment)?,
                        ));
                    }
                    for argument in 1..arguments.len() {
                        let argument = u16::try_from(argument).map_err(|_| {
                            LowerError::InvalidOperandArity {
                                expected: u16::MAX as usize,
                                actual: arguments.len(),
                            }
                        })?;
                        tag_program.push(super::identity::HashTagPart::BinaryArgument { argument });
                    }
                    let (target, public, base, digit_count, small) = match variant {
                        HashVariant::Plain => {
                            if base.is_some() || digit_count.is_some() {
                                return Err(LowerError::InvalidOperandArity {
                                    expected: 0,
                                    actual: 1,
                                });
                            }
                            let query = self.egraph.analysis.symbols.hash_queries.intern(
                                super::identity::HashQuerySpec {
                                    matrix_type: output_matrix.clone(),
                                    tag_program: tag_program.into_boxed_slice(),
                                },
                            );
                            (
                                self.egraph.add(MxxLang::HashPlain {
                                    query: super::identity::HashQuerySpecId(query),
                                    arguments: arguments.clone().into_boxed_slice(),
                                }),
                                None,
                                None,
                                None,
                                false,
                            )
                        }
                        HashVariant::Decomposed | HashVariant::SmallDecomposed => {
                            let (Some(base), Some(digit_count)) = (base, digit_count) else {
                                return Err(LowerError::InvalidOperandArity {
                                    expected: 2,
                                    actual: 0,
                                });
                            };
                            let base = self.resolve_int(&base, &environment)?;
                            let digit_count = self.resolve_int(&digit_count, &environment)?;
                            let Some(base_value) = resolved_integer(&base) else {
                                return Err(LowerError::NonExactIdentityIndex {
                                    expression: IntExpr::constant(0),
                                });
                            };
                            let Some(digit_count_value) = resolved_nonnegative(&digit_count) else {
                                return Err(LowerError::NonExactIdentityIndex {
                                    expression: IntExpr::constant(0),
                                });
                            };
                            let Some(rows) = resolved_nonnegative(&output_matrix.rows) else {
                                return Err(LowerError::NonExactIdentityIndex {
                                    expression: matrix_type.rows.clone(),
                                });
                            };
                            let Some(output_rows) = rows.to_usize() else {
                                return Err(LowerError::NonExactIdentityIndex {
                                    expression: matrix_type.rows.clone(),
                                });
                            };
                            let Some(digit_count_usize) = digit_count_value.to_usize() else {
                                return Err(LowerError::NonExactIdentityIndex {
                                    expression: IntExpr::constant(0),
                                });
                            };
                            if base_value <= BigInt::from(1) ||
                                digit_count_usize == 0 ||
                                output_rows == 0 ||
                                output_rows % digit_count_usize != 0
                            {
                                return Err(LowerError::InvalidOperandArity {
                                    expected: digit_count_usize,
                                    actual: output_rows,
                                });
                            }
                            let small = variant == HashVariant::SmallDecomposed;
                            let Some(modulus) = resolved_integer(&output_matrix.modulus) else {
                                return Err(LowerError::NonExactIdentityIndex {
                                    expression: matrix_type.modulus.clone(),
                                });
                            };
                            let Some(ring_dimension) =
                                resolved_nonnegative(&output_matrix.ring_dimension)
                                    .and_then(|value| value.to_usize())
                            else {
                                return Err(LowerError::NonExactIdentityIndex {
                                    expression: matrix_type.ring_dimension.clone(),
                                });
                            };
                            let layout_matches = self.request.layouts.iter().any(|layout| {
                                let layout_modulus = layout
                                    .crt_moduli
                                    .iter()
                                    .fold(BigInt::from(1), |product, modulus| {
                                        product * BigInt::from(*modulus)
                                    });
                                layout.ring_dimension == ring_dimension &&
                                    layout_modulus == modulus &&
                                    layout.base == base_value &&
                                    (if small {
                                        layout.small_digit_count
                                    } else {
                                        layout.regular_digit_count
                                    }) == digit_count_usize
                            });
                            if !layout_matches {
                                return Err(LowerError::InvalidOperandArity {
                                    expected: 1,
                                    actual: 0,
                                });
                            }
                            let plain_rows = output_rows / digit_count_usize;
                            let mut plain_matrix = output_matrix.clone();
                            plain_matrix.rows = ResolvedIntExpr::Const(BigInt::from(plain_rows));
                            let mut gadget_matrix = plain_matrix.clone();
                            gadget_matrix.columns =
                                ResolvedIntExpr::Const(BigInt::from(output_rows));
                            let query = self.egraph.analysis.symbols.hash_queries.intern(
                                super::identity::HashQuerySpec {
                                    matrix_type: plain_matrix,
                                    tag_program: tag_program.into_boxed_slice(),
                                },
                            );
                            let target = self.egraph.add(MxxLang::HashPlain {
                                query: super::identity::HashQuerySpecId(query),
                                arguments: arguments.clone().into_boxed_slice(),
                            });
                            let gadget = self.egraph.analysis.symbols.matrix_constants.intern(
                                super::identity::MatrixConstantSpec {
                                    matrix_type: gadget_matrix,
                                    value: super::identity::MatrixConstantValue::Gadget {
                                        base: base.clone(),
                                        small,
                                    },
                                },
                            );
                            let public = self.egraph.add(MxxLang::MatrixConstant(
                                super::identity::MatrixConstantSpecId(gadget),
                            ));
                            (target, Some(public), Some(base), Some(digit_count), small)
                        }
                    };
                    let value = if let Some(public) = public {
                        let range_proved = if small {
                            base.as_ref()
                                .and_then(resolved_nonnegative)
                                .zip(
                                    self.egraph[self.egraph.find(target)]
                                        .data
                                        .canonical_coefficient_exclusive_upper
                                        .as_ref(),
                                )
                                .is_some_and(|(limit, upper)| {
                                    super::identity::canonical_range_within_limit(
                                        Some(upper),
                                        &limit,
                                    )
                                })
                        } else {
                            false
                        };
                        let sampler = self.egraph.analysis.symbols.samplers.intern(
                            SamplerIdentity::DecomposedHash {
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
                                public,
                                target,
                                arguments: arguments.into_boxed_slice(),
                                matrix_type: output_matrix.clone(),
                                base: base.expect("decomposed hash base"),
                                digit_count: digit_count.expect("decomposed hash digits"),
                                small,
                                range_proved,
                            },
                        );
                        let source = self.egraph.analysis.symbols.atomic_sources.intern(
                            super::identity::AtomicSourceDescriptor {
                                key: super::identity::AtomicSourceKey::Sampler(
                                    SamplerDescriptorId(sampler),
                                ),
                                sort: MxxSort::Matrix(output_matrix),
                                integer_domain: None,
                                canonical_residue_convention: None,
                                relation_role: Some(if small {
                                    super::identity::AtomicRelationRole::SmallDecomposedHash {
                                        range_proved,
                                    }
                                } else {
                                    super::identity::AtomicRelationRole::DecomposedHash
                                }),
                            },
                        );
                        LoweredValue::Term(
                            self.egraph.add(MxxLang::Atom {
                                source: super::identity::AtomicSourceId(source),
                                indices: environment
                                    .active_coordinates
                                    .iter()
                                    .map(|coordinate| coordinate.index.term)
                                    .collect(),
                            }),
                        )
                    } else {
                        LoweredValue::Term(target)
                    };
                    self.finish_wire(&wire, value.clone());
                    values.push(value);
                }
                LoweringFrame::FinishGadgetDecompose {
                    wire,
                    environment,
                    base,
                    digit_count,
                    small,
                    output_type,
                } => {
                    let LoweredValue::Term(target) = values
                        .pop()
                        .ok_or(LowerError::InvalidOperandArity { expected: 1, actual: 0 })?
                    else {
                        return Err(LowerError::InvalidOperandArity { expected: 1, actual: 0 });
                    };
                    let (WireType::Matrix(matrix) | WireType::Preimage(matrix)) = output_type
                    else {
                        return Err(LowerError::InvalidOperandSort {
                            expected: WireType::Int,
                            actual: output_type,
                        });
                    };
                    let output_matrix = self.resolve_matrix_type(&matrix, &environment)?;
                    let base = self.resolve_int(&base, &environment)?;
                    let digit_count = self.resolve_int(&digit_count, &environment)?;
                    let Some(base_limit) = resolved_nonnegative(&base) else {
                        return Err(LowerError::NonExactIdentityIndex {
                            expression: IntExpr::constant(0),
                        });
                    };
                    let range_proved = small &&
                        super::identity::canonical_range_within_limit(
                            self.egraph[self.egraph.find(target)]
                                .data
                                .canonical_coefficient_exclusive_upper
                                .as_ref(),
                            &base_limit,
                        );
                    let Some(digits) =
                        resolved_nonnegative(&digit_count).and_then(|v| v.to_usize())
                    else {
                        return Err(LowerError::NonExactIdentityIndex {
                            expression: IntExpr::constant(0),
                        });
                    };
                    let Some(output_rows) = resolved_nonnegative(&output_matrix.rows) else {
                        return Err(LowerError::NonExactIdentityIndex {
                            expression: matrix.rows.clone(),
                        });
                    };
                    let Some(target_rows) = output_rows
                        .to_usize()
                        .filter(|rows| digits != 0 && rows % digits == 0)
                        .map(|rows| rows / digits)
                    else {
                        return Err(LowerError::InvalidOperandArity { expected: digits, actual: 0 });
                    };
                    let mut gadget_matrix = output_matrix.clone();
                    gadget_matrix.rows = ResolvedIntExpr::Const(BigInt::from(target_rows));
                    gadget_matrix.columns = ResolvedIntExpr::Const(BigInt::from(output_rows));
                    let gadget = self.egraph.analysis.symbols.matrix_constants.intern(
                        super::identity::MatrixConstantSpec {
                            matrix_type: gadget_matrix,
                            value: super::identity::MatrixConstantValue::Gadget {
                                base: base.clone(),
                                small,
                            },
                        },
                    );
                    let public = self.egraph.add(MxxLang::MatrixConstant(
                        super::identity::MatrixConstantSpecId(gadget),
                    ));
                    let indices: Box<[Id]> = environment
                        .active_coordinates
                        .iter()
                        .map(|coordinate| coordinate.index.term)
                        .collect();
                    let sampler = self.egraph.analysis.symbols.samplers.intern(
                        SamplerIdentity::GadgetDecomposition {
                            source: super::identity::GraphWireSourceKey {
                                wire: wire.source.clone(),
                                coordinate_binders: environment
                                    .active_coordinates
                                    .iter()
                                    .map(|coordinate| coordinate.binder.clone())
                                    .collect(),
                            },
                            indices: indices.clone(),
                            public,
                            target,
                            base,
                            digit_count,
                            small,
                            range_proved,
                        },
                    );
                    let source = self.egraph.analysis.symbols.atomic_sources.intern(
                        super::identity::AtomicSourceDescriptor {
                            key: super::identity::AtomicSourceKey::Sampler(SamplerDescriptorId(
                                sampler,
                            )),
                            sort: MxxSort::Matrix(output_matrix),
                            integer_domain: None,
                            canonical_residue_convention: None,
                            relation_role: Some(if small {
                                super::identity::AtomicRelationRole::SmallGadgetDecomposition {
                                    range_proved,
                                }
                            } else {
                                super::identity::AtomicRelationRole::GadgetDecomposition
                            }),
                        },
                    );
                    let value = LoweredValue::Term(self.egraph.add(MxxLang::Atom {
                        source: super::identity::AtomicSourceId(source),
                        indices,
                    }));
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

    fn lower_source_family(
        &mut self,
        wire: &LoweringWire,
        environment: &LowerEnv,
        element: WireType,
        count: IntExpr,
    ) -> Result<LoweredValue, LowerError> {
        let count_value = self.lower_int_expr(&count, environment)?;
        let count_range = self
            .integer_analysis(count_value.term)
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
        let binder_id =
            self.egraph.analysis.symbols.binders.intern(super::identity::BinderDescriptor {
                key: binder.clone(),
                minimum: BigInt::zero(),
                maximum: count_range.minimum - BigInt::from(1_u8),
            });
        let index = LoweredInt {
            term: self.egraph.add(MxxLang::IntBinder(super::identity::BinderId(binder_id))),
            stable_identity: Some(ResolvedIntExpr::Binder(binder.clone())),
        };
        let mut indexed_wire = wire.clone();
        indexed_wire.indices = wire.indices.iter().cloned().chain([index.clone()]).collect();
        let mut indexed_environment = environment.clone();
        indexed_environment.active_coordinates.push(Coordinate { binder: binder.clone(), index });
        let element_type = self.resolve_family_element_sort(&element, &indexed_environment)?;
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
    ) -> Result<Id, LowerError> {
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
        let (key, integer_domain, canonical_residue_convention) =
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
                    None,
                )
            });
        let descriptor = super::identity::AtomicSourceDescriptor {
            key,
            sort,
            integer_domain,
            canonical_residue_convention,
            relation_role,
        };
        let source = self.egraph.analysis.symbols.atomic_sources.intern(descriptor);
        let atom = self.egraph.add(MxxLang::Atom {
            source: super::identity::AtomicSourceId(source),
            indices: environment
                .active_coordinates
                .iter()
                .map(|coordinate| coordinate.index.term)
                .collect(),
        });
        Ok(atom)
    }

    /// Normalizes a non-artifact root-stage input to its closed protocol input identity.  The
    /// graph's local input name is never used as an analysis identity.
    fn protocol_input_source(
        &mut self,
        wire: &LoweringWire,
        environment: &LowerEnv,
        sort: &MxxSort,
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
            (Some((lower, upper)), MxxSort::Int) => {
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
            (Some(upper), MxxSort::Matrix(matrix)) => {
                let modulus = resolved_nonnegative(&matrix.modulus).ok_or_else(|| {
                    LowerError::InvalidExtractCoefficientCanonicalUpper {
                        upper: num_bigint::BigUint::from(0_u8),
                        modulus: num_bigint::BigUint::from(0_u8),
                    }
                })?;
                let upper = match upper {
                    Some(upper) => match self.resolve_int(upper, environment)? {
                        ResolvedIntExpr::Const(upper) => upper.to_biguint().ok_or_else(|| {
                            LowerError::InvalidExtractCoefficientCanonicalUpper {
                                upper: num_bigint::BigUint::from(0_u8),
                                modulus: modulus.clone(),
                            }
                        })?,
                        _ => {
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
        let public = self.egraph.analysis.symbols.atomic_sources.intern(
            super::identity::AtomicSourceDescriptor {
                key: super::identity::AtomicSourceKey::ProtocolInput(public_input.clone()),
                sort: MxxSort::Matrix(matrix_type.clone()),
                integer_domain: None,
                canonical_residue_convention: None,
                relation_role: None,
            },
        );
        let public = self.egraph.add(MxxLang::Atom {
            source: super::identity::AtomicSourceId(public),
            indices: environment
                .active_coordinates
                .iter()
                .map(|coordinate| coordinate.index.term)
                .collect(),
        });
        let descriptor = TrapdoorIdentity {
            source: TrapdoorSourceKey::ProtocolInput(input),
            indices: environment
                .active_coordinates
                .iter()
                .map(|coordinate| coordinate.index.term)
                .collect(),
            matrix_type,
            public,
            sigma_bits: self.resolve_real(sigma, environment)?.to_bits(),
            gadget_base: self.resolve_int(gadget_base, environment)?,
            digit_count: self.resolve_int(digit_count, environment)?,
            preimage_cutoff: self.resolve_int(preimage_max_coefficient_bound, environment)?,
        };
        Ok(Some(TrapdoorDescriptorId(self.egraph.analysis.symbols.trapdoors.intern(descriptor))))
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
        let stable_identity = left
            .stable_identity
            .zip(right.stable_identity)
            .map(|(left, right)| identity(Box::new(left), Box::new(right)));
        let term = self.egraph.add(node([left.term, right.term]));
        let stable_identity = self
            .integer_analysis(term)
            .and_then(|(domain, _)| domain.interval().ok())
            .and_then(|interval| {
                (interval.minimum == interval.maximum)
                    .then_some(ResolvedIntExpr::Const(interval.minimum))
            })
            .or(stable_identity);
        Ok(LoweredInt { term, stable_identity })
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
                self.validate_boolean_consumer(values[0], SelectorOnlyConsumer::BoolToInt, false)?;
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
                self.egraph.add(MxxLang::MatrixScale([scalar.term, matrix]))
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
                        LoweredValue::Trapdoor(_) | LoweredValue::TrapdoorFamily { .. } => {
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
            NodeKind::ExtractCoefficient { position, canonical_input_exclusive_upper } => {
                let matrix = terms(1)?[0];
                let position = self.lower_int_expr(position, environment)?;
                if let Some(upper) = canonical_input_exclusive_upper {
                    let modulus = match &self.egraph[self.egraph.find(matrix)].data.sort {
                        Ok(MxxSort::Matrix(matrix)) => {
                            resolved_nonnegative(&matrix.modulus).unwrap_or_default()
                        }
                        _ => num_bigint::BigUint::default(),
                    };
                    if upper.is_zero() || upper > &modulus {
                        return Err(LowerError::InvalidExtractCoefficientCanonicalUpper {
                            upper: upper.clone(),
                            modulus,
                        });
                    }
                }
                self.egraph.add(MxxLang::ExtractCoefficient {
                    canonical_exclusive_upper: canonical_input_exclusive_upper.clone(),
                    input: [matrix, position.term],
                })
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
                            LoweredValue::Trapdoor(_) | LoweredValue::TrapdoorFamily { .. } => {
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
                let element_wire_type = *element;
                let element_type =
                    self.resolve_family_element_sort(&element_wire_type, environment)?;
                let elements = arguments
                    .iter()
                    .map(|argument| match argument {
                        LoweredValue::Term(term)
                            if self.egraph[self.egraph.find(*term)].data.sort ==
                                Ok(element_type.clone()) =>
                        {
                            Ok(*term)
                        }
                        LoweredValue::Term(term) => Err(LowerError::FamilyElementTypeMismatch {
                            expected: element_wire_type.clone(),
                            actual: self.scalar_wire_type(*term).unwrap_or(WireType::Int),
                        }),
                        LoweredValue::Family(_) |
                        LoweredValue::Trapdoor(_) |
                        LoweredValue::TrapdoorFamily { .. } => {
                            Err(LowerError::FamilyElementTypeMismatch {
                                expected: element_wire_type.clone(),
                                actual: WireType::Int,
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
                if let [
                    LoweredValue::TrapdoorFamily { representative, binder, logical_count },
                    LoweredValue::Term(selector),
                ] = arguments
                {
                    return self.trapdoor_family_element(
                        *representative,
                        binder,
                        logical_count,
                        &LoweredInt { term: *selector, stable_identity: None },
                    );
                }
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
                let mut families = cases.iter().cloned().collect::<Vec<_>>();
                if !matches!(output_type, WireType::IndexedFamily { .. }) {
                    let zero = self.add_int(BigInt::zero(), ResolvedIntExpr::Const(BigInt::zero()));
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
                    return family::select_family(&mut self.egraph, *selector, &families)
                        .map(LoweredValue::Family)
                        .map_err(|_| LowerError::IncompatibleFamilyCoverage {
                            expected: output_type.clone(),
                            actual: output_type,
                        });
                }
                let terms = families
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

    /// A Select chooses one value at one logical family index.  Branch-local
    /// parallel loops may use distinct binder identities for that same index;
    /// align those alpha-equivalent templates to the first case without
    /// enumerating their logical elements.
    fn align_selected_shared_families(
        &mut self,
        mut families: Vec<FamilyLoweringValue>,
    ) -> Result<Vec<FamilyLoweringValue>, ()> {
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
        let common_binder_id = self
            .egraph
            .analysis
            .symbols
            .binders
            .values
            .iter()
            .position(|descriptor| descriptor.key == common_domain.binder)
            .and_then(|id| u32::try_from(id).ok())
            .ok_or(())?;
        let common_binder_term =
            self.egraph.add(MxxLang::IntBinder(super::identity::BinderId(common_binder_id)));
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
            let binder_id = self
                .egraph
                .analysis
                .symbols
                .binders
                .values
                .iter()
                .position(|descriptor| descriptor.key == domain.binder)
                .and_then(|id| u32::try_from(id).ok())
                .ok_or(())?;
            let scope = domain.binder.loop_scope.clone();
            let node = domain.binder.loop_node;
            let control = &mut self.control;
            let representative = family::instantiate_shared_element(
                &mut self.egraph,
                &mut self.shared_templates,
                *representative,
                super::identity::BinderId(binder_id),
                common_binder_term,
                &mut || {
                    if let Some(control) = control.as_deref_mut() {
                        control.work(&scope, node).map_err(|_| ())?;
                    }
                    Ok(())
                },
            )?;
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
        family: &FamilyLoweringValue,
        index: &LoweredInt,
    ) -> Result<LoweredValue, LowerError> {
        let (representative, domain, _) = family::shared_element(family)
            .map_err(|_| LowerError::InvalidFamilyCount { count: IntExpr::constant(0) })?;
        let Some(index_analysis) = self.integer_analysis(index.term) else {
            return Err(LowerError::MissingIntegerAnalysis { term: index.term });
        };
        let index_domain = index_analysis.0;
        if family::validate_family_index(index_domain, &domain.logical_count).is_err() {
            return Err(LowerError::FamilyAccessOutOfRange {
                index: index
                    .stable_identity
                    .as_ref()
                    .and_then(|value| match value {
                        ResolvedIntExpr::Const(value) => Some(IntExpr::constant(value.clone())),
                        _ => None,
                    })
                    .unwrap_or_else(|| IntExpr::constant(-1)),
                count: IntExpr::constant(domain.logical_count.clone()),
            });
        }
        if index.stable_identity.as_ref() == Some(&ResolvedIntExpr::Binder(domain.binder.clone())) {
            return Ok(LoweredValue::Term(representative));
        }
        let binder_id = self
            .egraph
            .analysis
            .symbols
            .binders
            .values
            .iter()
            .position(|descriptor| descriptor.key == domain.binder)
            .and_then(|id| u32::try_from(id).ok())
            .ok_or_else(|| LowerError::InvalidFamilyCount {
                count: IntExpr::constant(domain.logical_count.clone()),
            })?;
        let scope = domain.binder.loop_scope.clone();
        let node = domain.binder.loop_node;
        let control = &mut self.control;
        family::instantiate_shared_element(
            &mut self.egraph,
            &mut self.shared_templates,
            representative,
            super::identity::BinderId(binder_id),
            index.term,
            &mut || {
                if let Some(control) = control.as_deref_mut() {
                    control.work(&scope, node)?;
                }
                Ok(())
            },
        )
        .map(LoweredValue::Term)
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

    fn trapdoor_family_element(
        &mut self,
        representative: TrapdoorDescriptorId,
        binder: &BinderKey,
        logical_count: &num_bigint::BigUint,
        index: &LoweredInt,
    ) -> Result<LoweredValue, LowerError> {
        let index_domain =
            self.integer_analysis(index.term).map(|(domain, _)| domain).ok_or_else(|| {
                LowerError::FamilyAccessOutOfRange {
                    index: IntExpr::constant(-1),
                    count: IntExpr::constant(logical_count.clone()),
                }
            })?;
        family::validate_family_index(index_domain, logical_count).map_err(|_| {
            LowerError::FamilyAccessOutOfRange {
                index: IntExpr::constant(-1),
                count: IntExpr::constant(logical_count.clone()),
            }
        })?;
        let binder_id = self
            .egraph
            .analysis
            .symbols
            .binders
            .values
            .iter()
            .position(|descriptor| &descriptor.key == binder)
            .and_then(|id| u32::try_from(id).ok())
            .ok_or_else(|| LowerError::InvalidFamilyCount {
                count: IntExpr::constant(logical_count.clone()),
            })?;
        let template =
            self.egraph.analysis.symbols.trapdoors.get(representative.0).cloned().ok_or(
                LowerError::FamilyProducerNotResolved {
                    family: WireRef {
                        node: binder.loop_node,
                        port: mxx_ir_core::Port(binder.slot),
                    },
                },
            )?;
        let mut instantiate = |term| {
            family::instantiate_shared_element(
                &mut self.egraph,
                &mut self.shared_templates,
                term,
                super::identity::BinderId(binder_id),
                index.term,
                &mut || Ok::<(), LowerError>(()),
            )
        };
        let public = instantiate(template.public)?;
        let indices = template
            .indices
            .iter()
            .copied()
            .map(&mut instantiate)
            .collect::<Result<Vec<_>, _>>()?;
        let descriptor =
            TrapdoorIdentity { public, indices: indices.into_boxed_slice(), ..template };
        let descriptor = self.egraph.analysis.symbols.trapdoors.intern(descriptor);
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
        let zero = self.add_int(BigInt::zero(), ResolvedIntExpr::Const(BigInt::zero()));
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
            child_inputs.iter().copied().zip(arguments).zip(specification.input_modes.iter())
        {
            let value = match mode {
                LoopInputMode::Broadcast => argument,
                LoopInputMode::Zip => match argument {
                    LoweredValue::Family(family) => self.family_element(&family, &index)?,
                    LoweredValue::TrapdoorFamily { representative, binder, logical_count } => self
                        .trapdoor_family_element(representative, &binder, &logical_count, &index)?,
                    value => value,
                },
                LoopInputMode::ZipOffset { offset } => {
                    let offset = self.add_int(
                        BigInt::from(*offset),
                        ResolvedIntExpr::Const(BigInt::from(*offset)),
                    );
                    let offset_index = self.combine_int(
                        vec![index.clone(), offset],
                        MxxLang::IntAdd,
                        ResolvedIntExpr::Add,
                    )?;
                    match argument {
                        LoweredValue::Family(family) => {
                            self.family_element(&family, &offset_index)?
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
            child.state_inputs.insert(input, value);
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
        let element_wire_type = *element;
        let element_type = self.resolve_family_element_sort(&element_wire_type, environment)?;
        let actual_sort = &self.egraph[self.egraph.find(representative)].data.sort;
        let sort_matches = match (&element_type, actual_sort) {
            (MxxSort::Matrix(expected), Ok(MxxSort::Matrix(actual))) => {
                super::analysis::matrix_types_equal(expected, actual)
            }
            (expected, Ok(actual)) => expected == actual,
            (_, Err(_)) => false,
        };
        if !sort_matches {
            return Err(LowerError::FamilyElementTypeMismatch {
                expected: element_wire_type,
                actual: self.scalar_wire_type(representative).unwrap_or(WireType::Int),
            });
        }
        let mut binder_domains = environment
            .active_coordinates
            .iter()
            .map(|coordinate| {
                let interval = self
                    .integer_analysis(coordinate.index.term)
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
        let single_iteration = count_identity == ResolvedIntExpr::Const(BigInt::from(1_u8));
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
            if single_iteration {
                let input_type = {
                    let graph = self.graph_for_program(&child.occurrence.program)?;
                    let scope = graph
                        .scope(&child.occurrence.definition)
                        .ok_or(LowerError::MissingWire { wire: child_inputs[position] })?;
                    let input_node = scope
                        .node(child_inputs[position].node)
                        .ok_or(LowerError::MissingNode { node: child_inputs[position].node })?;
                    input_node.output_types()[child_inputs[position].port.0 as usize].clone()
                };
                let value =
                    self.normalize_singleton_for_input(arguments[position].clone(), &input_type)?;
                child.state_inputs.insert(child_inputs[position], value);
                continue;
            }
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
            .zip(arguments.iter().skip(specification.carried_count).cloned())
        {
            child.state_inputs.insert(input, argument);
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
        if count == ResolvedIntExpr::Const(BigInt::from(1_u8)) {
            return transition.get(carried_index).copied().map(LoweredValue::Term).ok_or(
                LowerError::InvalidOutputPort {
                    wire: wire.source.wire,
                    output_count: transition.len(),
                },
            );
        }
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

    fn resolve_family_element_sort(
        &mut self,
        element: &WireType,
        environment: &LowerEnv,
    ) -> Result<MxxSort, LowerError> {
        match element {
            WireType::Int => Ok(MxxSort::Int),
            WireType::Bool => Ok(MxxSort::Bool),
            WireType::Matrix(matrix) => {
                self.resolve_matrix_type(matrix, environment).map(MxxSort::Matrix)
            }
            actual => Err(LowerError::FamilyElementTypeMismatch {
                expected: WireType::Int,
                actual: actual.clone(),
            }),
        }
    }

    fn scalar_wire_type(&self, term: Id) -> Option<WireType> {
        match self.egraph[self.egraph.find(term)].data.sort.as_ref().ok()? {
            MxxSort::Int => Some(WireType::Int),
            MxxSort::Bool => Some(WireType::Bool),
            MxxSort::Real => Some(WireType::Real),
            _ => None,
        }
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

    #[test]
    fn extract_coefficient_uses_direct_upper_and_rejects_oversized_upper() {
        let protocol = crate::toy_example::protocol();
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "extract-upper".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
        let matrix_type = super::super::identity::ResolvedMatrixType {
            modulus: ResolvedIntExpr::Const(BigInt::from(17)),
            ring_dimension: ResolvedIntExpr::Const(BigInt::from(1)),
            rows: ResolvedIntExpr::Const(BigInt::from(1)),
            columns: ResolvedIntExpr::Const(BigInt::from(1)),
        };
        let source = lowerer.egraph.analysis.symbols.atomic_sources.intern(
            super::super::identity::AtomicSourceDescriptor {
                key: super::super::identity::AtomicSourceKey::ProtocolInput(
                    crate::ProtocolInputId::from("extract-input"),
                ),
                sort: MxxSort::Matrix(matrix_type),
                integer_domain: None,
                canonical_residue_convention: Some(CanonicalResidueConvention::Nonnegative),
                relation_role: None,
            },
        );
        let matrix = lowerer.egraph.add(MxxLang::Atom {
            source: super::super::identity::AtomicSourceId(source),
            indices: Box::new([]),
        });
        let environment = root_test_environment();
        let kind = NodeKind::ExtractCoefficient {
            position: IntExpr::constant(0),
            canonical_input_exclusive_upper: Some(4_u8.into()),
        };
        let LoweredValue::Term(extract) =
            lowerer.lower_node(&kind, &[LoweredValue::Term(matrix)], &environment).unwrap()
        else {
            unreachable!()
        };
        assert_eq!(
            lowerer.integer_analysis(extract).unwrap().0.interval().unwrap(),
            super::super::analysis::IntegerInterval::new(0.into(), 3.into()).unwrap()
        );

        let invalid = NodeKind::ExtractCoefficient {
            position: IntExpr::constant(0),
            canonical_input_exclusive_upper: Some(18_u8.into()),
        };
        assert!(matches!(
            lowerer.lower_node(&invalid, &[LoweredValue::Term(matrix)], &environment),
            Err(LowerError::InvalidExtractCoefficientCanonicalUpper { upper, modulus })
                if upper == 18_u8.into() && modulus == 17_u8.into()
        ));
    }

    #[test]
    fn plain_extract_uses_authoritative_source_full_modulus_fallback() {
        let protocol = crate::toy_example::protocol();
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "extract-fallback".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
        let matrix_type = super::super::identity::ResolvedMatrixType {
            modulus: ResolvedIntExpr::Const(BigInt::from(17)),
            ring_dimension: ResolvedIntExpr::Const(BigInt::from(1)),
            rows: ResolvedIntExpr::Const(BigInt::from(1)),
            columns: ResolvedIntExpr::Const(BigInt::from(1)),
        };
        let source = lowerer.egraph.analysis.symbols.atomic_sources.intern(
            super::super::identity::AtomicSourceDescriptor {
                key: super::super::identity::AtomicSourceKey::ProtocolInput(
                    crate::ProtocolInputId::from("fallback-input"),
                ),
                sort: MxxSort::Matrix(matrix_type),
                integer_domain: None,
                canonical_residue_convention: Some(CanonicalResidueConvention::Nonnegative),
                relation_role: None,
            },
        );
        let matrix = lowerer.egraph.add(MxxLang::Atom {
            source: super::super::identity::AtomicSourceId(source),
            indices: Box::new([]),
        });
        let kind = NodeKind::ExtractCoefficient {
            position: IntExpr::constant(0),
            canonical_input_exclusive_upper: None,
        };
        let LoweredValue::Term(extract) = lowerer
            .lower_node(&kind, &[LoweredValue::Term(matrix)], &root_test_environment())
            .unwrap()
        else {
            unreachable!()
        };
        assert_eq!(
            lowerer.integer_analysis(extract).unwrap().0.interval().unwrap(),
            super::super::analysis::IntegerInterval::new(0.into(), 16.into()).unwrap()
        );
    }

    #[test]
    fn scalar_consumer_validation_uses_the_consumed_boolean_sort() {
        let protocol = crate::toy_example::protocol();
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "scalar-consumer".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
        let boolean = lowerer.egraph.add(MxxLang::BoolConst(true));
        lowerer
            .validate_boolean_consumer(boolean, SelectorOnlyConsumer::BoolToInt, false)
            .expect("an ordinary boolean is a valid BoolToInt operand");

        let integer = lowerer.egraph.add(MxxLang::IntConst(BigInt::from(1)));
        assert_eq!(
            lowerer.validate_boolean_consumer(integer, SelectorOnlyConsumer::BoolToInt, false,),
            Err(LowerError::InvalidOperandSort { expected: WireType::Bool, actual: WireType::Int })
        );
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
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
        let lowered = lowerer.lower_stage_wire(&stage, output).expect("lower decomposed hash");
        assert!(matches!(lowered, LoweredValue::Term(_)));
        let samplers = &lowerer.egraph.analysis.symbols.samplers.values;
        let [
            SamplerIdentity::DecomposedHash {
                public,
                target,
                arguments,
                matrix_type,
                base,
                digit_count,
                small,
                ..
            },
        ] = samplers.as_slice()
        else {
            panic!("one decomposed hash sampler")
        };
        assert_eq!(arguments.len(), 2, "key then runtime tag arguments are retained");
        assert_eq!(matrix_type.rows, ResolvedIntExpr::Const(BigInt::from(3)));
        assert_eq!(*base, ResolvedIntExpr::Const(BigInt::from(4)));
        assert_eq!(*digit_count, ResolvedIntExpr::Const(BigInt::from(3)));
        assert!(!small);
        let target = lowerer.egraph.find(*target);
        let MxxLang::HashPlain { query, arguments: hash_arguments } =
            lowerer.egraph[target].nodes.first().expect("hash target")
        else {
            panic!("decomposed target is HashPlain")
        };
        assert_eq!(hash_arguments.len(), 2);
        let query = lowerer.egraph.analysis.symbols.hash_queries.get(query.0).expect("query spec");
        assert_eq!(query.matrix_type.rows, ResolvedIntExpr::Const(BigInt::from(1)));
        assert!(matches!(query.tag_program.as_ref(), [
            super::super::identity::HashTagPart::Literal(prefix),
            super::super::identity::HashTagPart::BinaryStatic(_),
            super::super::identity::HashTagPart::DecimalStatic(_),
            super::super::identity::HashTagPart::U64LeStatic(_),
            super::super::identity::HashTagPart::BinaryArgument { argument: 1 },
        ] if prefix.as_ref() == b"hash-fixture"));
        let public = lowerer.egraph.find(*public);
        let MxxLang::MatrixConstant(spec) = lowerer.egraph[public].nodes.first().expect("gadget")
        else {
            panic!("public is gadget matrix constant")
        };
        let gadget =
            lowerer.egraph.analysis.symbols.matrix_constants.get(spec.0).expect("gadget spec");
        assert_eq!(gadget.matrix_type.rows, ResolvedIntExpr::Const(BigInt::from(1)));
        assert_eq!(gadget.matrix_type.columns, ResolvedIntExpr::Const(BigInt::from(3)));
        assert!(matches!(
            gadget.value,
            super::super::identity::MatrixConstantValue::Gadget { small: false, .. }
        ));
        assert_eq!(lowerer.relation_registrations().len(), 1);
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
            MxxAnalysis::default(),
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
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
        assert!(matches!(
            lowerer.lower_stage_wire(&stage, output),
            Err(LowerError::InvalidOperandArity { .. })
        ));
    }

    #[test]
    fn small_decomposed_hash_interns_a_small_sampler_descriptor() {
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
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
        lowerer.lower_stage_wire(&stage, output).expect("lower small decomposed hash");
        assert!(matches!(
            lowerer.egraph.analysis.symbols.samplers.values.as_slice(),
            [SamplerIdentity::DecomposedHash { small: true, range_proved: false, .. }]
        ));
        assert_eq!(lowerer.relation_registrations().len(), 1);
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
    fn completed_shared_dependency_is_reused_but_an_active_back_edge_is_rejected() {
        let protocol = crate::toy_example::protocol();
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "memo-colors".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
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
        let value = LoweredValue::Term(lowerer.egraph.add(MxxLang::IntConst(1.into())));
        lowerer.finish_wire(&wire, value.clone());
        assert!(matches!(
            (lowerer.begin_wire(&wire), &value),
            (Ok(Some(LoweredValue::Term(reused))), LoweredValue::Term(value)) if reused == *value
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
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
        let scope = root_test_environment().occurrence;
        let first =
            BinderKey { loop_scope: scope.clone(), loop_node: mxx_ir_core::NodeId(1), slot: 0 };
        let second = BinderKey { loop_scope: scope, loop_node: mxx_ir_core::NodeId(2), slot: 0 };
        let first_id =
            super::super::identity::BinderId(lowerer.egraph.analysis.symbols.binders.intern(
                super::super::identity::BinderDescriptor {
                    key: first.clone(),
                    minimum: BigInt::zero(),
                    maximum: BigInt::from(511_u16),
                },
            ));
        let second_id =
            super::super::identity::BinderId(lowerer.egraph.analysis.symbols.binders.intern(
                super::super::identity::BinderDescriptor {
                    key: second.clone(),
                    minimum: BigInt::zero(),
                    maximum: BigInt::from(511_u16),
                },
            ));
        let family = |binder: BinderKey, representative| FamilyLoweringValue {
            element_type: MxxSort::Int,
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
        let first_family = family(first.clone(), lowerer.egraph.add(MxxLang::IntBinder(first_id)));
        let second_family = family(second, lowerer.egraph.add(MxxLang::IntBinder(second_id)));
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
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
        let LoweredValue::Trapdoor(id) =
            lowerer.lower_stage_wire(&StageId("encrypt".to_owned()), output).unwrap()
        else {
            panic!("protocol trapdoor input")
        };
        let descriptor = lowerer.egraph.analysis.symbols.trapdoors.get(id.0).unwrap();
        assert_eq!(
            descriptor.source,
            TrapdoorSourceKey::ProtocolInput(crate::ProtocolInputId::from("trapdoor"))
        );
        assert_eq!(descriptor.gadget_base, ResolvedIntExpr::Const(BigInt::from(2)));
        assert_eq!(descriptor.digit_count, ResolvedIntExpr::Const(BigInt::from(3)));
        assert_eq!(descriptor.preimage_cutoff, ResolvedIntExpr::Const(BigInt::from(5)));
        let MxxLang::Atom { source, .. } =
            lowerer.egraph[lowerer.egraph.find(descriptor.public)].nodes.first().unwrap()
        else {
            panic!("public protocol atom")
        };
        assert!(
            matches!(lowerer.egraph.analysis.symbols.atomic_sources.get(source.0).unwrap().key, super::super::identity::AtomicSourceKey::ProtocolInput(ref id) if id == &crate::ProtocolInputId::from("public"))
        );
    }

    #[test]
    fn protocol_trapdoor_input_rejects_a_non_matrix_declared_public_contract() {
        let (protocol, output) = protocol_trapdoor_input_fixture(InputValueContract::Boolean);
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "trapdoor-input".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
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
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
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
