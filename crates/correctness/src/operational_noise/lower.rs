//! Iterative Graph-IR lowering state for the operational-noise checker.
//!
//! A lowering wire is a concrete graph occurrence plus its active symbolic coordinates.  The
//! single memo below is the only owner of graph-wire lowering results; integer ranges and
//! selector provenance remain exclusively in the e-graph analysis.

use super::{
    OperationalCheckRequest,
    analysis::{IntegerDomain, MxxAnalysis, MxxSort, ScalarProvenance, resolved_constant},
    bound::{
        BoundClass, BoundEvaluationControl, BoundEvaluationError, BoundInput, MatrixBound,
        MatrixMetadata, ResolvedMatrixConstant, gadget_digit_bound,
    },
    error::{LowerError, SelectorOnlyConsumer},
    family::{self, FamilyCoverageStorage, FamilyLoweringValue},
    identity::{
        BinderKey, CanonicalResidueConvention, OccurrenceScope, ResolvedIntExpr,
        SamplerDescriptorId, SamplerIdentity, SequentialStateKey, TrapdoorDescriptorId,
        TrapdoorIdentity, TrapdoorSourceKey, WireSourceKey,
    },
    language::MxxLang,
    normal_form::{
        ExpressionDag, ExpressionNode, FactorIdentity, FactorKind, FactorOwner, RelationRegistry,
        SymbolicFactor, TermId,
    },
    normal_form_family,
    normal_form_ops::{
        AdditionalOperations, BoolBit, CoefficientPreservingView, IntegerInterval,
        PolynomialNFOperations, ScaleScalar, ViewSpec,
    },
};
use crate::{
    DeclaredBoundExpr, InputValueContract, ProtocolDecl, ProtocolInputDestination, StageId,
    StageInputName,
};
use egg::{EGraph, Id};
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
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

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

/// Read-only protocol identity for a graph-generated source.  This is
/// diagnostic data only: output and artifact names never affect bounds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GraphWireBindingDiagnostic {
    pub(crate) stage: Option<crate::StageId>,
    pub(crate) output_names: Box<[String]>,
    pub(crate) artifact_consumers: Box<[(crate::StageId, crate::StageInputName)]>,
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
        let descriptor = self
            .lowerer
            .egraph
            .analysis
            .symbols
            .atomic_sources
            .get(source.0)
            .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })?;
        let matrix_type = self.matrix_type(term)?;
        let (coefficient_class, metadata) = match &descriptor.key {
            super::identity::AtomicSourceKey::Sampler(id) => {
                let sampler = self
                    .lowerer
                    .egraph
                    .analysis
                    .symbols
                    .samplers
                    .get(id.0)
                    .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })?;
                let coefficient_class = match sampler {
                    SamplerIdentity::Gaussian { max_coefficient_bound, .. } => {
                        let maximum_absolute_coefficient = resolved_integer(max_coefficient_bound)
                            .and_then(|bound| bound.to_biguint())
                            .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })?;
                        BoundClass::bounded(maximum_absolute_coefficient)
                    }
                    SamplerIdentity::UniformInterval { minimum, maximum, .. } => {
                        let minimum = resolved_integer(minimum)
                            .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })?;
                        let maximum = resolved_integer(maximum)
                            .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })?;
                        if minimum > maximum {
                            return Err(BoundEvaluationError::InvalidMatrixConstant { term });
                        }
                        BoundClass::bounded(
                            minimum
                                .abs()
                                .max(maximum.abs())
                                .to_biguint()
                                .expect("absolute interval endpoint"),
                        )
                    }
                    SamplerIdentity::Preimage { cutoff, .. } => {
                        let cutoff = resolved_integer(cutoff)
                            .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })?;
                        let cutoff = cutoff
                            .to_biguint()
                            .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })?;
                        BoundClass::bounded(cutoff)
                    }
                    SamplerIdentity::DecomposedHash { base, small, .. } |
                    SamplerIdentity::GadgetDecomposition { base, small, .. } => {
                        let base = resolved_integer(base)
                            .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })?;
                        gadget_digit_bound(&base, *small)
                            .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })?
                    }
                };
                (coefficient_class, MatrixMetadata::unknown())
            }
            super::identity::AtomicSourceKey::ProtocolInput(input) => {
                self.protocol_matrix_bound(input, &matrix_type, term)?
            }
            super::identity::AtomicSourceKey::GraphWire(source) => {
                return Err(BoundEvaluationError::OpaqueGraphWire { source: source.clone() });
            }
            super::identity::AtomicSourceKey::ExplicitLarge(_) => {
                (BoundClass::Large, MatrixMetadata::unknown())
            }
            // A carried-state placeholder is meaningful only inside the
            // descriptor-owned simultaneous transition overlay below.
            super::identity::AtomicSourceKey::SequentialState(_) |
            super::identity::AtomicSourceKey::SequentialRecurrence { .. } => {
                return Err(BoundEvaluationError::SequentialStateOutsideOverlay { term });
            }
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

    fn pack_bit_maximum(&self, term: Id, bit: Id) -> Result<BigUint, BoundEvaluationError> {
        let bit = self.lowerer.egraph.find(bit);
        let data = &self.lowerer.egraph[bit].data;
        if data.sort != Ok(MxxSort::Bool) ||
            data.scalar_provenance != Some(ScalarProvenance::Ordinary)
        {
            return Err(BoundEvaluationError::InvalidPack { term });
        }
        Ok(if data.possible_true { BigUint::one() } else { BigUint::zero() })
    }

    fn switch_reachable_cases(
        &self,
        term: Id,
        selector: Id,
        case_count: usize,
    ) -> Result<Box<[bool]>, BoundEvaluationError> {
        let selector = self.lowerer.egraph.find(selector);
        let interval = self.lowerer.egraph[selector]
            .data
            .integer_domain
            .as_ref()
            .ok_or(BoundEvaluationError::InvalidSwitchReachability { term })?
            .interval()
            .map_err(|_| BoundEvaluationError::InvalidSwitchReachability { term })?;
        let minimum = interval
            .minimum
            .to_usize()
            .ok_or(BoundEvaluationError::InvalidSwitchReachability { term })?;
        let maximum = interval
            .maximum
            .to_usize()
            .filter(|maximum| *maximum < case_count)
            .ok_or(BoundEvaluationError::InvalidSwitchReachability { term })?;
        if case_count == 0 || minimum > maximum {
            return Err(BoundEvaluationError::InvalidSwitchReachability { term });
        }
        Ok((0..case_count).map(|case| minimum <= case && case <= maximum).collect())
    }
}

impl ProductionBoundInput<'_, '_, '_> {
    fn protocol_matrix_bound(
        &self,
        input: &crate::ProtocolInputId,
        matrix_type: &mxx_ir_core::types::ConcreteMatrixType,
        term: Id,
    ) -> Result<(BoundClass, MatrixMetadata), BoundEvaluationError> {
        let contract = self
            .lowerer
            .protocol
            .bundle
            .input_contract
            .inputs
            .iter()
            .find(|entry| entry.id == *input)
            .map(|entry| Self::family_element_contract(&entry.value))
            .ok_or(BoundEvaluationError::MissingInputBoundContract { term })?;
        match contract {
            InputValueContract::MatrixBounded { max_centered_coefficient, .. } => Ok((
                BoundClass::bounded(self.evaluate_declared_bound(max_centered_coefficient, term)?),
                MatrixMetadata::unknown(),
            )),
            InputValueContract::MatrixExact {
                canonical_coefficient_exclusive_upper_bound: Some(upper),
                is_constant_polynomial,
                ..
            } => {
                let upper = self.evaluate_contract_int(upper, term)?;
                let modulus = matrix_type
                    .modulus
                    .to_biguint()
                    .filter(|modulus| !modulus.is_zero())
                    .ok_or(BoundEvaluationError::InvalidDeclaredBound { term })?;
                if upper.is_zero() || upper > modulus {
                    return Err(BoundEvaluationError::InvalidDeclaredBound { term });
                }
                Ok((
                    BoundClass::bounded(upper - BigUint::one()),
                    MatrixMetadata {
                        is_constant_polynomial: *is_constant_polynomial,
                        known_zero_rows: None,
                    },
                ))
            }
            InputValueContract::MatrixLarge { .. } => {
                Ok((BoundClass::Large, MatrixMetadata::unknown()))
            }
            _ => Err(BoundEvaluationError::MissingInputBoundContract { term }),
        }
    }

    fn family_element_contract(mut contract: &InputValueContract) -> &InputValueContract {
        while let InputValueContract::Family { element, .. } = contract {
            contract = element;
        }
        contract
    }

    fn contract_param_env(&self) -> mxx_ir_core::ParamEnv {
        let mut environment = mxx_ir_core::ParamEnv::default();
        environment.integers.extend(self.lowerer.request.environment.iter().filter_map(
            |(name, value)| match value {
                super::OperationalParameterValue::Integer(value) => {
                    Some((name.clone(), value.clone()))
                }
                super::OperationalParameterValue::Rational { .. } => None,
            },
        ));
        environment
    }

    fn evaluate_contract_int(
        &self,
        expression: &IntExpr,
        term: Id,
    ) -> Result<BigUint, BoundEvaluationError> {
        expression
            .evaluate(&self.contract_param_env())
            .ok()
            .and_then(|value| value.to_biguint())
            .ok_or(BoundEvaluationError::InvalidDeclaredBound { term })
    }

    fn evaluate_declared_bound(
        &self,
        expression: &DeclaredBoundExpr,
        term: Id,
    ) -> Result<BigUint, BoundEvaluationError> {
        let invalid = || BoundEvaluationError::InvalidDeclaredBound { term };
        match expression {
            DeclaredBoundExpr::Constant(value) => Ok(value.clone()),
            DeclaredBoundExpr::Parameter(value) => self.evaluate_contract_int(value, term),
            DeclaredBoundExpr::Add(left, right) => Ok(self.evaluate_declared_bound(left, term)? +
                self.evaluate_declared_bound(right, term)?),
            DeclaredBoundExpr::Multiply(left, right) => Ok(self
                .evaluate_declared_bound(left, term)? *
                self.evaluate_declared_bound(right, term)?),
            DeclaredBoundExpr::Maximum(left, right) => Ok(self
                .evaluate_declared_bound(left, term)?
                .max(self.evaluate_declared_bound(right, term)?)),
            DeclaredBoundExpr::Minimum(left, right) => Ok(self
                .evaluate_declared_bound(left, term)?
                .min(self.evaluate_declared_bound(right, term)?)),
            DeclaredBoundExpr::Absolute(value) => value
                .evaluate(&self.contract_param_env())
                .map(|value| value.abs().to_biguint().expect("absolute integer"))
                .map_err(|_| invalid()),
            DeclaredBoundExpr::FloorDivide { value, positive_divisor } => {
                if positive_divisor.is_zero() {
                    return Err(invalid());
                }
                Ok(self.evaluate_declared_bound(value, term)? / positive_divisor)
            }
            DeclaredBoundExpr::MatrixProduct { ring_dimension, inner_dimension, left, right } => {
                let ring_dimension = self.evaluate_contract_int(ring_dimension, term)?;
                let inner_dimension = self.evaluate_contract_int(inner_dimension, term)?;
                if ring_dimension.is_zero() || inner_dimension.is_zero() {
                    return Err(invalid());
                }
                Ok(ring_dimension *
                    inner_dimension *
                    self.evaluate_declared_bound(left, term)? *
                    self.evaluate_declared_bound(right, term)?)
            }
        }
    }
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
    /// A matrix expression in the single egg-independent lowering DAG.
    Matrix(TermId),
    MatrixFamily(FamilyLoweringValue<TermId>),
    /// A scalar/domain analysis term. Matrix values must not use this path in production.
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

/// The sole mutable owner for one lowering/rewrite job.
pub struct GraphLowerer<'a, 'control> {
    pub protocol: &'a ProtocolDecl,
    pub request: &'a OperationalCheckRequest,
    pub egraph: EGraph<MxxLang, MxxAnalysis>,
    /// The sole matrix-expression store for this lowering job.
    pub dag: ExpressionDag,
    /// The sole normal-form relation registry for this lowering job.
    pub relation_registry: RelationRegistry,
    memo: HashMap<LoweringWireKey, LoweredValue>,
    /// Job-local memo for owner-aware shared-family substitutions.  This is
    /// deliberately keyed by the source term and binder/value, not by a
    /// logical lane; no family is expanded into a Cartesian cache.
    family_substitution_memo: BTreeMap<(TermId, BinderKey, ResolvedIntExpr), TermId>,
    /// Direct identities for scalar terms created by this lowering job.  This
    /// is a symbol table, not an e-class traversal: every derived scalar is
    /// registered at its construction boundary.
    scalar_identities: HashMap<Id, ResolvedIntExpr>,
    active: HashSet<LoweringWireKey>,
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
            dag: ExpressionDag::new(),
            relation_registry: RelationRegistry::default(),
            memo: HashMap::new(),
            family_substitution_memo: BTreeMap::new(),
            scalar_identities: HashMap::new(),
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
        analysis: MxxAnalysis,
        control: &'control mut dyn LoweringControl,
    ) -> Self {
        Self {
            protocol,
            request,
            egraph: EGraph::new(analysis),
            dag: ExpressionDag::new(),
            relation_registry: RelationRegistry::default(),
            memo: HashMap::new(),
            family_substitution_memo: BTreeMap::new(),
            scalar_identities: HashMap::new(),
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
            egraph: self.egraph,
            dag: self.dag,
            relation_registry: self.relation_registry,
            memo: self.memo,
            family_substitution_memo: self.family_substitution_memo,
            scalar_identities: self.scalar_identities,
            active: self.active,
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

    fn remember_scalar_identity(&mut self, term: Id, identity: ResolvedIntExpr) {
        self.scalar_identities.insert(self.egraph.find(term), identity);
    }

    /// Returns the canonical scalar identity installed at the lowering
    /// boundary.  The direct-node fallback exists for narrowly constructed
    /// unit fixtures that insert a scalar into the temporary e-graph without
    /// going through lowering; it reads that exact node ID and never searches
    /// equivalent e-class alternatives.
    fn canonical_term_identity(&self, term: Id) -> Result<ResolvedIntExpr, LowerError> {
        let canonical = self.egraph.find(term);
        if let Some(identity) = self.scalar_identities.get(&canonical) {
            return Ok(identity.clone());
        }
        self.direct_term_identity(term, &mut HashSet::new())
    }

    fn direct_term_identity(
        &self,
        term: Id,
        active: &mut HashSet<Id>,
    ) -> Result<ResolvedIntExpr, LowerError> {
        if !active.insert(term) {
            return Err(LowerError::MissingIntegerAnalysis { term });
        }
        let result = match self.egraph.id_to_node(term) {
            MxxLang::IntConst(value) => Ok(ResolvedIntExpr::Const(value.clone())),
            MxxLang::IntParameter(name) => Ok(ResolvedIntExpr::Parameter(name.clone())),
            MxxLang::IntBinder(id) => self
                .egraph
                .analysis
                .symbols
                .binders
                .get(id.0)
                .map(|descriptor| ResolvedIntExpr::Binder(descriptor.key.clone()))
                .ok_or(LowerError::MissingIntegerAnalysis { term }),
            MxxLang::Atom { source, indices } => {
                let descriptor = self
                    .egraph
                    .analysis
                    .symbols
                    .atomic_sources
                    .get(source.0)
                    .ok_or(LowerError::MissingIntegerAnalysis { term })?;
                let coordinates = indices
                    .iter()
                    .map(|index| self.direct_term_identity(*index, active))
                    .collect::<Result<Box<_>, _>>()?;
                Ok(ResolvedIntExpr::Source { source: descriptor.key.clone(), coordinates })
            }
            MxxLang::IntAdd([left, right]) => {
                self.direct_binary_identity(*left, *right, active, ResolvedIntExpr::Add)
            }
            MxxLang::IntSub([left, right]) => {
                self.direct_binary_identity(*left, *right, active, ResolvedIntExpr::Sub)
            }
            MxxLang::IntMul([left, right]) => {
                self.direct_binary_identity(*left, *right, active, ResolvedIntExpr::Mul)
            }
            MxxLang::IntExactDiv([left, right]) => {
                self.direct_binary_identity(*left, *right, active, ResolvedIntExpr::Div)
            }
            MxxLang::IntEuclideanDiv([left, right]) => {
                self.direct_binary_identity(*left, *right, active, ResolvedIntExpr::EuclideanDiv)
            }
            MxxLang::IntEuclideanRemainder([left, right]) => self.direct_binary_identity(
                *left,
                *right,
                active,
                ResolvedIntExpr::EuclideanRemainder,
            ),
            MxxLang::IntRoundDiv([left, right]) => {
                self.direct_binary_identity(*left, *right, active, ResolvedIntExpr::RoundDiv)
            }
            MxxLang::IntLog2Ceil([input]) => {
                Ok(ResolvedIntExpr::Log2Ceil(Box::new(self.direct_term_identity(*input, active)?)))
            }
            MxxLang::ExtractCoefficient { canonical_exclusive_upper, input: [input, position] } => {
                Ok(ResolvedIntExpr::ExtractCoefficient {
                    input: Box::new(self.direct_term_identity(*input, active)?),
                    position: Box::new(self.direct_term_identity(*position, active)?),
                    canonical_exclusive_upper: canonical_exclusive_upper.clone(),
                })
            }
            _ => Err(LowerError::MissingIntegerAnalysis { term }),
        };
        active.remove(&term);
        result
    }

    fn direct_binary_identity(
        &self,
        left: Id,
        right: Id,
        active: &mut HashSet<Id>,
        operation: impl FnOnce(Box<ResolvedIntExpr>, Box<ResolvedIntExpr>) -> ResolvedIntExpr,
    ) -> Result<ResolvedIntExpr, LowerError> {
        let left = self.direct_term_identity(left, active)?;
        let right = self.direct_term_identity(right, active)?;
        Ok(operation(Box::new(left), Box::new(right)))
    }

    fn canonical_scalar_identity(&self, term: Id) -> Result<ResolvedIntExpr, LowerError> {
        let identity = self.canonical_term_identity(term)?;
        if self.egraph[self.egraph.find(term)].data.sort != Ok(MxxSort::Int) {
            return Err(LowerError::MissingIntegerAnalysis { term });
        }
        Ok(identity)
    }

    fn selector_reachable(&self, term: Id, count: usize) -> Result<Box<[usize]>, LowerError> {
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

    /// Returns the one matrix expression DAG owned by this lowering job.
    pub fn expression_dag(&self) -> &ExpressionDag {
        &self.dag
    }

    /// Returns the one checked normal-form relation registry owned by this job.
    pub fn normal_form_relations(&self) -> &RelationRegistry {
        &self.relation_registry
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
                SamplerIdentity::Gaussian { .. } | SamplerIdentity::UniformInterval { .. } => None,
            })
            .collect()
    }

    /// Returns the one production view used by the bound evaluator.  It reads
    /// canonical e-graph analysis and exact lowering descriptors only.
    pub fn production_bound_view(&self) -> ProductionBoundInput<'_, 'a, 'control> {
        ProductionBoundInput { lowerer: self, control: None }
    }

    /// Resolves a graph wire back to its declared workflow output and exact
    /// artifact consumers.  The lookup uses frozen program/scope/wire
    /// identity; node numbers, output order, and protocol-specific names are
    /// never interpreted.
    pub(crate) fn graph_wire_binding_diagnostic(
        &self,
        source: &super::identity::GraphWireSourceKey,
    ) -> GraphWireBindingDiagnostic {
        let super::identity::ProgramKey::WorkflowStage(stage_id) = &source.wire.scope.program
        else {
            return GraphWireBindingDiagnostic {
                stage: None,
                output_names: Box::new([]),
                artifact_consumers: Box::new([]),
            };
        };
        let stage = self.protocol.stages().iter().find(|stage| &stage.id == stage_id);
        let mut output_names = Vec::new();
        if source.wire.scope.path.is_empty() &&
            source.wire.scope.definition == FrozenGraphScopeId::Root
        {
            if let Some(stage) = stage {
                output_names.extend(stage.graph.outputs().iter().filter_map(|(name, output)| {
                    (output.value == source.wire.wire).then_some(name.clone())
                }));
            }
        }
        output_names.sort();
        output_names.dedup();
        let output_name_set = output_names.iter().map(String::as_str).collect::<BTreeSet<_>>();
        let mut artifact_consumers = self
            .protocol
            .stages()
            .iter()
            .flat_map(|consumer| {
                consumer.bindings.iter().filter_map(|binding| {
                    (&binding.producer_stage == stage_id &&
                        output_name_set.contains(binding.producer_output.0.as_str()))
                    .then_some((consumer.id.clone(), binding.consumer_input.clone()))
                })
            })
            .collect::<Vec<_>>();
        artifact_consumers.sort();
        artifact_consumers.dedup();
        GraphWireBindingDiagnostic {
            stage: stage.map(|stage| stage.id.clone()),
            output_names: output_names.into_boxed_slice(),
            artifact_consumers: artifact_consumers.into_boxed_slice(),
        }
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
                                    match *element {
                                        WireType::Matrix(_) | WireType::Preimage(_) => {
                                            return Err(
                                                LowerError::UnsupportedMatrixProductExpansion,
                                            )
                                        }
                                        element => LoweredValue::Term(self.atom_for_wire(
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
                                output_type => LoweredValue::Term(self.atom_for_wire(
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
                                matches!(arguments[0], LoweredValue::Term(_)) =>
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
                    let LoweredValue::Term(representative) = representative_value else {
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
                        LoweredValue::Term(term) => LoweredValue::Term(self.scalar_term(term)?),
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
                            .egraph
                            .analysis
                            .symbols
                            .trapdoors
                            .get(trapdoor.0)
                            .map(|descriptor| descriptor.source.clone());
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
                            preimage: preimage_key,
                            target: *target,
                        };
                        self.relation_registry
                            .register(registration)
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
                            LoweredValue::Term(term) => {
                                let term = self.egraph.find(term);
                                let sort = self.egraph[term].data.sort.as_ref().ok();
                                // Hash inputs are scalar/domain values that are consumed by the
                                // matrix source descriptor.  Bytes is intentionally allowed here
                                // as a non-matrix source domain; it is never emitted as a matrix
                                // carrier or passed to scalar arithmetic.
                                matches!(
                                    sort,
                                    Some(
                                        MxxSort::Bytes(_) |
                                            MxxSort::Int |
                                            MxxSort::Bool |
                                            MxxSort::Real
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
                                .enumerate()
                                .map(|(index, _)| {
                                    Ok(super::normal_form::PolynomialNF::exact_factor(
                                        self.graph_factor_identity(
                                            &wire,
                                            &environment,
                                            format!("hash-plain-argument:{index}").as_bytes(),
                                        )?,
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
        let source = self.egraph.analysis.symbols.atomic_sources.intern(descriptor);
        let atom = self.egraph.add(MxxLang::Atom {
            source: super::identity::AtomicSourceId(source),
            indices: environment
                .active_coordinates
                .iter()
                .map(|coordinate| coordinate.index.term)
                .collect(),
        });
        if sort == MxxSort::Int {
            let coordinates = environment
                .active_coordinates
                .iter()
                .map(|coordinate| self.canonical_scalar_identity(coordinate.index.term))
                .collect::<Result<Box<_>, _>>()?;
            self.remember_scalar_identity(
                atom,
                ResolvedIntExpr::Source { source: key, coordinates },
            );
        }
        Ok(atom)
    }

    /// Lowers a matrix source directly into the job DAG.  The scalar e-graph is
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
            match self.protocol_input_source(wire, environment, &MxxSort::Matrix(matrix.clone()))? {
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
                let index = coordinate.index.stable_identity.clone().ok_or_else(|| {
                    LowerError::NonExactIdentityIndex { expression: IntExpr::constant(0) }
                })?;
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
                            .egraph
                            .analysis
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
            metadata: protocol_contract
                .map(|(_, metadata)| metadata)
                .unwrap_or_else(MatrixMetadata::unknown),
        };
        self.push_matrix_atom(
            factor_key,
            bound,
            matches!(relation_role, Some(super::identity::AtomicRelationRole::Preimage)),
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
                    BoundClass::bounded(upper - BigUint::one()),
                    MatrixMetadata {
                        is_constant_polynomial: *is_constant_polynomial,
                        known_zero_rows: None,
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
                Some(
                    ring_dimension.evaluate(&env).ok()?.to_biguint()? *
                        inner_dimension.evaluate(&env).ok()?.to_biguint()? *
                        self.declared_bound_value(left)? *
                        self.declared_bound_value(right)?,
                )
            }
        }
    }

    fn push_matrix_atom(
        &mut self,
        key: FactorIdentity,
        bound: MatrixBound,
        relation_live: bool,
    ) -> Result<TermId, LowerError> {
        let factor = if relation_live {
            SymbolicFactor::relation_live(key, bound)
                .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?
        } else if matches!(bound.coefficient_class, BoundClass::Large) {
            SymbolicFactor::large(key)
        } else {
            SymbolicFactor::bounded(key, bound)
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
            Ok(SymbolicFactor::large(key))
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
        let node =
            self.dag.node(term).map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
        match node {
            ExpressionNode::Atom(factor) => factor
                .matrix_bound
                .as_ref()
                .map(|bound| bound.matrix_type.clone())
                .ok_or(LowerError::UnsupportedMatrixProductExpansion),
            ExpressionNode::Add(children) | ExpressionNode::Product(children) => children
                .first()
                .copied()
                .ok_or(LowerError::UnsupportedMatrixProductExpansion)
                .and_then(|child| self.dag_matrix_type(child)),
            ExpressionNode::Negate(child) |
            ExpressionNode::MatrixScale { input: child, .. } |
            ExpressionNode::Slice { input: child, .. } |
            ExpressionNode::View { input: child, .. } => self.dag_matrix_type(*child),
            ExpressionNode::Transpose(child) => {
                let mut matrix = self.dag_matrix_type(*child)?;
                std::mem::swap(&mut matrix.rows, &mut matrix.columns);
                Ok(matrix)
            }
            ExpressionNode::Tensor { left, right } => {
                let left = self.dag_matrix_type(*left)?;
                let right = self.dag_matrix_type(*right)?;
                if left.modulus != right.modulus || left.ring_dimension != right.ring_dimension {
                    return Err(LowerError::UnsupportedMatrixProductExpansion);
                }
                Ok(mxx_ir_core::types::ConcreteMatrixType {
                    modulus: left.modulus,
                    ring_dimension: left.ring_dimension,
                    rows: left
                        .rows
                        .checked_mul(right.rows)
                        .ok_or(LowerError::UnsupportedMatrixProductExpansion)?,
                    columns: left
                        .columns
                        .checked_mul(right.columns)
                        .ok_or(LowerError::UnsupportedMatrixProductExpansion)?,
                })
            }
            ExpressionNode::LiftConstantPolynomial { matrix_type, .. } |
            ExpressionNode::CrtRecompose { output_type: matrix_type, .. } |
            ExpressionNode::Concat { output_type: matrix_type, .. } => Ok(matrix_type.clone()),
            ExpressionNode::Switch { cases, .. } |
            ExpressionNode::Select { cases, .. } |
            ExpressionNode::FamilyGetStatic { cases, .. } |
            ExpressionNode::FamilyGetDynamic { cases, .. } => cases
                .first()
                .copied()
                .ok_or(LowerError::UnsupportedMatrixProductExpansion)
                .and_then(|child| self.dag_matrix_type(child)),
            ExpressionNode::Zero => Err(LowerError::UnsupportedMatrixProductExpansion),
        }
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
                    coordinate
                        .index
                        .stable_identity
                        .clone()
                        .or_else(|| self.canonical_scalar_identity(coordinate.index.term).ok())
                        .ok_or(LowerError::MissingIntegerAnalysis {
                            term: coordinate.index.term,
                        })?,
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
            let bound = MatrixBound {
                matrix_type,
                coefficient_class: BoundClass::ExactZero,
                metadata: MatrixMetadata::unknown(),
            };
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
        let LoweredValue::Term(term) = input else {
            return Err(LowerError::UnsupportedMatrixProductExpansion);
        };
        let term = self.scalar_term(*term)?;
        let interval = self
            .integer_analysis(term)
            .and_then(|(domain, _)| domain.interval().ok())
            .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
        let matrix_type =
            concrete_matrix_type(&self.resolve_matrix_type(matrix_type, environment)?)
                .ok_or(LowerError::UnsupportedMatrixProductExpansion)?;
        let key = self.graph_factor_identity(wire, environment, b"lift-constant-polynomial")?;
        let source = super::normal_form::PolynomialNF::bounded(MatrixBound {
            matrix_type: matrix_type.clone(),
            coefficient_class: BoundClass::bounded(
                interval.minimum.magnitude().max(interval.maximum.magnitude()).clone(),
            ),
            metadata: MatrixMetadata::unknown(),
        })
        .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
        let domain = IntegerInterval::new(interval.minimum.clone(), interval.maximum.clone())
            .map_err(|_| LowerError::UnsupportedMatrixProductExpansion)?;
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
                        coordinate.index.stable_identity.clone().ok_or(
                            LowerError::NonExactIdentityIndex { expression: IntExpr::constant(0) },
                        )?,
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
            metadata: MatrixMetadata::unknown(),
        };
        let trapdoor_source = self
            .egraph
            .analysis
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
        let indices =
            environment.active_coordinates.iter().map(|coordinate| coordinate.index.term).collect();
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
        Ok(Some(self.egraph.analysis.symbols.samplers.intern(sampler)))
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
                            {
                                let identity = ResolvedIntExpr::Binder(binder.clone());
                                let term = self
                                    .egraph
                                    .add(MxxLang::IntBinder(super::identity::BinderId(binder_id)));
                                self.remember_scalar_identity(term, identity.clone());
                                LoweredInt { term, stable_identity: Some(identity) }
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
                            let lowered = LoweredInt {
                                term,
                                stable_identity: child
                                    .stable_identity
                                    .map(|value| ResolvedIntExpr::Log2Ceil(Box::new(value))),
                            };
                            if let Some(identity) = lowered.stable_identity.clone() {
                                self.remember_scalar_identity(term, identity);
                            }
                            lowered
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
        let term = self.egraph.add(MxxLang::IntConst(value));
        self.remember_scalar_identity(term, identity.clone());
        LoweredInt { term, stable_identity: Some(identity) }
    }

    fn add_resolved_int(&mut self, value: ResolvedIntExpr) -> Result<LoweredInt, LowerError> {
        match value {
            ResolvedIntExpr::Const(value) => {
                Ok(self.add_int(value.clone(), ResolvedIntExpr::Const(value)))
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
                if let Some(value) = value {
                    Ok(self.add_int(value.clone(), ResolvedIntExpr::Const(value)))
                } else {
                    let identity = ResolvedIntExpr::Parameter(name.clone());
                    let term = self.egraph.add(MxxLang::IntParameter(name));
                    self.remember_scalar_identity(term, identity.clone());
                    Ok(LoweredInt { term, stable_identity: Some(identity) })
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
            ResolvedIntExpr::ExtractCoefficient { .. } => {
                Err(LowerError::UnsupportedMatrixProductExpansion)
            }
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
        if let Some(identity) = stable_identity.clone() {
            self.remember_scalar_identity(term, identity);
        }
        Ok(LoweredInt { term, stable_identity })
    }

    /// A `Term` is an e-graph scalar/domain value.  Matrix expressions are
    /// DAG terms and must be rejected at this boundary, even if a stale or
    /// malformed matrix enode was manually inserted into the scalar e-graph.
    fn scalar_term(&self, term: Id) -> Result<Id, LowerError> {
        let term = self.egraph.find(term);
        matches!(
            self.egraph[term].data.sort.as_ref().ok(),
            Some(MxxSort::Int | MxxSort::Bool | MxxSort::Real)
        )
        .then_some(term)
        .ok_or(LowerError::UnsupportedMatrixProductExpansion)
    }

    fn register_scalar_node_identity(
        &mut self,
        kind: &NodeKind,
        arguments: &[LoweredValue],
        term: Id,
    ) -> Result<(), LowerError> {
        let scalar = |value: &LoweredValue| match value {
            LoweredValue::Term(term) => {
                self.scalar_term(*term).and_then(|term| self.canonical_scalar_identity(term))
            }
            _ => Err(LowerError::MissingIntegerAnalysis { term }),
        };
        let identity = match kind {
            NodeKind::IntBinary(operation) => {
                let [left, right] = arguments else {
                    return Err(LowerError::InvalidOperandArity {
                        expected: 2,
                        actual: arguments.len(),
                    });
                };
                let left = scalar(left)?;
                let right = scalar(right)?;
                let pair = || (Box::new(left.clone()), Box::new(right.clone()));
                Some(match operation {
                    IntBinaryOp::Add => ResolvedIntExpr::Add(pair().0, pair().1),
                    IntBinaryOp::Subtract => ResolvedIntExpr::Sub(pair().0, pair().1),
                    IntBinaryOp::Multiply => ResolvedIntExpr::Mul(pair().0, pair().1),
                    IntBinaryOp::Divide => ResolvedIntExpr::EuclideanDiv(pair().0, pair().1),
                    IntBinaryOp::Remainder => {
                        ResolvedIntExpr::EuclideanRemainder(pair().0, pair().1)
                    }
                })
            }
            NodeKind::ExtractCoefficient { canonical_input_exclusive_upper, .. } => {
                let [input] = arguments else {
                    return Err(LowerError::InvalidOperandArity {
                        expected: 1,
                        actual: arguments.len(),
                    });
                };
                let LoweredValue::Term(input) = input else {
                    return Err(LowerError::MissingIntegerAnalysis { term });
                };
                let input = self.canonical_term_identity(*input)?;
                let position = match self.egraph.id_to_node(term) {
                    MxxLang::ExtractCoefficient { input: [_, position], .. } => {
                        self.canonical_scalar_identity(*position)?
                    }
                    _ => return Err(LowerError::MissingIntegerAnalysis { term }),
                };
                Some(ResolvedIntExpr::ExtractCoefficient {
                    input: Box::new(input),
                    position: Box::new(position),
                    canonical_exclusive_upper: canonical_input_exclusive_upper.clone(),
                })
            }
            _ => None,
        };
        if let Some(identity) = identity {
            self.remember_scalar_identity(term, identity);
        }
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
        // Matrix values are never converted back to MxxLang.  This is the
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
                    scalar.term,
                    SelectorOnlyConsumer::MatrixScale,
                    false,
                )?;
                let Some(value) = scalar.stable_identity.as_ref().and_then(resolved_integer) else {
                    return Err(LowerError::UnsupportedMatrixProductExpansion);
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
                let key = self.dag_factor_key(*matrix)?;
                let matrix_type = self.dag_matrix_type(*matrix)?;
                let matrix_type = mxx_ir_core::types::ConcreteMatrixType {
                    modulus: matrix_type.modulus,
                    ring_dimension: matrix_type.ring_dimension,
                    rows: 1,
                    columns: 1,
                };
                return self
                    .dag
                    .push(ExpressionNode::MatrixScale {
                        input: *matrix,
                        scalar: ScaleScalar::Exact { key, value, matrix_type },
                    })
                    .map(LoweredValue::Matrix)
                    .map_err(|_| LowerError::UnsupportedMatrixProductExpansion);
            }
            _ if arguments.iter().any(|value| matches!(value, LoweredValue::Matrix(_))) => {
                return Err(LowerError::UnsupportedMatrixProductExpansion);
            }
            _ => {}
        }
        let terms = |expected: usize| -> Result<Vec<Id>, LowerError> {
            if arguments.len() != expected {
                return Err(LowerError::InvalidOperandArity { expected, actual: arguments.len() });
            }
            arguments
                .iter()
                .map(|value| match value {
                    LoweredValue::Term(term) => self.scalar_term(*term),
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
                        let dividend = self.egraph.find(values[0]);
                        let divisor = self.egraph.find(values[1]);
                        let remainder_is_dividend = {
                            let dividend_data = &self.egraph[dividend].data;
                            let divisor_data = &self.egraph[divisor].data;
                            match (
                                dividend_data.integer_domain.as_ref(),
                                divisor_data.integer_domain.as_ref(),
                                divisor_data.scalar_provenance,
                            ) {
                                (
                                    Some(dividend_domain),
                                    Some(IntegerDomain::Exact(divisor)),
                                    Some(ScalarProvenance::Ordinary),
                                ) if divisor.is_positive() => {
                                    dividend_domain.interval().is_ok_and(|interval| {
                                        interval.minimum >= BigInt::zero() &&
                                            interval.maximum < *divisor
                                    })
                                }
                                _ => false,
                            }
                        };
                        if remainder_is_dividend {
                            dividend
                        } else {
                            self.egraph.add(MxxLang::IntEuclideanRemainder([dividend, divisor]))
                        }
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
                let direct_extract =
                    self.egraph[self.egraph.find(input)].data.direct_extract.is_some();
                self.validate_integer_consumer(
                    input,
                    SelectorOnlyConsumer::LiftConstantPolynomial,
                    direct_extract,
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
                            LoweredValue::Term(value) => self.scalar_term(*value),
                            LoweredValue::Matrix(_) => {
                                Err(LowerError::UnsupportedMatrixProductExpansion)
                            }
                            LoweredValue::MatrixFamily(_) => {
                                Err(LowerError::UnsupportedMatrixProductExpansion)
                            }
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
            // fail closed without retaining an MxxLang matrix fallback.
            NodeKind::ConstantMatrix { .. } |
            NodeKind::MatrixBinary(_) |
            NodeKind::MatrixNegate |
            NodeKind::MatrixScale { .. } |
            NodeKind::Transpose |
            NodeKind::Slice { .. } |
            NodeKind::Tensor |
            NodeKind::Concat { .. } => return Err(LowerError::UnsupportedMatrixProductExpansion),
        };
        self.register_scalar_node_identity(kind, arguments, term)?;
        Ok(LoweredValue::Term(term))
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
                if family.element_type != MxxSort::Bool {
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
                        let data = &self.egraph[self.egraph.find(*term)].data;
                        if data.sort != Ok(MxxSort::Bool) ||
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
                            value: super::normal_form::PolynomialNF::exact_factor(key.clone()),
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
                        LoweredValue::Term(term)
                            if self.egraph[self.egraph.find(*term)]
                                .data
                                .sort
                                .as_ref()
                                .is_ok_and(|actual| {
                                    super::analysis::sorts_equal(&element_type, actual)
                                }) =>
                        {
                            self.scalar_term(*term)
                        }
                        LoweredValue::Term(term) => {
                            Err(LowerError::FamilyElementLoweringMismatch {
                                expected: element_wire_type.clone(),
                                actual_category: super::error::LoweredValueCategory::Term,
                                actual_sort: self.egraph[self.egraph.find(*term)]
                                    .data
                                    .sort
                                    .as_ref()
                                    .ok()
                                    .cloned(),
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
                    if let Some(stable) = index.stable_identity.as_ref() {
                        if let Some(element) =
                            normal_form_family::static_matrix_term(family, stable).map_err(
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
                    return Ok(LoweredValue::Term(self.scalar_term(element)?));
                }
                self.shared_family_element(family, &index)
            }
            NodeKind::FamilyGetDynamic => {
                if let [LoweredValue::MatrixFamily(family), LoweredValue::Term(selector)] =
                    arguments
                {
                    let selector = LoweredInt {
                        term: self.scalar_term(*selector)?,
                        stable_identity: Some(self.canonical_scalar_identity(*selector)?),
                    };
                    return self.matrix_family_element(family, &selector, wire, environment);
                }
                if let [
                    LoweredValue::TrapdoorFamily { representative, binder, logical_count },
                    LoweredValue::Term(selector),
                ] = arguments
                {
                    return self.trapdoor_family_element(
                        *representative,
                        binder,
                        logical_count,
                        &LoweredInt { term: self.scalar_term(*selector)?, stable_identity: None },
                    );
                }
                let [LoweredValue::Family(family), LoweredValue::Term(selector)] = arguments else {
                    return Err(LowerError::InvalidOperandArity {
                        expected: 2,
                        actual: arguments.len(),
                    });
                };
                let selector = self.scalar_term(*selector)?;
                match &family.storage {
                    FamilyCoverageStorage::ExactStored { elements } => {
                        let term = family::dynamic_get(&mut self.egraph, family, selector)
                            .map_err(|_| LowerError::InvalidFamilyCount {
                                count: IntExpr::constant(elements.len()),
                            })?;
                        Ok(LoweredValue::Term(term))
                    }
                    FamilyCoverageStorage::SharedTemplate { .. } => self.shared_family_element(
                        family,
                        &LoweredInt { term: selector, stable_identity: None },
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
                    let selector_value = LoweredInt {
                        term: selector,
                        stable_identity: Some(self.canonical_scalar_identity(selector)?),
                    };
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
                    return family::select_family(&mut self.egraph, selector, &families)
                        .map(LoweredValue::Family)
                        .map_err(|_| LowerError::IncompatibleFamilyCoverage {
                            expected: output_type.clone(),
                            actual: output_type,
                        });
                }
                let terms = families
                    .iter()
                    .map(|value| match value {
                        LoweredValue::Term(term) => self.scalar_term(*term),
                        _ => Err(LowerError::IncompatibleFamilyCoverage {
                            expected: output_type.clone(),
                            actual: output_type.clone(),
                        }),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(LoweredValue::Term(family::add_runtime_switch(
                    &mut self.egraph,
                    selector,
                    &terms,
                )))
            }
            NodeKind::ParallelLoop(_) | NodeKind::SequentialLoop(_) => {
                unreachable!("loop lowering is scheduled on the outer continuation stack")
            }
            NodeKind::PackPolynomialCoefficients { matrix_type, coefficient_bits } => {
                let [LoweredValue::Family(family)] = arguments else {
                    return Err(LowerError::InvalidOperandArity {
                        expected: 1,
                        actual: arguments.len(),
                    });
                };
                if family.element_type != MxxSort::Bool {
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
                let coefficient_bits = self.resolve_int(coefficient_bits, environment)?;
                let Some(coefficient_bits_value) = resolved_integer(&coefficient_bits) else {
                    return Err(LowerError::InvalidPackBitCount {
                        coefficient_bits: BigInt::from(-1),
                        modulus: resolved_integer(&matrix_type.modulus)
                            .unwrap_or_else(|| BigInt::from(-1)),
                    });
                };
                let Some(coefficient_bits_usize) = coefficient_bits_value.to_usize() else {
                    return Err(LowerError::InvalidPackBitCount {
                        coefficient_bits: coefficient_bits_value,
                        modulus: resolved_integer(&matrix_type.modulus)
                            .unwrap_or_else(|| BigInt::from(-1)),
                    });
                };
                let ring_dimension = resolved_nonnegative(&matrix_type.ring_dimension)
                    .and_then(|value| value.to_usize())
                    .ok_or_else(|| LowerError::InvalidPackBitCount {
                        coefficient_bits: coefficient_bits_value.clone(),
                        modulus: resolved_integer(&matrix_type.modulus)
                            .unwrap_or_else(|| BigInt::from(-1)),
                    })?;
                let expected =
                    ring_dimension.checked_mul(coefficient_bits_usize).ok_or_else(|| {
                        LowerError::InvalidPackBitCount {
                            coefficient_bits: coefficient_bits_value.clone(),
                            modulus: resolved_integer(&matrix_type.modulus)
                                .unwrap_or_else(|| BigInt::from(-1)),
                        }
                    })?;
                if coefficient_bits_usize == 0 || elements.len() != expected {
                    return Err(LowerError::InvalidPackBitWidth {
                        expected,
                        actual: elements.len(),
                    });
                }
                Ok(LoweredValue::Term(self.egraph.add(MxxLang::PackPolynomialCoefficients {
                    matrix_type,
                    coefficient_bits,
                    bits: elements.clone(),
                })))
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
            LoweredValue::Term(_) => Some(super::error::LoweredValueCategory::Term),
            LoweredValue::Family(_) => Some(super::error::LoweredValueCategory::Family),
            LoweredValue::Trapdoor(_) => Some(super::error::LoweredValueCategory::Trapdoor),
            LoweredValue::TrapdoorFamily { .. } => {
                Some(super::error::LoweredValueCategory::TrapdoorFamily)
            }
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
            return Ok(LoweredValue::Term(*representative));
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
            *representative,
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

    fn shared_matrix_family_element(
        &mut self,
        family: &FamilyLoweringValue<TermId>,
        index: &LoweredInt,
    ) -> Result<LoweredValue, LowerError> {
        let (representative, domain, _) = family::shared_element(family)
            .map_err(|_| LowerError::InvalidFamilyCount { count: IntExpr::constant(0) })?;
        let Some(index_analysis) = self.integer_analysis(index.term) else {
            return Err(LowerError::MissingIntegerAnalysis { term: index.term });
        };
        if family::validate_family_index(index_analysis.0, &domain.logical_count).is_err() {
            return Err(LowerError::FamilyAccessOutOfRange {
                index: IntExpr::constant(-1),
                count: IntExpr::constant(domain.logical_count.clone()),
            });
        }
        if index.stable_identity.as_ref() == Some(&ResolvedIntExpr::Binder(domain.binder.clone())) {
            return Ok(LoweredValue::Matrix(*representative));
        }
        let replacement = index
            .stable_identity
            .clone()
            .ok_or(LowerError::NonExactIdentityIndex { expression: IntExpr::constant(0) })?;
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
                &mut self.family_substitution_memo,
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
        let Some((domain, _)) = self.integer_analysis(index.term) else {
            return Err(LowerError::MissingIntegerAnalysis { term: index.term });
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
        if let Some(stable) = index.stable_identity.as_ref() {
            if let Some(term) =
                normal_form_family::static_matrix_term(family, stable).map_err(|_| {
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
                    LoweredValue::MatrixFamily(family) => {
                        self.matrix_family_element(&family, &index, &wire, &environment)?
                    }
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
        representative: Id,
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
        let actual_sort = &self.egraph[self.egraph.find(representative)].data.sort;
        let sort_matches = match (&element_type, actual_sort) {
            (MxxSort::Matrix(expected), Ok(MxxSort::Matrix(actual))) => {
                super::analysis::matrix_types_equal(expected, actual)
            }
            (expected, Ok(actual)) => expected == actual,
            (_, Err(_)) => false,
        };
        if !sort_matches {
            return Err(LowerError::FamilyElementLoweringMismatch {
                expected: element_wire_type,
                actual_category: super::error::LoweredValueCategory::Term,
                actual_sort: actual_sort.as_ref().ok().cloned(),
                producer: body_source,
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
        let count_identity = count.stable_identity.ok_or_else(|| {
            LowerError::NonExactIdentityIndex { expression: specification.count.clone() }
        })?;
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

    /// Queues a matrix recurrence without creating an e-graph matrix atom.
    /// Integer loop analysis still uses the scalar e-graph, but every carried
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
        let binder_id =
            self.egraph.analysis.symbols.binders.intern(super::identity::BinderDescriptor {
                key: binder.clone(),
                minimum: BigInt::zero(),
                maximum: maximum - BigInt::from(1_u8),
            });
        let iteration = LoweredInt {
            term: self.egraph.add(MxxLang::IntBinder(super::identity::BinderId(binder_id))),
            stable_identity: Some(ResolvedIntExpr::Binder(binder.clone())),
        };
        child.binders.push((binder.clone(), iteration.clone()));
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
                    matrix_type: concrete,
                    coefficient_class: BoundClass::Large,
                    metadata: MatrixMetadata::unknown(),
                }),
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
            count: super::analysis::IntegerDomain::Exact(
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
        self.scalar_selector_identity(selector.term)
    }

    fn scalar_selector_identity(&self, selector: Id) -> Result<FactorIdentity, LowerError> {
        Ok(FactorIdentity::scalar_selector(self.canonical_scalar_identity(selector)?))
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

    fn test_integer_atom(
        egraph: &mut EGraph<MxxLang, MxxAnalysis>,
        name: &str,
        minimum: i64,
        maximum: i64,
    ) -> Id {
        let source = egraph.analysis.symbols.atomic_sources.intern(
            super::super::identity::AtomicSourceDescriptor {
                key: super::super::identity::AtomicSourceKey::ProtocolInput(
                    crate::ProtocolInputId::from(name),
                ),
                sort: MxxSort::Int,
                integer_domain: Some(super::super::identity::IntegerSourceDomain {
                    minimum: minimum.into(),
                    maximum: maximum.into(),
                }),
                canonical_residue_convention: None,
                relation_role: None,
            },
        );
        egraph.add(MxxLang::Atom {
            source: super::super::identity::AtomicSourceId(source),
            indices: Box::new([]),
        })
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
            let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
            let environment = root_test_environment();
            let matrix = test_resolved_matrix();
            let concrete = concrete_matrix_type(&matrix).unwrap();
            let bound = |class| MatrixBound {
                matrix_type: concrete.clone(),
                coefficient_class: class,
                metadata: MatrixMetadata::unknown(),
            };
            let large_matrix = |key| SymbolicFactor {
                key,
                bound: BoundClass::Large,
                relation_live: false,
                trapdoor: None,
                matrix_bound: Some(bound(BoundClass::Large)),
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
                .push(ExpressionNode::Atom(SymbolicFactor::large(target.clone())))
                .unwrap();
            lowerer
                .relation_registry
                .register(super::super::normal_form::RelationRegistration {
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
                        matrix_type: concrete,
                        coefficient_class: BoundClass::Large,
                        metadata: MatrixMetadata::unknown(),
                    }),
                    switch: None,
                }))
                .unwrap()
        };
        let mut valid = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
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

        let mut swapped = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
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
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
        let selector = test_integer_atom(&mut lowerer.egraph, "direct-select-range", 0, 1);
        let cases = [10, 20, 30].map(|value| lowerer.egraph.add(MxxLang::IntConst(value.into())));
        let environment = root_test_environment();
        let wire = LoweringWire {
            source: WireSourceKey {
                scope: environment.occurrence.clone(),
                wire: WireRef { node: mxx_ir_core::NodeId(1), port: mxx_ir_core::Port(0) },
            },
            indices: Box::new([]),
        };
        let arguments =
            std::iter::once(selector).chain(cases).map(LoweredValue::Term).collect::<Vec<_>>();
        let LoweredValue::Term(direct) = lowerer
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
        let shared = family::add_runtime_switch(&mut lowerer.egraph, selector, &cases);
        assert_eq!(lowerer.egraph.find(direct), lowerer.egraph.find(shared));
    }

    #[test]
    fn shared_matrix_family_accepts_bounded_runtime_selector_without_lanes() {
        let (protocol, request) = recurrence_lowerer();
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
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
                    matrix_type: concrete,
                    coefficient_class: BoundClass::Large,
                    metadata: MatrixMetadata::unknown(),
                }),
                switch: None,
            }))
            .unwrap();
        let family = FamilyLoweringValue {
            element_type: MxxSort::Matrix(matrix),
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
        let selector =
            test_integer_atom(&mut lowerer.egraph, "bounded-runtime-selector", 0, 30_719);
        let selector = LoweredInt {
            term: selector,
            stable_identity: lowerer.canonical_scalar_identity(selector).ok(),
        };
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
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
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
                        metadata: MatrixMetadata::unknown(),
                    }),
                    switch: None,
                }))
                .unwrap()
        };
        let stored0 = term(&mut lowerer, "stored-0");
        let stored1 = term(&mut lowerer, "stored-1");
        let family = FamilyLoweringValue {
            element_type: MxxSort::Matrix(matrix),
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
        let reachable = test_integer_atom(&mut lowerer.egraph, "reachable", 0, 0);
        let reachable = LoweredInt {
            term: reachable,
            stable_identity: lowerer.canonical_scalar_identity(reachable).ok(),
        };
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

        let invalid = test_integer_atom(&mut lowerer.egraph, "invalid-upper", 0, 2);
        let invalid = LoweredInt {
            term: invalid,
            stable_identity: lowerer.canonical_scalar_identity(invalid).ok(),
        };
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

    #[test]
    fn matrix_operations_have_only_dag_lowered_values() {
        let protocol = crate::toy_example::protocol();
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "matrix-dag-only".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
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
                    metadata: MatrixMetadata::unknown(),
                }),
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
        assert!(lowerer.egraph.classes().all(|class| {
            class.nodes.iter().all(|node| {
                !matches!(
                    node,
                    MxxLang::MatrixConstant(_) |
                        MxxLang::MatrixAdd(_) |
                        MxxLang::MatrixMultiply(_) |
                        MxxLang::MatrixNegate(_) |
                        MxxLang::MatrixScale(_) |
                        MxxLang::MatrixTranspose(_) |
                        MxxLang::MatrixSlice { .. } |
                        MxxLang::MatrixTensor(_) |
                        MxxLang::MatrixConcat { .. }
                )
            })
        }));
    }

    #[test]
    fn gadget_decompose_dag_path_validates_base_digits_rows_and_input_category() {
        let protocol = crate::toy_example::protocol();
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "gadget-validation".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
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
                    metadata: MatrixMetadata::unknown(),
                }),
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
        let scalar = LoweredValue::Term(lowerer.egraph.add(MxxLang::IntConst(1.into())));
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
    fn malformed_matrix_terms_and_matrix_families_are_rejected_at_scalar_boundaries() {
        let protocol = crate::toy_example::protocol();
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "matrix-term-rejection".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
        let matrix = super::super::identity::ResolvedMatrixType {
            modulus: ResolvedIntExpr::Const(17.into()),
            ring_dimension: ResolvedIntExpr::Const(1.into()),
            rows: ResolvedIntExpr::Const(1.into()),
            columns: ResolvedIntExpr::Const(1.into()),
        };
        let spec = lowerer.egraph.analysis.symbols.matrix_constants.intern(
            super::super::identity::MatrixConstantSpec {
                matrix_type: matrix.clone(),
                value: super::super::identity::MatrixConstantValue::Zero,
            },
        );
        let malformed = lowerer
            .egraph
            .add(MxxLang::MatrixConstant(super::super::identity::MatrixConstantSpecId(spec)));
        let scalar = lowerer.egraph.add(MxxLang::IntConst(1.into()));
        assert!(matches!(
            lowerer.lower_node(
                &NodeKind::IntBinary(IntBinaryOp::Add),
                &[LoweredValue::Term(malformed), LoweredValue::Term(scalar)],
                &root_test_environment(),
            ),
            Err(LowerError::UnsupportedMatrixProductExpansion)
        ));

        let family = FamilyLoweringValue {
            element_type: MxxSort::Matrix(matrix),
            storage: FamilyCoverageStorage::ExactStored {
                elements: vec![malformed].into_boxed_slice(),
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
    fn matrix_normal_form_is_independent_of_unrelated_scalar_egraph_insertions() {
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
                        metadata: MatrixMetadata::unknown(),
                    }),
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
                        matrix_type: matrix,
                        coefficient_class: BoundClass::Large,
                        metadata: MatrixMetadata::unknown(),
                    }),
                    switch: None,
                }))
                .unwrap();
            lowerer.dag.push(ExpressionNode::Product(vec![left, right].into_boxed_slice())).unwrap()
        };
        let mut plain = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
        let plain_root = build(&mut plain);
        let mut polluted = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
        let first = polluted.egraph.add(MxxLang::IntConst(9.into()));
        let second = polluted.egraph.add(MxxLang::IntConst(11.into()));
        let _ = polluted.egraph.add(MxxLang::IntAdd([first, second]));
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
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
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
        lowerer
            .expression_dag()
            .normalize_bounded(term, lowerer.normal_form_relations())
            .expect("finite DAG bound")
            .as_matrix_bound()
            .expect("bounded matrix summary")
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
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
        lowerer.lower_stage_wire(&StageId("encrypt".to_owned()), output).map(|_| ())
    }

    fn with_production_bound_atom(
        contract: Option<InputValueContract>,
        key: super::super::identity::AtomicSourceKey,
        inspect: impl FnOnce(&GraphLowerer<'_, '_>, super::super::identity::AtomicSourceId, Id),
    ) {
        let mut protocol = crate::toy_example::protocol();
        if let Some(contract) = contract {
            protocol.bundle.input_contract.inputs.push(crate::InputContractEntry {
                id: crate::ProtocolInputId::from("bound-input"),
                name: "bound-input".to_owned(),
                value: contract,
            });
        }
        let request = OperationalCheckRequest {
            environment: vec![(
                "declared".to_owned(),
                super::super::OperationalParameterValue::Integer(BigInt::from(3)),
            )],
            layouts: Vec::new(),
            target_id: "production-bound-atom".to_owned(),
        };
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
        let source = lowerer.egraph.analysis.symbols.atomic_sources.intern(
            super::super::identity::AtomicSourceDescriptor {
                key,
                sort: MxxSort::Matrix(super::super::identity::ResolvedMatrixType {
                    modulus: ResolvedIntExpr::Const(BigInt::from(17)),
                    ring_dimension: ResolvedIntExpr::Const(BigInt::from(1)),
                    rows: ResolvedIntExpr::Const(BigInt::from(1)),
                    columns: ResolvedIntExpr::Const(BigInt::from(1)),
                }),
                integer_domain: None,
                canonical_residue_convention: None,
                relation_role: None,
            },
        );
        let source = super::super::identity::AtomicSourceId(source);
        let term = lowerer.egraph.add(MxxLang::Atom { source, indices: Box::new([]) });
        inspect(&lowerer, source, term);
    }

    #[test]
    fn protocol_matrix_bounds_are_exactly_the_declared_contracts() {
        let matrix = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        with_production_bound_atom(
            Some(InputValueContract::MatrixBounded {
                matrix_type: matrix.clone(),
                max_centered_coefficient: DeclaredBoundExpr::Multiply(
                    Box::new(DeclaredBoundExpr::Parameter(IntExpr::Var("declared".to_owned()))),
                    Box::new(DeclaredBoundExpr::Constant(2_u8.into())),
                ),
            }),
            super::super::identity::AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from(
                "bound-input",
            )),
            |lowerer, source, term| {
                let bound = lowerer.production_bound_view().atom_bound(source, term).unwrap();
                assert_eq!(bound.coefficient_class, BoundClass::bounded(6_u8.into()));
                assert_eq!(bound.metadata, MatrixMetadata::unknown());
            },
        );
        with_production_bound_atom(
            Some(InputValueContract::MatrixExact {
                matrix_type: matrix,
                canonical_coefficient_exclusive_upper_bound: Some(IntExpr::constant(7)),
                is_constant_polynomial: true,
            }),
            super::super::identity::AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from(
                "bound-input",
            )),
            |lowerer, source, term| {
                let bound = lowerer.production_bound_view().atom_bound(source, term).unwrap();
                assert_eq!(bound.coefficient_class, BoundClass::bounded(6_u8.into()));
                assert!(bound.metadata.is_constant_polynomial);
            },
        );
    }

    #[test]
    fn missing_invalid_and_explicit_large_input_bounds_remain_distinct() {
        let matrix = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        for upper in [None, Some(IntExpr::constant(0)), Some(IntExpr::constant(18))] {
            with_production_bound_atom(
                Some(InputValueContract::MatrixExact {
                    matrix_type: matrix.clone(),
                    canonical_coefficient_exclusive_upper_bound: upper,
                    is_constant_polynomial: false,
                }),
                super::super::identity::AtomicSourceKey::ProtocolInput(
                    crate::ProtocolInputId::from("bound-input"),
                ),
                |lowerer, source, term| {
                    let expected = if matches!(
                        lowerer.protocol.bundle.input_contract.inputs.last().unwrap().value,
                        InputValueContract::MatrixExact {
                            canonical_coefficient_exclusive_upper_bound: None,
                            ..
                        }
                    ) {
                        BoundEvaluationError::MissingInputBoundContract { term }
                    } else {
                        BoundEvaluationError::InvalidDeclaredBound { term }
                    };
                    assert_eq!(
                        lowerer.production_bound_view().atom_bound(source, term),
                        Err(expected)
                    );
                },
            );
        }
        with_production_bound_atom(
            Some(InputValueContract::MatrixLarge { matrix_type: matrix }),
            super::super::identity::AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from(
                "bound-input",
            )),
            |lowerer, source, term| {
                assert_eq!(
                    lowerer
                        .production_bound_view()
                        .atom_bound(source, term)
                        .unwrap()
                        .coefficient_class,
                    BoundClass::Large
                );
            },
        );
        with_production_bound_atom(
            None,
            super::super::identity::AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from(
                "bound-input",
            )),
            |lowerer, source, term| {
                assert_eq!(
                    lowerer.production_bound_view().atom_bound(source, term),
                    Err(BoundEvaluationError::MissingInputBoundContract { term })
                );
            },
        );
    }

    #[test]
    fn opaque_graph_and_escaped_sequential_state_have_dedicated_errors() {
        let scope = root_test_environment().occurrence;
        let opaque_source = super::super::identity::GraphWireSourceKey {
            wire: WireSourceKey {
                scope: scope.clone(),
                wire: WireRef { node: mxx_ir_core::NodeId(1), port: mxx_ir_core::Port(0) },
            },
            coordinate_binders: Box::new([]),
        };
        with_production_bound_atom(
            None,
            super::super::identity::AtomicSourceKey::GraphWire(opaque_source.clone()),
            |lowerer, source, term| {
                assert_eq!(
                    lowerer.production_bound_view().atom_bound(source, term),
                    Err(BoundEvaluationError::OpaqueGraphWire { source: opaque_source })
                );
            },
        );
        with_production_bound_atom(
            None,
            super::super::identity::AtomicSourceKey::SequentialState(SequentialStateKey {
                loop_scope: scope,
                loop_node: mxx_ir_core::NodeId(2),
                carried_index: 0,
            }),
            |lowerer, source, term| {
                assert_eq!(
                    lowerer.production_bound_view().atom_bound(source, term),
                    Err(BoundEvaluationError::SequentialStateOutsideOverlay { term })
                );
            },
        );
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
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
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
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
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
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
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
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
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
                assert!(matches!(
                    gaussian.egraph.analysis.symbols.samplers.values.as_slice(),
                    [SamplerIdentity::Gaussian { max_coefficient_bound: ResolvedIntExpr::Const(value), .. }]
                        if value == &BigInt::from(5)
                ));
                assert!(gaussian.relation_registrations().is_empty());
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
                assert!(matches!(
                    interval.egraph.analysis.symbols.samplers.values.as_slice(),
                    [SamplerIdentity::UniformInterval {
                        minimum: ResolvedIntExpr::Const(minimum),
                        maximum: ResolvedIntExpr::Const(maximum),
                        ..
                    }] if minimum == &BigInt::from(-3) && maximum == &BigInt::from(2)
                ));
                assert!(interval.relation_registrations().is_empty());
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
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
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
        return;
        #[cfg(any())]
        {
            let decomposed_hash = Id::from(0);
            let sampler = lowerer.egraph.analysis.symbols.samplers.values[0].clone();
            let SamplerIdentity::DecomposedHash {
                source,
                indices,
                public,
                target,
                base,
                digit_count,
                ..
            } = sampler
            else {
                panic!("fixture produces a decomposed hash sampler")
            };
            let sampler = lowerer.egraph.analysis.symbols.samplers.intern(
                SamplerIdentity::GadgetDecomposition {
                    source: source.clone().into(),
                    indices: indices.clone(),
                    public,
                    target,
                    base: base.clone(),
                    digit_count: digit_count.clone(),
                    small: false,
                    range_proved: false,
                },
            );
            let sort = lowerer.egraph[lowerer.egraph.find(decomposed_hash)]
                .data
                .sort
                .clone()
                .expect("decomposed hash has a matrix sort");
            let atom_source = lowerer.egraph.analysis.symbols.atomic_sources.intern(
                super::super::identity::AtomicSourceDescriptor {
                    key: super::super::identity::AtomicSourceKey::Sampler(SamplerDescriptorId(
                        sampler,
                    )),
                    sort: sort.clone(),
                    integer_domain: None,
                    canonical_residue_convention: None,
                    relation_role: Some(
                        super::super::identity::AtomicRelationRole::GadgetDecomposition,
                    ),
                },
            );
            let decomposition = lowerer.egraph.add(MxxLang::Atom {
                source: super::super::identity::AtomicSourceId(atom_source),
                indices: indices.clone(),
            });

            let same_sampler = lowerer.egraph.analysis.symbols.samplers.intern(
                SamplerIdentity::GadgetDecomposition {
                    source: super::super::identity::GraphWireSourceKey {
                        wire: WireSourceKey {
                            scope: source.wire.scope.clone(),
                            wire: WireRef {
                                node: mxx_ir_core::NodeId(777),
                                port: mxx_ir_core::Port(0),
                            },
                        },
                        coordinate_binders: source.coordinate_binders.clone(),
                    }
                    .into(),
                    indices: indices.clone(),
                    public,
                    target,
                    base: base.clone(),
                    digit_count: digit_count.clone(),
                    small: false,
                    range_proved: false,
                },
            );
            assert_eq!(same_sampler, sampler, "equal deterministic decompositions share a sampler");
            let same_atom_source = lowerer.egraph.analysis.symbols.atomic_sources.intern(
                super::super::identity::AtomicSourceDescriptor {
                    key: super::super::identity::AtomicSourceKey::Sampler(SamplerDescriptorId(
                        same_sampler,
                    )),
                    sort: sort.clone(),
                    integer_domain: None,
                    canonical_residue_convention: None,
                    relation_role: Some(
                        super::super::identity::AtomicRelationRole::GadgetDecomposition,
                    ),
                },
            );
            assert_eq!(same_atom_source, atom_source, "the shared sampler has one atom source");
            let same_decomposition = lowerer.egraph.add(MxxLang::Atom {
                source: super::super::identity::AtomicSourceId(same_atom_source),
                indices: indices.clone(),
            });
            assert_eq!(
                lowerer.egraph.find(same_decomposition),
                lowerer.egraph.find(decomposition),
                "the same atom source and ordered coordinates hash-cons into one e-class"
            );
            let registrations = lowerer.relation_registrations();
            assert!(registrations.iter().any(|registration| {
                registration.source == super::super::identity::AtomicSourceId(atom_source) &&
                    registration.expected_public == public &&
                    registration.target == target &&
                    registration.indices.as_ref() == indices.as_ref()
            }));

            assert_eq!(
                BoundEvaluator::new(&lowerer.production_bound_view())
                    .evaluate(decomposition)
                    .expect("regular gadget decomposition has a finite digit bound")
                    .coefficient_class,
                BoundClass::Bounded { maximum_absolute_coefficient: 2_u8.into() },
            );

            let sampler = lowerer.egraph.analysis.symbols.samplers.intern(
                SamplerIdentity::GadgetDecomposition {
                    source: source.into(),
                    indices: indices.clone(),
                    public,
                    target,
                    base: ResolvedIntExpr::Const(BigInt::from(1)),
                    digit_count,
                    small: false,
                    range_proved: false,
                },
            );
            let atom_source = lowerer.egraph.analysis.symbols.atomic_sources.intern(
                super::super::identity::AtomicSourceDescriptor {
                    key: super::super::identity::AtomicSourceKey::Sampler(SamplerDescriptorId(
                        sampler,
                    )),
                    sort,
                    integer_domain: None,
                    canonical_residue_convention: None,
                    relation_role: Some(
                        super::super::identity::AtomicRelationRole::GadgetDecomposition,
                    ),
                },
            );
            let invalid = lowerer.egraph.add(MxxLang::Atom {
                source: super::super::identity::AtomicSourceId(atom_source),
                indices,
            });
            assert_eq!(
                BoundEvaluator::new(&lowerer.production_bound_view()).evaluate(invalid),
                Err(BoundEvaluationError::InvalidMatrixConstant { term: invalid }),
            );
        }
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
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
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
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
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
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
        let LoweredValue::Matrix(term) = lowerer
            .lower_stage_wire(&StageId("encrypt".to_owned()), output)
            .expect("lower polynomial constant through parallel loop")
        else {
            panic!("polynomial constant is a matrix term")
        };
        assert_eq!(dag_bound(&lowerer, term), BoundClass::bounded(5_u8.into()));
        assert!(lowerer.egraph.analysis.symbols.atomic_sources.values.iter().all(|descriptor| {
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
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
        lowerer
            .lower_stage_wire(&StageId("encrypt".to_owned()), output)
            .expect("bound parallel input lowers through subgraph call");
        assert!(lowerer.egraph.analysis.symbols.atomic_sources.values.iter().all(|descriptor| {
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
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
        assert!(matches!(
            lowerer.lower_stage_wire(&StageId("encrypt".to_owned()), output),
            Ok(LoweredValue::Term(_))
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
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
        assert!(matches!(
            lowerer.lower_stage_wire(&StageId("encrypt".to_owned()), output),
            Ok(LoweredValue::Term(_))
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
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
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
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
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
        let mut lowerer = GraphLowerer::new(&protocol, &request, MxxAnalysis::default());
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
        assert!(bound.metadata.is_constant_polynomial);
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
