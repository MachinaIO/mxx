//! Typed e-graph facts for the operational-noise checker.
//!
//! Integer ranges live here rather than on lowering handles.  In particular,
//! [`ScalarProvenance::SelectorOnly`] is sticky when e-classes merge: a value
//! derived from a canonical matrix residue remains usable for lookup selection
//! but can never become an ordinary numeric/noise scalar through congruence.

use super::{
    error::{AnalysisError, ResourceLimitKind, ResourceObserved},
    identity::{
        AtomicRelationRole, AtomicSourceId, BinderKey, CanonicalResidueConvention, ResolvedIntExpr,
        ResolvedMatrixType, SymbolTables,
    },
    language::MxxLang,
};
use egg::{Analysis, DidMerge, EGraph, Language};
use mxx_ir_core::{IntExpr, ParamEnv, expr::euclidean_div_rem};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Signed, Zero};
use smallvec::SmallVec;
use std::{
    cell::RefCell,
    collections::BTreeMap,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

/// The complete sort carried by every checker e-class.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum MxxSort {
    Int,
    Bool,
    Real,
    Bytes(ResolvedIntExpr),
    TypedBlob { type_name: String, schema_hash: [u8; 32] },
    Matrix(ResolvedMatrixType),
}

impl MxxSort {
    pub const fn is_scalar(&self) -> bool {
        matches!(self, Self::Int | Self::Bool | Self::Real)
    }

    pub const fn permits_scalar_provenance(&self) -> bool {
        matches!(self, Self::Int | Self::Bool)
    }
}

/// A closed inclusive integer interval.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct IntegerInterval {
    pub minimum: BigInt,
    pub maximum: BigInt,
}

impl IntegerInterval {
    pub fn new(minimum: BigInt, maximum: BigInt) -> Option<Self> {
        (minimum <= maximum).then_some(Self { minimum, maximum })
    }

    pub fn exact(value: BigInt) -> Self {
        Self { minimum: value.clone(), maximum: value }
    }

    pub fn contains_zero(&self) -> bool {
        self.minimum <= BigInt::zero() && BigInt::zero() <= self.maximum
    }

    pub fn is_exact(&self) -> bool {
        self.minimum == self.maximum
    }

    pub fn add(&self, other: &Self) -> Self {
        Self { minimum: &self.minimum + &other.minimum, maximum: &self.maximum + &other.maximum }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self { minimum: &self.minimum - &other.maximum, maximum: &self.maximum - &other.minimum }
    }

    /// Uses exactly four endpoint products.  This is constant work, rather
    /// than an enumeration of the binders that produced either interval.
    pub fn mul(&self, other: &Self) -> Self {
        let products = [
            &self.minimum * &other.minimum,
            &self.minimum * &other.maximum,
            &self.maximum * &other.minimum,
            &self.maximum * &other.maximum,
        ];
        let minimum = products.iter().min().expect("four products").clone();
        let maximum = products.iter().max().expect("four products").clone();
        Self { minimum, maximum }
    }

    pub fn hull(&self, other: &Self) -> Self {
        Self {
            minimum: self.minimum.clone().min(other.minimum.clone()),
            maximum: self.maximum.clone().max(other.maximum.clone()),
        }
    }
}

/// The one owner for both compile-time affine ranges and runtime integer
/// ranges.  `Affine` never expands a Cartesian product of binders.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IntegerDomain {
    Exact(BigInt),
    Affine {
        constant: BigInt,
        coefficients: BTreeMap<BinderKey, BigInt>,
        binders: BTreeMap<BinderKey, IntegerInterval>,
    },
    IntervalOnly(IntegerInterval),
}

/// Domain-operation failures deliberately contain no guessed Graph IR
/// expression.  The lowerer owns that expression and maps this closed reason
/// to the corresponding typed `LowerError` at the use-site.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntegerDomainError {
    DivisionByZero,
    NonExactDivisor,
    InexactDivision,
    InvalidRoundDivisor,
    InvalidLog2CeilArgument,
    InvalidAffine,
}

impl IntegerDomain {
    pub fn exact(value: impl Into<BigInt>) -> Self {
        Self::Exact(value.into())
    }

    pub fn interval(&self) -> Result<IntegerInterval, IntegerDomainError> {
        match self {
            Self::Exact(value) => Ok(IntegerInterval::exact(value.clone())),
            Self::IntervalOnly(interval) => Ok(interval.clone()),
            Self::Affine { constant, coefficients, binders } => {
                let mut minimum = constant.clone();
                let mut maximum = constant.clone();
                for (binder, coefficient) in coefficients {
                    let interval = binders.get(binder).ok_or(IntegerDomainError::InvalidAffine)?;
                    if coefficient.is_negative() {
                        minimum += coefficient * &interval.maximum;
                        maximum += coefficient * &interval.minimum;
                    } else {
                        minimum += coefficient * &interval.minimum;
                        maximum += coefficient * &interval.maximum;
                    }
                }
                if binders.keys().any(|binder| !coefficients.contains_key(binder)) {
                    return Err(IntegerDomainError::InvalidAffine);
                }
                Ok(IntegerInterval { minimum, maximum })
            }
        }
    }

    pub fn hull(&self, other: &Self) -> Result<Self, IntegerDomainError> {
        if self == other {
            return Ok(self.clone());
        }
        Ok(Self::IntervalOnly(self.interval()?.hull(&other.interval()?)))
    }

    pub fn add(&self, other: &Self) -> Result<Self, IntegerDomainError> {
        self.combine_affine(other, false)
    }

    pub fn sub(&self, other: &Self) -> Result<Self, IntegerDomainError> {
        self.combine_affine(other, true)
    }

    fn combine_affine(&self, other: &Self, subtract: bool) -> Result<Self, IntegerDomainError> {
        let Some((left_constant, left_coefficients, left_binders)) = self.affine_parts() else {
            let left = self.interval()?;
            let right = other.interval()?;
            return Ok(Self::IntervalOnly(if subtract {
                left.sub(&right)
            } else {
                left.add(&right)
            }));
        };
        let Some((right_constant, right_coefficients, right_binders)) = other.affine_parts() else {
            let left = self.interval()?;
            let right = other.interval()?;
            return Ok(Self::IntervalOnly(if subtract {
                left.sub(&right)
            } else {
                left.add(&right)
            }));
        };
        let mut coefficients = left_coefficients;
        let mut binders = left_binders;
        for (binder, coefficient) in right_coefficients {
            let delta = if subtract { -coefficient } else { coefficient };
            let entry = coefficients.entry(binder.clone()).or_insert_with(BigInt::zero);
            *entry += delta;
            if entry.is_zero() {
                coefficients.remove(&binder);
            }
        }
        for (binder, interval) in right_binders {
            match binders.get(&binder) {
                Some(existing) if existing != &interval => {
                    return Err(IntegerDomainError::InvalidAffine)
                }
                Some(_) => {}
                None => {
                    binders.insert(binder, interval);
                }
            }
        }
        binders.retain(|binder, _| coefficients.contains_key(binder));
        Ok(Self::from_affine(
            if subtract { left_constant - right_constant } else { left_constant + right_constant },
            coefficients,
            binders,
        ))
    }

    pub fn mul(&self, other: &Self) -> Result<Self, IntegerDomainError> {
        match (self.exact_value(), other.exact_value()) {
            (Some(left), _) => other.scale(left),
            (_, Some(right)) => self.scale(right),
            _ => Ok(Self::IntervalOnly(self.interval()?.mul(&other.interval()?))),
        }
    }

    pub fn exact_div(&self, divisor: &Self) -> Result<Self, IntegerDomainError> {
        let divisor = divisor.exact_value().ok_or(IntegerDomainError::NonExactDivisor)?;
        if divisor.is_zero() {
            return Err(IntegerDomainError::DivisionByZero);
        }
        match self {
            Self::Exact(value) => exact_quotient(value, divisor).map(Self::Exact),
            Self::Affine { constant, coefficients, binders } => {
                let constant = exact_quotient(constant, divisor)?;
                let mut divided = BTreeMap::new();
                for (binder, coefficient) in coefficients {
                    divided.insert(binder.clone(), exact_quotient(coefficient, divisor)?);
                }
                Ok(Self::from_affine(constant, divided, binders.clone()))
            }
            Self::IntervalOnly(interval) if interval.is_exact() => {
                exact_quotient(&interval.minimum, divisor).map(Self::Exact)
            }
            Self::IntervalOnly(_) => Err(IntegerDomainError::InexactDivision),
        }
    }

    pub fn round_div(&self, divisor: &Self) -> Result<Self, IntegerDomainError> {
        let divisor = divisor.exact_value().ok_or(IntegerDomainError::NonExactDivisor)?;
        if *divisor <= BigInt::zero() {
            return Err(IntegerDomainError::InvalidRoundDivisor);
        }
        let interval = self.interval()?;
        Ok(Self::IntervalOnly(IntegerInterval {
            minimum: evaluate_round_div(&interval.minimum, divisor)?,
            maximum: evaluate_round_div(&interval.maximum, divisor)?,
        }))
    }

    pub fn log2_ceil(&self) -> Result<Self, IntegerDomainError> {
        let interval = self.interval()?;
        if interval.minimum < BigInt::one() {
            return Err(IntegerDomainError::InvalidLog2CeilArgument);
        }
        Ok(Self::IntervalOnly(IntegerInterval {
            minimum: log2_ceil(&interval.minimum),
            maximum: log2_ceil(&interval.maximum),
        }))
    }

    /// Runtime Euclidean division.  Unlike [`Self::exact_div`], this accepts a
    /// non-singleton dividend and uses the runtime's nonnegative remainder
    /// convention even for a negative divisor.
    pub fn euclidean_div(&self, divisor: &Self) -> Result<Self, IntegerDomainError> {
        let divisor = divisor.exact_value().ok_or(IntegerDomainError::NonExactDivisor)?;
        if divisor.is_zero() {
            return Err(IntegerDomainError::DivisionByZero);
        }
        let interval = self.interval()?;
        let (_, _) = euclidean_div_rem(&BigInt::zero(), divisor)
            .map_err(|_| IntegerDomainError::DivisionByZero)?;
        let (minimum, _) = euclidean_div_rem(&interval.minimum, divisor)
            .map_err(|_| IntegerDomainError::DivisionByZero)?;
        let (maximum, _) = euclidean_div_rem(&interval.maximum, divisor)
            .map_err(|_| IntegerDomainError::DivisionByZero)?;
        Ok(Self::IntervalOnly(IntegerInterval { minimum, maximum }))
    }

    pub fn euclidean_remainder(&self, divisor: &Self) -> Result<Self, IntegerDomainError> {
        let divisor = divisor.exact_value().ok_or(IntegerDomainError::NonExactDivisor)?;
        if divisor.is_zero() {
            return Err(IntegerDomainError::DivisionByZero);
        }
        if let Some(value) = self.exact_value() {
            let (_, remainder) = euclidean_div_rem(value, divisor)
                .map_err(|_| IntegerDomainError::DivisionByZero)?;
            return Ok(Self::Exact(remainder));
        }
        let upper = divisor.abs() - BigInt::one();
        Ok(Self::IntervalOnly(IntegerInterval { minimum: BigInt::zero(), maximum: upper }))
    }

    fn exact_value(&self) -> Option<&BigInt> {
        match self {
            Self::Exact(value) => Some(value),
            Self::IntervalOnly(interval) if interval.is_exact() => Some(&interval.minimum),
            Self::Affine { constant, coefficients, .. } if coefficients.is_empty() => {
                Some(constant)
            }
            _ => None,
        }
    }

    fn affine_parts(
        &self,
    ) -> Option<(BigInt, BTreeMap<BinderKey, BigInt>, BTreeMap<BinderKey, IntegerInterval>)> {
        match self {
            Self::Exact(value) => Some((value.clone(), BTreeMap::new(), BTreeMap::new())),
            Self::Affine { constant, coefficients, binders } => {
                Some((constant.clone(), coefficients.clone(), binders.clone()))
            }
            Self::IntervalOnly(_) => None,
        }
    }

    fn scale(&self, scalar: &BigInt) -> Result<Self, IntegerDomainError> {
        match self {
            Self::Exact(value) => Ok(Self::Exact(value * scalar)),
            Self::Affine { constant, coefficients, binders } => Ok(Self::from_affine(
                constant * scalar,
                coefficients
                    .iter()
                    .filter_map(|(binder, coefficient)| {
                        let product = coefficient * scalar;
                        (!product.is_zero()).then_some((binder.clone(), product))
                    })
                    .collect(),
                binders.clone(),
            )),
            Self::IntervalOnly(interval) => {
                let factor = IntegerInterval::exact(scalar.clone());
                Ok(Self::IntervalOnly(interval.mul(&factor)))
            }
        }
    }

    fn from_affine(
        constant: BigInt,
        coefficients: BTreeMap<BinderKey, BigInt>,
        mut binders: BTreeMap<BinderKey, IntegerInterval>,
    ) -> Self {
        let coefficients = coefficients
            .into_iter()
            .filter(|(_, coefficient)| !coefficient.is_zero())
            .collect::<BTreeMap<_, _>>();
        binders.retain(|binder, _| coefficients.contains_key(binder));
        if coefficients.is_empty() {
            Self::Exact(constant)
        } else {
            Self::Affine { constant, coefficients, binders }
        }
    }
}

fn exact_quotient(value: &BigInt, divisor: &BigInt) -> Result<BigInt, IntegerDomainError> {
    if divisor.is_zero() {
        return Err(IntegerDomainError::DivisionByZero);
    }
    let quotient = value / divisor;
    (value % divisor).is_zero().then_some(quotient).ok_or(IntegerDomainError::InexactDivision)
}

fn evaluate_round_div(value: &BigInt, divisor: &BigInt) -> Result<BigInt, IntegerDomainError> {
    IntExpr::RoundDiv(
        Box::new(IntExpr::constant(value.clone())),
        Box::new(IntExpr::constant(divisor.clone())),
    )
    .evaluate(&ParamEnv::default())
    .map_err(|_| IntegerDomainError::InvalidRoundDivisor)
}

fn log2_ceil(value: &BigInt) -> BigInt {
    let value = value.to_biguint().expect("positive input checked by caller");
    let floor = value.bits() - 1;
    let power = BigUint::one() << usize::try_from(floor).expect("bit count fits usize");
    BigInt::from(if value == power { floor } else { floor + 1 })
}

/// A scalar derived from a canonical matrix coefficient can be a selector but
/// never ordinary arithmetic/noise data.  Merge uses logical OR, so this tag
/// cannot be erased by an equality or constant-fold union.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarProvenance {
    Ordinary,
    SelectorOnly,
}

impl ScalarProvenance {
    pub fn merge(self, other: Self) -> Self {
        if matches!(self, Self::SelectorOnly) || matches!(other, Self::SelectorOnly) {
            Self::SelectorOnly
        } else {
            Self::Ordinary
        }
    }

    pub fn runtime_arithmetic(self, other: Self) -> Option<Self> {
        matches!((self, other), (Self::Ordinary, Self::Ordinary)).then_some(Self::Ordinary)
    }

    pub fn euclidean(self, divisor: Self) -> Option<Self> {
        matches!(divisor, Self::Ordinary).then_some(self)
    }
}

/// The direct sampler output identity that may support a checked relation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelationSource {
    pub source: AtomicSourceId,
    pub indices: Box<[egg::Id]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelationUnavailableReason {
    SmallDecompositionRangeNotProved,
}

/// A shallow, interned relation-provenance node.  The arena owns all nodes and
/// switch branch slots; e-class facts only retain this handle.  Consequently a
/// deep switch chain neither recursively clones nor recursively drops.
#[derive(Clone)]
pub struct RelationProvenance {
    arena: Rc<RefCell<RelationProvenanceArena>>,
    node: usize,
}

impl std::fmt::Debug for RelationProvenance {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("RelationProvenance").field("node", &self.node).finish()
    }
}

impl PartialEq for RelationProvenance {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && Rc::ptr_eq(&self.arena, &other.arena)
    }
}

impl Eq for RelationProvenance {}

#[derive(Debug)]
enum RelationProvenanceNode {
    Direct(RelationSource),
    Unavailable { source: RelationSource, reason: RelationUnavailableReason },
    Switch { originating_selector: egg::Id, branches: Box<[Box<[usize]>]> },
}

#[derive(Debug, Default)]
struct RelationProvenanceArena {
    nodes: Vec<RelationProvenanceNode>,
}

impl RelationProvenanceArena {
    fn intern(&mut self, node: RelationProvenanceNode) -> usize {
        let id = self.nodes.len();
        self.nodes.push(node);
        id
    }
}

/// Item yielded by [`visit_relation_provenance`] without exposing the arena's
/// storage or encouraging recursive provenance traversal in consumers.
#[allow(dead_code)] // Some checker phases deliberately need only relation leaves.
pub(crate) enum RelationProvenanceVisit<'a> {
    Direct(&'a RelationSource),
    Unavailable { source: &'a RelationSource, reason: RelationUnavailableReason },
    Switch { originating_selector: egg::Id, branch_count: usize },
}

/// Iterates provenance without cloning it or descending through the Rust call
/// stack.  Consumers receive relation leaves in depth-first order and can
/// ignore switch metadata when only relation candidates matter.
pub(crate) fn visit_relation_provenance(
    values: &[RelationProvenance],
    mut visit: impl FnMut(RelationProvenanceVisit<'_>),
) {
    for root in values {
        let arena = root.arena.borrow();
        let mut work = Box::new(ProvenanceWorkChunk::empty());
        work.values[0] = Some(root.node);
        work.len = 1;
        loop {
            if work.len == 0 {
                let Some(previous) = work.previous.take() else {
                    break;
                };
                work = previous;
                continue;
            }
            work.len -= 1;
            let node = work.values[work.len].take().expect("initialized work item");
            match &arena.nodes[node] {
                RelationProvenanceNode::Direct(source) => {
                    visit(RelationProvenanceVisit::Direct(source))
                }
                RelationProvenanceNode::Unavailable { source, reason } => {
                    visit(RelationProvenanceVisit::Unavailable { source, reason: *reason });
                }
                RelationProvenanceNode::Switch { originating_selector, branches } => {
                    visit(RelationProvenanceVisit::Switch {
                        originating_selector: *originating_selector,
                        branch_count: branches.len(),
                    });
                    for branch in branches.iter().rev() {
                        for node in branch.iter().rev() {
                            if work.len == PROVENANCE_WORK_CHUNK_SIZE {
                                work = Box::new(ProvenanceWorkChunk {
                                    values: [None; PROVENANCE_WORK_CHUNK_SIZE],
                                    len: 0,
                                    previous: Some(work),
                                });
                            }
                            work.values[work.len] = Some(*node);
                            work.len += 1;
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectExtractFact {
    pub canonical_upper: Option<BigUint>,
}

/// All facts made by the sole egg analysis owner.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnalysisData {
    pub sort: Result<MxxSort, AnalysisError>,
    pub integer_domain: Option<IntegerDomain>,
    pub scalar_provenance: Option<ScalarProvenance>,
    pub possible_false: bool,
    pub possible_true: bool,
    pub real_constant_bits: Option<u64>,
    pub canonical_coefficient_exclusive_upper: Option<BigUint>,
    pub canonical_residue_convention: Option<CanonicalResidueConvention>,
    pub direct_extract: Option<DirectExtractFact>,
    pub(crate) relation_provenance: SmallVec<[RelationProvenance; 1]>,
}

impl AnalysisData {
    pub fn scalar(
        sort: MxxSort,
        domain: Option<IntegerDomain>,
        provenance: ScalarProvenance,
    ) -> Self {
        debug_assert!(sort.permits_scalar_provenance());
        Self {
            sort: Ok(sort),
            integer_domain: domain,
            scalar_provenance: Some(provenance),
            possible_false: false,
            possible_true: false,
            real_constant_bits: None,
            canonical_coefficient_exclusive_upper: None,
            canonical_residue_convention: None,
            direct_extract: None,
            relation_provenance: SmallVec::new(),
        }
    }

    pub fn matrix(sort: ResolvedMatrixType, canonical_upper: Option<BigUint>) -> Self {
        Self::matrix_with_convention(sort, canonical_upper, None)
    }

    pub fn matrix_with_convention(
        sort: ResolvedMatrixType,
        canonical_upper: Option<BigUint>,
        canonical_residue_convention: Option<CanonicalResidueConvention>,
    ) -> Self {
        Self {
            sort: Ok(MxxSort::Matrix(sort)),
            integer_domain: None,
            scalar_provenance: None,
            possible_false: false,
            possible_true: false,
            real_constant_bits: None,
            canonical_coefficient_exclusive_upper: canonical_upper,
            canonical_residue_convention,
            direct_extract: None,
            relation_provenance: SmallVec::new(),
        }
    }

    pub fn boolean(
        possible_false: bool,
        possible_true: bool,
        provenance: ScalarProvenance,
    ) -> Self {
        let mut data = Self::scalar(MxxSort::Bool, None, provenance);
        data.possible_false = possible_false;
        data.possible_true = possible_true;
        data
    }

    /// Applies the specified sticky e-class merge.  It does not canonicalize
    /// relation IDs: egg canonicalization belongs to the post-rebuild relation
    /// phase, so merge order cannot affect the raw stored provenance.
    pub(crate) fn merge_from(
        &mut self,
        from: Self,
        relation_limit: usize,
        budget: &mut ResourceBudget,
    ) -> (bool, bool) {
        // Complete the relation-limit check before changing any scalar facts
        // or moving a provenance collection.  The analysis owner separately
        // reserves the exact owned-element delta before calling this method.
        let Some((_, relation_overflow)) =
            merge_provenance_delta(self, &from, relation_limit, budget)
        else {
            return (false, true);
        };
        if relation_overflow {
            return (false, true);
        }

        // `egg::DidMerge` permits conservative change reporting.  Comparing
        // the inputs avoids cloning their potentially deep provenance trees.
        let inputs_differ = *self != from;
        self.sort = merge_sort(self.sort.clone(), from.sort);
        let missing_integer_domain = matches!(&self.sort, Ok(MxxSort::Int)) &&
            (self.integer_domain.is_none() || from.integer_domain.is_none());
        self.integer_domain = match (&self.integer_domain, &from.integer_domain) {
            (Some(left), Some(right)) => left.hull(right).ok(),
            (None, None) => None,
            _ => None,
        };
        let permits_scalar_provenance =
            self.sort.as_ref().is_ok_and(MxxSort::permits_scalar_provenance);
        let missing_scalar_provenance = permits_scalar_provenance &&
            (self.scalar_provenance.is_none() || from.scalar_provenance.is_none());
        self.scalar_provenance = match (self.scalar_provenance, from.scalar_provenance) {
            (Some(left), Some(right)) => Some(left.merge(right)),
            _ => None,
        };
        self.possible_false |= from.possible_false;
        self.possible_true |= from.possible_true;
        let missing_real_constant = matches!(&self.sort, Ok(MxxSort::Real)) &&
            (self.real_constant_bits.is_none() || from.real_constant_bits.is_none());
        self.real_constant_bits = match (self.real_constant_bits, from.real_constant_bits) {
            (Some(left), Some(right)) if left == right => Some(left),
            (None, None) => None,
            _ => None,
        };
        self.canonical_coefficient_exclusive_upper = match (
            self.canonical_coefficient_exclusive_upper.take(),
            from.canonical_coefficient_exclusive_upper,
        ) {
            (Some(left), Some(right)) => Some(left.min(right)),
            (left @ Some(_), None) | (None, left @ Some(_)) => left,
            (None, None) => None,
        };
        self.canonical_residue_convention =
            match (self.canonical_residue_convention, from.canonical_residue_convention) {
                (Some(left), Some(right)) if left == right => Some(left),
                (left @ Some(_), None) | (None, left @ Some(_)) => left,
                (Some(_), Some(_)) => {
                    make_sort_conflict_sticky(&mut self.sort);
                    None
                }
                (None, None) => None,
            };
        self.direct_extract = match (self.direct_extract.take(), from.direct_extract) {
            (Some(left), Some(right)) if left == right => Some(left),
            _ => None,
        };
        for provenance in from.relation_provenance {
            if !self.relation_provenance.contains(&provenance) {
                debug_assert!(self.relation_provenance.len() < relation_limit);
                self.relation_provenance.push(provenance);
            }
        }
        if missing_integer_domain || missing_scalar_provenance || missing_real_constant {
            make_sort_conflict_sticky(&mut self.sort);
        }
        (inputs_differ, false)
    }
}

fn make_sort_conflict_sticky(sort: &mut Result<MxxSort, AnalysisError>) {
    if let Ok(value) = sort {
        let value = value.clone();
        *sort = Err(AnalysisError::EClassSortConflict { expected: value.clone(), actual: value });
    }
}

fn merge_sort(
    left: Result<MxxSort, AnalysisError>,
    right: Result<MxxSort, AnalysisError>,
) -> Result<MxxSort, AnalysisError> {
    match (left, right) {
        (Ok(left), Ok(right)) if left == right => Ok(left),
        // The current public error registry carries `WireType` values, while
        // analysis works over owner-resolved MxxSort values.  Lowering performs
        // that lossless boundary conversion; until then preserve the first
        // existing typed error rather than fabricate a WireType.
        (Err(error), _) | (_, Err(error)) => Err(error),
        (Ok(expected), Ok(actual)) => Err(AnalysisError::EClassSortConflict { expected, actual }),
    }
}

/// Sole owner of analysis facts and job-local symbols for one e-graph.
#[derive(Clone, Debug)]
pub struct MxxAnalysis {
    pub symbols: SymbolTables,
    pub resource_failure: Option<ResourceObserved>,
    pub resource_failure_kind: Option<ResourceLimitKind>,
    relation_sources_per_eclass: usize,
    switch_case_limit: usize,
    pub(crate) resource_budget: ResourceBudget,
    provenance_arena: Rc<RefCell<RelationProvenanceArena>>,
}

#[derive(Clone, Debug)]
pub(crate) struct ResourceBudget {
    pub(crate) total_owned_element_limit: usize,
    total_owned_elements: Arc<AtomicUsize>,
    started: Instant,
    deadline: Instant,
    total_time_limit: Duration,
    temporary_elements: usize,
    temporary_failure: Option<(ResourceLimitKind, ResourceObserved)>,
}

impl ResourceBudget {
    fn standalone_for_tests() -> Self {
        let started = Instant::now();
        Self {
            total_owned_element_limit: 2_000_000,
            total_owned_elements: Arc::new(AtomicUsize::new(0)),
            started,
            deadline: started + Duration::from_secs(120),
            total_time_limit: Duration::from_secs(120),
            temporary_elements: 0,
            temporary_failure: None,
        }
    }

    #[cfg(test)]
    fn production() -> Self {
        Self::standalone_for_tests()
    }

    #[cfg(test)]
    fn owned(&self) -> usize {
        self.total_owned_elements.load(Ordering::Relaxed)
    }

    pub(crate) fn from_shared(
        total_owned_element_limit: usize,
        total_owned_elements: Arc<AtomicUsize>,
        started: Instant,
        deadline: Instant,
        total_time_limit: Duration,
    ) -> Self {
        Self {
            total_owned_element_limit,
            total_owned_elements,
            started,
            deadline,
            total_time_limit,
            temporary_elements: 0,
            temporary_failure: None,
        }
    }

    pub(crate) fn reserve(
        &mut self,
        additional: usize,
    ) -> Result<(), (ResourceLimitKind, ResourceObserved)> {
        let now = Instant::now();
        if now >= self.deadline {
            return Err((
                ResourceLimitKind::TotalTime,
                ResourceObserved::Duration {
                    limit: self.total_time_limit,
                    observed: now.duration_since(self.started),
                },
            ));
        }
        let mut current = self.total_owned_elements.load(Ordering::Relaxed);
        loop {
            let observed = current.checked_add(additional).unwrap_or(usize::MAX);
            if observed > self.total_owned_element_limit {
                return Err((
                    ResourceLimitKind::TotalOwnedElements,
                    ResourceObserved::Counter {
                        limit: self.total_owned_element_limit.min(u64::MAX as usize) as u64,
                        observed: observed.min(u64::MAX as usize) as u64,
                    },
                ));
            }
            match self.total_owned_elements.compare_exchange_weak(
                current,
                observed,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(()),
                Err(next) => current = next,
            }
        }
    }

    fn reserve_temporary(&mut self, additional: usize) -> Result<(), ResourceObserved> {
        if let Err((kind, observed)) = self.reserve(additional) {
            self.temporary_failure = Some((kind, observed.clone()));
            return Err(observed);
        }
        self.temporary_elements =
            self.temporary_elements.checked_add(additional).unwrap_or(usize::MAX);
        Ok(())
    }

    fn release_temporary(&mut self, released: usize) {
        self.temporary_elements = self
            .temporary_elements
            .checked_sub(released)
            .expect("temporary reservations must be released exactly once");
    }
}

impl Default for MxxAnalysis {
    fn default() -> Self {
        Self::new(SymbolTables::default())
    }
}

impl MxxAnalysis {
    pub fn new(symbols: SymbolTables) -> Self {
        Self {
            symbols,
            resource_failure: None,
            resource_failure_kind: None,
            relation_sources_per_eclass: 64,
            switch_case_limit: 65_536,
            resource_budget: ResourceBudget::standalone_for_tests(),
            provenance_arena: Rc::default(),
        }
    }

    pub(crate) fn with_resource_budget(
        symbols: SymbolTables,
        relation_sources_per_eclass: usize,
        switch_case_limit: usize,
        resource_budget: ResourceBudget,
    ) -> Self {
        Self {
            symbols,
            resource_failure: None,
            resource_failure_kind: None,
            relation_sources_per_eclass,
            switch_case_limit,
            resource_budget,
            provenance_arena: Rc::default(),
        }
    }

    #[cfg(test)]
    fn with_relation_limit(symbols: SymbolTables, relation_sources_per_eclass: usize) -> Self {
        Self {
            symbols,
            resource_failure: None,
            resource_failure_kind: None,
            relation_sources_per_eclass,
            switch_case_limit: 65_536,
            resource_budget: ResourceBudget::standalone_for_tests(),
            provenance_arena: Rc::default(),
        }
    }

    fn reserve_owned_elements(&mut self, additional: Option<usize>) -> bool {
        let reservation = additional
            .ok_or_else(|| {
                (
                    ResourceLimitKind::TotalOwnedElements,
                    ResourceObserved::Counter {
                        limit: self.resource_budget.total_owned_element_limit as u64,
                        observed: u64::MAX,
                    },
                )
            })
            .and_then(|additional| self.resource_budget.reserve(additional));
        if let Err((kind, observed)) = &reservation &&
            self.resource_failure.is_none()
        {
            self.resource_failure = Some(observed.clone());
            self.resource_failure_kind = Some(kind.clone());
        }
        reservation.is_ok()
    }

    fn fail_resource(&mut self, kind: ResourceLimitKind, observed: ResourceObserved) {
        if self.resource_failure.is_none() {
            self.resource_failure = Some(observed);
            self.resource_failure_kind = Some(kind);
        }
    }

    /// Makes the selector domain for `ExtractCoefficient` from an authoritative
    /// upper bound.  `None` is the validated full-modulus fallback; a malformed
    /// authoritative bound is never silently widened.
    pub fn extract_coefficient_domain(
        matrix: &ResolvedMatrixType,
        modulus: &BigUint,
        authoritative_upper: Option<&BigUint>,
        convention: CanonicalResidueConvention,
    ) -> Result<IntegerDomain, AnalysisError> {
        if modulus.is_zero() {
            return Err(AnalysisError::UnknownCanonicalResidueRange {
                matrix: MxxSort::Matrix(matrix.clone()),
            });
        }
        if let Some(upper) = authoritative_upper {
            if convention != CanonicalResidueConvention::Nonnegative ||
                upper.is_zero() ||
                upper > modulus
            {
                return Err(AnalysisError::UnknownCanonicalResidueRange {
                    matrix: MxxSort::Matrix(matrix.clone()),
                });
            }
            return Ok(IntegerDomain::IntervalOnly(IntegerInterval {
                minimum: BigInt::zero(),
                maximum: BigInt::from(upper - BigUint::one()),
            }));
        }
        let (minimum, maximum) = match convention {
            CanonicalResidueConvention::Nonnegative => {
                (BigInt::zero(), BigInt::from(modulus - BigUint::one()))
            }
            CanonicalResidueConvention::Centered => {
                let floor = modulus / BigUint::from(2_u8);
                let ceil = (modulus + BigUint::one()) / BigUint::from(2_u8);
                (-BigInt::from(floor), BigInt::from(ceil) - BigInt::one())
            }
        };
        if minimum > maximum {
            return Err(AnalysisError::UnknownCanonicalResidueRange {
                matrix: MxxSort::Matrix(matrix.clone()),
            });
        }
        Ok(IntegerDomain::IntervalOnly(IntegerInterval { minimum, maximum }))
    }
}

impl Analysis<MxxLang> for MxxAnalysis {
    type Data = AnalysisData;

    fn make(egraph: &mut EGraph<MxxLang, Self>, enode: &MxxLang) -> Self::Data {
        if let MxxLang::Switch(children) = enode {
            let case_count = children.len().saturating_sub(1);
            if case_count > egraph.analysis.switch_case_limit {
                egraph.analysis.fail_resource(
                    ResourceLimitKind::SwitchCases,
                    ResourceObserved::Counter {
                        limit: egraph.analysis.switch_case_limit as u64,
                        observed: case_count.min(u64::MAX as usize) as u64,
                    },
                );
                return invalid_analysis_data();
            }
        }
        // Reserve structural children and the complete prospective switch
        // provenance before its owned branch collections are constructed.
        let prospective_provenance = prospective_provenance_owned_elements(egraph, enode);
        let prospective_total = prospective_provenance
            .and_then(|provenance| enode.children().len().checked_add(provenance));
        if !egraph.analysis.reserve_owned_elements(prospective_total) {
            return invalid_analysis_data();
        }
        let mut data = make_analysis_data(egraph, enode);
        // Only the syntactic ExtractCoefficient node is a direct extraction.
        // Every operator or wrapper, including an all-equal Switch, clears it.
        if !matches!(enode, MxxLang::ExtractCoefficient(_)) {
            data.direct_extract = None;
        }
        data
    }

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        let preflight = merge_provenance_delta(
            to,
            &from,
            self.relation_sources_per_eclass,
            &mut self.resource_budget,
        );
        if let Some((kind, observed)) = self.resource_budget.temporary_failure.take() {
            self.fail_resource(kind, observed);
        }
        let Some((additional, relation_overflow)) = preflight else {
            return DidMerge(false, false);
        };
        if relation_overflow {
            if self.resource_failure.is_none() {
                self.resource_failure = Some(ResourceObserved::Counter {
                    limit: self.relation_sources_per_eclass as u64,
                    observed: self.relation_sources_per_eclass.checked_add(1).unwrap_or(usize::MAX)
                        as u64,
                });
                self.resource_failure_kind = Some(ResourceLimitKind::RelationSourcesPerEClass);
            }
            return DidMerge(false, false);
        }
        if !self.reserve_owned_elements(Some(additional)) {
            return DidMerge(false, false);
        }
        let (changed, overflow) =
            to.merge_from(from, self.relation_sources_per_eclass, &mut self.resource_budget);
        debug_assert!(!overflow, "relation provenance preflight must prevent overflow");
        // The second flag may conservatively report that `from` changed.  It
        // shares the clone-free input comparison from `merge_from`.
        DidMerge(changed, changed)
    }
}

fn make_analysis_data(egraph: &EGraph<MxxLang, MxxAnalysis>, enode: &MxxLang) -> AnalysisData {
    use MxxLang::*;
    match enode {
        Atom { source, indices } => {
            let Some(descriptor) = egraph.analysis.symbols.atomic_sources.get(source.0) else {
                return invalid_analysis_data();
            };
            if !graph_wire_coordinates_are_authoritative(egraph, descriptor, indices) {
                return AnalysisData {
                    sort: Err(AnalysisError::InvalidSamplerDescriptor {
                        source: descriptor.key.clone(),
                    }),
                    ..invalid_analysis_data()
                };
            }
            if descriptor.relation_role.is_some() && !matches!(descriptor.sort, MxxSort::Matrix(_))
            {
                return AnalysisData {
                    sort: Err(AnalysisError::InvalidSamplerDescriptor {
                        source: descriptor.key.clone(),
                    }),
                    ..invalid_analysis_data()
                };
            }
            let mut data = match &descriptor.sort {
                MxxSort::Int => {
                    let Some(domain) = &descriptor.integer_domain else {
                        return invalid_analysis_data();
                    };
                    let Some(interval) =
                        IntegerInterval::new(domain.minimum.clone(), domain.maximum.clone())
                    else {
                        return invalid_analysis_data();
                    };
                    AnalysisData::scalar(
                        MxxSort::Int,
                        Some(if interval.is_exact() {
                            IntegerDomain::Exact(interval.minimum)
                        } else {
                            IntegerDomain::IntervalOnly(interval)
                        }),
                        ScalarProvenance::Ordinary,
                    )
                }
                MxxSort::Matrix(matrix) => AnalysisData::matrix_with_convention(
                    matrix.clone(),
                    None,
                    descriptor.canonical_residue_convention,
                ),
                other => data_for_sort(other.clone()),
            };
            if let Some(role) = descriptor.relation_role {
                let relation_source = RelationSource { source: *source, indices: indices.clone() };
                let node = match role {
                    AtomicRelationRole::Preimage |
                    AtomicRelationRole::GadgetDecomposition |
                    AtomicRelationRole::DecomposedHash => {
                        RelationProvenanceNode::Direct(relation_source)
                    }
                    AtomicRelationRole::SmallGadgetDecomposition { range_proved: true } |
                    AtomicRelationRole::SmallDecomposedHash { range_proved: true } => {
                        RelationProvenanceNode::Direct(relation_source)
                    }
                    AtomicRelationRole::SmallGadgetDecomposition { range_proved: false } |
                    AtomicRelationRole::SmallDecomposedHash { range_proved: false } => {
                        RelationProvenanceNode::Unavailable {
                            source: relation_source,
                            reason: RelationUnavailableReason::SmallDecompositionRangeNotProved,
                        }
                    }
                };
                let node = egraph.analysis.provenance_arena.borrow_mut().intern(node);
                data.relation_provenance.push(RelationProvenance {
                    arena: Rc::clone(&egraph.analysis.provenance_arena),
                    node,
                });
            }
            data
        }
        IntConst(value) => AnalysisData::scalar(
            MxxSort::Int,
            Some(IntegerDomain::Exact(value.clone())),
            ScalarProvenance::Ordinary,
        ),
        IntParameter(name) => egraph
            .analysis
            .symbols
            .integer_parameters
            .get(name)
            .map(|value| {
                AnalysisData::scalar(
                    MxxSort::Int,
                    Some(IntegerDomain::Exact(value.clone())),
                    ScalarProvenance::Ordinary,
                )
            })
            .unwrap_or_else(invalid_analysis_data),
        IntBinder(binder_id) => {
            let Some(descriptor) = egraph.analysis.symbols.binders.get(binder_id.0).cloned() else {
                return invalid_analysis_data();
            };
            let Some(interval) = IntegerInterval::new(descriptor.minimum, descriptor.maximum)
            else {
                return invalid_analysis_data();
            };
            AnalysisData::scalar(
                MxxSort::Int,
                Some(IntegerDomain::Affine {
                    constant: BigInt::zero(),
                    coefficients: BTreeMap::from([(descriptor.key.clone(), BigInt::one())]),
                    binders: BTreeMap::from([(descriptor.key, interval)]),
                }),
                ScalarProvenance::Ordinary,
            )
        }
        IntAdd(children) => integer_binary(egraph, children, IntegerDomain::add, false),
        IntSub(children) => integer_binary(egraph, children, IntegerDomain::sub, false),
        IntMul(children) => integer_binary(egraph, children, IntegerDomain::mul, false),
        IntExactDiv(children) => integer_binary(egraph, children, IntegerDomain::exact_div, false),
        IntEuclideanDiv(children) => {
            integer_binary(egraph, children, IntegerDomain::euclidean_div, true)
        }
        IntEuclideanRemainder(children) => {
            integer_binary(egraph, children, IntegerDomain::euclidean_remainder, true)
        }
        IntRoundDiv(children) => integer_binary(egraph, children, IntegerDomain::round_div, false),
        IntLog2Ceil(children) => integer_unary(egraph, children[0], IntegerDomain::log2_ceil),
        BoolConst(value) => AnalysisData::boolean(!value, *value, ScalarProvenance::Ordinary),
        IntEqual(children) => integer_compare(egraph, children, |left, right| {
            let left = left.interval().ok()?;
            let right = right.interval().ok()?;
            Some(if left.maximum < right.minimum || right.maximum < left.minimum {
                (true, false)
            } else if left.is_exact() && right.is_exact() && left.minimum == right.minimum {
                (false, true)
            } else {
                (true, true)
            })
        }),
        IntLess(children) => integer_compare(egraph, children, |left, right| {
            let left = left.interval().ok()?;
            let right = right.interval().ok()?;
            Some(if left.maximum < right.minimum {
                (false, true)
            } else if left.minimum >= right.maximum {
                (true, false)
            } else {
                (true, true)
            })
        }),
        IntLessEqual(children) => integer_compare(egraph, children, |left, right| {
            let left = left.interval().ok()?;
            let right = right.interval().ok()?;
            Some(if left.maximum <= right.minimum {
                (false, true)
            } else if left.minimum > right.maximum {
                (true, false)
            } else {
                (true, true)
            })
        }),
        BitExtract { input, .. } => {
            let Some(source) = int_data(egraph, input[0]) else {
                return invalid_analysis_data();
            };
            AnalysisData::boolean(true, true, source.scalar_provenance.unwrap())
        }
        BoolToInt(children) => {
            let source = &egraph[children[0]].data;
            if source.sort != Ok(MxxSort::Bool) ||
                source.scalar_provenance != Some(ScalarProvenance::Ordinary)
            {
                return invalid_analysis_data();
            }
            let domain = match (source.possible_false, source.possible_true) {
                (true, false) => IntegerDomain::Exact(BigInt::zero()),
                (false, true) => IntegerDomain::Exact(BigInt::one()),
                (true, true) => IntegerDomain::IntervalOnly(IntegerInterval {
                    minimum: BigInt::zero(),
                    maximum: BigInt::one(),
                }),
                (false, false) => return invalid_analysis_data(),
            };
            AnalysisData::scalar(MxxSort::Int, Some(domain), ScalarProvenance::Ordinary)
        }
        RealConst(bits) => real_data(f64::from_bits(*bits)),
        IntToReal(children) => {
            let Some(source) = int_data(egraph, children[0]) else {
                return invalid_analysis_data();
            };
            if source.scalar_provenance != Some(ScalarProvenance::Ordinary) {
                return invalid_analysis_data();
            }
            let Some(value) = source.integer_domain.as_ref().and_then(IntegerDomain::exact_value)
            else {
                return invalid_analysis_data();
            };
            use num_traits::ToPrimitive;
            value.to_f64().map(real_data).unwrap_or_else(invalid_analysis_data)
        }
        RealAdd(children) => real_binary(egraph, children, |left, right| left + right),
        RealSub(children) => real_binary(egraph, children, |left, right| left - right),
        RealMul(children) => real_binary(egraph, children, |left, right| left * right),
        RealDiv(children) => real_binary(egraph, children, |left, right| left / right),
        RealSqrt(children) => real_unary(egraph, children[0], f64::sqrt),
        MatrixConstant(spec_id) => egraph
            .analysis
            .symbols
            .matrix_constants
            .get(spec_id.0)
            .map(|spec| {
                let upper = spec.canonical_coefficient_exclusive_upper();
                AnalysisData::matrix_with_convention(
                    spec.matrix_type.clone(),
                    upper,
                    Some(CanonicalResidueConvention::Nonnegative),
                )
            })
            .unwrap_or_else(invalid_analysis_data),
        MatrixCanonicalRangeContract { upper, input } => {
            let mut data = egraph[input[0]].data.clone();
            if !matches!(data.sort, Ok(MxxSort::Matrix(_))) ||
                upper.is_zero() ||
                data.canonical_residue_convention !=
                    Some(CanonicalResidueConvention::Nonnegative)
            {
                return invalid_analysis_data();
            }
            data.canonical_coefficient_exclusive_upper = Some(
                data.canonical_coefficient_exclusive_upper
                    .map_or_else(|| upper.clone(), |existing| existing.min(upper.clone())),
            );
            data
        }
        HashPlain { query, arguments } => {
            let Some(spec) = egraph.analysis.symbols.hash_queries.get(query.0) else {
                return invalid_analysis_data();
            };
            if arguments
                .first()
                .is_none_or(|key| !matches!(egraph[*key].data.sort, Ok(MxxSort::Bytes(_))))
            {
                return invalid_analysis_data();
            }
            AnalysisData::matrix(spec.matrix_type.clone(), None)
        }
        MatrixAdd(children) => matrix_add(egraph, children),
        MatrixMultiply(children) => matrix_multiply(egraph, children),
        MatrixNegate(children) => matrix_passthrough(egraph, children[0], false),
        MatrixScale(children) => {
            if int_data(egraph, children[0])
                .is_none_or(|value| value.scalar_provenance != Some(ScalarProvenance::Ordinary))
            {
                invalid_analysis_data()
            } else {
                matrix_passthrough(egraph, children[1], false)
            }
        }
        MatrixTranspose(children) => {
            let Some(matrix) = matrix_sort(egraph, children[0]) else {
                return invalid_analysis_data();
            };
            AnalysisData::matrix_with_convention(
                ResolvedMatrixType {
                    modulus: matrix.modulus.clone(),
                    ring_dimension: matrix.ring_dimension.clone(),
                    rows: matrix.columns.clone(),
                    columns: matrix.rows.clone(),
                },
                egraph[children[0]].data.canonical_coefficient_exclusive_upper.clone(),
                egraph[children[0]].data.canonical_residue_convention,
            )
        }
        MatrixSlice { spec, input } => {
            let Some(matrix) = matrix_sort(egraph, input[0]) else {
                return invalid_analysis_data();
            };
            let Some(slice) = egraph.analysis.symbols.slices.get(spec.0) else {
                return invalid_analysis_data();
            };
            AnalysisData::matrix_with_convention(
                ResolvedMatrixType {
                    modulus: matrix.modulus.clone(),
                    ring_dimension: matrix.ring_dimension.clone(),
                    rows: slice.rows.as_ref().map_or_else(
                        || matrix.rows.clone(),
                        |range| {
                            ResolvedIntExpr::Sub(
                                Box::new(range.end.clone()),
                                Box::new(range.start.clone()),
                            )
                        },
                    ),
                    columns: slice.columns.as_ref().map_or_else(
                        || matrix.columns.clone(),
                        |range| {
                            ResolvedIntExpr::Sub(
                                Box::new(range.end.clone()),
                                Box::new(range.start.clone()),
                            )
                        },
                    ),
                },
                egraph[input[0]].data.canonical_coefficient_exclusive_upper.clone(),
                egraph[input[0]].data.canonical_residue_convention,
            )
        }
        MatrixTensor(children) => matrix_tensor(egraph, children),
        MatrixConcat { axis, inputs } => matrix_concat(egraph, *axis, inputs),
        Switch(children) => switch_data(egraph, children),
        ExtractCoefficient(children) => extract_coefficient_data(egraph, children),
        LiftConstantPolynomial { matrix_type, input } => {
            let Some(source) = int_data(egraph, input[0]) else {
                return invalid_analysis_data();
            };
            match source.scalar_provenance {
                Some(ScalarProvenance::Ordinary) => AnalysisData::matrix(matrix_type.clone(), None),
                Some(ScalarProvenance::SelectorOnly) => {
                    let Some(extract) = &source.direct_extract else {
                        return invalid_analysis_data();
                    };
                    AnalysisData::matrix_with_convention(
                        matrix_type.clone(),
                        extract.canonical_upper.clone(),
                        Some(CanonicalResidueConvention::Nonnegative),
                    )
                }
                None => invalid_analysis_data(),
            }
        }
        CrtRecompose { spec, inputs } => {
            let Some(spec) = egraph.analysis.symbols.crts.get(spec.0) else {
                return invalid_analysis_data();
            };
            if inputs.len() != spec.plaintext_moduli.len() || inputs.is_empty() {
                return invalid_analysis_data();
            }
            let Some(first) = matrix_sort(egraph, inputs[0]) else {
                return invalid_analysis_data();
            };
            if inputs.iter().any(|id| {
                matrix_sort(egraph, *id).is_none_or(|matrix| {
                    matrix.ring_dimension != first.ring_dimension ||
                        matrix.rows != first.rows ||
                        matrix.columns != first.columns
                })
            }) {
                return invalid_analysis_data();
            }
            let modulus = spec
                .plaintext_moduli
                .iter()
                .cloned()
                .reduce(|left, right| ResolvedIntExpr::Mul(Box::new(left), Box::new(right)))
                .expect("nonempty checked");
            AnalysisData::matrix(
                ResolvedMatrixType {
                    modulus,
                    ring_dimension: first.ring_dimension.clone(),
                    rows: first.rows.clone(),
                    columns: first.columns.clone(),
                },
                None,
            )
        }
        PackPolynomialCoefficients { matrix_type, bits, .. } => {
            if bits.iter().any(|id| egraph[*id].data.sort != Ok(MxxSort::Bool)) {
                invalid_analysis_data()
            } else {
                AnalysisData::matrix(matrix_type.clone(), None)
            }
        }
    }
}

fn graph_wire_coordinates_are_authoritative(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    descriptor: &super::identity::AtomicSourceDescriptor,
    indices: &[egg::Id],
) -> bool {
    let binders = match &descriptor.key {
        super::identity::AtomicSourceKey::GraphWire(source) => source.coordinate_binders.as_ref(),
        super::identity::AtomicSourceKey::Sampler(id) => {
            let Some(super::identity::SamplerIdentity::Preimage {
                source, indices: recorded, ..
            }) = egraph.analysis.symbols.samplers.get(id.0)
            else {
                return false;
            };
            if recorded.as_ref() != indices {
                return false;
            }
            source.coordinate_binders.as_ref()
        }
        _ => return indices.is_empty(),
    };
    if binders.len() != indices.len() {
        return false;
    }
    binders.iter().zip(indices).all(|(expected, index)| {
        let Some(id) = egraph
            .analysis
            .symbols
            .binders
            .values
            .iter()
            .position(|descriptor| &descriptor.key == expected)
        else {
            return false;
        };
        egraph[*index]
            .nodes
            .iter()
            .any(|node| matches!(node, MxxLang::IntBinder(binder) if binder.0 as usize == id))
    })
}

fn prospective_provenance_owned_elements(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    enode: &MxxLang,
) -> Option<usize> {
    match enode {
        MxxLang::Atom { source, indices } => egraph
            .analysis
            .symbols
            .atomic_sources
            .get(source.0)
            .and_then(|descriptor| descriptor.relation_role)
            .map_or(Some(0), |_| 1_usize.checked_add(indices.len())),
        MxxLang::Switch(children) => children.get(1..).map_or(Some(0), |cases| {
            // One arena node plus one immediate handle slot per branch.  The
            // branch provenance stays in its existing arena nodes.
            1_usize.checked_add(cases.len())
        }),
        _ => Some(0),
    }
}

const PROVENANCE_WORK_CHUNK_SIZE: usize = 64;

struct ProvenanceWorkChunk {
    values: [Option<usize>; PROVENANCE_WORK_CHUNK_SIZE],
    len: usize,
    previous: Option<Box<ProvenanceWorkChunk>>,
}

impl ProvenanceWorkChunk {
    fn empty() -> Self {
        Self { values: [None; PROVENANCE_WORK_CHUNK_SIZE], len: 0, previous: None }
    }
}

/// Counts a prospective copied provenance tree without an unbounded `Vec`.
/// Each fixed-size work chunk is reserved from the job budget before its box
/// is allocated, then released when traversal drops the chunk.
#[cfg(test)]
fn provenance_owned_elements_with_budget(
    data: &AnalysisData,
    budget: &mut ResourceBudget,
) -> Option<usize> {
    let mut total = 0_usize;
    for provenance in &data.relation_provenance {
        total =
            total.checked_add(provenance_owned_elements_for_with_budget(provenance, budget)?)?;
    }
    Some(total)
}

#[cfg(test)]
fn provenance_owned_elements(data: &AnalysisData) -> Option<usize> {
    let mut budget = ResourceBudget::standalone_for_tests();
    let total = provenance_owned_elements_with_budget(data, &mut budget)?;
    debug_assert_eq!(budget.temporary_elements, 0);
    Some(total)
}

fn provenance_owned_elements_for_with_budget(
    provenance: &RelationProvenance,
    budget: &mut ResourceBudget,
) -> Option<usize> {
    let arena = provenance.arena.borrow();
    budget.reserve_temporary(PROVENANCE_WORK_CHUNK_SIZE).ok()?;
    let mut reserved_chunks = 1_usize;
    let mut current = Box::new(ProvenanceWorkChunk::empty());
    current.values[current.len] = Some(provenance.node);
    current.len += 1;
    let mut total = 0_usize;

    let result = (|| {
        loop {
            if current.len == 0 {
                let Some(previous) = current.previous.take() else {
                    break;
                };
                budget.release_temporary(PROVENANCE_WORK_CHUNK_SIZE);
                reserved_chunks -= 1;
                current = previous;
                continue;
            }
            current.len -= 1;
            let node = current.values[current.len]
                .take()
                .expect("work-stack entries are initialized before they are counted");
            match &arena.nodes[node] {
                RelationProvenanceNode::Direct(source) |
                RelationProvenanceNode::Unavailable { source, .. } => {
                    total = total.checked_add(1_usize.checked_add(source.indices.len())?)?;
                }
                RelationProvenanceNode::Switch { branches, .. } => {
                    total = total.checked_add(1_usize.checked_add(branches.len())?)?;
                    for branch in branches {
                        for node in branch {
                            if current.len == PROVENANCE_WORK_CHUNK_SIZE {
                                budget.reserve_temporary(PROVENANCE_WORK_CHUNK_SIZE).ok()?;
                                reserved_chunks = reserved_chunks.checked_add(1)?;
                                current = Box::new(ProvenanceWorkChunk {
                                    values: [None; PROVENANCE_WORK_CHUNK_SIZE],
                                    len: 0,
                                    previous: Some(current),
                                });
                            }
                            current.values[current.len] = Some(*node);
                            current.len += 1;
                        }
                    }
                }
            }
        }
        Some(total)
    })();
    budget.release_temporary(reserved_chunks * PROVENANCE_WORK_CHUNK_SIZE);
    result
}

/// Counts only provenance that merge will transfer, before `merge_from` can
/// allocate branch storage or move a provenance collection into `to`.
fn merge_provenance_delta(
    to: &AnalysisData,
    from: &AnalysisData,
    relation_limit: usize,
    budget: &mut ResourceBudget,
) -> Option<(usize, bool)> {
    let additional = 0_usize;
    let mut resulting_relations = to.relation_provenance.len();
    for provenance in &from.relation_provenance {
        if !to.relation_provenance.iter().any(|existing| existing == provenance) {
            resulting_relations = resulting_relations.checked_add(1)?;
            if resulting_relations > relation_limit {
                return Some((additional, true));
            }
            // Dedup walks the arena iteratively under the temporary budget,
            // but a successful merge transfers only an existing handle.
            let _ = provenance_owned_elements_for_with_budget(provenance, budget)?;
        }
    }
    Some((additional, false))
}

fn invalid_analysis_data() -> AnalysisData {
    AnalysisData {
        sort: Err(AnalysisError::EClassSortConflict {
            expected: MxxSort::Int,
            actual: MxxSort::Bool,
        }),
        integer_domain: None,
        scalar_provenance: None,
        possible_false: false,
        possible_true: false,
        real_constant_bits: None,
        canonical_coefficient_exclusive_upper: None,
        canonical_residue_convention: None,
        direct_extract: None,
        relation_provenance: SmallVec::new(),
    }
}

fn data_for_sort(sort: MxxSort) -> AnalysisData {
    match sort {
        MxxSort::Int => AnalysisData::scalar(MxxSort::Int, None, ScalarProvenance::Ordinary),
        MxxSort::Bool => AnalysisData::boolean(true, true, ScalarProvenance::Ordinary),
        MxxSort::Real => AnalysisData {
            sort: Ok(MxxSort::Real),
            integer_domain: None,
            scalar_provenance: None,
            possible_false: false,
            possible_true: false,
            real_constant_bits: None,
            canonical_coefficient_exclusive_upper: None,
            canonical_residue_convention: None,
            direct_extract: None,
            relation_provenance: SmallVec::new(),
        },
        MxxSort::Matrix(matrix) => AnalysisData::matrix(matrix, None),
        other => AnalysisData {
            sort: Ok(other),
            integer_domain: None,
            scalar_provenance: None,
            possible_false: false,
            possible_true: false,
            real_constant_bits: None,
            canonical_coefficient_exclusive_upper: None,
            canonical_residue_convention: None,
            direct_extract: None,
            relation_provenance: SmallVec::new(),
        },
    }
}

fn int_data(egraph: &EGraph<MxxLang, MxxAnalysis>, id: egg::Id) -> Option<&AnalysisData> {
    let data = &egraph[id].data;
    (data.sort == Ok(MxxSort::Int) &&
        data.integer_domain.is_some() &&
        data.scalar_provenance.is_some())
    .then_some(data)
}

fn integer_binary(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    children: &[egg::Id; 2],
    operation: fn(&IntegerDomain, &IntegerDomain) -> Result<IntegerDomain, IntegerDomainError>,
    euclidean: bool,
) -> AnalysisData {
    let (Some(left), Some(right)) = (int_data(egraph, children[0]), int_data(egraph, children[1]))
    else {
        return invalid_analysis_data();
    };
    let Some(domain) =
        operation(left.integer_domain.as_ref().unwrap(), right.integer_domain.as_ref().unwrap())
            .ok()
    else {
        return invalid_analysis_data();
    };
    let left_provenance = left.scalar_provenance.unwrap();
    let right_provenance = right.scalar_provenance.unwrap();
    let provenance = if euclidean {
        left_provenance.euclidean(right_provenance)
    } else {
        left_provenance.runtime_arithmetic(right_provenance)
    };
    provenance
        .map(|provenance| AnalysisData::scalar(MxxSort::Int, Some(domain), provenance))
        .unwrap_or_else(invalid_analysis_data)
}

fn integer_unary(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    child: egg::Id,
    operation: fn(&IntegerDomain) -> Result<IntegerDomain, IntegerDomainError>,
) -> AnalysisData {
    let Some(source) = int_data(egraph, child) else {
        return invalid_analysis_data();
    };
    if source.scalar_provenance != Some(ScalarProvenance::Ordinary) {
        return invalid_analysis_data();
    }
    operation(source.integer_domain.as_ref().unwrap())
        .map(|domain| AnalysisData::scalar(MxxSort::Int, Some(domain), ScalarProvenance::Ordinary))
        .unwrap_or_else(|_| invalid_analysis_data())
}

fn integer_compare(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    children: &[egg::Id; 2],
    operation: impl FnOnce(&IntegerDomain, &IntegerDomain) -> Option<(bool, bool)>,
) -> AnalysisData {
    let (Some(left), Some(right)) = (int_data(egraph, children[0]), int_data(egraph, children[1]))
    else {
        return invalid_analysis_data();
    };
    let Some((possible_false, possible_true)) =
        operation(left.integer_domain.as_ref().unwrap(), right.integer_domain.as_ref().unwrap())
    else {
        return invalid_analysis_data();
    };
    AnalysisData::boolean(
        possible_false,
        possible_true,
        left.scalar_provenance.unwrap().merge(right.scalar_provenance.unwrap()),
    )
}

fn real_data(value: f64) -> AnalysisData {
    if !value.is_finite() {
        return invalid_analysis_data();
    }
    AnalysisData {
        sort: Ok(MxxSort::Real),
        integer_domain: None,
        scalar_provenance: None,
        possible_false: false,
        possible_true: false,
        real_constant_bits: Some(value.to_bits()),
        canonical_coefficient_exclusive_upper: None,
        canonical_residue_convention: None,
        direct_extract: None,
        relation_provenance: SmallVec::new(),
    }
}

fn real_binary(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    children: &[egg::Id; 2],
    operation: impl FnOnce(f64, f64) -> f64,
) -> AnalysisData {
    let (Some(left), Some(right)) =
        (egraph[children[0]].data.real_constant_bits, egraph[children[1]].data.real_constant_bits)
    else {
        return invalid_analysis_data();
    };
    real_data(operation(f64::from_bits(left), f64::from_bits(right)))
}

fn real_unary(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    child: egg::Id,
    operation: impl FnOnce(f64) -> f64,
) -> AnalysisData {
    egraph[child]
        .data
        .real_constant_bits
        .map(|value| real_data(operation(f64::from_bits(value))))
        .unwrap_or_else(invalid_analysis_data)
}

fn matrix_sort(egraph: &EGraph<MxxLang, MxxAnalysis>, id: egg::Id) -> Option<&ResolvedMatrixType> {
    match &egraph[id].data.sort {
        Ok(MxxSort::Matrix(matrix)) => Some(matrix),
        _ => None,
    }
}

fn matrix_passthrough(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    child: egg::Id,
    preserve_canonical_upper: bool,
) -> AnalysisData {
    let Some(matrix) = matrix_sort(egraph, child) else {
        return invalid_analysis_data();
    };
    AnalysisData::matrix_with_convention(
        matrix.clone(),
        preserve_canonical_upper
            .then(|| egraph[child].data.canonical_coefficient_exclusive_upper.clone())
            .flatten(),
        preserve_canonical_upper
            .then_some(egraph[child].data.canonical_residue_convention)
            .flatten(),
    )
}

fn matrix_add(egraph: &EGraph<MxxLang, MxxAnalysis>, children: &[egg::Id]) -> AnalysisData {
    let Some(first_id) = children.first() else {
        return invalid_analysis_data();
    };
    let Some(first) = matrix_sort(egraph, *first_id) else {
        return invalid_analysis_data();
    };
    if children.iter().any(|id| matrix_sort(egraph, *id) != Some(first)) {
        return invalid_analysis_data();
    }
    AnalysisData::matrix(first.clone(), None)
}

fn matrix_multiply(egraph: &EGraph<MxxLang, MxxAnalysis>, children: &[egg::Id]) -> AnalysisData {
    let Some(first_id) = children.first() else {
        return invalid_analysis_data();
    };
    let Some(first) = matrix_sort(egraph, *first_id).cloned() else {
        return invalid_analysis_data();
    };
    let mut columns = first.columns.clone();
    for child in &children[1..] {
        let Some(next) = matrix_sort(egraph, *child) else {
            return invalid_analysis_data();
        };
        if next.modulus != first.modulus ||
            next.ring_dimension != first.ring_dimension ||
            columns != next.rows
        {
            return invalid_analysis_data();
        }
        columns = next.columns.clone();
    }
    AnalysisData::matrix(
        ResolvedMatrixType {
            modulus: first.modulus,
            ring_dimension: first.ring_dimension,
            rows: first.rows,
            columns,
        },
        None,
    )
}

fn matrix_tensor(egraph: &EGraph<MxxLang, MxxAnalysis>, children: &[egg::Id; 2]) -> AnalysisData {
    let (Some(left), Some(right)) =
        (matrix_sort(egraph, children[0]), matrix_sort(egraph, children[1]))
    else {
        return invalid_analysis_data();
    };
    if left.modulus != right.modulus || left.ring_dimension != right.ring_dimension {
        return invalid_analysis_data();
    }
    AnalysisData::matrix(
        ResolvedMatrixType {
            modulus: left.modulus.clone(),
            ring_dimension: left.ring_dimension.clone(),
            rows: ResolvedIntExpr::Mul(Box::new(left.rows.clone()), Box::new(right.rows.clone())),
            columns: ResolvedIntExpr::Mul(
                Box::new(left.columns.clone()),
                Box::new(right.columns.clone()),
            ),
        },
        None,
    )
}

fn matrix_concat(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    axis: super::identity::Axis,
    inputs: &[egg::Id],
) -> AnalysisData {
    let Some(first_id) = inputs.first() else {
        return invalid_analysis_data();
    };
    let Some(first) = matrix_sort(egraph, *first_id).cloned() else {
        return invalid_analysis_data();
    };
    let mut rows = first.rows.clone();
    let mut columns = first.columns.clone();
    for input in &inputs[1..] {
        let Some(next) = matrix_sort(egraph, *input) else {
            return invalid_analysis_data();
        };
        if next.modulus != first.modulus || next.ring_dimension != first.ring_dimension {
            return invalid_analysis_data();
        }
        match axis {
            super::identity::Axis::Rows if next.columns == columns => {
                rows = ResolvedIntExpr::Add(Box::new(rows), Box::new(next.rows.clone()));
            }
            super::identity::Axis::Columns if next.rows == rows => {
                columns = ResolvedIntExpr::Add(Box::new(columns), Box::new(next.columns.clone()));
            }
            super::identity::Axis::Diagonal => {
                rows = ResolvedIntExpr::Add(Box::new(rows), Box::new(next.rows.clone()));
                columns = ResolvedIntExpr::Add(Box::new(columns), Box::new(next.columns.clone()));
            }
            _ => return invalid_analysis_data(),
        }
    }
    let canonical_upper = inputs
        .iter()
        .map(|id| egraph[*id].data.canonical_coefficient_exclusive_upper.as_ref())
        .try_fold(BigUint::zero(), |maximum, upper| Some(maximum.max(upper?.clone())));
    let convention = inputs
        .iter()
        .map(|id| egraph[*id].data.canonical_residue_convention)
        .try_fold(None, |current, convention| {
            let convention = convention?;
            match current {
                Some(current) if current != convention => None,
                Some(current) => Some(Some(current)),
                None => Some(Some(convention)),
            }
        })
        .flatten();
    AnalysisData::matrix_with_convention(
        ResolvedMatrixType {
            modulus: first.modulus,
            ring_dimension: first.ring_dimension,
            rows,
            columns,
        },
        canonical_upper,
        convention,
    )
}

fn switch_data(egraph: &EGraph<MxxLang, MxxAnalysis>, children: &[egg::Id]) -> AnalysisData {
    let Some((selector, cases)) = children.split_first() else {
        return invalid_analysis_data();
    };
    let Some(selector_data) = int_data(egraph, *selector) else {
        return invalid_analysis_data();
    };
    if cases.is_empty() {
        return invalid_analysis_data();
    }
    let Some(selector_range) =
        selector_data.integer_domain.as_ref().and_then(|domain| domain.interval().ok())
    else {
        return invalid_analysis_data();
    };
    if selector_range.minimum < BigInt::zero() ||
        selector_range.maximum >= BigInt::from(cases.len())
    {
        return invalid_analysis_data();
    }
    let first = &egraph[cases[0]].data;
    if cases.iter().any(|id| egraph[*id].data.sort != first.sort) {
        return invalid_analysis_data();
    }
    // A Switch constructs only scalar metadata plus one arena node.  Its
    // branches copy handles, never recursive provenance subtrees.
    let integer_domain = cases
        .iter()
        .map(|id| egraph[*id].data.integer_domain.as_ref())
        .try_fold(None, |acc: Option<IntegerDomain>, domain| {
            let domain = domain?;
            Some(Some(match acc {
                Some(acc) => acc.hull(domain).ok()?,
                None => domain.clone(),
            }))
        })
        .flatten();
    let scalar_provenance = cases
        .iter()
        .map(|id| egraph[*id].data.scalar_provenance)
        .try_fold(None, |acc, provenance| {
            let provenance = provenance?;
            Some(Some(acc.map_or(provenance, |acc: ScalarProvenance| acc.merge(provenance))))
        })
        .flatten();
    let possible_false = cases.iter().any(|id| egraph[*id].data.possible_false);
    let possible_true = cases.iter().any(|id| egraph[*id].data.possible_true);
    let real_constant_bits = cases
        .iter()
        .map(|id| egraph[*id].data.real_constant_bits)
        .try_fold(None, |acc, bits| {
            let bits = bits?;
            match acc {
                Some(existing) if existing != bits => None,
                Some(existing) => Some(Some(existing)),
                None => Some(Some(bits)),
            }
        })
        .flatten();
    let canonical_coefficient_exclusive_upper = cases
        .iter()
        .map(|id| egraph[*id].data.canonical_coefficient_exclusive_upper.as_ref())
        .try_fold(BigUint::zero(), |maximum, upper| Some(maximum.max(upper?.clone())));
    let canonical_residue_convention = cases
        .iter()
        .map(|id| egraph[*id].data.canonical_residue_convention)
        .try_fold(None, |current, convention| {
            let convention = convention?;
            match current {
                Some(current) if current != convention => None,
                Some(current) => Some(Some(current)),
                None => Some(Some(convention)),
            }
        })
        .flatten();
    match &first.sort {
        Ok(MxxSort::Int) if integer_domain.is_none() || scalar_provenance.is_none() => {
            return invalid_analysis_data()
        }
        Ok(MxxSort::Bool) if scalar_provenance.is_none() => return invalid_analysis_data(),
        Ok(MxxSort::Real) if real_constant_bits.is_none() => return invalid_analysis_data(),
        _ => {}
    }
    let branches = cases
        .iter()
        .map(|id| {
            egraph[*id]
                .data
                .relation_provenance
                .iter()
                .map(|provenance| {
                    debug_assert!(Rc::ptr_eq(&provenance.arena, &egraph.analysis.provenance_arena));
                    provenance.node
                })
                .collect()
        })
        .collect();
    let node = egraph
        .analysis
        .provenance_arena
        .borrow_mut()
        .intern(RelationProvenanceNode::Switch { originating_selector: *selector, branches });
    AnalysisData {
        sort: first.sort.clone(),
        integer_domain,
        scalar_provenance,
        possible_false,
        possible_true,
        real_constant_bits,
        canonical_coefficient_exclusive_upper,
        canonical_residue_convention,
        direct_extract: None,
        relation_provenance: SmallVec::from_vec(vec![RelationProvenance {
            arena: Rc::clone(&egraph.analysis.provenance_arena),
            node,
        }]),
    }
}

fn extract_coefficient_data(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    children: &[egg::Id; 2],
) -> AnalysisData {
    let Some(matrix) = matrix_sort(egraph, children[0]) else {
        return invalid_analysis_data();
    };
    if int_data(egraph, children[1]).is_none() {
        return invalid_analysis_data();
    }
    let ResolvedIntExpr::Const(modulus) = &matrix.modulus else {
        return invalid_analysis_data();
    };
    let Some(modulus) = modulus.to_biguint() else {
        return invalid_analysis_data();
    };
    let Some(convention) = egraph[children[0]].data.canonical_residue_convention else {
        return invalid_analysis_data();
    };
    let Ok(domain) = MxxAnalysis::extract_coefficient_domain(
        matrix,
        &modulus,
        egraph[children[0]].data.canonical_coefficient_exclusive_upper.as_ref(),
        convention,
    ) else {
        return invalid_analysis_data();
    };
    let mut data =
        AnalysisData::scalar(MxxSort::Int, Some(domain.clone()), ScalarProvenance::SelectorOnly);
    let interval = domain.interval().ok();
    data.direct_extract = Some(DirectExtractFact {
        canonical_upper: interval.and_then(|interval| {
            (interval.minimum == BigInt::zero())
                .then(|| (&interval.maximum + BigInt::one()).to_biguint())
                .flatten()
        }),
    });
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::identity::{
        AtomicRelationRole, AtomicSourceDescriptor, AtomicSourceKey, BinderDescriptor,
        GraphWireSourceKey, IntegerSourceDomain, MatrixConstantSpec, MatrixConstantValue,
        OccurrenceScope, ProgramKey, SliceSpec, WireSourceKey,
    };

    fn scalar_matrix_type(columns: i64) -> ResolvedMatrixType {
        ResolvedMatrixType {
            modulus: ResolvedIntExpr::Const(17.into()),
            ring_dimension: ResolvedIntExpr::Const(1.into()),
            rows: ResolvedIntExpr::Const(1.into()),
            columns: ResolvedIntExpr::Const(columns.into()),
        }
    }

    fn test_direct_provenance() -> RelationProvenance {
        let arena = Rc::new(RefCell::new(RelationProvenanceArena::default()));
        let node = arena.borrow_mut().intern(RelationProvenanceNode::Direct(RelationSource {
            source: AtomicSourceId(0),
            indices: Box::new([]),
        }));
        RelationProvenance { arena, node }
    }

    fn constant_matrix(
        egraph: &mut EGraph<MxxLang, MxxAnalysis>,
        matrix_type: ResolvedMatrixType,
        value: MatrixConstantValue,
    ) -> egg::Id {
        let id = egraph
            .analysis
            .symbols
            .matrix_constants
            .intern(MatrixConstantSpec { matrix_type, value });
        egraph.add(MxxLang::MatrixConstant(super::super::identity::MatrixConstantSpecId(id)))
    }

    #[test]
    fn affine_endpoints_are_linear_in_binder_count() {
        let key = |slot| BinderKey {
            loop_scope: super::super::identity::OccurrenceScope {
                program: super::super::identity::ProgramKey::Ideal,
                definition: mxx_ir_core::FrozenGraphScopeId::Root,
                path: Box::new([]),
            },
            loop_node: mxx_ir_core::NodeId(slot),
            slot: 0,
        };
        let first = key(1);
        let second = key(2);
        let domain = IntegerDomain::Affine {
            constant: BigInt::from(5),
            coefficients: BTreeMap::from([
                (first.clone(), BigInt::from(3)),
                (second.clone(), BigInt::from(-2)),
            ]),
            binders: BTreeMap::from([
                (first, IntegerInterval::new(0.into(), 4.into()).unwrap()),
                (second, IntegerInterval::new(1.into(), 6.into()).unwrap()),
            ]),
        };
        assert_eq!(
            domain.interval().unwrap(),
            IntegerInterval::new((-7).into(), 15.into()).unwrap()
        );
    }

    #[test]
    fn runtime_euclidean_domains_match_runtime_convention() {
        let one_to_two =
            IntegerDomain::IntervalOnly(IntegerInterval::new(1.into(), 2.into()).unwrap());
        assert_eq!(
            one_to_two.euclidean_div(&IntegerDomain::exact(2)).unwrap().interval().unwrap(),
            IntegerInterval::new(0.into(), 1.into()).unwrap()
        );
        let negative =
            IntegerDomain::IntervalOnly(IntegerInterval::new((-3).into(), (-1).into()).unwrap());
        assert_eq!(
            negative.euclidean_div(&IntegerDomain::exact(-2)).unwrap().interval().unwrap(),
            IntegerInterval::new((-2).into(), (-1).into()).unwrap()
        );
        let range =
            IntegerDomain::IntervalOnly(IntegerInterval::new((-3).into(), 4.into()).unwrap());
        assert_eq!(
            range.euclidean_remainder(&IntegerDomain::exact(-2)).unwrap().interval().unwrap(),
            IntegerInterval::new(0.into(), 1.into()).unwrap()
        );
    }

    #[test]
    fn round_div_uses_shared_floor_semantics_for_negative_values() {
        let domain = IntegerDomain::exact(-4).round_div(&IntegerDomain::exact(3)).unwrap();
        assert_eq!(domain.interval().unwrap(), IntegerInterval::exact((-1).into()));
    }

    #[test]
    fn intrinsic_integer_nodes_read_authoritative_symbol_facts() {
        let binder = BinderKey {
            loop_scope: super::super::identity::OccurrenceScope {
                program: super::super::identity::ProgramKey::Ideal,
                definition: mxx_ir_core::FrozenGraphScopeId::Root,
                path: Box::new([]),
            },
            loop_node: mxx_ir_core::NodeId(7),
            slot: 0,
        };
        let mut symbols = SymbolTables::default();
        let binder_id = symbols.binders.intern(BinderDescriptor {
            key: binder,
            minimum: 0.into(),
            maximum: 8.into(),
        });
        symbols.integer_parameters.insert("p".to_owned(), 11.into());
        let atom_id = symbols.atomic_sources.intern(AtomicSourceDescriptor {
            key: AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from("selector")),
            sort: MxxSort::Int,
            integer_domain: Some(IntegerSourceDomain { minimum: 2.into(), maximum: 5.into() }),
            canonical_residue_convention: None,
            relation_role: None,
        });
        let mut egraph = EGraph::new(MxxAnalysis::new(symbols));

        let binder_term =
            egraph.add(MxxLang::IntBinder(super::super::identity::BinderId(binder_id)));
        let parameter_term = egraph.add(MxxLang::IntParameter("p".to_owned()));
        let atom_term =
            egraph.add(MxxLang::Atom { source: AtomicSourceId(atom_id), indices: Box::new([]) });

        assert_eq!(
            egraph[binder_term].data.integer_domain.as_ref().unwrap().interval().unwrap(),
            IntegerInterval::new(0.into(), 8.into()).unwrap()
        );
        assert_eq!(
            egraph[parameter_term].data.integer_domain,
            Some(IntegerDomain::Exact(11.into()))
        );
        assert_eq!(
            egraph[atom_term].data.integer_domain.as_ref().unwrap().interval().unwrap(),
            IntegerInterval::new(2.into(), 5.into()).unwrap()
        );
    }

    #[test]
    fn switch_rejects_a_selector_domain_outside_its_cases() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(2.into()));
        let first = egraph.add(MxxLang::IntConst(10.into()));
        let second = egraph.add(MxxLang::IntConst(11.into()));
        let selected =
            egraph.add(MxxLang::Switch(vec![selector, first, second].into_boxed_slice()));

        assert!(egraph[selected].data.sort.is_err());
    }

    #[test]
    fn relation_roles_are_analysis_owned_and_switch_preserves_branches() {
        let mut symbols = SymbolTables::default();
        let source = |name: &str, role| AtomicSourceDescriptor {
            key: AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from(name)),
            sort: MxxSort::Matrix(scalar_matrix_type(1)),
            integer_domain: None,
            canonical_residue_convention: Some(CanonicalResidueConvention::Nonnegative),
            relation_role: Some(role),
        };
        let direct =
            symbols.atomic_sources.intern(source("preimage", AtomicRelationRole::Preimage));
        let unavailable = symbols.atomic_sources.intern(source(
            "small",
            AtomicRelationRole::SmallDecomposedHash { range_proved: false },
        ));
        let mut egraph = EGraph::new(MxxAnalysis::new(symbols));
        let direct =
            egraph.add(MxxLang::Atom { source: AtomicSourceId(direct), indices: Box::new([]) });
        let unavailable = egraph
            .add(MxxLang::Atom { source: AtomicSourceId(unavailable), indices: Box::new([]) });
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let selected =
            egraph.add(MxxLang::Switch(vec![selector, direct, unavailable].into_boxed_slice()));

        let mut direct_count = 0;
        visit_relation_provenance(&egraph[direct].data.relation_provenance, |visit| {
            direct_count += matches!(visit, RelationProvenanceVisit::Direct(_)) as usize;
        });
        assert_eq!(direct_count, 1);
        let mut unavailable_count = 0;
        visit_relation_provenance(&egraph[unavailable].data.relation_provenance, |visit| {
            unavailable_count +=
                matches!(visit, RelationProvenanceVisit::Unavailable { .. }) as usize;
        });
        assert_eq!(unavailable_count, 1);
        let mut switch_branch_count = None;
        visit_relation_provenance(&egraph[selected].data.relation_provenance, |visit| {
            if let RelationProvenanceVisit::Switch { branch_count, .. } = visit {
                switch_branch_count = Some(branch_count);
            }
        });
        assert_eq!(switch_branch_count, Some(2));
    }

    #[test]
    fn selector_only_forbidden_consumers_and_direct_lift_are_closed() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let matrix_type = scalar_matrix_type(1);
        let matrix =
            constant_matrix(&mut egraph, matrix_type.clone(), MatrixConstantValue::Identity);
        let position = egraph.add(MxxLang::IntConst(0.into()));
        let extract = egraph.add(MxxLang::ExtractCoefficient([matrix, position]));
        let direct_lift = egraph.add(MxxLang::LiftConstantPolynomial {
            matrix_type: matrix_type.clone(),
            input: [extract],
        });
        let to_real = egraph.add(MxxLang::IntToReal([extract]));
        let zero = egraph.add(MxxLang::IntConst(0.into()));
        let laundered = egraph.add(MxxLang::IntAdd([extract, zero]));
        let indirect_lift =
            egraph.add(MxxLang::LiftConstantPolynomial { matrix_type, input: [laundered] });

        assert_eq!(
            egraph[direct_lift].data.canonical_coefficient_exclusive_upper,
            Some(BigUint::from(2_u8))
        );
        assert!(egraph[to_real].data.sort.is_err());
        assert!(egraph[indirect_lift].data.sort.is_err());
    }

    #[test]
    fn switch_cannot_launder_a_direct_extract_even_when_all_cases_are_the_same() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let matrix_type = scalar_matrix_type(1);
        let matrix =
            constant_matrix(&mut egraph, matrix_type.clone(), MatrixConstantValue::Identity);
        let position = egraph.add(MxxLang::IntConst(0.into()));
        let extract = egraph.add(MxxLang::ExtractCoefficient([matrix, position]));
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let switched =
            egraph.add(MxxLang::Switch(vec![selector, extract, extract].into_boxed_slice()));
        let lift = egraph.add(MxxLang::LiftConstantPolynomial { matrix_type, input: [switched] });

        assert!(egraph[extract].data.direct_extract.is_some());
        assert!(egraph[switched].data.direct_extract.is_none());
        assert!(egraph[lift].data.sort.is_err());
    }

    #[test]
    fn arithmetic_and_matrix_negate_clear_direct_or_canonical_facts() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let matrix_type = scalar_matrix_type(1);
        let matrix =
            constant_matrix(&mut egraph, matrix_type.clone(), MatrixConstantValue::Identity);
        let position = egraph.add(MxxLang::IntConst(0.into()));
        let extract = egraph.add(MxxLang::ExtractCoefficient([matrix, position]));
        let zero = egraph.add(MxxLang::IntConst(0.into()));
        let added = egraph.add(MxxLang::IntAdd([extract, zero]));
        let negated = egraph.add(MxxLang::MatrixNegate([matrix]));

        assert!(egraph[added].data.direct_extract.is_none());
        assert!(egraph[added].data.sort.is_err());
        assert_eq!(egraph[negated].data.canonical_coefficient_exclusive_upper, None);
    }

    #[test]
    fn incompatible_matrix_and_switch_sorts_fail_closed() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let left = constant_matrix(&mut egraph, scalar_matrix_type(1), MatrixConstantValue::Zero);
        let right = constant_matrix(&mut egraph, scalar_matrix_type(2), MatrixConstantValue::Zero);
        let add = egraph.add(MxxLang::MatrixAdd(vec![left, right].into_boxed_slice()));
        let multiply = egraph.add(MxxLang::MatrixMultiply(vec![right, right].into_boxed_slice()));
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let int_case = egraph.add(MxxLang::IntConst(1.into()));
        let switch = egraph.add(MxxLang::Switch(vec![selector, left, int_case].into_boxed_slice()));

        assert!(egraph[add].data.sort.is_err());
        assert!(egraph[multiply].data.sort.is_err());
        assert!(egraph[switch].data.sort.is_err());
    }

    #[test]
    fn canonical_transfer_is_complete_for_views_concat_and_switch() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let matrix_type = scalar_matrix_type(1);
        let zero = constant_matrix(&mut egraph, matrix_type.clone(), MatrixConstantValue::Zero);
        let identity =
            constant_matrix(&mut egraph, matrix_type.clone(), MatrixConstantValue::Identity);
        let transpose = egraph.add(MxxLang::MatrixTranspose([identity]));
        let slice_id =
            egraph.analysis.symbols.slices.intern(SliceSpec { rows: None, columns: None });
        let slice = egraph.add(MxxLang::MatrixSlice {
            spec: super::super::identity::SliceSpecId(slice_id),
            input: [identity],
        });
        let concat = egraph.add(MxxLang::MatrixConcat {
            axis: super::super::identity::Axis::Rows,
            inputs: vec![zero, identity].into_boxed_slice(),
        });
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let switch = egraph.add(MxxLang::Switch(vec![selector, zero, identity].into_boxed_slice()));

        for id in [transpose, slice, concat, switch] {
            assert_eq!(
                egraph[id].data.canonical_coefficient_exclusive_upper,
                Some(BigUint::from(2_u8))
            );
        }
    }

    #[test]
    fn concat_and_switch_drop_unknown_canonical_ranges() {
        let mut symbols = SymbolTables::default();
        let unknown = symbols.atomic_sources.intern(AtomicSourceDescriptor {
            key: AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from("unknown")),
            sort: MxxSort::Matrix(scalar_matrix_type(1)),
            integer_domain: None,
            canonical_residue_convention: Some(CanonicalResidueConvention::Nonnegative),
            relation_role: None,
        });
        let mut egraph = EGraph::new(MxxAnalysis::new(symbols));
        let unknown =
            egraph.add(MxxLang::Atom { source: AtomicSourceId(unknown), indices: Box::new([]) });
        let known =
            constant_matrix(&mut egraph, scalar_matrix_type(1), MatrixConstantValue::Identity);
        let concat = egraph.add(MxxLang::MatrixConcat {
            axis: super::super::identity::Axis::Rows,
            inputs: vec![known, unknown].into_boxed_slice(),
        });
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let switch = egraph.add(MxxLang::Switch(vec![selector, known, unknown].into_boxed_slice()));

        assert_eq!(egraph[concat].data.canonical_coefficient_exclusive_upper, None);
        assert_eq!(egraph[switch].data.canonical_coefficient_exclusive_upper, None);
    }

    #[test]
    fn relation_atoms_require_matrix_sort_and_exact_ordered_graph_coordinates() {
        let scope = OccurrenceScope {
            program: ProgramKey::Ideal,
            definition: mxx_ir_core::FrozenGraphScopeId::Root,
            path: Box::new([]),
        };
        let first_key =
            BinderKey { loop_scope: scope.clone(), loop_node: mxx_ir_core::NodeId(1), slot: 0 };
        let second_key =
            BinderKey { loop_scope: scope.clone(), loop_node: mxx_ir_core::NodeId(2), slot: 0 };
        let graph_key = |binders: Box<[BinderKey]>| {
            AtomicSourceKey::GraphWire(GraphWireSourceKey {
                wire: WireSourceKey {
                    scope: scope.clone(),
                    wire: mxx_ir_core::WireRef {
                        node: mxx_ir_core::NodeId(3),
                        port: mxx_ir_core::Port(0),
                    },
                },
                coordinate_binders: binders,
            })
        };
        let mut symbols = SymbolTables::default();
        let first = symbols.binders.intern(BinderDescriptor {
            key: first_key.clone(),
            minimum: 0.into(),
            maximum: 1.into(),
        });
        let second = symbols.binders.intern(BinderDescriptor {
            key: second_key.clone(),
            minimum: 0.into(),
            maximum: 1.into(),
        });
        let valid = symbols.atomic_sources.intern(AtomicSourceDescriptor {
            key: graph_key(Box::new([first_key.clone(), second_key.clone()])),
            sort: MxxSort::Matrix(scalar_matrix_type(1)),
            integer_domain: None,
            canonical_residue_convention: None,
            relation_role: Some(AtomicRelationRole::Preimage),
        });
        let nonmatrix = symbols.atomic_sources.intern(AtomicSourceDescriptor {
            key: AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from("bad-role")),
            sort: MxxSort::Int,
            integer_domain: Some(IntegerSourceDomain { minimum: 0.into(), maximum: 1.into() }),
            canonical_residue_convention: None,
            relation_role: Some(AtomicRelationRole::Preimage),
        });
        let mut egraph = EGraph::new(MxxAnalysis::new(symbols));
        let first = egraph.add(MxxLang::IntBinder(super::super::identity::BinderId(first)));
        let second = egraph.add(MxxLang::IntBinder(super::super::identity::BinderId(second)));
        let ordered = egraph.add(MxxLang::Atom {
            source: AtomicSourceId(valid),
            indices: Box::new([first, second]),
        });
        let reversed = egraph.add(MxxLang::Atom {
            source: AtomicSourceId(valid),
            indices: Box::new([second, first]),
        });
        let short =
            egraph.add(MxxLang::Atom { source: AtomicSourceId(valid), indices: Box::new([first]) });
        let nonmatrix =
            egraph.add(MxxLang::Atom { source: AtomicSourceId(nonmatrix), indices: Box::new([]) });

        assert_eq!(egraph[ordered].data.relation_provenance.len(), 1);
        for invalid in [reversed, short, nonmatrix] {
            assert!(matches!(
                egraph[invalid].data.sort,
                Err(AnalysisError::InvalidSamplerDescriptor { .. })
            ));
            assert!(egraph[invalid].data.relation_provenance.is_empty());
        }
    }

    #[test]
    fn deep_egraph_switch_add_uses_shallow_arena_handles() {
        const DEPTH: usize = 20_000;
        let mut symbols = SymbolTables::default();
        let relation = symbols.atomic_sources.intern(AtomicSourceDescriptor {
            key: AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from("relation")),
            sort: MxxSort::Matrix(scalar_matrix_type(1)),
            integer_domain: None,
            canonical_residue_convention: None,
            relation_role: Some(AtomicRelationRole::Preimage),
        });
        std::thread::Builder::new()
            .stack_size(512 * 1024)
            .spawn(move || {
                let mut egraph = EGraph::new(MxxAnalysis::new(symbols));
                let selector = egraph.add(MxxLang::IntConst(0.into()));
                let mut selected = egraph
                    .add(MxxLang::Atom { source: AtomicSourceId(relation), indices: Box::new([]) });
                for _ in 0..DEPTH {
                    selected =
                        egraph.add(MxxLang::Switch(vec![selector, selected].into_boxed_slice()));
                }
                assert_eq!(provenance_owned_elements(&egraph[selected].data), Some(1 + 2 * DEPTH));
                assert_eq!(egraph.analysis.resource_budget.temporary_elements, 0);
            })
            .expect("constrained-stack egraph worker must start")
            .join()
            .expect("constrained-stack egraph worker must complete");
    }

    #[test]
    fn cumulative_owned_element_budget_is_sticky() {
        let mut analysis = MxxAnalysis::default();
        analysis.resource_budget.total_owned_element_limit = 1;
        let mut egraph = EGraph::new(analysis);
        let first = egraph.add(MxxLang::IntConst(0.into()));
        let second = egraph.add(MxxLang::IntConst(1.into()));
        let _ = egraph.add(MxxLang::IntAdd([first, second]));

        assert!(matches!(
            egraph.analysis.resource_failure,
            Some(ResourceObserved::Counter { limit: 1, observed: 2 })
        ));
    }

    #[test]
    fn make_budget_failure_does_not_construct_switch_provenance() {
        let mut symbols = SymbolTables::default();
        let relation = symbols.atomic_sources.intern(AtomicSourceDescriptor {
            key: AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from("relation")),
            sort: MxxSort::Matrix(scalar_matrix_type(1)),
            integer_domain: None,
            canonical_residue_convention: None,
            relation_role: Some(AtomicRelationRole::Preimage),
        });
        let mut egraph = EGraph::new(MxxAnalysis::new(symbols));
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let first =
            egraph.add(MxxLang::Atom { source: AtomicSourceId(relation), indices: Box::new([]) });
        let before = egraph.analysis.resource_budget.owned();
        // The Switch needs only its arena node and immediate branch handles,
        // plus the language node's children: 66 + 66 owned elements.
        egraph.analysis.resource_budget.total_owned_element_limit = before + 131;
        let mut children = vec![selector];
        children.extend(std::iter::repeat(first).take(65));

        let switch = egraph.add(MxxLang::Switch(children.into_boxed_slice()));

        assert!(matches!(
            egraph.analysis.resource_failure,
            Some(ResourceObserved::Counter { limit, observed }) if limit == (before + 131) as u64 && observed == (before + 132) as u64
        ));
        assert!(egraph[switch].data.relation_provenance.is_empty());
    }

    #[test]
    fn deeply_nested_switch_preflight_charges_only_the_new_shallow_switch() {
        const DEPTH: usize = 96;
        let mut symbols = SymbolTables::default();
        let relation = symbols.atomic_sources.intern(AtomicSourceDescriptor {
            key: AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from("relation")),
            sort: MxxSort::Matrix(scalar_matrix_type(1)),
            integer_domain: None,
            canonical_residue_convention: None,
            relation_role: Some(AtomicRelationRole::Preimage),
        });
        let mut egraph = EGraph::new(MxxAnalysis::new(symbols));
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let mut selected =
            egraph.add(MxxLang::Atom { source: AtomicSourceId(relation), indices: Box::new([]) });
        for _ in 0..DEPTH {
            selected = egraph.add(MxxLang::Switch(vec![selector, selected].into_boxed_slice()));
        }

        let before = egraph.analysis.resource_budget.owned();
        egraph.analysis.resource_budget.total_owned_element_limit = before + 3;
        let overflow = egraph.add(MxxLang::Switch(vec![selector, selected].into_boxed_slice()));

        assert_eq!(
            egraph.analysis.resource_failure,
            Some(ResourceObserved::Counter {
                limit: (before + 3) as u64,
                observed: (before + 4) as u64,
            })
        );
        assert!(egraph[overflow].data.relation_provenance.is_empty());
    }

    #[test]
    fn selector_provenance_survives_euclidean_operations_and_merge() {
        let selector = ScalarProvenance::SelectorOnly;
        assert_eq!(selector.euclidean(ScalarProvenance::Ordinary), Some(selector));
        assert_eq!(selector.runtime_arithmetic(ScalarProvenance::Ordinary), None);
        assert_eq!(selector.merge(ScalarProvenance::Ordinary), selector);
    }

    #[test]
    fn authoritative_extract_range_is_narrower_than_modulus() {
        let matrix = ResolvedMatrixType {
            modulus: ResolvedIntExpr::Const(97.into()),
            ring_dimension: ResolvedIntExpr::Const(1.into()),
            rows: ResolvedIntExpr::Const(1.into()),
            columns: ResolvedIntExpr::Const(1.into()),
        };
        let domain = MxxAnalysis::extract_coefficient_domain(
            &matrix,
            &BigUint::from(97_u32),
            Some(&BigUint::from(11_u32)),
            CanonicalResidueConvention::Nonnegative,
        )
        .unwrap();
        assert_eq!(domain.interval().unwrap(), IntegerInterval::new(0.into(), 10.into()).unwrap());
        assert!(
            MxxAnalysis::extract_coefficient_domain(
                &matrix,
                &BigUint::from(97_u32),
                Some(&BigUint::from(98_u32)),
                CanonicalResidueConvention::Nonnegative,
            )
            .is_err()
        );
        let centered = MxxAnalysis::extract_coefficient_domain(
            &matrix,
            &BigUint::from(97_u32),
            None,
            CanonicalResidueConvention::Centered,
        )
        .unwrap();
        assert_eq!(
            centered.interval().unwrap(),
            IntegerInterval::new((-48).into(), 48.into()).unwrap()
        );
        assert!(
            MxxAnalysis::extract_coefficient_domain(
                &matrix,
                &BigUint::from(97_u32),
                Some(&BigUint::from(11_u32)),
                CanonicalResidueConvention::Centered,
            )
            .is_err()
        );
    }

    #[test]
    fn merge_keeps_missing_required_scalar_facts_as_a_sticky_sort_error() {
        let mut complete = AnalysisData::scalar(
            MxxSort::Int,
            Some(IntegerDomain::exact(1)),
            ScalarProvenance::Ordinary,
        );
        let incomplete = AnalysisData::scalar(MxxSort::Int, None, ScalarProvenance::Ordinary);

        let (_, overflow) = complete.merge_from(incomplete, 64, &mut ResourceBudget::production());

        assert!(!overflow);
        assert!(matches!(complete.sort, Err(AnalysisError::EClassSortConflict { .. })));
    }

    #[test]
    fn relation_limit_failure_is_sticky_on_the_analysis_owner() {
        let mut to = AnalysisData::matrix(
            ResolvedMatrixType {
                modulus: ResolvedIntExpr::Const(17.into()),
                ring_dimension: ResolvedIntExpr::Const(1.into()),
                rows: ResolvedIntExpr::Const(1.into()),
                columns: ResolvedIntExpr::Const(1.into()),
            },
            None,
        );
        let mut from = to.clone();
        from.relation_provenance.push(test_direct_provenance());
        let mut analysis = MxxAnalysis::with_relation_limit(SymbolTables::default(), 0);

        let _ = <MxxAnalysis as Analysis<MxxLang>>::merge(&mut analysis, &mut to, from);

        assert_eq!(
            analysis.resource_failure,
            Some(ResourceObserved::Counter { limit: 0, observed: 1 })
        );
    }

    #[test]
    fn merge_budget_failure_does_not_insert_new_provenance() {
        let mut to = AnalysisData::matrix(scalar_matrix_type(1), None);
        let mut from = AnalysisData::matrix(scalar_matrix_type(1), None);
        from.relation_provenance.push(test_direct_provenance());
        let mut analysis = MxxAnalysis::default();
        analysis.resource_budget.total_owned_element_limit = 0;

        let merged = <MxxAnalysis as Analysis<MxxLang>>::merge(&mut analysis, &mut to, from);

        assert!(!merged.0 && !merged.1);
        assert!(to.relation_provenance.is_empty());
        assert_eq!(analysis.resource_budget.owned(), 0);
        assert_eq!(
            analysis.resource_failure,
            Some(ResourceObserved::Counter {
                limit: 0,
                observed: PROVENANCE_WORK_CHUNK_SIZE as u64,
            })
        );
    }

    #[test]
    fn deep_merge_preflight_releases_scratch_but_keeps_cumulative_reservations() {
        const DEPTH: usize = 256;
        let mut symbols = SymbolTables::default();
        let relation = symbols.atomic_sources.intern(AtomicSourceDescriptor {
            key: AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from("relation")),
            sort: MxxSort::Matrix(scalar_matrix_type(1)),
            integer_domain: None,
            canonical_residue_convention: None,
            relation_role: Some(AtomicRelationRole::Preimage),
        });
        let mut egraph = EGraph::new(MxxAnalysis::new(symbols));
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let mut selected =
            egraph.add(MxxLang::Atom { source: AtomicSourceId(relation), indices: Box::new([]) });
        for _ in 0..DEPTH {
            selected = egraph.add(MxxLang::Switch(vec![selector, selected].into_boxed_slice()));
        }
        let mut to = AnalysisData::matrix(scalar_matrix_type(1), None);
        let from = egraph[selected].data.clone();
        let before = egraph.analysis.resource_budget.owned();
        let merged = <MxxAnalysis as Analysis<MxxLang>>::merge(&mut egraph.analysis, &mut to, from);
        assert!(merged.0 && merged.1);
        assert_eq!(egraph.analysis.resource_budget.temporary_elements, 0);
        assert!(egraph.analysis.resource_budget.owned() > before);

        let mut failed_to = AnalysisData::matrix(scalar_matrix_type(1), None);
        let failed_from = egraph[selected].data.clone();
        let before = egraph.analysis.resource_budget.owned();
        egraph.analysis.resource_budget.total_owned_element_limit = before + 1;
        let merged = <MxxAnalysis as Analysis<MxxLang>>::merge(
            &mut egraph.analysis,
            &mut failed_to,
            failed_from,
        );
        assert!(!merged.0 && !merged.1);
        assert!(failed_to.relation_provenance.is_empty());
        assert_eq!(egraph.analysis.resource_budget.temporary_elements, 0);
    }
}
