//! Egg-independent expression DAG and ordered polynomial normal form.
//!
//! `TermId` is only a job-local edge into [`ExpressionDag`].  It is never used
//! as a canonical equality key: symbolic equality is defined by the complete
//! owner-aware [`FactorIdentity`] carried by a factor and by ordered factor
//! lists.  This keeps the normal form independent of e-graph insertion order.

use super::{
    bound::{
        BoundClass, MatrixBound, MatrixMetadata, MatrixProductFacts, product_bound,
        product_bound_with_facts,
    },
    identity::{
        AtomicSourceKey, Axis, BinderKey, CrtSpec, ResolvedIndexRange, ResolvedIntExpr,
        ResolvedMatrixType, SliceSpec, TrapdoorSourceKey, substitute_resolved_int_expr,
    },
    normal_form_ops::{IntegerInterval, ScaleScalar, ViewSpec},
};
use mxx_ir_core::{Port, types::ConcreteMatrixType};
use num_bigint::{BigInt, BigUint};
use num_traits::{ToPrimitive, Zero};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    sync::Arc,
};

#[path = "normal_form_product.rs"]
pub(crate) mod normal_form_product;
pub(crate) use normal_form_product::NormalizationCounters;

pub use super::normal_form_relation::{FullRelationKey, RelationRegistration, RelationRegistry};

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FactorLayoutIdentity {
    pub matrix: ResolvedMatrixType,
    pub view: Option<SliceSpec>,
}

/// Stable index into one job-local expression DAG.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TermId(pub u32);

/// Hash-consed structural identity handle.  It is not a DAG `TermId` and is
/// never exposed as a semantic node number outside this job-local arena.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct MatrixValueIdentityId(pub u32);

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum MatrixValueOperation {
    Zero,
    Atom,
    Add,
    Negate,
    Product,
    Switch { reachable: Box<[usize]> },
    Select { reachable: Box<[usize]> },
    FamilyGetStatic { index: usize },
    FamilyGetDynamic { stored_indices: Box<[BigUint]>, domain_upper: BigUint },
    MatrixScale { scalar: ScaleScalar },
    Transpose,
    Slice { spec: SliceSpec },
    Tensor,
    LiftConstantPolynomial { matrix_type: ConcreteMatrixType, domain: IntegerInterval },
    View { view: ViewSpec, output_type: ConcreteMatrixType },
    CrtRecompose { spec: CrtSpec, output_type: ConcreteMatrixType },
    Concat { axis: Axis, output_type: ConcreteMatrixType },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct MatrixValueIdentityNode {
    pub operation: MatrixValueOperation,
    pub children: Box<[MatrixValueIdentityId]>,
    pub owner: Option<FactorIdentity>,
    pub selector: Option<FactorIdentity>,
}

/// Facts computed once when an expression enters the job-local DAG.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatrixValueFacts {
    pub concrete_type: Option<ConcreteMatrixType>,
    pub metadata: MatrixMetadata,
    pub identity: MatrixValueIdentityId,
}

/// Typed kind of one owner-resolved symbolic factor.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum FactorKind {
    Signal,
    RelationTarget,
    SwitchBarrier,
    #[cfg(test)]
    Test(Box<str>),
}

/// Typed identity of one plain-hash argument. Finite summaries remain value
/// contracts rather than being converted to debug text.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum HashPlainArgumentIdentity {
    Exact(FactorIdentity),
    Bounded { matrix_type: ConcreteMatrixType, coefficient_class: BoundClass },
    ExactZero,
}

/// The owner of a factor.  All production variants are existing typed
/// identity components; the named variant exists only for unit fixtures.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum FactorOwner {
    Atomic(AtomicSourceKey),
    Trapdoor(TrapdoorSourceKey),
    /// Canonical owner-aware scalar identity used by selector barriers.  It
    /// deliberately does not mention the graph output node that materialized
    /// the selector, so equal selectors share one barrier identity.
    Scalar(ResolvedIntExpr),
    /// A plain hash query together with its ordered, canonical scalar input
    /// identities.  The query and argument identities are part of the key;
    /// no compact arena ID or debug rendering is persistent identity.
    HashPlain {
        query: Box<FactorIdentity>,
        arguments: Box<[HashPlainArgumentIdentity]>,
    },
    Derived {
        parent: Box<FactorIdentity>,
        tag: Box<[u8]>,
    },
    #[cfg(test)]
    Named(Box<str>),
}

#[cfg(test)]
impl From<&str> for FactorOwner {
    fn from(value: &str) -> Self {
        Self::Named(value.into())
    }
}

/// Complete owner-resolved identity of one exact symbolic factor.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FactorIdentity {
    pub owner: FactorOwner,
    pub kind: FactorKind,
    pub port: Port,
    /// Runtime coordinates are paired with their introducing binders.  A
    /// production identity cannot carry an unowned coordinate or binder.
    pub coordinates: Box<[(BinderKey, ResolvedIntExpr)]>,
    pub public: Option<AtomicSourceKey>,
    pub layout: Option<FactorLayoutIdentity>,
    pub selector: Option<Box<FactorIdentity>>,
    /// Actual trapdoor provenance is symbolic identity, never a lookup wildcard.
    pub trapdoor: Option<TrapdoorSourceKey>,
    /// Ordered reachable-case mapping for selector-owned barriers.
    pub selector_mapping: Box<[BigUint]>,
}

impl FactorIdentity {
    pub fn atomic(
        source: AtomicSourceKey,
        coordinates: impl IntoIterator<Item = (BinderKey, ResolvedIntExpr)>,
    ) -> Self {
        Self {
            owner: FactorOwner::Atomic(source),
            kind: FactorKind::Signal,
            port: Port(0),
            coordinates: coordinates.into_iter().collect(),
            public: None,
            layout: None,
            selector: None,
            trapdoor: None,
            selector_mapping: Box::new([]),
        }
    }

    pub fn scalar_selector(identity: ResolvedIntExpr) -> Self {
        Self {
            owner: FactorOwner::Scalar(identity),
            kind: FactorKind::Signal,
            port: Port(0),
            coordinates: Box::new([]),
            public: None,
            layout: None,
            selector: None,
            trapdoor: None,
            selector_mapping: Box::new([]),
        }
    }

    pub fn trapdoor(
        source: TrapdoorSourceKey,
        coordinates: impl IntoIterator<Item = (BinderKey, ResolvedIntExpr)>,
    ) -> Self {
        Self {
            owner: FactorOwner::Trapdoor(source.clone()),
            kind: FactorKind::Signal,
            port: Port(0),
            coordinates: coordinates.into_iter().collect(),
            public: None,
            layout: None,
            selector: None,
            trapdoor: Some(source.clone()),
            selector_mapping: Box::new([]),
        }
    }

    fn switch_barrier(
        selector: FactorIdentity,
        selector_mapping: Box<[BigUint]>,
        fingerprint: Box<[u8]>,
    ) -> Self {
        Self {
            owner: FactorOwner::Derived { parent: Box::new(selector.clone()), tag: fingerprint },
            kind: FactorKind::SwitchBarrier,
            port: Port(0),
            coordinates: Box::new([]),
            public: None,
            layout: None,
            selector: Some(Box::new(selector)),
            trapdoor: None,
            selector_mapping,
        }
    }

    pub fn coordinates(&self) -> &[(BinderKey, ResolvedIntExpr)] {
        &self.coordinates
    }

    #[cfg(test)]
    pub fn named(name: impl Into<Box<str>>) -> Self {
        let name = name.into();
        Self {
            owner: FactorOwner::Named("named".into()),
            kind: FactorKind::Test(name),
            port: Port(0),
            coordinates: Box::new([]),
            public: None,
            layout: None,
            selector: None,
            trapdoor: None,
            selector_mapping: Box::new([]),
        }
    }
}

/// A symbolic factor.  Bounds and relation liveness are contracts, not part
/// of symbolic identity, so equal keys can never become distinct monomials.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SymbolicFactor {
    pub key: FactorIdentity,
    pub bound: BoundClass,
    pub relation_live: bool,
    /// Actual trapdoor provenance carried by the factor.  This is distinct
    /// from the preimage factor identity used to look up a relation.
    pub trapdoor: Option<TrapdoorSourceKey>,
    /// Full shape/metadata contract used whenever this finite factor is
    /// multiplied into a summary.  The class above remains the symbolic
    /// contract exposed to canonical equality; this field is never a key.
    pub matrix_bound: Option<MatrixBound>,
    /// Value facts are intentionally separate from the noise bound.
    pub matrix_value_metadata: MatrixMetadata,
    pub switch: Option<Arc<SwitchData>>,
}

impl SymbolicFactor {
    pub fn large(key: FactorIdentity) -> Self {
        Self {
            key,
            bound: BoundClass::Large,
            relation_live: false,
            trapdoor: None,
            matrix_bound: None,
            matrix_value_metadata: MatrixMetadata::unknown(),
            switch: None,
        }
    }

    pub fn bounded(key: FactorIdentity, bound: MatrixBound) -> Result<Self, NormalFormError> {
        if matches!(bound.coefficient_class, BoundClass::Large) {
            return Err(NormalFormError::LargeFactorCannotBeBounded);
        }
        Ok(Self {
            key,
            bound: bound.coefficient_class.clone(),
            relation_live: false,
            trapdoor: None,
            matrix_bound: Some(bound),
            matrix_value_metadata: MatrixMetadata::unknown(),
            switch: None,
        })
    }

    pub fn relation_live(key: FactorIdentity, bound: MatrixBound) -> Result<Self, NormalFormError> {
        if matches!(bound.coefficient_class, BoundClass::Large) {
            return Err(NormalFormError::RelationLiveRequiresFiniteBound);
        }
        Ok(Self {
            key,
            bound: bound.coefficient_class.clone(),
            relation_live: true,
            trapdoor: None,
            matrix_bound: Some(bound),
            matrix_value_metadata: MatrixMetadata::unknown(),
            switch: None,
        })
    }

    pub fn with_trapdoor(mut self, trapdoor: TrapdoorSourceKey) -> Self {
        self.key.trapdoor = Some(trapdoor.clone());
        self.trapdoor = Some(trapdoor);
        self
    }

    fn switch(
        key: FactorIdentity,
        bound: BoundClass,
        matrix_bound: Option<MatrixBound>,
        data: Arc<SwitchData>,
    ) -> Self {
        Self {
            key,
            bound,
            relation_live: false,
            trapdoor: None,
            matrix_bound,
            matrix_value_metadata: MatrixMetadata::unknown(),
            switch: Some(data),
        }
    }
}

/// A selector barrier retains only reachable, already-normalized cases.  Its
/// structural identity is independent of numeric bounds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SwitchData {
    pub selector: FactorIdentity,
    pub cases: Box<[PolynomialNF]>,
    /// Source-case mapping in the order represented by `cases`.  The values
    /// are semantic indices, never DAG `TermId`s.
    pub case_indices: Box<[BigUint]>,
    /// Full owner-resolved source structure for each case.  This is a compact
    /// fingerprint, not an expression-tree cache; numeric coefficient caps
    /// are deliberately omitted from it.
    pub case_fingerprints: Box<[Box<str>]>,
}

/// One monomial with a significant factor order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Monomial {
    factors: Box<[SymbolicFactor]>,
}

impl Monomial {
    pub fn one() -> Self {
        Self { factors: Box::new([]) }
    }
    pub fn from_factor(factor: SymbolicFactor) -> Self {
        Self { factors: Box::new([factor]) }
    }
    pub fn factors(&self) -> &[SymbolicFactor] {
        &self.factors
    }
    pub(crate) fn from_factors(factors: impl IntoIterator<Item = SymbolicFactor>) -> Self {
        Self { factors: factors.into_iter().collect() }
    }
    pub fn key(&self) -> MonomialKey {
        MonomialKey(self.factors.iter().map(|factor| factor.key.clone()).collect())
    }
    fn concat(&self, other: &Self) -> Self {
        let mut factors = Vec::with_capacity(self.factors.len() + other.factors.len());
        factors.extend(self.factors.iter().cloned());
        factors.extend(other.factors.iter().cloned());
        Self { factors: factors.into_boxed_slice() }
    }
}

/// Canonical ordered factor identity list.  No bound or TermId participates.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct MonomialKey(Box<[FactorIdentity]>);

impl MonomialKey {
    pub fn factors(&self) -> &[FactorIdentity] {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BoundedSummary {
    ExactZero,
    Bounded(MatrixBound),
}

impl BoundedSummary {
    pub fn is_exact_zero(&self) -> bool {
        matches!(self, Self::ExactZero)
    }
    pub fn as_matrix_bound(&self) -> Option<&MatrixBound> {
        match self {
            Self::Bounded(bound) => Some(bound),
            Self::ExactZero => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SignedMonomial {
    pub monomial: Monomial,
    pub multiplicity: BigInt,
}

/// A canonical sum of ordered monomials plus one finite noise summary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolynomialNF {
    exact_terms: BTreeMap<MonomialKey, SignedMonomial>,
    bounded_summary: BoundedSummary,
}

impl Default for PolynomialNF {
    fn default() -> Self {
        Self::zero()
    }
}

impl PolynomialNF {
    pub fn zero() -> Self {
        Self { exact_terms: BTreeMap::new(), bounded_summary: BoundedSummary::ExactZero }
    }

    /// Multiplicative identity used only as the seed for ordered products.
    pub fn one() -> Self {
        Self::from_monomial(Monomial::one())
    }

    pub fn exact_factor(key: FactorIdentity) -> Self {
        Self::from_monomial(Monomial::from_factor(SymbolicFactor::large(key)))
    }

    pub fn bounded(bound: MatrixBound) -> Result<Self, NormalFormError> {
        if matches!(bound.coefficient_class, BoundClass::Large) {
            return Err(NormalFormError::LargeBoundCannotBeSummarized);
        }
        let summary = if matches!(bound.coefficient_class, BoundClass::ExactZero) {
            BoundedSummary::ExactZero
        } else {
            BoundedSummary::Bounded(bound)
        };
        Ok(Self { exact_terms: BTreeMap::new(), bounded_summary: summary })
    }

    pub fn relation_live_factor(
        key: FactorIdentity,
        bound: MatrixBound,
    ) -> Result<Self, NormalFormError> {
        Ok(Self::from_monomial(Monomial::from_factor(SymbolicFactor::relation_live(key, bound)?)))
    }

    /// Finite non-relation factors are noise and are folded at construction.
    pub fn bounded_factor(
        key: FactorIdentity,
        bound: MatrixBound,
    ) -> Result<Self, NormalFormError> {
        Ok(Self::from_monomial(Monomial::from_factor(SymbolicFactor::bounded(key, bound)?)))
    }

    pub fn exact_terms(&self) -> &BTreeMap<MonomialKey, SignedMonomial> {
        &self.exact_terms
    }
    pub fn bounded_summary(&self) -> &BoundedSummary {
        &self.bounded_summary
    }
    pub(crate) fn from_parts(
        exact_terms: BTreeMap<MonomialKey, SignedMonomial>,
        bounded_summary: BoundedSummary,
    ) -> Self {
        Self { exact_terms, bounded_summary }
    }
    pub fn is_exact_zero(&self) -> bool {
        self.exact_terms.is_empty() && self.bounded_summary.is_exact_zero()
    }

    pub fn add(mut self, other: Self) -> Result<Self, NormalFormError> {
        self.bounded_summary = add_summary(self.bounded_summary, other.bounded_summary)?;
        for term in other.exact_terms.into_values() {
            self.insert(term.monomial, term.multiplicity)?;
        }
        Ok(self)
    }

    pub fn sum<I>(children: I) -> Result<Self, NormalFormError>
    where
        I: IntoIterator<Item = Result<Self, NormalFormError>>,
    {
        children.into_iter().try_fold(Self::zero(), |left, right| left.add(right?))
    }

    pub fn negate(mut self) -> Self {
        for term in self.exact_terms.values_mut() {
            term.multiplicity = -std::mem::take(&mut term.multiplicity);
        }
        self
    }

    pub fn subtract(self, other: Self) -> Result<Self, NormalFormError> {
        self.add(other.negate())
    }

    pub fn first_large_witness(&self) -> Option<LargeWitness> {
        self.exact_terms.values().find_map(|term| {
            term.monomial.factors.iter().enumerate().find_map(|(index, factor)| {
                matches!(factor.bound, BoundClass::Large).then(|| LargeWitness {
                    monomial: term.monomial.key(),
                    factor_index: index,
                    identity: factor.key.clone(),
                })
            })
        })
    }

    /// Finish relation processing at the root and fold every finite term that
    /// remains unmatched. A live identity survives only when it is not safely
    /// finite (for example, a malformed Large contract).
    pub fn finish_relation_live(mut self) -> Result<Self, NormalFormError> {
        let mut ignored_fold_count = 0_u64;
        self.fold_finite_non_live_terms(&mut ignored_fold_count)?;
        Ok(self)
    }

    pub(crate) fn finish_relation_live_counted(
        mut self,
        fold_count: &mut u64,
    ) -> Result<Self, NormalFormError> {
        self.fold_finite_non_live_terms(fold_count)?;
        Ok(self)
    }

    fn fold_finite_non_live_terms(&mut self, fold_count: &mut u64) -> Result<(), NormalFormError> {
        let keys = self.exact_terms.keys().cloned().collect::<Vec<_>>();
        for key in keys {
            let Some(term) = self.exact_terms.remove(&key) else { continue };
            if term.monomial.factors.iter().any(|factor| matches!(factor.bound, BoundClass::Large))
            {
                self.exact_terms.insert(key, term);
                continue;
            }
            let bound = monomial_bound(&term.monomial)?;
            self.bounded_summary = add_summary(
                self.bounded_summary.clone(),
                summary_from_bound(scale_by_multiplicity(bound, &term.multiplicity)),
            )?;
            *fold_count = fold_count.saturating_add(1);
        }
        Ok(())
    }

    /// Final acceptance gate for the root residual.
    pub fn validate_bounded_only(&self) -> Result<&BoundedSummary, NormalFormError> {
        if let Some((key, term)) = self.exact_terms.iter().next() {
            let _ = term;
            return Err(NormalFormError::UnconsumedExactTerm { key: key.clone() });
        }
        Ok(&self.bounded_summary)
    }

    fn from_monomial(monomial: Monomial) -> Self {
        if monomial.factors.iter().all(|factor| {
            !factor.relation_live &&
                !matches!(factor.bound, BoundClass::Large) &&
                factor.switch.is_none()
        }) {
            if let Ok(bound) = monomial_bound(&monomial) {
                return Self {
                    exact_terms: BTreeMap::new(),
                    bounded_summary: summary_from_bound(bound),
                };
            }
        }
        let key = monomial.key();
        let mut exact_terms = BTreeMap::new();
        exact_terms.insert(key, SignedMonomial { monomial, multiplicity: BigInt::from(1) });
        Self { exact_terms, bounded_summary: BoundedSummary::ExactZero }
    }

    pub(crate) fn insert(
        &mut self,
        monomial: Monomial,
        multiplicity: BigInt,
    ) -> Result<(), NormalFormError> {
        if multiplicity.is_zero() {
            return Ok(());
        }
        let key = monomial.key();
        if let Some(existing) = self.exact_terms.get_mut(&key) {
            if existing
                .monomial
                .factors
                .iter()
                .map(|f| (&f.bound, f.relation_live, &f.trapdoor, &f.matrix_bound))
                .ne(monomial
                    .factors
                    .iter()
                    .map(|f| (&f.bound, f.relation_live, &f.trapdoor, &f.matrix_bound)))
            {
                return Err(NormalFormError::ConflictingFactorContracts { key });
            }
            existing.multiplicity += multiplicity;
            if existing.multiplicity.is_zero() {
                self.exact_terms.remove(&key);
            }
        } else {
            self.exact_terms.insert(key, SignedMonomial { monomial, multiplicity });
        }
        Ok(())
    }

    pub(crate) fn multiply_summary_counted(
        mut self,
        summary: &BoundedSummary,
        terms: &BTreeMap<MonomialKey, SignedMonomial>,
        fold_count: &mut u64,
    ) -> Result<Self, NormalFormError> {
        for term in terms.values() {
            if term.monomial.factors.iter().any(|factor| matches!(factor.bound, BoundClass::Large))
            {
                return Err(NormalFormError::BoundedSummaryMixedWithLarge);
            }
            let product = match summary {
                BoundedSummary::ExactZero => BoundedSummary::ExactZero,
                BoundedSummary::Bounded(left) => {
                    if term.monomial.factors.is_empty() {
                        BoundedSummary::Bounded(scale_by_multiplicity(
                            left.clone(),
                            &term.multiplicity,
                        ))
                    } else {
                        let factor_bound = monomial_bound(&term.monomial)?;
                        BoundedSummary::Bounded(scale_by_multiplicity(
                            product_bound(left, &factor_bound).map_err(NormalFormError::bound)?,
                            &term.multiplicity,
                        ))
                    }
                }
            };
            let contributes = !matches!(product, BoundedSummary::ExactZero);
            self.bounded_summary = add_summary(self.bounded_summary, product)?;
            if contributes {
                *fold_count = fold_count.saturating_add(1);
            }
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LargeWitness {
    pub monomial: MonomialKey,
    pub factor_index: usize,
    pub identity: FactorIdentity,
}

/// Expression node.  Terms are immutable after insertion, making DFS memo
/// and visiting-state cycle detection sufficient for honest producer DAGs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExpressionNode {
    Zero,
    Atom(SymbolicFactor),
    Add(Box<[TermId]>),
    Negate(TermId),
    Product(Box<[TermId]>),
    Switch {
        selector: FactorIdentity,
        cases: Box<[TermId]>,
        reachable: Box<[usize]>,
    },
    /// Select is the canonical single-selector spelling used by lowered
    /// protocol families.  It has the same checked semantics as Switch;
    /// keeping the two spellings explicit avoids making callers encode a
    /// protocol operation as an arbitrary product or add node.
    Select {
        selector: FactorIdentity,
        cases: Box<[TermId]>,
        reachable: Box<[usize]>,
    },
    FamilyGetStatic {
        cases: Box<[TermId]>,
        index: usize,
    },
    FamilyGetDynamic {
        selector: FactorIdentity,
        cases: Box<[TermId]>,
        stored_indices: Box<[BigUint]>,
        domain_upper: BigUint,
    },
    MatrixScale {
        input: TermId,
        scalar: ScaleScalar,
    },
    Transpose(TermId),
    Slice {
        input: TermId,
        spec: SliceSpec,
    },
    Tensor {
        left: TermId,
        right: TermId,
    },
    LiftConstantPolynomial {
        input: TermId,
        matrix_type: ConcreteMatrixType,
        domain: IntegerInterval,
    },
    View {
        input: TermId,
        view: ViewSpec,
        output_type: ConcreteMatrixType,
    },
    CrtRecompose {
        inputs: Box<[TermId]>,
        spec: CrtSpec,
        output_type: ConcreteMatrixType,
    },
    Concat {
        inputs: Box<[TermId]>,
        axis: Axis,
        output_type: ConcreteMatrixType,
    },
}

#[derive(Clone, Debug, Default)]
pub struct ExpressionDag {
    nodes: Vec<ExpressionNode>,
    facts: Vec<MatrixValueFacts>,
    identity_nodes: Vec<MatrixValueIdentityNode>,
    identity_index: BTreeMap<MatrixValueIdentityNode, MatrixValueIdentityId>,
}

fn slice_concrete_type(
    mut matrix: Option<ConcreteMatrixType>,
    spec: &SliceSpec,
) -> Option<ConcreteMatrixType> {
    if let Some(range) = spec.rows.as_ref() {
        let start = match &range.start {
            ResolvedIntExpr::Const(value) => value.to_usize()?,
            _ => return None,
        };
        let end = match &range.end {
            ResolvedIntExpr::Const(value) => value.to_usize()?,
            _ => return None,
        };
        let matrix_ref = matrix.as_mut()?;
        if start >= end || end > matrix_ref.rows {
            return None;
        }
        matrix_ref.rows = end - start;
    }
    if let Some(range) = spec.columns.as_ref() {
        let start = match &range.start {
            ResolvedIntExpr::Const(value) => value.to_usize()?,
            _ => return None,
        };
        let end = match &range.end {
            ResolvedIntExpr::Const(value) => value.to_usize()?,
            _ => return None,
        };
        let matrix_ref = matrix.as_mut()?;
        if start >= end || end > matrix_ref.columns {
            return None;
        }
        matrix_ref.columns = end - start;
    }
    matrix
}

impl ExpressionDag {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn push(&mut self, node: ExpressionNode) -> Result<TermId, NormalFormError> {
        let id =
            TermId(u32::try_from(self.nodes.len()).map_err(|_| NormalFormError::TooManyTerms)?);
        if node.children().iter().any(|child| child.0 as usize >= self.nodes.len()) {
            return Err(NormalFormError::InvalidTermId);
        }
        let facts = self.derive_facts(&node)?;
        self.nodes.push(node);
        self.facts.push(facts);
        Ok(id)
    }
    pub fn node(&self, id: TermId) -> Result<&ExpressionNode, NormalFormError> {
        self.nodes.get(id.0 as usize).ok_or(NormalFormError::InvalidTermId)
    }
    pub fn facts(&self, id: TermId) -> Result<&MatrixValueFacts, NormalFormError> {
        self.facts.get(id.0 as usize).ok_or(NormalFormError::InvalidTermId)
    }

    pub fn identity_node(&self, id: MatrixValueIdentityId) -> Option<&MatrixValueIdentityNode> {
        self.identity_nodes.get(id.0 as usize)
    }

    fn intern_identity(&mut self, node: MatrixValueIdentityNode) -> MatrixValueIdentityId {
        if let Some(id) = self.identity_index.get(&node).copied() {
            return id;
        }
        let id = MatrixValueIdentityId(self.identity_nodes.len() as u32);
        self.identity_index.insert(node.clone(), id);
        self.identity_nodes.push(node);
        id
    }

    fn derive_facts(&mut self, node: &ExpressionNode) -> Result<MatrixValueFacts, NormalFormError> {
        let children = node.children();
        let inputs =
            children.iter().map(|child| self.facts(*child)).collect::<Result<Vec<_>, _>>()?;
        let child_ids = inputs.iter().map(|fact| fact.identity).collect::<Vec<_>>();
        let operation = node.operation_identity_tag();
        let selector = match node {
            ExpressionNode::Switch { selector, .. } |
            ExpressionNode::Select { selector, .. } |
            ExpressionNode::FamilyGetDynamic { selector, .. } => Some(selector.clone()),
            _ => None,
        };
        let concrete_type = match node {
            ExpressionNode::Transpose(_) => inputs.first().and_then(|fact| {
                let mut matrix = fact.concrete_type.clone()?;
                std::mem::swap(&mut matrix.rows, &mut matrix.columns);
                Some(matrix)
            }),
            ExpressionNode::Tensor { .. } => {
                let left = inputs.first().and_then(|fact| fact.concrete_type.clone());
                let right = inputs.get(1).and_then(|fact| fact.concrete_type.clone());
                match (left, right) {
                    (Some(left), Some(right))
                        if left.modulus == right.modulus &&
                            left.ring_dimension == right.ring_dimension =>
                    {
                        let rows = left.rows.checked_mul(right.rows);
                        let columns = left.columns.checked_mul(right.columns);
                        rows.zip(columns).map(|(rows, columns)| ConcreteMatrixType {
                            modulus: left.modulus,
                            ring_dimension: left.ring_dimension,
                            rows,
                            columns,
                        })
                    }
                    _ => None,
                }
            }
            ExpressionNode::Slice { spec, .. } => slice_concrete_type(
                inputs.first().and_then(|fact| fact.concrete_type.clone()),
                spec,
            ),
            _ => node
                .explicit_matrix_type()
                .cloned()
                .or_else(|| inputs.first().and_then(|fact| fact.concrete_type.clone())),
        };
        let metadata = match node {
            ExpressionNode::Atom(factor) => factor.matrix_value_metadata.clone(),
            ExpressionNode::Transpose(_) |
            ExpressionNode::Slice { .. } |
            ExpressionNode::View { .. } => inputs
                .first()
                .map(|fact| fact.metadata.clone())
                .unwrap_or_else(MatrixMetadata::unknown),
            ExpressionNode::FamilyGetStatic { index, .. } => inputs
                .get(*index)
                .map(|fact| fact.metadata.clone())
                .unwrap_or_else(MatrixMetadata::unknown),
            ExpressionNode::FamilyGetDynamic { cases, .. } => {
                merge_reachable_metadata(&inputs, &(0..cases.len()).collect::<Vec<_>>())
            }
            ExpressionNode::Switch { .. } | ExpressionNode::Select { .. } => {
                merge_reachable_metadata(&inputs, node.reachable_indices())
            }
            ExpressionNode::LiftConstantPolynomial { domain, .. } => MatrixMetadata {
                canonical_coefficient_exclusive_upper: domain.direct_extract_upper.clone(),
                is_constant_polynomial: true,
                known_zero_rows: None,
            },
            _ => MatrixMetadata::unknown(),
        };
        let identity = self.intern_identity(MatrixValueIdentityNode {
            operation,
            children: child_ids.into_boxed_slice(),
            owner: match node {
                ExpressionNode::Atom(factor) => Some(factor.key.clone()),
                _ => None,
            },
            selector,
        });
        Ok(MatrixValueFacts { concrete_type, metadata, identity })
    }
    #[cfg(test)]
    pub(crate) fn term_count(&self) -> usize {
        self.nodes.len()
    }
    #[cfg(test)]
    pub(crate) fn identity_count(&self) -> usize {
        self.identity_nodes.len()
    }
    pub fn normalize(
        &self,
        root: TermId,
        registry: &RelationRegistry,
    ) -> Result<PolynomialNF, NormalFormError> {
        normal_form_product::Normalizer::new(self, registry).normalize(root)?.finish_relation_live()
    }

    /// Normalize one root and return the counters owned by this DAG job.
    /// Counters are collected at the normalizer's actual traversal and
    /// relation/folding boundaries; they are not inferred from lowered wires.
    pub(crate) fn normalize_with_counters(
        &self,
        root: TermId,
        registry: &RelationRegistry,
    ) -> Result<(PolynomialNF, NormalizationCounters), NormalFormError> {
        normal_form_product::Normalizer::new(self, registry).normalize_with_counters(root)
    }

    /// Strict root gate that additionally rejects any exact residual.
    pub fn normalize_bounded(
        &self,
        root: TermId,
        registry: &RelationRegistry,
    ) -> Result<BoundedSummary, NormalFormError> {
        let normalized = self.normalize(root, registry)?;
        Ok(normalized.validate_bounded_only()?.clone())
    }

    /// Bind one owner-resolved loop coordinate in an arbitrary DAG subtree.
    ///
    /// This is deliberately a caller-owned memo operation: a lowering job can
    /// reuse the same memo for every family access without introducing a
    /// second expression cache.  The walk rebuilds every node shape, including
    /// nested switches, family accesses, and structural operations; it is not
    /// limited to atom/zero representatives.
    pub(crate) fn substitute_binder(
        &mut self,
        root: TermId,
        binder: &BinderKey,
        replacement: &ResolvedIntExpr,
        memo: &mut BTreeMap<(TermId, BinderKey, ResolvedIntExpr), TermId>,
    ) -> Result<TermId, NormalFormError> {
        // Keep the overwhelmingly common atom/zero path in a small stack
        // frame.  The general structural dispatcher below is intentionally
        // separate: its large enum match must not consume the constrained
        // stack used by the deep lowering stress test.
        let memo_key = (root, binder.clone(), replacement.clone());
        if let Some(term) = memo.get(&memo_key) {
            return Ok(*term);
        }
        match self.node(root)?.clone() {
            ExpressionNode::Zero => {
                let result = self.push(ExpressionNode::Zero)?;
                memo.insert(memo_key, result);
                Ok(result)
            }
            ExpressionNode::Atom(mut factor) => {
                factor.key = substitute_factor_identity(&factor.key, binder, replacement);
                factor.trapdoor = factor
                    .trapdoor
                    .map(|source| substitute_trapdoor_source(source, binder, replacement));
                factor.key.trapdoor = factor
                    .key
                    .trapdoor
                    .clone()
                    .map(|source| substitute_trapdoor_source(source, binder, replacement));
                factor.switch = factor.switch.map(|switch| {
                    Arc::new(SwitchData {
                        selector: substitute_factor_identity(&switch.selector, binder, replacement),
                        cases: switch
                            .cases
                            .iter()
                            .map(|case| substitute_nf(case, binder, replacement))
                            .collect(),
                        case_indices: switch.case_indices.clone(),
                        case_fingerprints: switch.case_fingerprints.clone(),
                    })
                });
                let result = self.push(ExpressionNode::Atom(factor))?;
                memo.insert(memo_key, result);
                Ok(result)
            }
            _ => self.substitute_binder_complex(root, binder, replacement, memo),
        }
    }

    fn substitute_binder_complex(
        &mut self,
        root: TermId,
        binder: &BinderKey,
        replacement: &ResolvedIntExpr,
        memo: &mut BTreeMap<(TermId, BinderKey, ResolvedIntExpr), TermId>,
    ) -> Result<TermId, NormalFormError> {
        let memo_key = (root, binder.clone(), replacement.clone());
        if let Some(term) = memo.get(&memo_key) {
            return Ok(*term);
        }
        let node = self.node(root)?.clone();
        let rebuilt = match node {
            ExpressionNode::Zero | ExpressionNode::Atom(_) => unreachable!("fast path handled"),
            ExpressionNode::Add(children) => ExpressionNode::Add(
                children
                    .iter()
                    .map(|child| self.substitute_binder(*child, binder, replacement, memo))
                    .collect::<Result<Box<_>, _>>()?,
            ),
            ExpressionNode::Negate(child) => {
                ExpressionNode::Negate(self.substitute_binder(child, binder, replacement, memo)?)
            }
            ExpressionNode::Product(children) => ExpressionNode::Product(
                children
                    .iter()
                    .map(|child| self.substitute_binder(*child, binder, replacement, memo))
                    .collect::<Result<Box<_>, _>>()?,
            ),
            ExpressionNode::Switch { selector, cases, reachable } => ExpressionNode::Switch {
                selector: substitute_factor_identity(&selector, binder, replacement),
                cases: cases
                    .iter()
                    .map(|child| self.substitute_binder(*child, binder, replacement, memo))
                    .collect::<Result<Box<_>, _>>()?,
                reachable,
            },
            ExpressionNode::Select { selector, cases, reachable } => ExpressionNode::Select {
                selector: substitute_factor_identity(&selector, binder, replacement),
                cases: cases
                    .iter()
                    .map(|child| self.substitute_binder(*child, binder, replacement, memo))
                    .collect::<Result<Box<_>, _>>()?,
                reachable,
            },
            ExpressionNode::FamilyGetStatic { cases, index } => ExpressionNode::FamilyGetStatic {
                cases: cases
                    .iter()
                    .map(|child| self.substitute_binder(*child, binder, replacement, memo))
                    .collect::<Result<Box<_>, _>>()?,
                index,
            },
            ExpressionNode::FamilyGetDynamic { selector, cases, stored_indices, domain_upper } => {
                ExpressionNode::FamilyGetDynamic {
                    selector: substitute_factor_identity(&selector, binder, replacement),
                    cases: cases
                        .iter()
                        .map(|child| self.substitute_binder(*child, binder, replacement, memo))
                        .collect::<Result<Box<_>, _>>()?,
                    stored_indices,
                    domain_upper,
                }
            }
            ExpressionNode::MatrixScale { input, scalar } => ExpressionNode::MatrixScale {
                input: self.substitute_binder(input, binder, replacement, memo)?,
                scalar: substitute_scale_scalar(scalar, binder, replacement),
            },
            ExpressionNode::Transpose(input) => ExpressionNode::Transpose(self.substitute_binder(
                input,
                binder,
                replacement,
                memo,
            )?),
            ExpressionNode::Slice { input, spec } => ExpressionNode::Slice {
                input: self.substitute_binder(input, binder, replacement, memo)?,
                spec: substitute_slice_spec(spec, binder, replacement),
            },
            ExpressionNode::Tensor { left, right } => ExpressionNode::Tensor {
                left: self.substitute_binder(left, binder, replacement, memo)?,
                right: self.substitute_binder(right, binder, replacement, memo)?,
            },
            ExpressionNode::LiftConstantPolynomial { input, matrix_type, domain } => {
                ExpressionNode::LiftConstantPolynomial {
                    input: self.substitute_binder(input, binder, replacement, memo)?,
                    matrix_type,
                    domain,
                }
            }
            ExpressionNode::View { input, view, output_type } => ExpressionNode::View {
                input: self.substitute_binder(input, binder, replacement, memo)?,
                view,
                output_type,
            },
            ExpressionNode::CrtRecompose { inputs, spec, output_type } => {
                ExpressionNode::CrtRecompose {
                    inputs: inputs
                        .iter()
                        .map(|child| self.substitute_binder(*child, binder, replacement, memo))
                        .collect::<Result<Box<_>, _>>()?,
                    spec: substitute_crt_spec(spec, binder, replacement),
                    output_type,
                }
            }
            ExpressionNode::Concat { inputs, axis, output_type } => ExpressionNode::Concat {
                inputs: inputs
                    .iter()
                    .map(|child| self.substitute_binder(*child, binder, replacement, memo))
                    .collect::<Result<Box<_>, _>>()?,
                axis,
                output_type,
            },
        };
        let result = self.push(rebuilt)?;
        memo.insert(memo_key, result);
        Ok(result)
    }

    /// Replaces owner-resolved placeholder factors in a DAG subtree.  The
    /// replacement map is supplied by the caller for one recurrence step, so
    /// every next state observes the same previous state vector.  `memo` is
    /// intentionally job-local and can be discarded when that vector changes.
    pub(crate) fn substitute_factors(
        &mut self,
        root: TermId,
        replacements: &BTreeMap<FactorIdentity, TermId>,
        memo: &mut BTreeMap<TermId, TermId>,
    ) -> Result<TermId, NormalFormError> {
        if let Some(term) = memo.get(&root) {
            return Ok(*term);
        }
        let node = self.node(root)?.clone();
        if let ExpressionNode::Atom(factor) = &node {
            if let Some(term) = replacements.get(&factor.key) {
                memo.insert(root, *term);
                return Ok(*term);
            }
        }
        let rebuilt = match node {
            ExpressionNode::Zero => ExpressionNode::Zero,
            ExpressionNode::Atom(factor) => ExpressionNode::Atom(factor),
            ExpressionNode::Add(children) => ExpressionNode::Add(
                children
                    .iter()
                    .map(|child| self.substitute_factors(*child, replacements, memo))
                    .collect::<Result<Box<_>, _>>()?,
            ),
            ExpressionNode::Negate(child) => {
                ExpressionNode::Negate(self.substitute_factors(child, replacements, memo)?)
            }
            ExpressionNode::Product(children) => ExpressionNode::Product(
                children
                    .iter()
                    .map(|child| self.substitute_factors(*child, replacements, memo))
                    .collect::<Result<Box<_>, _>>()?,
            ),
            ExpressionNode::Switch { selector, cases, reachable } => ExpressionNode::Switch {
                selector,
                cases: cases
                    .iter()
                    .map(|child| self.substitute_factors(*child, replacements, memo))
                    .collect::<Result<Box<_>, _>>()?,
                reachable,
            },
            ExpressionNode::Select { selector, cases, reachable } => ExpressionNode::Select {
                selector,
                cases: cases
                    .iter()
                    .map(|child| self.substitute_factors(*child, replacements, memo))
                    .collect::<Result<Box<_>, _>>()?,
                reachable,
            },
            ExpressionNode::FamilyGetStatic { cases, index } => ExpressionNode::FamilyGetStatic {
                cases: cases
                    .iter()
                    .map(|child| self.substitute_factors(*child, replacements, memo))
                    .collect::<Result<Box<_>, _>>()?,
                index,
            },
            ExpressionNode::FamilyGetDynamic { selector, cases, stored_indices, domain_upper } => {
                ExpressionNode::FamilyGetDynamic {
                    selector,
                    cases: cases
                        .iter()
                        .map(|child| self.substitute_factors(*child, replacements, memo))
                        .collect::<Result<Box<_>, _>>()?,
                    stored_indices,
                    domain_upper,
                }
            }
            ExpressionNode::MatrixScale { input, scalar } => ExpressionNode::MatrixScale {
                input: self.substitute_factors(input, replacements, memo)?,
                scalar,
            },
            ExpressionNode::Transpose(input) => {
                ExpressionNode::Transpose(self.substitute_factors(input, replacements, memo)?)
            }
            ExpressionNode::Slice { input, spec } => ExpressionNode::Slice {
                input: self.substitute_factors(input, replacements, memo)?,
                spec,
            },
            ExpressionNode::Tensor { left, right } => ExpressionNode::Tensor {
                left: self.substitute_factors(left, replacements, memo)?,
                right: self.substitute_factors(right, replacements, memo)?,
            },
            ExpressionNode::LiftConstantPolynomial { input, matrix_type, domain } => {
                ExpressionNode::LiftConstantPolynomial {
                    input: self.substitute_factors(input, replacements, memo)?,
                    matrix_type,
                    domain,
                }
            }
            ExpressionNode::View { input, view, output_type } => ExpressionNode::View {
                input: self.substitute_factors(input, replacements, memo)?,
                view,
                output_type,
            },
            ExpressionNode::CrtRecompose { inputs, spec, output_type } => {
                ExpressionNode::CrtRecompose {
                    inputs: inputs
                        .iter()
                        .map(|child| self.substitute_factors(*child, replacements, memo))
                        .collect::<Result<Box<_>, _>>()?,
                    spec,
                    output_type,
                }
            }
            ExpressionNode::Concat { inputs, axis, output_type } => ExpressionNode::Concat {
                inputs: inputs
                    .iter()
                    .map(|child| self.substitute_factors(*child, replacements, memo))
                    .collect::<Result<Box<_>, _>>()?,
                axis,
                output_type,
            },
        };
        let result = self.push(rebuilt)?;
        memo.insert(root, result);
        Ok(result)
    }
}

fn substitute_factor_identity(
    value: &FactorIdentity,
    binder: &BinderKey,
    replacement: &ResolvedIntExpr,
) -> FactorIdentity {
    let mut value = value.clone();
    value.coordinates = value
        .coordinates
        .iter()
        .map(|(owner, coordinate)| {
            (owner.clone(), substitute_resolved_int_expr(coordinate, binder, replacement))
        })
        .collect();
    value.public = value.public.map(|source| substitute_atomic_source(source, binder, replacement));
    value.selector = value
        .selector
        .map(|selector| Box::new(substitute_factor_identity(&selector, binder, replacement)));
    value.trapdoor =
        value.trapdoor.map(|source| substitute_trapdoor_source(source, binder, replacement));
    value.owner = match value.owner {
        FactorOwner::Derived { parent, tag } => FactorOwner::Derived {
            parent: Box::new(substitute_factor_identity(&parent, binder, replacement)),
            tag,
        },
        FactorOwner::Scalar(identity) => {
            FactorOwner::Scalar(substitute_resolved_int_expr(&identity, binder, replacement))
        }
        owner => owner,
    };
    value
}

fn substitute_atomic_source(
    source: AtomicSourceKey,
    _binder: &BinderKey,
    _replacement: &ResolvedIntExpr,
) -> AtomicSourceKey {
    // Runtime coordinates are carried by FactorIdentity.  Source descriptors
    // are owner identities and must not be inferred or rewritten by position.
    source
}

fn substitute_trapdoor_source(
    source: TrapdoorSourceKey,
    _binder: &BinderKey,
    _replacement: &ResolvedIntExpr,
) -> TrapdoorSourceKey {
    source
}

fn substitute_nf(
    value: &PolynomialNF,
    binder: &BinderKey,
    replacement: &ResolvedIntExpr,
) -> PolynomialNF {
    let exact_terms = value
        .exact_terms
        .values()
        .map(|term| {
            let monomial =
                Monomial::from_factors(term.monomial.factors.iter().cloned().map(|mut factor| {
                    factor.key = substitute_factor_identity(&factor.key, binder, replacement);
                    factor
                }));
            (monomial.key(), SignedMonomial { monomial, multiplicity: term.multiplicity.clone() })
        })
        .collect();
    PolynomialNF::from_parts(exact_terms, value.bounded_summary.clone())
}

fn substitute_slice_spec(
    spec: SliceSpec,
    binder: &BinderKey,
    replacement: &ResolvedIntExpr,
) -> SliceSpec {
    let map_range = |range: ResolvedIndexRange| ResolvedIndexRange {
        start: substitute_resolved_int_expr(&range.start, binder, replacement),
        end: substitute_resolved_int_expr(&range.end, binder, replacement),
    };
    SliceSpec { rows: spec.rows.map(map_range), columns: spec.columns.map(map_range) }
}

fn substitute_crt_spec(
    spec: CrtSpec,
    binder: &BinderKey,
    replacement: &ResolvedIntExpr,
) -> CrtSpec {
    CrtSpec {
        plaintext_moduli: spec
            .plaintext_moduli
            .iter()
            .map(|value| substitute_resolved_int_expr(value, binder, replacement))
            .collect(),
        reconstruction_coefficients: spec
            .reconstruction_coefficients
            .iter()
            .map(|value| substitute_resolved_int_expr(value, binder, replacement))
            .collect(),
    }
}

fn substitute_scale_scalar(
    scalar: ScaleScalar,
    binder: &BinderKey,
    replacement: &ResolvedIntExpr,
) -> ScaleScalar {
    match scalar {
        ScaleScalar::Exact { key, value, matrix_type } => ScaleScalar::Exact {
            key: substitute_factor_identity(&key, binder, replacement),
            value,
            matrix_type,
        },
        ScaleScalar::Interval(interval) => ScaleScalar::Interval(interval),
    }
}

/// Proof-only extraction caps are facts, not value identity.  Keep the
/// structural scale domain while removing that analysis annotation before it
/// enters the matrix identity interner.
fn identity_interval(interval: &IntegerInterval) -> IntegerInterval {
    let mut identity = interval.clone();
    identity.direct_extract_upper = None;
    identity
}

fn scale_identity_scalar(scalar: &ScaleScalar) -> ScaleScalar {
    match scalar {
        ScaleScalar::Exact { key, value, matrix_type } => ScaleScalar::Exact {
            key: key.clone(),
            value: value.clone(),
            matrix_type: matrix_type.clone(),
        },
        ScaleScalar::Interval(interval) => ScaleScalar::Interval(identity_interval(interval)),
    }
}

impl ExpressionNode {
    fn operation_identity_tag(&self) -> MatrixValueOperation {
        match self {
            Self::Zero => MatrixValueOperation::Zero,
            Self::Atom(_) => MatrixValueOperation::Atom,
            Self::Add(_) => MatrixValueOperation::Add,
            Self::Negate(_) => MatrixValueOperation::Negate,
            Self::Product(_) => MatrixValueOperation::Product,
            Self::Switch { reachable, .. } => {
                MatrixValueOperation::Switch { reachable: reachable.clone() }
            }
            Self::Select { reachable, .. } => {
                MatrixValueOperation::Select { reachable: reachable.clone() }
            }
            Self::FamilyGetStatic { index, .. } => {
                MatrixValueOperation::FamilyGetStatic { index: *index }
            }
            Self::FamilyGetDynamic { stored_indices, domain_upper, .. } => {
                MatrixValueOperation::FamilyGetDynamic {
                    stored_indices: stored_indices.clone(),
                    domain_upper: domain_upper.clone(),
                }
            }
            Self::MatrixScale { scalar, .. } => {
                MatrixValueOperation::MatrixScale { scalar: scale_identity_scalar(scalar) }
            }
            Self::Transpose(_) => MatrixValueOperation::Transpose,
            Self::Slice { spec, .. } => MatrixValueOperation::Slice { spec: spec.clone() },
            Self::Tensor { .. } => MatrixValueOperation::Tensor,
            Self::LiftConstantPolynomial { matrix_type, domain, .. } => {
                MatrixValueOperation::LiftConstantPolynomial {
                    matrix_type: matrix_type.clone(),
                    domain: identity_interval(domain),
                }
            }
            Self::View { view, output_type, .. } => {
                MatrixValueOperation::View { view: view.clone(), output_type: output_type.clone() }
            }
            Self::CrtRecompose { spec, output_type, .. } => MatrixValueOperation::CrtRecompose {
                spec: spec.clone(),
                output_type: output_type.clone(),
            },
            Self::Concat { axis, output_type, .. } => {
                MatrixValueOperation::Concat { axis: *axis, output_type: output_type.clone() }
            }
        }
    }

    fn explicit_matrix_type(&self) -> Option<&ConcreteMatrixType> {
        match self {
            Self::Atom(factor) => factor.matrix_bound.as_ref().map(|bound| &bound.matrix_type),
            Self::LiftConstantPolynomial { matrix_type, .. } |
            Self::View { output_type: matrix_type, .. } |
            Self::CrtRecompose { output_type: matrix_type, .. } |
            Self::Concat { output_type: matrix_type, .. } => Some(matrix_type),
            _ => None,
        }
    }

    fn reachable_indices(&self) -> &[usize] {
        match self {
            Self::Switch { reachable, .. } | Self::Select { reachable, .. } => reachable,
            _ => &[],
        }
    }

    fn children(&self) -> Vec<TermId> {
        match self {
            Self::Zero | Self::Atom(_) => Vec::new(),
            Self::Add(children) | Self::Product(children) => children.to_vec(),
            Self::Negate(child) => vec![*child],
            Self::Switch { cases, .. } |
            Self::Select { cases, .. } |
            Self::FamilyGetStatic { cases, .. } |
            Self::FamilyGetDynamic { cases, .. } => cases.to_vec(),
            Self::MatrixScale { input, .. } |
            Self::Transpose(input) |
            Self::Slice { input, .. } |
            Self::LiftConstantPolynomial { input, .. } |
            Self::View { input, .. } => vec![*input],
            Self::Tensor { left, right } => vec![*left, *right],
            Self::CrtRecompose { inputs, .. } | Self::Concat { inputs, .. } => inputs.to_vec(),
        }
    }
}

fn merge_reachable_metadata(inputs: &[&MatrixValueFacts], reachable: &[usize]) -> MatrixMetadata {
    let mut selected = reachable.iter().filter_map(|index| inputs.get(*index));
    let Some(first) = selected.next() else { return MatrixMetadata::unknown() };
    let mut metadata = first.metadata.clone();
    for fact in selected {
        metadata.canonical_coefficient_exclusive_upper = metadata
            .canonical_coefficient_exclusive_upper
            .take()
            .zip(fact.metadata.canonical_coefficient_exclusive_upper.clone())
            .map(|(left, right)| left.max(right));
        metadata.is_constant_polynomial &= fact.metadata.is_constant_polynomial;
        metadata.known_zero_rows = metadata
            .known_zero_rows
            .take()
            .zip(fact.metadata.known_zero_rows.clone())
            .and_then(|(left, right)| (left == right).then_some(left));
    }
    metadata
}

fn bound_kind(bound: &BoundClass) -> &'static str {
    match bound {
        BoundClass::ExactZero => "zero",
        BoundClass::Bounded { .. } => "bounded",
        BoundClass::Large => "large",
    }
}

fn factor_structural_fingerprint(factor: &SymbolicFactor) -> String {
    let shape = factor.matrix_bound.as_ref().map(|bound| {
        // The coefficient cap is intentionally absent: it is a bound, not a
        // symbolic identity. Matrix shape and proof metadata remain part of
        // the resolved factor contract.
        &bound.matrix_type
    });
    let switch = factor
        .switch
        .as_ref()
        .map(|data| (&data.selector, &data.case_indices, &data.case_fingerprints));
    format!(
        "factor(key={:?},kind={},live={},trapdoor={:?},shape={:?},switch={:?})",
        factor.key,
        bound_kind(&factor.bound),
        factor.relation_live,
        factor.trapdoor,
        shape,
        switch
    )
}

/// Fingerprint the owner-resolved DAG structure without retaining the DAG or
/// using its numeric IDs.  In particular, bounded atoms retain their complete
/// factor provenance while their numeric coefficient caps do not participate.
fn dag_structure_fingerprint(
    dag: &ExpressionDag,
    id: TermId,
    visiting: &mut BTreeSet<TermId>,
) -> Result<String, NormalFormError> {
    if !visiting.insert(id) {
        return Err(NormalFormError::CyclicExpression { term: id });
    }
    let node = dag.node(id)?.clone();
    let result = match node {
        ExpressionNode::Zero => "zero".to_owned(),
        ExpressionNode::Atom(factor) => factor_structural_fingerprint(&factor),
        ExpressionNode::Add(children) => {
            let children = children
                .iter()
                .map(|child| dag_structure_fingerprint(dag, *child, visiting))
                .collect::<Result<Vec<_>, _>>()?;
            format!("add({children:?})")
        }
        ExpressionNode::Negate(child) => {
            format!("neg({})", dag_structure_fingerprint(dag, child, visiting)?)
        }
        ExpressionNode::Product(children) => {
            let children = children
                .iter()
                .map(|child| dag_structure_fingerprint(dag, *child, visiting))
                .collect::<Result<Vec<_>, _>>()?;
            format!("product({children:?})")
        }
        ExpressionNode::Switch { selector, cases, reachable } |
        ExpressionNode::Select { selector, cases, reachable } => {
            let cases = reachable
                .iter()
                .map(|index| dag_structure_fingerprint(dag, cases[*index], visiting))
                .collect::<Result<Vec<_>, _>>()?;
            format!("select(selector={selector:?},indices={reachable:?},cases={cases:?})")
        }
        ExpressionNode::FamilyGetStatic { cases, index } => {
            let cases = cases
                .iter()
                .map(|child| dag_structure_fingerprint(dag, *child, visiting))
                .collect::<Result<Vec<_>, _>>()?;
            format!("family-static(index={index},cases={cases:?})")
        }
        ExpressionNode::FamilyGetDynamic { selector, cases, stored_indices, domain_upper } => {
            let cases = cases
                .iter()
                .map(|child| dag_structure_fingerprint(dag, *child, visiting))
                .collect::<Result<Vec<_>, _>>()?;
            format!(
                "family-dynamic(selector={selector:?},indices={stored_indices:?},domain={domain_upper:?},cases={cases:?})"
            )
        }
        ExpressionNode::MatrixScale { input, scalar } => format!(
            "scale(scalar={scalar:?},input={})",
            dag_structure_fingerprint(dag, input, visiting)?
        ),
        ExpressionNode::Transpose(input) => {
            format!("transpose({})", dag_structure_fingerprint(dag, input, visiting)?)
        }
        ExpressionNode::Slice { input, spec } => format!(
            "slice(spec={spec:?},input={})",
            dag_structure_fingerprint(dag, input, visiting)?
        ),
        ExpressionNode::Tensor { left, right } => format!(
            "tensor(left={},right={})",
            dag_structure_fingerprint(dag, left, visiting)?,
            dag_structure_fingerprint(dag, right, visiting)?
        ),
        ExpressionNode::LiftConstantPolynomial { input, matrix_type, domain } => format!(
            "lift(type={matrix_type:?},domain={domain:?},input={})",
            dag_structure_fingerprint(dag, input, visiting)?
        ),
        ExpressionNode::View { input, view, output_type } => format!(
            "view(type={output_type:?},view={view:?},input={})",
            dag_structure_fingerprint(dag, input, visiting)?
        ),
        ExpressionNode::CrtRecompose { inputs, spec, output_type } => format!(
            "crt(type={output_type:?},spec={spec:?},inputs={:?})",
            inputs
                .iter()
                .map(|child| dag_structure_fingerprint(dag, *child, visiting))
                .collect::<Result<Vec<_>, _>>()?
        ),
        ExpressionNode::Concat { inputs, axis, output_type } => format!(
            "concat(axis={axis:?},type={output_type:?},inputs={:?})",
            inputs
                .iter()
                .map(|child| dag_structure_fingerprint(dag, *child, visiting))
                .collect::<Result<Vec<_>, _>>()?
        ),
    };
    visiting.remove(&id);
    Ok(result)
}

pub(crate) fn add_summary(
    left: BoundedSummary,
    right: BoundedSummary,
) -> Result<BoundedSummary, NormalFormError> {
    match (left, right) {
        (BoundedSummary::ExactZero, value) | (value, BoundedSummary::ExactZero) => Ok(value),
        (BoundedSummary::Bounded(left), BoundedSummary::Bounded(right)) => {
            if left.matrix_type != right.matrix_type {
                return Err(NormalFormError::IncompatibleMatrixAddition {
                    left: left.matrix_type,
                    right: right.matrix_type,
                });
            }
            let class = match (left.coefficient_class, right.coefficient_class) {
                (BoundClass::ExactZero, value) | (value, BoundClass::ExactZero) => value,
                (
                    BoundClass::Bounded { maximum_absolute_coefficient: left },
                    BoundClass::Bounded { maximum_absolute_coefficient: right },
                ) => BoundClass::bounded(left + right),
                _ => return Err(NormalFormError::LargeBoundCannotBeSummarized),
            };
            Ok(BoundedSummary::Bounded(MatrixBound {
                matrix_type: left.matrix_type,
                coefficient_class: class,
            }))
        }
    }
}

pub(crate) fn summary_from_bound(bound: MatrixBound) -> BoundedSummary {
    if matches!(bound.coefficient_class, BoundClass::ExactZero) {
        BoundedSummary::ExactZero
    } else {
        BoundedSummary::Bounded(bound)
    }
}

pub(crate) fn monomial_bound(monomial: &Monomial) -> Result<MatrixBound, NormalFormError> {
    let Some(first) = monomial.factors.first() else { return Err(NormalFormError::EmptyMonomial) };
    let Some(mut result) = first.matrix_bound.clone() else {
        return Err(NormalFormError::MissingMatrixBound);
    };
    let mut left_metadata = first.matrix_value_metadata.clone();
    for factor in &monomial.factors[1..] {
        let Some(next) = factor.matrix_bound.as_ref() else {
            return Err(NormalFormError::MissingMatrixBound);
        };
        result = product_bound_with_facts(
            &result,
            next,
            &MatrixProductFacts {
                left_is_constant_polynomial: left_metadata.is_constant_polynomial,
                right_is_constant_polynomial: factor.matrix_value_metadata.is_constant_polynomial,
                right_known_zero_rows: factor.matrix_value_metadata.known_zero_rows.clone(),
            },
        )
        .map_err(NormalFormError::bound)?;
        left_metadata = MatrixMetadata {
            canonical_coefficient_exclusive_upper: None,
            is_constant_polynomial: left_metadata.is_constant_polynomial &&
                factor.matrix_value_metadata.is_constant_polynomial,
            known_zero_rows: None,
        };
    }
    Ok(result)
}

fn polynomial_bound(value: &PolynomialNF) -> Result<Option<MatrixBound>, NormalFormError> {
    if value.first_large_witness().is_some() {
        return Ok(None);
    }
    let mut summary = value.bounded_summary.clone();
    for term in value.exact_terms.values() {
        summary = add_summary(
            summary,
            summary_from_bound(scale_by_multiplicity(
                monomial_bound(&term.monomial)?,
                &term.multiplicity,
            )),
        )?;
    }
    Ok(summary.as_matrix_bound().cloned())
}

fn switch_normalize(
    selector: FactorIdentity,
    cases: Box<[PolynomialNF]>,
    case_indices: Box<[BigUint]>,
    case_fingerprints: Box<[Box<str>]>,
) -> Result<PolynomialNF, NormalFormError> {
    if cases.is_empty() ||
        cases.len() != case_indices.len() ||
        cases.len() != case_fingerprints.len()
    {
        return Err(NormalFormError::InvalidSwitchReachability);
    }
    if cases
        .iter()
        .zip(case_fingerprints.iter())
        .skip(1)
        .all(|(case, fingerprint)| case == &cases[0] && fingerprint == &case_fingerprints[0])
    {
        return Ok(cases[0].clone());
    }

    // Hoist exact additive terms common to every reachable case. Numeric
    // bounded summaries deliberately never participate in this comparison.
    let mut common = PolynomialNF::zero();
    for (key, term) in &cases[0].exact_terms {
        if cases.iter().skip(1).all(|case| {
            case.exact_terms.get(key).is_some_and(|candidate| {
                candidate.multiplicity == term.multiplicity && candidate.monomial == term.monomial
            })
        }) {
            common.insert(term.monomial.clone(), term.multiplicity.clone())?;
        }
    }
    let residuals = cases
        .iter()
        .map(|case| case.clone().subtract(common.clone()))
        .collect::<Result<Vec<_>, _>>()?;

    let same_structure =
        case_fingerprints.iter().all(|fingerprint| fingerprint == &case_fingerprints[0]);
    if same_structure && residuals.iter().all(|case| case.exact_terms.is_empty()) {
        let mut maximum = BoundedSummary::ExactZero;
        for case in &residuals {
            if let Some(bound) = polynomial_bound(case)? {
                maximum = maximum_bound(maximum, bound)?;
            }
        }
        return common.add(match maximum {
            BoundedSummary::ExactZero => PolynomialNF::zero(),
            BoundedSummary::Bounded(bound) => PolynomialNF::bounded(bound)?,
        });
    }

    // Hoist common ordered prefix/suffix only for one symbolic monomial (or
    // exact zero) per case. This never compares bounded summary values.
    if residuals.iter().all(|case| {
        case.bounded_summary.is_exact_zero() &&
            case.exact_terms.len() == 1 &&
            case.exact_terms
                .values()
                .next()
                .is_some_and(|term| term.multiplicity == BigInt::from(1))
    }) {
        let monomials = residuals
            .iter()
            .map(|case| case.exact_terms.values().next().unwrap().monomial.clone())
            .collect::<Vec<_>>();
        let prefix_len = (0..monomials[0].factors.len())
            .take_while(|index| {
                monomials.iter().all(|monomial| {
                    monomial.factors[*index].key == monomials[0].factors[*index].key
                })
            })
            .count();
        let suffix_len = (0..monomials[0].factors.len().saturating_sub(prefix_len))
            .take_while(|offset| {
                let first = &monomials[0].factors[monomials[0].factors.len() - 1 - offset].key;
                monomials.iter().all(|monomial| {
                    monomial.factors[monomial.factors.len() - 1 - offset].key == *first
                })
            })
            .count();
        if prefix_len != 0 || suffix_len != 0 {
            let prefix = Monomial {
                factors: monomials[0].factors[..prefix_len].to_vec().into_boxed_slice(),
            };
            let suffix_start = monomials[0].factors.len() - suffix_len;
            let suffix = Monomial {
                factors: monomials[0].factors[suffix_start..].to_vec().into_boxed_slice(),
            };
            let middle = monomials
                .iter()
                .map(|monomial| {
                    PolynomialNF::from_monomial(Monomial {
                        factors: monomial.factors[prefix_len..suffix_start]
                            .to_vec()
                            .into_boxed_slice(),
                    })
                })
                .collect::<Vec<_>>();
            let inner = switch_normalize(
                selector,
                middle.into_boxed_slice(),
                case_indices.clone(),
                case_fingerprints.clone(),
            )?;
            let mut output = PolynomialNF::one();
            output = normal_form_product::product(output, PolynomialNF::from_monomial(prefix))?;
            output = normal_form_product::product(output, inner)?;
            output = normal_form_product::product(output, PolynomialNF::from_monomial(suffix))?;
            return common.add(output);
        }
    }

    let data =
        Arc::new(SwitchData { selector: selector.clone(), cases, case_indices, case_fingerprints });
    let key = FactorIdentity::switch_barrier(
        selector,
        data.case_indices.clone(),
        switch_fingerprint(&data.case_indices, &data.case_fingerprints),
    );
    let (bound, matrix_bound) = switch_bound(&data.cases)?;
    common.add(PolynomialNF::from_monomial(Monomial::from_factor(SymbolicFactor::switch(
        key,
        bound,
        matrix_bound,
        data,
    ))))
}

fn switch_fingerprint(indices: &[BigUint], cases: &[Box<str>]) -> Box<[u8]> {
    let mut fingerprint = Vec::new();
    for index in indices {
        let bytes = index.to_bytes_be();
        fingerprint.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
        fingerprint.extend_from_slice(&bytes);
    }
    for case in cases {
        let bytes = case.as_bytes();
        fingerprint.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
        fingerprint.extend_from_slice(bytes);
    }
    fingerprint.into_boxed_slice()
}

fn maximum_bound(
    current: BoundedSummary,
    candidate: MatrixBound,
) -> Result<BoundedSummary, NormalFormError> {
    match current {
        BoundedSummary::ExactZero => Ok(summary_from_bound(candidate)),
        BoundedSummary::Bounded(existing) => {
            if existing.matrix_type != candidate.matrix_type {
                return Err(NormalFormError::IncompatibleMatrixAddition {
                    left: existing.matrix_type,
                    right: candidate.matrix_type,
                });
            }
            let coefficient_class = match (existing.coefficient_class, candidate.coefficient_class)
            {
                (BoundClass::ExactZero, value) | (value, BoundClass::ExactZero) => value,
                (
                    BoundClass::Bounded { maximum_absolute_coefficient: left },
                    BoundClass::Bounded { maximum_absolute_coefficient: right },
                ) => BoundClass::bounded(left.max(right)),
                _ => BoundClass::Large,
            };
            Ok(summary_from_bound(MatrixBound {
                matrix_type: existing.matrix_type,
                coefficient_class,
            }))
        }
    }
}

fn switch_bound(
    cases: &[PolynomialNF],
) -> Result<(BoundClass, Option<MatrixBound>), NormalFormError> {
    let mut maximum = BoundedSummary::ExactZero;
    for case in cases {
        if case.first_large_witness().is_some() {
            return Ok((BoundClass::Large, None));
        }
        if let Some(bound) = polynomial_bound(case)? {
            maximum = maximum_bound(maximum, bound)?;
        }
    }
    Ok(match maximum {
        BoundedSummary::ExactZero => (BoundClass::ExactZero, None),
        BoundedSummary::Bounded(bound) => (bound.coefficient_class.clone(), Some(bound)),
    })
}

fn combine_same_selector_switches(
    monomial: &Monomial,
) -> Result<Option<PolynomialNF>, NormalFormError> {
    let switches =
        monomial.factors.iter().filter_map(|factor| factor.switch.clone()).collect::<Vec<_>>();
    if switches.len() < 2 {
        return Ok(None);
    }
    let selector = switches[0].selector.clone();
    if switches.iter().any(|switch| switch.selector != selector) {
        return Ok(None);
    }
    if switches.iter().any(|switch| switch.cases.len() != switches[0].cases.len()) {
        return Err(NormalFormError::AmbiguousSwitchMapping);
    }
    if switches.iter().any(|switch| switch.case_indices != switches[0].case_indices) {
        return Err(NormalFormError::AmbiguousSwitchMapping);
    }
    let mut cases = Vec::with_capacity(switches[0].cases.len());
    let mut case_fingerprints = Vec::with_capacity(switches[0].cases.len());
    for index in 0..switches[0].cases.len() {
        let mut case = PolynomialNF::one();
        let mut fingerprint_parts = Vec::with_capacity(monomial.factors.len());
        for factor in monomial.factors.iter() {
            if let Some(switch) = &factor.switch {
                case = normal_form_product::product(case, switch.cases[index].clone())?;
                fingerprint_parts.push(switch.case_fingerprints[index].to_string());
            } else {
                case = normal_form_product::product(
                    case,
                    PolynomialNF::from_monomial(Monomial::from_factor(factor.clone())),
                )?;
                fingerprint_parts.push(factor_structural_fingerprint(factor));
            }
        }
        cases.push(case);
        case_fingerprints.push(format!("product({fingerprint_parts:?}").into_boxed_str());
    }
    Ok(Some(switch_normalize(
        selector,
        cases.into_boxed_slice(),
        switches[0].case_indices.clone(),
        case_fingerprints.into_iter().collect(),
    )?))
}

pub(crate) fn scale_by_multiplicity(mut bound: MatrixBound, multiplicity: &BigInt) -> MatrixBound {
    let absolute = multiplicity.magnitude().clone();
    if let BoundClass::Bounded { maximum_absolute_coefficient } = &mut bound.coefficient_class {
        *maximum_absolute_coefficient *= absolute;
    }
    bound
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NormalFormError {
    InvalidTermId,
    TooManyTerms,
    EmptyMonomial,
    LargeFactorCannotBeBounded,
    LargeBoundCannotBeSummarized,
    RelationLiveRequiresFiniteBound,
    MissingMatrixBound,
    BoundedSummaryMixedWithLarge,
    ConflictingFactorContracts { key: MonomialKey },
    IncompatibleMatrixAddition { left: ConcreteMatrixType, right: ConcreteMatrixType },
    IncompatibleMatrixProduct { left: ConcreteMatrixType, right: ConcreteMatrixType },
    InvalidKnownZeroRows { known_zero_rows: BigUint, row_count: BigUint },
    BoundArithmetic,
    AmbiguousRelation { keys: Vec<FactorIdentity> },
    ConflictingRelationTarget { key: FullRelationKey },
    CyclicExpression { term: TermId },
    CyclicRelationDependency { key: FullRelationKey },
    UnconsumedRelationLive { key: MonomialKey },
    UnconsumedExactTerm { key: MonomialKey },
    InvalidSwitchReachability,
    AmbiguousSwitchMapping,
    InvalidFamilyIndex,
    InvalidFamilyDomain,
}

impl NormalFormError {
    fn bound(error: super::bound::BoundArithmeticError) -> Self {
        match error {
            super::bound::BoundArithmeticError::IncompatibleMatrixProduct { left, right } => {
                Self::IncompatibleMatrixProduct { left, right }
            }
            super::bound::BoundArithmeticError::InvalidKnownZeroRows {
                known_zero_rows,
                row_count,
            } => Self::InvalidKnownZeroRows { known_zero_rows, row_count },
        }
    }
}

impl fmt::Display for NormalFormError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}
impl std::error::Error for NormalFormError {}

#[cfg(test)]
mod tests {
    use super::*;
    fn bound(value: u64) -> MatrixBound {
        MatrixBound {
            matrix_type: ConcreteMatrixType {
                modulus: 17.into(),
                ring_dimension: 1,
                rows: 1,
                columns: 1,
            },
            coefficient_class: BoundClass::bounded(value.into()),
        }
    }

    fn rectangular_bound(
        value: u64,
        rows: usize,
        columns: usize,
        ring_dimension: usize,
    ) -> MatrixBound {
        MatrixBound {
            matrix_type: ConcreteMatrixType { modulus: 17.into(), ring_dimension, rows, columns },
            coefficient_class: BoundClass::bounded(value.into()),
        }
    }
    #[test]
    fn zero_annihilates_large() {
        assert!(
            normal_form_product::product(
                PolynomialNF::zero(),
                PolynomialNF::exact_factor(FactorIdentity::named("L")),
            )
            .unwrap()
            .is_exact_zero()
        );
    }
    #[test]
    fn signs_cancel_and_order_is_preserved() {
        let a = PolynomialNF::exact_factor(FactorIdentity::named("A"));
        let b = PolynomialNF::exact_factor(FactorIdentity::named("B"));
        assert!(a.clone().subtract(a.clone().negate().negate()).unwrap().is_exact_zero());
        assert_ne!(
            normal_form_product::product(a.clone(), b.clone()).unwrap(),
            normal_form_product::product(b, a).unwrap()
        );
    }
    #[test]
    fn bounded_terms_fold() {
        let x = PolynomialNF::bounded(bound(2))
            .unwrap()
            .add(PolynomialNF::bounded(bound(3)).unwrap())
            .unwrap();
        assert_eq!(
            x.bounded_summary().as_matrix_bound().unwrap().coefficient_class,
            BoundClass::bounded(5_u8.into())
        );
    }

    #[test]
    fn product_retains_all_bounded_cross_terms_around_live_factors() {
        let left = PolynomialNF::relation_live_factor(FactorIdentity::named("K"), bound(1))
            .unwrap()
            .add(PolynomialNF::bounded(bound(2)).unwrap())
            .unwrap();
        let right = PolynomialNF::relation_live_factor(FactorIdentity::named("K2"), bound(1))
            .unwrap()
            .add(PolynomialNF::bounded(bound(3)).unwrap())
            .unwrap();
        let product = normal_form_product::product_bound_only(left, right).unwrap();
        assert_eq!(product.exact_terms().len(), 1);
        assert_eq!(
            product.bounded_summary().as_matrix_bound().unwrap().coefficient_class,
            BoundClass::bounded(11_u8.into())
        );
    }
    #[test]
    fn dag_term_ids_are_not_symbolic_keys() {
        let mut dag = ExpressionDag::new();
        let a = dag
            .push(ExpressionNode::Atom(SymbolicFactor::large(FactorIdentity::named("A"))))
            .unwrap();
        let b = dag
            .push(ExpressionNode::Atom(SymbolicFactor::large(FactorIdentity::named("A"))))
            .unwrap();
        let root = dag.push(ExpressionNode::Add(vec![a, b].into_boxed_slice())).unwrap();
        let nf = dag.normalize(root, &RelationRegistry::default()).unwrap();
        assert_eq!(nf.exact_terms().values().next().unwrap().multiplicity, 2.into());
    }

    #[test]
    fn deep_expression_dag_uses_shallow_identity_arena_and_reuses_shared_values() {
        let mut dag = ExpressionDag::new();
        let zero = dag.push(ExpressionNode::Zero).unwrap();
        let depth = 4096;
        let first_transpose = dag.push(ExpressionNode::Transpose(zero)).unwrap();
        let mut current = first_transpose;
        for _ in 1..depth {
            current = dag.push(ExpressionNode::Transpose(current)).unwrap();
        }
        let duplicate = dag.push(ExpressionNode::Transpose(zero)).unwrap();

        assert_eq!(dag.term_count(), depth + 2);
        assert_eq!(dag.identity_count(), depth + 1);
        assert_eq!(
            dag.facts(duplicate).unwrap().identity,
            dag.facts(first_transpose).unwrap().identity
        );
    }

    #[test]
    fn relation_rewrites_leftmost_ordered_boundary() {
        let public = FactorIdentity::named("B");
        let preimage = FactorIdentity::named("K");
        let target = FactorIdentity::named("P");
        let mut dag = ExpressionDag::new();
        let b = dag.push(ExpressionNode::Atom(SymbolicFactor::large(public.clone()))).unwrap();
        let k = dag
            .push(ExpressionNode::Atom(
                SymbolicFactor::relation_live(preimage.clone(), bound(1)).unwrap(),
            ))
            .unwrap();
        let p = dag.push(ExpressionNode::Atom(SymbolicFactor::large(target.clone()))).unwrap();
        let root = dag.push(ExpressionNode::Product(vec![b, k].into_boxed_slice())).unwrap();
        let key = FullRelationKey {
            source: "named".into(),
            ordered_indices: Box::new([]),
            public: public.clone(),
            target: target.clone(),
            matrix_type: None,
            layout: None,
            trapdoor: None,
            selector: None,
        };
        let mut registry = RelationRegistry::default();
        registry.register(RelationRegistration { key, preimage, target: p }).unwrap();
        let nf = dag.normalize(root, &registry).unwrap();
        assert_eq!(nf.exact_terms().len(), 1);
        assert_eq!(nf.exact_terms().keys().next().unwrap().factors(), &[target]);
    }

    #[test]
    fn normalization_counters_report_real_dag_and_relation_work() {
        let public = FactorIdentity::named("counter-B");
        let preimage = FactorIdentity::named("counter-K");
        let target = FactorIdentity::named("counter-P");
        let mut dag = ExpressionDag::new();
        let b = dag.push(ExpressionNode::Atom(SymbolicFactor::large(public.clone()))).unwrap();
        let k = dag
            .push(ExpressionNode::Atom(
                SymbolicFactor::relation_live(preimage.clone(), bound(1)).unwrap(),
            ))
            .unwrap();
        let p = dag.push(ExpressionNode::Atom(SymbolicFactor::large(target.clone()))).unwrap();
        let root = dag.push(ExpressionNode::Product(vec![b, k].into_boxed_slice())).unwrap();
        let key = FullRelationKey {
            source: "named".into(),
            ordered_indices: Box::new([]),
            public: public.clone(),
            target: target.clone(),
            matrix_type: None,
            layout: None,
            trapdoor: None,
            selector: None,
        };
        let mut registry = RelationRegistry::default();
        registry.register(RelationRegistration { key, preimage, target: p }).unwrap();

        let (normalized, counters) = dag.normalize_with_counters(root, &registry).unwrap();
        assert_eq!(normalized.exact_terms().len(), 1);
        assert_eq!(counters.nodes_processed, 4);
        assert_eq!(counters.relation_candidates, 1);
        assert_eq!(counters.relations_applied, 1);
        assert_eq!(counters.relations_remaining, 0);
    }

    #[test]
    fn normalization_counters_report_bounded_product_fold() {
        let mut dag = ExpressionDag::new();
        let left = dag
            .push(ExpressionNode::Atom(
                SymbolicFactor::bounded(FactorIdentity::named("fold-left"), bound(2)).unwrap(),
            ))
            .unwrap();
        let right = dag
            .push(ExpressionNode::Atom(
                SymbolicFactor::bounded(FactorIdentity::named("fold-right"), bound(3)).unwrap(),
            ))
            .unwrap();
        let root = dag.push(ExpressionNode::Product(vec![left, right].into_boxed_slice())).unwrap();
        let (normalized, counters) =
            dag.normalize_with_counters(root, &RelationRegistry::default()).unwrap();
        assert!(normalized.validate_bounded_only().is_ok());
        assert_eq!(counters.nodes_processed, 3);
        assert!(counters.bounded_fold_count >= 1);
    }

    #[test]
    fn normalization_counters_report_reachable_switch_cases() {
        let mut dag = ExpressionDag::new();
        let zero = dag.push(ExpressionNode::Zero).unwrap();
        let root = dag
            .push(ExpressionNode::Switch {
                selector: FactorIdentity::named("counter-selector"),
                cases: vec![zero, zero].into_boxed_slice(),
                reachable: vec![0, 1].into_boxed_slice(),
            })
            .unwrap();
        let (normalized, counters) =
            dag.normalize_with_counters(root, &RelationRegistry::default()).unwrap();
        assert!(normalized.is_exact_zero());
        assert_eq!(counters.nodes_processed, 2);
        assert_eq!(counters.switch_cases_processed, 2);
    }

    #[test]
    fn exact_zero_factor_atom_is_zero_and_finite_cross_term_uses_shape_and_ring_factor() {
        let zero = SymbolicFactor::bounded(FactorIdentity::named("Z"), bound(0)).unwrap();
        let large = PolynomialNF::exact_factor(FactorIdentity::named("L"));
        let mut dag = ExpressionDag::new();
        let zero_id = dag.push(ExpressionNode::Atom(zero)).unwrap();
        let large_id = dag
            .push(ExpressionNode::Atom(SymbolicFactor::large(FactorIdentity::named("L"))))
            .unwrap();
        let product = dag.push(ExpressionNode::Product(vec![zero_id, large_id].into())).unwrap();
        assert!(dag.normalize(product, &RelationRegistry::default()).unwrap().is_exact_zero());

        let summary = PolynomialNF::bounded(rectangular_bound(2, 2, 2, 3)).unwrap();
        let finite = PolynomialNF::relation_live_factor(
            FactorIdentity::named("K"),
            rectangular_bound(3, 2, 4, 3),
        )
        .unwrap();
        let folded = normal_form_product::product(summary, finite).unwrap();
        let matrix = folded.bounded_summary().as_matrix_bound().unwrap();
        assert_eq!((matrix.matrix_type.rows, matrix.matrix_type.columns), (2, 4));
        assert_eq!(matrix.coefficient_class, BoundClass::bounded(36_u8.into()));
        assert!(
            normal_form_product::product(large, PolynomialNF::bounded(bound(2)).unwrap()).is_err()
        );
    }

    #[test]
    fn final_finish_folds_unmatched_finite_and_rejects_large_residuals() {
        let finite = PolynomialNF::bounded_factor(FactorIdentity::named("E"), bound(4)).unwrap();
        let finished = finite.finish_relation_live().unwrap();
        assert!(finished.validate_bounded_only().is_ok());
        let live =
            PolynomialNF::relation_live_factor(FactorIdentity::named("K"), bound(1)).unwrap();
        assert!(live.finish_relation_live().unwrap().validate_bounded_only().is_ok());
        let large = PolynomialNF::exact_factor(FactorIdentity::named("L"));
        assert!(matches!(
            large.validate_bounded_only(),
            Err(NormalFormError::UnconsumedExactTerm { .. })
        ));
    }

    #[test]
    fn relation_full_key_mismatch_is_not_applicable_and_conflicts_fail_closed() {
        let public = FactorIdentity::named("B");
        let preimage = FactorIdentity::named("K");
        let target = FactorIdentity::named("P");
        let mut dag = ExpressionDag::new();
        let b = dag.push(ExpressionNode::Atom(SymbolicFactor::large(public.clone()))).unwrap();
        let k = dag
            .push(ExpressionNode::Atom(
                SymbolicFactor::relation_live(preimage.clone(), bound(1)).unwrap(),
            ))
            .unwrap();
        let p = dag.push(ExpressionNode::Atom(SymbolicFactor::large(target.clone()))).unwrap();
        let root = dag.push(ExpressionNode::Product(vec![b, k].into())).unwrap();
        let key = FullRelationKey {
            source: "wrong-owner".into(),
            ordered_indices: Box::new([]),
            public: public.clone(),
            target: target.clone(),
            matrix_type: None,
            layout: None,
            trapdoor: None,
            selector: None,
        };
        let mut registry = RelationRegistry::default();
        registry
            .register(RelationRegistration {
                key: key.clone(),
                preimage: preimage.clone(),
                target: p,
            })
            .unwrap();
        registry
            .register(RelationRegistration {
                key: key.clone(),
                preimage: preimage.clone(),
                target: p,
            })
            .unwrap();
        assert!(matches!(
            dag.normalize_bounded(root, &registry),
            Err(NormalFormError::UnconsumedExactTerm { .. })
        ));
        let conflicting = dag
            .push(ExpressionNode::Atom(SymbolicFactor::large(FactorIdentity::named("Q"))))
            .unwrap();
        assert!(matches!(
            registry.register(RelationRegistration {
                key: key.clone(),
                preimage: preimage.clone(),
                target: conflicting
            }),
            Err(NormalFormError::ConflictingRelationTarget { .. })
        ));
        let correct_key = FullRelationKey { source: "named".into(), ..key.clone() };
        registry
            .register(RelationRegistration {
                key: correct_key.clone(),
                preimage: preimage.clone(),
                target: p,
            })
            .unwrap();
        let alternate_key = FullRelationKey { target: FactorIdentity::named("Q"), ..correct_key };
        registry
            .register(RelationRegistration { key: alternate_key, preimage, target: conflicting })
            .unwrap();
        assert!(matches!(
            dag.normalize(root, &registry),
            Err(NormalFormError::AmbiguousRelation { .. })
        ));
    }

    #[test]
    fn active_relation_key_covers_reconnected_target_fixed_point() {
        let public = FactorIdentity::named("B");
        let preimage = FactorIdentity::named("K");
        let target = FactorIdentity::named("P");
        let mut dag = ExpressionDag::new();
        let b = dag.push(ExpressionNode::Atom(SymbolicFactor::large(public.clone()))).unwrap();
        let k = dag
            .push(ExpressionNode::Atom(
                SymbolicFactor::relation_live(preimage.clone(), bound(1)).unwrap(),
            ))
            .unwrap();
        let target_product = dag.push(ExpressionNode::Product(vec![b, k].into())).unwrap();
        let prefix = dag
            .push(ExpressionNode::Atom(SymbolicFactor::large(FactorIdentity::named("A"))))
            .unwrap();
        let suffix = dag
            .push(ExpressionNode::Atom(SymbolicFactor::large(FactorIdentity::named("D"))))
            .unwrap();
        let root = dag.push(ExpressionNode::Product(vec![prefix, b, k, suffix].into())).unwrap();
        let key = FullRelationKey {
            source: "named".into(),
            ordered_indices: Box::new([]),
            public: public.clone(),
            target,
            matrix_type: None,
            layout: None,
            trapdoor: None,
            selector: None,
        };
        let mut registry = RelationRegistry::default();
        registry.register(RelationRegistration { key, preimage, target: target_product }).unwrap();
        assert!(matches!(
            dag.normalize(root, &registry),
            Err(NormalFormError::CyclicRelationDependency { .. })
        ));
    }

    #[test]
    fn dag_root_normalize_cannot_omit_relation_live_finalization() {
        let mut dag = ExpressionDag::new();
        let live = dag
            .push(ExpressionNode::Atom(
                SymbolicFactor::relation_live(FactorIdentity::named("K"), bound(1)).unwrap(),
            ))
            .unwrap();
        assert!(dag.normalize_bounded(live, &RelationRegistry::default()).is_ok());
    }

    #[test]
    fn relation_matching_uses_actual_trapdoor_provenance() {
        let public = FactorIdentity::named("B");
        let preimage = FactorIdentity::named("K");
        let target = FactorIdentity::named("P");
        let trapdoor = TrapdoorSourceKey::ProtocolInput(crate::ProtocolInputId::from("td"));
        let wrong_trapdoor =
            TrapdoorSourceKey::ProtocolInput(crate::ProtocolInputId::from("wrong-td"));
        let mut dag = ExpressionDag::new();
        let b = dag.push(ExpressionNode::Atom(SymbolicFactor::large(public.clone()))).unwrap();
        let k = dag
            .push(ExpressionNode::Atom(
                SymbolicFactor::relation_live(preimage.clone(), bound(1))
                    .unwrap()
                    .with_trapdoor(wrong_trapdoor),
            ))
            .unwrap();
        let p = dag.push(ExpressionNode::Atom(SymbolicFactor::large(target.clone()))).unwrap();
        let root = dag.push(ExpressionNode::Product(vec![b, k].into())).unwrap();
        let key = FullRelationKey {
            source: "named".into(),
            ordered_indices: Box::new([]),
            public,
            target,
            matrix_type: None,
            layout: None,
            trapdoor: Some(trapdoor.clone()),
            selector: None,
        };
        let mut registry = RelationRegistry::default();
        registry
            .register(RelationRegistration {
                key: key.clone(),
                preimage: preimage.clone(),
                target: p,
            })
            .unwrap();
        assert!(matches!(
            dag.normalize_bounded(root, &registry),
            Err(NormalFormError::UnconsumedExactTerm { .. })
        ));

        let mut matching_dag = ExpressionDag::new();
        let b = matching_dag
            .push(ExpressionNode::Atom(SymbolicFactor::large(key.public.clone())))
            .unwrap();
        let k = matching_dag
            .push(ExpressionNode::Atom(
                SymbolicFactor::relation_live(preimage, bound(1)).unwrap().with_trapdoor(trapdoor),
            ))
            .unwrap();
        let p = matching_dag
            .push(ExpressionNode::Atom(SymbolicFactor::large(key.target.clone())))
            .unwrap();
        let root = matching_dag.push(ExpressionNode::Product(vec![b, k].into())).unwrap();
        let mut matching_registry = RelationRegistry::default();
        matching_registry
            .register(RelationRegistration { key, preimage: FactorIdentity::named("K"), target: p })
            .unwrap();
        assert!(matching_dag.normalize(root, &matching_registry).is_ok());
    }

    #[test]
    fn product_exposes_single_switch_for_case_local_relations() {
        let public = FactorIdentity::named("B");
        let selector = FactorIdentity::named("s");
        let preimages = [FactorIdentity::named("K0"), FactorIdentity::named("K1")];
        let targets = [FactorIdentity::named("P0"), FactorIdentity::named("P1")];
        let extras = [FactorIdentity::named("U0"), FactorIdentity::named("U1")];
        let mut dag = ExpressionDag::new();
        let b = dag.push(ExpressionNode::Atom(SymbolicFactor::large(public.clone()))).unwrap();
        let mut cases = Vec::new();
        let mut target_terms = Vec::new();
        for (preimage, target) in preimages.iter().zip(targets.iter()) {
            cases.push(
                dag.push(ExpressionNode::Atom(
                    SymbolicFactor::relation_live(preimage.clone(), bound(1)).unwrap(),
                ))
                .unwrap(),
            );
            target_terms.push(
                dag.push(ExpressionNode::Atom(SymbolicFactor::large(target.clone()))).unwrap(),
            );
        }
        let switched = dag
            .push(ExpressionNode::Switch {
                selector: selector.clone(),
                cases: cases.into_boxed_slice(),
                reachable: vec![0, 1].into_boxed_slice(),
            })
            .unwrap();
        let extra_cases = extras
            .iter()
            .map(|extra| {
                dag.push(ExpressionNode::Atom(SymbolicFactor::large(extra.clone()))).unwrap()
            })
            .collect::<Vec<_>>();
        let switched_extra = dag
            .push(ExpressionNode::Switch {
                selector,
                cases: extra_cases.into_boxed_slice(),
                reachable: vec![0, 1].into_boxed_slice(),
            })
            .unwrap();
        let root = dag
            .push(ExpressionNode::Product(vec![b, switched, switched_extra].into_boxed_slice()))
            .unwrap();
        let mut registry = RelationRegistry::default();
        for ((preimage, target), target_term) in
            preimages.iter().zip(targets.iter()).zip(target_terms)
        {
            registry
                .register(RelationRegistration {
                    key: FullRelationKey {
                        source: "named".into(),
                        ordered_indices: Box::new([]),
                        public: public.clone(),
                        target: target.clone(),
                        matrix_type: None,
                        layout: None,
                        trapdoor: None,
                        selector: None,
                    },
                    preimage: preimage.clone(),
                    target: target_term,
                })
                .unwrap();
        }
        let normalized = dag.normalize(root, &registry).unwrap();
        let switch = normalized
            .exact_terms()
            .values()
            .flat_map(|term| term.monomial.factors())
            .find_map(|factor| factor.switch.clone())
            .expect("single selector must remain a barrier");
        let case_keys = switch
            .cases
            .iter()
            .map(|case| {
                case.exact_terms()
                    .values()
                    .next()
                    .unwrap()
                    .monomial
                    .factors()
                    .iter()
                    .map(|factor| factor.key.clone())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            case_keys,
            vec![
                vec![targets[0].clone(), extras[0].clone()],
                vec![targets[1].clone(), extras[1].clone()]
            ]
        );
    }

    #[test]
    fn finite_relation_target_is_folded_before_root_returns() {
        let public = FactorIdentity::named("B");
        let preimage = FactorIdentity::named("K");
        let target = FactorIdentity::named("E");
        let mut dag = ExpressionDag::new();
        let b = dag.push(ExpressionNode::Atom(SymbolicFactor::large(public.clone()))).unwrap();
        let k = dag
            .push(ExpressionNode::Atom(
                SymbolicFactor::relation_live(preimage.clone(), bound(1)).unwrap(),
            ))
            .unwrap();
        let e = dag
            .push(ExpressionNode::Atom(SymbolicFactor::bounded(target.clone(), bound(2)).unwrap()))
            .unwrap();
        let root = dag.push(ExpressionNode::Product(vec![b, k].into())).unwrap();
        let key = FullRelationKey {
            source: "named".into(),
            ordered_indices: Box::new([]),
            public,
            target,
            matrix_type: None,
            layout: None,
            trapdoor: None,
            selector: None,
        };
        let mut registry = RelationRegistry::default();
        registry.register(RelationRegistration { key, preimage, target: e }).unwrap();
        let summary = dag.normalize_bounded(root, &registry).unwrap();
        assert_eq!(
            summary.as_matrix_bound().unwrap().coefficient_class,
            BoundClass::bounded(2_u8.into())
        );
    }

    #[test]
    fn switch_does_not_hoist_distinct_noise_with_equal_bounds() {
        let mut dag = ExpressionDag::new();
        let common = dag
            .push(ExpressionNode::Atom(SymbolicFactor::large(FactorIdentity::named("X"))))
            .unwrap();
        let noise = |dag: &mut ExpressionDag, name: &str| {
            dag.push(ExpressionNode::Atom(
                SymbolicFactor::bounded(FactorIdentity::named(name), bound(2)).unwrap(),
            ))
            .unwrap()
        };
        let n0 = noise(&mut dag, "N0");
        let n1 = noise(&mut dag, "N1");
        let n2 = noise(&mut dag, "N2");
        let n3 = noise(&mut dag, "N3");
        let first = dag.push(ExpressionNode::Add(vec![common, n0].into_boxed_slice())).unwrap();
        let second = dag.push(ExpressionNode::Add(vec![common, n1].into_boxed_slice())).unwrap();
        let first_switch = dag
            .push(ExpressionNode::Switch {
                selector: FactorIdentity::named("selector"),
                cases: vec![first, second].into_boxed_slice(),
                reachable: vec![0, 1].into_boxed_slice(),
            })
            .unwrap();
        let third = dag.push(ExpressionNode::Add(vec![common, n2].into_boxed_slice())).unwrap();
        let fourth = dag.push(ExpressionNode::Add(vec![common, n3].into_boxed_slice())).unwrap();
        let second_switch = dag
            .push(ExpressionNode::Switch {
                selector: FactorIdentity::named("selector"),
                cases: vec![third, fourth].into_boxed_slice(),
                reachable: vec![0, 1].into_boxed_slice(),
            })
            .unwrap();
        let left = dag.normalize(first_switch, &RelationRegistry::default()).unwrap();
        let right = dag.normalize(second_switch, &RelationRegistry::default()).unwrap();
        let left_barrier = left
            .exact_terms
            .values()
            .find_map(|term| term.monomial.factors.iter().find_map(|factor| factor.switch.clone()))
            .unwrap();
        let right_barrier = right
            .exact_terms
            .values()
            .find_map(|term| term.monomial.factors.iter().find_map(|factor| factor.switch.clone()))
            .unwrap();
        assert_ne!(left_barrier.case_fingerprints, right_barrier.case_fingerprints);
        let difference = left.add(right.negate()).unwrap();
        assert_eq!(difference.exact_terms.len(), 2);
        assert!(
            difference
                .exact_terms
                .values()
                .all(|term| { term.monomial.factors.iter().any(|factor| factor.switch.is_some()) })
        );
    }

    #[test]
    fn same_selector_is_casewise_for_three_factors_and_different_selectors_are_barriers() {
        let mut dag = ExpressionDag::new();
        let make_switch = |dag: &mut ExpressionDag, selector: FactorIdentity, names: [&str; 2]| {
            let left = dag
                .push(ExpressionNode::Atom(SymbolicFactor::large(FactorIdentity::named(names[0]))))
                .unwrap();
            let right = dag
                .push(ExpressionNode::Atom(SymbolicFactor::large(FactorIdentity::named(names[1]))))
                .unwrap();
            dag.push(ExpressionNode::Switch {
                selector,
                cases: vec![left, right].into_boxed_slice(),
                reachable: vec![0, 1].into_boxed_slice(),
            })
            .unwrap()
        };
        let selector = FactorIdentity::named("s");
        let a = make_switch(&mut dag, selector.clone(), ["A0", "A1"]);
        let b = make_switch(&mut dag, selector.clone(), ["B0", "B1"]);
        let c = make_switch(&mut dag, selector.clone(), ["C0", "C1"]);
        let same = dag.push(ExpressionNode::Product(vec![a, b, c].into_boxed_slice())).unwrap();
        assert_eq!(
            dag.normalize(same, &RelationRegistry::default()).unwrap().exact_terms().len(),
            1
        );
        let same_nf = dag.normalize(same, &RelationRegistry::default()).unwrap();
        let same_data = same_nf
            .exact_terms
            .values()
            .next()
            .unwrap()
            .monomial
            .factors
            .iter()
            .find_map(|factor| factor.switch.clone())
            .unwrap();
        let expected_cases = [["A0", "B0", "C0"], ["A1", "B1", "C1"]];
        for (case, expected) in same_data.cases.iter().zip(expected_cases) {
            let term = case.exact_terms.values().next().unwrap();
            assert_eq!(
                term.monomial.factors.iter().map(|factor| factor.key.clone()).collect::<Vec<_>>(),
                expected.into_iter().map(FactorIdentity::named).collect::<Vec<_>>()
            );
        }
        let other = make_switch(&mut dag, FactorIdentity::named("t"), ["T0", "T1"]);
        let different =
            dag.push(ExpressionNode::Product(vec![a, other].into_boxed_slice())).unwrap();
        let different_nf = dag.normalize(different, &RelationRegistry::default()).unwrap();
        assert_eq!(different_nf.exact_terms.len(), 1);
        let different_term = different_nf.exact_terms.values().next().unwrap();
        assert_eq!(different_term.monomial.factors.len(), 2);
        assert_ne!(
            different_term.monomial.factors[0].switch.as_ref().unwrap().selector,
            different_term.monomial.factors[1].switch.as_ref().unwrap().selector
        );

        let reversed = make_switch(&mut dag, FactorIdentity::named("s"), ["R0", "R1"]);
        let reversed_node = match dag.node(reversed).unwrap() {
            ExpressionNode::Switch { selector, cases, .. } => dag
                .push(ExpressionNode::Switch {
                    selector: selector.clone(),
                    cases: cases.clone(),
                    reachable: vec![1, 0].into_boxed_slice(),
                })
                .unwrap(),
            _ => unreachable!(),
        };
        let mismatched = dag.push(ExpressionNode::Product(vec![a, reversed_node].into())).unwrap();
        assert!(matches!(
            dag.normalize(mismatched, &RelationRegistry::default()),
            Err(NormalFormError::AmbiguousSwitchMapping)
        ));
    }

    #[test]
    fn scalar_selector_identity_is_shared_by_separate_matrix_select_nodes() {
        let mut dag = ExpressionDag::new();
        let make_select = |dag: &mut ExpressionDag, selector: FactorIdentity| {
            let left = dag
                .push(ExpressionNode::Atom(SymbolicFactor::large(FactorIdentity::named("L"))))
                .unwrap();
            let right = dag
                .push(ExpressionNode::Atom(SymbolicFactor::large(FactorIdentity::named("R"))))
                .unwrap();
            dag.push(ExpressionNode::Select {
                selector,
                cases: vec![left, right].into_boxed_slice(),
                reachable: vec![0, 1].into_boxed_slice(),
            })
            .unwrap()
        };
        let selector =
            FactorIdentity::scalar_selector(ResolvedIntExpr::Parameter("selector".to_owned()));
        let first = make_select(&mut dag, selector.clone());
        let second = make_select(&mut dag, selector);
        let cancellation =
            dag.push(ExpressionNode::Product(vec![first, second].into_boxed_slice())).unwrap();
        let normalized = dag.normalize(cancellation, &RelationRegistry::default()).unwrap();
        assert_eq!(normalized.exact_terms().len(), 1);
        assert!(
            normalized
                .exact_terms()
                .values()
                .next()
                .unwrap()
                .monomial
                .factors
                .iter()
                .any(|factor| factor.switch.is_some())
        );

        let first = make_select(
            &mut dag,
            FactorIdentity::scalar_selector(ResolvedIntExpr::Parameter("selector".to_owned())),
        );
        let distinct = make_select(
            &mut dag,
            FactorIdentity::scalar_selector(ResolvedIntExpr::Parameter(
                "distinct-selector".to_owned(),
            )),
        );
        let noncollision =
            dag.push(ExpressionNode::Product(vec![first, distinct].into_boxed_slice())).unwrap();
        let normalized = dag.normalize(noncollision, &RelationRegistry::default()).unwrap();
        assert_eq!(normalized.exact_terms().len(), 1);
        let term = normalized.exact_terms().values().next().unwrap();
        assert_eq!(term.monomial.factors.len(), 2);
        assert_ne!(
            term.monomial.factors[0].switch.as_ref().unwrap().selector,
            term.monomial.factors[1].switch.as_ref().unwrap().selector
        );
    }

    #[test]
    fn switch_hoists_common_additive_and_ordered_prefix_suffix() {
        let mut dag = ExpressionDag::new();
        let common = dag
            .push(ExpressionNode::Atom(SymbolicFactor::large(FactorIdentity::named("A"))))
            .unwrap();
        let x0 = dag
            .push(ExpressionNode::Atom(SymbolicFactor::large(FactorIdentity::named("X0"))))
            .unwrap();
        let x1 = dag
            .push(ExpressionNode::Atom(SymbolicFactor::large(FactorIdentity::named("X1"))))
            .unwrap();
        let add0 = dag.push(ExpressionNode::Add(vec![common, x0].into_boxed_slice())).unwrap();
        let add1 = dag.push(ExpressionNode::Add(vec![common, x1].into_boxed_slice())).unwrap();
        let sum_switch = dag
            .push(ExpressionNode::Switch {
                selector: FactorIdentity::named("s"),
                cases: vec![add0, add1].into_boxed_slice(),
                reachable: vec![0, 1].into_boxed_slice(),
            })
            .unwrap();
        assert_eq!(
            dag.normalize(sum_switch, &RelationRegistry::default()).unwrap().exact_terms().len(),
            2
        );
        let prefix = dag
            .push(ExpressionNode::Atom(SymbolicFactor::large(FactorIdentity::named("P"))))
            .unwrap();
        let suffix = dag
            .push(ExpressionNode::Atom(SymbolicFactor::large(FactorIdentity::named("D"))))
            .unwrap();
        let left =
            dag.push(ExpressionNode::Product(vec![prefix, x0, suffix].into_boxed_slice())).unwrap();
        let right =
            dag.push(ExpressionNode::Product(vec![prefix, x1, suffix].into_boxed_slice())).unwrap();
        let product_switch = dag
            .push(ExpressionNode::Switch {
                selector: FactorIdentity::named("s"),
                cases: vec![left, right].into_boxed_slice(),
                reachable: vec![0, 1].into_boxed_slice(),
            })
            .unwrap();
        let product = dag.normalize(product_switch, &RelationRegistry::default()).unwrap();
        assert_eq!(product.exact_terms().len(), 1);
        assert_eq!(product.exact_terms().keys().next().unwrap().factors().len(), 3);
    }

    #[test]
    fn family_static_and_dynamic_access_are_validated_without_enumeration() {
        let mut dag = ExpressionDag::new();
        let first = dag
            .push(ExpressionNode::Atom(SymbolicFactor::large(FactorIdentity::named("F0"))))
            .unwrap();
        let second = dag
            .push(ExpressionNode::Atom(SymbolicFactor::large(FactorIdentity::named("F1"))))
            .unwrap();
        let static_case = dag
            .push(ExpressionNode::FamilyGetStatic {
                cases: vec![first, second].into_boxed_slice(),
                index: 1,
            })
            .unwrap();
        assert_eq!(
            dag.normalize(static_case, &RelationRegistry::default())
                .unwrap()
                .first_large_witness()
                .unwrap()
                .identity,
            FactorIdentity::named("F1")
        );
        let invalid_static = dag
            .push(ExpressionNode::FamilyGetStatic {
                cases: vec![first, second].into_boxed_slice(),
                index: 2,
            })
            .unwrap();
        assert!(matches!(
            dag.normalize(invalid_static, &RelationRegistry::default()),
            Err(NormalFormError::InvalidFamilyIndex)
        ));
        let dynamic = dag
            .push(ExpressionNode::FamilyGetDynamic {
                selector: FactorIdentity::named("s"),
                cases: vec![first, second].into_boxed_slice(),
                stored_indices: vec![0_u8.into(), 1_u8.into()].into_boxed_slice(),
                domain_upper: 2_u8.into(),
            })
            .unwrap();
        let dynamic_nf = dag.normalize(dynamic, &RelationRegistry::default()).unwrap();
        let dynamic_data = dynamic_nf
            .exact_terms
            .values()
            .next()
            .unwrap()
            .monomial
            .factors
            .iter()
            .find_map(|factor| factor.switch.clone())
            .unwrap();
        assert_eq!(dynamic_data.case_indices, vec![0_u8.into(), 1_u8.into()].into_boxed_slice());
        let invalid_dynamic = dag
            .push(ExpressionNode::FamilyGetDynamic {
                selector: FactorIdentity::named("s"),
                cases: vec![first, second].into_boxed_slice(),
                stored_indices: vec![0_u8.into(), 1_u8.into()].into_boxed_slice(),
                domain_upper: 3_u8.into(),
            })
            .unwrap();
        assert!(matches!(
            dag.normalize(invalid_dynamic, &RelationRegistry::default()),
            Err(NormalFormError::InvalidFamilyDomain)
        ));
    }

    fn relation_key(public: FactorIdentity, target: FactorIdentity) -> FullRelationKey {
        FullRelationKey {
            source: "named".into(),
            ordered_indices: Box::new([]),
            public,
            target,
            matrix_type: None,
            layout: None,
            trapdoor: None,
            selector: None,
        }
    }

    fn associated_product_fixture(
        right_associated: bool,
        reverse_leaf_insertion: bool,
    ) -> (ExpressionDag, TermId, RelationRegistry) {
        let a = FactorIdentity::named("assoc-A");
        let b = FactorIdentity::named("assoc-B");
        let k = FactorIdentity::named("assoc-K");
        let p = FactorIdentity::named("assoc-P");
        let mut dag = ExpressionDag::new();
        let (a_id, b_id, k_id, p_id) = if reverse_leaf_insertion {
            let p_id = dag.push(ExpressionNode::Atom(SymbolicFactor::large(p.clone()))).unwrap();
            let k_id = dag
                .push(ExpressionNode::Atom(
                    SymbolicFactor::relation_live(k.clone(), bound(1)).unwrap(),
                ))
                .unwrap();
            let b_id = dag.push(ExpressionNode::Atom(SymbolicFactor::large(b.clone()))).unwrap();
            let a_id = dag.push(ExpressionNode::Atom(SymbolicFactor::large(a.clone()))).unwrap();
            (a_id, b_id, k_id, p_id)
        } else {
            let a_id = dag.push(ExpressionNode::Atom(SymbolicFactor::large(a.clone()))).unwrap();
            let b_id = dag.push(ExpressionNode::Atom(SymbolicFactor::large(b.clone()))).unwrap();
            let k_id = dag
                .push(ExpressionNode::Atom(
                    SymbolicFactor::relation_live(k.clone(), bound(1)).unwrap(),
                ))
                .unwrap();
            let p_id = dag.push(ExpressionNode::Atom(SymbolicFactor::large(p.clone()))).unwrap();
            (a_id, b_id, k_id, p_id)
        };
        let root = if right_associated {
            let inner =
                dag.push(ExpressionNode::Product(vec![b_id, k_id].into_boxed_slice())).unwrap();
            dag.push(ExpressionNode::Product(vec![a_id, inner].into_boxed_slice())).unwrap()
        } else {
            let inner =
                dag.push(ExpressionNode::Product(vec![a_id, b_id].into_boxed_slice())).unwrap();
            dag.push(ExpressionNode::Product(vec![inner, k_id].into_boxed_slice())).unwrap()
        };
        let mut registry = RelationRegistry::default();
        registry
            .register(RelationRegistration {
                key: relation_key(b, p.clone()),
                preimage: k,
                target: p_id,
            })
            .unwrap();
        (dag, root, registry)
    }

    #[test]
    fn prefix_suffix_and_nested_target_relations_reach_one_fixed_point() {
        let a = FactorIdentity::named("fixed-A");
        let b = FactorIdentity::named("fixed-B");
        let k = FactorIdentity::named("fixed-K");
        let c = FactorIdentity::named("fixed-C");
        let l = FactorIdentity::named("fixed-L");
        let p = FactorIdentity::named("fixed-P");
        let q = FactorIdentity::named("fixed-Q");
        let d = FactorIdentity::named("fixed-D");
        let mut dag = ExpressionDag::new();
        let atom = |dag: &mut ExpressionDag, factor: SymbolicFactor| {
            dag.push(ExpressionNode::Atom(factor)).unwrap()
        };
        let a_id = atom(&mut dag, SymbolicFactor::large(a.clone()));
        let b_id = atom(&mut dag, SymbolicFactor::large(b.clone()));
        let k_id = atom(&mut dag, SymbolicFactor::relation_live(k.clone(), bound(1)).unwrap());
        let c_id = atom(&mut dag, SymbolicFactor::large(c.clone()));
        let l_id = atom(&mut dag, SymbolicFactor::relation_live(l.clone(), bound(1)).unwrap());
        let q_id = atom(&mut dag, SymbolicFactor::large(q.clone()));
        let d_id = atom(&mut dag, SymbolicFactor::large(d.clone()));
        let nested_target =
            dag.push(ExpressionNode::Product(vec![c_id, l_id].into_boxed_slice())).unwrap();
        let root = dag
            .push(ExpressionNode::Product(vec![a_id, b_id, k_id, d_id].into_boxed_slice()))
            .unwrap();
        let mut registry = RelationRegistry::default();
        registry
            .register(RelationRegistration {
                key: relation_key(c.clone(), q.clone()),
                preimage: l,
                target: q_id,
            })
            .unwrap();
        registry
            .register(RelationRegistration {
                key: relation_key(b, p),
                preimage: k,
                target: nested_target,
            })
            .unwrap();

        let normalized = dag.normalize(root, &registry).unwrap();
        let key = normalized.exact_terms().keys().next().unwrap();
        assert_eq!(
            key.factors(),
            &[
                FactorIdentity::named("fixed-A"),
                FactorIdentity::named("fixed-Q"),
                FactorIdentity::named("fixed-D")
            ]
        );
        assert!(normalized.exact_terms().keys().all(|key| {
            key.factors().iter().all(|factor| {
                !matches!(factor, FactorIdentity { kind: FactorKind::Test(name), .. } if &**name == "fixed-B" || &**name == "fixed-K" || &**name == "fixed-P")
            })
        }));
    }

    #[test]
    fn association_and_insertion_order_preserve_nf_bound_and_counters() {
        let (left_dag, left_root, left_registry) = associated_product_fixture(false, false);
        let (right_dag, right_root, right_registry) = associated_product_fixture(true, true);
        let (left_nf, left_counters) =
            left_dag.normalize_with_counters(left_root, &left_registry).unwrap();
        let (right_nf, right_counters) =
            right_dag.normalize_with_counters(right_root, &right_registry).unwrap();
        assert_eq!(left_nf, right_nf);
        assert_eq!(left_nf.bounded_summary(), right_nf.bounded_summary());
        assert_eq!(left_counters, right_counters);
        assert_eq!(left_counters.relation_candidates, 1);
        assert_eq!(left_counters.relations_applied, 1);

        let mut first = ExpressionDag::new();
        let a = first
            .push(ExpressionNode::Atom(SymbolicFactor::large(FactorIdentity::named("add-A"))))
            .unwrap();
        let b = first
            .push(ExpressionNode::Atom(SymbolicFactor::large(FactorIdentity::named("add-B"))))
            .unwrap();
        let neg_a = first.push(ExpressionNode::Negate(a)).unwrap();
        let first_inner =
            first.push(ExpressionNode::Add(vec![b, neg_a].into_boxed_slice())).unwrap();
        let first_root =
            first.push(ExpressionNode::Add(vec![a, first_inner].into_boxed_slice())).unwrap();

        let mut second = ExpressionDag::new();
        let b2 = second
            .push(ExpressionNode::Atom(SymbolicFactor::large(FactorIdentity::named("add-B"))))
            .unwrap();
        let a2 = second
            .push(ExpressionNode::Atom(SymbolicFactor::large(FactorIdentity::named("add-A"))))
            .unwrap();
        let neg_a2 = second.push(ExpressionNode::Negate(a2)).unwrap();
        let second_inner =
            second.push(ExpressionNode::Add(vec![a2, b2].into_boxed_slice())).unwrap();
        let second_root = second
            .push(ExpressionNode::Add(vec![neg_a2, second_inner].into_boxed_slice()))
            .unwrap();
        let (first_nf, first_counters) =
            first.normalize_with_counters(first_root, &RelationRegistry::default()).unwrap();
        let (second_nf, second_counters) =
            second.normalize_with_counters(second_root, &RelationRegistry::default()).unwrap();
        assert_eq!(first_nf, second_nf);
        assert_eq!(first_nf.bounded_summary(), second_nf.bounded_summary());
        assert_eq!(first_counters, second_counters);
    }

    #[test]
    fn relation_registration_permutations_ignore_mismatches_and_report_same_ambiguity() {
        let public = FactorIdentity::named("perm-B");
        let preimage = FactorIdentity::named("perm-K");
        let first_target = FactorIdentity::named("perm-P");
        let second_target = FactorIdentity::named("perm-Q");
        let mut dag = ExpressionDag::new();
        let b = dag.push(ExpressionNode::Atom(SymbolicFactor::large(public.clone()))).unwrap();
        let k = dag
            .push(ExpressionNode::Atom(
                SymbolicFactor::relation_live(preimage.clone(), bound(1)).unwrap(),
            ))
            .unwrap();
        let p =
            dag.push(ExpressionNode::Atom(SymbolicFactor::large(first_target.clone()))).unwrap();
        let q =
            dag.push(ExpressionNode::Atom(SymbolicFactor::large(second_target.clone()))).unwrap();
        let root = dag.push(ExpressionNode::Product(vec![b, k].into_boxed_slice())).unwrap();
        let good = RelationRegistration {
            key: relation_key(public.clone(), first_target.clone()),
            preimage: preimage.clone(),
            target: p,
        };
        let mismatched = RelationRegistration {
            key: FullRelationKey {
                source: "wrong-owner".into(),
                target: second_target.clone(),
                ..good.key.clone()
            },
            preimage: preimage.clone(),
            target: q,
        };
        let alternate =
            RelationRegistration { key: relation_key(public, second_target), preimage, target: q };

        let mut mismatch_only = RelationRegistry::default();
        mismatch_only.register(good.clone()).unwrap();
        mismatch_only.register(mismatched.clone()).unwrap();
        let mismatch_nf = dag.normalize(root, &mismatch_only).unwrap();
        assert_eq!(mismatch_nf.exact_terms().keys().next().unwrap().factors(), &[first_target]);

        let mut first = RelationRegistry::default();
        first.register(good.clone()).unwrap();
        first.register(mismatched.clone()).unwrap();
        first.register(alternate.clone()).unwrap();
        let mut second = RelationRegistry::default();
        second.register(alternate).unwrap();
        second.register(mismatched).unwrap();
        second.register(good).unwrap();
        let first_error = dag.normalize(root, &first).unwrap_err();
        let second_error = dag.normalize(root, &second).unwrap_err();
        assert_eq!(first_error, second_error);
        assert!(matches!(first_error, NormalFormError::AmbiguousRelation { .. }));
    }

    #[test]
    fn immutable_dag_normalization_is_deterministic_across_sixteen_threads() {
        let (dag, root, registry) = associated_product_fixture(true, true);
        std::thread::scope(|scope| {
            let handles = (0..16)
                .map(|_| scope.spawn(|| dag.normalize_with_counters(root, &registry)))
                .collect::<Vec<_>>();
            let results = handles
                .into_iter()
                .map(|handle| handle.join().unwrap().unwrap())
                .collect::<Vec<_>>();
            let (expected_nf, expected_counters) = &results[0];
            for (nf, counters) in &results[1..] {
                assert_eq!(nf, expected_nf);
                assert_eq!(counters, expected_counters);
            }
        });
    }

    #[test]
    fn first_large_witness_uses_canonical_key_order_not_insertion_order() {
        let build = |reverse: bool| {
            let mut dag = ExpressionDag::new();
            let a = dag
                .push(ExpressionNode::Atom(SymbolicFactor::large(FactorIdentity::named(
                    "witness-A",
                ))))
                .unwrap();
            let b = dag
                .push(ExpressionNode::Atom(SymbolicFactor::large(FactorIdentity::named(
                    "witness-B",
                ))))
                .unwrap();
            let root = if reverse {
                dag.push(ExpressionNode::Add(vec![b, a].into_boxed_slice())).unwrap()
            } else {
                dag.push(ExpressionNode::Add(vec![a, b].into_boxed_slice())).unwrap()
            };
            dag.normalize(root, &RelationRegistry::default())
                .unwrap()
                .first_large_witness()
                .unwrap()
        };
        let forward = build(false);
        let reverse = build(true);
        assert_eq!(forward, reverse);
        assert_eq!(forward.factor_index, 0);
        assert_eq!(forward.identity, FactorIdentity::named("witness-A"));
    }
}
