//! Checked one-way sampler relations for the operational-noise e-graph.
//!
//! This module intentionally has no Graph-IR cache.  Lowering records the
//! complete sampler identity in [`RelationSource`] and the caller registers
//! the corresponding public/target pair once.  Every use compares rebuilt
//! e-class identities, including the ordered coordinate children.

use super::{
    analysis::{
        MxxAnalysis, MxxSort, RelationProvenance, RelationProvenanceVisit, RelationSource,
        matrix_types_equal, resolved_constant, resolved_equal, try_visit_relation_provenance,
    },
    family,
    identity::{
        AtomicRelationRole, AtomicSourceId, AtomicSourceKey, Axis, BinderId, MatrixConstantValue,
        SamplerIdentity, TrapdoorDescriptorId,
    },
    language::MxxLang,
};
use egg::{Applier, EGraph, Id, Language, RecExpr, SearchMatches, Searcher, Subst, Symbol, Var};
use num_bigint::BigInt;
use num_traits::Zero;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

/// A closed relation registration supplied by source lowering.
///
/// `source` is deliberately an atom source rather than a node number: the
/// atom's ordered index children are checked separately after rebuild.
/// This compact rewrite snapshot is intentionally distinct from
/// [`SamplerIdentity`]: the sampler owns producer semantics, while egg's
/// callbacks require an immutable lookup keyed by the final `AtomicSourceId`.
/// Reconstructing that reverse mapping inside every callback would either lose
/// the atom identity or repeatedly scan all sampler/source descriptors.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct RelationRegistration {
    pub source: AtomicSourceId,
    pub expected_public: Id,
    pub target: Id,
    pub trapdoor: Option<TrapdoorDescriptorId>,
    pub indices: Box<[Id]>,
}

/// Fail-closed outcomes retained by the rewrite context.  Egg's callback API
/// cannot return a `Result`; callers must inspect this after every rewrite
/// pass before extraction.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum RelationFailure {
    MissingRegistration { source: AtomicSourceId },
    MismatchedIndex { source: AtomicSourceId },
    MismatchedPublic { source: AtomicSourceId },
    AmbiguousReplacement { sources: Box<[AtomicSourceId]> },
    DifferentSelectorBlocked,
    TransformedOperand,
    UnavailableRelation { source: AtomicSourceId },
    InvalidRelationProducer { source: AtomicSourceId },
    MismatchedType { source: AtomicSourceId },
    MismatchedLayout { source: AtomicSourceId },
    MismatchedTrapdoor { source: AtomicSourceId },
    MismatchedTarget { source: AtomicSourceId },
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RelationCounters {
    pub candidates: u64,
    pub rewrites: u64,
    pub selector_distributions: u64,
}

#[derive(Debug)]
struct RelationState {
    registrations: BTreeMap<AtomicSourceId, Vec<RelationRegistration>>,
    failure: Option<RelationFailure>,
    counters: RelationCounters,
}

/// Job-owned controls shared with all checker phases.  Relation rewriting only
/// reserves through this object; it never starts a phase-local clock or count.
#[derive(Clone, Debug)]
pub struct SharedRewriteBudget {
    owned: Arc<AtomicUsize>,
}

impl SharedRewriteBudget {
    pub fn new() -> Self {
        Self { owned: Arc::new(AtomicUsize::new(0)) }
    }
    /// Uses the simulation driver's cumulative counter.  This is the
    /// production constructor: relation callbacks never own a second budget.
    pub(crate) fn from_shared(owned: Arc<AtomicUsize>) -> Self {
        Self { owned }
    }

    pub fn owned(&self) -> usize {
        self.owned.load(Ordering::Relaxed)
    }

    fn reserve(&self, additional: usize) -> Result<(), RelationFailure> {
        let mut observed = self.owned.load(Ordering::Relaxed);
        loop {
            let next = observed.saturating_add(additional);
            match self.owned.compare_exchange_weak(
                observed,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(()),
                Err(current) => observed = current,
            }
        }
    }
}

/// Shared, simulation-wide relation state.  The deadline and cumulative
/// reservation counter are intentionally shared by search and application;
/// relation matching never creates a second per-rewrite budget.
#[derive(Clone, Debug)]
pub struct RewriteContext {
    state: Arc<Mutex<RelationState>>,
    budget: SharedRewriteBudget,
}

impl RewriteContext {
    pub fn new(budget: SharedRewriteBudget) -> Self {
        Self {
            state: Arc::new(Mutex::new(RelationState {
                registrations: BTreeMap::new(),
                failure: None,
                counters: RelationCounters::default(),
            })),
            budget,
        }
    }

    pub fn register(&self, registration: RelationRegistration) {
        let mut state = self.state.lock().expect("relation context lock");
        state.registrations.entry(registration.source).or_default().push(registration);
    }

    pub fn failure(&self) -> Option<RelationFailure> {
        self.state.lock().expect("relation context lock").failure.clone()
    }

    pub fn counters(&self) -> RelationCounters {
        self.state.lock().expect("relation context lock").counters.clone()
    }

    pub(crate) fn reserve(&self, additional: usize) -> bool {
        let mut state = self.state.lock().expect("relation context lock");
        if state.failure.is_some() {
            return false;
        }
        if let Err(failure) = self.budget.reserve(additional) {
            state.failure = Some(failure);
            return false;
        }
        true
    }

    fn note_candidate(&self) {
        self.state.lock().expect("relation context lock").counters.candidates += 1;
    }

    pub(crate) fn note_rewrite(&self, selector_distribution: bool) {
        let mut state = self.state.lock().expect("relation context lock");
        state.counters.rewrites += 1;
        if selector_distribution {
            state.counters.selector_distributions += 1;
        }
    }

    fn fail(&self, failure: RelationFailure) {
        let mut state = self.state.lock().expect("relation context lock");
        if state.failure.is_none() {
            state.failure = Some(failure);
        }
    }

    fn registrations(&self, source: AtomicSourceId) -> Vec<RelationRegistration> {
        self.state
            .lock()
            .expect("relation context lock")
            .registrations
            .get(&source)
            .cloned()
            .unwrap_or_default()
    }
}

/// A relation-bearing factor searcher.  It matches an ordered matrix product
/// containing a relation factor at any position; multiplication is never made
/// commutative to expose a relation.
#[derive(Clone, Debug)]
pub struct RelationSearcher {
    context: RewriteContext,
}

impl RelationSearcher {
    pub fn new(context: RewriteContext) -> Self {
        Self { context }
    }
}

impl Searcher<MxxLang, MxxAnalysis> for RelationSearcher {
    fn search_eclass_with_limit(
        &self,
        egraph: &EGraph<MxxLang, MxxAnalysis>,
        eclass: Id,
        limit: usize,
    ) -> Option<SearchMatches<'_, MxxLang>> {
        if limit == 0 || self.context.failure().is_some() || !self.context.reserve(1) {
            return None;
        }
        let class = &egraph[egraph.find(eclass)];
        let relation_match = class.nodes.iter().any(|node| {
            matches!(node, MxxLang::MatrixMultiply(factors)
            if factors.iter().any(|factor| {
                !egraph[egraph.find(*factor)].data.relation_provenance.is_empty()
            }) || right_nested_relation_factor_candidates(egraph, factors, &self.context)
                .is_some_and(|candidates| !candidates.is_empty()))
        });
        let has_physical_add = class.nodes.iter().any(|node| matches!(node, MxxLang::MatrixAdd(_)));
        let matched = relation_match ||
            (has_physical_add && pointwise_add_switch_cancellation_possible(egraph, eclass));
        matched.then(|| SearchMatches { eclass, substs: vec![Subst::default()], ast: None })
    }

    fn vars(&self) -> Vec<Var> {
        Vec::new()
    }
}

/// Applies one checked, directional relation rewrite.  It recomputes the
/// concrete candidate at application time because egg may have rebuilt an
/// e-class since search.  This makes canonical identity, not raw `Id`, the
/// authority for public operands, targets, layouts and index children.
#[derive(Clone, Debug)]
pub struct RelationApplier {
    context: RewriteContext,
}

impl RelationApplier {
    pub fn new(context: RewriteContext) -> Self {
        Self { context }
    }
}

impl Applier<MxxLang, MxxAnalysis> for RelationApplier {
    fn apply_one(
        &self,
        egraph: &mut EGraph<MxxLang, MxxAnalysis>,
        eclass: Id,
        _subst: &Subst,
        _searcher_ast: Option<&egg::PatternAst<MxxLang>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        if self.context.failure().is_some() || !self.context.reserve(1) {
            return Vec::new();
        }
        let root = egraph.find(eclass);
        let nodes = egraph[root].nodes.clone();
        for plan in pointwise_add_switch_cancellation_plans(egraph, root) {
            self.context.note_candidate();
            if !self.context.reserve(1) {
                return Vec::new();
            }
            if let Some(replacement) = build_pointwise_add_switch_cancellation_with_context(
                egraph,
                root,
                plan,
                &self.context,
            ) && egraph.union(root, replacement)
            {
                self.context.note_rewrite(true);
                return vec![replacement];
            }
        }
        for node in nodes {
            let MxxLang::MatrixMultiply(factors) = node else { continue };
            let nested_candidates =
                right_nested_relation_factor_candidates(egraph, &factors, &self.context);
            for candidate in std::iter::once(factors).chain(nested_candidates.into_iter().flatten())
            {
                if let Some(plan) =
                    checked_product_replacement_plan(egraph, &self.context, &candidate)
                {
                    if replacement_plan_satisfied(egraph, root, &plan.replacement) {
                        continue;
                    }
                    if let Some(replacement) =
                        materialize_replacement_plan(egraph, &self.context, &plan.replacement) &&
                        egraph.union(root, replacement)
                    {
                        self.context.note_rewrite(plan.selector_distribution);
                        return vec![replacement];
                    }
                }
                if self.context.failure().is_some() {
                    return Vec::new();
                }
            }
        }
        Vec::new()
    }
}

/// Enumerates only immediate, ordered right-nested relation boundaries.  A
/// candidate is `prefix * K * (R * tail) * suffix`, exposed as the ephemeral
/// factor sequence `prefix, K, R, tail, suffix`.  Physical inner witnesses
/// remain independent; this never combines e-class alternatives or inserts an
/// associativity e-node.
fn right_nested_relation_factor_candidates_with_reserve(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    factors: &[Id],
    reserve: &mut dyn FnMut() -> bool,
) -> Option<Vec<Box<[Id]>>> {
    let mut candidates = Vec::new();
    for outer_position in 1..factors.len() {
        if !reserve() {
            return None;
        }
        let inner = egraph.find(factors[outer_position]);
        for node in &egraph[inner].nodes {
            if !reserve() {
                return None;
            }
            let MxxLang::MatrixMultiply(inner_factors) = node else { continue };
            let Some(relation) = inner_factors.first().copied() else { continue };
            if egraph[egraph.find(relation)].data.relation_provenance.is_empty() {
                continue;
            }
            if !reserve() {
                return None;
            }
            let mut candidate = Vec::new();
            candidate
                .try_reserve_exact(
                    factors.len().checked_add(inner_factors.len())?.saturating_sub(1),
                )
                .ok()?;
            for factor in factors[..outer_position]
                .iter()
                .chain(inner_factors)
                .chain(&factors[outer_position + 1..])
            {
                if !reserve() {
                    return None;
                }
                candidate.push(egraph.find(*factor));
            }
            if !reserve() {
                return None;
            }
            candidates.try_reserve(1).ok()?;
            candidates.push(candidate.into_boxed_slice());
        }
    }
    Some(candidates)
}

fn right_nested_relation_factor_candidates(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    factors: &[Id],
    context: &RewriteContext,
) -> Option<Vec<Box<[Id]>>> {
    right_nested_relation_factor_candidates_with_reserve(egraph, factors, &mut || {
        context.reserve(1)
    })
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum ReplacementPlan {
    Existing(Id),
    Product(Box<[ReplacementPlan]>),
    Add(Box<[ReplacementPlan]>),
    Negate(Box<ReplacementPlan>),
    Switch(Box<[ReplacementPlan]>),
    Concat {
        axis: Axis,
        inputs: Box<[ReplacementPlan]>,
    },
    /// Every member is an exact physical witness of one relation result.  The
    /// applier unions them just as the old distribution materializer did.
    Equivalent(Box<[ReplacementPlan]>),
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CheckedProductReplacementPlan {
    replacement: ReplacementPlan,
    selector_distribution: bool,
}

/// Extraction-only classification of one exact product node. Both facts are
/// derived from the same ephemeral checked replacement-plan scan.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProposalRelationClassification {
    /// The checked replacement is not yet represented by the enclosing class.
    pub relation_redex: bool,
    /// The number of distinct canonical ordered factor-sequence boundaries
    /// with one authoritative checked replacement. This is e-node-local and
    /// never an extraction obligation, including after a replacement has been
    /// unioned.
    pub local_checked_relation_count: u64,
}

fn existing_product_plan(factors: &[Id]) -> ReplacementPlan {
    let factors = factors.iter().copied().map(ReplacementPlan::Existing).collect::<Vec<_>>();
    if factors.len() == 1 {
        factors.into_iter().next().expect("one product factor")
    } else {
        ReplacementPlan::Product(factors.into_boxed_slice())
    }
}

fn splice_product_plan(
    prefix: &[Id],
    middle: &[ReplacementPlan],
    suffix: &[Id],
) -> ReplacementPlan {
    let mut factors = Vec::with_capacity(prefix.len() + middle.len() + suffix.len());
    factors.extend(prefix.iter().copied().map(ReplacementPlan::Existing));
    factors.extend_from_slice(middle);
    factors.extend(suffix.iter().copied().map(ReplacementPlan::Existing));
    if factors.len() == 1 {
        factors.into_iter().next().expect("one product factor")
    } else {
        ReplacementPlan::Product(factors.into_boxed_slice())
    }
}

fn checked_product_replacement_plan(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    context: &RewriteContext,
    factors: &[Id],
) -> Option<CheckedProductReplacementPlan> {
    for relation_position in 1..factors.len() {
        if let Some(replacement) =
            checked_product_replacement_plan_at(egraph, context, factors, relation_position)
        {
            return Some(replacement);
        }
        if context.failure().is_some() {
            return None;
        }
    }
    None
}

/// Checks exactly one ordered public/relation boundary in one factor sequence.
/// Callers that need a first rewrite keep the historical left-to-right scan;
/// extraction instead visits every such boundary to rank all checked plans.
fn checked_product_replacement_plan_at(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    context: &RewriteContext,
    factors: &[Id],
    relation_position: usize,
) -> Option<CheckedProductReplacementPlan> {
    let relation = *factors.get(relation_position)?;
    if relation_position == 0 || egraph[egraph.find(relation)].data.relation_provenance.is_empty() {
        return None;
    }
    let public = factors[relation_position - 1];
    match pointwise_same_selector_plan(egraph, public, relation, true) {
        Ok(Some(product)) => {
            return Some(CheckedProductReplacementPlan {
                replacement: splice_product_plan(
                    &factors[..relation_position - 1],
                    &[product],
                    &factors[relation_position + 1..],
                ),
                selector_distribution: true,
            });
        }
        Ok(None)
            if switch_node(egraph, public).is_some() || switch_node(egraph, relation).is_some() =>
        {
            context.fail(RelationFailure::TransformedOperand);
            return None;
        }
        Err(RelationFailure::DifferentSelectorBlocked) => return None,
        Err(failure) => {
            context.fail(failure);
            return None;
        }
        Ok(None) => {}
    }
    checked_replacement_plan(egraph, context, factors, relation_position)
}

/// A physical representation query never conflates an absent operation with
/// competing operations.  Both are fail-closed for transformations, but only
/// the latter must not be treated as an atomic leaf.
enum PhysicalStructure<T> {
    Absent,
    Unique(T),
    Ambiguous,
}

fn physical_add_terms(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    id: Id,
) -> PhysicalStructure<Box<[Id]>> {
    let mut unique = None;
    for node in &egraph[egraph.find(id)].nodes {
        let MxxLang::MatrixAdd(terms) = node else { continue };
        let candidate =
            terms.iter().map(|term| egraph.find(*term)).collect::<Vec<_>>().into_boxed_slice();
        match &unique {
            Some(previous) if previous != &candidate => return PhysicalStructure::Ambiguous,
            Some(_) => {}
            None => unique = Some(candidate),
        }
    }
    match unique {
        Some(terms) => PhysicalStructure::Unique(terms),
        None => PhysicalStructure::Absent,
    }
}

/// Returns Switch cases only when every physical Switch representation agrees
/// after e-class canonicalization.  A selector diagonalization is an exact
/// equality, so it must not choose one competing stored-case vector.
fn physical_switch_cases(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    id: Id,
) -> PhysicalStructure<Box<[Id]>> {
    let mut unique = None;
    for node in &egraph[egraph.find(id)].nodes {
        let MxxLang::Switch(cases) = node else { continue };
        let candidate =
            cases.iter().map(|case| egraph.find(*case)).collect::<Vec<_>>().into_boxed_slice();
        match &unique {
            Some(previous) if previous != &candidate => return PhysicalStructure::Ambiguous,
            Some(_) => {}
            None => unique = Some(candidate),
        }
    }
    match unique {
        Some(cases) => PhysicalStructure::Unique(cases),
        None => PhysicalStructure::Absent,
    }
}

fn physical_product_factors(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    id: Id,
) -> PhysicalStructure<Box<[Id]>> {
    let mut unique = None;
    for node in &egraph[egraph.find(id)].nodes {
        let MxxLang::MatrixMultiply(factors) = node else { continue };
        let candidate = factors
            .iter()
            .map(|factor| egraph.find(*factor))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        match &unique {
            Some(previous) if previous != &candidate => return PhysicalStructure::Ambiguous,
            Some(_) => {}
            None => unique = Some(candidate),
        }
    }
    match unique {
        Some(factors) => PhysicalStructure::Unique(factors),
        None => PhysicalStructure::Absent,
    }
}

fn unique_add_terms(egraph: &EGraph<MxxLang, MxxAnalysis>, id: Id) -> Option<Box<[Id]>> {
    match physical_add_terms(egraph, id) {
        PhysicalStructure::Unique(terms) => Some(terms),
        PhysicalStructure::Absent | PhysicalStructure::Ambiguous => None,
    }
}

/// Returns a negated base only when the e-class has exactly one physical
/// negate representation.  This avoids silently choosing a sign from an
/// ambiguous e-class.
fn physical_negated_base(egraph: &EGraph<MxxLang, MxxAnalysis>, term: Id) -> PhysicalStructure<Id> {
    let mut unique = None;
    for node in &egraph[egraph.find(term)].nodes {
        let MxxLang::MatrixNegate([input]) = node else { continue };
        let candidate = egraph.find(*input);
        match unique {
            Some(previous) if previous != candidate => return PhysicalStructure::Ambiguous,
            Some(_) => {}
            None => unique = Some(candidate),
        }
    }
    match unique {
        Some(base) => PhysicalStructure::Unique(base),
        None => PhysicalStructure::Absent,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PointwiseCaseFlattenReject {
    AmbiguousAdd,
    NestedSwitchOrCycle,
}

/// Failure-only metadata captured with the signed case plan.  It prevents a
/// later diagnostic from reinterpreting an e-class after saturation changed
/// its physical representation.
struct FlattenedPointwiseCase {
    terms: Vec<SignedPointwiseTerm>,
    physical_adds: usize,
    grouped_add_children: usize,
}

/// Flattens one stored outer case while diagonally selecting nested Switches
/// that are governed by exactly the same selector.  The traversal is local to
/// one stored case and read-only: it neither enumerates selector products nor
/// constructs an e-node.  A nested Switch is usable only when its sole
/// physical case vector agrees with the outer selector and case count.
fn flatten_pointwise_case(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    root: Id,
    outer_selector: Id,
    outer_case_count: usize,
    case_index: usize,
    input: Id,
) -> Result<FlattenedPointwiseCase, PointwiseCaseFlattenReject> {
    let root = egraph.find(root);
    let outer_selector = egraph.find(outer_selector);
    if case_index >= outer_case_count ||
        !selector_domain_matches_cases(egraph, outer_selector, outer_case_count)
    {
        return Err(PointwiseCaseFlattenReject::NestedSwitchOrCycle);
    }

    if egraph.find(input) == root {
        return Err(PointwiseCaseFlattenReject::NestedSwitchOrCycle);
    }
    let case = egraph.find(input);
    let physical_adds =
        egraph[case].nodes.iter().filter(|node| matches!(node, MxxLang::MatrixAdd(_))).count();
    let grouped_add_children = egraph[case]
        .nodes
        .iter()
        .filter_map(|node| match node {
            MxxLang::MatrixAdd(children) => Some(children.iter()),
            _ => None,
        })
        .flatten()
        .filter(|child| unique_add_terms(egraph, **child).is_some())
        .count();
    let mut specialize = |term| match physical_switch_cases(egraph, term) {
        PhysicalStructure::Absent => Ok(None),
        PhysicalStructure::Ambiguous => Err(()),
        PhysicalStructure::Unique(cases) => {
            let Some((&selector, cases)) = cases.split_first() else { return Err(()) };
            if selector != outer_selector || cases.len() != outer_case_count {
                return Err(());
            }
            let selected = cases.get(case_index).copied().ok_or(())?;
            Ok(Some((selected != root).then_some((selected, false)).ok_or(())?))
        }
    };
    signed_additive_leaves_with_specialization_and_progress(
        egraph,
        &[input],
        |_| {},
        &mut specialize,
        &mut || Ok(()),
    )
    .map(|terms| FlattenedPointwiseCase {
        terms: terms
            .into_iter()
            .map(|(base, negative)| SignedPointwiseTerm { base, negative })
            .collect(),
        physical_adds,
        grouped_add_children,
    })
    .ok_or(PointwiseCaseFlattenReject::AmbiguousAdd)
}

/// A binder selector carries the authoritative stored-case domain.  Plain
/// selectors have no owner descriptor, so their physical case vector itself
/// remains the complete contract used by the direct planner.
fn selector_domain_matches_cases(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    selector: Id,
    case_count: usize,
) -> bool {
    let selector = egraph.find(selector);
    match egraph[selector].nodes.as_slice() {
        [MxxLang::IntBinder(binder)] => {
            egraph.analysis.symbols.binders.get(binder.0).is_some_and(|descriptor| {
                descriptor.minimum == BigInt::zero() &&
                    descriptor.maximum == BigInt::from(case_count.saturating_sub(1))
            })
        }
        nodes if nodes.iter().any(|node| matches!(node, MxxLang::IntBinder(_))) => false,
        _ => true,
    }
}

/// Returns whether materializing `remaining` would add a genuinely new,
/// strictly simpler representative to `root`.
///
/// Empty remainders are materialized as a zero constant; every nonempty
/// remainder, including a singleton, is materialized as one MatrixAdd node.
/// Keeping that singleton wrapper avoids making an old Add node refer to one
/// of its own children after canonicalization.
fn signed_additive_child_matches(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    child: Id,
    (base, negative): (Id, bool),
) -> bool {
    let child = egraph.find(child);
    let direct = if !negative {
        child == base
    } else {
        matches!(
            physical_negated_base(egraph, child),
            PhysicalStructure::Unique(negated) if negated == base && negated != child
        )
    };
    if direct {
        return true;
    }
    match physical_add_terms(egraph, child) {
        // Singleton wrappers are intentionally materialized when a selected
        // Switch result would otherwise be an input of the Add being unioned.
        // They preserve the exact signed leaf while preventing an e-class
        // self-cycle; treat that wrapper as the same retained term here.
        PhysicalStructure::Unique(terms) if terms.len() == 1 && egraph.find(terms[0]) != child => {
            signed_additive_child_matches(egraph, terms[0], (base, negative))
        }
        PhysicalStructure::Absent | PhysicalStructure::Ambiguous | PhysicalStructure::Unique(_) => {
            false
        }
    }
}

fn cancelled_signed_additive_terms(terms: &[(Id, bool)]) -> (Vec<bool>, bool) {
    let mut cancelled = vec![false; terms.len()];
    let mut positive = HashMap::<Id, Vec<usize>>::new();
    let mut negative = HashMap::<Id, Vec<usize>>::new();
    let mut any = false;
    for (index, (base, is_negative)) in terms.iter().enumerate() {
        let opposite = if *is_negative { &mut positive } else { &mut negative };
        if let Some(match_index) = opposite.get_mut(base).and_then(Vec::pop) {
            cancelled[index] = true;
            cancelled[match_index] = true;
            any = true;
        } else {
            let same = if *is_negative { &mut negative } else { &mut positive };
            same.entry(*base).or_default().push(index);
        }
    }
    (cancelled, any)
}

/// A fully validated pointwise Add/Switch cancellation.  Planning is read-only
/// so a failing case cannot leave partial e-nodes in the graph.
struct PointwiseAddSwitchPlan {
    selector: Id,
    cases: Vec<Vec<SignedPointwiseTerm>>,
    binder_aware: Option<Box<[BinderAwarePointwiseAddSwitchPlan]>>,
}

/// A signed leaf selected while planning a pointwise Add/Switch rewrite.  It
/// is intentionally only a plan: a negation required by selector
/// diagonalization is materialized after all stored cases have passed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SignedPointwiseTerm {
    base: Id,
    negative: bool,
}

/// A read-only description of the symbolic fixed portion of a pointwise
/// cancellation.  Instantiation is deliberately deferred to the applier:
/// search must not add speculative e-nodes.
struct BinderAwarePointwiseAddSwitchPlan {
    binder: BinderId,
    /// Direct terms of every physical stored case, structurally checked by the
    /// read-only planner before the applier creates its first e-node.
    case_terms: Box<[Box<[SignedPointwiseTerm]>]>,
    fixed_bases: Box<[Id]>,
    fixed_occurrences: Box<[FixedOccurrence]>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct FixedOccurrence {
    base_index: usize,
    negative: bool,
}

/// A bounded diagnostic summary of one signed direct Add identity.  `eclass`
/// is canonical at the point the pointwise plan was inspected; it is only a
/// failure log coordinate and is never retained in e-graph analysis.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct SignedCanonicalMultiplicity {
    pub eclass: usize,
    pub negative: bool,
    pub multiplicity: usize,
}

const MAX_UNMATCHED_TERM_IDENTITIES: usize = 16;
const MAX_POINTWISE_PROBE_STRUCTURES: usize = 8;

/// Failure-only result of the symbolic binder preflight.  It is deliberately
/// separate from the direct Add/Switch cancellation result: neither failure
/// changes the rewrite's eligibility.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum BinderPreflightReject {
    SelectorNotUniqueBinder,
    MissingDescriptor { binder: BinderId },
    DomainCaseCountMismatch { binder: BinderId, case_count: usize },
    FixedTermsEmptyOrSwitch,
    FixedSignedFlatten,
    CaseAmbiguous { case_index: usize },
    CaseSelfCycle { case_index: usize },
    CaseNestedSwitch { case_index: usize },
}

/// Read-only summary of the exact symbolic work the binder planner would
/// instantiate.  It is created before instantiation and therefore never adds
/// e-nodes or changes relation availability.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BinderPreflightReady {
    pub fixed_terms: Box<[SignedCanonicalMultiplicity]>,
    pub fixed_terms_omitted_occurrences: usize,
    pub unique_base_count: usize,
    pub case_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum PointwiseDirectProbe {
    Ready,
    Rejected(PointwiseAddSwitchReject),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PointwiseAddSwitchProbeOutcome {
    pub direct: PointwiseDirectProbe,
    pub binder: Result<BinderPreflightReady, BinderPreflightReject>,
}

/// Bounded, deterministic diagnostics for every eligible physical Add/Switch
/// structure.  Production only consumes successful plans; this probe is used
/// solely after a selected Large residual and owns no cache or e-graph state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PointwiseAddSwitchProbe {
    pub physical_root_adds: usize,
    pub eligible_single_switch_adds: usize,
    pub direct_switch_children: usize,
    pub direct_grouped_add_children: usize,
    pub outcomes: Box<[PointwiseAddSwitchProbeOutcome]>,
    pub omitted_eligible_structures: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum PointwiseAddSwitchReject {
    #[cfg(test)]
    Structural {
        physical_root_adds: usize,
        eligible_single_switch_adds: usize,
        direct_switch_children: usize,
        direct_grouped_add_children: usize,
    },
    FixedTermsEmptyOrSwitch,
    /// `case_index` is zero-based among the stored cases (the selector is excluded).
    CaseCycleOrNestedSwitch {
        case_index: usize,
    },
    /// `case_index` is zero-based among the stored cases (the selector is excluded).
    UnmatchedFixedTerms {
        physical_root_adds: usize,
        eligible_single_switch_adds: usize,
        /// Exact canonical selector and physical stored cases chosen by the
        /// planner before the first unmatched case.  These are failure-only
        /// coordinates for extraction diagnostics, not relation state.
        selector: Id,
        switch_cases: Box<[Id]>,
        case_index: usize,
        matched: usize,
        required: usize,
        direct_terms: usize,
        negated_terms: usize,
        fixed_unique_add_children: usize,
        case_physical_adds: usize,
        case_grouped_add_children: usize,
        /// Signed canonical fixed identities, sorted by e-class then polarity.
        /// At most `MAX_UNMATCHED_TERM_IDENTITIES` entries are retained.
        fixed_terms: Box<[SignedCanonicalMultiplicity]>,
        fixed_terms_omitted_occurrences: usize,
        /// Signed canonical direct identities in the first failing stored case,
        /// sorted by e-class then polarity.  It is built only after failure.
        case_terms: Box<[SignedCanonicalMultiplicity]>,
        case_terms_omitted_occurrences: usize,
    },
    EquivalentResult,
}

#[derive(Clone)]
struct PointwiseAddSwitchStructure {
    terms: Vec<Id>,
    switch_index: usize,
    switch: Box<[Id]>,
}

struct PointwiseAddSwitchStructures {
    physical_root_adds: usize,
    direct_switch_children: usize,
    direct_grouped_add_children: usize,
    eligible: Vec<PointwiseAddSwitchStructure>,
}

fn pointwise_add_switch_structures(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    root: Id,
) -> PointwiseAddSwitchStructures {
    let root = egraph.find(root);
    let mut physical_root_adds = 0;
    let mut direct_switch_children = 0;
    let mut direct_grouped_add_children = 0;
    // The node itself is the canonical ordering and deduplication key.  This
    // deliberately keeps physically distinct candidate representations while
    // coalescing duplicate e-nodes introduced by congruence closure.
    let mut eligible = BTreeMap::new();
    for node in &egraph[root].nodes {
        let MxxLang::MatrixAdd(terms) = node else { continue };
        physical_root_adds += 1;
        for term in terms {
            let term = egraph.find(*term);
            direct_switch_children += usize::from(has_physical_switch(egraph, term));
            direct_grouped_add_children += usize::from(unique_add_terms(egraph, term).is_some());
        }
        if let Some((key, structure)) = pointwise_add_switch_structure(egraph, root, terms) {
            eligible.entry(key).or_insert(structure);
        }
    }
    PointwiseAddSwitchStructures {
        physical_root_adds,
        direct_switch_children,
        direct_grouped_add_children,
        eligible: eligible.into_values().collect(),
    }
}

fn pointwise_add_switch_cancellation_possible(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    root: Id,
) -> bool {
    !pointwise_add_switch_cancellation_plans(egraph, root).is_empty()
}

#[cfg(test)]
fn pointwise_add_switch_cancellation_plan(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    root: Id,
) -> Option<PointwiseAddSwitchPlan> {
    pointwise_add_switch_cancellation_plans(egraph, root).into_iter().next()
}

/// Enumerates every structurally distinct physical Add candidate in a stable
/// order.  A plan is entirely read-only; application may still fail while
/// instantiating a binder-owned element, in which case the next candidate is
/// attempted without unioning the root to any partial result.
fn pointwise_add_switch_cancellation_plans(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    root: Id,
) -> Vec<PointwiseAddSwitchPlan> {
    let root = egraph.find(root);
    let structures = pointwise_add_switch_structures(egraph, root);
    let physical_root_adds = structures.physical_root_adds;
    let eligible_single_switch_adds = structures.eligible.len();
    let mut plans = structures
        .eligible
        .iter()
        .filter_map(|structure| {
            pointwise_add_switch_cancellation_for_structure(
                egraph,
                root,
                structure,
                physical_root_adds,
                eligible_single_switch_adds,
            )
            .ok()
        })
        .collect::<Vec<_>>();
    plans.extend(structures.eligible.into_iter().filter_map(|structure| {
        let selector = egraph.find(structure.switch[0]);
        binder_aware_pointwise_add_switch_cancellation_for_structure(egraph, root, &structure)
            .ok()
            .map(|binder_aware| PointwiseAddSwitchPlan {
                selector,
                cases: Vec::new(),
                binder_aware: Some(Box::new([binder_aware])),
            })
    }));
    plans
}

pub(crate) fn selected_polynomial_monomials(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    expression: &RecExpr<MxxLang>,
    origins: &[Id],
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<Vec<Vec<(Box<[Id]>, bool)>>> {
    if origins.len() != expression.as_ref().len() {
        return None;
    }
    let mut memo = Vec::<Vec<(Box<[Id]>, bool)>>::with_capacity(expression.as_ref().len());
    for (index, node) in expression.as_ref().iter().enumerate() {
        let mut terms = match node {
            MxxLang::MatrixAdd(children) => {
                let mut terms = Vec::new();
                for child in children {
                    for term in memo.get(usize::from(*child))? {
                        progress().ok()?;
                        terms.try_reserve(1).ok()?;
                        terms.push(term.clone());
                    }
                }
                terms
            }
            MxxLang::MatrixNegate([child]) => {
                let mut terms = Vec::new();
                for (spine, negative) in memo.get(usize::from(*child))? {
                    progress().ok()?;
                    terms.try_reserve(1).ok()?;
                    terms.push((spine.clone(), !negative));
                }
                terms
            }
            MxxLang::MatrixMultiply(factors) => {
                let mut product = vec![(Box::<[Id]>::default(), false)];
                for factor in factors {
                    let terms = memo.get(usize::from(*factor))?;
                    product = ordered_cartesian_multiply_signed_spines(product, terms, progress)?;
                    canonicalize_central_constant_scalar_spines(egraph, &mut product, progress)?;
                    cancel_signed_spines(&mut product);
                }
                product
            }
            _ => {
                progress().ok()?;
                let mut spine = Vec::new();
                spine.try_reserve_exact(1).ok()?;
                spine.push(egraph.find(*origins.get(index)?));
                vec![(spine.into_boxed_slice(), false)]
            }
        };
        canonicalize_central_constant_scalar_spines(egraph, &mut terms, progress)?;
        cancel_signed_spines(&mut terms);
        progress().ok()?;
        memo.push(terms);
    }
    Some(memo)
}

/// Canonicalizes only the already-reviewed central constant scalar factors in
/// each selected ordered monomial. Noncentral factor order, multiplicity, and
/// sign are preserved. The replacement is committed only after every checked
/// spine and progress step succeeds.
fn canonicalize_central_constant_scalar_spines(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    terms: &mut Vec<(Box<[Id]>, bool)>,
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<()> {
    let mut canonical = Vec::new();
    canonical.try_reserve_exact(terms.len()).ok()?;
    for (spine, negative) in terms.iter() {
        let mut central = Vec::new();
        let mut noncentral = Vec::new();
        for factor in spine {
            progress().ok()?;
            let factor = egraph.find(*factor);
            if is_central_constant_scalar(egraph, factor, progress)? {
                central.try_reserve(1).ok()?;
                central.push(factor);
            } else {
                noncentral.try_reserve(1).ok()?;
                noncentral.push(factor);
            }
        }
        central.sort_unstable();
        central.try_reserve(noncentral.len()).ok()?;
        central.extend(noncentral);
        canonical.try_reserve(1).ok()?;
        canonical.push((central.into_boxed_slice(), *negative));
    }
    *terms = canonical;
    Some(())
}

/// Collects every independently valid Switch hoist from one immutable selected
/// snapshot. Canonical origins occur at most once. The root polynomial plan is
/// deliberately deferred until that snapshot contains no pending Switch plan.
pub(crate) fn selected_polynomial_redexes(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    expression: &RecExpr<MxxLang>,
    origins: &[Id],
    root_index: usize,
    monomials: &[Vec<(Box<[Id]>, bool)>],
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<Vec<(Id, ReplacementPlan)>> {
    if origins.len() != expression.as_ref().len() {
        return None;
    }
    let mut seen_origins = BTreeSet::new();
    let mut redexes = Vec::new();
    for (index, node) in expression.as_ref().iter().enumerate() {
        progress().ok()?;
        if matches!(node, MxxLang::Switch(_)) &&
            let Some(redex) =
                selected_switch_hoist_plan(egraph, origins, index, node, monomials, progress) &&
            seen_origins.insert(redex.0)
        {
            redexes.try_reserve(1).ok()?;
            redexes.push(redex);
        }
    }
    if !redexes.is_empty() {
        return Some(redexes);
    }
    let mut selected_indices = BTreeMap::<Id, Vec<usize>>::new();
    let mut selected_switches = BTreeMap::<Id, Option<usize>>::new();
    for (index, node) in expression.as_ref().iter().enumerate() {
        progress().ok()?;
        let origin = egraph.find(*origins.get(index)?);
        selected_indices.entry(origin).or_default().push(index);
        if matches!(node, MxxLang::Switch(_)) {
            selected_switches
                .entry(origin)
                .and_modify(|stored| *stored = None)
                .or_insert(Some(index));
        }
    }
    let mut seen_add_origins = BTreeSet::new();
    let mut accepted_in_additive_subtree = vec![false; expression.as_ref().len()];
    for (index, node) in expression.as_ref().iter().enumerate() {
        progress().ok()?;
        match node {
            MxxLang::MatrixNegate([child]) => {
                accepted_in_additive_subtree[index] =
                    accepted_in_additive_subtree[usize::from(*child)];
            }
            MxxLang::MatrixAdd(children) => {
                let descendant_accepted =
                    children.iter().any(|child| accepted_in_additive_subtree[usize::from(*child)]);
                if descendant_accepted {
                    accepted_in_additive_subtree[index] = true;
                    continue;
                }
                let origin = egraph.find(*origins.get(index)?);
                if let Some(plan) = selected_same_selector_add_hoist_plan(
                    egraph,
                    expression,
                    origins,
                    &selected_indices,
                    &selected_switches,
                    index,
                    monomials,
                    progress,
                )? && !replacement_plan_satisfied(egraph, origin, &plan)
                {
                    accepted_in_additive_subtree[index] = true;
                    if seen_add_origins.insert(origin) {
                        redexes.try_reserve(1).ok()?;
                        redexes.push((origin, plan));
                    }
                }
            }
            _ => {}
        }
    }
    if !redexes.is_empty() {
        return Some(redexes);
    }
    if !matches!(
        expression.as_ref().get(root_index)?,
        MxxLang::MatrixAdd(_) | MxxLang::MatrixNegate(_) | MxxLang::MatrixMultiply(_)
    ) {
        return Some(redexes);
    }
    let origin = egraph.find(*origins.get(root_index)?);
    progress().ok()?;
    let plan = signed_spines_replacement_plan(monomials.get(root_index)?, None)?;
    if !replacement_plan_satisfied(egraph, origin, &plan) {
        redexes.push((origin, plan));
    }
    Some(redexes)
}

/// Removes selector scope from a signed sum of one-Switch monomials only when
/// the already-selected polynomial in every corresponding case is identical.
/// Unlike a pointwise Switch merge, the replacement contains no Switch.
fn selected_subtree_contains_origin(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    expression: &RecExpr<MxxLang>,
    origins: &[Id],
    root_index: usize,
    target: Id,
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<bool> {
    let mut work = vec![(root_index, false)];
    let mut active = HashSet::new();
    let mut complete = HashSet::new();
    while let Some((index, exiting)) = work.pop() {
        progress().ok()?;
        if exiting {
            active.remove(&index);
            complete.insert(index);
            continue;
        }
        if complete.contains(&index) {
            continue;
        }
        if !active.insert(index) {
            return None;
        }
        if egraph.find(*origins.get(index)?) == target {
            return Some(true);
        }
        work.try_reserve(1).ok()?;
        work.push((index, true));
        let node = expression.as_ref().get(index)?;
        work.try_reserve(node.children().len()).ok()?;
        for child in node.children().iter().rev() {
            work.push((usize::from(*child), false));
        }
    }
    Some(false)
}

fn selected_context_has_independent_representative(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    expression: &RecExpr<MxxLang>,
    origins: &[Id],
    context_indices: &[usize],
    selector_origin: Id,
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<bool> {
    if context_indices.is_empty() {
        return Some(false);
    }
    for context_index in context_indices {
        progress().ok()?;
        if !selected_subtree_contains_origin(
            egraph,
            expression,
            origins,
            *context_index,
            selector_origin,
            progress,
        )? {
            return Some(true);
        }
    }
    Some(false)
}

const SELECTED_MISMATCH_SPINE_LIMIT: usize = 6;
const SELECTED_MISMATCH_FACTOR_LIMIT: usize = 8;
const SELECTED_MISMATCH_FACTOR_VIEW_LIMIT: usize = 12;

fn selected_mismatch_spines(
    terms: &[(Box<[Id]>, bool)],
) -> (Vec<(bool, Box<[usize]>, usize, usize)>, usize) {
    let mut retained = Vec::new();
    let mut omitted = 0usize;
    let mut index = 0usize;
    while index < terms.len() {
        let (factors, negative) = &terms[index];
        let mut end = index + 1;
        while end < terms.len() && terms[end] == terms[index] {
            end += 1;
        }
        if retained.len() < SELECTED_MISMATCH_SPINE_LIMIT {
            retained.push((
                *negative,
                factors
                    .iter()
                    .take(SELECTED_MISMATCH_FACTOR_LIMIT)
                    .map(|factor| usize::from(*factor))
                    .collect(),
                factors.len().saturating_sub(SELECTED_MISMATCH_FACTOR_LIMIT),
                end - index,
            ));
        } else {
            omitted += end - index;
        }
        index = end;
    }
    (retained, omitted)
}

fn selected_mismatch_factor_views(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    baseline: &[(Box<[Id]>, bool)],
    differing: &[(Box<[Id]>, bool)],
) -> (Vec<LeafView>, usize) {
    let baseline_counts = baseline.iter().fold(BTreeMap::new(), |mut counts, term| {
        *counts.entry(term).or_insert(0usize) += 1;
        counts
    });
    let differing_counts = differing.iter().fold(BTreeMap::new(), |mut counts, term| {
        *counts.entry(term).or_insert(0usize) += 1;
        counts
    });
    let mut factors = BTreeSet::new();
    for term in baseline_counts.keys().chain(differing_counts.keys()) {
        if baseline_counts.get(term) != differing_counts.get(term) {
            factors.extend(term.0.iter().map(|factor| egraph.find(*factor)));
        }
    }
    let omitted = factors.len().saturating_sub(SELECTED_MISMATCH_FACTOR_VIEW_LIMIT);
    (
        factors
            .into_iter()
            .take(SELECTED_MISMATCH_FACTOR_VIEW_LIMIT)
            .map(|factor| leaf_view(egraph, factor))
            .collect(),
        omitted,
    )
}

fn selected_spine_peel_term(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    spine: &[Id],
    negative: bool,
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<PeelTerm> {
    match spine {
        [] => None,
        [base] => Some(PeelTerm::Concrete { base: egraph.find(*base), negative }),
        factors => {
            let factors = copy_ids(factors, progress)?;
            if let Some(product) = egraph.lookup(MxxLang::MatrixMultiply(factors.clone())) {
                return Some(PeelTerm::Concrete { base: egraph.find(product), negative });
            }
            let (&base, suffix) = factors.split_first()?;
            Some(PeelTerm::ProductFactor {
                prefix: Box::default(),
                terms: vec![(base, false)],
                suffix: suffix.to_vec().into_boxed_slice(),
                negative,
            })
        }
    }
}

fn selected_peel_terms_to_spines(
    terms: &[PeelTerm],
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<Vec<(Box<[Id]>, bool)>> {
    let mut spines = Vec::new();
    for term in terms {
        match term {
            PeelTerm::Concrete { base, negative } => {
                progress().ok()?;
                spines.try_reserve(1).ok()?;
                spines.push((vec![*base].into_boxed_slice(), *negative));
            }
            PeelTerm::ProductFactor { prefix, terms, suffix, negative } => {
                for (base, base_negative) in terms {
                    progress().ok()?;
                    let mut spine = Vec::new();
                    spine.try_reserve_exact(prefix.len() + 1 + suffix.len()).ok()?;
                    spine.extend(prefix.iter().copied());
                    spine.push(*base);
                    spine.extend(suffix.iter().copied());
                    spines.try_reserve(1).ok()?;
                    spines.push((spine.into_boxed_slice(), *negative != *base_negative));
                }
            }
        }
    }
    Some(spines)
}

fn selected_same_selector_add_hoist_plan(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    expression: &RecExpr<MxxLang>,
    origins: &[Id],
    selected_indices: &BTreeMap<Id, Vec<usize>>,
    selected_switches: &BTreeMap<Id, Option<usize>>,
    root_index: usize,
    monomials: &[Vec<(Box<[Id]>, bool)>],
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<Option<ReplacementPlan>> {
    if !matches!(expression.as_ref().get(root_index)?, MxxLang::MatrixAdd(_)) {
        return Some(None);
    }
    let root_origin = egraph.find(*origins.get(root_index)?);
    if !matches!(egraph[root_origin].data.sort, Ok(MxxSort::Matrix(_))) {
        return Some(None);
    }
    let root_terms = monomials.get(root_index)?;
    let mut groups = BTreeMap::<
        Id,
        (Option<usize>, bool, Vec<(usize, Box<[Id]>, bool, Box<[Id]>, Box<[Id]>)>),
    >::new();
    let diagnose = tracing::enabled!(tracing::Level::DEBUG);
    let mut rejected_by_selector = BTreeMap::<
        Id,
        Vec<(
            usize,
            bool,
            Vec<usize>,
            usize,
            usize,
            usize,
            usize,
            Vec<(usize, usize)>,
            &'static str,
            Option<(usize, usize, usize)>,
        )>,
    >::new();

    for (ordinal, (spine, negative)) in root_terms.iter().enumerate() {
        progress().ok()?;
        let mut occurrences = Vec::new();
        let mut unresolved = false;
        for (position, factor) in spine.iter().enumerate() {
            progress().ok()?;
            let factor = egraph.find(*factor);
            if let Some(selected_index) = selected_switches.get(&factor) {
                let Some(selected_index) = *selected_index else {
                    if diagnose {
                        let mut alternatives = BTreeSet::new();
                        let mut switch_occurrence_count = 0usize;
                        if let Some(indices) = selected_indices.get(&factor) {
                            for index in indices {
                                if let Some(MxxLang::Switch(switch)) =
                                    expression.as_ref().get(*index)
                                {
                                    let Some((&selector, cases)) = switch.split_first() else {
                                        continue;
                                    };
                                    switch_occurrence_count += 1;
                                    alternatives.insert((
                                        egraph.find(*origins.get(usize::from(selector))?),
                                        cases.len(),
                                    ));
                                }
                            }
                        }
                        if alternatives.len() == 1 &&
                            let Some(&(selector, case_count)) = alternatives.iter().next()
                        {
                            rejected_by_selector.entry(selector).or_default().push((
                                ordinal,
                                *negative,
                                spine
                                    .iter()
                                    .take(8)
                                    .map(|factor| usize::from(egraph.find(*factor)))
                                    .collect(),
                                spine.len().saturating_sub(8),
                                switch_occurrence_count,
                                usize::from(factor),
                                case_count,
                                Vec::new(),
                                "duplicate-selected-switch-occurrences",
                                None,
                            ));
                        }
                    }
                    unresolved = true;
                    break;
                };
                let Some(MxxLang::Switch(switch)) = expression.as_ref().get(selected_index) else {
                    unresolved = true;
                    break;
                };
                let Some(selector) = switch.first() else {
                    unresolved = true;
                    break;
                };
                occurrences.push((
                    position,
                    factor,
                    selected_index,
                    egraph.find(*origins.get(usize::from(*selector))?),
                ));
            }
        }
        if unresolved || occurrences.is_empty() {
            continue;
        }
        let mut visited_selectors = BTreeSet::new();
        for (switch_position, switch_origin, switch_index, selector_origin) in &occurrences {
            progress().ok()?;
            if !visited_selectors.insert(*selector_origin) ||
                occurrences.iter().filter(|occurrence| occurrence.3 == *selector_origin).count() !=
                    1
            {
                continue;
            }
            let MxxLang::Switch(switch) = expression.as_ref().get(*switch_index)? else { continue };
            let (_, cases) = switch.split_first()?;
            if cases.is_empty() {
                continue;
            }
            let mut canonical_cases = Vec::new();
            canonical_cases.try_reserve_exact(cases.len()).ok()?;
            for case in cases {
                progress().ok()?;
                canonical_cases.push(egraph.find(*origins.get(usize::from(*case))?));
            }
            let canonical_cases = canonical_cases.into_boxed_slice();
            let prefix = copy_ids(&spine[..*switch_position], progress)?;
            let suffix = copy_ids(&spine[*switch_position + 1..], progress)?;
            let mut context_valid = true;
            let mut context_occurrences = Vec::new();
            let mut context_rejection = None;
            for factor in prefix.iter().chain(suffix.iter()) {
                progress().ok()?;
                let factor = egraph.find(*factor);
                let Some(context_indices) = selected_indices.get(&factor) else {
                    context_valid = false;
                    context_rejection = Some("missing-context-origin");
                    context_occurrences.push((usize::from(factor), 0));
                    break;
                };
                context_occurrences.push((usize::from(factor), context_indices.len()));
                if !selected_context_has_independent_representative(
                    egraph,
                    expression,
                    origins,
                    context_indices,
                    *selector_origin,
                    progress,
                )? {
                    context_valid = false;
                    context_rejection = Some("all-context-representatives-selector-dependent");
                    break;
                }
            }
            if !context_valid {
                if diagnose {
                    rejected_by_selector.entry(*selector_origin).or_default().push((
                        ordinal,
                        *negative,
                        spine
                            .iter()
                            .take(8)
                            .map(|factor| usize::from(egraph.find(*factor)))
                            .collect(),
                        spine.len().saturating_sub(8),
                        1,
                        usize::from(*switch_origin),
                        canonical_cases.len(),
                        context_occurrences,
                        context_rejection.unwrap_or("context-rejected"),
                        None,
                    ));
                }
                continue;
            }
            let mut offending_case = None;
            let switch_sort = match &egraph[*switch_origin].data.sort {
                Ok(MxxSort::Matrix(matrix)) => Some(matrix),
                _ => None,
            };
            let rejection = if egraph[*selector_origin].data.sort != Ok(MxxSort::Int) {
                Some("selector-sort-mismatch")
            } else if !selector_domain_matches_cases(
                egraph,
                *selector_origin,
                canonical_cases.len(),
            ) {
                Some("selector-domain-case-count-mismatch")
            } else if *switch_origin == root_origin {
                Some("switch-root-cycle")
            } else if switch_sort.is_none() {
                Some("switch-output-sort-mismatch")
            } else {
                let switch_sort = switch_sort.expect("checked above");
                canonical_cases.iter().enumerate().find_map(|(case_ordinal, case)| {
                    let reason = if *case == root_origin || *case == *switch_origin {
                        Some("case-cycle")
                    } else {
                        match &egraph[*case].data.sort {
                            Ok(MxxSort::Matrix(case_sort))
                                if matrix_types_equal(case_sort, switch_sort) =>
                            {
                                let count = selected_indices.get(case).map_or(0, Vec::len);
                                (count != 1).then_some(if count == 0 {
                                    "missing-case-selected-index"
                                } else {
                                    "duplicate-case-selected-indices"
                                })
                            }
                            _ => Some("case-sort-mismatch"),
                        }
                    };
                    reason.map(|reason| {
                        offending_case = Some((
                            case_ordinal,
                            usize::from(*case),
                            selected_indices.get(case).map_or(0, Vec::len),
                        ));
                        reason
                    })
                })
            };
            let valid = rejection.is_none();
            if diagnose && let Some(rejection) = rejection {
                rejected_by_selector.entry(*selector_origin).or_default().push((
                    ordinal,
                    *negative,
                    spine.iter().take(8).map(|factor| usize::from(egraph.find(*factor))).collect(),
                    spine.len().saturating_sub(8),
                    1,
                    usize::from(*switch_origin),
                    canonical_cases.len(),
                    context_occurrences,
                    rejection,
                    offending_case,
                ));
            }
            let entry = groups.entry(*selector_origin).or_insert((
                Some(canonical_cases.len()),
                true,
                Vec::new(),
            ));
            entry.1 &= valid && entry.0 == Some(canonical_cases.len());
            entry.2.try_reserve(1).ok()?;
            entry.2.push((ordinal, canonical_cases, *negative, prefix, suffix));
        }
    }

    if diagnose {
        for (selector, (_, _, participants)) in &groups {
            let Some(rejected) = rejected_by_selector.get(selector) else { continue };
            let rejected_after_context = rejected
                .iter()
                .filter(|entry| {
                    matches!(
                        entry.8,
                        "selector-sort-mismatch" |
                            "selector-domain-case-count-mismatch" |
                            "switch-root-cycle" |
                            "switch-output-sort-mismatch" |
                            "case-cycle" |
                            "case-sort-mismatch" |
                            "missing-case-selected-index" |
                            "duplicate-case-selected-indices"
                    )
                })
                .count();
            let accepted_participant_count =
                participants.len().saturating_sub(rejected_after_context);
            if accepted_participant_count < 2 || rejected.is_empty() {
                continue;
            }
            tracing::debug!(
                event = "operational_selected_cross_switch_rejected_monomials",
                add_origin = usize::from(root_origin),
                selector = usize::from(*selector),
                accepted_participant_count,
                rejected = ?rejected.iter().take(4).collect::<Vec<_>>(),
                omitted_rejected_count = rejected.len().saturating_sub(4),
                "same-selector Add has rejected selected monomials"
            );
        }
    }

    let mut replacements = BTreeMap::<usize, Option<ReplacementPlan>>::new();
    for (selector_origin, (case_count, valid, participants)) in groups {
        progress().ok()?;
        if !valid || participants.len() < 2 {
            continue;
        }
        let case_count = case_count?;
        let participant_ordinals =
            participants.iter().map(|(ordinal, ..)| *ordinal).collect::<BTreeSet<_>>();
        let mut exterior = Vec::new();
        let mut candidate_valid = true;
        let mut whole_rejection = None;
        for (ordinal, (spine, negative)) in root_terms.iter().enumerate() {
            progress().ok()?;
            let mut switch_count = 0usize;
            let mut same_selector = false;
            let mut switch_occurrences = Vec::new();
            for factor in spine {
                progress().ok()?;
                let factor = egraph.find(*factor);
                let Some(selected_switch) = selected_switches.get(&factor) else { continue };
                let Some(selected_switch) = *selected_switch else {
                    candidate_valid = false;
                    if switch_occurrences.len() < 4 {
                        for index in selected_indices.get(&factor).into_iter().flatten() {
                            let Some(MxxLang::Switch(children)) = expression.as_ref().get(*index)
                            else {
                                continue;
                            };
                            let Some(selector) = children.first() else { continue };
                            switch_occurrences.push((
                                usize::from(factor),
                                usize::from(egraph.find(*origins.get(usize::from(*selector))?)),
                                children.len().saturating_sub(1),
                                selected_indices.get(&factor).map_or(0, Vec::len),
                            ));
                            if switch_occurrences.len() == 4 {
                                break;
                            }
                        }
                    }
                    whole_rejection = Some((
                        "unresolved-selected-switch",
                        ordinal,
                        *negative,
                        std::mem::take(&mut switch_occurrences),
                    ));
                    break;
                };
                let MxxLang::Switch(children) = expression.as_ref().get(selected_switch)? else {
                    candidate_valid = false;
                    whole_rejection =
                        Some(("unresolved-selected-switch", ordinal, *negative, Vec::new()));
                    break;
                };
                let Some(selector) = children.first() else {
                    candidate_valid = false;
                    whole_rejection =
                        Some(("unresolved-selected-switch", ordinal, *negative, Vec::new()));
                    break;
                };
                let factor_selector = egraph.find(*origins.get(usize::from(*selector))?);
                if factor_selector == selector_origin {
                    switch_count += 1;
                    same_selector = true;
                    if switch_occurrences.len() < 4 {
                        switch_occurrences.push((
                            usize::from(factor),
                            usize::from(factor_selector),
                            children.len().saturating_sub(1),
                            selected_indices.get(&factor).map_or(0, Vec::len),
                        ));
                    }
                }
            }
            if !candidate_valid || switch_count > 1 {
                if candidate_valid {
                    whole_rejection =
                        Some(("multiple-switches", ordinal, *negative, switch_occurrences));
                }
                candidate_valid = false;
                break;
            }
            if switch_count == 1 {
                if !same_selector || !participant_ordinals.contains(&ordinal) {
                    candidate_valid = false;
                    whole_rejection = Some((
                        "unresolved-selected-switch",
                        ordinal,
                        *negative,
                        switch_occurrences,
                    ));
                    break;
                }
                continue;
            }
            for factor in spine {
                let factor = egraph.find(*factor);
                let Some(indices) = selected_indices.get(&factor) else {
                    candidate_valid = false;
                    whole_rejection =
                        Some(("missing-selected-representation", ordinal, *negative, Vec::new()));
                    break;
                };
                if !selected_context_has_independent_representative(
                    egraph,
                    expression,
                    origins,
                    indices,
                    selector_origin,
                    progress,
                )? {
                    candidate_valid = false;
                    whole_rejection = Some(("dependent-exterior", ordinal, *negative, Vec::new()));
                    break;
                }
            }
            if !candidate_valid {
                break;
            }
            exterior.try_reserve(1).ok()?;
            exterior.push((ordinal, spine.clone(), *negative));
        }
        if !candidate_valid {
            if diagnose && participants.len() >= 2 {
                let (reason, ordinal, negative, switch_occurrences) =
                    whole_rejection.unwrap_or(("whole-candidate-invalid", 0, false, Vec::new()));
                let (spine, _) = root_terms.get(ordinal)?;
                tracing::debug!(
                    event = "operational_selected_full_case_candidate_rejected",
                    add_origin = usize::from(root_origin),
                    selector = usize::from(selector_origin),
                    reason,
                    monomial_ordinal = ordinal,
                    negative,
                    spine = ?spine
                        .iter()
                        .take(8)
                        .map(|factor| usize::from(egraph.find(*factor)))
                        .collect::<Vec<_>>(),
                    spine_omitted = spine.len().saturating_sub(8),
                    switch_occurrences = ?switch_occurrences,
                    participant_count = participants.len(),
                    exterior_count = exterior.len(),
                    total_monomial_count = root_terms.len(),
                    "selected full-case candidate rejects one whole-root monomial"
                );
            }
            continue;
        }
        // Cases are streamed.  The existing peel matcher is linear in stored
        // cases and, per case, worst-case quadratic in actual versus fixed
        // spines (times their ordered-factor traversal); no selector product
        // or case polynomial is retained.
        let mut common_result: Option<Vec<(Box<[Id]>, bool)>> = None;
        let mut zero_witness = None;
        let mut peeled_case_count = 0usize;
        let mut total_actual_count = 0usize;
        let mut total_fixed_count = 0usize;
        let mut total_unmatched_fixed_count = 0usize;
        for case_ordinal in 0..case_count {
            let mut combined = Vec::new();
            for (_, cases, outer_negative, prefix, suffix) in &participants {
                progress().ok()?;
                let case = *cases.get(case_ordinal)?;
                let Some([selected_case_index]) = selected_indices.get(&case).map(Vec::as_slice)
                else {
                    return Some(None);
                };
                zero_witness.get_or_insert(case);
                for (case_spine, case_negative) in monomials.get(*selected_case_index)? {
                    progress().ok()?;
                    let length =
                        prefix.len().checked_add(case_spine.len())?.checked_add(suffix.len())?;
                    let mut contextual = Vec::new();
                    contextual.try_reserve_exact(length).ok()?;
                    contextual.extend(prefix.iter().copied());
                    contextual.extend(case_spine.iter().copied());
                    contextual.extend(suffix.iter().copied());
                    combined.try_reserve(1).ok()?;
                    combined
                        .push((contextual.into_boxed_slice(), *case_negative != *outer_negative));
                }
            }
            for (_, spine, negative) in &exterior {
                progress().ok()?;
                combined.try_reserve(1).ok()?;
                combined.push((copy_ids(spine, progress)?, *negative));
            }
            canonicalize_central_constant_scalar_spines(egraph, &mut combined, progress)?;
            cancel_signed_spines(&mut combined);
            let mut actual = Vec::new();
            let mut fixed_spines = Vec::new();
            for (spine, negative) in combined {
                progress().ok()?;
                let term = selected_spine_peel_term(egraph, &spine, negative, progress)?;
                let peelable = match &term {
                    PeelTerm::Concrete { base, .. } => {
                        factor_can_peel_fixed_target(egraph, *base, progress)?
                    }
                    PeelTerm::ProductFactor { terms, .. } => {
                        let mut peelable = false;
                        for (base, _) in terms {
                            if factor_can_peel_fixed_target(egraph, *base, progress)? {
                                peelable = true;
                                break;
                            }
                        }
                        peelable
                    }
                };
                if peelable {
                    actual.try_reserve(1).ok()?;
                    actual.push(term);
                } else {
                    fixed_spines.try_reserve(1).ok()?;
                    fixed_spines.push((spine, negative));
                }
            }
            let actual_count = actual.len();
            let fixed_count = fixed_spines.len();
            let (peeled, unmatched_fixed) =
                peel_fixed_targets(egraph, &mut actual, &fixed_spines, progress)?;
            let unmatched_fixed_count = unmatched_fixed.len();
            peeled_case_count += usize::from(peeled);
            total_actual_count = total_actual_count.checked_add(actual_count)?;
            total_fixed_count = total_fixed_count.checked_add(fixed_count)?;
            total_unmatched_fixed_count =
                total_unmatched_fixed_count.checked_add(unmatched_fixed_count)?;
            let mut combined = selected_peel_terms_to_spines(&actual, progress)?;
            for term in unmatched_fixed {
                progress().ok()?;
                combined.try_reserve(1).ok()?;
                combined.push(term);
            }
            canonicalize_central_constant_scalar_spines(egraph, &mut combined, progress)?;
            cancel_signed_spines(&mut combined);
            if let Some(previous) = &common_result {
                if previous != &combined {
                    if tracing::enabled!(tracing::Level::DEBUG) {
                        let contexts = participants
                            .iter()
                            .map(|(_, _, negative, prefix, suffix)| {
                                (*negative, prefix.len(), suffix.len())
                            })
                            .collect::<Vec<_>>();
                        let (baseline_spines, baseline_omitted) =
                            selected_mismatch_spines(previous);
                        let (differing_spines, differing_omitted) =
                            selected_mismatch_spines(&combined);
                        let (differing_factor_views, differing_factor_views_omitted) =
                            selected_mismatch_factor_views(egraph, previous, &combined);
                        tracing::debug!(
                            event = "operational_selected_full_case_residual_mismatch",
                            add_origin = usize::from(root_origin),
                            selector = usize::from(selector_origin),
                            participant_count = participants.len(),
                            exterior_count = exterior.len(),
                            peeled,
                            actual_count,
                            fixed_count,
                            unmatched_fixed_count,
                            case_count,
                            baseline_case = 0,
                            differing_case = case_ordinal,
                            contexts = ?contexts,
                            baseline_monomial_count = previous.len(),
                            differing_monomial_count = combined.len(),
                            baseline_spines = ?baseline_spines,
                            differing_spines = ?differing_spines,
                            baseline_omitted,
                            differing_omitted,
                            differing_factor_views = ?differing_factor_views,
                            differing_factor_views_omitted,
                            "selected same-selector full-case residuals differ"
                        );
                    }
                    common_result = None;
                    break;
                }
            } else {
                common_result = Some(combined);
            }
        }
        let Some(common_result) = common_result else {
            continue;
        };
        tracing::debug!(
            event = "operational_selected_full_case_residual_equal",
            add_origin = usize::from(root_origin),
            selector = usize::from(selector_origin),
            participant_count = participants.len(),
            exterior_count = exterior.len(),
            case_count,
            peeled_case_count,
            total_actual_count,
            total_fixed_count,
            total_unmatched_fixed_count,
            residual_monomial_count = common_result.len(),
            "selected same-selector cases have one complete residual"
        );
        let plan = if common_result.is_empty() {
            let witness = ReplacementPlan::Existing(zero_witness?);
            ReplacementPlan::Add(
                vec![witness.clone(), ReplacementPlan::Negate(Box::new(witness))]
                    .into_boxed_slice(),
            )
        } else {
            signed_spines_replacement_plan(&common_result, None)?
        };
        let mut replaced_ordinals = participant_ordinals;
        replaced_ordinals.extend(exterior.iter().map(|(ordinal, ..)| *ordinal));
        let first = *replaced_ordinals.first()?;
        if replaced_ordinals.iter().any(|ordinal| replacements.contains_key(ordinal)) {
            return Some(None);
        }
        replacements.insert(first, Some(plan));
        for ordinal in replaced_ordinals {
            if ordinal != first {
                replacements.insert(ordinal, None);
            }
        }
    }
    if replacements.is_empty() {
        return Some(None);
    }
    let mut terms = Vec::new();
    for (ordinal, (spine, negative)) in root_terms.iter().enumerate() {
        progress().ok()?;
        match replacements.get(&ordinal) {
            Some(Some(plan)) => terms.push(plan.clone()),
            Some(None) => {}
            None => terms.push(signed_spine_replacement_plan(spine, *negative)?),
        }
    }
    Some(Some(ReplacementPlan::Add(terms.into_boxed_slice())))
}

fn signed_spine_replacement_plan(factors: &[Id], negative: bool) -> Option<ReplacementPlan> {
    let product = existing_product_plan(factors);
    Some(if negative { ReplacementPlan::Negate(Box::new(product)) } else { product })
}

fn signed_spines_replacement_plan(
    terms: &[(Box<[Id]>, bool)],
    zero_witness: Option<&ReplacementPlan>,
) -> Option<ReplacementPlan> {
    let terms = terms
        .iter()
        .map(|(factors, negative)| signed_spine_replacement_plan(factors, *negative))
        .collect::<Option<Vec<_>>>()?;
    match terms.len() {
        0 => {
            let witness = zero_witness?.clone();
            Some(ReplacementPlan::Add(
                vec![witness.clone(), ReplacementPlan::Negate(Box::new(witness))]
                    .into_boxed_slice(),
            ))
        }
        1 => terms.into_iter().next(),
        _ => Some(ReplacementPlan::Add(terms.into_boxed_slice())),
    }
}

fn selected_switch_hoist_plan(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    origins: &[Id],
    switch_index: usize,
    node: &MxxLang,
    monomials: &[Vec<(Box<[Id]>, bool)>],
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<(Id, ReplacementPlan)> {
    let MxxLang::Switch(children) = node else { return None };
    let (&selector, cases) = children.split_first()?;
    if cases.is_empty() ||
        !selector_domain_matches_cases(
            egraph,
            egraph.find(*origins.get(usize::from(selector))?),
            cases.len(),
        )
    {
        return None;
    }
    let origin = egraph.find(*origins.get(switch_index)?);
    let mut case_terms = Vec::new();
    for case in cases {
        progress().ok()?;
        case_terms.try_reserve(1).ok()?;
        case_terms.push(copy_signed_spines(monomials.get(usize::from(*case))?, progress)?);
    }
    for terms in &mut case_terms {
        progress().ok()?;
        cancel_signed_spines(terms);
    }

    let mut common = copy_signed_spines(case_terms.first()?, progress)?;
    for terms in &case_terms[1..] {
        common = sorted_signed_spine_intersection(&common, terms, progress)?;
    }
    for terms in &mut case_terms {
        *terms = sorted_signed_spine_subtraction(terms, &common, progress)?;
    }

    if case_terms.iter().all(Vec::is_empty) {
        let plan = signed_spines_replacement_plan(&common, None)?;
        return (!replacement_plan_satisfied(egraph, origin, &plan)).then_some((origin, plan));
    }

    let mut residual_spines = Vec::new();
    for (spine, _) in case_terms.iter().flatten() {
        progress().ok()?;
        residual_spines.try_reserve(1).ok()?;
        residual_spines.push(spine);
    }
    // An empty residual case is a zero of the complete Switch output shape.
    // It cannot serve as the middle zero of a factored product without a
    // separately proved intermediate shape, so only additive hoisting remains
    // eligible in that situation.
    let minimum_length = if case_terms.iter().any(Vec::is_empty) {
        0
    } else {
        residual_spines.iter().map(|spine| spine.len()).min().unwrap_or(0)
    };
    let mut prefix_length = 0;
    while prefix_length + 1 < minimum_length {
        let mut shared = true;
        for spine in &residual_spines {
            progress().ok()?;
            shared &= spine[prefix_length] == residual_spines[0][prefix_length];
        }
        if !shared {
            break;
        }
        prefix_length += 1;
    }
    let mut suffix_length = 0;
    while prefix_length + suffix_length + 1 < minimum_length {
        let mut shared = true;
        for spine in &residual_spines {
            progress().ok()?;
            shared &= spine[spine.len() - suffix_length - 1] ==
                residual_spines[0][residual_spines[0].len() - suffix_length - 1];
        }
        if !shared {
            break;
        }
        suffix_length += 1;
    }
    let prefix = copy_ids(
        residual_spines.first().map(|spine| &spine[..prefix_length]).unwrap_or_default(),
        progress,
    )?;
    let suffix = copy_ids(
        residual_spines
            .first()
            .map(|spine| &spine[spine.len() - suffix_length..])
            .unwrap_or_default(),
        progress,
    )?;
    if prefix.is_empty() && suffix.is_empty() && common.is_empty() {
        return None;
    }

    let common_plan = signed_spines_replacement_plan(&common, None);
    let zero_witness = common_plan.as_ref();
    let mut switch_cases = Vec::with_capacity(cases.len() + 1);
    switch_cases.push(ReplacementPlan::Existing(egraph.find(*origins.get(usize::from(selector))?)));
    for terms in &case_terms {
        let mut middle = Vec::new();
        for (spine, negative) in terms {
            progress().ok()?;
            middle.try_reserve(1).ok()?;
            let end = spine.len().checked_sub(suffix_length)?;
            middle.push((copy_ids(spine.get(prefix_length..end)?, progress)?, *negative));
        }
        switch_cases.push(signed_spines_replacement_plan(&middle, zero_witness)?);
    }
    let switch = ReplacementPlan::Switch(switch_cases.into_boxed_slice());
    let switched = splice_product_plan(&prefix, &[switch], &suffix);
    let plan = match common_plan {
        Some(common) => ReplacementPlan::Add(vec![switched, common].into_boxed_slice()),
        None => switched,
    };
    (!replacement_plan_satisfied(egraph, origin, &plan)).then_some((origin, plan))
}

fn copy_signed_spines(
    terms: &[(Box<[Id]>, bool)],
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<Vec<(Box<[Id]>, bool)>> {
    let mut output = Vec::new();
    for (spine, negative) in terms {
        progress().ok()?;
        output.try_reserve(1).ok()?;
        output.push((copy_ids(spine, progress)?, *negative));
    }
    Some(output)
}

fn sorted_signed_spine_intersection(
    left: &[(Box<[Id]>, bool)],
    right: &[(Box<[Id]>, bool)],
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<Vec<(Box<[Id]>, bool)>> {
    let mut output = Vec::new();
    let (mut left_index, mut right_index) = (0, 0);
    while left_index < left.len() && right_index < right.len() {
        progress().ok()?;
        match left[left_index].cmp(&right[right_index]) {
            std::cmp::Ordering::Less => left_index += 1,
            std::cmp::Ordering::Greater => right_index += 1,
            std::cmp::Ordering::Equal => {
                output.try_reserve(1).ok()?;
                output.push((copy_ids(&left[left_index].0, progress)?, left[left_index].1));
                left_index += 1;
                right_index += 1;
            }
        }
    }
    Some(output)
}

fn sorted_signed_spine_subtraction(
    terms: &[(Box<[Id]>, bool)],
    removed: &[(Box<[Id]>, bool)],
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<Vec<(Box<[Id]>, bool)>> {
    let mut output = Vec::new();
    let mut removed_index = 0;
    for term in terms {
        progress().ok()?;
        if removed_index < removed.len() && *term == removed[removed_index] {
            removed_index += 1;
        } else {
            output.try_reserve(1).ok()?;
            output.push((copy_ids(&term.0, progress)?, term.1));
        }
    }
    (removed_index == removed.len()).then_some(output)
}

pub(crate) fn materialize_selected_polynomial_redex(
    egraph: &mut EGraph<MxxLang, MxxAnalysis>,
    (origin, plan): (Id, ReplacementPlan),
    context: &RewriteContext,
) -> Option<(Id, Id)> {
    context.note_candidate();
    let replacement = materialize_replacement_plan(egraph, context, &plan)?;
    Some((origin, replacement))
}

#[cfg(test)]
fn pointwise_add_switch_cancellation_result(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    root: Id,
) -> Result<PointwiseAddSwitchPlan, PointwiseAddSwitchReject> {
    let root = egraph.find(root);
    let structures = pointwise_add_switch_structures(egraph, root);
    let PointwiseAddSwitchStructures {
        physical_root_adds,
        direct_switch_children,
        direct_grouped_add_children,
        eligible,
    } = structures;
    let eligible_single_switch_adds = eligible.len();
    let structure = eligible.into_iter().next().ok_or(PointwiseAddSwitchReject::Structural {
        physical_root_adds,
        eligible_single_switch_adds,
        direct_switch_children,
        direct_grouped_add_children,
    })?;
    pointwise_add_switch_cancellation_for_structure(
        egraph,
        root,
        &structure,
        physical_root_adds,
        eligible_single_switch_adds,
    )
}

fn pointwise_add_switch_cancellation_for_structure(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    root: Id,
    structure: &PointwiseAddSwitchStructure,
    physical_root_adds: usize,
    eligible_single_switch_adds: usize,
) -> Result<PointwiseAddSwitchPlan, PointwiseAddSwitchReject> {
    let root = egraph.find(root);
    let fixed = structure
        .terms
        .iter()
        .enumerate()
        .filter_map(|(index, term)| (index != structure.switch_index).then_some(*term))
        .collect::<Vec<_>>();
    if fixed.is_empty() || fixed.iter().any(|term| has_physical_switch(egraph, *term)) {
        return Err(PointwiseAddSwitchReject::FixedTermsEmptyOrSwitch);
    }
    let mut fixed_signed = HashMap::<(Id, bool), (usize, usize, usize)>::new();
    for &term in &fixed {
        let Some(signed) = signed_additive_term(egraph, term) else {
            return Err(PointwiseAddSwitchReject::FixedTermsEmptyOrSwitch);
        };
        fixed_signed.entry(signed).or_insert((0, 0, 0)).0 += 1;
    }
    let fixed_count = fixed.len();
    let mut normalized_cases = Vec::with_capacity(structure.switch.len() - 1);
    for (case_epoch, case) in structure.switch[1..].iter().enumerate() {
        let flattened = flatten_pointwise_case(
            egraph,
            root,
            structure.switch[0],
            structure.switch.len() - 1,
            case_epoch,
            *case,
        )
        .map_err(|_| PointwiseAddSwitchReject::CaseCycleOrNestedSwitch {
            case_index: case_epoch,
        })?;
        let case_physical_adds = flattened.physical_adds;
        let case_grouped_add_children = flattened.grouped_add_children;
        let terms = flattened.terms;
        let mut consumed = vec![false; terms.len()];
        let mut matched = 0;
        for (index, term) in terms.iter().enumerate() {
            if let Some((required, seen_epoch, used)) =
                fixed_signed.get_mut(&(term.base, !term.negative))
            {
                if *seen_epoch != case_epoch + 1 {
                    *seen_epoch = case_epoch + 1;
                    *used = 0;
                }
                if *used < *required {
                    *used += 1;
                    matched += 1;
                    consumed[index] = true;
                }
            }
        }
        if matched != fixed_count {
            let diagnostic = unmatched_fixed_terms_diagnostic(
                egraph,
                &fixed,
                &terms,
                case_physical_adds,
                case_grouped_add_children,
            );
            return Err(PointwiseAddSwitchReject::UnmatchedFixedTerms {
                physical_root_adds,
                eligible_single_switch_adds,
                selector: egraph.find(structure.switch[0]),
                switch_cases: structure.switch.clone(),
                case_index: case_epoch,
                matched,
                required: fixed_count,
                direct_terms: diagnostic.direct_terms,
                negated_terms: diagnostic.negated_terms,
                fixed_unique_add_children: diagnostic.fixed_unique_add_children,
                case_physical_adds: diagnostic.case_physical_adds,
                case_grouped_add_children: diagnostic.case_grouped_add_children,
                fixed_terms: diagnostic.fixed_terms,
                fixed_terms_omitted_occurrences: diagnostic.fixed_terms_omitted_occurrences,
                case_terms: diagnostic.case_terms,
                case_terms_omitted_occurrences: diagnostic.case_terms_omitted_occurrences,
            });
        }
        let after_cross = terms
            .into_iter()
            .zip(consumed)
            .filter_map(|(term, consumed)| (!consumed).then_some(term))
            .collect::<Vec<_>>();
        let signed_after_cross =
            after_cross.iter().map(|term| (term.base, term.negative)).collect::<Vec<_>>();
        let (cancelled, _) = cancelled_signed_additive_terms(&signed_after_cross);
        let remaining = after_cross
            .into_iter()
            .zip(cancelled)
            .filter_map(|(term, cancelled)| (!cancelled).then_some(term))
            .collect::<Vec<_>>();
        if remaining.iter().any(|term| egraph.find(term.base) == root) {
            return Err(PointwiseAddSwitchReject::CaseCycleOrNestedSwitch {
                case_index: case_epoch,
            });
        }
        normalized_cases.push(remaining);
    }
    let selector = egraph.find(structure.switch[0]);
    if equivalent_signed_switch_exists(egraph, root, selector, &normalized_cases) {
        return Err(PointwiseAddSwitchReject::EquivalentResult);
    }
    Ok(PointwiseAddSwitchPlan { selector, cases: normalized_cases, binder_aware: None })
}

/// Plans one symbolic physical Add candidate.  It intentionally consults the
/// binder descriptor rather than the e-class integer-domain hull: the
/// descriptor is the owner-authoritative physical stored-case domain.
fn binder_aware_pointwise_add_switch_cancellation_for_structure(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    root: Id,
    structure: &PointwiseAddSwitchStructure,
) -> Result<BinderAwarePointwiseAddSwitchPlan, BinderPreflightReject> {
    let root = egraph.find(root);
    let fixed = structure
        .terms
        .iter()
        .enumerate()
        .filter_map(|(index, term)| (index != structure.switch_index).then_some(*term))
        .collect::<Vec<_>>();
    if fixed.is_empty() || fixed.iter().any(|term| has_physical_switch(egraph, *term)) {
        return Err(BinderPreflightReject::FixedTermsEmptyOrSwitch);
    }

    let selector = egraph.find(structure.switch[0]);
    let selector_nodes = &egraph[selector].nodes;
    let [MxxLang::IntBinder(binder)] = selector_nodes.as_slice() else {
        return Err(BinderPreflightReject::SelectorNotUniqueBinder);
    };
    let descriptor = egraph
        .analysis
        .symbols
        .binders
        .get(binder.0)
        .ok_or(BinderPreflightReject::MissingDescriptor { binder: *binder })?;
    let case_count = structure.switch.len().checked_sub(1).expect("eligible switch has one case");
    if descriptor.minimum != BigInt::zero() || descriptor.maximum != BigInt::from(case_count - 1) {
        return Err(BinderPreflightReject::DomainCaseCountMismatch { binder: *binder, case_count });
    }

    let mut base_indices = BTreeMap::<Id, usize>::new();
    let mut fixed_bases = Vec::new();
    let mut fixed_occurrences = Vec::with_capacity(fixed.len());
    let fixed_leaves =
        signed_additive_leaves(egraph, &fixed).ok_or(BinderPreflightReject::FixedSignedFlatten)?;
    for (base, negative) in fixed_leaves {
        let base_index = match base_indices.get(&base) {
            Some(index) => *index,
            None => {
                let index = fixed_bases.len();
                fixed_bases.push(base);
                base_indices.insert(base, index);
                index
            }
        };
        fixed_occurrences.push(FixedOccurrence { base_index, negative });
    }
    let mut case_terms = Vec::with_capacity(case_count);
    for (case_index, case) in structure.switch[1..].iter().enumerate() {
        let terms =
            match flatten_pointwise_case(egraph, root, selector, case_count, case_index, *case) {
                Ok(flattened) => flattened.terms,
                Err(PointwiseCaseFlattenReject::AmbiguousAdd) => {
                    return Err(BinderPreflightReject::CaseAmbiguous { case_index });
                }
                Err(PointwiseCaseFlattenReject::NestedSwitchOrCycle)
                    if egraph.find(*case) == root =>
                {
                    return Err(BinderPreflightReject::CaseSelfCycle { case_index });
                }
                Err(PointwiseCaseFlattenReject::NestedSwitchOrCycle) => {
                    return Err(BinderPreflightReject::CaseNestedSwitch { case_index });
                }
            };
        case_terms.push(terms.into_boxed_slice());
    }
    Ok(BinderAwarePointwiseAddSwitchPlan {
        binder: *binder,
        case_terms: case_terms.into_boxed_slice(),
        fixed_bases: fixed_bases.into_boxed_slice(),
        fixed_occurrences: fixed_occurrences.into_boxed_slice(),
    })
}

fn binder_preflight_ready(plan: &BinderAwarePointwiseAddSwitchPlan) -> BinderPreflightReady {
    let leaves = plan
        .fixed_occurrences
        .iter()
        .map(|occurrence| (plan.fixed_bases[occurrence.base_index], occurrence.negative))
        .collect::<Vec<_>>();
    let (fixed_terms, fixed_terms_omitted_occurrences) = signed_leaf_multiplicity_summary(&leaves);
    BinderPreflightReady {
        fixed_terms,
        fixed_terms_omitted_occurrences,
        unique_base_count: plan.fixed_bases.len(),
        case_count: plan.case_terms.len(),
    }
}

/// Inspects the same stable physical candidates used by the rewrite planner.
/// It is bounded independently of e-graph size and performs no construction.
pub(crate) fn pointwise_add_switch_probe(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    root: Id,
) -> PointwiseAddSwitchProbe {
    let root = egraph.find(root);
    let structures = pointwise_add_switch_structures(egraph, root);
    let eligible_single_switch_adds = structures.eligible.len();
    let outcomes = structures
        .eligible
        .iter()
        .take(MAX_POINTWISE_PROBE_STRUCTURES)
        .map(|structure| {
            let direct = match pointwise_add_switch_cancellation_for_structure(
                egraph,
                root,
                structure,
                structures.physical_root_adds,
                eligible_single_switch_adds,
            ) {
                Ok(_) => PointwiseDirectProbe::Ready,
                Err(reject) => PointwiseDirectProbe::Rejected(reject),
            };
            let binder = binder_aware_pointwise_add_switch_cancellation_for_structure(
                egraph, root, structure,
            )
            .map(|plan| binder_preflight_ready(&plan));
            PointwiseAddSwitchProbeOutcome { direct, binder }
        })
        .collect::<Vec<_>>();
    PointwiseAddSwitchProbe {
        physical_root_adds: structures.physical_root_adds,
        eligible_single_switch_adds,
        direct_switch_children: structures.direct_switch_children,
        direct_grouped_add_children: structures.direct_grouped_add_children,
        outcomes: outcomes.into_boxed_slice(),
        omitted_eligible_structures: eligible_single_switch_adds
            .saturating_sub(MAX_POINTWISE_PROBE_STRUCTURES),
    }
}

struct UnmatchedFixedTermsDiagnostic {
    direct_terms: usize,
    negated_terms: usize,
    fixed_unique_add_children: usize,
    case_physical_adds: usize,
    case_grouped_add_children: usize,
    fixed_terms: Box<[SignedCanonicalMultiplicity]>,
    fixed_terms_omitted_occurrences: usize,
    case_terms: Box<[SignedCanonicalMultiplicity]>,
    case_terms_omitted_occurrences: usize,
}

/// Builds a bounded failure-only report.  The success path never calls this
/// helper: it retains at most sixteen identities from the already validated
/// signed case plan and its direct fixed-term sequence.
fn unmatched_fixed_terms_diagnostic(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    fixed: &[Id],
    case_terms: &[SignedPointwiseTerm],
    case_physical_adds: usize,
    case_grouped_add_children: usize,
) -> UnmatchedFixedTermsDiagnostic {
    let (fixed_terms, fixed_terms_omitted_occurrences) = signed_multiplicity_summary(egraph, fixed);
    let signed_case_terms =
        case_terms.iter().map(|term| (term.base, term.negative)).collect::<Vec<_>>();
    let (case_terms, case_terms_omitted_occurrences) =
        signed_leaf_multiplicity_summary(&signed_case_terms);
    UnmatchedFixedTermsDiagnostic {
        direct_terms: signed_case_terms.len(),
        negated_terms: signed_case_terms.iter().filter(|(_, negative)| *negative).count(),
        fixed_unique_add_children: fixed
            .iter()
            .filter(|term| unique_add_terms(egraph, **term).is_some())
            .count(),
        case_physical_adds,
        case_grouped_add_children,
        fixed_terms,
        fixed_terms_omitted_occurrences,
        case_terms,
        case_terms_omitted_occurrences,
    }
}

fn signed_multiplicity_summary(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    terms: &[Id],
) -> (Box<[SignedCanonicalMultiplicity]>, usize) {
    let mut retained = Vec::with_capacity(MAX_UNMATCHED_TERM_IDENTITIES);
    let mut omitted_occurrences = 0;
    for &term in terms {
        let Some((base, negative)) = signed_additive_term(egraph, term) else { continue };
        let eclass = usize::from(egraph.find(base));
        if let Some(summary) =
            retained.iter_mut().find(|summary: &&mut SignedCanonicalMultiplicity| {
                summary.eclass == eclass && summary.negative == negative
            })
        {
            summary.multiplicity += 1;
        } else if retained.len() < MAX_UNMATCHED_TERM_IDENTITIES {
            retained.push(SignedCanonicalMultiplicity { eclass, negative, multiplicity: 1 });
        } else {
            omitted_occurrences += 1;
        }
    }
    retained.sort_unstable();
    (retained.into_boxed_slice(), omitted_occurrences)
}

fn signed_leaf_multiplicity_summary(
    terms: &[(Id, bool)],
) -> (Box<[SignedCanonicalMultiplicity]>, usize) {
    let mut summaries = BTreeMap::<(usize, bool), usize>::new();
    for (base, negative) in terms {
        *summaries.entry((usize::from(*base), *negative)).or_default() += 1;
    }
    let retained = summaries
        .iter()
        .take(MAX_UNMATCHED_TERM_IDENTITIES)
        .map(|((eclass, negative), multiplicity)| SignedCanonicalMultiplicity {
            eclass: *eclass,
            negative: *negative,
            multiplicity: *multiplicity,
        })
        .collect::<Vec<_>>();
    let omitted_occurrences = summaries
        .iter()
        .skip(MAX_UNMATCHED_TERM_IDENTITIES)
        .map(|(_, multiplicity)| *multiplicity)
        .sum();
    (retained.into_boxed_slice(), omitted_occurrences)
}

#[cfg(test)]
pub(crate) fn pointwise_add_switch_cancellation_reason(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    root: Id,
) -> Result<(), PointwiseAddSwitchReject> {
    pointwise_add_switch_cancellation_result(egraph, root).map(|_| ())
}

fn pointwise_add_switch_structure(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    root: Id,
    terms: &[Id],
) -> Option<(MxxLang, PointwiseAddSwitchStructure)> {
    let terms = terms.iter().map(|term| egraph.find(*term)).collect::<Vec<_>>();
    if terms.iter().any(|term| *term == root) {
        return None;
    }
    let switches = terms
        .iter()
        .enumerate()
        .filter_map(|(index, term)| unique_switch_cases(egraph, *term).map(|cases| (index, cases)))
        .collect::<Vec<_>>();
    if switches.len() != 1 {
        return None;
    }
    let (switch_index, switch) = &switches[0];
    if switch.len() < 2 ||
        switch[1..].iter().any(|term| egraph.find(*term) == root) ||
        terms
            .iter()
            .enumerate()
            .any(|(index, term)| index != *switch_index && has_physical_switch(egraph, *term))
    {
        return None;
    }
    Some((
        MxxLang::MatrixAdd(terms.clone().into_boxed_slice()),
        PointwiseAddSwitchStructure { terms, switch_index: *switch_index, switch: switch.clone() },
    ))
}

#[cfg(test)]
fn build_pointwise_add_switch_cancellation(
    egraph: &mut EGraph<MxxLang, MxxAnalysis>,
    root: Id,
    plan: PointwiseAddSwitchPlan,
) -> Option<Id> {
    let mut do_not_collect = || false;
    let mut ignore = ignore_binder_build_reject;
    build_pointwise_add_switch_cancellation_inner(
        egraph,
        root,
        plan,
        &mut || Ok(()),
        &mut do_not_collect,
        &mut ignore,
    )
}

fn build_pointwise_add_switch_cancellation_with_context(
    egraph: &mut EGraph<MxxLang, MxxAnalysis>,
    root: Id,
    plan: PointwiseAddSwitchPlan,
    context: &RewriteContext,
) -> Option<Id> {
    let selector = plan.selector;
    let mut diagnostic_available = tracing::enabled!(tracing::Level::DEBUG);
    let mut log_first_reject = || std::mem::take(&mut diagnostic_available);
    let mut emit_reject = |reject: &BinderBuildReject| {
        emit_binder_build_reject(root, selector, reject);
    };
    build_pointwise_add_switch_cancellation_inner(
        egraph,
        root,
        plan,
        &mut || context.reserve(1).then_some(()).ok_or(()),
        &mut log_first_reject,
        &mut emit_reject,
    )
}

fn build_pointwise_add_switch_cancellation_inner(
    egraph: &mut EGraph<MxxLang, MxxAnalysis>,
    root: Id,
    plan: PointwiseAddSwitchPlan,
    progress: &mut dyn FnMut() -> Result<(), ()>,
    should_collect: &mut dyn FnMut() -> bool,
    sink: &mut dyn FnMut(&BinderBuildReject),
) -> Option<Id> {
    if let Some(binder_plans) = plan.binder_aware {
        for binder_plan in binder_plans.into_vec() {
            let replacement = build_binder_aware_pointwise_add_switch_cancellation_with_sink(
                egraph,
                root,
                plan.selector,
                binder_plan,
                progress,
                should_collect,
                sink,
            );
            if let Some(replacement) = replacement {
                return Some(replacement);
            }
        }
        return None;
    }
    let mut cases = Vec::with_capacity(plan.cases.len() + 1);
    cases.push(plan.selector);
    for terms in plan.cases {
        let mut materialized = Vec::with_capacity(terms.len());
        for term in terms {
            materialized.push(if term.negative {
                egraph.add(MxxLang::MatrixNegate([term.base]))
            } else {
                term.base
            });
        }
        cases.push(build_detached_pointwise_terms(egraph, root, materialized));
    }
    Some(egraph.add(MxxLang::Switch(cases.into_boxed_slice())))
}

/// Instantiates each distinct fixed base once per physical stored case, then
/// normalizes every case with the reconstructed fixed sequence. The caller
/// unions only after every case has completed.
#[cfg(test)]
fn build_binder_aware_pointwise_add_switch_cancellation(
    egraph: &mut EGraph<MxxLang, MxxAnalysis>,
    root: Id,
    selector: Id,
    plan: BinderAwarePointwiseAddSwitchPlan,
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<Id> {
    let mut log_enabled = || tracing::enabled!(tracing::Level::DEBUG);
    let mut log_reject = |reject: &BinderBuildReject| {
        emit_binder_build_reject(root, selector, reject);
    };
    build_binder_aware_pointwise_add_switch_cancellation_with_sink(
        egraph,
        root,
        selector,
        plan,
        progress,
        &mut log_enabled,
        &mut log_reject,
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BinderBuildRejectStage {
    FixedSignedReject,
    NoExactCancellation,
    RootCycle,
    Equivalent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BinderBuildSignedLeaf {
    eclass: usize,
    negative: bool,
}

/// Failure-only binder-build evidence.  It is passed directly to the local
/// logger (or a cfg(test) sink), and is never stored in analysis or the graph.
#[derive(Clone, Debug, Eq, PartialEq)]
struct BinderBuildReject {
    case_index: usize,
    stage: BinderBuildRejectStage,
    actual: Box<[BinderBuildSignedLeaf]>,
    actual_omitted: usize,
    fixed: Box<[BinderBuildSignedLeaf]>,
    fixed_omitted: usize,
    actual_product_spines: Box<[RetainedProductSpine]>,
    fixed_product_spines: Box<[RetainedProductSpine]>,
    actual_product_leaves: Box<[ProductLeafDiagnostic]>,
    fixed_product_leaves: Box<[ProductLeafDiagnostic]>,
}

#[cfg(test)]
fn ignore_binder_build_reject(_: &BinderBuildReject) {}

fn summarize_binder_build_leaves(leaves: &[(Id, bool)]) -> (Box<[BinderBuildSignedLeaf]>, usize) {
    const DIAGNOSTIC_LEAF_LIMIT: usize = 16;
    let retained = leaves
        .iter()
        .take(DIAGNOSTIC_LEAF_LIMIT)
        .map(|(id, negative)| BinderBuildSignedLeaf {
            eclass: usize::from(*id),
            negative: *negative,
        })
        .collect();
    (retained, leaves.len().saturating_sub(DIAGNOSTIC_LEAF_LIMIT))
}

fn report_binder_build_reject(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    case_index: usize,
    stage: BinderBuildRejectStage,
    actual: &[(Id, bool)],
    fixed: &[(Id, bool)],
    should_collect: &mut dyn FnMut() -> bool,
    sink: &mut dyn FnMut(&BinderBuildReject),
) {
    if !should_collect() {
        return;
    }
    let (actual, actual_omitted) = summarize_binder_build_leaves(actual);
    let (fixed, fixed_omitted) = summarize_binder_build_leaves(fixed);
    let actual_ids = actual.iter().map(|leaf| Id::from(leaf.eclass)).collect::<Vec<_>>();
    let fixed_ids = fixed.iter().map(|leaf| Id::from(leaf.eclass)).collect::<Vec<_>>();
    let reject = BinderBuildReject {
        case_index,
        stage,
        actual_product_spines: retained_product_spines(egraph, &actual_ids).into_boxed_slice(),
        fixed_product_spines: retained_product_spines(egraph, &fixed_ids).into_boxed_slice(),
        actual_product_leaves: retained_product_leaves(egraph, &actual_ids).into_boxed_slice(),
        fixed_product_leaves: retained_product_leaves(egraph, &fixed_ids).into_boxed_slice(),
        actual,
        actual_omitted,
        fixed,
        fixed_omitted,
    };
    sink(&reject);
}

fn emit_binder_build_reject(root: Id, selector: Id, reject: &BinderBuildReject) {
    if tracing::enabled!(tracing::Level::DEBUG) {
        tracing::debug!(
            event = "pointwise_binder_build_failure",
            root = usize::from(root),
            selector = usize::from(selector),
            case_index = reject.case_index,
            stage = ?reject.stage,
            actual = ?reject.actual,
            actual_omitted = reject.actual_omitted,
            fixed = ?reject.fixed,
            fixed_omitted = reject.fixed_omitted,
            actual_product_spines = ?reject.actual_product_spines,
            fixed_product_spines = ?reject.fixed_product_spines,
            actual_product_leaves = ?reject.actual_product_leaves,
            fixed_product_leaves = ?reject.fixed_product_leaves,
        );
    }
}

/// A temporary actual-case term used while peeling fixed monomials.  It is a
/// plan only: no e-node is created until every fixed target has been removed.
#[derive(Clone, Debug, Eq, PartialEq)]
enum PeelTerm {
    Concrete { base: Id, negative: bool },
    ProductFactor { prefix: Box<[Id]>, terms: Vec<(Id, bool)>, suffix: Box<[Id]>, negative: bool },
}

const PEEL_DIAGNOSTIC_LIMIT: usize = 32;
const PEEL_DIAGNOSTIC_SPINE_LIMIT: usize = 16;
const PEEL_DIAGNOSTIC_NODE_LIMIT: usize = 8;

/// DEBUG-only description of one retained fixed-product factor.  It is built
/// only after the fixed spine itself has passed the existing cap, so it cannot
/// influence the ordered matching plan.
#[derive(Debug)]
struct FixedSpineFactorDiagnostic {
    eclass: usize,
    matrix_shape: Option<super::identity::ResolvedMatrixType>,
    semantic_nodes: Box<[String]>,
    semantic_nodes_omitted: usize,
}

impl FixedSpineFactorDiagnostic {
    fn log_fields(
        &self,
    ) -> (usize, Option<&super::identity::ResolvedMatrixType>, &[String], usize) {
        (self.eclass, self.matrix_shape.as_ref(), &self.semantic_nodes, self.semantic_nodes_omitted)
    }
}

fn fixed_spine_factor_diagnostic(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    id: Id,
) -> FixedSpineFactorDiagnostic {
    let id = egraph.find(id);
    let class = &egraph[id];
    let matrix_shape = match &class.data.sort {
        Ok(MxxSort::Matrix(matrix_shape)) => Some(matrix_shape.clone()),
        _ => None,
    };
    let semantic_nodes = class
        .nodes
        .iter()
        .take(PEEL_DIAGNOSTIC_NODE_LIMIT)
        .map(|node| match node {
            MxxLang::Atom { source, indices } => {
                let descriptor = egraph.analysis.symbols.atomic_sources.get(source.0);
                format!(
                    "atom source_kind={:?} relation_role={:?} indices={}",
                    descriptor.map(|descriptor| &descriptor.key),
                    descriptor.and_then(|descriptor| descriptor.relation_role),
                    indices.len(),
                )
            }
            MxxLang::MatrixConstant(spec) => format!(
                "matrix_constant spec={:?}",
                egraph.analysis.symbols.matrix_constants.get(spec.0),
            ),
            MxxLang::Switch(cases) => format!(
                "switch selector={} case_count={}",
                cases.first().map_or(usize::MAX, |selector| usize::from(egraph.find(*selector))),
                cases.len().saturating_sub(1),
            ),
            MxxLang::HashPlain { query, arguments } => format!(
                "hash_plain query={:?} arguments={}",
                egraph.analysis.symbols.hash_queries.get(query.0),
                arguments.len(),
            ),
            MxxLang::ExtractCoefficient { canonical_exclusive_upper, .. } => format!(
                "extract_coefficient canonical_exclusive_upper={canonical_exclusive_upper:?}",
            ),
            MxxLang::LiftConstantPolynomial { matrix_type, .. } => {
                format!("lift_constant_polynomial matrix_type={matrix_type:?}")
            }
            MxxLang::CrtRecompose { spec, inputs } => format!(
                "crt_recompose spec={:?} inputs={}",
                egraph.analysis.symbols.crts.get(spec.0),
                inputs.len(),
            ),
            MxxLang::PackPolynomialCoefficients { matrix_type, coefficient_bits, bits } => {
                format!(
                    "pack_polynomial_coefficients matrix_type={matrix_type:?} coefficient_bits={coefficient_bits:?} bits={}",
                    bits.len(),
                )
            }
            _ => format!("operator={} children={}", node.operator_name(), node.children().len()),
        })
        .collect::<Vec<_>>();
    FixedSpineFactorDiagnostic {
        eclass: usize::from(id),
        matrix_shape,
        semantic_nodes: semantic_nodes.into_boxed_slice(),
        semantic_nodes_omitted: class.nodes.len().saturating_sub(PEEL_DIAGNOSTIC_NODE_LIMIT),
    }
}

fn capped_peel_diagnostic_ids(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    ids: &[Id],
) -> (Vec<FixedSpineFactorDiagnostic>, usize) {
    (
        ids.iter()
            .take(PEEL_DIAGNOSTIC_SPINE_LIMIT)
            .map(|id| fixed_spine_factor_diagnostic(egraph, *id))
            .collect(),
        ids.len().saturating_sub(PEEL_DIAGNOSTIC_SPINE_LIMIT),
    )
}

/// Logs the fixed occurrence-to-base mapping before monomial expansion.  An
/// expanded spine can originate from an additive factor, so this is the only
/// cheap, exact occurrence provenance retained by the DEBUG diagnostic.
fn log_mapped_fixed_occurrences(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    case_index: usize,
    occurrences: &[FixedOccurrence],
    mapped_fixed: &[(Id, bool)],
) {
    if !tracing::enabled!(tracing::Level::DEBUG) {
        return;
    }
    for (occurrence_index, (occurrence, (base, negative))) in
        occurrences.iter().zip(mapped_fixed).take(PEEL_DIAGNOSTIC_LIMIT).enumerate()
    {
        tracing::debug!(
            event = "binder_mapped_fixed_occurrence",
            case_index,
            occurrence_index,
            base_index = occurrence.base_index,
            negative,
            mapped_base = ?fixed_spine_factor_diagnostic(egraph, *base),
        );
    }
    tracing::debug!(
        event = "binder_mapped_fixed_occurrence_summary",
        case_index,
        occurrence_count = mapped_fixed.len(),
        occurrence_count_omitted = mapped_fixed.len().saturating_sub(PEEL_DIAGNOSTIC_LIMIT),
    );
}

/// Logs the mapped fixed targets before exact opposite-spine cancellation.
/// This preserves the evidence needed to tell whether a signed fixed product
/// existed even when it is subsequently eliminated from the peel input.
fn log_pre_cancel_mapped_fixed_spines(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    case_index: usize,
    fixed_spines: &[(Box<[Id]>, bool)],
) {
    if !tracing::enabled!(tracing::Level::DEBUG) {
        return;
    }
    for (spine_index, (spine, negative)) in
        fixed_spines.iter().take(PEEL_DIAGNOSTIC_LIMIT).enumerate()
    {
        let (factors, factors_omitted) = capped_peel_diagnostic_ids(egraph, spine);
        let factor_fields =
            factors.iter().map(FixedSpineFactorDiagnostic::log_fields).collect::<Vec<_>>();
        tracing::debug!(
            event = "binder_pre_cancel_mapped_fixed_spine",
            case_index,
            spine_index,
            negative,
            factors = ?factor_fields,
            factors_omitted,
        );
    }
    tracing::debug!(
        event = "binder_pre_cancel_mapped_fixed_spine_summary",
        case_index,
        spine_count = fixed_spines.len(),
        spine_count_omitted = fixed_spines.len().saturating_sub(PEEL_DIAGNOSTIC_LIMIT),
    );
}

fn log_peel_term(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    target_index: Option<usize>,
    contribution: &'static str,
    position: usize,
    term: &PeelTerm,
) {
    if !tracing::enabled!(tracing::Level::DEBUG) {
        return;
    }
    match term {
        PeelTerm::Concrete { base, negative } => {
            let base = fixed_spine_factor_diagnostic(egraph, *base);
            let (base_eclass, base_matrix_shape, base_semantic_nodes, base_semantic_nodes_omitted) =
                base.log_fields();
            tracing::debug!(
                event = "fixed_target_peel_term",
                target_index = ?target_index,
                contribution,
                position,
                kind = "concrete",
                base_eclass,
                base_matrix_shape = ?base_matrix_shape,
                base_semantic_nodes = ?base_semantic_nodes,
                base_semantic_nodes_omitted,
                negative,
            )
        }
        PeelTerm::ProductFactor { prefix, terms, suffix, negative } => {
            let (prefix, prefix_omitted) = capped_peel_diagnostic_ids(egraph, prefix);
            let selected_additive_leaves = terms
                .iter()
                .take(PEEL_DIAGNOSTIC_SPINE_LIMIT)
                .map(|(id, _)| fixed_spine_factor_diagnostic(egraph, *id))
                .collect::<Vec<_>>();
            let selected_additive_leaf_negative = terms
                .iter()
                .take(PEEL_DIAGNOSTIC_SPINE_LIMIT)
                .map(|(_, negative)| *negative)
                .collect::<Vec<_>>();
            let selected_additive_leaves_omitted =
                terms.len().saturating_sub(PEEL_DIAGNOSTIC_SPINE_LIMIT);
            let (suffix, suffix_omitted) = capped_peel_diagnostic_ids(egraph, suffix);
            let prefix_fields =
                prefix.iter().map(FixedSpineFactorDiagnostic::log_fields).collect::<Vec<_>>();
            let selected_additive_leaf_fields = selected_additive_leaves
                .iter()
                .map(FixedSpineFactorDiagnostic::log_fields)
                .collect::<Vec<_>>();
            let suffix_fields =
                suffix.iter().map(FixedSpineFactorDiagnostic::log_fields).collect::<Vec<_>>();
            tracing::debug!(
                event = "fixed_target_peel_term",
                target_index = ?target_index,
                contribution,
                position,
                kind = "product_factor",
                negative,
                prefix = ?prefix_fields,
                prefix_omitted,
                selected_additive_leaves = ?selected_additive_leaf_fields,
                selected_additive_leaf_negative = ?selected_additive_leaf_negative,
                selected_additive_leaves_omitted,
                suffix = ?suffix_fields,
                suffix_omitted,
            );
        }
    }
}

fn uncontested_product_factors_with_progress(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    term: Id,
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<Box<[Id]>> {
    let term = egraph.find(term);
    let mut unique: Option<Box<[Id]>> = None;
    for node in &egraph[term].nodes {
        progress().ok()?;
        let MxxLang::MatrixMultiply(factors) = node else {
            return None;
        };
        let mut candidate = Vec::new();
        for factor in factors {
            progress().ok()?;
            candidate.try_reserve(1).ok()?;
            candidate.push(egraph.find(*factor));
        }
        let candidate = candidate.into_boxed_slice();
        if let Some(previous) = &unique {
            if previous.len() != candidate.len() {
                return None;
            }
            for (left, right) in previous.iter().zip(&candidate) {
                progress().ok()?;
                if left != right {
                    return None;
                }
            }
        } else {
            unique = Some(candidate);
        }
    }
    unique
}

fn is_exact_zero_matrix(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    term: Id,
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<bool> {
    for node in &egraph[egraph.find(term)].nodes {
        progress().ok()?;
        if matches!(
            node,
            MxxLang::MatrixConstant(spec)
                if matches!(
                    egraph.analysis.symbols.matrix_constants.get(spec.0),
                    Some(super::identity::MatrixConstantSpec {
                        value: MatrixConstantValue::Zero,
                        ..
                    })
                )
        ) {
            return Some(true);
        }
    }
    Some(false)
}

/// Returns whether `factor` is a scalar ring constant whose polynomial has no
/// nonconstant coefficients.  These are the only factors that may move across
/// another matrix factor while matching a fixed product spine: a 1-by-1
/// constant polynomial is central in the coefficient ring, whereas neither a
/// general 1-by-1 polynomial nor a larger matrix is.
fn is_central_constant_scalar(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    factor: Id,
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<bool> {
    let factor = egraph.find(factor);
    let Ok(MxxSort::Matrix(matrix_type)) = &egraph[factor].data.sort else {
        return Some(false);
    };
    if resolved_constant(&matrix_type.rows) != Some(BigInt::from(1_u8)) ||
        resolved_constant(&matrix_type.columns) != Some(BigInt::from(1_u8))
    {
        return Some(false);
    }
    for node in &egraph[factor].nodes {
        progress().ok()?;
        let MxxLang::MatrixConstant(spec) = node else { continue };
        let Some(super::identity::MatrixConstantSpec {
            value: MatrixConstantValue::Polynomial { coefficients },
            ..
        }) = egraph.analysis.symbols.matrix_constants.get(spec.0)
        else {
            continue;
        };
        let mut constant = true;
        for coefficient in coefficients.iter().skip(1) {
            progress().ok()?;
            if resolved_constant(coefficient).is_none_or(|coefficient| !coefficient.is_zero()) {
                constant = false;
                break;
            }
        }
        if constant {
            return Some(true);
        }
    }
    Some(false)
}

/// Matches two already-fixed product spines after removing precisely the
/// central scalar factors accepted by [`is_central_constant_scalar`].  The
/// non-scalar sequence stays ordered, and scalar identities retain exact
/// multiplicity, so this is not a general matrix-commutation rule.
fn scalar_reordered_spines_match(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    actual: &[Id],
    target: &[Id],
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<bool> {
    let mut actual_scalars = HashMap::<Id, usize>::new();
    let mut target_scalars = HashMap::<Id, usize>::new();
    let mut actual_nonscalars = Vec::new();
    let mut target_nonscalars = Vec::new();
    for factor in actual {
        let factor = egraph.find(*factor);
        progress().ok()?;
        if is_central_constant_scalar(egraph, factor, progress)? {
            *actual_scalars.entry(factor).or_default() += 1;
        } else {
            actual_nonscalars.try_reserve(1).ok()?;
            actual_nonscalars.push(factor);
        }
    }
    for factor in target {
        let factor = egraph.find(*factor);
        progress().ok()?;
        if is_central_constant_scalar(egraph, factor, progress)? {
            *target_scalars.entry(factor).or_default() += 1;
        } else {
            target_nonscalars.try_reserve(1).ok()?;
            target_nonscalars.push(factor);
        }
    }
    if actual_scalars.is_empty() ||
        actual_scalars != target_scalars ||
        actual_nonscalars.len() != target_nonscalars.len()
    {
        return Some(false);
    }
    for (actual, target) in actual_nonscalars.iter().zip(&target_nonscalars) {
        progress().ok()?;
        if actual != target {
            return Some(false);
        }
    }
    Some(true)
}

fn scalar_reordered_direct_span_matches(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    term: Id,
    target: &[Id],
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<bool> {
    let Some(Some(actual)) = flatten_uncontested_product_factors(egraph, term, progress) else {
        return Some(false);
    };
    scalar_reordered_spines_match(egraph, &actual, target, progress)
}

/// Returns every target position reachable from one e-class.  Each state is
/// `(canonical eclass, target position, sign)`: singleton identity is always
/// admitted, then only direct Multiply/Negate e-nodes are explored. Add and
/// Switch stay atomic.  This proves a direct ordered witness across arbitrary
/// product association without materializing a Cartesian spine list.
fn direct_signed_target_positions(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    term: Id,
    position: usize,
    negative: bool,
    target: &[Id],
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<BTreeSet<(usize, bool)>> {
    type State = (Id, usize, bool);
    fn visit(
        egraph: &EGraph<MxxLang, MxxAnalysis>,
        state: State,
        target: &[Id],
        memo: &mut HashMap<State, BTreeSet<(usize, bool)>>,
        active: &mut HashSet<State>,
        progress: &mut dyn FnMut() -> Result<(), ()>,
    ) -> Option<BTreeSet<(usize, bool)>> {
        progress().ok()?;
        if let Some(previous) = memo.get(&state) {
            let mut copied = BTreeSet::new();
            for position in previous {
                progress().ok()?;
                copied.insert(*position);
            }
            return Some(copied);
        }
        if !active.insert(state) {
            // A recursive structural representation is simply not a witness
            // for this branch.  The canonical singleton check above remains
            // valid in its caller and must not be discarded with the branch.
            return Some(BTreeSet::new());
        }
        let (term, position, negative) = state;
        let mut output = BTreeSet::new();
        if position < target.len() {
            progress().ok()?;
            if term == egraph.find(target[position]) {
                progress().ok()?;
                output.insert((position + 1, negative));
            }
        }
        for node in &egraph[term].nodes {
            progress().ok()?;
            match node {
                MxxLang::MatrixNegate([input]) => {
                    for result in visit(
                        egraph,
                        (egraph.find(*input), position, !negative),
                        target,
                        memo,
                        active,
                        progress,
                    )? {
                        progress().ok()?;
                        output.insert(result);
                    }
                }
                MxxLang::MatrixMultiply(factors) => {
                    progress().ok()?;
                    let mut states = BTreeSet::from([(position, negative)]);
                    for factor in factors {
                        let mut next = BTreeSet::new();
                        for (child_position, child_negative) in states {
                            for result in visit(
                                egraph,
                                (egraph.find(*factor), child_position, child_negative),
                                target,
                                memo,
                                active,
                                progress,
                            )? {
                                progress().ok()?;
                                next.insert(result);
                            }
                        }
                        states = next;
                        if states.is_empty() {
                            break;
                        }
                    }
                    for result in states {
                        progress().ok()?;
                        output.insert(result);
                    }
                }
                _ => {}
            }
        }
        active.remove(&state);
        let mut stored = BTreeSet::new();
        for result in &output {
            progress().ok()?;
            stored.insert(*result);
        }
        progress().ok()?;
        memo.insert(state, stored);
        Some(output)
    }

    progress().ok()?;
    let mut memo = HashMap::new();
    progress().ok()?;
    let mut active = HashSet::new();
    visit(egraph, (egraph.find(term), position, negative), target, &mut memo, &mut active, progress)
}

fn has_opposite_direct_span(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    term: Id,
    target: &[Id],
    actual_negative: bool,
    target_negative: bool,
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<bool> {
    let positions = direct_signed_target_positions(egraph, term, 0, false, target, progress)?;
    for (position, negative) in positions {
        progress().ok()?;
        if position == target.len() && (actual_negative != negative) != target_negative {
            return Some(true);
        }
    }
    scalar_reordered_direct_span_matches(egraph, term, target, progress)
        .map(|matched| matched && actual_negative != target_negative)
}

/// Matches a product with one selected additive leaf using a small dynamic
/// program over the fixed target positions.  This normalizes direct Negate
/// signs in every factor without enumerating products of alternatives.
fn product_factor_span_matches(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    prefix: &[Id],
    term: Id,
    term_negative: bool,
    suffix: &[Id],
    target: &[Id],
    actual_negative: bool,
    target_negative: bool,
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<bool> {
    let factors = prefix
        .iter()
        .copied()
        .map(|factor| (factor, false))
        .chain(std::iter::once((term, term_negative)))
        .chain(suffix.iter().copied().map(|factor| (factor, false)));
    progress().ok()?;
    let mut states = BTreeSet::new();
    progress().ok()?;
    states.insert((0usize, false));
    for (factor, factor_negative) in factors {
        let mut next = BTreeSet::new();
        for (position, negative) in states {
            for (end, span_negative) in direct_signed_target_positions(
                egraph,
                factor,
                position,
                factor_negative,
                target,
                progress,
            )? {
                progress().ok()?;
                next.insert((end, negative != span_negative));
            }
        }
        states = next;
        if states.is_empty() {
            break;
        }
    }
    for (position, negative) in states {
        progress().ok()?;
        if position == target.len() && (actual_negative != negative) != target_negative {
            return Some(true);
        }
    }
    let mut actual = Vec::new();
    for factor in prefix {
        progress().ok()?;
        actual.try_reserve(1).ok()?;
        actual.push(*factor);
    }
    progress().ok()?;
    actual.try_reserve(1).ok()?;
    actual.push(term);
    for factor in suffix {
        progress().ok()?;
        actual.try_reserve(1).ok()?;
        actual.push(*factor);
    }
    scalar_reordered_spines_match(egraph, &actual, target, progress)
        .map(|matched| matched && (actual_negative != term_negative) != target_negative)
}

fn copy_ids(ids: &[Id], progress: &mut dyn FnMut() -> Result<(), ()>) -> Option<Box<[Id]>> {
    let mut output = Vec::new();
    for id in ids {
        progress().ok()?;
        output.try_reserve(1).ok()?;
        output.push(*id);
    }
    Some(output.into_boxed_slice())
}

fn copy_signed_terms(
    terms: &[(Id, bool)],
    skip: Option<usize>,
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<Vec<(Id, bool)>> {
    let mut output = Vec::new();
    for (index, term) in terms.iter().enumerate() {
        if Some(index) != skip {
            progress().ok()?;
            output.try_reserve(1).ok()?;
            output.push(*term);
        }
    }
    Some(output)
}

/// A smaller residual is always preferable: it retains less exact signal
/// structure and therefore cannot hide a cancellation available through a
/// different physical e-node.  The structural order only breaks equal-size
/// ties, making the chosen plan independent of e-node iteration order.
fn compare_ids_with_progress(
    left: &[Id],
    right: &[Id],
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<std::cmp::Ordering> {
    for (left, right) in left.iter().zip(right) {
        progress().ok()?;
        let order = left.cmp(right);
        if !order.is_eq() {
            return Some(order);
        }
    }
    Some(left.len().cmp(&right.len()))
}

fn compare_signed_ids_with_progress(
    left: &[(Id, bool)],
    right: &[(Id, bool)],
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<std::cmp::Ordering> {
    for ((left_id, left_negative), (right_id, right_negative)) in left.iter().zip(right) {
        progress().ok()?;
        let order = left_id.cmp(right_id);
        if !order.is_eq() {
            return Some(order);
        }
        progress().ok()?;
        let order = left_negative.cmp(right_negative);
        if !order.is_eq() {
            return Some(order);
        }
    }
    Some(left.len().cmp(&right.len()))
}

fn compare_peel_term_with_progress(
    left: &PeelTerm,
    right: &PeelTerm,
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<std::cmp::Ordering> {
    match (left, right) {
        (
            PeelTerm::Concrete { base: left_base, negative: left_negative },
            PeelTerm::Concrete { base: right_base, negative: right_negative },
        ) => {
            progress().ok()?;
            let order = left_base.cmp(right_base);
            if !order.is_eq() {
                return Some(order);
            }
            progress().ok()?;
            Some(left_negative.cmp(right_negative))
        }
        (PeelTerm::Concrete { .. }, PeelTerm::ProductFactor { .. }) => {
            Some(std::cmp::Ordering::Less)
        }
        (PeelTerm::ProductFactor { .. }, PeelTerm::Concrete { .. }) => {
            Some(std::cmp::Ordering::Greater)
        }
        (
            PeelTerm::ProductFactor {
                prefix: left_prefix,
                terms: left_terms,
                suffix: left_suffix,
                negative: left_negative,
            },
            PeelTerm::ProductFactor {
                prefix: right_prefix,
                terms: right_terms,
                suffix: right_suffix,
                negative: right_negative,
            },
        ) => {
            let order = compare_ids_with_progress(left_prefix, right_prefix, progress)?;
            if !order.is_eq() {
                return Some(order);
            }
            let order = compare_signed_ids_with_progress(left_terms, right_terms, progress)?;
            if !order.is_eq() {
                return Some(order);
            }
            let order = compare_ids_with_progress(left_suffix, right_suffix, progress)?;
            if !order.is_eq() {
                return Some(order);
            }
            progress().ok()?;
            Some(left_negative.cmp(right_negative))
        }
    }
}

fn peel_term_count_with_progress(
    terms: &[PeelTerm],
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<usize> {
    let mut count = 0usize;
    for term in terms {
        progress().ok()?;
        match term {
            PeelTerm::Concrete { .. } => count = count.checked_add(1)?,
            PeelTerm::ProductFactor { terms, .. } => {
                for _ in terms {
                    progress().ok()?;
                    count = count.checked_add(1)?;
                }
            }
        }
    }
    Some(count)
}

fn residual_is_better(
    candidate: &[PeelTerm],
    previous: &[PeelTerm],
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<bool> {
    let candidate_count = peel_term_count_with_progress(candidate, progress)?;
    let previous_count = peel_term_count_with_progress(previous, progress)?;
    if candidate_count != previous_count {
        return Some(candidate_count < previous_count);
    }
    for (left, right) in candidate.iter().zip(previous) {
        let order = compare_peel_term_with_progress(left, right, progress)?;
        if !order.is_eq() {
            return Some(order.is_lt());
        }
    }
    Some(candidate.len() < previous.len())
}

fn retain_best_residual(
    best: &mut Option<Vec<PeelTerm>>,
    candidate: Vec<PeelTerm>,
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<()> {
    let replace = match best.as_ref() {
        Some(previous) => residual_is_better(&candidate, previous, progress)?,
        None => true,
    };
    if replace {
        *best = Some(candidate);
    }
    Some(())
}

/// Plans removal of one fixed target from a selected factor of an ordered
/// product.  Every returned term already contains `prefix` and `suffix`.
/// This is what permits one direct, uncontested nested product to expose an
/// additive factor without distributing any other product.
///
/// Physical Add/Negate/Multiply nodes are alternatives, not a consensus: we
/// test every exact candidate, keep its untouched siblings, and select the
/// smallest residual.  Switches remain atomic.  In particular, this does not
/// enumerate products of additive alternatives or mix factor lists from
/// separate physical product witnesses.
fn peel_direct_product_factors(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    base: Id,
    factor_negative: bool,
    prefix: &[Id],
    factors: &[Id],
    suffix: &[Id],
    target: &(Box<[Id]>, bool),
    product_negative: bool,
    active: &mut HashSet<Id>,
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<Option<Vec<PeelTerm>>> {
    if factors.is_empty() {
        return Some(None);
    }
    let base = egraph.find(base);
    let mut canonical_factors = Vec::new();
    for factor in factors {
        progress().ok()?;
        let factor = egraph.find(*factor);
        if factor == base {
            return Some(None);
        }
        canonical_factors.try_reserve(1).ok()?;
        canonical_factors.push(factor);
    }
    // The sign accumulated while reaching this product belongs to the whole
    // product, not to each selected factor.
    let nested_product_negative = product_negative != factor_negative;
    let mut context = Vec::new();
    for factor in prefix.iter().chain(canonical_factors.iter()).chain(suffix) {
        progress().ok()?;
        context.try_reserve(1).ok()?;
        context.push(*factor);
    }
    let context_offset = prefix.len();
    let mut best = None;
    for factor_index in 0..canonical_factors.len() {
        if !factor_can_peel_fixed_target(egraph, canonical_factors[factor_index], progress)? {
            continue;
        }
        let selected_index = context_offset.checked_add(factor_index)?;
        if let Some(residual) = peel_product_factor_target(
            egraph,
            canonical_factors[factor_index],
            false,
            &context[..selected_index],
            &context[selected_index + 1..],
            target,
            nested_product_negative,
            active,
            progress,
        )? {
            retain_best_residual(&mut best, residual, progress)?;
        }
    }
    Some(best)
}

fn peel_product_factor_target(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    base: Id,
    factor_negative: bool,
    prefix: &[Id],
    suffix: &[Id],
    target: &(Box<[Id]>, bool),
    product_negative: bool,
    active: &mut HashSet<Id>,
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<Option<Vec<PeelTerm>>> {
    let base = egraph.find(base);
    if product_factor_span_matches(
        egraph,
        prefix,
        base,
        factor_negative,
        suffix,
        &target.0,
        product_negative,
        target.1,
        progress,
    )? {
        return Some(Some(Vec::new()));
    }
    progress().ok()?;
    if !active.insert(base) {
        return Some(None);
    }
    let result = (|| -> Option<Option<Vec<PeelTerm>>> {
        let uncontested_factors = flatten_uncontested_product_factors(egraph, base, progress)?;
        let mut best = None;
        for node in &egraph[base].nodes {
            progress().ok()?;
            match node {
                MxxLang::MatrixNegate([input]) => {
                    if let Some(residual) = peel_product_factor_target(
                        egraph,
                        *input,
                        !factor_negative,
                        prefix,
                        suffix,
                        target,
                        product_negative,
                        active,
                        progress,
                    )? {
                        retain_best_residual(&mut best, residual, progress)?;
                    }
                }
                MxxLang::MatrixAdd(children) => {
                    if children.is_empty() {
                        continue;
                    }
                    let mut self_child = false;
                    for child in children {
                        progress().ok()?;
                        if egraph.find(*child) == base {
                            self_child = true;
                            break;
                        }
                    }
                    if self_child {
                        continue;
                    }
                    for (selected_index, child) in children.iter().enumerate() {
                        progress().ok()?;
                        let Some(selected) = peel_product_factor_target(
                            egraph,
                            *child,
                            factor_negative,
                            prefix,
                            suffix,
                            target,
                            product_negative,
                            active,
                            progress,
                        )?
                        else {
                            continue;
                        };
                        let mut residual = Vec::new();
                        let mut selected = Some(selected);
                        for (child_index, sibling) in children.iter().enumerate() {
                            if child_index == selected_index {
                                for term in selected.take().expect("selected child is unique") {
                                    progress().ok()?;
                                    residual.try_reserve(1).ok()?;
                                    residual.push(term);
                                }
                            } else {
                                progress().ok()?;
                                residual.try_reserve(1).ok()?;
                                residual.push(PeelTerm::ProductFactor {
                                    prefix: copy_ids(prefix, progress)?,
                                    terms: vec![(egraph.find(*sibling), factor_negative)],
                                    suffix: copy_ids(suffix, progress)?,
                                    negative: product_negative,
                                });
                            }
                        }
                        retain_best_residual(&mut best, residual, progress)?;
                    }
                }
                MxxLang::MatrixMultiply(factors) if uncontested_factors.is_none() => {
                    if let Some(residual) = peel_direct_product_factors(
                        egraph,
                        base,
                        factor_negative,
                        prefix,
                        factors,
                        suffix,
                        target,
                        product_negative,
                        active,
                        progress,
                    )? {
                        if residual.is_empty() {
                            return Some(Some(Vec::new()));
                        }
                        retain_best_residual(&mut best, residual, progress)?;
                    }
                }
                _ => {}
            }
        }
        if let Some(factors) = uncontested_factors {
            if let Some(residual) = peel_direct_product_factors(
                egraph,
                base,
                factor_negative,
                prefix,
                &factors,
                suffix,
                target,
                product_negative,
                active,
                progress,
            )? {
                retain_best_residual(&mut best, residual, progress)?;
            }
        }
        Some(best)
    })();
    active.remove(&base);
    result
}

/// Flattens only an entirely direct, uncontested product representation for the
/// associated-product fast path.  Competing physical product witnesses are
/// considered separately by [`peel_direct_product_factors`].  This local view
/// is discarded after one peel attempt; it is not a cache.
fn flatten_uncontested_product_factors(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    base: Id,
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<Option<Box<[Id]>>> {
    fn append(
        egraph: &EGraph<MxxLang, MxxAnalysis>,
        base: Id,
        output: &mut Vec<Id>,
        active: &mut HashSet<Id>,
        progress: &mut dyn FnMut() -> Result<(), ()>,
    ) -> Option<bool> {
        let base = egraph.find(base);
        progress().ok()?;
        if !active.insert(base) {
            return Some(false);
        }
        let result = if let Some(factors) =
            uncontested_product_factors_with_progress(egraph, base, progress)
        {
            let mut valid = true;
            for factor in factors.iter().copied() {
                if !append(egraph, factor, output, active, progress)? {
                    valid = false;
                    break;
                }
            }
            valid
        } else {
            progress().ok()?;
            output.try_reserve(1).ok()?;
            output.push(base);
            true
        };
        active.remove(&base);
        Some(result)
    }

    if uncontested_product_factors_with_progress(egraph, base, progress).is_none() {
        return Some(None);
    }
    let mut output = Vec::new();
    let mut active = HashSet::new();
    if !append(egraph, base, &mut output, &mut active, progress)? {
        return Some(None);
    }
    Some(Some(output.into_boxed_slice()))
}

/// Checks whether a factor has a physical Add/Negate/Multiply path that can
/// expose a partial fixed target, avoiding a repeated full-context scan for
/// atoms of a long associated product.
fn factor_can_peel_fixed_target(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    factor: Id,
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<bool> {
    fn visit(
        egraph: &EGraph<MxxLang, MxxAnalysis>,
        factor: Id,
        active: &mut HashSet<Id>,
        progress: &mut dyn FnMut() -> Result<(), ()>,
    ) -> Option<bool> {
        let factor = egraph.find(factor);
        progress().ok()?;
        if !active.insert(factor) {
            return Some(false);
        }
        let result = (|| -> Option<bool> {
            for node in &egraph[factor].nodes {
                progress().ok()?;
                match node {
                    MxxLang::MatrixAdd(_) => return Some(true),
                    MxxLang::MatrixNegate([input]) => {
                        if visit(egraph, *input, active, progress)? {
                            return Some(true);
                        }
                    }
                    MxxLang::MatrixMultiply(factors) => {
                        for factor in factors {
                            progress().ok()?;
                            if visit(egraph, *factor, active, progress)? {
                                return Some(true);
                            }
                        }
                    }
                    _ => {}
                }
            }
            Some(false)
        })();
        active.remove(&factor);
        result
    }

    let mut active = HashSet::new();
    visit(egraph, factor, &mut active, progress)
}

fn peel_concrete_fixed_target(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    base: Id,
    negative: bool,
    target: &(Box<[Id]>, bool),
    active: &mut HashSet<Id>,
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<Option<Vec<PeelTerm>>> {
    let (target_spine, target_negative) = target;
    let base = egraph.find(base);
    if has_opposite_direct_span(egraph, base, target_spine, negative, *target_negative, progress)? {
        return Some(Some(Vec::new()));
    }
    progress().ok()?;
    if !active.insert(base) {
        return Some(None);
    }
    let result = (|| -> Option<Option<Vec<PeelTerm>>> {
        let uncontested_factors = uncontested_product_factors_with_progress(egraph, base, progress);
        let mut best = None;
        // Each direct Add/Negate node is a separate exact witness.  We open
        // only the selected child; siblings remain raw, ordered residuals.
        for node in &egraph[base].nodes {
            progress().ok()?;
            match node {
                MxxLang::MatrixNegate([input]) => {
                    if let Some(residual) = peel_concrete_fixed_target(
                        egraph, *input, !negative, target, active, progress,
                    )? {
                        retain_best_residual(&mut best, residual, progress)?;
                    }
                }
                MxxLang::MatrixAdd(children) => {
                    if children.is_empty() {
                        continue;
                    }
                    let mut self_child = false;
                    for child in children {
                        progress().ok()?;
                        if egraph.find(*child) == base {
                            self_child = true;
                            break;
                        }
                    }
                    if self_child {
                        continue;
                    }
                    for (selected_index, child) in children.iter().enumerate() {
                        progress().ok()?;
                        let Some(selected) = peel_concrete_fixed_target(
                            egraph, *child, negative, target, active, progress,
                        )?
                        else {
                            continue;
                        };
                        let mut residual = Vec::new();
                        let mut selected = Some(selected);
                        for (child_index, sibling) in children.iter().enumerate() {
                            if child_index == selected_index {
                                for term in selected.take().expect("selected child is unique") {
                                    progress().ok()?;
                                    residual.try_reserve(1).ok()?;
                                    residual.push(term);
                                }
                            } else {
                                progress().ok()?;
                                residual.try_reserve(1).ok()?;
                                residual.push(PeelTerm::Concrete {
                                    base: egraph.find(*sibling),
                                    negative,
                                });
                            }
                        }
                        retain_best_residual(&mut best, residual, progress)?;
                    }
                }
                MxxLang::MatrixMultiply(factors) if uncontested_factors.is_none() => {
                    if let Some(residual) = peel_direct_product_factors(
                        egraph,
                        base,
                        false,
                        &[],
                        factors,
                        &[],
                        target,
                        negative,
                        active,
                        progress,
                    )? {
                        retain_best_residual(&mut best, residual, progress)?;
                    }
                }
                _ => {}
            }
        }
        if let Some(factors) = uncontested_factors {
            if let Some(residual) = peel_direct_product_factors(
                egraph,
                base,
                false,
                &[],
                &factors,
                &[],
                target,
                negative,
                active,
                progress,
            )? {
                if residual.is_empty() {
                    return Some(Some(Vec::new()));
                }
                retain_best_residual(&mut best, residual, progress)?;
            }
        }
        Some(best)
    })();
    active.remove(&base);
    result
}

fn peel_fixed_target_from_term(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    term: &PeelTerm,
    target: &(Box<[Id]>, bool),
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<Option<Vec<PeelTerm>>> {
    match term {
        PeelTerm::Concrete { base, negative } => {
            let mut active = HashSet::new();
            peel_concrete_fixed_target(egraph, *base, *negative, target, &mut active, progress)
        }
        PeelTerm::ProductFactor { prefix, terms, suffix, negative } => {
            let mut best = None;
            for (index, (leaf, leaf_negative)) in terms.iter().enumerate() {
                progress().ok()?;
                let mut active = HashSet::new();
                let Some(selected) = peel_product_factor_target(
                    egraph,
                    *leaf,
                    *leaf_negative,
                    prefix,
                    suffix,
                    target,
                    *negative,
                    &mut active,
                    progress,
                )?
                else {
                    continue;
                };
                let mut remaining = Vec::new();
                let mut selected = Some(selected);
                for (term_index, sibling) in terms.iter().enumerate() {
                    if term_index == index {
                        for term in selected.take().expect("selected factor is unique") {
                            progress().ok()?;
                            remaining.try_reserve(1).ok()?;
                            remaining.push(term);
                        }
                    } else {
                        progress().ok()?;
                        remaining.try_reserve(1).ok()?;
                        remaining.push(PeelTerm::ProductFactor {
                            prefix: copy_ids(prefix, progress)?,
                            terms: vec![*sibling],
                            suffix: copy_ids(suffix, progress)?,
                            negative: *negative,
                        });
                    }
                }
                retain_best_residual(&mut best, remaining, progress)?;
            }
            Some(best)
        }
    }
}

fn copy_peel_term(
    term: &PeelTerm,
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<PeelTerm> {
    match term {
        PeelTerm::Concrete { base, negative } => {
            progress().ok()?;
            Some(PeelTerm::Concrete { base: *base, negative: *negative })
        }
        PeelTerm::ProductFactor { prefix, terms, suffix, negative } => {
            let prefix = copy_ids(prefix, progress)?;
            let terms = copy_signed_terms(terms, None, progress)?;
            let suffix = copy_ids(suffix, progress)?;
            progress().ok()?;
            Some(PeelTerm::ProductFactor { prefix, terms, suffix, negative: *negative })
        }
    }
}

fn copy_peel_terms(
    terms: &[PeelTerm],
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<Vec<PeelTerm>> {
    let mut copied = Vec::new();
    for term in terms {
        let term = copy_peel_term(term, progress)?;
        progress().ok()?;
        copied.try_reserve(1).ok()?;
        copied.push(term);
    }
    Some(copied)
}

fn replace_peel_term(
    terms: &[PeelTerm],
    index: usize,
    replacement: Vec<PeelTerm>,
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<Vec<PeelTerm>> {
    let mut replaced = Vec::new();
    for term in &terms[..index] {
        let term = copy_peel_term(term, progress)?;
        progress().ok()?;
        replaced.try_reserve(1).ok()?;
        replaced.push(term);
    }
    for term in replacement {
        progress().ok()?;
        replaced.try_reserve(1).ok()?;
        replaced.push(term);
    }
    for term in &terms[index + 1..] {
        let term = copy_peel_term(term, progress)?;
        progress().ok()?;
        replaced.try_reserve(1).ok()?;
        replaced.push(term);
    }
    Some(replaced)
}

fn peel_fixed_targets(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    actual: &mut Vec<PeelTerm>,
    fixed: &[(Box<[Id]>, bool)],
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<(bool, Vec<(Box<[Id]>, bool)>)> {
    // This function is a transaction: it plans every target against a deep
    // private copy and commits only after the whole fixed list has succeeded.
    let mut planned = copy_peel_terms(actual, progress)?;
    let mut unmatched = Vec::new();
    let mut any_peeled = false;
    let trace_fixed_targets = tracing::enabled!(tracing::Level::DEBUG);
    let mut omitted_fixed_targets = 0usize;
    for (target_index, target) in fixed.iter().enumerate() {
        let mut exact_zero = false;
        for factor in &target.0 {
            if is_exact_zero_matrix(egraph, *factor, progress)? {
                exact_zero = true;
                break;
            }
        }
        if exact_zero {
            any_peeled = true;
            if trace_fixed_targets && target_index < PEEL_DIAGNOSTIC_LIMIT {
                let (fixed_spine, fixed_spine_omitted) =
                    capped_peel_diagnostic_ids(egraph, &target.0);
                tracing::debug!(
                    event = "fixed_target_peel",
                    target_index,
                    result = "exact_zero",
                    negative = target.1,
                    fixed_spine = ?fixed_spine,
                    fixed_spine_omitted,
                );
            } else if trace_fixed_targets {
                omitted_fixed_targets = omitted_fixed_targets.saturating_add(1);
            }
            continue;
        }
        let mut peeled = false;
        for index in 0..planned.len() {
            progress().ok()?;
            let Some(replacement) =
                peel_fixed_target_from_term(egraph, &planned[index], target, progress)?
            else {
                continue;
            };
            if trace_fixed_targets && target_index < PEEL_DIAGNOSTIC_LIMIT {
                tracing::debug!(
                    event = "fixed_target_peel_match",
                    target_index,
                    matched_actual_position = index,
                    replacement_terms = replacement.len(),
                    replacement_terms_omitted =
                        replacement.len().saturating_sub(PEEL_DIAGNOSTIC_LIMIT),
                );
                log_peel_term(
                    egraph,
                    Some(target_index),
                    "matched_actual_before_replacement",
                    index,
                    &planned[index],
                );
                for (replacement_position, term) in
                    replacement.iter().take(PEEL_DIAGNOSTIC_LIMIT).enumerate()
                {
                    log_peel_term(
                        egraph,
                        Some(target_index),
                        "replacement_after_peeling",
                        replacement_position,
                        term,
                    );
                }
            }
            planned = replace_peel_term(&planned, index, replacement, progress)?;
            peeled = true;
            any_peeled = true;
            break;
        }
        if !peeled {
            let spine = copy_ids(&target.0, progress)?;
            progress().ok()?;
            unmatched.try_reserve(1).ok()?;
            unmatched.push((spine, target.1));
        }
        if trace_fixed_targets && target_index < PEEL_DIAGNOSTIC_LIMIT {
            let (fixed_spine, fixed_spine_omitted) = capped_peel_diagnostic_ids(egraph, &target.0);
            tracing::debug!(
                event = "fixed_target_peel",
                target_index,
                result = if peeled { "peeled" } else { "unmatched" },
                negative = target.1,
                fixed_spine = ?fixed_spine,
                fixed_spine_omitted,
                planned_actual_terms = planned.len(),
            );
        } else if trace_fixed_targets {
            omitted_fixed_targets = omitted_fixed_targets.saturating_add(1);
        }
    }
    if trace_fixed_targets {
        for (position, term) in planned.iter().take(PEEL_DIAGNOSTIC_LIMIT).enumerate() {
            log_peel_term(egraph, None, "actual_residual", position, term);
        }
        for (position, (spine, negative)) in
            unmatched.iter().take(PEEL_DIAGNOSTIC_LIMIT).enumerate()
        {
            let (fixed_spine, fixed_spine_omitted) = capped_peel_diagnostic_ids(egraph, spine);
            tracing::debug!(
                event = "fixed_target_residual_plan",
                contribution = "unmatched_fixed",
                position,
                negative,
                fixed_spine = ?fixed_spine,
                fixed_spine_omitted,
            );
        }
        tracing::debug!(
            event = "fixed_target_peel_summary",
            fixed_target_count = fixed.len(),
            omitted_fixed_targets,
            actual_residual_terms = planned.len(),
            actual_residual_terms_omitted = planned.len().saturating_sub(PEEL_DIAGNOSTIC_LIMIT),
            unmatched_fixed_terms = unmatched.len(),
            unmatched_fixed_terms_omitted = unmatched.len().saturating_sub(PEEL_DIAGNOSTIC_LIMIT),
            any_peeled,
        );
    }
    for _ in actual.iter() {
        progress().ok()?;
    }
    *actual = planned;
    Some((any_peeled, unmatched))
}

fn materialize_fixed_spines(
    egraph: &mut EGraph<MxxLang, MxxAnalysis>,
    fixed: Vec<(Box<[Id]>, bool)>,
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<Vec<Id>> {
    let mut output = Vec::new();
    for (factors, negative) in fixed {
        let base = match factors.as_ref() {
            [] => return None,
            [factor] => *factor,
            _ => {
                progress().ok()?;
                egraph.add(MxxLang::MatrixMultiply(factors))
            }
        };
        let materialized = if negative {
            progress().ok()?;
            egraph.add(MxxLang::MatrixNegate([base]))
        } else {
            base
        };
        progress().ok()?;
        output.try_reserve(1).ok()?;
        output.push(materialized);
    }
    Some(output)
}

/// Fixed targets may themselves become equal after binder instantiation.  Drop
/// only exact opposite spines before touching the grouped actual case.
fn cancel_fixed_spines(
    fixed: Vec<(Box<[Id]>, bool)>,
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<(Vec<(Box<[Id]>, bool)>, bool)> {
    let mut cancelled = Vec::new();
    let mut positive = HashMap::<Box<[Id]>, Vec<usize>>::new();
    let mut negative = HashMap::<Box<[Id]>, Vec<usize>>::new();
    let mut any = false;
    for (index, (spine, negative_sign)) in fixed.iter().enumerate() {
        progress().ok()?;
        cancelled.try_reserve(1).ok()?;
        cancelled.push(false);
        let opposite = if *negative_sign { &mut positive } else { &mut negative };
        if let Some(other) = opposite.get_mut(spine).and_then(Vec::pop) {
            cancelled[index] = true;
            cancelled[other] = true;
            any = true;
        } else {
            let same = if *negative_sign { &mut negative } else { &mut positive };
            let spine = copy_ids(spine, progress)?;
            progress().ok()?;
            let entries = same.entry(spine).or_default();
            progress().ok()?;
            entries.try_reserve(1).ok()?;
            entries.push(index);
        }
    }
    let mut remaining = Vec::new();
    for (term, cancelled) in fixed.into_iter().zip(cancelled) {
        if !cancelled {
            progress().ok()?;
            remaining.try_reserve(1).ok()?;
            remaining.push(term);
        }
    }
    Some((remaining, any))
}

fn materialize_peel_terms(
    egraph: &mut EGraph<MxxLang, MxxAnalysis>,
    root: Id,
    terms: Vec<PeelTerm>,
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<Vec<Id>> {
    let mut output = Vec::new();
    for term in terms {
        let (base, negative) = match term {
            PeelTerm::Concrete { base, negative } => (base, negative),
            PeelTerm::ProductFactor { prefix, terms, suffix, negative } => {
                let mut factor_terms = Vec::new();
                for (base, negative) in terms {
                    progress().ok()?;
                    factor_terms.try_reserve(1).ok()?;
                    factor_terms.push(PeelTerm::Concrete { base, negative });
                }
                let factor = materialize_peel_terms(egraph, root, factor_terms, progress)?;
                let factor = build_additive_terms(egraph, root, factor);
                let mut factors = Vec::new();
                for prefix_factor in prefix {
                    progress().ok()?;
                    factors.try_reserve(1).ok()?;
                    factors.push(prefix_factor);
                }
                progress().ok()?;
                factors.try_reserve(1).ok()?;
                factors.push(factor);
                for suffix_factor in suffix {
                    progress().ok()?;
                    factors.try_reserve(1).ok()?;
                    factors.push(suffix_factor);
                }
                let base = match factors.as_slice() {
                    [] => return None,
                    [factor] => *factor,
                    _ => {
                        progress().ok()?;
                        egraph.add(MxxLang::MatrixMultiply(factors.into_boxed_slice()))
                    }
                };
                (base, negative)
            }
        };
        let materialized = if negative {
            progress().ok()?;
            egraph.add(MxxLang::MatrixNegate([base]))
        } else {
            base
        };
        progress().ok()?;
        output.try_reserve(1).ok()?;
        output.push(materialized);
    }
    Some(output)
}

/// Internal build path shared by production logging and the cfg(test) sink.
/// The sink is synchronous and ephemeral; it cannot affect construction.
fn build_binder_aware_pointwise_add_switch_cancellation_with_sink(
    egraph: &mut EGraph<MxxLang, MxxAnalysis>,
    root: Id,
    selector: Id,
    plan: BinderAwarePointwiseAddSwitchPlan,
    progress: &mut dyn FnMut() -> Result<(), ()>,
    should_collect: &mut dyn FnMut() -> bool,
    sink: &mut dyn FnMut(&BinderBuildReject),
) -> Option<Id> {
    let mut normalized_cases = Vec::new();
    for (case_index, terms) in plan.case_terms.iter().enumerate() {
        progress().ok()?;
        let index = egraph.add(MxxLang::IntConst(BigInt::from(case_index)));
        let mut mapped_bases = Vec::new();
        for fixed in &plan.fixed_bases {
            let instantiated =
                family::instantiate_shared_element(egraph, *fixed, plan.binder, index, progress)
                    .ok()?;
            progress().ok()?;
            mapped_bases.try_reserve(1).ok()?;
            mapped_bases.push(egraph.find(instantiated));
        }
        let mut actual = Vec::new();
        for term in terms {
            progress().ok()?;
            actual.try_reserve(1).ok()?;
            actual.push(PeelTerm::Concrete { base: term.base, negative: term.negative });
        }
        let mut mapped_fixed = Vec::new();
        for occurrence in &plan.fixed_occurrences {
            let mapped = mapped_bases[occurrence.base_index];
            progress().ok()?;
            mapped_fixed.try_reserve(1).ok()?;
            mapped_fixed.push((mapped, occurrence.negative));
        }
        log_mapped_fixed_occurrences(egraph, case_index, &plan.fixed_occurrences, &mapped_fixed);
        let Some(mapped_fixed_spines) =
            signed_ordered_monomial_spines(egraph, &mapped_fixed, progress)
        else {
            report_binder_build_reject(
                egraph,
                case_index,
                BinderBuildRejectStage::FixedSignedReject,
                &[],
                &[],
                should_collect,
                sink,
            );
            return None;
        };
        log_pre_cancel_mapped_fixed_spines(egraph, case_index, &mapped_fixed_spines);
        let (mapped_fixed_spines, fixed_cancelled) =
            cancel_fixed_spines(mapped_fixed_spines, progress)?;
        let (any_peeled, unmatched_fixed) =
            peel_fixed_targets(egraph, &mut actual, &mapped_fixed_spines, progress)?;
        if !any_peeled && !fixed_cancelled {
            let mut diagnostic_actual = Vec::new();
            for term in terms {
                progress().ok()?;
                progress().ok()?;
                diagnostic_actual.try_reserve(1).ok()?;
                diagnostic_actual.push((term.base, term.negative));
            }
            report_binder_build_reject(
                egraph,
                case_index,
                BinderBuildRejectStage::NoExactCancellation,
                &diagnostic_actual,
                &mapped_fixed,
                should_collect,
                sink,
            );
            return None;
        }
        // Peeling is entirely read-only.  Only the surviving residual path is
        // materialized after every fixed target has matched.
        let mut remaining = materialize_peel_terms(egraph, root, actual, progress)?;
        for term in materialize_fixed_spines(egraph, unmatched_fixed, progress)? {
            progress().ok()?;
            remaining.try_reserve(1).ok()?;
            remaining.push(term);
        }
        if remaining.iter().any(|term| egraph.find(*term) == egraph.find(root)) {
            report_binder_build_reject(
                egraph,
                case_index,
                BinderBuildRejectStage::RootCycle,
                &[],
                &[],
                should_collect,
                sink,
            );
            return None;
        }
        progress().ok()?;
        normalized_cases.try_reserve(1).ok()?;
        normalized_cases.push(remaining);
    }
    if equivalent_switch_exists(egraph, root, selector, &normalized_cases) {
        report_binder_build_reject(
            egraph,
            0,
            BinderBuildRejectStage::Equivalent,
            &[],
            &[],
            should_collect,
            sink,
        );
        return None;
    }
    let mut cases = Vec::new();
    progress().ok()?;
    cases.try_reserve(1).ok()?;
    cases.push(selector);
    for terms in normalized_cases {
        progress().ok()?;
        cases.try_reserve(1).ok()?;
        cases.push(build_additive_terms(egraph, root, terms));
    }
    progress().ok()?;
    let candidate = MxxLang::Switch(cases.into_boxed_slice());
    let candidate_id = egraph.lookup(candidate.clone());
    let root = egraph.find(root);
    let candidate_is_direct_child = if let Some(candidate_id) = candidate_id {
        let candidate_id = egraph.find(candidate_id);
        let mut direct = false;
        for node in &egraph[root].nodes {
            progress().ok()?;
            let MxxLang::MatrixAdd(children) = node else { continue };
            for child in children {
                progress().ok()?;
                if egraph.find(*child) == candidate_id {
                    direct = true;
                    break;
                }
            }
            if direct {
                break;
            }
        }
        direct
    } else {
        false
    };
    if candidate_is_direct_child {
        let MxxLang::Switch(mut guarded_cases) = candidate else { return None };
        let first_case = *guarded_cases.get(1)?;
        progress().ok()?;
        guarded_cases[1] = egraph.add(MxxLang::MatrixAdd(vec![first_case].into_boxed_slice()));
        let guarded = MxxLang::Switch(guarded_cases);
        progress().ok()?;
        if egraph.lookup(guarded.clone()).is_some_and(|existing| egraph.find(existing) == root) {
            return None;
        }
        return Some(egraph.add(guarded));
    }
    Some(egraph.add(candidate))
}

#[cfg(test)]
fn build_binder_aware_pointwise_add_switch_cancellation_with_diagnostic(
    egraph: &mut EGraph<MxxLang, MxxAnalysis>,
    root: Id,
    selector: Id,
    plan: BinderAwarePointwiseAddSwitchPlan,
    sink: &mut dyn FnMut(&BinderBuildReject),
) -> Option<Id> {
    let mut progress = || Ok(());
    let mut always_collect = || true;
    build_binder_aware_pointwise_add_switch_cancellation_with_sink(
        egraph,
        root,
        selector,
        plan,
        &mut progress,
        &mut always_collect,
        sink,
    )
}

/// Diagnostic-only physical product view. It never associates products, and
/// reports up to eight ordered factors even when one is itself a product.
#[derive(Clone, Debug, Eq, PartialEq)]
enum RetainedProductSpine {
    Absent {
        leaf: usize,
    },
    Ambiguous {
        leaf: usize,
    },
    Direct {
        leaf: usize,
        factors: Box<[usize]>,
        factor_adds: Box<[RetainedFactorAdd]>,
        omitted: usize,
    },
}

/// Bounded, failure-only physical Add evidence for one retained product
/// factor. It distinguishes no physical Add from exactly one and competing
/// physical Adds without retaining e-graph state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RetainedFactorAdd {
    Absent,
    Unique,
    Ambiguous,
}

/// Failure-only ordered-product evidence for one retained signed root.  This
/// is deliberately derived after the one-shot diagnostic gate: no traversal
/// or descriptor allocation occurs on the normal rewrite path.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct ProductLeafDiagnostic {
    root: usize,
    leaves: Box<[LeafView]>,
    leaf_omitted: ProductLeafOmission,
    status: ProductLeafStatus,
}

/// Exact only when the bounded traversal has ruled out any unvisited product
/// branch. Otherwise the retained prefix is truthful and this reports the
/// minimum number of flattened leaves known to be absent from it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum ProductLeafOmission {
    Exact(usize),
    AtLeast(usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum ProductLeafStatus {
    Complete,
    Ambiguous { at: usize },
    Cycle { at: usize },
    Truncated { pending: usize },
}

/// A bounded view of one physical non-product leaf e-class.  Child ids are
/// canonicalized at collection time, so the view is stable across a rebuilt
/// e-graph without retaining nodes or paths.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct LeafView {
    eclass: usize,
    nodes: Box<[LeafNodeDescriptor]>,
    nodes_omitted: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum LeafNodeDescriptor {
    Atom {
        source_id: u32,
        source_kind: &'static str,
        relation_role: Option<AtomicRelationRole>,
        indices: Box<[usize]>,
        indices_omitted: usize,
    },
    MatrixConstant {
        spec: u32,
        value_kind: &'static str,
    },
    HashPlain {
        query: u32,
        arguments: Box<[usize]>,
        arguments_omitted: usize,
    },
    Switch {
        selector: Option<usize>,
        cases: Box<[usize]>,
        cases_omitted: usize,
    },
    Other {
        operator_name: &'static str,
        children: Box<[usize]>,
        children_omitted: usize,
    },
}

fn atomic_source_kind(source: &AtomicSourceKey) -> &'static str {
    match source {
        AtomicSourceKey::ProtocolInput(_) => "protocol-input",
        AtomicSourceKey::GraphWire(_) => "graph-wire",
        AtomicSourceKey::ExplicitLarge(_) => "explicit-large",
        AtomicSourceKey::SequentialState(_) => "sequential-state",
        AtomicSourceKey::SequentialRecurrence { .. } => "sequential-recurrence",
        AtomicSourceKey::Sampler(_) => "sampler",
    }
}

fn matrix_constant_value_kind(value: &MatrixConstantValue) -> &'static str {
    match value {
        MatrixConstantValue::Zero => "zero",
        MatrixConstantValue::Identity => "identity",
        MatrixConstantValue::UnitRow { .. } => "unit-row",
        MatrixConstantValue::UnitColumn { .. } => "unit-column",
        MatrixConstantValue::Gadget { .. } => "gadget",
        MatrixConstantValue::PowerOfBase { .. } => "power-of-base",
        MatrixConstantValue::Rotation { .. } => "rotation",
        MatrixConstantValue::Polynomial { .. } => "polynomial",
    }
}

fn capped_canonical_children(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    children: &[Id],
) -> (Box<[usize]>, usize) {
    const CHILD_LIMIT: usize = 8;
    (
        children.iter().take(CHILD_LIMIT).map(|child| usize::from(egraph.find(*child))).collect(),
        children.len().saturating_sub(CHILD_LIMIT),
    )
}

fn leaf_node_descriptor(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    node: &MxxLang,
) -> LeafNodeDescriptor {
    match node {
        MxxLang::Atom { source, indices } => {
            let (indices, indices_omitted) = capped_canonical_children(egraph, indices);
            let source_kind = egraph
                .analysis
                .symbols
                .atomic_sources
                .get(source.0)
                .map(|descriptor| atomic_source_kind(&descriptor.key))
                .unwrap_or("missing-source");
            let relation_role = egraph
                .analysis
                .symbols
                .atomic_sources
                .get(source.0)
                .and_then(|descriptor| descriptor.relation_role);
            LeafNodeDescriptor::Atom {
                source_id: source.0,
                source_kind,
                relation_role,
                indices,
                indices_omitted,
            }
        }
        MxxLang::MatrixConstant(spec) => LeafNodeDescriptor::MatrixConstant {
            spec: spec.0,
            value_kind: egraph
                .analysis
                .symbols
                .matrix_constants
                .get(spec.0)
                .map(|constant| matrix_constant_value_kind(&constant.value))
                .unwrap_or("missing-spec"),
        },
        MxxLang::HashPlain { query, arguments } => {
            let (arguments, arguments_omitted) = capped_canonical_children(egraph, arguments);
            LeafNodeDescriptor::HashPlain { query: query.0, arguments, arguments_omitted }
        }
        MxxLang::Switch(cases) => {
            let selector = cases.first().map(|selector| usize::from(egraph.find(*selector)));
            let (cases, cases_omitted) =
                capped_canonical_children(egraph, cases.get(1..).unwrap_or_default());
            LeafNodeDescriptor::Switch { selector, cases, cases_omitted }
        }
        other => {
            let (children, children_omitted) = capped_canonical_children(egraph, other.children());
            LeafNodeDescriptor::Other {
                operator_name: other.operator_name(),
                children,
                children_omitted,
            }
        }
    }
}

fn leaf_view(egraph: &EGraph<MxxLang, MxxAnalysis>, id: Id) -> LeafView {
    const NODE_LIMIT: usize = 4;
    let id = egraph.find(id);
    let mut nodes = egraph[id].nodes.iter().collect::<Vec<_>>();
    nodes.sort_unstable();
    let nodes_omitted = nodes.len().saturating_sub(NODE_LIMIT);
    LeafView {
        eclass: usize::from(id),
        nodes: nodes
            .into_iter()
            .take(NODE_LIMIT)
            .map(|node| leaf_node_descriptor(egraph, node))
            .collect(),
        nodes_omitted,
    }
}

fn retained_product_leaves(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    roots: &[Id],
) -> Vec<ProductLeafDiagnostic> {
    const LEAF_LIMIT: usize = 16;
    const PRODUCT_VISIT_LIMIT: usize = 32;

    enum ProductWork {
        Visit(Id),
        Exit(Id),
        Schedule { factors: Box<[Id]>, next: usize },
    }

    fn pending_product_work(work: &[ProductWork]) -> usize {
        work.iter()
            .map(|item| match item {
                ProductWork::Visit(_) => 1,
                ProductWork::Exit(_) => 0,
                ProductWork::Schedule { factors, next } => factors.len().saturating_sub(*next),
            })
            .sum()
    }

    roots
        .iter()
        .map(|root| {
            let root = egraph.find(*root);
            let mut leaves = Vec::new();
            let mut work = vec![ProductWork::Visit(root)];
            let mut active = HashSet::new();
            let mut visits = 0;
            let (status, leaf_omitted) = loop {
                let Some(item) = work.pop() else {
                    break (ProductLeafStatus::Complete, ProductLeafOmission::Exact(0));
                };
                match item {
                    ProductWork::Exit(term) => {
                        active.remove(&term);
                    }
                    ProductWork::Schedule { factors, next } => {
                        if let Some(factor) = factors.get(next).copied() {
                            work.push(ProductWork::Schedule { factors, next: next + 1 });
                            work.push(ProductWork::Visit(egraph.find(factor)));
                        }
                    }
                    ProductWork::Visit(term) => {
                        let term = egraph.find(term);
                        if visits == PRODUCT_VISIT_LIMIT {
                            break (
                                ProductLeafStatus::Truncated {
                                    pending: pending_product_work(&work) + 1,
                                },
                                ProductLeafOmission::AtLeast(0),
                            );
                        }
                        visits += 1;
                        match physical_product_factors(egraph, term) {
                            PhysicalStructure::Absent => {
                                if leaves.len() == LEAF_LIMIT {
                                    let pending = pending_product_work(&work) + 1;
                                    break (
                                        ProductLeafStatus::Truncated { pending },
                                        if pending == 1 {
                                            ProductLeafOmission::Exact(1)
                                        } else {
                                            ProductLeafOmission::AtLeast(1)
                                        },
                                    );
                                }
                                leaves.push(leaf_view(egraph, term));
                            }
                            PhysicalStructure::Ambiguous => {
                                break (
                                    ProductLeafStatus::Ambiguous { at: usize::from(term) },
                                    if pending_product_work(&work) == 0 {
                                        ProductLeafOmission::Exact(0)
                                    } else {
                                        ProductLeafOmission::AtLeast(0)
                                    },
                                );
                            }
                            PhysicalStructure::Unique(factors) => {
                                if !active.insert(term) {
                                    break (
                                        ProductLeafStatus::Cycle { at: usize::from(term) },
                                        if pending_product_work(&work) == 0 {
                                            ProductLeafOmission::Exact(0)
                                        } else {
                                            ProductLeafOmission::AtLeast(0)
                                        },
                                    );
                                }
                                work.push(ProductWork::Exit(term));
                                work.push(ProductWork::Schedule { factors, next: 0 });
                            }
                        }
                    }
                }
            };
            ProductLeafDiagnostic {
                root: usize::from(root),
                leaf_omitted,
                leaves: leaves.into_boxed_slice(),
                status,
            }
        })
        .collect()
}

fn retained_factor_add(egraph: &EGraph<MxxLang, MxxAnalysis>, factor: Id) -> RetainedFactorAdd {
    match physical_add_terms(egraph, factor) {
        PhysicalStructure::Absent => RetainedFactorAdd::Absent,
        PhysicalStructure::Unique(_) => RetainedFactorAdd::Unique,
        PhysicalStructure::Ambiguous => RetainedFactorAdd::Ambiguous,
    }
}

fn retained_product_spines(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    leaves: &[Id],
) -> Vec<RetainedProductSpine> {
    const FACTOR_LIMIT: usize = 8;
    leaves
        .iter()
        .map(|leaf| {
            let leaf = egraph.find(*leaf);
            let products = egraph[leaf]
                .nodes
                .iter()
                .filter_map(|node| match node {
                    MxxLang::MatrixMultiply(factors) => {
                        Some(factors.iter().map(|factor| egraph.find(*factor)).collect::<Vec<_>>())
                    }
                    _ => None,
                })
                .collect::<BTreeSet<_>>();
            match products.len() {
                0 => RetainedProductSpine::Absent { leaf: usize::from(leaf) },
                1 => {
                    let factors = products.into_iter().next().expect("one product");
                    let omitted = factors.len().saturating_sub(FACTOR_LIMIT);
                    RetainedProductSpine::Direct {
                        leaf: usize::from(leaf),
                        factors: factors
                            .iter()
                            .copied()
                            .take(FACTOR_LIMIT)
                            .map(usize::from)
                            .collect(),
                        factor_adds: factors
                            .iter()
                            .copied()
                            .take(FACTOR_LIMIT)
                            .map(|factor| retained_factor_add(egraph, factor))
                            .collect(),
                        omitted,
                    }
                }
                _ => RetainedProductSpine::Ambiguous { leaf: usize::from(leaf) },
            }
        })
        .collect()
}

fn build_additive_terms(egraph: &mut EGraph<MxxLang, MxxAnalysis>, root: Id, terms: Vec<Id>) -> Id {
    match terms.len() {
        0 => match egraph[egraph.find(root)].data.sort.clone() {
            Ok(MxxSort::Matrix(matrix_type)) => {
                let spec = egraph.analysis.symbols.matrix_constants.intern(
                    super::identity::MatrixConstantSpec {
                        matrix_type,
                        value: MatrixConstantValue::Zero,
                    },
                );
                egraph.add(MxxLang::MatrixConstant(super::identity::MatrixConstantSpecId(spec)))
            }
            _ => unreachable!("pointwise matrix Add has matrix sort"),
        },
        1 => terms[0],
        _ => egraph.add(MxxLang::MatrixAdd(terms.into_boxed_slice())),
    }
}

/// The direct pointwise Switch builder unions its output with an enclosing
/// Add. A singleton case can therefore be one of that Add's original
/// children. Keep a one-child Add wrapper for exactly this path, so rebuild
/// never canonicalizes an old parent e-node into a self-cycle. Binder-aware
/// builders have their own instantiation safety checks and retain their
/// original singleton representation.
fn build_detached_pointwise_terms(
    egraph: &mut EGraph<MxxLang, MxxAnalysis>,
    root: Id,
    terms: Vec<Id>,
) -> Id {
    match terms.len() {
        1 => egraph.add(MxxLang::MatrixAdd(terms.into_boxed_slice())),
        _ => build_additive_terms(egraph, root, terms),
    }
}

fn unique_switch_cases(egraph: &EGraph<MxxLang, MxxAnalysis>, term: Id) -> Option<Box<[Id]>> {
    match physical_switch_cases(egraph, term) {
        PhysicalStructure::Unique(cases) => Some(cases),
        PhysicalStructure::Absent | PhysicalStructure::Ambiguous => None,
    }
}

fn has_physical_switch(egraph: &EGraph<MxxLang, MxxAnalysis>, term: Id) -> bool {
    egraph[egraph.find(term)].nodes.iter().any(|node| matches!(node, MxxLang::Switch(_)))
}

fn signed_additive_term(egraph: &EGraph<MxxLang, MxxAnalysis>, term: Id) -> Option<(Id, bool)> {
    match physical_negated_base(egraph, term) {
        PhysicalStructure::Unique(base) => (base != egraph.find(term)).then_some((base, true)),
        PhysicalStructure::Absent => Some((egraph.find(term), false)),
        PhysicalStructure::Ambiguous => None,
    }
}

/// Expands every finite physical Add/Negate representation into signed
/// leaves, then accepts a representative only when every additive
/// representation has the same canonical signed polynomial. Non-additive
/// e-nodes are not alternative decompositions. The representative preserves
/// its physical leaf ids for later binder instantiation; consensus is checked
/// with ordered product keys, so it is insensitive to product association but
/// not product order. Cycles, empty Adds, and competing representations with
/// different polynomials remain fail-closed.
fn signed_additive_leaves(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    terms: &[Id],
) -> Option<Vec<(Id, bool)>> {
    let mut no_progress = || Ok(());
    signed_additive_leaves_with_visit_and_progress(egraph, terms, |_| {}, &mut no_progress)
}

/// The callback is test instrumentation only at its call sites.
#[cfg(test)]
fn signed_additive_leaves_with_visit<F>(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    terms: &[Id],
    visit: F,
) -> Option<Vec<(Id, bool)>>
where
    F: FnMut(Id),
{
    let mut no_progress = || Ok(());
    signed_additive_leaves_with_visit_and_progress(egraph, terms, visit, &mut no_progress)
}

/// The binder polynomial path uses the same additive consensus while charging
/// the existing shared rewrite budget for every representative occurrence it
/// copies or allocates.  Other callers keep their read-only behavior.
fn signed_additive_leaves_with_progress(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    terms: &[Id],
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<Vec<(Id, bool)>> {
    signed_additive_leaves_with_visit_and_progress(egraph, terms, |_| {}, progress)
}

fn signed_additive_leaves_with_visit_and_progress<F>(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    terms: &[Id],
    visit: F,
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<Vec<(Id, bool)>>
where
    F: FnMut(Id),
{
    signed_additive_leaves_with_specialization_and_progress(
        egraph,
        terms,
        visit,
        |_| Ok(None),
        progress,
    )
}

fn signed_additive_leaves_with_specialization_and_progress<F, S>(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    terms: &[Id],
    mut visit: F,
    mut specialize: S,
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<Vec<(Id, bool)>>
where
    F: FnMut(Id),
    S: FnMut(Id) -> Result<Option<(Id, bool)>, ()>,
{
    type Canonical = Vec<(Box<[Id]>, bool, usize)>;

    #[derive(Clone)]
    struct Consensus {
        representative: Vec<(Id, bool)>,
        canonical: Canonical,
    }

    fn normalized_counts(counts: BTreeMap<Box<[Id]>, (usize, usize)>) -> Canonical {
        counts
            .into_iter()
            .filter_map(|(key, (positive, negative))| match positive.cmp(&negative) {
                std::cmp::Ordering::Greater => Some((key, false, positive - negative)),
                std::cmp::Ordering::Less => Some((key, true, negative - positive)),
                std::cmp::Ordering::Equal => None,
            })
            .collect()
    }

    /// Add/Negate consensus needs a stable key for an outer product even
    /// when one factor is relation-unioned with another product shape.  Such
    /// a factor is an atomic key here; a physical product cycle still rejects.
    fn additive_product_leaves(
        egraph: &EGraph<MxxLang, MxxAnalysis>,
        term: Id,
    ) -> Option<Box<[Id]>> {
        fn collect(
            egraph: &EGraph<MxxLang, MxxAnalysis>,
            term: Id,
            active: &mut HashSet<Id>,
            leaves: &mut Vec<Id>,
        ) -> Option<()> {
            let term = egraph.find(term);
            match physical_product_factors(egraph, term) {
                PhysicalStructure::Absent | PhysicalStructure::Ambiguous => leaves.push(term),
                PhysicalStructure::Unique(factors) => {
                    if !active.insert(term) {
                        return None;
                    }
                    for factor in factors.iter() {
                        collect(egraph, *factor, active, leaves)?;
                    }
                    active.remove(&term);
                }
            }
            Some(())
        }

        let mut leaves = Vec::new();
        collect(egraph, term, &mut HashSet::new(), &mut leaves)?;
        Some(leaves.into_boxed_slice())
    }

    fn canonical_from_terms(
        egraph: &EGraph<MxxLang, MxxAnalysis>,
        terms: &[(Id, bool)],
    ) -> Option<Canonical> {
        let mut counts = BTreeMap::<Box<[Id]>, (usize, usize)>::new();
        for (term, negative) in terms {
            // A relation may legitimately union a product with a structurally
            // different target.  Add/Negate consensus must still be able to
            // treat that exact e-class as one atomic additive leaf; only
            // callers that need a product cancellation key reject it later.
            let key = additive_product_leaves(egraph, *term)?;
            let counts = counts.entry(key).or_default();
            let count = if *negative { &mut counts.1 } else { &mut counts.0 };
            *count = count.checked_add(1)?;
        }
        Some(normalized_counts(counts))
    }

    fn add_canonical(
        counts: &mut BTreeMap<Box<[Id]>, (usize, usize)>,
        canonical: &Canonical,
    ) -> Option<()> {
        for (key, negative, count) in canonical {
            let counts = counts.entry(key.clone()).or_default();
            let total = if *negative { &mut counts.1 } else { &mut counts.0 };
            *total = total.checked_add(*count)?;
        }
        Some(())
    }

    fn consensus<F, S>(
        egraph: &EGraph<MxxLang, MxxAnalysis>,
        term: Id,
        memo: &mut HashMap<Id, Consensus>,
        active: &mut HashSet<Id>,
        visit: &mut F,
        specialize: &mut S,
        progress: &mut dyn FnMut() -> Result<(), ()>,
    ) -> Option<Consensus>
    where
        F: FnMut(Id),
        S: FnMut(Id) -> Result<Option<(Id, bool)>, ()>,
    {
        let term = egraph.find(term);
        if let Some(existing) = memo.get(&term) {
            for _ in &existing.representative {
                progress().ok()?;
            }
            return Some(existing.clone());
        }
        if !active.insert(term) {
            return None;
        }
        visit(term);
        let result = (|| {
            let mut agreed: Option<Consensus> = None;
            let mut has_additive_structure = false;
            if let Some((base, negative)) = specialize(term).ok()? {
                has_additive_structure = true;
                let mut candidate =
                    consensus(egraph, base, memo, active, visit, specialize, progress)?;
                if negative {
                    for (_, negative) in &mut candidate.representative {
                        *negative = !*negative;
                    }
                    for (_, negative, _) in &mut candidate.canonical {
                        *negative = !*negative;
                    }
                }
                agreed = Some(candidate);
            }
            for node in &egraph[term].nodes {
                let candidate = match node {
                    MxxLang::MatrixNegate([base]) => {
                        has_additive_structure = true;
                        let mut candidate =
                            consensus(egraph, *base, memo, active, visit, specialize, progress)?;
                        for (_, negative) in &mut candidate.representative {
                            *negative = !*negative;
                        }
                        for (_, negative, _) in &mut candidate.canonical {
                            *negative = !*negative;
                        }
                        candidate
                    }
                    MxxLang::MatrixAdd(children) => {
                        has_additive_structure = true;
                        if children.is_empty() {
                            return None;
                        }
                        let mut representative = Vec::new();
                        let mut counts = BTreeMap::new();
                        for child in children.iter() {
                            let child = consensus(
                                egraph, *child, memo, active, visit, specialize, progress,
                            )?;
                            for occurrence in child.representative {
                                progress().ok()?;
                                representative.try_reserve(1).ok()?;
                                representative.push(occurrence);
                            }
                            add_canonical(&mut counts, &child.canonical)?;
                        }
                        Consensus { representative, canonical: normalized_counts(counts) }
                    }
                    _ => continue,
                };
                if let Some(previous) = &agreed {
                    if previous.canonical != candidate.canonical {
                        return None;
                    }
                } else {
                    agreed = Some(candidate);
                }
            }
            if has_additive_structure {
                agreed
            } else {
                progress().ok()?;
                let mut representative = Vec::new();
                representative.try_reserve_exact(1).ok()?;
                representative.push((term, false));
                Some(Consensus {
                    canonical: canonical_from_terms(egraph, &representative)?,
                    representative,
                })
            }
        })();
        active.remove(&term);
        let result = result?;
        for _ in &result.representative {
            progress().ok()?;
        }
        progress().ok()?;
        memo.insert(term, result.clone());
        Some(result)
    }

    let mut memo = HashMap::new();
    let mut active = HashSet::new();
    let mut output = Vec::new();
    for term in terms {
        let consensus = consensus(
            egraph,
            *term,
            &mut memo,
            &mut active,
            &mut visit,
            &mut specialize,
            progress,
        )?;
        for occurrence in consensus.representative {
            progress().ok()?;
            output.try_reserve(1).ok()?;
            output.push(occurrence);
        }
    }
    Some(output)
}

/// A read-only signed noncommutative polynomial view used only for mapped
/// fixed targets while building one binder case.  Every direct product root is
/// flattened and must agree exactly. Nested factors inspect every direct
/// product witness and retain the lexicographically smallest finite acyclic
/// output. Product-only cycles reject, while any non-product representative
/// (including `Switch`) stays opaque and is never enumerated.
fn mapped_fixed_product_consensus_with_progress(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    root: Id,
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<Option<Box<[Id]>>> {
    fn product_witnesses(
        egraph: &EGraph<MxxLang, MxxAnalysis>,
        term: Id,
        progress: &mut dyn FnMut() -> Result<(), ()>,
    ) -> Option<(Vec<Box<[Id]>>, bool)> {
        let term = egraph.find(term);
        let mut witnesses = Vec::new();
        let mut has_non_product = false;
        for node in &egraph[term].nodes {
            progress().ok()?;
            let MxxLang::MatrixMultiply(factors) = node else {
                has_non_product = true;
                continue;
            };
            progress().ok()?;
            let mut canonical = Vec::new();
            canonical.try_reserve_exact(factors.len()).ok()?;
            for factor in factors {
                progress().ok()?;
                canonical.push(egraph.find(*factor));
            }
            progress().ok()?;
            witnesses.try_reserve(1).ok()?;
            witnesses.push(canonical.into_boxed_slice());
        }
        Some((witnesses, has_non_product))
    }

    fn compare_sequences(
        left: &[Id],
        right: &[Id],
        progress: &mut dyn FnMut() -> Result<(), ()>,
    ) -> Option<std::cmp::Ordering> {
        for (left, right) in left.iter().zip(right) {
            progress().ok()?;
            match left.cmp(right) {
                std::cmp::Ordering::Equal => {}
                order => return Some(order),
            }
        }
        progress().ok()?;
        Some(left.len().cmp(&right.len()))
    }

    fn flatten_factor(
        egraph: &EGraph<MxxLang, MxxAnalysis>,
        term: Id,
        active: &mut HashSet<Id>,
        progress: &mut dyn FnMut() -> Result<(), ()>,
        output: &mut Vec<Id>,
    ) -> Option<bool> {
        let term = egraph.find(term);
        if active.contains(&term) {
            return Some(false);
        }
        let (witnesses, has_non_product) = product_witnesses(egraph, term, progress)?;
        if witnesses.is_empty() {
            progress().ok()?;
            output.try_reserve(1).ok()?;
            output.push(term);
            return Some(true);
        }
        debug_assert!(active.insert(term));
        let checkpoint = output.len();
        let mut best: Option<Vec<Id>> = None;
        for factors in witnesses {
            output.truncate(checkpoint);
            if factors.is_empty() {
                continue;
            }
            let mut finite = true;
            for factor in factors {
                progress().ok()?;
                if !flatten_factor(egraph, factor, active, progress, output)? {
                    finite = false;
                    break;
                }
            }
            if finite {
                let candidate_len = output.len().checked_sub(checkpoint)?;
                progress().ok()?;
                let mut candidate = Vec::new();
                candidate.try_reserve_exact(candidate_len).ok()?;
                for factor in &output[checkpoint..] {
                    progress().ok()?;
                    candidate.push(*factor);
                }
                let replace = match &best {
                    None => true,
                    Some(previous) => {
                        compare_sequences(&candidate, previous, progress)? ==
                            std::cmp::Ordering::Less
                    }
                };
                if replace {
                    best = Some(candidate);
                }
            }
        }
        output.truncate(checkpoint);
        active.remove(&term);
        if let Some(best) = best {
            output.try_reserve_exact(best.len()).ok()?;
            for factor in best {
                progress().ok()?;
                output.push(factor);
            }
            return Some(true);
        }
        if has_non_product {
            progress().ok()?;
            output.try_reserve(1).ok()?;
            output.push(term);
            Some(true)
        } else {
            Some(false)
        }
    }

    fn flatten_witness(
        egraph: &EGraph<MxxLang, MxxAnalysis>,
        factors: &[Id],
        active: &mut HashSet<Id>,
        progress: &mut dyn FnMut() -> Result<(), ()>,
    ) -> Option<Vec<Id>> {
        let mut output = Vec::new();
        for factor in factors {
            progress().ok()?;
            if !flatten_factor(egraph, *factor, active, progress, &mut output)? {
                return None;
            }
        }
        Some(output)
    }

    let root = egraph.find(root);
    let (witnesses, _) = product_witnesses(egraph, root, progress)?;
    if witnesses.is_empty() {
        return Some(None);
    }
    if witnesses.iter().any(|factors| factors.is_empty()) {
        return None;
    }
    let mut agreed: Option<Vec<Id>> = None;
    for factors in witnesses {
        let flattened = flatten_witness(egraph, &factors, &mut HashSet::new(), progress)?;
        if let Some(previous) = &agreed {
            if previous.len() != flattened.len() {
                return None;
            }
            for (left, right) in previous.iter().zip(&flattened) {
                progress().ok()?;
                if left != right {
                    return None;
                }
            }
        } else {
            agreed = Some(flattened);
        }
    }
    Some(Some(agreed?.into_boxed_slice()))
}

fn cancel_signed_spines(terms: &mut Vec<(Box<[Id]>, bool)>) {
    let mut cancelled = vec![false; terms.len()];
    let mut positive = HashMap::<Box<[Id]>, Vec<usize>>::new();
    let mut negative = HashMap::<Box<[Id]>, Vec<usize>>::new();
    for (index, (spine, negative_sign)) in terms.iter().enumerate() {
        let opposite = if *negative_sign { &mut positive } else { &mut negative };
        if let Some(other) = opposite.get_mut(spine).and_then(Vec::pop) {
            cancelled[index] = true;
            cancelled[other] = true;
        } else {
            let same = if *negative_sign { &mut negative } else { &mut positive };
            same.entry(spine.clone()).or_default().push(index);
        }
    }
    let mut retained = Vec::with_capacity(terms.len());
    for (term, cancelled) in std::mem::take(terms).into_iter().zip(cancelled) {
        if !cancelled {
            retained.push(term);
        }
    }
    retained.sort_unstable();
    *terms = retained;
}

fn ordered_cartesian_multiply_signed_spines(
    product: Vec<(Box<[Id]>, bool)>,
    expanded_factor: &[(Box<[Id]>, bool)],
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<Vec<(Box<[Id]>, bool)>> {
    let mut next = Vec::new();
    for (prefix, prefix_negative) in &product {
        for (suffix, suffix_negative) in expanded_factor {
            progress().ok()?;
            next.try_reserve(1).ok()?;
            let length = prefix.len().checked_add(suffix.len())?;
            progress().ok()?;
            let mut combined = Vec::new();
            combined.try_reserve_exact(length).ok()?;
            for factor in prefix {
                progress().ok()?;
                combined.push(*factor);
            }
            for factor in suffix {
                progress().ok()?;
                combined.push(*factor);
            }
            next.push((combined.into_boxed_slice(), *prefix_negative != *suffix_negative));
        }
    }
    Some(next)
}

fn signed_ordered_monomial_spines(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    terms: &[(Id, bool)],
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<Vec<(Box<[Id]>, bool)>> {
    fn expand_leaf(
        egraph: &EGraph<MxxLang, MxxAnalysis>,
        term: Id,
        negative: bool,
        active: &mut HashSet<Id>,
        progress: &mut dyn FnMut() -> Result<(), ()>,
    ) -> Option<Vec<(Box<[Id]>, bool)>> {
        let term = egraph.find(term);
        let factors = mapped_fixed_product_consensus_with_progress(egraph, term, progress)?;
        let Some(factors) = factors else {
            progress().ok()?;
            let mut output = Vec::new();
            output.try_reserve_exact(1).ok()?;
            let mut singleton = Vec::new();
            progress().ok()?;
            singleton.try_reserve_exact(1).ok()?;
            singleton.push(term);
            output.push((singleton.into_boxed_slice(), negative));
            return Some(output);
        };
        if factors.is_empty() || !active.insert(term) {
            return None;
        }
        let result = (|| -> Option<Vec<(Box<[Id]>, bool)>> {
            progress().ok()?;
            let mut product = Vec::new();
            progress().ok()?;
            product.try_reserve_exact(1).ok()?;
            product.push((Box::<[Id]>::default(), false));
            for factor in factors {
                let additive = signed_additive_leaves_with_progress(egraph, &[factor], progress)?;
                let mut expanded_factor = Vec::new();
                for (leaf, leaf_negative) in additive {
                    let expanded = expand_leaf(egraph, leaf, leaf_negative, active, progress)?;
                    for monomial in expanded {
                        progress().ok()?;
                        expanded_factor.try_reserve(1).ok()?;
                        expanded_factor.push(monomial);
                    }
                }
                product =
                    ordered_cartesian_multiply_signed_spines(product, &expanded_factor, progress)?;
            }
            Some(product)
        })();
        active.remove(&term);
        let mut result = result?;
        for (_, sign) in &mut result {
            progress().ok()?;
            *sign = *sign != negative;
        }
        Some(result)
    }

    let mut active = HashSet::new();
    let mut output = Vec::new();
    for (term, outer_negative) in terms {
        let additive = signed_additive_leaves_with_progress(egraph, &[*term], progress)?;
        for (leaf, leaf_negative) in additive {
            let expanded =
                expand_leaf(egraph, leaf, leaf_negative != *outer_negative, &mut active, progress)?;
            for monomial in expanded {
                progress().ok()?;
                output.try_reserve(1).ok()?;
                output.push(monomial);
            }
        }
    }
    Some(output)
}

/// Returns the sole ordered MatrixMultiply leaf sequence for `term`, flattening
/// association without changing factor order.  A class with no physical
/// product is its own singleton leaf; competing product layouts and product
/// cycles are not given a cancellation key.
#[cfg(test)]
fn ordered_product_leaves(egraph: &EGraph<MxxLang, MxxAnalysis>, term: Id) -> Option<Box<[Id]>> {
    fn collect(
        egraph: &EGraph<MxxLang, MxxAnalysis>,
        term: Id,
        active: &mut HashSet<Id>,
        leaves: &mut Vec<Id>,
    ) -> Option<()> {
        let term = egraph.find(term);
        match physical_product_factors(egraph, term) {
            PhysicalStructure::Absent => leaves.push(term),
            PhysicalStructure::Ambiguous => return None,
            PhysicalStructure::Unique(factors) => {
                if !active.insert(term) {
                    return None;
                }
                for factor in factors.iter() {
                    collect(egraph, *factor, active, leaves)?;
                }
                active.remove(&term);
            }
        }
        Some(())
    }

    let mut leaves = Vec::new();
    collect(egraph, term, &mut HashSet::new(), &mut leaves)?;
    Some(leaves.into_boxed_slice())
}

#[cfg(test)]
fn cancelled_signed_additive_leaves(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    terms: &[(Id, bool)],
) -> (Vec<bool>, bool) {
    let mut cancelled = vec![false; terms.len()];
    let mut positive = HashMap::<Box<[Id]>, Vec<usize>>::new();
    let mut negative = HashMap::<Box<[Id]>, Vec<usize>>::new();
    let mut any = false;
    for (index, (base, is_negative)) in terms.iter().enumerate() {
        let Some(key) = ordered_product_leaves(egraph, *base) else { continue };
        let opposite = if *is_negative { &mut positive } else { &mut negative };
        if let Some(other) = opposite.get_mut(&key).and_then(Vec::pop) {
            cancelled[index] = true;
            cancelled[other] = true;
            any = true;
        } else {
            let same = if *is_negative { &mut negative } else { &mut positive };
            same.entry(key).or_default().push(index);
        }
    }
    (cancelled, any)
}

fn equivalent_switch_exists(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    root: Id,
    selector: Id,
    cases: &[Vec<Id>],
) -> bool {
    egraph[egraph.find(root)].nodes.iter().any(|node| match node {
        MxxLang::Switch(existing) if existing.len() == cases.len() + 1 && egraph.find(existing[0]) == selector => {
            existing[1..].iter().zip(cases).all(|(case, terms)| {
                if terms.is_empty() {
                    return egraph[egraph.find(*case)].nodes.iter().any(|node| matches!(node, MxxLang::MatrixConstant(spec)
                        if matches!(egraph.analysis.symbols.matrix_constants.get(spec.0), Some(super::identity::MatrixConstantSpec { value: MatrixConstantValue::Zero, .. }))));
                }
                if terms.len() == 1 {
                    return egraph.find(*case) == egraph.find(terms[0]);
                }
                egraph[egraph.find(*case)].nodes.iter().any(|node| matches!(node, MxxLang::MatrixAdd(existing_terms)
                    if existing_terms.len() == terms.len() && existing_terms.iter().zip(terms).all(|(left, right)| egraph.find(*left) == egraph.find(*right))))
            })
        }
        _ => false,
    })
}

fn equivalent_signed_switch_exists(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    root: Id,
    selector: Id,
    cases: &[Vec<SignedPointwiseTerm>],
) -> bool {
    egraph[egraph.find(root)].nodes.iter().any(|node| match node {
        MxxLang::Switch(existing)
            if existing.len() == cases.len() + 1 && egraph.find(existing[0]) == selector =>
        {
            existing[1..].iter().zip(cases).all(|(case, terms)| match terms.len() {
                0 => egraph[egraph.find(*case)].nodes.iter().any(|node| {
                    matches!(node, MxxLang::MatrixConstant(spec)
                        if matches!(egraph.analysis.symbols.matrix_constants.get(spec.0), Some(super::identity::MatrixConstantSpec { value: MatrixConstantValue::Zero, .. })))
                }),
                1 => signed_additive_child_matches(
                    egraph,
                    *case,
                    (terms[0].base, terms[0].negative),
                ),
                _ => egraph[egraph.find(*case)].nodes.iter().any(|node| {
                    matches!(node, MxxLang::MatrixAdd(existing_terms)
                        if existing_terms.len() == terms.len()
                            && existing_terms.iter().zip(terms).all(|(child, term)|
                                signed_additive_child_matches(
                                    egraph,
                                    *child,
                                    (term.base, term.negative),
                                )))
                }),
            })
        }
        _ => false,
    })
}

fn checked_replacement_plan(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    context: &RewriteContext,
    factors: &[Id],
    relation_position: usize,
) -> Option<CheckedProductReplacementPlan> {
    let relation = egraph.find(factors[relation_position]);
    let actual_public = egraph.find(factors[relation_position - 1]);
    let provenance = &egraph[relation].data.relation_provenance;
    let mut candidates = Vec::new();
    if !flatten_provenance(provenance, context, &mut candidates) {
        return None;
    }
    let mut replacements = BTreeSet::new();
    // Distribution candidates remain purely semantic until all registrations
    // agree on exactly one result.  Unlike ordinary relation rewrites, a
    // distribution candidate can require several e-nodes, so constructing it
    // while probing a later registration would leak a partial rewrite.
    let mut distribution_plans = BTreeSet::new();
    let mut sources = BTreeSet::new();
    let mut failures = BTreeSet::new();
    for candidate in candidates {
        context.note_candidate();
        let source = match candidate {
            RelationCandidate::Direct(source) => source,
            RelationCandidate::Unavailable(source) => {
                failures.insert(RelationFailure::UnavailableRelation { source: source.source });
                continue;
            }
        };
        let registrations = context.registrations(source.source);
        if registrations.is_empty() {
            failures.insert(RelationFailure::MissingRegistration { source: source.source });
            continue;
        }
        for registration in registrations {
            // Section 27 permits distribution only through a physical summand
            // ending in this registration's exact public operand.  Validate that
            // summand before accepting the relation; the enclosing addition is
            // not itself the sampler public key.
            let distributed_public = distribution_public_operand(
                egraph,
                actual_public,
                registration.expected_public,
                context,
            );
            if context.failure().is_some() {
                return None;
            }
            let affine_plan =
                affine_concat_plan(egraph, actual_public, registration.expected_public);
            if distributed_public.is_none() &&
                affine_plan.is_none() &&
                egraph.find(registration.expected_public) != actual_public
            {
                // This registration is not applicable to this product.  It is
                // not a malformed relation: another registration or another
                // product e-node may be the matching use.
                continue;
            }
            let preflight_public = if distributed_public.is_some() || affine_plan.is_some() {
                registration.expected_public
            } else {
                actual_public
            };
            if let Err(failure) =
                preflight_registration(egraph, relation, &source, &registration, preflight_public)
            {
                failures.insert(failure);
                continue;
            }
            if !same_canonical_indices(egraph, &source.indices, &registration.indices) {
                failures.insert(RelationFailure::MismatchedIndex { source: source.source });
                continue;
            }
            let target = egraph.find(registration.target);
            if let Some(plan) = affine_plan {
                let normalized_public =
                    affine_concat_replacement_plan(registration.expected_public, &plan);
                replacements.insert(splice_product_plan(
                    &factors[..relation_position - 1],
                    &[normalized_public, ReplacementPlan::Existing(relation)],
                    &factors[relation_position + 1..],
                ));
                sources.insert(source.source);
                continue;
            }
            let distributed = relation_guided_distribution(
                egraph,
                factors,
                relation_position,
                registration.expected_public,
                target,
                context,
            );
            let Some(distributed) = distributed else { return None };
            // If the left factor is additive, consuming the relation without
            // an exact matching summand would be an unsound general-product
            // rewrite.  Fail closed instead of silently taking that fallback.
            let additive_public = egraph[egraph.find(actual_public)]
                .nodes
                .iter()
                .any(|node| matches!(node, MxxLang::MatrixAdd(_)));
            let replacement = match (additive_public, distributed.plans.is_empty()) {
                (_, false) => {
                    if !context.reserve(1) {
                        return None;
                    }
                    distribution_plans.insert(DistributionPlan {
                        witnesses: distributed.plans.into_iter().collect(),
                    });
                    sources.insert(source.source);
                    continue;
                }
                (true, true) => {
                    failures.insert(RelationFailure::TransformedOperand);
                    continue;
                }
                (false, true) => target_spliced_replacement_plan(
                    egraph,
                    &factors[..relation_position - 1],
                    &[],
                    target,
                    &factors[relation_position + 1..],
                ),
            };
            replacements.insert(replacement);
            sources.insert(source.source);
        }
    }
    if replacements.len() > 1 ||
        (!replacements.is_empty() && !distribution_plans.is_empty()) ||
        distribution_plans.len() > 1
    {
        context
            .fail(RelationFailure::AmbiguousReplacement { sources: sources.into_iter().collect() });
        return None;
    }
    let replacement = if let Some(replacement) = replacements.into_iter().next() {
        replacement
    } else if let Some(plan) = distribution_plans.into_iter().next() {
        distribution_replacement_plan(&plan)
    } else {
        if let Some(failure) = failures.into_iter().next() {
            context.fail(failure);
        }
        return None;
    };
    let selector_distribution =
        switch_node(egraph, actual_public).is_some() || switch_node(egraph, relation).is_some();
    Some(CheckedProductReplacementPlan { replacement, selector_distribution })
}

#[cfg(test)]
fn checked_replacement(
    egraph: &mut EGraph<MxxLang, MxxAnalysis>,
    context: &RewriteContext,
    factors: &[Id],
    relation_position: usize,
) -> Option<(Id, bool)> {
    let plan = checked_replacement_plan(egraph, context, factors, relation_position)?;
    let replacement = materialize_replacement_plan(egraph, context, &plan.replacement)?;
    Some((replacement, plan.selector_distribution))
}

/// Returns whether `root` already has the exact e-node tree described by the
/// checked relation result.  This is deliberately a structural lookup, not a
/// second e-graph or a memo table: the plan is small, ephemeral, and carries
/// only the concrete result the applier would otherwise insert.
fn replacement_plan_satisfied(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    root: Id,
    plan: &ReplacementPlan,
) -> bool {
    fn node_matches(
        egraph: &EGraph<MxxLang, MxxAnalysis>,
        root: Id,
        plan: &ReplacementPlan,
    ) -> bool {
        let root = egraph.find(root);
        match plan {
            ReplacementPlan::Existing(existing) => root == egraph.find(*existing),
            ReplacementPlan::Product(factors) => egraph[root].nodes.iter().any(|node| {
                matches!(node, MxxLang::MatrixMultiply(existing)
                    if existing.len() == factors.len() &&
                        existing.iter().zip(factors).all(|(child, factor)|
                            node_matches(egraph, *child, factor)))
            }),
            ReplacementPlan::Add(terms) => egraph[root].nodes.iter().any(|node| {
                matches!(node, MxxLang::MatrixAdd(existing)
                    if existing.len() == terms.len() &&
                        existing.iter().zip(terms).all(|(child, term)|
                            node_matches(egraph, *child, term)))
            }),
            ReplacementPlan::Negate(term) => egraph[root].nodes.iter().any(|node| {
                matches!(node, MxxLang::MatrixNegate([existing])
                    if node_matches(egraph, *existing, term))
            }),
            ReplacementPlan::Switch(cases) => egraph[root].nodes.iter().any(|node| {
                matches!(node, MxxLang::Switch(existing)
                    if existing.len() == cases.len() &&
                        existing.iter().zip(cases).all(|(child, case)|
                            node_matches(egraph, *child, case)))
            }),
            ReplacementPlan::Concat { axis, inputs } => egraph[root].nodes.iter().any(|node| {
                matches!(node, MxxLang::MatrixConcat { axis: existing_axis, inputs: existing }
                    if existing_axis == axis && existing.len() == inputs.len() &&
                        existing.iter().zip(inputs).all(|(child, input)|
                            node_matches(egraph, *child, input)))
            }),
            ReplacementPlan::Equivalent(plans) => {
                plans.iter().all(|plan| node_matches(egraph, root, plan))
            }
        }
    }

    node_matches(egraph, root, plan)
}

/// Materializes precisely the tree inspected by `replacement_plan_satisfied`.
/// `Equivalent` retains the prior distribution behaviour by unifying every
/// independently witnessed result, while all other variants add one ordinary
/// e-node using canonical child identities.
fn materialize_replacement_plan(
    egraph: &mut EGraph<MxxLang, MxxAnalysis>,
    context: &RewriteContext,
    plan: &ReplacementPlan,
) -> Option<Id> {
    match plan {
        ReplacementPlan::Existing(id) => Some(egraph.find(*id)),
        ReplacementPlan::Product(factors) => {
            let mut materialized = Vec::with_capacity(factors.len());
            for factor in factors {
                context.reserve(1).then_some(())?;
                materialized.push(materialize_replacement_plan(egraph, context, factor)?);
            }
            context.reserve(1).then_some(())?;
            Some(egraph.add(MxxLang::MatrixMultiply(materialized.into_boxed_slice())))
        }
        ReplacementPlan::Add(terms) => {
            let mut materialized = Vec::with_capacity(terms.len());
            for term in terms {
                context.reserve(1).then_some(())?;
                materialized.push(materialize_replacement_plan(egraph, context, term)?);
            }
            context.reserve(1).then_some(())?;
            Some(egraph.add(MxxLang::MatrixAdd(materialized.into_boxed_slice())))
        }
        ReplacementPlan::Negate(term) => {
            let term = materialize_replacement_plan(egraph, context, term)?;
            context.reserve(1).then_some(())?;
            Some(egraph.add(MxxLang::MatrixNegate([term])))
        }
        ReplacementPlan::Switch(cases) => {
            let mut materialized = Vec::with_capacity(cases.len());
            for case in cases {
                context.reserve(1).then_some(())?;
                materialized.push(materialize_replacement_plan(egraph, context, case)?);
            }
            context.reserve(1).then_some(())?;
            Some(egraph.add(MxxLang::Switch(materialized.into_boxed_slice())))
        }
        ReplacementPlan::Concat { axis, inputs } => {
            let mut materialized = Vec::with_capacity(inputs.len());
            for input in inputs {
                context.reserve(1).then_some(())?;
                materialized.push(materialize_replacement_plan(egraph, context, input)?);
            }
            context.reserve(1).then_some(())?;
            Some(egraph.add(MxxLang::MatrixConcat {
                axis: *axis,
                inputs: materialized.into_boxed_slice(),
            }))
        }
        ReplacementPlan::Equivalent(plans) => {
            let mut plans = plans.iter();
            let representative = materialize_replacement_plan(egraph, context, plans.next()?)?;
            for alternative in plans {
                let alternative = materialize_replacement_plan(egraph, context, alternative)?;
                context.reserve(1).then_some(())?;
                egraph.union(representative, alternative);
            }
            Some(representative)
        }
    }
}

fn distribution_public_operand(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    actual_public: Id,
    expected_public: Id,
    context: &RewriteContext,
) -> Option<Id> {
    let expected_public = egraph.find(expected_public);
    egraph[egraph.find(actual_public)]
        .nodes
        .iter()
        .any(|node| {
            if !context.reserve(1) {
                return false;
            }
            let MxxLang::MatrixAdd(children) = node else { return false };
            children.iter().any(|term| {
                if !context.reserve(1) {
                    return false;
                }
                egraph[egraph.find(*term)].nodes.iter().any(|node| {
                    if !context.reserve(1) {
                        return false;
                    }
                    let MxxLang::MatrixMultiply(factors) = node else { return false };
                    factors.last().is_some_and(|last| {
                        context.reserve(1) && egraph.find(*last) == expected_public
                    })
                })
            })
        })
        .then_some(expected_public)
}

/// A local, read-only proof that a column concat is an affine view of one
/// exact public operand.  It is intentionally available only while consuming
/// a registered relation: this is not a general e-graph distribution rule.
#[derive(Clone, Debug)]
struct AffineConcatPlan {
    prefix: Box<[Id]>,
    residuals: Option<Box<[Box<[Id]>]>>,
    outside: Box<[Id]>,
}

/// One locally witnessed decomposition of a concat chunk.  The signal and
/// residual terms remain canonical e-class ids so duplicate physical add
/// nodes do not create spurious ambiguity.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct AffineChunkDecomposition {
    signal: Id,
    prefix: Box<[Id]>,
    end: BigInt,
    residual: Box<[Id]>,
}

fn affine_concat_plan(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    actual_public: Id,
    expected_public: Id,
) -> Option<AffineConcatPlan> {
    let expected_public = egraph.find(expected_public);
    let (concat_id, concat, outside) = affine_concat_operand(egraph, actual_public)?;
    let (Ok(MxxSort::Matrix(actual_type)), Ok(MxxSort::Matrix(concat_type))) =
        (&egraph[egraph.find(actual_public)].data.sort, &egraph[egraph.find(concat_id)].data.sort)
    else {
        return None;
    };
    if !matrix_types_equal(actual_type, concat_type) {
        return None;
    }
    if concat.is_empty() {
        return None;
    }
    affine_concat_plan_for_inputs(egraph, expected_public, concat, outside)
}

fn affine_concat_operand(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    actual_public: Id,
) -> Option<(Id, Box<[Id]>, Box<[Id]>)> {
    if let Some(concat) = unique_concat_columns(egraph, actual_public) {
        return Some((egraph.find(actual_public), concat, Box::new([])));
    }
    let terms = unique_add_terms(egraph, actual_public)?;
    let mut matching = terms.iter().enumerate().filter_map(|(index, term)| {
        unique_concat_columns(egraph, *term).map(|inputs| (index, inputs))
    });
    let (index, concat) = matching.next()?;
    matching.next().is_none().then(|| {
        let outside = terms
            .iter()
            .enumerate()
            .filter_map(|(other, term)| (other != index).then_some(*term))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        (egraph.find(terms[index]), concat, outside)
    })
}

fn affine_concat_plan_for_inputs(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    expected_public: Id,
    concat: Box<[Id]>,
    outside: Box<[Id]>,
) -> Option<AffineConcatPlan> {
    let full_columns = match &egraph[expected_public].data.sort {
        Ok(MxxSort::Matrix(matrix)) => resolved_constant(&matrix.columns)?,
        _ => return None,
    };
    if full_columns <= BigInt::zero() || concat.is_empty() {
        return None;
    }
    let mut next_column = BigInt::zero();
    let mut shared_prefix: Option<Box<[Id]>> = None;
    let mut residuals = Vec::with_capacity(concat.len());
    let mut has_residual = None;
    for chunk in concat.iter() {
        let decompositions = affine_chunk_decompositions(
            egraph,
            *chunk,
            expected_public,
            &next_column,
            &full_columns,
            shared_prefix.as_deref(),
        );
        if decompositions.len() != 1 {
            return None;
        }
        let decomposition = decompositions.into_iter().next().expect("checked singleton");
        if shared_prefix.is_none() {
            shared_prefix = Some(decomposition.prefix.clone());
        }
        let remaining = decomposition.residual;
        // A concat can be entirely signal-only, in which case no zero matrix
        // needs to be invented.  Mixing signal-only and affine chunks would
        // require a typed zero residual for the missing columns, so reject it
        // rather than adding a checker-only zero primitive.
        match (has_residual, remaining.is_empty()) {
            (Some(false), false) | (Some(true), true) => return None,
            (None, empty) => has_residual = Some(!empty),
            _ => {}
        }
        if !remaining.is_empty() {
            residuals.push(remaining);
        }
        next_column = decomposition.end;
    }
    (next_column == full_columns).then(|| AffineConcatPlan {
        prefix: shared_prefix.unwrap_or_default(),
        residuals: has_residual.filter(|has| *has).map(|_| residuals.into_boxed_slice()),
        outside,
    })
}

fn unique_concat_columns(egraph: &EGraph<MxxLang, MxxAnalysis>, id: Id) -> Option<Box<[Id]>> {
    let matches = egraph[egraph.find(id)]
        .nodes
        .iter()
        .filter_map(|node| match node {
            MxxLang::MatrixConcat { axis: Axis::Columns, inputs } => Some(
                inputs
                    .iter()
                    .map(|input| egraph.find(*input))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            ),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    (matches.len() == 1).then(|| matches.into_iter().next().expect("checked singleton"))
}

/// Enumerates only the physical add witnesses of one concat chunk.  Each
/// witness is examined independently; this deliberately never combines
/// alternatives across chunks.
fn affine_chunk_decompositions(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    chunk: Id,
    expected_public: Id,
    next_column: &BigInt,
    full_columns: &BigInt,
    shared_prefix: Option<&[Id]>,
) -> BTreeSet<AffineChunkDecomposition> {
    let chunk = egraph.find(chunk);
    let mut witnesses = BTreeSet::new();
    for node in &egraph[chunk].nodes {
        let MxxLang::MatrixAdd(terms) = node else { continue };
        witnesses.insert(
            terms.iter().map(|term| egraph.find(*term)).collect::<Vec<_>>().into_boxed_slice(),
        );
    }
    if witnesses.is_empty() {
        witnesses.insert(vec![chunk].into_boxed_slice());
    }

    let mut decompositions = BTreeSet::new();
    for terms in witnesses {
        for (signal_index, signal) in terms.iter().copied().enumerate() {
            let Some((prefix, start, end)) = slice_product(egraph, signal, expected_public) else {
                continue;
            };
            if start != *next_column ||
                end <= start ||
                end > *full_columns ||
                shared_prefix
                    .is_some_and(|previous| !same_canonical_indices(egraph, previous, &prefix))
            {
                continue;
            }
            let residual = terms
                .iter()
                .enumerate()
                .filter_map(|(index, term)| (index != signal_index).then_some(*term))
                .collect::<Vec<_>>()
                .into_boxed_slice();
            decompositions.insert(AffineChunkDecomposition { signal, prefix, end, residual });
        }
    }
    decompositions
}

fn unique_product_factors(egraph: &EGraph<MxxLang, MxxAnalysis>, id: Id) -> Option<Box<[Id]>> {
    match physical_product_factors(egraph, id) {
        PhysicalStructure::Unique(factors) => Some(factors),
        PhysicalStructure::Absent | PhysicalStructure::Ambiguous => None,
    }
}

fn slice_product(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    term: Id,
    expected_public: Id,
) -> Option<(Box<[Id]>, BigInt, BigInt)> {
    let mut candidates: BTreeSet<(Box<[Id]>, BigInt, BigInt)> = BTreeSet::new();
    for node in &egraph[egraph.find(term)].nodes {
        let MxxLang::MatrixSlice { spec, input } = node else { continue };
        if egraph.find(input[0]) != egraph.find(expected_public) {
            continue;
        }
        let Some(spec) = egraph.analysis.symbols.slices.get(spec.0) else { continue };
        let Some(columns) = spec.columns.as_ref() else { continue };
        if spec.rows.is_none() {
            if let (Some(start), Some(end)) =
                (resolved_constant(&columns.start), resolved_constant(&columns.end))
            {
                candidates.insert((Vec::new().into_boxed_slice(), start, end));
            }
        }
    }
    for node in &egraph[egraph.find(term)].nodes {
        let MxxLang::MatrixMultiply(factors) = node else { continue };
        let Some(last) = factors.last() else { continue };
        for node in &egraph[egraph.find(*last)].nodes {
            let MxxLang::MatrixSlice { spec, input } = node else { continue };
            if egraph.find(input[0]) != egraph.find(expected_public) {
                continue;
            }
            let Some(spec) = egraph.analysis.symbols.slices.get(spec.0) else { continue };
            let Some(columns) = spec.columns.as_ref() else { continue };
            if spec.rows.is_some() {
                continue;
            }
            let (Some(start), Some(end)) =
                (resolved_constant(&columns.start), resolved_constant(&columns.end))
            else {
                continue;
            };
            let prefix = factors[..factors.len() - 1]
                .iter()
                .map(|factor| egraph.find(*factor))
                .collect::<Vec<_>>()
                .into_boxed_slice();
            candidates.insert((prefix, start, end));
        }
    }
    if candidates.len() != 1 {
        return None;
    }
    candidates.into_iter().next()
}

fn affine_concat_replacement_plan(expected_public: Id, plan: &AffineConcatPlan) -> ReplacementPlan {
    let leading =
        splice_product_plan(&plan.prefix, &[ReplacementPlan::Existing(expected_public)], &[]);
    let residual_chunks = plan
        .residuals
        .as_ref()
        .into_iter()
        .flatten()
        .map(|terms| {
            if terms.len() == 1 {
                ReplacementPlan::Existing(terms[0])
            } else {
                ReplacementPlan::Add(
                    terms
                        .iter()
                        .copied()
                        .map(ReplacementPlan::Existing)
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                )
            }
        })
        .collect::<Vec<_>>();
    let residual = match residual_chunks.len() {
        0 => None,
        1 => residual_chunks.into_iter().next(),
        _ => Some(ReplacementPlan::Concat {
            axis: Axis::Columns,
            inputs: residual_chunks.into_boxed_slice(),
        }),
    };
    let mut terms = Vec::with_capacity(plan.outside.len() + 1 + usize::from(residual.is_some()));
    terms.push(leading);
    if let Some(residual) = residual {
        terms.push(residual);
    }
    terms.extend(plan.outside.iter().copied().map(ReplacementPlan::Existing));
    if terms.len() == 1 {
        terms.into_iter().next().expect("affine leading term")
    } else {
        ReplacementPlan::Add(terms.into_boxed_slice())
    }
}

/// Closed relation-redex classification used by extraction.  Matrix bounds
/// are resolved independently through the authoritative bound input and the
/// shared node transfer; source syntax is never treated as a bound contract.
pub fn classify_proposal_node(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    origin: Id,
    node: &MxxLang,
    context: &RewriteContext,
) -> Result<ProposalRelationClassification, RelationFailure> {
    let MxxLang::MatrixMultiply(factors) = node else {
        return Ok(ProposalRelationClassification::default());
    };
    let origin = egraph.find(origin);
    let Some(nested_candidates) = right_nested_relation_factor_candidates(egraph, factors, context)
    else {
        return match context.failure() {
            Some(failure) => Err(failure),
            None => Ok(ProposalRelationClassification::default()),
        };
    };
    let mut classification = ProposalRelationClassification::default();
    let mut unique_plans = BTreeSet::new();
    let mut scan = |candidate: &[Id]| -> Result<(), RelationFailure> {
        for relation_position in 1..candidate.len() {
            let Some(plan) =
                checked_product_replacement_plan_at(egraph, context, candidate, relation_position)
            else {
                if let Some(failure) = context.failure() {
                    return Err(failure);
                }
                continue;
            };
            let unsatisfied = !replacement_plan_satisfied(egraph, origin, &plan.replacement);
            // A checked plan itself is the canonical boundary identity: it
            // contains the ordered public/relation splice context. Moving it
            // into the set avoids cloning factor sequences while deduplicating
            // direct and immediate right-nested physical exposures.
            classification.relation_redex |= unsatisfied;
            if unique_plans.insert(plan) {
                classification.local_checked_relation_count =
                    classification.local_checked_relation_count.saturating_add(1);
            }
        }
        Ok(())
    };
    scan(factors)?;
    for nested in nested_candidates {
        scan(&nested)?;
    }
    Ok(classification)
}

/// Validates the exact Graph-derived records before accepting an e-class
/// equality.  The registration only points at graph-built e-classes; this
/// function verifies the source role, bare operand, matrix layout, public and
/// trapdoor endpoints at the point a rewrite is attempted.
fn preflight_registration(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    relation: Id,
    source: &RelationSource,
    registration: &RelationRegistration,
    actual_public: Id,
) -> Result<(), RelationFailure> {
    let source_id = egraph.find(relation);
    let source_descriptor = egraph.analysis.symbols.atomic_sources.get(source.source.0);
    let Some(source_descriptor) = source_descriptor else {
        return Err(RelationFailure::InvalidRelationProducer { source: source.source });
    };
    if !matches!(
        source_descriptor.relation_role,
        Some(
            AtomicRelationRole::Preimage |
                AtomicRelationRole::GadgetDecomposition |
                AtomicRelationRole::DecomposedHash |
                AtomicRelationRole::SmallGadgetDecomposition { range_proved: true } |
                AtomicRelationRole::SmallDecomposedHash { range_proved: true }
        )
    ) {
        return Err(RelationFailure::InvalidRelationProducer { source: source.source });
    }
    // A relation is consumed only as its untransformed Atom.  An e-class may
    // contain equal aliases, but one exact atom with its original ordered
    // coordinate children must still be present.
    if !egraph[source_id].nodes.iter().any(|node| matches!(node,
        MxxLang::Atom { source: atom_source, indices }
        if *atom_source == source.source && same_canonical_indices(egraph, indices, &source.indices)
    )) {
        return Err(RelationFailure::TransformedOperand);
    }
    let expected_public = egraph.find(registration.expected_public);
    let target = egraph.find(registration.target);
    let (
        Ok(MxxSort::Matrix(source_matrix)),
        Ok(MxxSort::Matrix(public_matrix)),
        Ok(MxxSort::Matrix(target_matrix)),
    ) = (
        &egraph[source_id].data.sort,
        &egraph[expected_public].data.sort,
        &egraph[target].data.sort,
    )
    else {
        return Err(RelationFailure::MismatchedType { source: source.source });
    };
    if !resolved_equal(&source_matrix.modulus, &public_matrix.modulus) ||
        !resolved_equal(&source_matrix.ring_dimension, &public_matrix.ring_dimension) ||
        !resolved_equal(&source_matrix.modulus, &target_matrix.modulus) ||
        !resolved_equal(&source_matrix.ring_dimension, &target_matrix.ring_dimension)
    {
        return Err(RelationFailure::MismatchedLayout { source: source.source });
    }
    if !resolved_equal(&public_matrix.columns, &source_matrix.rows) ||
        !resolved_equal(&target_matrix.rows, &public_matrix.rows) ||
        !resolved_equal(&target_matrix.columns, &source_matrix.columns)
    {
        return Err(RelationFailure::MismatchedType { source: source.source });
    }
    if expected_public != actual_public {
        return Err(RelationFailure::MismatchedPublic { source: source.source });
    }
    if matches!(
        source_descriptor.relation_role,
        Some(
            AtomicRelationRole::DecomposedHash |
                AtomicRelationRole::SmallDecomposedHash { range_proved: true }
        )
    ) {
        let AtomicSourceKey::Sampler(sampler_id) = source_descriptor.key else {
            return Err(RelationFailure::InvalidRelationProducer { source: source.source });
        };
        let Some(SamplerIdentity::DecomposedHash {
            public,
            target: sampler_target,
            arguments,
            matrix_type,
            base,
            digit_count,
            small,
            ..
        }) = egraph.analysis.symbols.samplers.get(sampler_id.0)
        else {
            return Err(RelationFailure::InvalidRelationProducer { source: source.source });
        };
        if egraph.find(*public) != expected_public ||
            egraph.find(*sampler_target) != target ||
            !matrix_types_equal(matrix_type, source_matrix)
        {
            return Err(RelationFailure::MismatchedTarget { source: source.source });
        }
        let public_is_exact_gadget = egraph[expected_public].nodes.iter().any(|node| {
            let MxxLang::MatrixConstant(spec_id) = node else { return false };
            egraph.analysis.symbols.matrix_constants.get(spec_id.0).is_some_and(|spec| {
                matrix_types_equal(&spec.matrix_type, public_matrix) &&
                    matches!(
                        &spec.value,
                        MatrixConstantValue::Gadget { base: spec_base, small: spec_small }
                        if spec_base == base && spec_small == small
                    )
            })
        });
        if !public_is_exact_gadget {
            return Err(RelationFailure::MismatchedPublic { source: source.source });
        }
        let target_is_exact_hash = egraph[target].nodes.iter().any(|node| {
            let MxxLang::HashPlain { query, arguments: hash_arguments } = node else {
                return false;
            };
            egraph.analysis.symbols.hash_queries.get(query.0).is_some_and(|query| {
                matrix_types_equal(&query.matrix_type, target_matrix) &&
                    same_canonical_indices(egraph, hash_arguments, arguments)
            })
        });
        let layout_is_exact = resolved_constant(&source_matrix.rows)
            .zip(resolved_constant(&public_matrix.rows))
            .zip(resolved_constant(digit_count))
            .is_some_and(|((source_rows, public_rows), digits)| {
                digits > BigInt::zero() && source_rows == public_rows * digits
            });
        if !target_is_exact_hash || !layout_is_exact {
            return Err(RelationFailure::MismatchedTarget { source: source.source });
        }
    }
    if matches!(
        source_descriptor.relation_role,
        Some(
            AtomicRelationRole::GadgetDecomposition |
                AtomicRelationRole::SmallGadgetDecomposition { range_proved: true }
        )
    ) {
        let AtomicSourceKey::Sampler(sampler_id) = source_descriptor.key else {
            return Err(RelationFailure::InvalidRelationProducer { source: source.source });
        };
        let Some(SamplerIdentity::GadgetDecomposition {
            public,
            target: sampler_target,
            base,
            digit_count,
            small,
            ..
        }) = egraph.analysis.symbols.samplers.get(sampler_id.0)
        else {
            return Err(RelationFailure::InvalidRelationProducer { source: source.source });
        };
        let public_is_exact_gadget = egraph[expected_public].nodes.iter().any(|node| {
            let MxxLang::MatrixConstant(spec_id) = node else { return false };
            egraph.analysis.symbols.matrix_constants.get(spec_id.0).is_some_and(|spec| {
                matrix_types_equal(&spec.matrix_type, public_matrix) &&
                    matches!(&spec.value,
                        MatrixConstantValue::Gadget { base: spec_base, small: spec_small }
                        if spec_base == base && spec_small == small)
            })
        });
        let layout_is_exact = resolved_constant(&source_matrix.rows)
            .zip(resolved_constant(&target_matrix.rows))
            .zip(resolved_constant(digit_count))
            .is_some_and(|((source_rows, target_rows), digits)| {
                digits > BigInt::zero() && source_rows == target_rows * digits
            });
        if egraph.find(*public) != expected_public ||
            egraph.find(*sampler_target) != target ||
            !public_is_exact_gadget ||
            !layout_is_exact
        {
            return Err(RelationFailure::MismatchedTarget { source: source.source });
        }
    }
    if !registration.trapdoor.is_none_or(|trapdoor_id| {
        egraph.analysis.symbols.trapdoors.get(trapdoor_id.0).is_some_and(|trapdoor| {
            egraph.find(trapdoor.public) == expected_public &&
                matrix_types_equal(&trapdoor.matrix_type, public_matrix)
        })
    }) {
        return Err(RelationFailure::MismatchedTrapdoor { source: source.source });
    }
    Ok(())
}

/// The restricted distribution rule from Section 27.  It only examines the
/// physical summands of the actual left operand while consuming this exact
/// right-hand relation factor.  It never distributes a general product.
struct RelationGuidedDistribution {
    plans: BTreeSet<DistributionAddPlan>,
}

/// An ephemeral description of one physical Add witness.  It deliberately
/// carries only canonical Id sequences and operation nesting: no e-node is
/// inserted while registrations are still being compared.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct DistributionPlan {
    witnesses: Box<[DistributionAddPlan]>,
}

/// One plan is retained for every physical Add witness.  The enclosing plan
/// keeps these witnesses as union alternatives rather than combining them.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct DistributionAddPlan {
    terms: Box<[DistributionTerm]>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct DistributionTerm {
    primary: DistributionOperation,
    alternatives: Box<[DistributionOperation]>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum DistributionOperation {
    Product(Box<[Id]>),
    Existing(Id),
}

fn distribution_replacement_plan(plan: &DistributionPlan) -> ReplacementPlan {
    let witnesses = plan
        .witnesses
        .iter()
        .map(|witness| {
            ReplacementPlan::Add(
                witness
                    .terms
                    .iter()
                    .map(|term| {
                        let plans = std::iter::once(&term.primary)
                            .chain(term.alternatives.iter())
                            .map(distribution_operation_replacement_plan)
                            .collect::<Vec<_>>();
                        if plans.len() == 1 {
                            plans.into_iter().next().expect("one primary distribution operation")
                        } else {
                            ReplacementPlan::Equivalent(plans.into_boxed_slice())
                        }
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            )
        })
        .collect::<Vec<_>>();
    if witnesses.len() == 1 {
        witnesses.into_iter().next().expect("one distribution witness")
    } else {
        ReplacementPlan::Equivalent(witnesses.into_boxed_slice())
    }
}

fn distribution_operation_replacement_plan(operation: &DistributionOperation) -> ReplacementPlan {
    match operation {
        DistributionOperation::Existing(id) => ReplacementPlan::Existing(*id),
        DistributionOperation::Product(factors) => existing_product_plan(factors),
    }
}

fn relation_guided_distribution(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    factors: &[Id],
    relation_position: usize,
    expected_public: Id,
    target: Id,
    context: &RewriteContext,
) -> Option<RelationGuidedDistribution> {
    let actual_public = factors[relation_position - 1];
    let relation = factors[relation_position];
    let expected_public = egraph.find(expected_public);
    let mut plans = BTreeSet::new();
    for node in &egraph[egraph.find(actual_public)].nodes {
        context.reserve(1).then_some(())?;
        let MxxLang::MatrixAdd(add) = node else { continue };
        context.reserve(1).then_some(())?;
        let mut terms = Vec::with_capacity(add.len());
        let mut consumed = false;
        for matched_term in add.iter() {
            context.reserve(1).then_some(())?;
            let mut alternatives = BTreeSet::new();
            for product in &egraph[egraph.find(*matched_term)].nodes {
                context.reserve(1).then_some(())?;
                let MxxLang::MatrixMultiply(product_factors) = product else { continue };
                let Some(last) = product_factors.last() else { continue };
                context.reserve(1).then_some(())?;
                if egraph.find(*last) != expected_public {
                    continue;
                }
                let operations = distribution_target_operations(
                    egraph,
                    &factors[..relation_position - 1],
                    &product_factors[..product_factors.len() - 1],
                    target,
                    &factors[relation_position + 1..],
                    context,
                )?;
                context.reserve(1).then_some(())?;
                alternatives.insert(operations);
            }
            if let Some(representative) = alternatives.iter().next() {
                consumed = true;
                context.reserve(1).then_some(())?;
                let alternate_sets = alternatives.iter().skip(1).collect::<Vec<_>>();
                for (position, primary) in representative.iter().enumerate() {
                    context.reserve(1).then_some(())?;
                    let mut local_alternatives = Vec::with_capacity(alternate_sets.len());
                    for alternate in &alternate_sets {
                        context.reserve(1).then_some(())?;
                        local_alternatives.push(alternate[position].clone());
                    }
                    context.reserve(1).then_some(())?;
                    terms.push(DistributionTerm {
                        primary: primary.clone(),
                        alternatives: local_alternatives.into_boxed_slice(),
                    });
                }
            } else {
                let primary = distribution_product_operation(
                    egraph,
                    &factors[..relation_position - 1],
                    &[*matched_term, relation],
                    &factors[relation_position + 1..],
                    context,
                )?;
                context.reserve(1).then_some(())?;
                terms.push(DistributionTerm { primary, alternatives: Box::new([]) });
            }
        }
        if consumed {
            context.reserve(1).then_some(())?;
            plans.insert(DistributionAddPlan { terms: terms.into_boxed_slice() });
        }
    }
    Some(RelationGuidedDistribution { plans })
}

fn distribution_product_operation(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    prefix: &[Id],
    middle: &[Id],
    suffix: &[Id],
    context: &RewriteContext,
) -> Option<DistributionOperation> {
    context.reserve(1).then_some(())?;
    let mut factors = Vec::with_capacity(prefix.len() + middle.len() + suffix.len());
    for factor in prefix.iter().chain(middle).chain(suffix) {
        context.reserve(1).then_some(())?;
        factors.push(egraph.find(*factor));
    }
    Some(if factors.len() == 1 {
        DistributionOperation::Existing(factors[0])
    } else {
        DistributionOperation::Product(factors.into_boxed_slice())
    })
}

fn distribution_target_operations(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    prefix: &[Id],
    target_prefix: &[Id],
    target: Id,
    suffix: &[Id],
    context: &RewriteContext,
) -> Option<Box<[DistributionOperation]>> {
    if let Some(terms) = unique_add_terms(egraph, target) {
        if terms.is_empty() {
            return Some(
                vec![DistributionOperation::Existing(egraph.find(target))].into_boxed_slice(),
            );
        }
        context.reserve(1).then_some(())?;
        let mut products = Vec::with_capacity(terms.len());
        for term in terms.iter() {
            let expanded = unique_product_factors(egraph, *term);
            context.reserve(1).then_some(())?;
            let mut middle = Vec::with_capacity(
                target_prefix.len() + expanded.as_ref().map_or(1, |factors| factors.len()),
            );
            for factor in target_prefix {
                context.reserve(1).then_some(())?;
                middle.push(egraph.find(*factor));
            }
            if let Some(factors) = expanded {
                for factor in factors.iter() {
                    context.reserve(1).then_some(())?;
                    middle.push(egraph.find(*factor));
                }
            } else {
                context.reserve(1).then_some(())?;
                middle.push(egraph.find(*term));
            }
            context.reserve(1).then_some(())?;
            products
                .push(distribution_product_operation(egraph, prefix, &middle, suffix, context)?);
        }
        return Some(products.into_boxed_slice());
    }
    context.reserve(1).then_some(())?;
    let mut middle = Vec::with_capacity(target_prefix.len() + 1);
    for factor in target_prefix {
        context.reserve(1).then_some(())?;
        middle.push(egraph.find(*factor));
    }
    context.reserve(1).then_some(())?;
    middle.push(egraph.find(target));
    Some(
        vec![distribution_product_operation(egraph, prefix, &middle, suffix, context)?]
            .into_boxed_slice(),
    )
}

fn target_spliced_replacement_plan(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    prefix: &[Id],
    target_prefix: &[Id],
    target: Id,
    suffix: &[Id],
) -> ReplacementPlan {
    if let Some(terms) = unique_add_terms(egraph, target) {
        if terms.is_empty() {
            return ReplacementPlan::Existing(target);
        }
        return ReplacementPlan::Add(
            terms
                .iter()
                .map(|term| {
                    let mut middle = Vec::with_capacity(
                        target_prefix.len() +
                            unique_product_factors(egraph, *term)
                                .as_ref()
                                .map_or(1, |factors| factors.len()),
                    );
                    middle.extend(target_prefix.iter().copied().map(ReplacementPlan::Existing));
                    if let Some(factors) = unique_product_factors(egraph, *term) {
                        middle.extend(factors.iter().copied().map(ReplacementPlan::Existing));
                    } else {
                        middle.push(ReplacementPlan::Existing(*term));
                    }
                    splice_product_plan(prefix, &middle, suffix)
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
    }
    let mut middle =
        target_prefix.iter().copied().map(ReplacementPlan::Existing).collect::<Vec<_>>();
    middle.push(ReplacementPlan::Existing(target));
    splice_product_plan(prefix, &middle, suffix)
}

fn same_canonical_indices(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    left: &[Id],
    right: &[Id],
) -> bool {
    left.len() == right.len() &&
        left.iter().zip(right).all(|(left, right)| egraph.find(*left) == egraph.find(*right))
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum RelationCandidate {
    Direct(RelationSource),
    Unavailable(RelationSource),
}

fn flatten_provenance(
    values: &[RelationProvenance],
    context: &RewriteContext,
    out: &mut Vec<RelationCandidate>,
) -> bool {
    let completed = try_visit_relation_provenance(
        values,
        || context.reserve(1),
        |visit| match visit {
            RelationProvenanceVisit::Direct(source) => {
                out.push(RelationCandidate::Direct(source.clone()))
            }
            RelationProvenanceVisit::Unavailable { source, .. } => {
                out.push(RelationCandidate::Unavailable(source.clone()));
            }
            RelationProvenanceVisit::Switch { .. } => {}
        },
    );
    out.sort();
    out.dedup();
    completed
}

/// Builds a pointwise selector product.  A selector may be present on one
/// operand only; the fixed operand is then used in every branch.  Two
/// selectors must still be identical, so this never creates a Cartesian
/// product of branch families.
pub fn pointwise_same_selector(
    egraph: &mut EGraph<MxxLang, MxxAnalysis>,
    left: Id,
    right: Id,
    multiply: bool,
) -> Result<Option<Id>, RelationFailure> {
    let left = egraph.find(left);
    let right = egraph.find(right);
    if !pointwise_selector_is_distributable(egraph, left, right)? {
        return Ok(None);
    }
    let left_switch = switch_node(egraph, left);
    let right_switch = switch_node(egraph, right);
    match (left_switch, right_switch) {
        (Some(left_cases), Some(right_cases)) => {
            let mut cases = Vec::with_capacity(left_cases.len());
            cases.push(left_cases[0]);
            for (left, right) in left_cases[1..].iter().zip(&right_cases[1..]) {
                cases.push(egraph.add(if multiply {
                    MxxLang::MatrixMultiply(vec![*left, *right].into_boxed_slice())
                } else {
                    MxxLang::MatrixAdd(vec![*left, *right].into_boxed_slice())
                }));
            }
            Ok(Some(egraph.add(MxxLang::Switch(cases.into_boxed_slice()))))
        }
        (Some(cases), None) => {
            Ok(pointwise_switch_with_fixed_operand(egraph, cases, right, true, multiply))
        }
        (None, Some(cases)) => {
            Ok(pointwise_switch_with_fixed_operand(egraph, cases, left, false, multiply))
        }
        (None, None) => unreachable!("a distributable selector has a switch"),
    }
}

/// Read-only counterpart of `pointwise_same_selector`.  It captures the exact
/// Switch/Product tree before an e-node is inserted, so extraction can charge
/// only an equality that has not already been recorded in the e-graph.
fn pointwise_same_selector_plan(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    left: Id,
    right: Id,
    multiply: bool,
) -> Result<Option<ReplacementPlan>, RelationFailure> {
    let left = egraph.find(left);
    let right = egraph.find(right);
    if !pointwise_selector_is_distributable(egraph, left, right)? {
        return Ok(None);
    }
    let left_switch = switch_node(egraph, left);
    let right_switch = switch_node(egraph, right);
    let combine = |left: Id, right: Id| {
        if multiply {
            ReplacementPlan::Product(
                vec![ReplacementPlan::Existing(left), ReplacementPlan::Existing(right)].into(),
            )
        } else {
            ReplacementPlan::Add(
                vec![ReplacementPlan::Existing(left), ReplacementPlan::Existing(right)].into(),
            )
        }
    };
    match (left_switch, right_switch) {
        (Some(left_cases), Some(right_cases)) => {
            let mut cases = Vec::with_capacity(left_cases.len());
            cases.push(ReplacementPlan::Existing(left_cases[0]));
            for (left, right) in left_cases[1..].iter().zip(&right_cases[1..]) {
                cases.push(combine(*left, *right));
            }
            Ok(Some(ReplacementPlan::Switch(cases.into_boxed_slice())))
        }
        (Some(cases), None) => {
            let mut output_cases = Vec::with_capacity(cases.len());
            output_cases.push(ReplacementPlan::Existing(cases[0]));
            output_cases.extend(cases[1..].iter().map(|case| combine(*case, right)));
            Ok(Some(ReplacementPlan::Switch(output_cases.into_boxed_slice())))
        }
        (None, Some(cases)) => {
            let mut output_cases = Vec::with_capacity(cases.len());
            output_cases.push(ReplacementPlan::Existing(cases[0]));
            output_cases.extend(cases[1..].iter().map(|case| combine(left, *case)));
            Ok(Some(ReplacementPlan::Switch(output_cases.into_boxed_slice())))
        }
        (None, None) => unreachable!("a distributable selector has a switch"),
    }
}

/// Checks the selector shape shared by pointwise construction and extraction.
/// It reads only canonical selector identities and case counts; it neither
/// clones cases nor visits alternatives.
fn pointwise_selector_is_distributable(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    left: Id,
    right: Id,
) -> Result<bool, RelationFailure> {
    let left = switch_shape(egraph, left);
    let right = switch_shape(egraph, right);
    match (left, right) {
        (Some((left_selector, left_count)), Some((right_selector, right_count))) => {
            if left_selector != right_selector {
                Err(RelationFailure::DifferentSelectorBlocked)
            } else {
                Ok(left_count == right_count && left_count >= 2)
            }
        }
        (Some((_, count)), None) | (None, Some((_, count))) => Ok(count >= 2),
        (None, None) => Ok(false),
    }
}

fn switch_shape(egraph: &EGraph<MxxLang, MxxAnalysis>, id: Id) -> Option<(Id, usize)> {
    egraph[egraph.find(id)].nodes.iter().find_map(|node| match node {
        MxxLang::Switch(cases) => {
            cases.first().map(|selector| (egraph.find(*selector), cases.len()))
        }
        _ => None,
    })
}

fn pointwise_switch_with_fixed_operand(
    egraph: &mut EGraph<MxxLang, MxxAnalysis>,
    cases: Box<[Id]>,
    fixed: Id,
    switch_is_left: bool,
    multiply: bool,
) -> Option<Id> {
    if cases.len() < 2 {
        return None;
    }
    let mut output_cases = Vec::with_capacity(cases.len());
    output_cases.push(cases[0]);
    for case in &cases[1..] {
        let (left, right) = if switch_is_left { (*case, fixed) } else { (fixed, *case) };
        output_cases.push(egraph.add(if multiply {
            MxxLang::MatrixMultiply(vec![left, right].into_boxed_slice())
        } else {
            MxxLang::MatrixAdd(vec![left, right].into_boxed_slice())
        }));
    }
    Some(egraph.add(MxxLang::Switch(output_cases.into_boxed_slice())))
}

fn switch_node(egraph: &EGraph<MxxLang, MxxAnalysis>, id: Id) -> Option<Box<[Id]>> {
    egraph[egraph.find(id)].nodes.iter().find_map(|node| match node {
        MxxLang::Switch(cases) => Some(cases.clone()),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::identity::{
        AtomicSourceDescriptor, AtomicSourceKey, BinderDescriptor, BinderKey,
        CanonicalResidueConvention, GraphWireSourceKey, HashQuerySpec, HashQuerySpecId,
        IntegerSourceDomain, MatrixConstantSpec, MatrixConstantSpecId, OccurrenceScope, ProgramKey,
        ResolvedIndexRange, ResolvedIntExpr, ResolvedMatrixType, SamplerDescriptorId, SliceSpec,
        SliceSpecId, WireSourceKey,
    };
    use std::{
        collections::BTreeMap,
        fmt,
        sync::{Arc, Mutex},
    };
    use tracing::{
        Event, Subscriber,
        field::{Field, Visit},
        level_filters::LevelFilter,
    };
    use tracing_subscriber::{Layer, layer::Context, prelude::*, registry::LookupSpan};

    #[derive(Default)]
    struct EventFields {
        fields: BTreeMap<String, String>,
    }

    impl Visit for EventFields {
        fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
            self.fields.insert(field.name().to_owned(), format!("{value:?}"));
        }
    }

    #[derive(Clone, Default)]
    struct BinderFailureEventCapture(Arc<Mutex<Vec<BTreeMap<String, String>>>>);

    impl<S> Layer<S> for BinderFailureEventCapture
    where
        S: Subscriber + for<'lookup> LookupSpan<'lookup>,
    {
        fn max_level_hint(&self) -> Option<LevelFilter> {
            Some(LevelFilter::DEBUG)
        }

        fn on_event(&self, event: &Event<'_>, _: Context<'_, S>) {
            let mut fields = EventFields::default();
            event.record(&mut fields);
            if fields
                .fields
                .get("event")
                .is_some_and(|event| event.contains("pointwise_binder_build_failure"))
            {
                self.0.lock().expect("event capture lock").push(fields.fields);
            }
        }
    }

    #[derive(Clone, Default)]
    struct FixedPeelEventCapture(Arc<Mutex<Vec<BTreeMap<String, String>>>>);

    impl<S> Layer<S> for FixedPeelEventCapture
    where
        S: Subscriber + for<'lookup> LookupSpan<'lookup>,
    {
        fn max_level_hint(&self) -> Option<LevelFilter> {
            Some(LevelFilter::DEBUG)
        }

        fn on_event(&self, event: &Event<'_>, _: Context<'_, S>) {
            let mut fields = EventFields::default();
            event.record(&mut fields);
            if fields.fields.get("event").is_some_and(|event| {
                event.contains("fixed_target_peel_match") ||
                    event.contains("fixed_target_peel_term") ||
                    event.contains("binder_pre_cancel_mapped_fixed_spine") ||
                    event.contains("operational_selected_full_case_residual_mismatch") ||
                    event.contains("operational_selected_full_case_candidate_rejected")
            }) {
                self.0.lock().expect("event capture lock").push(fields.fields);
            }
        }
    }

    fn scalar_matrix_type() -> ResolvedMatrixType {
        ResolvedMatrixType {
            modulus: ResolvedIntExpr::Const(17.into()),
            ring_dimension: ResolvedIntExpr::Const(1.into()),
            rows: ResolvedIntExpr::Const(1.into()),
            columns: ResolvedIntExpr::Const(1.into()),
        }
    }

    fn concrete_matrix_type(
        modulus: i64,
        ring_dimension: i64,
        rows: i64,
        columns: i64,
    ) -> ResolvedMatrixType {
        ResolvedMatrixType {
            modulus: ResolvedIntExpr::Const(modulus.into()),
            ring_dimension: ResolvedIntExpr::Const(ring_dimension.into()),
            rows: ResolvedIntExpr::Const(rows.into()),
            columns: ResolvedIntExpr::Const(columns.into()),
        }
    }

    fn matrix_atom(
        egraph: &mut EGraph<MxxLang, MxxAnalysis>,
        name: &str,
        relation_role: Option<AtomicRelationRole>,
    ) -> (Id, AtomicSourceId) {
        matrix_atom_with_type(egraph, name, scalar_matrix_type(), relation_role)
    }

    fn matrix_atom_with_type(
        egraph: &mut EGraph<MxxLang, MxxAnalysis>,
        name: &str,
        matrix_type: ResolvedMatrixType,
        relation_role: Option<AtomicRelationRole>,
    ) -> (Id, AtomicSourceId) {
        let source = egraph.analysis.symbols.atomic_sources.intern(AtomicSourceDescriptor {
            key: AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from(name)),
            sort: MxxSort::Matrix(matrix_type),
            integer_domain: None,
            canonical_residue_convention: Some(CanonicalResidueConvention::Nonnegative),
            relation_role,
        });
        let source = AtomicSourceId(source);
        let term = egraph.add(MxxLang::Atom { source, indices: Box::new([]) });
        (term, source)
    }

    fn integer_atom(egraph: &mut EGraph<MxxLang, MxxAnalysis>, name: &str) -> Id {
        let source = egraph.analysis.symbols.atomic_sources.intern(AtomicSourceDescriptor {
            key: AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from(name)),
            sort: MxxSort::Int,
            integer_domain: Some(IntegerSourceDomain { minimum: 0.into(), maximum: 1.into() }),
            canonical_residue_convention: None,
            relation_role: None,
        });
        egraph.add(MxxLang::Atom { source: AtomicSourceId(source), indices: Box::new([]) })
    }

    fn scalar_polynomial_constant(
        egraph: &mut EGraph<MxxLang, MxxAnalysis>,
        coefficients: &[i64],
    ) -> Id {
        let spec = egraph.analysis.symbols.matrix_constants.intern(
            super::super::identity::MatrixConstantSpec {
                matrix_type: scalar_matrix_type(),
                value: MatrixConstantValue::Polynomial {
                    coefficients: coefficients
                        .iter()
                        .map(|coefficient| ResolvedIntExpr::Const((*coefficient).into()))
                        .collect(),
                },
            },
        );
        egraph.add(MxxLang::MatrixConstant(super::super::identity::MatrixConstantSpecId(spec)))
    }

    fn regular_scalar_gadget(egraph: &mut EGraph<MxxLang, MxxAnalysis>) -> Id {
        let spec = egraph.analysis.symbols.matrix_constants.intern(
            super::super::identity::MatrixConstantSpec {
                matrix_type: scalar_matrix_type(),
                value: MatrixConstantValue::Gadget {
                    base: ResolvedIntExpr::Const(2.into()),
                    small: false,
                },
            },
        );
        egraph.add(MxxLang::MatrixConstant(super::super::identity::MatrixConstantSpecId(spec)))
    }

    fn binder_matrix_atom(egraph: &mut EGraph<MxxLang, MxxAnalysis>, binder: Id, name: &str) -> Id {
        indexed_matrix_atom(egraph, &[binder], name)
    }

    fn indexed_matrix_atom(
        egraph: &mut EGraph<MxxLang, MxxAnalysis>,
        indices: &[Id],
        name: &str,
    ) -> Id {
        let source = egraph.analysis.symbols.atomic_sources.intern(AtomicSourceDescriptor {
            key: AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from(name)),
            sort: MxxSort::Matrix(scalar_matrix_type()),
            integer_domain: None,
            canonical_residue_convention: Some(CanonicalResidueConvention::Nonnegative),
            relation_role: None,
        });
        egraph.add(MxxLang::Atom {
            source: AtomicSourceId(source),
            indices: indices.to_vec().into_boxed_slice(),
        })
    }

    fn symbolic_two_case_root(
        egraph: &mut EGraph<MxxLang, MxxAnalysis>,
        selector: Id,
        shared: Id,
        first_case: Id,
        second_case: Id,
        fixed_multiplicity: usize,
    ) -> Id {
        let switch =
            egraph.add(MxxLang::Switch(vec![selector, first_case, second_case].into_boxed_slice()));
        let fixed = egraph.add(MxxLang::MatrixNegate([shared]));
        let mut terms = vec![switch];
        terms.extend(std::iter::repeat_n(fixed, fixed_multiplicity));
        egraph.add(MxxLang::MatrixAdd(terms.into_boxed_slice()))
    }

    fn only_binder_plan(
        egraph: &EGraph<MxxLang, MxxAnalysis>,
        root: Id,
    ) -> (Id, BinderAwarePointwiseAddSwitchPlan) {
        let plan = pointwise_add_switch_cancellation_plan(egraph, root)
            .expect("binder-aware pointwise plan");
        let selector = plan.selector;
        let mut plans = plan.binder_aware.expect("one binder-aware candidate").into_vec();
        assert_eq!(plans.len(), 1);
        (selector, plans.pop().expect("one binder-aware plan"))
    }

    fn test_binder(
        egraph: &mut EGraph<MxxLang, MxxAnalysis>,
        minimum: i64,
        maximum: i64,
    ) -> BinderId {
        test_binder_at(egraph, minimum, maximum, 1)
    }

    fn test_binder_at(
        egraph: &mut EGraph<MxxLang, MxxAnalysis>,
        minimum: i64,
        maximum: i64,
        node: u64,
    ) -> BinderId {
        BinderId(egraph.analysis.symbols.binders.intern(BinderDescriptor {
            key: BinderKey {
                loop_scope: OccurrenceScope {
                    program: ProgramKey::Ideal,
                    definition: mxx_ir_core::FrozenGraphScopeId::Root,
                    path: Box::new([]),
                },
                loop_node: mxx_ir_core::NodeId(node),
                slot: 0,
            },
            minimum: minimum.into(),
            maximum: maximum.into(),
        }))
    }

    #[test]
    fn affine_concat_plan_preserves_wrapper_and_all_residual_terms() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let matrix = |rows: i32, columns: i32| ResolvedMatrixType {
            modulus: ResolvedIntExpr::Const(17.into()),
            ring_dimension: ResolvedIntExpr::Const(1.into()),
            rows: ResolvedIntExpr::Const(rows.into()),
            columns: ResolvedIntExpr::Const(columns.into()),
        };
        let (expected, _) = matrix_atom_with_type(&mut egraph, "expected", matrix(2, 4), None);
        let (prefix, _) = matrix_atom_with_type(&mut egraph, "prefix", matrix(1, 2), None);
        let (error0, _) = matrix_atom_with_type(&mut egraph, "error0", matrix(1, 2), None);
        let (error1, _) = matrix_atom_with_type(&mut egraph, "error1", matrix(1, 2), None);
        let (extra0, _) = matrix_atom_with_type(&mut egraph, "extra0", matrix(1, 2), None);
        let (extra1, _) = matrix_atom_with_type(&mut egraph, "extra1", matrix(1, 2), None);
        let (outside, _) = matrix_atom_with_type(&mut egraph, "outside", matrix(1, 4), None);
        let slice = |egraph: &mut EGraph<MxxLang, MxxAnalysis>, start: i32, end: i32| {
            let spec = SliceSpecId(egraph.analysis.symbols.slices.intern(SliceSpec {
                rows: None,
                columns: Some(ResolvedIndexRange {
                    start: ResolvedIntExpr::Const(start.into()),
                    end: ResolvedIntExpr::Const(end.into()),
                }),
            }));
            egraph.add(MxxLang::MatrixSlice { spec, input: [expected] })
        };
        let slice0 = slice(&mut egraph, 0, 2);
        let signal0 = egraph.add(MxxLang::MatrixMultiply(vec![prefix, slice0].into()));
        let chunk0 = egraph.add(MxxLang::MatrixAdd(vec![signal0, error0, extra0].into()));
        let slice1 = slice(&mut egraph, 2, 4);
        let signal1 = egraph.add(MxxLang::MatrixMultiply(vec![prefix, slice1].into()));
        let chunk1 = egraph.add(MxxLang::MatrixAdd(vec![signal1, error1, extra1].into()));
        let concat = egraph.add(MxxLang::MatrixConcat {
            axis: Axis::Columns,
            inputs: vec![chunk0, chunk1].into(),
        });
        let (other_public, _) =
            matrix_atom_with_type(&mut egraph, "other-public", matrix(2, 4), None);
        let (other_prefix, _) =
            matrix_atom_with_type(&mut egraph, "other-prefix", matrix(1, 2), None);
        let mismatched_signal =
            egraph.add(MxxLang::MatrixMultiply(vec![other_prefix, slice1].into()));
        let mismatched_chunk =
            egraph.add(MxxLang::MatrixAdd(vec![mismatched_signal, error1].into()));
        let prefix_mismatch = egraph.add(MxxLang::MatrixConcat {
            axis: Axis::Columns,
            inputs: vec![chunk0, mismatched_chunk].into(),
        });
        let invalid_witness = egraph.add(MxxLang::MatrixAdd(vec![other_prefix, error0].into()));
        egraph.union(chunk0, invalid_witness);
        let wrapper = egraph.add(MxxLang::MatrixAdd(vec![concat, outside].into()));
        egraph.rebuild();

        let plan = affine_concat_plan(&egraph, wrapper, expected).expect("exact column partition");
        assert!(affine_concat_plan(&egraph, concat, other_public).is_none());
        assert!(affine_concat_plan(&egraph, prefix_mismatch, expected).is_none());
        assert_eq!(plan.prefix.len(), 1);
        assert_eq!(egraph.find(plan.prefix[0]), egraph.find(prefix));
        let residuals = plan.residuals.as_ref().expect("every chunk has residual terms");
        assert_eq!(residuals.len(), 2);
        assert!(residuals.iter().all(|terms| terms.len() == 2));
        assert_eq!(plan.outside.len(), 1);
        assert_eq!(egraph.find(plan.outside[0]), egraph.find(outside));

        let (relation, source) = matrix_atom_with_type(
            &mut egraph,
            "relation",
            matrix(4, 1),
            Some(AtomicRelationRole::Preimage),
        );
        let (target, _) = matrix_atom_with_type(&mut egraph, "target", matrix(2, 1), None);
        let context = RewriteContext::new(SharedRewriteBudget::new());
        context.register(registration(source, expected, target));
        let root = egraph.add(MxxLang::MatrixMultiply(vec![wrapper, relation].into()));
        let rewrite = egg::Rewrite::new(
            "test-affine-concat-relation",
            RelationSearcher::new(context.clone()),
            RelationApplier::new(context.clone()),
        )
        .expect("closed relation rewrite");
        let egraph = egg::Runner::default().with_egraph(egraph).run(&[rewrite]).egraph;
        assert!(
            egraph[egraph.find(root)]
                .nodes
                .iter()
                .any(|node| { matches!(node, MxxLang::MatrixAdd(terms) if terms.len() >= 2) })
        );
        assert!(context.counters().rewrites >= 2);
        assert_eq!(context.failure(), None);
    }

    #[test]
    fn affine_concat_plan_rejects_distinct_valid_chunk_witnesses() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let matrix = |rows: i32, columns: i32| ResolvedMatrixType {
            modulus: ResolvedIntExpr::Const(17.into()),
            ring_dimension: ResolvedIntExpr::Const(1.into()),
            rows: ResolvedIntExpr::Const(rows.into()),
            columns: ResolvedIntExpr::Const(columns.into()),
        };
        let (expected, _) = matrix_atom_with_type(&mut egraph, "expected", matrix(1, 1), None);
        let (prefix, _) = matrix_atom_with_type(&mut egraph, "prefix", matrix(1, 1), None);
        let (error0, _) = matrix_atom_with_type(&mut egraph, "error0", matrix(1, 1), None);
        let (error1, _) = matrix_atom_with_type(&mut egraph, "error1", matrix(1, 1), None);
        let spec = SliceSpecId(egraph.analysis.symbols.slices.intern(SliceSpec {
            rows: None,
            columns: Some(ResolvedIndexRange {
                start: ResolvedIntExpr::Const(0.into()),
                end: ResolvedIntExpr::Const(1.into()),
            }),
        }));
        let slice = egraph.add(MxxLang::MatrixSlice { spec, input: [expected] });
        let prefixed_signal = egraph.add(MxxLang::MatrixMultiply(vec![prefix, slice].into()));
        let first_witness = egraph.add(MxxLang::MatrixAdd(vec![prefixed_signal, error0].into()));
        let second_witness = egraph.add(MxxLang::MatrixAdd(vec![slice, error1].into()));
        egraph.union(first_witness, second_witness);
        let actual = egraph
            .add(MxxLang::MatrixConcat { axis: Axis::Columns, inputs: vec![first_witness].into() });
        egraph.rebuild();

        assert!(affine_concat_plan(&egraph, actual, expected).is_none());
    }

    #[test]
    fn affine_concat_plan_rejects_a_column_gap() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let matrix = |rows: i32, columns: i32| ResolvedMatrixType {
            modulus: ResolvedIntExpr::Const(17.into()),
            ring_dimension: ResolvedIntExpr::Const(1.into()),
            rows: ResolvedIntExpr::Const(rows.into()),
            columns: ResolvedIntExpr::Const(columns.into()),
        };
        let (expected, _) = matrix_atom_with_type(&mut egraph, "expected", matrix(1, 4), None);
        let slice = |egraph: &mut EGraph<MxxLang, MxxAnalysis>, start: i32, end: i32| {
            let spec = SliceSpecId(egraph.analysis.symbols.slices.intern(SliceSpec {
                rows: None,
                columns: Some(ResolvedIndexRange {
                    start: ResolvedIntExpr::Const(start.into()),
                    end: ResolvedIntExpr::Const(end.into()),
                }),
            }));
            egraph.add(MxxLang::MatrixSlice { spec, input: [expected] })
        };
        let (error0, _) = matrix_atom_with_type(&mut egraph, "error0", matrix(1, 2), None);
        let (error1, _) = matrix_atom_with_type(&mut egraph, "error1", matrix(1, 1), None);
        let slice0 = slice(&mut egraph, 0, 2);
        let chunk0 = egraph.add(MxxLang::MatrixAdd(vec![slice0, error0].into()));
        let slice1 = slice(&mut egraph, 3, 4);
        let chunk1 = egraph.add(MxxLang::MatrixAdd(vec![slice1, error1].into()));
        let actual = egraph.add(MxxLang::MatrixConcat {
            axis: Axis::Columns,
            inputs: vec![chunk0, chunk1].into(),
        });
        let wrong_axis = egraph
            .add(MxxLang::MatrixConcat { axis: Axis::Rows, inputs: vec![chunk0, chunk1].into() });
        egraph.rebuild();
        assert!(affine_concat_plan(&egraph, actual, expected).is_none());
        assert!(affine_concat_plan(&egraph, wrong_axis, expected).is_none());
    }

    #[test]
    fn affine_concat_plan_accepts_direct_pure_signal_chunk() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (expected, _) = matrix_atom(&mut egraph, "expected", None);
        let spec = SliceSpecId(egraph.analysis.symbols.slices.intern(SliceSpec {
            rows: None,
            columns: Some(ResolvedIndexRange {
                start: ResolvedIntExpr::Const(0.into()),
                end: ResolvedIntExpr::Const(1.into()),
            }),
        }));
        let chunk = egraph.add(MxxLang::MatrixSlice { spec, input: [expected] });
        let actual =
            egraph.add(MxxLang::MatrixConcat { axis: Axis::Columns, inputs: vec![chunk].into() });
        egraph.rebuild();
        let plan = affine_concat_plan(&egraph, actual, expected).expect("complete pure partition");
        assert!(plan.prefix.is_empty());
        assert!(plan.residuals.is_none());
        assert!(plan.outside.is_empty());
    }

    #[test]
    fn source_kinds_do_not_affect_relation_redex_classification() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let protocol_atom = |egraph: &mut EGraph<MxxLang, MxxAnalysis>, name: &str, sort| {
            let source = AtomicSourceId(egraph.analysis.symbols.atomic_sources.intern(
                AtomicSourceDescriptor {
                    key: AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from(name)),
                    sort,
                    integer_domain: None,
                    canonical_residue_convention: None,
                    relation_role: None,
                },
            ));
            egraph.add(MxxLang::Atom { source, indices: Box::new([]) })
        };
        let bytes = protocol_atom(
            &mut egraph,
            "hash-key",
            MxxSort::Bytes(ResolvedIntExpr::Const(32.into())),
        );
        let integer = protocol_atom(&mut egraph, "counter", MxxSort::Int);
        let (matrix, _) = matrix_atom(&mut egraph, "plaintext", None);
        let sampler_source =
            AtomicSourceId(egraph.analysis.symbols.atomic_sources.intern(AtomicSourceDescriptor {
                key: AtomicSourceKey::Sampler(SamplerDescriptorId(0)),
                sort: MxxSort::Matrix(scalar_matrix_type()),
                integer_domain: None,
                canonical_residue_convention: None,
                relation_role: None,
            }));
        let sampler = egraph.add(MxxLang::Atom { source: sampler_source, indices: Box::new([]) });
        let context = RewriteContext::new(SharedRewriteBudget::new());

        for term in [bytes, integer, sampler] {
            let node = egraph[egraph.find(term)].nodes.first().expect("atom node");
            assert!(!classify_proposal_node(&egraph, term, node, &context).unwrap().relation_redex);
        }
        let node = egraph[egraph.find(matrix)].nodes.first().expect("matrix atom node");
        assert!(!classify_proposal_node(&egraph, matrix, node, &context).unwrap().relation_redex);
    }

    fn registration(
        source: AtomicSourceId,
        expected_public: Id,
        target: Id,
    ) -> RelationRegistration {
        RelationRegistration {
            source,
            expected_public,
            target,
            trapdoor: None,
            indices: Box::new([]),
        }
    }

    #[test]
    fn pointwise_selector_product_distributes_a_single_switch_linearly() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let left_cases =
            [egraph.add(MxxLang::IntConst(2.into())), egraph.add(MxxLang::IntConst(3.into()))];
        let left = egraph
            .add(MxxLang::Switch(vec![selector, left_cases[0], left_cases[1]].into_boxed_slice()));
        let right = egraph.add(MxxLang::IntConst(5.into()));

        let product = pointwise_same_selector(&mut egraph, left, right, true)
            .expect("one-sided selection is valid")
            .expect("the selection is distributed");
        let cases = switch_node(&egraph, product).expect("selector is retained");
        assert_eq!(egraph.find(cases[0]), egraph.find(selector));
        assert_eq!(cases.len(), 3);
        for (case, left_case) in cases[1..].iter().zip(left_cases) {
            assert!(egraph[egraph.find(*case)].nodes.iter().any(|node| {
                matches!(node, MxxLang::MatrixMultiply(factors)
                    if factors.len() == 2 &&
                        egraph.find(factors[0]) == egraph.find(left_case) &&
                        egraph.find(factors[1]) == egraph.find(right))
            }));
        }

        let right_cases =
            [egraph.add(MxxLang::IntConst(7.into())), egraph.add(MxxLang::IntConst(11.into()))];
        let selected_right = egraph.add(MxxLang::Switch(
            vec![selector, right_cases[0], right_cases[1]].into_boxed_slice(),
        ));
        let product = pointwise_same_selector(&mut egraph, right, selected_right, true)
            .expect("one-sided selection is valid")
            .expect("the selection is distributed");
        let cases = switch_node(&egraph, product).expect("selector is retained");
        for (case, right_case) in cases[1..].iter().zip(right_cases) {
            assert!(egraph[egraph.find(*case)].nodes.iter().any(|node| {
                matches!(node, MxxLang::MatrixMultiply(factors)
                    if factors.len() == 2 &&
                        egraph.find(factors[0]) == egraph.find(right) &&
                        egraph.find(factors[1]) == egraph.find(right_case))
            }));
        }
    }

    #[test]
    fn classifier_skips_different_selector_relation_position_without_failure() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let left_selector = egraph.add(MxxLang::IntConst(0.into()));
        let right_selector = egraph.add(MxxLang::IntConst(1.into()));
        let (public_case, _) = matrix_atom(&mut egraph, "public-case", None);
        let (relation_case, _) =
            matrix_atom(&mut egraph, "relation-case", Some(AtomicRelationRole::Preimage));
        let public =
            egraph.add(MxxLang::Switch(vec![left_selector, public_case].into_boxed_slice()));
        let relation =
            egraph.add(MxxLang::Switch(vec![right_selector, relation_case].into_boxed_slice()));
        let product =
            egraph.add(MxxLang::MatrixMultiply(vec![public, relation].into_boxed_slice()));
        egraph.rebuild();
        let context = RewriteContext::new(SharedRewriteBudget::new());
        let node = egraph[egraph.find(product)]
            .nodes
            .iter()
            .find(|node| matches!(node, MxxLang::MatrixMultiply(_)))
            .expect("product node");
        assert!(
            !classify_proposal_node(&egraph, product, node, &context)
                .expect("selector mismatch is local")
                .relation_redex
        );
        assert_eq!(context.failure(), None);
    }

    #[test]
    fn classifier_marks_same_selector_relation_product_for_distribution() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let (public_case, _) = matrix_atom(&mut egraph, "same-public", None);
        let (relation_case, _) =
            matrix_atom(&mut egraph, "same-relation", Some(AtomicRelationRole::Preimage));
        let public = egraph.add(MxxLang::Switch(vec![selector, public_case].into_boxed_slice()));
        let relation =
            egraph.add(MxxLang::Switch(vec![selector, relation_case].into_boxed_slice()));
        let product =
            egraph.add(MxxLang::MatrixMultiply(vec![public, relation].into_boxed_slice()));
        egraph.rebuild();
        let context = RewriteContext::new(SharedRewriteBudget::new());
        let node = egraph[egraph.find(product)]
            .nodes
            .iter()
            .find(|node| matches!(node, MxxLang::MatrixMultiply(_)))
            .expect("product node");
        let classification = classify_proposal_node(&egraph, product, node, &context)
            .expect("same selector distributes");
        assert!(classification.relation_redex);
        assert_eq!(classification.local_checked_relation_count, 1);
        assert_eq!(context.failure(), None);
    }

    #[test]
    fn right_nested_relation_boundary_rewrites_and_classifies() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (public, _) = matrix_atom(&mut egraph, "right-nested-public", None);
        let (relation, source) =
            matrix_atom(&mut egraph, "right-nested-relation", Some(AtomicRelationRole::Preimage));
        let (target, _) = matrix_atom(&mut egraph, "right-nested-target", None);
        let (tail, _) = matrix_atom(&mut egraph, "right-nested-tail", None);
        let inner = egraph.add(MxxLang::MatrixMultiply(vec![relation, tail].into_boxed_slice()));
        let root = egraph.add(MxxLang::MatrixMultiply(vec![public, inner].into_boxed_slice()));
        let expected = egraph.add(MxxLang::MatrixMultiply(vec![target, tail].into_boxed_slice()));
        egraph.rebuild();

        let context = RewriteContext::new(SharedRewriteBudget::new());
        context.register(registration(source, public, target));
        let node = egraph[egraph.find(root)]
            .nodes
            .iter()
            .find(|node| matches!(node, MxxLang::MatrixMultiply(_)))
            .expect("outer product witness");
        assert!(
            classify_proposal_node(&egraph, root, node, &context)
                .expect("closed relation")
                .relation_redex
        );
        let searcher = RelationSearcher::new(context.clone());
        assert!(Searcher::search_eclass_with_limit(&searcher, &egraph, root, 1).is_some());

        let applier = RelationApplier::new(context.clone());
        assert!(
            !Applier::apply_one(
                &applier,
                &mut egraph,
                root,
                &Subst::default(),
                None,
                Symbol::from("right-nested-relation"),
            )
            .is_empty()
        );
        assert_eq!(egraph.find(root), egraph.find(expected));
        let raw_node = egraph[egraph.find(root)]
            .nodes
            .iter()
            .find(|node| {
                matches!(node, MxxLang::MatrixMultiply(factors) if factors.len() == 2 &&
                    egraph.find(factors[0]) == egraph.find(public) &&
                    egraph.find(factors[1]) == egraph.find(inner))
            })
            .expect("the rewritten e-class retains its raw relation representation");
        let classification = classify_proposal_node(&egraph, root, raw_node, &context)
            .expect("canonical-equivalent replacement is satisfied");
        assert!(
            !classification.relation_redex,
            "an already-unioned relation must not remain an extraction obligation"
        );
        assert_eq!(
            classification.local_checked_relation_count, 1,
            "the raw relation node retains one checked relation boundary after replacement"
        );
        assert_eq!(context.failure(), None);
    }

    #[test]
    fn right_nested_relation_boundary_preserves_noncommutative_order() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (public, _) = matrix_atom(&mut egraph, "right-nested-order-public", None);
        let (relation, source) = matrix_atom(
            &mut egraph,
            "right-nested-order-relation",
            Some(AtomicRelationRole::Preimage),
        );
        let (target, _) = matrix_atom(&mut egraph, "right-nested-order-target", None);
        let (tail, _) = matrix_atom(&mut egraph, "right-nested-order-tail", None);
        let inner = egraph.add(MxxLang::MatrixMultiply(vec![tail, relation].into_boxed_slice()));
        let root = egraph.add(MxxLang::MatrixMultiply(vec![public, inner].into_boxed_slice()));
        egraph.rebuild();

        let context = RewriteContext::new(SharedRewriteBudget::new());
        context.register(registration(source, public, target));
        let node = egraph[egraph.find(root)]
            .nodes
            .iter()
            .find(|node| matches!(node, MxxLang::MatrixMultiply(_)))
            .expect("outer product witness");
        assert!(
            !classify_proposal_node(&egraph, root, node, &context)
                .expect("not an ordered match")
                .relation_redex
        );

        let applier = RelationApplier::new(context.clone());
        assert!(
            Applier::apply_one(
                &applier,
                &mut egraph,
                root,
                &Subst::default(),
                None,
                Symbol::from("right-nested-order"),
            )
            .is_empty()
        );
        assert_ne!(egraph.find(root), egraph.find(target));
        assert_eq!(context.failure(), None);
    }

    #[test]
    fn right_nested_relation_witnesses_are_independent_and_budgeted() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (public, _) = matrix_atom(&mut egraph, "right-nested-witness-public", None);
        let (relation, _) = matrix_atom(
            &mut egraph,
            "right-nested-witness-relation",
            Some(AtomicRelationRole::Preimage),
        );
        let (first_tail, _) = matrix_atom(&mut egraph, "right-nested-witness-first", None);
        let (second_tail, _) = matrix_atom(&mut egraph, "right-nested-witness-second", None);
        let first =
            egraph.add(MxxLang::MatrixMultiply(vec![relation, first_tail].into_boxed_slice()));
        let second =
            egraph.add(MxxLang::MatrixMultiply(vec![relation, second_tail].into_boxed_slice()));
        egraph.union(first, second);
        let root = egraph.add(MxxLang::MatrixMultiply(vec![public, first].into_boxed_slice()));
        egraph.rebuild();
        let factors = egraph[egraph.find(root)]
            .nodes
            .iter()
            .find_map(|node| match node {
                MxxLang::MatrixMultiply(factors) => Some(factors.clone()),
                _ => None,
            })
            .expect("outer product witness");

        let mut full_charges = 0;
        let candidates =
            right_nested_relation_factor_candidates_with_reserve(&egraph, &factors, &mut || {
                full_charges += 1;
                true
            })
            .expect("funded witness scan");
        assert_eq!(candidates.len(), 2, "physical witnesses remain independent");
        assert!(candidates.iter().all(|candidate| candidate.len() == 3));
        assert!(candidates.iter().any(|candidate| {
            candidate.as_ref() ==
                [egraph.find(public), egraph.find(relation), egraph.find(first_tail)]
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate.as_ref() ==
                [egraph.find(public), egraph.find(relation), egraph.find(second_tail)]
        }));

        let before = egraph.total_size();
        let mut charges = 0;
        assert!(
            right_nested_relation_factor_candidates_with_reserve(&egraph, &factors, &mut || {
                charges += 1;
                charges < full_charges
            },)
            .is_none()
        );
        assert_eq!(egraph.total_size(), before, "budget exhaustion is pre-mutation");

        let charges_for = |tail_count| {
            let mut egraph = EGraph::new(MxxAnalysis::default());
            let (public, _) = matrix_atom(&mut egraph, "right-nested-linear-public", None);
            let (relation, _) = matrix_atom(
                &mut egraph,
                "right-nested-linear-relation",
                Some(AtomicRelationRole::Preimage),
            );
            let tails = (0..tail_count)
                .map(|index| {
                    matrix_atom(&mut egraph, &format!("right-nested-linear-{index}"), None).0
                })
                .collect::<Vec<_>>();
            let inner = egraph.add(MxxLang::MatrixMultiply(
                std::iter::once(relation).chain(tails).collect::<Vec<_>>().into_boxed_slice(),
            ));
            let root = egraph.add(MxxLang::MatrixMultiply(vec![public, inner].into_boxed_slice()));
            egraph.rebuild();
            let factors = egraph[egraph.find(root)]
                .nodes
                .iter()
                .find_map(|node| match node {
                    MxxLang::MatrixMultiply(factors) => Some(factors.clone()),
                    _ => None,
                })
                .expect("outer product witness");
            let mut charges = 0;
            assert!(
                right_nested_relation_factor_candidates_with_reserve(
                    &egraph,
                    &factors,
                    &mut || {
                        charges += 1;
                        true
                    },
                )
                .is_some()
            );
            charges
        };
        let eight = charges_for(8);
        let sixteen = charges_for(16);
        assert!(sixteen > eight && sixteen <= eight * 3, "linear scan: {eight} -> {sixteen}");
    }

    #[test]
    fn applier_skips_different_selector_position_and_rewrites_later_relation() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let left_selector = egraph.add(MxxLang::IntConst(0.into()));
        let right_selector = egraph.add(MxxLang::IntConst(1.into()));
        let (public_case, _) = matrix_atom(&mut egraph, "skip-public", None);
        let (relation_case, _) =
            matrix_atom(&mut egraph, "skip-relation", Some(AtomicRelationRole::Preimage));
        let skipped_public =
            egraph.add(MxxLang::Switch(vec![left_selector, public_case].into_boxed_slice()));
        let skipped_relation =
            egraph.add(MxxLang::Switch(vec![right_selector, relation_case].into_boxed_slice()));
        let (expected_public, _) = matrix_atom(&mut egraph, "expected-public", None);
        let (relation, source) =
            matrix_atom(&mut egraph, "registered-relation", Some(AtomicRelationRole::Preimage));
        let (target, _) = matrix_atom(&mut egraph, "target", None);
        let root = egraph.add(MxxLang::MatrixMultiply(
            vec![skipped_public, skipped_relation, expected_public, relation].into_boxed_slice(),
        ));
        egraph.rebuild();
        let context = RewriteContext::new(SharedRewriteBudget::new());
        context.register(registration(source, expected_public, target));
        let applier = RelationApplier::new(context.clone());
        let rewrites = Applier::apply_one(
            &applier,
            &mut egraph,
            root,
            &Subst::default(),
            None,
            Symbol::from("skip-then-rewrite"),
        );
        assert!(!rewrites.is_empty(), "later registered relation rewrites");
        assert_eq!(context.failure(), None);
    }

    #[test]
    fn applier_keeps_genuine_registration_mismatch_sticky() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (public, _) = matrix_atom(&mut egraph, "actual-public", None);
        let (relation, source) =
            matrix_atom(&mut egraph, "mismatch-relation", Some(AtomicRelationRole::Preimage));
        let target = egraph.add(MxxLang::IntConst(0.into()));
        let root = egraph.add(MxxLang::MatrixMultiply(vec![public, relation].into_boxed_slice()));
        egraph.rebuild();
        let context = RewriteContext::new(SharedRewriteBudget::new());
        context.register(registration(source, public, target));
        let applier = RelationApplier::new(context.clone());
        assert!(
            Applier::apply_one(
                &applier,
                &mut egraph,
                root,
                &Subst::default(),
                None,
                Symbol::from("sticky-mismatch")
            )
            .is_empty()
        );
        assert_eq!(context.failure(), Some(RelationFailure::MismatchedType { source }));
    }

    #[test]
    fn pointwise_selector_product_rejects_different_two_sided_selectors() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let left_selector = egraph.add(MxxLang::IntConst(0.into()));
        let right_selector = egraph.add(MxxLang::IntConst(1.into()));
        let left_cases =
            [egraph.add(MxxLang::IntConst(2.into())), egraph.add(MxxLang::IntConst(3.into()))];
        let right_cases =
            [egraph.add(MxxLang::IntConst(5.into())), egraph.add(MxxLang::IntConst(7.into()))];
        let left = egraph.add(MxxLang::Switch(
            vec![left_selector, left_cases[0], left_cases[1]].into_boxed_slice(),
        ));
        let right = egraph.add(MxxLang::Switch(
            vec![right_selector, right_cases[0], right_cases[1]].into_boxed_slice(),
        ));
        assert_eq!(
            pointwise_same_selector(&mut egraph, left, right, true),
            Err(RelationFailure::DifferentSelectorBlocked),
        );
    }

    #[test]
    fn unmatched_public_registration_is_not_a_global_relation_failure() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (expected_public, _) = matrix_atom(&mut egraph, "expected", None);
        let (actual_public, _) = matrix_atom(&mut egraph, "actual", None);
        let (target, _) = matrix_atom(&mut egraph, "target", None);
        let (relation, source) =
            matrix_atom(&mut egraph, "relation", Some(AtomicRelationRole::Preimage));
        let context = RewriteContext::new(SharedRewriteBudget::new());
        context.register(registration(source, expected_public, target));

        assert!(
            checked_replacement(&mut egraph, &context, &[actual_public, relation], 1).is_none()
        );
        assert_eq!(context.failure(), None);
    }

    #[test]
    fn additive_distribution_keeps_nonmatching_residual_without_failure() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (prefix, _) = matrix_atom(&mut egraph, "prefix", None);
        let (expected_public, _) = matrix_atom(&mut egraph, "expected", None);
        let (residual, _) = matrix_atom(&mut egraph, "residual", None);
        let (target, _) = matrix_atom(&mut egraph, "target", None);
        let (relation, source) =
            matrix_atom(&mut egraph, "relation", Some(AtomicRelationRole::Preimage));
        let matching_summand =
            egraph.add(MxxLang::MatrixMultiply(vec![prefix, expected_public].into_boxed_slice()));
        let actual_public =
            egraph.add(MxxLang::MatrixAdd(vec![matching_summand, residual].into_boxed_slice()));
        let context = RewriteContext::new(SharedRewriteBudget::new());
        context.register(registration(source, expected_public, target));

        let replacement = checked_replacement(&mut egraph, &context, &[actual_public, relation], 1)
            .expect("matching additive summand is rewritten")
            .0;
        assert_eq!(context.failure(), None);
        let MxxLang::MatrixAdd(terms) = egraph[egraph.find(replacement)]
            .nodes
            .iter()
            .find(|node| matches!(node, MxxLang::MatrixAdd(_)))
            .expect("distribution produces an additive replacement")
        else {
            unreachable!()
        };
        assert!(terms.iter().any(|term| {
            egraph[egraph.find(*term)].nodes.iter().any(|node| {
                matches!(node, MxxLang::MatrixMultiply(factors)
                    if factors.len() == 2 &&
                        egraph.find(factors[0]) == egraph.find(residual) &&
                        egraph.find(factors[1]) == egraph.find(relation))
            })
        }));
    }

    #[test]
    fn additive_distribution_scans_later_physical_add_witnesses() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (prefix, _) = matrix_atom(&mut egraph, "later-add-prefix", None);
        let (expected_public, _) = matrix_atom(&mut egraph, "later-add-expected", None);
        let (nonmatching, _) = matrix_atom(&mut egraph, "later-add-nonmatching", None);
        let (residual, _) = matrix_atom(&mut egraph, "later-add-residual", None);
        let (target, _) = matrix_atom(&mut egraph, "later-add-target", None);
        let (relation, source) =
            matrix_atom(&mut egraph, "later-add-relation", Some(AtomicRelationRole::Preimage));
        let first = egraph.add(MxxLang::MatrixAdd(vec![nonmatching].into_boxed_slice()));
        let matching =
            egraph.add(MxxLang::MatrixMultiply(vec![prefix, expected_public].into_boxed_slice()));
        let second = egraph.add(MxxLang::MatrixAdd(vec![matching, residual].into_boxed_slice()));
        egraph.union(first, second);
        egraph.rebuild();
        let context = RewriteContext::new(SharedRewriteBudget::new());
        context.register(registration(source, expected_public, target));

        let replacement = checked_replacement(&mut egraph, &context, &[first, relation], 1)
            .expect("a later physical Add witness is applicable")
            .0;
        egraph.rebuild();
        assert_eq!(context.failure(), None);
        let is_residual_relation_product = |term: Id, expected_residual: Id| {
            egraph[egraph.find(term)].nodes.iter().any(|node| {
                matches!(node, MxxLang::MatrixMultiply(factors)
                    if factors.len() == 2 &&
                        egraph.find(factors[0]) == egraph.find(expected_residual) &&
                        egraph.find(factors[1]) == egraph.find(relation))
            })
        };
        assert!(egraph[egraph.find(replacement)].nodes.iter().any(|node| {
            let MxxLang::MatrixAdd(terms) = node else { return false };
            terms.iter().any(|term| is_residual_relation_product(*term, residual))
        }));
    }

    #[test]
    fn additive_distribution_scans_later_physical_multiply_witnesses() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (prefix, _) = matrix_atom(&mut egraph, "later-multiply-prefix", None);
        let (expected_public, _) = matrix_atom(&mut egraph, "later-multiply-expected", None);
        let (other_public, _) = matrix_atom(&mut egraph, "later-multiply-other", None);
        let (target, _) = matrix_atom(&mut egraph, "later-multiply-target", None);
        let (relation, source) =
            matrix_atom(&mut egraph, "later-multiply-relation", Some(AtomicRelationRole::Preimage));
        let first =
            egraph.add(MxxLang::MatrixMultiply(vec![prefix, other_public].into_boxed_slice()));
        let second =
            egraph.add(MxxLang::MatrixMultiply(vec![prefix, expected_public].into_boxed_slice()));
        egraph.union(first, second);
        let actual_public = egraph.add(MxxLang::MatrixAdd(vec![first].into_boxed_slice()));
        egraph.rebuild();
        let context = RewriteContext::new(SharedRewriteBudget::new());
        context.register(registration(source, expected_public, target));

        assert!(
            checked_replacement(&mut egraph, &context, &[actual_public, relation], 1).is_some(),
            "a later physical Multiply witness ending in the exact public operand is applicable"
        );
        assert_eq!(context.failure(), None);
    }

    #[test]
    fn additive_distribution_unions_all_applicable_physical_add_witnesses() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (first_prefix, _) = matrix_atom(&mut egraph, "all-add-first-prefix", None);
        let (second_prefix, _) = matrix_atom(&mut egraph, "all-add-second-prefix", None);
        let (expected_public, _) = matrix_atom(&mut egraph, "all-add-expected", None);
        let (first_residual, _) = matrix_atom(&mut egraph, "all-add-first-residual", None);
        let (second_residual, _) = matrix_atom(&mut egraph, "all-add-second-residual", None);
        let (target, _) = matrix_atom(&mut egraph, "all-add-target", None);
        let (relation, source) =
            matrix_atom(&mut egraph, "all-add-relation", Some(AtomicRelationRole::Preimage));
        let first_match = egraph
            .add(MxxLang::MatrixMultiply(vec![first_prefix, expected_public].into_boxed_slice()));
        let second_match = egraph
            .add(MxxLang::MatrixMultiply(vec![second_prefix, expected_public].into_boxed_slice()));
        let first =
            egraph.add(MxxLang::MatrixAdd(vec![first_match, first_residual].into_boxed_slice()));
        let second =
            egraph.add(MxxLang::MatrixAdd(vec![second_match, second_residual].into_boxed_slice()));
        egraph.union(first, second);
        egraph.rebuild();
        let context = RewriteContext::new(SharedRewriteBudget::new());
        context.register(registration(source, expected_public, target));

        let replacement = checked_replacement(&mut egraph, &context, &[first, relation], 1)
            .expect("both physical Add witnesses are applicable")
            .0;
        egraph.rebuild();
        assert_eq!(context.failure(), None);
        let residual_is_preserved = |residual| {
            egraph[egraph.find(replacement)].nodes.iter().any(|node| {
                let MxxLang::MatrixAdd(terms) = node else { return false };
                terms.iter().any(|term| {
                    egraph[egraph.find(*term)].nodes.iter().any(|node| {
                        matches!(node, MxxLang::MatrixMultiply(factors)
                            if factors.len() == 2 &&
                                egraph.find(factors[0]) == egraph.find(residual) &&
                                egraph.find(factors[1]) == egraph.find(relation))
                    })
                })
            })
        };
        assert!(residual_is_preserved(first_residual));
        assert!(residual_is_preserved(second_residual));
    }

    #[test]
    fn additive_distribution_consumes_all_matching_summands_in_one_add_witness() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (left, _) = matrix_atom(&mut egraph, "all-summands-left", None);
        let (right, _) = matrix_atom(&mut egraph, "all-summands-right", None);
        let (public, _) = matrix_atom(&mut egraph, "all-summands-public", None);
        let (target, _) = matrix_atom(&mut egraph, "all-summands-target", None);
        let (relation, source) =
            matrix_atom(&mut egraph, "all-summands-relation", Some(AtomicRelationRole::Preimage));
        let first = egraph.add(MxxLang::MatrixMultiply(vec![left, public].into_boxed_slice()));
        let second = egraph.add(MxxLang::MatrixMultiply(vec![right, public].into_boxed_slice()));
        let actual = egraph.add(MxxLang::MatrixAdd(vec![first, second].into_boxed_slice()));
        egraph.rebuild();
        let context = RewriteContext::new(SharedRewriteBudget::new());
        context.register(registration(source, public, target));

        let replacement = checked_replacement(&mut egraph, &context, &[actual, relation], 1)
            .expect("both matching summands are consumed together")
            .0;
        egraph.rebuild();
        let matching_products = egraph[egraph.find(replacement)]
            .nodes
            .iter()
            .filter_map(|node| match node {
                MxxLang::MatrixAdd(terms) => Some(terms),
                _ => None,
            })
            .any(|terms| {
                terms
                    .iter()
                    .filter(|term| {
                        egraph[egraph.find(**term)].nodes.iter().any(|node| {
                            matches!(node, MxxLang::MatrixMultiply(factors)
                        if factors.len() == 2 && egraph.find(factors[1]) == egraph.find(target))
                        })
                    })
                    .count() ==
                    2
            });
        assert!(matching_products);
    }

    #[test]
    fn additive_distribution_splices_an_additive_target_into_one_flat_add() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (prefix, _) = matrix_atom(&mut egraph, "flat-target-prefix", None);
        let (public, _) = matrix_atom(&mut egraph, "flat-target-public", None);
        let (first_target, _) = matrix_atom(&mut egraph, "flat-target-first", None);
        let (second_target, _) = matrix_atom(&mut egraph, "flat-target-second", None);
        let target =
            egraph.add(MxxLang::MatrixAdd(vec![first_target, second_target].into_boxed_slice()));
        let (relation, source) =
            matrix_atom(&mut egraph, "flat-target-relation", Some(AtomicRelationRole::Preimage));
        let matching = egraph.add(MxxLang::MatrixMultiply(vec![prefix, public].into_boxed_slice()));
        let actual = egraph.add(MxxLang::MatrixAdd(vec![matching].into_boxed_slice()));
        egraph.rebuild();
        let context = RewriteContext::new(SharedRewriteBudget::new());
        context.register(registration(source, public, target));

        let replacement = checked_replacement(&mut egraph, &context, &[actual, relation], 1)
            .expect("additive target distributes")
            .0;
        egraph.rebuild();
        let outer_terms = egraph[egraph.find(replacement)]
            .nodes
            .iter()
            .find_map(|node| match node {
                MxxLang::MatrixAdd(terms) => Some(terms),
                _ => None,
            })
            .expect("one flat distributed Add");
        assert_eq!(outer_terms.len(), 2);
        for target_term in [first_target, second_target] {
            assert!(outer_terms.iter().any(|term| {
                egraph[egraph.find(*term)].nodes.iter().any(|node| {
                    matches!(node, MxxLang::MatrixMultiply(factors)
                        if factors.len() == 2 &&
                            egraph.find(factors[0]) == egraph.find(prefix) &&
                            egraph.find(factors[1]) == egraph.find(target_term))
                })
            }));
        }
        assert!(outer_terms.iter().all(|term| {
            !egraph[egraph.find(*term)]
                .nodes
                .iter()
                .any(|node| matches!(node, MxxLang::MatrixAdd(_)))
        }));
    }

    #[test]
    fn additive_distribution_unions_multiple_matching_product_witnesses_without_cartesian_adds() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (first_prefix, _) = matrix_atom(&mut egraph, "multi-product-first", None);
        let (second_prefix, _) = matrix_atom(&mut egraph, "multi-product-second", None);
        let (public, _) = matrix_atom(&mut egraph, "multi-product-public", None);
        let (target, _) = matrix_atom(&mut egraph, "multi-product-target", None);
        let (relation, source) =
            matrix_atom(&mut egraph, "multi-product-relation", Some(AtomicRelationRole::Preimage));
        let first =
            egraph.add(MxxLang::MatrixMultiply(vec![first_prefix, public].into_boxed_slice()));
        let second =
            egraph.add(MxxLang::MatrixMultiply(vec![second_prefix, public].into_boxed_slice()));
        egraph.union(first, second);
        let actual = egraph.add(MxxLang::MatrixAdd(vec![first].into_boxed_slice()));
        egraph.rebuild();
        let context = RewriteContext::new(SharedRewriteBudget::new());
        context.register(registration(source, public, target));

        let replacement = checked_replacement(&mut egraph, &context, &[actual, relation], 1)
            .expect("matching product alternatives are retained")
            .0;
        egraph.rebuild();
        assert_eq!(
            egraph[egraph.find(replacement)]
                .nodes
                .iter()
                .filter(|node| matches!(node, MxxLang::MatrixAdd(_)))
                .count(),
            1,
            "one Add replacement retains locally unioned product alternatives without combinations"
        );
    }

    #[test]
    fn ambiguous_distribution_plans_do_not_materialize_nodes() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (prefix, _) = matrix_atom(&mut egraph, "ambiguous-plan-prefix", None);
        let (public, _) = matrix_atom(&mut egraph, "ambiguous-plan-public", None);
        let (first_target, _) = matrix_atom(&mut egraph, "ambiguous-plan-first-target", None);
        let (second_target, _) = matrix_atom(&mut egraph, "ambiguous-plan-second-target", None);
        let (relation, source) =
            matrix_atom(&mut egraph, "ambiguous-plan-relation", Some(AtomicRelationRole::Preimage));
        let matching = egraph.add(MxxLang::MatrixMultiply(vec![prefix, public].into_boxed_slice()));
        let actual = egraph.add(MxxLang::MatrixAdd(vec![matching].into_boxed_slice()));
        egraph.rebuild();
        let before = egraph.total_size();
        let context = RewriteContext::new(SharedRewriteBudget::new());
        context.register(registration(source, public, first_target));
        context.register(registration(source, public, second_target));

        assert!(checked_replacement(&mut egraph, &context, &[actual, relation], 1).is_none());
        assert_eq!(
            context.failure(),
            Some(RelationFailure::AmbiguousReplacement {
                sources: vec![source].into_boxed_slice()
            })
        );
        assert_eq!(egraph.total_size(), before, "ambiguous plans remain unmaterialized");
    }

    #[test]
    fn failed_distribution_preflight_does_not_materialize_nodes() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (prefix, _) = matrix_atom(&mut egraph, "failed-plan-prefix", None);
        let (public, _) = matrix_atom(&mut egraph, "failed-plan-public", None);
        let invalid_target = egraph.add(MxxLang::IntConst(0.into()));
        let (relation, source) =
            matrix_atom(&mut egraph, "failed-plan-relation", Some(AtomicRelationRole::Preimage));
        let matching = egraph.add(MxxLang::MatrixMultiply(vec![prefix, public].into_boxed_slice()));
        let actual = egraph.add(MxxLang::MatrixAdd(vec![matching].into_boxed_slice()));
        egraph.rebuild();
        let before = egraph.total_size();
        let context = RewriteContext::new(SharedRewriteBudget::new());
        context.register(registration(source, public, invalid_target));

        assert!(checked_replacement(&mut egraph, &context, &[actual, relation], 1).is_none());
        assert_eq!(context.failure(), Some(RelationFailure::MismatchedType { source }));
        assert_eq!(egraph.total_size(), before, "failed distribution remains unmaterialized");
    }

    #[test]
    fn distribution_plan_work_grows_nearly_linearly_with_stored_children() {
        let charged_work = |children| {
            let mut egraph = EGraph::new(MxxAnalysis::default());
            let (public, _) = matrix_atom(&mut egraph, "linear-plan-public", None);
            let (target, _) = matrix_atom(&mut egraph, "linear-plan-target", None);
            let (relation, source) = matrix_atom(
                &mut egraph,
                "linear-plan-relation",
                Some(AtomicRelationRole::Preimage),
            );
            let summands = (0..children)
                .map(|index| {
                    let (prefix, _) =
                        matrix_atom(&mut egraph, &format!("linear-plan-prefix-{index}"), None);
                    egraph.add(MxxLang::MatrixMultiply(vec![prefix, public].into_boxed_slice()))
                })
                .collect::<Vec<_>>();
            let actual = egraph.add(MxxLang::MatrixAdd(summands.into_boxed_slice()));
            egraph.rebuild();
            let budget = SharedRewriteBudget::new();
            let context = RewriteContext::new(budget.clone());
            context.register(registration(source, public, target));
            assert!(checked_replacement(&mut egraph, &context, &[actual, relation], 1).is_some());
            budget.owned()
        };

        let work_at_eight = charged_work(8);
        let work_at_sixteen = charged_work(16);
        assert!(work_at_sixteen > work_at_eight);
        assert!(
            work_at_sixteen <= work_at_eight * 3,
            "doubling stored children must retain near-linear charged copy work: {work_at_eight} -> {work_at_sixteen}"
        );
    }

    #[test]
    fn applicable_relation_with_malformed_target_remains_fail_closed() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (public, _) = matrix_atom(&mut egraph, "public", None);
        let target = egraph.add(MxxLang::IntConst(0.into()));
        let (relation, source) =
            matrix_atom(&mut egraph, "relation", Some(AtomicRelationRole::Preimage));
        let context = RewriteContext::new(SharedRewriteBudget::new());
        context.register(registration(source, public, target));

        assert!(checked_replacement(&mut egraph, &context, &[public, relation], 1).is_none());
        assert_eq!(context.failure(), Some(RelationFailure::MismatchedType { source }));
    }

    #[test]
    fn preimage_relation_accepts_equal_closed_dimension_expressions() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let public_type = ResolvedMatrixType {
            modulus: ResolvedIntExpr::Const(17.into()),
            ring_dimension: ResolvedIntExpr::Const(1.into()),
            rows: ResolvedIntExpr::Const(2.into()),
            columns: ResolvedIntExpr::Add(
                Box::new(ResolvedIntExpr::Const(1.into())),
                Box::new(ResolvedIntExpr::Const(1.into())),
            ),
        };
        let relation_type = ResolvedMatrixType {
            modulus: ResolvedIntExpr::Add(
                Box::new(ResolvedIntExpr::Const(16.into())),
                Box::new(ResolvedIntExpr::Const(1.into())),
            ),
            ring_dimension: ResolvedIntExpr::Sub(
                Box::new(ResolvedIntExpr::Const(2.into())),
                Box::new(ResolvedIntExpr::Const(1.into())),
            ),
            rows: ResolvedIntExpr::Mul(
                Box::new(ResolvedIntExpr::Const(1.into())),
                Box::new(ResolvedIntExpr::Const(2.into())),
            ),
            columns: ResolvedIntExpr::Const(3.into()),
        };
        let target_type = ResolvedMatrixType {
            modulus: ResolvedIntExpr::Const(17.into()),
            ring_dimension: ResolvedIntExpr::Const(1.into()),
            rows: ResolvedIntExpr::Sub(
                Box::new(ResolvedIntExpr::Const(3.into())),
                Box::new(ResolvedIntExpr::Const(1.into())),
            ),
            columns: ResolvedIntExpr::Add(
                Box::new(ResolvedIntExpr::Const(2.into())),
                Box::new(ResolvedIntExpr::Const(1.into())),
            ),
        };
        let (public, _) = matrix_atom_with_type(&mut egraph, "public", public_type, None);
        let (target, _) = matrix_atom_with_type(&mut egraph, "target", target_type, None);
        let (relation, source) = matrix_atom_with_type(
            &mut egraph,
            "relation",
            relation_type,
            Some(AtomicRelationRole::Preimage),
        );
        let context = RewriteContext::new(SharedRewriteBudget::new());
        context.register(registration(source, public, target));

        let product = egraph.add(MxxLang::MatrixMultiply(vec![public, relation].into()));
        let replacement = checked_replacement(&mut egraph, &context, &[public, relation], 1)
            .expect("closed-equivalent relation layout is rewritten")
            .0;
        egraph.union(product, replacement);
        egraph.rebuild();

        assert!(matches!(egraph[egraph.find(product)].data.sort, Ok(MxxSort::Matrix(_))));
        assert_eq!(context.failure(), None);
    }

    #[test]
    fn preimage_relation_rejects_unequal_closed_dimensions() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let matrix = |rows: i32, columns: i32| ResolvedMatrixType {
            modulus: ResolvedIntExpr::Const(17.into()),
            ring_dimension: ResolvedIntExpr::Const(1.into()),
            rows: ResolvedIntExpr::Const(rows.into()),
            columns: ResolvedIntExpr::Const(columns.into()),
        };
        let (public, _) = matrix_atom_with_type(&mut egraph, "public", matrix(2, 2), None);
        let (target, _) = matrix_atom_with_type(&mut egraph, "target", matrix(3, 3), None);
        let (relation, source) = matrix_atom_with_type(
            &mut egraph,
            "relation",
            matrix(2, 3),
            Some(AtomicRelationRole::Preimage),
        );
        let context = RewriteContext::new(SharedRewriteBudget::new());
        context.register(registration(source, public, target));

        assert!(checked_replacement(&mut egraph, &context, &[public, relation], 1).is_none());
        assert_eq!(context.failure(), Some(RelationFailure::MismatchedType { source }));
    }

    #[test]
    fn preimage_relation_rejects_different_runtime_dimension_binders() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let matrix = |rows, columns| ResolvedMatrixType {
            modulus: ResolvedIntExpr::Const(17.into()),
            ring_dimension: ResolvedIntExpr::Const(1.into()),
            rows,
            columns,
        };
        let left = super::super::identity::BinderKey {
            loop_scope: super::super::identity::OccurrenceScope {
                program: super::super::identity::ProgramKey::Ideal,
                definition: mxx_ir_core::FrozenGraphScopeId::Root,
                path: Box::new([]),
            },
            loop_node: mxx_ir_core::NodeId(1),
            slot: 0,
        };
        let right =
            super::super::identity::BinderKey { loop_node: mxx_ir_core::NodeId(2), ..left.clone() };
        let (public, _) = matrix_atom_with_type(
            &mut egraph,
            "public",
            matrix(ResolvedIntExpr::Const(2.into()), ResolvedIntExpr::Binder(left)),
            None,
        );
        let (target, _) = matrix_atom_with_type(
            &mut egraph,
            "target",
            matrix(ResolvedIntExpr::Const(2.into()), ResolvedIntExpr::Const(3.into())),
            None,
        );
        let (relation, source) = matrix_atom_with_type(
            &mut egraph,
            "relation",
            matrix(ResolvedIntExpr::Binder(right), ResolvedIntExpr::Const(3.into())),
            Some(AtomicRelationRole::Preimage),
        );
        let context = RewriteContext::new(SharedRewriteBudget::new());
        context.register(registration(source, public, target));

        assert!(checked_replacement(&mut egraph, &context, &[public, relation], 1).is_none());
        assert_eq!(context.failure(), Some(RelationFailure::MismatchedType { source }));
    }

    #[test]
    fn pointwise_add_switch_cancels_fixed_terms_in_every_stored_case() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let (signal, _) = matrix_atom(&mut egraph, "signal", None);
        let negated = egraph.add(MxxLang::MatrixNegate([signal]));
        let switch = egraph.add(MxxLang::Switch(vec![selector, signal, signal].into_boxed_slice()));
        let root = egraph.add(MxxLang::MatrixAdd(vec![switch, negated].into_boxed_slice()));
        egraph.rebuild();

        let plan = pointwise_add_switch_cancellation_plan(&egraph, root).expect("all cases cancel");
        let replacement = build_pointwise_add_switch_cancellation(&mut egraph, root, plan)
            .expect("direct plan builds");
        let MxxLang::Switch(cases) = egraph[egraph.find(replacement)]
            .nodes
            .iter()
            .find(|node| matches!(node, MxxLang::Switch(_)))
            .expect("one pointwise switch")
        else {
            unreachable!()
        };
        assert_eq!(egraph.find(cases[0]), egraph.find(selector));
        assert_eq!(cases.len(), 3);
        for case in &cases[1..] {
            assert!(
                matches!(egraph[egraph.find(*case)].nodes.as_slice(), [MxxLang::MatrixConstant(spec)]
                if matches!(egraph.analysis.symbols.matrix_constants.get(spec.0), Some(super::super::identity::MatrixConstantSpec { value: MatrixConstantValue::Zero, .. })))
            );
        }
    }

    #[test]
    fn pointwise_add_switch_instantiates_one_owner_binder_per_stored_case() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let binder = test_binder(&mut egraph, 0, 1);
        let selector = egraph.add(MxxLang::IntBinder(binder));
        let shared = binder_matrix_atom(&mut egraph, selector, "shared");
        let zero = egraph.add(MxxLang::IntConst(0.into()));
        let one = egraph.add(MxxLang::IntConst(1.into()));
        let first =
            family::instantiate_shared_element(&mut egraph, shared, binder, zero, &mut || {
                Ok::<(), ()>(())
            })
            .expect("test instantiation");
        let second =
            family::instantiate_shared_element(&mut egraph, shared, binder, one, &mut || {
                Ok::<(), ()>(())
            })
            .expect("test instantiation");
        let (left, _) = matrix_atom(&mut egraph, "left", None);
        let (right, _) = matrix_atom(&mut egraph, "right", None);
        let first_case =
            egraph.add(MxxLang::MatrixAdd(vec![first, left, right].into_boxed_slice()));
        let second_case =
            egraph.add(MxxLang::MatrixAdd(vec![second, right, left].into_boxed_slice()));
        let switch =
            egraph.add(MxxLang::Switch(vec![selector, first_case, second_case].into_boxed_slice()));
        let fixed = egraph.add(MxxLang::MatrixNegate([shared]));
        let root = egraph.add(MxxLang::MatrixAdd(vec![switch, fixed].into_boxed_slice()));
        egraph.rebuild();

        let plan = pointwise_add_switch_cancellation_plan(&egraph, root).expect("binder plan");
        assert!(plan.binder_aware.is_some());
        let replacement = build_pointwise_add_switch_cancellation(&mut egraph, root, plan)
            .expect("all physical cases match");
        let cases = switch_node(&egraph, replacement).expect("replacement switch");
        assert_eq!(egraph.find(cases[0]), egraph.find(selector));
        for (case, expected) in [(cases[1], [left, right]), (cases[2], [right, left])] {
            assert!(egraph[egraph.find(case)].nodes.iter().any(|node| {
                matches!(node, MxxLang::MatrixAdd(terms)
                    if terms.iter().map(|term| egraph.find(*term)).eq(expected.into_iter().map(|term| egraph.find(term))))
            }));
        }
    }

    #[test]
    fn pointwise_probe_reports_binder_ready_and_preflight_rejections() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let binder = test_binder(&mut egraph, 0, 1);
        let selector = egraph.add(MxxLang::IntBinder(binder));
        let shared = binder_matrix_atom(&mut egraph, selector, "probe-shared");
        let (first, _) = matrix_atom(&mut egraph, "probe-first", None);
        let (second, _) = matrix_atom(&mut egraph, "probe-second", None);
        let fixed = egraph.add(MxxLang::MatrixNegate([shared]));
        let switch = egraph.add(MxxLang::Switch(vec![selector, first, second].into_boxed_slice()));
        let root = egraph.add(MxxLang::MatrixAdd(vec![switch, fixed].into_boxed_slice()));
        egraph.rebuild();
        let before = egraph.total_size();

        let probe = pointwise_add_switch_probe(&egraph, root);
        assert_eq!(probe.eligible_single_switch_adds, 1);
        assert_eq!(probe.outcomes.len(), 1);
        assert!(matches!(
            &probe.outcomes[0].binder,
            Ok(BinderPreflightReady {
                fixed_terms,
                fixed_terms_omitted_occurrences: 0,
                unique_base_count: 1,
                case_count: 2,
            }) if fixed_terms.as_ref() == [SignedCanonicalMultiplicity {
                eclass: usize::from(egraph.find(shared)), negative: true, multiplicity: 1
            }]
        ));
        assert_eq!(egraph.total_size(), before);

        let non_binder_selector = egraph.add(MxxLang::IntConst(0.into()));
        let non_binder_switch =
            egraph.add(MxxLang::Switch(vec![non_binder_selector, shared].into_boxed_slice()));
        let non_binder_root =
            egraph.add(MxxLang::MatrixAdd(vec![non_binder_switch, fixed].into_boxed_slice()));
        egraph.rebuild();
        assert!(matches!(
            pointwise_add_switch_probe(&egraph, non_binder_root).outcomes[0].binder,
            Err(BinderPreflightReject::SelectorNotUniqueBinder)
        ));

        let bad_domain = test_binder_at(&mut egraph, 1, 2, 99);
        let bad_selector = egraph.add(MxxLang::IntBinder(bad_domain));
        let bad_switch =
            egraph.add(MxxLang::Switch(vec![bad_selector, first, second].into_boxed_slice()));
        let bad_root = egraph.add(MxxLang::MatrixAdd(vec![bad_switch, fixed].into_boxed_slice()));
        egraph.rebuild();
        assert!(matches!(
            pointwise_add_switch_probe(&egraph, bad_root).outcomes[0].binder,
            Err(BinderPreflightReject::DomainCaseCountMismatch { binder: actual, case_count: 2 })
                if actual == bad_domain
        ));

        let mut missing = EGraph::new(MxxAnalysis::default());
        let missing_selector = missing.add(MxxLang::IntBinder(BinderId(u32::MAX)));
        let (case, _) = matrix_atom(&mut missing, "missing-binder-case", None);
        let fixed = missing.add(MxxLang::MatrixNegate([case]));
        let switch = missing.add(MxxLang::Switch(vec![missing_selector, case].into_boxed_slice()));
        let root = missing.add(MxxLang::MatrixAdd(vec![switch, fixed].into_boxed_slice()));
        missing.rebuild();
        assert!(matches!(
            pointwise_add_switch_probe(&missing, root).outcomes[0].binder,
            Err(BinderPreflightReject::MissingDescriptor { binder: actual })
                if actual == BinderId(u32::MAX)
        ));
    }

    #[test]
    fn pointwise_probe_caps_physical_structures_in_stable_order() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let (signal, _) = matrix_atom(&mut egraph, "probe-signal", None);
        let fixed = egraph.add(MxxLang::MatrixNegate([signal]));
        let mut root = None;
        for index in 0..(MAX_POINTWISE_PROBE_STRUCTURES + 1) {
            let (first, _) = matrix_atom(&mut egraph, &format!("probe-case-{index}"), None);
            let switch = egraph.add(MxxLang::Switch(vec![selector, first].into_boxed_slice()));
            let candidate = egraph.add(MxxLang::MatrixAdd(vec![switch, fixed].into_boxed_slice()));
            if let Some(root) = root {
                egraph.union(root, candidate);
            } else {
                root = Some(candidate);
            }
        }
        let root = root.expect("one physical Add");
        egraph.rebuild();
        let before = egraph.total_size();

        let first = pointwise_add_switch_probe(&egraph, root);
        let second = pointwise_add_switch_probe(&egraph, root);
        assert_eq!(first, second);
        assert_eq!(first.eligible_single_switch_adds, MAX_POINTWISE_PROBE_STRUCTURES + 1);
        assert_eq!(first.outcomes.len(), MAX_POINTWISE_PROBE_STRUCTURES);
        assert_eq!(first.omitted_eligible_structures, 1);
        assert_eq!(egraph.total_size(), before);
    }

    #[test]
    fn pointwise_probe_names_fixed_and_case_preflight_failures() {
        let build =
            |egraph: &mut EGraph<MxxLang, MxxAnalysis>, selector: Id, fixed: Id, cases: Vec<Id>| {
                let mut children = vec![selector];
                children.extend(cases);
                let switch = egraph.add(MxxLang::Switch(children.into_boxed_slice()));
                egraph.add(MxxLang::MatrixAdd(vec![switch, fixed].into_boxed_slice()))
            };

        let mut egraph = EGraph::new(MxxAnalysis::default());
        let binder = test_binder(&mut egraph, 0, 1);
        let selector = egraph.add(MxxLang::IntBinder(binder));
        let (signal, _) = matrix_atom(&mut egraph, "flatten-signal", None);
        let fixed = egraph.add(MxxLang::MatrixAdd(Vec::new().into_boxed_slice()));
        let (first, _) = matrix_atom(&mut egraph, "flatten-first", None);
        let (second, _) = matrix_atom(&mut egraph, "flatten-second", None);
        let root = build(&mut egraph, selector, fixed, vec![first, second]);
        egraph.rebuild();
        assert!(matches!(
            pointwise_add_switch_probe(&egraph, root).outcomes[0].binder,
            Err(BinderPreflightReject::FixedSignedFlatten)
        ));

        let nested = egraph.add(MxxLang::Switch(vec![selector, first, second].into_boxed_slice()));
        let fixed = egraph.add(MxxLang::MatrixNegate([signal]));
        let root = build(&mut egraph, selector, fixed, vec![nested, second]);
        egraph.rebuild();
        assert!(matches!(
            pointwise_add_switch_probe(&egraph, root).outcomes[0].binder,
            Ok(BinderPreflightReady { case_count: 2, .. })
        ));

        let ambiguous_left = egraph.add(MxxLang::MatrixAdd(vec![first].into_boxed_slice()));
        let ambiguous_right = egraph.add(MxxLang::MatrixAdd(vec![second].into_boxed_slice()));
        egraph.union(ambiguous_left, ambiguous_right);
        let root = build(&mut egraph, selector, fixed, vec![ambiguous_left, second]);
        egraph.rebuild();
        assert!(matches!(
            pointwise_add_switch_probe(&egraph, root).outcomes[0].binder,
            Err(BinderPreflightReject::CaseAmbiguous { case_index: 0 })
        ));
    }

    #[test]
    fn binder_preflight_reports_fixed_switch_and_root_case_self_cycle_without_rebuild() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let binder = test_binder(&mut egraph, 0, 1);
        let selector = egraph.add(MxxLang::IntBinder(binder));
        let (signal, _) = matrix_atom(&mut egraph, "preflight-signal", None);
        let (second, _) = matrix_atom(&mut egraph, "preflight-second", None);
        let nested = egraph.add(MxxLang::Switch(vec![selector, signal, second].into_boxed_slice()));
        let fixed = egraph.add(MxxLang::MatrixNegate([signal]));
        let switch = egraph.add(MxxLang::Switch(vec![selector, signal, second].into_boxed_slice()));
        let root = egraph.add(MxxLang::MatrixAdd(vec![switch, fixed].into_boxed_slice()));
        let fixed_switch_structure = PointwiseAddSwitchStructure {
            terms: vec![switch, nested],
            switch_index: 0,
            switch: vec![selector, signal, second].into_boxed_slice(),
        };
        assert!(matches!(
            binder_aware_pointwise_add_switch_cancellation_for_structure(
                &egraph,
                root,
                &fixed_switch_structure,
            ),
            Err(BinderPreflightReject::FixedTermsEmptyOrSwitch)
        ));
        // Construct the preflight boundary directly instead of unioning a
        // root with one of its descendants.  Such a rebuild is intentionally
        // outside the honest-graph model and can make egg saturation loop.
        let structure = PointwiseAddSwitchStructure {
            terms: vec![switch, fixed],
            switch_index: 0,
            switch: vec![selector, root, second].into_boxed_slice(),
        };
        assert!(matches!(
            binder_aware_pointwise_add_switch_cancellation_for_structure(&egraph, root, &structure),
            Err(BinderPreflightReject::CaseSelfCycle { case_index: 0 })
        ));
    }

    #[test]
    fn binder_preflight_ready_caps_sorted_signed_leaf_summary() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let binder = test_binder(&mut egraph, 0, 1);
        let selector = egraph.add(MxxLang::IntBinder(binder));
        let (first, _) = matrix_atom(&mut egraph, "cap-first", None);
        let (second, _) = matrix_atom(&mut egraph, "cap-second", None);
        let switch = egraph.add(MxxLang::Switch(vec![selector, first, second].into_boxed_slice()));
        let mut terms = vec![switch];
        for index in 0..(MAX_UNMATCHED_TERM_IDENTITIES + 1) {
            let (term, _) = matrix_atom(&mut egraph, &format!("cap-fixed-{index}"), None);
            terms.push(egraph.add(MxxLang::MatrixNegate([term])));
        }
        let root = egraph.add(MxxLang::MatrixAdd(terms.into_boxed_slice()));
        egraph.rebuild();

        let probe = pointwise_add_switch_probe(&egraph, root);
        let ready = probe.outcomes[0].binder.as_ref().expect("symbolic preflight is ready");
        assert_eq!(ready.fixed_terms.len(), MAX_UNMATCHED_TERM_IDENTITIES);
        assert_eq!(ready.fixed_terms_omitted_occurrences, 1);
        assert_eq!(ready.unique_base_count, MAX_UNMATCHED_TERM_IDENTITIES + 1);
        assert_eq!(ready.case_count, 2);
        assert!(ready.fixed_terms.iter().all(|term| term.negative && term.multiplicity == 1));
        assert!(ready.fixed_terms.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn pointwise_add_switch_rejects_non_authoritative_domain_and_retains_foreign_owner() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let binder = test_binder(&mut egraph, 1, 2);
        let selector = egraph.add(MxxLang::IntBinder(binder));
        let shared = binder_matrix_atom(&mut egraph, selector, "shared");
        let first = egraph.add(MxxLang::IntConst(0.into()));
        let second = egraph.add(MxxLang::IntConst(1.into()));
        let first =
            family::instantiate_shared_element(&mut egraph, shared, binder, first, &mut || {
                Ok::<(), ()>(())
            })
            .expect("test instantiation");
        let second =
            family::instantiate_shared_element(&mut egraph, shared, binder, second, &mut || {
                Ok::<(), ()>(())
            })
            .expect("test instantiation");
        let switch = egraph.add(MxxLang::Switch(vec![selector, first, second].into_boxed_slice()));
        let fixed = egraph.add(MxxLang::MatrixNegate([shared]));
        let root = egraph.add(MxxLang::MatrixAdd(vec![switch, fixed].into_boxed_slice()));
        egraph.rebuild();
        assert!(pointwise_add_switch_cancellation_plan(&egraph, root).is_none());

        let owner = test_binder(&mut egraph, 0, 1);
        let foreign = test_binder_at(&mut egraph, 0, 1, 2);
        let owner_selector = egraph.add(MxxLang::IntBinder(owner));
        let foreign_selector = egraph.add(MxxLang::IntBinder(foreign));
        let foreign_shared = binder_matrix_atom(&mut egraph, foreign_selector, "foreign-shared");
        let foreign_zero = egraph.add(MxxLang::IntConst(0.into()));
        let foreign_one = egraph.add(MxxLang::IntConst(1.into()));
        let foreign_first = family::instantiate_shared_element(
            &mut egraph,
            foreign_shared,
            foreign,
            foreign_zero,
            &mut || Ok::<(), ()>(()),
        )
        .expect("test instantiation");
        let foreign_second = family::instantiate_shared_element(
            &mut egraph,
            foreign_shared,
            foreign,
            foreign_one,
            &mut || Ok::<(), ()>(()),
        )
        .expect("test instantiation");
        let foreign_fixed = egraph.add(MxxLang::MatrixNegate([foreign_shared]));
        let foreign_switch = egraph.add(MxxLang::Switch(
            vec![owner_selector, foreign_first, foreign_second].into_boxed_slice(),
        ));
        let foreign_root =
            egraph.add(MxxLang::MatrixAdd(vec![foreign_switch, foreign_fixed].into_boxed_slice()));
        egraph.rebuild();
        let plan = pointwise_add_switch_cancellation_plan(&egraph, foreign_root)
            .expect("foreign binder is retained rather than rejected");
        assert!(plan.binder_aware.is_some());
        assert!(build_pointwise_add_switch_cancellation(&mut egraph, foreign_root, plan).is_none());
    }

    #[test]
    fn pointwise_binder_fallback_distributes_wrong_indices_and_multiplicity_but_rejects_competing_selector()
     {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let binder = test_binder(&mut egraph, 0, 1);
        let selector = egraph.add(MxxLang::IntBinder(binder));
        let shared = binder_matrix_atom(&mut egraph, selector, "symbolic");
        let zero = egraph.add(MxxLang::IntConst(0.into()));
        let one = egraph.add(MxxLang::IntConst(1.into()));
        let at_zero =
            family::instantiate_shared_element(&mut egraph, shared, binder, zero, &mut || {
                Ok::<(), ()>(())
            })
            .expect("test instantiation");
        let at_one =
            family::instantiate_shared_element(&mut egraph, shared, binder, one, &mut || {
                Ok::<(), ()>(())
            })
            .expect("test instantiation");

        let wrong_constant =
            symbolic_two_case_root(&mut egraph, selector, shared, at_one, at_zero, 1);
        egraph.rebuild();
        let plan = pointwise_add_switch_cancellation_plan(&egraph, wrong_constant)
            .expect("only the binder fallback is eligible");
        assert!(plan.binder_aware.is_some());
        assert!(
            build_pointwise_add_switch_cancellation(&mut egraph, wrong_constant, plan).is_none()
        );

        let nonconstant = symbolic_two_case_root(&mut egraph, selector, shared, shared, at_one, 1);
        egraph.rebuild();
        let plan = pointwise_add_switch_cancellation_plan(&egraph, nonconstant)
            .expect("symbolic case is not directly cancellable in every case");
        assert!(build_pointwise_add_switch_cancellation(&mut egraph, nonconstant, plan).is_none());

        let insufficient =
            symbolic_two_case_root(&mut egraph, selector, shared, at_zero, at_one, 2);
        egraph.rebuild();
        let plan = pointwise_add_switch_cancellation_plan(&egraph, insufficient)
            .expect("insufficient multiplicity reaches the fallback");
        assert!(build_pointwise_add_switch_cancellation(&mut egraph, insufficient, plan).is_some());

        let excess_zero = egraph.add(MxxLang::MatrixAdd(vec![at_zero, at_zero].into_boxed_slice()));
        let excess_one = egraph.add(MxxLang::MatrixAdd(vec![at_one, at_one].into_boxed_slice()));
        let excess =
            symbolic_two_case_root(&mut egraph, selector, shared, excess_zero, excess_one, 1);
        egraph.rebuild();
        let plan = pointwise_add_switch_cancellation_plan(&egraph, excess)
            .expect("excess multiplicity still has one cancellable occurrence per case");
        let replacement = build_pointwise_add_switch_cancellation(&mut egraph, excess, plan)
            .expect("the excess occurrence is retained as the residual");
        let cases = switch_node(&egraph, replacement).expect("residual switch");
        assert_eq!(egraph.find(cases[1]), egraph.find(at_zero));
        assert_eq!(egraph.find(cases[2]), egraph.find(at_one));

        let alternate = test_binder_at(&mut egraph, 0, 1, 11);
        let alternate_selector = egraph.add(MxxLang::IntBinder(alternate));
        let competing = symbolic_two_case_root(&mut egraph, selector, shared, at_zero, at_one, 1);
        egraph.union(selector, alternate_selector);
        egraph.rebuild();
        assert!(pointwise_add_switch_cancellation_plan(&egraph, competing).is_none());
    }

    #[test]
    fn binder_build_diagnostic_captures_instantiated_actual_and_fixed_leaves() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let binder = test_binder(&mut egraph, 0, 1);
        let selector = egraph.add(MxxLang::IntBinder(binder));
        let shared = binder_matrix_atom(&mut egraph, selector, "diagnostic-instantiated");
        let zero = egraph.add(MxxLang::IntConst(0.into()));
        let one = egraph.add(MxxLang::IntConst(1.into()));
        let at_zero =
            family::instantiate_shared_element(&mut egraph, shared, binder, zero, &mut || {
                Ok::<(), ()>(())
            })
            .expect("test instantiation");
        let at_one =
            family::instantiate_shared_element(&mut egraph, shared, binder, one, &mut || {
                Ok::<(), ()>(())
            })
            .expect("test instantiation");
        let root = symbolic_two_case_root(&mut egraph, selector, shared, at_one, at_zero, 1);
        egraph.rebuild();
        let (selector, plan) = only_binder_plan(&egraph, root);
        let mut rejects = Vec::new();

        assert!(
            build_binder_aware_pointwise_add_switch_cancellation_with_diagnostic(
                &mut egraph,
                root,
                selector,
                plan,
                &mut |reject| rejects.push(reject.clone()),
            )
            .is_none()
        );

        assert_eq!(rejects.len(), 1);
        let reject = &rejects[0];
        assert_eq!(reject.case_index, 0);
        assert_eq!(reject.stage, BinderBuildRejectStage::NoExactCancellation);
        assert_eq!(
            reject.actual.as_ref(),
            &[BinderBuildSignedLeaf { eclass: usize::from(egraph.find(at_one)), negative: false }]
        );
        assert_eq!(
            reject.fixed.as_ref(),
            &[BinderBuildSignedLeaf { eclass: usize::from(egraph.find(at_zero)), negative: true }]
        );
    }

    #[test]
    fn binder_build_product_diagnostics_preserve_direct_order_and_classify_shapes() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (left, _) = matrix_atom(&mut egraph, "diagnostic-product-left", None);
        let (middle, _) = matrix_atom(&mut egraph, "diagnostic-product-middle", None);
        let (right, _) = matrix_atom(&mut egraph, "diagnostic-product-right", None);
        let direct = egraph.add(MxxLang::MatrixMultiply(vec![left, middle].into_boxed_slice()));
        let reordered = egraph.add(MxxLang::MatrixMultiply(vec![middle, left].into_boxed_slice()));
        let nested = egraph.add(MxxLang::MatrixMultiply(vec![direct, right].into_boxed_slice()));
        egraph.rebuild();

        assert_eq!(
            retained_product_spines(&egraph, &[left]),
            vec![RetainedProductSpine::Absent { leaf: usize::from(egraph.find(left)) }]
        );
        assert_eq!(
            retained_product_spines(&egraph, &[direct]),
            vec![RetainedProductSpine::Direct {
                leaf: usize::from(egraph.find(direct)),
                factors: vec![usize::from(egraph.find(left)), usize::from(egraph.find(middle))]
                    .into_boxed_slice(),
                factor_adds: vec![RetainedFactorAdd::Absent, RetainedFactorAdd::Absent]
                    .into_boxed_slice(),
                omitted: 0,
            }]
        );
        assert_eq!(
            retained_product_spines(&egraph, &[reordered]),
            vec![RetainedProductSpine::Direct {
                leaf: usize::from(egraph.find(reordered)),
                factors: vec![usize::from(egraph.find(middle)), usize::from(egraph.find(left))]
                    .into_boxed_slice(),
                factor_adds: vec![RetainedFactorAdd::Absent, RetainedFactorAdd::Absent]
                    .into_boxed_slice(),
                omitted: 0,
            }]
        );
        assert_eq!(
            retained_product_spines(&egraph, &[nested]),
            vec![RetainedProductSpine::Direct {
                leaf: usize::from(egraph.find(nested)),
                factors: vec![usize::from(egraph.find(direct)), usize::from(egraph.find(right))]
                    .into_boxed_slice(),
                factor_adds: vec![RetainedFactorAdd::Absent, RetainedFactorAdd::Absent]
                    .into_boxed_slice(),
                omitted: 0,
            }]
        );

        let alternate = egraph.add(MxxLang::MatrixMultiply(vec![right, left].into_boxed_slice()));
        egraph.union(direct, alternate);
        egraph.rebuild();
        assert_eq!(
            retained_product_spines(&egraph, &[direct]),
            vec![RetainedProductSpine::Ambiguous { leaf: usize::from(egraph.find(direct)) }]
        );
    }

    #[test]
    fn fixed_spine_diagnostic_records_canonical_shape_and_semantics() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (atom, _) =
            matrix_atom(&mut egraph, "fixed-spine-atom", Some(AtomicRelationRole::Preimage));
        let constant = regular_scalar_gadget(&mut egraph);
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let switch = egraph.add(MxxLang::Switch(vec![selector, atom, constant].into_boxed_slice()));
        egraph.rebuild();

        let (factors, omitted) = capped_peel_diagnostic_ids(&egraph, &[atom, constant, switch]);

        assert_eq!(omitted, 0);
        assert_eq!(factors.len(), 3);
        assert_eq!(factors[0].eclass, usize::from(egraph.find(atom)));
        assert_eq!(factors[0].matrix_shape, Some(scalar_matrix_type()));
        assert!(factors[0].semantic_nodes[0].contains("atom source_kind="));
        assert!(factors[0].semantic_nodes[0].contains("Preimage"));
        assert!(factors[1].semantic_nodes[0].contains("matrix_constant spec="));
        assert!(factors[2].semantic_nodes[0].contains("switch selector="));
        assert_eq!(factors[2].semantic_nodes_omitted, 0);
    }

    #[test]
    fn binder_build_product_diagnostics_classify_bounded_factor_adds() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (left, _) = matrix_atom(&mut egraph, "diagnostic-factor-add-left", None);
        let (first, _) = matrix_atom(&mut egraph, "diagnostic-factor-add-first", None);
        let (second, _) = matrix_atom(&mut egraph, "diagnostic-factor-add-second", None);
        let (third, _) = matrix_atom(&mut egraph, "diagnostic-factor-add-third", None);
        let unique_add = egraph.add(MxxLang::MatrixAdd(vec![first, second].into_boxed_slice()));
        let ambiguous_add = egraph.add(MxxLang::MatrixAdd(vec![second, third].into_boxed_slice()));
        let competing_add = egraph.add(MxxLang::MatrixAdd(vec![third, first].into_boxed_slice()));
        egraph.union(ambiguous_add, competing_add);
        let nested_factor =
            egraph.add(MxxLang::MatrixMultiply(vec![left, first].into_boxed_slice()));
        let product = egraph.add(MxxLang::MatrixMultiply(
            vec![nested_factor, unique_add, ambiguous_add].into_boxed_slice(),
        ));
        egraph.rebuild();

        assert_eq!(
            retained_product_spines(&egraph, &[product]),
            vec![RetainedProductSpine::Direct {
                leaf: usize::from(egraph.find(product)),
                factors: vec![
                    usize::from(egraph.find(nested_factor)),
                    usize::from(egraph.find(unique_add)),
                    usize::from(egraph.find(ambiguous_add)),
                ]
                .into_boxed_slice(),
                factor_adds: vec![
                    RetainedFactorAdd::Absent,
                    RetainedFactorAdd::Unique,
                    RetainedFactorAdd::Ambiguous,
                ]
                .into_boxed_slice(),
                omitted: 0,
            }]
        );
    }

    #[test]
    fn binder_build_diagnostics_bound_leaf_and_product_coordinates() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let leaves = (0..17)
            .map(|index| matrix_atom(&mut egraph, &format!("diagnostic-leaf-{index}"), None).0)
            .collect::<Vec<_>>();
        let signed = leaves.iter().copied().map(|leaf| (leaf, false)).collect::<Vec<_>>();
        let (summary, omitted) = summarize_binder_build_leaves(&signed);
        assert_eq!(summary.len(), 16);
        assert_eq!(omitted, 1);

        let product = egraph.add(MxxLang::MatrixMultiply(leaves[..9].to_vec().into_boxed_slice()));
        egraph.rebuild();
        assert_eq!(
            retained_product_spines(&egraph, &[product]),
            vec![RetainedProductSpine::Direct {
                leaf: usize::from(egraph.find(product)),
                factors: leaves[..8]
                    .iter()
                    .map(|factor| usize::from(egraph.find(*factor)))
                    .collect(),
                factor_adds: vec![RetainedFactorAdd::Absent; 8].into_boxed_slice(),
                omitted: 1,
            }]
        );
    }

    #[test]
    fn binder_build_flattened_product_views_preserve_association_and_order() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (first, _) = matrix_atom(&mut egraph, "flatten-first", None);
        let (second, _) = matrix_atom(&mut egraph, "flatten-second", None);
        let (third, _) = matrix_atom(&mut egraph, "flatten-third", None);
        let first_second =
            egraph.add(MxxLang::MatrixMultiply(vec![first, second].into_boxed_slice()));
        let left_associated =
            egraph.add(MxxLang::MatrixMultiply(vec![first_second, third].into_boxed_slice()));
        let second_third =
            egraph.add(MxxLang::MatrixMultiply(vec![second, third].into_boxed_slice()));
        let right_associated =
            egraph.add(MxxLang::MatrixMultiply(vec![first, second_third].into_boxed_slice()));
        let reordered =
            egraph.add(MxxLang::MatrixMultiply(vec![second, first, third].into_boxed_slice()));
        let shorter = egraph.add(MxxLang::MatrixMultiply(vec![first, second].into_boxed_slice()));
        egraph.rebuild();

        let views = retained_product_leaves(
            &egraph,
            &[left_associated, right_associated, reordered, shorter],
        );
        assert_eq!(views[0].status, ProductLeafStatus::Complete);
        assert_eq!(views[0].leaves, views[1].leaves, "association is flattened");
        assert_ne!(views[0].leaves, views[2].leaves, "factor order is retained");
        assert_ne!(views[0].leaves, views[3].leaves, "arity is retained");
    }

    #[test]
    fn binder_build_flattened_product_views_describe_generic_leaf_nodes() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let index = egraph.add(MxxLang::IntConst(7.into()));
        let atom = indexed_matrix_atom(&mut egraph, &[index], "view-atom");
        let constant_spec = egraph.analysis.symbols.matrix_constants.intern(
            super::super::identity::MatrixConstantSpec {
                matrix_type: scalar_matrix_type(),
                value: MatrixConstantValue::Identity,
            },
        );
        let constant = egraph.add(MxxLang::MatrixConstant(
            super::super::identity::MatrixConstantSpecId(constant_spec),
        ));
        let query = HashQuerySpecId(egraph.analysis.symbols.hash_queries.intern(HashQuerySpec {
            matrix_type: scalar_matrix_type(),
            tag_program: Box::new([]),
        }));
        let hash = egraph
            .add(MxxLang::HashPlain { query, arguments: vec![atom, constant].into_boxed_slice() });
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let switch = egraph.add(MxxLang::Switch(vec![selector, atom, constant].into_boxed_slice()));
        let other = egraph.add(MxxLang::MatrixAdd(vec![atom, constant].into_boxed_slice()));
        egraph.rebuild();

        let views = retained_product_leaves(&egraph, &[atom, constant, hash, switch, other]);
        assert!(matches!(
            views[0].leaves[0].nodes[0],
            LeafNodeDescriptor::Atom { source_kind: "protocol-input", indices_omitted: 0, .. }
        ));
        assert!(matches!(
            views[1].leaves[0].nodes[0],
            LeafNodeDescriptor::MatrixConstant { value_kind: "identity", .. }
        ));
        assert!(matches!(
            views[2].leaves[0].nodes[0],
            LeafNodeDescriptor::HashPlain { query: 0, arguments_omitted: 0, .. }
        ));
        assert!(matches!(
            views[3].leaves[0].nodes[0],
            LeafNodeDescriptor::Switch { selector: Some(_), cases_omitted: 0, .. }
        ));
        assert!(matches!(
            views[4].leaves[0].nodes[0],
            LeafNodeDescriptor::Other { operator_name: "matrix-add", children_omitted: 0, .. }
        ));
    }

    #[test]
    fn binder_build_flattened_product_views_report_ambiguous_cycle_and_truncation() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (first, _) = matrix_atom(&mut egraph, "flatten-status-first", None);
        let (second, _) = matrix_atom(&mut egraph, "flatten-status-second", None);
        let ambiguous = egraph.add(MxxLang::MatrixMultiply(vec![first, second].into_boxed_slice()));
        let alternate = egraph.add(MxxLang::MatrixMultiply(vec![second, first].into_boxed_slice()));
        egraph.union(ambiguous, alternate);
        egraph.rebuild();
        let (cycle_leaf, _) = matrix_atom(&mut egraph, "flatten-status-cycle", None);
        let cyclic = egraph.add(MxxLang::MatrixMultiply(vec![cycle_leaf].into_boxed_slice()));
        egraph.union(cyclic, cycle_leaf);
        let (trailing, _) = matrix_atom(&mut egraph, "flatten-status-trailing", None);
        let ambiguous_then_trailing =
            egraph.add(MxxLang::MatrixMultiply(vec![ambiguous, trailing].into_boxed_slice()));
        let cycle_then_trailing =
            egraph.add(MxxLang::MatrixMultiply(vec![cyclic, trailing].into_boxed_slice()));
        let wide_factors = (0..65)
            .map(|index| matrix_atom(&mut egraph, &format!("flatten-wide-{index}"), None).0)
            .collect::<Vec<_>>();
        let wide = egraph.add(MxxLang::MatrixMultiply(wide_factors.into_boxed_slice()));
        let (deep_leaf, _) = matrix_atom(&mut egraph, "flatten-deep", None);
        let deep = (0..33).fold(deep_leaf, |child, _| {
            egraph.add(MxxLang::MatrixMultiply(vec![child].into_boxed_slice()))
        });

        let views = retained_product_leaves(
            &egraph,
            &[ambiguous, cyclic, wide, deep, ambiguous_then_trailing, cycle_then_trailing],
        );
        assert!(matches!(views[0].status, ProductLeafStatus::Ambiguous { .. }));
        assert!(matches!(views[1].status, ProductLeafStatus::Cycle { .. }));
        assert_eq!(views[2].leaves.len(), 16);
        assert_eq!(views[2].leaf_omitted, ProductLeafOmission::AtLeast(1));
        assert_eq!(views[2].status, ProductLeafStatus::Truncated { pending: 49 });
        assert!(views[3].leaves.is_empty());
        assert!(matches!(views[3].status, ProductLeafStatus::Truncated { .. }));
        assert!(matches!(views[4].status, ProductLeafStatus::Ambiguous { .. }));
        assert_eq!(views[4].leaf_omitted, ProductLeafOmission::AtLeast(0));
        assert!(matches!(views[5].status, ProductLeafStatus::Cycle { .. }));
        assert_eq!(views[5].leaf_omitted, ProductLeafOmission::AtLeast(0));
    }

    #[test]
    fn binder_build_disabled_diagnostic_gate_does_not_traverse_rejected_roots() {
        let egraph = EGraph::new(MxxAnalysis::default());
        let mut checks = 0;
        let mut sink_called = false;
        report_binder_build_reject(
            &egraph,
            0,
            BinderBuildRejectStage::NoExactCancellation,
            &[(Id::from(usize::MAX), false)],
            &[],
            &mut || {
                checks += 1;
                false
            },
            &mut |_| sink_called = true,
        );
        assert_eq!(checks, 1);
        assert!(!sink_called);
    }

    #[test]
    fn binder_build_diagnostic_sink_does_not_construct_extra_nodes() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let binder = test_binder(&mut egraph, 0, 1);
        let selector = egraph.add(MxxLang::IntBinder(binder));
        let shared = binder_matrix_atom(&mut egraph, selector, "diagnostic-node-count");
        let zero = egraph.add(MxxLang::IntConst(0.into()));
        let one = egraph.add(MxxLang::IntConst(1.into()));
        let at_zero =
            family::instantiate_shared_element(&mut egraph, shared, binder, zero, &mut || {
                Ok::<(), ()>(())
            })
            .expect("test instantiation");
        let at_one =
            family::instantiate_shared_element(&mut egraph, shared, binder, one, &mut || {
                Ok::<(), ()>(())
            })
            .expect("test instantiation");
        let root = symbolic_two_case_root(&mut egraph, selector, shared, at_one, at_zero, 1);
        egraph.rebuild();
        let mut without_sink = egraph.clone();
        let mut with_sink = egraph;
        let (selector, plan) = only_binder_plan(&without_sink, root);
        assert!(
            build_binder_aware_pointwise_add_switch_cancellation(
                &mut without_sink,
                root,
                selector,
                plan,
                &mut || Ok(()),
            )
            .is_none()
        );
        let (selector, plan) = only_binder_plan(&with_sink, root);
        let mut rejects = Vec::new();
        assert!(
            build_binder_aware_pointwise_add_switch_cancellation_with_diagnostic(
                &mut with_sink,
                root,
                selector,
                plan,
                &mut |reject| rejects.push(reject.clone()),
            )
            .is_none()
        );
        assert_eq!(rejects.len(), 1);
        assert_eq!(with_sink.total_size(), without_sink.total_size());
        assert_eq!(with_sink.number_of_classes(), without_sink.number_of_classes());
    }

    #[test]
    fn binder_build_failure_event_is_single_and_includes_capped_product_fields() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let binder = test_binder(&mut egraph, 0, 1);
        let selector = egraph.add(MxxLang::IntBinder(binder));
        let shared = binder_matrix_atom(&mut egraph, selector, "diagnostic-event-shared");
        let zero = egraph.add(MxxLang::IntConst(0.into()));
        let at_zero =
            family::instantiate_shared_element(&mut egraph, shared, binder, zero, &mut || {
                Ok::<(), ()>(())
            })
            .expect("test instantiation");
        let factors = (0..9)
            .map(|index| matrix_atom(&mut egraph, &format!("diagnostic-event-{index}"), None).0)
            .collect::<Vec<_>>();
        let unmatched = egraph.add(MxxLang::MatrixMultiply(factors.into_boxed_slice()));
        let switch =
            egraph.add(MxxLang::Switch(vec![selector, at_zero, unmatched].into_boxed_slice()));
        let fixed = egraph.add(MxxLang::MatrixNegate([shared]));
        let root = egraph.add(MxxLang::MatrixAdd(vec![switch, fixed].into_boxed_slice()));
        egraph.rebuild();
        let (selector, plan) = only_binder_plan(&egraph, root);
        let capture = BinderFailureEventCapture::default();
        let subscriber = tracing_subscriber::registry().with(capture.clone());

        tracing::subscriber::with_default(subscriber, || {
            assert!(
                build_binder_aware_pointwise_add_switch_cancellation(
                    &mut egraph,
                    root,
                    selector,
                    plan,
                    &mut || Ok(()),
                )
                .is_none()
            );
        });

        let events = capture.0.lock().expect("event capture lock");
        assert_eq!(events.len(), 1, "only the first failed case is logged");
        assert_eq!(events[0].get("case_index").map(String::as_str), Some("1"));
        assert!(
            events[0]
                .get("actual_product_spines")
                .is_some_and(|value| value.contains("Direct") && value.contains("omitted: 1"))
        );
        assert!(
            events[0].get("fixed_product_spines").is_some_and(|value| value.contains("Absent"))
        );
    }

    #[test]
    fn production_binder_build_failure_logs_once_per_applier_attempt() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let binder = test_binder(&mut egraph, 0, 1);
        let selector = egraph.add(MxxLang::IntBinder(binder));
        let shared = binder_matrix_atom(&mut egraph, selector, "diagnostic-production-shared");
        let zero = egraph.add(MxxLang::IntConst(0.into()));
        let at_zero =
            family::instantiate_shared_element(&mut egraph, shared, binder, zero, &mut || {
                Ok::<(), ()>(())
            })
            .expect("test instantiation");
        let (unmatched, _) = matrix_atom(&mut egraph, "diagnostic-production-unmatched", None);
        let switch =
            egraph.add(MxxLang::Switch(vec![selector, at_zero, unmatched].into_boxed_slice()));
        let fixed = egraph.add(MxxLang::MatrixNegate([shared]));
        let root = egraph.add(MxxLang::MatrixAdd(vec![switch, fixed].into_boxed_slice()));
        egraph.rebuild();
        let capture = BinderFailureEventCapture::default();
        let subscriber = tracing_subscriber::registry().with(capture.clone());
        let context = RewriteContext::new(SharedRewriteBudget::new());

        tracing::subscriber::with_default(subscriber, || {
            for _ in 0..2 {
                let plan = pointwise_add_switch_cancellation_plan(&egraph, root)
                    .expect("binder-aware cancellation plan");
                assert!(
                    build_pointwise_add_switch_cancellation_with_context(
                        &mut egraph,
                        root,
                        plan,
                        &context,
                    )
                    .is_none()
                );
            }
        });

        assert_eq!(capture.0.lock().expect("event capture lock").len(), 2);
    }

    #[test]
    fn pointwise_binder_fallback_keeps_source_and_coordinate_lookalikes_uncancelled() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let binder = test_binder(&mut egraph, 0, 1);
        let selector = egraph.add(MxxLang::IntBinder(binder));
        let shared = binder_matrix_atom(&mut egraph, selector, "expected-source");
        let other_source = binder_matrix_atom(&mut egraph, selector, "other-source");
        let zero = egraph.add(MxxLang::IntConst(0.into()));
        let one = egraph.add(MxxLang::IntConst(1.into()));
        let expected_zero =
            family::instantiate_shared_element(&mut egraph, shared, binder, zero, &mut || {
                Ok::<(), ()>(())
            })
            .expect("test instantiation");
        let other_one =
            family::instantiate_shared_element(&mut egraph, other_source, binder, one, &mut || {
                Ok::<(), ()>(())
            })
            .expect("test instantiation");
        let source_root =
            symbolic_two_case_root(&mut egraph, selector, shared, expected_zero, other_one, 1);
        egraph.rebuild();
        let plan = pointwise_add_switch_cancellation_plan(&egraph, source_root)
            .expect("different source is not directly cancellable");
        assert!(build_pointwise_add_switch_cancellation(&mut egraph, source_root, plan).is_none());

        let coordinate = egraph.add(MxxLang::IntConst(7.into()));
        let indexed_shared =
            indexed_matrix_atom(&mut egraph, &[selector, coordinate], "coordinates");
        let expected_zero = family::instantiate_shared_element(
            &mut egraph,
            indexed_shared,
            binder,
            zero,
            &mut || Ok::<(), ()>(()),
        )
        .expect("test instantiation");
        let reordered_zero = indexed_matrix_atom(&mut egraph, &[coordinate, zero], "coordinates");
        let expected_one = family::instantiate_shared_element(
            &mut egraph,
            indexed_shared,
            binder,
            one,
            &mut || Ok::<(), ()>(()),
        )
        .expect("test instantiation");
        let coordinate_root = symbolic_two_case_root(
            &mut egraph,
            selector,
            indexed_shared,
            reordered_zero,
            expected_one,
            1,
        );
        egraph.rebuild();
        let plan = pointwise_add_switch_cancellation_plan(&egraph, coordinate_root)
            .expect("reordered coordinates are not directly cancellable");
        assert!(
            build_pointwise_add_switch_cancellation(&mut egraph, coordinate_root, plan).is_none()
        );
        assert_ne!(egraph.find(expected_zero), egraph.find(reordered_zero));
    }

    #[test]
    fn pointwise_binder_fallback_retains_unchanged_terms_and_cancels_them_locally() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let binder = test_binder(&mut egraph, 0, 1);
        let selector = egraph.add(MxxLang::IntBinder(binder));
        let shared = binder_matrix_atom(&mut egraph, selector, "indexed");
        let (unchanged, _) = matrix_atom(&mut egraph, "binder-free", None);
        let zero = egraph.add(MxxLang::IntConst(0.into()));
        let one = egraph.add(MxxLang::IntConst(1.into()));
        let at_zero =
            family::instantiate_shared_element(&mut egraph, shared, binder, zero, &mut || {
                Ok::<(), ()>(())
            })
            .expect("test instantiation");
        let at_one =
            family::instantiate_shared_element(&mut egraph, shared, binder, one, &mut || {
                Ok::<(), ()>(())
            })
            .expect("test instantiation");
        let first = egraph.add(MxxLang::MatrixAdd(vec![at_zero, unchanged].into_boxed_slice()));
        let second = egraph.add(MxxLang::MatrixAdd(vec![at_one, unchanged].into_boxed_slice()));
        let switch = egraph.add(MxxLang::Switch(vec![selector, first, second].into_boxed_slice()));
        let negative_shared = egraph.add(MxxLang::MatrixNegate([shared]));
        let negative_unchanged = egraph.add(MxxLang::MatrixNegate([unchanged]));
        let root = egraph.add(MxxLang::MatrixAdd(
            vec![switch, negative_shared, negative_unchanged].into_boxed_slice(),
        ));
        egraph.rebuild();

        let plan = pointwise_add_switch_cancellation_plan(&egraph, root)
            .expect("mixed per-case classification");
        assert!(plan.binder_aware.is_some());
        let replacement = build_pointwise_add_switch_cancellation(&mut egraph, root, plan)
            .expect("changed terms cancel across the case and unchanged terms locally");
        let cases = switch_node(&egraph, replacement).expect("normalized switch");
        assert!(cases[1..].iter().all(|case| {
            egraph[egraph.find(*case)].nodes.iter().any(|node| {
                matches!(node, MxxLang::MatrixConstant(spec)
                    if matches!(egraph.analysis.symbols.matrix_constants.get(spec.0), Some(super::super::identity::MatrixConstantSpec { value: MatrixConstantValue::Zero, .. })))
            })
        }));
    }

    #[test]
    fn pointwise_binder_distribution_preserves_interleaved_fixed_occurrence_order() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let binder = test_binder(&mut egraph, 0, 1);
        let selector = egraph.add(MxxLang::IntBinder(binder));
        let a = binder_matrix_atom(&mut egraph, selector, "a");
        let b = binder_matrix_atom(&mut egraph, selector, "b");
        let c = binder_matrix_atom(&mut egraph, selector, "c");
        let zero = egraph.add(MxxLang::IntConst(0.into()));
        let one = egraph.add(MxxLang::IntConst(1.into()));
        let instantiate =
            |egraph: &mut EGraph<MxxLang, MxxAnalysis>, base, index| {
                family::instantiate_shared_element(egraph, base, binder, index, &mut || {
                    Ok::<(), ()>(())
                })
                .expect("test instantiation")
            };
        let a_zero = instantiate(&mut egraph, a, zero);
        let b_zero = instantiate(&mut egraph, b, zero);
        let a_one = instantiate(&mut egraph, a, one);
        let b_one = instantiate(&mut egraph, b, one);
        let first = egraph.add(MxxLang::MatrixAdd(vec![a_zero, b_zero].into_boxed_slice()));
        let second = egraph.add(MxxLang::MatrixAdd(vec![a_one, b_one].into_boxed_slice()));
        let switch = egraph.add(MxxLang::Switch(vec![selector, first, second].into_boxed_slice()));
        let negative_c = egraph.add(MxxLang::MatrixNegate([c]));
        let root =
            egraph.add(MxxLang::MatrixAdd(vec![switch, a, b, a, negative_c].into_boxed_slice()));
        egraph.rebuild();
        let plan =
            pointwise_add_switch_cancellation_plan(&egraph, root).expect("distribution plan");
        assert!(build_pointwise_add_switch_cancellation(&mut egraph, root, plan).is_none());
    }

    #[test]
    fn pointwise_binder_distribution_cancels_distinct_bases_after_canonical_coalescing() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let binder = test_binder(&mut egraph, 0, 1);
        let selector = egraph.add(MxxLang::IntBinder(binder));
        let left = binder_matrix_atom(&mut egraph, selector, "left");
        let right = binder_matrix_atom(&mut egraph, selector, "right");
        let zero = egraph.add(MxxLang::IntConst(0.into()));
        let one = egraph.add(MxxLang::IntConst(1.into()));
        let left_zero =
            family::instantiate_shared_element(&mut egraph, left, binder, zero, &mut || {
                Ok::<(), ()>(())
            })
            .expect("test instantiation");
        let right_zero =
            family::instantiate_shared_element(&mut egraph, right, binder, zero, &mut || {
                Ok::<(), ()>(())
            })
            .expect("test instantiation");
        let left_one =
            family::instantiate_shared_element(&mut egraph, left, binder, one, &mut || {
                Ok::<(), ()>(())
            })
            .expect("test instantiation");
        let right_one =
            family::instantiate_shared_element(&mut egraph, right, binder, one, &mut || {
                Ok::<(), ()>(())
            })
            .expect("test instantiation");
        egraph.union(left_zero, right_zero);
        egraph.union(left_one, right_one);
        let zero_spec = egraph.analysis.symbols.matrix_constants.intern(
            super::super::identity::MatrixConstantSpec {
                matrix_type: scalar_matrix_type(),
                value: MatrixConstantValue::Zero,
            },
        );
        let first = egraph
            .add(MxxLang::MatrixConstant(super::super::identity::MatrixConstantSpecId(zero_spec)));
        let switch = egraph.add(MxxLang::Switch(vec![selector, first, first].into_boxed_slice()));
        let negative_right = egraph.add(MxxLang::MatrixNegate([right]));
        let root =
            egraph.add(MxxLang::MatrixAdd(vec![switch, left, negative_right].into_boxed_slice()));
        egraph.rebuild();
        let plan =
            pointwise_add_switch_cancellation_plan(&egraph, root).expect("distribution plan");
        let replacement = build_pointwise_add_switch_cancellation(&mut egraph, root, plan)
            .expect("coalesced cancellation");
        let cases = switch_node(&egraph, replacement).expect("replacement switch");
        assert!(cases[1..].iter().all(|case| {
            egraph[egraph.find(*case)].nodes.iter().any(|node| {
                matches!(node, MxxLang::MatrixConstant(spec)
                    if matches!(egraph.analysis.symbols.matrix_constants.get(spec.0), Some(super::super::identity::MatrixConstantSpec { value: MatrixConstantValue::Zero, .. }))) ||
                    matches!(node, MxxLang::MatrixAdd(children)
                        if children.len() == 1 && egraph[egraph.find(children[0])].nodes.iter().any(|child| matches!(child, MxxLang::MatrixConstant(spec)
                            if matches!(egraph.analysis.symbols.matrix_constants.get(spec.0), Some(super::super::identity::MatrixConstantSpec { value: MatrixConstantValue::Zero, .. })))))
            })
        }));
    }

    #[test]
    fn pointwise_binder_distribution_instantiates_descriptor_only_sampler_index() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let binder = test_binder(&mut egraph, 0, 1);
        let selector = egraph.add(MxxLang::IntBinder(binder));
        let sampler = egraph.analysis.symbols.samplers.intern(SamplerIdentity::UniformInterval {
            source: GraphWireSourceKey {
                wire: WireSourceKey {
                    scope: OccurrenceScope {
                        program: ProgramKey::Ideal,
                        definition: mxx_ir_core::FrozenGraphScopeId::Root,
                        path: Box::new([]),
                    },
                    wire: mxx_ir_core::WireRef {
                        node: mxx_ir_core::NodeId(42),
                        port: mxx_ir_core::Port(0),
                    },
                },
                coordinate_binders: Box::new([]),
            },
            indices: Box::new([selector]),
            minimum: ResolvedIntExpr::Const(0.into()),
            maximum: ResolvedIntExpr::Const(1.into()),
        });
        let source = egraph.analysis.symbols.atomic_sources.intern(AtomicSourceDescriptor {
            key: AtomicSourceKey::Sampler(SamplerDescriptorId(sampler)),
            sort: MxxSort::Matrix(scalar_matrix_type()),
            integer_domain: None,
            canonical_residue_convention: Some(CanonicalResidueConvention::Nonnegative),
            relation_role: None,
        });
        // The Atom itself has no index child; the only binder occurrence is in
        // the sampler descriptor retained by its source metadata.
        let shared =
            egraph.add(MxxLang::Atom { source: AtomicSourceId(source), indices: Box::new([]) });
        let zero = egraph.add(MxxLang::IntConst(0.into()));
        let one = egraph.add(MxxLang::IntConst(1.into()));
        let first =
            family::instantiate_shared_element(&mut egraph, shared, binder, zero, &mut || {
                Ok::<(), ()>(())
            })
            .expect("descriptor instantiation");
        let second =
            family::instantiate_shared_element(&mut egraph, shared, binder, one, &mut || {
                Ok::<(), ()>(())
            })
            .expect("descriptor instantiation");
        let (residual, _) = matrix_atom(&mut egraph, "descriptor-residual", None);
        let first_case = egraph.add(MxxLang::MatrixAdd(vec![first, residual].into_boxed_slice()));
        let second_case = egraph.add(MxxLang::MatrixAdd(vec![second, residual].into_boxed_slice()));
        let switch =
            egraph.add(MxxLang::Switch(vec![selector, first_case, second_case].into_boxed_slice()));
        let grouped_shared = egraph.add(MxxLang::MatrixAdd(vec![shared].into_boxed_slice()));
        let negated_grouped_shared = egraph.add(MxxLang::MatrixNegate([grouped_shared]));
        let root =
            egraph.add(MxxLang::MatrixAdd(vec![switch, negated_grouped_shared].into_boxed_slice()));
        egraph.rebuild();
        let plan = pointwise_add_switch_cancellation_plan(&egraph, root).expect("descriptor plan");
        let replacement = build_pointwise_add_switch_cancellation(&mut egraph, root, plan)
            .expect("descriptor-only binder remaps and cancels");
        let cases = switch_node(&egraph, replacement).expect("normalized switch");
        assert!(cases[1..].iter().all(|case| egraph.find(*case) == egraph.find(residual)));
    }

    #[test]
    fn pointwise_binder_candidates_try_a_later_physical_add_after_a_failed_candidate() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let binder = test_binder(&mut egraph, 0, 1);
        let selector = egraph.add(MxxLang::IntBinder(binder));
        let shared = binder_matrix_atom(&mut egraph, selector, "candidate-shared");
        let zero = egraph.add(MxxLang::IntConst(0.into()));
        let one = egraph.add(MxxLang::IntConst(1.into()));
        let at_zero =
            family::instantiate_shared_element(&mut egraph, shared, binder, zero, &mut || {
                Ok::<(), ()>(())
            })
            .expect("test instantiation");
        let at_one =
            family::instantiate_shared_element(&mut egraph, shared, binder, one, &mut || {
                Ok::<(), ()>(())
            })
            .expect("test instantiation");
        // This physical Add sorts first because its switch is created first,
        // but its stored cases disagree with the binder indices.
        let first = symbolic_two_case_root(&mut egraph, selector, shared, at_one, at_zero, 1);
        let second = symbolic_two_case_root(&mut egraph, selector, shared, at_zero, at_one, 1);
        egraph.union(first, second);
        egraph.rebuild();

        let mut plans = pointwise_add_switch_cancellation_plans(&egraph, first);
        assert_eq!(plans.len(), 2, "equivalent physical candidates are deduplicated by node");
        let mut probe = egraph.clone();
        assert!(
            build_pointwise_add_switch_cancellation(&mut probe, first, plans.remove(0)).is_none()
        );
        assert!(
            build_pointwise_add_switch_cancellation(&mut probe, first, plans.remove(0)).is_some()
        );

        let owned = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let context =
            RewriteContext::new(SharedRewriteBudget::from_shared(std::sync::Arc::clone(&owned)));
        let applier = RelationApplier::new(context.clone());
        assert!(
            !Applier::apply_one(
                &applier,
                &mut egraph,
                first,
                &Subst::default(),
                None,
                Symbol::from("pointwise-later-candidate"),
            )
            .is_empty()
        );
        assert_eq!(context.counters().candidates, 2);
        assert_eq!(context.counters().rewrites, 1);
        assert!(
            owned.load(std::sync::atomic::Ordering::Relaxed) >= 3,
            "the applier reservation and both candidate reservations are owned work"
        );
    }

    #[test]
    fn pointwise_zero_fixed_product_guard_avoids_a_parent_child_cycle() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let binder = test_binder(&mut egraph, 0, 1);
        let selector = egraph.add(MxxLang::IntBinder(binder));
        let shared = binder_matrix_atom(&mut egraph, selector, "zero-cycle-shared");
        let (first, _) = matrix_atom(&mut egraph, "zero-cycle-first", None);
        let (second, _) = matrix_atom(&mut egraph, "zero-cycle-second", None);
        let zero_spec = egraph.analysis.symbols.matrix_constants.intern(
            super::super::identity::MatrixConstantSpec {
                matrix_type: scalar_matrix_type(),
                value: MatrixConstantValue::Zero,
            },
        );
        let zero = egraph
            .add(MxxLang::MatrixConstant(super::super::identity::MatrixConstantSpecId(zero_spec)));
        let switch = egraph.add(MxxLang::Switch(vec![selector, first, second].into_boxed_slice()));
        let zero_product =
            egraph.add(MxxLang::MatrixMultiply(vec![shared, zero].into_boxed_slice()));
        let fixed = egraph.add(MxxLang::MatrixNegate([zero_product]));
        let root = egraph.add(MxxLang::MatrixAdd(vec![switch, fixed].into_boxed_slice()));
        egraph.rebuild();

        let context = RewriteContext::new(SharedRewriteBudget::new());
        let applier = RelationApplier::new(context);
        assert!(
            !Applier::apply_one(
                &applier,
                &mut egraph,
                root,
                &Subst::default(),
                None,
                Symbol::from("pointwise-zero-cycle"),
            )
            .is_empty()
        );
        egraph.rebuild();
        let root = egraph.find(root);
        assert_ne!(
            egraph.find(switch),
            root,
            "the original direct Switch child must not be unioned with its Add parent"
        );
        assert!(egraph[root].nodes.iter().any(|node| {
            let MxxLang::Switch(cases) = node else { return false };
            cases.get(1).is_some_and(|case| {
                egraph[egraph.find(*case)]
                    .nodes
                    .iter()
                    .any(|node| matches!(node, MxxLang::MatrixAdd(children) if children.len() == 1 && egraph.find(children[0]) == egraph.find(first)))
            })
        }));

        let after_first = egraph.total_size();
        assert!(
            Applier::apply_one(
                &applier,
                &mut egraph,
                root,
                &Subst::default(),
                None,
                Symbol::from("pointwise-zero-cycle-repeat"),
            )
            .is_empty()
        );
        egraph.rebuild();
        assert_eq!(egraph.total_size(), after_first, "a repeated apply cannot grow guarded nodes");
    }

    #[test]
    fn pointwise_binder_cancels_real_fixed_terms_alongside_zero_products() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let binder = test_binder(&mut egraph, 0, 1);
        let selector = egraph.add(MxxLang::IntBinder(binder));
        let shared = binder_matrix_atom(&mut egraph, selector, "mixed-zero-real-shared");
        let zero_spec = egraph.analysis.symbols.matrix_constants.intern(
            super::super::identity::MatrixConstantSpec {
                matrix_type: scalar_matrix_type(),
                value: MatrixConstantValue::Zero,
            },
        );
        let zero = egraph
            .add(MxxLang::MatrixConstant(super::super::identity::MatrixConstantSpecId(zero_spec)));
        let zero_product =
            egraph.add(MxxLang::MatrixMultiply(vec![shared, zero].into_boxed_slice()));
        let mut cases = vec![selector];
        let mut residuals = Vec::new();
        for case_index in 0..2 {
            let index = egraph.add(MxxLang::IntConst(BigInt::from(case_index)));
            let mapped =
                family::instantiate_shared_element(&mut egraph, shared, binder, index, &mut || {
                    Ok::<(), ()>(())
                })
                .expect("test instantiation");
            let (residual, _) =
                matrix_atom(&mut egraph, &format!("mixed-zero-real-residual-{case_index}"), None);
            residuals.push(residual);
            cases.push(egraph.add(MxxLang::MatrixAdd(vec![mapped, residual].into_boxed_slice())));
        }
        let switch = egraph.add(MxxLang::Switch(cases.into_boxed_slice()));
        let negative_shared = egraph.add(MxxLang::MatrixNegate([shared]));
        let negative_zero_product = egraph.add(MxxLang::MatrixNegate([zero_product]));
        let root = egraph.add(MxxLang::MatrixAdd(
            vec![switch, negative_shared, negative_zero_product].into_boxed_slice(),
        ));
        egraph.rebuild();

        let applier = RelationApplier::new(RewriteContext::new(SharedRewriteBudget::new()));
        assert!(
            !Applier::apply_one(
                &applier,
                &mut egraph,
                root,
                &Subst::default(),
                None,
                Symbol::from("pointwise-mixed-zero-real"),
            )
            .is_empty()
        );
        egraph.rebuild();
        let root = egraph.find(root);
        assert!(
            egraph[root].nodes.iter().any(|node| {
                let MxxLang::Switch(cases) = node else { return false };
                cases.len() == 3 &&
                    egraph.find(cases[1]) == egraph.find(residuals[0]) &&
                    egraph.find(cases[2]) == egraph.find(residuals[1])
            }),
            "the zero product is ignored while the real binder-owned term is cancelled"
        );
    }

    #[test]
    fn pointwise_zero_fixed_product_guard_grows_linearly_with_stored_cases() {
        fn guarded_growth(case_count: usize) -> usize {
            let mut egraph = EGraph::new(MxxAnalysis::default());
            let binder = test_binder(&mut egraph, 0, (case_count - 1) as i64);
            let selector = egraph.add(MxxLang::IntBinder(binder));
            let shared = binder_matrix_atom(&mut egraph, selector, "zero-cycle-linear-shared");
            let zero_spec = egraph.analysis.symbols.matrix_constants.intern(
                super::super::identity::MatrixConstantSpec {
                    matrix_type: scalar_matrix_type(),
                    value: MatrixConstantValue::Zero,
                },
            );
            let zero = egraph.add(MxxLang::MatrixConstant(
                super::super::identity::MatrixConstantSpecId(zero_spec),
            ));
            let mut cases = vec![selector];
            for case_index in 0..case_count {
                cases.push(
                    matrix_atom(
                        &mut egraph,
                        &format!("zero-cycle-linear-case-{case_count}-{case_index}"),
                        None,
                    )
                    .0,
                );
            }
            let switch = egraph.add(MxxLang::Switch(cases.into_boxed_slice()));
            let zero_product =
                egraph.add(MxxLang::MatrixMultiply(vec![shared, zero].into_boxed_slice()));
            let fixed = egraph.add(MxxLang::MatrixNegate([zero_product]));
            let root = egraph.add(MxxLang::MatrixAdd(vec![switch, fixed].into_boxed_slice()));
            egraph.rebuild();
            let before = egraph.total_size();

            let applier = RelationApplier::new(RewriteContext::new(SharedRewriteBudget::new()));
            assert!(
                !Applier::apply_one(
                    &applier,
                    &mut egraph,
                    root,
                    &Subst::default(),
                    None,
                    Symbol::from("pointwise-zero-cycle-linear"),
                )
                .is_empty()
            );
            egraph.rebuild();
            egraph.total_size() - before
        }

        let eight = guarded_growth(8);
        let sixteen = guarded_growth(16);
        assert!(
            sixteen <= eight.saturating_mul(3),
            "doubling stored cases must stay linear: 8={eight}, 16={sixteen}"
        );
    }

    #[test]
    fn pointwise_binder_candidates_do_not_union_when_every_physical_add_fails() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let binder = test_binder(&mut egraph, 0, 1);
        let selector = egraph.add(MxxLang::IntBinder(binder));
        let shared = binder_matrix_atom(&mut egraph, selector, "all-fail-shared");
        let zero = egraph.add(MxxLang::IntConst(0.into()));
        let one = egraph.add(MxxLang::IntConst(1.into()));
        let at_zero =
            family::instantiate_shared_element(&mut egraph, shared, binder, zero, &mut || {
                Ok::<(), ()>(())
            })
            .expect("test instantiation");
        let at_one =
            family::instantiate_shared_element(&mut egraph, shared, binder, one, &mut || {
                Ok::<(), ()>(())
            })
            .expect("test instantiation");
        let first = symbolic_two_case_root(&mut egraph, selector, shared, at_one, at_zero, 1);
        let wrapped_at_one = egraph.add(MxxLang::MatrixAdd(vec![at_one].into_boxed_slice()));
        let second =
            symbolic_two_case_root(&mut egraph, selector, shared, wrapped_at_one, at_zero, 1);
        egraph.union(first, second);
        egraph.rebuild();

        let owned = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let context =
            RewriteContext::new(SharedRewriteBudget::from_shared(std::sync::Arc::clone(&owned)));
        let applier = RelationApplier::new(context.clone());
        assert!(
            Applier::apply_one(
                &applier,
                &mut egraph,
                first,
                &Subst::default(),
                None,
                Symbol::from("pointwise-all-fail"),
            )
            .is_empty()
        );
        assert_eq!(context.counters().candidates, 2);
        assert_eq!(context.counters().rewrites, 0);
        assert!(
            owned.load(std::sync::atomic::Ordering::Relaxed) >= 3,
            "failed candidates still consume their explicit owned reservations"
        );
    }

    #[test]
    fn signed_additive_leaves_preserve_grouped_sign_order_and_multiplicity() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (left, _) = matrix_atom(&mut egraph, "signed-left", None);
        let (right, _) = matrix_atom(&mut egraph, "signed-right", None);
        let grouped = egraph.add(MxxLang::MatrixAdd(vec![left, right, left].into_boxed_slice()));
        let negated_group = egraph.add(MxxLang::MatrixNegate([grouped]));
        let double_negated = egraph.add(MxxLang::MatrixNegate([negated_group]));
        egraph.rebuild();

        assert_eq!(
            signed_additive_leaves(&egraph, &[negated_group]),
            Some(vec![
                (egraph.find(left), true),
                (egraph.find(right), true),
                (egraph.find(left), true),
            ])
        );
        assert_eq!(
            signed_additive_leaves(&egraph, &[double_negated]),
            Some(vec![
                (egraph.find(left), false),
                (egraph.find(right), false),
                (egraph.find(left), false),
            ])
        );

        let shared = egraph.add(MxxLang::MatrixAdd(vec![left, right].into_boxed_slice()));
        assert_eq!(
            signed_additive_leaves(&egraph, &[shared, shared]),
            Some(vec![
                (egraph.find(left), false),
                (egraph.find(right), false),
                (egraph.find(left), false),
                (egraph.find(right), false),
            ])
        );
    }

    #[test]
    fn signed_additive_leaves_accept_equivalent_competing_add_associations() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (first, _) = matrix_atom(&mut egraph, "association-first", None);
        let (second, _) = matrix_atom(&mut egraph, "association-second", None);
        let (third, _) = matrix_atom(&mut egraph, "association-third", None);
        let left = egraph.add(MxxLang::MatrixAdd(vec![first, second].into_boxed_slice()));
        let left_associated = egraph.add(MxxLang::MatrixAdd(vec![left, third].into_boxed_slice()));
        let right = egraph.add(MxxLang::MatrixAdd(vec![second, third].into_boxed_slice()));
        let right_associated =
            egraph.add(MxxLang::MatrixAdd(vec![first, right].into_boxed_slice()));
        egraph.union(left_associated, right_associated);
        egraph.rebuild();
        assert!(matches!(
            physical_add_terms(&egraph, left_associated),
            PhysicalStructure::Ambiguous
        ));
        let leaves = signed_additive_leaves(&egraph, &[left_associated])
            .expect("equivalent Add associations have one signed polynomial");
        assert_eq!(
            cancelled_signed_additive_leaves(
                &egraph,
                &leaves
                    .iter()
                    .copied()
                    .chain(leaves.iter().map(|(term, negative)| (*term, !negative)))
                    .collect::<Vec<_>>(),
            ),
            (vec![true; leaves.len() * 2], true),
        );
    }

    #[test]
    fn signed_additive_leaves_accept_equivalent_competing_negate_and_add_layouts() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (left, _) = matrix_atom(&mut egraph, "negate-add-left", None);
        let (right, _) = matrix_atom(&mut egraph, "negate-add-right", None);
        let sum = egraph.add(MxxLang::MatrixAdd(vec![left, right].into_boxed_slice()));
        let negated_sum = egraph.add(MxxLang::MatrixNegate([sum]));
        let negated_left = egraph.add(MxxLang::MatrixNegate([left]));
        let negated_right = egraph.add(MxxLang::MatrixNegate([right]));
        let expanded =
            egraph.add(MxxLang::MatrixAdd(vec![negated_left, negated_right].into_boxed_slice()));
        egraph.union(negated_sum, expanded);
        egraph.rebuild();
        assert!(signed_additive_leaves(&egraph, &[negated_sum]).is_some());
    }

    #[test]
    fn signed_additive_consensus_cancels_internal_opposites_before_comparison() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (cancelled, _) = matrix_atom(&mut egraph, "consensus-cancelled", None);
        let (retained, _) = matrix_atom(&mut egraph, "consensus-retained", None);
        let negated = egraph.add(MxxLang::MatrixNegate([cancelled]));
        let expanded =
            egraph.add(MxxLang::MatrixAdd(vec![cancelled, negated, retained].into_boxed_slice()));
        let direct = egraph.add(MxxLang::MatrixAdd(vec![retained].into_boxed_slice()));
        egraph.union(expanded, direct);
        egraph.rebuild();

        assert!(signed_additive_leaves(&egraph, &[expanded]).is_some());
    }

    #[test]
    fn signed_additive_consensus_visits_shared_equivalent_children_once() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let mut children = Vec::new();
        for index in 0..12 {
            let (left, _) = matrix_atom(&mut egraph, &format!("consensus-left-{index}"), None);
            let (right, _) = matrix_atom(&mut egraph, &format!("consensus-right-{index}"), None);
            let forward = egraph.add(MxxLang::MatrixAdd(vec![left, right].into_boxed_slice()));
            let reverse = egraph.add(MxxLang::MatrixAdd(vec![right, left].into_boxed_slice()));
            egraph.union(forward, reverse);
            children.push(forward);
        }
        let root = egraph.add(MxxLang::MatrixAdd(children.into_boxed_slice()));
        egraph.rebuild();

        let mut visits = Vec::new();
        assert!(
            signed_additive_leaves_with_visit(&egraph, &[root], |id| visits.push(id)).is_some()
        );
        let unique = visits.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(visits.len(), unique.len(), "each e-class is processed once per call");
        assert_eq!(visits.len(), 1 + 12 + 24, "root, children, and their atomic leaves");
    }

    #[test]
    fn signed_additive_leaves_reject_genuinely_different_competing_adds_and_cycles() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (left, _) = matrix_atom(&mut egraph, "different-add-left", None);
        let (right, _) = matrix_atom(&mut egraph, "different-add-right", None);
        let first_add = egraph.add(MxxLang::MatrixAdd(vec![left].into_boxed_slice()));
        let second_add = egraph.add(MxxLang::MatrixAdd(vec![right].into_boxed_slice()));
        egraph.union(first_add, second_add);
        egraph.rebuild();
        assert!(signed_additive_leaves(&egraph, &[first_add]).is_none());

        let cyclic = egraph.add(MxxLang::MatrixAdd(vec![left].into_boxed_slice()));
        egraph.union(cyclic, left);
        assert!(signed_additive_leaves(&egraph, &[cyclic]).is_none());
    }

    #[test]
    fn signed_ordered_monomials_distribute_products_and_preserve_order_and_sign() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (first, _) = matrix_atom(&mut egraph, "monomial-first", None);
        let (second, _) = matrix_atom(&mut egraph, "monomial-second", None);
        let (factor, _) = matrix_atom(&mut egraph, "monomial-factor", None);
        let sum = egraph.add(MxxLang::MatrixAdd(vec![first, second].into_boxed_slice()));
        let product = egraph.add(MxxLang::MatrixMultiply(vec![sum, factor].into_boxed_slice()));
        let negated = egraph.add(MxxLang::MatrixNegate([product]));
        let reordered = egraph.add(MxxLang::MatrixMultiply(vec![factor, sum].into_boxed_slice()));
        egraph.rebuild();

        let mut progress = || Ok(());
        assert_eq!(
            signed_ordered_monomial_spines(&egraph, &[(negated, false)], &mut progress),
            Some(vec![
                (vec![egraph.find(first), egraph.find(factor)].into_boxed_slice(), true),
                (vec![egraph.find(second), egraph.find(factor)].into_boxed_slice(), true),
            ])
        );
        let mut progress = || Ok(());
        assert_ne!(
            signed_ordered_monomial_spines(&egraph, &[(product, false)], &mut progress),
            signed_ordered_monomial_spines(&egraph, &[(reordered, false)], &mut progress),
            "matrix factor order is noncommutative"
        );

        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let switch = egraph.add(MxxLang::Switch(vec![selector, first, second].into_boxed_slice()));
        let switch_factor =
            egraph.add(MxxLang::MatrixMultiply(vec![sum, switch].into_boxed_slice()));
        let mut progress = || Ok(());
        assert_eq!(
            signed_ordered_monomial_spines(&egraph, &[(switch_factor, false)], &mut progress),
            Some(vec![
                (vec![egraph.find(first), egraph.find(switch)].into_boxed_slice(), false),
                (vec![egraph.find(second), egraph.find(switch)].into_boxed_slice(), false),
            ]),
            "Switch remains one opaque factor rather than enumerating its cases"
        );
    }

    #[test]
    fn selected_polynomial_expands_all_add_factors_and_cancels_interior_negation() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let a = egraph.add(MxxLang::IntConst(2.into()));
        let b = egraph.add(MxxLang::IntConst(3.into()));
        let c = egraph.add(MxxLang::IntConst(5.into()));
        let d = egraph.add(MxxLang::IntConst(7.into()));
        let add_left = egraph.add(MxxLang::MatrixAdd(vec![a, b].into_boxed_slice()));
        let add_right = egraph.add(MxxLang::MatrixAdd(vec![c, d].into_boxed_slice()));
        let root =
            egraph.add(MxxLang::MatrixMultiply(vec![add_left, add_right].into_boxed_slice()));
        let mut expression = RecExpr::default();
        let ae = expression.add(MxxLang::IntConst(2.into()));
        let be = expression.add(MxxLang::IntConst(3.into()));
        let ce = expression.add(MxxLang::IntConst(5.into()));
        let de = expression.add(MxxLang::IntConst(7.into()));
        let left = expression.add(MxxLang::MatrixAdd(vec![ae, be].into_boxed_slice()));
        let right = expression.add(MxxLang::MatrixAdd(vec![ce, de].into_boxed_slice()));
        let root_expression =
            expression.add(MxxLang::MatrixMultiply(vec![left, right].into_boxed_slice()));
        let origins = vec![a, b, c, d, add_left, add_right, root];
        let mut progress = || Ok(());
        let monomials =
            selected_polynomial_monomials(&egraph, &expression, &origins, &mut progress)
                .expect("selected RecExpr is topological");
        assert_eq!(monomials.len(), expression.as_ref().len(), "one local entry per selected node");
        let redex = selected_polynomial_redexes(
            &egraph,
            &expression,
            &origins,
            usize::from(root_expression),
            &monomials,
            &mut progress,
        )
        .expect("polynomial scan completes")
        .into_iter()
        .next()
        .expect("two Add factors expand together");
        assert!(matches!(redex.1, ReplacementPlan::Add(ref terms) if terms.len() == 4));

        let negated_d = egraph.add(MxxLang::MatrixNegate([d]));
        let cancelling = egraph.add(MxxLang::MatrixAdd(vec![c, negated_d, d].into_boxed_slice()));
        let product =
            egraph.add(MxxLang::MatrixMultiply(vec![add_left, cancelling].into_boxed_slice()));
        let mut interior = RecExpr::default();
        let ia = interior.add(MxxLang::IntConst(2.into()));
        let ib = interior.add(MxxLang::IntConst(3.into()));
        let ic = interior.add(MxxLang::IntConst(5.into()));
        let id = interior.add(MxxLang::IntConst(7.into()));
        let ineg = interior.add(MxxLang::MatrixNegate([id]));
        let isum = interior.add(MxxLang::MatrixAdd(vec![ia, ib].into_boxed_slice()));
        let inner = interior.add(MxxLang::MatrixAdd(vec![ic, ineg, id].into_boxed_slice()));
        let product_expression =
            interior.add(MxxLang::MatrixMultiply(vec![isum, inner].into_boxed_slice()));
        let interior_origins = vec![a, b, c, d, negated_d, add_left, cancelling, product];
        let mut progress = || Ok(());
        let interior_monomials =
            selected_polynomial_monomials(&egraph, &interior, &interior_origins, &mut progress)
                .expect("interior negate is selected");
        let redex = selected_polynomial_redexes(
            &egraph,
            &interior,
            &interior_origins,
            usize::from(product_expression),
            &interior_monomials,
            &mut progress,
        )
        .expect("polynomial scan completes")
        .into_iter()
        .next()
        .expect("interior cancellation leaves two ordered products");
        assert!(matches!(redex.1, ReplacementPlan::Add(ref terms) if terms.len() == 2));
    }

    #[test]
    fn selected_polynomial_cancels_lookup_signal_through_a_central_constant_scalar() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (secret, _) = matrix_atom(&mut egraph, "central-selected-secret", None);
        let gadget = regular_scalar_gadget(&mut egraph);
        let output_scalar = scalar_polynomial_constant(&mut egraph, &[9]);
        let relation_side = egraph
            .add(MxxLang::MatrixMultiply(vec![secret, gadget, output_scalar].into_boxed_slice()));
        let residual_side = egraph
            .add(MxxLang::MatrixMultiply(vec![output_scalar, secret, gadget].into_boxed_slice()));
        let negative_residual = egraph.add(MxxLang::MatrixNegate([residual_side]));
        let root = egraph
            .add(MxxLang::MatrixAdd(vec![relation_side, negative_residual].into_boxed_slice()));
        egraph.rebuild();

        let mut expression = RecExpr::default();
        let secret_expression = expression.add(MxxLang::IntConst(1.into()));
        let gadget_expression = expression.add(MxxLang::IntConst(2.into()));
        let scalar_expression = expression.add(MxxLang::IntConst(3.into()));
        let relation_expression = expression.add(MxxLang::MatrixMultiply(
            vec![secret_expression, gadget_expression, scalar_expression].into_boxed_slice(),
        ));
        let residual_expression = expression.add(MxxLang::MatrixMultiply(
            vec![scalar_expression, secret_expression, gadget_expression].into_boxed_slice(),
        ));
        let negative_expression = expression.add(MxxLang::MatrixNegate([residual_expression]));
        let root_expression = expression.add(MxxLang::MatrixAdd(
            vec![relation_expression, negative_expression].into_boxed_slice(),
        ));
        let origins =
            [secret, gadget, output_scalar, relation_side, residual_side, negative_residual, root];
        let mut progress = || Ok(());
        let monomials =
            selected_polynomial_monomials(&egraph, &expression, &origins, &mut progress)
                .expect("selected lookup signal evaluates");
        assert!(monomials[usize::from(root_expression)].is_empty());
        assert_eq!(
            monomials[usize::from(relation_expression)],
            vec![(
                vec![egraph.find(output_scalar), egraph.find(secret), egraph.find(gadget)]
                    .into_boxed_slice(),
                false,
            )]
        );
    }

    #[test]
    fn selected_polynomial_central_scalars_are_sorted_with_multiplicity_and_materialized() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (left, _) = matrix_atom(&mut egraph, "central-selected-left", None);
        let (right, _) = matrix_atom(&mut egraph, "central-selected-right", None);
        let first = scalar_polynomial_constant(&mut egraph, &[5]);
        let second = scalar_polynomial_constant(&mut egraph, &[7]);
        let product = egraph.add(MxxLang::MatrixMultiply(
            vec![left, second, first, first, right].into_boxed_slice(),
        ));
        egraph.rebuild();
        let mut expression = RecExpr::default();
        let left_expression = expression.add(MxxLang::IntConst(1.into()));
        let second_expression = expression.add(MxxLang::IntConst(2.into()));
        let first_expression = expression.add(MxxLang::IntConst(3.into()));
        let right_expression = expression.add(MxxLang::IntConst(4.into()));
        let product_expression = expression.add(MxxLang::MatrixMultiply(
            vec![
                left_expression,
                second_expression,
                first_expression,
                first_expression,
                right_expression,
            ]
            .into_boxed_slice(),
        ));
        let origins = [left, second, first, right, product];
        let mut progress = || Ok(());
        let monomials =
            selected_polynomial_monomials(&egraph, &expression, &origins, &mut progress)
                .expect("selected scalar product evaluates");
        let mut scalars = vec![egraph.find(second), egraph.find(first), egraph.find(first)];
        scalars.sort_unstable();
        let mut expected = scalars;
        expected.extend([egraph.find(left), egraph.find(right)]);
        assert_eq!(
            monomials[usize::from(product_expression)],
            vec![(expected.clone().into_boxed_slice(), false)]
        );
        let redex = selected_polynomial_redexes(
            &egraph,
            &expression,
            &origins,
            usize::from(product_expression),
            &monomials,
            &mut progress,
        )
        .expect("selected scan completes")
        .into_iter()
        .next()
        .expect("canonical product is a redex");
        assert!(matches!(redex.1, ReplacementPlan::Product(ref factors)
            if factors.iter().zip(expected).all(|(factor, expected)|
                matches!(factor, ReplacementPlan::Existing(actual) if *actual == expected))));
    }

    #[test]
    fn selected_polynomial_central_scalar_boundaries_fail_closed() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let zero = scalar_polynomial_constant(&mut egraph, &[0]);
        let nonconstant = scalar_polynomial_constant(&mut egraph, &[1, 1]);
        let gadget = regular_scalar_gadget(&mut egraph);
        let (atom, _) = matrix_atom(&mut egraph, "central-selected-atom", None);
        let unresolved_spec = egraph.analysis.symbols.matrix_constants.intern(MatrixConstantSpec {
            matrix_type: scalar_matrix_type(),
            value: MatrixConstantValue::Polynomial {
                coefficients: vec![
                    ResolvedIntExpr::Const(1.into()),
                    ResolvedIntExpr::Parameter("unresolved-central-coefficient".to_owned()),
                ]
                .into_boxed_slice(),
            },
        });
        let unresolved = egraph.add(MxxLang::MatrixConstant(MatrixConstantSpecId(unresolved_spec)));
        let missing = egraph.add(MxxLang::MatrixConstant(MatrixConstantSpecId(u32::MAX)));
        egraph.rebuild();
        let mut progress = || Ok(());
        assert_eq!(is_central_constant_scalar(&egraph, zero, &mut progress), Some(true));
        for term in [nonconstant, gadget, atom, unresolved, missing] {
            assert_eq!(is_central_constant_scalar(&egraph, term, &mut progress), Some(false));
        }
        let original = vec![(vec![atom, zero].into_boxed_slice(), false)];
        let mut interrupted = original.clone();
        let mut progress = || Err(());
        assert!(
            canonicalize_central_constant_scalar_spines(&egraph, &mut interrupted, &mut progress)
                .is_none()
        );
        assert_eq!(interrupted, original, "an interrupted canonicalization cannot commit");
    }

    #[test]
    fn selected_polynomial_keeps_product_order_distinct_and_evaluates_once_per_root() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let first = egraph.add(MxxLang::IntConst(2.into()));
        let second = egraph.add(MxxLang::IntConst(3.into()));
        let forward = egraph.add(MxxLang::MatrixMultiply(vec![first, second].into_boxed_slice()));
        let reverse = egraph.add(MxxLang::MatrixMultiply(vec![second, first].into_boxed_slice()));
        let expression_for = |left, right, root| {
            let mut expression = RecExpr::default();
            let left_expression = expression.add(MxxLang::IntConst(2.into()));
            let right_expression = expression.add(MxxLang::IntConst(3.into()));
            let root_expression = expression.add(MxxLang::MatrixMultiply(
                vec![left_expression, right_expression].into_boxed_slice(),
            ));
            (expression, vec![left, right, root], root_expression)
        };
        let (forward_expression, forward_origins, forward_root) =
            expression_for(first, second, forward);
        let (reverse_expression, reverse_origins, reverse_root) =
            expression_for(second, first, reverse);
        let mut evaluations = 0;
        evaluations += 1;
        let mut progress = || Ok(());
        let forward_monomials = selected_polynomial_monomials(
            &egraph,
            &forward_expression,
            &forward_origins,
            &mut progress,
        )
        .expect("one selected root evaluation");
        evaluations += 1;
        let mut progress = || Ok(());
        let reverse_monomials = selected_polynomial_monomials(
            &egraph,
            &reverse_expression,
            &reverse_origins,
            &mut progress,
        )
        .expect("one selected root evaluation");
        assert_eq!(evaluations, 2, "the evaluator runs once for each selected root");
        assert_ne!(
            forward_monomials[usize::from(forward_root)],
            reverse_monomials[usize::from(reverse_root)],
            "noncommutative factor order is part of the canonical spine"
        );
    }

    fn two_case_switch_hoist_plan(
        egraph: &EGraph<MxxLang, MxxAnalysis>,
        selector: Id,
        first_case: Id,
        second_case: Id,
        switch: Id,
        first_terms: Vec<(Box<[Id]>, bool)>,
        second_terms: Vec<(Box<[Id]>, bool)>,
    ) -> Option<ReplacementPlan> {
        let node = MxxLang::Switch(vec![Id::from(0), Id::from(1), Id::from(2)].into_boxed_slice());
        let monomials = vec![
            vec![(vec![selector].into_boxed_slice(), false)],
            first_terms,
            second_terms,
            vec![(vec![switch].into_boxed_slice(), false)],
        ];
        let mut progress = || Ok(());
        selected_switch_hoist_plan(
            egraph,
            &[selector, first_case, second_case, switch],
            3,
            &node,
            &monomials,
            &mut progress,
        )
        .map(|(_, plan)| plan)
    }

    fn selected_lookup_maps_for_test(
        egraph: &EGraph<MxxLang, MxxAnalysis>,
        expression: &RecExpr<MxxLang>,
        origins: &[Id],
    ) -> (BTreeMap<Id, Vec<usize>>, BTreeMap<Id, Option<usize>>) {
        let mut indices = BTreeMap::<Id, Vec<usize>>::new();
        let mut switches = BTreeMap::new();
        for (index, node) in expression.as_ref().iter().enumerate() {
            let origin = egraph.find(origins[index]);
            indices.entry(origin).or_default().push(index);
            if matches!(node, MxxLang::Switch(_)) {
                switches.entry(origin).and_modify(|stored| *stored = None).or_insert(Some(index));
            }
        }
        (indices, switches)
    }

    #[test]
    fn selected_switch_hoists_common_terms_and_whole_polynomial_factors() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let (common, _) = matrix_atom(&mut egraph, "hoist-common", None);
        let (prefix, _) = matrix_atom(&mut egraph, "hoist-prefix", None);
        let (suffix, _) = matrix_atom(&mut egraph, "hoist-suffix", None);
        let (left, _) = matrix_atom(&mut egraph, "hoist-left", None);
        let (right, _) = matrix_atom(&mut egraph, "hoist-right", None);
        let (first_case, _) = matrix_atom(&mut egraph, "hoist-first-case", None);
        let (second_case, _) = matrix_atom(&mut egraph, "hoist-second-case", None);
        let switch =
            egraph.add(MxxLang::Switch(vec![selector, first_case, second_case].into_boxed_slice()));
        egraph.rebuild();
        let mut expression = RecExpr::default();
        let selector_expression = expression.add(MxxLang::IntConst(0.into()));
        let first_expression = expression.add(MxxLang::IntConst(1.into()));
        let second_expression = expression.add(MxxLang::IntConst(2.into()));
        let switch_expression = expression.add(MxxLang::Switch(
            vec![selector_expression, first_expression, second_expression].into_boxed_slice(),
        ));
        let monomials = vec![
            vec![(vec![selector].into_boxed_slice(), false)],
            vec![
                (vec![prefix, left, suffix].into_boxed_slice(), false),
                (vec![common].into_boxed_slice(), false),
            ],
            vec![
                (vec![common].into_boxed_slice(), false),
                (vec![prefix, right, suffix].into_boxed_slice(), false),
            ],
            vec![(vec![switch].into_boxed_slice(), false)],
        ];
        let mut progress = || Ok(());
        let (_, plan) = selected_polynomial_redexes(
            &egraph,
            &expression,
            &[selector, first_case, second_case, switch],
            usize::from(switch_expression),
            &monomials,
            &mut progress,
        )
        .expect("polynomial scan completes")
        .into_iter()
        .next()
        .expect("strict Switch hoist");
        let ReplacementPlan::Add(terms) = plan else { panic!("common term stays outside") };
        assert!(matches!(terms[1], ReplacementPlan::Existing(id) if id == egraph.find(common)));
        let ReplacementPlan::Product(factors) = &terms[0] else { panic!("factored product") };
        assert!(matches!(factors[0], ReplacementPlan::Existing(id) if id == egraph.find(prefix)));
        assert!(matches!(factors[2], ReplacementPlan::Existing(id) if id == egraph.find(suffix)));
        let ReplacementPlan::Switch(cases) = &factors[1] else { panic!("one residual Switch") };
        assert!(matches!(cases[0], ReplacementPlan::Existing(id) if id == egraph.find(selector)));
        assert!(matches!(cases[1], ReplacementPlan::Existing(id) if id == egraph.find(left)));
        assert!(matches!(cases[2], ReplacementPlan::Existing(id) if id == egraph.find(right)));
    }

    #[test]
    fn selected_switch_hoist_boundaries_preserve_sign_order_scope_and_selector() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let invalid_binder = test_binder_at(&mut egraph, 0, 2, 777);
        let invalid_selector = egraph.add(MxxLang::IntBinder(invalid_binder));
        let atoms = (0..8)
            .map(|index| matrix_atom(&mut egraph, &format!("hoist-boundary-{index}"), None).0)
            .collect::<Vec<_>>();
        let [common, prefix, suffix, left, right, other, first_case, second_case] =
            atoms.as_slice()
        else {
            unreachable!()
        };
        let switch = egraph
            .add(MxxLang::Switch(vec![selector, *first_case, *second_case].into_boxed_slice()));
        let invalid_switch = egraph.add(MxxLang::Switch(
            vec![invalid_selector, *first_case, *second_case].into_boxed_slice(),
        ));
        egraph.rebuild();

        let signed = two_case_switch_hoist_plan(
            &egraph,
            selector,
            *first_case,
            *second_case,
            switch,
            vec![
                (vec![*common].into_boxed_slice(), true),
                (vec![*common].into_boxed_slice(), true),
                (vec![*left].into_boxed_slice(), false),
            ],
            vec![
                (vec![*common].into_boxed_slice(), true),
                (vec![*right].into_boxed_slice(), false),
            ],
        )
        .expect("minimum signed multiplicity is hoisted");
        assert!(matches!(signed, ReplacementPlan::Add(ref terms)
            if matches!(terms.last(), Some(ReplacementPlan::Negate(_)))));

        let prefix_only = two_case_switch_hoist_plan(
            &egraph,
            selector,
            *first_case,
            *second_case,
            switch,
            vec![(vec![*prefix, *left].into_boxed_slice(), false)],
            vec![(vec![*prefix, *right].into_boxed_slice(), false)],
        )
        .expect("prefix-only hoist");
        assert!(matches!(prefix_only, ReplacementPlan::Product(ref factors)
            if factors.len() == 2 && matches!(factors[0], ReplacementPlan::Existing(id) if id == egraph.find(*prefix))));
        let suffix_only = two_case_switch_hoist_plan(
            &egraph,
            selector,
            *first_case,
            *second_case,
            switch,
            vec![(vec![*left, *suffix].into_boxed_slice(), false)],
            vec![(vec![*right, *suffix].into_boxed_slice(), false)],
        )
        .expect("suffix-only hoist");
        assert!(matches!(suffix_only, ReplacementPlan::Product(ref factors)
            if factors.len() == 2 && matches!(factors[1], ReplacementPlan::Existing(id) if id == egraph.find(*suffix))));

        assert!(
            two_case_switch_hoist_plan(
                &egraph,
                selector,
                *first_case,
                *second_case,
                switch,
                vec![(vec![*left, *right].into_boxed_slice(), false)],
                vec![(vec![*right, *left].into_boxed_slice(), false)],
            )
            .is_none(),
            "noncommutative middle order is not rewritten"
        );
        assert!(
            two_case_switch_hoist_plan(
                &egraph,
                selector,
                *first_case,
                *second_case,
                switch,
                vec![
                    (vec![*prefix, *left].into_boxed_slice(), false),
                    (vec![*other].into_boxed_slice(), false),
                ],
                vec![
                    (vec![*prefix, *right].into_boxed_slice(), false),
                    (vec![*suffix].into_boxed_slice(), false),
                ],
            )
            .is_none(),
            "a factor shared by only a subset of monomials is not hoisted"
        );
        assert!(
            two_case_switch_hoist_plan(
                &egraph,
                invalid_selector,
                *first_case,
                *second_case,
                invalid_switch,
                vec![(vec![*prefix, *left].into_boxed_slice(), false)],
                vec![(vec![*prefix, *right].into_boxed_slice(), false)],
            )
            .is_none(),
            "an out-of-domain selector is unchanged"
        );
        let one_zero_residual = two_case_switch_hoist_plan(
            &egraph,
            selector,
            *first_case,
            *second_case,
            switch,
            vec![
                (vec![*common].into_boxed_slice(), false),
                (vec![*prefix, *left].into_boxed_slice(), false),
            ],
            vec![(vec![*common].into_boxed_slice(), false)],
        )
        .expect("common additive term is still hoisted");
        assert!(
            matches!(one_zero_residual, ReplacementPlan::Add(ref terms)
            if matches!(terms[0], ReplacementPlan::Switch(_))),
            "a full-output zero residual prevents unsound prefix factoring"
        );
    }

    #[test]
    fn selected_polynomial_keeps_distinct_selector_switches_separate() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let first_selector = egraph.add(MxxLang::IntConst(0.into()));
        let second_selector = egraph.add(MxxLang::IntConst(1.into()));
        let cases = (0..4)
            .map(|index| matrix_atom(&mut egraph, &format!("distinct-case-{index}"), None).0)
            .collect::<Vec<_>>();
        let first_switch = egraph
            .add(MxxLang::Switch(vec![first_selector, cases[0], cases[1]].into_boxed_slice()));
        let second_switch = egraph
            .add(MxxLang::Switch(vec![second_selector, cases[2], cases[3]].into_boxed_slice()));
        let root =
            egraph.add(MxxLang::MatrixAdd(vec![first_switch, second_switch].into_boxed_slice()));
        egraph.rebuild();
        let mut expression = RecExpr::default();
        let selector0 = expression.add(MxxLang::IntConst(0.into()));
        let case0 = expression.add(MxxLang::IntConst(2.into()));
        let case1 = expression.add(MxxLang::IntConst(3.into()));
        let switch0 = expression.add(MxxLang::Switch(vec![selector0, case0, case1].into()));
        let selector1 = expression.add(MxxLang::IntConst(1.into()));
        let case2 = expression.add(MxxLang::IntConst(5.into()));
        let case3 = expression.add(MxxLang::IntConst(7.into()));
        let switch1 = expression.add(MxxLang::Switch(vec![selector1, case2, case3].into()));
        let root_expression =
            expression.add(MxxLang::MatrixAdd(vec![switch0, switch1].into_boxed_slice()));
        let origins = [
            first_selector,
            cases[0],
            cases[1],
            first_switch,
            second_selector,
            cases[2],
            cases[3],
            second_switch,
            root,
        ];
        let mut progress = || Ok(());
        let monomials =
            selected_polynomial_monomials(&egraph, &expression, &origins, &mut progress)
                .expect("one evaluator invocation");
        assert_eq!(
            monomials[usize::from(root_expression)],
            vec![
                (vec![egraph.find(first_switch)].into_boxed_slice(), false),
                (vec![egraph.find(second_switch)].into_boxed_slice(), false),
            ]
        );
        assert!(
            selected_polynomial_redexes(
                &egraph,
                &expression,
                &origins,
                usize::from(root_expression),
                &monomials,
                &mut progress,
            )
            .is_some_and(|redexes| redexes.is_empty()),
            "the existing outer Add remains unchanged; no selector combination is planned"
        );
    }

    #[test]
    fn selected_full_case_keeps_a_foreign_selector_opaque() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let foreign_selector = egraph.add(MxxLang::IntConst(1.into()));
        let cases = (0..6)
            .map(|index| matrix_atom(&mut egraph, &format!("rejected-case-{index}"), None).0)
            .collect::<Vec<_>>();
        let first =
            egraph.add(MxxLang::Switch(vec![selector, cases[0], cases[1]].into_boxed_slice()));
        let second =
            egraph.add(MxxLang::Switch(vec![selector, cases[2], cases[3]].into_boxed_slice()));
        let foreign = egraph
            .add(MxxLang::Switch(vec![foreign_selector, cases[4], cases[5]].into_boxed_slice()));
        let root = egraph.add(MxxLang::MatrixAdd(vec![first, second, foreign].into_boxed_slice()));
        egraph.rebuild();

        let mut expression = RecExpr::default();
        let selector_expression = expression.add(MxxLang::IntConst(0.into()));
        let case0 = expression.add(MxxLang::IntConst(2.into()));
        let case1 = expression.add(MxxLang::IntConst(3.into()));
        let first_expression = expression
            .add(MxxLang::Switch(vec![selector_expression, case0, case1].into_boxed_slice()));
        let case2 = expression.add(MxxLang::IntConst(4.into()));
        let case3 = expression.add(MxxLang::IntConst(5.into()));
        let second_expression = expression
            .add(MxxLang::Switch(vec![selector_expression, case2, case3].into_boxed_slice()));
        let foreign_selector_expression = expression.add(MxxLang::IntConst(1.into()));
        let case4 = expression.add(MxxLang::IntConst(6.into()));
        let case5 = expression.add(MxxLang::IntConst(7.into()));
        let foreign_expression = expression.add(MxxLang::Switch(
            vec![foreign_selector_expression, case4, case5].into_boxed_slice(),
        ));
        let root_expression = expression.add(MxxLang::MatrixAdd(
            vec![first_expression, second_expression, foreign_expression].into_boxed_slice(),
        ));
        let origins = [
            selector,
            cases[0],
            cases[1],
            first,
            cases[2],
            cases[3],
            second,
            foreign_selector,
            cases[4],
            cases[5],
            foreign,
            root,
        ];
        let mut progress = || Ok(());
        let monomials =
            selected_polynomial_monomials(&egraph, &expression, &origins, &mut progress)
                .expect("selected polynomial");
        let (selected_indices, selected_switches) =
            selected_lookup_maps_for_test(&egraph, &expression, &origins);
        let plan = selected_same_selector_add_hoist_plan(
            &egraph,
            &expression,
            &origins,
            &selected_indices,
            &selected_switches,
            usize::from(root_expression),
            &monomials,
            &mut progress,
        );
        assert!(plan.is_some_and(|plan| plan.is_none()));
    }

    #[test]
    fn selected_polynomial_eliminates_equal_same_selector_case_sums_without_a_switch_merge() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let atoms = (0..6)
            .map(|index| matrix_atom(&mut egraph, &format!("cross-switch-zero-{index}"), None).0)
            .collect::<Vec<_>>();
        let first =
            egraph.add(MxxLang::Switch(vec![selector, atoms[0], atoms[1]].into_boxed_slice()));
        let first_product =
            egraph.add(MxxLang::MatrixMultiply(vec![atoms[5], first, atoms[4]].into_boxed_slice()));
        let second =
            egraph.add(MxxLang::Switch(vec![selector, atoms[2], atoms[3]].into_boxed_slice()));
        let first_case_product = egraph
            .add(MxxLang::MatrixMultiply(vec![atoms[5], atoms[0], atoms[4]].into_boxed_slice()));
        let second_case_product = egraph
            .add(MxxLang::MatrixMultiply(vec![atoms[5], atoms[1], atoms[4]].into_boxed_slice()));
        let competing_product =
            egraph.add(MxxLang::MatrixMultiply(vec![atoms[4], first, atoms[5]].into_boxed_slice()));
        let direct_switch = egraph.add(MxxLang::Switch(
            vec![selector, first_case_product, second_case_product].into_boxed_slice(),
        ));
        egraph.union(first_product, competing_product);
        egraph.union(first_product, direct_switch);
        let sums = [
            egraph.add(MxxLang::MatrixAdd(vec![first_case_product, atoms[2]].into_boxed_slice())),
            egraph.add(MxxLang::MatrixAdd(vec![second_case_product, atoms[3]].into_boxed_slice())),
        ];
        let third =
            egraph.add(MxxLang::Switch(vec![selector, sums[0], sums[1]].into_boxed_slice()));
        let negative_first = egraph.add(MxxLang::MatrixNegate([first_product]));
        let negative_second = egraph.add(MxxLang::MatrixNegate([second]));
        let inner = egraph.add(MxxLang::MatrixAdd(
            vec![negative_first, negative_second, third].into_boxed_slice(),
        ));
        let root = egraph.add(MxxLang::MatrixNegate([inner]));
        egraph.rebuild();

        let mut expression = RecExpr::default();
        let selector_expression = expression.add(MxxLang::IntConst(0.into()));
        let a0 = expression.add(MxxLang::IntConst(1.into()));
        let a1 = expression.add(MxxLang::IntConst(2.into()));
        let first_expression =
            expression.add(MxxLang::Switch(vec![selector_expression, a0, a1].into_boxed_slice()));
        let suffix_expression = expression.add(MxxLang::IntConst(8.into()));
        let prefix_expression = expression.add(MxxLang::IntConst(9.into()));
        let first_product_expression = expression.add(MxxLang::MatrixMultiply(
            vec![prefix_expression, first_expression, suffix_expression].into_boxed_slice(),
        ));
        let negative_first_expression =
            expression.add(MxxLang::MatrixNegate([first_product_expression]));
        let b0 = expression.add(MxxLang::IntConst(3.into()));
        let b1 = expression.add(MxxLang::IntConst(4.into()));
        let second_expression =
            expression.add(MxxLang::Switch(vec![selector_expression, b0, b1].into_boxed_slice()));
        let negative_second_expression = expression.add(MxxLang::MatrixNegate([second_expression]));
        let first_case_product_expression = expression.add(MxxLang::MatrixMultiply(
            vec![prefix_expression, a0, suffix_expression].into_boxed_slice(),
        ));
        let second_case_product_expression = expression.add(MxxLang::MatrixMultiply(
            vec![prefix_expression, a1, suffix_expression].into_boxed_slice(),
        ));
        let sum0 = expression
            .add(MxxLang::MatrixAdd(vec![first_case_product_expression, b0].into_boxed_slice()));
        let sum1 = expression
            .add(MxxLang::MatrixAdd(vec![second_case_product_expression, b1].into_boxed_slice()));
        let third_expression = expression
            .add(MxxLang::Switch(vec![selector_expression, sum0, sum1].into_boxed_slice()));
        let inner_expression = expression.add(MxxLang::MatrixAdd(
            vec![negative_first_expression, negative_second_expression, third_expression]
                .into_boxed_slice(),
        ));
        let root_expression = expression.add(MxxLang::MatrixNegate([inner_expression]));
        let origins = [
            selector,
            atoms[0],
            atoms[1],
            first,
            atoms[4],
            atoms[5],
            first_product,
            negative_first,
            atoms[2],
            atoms[3],
            second,
            negative_second,
            first_case_product,
            second_case_product,
            sums[0],
            sums[1],
            third,
            inner,
            root,
        ];
        let mut progress = || Ok(());
        let mut monomials =
            selected_polynomial_monomials(&egraph, &expression, &origins, &mut progress)
                .expect("selected cases evaluate");
        monomials[usize::from(negative_first_expression)] =
            vec![(vec![egraph.find(first_product)].into_boxed_slice(), true)];
        let redexes = selected_polynomial_redexes(
            &egraph,
            &expression,
            &origins,
            usize::from(root_expression),
            &monomials,
            &mut progress,
        )
        .expect("same-selector scan completes");
        assert_eq!(redexes.len(), 1);
        assert_eq!(redexes[0].0, egraph.find(inner), "the deep Add is scanned directly");
        fn contains_switch(plan: &ReplacementPlan) -> bool {
            match plan {
                ReplacementPlan::Switch(_) => true,
                ReplacementPlan::Product(children) |
                ReplacementPlan::Add(children) |
                ReplacementPlan::Equivalent(children) => children.iter().any(contains_switch),
                ReplacementPlan::Negate(child) => contains_switch(child),
                ReplacementPlan::Concat { inputs, .. } => inputs.iter().any(contains_switch),
                ReplacementPlan::Existing(_) => false,
            }
        }
        assert!(
            !contains_switch(&redexes[0].1),
            "the equality removes rather than merges Switches"
        );
        assert!(matches!(redexes[0].1, ReplacementPlan::Add(ref terms)
            if terms.len() == 1 && matches!(terms[0], ReplacementPlan::Add(ref zero) if zero.len() == 2)));
    }

    #[test]
    fn selected_polynomial_outer_add_owns_reassociated_same_selector_terms() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let atoms = (0..4)
            .map(|index| matrix_atom(&mut egraph, &format!("associated-switch-{index}"), None).0)
            .collect::<Vec<_>>();
        let first =
            egraph.add(MxxLang::Switch(vec![selector, atoms[0], atoms[1]].into_boxed_slice()));
        let second =
            egraph.add(MxxLang::Switch(vec![selector, atoms[2], atoms[3]].into_boxed_slice()));
        let sum0 = egraph.add(MxxLang::MatrixAdd(vec![atoms[0], atoms[2]].into_boxed_slice()));
        let sum1 = egraph.add(MxxLang::MatrixAdd(vec![atoms[1], atoms[3]].into_boxed_slice()));
        let combined = egraph.add(MxxLang::Switch(vec![selector, sum0, sum1].into_boxed_slice()));
        let negative_first = egraph.add(MxxLang::MatrixNegate([first]));
        let negative_second = egraph.add(MxxLang::MatrixNegate([second]));
        let inner =
            egraph.add(MxxLang::MatrixAdd(vec![negative_second, combined].into_boxed_slice()));
        let outer = egraph.add(MxxLang::MatrixAdd(vec![negative_first, inner].into_boxed_slice()));
        let unrelated =
            egraph.add(MxxLang::Switch(vec![selector, atoms[0], atoms[3]].into_boxed_slice()));
        let mismatching_parent =
            egraph.add(MxxLang::MatrixAdd(vec![outer, unrelated].into_boxed_slice()));
        egraph.rebuild();

        let mut expression = RecExpr::default();
        let selector_expression = expression.add(MxxLang::IntConst(0.into()));
        let a0 = expression.add(MxxLang::IntConst(1.into()));
        let a1 = expression.add(MxxLang::IntConst(2.into()));
        let first_expression =
            expression.add(MxxLang::Switch(vec![selector_expression, a0, a1].into_boxed_slice()));
        let negative_first_expression = expression.add(MxxLang::MatrixNegate([first_expression]));
        let b0 = expression.add(MxxLang::IntConst(3.into()));
        let b1 = expression.add(MxxLang::IntConst(4.into()));
        let second_expression =
            expression.add(MxxLang::Switch(vec![selector_expression, b0, b1].into_boxed_slice()));
        let negative_second_expression = expression.add(MxxLang::MatrixNegate([second_expression]));
        let sum0_expression = expression.add(MxxLang::MatrixAdd(vec![a0, b0].into_boxed_slice()));
        let sum1_expression = expression.add(MxxLang::MatrixAdd(vec![a1, b1].into_boxed_slice()));
        let combined_expression = expression.add(MxxLang::Switch(
            vec![selector_expression, sum0_expression, sum1_expression].into_boxed_slice(),
        ));
        let inner_expression = expression.add(MxxLang::MatrixAdd(
            vec![negative_second_expression, combined_expression].into_boxed_slice(),
        ));
        let outer_expression = expression.add(MxxLang::MatrixAdd(
            vec![negative_first_expression, inner_expression].into_boxed_slice(),
        ));
        let unrelated_expression =
            expression.add(MxxLang::Switch(vec![selector_expression, a0, b1].into_boxed_slice()));
        let parent_expression = expression.add(MxxLang::MatrixAdd(
            vec![outer_expression, unrelated_expression].into_boxed_slice(),
        ));
        let origins = [
            selector,
            atoms[0],
            atoms[1],
            first,
            negative_first,
            atoms[2],
            atoms[3],
            second,
            negative_second,
            sum0,
            sum1,
            combined,
            inner,
            outer,
            unrelated,
            mismatching_parent,
        ];
        let mut progress = || Ok(());
        let monomials =
            selected_polynomial_monomials(&egraph, &expression, &origins, &mut progress)
                .expect("associated polynomial memo");
        let redexes = selected_polynomial_redexes(
            &egraph,
            &expression,
            &origins,
            usize::from(parent_expression),
            &monomials,
            &mut progress,
        )
        .expect("associated Add scan");
        assert_eq!(
            redexes.len(),
            1,
            "the successful intermediate Add suppresses its mismatching parent"
        );
        assert_eq!(redexes[0].0, egraph.find(outer));
        fn contains_switch(plan: &ReplacementPlan) -> bool {
            match plan {
                ReplacementPlan::Switch(_) => true,
                ReplacementPlan::Product(children) |
                ReplacementPlan::Add(children) |
                ReplacementPlan::Equivalent(children) => children.iter().any(contains_switch),
                ReplacementPlan::Negate(child) => contains_switch(child),
                ReplacementPlan::Concat { inputs, .. } => inputs.iter().any(contains_switch),
                ReplacementPlan::Existing(_) => false,
            }
        }
        assert!(!contains_switch(&redexes[0].1));
    }

    #[test]
    fn selected_same_selector_case_sum_hoist_preserves_fixed_terms_and_fails_closed() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let selector_alias = egraph.add(MxxLang::IntConst(1.into()));
        egraph.union(selector, selector_alias);
        let atoms = (0..5)
            .map(|index| matrix_atom(&mut egraph, &format!("cross-switch-common-{index}"), None).0)
            .collect::<Vec<_>>();
        let first =
            egraph.add(MxxLang::Switch(vec![selector, atoms[0], atoms[1]].into_boxed_slice()));
        let second = egraph
            .add(MxxLang::Switch(vec![selector_alias, atoms[2], atoms[3]].into_boxed_slice()));
        let negative = egraph.add(MxxLang::MatrixNegate([first]));
        let root =
            egraph.add(MxxLang::MatrixAdd(vec![negative, second, atoms[4]].into_boxed_slice()));
        egraph.rebuild();

        let mut expression = RecExpr::default();
        let selector_expression = expression.add(MxxLang::IntConst(0.into()));
        let a0 = expression.add(MxxLang::IntConst(1.into()));
        let a1 = expression.add(MxxLang::IntConst(2.into()));
        let first_expression =
            expression.add(MxxLang::Switch(vec![selector_expression, a0, a1].into_boxed_slice()));
        let negative_expression = expression.add(MxxLang::MatrixNegate([first_expression]));
        let selector_alias_expression = expression.add(MxxLang::IntConst(3.into()));
        let b0 = expression.add(MxxLang::IntConst(4.into()));
        let b1 = expression.add(MxxLang::IntConst(5.into()));
        let second_expression = expression
            .add(MxxLang::Switch(vec![selector_alias_expression, b0, b1].into_boxed_slice()));
        let fixed_expression = expression.add(MxxLang::IntConst(6.into()));
        let root_expression = expression.add(MxxLang::MatrixAdd(
            vec![negative_expression, second_expression, fixed_expression].into_boxed_slice(),
        ));
        let origins = [
            selector,
            atoms[0],
            atoms[1],
            first,
            negative,
            selector_alias,
            atoms[2],
            atoms[3],
            second,
            atoms[4],
            root,
        ];
        let common = atoms[4];
        let monomials = vec![
            vec![(vec![selector].into(), false)],
            vec![(vec![atoms[0]].into(), false)],
            vec![(vec![atoms[1]].into(), false), (vec![common].into(), true)],
            vec![(vec![first].into(), false)],
            vec![(vec![first].into(), true)],
            vec![(vec![selector].into(), false)],
            vec![(vec![atoms[0]].into(), false), (vec![common].into(), false)],
            vec![(vec![atoms[1]].into(), false)],
            vec![(vec![second].into(), false)],
            vec![(vec![atoms[4]].into(), false)],
            vec![
                (vec![first].into(), true),
                (vec![second].into(), false),
                (vec![atoms[4]].into(), false),
            ],
        ];
        let (selected_indices, selected_switches) =
            selected_lookup_maps_for_test(&egraph, &expression, &origins);
        let mut progress = || Ok(());
        let plan = selected_same_selector_add_hoist_plan(
            &egraph,
            &expression,
            &origins,
            &selected_indices,
            &selected_switches,
            usize::from(root_expression),
            &monomials,
            &mut progress,
        )
        .expect("scan resources")
        .expect("case sums are selector independent");
        let ReplacementPlan::Add(terms) = plan else { panic!("one root Add") };
        assert_eq!(terms.len(), 1, "the complete case residual occurs exactly once");
        assert!(terms.iter().all(|term| !matches!(term, ReplacementPlan::Switch(_))));

        let case_origin = egraph.find(atoms[0]);
        let mut repeated_case_indices = selected_indices.clone();
        let repeated = *repeated_case_indices
            .get(&case_origin)
            .and_then(|indices| indices.first())
            .expect("stored case representative");
        repeated_case_indices.get_mut(&case_origin).expect("stored case").push(repeated);
        assert!(
            selected_same_selector_add_hoist_plan(
                &egraph,
                &expression,
                &origins,
                &repeated_case_indices,
                &selected_switches,
                usize::from(root_expression),
                &monomials,
                &mut progress,
            )
            .is_some_and(|plan| plan.is_none()),
            "repeated case origins remain fail-closed"
        );
        let mut missing_case_indices = selected_indices.clone();
        missing_case_indices.remove(&case_origin);
        assert!(
            selected_same_selector_add_hoist_plan(
                &egraph,
                &expression,
                &origins,
                &missing_case_indices,
                &selected_switches,
                usize::from(root_expression),
                &monomials,
                &mut progress,
            )
            .is_some_and(|plan| plan.is_none()),
            "missing case origins remain fail-closed"
        );

        let mut unequal = monomials.clone();
        unequal[usize::from(b1)].push((vec![atoms[2]].into(), false));
        let capture = FixedPeelEventCapture::default();
        let subscriber = tracing_subscriber::registry().with(capture.clone());
        let unequal_plan = tracing::subscriber::with_default(subscriber, || {
            selected_same_selector_add_hoist_plan(
                &egraph,
                &expression,
                &origins,
                &selected_indices,
                &selected_switches,
                usize::from(root_expression),
                &unequal,
                &mut progress,
            )
        });
        assert!(unequal_plan.is_some_and(|plan| plan.is_none()));
        let events = capture.0.lock().expect("event capture lock");
        assert_eq!(events.len(), 1, "only the first differing case is logged");
        assert_eq!(events[0].get("baseline_case").map(String::as_str), Some("0"));
        assert_eq!(events[0].get("differing_case").map(String::as_str), Some("1"));
        assert!(events[0].contains_key("contexts"));
        assert!(events[0].get("baseline_spines").is_some_and(|value| value.contains("false")));
        assert!(events[0].get("differing_spines").is_some_and(|value| value.contains("false")));
        assert!(
            events[0]
                .get("differing_factor_views")
                .is_some_and(|value| value.contains("source_kind: \"protocol-input\"") &&
                    value.contains("relation_role: None"))
        );
        assert!(events[0].contains_key("baseline_omitted"));
        assert!(events[0].contains_key("differing_omitted"));
        assert!(events[0].contains_key("differing_factor_views_omitted"));
        drop(events);
        let mut interrupted = || Err(());
        assert!(
            selected_same_selector_add_hoist_plan(
                &egraph,
                &expression,
                &origins,
                &selected_indices,
                &selected_switches,
                usize::from(root_expression),
                &monomials,
                &mut interrupted,
            )
            .is_none()
        );
    }

    #[test]
    fn selected_same_selector_full_case_uses_preimage_union_before_comparing_cases() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let a_lt = matrix_atom(&mut egraph, "full-case-a-lt", None).0;
        let mut rows = Vec::new();
        for case in 0..2 {
            let x_g_low = matrix_atom(&mut egraph, &format!("full-case-xg-low-{case}"), None).0;
            let a_z_low = matrix_atom(&mut egraph, &format!("full-case-az-low-{case}"), None).0;
            let y_g = matrix_atom(&mut egraph, &format!("full-case-yg-{case}"), None).0;
            let b_high = matrix_atom(&mut egraph, &format!("full-case-b-high-{case}"), None).0;
            let neg_a_z_low = egraph.add(MxxLang::MatrixNegate([a_z_low]));
            let neg_y_g = egraph.add(MxxLang::MatrixNegate([y_g]));
            let adjusted = egraph.add(MxxLang::MatrixAdd(
                vec![a_lt, x_g_low, neg_a_z_low, neg_y_g].into_boxed_slice(),
            ));
            egraph.union(b_high, adjusted);
            rows.push((x_g_low, a_z_low, y_g, b_high));
        }
        let x_switch =
            egraph.add(MxxLang::Switch(vec![selector, rows[0].0, rows[1].0].into_boxed_slice()));
        let az_switch =
            egraph.add(MxxLang::Switch(vec![selector, rows[0].1, rows[1].1].into_boxed_slice()));
        let y_switch =
            egraph.add(MxxLang::Switch(vec![selector, rows[0].2, rows[1].2].into_boxed_slice()));
        let high_switch =
            egraph.add(MxxLang::Switch(vec![selector, rows[0].3, rows[1].3].into_boxed_slice()));
        let neg_az_switch = egraph.add(MxxLang::MatrixNegate([az_switch]));
        let neg_y_switch = egraph.add(MxxLang::MatrixNegate([y_switch]));
        let neg_high_switch = egraph.add(MxxLang::MatrixNegate([high_switch]));
        let root = egraph.add(MxxLang::MatrixAdd(
            vec![a_lt, x_switch, neg_az_switch, neg_y_switch, neg_high_switch].into_boxed_slice(),
        ));
        egraph.rebuild();

        let mut expression = RecExpr::default();
        let selector_expression = expression.add(MxxLang::IntConst(0.into()));
        let a_expression = expression.add(MxxLang::IntConst(1.into()));
        let mut row_expressions = Vec::new();
        for offset in 0..2 {
            row_expressions.push((
                expression.add(MxxLang::IntConst((2 + offset * 4).into())),
                expression.add(MxxLang::IntConst((3 + offset * 4).into())),
                expression.add(MxxLang::IntConst((4 + offset * 4).into())),
                expression.add(MxxLang::IntConst((5 + offset * 4).into())),
            ));
        }
        let x_expression = expression.add(MxxLang::Switch(
            vec![selector_expression, row_expressions[0].0, row_expressions[1].0]
                .into_boxed_slice(),
        ));
        let az_expression = expression.add(MxxLang::Switch(
            vec![selector_expression, row_expressions[0].1, row_expressions[1].1]
                .into_boxed_slice(),
        ));
        let y_expression = expression.add(MxxLang::Switch(
            vec![selector_expression, row_expressions[0].2, row_expressions[1].2]
                .into_boxed_slice(),
        ));
        let high_expression = expression.add(MxxLang::Switch(
            vec![selector_expression, row_expressions[0].3, row_expressions[1].3]
                .into_boxed_slice(),
        ));
        let neg_az_expression = expression.add(MxxLang::MatrixNegate([az_expression]));
        let neg_y_expression = expression.add(MxxLang::MatrixNegate([y_expression]));
        let neg_high_expression = expression.add(MxxLang::MatrixNegate([high_expression]));
        let root_expression = expression.add(MxxLang::MatrixAdd(
            vec![
                a_expression,
                x_expression,
                neg_az_expression,
                neg_y_expression,
                neg_high_expression,
            ]
            .into_boxed_slice(),
        ));
        let origins = vec![
            selector,
            a_lt,
            rows[0].0,
            rows[0].1,
            rows[0].2,
            rows[0].3,
            rows[1].0,
            rows[1].1,
            rows[1].2,
            rows[1].3,
            x_switch,
            az_switch,
            y_switch,
            high_switch,
            neg_az_switch,
            neg_y_switch,
            neg_high_switch,
            root,
        ];
        let mut progress = || Ok(());
        let monomials =
            selected_polynomial_monomials(&egraph, &expression, &origins, &mut progress)
                .expect("full-case selected polynomial");
        let (selected_indices, selected_switches) =
            selected_lookup_maps_for_test(&egraph, &expression, &origins);
        let plan = selected_same_selector_add_hoist_plan(
            &egraph,
            &expression,
            &origins,
            &selected_indices,
            &selected_switches,
            usize::from(root_expression),
            &monomials,
            &mut progress,
        )
        .expect("full-case scan")
        .expect("every preimage case has the same zero residual");
        let ReplacementPlan::Add(terms) = plan else { panic!("typed zero uses an Add witness") };
        assert_eq!(terms.len(), 1, "the root retains one complete residual");
        let ReplacementPlan::Add(zero) = &terms[0] else { panic!("residual is typed zero") };
        assert_eq!(zero.len(), 2);
        assert!(matches!(zero[1], ReplacementPlan::Negate(_)));
    }

    #[test]
    fn selected_same_selector_ignores_unselected_physical_switch_alternatives() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let atoms = (0..4)
            .map(|index| matrix_atom(&mut egraph, &format!("ambiguous-direct-{index}"), None).0)
            .collect::<Vec<_>>();
        let inner =
            egraph.add(MxxLang::Switch(vec![selector, atoms[0], atoms[1]].into_boxed_slice()));
        let product = egraph.add(MxxLang::MatrixMultiply(vec![inner, atoms[2]].into_boxed_slice()));
        let case0 =
            egraph.add(MxxLang::MatrixMultiply(vec![atoms[0], atoms[2]].into_boxed_slice()));
        let case1 =
            egraph.add(MxxLang::MatrixMultiply(vec![atoms[1], atoms[2]].into_boxed_slice()));
        let direct = egraph.add(MxxLang::Switch(vec![selector, case0, case1].into_boxed_slice()));
        let incompatible =
            egraph.add(MxxLang::Switch(vec![selector, case0, atoms[3]].into_boxed_slice()));
        egraph.union(product, direct);
        egraph.union(product, incompatible);
        let negative = egraph.add(MxxLang::MatrixNegate([direct]));
        let root = egraph.add(MxxLang::MatrixAdd(vec![product, negative].into_boxed_slice()));
        egraph.rebuild();

        let mut expression = RecExpr::default();
        let selector_expression = expression.add(MxxLang::IntConst(0.into()));
        let a = expression.add(MxxLang::IntConst(1.into()));
        let b = expression.add(MxxLang::IntConst(2.into()));
        let inner_expression =
            expression.add(MxxLang::Switch(vec![selector_expression, a, b].into_boxed_slice()));
        let context_expression = expression.add(MxxLang::IntConst(3.into()));
        let product_expression = expression.add(MxxLang::MatrixMultiply(
            vec![inner_expression, context_expression].into_boxed_slice(),
        ));
        let case0_expression =
            expression.add(MxxLang::MatrixMultiply(vec![a, context_expression].into_boxed_slice()));
        let case1_expression =
            expression.add(MxxLang::MatrixMultiply(vec![b, context_expression].into_boxed_slice()));
        let direct_expression = expression.add(MxxLang::Switch(
            vec![selector_expression, case0_expression, case1_expression].into_boxed_slice(),
        ));
        let negative_expression = expression.add(MxxLang::MatrixNegate([direct_expression]));
        let root_expression = expression.add(MxxLang::MatrixAdd(
            vec![product_expression, negative_expression].into_boxed_slice(),
        ));
        let origins = [
            selector, atoms[0], atoms[1], inner, atoms[2], product, case0, case1, direct, negative,
            root,
        ];
        let mut progress = || Ok(());
        let monomials =
            selected_polynomial_monomials(&egraph, &expression, &origins, &mut progress)
                .expect("selected alternatives evaluate");
        let (selected_indices, selected_switches) =
            selected_lookup_maps_for_test(&egraph, &expression, &origins);
        assert!(
            selected_same_selector_add_hoist_plan(
                &egraph,
                &expression,
                &origins,
                &selected_indices,
                &selected_switches,
                usize::from(root_expression),
                &monomials,
                &mut progress,
            )
            .is_some_and(|plan| plan.is_some())
        );
    }

    #[test]
    fn selected_same_selector_rejects_two_selected_switch_factors_in_one_monomial() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let atoms = (0..4)
            .map(|index| matrix_atom(&mut egraph, &format!("two-switch-factors-{index}"), None).0)
            .collect::<Vec<_>>();
        let first =
            egraph.add(MxxLang::Switch(vec![selector, atoms[0], atoms[1]].into_boxed_slice()));
        let second =
            egraph.add(MxxLang::Switch(vec![selector, atoms[2], atoms[3]].into_boxed_slice()));
        let product = egraph.add(MxxLang::MatrixMultiply(vec![first, second].into_boxed_slice()));
        let negative = egraph.add(MxxLang::MatrixNegate([product]));
        let root = egraph.add(MxxLang::MatrixAdd(vec![product, negative].into_boxed_slice()));
        egraph.rebuild();

        let mut expression = RecExpr::default();
        let selector_expression = expression.add(MxxLang::IntConst(0.into()));
        let a0 = expression.add(MxxLang::IntConst(1.into()));
        let a1 = expression.add(MxxLang::IntConst(2.into()));
        let first_expression =
            expression.add(MxxLang::Switch(vec![selector_expression, a0, a1].into_boxed_slice()));
        let b0 = expression.add(MxxLang::IntConst(3.into()));
        let b1 = expression.add(MxxLang::IntConst(4.into()));
        let second_expression =
            expression.add(MxxLang::Switch(vec![selector_expression, b0, b1].into_boxed_slice()));
        let product_expression = expression.add(MxxLang::MatrixMultiply(
            vec![first_expression, second_expression].into_boxed_slice(),
        ));
        let negative_expression = expression.add(MxxLang::MatrixNegate([product_expression]));
        let root_expression = expression.add(MxxLang::MatrixAdd(
            vec![product_expression, negative_expression].into_boxed_slice(),
        ));
        let origins = [
            selector, atoms[0], atoms[1], first, atoms[2], atoms[3], second, product, negative,
            root,
        ];
        let mut progress = || Ok(());
        let monomials =
            selected_polynomial_monomials(&egraph, &expression, &origins, &mut progress)
                .expect("two-Switch polynomial");
        let (selected_indices, selected_switches) =
            selected_lookup_maps_for_test(&egraph, &expression, &origins);
        assert!(
            selected_same_selector_add_hoist_plan(
                &egraph,
                &expression,
                &origins,
                &selected_indices,
                &selected_switches,
                usize::from(root_expression),
                &monomials,
                &mut progress,
            )
            .is_some_and(|plan| plan.is_none())
        );
    }

    #[test]
    fn selected_switch_context_dependency_walks_atom_indices_and_fails_closed() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let (_, source) = matrix_atom(&mut egraph, "selector-dependent-context", None);
        let indexed =
            egraph.add(MxxLang::Atom { source, indices: vec![selector].into_boxed_slice() });
        let (independent, independent_source) =
            matrix_atom(&mut egraph, "selector-independent-context", None);
        egraph.union(indexed, independent);
        egraph.rebuild();
        let mut expression = RecExpr::default();
        let selector_expression = expression.add(MxxLang::IntConst(0.into()));
        let indexed_expression = expression
            .add(MxxLang::Atom { source, indices: vec![selector_expression].into_boxed_slice() });
        let independent_expression =
            expression.add(MxxLang::Atom { source: independent_source, indices: Box::default() });
        let origins = [selector, indexed, independent];
        let mut progress = || Ok(());
        assert_eq!(
            selected_subtree_contains_origin(
                &egraph,
                &expression,
                &origins,
                usize::from(indexed_expression),
                egraph.find(selector),
                &mut progress,
            ),
            Some(true)
        );
        assert_eq!(
            selected_context_has_independent_representative(
                &egraph,
                &expression,
                &origins,
                &[usize::from(indexed_expression), usize::from(independent_expression)],
                egraph.find(selector),
                &mut progress,
            ),
            Some(true),
            "a later independent equal representative witnesses context independence"
        );
        assert_eq!(
            selected_context_has_independent_representative(
                &egraph,
                &expression,
                &origins,
                &[usize::from(indexed_expression)],
                egraph.find(selector),
                &mut progress,
            ),
            Some(false),
            "all selector-dependent representatives reject"
        );
        let mut interrupted = || Err(());
        assert!(
            selected_subtree_contains_origin(
                &egraph,
                &expression,
                &origins,
                usize::from(indexed_expression),
                egraph.find(selector),
                &mut interrupted,
            )
            .is_none()
        );
        assert!(
            selected_context_has_independent_representative(
                &egraph,
                &expression,
                &origins,
                &[usize::from(indexed_expression), usize::from(independent_expression)],
                egraph.find(selector),
                &mut interrupted,
            )
            .is_none()
        );
    }

    #[test]
    fn selected_same_selector_case_sum_scan_is_linear_over_512_stored_cases() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let cases = (0..512)
            .map(|index| matrix_atom(&mut egraph, &format!("cross-switch-linear-{index}"), None).0)
            .collect::<Vec<_>>();
        let switch = egraph.add(MxxLang::Switch(
            std::iter::once(selector)
                .chain(cases.iter().copied())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ));
        let (context, _) = matrix_atom(&mut egraph, "cross-switch-linear-context", None);
        let product = egraph.add(MxxLang::MatrixMultiply(vec![switch, context].into_boxed_slice()));
        let negative = egraph.add(MxxLang::MatrixNegate([product]));
        let root = egraph.add(MxxLang::MatrixAdd(vec![product, negative].into_boxed_slice()));
        egraph.rebuild();

        let mut expression = RecExpr::default();
        let selector_expression = expression.add(MxxLang::IntConst(0.into()));
        let case_expressions = (0..512)
            .map(|index| expression.add(MxxLang::IntConst(BigInt::from(index + 1))))
            .collect::<Vec<_>>();
        let switch_expression = expression.add(MxxLang::Switch(
            std::iter::once(selector_expression)
                .chain(case_expressions.iter().copied())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ));
        let context_expression = expression.add(MxxLang::IntConst(BigInt::from(513)));
        let _repeated_context_expression = expression.add(MxxLang::IntConst(BigInt::from(514)));
        let product_expression = expression.add(MxxLang::MatrixMultiply(
            vec![switch_expression, context_expression].into_boxed_slice(),
        ));
        let negative_expression = expression.add(MxxLang::MatrixNegate([product_expression]));
        let root_expression = expression.add(MxxLang::MatrixAdd(
            vec![product_expression, negative_expression].into_boxed_slice(),
        ));
        let mut origins = Vec::with_capacity(517);
        origins.push(selector);
        origins.extend(cases.iter().copied());
        origins.extend([switch, context, context, product, negative, root]);
        let mut monomials = Vec::with_capacity(origins.len());
        monomials.push(vec![(vec![selector].into(), false)]);
        monomials.extend(cases.iter().map(|case| vec![(vec![*case].into(), false)]));
        monomials.push(vec![(vec![switch].into(), false)]);
        monomials.push(vec![(vec![context].into(), false)]);
        monomials.push(vec![(vec![context].into(), false)]);
        monomials.push(vec![(vec![switch, context].into(), false)]);
        monomials.push(vec![(vec![switch, context].into(), true)]);
        monomials.push(vec![
            (vec![switch, context].into(), false),
            (vec![switch, context].into(), true),
        ]);
        let (selected_indices, selected_switches) =
            selected_lookup_maps_for_test(&egraph, &expression, &origins);
        let mut visits = 0usize;
        let plan = selected_same_selector_add_hoist_plan(
            &egraph,
            &expression,
            &origins,
            &selected_indices,
            &selected_switches,
            usize::from(root_expression),
            &monomials,
            &mut || {
                visits += 1;
                Ok(())
            },
        )
        .expect("linear scan resources")
        .expect("opposite switches cancel");
        assert!(matches!(plan, ReplacementPlan::Add(_)));
        assert!(visits < 20_000, "stored-case work stays linear and bounded: {visits}");
    }

    #[test]
    fn selected_same_selector_accepts_rectangular_switch_factor_context() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let prefix = matrix_atom_with_type(
            &mut egraph,
            "rectangular-prefix",
            concrete_matrix_type(17, 8, 2, 3),
            None,
        )
        .0;
        let cases = (0..2)
            .map(|index| {
                matrix_atom_with_type(
                    &mut egraph,
                    &format!("rectangular-case-{index}"),
                    concrete_matrix_type(17, 8, 3, 4),
                    None,
                )
                .0
            })
            .collect::<Vec<_>>();
        let suffix = matrix_atom_with_type(
            &mut egraph,
            "rectangular-suffix",
            concrete_matrix_type(17, 8, 4, 5),
            None,
        )
        .0;
        let switch = egraph.add(MxxLang::Switch(
            std::iter::once(selector)
                .chain(cases.iter().copied())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ));
        let product =
            egraph.add(MxxLang::MatrixMultiply(vec![prefix, switch, suffix].into_boxed_slice()));
        let negative = egraph.add(MxxLang::MatrixNegate([product]));
        let root = egraph.add(MxxLang::MatrixAdd(vec![product, negative].into_boxed_slice()));
        egraph.rebuild();
        assert_ne!(egraph[switch].data.sort, egraph[root].data.sort);

        let mut expression = RecExpr::default();
        let selector_expression = expression.add(MxxLang::IntConst(0.into()));
        let case_expressions = [
            expression.add(MxxLang::IntConst(1.into())),
            expression.add(MxxLang::IntConst(2.into())),
        ];
        let switch_expression = expression.add(MxxLang::Switch(
            std::iter::once(selector_expression)
                .chain(case_expressions)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ));
        let prefix_expression = expression.add(MxxLang::IntConst(3.into()));
        let suffix_expression = expression.add(MxxLang::IntConst(4.into()));
        let product_expression = expression.add(MxxLang::MatrixMultiply(
            vec![prefix_expression, switch_expression, suffix_expression].into_boxed_slice(),
        ));
        let negative_expression = expression.add(MxxLang::MatrixNegate([product_expression]));
        let root_expression = expression.add(MxxLang::MatrixAdd(
            vec![product_expression, negative_expression].into_boxed_slice(),
        ));
        let origins =
            [selector, cases[0], cases[1], switch, prefix, suffix, product, negative, root];
        let mut progress = || Ok(());
        let mut monomials =
            selected_polynomial_monomials(&egraph, &expression, &origins, &mut progress)
                .expect("rectangular selected polynomial");
        monomials[usize::from(root_expression)] = vec![
            (vec![prefix, switch, suffix].into_boxed_slice(), false),
            (vec![prefix, switch, suffix].into_boxed_slice(), true),
        ];
        let (selected_indices, selected_switches) =
            selected_lookup_maps_for_test(&egraph, &expression, &origins);
        assert!(
            selected_same_selector_add_hoist_plan(
                &egraph,
                &expression,
                &origins,
                &selected_indices,
                &selected_switches,
                usize::from(root_expression),
                &monomials,
                &mut progress,
            )
            .is_some_and(|plan| plan.is_some()),
            "same-sorted Switch cases remain valid inside a differently shaped product"
        );
    }

    #[test]
    fn selected_same_selector_rejects_cases_that_match_root_instead_of_switch() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let prefix = matrix_atom_with_type(
            &mut egraph,
            "wrong-case-prefix",
            concrete_matrix_type(17, 8, 2, 3),
            None,
        )
        .0;
        let switch_cases = (0..2)
            .map(|index| {
                matrix_atom_with_type(
                    &mut egraph,
                    &format!("right-switch-case-{index}"),
                    concrete_matrix_type(17, 8, 3, 5),
                    None,
                )
                .0
            })
            .collect::<Vec<_>>();
        let root_shaped_cases = (0..2)
            .map(|index| {
                matrix_atom_with_type(
                    &mut egraph,
                    &format!("wrong-root-shaped-case-{index}"),
                    concrete_matrix_type(17, 8, 2, 5),
                    None,
                )
                .0
            })
            .collect::<Vec<_>>();
        let switch = egraph.add(MxxLang::Switch(
            std::iter::once(selector).chain(switch_cases).collect::<Vec<_>>().into_boxed_slice(),
        ));
        let product = egraph.add(MxxLang::MatrixMultiply(vec![prefix, switch].into_boxed_slice()));
        let negative = egraph.add(MxxLang::MatrixNegate([product]));
        let root = egraph.add(MxxLang::MatrixAdd(vec![product, negative].into_boxed_slice()));
        egraph.rebuild();
        assert_eq!(egraph[root_shaped_cases[0]].data.sort, egraph[root].data.sort);

        let mut expression = RecExpr::default();
        let selector_expression = expression.add(MxxLang::IntConst(0.into()));
        let first_case = expression.add(MxxLang::IntConst(1.into()));
        let second_case = expression.add(MxxLang::IntConst(2.into()));
        let switch_expression = expression.add(MxxLang::Switch(
            vec![selector_expression, first_case, second_case].into_boxed_slice(),
        ));
        let prefix_expression = expression.add(MxxLang::IntConst(3.into()));
        let product_expression = expression.add(MxxLang::MatrixMultiply(
            vec![prefix_expression, switch_expression].into_boxed_slice(),
        ));
        let negative_expression = expression.add(MxxLang::MatrixNegate([product_expression]));
        let root_expression = expression.add(MxxLang::MatrixAdd(
            vec![product_expression, negative_expression].into_boxed_slice(),
        ));
        let origins = [
            selector,
            root_shaped_cases[0],
            root_shaped_cases[1],
            switch,
            prefix,
            product,
            negative,
            root,
        ];
        let mut progress = || Ok(());
        let monomials =
            selected_polynomial_monomials(&egraph, &expression, &origins, &mut progress)
                .expect("selected polynomial with inconsistent case origins");
        let (selected_indices, selected_switches) =
            selected_lookup_maps_for_test(&egraph, &expression, &origins);
        assert!(
            selected_same_selector_add_hoist_plan(
                &egraph,
                &expression,
                &origins,
                &selected_indices,
                &selected_switches,
                usize::from(root_expression),
                &monomials,
                &mut progress,
            )
            .is_some_and(|plan| plan.is_none()),
            "a root-shaped case cannot replace a differently shaped Switch factor"
        );
    }

    #[test]
    fn selected_same_selector_rejects_case_modulus_and_ring_mismatches() {
        for (label, mismatched_type) in [
            ("modulus", concrete_matrix_type(19, 8, 3, 4)),
            ("ring", concrete_matrix_type(17, 16, 3, 4)),
        ] {
            let mut egraph = EGraph::new(MxxAnalysis::default());
            let selector = egraph.add(MxxLang::IntConst(0.into()));
            let switch_cases = (0..2)
                .map(|index| {
                    matrix_atom_with_type(
                        &mut egraph,
                        &format!("{label}-switch-case-{index}"),
                        concrete_matrix_type(17, 8, 3, 4),
                        None,
                    )
                    .0
                })
                .collect::<Vec<_>>();
            let selected_cases = (0..2)
                .map(|index| {
                    matrix_atom_with_type(
                        &mut egraph,
                        &format!("{label}-selected-case-{index}"),
                        mismatched_type.clone(),
                        None,
                    )
                    .0
                })
                .collect::<Vec<_>>();
            let switch = egraph.add(MxxLang::Switch(
                std::iter::once(selector)
                    .chain(switch_cases)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            ));
            let negative = egraph.add(MxxLang::MatrixNegate([switch]));
            let root = egraph.add(MxxLang::MatrixAdd(vec![switch, negative].into_boxed_slice()));
            egraph.rebuild();

            let mut expression = RecExpr::default();
            let selector_expression = expression.add(MxxLang::IntConst(0.into()));
            let first_case = expression.add(MxxLang::IntConst(1.into()));
            let second_case = expression.add(MxxLang::IntConst(2.into()));
            let switch_expression = expression.add(MxxLang::Switch(
                vec![selector_expression, first_case, second_case].into_boxed_slice(),
            ));
            let negative_expression = expression.add(MxxLang::MatrixNegate([switch_expression]));
            let root_expression = expression.add(MxxLang::MatrixAdd(
                vec![switch_expression, negative_expression].into_boxed_slice(),
            ));
            let origins = [selector, selected_cases[0], selected_cases[1], switch, negative, root];
            let mut progress = || Ok(());
            let monomials =
                selected_polynomial_monomials(&egraph, &expression, &origins, &mut progress)
                    .expect("selected polynomial with mismatched case contract");
            let (selected_indices, selected_switches) =
                selected_lookup_maps_for_test(&egraph, &expression, &origins);
            assert!(
                selected_same_selector_add_hoist_plan(
                    &egraph,
                    &expression,
                    &origins,
                    &selected_indices,
                    &selected_switches,
                    usize::from(root_expression),
                    &monomials,
                    &mut progress,
                )
                .is_some_and(|plan| plan.is_none()),
                "{label} mismatch must reject"
            );
        }
    }

    #[test]
    fn selected_same_selector_accepts_resolved_equivalent_case_sorts() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let affine_rows = ResolvedIntExpr::Add(
            Box::new(ResolvedIntExpr::Const(1.into())),
            Box::new(ResolvedIntExpr::Const(2.into())),
        );
        let affine_case = matrix_atom_with_type(
            &mut egraph,
            "resolved-equivalent-affine-case",
            ResolvedMatrixType {
                modulus: ResolvedIntExpr::Const(17.into()),
                ring_dimension: ResolvedIntExpr::Const(8.into()),
                rows: affine_rows,
                columns: ResolvedIntExpr::Const(4.into()),
            },
            None,
        )
        .0;
        let literal_case = matrix_atom_with_type(
            &mut egraph,
            "resolved-equivalent-literal-case",
            concrete_matrix_type(17, 8, 3, 4),
            None,
        )
        .0;
        let switch = egraph
            .add(MxxLang::Switch(vec![selector, affine_case, literal_case].into_boxed_slice()));
        let negative = egraph.add(MxxLang::MatrixNegate([switch]));
        let root = egraph.add(MxxLang::MatrixAdd(vec![switch, negative].into_boxed_slice()));
        egraph.rebuild();

        let mut expression = RecExpr::default();
        let selector_expression = expression.add(MxxLang::IntConst(0.into()));
        let first_case = expression.add(MxxLang::IntConst(1.into()));
        let second_case = expression.add(MxxLang::IntConst(2.into()));
        let switch_expression = expression.add(MxxLang::Switch(
            vec![selector_expression, first_case, second_case].into_boxed_slice(),
        ));
        let negative_expression = expression.add(MxxLang::MatrixNegate([switch_expression]));
        let root_expression = expression.add(MxxLang::MatrixAdd(
            vec![switch_expression, negative_expression].into_boxed_slice(),
        ));
        let origins = [selector, affine_case, literal_case, switch, negative, root];
        let mut progress = || Ok(());
        let mut monomials =
            selected_polynomial_monomials(&egraph, &expression, &origins, &mut progress)
                .expect("resolved-equivalent selected polynomial");
        monomials[usize::from(root_expression)] =
            vec![(vec![switch].into_boxed_slice(), false), (vec![switch].into_boxed_slice(), true)];
        let (selected_indices, selected_switches) =
            selected_lookup_maps_for_test(&egraph, &expression, &origins);
        assert!(
            selected_same_selector_add_hoist_plan(
                &egraph,
                &expression,
                &origins,
                &selected_indices,
                &selected_switches,
                usize::from(root_expression),
                &monomials,
                &mut progress,
            )
            .is_some_and(|plan| plan.is_some()),
            "semantic matrix-type equality accepts resolved-equivalent dimensions"
        );
    }

    #[test]
    fn selected_polynomial_batches_sibling_switches_and_defers_the_root() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let atoms = (0..6)
            .map(|index| matrix_atom(&mut egraph, &format!("batch-sibling-{index}"), None).0)
            .collect::<Vec<_>>();
        let first_switch =
            egraph.add(MxxLang::Switch(vec![selector, atoms[0], atoms[1]].into_boxed_slice()));
        let second_switch =
            egraph.add(MxxLang::Switch(vec![selector, atoms[2], atoms[3]].into_boxed_slice()));
        let root =
            egraph.add(MxxLang::MatrixAdd(vec![first_switch, second_switch].into_boxed_slice()));
        egraph.rebuild();

        let mut expression = RecExpr::default();
        let selector_expression = expression.add(MxxLang::IntConst(0.into()));
        let first_left = expression.add(MxxLang::IntConst(1.into()));
        let first_right = expression.add(MxxLang::IntConst(2.into()));
        let first = expression.add(MxxLang::Switch(
            vec![selector_expression, first_left, first_right].into_boxed_slice(),
        ));
        let second_left = expression.add(MxxLang::IntConst(3.into()));
        let second_right = expression.add(MxxLang::IntConst(4.into()));
        let second = expression.add(MxxLang::Switch(
            vec![selector_expression, second_left, second_right].into_boxed_slice(),
        ));
        let root_expression =
            expression.add(MxxLang::MatrixAdd(vec![first, second].into_boxed_slice()));
        let origins =
            [selector, atoms[0], atoms[1], first_switch, atoms[2], atoms[3], second_switch, root];
        let monomials = vec![
            vec![(vec![selector].into(), false)],
            vec![(vec![atoms[4], atoms[0]].into(), false)],
            vec![(vec![atoms[4], atoms[1]].into(), false)],
            vec![(vec![first_switch].into(), false)],
            vec![(vec![atoms[5], atoms[2]].into(), false)],
            vec![(vec![atoms[5], atoms[3]].into(), false)],
            vec![(vec![second_switch].into(), false)],
            vec![(vec![first_switch].into(), false), (vec![second_switch].into(), false)],
        ];
        let mut progress = || Ok(());
        let redexes = selected_polynomial_redexes(
            &egraph,
            &expression,
            &origins,
            usize::from(root_expression),
            &monomials,
            &mut progress,
        )
        .expect("batch scan completes");
        assert_eq!(redexes.len(), 2);
        assert_eq!(redexes[0].0, egraph.find(first_switch));
        assert_eq!(redexes[1].0, egraph.find(second_switch));
        assert!(redexes.iter().all(|(origin, _)| *origin != egraph.find(root)));
    }

    #[test]
    fn selected_polynomial_batches_nested_switches_and_deduplicates_origins() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let atoms = (0..5)
            .map(|index| matrix_atom(&mut egraph, &format!("batch-nested-{index}"), None).0)
            .collect::<Vec<_>>();
        let inner_left =
            egraph.add(MxxLang::MatrixMultiply(vec![atoms[0], atoms[1]].into_boxed_slice()));
        let inner_right =
            egraph.add(MxxLang::MatrixMultiply(vec![atoms[0], atoms[2]].into_boxed_slice()));
        let inner =
            egraph.add(MxxLang::Switch(vec![selector, inner_left, inner_right].into_boxed_slice()));
        let outer_left = egraph.add(MxxLang::MatrixAdd(vec![atoms[3], inner].into_boxed_slice()));
        let outer_right =
            egraph.add(MxxLang::MatrixAdd(vec![atoms[3], atoms[4]].into_boxed_slice()));
        let outer =
            egraph.add(MxxLang::Switch(vec![selector, outer_left, outer_right].into_boxed_slice()));
        egraph.rebuild();

        let mut expression = RecExpr::default();
        let selector_expression = expression.add(MxxLang::IntConst(0.into()));
        let inner_prefix_expression = expression.add(MxxLang::IntConst(10.into()));
        let inner_left_atom = expression.add(MxxLang::IntConst(11.into()));
        let inner_left_expression = expression.add(MxxLang::MatrixMultiply(
            vec![inner_prefix_expression, inner_left_atom].into_boxed_slice(),
        ));
        let inner_right_atom = expression.add(MxxLang::IntConst(12.into()));
        let inner_right_expression = expression.add(MxxLang::MatrixMultiply(
            vec![inner_prefix_expression, inner_right_atom].into_boxed_slice(),
        ));
        let inner_expression = expression.add(MxxLang::Switch(
            vec![selector_expression, inner_left_expression, inner_right_expression]
                .into_boxed_slice(),
        ));
        let duplicate_inner_expression = expression.add(MxxLang::Switch(
            vec![selector_expression, inner_left_expression, inner_right_expression]
                .into_boxed_slice(),
        ));
        let common_expression = expression.add(MxxLang::IntConst(13.into()));
        let outer_right_atom = expression.add(MxxLang::IntConst(14.into()));
        let outer_left_expression = expression.add(MxxLang::MatrixAdd(
            vec![common_expression, duplicate_inner_expression].into_boxed_slice(),
        ));
        let outer_right_expression = expression
            .add(MxxLang::MatrixAdd(vec![common_expression, outer_right_atom].into_boxed_slice()));
        let outer_expression = expression.add(MxxLang::Switch(
            vec![selector_expression, outer_left_expression, outer_right_expression]
                .into_boxed_slice(),
        ));
        let origins = [
            selector,
            atoms[0],
            atoms[1],
            inner_left,
            atoms[2],
            inner_right,
            inner,
            inner,
            atoms[3],
            atoms[4],
            outer_left,
            outer_right,
            outer,
        ];
        let mut progress = || Ok(());
        let monomials =
            selected_polynomial_monomials(&egraph, &expression, &origins, &mut progress)
                .expect("nested selected snapshot evaluates");
        let redexes = selected_polynomial_redexes(
            &egraph,
            &expression,
            &origins,
            usize::from(outer_expression),
            &monomials,
            &mut progress,
        )
        .expect("nested batch scan completes");
        assert_eq!(redexes.len(), 2, "the duplicate inner origin contributes one plan");
        assert_eq!(redexes[0].0, egraph.find(inner));
        assert_eq!(redexes[1].0, egraph.find(outer));

        let context = RewriteContext::new(SharedRewriteBudget::new());
        let pending = redexes
            .into_iter()
            .map(|redex| {
                materialize_selected_polynomial_redex(&mut egraph, redex, &context)
                    .expect("snapshot-local plan materializes")
            })
            .collect::<Vec<_>>();
        for (origin, replacement) in &pending {
            assert!(egraph.union(*origin, *replacement));
        }
        egraph.rebuild();
        assert!(
            pending
                .iter()
                .all(|(origin, replacement)| egraph.find(*origin) == egraph.find(*replacement)),
            "every independently materialized snapshot equality is unioned"
        );
        let _ = inner_expression;
    }

    #[test]
    fn mapped_fixed_product_consensus_flattens_associations_and_ignores_switch_witnesses() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (y, _) = matrix_atom(&mut egraph, "mapped-consensus-y", None);
        let (s, _) = matrix_atom(&mut egraph, "mapped-consensus-s", None);
        let (r, _) = matrix_atom(&mut egraph, "mapped-consensus-r", None);
        let (g, _) = matrix_atom(&mut egraph, "mapped-consensus-g", None);
        let left_prefix = egraph.add(MxxLang::MatrixMultiply(vec![y, s].into_boxed_slice()));
        let left = egraph.add(MxxLang::MatrixMultiply(vec![left_prefix, r, g].into_boxed_slice()));
        let right_suffix = egraph.add(MxxLang::MatrixMultiply(vec![s, r, g].into_boxed_slice()));
        let right = egraph.add(MxxLang::MatrixMultiply(vec![y, right_suffix].into_boxed_slice()));
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let switch = egraph.add(MxxLang::Switch(vec![selector, y, s].into_boxed_slice()));
        egraph.union(left, right);
        egraph.union(left, switch);
        egraph.rebuild();

        let mut progress = || Ok(());
        assert_eq!(
            mapped_fixed_product_consensus_with_progress(&egraph, left, &mut progress),
            Some(Some(
                vec![egraph.find(y), egraph.find(s), egraph.find(r), egraph.find(g)]
                    .into_boxed_slice()
            ))
        );
    }

    #[test]
    fn mapped_fixed_product_consensus_rejects_reordering_empty_roots_and_budget_cutoff() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (y, _) = matrix_atom(&mut egraph, "mapped-reject-y", None);
        let (s, _) = matrix_atom(&mut egraph, "mapped-reject-s", None);
        let (r, _) = matrix_atom(&mut egraph, "mapped-reject-r", None);
        let forward = egraph.add(MxxLang::MatrixMultiply(vec![y, s, r].into_boxed_slice()));
        let reordered = egraph.add(MxxLang::MatrixMultiply(vec![r, s, y].into_boxed_slice()));
        egraph.union(forward, reordered);
        egraph.rebuild();
        let mut progress = || Ok(());
        assert!(
            mapped_fixed_product_consensus_with_progress(&egraph, forward, &mut progress).is_none()
        );

        let empty = egraph.add(MxxLang::MatrixMultiply(Box::default()));
        let mut progress = || Ok(());
        assert!(
            mapped_fixed_product_consensus_with_progress(&egraph, empty, &mut progress).is_none(),
            "an empty direct root witness is fail-closed"
        );
        let nested_empty = egraph.add(MxxLang::MatrixMultiply(Box::default()));
        let nested_empty_root =
            egraph.add(MxxLang::MatrixMultiply(vec![nested_empty, y].into_boxed_slice()));
        let mut progress = || Ok(());
        assert!(
            mapped_fixed_product_consensus_with_progress(&egraph, nested_empty_root, &mut progress)
                .is_none(),
            "an empty nested witness cannot erase a product factor"
        );

        let funded = egraph.add(MxxLang::MatrixMultiply(vec![y, s].into_boxed_slice()));
        let mut full_calls = 0;
        assert!(
            mapped_fixed_product_consensus_with_progress(&egraph, funded, &mut || {
                full_calls += 1;
                Ok(())
            })
            .is_some()
        );
        let mut calls = 0;
        assert!(
            mapped_fixed_product_consensus_with_progress(&egraph, funded, &mut || {
                calls += 1;
                (calls < full_calls).then_some(()).ok_or(())
            })
            .is_none()
        );
    }

    #[test]
    fn mapped_fixed_product_consensus_keeps_switch_cases_opaque_and_scales_linearly() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (y, _) = matrix_atom(&mut egraph, "mapped-opaque-y", None);
        let (s, _) = matrix_atom(&mut egraph, "mapped-opaque-s", None);
        let product = egraph.add(MxxLang::MatrixMultiply(vec![y, s].into_boxed_slice()));
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let mut cases = vec![selector];
        cases.extend(std::iter::repeat_n(y, 128));
        let switch = egraph.add(MxxLang::Switch(cases.into_boxed_slice()));
        egraph.union(product, switch);
        egraph.rebuild();
        let mut switch_charges = 0;
        assert!(
            mapped_fixed_product_consensus_with_progress(&egraph, product, &mut || {
                switch_charges += 1;
                Ok(())
            })
            .is_some()
        );
        assert!(switch_charges < 32, "Switch cases are opaque to product consensus");

        let charges_for = |width| {
            let mut egraph = EGraph::new(MxxAnalysis::default());
            let factors = (0..width)
                .map(|index| {
                    matrix_atom(&mut egraph, &format!("mapped-linear-{width}-{index}"), None).0
                })
                .collect::<Vec<_>>();
            let product = egraph.add(MxxLang::MatrixMultiply(factors.into_boxed_slice()));
            egraph.rebuild();
            let mut charges = 0;
            assert!(
                mapped_fixed_product_consensus_with_progress(&egraph, product, &mut || {
                    charges += 1;
                    Ok(())
                })
                .is_some()
            );
            charges
        };
        let eight = charges_for(8);
        let sixteen = charges_for(16);
        assert!(
            sixteen > eight && sixteen <= eight * 3,
            "near-linear charges: {eight} -> {sixteen}"
        );

        let association_charges_for = |depth| {
            let mut egraph = EGraph::new(MxxAnalysis::default());
            let factors = (0..depth)
                .map(|index| {
                    matrix_atom(&mut egraph, &format!("mapped-association-{depth}-{index}"), None).0
                })
                .collect::<Vec<_>>();
            let mut product = factors[0];
            for factor in &factors[1..] {
                product =
                    egraph.add(MxxLang::MatrixMultiply(vec![product, *factor].into_boxed_slice()));
            }
            let mut alternate = *factors.last().expect("nonempty association");
            for factor in factors[..factors.len() - 1].iter().rev() {
                alternate = egraph
                    .add(MxxLang::MatrixMultiply(vec![*factor, alternate].into_boxed_slice()));
            }
            egraph.union(product, alternate);
            egraph.rebuild();
            let physical_witnesses = egraph[egraph.find(product)]
                .nodes
                .iter()
                .filter(|node| matches!(node, MxxLang::MatrixMultiply(_)))
                .count();
            assert!(
                physical_witnesses >= 2,
                "both immediate association witnesses remain in the rebuilt root class"
            );
            let mut charges = 0;
            let leaves =
                mapped_fixed_product_consensus_with_progress(&egraph, product, &mut || {
                    charges += 1;
                    Ok(())
                })
                .expect("funded association")
                .expect("association is a product");
            assert_eq!(leaves.len(), depth);
            charges
        };
        let associated_eight = association_charges_for(8);
        let associated_sixteen = association_charges_for(16);
        assert!(
            associated_sixteen <= associated_eight * 4,
            "output-sensitive witness traversal scales within the retained association outputs: {associated_eight} -> {associated_sixteen}"
        );
    }

    #[test]
    fn signed_ordered_monomials_reject_different_product_witnesses() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (first, _) = matrix_atom(&mut egraph, "monomial-consensus-first", None);
        let (second, _) = matrix_atom(&mut egraph, "monomial-consensus-second", None);
        let (factor, _) = matrix_atom(&mut egraph, "monomial-consensus-factor", None);
        let left = egraph.add(MxxLang::MatrixAdd(vec![first, second].into_boxed_slice()));
        let right = egraph.add(MxxLang::MatrixAdd(vec![second, first].into_boxed_slice()));
        let left_product =
            egraph.add(MxxLang::MatrixMultiply(vec![left, factor].into_boxed_slice()));
        let right_product =
            egraph.add(MxxLang::MatrixMultiply(vec![right, factor].into_boxed_slice()));
        egraph.union(left_product, right_product);
        egraph.rebuild();
        let mut progress = || Ok(());
        assert!(
            signed_ordered_monomial_spines(&egraph, &[(left_product, false)], &mut progress)
                .is_none(),
            "mapped fixed products require one ordered canonical witness"
        );

        let (relation_target, _) = matrix_atom(&mut egraph, "monomial-relation-target", None);
        egraph.union(left_product, relation_target);
        egraph.rebuild();
        let mut progress = || Ok(());
        assert!(
            signed_ordered_monomial_spines(&egraph, &[(left_product, false)], &mut progress)
                .is_none(),
            "a non-Multiply relation alternative cannot hide conflicting product witnesses"
        );

        let (cycle_leaf, _) = matrix_atom(&mut egraph, "monomial-cycle", None);
        let cyclic = egraph.add(MxxLang::MatrixMultiply(vec![cycle_leaf].into_boxed_slice()));
        egraph.union(cyclic, cycle_leaf);
        let mut progress = || Ok(());
        assert!(
            signed_ordered_monomial_spines(&egraph, &[(cyclic, false)], &mut progress).is_none()
        );
    }

    #[test]
    fn signed_ordered_monomials_charge_every_generated_cartesian_monomial() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (first, _) = matrix_atom(&mut egraph, "monomial-shared-first", None);
        let (second, _) = matrix_atom(&mut egraph, "monomial-shared-second", None);
        let shared = egraph.add(MxxLang::MatrixAdd(vec![first, second].into_boxed_slice()));
        let product = egraph.add(MxxLang::MatrixMultiply(vec![shared, shared].into_boxed_slice()));
        egraph.rebuild();

        let mut charges = 0;
        let spines = signed_ordered_monomial_spines(&egraph, &[(product, false)], &mut || {
            charges += 1;
            Ok(())
        })
        .expect("shared additive factor has one consensus result");
        assert_eq!(spines.len(), 4, "only mathematically required product monomials are retained");
        assert_eq!(
            charges, 89,
            "the selected direct product witness and its required Cartesian monomials are charged"
        );
    }

    #[test]
    fn signed_ordered_monomials_stop_before_an_unfunded_cartesian_product() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (first, _) = matrix_atom(&mut egraph, "monomial-budget-first", None);
        let (second, _) = matrix_atom(&mut egraph, "monomial-budget-second", None);
        let shared = egraph.add(MxxLang::MatrixAdd(vec![first, second].into_boxed_slice()));
        let product = egraph.add(MxxLang::MatrixMultiply(vec![shared, shared].into_boxed_slice()));
        egraph.rebuild();

        let mut remaining = 5;
        assert!(
            signed_ordered_monomial_spines(&egraph, &[(product, false)], &mut || {
                if remaining == 0 {
                    Err(())
                } else {
                    remaining -= 1;
                    Ok(())
                }
            })
            .is_none(),
            "the sixth generated monomial is rejected before it is allocated"
        );
    }

    #[test]
    fn signed_ordered_monomials_stop_before_expanding_a_compact_shared_add_dag() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (first, _) = matrix_atom(&mut egraph, "monomial-dag-first", None);
        let (second, _) = matrix_atom(&mut egraph, "monomial-dag-second", None);
        let mut shared = egraph.add(MxxLang::MatrixAdd(vec![first, second].into_boxed_slice()));
        for _ in 0..12 {
            shared = egraph.add(MxxLang::MatrixAdd(vec![shared, shared].into_boxed_slice()));
        }
        let (factor, _) = matrix_atom(&mut egraph, "monomial-dag-factor", None);
        let product = egraph.add(MxxLang::MatrixMultiply(vec![shared, factor].into_boxed_slice()));
        egraph.rebuild();

        let mut remaining = 64;
        assert!(
            signed_ordered_monomial_spines(&egraph, &[(product, false)], &mut || {
                if remaining == 0 {
                    Err(())
                } else {
                    remaining -= 1;
                    Ok(())
                }
            })
            .is_none(),
            "representative copies stop at the shared budget before the compact DAG expands"
        );
    }

    #[test]
    fn binder_distribution_cancels_distributed_product_without_switch_cartesian_expansion() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let binder = test_binder(&mut egraph, 0, 1);
        let selector = egraph.add(MxxLang::IntBinder(binder));
        let first = binder_matrix_atom(&mut egraph, selector, "distributed-first");
        let second = binder_matrix_atom(&mut egraph, selector, "distributed-second");
        let (factor, _) = matrix_atom(&mut egraph, "distributed-factor", None);
        let shared_sum = egraph.add(MxxLang::MatrixAdd(vec![first, second].into_boxed_slice()));
        let shared =
            egraph.add(MxxLang::MatrixMultiply(vec![shared_sum, factor].into_boxed_slice()));
        let zero = egraph.add(MxxLang::IntConst(0.into()));
        let one = egraph.add(MxxLang::IntConst(1.into()));
        let instantiate =
            |egraph: &mut EGraph<MxxLang, MxxAnalysis>, term, index| {
                family::instantiate_shared_element(egraph, term, binder, index, &mut || {
                    Ok::<(), ()>(())
                })
                .expect("test instantiation")
            };
        let first_zero = instantiate(&mut egraph, first, zero);
        let second_zero = instantiate(&mut egraph, second, zero);
        let first_one = instantiate(&mut egraph, first, one);
        let second_one = instantiate(&mut egraph, second, one);
        let case = |egraph: &mut EGraph<MxxLang, MxxAnalysis>, left, right| {
            let left = egraph.add(MxxLang::MatrixMultiply(vec![left, factor].into_boxed_slice()));
            let right = egraph.add(MxxLang::MatrixMultiply(vec![right, factor].into_boxed_slice()));
            egraph.add(MxxLang::MatrixAdd(vec![left, right].into_boxed_slice()))
        };
        let zero_case = case(&mut egraph, first_zero, second_zero);
        let one_case = case(&mut egraph, first_one, second_one);
        let switch =
            egraph.add(MxxLang::Switch(vec![selector, zero_case, one_case].into_boxed_slice()));
        let fixed = egraph.add(MxxLang::MatrixNegate([shared]));
        let root = egraph.add(MxxLang::MatrixAdd(vec![switch, fixed].into_boxed_slice()));
        egraph.rebuild();

        let (selector, plan) = only_binder_plan(&egraph, root);
        let mut charges = 0;
        let replacement = build_binder_aware_pointwise_add_switch_cancellation(
            &mut egraph,
            root,
            selector,
            plan,
            &mut || {
                charges += 1;
                Ok(())
            },
        )
        .expect("distributed signal terms cancel per binder case");
        let cases =
            switch_node(&egraph, replacement).expect("replacement remains a two-case switch");
        assert_eq!(cases.len(), 3, "Switch remains opaque and has no selector Cartesian expansion");
        assert!(charges > 0, "read-only planning and construction share the rewrite budget");
        for case in &cases[1..] {
            assert!(egraph[egraph.find(*case)].nodes.iter().any(|node| {
                matches!(node, MxxLang::MatrixConstant(spec)
                    if matches!(egraph.analysis.symbols.matrix_constants.get(spec.0), Some(super::super::identity::MatrixConstantSpec { value: MatrixConstantValue::Zero, .. })))
            }));
        }
    }

    #[test]
    fn fixed_guided_peeling_removes_one_additive_product_summand_in_order() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (a, _) = matrix_atom(&mut egraph, "peel-a", None);
        let (b, _) = matrix_atom(&mut egraph, "peel-b", None);
        let (d, _) = matrix_atom(&mut egraph, "peel-d", None);
        let sum = egraph.add(MxxLang::MatrixAdd(vec![a, b].into_boxed_slice()));
        let actual = egraph.add(MxxLang::MatrixMultiply(vec![sum, d].into_boxed_slice()));
        egraph.rebuild();
        let mut terms = vec![PeelTerm::Concrete { base: actual, negative: false }];
        let mut progress = || Ok(());
        assert_eq!(
            peel_fixed_targets(
                &egraph,
                &mut terms,
                &[(vec![egraph.find(a), egraph.find(d)].into_boxed_slice(), true)],
                &mut progress,
            ),
            Some((true, Vec::new()))
        );
        assert!(matches!(
            terms.as_slice(),
            [PeelTerm::ProductFactor { prefix, terms, suffix, negative: false }]
                if prefix.is_empty() && suffix.as_ref() == [egraph.find(d)] && terms == &vec![(egraph.find(b), false)]
        ));
    }

    #[test]
    fn fixed_guided_peeling_debug_logs_matched_actual_and_replacement() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (a, _) = matrix_atom(&mut egraph, "peel-debug-a", None);
        let (b, _) = matrix_atom(&mut egraph, "peel-debug-b", None);
        let (d, _) = matrix_atom(&mut egraph, "peel-debug-d", None);
        let sum = egraph.add(MxxLang::MatrixAdd(vec![a, b].into_boxed_slice()));
        let actual = egraph.add(MxxLang::MatrixMultiply(vec![sum, d].into_boxed_slice()));
        egraph.rebuild();
        let mut terms = vec![PeelTerm::Concrete { base: actual, negative: false }];
        let capture = FixedPeelEventCapture::default();
        let subscriber = tracing_subscriber::registry().with(capture.clone());

        tracing::subscriber::with_default(subscriber, || {
            assert_eq!(
                peel_fixed_targets(
                    &egraph,
                    &mut terms,
                    &[(vec![egraph.find(a), egraph.find(d)].into_boxed_slice(), true)],
                    &mut || Ok(()),
                ),
                Some((true, Vec::new()))
            );
        });

        let events = capture.0.lock().expect("event capture lock");
        assert!(events.iter().any(|fields| {
            fields.get("event").is_some_and(|event| event.contains("fixed_target_peel_match")) &&
                fields.get("target_index").is_some_and(|index| index.contains('0'))
        }));
        assert!(events.iter().any(|fields| {
            fields
                .get("contribution")
                .is_some_and(|value| value.contains("matched_actual_before_replacement")) &&
                fields.get("kind").is_some_and(|value| value.contains("concrete"))
        }));
        assert!(events.iter().any(|fields| {
            fields
                .get("contribution")
                .is_some_and(|value| value.contains("replacement_after_peeling")) &&
                fields.get("kind").is_some_and(|value| value.contains("product_factor")) &&
                fields.get("selected_additive_leaves").is_some_and(|value| {
                    value.contains(&usize::from(egraph.find(b)).to_string())
                })
        }));
    }

    #[test]
    fn pre_cancel_mapped_fixed_spine_debug_log_preserves_sign_and_factor_order() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (left, _) = matrix_atom(&mut egraph, "pre-cancel-left", None);
        let (middle, _) = matrix_atom(&mut egraph, "pre-cancel-middle", None);
        let (right, _) = matrix_atom(&mut egraph, "pre-cancel-right", None);
        egraph.rebuild();
        let capture = FixedPeelEventCapture::default();
        let subscriber = tracing_subscriber::registry().with(capture.clone());

        tracing::subscriber::with_default(subscriber, || {
            log_pre_cancel_mapped_fixed_spines(
                &egraph,
                0,
                &[(
                    vec![egraph.find(left), egraph.find(middle), egraph.find(right)]
                        .into_boxed_slice(),
                    true,
                )],
            );
        });

        let events = capture.0.lock().expect("event capture lock");
        let spine = events
            .iter()
            .find(|fields| {
                fields
                    .get("event")
                    .is_some_and(|event| event.contains("binder_pre_cancel_mapped_fixed_spine\""))
            })
            .expect("the retained pre-cancel spine is logged");
        assert!(spine.get("negative").is_some_and(|negative| negative.contains("true")));
        let factors = spine.get("factors").expect("factor descriptors");
        let left = format!("({},", usize::from(egraph.find(left)));
        let middle = format!("({},", usize::from(egraph.find(middle)));
        let right = format!("({},", usize::from(egraph.find(right)));
        assert!(factors.find(&left) < factors.find(&middle));
        assert!(factors.find(&middle) < factors.find(&right));
    }

    #[test]
    fn fixed_guided_peeling_descends_one_uncontested_nested_product() {
        fn run(negations: usize) {
            let mut egraph = EGraph::new(MxxAnalysis::default());
            let (p, _) = matrix_atom(&mut egraph, "peel-nested-prefix", None);
            let (a, _) = matrix_atom(&mut egraph, "peel-nested-signal", None);
            let (noise, _) = matrix_atom(&mut egraph, "peel-nested-noise", None);
            let (other, _) = matrix_atom(&mut egraph, "peel-nested-other", None);
            let (d, _) = matrix_atom(&mut egraph, "peel-nested-suffix", None);
            let grouped = egraph.add(MxxLang::MatrixAdd(vec![a, noise].into_boxed_slice()));
            let signal = egraph.add(MxxLang::MatrixMultiply(vec![p, grouped].into_boxed_slice()));
            let outer = egraph.add(MxxLang::MatrixAdd(vec![signal, other].into_boxed_slice()));
            let mut actual = egraph.add(MxxLang::MatrixMultiply(vec![outer, d].into_boxed_slice()));
            for _ in 0..negations {
                actual = egraph.add(MxxLang::MatrixNegate([actual]));
            }
            egraph.rebuild();

            let sign = negations % 2 == 0;
            let signal_target =
                (vec![egraph.find(p), egraph.find(a), egraph.find(d)].into_boxed_slice(), sign);
            let other_target = (vec![egraph.find(other), egraph.find(d)].into_boxed_slice(), sign);
            let noise_target =
                (vec![egraph.find(p), egraph.find(noise), egraph.find(d)].into_boxed_slice(), sign);
            let mut terms = vec![PeelTerm::Concrete { base: actual, negative: false }];
            let mut progress = || Ok(());
            assert_eq!(
                peel_fixed_targets(
                    &egraph,
                    &mut terms,
                    &[signal_target, other_target],
                    &mut progress
                ),
                Some((true, Vec::new()))
            );
            assert!(matches!(
                terms.as_slice(),
                [PeelTerm::ProductFactor { prefix, terms, suffix, negative }]
                    if prefix.as_ref() == [egraph.find(p)]
                        && terms == &vec![(egraph.find(noise), false)]
                        && suffix.as_ref() == [egraph.find(d)]
                        && *negative == (negations % 2 == 1)
            ));
            let mut progress = || Ok(());
            assert_eq!(
                peel_fixed_targets(&egraph, &mut terms, &[noise_target], &mut progress),
                Some((true, Vec::new()))
            );
            assert!(terms.is_empty(), "a later target reopens the nested-product residual");
        }

        run(0);
        run(1);
        run(2);
    }

    #[test]
    fn fixed_guided_nested_product_peeling_folds_inner_negate_sign() {
        fn run(inner_negations: usize) {
            let mut egraph = EGraph::new(MxxAnalysis::default());
            let (p, _) = matrix_atom(&mut egraph, "peel-inner-negate-prefix", None);
            let (a, _) = matrix_atom(&mut egraph, "peel-inner-negate-signal", None);
            let (noise, _) = matrix_atom(&mut egraph, "peel-inner-negate-noise", None);
            let (other, _) = matrix_atom(&mut egraph, "peel-inner-negate-other", None);
            let (d, _) = matrix_atom(&mut egraph, "peel-inner-negate-suffix", None);
            let grouped = egraph.add(MxxLang::MatrixAdd(vec![a, noise].into_boxed_slice()));
            let mut signal =
                egraph.add(MxxLang::MatrixMultiply(vec![p, grouped].into_boxed_slice()));
            for _ in 0..inner_negations {
                signal = egraph.add(MxxLang::MatrixNegate([signal]));
            }
            let outer = egraph.add(MxxLang::MatrixAdd(vec![signal, other].into_boxed_slice()));
            let actual = egraph.add(MxxLang::MatrixMultiply(vec![outer, d].into_boxed_slice()));
            egraph.rebuild();

            let signal_target_negative = inner_negations % 2 == 0;
            let signal_target = (
                vec![egraph.find(p), egraph.find(a), egraph.find(d)].into_boxed_slice(),
                signal_target_negative,
            );
            let other_target = (vec![egraph.find(other), egraph.find(d)].into_boxed_slice(), true);
            let noise_target = (
                vec![egraph.find(p), egraph.find(noise), egraph.find(d)].into_boxed_slice(),
                signal_target_negative,
            );
            let mut terms = vec![PeelTerm::Concrete { base: actual, negative: false }];
            let mut progress = || Ok(());
            assert_eq!(
                peel_fixed_targets(
                    &egraph,
                    &mut terms,
                    &[signal_target, other_target],
                    &mut progress
                ),
                Some((true, Vec::new()))
            );
            assert!(matches!(
                terms.as_slice(),
                [PeelTerm::ProductFactor { prefix, terms, suffix, negative }]
                    if prefix.as_ref() == [egraph.find(p)]
                        && terms == &vec![(egraph.find(noise), false)]
                        && suffix.as_ref() == [egraph.find(d)]
                        && *negative == (inner_negations % 2 == 1)
            ));
            let mut progress = || Ok(());
            assert_eq!(
                peel_fixed_targets(&egraph, &mut terms, &[noise_target], &mut progress),
                Some((true, Vec::new()))
            );
            assert!(terms.is_empty());

            if inner_negations % 2 == 1 {
                let original = vec![PeelTerm::Concrete { base: actual, negative: false }];
                let mut wrong_sign = original.clone();
                let mut progress = || Ok(());
                let wrong_target =
                    (vec![egraph.find(p), egraph.find(a), egraph.find(d)].into_boxed_slice(), true);
                assert_eq!(
                    peel_fixed_targets(
                        &egraph,
                        &mut wrong_sign,
                        &[wrong_target.clone()],
                        &mut progress
                    ),
                    Some((false, vec![wrong_target]))
                );
                assert_eq!(
                    wrong_sign, original,
                    "a same-sign fixed term cannot erase an inner negative signal"
                );
            }
        }

        run(1);
        run(2);
    }

    #[test]
    fn fixed_guided_nested_product_peeling_scales_linearly_with_association_depth() {
        fn charges_for(depth: usize) -> usize {
            let mut egraph = EGraph::new(MxxAnalysis::default());
            let prefixes = (0..depth)
                .map(|index| {
                    matrix_atom(&mut egraph, &format!("peel-linear-prefix-{depth}-{index}"), None).0
                })
                .collect::<Vec<_>>();
            let (signal, _) = matrix_atom(&mut egraph, "peel-linear-signal", None);
            let (noise, _) = matrix_atom(&mut egraph, "peel-linear-noise", None);
            let (suffix, _) = matrix_atom(&mut egraph, "peel-linear-suffix", None);
            let grouped = egraph.add(MxxLang::MatrixAdd(vec![signal, noise].into_boxed_slice()));
            let mut product = prefixes[0];
            for prefix in &prefixes[1..] {
                product =
                    egraph.add(MxxLang::MatrixMultiply(vec![product, *prefix].into_boxed_slice()));
            }
            product =
                egraph.add(MxxLang::MatrixMultiply(vec![product, grouped].into_boxed_slice()));
            let actual =
                egraph.add(MxxLang::MatrixMultiply(vec![product, suffix].into_boxed_slice()));
            egraph.rebuild();
            let target = prefixes
                .iter()
                .map(|prefix| egraph.find(*prefix))
                .chain(std::iter::once(egraph.find(signal)))
                .chain(std::iter::once(egraph.find(suffix)))
                .collect::<Vec<_>>()
                .into_boxed_slice();
            let mut terms = vec![PeelTerm::Concrete { base: actual, negative: false }];
            let mut charges = 0;
            assert_eq!(
                peel_fixed_targets(&egraph, &mut terms, &[(target, true)], &mut || {
                    charges += 1;
                    Ok(())
                }),
                Some((true, Vec::new()))
            );
            assert!(matches!(
                terms.as_slice(),
                [PeelTerm::ProductFactor { prefix, terms, suffix: remaining_suffix, negative: false }]
                    if prefix.len() == depth
                        && terms == &vec![(egraph.find(noise), false)]
                        && remaining_suffix.as_ref() == [egraph.find(suffix)]
            ));
            charges
        }

        let shallow = charges_for(32);
        let deep = charges_for(128);
        assert!(
            deep < shallow.saturating_mul(8),
            "four times the direct association depth must not produce quadratic planning work: shallow={shallow}, deep={deep}"
        );
    }

    #[test]
    fn fixed_guided_nested_product_peeling_is_order_independent_and_transactional() {
        fn build(
            grouped_first: bool,
        ) -> (EGraph<MxxLang, MxxAnalysis>, Id, Box<[Id]>, Id, Id, Id, Id, Id) {
            let mut egraph = EGraph::new(MxxAnalysis::default());
            let (p, _) = matrix_atom(&mut egraph, "peel-nested-choice-prefix", None);
            let (a, _) = matrix_atom(&mut egraph, "peel-nested-choice-signal", None);
            let (noise, _) = matrix_atom(&mut egraph, "peel-nested-choice-noise", None);
            let (other, _) = matrix_atom(&mut egraph, "peel-nested-choice-other", None);
            let (wide_one, _) = matrix_atom(&mut egraph, "peel-nested-choice-wide-one", None);
            let (wide_two, _) = matrix_atom(&mut egraph, "peel-nested-choice-wide-two", None);
            let (d, _) = matrix_atom(&mut egraph, "peel-nested-choice-suffix", None);
            let grouped = egraph.add(MxxLang::MatrixAdd(vec![a, noise].into_boxed_slice()));
            let signal = egraph.add(MxxLang::MatrixMultiply(vec![p, grouped].into_boxed_slice()));
            let short = egraph.add(MxxLang::MatrixAdd(vec![signal, other].into_boxed_slice()));
            let wide =
                egraph.add(MxxLang::MatrixAdd(vec![signal, wide_one, wide_two].into_boxed_slice()));
            egraph.union(short, wide);
            egraph.rebuild();
            let outer = egraph.find(short);
            if !grouped_first {
                egraph[outer].nodes.reverse();
            }
            let actual = egraph.add(MxxLang::MatrixMultiply(vec![outer, d].into_boxed_slice()));
            egraph.rebuild();
            let target = vec![egraph.find(p), egraph.find(a), egraph.find(d)].into_boxed_slice();
            (egraph, actual, target, p, a, noise, other, d)
        }

        for grouped_first in [true, false] {
            let (egraph, actual, target, p, _, noise, other, _) = build(grouped_first);
            let original = vec![PeelTerm::Concrete { base: actual, negative: false }];
            let mut terms = original.clone();
            let mut progress = || Ok(());
            assert_eq!(
                peel_fixed_targets(&egraph, &mut terms, &[(target.clone(), true)], &mut progress),
                Some((true, Vec::new()))
            );
            assert_eq!(
                terms.len(),
                2,
                "the smaller physical Add witness is selected regardless of node order"
            );
            assert!(terms.iter().any(|term| matches!(
                term,
                PeelTerm::ProductFactor { prefix, terms, suffix, negative: false }
                    if prefix.as_ref() == [egraph.find(p)]
                        && terms == &vec![(egraph.find(noise), false)]
                        && suffix.len() == 1
            )));
            assert!(terms.iter().any(|term| matches!(
                term,
                PeelTerm::ProductFactor { prefix, terms, suffix, negative: false }
                    if prefix.is_empty()
                        && terms == &vec![(egraph.find(other), false)]
                        && suffix.len() == 1
            )));

            let before = egraph.total_size();
            let mut full_calls = 0;
            let mut funded = original.clone();
            assert!(
                peel_fixed_targets(&egraph, &mut funded, &[(target.clone(), true)], &mut || {
                    full_calls += 1;
                    Ok(())
                })
                .is_some()
            );
            let mut interrupted = original.clone();
            let mut calls = 0;
            assert!(
                peel_fixed_targets(&egraph, &mut interrupted, &[(target, true)], &mut || {
                    calls += 1;
                    (calls < full_calls).then_some(()).ok_or(())
                })
                .is_none()
            );
            assert_eq!(interrupted, original, "an interrupted nested plan cannot commit");
            assert_eq!(egraph.total_size(), before, "nested peeling stays read-only");
        }

        let (egraph, actual, _, p, a, _, _, d) = build(true);
        let reordered = vec![egraph.find(a), egraph.find(p), egraph.find(d)].into_boxed_slice();
        let original = vec![PeelTerm::Concrete { base: actual, negative: false }];
        let mut terms = original.clone();
        let mut progress = || Ok(());
        assert_eq!(
            peel_fixed_targets(&egraph, &mut terms, &[(reordered.clone(), true)], &mut progress),
            Some((false, vec![(reordered, true)]))
        );
        assert_eq!(terms, original, "noncommutative reordering remains ineligible");
    }

    #[test]
    fn fixed_guided_peeling_cancels_lookup_y_signal_through_a_constant_scalar() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (case_switch, _) = matrix_atom(&mut egraph, "lookup-case-switch", None);
        let output_scalar = scalar_polynomial_constant(&mut egraph, &[9]);
        let gadget = regular_scalar_gadget(&mut egraph);
        let (low_preimage, _) = matrix_atom(&mut egraph, "lookup-low-preimage", None);
        // The lookup preprocessing target contributes yG, while the encoded
        // signal reaches the relation pass as Gy.  A regular gadget is Large,
        // so this must be exact signal cancellation rather than a bound.
        let actual = egraph.add(MxxLang::MatrixMultiply(
            vec![case_switch, gadget, output_scalar, low_preimage].into_boxed_slice(),
        ));
        egraph.rebuild();
        let target = vec![
            egraph.find(case_switch),
            egraph.find(output_scalar),
            egraph.find(gadget),
            egraph.find(low_preimage),
        ]
        .into_boxed_slice();
        let mut terms = vec![PeelTerm::Concrete { base: actual, negative: false }];
        let mut progress = || Ok(());
        assert_eq!(
            peel_fixed_targets(&egraph, &mut terms, &[(target, true)], &mut progress),
            Some((true, Vec::new()))
        );
        assert!(terms.is_empty(), "the lookup y-signal must not survive as bounded noise");
    }

    #[test]
    fn fixed_guided_peeling_rejects_nonconstant_scalar_polynomial_reordering() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (case_switch, _) = matrix_atom(&mut egraph, "lookup-nonconstant-case-switch", None);
        let nonconstant_scalar = scalar_polynomial_constant(&mut egraph, &[9, 1]);
        let gadget = regular_scalar_gadget(&mut egraph);
        let (low_preimage, _) = matrix_atom(&mut egraph, "lookup-nonconstant-low-preimage", None);
        let actual = egraph.add(MxxLang::MatrixMultiply(
            vec![case_switch, gadget, nonconstant_scalar, low_preimage].into_boxed_slice(),
        ));
        egraph.rebuild();
        let target = vec![
            egraph.find(case_switch),
            egraph.find(nonconstant_scalar),
            egraph.find(gadget),
            egraph.find(low_preimage),
        ]
        .into_boxed_slice();
        let original = vec![PeelTerm::Concrete { base: actual, negative: false }];
        let mut terms = original.clone();
        let mut progress = || Ok(());
        assert_eq!(
            peel_fixed_targets(&egraph, &mut terms, &[(target.clone(), true)], &mut progress),
            Some((false, vec![(target, true)]))
        );
        assert_eq!(terms, original, "a nonconstant polynomial is not a central scalar");
    }

    #[test]
    fn fixed_guided_product_factor_scalar_fallback_preserves_the_selected_sign() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (case_switch, _) = matrix_atom(&mut egraph, "lookup-signed-case-switch", None);
        let output_scalar = scalar_polynomial_constant(&mut egraph, &[9]);
        let gadget = regular_scalar_gadget(&mut egraph);
        let (low_preimage, _) = matrix_atom(&mut egraph, "lookup-signed-low-preimage", None);
        egraph.rebuild();
        let target = vec![
            egraph.find(case_switch),
            egraph.find(output_scalar),
            egraph.find(gadget),
            egraph.find(low_preimage),
        ]
        .into_boxed_slice();
        let original = vec![PeelTerm::ProductFactor {
            prefix: vec![egraph.find(case_switch)].into_boxed_slice(),
            terms: vec![(egraph.find(gadget), true)],
            suffix: vec![egraph.find(output_scalar), egraph.find(low_preimage)].into_boxed_slice(),
            negative: false,
        }];

        let mut same_sign = original.clone();
        let mut progress = || Ok(());
        assert_eq!(
            peel_fixed_targets(&egraph, &mut same_sign, &[(target.clone(), true)], &mut progress),
            Some((false, vec![(target.clone(), true)]))
        );
        assert_eq!(same_sign, original);

        let mut opposite_sign = original;
        let mut progress = || Ok(());
        assert_eq!(
            peel_fixed_targets(&egraph, &mut opposite_sign, &[(target, false)], &mut progress),
            Some((true, Vec::new()))
        );
        assert!(opposite_sign.is_empty());
    }

    #[test]
    fn fixed_guided_scalar_fallback_requires_exact_duplicate_multiplicity() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (case_switch, _) = matrix_atom(&mut egraph, "lookup-duplicate-case-switch", None);
        let output_scalar = scalar_polynomial_constant(&mut egraph, &[9]);
        let gadget = regular_scalar_gadget(&mut egraph);
        let (low_preimage, _) = matrix_atom(&mut egraph, "lookup-duplicate-low-preimage", None);
        let actual = egraph.add(MxxLang::MatrixMultiply(
            vec![case_switch, gadget, output_scalar, output_scalar, low_preimage]
                .into_boxed_slice(),
        ));
        egraph.rebuild();
        let matching_target = vec![
            egraph.find(case_switch),
            egraph.find(output_scalar),
            egraph.find(output_scalar),
            egraph.find(gadget),
            egraph.find(low_preimage),
        ]
        .into_boxed_slice();
        let mut terms = vec![PeelTerm::Concrete { base: actual, negative: false }];
        let mut progress = || Ok(());
        assert_eq!(
            peel_fixed_targets(&egraph, &mut terms, &[(matching_target, true)], &mut progress),
            Some((true, Vec::new()))
        );

        let mismatched_target = vec![
            egraph.find(case_switch),
            egraph.find(output_scalar),
            egraph.find(gadget),
            egraph.find(low_preimage),
        ]
        .into_boxed_slice();
        let original = vec![PeelTerm::Concrete { base: actual, negative: false }];
        let mut terms = original.clone();
        let mut progress = || Ok(());
        assert_eq!(
            peel_fixed_targets(
                &egraph,
                &mut terms,
                &[(mismatched_target.clone(), true)],
                &mut progress,
            ),
            Some((false, vec![(mismatched_target, true)]))
        );
        assert_eq!(terms, original);
    }

    #[test]
    fn fixed_guided_scalar_fallback_keeps_nonscalar_order_exact() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (case_switch, _) = matrix_atom(&mut egraph, "lookup-order-case-switch", None);
        let output_scalar = scalar_polynomial_constant(&mut egraph, &[9]);
        let gadget = regular_scalar_gadget(&mut egraph);
        let (first, _) = matrix_atom(&mut egraph, "lookup-order-first", None);
        let (second, _) = matrix_atom(&mut egraph, "lookup-order-second", None);
        let actual = egraph.add(MxxLang::MatrixMultiply(
            vec![case_switch, gadget, output_scalar, first, second].into_boxed_slice(),
        ));
        egraph.rebuild();
        let target = vec![
            egraph.find(case_switch),
            egraph.find(output_scalar),
            egraph.find(gadget),
            egraph.find(second),
            egraph.find(first),
        ]
        .into_boxed_slice();
        let original = vec![PeelTerm::Concrete { base: actual, negative: false }];
        let mut terms = original.clone();
        let mut progress = || Ok(());
        assert_eq!(
            peel_fixed_targets(&egraph, &mut terms, &[(target.clone(), true)], &mut progress),
            Some((false, vec![(target, true)]))
        );
        assert_eq!(terms, original);
    }

    #[test]
    fn fixed_guided_peeling_preserves_product_prefix_suffix_and_rejects_reordering() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (p, _) = matrix_atom(&mut egraph, "peel-prefix", None);
        let (a, _) = matrix_atom(&mut egraph, "peel-middle-a", None);
        let (b, _) = matrix_atom(&mut egraph, "peel-middle-b", None);
        let (s, _) = matrix_atom(&mut egraph, "peel-suffix", None);
        let sum = egraph.add(MxxLang::MatrixAdd(vec![a, b].into_boxed_slice()));
        let actual = egraph.add(MxxLang::MatrixMultiply(vec![p, sum, s].into_boxed_slice()));
        egraph.rebuild();
        let target = vec![egraph.find(p), egraph.find(a), egraph.find(s)].into_boxed_slice();
        let mut terms = vec![PeelTerm::Concrete { base: actual, negative: false }];
        let mut progress = || Ok(());
        assert_eq!(
            peel_fixed_targets(&egraph, &mut terms, &[(target, true)], &mut progress),
            Some((true, Vec::new()))
        );
        let mut reordered = vec![PeelTerm::Concrete { base: actual, negative: false }];
        let mut progress = || Ok(());
        assert_eq!(
            peel_fixed_targets(
                &egraph,
                &mut reordered,
                &[(vec![egraph.find(a), egraph.find(p), egraph.find(s)].into_boxed_slice(), true)],
                &mut progress,
            ),
            Some((
                false,
                vec![(
                    vec![egraph.find(a), egraph.find(p), egraph.find(s)].into_boxed_slice(),
                    true
                )]
            ))
        );

        let negated_a = egraph.add(MxxLang::MatrixNegate([a]));
        let double_negated_a = egraph.add(MxxLang::MatrixNegate([negated_a]));
        let negative_product =
            egraph.add(MxxLang::MatrixMultiply(vec![p, negated_a, s].into_boxed_slice()));
        let double_negative_product =
            egraph.add(MxxLang::MatrixMultiply(vec![p, double_negated_a, s].into_boxed_slice()));
        egraph.rebuild();
        let mut negative_terms =
            vec![PeelTerm::Concrete { base: negative_product, negative: false }];
        let mut progress = || Ok(());
        assert_eq!(
            peel_fixed_targets(
                &egraph,
                &mut negative_terms,
                &[(vec![egraph.find(p), egraph.find(a), egraph.find(s)].into_boxed_slice(), false)],
                &mut progress,
            ),
            Some((true, Vec::new()))
        );
        assert!(negative_terms.is_empty());
        let mut double_negative_terms =
            vec![PeelTerm::Concrete { base: double_negative_product, negative: false }];
        let mut progress = || Ok(());
        assert_eq!(
            peel_fixed_targets(
                &egraph,
                &mut double_negative_terms,
                &[(vec![egraph.find(p), egraph.find(a), egraph.find(s)].into_boxed_slice(), true)],
                &mut progress,
            ),
            Some((true, Vec::new()))
        );
        assert!(double_negative_terms.is_empty());

        let ab = egraph.add(MxxLang::MatrixMultiply(vec![a, b].into_boxed_slice()));
        let abc = egraph.add(MxxLang::MatrixMultiply(vec![ab, p].into_boxed_slice()));
        let abcd = egraph.add(MxxLang::MatrixMultiply(vec![abc, s].into_boxed_slice()));
        egraph.rebuild();
        let mut associated = vec![PeelTerm::Concrete { base: abcd, negative: false }];
        let mut progress = || Ok(());
        assert_eq!(
            peel_fixed_targets(
                &egraph,
                &mut associated,
                &[(
                    vec![egraph.find(a), egraph.find(b), egraph.find(p), egraph.find(s)]
                        .into_boxed_slice(),
                    true
                )],
                &mut progress,
            ),
            Some((true, Vec::new()))
        );
        let mut reordered_associated = vec![PeelTerm::Concrete { base: abcd, negative: false }];
        let mut progress = || Ok(());
        assert_eq!(
            peel_fixed_targets(
                &egraph,
                &mut reordered_associated,
                &[(
                    vec![egraph.find(b), egraph.find(a), egraph.find(p), egraph.find(s)]
                        .into_boxed_slice(),
                    true
                )],
                &mut progress,
            ),
            Some((
                false,
                vec![(
                    vec![egraph.find(b), egraph.find(a), egraph.find(p), egraph.find(s)]
                        .into_boxed_slice(),
                    true
                )]
            ))
        );
    }

    #[test]
    fn fixed_guided_peeling_supports_signs_multiplicity_and_zero_fixed_terms() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (a, _) = matrix_atom(&mut egraph, "peel-sign-a", None);
        let (b, _) = matrix_atom(&mut egraph, "peel-sign-b", None);
        let grouped = egraph.add(MxxLang::MatrixAdd(vec![a, b].into_boxed_slice()));
        let negated = egraph.add(MxxLang::MatrixNegate([grouped]));
        let zero_spec = egraph.analysis.symbols.matrix_constants.intern(
            super::super::identity::MatrixConstantSpec {
                matrix_type: scalar_matrix_type(),
                value: MatrixConstantValue::Zero,
            },
        );
        let zero = egraph
            .add(MxxLang::MatrixConstant(super::super::identity::MatrixConstantSpecId(zero_spec)));
        egraph.rebuild();
        let mut terms = vec![PeelTerm::Concrete { base: negated, negative: false }];
        let mut progress = || Ok(());
        assert_eq!(
            peel_fixed_targets(
                &egraph,
                &mut terms,
                &[
                    (vec![egraph.find(zero)].into_boxed_slice(), false),
                    (vec![egraph.find(a)].into_boxed_slice(), false),
                    (vec![egraph.find(b)].into_boxed_slice(), false),
                ],
                &mut progress,
            ),
            Some((true, Vec::new()))
        );
        assert!(terms.is_empty());
    }

    #[test]
    fn fixed_guided_peeling_drops_zero_product_fixed_targets_before_planning() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (p, _) = matrix_atom(&mut egraph, "peel-zero-product-prefix", None);
        let (a, _) = matrix_atom(&mut egraph, "peel-zero-product-actual", None);
        let (d, _) = matrix_atom(&mut egraph, "peel-zero-product-suffix", None);
        let zero_spec = egraph.analysis.symbols.matrix_constants.intern(
            super::super::identity::MatrixConstantSpec {
                matrix_type: scalar_matrix_type(),
                value: MatrixConstantValue::Zero,
            },
        );
        let zero = egraph
            .add(MxxLang::MatrixConstant(super::super::identity::MatrixConstantSpecId(zero_spec)));
        let (zero_alias, _) = matrix_atom(&mut egraph, "peel-zero-product-opaque-alias", None);
        assert!(
            !egraph[zero_alias].nodes.iter().any(|node| matches!(node, MxxLang::MatrixConstant(_))),
            "the alias starts as a nonconstant physical node"
        );
        egraph.union(zero_alias, zero);
        egraph.rebuild();
        let mut zero_progress = || Ok(());
        assert!(
            is_exact_zero_matrix(&egraph, zero_alias, &mut zero_progress).unwrap(),
            "the original opaque alias resolves through its canonical e-class to the zero witness"
        );
        let actual = vec![PeelTerm::Concrete { base: a, negative: false }];
        for (spine, negative) in
            [(vec![zero, p, d], false), (vec![p, zero_alias, d], true), (vec![p, d, zero], false)]
        {
            let mut terms = actual.clone();
            let mut progress = || Ok(());
            assert_eq!(
                peel_fixed_targets(
                    &egraph,
                    &mut terms,
                    &[(spine.into_boxed_slice(), negative)],
                    &mut progress,
                ),
                Some((true, Vec::new())),
                "a zero factor in any ordered position is a zero fixed product"
            );
            assert_eq!(terms, actual, "zero normalization does not touch the actual residual");
        }

        let nested_actual = vec![PeelTerm::ProductFactor {
            prefix: vec![egraph.find(p)].into_boxed_slice(),
            terms: vec![(egraph.find(a), false)],
            suffix: vec![egraph.find(d)].into_boxed_slice(),
            negative: false,
        }];
        let zero_target =
            vec![egraph.find(p), egraph.find(zero), egraph.find(d)].into_boxed_slice();
        let mut nested_terms = nested_actual.clone();
        let mut progress = || Ok(());
        assert_eq!(
            peel_fixed_targets(
                &egraph,
                &mut nested_terms,
                &[(zero_target.clone(), true)],
                &mut progress,
            ),
            Some((true, Vec::new()))
        );
        assert_eq!(nested_terms, nested_actual, "zero products skip ProductFactor reopening too");

        let nonzero_target = vec![egraph.find(p), egraph.find(d)].into_boxed_slice();
        let mut nonzero_terms = actual.clone();
        let mut progress = || Ok(());
        assert_eq!(
            peel_fixed_targets(
                &egraph,
                &mut nonzero_terms,
                &[(nonzero_target.clone(), true)],
                &mut progress,
            ),
            Some((false, vec![(nonzero_target, true)])),
            "an ordinary fixed product remains unmatched without an exact witness"
        );
        assert_eq!(nonzero_terms, actual);

        let before = egraph.total_size();
        let mut full_calls = 0;
        let mut funded = actual.clone();
        assert!(
            peel_fixed_targets(&egraph, &mut funded, &[(zero_target.clone(), false)], &mut || {
                full_calls += 1;
                Ok(())
            },)
            .is_some()
        );
        let mut interrupted = actual.clone();
        let mut calls = 0;
        assert!(
            peel_fixed_targets(&egraph, &mut interrupted, &[(zero_target, false)], &mut || {
                calls += 1;
                (calls < full_calls).then_some(()).ok_or(())
            },)
            .is_none(),
            "an interrupted zero-factor scan cannot commit the private plan"
        );
        assert_eq!(interrupted, actual);
        assert_eq!(egraph.total_size(), before);
    }

    #[test]
    fn fixed_guided_peeling_keeps_switch_and_relation_unioned_products_atomic() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let (a, _) = matrix_atom(&mut egraph, "peel-switch-a", None);
        let (b, _) = matrix_atom(&mut egraph, "peel-switch-b", None);
        let (d, _) = matrix_atom(&mut egraph, "peel-switch-d", None);
        let switch = egraph.add(MxxLang::Switch(vec![selector, a, b].into_boxed_slice()));
        let switched = egraph.add(MxxLang::MatrixMultiply(vec![switch, d].into_boxed_slice()));
        let mut switch_terms = vec![PeelTerm::Concrete { base: switched, negative: false }];
        let before = egraph.total_size();
        let mut progress = || Ok(());
        assert_eq!(
            peel_fixed_targets(
                &egraph,
                &mut switch_terms,
                &[(vec![egraph.find(a), egraph.find(d)].into_boxed_slice(), true)],
                &mut progress,
            ),
            Some((false, vec![(vec![egraph.find(a), egraph.find(d)].into_boxed_slice(), true)]))
        );
        assert_eq!(egraph.total_size(), before, "failed peeling adds no e-nodes");

        let (s, _) = matrix_atom(&mut egraph, "peel-relation-s", None);
        let (h, _) = matrix_atom(&mut egraph, "peel-relation-h", None);
        let (noise, _) = matrix_atom(&mut egraph, "peel-relation-noise", None);
        let product = egraph.add(MxxLang::MatrixMultiply(vec![s, h].into_boxed_slice()));
        let (relation_target, _) = matrix_atom(&mut egraph, "peel-relation-target", None);
        egraph.union(product, relation_target);
        let grouped = egraph.add(MxxLang::MatrixAdd(vec![product, noise].into_boxed_slice()));
        let actual = egraph.add(MxxLang::MatrixMultiply(vec![grouped, d].into_boxed_slice()));
        egraph.rebuild();
        let mut terms = vec![PeelTerm::Concrete { base: actual, negative: false }];
        let mut progress = || Ok(());
        assert_eq!(
            peel_fixed_targets(
                &egraph,
                &mut terms,
                &[(vec![egraph.find(s), egraph.find(h), egraph.find(d)].into_boxed_slice(), true)],
                &mut progress,
            ),
            Some((true, Vec::new())),
            "the direct Multiply witness inside a relation-unioned e-class matches the target span"
        );
        assert!(matches!(
            terms.as_slice(),
            [PeelTerm::ProductFactor { terms, .. }] if terms == &vec![(egraph.find(noise), false)]
        ));
        let mut singleton_terms = vec![PeelTerm::Concrete { base: actual, negative: false }];
        let mut progress = || Ok(());
        assert_eq!(
            peel_fixed_targets(
                &egraph,
                &mut singleton_terms,
                &[(vec![egraph.find(product), egraph.find(d)].into_boxed_slice(), true)],
                &mut progress,
            ),
            Some((true, Vec::new())),
            "the relation-unioned product remains a canonical singleton identity"
        );

        let cyclic = egraph.add(MxxLang::MatrixMultiply(vec![relation_target].into_boxed_slice()));
        egraph.union(cyclic, relation_target);
        let mut progress = || Ok(());
        assert!(
            direct_signed_target_positions(
                &egraph,
                cyclic,
                0,
                false,
                &[egraph.find(cyclic)],
                &mut progress,
            )
            .is_some_and(|positions| positions.contains(&(1, false))),
            "a cyclic structural branch cannot discard the canonical singleton identity"
        );
        let mut visits = 0;
        assert_eq!(
            direct_signed_target_positions(
                &egraph,
                cyclic,
                0,
                false,
                &[egraph.find(s)],
                &mut || {
                    visits += 1;
                    Ok(())
                },
            ),
            Some(BTreeSet::new()),
            "a structural cycle alone is rejected as a witness without recursive traversal"
        );
        assert!(visits < 64, "cycle handling is local and bounded");
    }

    #[test]
    fn fixed_guided_peeling_tries_each_physical_add_candidate_without_expanding_siblings() {
        fn run(grouped_first: bool) {
            let mut egraph = EGraph::new(MxxAnalysis::default());
            let (left, _) = matrix_atom(&mut egraph, "peel-candidate-left", None);
            let (right, _) = matrix_atom(&mut egraph, "peel-candidate-right", None);
            let (residual, _) = matrix_atom(&mut egraph, "peel-candidate-residual", None);
            let (wide_one, _) = matrix_atom(&mut egraph, "peel-candidate-wide-one", None);
            let (wide_two, _) = matrix_atom(&mut egraph, "peel-candidate-wide-two", None);
            let (wide_three, _) = matrix_atom(&mut egraph, "peel-candidate-wide-three", None);
            let signal = egraph.add(MxxLang::MatrixMultiply(vec![left, right].into_boxed_slice()));
            let (first, second) = if grouped_first {
                (
                    egraph.add(MxxLang::MatrixAdd(vec![signal, residual].into_boxed_slice())),
                    egraph.add(MxxLang::MatrixAdd(
                        vec![signal, wide_one, wide_two, wide_three].into_boxed_slice(),
                    )),
                )
            } else {
                (
                    egraph.add(MxxLang::MatrixAdd(
                        vec![signal, wide_one, wide_two, wide_three].into_boxed_slice(),
                    )),
                    egraph.add(MxxLang::MatrixAdd(vec![signal, residual].into_boxed_slice())),
                )
            };
            egraph.union(first, second);
            egraph.rebuild();
            let factor = egraph.find(first);
            let physical_adds = egraph[factor]
                .nodes
                .iter()
                .filter_map(|node| match node {
                    MxxLang::MatrixAdd(children) => {
                        Some(children.iter().map(|child| egraph.find(*child)).collect::<Vec<_>>())
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(physical_adds.len(), 2, "both successful physical Add candidates remain");
            if !grouped_first {
                egraph[factor].nodes.reverse();
                let reversed = egraph[factor]
                    .nodes
                    .iter()
                    .filter_map(|node| match node {
                        MxxLang::MatrixAdd(children) => Some(
                            children.iter().map(|child| egraph.find(*child)).collect::<Vec<_>>(),
                        ),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                assert_eq!(reversed, physical_adds.iter().cloned().rev().collect::<Vec<_>>());
            }
            let mut actual = vec![PeelTerm::Concrete { base: first, negative: false }];
            let mut progress = || Ok(());
            assert_eq!(
                peel_fixed_targets(
                    &egraph,
                    &mut actual,
                    &[(vec![egraph.find(left), egraph.find(right)].into_boxed_slice(), true)],
                    &mut progress,
                ),
                Some((true, Vec::new()))
            );
            assert_eq!(
                actual,
                vec![PeelTerm::Concrete { base: egraph.find(residual), negative: false }],
                "only the selected grouped candidate is opened; wide siblings never form a Cartesian residual"
            );
        }
        run(true);
        run(false);
    }

    #[test]
    fn fixed_guided_peeling_tries_each_physical_product_candidate_without_mixing_factors() {
        fn run(nested: bool, reverse_nodes: bool) -> Vec<PeelTerm> {
            let mut egraph = EGraph::new(MxxAnalysis::default());
            let (p, _) = matrix_atom(&mut egraph, "peel-product-candidate-p", None);
            let (a, _) = matrix_atom(&mut egraph, "peel-product-candidate-a", None);
            let (noise, _) = matrix_atom(&mut egraph, "peel-product-candidate-noise", None);
            let (q, _) = matrix_atom(&mut egraph, "peel-product-candidate-q", None);
            let (r, _) = matrix_atom(&mut egraph, "peel-product-candidate-r", None);
            let (d, _) = matrix_atom(&mut egraph, "peel-product-candidate-d", None);
            let grouped = egraph.add(MxxLang::MatrixAdd(vec![a, noise].into_boxed_slice()));
            let (actual, competing_product) = if nested {
                let good = egraph.add(MxxLang::MatrixMultiply(vec![p, grouped].into_boxed_slice()));
                let bad = egraph.add(MxxLang::MatrixMultiply(vec![q, r].into_boxed_slice()));
                egraph.union(good, bad);
                (egraph.add(MxxLang::MatrixMultiply(vec![good, d].into_boxed_slice())), good)
            } else {
                let good =
                    egraph.add(MxxLang::MatrixMultiply(vec![p, grouped, d].into_boxed_slice()));
                let bad = egraph.add(MxxLang::MatrixMultiply(vec![q, r, d].into_boxed_slice()));
                egraph.union(good, bad);
                (good, good)
            };
            egraph.rebuild();
            let competing_product = egraph.find(competing_product);
            assert_eq!(
                egraph[competing_product]
                    .nodes
                    .iter()
                    .filter(|node| matches!(node, MxxLang::MatrixMultiply(_)))
                    .count(),
                2,
                "both physical product alternatives are retained"
            );
            if reverse_nodes {
                egraph[competing_product].nodes.reverse();
            }

            let original = vec![PeelTerm::Concrete { base: actual, negative: false }];
            let target = vec![egraph.find(p), egraph.find(a), egraph.find(d)].into_boxed_slice();
            let mut terms = original.clone();
            let mut full_calls = 0;
            assert_eq!(
                peel_fixed_targets(&egraph, &mut terms, &[(target.clone(), true)], &mut || {
                    full_calls += 1;
                    Ok(())
                }),
                Some((true, Vec::new()))
            );
            assert!(matches!(
                terms.as_slice(),
                [PeelTerm::ProductFactor { prefix, terms, suffix, negative: false }]
                    if prefix.as_ref() == [egraph.find(p)]
                        && terms == &vec![(egraph.find(noise), false)]
                        && suffix.as_ref() == [egraph.find(d)]
            ));

            let reordered = vec![egraph.find(a), egraph.find(p), egraph.find(d)].into_boxed_slice();
            let mut reordered_terms = original.clone();
            let mut progress = || Ok(());
            assert_eq!(
                peel_fixed_targets(
                    &egraph,
                    &mut reordered_terms,
                    &[(reordered.clone(), true)],
                    &mut progress
                ),
                Some((false, vec![(reordered, true)])),
                "a target cannot combine the p and a factors in reversed order"
            );
            assert_eq!(reordered_terms, original);

            let before = egraph.total_size();
            let mut interrupted = original;
            let mut calls = 0;
            assert!(
                peel_fixed_targets(&egraph, &mut interrupted, &[(target, true)], &mut || {
                    calls += 1;
                    (calls < full_calls).then_some(()).ok_or(())
                })
                .is_none(),
                "the final charged operation interrupts before any candidate plan commits"
            );
            assert_eq!(interrupted, vec![PeelTerm::Concrete { base: actual, negative: false }]);
            assert_eq!(egraph.total_size(), before, "physical candidates are read-only");
            terms
        }

        for nested in [false, true] {
            assert_eq!(
                run(nested, false),
                run(nested, true),
                "the minimum residual is independent of physical product e-node order"
            );
        }
    }

    #[test]
    fn fixed_guided_physical_product_candidates_choose_the_smallest_signed_residual() {
        fn run(reverse_nodes: bool) -> Vec<PeelTerm> {
            let mut egraph = EGraph::new(MxxAnalysis::default());
            let (p, _) = matrix_atom(&mut egraph, "peel-product-smallest-p", None);
            let (a, _) = matrix_atom(&mut egraph, "peel-product-smallest-a", None);
            let (short_noise, _) = matrix_atom(&mut egraph, "peel-product-smallest-short", None);
            let (wide_one, _) = matrix_atom(&mut egraph, "peel-product-smallest-wide-one", None);
            let (wide_two, _) = matrix_atom(&mut egraph, "peel-product-smallest-wide-two", None);
            let (d, _) = matrix_atom(&mut egraph, "peel-product-smallest-d", None);
            let short_sum =
                egraph.add(MxxLang::MatrixAdd(vec![a, a, short_noise].into_boxed_slice()));
            let wide_sum =
                egraph.add(MxxLang::MatrixAdd(vec![a, a, wide_one, wide_two].into_boxed_slice()));
            let short = egraph.add(MxxLang::MatrixNegate([short_sum]));
            let wide = egraph.add(MxxLang::MatrixNegate([wide_sum]));
            let short = egraph.add(MxxLang::MatrixMultiply(vec![p, short, d].into_boxed_slice()));
            let wide = egraph.add(MxxLang::MatrixMultiply(vec![p, wide, d].into_boxed_slice()));
            egraph.union(short, wide);
            egraph.rebuild();
            let actual = egraph.find(short);
            if reverse_nodes {
                egraph[actual].nodes.reverse();
            }

            let target = vec![egraph.find(p), egraph.find(a), egraph.find(d)].into_boxed_slice();
            let mut terms = vec![PeelTerm::Concrete { base: actual, negative: false }];
            let mut progress = || Ok(());
            assert_eq!(
                peel_fixed_targets(
                    &egraph,
                    &mut terms,
                    &[(target.clone(), false), (target, false)],
                    &mut progress,
                ),
                Some((true, Vec::new()))
            );
            assert!(matches!(
                terms.as_slice(),
                [PeelTerm::ProductFactor { prefix, terms, suffix, negative: false }]
                    if prefix.as_ref() == [egraph.find(p)]
                        && terms == &vec![(egraph.find(short_noise), true)]
                        && suffix.as_ref() == [egraph.find(d)]
            ));
            terms
        }

        assert_eq!(run(false), run(true));
    }

    #[test]
    fn fixed_guided_physical_product_cycles_are_atomic_and_interruptible() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (seed, _) = matrix_atom(&mut egraph, "peel-product-cycle-seed", None);
        let (target_atom, _) = matrix_atom(&mut egraph, "peel-product-cycle-target", None);
        let direct = egraph.add(MxxLang::MatrixMultiply(vec![seed].into_boxed_slice()));
        let indirect = egraph.add(MxxLang::MatrixMultiply(vec![direct].into_boxed_slice()));
        egraph.union(seed, indirect);
        egraph.rebuild();
        let actual = egraph.find(seed);
        let target = (vec![egraph.find(target_atom)].into_boxed_slice(), true);
        let original = vec![PeelTerm::Concrete { base: actual, negative: false }];
        let before = egraph.total_size();
        let mut full_calls = 0;
        let mut funded = original.clone();
        assert_eq!(
            peel_fixed_targets(&egraph, &mut funded, std::slice::from_ref(&target), &mut || {
                full_calls += 1;
                Ok(())
            }),
            Some((false, vec![target.clone()]))
        );
        assert_eq!(funded, original, "cyclic physical alternatives stay atomic");
        assert!(full_calls < 128, "direct and indirect cycles stop locally");

        let mut interrupted = original.clone();
        let mut calls = 0;
        assert!(
            peel_fixed_targets(&egraph, &mut interrupted, &[target], &mut || {
                calls += 1;
                (calls < full_calls).then_some(()).ok_or(())
            })
            .is_none(),
            "an interrupted cyclic candidate scan cannot commit"
        );
        assert_eq!(interrupted, original);
        assert_eq!(egraph.total_size(), before);
    }

    #[test]
    fn fixed_guided_physical_product_candidates_scale_linearly() {
        fn charges_for(candidate_count: usize) -> usize {
            let mut egraph = EGraph::new(MxxAnalysis::default());
            let (p, _) = matrix_atom(&mut egraph, "peel-product-linear-p", None);
            let (a, _) = matrix_atom(&mut egraph, "peel-product-linear-a", None);
            let (noise, _) = matrix_atom(&mut egraph, "peel-product-linear-noise", None);
            let (d, _) = matrix_atom(&mut egraph, "peel-product-linear-d", None);
            let grouped = egraph.add(MxxLang::MatrixAdd(vec![a, noise].into_boxed_slice()));
            let actual =
                egraph.add(MxxLang::MatrixMultiply(vec![p, grouped, d].into_boxed_slice()));
            for index in 1..candidate_count {
                let (left, _) = matrix_atom(
                    &mut egraph,
                    &format!("peel-product-linear-left-{candidate_count}-{index}"),
                    None,
                );
                let (right, _) = matrix_atom(
                    &mut egraph,
                    &format!("peel-product-linear-right-{candidate_count}-{index}"),
                    None,
                );
                let alternative =
                    egraph.add(MxxLang::MatrixMultiply(vec![left, right, d].into_boxed_slice()));
                egraph.union(actual, alternative);
            }
            egraph.rebuild();
            let target = vec![egraph.find(p), egraph.find(a), egraph.find(d)].into_boxed_slice();
            let mut terms = vec![PeelTerm::Concrete { base: actual, negative: false }];
            let mut charges = 0;
            assert_eq!(
                peel_fixed_targets(&egraph, &mut terms, &[(target, true)], &mut || {
                    charges += 1;
                    Ok(())
                }),
                Some((true, Vec::new()))
            );
            charges
        }

        let eight = charges_for(8);
        let sixteen = charges_for(16);
        assert!(eight > 0);
        assert!(
            sixteen < eight.saturating_mul(3),
            "independent physical candidates must not form a Cartesian product: {eight} -> {sixteen}"
        );
    }

    #[test]
    fn fixed_guided_peeling_chooses_the_smallest_physical_factor_candidate() {
        fn run(grouped_first: bool) {
            let mut egraph = EGraph::new(MxxAnalysis::default());
            let (left, _) = matrix_atom(&mut egraph, "peel-factor-candidate-left", None);
            let (right, _) = matrix_atom(&mut egraph, "peel-factor-candidate-right", None);
            let (residual, _) = matrix_atom(&mut egraph, "peel-factor-candidate-residual", None);
            let (wide_one, _) = matrix_atom(&mut egraph, "peel-factor-candidate-wide-one", None);
            let (wide_two, _) = matrix_atom(&mut egraph, "peel-factor-candidate-wide-two", None);
            let (wide_three, _) =
                matrix_atom(&mut egraph, "peel-factor-candidate-wide-three", None);
            let (suffix, _) = matrix_atom(&mut egraph, "peel-factor-candidate-suffix", None);
            let signal = egraph.add(MxxLang::MatrixMultiply(vec![left, right].into_boxed_slice()));
            let (first, second) = if grouped_first {
                (
                    egraph.add(MxxLang::MatrixAdd(vec![signal, residual].into_boxed_slice())),
                    egraph.add(MxxLang::MatrixAdd(
                        vec![signal, wide_one, wide_two, wide_three].into_boxed_slice(),
                    )),
                )
            } else {
                (
                    egraph.add(MxxLang::MatrixAdd(
                        vec![signal, wide_one, wide_two, wide_three].into_boxed_slice(),
                    )),
                    egraph.add(MxxLang::MatrixAdd(vec![signal, residual].into_boxed_slice())),
                )
            };
            egraph.union(first, second);
            let actual =
                egraph.add(MxxLang::MatrixMultiply(vec![first, suffix].into_boxed_slice()));
            egraph.rebuild();
            let factor = egraph.find(first);
            let physical_adds = egraph[factor]
                .nodes
                .iter()
                .filter_map(|node| match node {
                    MxxLang::MatrixAdd(children) => {
                        Some(children.iter().map(|child| egraph.find(*child)).collect::<Vec<_>>())
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(physical_adds.len(), 2, "both successful physical factor candidates remain");
            if !grouped_first {
                egraph[factor].nodes.reverse();
                let reversed = egraph[factor]
                    .nodes
                    .iter()
                    .filter_map(|node| match node {
                        MxxLang::MatrixAdd(children) => Some(
                            children.iter().map(|child| egraph.find(*child)).collect::<Vec<_>>(),
                        ),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                assert_eq!(reversed, physical_adds.iter().cloned().rev().collect::<Vec<_>>());
            }

            let mut terms = vec![PeelTerm::Concrete { base: actual, negative: false }];
            let mut progress = || Ok(());
            assert_eq!(
                peel_fixed_targets(
                    &egraph,
                    &mut terms,
                    &[(
                        vec![egraph.find(left), egraph.find(right), egraph.find(suffix),]
                            .into_boxed_slice(),
                        true
                    )],
                    &mut progress,
                ),
                Some((true, Vec::new()))
            );
            assert!(matches!(
                terms.as_slice(),
                [PeelTerm::ProductFactor { prefix, terms, suffix: remaining_suffix, negative: false }]
                    if prefix.is_empty()
                        && terms == &vec![(egraph.find(residual), false)]
                        && remaining_suffix.as_ref() == [egraph.find(suffix)]
            ));
        }
        run(true);
        run(false);
    }

    #[test]
    fn fixed_guided_peeling_reopens_a_nested_product_factor_for_later_targets() {
        fn run(negated: bool) {
            let mut egraph = EGraph::new(MxxAnalysis::default());
            let (a, _) = matrix_atom(&mut egraph, "peel-later-factor-a", None);
            let (b, _) = matrix_atom(&mut egraph, "peel-later-factor-b", None);
            let (noise, _) = matrix_atom(&mut egraph, "peel-later-factor-noise", None);
            let (d, _) = matrix_atom(&mut egraph, "peel-later-factor-d", None);
            let grouped = egraph.add(MxxLang::MatrixAdd(vec![a, noise].into_boxed_slice()));
            let grouped =
                if negated { egraph.add(MxxLang::MatrixNegate([grouped])) } else { grouped };
            let sum = egraph.add(MxxLang::MatrixAdd(vec![b, grouped].into_boxed_slice()));
            let actual = egraph.add(MxxLang::MatrixMultiply(vec![sum, d].into_boxed_slice()));
            egraph.rebuild();

            let mut terms = vec![PeelTerm::Concrete { base: actual, negative: false }];
            let first = (vec![egraph.find(b), egraph.find(d)].into_boxed_slice(), true);
            let second = (vec![egraph.find(a), egraph.find(d)].into_boxed_slice(), !negated);
            let mut progress = || Ok(());
            assert_eq!(
                peel_fixed_targets(&egraph, &mut terms, &[first, second], &mut progress),
                Some((true, Vec::new()))
            );
            assert!(matches!(
                terms.as_slice(),
                [PeelTerm::ProductFactor { prefix, terms, suffix, negative: false }]
                    if prefix.is_empty()
                        && terms == &vec![(egraph.find(noise), negated)]
                        && suffix.as_ref() == [egraph.find(d)]
            ));
        }
        run(false);
        run(true);
    }

    #[test]
    fn fixed_guided_residual_ranking_is_order_independent_and_interruptible() {
        let wide_term = PeelTerm::ProductFactor {
            prefix: Box::default(),
            terms: (0..16).map(|index| (Id::from(index), false)).collect(),
            suffix: Box::default(),
            negative: false,
        };
        let other_wide_term = PeelTerm::ProductFactor {
            prefix: Box::default(),
            terms: (16..32).map(|index| (Id::from(index), false)).collect(),
            suffix: Box::default(),
            negative: false,
        };
        let mut best = Some(vec![wide_term.clone()]);
        let mut full_calls = 0;
        retain_best_residual(&mut best, vec![other_wide_term.clone()], &mut || {
            full_calls += 1;
            Ok(())
        })
        .unwrap();
        assert!(full_calls > 16, "whole-residual ranking charges product-factor ids and signs");
        let original = Some(vec![wide_term]);
        let mut interrupted = original.clone();
        let mut calls = 0;
        assert!(
            retain_best_residual(&mut interrupted, vec![other_wide_term], &mut || {
                calls += 1;
                (calls < full_calls).then_some(()).ok_or(())
            })
            .is_none()
        );
        assert_eq!(interrupted, original, "whole-residual ranking is transactional too");
    }

    #[test]
    fn fixed_guided_physical_add_candidates_preserve_order_sign_and_transactionality() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (left, _) = matrix_atom(&mut egraph, "peel-candidate-order-left", None);
        let (right, _) = matrix_atom(&mut egraph, "peel-candidate-order-right", None);
        let (noise, _) = matrix_atom(&mut egraph, "peel-candidate-order-noise", None);
        let reversed = egraph.add(MxxLang::MatrixMultiply(vec![right, left].into_boxed_slice()));
        let bad = egraph.add(MxxLang::MatrixAdd(vec![reversed, noise].into_boxed_slice()));
        let before = egraph.total_size();
        let original = vec![PeelTerm::Concrete { base: bad, negative: false }];
        let mut actual = original.clone();
        let mut progress = || Ok(());
        assert_eq!(
            peel_fixed_targets(
                &egraph,
                &mut actual,
                &[(vec![egraph.find(left), egraph.find(right)].into_boxed_slice(), true)],
                &mut progress,
            ),
            Some((
                false,
                vec![(vec![egraph.find(left), egraph.find(right)].into_boxed_slice(), true)]
            ))
        );
        assert_eq!(actual, original, "noncommutative reordering cannot be peeled");
        assert_eq!(egraph.total_size(), before);

        let signal = egraph.add(MxxLang::MatrixMultiply(vec![left, right].into_boxed_slice()));
        let grouped = egraph.add(MxxLang::MatrixAdd(vec![signal, noise].into_boxed_slice()));
        let negated = egraph.add(MxxLang::MatrixNegate([grouped]));
        let mut actual = vec![PeelTerm::Concrete { base: negated, negative: false }];
        let mut progress = || Ok(());
        assert_eq!(
            peel_fixed_targets(
                &egraph,
                &mut actual,
                &[(vec![egraph.find(left), egraph.find(right)].into_boxed_slice(), false)],
                &mut progress,
            ),
            Some((true, Vec::new()))
        );
        assert_eq!(
            actual,
            vec![PeelTerm::Concrete { base: egraph.find(noise), negative: true }],
            "a direct Negate candidate toggles only the selected additive witness"
        );
    }

    #[test]
    fn fixed_guided_physical_add_candidates_skip_a_cyclic_or_unfunded_candidate() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (left, _) = matrix_atom(&mut egraph, "peel-candidate-cycle-left", None);
        let (right, _) = matrix_atom(&mut egraph, "peel-candidate-cycle-right", None);
        let (noise, _) = matrix_atom(&mut egraph, "peel-candidate-cycle-noise", None);
        let signal = egraph.add(MxxLang::MatrixMultiply(vec![left, right].into_boxed_slice()));
        let valid = egraph.add(MxxLang::MatrixAdd(vec![signal, noise].into_boxed_slice()));
        let (seed, _) = matrix_atom(&mut egraph, "peel-candidate-cycle-seed", None);
        let cyclic = egraph.add(MxxLang::MatrixAdd(vec![seed].into_boxed_slice()));
        egraph.union(cyclic, seed);
        egraph.union(seed, valid);

        let original = vec![PeelTerm::Concrete { base: cyclic, negative: false }];
        let target = (vec![egraph.find(left), egraph.find(right)].into_boxed_slice(), true);
        let mut funded = original.clone();
        let mut full_calls = 0;
        assert!(
            peel_fixed_targets(&egraph, &mut funded, std::slice::from_ref(&target), &mut || {
                full_calls += 1;
                Ok(())
            })
            .is_some()
        );
        assert_eq!(
            funded,
            vec![PeelTerm::Concrete { base: egraph.find(noise), negative: false }],
            "a cyclic earlier Add witness cannot mask a later valid Add witness"
        );

        let before = egraph.total_size();
        let mut interrupted = original.clone();
        let mut calls = 0;
        assert!(
            peel_fixed_targets(&egraph, &mut interrupted, &[target], &mut || {
                calls += 1;
                (calls < full_calls / 2).then_some(()).ok_or(())
            })
            .is_none(),
            "the calibrated midpoint interrupts while enumerating the physical candidates"
        );
        assert_eq!(
            interrupted, original,
            "candidate interruption cannot commit a partial residual"
        );
        assert_eq!(egraph.total_size(), before, "candidate enumeration is read-only");
    }

    #[test]
    fn fixed_guided_peeling_is_linear_in_a_grouped_additive_case() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let mut children = Vec::new();
        for index in 0..128 {
            children.push(matrix_atom(&mut egraph, &format!("peel-many-{index}"), None).0);
        }
        let grouped = egraph.add(MxxLang::MatrixAdd(children.clone().into_boxed_slice()));
        egraph.rebuild();
        let mut terms = vec![PeelTerm::Concrete { base: grouped, negative: false }];
        let mut charges = 0;
        assert_eq!(
            peel_fixed_targets(
                &egraph,
                &mut terms,
                &[(vec![egraph.find(children[64])].into_boxed_slice(), true)],
                &mut || {
                    charges += 1;
                    Ok(())
                },
            ),
            Some((true, Vec::new()))
        );
        assert_eq!(terms.len(), 127);
        assert!(charges < 5_000, "one target scans/copies the grouped case linearly");
    }

    #[test]
    fn fixed_guided_peeling_stops_before_unfunded_second_peel_or_fixed_cancellation() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (p, _) = matrix_atom(&mut egraph, "peel-budget-prefix", None);
        let (a, _) = matrix_atom(&mut egraph, "peel-budget-a", None);
        let (b, _) = matrix_atom(&mut egraph, "peel-budget-b", None);
        let (c, _) = matrix_atom(&mut egraph, "peel-budget-c", None);
        let (s, _) = matrix_atom(&mut egraph, "peel-budget-suffix", None);
        let sum = egraph.add(MxxLang::MatrixAdd(vec![a, b, c].into_boxed_slice()));
        let actual = egraph.add(MxxLang::MatrixMultiply(vec![p, sum, s].into_boxed_slice()));
        egraph.rebuild();
        let mut terms = vec![PeelTerm::Concrete { base: actual, negative: false }];
        let mut progress = || Ok(());
        assert_eq!(
            peel_fixed_targets(
                &egraph,
                &mut terms,
                &[(vec![egraph.find(p), egraph.find(a), egraph.find(s)].into_boxed_slice(), true)],
                &mut progress,
            ),
            Some((true, Vec::new()))
        );
        let before = egraph.total_size();
        let original_terms = terms.clone();
        let second_target = vec![egraph.find(p), egraph.find(b), egraph.find(s)].into_boxed_slice();
        let copy_probe = vec![egraph.find(p); 8];
        let mut full_copy_calls = 0;
        assert!(
            copy_ids(&copy_probe, &mut || {
                full_copy_calls += 1;
                Ok(())
            })
            .is_some()
        );
        assert_eq!(full_copy_calls, 8, "one progress callback guards each copied id");
        let mut copy_calls = 0;
        assert!(
            copy_ids(&copy_probe, &mut || {
                copy_calls += 1;
                (copy_calls < 5).then_some(()).ok_or(())
            })
            .is_none(),
            "the fifth callback interrupts the middle of an eight-id copy"
        );
        assert_eq!(copy_calls, 5);

        let mut full_peel_calls = 0;
        let mut funded_candidate = original_terms.clone();
        assert!(
            peel_fixed_targets(
                &egraph,
                &mut funded_candidate,
                &[(second_target.clone(), true)],
                &mut || {
                    full_peel_calls += 1;
                    Ok(())
                },
            )
            .is_some()
        );
        let mut candidate = original_terms.clone();
        let mut peel_calls = 0;
        assert!(
            peel_fixed_targets(
                &egraph,
                &mut candidate,
                &[(second_target.clone(), true)],
                &mut || {
                    peel_calls += 1;
                    (peel_calls < full_peel_calls - 1).then_some(()).ok_or(())
                },
            )
            .is_none(),
            "the calibrated late callback interrupts the ProductFactor peel"
        );
        assert_eq!(candidate, original_terms, "interrupted peeling keeps the plan unchanged");
        assert_eq!(egraph.total_size(), before, "interrupted planning creates no e-nodes");

        let nested = egraph.add(MxxLang::MatrixMultiply(vec![p, a].into_boxed_slice()));
        let nested = egraph.add(MxxLang::MatrixMultiply(vec![nested, b].into_boxed_slice()));
        egraph.rebuild();
        let after_nested = egraph.total_size();
        let target = vec![egraph.find(p), egraph.find(a), egraph.find(b)];
        let mut full_scan_calls = 0;
        assert!(
            direct_signed_target_positions(&egraph, nested, 0, false, &target, &mut || {
                full_scan_calls += 1;
                Ok(())
            })
            .is_some()
        );
        assert!(full_scan_calls > 8, "nested product reaches the structural DP phase");
        let mut scan_calls = 0;
        assert!(
            direct_signed_target_positions(&egraph, nested, 0, false, &target, &mut || {
                scan_calls += 1;
                (scan_calls < full_scan_calls / 2).then_some(()).ok_or(())
            })
            .is_none(),
            "the calibrated halfway callback interrupts the direct Mul DP scan"
        );
        assert!(scan_calls > 2, "the DP setup callbacks were funded before interruption");

        let fixed =
            (0..16).map(|_| (vec![egraph.find(a)].into_boxed_slice(), false)).collect::<Vec<_>>();
        let mut full_fixed_calls = 0;
        assert!(
            cancel_fixed_spines(fixed.clone(), &mut || {
                full_fixed_calls += 1;
                Ok(())
            })
            .is_some()
        );
        let mut fixed_calls = 0;
        assert!(
            cancel_fixed_spines(fixed, &mut || {
                fixed_calls += 1;
                (fixed_calls < full_fixed_calls - 1).then_some(()).ok_or(())
            })
            .is_none(),
            "the calibrated final callback interrupts fixed-map output materialization"
        );
        assert_eq!(egraph.total_size(), after_nested, "all interrupted paths are read-only");
    }

    #[test]
    fn fixed_guided_peeling_commits_only_after_every_fixed_target_is_funded() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (first, _) = matrix_atom(&mut egraph, "peel-transaction-first", None);
        let factors = (0..12)
            .map(|index| matrix_atom(&mut egraph, &format!("peel-transaction-{index}"), None).0)
            .collect::<Vec<_>>();
        let product = egraph.add(MxxLang::MatrixMultiply(factors.clone().into_boxed_slice()));
        let actual = egraph.add(MxxLang::MatrixAdd(vec![first, product].into_boxed_slice()));
        egraph.rebuild();
        let original = vec![PeelTerm::Concrete { base: actual, negative: false }];
        let first_target = (vec![egraph.find(first)].into_boxed_slice(), true);
        let second_target = (
            factors
                .iter()
                .map(|factor| egraph.find(*factor))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            true,
        );

        let mut first_only_calls = 0;
        let mut first_only = original.clone();
        assert!(
            peel_fixed_targets(
                &egraph,
                &mut first_only,
                std::slice::from_ref(&first_target),
                &mut || {
                    first_only_calls += 1;
                    Ok(())
                },
            )
            .is_some()
        );

        let mut full_calls = 0;
        let mut funded = original.clone();
        assert!(
            peel_fixed_targets(
                &egraph,
                &mut funded,
                &[first_target.clone(), second_target.clone()],
                &mut || {
                    full_calls += 1;
                    Ok(())
                },
            )
            .is_some()
        );
        let interruption = full_calls - 1;
        assert!(
            interruption > first_only_calls,
            "the final funded callback belongs to the large second-target plan"
        );

        let before = egraph.total_size();
        let mut interrupted = original.clone();
        let mut calls = 0;
        assert!(
            peel_fixed_targets(
                &egraph,
                &mut interrupted,
                &[first_target, second_target],
                &mut || {
                    calls += 1;
                    (calls < interruption).then_some(()).ok_or(())
                },
            )
            .is_none(),
            "the second target is interrupted after the first target has been fully planned"
        );
        assert_eq!(
            interrupted, original,
            "a failed later target cannot partially commit target one"
        );
        assert_eq!(egraph.total_size(), before, "transactional planning does not add e-nodes");
    }

    #[test]
    fn cancelled_signed_additive_leaves_associate_products_without_reordering() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (first, _) = matrix_atom(&mut egraph, "product-first", None);
        let (second, _) = matrix_atom(&mut egraph, "product-second", None);
        let (third, _) = matrix_atom(&mut egraph, "product-third", None);
        let left = egraph.add(MxxLang::MatrixMultiply(vec![first, second].into_boxed_slice()));
        let left_associated =
            egraph.add(MxxLang::MatrixMultiply(vec![left, third].into_boxed_slice()));
        let right = egraph.add(MxxLang::MatrixMultiply(vec![second, third].into_boxed_slice()));
        let right_associated =
            egraph.add(MxxLang::MatrixMultiply(vec![first, right].into_boxed_slice()));
        let reordered =
            egraph.add(MxxLang::MatrixMultiply(vec![second, first, third].into_boxed_slice()));
        egraph.rebuild();
        let before = egraph.total_size();

        assert_eq!(
            ordered_product_leaves(&egraph, left_associated),
            ordered_product_leaves(&egraph, right_associated),
        );
        assert_ne!(
            ordered_product_leaves(&egraph, left_associated),
            ordered_product_leaves(&egraph, reordered),
        );
        assert_eq!(
            cancelled_signed_additive_leaves(
                &egraph,
                &[(left_associated, false), (right_associated, true)],
            ),
            (vec![true, true], true),
        );
        assert_eq!(
            cancelled_signed_additive_leaves(
                &egraph,
                &[(left_associated, false), (reordered, true)],
            ),
            (vec![false, false], false),
        );
        assert_eq!(
            cancelled_signed_additive_leaves(
                &egraph,
                &[(left_associated, false), (left_associated, false), (right_associated, true),],
            ),
            (vec![false, true, true], true),
        );
        assert_eq!(egraph.total_size(), before, "cancellation key is read-only");
    }

    #[test]
    fn cancelled_signed_additive_leaves_reject_ambiguous_or_cyclic_products() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (first, _) = matrix_atom(&mut egraph, "ambiguous-product-first", None);
        let (second, _) = matrix_atom(&mut egraph, "ambiguous-product-second", None);
        let (third, _) = matrix_atom(&mut egraph, "ambiguous-product-third", None);
        let first_product =
            egraph.add(MxxLang::MatrixMultiply(vec![first, second].into_boxed_slice()));
        let second_product =
            egraph.add(MxxLang::MatrixMultiply(vec![first, third].into_boxed_slice()));
        egraph.union(first_product, second_product);
        egraph.rebuild();
        let cyclic = egraph.add(MxxLang::MatrixMultiply(vec![third].into_boxed_slice()));
        egraph.union(cyclic, third);

        assert_eq!(ordered_product_leaves(&egraph, first_product), None);
        assert_eq!(ordered_product_leaves(&egraph, cyclic), None);
        assert_eq!(
            cancelled_signed_additive_leaves(
                &egraph,
                &[(first_product, false), (first_product, true), (cyclic, false), (cyclic, true),],
            ),
            (vec![false, false, false, false], false),
        );
    }

    #[test]
    fn binder_distribution_stabilizes_nodes_and_charges_shared_work() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let binder = test_binder(&mut egraph, 0, 1);
        let selector = egraph.add(MxxLang::IntBinder(binder));
        let shared = binder_matrix_atom(&mut egraph, selector, "failed-apply");
        let zero = egraph.add(MxxLang::IntConst(0.into()));
        let one = egraph.add(MxxLang::IntConst(1.into()));
        let at_zero =
            family::instantiate_shared_element(&mut egraph, shared, binder, zero, &mut || {
                Ok::<(), ()>(())
            })
            .expect("test instantiation");
        let at_one =
            family::instantiate_shared_element(&mut egraph, shared, binder, one, &mut || {
                Ok::<(), ()>(())
            })
            .expect("test instantiation");
        let root = symbolic_two_case_root(&mut egraph, selector, shared, at_one, at_zero, 1);
        egraph.rebuild();
        let owned = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let context =
            RewriteContext::new(SharedRewriteBudget::from_shared(std::sync::Arc::clone(&owned)));
        let applier = RelationApplier::new(context);
        let invoke = |egraph: &mut EGraph<MxxLang, MxxAnalysis>| {
            Applier::apply_one(
                &applier,
                egraph,
                root,
                &Subst::default(),
                None,
                Symbol::from("binder-failure"),
            )
        };
        assert!(invoke(&mut egraph).is_empty());
        egraph.rebuild();
        let nodes_after_first = egraph.total_size();
        let charged_after_first = owned.load(std::sync::atomic::Ordering::Relaxed);
        assert!(charged_after_first > 1, "apply and instantiation visits are charged");
        assert!(invoke(&mut egraph).is_empty());
        egraph.rebuild();
        let charged_after_second = owned.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            egraph.total_size(),
            nodes_after_first,
            "equivalent retries reuse interned nodes"
        );
        assert!(charged_after_second > charged_after_first);
        assert!(charged_after_second - charged_after_first <= charged_after_first);
    }

    #[test]
    fn pointwise_add_switch_diagnostic_distinguishes_direct_switch_shapes() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let (signal, _) = matrix_atom(&mut egraph, "signal", None);
        let negated = egraph.add(MxxLang::MatrixNegate([signal]));

        let absent = egraph.add(MxxLang::MatrixAdd(vec![signal, negated].into_boxed_slice()));
        let switch = egraph.add(MxxLang::Switch(vec![selector, signal, signal].into_boxed_slice()));
        let grouped = egraph.add(MxxLang::MatrixAdd(vec![switch, negated].into_boxed_slice()));
        let hidden = egraph.add(MxxLang::MatrixAdd(vec![grouped].into_boxed_slice()));
        let second_switch =
            egraph.add(MxxLang::Switch(vec![selector, signal, signal].into_boxed_slice()));
        let multiple =
            egraph.add(MxxLang::MatrixAdd(vec![switch, second_switch, negated].into_boxed_slice()));
        egraph.rebuild();
        let before = egraph.total_size();

        assert_eq!(
            pointwise_add_switch_cancellation_reason(&egraph, absent),
            Err(PointwiseAddSwitchReject::Structural {
                physical_root_adds: 1,
                eligible_single_switch_adds: 0,
                direct_switch_children: 0,
                direct_grouped_add_children: 0,
            })
        );
        assert_eq!(
            pointwise_add_switch_cancellation_reason(&egraph, hidden),
            Err(PointwiseAddSwitchReject::Structural {
                physical_root_adds: 1,
                eligible_single_switch_adds: 0,
                direct_switch_children: 0,
                direct_grouped_add_children: 1,
            })
        );
        assert_eq!(
            pointwise_add_switch_cancellation_reason(&egraph, multiple),
            Err(PointwiseAddSwitchReject::Structural {
                physical_root_adds: 1,
                eligible_single_switch_adds: 0,
                direct_switch_children: 2,
                direct_grouped_add_children: 0,
            })
        );
        assert_eq!(egraph.total_size(), before);
    }

    #[test]
    fn pointwise_add_switch_rejects_one_failed_case_and_does_not_repeat_existing_result() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let (signal, _) = matrix_atom(&mut egraph, "signal", None);
        let (other, _) = matrix_atom(&mut egraph, "other", None);
        let negated = egraph.add(MxxLang::MatrixNegate([signal]));
        let failing = egraph.add(MxxLang::Switch(vec![selector, signal, other].into_boxed_slice()));
        let failing_root =
            egraph.add(MxxLang::MatrixAdd(vec![failing, negated].into_boxed_slice()));
        egraph.rebuild();
        let before = egraph.total_size();
        assert!(matches!(
            pointwise_add_switch_cancellation_reason(&egraph, failing_root),
            Err(PointwiseAddSwitchReject::UnmatchedFixedTerms {
                case_index: 1,
                matched: 0,
                required: 1,
                direct_terms: 1,
                negated_terms: 0,
                fixed_unique_add_children: 0,
                case_physical_adds: 0,
                case_grouped_add_children: 0,
                ..
            })
        ));
        assert_eq!(egraph.total_size(), before);

        let passing =
            egraph.add(MxxLang::Switch(vec![selector, signal, signal].into_boxed_slice()));
        let root = egraph.add(MxxLang::MatrixAdd(vec![passing, negated].into_boxed_slice()));
        egraph.rebuild();
        let plan =
            pointwise_add_switch_cancellation_plan(&egraph, root).expect("first normalization");
        let replacement = build_pointwise_add_switch_cancellation(&mut egraph, root, plan)
            .expect("direct plan builds");
        egraph.union(root, replacement);
        egraph.rebuild();
        let before = egraph.total_size();
        assert!(pointwise_add_switch_cancellation_plan(&egraph, root).is_none());
        assert_eq!(egraph.total_size(), before);
    }

    #[test]
    fn pointwise_add_switch_failure_reports_signed_canonical_multiplicities() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let (signal, _) = matrix_atom(&mut egraph, "signal", None);
        let (other, _) = matrix_atom(&mut egraph, "other", None);
        let negative_signal = egraph.add(MxxLang::MatrixNegate([signal]));
        let negative_other = egraph.add(MxxLang::MatrixNegate([other]));
        let first_case = egraph.add(MxxLang::MatrixAdd(vec![signal, negative_other].into()));
        let switch = egraph.add(MxxLang::Switch(vec![selector, first_case].into_boxed_slice()));
        let root = egraph.add(MxxLang::MatrixAdd(
            vec![switch, negative_signal, negative_signal, other].into_boxed_slice(),
        ));
        egraph.rebuild();

        let Err(PointwiseAddSwitchReject::UnmatchedFixedTerms {
            case_index,
            matched,
            required,
            fixed_terms,
            fixed_terms_omitted_occurrences,
            case_terms,
            case_terms_omitted_occurrences,
            ..
        }) = pointwise_add_switch_cancellation_reason(&egraph, root)
        else {
            panic!("one fixed signal occurrence must remain unmatched");
        };
        assert_eq!(case_index, 0);
        assert_eq!((matched, required), (2, 3));
        let mut expected_fixed = vec![
            SignedCanonicalMultiplicity {
                eclass: usize::from(egraph.find(signal)),
                negative: true,
                multiplicity: 2,
            },
            SignedCanonicalMultiplicity {
                eclass: usize::from(egraph.find(other)),
                negative: false,
                multiplicity: 1,
            },
        ];
        expected_fixed.sort_unstable();
        let mut expected_case = vec![
            SignedCanonicalMultiplicity {
                eclass: usize::from(egraph.find(signal)),
                negative: false,
                multiplicity: 1,
            },
            SignedCanonicalMultiplicity {
                eclass: usize::from(egraph.find(other)),
                negative: true,
                multiplicity: 1,
            },
        ];
        expected_case.sort_unstable();
        assert_eq!(fixed_terms.as_ref(), expected_fixed.as_slice());
        assert_eq!(case_terms.as_ref(), expected_case.as_slice());
        assert_eq!(fixed_terms_omitted_occurrences, 0);
        assert_eq!(case_terms_omitted_occurrences, 0);
    }

    #[test]
    fn pointwise_add_switch_failure_counts_competing_eligible_roots() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let (signal, _) = matrix_atom(&mut egraph, "signal", None);
        let (other, _) = matrix_atom(&mut egraph, "other", None);
        let negated_signal = egraph.add(MxxLang::MatrixNegate([signal]));
        let switch = egraph.add(MxxLang::Switch(vec![selector, signal, other].into_boxed_slice()));
        let first = egraph.add(MxxLang::MatrixAdd(vec![switch, negated_signal].into_boxed_slice()));
        let second =
            egraph.add(MxxLang::MatrixAdd(vec![negated_signal, switch].into_boxed_slice()));
        egraph.union(first, second);
        egraph.rebuild();

        assert!(matches!(
            pointwise_add_switch_cancellation_reason(&egraph, first),
            Err(PointwiseAddSwitchReject::UnmatchedFixedTerms {
                physical_root_adds: 2,
                eligible_single_switch_adds: 2,
                case_index: 1,
                matched: 0,
                required: 1,
                ..
            })
        ));
    }

    #[test]
    fn pointwise_add_switch_failure_caps_signed_identity_diagnostic() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let (case, _) = matrix_atom(&mut egraph, "case", None);
        let switch = egraph.add(MxxLang::Switch(vec![selector, case].into_boxed_slice()));
        let mut root_terms = vec![switch];
        let mut first_negated = None;
        for index in 0..16 {
            let (term, _) = matrix_atom(&mut egraph, &format!("fixed-{index}"), None);
            let negated = egraph.add(MxxLang::MatrixNegate([term]));
            if index == 0 {
                first_negated = Some(negated);
            }
            root_terms.push(negated);
        }
        root_terms.push(first_negated.expect("the first retained identity exists"));
        for index in 16..18 {
            let (term, _) = matrix_atom(&mut egraph, &format!("fixed-{index}"), None);
            root_terms.push(egraph.add(MxxLang::MatrixNegate([term])));
        }
        let root = egraph.add(MxxLang::MatrixAdd(root_terms.into_boxed_slice()));
        egraph.rebuild();

        let Err(PointwiseAddSwitchReject::UnmatchedFixedTerms {
            fixed_terms,
            fixed_terms_omitted_occurrences,
            case_terms,
            case_terms_omitted_occurrences,
            ..
        }) = pointwise_add_switch_cancellation_reason(&egraph, root)
        else {
            panic!("unmatched fixed terms must reject pointwise cancellation");
        };
        assert_eq!(fixed_terms.len(), MAX_UNMATCHED_TERM_IDENTITIES);
        assert_eq!(fixed_terms_omitted_occurrences, 2);
        assert!(fixed_terms.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(fixed_terms.iter().all(|term| term.negative));
        assert_eq!(fixed_terms.iter().map(|term| term.multiplicity).sum::<usize>(), 17);
        assert_eq!(
            case_terms.as_ref(),
            &[SignedCanonicalMultiplicity {
                eclass: usize::from(egraph.find(case)),
                negative: false,
                multiplicity: 1,
            }]
        );
        assert_eq!(case_terms_omitted_occurrences, 0);
    }

    #[test]
    fn pointwise_add_switch_visits_stored_cases_linearly() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let (signal, _) = matrix_atom(&mut egraph, "signal", None);
        let negated = egraph.add(MxxLang::MatrixNegate([signal]));
        let mut inner_cases = vec![selector];
        inner_cases.extend(std::iter::repeat_n(signal, 512));
        let inner = egraph.add(MxxLang::Switch(inner_cases.into_boxed_slice()));
        let mut outer_cases = vec![selector];
        outer_cases.extend(std::iter::repeat_n(inner, 512));
        let outer = egraph.add(MxxLang::Switch(outer_cases.into_boxed_slice()));
        let root = egraph.add(MxxLang::MatrixAdd(vec![outer, negated].into_boxed_slice()));
        egraph.rebuild();
        let before = egraph.total_size();
        let plan =
            pointwise_add_switch_cancellation_plan(&egraph, root).expect("all stored cases cancel");
        let replacement = build_pointwise_add_switch_cancellation(&mut egraph, root, plan)
            .expect("direct plan builds");
        assert!(
            matches!(egraph[egraph.find(replacement)].nodes.as_slice(), [MxxLang::Switch(cases)] if cases.len() == 513)
        );
        assert!(
            egraph.total_size() <= before + 513,
            "one switch plus at most one node per stored case"
        );
    }

    #[test]
    fn pointwise_add_switch_diagonalizes_same_selector_cases_and_negation() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = integer_atom(&mut egraph, "diagonal-selector");
        let selector_alias = integer_atom(&mut egraph, "diagonal-selector-alias");
        egraph.union(selector, selector_alias);
        egraph.rebuild();
        let (signal, _) = matrix_atom(&mut egraph, "diagonal-signal", None);
        let inner =
            egraph.add(MxxLang::Switch(vec![selector_alias, signal, signal].into_boxed_slice()));
        let outer = egraph.add(MxxLang::Switch(vec![selector, inner, inner].into_boxed_slice()));
        let negative_signal = egraph.add(MxxLang::MatrixNegate([signal]));
        let root = egraph.add(MxxLang::MatrixAdd(vec![outer, negative_signal].into_boxed_slice()));
        egraph.rebuild();

        let plan = pointwise_add_switch_cancellation_plan(&egraph, root).expect("same selector");
        let replacement =
            build_pointwise_add_switch_cancellation(&mut egraph, root, plan).expect("replacement");
        let cases = switch_node(&egraph, replacement).expect("result switch");
        assert!(egraph[egraph.find(cases[1])].nodes.iter().any(|node| {
            matches!(node, MxxLang::MatrixConstant(spec)
                if matches!(egraph.analysis.symbols.matrix_constants.get(spec.0), Some(super::super::identity::MatrixConstantSpec { value: MatrixConstantValue::Zero, .. })))
        }));
        assert!(egraph[egraph.find(cases[2])].nodes.iter().any(|node| {
            matches!(node, MxxLang::MatrixConstant(spec)
                if matches!(egraph.analysis.symbols.matrix_constants.get(spec.0), Some(super::super::identity::MatrixConstantSpec { value: MatrixConstantValue::Zero, .. })))
        }));

        let negative_inner = egraph.add(MxxLang::MatrixNegate([inner]));
        let negative_outer = egraph.add(MxxLang::Switch(
            vec![selector, negative_inner, negative_inner].into_boxed_slice(),
        ));
        let negated_root =
            egraph.add(MxxLang::MatrixAdd(vec![negative_outer, signal].into_boxed_slice()));
        egraph.rebuild();
        let plan = pointwise_add_switch_cancellation_plan(&egraph, negated_root)
            .expect("negated selector");
        let replacement = build_pointwise_add_switch_cancellation(&mut egraph, negated_root, plan)
            .expect("replacement");
        let cases = switch_node(&egraph, replacement).expect("result switch");
        assert!(egraph[egraph.find(cases[1])].nodes.iter().any(|node| {
            matches!(node, MxxLang::MatrixConstant(spec)
                if matches!(egraph.analysis.symbols.matrix_constants.get(spec.0), Some(super::super::identity::MatrixConstantSpec { value: MatrixConstantValue::Zero, .. })))
        }));
        assert!(egraph[egraph.find(cases[2])].nodes.iter().any(|node| {
            matches!(node, MxxLang::MatrixConstant(spec)
                if matches!(egraph.analysis.symbols.matrix_constants.get(spec.0), Some(super::super::identity::MatrixConstantSpec { value: MatrixConstantValue::Zero, .. })))
        }));
    }

    #[test]
    fn binder_pointwise_plan_diagonalizes_same_selector_nested_cases() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let binder = test_binder(&mut egraph, 0, 1);
        let selector = egraph.add(MxxLang::IntBinder(binder));
        let shared = binder_matrix_atom(&mut egraph, selector, "nested-binder-shared");
        let zero = egraph.add(MxxLang::IntConst(0.into()));
        let one = egraph.add(MxxLang::IntConst(1.into()));
        let at_zero =
            family::instantiate_shared_element(&mut egraph, shared, binder, zero, &mut || {
                Ok::<(), ()>(())
            })
            .expect("zero case");
        let at_one =
            family::instantiate_shared_element(&mut egraph, shared, binder, one, &mut || {
                Ok::<(), ()>(())
            })
            .expect("one case");
        let inner = egraph.add(MxxLang::Switch(vec![selector, at_zero, at_one].into_boxed_slice()));
        let outer = egraph.add(MxxLang::Switch(vec![selector, inner, inner].into_boxed_slice()));
        let fixed = egraph.add(MxxLang::MatrixNegate([shared]));
        let root = egraph.add(MxxLang::MatrixAdd(vec![outer, fixed].into_boxed_slice()));
        egraph.rebuild();

        let plan = pointwise_add_switch_cancellation_plan(&egraph, root).expect("binder plan");
        assert!(plan.binder_aware.is_some());
        let replacement =
            build_pointwise_add_switch_cancellation(&mut egraph, root, plan).expect("replacement");
        let cases = switch_node(&egraph, replacement).expect("result switch");
        for case in &cases[1..] {
            assert!(egraph[egraph.find(*case)].nodes.iter().any(|node| {
                matches!(node, MxxLang::MatrixConstant(spec)
                    if matches!(egraph.analysis.symbols.matrix_constants.get(spec.0), Some(super::super::identity::MatrixConstantSpec { value: MatrixConstantValue::Zero, .. })))
            }));
        }
    }

    #[test]
    fn pointwise_add_switch_rejects_non_consensus_nested_selector_cases() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let other_selector = egraph.add(MxxLang::IntConst(1.into()));
        let (signal, _) = matrix_atom(&mut egraph, "nested-signal", None);
        let (other, _) = matrix_atom(&mut egraph, "nested-other", None);
        let negative_signal = egraph.add(MxxLang::MatrixNegate([signal]));

        let distinct =
            egraph.add(MxxLang::Switch(vec![other_selector, signal, other].into_boxed_slice()));
        let outer =
            egraph.add(MxxLang::Switch(vec![selector, distinct, distinct].into_boxed_slice()));
        let root = egraph.add(MxxLang::MatrixAdd(vec![outer, negative_signal].into_boxed_slice()));
        egraph.rebuild();
        assert!(pointwise_add_switch_cancellation_plan(&egraph, root).is_none());

        let short = egraph.add(MxxLang::Switch(vec![selector, signal].into_boxed_slice()));
        let outer = egraph.add(MxxLang::Switch(vec![selector, short, short].into_boxed_slice()));
        let root = egraph.add(MxxLang::MatrixAdd(vec![outer, negative_signal].into_boxed_slice()));
        egraph.rebuild();
        assert!(pointwise_add_switch_cancellation_plan(&egraph, root).is_none());

        let first = egraph.add(MxxLang::Switch(vec![selector, signal, other].into_boxed_slice()));
        let second = egraph.add(MxxLang::Switch(vec![selector, other, signal].into_boxed_slice()));
        egraph.union(first, second);
        egraph.rebuild();
        let outer = egraph.add(MxxLang::Switch(vec![selector, first, first].into_boxed_slice()));
        let root = egraph.add(MxxLang::MatrixAdd(vec![outer, negative_signal].into_boxed_slice()));
        egraph.rebuild();
        assert!(pointwise_add_switch_cancellation_plan(&egraph, root).is_none());

        let cyclic_structure = PointwiseAddSwitchStructure {
            terms: vec![outer, negative_signal],
            switch_index: 0,
            switch: vec![selector, root, other].into_boxed_slice(),
        };
        assert!(matches!(
            pointwise_add_switch_cancellation_for_structure(&egraph, root, &cyclic_structure, 1, 1),
            Err(PointwiseAddSwitchReject::CaseCycleOrNestedSwitch { case_index: 0 })
        ));
    }

    #[test]
    fn pointwise_add_switch_reports_unmatched_diagonalized_case_without_rereading_it() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let (signal, _) = matrix_atom(&mut egraph, "diagnostic-signal", None);
        let (other, _) = matrix_atom(&mut egraph, "diagnostic-other", None);
        let inner = egraph.add(MxxLang::Switch(vec![selector, signal, other].into_boxed_slice()));
        let outer = egraph.add(MxxLang::Switch(vec![selector, inner, inner].into_boxed_slice()));
        let fixed = egraph.add(MxxLang::MatrixNegate([signal]));
        let root = egraph.add(MxxLang::MatrixAdd(vec![outer, fixed].into_boxed_slice()));
        egraph.rebuild();

        assert!(matches!(
            pointwise_add_switch_cancellation_reason(&egraph, root),
            Err(PointwiseAddSwitchReject::UnmatchedFixedTerms {
                case_index: 1,
                direct_terms: 1,
                negated_terms: 0,
                case_physical_adds: 0,
                case_grouped_add_children: 0,
                ..
            })
        ));
    }

    #[test]
    fn pointwise_add_switch_preserves_multiplicity_and_rejects_nested_or_competing_cases() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let (signal, _) = matrix_atom(&mut egraph, "signal", None);
        let (residual, _) = matrix_atom(&mut egraph, "residual", None);
        let (local, _) = matrix_atom(&mut egraph, "local", None);
        let negated_signal = egraph.add(MxxLang::MatrixNegate([signal]));
        let negated_local = egraph.add(MxxLang::MatrixNegate([local]));
        let case = egraph.add(MxxLang::MatrixAdd(
            vec![signal, residual, signal, local, negated_local].into_boxed_slice(),
        ));
        let switch = egraph.add(MxxLang::Switch(vec![selector, case, case].into_boxed_slice()));
        let root = egraph.add(MxxLang::MatrixAdd(
            vec![switch, negated_signal, negated_signal].into_boxed_slice(),
        ));
        egraph.rebuild();
        let plan = pointwise_add_switch_cancellation_plan(&egraph, root).expect("duplicates match");
        assert_eq!(
            plan.cases,
            vec![
                vec![SignedPointwiseTerm { base: egraph.find(residual), negative: false }],
                vec![SignedPointwiseTerm { base: egraph.find(residual), negative: false }],
            ]
        );

        let inner = egraph.add(MxxLang::Switch(vec![selector, signal, signal].into_boxed_slice()));
        let outer = egraph.add(MxxLang::Switch(vec![selector, inner, inner].into_boxed_slice()));
        let nested = egraph.add(MxxLang::MatrixAdd(vec![outer, negated_signal].into_boxed_slice()));
        let alternate =
            egraph.add(MxxLang::Switch(vec![selector, signal, negated_signal].into_boxed_slice()));
        egraph.union(switch, alternate);
        egraph.rebuild();
        assert!(pointwise_add_switch_cancellation_plan(&egraph, nested).is_some());
        assert_eq!(
            pointwise_add_switch_cancellation_reason(&egraph, root),
            Err(PointwiseAddSwitchReject::Structural {
                physical_root_adds: 1,
                eligible_single_switch_adds: 0,
                direct_switch_children: 1,
                direct_grouped_add_children: 0,
            })
        );
    }

    #[test]
    fn pointwise_add_switch_uses_minimum_eligible_root_and_rejects_case_self_cycle() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let (signal, _) = matrix_atom(&mut egraph, "signal", None);
        let negated = egraph.add(MxxLang::MatrixNegate([signal]));
        let switch = egraph.add(MxxLang::Switch(vec![selector, signal, signal].into_boxed_slice()));
        let eligible = egraph.add(MxxLang::MatrixAdd(vec![switch, negated].into_boxed_slice()));
        let ineligible = egraph.add(MxxLang::MatrixAdd(vec![signal, negated].into_boxed_slice()));
        egraph.union(eligible, ineligible);
        egraph.rebuild();
        assert_eq!(
            pointwise_add_switch_cancellation_plan(&egraph, eligible)
                .expect("the unique eligible physical Add is selected")
                .cases,
            vec![Vec::new(), Vec::new()]
        );

        let case = egraph.add(MxxLang::MatrixAdd(vec![signal].into_boxed_slice()));
        egraph.union(case, signal);
        let before = egraph.total_size();
        assert_eq!(egraph.find(case), egraph.find(signal));
        assert!(matches!(
            physical_add_terms(&egraph, case),
            PhysicalStructure::Unique(terms)
                if terms.iter().any(|child| *child == egraph.find(case))
        ));
        assert_eq!(egraph.total_size(), before);
    }

    #[test]
    fn pointwise_add_switch_does_not_repeat_existing_singleton_cases() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let (signal, _) = matrix_atom(&mut egraph, "signal", None);
        let (residual, _) = matrix_atom(&mut egraph, "residual", None);
        let negated = egraph.add(MxxLang::MatrixNegate([signal]));
        let case = egraph.add(MxxLang::MatrixAdd(vec![signal, residual].into_boxed_slice()));
        let switch = egraph.add(MxxLang::Switch(vec![selector, case, case].into_boxed_slice()));
        let root = egraph.add(MxxLang::MatrixAdd(vec![switch, negated].into_boxed_slice()));
        egraph.rebuild();
        let plan =
            pointwise_add_switch_cancellation_plan(&egraph, root).expect("singleton residual");
        let replacement = build_pointwise_add_switch_cancellation(&mut egraph, root, plan)
            .expect("direct plan builds");
        egraph.union(root, replacement);
        egraph.rebuild();
        let before = egraph.total_size();
        assert!(pointwise_add_switch_cancellation_plan(&egraph, root).is_none());
        assert_eq!(egraph.total_size(), before);
    }

    #[test]
    fn shared_budget_observes_owned_work() {
        let budget = SharedRewriteBudget::new();
        assert!(budget.reserve(1).is_ok());
        assert!(budget.reserve(1).is_ok());
        assert!(budget.reserve(1).is_ok());
        assert_eq!(budget.owned.load(Ordering::Relaxed), 3);
    }
}
