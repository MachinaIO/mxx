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
use egg::{Applier, EGraph, Id, Language, SearchMatches, Searcher, Subst, Symbol, Var};
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

    fn reserve(&self, additional: usize) -> bool {
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

    fn note_rewrite(&self, selector_distribution: bool) {
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
            }))
        });
        let has_physical_add = class.nodes.iter().any(|node| matches!(node, MxxLang::MatrixAdd(_)));
        let matched = relation_match ||
            (has_physical_add &&
                (exact_additive_cancellation_possible(egraph, eclass) ||
                    pointwise_add_switch_cancellation_possible(egraph, eclass)));
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
        if let Some(replacement) = exact_additive_remainder(egraph, root) {
            if egraph.union(root, replacement) {
                self.context.note_rewrite(false);
                return vec![replacement];
            }
        }
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
            for relation_position in 1..factors.len() {
                let relation = factors[relation_position];
                if egraph[egraph.find(relation)].data.relation_provenance.is_empty() {
                    continue;
                }
                let public = factors[relation_position - 1];
                match pointwise_same_selector(egraph, public, relation, true) {
                    Ok(Some(product)) => {
                        let replacement = ordered_product_sequence(
                            egraph,
                            &factors[..relation_position - 1],
                            &[product],
                            &factors[relation_position + 1..],
                        );
                        if egraph.union(root, replacement) {
                            self.context.note_rewrite(true);
                            return vec![replacement];
                        }
                        continue;
                    }
                    Ok(None)
                        if switch_node(egraph, public).is_some() ||
                            switch_node(egraph, relation).is_some() =>
                    {
                        self.context.fail(RelationFailure::TransformedOperand);
                        return Vec::new();
                    }
                    Err(RelationFailure::DifferentSelectorBlocked) => continue,
                    Err(failure) => {
                        self.context.fail(failure);
                        return Vec::new();
                    }
                    Ok(None) => {}
                }
                if let Some((replacement, selector_distribution)) =
                    checked_replacement(egraph, &self.context, &factors, relation_position)
                {
                    if egraph.union(root, replacement) {
                        self.context.note_rewrite(selector_distribution);
                        return vec![replacement];
                    }
                }
            }
        }
        Vec::new()
    }
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

/// Read-only explanation of why an exact additive normalization is, or is
/// not, available for one selected physical Add node.  This is used only by
/// failure diagnostics after extraction; it neither changes saturation nor
/// records state in the e-graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum AddNormalizationProbe {
    NoExactPair,
    CompetingPhysicalAdds,
    CycleOrSharedNestedAdd,
    RewriteStillPossible,
    NormalizedAlternativeSelected(MxxLang),
    NormalizedAlternativeNotSelected(MxxLang),
}

/// Inspects the existing physical Add representation once.  The returned
/// normalized node, when present, is already in the same e-class and is
/// suitable for a diagnostic-only cost comparison.
pub(crate) fn probe_exact_additive_normalization(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    root: Id,
    selected: &MxxLang,
) -> AddNormalizationProbe {
    let root = egraph.find(root);
    let selected_terms = match selected {
        MxxLang::MatrixAdd(terms) => {
            terms.iter().map(|term| egraph.find(*term)).collect::<Vec<_>>().into_boxed_slice()
        }
        _ => return AddNormalizationProbe::NoExactPair,
    };
    let add_structure = physical_add_terms(egraph, root);
    let competing = matches!(add_structure, PhysicalStructure::Ambiguous);
    if matches!(add_structure, PhysicalStructure::Absent) {
        return AddNormalizationProbe::NoExactPair;
    }
    let Some((terms, flattened)) = flattened_additive_terms_from(egraph, root, selected_terms)
    else {
        return AddNormalizationProbe::CycleOrSharedNestedAdd;
    };
    let Some((cancelled, any_cancelled)) = cancelled_additive_terms(egraph, &terms) else {
        return AddNormalizationProbe::CycleOrSharedNestedAdd;
    };
    if !flattened && !any_cancelled {
        return if competing {
            AddNormalizationProbe::CompetingPhysicalAdds
        } else {
            AddNormalizationProbe::NoExactPair
        };
    }
    let remaining = terms
        .into_iter()
        .zip(cancelled)
        .filter_map(|(term, cancelled)| (!cancelled).then_some(term))
        .collect::<Vec<_>>();
    if remaining.iter().any(|term| egraph.find(*term) == root) {
        return AddNormalizationProbe::CycleOrSharedNestedAdd;
    }
    let normalized = if remaining.is_empty() {
        egraph[root].nodes.iter().find_map(|node| match node {
            MxxLang::MatrixConstant(spec)
                if matches!(
                    egraph.analysis.symbols.matrix_constants.get(spec.0),
                    Some(super::identity::MatrixConstantSpec {
                        value: MatrixConstantValue::Zero,
                        ..
                    })
                ) =>
            {
                Some(node.clone())
            }
            _ => None,
        })
    } else {
        egraph[root].nodes.iter().find_map(|node| match node {
            MxxLang::MatrixAdd(candidate)
                if candidate.len() == remaining.len() &&
                    candidate
                        .iter()
                        .zip(&remaining)
                        .all(|(left, right)| egraph.find(*left) == egraph.find(*right)) =>
            {
                Some(node.clone())
            }
            _ => None,
        })
    };
    let Some(normalized) = normalized else {
        return if competing {
            AddNormalizationProbe::CompetingPhysicalAdds
        } else {
            AddNormalizationProbe::RewriteStillPossible
        };
    };
    if selected_node_matches(egraph, selected, &normalized) {
        AddNormalizationProbe::NormalizedAlternativeSelected(normalized)
    } else {
        AddNormalizationProbe::NormalizedAlternativeNotSelected(normalized)
    }
}

fn selected_node_matches(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    left: &MxxLang,
    right: &MxxLang,
) -> bool {
    match (left, right) {
        (MxxLang::MatrixAdd(left), MxxLang::MatrixAdd(right)) => {
            left.len() == right.len() &&
                left.iter()
                    .zip(right)
                    .all(|(left, right)| egraph.find(*left) == egraph.find(*right))
        }
        (MxxLang::MatrixConstant(left), MxxLang::MatrixConstant(right)) => left == right,
        _ => false,
    }
}

fn flattened_additive_terms_from(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    root: Id,
    terms: Box<[Id]>,
) -> Option<(Vec<Id>, bool)> {
    let mut output = Vec::with_capacity(terms.len());
    let mut work = terms.iter().rev().map(|term| (egraph.find(*term), false)).collect::<Vec<_>>();
    let mut visiting = HashSet::new();
    let mut expanded_adds = HashSet::new();
    visiting.insert(root);
    let mut flattened = false;

    while let Some((term, exiting)) = work.pop() {
        if exiting {
            visiting.remove(&term);
            continue;
        }
        let children = match physical_add_terms(egraph, term) {
            PhysicalStructure::Absent => {
                output.push(term);
                continue;
            }
            PhysicalStructure::Unique(children) => children,
            PhysicalStructure::Ambiguous => return None,
        };
        if !visiting.insert(term) {
            return None;
        }
        if !expanded_adds.insert(term) {
            return None;
        }
        flattened = true;
        work.push((term, true));
        for child in children.iter().rev() {
            work.push((egraph.find(*child), false));
        }
    }
    Some((output, flattened))
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

/// Finds exact opposite pairs in physical term order.  Every input term is
/// queued or paired once, so this is O(k) without signed counters or overflow
/// conversions.  The returned mask preserves the original retained order.
fn cancelled_additive_terms(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    terms: &[Id],
) -> Option<(Vec<bool>, bool)> {
    let mut cancelled = vec![false; terms.len()];
    let mut positive = HashMap::<Id, Vec<usize>>::new();
    let mut negative = HashMap::<Id, Vec<usize>>::new();
    let mut any = false;

    for (index, term) in terms.iter().enumerate() {
        let term = egraph.find(*term);
        let (base, is_negative) = match physical_negated_base(egraph, term) {
            PhysicalStructure::Unique(base) if base != term => (base, true),
            PhysicalStructure::Absent => (term, false),
            PhysicalStructure::Unique(_) | PhysicalStructure::Ambiguous => return None,
        };
        let opposite = if is_negative { &mut positive } else { &mut negative };
        if let Some(match_index) = opposite.get_mut(&base).and_then(Vec::pop) {
            cancelled[index] = true;
            cancelled[match_index] = true;
            any = true;
        } else {
            let same = if is_negative { &mut negative } else { &mut positive };
            same.entry(base).or_default().push(index);
        }
    }
    Some((cancelled, any))
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
    if !negative {
        return child == base;
    }
    matches!(
        physical_negated_base(egraph, child),
        PhysicalStructure::Unique(negated) if negated == base && negated != child
    )
}

fn strict_additive_remainder(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    root: Id,
    remaining: &[(Id, bool)],
) -> bool {
    let root = egraph.find(root);
    if remaining.is_empty() {
        return !egraph[root].nodes.iter().any(|node| {
            matches!(
                node,
                MxxLang::MatrixConstant(spec)
                    if matches!(
                        egraph.analysis.symbols.matrix_constants.get(spec.0),
                        Some(super::identity::MatrixConstantSpec {
                            value: MatrixConstantValue::Zero,
                            ..
                        })
                    )
            )
        });
    }
    if remaining.iter().any(|(term, _)| egraph.find(*term) == root) {
        return false;
    }
    !egraph[root].nodes.iter().any(|node| {
        matches!(
            node,
            MxxLang::MatrixAdd(existing)
                if existing.len() == remaining.len()
                    && existing.iter().zip(remaining).all(|(left, right)|
                        signed_additive_child_matches(egraph, *left, *right))
        )
    })
}

/// Opens one unambiguous Add/Negate tree into signed leaves.  This is local
/// to exact root normalization: unlike the binder consensus path it never
/// tries to reconcile competing representations.  A repeated structured
/// e-class is rejected rather than copied, keeping this pass linear in the
/// physical tree it accepts.
fn flattened_signed_additive_terms_from(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    root: Id,
    terms: Vec<Id>,
) -> Option<(Vec<(Id, bool)>, bool)> {
    let mut output = Vec::with_capacity(terms.len());
    let mut work =
        terms.iter().rev().map(|term| (egraph.find(*term), false, false)).collect::<Vec<_>>();
    let mut visiting = HashSet::new();
    let mut expanded = HashSet::new();
    visiting.insert(egraph.find(root));
    let mut flattened = false;

    while let Some((term, negative, exiting)) = work.pop() {
        if exiting {
            visiting.remove(&term);
            continue;
        }
        let add = physical_add_terms(egraph, term);
        let negate = physical_negated_base(egraph, term);
        match (add, negate) {
            (PhysicalStructure::Absent, PhysicalStructure::Absent) => output.push((term, negative)),
            (PhysicalStructure::Unique(children), PhysicalStructure::Absent) => {
                if !visiting.insert(term) || !expanded.insert(term) {
                    return None;
                }
                flattened = true;
                work.push((term, negative, true));
                for child in children.iter().rev() {
                    work.push((egraph.find(*child), negative, false));
                }
            }
            (PhysicalStructure::Absent, PhysicalStructure::Unique(base)) if base != term => {
                if !visiting.insert(term) || !expanded.insert(term) {
                    return None;
                }
                flattened = true;
                work.push((term, negative, true));
                work.push((base, !negative, false));
            }
            (PhysicalStructure::Ambiguous, _) |
            (_, PhysicalStructure::Ambiguous) |
            (PhysicalStructure::Unique(_), PhysicalStructure::Unique(_)) |
            (PhysicalStructure::Absent, PhysicalStructure::Unique(_)) => return None,
        }
    }
    Some((output, flattened))
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

/// Selects the first strict additive normalization among the root's physical
/// Add representations.  A non-strict earlier candidate must not mask a later
/// physical Add which cancels to a new result.
fn additive_remainder_terms(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    root: Id,
) -> Option<Vec<(Id, bool)>> {
    let root = egraph.find(root);
    let mut seen = BTreeSet::new();
    for node in &egraph[root].nodes {
        let MxxLang::MatrixAdd(children) = node else { continue };
        let children =
            children.iter().map(|child| egraph.find(*child)).collect::<Vec<_>>().into_boxed_slice();
        if !seen.insert(children.clone()) {
            continue;
        }
        // Cancel direct pairs first.  In particular, a grouped signal and its
        // unique Negate cancel without opening the signal expression.
        let Some((direct_cancelled, direct_pair)) = cancelled_additive_terms(egraph, &children)
        else {
            continue;
        };
        let direct_remaining = children
            .into_iter()
            .zip(direct_cancelled)
            .filter_map(|(term, cancelled)| (!cancelled).then_some(term))
            .collect::<Vec<_>>();
        let Some((terms, flattened)) =
            flattened_signed_additive_terms_from(egraph, root, direct_remaining)
        else {
            continue;
        };
        let (cancelled, nested_pair) = cancelled_signed_additive_terms(&terms);
        if !direct_pair && !flattened && !nested_pair {
            continue;
        }
        let remaining = terms
            .into_iter()
            .zip(cancelled)
            .filter_map(|(term, cancelled)| (!cancelled).then_some(term))
            .collect::<Vec<_>>();
        if strict_additive_remainder(egraph, root, &remaining) {
            return Some(remaining);
        }
    }
    None
}

fn exact_additive_cancellation_possible(egraph: &EGraph<MxxLang, MxxAnalysis>, root: Id) -> bool {
    additive_remainder_terms(egraph, root).is_some()
}

fn exact_additive_remainder(egraph: &mut EGraph<MxxLang, MxxAnalysis>, root: Id) -> Option<Id> {
    let root = egraph.find(root);
    let remaining = additive_remainder_terms(egraph, root)?;
    match remaining.len() {
        0 => match egraph[root].data.sort.clone() {
            Ok(MxxSort::Matrix(matrix_type)) => {
                let spec = egraph.analysis.symbols.matrix_constants.intern(
                    super::identity::MatrixConstantSpec {
                        matrix_type,
                        value: MatrixConstantValue::Zero,
                    },
                );
                Some(
                    egraph
                        .add(MxxLang::MatrixConstant(super::identity::MatrixConstantSpecId(spec))),
                )
            }
            _ => None,
        },
        // Keep the retained child behind one physical Add node.  Unioning an
        // Add e-class directly with one of its own children makes that old
        // e-node self-referential after egg canonicalization, which can keep
        // saturation active even though the cancellation is complete.
        _ => {
            let remaining = remaining
                .into_iter()
                .map(
                    |(term, negative)| {
                        if negative { egraph.add(MxxLang::MatrixNegate([term])) } else { term }
                    },
                )
                .collect::<Vec<_>>();
            Some(egraph.add(MxxLang::MatrixAdd(remaining.into_boxed_slice())))
        }
    }
}

/// A fully validated pointwise Add/Switch cancellation.  Planning is read-only
/// so a failing case cannot leave partial e-nodes in the graph.
struct PointwiseAddSwitchPlan {
    selector: Id,
    cases: Vec<Vec<Id>>,
    binder_aware: Option<Box<[BinderAwarePointwiseAddSwitchPlan]>>,
}

/// A read-only description of the symbolic fixed portion of a pointwise
/// cancellation.  Instantiation is deliberately deferred to the applier:
/// search must not add speculative e-nodes.
struct BinderAwarePointwiseAddSwitchPlan {
    binder: BinderId,
    /// Direct terms of every physical stored case, structurally checked by the
    /// read-only planner before the applier creates its first e-node.
    case_terms: Box<[Box<[Id]>]>,
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
        let terms = direct_add_terms_or_atomic(egraph, *case)
            .ok_or(PointwiseAddSwitchReject::CaseCycleOrNestedSwitch { case_index: case_epoch })?;
        let case = egraph.find(*case);
        if case == root || terms.iter().any(|term| has_physical_switch(egraph, *term)) {
            return Err(PointwiseAddSwitchReject::CaseCycleOrNestedSwitch {
                case_index: case_epoch,
            });
        }
        let mut consumed = vec![false; terms.len()];
        let mut matched = 0;
        for (index, term) in terms.iter().enumerate() {
            let Some((base, negative)) = signed_additive_term(egraph, *term) else {
                return Err(PointwiseAddSwitchReject::CaseCycleOrNestedSwitch {
                    case_index: case_epoch,
                });
            };
            if let Some((required, seen_epoch, used)) = fixed_signed.get_mut(&(base, !negative)) {
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
            let diagnostic = unmatched_fixed_terms_diagnostic(egraph, &fixed, case);
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
        let (cancelled, _) = cancelled_additive_terms(egraph, &after_cross)
            .ok_or(PointwiseAddSwitchReject::CaseCycleOrNestedSwitch { case_index: case_epoch })?;
        let remaining = after_cross
            .into_iter()
            .zip(cancelled)
            .filter_map(|(term, cancelled)| (!cancelled).then_some(term))
            .collect::<Vec<_>>();
        if remaining.iter().any(|term| egraph.find(*term) == root) {
            return Err(PointwiseAddSwitchReject::CaseCycleOrNestedSwitch {
                case_index: case_epoch,
            });
        }
        normalized_cases.push(remaining);
    }
    let selector = egraph.find(structure.switch[0]);
    if equivalent_switch_exists(egraph, root, selector, &normalized_cases) {
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
        let case = egraph.find(*case);
        if case == root {
            return Err(BinderPreflightReject::CaseSelfCycle { case_index });
        }
        let terms = match physical_add_terms(egraph, case) {
            PhysicalStructure::Absent => vec![case],
            PhysicalStructure::Ambiguous => {
                return Err(BinderPreflightReject::CaseAmbiguous { case_index });
            }
            PhysicalStructure::Unique(terms) => {
                let terms = terms.into_iter().collect::<Vec<_>>();
                if terms.iter().any(|child| *child == case) {
                    return Err(BinderPreflightReject::CaseSelfCycle { case_index });
                }
                terms
            }
        };
        if terms.iter().any(|term| has_physical_switch(egraph, *term)) {
            return Err(BinderPreflightReject::CaseNestedSwitch { case_index });
        }
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
/// helper: it retains at most sixteen identities while it rescans the already
/// selected failing case and its direct fixed-term sequence.
fn unmatched_fixed_terms_diagnostic(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    fixed: &[Id],
    case: Id,
) -> UnmatchedFixedTermsDiagnostic {
    let case = egraph.find(case);
    let case_direct_terms = direct_add_terms_or_atomic(egraph, case)
        .expect("validated pointwise case remains structurally stable during read-only diagnosis");
    let (fixed_terms, fixed_terms_omitted_occurrences) = signed_multiplicity_summary(egraph, fixed);
    let (case_terms, case_terms_omitted_occurrences) =
        signed_multiplicity_summary(egraph, &case_direct_terms);
    UnmatchedFixedTermsDiagnostic {
        direct_terms: case_direct_terms.len(),
        negated_terms: case_direct_terms
            .iter()
            .filter(|term| {
                signed_additive_term(egraph, **term).is_some_and(|(_, negative)| negative)
            })
            .count(),
        fixed_unique_add_children: fixed
            .iter()
            .filter(|term| unique_add_terms(egraph, **term).is_some())
            .count(),
        case_physical_adds: egraph[case]
            .nodes
            .iter()
            .filter(|node| matches!(node, MxxLang::MatrixAdd(_)))
            .count(),
        case_grouped_add_children: case_direct_terms
            .iter()
            .filter(|term| unique_add_terms(egraph, **term).is_some())
            .count(),
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
        cases.push(build_additive_terms(egraph, root, terms));
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
    CaseSignedReject,
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
    Some(false)
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
            return Some(false);
        }
    }
    for (position, negative) in states {
        progress().ok()?;
        if position == target.len() && (actual_negative != negative) != target_negative {
            return Some(true);
        }
    }
    Some(false)
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

fn peel_fixed_target_from_term(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    term: &PeelTerm,
    target: &(Box<[Id]>, bool),
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<Option<Vec<PeelTerm>>> {
    let (target_spine, target_negative) = target;
    match term {
        PeelTerm::Concrete { base, negative } => {
            if has_opposite_direct_span(
                egraph,
                *base,
                target_spine,
                *negative,
                *target_negative,
                progress,
            )? {
                return Some(Some(Vec::new()));
            }
            let leaves = signed_additive_leaves_with_progress(egraph, &[*base], progress)?;
            if leaves.len() > 1 {
                for (index, (leaf, leaf_negative)) in leaves.iter().enumerate() {
                    progress().ok()?;
                    if has_opposite_direct_span(
                        egraph,
                        *leaf,
                        target_spine,
                        *negative != *leaf_negative,
                        *target_negative,
                        progress,
                    )? {
                        let mut remaining = Vec::new();
                        for (sibling_index, (sibling, sibling_negative)) in
                            leaves.iter().enumerate()
                        {
                            if sibling_index != index {
                                progress().ok()?;
                                remaining.try_reserve(1).ok()?;
                                remaining.push(PeelTerm::Concrete {
                                    base: *sibling,
                                    negative: *negative != *sibling_negative,
                                });
                            }
                        }
                        return Some(Some(remaining));
                    }
                }
            }
            let Some(factors) = uncontested_product_factors_with_progress(egraph, *base, progress)
            else {
                return Some(None);
            };
            for factor_index in 0..factors.len() {
                let leaves = signed_additive_leaves_with_progress(
                    egraph,
                    &[factors[factor_index]],
                    progress,
                )?;
                for (leaf_index, (leaf, leaf_negative)) in leaves.iter().enumerate() {
                    progress().ok()?;
                    if product_factor_span_matches(
                        egraph,
                        &factors[..factor_index],
                        *leaf,
                        *leaf_negative,
                        &factors[factor_index + 1..],
                        target_spine,
                        *negative,
                        *target_negative,
                        progress,
                    )? {
                        let remaining = copy_signed_terms(&leaves, Some(leaf_index), progress)?;
                        if remaining.is_empty() {
                            return Some(Some(Vec::new()));
                        }
                        progress().ok()?;
                        return Some(Some(vec![PeelTerm::ProductFactor {
                            prefix: copy_ids(&factors[..factor_index], progress)?,
                            terms: remaining,
                            suffix: copy_ids(&factors[factor_index + 1..], progress)?,
                            negative: *negative,
                        }]));
                    }
                }
            }
            Some(None)
        }
        PeelTerm::ProductFactor { prefix, terms, suffix, negative } => {
            for (index, (leaf, leaf_negative)) in terms.iter().enumerate() {
                progress().ok()?;
                if product_factor_span_matches(
                    egraph,
                    prefix,
                    *leaf,
                    *leaf_negative,
                    suffix,
                    target_spine,
                    *negative,
                    *target_negative,
                    progress,
                )? {
                    let remaining = copy_signed_terms(terms, Some(index), progress)?;
                    if remaining.is_empty() {
                        return Some(Some(Vec::new()));
                    }
                    progress().ok()?;
                    return Some(Some(vec![PeelTerm::ProductFactor {
                        prefix: copy_ids(prefix, progress)?,
                        terms: remaining,
                        suffix: copy_ids(suffix, progress)?,
                        negative: *negative,
                    }]));
                }
            }
            Some(None)
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
    for target in fixed {
        if target.0.len() == 1 && is_exact_zero_matrix(egraph, target.0[0], progress)? {
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
        let mut actual_valid = true;
        for term in terms {
            let Some((base, negative)) = signed_additive_term(egraph, *term) else {
                actual_valid = false;
                break;
            };
            progress().ok()?;
            actual.try_reserve(1).ok()?;
            actual.push(PeelTerm::Concrete { base, negative });
        }
        if !actual_valid {
            report_binder_build_reject(
                egraph,
                case_index,
                BinderBuildRejectStage::CaseSignedReject,
                &[],
                &[],
                should_collect,
                sink,
            );
            return None;
        };
        let mut mapped_fixed = Vec::new();
        for occurrence in &plan.fixed_occurrences {
            let mapped = mapped_bases[occurrence.base_index];
            progress().ok()?;
            mapped_fixed.try_reserve(1).ok()?;
            mapped_fixed.push((mapped, occurrence.negative));
        }
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
        let (mapped_fixed_spines, fixed_cancelled) =
            cancel_fixed_spines(mapped_fixed_spines, progress)?;
        let (any_peeled, unmatched_fixed) =
            peel_fixed_targets(egraph, &mut actual, &mapped_fixed_spines, progress)?;
        if !any_peeled && !fixed_cancelled {
            let mut diagnostic_actual = Vec::new();
            for term in terms {
                progress().ok()?;
                if let Some(term) = signed_additive_term(egraph, *term) {
                    progress().ok()?;
                    diagnostic_actual.try_reserve(1).ok()?;
                    diagnostic_actual.push(term);
                }
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
    Some(egraph.add(MxxLang::Switch(cases.into_boxed_slice())))
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
            LeafNodeDescriptor::Atom { source_id: source.0, source_kind, indices, indices_omitted }
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

fn direct_add_terms_or_atomic(egraph: &EGraph<MxxLang, MxxAnalysis>, term: Id) -> Option<Vec<Id>> {
    let term = egraph.find(term);
    match physical_add_terms(egraph, term) {
        PhysicalStructure::Absent => Some(vec![term]),
        PhysicalStructure::Ambiguous => None,
        PhysicalStructure::Unique(terms) => {
            let terms = terms.into_iter().collect::<Vec<_>>();
            (!terms.iter().any(|child| *child == term)).then_some(terms)
        }
    }
}

fn unique_switch_cases(egraph: &EGraph<MxxLang, MxxAnalysis>, term: Id) -> Option<Box<[Id]>> {
    let mut unique = None;
    for node in &egraph[egraph.find(term)].nodes {
        let MxxLang::Switch(cases) = node else { continue };
        let cases =
            cases.iter().map(|term| egraph.find(*term)).collect::<Vec<_>>().into_boxed_slice();
        match &unique {
            Some(previous) if previous != &cases => return None,
            Some(_) => {}
            None => unique = Some(cases),
        }
    }
    unique
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
    mut visit: F,
    progress: &mut dyn FnMut() -> Result<(), ()>,
) -> Option<Vec<(Id, bool)>>
where
    F: FnMut(Id),
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

    fn consensus<F>(
        egraph: &EGraph<MxxLang, MxxAnalysis>,
        term: Id,
        memo: &mut HashMap<Id, Consensus>,
        active: &mut HashSet<Id>,
        visit: &mut F,
        progress: &mut dyn FnMut() -> Result<(), ()>,
    ) -> Option<Consensus>
    where
        F: FnMut(Id),
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
            for node in &egraph[term].nodes {
                let candidate = match node {
                    MxxLang::MatrixNegate([base]) => {
                        has_additive_structure = true;
                        let mut candidate =
                            consensus(egraph, *base, memo, active, visit, progress)?;
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
                            let child = consensus(egraph, *child, memo, active, visit, progress)?;
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
        let consensus = consensus(egraph, *term, &mut memo, &mut active, &mut visit, progress)?;
        for occurrence in consensus.representative {
            progress().ok()?;
            output.try_reserve(1).ok()?;
            output.push(occurrence);
        }
    }
    Some(output)
}

/// A read-only signed noncommutative polynomial view used only for mapped
/// fixed targets while building one binder case.  It first reuses the existing
/// Add/Negate consensus, then distributes only an unambiguous physical
/// MatrixMultiply spine.  The actual case uses fixed-guided peeling instead,
/// so this is the remaining fixed-target-only Cartesian expansion. A product
/// e-class with competing forms is an atomic leaf: relation saturation may
/// legitimately have unioned it with a structurally different target.
/// Switch is deliberately atomic and is never enumerated.
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
        let factors = uncontested_product_factors_with_progress(egraph, term, progress);
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
                let combinations = product.len().checked_mul(expanded_factor.len())?;
                let mut next = Vec::new();
                for (prefix, prefix_negative) in &product {
                    for (suffix, suffix_negative) in &expanded_factor {
                        // Every generated Cartesian monomial consumes the
                        // shared rewrite budget before allocating or pushing.
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
                        next.push((
                            combined.into_boxed_slice(),
                            *prefix_negative != *suffix_negative,
                        ));
                    }
                }
                debug_assert_eq!(next.len(), combinations);
                product = next;
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

fn checked_replacement(
    egraph: &mut EGraph<MxxLang, MxxAnalysis>,
    context: &RewriteContext,
    factors: &[Id],
    relation_position: usize,
) -> Option<(Id, bool)> {
    let relation = egraph.find(factors[relation_position]);
    let actual_public = egraph.find(factors[relation_position - 1]);
    let provenance = &egraph[relation].data.relation_provenance;
    let mut candidates = Vec::new();
    if !flatten_provenance(provenance, context, &mut candidates) {
        return None;
    }
    let mut replacements = BTreeSet::new();
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
            let distributed_public =
                distribution_public_operand(egraph, actual_public, registration.expected_public);
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
                    build_affine_concat(egraph, registration.expected_public, &plan);
                replacements.insert(ordered_product_sequence(
                    egraph,
                    &factors[..relation_position - 1],
                    &[normalized_public, relation],
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
            );
            // If the left factor is additive, consuming the relation without
            // an exact matching summand would be an unsound general-product
            // rewrite.  Fail closed instead of silently taking that fallback.
            let additive_public = egraph[egraph.find(actual_public)]
                .nodes
                .iter()
                .any(|node| matches!(node, MxxLang::MatrixAdd(_)));
            let replacement = match (additive_public, distributed) {
                (_, Some(replacement)) => replacement,
                (true, None) => {
                    failures.insert(RelationFailure::TransformedOperand);
                    continue;
                }
                (false, None) => target_spliced_product(
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
    if replacements.len() > 1 {
        context
            .fail(RelationFailure::AmbiguousReplacement { sources: sources.into_iter().collect() });
        return None;
    }
    let Some(replacement) = replacements.into_iter().next() else {
        if let Some(failure) = failures.into_iter().next() {
            context.fail(failure);
        }
        return None;
    };
    let selector_distribution =
        switch_node(egraph, actual_public).is_some() || switch_node(egraph, relation).is_some();
    Some((replacement, selector_distribution))
}

fn distribution_public_operand(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    actual_public: Id,
    expected_public: Id,
) -> Option<Id> {
    let add = egraph[egraph.find(actual_public)].nodes.iter().find_map(|node| match node {
        MxxLang::MatrixAdd(children) => Some(children),
        _ => None,
    })?;
    add.iter().find_map(|term| {
        egraph[egraph.find(*term)].nodes.iter().find_map(|node| match node {
            MxxLang::MatrixMultiply(factors)
                if factors
                    .last()
                    .is_some_and(|last| egraph.find(*last) == egraph.find(expected_public)) =>
            {
                Some(expected_public)
            }
            _ => None,
        })
    })
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
        let terms = chunk_terms(egraph, *chunk)?;
        let mut match_term = None;
        for (index, term) in terms.iter().enumerate() {
            let Some((prefix, start, end)) = slice_product(egraph, *term, expected_public) else {
                continue;
            };
            if start != next_column || end <= start || end > full_columns || match_term.is_some() {
                return None;
            }
            match &shared_prefix {
                Some(previous) if !same_canonical_indices(egraph, previous, &prefix) => {
                    return None;
                }
                None => shared_prefix = Some(prefix),
                _ => {}
            }
            match_term = Some((index, end));
        }
        let Some((matched_index, end)) = match_term else { return None };
        let remaining = terms
            .iter()
            .enumerate()
            .filter_map(|(index, term)| (index != matched_index).then_some(*term))
            .collect::<Vec<_>>();
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
            residuals.push(remaining.into_boxed_slice());
        }
        next_column = end;
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

/// A concat chunk is either one direct signal term or one unambiguous
/// physical addition.  An e-class with competing add representations is not
/// a basis for choosing which residual to preserve.
fn chunk_terms(egraph: &EGraph<MxxLang, MxxAnalysis>, id: Id) -> Option<Box<[Id]>> {
    unique_add_terms(egraph, id).or_else(|| {
        (!egraph[egraph.find(id)].nodes.iter().any(|node| matches!(node, MxxLang::MatrixAdd(_))))
            .then(|| vec![egraph.find(id)].into_boxed_slice())
    })
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

fn build_affine_concat(
    egraph: &mut EGraph<MxxLang, MxxAnalysis>,
    expected_public: Id,
    plan: &AffineConcatPlan,
) -> Id {
    let leading = ordered_product_sequence(egraph, &plan.prefix, &[expected_public], &[]);
    let residual_chunks = plan
        .residuals
        .as_ref()
        .into_iter()
        .flatten()
        .map(|terms| {
            if terms.len() == 1 { terms[0] } else { egraph.add(MxxLang::MatrixAdd(terms.clone())) }
        })
        .collect::<Vec<_>>();
    let residual = match residual_chunks.len() {
        0 => None,
        1 => Some(residual_chunks[0]),
        _ => Some(egraph.add(MxxLang::MatrixConcat {
            axis: Axis::Columns,
            inputs: residual_chunks.into_boxed_slice(),
        })),
    };
    let mut terms = Vec::with_capacity(plan.outside.len() + 1 + usize::from(residual.is_some()));
    terms.push(leading);
    if let Some(residual) = residual {
        terms.push(residual);
    }
    terms.extend_from_slice(&plan.outside);
    if terms.len() == 1 {
        terms[0]
    } else {
        egraph.add(MxxLang::MatrixAdd(terms.into_boxed_slice()))
    }
}

/// Closed relation-redex classification used by extraction.  Matrix bounds
/// are resolved independently through the authoritative bound input and the
/// shared node transfer; source syntax is never treated as a bound contract.
pub fn classify_proposal_node(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    node: &MxxLang,
    context: &RewriteContext,
) -> Result<bool, RelationFailure> {
    let MxxLang::MatrixMultiply(factors) = node else {
        return Ok(false);
    };
    for relation_position in 1..factors.len() {
        let relation = egraph.find(factors[relation_position]);
        if egraph[relation].data.relation_provenance.is_empty() {
            continue;
        }
        let public = egraph.find(factors[relation_position - 1]);
        // Distribution itself is a closed structural rewrite.  It must be
        // charged before checking a relation source inside a selected Switch:
        // that source is not yet a bare atom, but the applier will expose it
        // pointwise without enumerating selector combinations.
        match pointwise_selector_is_distributable(egraph, public, relation) {
            Ok(true) => return Ok(true),
            Ok(false) => {}
            Err(RelationFailure::DifferentSelectorBlocked) => continue,
            Err(failure) => return Err(failure),
        }
        let mut sources = Vec::new();
        if !flatten_provenance(&egraph[relation].data.relation_provenance, context, &mut sources) {
            return Err(context.failure().expect("failed provenance reservation records a failure"));
        }
        for candidate in sources {
            let RelationCandidate::Direct(source) = candidate else { continue };
            for registration in context.registrations(source.source) {
                let distributed_public =
                    distribution_public_operand(egraph, public, registration.expected_public);
                let affine_plan = affine_concat_plan(egraph, public, registration.expected_public);
                let preflight_public = distributed_public.unwrap_or(public);
                if preflight_registration(
                    egraph,
                    relation,
                    &source,
                    &registration,
                    if distributed_public.is_some() || affine_plan.is_some() {
                        registration.expected_public
                    } else {
                        preflight_public
                    },
                )
                .is_ok() &&
                    same_canonical_indices(egraph, &source.indices, &registration.indices) &&
                    (distributed_public.is_some() ||
                        affine_plan.is_some() ||
                        egraph.find(registration.expected_public) == public)
                {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
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
fn relation_guided_distribution(
    egraph: &mut EGraph<MxxLang, MxxAnalysis>,
    factors: &[Id],
    relation_position: usize,
    expected_public: Id,
    target: Id,
) -> Option<Id> {
    let actual_public = factors[relation_position - 1];
    let relation = factors[relation_position];
    let add = egraph[egraph.find(actual_public)].nodes.iter().find_map(|node| match node {
        MxxLang::MatrixAdd(children) => Some(children.clone()),
        _ => None,
    })?;
    let mut terms = Vec::with_capacity(add.len());
    let mut consumed = false;
    for term in add.iter() {
        let product = egraph[egraph.find(*term)].nodes.iter().find_map(|node| match node {
            MxxLang::MatrixMultiply(factors) => Some(factors.clone()),
            _ => None,
        });
        let has_expected_public = product.as_ref().is_some_and(|factors| {
            factors.last().is_some_and(|last| egraph.find(*last) == egraph.find(expected_public))
        });
        if has_expected_public {
            let product_factors = product.expect("checked above");
            let replacement = target_spliced_product(
                egraph,
                &factors[..relation_position - 1],
                &product_factors[..product_factors.len() - 1],
                target,
                &factors[relation_position + 1..],
            );
            // `target_spliced_product` distributes the already-validated
            // relation target through the fixed prefix.  Keep that exact
            // target addition at this physical level so a later registered
            // relation can recognize a concat of affine slices without
            // inventing a general product-distribution rewrite.
            if let Some(expanded) = unique_add_terms(egraph, replacement) {
                terms.extend(expanded.iter().copied());
            } else {
                terms.push(replacement);
            }
            consumed = true;
        } else {
            terms.push(ordered_product_sequence(
                egraph,
                &factors[..relation_position - 1],
                &[*term, relation],
                &factors[relation_position + 1..],
            ));
        }
    }
    consumed.then(|| egraph.add(MxxLang::MatrixAdd(terms.into_boxed_slice())))
}

fn ordered_product_sequence(
    egraph: &mut EGraph<MxxLang, MxxAnalysis>,
    prefix: &[Id],
    middle: &[Id],
    suffix: &[Id],
) -> Id {
    let mut factors = Vec::with_capacity(prefix.len() + middle.len() + suffix.len());
    factors.extend_from_slice(prefix);
    factors.extend_from_slice(middle);
    factors.extend_from_slice(suffix);
    if factors.len() == 1 {
        factors[0]
    } else {
        egraph.add(MxxLang::MatrixMultiply(factors.into_boxed_slice()))
    }
}

fn target_spliced_product(
    egraph: &mut EGraph<MxxLang, MxxAnalysis>,
    prefix: &[Id],
    target_prefix: &[Id],
    target: Id,
    suffix: &[Id],
) -> Id {
    if let Some(terms) = unique_add_terms(egraph, target) {
        if terms.is_empty() {
            return target;
        }
        let mut products = Vec::with_capacity(terms.len());
        for term in terms.iter() {
            let expanded = unique_product_factors(egraph, *term);
            let mut middle = Vec::with_capacity(
                target_prefix.len() + expanded.as_ref().map_or(1, |factors| factors.len()),
            );
            middle.extend_from_slice(target_prefix);
            if let Some(factors) = expanded {
                middle.extend_from_slice(&factors);
            } else {
                middle.push(*term);
            }
            products.push(ordered_product_sequence(egraph, prefix, &middle, suffix));
        }
        return egraph.add(MxxLang::MatrixAdd(products.into_boxed_slice()));
    }
    let mut middle = Vec::with_capacity(target_prefix.len() + 1);
    middle.extend_from_slice(target_prefix);
    middle.push(target);
    ordered_product_sequence(egraph, prefix, &middle, suffix)
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
    use crate::operational_noise::{
        bound::{
            BoundClass, BoundEvaluationError, BoundEvaluator, BoundInput, MatrixBound,
            MatrixMetadata, ResolvedMatrixConstant,
        },
        identity::{
            AtomicSourceDescriptor, AtomicSourceKey, BinderDescriptor, BinderKey,
            CanonicalResidueConvention, GraphWireSourceKey, HashQuerySpec, HashQuerySpecId,
            HashTagPart, OccurrenceScope, ProgramKey, ResolvedIndexRange, ResolvedIntExpr,
            ResolvedMatrixType, SamplerDescriptorId, SliceSpec, SliceSpecId, WireSourceKey,
        },
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

    fn scalar_matrix_type() -> ResolvedMatrixType {
        ResolvedMatrixType {
            modulus: ResolvedIntExpr::Const(17.into()),
            ring_dimension: ResolvedIntExpr::Const(1.into()),
            rows: ResolvedIntExpr::Const(1.into()),
            columns: ResolvedIntExpr::Const(1.into()),
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

    struct UnaryAddBoundInput {
        nodes: BTreeMap<Id, MxxLang>,
        source: AtomicSourceId,
        bound: MatrixBound,
    }

    impl BoundInput for UnaryAddBoundInput {
        fn node(&self, term: Id) -> Option<&MxxLang> {
            self.nodes.get(&term)
        }

        fn matrix_type(
            &self,
            _: Id,
        ) -> Result<mxx_ir_core::types::ConcreteMatrixType, BoundEvaluationError> {
            Ok(self.bound.matrix_type.clone())
        }

        fn atom_bound(
            &self,
            source: AtomicSourceId,
            term: Id,
        ) -> Result<MatrixBound, BoundEvaluationError> {
            (source == self.source)
                .then(|| self.bound.clone())
                .ok_or(BoundEvaluationError::MissingInputBoundContract { term })
        }

        fn matrix_constant(
            &self,
            _: super::super::identity::MatrixConstantSpecId,
            term: Id,
        ) -> Result<
            (mxx_ir_core::types::ConcreteMatrixType, ResolvedMatrixConstant),
            BoundEvaluationError,
        > {
            Err(BoundEvaluationError::InvalidMatrixConstant { term })
        }

        fn scalar_maximum_absolute(
            &self,
            term: Id,
        ) -> Result<num_bigint::BigUint, BoundEvaluationError> {
            Err(BoundEvaluationError::InvalidMatrixScale { term })
        }

        fn lift_constant_polynomial_class(
            &self,
            term: Id,
            _: Id,
        ) -> Result<BoundClass, BoundEvaluationError> {
            Err(BoundEvaluationError::InvalidMatrixScale { term })
        }

        fn crt_coefficients(
            &self,
            _: super::super::identity::CrtSpecId,
            term: Id,
        ) -> Result<Box<[BigInt]>, BoundEvaluationError> {
            Err(BoundEvaluationError::InvalidCrtRecompose { term })
        }

        fn validate_pack(&self, term: Id, _: usize) -> Result<(), BoundEvaluationError> {
            Err(BoundEvaluationError::InvalidPack { term })
        }
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

        let normalized = build_affine_concat(&mut egraph, expected, &plan);
        assert!(egraph[egraph.find(normalized)].nodes.iter().any(|node| {
            matches!(node, MxxLang::MatrixAdd(terms)
                if terms.iter().any(|term| egraph.find(*term) == egraph.find(outside)))
        }));

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
    fn target_add_splice_preserves_ordered_product_factors() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (prefix, _) = matrix_atom(&mut egraph, "prefix", None);
        let (left, _) = matrix_atom(&mut egraph, "left", None);
        let (right, _) = matrix_atom(&mut egraph, "right", None);
        let (error, _) = matrix_atom(&mut egraph, "error", None);
        let (suffix, _) = matrix_atom(&mut egraph, "suffix", None);
        let signal = egraph.add(MxxLang::MatrixMultiply(vec![left, right].into()));
        let target = egraph.add(MxxLang::MatrixAdd(vec![signal, error].into()));
        let replacement = target_spliced_product(&mut egraph, &[prefix], &[], target, &[suffix]);
        assert!(egraph[egraph.find(replacement)].nodes.iter().any(|node| {
            matches!(node, MxxLang::MatrixAdd(terms) if terms.iter().any(|term| {
                egraph[egraph.find(*term)].nodes.iter().any(|node| matches!(node,
                    MxxLang::MatrixMultiply(factors)
                        if factors.len() == 4 &&
                            egraph.find(factors[0]) == egraph.find(prefix) &&
                            egraph.find(factors[1]) == egraph.find(left) &&
                            egraph.find(factors[2]) == egraph.find(right) &&
                            egraph.find(factors[3]) == egraph.find(suffix)))
            }))
        }));
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
            assert!(!classify_proposal_node(&egraph, node, &context).unwrap());
        }
        let node = egraph[egraph.find(matrix)].nodes.first().expect("matrix atom node");
        assert!(!classify_proposal_node(&egraph, node, &context).unwrap());
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
            !classify_proposal_node(&egraph, node, &context).expect("selector mismatch is local")
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
        assert!(
            classify_proposal_node(&egraph, node, &context).expect("same selector distributes")
        );
        assert_eq!(context.failure(), None);
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
    fn registered_relation_cancels_the_newly_equal_additive_signal_in_one_saturation() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (public, _) = matrix_atom(&mut egraph, "public", None);
        let (target, _) = matrix_atom(&mut egraph, "target", None);
        let (residual, _) = matrix_atom(&mut egraph, "residual", None);
        let (relation, source) =
            matrix_atom(&mut egraph, "relation", Some(AtomicRelationRole::Preimage));
        let product = egraph.add(MxxLang::MatrixMultiply(vec![public, relation].into()));
        let negated_target = egraph.add(MxxLang::MatrixNegate([target]));
        let root = egraph.add(MxxLang::MatrixAdd(vec![product, negated_target, residual].into()));
        let context = RewriteContext::new(SharedRewriteBudget::new());
        context.register(registration(source, public, target));
        let rewrite = egg::Rewrite::new(
            "registered-relation-exact-additive-cancellation",
            RelationSearcher::new(context.clone()),
            RelationApplier::new(context.clone()),
        )
        .expect("closed relation rewrite");

        let egraph = egg::Runner::default().with_egraph(egraph).run(&[rewrite]).egraph;
        assert!(egraph[egraph.find(root)].nodes.iter().any(|node| {
            matches!(node, MxxLang::MatrixAdd(terms)
                if terms.len() == 1 && egraph.find(terms[0]) == egraph.find(residual))
        }));
        assert_eq!(context.failure(), None);
    }

    #[test]
    fn exact_additive_cancellation_flattens_once_and_preserves_multiplicity_order() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (term, _) = matrix_atom(&mut egraph, "term", None);
        let (residual, residual_source) = matrix_atom(&mut egraph, "residual", None);
        let negated = egraph.add(MxxLang::MatrixNegate([term]));
        let nested = egraph.add(MxxLang::MatrixAdd(vec![negated, term].into()));
        let root = egraph.add(MxxLang::MatrixAdd(vec![term, nested, negated, residual].into()));
        egraph.rebuild();

        let replacement = exact_additive_remainder(&mut egraph, root)
            .expect("two exact pairs reduce to the physical residual");
        assert!(egraph[egraph.find(replacement)].nodes.iter().any(|node| {
            matches!(node, MxxLang::MatrixAdd(terms)
                if terms.len() == 1 && egraph.find(terms[0]) == egraph.find(residual))
        }));
        let wrapper = egraph[egraph.find(replacement)]
            .nodes
            .iter()
            .find(|node| matches!(node, MxxLang::MatrixAdd(terms) if terms.len() == 1))
            .expect("singleton wrapper")
            .clone();
        let residual_node = egraph[egraph.find(residual)]
            .nodes
            .iter()
            .find(|node| matches!(node, MxxLang::Atom { .. }))
            .expect("residual atom")
            .clone();
        let matrix_type = mxx_ir_core::types::ConcreteMatrixType {
            modulus: 17.into(),
            ring_dimension: 1,
            rows: 1,
            columns: 1,
        };
        let residual_bound = MatrixBound {
            matrix_type,
            coefficient_class: BoundClass::bounded(7_u8.into()),
            metadata: MatrixMetadata::unknown(),
        };
        let input = UnaryAddBoundInput {
            nodes: BTreeMap::from([(replacement, wrapper), (residual, residual_node)]),
            source: residual_source,
            bound: residual_bound.clone(),
        };
        assert_eq!(BoundEvaluator::new(&input).evaluate(replacement), Ok(residual_bound));
    }

    #[test]
    fn exact_additive_cancellation_rejects_shared_nested_add_dags() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (term, _) = matrix_atom(&mut egraph, "term", None);
        let negated = egraph.add(MxxLang::MatrixNegate([term]));
        let shared = egraph.add(MxxLang::MatrixAdd(vec![term, negated].into_boxed_slice()));
        let root = egraph.add(MxxLang::MatrixAdd(vec![shared, shared].into_boxed_slice()));
        egraph.rebuild();

        assert!(exact_additive_remainder(&mut egraph, root).is_none());
        assert!(!exact_additive_cancellation_possible(&egraph, root));
    }

    #[test]
    fn exact_additive_cancellation_rejects_cyclic_add_eclasses() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (term, _) = matrix_atom(&mut egraph, "term", None);
        let root = egraph.add(MxxLang::MatrixAdd(vec![term].into_boxed_slice()));
        egraph.union(root, term);
        let canonical_root = egraph.find(root);
        assert_eq!(canonical_root, egraph.find(term));
        assert!(matches!(
            unique_add_terms(&egraph, root).as_deref(),
            Some([child]) if egraph.find(*child) == canonical_root
        ));

        assert!(exact_additive_remainder(&mut egraph, root).is_none());
        assert!(!exact_additive_cancellation_possible(&egraph, root));
    }

    #[test]
    fn exact_additive_cancellation_requires_the_same_selector() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let first_selector = egraph.add(MxxLang::IntConst(0.into()));
        let second_selector = egraph.add(MxxLang::IntConst(1.into()));
        let (first_case, _) = matrix_atom(&mut egraph, "first-case", None);
        let (second_case, _) = matrix_atom(&mut egraph, "second-case", None);
        let first = egraph
            .add(MxxLang::Switch(vec![first_selector, first_case, second_case].into_boxed_slice()));
        let second = egraph.add(MxxLang::Switch(
            vec![second_selector, first_case, second_case].into_boxed_slice(),
        ));
        let negated_second = egraph.add(MxxLang::MatrixNegate([second]));
        let root = egraph.add(MxxLang::MatrixAdd(vec![first, negated_second].into_boxed_slice()));
        egraph.rebuild();

        assert!(exact_additive_remainder(&mut egraph, root).is_none());
    }

    #[test]
    fn exact_additive_cancellation_emits_a_typed_zero() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (term, _) = matrix_atom(&mut egraph, "term", None);
        let negated = egraph.add(MxxLang::MatrixNegate([term]));
        let root = egraph.add(MxxLang::MatrixAdd(vec![term, negated].into_boxed_slice()));
        egraph.rebuild();

        let zero =
            exact_additive_remainder(&mut egraph, root).expect("complete exact cancellation");
        assert!(egraph[egraph.find(zero)].nodes.iter().any(|node| {
            matches!(node, MxxLang::MatrixConstant(spec)
            if matches!(
                egraph.analysis.symbols.matrix_constants.get(spec.0),
                Some(super::super::identity::MatrixConstantSpec {
                    matrix_type,
                    value: MatrixConstantValue::Zero,
                }) if matrix_type == &scalar_matrix_type()
            ))
        }));
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
            Err(BinderPreflightReject::CaseNestedSwitch { case_index: 0 })
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
        assert!(cases[1..].iter().all(|case| egraph[egraph.find(*case)].nodes.iter().any(|node| matches!(node, MxxLang::MatrixConstant(spec)
            if matches!(egraph.analysis.symbols.matrix_constants.get(spec.0), Some(super::super::identity::MatrixConstantSpec { value: MatrixConstantValue::Zero, .. }))))));
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
    fn signed_ordered_monomials_accept_equivalent_add_forms_and_keep_relation_unions_atomic() {
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
                .is_some()
        );

        let (relation_target, _) = matrix_atom(&mut egraph, "monomial-relation-target", None);
        egraph.union(left_product, relation_target);
        egraph.rebuild();
        let mut progress = || Ok(());
        assert!(
            signed_ordered_monomial_spines(&egraph, &[(left_product, false)], &mut progress)
                .is_some(),
            "a relation-unioned product remains one atomic leaf"
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
            charges, 81,
            "63 baseline charges plus 18 direct eclass/factor scans; changing either traversal changes this calibrated total"
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
        let interruption = full_calls / 2;
        assert!(
            interruption > first_only_calls,
            "the calibrated halfway point is after target one and inside the large second-target DP"
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
    fn pointwise_add_switch_rejects_two_switches_and_failed_case_without_mutation() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let (signal, _) = matrix_atom(&mut egraph, "signal", None);
        let (other, _) = matrix_atom(&mut egraph, "other", None);
        let negated = egraph.add(MxxLang::MatrixNegate([signal]));
        let first = egraph.add(MxxLang::Switch(vec![selector, signal, signal].into_boxed_slice()));
        let second = egraph.add(MxxLang::Switch(vec![selector, signal, other].into_boxed_slice()));
        let root = egraph.add(MxxLang::MatrixAdd(vec![first, second, negated].into_boxed_slice()));
        egraph.rebuild();
        let before = egraph.total_size();
        assert!(pointwise_add_switch_cancellation_plan(&egraph, root).is_none());
        assert_eq!(egraph.total_size(), before);
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
        let mut cases = vec![selector];
        cases.extend(std::iter::repeat_n(signal, 32));
        let switch = egraph.add(MxxLang::Switch(cases.into_boxed_slice()));
        let root = egraph.add(MxxLang::MatrixAdd(vec![switch, negated].into_boxed_slice()));
        egraph.rebuild();
        let before = egraph.total_size();
        let plan =
            pointwise_add_switch_cancellation_plan(&egraph, root).expect("all stored cases cancel");
        let replacement = build_pointwise_add_switch_cancellation(&mut egraph, root, plan)
            .expect("direct plan builds");
        assert!(
            matches!(egraph[egraph.find(replacement)].nodes.as_slice(), [MxxLang::Switch(cases)] if cases.len() == 33)
        );
        assert!(
            egraph.total_size() <= before + 33,
            "one switch plus at most one node per stored case"
        );
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
        assert_eq!(
            pointwise_add_switch_cancellation_plan(&egraph, root).expect("duplicates match").cases,
            vec![vec![egraph.find(residual)], vec![egraph.find(residual)]]
        );

        let inner = egraph.add(MxxLang::Switch(vec![selector, signal, signal].into_boxed_slice()));
        let outer = egraph.add(MxxLang::Switch(vec![selector, inner, inner].into_boxed_slice()));
        let nested = egraph.add(MxxLang::MatrixAdd(vec![outer, negated_signal].into_boxed_slice()));
        let alternate =
            egraph.add(MxxLang::Switch(vec![selector, signal, negated_signal].into_boxed_slice()));
        egraph.union(switch, alternate);
        egraph.rebuild();
        assert_eq!(
            pointwise_add_switch_cancellation_reason(&egraph, nested),
            Err(PointwiseAddSwitchReject::CaseCycleOrNestedSwitch { case_index: 0 })
        );
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
        assert!(direct_add_terms_or_atomic(&egraph, case).is_none());
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
    fn additive_probe_reports_a_normalized_alternative_not_selected() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (term, _) = matrix_atom(&mut egraph, "term", None);
        let (residual, _) = matrix_atom(&mut egraph, "residual", None);
        let negated = egraph.add(MxxLang::MatrixNegate([term]));
        let root = egraph.add(MxxLang::MatrixAdd(vec![term, negated, residual].into_boxed_slice()));
        let normalized = egraph.add(MxxLang::MatrixAdd(vec![residual].into_boxed_slice()));
        egraph.union(root, normalized);
        egraph.rebuild();

        let selected = MxxLang::MatrixAdd(vec![term, negated, residual].into_boxed_slice());
        assert!(matches!(
            probe_exact_additive_normalization(&egraph, root, &selected),
            AddNormalizationProbe::NormalizedAlternativeNotSelected(MxxLang::MatrixAdd(terms))
                if terms.len() == 1 && egraph.find(terms[0]) == egraph.find(residual)
        ));
    }

    #[test]
    fn additive_probe_reports_shared_nested_add_without_expanding_it() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (term, _) = matrix_atom(&mut egraph, "term", None);
        let negated = egraph.add(MxxLang::MatrixNegate([term]));
        let shared = egraph.add(MxxLang::MatrixAdd(vec![term, negated].into_boxed_slice()));
        let root = egraph.add(MxxLang::MatrixAdd(vec![shared, shared].into_boxed_slice()));
        egraph.rebuild();

        let selected = MxxLang::MatrixAdd(vec![shared, shared].into_boxed_slice());
        assert_eq!(
            probe_exact_additive_normalization(&egraph, root, &selected),
            AddNormalizationProbe::CycleOrSharedNestedAdd
        );
    }

    #[test]
    fn exact_additive_cancellation_accepts_a_cancellable_competing_root_add() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (first, _) = matrix_atom(&mut egraph, "first", None);
        let first_negative = egraph.add(MxxLang::MatrixNegate([first]));
        let (third, _) = matrix_atom(&mut egraph, "third", None);
        let (fourth, _) = matrix_atom(&mut egraph, "fourth", None);
        let first_add =
            egraph.add(MxxLang::MatrixAdd(vec![first, first_negative].into_boxed_slice()));
        let second_add = egraph.add(MxxLang::MatrixAdd(vec![third, fourth].into_boxed_slice()));
        egraph.union(first_add, second_add);
        egraph.rebuild();

        let remainder = exact_additive_remainder(&mut egraph, first_add)
            .expect("one physical root Add has an exact cancellation");
        assert!(matches!(
            egraph[egraph.find(remainder)].nodes.as_slice(),
            [MxxLang::MatrixConstant(spec)]
                if matches!(egraph.analysis.symbols.matrix_constants.get(spec.0), Some(super::super::identity::MatrixConstantSpec { value: MatrixConstantValue::Zero, .. }))
        ));
        assert!(exact_additive_cancellation_possible(&egraph, first_add));
        assert_eq!(
            probe_exact_additive_normalization(
                &egraph,
                first_add,
                &MxxLang::MatrixAdd(vec![first, first_negative].into_boxed_slice()),
            ),
            AddNormalizationProbe::CompetingPhysicalAdds
        );
    }

    #[test]
    fn exact_additive_cancellation_skips_an_existing_flattened_root_before_later_zero() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (first, _) = matrix_atom(&mut egraph, "first", None);
        let first_negative = egraph.add(MxxLang::MatrixNegate([first]));
        let (residual, _) = matrix_atom(&mut egraph, "residual", None);
        let nested = egraph.add(MxxLang::MatrixAdd(vec![first, first_negative].into_boxed_slice()));
        let first_root = egraph.add(MxxLang::MatrixAdd(vec![nested, residual].into_boxed_slice()));
        let existing_remainder = egraph.add(MxxLang::MatrixAdd(vec![residual].into_boxed_slice()));
        egraph.union(first_root, existing_remainder);

        let (later, _) = matrix_atom(&mut egraph, "later", None);
        let later_negative = egraph.add(MxxLang::MatrixNegate([later]));
        let later_root =
            egraph.add(MxxLang::MatrixAdd(vec![later, later_negative].into_boxed_slice()));
        egraph.union(first_root, later_root);
        egraph.rebuild();

        assert!(exact_additive_cancellation_possible(&egraph, first_root));
        let remainder = exact_additive_remainder(&mut egraph, first_root)
            .expect("later root Add cancels after the existing flattened remainder is skipped");
        assert!(egraph[egraph.find(remainder)].nodes.iter().any(|node| {
            matches!(
                node,
                MxxLang::MatrixConstant(spec)
                    if matches!(
                        egraph.analysis.symbols.matrix_constants.get(spec.0),
                        Some(super::super::identity::MatrixConstantSpec {
                            value: MatrixConstantValue::Zero,
                            ..
                        })
                    )
            )
        }));
    }

    #[test]
    fn exact_additive_cancellation_cancels_a_grouped_direct_pair_before_opening_it() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (left, _) = matrix_atom(&mut egraph, "grouped-left", None);
        let (right, _) = matrix_atom(&mut egraph, "grouped-right", None);
        let grouped = egraph.add(MxxLang::MatrixAdd(vec![left, right].into_boxed_slice()));
        let negated_group = egraph.add(MxxLang::MatrixNegate([grouped]));
        let root = egraph.add(MxxLang::MatrixAdd(vec![grouped, negated_group].into_boxed_slice()));
        egraph.rebuild();

        let zero = exact_additive_remainder(&mut egraph, root)
            .expect("a direct grouped Add and its Negate cancel without opening either child");
        assert!(egraph[egraph.find(zero)].nodes.iter().any(|node| {
            matches!(
                node,
                MxxLang::MatrixConstant(spec)
                    if matches!(
                        egraph.analysis.symbols.matrix_constants.get(spec.0),
                        Some(super::super::identity::MatrixConstantSpec {
                            value: MatrixConstantValue::Zero,
                            ..
                        })
                    )
            )
        }));
    }

    #[test]
    fn exact_additive_cancellation_flattens_nested_negate_with_its_sign() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (left, _) = matrix_atom(&mut egraph, "nested-left", None);
        let (right, _) = matrix_atom(&mut egraph, "nested-right", None);
        let grouped = egraph.add(MxxLang::MatrixAdd(vec![left, right].into_boxed_slice()));
        let negated_group = egraph.add(MxxLang::MatrixNegate([grouped]));
        let root = egraph.add(MxxLang::MatrixAdd(vec![negated_group, left].into_boxed_slice()));
        egraph.rebuild();

        let replacement = exact_additive_remainder(&mut egraph, root)
            .expect("nested signed leaves retain the unmatched negative term");
        assert!(egraph[egraph.find(replacement)].nodes.iter().any(|node| {
            matches!(
                node,
                MxxLang::MatrixAdd(terms)
                    if terms.len() == 1
                        && matches!(
                            physical_negated_base(&egraph, terms[0]),
                            PhysicalStructure::Unique(base) if base == egraph.find(right)
                        )
            )
        }));
    }

    #[test]
    fn exact_additive_cancellation_rejects_an_ambiguous_nested_add() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (first, _) = matrix_atom(&mut egraph, "ambiguous-first", None);
        let (second, _) = matrix_atom(&mut egraph, "ambiguous-second", None);
        let first_add = egraph.add(MxxLang::MatrixAdd(vec![first].into_boxed_slice()));
        let second_add = egraph.add(MxxLang::MatrixAdd(vec![second].into_boxed_slice()));
        egraph.union(first_add, second_add);
        let (residual, _) = matrix_atom(&mut egraph, "ambiguous-residual", None);
        let root = egraph.add(MxxLang::MatrixAdd(vec![first_add, residual].into_boxed_slice()));
        egraph.rebuild();

        assert!(exact_additive_remainder(&mut egraph, root).is_none());
        assert!(!exact_additive_cancellation_possible(&egraph, root));
    }

    #[test]
    fn exact_additive_cancellation_preserves_an_unmatched_grouped_residual() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (cancelled, _) = matrix_atom(&mut egraph, "residual-cancelled", None);
        let (retained, _) = matrix_atom(&mut egraph, "residual-retained", None);
        let grouped = egraph.add(MxxLang::MatrixAdd(vec![cancelled, retained].into_boxed_slice()));
        let negated = egraph.add(MxxLang::MatrixNegate([cancelled]));
        let root = egraph.add(MxxLang::MatrixAdd(vec![grouped, negated].into_boxed_slice()));
        egraph.rebuild();

        let replacement = exact_additive_remainder(&mut egraph, root)
            .expect("the unmatched grouped term remains after signed cancellation");
        assert!(egraph[egraph.find(replacement)].nodes.iter().any(|node| {
            matches!(
                node,
                MxxLang::MatrixAdd(terms)
                    if terms.len() == 1 && egraph.find(terms[0]) == egraph.find(retained)
            )
        }));
    }

    #[test]
    fn exact_additive_cancellation_keeps_noncanonical_lookalikes_atomic() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (source_left, _) = matrix_atom(&mut egraph, "source-left", None);
        let (source_right, _) = matrix_atom(&mut egraph, "source-right", None);
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let (first_case, _) = matrix_atom(&mut egraph, "first-case", None);
        let (second_case, _) = matrix_atom(&mut egraph, "second-case", None);
        let forward_switch =
            egraph.add(MxxLang::Switch(vec![selector, first_case, second_case].into_boxed_slice()));
        let reversed_switch =
            egraph.add(MxxLang::Switch(vec![selector, second_case, first_case].into_boxed_slice()));
        let left_slice_spec = SliceSpecId(egraph.analysis.symbols.slices.intern(SliceSpec {
            rows: None,
            columns: Some(ResolvedIndexRange {
                start: ResolvedIntExpr::Const(0.into()),
                end: ResolvedIntExpr::Const(1.into()),
            }),
        }));
        let right_slice_spec = SliceSpecId(egraph.analysis.symbols.slices.intern(SliceSpec {
            rows: None,
            columns: Some(ResolvedIndexRange {
                start: ResolvedIntExpr::Const(1.into()),
                end: ResolvedIntExpr::Const(2.into()),
            }),
        }));
        let left_slice =
            egraph.add(MxxLang::MatrixSlice { spec: left_slice_spec, input: [source_left] });
        let right_slice =
            egraph.add(MxxLang::MatrixSlice { spec: right_slice_spec, input: [source_left] });
        let first_query =
            HashQuerySpecId(egraph.analysis.symbols.hash_queries.intern(HashQuerySpec {
                matrix_type: scalar_matrix_type(),
                tag_program:
                    vec![HashTagPart::Literal(vec![1].into_boxed_slice())].into_boxed_slice(),
            }));
        let second_query =
            HashQuerySpecId(egraph.analysis.symbols.hash_queries.intern(HashQuerySpec {
                matrix_type: scalar_matrix_type(),
                tag_program:
                    vec![HashTagPart::Literal(vec![2].into_boxed_slice())].into_boxed_slice(),
            }));
        let first_hash = egraph.add(MxxLang::HashPlain {
            query: first_query,
            arguments: vec![source_left].into_boxed_slice(),
        });
        let second_hash = egraph.add(MxxLang::HashPlain {
            query: second_query,
            arguments: vec![source_right].into_boxed_slice(),
        });
        let pairs = [
            (source_left, source_right),
            (forward_switch, reversed_switch),
            (left_slice, right_slice),
            (first_hash, second_hash),
        ];
        let roots = pairs
            .into_iter()
            .map(|(left, right)| {
                let negate = egraph.add(MxxLang::MatrixNegate([right]));
                egraph.add(MxxLang::MatrixAdd(vec![left, negate].into_boxed_slice()))
            })
            .collect::<Vec<_>>();
        egraph.rebuild();

        for root in roots {
            assert!(
                exact_additive_remainder(&mut egraph, root).is_none(),
                "only canonical e-class equality can cancel"
            );
        }
    }

    #[test]
    fn exact_additive_cancellation_does_not_choose_an_ambiguous_negate() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (base, _) = matrix_atom(&mut egraph, "base", None);
        let (other, _) = matrix_atom(&mut egraph, "other", None);
        let first_negate = egraph.add(MxxLang::MatrixNegate([base]));
        let second_negate = egraph.add(MxxLang::MatrixNegate([other]));
        egraph.union(first_negate, second_negate);
        let root = egraph.add(MxxLang::MatrixAdd(vec![base, first_negate].into_boxed_slice()));
        egraph.rebuild();

        assert!(exact_additive_remainder(&mut egraph, root).is_none());
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
