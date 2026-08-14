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
    identity::{
        AtomicRelationRole, AtomicSourceId, AtomicSourceKey, Axis, MatrixConstantValue,
        SamplerIdentity, TrapdoorDescriptorId,
    },
    language::MxxLang,
};
use egg::{Applier, EGraph, Id, SearchMatches, Searcher, Subst, Symbol, Var};
use num_bigint::BigInt;
use num_traits::Zero;
use std::{
    collections::{BTreeMap, BTreeSet},
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
        let matched = class.nodes.iter().any(|node| {
            let MxxLang::MatrixMultiply(factors) = node else { return false };
            factors
                .iter()
                .any(|factor| !egraph[egraph.find(*factor)].data.relation_provenance.is_empty())
        });
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

fn unique_add_terms(egraph: &EGraph<MxxLang, MxxAnalysis>, id: Id) -> Option<Box<[Id]>> {
    let matches = egraph[egraph.find(id)]
        .nodes
        .iter()
        .filter_map(|node| match node {
            MxxLang::MatrixAdd(terms) => Some(
                terms.iter().map(|term| egraph.find(*term)).collect::<Vec<_>>().into_boxed_slice(),
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
    let adds = egraph[egraph.find(id)]
        .nodes
        .iter()
        .filter_map(|node| match node {
            MxxLang::MatrixAdd(terms) => Some(
                terms.iter().map(|term| egraph.find(*term)).collect::<Vec<_>>().into_boxed_slice(),
            ),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    match adds.len() {
        0 => Some(vec![egraph.find(id)].into_boxed_slice()),
        1 => adds.into_iter().next(),
        _ => None,
    }
}

fn unique_product_factors(egraph: &EGraph<MxxLang, MxxAnalysis>, id: Id) -> Option<Box<[Id]>> {
    let matches = egraph[egraph.find(id)]
        .nodes
        .iter()
        .filter_map(|node| match node {
            MxxLang::MatrixMultiply(factors) => Some(
                factors
                    .iter()
                    .map(|factor| egraph.find(*factor))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            ),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    (matches.len() == 1).then(|| matches.into_iter().next().expect("checked singleton"))
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
        if pointwise_selector_is_distributable(egraph, public, relation)? {
            return Ok(true);
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
    use crate::operational_noise::identity::{
        AtomicSourceDescriptor, AtomicSourceKey, CanonicalResidueConvention, ResolvedIndexRange,
        ResolvedIntExpr, ResolvedMatrixType, SamplerDescriptorId, SliceSpec, SliceSpecId,
    };

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
    fn shared_budget_observes_owned_work() {
        let budget = SharedRewriteBudget::new();
        assert!(budget.reserve(1).is_ok());
        assert!(budget.reserve(1).is_ok());
        assert!(budget.reserve(1).is_ok());
        assert_eq!(budget.owned.load(Ordering::Relaxed), 3);
    }
}
