//! Exact, job-local normalisation over the expression/program arenas.
//!
//! Expression IDs remain the semantic identity, while exact polynomial
//! terms contain only compact monomial IDs.  In particular, this module has no factor identity,
//! symbolic-factor, relation-protection, or provenance authority of its own.

use super::{
    arena::{
        ExprArena, ExprId, ExprNode, HashVariant, MatrixLayout, MatrixOperation,
        ResolvedMatrixType, ResolvedValueType, SamplerOperation, ScalarOperation, ScopeProof,
        ScopedExprId, TypedConstant, ValueOperator, ValueTransformOperation,
    },
    bound::{
        BoundClass, MatrixBound as CanonicalMatrixBound, MatrixProductFacts,
        product_bound_with_facts, tensor_bound_with_facts,
    },
    facts::{
        BoundExpression, CoefficientBound, FactError, FactStore, MatrixFacts, NumericContract,
        ValueFacts,
    },
    job::{ProofReachedUniversalLhs, ReachedUniversalLhs},
    monomial::{MonomialArena, MonomialError, MonomialId, TermMap},
    program::{ArenaError, ProgramArena, ValueProgramId},
    relation::{
        CanonicalLhsKey, GadgetRecompositionRegistry, NormalizationCache, RelationRegistry,
        RelationRegistryError, RelationResolution, RuntimeSpecializationKey,
        UniversalRelationRegistration,
    },
};
use mxx_ir_core::types::ConcreteMatrixType;
use num_bigint::{BigInt, BigUint};
use num_traits::{Signed, ToPrimitive, Zero};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt,
    sync::Arc,
};

fn checked_matrix_product_output(
    left: &ResolvedMatrixType,
    right: &ResolvedMatrixType,
) -> Option<ResolvedMatrixType> {
    if left.modulus != right.modulus || left.ring_dimension != right.ring_dimension {
        return None;
    }
    let (rows, columns) = if left.rows == 1 && left.columns == 1 {
        (right.rows, right.columns)
    } else if right.rows == 1 && right.columns == 1 {
        (left.rows, left.columns)
    } else if left.columns == right.rows {
        (left.rows, right.columns)
    } else {
        return None;
    };
    Some(ResolvedMatrixType {
        modulus: left.modulus.clone(),
        ring_dimension: left.ring_dimension,
        rows,
        columns,
    })
}

/// A compact bound summary kept alongside exact terms.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BoundedSummary {
    pub coefficient_bound: NumericContract<CoefficientBound>,
}

impl BoundedSummary {
    pub fn missing() -> Self {
        Self { coefficient_bound: NumericContract::Missing }
    }

    pub fn known(bound: CoefficientBound) -> Self {
        Self { coefficient_bound: NumericContract::Known(bound) }
    }
}

/// Exact polynomial terms plus a sound summary for values which cannot be reduced further.
/// Factor lists are owned by [`MonomialArena`], not by this map.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct PolynomialNF {
    pub exact_terms: TermMap<BigInt>,
    pub bounded_summary: BoundedSummary,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ProofResolutionOwned {
    NoMatch,
    Rewrite(Arc<PolynomialNF>),
    Ambiguous { candidate_count: usize },
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RelationMatch {
    prefix: Vec<ScopedExprId>,
    suffix: Vec<ScopedExprId>,
    remaining_central: Vec<ScopedExprId>,
    rhs: super::relation::CanonicalRhsId,
}

impl PolynomialNF {
    pub fn zero() -> Self {
        Self {
            exact_terms: BTreeMap::new(),
            bounded_summary: BoundedSummary::known(CoefficientBound::ExactZero),
        }
    }

    pub fn is_zero(&self) -> bool {
        self.exact_terms.is_empty() &&
            self.bounded_summary.coefficient_bound ==
                NumericContract::Known(CoefficientBound::ExactZero)
    }

    pub fn term_count(&self) -> usize {
        self.exact_terms.len()
    }
}

/// A semantic expression together with its exact normal form and independent numeric contract.
/// The `Arc` is only ownership/lifetime management for a shared immutable map; no copy-on-write
/// or whole-map clone is used by the normaliser.
#[derive(Clone, Debug)]
pub struct AnalyzedValue {
    pub semantic: ScopedExprId,
    pub exact_nf: Option<Arc<PolynomialNF>>,
    pub coefficient_bound: NumericContract<CoefficientBound>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NormalizationCounters {
    pub nodes_processed: u64,
    pub nodes_total: u64,
    /// Number of exact terms retained by the final normalized root.
    pub final_exact_term_count: u64,
    pub remaining_use_releases: u64,
    /// Number of exact terms presented to the relation matcher.
    pub relation_candidates: u64,
    /// Number of relation matches expanded by the relation worklist.
    pub relation_applied: u64,
    /// Number of relation-bearing exact terms still retained after normalization.
    pub relation_remaining: u64,
    /// Number of finite exact terms folded into the bounded summary.
    pub bounded_fold_count: u64,
    pub peak_cached_values: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NormalizeError {
    Arena(ArenaError),
    Facts(FactError),
    Monomial(MonomialError),
    InvalidScope { expected: ValueProgramId, actual: ValueProgramId },
    MissingCachedValue { expression: ExprId },
    SharedRootCacheValue { expression: ExprId },
    UnsupportedOperator { operator: String },
    ArithmeticOverflow,
    Relation(RelationRegistryError),
}

impl fmt::Display for NormalizeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for NormalizeError {}

impl From<ArenaError> for NormalizeError {
    fn from(error: ArenaError) -> Self {
        Self::Arena(error)
    }
}

impl From<FactError> for NormalizeError {
    fn from(error: FactError) -> Self {
        Self::Facts(error)
    }
}

impl From<MonomialError> for NormalizeError {
    fn from(error: MonomialError) -> Self {
        Self::Monomial(error)
    }
}

impl From<RelationRegistryError> for NormalizeError {
    fn from(error: RelationRegistryError) -> Self {
        Self::Relation(error)
    }
}

/// The Stage 3 exact normaliser.  One instance is scoped to one finalized value program and one
/// job-owned monomial arena.  All traversal state is iterative and is released at last use.
pub struct Normalizer<'a> {
    expressions: &'a mut ExprArena,
    programs: &'a ProgramArena,
    facts: &'a FactStore,
    monomials: &'a mut MonomialArena,
    scope: ValueProgramId,
    relations: Option<&'a RelationRegistry>,
    gadget_recompositions: Option<&'a GadgetRecompositionRegistry>,
    normalization: Option<&'a mut NormalizationCache>,
    cache: BTreeMap<ExprId, Arc<AnalyzedValue>>,
    /// Durable value-level transfer results for expressions which may be released from `cache`
    /// before the root's exact monomials are folded.  This is deliberately keyed by expression
    /// identity rather than by a monomial: it does not become semantic identity or duplicate
    /// exact factor storage, and it keeps derived values such as `Slice(Gaussian)` available for
    /// the final typed bound pass.
    expression_bounds: BTreeMap<ExprId, NumericContract<CoefficientBound>>,
    remaining_uses: BTreeMap<ExprId, usize>,
    /// Normalized inputs retained by the structural gadget-recomposition hold. A decomposition
    /// factor may be paired with several exact terms of the other operand, so the held NF must be
    /// reusable rather than consumed once per pair.
    gadget_input_nfs: BTreeMap<ExprId, Arc<PolynomialNF>>,
    counters: NormalizationCounters,
    relation_rewriting_enabled: bool,
    fold_final_no_match: bool,
}

impl<'a> Normalizer<'a> {
    pub fn new(
        expressions: &'a mut ExprArena,
        programs: &'a ProgramArena,
        facts: &'a FactStore,
        monomials: &'a mut MonomialArena,
    ) -> Result<Self, NormalizeError> {
        let scope = monomials.scope();
        programs.program(scope)?;
        if facts.arena() != expressions.token() {
            return Err(NormalizeError::Facts(FactError::ForeignExpression {
                expected: expressions.token(),
                actual: facts.arena(),
            }));
        }
        Ok(Self {
            expressions,
            programs,
            facts,
            monomials,
            scope,
            relations: None,
            gadget_recompositions: None,
            normalization: None,
            cache: BTreeMap::new(),
            expression_bounds: BTreeMap::new(),
            remaining_uses: BTreeMap::new(),
            gadget_input_nfs: BTreeMap::new(),
            counters: NormalizationCounters::default(),
            relation_rewriting_enabled: true,
            fold_final_no_match: true,
        })
    }

    pub fn with_relations(
        mut self,
        relations: &'a RelationRegistry,
        normalization: &'a mut NormalizationCache,
    ) -> Self {
        self.relations = Some(relations);
        self.normalization = Some(normalization);
        self
    }

    pub fn with_gadget_recompositions(
        mut self,
        gadget_recompositions: &'a GadgetRecompositionRegistry,
    ) -> Self {
        self.gadget_recompositions = Some(gadget_recompositions);
        self
    }

    pub fn counters(&self) -> NormalizationCounters {
        self.counters
    }

    pub fn normalize(&mut self, root: ScopedExprId) -> Result<AnalyzedValue, NormalizeError> {
        if root.program() != self.scope {
            return Err(NormalizeError::InvalidScope {
                expected: self.scope,
                actual: root.program(),
            });
        }
        // Relation closure is lexical over the complete root word. Defer it until all expression
        // children have been assembled; otherwise a child `B*K` rewrite would discard the active
        // relation before its parent exposes the boundary `B*K` again.
        let relation_rewriting_enabled = self.relation_rewriting_enabled;
        self.relation_rewriting_enabled = false;
        // The root may be a beta-reduced specialization derived in this scope rather than the
        // finalized program's original root. Validate that exact root against the registered
        // signature instead of proving reachability from a different canonical root.
        let mut scope_proof = self.expressions.scope_proof(root.program(), root.expression())?;
        self.cache.clear();
        self.expression_bounds.clear();
        self.remaining_uses.clear();
        self.gadget_input_nfs.clear();
        self.counters = NormalizationCounters::default();
        let reachable = self.compute_use_counts(root.expression())?;
        self.counters.nodes_total = reachable.len() as u64;
        let mut work = vec![(root.expression(), false)];
        let mut completed = BTreeSet::new();
        while let Some((expression, expanded)) = work.pop() {
            if completed.contains(&expression) {
                continue;
            }
            let node = self.expressions.node_arc(expression)?;
            if !expanded {
                work.push((expression, true));
                // Revisit shared dependencies as needed, but complete each node once. Pushing
                // source order makes the first input the deepest dependency in the stack; its
                // own children are completed before the parent consumes it.
                for child in &node.inputs {
                    work.push((*child, false));
                }
                continue;
            }
            let value = self.evaluate_node(&mut scope_proof, expression, node.as_ref())?;
            // Keep only the compact typed transfer, not the exact NF, after a node's last use.
            // The final root fold can therefore recover bounds for released derived factors.
            self.expression_bounds.insert(expression, value.coefficient_bound.clone());
            self.counters.nodes_processed = self.counters.nodes_processed.saturating_add(1);
            self.cache.insert(expression, Arc::new(value));
            completed.insert(expression);
            self.counters.peak_cached_values =
                self.counters.peak_cached_values.max(self.cache.len() as u64);
        }
        let value = self
            .cache
            .remove(&root.expression())
            .ok_or(NormalizeError::MissingCachedValue { expression: root.expression() })?;
        let mut value = Arc::try_unwrap(value)
            .map_err(|_| NormalizeError::SharedRootCacheValue { expression: root.expression() })?;
        self.relation_rewriting_enabled = relation_rewriting_enabled;
        if self.relations.is_some() && self.relation_rewriting_enabled {
            if let Some(exact_nf) = value.exact_nf.as_mut() {
                let normal_form = Arc::make_mut(exact_nf);
                let changed = self.rewrite_closed_relations(normal_form)?;
                if changed {
                    // Relation closure replaces the old exact word. Do not carry its summary
                    // (which may be Large because of a pre-rewrite plain hash) into the rebound.
                    normal_form.bounded_summary =
                        BoundedSummary::known(CoefficientBound::ExactZero);
                    let rebound = self.bound_normal_form(normal_form)?;
                    normal_form.bounded_summary.coefficient_bound = rebound.clone();
                    value.coefficient_bound = rebound;
                }
            }
        }
        if self.fold_final_no_match && self.relations.is_some() && self.relation_rewriting_enabled {
            if let Some(exact_nf) = value.exact_nf.as_mut() {
                let normal_form = Arc::make_mut(exact_nf);
                // Compute the total from the exact factors while they are still present.  The
                // summary produced by ordinary constructors is often only a placeholder
                // (`Missing`), so copying it here would discard newly available typed transfers.
                // The finite terms are then folded without being counted a second time.
                let rebound = match &normal_form.bounded_summary.coefficient_bound {
                    NumericContract::Known(bound) => NumericContract::Known(bound.clone()),
                    NumericContract::Missing => self.bound_normal_form(normal_form)?,
                };
                self.fold_finite_no_match_terms(normal_form)?;
                normal_form.bounded_summary.coefficient_bound = rebound.clone();
                value.coefficient_bound = rebound;
                if normal_form.is_zero() {
                    value.coefficient_bound = NumericContract::Known(CoefficientBound::ExactZero);
                    normal_form.bounded_summary =
                        BoundedSummary::known(CoefficientBound::ExactZero);
                }
            }
        }
        // The relation worklist reaches a fixed point before this stage; any retained exact term
        // therefore has no applicable relation boundary.  Ambiguous/unresolved registrations are
        // intentionally fail-closed and are represented by the retained exact term itself.
        self.counters.relation_remaining = value
            .exact_nf
            .as_deref()
            .map(|normal_form| self.count_relation_remaining(normal_form))
            .unwrap_or(0);
        self.counters.final_exact_term_count =
            value.exact_nf.as_ref().map_or(0, |normal_form| normal_form.exact_terms.len() as u64);
        Ok(value)
    }

    /// Stage A is one exact dispatch lookup. Stage B substitutes the identical index expression
    /// into every plan and canonicalizes through this normalizer. Runtime results enter only the
    /// ordinary memo owned by `NormalizationCache`.
    pub fn resolve_universal(
        &mut self,
        reached: &ReachedUniversalLhs,
    ) -> Result<RelationResolution, NormalizeError> {
        let (dispatch, index, index_range, layout, monomial) = reached.parts();
        let relations =
            self.relations.ok_or(NormalizeError::Relation(RelationRegistryError::NotFrozen))?;
        let generation = relations.frozen_generation()?;
        if relations
            .universal_candidates(dispatch)?
            .is_some_and(|bucket| bucket.keys().any(|lhs| !lhs.domain.contains(index_range)))
        {
            return Err(NormalizeError::Relation(RelationRegistryError::IndexOutOfDomain));
        }
        let key = RuntimeSpecializationKey { dispatch: dispatch.clone(), index, generation };
        if let Some(cached) =
            self.normalization.as_deref().and_then(|cache| cache.runtime_get(&key)).cloned()
        {
            return super::relation::resolve_candidates(
                cached.get(&CanonicalLhsKey { layout: layout.cloned(), monomial }),
            )
            .map_err(Into::into);
        }
        let specialized = self.specialize_universal(dispatch, index, index_range)?;
        let result = super::relation::resolve_candidates(
            specialized.get(&CanonicalLhsKey { layout: layout.cloned(), monomial }),
        )?;
        self.normalization
            .as_deref_mut()
            .ok_or(NormalizeError::Relation(RelationRegistryError::InvalidCanonicalRhs))?
            .runtime_insert(key, specialized);
        Ok(result)
    }

    /// Proof-root specialization consumes its capability and uses a fresh local map. No lookup or
    /// insertion touches the ordinary runtime memo.
    pub(crate) fn resolve_universal_proof(
        &mut self,
        reached: ProofReachedUniversalLhs<'_>,
    ) -> Result<ProofResolutionOwned, NormalizeError> {
        let (reached, generation) = reached.into_parts();
        let actual = self
            .relations
            .ok_or(NormalizeError::Relation(RelationRegistryError::NotFrozen))?
            .frozen_generation()?;
        if actual != generation {
            return Err(NormalizeError::Relation(RelationRegistryError::StaleGeneration {
                expected: actual.value(),
                actual: generation.value(),
            }));
        }
        let (dispatch, index, index_range, layout, monomial) = reached.parts();
        let checkpoint = self
            .normalization
            .as_deref()
            .ok_or(NormalizeError::Relation(RelationRegistryError::InvalidCanonicalRhs))?
            .checkpoint();
        let result = self.specialize_universal(dispatch, index, index_range).and_then(|local| {
            let candidates = local.get(&CanonicalLhsKey { layout: layout.cloned(), monomial });
            match candidates {
                None => Ok(ProofResolutionOwned::NoMatch),
                Some(candidates) if candidates.len() == 1 => {
                    let rhs = *candidates.iter().next().expect("one candidate checked");
                    let owned = self
                        .normalization
                        .as_deref()
                        .ok_or(NormalizeError::Relation(
                            RelationRegistryError::InvalidCanonicalRhs,
                        ))?
                        .get_arc(rhs)?;
                    Ok(ProofResolutionOwned::Rewrite(owned))
                }
                Some(candidates) => {
                    Ok(ProofResolutionOwned::Ambiguous { candidate_count: candidates.len() })
                }
            }
        });
        self.normalization
            .as_deref_mut()
            .expect("normalization cache checked above")
            .rollback(checkpoint);
        result
    }

    fn specialize_universal(
        &mut self,
        dispatch: &super::relation::UniversalDispatchKey,
        index: ScopedExprId,
        index_range: super::arena::TrustedIndexRange,
    ) -> Result<BTreeMap<CanonicalLhsKey, BTreeSet<super::relation::CanonicalRhsId>>, NormalizeError>
    {
        let registrations = self
            .relations
            .ok_or(NormalizeError::Relation(RelationRegistryError::NotFrozen))?
            .universal_candidates(dispatch)?
            .cloned();
        let Some(registrations) = registrations else {
            return Ok(BTreeMap::new());
        };
        let mut result = BTreeMap::<CanonicalLhsKey, BTreeSet<_>>::new();
        for (static_lhs, targets) in registrations {
            if !static_lhs.domain.contains(index_range) {
                return Err(NormalizeError::Relation(RelationRegistryError::IndexOutOfDomain));
            }
            for registration in targets.into_values() {
                let (lhs, rhs) = self.specialize_registration(index, index_range, &registration)?;
                result.entry(lhs).or_default().insert(rhs);
            }
        }
        Ok(result)
    }

    fn specialize_registration(
        &mut self,
        index: ScopedExprId,
        index_range: super::arena::TrustedIndexRange,
        registration: &UniversalRelationRegistration,
    ) -> Result<(CanonicalLhsKey, super::relation::CanonicalRhsId), NormalizeError> {
        // Keep the exact family-call provenance in the LHS.  Opaque producer families must stay
        // as `ProgramCall(plan, h(i))`; beta-reducing them to their body would erase the only
        // authority tying this relation to the reached preimage selector. Reducible generated
        // families still beta-reduce through the same typed family API.
        let public_root =
            self.specialize_family_plan(registration.lhs.public_plan, index, index_range)?;
        let preimage_root =
            self.specialize_family_plan(registration.lhs.preimage_plan, index, index_range)?;
        // These plans are part of the concrete authority even though neither contributes a factor.
        let trapdoor = self.programs.beta_reduce(
            self.expressions,
            registration.lhs.trapdoor_plan,
            &[index.expression()],
        )?;
        let pairing = self.programs.beta_reduce(
            self.expressions,
            registration.lhs.public_pairing,
            &[index.expression()],
        )?;
        if self.expressions.value_type(trapdoor)? != &registration.lhs.validation.trapdoor_type ||
            self.expressions.value_type(pairing)? != &registration.lhs.validation.public_type
        {
            return Err(NormalizeError::Relation(RelationRegistryError::Validation(
                super::relation::RelationValidationError::TypeMismatch,
            )));
        }
        let (first, second) = if registration.lhs.factor_order.public_precedes_preimage {
            (public_root, preimage_root)
        } else {
            (preimage_root, public_root)
        };
        let product_root = self
            .expressions
            .intern(ValueOperator::Matrix(MatrixOperation::Multiply), Box::new([first, second]))?;
        // Canonicalize the complete specialized product through the same exact normalizer entry
        // used for the RHS.  This is important for parent-local transforms such as
        // `Slice(Tensor(Concat(...), R))`: interning the two roots directly would preserve the
        // transform as an opaque factor and make the relation depend on an implementation detail
        // of the registration path.
        // Canonicalize the relation's own LHS without applying that same frozen relation while
        // constructing its key. Relation application is reserved for the ordinary fixed-point
        // pass over a reached term; otherwise a self-shaped LHS could consume itself during
        // registration/specialization.
        let product = self.normalize_specialized_root_without_relations(product_root)?;
        let monomial = canonical_lhs_monomial(product.exact_nf.as_deref())?;
        let (_, target) = self.normalize_plan(registration.target_plan, index)?;
        let rhs = self
            .normalization
            .as_deref_mut()
            .ok_or(NormalizeError::Relation(RelationRegistryError::InvalidCanonicalRhs))?
            .intern(target)?;
        Ok((CanonicalLhsKey { layout: registration.lhs.layout.clone(), monomial }, rhs))
    }

    fn specialize_family_plan(
        &mut self,
        plan: ValueProgramId,
        index: ScopedExprId,
        index_range: super::arena::TrustedIndexRange,
    ) -> Result<ExprId, NormalizeError> {
        self.programs
            .call_family_in_range(
                self.expressions,
                super::program::FamilyValueId::from_program(plan),
                index.expression(),
                index_range,
            )
            .map_err(Into::into)
    }

    /// Normalize one already-specialized root in an isolated exact-normalizer state. Universal
    /// relation products and target plans use this same entry point so transform-aware parent-local
    /// rules and ordinary relation closure cannot diverge between the two sides.
    fn normalize_specialized_root(
        &mut self,
        root: ExprId,
    ) -> Result<AnalyzedValue, NormalizeError> {
        let proof = self.expressions.scope_proof(self.scope, root)?;
        let scoped = self.expressions.scoped_from_proof(&proof, root)?;
        let saved_cache = std::mem::take(&mut self.cache);
        // `normalize` owns a complete root-local bounds map and clears it at entry. Keep the
        // outer map out of that nested invocation, then merge newly-derived entries back after
        // restoring it. This preserves the outer typed authority without retaining a stale
        // weaker result when the nested pass derived a stronger bound for the same expression.
        let saved_expression_bounds = std::mem::take(&mut self.expression_bounds);
        let saved_uses = std::mem::take(&mut self.remaining_uses);
        let saved_gadget_input_nfs = std::mem::take(&mut self.gadget_input_nfs);
        let saved_counters = self.counters;
        let saved_fold_final_no_match = self.fold_final_no_match;
        self.fold_final_no_match = false;
        let value = self.normalize(scoped);
        self.cache = saved_cache;
        let nested_expression_bounds = std::mem::take(&mut self.expression_bounds);
        self.expression_bounds = saved_expression_bounds;
        self.merge_expression_bounds(nested_expression_bounds);
        self.remaining_uses = saved_uses;
        self.gadget_input_nfs = saved_gadget_input_nfs;
        self.counters = saved_counters;
        self.fold_final_no_match = saved_fold_final_no_match;
        value
    }

    fn normalize_specialized_root_without_relations(
        &mut self,
        root: ExprId,
    ) -> Result<AnalyzedValue, NormalizeError> {
        let previous = self.relation_rewriting_enabled;
        self.relation_rewriting_enabled = false;
        let value = self.normalize_specialized_root(root);
        self.relation_rewriting_enabled = previous;
        value
    }

    fn normalize_plan(
        &mut self,
        plan: ValueProgramId,
        index: ScopedExprId,
    ) -> Result<(ExprId, PolynomialNF), NormalizeError> {
        let root = self.programs.beta_reduce(self.expressions, plan, &[index.expression()])?;
        let value = self.normalize_specialized_root(root)?;
        let normal_form = value.exact_nf.as_deref().cloned().ok_or_else(|| {
            NormalizeError::UnsupportedOperator {
                operator: "relation plan without exact normal form".into(),
            }
        })?;
        Ok((root, normal_form))
    }

    fn compute_use_counts(&mut self, root: ExprId) -> Result<BTreeSet<ExprId>, NormalizeError> {
        let mut reachable = BTreeSet::new();
        let mut work = vec![root];
        while let Some(expression) = work.pop() {
            if !reachable.insert(expression) {
                continue;
            }
            let node = self.expressions.node(expression)?;
            for child in &node.inputs {
                *self.remaining_uses.entry(*child).or_default() += 1;
                work.push(*child);
            }
            if matches!(node.operator, ValueOperator::Matrix(MatrixOperation::Slice { .. })) {
                if let Some(input) = node.inputs.first() {
                    if let ValueOperator::Matrix(MatrixOperation::Slice {
                        row_start,
                        row_end_exclusive,
                        column_start,
                        column_end_exclusive,
                        ..
                    }) = &node.operator
                    {
                        if self.slice_is_identity(
                            *input,
                            *row_start,
                            *row_end_exclusive,
                            *column_start,
                            *column_end_exclusive,
                        )? {
                            continue;
                        }
                    }
                    let input_node = self.expressions.node(*input)?;
                    if matches!(
                        input_node.operator,
                        ValueOperator::Matrix(MatrixOperation::Concat { .. })
                    ) {
                        // An exact concat/slice inverse consumes the selected component NF after
                        // the concat itself has been evaluated. Keep one explicit use alive for
                        // each component until that classifier runs.
                        for component in &input_node.inputs {
                            *self.remaining_uses.entry(*component).or_default() += 1;
                        }
                    }
                    let (column_start, column_end) = match &node.operator {
                        ValueOperator::Matrix(MatrixOperation::Slice {
                            column_start,
                            column_end_exclusive,
                            ..
                        }) => (*column_start, *column_end_exclusive),
                        _ => continue,
                    };
                    for held in self.slice_parent_hold_inputs(*input, column_start, column_end)? {
                        *self.remaining_uses.entry(held).or_default() += 1;
                    }
                }
            }
            if matches!(
                node.operator,
                ValueOperator::Matrix(MatrixOperation::LiftConstantPolynomial { .. })
            ) {
                if let Some(source) = self.lift_extraction_source(expression, &node)? {
                    *self.remaining_uses.entry(source).or_default() += 1;
                }
            }
            if matches!(node.operator, ValueOperator::Matrix(MatrixOperation::Transpose)) {
                if let Some(child) = node.inputs.first() {
                    let child_node = self.expressions.node(*child)?;
                    if matches!(
                        child_node.operator,
                        ValueOperator::Matrix(MatrixOperation::Transpose)
                    ) {
                        if let Some(grandchild) = child_node.inputs.first() {
                            *self.remaining_uses.entry(*grandchild).or_default() += 1;
                        }
                    }
                }
            }
            if matches!(
                node.operator,
                ValueOperator::Transform(ValueTransformOperation::GadgetDecompose { .. })
            ) && self.gadget_recompositions.is_some()
            {
                if let Some(input) = node.inputs.first() {
                    // Gadget recomposition consumes the already-normalized input NF after the
                    // decomposition node has been evaluated. Keep one explicit memo use alive;
                    // this is a structural hold, not a second semantic occurrence.
                    *self.remaining_uses.entry(*input).or_default() += 1;
                }
            }
        }
        *self.remaining_uses.entry(root).or_default() += 1;
        Ok(reachable)
    }

    fn child_value(&mut self, expression: ExprId) -> Result<Arc<AnalyzedValue>, NormalizeError> {
        let value = self
            .cache
            .get(&expression)
            .cloned()
            .ok_or(NormalizeError::MissingCachedValue { expression })?;
        let remaining = self
            .remaining_uses
            .get_mut(&expression)
            .ok_or(NormalizeError::MissingCachedValue { expression })?;
        *remaining = remaining.saturating_sub(1);
        if *remaining == 0 {
            self.cache.remove(&expression);
            self.counters.remaining_use_releases =
                self.counters.remaining_use_releases.saturating_add(1);
        }
        Ok(value)
    }

    fn gadget_input_nf(
        &mut self,
        expression: ExprId,
    ) -> Result<Option<Arc<PolynomialNF>>, NormalizeError> {
        if let Some(normal_form) = self.gadget_input_nfs.get(&expression).cloned() {
            return Ok(Some(normal_form));
        }
        let value = match self.child_value(expression) {
            Ok(value) => value,
            Err(NormalizeError::MissingCachedValue { .. }) => return Ok(None),
            Err(error) => return Err(error),
        };
        let Some(normal_form) = value.exact_nf.clone() else {
            return Ok(None);
        };
        self.gadget_input_nfs.insert(expression, normal_form.clone());
        Ok(Some(normal_form))
    }

    fn evaluate_node(
        &mut self,
        scope_proof: &mut ScopeProof,
        expression: ExprId,
        node: &ExprNode,
    ) -> Result<AnalyzedValue, NormalizeError> {
        // `normalize` validates the complete root once. Every expression reaching this point was
        // discovered below that validated root, so rebuilding the scoped view is an O(1) checked
        // projection. Calling `ProgramArena::scoped` here would walk the remaining sub-DAG once
        // per node and turn a linear chain into O(N^2).
        let semantic = self.expressions.scoped_from_proof(scope_proof, expression)?;
        let mut children = Vec::with_capacity(node.inputs.len());
        for child in &node.inputs {
            children.push(self.child_value(*child)?);
        }
        let output_type = self.expressions.value_type(expression)?.clone();
        let mut value = if matches!(output_type, ResolvedValueType::Matrix(_)) {
            self.evaluate_matrix(scope_proof, semantic, expression, node, &children)?
        } else {
            self.evaluate_nonmatrix(semantic, expression, node, &children)?
        };
        if let Some(normal_form) = value.exact_nf.as_mut().and_then(Arc::get_mut) {
            if self.relations.is_some() && self.relation_rewriting_enabled {
                let changed = self.rewrite_closed_relations(normal_form)?;
                if changed {
                    normal_form.bounded_summary =
                        BoundedSummary::known(CoefficientBound::ExactZero);
                    let rebound = self.bound_normal_form(normal_form)?;
                    normal_form.bounded_summary.coefficient_bound = rebound.clone();
                    value.coefficient_bound = rebound;
                }
            }
            if normal_form.is_zero() {
                value.coefficient_bound = NumericContract::Known(CoefficientBound::ExactZero);
                normal_form.bounded_summary = BoundedSummary::known(CoefficientBound::ExactZero);
            }
        }
        Ok(value)
    }

    fn evaluate_matrix(
        &mut self,
        scope_proof: &mut ScopeProof,
        semantic: ScopedExprId,
        expression: ExprId,
        node: &ExprNode,
        children: &[Arc<AnalyzedValue>],
    ) -> Result<AnalyzedValue, NormalizeError> {
        let bound = self.matrix_bound(expression, node, children)?;
        if let ValueOperator::Matrix(operation) = &node.operator {
            if let Some(exact_nf) = self.shared_identity_nf(node, operation, children)? {
                return Ok(AnalyzedValue {
                    semantic,
                    exact_nf: Some(exact_nf),
                    coefficient_bound: bound,
                });
            }
        }
        let exact = match &node.operator {
            ValueOperator::Matrix(operation) => Some(self.matrix_operation_exact(
                scope_proof,
                semantic,
                node,
                operation,
                children,
            )?),
            ValueOperator::Scalar(ScalarOperation::LiftConstantPolynomial { .. }) => Some(
                node.inputs
                    .first()
                    .copied()
                    .and_then(|input| self.integer_constant(input))
                    .filter(Zero::is_zero)
                    .map_or_else(|| self.atom_nf(semantic), |_| Ok(PolynomialNF::zero()))?,
            ),
            ValueOperator::Transform(ValueTransformOperation::GadgetDecompose { .. }) |
            ValueOperator::Transform(ValueTransformOperation::PackPolynomialCoefficients {
                ..
            }) |
            ValueOperator::Source(_) |
            ValueOperator::Sample { .. } |
            ValueOperator::Sampler { .. } |
            ValueOperator::DeterministicHash(_) |
            ValueOperator::OpaqueFamilyElement { .. } |
            ValueOperator::ExplicitElement { .. } |
            ValueOperator::ProgramCall { .. } |
            ValueOperator::Trapdoor(_) => Some(self.atom_nf(semantic)?),
            _ => Some(self.atom_nf(semantic)?),
        };
        Ok(AnalyzedValue {
            semantic,
            exact_nf: exact.map(|normal_form| Arc::new(with_summary(normal_form, bound.clone()))),
            coefficient_bound: bound,
        })
    }

    fn shared_identity_nf(
        &mut self,
        node: &ExprNode,
        operation: &MatrixOperation,
        children: &[Arc<AnalyzedValue>],
    ) -> Result<Option<Arc<PolynomialNF>>, NormalizeError> {
        match operation {
            MatrixOperation::Transpose => {
                let Some(child) = node.inputs.first().copied() else {
                    return Ok(None);
                };
                let child_node = self.expressions.node(child)?;
                if matches!(child_node.operator, ValueOperator::Matrix(MatrixOperation::Transpose))
                {
                    if let Some(grandchild) = child_node.inputs.first().copied() {
                        return Ok(self.child_value(grandchild)?.exact_nf.clone());
                    }
                }
            }
            MatrixOperation::Slice {
                row_start,
                row_end_exclusive,
                column_start,
                column_end_exclusive,
                ..
            } => {
                if self.slice_is_identity(
                    node.inputs[0],
                    *row_start,
                    *row_end_exclusive,
                    *column_start,
                    *column_end_exclusive,
                )? {
                    return Ok(children.first().and_then(|value| value.exact_nf.clone()));
                }
            }
            MatrixOperation::IndexedSlice { .. } => {}
            MatrixOperation::View { output, layout } => {
                if let ResolvedValueType::Matrix(input) =
                    self.expressions.value_type(node.inputs[0])?
                {
                    if input == output &&
                        *layout == MatrixLayout::row_major(input.rows, input.columns)
                    {
                        return Ok(children.first().and_then(|value| value.exact_nf.clone()));
                    }
                }
            }
            _ => {}
        }
        Ok(None)
    }

    fn evaluate_nonmatrix(
        &mut self,
        semantic: ScopedExprId,
        expression: ExprId,
        node: &ExprNode,
        children: &[Arc<AnalyzedValue>],
    ) -> Result<AnalyzedValue, NormalizeError> {
        let bound = self.nonmatrix_bound(expression, node, children)?;
        Ok(AnalyzedValue { semantic, exact_nf: None, coefficient_bound: bound })
    }

    fn matrix_operation_exact(
        &mut self,
        scope_proof: &mut ScopeProof,
        semantic: ScopedExprId,
        node: &ExprNode,
        operation: &MatrixOperation,
        children: &[Arc<AnalyzedValue>],
    ) -> Result<PolynomialNF, NormalizeError> {
        match operation {
            MatrixOperation::Add | MatrixOperation::Subtract => {
                let left = children.first().and_then(|value| value.exact_nf.as_ref());
                let right = children.get(1).and_then(|value| value.exact_nf.as_ref());
                match (left, right) {
                    (Some(left), Some(right)) => {
                        self.add_nf(left, right, matches!(operation, MatrixOperation::Subtract))
                    }
                    _ => Ok(self.atom_nf(semantic)?),
                }
            }
            MatrixOperation::Negate => {
                if let Some(value) = children.first().and_then(|value| value.exact_nf.as_ref()) {
                    Ok(self.negate_nf(value))
                } else {
                    Ok(self.atom_nf(semantic)?)
                }
            }
            MatrixOperation::Scale => {
                let scalar = node.inputs.get(1).copied().and_then(|id| self.integer_constant(id));
                if let (Some(scale), Some(value)) =
                    (scalar, children.first().and_then(|value| value.exact_nf.as_ref()))
                {
                    Ok(self.scale_nf(value, &scale))
                } else {
                    Ok(self.atom_nf(semantic)?)
                }
            }
            MatrixOperation::Multiply => {
                let left = children.first().and_then(|value| value.exact_nf.as_ref());
                let right = children.get(1).and_then(|value| value.exact_nf.as_ref());
                match (left, right) {
                    (Some(left), Some(right)) => self.product_nf(scope_proof, left, right),
                    _ => Ok(self.atom_nf(semantic)?),
                }
            }
            MatrixOperation::Transpose => {
                // A double transpose is an exact structural identity for every matrix shape. A
                // general transpose of a sum is retained as one semantic atom until the later
                // relation-aware matrix-view stage supplies factor-level transpose identities.
                if let Some(child) = node.inputs.first().copied() {
                    if let ValueOperator::Matrix(MatrixOperation::Transpose) =
                        &self.expressions.node(child)?.operator
                    {
                        let grandchild = self.expressions.node(child)?.inputs.first().copied();
                        if let Some(grandchild) = grandchild {
                            return Ok(self
                                .child_value(grandchild)?
                                .exact_nf
                                .as_deref()
                                .cloned()
                                .unwrap_or_else(PolynomialNF::zero));
                        }
                    }
                }
                let Some(input) = children.first().and_then(|value| value.exact_nf.as_ref()) else {
                    return Ok(self.atom_nf(semantic)?);
                };
                self.transform_nf(scope_proof, input, ValueOperator::Matrix(operation.clone()))
            }
            MatrixOperation::Slice {
                row_start,
                row_end_exclusive,
                column_start,
                column_end_exclusive,
                ..
            } => {
                if let Some(restored) =
                    self.parent_local_slice_nf(scope_proof, semantic.expression(), node, children)?
                {
                    return Ok(restored);
                }
                let Some(input) = children.first().and_then(|value| value.exact_nf.as_ref()) else {
                    return Ok(self.atom_nf(semantic)?);
                };
                if input.is_zero() {
                    return Ok(PolynomialNF::zero());
                }
                if let Some(restored) = self.concat_slice_inverse(
                    node.inputs[0],
                    *row_start,
                    *row_end_exclusive,
                    *column_start,
                    *column_end_exclusive,
                )? {
                    return Ok(restored);
                }
                if self.slice_is_identity(
                    node.inputs[0],
                    *row_start,
                    *row_end_exclusive,
                    *column_start,
                    *column_end_exclusive,
                )? {
                    unreachable!("identity slices are shared before owned normalization")
                }
                self.transform_nf(scope_proof, input, ValueOperator::Matrix(operation.clone()))
            }
            // Binder-open coordinates are structural semantic inputs. They are never rewritten
            // by polynomial identities; cancellation still works because the complete node is
            // retained as one atom and its negation uses the same semantic ID.
            MatrixOperation::IndexedSlice { .. } => Ok(self.atom_nf(semantic)?),
            MatrixOperation::View { output, layout } => {
                let input_type = self.expressions.value_type(node.inputs[0])?;
                if let ResolvedValueType::Matrix(input_type) = input_type {
                    if input_type == output &&
                        *layout ==
                            super::arena::MatrixLayout::row_major(
                                input_type.rows,
                                input_type.columns,
                            )
                    {
                        if children.first().and_then(|value| value.exact_nf.as_ref()).is_some() {
                            unreachable!("identity views are shared before owned normalization")
                        }
                    }
                }
                let Some(input) = children.first().and_then(|value| value.exact_nf.as_ref()) else {
                    return Ok(self.atom_nf(semantic)?);
                };
                self.transform_nf(scope_proof, input, ValueOperator::Matrix(operation.clone()))
            }
            MatrixOperation::Concat { .. } => {
                self.concat_nf(scope_proof, semantic, operation, node, children)
            }
            MatrixOperation::Tensor { .. } => {
                if children
                    .iter()
                    .any(|child| child.exact_nf.as_ref().is_some_and(|nf| nf.is_zero()))
                {
                    Ok(PolynomialNF::zero())
                } else {
                    let Some(left) = children.first().and_then(|value| value.exact_nf.as_ref())
                    else {
                        return Ok(self.atom_nf(semantic)?);
                    };
                    let Some(right) = children.get(1).and_then(|value| value.exact_nf.as_ref())
                    else {
                        return Ok(self.atom_nf(semantic)?);
                    };
                    if let Some(flattened) = self.tensor_scalar_action_nf(
                        scope_proof,
                        operation,
                        node.inputs[0],
                        node.inputs[1],
                        left,
                        right,
                    )? {
                        Ok(flattened)
                    } else {
                        // A non-scalar tensor remains a tensor factor. `tensor_nf` distributes
                        // only over exact polynomial terms; it never treats matrix tensor
                        // multiplication as an ordinary scalar product.
                        self.tensor_nf(scope_proof, operation, left, right)
                    }
                }
            }
            MatrixOperation::CrtRecompose { reconstruction_coefficients, .. } => {
                if reconstruction_coefficients.len() != children.len() {
                    return Ok(self.atom_nf(semantic)?);
                }
                let mut output = PolynomialNF::zero();
                for (child, coefficient) in children.iter().zip(reconstruction_coefficients) {
                    let Some(input) = child.exact_nf.as_ref() else {
                        return Ok(self.atom_nf(semantic)?);
                    };
                    let scaled = self.scale_nf(input, coefficient);
                    output = self.add_nf(&output, &scaled, false)?;
                }
                Ok(output)
            }
            MatrixOperation::LiftConstantPolynomial { .. } => {
                if node
                    .inputs
                    .first()
                    .copied()
                    .and_then(|input| self.integer_constant(input))
                    .is_some_and(|value| value.is_zero())
                {
                    Ok(PolynomialNF::zero())
                } else if let Some(restored) =
                    self.lifted_extracted_constant_nf(semantic.expression(), node)?
                {
                    Ok(restored)
                } else {
                    Ok(self.atom_nf(semantic)?)
                }
            }
            MatrixOperation::ExtractCoefficient { .. } => Ok(self.atom_nf(semantic)?),
        }
    }

    /// `lift(extract_0(X))` is exactly `X` when `X` normalizes to a central-only polynomial:
    /// every central factor is a 1x1 constant polynomial, so the canonical coefficient at
    /// position 0 carries the complete value and the lift reproduces it. Universal preimage
    /// targets are registered over a lifted table index while the reached online selector is an
    /// extracted plaintext coefficient; without this identity the two sides of one exact
    /// relation never share a monomial. Registration-side indices are materialized as
    /// `index + 0`, so exact zero addends are peeled first.
    fn lifted_extracted_constant_nf(
        &mut self,
        expression: ExprId,
        node: &ExprNode,
    ) -> Result<Option<PolynomialNF>, NormalizeError> {
        let Some(source) = self.lift_extraction_source(expression, node)? else {
            return Ok(None);
        };
        // `compute_use_counts` holds one extra use of the extraction source for exactly this
        // splice, so the memo entry is still alive here even though the extract itself already
        // consumed its ordinary graph use.
        let value = match self.child_value(source) {
            Ok(value) => value,
            Err(NormalizeError::MissingCachedValue { .. }) => return Ok(None),
            Err(error) => return Err(error),
        };
        let Some(normal_form) = value.exact_nf.as_deref() else { return Ok(None) };
        for monomial in normal_form.exact_terms.keys() {
            if !self.monomials.descriptor(*monomial)?.ordered_factors.is_empty() {
                return Ok(None);
            }
        }
        Ok(Some(normal_form.clone()))
    }

    /// Resolve the matrix whose canonical coefficient one lift-of-extract chain reproduces:
    /// `lift(extract_0(X) + 0)` for a source `X` of exactly the lifted output type. Exact zero
    /// addends are peeled because registration-side table indices are materialized as
    /// `index + 0`. Whether `X` is central-only is decided by the caller on its normal form.
    fn lift_extraction_source(
        &self,
        expression: ExprId,
        node: &ExprNode,
    ) -> Result<Option<ExprId>, NormalizeError> {
        let Some(mut scalar) = node.inputs.first().copied() else { return Ok(None) };
        loop {
            let scalar_node = self.expressions.node(scalar)?;
            if !matches!(scalar_node.operator, ValueOperator::Scalar(ScalarOperation::Add)) {
                break;
            }
            let [left, right] = scalar_node.inputs.as_ref() else { break };
            let (left, right) = (*left, *right);
            if self.integer_constant(right).is_some_and(|value| value.is_zero()) {
                scalar = left;
            } else if self.integer_constant(left).is_some_and(|value| value.is_zero()) {
                scalar = right;
            } else {
                break;
            }
        }
        let scalar_node = self.expressions.node(scalar)?;
        let ValueOperator::ExtractCoefficient { position: 0, .. } = &scalar_node.operator else {
            return Ok(None);
        };
        let Some(source) = scalar_node.inputs.first().copied() else { return Ok(None) };
        if self.expressions.value_type(source)? != self.expressions.value_type(expression)? {
            return Ok(None);
        }
        Ok(Some(source))
    }

    fn transform_nf(
        &mut self,
        scope_proof: &mut ScopeProof,
        input: &PolynomialNF,
        descriptor: ValueOperator,
    ) -> Result<PolynomialNF, NormalizeError> {
        let mut terms = BTreeMap::new();
        for (monomial, coefficient) in &input.exact_terms {
            if coefficient.is_zero() {
                continue;
            }
            let input = self.materialize_monomial(scope_proof, *monomial)?;
            let transformed = self.expressions.intern_scoped_transform(
                scope_proof,
                descriptor.clone(),
                &[input],
            )?;
            let transformed = self.atom_monomial(Some(scope_proof), transformed)?;
            merge_term(&mut terms, transformed, coefficient.clone());
        }
        Ok(PolynomialNF { exact_terms: terms, bounded_summary: BoundedSummary::missing() })
    }

    fn tensor_nf(
        &mut self,
        scope_proof: &mut ScopeProof,
        operation: &MatrixOperation,
        left: &PolynomialNF,
        right: &PolynomialNF,
    ) -> Result<PolynomialNF, NormalizeError> {
        let mut terms = BTreeMap::new();
        let mut expressions = BTreeMap::new();
        for (left_id, left_coefficient) in &left.exact_terms {
            let left_expression = if let Some(expression) = expressions.get(left_id).copied() {
                expression
            } else {
                let expression = self.materialize_monomial(scope_proof, *left_id)?;
                expressions.insert(*left_id, expression);
                expression
            };
            for (right_id, right_coefficient) in &right.exact_terms {
                let coefficient = left_coefficient * right_coefficient;
                if coefficient.is_zero() {
                    continue;
                }
                let right_expression = if let Some(expression) = expressions.get(right_id).copied()
                {
                    expression
                } else {
                    let expression = self.materialize_monomial(scope_proof, *right_id)?;
                    expressions.insert(*right_id, expression);
                    expression
                };
                let transformed = self.expressions.intern_scoped_transform(
                    scope_proof,
                    ValueOperator::Matrix(operation.clone()),
                    &[left_expression, right_expression],
                )?;
                let transformed = self.atom_monomial(Some(scope_proof), transformed)?;
                merge_term(&mut terms, transformed, coefficient);
            }
        }
        Ok(PolynomialNF { exact_terms: terms, bounded_summary: BoundedSummary::missing() })
    }

    /// Flatten a tensor when one operand is exactly a row-major 1x1 matrix. For matrices over the
    /// same polynomial ring, `Tensor(s, A)` is the ordinary left scalar action `s * A`, while
    /// `Tensor(A, s)` is `A * s`. This exact constructor law does not require `s` to be a constant
    /// polynomial; operand order is retained, so it does not introduce a commutativity rule for
    /// general ordered factors.
    fn tensor_scalar_action_nf(
        &mut self,
        scope_proof: &ScopeProof,
        operation: &MatrixOperation,
        left_expression: ExprId,
        right_expression: ExprId,
        left: &PolynomialNF,
        right: &PolynomialNF,
    ) -> Result<Option<PolynomialNF>, NormalizeError> {
        let MatrixOperation::Tensor { output, left_layout, right_layout, output_layout } =
            operation
        else {
            return Ok(None);
        };
        let ResolvedValueType::Matrix(left_type) = self.expressions.value_type(left_expression)?
        else {
            return Ok(None);
        };
        let ResolvedValueType::Matrix(right_type) =
            self.expressions.value_type(right_expression)?
        else {
            return Ok(None);
        };
        if output.modulus != left_type.modulus ||
            output.modulus != right_type.modulus ||
            output.ring_dimension != left_type.ring_dimension ||
            output.ring_dimension != right_type.ring_dimension ||
            *left_layout != MatrixLayout::row_major(left_type.rows, left_type.columns) ||
            *right_layout != MatrixLayout::row_major(right_type.rows, right_type.columns) ||
            *output_layout != MatrixLayout::row_major(output.rows, output.columns)
        {
            return Ok(None);
        }
        let left_scalar = left_type.rows == 1 &&
            left_type.columns == 1 &&
            *left_layout == MatrixLayout::row_major(1, 1);
        let right_scalar = right_type.rows == 1 &&
            right_type.columns == 1 &&
            *right_layout == MatrixLayout::row_major(1, 1);
        let operands = if left_scalar {
            Some((left, right, true, left_type, right_type))
        } else if right_scalar {
            Some((right, left, false, right_type, left_type))
        } else {
            None
        };
        let Some((scalar_nf, other_nf, scalar_on_left, scalar_type, other_type)) = operands else {
            return Ok(None);
        };
        if output != other_type ||
            scalar_nf.exact_terms.keys().any(|id| {
                self.monomials
                    .descriptor(*id)
                    .map(|descriptor| {
                        descriptor.ordered_factors.iter().any(|factor| {
                            let expression = factor.expression();
                            !matches!(
                                self.expressions.value_type(expression),
                                Ok(ResolvedValueType::Matrix(matrix)) if matrix == scalar_type
                            )
                        })
                    })
                    .unwrap_or(true)
            })
        {
            return Ok(None);
        }
        // Tensor scalar action is the sole authority for treating arbitrary polynomial-ring
        // 1x1 factors as commuting scalars. Reclassify each exact scalar term independently:
        // coefficients and multiplicities are preserved, and no opaque scalar expression is
        // materialized. Ordinary matrix products retain their ordered 1x1 factors.
        let mut reclassified_terms = BTreeMap::new();
        for (monomial, coefficient) in &scalar_nf.exact_terms {
            if coefficient.is_zero() {
                continue;
            }
            let (mut central, ordered) = {
                let descriptor = self.monomials.descriptor(*monomial)?;
                (descriptor.central_factors.to_vec(), descriptor.ordered_factors.to_vec())
            };
            central.extend_from_slice(&ordered);
            let reclassified = self.monomials.intern_with_proof(
                self.expressions,
                self.programs,
                scope_proof,
                &central,
                &[],
            )?;
            merge_term(&mut reclassified_terms, reclassified, coefficient.clone());
        }
        let reclassified_scalar = PolynomialNF {
            exact_terms: reclassified_terms,
            bounded_summary: BoundedSummary::missing(),
        };
        let (first, second) = if scalar_on_left {
            (&reclassified_scalar, other_nf)
        } else {
            (other_nf, &reclassified_scalar)
        };
        self.product_nf(scope_proof, first, second).map(Some)
    }

    fn concat_nf(
        &mut self,
        scope_proof: &mut ScopeProof,
        semantic: ScopedExprId,
        operation: &MatrixOperation,
        node: &ExprNode,
        children: &[Arc<AnalyzedValue>],
    ) -> Result<PolynomialNF, NormalizeError> {
        if children.iter().any(|child| child.exact_nf.is_none()) {
            return self.atom_nf(semantic);
        }

        let mut zero_inputs = Vec::new();
        zero_inputs.try_reserve(children.len()).map_err(|_| NormalizeError::ArithmeticOverflow)?;
        for input in &node.inputs {
            let ResolvedValueType::Matrix(input_type) = self.expressions.value_type(*input)? else {
                return self.atom_nf(semantic);
            };
            zero_inputs.push(self.zero_matrix(scope_proof, input_type.clone())?);
        }

        let mut terms = BTreeMap::new();
        for (position, child) in children.iter().enumerate() {
            let input = child.exact_nf.as_ref().expect("checked above");
            for (monomial, coefficient) in &input.exact_terms {
                if coefficient.is_zero() {
                    continue;
                }
                let expression = self.materialize_monomial(scope_proof, *monomial)?;
                let mut inputs = zero_inputs.clone();
                inputs[position] = expression;
                let transformed = self.expressions.intern_scoped_transform(
                    scope_proof,
                    ValueOperator::Matrix(operation.clone()),
                    &inputs,
                )?;
                let transformed = self.atom_monomial(Some(scope_proof), transformed)?;
                merge_term(&mut terms, transformed, coefficient.clone());
            }
        }
        Ok(PolynomialNF { exact_terms: terms, bounded_summary: BoundedSummary::missing() })
    }

    fn zero_matrix(
        &mut self,
        scope_proof: &mut ScopeProof,
        output: ResolvedMatrixType,
    ) -> Result<ScopedExprId, NormalizeError> {
        let zero = self.expressions.intern_scoped_transform(
            scope_proof,
            ValueOperator::Constant(TypedConstant::int(0)),
            &[],
        )?;
        Ok(self.expressions.intern_scoped_transform(
            scope_proof,
            ValueOperator::Matrix(MatrixOperation::LiftConstantPolynomial {
                output,
                coefficient_bits: 1,
            }),
            &[zero],
        )?)
    }

    fn materialize_monomial(
        &mut self,
        scope_proof: &mut ScopeProof,
        monomial: MonomialId,
    ) -> Result<ScopedExprId, NormalizeError> {
        let (central_count, ordered_count) = {
            let descriptor = self.monomials.descriptor(monomial)?;
            (descriptor.central_factors.len(), descriptor.ordered_factors.len())
        };
        let mut central = None;
        for position in 0..central_count {
            let factor = self.monomials.descriptor(monomial)?.central_factors[position];
            central = Some(if let Some(accumulator) = central {
                self.expressions.intern_scoped_transform(
                    scope_proof,
                    ValueOperator::Matrix(MatrixOperation::Multiply),
                    &[accumulator, factor],
                )?
            } else {
                factor
            });
        }
        let mut ordered = None;
        for position in 0..ordered_count {
            let factor = self.monomials.descriptor(monomial)?.ordered_factors[position];
            ordered = Some(if let Some(accumulator) = ordered {
                self.expressions.intern_scoped_transform(
                    scope_proof,
                    ValueOperator::Matrix(MatrixOperation::Multiply),
                    &[accumulator, factor],
                )?
            } else {
                factor
            });
        }
        match (central, ordered) {
            (Some(central), Some(ordered)) => {
                let ResolvedValueType::Matrix(central_type) =
                    self.expressions.value_type(central.expression())?.clone()
                else {
                    return Err(NormalizeError::UnsupportedOperator {
                        operator: "non-matrix central monomial factor".to_owned(),
                    });
                };
                let ResolvedValueType::Matrix(ordered_type) =
                    self.expressions.value_type(ordered.expression())?.clone()
                else {
                    return Err(NormalizeError::UnsupportedOperator {
                        operator: "non-matrix ordered monomial factor".to_owned(),
                    });
                };
                Ok(self.expressions.intern_scoped_transform(
                    scope_proof,
                    ValueOperator::Matrix(MatrixOperation::Tensor {
                        output: ordered_type.clone(),
                        left_layout: MatrixLayout::row_major(
                            central_type.rows,
                            central_type.columns,
                        ),
                        right_layout: MatrixLayout::row_major(
                            ordered_type.rows,
                            ordered_type.columns,
                        ),
                        output_layout: MatrixLayout::row_major(
                            ordered_type.rows,
                            ordered_type.columns,
                        ),
                    }),
                    &[central, ordered],
                )?)
            }
            (Some(central), None) => Ok(central),
            (None, Some(ordered)) => Ok(ordered),
            (None, None) => Err(NormalizeError::UnsupportedOperator {
                operator: "empty exact monomial".to_owned(),
            }),
        }
    }

    fn slice_is_identity(
        &self,
        input: ExprId,
        row_start: usize,
        row_end: usize,
        column_start: usize,
        column_end: usize,
    ) -> Result<bool, NormalizeError> {
        let ResolvedValueType::Matrix(input) = self.expressions.value_type(input)? else {
            return Ok(false);
        };
        Ok(row_start == 0 &&
            row_end == input.rows &&
            column_start == 0 &&
            column_end == input.columns)
    }

    /// Keep the operands needed by the parent-local concat projections alive until the slice is
    /// evaluated. These are explicit structural holds, consumed by `parent_local_slice_nf` even
    /// when validation fails closed.
    fn slice_parent_hold_inputs(
        &self,
        input: ExprId,
        column_start: usize,
        column_end: usize,
    ) -> Result<Vec<ExprId>, NormalizeError> {
        let node = self.expressions.node(input)?;
        let mut holds = Vec::new();
        match &node.operator {
            ValueOperator::Matrix(MatrixOperation::Multiply) if node.inputs.len() == 2 => {
                let right = self.expressions.node(node.inputs[1])?;
                if matches!(
                    right.operator,
                    ValueOperator::Matrix(MatrixOperation::Concat { axis: 1, .. })
                ) {
                    holds.push(node.inputs[0]);
                    holds.push(node.inputs[1]);
                    holds.extend(self.concat_projection_path(
                        node.inputs[1],
                        column_start,
                        column_end,
                        false,
                    )?);
                }
            }
            ValueOperator::Matrix(MatrixOperation::Tensor { .. }) if node.inputs.len() == 2 => {
                let left = self.expressions.node(node.inputs[0])?;
                if matches!(
                    left.operator,
                    ValueOperator::Matrix(MatrixOperation::Concat { axis: 1, .. })
                ) {
                    holds.push(node.inputs[0]);
                    holds.push(node.inputs[1]);
                    let ResolvedValueType::Matrix(right_type) =
                        self.expressions.value_type(node.inputs[1])?
                    else {
                        return Ok(holds);
                    };
                    if right_type.columns == 0 ||
                        column_start % right_type.columns != 0 ||
                        column_start.checked_add(right_type.columns) != Some(column_end)
                    {
                        return Ok(holds);
                    }
                    let start = column_start / right_type.columns;
                    holds.extend(self.concat_projection_path(
                        node.inputs[0],
                        start,
                        start.checked_add(1).ok_or(NormalizeError::ArithmeticOverflow)?,
                        true,
                    )?);
                }
            }
            _ => {}
        }
        Ok(holds)
    }

    fn concat_projection_path(
        &self,
        mut concat: ExprId,
        mut start: usize,
        mut end: usize,
        require_scalar: bool,
    ) -> Result<Vec<ExprId>, NormalizeError> {
        let mut path = vec![concat];
        loop {
            let Some((_, components)) = self.validated_concat_components(concat)? else {
                return Ok(Vec::new());
            };
            let Some((child, shape, child_start, child_end)) = components
                .iter()
                .find(|(_, _, child_start, child_end)| *child_start <= start && end <= *child_end)
                .cloned()
            else {
                return Ok(Vec::new());
            };
            let child_node = self.expressions.node(child)?;
            if child_start == start && child_end == end {
                if require_scalar && (shape.rows != 1 || shape.columns != 1) {
                    return Ok(Vec::new());
                }
                path.push(child);
                return Ok(path);
            }
            if shape.rows != 1 ||
                !matches!(
                    child_node.operator,
                    ValueOperator::Matrix(MatrixOperation::Concat { axis: 1, .. })
                )
            {
                return Ok(Vec::new());
            }
            start = start.checked_sub(child_start).ok_or(NormalizeError::ArithmeticOverflow)?;
            end = end.checked_sub(child_start).ok_or(NormalizeError::ArithmeticOverflow)?;
            concat = child;
            path.push(child);
        }
    }

    fn validated_concat_components(
        &self,
        concat: ExprId,
    ) -> Result<
        Option<(ResolvedMatrixType, Vec<(ExprId, ResolvedMatrixType, usize, usize)>)>,
        NormalizeError,
    > {
        let node = self.expressions.node(concat)?;
        let ValueOperator::Matrix(MatrixOperation::Concat { axis, output, layout }) =
            &node.operator
        else {
            return Ok(None);
        };
        if *axis != 1 || *layout != MatrixLayout::row_major(output.rows, output.columns) {
            return Ok(None);
        }
        let ResolvedValueType::Matrix(actual) = self.expressions.value_type(concat)? else {
            return Ok(None);
        };
        if actual != output {
            return Ok(None);
        }
        let mut offset = 0_usize;
        let mut components = Vec::new();
        for &component in &node.inputs {
            let ResolvedValueType::Matrix(shape) = self.expressions.value_type(component)? else {
                return Ok(None);
            };
            if shape.modulus != output.modulus ||
                shape.ring_dimension != output.ring_dimension ||
                shape.rows != output.rows
            {
                return Ok(None);
            }
            let end =
                offset.checked_add(shape.columns).ok_or(NormalizeError::ArithmeticOverflow)?;
            components.push((component, shape.clone(), offset, end));
            offset = end;
        }
        if offset != output.columns {
            return Ok(None);
        }
        Ok(Some((output.clone(), components)))
    }

    fn exact_concat_projection(
        &self,
        mut concat: ExprId,
        mut column_start: usize,
        mut column_end: usize,
        require_scalar: bool,
    ) -> Result<Option<(ExprId, ResolvedMatrixType)>, NormalizeError> {
        loop {
            let Some((_, components)) = self.validated_concat_components(concat)? else {
                return Ok(None);
            };
            let Some((component, shape, start, end)) = components
                .iter()
                .find(|(_, _, start, end)| *start <= column_start && column_end <= *end)
                .cloned()
            else {
                return Ok(None);
            };
            let exact = start == column_start && end == column_end;
            let component_node = self.expressions.node(component)?;
            if exact {
                if require_scalar && (shape.rows != 1 || shape.columns != 1) {
                    return Ok(None);
                }
                return Ok(Some((component, shape)));
            }
            if column_start < start || column_end > end || shape.rows != 1 {
                return Ok(None);
            }
            let ValueOperator::Matrix(MatrixOperation::Concat { .. }) = component_node.operator
            else {
                return Ok(None);
            };
            concat = component;
            column_start =
                column_start.checked_sub(start).ok_or(NormalizeError::ArithmeticOverflow)?;
            column_end = column_end.checked_sub(start).ok_or(NormalizeError::ArithmeticOverflow)?;
        }
    }

    /// Recover only the two accepted parent-local slice forms. The resulting NF is ordinary
    /// product NF, so the caller's normal relation-closure pass sees the same B/K factors as a
    /// graph-level multiplication.
    fn parent_local_slice_nf(
        &mut self,
        scope_proof: &ScopeProof,
        expression: ExprId,
        slice: &ExprNode,
        _children: &[Arc<AnalyzedValue>],
    ) -> Result<Option<PolynomialNF>, NormalizeError> {
        let ValueOperator::Matrix(MatrixOperation::Slice {
            row_start,
            row_end_exclusive,
            column_start,
            column_end_exclusive,
            layout: slice_layout,
        }) = &slice.operator
        else {
            return Ok(None);
        };
        let Some(&input) = slice.inputs.first() else {
            return Ok(None);
        };
        let holds = self.slice_parent_hold_inputs(input, *column_start, *column_end_exclusive)?;
        if holds.is_empty() {
            return Ok(None);
        }
        let held = holds
            .into_iter()
            .map(|expression| self.child_value(expression).map(|value| (expression, value)))
            .collect::<Result<Vec<_>, _>>()?;
        let value_for = |expression| {
            held.iter().find(|(id, _)| *id == expression).map(|(_, value)| value.clone())
        };
        let ResolvedValueType::Matrix(parent_type) = self.expressions.value_type(input)? else {
            return Ok(None);
        };
        let ResolvedValueType::Matrix(actual_output) = self.expressions.value_type(expression)?
        else {
            return Ok(None);
        };
        if *row_start != 0 ||
            *row_end_exclusive != parent_type.rows ||
            *slice_layout != MatrixLayout::row_major(actual_output.rows, actual_output.columns)
        {
            return Ok(None);
        }
        let Some(slice_columns) = column_end_exclusive.checked_sub(*column_start) else {
            return Ok(None);
        };
        if *column_end_exclusive > parent_type.columns ||
            actual_output.rows != parent_type.rows ||
            actual_output.columns != slice_columns ||
            slice_columns == 0
        {
            return Ok(None);
        }
        let parent = self.expressions.node(input)?;
        match &parent.operator {
            ValueOperator::Matrix(MatrixOperation::Multiply) if parent.inputs.len() == 2 => {
                let left = parent.inputs[0];
                let concat = parent.inputs[1];
                let Some((_, _components)) = self.validated_concat_components(concat)? else {
                    return Ok(None);
                };
                let Some((component, component_type)) = self.exact_concat_projection(
                    concat,
                    *column_start,
                    *column_end_exclusive,
                    false,
                )?
                else {
                    return Ok(None);
                };
                let ResolvedValueType::Matrix(left_type) = self.expressions.value_type(left)?
                else {
                    return Ok(None);
                };
                let Some(expected) = checked_matrix_product_output(left_type, &component_type)
                else {
                    return Ok(None);
                };
                if expected != *actual_output {
                    return Ok(None);
                }
                let Some(left_value) = value_for(left) else { return Ok(None) };
                let Some(component_value) = value_for(component) else { return Ok(None) };
                let (Some(left_nf), Some(component_nf)) =
                    (left_value.exact_nf.as_ref(), component_value.exact_nf.as_ref())
                else {
                    return Ok(None);
                };
                Ok(Some(self.product_nf(scope_proof, left_nf, component_nf)?))
            }
            ValueOperator::Matrix(MatrixOperation::Tensor {
                output,
                left_layout,
                right_layout,
                output_layout,
            }) if parent.inputs.len() == 2 => {
                let concat = parent.inputs[0];
                let right = parent.inputs[1];
                let Some((concat_type, _components)) = self.validated_concat_components(concat)?
                else {
                    return Ok(None);
                };
                let ResolvedValueType::Matrix(right_type) = self.expressions.value_type(right)?
                else {
                    return Ok(None);
                };
                let expected_rows = concat_type
                    .rows
                    .checked_mul(right_type.rows)
                    .ok_or(NormalizeError::ArithmeticOverflow)?;
                let expected_columns = concat_type
                    .columns
                    .checked_mul(right_type.columns)
                    .ok_or(NormalizeError::ArithmeticOverflow)?;
                let expected = ResolvedMatrixType {
                    modulus: concat_type.modulus.clone(),
                    ring_dimension: concat_type.ring_dimension,
                    rows: expected_rows,
                    columns: expected_columns,
                };
                if *output != expected ||
                    *left_layout !=
                        MatrixLayout::row_major(concat_type.rows, concat_type.columns) ||
                    *right_layout != MatrixLayout::row_major(right_type.rows, right_type.columns) ||
                    *output_layout != MatrixLayout::row_major(expected.rows, expected.columns) ||
                    concat_type.rows != 1
                {
                    return Ok(None);
                }
                if column_start % right_type.columns != 0 ||
                    column_start.checked_add(right_type.columns) != Some(*column_end_exclusive)
                {
                    return Ok(None);
                }
                // This rewrite is limited to a concat made entirely of scalar blocks. A
                // selected 1x1 child in a mixed-shape concat is not enough to establish the
                // tensor's parent-local layout contract.
                let Some((component, component_type)) = self.exact_concat_projection(
                    concat,
                    column_start / right_type.columns,
                    column_start
                        .checked_div(right_type.columns)
                        .and_then(|start| start.checked_add(1))
                        .ok_or(NormalizeError::ArithmeticOverflow)?,
                    true,
                )?
                else {
                    return Ok(None);
                };
                let _ = component_type;
                let Some(component_value) = value_for(component) else { return Ok(None) };
                let Some(right_value) = value_for(right) else { return Ok(None) };
                let (Some(component_nf), Some(right_nf)) =
                    (component_value.exact_nf.as_ref(), right_value.exact_nf.as_ref())
                else {
                    return Ok(None);
                };
                Ok(Some(self.product_nf(scope_proof, component_nf, right_nf)?))
            }
            _ => Ok(None),
        }
    }

    fn concat_slice_inverse(
        &mut self,
        input: ExprId,
        row_start: usize,
        row_end: usize,
        column_start: usize,
        column_end: usize,
    ) -> Result<Option<PolynomialNF>, NormalizeError> {
        let concat = self.expressions.node_arc(input)?;
        let ValueOperator::Matrix(MatrixOperation::Concat { axis, .. }) = &concat.operator else {
            return Ok(None);
        };
        let mut row_offset = 0_usize;
        let mut column_offset = 0_usize;
        let mut restored = None;
        for child in &concat.inputs {
            let ResolvedValueType::Matrix(shape) = self.expressions.value_type(*child)? else {
                return Ok(None);
            };
            let child_row_start = if *axis == 1 { 0 } else { row_offset };
            let child_column_start = if *axis == 0 { 0 } else { column_offset };
            let child_row_end = child_row_start
                .checked_add(shape.rows)
                .ok_or(NormalizeError::ArithmeticOverflow)?;
            let child_column_end = child_column_start
                .checked_add(shape.columns)
                .ok_or(NormalizeError::ArithmeticOverflow)?;
            let exact = row_start == child_row_start &&
                row_end == child_row_end &&
                column_start == child_column_start &&
                column_end == child_column_end;
            if exact {
                let value = self.child_value(*child)?;
                restored = value.exact_nf.as_ref().map(|normal_form| (**normal_form).clone());
            } else {
                // Consume the structural-use hold installed by `compute_use_counts` even for
                // disjoint and partially overlapping blocks.
                self.child_value(*child)?;
            }
            if *axis != 1 {
                row_offset = child_row_end;
            }
            if *axis != 0 {
                column_offset = child_column_end;
            }
        }
        Ok(restored)
    }

    fn atom_nf(&mut self, semantic: ScopedExprId) -> Result<PolynomialNF, NormalizeError> {
        let proof = self.expressions.scope_proof(self.scope, semantic.expression())?;
        let id = self.atom_monomial(Some(&proof), semantic)?;
        let mut terms = BTreeMap::new();
        terms.insert(id, BigInt::from(1_u8));
        Ok(PolynomialNF { exact_terms: terms, bounded_summary: BoundedSummary::missing() })
    }

    fn atom_monomial(
        &mut self,
        scope_proof: Option<&ScopeProof>,
        semantic: ScopedExprId,
    ) -> Result<MonomialId, NormalizeError> {
        let expression_type = self.expressions.value_type(semantic.expression())?;
        let ResolvedValueType::Matrix(matrix) = expression_type else {
            return Err(NormalizeError::UnsupportedOperator {
                operator: "non-matrix atom".to_owned(),
            });
        };
        let mut central = Vec::new();
        let mut ordered = Vec::new();
        if matrix.rows == 1 &&
            matrix.columns == 1 &&
            self.central_scalar_fact(semantic.expression(), matrix)
        {
            central.push(semantic);
        } else {
            ordered.push(semantic);
        }
        Ok(if let Some(scope_proof) = scope_proof {
            self.monomials.intern_with_proof(
                self.expressions,
                self.programs,
                scope_proof,
                &central,
                &ordered,
            )?
        } else {
            self.monomials.intern(self.expressions, self.programs, &central, &ordered)?
        })
    }

    fn central_scalar_fact(&self, expression: ExprId, matrix: &ResolvedMatrixType) -> bool {
        let direct = match self.facts.facts(expression) {
            Ok(ValueFacts::Matrix(facts)) => Some(facts),
            _ => None,
        };
        direct.or_else(|| self.program_call_matrix_facts(expression)).is_some_and(|facts| {
            facts.matrix_type == *matrix &&
                facts.metadata.is_constant_polynomial &&
                facts.metadata.layout == MatrixLayout::row_major(1, 1)
        })
    }

    /// Resolve matrix facts carried by the exact opaque family program behind a call.  Explicit
    /// family calls created while a program body is open cannot be inserted into the global
    /// expression-keyed fact store, so the family record is the scope-safe authority.  Opaque
    /// source and preimage families have no such summary and deliberately return `None`.
    fn program_call_matrix_facts(&self, expression: ExprId) -> Option<&MatrixFacts> {
        self.programs.program_call_family_matrix_facts(self.expressions, expression).ok().flatten()
    }

    fn add_nf(
        &mut self,
        left: &PolynomialNF,
        right: &PolynomialNF,
        subtract: bool,
    ) -> Result<PolynomialNF, NormalizeError> {
        let mut terms = BTreeMap::new();
        for (id, coefficient) in &left.exact_terms {
            terms.insert(*id, coefficient.clone());
        }
        for (id, coefficient) in &right.exact_terms {
            let signed = if subtract { -coefficient } else { coefficient.clone() };
            let entry = terms.entry(*id).or_insert_with(|| BigInt::from(0_u8));
            *entry += signed;
            if entry.is_zero() {
                terms.remove(id);
            }
        }
        Ok(PolynomialNF { exact_terms: terms, bounded_summary: BoundedSummary::missing() })
    }

    fn negate_nf(&self, value: &PolynomialNF) -> PolynomialNF {
        let mut terms = BTreeMap::new();
        for (id, coefficient) in &value.exact_terms {
            terms.insert(*id, -coefficient.clone());
        }
        PolynomialNF { exact_terms: terms, bounded_summary: BoundedSummary::missing() }
    }

    fn scale_nf(&self, value: &PolynomialNF, scale: &BigInt) -> PolynomialNF {
        if scale.is_zero() {
            return PolynomialNF::zero();
        }
        let mut terms = BTreeMap::new();
        for (id, coefficient) in &value.exact_terms {
            let result = coefficient * scale;
            if !result.is_zero() {
                terms.insert(*id, result);
            }
        }
        PolynomialNF { exact_terms: terms, bounded_summary: BoundedSummary::missing() }
    }

    fn product_nf(
        &mut self,
        scope_proof: &ScopeProof,
        left: &PolynomialNF,
        right: &PolynomialNF,
    ) -> Result<PolynomialNF, NormalizeError> {
        let mut worklist = VecDeque::new();
        for (left_id, left_coefficient) in &left.exact_terms {
            for (right_id, right_coefficient) in &right.exact_terms {
                let product = self.product_monomials(scope_proof, *left_id, *right_id)?;
                let coefficient = left_coefficient * right_coefficient;
                if coefficient.is_zero() {
                    continue;
                }
                worklist.push_back((product, coefficient));
            }
        }
        let mut terms = BTreeMap::new();
        while let Some((monomial, coefficient)) = worklist.pop_front() {
            if coefficient.is_zero() {
                continue;
            }
            let Some(rewritten) = self.rewrite_gadget_decomposition(monomial)? else {
                merge_term(&mut terms, monomial, coefficient);
                continue;
            };
            // Process every newly spliced NF term through the same deterministic queue. This
            // closes multiple adjacent gadget/decomposition pairs without ever reifying `A` as
            // an opaque raw expression factor.
            for (rewritten_monomial, rewritten_coefficient) in
                rewritten.exact_terms.into_iter().rev()
            {
                worklist
                    .push_front((rewritten_monomial, coefficient.clone() * rewritten_coefficient));
            }
        }
        Ok(PolynomialNF { exact_terms: terms, bounded_summary: BoundedSummary::missing() })
    }

    /// Apply the checked algebraic identity `G(base, small) * D(A) = A`. The relation is
    /// recognized only for the exact typed gadget source and the exact decomposition transform
    /// already present in this ordered word; no same-shaped source search is performed.
    fn rewrite_gadget_decomposition(
        &mut self,
        monomial: MonomialId,
    ) -> Result<Option<PolynomialNF>, NormalizeError> {
        let (central_factors, ordered_factors) = {
            let descriptor = self.monomials.descriptor(monomial)?;
            (descriptor.central_factors.to_vec(), descriptor.ordered_factors.to_vec())
        };
        let ordered = ordered_factors.as_slice();
        for index in 0..ordered.len().saturating_sub(1) {
            let gadget = ordered[index];
            let decomposition = ordered[index + 1];
            let decomposition_node = self.expressions.node(decomposition.expression())?;
            let ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
                base,
                small,
                digit_count,
                ..
            }) = &decomposition_node.operator
            else {
                continue;
            };
            let gadget_node = self.expressions.node(gadget.expression())?;
            let Some(super::arena::MatrixConstantKind::Gadget {
                base: gadget_base,
                small: gadget_small,
            }) = gadget_node.operator.source_matrix_constant()
            else {
                continue;
            };
            if gadget_base != base || gadget_small != small {
                continue;
            }
            let Some(input) = decomposition_node.inputs.first().copied() else {
                continue;
            };
            let super::arena::ResolvedValueType::Matrix(input_type) =
                self.expressions.value_type(input)?
            else {
                continue;
            };
            let super::arena::ResolvedValueType::Matrix(gadget_type) =
                self.expressions.value_type(gadget.expression())?
            else {
                continue;
            };
            let super::arena::ResolvedValueType::Matrix(decomposition_type) =
                self.expressions.value_type(decomposition.expression())?
            else {
                continue;
            };
            let super::arena::ResolvedValueType::Matrix(output_type) =
                self.expressions.value_type(input)?
            else {
                continue;
            };
            let layout = |expression| match self.facts.facts(expression) {
                Ok(ValueFacts::Matrix(facts)) => Some(facts.metadata.layout.clone()),
                _ => None,
            };
            let Some(registry) = self.gadget_recompositions else {
                continue;
            };
            if !registry.allows(
                *base,
                *small,
                *digit_count,
                gadget_type,
                decomposition_type,
                input_type,
                output_type,
                layout(gadget.expression()).as_ref(),
                layout(decomposition.expression()).as_ref(),
                layout(input).as_ref(),
            ) {
                continue;
            }
            // `D(A)` itself is an atom in the child NF, but the identity exposes the already
            // normalized polynomial NF of `A`, not the raw input expression. The use-count hold
            // installed during traversal keeps this memo entry alive until this splice.
            let Some(input_nf) = self.gadget_input_nf(input)? else { return Ok(None) };
            let left = if central_factors.is_empty() && index == 0 {
                None
            } else {
                Some(self.monomials.intern(
                    self.expressions,
                    self.programs,
                    &central_factors,
                    &ordered_factors[..index],
                )?)
            };
            let suffix = if index + 2 == ordered_factors.len() {
                None
            } else {
                Some(self.monomials.intern(
                    self.expressions,
                    self.programs,
                    &[],
                    &ordered_factors[index + 2..],
                )?)
            };
            let mut terms = BTreeMap::new();
            for (input_monomial, input_coefficient) in &input_nf.exact_terms {
                let mut replacement = *input_monomial;
                if let Some(left) = left {
                    replacement = self.monomials.combine_interned(self.scope, left, replacement)?;
                }
                if let Some(suffix) = suffix {
                    replacement =
                        self.monomials.combine_interned(self.scope, replacement, suffix)?;
                }
                merge_term(&mut terms, replacement, input_coefficient.clone());
            }
            return Ok(Some(PolynomialNF {
                exact_terms: terms,
                bounded_summary: BoundedSummary::missing(),
            }));
        }
        Ok(None)
    }

    fn product_monomials(
        &mut self,
        _scope_proof: &ScopeProof,
        left: MonomialId,
        right: MonomialId,
    ) -> Result<MonomialId, NormalizeError> {
        Ok(self.monomials.combine_interned(self.scope, left, right)?)
    }

    /// Close exact terms by repeatedly rewriting the leftmost adjacent subword. Recombined RHS
    /// terms go back onto the same deterministic worklist, so prefix, suffix, central factors,
    /// and coefficient multiplication all remain part of the next match.
    fn rewrite_closed_relations(
        &mut self,
        normal_form: &mut PolynomialNF,
    ) -> Result<bool, NormalizeError> {
        let initial = std::mem::take(&mut normal_form.exact_terms);
        let mut worklist = initial.into_iter().collect::<VecDeque<_>>();
        let mut result = BTreeMap::new();
        let mut changed = false;
        while let Some((monomial, coefficient)) = worklist.pop_front() {
            if coefficient.is_zero() {
                continue;
            }
            // Relation RHS splices recombine prefix, canonical RHS, and suffix words; a gadget
            // factor ending the prefix then sits adjacent to a decomposition opening the suffix.
            // Recomposition otherwise runs only under `product_nf`, so close those pairs here or
            // the spliced word can never cancel against its ordinarily-evaluated counterpart.
            if let Some(rewritten) = self.rewrite_gadget_decomposition(monomial)? {
                changed = true;
                for (rewritten_monomial, rewritten_coefficient) in
                    rewritten.exact_terms.into_iter().rev()
                {
                    worklist.push_front((
                        rewritten_monomial,
                        coefficient.clone() * rewritten_coefficient,
                    ));
                }
                continue;
            }
            self.counters.relation_candidates = self.counters.relation_candidates.saturating_add(1);
            let Some(relation_match) = self.find_relation_match(monomial)? else {
                merge_term(&mut result, monomial, coefficient);
                continue;
            };
            changed = true;
            self.counters.relation_applied = self.counters.relation_applied.saturating_add(1);
            let rhs = self
                .normalization
                .as_deref()
                .ok_or(NormalizeError::Relation(RelationRegistryError::InvalidCanonicalRhs))?
                .get_arc(relation_match.rhs)?;
            let left = if relation_match.remaining_central.is_empty() &&
                relation_match.prefix.is_empty()
            {
                None
            } else {
                Some(self.monomials.intern(
                    self.expressions,
                    self.programs,
                    &relation_match.remaining_central,
                    &relation_match.prefix,
                )?)
            };
            let suffix = if relation_match.suffix.is_empty() {
                None
            } else {
                Some(self.monomials.intern(
                    self.expressions,
                    self.programs,
                    &[],
                    &relation_match.suffix,
                )?)
            };
            let mut recombined = Vec::new();
            for (rhs_monomial, rhs_coefficient) in &rhs.exact_terms {
                let mut combined = *rhs_monomial;
                if let Some(left) = left {
                    combined = self.monomials.combine_interned(self.scope, left, combined)?;
                }
                if let Some(suffix) = suffix {
                    combined = self.monomials.combine_interned(self.scope, combined, suffix)?;
                }
                recombined.push((combined, &coefficient * rhs_coefficient));
            }
            for term in recombined.into_iter().rev() {
                worklist.push_front(term);
            }
        }
        normal_form.exact_terms = result;
        if normal_form.exact_terms.is_empty() {
            normal_form.bounded_summary = BoundedSummary::known(CoefficientBound::ExactZero);
        }
        Ok(changed)
    }

    fn fold_finite_no_match_terms(
        &mut self,
        normal_form: &mut PolynomialNF,
    ) -> Result<(), NormalizeError> {
        if normal_form.exact_terms.is_empty() {
            return Ok(());
        }
        let mut retained = BTreeMap::new();
        let mut folded = CoefficientBound::ExactZero;
        for (monomial, coefficient) in std::mem::take(&mut normal_form.exact_terms) {
            match self.bound_monomial(monomial, &coefficient)? {
                NumericContract::Known(CoefficientBound::ExactZero) => {}
                NumericContract::Known(bound @ CoefficientBound::Finite(_)) => {
                    self.counters.bounded_fold_count =
                        self.counters.bounded_fold_count.saturating_add(1);
                    folded = add_known_bounds(&folded, &bound);
                }
                NumericContract::Known(CoefficientBound::Large) | NumericContract::Missing => {
                    retained.insert(monomial, coefficient);
                }
            }
        }
        normal_form.exact_terms = retained;
        normal_form.bounded_summary.coefficient_bound =
            match (&normal_form.bounded_summary.coefficient_bound, folded) {
                (NumericContract::Known(existing), folded) => {
                    NumericContract::Known(add_known_bounds(existing, &folded))
                }
                (NumericContract::Missing, _) => NumericContract::Missing,
            };
        Ok(())
    }

    /// Count retained exact terms which still expose a uniquely dispatchable preimage call.
    /// This is deliberately a final structural scan: it does not specialize a selector, walk
    /// relation registrations, or attempt another rewrite. A plain exact residual therefore
    /// remains distinct from an unreduced relation-bearing term in diagnostics.
    fn count_relation_remaining(&self, normal_form: &PolynomialNF) -> u64 {
        let Some(relations) = self.relations else { return 0 };
        normal_form
            .exact_terms
            .keys()
            .filter(|monomial| {
                let Ok(descriptor) = self.monomials.descriptor(**monomial) else {
                    return false;
                };
                descriptor.central_factors.iter().chain(descriptor.ordered_factors.iter()).any(
                    |factor| {
                        let Ok(node) = self.expressions.node(factor.expression()) else {
                            return false;
                        };
                        let ValueOperator::ProgramCall { program } = node.operator else {
                            return false;
                        };
                        matches!(relations.dispatch_for_preimage_program(program), Ok(Some(_)))
                    },
                )
            })
            .count() as u64
    }

    fn merge_expression_bounds(
        &mut self,
        nested: BTreeMap<ExprId, NumericContract<CoefficientBound>>,
    ) {
        for (expression, incoming) in nested {
            match self.expression_bounds.get(&expression) {
                None => {
                    self.expression_bounds.insert(expression, incoming);
                }
                Some(existing) => {
                    let merged = stronger_bound(existing, &incoming);
                    if merged != *existing {
                        self.expression_bounds.insert(expression, merged);
                    }
                }
            }
        }
    }

    fn find_relation_match(
        &mut self,
        monomial: MonomialId,
    ) -> Result<Option<RelationMatch>, NormalizeError> {
        let Some(relations) = self.relations else {
            return Ok(None);
        };
        let (central, ordered) = {
            let descriptor = self.monomials.descriptor(monomial)?;
            (descriptor.central_factors.to_vec(), descriptor.ordered_factors.to_vec())
        };
        let layout = ordered.first().or_else(|| central.first()).and_then(|factor| {
            match self.facts.facts(factor.expression()) {
                Ok(ValueFacts::Matrix(facts)) => Some(facts.metadata.layout.clone()),
                _ => None,
            }
        });
        // Closed and universal relations share the same ordered-subword matcher. The whole-term
        // lookup remains a fast path, but it is not the semantic boundary of relation use.
        for candidate_layout in [layout.clone(), None] {
            let lhs = CanonicalLhsKey { layout: candidate_layout, monomial };
            if let RelationResolution::Rewrite(rhs) = relations.resolve_closed(&lhs)? {
                return Ok(Some(RelationMatch {
                    prefix: Vec::new(),
                    suffix: Vec::new(),
                    remaining_central: Vec::new(),
                    rhs,
                }));
            }
        }
        if let Some(result) = self.find_closed_subword_match(&central, &ordered)? {
            return Ok(Some(result));
        }
        self.find_universal_subword_match(&central, &ordered)
    }

    fn find_closed_subword_match(
        &mut self,
        central: &[ScopedExprId],
        ordered: &[ScopedExprId],
    ) -> Result<Option<RelationMatch>, NormalizeError> {
        let Some(relations) = self.relations else { return Ok(None) };
        // Leftmost boundary wins. At one boundary, try the longest word first so a registered
        // `B * X * K` relation is not shadowed by a shorter `X * K` relation.
        for start in 0..=ordered.len() {
            for width in (1..=ordered.len() - start).rev() {
                let Some(candidate) = self.monomials.find_interned(
                    self.scope,
                    &[],
                    &ordered[start..start + width],
                )?
                else {
                    continue
                };
                let remaining_central = central.to_vec();
                let candidate_layout = ordered[start..start + width].first().and_then(|factor| {
                    match self.facts.facts(factor.expression()) {
                        Ok(ValueFacts::Matrix(facts)) => Some(facts.metadata.layout.clone()),
                        _ => None,
                    }
                });
                for candidate_layout in [candidate_layout, None] {
                    let lhs = CanonicalLhsKey { layout: candidate_layout, monomial: candidate };
                    if let RelationResolution::Rewrite(rhs) = relations.resolve_closed(&lhs)? {
                        return Ok(Some(RelationMatch {
                            prefix: ordered[..start].to_vec(),
                            suffix: ordered[start + width..].to_vec(),
                            remaining_central,
                            rhs,
                        }));
                    }
                }
            }
        }
        Ok(None)
    }

    fn find_universal_subword_match(
        &mut self,
        central: &[ScopedExprId],
        ordered: &[ScopedExprId],
    ) -> Result<Option<RelationMatch>, NormalizeError> {
        let Some(relations) = self.relations else { return Ok(None) };
        let mut candidates = BTreeMap::<(usize, usize), BTreeSet<_>>::new();
        for (k_position, &k_factor) in ordered.iter().enumerate() {
            let node = self.expressions.node(k_factor.expression())?;
            let ValueOperator::ProgramCall { program } = node.operator else { continue };
            let Some(dispatch) = relations.dispatch_for_preimage_program(program)? else {
                continue;
            };
            let [selector] = node.inputs.as_ref() else { continue };
            let index = self.programs.scoped(self.expressions, self.scope, *selector)?;
            let Some(index_range) = self.universal_index_range(index)? else { continue };
            let specialized = self.specialized_universal_cached(dispatch, index, index_range)?;
            for (lhs, rhs_candidates) in specialized {
                let descriptor = self.monomials.descriptor(lhs.monomial)?;
                // Universal preimage relations consume an adjacent ordered word. A relation
                // whose LHS is central-only has no lexical boundary and is deliberately not
                // dispatched here.
                if descriptor.ordered_factors.is_empty() || !descriptor.central_factors.is_empty() {
                    continue;
                }
                // The relation layout belongs to the candidate subword, not to the first factor
                // of the complete monomial. A prefix may have a different view/layout; using
                // the full-term layout here would reject an otherwise exact universal match.
                if lhs.layout.is_some() {
                    let candidate_layout = descriptor
                        .ordered_factors
                        .first()
                        .or_else(|| descriptor.central_factors.first())
                        .and_then(|factor| match self.facts.facts(factor.expression()) {
                            Ok(ValueFacts::Matrix(facts)) => Some(facts.metadata.layout.clone()),
                            _ => None,
                        });
                    if lhs.layout != candidate_layout {
                        continue;
                    }
                }
                let mut lhs_k_positions = descriptor
                    .ordered_factors
                    .iter()
                    .enumerate()
                    .filter_map(|(position, factor)| (*factor == k_factor).then_some(position));
                while let Some(lhs_k_position) = lhs_k_positions.next() {
                    let ordered_len = descriptor.ordered_factors.len();
                    let Some(start) = k_position.checked_sub(lhs_k_position) else { continue };
                    let Some(end) = start.checked_add(ordered_len) else { continue };
                    if end > ordered.len() || ordered[start..end] != descriptor.ordered_factors[..]
                    {
                        continue;
                    }
                    if remove_central_subword(central, &descriptor.central_factors).is_none() {
                        continue
                    }
                    // Universal matching is selected globally, not by K occurrence or
                    // registration/map iteration order.  All universal LHSes have an empty
                    // central word, so the remaining central factors are identical for every
                    // candidate in this term; retain the computed proof only after selecting the
                    // winning span below.
                    candidates
                        .entry((start, end))
                        .or_default()
                        .extend(rhs_candidates.iter().copied());
                }
            }
        }
        let Some(((start, end), rhs_candidates)) = candidates.into_iter().min_by(
            |((left_start, left_end), _), ((right_start, right_end), _)| {
                left_start.cmp(right_start).then_with(|| {
                    right_end
                        .saturating_sub(*right_start)
                        .cmp(&left_end.saturating_sub(*left_start))
                })
            },
        ) else {
            return Ok(None);
        };
        let super::relation::RelationResolution::Rewrite(rhs) =
            super::relation::resolve_candidates(Some(&rhs_candidates))?
        else {
            return Ok(None);
        };
        Ok(Some(RelationMatch {
            prefix: ordered[..start].to_vec(),
            suffix: ordered[end..].to_vec(),
            remaining_central: central.to_vec(),
            rhs,
        }))
    }

    fn universal_index_range(
        &self,
        index: ScopedExprId,
    ) -> Result<Option<super::arena::TrustedIndexRange>, NormalizeError> {
        if let Ok(range) =
            self.facts.trusted_scoped_index_range(index.program(), index.expression())
        {
            return Ok(Some(range));
        }
        if let Ok(range) = self.facts.trusted_index_range(index.expression()) {
            return Ok(Some(range));
        }
        let node = self.expressions.node(index.expression())?;
        // Mirror the lowering-side selector authority: a closed `ExtractCoefficient` selector
        // carries its declared canonical exclusive upper bound, which is exactly the trusted
        // half-open range the family call was lowered with.
        if let ValueOperator::ExtractCoefficient {
            canonical_input_exclusive_upper: Some(upper),
            ..
        } = &node.operator
        {
            let Some(maximum_exclusive) = upper.to_u64() else { return Ok(None) };
            if maximum_exclusive == 0 {
                return Ok(None);
            }
            return Ok(Some(super::arena::TrustedIndexRange { minimum: 0, maximum_exclusive }));
        }
        if let ValueOperator::Argument { position: 0, value_type } = &node.operator {
            if *value_type != ResolvedValueType::Int {
                return Ok(None);
            }
            let program = self.programs.program(index.program())?;
            let [input] = program.signature.inputs.as_ref() else { return Ok(None) };
            return Ok(input.trusted_index_range);
        }
        let program = self.programs.program(index.program())?;
        let Some(input) = program.signature.inputs.first() else { return Ok(None) };
        let Some(input_range) = input.trusted_index_range else { return Ok(None) };
        let Some((coefficient, offset)) = self.scoped_affine_form(index.expression(), 0) else {
            return Ok(None);
        };
        let first = &coefficient * BigInt::from(input_range.minimum) + &offset;
        let second = &coefficient * BigInt::from(input_range.maximum_exclusive) + &offset;
        let (minimum, maximum_exclusive) =
            if coefficient.is_negative() { (second, first) } else { (first, second) };
        let (Some(minimum), Some(maximum_exclusive)) =
            (minimum.to_u64(), maximum_exclusive.to_u64())
        else {
            return Ok(None);
        };
        if minimum < maximum_exclusive {
            return Ok(Some(super::arena::TrustedIndexRange { minimum, maximum_exclusive }));
        }
        Ok(None)
    }

    fn scoped_affine_form(
        &self,
        expression: ExprId,
        argument_position: u32,
    ) -> Option<(BigInt, BigInt)> {
        let node = self.expressions.node(expression).ok()?;
        if let ValueOperator::Argument { position, value_type } = &node.operator {
            return (*position == argument_position && *value_type == ResolvedValueType::Int)
                .then_some((BigInt::from(1_u8), BigInt::from(0_u8)));
        }
        if let ValueOperator::Constant(TypedConstant {
            value: super::arena::ConstantValue::Int(value),
            ..
        }) = &node.operator
        {
            return Some((BigInt::from(0_u8), value.clone()));
        }
        let ValueOperator::Scalar(operation) = &node.operator else { return None };
        match operation {
            ScalarOperation::Negate if node.inputs.len() == 1 => {
                let (coefficient, offset) =
                    self.scoped_affine_form(node.inputs[0], argument_position)?;
                Some((-coefficient, -offset))
            }
            ScalarOperation::Add | ScalarOperation::Subtract | ScalarOperation::Multiply
                if node.inputs.len() == 2 =>
            {
                let left = self.scoped_affine_form(node.inputs[0], argument_position)?;
                let right = self.scoped_affine_form(node.inputs[1], argument_position)?;
                match operation {
                    ScalarOperation::Add => Some((left.0 + right.0, left.1 + right.1)),
                    ScalarOperation::Subtract => Some((left.0 - right.0, left.1 - right.1)),
                    ScalarOperation::Multiply if left.0.is_zero() => {
                        Some((right.0 * left.1.clone(), right.1 * left.1))
                    }
                    ScalarOperation::Multiply if right.0.is_zero() => {
                        Some((left.0 * right.1.clone(), left.1 * right.1))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn specialized_universal_cached(
        &mut self,
        dispatch: &super::relation::UniversalDispatchKey,
        index: ScopedExprId,
        index_range: super::arena::TrustedIndexRange,
    ) -> Result<BTreeMap<CanonicalLhsKey, BTreeSet<super::relation::CanonicalRhsId>>, NormalizeError>
    {
        let relations =
            self.relations.ok_or(NormalizeError::Relation(RelationRegistryError::NotFrozen))?;
        let generation = relations.frozen_generation()?;
        let key = RuntimeSpecializationKey { dispatch: dispatch.clone(), index, generation };
        if let Some(cached) =
            self.normalization.as_deref().and_then(|cache| cache.runtime_get(&key)).cloned()
        {
            return Ok(cached);
        }
        let specialized = self.specialize_universal(dispatch, index, index_range)?;
        self.normalization
            .as_deref_mut()
            .ok_or(NormalizeError::Relation(RelationRegistryError::InvalidCanonicalRhs))?
            .runtime_insert(key, specialized.clone());
        Ok(specialized)
    }

    /// Recompute the numeric transfer after exact relation rewriting. This is intentionally based
    /// on current exact factors; a pre-rewrite `Large` or `Missing` result is never retained.
    fn bound_normal_form(
        &self,
        normal_form: &PolynomialNF,
    ) -> Result<NumericContract<CoefficientBound>, NormalizeError> {
        if normal_form.exact_terms.is_empty() {
            return Ok(normal_form.bounded_summary.coefficient_bound.clone());
        }
        let mut total = match &normal_form.bounded_summary.coefficient_bound {
            NumericContract::Known(bound) => bound.clone(),
            NumericContract::Missing => CoefficientBound::ExactZero,
        };
        for (monomial, coefficient) in &normal_form.exact_terms {
            let NumericContract::Known(product) = self.bound_monomial(*monomial, coefficient)?
            else {
                return Ok(NumericContract::Missing);
            };
            total = add_known_bounds(&total, &product);
        }
        Ok(NumericContract::Known(total))
    }

    fn bound_monomial(
        &self,
        monomial: MonomialId,
        coefficient: &BigInt,
    ) -> Result<NumericContract<CoefficientBound>, NormalizeError> {
        let descriptor = self.monomials.descriptor(monomial)?;
        let mut product: Option<CanonicalMatrixBound> = None;
        let mut product_is_constant_polynomial = true;
        for factor in descriptor.central_factors.iter().chain(descriptor.ordered_factors.iter()) {
            let factor_bound = self.factor_bound(factor.expression())?;
            let factor_type = match self.expressions.value_type(factor.expression())? {
                ResolvedValueType::Matrix(matrix) => concrete_type(matrix),
                _ => return Ok(NumericContract::Missing),
            };
            let NumericContract::Known(factor_bound) = factor_bound else {
                return Ok(NumericContract::Missing);
            };
            let factor_facts = self.matrix_value_facts(factor.expression());
            let factor_is_constant_polynomial =
                factor_facts.is_some_and(|facts| facts.metadata.is_constant_polynomial);
            let factor_support_upper = factor_facts.and_then(|facts| match &facts.polynomial {
                NumericContract::Known(polynomial) => Some(polynomial.support_upper),
                NumericContract::Missing => None,
            });
            let factor_bound = CanonicalMatrixBound {
                matrix_type: factor_type,
                coefficient_class: canonical_class(&factor_bound),
            };
            product = Some(if let Some(left) = product {
                product_bound_with_facts(
                    &left,
                    &factor_bound,
                    &MatrixProductFacts {
                        left_is_constant_polynomial: product_is_constant_polynomial,
                        right_is_constant_polynomial: factor_is_constant_polynomial,
                        right_support_upper: factor_support_upper,
                        ..MatrixProductFacts::default()
                    },
                )
                .map_err(|_| NormalizeError::ArithmeticOverflow)?
            } else {
                factor_bound
            });
            product_is_constant_polynomial &= factor_is_constant_polynomial;
        }
        let Some(product) = product else {
            return Ok(NumericContract::Known(CoefficientBound::finite(
                coefficient.magnitude().clone(),
            )));
        };
        product_bounds_with_factor(
            &[
                NumericContract::Known(coefficient_bound(&product.coefficient_class)),
                NumericContract::Known(CoefficientBound::finite(coefficient.magnitude().clone())),
            ],
            &BigUint::from(1_u8),
        )
    }

    /// Resolve the compact value-level transfer for one exact factor.  A released child may no
    /// longer be in `cache`, so the durable expression-bound map is consulted first.  Missing
    /// entries are then filled only by typed authority; no display/debug identity is accepted as
    /// a bound source.
    fn factor_bound(
        &self,
        expression: ExprId,
    ) -> Result<NumericContract<CoefficientBound>, NormalizeError> {
        if let Some(bound) = self.expression_bounds.get(&expression) {
            if !bound.is_missing() {
                return Ok(bound.clone());
            }
        }
        if let Some(bound) = self.cache.get(&expression).map(|value| &value.coefficient_bound) {
            if !bound.is_missing() {
                return Ok(bound.clone());
            }
        }
        if let Ok(bound) = self.facts.coefficient_bound(expression) {
            if !bound.is_missing() {
                return Ok(bound.clone());
            }
        }
        if let Some(facts) = self.program_call_matrix_facts(expression) {
            if !facts.coefficient_bound.is_missing() {
                return Ok(facts.coefficient_bound.clone());
            }
        }
        let node = self.expressions.node(expression)?;
        let derived = match &node.operator {
            ValueOperator::Sampler { operation, .. } => sampler_bound(operation),
            ValueOperator::DeterministicHash(_) => NumericContract::Known(CoefficientBound::Large),
            ValueOperator::Transform(operation) => transform_bound(operation),
            ValueOperator::ProgramCall { program } => {
                self.relation_live_preimage_bound(expression, *program)?
            }
            _ => NumericContract::Missing,
        };
        Ok(derived)
    }

    /// An opaque `ProgramCall` is finite only when its exact program is the unique frozen
    /// preimage-family dispatch and that dispatch's source is the family body itself.  The source
    /// sampler's cutoff is the authority; a same-shaped or merely named program is insufficient.
    fn relation_live_preimage_bound(
        &self,
        expression: ExprId,
        program: ValueProgramId,
    ) -> Result<NumericContract<CoefficientBound>, NormalizeError> {
        let Some(relations) = self.relations else {
            return Ok(NumericContract::Missing);
        };
        let dispatch = match relations.dispatch_for_preimage_program(program) {
            Ok(Some(dispatch)) => dispatch,
            Ok(None) | Err(RelationRegistryError::AmbiguousPreimageDispatch) => {
                return Ok(NumericContract::Missing)
            }
            Err(error) => return Err(error.into()),
        };
        let [index] = self.expressions.node(expression)?.inputs.as_ref() else {
            return Ok(NumericContract::Missing);
        };
        if self.expressions.value_type(*index)? != &ResolvedValueType::Int {
            return Ok(NumericContract::Missing);
        }
        let family_body = self.programs.family_body(dispatch.preimage_family)?;
        if family_body != dispatch.preimage_source.expression {
            return Ok(NumericContract::Missing);
        }
        self.authoritative_source_bound(dispatch.preimage_source.expression)
    }

    fn authoritative_source_bound(
        &self,
        source: ExprId,
    ) -> Result<NumericContract<CoefficientBound>, NormalizeError> {
        if let Some(bound) = self.expression_bounds.get(&source) {
            if !bound.is_missing() {
                return Ok(bound.clone());
            }
        }
        if let Ok(bound) = self.facts.coefficient_bound(source) {
            if !bound.is_missing() {
                return Ok(bound.clone());
            }
        }
        let node = self.expressions.node(source)?;
        match &node.operator {
            ValueOperator::Sampler {
                operation: SamplerOperation::Preimage { max_coefficient_bound, .. },
                ..
            } => Ok(NumericContract::Known(CoefficientBound::finite(
                max_coefficient_bound.magnitude().clone(),
            ))),
            _ => Ok(NumericContract::Missing),
        }
    }

    fn matrix_bound(
        &self,
        expression: ExprId,
        node: &ExprNode,
        children: &[Arc<AnalyzedValue>],
    ) -> Result<NumericContract<CoefficientBound>, NormalizeError> {
        if let Some(bound) = self.fact_bound(expression)? {
            return Ok(bound);
        }
        let child_bounds =
            children.iter().map(|value| value.coefficient_bound.clone()).collect::<Vec<_>>();
        let bound = match &node.operator {
            ValueOperator::Matrix(operation) => {
                self.matrix_operation_bound(operation, node, &child_bounds)?
            }
            ValueOperator::Sampler { operation, .. } => sampler_bound(operation),
            ValueOperator::DeterministicHash(_) => NumericContract::Known(CoefficientBound::Large),
            ValueOperator::Source(_) | ValueOperator::Sample { .. } => NumericContract::Missing,
            ValueOperator::ProgramCall { .. } => self
                .program_call_matrix_facts(expression)
                .map(|facts| facts.coefficient_bound.clone())
                .unwrap_or(NumericContract::Missing),
            ValueOperator::Transform(_) => NumericContract::Missing,
            _ => child_bounds.into_iter().next().unwrap_or(NumericContract::Missing),
        };
        Ok(bound)
    }

    fn matrix_operation_bound(
        &self,
        operation: &MatrixOperation,
        node: &ExprNode,
        bounds: &[NumericContract<CoefficientBound>],
    ) -> Result<NumericContract<CoefficientBound>, NormalizeError> {
        match operation {
            MatrixOperation::Add | MatrixOperation::Subtract => add_bounds(bounds),
            MatrixOperation::Negate |
            MatrixOperation::Transpose |
            MatrixOperation::Slice { .. } |
            MatrixOperation::IndexedSlice { .. } |
            MatrixOperation::View { .. } => {
                Ok(bounds.first().cloned().unwrap_or(NumericContract::Missing))
            }
            MatrixOperation::Scale => product_bounds(bounds),
            MatrixOperation::Multiply => self.matrix_product_bound(node, bounds),
            MatrixOperation::Tensor { .. } => self.tensor_bound(node, bounds),
            MatrixOperation::Concat { .. } => max_bounds(bounds),
            MatrixOperation::CrtRecompose { reconstruction_coefficients, .. } => {
                weighted_sum_bounds(bounds, reconstruction_coefficients)
            }
            MatrixOperation::ExtractCoefficient { .. } |
            MatrixOperation::LiftConstantPolynomial { .. } => {
                Ok(bounds.first().cloned().unwrap_or(NumericContract::Missing))
            }
        }
    }

    fn tensor_bound(
        &self,
        node: &ExprNode,
        bounds: &[NumericContract<CoefficientBound>],
    ) -> Result<NumericContract<CoefficientBound>, NormalizeError> {
        let [left_bound, right_bound] = bounds else {
            return Ok(NumericContract::Missing);
        };
        let (NumericContract::Known(left_bound), NumericContract::Known(right_bound)) =
            (left_bound, right_bound)
        else {
            return Ok(NumericContract::Missing);
        };
        let (ResolvedValueType::Matrix(left_type), ResolvedValueType::Matrix(right_type)) = (
            self.expressions.value_type(node.inputs[0])?,
            self.expressions.value_type(node.inputs[1])?,
        ) else {
            return Ok(NumericContract::Missing);
        };
        let canonical = tensor_bound_with_facts(
            &CanonicalMatrixBound {
                matrix_type: concrete_type(left_type),
                coefficient_class: canonical_class(left_bound),
            },
            &CanonicalMatrixBound {
                matrix_type: concrete_type(right_type),
                coefficient_class: canonical_class(right_bound),
            },
            &MatrixProductFacts {
                left_is_constant_polynomial: self.constant_polynomial_fact(node.inputs[0]),
                right_is_constant_polynomial: self.constant_polynomial_fact(node.inputs[1]),
                ..MatrixProductFacts::default()
            },
        )
        .map_err(|_| NormalizeError::ArithmeticOverflow)?;
        Ok(NumericContract::Known(coefficient_bound(&canonical.coefficient_class)))
    }

    fn constant_polynomial_fact(&self, expression: ExprId) -> bool {
        self.matrix_value_facts(expression)
            .is_some_and(|facts| facts.metadata.is_constant_polynomial)
    }

    fn matrix_value_facts(&self, expression: ExprId) -> Option<&MatrixFacts> {
        match self.facts.facts(expression) {
            Ok(ValueFacts::Matrix(facts)) => Some(facts),
            _ => self.program_call_matrix_facts(expression),
        }
    }

    fn matrix_product_bound(
        &self,
        node: &ExprNode,
        bounds: &[NumericContract<CoefficientBound>],
    ) -> Result<NumericContract<CoefficientBound>, NormalizeError> {
        let [NumericContract::Known(left_bound), NumericContract::Known(right_bound)] = bounds
        else {
            return Ok(NumericContract::Missing);
        };
        let (ResolvedValueType::Matrix(left_type), ResolvedValueType::Matrix(right_type)) = (
            self.expressions.value_type(node.inputs[0])?,
            self.expressions.value_type(node.inputs[1])?,
        ) else {
            return Ok(NumericContract::Missing);
        };
        let left_facts = self.matrix_value_facts(node.inputs[0]);
        let right_facts = self.matrix_value_facts(node.inputs[1]);
        let support = |facts: Option<&MatrixFacts>| {
            facts.and_then(|facts| match &facts.polynomial {
                NumericContract::Known(polynomial) => Some(polynomial.support_upper),
                NumericContract::Missing => None,
            })
        };
        let result = product_bound_with_facts(
            &CanonicalMatrixBound {
                matrix_type: concrete_type(left_type),
                coefficient_class: canonical_class(left_bound),
            },
            &CanonicalMatrixBound {
                matrix_type: concrete_type(right_type),
                coefficient_class: canonical_class(right_bound),
            },
            &MatrixProductFacts {
                left_is_constant_polynomial: left_facts
                    .is_some_and(|facts| facts.metadata.is_constant_polynomial),
                right_is_constant_polynomial: right_facts
                    .is_some_and(|facts| facts.metadata.is_constant_polynomial),
                left_support_upper: support(left_facts),
                right_support_upper: support(right_facts),
                ..MatrixProductFacts::default()
            },
        )
        .map_err(|_| NormalizeError::ArithmeticOverflow)?;
        Ok(NumericContract::Known(coefficient_bound(&result.coefficient_class)))
    }

    fn nonmatrix_bound(
        &self,
        expression: ExprId,
        node: &ExprNode,
        children: &[Arc<AnalyzedValue>],
    ) -> Result<NumericContract<CoefficientBound>, NormalizeError> {
        if let Some(bound) = self.fact_bound(expression)? {
            return Ok(bound);
        }
        let bounds =
            children.iter().map(|value| value.coefficient_bound.clone()).collect::<Vec<_>>();
        let bound = match &node.operator {
            ValueOperator::Constant(constant) => match &constant.value {
                super::arena::ConstantValue::Int(value) => {
                    NumericContract::Known(CoefficientBound::finite(value.magnitude().clone()))
                }
                _ => NumericContract::Missing,
            },
            ValueOperator::Scalar(operation) => scalar_bound(operation, &bounds),
            ValueOperator::ExtractCoefficient { .. } => {
                bounds.first().cloned().unwrap_or(NumericContract::Missing)
            }
            ValueOperator::Sampler { operation, .. } => sampler_bound(operation),
            ValueOperator::DeterministicHash(_) => NumericContract::Known(CoefficientBound::Large),
            ValueOperator::Source(_) | ValueOperator::Sample { .. } => NumericContract::Missing,
            ValueOperator::Argument { .. } | ValueOperator::ProgramCall { .. } => {
                NumericContract::Missing
            }
            _ => bounds.first().cloned().unwrap_or(NumericContract::Missing),
        };
        Ok(bound)
    }

    fn fact_bound(
        &self,
        expression: ExprId,
    ) -> Result<Option<NumericContract<CoefficientBound>>, NormalizeError> {
        match self.facts.coefficient_bound(expression) {
            Ok(bound) => Ok(Some(bound.clone())),
            Err(FactError::MissingFacts { .. }) => Ok(None),
            Err(error) => Err(NormalizeError::Facts(error)),
        }
    }

    fn integer_constant(&self, expression: ExprId) -> Option<BigInt> {
        let mut current = expression;
        let mut negate = false;
        loop {
            let node = self.expressions.node(current).ok()?;
            match &node.operator {
                ValueOperator::Constant(super::arena::TypedConstant {
                    value: super::arena::ConstantValue::Int(value),
                    ..
                }) => return Some(if negate { -value.clone() } else { value.clone() }),
                ValueOperator::Scalar(ScalarOperation::Negate) if node.inputs.len() == 1 => {
                    negate = !negate;
                    current = node.inputs[0];
                }
                _ => return None,
            }
        }
    }
}

fn canonical_lhs_monomial(
    normal_form: Option<&PolynomialNF>,
) -> Result<MonomialId, NormalizeError> {
    let Some(normal_form) = normal_form else {
        return Err(NormalizeError::Relation(RelationRegistryError::NonCanonicalLhs(
            super::relation::CanonicalLhsError::MissingExactNormalForm,
        )));
    };
    let mut terms = normal_form.exact_terms.iter();
    let Some((monomial, coefficient)) = terms.next() else {
        return Err(NormalizeError::Relation(RelationRegistryError::NonCanonicalLhs(
            super::relation::CanonicalLhsError::Zero,
        )));
    };
    if terms.next().is_some() {
        return Err(NormalizeError::Relation(RelationRegistryError::NonCanonicalLhs(
            super::relation::CanonicalLhsError::MultipleTerms,
        )));
    }
    if coefficient != &BigInt::from(1_u8) {
        return Err(NormalizeError::Relation(RelationRegistryError::NonCanonicalLhs(
            super::relation::CanonicalLhsError::NonUnitCoefficient,
        )));
    }
    Ok(*monomial)
}

fn merge_term(terms: &mut TermMap<BigInt>, monomial: MonomialId, coefficient: BigInt) {
    if coefficient.is_zero() {
        return;
    }
    let entry = terms.entry(monomial).or_insert_with(|| BigInt::from(0_u8));
    *entry += coefficient;
    if entry.is_zero() {
        terms.remove(&monomial);
    }
}

fn remove_central_subword(
    actual: &[ScopedExprId],
    required: &[ScopedExprId],
) -> Option<Vec<ScopedExprId>> {
    let mut remaining = actual.to_vec();
    for factor in required {
        let position = remaining.iter().position(|candidate| candidate == factor)?;
        remaining.remove(position);
    }
    Some(remaining)
}

fn with_summary(
    mut normal_form: PolynomialNF,
    bound: NumericContract<CoefficientBound>,
) -> PolynomialNF {
    normal_form.bounded_summary.coefficient_bound = if normal_form.exact_terms.is_empty() {
        NumericContract::Known(CoefficientBound::ExactZero)
    } else {
        bound
    };
    normal_form
}

/// Merge two sound value-level contracts without replacing a known result with a weaker one.
/// Both contracts describe the same expression, so the tighter known upper bound is safe to
/// retain; `Missing` never displaces a known result.
fn stronger_bound(
    existing: &NumericContract<CoefficientBound>,
    incoming: &NumericContract<CoefficientBound>,
) -> NumericContract<CoefficientBound> {
    match (existing, incoming) {
        (NumericContract::Missing, incoming) => incoming.clone(),
        (existing, NumericContract::Missing) => existing.clone(),
        (NumericContract::Known(existing), NumericContract::Known(incoming)) => {
            let selected = match (existing, incoming) {
                (CoefficientBound::ExactZero, _) | (_, CoefficientBound::ExactZero) => {
                    CoefficientBound::ExactZero
                }
                (CoefficientBound::Finite(existing), CoefficientBound::Finite(incoming)) => {
                    if existing.maximum_absolute_coefficient <=
                        incoming.maximum_absolute_coefficient
                    {
                        CoefficientBound::Finite(existing.clone())
                    } else {
                        CoefficientBound::Finite(incoming.clone())
                    }
                }
                (CoefficientBound::Finite(existing), CoefficientBound::Large) => {
                    CoefficientBound::Finite(existing.clone())
                }
                (CoefficientBound::Large, CoefficientBound::Finite(incoming)) => {
                    CoefficientBound::Finite(incoming.clone())
                }
                (CoefficientBound::Large, CoefficientBound::Large) => CoefficientBound::Large,
            };
            NumericContract::Known(selected)
        }
    }
}

fn concrete_type(matrix: &super::arena::ResolvedMatrixType) -> ConcreteMatrixType {
    ConcreteMatrixType {
        modulus: matrix.modulus.clone().into(),
        ring_dimension: matrix.ring_dimension,
        rows: matrix.rows,
        columns: matrix.columns,
    }
}

fn canonical_class(bound: &CoefficientBound) -> BoundClass {
    match bound {
        CoefficientBound::ExactZero => BoundClass::ExactZero,
        CoefficientBound::Finite(bound) => {
            BoundClass::bounded(bound.maximum_absolute_coefficient.clone())
        }
        CoefficientBound::Large => BoundClass::Large,
    }
}

fn coefficient_bound(bound: &BoundClass) -> CoefficientBound {
    match bound {
        BoundClass::ExactZero => CoefficientBound::ExactZero,
        BoundClass::Bounded { maximum_absolute_coefficient } => {
            CoefficientBound::finite(maximum_absolute_coefficient.clone())
        }
        BoundClass::Large => CoefficientBound::Large,
    }
}

fn sampler_bound(operation: &SamplerOperation) -> NumericContract<CoefficientBound> {
    match operation {
        SamplerOperation::UniformResidue { .. } => NumericContract::Known(CoefficientBound::Large),
        SamplerOperation::UniformInterval { minimum, maximum, .. } => {
            let upper = minimum.abs().max(maximum.abs());
            NumericContract::Known(CoefficientBound::finite(upper.magnitude().clone()))
        }
        SamplerOperation::Gaussian { max_coefficient_bound, .. } |
        SamplerOperation::Trapdoor {
            preimage_max_coefficient_bound: max_coefficient_bound,
            ..
        } |
        SamplerOperation::Preimage { max_coefficient_bound, .. } => NumericContract::Known(
            CoefficientBound::finite(max_coefficient_bound.magnitude().clone()),
        ),
        SamplerOperation::Hash { variant, base, .. } => match variant {
            // Plain hashes are intentionally explicit large residuals.  A finite value is
            // accepted only when the caller supplied an authoritative fact, which is handled
            // before this fallback by `factor_bound`.
            HashVariant::Plain => NumericContract::Known(CoefficientBound::Large),
            HashVariant::Decomposed | HashVariant::SmallDecomposed => {
                let Some(base) = base else { return NumericContract::Missing };
                if *base < 2 {
                    return NumericContract::Missing;
                }
                let bound = if matches!(variant, HashVariant::SmallDecomposed) {
                    base.saturating_sub(1)
                } else {
                    (*base / 2).max(1)
                };
                NumericContract::Known(CoefficientBound::finite(BigUint::from(bound)))
            }
        },
    }
}

fn transform_bound(operation: &ValueTransformOperation) -> NumericContract<CoefficientBound> {
    match operation {
        ValueTransformOperation::GadgetDecompose { base, small, .. } => {
            if *base < 2 {
                return NumericContract::Missing;
            }
            let bound = if *small { base.saturating_sub(1) } else { (*base / 2).max(1) };
            NumericContract::Known(CoefficientBound::finite(BigUint::from(bound)))
        }
        ValueTransformOperation::PackPolynomialCoefficients { .. } => NumericContract::Missing,
    }
}

fn scalar_bound(
    operation: &ScalarOperation,
    bounds: &[NumericContract<CoefficientBound>],
) -> NumericContract<CoefficientBound> {
    match operation {
        ScalarOperation::Add | ScalarOperation::Subtract => {
            add_bounds(bounds).unwrap_or(NumericContract::Missing)
        }
        ScalarOperation::Multiply => product_bounds(bounds).unwrap_or(NumericContract::Missing),
        ScalarOperation::Negate |
        ScalarOperation::BoolToInt |
        ScalarOperation::IntToReal |
        ScalarOperation::ExtractCoefficient { .. } => {
            bounds.first().cloned().unwrap_or(NumericContract::Missing)
        }
        ScalarOperation::LiftConstantPolynomial { .. } => {
            bounds.first().cloned().unwrap_or(NumericContract::Missing)
        }
        _ => NumericContract::Missing,
    }
}

fn add_bounds(
    bounds: &[NumericContract<CoefficientBound>],
) -> Result<NumericContract<CoefficientBound>, NormalizeError> {
    let mut result = CoefficientBound::ExactZero;
    for bound in bounds {
        let NumericContract::Known(bound) = bound else {
            return Ok(NumericContract::Missing);
        };
        result = add_known_bounds(&result, bound);
    }
    Ok(NumericContract::Known(result))
}

fn add_known_bounds(left: &CoefficientBound, right: &CoefficientBound) -> CoefficientBound {
    match (left, right) {
        (CoefficientBound::ExactZero, right) => right.clone(),
        (left, CoefficientBound::ExactZero) => left.clone(),
        (CoefficientBound::Large, _) | (_, CoefficientBound::Large) => CoefficientBound::Large,
        (CoefficientBound::Finite(left), CoefficientBound::Finite(right)) => {
            CoefficientBound::finite(
                &left.maximum_absolute_coefficient + &right.maximum_absolute_coefficient,
            )
        }
    }
}

fn product_bounds(
    bounds: &[NumericContract<CoefficientBound>],
) -> Result<NumericContract<CoefficientBound>, NormalizeError> {
    product_bounds_with_factor(bounds, &BigUint::from(1_u8))
}

fn product_bounds_with_factor(
    bounds: &[NumericContract<CoefficientBound>],
    factor: &BigUint,
) -> Result<NumericContract<CoefficientBound>, NormalizeError> {
    let mut result = CoefficientBound::Finite(BoundExpression::new(BigUint::from(1_u8)));
    for bound in bounds {
        let NumericContract::Known(bound) = bound else {
            return Ok(NumericContract::Missing);
        };
        match (&result, bound) {
            (CoefficientBound::ExactZero, _) | (_, CoefficientBound::ExactZero) => {
                return Ok(NumericContract::Known(CoefficientBound::ExactZero));
            }
            (CoefficientBound::Large, _) | (_, CoefficientBound::Large) => {
                return Ok(NumericContract::Known(CoefficientBound::Large));
            }
            (CoefficientBound::Finite(left), CoefficientBound::Finite(right)) => {
                result = CoefficientBound::Finite(BoundExpression::new(
                    &left.maximum_absolute_coefficient * &right.maximum_absolute_coefficient,
                ));
            }
        }
    }
    if let CoefficientBound::Finite(value) = &mut result {
        value.maximum_absolute_coefficient *= factor;
    }
    Ok(NumericContract::Known(result))
}

fn max_bounds(
    bounds: &[NumericContract<CoefficientBound>],
) -> Result<NumericContract<CoefficientBound>, NormalizeError> {
    let mut result = NumericContract::Known(CoefficientBound::ExactZero);
    for bound in bounds {
        let NumericContract::Known(bound) = bound else {
            return Ok(NumericContract::Missing);
        };
        result = NumericContract::Known(max_bound(result.as_known().unwrap(), bound));
    }
    Ok(result)
}

fn max_bound(left: &CoefficientBound, right: &CoefficientBound) -> CoefficientBound {
    match (left, right) {
        (CoefficientBound::Large, _) | (_, CoefficientBound::Large) => CoefficientBound::Large,
        (CoefficientBound::ExactZero, right) => right.clone(),
        (left, CoefficientBound::ExactZero) => left.clone(),
        (CoefficientBound::Finite(left), CoefficientBound::Finite(right)) => {
            CoefficientBound::finite(
                left.maximum_absolute_coefficient
                    .clone()
                    .max(right.maximum_absolute_coefficient.clone()),
            )
        }
    }
}

fn weighted_sum_bounds(
    bounds: &[NumericContract<CoefficientBound>],
    weights: &[BigInt],
) -> Result<NumericContract<CoefficientBound>, NormalizeError> {
    if bounds.len() != weights.len() {
        return Ok(NumericContract::Missing);
    }
    let mut result = BigUint::from(0_u8);
    for (bound, weight) in bounds.iter().zip(weights) {
        // A zero reconstruction coefficient removes the lane semantically. Inspecting its
        // numeric class first would incorrectly let `0 * Large` poison an otherwise bounded CRT
        // recomposition.
        if weight.is_zero() {
            continue;
        }
        let NumericContract::Known(CoefficientBound::Finite(value)) = bound else {
            return Ok(NumericContract::Missing);
        };
        result += value.maximum_absolute_coefficient.clone() * weight.magnitude();
    }
    Ok(NumericContract::Known(CoefficientBound::finite(result)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::{
        arena::{
            ArenaToken, HashVariant, MatrixLayout, MatrixOperation, ProgramInput, ProgramSignature,
            ResolvedMatrixType, SampleEventId, SamplerOperation, SemanticFamilySourceIdentity,
            SemanticSourceIdentity, TrustedIndexRange,
        },
        facts::{MatrixFacts, MatrixMetadata, ValueFacts},
        job::{ProofReachedUniversalLhs, ReachedUniversalLhs},
        relation::{
            FactorOrderContract, GadgetRecompositionRegistry, GadgetRecompositionRule,
            RelationRegistry, RelationValidationAuthority, SamplerSourceContract, StaticLhsKey,
            TrapdoorSourceContract, UniversalDispatchKey, UniversalRelationRegistration,
        },
    };
    use std::time::{Duration, Instant};

    fn matrix_type() -> ResolvedMatrixType {
        ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 2).unwrap()
    }

    #[test]
    fn universal_lhs_canonicalization_is_typed_and_fail_closed() {
        let token = ArenaToken::fresh();
        let first = MonomialId::new(token, 0);
        let second = MonomialId::new(token, 1);
        let expected =
            |reason| Err(NormalizeError::Relation(RelationRegistryError::NonCanonicalLhs(reason)));
        assert_eq!(
            canonical_lhs_monomial(None),
            expected(crate::operational_noise::relation::CanonicalLhsError::MissingExactNormalForm)
        );
        assert_eq!(
            canonical_lhs_monomial(Some(&PolynomialNF::zero())),
            expected(crate::operational_noise::relation::CanonicalLhsError::Zero)
        );
        let multi = PolynomialNF {
            exact_terms: [(first, BigInt::from(1_u8)), (second, BigInt::from(1_u8))]
                .into_iter()
                .collect(),
            bounded_summary: BoundedSummary::missing(),
        };
        assert_eq!(
            canonical_lhs_monomial(Some(&multi)),
            expected(crate::operational_noise::relation::CanonicalLhsError::MultipleTerms)
        );
        let nonunit = PolynomialNF {
            exact_terms: [(first, BigInt::from(2_u8))].into_iter().collect(),
            bounded_summary: BoundedSummary::missing(),
        };
        assert_eq!(
            canonical_lhs_monomial(Some(&nonunit)),
            expected(crate::operational_noise::relation::CanonicalLhsError::NonUnitCoefficient)
        );
        let accepted = PolynomialNF {
            exact_terms: [(first, BigInt::from(1_u8))].into_iter().collect(),
            bounded_summary: BoundedSummary::missing(),
        };
        assert_eq!(canonical_lhs_monomial(Some(&accepted)), Ok(first));
    }

    fn setup(
        expressions: &mut ExprArena,
        programs: &mut ProgramArena,
        body: ExprId,
    ) -> (FactStore, MonomialArena, ScopedExprId) {
        let output = expressions.value_type(body).unwrap().clone();
        let domain = super::super::arena::FamilyDomain::new(0, 1).unwrap();
        let family = programs
            .generated_family(
                expressions,
                ProgramSignature {
                    inputs: Box::new([ProgramInput {
                        value_type: ResolvedValueType::Int,
                        trusted_index_range: Some(TrustedIndexRange {
                            minimum: domain.minimum,
                            maximum_exclusive: domain.maximum_exclusive,
                        }),
                    }]),
                    output,
                },
                body,
            )
            .unwrap();
        let facts = FactStore::new(expressions);
        let monomials = MonomialArena::new(expressions, programs, family.program()).unwrap();
        let semantic = programs.scoped(expressions, family.program(), body).unwrap();
        (facts, monomials, semantic)
    }

    fn source(expressions: &mut ExprArena) -> ExprId {
        source_with(expressions, matrix_type(), 1)
    }

    fn source_with(expressions: &mut ExprArena, output: ResolvedMatrixType, event: u64) -> ExprId {
        expressions
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(event),
                    operation: SamplerOperation::UniformResidue { output },
                },
                Box::new([]),
            )
            .unwrap()
    }

    fn interval_factor(
        expressions: &mut ExprArena,
        output: ResolvedMatrixType,
        event: u64,
        bound: i64,
    ) -> ExprId {
        expressions
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(event),
                    operation: SamplerOperation::UniformInterval {
                        output,
                        minimum: BigInt::from(-bound),
                        maximum: BigInt::from(bound),
                    },
                },
                Box::new([]),
            )
            .unwrap()
    }

    fn gaussian_factor(
        expressions: &mut ExprArena,
        output: ResolvedMatrixType,
        event: u64,
        bound: i64,
    ) -> ExprId {
        expressions
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(event),
                    operation: SamplerOperation::Gaussian {
                        output,
                        sigma: "1".to_owned(),
                        max_coefficient_bound: BigInt::from(bound),
                    },
                },
                Box::new([]),
            )
            .unwrap()
    }

    fn preimage_factor(
        expressions: &mut ExprArena,
        output: ResolvedMatrixType,
        event: u64,
        bound: i64,
    ) -> ExprId {
        expressions
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(event),
                    operation: SamplerOperation::Preimage {
                        output,
                        max_coefficient_bound: BigInt::from(bound),
                    },
                },
                Box::new([]),
            )
            .unwrap()
    }

    fn hash_factor(expressions: &mut ExprArena, output: ResolvedMatrixType, event: u64) -> ExprId {
        expressions
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(event),
                    operation: SamplerOperation::Hash {
                        output,
                        variant: HashVariant::Plain,
                        tag_prefix: Box::new([]),
                        tag_expressions: Box::new([]),
                        tag_decimal_expressions: Box::new([]),
                        tag_u64_le_expressions: Box::new([]),
                        base: None,
                        digit_count: None,
                    },
                },
                Box::new([]),
            )
            .unwrap()
    }

    fn product(expressions: &mut ExprArena, factors: &[ExprId]) -> ExprId {
        let (&first, rest) = factors.split_first().expect("non-empty product");
        rest.iter().copied().fold(first, |left, right| {
            expressions.intern_matrix_transform(MatrixOperation::Multiply, &[left, right]).unwrap()
        })
    }

    fn closed_relation_authority(
        matrix: &ResolvedMatrixType,
        source: ExprId,
        trapdoor: ExprId,
    ) -> RelationValidationAuthority {
        let value_type = ResolvedValueType::Matrix(matrix.clone());
        RelationValidationAuthority {
            source: SamplerSourceContract { expression: source },
            trapdoor_source: TrapdoorSourceContract { expression: trapdoor },
            matrix_type: matrix.clone(),
            public_type: value_type.clone(),
            preimage_type: value_type.clone(),
            target_type: value_type,
            trapdoor_type: ResolvedValueType::Trapdoor,
            layout: None,
            factor_order: FactorOrderContract::ordered_public_preimage(),
            domain: super::super::arena::FamilyDomain::new(0, 1).unwrap(),
            index_range: TrustedIndexRange::new(0, 1).unwrap(),
            gadget: None,
            decomposition: None,
        }
    }

    fn register_test_closed_relation(
        expressions: &mut ExprArena,
        programs: &ProgramArena,
        facts: &FactStore,
        monomials: &mut MonomialArena,
        lhs_expression: ExprId,
        rhs_expression: ExprId,
        source: ExprId,
        trapdoor: ExprId,
    ) -> (RelationRegistry, NormalizationCache, MonomialId) {
        let scope = monomials.scope();
        let lhs_proof = expressions.scope_proof(scope, lhs_expression).unwrap();
        let lhs_scoped = expressions.scoped_from_proof(&lhs_proof, lhs_expression).unwrap();
        let rhs_proof = expressions.scope_proof(scope, rhs_expression).unwrap();
        let rhs_scoped = expressions.scoped_from_proof(&rhs_proof, rhs_expression).unwrap();
        let (lhs, rhs) = {
            let mut normalizer = Normalizer::new(expressions, programs, facts, monomials).unwrap();
            let lhs = normalizer.normalize(lhs_scoped).unwrap();
            let rhs = normalizer.normalize(rhs_scoped).unwrap();
            (
                *lhs.exact_nf.unwrap().exact_terms.keys().next().unwrap(),
                (*rhs.exact_nf.unwrap()).clone(),
            )
        };
        let mut cache = NormalizationCache::new();
        let rhs = cache.intern(rhs).unwrap();
        let mut relations = RelationRegistry::new();
        relations
            .register_closed(
                CanonicalLhsKey { layout: None, monomial: lhs },
                rhs,
                &closed_relation_authority(&matrix_type(), source, trapdoor),
            )
            .unwrap();
        relations.freeze();
        (relations, cache, lhs)
    }

    fn register_test_closed_relation_into(
        expressions: &mut ExprArena,
        programs: &ProgramArena,
        facts: &FactStore,
        monomials: &mut MonomialArena,
        relations: &mut RelationRegistry,
        cache: &mut NormalizationCache,
        lhs_expression: ExprId,
        rhs_expression: ExprId,
        source: ExprId,
        trapdoor: ExprId,
    ) {
        let scope = monomials.scope();
        let lhs_proof = expressions.scope_proof(scope, lhs_expression).unwrap();
        let lhs_scoped = expressions.scoped_from_proof(&lhs_proof, lhs_expression).unwrap();
        let rhs_proof = expressions.scope_proof(scope, rhs_expression).unwrap();
        let rhs_scoped = expressions.scoped_from_proof(&rhs_proof, rhs_expression).unwrap();
        let (lhs, rhs) = {
            let mut normalizer = Normalizer::new(expressions, programs, facts, monomials).unwrap();
            let lhs = normalizer.normalize(lhs_scoped).unwrap();
            let rhs = normalizer.normalize(rhs_scoped).unwrap();
            (
                *lhs.exact_nf.unwrap().exact_terms.keys().next().unwrap(),
                (*rhs.exact_nf.unwrap()).clone(),
            )
        };
        let rhs = cache.intern(rhs).unwrap();
        relations
            .register_closed(
                CanonicalLhsKey { layout: None, monomial: lhs },
                rhs,
                &closed_relation_authority(&matrix_type(), source, trapdoor),
            )
            .unwrap();
    }

    fn insert_matrix_bound(
        facts: &mut FactStore,
        expressions: &ExprArena,
        expression: ExprId,
        bound: u64,
    ) {
        let ResolvedValueType::Matrix(matrix) = expressions.value_type(expression).unwrap() else {
            panic!("bound fixture must be a matrix")
        };
        let layout = MatrixLayout::row_major(matrix.rows, matrix.columns);
        let mut metadata = MatrixMetadata::new(layout);
        // Scalar interval fixtures used as central factors are explicitly constant-polynomial;
        // non-scalar interval factors retain the ordinary matrix bound only.
        metadata.is_constant_polynomial = matrix.rows == 1 && matrix.columns == 1;
        let mut matrix_facts = MatrixFacts::new(matrix.clone(), metadata);
        matrix_facts.coefficient_bound = NumericContract::Known(CoefficientBound::finite(bound));
        facts.insert(expressions, expression, ValueFacts::Matrix(matrix_facts)).unwrap();
    }

    fn mark_scalar_sources_constant(expressions: &ExprArena, facts: &mut FactStore, root: ExprId) {
        let mut seen = BTreeSet::new();
        let mut work = vec![root];
        while let Some(expression) = work.pop() {
            if !seen.insert(expression) {
                continue;
            }
            let node = expressions.node(expression).unwrap();
            work.extend(node.inputs.iter().copied());
            let ResolvedValueType::Matrix(matrix) = expressions.value_type(expression).unwrap()
            else {
                continue;
            };
            if !expressions.free_arguments(expression).unwrap().is_empty() {
                continue;
            }
            let is_leaf =
                matches!(node.operator, ValueOperator::Source(_) | ValueOperator::Sampler { .. });
            let metadata = MatrixMetadata {
                layout: MatrixLayout::row_major(matrix.rows, matrix.columns),
                is_constant_polynomial: is_leaf && matrix.rows == 1 && matrix.columns == 1,
                ..MatrixMetadata::new(MatrixLayout::row_major(matrix.rows, matrix.columns))
            };
            facts
                .insert(
                    expressions,
                    expression,
                    ValueFacts::Matrix(MatrixFacts::new(matrix.clone(), metadata)),
                )
                .unwrap();
        }
    }

    #[test]
    fn binder_open_explicit_family_call_uses_program_owned_scalar_facts() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let scalar = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap();
        let branch = source_with(&mut expressions, scalar.clone(), 240);
        let mut facts = FactStore::new(&expressions);
        let mut metadata = MatrixMetadata::new(MatrixLayout::row_major(1, 1));
        metadata.is_constant_polynomial = true;
        let mut branch_facts = MatrixFacts::new(scalar.clone(), metadata);
        branch_facts.coefficient_bound =
            NumericContract::Known(CoefficientBound::finite(BigUint::from(7_u8)));
        facts.insert(&expressions, branch, ValueFacts::Matrix(branch_facts)).unwrap();
        let domain = super::super::arena::FamilyDomain::new(0, 1).unwrap();
        let explicit =
            programs.explicit_family(&mut expressions, &facts, domain, Box::new([branch])).unwrap();
        let index = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let call = programs
            .call_family_in_range(
                &mut expressions,
                explicit,
                index,
                TrustedIndexRange::new(0, 1).unwrap(),
            )
            .unwrap();
        let ordered = source_with(&mut expressions, scalar, 241);
        let root = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[call, ordered])
            .unwrap();
        let outer =
            programs.opaque_generated_family_from_body(&mut expressions, domain, root).unwrap();
        let mut monomials = MonomialArena::new(&expressions, &programs, outer.program()).unwrap();
        let semantic = programs.scoped(&expressions, outer.program(), root).unwrap();
        let expected_central = programs.scoped(&expressions, outer.program(), call).unwrap();
        let expected_ordered = programs.scoped(&expressions, outer.program(), ordered).unwrap();
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let value = normalizer.normalize(semantic).unwrap();
        assert_eq!(
            normalizer.factor_bound(call).unwrap(),
            NumericContract::Known(CoefficientBound::finite(BigUint::from(7_u8)))
        );
        drop(normalizer);
        let normal_form = value.exact_nf.unwrap();
        let monomial = *normal_form.exact_terms.keys().next().unwrap();
        let descriptor = monomials.descriptor(monomial).unwrap();
        assert_eq!(descriptor.central_factors.as_ref(), &[expected_central]);
        assert_eq!(descriptor.ordered_factors.as_ref(), &[expected_ordered]);
    }

    fn insert_matrix_layout_fact(
        expressions: &ExprArena,
        facts: &mut FactStore,
        expression: ExprId,
        is_constant_polynomial: bool,
    ) {
        let ResolvedValueType::Matrix(matrix) = expressions.value_type(expression).unwrap() else {
            panic!("layout fixture must be a matrix")
        };
        let layout = MatrixLayout::row_major(matrix.rows, matrix.columns);
        let metadata = MatrixMetadata {
            layout: layout.clone(),
            is_constant_polynomial,
            ..MatrixMetadata::new(layout)
        };
        facts
            .insert(
                expressions,
                expression,
                ValueFacts::Matrix(MatrixFacts::new(matrix.clone(), metadata)),
            )
            .unwrap();
    }

    fn matrix_source(
        expressions: &mut ExprArena,
        name: &str,
        output: ResolvedMatrixType,
        gadget: Option<(u64, bool)>,
    ) -> ExprId {
        expressions
            .intern(
                ValueOperator::Source(SemanticSourceIdentity {
                    stable_definition: name.to_owned(),
                    invocation: name.to_owned(),
                    sample_event: None,
                    output_role: "value".to_owned(),
                    sampler: None,
                    artifact: None,
                    value_type: ResolvedValueType::Matrix(output),
                    coordinates: Box::new([]),
                    matrix_constant: gadget.map(|(base, small)| {
                        super::super::arena::MatrixConstantKind::Gadget { base, small }
                    }),
                }),
                Box::new([]),
            )
            .unwrap()
    }

    fn recomposition_registry(
        gadget_type: ResolvedMatrixType,
        decomposition_type: ResolvedMatrixType,
        input_type: ResolvedMatrixType,
        small: bool,
        digit_count: u32,
    ) -> GadgetRecompositionRegistry {
        let mut registry = GadgetRecompositionRegistry::new();
        registry
            .register(GadgetRecompositionRule {
                base: 2,
                small,
                digit_count,
                gadget_layout: Some(MatrixLayout::row_major(gadget_type.rows, gadget_type.columns)),
                decomposition_layout: Some(MatrixLayout::row_major(
                    decomposition_type.rows,
                    decomposition_type.columns,
                )),
                input_layout: Some(MatrixLayout::row_major(input_type.rows, input_type.columns)),
                output_type: input_type.clone(),
                gadget_type,
                decomposition_type,
                input_type,
            })
            .unwrap();
        registry.freeze();
        registry
    }

    fn gadget_product(
        expressions: &mut ExprArena,
        small: bool,
        digit_count: u32,
        gadget_type: ResolvedMatrixType,
        decomposition_type: ResolvedMatrixType,
        input_type: ResolvedMatrixType,
        gadget_constant: Option<(u64, bool)>,
    ) -> (ExprId, ExprId, ExprId) {
        let gadget = matrix_source(expressions, "gadget", gadget_type, gadget_constant);
        let input = matrix_source(expressions, "input", input_type, None);
        let decomposition = expressions
            .intern(
                ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
                    output: decomposition_type,
                    base: 2,
                    small,
                    digit_count,
                }),
                Box::new([input]),
            )
            .unwrap();
        (gadget, decomposition, input)
    }

    fn normalize_with_gadget_registry(
        expressions: &mut ExprArena,
        programs: &mut ProgramArena,
        body: ExprId,
        registry: &GadgetRecompositionRegistry,
    ) -> (PolynomialNF, MonomialArena) {
        let (mut facts, mut monomials, root) = setup(expressions, programs, body);
        mark_scalar_sources_constant(expressions, &mut facts, body);
        let exact_nf = {
            let mut normalizer = Normalizer::new(expressions, programs, &facts, &mut monomials)
                .unwrap()
                .with_gadget_recompositions(registry);
            normalizer.normalize(root).unwrap().exact_nf.unwrap().as_ref().clone()
        };
        (exact_nf, monomials)
    }

    #[test]
    fn gadget_recomposition_rewrites_regular_and_small_typed_constants() {
        for small in [false, true] {
            let mut expressions = ExprArena::new();
            let mut programs = ProgramArena::new();
            let input_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 1).unwrap();
            let decomposition_type =
                ResolvedMatrixType::new(BigUint::from(17_u8), 1, 3, 1).unwrap();
            let gadget_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 3).unwrap();
            let (gadget, decomposition, _) = gadget_product(
                &mut expressions,
                small,
                3,
                gadget_type.clone(),
                decomposition_type.clone(),
                input_type.clone(),
                Some((2, small)),
            );
            let product = expressions
                .intern_matrix_transform(MatrixOperation::Multiply, &[gadget, decomposition])
                .unwrap();
            let registry =
                recomposition_registry(gadget_type, decomposition_type, input_type, small, 3);
            let (normal_form, monomials) =
                normalize_with_gadget_registry(&mut expressions, &mut programs, product, &registry);
            assert_eq!(normal_form.exact_terms.len(), 1);
            let term = normal_form.exact_terms.keys().next().unwrap();
            let descriptor = monomials.descriptor(*term).unwrap();
            // The recomposed input is the normalized 1x1 NF itself; it must remain a central
            // factor rather than being reintroduced as an opaque ordered expression atom.
            assert_eq!(descriptor.central_factors.len(), 1);
            assert_eq!(descriptor.ordered_factors.len(), 0);
        }
    }

    #[test]
    fn tensor_scalar_action_routes_new_gadget_adjacency_through_recomposition() {
        for scalar_on_left in [false, true] {
            let mut expressions = ExprArena::new();
            let mut programs = ProgramArena::new();
            let input_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 1).unwrap();
            let decomposition_type =
                ResolvedMatrixType::new(BigUint::from(17_u8), 1, 3, 1).unwrap();
            let gadget_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 3).unwrap();
            let scalar_type = input_type.clone();
            let (gadget, decomposition, input) = gadget_product(
                &mut expressions,
                false,
                3,
                gadget_type.clone(),
                decomposition_type.clone(),
                input_type.clone(),
                Some((2, false)),
            );
            let scalar_body = matrix_source(
                &mut expressions,
                if scalar_on_left { "tensor-left-scalar" } else { "tensor-right-scalar" },
                scalar_type,
                None,
            );
            let domain = super::super::arena::FamilyDomain::new(0, 1).unwrap();
            let scalar_family = programs
                .opaque_generated_family_from_body(&mut expressions, domain, scalar_body)
                .unwrap();
            let index = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
            let scalar = programs
                .call_family_in_range(
                    &mut expressions,
                    scalar_family,
                    index,
                    TrustedIndexRange::new(0, 1).unwrap(),
                )
                .unwrap();
            let root = if scalar_on_left {
                let tensor = expressions
                    .intern_matrix_transform(
                        MatrixOperation::Tensor {
                            output: gadget_type.clone(),
                            left_layout: MatrixLayout::row_major(1, 1),
                            right_layout: MatrixLayout::row_major(1, 3),
                            output_layout: MatrixLayout::row_major(1, 3),
                        },
                        &[scalar, gadget],
                    )
                    .unwrap();
                expressions
                    .intern_matrix_transform(MatrixOperation::Multiply, &[tensor, decomposition])
                    .unwrap()
            } else {
                let tensor = expressions
                    .intern_matrix_transform(
                        MatrixOperation::Tensor {
                            output: decomposition_type.clone(),
                            left_layout: MatrixLayout::row_major(3, 1),
                            right_layout: MatrixLayout::row_major(1, 1),
                            output_layout: MatrixLayout::row_major(3, 1),
                        },
                        &[decomposition, scalar],
                    )
                    .unwrap();
                expressions
                    .intern_matrix_transform(MatrixOperation::Multiply, &[gadget, tensor])
                    .unwrap()
            };
            let registry =
                recomposition_registry(gadget_type, decomposition_type, input_type, false, 3);
            let (mut facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
            for expression in [gadget, decomposition, input] {
                insert_matrix_layout_fact(&expressions, &mut facts, expression, false);
            }
            let scalar_scoped = programs.scoped(&expressions, monomials.scope(), scalar).unwrap();
            let scalar_value = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                .unwrap()
                .normalize(scalar_scoped)
                .unwrap();
            assert_eq!(scalar_value.coefficient_bound, NumericContract::Missing);
            let scalar_nf = scalar_value.exact_nf.unwrap();
            let scalar_descriptor =
                monomials.descriptor(*scalar_nf.exact_terms.keys().next().unwrap()).unwrap();
            assert!(scalar_descriptor.central_factors.is_empty());
            assert_eq!(scalar_descriptor.ordered_factors.len(), 1);
            assert_eq!(scalar_descriptor.ordered_factors[0].expression(), scalar);
            let value = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                .unwrap()
                .with_gadget_recompositions(&registry)
                .normalize(semantic)
                .unwrap();
            let normal_form = value.exact_nf.unwrap();
            assert_eq!(normal_form.exact_terms.len(), 1);
            let descriptor =
                monomials.descriptor(*normal_form.exact_terms.keys().next().unwrap()).unwrap();
            assert_eq!(descriptor.central_factors.len(), 1);
            assert_eq!(descriptor.central_factors[0].expression(), scalar);
            assert_eq!(
                descriptor
                    .ordered_factors
                    .iter()
                    .map(|factor| factor.expression())
                    .collect::<Vec<_>>(),
                vec![input]
            );
        }
    }

    #[test]
    fn gadget_recomposition_is_binder_open_after_family_body_lowering() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let input_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 1).unwrap();
        let decomposition_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 3, 1).unwrap();
        let gadget_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 3).unwrap();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let input = expressions
            .intern(
                ValueOperator::OpaqueFamilyElement {
                    source: SemanticFamilySourceIdentity {
                        stable_definition: "binder-open-input".to_owned(),
                        invocation: "binder-open-input".to_owned(),
                        element_type: ResolvedValueType::Matrix(input_type.clone()),
                        domain: super::super::arena::FamilyDomain::new(0, 4).unwrap(),
                        artifact: None,
                    },
                },
                Box::new([argument]),
            )
            .unwrap();
        let gadget = matrix_source(
            &mut expressions,
            "binder-open-gadget",
            gadget_type.clone(),
            Some((2, false)),
        );
        let decomposition = expressions
            .intern(
                ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
                    output: decomposition_type.clone(),
                    base: 2,
                    small: false,
                    digit_count: 3,
                }),
                Box::new([input]),
            )
            .unwrap();
        let product = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[gadget, decomposition])
            .unwrap();
        let registry =
            recomposition_registry(gadget_type, decomposition_type, input_type, false, 3);
        let (normal_form, monomials) =
            normalize_with_gadget_registry(&mut expressions, &mut programs, product, &registry);
        let term = normal_form.exact_terms.keys().next().unwrap();
        let descriptor = monomials.descriptor(*term).unwrap();
        assert_eq!(descriptor.central_factors.len(), 0);
        assert_eq!(descriptor.ordered_factors.len(), 2);
    }

    #[test]
    fn nonconstant_one_by_one_recomposition_stays_ordered() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let scalar = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 1).unwrap();
        let (gadget, decomposition, input) = gadget_product(
            &mut expressions,
            false,
            1,
            scalar.clone(),
            scalar.clone(),
            scalar.clone(),
            Some((2, false)),
        );
        let product = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[gadget, decomposition])
            .unwrap();
        let registry = recomposition_registry(scalar.clone(), scalar.clone(), scalar, false, 1);
        let (mut facts, mut monomials, root) = setup(&mut expressions, &mut programs, product);
        // The identity remains valid for a nonconstant 1x1 A.  Supply the exact
        // layouts required by the registry while keeping every factor noncentral;
        // recomposition must expose A as one ordered factor.
        insert_matrix_layout_fact(&expressions, &mut facts, gadget, false);
        insert_matrix_layout_fact(&expressions, &mut facts, decomposition, false);
        insert_matrix_layout_fact(&expressions, &mut facts, input, false);
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_gadget_recompositions(&registry);
        let value = normalizer.normalize(root).unwrap();
        let id = *value.exact_nf.unwrap().exact_terms.keys().next().unwrap();
        let descriptor = monomials.descriptor(id).unwrap();
        assert!(descriptor.central_factors.is_empty());
        assert_eq!(descriptor.ordered_factors.len(), 1);
        assert_eq!(descriptor.ordered_factors[0].expression(), input);
    }

    #[test]
    fn gadget_recomposition_requires_order_and_typed_gadget_source() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let scalar = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 1).unwrap();
        let registry =
            recomposition_registry(scalar.clone(), scalar.clone(), scalar.clone(), false, 1);

        let (gadget, decomposition, _) = gadget_product(
            &mut expressions,
            false,
            1,
            scalar.clone(),
            scalar.clone(),
            scalar.clone(),
            Some((2, false)),
        );
        let reversed = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[decomposition, gadget])
            .unwrap();
        let (normal_form, monomials) =
            normalize_with_gadget_registry(&mut expressions, &mut programs, reversed, &registry);
        let term = normal_form.exact_terms.keys().next().unwrap();
        let reversed_descriptor = monomials.descriptor(*term).unwrap();
        // The scalar gadget source is central by fact; the scalar decomposition
        // transform is not a source fact and therefore remains ordered.
        assert_eq!(reversed_descriptor.central_factors.len(), 1);
        assert_eq!(reversed_descriptor.ordered_factors.len(), 1);

        let input_type = scalar.clone();
        let decomposition_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 3, 1).unwrap();
        let gadget_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 3).unwrap();
        let non_gadget_registry = recomposition_registry(
            gadget_type.clone(),
            decomposition_type.clone(),
            input_type.clone(),
            false,
            3,
        );
        let scalar_source =
            matrix_source(&mut expressions, "same-shaped-source", gadget_type, None);
        let input = matrix_source(&mut expressions, "same-shaped-input", input_type, None);
        let decomposition = expressions
            .intern(
                ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
                    output: decomposition_type,
                    base: 2,
                    small: false,
                    digit_count: 3,
                }),
                Box::new([input]),
            )
            .unwrap();
        let non_gadget_product = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[scalar_source, decomposition])
            .unwrap();
        let (normal_form, monomials) = normalize_with_gadget_registry(
            &mut expressions,
            &mut programs,
            non_gadget_product,
            &non_gadget_registry,
        );
        let term = normal_form.exact_terms.keys().next().unwrap();
        assert_eq!(monomials.descriptor(*term).unwrap().ordered_factors.len(), 2);
    }

    #[test]
    fn gadget_recomposition_preserves_central_scalar_and_rejects_hash_decomposition() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let scalar = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 1).unwrap();
        let input_type = scalar.clone();
        let decomposition_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 3, 1).unwrap();
        let gadget_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 3).unwrap();
        let registry = recomposition_registry(
            gadget_type.clone(),
            decomposition_type.clone(),
            input_type.clone(),
            false,
            3,
        );
        let central = matrix_source(&mut expressions, "central", scalar.clone(), None);
        let (gadget, decomposition, _) = gadget_product(
            &mut expressions,
            false,
            3,
            gadget_type,
            decomposition_type.clone(),
            input_type,
            Some((2, false)),
        );
        let central_product = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[central, gadget])
            .and_then(|left| {
                expressions
                    .intern_matrix_transform(MatrixOperation::Multiply, &[left, decomposition])
            })
            .unwrap();
        let (normal_form, monomials) = normalize_with_gadget_registry(
            &mut expressions,
            &mut programs,
            central_product,
            &registry,
        );
        let term = normal_form.exact_terms.keys().next().unwrap();
        let descriptor = monomials.descriptor(*term).unwrap();
        assert_eq!(descriptor.central_factors.len(), 2);
        assert_eq!(descriptor.ordered_factors.len(), 0);

        let hash = expressions
            .intern(
                ValueOperator::Sampler {
                    event: super::super::arena::SampleEventId(901),
                    operation: SamplerOperation::Hash {
                        output: decomposition_type,
                        variant: HashVariant::Decomposed,
                        tag_prefix: Box::new([]),
                        tag_expressions: Box::new([]),
                        tag_decimal_expressions: Box::new([]),
                        tag_u64_le_expressions: Box::new([]),
                        base: Some(2),
                        digit_count: Some(1),
                    },
                },
                Box::new([]),
            )
            .unwrap();
        let hash_product = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[gadget, hash])
            .unwrap();
        let (normal_form, monomials) = normalize_with_gadget_registry(
            &mut expressions,
            &mut programs,
            hash_product,
            &registry,
        );
        let term = normal_form.exact_terms.keys().next().unwrap();
        assert_eq!(monomials.descriptor(*term).unwrap().ordered_factors.len(), 2);
    }

    #[test]
    fn gadget_recomposition_splices_each_input_nf_term_without_raw_input_atom() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let input_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 2, 1).unwrap();
        let decomposition_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 6, 1).unwrap();
        let gadget_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 2, 6).unwrap();
        let first = matrix_source(&mut expressions, "sum-first", input_type.clone(), None);
        let second = matrix_source(&mut expressions, "sum-second", input_type.clone(), None);
        let input =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[first, second]).unwrap();
        let gadget =
            matrix_source(&mut expressions, "sum-gadget", gadget_type.clone(), Some((2, false)));
        let decomposition = expressions
            .intern(
                ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
                    output: decomposition_type.clone(),
                    base: 2,
                    small: false,
                    digit_count: 3,
                }),
                Box::new([input]),
            )
            .unwrap();
        let product = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[gadget, decomposition])
            .unwrap();
        let registry =
            recomposition_registry(gadget_type, decomposition_type, input_type, false, 3);
        let (normal_form, monomials) =
            normalize_with_gadget_registry(&mut expressions, &mut programs, product, &registry);
        assert_eq!(normal_form.exact_terms.len(), 2);
        let factors = normal_form
            .exact_terms
            .keys()
            .map(|id| {
                let descriptor = monomials.descriptor(*id).unwrap();
                assert!(descriptor.central_factors.is_empty());
                assert_eq!(descriptor.ordered_factors.len(), 1);
                descriptor.ordered_factors[0].expression()
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(factors, [first, second].into_iter().collect());
    }

    #[test]
    fn gadget_recomposition_splices_prefix_suffix_and_central_factors() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let scalar_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 1).unwrap();
        let input_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 2, 1).unwrap();
        let decomposition_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 6, 1).unwrap();
        let gadget_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 2, 6).unwrap();
        let input = matrix_source(&mut expressions, "surrounded-input", input_type.clone(), None);
        let gadget = matrix_source(
            &mut expressions,
            "surrounded-gadget",
            gadget_type.clone(),
            Some((2, false)),
        );
        let decomposition = expressions
            .intern(
                ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
                    output: decomposition_type.clone(),
                    base: 2,
                    small: false,
                    digit_count: 3,
                }),
                Box::new([input]),
            )
            .unwrap();
        let prefix =
            matrix_source(&mut expressions, "surrounded-prefix", scalar_type.clone(), None);
        let suffix =
            matrix_source(&mut expressions, "surrounded-suffix", scalar_type.clone(), None);
        let product = product(&mut expressions, &[prefix, gadget, decomposition, suffix]);
        let registry =
            recomposition_registry(gadget_type, decomposition_type, input_type, false, 3);
        let (normal_form, monomials) =
            normalize_with_gadget_registry(&mut expressions, &mut programs, product, &registry);
        assert_eq!(normal_form.exact_terms.len(), 1);
        let descriptor =
            monomials.descriptor(*normal_form.exact_terms.keys().next().unwrap()).unwrap();
        assert_eq!(descriptor.central_factors.len(), 2);
        assert_eq!(
            descriptor.ordered_factors.as_ref(),
            &[programs.scoped(&expressions, monomials.scope(), input).unwrap()]
        );
        let central = descriptor
            .central_factors
            .iter()
            .map(|factor| factor.expression())
            .collect::<BTreeSet<_>>();
        assert_eq!(central, [prefix, suffix].into_iter().collect());
    }

    #[test]
    fn tensor_flattens_typed_one_by_one_scalar_action_and_preserves_order() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let scalar_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 1).unwrap();
        let other_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 2, 2).unwrap();
        let scalar = matrix_source(&mut expressions, "tensor-scalar", scalar_type.clone(), None);
        let other = matrix_source(&mut expressions, "tensor-other", other_type.clone(), None);
        let tensor = expressions
            .intern_matrix_transform(
                MatrixOperation::Tensor {
                    output: other_type.clone(),
                    left_layout: MatrixLayout::row_major(1, 1),
                    right_layout: MatrixLayout::row_major(2, 2),
                    output_layout: MatrixLayout::row_major(2, 2),
                },
                &[scalar, other],
            )
            .unwrap();
        let (mut facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, tensor);
        let mut metadata = MatrixMetadata::new(MatrixLayout::row_major(1, 1));
        metadata.is_constant_polynomial = true;
        facts
            .insert(
                &expressions,
                scalar,
                ValueFacts::Matrix(MatrixFacts::new(scalar_type, metadata)),
            )
            .unwrap();
        let value = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(semantic)
            .unwrap();
        let id = *value.exact_nf.unwrap().exact_terms.keys().next().unwrap();
        let descriptor = monomials.descriptor(id).unwrap();
        assert_eq!(descriptor.central_factors.len(), 1);
        assert_eq!(descriptor.ordered_factors.len(), 1);
        assert_eq!(descriptor.central_factors[0].expression(), scalar);
        assert_eq!(descriptor.ordered_factors[0].expression(), other);

        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let scalar_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 1).unwrap();
        let other_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 2, 2).unwrap();
        let scalar = matrix_source(&mut expressions, "tensor-ring-scalar", scalar_type, None);
        let other = matrix_source(&mut expressions, "tensor-ring-other", other_type.clone(), None);
        let tensor = expressions
            .intern_matrix_transform(
                MatrixOperation::Tensor {
                    output: other_type,
                    left_layout: MatrixLayout::row_major(1, 1),
                    right_layout: MatrixLayout::row_major(2, 2),
                    output_layout: MatrixLayout::row_major(2, 2),
                },
                &[scalar, other],
            )
            .unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, tensor);
        let value = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(semantic)
            .unwrap();
        let id = *value.exact_nf.unwrap().exact_terms.keys().next().unwrap();
        let descriptor = monomials.descriptor(id).unwrap();
        assert_eq!(descriptor.central_factors.len(), 1);
        assert_eq!(descriptor.central_factors[0].expression(), scalar);
        assert_eq!(descriptor.ordered_factors.len(), 1);
        assert_eq!(descriptor.ordered_factors[0].expression(), other);
    }

    #[test]
    fn tensor_reclassifies_additive_scalar_terms_with_multiplicity_canonically() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let scalar_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap();
        let matrix_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 2).unwrap();
        let repeated =
            matrix_source(&mut expressions, "tensor-repeated-scalar", scalar_type.clone(), None);
        let additive = matrix_source(&mut expressions, "tensor-additive-scalar", scalar_type, None);
        let repeated_product = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[repeated, repeated])
            .unwrap();
        let scalar = expressions
            .intern_matrix_transform(MatrixOperation::Add, &[repeated_product, additive])
            .unwrap();
        let matrix = matrix_source(&mut expressions, "tensor-matrix", matrix_type.clone(), None);
        let tensor_operation = |scalar_on_left| MatrixOperation::Tensor {
            output: matrix_type.clone(),
            left_layout: if scalar_on_left {
                MatrixLayout::row_major(1, 1)
            } else {
                MatrixLayout::row_major(2, 2)
            },
            right_layout: if scalar_on_left {
                MatrixLayout::row_major(2, 2)
            } else {
                MatrixLayout::row_major(1, 1)
            },
            output_layout: MatrixLayout::row_major(2, 2),
        };
        let left =
            expressions.intern_matrix_transform(tensor_operation(true), &[scalar, matrix]).unwrap();
        let right = expressions
            .intern_matrix_transform(tensor_operation(false), &[matrix, scalar])
            .unwrap();
        let negated_right =
            expressions.intern_matrix_transform(MatrixOperation::Negate, &[right]).unwrap();
        let cancellation = expressions
            .intern_matrix_transform(MatrixOperation::Add, &[left, negated_right])
            .unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, cancellation);
        let scope = monomials.scope();
        let left = programs.scoped(&expressions, scope, left).unwrap();
        let left_nf = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(left)
            .unwrap()
            .exact_nf
            .unwrap();
        assert_eq!(left_nf.exact_terms.len(), 2);
        let mut central_multiplicities = left_nf
            .exact_terms
            .keys()
            .map(|monomial| {
                let descriptor = monomials.descriptor(*monomial).unwrap();
                assert_eq!(descriptor.ordered_factors.len(), 1);
                assert_eq!(descriptor.ordered_factors[0].expression(), matrix);
                descriptor
                    .central_factors
                    .iter()
                    .map(|factor| factor.expression())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        central_multiplicities.sort();
        let mut expected = vec![vec![additive], vec![repeated, repeated]];
        expected.sort();
        assert_eq!(central_multiplicities, expected);

        let cancelled = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(semantic)
            .unwrap();
        assert!(cancelled.exact_nf.unwrap().is_zero());
    }

    #[test]
    fn tensor_scalar_reclassification_rejects_non_scalar_composite_factors() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let row = matrix_source(
            &mut expressions,
            "tensor-row",
            ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 2).unwrap(),
            None,
        );
        let column = matrix_source(
            &mut expressions,
            "tensor-column",
            ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 1).unwrap(),
            None,
        );
        let scalar =
            expressions.intern_matrix_transform(MatrixOperation::Multiply, &[row, column]).unwrap();
        let matrix_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 2).unwrap();
        let matrix =
            matrix_source(&mut expressions, "tensor-rejection-matrix", matrix_type.clone(), None);
        let tensor = expressions
            .intern_matrix_transform(
                MatrixOperation::Tensor {
                    output: matrix_type,
                    left_layout: MatrixLayout::row_major(1, 1),
                    right_layout: MatrixLayout::row_major(2, 2),
                    output_layout: MatrixLayout::row_major(2, 2),
                },
                &[scalar, matrix],
            )
            .unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, tensor);
        let exact = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(semantic)
            .unwrap()
            .exact_nf
            .unwrap();
        assert_eq!(exact.exact_terms.len(), 1);
        let descriptor = monomials.descriptor(*exact.exact_terms.keys().next().unwrap()).unwrap();
        assert!(descriptor.central_factors.is_empty());
        assert_eq!(descriptor.ordered_factors.len(), 1);
        assert!(matches!(
            expressions.node(descriptor.ordered_factors[0].expression()).unwrap().operator,
            ValueOperator::Matrix(MatrixOperation::Tensor { .. })
        ));
    }

    #[test]
    fn addition_cancels_exact_terms() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let atom = source(&mut expressions);
        let neg = expressions
            .intern_slice(ValueOperator::Matrix(MatrixOperation::Negate), &[atom])
            .unwrap();
        let root = expressions
            .intern_slice(ValueOperator::Matrix(MatrixOperation::Add), &[atom, neg])
            .unwrap();
        let (facts, mut monomials, root_semantic) = setup(&mut expressions, &mut programs, root);
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let value = normalizer.normalize(root_semantic).unwrap();
        assert!(value.exact_nf.unwrap().is_zero());
    }

    #[test]
    fn double_transpose_reuses_the_grandchild_nf_for_sum_and_product_cancellation() {
        for product in [false, true] {
            let mut expressions = ExprArena::new();
            let mut programs = ProgramArena::new();
            let left = source_with(&mut expressions, matrix_type(), 101);
            let right = source_with(&mut expressions, matrix_type(), 102);
            let value = if product {
                expressions
                    .intern_matrix_transform(MatrixOperation::Multiply, &[left, right])
                    .unwrap()
            } else {
                expressions.intern_matrix_transform(MatrixOperation::Add, &[left, right]).unwrap()
            };
            let neg =
                expressions.intern_matrix_transform(MatrixOperation::Negate, &[value]).unwrap();
            let cancelled =
                expressions.intern_matrix_transform(MatrixOperation::Add, &[value, neg]).unwrap();
            let transposed = expressions
                .intern_matrix_transform(MatrixOperation::Transpose, &[cancelled])
                .unwrap();
            let root = expressions
                .intern_matrix_transform(MatrixOperation::Transpose, &[transposed])
                .unwrap();
            let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
            let value = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                .unwrap()
                .normalize(semantic)
                .unwrap();
            assert!(value.exact_nf.unwrap().is_zero());
        }
    }

    #[test]
    fn long_identity_slice_view_chain_shares_nf_and_keeps_cache_peak_constant() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let mut root = source_with(&mut expressions, matrix_type(), 103);
        for iteration in 0..2_000 {
            root = if iteration % 2 == 0 {
                expressions
                    .intern_matrix_transform(
                        MatrixOperation::Slice {
                            row_start: 0,
                            row_end_exclusive: 2,
                            column_start: 0,
                            column_end_exclusive: 2,
                            layout: MatrixLayout::row_major(2, 2),
                        },
                        &[root],
                    )
                    .unwrap()
            } else {
                expressions
                    .intern_matrix_transform(
                        MatrixOperation::View {
                            output: matrix_type(),
                            layout: MatrixLayout::row_major(2, 2),
                        },
                        &[root],
                    )
                    .unwrap()
            };
        }
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let value = normalizer.normalize(semantic).unwrap();
        assert_eq!(value.exact_nf.unwrap().term_count(), 1);
        assert!(
            normalizer.counters().peak_cached_values <= 2,
            "identity chain retained {} cached values",
            normalizer.counters().peak_cached_values
        );
        assert!(
            normalizer.counters().remaining_use_releases >= 1_999,
            "identity chain did not release intermediate values: {}",
            normalizer.counters().remaining_use_releases
        );
    }

    #[test]
    fn closed_relation_rewrites_ordered_subword_with_prefix_suffix_and_central_factor() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let matrix = matrix_type();
        let scalar = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap();
        let central = interval_factor(&mut expressions, scalar, 601, 2);
        let prefix = interval_factor(&mut expressions, matrix.clone(), 602, 1);
        let public = hash_factor(&mut expressions, matrix.clone(), 603);
        let preimage = preimage_factor(&mut expressions, matrix.clone(), 604, 1);
        let target = gaussian_factor(&mut expressions, matrix.clone(), 605, 1);
        let suffix = interval_factor(&mut expressions, matrix.clone(), 606, 1);
        let lhs_expression = product(&mut expressions, &[public, preimage]);
        let root = product(&mut expressions, &[central, prefix, public, preimage, suffix]);
        let (mut facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        insert_matrix_bound(&mut facts, &expressions, central, 2);
        insert_matrix_bound(&mut facts, &expressions, prefix, 1);
        insert_matrix_bound(&mut facts, &expressions, target, 1);
        insert_matrix_bound(&mut facts, &expressions, suffix, 1);
        let scope = monomials.scope();
        let scoped = |expressions: &ExprArena, id| {
            let proof = expressions.scope_proof(scope, id).unwrap();
            expressions.scoped_from_proof(&proof, id).unwrap()
        };
        let lhs_scoped = scoped(&expressions, lhs_expression);
        let target_scoped = scoped(&expressions, target);
        let (lhs, rhs_nf) = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
            let lhs_nf = normalizer.normalize(lhs_scoped).unwrap().exact_nf.unwrap();
            let rhs_nf = normalizer.normalize(target_scoped).unwrap().exact_nf.unwrap();
            (*lhs_nf.exact_terms.keys().next().unwrap(), rhs_nf.as_ref().clone())
        };
        let mut cache = NormalizationCache::new();
        let rhs = cache.intern(rhs_nf).unwrap();
        let mut relations = RelationRegistry::new();
        relations
            .register_closed(
                CanonicalLhsKey { layout: None, monomial: lhs },
                rhs,
                &closed_relation_authority(&matrix, preimage, public),
            )
            .unwrap();
        relations.freeze();
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_relations(&relations, &mut cache);
        let value = normalizer.normalize(semantic).unwrap();
        let exact_nf = value.exact_nf.unwrap();
        assert!(
            exact_nf.exact_terms.is_empty(),
            "finite rewritten term must be folded: nf={exact_nf:?}, bound={:?}",
            value.coefficient_bound
        );
        assert_eq!(
            value.coefficient_bound,
            NumericContract::Known(CoefficientBound::finite(128_u8))
        );
        assert_eq!(exact_nf.bounded_summary.coefficient_bound, value.coefficient_bound);
        let counters = normalizer.counters();
        assert_eq!(counters.relation_candidates, 2);
        assert_eq!(counters.relation_applied, 1);
        assert_eq!(counters.relation_remaining, 0);
        assert_eq!(counters.bounded_fold_count, 1);
        assert_eq!(counters.final_exact_term_count, 0);
    }

    #[test]
    fn relation_rhs_scalar_action_rebounds_with_fact_aware_ring_factor() {
        for scalar_on_left in [false, true] {
            for scalar_is_constant in [false, true] {
                let mut expressions = ExprArena::new();
                let mut programs = ProgramArena::new();
                let matrix_type = matrix_type();
                let scalar_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap();
                let public = hash_factor(&mut expressions, matrix_type.clone(), 650);
                let preimage = preimage_factor(&mut expressions, matrix_type.clone(), 651, 1);
                let lhs = product(&mut expressions, &[public, preimage]);
                let scalar = source_with(&mut expressions, scalar_type.clone(), 652);
                let matrix = source_with(&mut expressions, matrix_type.clone(), 653);
                let inputs = if scalar_on_left { [scalar, matrix] } else { [matrix, scalar] };
                let rhs = expressions
                    .intern_matrix_transform(
                        MatrixOperation::Tensor {
                            output: matrix_type.clone(),
                            left_layout: if scalar_on_left {
                                MatrixLayout::row_major(1, 1)
                            } else {
                                MatrixLayout::row_major(2, 2)
                            },
                            right_layout: if scalar_on_left {
                                MatrixLayout::row_major(2, 2)
                            } else {
                                MatrixLayout::row_major(1, 1)
                            },
                            output_layout: MatrixLayout::row_major(2, 2),
                        },
                        &inputs,
                    )
                    .unwrap();
                let (mut facts, mut monomials, semantic) =
                    setup(&mut expressions, &mut programs, lhs);
                for (expression, ty, bound, constant) in [
                    (scalar, scalar_type, 2_u8, scalar_is_constant),
                    (matrix, matrix_type, 3_u8, false),
                ] {
                    let mut metadata =
                        MatrixMetadata::new(MatrixLayout::row_major(ty.rows, ty.columns));
                    metadata.is_constant_polynomial = constant;
                    let mut matrix_facts = MatrixFacts::new(ty, metadata);
                    matrix_facts.coefficient_bound =
                        NumericContract::Known(CoefficientBound::finite(BigUint::from(bound)));
                    facts
                        .insert(&expressions, expression, ValueFacts::Matrix(matrix_facts))
                        .unwrap();
                }
                let (relations, mut cache, _) = register_test_closed_relation(
                    &mut expressions,
                    &programs,
                    &facts,
                    &mut monomials,
                    lhs,
                    rhs,
                    preimage,
                    public,
                );
                let mut normalizer =
                    Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                        .unwrap()
                        .with_relations(&relations, &mut cache);
                let value = normalizer.normalize(semantic).unwrap();
                let expected = if scalar_is_constant { 6_u8 } else { 24_u8 };
                assert_eq!(
                    value.coefficient_bound,
                    NumericContract::Known(CoefficientBound::finite(BigUint::from(expected))),
                    "scalar_on_left={scalar_on_left}, constant={scalar_is_constant}",
                );
                assert!(value.exact_nf.unwrap().exact_terms.is_empty());
                assert_eq!(normalizer.counters().relation_applied, 1);
                assert_eq!(normalizer.counters().bounded_fold_count, 1);
            }
        }
    }

    #[test]
    fn closed_relations_rewrite_two_ordered_occurrences_to_a_fixed_point() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let matrix = matrix_type();
        let public_one = hash_factor(&mut expressions, matrix.clone(), 611);
        let preimage_one = preimage_factor(&mut expressions, matrix.clone(), 612, 1);
        let target_one = gaussian_factor(&mut expressions, matrix.clone(), 613, 1);
        let public_two = hash_factor(&mut expressions, matrix.clone(), 614);
        let preimage_two = preimage_factor(&mut expressions, matrix.clone(), 615, 1);
        let target_two = gaussian_factor(&mut expressions, matrix.clone(), 616, 1);
        let lhs_one_expression = product(&mut expressions, &[public_one, preimage_one]);
        let lhs_two_expression = product(&mut expressions, &[public_two, preimage_two]);
        let root = product(&mut expressions, &[public_one, preimage_one, public_two, preimage_two]);
        let (mut facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        insert_matrix_bound(&mut facts, &expressions, target_one, 1);
        insert_matrix_bound(&mut facts, &expressions, target_two, 1);
        let scope = monomials.scope();
        let scoped = |expressions: &ExprArena, id| {
            let proof = expressions.scope_proof(scope, id).unwrap();
            expressions.scoped_from_proof(&proof, id).unwrap()
        };
        let lhs_one_scoped = scoped(&expressions, lhs_one_expression);
        let lhs_two_scoped = scoped(&expressions, lhs_two_expression);
        let target_one_scoped = scoped(&expressions, target_one);
        let target_two_scoped = scoped(&expressions, target_two);
        let (lhs_one, rhs_one, lhs_two, rhs_two) = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
            let lhs_one_nf = normalizer.normalize(lhs_one_scoped).unwrap().exact_nf.unwrap();
            let rhs_one_nf = normalizer.normalize(target_one_scoped).unwrap().exact_nf.unwrap();
            let lhs_two_nf = normalizer.normalize(lhs_two_scoped).unwrap().exact_nf.unwrap();
            let rhs_two_nf = normalizer.normalize(target_two_scoped).unwrap().exact_nf.unwrap();
            (
                *lhs_one_nf.exact_terms.keys().next().unwrap(),
                rhs_one_nf.as_ref().clone(),
                *lhs_two_nf.exact_terms.keys().next().unwrap(),
                rhs_two_nf.as_ref().clone(),
            )
        };
        let mut cache = NormalizationCache::new();
        let rhs_one = cache.intern(rhs_one).unwrap();
        let rhs_two = cache.intern(rhs_two).unwrap();
        let mut relations = RelationRegistry::new();
        relations
            .register_closed(
                CanonicalLhsKey { layout: None, monomial: lhs_one },
                rhs_one,
                &closed_relation_authority(&matrix, preimage_one, public_one),
            )
            .unwrap();
        relations
            .register_closed(
                CanonicalLhsKey { layout: None, monomial: lhs_two },
                rhs_two,
                &closed_relation_authority(&matrix, preimage_two, public_two),
            )
            .unwrap();
        relations.freeze();
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_relations(&relations, &mut cache);
        normalizer.fold_final_no_match = false;
        let value = normalizer.normalize(semantic).unwrap();
        let exact_nf = value.exact_nf.unwrap();
        let counters = normalizer.counters();
        drop(normalizer);
        assert_eq!(exact_nf.exact_terms.len(), 1);
        let descriptor =
            monomials.descriptor(*exact_nf.exact_terms.keys().next().unwrap()).unwrap();
        assert_eq!(descriptor.ordered_factors.as_ref(), &[target_one_scoped, target_two_scoped]);
        assert_eq!(value.coefficient_bound, NumericContract::Known(CoefficientBound::finite(8_u8)));
        assert_eq!(counters.relation_candidates, 3);
        assert_eq!(counters.relation_applied, 2);
        assert_eq!(counters.relation_remaining, 0);
        assert_eq!(counters.bounded_fold_count, 0);
        assert_eq!(counters.final_exact_term_count, 1);
    }

    #[test]
    fn relation_rhs_error_exposes_and_rewrites_a_second_preimage_relation() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let matrix = matrix_type();
        let public_one = hash_factor(&mut expressions, matrix.clone(), 621);
        let preimage_one = preimage_factor(&mut expressions, matrix.clone(), 622, 1);
        let public_two = hash_factor(&mut expressions, matrix.clone(), 623);
        let preimage_two = preimage_factor(&mut expressions, matrix.clone(), 624, 1);
        let signal_one = interval_factor(&mut expressions, matrix.clone(), 625, 1);
        let signal_two = interval_factor(&mut expressions, matrix.clone(), 626, 1);
        let target = gaussian_factor(&mut expressions, matrix.clone(), 627, 1);
        let error_one = gaussian_factor(&mut expressions, matrix.clone(), 628, 1);
        let error_two = gaussian_factor(&mut expressions, matrix.clone(), 629, 1);
        let lhs_one_expression = product(&mut expressions, &[public_one, preimage_one]);
        let lhs_two_expression = product(&mut expressions, &[public_two, preimage_two]);
        let rhs_one_signal = product(&mut expressions, &[signal_one, public_two]);
        let rhs_two_signal = product(&mut expressions, &[signal_two, target]);
        let rhs_one_expression = expressions
            .intern_matrix_transform(MatrixOperation::Add, &[rhs_one_signal, error_one])
            .unwrap();
        let rhs_two_expression = expressions
            .intern_matrix_transform(MatrixOperation::Add, &[rhs_two_signal, error_two])
            .unwrap();
        let root = product(&mut expressions, &[public_one, preimage_one, preimage_two]);
        let (mut facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        for (expression, bound) in [
            (preimage_two, 1),
            (signal_one, 1),
            (signal_two, 1),
            (target, 1),
            (error_one, 1),
            (error_two, 1),
        ] {
            insert_matrix_bound(&mut facts, &expressions, expression, bound);
        }
        let scope = monomials.scope();
        let scoped = |expressions: &ExprArena, id| {
            let proof = expressions.scope_proof(scope, id).unwrap();
            expressions.scoped_from_proof(&proof, id).unwrap()
        };
        let lhs_one_scoped = scoped(&expressions, lhs_one_expression);
        let lhs_two_scoped = scoped(&expressions, lhs_two_expression);
        let rhs_one_scoped = scoped(&expressions, rhs_one_expression);
        let rhs_two_scoped = scoped(&expressions, rhs_two_expression);
        let (lhs_one, rhs_one, lhs_two, rhs_two) = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
            let lhs_one_nf = normalizer.normalize(lhs_one_scoped).unwrap().exact_nf.unwrap();
            let rhs_one_nf = normalizer.normalize(rhs_one_scoped).unwrap().exact_nf.unwrap();
            let lhs_two_nf = normalizer.normalize(lhs_two_scoped).unwrap().exact_nf.unwrap();
            let rhs_two_nf = normalizer.normalize(rhs_two_scoped).unwrap().exact_nf.unwrap();
            (
                *lhs_one_nf.exact_terms.keys().next().unwrap(),
                rhs_one_nf.as_ref().clone(),
                *lhs_two_nf.exact_terms.keys().next().unwrap(),
                rhs_two_nf.as_ref().clone(),
            )
        };
        let mut cache = NormalizationCache::new();
        let rhs_one = cache.intern(rhs_one).unwrap();
        let rhs_two = cache.intern(rhs_two).unwrap();
        let mut relations = RelationRegistry::new();
        relations
            .register_closed(
                CanonicalLhsKey { layout: None, monomial: lhs_one },
                rhs_one,
                &closed_relation_authority(&matrix, preimage_one, public_one),
            )
            .unwrap();
        relations
            .register_closed(
                CanonicalLhsKey { layout: None, monomial: lhs_two },
                rhs_two,
                &closed_relation_authority(&matrix, preimage_two, public_two),
            )
            .unwrap();
        relations.freeze();
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_relations(&relations, &mut cache);
        let value = normalizer.normalize(semantic).unwrap();
        let exact_nf = value.exact_nf.unwrap();
        assert!(exact_nf.exact_terms.is_empty());
        // Ring dimension 4 contributes 8 to each 2x2 multiplication.  The surviving terms are
        // S1*S2*P (64), S1*E2 (8), and E1*K2 (8); the final term proves that the first relation's
        // error is retained and bounded even though K2 has no relation with E1.
        assert_eq!(
            value.coefficient_bound,
            NumericContract::Known(CoefficientBound::finite(80_u8))
        );
        assert_eq!(exact_nf.bounded_summary.coefficient_bound, value.coefficient_bound);
    }

    #[test]
    fn relation_rewrite_precedes_bound_and_turns_missing_s_b_k_into_finite_s_p() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let scalar_type = matrix_type();
        let hash = |expressions: &mut ExprArena, output: ResolvedMatrixType, event| {
            expressions
                .intern(
                    ValueOperator::Sampler {
                        event: SampleEventId(event),
                        operation: SamplerOperation::Hash {
                            output,
                            variant: HashVariant::Plain,
                            tag_prefix: Box::new([]),
                            tag_expressions: Box::new([]),
                            tag_decimal_expressions: Box::new([]),
                            tag_u64_le_expressions: Box::new([]),
                            base: None,
                            digit_count: None,
                        },
                    },
                    Box::new([]),
                )
                .unwrap()
        };
        let s = expressions
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(201),
                    operation: SamplerOperation::UniformInterval {
                        output: scalar_type.clone(),
                        minimum: BigInt::from(-3),
                        maximum: BigInt::from(3),
                    },
                },
                Box::new([]),
            )
            .unwrap();
        let b = hash(&mut expressions, matrix_type(), 202);
        let k = hash(&mut expressions, matrix_type(), 203);
        let p = expressions
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(204),
                    operation: SamplerOperation::Gaussian {
                        output: matrix_type(),
                        sigma: "1".into(),
                        max_coefficient_bound: BigInt::from(2),
                    },
                },
                Box::new([]),
            )
            .unwrap();
        let bk = expressions.intern_matrix_transform(MatrixOperation::Multiply, &[b, k]).unwrap();
        let root =
            expressions.intern_matrix_transform(MatrixOperation::Multiply, &[s, bk]).unwrap();
        let (mut facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        for (id, ty, bound) in [(s, scalar_type, 3_u8), (p, matrix_type(), 2_u8)] {
            let layout = MatrixLayout::row_major(ty.rows, ty.columns);
            let mut matrix_facts = MatrixFacts::new(ty, MatrixMetadata::new(layout));
            matrix_facts.coefficient_bound =
                NumericContract::Known(CoefficientBound::finite(bound));
            facts.insert(&expressions, id, ValueFacts::Matrix(matrix_facts)).unwrap();
        }
        let scope = monomials.scope();
        let bk_proof = expressions.scope_proof(scope, bk).unwrap();
        let bk_scoped = expressions.scoped_from_proof(&bk_proof, bk).unwrap();
        let p_proof = expressions.scope_proof(scope, p).unwrap();
        let p_scoped = expressions.scoped_from_proof(&p_proof, p).unwrap();
        let (lhs, rhs_nf) = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
            let lhs_nf = normalizer.normalize(bk_scoped).unwrap().exact_nf.unwrap();
            let rhs_nf = normalizer.normalize(p_scoped).unwrap().exact_nf.unwrap();
            (*lhs_nf.exact_terms.keys().next().unwrap(), (*rhs_nf).clone())
        };
        let mut cache = NormalizationCache::new();
        let rhs = cache.intern(rhs_nf).unwrap();
        let ty = ResolvedValueType::Matrix(matrix_type());
        let authority = RelationValidationAuthority {
            source: SamplerSourceContract { expression: k },
            trapdoor_source: TrapdoorSourceContract { expression: b },
            matrix_type: matrix_type(),
            public_type: ty.clone(),
            preimage_type: ty.clone(),
            target_type: ty.clone(),
            trapdoor_type: ResolvedValueType::Trapdoor,
            layout: None,
            factor_order: FactorOrderContract::ordered_public_preimage(),
            domain: super::super::arena::FamilyDomain::new(0, 1).unwrap(),
            index_range: TrustedIndexRange::new(0, 1).unwrap(),
            gadget: None,
            decomposition: None,
        };
        let mut relations = RelationRegistry::new();
        relations
            .register_closed(CanonicalLhsKey { layout: None, monomial: lhs }, rhs, &authority)
            .unwrap();
        relations.freeze();
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_relations(&relations, &mut cache);
        let value = normalizer.normalize(semantic).unwrap();
        assert_eq!(
            value.coefficient_bound,
            NumericContract::Known(CoefficientBound::finite(48_u8))
        );
        assert_eq!(value.exact_nf.unwrap().term_count(), 0);
    }

    #[test]
    fn closed_relations_rewrite_two_bk_occurrences_to_p_times_p() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let b = source_with(&mut expressions, matrix_type(), 226);
        let k = source_with(&mut expressions, matrix_type(), 227);
        let p = source_with(&mut expressions, matrix_type(), 228);
        let bk = product(&mut expressions, &[b, k]);
        let root = product(&mut expressions, &[b, k, b, k]);
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let (relations, mut cache, _) = register_test_closed_relation(
            &mut expressions,
            &programs,
            &facts,
            &mut monomials,
            bk,
            p,
            k,
            b,
        );
        let scope = monomials.scope();
        let p_scoped =
            expressions.scoped_from_proof(&expressions.scope_proof(scope, p).unwrap(), p).unwrap();
        let expected = {
            let mut no_relations =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
            let nf = no_relations.normalize(p_scoped).unwrap().exact_nf.unwrap();
            *nf.exact_terms.keys().next().unwrap()
        };
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_relations(&relations, &mut cache);
        normalizer.fold_final_no_match = false;
        let value = normalizer.normalize(semantic).unwrap();
        let exact = value.exact_nf.unwrap();
        let expected = monomials.combine_interned(monomials.scope(), expected, expected).unwrap();
        assert_eq!(exact.exact_terms, [(expected, BigInt::from(1_u8))].into_iter().collect());
    }

    #[test]
    fn closed_relations_allow_sibling_rhs_branches_to_reuse_relation() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let a = source_with(&mut expressions, matrix_type(), 229);
        let b = source_with(&mut expressions, matrix_type(), 229);
        let k = source_with(&mut expressions, matrix_type(), 230);
        let p = source_with(&mut expressions, matrix_type(), 231);
        let bk = product(&mut expressions, &[b, k]);
        let outer_rhs =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[bk, bk]).unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, a);
        let mut relations = RelationRegistry::new();
        let mut cache = NormalizationCache::new();
        register_test_closed_relation_into(
            &mut expressions,
            &programs,
            &facts,
            &mut monomials,
            &mut relations,
            &mut cache,
            bk,
            p,
            k,
            b,
        );
        register_test_closed_relation_into(
            &mut expressions,
            &programs,
            &facts,
            &mut monomials,
            &mut relations,
            &mut cache,
            a,
            outer_rhs,
            k,
            b,
        );
        relations.freeze();
        let scope = monomials.scope();
        let p_scoped =
            expressions.scoped_from_proof(&expressions.scope_proof(scope, p).unwrap(), p).unwrap();
        let expected = {
            let mut no_relations =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
            *no_relations
                .normalize(p_scoped)
                .unwrap()
                .exact_nf
                .unwrap()
                .exact_terms
                .keys()
                .next()
                .unwrap()
        };
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_relations(&relations, &mut cache);
        normalizer.fold_final_no_match = false;
        let exact = normalizer.normalize(semantic).unwrap().exact_nf.unwrap();
        assert_eq!(exact.exact_terms, [(expected, BigInt::from(2_u8))].into_iter().collect());
    }

    #[test]
    fn universal_specialization_accepts_k_of_h_i_and_rejects_out_of_domain_proof() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let domain = super::super::arena::FamilyDomain::new(0, 4).unwrap();
        let range = TrustedIndexRange::new(0, 4).unwrap();
        let index = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let generated = |expressions: &mut ExprArena, programs: &mut ProgramArena, name: &str| {
            let body = expressions
                .intern(
                    ValueOperator::OpaqueFamilyElement {
                        source: SemanticFamilySourceIdentity {
                            stable_definition: name.into(),
                            invocation: "fixture".into(),
                            element_type: ResolvedValueType::Matrix(matrix_type()),
                            domain,
                            artifact: None,
                        },
                    },
                    Box::new([index]),
                )
                .unwrap();
            programs.generated_family_from_body(expressions, domain, body).unwrap()
        };
        // Keep the relation fixture production-shaped: B is a generated family whose selected
        // value is a full-row slice of a tensor over a nested horizontal concat. The preimage
        // family below is intentionally opaque, so specialization must retain `ProgramCall(K,x)`.
        let public = {
            let scalar = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap();
            let mut scalar_atom = |name: &str| {
                expressions
                    .intern(
                        ValueOperator::OpaqueFamilyElement {
                            source: SemanticFamilySourceIdentity {
                                stable_definition: name.to_owned(),
                                invocation: "nested-fixture".to_owned(),
                                element_type: ResolvedValueType::Matrix(scalar.clone()),
                                domain,
                                artifact: None,
                            },
                        },
                        Box::new([index]),
                    )
                    .unwrap()
            };
            let first = scalar_atom("B0");
            let second = scalar_atom("B1");
            let third = scalar_atom("B2");
            let inner = expressions
                .intern_slice(
                    ValueOperator::Matrix(MatrixOperation::Concat {
                        axis: 1,
                        output: ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 2).unwrap(),
                        layout: MatrixLayout::row_major(1, 2),
                    }),
                    &[first, second],
                )
                .unwrap();
            let outer = expressions
                .intern_slice(
                    ValueOperator::Matrix(MatrixOperation::Concat {
                        axis: 1,
                        output: ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 3).unwrap(),
                        layout: MatrixLayout::row_major(1, 3),
                    }),
                    &[inner, third],
                )
                .unwrap();
            let right = expressions
                .intern(
                    ValueOperator::Sampler {
                        event: SampleEventId(63),
                        operation: SamplerOperation::UniformResidue { output: matrix_type() },
                    },
                    Box::new([]),
                )
                .unwrap();
            let tensor = expressions
                .intern_matrix_transform(
                    MatrixOperation::Tensor {
                        output: ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 6).unwrap(),
                        left_layout: MatrixLayout::row_major(1, 3),
                        right_layout: MatrixLayout::row_major(2, 2),
                        output_layout: MatrixLayout::row_major(2, 6),
                    },
                    &[outer, right],
                )
                .unwrap();
            let body = expressions
                .intern_slice(
                    ValueOperator::Matrix(MatrixOperation::Slice {
                        row_start: 0,
                        row_end_exclusive: 2,
                        column_start: 2,
                        column_end_exclusive: 4,
                        layout: MatrixLayout::row_major(2, 2),
                    }),
                    &[tensor],
                )
                .unwrap();
            programs.generated_family_from_body(&mut expressions, domain, body).unwrap()
        };
        let reducible_preimage = generated(&mut expressions, &mut programs, "K");
        let preimage_body = programs.family_body(reducible_preimage).unwrap();
        let preimage = programs
            .opaque_generated_family_from_body(&mut expressions, domain, preimage_body)
            .unwrap();
        let target = generated(&mut expressions, &mut programs, "P");
        let trapdoor_root = expressions
            .intern(
                ValueOperator::Trapdoor(super::super::arena::TrapdoorOperation::Generate {
                    descriptor: "fixture-trapdoor".into(),
                    parameters: Box::new([]),
                    paired_public_event: SampleEventId(51),
                    paired_public_output_role: "value".to_owned(),
                }),
                Box::new([]),
            )
            .unwrap();
        let trapdoor_family =
            programs.generated_family_from_body(&mut expressions, domain, trapdoor_root).unwrap();
        let mut facts = FactStore::new(&expressions);
        assert!(expressions.free_arguments(index).unwrap().contains(&(0, ResolvedValueType::Int)));
        facts.finalize_ranges();
        let zero = expressions
            .intern(
                ValueOperator::Constant(super::super::arena::TypedConstant::int(0)),
                Box::new([]),
            )
            .unwrap();
        let selector = expressions
            .intern(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                Box::new([index, zero]),
            )
            .unwrap();
        let b = programs.call_family_in_range(&mut expressions, public, selector, range).unwrap();
        let k = programs.call_family_in_range(&mut expressions, preimage, selector, range).unwrap();
        let ordinary_residual = source_with(&mut expressions, matrix_type(), 905);
        let product =
            expressions.intern_matrix_transform(MatrixOperation::Multiply, &[b, k]).unwrap();
        let scope_family =
            programs.generated_family_from_body(&mut expressions, domain, product).unwrap();
        let scope = scope_family.program();
        let root = programs.scoped(&expressions, scope, product).unwrap();
        let index = programs.scoped(&expressions, scope, selector).unwrap();
        let mut monomials = MonomialArena::new(&expressions, &programs, scope).unwrap();
        let lhs = {
            let value = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                .unwrap()
                .normalize(root)
                .unwrap();
            let normal_form = value.exact_nf.unwrap();
            assert_eq!(normal_form.exact_terms.len(), 1);
            assert_eq!(normal_form.exact_terms.values().next(), Some(&BigInt::from(1_u8)));
            *normal_form.exact_terms.keys().next().unwrap()
        };
        let lhs_descriptor = monomials.descriptor(lhs).unwrap();
        assert!(lhs_descriptor.ordered_factors.iter().any(|factor| {
            matches!(
                expressions.node(factor.expression()).unwrap().operator,
                ValueOperator::ProgramCall { program } if program == preimage.program()
            )
        }));
        let source = SamplerSourceContract { expression: programs.family_body(preimage).unwrap() };
        let trapdoor = TrapdoorSourceContract { expression: trapdoor_root };
        let dispatch = UniversalDispatchKey {
            preimage_family: preimage,
            preimage_source: source.clone(),
            matrix_type: matrix_type(),
            trapdoor_source: trapdoor.clone(),
        };
        let ty = ResolvedValueType::Matrix(matrix_type());
        let validation = RelationValidationAuthority {
            source,
            trapdoor_source: trapdoor,
            matrix_type: matrix_type(),
            public_type: ty.clone(),
            preimage_type: ty.clone(),
            target_type: ty.clone(),
            trapdoor_type: ResolvedValueType::Trapdoor,
            layout: None,
            factor_order: FactorOrderContract::ordered_public_preimage(),
            domain,
            index_range: range,
            gadget: None,
            decomposition: None,
        };
        let registration = UniversalRelationRegistration {
            dispatch: dispatch.clone(),
            lhs: StaticLhsKey {
                domain,
                public_plan: public.program(),
                preimage_plan: preimage.program(),
                trapdoor_plan: trapdoor_family.program(),
                public_pairing: public.program(),
                layout: None,
                factor_order: FactorOrderContract::ordered_public_preimage(),
                remaining_contracts: Box::new([]),
                validation,
            },
            target_plan: target.program(),
        };
        let mut relations = RelationRegistry::new();
        relations.register_universal(registration).unwrap();
        let generation = relations.freeze();
        let mut cache = NormalizationCache::new();
        // Construct the unmatched root before borrowing the expression arena through the
        // normalizer; it is used below for the final relation-remaining regression.
        let unmatched = expressions
            .intern_matrix_transform(MatrixOperation::Add, &[k, ordinary_residual])
            .unwrap();
        let unmatched_proof = expressions.scope_proof(scope, unmatched).unwrap();
        let unmatched = expressions.scoped_from_proof(&unmatched_proof, unmatched).unwrap();
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_relations(&relations, &mut cache);
        let reached = ReachedUniversalLhs::fixture(dispatch.clone(), index, range, None, lhs);
        assert_eq!(normalizer.normalization.as_deref().unwrap().runtime_entry_count(), 0);
        let canonical_count = normalizer.normalization.as_deref().unwrap().canonical_rhs_count();
        let canonical_fingerprint =
            normalizer.normalization.as_deref().unwrap().canonical_state_fingerprint();
        let proof = ProofReachedUniversalLhs::fixture(
            ReachedUniversalLhs::fixture(dispatch.clone(), index, range, None, lhs),
            generation,
        );
        let owned = normalizer.resolve_universal_proof(proof).unwrap();
        let ProofResolutionOwned::Rewrite(owned_rhs) = owned else {
            panic!("expected owned rewrite")
        };
        assert!(owned_rhs.term_count() > 0);
        assert_eq!(normalizer.normalization.as_deref().unwrap().runtime_entry_count(), 0);
        assert_eq!(
            normalizer.normalization.as_deref().unwrap().canonical_rhs_count(),
            canonical_count
        );
        assert_eq!(
            normalizer.normalization.as_deref().unwrap().canonical_state_fingerprint(),
            canonical_fingerprint
        );
        let repeat = ProofReachedUniversalLhs::fixture(
            ReachedUniversalLhs::fixture(dispatch.clone(), index, range, None, lhs),
            generation,
        );
        assert!(matches!(
            normalizer.resolve_universal_proof(repeat).unwrap(),
            ProofResolutionOwned::Rewrite(_)
        ));
        assert!(owned_rhs.term_count() > 0, "owned result remains usable after repeat rollback");
        let invalid_proof = ProofReachedUniversalLhs::fixture(
            ReachedUniversalLhs::fixture(
                dispatch.clone(),
                index,
                TrustedIndexRange::new(0, 5).unwrap(),
                None,
                lhs,
            ),
            generation,
        );
        assert_eq!(
            normalizer.resolve_universal_proof(invalid_proof),
            Err(NormalizeError::Relation(RelationRegistryError::IndexOutOfDomain))
        );
        assert_eq!(normalizer.normalization.as_deref().unwrap().runtime_entry_count(), 0);
        assert_eq!(
            normalizer.normalization.as_deref().unwrap().canonical_rhs_count(),
            canonical_count
        );
        assert_eq!(
            normalizer.normalization.as_deref().unwrap().canonical_state_fingerprint(),
            canonical_fingerprint
        );
        assert!(matches!(
            normalizer.resolve_universal(&reached).unwrap(),
            RelationResolution::Rewrite(_)
        ));
        assert_eq!(normalizer.normalization.as_deref().unwrap().runtime_entry_count(), 1);
        let out_of_domain = ReachedUniversalLhs::fixture(
            reached.dispatch().clone(),
            index,
            TrustedIndexRange::new(0, 5).unwrap(),
            None,
            lhs,
        );
        assert_eq!(
            normalizer.resolve_universal(&out_of_domain),
            Err(NormalizeError::Relation(RelationRegistryError::IndexOutOfDomain))
        );
        // The production path applies the same specialized relation while normalizing the
        // complete expression, including any surrounding ordered factors; it must not require a
        // job-level whole-monomial subtraction pass.
        let rewritten = normalizer.normalize(root).unwrap();
        assert!(rewritten.exact_nf.as_ref().is_some_and(|value| value.term_count() > 0));

        // A dispatchable K which has no adjacent matching public factor is retained, while the
        // ordinary residual in the same sum is not mislabeled as relation-bearing. This is a
        // final structural diagnostic only: no second universal specialization is performed.
        normalizer.fold_final_no_match = false;
        let unmatched_value = normalizer.normalize(unmatched).unwrap();
        assert_eq!(unmatched_value.exact_nf.as_ref().unwrap().exact_terms.len(), 2);
        assert_eq!(normalizer.counters().relation_remaining, 1);
    }

    #[test]
    fn universal_subword_match_is_globally_leftmost_longest() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let domain = super::super::arena::FamilyDomain::new(0, 1).unwrap();
        let range = TrustedIndexRange::new(0, 1).unwrap();
        let index = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let matrix = matrix_type();
        let opaque_matrix = |expressions: &mut ExprArena, name: &str| {
            expressions
                .intern(
                    ValueOperator::OpaqueFamilyElement {
                        source: SemanticFamilySourceIdentity {
                            stable_definition: name.to_owned(),
                            invocation: "leftmost-longest".to_owned(),
                            element_type: ResolvedValueType::Matrix(matrix.clone()),
                            domain,
                            artifact: None,
                        },
                    },
                    Box::new([index]),
                )
                .unwrap()
        };
        let b_body = opaque_matrix(&mut expressions, "B");
        let x_body = opaque_matrix(&mut expressions, "X");
        // Create the short plan first so a registry-order implementation sees [X,K] before
        // [B,X,K]. The selected result must still be determined by the complete term layout.
        let short_public =
            programs.generated_family_from_body(&mut expressions, domain, x_body).unwrap();
        let long_body = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[b_body, x_body])
            .unwrap();
        let long_public =
            programs.generated_family_from_body(&mut expressions, domain, long_body).unwrap();
        let preimage_body = preimage_factor(&mut expressions, matrix.clone(), 901, 3);
        let preimage = programs
            .opaque_generated_family_from_body(&mut expressions, domain, preimage_body)
            .unwrap();
        let public_b = b_body;
        let public_x = x_body;
        let k = programs.call_family_in_range(&mut expressions, preimage, index, range).unwrap();
        let root = product(&mut expressions, &[public_b, public_x, k]);
        let scope_family =
            programs.generated_family_from_body(&mut expressions, domain, root).unwrap();
        let scope = scope_family.program();
        let semantic = programs.scoped(&expressions, scope, root).unwrap();
        let mut facts = FactStore::new(&expressions);
        facts.finalize_ranges();
        let trapdoor_root = expressions
            .intern(
                ValueOperator::Trapdoor(super::super::arena::TrapdoorOperation::Generate {
                    descriptor: "leftmost-longest-trapdoor".into(),
                    parameters: Box::new([]),
                    paired_public_event: SampleEventId(902),
                    paired_public_output_role: "value".to_owned(),
                }),
                Box::new([]),
            )
            .unwrap();
        let trapdoor_family =
            programs.generated_family_from_body(&mut expressions, domain, trapdoor_root).unwrap();
        let target_short_body = source_with(&mut expressions, matrix.clone(), 903);
        let target_long_body = source_with(&mut expressions, matrix.clone(), 904);
        let target_short = programs
            .generated_family_from_body(&mut expressions, domain, target_short_body)
            .unwrap();
        let target_long = programs
            .generated_family_from_body(&mut expressions, domain, target_long_body)
            .unwrap();
        let source = SamplerSourceContract { expression: programs.family_body(preimage).unwrap() };
        let trapdoor = TrapdoorSourceContract { expression: trapdoor_root };
        let dispatch = UniversalDispatchKey {
            preimage_family: preimage,
            preimage_source: source.clone(),
            matrix_type: matrix.clone(),
            trapdoor_source: trapdoor.clone(),
        };
        let value_type = ResolvedValueType::Matrix(matrix.clone());
        let validation = || RelationValidationAuthority {
            source: source.clone(),
            trapdoor_source: trapdoor.clone(),
            matrix_type: matrix.clone(),
            public_type: value_type.clone(),
            preimage_type: value_type.clone(),
            target_type: value_type.clone(),
            trapdoor_type: ResolvedValueType::Trapdoor,
            layout: None,
            factor_order: FactorOrderContract::ordered_public_preimage(),
            domain,
            index_range: range,
            gadget: None,
            decomposition: None,
        };
        let registration = |public_plan, target_plan| UniversalRelationRegistration {
            dispatch: dispatch.clone(),
            lhs: StaticLhsKey {
                domain,
                public_plan,
                preimage_plan: preimage.program(),
                trapdoor_plan: trapdoor_family.program(),
                public_pairing: short_public.program(),
                layout: None,
                factor_order: FactorOrderContract::ordered_public_preimage(),
                remaining_contracts: Box::new([]),
                validation: validation(),
            },
            target_plan,
        };
        let mut relations = RelationRegistry::new();
        relations
            .register_universal(registration(short_public.program(), target_short.program()))
            .unwrap();
        relations
            .register_universal(registration(long_public.program(), target_long.program()))
            .unwrap();
        relations.freeze();
        let mut monomials = MonomialArena::new(&expressions, &programs, scope).unwrap();
        let mut cache = NormalizationCache::new();
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_relations(&relations, &mut cache);
        let value = normalizer.normalize(semantic).unwrap();
        let exact = value.exact_nf.unwrap();
        assert_eq!(exact.exact_terms.len(), 1);
        let selected = *exact.exact_terms.keys().next().unwrap();
        let selected_descriptor = monomials.descriptor(selected).unwrap();
        assert_eq!(selected_descriptor.ordered_factors.len(), 1);
        let selected_operator = expressions
            .node(selected_descriptor.ordered_factors[0].expression())
            .unwrap()
            .operator
            .clone();
        assert!(matches!(
            selected_operator,
            ValueOperator::Sampler { event: SampleEventId(904), .. }
        ));
    }

    #[test]
    fn ordered_products_do_not_commute() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let left = source(&mut expressions);
        let right = expressions
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(2),
                    operation: SamplerOperation::UniformResidue { output: matrix_type() },
                },
                Box::new([]),
            )
            .unwrap();
        let first = expressions
            .intern_slice(ValueOperator::Matrix(MatrixOperation::Multiply), &[left, right])
            .unwrap();
        let second = expressions
            .intern_slice(ValueOperator::Matrix(MatrixOperation::Multiply), &[right, left])
            .unwrap();
        let combined = expressions
            .intern_slice(ValueOperator::Matrix(MatrixOperation::Add), &[first, second])
            .unwrap();
        let (facts, mut monomials, _) = setup(&mut expressions, &mut programs, combined);
        let scope = monomials.scope();
        let first_semantic = programs.scoped(&expressions, scope, first).unwrap();
        let second_semantic = programs.scoped(&expressions, scope, second).unwrap();
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let first_nf = normalizer.normalize(first_semantic).unwrap().exact_nf.unwrap();
        let second_nf = normalizer.normalize(second_semantic).unwrap().exact_nf.unwrap();
        assert_ne!(first_nf.exact_terms, second_nf.exact_terms);
    }

    #[test]
    fn plain_hash_without_authoritative_range_is_explicitly_large() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let atom = expressions
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(3),
                    operation: SamplerOperation::Hash {
                        output: matrix_type(),
                        variant: HashVariant::Plain,
                        tag_prefix: Box::new([]),
                        tag_expressions: Box::new([]),
                        tag_decimal_expressions: Box::new([]),
                        tag_u64_le_expressions: Box::new([]),
                        base: None,
                        digit_count: None,
                    },
                },
                Box::new([]),
            )
            .unwrap();
        let root = expressions
            .intern_slice(ValueOperator::Matrix(MatrixOperation::Negate), &[atom])
            .unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let value = normalizer.normalize(semantic).unwrap();
        assert_eq!(value.coefficient_bound, NumericContract::Known(CoefficientBound::Large));
    }

    #[test]
    fn decomposed_hash_and_gadget_decompose_have_typed_finite_transfers() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let scalar = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 1).unwrap();
        let gaussian = gaussian_factor(&mut expressions, scalar.clone(), 901, 2);
        let hash = expressions
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(902),
                    operation: SamplerOperation::Hash {
                        output: scalar.clone(),
                        variant: HashVariant::Decomposed,
                        tag_prefix: Box::new([]),
                        tag_expressions: Box::new([]),
                        tag_decimal_expressions: Box::new([]),
                        tag_u64_le_expressions: Box::new([]),
                        base: Some(7),
                        digit_count: Some(1),
                    },
                },
                Box::new([]),
            )
            .unwrap();
        let input = matrix_source(&mut expressions, "typed-decompose-input", scalar.clone(), None);
        let decompose = expressions
            .intern(
                ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
                    output: scalar.clone(),
                    base: 7,
                    small: false,
                    digit_count: 1,
                }),
                Box::new([input]),
            )
            .unwrap();
        let root = product(&mut expressions, &[gaussian, hash, decompose]);
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let mut relations = RelationRegistry::new();
        relations.freeze();
        let mut cache = NormalizationCache::new();
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_relations(&relations, &mut cache);
        let value = normalizer.normalize(semantic).unwrap();
        assert!(value.exact_nf.as_ref().is_some_and(|nf| nf.exact_terms.is_empty()));
        // Gaussian=2, decomposed hash=floor(7/2)=3, regular gadget digits=3; all products are
        // scalar in this fixture, so no matrix support multiplier is introduced.
        assert_eq!(
            value.coefficient_bound,
            NumericContract::Known(CoefficientBound::finite(18_u8))
        );
    }

    #[test]
    fn released_derived_slice_keeps_gaussian_bound_for_final_fold() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let input = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 2, 1).unwrap();
        let gaussian = gaussian_factor(&mut expressions, input.clone(), 903, 4);
        let slice = expressions
            .intern(
                ValueOperator::Matrix(MatrixOperation::Slice {
                    row_start: 0,
                    row_end_exclusive: 1,
                    column_start: 0,
                    column_end_exclusive: 1,
                    layout: MatrixLayout::row_major(1, 1),
                }),
                Box::new([gaussian]),
            )
            .unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, slice);
        let mut relations = RelationRegistry::new();
        relations.freeze();
        let mut cache = NormalizationCache::new();
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_relations(&relations, &mut cache);
        let value = normalizer.normalize(semantic).unwrap();
        assert!(value.exact_nf.as_ref().is_some_and(|nf| nf.exact_terms.is_empty()));
        assert_eq!(value.coefficient_bound, NumericContract::Known(CoefficientBound::finite(4_u8)));
    }

    #[test]
    fn indexed_slice_is_structural_atom_and_cancels_exactly() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let input = matrix_source(&mut expressions, "indexed-slice-input", matrix_type(), None);
        let output = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap();
        let zero = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(0)), Box::new([]))
            .unwrap();
        let one = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(1)), Box::new([]))
            .unwrap();
        let slice = expressions
            .intern_matrix_transform(
                MatrixOperation::IndexedSlice {
                    output: output.clone(),
                    layout: MatrixLayout::row_major(1, 2),
                },
                &[input, zero, one, zero, one],
            )
            .unwrap();
        let negated =
            expressions.intern_matrix_transform(MatrixOperation::Negate, &[slice]).unwrap();
        let cancelled =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[slice, negated]).unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, cancelled);
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let value = normalizer.normalize(semantic).unwrap();
        assert!(value.exact_nf.unwrap().is_zero());
    }

    #[test]
    fn deep_shared_chain_uses_iterative_worklist() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let mut root = source(&mut expressions);
        for _ in 0..20_000 {
            root = expressions
                .intern_slice(ValueOperator::Matrix(MatrixOperation::Negate), &[root])
                .unwrap();
        }
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let started = Instant::now();
        let value = normalizer.normalize(semantic).unwrap();
        assert!(value.exact_nf.is_some());
        assert_eq!(normalizer.counters().nodes_processed, 20_001);
        assert_eq!(normalizer.counters().nodes_total, 20_001);
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn nonidentity_view_distributes_over_exact_terms() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let left = source_with(&mut expressions, matrix_type(), 60);
        let right = source_with(&mut expressions, matrix_type(), 61);
        let sum =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[left, right]).unwrap();
        let view = expressions
            .intern_matrix_transform(
                MatrixOperation::View {
                    output: ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 4).unwrap(),
                    layout: MatrixLayout::row_major(1, 4),
                },
                &[sum],
            )
            .unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, view);
        let value = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(semantic)
            .unwrap();
        let normal_form = value.exact_nf.unwrap();
        assert_eq!(normal_form.exact_terms.len(), 2);
        for id in normal_form.exact_terms.keys() {
            let factor = monomials.descriptor(*id).unwrap().ordered_factors[0];
            assert!(matches!(
                expressions.node(factor.expression()).unwrap().operator,
                ValueOperator::Matrix(MatrixOperation::View { .. })
            ));
        }
    }

    #[test]
    fn tensor_and_concat_distribute_with_operand_order_in_identity() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let a = source_with(&mut expressions, matrix_type(), 70);
        let b = source_with(&mut expressions, matrix_type(), 71);
        let c = source_with(&mut expressions, matrix_type(), 72);
        let d = source_with(&mut expressions, matrix_type(), 73);
        let left = expressions.intern_matrix_transform(MatrixOperation::Add, &[a, b]).unwrap();
        let right = expressions.intern_matrix_transform(MatrixOperation::Add, &[c, d]).unwrap();
        let tensor_output = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 4, 4).unwrap();
        let tensor = expressions
            .intern_matrix_transform(
                MatrixOperation::Tensor {
                    output: tensor_output,
                    left_layout: MatrixLayout::row_major(2, 2),
                    right_layout: MatrixLayout::row_major(2, 2),
                    output_layout: MatrixLayout::row_major(4, 4),
                },
                &[left, right],
            )
            .unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, tensor);
        let value = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(semantic)
            .unwrap();
        let normal_form = value.exact_nf.unwrap();
        assert_eq!(normal_form.exact_terms.len(), 4);
        for id in normal_form.exact_terms.keys() {
            let factor = monomials.descriptor(*id).unwrap().ordered_factors[0];
            let node = expressions.node(factor.expression()).unwrap();
            assert!(matches!(node.operator, ValueOperator::Matrix(MatrixOperation::Tensor { .. })));
            assert_eq!(node.inputs.len(), 2);
        }

        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let a = source_with(&mut expressions, matrix_type(), 74);
        let b = source_with(&mut expressions, matrix_type(), 75);
        let left = expressions.intern_matrix_transform(MatrixOperation::Add, &[a, b]).unwrap();
        let right = source_with(&mut expressions, matrix_type(), 76);
        let concat_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 4).unwrap();
        let operation = MatrixOperation::Concat {
            axis: 1,
            output: concat_type,
            layout: MatrixLayout::row_major(2, 4),
        };
        let forward =
            expressions.intern_matrix_transform(operation.clone(), &[left, right]).unwrap();
        let reverse = expressions.intern_matrix_transform(operation, &[right, left]).unwrap();
        let combined =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[forward, reverse]).unwrap();
        let (facts, mut monomials, _) = setup(&mut expressions, &mut programs, combined);
        let scope = monomials.scope();
        let forward = programs.scoped(&expressions, scope, forward).unwrap();
        let reverse = programs.scoped(&expressions, scope, reverse).unwrap();
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let forward = normalizer.normalize(forward).unwrap().exact_nf.unwrap();
        let reverse = normalizer.normalize(reverse).unwrap().exact_nf.unwrap();
        assert_eq!(forward.exact_terms.len(), 3);
        assert_eq!(reverse.exact_terms.len(), 3);
        assert_ne!(forward.exact_terms, reverse.exact_terms);
    }

    #[test]
    fn slice_ranges_remain_distinct_and_shared_prefix_growth_is_linear() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let left = source_with(&mut expressions, matrix_type(), 80);
        let right = source_with(&mut expressions, matrix_type(), 81);
        let sum =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[left, right]).unwrap();
        let slice = |expressions: &mut ExprArena, start| {
            expressions
                .intern_matrix_transform(
                    MatrixOperation::Slice {
                        row_start: 0,
                        row_end_exclusive: 2,
                        column_start: start,
                        column_end_exclusive: start + 1,
                        layout: MatrixLayout::row_major(2, 1),
                    },
                    &[sum],
                )
                .unwrap()
        };
        let first = slice(&mut expressions, 0);
        let second = slice(&mut expressions, 1);
        let combined =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[first, second]).unwrap();
        let (facts, mut monomials, _) = setup(&mut expressions, &mut programs, combined);
        let scope = monomials.scope();
        let first = programs.scoped(&expressions, scope, first).unwrap();
        let second = programs.scoped(&expressions, scope, second).unwrap();
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let first = normalizer.normalize(first).unwrap().exact_nf.unwrap();
        let second = normalizer.normalize(second).unwrap().exact_nf.unwrap();
        assert_eq!(first.exact_terms.len(), 2);
        assert_eq!(second.exact_terms.len(), 2);
        assert_ne!(first.exact_terms, second.exact_terms);

        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let left = source_with(&mut expressions, matrix_type(), 82);
        let right = source_with(&mut expressions, matrix_type(), 83);
        let mut root =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[left, right]).unwrap();
        for depth in 0..4_096 {
            let output = if depth % 2 == 0 {
                ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 4).unwrap()
            } else {
                matrix_type()
            };
            root = expressions
                .intern_matrix_transform(
                    MatrixOperation::View {
                        layout: MatrixLayout::row_major(output.rows, output.columns),
                        output,
                    },
                    &[root],
                )
                .unwrap();
        }
        let original_nodes = expressions.node_count();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let value = normalizer.normalize(semantic).unwrap();
        assert_eq!(value.exact_nf.unwrap().exact_terms.len(), 2);
        assert_eq!(normalizer.counters().nodes_processed, 4_099);
        drop(normalizer);
        assert!(expressions.node_count() <= original_nodes + 2 * 4_096);
    }

    #[test]
    fn tensor_bound_uses_ring_factor_unless_whole_operand_fact_is_constant() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let left = source_with(&mut expressions, matrix_type(), 10);
        let right = source_with(&mut expressions, matrix_type(), 11);
        let output = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 4, 4).unwrap();
        let tensor = expressions
            .intern_slice(
                ValueOperator::Matrix(MatrixOperation::Tensor {
                    output: output.clone(),
                    left_layout: MatrixLayout::row_major(2, 2),
                    right_layout: MatrixLayout::row_major(2, 2),
                    output_layout: MatrixLayout::row_major(4, 4),
                }),
                &[left, right],
            )
            .unwrap();
        let (mut facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, tensor);
        for (id, coefficient, constant) in [(left, 2_u8, false), (right, 3_u8, false)] {
            let mut metadata = MatrixMetadata::new(MatrixLayout::row_major(2, 2));
            metadata.is_constant_polynomial = constant;
            let mut value = MatrixFacts::new(matrix_type(), metadata);
            value.coefficient_bound =
                NumericContract::Known(CoefficientBound::finite(BigUint::from(coefficient)));
            facts.insert(&expressions, id, ValueFacts::Matrix(value)).unwrap();
        }
        let value = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(semantic)
            .unwrap();
        assert_eq!(
            value.coefficient_bound,
            NumericContract::Known(CoefficientBound::finite(BigUint::from(24_u8)))
        );

        let mut metadata = MatrixMetadata::new(MatrixLayout::row_major(2, 2));
        metadata.is_constant_polynomial = true;
        let mut left_facts = MatrixFacts::new(matrix_type(), metadata);
        left_facts.coefficient_bound =
            NumericContract::Known(CoefficientBound::finite(BigUint::from(2_u8)));
        let mut constant_facts = FactStore::new(&expressions);
        constant_facts.insert(&expressions, left, ValueFacts::Matrix(left_facts)).unwrap();
        let mut right_facts =
            MatrixFacts::new(matrix_type(), MatrixMetadata::new(MatrixLayout::row_major(2, 2)));
        right_facts.coefficient_bound =
            NumericContract::Known(CoefficientBound::finite(BigUint::from(3_u8)));
        constant_facts.insert(&expressions, right, ValueFacts::Matrix(right_facts)).unwrap();
        let value = Normalizer::new(&mut expressions, &programs, &constant_facts, &mut monomials)
            .unwrap()
            .normalize(semantic)
            .unwrap();
        assert_eq!(
            value.coefficient_bound,
            NumericContract::Known(CoefficientBound::finite(BigUint::from(6_u8)))
        );
    }

    #[test]
    fn one_by_one_scalar_products_use_ring_factor_unless_constant() {
        for tensor in [false, true] {
            for scalar_on_left in [false, true] {
                for scalar_is_constant in [false, true] {
                    let mut expressions = ExprArena::new();
                    let mut programs = ProgramArena::new();
                    let scalar_type =
                        ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap();
                    let matrix_type =
                        ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 2).unwrap();
                    let scalar = source_with(&mut expressions, scalar_type.clone(), 242);
                    let matrix = source_with(&mut expressions, matrix_type.clone(), 243);
                    let inputs = if scalar_on_left { [scalar, matrix] } else { [matrix, scalar] };
                    let operation = if tensor {
                        MatrixOperation::Tensor {
                            output: matrix_type.clone(),
                            left_layout: if scalar_on_left {
                                MatrixLayout::row_major(1, 1)
                            } else {
                                MatrixLayout::row_major(2, 2)
                            },
                            right_layout: if scalar_on_left {
                                MatrixLayout::row_major(2, 2)
                            } else {
                                MatrixLayout::row_major(1, 1)
                            },
                            output_layout: MatrixLayout::row_major(2, 2),
                        }
                    } else {
                        MatrixOperation::Multiply
                    };
                    let root = expressions.intern_matrix_transform(operation, &inputs).unwrap();
                    let (mut facts, mut monomials, semantic) =
                        setup(&mut expressions, &mut programs, root);
                    for (expression, ty, bound, constant) in [
                        (scalar, scalar_type, 2_u8, scalar_is_constant),
                        (matrix, matrix_type, 3_u8, false),
                    ] {
                        let mut metadata =
                            MatrixMetadata::new(MatrixLayout::row_major(ty.rows, ty.columns));
                        metadata.is_constant_polynomial = constant;
                        let mut matrix_facts = MatrixFacts::new(ty, metadata);
                        matrix_facts.coefficient_bound =
                            NumericContract::Known(CoefficientBound::finite(BigUint::from(bound)));
                        facts
                            .insert(&expressions, expression, ValueFacts::Matrix(matrix_facts))
                            .unwrap();
                    }
                    let value =
                        Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                            .unwrap()
                            .normalize(semantic)
                            .unwrap();
                    let expected = if scalar_is_constant { 6_u8 } else { 24_u8 };
                    assert_eq!(
                        value.coefficient_bound,
                        NumericContract::Known(CoefficientBound::finite(BigUint::from(expected))),
                        "tensor={tensor}, scalar_on_left={scalar_on_left}, constant={scalar_is_constant}",
                    );
                }
            }
        }
    }

    #[test]
    fn scalar_product_bound_is_association_independent() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let scalar_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap();
        let matrix_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 2).unwrap();
        let left_scalar = source_with(&mut expressions, scalar_type.clone(), 244);
        let matrix = source_with(&mut expressions, matrix_type.clone(), 245);
        let right_scalar = source_with(&mut expressions, scalar_type.clone(), 246);
        let left_pair = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[left_scalar, matrix])
            .unwrap();
        let left_associated = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[left_pair, right_scalar])
            .unwrap();
        let right_pair = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[matrix, right_scalar])
            .unwrap();
        let right_associated = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[left_scalar, right_pair])
            .unwrap();
        let combined = expressions
            .intern_matrix_transform(MatrixOperation::Add, &[left_associated, right_associated])
            .unwrap();
        let (mut facts, mut monomials, _) = setup(&mut expressions, &mut programs, combined);
        for (expression, ty, bound) in [
            (left_scalar, scalar_type.clone(), 2_u8),
            (matrix, matrix_type, 3_u8),
            (right_scalar, scalar_type, 5_u8),
        ] {
            let mut matrix_facts = MatrixFacts::new(
                ty.clone(),
                MatrixMetadata::new(MatrixLayout::row_major(ty.rows, ty.columns)),
            );
            matrix_facts.coefficient_bound =
                NumericContract::Known(CoefficientBound::finite(BigUint::from(bound)));
            facts.insert(&expressions, expression, ValueFacts::Matrix(matrix_facts)).unwrap();
        }
        let scope = monomials.scope();
        let left = programs.scoped(&expressions, scope, left_associated).unwrap();
        let right = programs.scoped(&expressions, scope, right_associated).unwrap();
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let left = normalizer.normalize(left).unwrap();
        let right = normalizer.normalize(right).unwrap();
        assert_eq!(left.coefficient_bound, right.coefficient_bound);
        assert_eq!(
            left.coefficient_bound,
            NumericContract::Known(CoefficientBound::finite(BigUint::from(480_u16)))
        );
    }

    #[test]
    fn concat_slice_restores_only_an_exact_block() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let component = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 2).unwrap();
        let concat_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 4).unwrap();
        let left = source_with(&mut expressions, component.clone(), 20);
        let right = source_with(&mut expressions, component, 21);
        let concat = expressions
            .intern_slice(
                ValueOperator::Matrix(MatrixOperation::Concat {
                    axis: 1,
                    output: concat_type,
                    layout: MatrixLayout::row_major(2, 4),
                }),
                &[left, right],
            )
            .unwrap();
        let exact_right = expressions
            .intern_slice(
                ValueOperator::Matrix(MatrixOperation::Slice {
                    row_start: 0,
                    row_end_exclusive: 2,
                    column_start: 2,
                    column_end_exclusive: 4,
                    layout: MatrixLayout::row_major(2, 2),
                }),
                &[concat],
            )
            .unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, exact_right);
        let scope = monomials.scope();
        let expected = programs.scoped(&expressions, scope, right).unwrap();
        let value = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(semantic)
            .unwrap();
        let id = *value.exact_nf.unwrap().exact_terms.keys().next().unwrap();
        assert_eq!(monomials.descriptor(id).unwrap().ordered_factors.as_ref(), &[expected]);
    }

    #[test]
    fn parent_local_slice_recovers_full_row_multiply_block() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let left_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 2).unwrap();
        let block_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 2).unwrap();
        let concat_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 4).unwrap();
        let output_type = left_type.clone();
        let left = source_with(&mut expressions, left_type, 200);
        let first = source_with(&mut expressions, block_type.clone(), 201);
        let second = source_with(&mut expressions, block_type, 202);
        let concat = expressions
            .intern_slice(
                ValueOperator::Matrix(MatrixOperation::Concat {
                    axis: 1,
                    output: concat_type,
                    layout: MatrixLayout::row_major(2, 4),
                }),
                &[first, second],
            )
            .unwrap();
        let product = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[left, concat])
            .unwrap();
        let slice = expressions
            .intern_slice(
                ValueOperator::Matrix(MatrixOperation::Slice {
                    row_start: 0,
                    row_end_exclusive: 2,
                    column_start: 2,
                    column_end_exclusive: 4,
                    layout: MatrixLayout::row_major(2, 2),
                }),
                &[product],
            )
            .unwrap();
        assert_eq!(expressions.value_type(slice).unwrap(), &ResolvedValueType::Matrix(output_type));
        let (mut facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, slice);
        mark_scalar_sources_constant(&expressions, &mut facts, slice);
        let scope = monomials.scope();
        let expected_left = programs.scoped(&expressions, scope, left).unwrap();
        let expected_second = programs.scoped(&expressions, scope, second).unwrap();
        let value = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(semantic)
            .unwrap();
        let id = *value.exact_nf.unwrap().exact_terms.keys().next().unwrap();
        let descriptor = monomials.descriptor(id).unwrap();
        assert_eq!(descriptor.central_factors.len(), 0);
        assert_eq!(descriptor.ordered_factors.as_ref(), &[expected_left, expected_second]);
    }

    #[test]
    fn parent_local_slice_rejects_partial_multiply_block() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let block_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 2).unwrap();
        let first = source_with(&mut expressions, block_type.clone(), 210);
        let second = source_with(&mut expressions, block_type, 211);
        let concat = expressions
            .intern_slice(
                ValueOperator::Matrix(MatrixOperation::Concat {
                    axis: 1,
                    output: ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 4).unwrap(),
                    layout: MatrixLayout::row_major(2, 4),
                }),
                &[first, second],
            )
            .unwrap();
        let left = source_with(
            &mut expressions,
            ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 2).unwrap(),
            212,
        );
        let product = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[left, concat])
            .unwrap();
        let partial = expressions
            .intern_slice(
                ValueOperator::Matrix(MatrixOperation::Slice {
                    row_start: 0,
                    row_end_exclusive: 2,
                    column_start: 1,
                    column_end_exclusive: 3,
                    layout: MatrixLayout::row_major(2, 2),
                }),
                &[product],
            )
            .unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, partial);
        let value = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(semantic)
            .unwrap();
        let normal_form = value.exact_nf.unwrap();
        assert_eq!(normal_form.exact_terms.len(), 2);
        for id in normal_form.exact_terms.keys() {
            let descriptor = monomials.descriptor(*id).unwrap();
            assert_eq!(descriptor.central_factors.len(), 0);
            assert_eq!(descriptor.ordered_factors.len(), 1);
            assert!(matches!(
                expressions.node(descriptor.ordered_factors[0].expression()).unwrap().operator,
                ValueOperator::Matrix(MatrixOperation::Slice { .. })
            ));
        }
    }

    #[test]
    fn parent_local_tensor_slice_exposes_central_block_and_ordered_right_factor() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let scalar_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap();
        let right_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 2).unwrap();
        let first = source_with(&mut expressions, scalar_type.clone(), 220);
        let second = source_with(&mut expressions, scalar_type.clone(), 221);
        let concat = expressions
            .intern_slice(
                ValueOperator::Matrix(MatrixOperation::Concat {
                    axis: 1,
                    output: ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 2).unwrap(),
                    layout: MatrixLayout::row_major(1, 2),
                }),
                &[first, second],
            )
            .unwrap();
        let right = source_with(&mut expressions, right_type.clone(), 222);
        let tensor = expressions
            .intern_matrix_transform(
                MatrixOperation::Tensor {
                    output: ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 4).unwrap(),
                    left_layout: MatrixLayout::row_major(1, 2),
                    right_layout: MatrixLayout::row_major(1, 2),
                    output_layout: MatrixLayout::row_major(1, 4),
                },
                &[concat, right],
            )
            .unwrap();
        let slice = expressions
            .intern_slice(
                ValueOperator::Matrix(MatrixOperation::Slice {
                    row_start: 0,
                    row_end_exclusive: 1,
                    column_start: 2,
                    column_end_exclusive: 4,
                    layout: MatrixLayout::row_major(1, 2),
                }),
                &[tensor],
            )
            .unwrap();
        let (mut facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, slice);
        mark_scalar_sources_constant(&expressions, &mut facts, slice);
        let scope = monomials.scope();
        let expected_component = programs
            .scoped(&expressions, scope, expressions.node(concat).unwrap().inputs[1])
            .unwrap();
        let expected_right = programs.scoped(&expressions, scope, right).unwrap();
        let value = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(semantic)
            .unwrap();
        let id = *value.exact_nf.unwrap().exact_terms.keys().next().unwrap();
        let descriptor = monomials.descriptor(id).unwrap();
        assert_eq!(descriptor.central_factors.as_ref(), &[expected_component]);
        assert_eq!(descriptor.ordered_factors.as_ref(), &[expected_right]);
    }

    #[test]
    fn parent_local_tensor_slice_rejects_misaligned_columns() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let scalar = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap();
        let right_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 2).unwrap();
        let first = source_with(&mut expressions, scalar.clone(), 223);
        let second = source_with(&mut expressions, scalar, 224);
        let concat = expressions
            .intern_slice(
                ValueOperator::Matrix(MatrixOperation::Concat {
                    axis: 1,
                    output: ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 2).unwrap(),
                    layout: MatrixLayout::row_major(1, 2),
                }),
                &[first, second],
            )
            .unwrap();
        let right = source_with(&mut expressions, right_type.clone(), 225);
        let tensor = expressions
            .intern_matrix_transform(
                MatrixOperation::Tensor {
                    output: ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 4).unwrap(),
                    left_layout: MatrixLayout::row_major(1, 2),
                    right_layout: MatrixLayout::row_major(1, 2),
                    output_layout: MatrixLayout::row_major(1, 4),
                },
                &[concat, right],
            )
            .unwrap();
        let slice = expressions
            .intern_slice(
                ValueOperator::Matrix(MatrixOperation::Slice {
                    row_start: 0,
                    row_end_exclusive: 1,
                    column_start: 1,
                    column_end_exclusive: 3,
                    layout: MatrixLayout::row_major(1, 2),
                }),
                &[tensor],
            )
            .unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, slice);
        let value = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(semantic)
            .unwrap();
        assert_eq!(value.exact_nf.unwrap().exact_terms.len(), 2);
    }

    #[test]
    fn nested_concat_parent_local_positive_regression() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let scalar = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap();
        let a = source_with(&mut expressions, scalar.clone(), 230);
        let b = source_with(&mut expressions, scalar.clone(), 231);
        let c = source_with(&mut expressions, scalar.clone(), 232);
        let inner = expressions
            .intern_slice(
                ValueOperator::Matrix(MatrixOperation::Concat {
                    axis: 1,
                    output: ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 2).unwrap(),
                    layout: MatrixLayout::row_major(1, 2),
                }),
                &[a, b],
            )
            .unwrap();
        let outer = expressions
            .intern_slice(
                ValueOperator::Matrix(MatrixOperation::Concat {
                    axis: 1,
                    output: ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 3).unwrap(),
                    layout: MatrixLayout::row_major(1, 3),
                }),
                &[inner, c],
            )
            .unwrap();
        let right = source_with(
            &mut expressions,
            ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 2).unwrap(),
            233,
        );
        let tensor = expressions
            .intern_matrix_transform(
                MatrixOperation::Tensor {
                    output: ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 6).unwrap(),
                    left_layout: MatrixLayout::row_major(1, 3),
                    right_layout: MatrixLayout::row_major(1, 2),
                    output_layout: MatrixLayout::row_major(1, 6),
                },
                &[outer, right],
            )
            .unwrap();
        let slice = expressions
            .intern_slice(
                ValueOperator::Matrix(MatrixOperation::Slice {
                    row_start: 0,
                    row_end_exclusive: 1,
                    column_start: 2,
                    column_end_exclusive: 4,
                    layout: MatrixLayout::row_major(1, 2),
                }),
                &[tensor],
            )
            .unwrap();
        let (mut facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, slice);
        mark_scalar_sources_constant(&expressions, &mut facts, slice);
        let scope = monomials.scope();
        let expected_central = programs.scoped(&expressions, scope, b).unwrap();
        let expected_ordered = programs.scoped(&expressions, scope, right).unwrap();
        let value = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(semantic)
            .unwrap();
        let normal_form = value.exact_nf.unwrap();
        assert_eq!(normal_form.exact_terms.len(), 1);
        assert_eq!(normal_form.exact_terms.values().next(), Some(&BigInt::from(1_u8)));
        let id = *normal_form.exact_terms.keys().next().unwrap();
        let d = monomials.descriptor(id).unwrap();
        assert_eq!(d.central_factors.as_ref(), &[expected_central]);
        assert_eq!(d.ordered_factors.as_ref(), &[expected_ordered]);
    }

    #[test]
    fn nested_concat_sibling_boundary_falls_back() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let scalar = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap();
        let a = source_with(&mut expressions, scalar.clone(), 234);
        let b = source_with(&mut expressions, scalar.clone(), 235);
        let inner = expressions
            .intern_slice(
                ValueOperator::Matrix(MatrixOperation::Concat {
                    axis: 1,
                    output: ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 2).unwrap(),
                    layout: MatrixLayout::row_major(1, 2),
                }),
                &[a, b],
            )
            .unwrap();
        let c = source_with(&mut expressions, scalar.clone(), 236);
        let outer = expressions
            .intern_slice(
                ValueOperator::Matrix(MatrixOperation::Concat {
                    axis: 1,
                    output: ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 3).unwrap(),
                    layout: MatrixLayout::row_major(1, 3),
                }),
                &[inner, c],
            )
            .unwrap();
        let right = source_with(
            &mut expressions,
            ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 2).unwrap(),
            237,
        );
        let tensor = expressions
            .intern_matrix_transform(
                MatrixOperation::Tensor {
                    output: ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 6).unwrap(),
                    left_layout: MatrixLayout::row_major(1, 3),
                    right_layout: MatrixLayout::row_major(1, 2),
                    output_layout: MatrixLayout::row_major(1, 6),
                },
                &[outer, right],
            )
            .unwrap();
        let slice = expressions
            .intern_slice(
                ValueOperator::Matrix(MatrixOperation::Slice {
                    row_start: 0,
                    row_end_exclusive: 1,
                    column_start: 1,
                    column_end_exclusive: 5,
                    layout: MatrixLayout::row_major(1, 4),
                }),
                &[tensor],
            )
            .unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, slice);
        let value = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(semantic)
            .unwrap();
        let normal_form = value.exact_nf.unwrap();
        assert_eq!(normal_form.exact_terms.len(), 3);
        for id in normal_form.exact_terms.keys() {
            let descriptor = monomials.descriptor(*id).unwrap();
            assert!(descriptor.central_factors.is_empty());
            assert_eq!(descriptor.ordered_factors.len(), 1);
            assert!(matches!(
                expressions.node(descriptor.ordered_factors[0].expression()).unwrap().operator,
                ValueOperator::Matrix(MatrixOperation::Slice {
                    column_start: 1,
                    column_end_exclusive: 5,
                    ..
                })
            ));
        }
    }

    #[test]
    fn deep_concat_projection_is_iterative_and_path_bounded() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let scalar = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap();
        let first = source_with(&mut expressions, scalar.clone(), 238);
        let mut root = first;
        let zero_constant = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(0)), Box::new([]))
            .unwrap();
        let zero = expressions
            .intern_matrix_transform(
                MatrixOperation::LiftConstantPolynomial {
                    output: scalar.clone(),
                    coefficient_bits: 1,
                },
                &[zero_constant],
            )
            .unwrap();
        let depth = 1_024;
        for level in 0..depth {
            let width = level + 2;
            root = expressions
                .intern_slice(
                    ValueOperator::Matrix(MatrixOperation::Concat {
                        axis: 1,
                        output: ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, width).unwrap(),
                        layout: MatrixLayout::row_major(1, width),
                    }),
                    &[root, zero],
                )
                .unwrap();
        }
        let right = source_with(
            &mut expressions,
            ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 2).unwrap(),
            239,
        );
        let output_columns = 2 * (depth + 1);
        let tensor = expressions
            .intern_matrix_transform(
                MatrixOperation::Tensor {
                    output: ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, output_columns)
                        .unwrap(),
                    left_layout: MatrixLayout::row_major(1, depth + 1),
                    right_layout: MatrixLayout::row_major(1, 2),
                    output_layout: MatrixLayout::row_major(1, output_columns),
                },
                &[root, right],
            )
            .unwrap();
        let slice = expressions
            .intern_slice(
                ValueOperator::Matrix(MatrixOperation::Slice {
                    row_start: 0,
                    row_end_exclusive: 1,
                    column_start: 0,
                    column_end_exclusive: 2,
                    layout: MatrixLayout::row_major(1, 2),
                }),
                &[tensor],
            )
            .unwrap();
        let (mut facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, slice);
        mark_scalar_sources_constant(&expressions, &mut facts, slice);
        let scope = monomials.scope();
        let expected_central = programs.scoped(&expressions, scope, first).unwrap();
        let expected_ordered = programs.scoped(&expressions, scope, right).unwrap();
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let value = normalizer.normalize(semantic).unwrap();
        let counters = normalizer.counters();
        drop(normalizer);
        let normal_form = value.exact_nf.unwrap();
        assert_eq!(normal_form.exact_terms.len(), 1);
        assert_eq!(normal_form.exact_terms.values().next(), Some(&BigInt::from(1_u8)));
        let id = *normal_form.exact_terms.keys().next().unwrap();
        let descriptor = monomials.descriptor(id).unwrap();
        assert_eq!(descriptor.central_factors.as_ref(), &[expected_central]);
        assert_eq!(descriptor.ordered_factors.as_ref(), &[expected_ordered]);
        assert!(counters.nodes_processed <= 4 * depth as u64 + 8);
        assert!(counters.peak_cached_values <= 3 * depth as u64 + 8);
        assert!(counters.remaining_use_releases >= depth as u64);
    }

    #[test]
    fn identity_slice_does_not_retain_parent_projection_holds() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let scalar = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap();
        let left = source_with(&mut expressions, scalar.clone(), 226);
        let first = source_with(&mut expressions, scalar.clone(), 227);
        let second = source_with(&mut expressions, scalar.clone(), 228);
        let concat = expressions
            .intern_matrix_transform(
                MatrixOperation::Concat {
                    axis: 1,
                    output: ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 2).unwrap(),
                    layout: MatrixLayout::row_major(1, 2),
                },
                &[first, second],
            )
            .unwrap();
        let product = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[left, concat])
            .unwrap();
        let identity = expressions
            .intern_slice(
                ValueOperator::Matrix(MatrixOperation::Slice {
                    row_start: 0,
                    row_end_exclusive: 1,
                    column_start: 0,
                    column_end_exclusive: 2,
                    layout: MatrixLayout::row_major(1, 2),
                }),
                &[product],
            )
            .unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, identity);
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let value = normalizer.normalize(semantic).unwrap();
        assert_eq!(value.exact_nf.unwrap().exact_terms.len(), 2);
        assert!(normalizer.counters().peak_cached_values <= 4);
        assert!(normalizer.counters().remaining_use_releases >= 2);
    }

    #[test]
    fn partial_concat_slice_does_not_use_containment_as_an_inverse() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let component = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 2).unwrap();
        let left = source_with(&mut expressions, component.clone(), 30);
        let right = source_with(&mut expressions, component, 31);
        let concat = expressions
            .intern_slice(
                ValueOperator::Matrix(MatrixOperation::Concat {
                    axis: 1,
                    output: ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 4).unwrap(),
                    layout: MatrixLayout::row_major(2, 4),
                }),
                &[left, right],
            )
            .unwrap();
        let partial = expressions
            .intern_slice(
                ValueOperator::Matrix(MatrixOperation::Slice {
                    row_start: 0,
                    row_end_exclusive: 2,
                    column_start: 1,
                    column_end_exclusive: 3,
                    layout: MatrixLayout::row_major(2, 2),
                }),
                &[concat],
            )
            .unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, partial);
        let value = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(semantic)
            .unwrap();
        let normal_form = value.exact_nf.unwrap();
        assert_eq!(normal_form.exact_terms.len(), 2);
        for id in normal_form.exact_terms.keys() {
            let factors = monomials.descriptor(*id).unwrap().ordered_factors.as_ref();
            assert_eq!(factors.len(), 1);
            assert_ne!(factors[0], semantic);
            assert!(matches!(
                expressions.node(factors[0].expression()).unwrap().operator,
                ValueOperator::Matrix(MatrixOperation::Slice {
                    row_start: 0,
                    row_end_exclusive: 2,
                    column_start: 1,
                    column_end_exclusive: 3,
                    ..
                })
            ));
        }
    }

    #[test]
    fn unequal_contained_concat_slice_remains_structural() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let component = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 2).unwrap();
        let left = source_with(&mut expressions, component.clone(), 40);
        let right = source_with(&mut expressions, component, 41);
        let concat = expressions
            .intern_slice(
                ValueOperator::Matrix(MatrixOperation::Concat {
                    axis: 1,
                    output: ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 4).unwrap(),
                    layout: MatrixLayout::row_major(2, 4),
                }),
                &[left, right],
            )
            .unwrap();
        let contained = expressions
            .intern_slice(
                ValueOperator::Matrix(MatrixOperation::Slice {
                    row_start: 0,
                    row_end_exclusive: 2,
                    column_start: 0,
                    column_end_exclusive: 1,
                    layout: MatrixLayout::row_major(2, 1),
                }),
                &[concat],
            )
            .unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, contained);
        let value = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(semantic)
            .unwrap();
        let normal_form = value.exact_nf.unwrap();
        assert_eq!(normal_form.exact_terms.len(), 2);
        for id in normal_form.exact_terms.keys() {
            let factors = monomials.descriptor(*id).unwrap().ordered_factors.as_ref();
            assert_eq!(factors.len(), 1);
            assert_ne!(factors[0], semantic);
            assert!(matches!(
                expressions.node(factors[0].expression()).unwrap().operator,
                ValueOperator::Matrix(MatrixOperation::Slice {
                    row_start: 0,
                    row_end_exclusive: 2,
                    column_start: 0,
                    column_end_exclusive: 1,
                    ..
                })
            ));
        }
    }

    #[test]
    fn crt_recompose_distributes_exact_coefficients() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let output = ResolvedMatrixType::new(BigUint::from(15_u8), 4, 2, 2).unwrap();
        // CRT recomposition consumes equal one-row matrices in the graph IR. The plaintext
        // moduli are lane metadata, not the operand ring moduli.
        let left = source_with(&mut expressions, output.clone(), 50);
        let right = source_with(&mut expressions, output.clone(), 51);
        let crt = expressions
            .intern_slice(
                ValueOperator::Matrix(MatrixOperation::CrtRecompose {
                    plaintext_moduli: Box::new([BigUint::from(3_u8), BigUint::from(5_u8)]),
                    reconstruction_coefficients: Box::new([
                        BigInt::from(2_u8),
                        BigInt::from(12_u8),
                    ]),
                    output,
                }),
                &[left, right],
            )
            .unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, crt);
        let value = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(semantic)
            .unwrap();
        let terms = &value.exact_nf.unwrap().exact_terms;
        assert_eq!(terms.len(), 2);
        assert_eq!(
            terms.values().cloned().collect::<BTreeSet<_>>(),
            BTreeSet::from([BigInt::from(2_u8), BigInt::from(12_u8),])
        );
    }

    #[test]
    fn zero_crt_coefficient_skips_a_large_lane_before_bound_inspection() {
        let bounds = [
            NumericContract::Known(CoefficientBound::Large),
            NumericContract::Known(CoefficientBound::finite(BigUint::from(7_u8))),
        ];
        assert_eq!(
            weighted_sum_bounds(&bounds, &[BigInt::from(0_u8), BigInt::from(3_u8)]).unwrap(),
            NumericContract::Known(CoefficientBound::finite(BigUint::from(21_u8)))
        );
    }
}
