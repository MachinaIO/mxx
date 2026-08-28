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
    monomial::{MonomialArena, MonomialError, MonomialId, TermMap},
    program::{ArenaError, BetaReason, FamilyValueId, ProgramArena, ValueProgramId},
    relation::{
        CanonicalLhsKey, GadgetRecompositionRegistry, GadgetRecompositionRule, NormalizationCache,
        RelationRegistry, RelationRegistryError, RuntimeSpecializationKey,
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
use tracing::info;

const MONOMIAL_GC_ALLOCATION_THRESHOLD_BYTES: u64 = 256 * 1024 * 1024;
const MONOMIAL_GC_ALLOCATION_THRESHOLD_ENV: &str = "MXX_OPERATIONAL_MONOMIAL_GC_THRESHOLD_BYTES";
const PARALLEL_GADGET_SPLICE_BATCH_TERMS: usize = 8 * 1024;

fn monomial_gc_allocation_threshold_bytes() -> u64 {
    std::env::var(MONOMIAL_GC_ALLOCATION_THRESHOLD_ENV)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|threshold| *threshold > 0)
        .unwrap_or(MONOMIAL_GC_ALLOCATION_THRESHOLD_BYTES)
}
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

/// The identityless contribution of terms whose every matrix factor has a finite bound.
///
/// This is not a cache of the complete value bound.  Large- or Missing-bearing monomials remain
/// in `PolynomialNF::exact_terms`; `bound_normal_form` combines those exact terms with this one
/// conservative noise contribution.  Addition and subtraction both add summary magnitudes, so
/// deliberately-forgotten noise identity is never used for cancellation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
enum FiniteSummaryBound {
    ExactZero,
    Finite(BoundExpression),
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BoundedSummary {
    bound: FiniteSummaryBound,
}

impl BoundedSummary {
    pub fn finite(bound: BoundExpression) -> Self {
        Self { bound: FiniteSummaryBound::Finite(bound) }
    }

    pub fn zero() -> Self {
        Self { bound: FiniteSummaryBound::ExactZero }
    }

    pub fn coefficient_bound(&self) -> NumericContract<CoefficientBound> {
        NumericContract::Known(match &self.bound {
            FiniteSummaryBound::ExactZero => CoefficientBound::ExactZero,
            FiniteSummaryBound::Finite(bound) => CoefficientBound::Finite(bound.clone()),
        })
    }

    fn is_zero(&self) -> bool {
        matches!(self.bound, FiniteSummaryBound::ExactZero)
    }

    pub(crate) fn from_contract(
        bound: NumericContract<CoefficientBound>,
    ) -> Result<Self, NormalizeError> {
        match bound {
            NumericContract::Known(CoefficientBound::ExactZero) => Ok(Self::zero()),
            NumericContract::Known(CoefficientBound::Finite(bound)) => Ok(Self::finite(bound)),
            NumericContract::Known(CoefficientBound::Large) | NumericContract::Missing => {
                Err(NormalizeError::InvalidExactPlan { reason: "bounded summary must be finite" })
            }
        }
    }
}

/// Exact polynomial terms plus a sound summary for values which cannot be reduced further.
/// Factor lists are owned by [`MonomialArena`], not by this map.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct PolynomialNF {
    pub exact_terms: TermMap<BigInt>,
    pub bounded_summary: BoundedSummary,
}

trait ExactTermAccumulator {
    fn merge(&mut self, monomial: MonomialId, coefficient: BigInt) -> Result<(), NormalizeError>;
}

impl ExactTermAccumulator for TermMap<BigInt> {
    fn merge(&mut self, monomial: MonomialId, coefficient: BigInt) -> Result<(), NormalizeError> {
        merge_term(self, monomial, coefficient);
        Ok(())
    }
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
        Self { exact_terms: BTreeMap::new(), bounded_summary: BoundedSummary::zero() }
    }

    pub fn is_zero(&self) -> bool {
        self.exact_terms.is_empty() &&
            self.bounded_summary.coefficient_bound() ==
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

struct ProductGadgetSplice {
    left: Option<MonomialId>,
    suffix: Option<MonomialId>,
    input_nf: Arc<PolynomialNF>,
    next_after: Option<MonomialId>,
    coefficient: BigInt,
    summary_pending: bool,
}

enum ProductWorkItem {
    Term(MonomialId, BigInt),
    GadgetSplice(ProductGadgetSplice),
}

/// Match the complete typed gadget/decomposition pair used by both compact preflight and
/// runtime rewriting. `require_frozen` is true for runtime use and false while compiling the
/// private plan, before the relation registry freeze barrier.
pub(crate) fn authorized_gadget_pair_rule_from(
    expressions: &ExprArena,
    facts: &FactStore,
    registry: Option<&GadgetRecompositionRegistry>,
    gadget: ExprId,
    decomposition: ExprId,
    require_frozen: bool,
) -> Result<Option<(ExprId, GadgetRecompositionRule)>, NormalizeError> {
    let decomposition_node = expressions.node(decomposition)?;
    let ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
        base,
        small,
        digit_count,
        ..
    }) = &decomposition_node.operator
    else {
        return Ok(None);
    };
    let Some(input) = decomposition_node.inputs.first().copied() else {
        return Ok(None);
    };
    let matrix_type = |expression| -> Result<Option<&ResolvedMatrixType>, NormalizeError> {
        Ok(match expressions.value_type(expression)? {
            ResolvedValueType::Matrix(matrix) => Some(matrix),
            _ => None,
        })
    };
    let (Some(input_type), Some(gadget_type), Some(decomposition_type), Some(output_type)) = (
        matrix_type(input)?,
        matrix_type(gadget)?,
        matrix_type(decomposition)?,
        matrix_type(input)?,
    ) else {
        return Ok(None);
    };
    let layout = |expression| match facts.facts(expression) {
        Ok(ValueFacts::Matrix(facts)) => Some(facts.metadata.layout.clone()),
        _ => None,
    };
    let Some(registry) = registry else { return Ok(None) };
    let gadget_layout = MatrixLayout::row_major(
        input_type.rows,
        input_type.rows.saturating_mul(*digit_count as usize),
    );
    let decomposition_layout = layout(decomposition);
    let input_layout = layout(input);
    let Some(rule) = (if require_frozen {
        registry.matching_rule(
            *base,
            *small,
            *digit_count,
            gadget_type,
            decomposition_type,
            input_type,
            output_type,
            Some(&gadget_layout),
            decomposition_layout.as_ref(),
            input_layout.as_ref(),
        )
    } else {
        registry.matching_rule_unfrozen(
            *base,
            *small,
            *digit_count,
            gadget_type,
            decomposition_type,
            input_type,
            output_type,
            Some(&gadget_layout),
            decomposition_layout.as_ref(),
            input_layout.as_ref(),
        )
    }) else {
        return Ok(None);
    };
    let mut work = vec![gadget];
    while let Some(expression) = work.pop() {
        let node = expressions.node(expression)?;
        if let Some(super::arena::MatrixConstantKind::Gadget {
            base: gadget_base,
            small: gadget_small,
        }) = node.operator.source_matrix_constant()
        {
            if *gadget_base != rule.base ||
                *gadget_small != rule.small ||
                expressions.value_type(expression)? !=
                    &ResolvedValueType::Matrix(rule.gadget_type.clone())
            {
                return Ok(None);
            }
            let layout = match facts.facts(expression) {
                Ok(ValueFacts::Matrix(facts)) => Some(facts.metadata.layout.clone()),
                _ => None,
            };
            if layout.as_ref() != rule.gadget_layout.as_ref() {
                return Ok(None);
            }
            continue;
        }
        if matches!(node.operator, ValueOperator::Matrix(MatrixOperation::Add)) &&
            node.inputs.len() == 2
        {
            work.extend(node.inputs.iter().copied());
            continue;
        }
        return Ok(None);
    }
    Ok(Some((input, rule)))
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NormalizationCounters {
    pub nodes_processed: u64,
    /// Ordinary roots count each reachable DAG node once. Compact roots report completed runtime
    /// Visit work, including nested semantic-binding visits; the separate preflight occurrence
    /// count lives only in the private compact plan.
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
    pub compact_virtual_calls: u64,
    pub compact_algebra_nodes: u64,
    pub compact_max_frames: u64,
    /// Compact evaluation does not retain a global memo for shared virtual DAGs; these counters
    /// therefore remain zero while each occurrence is evaluated independently.
    pub compact_memo_entries: u64,
    pub compact_peak_memo_entries: u64,
    pub compact_memo_term_refs: u64,
    pub compact_memo_bytes: u64,
    pub compact_live_frames: u64,
    pub compact_peak_live_frames: u64,
    pub compact_live_values: u64,
    pub compact_peak_live_values: u64,
    pub compact_logical_add_sub: u64,
    pub compact_logical_scale: u64,
    pub compact_strict_products: u64,
    pub compact_concrete_shell_nodes: u64,
    pub compact_max_virtual_frames: u64,
    pub compact_max_virtual_values: u64,
    /// Number of compact gadget-shell occurrences admitted by the lowering plan.
    pub compact_planned_shell_occurrences: u64,
    /// Number of distinct compact gadget shell identities admitted by the lowering plan.
    pub compact_planned_unique_shells: u64,
    pub compact_shell_allocated: u64,
    pub compact_shell_hits: u64,
    pub compact_shell_new: u64,
    pub compact_shell_holds_current: u64,
    pub compact_shell_holds_peak: u64,
    pub compact_shell_holds_released: u64,
    pub compact_shell_holds_unmatched: u64,
    pub compact_scalar_consumers: u64,
    pub compact_scalar_holds_current: u64,
    pub compact_scalar_holds_peak: u64,
    pub compact_scalar_holds_released: u64,
    pub compact_scalar_holds_unmatched: u64,
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
    InvalidExactPlan { reason: &'static str },
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

enum ScalarActionNormalization {
    NotApplicable,
    Opaque,
    Exact(PolynomialNF),
}

struct CompactValue {
    semantic: Option<ScopedExprId>,
    exact_nf: Option<Arc<PolynomialNF>>,
    coefficient_bound: NumericContract<CoefficientBound>,
    resolved_type: ResolvedValueType,
    hold: Option<ExprId>,
}

#[derive(Clone)]
struct CompactGadgetHold {
    normal_form: Arc<PolynomialNF>,
    remaining: u64,
    /// Shell identities with one ready product-consumer token each.  The product splice consumes
    /// the token for its exact shell; a generic lookup cannot release another shell's hold.
    consumers: BTreeMap<ExprId, u64>,
}

#[derive(Default)]
struct CompactProductContext {
    used_shells: BTreeSet<(ExprId, ExprId)>,
}

/// Private lowering witness for a compact root.  Expression IDs are deliberately retained in the
/// plan: the root normalizer must consume the exact shell/input pair that preflight inspected,
/// never a same-shaped replacement discovered after freezing.
#[derive(Clone, Debug, Default)]
pub(crate) struct CompactShellPlan {
    pub(crate) gadget_shells: BTreeMap<(ExprId, ExprId), CompactGadgetPlan>,
    pub(crate) scalar_factors: BTreeMap<ExprId, CompactScalarPlan>,
    /// Exact one-sided scalar ProgramCall authorizations. The side is part of the key because
    /// scalar-left and scalar-right products have different ordered semantics.
    pub(crate) scalar_program_calls: BTreeMap<(ExprId, ExprId, bool), CompactScalarProgramCallPlan>,
    /// Number of expression occurrences visited by the structural preflight traversal. Shared DAG
    /// edges count once per incoming occurrence; this is not a compact runtime work counter.
    pub(crate) preflight_node_occurrences: u64,
    pub(crate) shell_allocated: u64,
    pub(crate) shell_hits: u64,
    pub(crate) shell_new: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct CompactGadgetPlan {
    pub(crate) rule: GadgetRecompositionRule,
    pub(crate) occurrences: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct CompactScalarPlan {
    pub(crate) value_type: ResolvedMatrixType,
    pub(crate) occurrences: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct CompactScalarProgramCallPlan {
    pub(crate) value_type: ResolvedMatrixType,
    pub(crate) occurrences: u64,
}

impl CompactShellPlan {
    pub(crate) fn insert_gadget(
        &mut self,
        shell: ExprId,
        input: ExprId,
        rule: GadgetRecompositionRule,
    ) -> bool {
        self.insert_gadget_count(shell, input, rule, 1)
    }

    pub(crate) fn insert_gadget_count(
        &mut self,
        shell: ExprId,
        input: ExprId,
        rule: GadgetRecompositionRule,
        occurrences: u64,
    ) -> bool {
        let mut compatible = true;
        self.gadget_shells
            .entry((shell, input))
            .and_modify(|entry| {
                if entry.rule == rule {
                    if let Some(total) = entry.occurrences.checked_add(occurrences) {
                        entry.occurrences = total;
                    } else {
                        compatible = false;
                    }
                } else {
                    compatible = false;
                }
            })
            .or_insert(CompactGadgetPlan { rule, occurrences });
        compatible
    }

    pub(crate) fn gadget_occurrences(&self) -> u64 {
        self.gadget_shells.values().map(|entry| entry.occurrences).sum()
    }

    pub(crate) fn insert_scalar(
        &mut self,
        expression: ExprId,
        value_type: ResolvedMatrixType,
    ) -> bool {
        self.insert_scalar_count(expression, value_type, 1)
    }

    pub(crate) fn insert_scalar_count(
        &mut self,
        expression: ExprId,
        value_type: ResolvedMatrixType,
        occurrences: u64,
    ) -> bool {
        let mut compatible = true;
        self.scalar_factors
            .entry(expression)
            .and_modify(|entry| {
                if entry.value_type == value_type {
                    if let Some(total) = entry.occurrences.checked_add(occurrences) {
                        entry.occurrences = total;
                    } else {
                        compatible = false;
                    }
                } else {
                    compatible = false;
                }
            })
            .or_insert(CompactScalarPlan { value_type, occurrences });
        compatible
    }

    pub(crate) fn scalar_occurrences(&self) -> u64 {
        self.scalar_factors.values().map(|entry| entry.occurrences).sum::<u64>() +
            self.scalar_program_calls.values().map(|entry| entry.occurrences).sum::<u64>()
    }

    pub(crate) fn insert_scalar_program_call(
        &mut self,
        consumer: ExprId,
        call: ExprId,
        scalar_is_right: bool,
        value_type: ResolvedMatrixType,
    ) -> bool {
        self.insert_scalar_program_call_count(consumer, call, scalar_is_right, value_type, 1)
    }

    pub(crate) fn insert_scalar_program_call_count(
        &mut self,
        consumer: ExprId,
        call: ExprId,
        scalar_is_right: bool,
        value_type: ResolvedMatrixType,
        occurrences: u64,
    ) -> bool {
        let mut compatible = true;
        self.scalar_program_calls
            .entry((consumer, call, scalar_is_right))
            .and_modify(|entry| {
                if entry.value_type == value_type {
                    if let Some(total) = entry.occurrences.checked_add(occurrences) {
                        entry.occurrences = total;
                    } else {
                        compatible = false;
                    }
                } else {
                    compatible = false;
                }
            })
            .or_insert(CompactScalarProgramCallPlan { value_type, occurrences });
        compatible
    }
}

impl CompactValue {
    fn from_analyzed(value: AnalyzedValue, resolved_type: ResolvedValueType) -> Self {
        Self {
            semantic: Some(value.semantic),
            exact_nf: value.exact_nf,
            coefficient_bound: value.coefficient_bound,
            resolved_type,
            hold: None,
        }
    }

    fn as_analyzed(&self) -> Result<AnalyzedValue, NormalizeError> {
        Ok(AnalyzedValue {
            semantic: self.semantic.ok_or(NormalizeError::InvalidExactPlan {
                reason: "eliminable virtual value reached concrete operator",
            })?,
            exact_nf: self.exact_nf.clone(),
            coefficient_bound: self.coefficient_bound.clone(),
        })
    }
}

enum CompactFrame {
    Visit { expression: ExprId, bindings: Vec<ScopedExprId> },
    CombineMatrix { expression: ExprId, node: Arc<ExprNode> },
    CombineGadget { expression: ExprId, node: Arc<ExprNode> },
    CombineCall { expression: ExprId, node: Arc<ExprNode> },
    CombineConcrete { expression: ExprId, node: Arc<ExprNode> },
}

enum CompactRootSource {
    Closed(ScopedExprId),
    Family(super::program::FamilyValueId),
}

#[derive(Clone, Copy)]
enum CompactProgressSource {
    Closed,
    Family,
}

impl CompactProgressSource {
    const fn stage(self) -> &'static str {
        match self {
            Self::Closed => "normalize_compact_root",
            Self::Family => "normalize_compact_family",
        }
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
    /// Finite relation endpoints retained with identity for exactly one direct registered
    /// Multiply consumer. They remain bounded values; this set only delays numeric compression
    /// until that immediate boundary has either rewritten or folded them.
    retained_bounded_endpoints: BTreeSet<ExprId>,
    /// Finite factors below a uniquely-consumed 1x1 scalar operand of a non-scalar Multiply.
    /// Keeping this one lexical branch exact until the scalar-action boundary preserves the full
    /// ordered Large product without introducing a third durable polynomial lane.
    retained_bounded_scalar_factors: BTreeSet<ExprId>,
    /// Durable value-level transfer results for expressions which may be released from `cache`
    /// before the root's exact monomials are folded.
    expression_bounds: BTreeMap<ExprId, NumericContract<CoefficientBound>>,
    remaining_uses: BTreeMap<ExprId, usize>,
    /// Normalized inputs retained by the structural gadget-recomposition hold.
    gadget_input_nfs: BTreeMap<ExprId, CompactGadgetHold>,
    /// Finite scalar leaves retained until their exact scalar-action consumer has run.
    compact_scalar_holds: BTreeMap<ExprId, u64>,
    compact_plan: Option<CompactShellPlan>,
    compact_plan_remaining: Option<CompactShellPlan>,
    counters: NormalizationCounters,
    normalization_depth: u32,
    relation_rewriting_enabled: bool,
    suppress_product_relation_closure: bool,
    fold_final_no_match: bool,
    /// Slots below this outer-call high-water are externally observable and remain pinned even
    /// when no current normalization owner references them.
    protected_monomial_prefix: usize,
    monomial_gc_allocation_threshold_bytes: u64,
    gadget_splice_batch_terms: usize,
    compact_mode: bool,
    compact_product_contexts: Vec<CompactProductContext>,
    compact_progress_source: CompactProgressSource,
}

impl<'a> Normalizer<'a> {
    fn clear_value_cache(&mut self) {
        self.cache.clear();
    }

    fn log_compact_progress(&self) {
        let processed = self.counters.nodes_processed;
        if processed == 0 || !(processed.is_power_of_two() || processed % 1_000_000 == 0) {
            return;
        }
        let preflight =
            self.compact_plan.as_ref().map_or(0, |plan| plan.preflight_node_occurrences);
        info!(
            target: "mxx_correctness::operational_noise",
            stage = self.compact_progress_source.stage(),
            event = "progress",
            nodes_processed = processed,
            preflight_node_occurrences = preflight,
            frames = self.counters.compact_live_frames,
            values = self.counters.compact_live_values,
            exact_terms = self.counters.final_exact_term_count,
            shell_holds = self.counters.compact_shell_holds_current,
            scalar_holds = self.counters.compact_scalar_holds_current,
            "compact normalization progress"
        );
    }

    fn insert_value_cache(&mut self, expression: ExprId, value: Arc<AnalyzedValue>) {
        self.cache.insert(expression, value);
    }

    fn take_value_cache(&mut self, expression: ExprId) -> Option<Arc<AnalyzedValue>> {
        self.cache.remove(&expression)
    }

    fn remove_value_cache(&mut self, expression: ExprId) {
        self.cache.remove(&expression);
    }

    fn clear_gadget_holds(&mut self) {
        self.gadget_input_nfs.clear();
    }

    fn insert_gadget_hold(
        &mut self,
        expression: ExprId,
        normal_form: Arc<PolynomialNF>,
        shell: Option<ExprId>,
    ) {
        let entry = self.gadget_input_nfs.entry(expression).or_insert_with(|| CompactGadgetHold {
            normal_form: normal_form.clone(),
            remaining: 0,
            consumers: BTreeMap::new(),
        });
        entry.remaining = entry.remaining.saturating_add(1);
        entry.normal_form = normal_form;
        if let Some(shell) = shell {
            *entry.consumers.entry(shell).or_default() += 1;
        }
        if !self.compact_mode {
            return;
        }
        if self.compact_plan.is_none() {
            self.counters.compact_planned_shell_occurrences =
                self.counters.compact_planned_shell_occurrences.saturating_add(1);
        }
        self.counters.compact_shell_holds_current =
            self.counters.compact_shell_holds_current.saturating_add(1);
        self.counters.compact_shell_holds_peak =
            self.counters.compact_shell_holds_peak.max(self.counters.compact_shell_holds_current);
    }

    fn consume_compact_shell_plan(
        &mut self,
        shell: ExprId,
        input: ExprId,
    ) -> Result<(), NormalizeError> {
        let Some(plan) = self.compact_plan_remaining.as_mut() else { return Ok(()) };
        let Some(entry) = plan.gadget_shells.get_mut(&(shell, input)) else {
            return Err(NormalizeError::InvalidExactPlan {
                reason: "compact gadget shell was not preflighted",
            });
        };
        if entry.occurrences == 0 {
            return Err(NormalizeError::InvalidExactPlan {
                reason: "compact gadget shell occurrence over-consumed",
            });
        }
        entry.occurrences -= 1;
        Ok(())
    }

    /// Release one exact decomposition-input token after its authorized product boundary has
    /// completed. The entry itself is removed at its final direct consumer so a later accidental
    /// lookup cannot silently resurrect a consumed hold.
    fn release_compact_gadget_hold(
        &mut self,
        shell: ExprId,
        input: ExprId,
    ) -> Result<(), NormalizeError> {
        let remove = {
            let hold =
                self.gadget_input_nfs.get_mut(&input).ok_or(NormalizeError::InvalidExactPlan {
                    reason: "compact gadget hold disappeared",
                })?;
            let token = hold.consumers.get_mut(&shell).ok_or(NormalizeError::InvalidExactPlan {
                reason: "compact gadget consumer token disappeared",
            })?;
            if *token == 0 || hold.remaining == 0 {
                return Err(NormalizeError::InvalidExactPlan {
                    reason: "compact gadget consumer token over-consumed",
                });
            }
            *token -= 1;
            hold.remaining -= 1;
            if *token == 0 {
                hold.consumers.remove(&shell);
            }
            hold.remaining == 0
        };
        if remove {
            self.gadget_input_nfs.remove(&input);
        }
        self.counters.compact_shell_holds_current =
            self.counters.compact_shell_holds_current.saturating_sub(1);
        self.counters.compact_shell_holds_released =
            self.counters.compact_shell_holds_released.saturating_add(1);
        Ok(())
    }

    fn exact_compact_gadget_rule(
        &self,
        shell: ExprId,
        input: ExprId,
    ) -> Result<GadgetRecompositionRule, NormalizeError> {
        let node = self.expressions.node(shell)?;
        let ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
            base,
            small,
            digit_count,
            ..
        }) = &node.operator
        else {
            return Err(NormalizeError::InvalidExactPlan {
                reason: "compact shell operator changed",
            });
        };
        if node.inputs.as_ref() != [input] {
            return Err(NormalizeError::InvalidExactPlan { reason: "compact shell input changed" });
        }
        let ResolvedValueType::Matrix(input_type) = self.expressions.value_type(input)? else {
            return Err(NormalizeError::InvalidExactPlan {
                reason: "compact shell input type changed",
            });
        };
        let ResolvedValueType::Matrix(decomposition_type) = self.expressions.value_type(shell)?
        else {
            return Err(NormalizeError::InvalidExactPlan {
                reason: "compact shell output type changed",
            });
        };
        let gadget_type = ResolvedMatrixType::new(
            input_type.modulus.clone(),
            input_type.ring_dimension,
            input_type.rows,
            decomposition_type.rows,
        )?;
        let layout = |expression| match self.facts.facts(expression) {
            Ok(ValueFacts::Matrix(facts)) => Some(facts.metadata.layout.clone()),
            _ => None,
        };
        let gadget_layout = MatrixLayout::row_major(
            input_type.rows,
            input_type.rows.saturating_mul(*digit_count as usize),
        );
        self.gadget_recompositions
            .ok_or(NormalizeError::InvalidExactPlan { reason: "compact gadget registry missing" })?
            .matching_rule(
                *base,
                *small,
                *digit_count,
                &gadget_type,
                decomposition_type,
                input_type,
                input_type,
                Some(&gadget_layout),
                layout(shell).as_ref(),
                layout(input).as_ref(),
            )
            .ok_or(NormalizeError::InvalidExactPlan {
                reason: "compact gadget rule missing or ambiguous",
            })
    }

    fn retain_compact_scalar_hold(
        &mut self,
        expression: ExprId,
        value_type: &ResolvedValueType,
        bound: &NumericContract<CoefficientBound>,
    ) -> Option<ExprId> {
        let ResolvedValueType::Matrix(matrix) = value_type else { return None };
        if matrix.rows != 1 ||
            matrix.columns != 1 ||
            !matches!(bound, NumericContract::Known(CoefficientBound::Finite(_)))
        {
            return None;
        }
        if self.compact_mode {
            let planned = self.compact_plan_remaining.as_ref().is_some_and(|plan| {
                plan.scalar_factors
                    .get(&expression)
                    .is_some_and(|entry| entry.value_type == *matrix && entry.occurrences > 0)
            });
            if !planned {
                return None;
            }
        }
        *self.compact_scalar_holds.entry(expression).or_default() += 1;
        self.retained_bounded_scalar_factors.insert(expression);
        if !self.compact_mode {
            self.counters.compact_scalar_consumers =
                self.counters.compact_scalar_consumers.saturating_add(1);
        }
        self.counters.compact_scalar_holds_current =
            self.counters.compact_scalar_holds_current.saturating_add(1);
        self.counters.compact_scalar_holds_peak =
            self.counters.compact_scalar_holds_peak.max(self.counters.compact_scalar_holds_current);
        Some(expression)
    }

    fn release_compact_scalar_hold(&mut self, expression: ExprId) -> Result<(), NormalizeError> {
        let Some(remaining) = self.compact_scalar_holds.get_mut(&expression) else {
            return Err(NormalizeError::InvalidExactPlan {
                reason: "compact scalar hold was over-released",
            });
        };
        if *remaining == 0 {
            return Err(NormalizeError::InvalidExactPlan {
                reason: "compact scalar hold count was exhausted",
            });
        }
        *remaining -= 1;
        self.counters.compact_scalar_holds_current =
            self.counters.compact_scalar_holds_current.saturating_sub(1);
        self.counters.compact_scalar_holds_released =
            self.counters.compact_scalar_holds_released.saturating_add(1);
        if *remaining == 0 {
            self.compact_scalar_holds.remove(&expression);
            self.retained_bounded_scalar_factors.remove(&expression);
        }
        Ok(())
    }

    fn consume_compact_scalar_plan(
        &mut self,
        expression: ExprId,
        value_type: &ResolvedMatrixType,
    ) -> Result<(), NormalizeError> {
        let plan = self
            .compact_plan_remaining
            .as_mut()
            .ok_or(NormalizeError::InvalidExactPlan { reason: "compact scalar plan is missing" })?;
        let entry =
            plan.scalar_factors.get_mut(&expression).ok_or(NormalizeError::InvalidExactPlan {
                reason: "compact scalar consumer is unplanned",
            })?;
        if entry.value_type != *value_type || entry.occurrences == 0 {
            return Err(NormalizeError::InvalidExactPlan {
                reason: "compact scalar consumer count is inconsistent",
            });
        }
        entry.occurrences -= 1;
        Ok(())
    }

    fn consume_compact_scalar_program_call(
        &mut self,
        consumer: ExprId,
        expression: ExprId,
        scalar_is_right: bool,
        value_type: &ResolvedMatrixType,
    ) -> Result<(), NormalizeError> {
        let plan = self
            .compact_plan_remaining
            .as_mut()
            .ok_or(NormalizeError::InvalidExactPlan { reason: "compact scalar plan is missing" })?;
        let entry = plan
            .scalar_program_calls
            .get_mut(&(consumer, expression, scalar_is_right))
            .ok_or(NormalizeError::InvalidExactPlan {
                reason: "compact scalar ProgramCall consumer is unplanned",
            })?;
        if entry.value_type != *value_type || entry.occurrences == 0 {
            return Err(NormalizeError::InvalidExactPlan {
                reason: "compact scalar ProgramCall consumer count is inconsistent",
            });
        }
        entry.occurrences -= 1;
        Ok(())
    }

    /// Run only after a depth-one node has been fully committed to every durable owner. Product
    /// and relation worklists are lexical locals inside `evaluate_node` and are therefore gone at
    /// this boundary. Root collection is exact and fail-closed; the arena validates every ID
    /// before dropping any descriptor.
    fn sweep_monomials_at_node_commit(&mut self) -> Result<(), NormalizeError> {
        if self.normalization_depth != 1 ||
            self.monomials.allocated_payload_since_sweep() <
                self.monomial_gc_allocation_threshold_bytes
        {
            return Ok(());
        }
        let cache_roots = self.cache.values().flat_map(|value| {
            value.exact_nf.iter().flat_map(|normal_form| normal_form.exact_terms.keys().copied())
        });
        let gadget_roots = self
            .gadget_input_nfs
            .values()
            .flat_map(|hold| hold.normal_form.exact_terms.keys().copied());
        let canonical_roots =
            self.normalization.as_deref().into_iter().flat_map(NormalizationCache::monomial_roots);
        let arena = self.monomials.token();
        let canonical_roots = canonical_roots.filter(move |root| root.arena() == arena);
        self.monomials.sweep(
            self.protected_monomial_prefix,
            cache_roots.chain(gadget_roots).chain(canonical_roots),
        )?;
        Ok(())
    }

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
        let protected_monomial_prefix = monomials.len();
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
            retained_bounded_endpoints: BTreeSet::new(),
            retained_bounded_scalar_factors: BTreeSet::new(),
            expression_bounds: BTreeMap::new(),
            remaining_uses: BTreeMap::new(),
            gadget_input_nfs: BTreeMap::new(),
            compact_scalar_holds: BTreeMap::new(),
            compact_plan: None,
            compact_plan_remaining: None,
            counters: NormalizationCounters::default(),
            normalization_depth: 0,
            relation_rewriting_enabled: true,
            suppress_product_relation_closure: false,
            fold_final_no_match: true,
            protected_monomial_prefix,
            monomial_gc_allocation_threshold_bytes: monomial_gc_allocation_threshold_bytes(),
            gadget_splice_batch_terms: PARALLEL_GADGET_SPLICE_BATCH_TERMS,
            compact_mode: false,
            compact_product_contexts: Vec::new(),
            compact_progress_source: CompactProgressSource::Closed,
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

    pub(crate) fn with_compact_shell_plan(mut self, plan: CompactShellPlan) -> Self {
        self.compact_plan = Some(plan);
        self
    }

    pub fn counters(&self) -> NormalizationCounters {
        self.counters
    }

    pub fn normalize(&mut self, root: ScopedExprId) -> Result<AnalyzedValue, NormalizeError> {
        self.normalize_with_authority(root, None)
    }

    fn normalize_with_existing_scope_proof(
        &mut self,
        root: ScopedExprId,
        proof: ScopeProof,
    ) -> Result<AnalyzedValue, NormalizeError> {
        self.expressions.validate_scope_proof_for_root(
            &proof,
            root.program(),
            root.expression(),
        )?;
        self.normalize_with_authority(root, Some(proof))
    }

    fn normalize_with_authority(
        &mut self,
        root: ScopedExprId,
        scope_proof: Option<ScopeProof>,
    ) -> Result<AnalyzedValue, NormalizeError> {
        let outermost = self.normalization_depth == 0;
        if outermost {
            self.protected_monomial_prefix = self.monomials.len();
        }
        self.normalization_depth = self.normalization_depth.saturating_add(1);
        let result = self.normalize_inner(root, scope_proof);
        self.normalization_depth = self.normalization_depth.saturating_sub(1);
        result
    }

    fn normalize_inner(
        &mut self,
        root: ScopedExprId,
        scope_proof: Option<ScopeProof>,
    ) -> Result<AnalyzedValue, NormalizeError> {
        if root.program() != self.scope {
            return Err(NormalizeError::InvalidScope {
                expected: self.scope,
                actual: root.program(),
            });
        }
        // Relation closure is lexical over the complete root word. Defer it until all expression
        // children have been assembled; otherwise a child rewrite could consume a boundary that
        // only becomes meaningful in its parent.
        let saved_relation_rewriting = self.relation_rewriting_enabled;
        self.relation_rewriting_enabled = false;
        let result = (|| {
            let mut scope_proof = match scope_proof {
                Some(proof) => proof,
                None => self.expressions.scope_proof(root.program(), root.expression())?,
            };
            self.clear_value_cache();
            self.retained_bounded_endpoints.clear();
            self.retained_bounded_scalar_factors.clear();
            self.expression_bounds.clear();
            self.remaining_uses.clear();
            self.clear_gadget_holds();
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
                    for child in &node.inputs {
                        work.push((*child, false));
                    }
                    continue;
                }
                let value = self.evaluate_node(&mut scope_proof, expression, node.as_ref())?;
                self.expression_bounds.insert(expression, value.coefficient_bound.clone());
                self.counters.nodes_processed = self.counters.nodes_processed.saturating_add(1);
                self.insert_value_cache(expression, Arc::new(value));
                completed.insert(expression);
                self.sweep_monomials_at_node_commit()?;
                self.counters.peak_cached_values =
                    self.counters.peak_cached_values.max(self.cache.len() as u64);
            }

            let value = self
                .take_value_cache(root.expression())
                .ok_or(NormalizeError::MissingCachedValue { expression: root.expression() })?;
            let mut value = Arc::try_unwrap(value).map_err(|_| {
                NormalizeError::SharedRootCacheValue { expression: root.expression() }
            })?;
            self.relation_rewriting_enabled = saved_relation_rewriting;
            if self.relations.is_some() && self.relation_rewriting_enabled {
                if let Some(exact_nf) = value.exact_nf.as_mut() {
                    let normal_form = Arc::make_mut(exact_nf);
                    self.rewrite_relations(normal_form)?;
                    value.coefficient_bound = self.bound_normal_form(normal_form)?;
                }
            }
            if self.fold_final_no_match &&
                self.relations.is_some() &&
                self.relation_rewriting_enabled
            {
                if let Some(exact_nf) = value.exact_nf.as_mut() {
                    let normal_form = Arc::make_mut(exact_nf);
                    let rebound = self.bound_normal_form(normal_form)?;
                    self.fold_finite_no_match_terms(normal_form, false)?;
                    value.coefficient_bound = rebound;
                    if normal_form.is_zero() {
                        value.coefficient_bound =
                            NumericContract::Known(CoefficientBound::ExactZero);
                        normal_form.bounded_summary = BoundedSummary::zero();
                    }
                }
            }
            self.counters.relation_remaining = value
                .exact_nf
                .as_deref()
                .map(|normal_form| self.count_relation_remaining(normal_form))
                .unwrap_or(0);
            self.counters.final_exact_term_count = value
                .exact_nf
                .as_ref()
                .map_or(0, |normal_form| normal_form.exact_terms.len() as u64);
            Ok(value)
        })();
        self.relation_rewriting_enabled = saved_relation_rewriting;
        result
    }

    /// Stage A is one exact dispatch lookup. Stage B substitutes the identical index expression
    /// into every plan and canonicalizes through this normalizer. Runtime results enter only the
    /// ordinary memo owned by `NormalizationCache`.
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
        let trapdoor = self.programs.beta_reduce_materialized_with_reason(
            self.expressions,
            registration.lhs.trapdoor_plan,
            &[index.expression()],
            BetaReason::NormalizerSpecialization,
        )?;
        let pairing = self.programs.beta_reduce_materialized_with_reason(
            self.expressions,
            registration.lhs.public_pairing,
            &[index.expression()],
            BetaReason::NormalizerSpecialization,
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
            .intern_arc(target)?;
        Ok((CanonicalLhsKey { layout: registration.lhs.layout.clone(), monomial }, rhs))
    }

    fn specialize_family_plan(
        &mut self,
        plan: ValueProgramId,
        index: ScopedExprId,
        index_range: super::arena::TrustedIndexRange,
    ) -> Result<ExprId, NormalizeError> {
        self.programs
            .call_family_in_range_with_reason(
                self.expressions,
                super::program::FamilyValueId::from_program(plan),
                index.expression(),
                index_range,
                BetaReason::NormalizerSpecialization,
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
        let saved_retained_bounded_endpoints = std::mem::take(&mut self.retained_bounded_endpoints);
        let saved_retained_bounded_scalar_factors =
            std::mem::take(&mut self.retained_bounded_scalar_factors);
        let saved_expression_bounds = std::mem::take(&mut self.expression_bounds);
        let saved_uses = std::mem::take(&mut self.remaining_uses);
        let saved_gadget_input_nfs = std::mem::take(&mut self.gadget_input_nfs);
        let saved_compact_scalar_holds = std::mem::take(&mut self.compact_scalar_holds);
        let saved_counters = self.counters;
        let saved_fold_final_no_match = self.fold_final_no_match;
        let saved_compact_mode = self.compact_mode;
        let saved_suppress_product_relation_closure = self.suppress_product_relation_closure;
        self.fold_final_no_match = false;
        // Relation specialization is an isolated ordinary normalization.  In particular, its
        // product may contain the same gadget/scalar-shaped children as the outer compact root,
        // but it has no private lowering plan or hold tokens to consume.
        self.compact_mode = false;
        self.suppress_product_relation_closure = saved_suppress_product_relation_closure;

        let value = self.normalize_with_existing_scope_proof(scoped, proof);
        let nested_expression_bounds = std::mem::take(&mut self.expression_bounds);
        self.cache = saved_cache;
        self.retained_bounded_endpoints = saved_retained_bounded_endpoints;
        self.retained_bounded_scalar_factors = saved_retained_bounded_scalar_factors;
        self.expression_bounds = saved_expression_bounds;
        if value.is_ok() {
            self.merge_expression_bounds(nested_expression_bounds);
        }
        self.remaining_uses = saved_uses;
        self.gadget_input_nfs = saved_gadget_input_nfs;
        self.compact_scalar_holds = saved_compact_scalar_holds;
        self.counters = saved_counters;
        self.fold_final_no_match = saved_fold_final_no_match;
        self.compact_mode = saved_compact_mode;
        self.suppress_product_relation_closure = saved_suppress_product_relation_closure;
        value
    }

    fn normalize_specialized_root_without_relations(
        &mut self,
        root: ExprId,
    ) -> Result<AnalyzedValue, NormalizeError> {
        let previous_relation_rewriting = self.relation_rewriting_enabled;
        let previous_suppress_product_relation_closure = self.suppress_product_relation_closure;
        self.relation_rewriting_enabled = false;
        self.suppress_product_relation_closure = true;
        let value = self.normalize_specialized_root(root);
        self.relation_rewriting_enabled = previous_relation_rewriting;
        self.suppress_product_relation_closure = previous_suppress_product_relation_closure;
        value
    }

    fn normalize_plan(
        &mut self,
        plan: ValueProgramId,
        index: ScopedExprId,
    ) -> Result<(ExprId, Arc<PolynomialNF>), NormalizeError> {
        let root = self.programs.beta_reduce_materialized_with_reason(
            self.expressions,
            plan,
            &[index.expression()],
            BetaReason::NormalizerSpecialization,
        )?;
        let value = self.normalize_specialized_root(root)?;
        let normal_form =
            value.exact_nf.clone().ok_or_else(|| NormalizeError::UnsupportedOperator {
                operator: "relation plan without exact normal form".into(),
            })?;
        Ok((root, normal_form))
    }

    fn compute_use_counts(&mut self, root: ExprId) -> Result<BTreeSet<ExprId>, NormalizeError> {
        let mut reachable = BTreeSet::new();
        let mut real_consumers = BTreeMap::<ExprId, BTreeSet<ExprId>>::new();
        let mut work = vec![root];
        while let Some(expression) = work.pop() {
            if !reachable.insert(expression) {
                continue;
            }
            let node = self.expressions.node(expression)?;
            for child in &node.inputs {
                *self.remaining_uses.entry(*child).or_default() += 1;
                real_consumers.entry(*child).or_default().insert(expression);
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

        // Tall's `right * plaintext` is a one-sided scalar action.  If the finite 1x1 plaintext
        // branch were collapsed before this edge, a later Large row product could not recover the
        // exact scalar factors required by the two-lane contract.  Delay only uniquely-consumed
        // scalar branches, and only until their immediate non-scalar Multiply.
        let mut scalar_factor_work = Vec::new();
        for expression in &reachable {
            let node = self.expressions.node(*expression)?;
            if !matches!(node.operator, ValueOperator::Matrix(MatrixOperation::Multiply)) ||
                node.inputs.len() != 2
            {
                continue;
            }
            let mut scalar = None;
            let mut non_scalar = false;
            for input in &node.inputs {
                let ResolvedValueType::Matrix(matrix) = self.expressions.value_type(*input)? else {
                    continue;
                };
                if matrix.rows == 1 && matrix.columns == 1 {
                    scalar = Some(*input);
                } else {
                    non_scalar = true;
                }
            }
            let Some(scalar) = scalar.filter(|_| non_scalar) else { continue };
            if node.inputs.iter().filter(|input| **input == scalar).count() != 1 ||
                !real_consumers.get(&scalar).is_some_and(|consumers| {
                    consumers.len() == 1 && consumers.contains(expression)
                })
            {
                continue;
            }
            scalar_factor_work.push(scalar);
        }
        while let Some(expression) = scalar_factor_work.pop() {
            if !self.retained_bounded_scalar_factors.insert(expression) {
                continue;
            }
            scalar_factor_work.extend(self.expressions.node(expression)?.inputs.iter().copied());
        }

        // A finite endpoint is retained only for one semantic input edge of one direct Multiply.
        // Closed and universal endpoints must match the complete registered ordered pair. A
        // gadget decomposition may use the registry's exact typed half here because the opposite
        // operand can be an Additive NF; the product executor revalidates the complete G|D pair
        // before rewriting and otherwise treats the endpoint as ordinary bounded input.
        let mut universal_endpoint_pairs = BTreeSet::new();
        let mut universal_direct_public_pairs = BTreeSet::new();
        if let Some(relations) = self.relations {
            for (preimage, public, _) in relations.universal_plan_roles() {
                universal_endpoint_pairs.insert((public, preimage));
                if let Some(public_family) = self.programs.family_for_program(public) {
                    let body = self.programs.family_body(public_family)?;
                    if self.expressions.free_arguments(body)?.is_empty() {
                        universal_direct_public_pairs.insert((body, preimage));
                    }
                }
            }
        }
        for expression in &reachable {
            let Some(consumers) = real_consumers.get(expression) else { continue };
            if consumers.len() != 1 {
                continue;
            }
            let consumer = *consumers.first().expect("length checked above");
            let consumer_node = self.expressions.node(consumer)?;
            if !matches!(consumer_node.operator, ValueOperator::Matrix(MatrixOperation::Multiply)) ||
                consumer_node.inputs.len() != 2 ||
                consumer_node.inputs.iter().filter(|input| *input == expression).count() != 1
            {
                continue;
            }
            let left_expression = consumer_node.inputs[0];
            let right_expression = consumer_node.inputs[1];
            let left_node = self.expressions.node(left_expression)?;
            let right_node = self.expressions.node(right_expression)?;
            let universal = match (&left_node.operator, &right_node.operator) {
                (
                    ValueOperator::ProgramCall { program: public },
                    ValueOperator::ProgramCall { program: preimage },
                ) => universal_endpoint_pairs.contains(&(*public, *preimage)),
                (_, ValueOperator::ProgramCall { program: preimage }) => {
                    universal_direct_public_pairs.contains(&(left_expression, *preimage))
                }
                _ => false,
            };
            let gadget = if *expression == right_expression {
                if let ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
                    base,
                    small,
                    digit_count,
                    ..
                }) = &right_node.operator
                {
                    let Some(input) = right_node.inputs.first().copied() else { continue };
                    let (
                        ResolvedValueType::Matrix(decomposition_type),
                        ResolvedValueType::Matrix(input_type),
                    ) = (
                        self.expressions.value_type(right_expression)?,
                        self.expressions.value_type(input)?,
                    )
                    else {
                        continue
                    };
                    let decomposition_layout = self
                        .matrix_value_facts(right_expression)
                        .map(|facts| &facts.metadata.layout);
                    let input_layout =
                        self.matrix_value_facts(input).map(|facts| &facts.metadata.layout);
                    self.gadget_recompositions
                        .map(|registry| {
                            registry.allows_decomposition_half(
                                *base,
                                *small,
                                *digit_count,
                                decomposition_type,
                                input_type,
                                decomposition_layout,
                                input_layout,
                            )
                        })
                        .transpose()?
                        .unwrap_or(false)
                } else {
                    false
                }
            } else {
                false
            };
            if universal || gadget {
                self.retained_bounded_endpoints.insert(*expression);
            }
        }

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
            self.remove_value_cache(expression);
            self.counters.remaining_use_releases =
                self.counters.remaining_use_releases.saturating_add(1);
        }
        Ok(value)
    }

    fn gadget_input_nf(
        &mut self,
        shell: ExprId,
        expression: ExprId,
    ) -> Result<Option<Arc<PolynomialNF>>, NormalizeError> {
        if self.gadget_input_nfs.contains_key(&expression) {
            let (normal_form, has_token) = {
                let hold = self.gadget_input_nfs.get(&expression).ok_or(
                    NormalizeError::InvalidExactPlan { reason: "compact gadget hold disappeared" },
                )?;
                (hold.normal_form.clone(), hold.consumers.get(&shell).copied().unwrap_or(0) != 0)
            };
            if self.compact_mode {
                if !has_token {
                    return Err(NormalizeError::InvalidExactPlan {
                        reason: "compact gadget consumer token missing",
                    });
                }
                let expected_rule = self
                    .compact_plan
                    .as_ref()
                    .and_then(|plan| plan.gadget_shells.get(&(shell, expression)))
                    .map(|entry| entry.rule.clone())
                    .ok_or(NormalizeError::InvalidExactPlan {
                        reason: "compact gadget consumer plan missing",
                    })?;
                if self.exact_compact_gadget_rule(shell, expression)? != expected_rule {
                    return Err(NormalizeError::InvalidExactPlan {
                        reason: "compact gadget rule identity changed",
                    });
                }
                if let Some(context) = self.compact_product_contexts.last_mut() {
                    context.used_shells.insert((shell, expression));
                    return Ok(Some(normal_form));
                }
                // A compact plan may be consumed only by the exact product boundary that
                // preflight authorized. A relation pass or standalone lookup has no such token.
                return Err(NormalizeError::InvalidExactPlan {
                    reason: "compact gadget lookup outside product boundary",
                });
            }
            return Ok(Some(normal_form));
        }
        if self.compact_mode {
            return Err(NormalizeError::InvalidExactPlan {
                reason: "compact gadget shell hold missing",
            });
        }
        let value = match self.child_value(expression) {
            Ok(value) => value,
            Err(NormalizeError::MissingCachedValue { .. }) => return Ok(None),
            Err(error) => return Err(error),
        };
        let Some(normal_form) = value.exact_nf.clone() else {
            return Ok(None);
        };
        self.insert_gadget_hold(expression, normal_form.clone(), None);
        Ok(Some(normal_form))
    }

    fn compact_product_nf(
        &mut self,
        scope_proof: &ScopeProof,
        left_type: &ResolvedMatrixType,
        right_type: &ResolvedMatrixType,
        left: &PolynomialNF,
        right: &PolynomialNF,
    ) -> Result<PolynomialNF, NormalizeError> {
        self.with_compact_product_context(|normalizer| {
            normalizer.product_nf(scope_proof, left_type, right_type, left, right)
        })
    }

    /// Evaluate one compact product boundary with shell-use tracking. The frame is always
    /// removed before propagating the evaluator result; shell plan/hold counts are committed only
    /// after a successful product, so an error cannot release a prefix or leave a stale context.
    fn with_compact_product_context<T>(
        &mut self,
        evaluate: impl FnOnce(&mut Self) -> Result<T, NormalizeError>,
    ) -> Result<T, NormalizeError> {
        self.compact_product_contexts.push(CompactProductContext::default());
        let result = evaluate(self);
        let context =
            self.compact_product_contexts.pop().ok_or(NormalizeError::InvalidExactPlan {
                reason: "compact product context disappeared",
            })?;
        let result = result?;
        self.commit_compact_product_context(context.used_shells)?;
        Ok(result)
    }

    fn commit_compact_product_context(
        &mut self,
        used_shells: BTreeSet<(ExprId, ExprId)>,
    ) -> Result<(), NormalizeError> {
        if used_shells.is_empty() {
            return Ok(());
        }
        // Validate every token and plan count before mutating either side.  This keeps a
        // malformed boundary fail-closed instead of releasing only a prefix of its shells.
        for (shell, input) in &used_shells {
            let available = self
                .gadget_input_nfs
                .get(input)
                .and_then(|hold| hold.consumers.get(shell))
                .copied()
                .unwrap_or(0);
            if available == 0 {
                return Err(NormalizeError::InvalidExactPlan {
                    reason: "compact gadget consumer token over-consumed",
                });
            }
            let planned = self
                .compact_plan_remaining
                .as_ref()
                .and_then(|plan| plan.gadget_shells.get(&(*shell, *input)))
                .map(|entry| entry.occurrences)
                .unwrap_or(0);
            if planned == 0 {
                return Err(NormalizeError::InvalidExactPlan {
                    reason: "compact gadget plan occurrence over-consumed",
                });
            }
        }
        for (shell, input) in used_shells {
            self.consume_compact_shell_plan(shell, input)?;
            self.release_compact_gadget_hold(shell, input)?;
        }
        Ok(())
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
                self.rewrite_relations(normal_form)?;
                value.coefficient_bound = self.bound_normal_form(normal_form)?;
            }
        }
        if let Some(normal_form) = value.exact_nf.as_mut().and_then(Arc::get_mut) {
            if normal_form.is_zero() {
                value.coefficient_bound = NumericContract::Known(CoefficientBound::ExactZero);
                normal_form.bounded_summary = BoundedSummary::zero();
            }
        }
        Ok(value)
    }

    /// Normalize a root selected by the lowering preflight.  Reducible generated calls are
    /// interpreted as a compact algebraic island; concrete leaves are converted to atoms only at
    /// their own boundary.  The work stack is explicit so a 20k-node generated chain cannot
    /// overflow the Rust call stack.
    pub(crate) fn normalize_compact_root(
        &mut self,
        root: ScopedExprId,
    ) -> Result<AnalyzedValue, NormalizeError> {
        self.normalize_compact_source(CompactRootSource::Closed(root))
    }

    pub(crate) fn normalize_compact_family_root(
        &mut self,
        family: FamilyValueId,
    ) -> Result<AnalyzedValue, NormalizeError> {
        self.normalize_compact_source(CompactRootSource::Family(family))
    }

    fn normalize_compact_source(
        &mut self,
        source: CompactRootSource,
    ) -> Result<AnalyzedValue, NormalizeError> {
        self.compact_progress_source = match source {
            CompactRootSource::Closed(_) => CompactProgressSource::Closed,
            CompactRootSource::Family(_) => CompactProgressSource::Family,
        };
        let (mut proof, body, bindings, result_semantic) = match source {
            CompactRootSource::Closed(root) => {
                if root.program() != self.scope {
                    return Err(NormalizeError::InvalidScope {
                        expected: self.scope,
                        actual: root.program(),
                    });
                }
                let proof = self.expressions.scope_proof(root.program(), root.expression())?;
                let mut root_expression = root.expression();
                let mut root_node = self.expressions.node(root_expression)?.clone();
                let ValueOperator::ProgramCall { mut program } = root_node.operator else {
                    // A finalized zero-argument wrapper may canonicalize an already-concrete root.
                    // The lowering marker is conservative, so retaining the
                    // ordinary evaluator here is semantically safe and
                    // preserves the eager-path error ordering.
                    return self.normalize(root);
                };
                // `normalize_compact_closed_root` wraps a closed expression in a private
                // zero-argument program.  Unwrap that transport shell while
                // retaining the shell's scope proof; the selected residual itself
                // is the inner generated family call.
                if self.programs.family_for_program(program).is_none() {
                    let wrapper_signature = self.programs.program_signature(program)?;
                    if !wrapper_signature.inputs.is_empty() {
                        return self.normalize(root);
                    }
                    root_expression = self.programs.program(program)?.root;
                    root_node = self.expressions.node(root_expression)?.clone();
                    let ValueOperator::ProgramCall { program: inner } = root_node.operator else {
                        return self.normalize(root);
                    };
                    program = inner;
                }
                let family = self.programs.family_for_program(program).ok_or(
                    NormalizeError::InvalidExactPlan { reason: "compact root family authority" },
                )?;
                if !self.programs.family_is_reducible(family)? {
                    return Err(NormalizeError::InvalidExactPlan {
                        reason: "compact root is opaque",
                    });
                }
                let signature = self.programs.program_signature(program)?;
                if root_node.inputs.len() != signature.inputs.len() {
                    return Err(NormalizeError::InvalidExactPlan { reason: "compact root arity" });
                }
                let mut bindings = Vec::with_capacity(root_node.inputs.len());
                for (input, expected) in root_node.inputs.iter().zip(signature.inputs.iter()) {
                    if self.expressions.value_type(*input)? != &expected.value_type {
                        return Err(NormalizeError::InvalidExactPlan {
                            reason: "compact root input type",
                        });
                    }
                    if !self.expressions.is_closed(*input)? {
                        return Err(NormalizeError::InvalidExactPlan {
                            reason: "compact root input is open",
                        });
                    }
                    bindings.push(self.expressions.scoped_from_proof(&proof, *input)?);
                }
                let body = self.programs.family_body(family)?;
                if self.expressions.value_type(body)? != &signature.output {
                    return Err(NormalizeError::InvalidExactPlan { reason: "compact root output" });
                }
                (proof, body, bindings, root)
            }
            CompactRootSource::Family(family) => {
                if family.program() != self.scope || !self.programs.family_is_reducible(family)? {
                    return Err(NormalizeError::InvalidExactPlan {
                        reason: "compact family authority",
                    });
                }
                let domain = self.programs.family_domain(family)?;
                let signature = self.programs.program_signature(family.program())?.clone();
                let element_type = self.programs.family_element_type(family)?;
                if signature.inputs.len() != 1 ||
                    signature.inputs[0].value_type != ResolvedValueType::Int ||
                    signature.inputs[0].trusted_index_range !=
                        Some(super::arena::TrustedIndexRange {
                            minimum: domain.minimum,
                            maximum_exclusive: domain.maximum_exclusive,
                        }) ||
                    signature.output != element_type
                {
                    return Err(NormalizeError::InvalidExactPlan {
                        reason: "compact family signature",
                    });
                }
                let body = self.programs.family_body(family)?;
                if self.expressions.value_type(body)? != &signature.output {
                    return Err(NormalizeError::InvalidExactPlan {
                        reason: "compact family output",
                    });
                }
                let free_arguments = self.expressions.free_arguments(body)?;
                if free_arguments.iter().any(|(position, value_type)| {
                    *position != 0 || *value_type != ResolvedValueType::Int
                }) {
                    return Err(NormalizeError::InvalidExactPlan {
                        reason: "compact family formal argument",
                    });
                }
                let proof = self.expressions.scope_proof(family.program(), body)?;
                let argument = self.expressions.intern_argument(0, ResolvedValueType::Int)?;
                let bindings = if free_arguments.contains(&(0, ResolvedValueType::Int)) {
                    vec![self.expressions.scoped_from_proof(&proof, argument)?]
                } else {
                    Vec::new()
                };
                let result_semantic = self.expressions.scoped_from_proof(&proof, body)?;
                (proof, body, bindings, result_semantic)
            }
        };
        self.counters.compact_virtual_calls = self.counters.compact_virtual_calls.saturating_add(1);
        let mut frames = vec![CompactFrame::Visit { expression: body, bindings }];
        let mut values = Vec::<CompactValue>::new();
        let mut max_frames = frames.len() as u64;
        self.gadget_input_nfs.clear();
        self.compact_scalar_holds.clear();
        self.retained_bounded_scalar_factors.clear();
        self.compact_plan_remaining = self.compact_plan.clone();
        if let Some(plan) = self.compact_plan.as_ref() {
            self.counters.compact_planned_shell_occurrences = plan.gadget_occurrences();
            self.counters.compact_planned_unique_shells = plan.gadget_shells.len() as u64;
            self.counters.compact_shell_allocated = plan.shell_allocated;
            self.counters.compact_shell_hits = plan.shell_hits;
            self.counters.compact_shell_new = plan.shell_new;
            self.counters.compact_scalar_consumers = plan.scalar_occurrences();
        }
        self.compact_mode = true;
        self.counters.compact_shell_holds_current = 0;
        self.counters.compact_scalar_holds_current = 0;
        while let Some(frame) = frames.pop() {
            match frame {
                CompactFrame::Visit { expression, bindings } => {
                    self.counters.nodes_processed = self.counters.nodes_processed.saturating_add(1);
                    self.log_compact_progress();
                    let node = self.expressions.node(expression)?.clone();
                    // Closed concrete leaves already have their canonical identity.  Preserve
                    // that identity directly; only matrix leaves cross the atom boundary.  An
                    // open transform/call is handled below after its arguments are substituted.
                    let virtual_node = matches!(
                        node.operator,
                        ValueOperator::Matrix(
                            MatrixOperation::Add |
                                MatrixOperation::Subtract |
                                MatrixOperation::Negate |
                                MatrixOperation::Scale |
                                MatrixOperation::Multiply |
                                MatrixOperation::Tensor { .. }
                        )
                    ) || matches!(
                        node.operator,
                        ValueOperator::ProgramCall { program }
                            if self
                                .programs
                                .family_for_program(program)
                                .map(|family| self.programs.family_is_reducible(family))
                                .transpose()?
                                .unwrap_or(false)
                    );
                    if !matches!(node.operator, ValueOperator::Argument { .. }) &&
                        !virtual_node &&
                        node.inputs.is_empty()
                    {
                        if let Ok(semantic) = self.expressions.scoped_from_proof(&proof, expression)
                        {
                            let coefficient_bound = if matches!(
                                node.operator,
                                ValueOperator::Constant(super::arena::TypedConstant {
                                    value: super::arena::ConstantValue::Int(_),
                                    ..
                                })
                            ) {
                                self.nonmatrix_bound(expression, &node, &[])?
                            } else {
                                self.factor_bound(expression)?
                            };
                            let resolved_type = self.expressions.value_type(expression)?.clone();
                            let hold = self.retain_compact_scalar_hold(
                                expression,
                                &resolved_type,
                                &coefficient_bound,
                            );
                            let exact = if matches!(
                                self.expressions.value_type(semantic.expression())?,
                                ResolvedValueType::Matrix(_)
                            ) {
                                Some(Arc::new(self.atom_nf(&proof, semantic)?))
                            } else {
                                None
                            };
                            values.push(CompactValue {
                                semantic: Some(semantic),
                                exact_nf: exact,
                                coefficient_bound,
                                resolved_type,
                                hold,
                            });
                            continue;
                        }
                    }
                    match node.operator {
                        ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
                            ..
                        }) => {
                            let planned = node.inputs.first().is_some_and(|input| {
                                self.compact_plan.as_ref().is_some_and(|plan| {
                                    plan.gadget_shells.contains_key(&(expression, *input))
                                })
                            });
                            frames.push(if planned {
                                CompactFrame::CombineGadget {
                                    expression,
                                    node: Arc::new(node.clone()),
                                }
                            } else {
                                CompactFrame::CombineConcrete {
                                    expression,
                                    node: Arc::new(node.clone()),
                                }
                            });
                            for input in node.inputs.iter().rev() {
                                frames.push(CompactFrame::Visit {
                                    expression: *input,
                                    bindings: bindings.clone(),
                                });
                            }
                        }
                        ValueOperator::Argument { position, value_type } => {
                            let semantic = *bindings.get(position as usize).ok_or(
                                NormalizeError::InvalidExactPlan {
                                    reason: "compact argument binding",
                                },
                            )?;
                            if self.expressions.value_type(semantic.expression())? != &value_type {
                                return Err(NormalizeError::InvalidExactPlan {
                                    reason: "compact argument type",
                                });
                            }
                            let exact = if matches!(value_type, ResolvedValueType::Matrix(_)) {
                                Some(Arc::new(self.atom_nf(&proof, semantic)?))
                            } else {
                                None
                            };
                            let coefficient_bound = if exact.is_some() {
                                self.authoritative_source_bound(semantic.expression())?
                            } else if matches!(value_type, ResolvedValueType::Int) &&
                                self.direct_integer_constant(semantic.expression()).is_some()
                            {
                                self.nonmatrix_bound(
                                    semantic.expression(),
                                    self.expressions.node(semantic.expression())?,
                                    &[],
                                )?
                            } else {
                                NumericContract::Missing
                            };
                            let hold = self.retain_compact_scalar_hold(
                                semantic.expression(),
                                &value_type,
                                &coefficient_bound,
                            );
                            values.push(CompactValue {
                                semantic: Some(semantic),
                                exact_nf: exact,
                                coefficient_bound,
                                resolved_type: value_type,
                                hold,
                            });
                        }
                        ValueOperator::ProgramCall { program } => {
                            let family = self.programs.family_for_program(program);
                            let reducible = family
                                .map(|family| self.programs.family_is_reducible(family))
                                .transpose()?
                                .unwrap_or(false);
                            if reducible {
                                let mut nested = Vec::with_capacity(node.inputs.len());
                                for input in node.inputs.iter().copied() {
                                    nested.push(self.compact_semantic_iterative(
                                        &mut proof, input, &bindings,
                                    )?);
                                }
                                let family = family.ok_or(NormalizeError::InvalidExactPlan {
                                    reason: "compact call family authority",
                                })?;
                                frames.push(CompactFrame::Visit {
                                    expression: self.programs.family_body(family)?,
                                    bindings: nested,
                                });
                                self.counters.compact_virtual_calls =
                                    self.counters.compact_virtual_calls.saturating_add(1);
                                continue;
                            }
                            frames.push(CompactFrame::CombineCall {
                                expression,
                                node: Arc::new(node.clone()),
                            });
                            for input in node.inputs.iter().rev() {
                                frames.push(CompactFrame::Visit {
                                    expression: *input,
                                    bindings: bindings.clone(),
                                });
                            }
                        }
                        ValueOperator::Matrix(
                            MatrixOperation::Add |
                            MatrixOperation::Subtract |
                            MatrixOperation::Negate |
                            MatrixOperation::Scale |
                            MatrixOperation::Multiply |
                            MatrixOperation::Tensor { .. },
                        ) => {
                            self.counters.compact_algebra_nodes =
                                self.counters.compact_algebra_nodes.saturating_add(1);
                            match node.operator {
                                ValueOperator::Matrix(
                                    MatrixOperation::Add | MatrixOperation::Subtract,
                                ) => {
                                    self.counters.compact_logical_add_sub =
                                        self.counters.compact_logical_add_sub.saturating_add(1)
                                }
                                ValueOperator::Matrix(MatrixOperation::Scale) => {
                                    self.counters.compact_logical_scale =
                                        self.counters.compact_logical_scale.saturating_add(1)
                                }
                                ValueOperator::Matrix(MatrixOperation::Multiply) => {
                                    self.counters.compact_strict_products =
                                        self.counters.compact_strict_products.saturating_add(1)
                                }
                                _ => {}
                            }
                            frames.push(CompactFrame::CombineMatrix {
                                expression,
                                node: Arc::new(node.clone()),
                            });
                            for input in node.inputs.iter().rev() {
                                frames.push(CompactFrame::Visit {
                                    expression: *input,
                                    bindings: bindings.clone(),
                                });
                            }
                        }
                        _ => {
                            frames.push(CompactFrame::CombineConcrete {
                                expression,
                                node: Arc::new(node.clone()),
                            });
                            for input in node.inputs.iter().rev() {
                                frames.push(CompactFrame::Visit {
                                    expression: *input,
                                    bindings: bindings.clone(),
                                });
                            }
                        }
                    }
                }
                CompactFrame::CombineMatrix { expression, node } => {
                    let count = node.inputs.len();
                    if values.len() < count {
                        return Err(NormalizeError::InvalidExactPlan {
                            reason: "compact value stack",
                        });
                    }
                    let start = values.len() - count;
                    let children = values.split_off(start);
                    let child_holds =
                        children.iter().filter_map(|child| child.hold).collect::<Vec<_>>();
                    let bounds = children
                        .iter()
                        .map(|child| child.coefficient_bound.clone())
                        .collect::<Vec<_>>();
                    let bound = self.matrix_operation_bound(
                        match &node.operator {
                            ValueOperator::Matrix(operation) => operation,
                            _ => unreachable!(),
                        },
                        &node,
                        &bounds,
                    )?;
                    let exact = match node.operator {
                        ValueOperator::Matrix(MatrixOperation::Add) => Some(self.add_nf(
                            children[0].exact_nf.as_deref().ok_or(
                                NormalizeError::InvalidExactPlan { reason: "compact add child" },
                            )?,
                            children[1].exact_nf.as_deref().ok_or(
                                NormalizeError::InvalidExactPlan { reason: "compact add child" },
                            )?,
                            false,
                        )?),
                        ValueOperator::Matrix(MatrixOperation::Subtract) => Some(self.add_nf(
                            children[0].exact_nf.as_deref().ok_or(
                                NormalizeError::InvalidExactPlan {
                                    reason: "compact subtract child",
                                },
                            )?,
                            children[1].exact_nf.as_deref().ok_or(
                                NormalizeError::InvalidExactPlan {
                                    reason: "compact subtract child",
                                },
                            )?,
                            true,
                        )?),
                        ValueOperator::Matrix(MatrixOperation::Negate) => {
                            Some(self.negate_nf(children[0].exact_nf.as_deref().ok_or(
                                NormalizeError::InvalidExactPlan { reason: "compact negate child" },
                            )?))
                        }
                        ValueOperator::Matrix(MatrixOperation::Scale) => {
                            let scale = self.direct_integer_constant(node.inputs[1]).ok_or(
                                NormalizeError::InvalidExactPlan {
                                    reason: "compact scale coefficient",
                                },
                            )?;
                            Some(self.scale_nf(
                                children[0].exact_nf.as_deref().ok_or(
                                    NormalizeError::InvalidExactPlan {
                                        reason: "compact scale child",
                                    },
                                )?,
                                &scale,
                            ))
                        }
                        ValueOperator::Matrix(MatrixOperation::Multiply) => {
                            let ResolvedValueType::Matrix(left_type) =
                                self.expressions.value_type(node.inputs[0])?
                            else {
                                return Err(NormalizeError::InvalidExactPlan {
                                    reason: "compact product left type",
                                });
                            };
                            let ResolvedValueType::Matrix(right_type) =
                                self.expressions.value_type(node.inputs[1])?
                            else {
                                return Err(NormalizeError::InvalidExactPlan {
                                    reason: "compact product right type",
                                });
                            };
                            let left_type = left_type.clone();
                            let right_type = right_type.clone();
                            let left_nf = children[0].exact_nf.as_deref().ok_or(
                                NormalizeError::InvalidExactPlan { reason: "compact product left" },
                            )?;
                            let right_nf = children[1].exact_nf.as_deref().ok_or(
                                NormalizeError::InvalidExactPlan {
                                    reason: "compact product right",
                                },
                            )?;
                            let left_scalar = left_type.rows == 1 && left_type.columns == 1;
                            let right_scalar = right_type.rows == 1 && right_type.columns == 1;
                            let one_sided_scalar = left_scalar ^ right_scalar;
                            let both_scalar = left_scalar && right_scalar;
                            if one_sided_scalar || both_scalar {
                                let output_type = match self.expressions.value_type(expression)? {
                                    ResolvedValueType::Matrix(output_type) => output_type.clone(),
                                    _ => {
                                        return Err(NormalizeError::InvalidExactPlan {
                                            reason: "compact scalar action output type",
                                        })
                                    }
                                };
                                let left_bound = children[0].coefficient_bound.clone();
                                let right_bound = children[1].coefficient_bound.clone();
                                let mut value = match self.scalar_action_nf_typed(
                                    &proof,
                                    &output_type,
                                    &left_type,
                                    &right_type,
                                    &left_bound,
                                    &right_bound,
                                    left_nf,
                                    right_nf,
                                )? {
                                    ScalarActionNormalization::Exact(value) => Some(value),
                                    _ => {
                                        return Err(NormalizeError::InvalidExactPlan {
                                            reason: "compact scalar action was not exact",
                                        })
                                    }
                                };
                                // A fully finite scalar action follows the eager bounded-only
                                // contract: retain its summary but do not publish an exact atom.
                                // The authorized scalar token is consumed only after the typed
                                // action above returns Exact.
                                if matches!(
                                    bound,
                                    NumericContract::Known(
                                        CoefficientBound::ExactZero | CoefficientBound::Finite(_)
                                    )
                                ) {
                                    value = Some(PolynomialNF {
                                        exact_terms: BTreeMap::new(),
                                        bounded_summary: BoundedSummary::from_contract(
                                            bound.clone(),
                                        )?,
                                    });
                                }
                                if one_sided_scalar {
                                    let (scalar_expression, scalar_type) = if left_scalar {
                                        (node.inputs[0], &left_type)
                                    } else {
                                        (node.inputs[1], &right_type)
                                    };
                                    let scalar_is_right = right_scalar;
                                    if matches!(
                                        self.expressions.node(scalar_expression)?.operator,
                                        ValueOperator::ProgramCall { .. }
                                    ) {
                                        self.consume_compact_scalar_program_call(
                                            expression,
                                            scalar_expression,
                                            scalar_is_right,
                                            scalar_type,
                                        )?;
                                    } else {
                                        self.consume_compact_scalar_plan(
                                            scalar_expression,
                                            scalar_type,
                                        )?;
                                    }
                                }
                                value
                            } else {
                                Some(self.compact_product_nf(
                                    &proof,
                                    &left_type,
                                    &right_type,
                                    left_nf,
                                    right_nf,
                                )?)
                            }
                        }
                        ValueOperator::Matrix(MatrixOperation::Tensor { .. }) => {
                            let ResolvedValueType::Matrix(left_type) =
                                self.expressions.value_type(node.inputs[0])?
                            else {
                                return Err(NormalizeError::InvalidExactPlan {
                                    reason: "compact tensor left type",
                                });
                            };
                            let ResolvedValueType::Matrix(right_type) =
                                self.expressions.value_type(node.inputs[1])?
                            else {
                                return Err(NormalizeError::InvalidExactPlan {
                                    reason: "compact tensor right type",
                                });
                            };
                            let left_type = left_type.clone();
                            let right_type = right_type.clone();
                            let left_nf = children[0].exact_nf.as_deref().ok_or(
                                NormalizeError::InvalidExactPlan { reason: "compact tensor left" },
                            )?;
                            let right_nf = children[1].exact_nf.as_deref().ok_or(
                                NormalizeError::InvalidExactPlan { reason: "compact tensor right" },
                            )?;
                            let mut value = match self.tensor_scalar_action_nf(
                                &proof,
                                match &node.operator {
                                    ValueOperator::Matrix(operation) => operation,
                                    _ => unreachable!(),
                                },
                                expression,
                                node.inputs[0],
                                node.inputs[1],
                                left_nf,
                                right_nf,
                            )? {
                                ScalarActionNormalization::Exact(value) => value,
                                _ => {
                                    return Err(NormalizeError::InvalidExactPlan {
                                        reason: "compact tensor scalar action was not exact",
                                    })
                                }
                            };
                            if matches!(
                                bound,
                                NumericContract::Known(
                                    CoefficientBound::ExactZero | CoefficientBound::Finite(_)
                                )
                            ) {
                                value = PolynomialNF {
                                    exact_terms: BTreeMap::new(),
                                    bounded_summary: BoundedSummary::from_contract(bound.clone())?,
                                };
                            }
                            let scalar_is_right = right_type.rows == 1 && right_type.columns == 1;
                            let scalar_expression =
                                if scalar_is_right { node.inputs[1] } else { node.inputs[0] };
                            if matches!(
                                self.expressions.node(scalar_expression)?.operator,
                                ValueOperator::ProgramCall { .. }
                            ) {
                                self.consume_compact_scalar_program_call(
                                    expression,
                                    scalar_expression,
                                    scalar_is_right,
                                    if scalar_is_right { &right_type } else { &left_type },
                                )?;
                            } else {
                                return Err(NormalizeError::InvalidExactPlan {
                                    reason: "compact tensor scalar factor is not planned",
                                });
                            }
                            Some(value)
                        }
                        _ => unreachable!(),
                    };
                    for hold in child_holds {
                        self.release_compact_scalar_hold(hold)?;
                    }
                    values.push(CompactValue {
                        // This value is an eliminable virtual algebra result.  It deliberately
                        // carries no semantic expression; the original root identity is restored
                        // only once, after the complete compact value has been produced.
                        semantic: None,
                        exact_nf: exact.map(Arc::new),
                        coefficient_bound: bound,
                        resolved_type: self.expressions.value_type(expression)?.clone(),
                        hold: None,
                    });
                }
                CompactFrame::CombineGadget { expression, node } => {
                    if node.inputs.len() != 1 || self.compact_plan.is_none() {
                        return Err(NormalizeError::InvalidExactPlan {
                            reason: "compact gadget frame is not preflighted",
                        });
                    }
                    let start =
                        values.len().checked_sub(1).ok_or(NormalizeError::InvalidExactPlan {
                            reason: "compact gadget stack",
                        })?;
                    let children = values.split_off(start);
                    let child =
                        children.into_iter().next().ok_or(NormalizeError::InvalidExactPlan {
                            reason: "compact gadget child",
                        })?;
                    let input = node.inputs[0];
                    let plan_entry = self
                        .compact_plan
                        .as_ref()
                        .and_then(|plan| plan.gadget_shells.get(&(expression, input)))
                        .ok_or(NormalizeError::InvalidExactPlan {
                            reason: "compact gadget shell was not preflighted",
                        })?;
                    let rule = self.exact_compact_gadget_rule(expression, input)?;
                    if rule != plan_entry.rule {
                        return Err(NormalizeError::InvalidExactPlan {
                            reason: "compact gadget rule identity changed",
                        });
                    }
                    let input_nf = child.exact_nf.ok_or(NormalizeError::InvalidExactPlan {
                        reason: "compact gadget input has no exact normal form",
                    })?;
                    let bound = child.coefficient_bound.clone();
                    if let Some(child_hold) = child.hold {
                        self.release_compact_scalar_hold(child_hold)?;
                    }
                    let semantic = self.expressions.intern_scoped_transform(
                        &mut proof,
                        node.operator.clone(),
                        &[child.semantic.ok_or(NormalizeError::InvalidExactPlan {
                            reason: "compact gadget input semantic missing",
                        })?],
                    )?;
                    self.insert_gadget_hold(input, input_nf, Some(expression));
                    let resolved_type = self.expressions.value_type(expression)?.clone();
                    let exact_nf = Some(Arc::new(self.atom_nf(&proof, semantic)?));
                    values.push(CompactValue {
                        semantic: Some(semantic),
                        exact_nf,
                        coefficient_bound: bound,
                        resolved_type,
                        hold: None,
                    });
                }
                CompactFrame::CombineCall { expression, node } => {
                    let count = node.inputs.len();
                    if values.len() < count {
                        return Err(NormalizeError::InvalidExactPlan {
                            reason: "compact call stack",
                        });
                    }
                    let start = values.len() - count;
                    let children = values.split_off(start);
                    let child_holds =
                        children.iter().filter_map(|child| child.hold).collect::<Vec<_>>();
                    let semantics = children
                        .iter()
                        .map(|child| child.semantic)
                        .collect::<Option<Vec<_>>>()
                        .ok_or(NormalizeError::InvalidExactPlan {
                            reason: "eliminable virtual call input",
                        })?;
                    let semantic = self.expressions.intern_scoped_transform(
                        &mut proof,
                        node.operator.clone(),
                        &semantics,
                    )?;
                    let child_values = children
                        .iter()
                        .map(CompactValue::as_analyzed)
                        .map(|value| value.map(Arc::new))
                        .collect::<Result<Vec<_>, _>>()?;
                    let mut value = if matches!(
                        self.expressions.value_type(semantic.expression())?,
                        ResolvedValueType::Matrix(_)
                    ) {
                        self.evaluate_matrix(
                            &mut proof,
                            semantic,
                            expression,
                            &node,
                            &child_values,
                        )?
                    } else {
                        self.evaluate_nonmatrix(semantic, expression, &node, &child_values)?
                    };
                    // Generated body facts may remain `Missing` even when this compact
                    // traversal has exact finite child transfers locally. Derive the operation
                    // bound from those children without a second graph traversal.
                    if value.coefficient_bound.is_missing() {
                        if let ValueOperator::Matrix(operation) = &node.operator {
                            let child_bounds = child_values
                                .iter()
                                .map(|child| child.coefficient_bound.clone())
                                .collect::<Vec<_>>();
                            value.coefficient_bound =
                                self.matrix_operation_bound(operation, &node, &child_bounds)?;
                        }
                    }
                    let resolved_type = self.expressions.value_type(expression)?.clone();
                    let hold = self.retain_compact_scalar_hold(
                        value.semantic.expression(),
                        &resolved_type,
                        &value.coefficient_bound,
                    );
                    let mut compact_value = CompactValue::from_analyzed(value, resolved_type);
                    compact_value.hold = hold;
                    for child_hold in child_holds {
                        self.release_compact_scalar_hold(child_hold)?;
                    }
                    values.push(compact_value);
                }
                CompactFrame::CombineConcrete { expression, node } => {
                    self.counters.compact_concrete_shell_nodes =
                        self.counters.compact_concrete_shell_nodes.saturating_add(1);
                    let count = node.inputs.len();
                    if values.len() < count {
                        return Err(NormalizeError::InvalidExactPlan {
                            reason: "compact concrete stack",
                        });
                    }
                    let start = values.len() - count;
                    let children = values.split_off(start);
                    let child_holds =
                        children.iter().filter_map(|child| child.hold).collect::<Vec<_>>();
                    let semantics = children
                        .iter()
                        .map(|child| child.semantic)
                        .collect::<Option<Vec<_>>>()
                        .ok_or(NormalizeError::InvalidExactPlan {
                            reason: "eliminable virtual concrete input",
                        })?;
                    let semantic = self.expressions.intern_scoped_transform(
                        &mut proof,
                        node.operator.clone(),
                        &semantics,
                    )?;
                    let child_values = children
                        .iter()
                        .map(CompactValue::as_analyzed)
                        .map(|value| value.map(Arc::new))
                        .collect::<Result<Vec<_>, _>>()?;
                    let value = if matches!(
                        self.expressions.value_type(semantic.expression())?,
                        ResolvedValueType::Matrix(_)
                    ) {
                        self.evaluate_matrix(
                            &mut proof,
                            semantic,
                            expression,
                            &node,
                            &child_values,
                        )?
                    } else {
                        self.evaluate_nonmatrix(semantic, expression, &node, &child_values)?
                    };
                    let resolved_type = self.expressions.value_type(expression)?.clone();
                    let hold = self.retain_compact_scalar_hold(
                        value.semantic.expression(),
                        &resolved_type,
                        &value.coefficient_bound,
                    );
                    let mut compact_value = CompactValue::from_analyzed(value, resolved_type);
                    compact_value.hold = hold;
                    for child_hold in child_holds {
                        self.release_compact_scalar_hold(child_hold)?;
                    }
                    values.push(compact_value);
                }
            }
            let live_frames = frames.len() as u64;
            let live_values = values.len() as u64;
            self.counters.compact_live_frames = live_frames;
            self.counters.compact_live_values = live_values;
            self.counters.compact_peak_live_frames =
                self.counters.compact_peak_live_frames.max(live_frames);
            self.counters.compact_peak_live_values =
                self.counters.compact_peak_live_values.max(live_values);
            max_frames = max_frames.max(live_frames);
            self.counters.compact_max_virtual_frames =
                self.counters.compact_max_virtual_frames.max(live_frames);
            self.counters.compact_max_virtual_values =
                self.counters.compact_max_virtual_values.max(live_values);
        }
        self.counters.compact_max_frames = max_frames;
        let result_value = values
            .pop()
            .ok_or(NormalizeError::InvalidExactPlan { reason: "compact root value" })?;
        let mut result = AnalyzedValue {
            semantic: result_semantic,
            exact_nf: result_value.exact_nf,
            coefficient_bound: result_value.coefficient_bound,
        };
        self.counters.compact_live_frames = 0;
        self.counters.compact_live_values = 0;
        result.semantic = result_semantic;
        if let Some(exact_nf) = result.exact_nf.as_mut() {
            if self.relations.is_some() {
                let normal_form = Arc::make_mut(exact_nf);
                self.rewrite_relations(normal_form)?;
                result.coefficient_bound = self.bound_normal_form(normal_form)?;
                if self.fold_final_no_match {
                    self.fold_finite_no_match_terms(normal_form, false)?;
                }
                if normal_form.is_zero() {
                    result.coefficient_bound = NumericContract::Known(CoefficientBound::ExactZero);
                    normal_form.bounded_summary = BoundedSummary::zero();
                }
            }
            self.counters.relation_remaining = self.count_relation_remaining(exact_nf) as u64;
            self.counters.final_exact_term_count = exact_nf.exact_terms.len() as u64;
        }
        let (gadget_plan_unmatched, scalar_plan_unmatched) = self
            .compact_plan_remaining
            .as_ref()
            .map(|plan| {
                (
                    plan.gadget_shells.values().map(|entry| entry.occurrences).sum::<u64>(),
                    plan.scalar_occurrences(),
                )
            })
            .unwrap_or((0, 0));
        let shell_holds = self.gadget_input_nfs.values().map(|hold| hold.remaining).sum::<u64>();
        let scalar_holds = self.compact_scalar_holds.values().copied().sum::<u64>();
        self.counters.compact_shell_holds_unmatched =
            gadget_plan_unmatched.saturating_add(shell_holds);
        self.counters.compact_scalar_holds_unmatched =
            scalar_plan_unmatched.saturating_add(scalar_holds);
        if gadget_plan_unmatched != 0 ||
            scalar_plan_unmatched != 0 ||
            shell_holds != 0 ||
            scalar_holds != 0
        {
            self.compact_mode = false;
            return Err(NormalizeError::InvalidExactPlan {
                reason: "compact plan or hold was not fully consumed",
            });
        }
        // Compact runtime work is the actual evaluator Visit count. The structural preflight
        // occurrence count remains private plan diagnostics and is deliberately not equated with
        // this completed runtime total.
        self.counters.nodes_total = self.counters.nodes_processed;
        self.compact_plan_remaining = None;
        self.compact_mode = false;
        Ok(result)
    }

    /// Build the semantic identity of a call's ordered binding with a local environment stack.
    /// Reducible calls
    /// are beta-entered here, so nested index bindings have the same scoped identity as eager
    /// evaluation without a bare-expression cache or domain materialization.
    fn compact_semantic_iterative(
        &mut self,
        proof: &mut ScopeProof,
        expression: ExprId,
        bindings: &[ScopedExprId],
    ) -> Result<ScopedExprId, NormalizeError> {
        enum Frame {
            Visit {
                expression: ExprId,
                environment: usize,
            },
            Build {
                expression: ExprId,
                environment: usize,
            },
            EnterCall {
                expression: ExprId,
                environment: usize,
                body: ExprId,
                body_environment: usize,
            },
            FinishCall {
                expression: ExprId,
                environment: usize,
                body: ExprId,
                body_environment: usize,
            },
        }

        let mut environments = vec![bindings.to_vec()];
        let mut values = BTreeMap::<(ExprId, usize), ScopedExprId>::new();
        let mut active = BTreeSet::<(ExprId, usize)>::new();
        let mut frames = vec![Frame::Visit { expression, environment: 0 }];
        while let Some(frame) = frames.pop() {
            match frame {
                Frame::Visit { expression, environment } => {
                    self.counters.nodes_processed = self.counters.nodes_processed.saturating_add(1);
                    self.log_compact_progress();
                    let key = (expression, environment);
                    if values.contains_key(&key) {
                        continue;
                    }
                    if !active.insert(key) {
                        return Err(NormalizeError::InvalidExactPlan {
                            reason: "compact semantic binding cycle",
                        });
                    }
                    let node = self.expressions.node(expression)?.clone();
                    if let ValueOperator::Argument { position, .. } = node.operator {
                        let value = *environments
                            .get(environment)
                            .and_then(|bindings| bindings.get(position as usize))
                            .ok_or(NormalizeError::InvalidExactPlan {
                                reason: "compact argument binding",
                            })?;
                        values.insert(key, value);
                        active.remove(&key);
                        continue;
                    }
                    if let ValueOperator::ProgramCall { program } = node.operator {
                        let family = self.programs.family_for_program(program);
                        let reducible = family
                            .map(|family| self.programs.family_is_reducible(family))
                            .transpose()?
                            .unwrap_or(false);
                        if reducible {
                            let family = family.ok_or(NormalizeError::InvalidExactPlan {
                                reason: "compact call family authority",
                            })?;
                            let signature = self.programs.program_signature(program)?.clone();
                            if node.inputs.len() != signature.inputs.len() {
                                return Err(NormalizeError::InvalidExactPlan {
                                    reason: "compact call arity",
                                });
                            }
                            for (input, expected) in node.inputs.iter().zip(signature.inputs.iter())
                            {
                                if self.expressions.value_type(*input)? != &expected.value_type {
                                    return Err(NormalizeError::InvalidExactPlan {
                                        reason: "compact call input type",
                                    });
                                }
                            }
                            let body = self.programs.family_body(family)?;
                            if self.expressions.value_type(body)? != &signature.output {
                                return Err(NormalizeError::InvalidExactPlan {
                                    reason: "compact call output type",
                                });
                            }
                            let body_environment = environments.len();
                            environments.push(Vec::with_capacity(node.inputs.len()));
                            frames.push(Frame::EnterCall {
                                expression,
                                environment,
                                body,
                                body_environment,
                            });
                            for input in node.inputs.iter().rev() {
                                frames.push(Frame::Visit { expression: *input, environment });
                            }
                            continue;
                        }
                    }
                    if let Ok(value) = self.expressions.scoped_from_proof(proof, expression) {
                        values.insert(key, value);
                        active.remove(&key);
                        continue;
                    }
                    frames.push(Frame::Build { expression, environment });
                    for input in node.inputs.iter().rev() {
                        frames.push(Frame::Visit { expression: *input, environment });
                    }
                }
                Frame::Build { expression, environment } => {
                    let key = (expression, environment);
                    let node = self.expressions.node(expression)?.clone();
                    let inputs = node
                        .inputs
                        .iter()
                        .map(|input| {
                            values.get(&(*input, environment)).copied().ok_or(
                                NormalizeError::InvalidExactPlan {
                                    reason: "compact semantic binding",
                                },
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let value =
                        self.expressions.intern_scoped_transform(proof, node.operator, &inputs)?;
                    values.insert(key, value);
                    active.remove(&key);
                }
                Frame::EnterCall { expression, environment, body, body_environment } => {
                    let call_node = self.expressions.node(expression)?.clone();
                    let inputs = call_node
                        .inputs
                        .iter()
                        .map(|input| {
                            values.get(&(*input, environment)).copied().ok_or(
                                NormalizeError::InvalidExactPlan {
                                    reason: "compact semantic call input",
                                },
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    environments[body_environment] = inputs;
                    frames.push(Frame::FinishCall {
                        expression,
                        environment,
                        body,
                        body_environment,
                    });
                    frames.push(Frame::Visit { expression: body, environment: body_environment });
                }
                Frame::FinishCall { expression, environment, body, body_environment } => {
                    let body_value = values.get(&(body, body_environment)).copied().ok_or(
                        NormalizeError::InvalidExactPlan { reason: "compact semantic call body" },
                    )?;
                    values.insert((expression, environment), body_value);
                    active.remove(&(expression, environment));
                }
            }
        }
        values
            .get(&(expression, 0))
            .copied()
            .ok_or(NormalizeError::InvalidExactPlan { reason: "compact semantic root" })
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
                    .map_or_else(
                        || self.atom_nf(scope_proof, semantic),
                        |_| Ok(PolynomialNF::zero()),
                    )?,
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
            ValueOperator::Trapdoor(_) => Some(self.atom_nf(scope_proof, semantic)?),
            _ => Some(self.atom_nf(scope_proof, semantic)?),
        };
        Ok(AnalyzedValue { semantic, exact_nf: exact.map(Arc::new), coefficient_bound: bound })
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
                    _ => Ok(self.atom_nf(scope_proof, semantic)?),
                }
            }
            MatrixOperation::Negate => {
                if let Some(value) = children.first().and_then(|value| value.exact_nf.as_ref()) {
                    Ok(self.negate_nf(value))
                } else {
                    Ok(self.atom_nf(scope_proof, semantic)?)
                }
            }
            MatrixOperation::Scale => {
                let scalar = node.inputs.get(1).copied().and_then(|id| self.integer_constant(id));
                if let (Some(scale), Some(value)) =
                    (scalar, children.first().and_then(|value| value.exact_nf.as_ref()))
                {
                    Ok(self.scale_nf(value, &scale))
                } else {
                    Ok(self.atom_nf(scope_proof, semantic)?)
                }
            }
            MatrixOperation::Multiply => {
                let left = children.first().and_then(|value| value.exact_nf.as_ref());
                let right = children.get(1).and_then(|value| value.exact_nf.as_ref());
                match (left, right) {
                    (Some(left), Some(right)) => match self.scalar_action_nf(
                        scope_proof,
                        semantic.expression(),
                        node.inputs[0],
                        node.inputs[1],
                        left,
                        right,
                    )? {
                        ScalarActionNormalization::Exact(normal_form) => Ok(normal_form),
                        ScalarActionNormalization::Opaque => self.atom_nf(scope_proof, semantic),
                        ScalarActionNormalization::NotApplicable => {
                            let ResolvedValueType::Matrix(left_type) =
                                self.expressions.value_type(node.inputs[0])?
                            else {
                                return Ok(self.atom_nf(scope_proof, semantic)?);
                            };
                            let left_type = left_type.clone();
                            let ResolvedValueType::Matrix(right_type) =
                                self.expressions.value_type(node.inputs[1])?
                            else {
                                return Ok(self.atom_nf(scope_proof, semantic)?);
                            };
                            let right_type = right_type.clone();
                            self.product_nf(scope_proof, &left_type, &right_type, left, right)
                        }
                    },
                    _ => Ok(self.atom_nf(scope_proof, semantic)?),
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
                    return Ok(self.atom_nf(scope_proof, semantic)?);
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
                    return Ok(self.atom_nf(scope_proof, semantic)?);
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
            // Binder-open coordinates are structural semantic inputs; the atom below carries
            // the complete node. Coordinates are first reduced to their range-proved canonical
            // affine form so rotation-composed views of the same row share one semantic ID and
            // their q-scale +/- pairs cancel exactly.
            MatrixOperation::IndexedSlice { .. } => {
                if let Some(canonical) =
                    self.canonical_indexed_slice(scope_proof, node, operation, children)?
                {
                    return Ok(self.atom_nf(scope_proof, canonical)?);
                }
                Ok(self.atom_nf(scope_proof, semantic)?)
            }
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
                    return Ok(self.atom_nf(scope_proof, semantic)?);
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
                        return Ok(self.atom_nf(scope_proof, semantic)?);
                    };
                    let Some(right) = children.get(1).and_then(|value| value.exact_nf.as_ref())
                    else {
                        return Ok(self.atom_nf(scope_proof, semantic)?);
                    };
                    match self.tensor_scalar_action_nf(
                        scope_proof,
                        operation,
                        semantic.expression(),
                        node.inputs[0],
                        node.inputs[1],
                        left,
                        right,
                    )? {
                        ScalarActionNormalization::Exact(normal_form) => Ok(normal_form),
                        ScalarActionNormalization::Opaque => self.atom_nf(scope_proof, semantic),
                        ScalarActionNormalization::NotApplicable => {
                            // A non-scalar tensor remains a tensor factor. `tensor_nf` distributes
                            // only over exact polynomial terms; it never treats matrix tensor
                            // multiplication as an ordinary scalar product.
                            self.tensor_nf(
                                scope_proof,
                                operation,
                                node.inputs[0],
                                node.inputs[1],
                                left,
                                right,
                            )
                        }
                    }
                }
            }
            MatrixOperation::CrtRecompose { reconstruction_coefficients, .. } => {
                if reconstruction_coefficients.len() != children.len() {
                    return Ok(self.atom_nf(scope_proof, semantic)?);
                }
                let mut output = PolynomialNF::zero();
                for (child, coefficient) in children.iter().zip(reconstruction_coefficients) {
                    let Some(input) = child.exact_nf.as_ref() else {
                        return Ok(self.atom_nf(scope_proof, semantic)?);
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
                    Ok(self.atom_nf(scope_proof, semantic)?)
                }
            }
            MatrixOperation::ExtractCoefficient { .. } => Ok(self.atom_nf(scope_proof, semantic)?),
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

    /// Canonicalize the four binder-open Int coordinates of one indexed slice by exact
    /// range-aware affine reduction. Rotation-style index compositions materialize the same
    /// source row as `(a*i + b + k*m) mod m`; the binder's trusted range proves which multiple
    /// of `m` is active, so the remainder is removed as an integer identity. Semantically equal
    /// but syntactically different slices then intern to one node, and their modulus-scale
    /// +/- pairs cancel exactly instead of surviving as unfoldable Large residuals.
    fn canonical_indexed_slice(
        &mut self,
        scope_proof: &mut ScopeProof,
        node: &ExprNode,
        operation: &MatrixOperation,
        children: &[Arc<AnalyzedValue>],
    ) -> Result<Option<ScopedExprId>, NormalizeError> {
        let Some(range) = self.scope_argument_range() else {
            return Ok(None);
        };
        if range.minimum >= range.maximum_exclusive {
            return Ok(None);
        }
        let mut inputs = Vec::with_capacity(node.inputs.len());
        let mut changed = false;
        for (position, input) in node.inputs.iter().copied().enumerate() {
            if position == 0 {
                inputs.push(self.expressions.scoped_from_proof(scope_proof, input)?);
                continue;
            }
            let Some((argument, a, b)) = self.range_reduced_affine_form(input, range)? else {
                inputs.push(self.expressions.scoped_from_proof(scope_proof, input)?);
                continue;
            };
            let canonical = self.intern_affine_index(scope_proof, argument, &a, &b)?;
            if canonical.expression() != input {
                changed = true;
            }
            inputs.push(canonical);
        }
        if !changed {
            return Ok(None);
        }
        let rewritten = self.expressions.intern_scoped_transform(
            scope_proof,
            ValueOperator::Matrix(operation.clone()),
            &inputs,
        )?;
        // The rewritten atom shares the source matrix, so it shares the source's value-level
        // transfer; record it so a term retaining this factor keeps a usable bound.
        if let Some(bound) = children.first().map(|value| value.coefficient_bound.clone()) {
            self.expression_bounds.entry(rewritten.expression()).or_insert(bound);
        }
        Ok(Some(rewritten))
    }

    /// The trusted range of this scope's single Int binder, when it has one.
    fn scope_argument_range(&self) -> Option<super::arena::TrustedIndexRange> {
        let program = self.programs.program(self.scope).ok()?;
        let [input] = program.signature.inputs.as_ref() else {
            return None;
        };
        if input.value_type != ResolvedValueType::Int {
            return None;
        }
        input.trusted_index_range
    }

    /// Exact affine form `a * argument + b` of one binder-open Int expression, with
    /// range-proved remainder elimination: `x mod m` reduces to `x - k*m` only when the
    /// binder's trusted range confines `x` to the single window `[k*m, (k+1)*m)`. Every
    /// non-provable shape returns `None`, keeping the original expression (fail closed).
    #[allow(clippy::type_complexity)]
    fn range_reduced_affine_form(
        &self,
        expression: ExprId,
        range: super::arena::TrustedIndexRange,
    ) -> Result<Option<(Option<ExprId>, BigInt, BigInt)>, NormalizeError> {
        let node = self.expressions.node(expression)?;
        let merge_arguments = |left: Option<ExprId>, right: Option<ExprId>| match (left, right) {
            (None, argument) | (argument, None) => Some(argument),
            (Some(left), Some(right)) if left == right => Some(Some(left)),
            _ => None,
        };
        match &node.operator {
            ValueOperator::Argument { position: 0, value_type } => {
                if *value_type != ResolvedValueType::Int {
                    return Ok(None);
                }
                Ok(Some((Some(expression), BigInt::from(1_u8), BigInt::from(0_u8))))
            }
            ValueOperator::Constant(TypedConstant {
                value: super::arena::ConstantValue::Int(value),
                ..
            }) => Ok(Some((None, BigInt::from(0_u8), value.clone()))),
            ValueOperator::Scalar(operation) => match operation {
                ScalarOperation::Negate if node.inputs.len() == 1 => {
                    let Some((argument, a, b)) =
                        self.range_reduced_affine_form(node.inputs[0], range)?
                    else {
                        return Ok(None);
                    };
                    Ok(Some((argument, -a, -b)))
                }
                ScalarOperation::Add | ScalarOperation::Subtract if node.inputs.len() == 2 => {
                    let Some((left_argument, left_a, left_b)) =
                        self.range_reduced_affine_form(node.inputs[0], range)?
                    else {
                        return Ok(None);
                    };
                    let Some((right_argument, right_a, right_b)) =
                        self.range_reduced_affine_form(node.inputs[1], range)?
                    else {
                        return Ok(None);
                    };
                    let Some(argument) = merge_arguments(left_argument, right_argument) else {
                        return Ok(None);
                    };
                    if matches!(operation, ScalarOperation::Add) {
                        Ok(Some((argument, left_a + right_a, left_b + right_b)))
                    } else {
                        Ok(Some((argument, left_a - right_a, left_b - right_b)))
                    }
                }
                ScalarOperation::Multiply if node.inputs.len() == 2 => {
                    let Some((left_argument, left_a, left_b)) =
                        self.range_reduced_affine_form(node.inputs[0], range)?
                    else {
                        return Ok(None);
                    };
                    let Some((right_argument, right_a, right_b)) =
                        self.range_reduced_affine_form(node.inputs[1], range)?
                    else {
                        return Ok(None);
                    };
                    if left_a.is_zero() {
                        Ok(Some((right_argument, right_a * &left_b, right_b * left_b)))
                    } else if right_a.is_zero() {
                        Ok(Some((left_argument, left_a * &right_b, left_b * right_b)))
                    } else {
                        Ok(None)
                    }
                }
                ScalarOperation::Remainder if node.inputs.len() == 2 => {
                    let Some((argument, a, b)) =
                        self.range_reduced_affine_form(node.inputs[0], range)?
                    else {
                        return Ok(None);
                    };
                    let Some((None, modulus_a, modulus)) =
                        self.range_reduced_affine_form(node.inputs[1], range)?
                    else {
                        return Ok(None);
                    };
                    if !modulus_a.is_zero() || modulus <= BigInt::from(0_u8) {
                        return Ok(None);
                    }
                    let first = BigInt::from(range.minimum);
                    let last = BigInt::from(range.maximum_exclusive) - BigInt::from(1_u8);
                    let (minimum, maximum) = if argument.is_none() || a.is_zero() {
                        (b.clone(), b.clone())
                    } else {
                        let low = &a * &first + &b;
                        let high = &a * &last + &b;
                        if low <= high { (low, high) } else { (high, low) }
                    };
                    let window = floor_div(&minimum, &modulus);
                    if floor_div(&maximum, &modulus) != window {
                        return Ok(None);
                    }
                    Ok(Some((argument, a, b - window * modulus)))
                }
                _ => Ok(None),
            },
            _ => Ok(None),
        }
    }

    /// Intern the canonical expression for `a * argument + b`: `Const(b)` for constants, the
    /// bare argument for the identity map, otherwise `Add/Subtract(Multiply(argument, a), |b|)`.
    fn intern_affine_index(
        &mut self,
        scope_proof: &mut ScopeProof,
        argument: Option<ExprId>,
        a: &BigInt,
        b: &BigInt,
    ) -> Result<ScopedExprId, NormalizeError> {
        let Some(argument) = argument.filter(|_| !a.is_zero()) else {
            return self.intern_scoped_int_constant(scope_proof, b);
        };
        let argument = self.expressions.scoped_from_proof(scope_proof, argument)?;
        let base = if *a == BigInt::from(1_u8) {
            argument
        } else {
            let factor = self.intern_scoped_int_constant(scope_proof, a)?;
            self.expressions.intern_scoped_transform(
                scope_proof,
                ValueOperator::Scalar(ScalarOperation::Multiply),
                &[argument, factor],
            )?
        };
        if b.is_zero() {
            return Ok(base);
        }
        let (operation, magnitude) = if *b < BigInt::from(0_u8) {
            (ScalarOperation::Subtract, -b.clone())
        } else {
            (ScalarOperation::Add, b.clone())
        };
        let offset = self.intern_scoped_int_constant(scope_proof, &magnitude)?;
        Ok(self.expressions.intern_scoped_transform(
            scope_proof,
            ValueOperator::Scalar(operation),
            &[base, offset],
        )?)
    }

    fn intern_scoped_int_constant(
        &mut self,
        scope_proof: &mut ScopeProof,
        value: &BigInt,
    ) -> Result<ScopedExprId, NormalizeError> {
        Ok(self.expressions.intern_scoped_transform(
            scope_proof,
            ValueOperator::Constant(TypedConstant::int(value.clone())),
            &[],
        )?)
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
        let mut result =
            PolynomialNF { exact_terms: terms, bounded_summary: input.bounded_summary.clone() };
        self.fold_finite_no_match_terms(&mut result, true)?;
        Ok(result)
    }

    fn tensor_nf(
        &mut self,
        scope_proof: &mut ScopeProof,
        operation: &MatrixOperation,
        left_expression: ExprId,
        right_expression: ExprId,
        left: &PolynomialNF,
        right: &PolynomialNF,
    ) -> Result<PolynomialNF, NormalizeError> {
        let ResolvedValueType::Matrix(left_type) = self.expressions.value_type(left_expression)?
        else {
            return Err(NormalizeError::InvalidExactPlan {
                reason: "tensor left input is not a matrix",
            })
        };
        let left_type = left_type.clone();
        let ResolvedValueType::Matrix(right_type) =
            self.expressions.value_type(right_expression)?
        else {
            return Err(NormalizeError::InvalidExactPlan {
                reason: "tensor right input is not a matrix",
            })
        };
        let right_type = right_type.clone();
        let left_bound = self.bound_normal_form(left)?;
        let right_bound = self.bound_normal_form(right)?;
        if matches!(
            left_bound,
            NumericContract::Known(CoefficientBound::ExactZero | CoefficientBound::Finite(_))
        ) && matches!(
            right_bound,
            NumericContract::Known(CoefficientBound::ExactZero | CoefficientBound::Finite(_))
        ) {
            return Ok(PolynomialNF {
                exact_terms: BTreeMap::new(),
                bounded_summary: BoundedSummary::from_contract(self.typed_tensor_contract(
                    &left_type,
                    &left_bound,
                    &right_type,
                    &right_bound,
                )?)?,
            })
        }
        let noise = self.tensor_summary_contract(&left_type, left, &right_type, right)?;
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
        let mut result = PolynomialNF {
            exact_terms: terms,
            bounded_summary: BoundedSummary::from_contract(noise)?,
        };
        self.fold_finite_no_match_terms(&mut result, true)?;
        Ok(result)
    }

    /// Flatten a tensor when one operand is exactly a row-major 1x1 matrix, using the same typed
    /// scalar-action authority as ordinary multiplication.
    fn tensor_scalar_action_nf(
        &mut self,
        scope_proof: &ScopeProof,
        operation: &MatrixOperation,
        output_expression: ExprId,
        left_expression: ExprId,
        right_expression: ExprId,
        left: &PolynomialNF,
        right: &PolynomialNF,
    ) -> Result<ScalarActionNormalization, NormalizeError> {
        let MatrixOperation::Tensor { output, left_layout, right_layout, output_layout } =
            operation
        else {
            return Ok(ScalarActionNormalization::NotApplicable);
        };
        let ResolvedValueType::Matrix(left_type) = self.expressions.value_type(left_expression)?
        else {
            return Ok(ScalarActionNormalization::NotApplicable);
        };
        let ResolvedValueType::Matrix(right_type) =
            self.expressions.value_type(right_expression)?
        else {
            return Ok(ScalarActionNormalization::NotApplicable);
        };
        let left_scalar = left_type.rows == 1 &&
            left_type.columns == 1 &&
            *left_layout == MatrixLayout::row_major(1, 1);
        let right_scalar = right_type.rows == 1 &&
            right_type.columns == 1 &&
            *right_layout == MatrixLayout::row_major(1, 1);
        if !left_scalar && !right_scalar {
            return Ok(ScalarActionNormalization::NotApplicable);
        }
        if output.modulus != left_type.modulus ||
            output.modulus != right_type.modulus ||
            output.ring_dimension != left_type.ring_dimension ||
            output.ring_dimension != right_type.ring_dimension ||
            *left_layout != MatrixLayout::row_major(left_type.rows, left_type.columns) ||
            *right_layout != MatrixLayout::row_major(right_type.rows, right_type.columns) ||
            *output_layout != MatrixLayout::row_major(output.rows, output.columns)
        {
            return Ok(ScalarActionNormalization::Opaque);
        }
        self.scalar_action_nf(
            scope_proof,
            output_expression,
            left_expression,
            right_expression,
            left,
            right,
        )
    }

    /// Canonicalize a typed polynomial-ring scalar action for both ordinary multiplication and
    /// scalar-shaped tensors. Every exact term of a scalar operand must consist solely of 1x1
    /// factors of that exact type. A composite 1x1 result built from non-scalar ordered factors is
    /// retained as one opaque expression rather than being partially commuted.
    fn scalar_action_nf(
        &mut self,
        scope_proof: &ScopeProof,
        output_expression: ExprId,
        left_expression: ExprId,
        right_expression: ExprId,
        left: &PolynomialNF,
        right: &PolynomialNF,
    ) -> Result<ScalarActionNormalization, NormalizeError> {
        let ResolvedValueType::Matrix(output_type) =
            self.expressions.value_type(output_expression)?
        else {
            return Ok(ScalarActionNormalization::NotApplicable);
        };
        let output_type = output_type.clone();
        let ResolvedValueType::Matrix(left_type) = self.expressions.value_type(left_expression)?
        else {
            return Ok(ScalarActionNormalization::NotApplicable);
        };
        let left_type = left_type.clone();
        let ResolvedValueType::Matrix(right_type) =
            self.expressions.value_type(right_expression)?
        else {
            return Ok(ScalarActionNormalization::NotApplicable);
        };
        let right_type = right_type.clone();
        let left_bound = self.bound_exact_terms(left)?;
        let right_bound = self.bound_exact_terms(right)?;
        self.scalar_action_nf_typed(
            scope_proof,
            &output_type,
            &left_type,
            &right_type,
            &left_bound,
            &right_bound,
            left,
            right,
        )
    }

    /// Apply scalar action using already-resolved typed operands. Compact algebra values may have
    /// no semantic parent expression, so this helper never consults a parent ExprId.
    fn scalar_action_nf_typed(
        &mut self,
        scope_proof: &ScopeProof,
        output_type: &ResolvedMatrixType,
        left_type: &ResolvedMatrixType,
        right_type: &ResolvedMatrixType,
        left_bound: &NumericContract<CoefficientBound>,
        right_bound: &NumericContract<CoefficientBound>,
        left: &PolynomialNF,
        right: &PolynomialNF,
    ) -> Result<ScalarActionNormalization, NormalizeError> {
        // Keep these bounds in the typed contract for callers which have already proved them;
        // exact algebra itself remains valid when an ordinary eager caller has no bound fact.
        let _ = (left_bound, right_bound);
        let left_scalar = left_type.rows == 1 && left_type.columns == 1;
        let right_scalar = right_type.rows == 1 && right_type.columns == 1;
        if !left_scalar && !right_scalar {
            return Ok(ScalarActionNormalization::NotApplicable);
        }
        let expected_output = if left_scalar { right_type } else { left_type };
        if left_type.modulus != right_type.modulus ||
            left_type.ring_dimension != right_type.ring_dimension ||
            output_type != expected_output
        {
            return Ok(ScalarActionNormalization::Opaque);
        }

        if (left_scalar && !self.scalar_nf_ordered_factors_match_type(left, &left_type)?) ||
            (right_scalar && !self.scalar_nf_ordered_factors_match_type(right, &right_type)?)
        {
            return Ok(ScalarActionNormalization::Opaque);
        }

        let ordered_scalar_product = if left_scalar && right_scalar {
            // Preserve ordered exact relations such as G * Decompose(A) = A before using the
            // commutativity of the scalar result. The product is centralized only after every
            // surviving ordered factor is proven to have the declared 1x1 output type. In the
            // reversed order no relation applies, so both typed scalar factors remain present.
            let product = self.with_compact_product_context(|normalizer| {
                normalizer.product_nf(scope_proof, &left_type, &right_type, left, right)
            })?;
            if !self.scalar_nf_ordered_factors_match_type(&product, output_type)? {
                return Ok(ScalarActionNormalization::Opaque);
            }
            Some(product)
        } else {
            None
        };

        if let Some(ordered_product) = ordered_scalar_product {
            return Ok(ScalarActionNormalization::Exact(
                self.centralize_scalar_nf(scope_proof, &ordered_product)?,
            ));
        }

        let reclassified_left =
            if left_scalar { self.centralize_scalar_nf(scope_proof, left)? } else { left.clone() };
        let reclassified_right = if right_scalar {
            self.centralize_scalar_nf(scope_proof, right)?
        } else {
            right.clone()
        };
        Ok(ScalarActionNormalization::Exact(self.product_nf(
            scope_proof,
            &left_type,
            &right_type,
            &reclassified_left,
            &reclassified_right,
        )?))
    }

    /// Move every typed scalar factor into the commutative part of its monomial. Callers must
    /// first prove that all ordered factors have the declared 1x1 matrix type.
    fn centralize_scalar_nf(
        &mut self,
        scope_proof: &ScopeProof,
        normal_form: &PolynomialNF,
    ) -> Result<PolynomialNF, NormalizeError> {
        let mut reclassified_terms = BTreeMap::new();
        for (monomial, coefficient) in &normal_form.exact_terms {
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
        Ok(PolynomialNF {
            exact_terms: reclassified_terms,
            bounded_summary: normal_form.bounded_summary.clone(),
        })
    }

    fn scalar_nf_ordered_factors_match_type(
        &self,
        normal_form: &PolynomialNF,
        scalar_type: &ResolvedMatrixType,
    ) -> Result<bool, NormalizeError> {
        for monomial in normal_form.exact_terms.keys() {
            let descriptor = self.monomials.descriptor(*monomial)?;
            for factor in descriptor.ordered_factors.iter() {
                if !matches!(
                    self.expressions.value_type(factor.expression()),
                    Ok(ResolvedValueType::Matrix(matrix)) if matrix == scalar_type
                ) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
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
            return self.atom_nf(scope_proof, semantic);
        }

        let mut zero_inputs = Vec::new();
        zero_inputs.try_reserve(children.len()).map_err(|_| NormalizeError::ArithmeticOverflow)?;
        for input in &node.inputs {
            let ResolvedValueType::Matrix(input_type) = self.expressions.value_type(*input)? else {
                return self.atom_nf(scope_proof, semantic);
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
        let summaries = children
            .iter()
            .filter_map(|child| child.exact_nf.as_ref())
            .map(|normal_form| normal_form.bounded_summary.coefficient_bound())
            .collect::<Vec<_>>();
        let mut result = PolynomialNF {
            exact_terms: terms,
            bounded_summary: BoundedSummary::from_contract(max_bounds(&summaries)?)?,
        };
        self.fold_finite_no_match_terms(&mut result, true)?;
        Ok(result)
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
                let left_type = left_type.clone();
                let Some(expected) = checked_matrix_product_output(&left_type, &component_type)
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
                Ok(Some(self.product_nf(
                    scope_proof,
                    &left_type,
                    &component_type,
                    left_nf,
                    component_nf,
                )?))
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
                let right_type = right_type.clone();
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
                Ok(Some(self.product_nf(
                    scope_proof,
                    &component_type,
                    &right_type,
                    component_nf,
                    right_nf,
                )?))
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

    fn atom_nf(
        &mut self,
        scope_proof: &ScopeProof,
        semantic: ScopedExprId,
    ) -> Result<PolynomialNF, NormalizeError> {
        self.expressions.validate_scoped_from_proof(scope_proof, semantic)?;
        let id = self.atom_monomial(Some(scope_proof), semantic)?;
        let mut terms = BTreeMap::new();
        terms.insert(id, BigInt::from(1_u8));
        Ok(PolynomialNF { exact_terms: terms, bounded_summary: BoundedSummary::zero() })
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
        let mut result = PolynomialNF {
            exact_terms: terms,
            bounded_summary: BoundedSummary::from_contract(add_noise_summaries(
                &left.bounded_summary.coefficient_bound(),
                &right.bounded_summary.coefficient_bound(),
            ))?,
        };
        self.fold_finite_no_match_terms(&mut result, true)?;
        Ok(result)
    }

    fn negate_nf(&self, value: &PolynomialNF) -> PolynomialNF {
        let mut terms = BTreeMap::new();
        for (id, coefficient) in &value.exact_terms {
            terms.insert(*id, -coefficient.clone());
        }
        PolynomialNF { exact_terms: terms, bounded_summary: value.bounded_summary.clone() }
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
        PolynomialNF {
            exact_terms: terms,
            bounded_summary: BoundedSummary::from_contract(scale_noise_summary(
                &value.bounded_summary.coefficient_bound(),
                scale.magnitude(),
            ))
            .expect("scaling a finite summary remains finite"),
        }
    }

    fn product_nf(
        &mut self,
        _scope_proof: &ScopeProof,
        left_type: &ResolvedMatrixType,
        right_type: &ResolvedMatrixType,
        left: &PolynomialNF,
        right: &PolynomialNF,
    ) -> Result<PolynomialNF, NormalizeError> {
        let mut terms = BTreeMap::new();
        let mut noise = NumericContract::Known(CoefficientBound::ExactZero);
        self.execute_product_into(
            left_type,
            right_type,
            left,
            right,
            &BigInt::from(1_u8),
            &mut terms,
            &mut noise,
        )?;

        Ok(PolynomialNF {
            exact_terms: terms,
            bounded_summary: BoundedSummary::from_contract(noise)?,
        })
    }

    fn bound_exact_terms(
        &self,
        normal_form: &PolynomialNF,
    ) -> Result<NumericContract<CoefficientBound>, NormalizeError> {
        let mut total = CoefficientBound::ExactZero;
        for (&monomial, coefficient) in &normal_form.exact_terms {
            let NumericContract::Known(bound) = self.bound_monomial(monomial, coefficient)? else {
                return Ok(NumericContract::Missing);
            };
            total = add_known_bounds(&total, &bound);
        }
        Ok(NumericContract::Known(total))
    }

    /// A bounded relation endpoint is not a Large/exact semantic term.  It is kept with identity
    /// for exactly one direct registered boundary, then folded like every other finite term.
    fn is_retained_bounded_monomial(&self, monomial: MonomialId) -> Result<bool, NormalizeError> {
        let descriptor = self.monomials.descriptor(monomial)?;
        if descriptor.central_factors.is_empty() && descriptor.ordered_factors.len() == 1 {
            let factor = descriptor.ordered_factors[0];
            if factor.program() == self.scope &&
                self.retained_bounded_endpoints.contains(&factor.expression())
            {
                return Ok(true)
            }
        }
        let mut factors =
            descriptor.central_factors.iter().chain(descriptor.ordered_factors.iter());
        let Some(first) = factors.next() else { return Ok(false) };
        Ok(first.program() == self.scope &&
            self.retained_bounded_scalar_factors.contains(&first.expression()) &&
            factors.all(|factor| {
                factor.program() == self.scope &&
                    self.retained_bounded_scalar_factors.contains(&factor.expression())
            }))
    }

    fn typed_product_contract(
        &self,
        left_type: &ResolvedMatrixType,
        left: &NumericContract<CoefficientBound>,
        right_type: &ResolvedMatrixType,
        right: &NumericContract<CoefficientBound>,
    ) -> Result<NumericContract<CoefficientBound>, NormalizeError> {
        let (NumericContract::Known(left), NumericContract::Known(right)) = (left, right) else {
            return Ok(NumericContract::Missing);
        };
        let product = product_bound_with_facts(
            &CanonicalMatrixBound {
                matrix_type: concrete_type(left_type),
                coefficient_class: canonical_class(left),
            },
            &CanonicalMatrixBound {
                matrix_type: concrete_type(right_type),
                coefficient_class: canonical_class(right),
            },
            &MatrixProductFacts::default(),
        )
        .map_err(|_| NormalizeError::ArithmeticOverflow)?;
        Ok(NumericContract::Known(coefficient_bound(&product.coefficient_class)))
    }

    fn typed_tensor_contract(
        &self,
        left_type: &ResolvedMatrixType,
        left: &NumericContract<CoefficientBound>,
        right_type: &ResolvedMatrixType,
        right: &NumericContract<CoefficientBound>,
    ) -> Result<NumericContract<CoefficientBound>, NormalizeError> {
        let (NumericContract::Known(left), NumericContract::Known(right)) = (left, right) else {
            return Ok(NumericContract::Missing)
        };
        let tensor = tensor_bound_with_facts(
            &CanonicalMatrixBound {
                matrix_type: concrete_type(left_type),
                coefficient_class: canonical_class(left),
            },
            &CanonicalMatrixBound {
                matrix_type: concrete_type(right_type),
                coefficient_class: canonical_class(right),
            },
            &MatrixProductFacts::default(),
        )
        .map_err(|_| NormalizeError::ArithmeticOverflow)?;
        Ok(NumericContract::Known(coefficient_bound(&tensor.coefficient_class)))
    }

    fn tensor_summary_contract(
        &mut self,
        left_type: &ResolvedMatrixType,
        left: &PolynomialNF,
        right_type: &ResolvedMatrixType,
        right: &PolynomialNF,
    ) -> Result<NumericContract<CoefficientBound>, NormalizeError> {
        let left_summary = left.bounded_summary.coefficient_bound();
        let right_summary = right.bounded_summary.coefficient_bound();
        let mut contribution =
            self.typed_tensor_contract(left_type, &left_summary, right_type, &right_summary)?;
        for (summary, summary_type, exact, exact_type, summary_on_left) in [
            (&left_summary, left_type, right, right_type, true),
            (&right_summary, right_type, left, left_type, false),
        ] {
            if summary.as_known().is_none_or(|bound| bound == &CoefficientBound::ExactZero) ||
                exact.exact_terms.is_empty()
            {
                continue
            }
            let exact_bound = self.bound_exact_terms(exact)?;
            if !matches!(
                exact_bound,
                NumericContract::Known(CoefficientBound::ExactZero | CoefficientBound::Finite(_))
            ) {
                return Err(NormalizeError::InvalidExactPlan {
                    reason: "compressed bounded summary tensored with Large or Missing exact value",
                })
            }
            let cross = if summary_on_left {
                self.typed_tensor_contract(summary_type, summary, exact_type, &exact_bound)?
            } else {
                self.typed_tensor_contract(exact_type, &exact_bound, summary_type, summary)?
            };
            contribution = add_noise_summaries(&contribution, &cross);
        }
        Ok(contribution)
    }

    /// Precompute every contribution involving an already-erased all-bounded summary.
    ///
    /// This runs before the exact product mutates the arena or destination.  A summary may be
    /// multiplied only by another finite component (including a retained finite relation
    /// endpoint).  Multiplying it by a Large/Missing exact value would lose the required ordered
    /// factor identity and therefore fails closed.
    fn product_summary_contract(
        &mut self,
        left_type: &ResolvedMatrixType,
        left: &PolynomialNF,
        right_type: &ResolvedMatrixType,
        right: &PolynomialNF,
        weight: &BigInt,
    ) -> Result<NumericContract<CoefficientBound>, NormalizeError> {
        if weight.is_zero() {
            return Ok(NumericContract::Known(CoefficientBound::ExactZero))
        }
        let left_summary = left.bounded_summary.coefficient_bound();
        let right_summary = right.bounded_summary.coefficient_bound();
        let mut contribution =
            self.typed_product_contract(left_type, &left_summary, right_type, &right_summary)?;

        // An identityless summary cannot become an ordered exact word again. Cross products with
        // retained exact terms therefore use the complete exact-side bound and fail closed as
        // Large/Missing when that is what the typed transfer proves.
        for (summary, summary_type, exact, exact_type, summary_on_left) in [
            (&left_summary, left_type, right, right_type, true),
            (&right_summary, right_type, left, left_type, false),
        ] {
            if summary.as_known().is_none_or(|bound| bound == &CoefficientBound::ExactZero) ||
                exact.exact_terms.is_empty()
            {
                continue
            }
            let exact_bound = self.bound_exact_terms(exact)?;
            if !matches!(
                exact_bound,
                NumericContract::Known(CoefficientBound::ExactZero | CoefficientBound::Finite(_))
            ) {
                return Err(NormalizeError::InvalidExactPlan {
                    reason: "compressed bounded summary multiplied by Large or Missing exact value",
                });
            }
            let cross = if summary_on_left {
                self.typed_product_contract(summary_type, summary, exact_type, &exact_bound)?
            } else {
                self.typed_product_contract(exact_type, &exact_bound, summary_type, summary)?
            };
            contribution = add_noise_summaries(&contribution, &cross);
        }
        Ok(scale_noise_summary(&contribution, weight.magnitude()))
    }

    /// Execute one complete eager product lifecycle.
    fn execute_product_into<A: ExactTermAccumulator>(
        &mut self,
        left_type: &ResolvedMatrixType,
        right_type: &ResolvedMatrixType,
        left: &PolynomialNF,
        right: &PolynomialNF,
        weight: &BigInt,
        terms: &mut A,
        noise: &mut NumericContract<CoefficientBound>,
    ) -> Result<(), NormalizeError> {
        let summary_contribution =
            self.product_summary_contract(left_type, left, right_type, right, weight)?;
        self.product_into_body(left, right, weight, terms, noise)?;
        *noise = add_noise_summaries(noise, &summary_contribution);
        Ok(())
    }

    fn product_into_body<A: ExactTermAccumulator>(
        &mut self,
        left: &PolynomialNF,
        right: &PolynomialNF,
        weight: &BigInt,
        terms: &mut A,
        noise: &mut NumericContract<CoefficientBound>,
    ) -> Result<(), NormalizeError> {
        let mut worklist = VecDeque::<ProductWorkItem>::new();
        for (left_id, left_coefficient) in &left.exact_terms {
            for (right_id, right_coefficient) in &right.exact_terms {
                let coefficient = left_coefficient * right_coefficient * weight;
                if coefficient.is_zero() {
                    continue;
                }
                let product = self.product_monomials(*left_id, *right_id)?;
                worklist.push_back(ProductWorkItem::Term(product, coefficient));
                // Drain each completed Cartesian pair before generating the next one. The same
                // rewrite queue remains authoritative, but its live size follows one pair's
                // recursive splice instead of the full product cardinality.
                self.drain_product_worklist(terms, noise, &mut worklist)?;
            }
        }
        self.drain_product_worklist(terms, noise, &mut worklist)
    }

    fn drain_product_worklist<A: ExactTermAccumulator>(
        &mut self,
        terms: &mut A,
        noise: &mut NumericContract<CoefficientBound>,
        worklist: &mut VecDeque<ProductWorkItem>,
    ) -> Result<(), NormalizeError> {
        while let Some(item) = worklist.pop_front() {
            let ProductWorkItem::Term(monomial, coefficient) = item else {
                let ProductWorkItem::GadgetSplice(mut splice) = item else { unreachable!() };
                let lower = splice
                    .next_after
                    .map(std::ops::Bound::Excluded)
                    .unwrap_or(std::ops::Bound::Unbounded);
                let mut input_terms =
                    splice.input_nf.exact_terms.range((lower, std::ops::Bound::Unbounded));
                let batch = input_terms
                    .by_ref()
                    .take(self.gadget_splice_batch_terms.max(1))
                    .map(|(&monomial, coefficient)| (monomial, coefficient.clone()))
                    .collect::<Vec<_>>();
                let has_more = input_terms.next().is_some();
                if batch.is_empty() {
                    if splice.summary_pending {
                        *noise = add_noise_summaries(
                            noise,
                            &scale_noise_summary(
                                &splice.input_nf.bounded_summary.coefficient_bound(),
                                splice.coefficient.magnitude(),
                            ),
                        );
                    }
                    continue;
                }
                let input_monomials =
                    batch.iter().map(|(monomial, _)| *monomial).collect::<Vec<_>>();
                let replacements = self.monomials.combine_interned_wrapped_batch(
                    self.scope,
                    splice.left,
                    &input_monomials,
                    splice.suffix,
                )?;
                if replacements.len() != batch.len() {
                    return Err(NormalizeError::InvalidExactPlan {
                        reason: "streamed gadget splice batch length mismatch",
                    });
                }
                let outer_coefficient = splice.coefficient.clone();
                if has_more {
                    splice.next_after = batch.last().map(|(monomial, _)| *monomial);
                    worklist.push_front(ProductWorkItem::GadgetSplice(splice));
                } else if splice.summary_pending {
                    *noise = add_noise_summaries(
                        noise,
                        &scale_noise_summary(
                            &splice.input_nf.bounded_summary.coefficient_bound(),
                            splice.coefficient.magnitude(),
                        ),
                    );
                }
                for ((_, input_coefficient), replacement) in
                    batch.into_iter().zip(replacements).rev()
                {
                    worklist.push_front(ProductWorkItem::Term(
                        replacement,
                        outer_coefficient.clone() * input_coefficient,
                    ));
                }
                continue;
            };
            if coefficient.is_zero() {
                continue;
            }
            let Some(splice) = self.product_gadget_splice(monomial, coefficient.clone())? else {
                self.merge_product_result_early(monomial, coefficient, terms, noise)?;
                continue;
            };
            worklist.push_front(ProductWorkItem::GadgetSplice(splice));
        }
        Ok(())
    }

    /// Merge one fully gadget-closed product candidate.  Relation closure happens before numeric
    /// erasure, then every all-finite result is absorbed immediately into the single noise
    /// summary.  A Large/Missing result keeps its complete ordered monomial identity, including
    /// every bounded factor, so only an identical full matrix product can cancel it later.
    fn merge_product_result_early<A: ExactTermAccumulator>(
        &mut self,
        monomial: MonomialId,
        coefficient: BigInt,
        terms: &mut A,
        noise: &mut NumericContract<CoefficientBound>,
    ) -> Result<(), NormalizeError> {
        if coefficient.is_zero() {
            return Ok(())
        }
        let initial_bound = self.bound_monomial(monomial, &coefficient)?;
        let needs_relation_closure = matches!(
            initial_bound,
            NumericContract::Known(CoefficientBound::Large) | NumericContract::Missing
        ) && self.relations.is_some() &&
            self.normalization_depth == 1 &&
            !self.suppress_product_relation_closure;
        if !needs_relation_closure {
            return self.merge_product_term(monomial, coefficient, initial_bound, terms, noise)
        }

        let mut candidate = PolynomialNF {
            exact_terms: BTreeMap::from([(monomial, coefficient)]),
            bounded_summary: BoundedSummary::zero(),
        };
        self.rewrite_relations(&mut candidate)?;
        *noise = add_noise_summaries(noise, &candidate.bounded_summary.coefficient_bound());
        for (rewritten, rewritten_coefficient) in candidate.exact_terms {
            let bound = self.bound_monomial(rewritten, &rewritten_coefficient)?;
            self.merge_product_term(rewritten, rewritten_coefficient, bound, terms, noise)?;
        }
        Ok(())
    }

    fn merge_product_term<A: ExactTermAccumulator>(
        &mut self,
        monomial: MonomialId,
        coefficient: BigInt,
        bound: NumericContract<CoefficientBound>,
        terms: &mut A,
        noise: &mut NumericContract<CoefficientBound>,
    ) -> Result<(), NormalizeError> {
        match bound {
            NumericContract::Known(CoefficientBound::ExactZero) => Ok(()),
            NumericContract::Known(bound @ CoefficientBound::Finite(_)) => {
                if self.is_retained_bounded_monomial(monomial)? {
                    return terms.merge(monomial, coefficient);
                }
                self.counters.bounded_fold_count =
                    self.counters.bounded_fold_count.saturating_add(1);
                *noise = add_noise_summaries(noise, &NumericContract::Known(bound));
                Ok(())
            }
            NumericContract::Known(CoefficientBound::Large) | NumericContract::Missing => {
                terms.merge(monomial, coefficient)
            }
        }
    }

    fn product_gadget_splice(
        &mut self,
        monomial: MonomialId,
        coefficient: BigInt,
    ) -> Result<Option<ProductGadgetSplice>, NormalizeError> {
        let (central_factors, ordered_factors) = {
            let descriptor = self.monomials.descriptor(monomial)?;
            (descriptor.central_factors.to_vec(), descriptor.ordered_factors.to_vec())
        };
        for index in 0..ordered_factors.len().saturating_sub(1) {
            let Some(input) = self
                .authorized_gadget_pair_input(ordered_factors[index], ordered_factors[index + 1])?
            else {
                continue;
            };
            let Some(input_nf) =
                self.gadget_input_nf(ordered_factors[index + 1].expression(), input)?
            else {
                if self.compact_mode {
                    return Err(NormalizeError::InvalidExactPlan {
                        reason: "compact gadget shell hold missing",
                    });
                }
                return Ok(None);
            };
            let has_left_context = !central_factors.is_empty() || index != 0;
            let has_suffix_context = index + 2 != ordered_factors.len();
            if !input_nf.bounded_summary.is_zero() && (has_left_context || has_suffix_context) {
                return Err(NormalizeError::InvalidExactPlan {
                    reason: "gadget input noise summary cannot be recombined with an exact prefix or suffix",
                })
            }
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
            return Ok(Some(ProductGadgetSplice {
                left,
                suffix,
                input_nf,
                next_after: None,
                coefficient,
                summary_pending: true,
            }));
        }
        Ok(None)
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
            let Some(input) = self.authorized_gadget_pair_input(gadget, decomposition)? else {
                continue;
            };
            let Some(normal_form) =
                self.splice_gadget_decomposition(&central_factors, &ordered_factors, index, input)?
            else {
                return Ok(None);
            };
            return Ok(Some(normal_form));
        }
        Ok(None)
    }

    /// Rewrite a gadget/decomposition pair which lies exactly across a product boundary without
    /// first interning the transient concatenated descriptor. This is the same typed splice used
    /// by `rewrite_gadget_decomposition`; non-matching pairs retain the ordinary product path.
    fn splice_gadget_decomposition(
        &mut self,
        central_factors: &[ScopedExprId],
        ordered_factors: &[ScopedExprId],
        index: usize,
        input: ExprId,
    ) -> Result<Option<PolynomialNF>, NormalizeError> {
        // `D(A)` itself is an atom in the child NF, but the identity exposes the already
        // normalized polynomial NF of `A`, not the raw input expression. The use-count hold
        // installed during traversal keeps this memo entry alive until this splice.
        let Some(input_nf) =
            self.gadget_input_nf(ordered_factors[index + 1].expression(), input)?
        else {
            if self.compact_mode {
                return Err(NormalizeError::InvalidExactPlan {
                    reason: "compact gadget shell hold missing",
                });
            }
            return Ok(None);
        };
        let has_left_context = !central_factors.is_empty() || index != 0;
        let has_suffix_context = index + 2 != ordered_factors.len();
        if !input_nf.bounded_summary.is_zero() && (has_left_context || has_suffix_context) {
            return Err(NormalizeError::InvalidExactPlan {
                reason: "gadget input noise summary cannot be recombined with an exact prefix or suffix",
            })
        }
        let left = if central_factors.is_empty() && index == 0 {
            None
        } else {
            Some(self.monomials.intern(
                self.expressions,
                self.programs,
                central_factors,
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
        let mut input_terms = input_nf.exact_terms.iter();
        loop {
            let batch = input_terms
                .by_ref()
                .take(self.gadget_splice_batch_terms.max(1))
                .collect::<Vec<_>>();
            if batch.is_empty() {
                break;
            }
            let input_monomials = batch.iter().map(|(id, _)| **id).collect::<Vec<_>>();
            let replacements = self.monomials.combine_interned_wrapped_batch(
                self.scope,
                left,
                &input_monomials,
                suffix,
            )?;
            for ((_, input_coefficient), replacement) in batch.into_iter().zip(replacements) {
                merge_term(&mut terms, replacement, input_coefficient.clone());
            }
        }
        Ok(Some(PolynomialNF {
            exact_terms: terms,
            bounded_summary: input_nf.bounded_summary.clone(),
        }))
    }

    fn authorized_gadget_pair_input(
        &self,
        gadget: ScopedExprId,
        decomposition: ScopedExprId,
    ) -> Result<Option<ExprId>, NormalizeError> {
        Ok(authorized_gadget_pair_rule_from(
            self.expressions,
            self.facts,
            self.gadget_recompositions,
            gadget.expression(),
            decomposition.expression(),
            true,
        )?
        .map(|(input, _)| input))
    }

    fn product_monomials(
        &mut self,
        left: MonomialId,
        right: MonomialId,
    ) -> Result<MonomialId, NormalizeError> {
        Ok(self.monomials.combine_interned(self.scope, left, right)?)
    }

    /// Close exact terms by repeatedly rewriting the leftmost adjacent subword. Recombined RHS
    /// terms go back onto the same deterministic worklist, so prefix, suffix, central factors,
    /// and coefficient multiplication all remain part of the next match.
    fn rewrite_relations(
        &mut self,
        normal_form: &mut PolynomialNF,
    ) -> Result<bool, NormalizeError> {
        let initial = std::mem::take(&mut normal_form.exact_terms);
        let mut worklist = initial.into_iter().collect::<VecDeque<_>>();
        let mut result = BTreeMap::new();
        let mut relation_noise = normal_form.bounded_summary.coefficient_bound();
        let mut changed = false;
        while let Some((monomial, coefficient)) = worklist.pop_front() {
            if coefficient.is_zero() {
                continue;
            }
            // Relation RHS splices can create a new gadget/decomposition adjacency, so ordinary
            // gadget closure must run on every item returned to this worklist.
            if let Some(rewritten) = self.rewrite_gadget_decomposition(monomial)? {
                changed = true;
                relation_noise = add_noise_summaries(
                    &relation_noise,
                    &scale_noise_summary(
                        &rewritten.bounded_summary.coefficient_bound(),
                        coefficient.magnitude(),
                    ),
                );
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
            self.validate_relation_rhs(&rhs)?;
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
            let rhs_noise = scale_noise_summary(
                &rhs.bounded_summary.coefficient_bound(),
                coefficient.magnitude(),
            );
            if rhs_noise != NumericContract::Known(CoefficientBound::ExactZero) {
                // A summary has no multiplicative identity. It may be preserved only when the
                // relation replaces the complete monomial without an exact prefix or suffix.
                if left.is_some() || suffix.is_some() {
                    return Err(NormalizeError::Relation(
                        RelationRegistryError::InvalidCanonicalRhs,
                    ));
                }
                relation_noise = add_noise_summaries(&relation_noise, &rhs_noise);
            }
            let mut recombined = Vec::with_capacity(rhs.exact_terms.len());
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
        normal_form.bounded_summary = BoundedSummary::from_contract(relation_noise)?;
        Ok(changed)
    }

    /// The compressed contract supports only preimage/recomposition right-hand sides which
    /// retain at least one genuinely Large exact term. Finite terms belong in the summary lane;
    /// a finite-only RHS is deliberately outside the supported protocol surface.
    fn validate_relation_rhs(&self, rhs: &PolynomialNF) -> Result<(), NormalizeError> {
        let mut has_large = false;
        for (monomial, coefficient) in &rhs.exact_terms {
            match self.bound_monomial(*monomial, coefficient)? {
                NumericContract::Known(CoefficientBound::Large) => has_large = true,
                NumericContract::Missing => {}
                NumericContract::Known(
                    CoefficientBound::ExactZero | CoefficientBound::Finite(_),
                ) => {
                    return Err(NormalizeError::Relation(RelationRegistryError::InvalidCanonicalRhs))
                }
            }
        }
        if !has_large {
            return Err(NormalizeError::Relation(RelationRegistryError::InvalidCanonicalRhs))
        }
        Ok(())
    }

    fn fold_finite_no_match_terms(
        &mut self,
        normal_form: &mut PolynomialNF,
        preserve_relation_endpoints: bool,
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
                    if preserve_relation_endpoints && self.is_retained_bounded_monomial(monomial)? {
                        retained.insert(monomial, coefficient);
                        continue;
                    }
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
        let existing = normal_form.bounded_summary.coefficient_bound();
        let NumericContract::Known(existing) = existing else {
            unreachable!("bounded summaries are finite-only")
        };
        normal_form.bounded_summary = BoundedSummary::from_contract(NumericContract::Known(
            add_known_bounds(&existing, &folded),
        ))?;
        Ok(())
    }

    /// Count retained exact terms which still expose a uniquely dispatchable preimage call.
    /// This is deliberately a final structural scan: it does not specialize a selector, walk
    /// relation registrations, or attempt another rewrite. A plain exact residual therefore
    /// remains distinct from an unreduced relation-bearing term in the final counters.
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
        if self.relations.is_none() {
            return Ok(None);
        }
        let (central, ordered) = {
            let descriptor = self.monomials.descriptor(monomial)?;
            (descriptor.central_factors.to_vec(), descriptor.ordered_factors.to_vec())
        };
        self.find_universal_subword_match(&central, &ordered)
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
            if node.inputs.len() != 1 {
                continue;
            }
            let Some(dispatch) = relations.dispatch_for_preimage_program(program)? else {
                continue;
            };
            let index = self.expressions.scoped_only_input(k_factor)?;
            if index.program() != self.scope {
                return Err(ArenaError::ScopeMismatch {
                    expected: self.scope,
                    actual: index.program(),
                }
                .into());
            }
            let Some(index_range) = self.universal_index_range(index)? else { continue };
            let specialized = self.specialized_universal_cached(dispatch, index, index_range)?;
            for (lhs, rhs_candidates) in specialized {
                let descriptor = self.monomials.descriptor(lhs.monomial)?;
                if descriptor.ordered_factors.is_empty() || !descriptor.central_factors.is_empty() {
                    continue;
                }
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
                for lhs_k_position in descriptor
                    .ordered_factors
                    .iter()
                    .enumerate()
                    .filter_map(|(position, factor)| (*factor == k_factor).then_some(position))
                {
                    let ordered_len = descriptor.ordered_factors.len();
                    let Some(start) = k_position.checked_sub(lhs_k_position) else { continue };
                    let Some(end) = start.checked_add(ordered_len) else { continue };
                    if end > ordered.len() || ordered[start..end] != descriptor.ordered_factors[..]
                    {
                        continue;
                    }
                    if remove_central_subword(central, &descriptor.central_factors).is_none() {
                        continue;
                    }
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
        if let ValueOperator::Constant(TypedConstant {
            value: super::arena::ConstantValue::Int(value),
            ..
        }) = &node.operator
        {
            let Some(minimum) = value.to_u64() else { return Ok(None) };
            let Some(maximum_exclusive) = minimum.checked_add(1) else { return Ok(None) };
            return Ok(Some(super::arena::TrustedIndexRange { minimum, maximum_exclusive }));
        }
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
            return Ok(normal_form.bounded_summary.coefficient_bound());
        }
        let NumericContract::Known(mut total) = normal_form.bounded_summary.coefficient_bound()
        else {
            unreachable!("bounded summaries are finite-only")
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
            ValueOperator::Matrix(
                MatrixOperation::Slice { .. } | MatrixOperation::IndexedSlice { .. },
            ) => node
                .inputs
                .first()
                .copied()
                .map_or(Ok(NumericContract::Missing), |input| self.factor_bound(input))?,
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
            if !bound.is_missing() {
                return Ok(bound);
            }
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
            // Input zero is the selector. Arena validation proves the remaining nonempty inputs
            // are the complete, same-typed branch set, so their maximum is the exact compact
            // transfer and a missing branch remains fail-closed.
            ValueOperator::ExplicitElement { .. } => max_bounds(&child_bounds[1..])?,
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
                let bound = bounds.first().cloned().unwrap_or(NumericContract::Missing);
                if bound.is_missing() && matches!(operation, MatrixOperation::Slice { .. }) {
                    self.factor_bound(node.inputs[0])
                } else {
                    Ok(bound)
                }
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

    fn direct_integer_constant(&self, expression: ExprId) -> Option<BigInt> {
        let node = self.expressions.node(expression).ok()?;
        let ValueOperator::Constant(super::arena::TypedConstant {
            value: super::arena::ConstantValue::Int(value),
            ..
        }) = &node.operator
        else {
            return None;
        };
        Some(value.clone())
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
        SamplerOperation::UniformInterval { output, minimum, maximum } => {
            let upper = minimum.abs().max(maximum.abs());
            // An interval that reaches the centered halfway point carries no more information
            // than a uniform residue; report it as Large instead of a modulus-scale finite
            // bound. Small designed intervals (ternary secrets, bits) keep their exact bound.
            if upper.magnitude() * 2_u8 >= output.modulus {
                NumericContract::Known(CoefficientBound::Large)
            } else {
                NumericContract::Known(CoefficientBound::finite(upper.magnitude().clone()))
            }
        }
        SamplerOperation::Gaussian { max_coefficient_bound, .. } |
        SamplerOperation::Preimage { max_coefficient_bound, .. } => NumericContract::Known(
            CoefficientBound::finite(max_coefficient_bound.magnitude().clone()),
        ),
        // The matrix-valued trapdoor sample port is the uniform public matrix `B`; its
        // `preimage_max_coefficient_bound` is metadata for preimages sampled against this
        // trapdoor later, never a bound on `B` itself.
        SamplerOperation::Trapdoor { .. } => NumericContract::Known(CoefficientBound::Large),
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

/// Floor division for a strictly positive divisor, matching `div_euclid` on integers.
fn floor_div(value: &BigInt, divisor: &BigInt) -> BigInt {
    let quotient = value / divisor;
    if (value - &quotient * divisor) < BigInt::from(0_u8) { quotient - 1 } else { quotient }
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

/// Add two identityless noise contributions. Production exact values canonicalize an absent
/// summary to `ExactZero` before arithmetic; a `Missing` observed here is therefore a real
/// fail-closed numeric contract and must propagate.
fn add_noise_summaries(
    left: &NumericContract<CoefficientBound>,
    right: &NumericContract<CoefficientBound>,
) -> NumericContract<CoefficientBound> {
    match (left, right) {
        (NumericContract::Known(left), NumericContract::Known(right)) => {
            NumericContract::Known(add_known_bounds(left, right))
        }
        _ => NumericContract::Missing,
    }
}

fn scale_noise_summary(
    summary: &NumericContract<CoefficientBound>,
    factor: &BigUint,
) -> NumericContract<CoefficientBound> {
    if factor.is_zero() {
        return NumericContract::Known(CoefficientBound::ExactZero)
    }
    let NumericContract::Known(bound) = summary else { return NumericContract::Missing };
    match bound {
        CoefficientBound::ExactZero => NumericContract::Known(CoefficientBound::ExactZero),
        CoefficientBound::Finite(value) => NumericContract::Known(CoefficientBound::finite(
            &value.maximum_absolute_coefficient * factor,
        )),
        CoefficientBound::Large => NumericContract::Known(CoefficientBound::Large),
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
            ArenaToken, FamilyDomain, HashVariant, MatrixLayout, MatrixOperation, ProgramInput,
            ProgramSignature, ResolvedMatrixType, SampleEventId, SamplerOperation,
            SemanticFamilySourceIdentity, SemanticSourceIdentity, TrustedIndexRange,
        },
        facts::{MatrixFacts, MatrixMetadata, ValueFacts},
        relation::{
            CanonicalLhsKey, CanonicalRhsId, FactorOrderContract, GadgetRecompositionRegistry,
            GadgetRecompositionRule, RelationRegistry, RelationValidationAuthority,
            SamplerSourceContract, StaticLhsKey, TrapdoorSourceContract, UniversalDispatchKey,
            UniversalRelationRegistration,
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
            bounded_summary: BoundedSummary::zero(),
        };
        assert_eq!(
            canonical_lhs_monomial(Some(&multi)),
            expected(crate::operational_noise::relation::CanonicalLhsError::MultipleTerms)
        );
        let nonunit = PolynomialNF {
            exact_terms: [(first, BigInt::from(2_u8))].into_iter().collect(),
            bounded_summary: BoundedSummary::zero(),
        };
        assert_eq!(
            canonical_lhs_monomial(Some(&nonunit)),
            expected(crate::operational_noise::relation::CanonicalLhsError::NonUnitCoefficient)
        );
        let accepted = PolynomialNF {
            exact_terms: [(first, BigInt::from(1_u8))].into_iter().collect(),
            bounded_summary: BoundedSummary::zero(),
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

    fn descriptor_coefficient_multiset(
        normal_form: &PolynomialNF,
        monomials: &MonomialArena,
    ) -> BTreeSet<(Vec<ScopedExprId>, Vec<ScopedExprId>, BigInt)> {
        normal_form
            .exact_terms
            .iter()
            .map(|(monomial, coefficient)| {
                let descriptor = monomials.descriptor(*monomial).unwrap();
                (
                    descriptor.central_factors.to_vec(),
                    descriptor.ordered_factors.to_vec(),
                    coefficient.clone(),
                )
            })
            .collect()
    }

    #[test]
    fn forced_monomial_gc_preserves_exact_nf_bound_and_counters_at_node_commit() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let left = gaussian_factor(&mut expressions, matrix_type(), 81_001, 3);
        let middle = gaussian_factor(&mut expressions, matrix_type(), 81_002, 5);
        let right = gaussian_factor(&mut expressions, matrix_type(), 81_003, 7);
        let root = product(&mut expressions, &[left, middle, right]);
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);

        let (forced, forced_counters) = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
            normalizer.monomial_gc_allocation_threshold_bytes = 1;
            let value = normalizer.normalize(semantic).unwrap();
            (value, normalizer.counters())
        };
        let high_water_after_gc = monomials.len();
        assert!(monomials.occupied_len() < high_water_after_gc);
        for monomial in forced.exact_nf.as_ref().unwrap().exact_terms.keys() {
            assert!(monomials.descriptor(*monomial).is_ok(), "committed root NF must stay live");
        }

        let (second_forced, second_forced_counters) = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
            normalizer.monomial_gc_allocation_threshold_bytes = 1;
            let value = normalizer.normalize(semantic).unwrap();
            (value, normalizer.counters())
        };
        for monomial in forced.exact_nf.as_ref().unwrap().exact_terms.keys() {
            assert!(
                monomials.descriptor(*monomial).is_ok(),
                "a prior-call external result is protected by the next outer prefix"
            );
        }
        let (disabled, disabled_counters) = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
            normalizer.monomial_gc_allocation_threshold_bytes = u64::MAX;
            let value = normalizer.normalize(semantic).unwrap();
            (value, normalizer.counters())
        };
        assert_eq!(forced.exact_nf, second_forced.exact_nf);
        assert_eq!(forced.coefficient_bound, second_forced.coefficient_bound);
        assert_eq!(forced_counters, second_forced_counters);
        assert_eq!(forced.semantic, disabled.semantic);
        assert_eq!(forced.coefficient_bound, disabled.coefficient_bound);
        assert_eq!(forced.exact_nf, disabled.exact_nf);
        assert_eq!(forced_counters, disabled_counters);
        assert!(monomials.len() >= high_water_after_gc, "collected slots are never reused");
    }

    #[test]
    fn bounded_product_chain_compresses_without_retaining_exact_monomials() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let factors = (0_u64..128)
            .map(|index| gaussian_factor(&mut expressions, matrix_type(), 82_000 + index, 3))
            .collect::<Vec<_>>();
        let root = product(&mut expressions, &factors);
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let value = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
            normalizer.monomial_gc_allocation_threshold_bytes = 1;
            normalizer.normalize(semantic).unwrap()
        };
        let normal_form = value.exact_nf.as_ref().unwrap();
        assert!(normal_form.exact_terms.is_empty());
        assert!(matches!(
            normal_form.bounded_summary.coefficient_bound(),
            NumericContract::Known(CoefficientBound::Finite(_))
        ));
        assert!(monomials.len() >= 128, "slot high-water remains monotonic");
    }

    #[test]
    fn monomial_gc_rejects_foreign_local_cache_root_before_mutation() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let root_a = source_with(&mut expressions, matrix_type(), 83_011);
        let (_, mut arena_a, semantic_a) = setup(&mut expressions, &mut programs, root_a);
        let foreign = arena_a.intern(&expressions, &programs, &[], &[semantic_a]).unwrap();

        let root_b = source_with(&mut expressions, matrix_type(), 83_012);
        let (facts_b, mut arena_b, semantic_b) = setup(&mut expressions, &mut programs, root_b);
        let local_live = arena_b.intern(&expressions, &programs, &[], &[semantic_b]).unwrap();
        let local_dead =
            arena_b.intern(&expressions, &programs, &[], &[semantic_b, semantic_b]).unwrap();
        let occupied_before = arena_b.occupied_len();
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts_b, &mut arena_b).unwrap();
        normalizer.protected_monomial_prefix = 0;
        normalizer.monomial_gc_allocation_threshold_bytes = 1;
        normalizer.normalization_depth = 1;
        normalizer.insert_value_cache(
            semantic_b.expression(),
            Arc::new(AnalyzedValue {
                semantic: semantic_b,
                exact_nf: Some(Arc::new(PolynomialNF {
                    exact_terms: BTreeMap::from([(foreign, BigInt::from(1))]),
                    bounded_summary: BoundedSummary::zero(),
                })),
                coefficient_bound: NumericContract::Missing,
            }),
        );
        assert!(matches!(
            normalizer.sweep_monomials_at_node_commit(),
            Err(NormalizeError::Monomial(MonomialError::InvalidMonomialId { .. }))
        ));
        assert_eq!(normalizer.monomials.occupied_len(), occupied_before);
        assert!(normalizer.monomials.descriptor(local_live).is_ok());
        assert!(normalizer.monomials.descriptor(local_dead).is_ok());
    }

    #[test]
    fn monomial_gc_rejects_tombstoned_local_cache_root_before_mutation() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let root = source_with(&mut expressions, matrix_type(), 83_013);
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let live = monomials.intern(&expressions, &programs, &[], &[semantic]).unwrap();
        let tombstone =
            monomials.intern(&expressions, &programs, &[], &[semantic, semantic]).unwrap();
        monomials.sweep(0, [live]).unwrap();
        let later = monomials
            .intern(&expressions, &programs, &[], &[semantic, semantic, semantic])
            .unwrap();
        let occupied_before = monomials.occupied_len();
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        normalizer.protected_monomial_prefix = 0;
        normalizer.monomial_gc_allocation_threshold_bytes = 1;
        normalizer.normalization_depth = 1;
        normalizer.insert_value_cache(
            semantic.expression(),
            Arc::new(AnalyzedValue {
                semantic,
                exact_nf: Some(Arc::new(PolynomialNF {
                    exact_terms: BTreeMap::from([(tombstone, BigInt::from(1))]),
                    bounded_summary: BoundedSummary::zero(),
                })),
                coefficient_bound: NumericContract::Missing,
            }),
        );
        assert!(matches!(
            normalizer.sweep_monomials_at_node_commit(),
            Err(NormalizeError::Monomial(MonomialError::CollectedMonomialId { .. }))
        ));
        assert_eq!(normalizer.monomials.occupied_len(), occupied_before);
        assert!(normalizer.monomials.descriptor(live).is_ok());
        assert!(normalizer.monomials.descriptor(later).is_ok());
    }

    #[test]
    fn monomial_gc_threshold_and_depth_gates_are_deterministic() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let root = source_with(&mut expressions, matrix_type(), 83_003);
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let live = monomials.intern(&expressions, &programs, &[], &[semantic]).unwrap();
        let dead = monomials.intern(&expressions, &programs, &[], &[semantic, semantic]).unwrap();
        let allocated = monomials.allocated_payload_since_sweep();
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        normalizer.protected_monomial_prefix = 0;
        normalizer.insert_value_cache(
            semantic.expression(),
            Arc::new(AnalyzedValue {
                semantic,
                exact_nf: Some(Arc::new(PolynomialNF {
                    exact_terms: BTreeMap::from([(live, BigInt::from(1))]),
                    bounded_summary: BoundedSummary::zero(),
                })),
                coefficient_bound: NumericContract::Missing,
            }),
        );
        normalizer.normalization_depth = 1;
        normalizer.monomial_gc_allocation_threshold_bytes = allocated.saturating_add(1);
        normalizer.sweep_monomials_at_node_commit().unwrap();
        assert!(normalizer.monomials.descriptor(dead).is_ok(), "below threshold is a no-op");
        normalizer.monomial_gc_allocation_threshold_bytes = 1;
        normalizer.normalization_depth = 2;
        normalizer.sweep_monomials_at_node_commit().unwrap();
        assert!(normalizer.monomials.descriptor(dead).is_ok(), "nested depth is a no-op");
        normalizer.normalization_depth = 1;
        normalizer.sweep_monomials_at_node_commit().unwrap();
        assert!(normalizer.monomials.descriptor(live).is_ok());
        assert!(matches!(
            normalizer.monomials.descriptor(dead),
            Err(MonomialError::CollectedMonomialId { .. })
        ));
    }

    #[test]
    fn one_scope_proof_serves_all_atoms_and_scoped_derivations_in_one_root() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let matrix = matrix_type();
        let uniform = source_with(&mut expressions, matrix.clone(), 701);
        let semantic_source =
            matrix_source(&mut expressions, "scope-proof-source", matrix.clone(), None);
        let gaussian = gaussian_factor(&mut expressions, matrix.clone(), 702, 3);
        let preimage = preimage_factor(&mut expressions, matrix, 703, 5);
        let atoms = [uniform, semantic_source, gaussian, preimage];
        let mut root = atoms[0];
        for atom in atoms.iter().copied().cycle().skip(1).take(31) {
            root =
                expressions.intern_matrix_transform(MatrixOperation::Add, &[root, atom]).unwrap();
        }
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);

        // Exercise expressions interned after the program was finalized. They remain under the
        // same non-forgeable proof authority and must not trigger one proof build per atom.
        let mut proof = expressions.scope_proof(semantic.program(), semantic.expression()).unwrap();
        let mut derived = semantic;
        for _ in 0..32 {
            derived = expressions
                .intern_scoped_transform(
                    &mut proof,
                    ValueOperator::Matrix(MatrixOperation::Negate),
                    &[derived],
                )
                .unwrap();
        }
        expressions.reset_scope_proof_build_count();

        let value = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(derived)
            .unwrap();
        assert_eq!(expressions.scope_proof_build_count(), 1);
        assert_eq!(value.coefficient_bound, NumericContract::Missing);
        let normal_form = value.exact_nf.unwrap();
        assert_eq!(normal_form.exact_terms.len(), 2);
        assert!(
            normal_form.exact_terms.values().all(|coefficient| *coefficient == BigInt::from(8))
        );
        let retained = normal_form
            .exact_terms
            .keys()
            .map(|monomial| {
                let descriptor = monomials.descriptor(*monomial).unwrap();
                assert!(descriptor.central_factors.is_empty());
                assert_eq!(descriptor.ordered_factors.len(), 1);
                descriptor.ordered_factors[0].expression()
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(retained, [uniform, semantic_source].into_iter().collect());
        assert!(matches!(
            expressions.node(uniform).unwrap().operator,
            ValueOperator::Sampler { event: SampleEventId(701), .. }
        ));
        assert!(matches!(
            expressions.node(gaussian).unwrap().operator,
            ValueOperator::Sampler {
                event: SampleEventId(702),
                operation: SamplerOperation::Gaussian { .. }
            }
        ));
        assert!(matches!(
            expressions.node(preimage).unwrap().operator,
            ValueOperator::Sampler {
                event: SampleEventId(703),
                operation: SamplerOperation::Preimage { .. }
            }
        ));
        assert!(matches!(
            expressions.node(semantic_source).unwrap().operator,
            ValueOperator::Source(ref identity)
                if identity.stable_definition == "scope-proof-source"
        ));
    }

    #[test]
    fn specialized_root_reuses_one_owned_scope_proof_without_semantic_drift() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let left = source_with(&mut expressions, matrix_type(), 710);
        let right = source_with(&mut expressions, matrix_type(), 711);
        let root =
            expressions.intern_matrix_transform(MatrixOperation::Subtract, &[left, right]).unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let mut transform_proof =
            expressions.scope_proof(semantic.program(), semantic.expression()).unwrap();
        let transformed = expressions
            .intern_scoped_transform(
                &mut transform_proof,
                ValueOperator::Matrix(MatrixOperation::Negate),
                &[semantic],
            )
            .unwrap();

        let public = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
            normalizer.normalize(transformed).unwrap()
        };
        expressions.reset_scope_proof_build_count();
        let specialized = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
            let value = normalizer.normalize_specialized_root(transformed.expression()).unwrap();
            value
        };
        assert_eq!(expressions.scope_proof_build_count(), 1);
        assert_eq!(specialized.semantic, public.semantic);
        assert_eq!(specialized.exact_nf, public.exact_nf);
        assert_eq!(specialized.coefficient_bound, public.coefficient_bound);
    }

    #[test]
    fn specialized_relation_suppression_restores_after_error() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let body = source_with(&mut expressions, matrix_type(), 712);
        let (facts, mut monomials, _semantic) = setup(&mut expressions, &mut programs, body);
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let invalid = ExprId::new(ArenaToken::fresh(), 1);
        assert!(normalizer.normalize_specialized_root_without_relations(invalid).is_err());
        assert!(!normalizer.suppress_product_relation_closure);
    }

    #[test]
    fn universal_specialization_isolated_from_outer_compact_gadget_holds() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let domain = FamilyDomain::new(0, 1).unwrap();
        let range = TrustedIndexRange::new(0, 1).unwrap();
        let input_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 1).unwrap();
        let gadget_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 3).unwrap();
        let decomposition_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 3, 1).unwrap();
        let gadget = matrix_source(
            &mut expressions,
            "specialized-gadget-1",
            gadget_type.clone(),
            Some((2, false)),
        );
        let gadget_two = matrix_source(
            &mut expressions,
            "specialized-gadget-2",
            gadget_type.clone(),
            Some((2, false)),
        );
        let input = matrix_source(&mut expressions, "specialized-input", input_type.clone(), None);
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
        let gadget_sum = expressions
            .intern_matrix_transform(MatrixOperation::Add, &[gadget, gadget_two])
            .unwrap();
        let public_root = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[gadget_sum, decomposition])
            .unwrap();
        let public_family =
            programs.generated_family_from_body(&mut expressions, domain, public_root).unwrap();
        let preimage_family =
            programs.generated_family_from_body(&mut expressions, domain, input).unwrap();
        let target_body =
            matrix_source(&mut expressions, "specialized-target", input_type.clone(), None);
        let target_family =
            programs.generated_family_from_body(&mut expressions, domain, target_body).unwrap();
        let trapdoor_root = expressions
            .intern(
                ValueOperator::Trapdoor(super::super::arena::TrapdoorOperation::Generate {
                    descriptor: "specialized-gadget-trapdoor".to_owned(),
                    parameters: Box::new([]),
                    paired_public_event: SampleEventId(991),
                    paired_public_output_role: "value".to_owned(),
                }),
                Box::new([]),
            )
            .unwrap();
        let trapdoor_family =
            programs.generated_family_from_body(&mut expressions, domain, trapdoor_root).unwrap();
        let index = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let scope_root = expressions
            .intern(ValueOperator::Matrix(MatrixOperation::Scale), Box::new([public_root, index]))
            .unwrap();
        let scope_family =
            programs.generated_family_from_body(&mut expressions, domain, scope_root).unwrap();
        let scope = scope_family.program();
        let source =
            SamplerSourceContract { expression: programs.family_body(preimage_family).unwrap() };
        let trapdoor = TrapdoorSourceContract { expression: trapdoor_root };
        let dispatch = UniversalDispatchKey {
            preimage_family,
            preimage_source: source.clone(),
            matrix_type: input_type.clone(),
            trapdoor_source: trapdoor.clone(),
        };
        let validation = || RelationValidationAuthority {
            source: source.clone(),
            trapdoor_source: trapdoor.clone(),
            matrix_type: input_type.clone(),
            public_type: ResolvedValueType::Matrix(input_type.clone()),
            preimage_type: ResolvedValueType::Matrix(input_type.clone()),
            target_type: ResolvedValueType::Matrix(input_type.clone()),
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
                public_plan: public_family.program(),
                preimage_plan: preimage_family.program(),
                trapdoor_plan: trapdoor_family.program(),
                public_pairing: public_family.program(),
                layout: None,
                factor_order: FactorOrderContract::ordered_public_preimage(),
                validation: validation(),
            },
            target_plan: target_family.program(),
        };
        let mut relations = RelationRegistry::new();
        relations.register_universal(registration).unwrap();
        relations.freeze();
        let registry = recomposition_registry(
            gadget_type.clone(),
            decomposition_type.clone(),
            input_type.clone(),
            false,
            3,
        );
        let mut facts = FactStore::new(&expressions);
        facts.finalize_ranges();
        let mut monomials = MonomialArena::new(&expressions, &programs, scope).unwrap();
        let mut cache = NormalizationCache::new();
        let index = programs.scoped(&expressions, scope, index).unwrap();
        let outer_proof = expressions.scope_proof(scope, scope_root).unwrap();
        let outer_input = expressions.scoped_from_proof(&outer_proof, input).unwrap();
        let outer_rule = registry
            .matching_rule(
                2,
                false,
                3,
                &gadget_type,
                &decomposition_type,
                &input_type,
                &input_type,
                Some(&MatrixLayout::row_major(gadget_type.rows, gadget_type.columns)),
                Some(&MatrixLayout::row_major(decomposition_type.rows, decomposition_type.columns)),
                Some(&MatrixLayout::row_major(input_type.rows, input_type.columns)),
            )
            .unwrap();
        let mut outer_plan = CompactShellPlan::default();
        outer_plan.preflight_node_occurrences = 3;
        outer_plan
            .gadget_shells
            .insert((decomposition, input), CompactGadgetPlan { rule: outer_rule, occurrences: 1 });
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_relations(&relations, &mut cache)
            .with_gadget_recompositions(&registry);
        normalizer.compact_mode = true;
        normalizer.compact_plan = Some(outer_plan);
        let outer_input_nf = normalizer.atom_nf(&outer_proof, outer_input).unwrap();
        normalizer.gadget_input_nfs.insert(
            input,
            CompactGadgetHold {
                normal_form: Arc::new(outer_input_nf),
                remaining: 1,
                consumers: BTreeMap::from([(decomposition, 1)]),
            },
        );
        let outer_plan_before = (
            normalizer.compact_plan.as_ref().unwrap().preflight_node_occurrences,
            normalizer
                .compact_plan
                .as_ref()
                .unwrap()
                .gadget_shells
                .iter()
                .map(|((shell, held_input), plan)| {
                    (*shell, *held_input, plan.rule.clone(), plan.occurrences)
                })
                .collect::<Vec<_>>(),
        );
        let outer_hold_before = normalizer
            .gadget_input_nfs
            .get(&input)
            .map(|hold| (hold.remaining, hold.consumers.clone()));
        let specialized = normalizer
            .specialize_universal(&dispatch, index, range)
            .expect("specialized relation with gadget product");
        assert!(!specialized.is_empty());
        assert!(!normalizer.suppress_product_relation_closure);
        assert!(normalizer.compact_mode);
        assert_eq!(
            normalizer.compact_plan.as_ref().map(|plan| (
                plan.preflight_node_occurrences,
                plan.gadget_shells
                    .iter()
                    .map(|((shell, held_input), plan)| {
                        (*shell, *held_input, plan.rule.clone(), plan.occurrences)
                    })
                    .collect::<Vec<_>>()
            )),
            Some(outer_plan_before.clone())
        );
        assert_eq!(
            normalizer
                .gadget_input_nfs
                .get(&input)
                .map(|hold| (hold.remaining, hold.consumers.clone())),
            outer_hold_before
        );
        assert!(normalizer.compact_scalar_holds.is_empty());
        assert_eq!(normalizer.counters, NormalizationCounters::default());
        let rejected = normalizer.specialize_universal(
            &dispatch,
            index,
            TrustedIndexRange::new(1, 2).unwrap(),
        );
        assert!(matches!(
            rejected,
            Err(NormalizeError::Relation(RelationRegistryError::IndexOutOfDomain))
        ));
        assert!(!normalizer.suppress_product_relation_closure);
        assert!(normalizer.compact_mode);
        assert_eq!(
            normalizer.compact_plan.as_ref().map(|plan| (
                plan.preflight_node_occurrences,
                plan.gadget_shells
                    .iter()
                    .map(|((shell, held_input), plan)| {
                        (*shell, *held_input, plan.rule.clone(), plan.occurrences)
                    })
                    .collect::<Vec<_>>()
            )),
            Some(outer_plan_before)
        );
        assert_eq!(
            normalizer
                .gadget_input_nfs
                .get(&input)
                .map(|hold| (hold.remaining, hold.consumers.clone())),
            outer_hold_before
        );
        drop(normalizer);

        let (compact_lhs_signature, compact_rhs_signatures) = {
            let factor_signature = |factor: ScopedExprId| {
                (
                    factor.expression(),
                    format!("{:?}", expressions.node(factor.expression()).unwrap().operator),
                    format!("{:?}", expressions.value_type(factor.expression()).unwrap()),
                )
            };
            let lhs_signature = |map: &BTreeMap<CanonicalLhsKey, BTreeSet<CanonicalRhsId>>,
                                 arena: &MonomialArena| {
                map.keys()
                    .map(|key| {
                        let descriptor = arena.descriptor(key.monomial).unwrap();
                        (
                            BigInt::from(1_u8),
                            descriptor
                                .central_factors
                                .iter()
                                .copied()
                                .map(factor_signature)
                                .collect::<Vec<_>>(),
                            descriptor
                                .ordered_factors
                                .iter()
                                .copied()
                                .map(factor_signature)
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>()
            };
            let polynomial_signature = |normal_form: &PolynomialNF, arena: &MonomialArena| {
                (
                    normal_form
                        .exact_terms
                        .iter()
                        .map(|(monomial, coefficient)| {
                            let descriptor = arena.descriptor(*monomial).unwrap();
                            (
                                coefficient.clone(),
                                descriptor
                                    .central_factors
                                    .iter()
                                    .copied()
                                    .map(factor_signature)
                                    .collect::<Vec<_>>(),
                                descriptor
                                    .ordered_factors
                                    .iter()
                                    .copied()
                                    .map(factor_signature)
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect::<Vec<_>>(),
                    normal_form.bounded_summary.clone(),
                )
            };
            let lhs = lhs_signature(&specialized, &monomials);
            let rhs = specialized
                .values()
                .flatten()
                .map(|rhs| polynomial_signature(cache.get(*rhs).unwrap(), &monomials))
                .collect::<Vec<_>>();
            (lhs, rhs)
        };
        let mut eager_monomials = MonomialArena::new(&expressions, &programs, scope).unwrap();
        let mut eager_cache = NormalizationCache::new();
        let mut eager = Normalizer::new(&mut expressions, &programs, &facts, &mut eager_monomials)
            .unwrap()
            .with_relations(&relations, &mut eager_cache)
            .with_gadget_recompositions(&registry);
        let eager_specialized = eager
            .specialize_universal(&dispatch, index, range)
            .expect("forced-eager specialized relation with gadget product");
        let eager_relation_applied = eager.counters.relation_applied;
        drop(eager);
        let (eager_lhs_signature, eager_rhs_signatures) = {
            let factor_signature = |factor: ScopedExprId| {
                (
                    factor.expression(),
                    format!("{:?}", expressions.node(factor.expression()).unwrap().operator),
                    format!("{:?}", expressions.value_type(factor.expression()).unwrap()),
                )
            };
            let lhs_signature = |map: &BTreeMap<CanonicalLhsKey, BTreeSet<CanonicalRhsId>>,
                                 arena: &MonomialArena| {
                map.keys()
                    .map(|key| {
                        let descriptor = arena.descriptor(key.monomial).unwrap();
                        (
                            BigInt::from(1_u8),
                            descriptor
                                .central_factors
                                .iter()
                                .copied()
                                .map(factor_signature)
                                .collect::<Vec<_>>(),
                            descriptor
                                .ordered_factors
                                .iter()
                                .copied()
                                .map(factor_signature)
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>()
            };
            let polynomial_signature = |normal_form: &PolynomialNF, arena: &MonomialArena| {
                (
                    normal_form
                        .exact_terms
                        .iter()
                        .map(|(monomial, coefficient)| {
                            let descriptor = arena.descriptor(*monomial).unwrap();
                            (
                                coefficient.clone(),
                                descriptor
                                    .central_factors
                                    .iter()
                                    .copied()
                                    .map(factor_signature)
                                    .collect::<Vec<_>>(),
                                descriptor
                                    .ordered_factors
                                    .iter()
                                    .copied()
                                    .map(factor_signature)
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect::<Vec<_>>(),
                    normal_form.bounded_summary.clone(),
                )
            };
            let lhs = lhs_signature(&eager_specialized, &eager_monomials);
            let rhs = eager_specialized
                .values()
                .flatten()
                .map(|rhs| polynomial_signature(eager_cache.get(*rhs).unwrap(), &eager_monomials))
                .collect::<Vec<_>>();
            (lhs, rhs)
        };
        assert_eq!(specialized.len(), eager_specialized.len());
        assert_eq!(compact_lhs_signature, eager_lhs_signature);
        assert_eq!(compact_rhs_signatures, eager_rhs_signatures);
        assert_eq!(eager_relation_applied, 0);
    }

    #[test]
    fn compact_runtime_counts_nested_program_call_visits_separately_from_preflight() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let matrix = matrix_type();
        let domain = FamilyDomain::new(0, 1).unwrap();
        // Families are intentionally one-index programs.  The shared input below is the same
        // Argument(0) expression on both reducible nested calls; compact semantic binding must
        // account for each actual Visit even though the argument is shared.
        let inner_body = source_with(&mut expressions, matrix.clone(), 720);
        let inner_family =
            programs.generated_family_from_body(&mut expressions, domain, inner_body).unwrap();
        let outer_argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let first_call =
            programs.call(&mut expressions, inner_family.program(), &[outer_argument]).unwrap();
        let second_call =
            programs.call(&mut expressions, inner_family.program(), &[outer_argument]).unwrap();
        let outer_body = expressions
            .intern_matrix_transform(MatrixOperation::Add, &[first_call, second_call])
            .unwrap();
        let outer_family =
            programs.generated_family_from_body(&mut expressions, domain, outer_body).unwrap();
        let shared_input = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(0)), Box::new([]))
            .unwrap();
        let root_call =
            programs.call(&mut expressions, outer_family.program(), &[shared_input]).unwrap();
        let wrapper = programs
            .finalize(
                &mut expressions,
                ProgramSignature {
                    inputs: Box::new([]),
                    output: programs.family_element_type(outer_family).unwrap(),
                },
                root_call,
            )
            .unwrap();
        let scoped = programs.root(&expressions, wrapper).unwrap();
        let facts = FactStore::new(&expressions);
        let mut monomials = MonomialArena::new(&expressions, &programs, wrapper).unwrap();
        let mut plan = CompactShellPlan::default();
        // The preflight count includes each incoming edge to the two nested calls and their
        // shared Argument(0), while runtime counts every evaluator Visit, including the nested
        // semantic-binding walks.  Keep this structural count private and deliberately separate.
        plan.preflight_node_occurrences = 7;
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_compact_shell_plan(plan);
        normalizer.compact_mode = true;
        let value = normalizer.normalize_compact_root(scoped).unwrap();
        let counters = normalizer.counters();
        assert!(counters.nodes_processed > 0);
        assert_eq!(counters.nodes_total, counters.nodes_processed);
        assert_eq!(normalizer.compact_plan.as_ref().unwrap().preflight_node_occurrences, 7);
        assert_eq!(counters.compact_live_frames, 0);
        assert_eq!(counters.compact_live_values, 0);
        assert!(value.exact_nf.is_some());
    }

    #[test]
    fn compact_shared_diamond_revisits_occurrences_with_depth_bounded_live_state() {
        const DEPTH: usize = 16;
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let mut root = source_with(&mut expressions, matrix_type(), 721);
        for level in 0..DEPTH {
            let left_leaf = source_with(&mut expressions, matrix_type(), 10_000 + level as u64 * 2);
            let right_leaf =
                source_with(&mut expressions, matrix_type(), 10_001 + level as u64 * 2);
            let left = expressions
                .intern_matrix_transform(MatrixOperation::Add, &[root, left_leaf])
                .unwrap();
            let right = expressions
                .intern_matrix_transform(MatrixOperation::Add, &[root, right_leaf])
                .unwrap();
            root =
                expressions.intern_matrix_transform(MatrixOperation::Add, &[left, right]).unwrap();
        }
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let family = programs.family_for_program(semantic.program()).unwrap();
        let mut plan = CompactShellPlan::default();
        plan.preflight_node_occurrences = 6 * (1_u64 << DEPTH) - 5;
        let planned_occurrences = plan.preflight_node_occurrences;
        let mut compact = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_compact_shell_plan(plan);
        compact.compact_mode = true;
        let compact_value = compact.normalize_compact_family_root(family).unwrap();
        let compact_counters = compact.counters();
        assert_eq!(compact_counters.compact_memo_entries, 0);
        assert_eq!(compact_counters.compact_peak_memo_entries, 0);
        assert_eq!(compact_counters.nodes_processed, planned_occurrences);
        assert!(
            compact_counters.compact_peak_live_frames <= 4 * DEPTH as u64 + 1,
            "peak frames={} depth={DEPTH}",
            compact_counters.compact_peak_live_frames
        );
        assert!(
            compact_counters.compact_peak_live_values <= 4 * DEPTH as u64 + 1,
            "peak values={} depth={DEPTH}",
            compact_counters.compact_peak_live_values
        );
        assert_eq!(compact_counters.compact_live_frames, 0);
        assert_eq!(compact_counters.compact_live_values, 0);
        assert_eq!(compact_counters.nodes_total, compact_counters.nodes_processed);
        drop(compact);

        let eager_value = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(semantic)
            .unwrap();
        assert_eq!(compact_value.exact_nf, eager_value.exact_nf);
        assert_eq!(compact_value.coefficient_bound, eager_value.coefficient_bound);
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

    fn product(expressions: &mut ExprArena, factors: &[ExprId]) -> ExprId {
        let (&first, rest) = factors.split_first().expect("non-empty product");
        rest.iter().copied().fold(first, |left, right| {
            expressions.intern_matrix_transform(MatrixOperation::Multiply, &[left, right]).unwrap()
        })
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
        let mut expected = vec![expected_central, expected_ordered];
        expected.sort_unstable();
        assert_eq!(descriptor.central_factors.as_ref(), expected.as_slice());
        assert!(descriptor.ordered_factors.is_empty());
    }

    #[test]
    fn nested_explicit_element_uses_branch_max_and_folds_without_changing_exact_identity() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let matrix = matrix_type();
        let left = source_with(&mut expressions, matrix.clone(), 242);
        let right = source_with(&mut expressions, matrix, 243);
        let mut facts = FactStore::new(&expressions);
        insert_matrix_bound(&mut facts, &expressions, left, 3);
        insert_matrix_bound(&mut facts, &expressions, right, 7);
        let domain = super::super::arena::FamilyDomain::new(0, 2).unwrap();
        let selector = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let explicit = expressions
            .intern(
                ValueOperator::ExplicitElement {
                    domain,
                    element_type: ResolvedValueType::Matrix(matrix_type()),
                },
                Box::new([selector, left, right]),
            )
            .unwrap();
        let explicit_node = expressions.node(explicit).unwrap();
        assert_eq!(explicit_node.inputs.as_ref(), &[selector, left, right]);
        assert!(matches!(
            explicit_node.operator,
            ValueOperator::ExplicitElement { domain: actual, .. } if actual == domain
        ));
        let root =
            expressions.intern_matrix_transform(MatrixOperation::Negate, &[explicit]).unwrap();
        let family = programs.generated_family_from_body(&mut expressions, domain, root).unwrap();
        assert_ne!(programs.family_body(family).unwrap(), explicit);
        let semantic = programs.scoped(&expressions, family.program(), root).unwrap();
        let explicit_semantic = programs.scoped(&expressions, family.program(), explicit).unwrap();
        facts.finalize_ranges();
        let mut monomials = MonomialArena::new(&expressions, &programs, family.program()).unwrap();

        let baseline = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
            normalizer.normalize(semantic).unwrap()
        };
        assert_eq!(
            baseline.coefficient_bound,
            NumericContract::Known(CoefficientBound::finite(7_u8))
        );
        let baseline_nf = baseline.exact_nf.unwrap();
        assert_eq!(baseline_nf.exact_terms.len(), 1);
        let baseline_monomial = *baseline_nf.exact_terms.keys().next().unwrap();
        assert_eq!(baseline_nf.exact_terms[&baseline_monomial], BigInt::from(-1));
        let descriptor = monomials.descriptor(baseline_monomial).unwrap();
        assert!(descriptor.central_factors.is_empty());
        assert_eq!(descriptor.ordered_factors.as_ref(), &[explicit_semantic]);

        let mut relations = RelationRegistry::new();
        relations.freeze();
        let mut cache = NormalizationCache::new();
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_relations(&relations, &mut cache);
        let folded = normalizer.normalize(semantic).unwrap();
        assert_eq!(
            folded.coefficient_bound,
            NumericContract::Known(CoefficientBound::finite(7_u8))
        );
        assert!(folded.exact_nf.unwrap().exact_terms.is_empty());
        assert_eq!(normalizer.counters().bounded_fold_count, 1);
        assert_eq!(normalizer.counters().final_exact_term_count, 0);
        assert_eq!(normalizer.counters().relation_remaining, 0);
    }

    #[test]
    fn explicit_element_branch_bound_transfer_is_fail_closed_and_respects_fact_precedence() {
        let run = |left_bound: Option<CoefficientBound>,
                   right_bound: Option<CoefficientBound>,
                   explicit_fact: Option<CoefficientBound>| {
            let mut expressions = ExprArena::new();
            let mut programs = ProgramArena::new();
            let matrix = matrix_type();
            let left = matrix_source(&mut expressions, "explicit-left", matrix.clone(), None);
            let right = matrix_source(&mut expressions, "explicit-right", matrix, None);
            let mut facts = FactStore::new(&expressions);
            for (expression, bound) in [(left, left_bound), (right, right_bound)] {
                if let Some(bound) = bound {
                    let ResolvedValueType::Matrix(matrix) =
                        expressions.value_type(expression).unwrap()
                    else {
                        panic!("explicit branch must be a matrix")
                    };
                    let mut matrix_facts = MatrixFacts::new(
                        matrix.clone(),
                        MatrixMetadata::new(MatrixLayout::row_major(matrix.rows, matrix.columns)),
                    );
                    matrix_facts.coefficient_bound = NumericContract::Known(bound);
                    facts
                        .insert(&expressions, expression, ValueFacts::Matrix(matrix_facts))
                        .unwrap();
                }
            }
            let domain = super::super::arena::FamilyDomain::new(0, 2).unwrap();
            let selector = if explicit_fact.is_some() {
                expressions
                    .intern(ValueOperator::Constant(TypedConstant::int(0)), Box::new([]))
                    .unwrap()
            } else {
                expressions.intern_argument(0, ResolvedValueType::Int).unwrap()
            };
            let explicit = expressions
                .intern(
                    ValueOperator::ExplicitElement {
                        domain,
                        element_type: ResolvedValueType::Matrix(matrix_type()),
                    },
                    Box::new([selector, left, right]),
                )
                .unwrap();
            if let Some(bound) = explicit_fact {
                let ResolvedValueType::Matrix(matrix) = expressions.value_type(explicit).unwrap()
                else {
                    panic!("explicit value must be a matrix")
                };
                let mut matrix_facts = MatrixFacts::new(
                    matrix.clone(),
                    MatrixMetadata::new(MatrixLayout::row_major(matrix.rows, matrix.columns)),
                );
                matrix_facts.coefficient_bound = NumericContract::Known(bound);
                facts.insert(&expressions, explicit, ValueFacts::Matrix(matrix_facts)).unwrap();
            }
            let family =
                programs.generated_family_from_body(&mut expressions, domain, explicit).unwrap();
            let semantic = programs.scoped(&expressions, family.program(), explicit).unwrap();
            facts.finalize_ranges();
            let mut monomials =
                MonomialArena::new(&expressions, &programs, family.program()).unwrap();
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
            normalizer.normalize(semantic).unwrap().coefficient_bound
        };

        assert_eq!(
            run(Some(CoefficientBound::finite(3_u8)), Some(CoefficientBound::finite(7_u8)), None,),
            NumericContract::Known(CoefficientBound::finite(7_u8))
        );
        assert_eq!(run(Some(CoefficientBound::finite(3_u8)), None, None), NumericContract::Missing);
        assert_eq!(
            run(Some(CoefficientBound::finite(3_u8)), Some(CoefficientBound::Large), None,),
            NumericContract::Known(CoefficientBound::Large)
        );
        assert_eq!(
            run(Some(CoefficientBound::ExactZero), Some(CoefficientBound::ExactZero), None,),
            NumericContract::Known(CoefficientBound::ExactZero)
        );
        assert_eq!(
            run(
                Some(CoefficientBound::finite(3_u8)),
                Some(CoefficientBound::finite(7_u8)),
                Some(CoefficientBound::finite(11_u8)),
            ),
            NumericContract::Known(CoefficientBound::finite(11_u8))
        );
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
    fn compact_shell_plan_deduplicates_rule_and_counts_shared_occurrences() {
        let input_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 1).unwrap();
        let decomposition_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 3, 1).unwrap();
        let gadget_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 3).unwrap();
        let rule = GadgetRecompositionRule {
            base: 2,
            small: false,
            digit_count: 3,
            gadget_type,
            decomposition_type,
            input_type: input_type.clone(),
            output_type: input_type,
            gadget_layout: Some(MatrixLayout::row_major(1, 3)),
            decomposition_layout: Some(MatrixLayout::row_major(3, 1)),
            input_layout: Some(MatrixLayout::row_major(1, 1)),
        };
        let mut plan = CompactShellPlan::default();
        assert!(plan.insert_gadget(
            ExprId::new(ArenaToken::fresh(), 1),
            ExprId::new(ArenaToken::fresh(), 2),
            rule.clone()
        ));
        // Same shell/input identity is one interned rule with two direct product consumers.
        let shell = plan.gadget_shells.keys().next().unwrap().0;
        let input = plan.gadget_shells.keys().next().unwrap().1;
        assert!(plan.insert_gadget(shell, input, rule));
        assert_eq!(plan.gadget_shells.len(), 1);
        assert_eq!(plan.gadget_occurrences(), 2);
    }

    #[test]
    fn compact_scalar_plan_rejects_unmatched_and_overconsumed_tokens() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let scalar_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap();
        let scalar = source_with(&mut expressions, scalar_type.clone(), 401);
        let large = source(&mut expressions);
        let product = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[large, scalar])
            .unwrap();
        let (mut facts, mut monomials, _root) = setup(&mut expressions, &mut programs, product);
        let mut scalar_facts = MatrixFacts::new(
            scalar_type.clone(),
            MatrixMetadata::new(MatrixLayout::row_major(1, 1)),
        );
        scalar_facts.coefficient_bound =
            NumericContract::Known(CoefficientBound::finite(BigUint::from(1_u8)));
        facts.insert(&expressions, scalar, ValueFacts::Matrix(scalar_facts)).unwrap();
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let mut plan = CompactShellPlan::default();
        assert!(plan.insert_scalar(scalar, scalar_type.clone()));
        normalizer.compact_plan_remaining = Some(plan);
        normalizer.compact_mode = true;
        assert!(normalizer.consume_compact_scalar_plan(scalar, &scalar_type).is_ok());
        assert!(matches!(
            normalizer.consume_compact_scalar_plan(scalar, &scalar_type),
            Err(NormalizeError::InvalidExactPlan { .. })
        ));
        assert!(matches!(
            normalizer.release_compact_scalar_hold(scalar),
            Err(NormalizeError::InvalidExactPlan { .. })
        ));

        // Consumer identity is part of the private token: a call shared by two
        // product kinds cannot be consumed through a different product or side.
        let mut program_plan = CompactShellPlan::default();
        assert!(program_plan.insert_scalar_program_call(
            product,
            scalar,
            false,
            scalar_type.clone()
        ));
        normalizer.compact_plan_remaining = Some(program_plan);
        assert!(matches!(
            normalizer.consume_compact_scalar_program_call(large, scalar, false, &scalar_type),
            Err(NormalizeError::InvalidExactPlan {
                reason: "compact scalar ProgramCall consumer is unplanned"
            })
        ));
        assert!(matches!(
            normalizer.consume_compact_scalar_program_call(product, scalar, true, &scalar_type),
            Err(NormalizeError::InvalidExactPlan {
                reason: "compact scalar ProgramCall consumer is unplanned"
            })
        ));
        assert_eq!(
            normalizer
                .compact_plan_remaining
                .as_ref()
                .unwrap()
                .scalar_program_calls
                .get(&(product, scalar, false))
                .unwrap()
                .occurrences,
            1
        );
        assert!(
            normalizer
                .consume_compact_scalar_program_call(product, scalar, false, &scalar_type)
                .is_ok()
        );
    }

    #[test]
    fn compact_both_scalar_action_rejects_non_exact_child_without_scalar_hold() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let left_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 2).unwrap();
        let right_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 1).unwrap();
        let scalar_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap();
        let left = source_with(&mut expressions, left_type, 1401);
        let right = source_with(&mut expressions, right_type, 1402);
        let non_exact_scalar =
            expressions.intern_matrix_transform(MatrixOperation::Multiply, &[left, right]).unwrap();
        let scalar = source_with(&mut expressions, scalar_type, 1403);
        let body = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[non_exact_scalar, scalar])
            .unwrap();
        let family = programs
            .generated_family_from_body(&mut expressions, FamilyDomain::new(0, 1).unwrap(), body)
            .unwrap();
        let facts = FactStore::new(&expressions);
        let mut monomials = MonomialArena::new(&expressions, &programs, family.program()).unwrap();
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_compact_shell_plan(CompactShellPlan::default());
        normalizer.compact_mode = true;
        let error = normalizer
            .normalize_compact_family_root(family)
            .expect_err("non-scalar factors cannot enter both-scalar centralization");
        assert!(matches!(
            error,
            NormalizeError::InvalidExactPlan { reason: "compact scalar action was not exact" }
        ));
        assert_eq!(normalizer.counters.compact_scalar_consumers, 0);
        assert_eq!(normalizer.counters.compact_scalar_holds_released, 0);
        assert_eq!(normalizer.counters.compact_scalar_holds_unmatched, 0);
    }

    #[test]
    fn compact_product_context_records_each_shell_once_per_boundary() {
        let shell = ExprId::new(ArenaToken::fresh(), 1);
        let input = ExprId::new(ArenaToken::fresh(), 2);
        let mut context = CompactProductContext::default();
        assert!(context.used_shells.insert((shell, input)));
        assert!(!context.used_shells.insert((shell, input)));
        assert_eq!(context.used_shells.len(), 1);
    }

    #[test]
    fn compact_product_context_error_pops_without_committing_shells() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let input_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 1).unwrap();
        let input = source_with(&mut expressions, input_type.clone(), 1501);
        let product = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[input, input])
            .unwrap();
        let (facts, mut monomials, _) = setup(&mut expressions, &mut programs, product);
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let shell = ExprId::new(ArenaToken::fresh(), 1);
        let held_input = ExprId::new(ArenaToken::fresh(), 2);
        normalizer.compact_mode = true;
        normalizer.compact_plan_remaining = Some(CompactShellPlan::default());
        let plan_shape = |plan: &CompactShellPlan| {
            (
                plan.gadget_shells.len(),
                plan.scalar_factors.len(),
                plan.scalar_program_calls.len(),
                plan.preflight_node_occurrences,
            )
        };
        let before_plan = normalizer.compact_plan_remaining.as_ref().map(plan_shape);
        let error: Result<(), NormalizeError> =
            normalizer.with_compact_product_context(|normalizer| {
                normalizer
                    .compact_product_contexts
                    .last_mut()
                    .unwrap()
                    .used_shells
                    .insert((shell, held_input));
                Err(NormalizeError::InvalidExactPlan { reason: "injected product failure" })
            });
        assert!(matches!(
            error,
            Err(NormalizeError::InvalidExactPlan { reason: "injected product failure" })
        ));
        assert!(normalizer.compact_product_contexts.is_empty());
        assert_eq!(normalizer.compact_plan_remaining.as_ref().map(plan_shape), before_plan);
        assert_eq!(normalizer.counters.compact_shell_holds_released, 0);
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
    fn tall_shaped_large_gadget_plus_noise_rewrites_and_compresses_immediately() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let modulus = BigUint::from(17_u8);
        let scalar_type = ResolvedMatrixType::new(modulus.clone(), 1, 1, 1).unwrap();
        let input_type = ResolvedMatrixType::new(modulus.clone(), 1, 1, 40).unwrap();
        let decomposition_type = ResolvedMatrixType::new(modulus.clone(), 1, 40, 40).unwrap();
        let gadget_type = input_type.clone();
        let large = source_with(&mut expressions, scalar_type, 95_001);
        let noise = gaussian_factor(&mut expressions, input_type.clone(), 95_002, 3);
        let (gadget, decomposition, input) = gadget_product(
            &mut expressions,
            false,
            40,
            gadget_type.clone(),
            decomposition_type.clone(),
            input_type.clone(),
            Some((2, false)),
        );
        let large_gadget = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[large, gadget])
            .unwrap();
        let left = expressions
            .intern_matrix_transform(MatrixOperation::Add, &[large_gadget, noise])
            .unwrap();
        let root = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[left, decomposition])
            .unwrap();
        let registry =
            recomposition_registry(gadget_type, decomposition_type, input_type, false, 40);
        let (mut facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let ResolvedValueType::Matrix(large_type) = expressions.value_type(large).unwrap() else {
            panic!("large test input must be a matrix")
        };
        let mut large_facts = MatrixFacts::new(
            large_type.clone(),
            MatrixMetadata::new(MatrixLayout::row_major(1, 1)),
        );
        large_facts.coefficient_bound = NumericContract::Known(CoefficientBound::Large);
        facts.insert(&expressions, large, ValueFacts::Matrix(large_facts)).unwrap();
        for finite in [noise, gadget, decomposition, input] {
            insert_matrix_bound(&mut facts, &expressions, finite, 3);
        }
        let mut relations = RelationRegistry::new();
        relations.freeze();
        let mut cache = NormalizationCache::new();
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_relations(&relations, &mut cache)
            .with_gadget_recompositions(&registry);
        let value = normalizer.normalize(semantic).unwrap();
        let normal_form = value.exact_nf.as_ref().unwrap();

        assert_eq!(normal_form.exact_terms.len(), 1);
        assert!(matches!(
            normal_form.bounded_summary.coefficient_bound(),
            NumericContract::Known(CoefficientBound::Finite(_))
        ));
        assert_eq!(value.coefficient_bound, NumericContract::Known(CoefficientBound::Large));
        assert!(normalizer.counters.bounded_fold_count > 0);

        let descriptor =
            monomials.descriptor(*normal_form.exact_terms.keys().next().unwrap()).unwrap();
        let factor_expressions = descriptor
            .central_factors
            .iter()
            .chain(descriptor.ordered_factors.iter())
            .map(|factor| factor.expression())
            .collect::<BTreeSet<_>>();
        assert!(factor_expressions.contains(&large));
        assert!(factor_expressions.contains(&input));
        assert!(!factor_expressions.contains(&gadget));
        assert!(!factor_expressions.contains(&decomposition));
        assert!(!factor_expressions.contains(&noise));
    }

    #[test]
    fn bounded_summary_rejects_missing_and_large_contracts() {
        assert!(matches!(
            BoundedSummary::from_contract(NumericContract::Missing),
            Err(NormalizeError::InvalidExactPlan { .. })
        ));
        assert!(matches!(
            BoundedSummary::from_contract(NumericContract::Known(CoefficientBound::Large)),
            Err(NormalizeError::InvalidExactPlan { .. })
        ));
        assert_eq!(
            BoundedSummary::zero().coefficient_bound(),
            NumericContract::Known(CoefficientBound::ExactZero)
        );
    }

    #[test]
    fn bounded_only_addition_is_one_summary_without_exact_terms() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let left = gaussian_factor(&mut expressions, matrix_type(), 95_011, 3);
        let right = gaussian_factor(&mut expressions, matrix_type(), 95_012, 5);
        let root =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[left, right]).unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let mut relations = RelationRegistry::new();
        relations.freeze();
        let mut cache = NormalizationCache::new();
        let value = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_relations(&relations, &mut cache)
            .normalize(semantic)
            .unwrap();
        let normal_form = value.exact_nf.unwrap();
        assert!(normal_form.exact_terms.is_empty());
        assert_eq!(
            normal_form.bounded_summary.coefficient_bound(),
            NumericContract::Known(CoefficientBound::finite(8_u8))
        );
        assert_eq!(value.coefficient_bound, normal_form.bounded_summary.coefficient_bound());
    }

    #[test]
    fn tall_shaped_bounded_scalar_addition_keeps_identity_until_large_product() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let modulus = BigUint::from(17_u8);
        let scalar_type = ResolvedMatrixType::new(modulus.clone(), 4, 1, 1).unwrap();
        let row_type = ResolvedMatrixType::new(modulus, 4, 1, 2).unwrap();
        let large = source_with(&mut expressions, row_type, 95_014);
        let first = gaussian_factor(&mut expressions, scalar_type.clone(), 95_015, 3);
        let second = gaussian_factor(&mut expressions, scalar_type, 95_016, 5);
        let scalar =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[first, second]).unwrap();
        let root = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[large, scalar])
            .unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let mut relations = RelationRegistry::new();
        relations.freeze();
        let mut cache = NormalizationCache::new();
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_relations(&relations, &mut cache);
        let value = normalizer.normalize(semantic).unwrap();
        let normal_form = value.exact_nf.unwrap();

        assert_eq!(normal_form.exact_terms.len(), 2);
        assert!(normal_form.bounded_summary.is_zero());
        assert_eq!(value.coefficient_bound, NumericContract::Known(CoefficientBound::Large));
        let factor_sets = normal_form
            .exact_terms
            .keys()
            .map(|monomial| {
                let descriptor = monomials.descriptor(*monomial).unwrap();
                descriptor
                    .central_factors
                    .iter()
                    .chain(descriptor.ordered_factors.iter())
                    .map(|factor| factor.expression())
                    .collect::<BTreeSet<_>>()
            })
            .collect::<Vec<_>>();
        assert!(factor_sets.iter().all(|factors| factors.contains(&large)));
        assert!(factor_sets.iter().any(|factors| factors.contains(&first)));
        assert!(factor_sets.iter().any(|factors| factors.contains(&second)));
    }

    #[test]
    fn bounded_tensor_is_one_summary_without_exact_terms() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let input_type = matrix_type();
        let output_type = ResolvedMatrixType::new(
            input_type.modulus.clone(),
            input_type.ring_dimension,
            input_type.rows * input_type.rows,
            input_type.columns * input_type.columns,
        )
        .unwrap();
        let left = gaussian_factor(&mut expressions, input_type.clone(), 95_014, 3);
        let right = gaussian_factor(&mut expressions, input_type.clone(), 95_015, 5);
        let root = expressions
            .intern_matrix_transform(
                MatrixOperation::Tensor {
                    output: output_type.clone(),
                    left_layout: MatrixLayout::row_major(input_type.rows, input_type.columns),
                    right_layout: MatrixLayout::row_major(input_type.rows, input_type.columns),
                    output_layout: MatrixLayout::row_major(output_type.rows, output_type.columns),
                },
                &[left, right],
            )
            .unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let value = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(semantic)
            .unwrap();
        let normal_form = value.exact_nf.unwrap();
        assert!(normal_form.exact_terms.is_empty());
        assert!(matches!(
            normal_form.bounded_summary.coefficient_bound(),
            NumericContract::Known(CoefficientBound::Finite(_))
        ));
    }

    #[test]
    fn identical_large_products_cancel_exactly() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let matrix = matrix_type();
        let left = source_with(&mut expressions, matrix.clone(), 95_016);
        let right = source_with(&mut expressions, matrix.clone(), 95_017);
        let product =
            expressions.intern_matrix_transform(MatrixOperation::Multiply, &[left, right]).unwrap();
        let root = expressions
            .intern_matrix_transform(MatrixOperation::Subtract, &[product, product])
            .unwrap();
        let (mut facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        for expression in [left, right] {
            let mut matrix_facts = MatrixFacts::new(
                matrix.clone(),
                MatrixMetadata::new(MatrixLayout::row_major(matrix.rows, matrix.columns)),
            );
            matrix_facts.coefficient_bound = NumericContract::Known(CoefficientBound::Large);
            facts.insert(&expressions, expression, ValueFacts::Matrix(matrix_facts)).unwrap();
        }
        let value = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(semantic)
            .unwrap();
        assert!(value.exact_nf.unwrap().is_zero());
    }

    #[test]
    fn large_times_compressed_summary_fails_before_product_mutation() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let large = source_with(&mut expressions, matrix_type(), 95_021);
        let (mut facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, large);
        let mut large_facts =
            MatrixFacts::new(matrix_type(), MatrixMetadata::new(MatrixLayout::row_major(2, 2)));
        large_facts.coefficient_bound = NumericContract::Known(CoefficientBound::Large);
        facts.insert(&expressions, large, ValueFacts::Matrix(large_facts)).unwrap();
        let large_id = monomials.intern(&expressions, &programs, &[], &[semantic]).unwrap();
        let summary = PolynomialNF {
            exact_terms: BTreeMap::new(),
            bounded_summary: BoundedSummary::finite(BoundExpression::new(BigUint::from(3_u8))),
        };
        let exact_large = PolynomialNF {
            exact_terms: BTreeMap::from([(large_id, BigInt::from(1_u8))]),
            bounded_summary: BoundedSummary::zero(),
        };
        let before_len = monomials.len();
        let before_occupied = monomials.occupied_len();
        let mut terms = BTreeMap::new();
        let mut noise = NumericContract::Known(CoefficientBound::ExactZero);
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let error = normalizer
            .execute_product_into(
                &matrix_type(),
                &matrix_type(),
                &summary,
                &exact_large,
                &BigInt::from(1_u8),
                &mut terms,
                &mut noise,
            )
            .unwrap_err();
        assert!(matches!(error, NormalizeError::InvalidExactPlan { .. }));
        assert!(terms.is_empty());
        assert_eq!(noise, NumericContract::Known(CoefficientBound::ExactZero));
        assert_eq!(normalizer.monomials.len(), before_len);
        assert_eq!(normalizer.monomials.occupied_len(), before_occupied);
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
    fn one_by_one_gadget_product_recomposes_to_central_input() {
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
        // Both operands are declared 1x1, but their ordered product must apply the exact gadget
        // relation before the proven scalar result is centralized.
        insert_matrix_layout_fact(&expressions, &mut facts, gadget, false);
        insert_matrix_layout_fact(&expressions, &mut facts, decomposition, false);
        insert_matrix_layout_fact(&expressions, &mut facts, input, false);
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_gadget_recompositions(&registry);
        let value = normalizer.normalize(root).unwrap();
        let id = *value.exact_nf.unwrap().exact_terms.keys().next().unwrap();
        let descriptor = monomials.descriptor(id).unwrap();
        assert_eq!(descriptor.central_factors.len(), 1);
        assert_eq!(descriptor.central_factors[0].expression(), input);
        assert!(descriptor.ordered_factors.is_empty());
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
        // Both operands are typed 1x1 scalars, so they commute centrally. The ordered
        // gadget-decomposition rewrite is deliberately unavailable in the reversed product.
        assert_eq!(reversed_descriptor.central_factors.len(), 2);
        assert!(reversed_descriptor.ordered_factors.is_empty());

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
            .intern_matrix_transform(MatrixOperation::Multiply, &[gadget, central])
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
    fn single_scalar_action_matches_eager_on_both_sides_without_deferral() {
        for scalar_on_left in [false, true] {
            let mut expressions = ExprArena::new();
            let mut programs = ProgramArena::new();
            let scalar_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap();
            let matrix_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 2).unwrap();
            let matrix = matrix_source(
                &mut expressions,
                "deferred-scalar-action-matrix",
                matrix_type.clone(),
                None,
            );
            let scalar =
                matrix_source(&mut expressions, "deferred-scalar-action-scalar", scalar_type, None);
            let inputs = if scalar_on_left { [scalar, matrix] } else { [matrix, scalar] };
            let action =
                expressions.intern_matrix_transform(MatrixOperation::Multiply, &inputs).unwrap();
            let zero =
                matrix_source(&mut expressions, "deferred-scalar-action-zero", matrix_type, None);
            let zero = expressions
                .intern_matrix_transform(MatrixOperation::Subtract, &[zero, zero])
                .unwrap();
            let root =
                expressions.intern_matrix_transform(MatrixOperation::Add, &[action, zero]).unwrap();
            let (facts, mut monomials, root_semantic) =
                setup(&mut expressions, &mut programs, root);
            let action_semantic = programs.scoped(&expressions, monomials.scope(), action).unwrap();
            let eager = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                .unwrap()
                .normalize(action_semantic)
                .unwrap();
            let deferred = {
                let mut normalizer =
                    Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
                normalizer.normalize(root_semantic).unwrap()
            };
            assert_eq!(eager.coefficient_bound, deferred.coefficient_bound);
            assert_eq!(
                descriptor_coefficient_multiset(eager.exact_nf.as_ref().unwrap(), &monomials),
                descriptor_coefficient_multiset(deferred.exact_nf.as_ref().unwrap(), &monomials)
            );
            let exact = deferred.exact_nf.unwrap();
            let descriptor =
                monomials.descriptor(*exact.exact_terms.keys().next().unwrap()).unwrap();
            assert_eq!(descriptor.central_factors.len(), 1);
            assert_eq!(descriptor.ordered_factors.len(), 1);
        }
    }

    #[test]
    fn ordinary_scalar_action_is_commutative_and_associative() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let scalar_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap();
        let matrix_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 2).unwrap();
        let matrix_a =
            matrix_source(&mut expressions, "ordinary-scalar-matrix-a", matrix_type.clone(), None);
        let matrix_b =
            matrix_source(&mut expressions, "ordinary-scalar-matrix-b", matrix_type, None);
        let scalar = matrix_source(&mut expressions, "ordinary-scalar", scalar_type.clone(), None);
        let distinct_scalar =
            matrix_source(&mut expressions, "ordinary-distinct-scalar", scalar_type, None);
        let matrix_times_scalar = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[matrix_a, scalar])
            .unwrap();
        let scalar_times_matrix = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[scalar, matrix_a])
            .unwrap();
        let commutator = expressions
            .intern_matrix_transform(
                MatrixOperation::Subtract,
                &[matrix_times_scalar, scalar_times_matrix],
            )
            .unwrap();
        let distinct = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[distinct_scalar, matrix_a])
            .and_then(|right| {
                expressions.intern_matrix_transform(
                    MatrixOperation::Subtract,
                    &[matrix_times_scalar, right],
                )
            })
            .unwrap();
        let both_scalar = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[scalar, distinct_scalar])
            .unwrap();
        let both_scalar_action = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[matrix_a, both_scalar])
            .unwrap();
        let left_associated = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[matrix_times_scalar, matrix_b])
            .unwrap();
        let right_associated = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[scalar, matrix_b])
            .and_then(|right| {
                expressions.intern_matrix_transform(MatrixOperation::Multiply, &[matrix_a, right])
            })
            .unwrap();
        let associator = expressions
            .intern_matrix_transform(
                MatrixOperation::Subtract,
                &[left_associated, right_associated],
            )
            .unwrap();
        let root = expressions
            .intern_matrix_transform(MatrixOperation::Add, &[commutator, distinct])
            .and_then(|value| {
                expressions.intern_matrix_transform(MatrixOperation::Add, &[value, associator])
            })
            .and_then(|value| {
                expressions
                    .intern_matrix_transform(MatrixOperation::Add, &[value, both_scalar_action])
            })
            .unwrap();
        let (facts, mut monomials, _) = setup(&mut expressions, &mut programs, root);
        let scope = monomials.scope();

        let commutator = programs.scoped(&expressions, scope, commutator).unwrap();
        let commutator_nf = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(commutator)
            .unwrap()
            .exact_nf
            .unwrap();
        assert!(commutator_nf.is_zero());

        let associator = programs.scoped(&expressions, scope, associator).unwrap();
        let associator_nf = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(associator)
            .unwrap()
            .exact_nf
            .unwrap();
        assert!(associator_nf.is_zero());

        let distinct = programs.scoped(&expressions, scope, distinct).unwrap();
        let distinct_nf = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(distinct)
            .unwrap()
            .exact_nf
            .unwrap();
        assert_eq!(distinct_nf.exact_terms.len(), 2);

        let both_scalar = programs.scoped(&expressions, scope, both_scalar).unwrap();
        let both_scalar_nf = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(both_scalar)
            .unwrap()
            .exact_nf
            .unwrap();
        assert_eq!(both_scalar_nf.exact_terms.len(), 1);
        let descriptor =
            monomials.descriptor(*both_scalar_nf.exact_terms.keys().next().unwrap()).unwrap();
        assert_eq!(descriptor.central_factors.len(), 2);
        assert!(descriptor.ordered_factors.is_empty());
    }

    #[test]
    fn ordinary_scalar_action_keeps_composite_scalar_opaque() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let row = matrix_source(
            &mut expressions,
            "ordinary-composite-row",
            ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 2).unwrap(),
            None,
        );
        let column = matrix_source(
            &mut expressions,
            "ordinary-composite-column",
            ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 1).unwrap(),
            None,
        );
        let composite_scalar =
            expressions.intern_matrix_transform(MatrixOperation::Multiply, &[row, column]).unwrap();
        let matrix = matrix_source(
            &mut expressions,
            "ordinary-composite-matrix",
            ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 2).unwrap(),
            None,
        );
        let scalar_action = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[composite_scalar, matrix])
            .unwrap();
        let (facts, mut monomials, semantic) =
            setup(&mut expressions, &mut programs, scalar_action);
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
        assert_eq!(descriptor.ordered_factors[0].expression(), scalar_action);
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
    fn forced_gc_marks_unique_additive_plan_leaves() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let x = source_with(&mut expressions, matrix_type(), 10_905);
        let y = source_with(&mut expressions, matrix_type(), 10_906);
        let sum = expressions.intern_matrix_transform(MatrixOperation::Add, &[x, y]).unwrap();
        let root = expressions.intern_matrix_transform(MatrixOperation::Add, &[sum, sum]).unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        normalizer.monomial_gc_allocation_threshold_bytes = 0;
        let value = normalizer.normalize(semantic).unwrap();
        let exact = value.exact_nf.unwrap();
        assert_eq!(exact.exact_terms.len(), 2);
        assert!(exact.exact_terms.values().all(|coefficient| coefficient == &BigInt::from(2_u8)));
        for monomial in exact.exact_terms.keys() {
            normalizer.monomials.descriptor(*monomial).unwrap();
        }
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
