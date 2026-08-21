//! Exact, job-local normalisation over the expression/program arenas.
//!
//! Expression IDs remain the semantic identity, while exact polynomial
//! terms contain only compact monomial IDs.  In particular, this module has no factor identity,
//! symbolic-factor, relation-protection, or provenance authority of its own.

use super::{
    arena::{
        ArenaToken, ExprArena, ExprId, ExprNode, HashVariant, MatrixConstantKind, MatrixLayout,
        MatrixOperation, ResolvedMatrixType, ResolvedValueType, SamplerOperation, ScalarOperation,
        ScopeProof, ScopedExprId, SemanticSourceIdentity, TypedConstant, ValueOperator,
        ValueTransformOperation,
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
    monomial::{MonomialArena, MonomialError, MonomialId, MonomialSweepOwnerReport, TermMap},
    program::{ArenaError, ProgramArena, ValueProgramId},
    relation::{
        CanonicalLhsKey, FrozenGeneration, GadgetRecompositionRegistry, NormalizationCache,
        RelationRegistry, RelationRegistryError, RelationResolution, RuntimeSpecializationKey,
        UniversalRelationRegistration,
    },
};
use mxx_ir_core::types::ConcreteMatrixType;
use num_bigint::{BigInt, BigUint};
use num_traits::{Signed, ToPrimitive, Zero};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    fmt,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
use tracing::info;

const NORMALIZATION_TRACE_TARGET: &str = "mxx_correctness::operational_noise::normalization";
const NORMALIZATION_TRACE_LINE_BUDGET: u8 = 32;
const NORMALIZATION_TRACE_SUBPHASE_LINE_BUDGET: u8 = 8;
const NORMALIZATION_TRACE_POST_LINE_BUDGET: u8 = 8;
const NORMALIZATION_TRACE_CRITICAL_CALLER_RESERVE: u8 = 7;
const NORMALIZATION_TRACE_FOCUS_CALL_ENV: &str = "MXX_OPERATIONAL_TRACE_FOCUS_CALL";
const NORMALIZATION_TRACE_FOCUS_EXPRESSION_SLOT_ENV: &str =
    "MXX_OPERATIONAL_TRACE_FOCUS_EXPRESSION_SLOT";
const NORMALIZATION_TRACE_FOCUS_TAIL_NODES_ENV: &str = "MXX_OPERATIONAL_TRACE_FOCUS_TAIL_NODES";
const NORMALIZATION_NODE_HEARTBEAT: u64 = 100_000;
const LARGE_PRODUCT_PLANNED_PAIRS: u64 = 100_000;
const MONOMIAL_GC_ALLOCATION_THRESHOLD_BYTES: u64 = 256 * 1024 * 1024;
const PRODUCT_PROCESSED_HEARTBEAT: u64 = 1_000_000;
const NORMALIZATION_WATCHDOG_ENV: &str = "MXX_OPERATIONAL_WATCHDOG_TRACE";
const NORMALIZATION_WATCHDOG_INTERVAL_ENV: &str = "MXX_OPERATIONAL_WATCHDOG_INTERVAL_SECS";
const NORMALIZATION_FOUR_CLASS_CENSUS_ENV: &str = "MXX_OPERATIONAL_FOUR_CLASS_CENSUS";
const NORMALIZATION_WATCHDOG_INTERVAL: Duration = Duration::from_secs(4);
const NORMALIZATION_WATCHDOG_MAX_INTERVAL_SECS: u64 = 3600;
const NORMALIZATION_WATCHDOG_MAX_SNAPSHOTS: u8 = 18;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DiagnosticPhase {
    ScopeProof,
    ScopeProofDone,
    StateReset,
    UseCounts,
    UseCountsDone,
    NodeWalk,
    EvaluateNode,
    Post,
    CallerMerge,
    CallReturn,
    RuntimeLookup,
    UniversalSpecialization,
    Registration,
    RhsIntern,
    ProofRollback,
    RelationClosure,
    RelationSearch,
    ProductGeneration,
    ProductGenerationEnd,
    ProductDrain,
    ProductEnd,
    Error,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OwnerCensusReason {
    RetainedArenaMilestone,
    LargeProductGenerationEnd,
    LargeProductEnd,
    OuterTerminal,
    OuterError,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DiagnosticRelationCounters {
    closures_started: u64,
    closures_completed: u64,
    closures_errored: u64,
    active_depth: u64,
    closed_relations_present: bool,
    initial_terms: u64,
    dequeued: u64,
    zero_skipped: u64,
    nonzero_dequeued: u64,
    enqueued: u64,
    queue_peak: u64,
    duplicate_same_outcome: u64,
    duplicate_changed_outcome: u64,
    central_factors_total: u64,
    central_factors_max: u64,
    ordered_factors_total: u64,
    ordered_factors_max: u64,
    gadget_attempts: u64,
    gadget_matches: u64,
    gadget_output_terms_total: u64,
    gadget_output_terms_max: u64,
    whole_closed_probes: u64,
    whole_closed_resolves: u64,
    whole_closed_matches: u64,
    whole_closed_ambiguities: u64,
    closed_window_probes: u64,
    closed_window_interned_hits: u64,
    closed_window_resolves: u64,
    closed_window_matches: u64,
    closed_window_ambiguities: u64,
    closed_subword_matches: u64,
    universal_probes: u64,
    universal_dispatch_hits: u64,
    universal_specializations: u64,
    universal_lhs_candidates: u64,
    universal_span_candidates: u64,
    universal_matches: u64,
    universal_ambiguities: u64,
    universal_rewrites: u64,
    no_matches: u64,
    match_errors: u64,
    rhs_splices: u64,
    rhs_terms_total: u64,
    rhs_terms_max: u64,
    rhs_terms_enqueued: u64,
    monomial_combines: u64,
    prefix_combines: u64,
    suffix_combines: u64,
    result_terms: u64,
    final_terms: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RelationOutcomeKind {
    Gadget,
    WholeClosed,
    ClosedWindow,
    Universal,
    NoMatch,
    Error,
}

struct RelationClosureDiagnostic {
    counters: DiagnosticRelationCounters,
    outcomes: HashMap<MonomialId, RelationOutcomeKind>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DiagnosticSpecializationCounters {
    runtime_lookup_hits: u64,
    runtime_lookup_misses: u64,
    ordinary_specializations_started: u64,
    ordinary_specializations_completed: u64,
    proof_specializations_started: u64,
    proof_specializations_completed: u64,
    registrations_started: u64,
    registrations_completed: u64,
    rhs_exact_terms_total: u64,
    rhs_exact_terms_max: u64,
    interner_existing: u64,
    interner_inserted: u64,
    proof_rollbacks_completed: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DiagnosticTimingCounter {
    calls: u64,
    total_ns: u64,
    max_ns: u64,
}

impl DiagnosticTimingCounter {
    fn record(&mut self, elapsed_ns: u64) {
        self.calls = self.calls.saturating_add(1);
        self.total_ns = self.total_ns.saturating_add(elapsed_ns);
        self.max_ns = self.max_ns.max(elapsed_ns);
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DiagnosticTimings {
    // Timing counters are a lexical hierarchy, not disjoint buckets to sum. Inclusive parents
    // deliberately contain their child work: `outer_relation_rebound`,
    // `universal_search_total`, `universal_specialized_cached`, `cached_miss_specialize`, and
    // `specialized_nested_normalize`. All other fields describe exclusive lexical leaf siblings.
    closure_setup: DiagnosticTimingCounter,
    descriptor_and_gadget: DiagnosticTimingCounter,
    closed_search: DiagnosticTimingCounter,
    universal_search_total: DiagnosticTimingCounter,
    rhs_fetch_prefix_suffix: DiagnosticTimingCounter,
    rhs_recombine_enqueue: DiagnosticTimingCounter,
    no_match_result_merge: DiagnosticTimingCounter,
    closure_final_assignment: DiagnosticTimingCounter,
    universal_factor_dispatch: DiagnosticTimingCounter,
    universal_selector_range: DiagnosticTimingCounter,
    universal_specialized_cached: DiagnosticTimingCounter,
    universal_lhs_layout_span: DiagnosticTimingCounter,
    universal_global_selection: DiagnosticTimingCounter,
    cached_key_lookup: DiagnosticTimingCounter,
    cached_hit_clone: DiagnosticTimingCounter,
    cached_miss_specialize: DiagnosticTimingCounter,
    cached_insert_return_clone: DiagnosticTimingCounter,
    specialized_nested_normalize: DiagnosticTimingCounter,
    specialized_extraction: DiagnosticTimingCounter,
    specialized_merge_bounds: DiagnosticTimingCounter,
    specialized_state_restore: DiagnosticTimingCounter,
    outer_scope_proof: DiagnosticTimingCounter,
    outer_use_counts: DiagnosticTimingCounter,
    outer_relation_rebound: DiagnosticTimingCounter,
    outer_bound_fold: DiagnosticTimingCounter,
    cached_hit_returned_entries_total: u64,
    cached_hit_returned_entries_max: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SpecializationKind {
    Ordinary,
    Proof,
}

#[derive(Clone, Copy, Debug)]
struct DiagnosticProgress {
    generation: u64,
    current_call: u64,
    last_completed: u64,
    depth: u32,
    phase: DiagnosticPhase,
    expression_slot: u64,
    operator: &'static str,
    nodes_done: u64,
    nodes_total: u64,
    product_processed: u64,
    product_generated: u64,
    product_enqueued: u64,
    product_processed_current: u64,
    product_planned_current: u64,
    product_generation_current: u64,
    product_enqueued_current: u64,
    product_queue_current: u64,
    product_output_current: u64,
    relation_processed: u64,
    specialization: DiagnosticSpecializationCounters,
    relation_closure: DiagnosticRelationCounters,
    timings: DiagnosticTimings,
    owners: DiagnosticOwnerCensus,
    owner_census_seq: u64,
    owner_census_depth: u32,
    owner_census_call: u64,
    owner_census_product_call_id: u64,
    owner_census_phase: DiagnosticPhase,
    owner_census_reason: Option<OwnerCensusReason>,
    gc: DiagnosticGcCounters,
    last_change: Instant,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DiagnosticGcCounters {
    sweep_count: u64,
    last_sweep_node: u64,
    last_high_water_slots: u64,
    last_occupied_slots: u64,
    last_reclaimed_slots: u64,
    last_reclaimed_payload_bytes: u64,
    cumulative_reclaimed_slots: u64,
    cumulative_reclaimed_payload_bytes: u64,
    last_allocated_payload_before_bytes: u64,
    last_bucket_entries: u64,
    last_protected_prefix_occupied_slots: u64,
    last_occupied_central_factor_entries: u64,
    last_occupied_ordered_factor_entries: u64,
    last_occupied_factor_payload_lower_bound_bytes: u64,
    last_protected_prefix: MonomialSweepOwnerReport,
    last_value_cache: MonomialSweepOwnerReport,
    last_exact_plan: MonomialSweepOwnerReport,
    last_gadget: MonomialSweepOwnerReport,
    last_canonical_runtime: MonomialSweepOwnerReport,
    last_closed: MonomialSweepOwnerReport,
    last_suspended: MonomialSweepOwnerReport,
    value_cache_entries: u64,
    value_cache_exact_term_refs: u64,
    value_cache_top8_exact_term_refs: u64,
    value_cache_top8_len: u8,
    value_cache_top8: [DiagnosticValueCacheTopEntry; 8],
    materialized_leaf_top8: DiagnosticMaterializedLeafTop8,
    exact_plan_four_class: DiagnosticFourClassCensus,
    four_class_total_ns: u64,
    four_class_max_ns: u64,
    four_class_last_ns: u64,
    sweep_total_ns: u64,
    sweep_max_ns: u64,
    sweep_last_ns: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DiagnosticClassStats {
    unique_monomials: u64,
    term_refs: u64,
    payload_lower_bound_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DiagnosticFourClassTopEntry {
    nf_ordinal: u64,
    under_product: bool,
    finite_no_relation_refs: u64,
    finite_relation_frontier_refs: u64,
    missing_refs: u64,
    large_refs: u64,
    finite_no_relation_payload: u64,
    finite_relation_frontier_payload: u64,
    missing_payload: u64,
    large_payload: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DiagnosticFourClassCensus {
    finite_no_relation: DiagnosticClassStats,
    finite_relation_frontier: DiagnosticClassStats,
    missing: DiagnosticClassStats,
    large: DiagnosticClassStats,
    // Reason statistics overlap by design; the union fields prove exact frontier coverage.
    current_exact_preimage: DiagnosticClassStats,
    // Ambiguous universal dispatch is relation-relevant but is not proven current authority.
    ambiguous_universal_dispatch: DiagnosticClassStats,
    current_authorized_gadget_pair: DiagnosticClassStats,
    closed_blanket: DiagnosticClassStats,
    future_typed_gadget: DiagnosticClassStats,
    future_universal_blanket: DiagnosticClassStats,
    frontier_unique_union: u64,
    frontier_reason_unique_union: u64,
    frontier_reason_term_ref_union: u64,
    frontier_reason_payload_union: u64,
    top_len: u8,
    top: [DiagnosticFourClassTopEntry; 8],
}

impl DiagnosticFourClassCensus {
    fn class_mut(&mut self, class: DiagnosticTermClass) -> &mut DiagnosticClassStats {
        match class {
            DiagnosticTermClass::FiniteNoRelation => &mut self.finite_no_relation,
            DiagnosticTermClass::FiniteRelationFrontier => &mut self.finite_relation_frontier,
            DiagnosticTermClass::Missing => &mut self.missing,
            DiagnosticTermClass::Large => &mut self.large,
        }
    }

    fn reason_mut(&mut self, index: usize) -> Option<&mut DiagnosticClassStats> {
        Some(match index {
            0 => &mut self.current_exact_preimage,
            1 => &mut self.ambiguous_universal_dispatch,
            2 => &mut self.current_authorized_gadget_pair,
            3 => &mut self.closed_blanket,
            4 => &mut self.future_typed_gadget,
            5 => &mut self.future_universal_blanket,
            _ => return None,
        })
    }
}

impl DiagnosticFourClassTopEntry {
    fn observe(&mut self, class: DiagnosticTermClass, payload: u64) {
        let (refs, bytes) = match class {
            DiagnosticTermClass::FiniteNoRelation => {
                (&mut self.finite_no_relation_refs, &mut self.finite_no_relation_payload)
            }
            DiagnosticTermClass::FiniteRelationFrontier => (
                &mut self.finite_relation_frontier_refs,
                &mut self.finite_relation_frontier_payload,
            ),
            DiagnosticTermClass::Missing => (&mut self.missing_refs, &mut self.missing_payload),
            DiagnosticTermClass::Large => (&mut self.large_refs, &mut self.large_payload),
        };
        *refs = refs.saturating_add(1);
        *bytes = bytes.saturating_add(payload);
    }

    fn total_refs(self) -> u64 {
        self.finite_no_relation_refs
            .saturating_add(self.finite_relation_frontier_refs)
            .saturating_add(self.missing_refs)
            .saturating_add(self.large_refs)
    }
}

#[derive(Clone)]
struct DiagnosticExactNf {
    normal_form: Arc<PolynomialNF>,
    ordinal: u64,
    under_product: bool,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum DiagnosticTermClass {
    FiniteNoRelation,
    FiniteRelationFrontier,
    Large,
    Missing,
}

impl DiagnosticTermClass {
    fn rank(self) -> u8 {
        match self {
            Self::FiniteNoRelation => 0,
            Self::FiniteRelationFrontier => 1,
            Self::Large => 2,
            Self::Missing => 3,
        }
    }
}

const FRONTIER_CURRENT_EXACT_PREIMAGE: u8 = 1 << 0;
const FRONTIER_AMBIGUOUS_UNIVERSAL_DISPATCH: u8 = 1 << 1;
const FRONTIER_CURRENT_AUTHORIZED_GADGET: u8 = 1 << 2;
const FRONTIER_CLOSED_BLANKET: u8 = 1 << 3;
const FRONTIER_FUTURE_TYPED_GADGET: u8 = 1 << 4;
const FRONTIER_FUTURE_UNIVERSAL_BLANKET: u8 = 1 << 5;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DiagnosticValueCacheTopEntry {
    expression_slot: u64,
    operator_category: &'static str,
    term_count: u64,
    remaining_uses: u64,
    producer_input_count: u64,
    cached_input_exact_term_refs_sum: u64,
    cached_input_exact_term_refs_max: u64,
    multiply_scalar_classification: &'static str,
    multiply_left_rows: u64,
    multiply_left_columns: u64,
    multiply_right_rows: u64,
    multiply_right_columns: u64,
    multiply_add_sub_consumers: u64,
    multiply_consumers: u64,
    multiply_structural_holds: u64,
    multiply_root_other_consumers: u64,
    multiply_deferral_rejection: &'static str,
    additive_materialized_leaf: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DiagnosticProductConsumerCounts {
    add_sub: u64,
    multiply: u64,
    structural: u64,
    root_other: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DiagnosticProductEvaluationSnapshot {
    had_left_exact: bool,
    had_right_exact: bool,
    gadget_boundary: bool,
    scalar_classification: &'static str,
    left_rows: u64,
    left_columns: u64,
    right_rows: u64,
    right_columns: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
enum DiagnosticMaterializationReason {
    #[default]
    OrdinaryProducer,
    Root,
    RelationBoundary,
    SpecializedReturn,
    NonAddConsumer,
    NestedAdditiveReturn,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DiagnosticMaterializedLeafOrigin {
    producer: Option<ExprId>,
    producer_operator: &'static str,
    reason: DiagnosticMaterializationReason,
    consumer: Option<ExprId>,
    consumer_operator: &'static str,
    consumer_category: &'static str,
    remaining_uses: u64,
    scalar_classification: &'static str,
    forced_input_count: u64,
    forced_terms_sum: u64,
    forced_terms_max: u64,
    retained_term_count: u64,
}

#[derive(Clone, Debug)]
struct DiagnosticMaterializedLeafAttachment {
    normal_form: Weak<PolynomialNF>,
    origin: DiagnosticMaterializedLeafOrigin,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DiagnosticMaterializedLeafTop8 {
    exact_term_refs: u64,
    len: u8,
    top: [DiagnosticMaterializedLeafOrigin; 8],
}

impl DiagnosticMaterializedLeafTop8 {
    fn observe(&mut self, mut origin: DiagnosticMaterializedLeafOrigin, terms: u64) {
        origin.retained_term_count = terms;
        self.exact_term_refs = self.exact_term_refs.saturating_add(terms);
        let len = usize::from(self.len);
        let key = |entry: &DiagnosticMaterializedLeafOrigin| {
            (
                std::cmp::Reverse(entry.retained_term_count),
                entry.producer.map(ExprId::slot).unwrap_or(u32::MAX),
                entry.consumer.map(ExprId::slot).unwrap_or(u32::MAX),
            )
        };
        let insert = (0..len).find(|index| key(&origin) < key(&self.top[*index])).unwrap_or(len);
        if insert >= self.top.len() {
            return;
        }
        let new_len = (len + 1).min(self.top.len());
        for index in (insert + 1..new_len).rev() {
            self.top[index] = self.top[index - 1];
        }
        self.top[insert] = origin;
        self.len = u8::try_from(new_len).unwrap_or(8);
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DiagnosticValueCacheTop8 {
    entries: u64,
    exact_term_refs: u64,
    top_exact_term_refs: u64,
    len: u8,
    top: [DiagnosticValueCacheTopEntry; 8],
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DiagnosticOwnerCensus {
    monomial_allocated_descriptor_slots: u64,
    monomial_retained_descriptor_slots: u64,
    monomial_reclaimed_descriptor_slots: u64,
    monomial_reachable_descriptor_slots: u64,
    monomial_reachable_central_factor_entries: u64,
    monomial_reachable_ordered_factor_entries: u64,
    monomial_reachable_max_factor_word: u64,
    monomial_owned_payload_lower_bound_bytes: u64,
    monomial_unreachable_descriptor_slots: u64,
    monomial_unreachable_central_factor_entries: u64,
    monomial_unreachable_ordered_factor_entries: u64,
    monomial_unreachable_payload_lower_bound_bytes: u64,
    monomial_invalid_root_count: u64,
    // Owner-local term counts are reference counts and are deliberately non-additive: the unique
    // reachable monomial fields above use a slot bitset to deduplicate IDs shared by owners.
    cache_entries: u64,
    cache_exact_terms: u64,
    cache_exact_terms_peak: u64,
    cache_largest_nf_terms_seen: u64,
    gadget_entries: u64,
    gadget_exact_terms: u64,
    gadget_exact_terms_peak: u64,
    gadget_largest_nf_terms_seen: u64,
    canonical_rhs_entries: u64,
    canonical_rhs_exact_terms: u64,
    canonical_rhs_exact_terms_peak: u64,
    canonical_rhs_largest_nf_terms: u64,
    runtime_entries: u64,
    runtime_lhs_keys: u64,
    additive_plan_nodes: u64,
    ordinary_product_plan_nodes: u64,
    gadget_product_plan_nodes: u64,
    additive_unique_leaf_refs: u64,
    additive_unique_leaf_exact_term_refs: u64,
    additive_largest_leaf_exact_terms: u64,
    gadget_product_unique_operand_refs: u64,
    gadget_product_operand_exact_term_refs: u64,
    gadget_product_largest_operand_exact_terms: u64,
    materialized_leaf_top8: DiagnosticMaterializedLeafTop8,
    gadget_product_plans_created: u64,
    gadget_product_streamed_executions: u64,
    gadget_product_zero_weight_skips: u64,
    gadget_product_standalone_materializations: u64,
    gadget_product_planned_pairs: u64,
    gadget_product_max_streamed_output_terms: u64,
    ordinary_product_plans_created: u64,
    typed_product_candidate_plans: u64,
    typed_product_direct_executions: u64,
    typed_product_pair_attempts: u64,
    typed_product_pair_matches: u64,
    typed_product_pair_ordinary_fallbacks: u64,
    typed_product_standalone_materializations: u64,
    ordinary_product_streamed_executions: u64,
    ordinary_product_zero_weight_skips: u64,
    ordinary_product_standalone_materializations: u64,
    ordinary_product_planned_pairs: u64,
    ordinary_product_max_streamed_output_terms: u64,
    scalar_action_plans_created: u64,
    scalar_action_streamed_executions: u64,
    scalar_action_zero_weight_skips: u64,
    scalar_action_standalone_materializations: u64,
    scalar_action_reclassified_terms: u64,
    scalar_action_reclassified_factors_max: u64,
    additive_materializations: u64,
    additive_materialization_output_terms_total: u64,
    additive_materialization_output_terms_max: u64,
}

/// A bounded, immutable owner-census observation. The product fields are copied at the same
/// instant as the owner roots are scanned, so a subsequent phase transition cannot relabel or
/// overwrite a generation-end sample before the reporter thread wakes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DiagnosticOwnerSample {
    owners: DiagnosticOwnerCensus,
    seq: u64,
    depth: u32,
    call: u64,
    product_call_id: u64,
    reason: OwnerCensusReason,
    phase: DiagnosticPhase,
    product_generated: u64,
    product_enqueued: u64,
    product_queue_current: u64,
    product_output_current: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct NormalizerOwnerCounters {
    cache_exact_terms: u64,
    cache_exact_terms_peak: u64,
    cache_largest_nf_terms_seen: u64,
    gadget_exact_terms: u64,
    gadget_exact_terms_peak: u64,
    gadget_largest_nf_terms_seen: u64,
}

#[derive(Clone, Copy)]
struct DiagnosticParentState {
    current_call: u64,
    depth: u32,
    phase: DiagnosticPhase,
    expression_slot: u64,
    operator: &'static str,
    nodes_done: u64,
    nodes_total: u64,
}

struct DiagnosticShared {
    progress: Mutex<DiagnosticProgress>,
    stop: AtomicBool,
    emitted_lines: Mutex<u8>,
    owner_samples: Mutex<Vec<DiagnosticOwnerSample>>,
    #[cfg(test)]
    events: Mutex<Vec<&'static str>>,
    #[cfg(test)]
    snapshots: Mutex<Vec<DiagnosticProgress>>,
}

struct DiagnosticWatchdog {
    shared: Arc<DiagnosticShared>,
    reporter: Option<JoinHandle<u8>>,
    next_call: u64,
}

impl DiagnosticWatchdog {
    fn start(generation: u64, interval: Duration) -> Option<Self> {
        let shared = Arc::new(DiagnosticShared {
            progress: Mutex::new(DiagnosticProgress {
                generation,
                current_call: 0,
                last_completed: 0,
                depth: 0,
                phase: DiagnosticPhase::StateReset,
                expression_slot: 0,
                operator: "none",
                nodes_done: 0,
                nodes_total: 0,
                product_processed: 0,
                product_generated: 0,
                product_enqueued: 0,
                product_processed_current: 0,
                product_planned_current: 0,
                product_generation_current: 0,
                product_enqueued_current: 0,
                product_queue_current: 0,
                product_output_current: 0,
                relation_processed: 0,
                specialization: DiagnosticSpecializationCounters::default(),
                relation_closure: DiagnosticRelationCounters::default(),
                timings: DiagnosticTimings::default(),
                owners: DiagnosticOwnerCensus::default(),
                owner_census_seq: 0,
                owner_census_depth: 0,
                owner_census_call: 0,
                owner_census_product_call_id: 0,
                owner_census_phase: DiagnosticPhase::StateReset,
                owner_census_reason: None,
                gc: DiagnosticGcCounters::default(),
                last_change: Instant::now(),
            }),
            stop: AtomicBool::new(false),
            emitted_lines: Mutex::new(0),
            owner_samples: Mutex::new(Vec::new()),
            #[cfg(test)]
            events: Mutex::new(Vec::new()),
            #[cfg(test)]
            snapshots: Mutex::new(Vec::new()),
        });
        let reporter_shared = Arc::clone(&shared);
        let reporter = thread::Builder::new()
            .name("mxx-normalization-watchdog".to_owned())
            .spawn(move || {
                // The starter emits the initial line synchronously before releasing this thread,
                // so immediate owner samples cannot race ahead of the session header.
                thread::park();
                let mut emitted = 1_u8;
                // The startup and finish unparks may coalesce before this thread first runs.
                // Observe the released stop flag before entering a potentially long timeout.
                if reporter_shared.stop.load(Ordering::Acquire) {
                    return emitted;
                }
                while emitted <= NORMALIZATION_WATCHDOG_MAX_SNAPSHOTS {
                    thread::park_timeout(interval);
                    if reporter_shared.stop.load(Ordering::Acquire) {
                        break;
                    }
                    diagnostic_watchdog_emit(&reporter_shared, "watchdog_snapshot");
                    emitted = emitted.saturating_add(1);
                }
                emitted
            })
            .ok()?;
        diagnostic_watchdog_emit(&shared, "watchdog_initial");
        reporter.thread().unpark();
        Some(Self { shared, reporter: Some(reporter), next_call: 0 })
    }

    fn update(&self, update: impl FnOnce(&mut DiagnosticProgress)) {
        let mut progress =
            self.shared.progress.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        update(&mut progress);
        progress.last_change = Instant::now();
    }

    fn emit_owner_sample(&self, owner_sample: DiagnosticOwnerSample, progress: DiagnosticProgress) {
        let mut samples =
            self.shared.owner_samples.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if samples.len() < 12 {
            samples.push(owner_sample);
        }
        drop(samples);
        diagnostic_watchdog_emit_snapshot(&self.shared, "watchdog_owner_sample", Some(progress));
    }

    fn enter_call(&mut self, depth: u32) -> DiagnosticParentState {
        self.next_call = self.next_call.saturating_add(1);
        let call = self.next_call;
        let mut parent = None;
        self.update(|progress| {
            parent = Some(DiagnosticParentState {
                current_call: progress.current_call,
                depth: progress.depth,
                phase: progress.phase,
                expression_slot: progress.expression_slot,
                operator: progress.operator,
                nodes_done: progress.nodes_done,
                nodes_total: progress.nodes_total,
            });
            progress.current_call = call;
            progress.depth = depth;
            progress.phase = DiagnosticPhase::ScopeProof;
            progress.expression_slot = 0;
            progress.operator = "none";
            progress.nodes_done = 0;
            progress.nodes_total = 0;
        });
        parent.expect("watchdog update executes")
    }

    fn complete_call(&self, parent: DiagnosticParentState, error: bool) {
        self.update(|progress| {
            let completed = progress.current_call;
            progress.last_completed = completed;
            progress.phase =
                if error { DiagnosticPhase::Error } else { DiagnosticPhase::CallReturn };
            if parent.current_call != 0 {
                progress.current_call = parent.current_call;
                progress.depth = parent.depth;
                progress.phase = parent.phase;
                progress.expression_slot = parent.expression_slot;
                progress.operator = parent.operator;
                progress.nodes_done = parent.nodes_done;
                progress.nodes_total = parent.nodes_total;
            }
        });
    }

    fn finish(&mut self, error: bool) {
        let Some(reporter) = self.reporter.take() else { return };
        self.update(|progress| {
            progress.phase =
                if error { DiagnosticPhase::Error } else { DiagnosticPhase::CallReturn };
        });
        self.shared.stop.store(true, Ordering::Release);
        reporter.thread().unpark();
        let _ = reporter.join();
        diagnostic_watchdog_emit(&self.shared, "watchdog_terminal");
    }
}

impl Drop for DiagnosticWatchdog {
    fn drop(&mut self) {
        self.finish(true);
    }
}

fn diagnostic_watchdog_emit(shared: &DiagnosticShared, event: &'static str) {
    diagnostic_watchdog_emit_snapshot(shared, event, None);
}

fn diagnostic_watchdog_emit_snapshot(
    shared: &DiagnosticShared,
    event: &'static str,
    frozen: Option<DiagnosticProgress>,
) {
    let mut emitted = shared.emitted_lines.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if *emitted >= 32 {
        return;
    }
    *emitted = emitted.saturating_add(1);
    drop(emitted);
    let progress = frozen.unwrap_or_else(|| {
        *shared.progress.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    });
    #[cfg(test)]
    shared.events.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push(event);
    #[cfg(test)]
    shared.snapshots.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push(progress);
    info!(
        target: NORMALIZATION_TRACE_TARGET,
        event,
        generation = progress.generation,
        current_call = progress.current_call,
        last_completed = progress.last_completed,
        depth = progress.depth,
        phase = ?progress.phase,
        expression_slot = progress.expression_slot,
        operator = progress.operator,
        nodes_done = progress.nodes_done,
        nodes_total = progress.nodes_total,
        product_processed = progress.product_processed,
        product_generated = progress.product_generated,
        product_enqueued = progress.product_enqueued,
        product_processed_current = progress.product_processed_current,
        product_planned_current = progress.product_planned_current,
        product_generation_current = progress.product_generation_current,
        product_enqueued_current = progress.product_enqueued_current,
        product_queue_current = progress.product_queue_current,
        product_output_current = progress.product_output_current,
        relation_processed = progress.relation_processed,
        gc = ?progress.gc,
        owner_census_seq = progress.owner_census_seq,
        owner_census_depth = progress.owner_census_depth,
        owner_census_call = progress.owner_census_call,
        owner_census_product_call_id = progress.owner_census_product_call_id,
        owner_census_phase = ?progress.owner_census_phase,
        owner_census_reason = ?progress.owner_census_reason,
        monomial_allocated_descriptor_slots = progress.owners.monomial_allocated_descriptor_slots,
        monomial_retained_descriptor_slots = progress.owners.monomial_retained_descriptor_slots,
        monomial_reclaimed_descriptor_slots = progress.owners.monomial_reclaimed_descriptor_slots,
        monomial_reachable_descriptor_slots = progress.owners.monomial_reachable_descriptor_slots,
            monomial_reachable_central_factor_entries = progress
                .owners
                .monomial_reachable_central_factor_entries,
            monomial_reachable_ordered_factor_entries = progress
                .owners
                .monomial_reachable_ordered_factor_entries,
            monomial_reachable_max_factor_word = progress
                .owners
                .monomial_reachable_max_factor_word,
        monomial_owned_payload_lower_bound_bytes = progress.owners.monomial_owned_payload_lower_bound_bytes,
        monomial_unreachable_descriptor_slots = progress.owners.monomial_unreachable_descriptor_slots,
        monomial_unreachable_central_factor_entries = progress.owners.monomial_unreachable_central_factor_entries,
        monomial_unreachable_ordered_factor_entries = progress.owners.monomial_unreachable_ordered_factor_entries,
        monomial_unreachable_payload_lower_bound_bytes = progress.owners.monomial_unreachable_payload_lower_bound_bytes,
        monomial_invalid_root_count = progress.owners.monomial_invalid_root_count,
        cache_owner_entries = progress.owners.cache_entries,
        cache_exact_terms = progress.owners.cache_exact_terms,
        cache_exact_terms_peak = progress.owners.cache_exact_terms_peak,
        cache_largest_nf_terms_seen = progress.owners.cache_largest_nf_terms_seen,
        gadget_owner_entries = progress.owners.gadget_entries,
        gadget_exact_terms = progress.owners.gadget_exact_terms,
        gadget_exact_terms_peak = progress.owners.gadget_exact_terms_peak,
        gadget_largest_nf_terms_seen = progress.owners.gadget_largest_nf_terms_seen,
        canonical_rhs_entries = progress.owners.canonical_rhs_entries,
        canonical_rhs_exact_terms = progress.owners.canonical_rhs_exact_terms,
        canonical_rhs_exact_terms_peak = progress.owners.canonical_rhs_exact_terms_peak,
        canonical_rhs_largest_nf_terms = progress.owners.canonical_rhs_largest_nf_terms,
        runtime_entries = progress.owners.runtime_entries,
        runtime_lhs_keys = progress.owners.runtime_lhs_keys,
        additive_plan_nodes = progress.owners.additive_plan_nodes,
        ordinary_product_plan_nodes = progress.owners.ordinary_product_plan_nodes,
        gadget_product_plan_nodes = progress.owners.gadget_product_plan_nodes,
        additive_unique_leaf_refs = progress.owners.additive_unique_leaf_refs,
        additive_unique_leaf_exact_term_refs = progress
            .owners
            .additive_unique_leaf_exact_term_refs,
        additive_largest_leaf_exact_terms = progress
            .owners
            .additive_largest_leaf_exact_terms,
        gadget_product_unique_operand_refs = progress.owners.gadget_product_unique_operand_refs,
        gadget_product_operand_exact_term_refs = progress
            .owners
            .gadget_product_operand_exact_term_refs,
        gadget_product_largest_operand_exact_terms = progress
            .owners
            .gadget_product_largest_operand_exact_terms,
        materialized_leaf_top8 = ?progress.owners.materialized_leaf_top8,
        gadget_product_plans_created = progress.owners.gadget_product_plans_created,
        gadget_product_streamed_executions = progress.owners.gadget_product_streamed_executions,
        gadget_product_zero_weight_skips = progress.owners.gadget_product_zero_weight_skips,
        gadget_product_standalone_materializations = progress
            .owners
            .gadget_product_standalone_materializations,
        gadget_product_planned_pairs = progress.owners.gadget_product_planned_pairs,
        gadget_product_max_streamed_output_terms = progress
            .owners
            .gadget_product_max_streamed_output_terms,
        ordinary_product_plans_created = progress.owners.ordinary_product_plans_created,
        typed_product_candidate_plans = progress.owners.typed_product_candidate_plans,
        typed_product_direct_executions = progress.owners.typed_product_direct_executions,
        typed_product_pair_attempts = progress.owners.typed_product_pair_attempts,
        typed_product_pair_matches = progress.owners.typed_product_pair_matches,
        typed_product_pair_ordinary_fallbacks = progress
            .owners
            .typed_product_pair_ordinary_fallbacks,
        typed_product_standalone_materializations = progress
            .owners
            .typed_product_standalone_materializations,
        ordinary_product_streamed_executions = progress
            .owners
            .ordinary_product_streamed_executions,
        ordinary_product_zero_weight_skips = progress.owners.ordinary_product_zero_weight_skips,
        ordinary_product_standalone_materializations = progress
            .owners
            .ordinary_product_standalone_materializations,
        ordinary_product_planned_pairs = progress.owners.ordinary_product_planned_pairs,
        ordinary_product_max_streamed_output_terms = progress
            .owners
            .ordinary_product_max_streamed_output_terms,
        scalar_action_plans_created = progress.owners.scalar_action_plans_created,
        scalar_action_streamed_executions = progress.owners.scalar_action_streamed_executions,
        scalar_action_zero_weight_skips = progress.owners.scalar_action_zero_weight_skips,
        scalar_action_standalone_materializations = progress
            .owners
            .scalar_action_standalone_materializations,
        scalar_action_reclassified_terms = progress.owners.scalar_action_reclassified_terms,
        scalar_action_reclassified_factors_max = progress
            .owners
            .scalar_action_reclassified_factors_max,
        additive_materializations = progress.owners.additive_materializations,
        additive_materialization_output_terms_total = progress
            .owners
            .additive_materialization_output_terms_total,
        additive_materialization_output_terms_max = progress
            .owners
            .additive_materialization_output_terms_max,
        runtime_lookup_hits = progress.specialization.runtime_lookup_hits,
        runtime_lookup_misses = progress.specialization.runtime_lookup_misses,
        ordinary_specializations_started = progress.specialization.ordinary_specializations_started,
        ordinary_specializations_completed = progress.specialization.ordinary_specializations_completed,
        proof_specializations_started = progress.specialization.proof_specializations_started,
        proof_specializations_completed = progress.specialization.proof_specializations_completed,
        registrations_started = progress.specialization.registrations_started,
        registrations_completed = progress.specialization.registrations_completed,
        rhs_exact_terms_total = progress.specialization.rhs_exact_terms_total,
        rhs_exact_terms_max = progress.specialization.rhs_exact_terms_max,
        interner_existing = progress.specialization.interner_existing,
        interner_inserted = progress.specialization.interner_inserted,
        proof_rollbacks_completed = progress.specialization.proof_rollbacks_completed,
        relation_closures_started = progress.relation_closure.closures_started,
        relation_closures_completed = progress.relation_closure.closures_completed,
        relation_closures_errored = progress.relation_closure.closures_errored,
        relation_active_depth = progress.relation_closure.active_depth,
        relation_closed_relations_present = progress.relation_closure.closed_relations_present,
        relation_initial_terms = progress.relation_closure.initial_terms,
        relation_dequeued = progress.relation_closure.dequeued,
        relation_zero_skipped = progress.relation_closure.zero_skipped,
        relation_nonzero_dequeued = progress.relation_closure.nonzero_dequeued,
        relation_enqueued = progress.relation_closure.enqueued,
        relation_queue_peak = progress.relation_closure.queue_peak,
        relation_duplicate_same_outcome = progress.relation_closure.duplicate_same_outcome,
        relation_duplicate_changed_outcome = progress.relation_closure.duplicate_changed_outcome,
        relation_central_factors_total = progress.relation_closure.central_factors_total,
        relation_central_factors_max = progress.relation_closure.central_factors_max,
        relation_ordered_factors_total = progress.relation_closure.ordered_factors_total,
        relation_ordered_factors_max = progress.relation_closure.ordered_factors_max,
        relation_gadget_attempts = progress.relation_closure.gadget_attempts,
        relation_gadget_matches = progress.relation_closure.gadget_matches,
        relation_gadget_output_terms_total = progress.relation_closure.gadget_output_terms_total,
        relation_gadget_output_terms_max = progress.relation_closure.gadget_output_terms_max,
        relation_whole_closed_probes = progress.relation_closure.whole_closed_probes,
        relation_whole_closed_resolves = progress.relation_closure.whole_closed_resolves,
        relation_whole_closed_matches = progress.relation_closure.whole_closed_matches,
        relation_whole_closed_ambiguities = progress.relation_closure.whole_closed_ambiguities,
        relation_closed_window_probes = progress.relation_closure.closed_window_probes,
        relation_closed_window_interned_hits = progress.relation_closure.closed_window_interned_hits,
        relation_closed_window_resolves = progress.relation_closure.closed_window_resolves,
        relation_closed_window_matches = progress.relation_closure.closed_window_matches,
        relation_closed_window_ambiguities = progress.relation_closure.closed_window_ambiguities,
        relation_closed_subword_matches = progress.relation_closure.closed_subword_matches,
        relation_universal_factors = progress.relation_closure.universal_probes,
        relation_universal_dispatch_hits = progress.relation_closure.universal_dispatch_hits,
        relation_universal_specializations = progress.relation_closure.universal_specializations,
        relation_universal_lhs_candidates = progress.relation_closure.universal_lhs_candidates,
        relation_universal_span_candidates = progress.relation_closure.universal_span_candidates,
        relation_universal_matches = progress.relation_closure.universal_matches,
        relation_universal_ambiguities = progress.relation_closure.universal_ambiguities,
        relation_universal_rewrites = progress.relation_closure.universal_rewrites,
        relation_no_matches = progress.relation_closure.no_matches,
        relation_match_errors = progress.relation_closure.match_errors,
        relation_rhs_splices = progress.relation_closure.rhs_splices,
        relation_rhs_terms_total = progress.relation_closure.rhs_terms_total,
        relation_rhs_terms_max = progress.relation_closure.rhs_terms_max,
        relation_rhs_terms_enqueued = progress.relation_closure.rhs_terms_enqueued,
        relation_monomial_combines = progress.relation_closure.monomial_combines,
        relation_prefix_combines = progress.relation_closure.prefix_combines,
        relation_suffix_combines = progress.relation_closure.suffix_combines,
        relation_result_terms = progress.relation_closure.result_terms,
        relation_final_terms = progress.relation_closure.final_terms,
        timings = ?progress.timings,
        unchanged_ms = u64::try_from(progress.last_change.elapsed().as_millis()).unwrap_or(u64::MAX),
    );
}

fn normalization_watchdog_enabled() -> bool {
    std::env::var(NORMALIZATION_WATCHDOG_ENV)
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn normalization_four_class_census_enabled() -> bool {
    normalization_four_class_census_enabled_from_value(
        std::env::var(NORMALIZATION_FOUR_CLASS_CENSUS_ENV).ok().as_deref(),
    )
}

fn normalization_four_class_census_enabled_from_value(value: Option<&str>) -> bool {
    value.is_some_and(|value| matches!(value, "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn normalization_watchdog_interval() -> Duration {
    let value = std::env::var(NORMALIZATION_WATCHDOG_INTERVAL_ENV).ok();
    normalization_watchdog_interval_from_value(value.as_deref())
}

fn normalization_watchdog_interval_from_value(value: Option<&str>) -> Duration {
    let seconds = value
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map(|seconds| seconds.min(NORMALIZATION_WATCHDOG_MAX_INTERVAL_SECS))
        .unwrap_or(NORMALIZATION_WATCHDOG_INTERVAL.as_secs());
    Duration::from_secs(seconds)
}

#[derive(Debug)]
struct NormalizationTrace {
    active: bool,
    started_at: Option<Instant>,
    lines_emitted: u8,
    terminal_emitted: bool,
    terminal_event: &'static str,
    normalization_calls: u64,
    max_normalization_depth: u32,
    outer_nodes_total: u64,
    nodes_processed: u64,
    nodes_total: u64,
    current_expression_slot: u64,
    current_operator: &'static str,
    current_subphase: &'static str,
    current_normalization_call: u64,
    last_completed_normalization_call: u64,
    focus_normalization_call: Option<u64>,
    focus_expression_slot: Option<u64>,
    focus_tail_nodes: Option<u64>,
    remaining_in_current_normalization: u64,
    subphase_lines_emitted: u8,
    #[cfg(test)]
    subphase_history: Vec<&'static str>,
    #[cfg(test)]
    node_start_history: Vec<(u64, &'static str, u64)>,
    post_lines_emitted: u8,
    #[cfg(test)]
    post_history: Vec<&'static str>,
    #[cfg(test)]
    caller_history: Vec<&'static str>,
    caller_nested_bounds_len: u64,
    caller_nested_uses_len: u64,
    caller_nested_cache_len: u64,
    caller_outer_bounds_len: u64,
    caller_after_bounds_len: u64,
    critical_caller_lines_reserved: u8,
    next_specialized_root_armed: bool,
    next_root_normalize_proof_pending: bool,
    root_exact_terms: u64,
    root_sum_central_factors: u64,
    root_max_central_factors: u64,
    root_sum_ordered_factors: u64,
    root_max_ordered_factors: u64,
    relation_initial: u64,
    relation_processed: u64,
    relation_worklist: u64,
    relation_result: u64,
    relation_peak_worklist: u64,
    relation_rewrites: u64,
    relation_enqueues: u64,
    relation_closed_window_probes: u64,
    relation_universal_factor_probes: u64,
    next_relation_processed_heartbeat: u64,
    next_relation_probe_heartbeat: u64,
    cache_len: u64,
    cache_peak: u64,
    monomial_len: u64,
    owners: DiagnosticOwnerCensus,
    next_node_heartbeat: u64,
    product_heartbeat_interval: u64,
    next_product_generated_heartbeat: u64,
    next_product_processed_heartbeat: u64,
    product_calls: u64,
    product_planned: u64,
    product_generated: u64,
    product_processed: u64,
    product_rewrites: u64,
    product_enqueued: u64,
    product_peak_queue: u64,
    product_current_queue: u64,
    product_current_output: u64,
    product_max_left_terms: u64,
    product_max_right_terms: u64,
    product_max_output_terms: u64,
    scalar_calls: u64,
    scalar_not_applicable: u64,
    scalar_opaque: u64,
    scalar_left: u64,
    scalar_right: u64,
    scalar_both: u64,
    scalar_reclassified_terms: u64,
    scalar_reclassified_factors: u64,
    last_product_heartbeat_operator: &'static str,
    product_heartbeat_saw_matrix_multiply: bool,
}

impl NormalizationTrace {
    fn new() -> Self {
        Self {
            active: false,
            started_at: None,
            lines_emitted: 0,
            terminal_emitted: false,
            terminal_event: "none",
            normalization_calls: 0,
            max_normalization_depth: 0,
            outer_nodes_total: 0,
            nodes_processed: 0,
            nodes_total: 0,
            current_expression_slot: 0,
            current_operator: "none",
            current_subphase: "none",
            current_normalization_call: 0,
            last_completed_normalization_call: 0,
            focus_normalization_call: None,
            focus_expression_slot: None,
            focus_tail_nodes: None,
            remaining_in_current_normalization: 0,
            subphase_lines_emitted: 0,
            #[cfg(test)]
            subphase_history: Vec::new(),
            #[cfg(test)]
            node_start_history: Vec::new(),
            post_lines_emitted: 0,
            #[cfg(test)]
            post_history: Vec::new(),
            #[cfg(test)]
            caller_history: Vec::new(),
            caller_nested_bounds_len: 0,
            caller_nested_uses_len: 0,
            caller_nested_cache_len: 0,
            caller_outer_bounds_len: 0,
            caller_after_bounds_len: 0,
            critical_caller_lines_reserved: 0,
            next_specialized_root_armed: false,
            next_root_normalize_proof_pending: false,
            root_exact_terms: 0,
            root_sum_central_factors: 0,
            root_max_central_factors: 0,
            root_sum_ordered_factors: 0,
            root_max_ordered_factors: 0,
            relation_initial: 0,
            relation_processed: 0,
            relation_worklist: 0,
            relation_result: 0,
            relation_peak_worklist: 0,
            relation_rewrites: 0,
            relation_enqueues: 0,
            relation_closed_window_probes: 0,
            relation_universal_factor_probes: 0,
            next_relation_processed_heartbeat: 1_000,
            next_relation_probe_heartbeat: 100_000,
            cache_len: 0,
            cache_peak: 0,
            monomial_len: 0,
            owners: DiagnosticOwnerCensus::default(),
            next_node_heartbeat: NORMALIZATION_NODE_HEARTBEAT,
            product_heartbeat_interval: PRODUCT_PROCESSED_HEARTBEAT,
            next_product_generated_heartbeat: PRODUCT_PROCESSED_HEARTBEAT,
            next_product_processed_heartbeat: PRODUCT_PROCESSED_HEARTBEAT,
            product_calls: 0,
            product_planned: 0,
            product_generated: 0,
            product_processed: 0,
            product_rewrites: 0,
            product_enqueued: 0,
            product_peak_queue: 0,
            product_current_queue: 0,
            product_current_output: 0,
            product_max_left_terms: 0,
            product_max_right_terms: 0,
            product_max_output_terms: 0,
            scalar_calls: 0,
            scalar_not_applicable: 0,
            scalar_opaque: 0,
            scalar_left: 0,
            scalar_right: 0,
            scalar_both: 0,
            scalar_reclassified_terms: 0,
            scalar_reclassified_factors: 0,
            last_product_heartbeat_operator: "none",
            product_heartbeat_saw_matrix_multiply: false,
        }
    }

    fn activate(&mut self, nodes_total: u64, depth: u32, monomial_len: usize) {
        self.active = true;
        self.started_at = Some(Instant::now());
        self.normalization_calls = 1;
        self.current_normalization_call = 1;
        self.max_normalization_depth = depth;
        self.outer_nodes_total = nodes_total;
        self.nodes_total = nodes_total;
        self.monomial_len = u64::try_from(monomial_len).unwrap_or(u64::MAX);
        if self.focus_normalization_call.is_some_and(|call| call > 1) {
            self.critical_caller_lines_reserved = NORMALIZATION_TRACE_CRITICAL_CALLER_RESERVE;
        }
    }

    fn record_nested_normalization(&mut self, nodes_total: u64, depth: u32) {
        if !self.active {
            return;
        }
        self.normalization_calls = self.normalization_calls.saturating_add(1);
        self.current_normalization_call = self.normalization_calls;
        self.max_normalization_depth = self.max_normalization_depth.max(depth);
        self.nodes_total = self.nodes_total.saturating_add(nodes_total);
    }

    fn record_nested_nodes(&mut self, nodes_total: u64) {
        if self.active {
            self.nodes_total = self.nodes_total.saturating_add(nodes_total);
        }
    }

    fn record_scalar_call(&mut self) {
        if self.active {
            self.scalar_calls = self.scalar_calls.saturating_add(1);
        }
    }

    fn record_scalar_not_applicable(&mut self) {
        if self.active {
            self.scalar_not_applicable = self.scalar_not_applicable.saturating_add(1);
        }
    }

    fn record_scalar_opaque(&mut self) {
        if self.active {
            self.scalar_opaque = self.scalar_opaque.saturating_add(1);
        }
    }

    fn record_scalar_left(&mut self) {
        if self.active {
            self.scalar_left = self.scalar_left.saturating_add(1);
        }
    }

    fn record_scalar_right(&mut self) {
        if self.active {
            self.scalar_right = self.scalar_right.saturating_add(1);
        }
    }

    fn record_scalar_both(&mut self) {
        if self.active {
            self.scalar_both = self.scalar_both.saturating_add(1);
        }
    }

    fn record_scalar_reclassification(&mut self, terms: usize, factors: usize) {
        if !self.active {
            return;
        }
        self.scalar_reclassified_terms =
            self.scalar_reclassified_terms.max(u64::try_from(terms).unwrap_or(u64::MAX));
        self.scalar_reclassified_factors =
            self.scalar_reclassified_factors.max(u64::try_from(factors).unwrap_or(u64::MAX));
    }

    fn enter_subphase(&mut self, subphase: &'static str) {
        self.current_subphase = subphase;
        if !self.active ||
            self.focus_normalization_call != Some(self.current_normalization_call) ||
            self.focus_expression_slot.is_some_and(|slot| slot != self.current_expression_slot) ||
            self.subphase_lines_emitted >= NORMALIZATION_TRACE_SUBPHASE_LINE_BUDGET
        {
            return;
        }
        if self.emit("normalize_subphase", self.nodes_processed, self.nodes_total, false) {
            self.subphase_lines_emitted = self.subphase_lines_emitted.saturating_add(1);
            #[cfg(test)]
            self.subphase_history.push(subphase);
        }
    }

    fn emit_node_start(&mut self, remaining_in_current_normalization: u64) {
        self.current_subphase = "evaluate_node:start";
        self.remaining_in_current_normalization = remaining_in_current_normalization;
        if !self.active ||
            self.focus_normalization_call != Some(self.current_normalization_call) ||
            !self.focus_tail_nodes.is_some_and(|tail| remaining_in_current_normalization <= tail)
        {
            return;
        }
        if self.emit("normalize_node_start", self.nodes_processed, self.nodes_total, false) {
            #[cfg(test)]
            self.node_start_history.push((
                self.current_expression_slot,
                self.current_operator,
                remaining_in_current_normalization,
            ));
        }
    }

    fn enter_postphase(&mut self, postphase: &'static str) {
        self.current_subphase = postphase;
        if !self.active ||
            self.focus_normalization_call != Some(self.current_normalization_call) ||
            self.post_lines_emitted >= NORMALIZATION_TRACE_POST_LINE_BUDGET
        {
            return;
        }
        if self.emit("normalize_post", self.nodes_processed, self.nodes_total, false) {
            self.post_lines_emitted = self.post_lines_emitted.saturating_add(1);
            #[cfg(test)]
            self.post_history.push(postphase);
        }
    }

    fn caller_trace_selected(&self) -> bool {
        self.active && self.focus_normalization_call == Some(self.last_completed_normalization_call)
    }

    fn focused_invocation_selected(&self) -> bool {
        self.active && self.focus_normalization_call == Some(self.current_normalization_call)
    }

    fn record_completed_invocation(&mut self, completed: u64) {
        self.last_completed_normalization_call = completed;
        if self.active && self.focus_normalization_call == Some(completed) {
            self.next_specialized_root_armed = true;
        }
    }

    fn claim_next_specialized_root(&mut self) -> bool {
        if !self.active || !self.next_specialized_root_armed {
            return false;
        }
        self.next_specialized_root_armed = false;
        true
    }

    fn enter_caller_phase(&mut self, phase: &'static str, critical: bool) {
        self.current_subphase = phase;
        if !self.caller_trace_selected() {
            return;
        }
        let emitted = if critical {
            self.emit_critical_caller("normalize_caller", self.nodes_processed, self.nodes_total)
        } else {
            self.emit("normalize_caller", self.nodes_processed, self.nodes_total, false)
        };
        if emitted {
            #[cfg(test)]
            self.caller_history.push(phase);
        }
    }

    fn enter_focused_invocation_phase(&mut self, phase: &'static str) {
        self.current_subphase = phase;
        if !self.focused_invocation_selected() {
            return;
        }
        if self.emit("normalize_caller", self.nodes_processed, self.nodes_total, false) {
            #[cfg(test)]
            self.caller_history.push(phase);
        }
    }

    fn enter_next_root_phase(&mut self, phase: &'static str, claimed: bool) {
        self.current_subphase = phase;
        if !claimed {
            return;
        }
        if self.emit_critical_caller("normalize_caller", self.nodes_processed, self.nodes_total) {
            #[cfg(test)]
            self.caller_history.push(phase);
        }
    }

    fn emit_critical_caller(
        &mut self,
        event: &'static str,
        nodes_processed: u64,
        nodes_total: u64,
    ) -> bool {
        if self.critical_caller_lines_reserved == 0 {
            return false;
        }
        if self.emit_with_reservation(event, nodes_processed, nodes_total, false, true) {
            self.critical_caller_lines_reserved =
                self.critical_caller_lines_reserved.saturating_sub(1);
            true
        } else {
            false
        }
    }

    fn relation_trace_selected(&self) -> bool {
        self.active && self.focus_normalization_call == Some(self.current_normalization_call)
    }

    fn record_relation_processed(&mut self, worklist: usize, result: usize) {
        if !self.relation_trace_selected() {
            return;
        }
        self.relation_processed = self.relation_processed.saturating_add(1);
        self.relation_worklist = u64::try_from(worklist).unwrap_or(u64::MAX);
        self.relation_result = u64::try_from(result).unwrap_or(u64::MAX);
        self.relation_peak_worklist = self.relation_peak_worklist.max(self.relation_worklist);
        if self.relation_processed >= self.next_relation_processed_heartbeat {
            self.emit("relation_heartbeat", self.nodes_processed, self.nodes_total, false);
            self.next_relation_processed_heartbeat =
                self.next_relation_processed_heartbeat.saturating_add(1_000);
        }
    }

    fn record_relation_probe(&mut self, closed: bool) {
        if !self.relation_trace_selected() {
            return;
        }
        if closed {
            self.relation_closed_window_probes =
                self.relation_closed_window_probes.saturating_add(1);
        } else {
            self.relation_universal_factor_probes =
                self.relation_universal_factor_probes.saturating_add(1);
        }
        let probes = self
            .relation_closed_window_probes
            .saturating_add(self.relation_universal_factor_probes);
        if probes >= self.next_relation_probe_heartbeat {
            self.emit("relation_heartbeat", self.nodes_processed, self.nodes_total, false);
            self.next_relation_probe_heartbeat =
                self.next_relation_probe_heartbeat.saturating_add(100_000);
        }
    }

    fn emit(
        &mut self,
        event: &'static str,
        nodes_processed: u64,
        nodes_total: u64,
        final_line: bool,
    ) -> bool {
        self.emit_with_reservation(event, nodes_processed, nodes_total, final_line, false)
    }

    fn emit_with_reservation(
        &mut self,
        event: &'static str,
        nodes_processed: u64,
        nodes_total: u64,
        final_line: bool,
        consume_caller_reservation: bool,
    ) -> bool {
        if !self.active {
            return false;
        }
        let limit = if final_line {
            NORMALIZATION_TRACE_LINE_BUDGET
        } else if consume_caller_reservation {
            NORMALIZATION_TRACE_LINE_BUDGET.saturating_sub(1)
        } else {
            NORMALIZATION_TRACE_LINE_BUDGET
                .saturating_sub(1)
                .saturating_sub(self.critical_caller_lines_reserved)
        };
        if self.lines_emitted >= limit {
            return false;
        }
        self.lines_emitted = self.lines_emitted.saturating_add(1);
        if final_line {
            self.terminal_emitted = true;
            self.terminal_event = event;
        }
        info!(
            target: NORMALIZATION_TRACE_TARGET,
            event,
            elapsed_ms = self
                .started_at
                .map(|started| u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX))
                .unwrap_or(0),
            normalization_calls = self.normalization_calls,
            max_normalization_depth = self.max_normalization_depth,
            terminal_emitted = self.terminal_emitted,
            terminal_event = self.terminal_event,
            outer_nodes_total = self.outer_nodes_total,
            nodes_processed,
            nodes_total,
            current_expression_slot = self.current_expression_slot,
            current_operator = self.current_operator,
            current_subphase = self.current_subphase,
            current_normalization_call = self.current_normalization_call,
            last_completed_normalization_call = self.last_completed_normalization_call,
            focus_normalization_call = self.focus_normalization_call,
            focus_expression_slot = self.focus_expression_slot,
            focus_tail_nodes = self.focus_tail_nodes,
            remaining_in_current_normalization = self.remaining_in_current_normalization,
            subphase_lines_emitted = self.subphase_lines_emitted,
            post_lines_emitted = self.post_lines_emitted,
            caller_nested_bounds_len = self.caller_nested_bounds_len,
            caller_nested_uses_len = self.caller_nested_uses_len,
            caller_nested_cache_len = self.caller_nested_cache_len,
            caller_outer_bounds_len = self.caller_outer_bounds_len,
            caller_after_bounds_len = self.caller_after_bounds_len,
            critical_caller_lines_reserved = self.critical_caller_lines_reserved,
            next_specialized_root_armed = self.next_specialized_root_armed,
            next_root_normalize_proof_pending = self.next_root_normalize_proof_pending,
            root_exact_terms = self.root_exact_terms,
            root_sum_central_factors = self.root_sum_central_factors,
            root_max_central_factors = self.root_max_central_factors,
            root_sum_ordered_factors = self.root_sum_ordered_factors,
            root_max_ordered_factors = self.root_max_ordered_factors,
            relation_initial = self.relation_initial,
            relation_processed = self.relation_processed,
            relation_worklist = self.relation_worklist,
            relation_result = self.relation_result,
            relation_peak_worklist = self.relation_peak_worklist,
            relation_rewrites = self.relation_rewrites,
            relation_enqueues = self.relation_enqueues,
            relation_closed_window_probes = self.relation_closed_window_probes,
            relation_universal_factor_probes = self.relation_universal_factor_probes,
            cache_len = self.cache_len,
            cache_peak = self.cache_peak,
            monomial_len = self.monomial_len,
            monomial_allocated_descriptor_slots = self.owners.monomial_allocated_descriptor_slots,
            monomial_retained_descriptor_slots = self.owners.monomial_retained_descriptor_slots,
            monomial_reclaimed_descriptor_slots = self.owners.monomial_reclaimed_descriptor_slots,
            monomial_reachable_descriptor_slots = self.owners.monomial_reachable_descriptor_slots,
            monomial_reachable_central_factor_entries = self
                .owners
                .monomial_reachable_central_factor_entries,
            monomial_reachable_ordered_factor_entries = self
                .owners
                .monomial_reachable_ordered_factor_entries,
            monomial_reachable_max_factor_word = self
                .owners
                .monomial_reachable_max_factor_word,
            monomial_owned_payload_lower_bound_bytes = self.owners.monomial_owned_payload_lower_bound_bytes,
            monomial_unreachable_descriptor_slots = self.owners.monomial_unreachable_descriptor_slots,
            monomial_unreachable_central_factor_entries = self.owners.monomial_unreachable_central_factor_entries,
            monomial_unreachable_ordered_factor_entries = self.owners.monomial_unreachable_ordered_factor_entries,
            monomial_unreachable_payload_lower_bound_bytes = self.owners.monomial_unreachable_payload_lower_bound_bytes,
            cache_owner_entries = self.owners.cache_entries,
            cache_exact_terms = self.owners.cache_exact_terms,
            cache_exact_terms_peak = self.owners.cache_exact_terms_peak,
            cache_largest_nf_terms_seen = self.owners.cache_largest_nf_terms_seen,
            gadget_owner_entries = self.owners.gadget_entries,
            gadget_exact_terms = self.owners.gadget_exact_terms,
            gadget_exact_terms_peak = self.owners.gadget_exact_terms_peak,
            gadget_largest_nf_terms_seen = self.owners.gadget_largest_nf_terms_seen,
            canonical_rhs_entries = self.owners.canonical_rhs_entries,
            canonical_rhs_exact_terms = self.owners.canonical_rhs_exact_terms,
            canonical_rhs_exact_terms_peak = self.owners.canonical_rhs_exact_terms_peak,
            canonical_rhs_largest_nf_terms = self.owners.canonical_rhs_largest_nf_terms,
            runtime_entries = self.owners.runtime_entries,
            runtime_lhs_keys = self.owners.runtime_lhs_keys,
            additive_plan_nodes = self.owners.additive_plan_nodes,
            ordinary_product_plan_nodes = self.owners.ordinary_product_plan_nodes,
            gadget_product_plan_nodes = self.owners.gadget_product_plan_nodes,
            additive_unique_leaf_refs = self.owners.additive_unique_leaf_refs,
            additive_unique_leaf_exact_term_refs = self
                .owners
                .additive_unique_leaf_exact_term_refs,
            additive_largest_leaf_exact_terms = self
                .owners
                .additive_largest_leaf_exact_terms,
            gadget_product_unique_operand_refs = self
                .owners
                .gadget_product_unique_operand_refs,
            gadget_product_operand_exact_term_refs = self
                .owners
                .gadget_product_operand_exact_term_refs,
            gadget_product_largest_operand_exact_terms = self
                .owners
                .gadget_product_largest_operand_exact_terms,
            materialized_leaf_top8 = ?self.owners.materialized_leaf_top8,
            gadget_product_plans_created = self.owners.gadget_product_plans_created,
            gadget_product_streamed_executions = self
                .owners
                .gadget_product_streamed_executions,
            gadget_product_zero_weight_skips = self.owners.gadget_product_zero_weight_skips,
            gadget_product_standalone_materializations = self
                .owners
                .gadget_product_standalone_materializations,
            gadget_product_planned_pairs = self.owners.gadget_product_planned_pairs,
            gadget_product_max_streamed_output_terms = self
                .owners
                .gadget_product_max_streamed_output_terms,
            ordinary_product_plans_created = self.owners.ordinary_product_plans_created,
            typed_product_candidate_plans = self.owners.typed_product_candidate_plans,
            typed_product_direct_executions = self.owners.typed_product_direct_executions,
            typed_product_pair_attempts = self.owners.typed_product_pair_attempts,
            typed_product_pair_matches = self.owners.typed_product_pair_matches,
            typed_product_pair_ordinary_fallbacks = self
                .owners
                .typed_product_pair_ordinary_fallbacks,
            typed_product_standalone_materializations = self
                .owners
                .typed_product_standalone_materializations,
            ordinary_product_streamed_executions = self
                .owners
                .ordinary_product_streamed_executions,
            ordinary_product_zero_weight_skips = self.owners.ordinary_product_zero_weight_skips,
            ordinary_product_standalone_materializations = self
                .owners
                .ordinary_product_standalone_materializations,
            ordinary_product_planned_pairs = self.owners.ordinary_product_planned_pairs,
            ordinary_product_max_streamed_output_terms = self
                .owners
                .ordinary_product_max_streamed_output_terms,
            scalar_action_plans_created = self.owners.scalar_action_plans_created,
            scalar_action_streamed_executions = self.owners.scalar_action_streamed_executions,
            scalar_action_zero_weight_skips = self.owners.scalar_action_zero_weight_skips,
            scalar_action_standalone_materializations = self
                .owners
                .scalar_action_standalone_materializations,
            scalar_action_reclassified_terms = self.owners.scalar_action_reclassified_terms,
            scalar_action_reclassified_factors_max = self
                .owners
                .scalar_action_reclassified_factors_max,
            additive_materializations = self.owners.additive_materializations,
            additive_materialization_output_terms_total = self
                .owners
                .additive_materialization_output_terms_total,
            additive_materialization_output_terms_max = self
                .owners
                .additive_materialization_output_terms_max,
            product_calls = self.product_calls,
            product_planned = self.product_planned,
            product_generated = self.product_generated,
            product_processed = self.product_processed,
            product_rewrites = self.product_rewrites,
            product_enqueued = self.product_enqueued,
            product_peak_queue = self.product_peak_queue,
            product_current_queue = self.product_current_queue,
            product_current_output = self.product_current_output,
            product_max_left_terms = self.product_max_left_terms,
            product_max_right_terms = self.product_max_right_terms,
            product_max_output_terms = self.product_max_output_terms,
            scalar_calls = self.scalar_calls,
            scalar_not_applicable = self.scalar_not_applicable,
            scalar_opaque = self.scalar_opaque,
            scalar_left = self.scalar_left,
            scalar_right = self.scalar_right,
            scalar_both = self.scalar_both,
            scalar_reclassified_terms = self.scalar_reclassified_terms,
            scalar_reclassified_factors = self.scalar_reclassified_factors,
            last_product_heartbeat_operator = self.last_product_heartbeat_operator,
            product_heartbeat_saw_matrix_multiply = self.product_heartbeat_saw_matrix_multiply,
            "operational normalization trace"
        );
        true
    }
}

fn normalization_operator_category(operator: &ValueOperator) -> &'static str {
    match operator {
        ValueOperator::Matrix(MatrixOperation::Add) => "add",
        ValueOperator::Matrix(MatrixOperation::Subtract) => "subtract",
        ValueOperator::Matrix(MatrixOperation::Multiply) => "multiply",
        ValueOperator::Matrix(MatrixOperation::Negate) => "negate",
        ValueOperator::Matrix(MatrixOperation::Scale) => "scale",
        ValueOperator::Matrix(MatrixOperation::Transpose) => "transpose",
        ValueOperator::Matrix(MatrixOperation::Slice { .. }) => "slice",
        ValueOperator::Matrix(MatrixOperation::IndexedSlice { .. }) => "indexed_slice",
        ValueOperator::Matrix(MatrixOperation::View { .. }) => "view",
        ValueOperator::Matrix(MatrixOperation::Concat { .. }) => "concat",
        ValueOperator::Matrix(MatrixOperation::Tensor { .. }) => "tensor",
        ValueOperator::Matrix(MatrixOperation::CrtRecompose { .. }) => "crt_recompose",
        ValueOperator::Matrix(MatrixOperation::ExtractCoefficient { .. }) => "extract_coefficient",
        ValueOperator::Matrix(MatrixOperation::LiftConstantPolynomial { .. }) => {
            "lift_constant_polynomial"
        }
        ValueOperator::Scalar(_) => "scalar",
        ValueOperator::Transform(_) => "transform",
        ValueOperator::Source(_) => "source",
        ValueOperator::Sampler { .. } | ValueOperator::Sample { .. } => "sample",
        ValueOperator::ProgramCall { .. } => "program_call",
        ValueOperator::OpaqueFamilyElement { .. } | ValueOperator::ExplicitElement { .. } => {
            "family_element"
        }
        ValueOperator::DeterministicHash(_) => "deterministic_hash",
        ValueOperator::Trapdoor(_) => "trapdoor",
        ValueOperator::Argument { .. } => "argument",
        ValueOperator::Constant(_) => "constant",
        ValueOperator::IndexMap { .. } => "index_map",
        ValueOperator::ExtractCoefficient { .. } => "extract_coefficient",
    }
}

fn normalization_trace_focus_call_from_env() -> Option<u64> {
    std::env::var(NORMALIZATION_TRACE_FOCUS_CALL_ENV)
        .ok()
        .and_then(|value| normalization_trace_positive_u64(&value))
}

fn normalization_trace_focus_expression_slot_from_env() -> Option<u64> {
    std::env::var(NORMALIZATION_TRACE_FOCUS_EXPRESSION_SLOT_ENV)
        .ok()
        .and_then(|value| normalization_trace_expression_slot(&value))
}

fn normalization_trace_focus_tail_nodes_from_env() -> Option<u64> {
    std::env::var(NORMALIZATION_TRACE_FOCUS_TAIL_NODES_ENV)
        .ok()
        .and_then(|value| normalization_trace_positive_u64(&value))
}

fn normalization_trace_positive_u64(value: &str) -> Option<u64> {
    value.parse::<u64>().ok().filter(|value| *value > 0)
}

fn normalization_trace_expression_slot(value: &str) -> Option<u64> {
    value.parse::<u64>().ok()
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
    kind: RelationOutcomeKind,
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

/// Exact state retained only while one root is being walked. Addition nodes are persistent and
/// immutable; public results and every non-additive operator still receive a materialized NF.
#[derive(Clone, Debug)]
enum NodeExactState {
    Materialized { authority: ExactPlanAuthority, normal_form: Arc<PolynomialNF> },
    Additive(Arc<AdditiveExactPlan>),
    Product(Arc<ProductExactPlan>),
    GadgetProduct(Arc<GadgetProductExactPlan>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ExactPlanAuthority {
    expressions: ArenaToken,
    monomials: ArenaToken,
    scope: ValueProgramId,
    relations: Option<FrozenGeneration>,
    matrix_type: ResolvedMatrixType,
}

#[derive(Debug)]
struct AdditiveExactPlan {
    id: u64,
    authority: ExactPlanAuthority,
    left: NodeExactState,
    right: NodeExactState,
    subtract_right: bool,
}

#[derive(Debug)]
struct ProductExactPlan {
    id: u64,
    authority: ExactPlanAuthority,
    expression: ExprId,
    left_expression: ExprId,
    right_expression: ExprId,
    left_type: ResolvedMatrixType,
    right_type: ResolvedMatrixType,
    mode: ProductMode,
    left: NodeExactState,
    right: NodeExactState,
}

#[derive(Clone, Debug)]
enum ProductMode {
    Ordinary,
    TypedGadgetCandidate,
    ScalarAction(ScalarActionExactPlan),
}

#[derive(Clone, Debug)]
struct ScalarActionExactPlan {
    scalar_on_left: bool,
    scalar_expression: ExprId,
    matrix_expression: ExprId,
    scalar_type: ResolvedMatrixType,
    matrix_type: ResolvedMatrixType,
    centralized_scalar: Arc<PolynomialNF>,
}

#[derive(Debug)]
struct FlattenedAdditiveExactPlan {
    id: u64,
    additive_ids: BTreeSet<u64>,
    leaves: HashMap<usize, (Arc<PolynomialNF>, BigInt)>,
    additive_outputs: BTreeMap<u64, (Arc<AdditiveExactPlan>, BigInt)>,
    products: BTreeMap<u64, (Arc<ProductExactPlan>, BigInt, usize, bool)>,
    gadget_products: BTreeMap<u64, (Arc<GadgetProductExactPlan>, BigInt)>,
}

#[derive(Debug)]
enum ExactMaterializationFrame {
    Enter(NodeExactState),
    FinishAdditive(FlattenedAdditiveExactPlan),
    FinishProduct(Arc<ProductExactPlan>),
    FinishGadgetProduct(Arc<GadgetProductExactPlan>),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ProductEligibilityStats {
    candidates: usize,
    consumer_edges: usize,
    queue_pops: usize,
}

#[derive(Debug)]
struct GadgetProductExactPlan {
    id: u64,
    authority: ExactPlanAuthority,
    expression: ExprId,
    left_expression: ExprId,
    right_expression: ExprId,
    left_type: ResolvedMatrixType,
    right_type: ResolvedMatrixType,
    left: Arc<PolynomialNF>,
    right: Arc<PolynomialNF>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct GadgetProductPlanCounters {
    plans_created: u64,
    streamed_executions: u64,
    zero_weight_skips: u64,
    standalone_materializations: u64,
    planned_pairs: u64,
    max_streamed_output_terms: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ProductPlanCounters {
    plans_created: u64,
    typed_candidate_plans: u64,
    typed_direct_executions: u64,
    typed_pair_attempts: u64,
    typed_pair_matches: u64,
    typed_pair_ordinary_fallbacks: u64,
    typed_standalone_materializations: u64,
    streamed_executions: u64,
    zero_weight_skips: u64,
    standalone_materializations: u64,
    planned_pairs: u64,
    max_streamed_output_terms: u64,
    scalar_action_plans_created: u64,
    scalar_action_streamed_executions: u64,
    scalar_action_zero_weight_skips: u64,
    scalar_action_standalone_materializations: u64,
    scalar_action_reclassified_terms: u64,
    scalar_action_reclassified_factors_max: u64,
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
    /// Add/Sub exact plans corresponding to cache entries whose public `exact_nf` is temporarily
    /// absent. Entries are removed at the same last-use boundary as the value cache.
    exact_plans: BTreeMap<ExprId, Arc<AdditiveExactPlan>>,
    product_plans: BTreeMap<ExprId, Arc<ProductExactPlan>>,
    gadget_product_plans: BTreeMap<ExprId, Arc<GadgetProductExactPlan>>,
    next_exact_plan_id: u64,
    product_plan_counters: ProductPlanCounters,
    gadget_product_counters: GadgetProductPlanCounters,
    /// Semantic reverse-edge authority for the narrow deferred gadget-product optimization.
    /// This is populated independently of watchdog diagnostics and contains only non-root
    /// Multiply nodes whose complete real consumer set consists of Add/Sub nodes.
    deferred_gadget_products: BTreeSet<ExprId>,
    /// Semantic reverse-edge authority for ordinary product plans. A member is a non-root,
    /// non-scalar Multiply whose complete real consumer set is Add/Sub or another member and
    /// which has no structural hold.
    deferred_products: BTreeSet<ExprId>,
    deferred_scalar_actions: BTreeSet<ExprId>,
    /// Watchdog-only reverse-edge summary used to explain why a materialized Multiply would not
    /// qualify for the experimental ordinary-product deferral policy. It is never consulted by
    /// normalization semantics.
    diagnostic_product_consumers: BTreeMap<ExprId, DiagnosticProductConsumerCounts>,
    diagnostic_product_evaluations: BTreeMap<ExprId, DiagnosticProductEvaluationSnapshot>,
    diagnostic_product_root: Option<ExprId>,
    /// Watchdog-only provenance for exact NFs produced after forcing a deferred plan. Keying by
    /// the resulting expression keeps this out of semantic identity and all normalization keys.
    diagnostic_materialization_origins: BTreeMap<ExprId, DiagnosticMaterializedLeafOrigin>,
    /// Watchdog-only attachment from immutable NF allocation identity to its origin. This keeps
    /// diagnostic metadata out of every exact-plan node when the watchdog is disabled.
    diagnostic_materialized_leaf_origins: HashMap<usize, DiagnosticMaterializedLeafAttachment>,
    diagnostic_current_expression: Option<ExprId>,
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
    owner_counters: NormalizerOwnerCounters,
    counters: NormalizationCounters,
    trace: NormalizationTrace,
    watchdog: Option<DiagnosticWatchdog>,
    /// Expensive exact-plan relation census. It is independent of the lightweight watchdog and
    /// remains disabled unless explicitly requested for a bounded diagnostic run.
    four_class_census_enabled: bool,
    watchdog_generation: u64,
    watchdog_product_processed: u64,
    watchdog_product_generated: u64,
    watchdog_product_enqueued: u64,
    watchdog_product_processed_current: u64,
    watchdog_product_planned_current: u64,
    watchdog_product_generation_current: u64,
    watchdog_product_enqueued_current: u64,
    watchdog_product_queue_current: u64,
    watchdog_product_output_current: u64,
    watchdog_relation_processed: u64,
    watchdog_specialization: DiagnosticSpecializationCounters,
    watchdog_relation_closure: DiagnosticRelationCounters,
    watchdog_timings: DiagnosticTimings,
    /// Monomial IDs retained solely by suspended nested-normalizer callers. Diagnostic only.
    suspended_owner_roots: Vec<MonomialId>,
    owner_census_samples: u8,
    owner_census_seq: u64,
    watchdog_product_call_id: u64,
    largest_sampled_product_planned: u64,
    large_product_pairs_sampled: u8,
    current_large_product_sampled: bool,
    next_retained_census_milestone: u64,
    #[cfg(test)]
    watchdog_hot_publish_count: u64,
    normalization_depth: u32,
    #[cfg(test)]
    trace_product_heartbeat_interval: u64,
    #[cfg(test)]
    trace_focus_call_override: Option<Option<u64>>,
    #[cfg(test)]
    trace_focus_expression_slot_override: Option<Option<u64>>,
    #[cfg(test)]
    trace_focus_tail_nodes_override: Option<Option<u64>>,
    #[cfg(test)]
    watchdog_enabled_override: Option<bool>,
    #[cfg(test)]
    watchdog_interval_override: Option<Duration>,
    #[cfg(test)]
    last_watchdog_events: Vec<&'static str>,
    #[cfg(test)]
    last_watchdog_snapshots: Vec<DiagnosticProgress>,
    #[cfg(test)]
    relation_matcher_publish_observer: Option<Box<dyn FnMut(DiagnosticRelationCounters)>>,
    relation_rewriting_enabled: bool,
    fold_final_no_match: bool,
    /// Slots below this outer-call high-water are externally observable and remain pinned even
    /// when no current normalization owner references them.
    protected_monomial_prefix: usize,
    monomial_gc_allocation_threshold_bytes: u64,
    gc_counters: DiagnosticGcCounters,
    exact_plan_materializations: u64,
    exact_plan_materialization_output_terms_total: u64,
    exact_plan_materialization_output_terms_max: u64,
}

impl<'a> Normalizer<'a> {
    fn restored_owner_counters(
        saved: NormalizerOwnerCounters,
        nested: NormalizerOwnerCounters,
    ) -> NormalizerOwnerCounters {
        NormalizerOwnerCounters {
            cache_exact_terms: saved.cache_exact_terms,
            cache_exact_terms_peak: saved.cache_exact_terms_peak.max(nested.cache_exact_terms_peak),
            cache_largest_nf_terms_seen: saved
                .cache_largest_nf_terms_seen
                .max(nested.cache_largest_nf_terms_seen),
            gadget_exact_terms: saved.gadget_exact_terms,
            gadget_exact_terms_peak: saved
                .gadget_exact_terms_peak
                .max(nested.gadget_exact_terms_peak),
            gadget_largest_nf_terms_seen: saved
                .gadget_largest_nf_terms_seen
                .max(nested.gadget_largest_nf_terms_seen),
        }
    }

    fn exact_terms(value: &AnalyzedValue) -> u64 {
        value.exact_nf.as_ref().map_or(0, |normal_form| {
            u64::try_from(normal_form.exact_terms.len()).unwrap_or(u64::MAX)
        })
    }

    fn exact_plan_authority(
        &self,
        expression: ExprId,
    ) -> Result<ExactPlanAuthority, NormalizeError> {
        let ResolvedValueType::Matrix(matrix_type) = self.expressions.value_type(expression)?
        else {
            return Err(NormalizeError::InvalidExactPlan { reason: "non-matrix exact state" });
        };
        let relations = self.relations.map(RelationRegistry::frozen_generation).transpose()?;
        Ok(ExactPlanAuthority {
            expressions: self.expressions.token(),
            monomials: self.monomials.token(),
            scope: self.scope,
            relations,
            matrix_type: matrix_type.clone(),
        })
    }

    fn validate_exact_plan_authority(
        &self,
        authority: &ExactPlanAuthority,
    ) -> Result<(), NormalizeError> {
        if authority.expressions != self.expressions.token() ||
            authority.monomials != self.monomials.token() ||
            authority.scope != self.scope
        {
            return Err(NormalizeError::InvalidExactPlan { reason: "foreign exact authority" });
        }
        let relations = self.relations.map(RelationRegistry::frozen_generation).transpose()?;
        if authority.relations != relations {
            return Err(NormalizeError::InvalidExactPlan { reason: "stale relation context" });
        }
        Ok(())
    }

    fn materialized_exact_state(
        &mut self,
        expression: ExprId,
        normal_form: Arc<PolynomialNF>,
    ) -> Result<NodeExactState, NormalizeError> {
        if self.watchdog.is_some() {
            let consumer =
                self.diagnostic_current_expression.filter(|current| *current != expression);
            let mut origin = self
                .diagnostic_materialization_origins
                .get(&expression)
                .copied()
                .unwrap_or_else(|| DiagnosticMaterializedLeafOrigin {
                    producer: Some(expression),
                    producer_operator: self
                        .expressions
                        .node(expression)
                        .map_or("invalid", |node| normalization_operator_category(&node.operator)),
                    consumer,
                    consumer_operator: consumer.map_or("root", |id| {
                        self.expressions.node(id).map_or("invalid", |node| {
                            normalization_operator_category(&node.operator)
                        })
                    }),
                    consumer_category: consumer
                        .map_or("root", |id| self.diagnostic_consumer_category(id)),
                    remaining_uses: u64::try_from(
                        *self.remaining_uses.get(&expression).unwrap_or(&0),
                    )
                    .unwrap_or(u64::MAX),
                    scalar_classification: consumer
                        .map_or("not_multiply", |id| self.diagnostic_scalar_classification(id)),
                    retained_term_count: u64::try_from(normal_form.exact_terms.len())
                        .unwrap_or(u64::MAX),
                    ..DiagnosticMaterializedLeafOrigin::default()
                });
            origin.retained_term_count =
                u64::try_from(normal_form.exact_terms.len()).unwrap_or(u64::MAX);
            let key = Arc::as_ptr(&normal_form) as usize;
            let same_allocation = self
                .diagnostic_materialized_leaf_origins
                .get(&key)
                .and_then(|attachment| attachment.normal_form.upgrade())
                .is_some_and(|attached| Arc::ptr_eq(&attached, &normal_form));
            if !same_allocation {
                self.diagnostic_materialized_leaf_origins.insert(
                    key,
                    DiagnosticMaterializedLeafAttachment {
                        normal_form: Arc::downgrade(&normal_form),
                        origin,
                    },
                );
            }
        }
        Ok(NodeExactState::Materialized {
            authority: self.exact_plan_authority(expression)?,
            normal_form,
        })
    }

    fn diagnostic_materialized_leaf_origin(
        &self,
        normal_form: &Arc<PolynomialNF>,
    ) -> Option<DiagnosticMaterializedLeafOrigin> {
        self.diagnostic_materialized_leaf_origins
            .get(&(Arc::as_ptr(normal_form) as usize))
            .and_then(|attachment| {
                attachment
                    .normal_form
                    .upgrade()
                    .filter(|attached| Arc::ptr_eq(attached, normal_form))
                    .map(|_| attachment.origin)
            })
    }

    fn diagnostic_scalar_classification(&self, expression: ExprId) -> &'static str {
        let Ok(node) = self.expressions.node(expression) else { return "" };
        if !matches!(node.operator, ValueOperator::Matrix(MatrixOperation::Multiply)) ||
            node.inputs.len() != 2
        {
            return "not_multiply";
        }
        let scalar = |input: ExprId| {
            matches!(
                self.expressions.value_type(input),
                Ok(ResolvedValueType::Matrix(matrix)) if matrix.rows == 1 && matrix.columns == 1
            )
        };
        match (scalar(node.inputs[0]), scalar(node.inputs[1])) {
            (false, false) => "ordinary",
            (true, false) => "scalar_left",
            (false, true) => "scalar_right",
            (true, true) => "scalar_both",
        }
    }

    fn diagnostic_consumer_category(&self, expression: ExprId) -> &'static str {
        let Ok(node) = self.expressions.node(expression) else { return "invalid" };
        match node.operator {
            ValueOperator::Matrix(MatrixOperation::Add | MatrixOperation::Subtract) => "add_sub",
            ValueOperator::Matrix(MatrixOperation::Multiply) => "multiply",
            ValueOperator::Matrix(_) | ValueOperator::Transform(_) => "structural",
            _ => "root_other",
        }
    }

    fn record_materialization_origin(
        &mut self,
        producer: ExprId,
        reason: DiagnosticMaterializationReason,
        consumer: Option<ExprId>,
        retained_term_count: u64,
    ) {
        if self.watchdog.is_none() {
            return;
        }
        let key = consumer.unwrap_or(producer);
        let remaining_uses =
            u64::try_from(*self.remaining_uses.get(&producer).unwrap_or(&0)).unwrap_or(u64::MAX);
        let origin = DiagnosticMaterializedLeafOrigin {
            producer: Some(producer),
            producer_operator: self
                .expressions
                .node(producer)
                .map_or("invalid", |node| normalization_operator_category(&node.operator)),
            reason,
            consumer,
            consumer_operator: consumer.map_or("root", |id| {
                self.expressions
                    .node(id)
                    .map_or("invalid", |node| normalization_operator_category(&node.operator))
            }),
            consumer_category: consumer.map_or("root", |id| self.diagnostic_consumer_category(id)),
            remaining_uses,
            scalar_classification: consumer
                .map_or("not_multiply", |id| self.diagnostic_scalar_classification(id)),
            forced_input_count: 1,
            forced_terms_sum: retained_term_count,
            forced_terms_max: retained_term_count,
            retained_term_count,
        };
        self.diagnostic_materialization_origins
            .entry(key)
            .and_modify(|current| {
                current.reason = current.reason.max(reason);
                if current.producer != Some(producer) {
                    current.producer = None;
                    current.producer_operator = "multiple";
                }
                current.forced_input_count = current.forced_input_count.saturating_add(1);
                current.forced_terms_sum =
                    current.forced_terms_sum.saturating_add(retained_term_count);
                current.forced_terms_max = current.forced_terms_max.max(retained_term_count);
                current.remaining_uses = current.remaining_uses.max(remaining_uses);
                current.retained_term_count = current.retained_term_count.max(retained_term_count);
            })
            .or_insert(origin);
    }

    fn cached_exact_state(
        &mut self,
        expression: ExprId,
        value: &AnalyzedValue,
    ) -> Result<Option<NodeExactState>, NormalizeError> {
        let additive = self.exact_plans.get(&expression).cloned();
        let product = self.product_plans.get(&expression).cloned();
        let gadget_product = self.gadget_product_plans.get(&expression).cloned();
        if usize::from(additive.is_some()) +
            usize::from(product.is_some()) +
            usize::from(gadget_product.is_some()) >
            1
        {
            return Err(NormalizeError::InvalidExactPlan {
                reason: "cache entry has two deferred exact representations",
            });
        }
        if let Some(plan) = additive {
            if value.exact_nf.is_some() {
                return Err(NormalizeError::InvalidExactPlan {
                    reason: "cache entry has two exact representations",
                });
            }
            return Ok(Some(NodeExactState::Additive(plan)));
        }
        if let Some(plan) = product {
            if value.exact_nf.is_some() {
                return Err(NormalizeError::InvalidExactPlan {
                    reason: "cache entry has two exact representations",
                });
            }
            return Ok(Some(NodeExactState::Product(plan)));
        }
        if let Some(plan) = gadget_product {
            if value.exact_nf.is_some() {
                return Err(NormalizeError::InvalidExactPlan {
                    reason: "cache entry has two exact representations",
                });
            }
            return Ok(Some(NodeExactState::GadgetProduct(plan)));
        }
        value
            .exact_nf
            .as_ref()
            .cloned()
            .map(|normal_form| self.materialized_exact_state(expression, normal_form))
            .transpose()
    }

    fn node_exact_authority(state: &NodeExactState) -> &ExactPlanAuthority {
        match state {
            NodeExactState::Materialized { authority, .. } => authority,
            NodeExactState::Additive(plan) => &plan.authority,
            NodeExactState::Product(plan) => &plan.authority,
            NodeExactState::GadgetProduct(plan) => &plan.authority,
        }
    }

    fn node_exact_plan_id(state: &NodeExactState) -> Option<u64> {
        match state {
            NodeExactState::Materialized { .. } => None,
            NodeExactState::Additive(plan) => Some(plan.id),
            NodeExactState::Product(plan) => Some(plan.id),
            NodeExactState::GadgetProduct(plan) => Some(plan.id),
        }
    }

    fn validate_product_plan(&self, plan: &ProductExactPlan) -> Result<(), NormalizeError> {
        self.validate_exact_plan_authority(&plan.authority)?;
        let node = self.expressions.node(plan.expression)?;
        if !matches!(node.operator, ValueOperator::Matrix(MatrixOperation::Multiply)) ||
            node.inputs.as_ref() != [plan.left_expression, plan.right_expression]
        {
            return Err(NormalizeError::InvalidExactPlan { reason: "stale product expression" });
        }
        let matrix_type = |expression| -> Result<&ResolvedMatrixType, NormalizeError> {
            match self.expressions.value_type(expression)? {
                ResolvedValueType::Matrix(matrix) => Ok(matrix),
                _ => Err(NormalizeError::InvalidExactPlan { reason: "non-matrix product operand" }),
            }
        };
        if matrix_type(plan.expression)? != &plan.authority.matrix_type ||
            matrix_type(plan.left_expression)? != &plan.left_type ||
            matrix_type(plan.right_expression)? != &plan.right_type
        {
            return Err(NormalizeError::InvalidExactPlan { reason: "product type drift" });
        }
        if Self::node_exact_authority(&plan.left).matrix_type != plan.left_type ||
            Self::node_exact_authority(&plan.right).matrix_type != plan.right_type
        {
            return Err(NormalizeError::InvalidExactPlan { reason: "product operand type drift" });
        }
        if matches!(plan.mode, ProductMode::TypedGadgetCandidate) &&
            !self.gadget_recompositions.is_some_and(GadgetRecompositionRegistry::is_frozen)
        {
            return Err(NormalizeError::InvalidExactPlan {
                reason: "typed gadget product registry is not frozen",
            });
        }
        for child in [&plan.left, &plan.right] {
            self.validate_exact_plan_authority(Self::node_exact_authority(child))?;
            if Self::node_exact_plan_id(child).is_some_and(|id| id >= plan.id) {
                return Err(NormalizeError::InvalidExactPlan { reason: "noncausal product edge" });
            }
        }
        if let ProductMode::ScalarAction(scalar) = &plan.mode {
            let (scalar_expression, matrix_expression, scalar_type, matrix_type, scalar_state) =
                if scalar.scalar_on_left {
                    (
                        plan.left_expression,
                        plan.right_expression,
                        &plan.left_type,
                        &plan.right_type,
                        &plan.left,
                    )
                } else {
                    (
                        plan.right_expression,
                        plan.left_expression,
                        &plan.right_type,
                        &plan.left_type,
                        &plan.right,
                    )
                };
            if scalar.scalar_expression != scalar_expression ||
                scalar.matrix_expression != matrix_expression ||
                scalar.scalar_type != *scalar_type ||
                scalar.matrix_type != *matrix_type ||
                scalar.scalar_type.rows != 1 ||
                scalar.scalar_type.columns != 1 ||
                scalar.matrix_type.rows == 1 && scalar.matrix_type.columns == 1
            {
                return Err(NormalizeError::InvalidExactPlan {
                    reason: "scalar action authority drift",
                });
            }
            let NodeExactState::Materialized { normal_form, .. } = scalar_state else {
                return Err(NormalizeError::InvalidExactPlan { reason: "deferred scalar operand" });
            };
            if !Arc::ptr_eq(normal_form, &scalar.centralized_scalar) {
                return Err(NormalizeError::InvalidExactPlan {
                    reason: "scalar action normal form drift",
                });
            }
            for monomial in scalar.centralized_scalar.exact_terms.keys() {
                if !self.monomials.descriptor(*monomial)?.ordered_factors.is_empty() {
                    return Err(NormalizeError::InvalidExactPlan {
                        reason: "scalar action was not centralized",
                    });
                }
            }
        }
        Ok(())
    }

    fn validate_gadget_product_plan(
        &self,
        plan: &GadgetProductExactPlan,
    ) -> Result<(), NormalizeError> {
        self.validate_exact_plan_authority(&plan.authority)?;
        let node = self.expressions.node(plan.expression)?;
        if !matches!(node.operator, ValueOperator::Matrix(MatrixOperation::Multiply)) ||
            node.inputs.as_ref() != [plan.left_expression, plan.right_expression]
        {
            return Err(NormalizeError::InvalidExactPlan {
                reason: "stale gadget product expression",
            });
        }
        let matrix_type = |expression| -> Result<&ResolvedMatrixType, NormalizeError> {
            match self.expressions.value_type(expression)? {
                ResolvedValueType::Matrix(matrix) => Ok(matrix),
                _ => Err(NormalizeError::InvalidExactPlan {
                    reason: "non-matrix gadget product operand",
                }),
            }
        };
        if matrix_type(plan.expression)? != &plan.authority.matrix_type ||
            matrix_type(plan.left_expression)? != &plan.left_type ||
            matrix_type(plan.right_expression)? != &plan.right_type
        {
            return Err(NormalizeError::InvalidExactPlan { reason: "gadget product type drift" });
        }
        // Validate every pending operand ID before any product in an additive materialization can
        // mutate the destination or append arena descriptors. This also makes a collected or
        // foreign operand fail closed at the plan boundary rather than midway through execution.
        for monomial in plan.left.exact_terms.keys().chain(plan.right.exact_terms.keys()) {
            self.monomials.descriptor(*monomial)?;
        }
        Ok(())
    }

    fn new_additive_plan(
        &mut self,
        expression: ExprId,
        left: NodeExactState,
        right: NodeExactState,
        subtract_right: bool,
    ) -> Result<Arc<AdditiveExactPlan>, NormalizeError> {
        let authority = self.exact_plan_authority(expression)?;
        for child in [&left, &right] {
            let child_authority = Self::node_exact_authority(child);
            self.validate_exact_plan_authority(child_authority)?;
            if child_authority.matrix_type != authority.matrix_type {
                return Err(NormalizeError::InvalidExactPlan { reason: "additive type mismatch" });
            }
        }
        self.next_exact_plan_id =
            self.next_exact_plan_id.checked_add(1).ok_or(NormalizeError::ArithmeticOverflow)?;
        if [&left, &right]
            .into_iter()
            .filter_map(Self::node_exact_plan_id)
            .any(|id| id >= self.next_exact_plan_id)
        {
            return Err(NormalizeError::InvalidExactPlan { reason: "noncausal additive edge" });
        }
        Ok(Arc::new(AdditiveExactPlan {
            id: self.next_exact_plan_id,
            authority,
            left,
            right,
            subtract_right,
        }))
    }

    fn new_product_plan(
        &mut self,
        scope_proof: &ScopeProof,
        expression: ExprId,
        node: &ExprNode,
        mut left: NodeExactState,
        mut right: NodeExactState,
    ) -> Result<Option<Arc<ProductExactPlan>>, NormalizeError> {
        let scalar_action_requested = self.deferred_scalar_actions.contains(&expression);
        let typed_gadget_requested = self.deferred_gadget_products.contains(&expression) &&
            self.gadget_recompositions.is_some_and(GadgetRecompositionRegistry::is_frozen);
        let ordinary_requested = self.deferred_products.contains(&expression);
        if !scalar_action_requested && !typed_gadget_requested && !ordinary_requested {
            return Ok(None);
        }
        if self.relation_rewriting_enabled ||
            !matches!(node.operator, ValueOperator::Matrix(MatrixOperation::Multiply)) ||
            node.inputs.len() != 2
        {
            return Ok(None);
        }
        let matrix_type = |expression| -> Result<ResolvedMatrixType, NormalizeError> {
            match self.expressions.value_type(expression)? {
                ResolvedValueType::Matrix(matrix) => Ok(matrix.clone()),
                _ => Err(NormalizeError::InvalidExactPlan { reason: "non-matrix product operand" }),
            }
        };
        let left_type = matrix_type(node.inputs[0])?;
        let right_type = matrix_type(node.inputs[1])?;
        let authority = self.exact_plan_authority(expression)?;
        let left_scalar = left_type.rows == 1 && left_type.columns == 1;
        let right_scalar = right_type.rows == 1 && right_type.columns == 1;
        for (child, expected) in [(&left, &left_type), (&right, &right_type)] {
            self.validate_exact_plan_authority(Self::node_exact_authority(child))?;
            if &Self::node_exact_authority(child).matrix_type != expected {
                return Err(NormalizeError::InvalidExactPlan {
                    reason: "product operand type mismatch",
                });
            }
        }
        let mode = if scalar_action_requested {
            if left_scalar == right_scalar {
                return Ok(None);
            }
            let (scalar_expression, matrix_expression, scalar_type, matrix_type, scalar_state) =
                if left_scalar {
                    (node.inputs[0], node.inputs[1], &left_type, &right_type, &left)
                } else {
                    (node.inputs[1], node.inputs[0], &right_type, &left_type, &right)
                };
            if scalar_type.modulus != matrix_type.modulus ||
                scalar_type.ring_dimension != matrix_type.ring_dimension ||
                authority.matrix_type != *matrix_type
            {
                return Ok(None);
            }
            let NodeExactState::Materialized { normal_form, .. } = scalar_state else {
                return Ok(None);
            };
            if !self.scalar_nf_ordered_factors_match_type(normal_form, scalar_type)? {
                return Ok(None);
            }
            let (centralized_scalar, terms, factors_max) =
                self.centralize_scalar_nf(scope_proof, normal_form)?;
            let centralized_scalar = Arc::new(centralized_scalar);
            let centralized_state =
                self.materialized_exact_state(scalar_expression, Arc::clone(&centralized_scalar))?;
            if left_scalar {
                left = centralized_state;
            } else {
                right = centralized_state;
            }
            self.product_plan_counters.scalar_action_reclassified_terms = self
                .product_plan_counters
                .scalar_action_reclassified_terms
                .saturating_add(u64::try_from(terms).unwrap_or(u64::MAX));
            self.product_plan_counters.scalar_action_reclassified_factors_max = self
                .product_plan_counters
                .scalar_action_reclassified_factors_max
                .max(u64::try_from(factors_max).unwrap_or(u64::MAX));
            self.trace.record_scalar_call();
            if left_scalar {
                self.trace.record_scalar_left();
            } else {
                self.trace.record_scalar_right();
            }
            ProductMode::ScalarAction(ScalarActionExactPlan {
                scalar_on_left: left_scalar,
                scalar_expression,
                matrix_expression,
                scalar_type: scalar_type.clone(),
                matrix_type: matrix_type.clone(),
                centralized_scalar,
            })
        } else {
            if left_scalar || right_scalar {
                return Ok(None);
            }
            if typed_gadget_requested {
                ProductMode::TypedGadgetCandidate
            } else {
                ProductMode::Ordinary
            }
        };
        for (child, expected) in [(&left, &left_type), (&right, &right_type)] {
            self.validate_exact_plan_authority(Self::node_exact_authority(child))?;
            if &Self::node_exact_authority(child).matrix_type != expected {
                return Err(NormalizeError::InvalidExactPlan {
                    reason: "product operand type mismatch",
                });
            }
        }
        self.next_exact_plan_id =
            self.next_exact_plan_id.checked_add(1).ok_or(NormalizeError::ArithmeticOverflow)?;
        let typed_candidate = matches!(mode, ProductMode::TypedGadgetCandidate);
        let scalar_action = matches!(mode, ProductMode::ScalarAction(_));
        let plan = Arc::new(ProductExactPlan {
            id: self.next_exact_plan_id,
            authority,
            expression,
            left_expression: node.inputs[0],
            right_expression: node.inputs[1],
            left_type,
            right_type,
            mode,
            left,
            right,
        });
        self.product_plan_counters.plans_created =
            self.product_plan_counters.plans_created.saturating_add(1);
        if typed_candidate {
            self.product_plan_counters.typed_candidate_plans =
                self.product_plan_counters.typed_candidate_plans.saturating_add(1);
        }
        if scalar_action {
            self.product_plan_counters.scalar_action_plans_created =
                self.product_plan_counters.scalar_action_plans_created.saturating_add(1);
        }
        Ok(Some(plan))
    }

    fn product_plan_uses_direct_gadget_rewrite(
        &self,
        plan: &ProductExactPlan,
        left: &PolynomialNF,
        right: &PolynomialNF,
    ) -> Result<bool, NormalizeError> {
        self.validate_product_plan(plan)?;
        if !matches!(plan.mode, ProductMode::TypedGadgetCandidate) {
            return Ok(false);
        }
        let Some(registry) = self.gadget_recompositions else { return Ok(false) };
        if !registry.is_frozen() {
            return Err(NormalizeError::InvalidExactPlan {
                reason: "typed gadget product registry is not frozen",
            });
        }
        let mut left_endpoints = BTreeSet::new();
        for monomial in left.exact_terms.keys() {
            let descriptor = self.monomials.descriptor(*monomial)?;
            if let Some(endpoint) = descriptor.ordered_factors.last() {
                left_endpoints.insert(*endpoint);
            }
        }
        let mut right_endpoints = BTreeSet::new();
        for monomial in right.exact_terms.keys() {
            let descriptor = self.monomials.descriptor(*monomial)?;
            if let Some(endpoint) = descriptor.ordered_factors.first() {
                right_endpoints.insert(*endpoint);
            }
        }
        for gadget in left_endpoints {
            for decomposition in &right_endpoints {
                if self.authorized_gadget_pair_input(gadget, *decomposition)?.is_some() {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    fn new_gadget_product_plan(
        &mut self,
        expression: ExprId,
        node: &ExprNode,
        left: Arc<PolynomialNF>,
        right: Arc<PolynomialNF>,
    ) -> Result<Option<Arc<GadgetProductExactPlan>>, NormalizeError> {
        if self.relation_rewriting_enabled ||
            !self.deferred_gadget_products.contains(&expression) ||
            !matches!(node.operator, ValueOperator::Matrix(MatrixOperation::Multiply)) ||
            node.inputs.len() != 2
        {
            return Ok(None);
        }
        let matrix_type = |expression| -> Result<ResolvedMatrixType, NormalizeError> {
            match self.expressions.value_type(expression)? {
                ResolvedValueType::Matrix(matrix) => Ok(matrix.clone()),
                _ => Err(NormalizeError::InvalidExactPlan {
                    reason: "non-matrix gadget product operand",
                }),
            }
        };
        let left_type = matrix_type(node.inputs[0])?;
        let right_type = matrix_type(node.inputs[1])?;
        if (left_type.rows == 1 && left_type.columns == 1) ||
            (right_type.rows == 1 && right_type.columns == 1)
        {
            return Ok(None);
        }
        let mut left_endpoints = BTreeSet::new();
        for monomial in left.exact_terms.keys() {
            let descriptor = self.monomials.descriptor(*monomial)?;
            if let Some(endpoint) = descriptor.ordered_factors.last() {
                left_endpoints.insert(*endpoint);
            }
        }
        let mut right_endpoints = BTreeSet::new();
        for monomial in right.exact_terms.keys() {
            let descriptor = self.monomials.descriptor(*monomial)?;
            if let Some(endpoint) = descriptor.ordered_factors.first() {
                right_endpoints.insert(*endpoint);
            }
        }
        let mut authorized = false;
        'outer: for gadget in left_endpoints {
            for decomposition in &right_endpoints {
                if self.authorized_gadget_pair_input(gadget, *decomposition)?.is_some() {
                    authorized = true;
                    break 'outer;
                }
            }
        }
        if !authorized {
            return Ok(None);
        }
        self.next_exact_plan_id =
            self.next_exact_plan_id.checked_add(1).ok_or(NormalizeError::ArithmeticOverflow)?;
        let plan = Arc::new(GadgetProductExactPlan {
            id: self.next_exact_plan_id,
            authority: self.exact_plan_authority(expression)?,
            expression,
            left_expression: node.inputs[0],
            right_expression: node.inputs[1],
            left_type,
            right_type,
            left,
            right,
        });
        self.gadget_product_counters.plans_created =
            self.gadget_product_counters.plans_created.saturating_add(1);
        Ok(Some(plan))
    }

    /// Materialize a persistent exact-plan DAG with an explicit causal postorder stack. Product
    /// chains never recurse on the Rust call stack. Additive parents still stream products directly
    /// into their one destination map unless a shared product needs a standalone output.
    fn materialize_exact_state(
        &mut self,
        state: &NodeExactState,
    ) -> Result<Arc<PolynomialNF>, NormalizeError> {
        self.validate_exact_state_dag(state)?;
        let product_operand_additives = Self::product_operand_additives(state);
        let mut product_uses = self.product_plan_use_counts(state, &product_operand_additives)?;
        let mut outputs = BTreeMap::<(u8, u64), Arc<PolynomialNF>>::new();
        let mut scheduled = BTreeSet::<(u8, u64)>::new();
        let mut consumed_additives = BTreeSet::<u64>::new();
        let mut frames = vec![ExactMaterializationFrame::Enter(state.clone())];

        while let Some(frame) = frames.pop() {
            match frame {
                ExactMaterializationFrame::Enter(state) => {
                    let Some(key) = Self::node_exact_output_key(&state) else { continue };
                    if outputs.contains_key(&key) || !scheduled.insert(key) {
                        continue;
                    }
                    match state {
                        NodeExactState::Materialized { .. } => unreachable!(),
                        NodeExactState::GadgetProduct(plan) => {
                            frames.push(ExactMaterializationFrame::FinishGadgetProduct(plan));
                        }
                        NodeExactState::Product(plan) => {
                            frames
                                .push(ExactMaterializationFrame::FinishProduct(Arc::clone(&plan)));
                            let mut dependencies = vec![plan.left.clone(), plan.right.clone()];
                            dependencies.sort_by_key(|state| Self::node_exact_plan_id(state));
                            for dependency in dependencies.into_iter().rev() {
                                frames.push(ExactMaterializationFrame::Enter(dependency));
                            }
                        }
                        NodeExactState::Additive(plan) => {
                            let mut flattened = self.flatten_additive_plan(
                                &plan,
                                &outputs,
                                &product_operand_additives,
                            )?;
                            let mut dependencies = Vec::new();
                            for (additive, weight) in flattened.additive_outputs.values() {
                                if !weight.is_zero() && !outputs.contains_key(&(0, additive.id)) {
                                    dependencies
                                        .push(NodeExactState::Additive(Arc::clone(additive)));
                                }
                            }
                            for (product, weight, occurrences, standalone) in
                                flattened.products.values_mut()
                            {
                                if weight.is_zero() {
                                    continue;
                                }
                                let remaining = *product_uses.get(&product.id).ok_or(
                                    NormalizeError::InvalidExactPlan {
                                        reason: "missing product use count",
                                    },
                                )?;
                                *standalone = outputs.contains_key(&(1, product.id)) ||
                                    remaining > *occurrences;
                                if *standalone {
                                    dependencies.push(NodeExactState::Product(Arc::clone(product)));
                                } else {
                                    dependencies.push(product.left.clone());
                                    dependencies.push(product.right.clone());
                                }
                            }
                            frames.push(ExactMaterializationFrame::FinishAdditive(flattened));
                            dependencies.sort_by_key(|state| Self::node_exact_plan_id(state));
                            for dependency in dependencies.into_iter().rev() {
                                frames.push(ExactMaterializationFrame::Enter(dependency));
                            }
                        }
                    }
                }
                ExactMaterializationFrame::FinishGadgetProduct(plan) => {
                    let mut terms = BTreeMap::new();
                    self.execute_product_into(
                        plan.left.as_ref(),
                        plan.right.as_ref(),
                        &BigInt::from(1_u8),
                        &mut terms,
                        true,
                        false,
                    )?;
                    self.gadget_product_counters.standalone_materializations =
                        self.gadget_product_counters.standalone_materializations.saturating_add(1);
                    outputs.insert(
                        (2, plan.id),
                        Arc::new(PolynomialNF {
                            exact_terms: terms,
                            bounded_summary: BoundedSummary::missing(),
                        }),
                    );
                }
                ExactMaterializationFrame::FinishProduct(plan) => {
                    let left = Self::exact_state_output(&plan.left, &outputs)?;
                    let right = Self::exact_state_output(&plan.right, &outputs)?;
                    let mut terms = BTreeMap::new();
                    let direct = self.product_plan_uses_direct_gadget_rewrite(
                        &plan,
                        left.as_ref(),
                        right.as_ref(),
                    )?;
                    self.execute_product_into(
                        left.as_ref(),
                        right.as_ref(),
                        &BigInt::from(1_u8),
                        &mut terms,
                        direct,
                        matches!(plan.mode, ProductMode::TypedGadgetCandidate),
                    )?;
                    let planned = u64::try_from(left.exact_terms.len())
                        .unwrap_or(u64::MAX)
                        .saturating_mul(u64::try_from(right.exact_terms.len()).unwrap_or(u64::MAX));
                    self.product_plan_counters.standalone_materializations =
                        self.product_plan_counters.standalone_materializations.saturating_add(1);
                    if matches!(plan.mode, ProductMode::ScalarAction(_)) {
                        self.product_plan_counters.scalar_action_standalone_materializations = self
                            .product_plan_counters
                            .scalar_action_standalone_materializations
                            .saturating_add(1);
                    }
                    if matches!(plan.mode, ProductMode::TypedGadgetCandidate) {
                        self.product_plan_counters.typed_standalone_materializations = self
                            .product_plan_counters
                            .typed_standalone_materializations
                            .saturating_add(1);
                    }
                    self.product_plan_counters.planned_pairs =
                        self.product_plan_counters.planned_pairs.saturating_add(planned);
                    Self::release_product_output(&plan.left, &mut outputs, &mut product_uses)?;
                    Self::release_product_output(&plan.right, &mut outputs, &mut product_uses)?;
                    outputs.insert(
                        (1, plan.id),
                        Arc::new(PolynomialNF {
                            exact_terms: terms,
                            bounded_summary: BoundedSummary::missing(),
                        }),
                    );
                }
                ExactMaterializationFrame::FinishAdditive(flattened) => {
                    let normal_form = self.finish_additive_plan(
                        flattened,
                        &mut outputs,
                        &mut product_uses,
                        &mut consumed_additives,
                    )?;
                    outputs.insert((0, normal_form.0), normal_form.1);
                }
            }
        }

        Self::exact_state_output(state, &outputs)
    }

    fn node_exact_output_key(state: &NodeExactState) -> Option<(u8, u64)> {
        match state {
            NodeExactState::Materialized { .. } => None,
            NodeExactState::Additive(plan) => Some((0, plan.id)),
            NodeExactState::Product(plan) => Some((1, plan.id)),
            NodeExactState::GadgetProduct(plan) => Some((2, plan.id)),
        }
    }

    fn exact_state_output(
        state: &NodeExactState,
        outputs: &BTreeMap<(u8, u64), Arc<PolynomialNF>>,
    ) -> Result<Arc<PolynomialNF>, NormalizeError> {
        if let NodeExactState::Materialized { normal_form, .. } = state {
            return Ok(Arc::clone(normal_form));
        }
        let key = Self::node_exact_output_key(state)
            .ok_or(NormalizeError::InvalidExactPlan { reason: "missing exact state output key" })?;
        outputs
            .get(&key)
            .cloned()
            .ok_or(NormalizeError::InvalidExactPlan { reason: "missing exact plan output" })
    }

    fn product_plan_use_counts(
        &self,
        root: &NodeExactState,
        product_operand_additives: &BTreeSet<u64>,
    ) -> Result<BTreeMap<u64, usize>, NormalizeError> {
        // Count product-to-product definition edges once, then mirror the additive flattening
        // work once per materialized additive output. A shared additive definition can feed more
        // than one product destination, so counting only its definition edges would release a
        // shared product output too early. This prepass is bounded by the same plan memberships
        // that the materializer subsequently visits; it does not expand polynomial terms.
        let mut result = BTreeMap::<u64, usize>::new();
        let mut pending = vec![root.clone()];
        let mut seen = BTreeSet::new();
        let mut additives = BTreeMap::<u64, Arc<AdditiveExactPlan>>::new();
        while let Some(state) = pending.pop() {
            let (kind, id) = match &state {
                NodeExactState::Materialized { .. } | NodeExactState::GadgetProduct(_) => continue,
                NodeExactState::Additive(plan) => (0, plan.id),
                NodeExactState::Product(plan) => (1, plan.id),
            };
            if !seen.insert((kind, id)) {
                continue;
            }
            let is_product = matches!(state, NodeExactState::Product(_));
            let children = match &state {
                NodeExactState::Additive(plan) => {
                    additives.insert(plan.id, Arc::clone(plan));
                    [&plan.left, &plan.right]
                }
                NodeExactState::Product(plan) => [&plan.left, &plan.right],
                NodeExactState::Materialized { .. } | NodeExactState::GadgetProduct(_) => continue,
            };
            for child in children {
                if is_product && let NodeExactState::Product(product) = child {
                    let count = result.entry(product.id).or_default();
                    *count = count.checked_add(1).ok_or(NormalizeError::ArithmeticOverflow)?;
                }
                pending.push(child.clone());
            }
        }
        let mut additive_outputs = product_operand_additives.clone();
        if let NodeExactState::Additive(plan) = root {
            additive_outputs.insert(plan.id);
        }
        let no_outputs = BTreeMap::new();
        for additive in additive_outputs {
            let plan = additives.get(&additive).ok_or(NormalizeError::InvalidExactPlan {
                reason: "missing additive output definition",
            })?;
            let flattened =
                self.flatten_additive_plan(plan, &no_outputs, product_operand_additives)?;
            for (product, _, occurrences, _) in flattened.products.into_values() {
                let count = result.entry(product.id).or_default();
                *count =
                    count.checked_add(occurrences).ok_or(NormalizeError::ArithmeticOverflow)?;
            }
        }
        Ok(result)
    }

    fn product_operand_additives(root: &NodeExactState) -> BTreeSet<u64> {
        let mut result = BTreeSet::new();
        let mut pending = vec![root.clone()];
        let mut seen = BTreeSet::new();
        while let Some(state) = pending.pop() {
            let Some((kind, id)) = Self::node_exact_output_key(&state) else { continue };
            if !seen.insert((kind, id)) {
                continue;
            }
            match state {
                NodeExactState::Product(plan) => {
                    for child in [&plan.left, &plan.right] {
                        if let NodeExactState::Additive(additive) = child {
                            result.insert(additive.id);
                        }
                        pending.push(child.clone());
                    }
                }
                NodeExactState::Additive(plan) => {
                    pending.push(plan.left.clone());
                    pending.push(plan.right.clone());
                }
                NodeExactState::Materialized { .. } | NodeExactState::GadgetProduct(_) => {}
            }
        }
        result
    }

    fn release_product_output(
        state: &NodeExactState,
        outputs: &mut BTreeMap<(u8, u64), Arc<PolynomialNF>>,
        product_uses: &mut BTreeMap<u64, usize>,
    ) -> Result<(), NormalizeError> {
        let NodeExactState::Product(plan) = state else { return Ok(()) };
        let remaining = product_uses
            .get_mut(&plan.id)
            .ok_or(NormalizeError::InvalidExactPlan { reason: "missing product use count" })?;
        *remaining = remaining
            .checked_sub(1)
            .ok_or(NormalizeError::InvalidExactPlan { reason: "product use count underflow" })?;
        if *remaining == 0 {
            outputs.remove(&(1, plan.id));
        }
        Ok(())
    }

    fn validate_exact_state_dag(&self, root: &NodeExactState) -> Result<(), NormalizeError> {
        let mut pending = vec![root.clone()];
        let mut plans = BTreeMap::<u64, (u8, usize)>::new();
        let register = |plans: &mut BTreeMap<u64, (u8, usize)>,
                        id: u64,
                        kind: u8,
                        pointer: usize|
         -> Result<bool, NormalizeError> {
            if let Some(existing) = plans.get(&id) {
                if *existing != (kind, pointer) {
                    return Err(NormalizeError::InvalidExactPlan {
                        reason: "conflicting exact plan id",
                    });
                }
                return Ok(false);
            }
            plans.insert(id, (kind, pointer));
            Ok(true)
        };
        while let Some(state) = pending.pop() {
            match state {
                NodeExactState::Materialized { authority, normal_form } => {
                    self.validate_exact_plan_authority(&authority)?;
                    for monomial in normal_form.exact_terms.keys() {
                        self.monomials.descriptor(*monomial)?;
                    }
                }
                NodeExactState::Additive(plan) => {
                    if !register(&mut plans, plan.id, 0, Arc::as_ptr(&plan) as usize)? {
                        continue;
                    }
                    self.validate_exact_plan_authority(&plan.authority)?;
                    for child in [&plan.left, &plan.right] {
                        self.validate_exact_plan_authority(Self::node_exact_authority(child))?;
                        if Self::node_exact_authority(child).matrix_type !=
                            plan.authority.matrix_type
                        {
                            return Err(NormalizeError::InvalidExactPlan {
                                reason: "additive type drift",
                            });
                        }
                        if Self::node_exact_plan_id(child).is_some_and(|id| id >= plan.id) {
                            return Err(NormalizeError::InvalidExactPlan {
                                reason: "noncausal additive edge",
                            });
                        }
                        pending.push(child.clone());
                    }
                }
                NodeExactState::Product(plan) => {
                    if !register(&mut plans, plan.id, 1, Arc::as_ptr(&plan) as usize)? {
                        continue;
                    }
                    self.validate_product_plan(&plan)?;
                    pending.push(plan.left.clone());
                    pending.push(plan.right.clone());
                }
                NodeExactState::GadgetProduct(plan) => {
                    if register(&mut plans, plan.id, 2, Arc::as_ptr(&plan) as usize)? {
                        self.validate_gadget_product_plan(&plan)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn flatten_additive_plan(
        &self,
        root: &Arc<AdditiveExactPlan>,
        outputs: &BTreeMap<(u8, u64), Arc<PolynomialNF>>,
        product_operand_additives: &BTreeSet<u64>,
    ) -> Result<FlattenedAdditiveExactPlan, NormalizeError> {
        let mut pending = BTreeMap::<u64, (Arc<AdditiveExactPlan>, BigInt)>::new();
        pending.insert(root.id, (Arc::clone(root), BigInt::from(1_u8)));
        let mut additive_ids = BTreeSet::new();
        let mut leaves = HashMap::<usize, (Arc<PolynomialNF>, BigInt)>::new();
        let mut additive_outputs = BTreeMap::<u64, (Arc<AdditiveExactPlan>, BigInt)>::new();
        let mut products = BTreeMap::<u64, (Arc<ProductExactPlan>, BigInt, usize, bool)>::new();
        let mut gadget_products = BTreeMap::<u64, (Arc<GadgetProductExactPlan>, BigInt)>::new();
        while let Some((_, (plan, weight))) = pending.pop_last() {
            if !additive_ids.insert(plan.id) {
                continue;
            }
            for (child, sign) in
                [(&plan.left, 1_i8), (&plan.right, if plan.subtract_right { -1 } else { 1 })]
            {
                let signed = if sign < 0 { -weight.clone() } else { weight.clone() };
                match child {
                    NodeExactState::Additive(child) => {
                        if let Some(normal_form) = outputs.get(&(0, child.id)) {
                            let key = Arc::as_ptr(normal_form) as usize;
                            leaves
                                .entry(key)
                                .and_modify(|(_, total)| *total += &signed)
                                .or_insert_with(|| (Arc::clone(normal_form), signed));
                        } else if product_operand_additives.contains(&child.id) {
                            additive_outputs
                                .entry(child.id)
                                .and_modify(|(_, total)| *total += &signed)
                                .or_insert_with(|| (Arc::clone(child), signed));
                        } else {
                            pending
                                .entry(child.id)
                                .and_modify(|(_, total)| *total += &signed)
                                .or_insert_with(|| (Arc::clone(child), signed));
                        }
                    }
                    NodeExactState::Product(child) => match products.entry(child.id) {
                        std::collections::btree_map::Entry::Vacant(entry) => {
                            entry.insert((Arc::clone(child), signed, 1, false));
                        }
                        std::collections::btree_map::Entry::Occupied(mut entry) => {
                            let (_, total, occurrences, _) = entry.get_mut();
                            *total += &signed;
                            *occurrences = occurrences
                                .checked_add(1)
                                .ok_or(NormalizeError::ArithmeticOverflow)?;
                        }
                    },
                    NodeExactState::GadgetProduct(child) => {
                        gadget_products
                            .entry(child.id)
                            .and_modify(|(_, total)| *total += &signed)
                            .or_insert_with(|| (Arc::clone(child), signed));
                    }
                    NodeExactState::Materialized { normal_form, .. } => {
                        let key = Arc::as_ptr(normal_form) as usize;
                        leaves
                            .entry(key)
                            .and_modify(|(_, total)| *total += &signed)
                            .or_insert_with(|| (Arc::clone(normal_form), signed));
                    }
                }
            }
        }
        Ok(FlattenedAdditiveExactPlan {
            id: root.id,
            additive_ids,
            leaves,
            additive_outputs,
            products,
            gadget_products,
        })
    }

    fn finish_additive_plan(
        &mut self,
        flattened: FlattenedAdditiveExactPlan,
        outputs: &mut BTreeMap<(u8, u64), Arc<PolynomialNF>>,
        product_uses: &mut BTreeMap<u64, usize>,
        consumed_additives: &mut BTreeSet<u64>,
    ) -> Result<(u64, Arc<PolynomialNF>), NormalizeError> {
        // These additive definitions own the Product edges collected below. Mark them before a
        // canceled sibling can recursively discard a shared additive operand.
        consumed_additives.extend(flattened.additive_ids.iter().copied());
        let mut terms = BTreeMap::new();
        for (_, (additive, weight)) in flattened.additive_outputs {
            if weight.is_zero() {
                if !outputs.contains_key(&(0, additive.id)) {
                    self.discard_unexecuted_exact_states(
                        vec![NodeExactState::Additive(additive)],
                        outputs,
                        product_uses,
                        consumed_additives,
                    )?;
                }
                continue;
            }
            let normal_form =
                outputs.get(&(0, additive.id)).ok_or(NormalizeError::InvalidExactPlan {
                    reason: "missing scheduled additive output",
                })?;
            for (monomial, coefficient) in &normal_form.exact_terms {
                merge_term(&mut terms, *monomial, coefficient * &weight);
            }
        }
        for (_, (normal_form, weight)) in flattened.leaves {
            if weight.is_zero() {
                continue;
            }
            for (monomial, coefficient) in &normal_form.exact_terms {
                merge_term(&mut terms, *monomial, coefficient * &weight);
            }
        }
        for (_, (plan, weight, occurrences, standalone)) in flattened.products.into_iter().rev() {
            let mut executed = false;
            if !weight.is_zero() {
                if standalone || outputs.contains_key(&(1, plan.id)) {
                    let normal_form =
                        outputs.get(&(1, plan.id)).ok_or(NormalizeError::InvalidExactPlan {
                            reason: "missing scheduled product output",
                        })?;
                    for (monomial, coefficient) in &normal_form.exact_terms {
                        merge_term(&mut terms, *monomial, coefficient * &weight);
                    }
                } else {
                    let left = Self::exact_state_output(&plan.left, outputs)?;
                    let right = Self::exact_state_output(&plan.right, outputs)?;
                    let direct = self.product_plan_uses_direct_gadget_rewrite(
                        &plan,
                        left.as_ref(),
                        right.as_ref(),
                    )?;
                    self.execute_product_into(
                        left.as_ref(),
                        right.as_ref(),
                        &weight,
                        &mut terms,
                        direct,
                        matches!(plan.mode, ProductMode::TypedGadgetCandidate),
                    )?;
                    let planned = u64::try_from(left.exact_terms.len())
                        .unwrap_or(u64::MAX)
                        .saturating_mul(u64::try_from(right.exact_terms.len()).unwrap_or(u64::MAX));
                    self.product_plan_counters.streamed_executions =
                        self.product_plan_counters.streamed_executions.saturating_add(1);
                    if matches!(plan.mode, ProductMode::ScalarAction(_)) {
                        self.product_plan_counters.scalar_action_streamed_executions = self
                            .product_plan_counters
                            .scalar_action_streamed_executions
                            .saturating_add(1);
                    }
                    self.product_plan_counters.planned_pairs =
                        self.product_plan_counters.planned_pairs.saturating_add(planned);
                    self.product_plan_counters.max_streamed_output_terms = self
                        .product_plan_counters
                        .max_streamed_output_terms
                        .max(u64::try_from(terms.len()).unwrap_or(u64::MAX));
                    Self::release_product_output(&plan.left, outputs, product_uses)?;
                    Self::release_product_output(&plan.right, outputs, product_uses)?;
                    executed = true;
                }
            } else {
                self.product_plan_counters.zero_weight_skips =
                    self.product_plan_counters.zero_weight_skips.saturating_add(1);
                if matches!(plan.mode, ProductMode::ScalarAction(_)) {
                    self.product_plan_counters.scalar_action_zero_weight_skips = self
                        .product_plan_counters
                        .scalar_action_zero_weight_skips
                        .saturating_add(1);
                }
            }
            for _ in 0..occurrences {
                let had_output = outputs.contains_key(&(1, plan.id));
                Self::release_product_output(
                    &NodeExactState::Product(Arc::clone(&plan)),
                    outputs,
                    product_uses,
                )?;
                if !executed && !had_output && product_uses.get(&plan.id).copied() == Some(0) {
                    self.discard_unexecuted_product_dependencies(
                        &plan,
                        outputs,
                        product_uses,
                        consumed_additives,
                    )?;
                }
            }
        }
        for (_, (plan, weight)) in flattened.gadget_products {
            if weight.is_zero() {
                self.gadget_product_counters.zero_weight_skips =
                    self.gadget_product_counters.zero_weight_skips.saturating_add(1);
                continue;
            }
            self.execute_product_into(
                plan.left.as_ref(),
                plan.right.as_ref(),
                &weight,
                &mut terms,
                true,
                false,
            )?;
        }
        let output_terms = u64::try_from(terms.len()).unwrap_or(u64::MAX);
        self.exact_plan_materializations = self.exact_plan_materializations.saturating_add(1);
        self.exact_plan_materialization_output_terms_total =
            self.exact_plan_materialization_output_terms_total.saturating_add(output_terms);
        self.exact_plan_materialization_output_terms_max =
            self.exact_plan_materialization_output_terms_max.max(output_terms);
        Ok((
            flattened.id,
            Arc::new(PolynomialNF {
                exact_terms: terms,
                bounded_summary: BoundedSummary::missing(),
            }),
        ))
    }

    fn discard_unexecuted_product_dependencies(
        &mut self,
        product: &Arc<ProductExactPlan>,
        outputs: &mut BTreeMap<(u8, u64), Arc<PolynomialNF>>,
        product_uses: &mut BTreeMap<u64, usize>,
        consumed_additives: &mut BTreeSet<u64>,
    ) -> Result<(), NormalizeError> {
        self.discard_unexecuted_exact_states(
            vec![product.left.clone(), product.right.clone()],
            outputs,
            product_uses,
            consumed_additives,
        )
    }

    fn discard_unexecuted_exact_states(
        &mut self,
        mut pending: Vec<NodeExactState>,
        outputs: &mut BTreeMap<(u8, u64), Arc<PolynomialNF>>,
        product_uses: &mut BTreeMap<u64, usize>,
        consumed_additives: &mut BTreeSet<u64>,
    ) -> Result<(), NormalizeError> {
        while let Some(state) = pending.pop() {
            match state {
                NodeExactState::Materialized { .. } | NodeExactState::GadgetProduct(_) => {}
                NodeExactState::Additive(plan) => {
                    if outputs.contains_key(&(0, plan.id)) || !consumed_additives.insert(plan.id) {
                        continue;
                    }
                    pending.push(plan.left.clone());
                    pending.push(plan.right.clone());
                }
                NodeExactState::Product(plan) => {
                    let had_output = outputs.contains_key(&(1, plan.id));
                    Self::release_product_output(
                        &NodeExactState::Product(Arc::clone(&plan)),
                        outputs,
                        product_uses,
                    )?;
                    if !had_output && product_uses.get(&plan.id).copied() == Some(0) {
                        pending.push(plan.left.clone());
                        pending.push(plan.right.clone());
                    }
                }
            }
        }
        Ok(())
    }

    fn materialize_exact_state_for(
        &mut self,
        state: &NodeExactState,
        producer: ExprId,
        reason: DiagnosticMaterializationReason,
        consumer: Option<ExprId>,
    ) -> Result<Arc<PolynomialNF>, NormalizeError> {
        let normal_form = self.materialize_exact_state(state)?;
        self.record_materialization_origin(
            producer,
            reason,
            consumer,
            u64::try_from(normal_form.exact_terms.len()).unwrap_or(u64::MAX),
        );
        Ok(normal_form)
    }

    fn exact_plan_leaf_roots_and_top8(
        &self,
        validate: bool,
    ) -> Result<
        (Vec<MonomialId>, DiagnosticMaterializedLeafTop8, Vec<DiagnosticExactNf>),
        NormalizeError,
    > {
        let mut pending = self
            .exact_plans
            .values()
            .cloned()
            .map(|plan| (NodeExactState::Additive(plan), false))
            .chain(
                self.product_plans
                    .values()
                    .cloned()
                    .map(|plan| (NodeExactState::Product(plan), true)),
            )
            .chain(
                self.gadget_product_plans
                    .values()
                    .cloned()
                    .map(|plan| (NodeExactState::GadgetProduct(plan), true)),
            )
            .collect::<Vec<_>>();
        if validate {
            for (state, _) in &pending {
                self.validate_exact_state_dag(state)?;
            }
        }
        let mut seen_plans = BTreeSet::<(u8, u64, bool)>::new();
        let mut seen_leaves = HashMap::<usize, usize>::new();
        let mut roots = Vec::new();
        let mut top8 = DiagnosticMaterializedLeafTop8::default();
        let collect_origins = self.watchdog.is_some();
        let collect_four_class = collect_origins && self.four_class_census_enabled;
        let mut diagnostic_nfs = Vec::<DiagnosticExactNf>::new();
        let observe_leaf =
            |normal_form: Arc<PolynomialNF>,
             under_product: bool,
             roots: &mut Vec<MonomialId>,
             collect_roots: bool,
             seen_leaves: &mut HashMap<usize, usize>,
             diagnostic_nfs: &mut Vec<DiagnosticExactNf>| {
                let key = Arc::as_ptr(&normal_form) as usize;
                if let Some(index) = seen_leaves.get(&key).copied() {
                    if collect_four_class {
                        diagnostic_nfs[index].under_product |= under_product;
                    }
                    return false;
                }
                if collect_roots {
                    roots.extend(normal_form.exact_terms.keys().copied());
                }
                let diagnostic_index = diagnostic_nfs.len();
                seen_leaves.insert(key, diagnostic_index);
                if collect_four_class {
                    diagnostic_nfs.push(DiagnosticExactNf {
                        normal_form,
                        ordinal: u64::try_from(diagnostic_nfs.len()).unwrap_or(u64::MAX),
                        under_product,
                    });
                }
                true
            };
        while let Some((state, under_product)) = pending.pop() {
            match state {
                NodeExactState::Additive(plan) => {
                    if seen_plans.insert((0, plan.id, collect_four_class && under_product)) {
                        pending.push((plan.left.clone(), under_product));
                        pending.push((plan.right.clone(), under_product));
                    }
                }
                NodeExactState::Product(plan) => {
                    if seen_plans.insert((1, plan.id, collect_four_class && under_product)) {
                        pending.push((plan.left.clone(), true));
                        pending.push((plan.right.clone(), true));
                    }
                }
                NodeExactState::GadgetProduct(plan) => {
                    if seen_plans.insert((2, plan.id, collect_four_class && under_product)) {
                        for normal_form in [&plan.left, &plan.right] {
                            observe_leaf(
                                Arc::clone(normal_form),
                                true,
                                &mut roots,
                                true,
                                &mut seen_leaves,
                                &mut diagnostic_nfs,
                            );
                        }
                    }
                }
                NodeExactState::Materialized { normal_form, .. } => {
                    if observe_leaf(
                        Arc::clone(&normal_form),
                        under_product,
                        &mut roots,
                        true,
                        &mut seen_leaves,
                        &mut diagnostic_nfs,
                    ) {
                        if collect_origins &&
                            let Some(origin) =
                                self.diagnostic_materialized_leaf_origin(&normal_form)
                        {
                            top8.observe(
                                origin,
                                u64::try_from(normal_form.exact_terms.len()).unwrap_or(u64::MAX),
                            );
                        }
                    }
                }
            }
        }
        Ok((roots, top8, diagnostic_nfs))
    }

    fn exact_plan_leaf_roots(&self, validate: bool) -> Result<Vec<MonomialId>, NormalizeError> {
        self.exact_plan_leaf_roots_and_top8(validate).map(|(roots, _, _)| roots)
    }

    fn diagnostic_frontier_reasons(
        &self,
        monomial: MonomialId,
        under_product: bool,
        has_closed: bool,
        has_universal: bool,
    ) -> Result<u8, NormalizeError> {
        let descriptor = self.monomials.descriptor(monomial)?;
        let mut reasons = 0_u8;
        if has_closed &&
            (!descriptor.central_factors.is_empty() || !descriptor.ordered_factors.is_empty())
        {
            reasons |= FRONTIER_CLOSED_BLANKET;
        }
        if let Some(relations) = self.relations {
            for factor in &descriptor.ordered_factors {
                let node = self.expressions.node(factor.expression())?;
                if let ValueOperator::ProgramCall { program } = node.operator {
                    match relations.dispatch_for_preimage_program(program) {
                        Ok(Some(_)) => {
                            reasons |= FRONTIER_CURRENT_EXACT_PREIMAGE;
                        }
                        Err(RelationRegistryError::AmbiguousPreimageDispatch) => {
                            reasons |= FRONTIER_AMBIGUOUS_UNIVERSAL_DISPATCH;
                        }
                        Ok(None) => {}
                        Err(error) => return Err(error.into()),
                    }
                }
            }
        }
        for pair in descriptor.ordered_factors.windows(2) {
            if self.authorized_gadget_pair_input(pair[0], pair[1])?.is_some() {
                reasons |= FRONTIER_CURRENT_AUTHORIZED_GADGET;
            }
        }
        if under_product && !descriptor.ordered_factors.is_empty() {
            if has_universal {
                // A future product can place the narrow public factor outside this retained
                // word. Without that boundary, absence of a current preimage call is not proof
                // that universal rewriting is impossible.
                reasons |= FRONTIER_FUTURE_UNIVERSAL_BLANKET;
            }
            if self.diagnostic_future_typed_gadget_boundary(descriptor)? {
                reasons |= FRONTIER_FUTURE_TYPED_GADGET;
            }
        }
        Ok(reasons)
    }

    fn diagnostic_future_typed_gadget_boundary(
        &self,
        descriptor: &super::monomial::MonomialDescriptor,
    ) -> Result<bool, NormalizeError> {
        let Some(registry) = self.gadget_recompositions else { return Ok(false) };
        let layout = |expression| match self.facts.facts(expression) {
            Ok(ValueFacts::Matrix(facts)) => Some(facts.metadata.layout.clone()),
            _ => None,
        };
        for factor in [descriptor.ordered_factors.first(), descriptor.ordered_factors.last()]
            .into_iter()
            .flatten()
        {
            let node = self.expressions.node(factor.expression())?;
            let factor_type = match self.expressions.value_type(factor.expression())? {
                ResolvedValueType::Matrix(matrix) => matrix,
                _ => continue,
            };
            if let Some(super::arena::MatrixConstantKind::Gadget { base, small }) =
                node.operator.source_matrix_constant() &&
                registry.allows_gadget_half(
                    *base,
                    *small,
                    factor_type,
                    layout(factor.expression()).as_ref(),
                )?
            {
                return Ok(true);
            }
            if let ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
                base,
                small,
                digit_count,
                ..
            }) = &node.operator &&
                let Some(input) = node.inputs.first().copied() &&
                let ResolvedValueType::Matrix(input_type) = self.expressions.value_type(input)? &&
                registry.allows_decomposition_half(
                    *base,
                    *small,
                    *digit_count,
                    factor_type,
                    input_type,
                    layout(factor.expression()).as_ref(),
                    layout(input).as_ref(),
                )?
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn diagnostic_four_class_census(
        &self,
        normal_forms: &[DiagnosticExactNf],
    ) -> Result<DiagnosticFourClassCensus, NormalizeError> {
        #[derive(Clone, Copy)]
        struct Record {
            class: DiagnosticTermClass,
            reasons: u8,
            payload: u64,
        }
        let has_closed = self
            .relations
            .map(RelationRegistry::has_closed_relations)
            .transpose()?
            .unwrap_or(false);
        let has_universal = self
            .relations
            .map(RelationRegistry::has_universal_relations)
            .transpose()?
            .unwrap_or(false);
        if self.gadget_recompositions.is_some_and(|registry| !registry.is_frozen()) {
            return Err(RelationRegistryError::NotFrozen.into());
        }
        let mut records = BTreeMap::<MonomialId, Record>::new();
        let mut census = DiagnosticFourClassCensus::default();
        for nf in normal_forms {
            let mut entry = DiagnosticFourClassTopEntry {
                nf_ordinal: nf.ordinal,
                under_product: nf.under_product,
                ..DiagnosticFourClassTopEntry::default()
            };
            for (monomial, coefficient) in &nf.normal_form.exact_terms {
                let bound_class = match self.bound_monomial(*monomial, coefficient)? {
                    NumericContract::Missing => DiagnosticTermClass::Missing,
                    NumericContract::Known(CoefficientBound::Large) => DiagnosticTermClass::Large,
                    NumericContract::Known(_) => DiagnosticTermClass::FiniteNoRelation,
                };
                let reasons = if bound_class == DiagnosticTermClass::FiniteNoRelation {
                    self.diagnostic_frontier_reasons(
                        *monomial,
                        nf.under_product,
                        has_closed,
                        has_universal,
                    )?
                } else {
                    0
                };
                let class = if reasons == 0 {
                    bound_class
                } else {
                    DiagnosticTermClass::FiniteRelationFrontier
                };
                let payload = self.monomials.descriptor_payload_lower_bound_bytes(*monomial)?;
                let stats = census.class_mut(class);
                stats.term_refs = stats.term_refs.saturating_add(1);
                entry.observe(class, payload);
                if class == DiagnosticTermClass::FiniteRelationFrontier {
                    census.frontier_reason_term_ref_union =
                        census.frontier_reason_term_ref_union.saturating_add(1);
                    for (index, bit) in [1_u8, 2, 4, 8, 16, 32].into_iter().enumerate() {
                        if reasons & bit != 0 &&
                            let Some(reason) = census.reason_mut(index)
                        {
                            reason.term_refs = reason.term_refs.saturating_add(1);
                        }
                    }
                }
                let record =
                    records.entry(*monomial).or_insert(Record { class, reasons: 0, payload });
                if class.rank() > record.class.rank() {
                    record.class = class;
                }
                record.reasons |= reasons;
            }
            let score = entry.total_refs();
            let len = usize::from(census.top_len);
            let insert = (0..len)
                .find(|index| {
                    let existing = &census.top[*index];
                    score > existing.total_refs() ||
                        (score == existing.total_refs() &&
                            entry.nf_ordinal < existing.nf_ordinal)
                })
                .unwrap_or(len);
            if insert < census.top.len() {
                let new_len = (len + 1).min(census.top.len());
                for index in (insert + 1..new_len).rev() {
                    census.top[index] = census.top[index - 1];
                }
                census.top[insert] = entry;
                census.top_len = u8::try_from(new_len).unwrap_or(8);
            }
        }
        for record in records.values() {
            let stats = census.class_mut(record.class);
            stats.unique_monomials = stats.unique_monomials.saturating_add(1);
            stats.payload_lower_bound_bytes =
                stats.payload_lower_bound_bytes.saturating_add(record.payload);
            if record.class == DiagnosticTermClass::FiniteRelationFrontier {
                census.frontier_unique_union = census.frontier_unique_union.saturating_add(1);
                if record.reasons != 0 {
                    census.frontier_reason_unique_union =
                        census.frontier_reason_unique_union.saturating_add(1);
                    census.frontier_reason_payload_union =
                        census.frontier_reason_payload_union.saturating_add(record.payload);
                }
                for (index, bit) in [1_u8, 2, 4, 8, 16, 32].into_iter().enumerate() {
                    if record.reasons & bit == 0 {
                        continue;
                    }
                    let Some(reason) = census.reason_mut(index) else { continue };
                    reason.unique_monomials = reason.unique_monomials.saturating_add(1);
                    reason.payload_lower_bound_bytes =
                        reason.payload_lower_bound_bytes.saturating_add(record.payload);
                }
            }
        }
        debug_assert_eq!(census.frontier_unique_union, census.frontier_reason_unique_union);
        debug_assert_eq!(
            census.finite_relation_frontier.term_refs,
            census.frontier_reason_term_ref_union
        );
        debug_assert_eq!(
            census.finite_relation_frontier.payload_lower_bound_bytes,
            census.frontier_reason_payload_union
        );
        Ok(census)
    }

    fn exact_plan_diagnostic_shape(
        &self,
    ) -> (u64, u64, u64, u64, u64, u64, u64, u64, DiagnosticMaterializedLeafTop8, u64) {
        let mut pending = self
            .exact_plans
            .values()
            .cloned()
            .map(|plan| (NodeExactState::Additive(plan), false))
            .chain(
                self.product_plans
                    .values()
                    .cloned()
                    .map(|plan| (NodeExactState::Product(plan), true)),
            )
            .chain(
                self.gadget_product_plans
                    .values()
                    .cloned()
                    .map(|plan| (NodeExactState::GadgetProduct(plan), true)),
            )
            .collect::<Vec<_>>();
        let mut seen_walk = BTreeSet::new();
        let mut seen_additive = BTreeSet::new();
        let mut seen_products = BTreeSet::<(u8, u64)>::new();
        let mut seen_leaves = BTreeSet::new();
        let mut seen_product_operands = BTreeSet::new();
        let mut leaf_exact_term_refs = 0_u64;
        let mut largest_leaf_exact_terms = 0_u64;
        let mut product_operand_exact_term_refs = 0_u64;
        let mut largest_product_operand_exact_terms = 0_u64;
        let mut materialized_top8 = DiagnosticMaterializedLeafTop8::default();
        let collect_origins = self.watchdog.is_some();
        while let Some((state, under_product)) = pending.pop() {
            match state {
                NodeExactState::Additive(plan) => {
                    seen_additive.insert(plan.id);
                    if seen_walk.insert((0, plan.id, under_product)) {
                        pending.push((plan.left.clone(), under_product));
                        pending.push((plan.right.clone(), under_product));
                    }
                }
                NodeExactState::Product(plan) => {
                    seen_products.insert((1, plan.id));
                    if seen_walk.insert((1, plan.id, under_product)) {
                        pending.push((plan.left.clone(), true));
                        pending.push((plan.right.clone(), true));
                    }
                }
                NodeExactState::GadgetProduct(plan) => {
                    seen_products.insert((2, plan.id));
                    for normal_form in [&plan.left, &plan.right] {
                        let key = Arc::as_ptr(normal_form) as usize;
                        if seen_product_operands.insert(key) {
                            let terms =
                                u64::try_from(normal_form.exact_terms.len()).unwrap_or(u64::MAX);
                            product_operand_exact_term_refs =
                                product_operand_exact_term_refs.saturating_add(terms);
                            largest_product_operand_exact_terms =
                                largest_product_operand_exact_terms.max(terms);
                        }
                    }
                }
                NodeExactState::Materialized { normal_form, .. } => {
                    let key = Arc::as_ptr(&normal_form) as usize;
                    let terms = u64::try_from(normal_form.exact_terms.len()).unwrap_or(u64::MAX);
                    if seen_leaves.insert(key) {
                        leaf_exact_term_refs = leaf_exact_term_refs.saturating_add(terms);
                        largest_leaf_exact_terms = largest_leaf_exact_terms.max(terms);
                        if collect_origins &&
                            let Some(origin) =
                                self.diagnostic_materialized_leaf_origin(&normal_form)
                        {
                            materialized_top8.observe(origin, terms);
                        }
                    }
                    if under_product && seen_product_operands.insert(key) {
                        product_operand_exact_term_refs =
                            product_operand_exact_term_refs.saturating_add(terms);
                        largest_product_operand_exact_terms =
                            largest_product_operand_exact_terms.max(terms);
                    }
                }
            }
        }
        (
            u64::try_from(seen_additive.len()).unwrap_or(u64::MAX),
            u64::try_from(seen_products.iter().filter(|(kind, _)| *kind == 2).count())
                .unwrap_or(u64::MAX),
            u64::try_from(seen_leaves.len()).unwrap_or(u64::MAX),
            leaf_exact_term_refs,
            largest_leaf_exact_terms,
            u64::try_from(seen_product_operands.len()).unwrap_or(u64::MAX),
            product_operand_exact_term_refs,
            largest_product_operand_exact_terms,
            materialized_top8,
            u64::try_from(seen_products.iter().filter(|(kind, _)| *kind == 1).count())
                .unwrap_or(u64::MAX),
        )
    }

    fn diagnostic_value_cache_top8(&self) -> DiagnosticValueCacheTop8 {
        let additive_leaf_ptrs = self.diagnostic_additive_leaf_ptrs();
        let mut result = DiagnosticValueCacheTop8 {
            entries: u64::try_from(self.cache.len()).unwrap_or(u64::MAX),
            ..DiagnosticValueCacheTop8::default()
        };
        for (&expression, value) in &self.cache {
            let Some(normal_form) = value.exact_nf.as_ref() else { continue };
            let term_count = u64::try_from(normal_form.exact_terms.len()).unwrap_or(u64::MAX);
            result.exact_term_refs = result.exact_term_refs.saturating_add(term_count);
            let candidate = DiagnosticValueCacheTopEntry {
                expression_slot: u64::from(expression.slot()),
                operator_category: self
                    .expressions
                    .node(expression)
                    .map(|node| normalization_operator_category(&node.operator))
                    .unwrap_or("invalid"),
                term_count,
                remaining_uses: self
                    .remaining_uses
                    .get(&expression)
                    .copied()
                    .map(|uses| u64::try_from(uses).unwrap_or(u64::MAX))
                    .unwrap_or(0),
                producer_input_count: 0,
                cached_input_exact_term_refs_sum: 0,
                cached_input_exact_term_refs_max: 0,
                additive_materialized_leaf: additive_leaf_ptrs
                    .contains(&(Arc::as_ptr(normal_form) as usize)),
                ..DiagnosticValueCacheTopEntry::default()
            };
            let current_len = usize::from(result.len);
            let insertion = (0..current_len).find(|&index| {
                candidate.term_count > result.top[index].term_count ||
                    (candidate.term_count == result.top[index].term_count &&
                        candidate.expression_slot < result.top[index].expression_slot)
            });
            let insertion = insertion.unwrap_or(current_len);
            if insertion >= result.top.len() {
                continue;
            }
            let new_len = (current_len + 1).min(result.top.len());
            for index in (insertion + 1..new_len).rev() {
                result.top[index] = result.top[index - 1];
            }
            result.top[insertion] = candidate;
            result.len = u8::try_from(new_len).unwrap_or(8);
        }
        result.top_exact_term_refs = result.top[..usize::from(result.len)]
            .iter()
            .fold(0_u64, |total, entry| total.saturating_add(entry.term_count));
        // Input ownership is intentionally inspected only after the bounded top-eight selection.
        // This keeps the diagnostic scan O(cache entries + 8 * producer arity * log(cache)) and
        // records only aggregate sizes, never semantic input identities or values.
        for entry in &mut result.top[..usize::from(result.len)] {
            let Ok(slot) = u32::try_from(entry.expression_slot) else { continue };
            let expression = ExprId::new(self.expressions.token(), slot);
            let Ok(node) = self.expressions.node(expression) else { continue };
            entry.producer_input_count = u64::try_from(node.inputs.len()).unwrap_or(u64::MAX);
            for input in &node.inputs {
                let term_count = self
                    .cache
                    .get(input)
                    .and_then(|value| value.exact_nf.as_ref())
                    .map(|normal_form| {
                        u64::try_from(normal_form.exact_terms.len()).unwrap_or(u64::MAX)
                    })
                    .unwrap_or(0);
                entry.cached_input_exact_term_refs_sum =
                    entry.cached_input_exact_term_refs_sum.saturating_add(term_count);
                entry.cached_input_exact_term_refs_max =
                    entry.cached_input_exact_term_refs_max.max(term_count);
            }
            self.populate_product_deferral_diagnostic(expression, node, entry);
        }
        result
    }

    fn diagnostic_additive_leaf_ptrs(&self) -> BTreeSet<usize> {
        let mut pending = self
            .exact_plans
            .values()
            .cloned()
            .map(NodeExactState::Additive)
            .chain(self.product_plans.values().cloned().map(NodeExactState::Product))
            .chain(self.gadget_product_plans.values().cloned().map(NodeExactState::GadgetProduct))
            .collect::<Vec<_>>();
        let mut seen_plans = BTreeSet::<(u8, u64)>::new();
        let mut leaves = BTreeSet::new();
        while let Some(state) = pending.pop() {
            match state {
                NodeExactState::Additive(plan) => {
                    if seen_plans.insert((0, plan.id)) {
                        pending.push(plan.left.clone());
                        pending.push(plan.right.clone());
                    }
                }
                NodeExactState::Product(plan) => {
                    if seen_plans.insert((1, plan.id)) {
                        pending.push(plan.left.clone());
                        pending.push(plan.right.clone());
                    }
                }
                NodeExactState::GadgetProduct(plan) => {
                    if seen_plans.insert((2, plan.id)) {
                        leaves.insert(Arc::as_ptr(&plan.left) as usize);
                        leaves.insert(Arc::as_ptr(&plan.right) as usize);
                    }
                }
                NodeExactState::Materialized { normal_form, .. } => {
                    leaves.insert(Arc::as_ptr(&normal_form) as usize);
                }
            }
        }
        leaves
    }

    fn populate_product_deferral_diagnostic(
        &self,
        expression: ExprId,
        node: &ExprNode,
        entry: &mut DiagnosticValueCacheTopEntry,
    ) {
        if !matches!(node.operator, ValueOperator::Matrix(MatrixOperation::Multiply)) {
            return;
        }
        let Some(snapshot) = self.diagnostic_product_evaluations.get(&expression).copied() else {
            entry.multiply_deferral_rejection = "missing_exact_operand";
            return;
        };
        entry.multiply_left_rows = snapshot.left_rows;
        entry.multiply_left_columns = snapshot.left_columns;
        entry.multiply_right_rows = snapshot.right_rows;
        entry.multiply_right_columns = snapshot.right_columns;
        entry.multiply_scalar_classification = snapshot.scalar_classification;
        let consumers =
            self.diagnostic_product_consumers.get(&expression).copied().unwrap_or_default();
        entry.multiply_add_sub_consumers = consumers.add_sub;
        entry.multiply_consumers = consumers.multiply;
        entry.multiply_structural_holds = consumers.structural;
        entry.multiply_root_other_consumers = consumers.root_other;
        let is_root = self.diagnostic_product_root == Some(expression);
        let non_root_other = consumers.root_other.saturating_sub(u64::from(is_root));
        if snapshot.scalar_classification != "ordinary" {
            entry.multiply_deferral_rejection = "scalar_shape";
        } else if consumers.multiply > 0 || non_root_other > 0 {
            entry.multiply_deferral_rejection = "non_additive_consumer";
        } else if consumers.structural > 0 {
            entry.multiply_deferral_rejection = "structural_hold";
        } else if snapshot.gadget_boundary {
            entry.multiply_deferral_rejection = "gadget_boundary";
        } else if is_root {
            entry.multiply_deferral_rejection = "root";
        } else if !snapshot.had_left_exact || !snapshot.had_right_exact {
            entry.multiply_deferral_rejection = "missing_exact_operand";
        } else {
            entry.multiply_deferral_rejection = "eligible";
        }
    }

    fn diagnostic_product_evaluation_snapshot(
        &self,
        node: &ExprNode,
        children: &[Arc<AnalyzedValue>],
    ) -> DiagnosticProductEvaluationSnapshot {
        let mut snapshot = DiagnosticProductEvaluationSnapshot {
            had_left_exact: children.first().is_some_and(|value| value.exact_nf.is_some()),
            had_right_exact: children.get(1).is_some_and(|value| value.exact_nf.is_some()),
            ..DiagnosticProductEvaluationSnapshot::default()
        };
        let left_type = node
            .inputs
            .first()
            .and_then(|input| self.expressions.value_type(*input).ok())
            .and_then(|value| match value {
                ResolvedValueType::Matrix(matrix) => Some(matrix),
                _ => None,
            });
        let right_type =
            node.inputs.get(1).and_then(|input| self.expressions.value_type(*input).ok()).and_then(
                |value| match value {
                    ResolvedValueType::Matrix(matrix) => Some(matrix),
                    _ => None,
                },
            );
        if let (Some(left), Some(right)) = (left_type, right_type) {
            snapshot.left_rows = u64::try_from(left.rows).unwrap_or(u64::MAX);
            snapshot.left_columns = u64::try_from(left.columns).unwrap_or(u64::MAX);
            snapshot.right_rows = u64::try_from(right.rows).unwrap_or(u64::MAX);
            snapshot.right_columns = u64::try_from(right.columns).unwrap_or(u64::MAX);
            let left_scalar = left.rows == 1 && left.columns == 1;
            let right_scalar = right.rows == 1 && right.columns == 1;
            snapshot.scalar_classification = match (left_scalar, right_scalar) {
                (false, false) => "ordinary",
                (true, false) => "scalar_left",
                (false, true) => "scalar_right",
                (true, true) => "scalar_both",
            };
        }
        snapshot.gadget_boundary = self.diagnostic_product_has_gadget_boundary(
            node,
            children.first().and_then(|value| value.exact_nf.as_deref()),
            children.get(1).and_then(|value| value.exact_nf.as_deref()),
        );
        snapshot
    }

    fn diagnostic_product_has_gadget_boundary(
        &self,
        node: &ExprNode,
        left: Option<&PolynomialNF>,
        right: Option<&PolynomialNF>,
    ) -> bool {
        if node.inputs.iter().any(|input| {
            self.expressions.node(*input).is_ok_and(|input| {
                matches!(
                    input.operator,
                    ValueOperator::Transform(ValueTransformOperation::GadgetDecompose { .. }) |
                        ValueOperator::Source(SemanticSourceIdentity {
                            matrix_constant: Some(MatrixConstantKind::Gadget { .. }),
                            ..
                        })
                )
            })
        }) {
            return true;
        }
        if self.gadget_recompositions.is_none() || node.inputs.len() != 2 {
            return false;
        }
        let (Some(left), Some(right)) = (left, right) else { return false };
        let mut gadget_endpoints = BTreeSet::new();
        for monomial in left.exact_terms.keys() {
            let Ok(descriptor) = self.monomials.descriptor(*monomial) else { continue };
            let Some(endpoint) = descriptor.ordered_factors.last() else { continue };
            if let Ok(input) = self.expressions.node(endpoint.expression()) {
                if let Some(MatrixConstantKind::Gadget { base, small }) =
                    input.operator.source_matrix_constant()
                {
                    gadget_endpoints.insert((*base, *small));
                }
            }
        }
        right.exact_terms.keys().any(|monomial| {
            let Ok(descriptor) = self.monomials.descriptor(*monomial) else { return false };
            let Some(endpoint) = descriptor.ordered_factors.first() else { return false };
            self.expressions.node(endpoint.expression()).is_ok_and(|input| {
                matches!(
                    &input.operator,
                    ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
                        base,
                        small,
                        ..
                    }) if gadget_endpoints.contains(&(*base, *small))
                )
            })
        })
    }

    fn clear_value_cache(&mut self) {
        self.cache.clear();
        self.exact_plans.clear();
        self.product_plans.clear();
        self.gadget_product_plans.clear();
        self.owner_counters.cache_exact_terms = 0;
    }

    fn insert_value_cache(&mut self, expression: ExprId, value: Arc<AnalyzedValue>) {
        if value.exact_nf.is_some() {
            self.exact_plans.remove(&expression);
            self.product_plans.remove(&expression);
            self.gadget_product_plans.remove(&expression);
        }
        let terms = Self::exact_terms(value.as_ref());
        if let Some(previous) = self.cache.insert(expression, value) {
            self.owner_counters.cache_exact_terms = self
                .owner_counters
                .cache_exact_terms
                .saturating_sub(Self::exact_terms(previous.as_ref()));
        }
        self.owner_counters.cache_exact_terms =
            self.owner_counters.cache_exact_terms.saturating_add(terms);
        self.owner_counters.cache_exact_terms_peak =
            self.owner_counters.cache_exact_terms_peak.max(self.owner_counters.cache_exact_terms);
        self.owner_counters.cache_largest_nf_terms_seen =
            self.owner_counters.cache_largest_nf_terms_seen.max(terms);
    }

    fn take_value_cache(&mut self, expression: ExprId) -> Option<Arc<AnalyzedValue>> {
        self.exact_plans.remove(&expression);
        self.product_plans.remove(&expression);
        self.gadget_product_plans.remove(&expression);
        let previous = self.cache.remove(&expression);
        if let Some(previous) = previous.as_ref() {
            self.owner_counters.cache_exact_terms = self
                .owner_counters
                .cache_exact_terms
                .saturating_sub(Self::exact_terms(previous.as_ref()));
        }
        previous
    }

    fn remove_value_cache(&mut self, expression: ExprId) {
        self.take_value_cache(expression);
    }

    fn clear_gadget_holds(&mut self) {
        self.gadget_input_nfs.clear();
        self.owner_counters.gadget_exact_terms = 0;
    }

    fn insert_gadget_hold(&mut self, expression: ExprId, normal_form: Arc<PolynomialNF>) {
        let terms = u64::try_from(normal_form.exact_terms.len()).unwrap_or(u64::MAX);
        if let Some(previous) = self.gadget_input_nfs.insert(expression, normal_form) {
            self.owner_counters.gadget_exact_terms = self
                .owner_counters
                .gadget_exact_terms
                .saturating_sub(u64::try_from(previous.exact_terms.len()).unwrap_or(u64::MAX));
        }
        self.owner_counters.gadget_exact_terms =
            self.owner_counters.gadget_exact_terms.saturating_add(terms);
        self.owner_counters.gadget_exact_terms_peak =
            self.owner_counters.gadget_exact_terms_peak.max(self.owner_counters.gadget_exact_terms);
        self.owner_counters.gadget_largest_nf_terms_seen =
            self.owner_counters.gadget_largest_nf_terms_seen.max(terms);
    }

    fn owner_census_with_active(
        &self,
        active: impl IntoIterator<Item = MonomialId>,
    ) -> DiagnosticOwnerCensus {
        let plan_roots = self.exact_plan_leaf_roots(false).unwrap_or_default();
        let cache_roots = self.cache.values().flat_map(|value| {
            value.exact_nf.iter().flat_map(|normal_form| normal_form.exact_terms.keys().copied())
        });
        let gadget_roots = self
            .gadget_input_nfs
            .values()
            .flat_map(|normal_form| normal_form.exact_terms.keys().copied());
        let canonical_roots =
            self.normalization.as_deref().into_iter().flat_map(NormalizationCache::monomial_roots);
        let closed_roots =
            self.relations.into_iter().flat_map(RelationRegistry::closed_monomial_roots);
        let suspended_roots = self.suspended_owner_roots.iter().copied();
        let monomial = self.monomials.owner_census(
            cache_roots
                .chain(plan_roots)
                .chain(gadget_roots)
                .chain(canonical_roots)
                .chain(closed_roots)
                .chain(suspended_roots)
                .chain(active),
        );
        let canonical =
            self.normalization.as_deref().map(|cache| cache.owner_census()).unwrap_or_default();
        let (
            additive_plan_nodes,
            gadget_product_plan_nodes,
            additive_unique_leaf_refs,
            additive_unique_leaf_exact_term_refs,
            additive_largest_leaf_exact_terms,
            gadget_product_unique_operand_refs,
            gadget_product_operand_exact_term_refs,
            gadget_product_largest_operand_exact_terms,
            materialized_leaf_top8,
            ordinary_product_plan_nodes,
        ) = self.exact_plan_diagnostic_shape();
        DiagnosticOwnerCensus {
            monomial_allocated_descriptor_slots: monomial.allocated_descriptor_slots,
            monomial_retained_descriptor_slots: monomial.retained_descriptor_slots,
            monomial_reclaimed_descriptor_slots: monomial.reclaimed_descriptor_slots,
            monomial_reachable_descriptor_slots: monomial.reachable_descriptor_slots,
            monomial_reachable_central_factor_entries: monomial.reachable_central_factor_entries,
            monomial_reachable_ordered_factor_entries: monomial.reachable_ordered_factor_entries,
            monomial_reachable_max_factor_word: monomial.reachable_max_factor_word,
            monomial_owned_payload_lower_bound_bytes: monomial.owned_payload_lower_bound_bytes,
            monomial_unreachable_descriptor_slots: monomial.unreachable_descriptor_slots,
            monomial_unreachable_central_factor_entries: monomial
                .unreachable_central_factor_entries,
            monomial_unreachable_ordered_factor_entries: monomial
                .unreachable_ordered_factor_entries,
            monomial_unreachable_payload_lower_bound_bytes: monomial
                .unreachable_payload_lower_bound_bytes,
            monomial_invalid_root_count: monomial.invalid_root_count,
            cache_entries: u64::try_from(self.cache.len()).unwrap_or(u64::MAX),
            cache_exact_terms: self.owner_counters.cache_exact_terms,
            cache_exact_terms_peak: self.owner_counters.cache_exact_terms_peak,
            cache_largest_nf_terms_seen: self.owner_counters.cache_largest_nf_terms_seen,
            gadget_entries: u64::try_from(self.gadget_input_nfs.len()).unwrap_or(u64::MAX),
            gadget_exact_terms: self.owner_counters.gadget_exact_terms,
            gadget_exact_terms_peak: self.owner_counters.gadget_exact_terms_peak,
            gadget_largest_nf_terms_seen: self.owner_counters.gadget_largest_nf_terms_seen,
            canonical_rhs_entries: canonical.canonical_rhs_entries,
            canonical_rhs_exact_terms: canonical.canonical_rhs_exact_terms,
            canonical_rhs_exact_terms_peak: canonical.canonical_rhs_exact_terms_peak,
            canonical_rhs_largest_nf_terms: canonical.canonical_rhs_largest_nf_terms,
            runtime_entries: canonical.runtime_entries,
            runtime_lhs_keys: canonical.runtime_lhs_keys,
            additive_plan_nodes,
            ordinary_product_plan_nodes,
            gadget_product_plan_nodes,
            additive_unique_leaf_refs,
            additive_unique_leaf_exact_term_refs,
            additive_largest_leaf_exact_terms,
            gadget_product_unique_operand_refs,
            gadget_product_operand_exact_term_refs,
            gadget_product_largest_operand_exact_terms,
            materialized_leaf_top8,
            gadget_product_plans_created: self.gadget_product_counters.plans_created,
            gadget_product_streamed_executions: self.gadget_product_counters.streamed_executions,
            gadget_product_zero_weight_skips: self.gadget_product_counters.zero_weight_skips,
            gadget_product_standalone_materializations: self
                .gadget_product_counters
                .standalone_materializations,
            gadget_product_planned_pairs: self.gadget_product_counters.planned_pairs,
            gadget_product_max_streamed_output_terms: self
                .gadget_product_counters
                .max_streamed_output_terms,
            ordinary_product_plans_created: self.product_plan_counters.plans_created,
            typed_product_candidate_plans: self.product_plan_counters.typed_candidate_plans,
            typed_product_direct_executions: self.product_plan_counters.typed_direct_executions,
            typed_product_pair_attempts: self.product_plan_counters.typed_pair_attempts,
            typed_product_pair_matches: self.product_plan_counters.typed_pair_matches,
            typed_product_pair_ordinary_fallbacks: self
                .product_plan_counters
                .typed_pair_ordinary_fallbacks,
            typed_product_standalone_materializations: self
                .product_plan_counters
                .typed_standalone_materializations,
            ordinary_product_streamed_executions: self.product_plan_counters.streamed_executions,
            ordinary_product_zero_weight_skips: self.product_plan_counters.zero_weight_skips,
            ordinary_product_standalone_materializations: self
                .product_plan_counters
                .standalone_materializations,
            ordinary_product_planned_pairs: self.product_plan_counters.planned_pairs,
            ordinary_product_max_streamed_output_terms: self
                .product_plan_counters
                .max_streamed_output_terms,
            scalar_action_plans_created: self.product_plan_counters.scalar_action_plans_created,
            scalar_action_streamed_executions: self
                .product_plan_counters
                .scalar_action_streamed_executions,
            scalar_action_zero_weight_skips: self
                .product_plan_counters
                .scalar_action_zero_weight_skips,
            scalar_action_standalone_materializations: self
                .product_plan_counters
                .scalar_action_standalone_materializations,
            scalar_action_reclassified_terms: self
                .product_plan_counters
                .scalar_action_reclassified_terms,
            scalar_action_reclassified_factors_max: self
                .product_plan_counters
                .scalar_action_reclassified_factors_max,
            additive_materializations: self.exact_plan_materializations,
            additive_materialization_output_terms_total: self
                .exact_plan_materialization_output_terms_total,
            additive_materialization_output_terms_max: self
                .exact_plan_materialization_output_terms_max,
        }
    }

    fn owner_census(&self) -> DiagnosticOwnerCensus {
        self.owner_census_with_active(std::iter::empty())
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
        let (plan_roots, materialized_leaf_top8, diagnostic_nfs) =
            self.exact_plan_leaf_roots_and_top8(true)?;
        // Classification is a separate expensive opt-in layered on the watchdog. It is computed
        // before mutation so an authority error aborts the sweep atomically, and committed only
        // after the real sweep succeeds. Ordinary watchdog owner telemetry remains enabled
        // without paying for this factor walk.
        let collect_four_class = self.watchdog.is_some() && self.four_class_census_enabled;
        let four_class_started = collect_four_class.then(Instant::now);
        let four_class = collect_four_class
            .then(|| self.diagnostic_four_class_census(&diagnostic_nfs))
            .transpose()?;
        let four_class_elapsed_ns = four_class_started
            .map(|started| u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX));
        let cache_roots = self.cache.values().flat_map(|value| {
            value.exact_nf.iter().flat_map(|normal_form| normal_form.exact_terms.keys().copied())
        });
        let gadget_roots = self
            .gadget_input_nfs
            .values()
            .flat_map(|normal_form| normal_form.exact_terms.keys().copied());
        let canonical_roots =
            self.normalization.as_deref().into_iter().flat_map(NormalizationCache::monomial_roots);
        let closed_roots =
            self.relations.into_iter().flat_map(RelationRegistry::closed_monomial_roots);
        let suspended_roots = self.suspended_owner_roots.iter().copied();
        let arena = self.monomials.token();
        // Value-cache, gadget, and suspended owners are local by construction. Do not filter
        // them: a foreign/tombstoned/out-of-range ID is an invariant violation and must reach the
        // arena's validate-before-mutate boundary. Canonical/runtime and closed-relation owners
        // are checker-global, so only their roots for this exact scoped arena participate.
        let canonical_roots = canonical_roots.filter(move |root| root.arena() == arena);
        let closed_roots = closed_roots.filter(move |root| root.arena() == arena);
        let allocated_payload_before = self.monomials.allocated_payload_since_sweep();
        // Timing starts only after the deterministic depth/allocation gate admits an actual
        // sweep attempt. A validation error leaves both the arena and telemetry unchanged.
        let started = Instant::now();
        let report = self.monomials.sweep_with_owners(
            self.protected_monomial_prefix,
            cache_roots,
            plan_roots,
            gadget_roots,
            canonical_roots,
            closed_roots,
            suspended_roots,
        )?;
        let elapsed_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        // This scan is diagnostic-only and is skipped entirely unless a watchdog exists.
        let value_cache_top8 = self.watchdog.as_ref().map(|_| self.diagnostic_value_cache_top8());
        self.gc_counters.sweep_count = self.gc_counters.sweep_count.saturating_add(1);
        self.gc_counters.last_sweep_node = self.counters.nodes_processed;
        self.gc_counters.last_high_water_slots = report.high_water_slots;
        self.gc_counters.last_occupied_slots = report.occupied_slots;
        self.gc_counters.last_reclaimed_slots = report.reclaimed_slots;
        self.gc_counters.last_reclaimed_payload_bytes = report.reclaimed_payload_lower_bound_bytes;
        self.gc_counters.cumulative_reclaimed_slots =
            self.gc_counters.cumulative_reclaimed_slots.saturating_add(report.reclaimed_slots);
        self.gc_counters.cumulative_reclaimed_payload_bytes = self
            .gc_counters
            .cumulative_reclaimed_payload_bytes
            .saturating_add(report.reclaimed_payload_lower_bound_bytes);
        self.gc_counters.last_allocated_payload_before_bytes = allocated_payload_before;
        self.gc_counters.last_bucket_entries = report.bucket_entries;
        self.gc_counters.last_protected_prefix_occupied_slots =
            report.protected_prefix_occupied_slots;
        self.gc_counters.last_occupied_central_factor_entries =
            report.occupied_central_factor_entries;
        self.gc_counters.last_occupied_ordered_factor_entries =
            report.occupied_ordered_factor_entries;
        self.gc_counters.last_occupied_factor_payload_lower_bound_bytes =
            report.occupied_factor_payload_lower_bound_bytes;
        self.gc_counters.last_protected_prefix = report.protected_prefix;
        self.gc_counters.last_value_cache = report.value_cache;
        self.gc_counters.last_exact_plan = report.exact_plan;
        self.gc_counters.last_gadget = report.gadget;
        self.gc_counters.last_canonical_runtime = report.canonical_runtime;
        self.gc_counters.last_closed = report.closed;
        self.gc_counters.last_suspended = report.suspended;
        if let Some(top8) = value_cache_top8 {
            self.gc_counters.value_cache_entries = top8.entries;
            self.gc_counters.value_cache_exact_term_refs = top8.exact_term_refs;
            self.gc_counters.value_cache_top8_exact_term_refs = top8.top_exact_term_refs;
            self.gc_counters.value_cache_top8_len = top8.len;
            self.gc_counters.value_cache_top8 = top8.top;
            self.gc_counters.materialized_leaf_top8 = materialized_leaf_top8;
            self.gc_counters.exact_plan_four_class = four_class.unwrap_or_default();
            if let Some(elapsed_ns) = four_class_elapsed_ns {
                self.gc_counters.four_class_total_ns =
                    self.gc_counters.four_class_total_ns.saturating_add(elapsed_ns);
                self.gc_counters.four_class_max_ns =
                    self.gc_counters.four_class_max_ns.max(elapsed_ns);
                self.gc_counters.four_class_last_ns = elapsed_ns;
            }
        }
        self.gc_counters.sweep_total_ns =
            self.gc_counters.sweep_total_ns.saturating_add(elapsed_ns);
        self.gc_counters.sweep_max_ns = self.gc_counters.sweep_max_ns.max(elapsed_ns);
        self.gc_counters.sweep_last_ns = elapsed_ns;
        let gc = self.gc_counters;
        // `watchdog_update` exits before locking when diagnostics are disabled.
        self.watchdog_update(|progress| progress.gc = gc);
        Ok(())
    }

    fn refresh_owner_diagnostics(&mut self) {
        // Cheap legacy trace boundary: fresh O(D) census is scheduled only by
        // `sample_owner_census` with an explicit reason.
    }

    fn sample_owner_census(
        &mut self,
        reason: OwnerCensusReason,
        active: impl IntoIterator<Item = MonomialId>,
    ) {
        if (self.watchdog.is_none() && !self.trace.active) || self.owner_census_samples >= 12 {
            return;
        }
        let owners = self.owner_census_with_active(active);
        self.owner_census_samples = self.owner_census_samples.saturating_add(1);
        self.owner_census_seq = self.owner_census_seq.saturating_add(1);
        let seq = self.owner_census_seq;
        let product_call_id = self.watchdog_product_call_id;
        if self.trace.active {
            self.trace.owners = owners;
        }
        let mut frozen = None;
        let mut owner_sample = None;
        self.watchdog_update(|progress| {
            progress.owners = owners;
            progress.owner_census_seq = seq;
            progress.owner_census_depth = progress.depth;
            progress.owner_census_call = progress.current_call;
            progress.owner_census_product_call_id = product_call_id;
            progress.owner_census_phase = progress.phase;
            progress.owner_census_reason = Some(reason);
            owner_sample = Some(DiagnosticOwnerSample {
                owners,
                seq,
                depth: progress.depth,
                call: progress.current_call,
                product_call_id,
                reason,
                phase: progress.phase,
                product_generated: progress.product_generated,
                product_enqueued: progress.product_enqueued,
                product_queue_current: progress.product_queue_current,
                product_output_current: progress.product_output_current,
            });
            frozen = Some(*progress);
        });
        if let (Some(watchdog), Some(owner_sample), Some(progress)) =
            (&self.watchdog, owner_sample, frozen)
        {
            watchdog.emit_owner_sample(owner_sample, progress);
        }
    }

    fn admits_large_product_census_pair(&self, planned: u64) -> bool {
        planned >= 100_000 &&
            self.large_product_pairs_sampled < 5 &&
            (self.largest_sampled_product_planned == 0 ||
                planned > self.largest_sampled_product_planned) &&
            self.owner_census_samples <= 9
    }

    fn admits_retained_arena_census(&mut self, retained: u64) -> bool {
        if self.owner_census_samples >= 8 || retained < self.next_retained_census_milestone {
            return false;
        }
        while retained >= self.next_retained_census_milestone {
            self.next_retained_census_milestone =
                self.next_retained_census_milestone.saturating_mul(2);
            if self.next_retained_census_milestone == u64::MAX {
                break;
            }
        }
        true
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
            exact_plans: BTreeMap::new(),
            product_plans: BTreeMap::new(),
            gadget_product_plans: BTreeMap::new(),
            next_exact_plan_id: 0,
            product_plan_counters: ProductPlanCounters::default(),
            gadget_product_counters: GadgetProductPlanCounters::default(),
            deferred_gadget_products: BTreeSet::new(),
            deferred_products: BTreeSet::new(),
            deferred_scalar_actions: BTreeSet::new(),
            diagnostic_product_consumers: BTreeMap::new(),
            diagnostic_product_evaluations: BTreeMap::new(),
            diagnostic_product_root: None,
            diagnostic_materialization_origins: BTreeMap::new(),
            diagnostic_materialized_leaf_origins: HashMap::new(),
            diagnostic_current_expression: None,
            expression_bounds: BTreeMap::new(),
            remaining_uses: BTreeMap::new(),
            gadget_input_nfs: BTreeMap::new(),
            owner_counters: NormalizerOwnerCounters::default(),
            counters: NormalizationCounters::default(),
            trace: NormalizationTrace::new(),
            watchdog: None,
            four_class_census_enabled: normalization_four_class_census_enabled(),
            watchdog_generation: 0,
            watchdog_product_processed: 0,
            watchdog_product_generated: 0,
            watchdog_product_enqueued: 0,
            watchdog_product_processed_current: 0,
            watchdog_product_planned_current: 0,
            watchdog_product_generation_current: 0,
            watchdog_product_enqueued_current: 0,
            watchdog_product_queue_current: 0,
            watchdog_product_output_current: 0,
            watchdog_relation_processed: 0,
            watchdog_specialization: DiagnosticSpecializationCounters::default(),
            watchdog_relation_closure: DiagnosticRelationCounters::default(),
            watchdog_timings: DiagnosticTimings::default(),
            suspended_owner_roots: Vec::new(),
            owner_census_samples: 0,
            owner_census_seq: 0,
            watchdog_product_call_id: 0,
            largest_sampled_product_planned: 0,
            large_product_pairs_sampled: 0,
            current_large_product_sampled: false,
            next_retained_census_milestone: 1 << 20,
            #[cfg(test)]
            watchdog_hot_publish_count: 0,
            normalization_depth: 0,
            #[cfg(test)]
            trace_product_heartbeat_interval: PRODUCT_PROCESSED_HEARTBEAT,
            #[cfg(test)]
            trace_focus_call_override: None,
            #[cfg(test)]
            trace_focus_expression_slot_override: None,
            #[cfg(test)]
            trace_focus_tail_nodes_override: None,
            #[cfg(test)]
            watchdog_enabled_override: None,
            #[cfg(test)]
            watchdog_interval_override: None,
            #[cfg(test)]
            last_watchdog_events: Vec::new(),
            #[cfg(test)]
            last_watchdog_snapshots: Vec::new(),
            #[cfg(test)]
            relation_matcher_publish_observer: None,
            relation_rewriting_enabled: true,
            fold_final_no_match: true,
            protected_monomial_prefix,
            monomial_gc_allocation_threshold_bytes: MONOMIAL_GC_ALLOCATION_THRESHOLD_BYTES,
            gc_counters: DiagnosticGcCounters::default(),
            exact_plan_materializations: 0,
            exact_plan_materialization_output_terms_total: 0,
            exact_plan_materialization_output_terms_max: 0,
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

    #[cfg(test)]
    fn with_trace_product_heartbeat_interval(mut self, interval: u64) -> Self {
        self.trace_product_heartbeat_interval = interval.max(1);
        self
    }

    #[cfg(test)]
    fn with_trace_focus_call_override(mut self, call: Option<u64>) -> Self {
        self.trace_focus_call_override = Some(call);
        self
    }

    #[cfg(test)]
    fn with_trace_focus_expression_slot_override(mut self, slot: Option<u64>) -> Self {
        self.trace_focus_expression_slot_override = Some(slot);
        self
    }

    #[cfg(test)]
    fn with_trace_focus_tail_nodes_override(mut self, tail: Option<u64>) -> Self {
        self.trace_focus_tail_nodes_override = Some(tail);
        self
    }

    #[cfg(test)]
    fn with_watchdog_override(mut self, enabled: bool, interval: Duration) -> Self {
        self.watchdog_enabled_override = Some(enabled);
        self.watchdog_interval_override = Some(interval);
        self
    }

    fn watchdog_update(&self, update: impl FnOnce(&mut DiagnosticProgress)) {
        if let Some(watchdog) = &self.watchdog {
            watchdog.update(update);
        }
    }

    fn watchdog_timing_start(&self) -> Option<Instant> {
        self.watchdog.as_ref().map(|_| Instant::now())
    }

    fn watchdog_record_timing(
        &mut self,
        started: Option<Instant>,
        update: fn(&mut DiagnosticTimings) -> &mut DiagnosticTimingCounter,
    ) {
        let Some(started) = started else { return };
        let elapsed_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        update(&mut self.watchdog_timings).record(elapsed_ns);
    }

    fn watchdog_record_product_processed(
        &mut self,
        queue_current: u64,
        output_current: u64,
        active: impl IntoIterator<Item = MonomialId>,
    ) {
        if self.watchdog.is_none() {
            return;
        }
        self.watchdog_product_queue_current = queue_current;
        self.watchdog_product_output_current = output_current;
        self.watchdog_product_processed = self.watchdog_product_processed.saturating_add(1);
        self.watchdog_product_processed_current =
            self.watchdog_product_processed_current.saturating_add(1);
        let processed = self.watchdog_product_processed;
        let processed_current = self.watchdog_product_processed_current;
        if processed_current >= 1_024 && processed_current.is_power_of_two() {
            let _ = active;
            self.watchdog_update(|progress| {
                progress.phase = DiagnosticPhase::ProductDrain;
                progress.product_processed = processed;
                progress.product_processed_current = processed_current;
                progress.product_queue_current = queue_current;
                progress.product_output_current = output_current;
            });
            #[cfg(test)]
            {
                self.watchdog_hot_publish_count = self.watchdog_hot_publish_count.saturating_add(1);
            }
        }
    }

    fn watchdog_record_product_generated(
        &mut self,
        enqueued: bool,
        queue_current: u64,
        active: impl IntoIterator<Item = MonomialId>,
    ) {
        if self.watchdog.is_none() {
            return;
        }
        self.watchdog_product_generated = self.watchdog_product_generated.saturating_add(1);
        self.watchdog_product_generation_current =
            self.watchdog_product_generation_current.saturating_add(1);
        if enqueued {
            self.watchdog_product_enqueued = self.watchdog_product_enqueued.saturating_add(1);
            self.watchdog_product_enqueued_current =
                self.watchdog_product_enqueued_current.saturating_add(1);
        }
        self.watchdog_product_queue_current = queue_current;
        self.watchdog_product_output_current = 0;
        let generated = self.watchdog_product_generated;
        let enqueued_count = self.watchdog_product_enqueued;
        let generation_current = self.watchdog_product_generation_current;
        let enqueued_current = self.watchdog_product_enqueued_current;
        let generated_crossing = generation_current == 1 || generation_current.is_power_of_two();
        let enqueued_crossing =
            enqueued && (enqueued_current == 1 || enqueued_current.is_power_of_two());
        if generated_crossing || enqueued_crossing {
            let _ = active;
            self.watchdog_update(|progress| {
                progress.phase = DiagnosticPhase::ProductGeneration;
                progress.product_generated = generated;
                progress.product_enqueued = enqueued_count;
                progress.product_generation_current = generation_current;
                progress.product_enqueued_current = enqueued_current;
                progress.product_queue_current = queue_current;
                progress.product_output_current = 0;
            });
            #[cfg(test)]
            {
                self.watchdog_hot_publish_count = self.watchdog_hot_publish_count.saturating_add(1);
            }
        }
    }

    fn watchdog_record_relation_processed(&mut self, active: impl IntoIterator<Item = MonomialId>) {
        if self.watchdog.is_none() {
            return;
        }
        self.watchdog_relation_processed = self.watchdog_relation_processed.saturating_add(1);
        let processed = self.watchdog_relation_processed;
        if processed >= 1_024 && processed.is_power_of_two() {
            let _ = active;
            self.watchdog_update(|progress| {
                progress.phase = DiagnosticPhase::RelationSearch;
                progress.relation_processed = processed;
            });
            #[cfg(test)]
            {
                self.watchdog_hot_publish_count = self.watchdog_hot_publish_count.saturating_add(1);
            }
        }
    }

    fn watchdog_record_specialization(
        &mut self,
        phase: DiagnosticPhase,
        update: impl FnOnce(&mut DiagnosticSpecializationCounters),
    ) {
        if self.watchdog.is_none() {
            return;
        }
        let before = self.watchdog_specialization;
        update(&mut self.watchdog_specialization);
        let after = self.watchdog_specialization;
        let crossed_publication_threshold = |old: u64, new: u64| {
            if new <= old {
                return false;
            }
            if old == 0 {
                return true;
            }
            old.checked_add(1)
                .and_then(u64::checked_next_power_of_two)
                .is_some_and(|threshold| threshold <= new)
        };
        let changed = [
            (before.runtime_lookup_hits, after.runtime_lookup_hits),
            (before.runtime_lookup_misses, after.runtime_lookup_misses),
            (before.ordinary_specializations_started, after.ordinary_specializations_started),
            (before.ordinary_specializations_completed, after.ordinary_specializations_completed),
            (before.proof_specializations_started, after.proof_specializations_started),
            (before.proof_specializations_completed, after.proof_specializations_completed),
            (before.registrations_started, after.registrations_started),
            (before.registrations_completed, after.registrations_completed),
            (before.rhs_exact_terms_total, after.rhs_exact_terms_total),
            (before.interner_existing, after.interner_existing),
            (before.interner_inserted, after.interner_inserted),
            (before.proof_rollbacks_completed, after.proof_rollbacks_completed),
        ]
        .into_iter()
        .chain([(before.rhs_exact_terms_max, after.rhs_exact_terms_max)])
        .any(|(old, new)| crossed_publication_threshold(old, new));
        if changed {
            self.watchdog_update(|progress| {
                progress.phase = phase;
                progress.specialization = after;
                progress.timings = self.watchdog_timings;
            });
            #[cfg(test)]
            {
                self.watchdog_hot_publish_count = self.watchdog_hot_publish_count.saturating_add(1);
            }
        }
    }

    fn watchdog_publish_relation_closure(
        &mut self,
        phase: DiagnosticPhase,
        counters: DiagnosticRelationCounters,
    ) {
        if self.watchdog.is_none() {
            return;
        }
        self.watchdog_relation_closure = counters;
        self.watchdog_update(|progress| {
            progress.phase = phase;
            progress.relation_closure = counters;
            progress.timings = self.watchdog_timings;
        });
        #[cfg(test)]
        {
            self.watchdog_hot_publish_count = self.watchdog_hot_publish_count.saturating_add(1);
        }
    }

    fn watchdog_publish_relation_closure_with_active(
        &mut self,
        phase: DiagnosticPhase,
        counters: DiagnosticRelationCounters,
        active: impl IntoIterator<Item = MonomialId>,
    ) {
        if self.watchdog.is_none() {
            return;
        }
        let _ = active;
        self.watchdog_relation_closure = counters;
        self.watchdog_update(|progress| {
            progress.phase = phase;
            progress.relation_closure = counters;
            progress.timings = self.watchdog_timings;
        });
        #[cfg(test)]
        {
            self.watchdog_hot_publish_count = self.watchdog_hot_publish_count.saturating_add(1);
        }
    }

    fn watchdog_maybe_publish_relation_progress(
        &mut self,
        before: DiagnosticRelationCounters,
        after: DiagnosticRelationCounters,
    ) {
        let crossed = |old: u64, new: u64| {
            new > old &&
                (old == 0 ||
                    old.checked_add(1)
                        .and_then(u64::checked_next_power_of_two)
                        .is_some_and(|threshold| threshold <= new))
        };
        let selected = [
            (before.dequeued, after.dequeued),
            (before.zero_skipped, after.zero_skipped),
            (before.nonzero_dequeued, after.nonzero_dequeued),
            (before.enqueued, after.enqueued),
            (before.queue_peak, after.queue_peak),
            (before.central_factors_total, after.central_factors_total),
            (before.central_factors_max, after.central_factors_max),
            (before.ordered_factors_total, after.ordered_factors_total),
            (before.ordered_factors_max, after.ordered_factors_max),
            (before.gadget_attempts, after.gadget_attempts),
            (before.gadget_output_terms_total, after.gadget_output_terms_total),
            (before.gadget_output_terms_max, after.gadget_output_terms_max),
            (before.whole_closed_probes, after.whole_closed_probes),
            (before.closed_window_probes, after.closed_window_probes),
            (before.universal_probes, after.universal_probes),
            (before.universal_lhs_candidates, after.universal_lhs_candidates),
            (before.universal_span_candidates, after.universal_span_candidates),
            (before.rhs_terms_total, after.rhs_terms_total),
            (before.rhs_terms_max, after.rhs_terms_max),
            (before.monomial_combines, after.monomial_combines),
        ];
        if selected.into_iter().any(|(old, new)| crossed(old, new)) {
            self.watchdog_publish_relation_closure(DiagnosticPhase::RelationSearch, after);
        }
    }

    /// Publish progress from matcher-internal scans before they return to the outer relation
    /// worklist. These counters advance one at a time, so the local power-of-two test is the only
    /// hot-loop cost; the watchdog mutex is acquired only at the bounded thresholds.
    fn watchdog_publish_relation_matcher_counter(
        watchdog: Option<&DiagnosticWatchdog>,
        aggregate: &mut DiagnosticRelationCounters,
        before: u64,
        after: u64,
        counters: DiagnosticRelationCounters,
    ) -> bool {
        if after <= before || !after.is_power_of_two() {
            return false;
        }
        let Some(watchdog) = watchdog else { return false };
        *aggregate = counters;
        watchdog.update(|progress| {
            progress.phase = DiagnosticPhase::RelationSearch;
            progress.relation_closure = counters;
        });
        true
    }

    pub fn normalize(&mut self, root: ScopedExprId) -> Result<AnalyzedValue, NormalizeError> {
        self.normalize_with_trace_authority(root, false, None)
    }

    pub(super) fn normalize_with_trace(
        &mut self,
        root: ScopedExprId,
    ) -> Result<AnalyzedValue, NormalizeError> {
        self.normalize_with_trace_authority(root, true, None)
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
        self.normalize_with_trace_authority(root, false, Some(proof))
    }

    fn normalize_with_trace_authority(
        &mut self,
        root: ScopedExprId,
        force_outer_trace: bool,
        scope_proof: Option<ScopeProof>,
    ) -> Result<AnalyzedValue, NormalizeError> {
        let outermost = self.normalization_depth == 0;
        if outermost {
            self.protected_monomial_prefix = self.monomials.len();
            self.gc_counters = DiagnosticGcCounters::default();
            self.trace = NormalizationTrace::new();
            let enabled = normalization_watchdog_enabled();
            #[cfg(test)]
            let enabled = self.watchdog_enabled_override.unwrap_or(enabled);
            if enabled {
                self.watchdog_generation = self.watchdog_generation.saturating_add(1);
                self.watchdog_product_processed = 0;
                self.watchdog_product_generated = 0;
                self.watchdog_product_enqueued = 0;
                self.watchdog_product_processed_current = 0;
                self.watchdog_product_planned_current = 0;
                self.watchdog_product_generation_current = 0;
                self.watchdog_product_enqueued_current = 0;
                self.watchdog_product_queue_current = 0;
                self.watchdog_product_output_current = 0;
                self.watchdog_relation_processed = 0;
                self.watchdog_specialization = DiagnosticSpecializationCounters::default();
                self.watchdog_relation_closure = DiagnosticRelationCounters::default();
                self.watchdog_timings = DiagnosticTimings::default();
                self.suspended_owner_roots.clear();
                self.owner_census_samples = 0;
                self.owner_census_seq = 0;
                self.watchdog_product_call_id = 0;
                self.largest_sampled_product_planned = 0;
                self.large_product_pairs_sampled = 0;
                self.current_large_product_sampled = false;
                self.next_retained_census_milestone = 1 << 20;
                #[cfg(test)]
                {
                    self.watchdog_hot_publish_count = 0;
                }
                let interval = normalization_watchdog_interval();
                #[cfg(test)]
                let interval = self.watchdog_interval_override.unwrap_or(interval);
                self.watchdog = DiagnosticWatchdog::start(self.watchdog_generation, interval);
            }
        }
        let watchdog_parent = self
            .watchdog
            .as_mut()
            .map(|watchdog| watchdog.enter_call(self.normalization_depth.saturating_add(1)));
        let previous_trace_call = self.trace.current_normalization_call;
        let previous_subphase = self.trace.current_subphase;
        self.normalization_depth = self.normalization_depth.saturating_add(1);
        let result = self.normalize_traced(root, outermost, force_outer_trace, scope_proof);
        self.normalization_depth = self.normalization_depth.saturating_sub(1);
        if !outermost {
            self.trace.record_completed_invocation(self.trace.current_normalization_call);
            self.trace.current_normalization_call = previous_trace_call;
            self.trace.current_subphase = previous_subphase;
        }
        if let (Some(watchdog), Some(parent)) = (&self.watchdog, watchdog_parent) {
            watchdog.complete_call(parent, result.is_err());
        }
        if outermost {
            self.sample_owner_census(
                if result.is_ok() {
                    OwnerCensusReason::OuterTerminal
                } else {
                    OwnerCensusReason::OuterError
                },
                result
                    .as_ref()
                    .ok()
                    .and_then(|value| value.exact_nf.as_ref())
                    .into_iter()
                    .flat_map(|normal_form| normal_form.exact_terms.keys().copied()),
            );
        }
        if outermost {
            if self.trace.active {
                self.trace.cache_len = u64::try_from(self.cache.len()).unwrap_or(u64::MAX);
                self.trace.monomial_len = u64::try_from(self.monomials.len()).unwrap_or(u64::MAX);
            }
            self.trace.emit(
                if result.is_ok() { "normalize_end" } else { "normalize_error" },
                self.trace.nodes_processed,
                self.trace.nodes_total,
                true,
            );
            let product_processed = self.watchdog_product_processed;
            let product_generated = self.watchdog_product_generated;
            let product_enqueued = self.watchdog_product_enqueued;
            let product_processed_current = self.watchdog_product_processed_current;
            let product_planned_current = self.watchdog_product_planned_current;
            let product_generation_current = self.watchdog_product_generation_current;
            let product_enqueued_current = self.watchdog_product_enqueued_current;
            let product_queue_current = self.watchdog_product_queue_current;
            let product_output_current = self.watchdog_product_output_current;
            let relation_processed = self.watchdog_relation_processed;
            let specialization = self.watchdog_specialization;
            let relation_closure = self.watchdog_relation_closure;
            let timings = self.watchdog_timings;
            self.watchdog_update(|progress| {
                progress.product_processed = product_processed;
                progress.product_generated = product_generated;
                progress.product_enqueued = product_enqueued;
                progress.product_processed_current = product_processed_current;
                progress.product_planned_current = product_planned_current;
                progress.product_generation_current = product_generation_current;
                progress.product_enqueued_current = product_enqueued_current;
                progress.product_queue_current = product_queue_current;
                progress.product_output_current = product_output_current;
                progress.relation_processed = relation_processed;
                progress.specialization = specialization;
                progress.relation_closure = relation_closure;
                progress.timings = timings;
            });
            if let Some(watchdog) = self.watchdog.as_mut() {
                watchdog.finish(result.is_err());
                #[cfg(test)]
                {
                    self.last_watchdog_events = watchdog
                        .shared
                        .events
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .clone();
                    self.last_watchdog_snapshots = watchdog
                        .shared
                        .snapshots
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .clone();
                }
            }
            self.watchdog = None;
        }
        result
    }

    fn normalize_traced(
        &mut self,
        root: ScopedExprId,
        outermost: bool,
        force_outer_trace: bool,
        scope_proof: Option<ScopeProof>,
    ) -> Result<AnalyzedValue, NormalizeError> {
        if root.program() != self.scope {
            return Err(NormalizeError::InvalidScope {
                expected: self.scope,
                actual: root.program(),
            });
        }
        if !outermost {
            self.trace.record_nested_normalization(0, self.normalization_depth);
        }
        // Relation closure is lexical over the complete root word. Defer it until all expression
        // children have been assembled; otherwise a child `B*K` rewrite would discard the active
        // relation before its parent exposes the boundary `B*K` again.
        let relation_rewriting_enabled = self.relation_rewriting_enabled;
        self.relation_rewriting_enabled = false;
        // The root may be a beta-reduced specialization derived in this scope rather than the
        // finalized program's original root. Validate that exact root against the registered
        // signature instead of proving reachability from a different canonical root.
        let trace_next_root_proof =
            !outermost && std::mem::take(&mut self.trace.next_root_normalize_proof_pending);
        self.trace.enter_next_root_phase("next_root:normalize_proof_start", trace_next_root_proof);
        self.watchdog_update(|progress| progress.phase = DiagnosticPhase::ScopeProof);
        let scope_proof_started = self.watchdog_timing_start();
        let scope_proof = match scope_proof {
            Some(proof) => Ok(proof),
            None => self.expressions.scope_proof(root.program(), root.expression()),
        };
        self.watchdog_record_timing(scope_proof_started, |timings| &mut timings.outer_scope_proof);
        let mut scope_proof = scope_proof?;
        self.watchdog_update(|progress| progress.phase = DiagnosticPhase::ScopeProofDone);
        self.trace.enter_next_root_phase("next_root:normalize_proof_end", trace_next_root_proof);
        self.watchdog_update(|progress| progress.phase = DiagnosticPhase::StateReset);
        self.clear_value_cache();
        self.deferred_gadget_products.clear();
        self.deferred_products.clear();
        self.deferred_scalar_actions.clear();
        self.diagnostic_product_consumers.clear();
        self.diagnostic_product_evaluations.clear();
        self.diagnostic_product_root = None;
        self.diagnostic_materialization_origins.clear();
        self.diagnostic_materialized_leaf_origins.clear();
        self.diagnostic_current_expression = None;
        self.expression_bounds.clear();
        self.remaining_uses.clear();
        self.clear_gadget_holds();
        self.owner_counters = NormalizerOwnerCounters::default();
        if outermost {
            self.exact_plan_materializations = 0;
            self.exact_plan_materialization_output_terms_total = 0;
            self.exact_plan_materialization_output_terms_max = 0;
            self.gadget_product_counters = GadgetProductPlanCounters::default();
        }
        self.counters = NormalizationCounters::default();
        self.watchdog_update(|progress| progress.phase = DiagnosticPhase::UseCounts);
        let use_counts_started = self.watchdog_timing_start();
        let reachable = self.compute_use_counts(root.expression());
        self.watchdog_record_timing(use_counts_started, |timings| &mut timings.outer_use_counts);
        let reachable = reachable?;
        self.counters.nodes_total = reachable.len() as u64;
        let nodes_total = self.counters.nodes_total;
        self.watchdog_update(|progress| {
            progress.phase = DiagnosticPhase::UseCountsDone;
            progress.nodes_total = nodes_total;
        });
        if outermost {
            if force_outer_trace && self.watchdog.is_none() {
                let focus_normalization_call = normalization_trace_focus_call_from_env();
                let focus_expression_slot = normalization_trace_focus_expression_slot_from_env();
                let focus_tail_nodes = normalization_trace_focus_tail_nodes_from_env();
                #[cfg(test)]
                let focus_normalization_call =
                    self.trace_focus_call_override.unwrap_or(focus_normalization_call);
                #[cfg(test)]
                let focus_expression_slot =
                    self.trace_focus_expression_slot_override.unwrap_or(focus_expression_slot);
                #[cfg(test)]
                let focus_tail_nodes =
                    self.trace_focus_tail_nodes_override.unwrap_or(focus_tail_nodes);
                self.trace.focus_normalization_call = focus_normalization_call;
                self.trace.focus_expression_slot = focus_expression_slot;
                self.trace.focus_tail_nodes = focus_tail_nodes;
                self.trace.activate(
                    self.counters.nodes_total,
                    self.normalization_depth,
                    self.monomials.len(),
                );
                #[cfg(test)]
                {
                    self.trace.product_heartbeat_interval = self.trace_product_heartbeat_interval;
                    self.trace.next_product_generated_heartbeat =
                        self.trace_product_heartbeat_interval;
                    self.trace.next_product_processed_heartbeat =
                        self.trace_product_heartbeat_interval;
                }
                self.refresh_owner_diagnostics();
                self.trace.emit(
                    "normalize_start",
                    self.trace.nodes_processed,
                    self.trace.nodes_total,
                    false,
                );
            }
        } else {
            self.trace.record_nested_nodes(self.counters.nodes_total);
            self.trace.enter_focused_invocation_phase("specialized_root:normalize_enter");
        }
        self.watchdog_update(|progress| progress.phase = DiagnosticPhase::NodeWalk);
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
            if self.trace.active {
                self.trace.current_expression_slot = u64::from(expression.slot());
                self.trace.current_operator = normalization_operator_category(&node.operator);
            }
            let remaining_in_current_normalization =
                self.counters.nodes_total.saturating_sub(self.counters.nodes_processed);
            self.trace.emit_node_start(remaining_in_current_normalization);
            let expression_slot = u64::from(expression.slot());
            let operator = normalization_operator_category(&node.operator);
            self.watchdog_update(|progress| {
                progress.phase = DiagnosticPhase::EvaluateNode;
                progress.expression_slot = expression_slot;
                progress.operator = operator;
            });
            let value = self.evaluate_node(&mut scope_proof, expression, node.as_ref())?;
            // Keep only the compact typed transfer, not the exact NF, after a node's last use.
            // The final root fold can therefore recover bounds for released derived factors.
            self.expression_bounds.insert(expression, value.coefficient_bound.clone());
            self.counters.nodes_processed = self.counters.nodes_processed.saturating_add(1);
            let nodes_done = self.counters.nodes_processed;
            self.watchdog_update(|progress| {
                progress.phase = DiagnosticPhase::NodeWalk;
                progress.nodes_done = nodes_done;
            });
            self.insert_value_cache(expression, Arc::new(value));
            completed.insert(expression);
            self.sweep_monomials_at_node_commit()?;
            self.counters.peak_cached_values =
                self.counters.peak_cached_values.max(self.cache.len() as u64);
            if self.trace.active {
                self.trace.nodes_processed = self.trace.nodes_processed.saturating_add(1);
                self.trace.cache_len = u64::try_from(self.cache.len()).unwrap_or(u64::MAX);
                self.trace.cache_peak = self.trace.cache_peak.max(self.trace.cache_len);
                self.trace.monomial_len = u64::try_from(self.monomials.len()).unwrap_or(u64::MAX);
                if self.trace.nodes_processed >= self.trace.next_node_heartbeat {
                    self.refresh_owner_diagnostics();
                    self.trace.emit(
                        "normalize_heartbeat",
                        self.trace.nodes_processed,
                        self.trace.nodes_total,
                        false,
                    );
                    self.trace.next_node_heartbeat =
                        self.trace.next_node_heartbeat.saturating_add(NORMALIZATION_NODE_HEARTBEAT);
                }
            }
        }
        self.watchdog_update(|progress| progress.phase = DiagnosticPhase::Post);
        let deferred_root = self
            .exact_plans
            .remove(&root.expression())
            .map(NodeExactState::Additive)
            .or_else(|| self.product_plans.remove(&root.expression()).map(NodeExactState::Product))
            .or_else(|| {
                self.gadget_product_plans
                    .remove(&root.expression())
                    .map(NodeExactState::GadgetProduct)
            });
        if let Some(deferred_root) = deferred_root {
            let reason = if self.normalization_depth > 1 {
                DiagnosticMaterializationReason::NestedAdditiveReturn
            } else {
                DiagnosticMaterializationReason::Root
            };
            let materialized =
                self.materialize_exact_state_for(&deferred_root, root.expression(), reason, None)?;
            let cached = self
                .cache
                .get(&root.expression())
                .cloned()
                .ok_or(NormalizeError::MissingCachedValue { expression: root.expression() })?;
            let materialized_value =
                synchronize_materialized_value(cached.as_ref().clone(), materialized);
            self.insert_value_cache(root.expression(), Arc::new(materialized_value));
        }
        let root_nf = self.cache.get(&root.expression()).and_then(|value| value.exact_nf.clone());
        self.update_trace_root_shape(root_nf.as_deref());
        self.trace.enter_postphase("post:root_take");
        let value = self
            .take_value_cache(root.expression())
            .ok_or(NormalizeError::MissingCachedValue { expression: root.expression() })?;
        let mut value = Arc::try_unwrap(value)
            .map_err(|_| NormalizeError::SharedRootCacheValue { expression: root.expression() })?;
        self.update_trace_root_shape(value.exact_nf.as_deref());
        self.trace.enter_postphase("post:root_unwrap");
        self.relation_rewriting_enabled = relation_rewriting_enabled;
        self.trace.enter_postphase("post:relation_rewrite");
        let relation_rebound_started = self.watchdog_timing_start();
        let relation_rebound = (|| -> Result<(), NormalizeError> {
            if self.relations.is_some() && self.relation_rewriting_enabled {
                if let Some(exact_nf) = value.exact_nf.as_mut() {
                    let normal_form = Arc::make_mut(exact_nf);
                    let changed = self.rewrite_closed_relations(normal_form)?;
                    if changed {
                        // Relation closure replaces the old exact word. Do not carry its summary
                        // (which may be Large because of a pre-rewrite plain hash) into rebound.
                        normal_form.bounded_summary =
                            BoundedSummary::known(CoefficientBound::ExactZero);
                        let rebound = self.bound_normal_form(normal_form)?;
                        normal_form.bounded_summary.coefficient_bound = rebound.clone();
                        value.coefficient_bound = rebound;
                    }
                }
            }
            Ok(())
        })();
        self.watchdog_record_timing(relation_rebound_started, |timings| {
            &mut timings.outer_relation_rebound
        });
        relation_rebound?;
        self.update_trace_root_shape(value.exact_nf.as_deref());
        self.trace.enter_postphase("post:relation_rebound");
        self.trace.enter_postphase("post:fold_bound");
        let bound_fold_started = self.watchdog_timing_start();
        let bound_fold = (|| -> Result<(), NormalizeError> {
            if self.fold_final_no_match &&
                self.relations.is_some() &&
                self.relation_rewriting_enabled
            {
                if let Some(exact_nf) = value.exact_nf.as_mut() {
                    let normal_form = Arc::make_mut(exact_nf);
                    // Compute the total while exact factors are still present, then fold finite
                    // terms without counting them a second time.
                    let rebound = match &normal_form.bounded_summary.coefficient_bound {
                        NumericContract::Known(bound) => NumericContract::Known(bound.clone()),
                        NumericContract::Missing => self.bound_normal_form(normal_form)?,
                    };
                    self.trace.enter_postphase("post:fold_terms");
                    self.fold_finite_no_match_terms(normal_form)?;
                    normal_form.bounded_summary.coefficient_bound = rebound.clone();
                    value.coefficient_bound = rebound;
                    if normal_form.is_zero() {
                        value.coefficient_bound =
                            NumericContract::Known(CoefficientBound::ExactZero);
                        normal_form.bounded_summary =
                            BoundedSummary::known(CoefficientBound::ExactZero);
                    }
                }
            }
            Ok(())
        })();
        self.watchdog_record_timing(bound_fold_started, |timings| &mut timings.outer_bound_fold);
        bound_fold?;
        // The relation worklist reaches a fixed point before this stage; any retained exact term
        // therefore has no applicable relation boundary.  Ambiguous/unresolved registrations are
        // intentionally fail-closed and are represented by the retained exact term itself.
        self.update_trace_root_shape(value.exact_nf.as_deref());
        self.trace.enter_postphase("post:relation_remaining");
        self.counters.relation_remaining = value
            .exact_nf
            .as_deref()
            .map(|normal_form| self.count_relation_remaining(normal_form))
            .unwrap_or(0);
        self.counters.final_exact_term_count =
            value.exact_nf.as_ref().map_or(0, |normal_form| normal_form.exact_terms.len() as u64);
        self.update_trace_root_shape(value.exact_nf.as_deref());
        self.trace.enter_postphase("post:complete");
        Ok(value)
    }

    fn update_trace_root_shape(&mut self, normal_form: Option<&PolynomialNF>) {
        if !self.trace.relation_trace_selected() {
            return;
        }
        let Some(normal_form) = normal_form else {
            self.trace.root_exact_terms = 0;
            self.trace.root_sum_central_factors = 0;
            self.trace.root_max_central_factors = 0;
            self.trace.root_sum_ordered_factors = 0;
            self.trace.root_max_ordered_factors = 0;
            return;
        };
        let mut sum_central = 0_u64;
        let mut max_central = 0_u64;
        let mut sum_ordered = 0_u64;
        let mut max_ordered = 0_u64;
        for monomial in normal_form.exact_terms.keys() {
            let Ok(descriptor) = self.monomials.descriptor(*monomial) else { continue };
            let central = u64::try_from(descriptor.central_factors.len()).unwrap_or(u64::MAX);
            let ordered = u64::try_from(descriptor.ordered_factors.len()).unwrap_or(u64::MAX);
            sum_central = sum_central.saturating_add(central);
            max_central = max_central.max(central);
            sum_ordered = sum_ordered.saturating_add(ordered);
            max_ordered = max_ordered.max(ordered);
        }
        self.trace.root_exact_terms =
            u64::try_from(normal_form.exact_terms.len()).unwrap_or(u64::MAX);
        self.trace.root_sum_central_factors = sum_central;
        self.trace.root_max_central_factors = max_central;
        self.trace.root_sum_ordered_factors = sum_ordered;
        self.trace.root_max_ordered_factors = max_ordered;
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
            self.watchdog_record_specialization(DiagnosticPhase::RuntimeLookup, |counters| {
                counters.runtime_lookup_hits = counters.runtime_lookup_hits.saturating_add(1);
            });
            return super::relation::resolve_candidates(
                cached.get(&CanonicalLhsKey { layout: layout.cloned(), monomial }),
            )
            .map_err(Into::into);
        }
        self.watchdog_record_specialization(DiagnosticPhase::RuntimeLookup, |counters| {
            counters.runtime_lookup_misses = counters.runtime_lookup_misses.saturating_add(1);
        });
        let specialized =
            self.specialize_universal(dispatch, index, index_range, SpecializationKind::Ordinary)?;
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
        let result = self
            .specialize_universal(dispatch, index, index_range, SpecializationKind::Proof)
            .and_then(|local| {
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
        if self.watchdog.is_some() {
            if let Ok(ProofResolutionOwned::Rewrite(rhs)) = &result {
                self.suspended_owner_roots.extend(rhs.exact_terms.keys().copied());
            }
        }
        self.normalization
            .as_deref_mut()
            .expect("normalization cache checked above")
            .rollback(checkpoint);
        self.watchdog_record_specialization(DiagnosticPhase::ProofRollback, |counters| {
            counters.proof_rollbacks_completed =
                counters.proof_rollbacks_completed.saturating_add(1);
        });
        result
    }

    fn specialize_universal(
        &mut self,
        dispatch: &super::relation::UniversalDispatchKey,
        index: ScopedExprId,
        index_range: super::arena::TrustedIndexRange,
        kind: SpecializationKind,
    ) -> Result<BTreeMap<CanonicalLhsKey, BTreeSet<super::relation::CanonicalRhsId>>, NormalizeError>
    {
        let suspended_checkpoint = self.suspended_owner_roots.len();
        self.watchdog_record_specialization(DiagnosticPhase::UniversalSpecialization, |counters| {
            match kind {
                SpecializationKind::Ordinary => {
                    counters.ordinary_specializations_started =
                        counters.ordinary_specializations_started.saturating_add(1);
                }
                SpecializationKind::Proof => {
                    counters.proof_specializations_started =
                        counters.proof_specializations_started.saturating_add(1);
                }
            }
        });
        let registrations = self
            .relations
            .ok_or(NormalizeError::Relation(RelationRegistryError::NotFrozen))?
            .universal_candidates(dispatch)?
            .cloned();
        let Some(registrations) = registrations else {
            self.watchdog_record_specialization(
                DiagnosticPhase::UniversalSpecialization,
                |counters| match kind {
                    SpecializationKind::Ordinary => {
                        counters.ordinary_specializations_completed =
                            counters.ordinary_specializations_completed.saturating_add(1);
                    }
                    SpecializationKind::Proof => {
                        counters.proof_specializations_completed =
                            counters.proof_specializations_completed.saturating_add(1);
                    }
                },
            );
            return Ok(BTreeMap::new());
        };
        let mut result = BTreeMap::<CanonicalLhsKey, BTreeSet<_>>::new();
        for (static_lhs, targets) in registrations {
            if !static_lhs.domain.contains(index_range) {
                self.suspended_owner_roots.truncate(suspended_checkpoint);
                return Err(NormalizeError::Relation(RelationRegistryError::IndexOutOfDomain));
            }
            for registration in targets.into_values() {
                self.trace.enter_caller_phase("specialize:next_registration", false);
                self.watchdog_record_specialization(DiagnosticPhase::Registration, |counters| {
                    counters.registrations_started =
                        counters.registrations_started.saturating_add(1);
                });
                let (lhs, rhs) =
                    match self.specialize_registration(index, index_range, &registration) {
                        Ok(value) => value,
                        Err(error) => {
                            self.suspended_owner_roots.truncate(suspended_checkpoint);
                            return Err(error);
                        }
                    };
                self.watchdog_record_specialization(DiagnosticPhase::Registration, |counters| {
                    counters.registrations_completed =
                        counters.registrations_completed.saturating_add(1);
                });
                let lhs_monomial = lhs.monomial;
                result.entry(lhs).or_default().insert(rhs);
                if self.watchdog.is_some() {
                    self.suspended_owner_roots.push(lhs_monomial);
                }
            }
        }
        self.watchdog_record_specialization(DiagnosticPhase::UniversalSpecialization, |counters| {
            match kind {
                SpecializationKind::Ordinary => {
                    counters.ordinary_specializations_completed =
                        counters.ordinary_specializations_completed.saturating_add(1);
                }
                SpecializationKind::Proof => {
                    counters.proof_specializations_completed =
                        counters.proof_specializations_completed.saturating_add(1);
                }
            }
        });
        self.suspended_owner_roots.truncate(suspended_checkpoint);
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
        let lhs_pin_checkpoint = self.suspended_owner_roots.len();
        if self.watchdog.is_some() {
            self.suspended_owner_roots.push(monomial);
        }
        let (_, target) = match self.normalize_plan(registration.target_plan, index) {
            Ok(value) => value,
            Err(error) => {
                self.suspended_owner_roots.truncate(lhs_pin_checkpoint);
                return Err(error);
            }
        };
        if self.watchdog.is_some() {
            let exact_terms = u64::try_from(target.exact_terms.len()).unwrap_or(u64::MAX);
            self.watchdog_record_specialization(DiagnosticPhase::RhsIntern, |counters| {
                counters.rhs_exact_terms_total =
                    counters.rhs_exact_terms_total.saturating_add(exact_terms);
                counters.rhs_exact_terms_max = counters.rhs_exact_terms_max.max(exact_terms);
            });
        }
        self.trace.enter_caller_phase("registration:rhs_intern_start", false);
        let track_interner_outcome = self.watchdog.is_some();
        let (rhs, inserted) = {
            let normalization = self
                .normalization
                .as_deref_mut()
                .ok_or(NormalizeError::Relation(RelationRegistryError::InvalidCanonicalRhs))?;
            let before = track_interner_outcome.then(|| normalization.canonical_rhs_count());
            let rhs = match normalization.intern_arc(target) {
                Ok(rhs) => rhs,
                Err(error) => {
                    self.suspended_owner_roots.truncate(lhs_pin_checkpoint);
                    return Err(error.into());
                }
            };
            (rhs, before.map(|before| normalization.canonical_rhs_count() > before))
        };
        if let Some(inserted) = inserted {
            self.watchdog_record_specialization(DiagnosticPhase::RhsIntern, |counters| {
                if inserted {
                    counters.interner_inserted = counters.interner_inserted.saturating_add(1);
                } else {
                    counters.interner_existing = counters.interner_existing.saturating_add(1);
                }
            });
        }
        self.trace.enter_caller_phase("registration:rhs_intern_end", false);
        self.suspended_owner_roots.truncate(lhs_pin_checkpoint);
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
        let trace_next_root = self.trace.claim_next_specialized_root();
        self.trace.enter_next_root_phase("next_root:preproof_start", trace_next_root);
        self.watchdog_update(|progress| progress.phase = DiagnosticPhase::ScopeProof);
        let proof = self.expressions.scope_proof(self.scope, root)?;
        self.watchdog_update(|progress| progress.phase = DiagnosticPhase::ScopeProofDone);
        self.trace.enter_next_root_phase("next_root:preproof_end", trace_next_root);
        let scoped = self.expressions.scoped_from_proof(&proof, root)?;
        self.trace.next_root_normalize_proof_pending = trace_next_root;
        let suspended_checkpoint = self.suspended_owner_roots.len();
        if self.watchdog.is_some() {
            self.suspended_owner_roots.extend(self.cache.values().flat_map(|value| {
                value
                    .exact_nf
                    .iter()
                    .flat_map(|normal_form| normal_form.exact_terms.keys().copied())
            }));
            self.suspended_owner_roots.extend(
                self.gadget_input_nfs
                    .values()
                    .flat_map(|normal_form| normal_form.exact_terms.keys().copied()),
            );
            self.suspended_owner_roots.extend(self.exact_plan_leaf_roots(true)?);
        }
        let saved_cache = std::mem::take(&mut self.cache);
        let saved_exact_plans = std::mem::take(&mut self.exact_plans);
        let saved_product_plans = std::mem::take(&mut self.product_plans);
        let saved_gadget_product_plans = std::mem::take(&mut self.gadget_product_plans);
        let saved_deferred_gadget_products = std::mem::take(&mut self.deferred_gadget_products);
        let saved_deferred_products = std::mem::take(&mut self.deferred_products);
        let saved_deferred_scalar_actions = std::mem::take(&mut self.deferred_scalar_actions);
        let saved_diagnostic_product_consumers =
            std::mem::take(&mut self.diagnostic_product_consumers);
        let saved_diagnostic_product_evaluations =
            std::mem::take(&mut self.diagnostic_product_evaluations);
        let saved_diagnostic_product_root = self.diagnostic_product_root.take();
        let saved_diagnostic_materialization_origins =
            std::mem::take(&mut self.diagnostic_materialization_origins);
        let saved_diagnostic_materialized_leaf_origins =
            std::mem::take(&mut self.diagnostic_materialized_leaf_origins);
        let saved_diagnostic_current_expression = self.diagnostic_current_expression.take();
        // `normalize` owns a complete root-local bounds map and clears it at entry. Keep the
        // outer map out of that nested invocation, then merge newly-derived entries back after
        // restoring it. This preserves the outer typed authority without retaining a stale
        // weaker result when the nested pass derived a stronger bound for the same expression.
        let saved_expression_bounds = std::mem::take(&mut self.expression_bounds);
        let saved_uses = std::mem::take(&mut self.remaining_uses);
        let saved_gadget_input_nfs = std::mem::take(&mut self.gadget_input_nfs);
        let saved_owner_counters = self.owner_counters;
        self.owner_counters = NormalizerOwnerCounters::default();
        let saved_counters = self.counters;
        let saved_trace_expression_slot = self.trace.current_expression_slot;
        let saved_trace_operator = self.trace.current_operator;
        let saved_fold_final_no_match = self.fold_final_no_match;
        self.fold_final_no_match = false;
        let nested_normalize_started = self.watchdog_timing_start();
        let value = self.normalize_with_existing_scope_proof(scoped, proof);
        let nested_return_origin = value
            .as_ref()
            .ok()
            .and_then(|_| self.diagnostic_materialization_origins.get(&root).copied())
            .map(|mut origin| {
                origin.reason = DiagnosticMaterializationReason::SpecializedReturn;
                origin
            });
        self.watchdog_record_timing(nested_normalize_started, |timings| {
            &mut timings.specialized_nested_normalize
        });
        let extraction_started = self.watchdog_timing_start();
        self.trace.caller_nested_bounds_len =
            u64::try_from(self.expression_bounds.len()).unwrap_or(u64::MAX);
        self.trace.caller_nested_uses_len =
            u64::try_from(self.remaining_uses.len()).unwrap_or(u64::MAX);
        self.trace.caller_nested_cache_len = u64::try_from(self.cache.len()).unwrap_or(u64::MAX);
        self.watchdog_update(|progress| progress.phase = DiagnosticPhase::CallerMerge);
        self.trace.enter_caller_phase("caller:nested_return", true);
        self.cache = saved_cache;
        self.exact_plans = saved_exact_plans;
        self.product_plans = saved_product_plans;
        self.gadget_product_plans = saved_gadget_product_plans;
        self.deferred_gadget_products = saved_deferred_gadget_products;
        self.deferred_products = saved_deferred_products;
        self.deferred_scalar_actions = saved_deferred_scalar_actions;
        self.diagnostic_product_consumers = saved_diagnostic_product_consumers;
        self.diagnostic_product_evaluations = saved_diagnostic_product_evaluations;
        self.diagnostic_product_root = saved_diagnostic_product_root;
        self.diagnostic_materialization_origins = saved_diagnostic_materialization_origins;
        self.diagnostic_materialized_leaf_origins = saved_diagnostic_materialized_leaf_origins;
        self.diagnostic_current_expression = saved_diagnostic_current_expression;
        if let Some(origin) = nested_return_origin {
            self.diagnostic_materialization_origins.insert(root, origin);
        }
        self.trace.enter_caller_phase("caller:cache_restored", false);
        let nested_expression_bounds = std::mem::take(&mut self.expression_bounds);
        self.expression_bounds = saved_expression_bounds;
        self.watchdog_record_timing(extraction_started, |timings| {
            &mut timings.specialized_extraction
        });
        self.trace.caller_outer_bounds_len =
            u64::try_from(self.expression_bounds.len()).unwrap_or(u64::MAX);
        self.trace.enter_caller_phase("caller:bounds_merge_start", true);
        let merge_bounds_started = self.watchdog_timing_start();
        if value.is_ok() {
            self.merge_expression_bounds(nested_expression_bounds);
        }
        self.watchdog_record_timing(merge_bounds_started, |timings| {
            &mut timings.specialized_merge_bounds
        });
        self.trace.caller_after_bounds_len =
            u64::try_from(self.expression_bounds.len()).unwrap_or(u64::MAX);
        self.trace.enter_caller_phase("caller:bounds_merge_end", true);
        self.trace.enter_caller_phase("caller:uses_restore_start", false);
        let state_restore_started = self.watchdog_timing_start();
        let nested_owner_counters = self.owner_counters;
        self.remaining_uses = saved_uses;
        self.trace.enter_caller_phase("caller:uses_restore_end", false);
        self.gadget_input_nfs = saved_gadget_input_nfs;
        self.owner_counters =
            Self::restored_owner_counters(saved_owner_counters, nested_owner_counters);
        self.counters = saved_counters;
        self.trace.current_expression_slot = saved_trace_expression_slot;
        self.trace.current_operator = saved_trace_operator;
        self.fold_final_no_match = saved_fold_final_no_match;
        self.suspended_owner_roots.truncate(suspended_checkpoint);
        self.watchdog_record_timing(state_restore_started, |timings| {
            &mut timings.specialized_state_restore
        });
        self.refresh_owner_diagnostics();
        self.trace.enter_caller_phase("caller:state_restored", false);
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
    ) -> Result<(ExprId, Arc<PolynomialNF>), NormalizeError> {
        let root = self.programs.beta_reduce(self.expressions, plan, &[index.expression()])?;
        let value = self.normalize_specialized_root(root)?;
        let normal_form =
            value.exact_nf.clone().ok_or_else(|| NormalizeError::UnsupportedOperator {
                operator: "relation plan without exact normal form".into(),
            })?;
        self.trace.enter_caller_phase("plan:nf_arc_cloned", false);
        Ok((root, normal_form))
    }

    fn compute_use_counts(&mut self, root: ExprId) -> Result<BTreeSet<ExprId>, NormalizeError> {
        let mut reachable = BTreeSet::new();
        let collect_product_diagnostics = self.watchdog.is_some();
        let mut product_consumers = BTreeMap::<ExprId, DiagnosticProductConsumerCounts>::new();
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
                let counts = product_consumers.entry(*child).or_default();
                match node.operator {
                    ValueOperator::Matrix(MatrixOperation::Add | MatrixOperation::Subtract) => {
                        counts.add_sub = counts.add_sub.saturating_add(1)
                    }
                    ValueOperator::Matrix(MatrixOperation::Multiply) => {
                        counts.multiply = counts.multiply.saturating_add(1)
                    }
                    _ => counts.root_other = counts.root_other.saturating_add(1),
                }
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
                            let counts = product_consumers.entry(*component).or_default();
                            counts.structural = counts.structural.saturating_add(1);
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
                        let counts = product_consumers.entry(held).or_default();
                        counts.structural = counts.structural.saturating_add(1);
                    }
                }
            }
            if matches!(
                node.operator,
                ValueOperator::Matrix(MatrixOperation::LiftConstantPolynomial { .. })
            ) {
                if let Some(source) = self.lift_extraction_source(expression, &node)? {
                    *self.remaining_uses.entry(source).or_default() += 1;
                    let counts = product_consumers.entry(source).or_default();
                    counts.structural = counts.structural.saturating_add(1);
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
                            let counts = product_consumers.entry(*grandchild).or_default();
                            counts.structural = counts.structural.saturating_add(1);
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
                    let counts = product_consumers.entry(*input).or_default();
                    counts.structural = counts.structural.saturating_add(1);
                }
            }
        }
        *self.remaining_uses.entry(root).or_default() += 1;
        let counts = product_consumers.entry(root).or_default();
        counts.root_other = counts.root_other.saturating_add(1);

        let mut candidates = BTreeSet::new();
        let mut gadget_candidates = BTreeSet::new();
        let mut scalar_action_candidates = BTreeSet::new();
        for expression in &reachable {
            if *expression == root ||
                product_consumers.get(expression).is_some_and(|counts| counts.structural != 0)
            {
                continue;
            }
            let node = self.expressions.node(*expression)?;
            if !matches!(node.operator, ValueOperator::Matrix(MatrixOperation::Multiply)) ||
                node.inputs.len() != 2
            {
                continue;
            }
            let scalar_sides = node.inputs.iter().try_fold(0_u8, |count, input| {
                let ResolvedValueType::Matrix(matrix) = self.expressions.value_type(*input)? else {
                    return Ok::<_, NormalizeError>(u8::MAX);
                };
                Ok(count.saturating_add(u8::from(matrix.rows == 1 && matrix.columns == 1)))
            })?;
            if scalar_sides == 1 {
                scalar_action_candidates.insert(*expression);
            } else if scalar_sides == 0 {
                gadget_candidates.insert(*expression);
                // A direct decomposition operand is the existing typed gadget boundary. Keep
                // it out of the ordinary policy so the specialized plan remains authoritative.
                if !node.inputs.iter().any(|input| {
                    self.expressions.node(*input).is_ok_and(|input| {
                        matches!(
                            input.operator,
                            ValueOperator::Transform(
                                ValueTransformOperation::GadgetDecompose { .. }
                            )
                        )
                    })
                }) {
                    candidates.insert(*expression);
                }
            }
        }
        let (eligible, _) = self.propagate_product_eligibility(&candidates, &real_consumers)?;
        self.deferred_products = eligible;
        self.deferred_scalar_actions = scalar_action_candidates
            .into_iter()
            .filter(|expression| {
                real_consumers.get(expression).is_some_and(|consumers| {
                    !consumers.is_empty() &&
                        consumers.iter().all(|consumer| {
                            self.expressions.node(*consumer).is_ok_and(|node| {
                                matches!(
                                    node.operator,
                                    ValueOperator::Matrix(
                                        MatrixOperation::Add | MatrixOperation::Subtract
                                    )
                                )
                            })
                        })
                })
            })
            .collect();
        if self.gadget_recompositions.is_some() {
            self.deferred_gadget_products = gadget_candidates
                .iter()
                .filter(|expression| {
                    real_consumers.get(expression).is_some_and(|consumers| {
                        !consumers.is_empty() &&
                            consumers.iter().all(|consumer| {
                                self.expressions.node(*consumer).is_ok_and(|node| {
                                    matches!(
                                        node.operator,
                                        ValueOperator::Matrix(
                                            MatrixOperation::Add | MatrixOperation::Subtract
                                        )
                                    ) || self.deferred_products.contains(consumer)
                                })
                            })
                    })
                })
                .copied()
                .collect();
        }
        if collect_product_diagnostics {
            self.diagnostic_product_consumers = product_consumers;
            self.diagnostic_product_root = Some(root);
        }
        Ok(reachable)
    }

    fn propagate_product_eligibility(
        &self,
        candidates: &BTreeSet<ExprId>,
        real_consumers: &BTreeMap<ExprId, BTreeSet<ExprId>>,
    ) -> Result<(BTreeSet<ExprId>, ProductEligibilityStats), NormalizeError> {
        let mut stats = ProductEligibilityStats {
            candidates: candidates.len(),
            ..ProductEligibilityStats::default()
        };
        let mut pending_candidate_consumers = BTreeMap::<ExprId, usize>::new();
        let mut predecessors = BTreeMap::<ExprId, Vec<ExprId>>::new();
        let mut queue = VecDeque::new();
        for candidate in candidates {
            let Some(consumers) = real_consumers.get(candidate) else { continue };
            if consumers.is_empty() {
                continue;
            }
            let mut blocked = false;
            let mut pending = 0_usize;
            for consumer in consumers {
                stats.consumer_edges = stats
                    .consumer_edges
                    .checked_add(1)
                    .ok_or(NormalizeError::ArithmeticOverflow)?;
                if candidates.contains(consumer) {
                    pending = pending.checked_add(1).ok_or(NormalizeError::ArithmeticOverflow)?;
                    predecessors.entry(*consumer).or_default().push(*candidate);
                    continue;
                }
                let node = self.expressions.node(*consumer)?;
                if !matches!(
                    node.operator,
                    ValueOperator::Matrix(MatrixOperation::Add | MatrixOperation::Subtract)
                ) {
                    blocked = true;
                }
            }
            if !blocked {
                pending_candidate_consumers.insert(*candidate, pending);
                if pending == 0 {
                    queue.push_back(*candidate);
                }
            }
        }
        let mut eligible = BTreeSet::new();
        while let Some(candidate) = queue.pop_front() {
            stats.queue_pops =
                stats.queue_pops.checked_add(1).ok_or(NormalizeError::ArithmeticOverflow)?;
            if !eligible.insert(candidate) {
                continue;
            }
            for predecessor in predecessors.get(&candidate).into_iter().flatten() {
                let Some(remaining) = pending_candidate_consumers.get_mut(predecessor) else {
                    continue;
                };
                *remaining = remaining.checked_sub(1).ok_or(NormalizeError::InvalidExactPlan {
                    reason: "product eligibility dependency underflow",
                })?;
                if *remaining == 0 {
                    queue.push_back(*predecessor);
                }
            }
        }
        Ok((eligible, stats))
    }

    fn child_value_with_exact(
        &mut self,
        expression: ExprId,
    ) -> Result<(Arc<AnalyzedValue>, Option<NodeExactState>), NormalizeError> {
        let value = self
            .cache
            .get(&expression)
            .cloned()
            .ok_or(NormalizeError::MissingCachedValue { expression })?;
        let exact = self.cached_exact_state(expression, value.as_ref())?;
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
        Ok((value, exact))
    }

    fn child_value(&mut self, expression: ExprId) -> Result<Arc<AnalyzedValue>, NormalizeError> {
        let (value, exact) = self.child_value_with_exact(expression)?;
        let Some(
            exact @ (NodeExactState::Additive(_) |
            NodeExactState::Product(_) |
            NodeExactState::GadgetProduct(_)),
        ) = exact
        else {
            return Ok(value);
        };
        let materialized = self.materialize_exact_state_for(
            &exact,
            expression,
            DiagnosticMaterializationReason::NonAddConsumer,
            self.diagnostic_current_expression,
        )?;
        Ok(Arc::new(synchronize_materialized_value(value.as_ref().clone(), materialized)))
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
        self.insert_gadget_hold(expression, normal_form.clone());
        Ok(Some(normal_form))
    }

    fn evaluate_node(
        &mut self,
        scope_proof: &mut ScopeProof,
        expression: ExprId,
        node: &ExprNode,
    ) -> Result<AnalyzedValue, NormalizeError> {
        if self.watchdog.is_some() {
            self.diagnostic_current_expression = Some(expression);
        }
        // `normalize` validates the complete root once. Every expression reaching this point was
        // discovered below that validated root, so rebuilding the scoped view is an O(1) checked
        // projection. Calling `ProgramArena::scoped` here would walk the remaining sub-DAG once
        // per node and turn a linear chain into O(N^2).
        let semantic = self.expressions.scoped_from_proof(scope_proof, expression)?;
        let additive = matches!(
            node.operator,
            ValueOperator::Matrix(MatrixOperation::Add | MatrixOperation::Subtract)
        );
        let compositional_product =
            matches!(node.operator, ValueOperator::Matrix(MatrixOperation::Multiply)) &&
                (self.deferred_products.contains(&expression) ||
                    self.deferred_gadget_products.contains(&expression) ||
                    self.deferred_scalar_actions.contains(&expression));
        let mut children = Vec::with_capacity(node.inputs.len());
        let mut child_exact = Vec::with_capacity(node.inputs.len());
        for child in &node.inputs {
            if additive || compositional_product {
                let (value, exact) = self.child_value_with_exact(*child)?;
                children.push(value);
                child_exact.push(exact);
            } else {
                children.push(self.child_value(*child)?);
            }
        }
        if self.watchdog.is_some() &&
            matches!(node.operator, ValueOperator::Matrix(MatrixOperation::Multiply))
        {
            let snapshot = self.diagnostic_product_evaluation_snapshot(node, &children);
            self.diagnostic_product_evaluations.insert(expression, snapshot);
        }
        let output_type = self.expressions.value_type(expression)?.clone();
        let mut value = if additive && matches!(output_type, ResolvedValueType::Matrix(_)) {
            let bound = self.matrix_bound(expression, node, &children)?;
            match (
                child_exact.first().and_then(Clone::clone),
                child_exact.get(1).and_then(Clone::clone),
            ) {
                (Some(left), Some(right)) => {
                    let plan = self.new_additive_plan(
                        expression,
                        left,
                        right,
                        matches!(node.operator, ValueOperator::Matrix(MatrixOperation::Subtract)),
                    )?;
                    if self.relations.is_some() && self.relation_rewriting_enabled {
                        let normal_form = self.materialize_exact_state_for(
                            &NodeExactState::Additive(plan),
                            expression,
                            DiagnosticMaterializationReason::RelationBoundary,
                            Some(expression),
                        )?;
                        synchronize_materialized_value(
                            AnalyzedValue { semantic, exact_nf: None, coefficient_bound: bound },
                            normal_form,
                        )
                    } else {
                        self.exact_plans.insert(expression, plan);
                        AnalyzedValue { semantic, exact_nf: None, coefficient_bound: bound }
                    }
                }
                _ => self.evaluate_matrix(scope_proof, semantic, expression, node, &children)?,
            }
        } else if matches!(output_type, ResolvedValueType::Matrix(_)) {
            let mut deferred_gadget = None;
            let mut deferred_product = None;
            if compositional_product && child_exact.len() == 2 {
                let left = child_exact[0].clone();
                let right = child_exact[1].clone();
                if let (
                    Some(NodeExactState::Materialized { normal_form: left, .. }),
                    Some(NodeExactState::Materialized { normal_form: right, .. }),
                ) = (&left, &right)
                {
                    deferred_gadget = self.new_gadget_product_plan(
                        expression,
                        node,
                        left.clone(),
                        right.clone(),
                    )?;
                }
                if deferred_gadget.is_none() &&
                    let (Some(left), Some(right)) = (left, right)
                {
                    deferred_product =
                        self.new_product_plan(scope_proof, expression, node, left, right)?;
                }
            }
            if let Some(plan) = deferred_gadget {
                let bound = self.matrix_bound(expression, node, &children)?;
                self.gadget_product_plans.insert(expression, plan);
                AnalyzedValue { semantic, exact_nf: None, coefficient_bound: bound }
            } else if let Some(plan) = deferred_product {
                let bound = self.matrix_bound(expression, node, &children)?;
                self.product_plans.insert(expression, plan);
                AnalyzedValue { semantic, exact_nf: None, coefficient_bound: bound }
            } else {
                for (index, exact) in child_exact.into_iter().enumerate() {
                    let Some(exact) = exact else { continue };
                    if matches!(exact, NodeExactState::Materialized { .. }) {
                        continue;
                    }
                    let child = node.inputs[index];
                    let materialized = self.materialize_exact_state_for(
                        &exact,
                        child,
                        DiagnosticMaterializationReason::NonAddConsumer,
                        Some(expression),
                    )?;
                    children[index] = Arc::new(synchronize_materialized_value(
                        children[index].as_ref().clone(),
                        materialized,
                    ));
                }
                self.evaluate_matrix(scope_proof, semantic, expression, node, &children)?
            }
        } else {
            self.evaluate_nonmatrix(semantic, expression, node, &children)?
        };
        self.trace.enter_subphase("evaluate_node:relation_rewrite");
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
        }
        self.trace.enter_subphase("evaluate_node:zero_check");
        if let Some(normal_form) = value.exact_nf.as_mut().and_then(Arc::get_mut) {
            if normal_form.is_zero() {
                value.coefficient_bound = NumericContract::Known(CoefficientBound::ExactZero);
                normal_form.bounded_summary = BoundedSummary::known(CoefficientBound::ExactZero);
            }
        }
        self.trace.enter_subphase("evaluate_node:complete");
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
        self.trace.enter_subphase("evaluate_matrix:bound");
        let bound = self.matrix_bound(expression, node, children)?;
        self.trace.enter_subphase("evaluate_matrix:exact");
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
                            self.product_nf(scope_proof, left, right)
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
                            self.tensor_nf(scope_proof, operation, left, right)
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
        self.trace.record_scalar_call();
        let ResolvedValueType::Matrix(output_type) =
            self.expressions.value_type(output_expression)?
        else {
            self.trace.record_scalar_not_applicable();
            return Ok(ScalarActionNormalization::NotApplicable);
        };
        let output_type = output_type.clone();
        let ResolvedValueType::Matrix(left_type) = self.expressions.value_type(left_expression)?
        else {
            self.trace.record_scalar_not_applicable();
            return Ok(ScalarActionNormalization::NotApplicable);
        };
        let left_type = left_type.clone();
        let ResolvedValueType::Matrix(right_type) =
            self.expressions.value_type(right_expression)?
        else {
            self.trace.record_scalar_not_applicable();
            return Ok(ScalarActionNormalization::NotApplicable);
        };
        let right_type = right_type.clone();
        let left_scalar = left_type.rows == 1 && left_type.columns == 1;
        let right_scalar = right_type.rows == 1 && right_type.columns == 1;
        if !left_scalar && !right_scalar {
            self.trace.record_scalar_not_applicable();
            return Ok(ScalarActionNormalization::NotApplicable);
        }
        let expected_output = if left_scalar { &right_type } else { &left_type };
        if left_type.modulus != right_type.modulus ||
            left_type.ring_dimension != right_type.ring_dimension ||
            &output_type != expected_output
        {
            self.trace.record_scalar_opaque();
            return Ok(ScalarActionNormalization::Opaque);
        }

        if (left_scalar && !self.scalar_nf_ordered_factors_match_type(left, &left_type)?) ||
            (right_scalar && !self.scalar_nf_ordered_factors_match_type(right, &right_type)?)
        {
            self.trace.record_scalar_opaque();
            return Ok(ScalarActionNormalization::Opaque);
        }

        let ordered_scalar_product = if left_scalar && right_scalar {
            // Preserve ordered exact relations such as G * Decompose(A) = A before using the
            // commutativity of the scalar result. The product is centralized only after every
            // surviving ordered factor is proven to have the declared 1x1 output type. In the
            // reversed order no relation applies, so both typed scalar factors remain present.
            let product = self.product_nf(scope_proof, left, right)?;
            if !self.scalar_nf_ordered_factors_match_type(&product, &output_type)? {
                self.trace.record_scalar_opaque();
                return Ok(ScalarActionNormalization::Opaque);
            }
            self.trace.record_scalar_both();
            Some(product)
        } else {
            if left_scalar {
                self.trace.record_scalar_left();
            } else {
                self.trace.record_scalar_right();
            }
            None
        };

        if let Some(ordered_product) = ordered_scalar_product {
            let (reclassified, _, _) = self.centralize_scalar_nf(scope_proof, &ordered_product)?;
            return Ok(ScalarActionNormalization::Exact(reclassified));
        }

        let reclassified_left = if left_scalar {
            self.centralize_scalar_nf(scope_proof, left)?.0
        } else {
            left.clone()
        };
        let reclassified_right = if right_scalar {
            self.centralize_scalar_nf(scope_proof, right)?.0
        } else {
            right.clone()
        };
        Ok(ScalarActionNormalization::Exact(self.product_nf(
            scope_proof,
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
    ) -> Result<(PolynomialNF, usize, usize), NormalizeError> {
        let mut reclassified_terms = BTreeMap::new();
        let mut max_reclassified_factors = 0usize;
        for (monomial, coefficient) in &normal_form.exact_terms {
            if coefficient.is_zero() {
                continue;
            }
            let (mut central, ordered) = {
                let descriptor = self.monomials.descriptor(*monomial)?;
                (descriptor.central_factors.to_vec(), descriptor.ordered_factors.to_vec())
            };
            central.extend_from_slice(&ordered);
            max_reclassified_factors = max_reclassified_factors.max(central.len());
            let reclassified = self.monomials.intern_with_proof(
                self.expressions,
                self.programs,
                scope_proof,
                &central,
                &[],
            )?;
            merge_term(&mut reclassified_terms, reclassified, coefficient.clone());
        }
        self.trace
            .record_scalar_reclassification(reclassified_terms.len(), max_reclassified_factors);
        let term_count = reclassified_terms.len();
        Ok((
            PolynomialNF {
                exact_terms: reclassified_terms,
                bounded_summary: BoundedSummary::missing(),
            },
            term_count,
            max_reclassified_factors,
        ))
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

    fn atom_nf(
        &mut self,
        scope_proof: &ScopeProof,
        semantic: ScopedExprId,
    ) -> Result<PolynomialNF, NormalizeError> {
        self.trace.enter_subphase("atom:scope_validate");
        self.expressions.validate_scoped_from_proof(scope_proof, semantic)?;
        self.trace.enter_subphase("atom:monomial_intern");
        let id = self.atom_monomial(Some(scope_proof), semantic)?;
        self.trace.enter_subphase("atom:term_insert");
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
        _scope_proof: &ScopeProof,
        left: &PolynomialNF,
        right: &PolynomialNF,
    ) -> Result<PolynomialNF, NormalizeError> {
        let mut terms = BTreeMap::new();
        self.execute_product_into(left, right, &BigInt::from(1_u8), &mut terms, false, false)?;
        Ok(PolynomialNF { exact_terms: terms, bounded_summary: BoundedSummary::missing() })
    }

    /// Execute one complete product lifecycle. Both eager products and deferred weighted gadget
    /// products pass through this wrapper, so per-call progress never inherits a preceding
    /// product's queue, generation, or large-product sampling state.
    fn execute_product_into(
        &mut self,
        left: &PolynomialNF,
        right: &PolynomialNF,
        weight: &BigInt,
        terms: &mut BTreeMap<MonomialId, BigInt>,
        direct_gadget_boundary: bool,
        typed_product_plan: bool,
    ) -> Result<(), NormalizeError> {
        if self.watchdog.is_some() {
            self.watchdog_product_call_id = self.watchdog_product_call_id.saturating_add(1);
            self.watchdog_product_processed_current = 0;
            self.watchdog_product_planned_current = 0;
            self.watchdog_product_generation_current = 0;
            self.watchdog_product_enqueued_current = 0;
            self.watchdog_product_queue_current = 0;
            self.watchdog_product_output_current = 0;
        }
        let planned = u64::try_from(left.exact_terms.len())
            .unwrap_or(u64::MAX)
            .saturating_mul(u64::try_from(right.exact_terms.len()).unwrap_or(u64::MAX));
        if self.watchdog.is_some() {
            self.watchdog_product_planned_current = planned;
            self.watchdog_update(|progress| {
                progress.product_processed_current = 0;
                progress.product_planned_current = planned;
                progress.product_generation_current = 0;
                progress.product_enqueued_current = 0;
                progress.product_queue_current = 0;
                progress.product_output_current = 0;
            });
        }
        self.current_large_product_sampled = false;
        let retained = u64::try_from(self.monomials.len()).unwrap_or(u64::MAX);
        if self.admits_retained_arena_census(retained) {
            self.sample_owner_census(
                OwnerCensusReason::RetainedArenaMilestone,
                left.exact_terms.keys().chain(right.exact_terms.keys()).copied(),
            );
        }
        if self.trace.active {
            self.trace.product_calls = self.trace.product_calls.saturating_add(1);
            self.trace.product_planned = self.trace.product_planned.saturating_add(planned);
            self.trace.product_max_left_terms = self
                .trace
                .product_max_left_terms
                .max(u64::try_from(left.exact_terms.len()).unwrap_or(u64::MAX));
            self.trace.product_max_right_terms = self
                .trace
                .product_max_right_terms
                .max(u64::try_from(right.exact_terms.len()).unwrap_or(u64::MAX));
        }
        let trace_large_product = self.trace.active && planned >= LARGE_PRODUCT_PLANNED_PAIRS;
        self.trace.product_current_queue = 0;
        self.trace.product_current_output = 0;
        if trace_large_product {
            self.trace.emit(
                "product_start",
                self.trace.nodes_processed,
                self.trace.nodes_total,
                false,
            );
        }
        if direct_gadget_boundary && typed_product_plan {
            self.product_plan_counters.typed_direct_executions =
                self.product_plan_counters.typed_direct_executions.saturating_add(1);
        } else if direct_gadget_boundary {
            self.gadget_product_counters.streamed_executions =
                self.gadget_product_counters.streamed_executions.saturating_add(1);
            self.gadget_product_counters.planned_pairs =
                self.gadget_product_counters.planned_pairs.saturating_add(planned);
        }
        let result = self.product_into_body(
            left,
            right,
            weight,
            terms,
            direct_gadget_boundary,
            typed_product_plan,
        );
        if self.trace.active {
            if result.is_ok() {
                self.trace.product_max_output_terms = self
                    .trace
                    .product_max_output_terms
                    .max(u64::try_from(terms.len()).unwrap_or(u64::MAX));
            }
        }
        if direct_gadget_boundary && !typed_product_plan && result.is_ok() {
            self.gadget_product_counters.max_streamed_output_terms = self
                .gadget_product_counters
                .max_streamed_output_terms
                .max(u64::try_from(terms.len()).unwrap_or(u64::MAX));
        }
        if trace_large_product {
            self.trace.emit(
                "product_end",
                self.trace.nodes_processed,
                self.trace.nodes_total,
                false,
            );
        }
        result
    }

    fn product_into_body(
        &mut self,
        left: &PolynomialNF,
        right: &PolynomialNF,
        weight: &BigInt,
        terms: &mut BTreeMap<MonomialId, BigInt>,
        direct_gadget_boundary: bool,
        typed_product_plan: bool,
    ) -> Result<(), NormalizeError> {
        let mut worklist = VecDeque::new();
        for (left_id, left_coefficient) in &left.exact_terms {
            for (right_id, right_coefficient) in &right.exact_terms {
                let coefficient = left_coefficient * right_coefficient * weight;
                if coefficient.is_zero() {
                    self.watchdog_record_product_generated(
                        false,
                        u64::try_from(worklist.len()).unwrap_or(u64::MAX),
                        left.exact_terms
                            .keys()
                            .chain(right.exact_terms.keys())
                            .chain(worklist.iter().map(|(id, _)| id))
                            .chain(terms.keys())
                            .copied(),
                    );
                    continue;
                }
                let direct_rewrite = if direct_gadget_boundary {
                    if typed_product_plan {
                        self.product_plan_counters.typed_pair_attempts =
                            self.product_plan_counters.typed_pair_attempts.saturating_add(1);
                    }
                    let rewritten = self.rewrite_gadget_product_pair(*left_id, *right_id)?;
                    if typed_product_plan {
                        if rewritten.is_some() {
                            self.product_plan_counters.typed_pair_matches =
                                self.product_plan_counters.typed_pair_matches.saturating_add(1);
                        } else {
                            self.product_plan_counters.typed_pair_ordinary_fallbacks = self
                                .product_plan_counters
                                .typed_pair_ordinary_fallbacks
                                .saturating_add(1);
                        }
                    }
                    rewritten
                } else {
                    None
                };
                if self.trace.active {
                    self.trace.product_generated = self.trace.product_generated.saturating_add(1);
                    if self.trace.product_generated >= self.trace.next_product_generated_heartbeat {
                        self.trace.last_product_heartbeat_operator = self.trace.current_operator;
                        self.trace.product_heartbeat_saw_matrix_multiply |=
                            self.trace.current_operator == "multiply";
                        self.refresh_owner_diagnostics();
                        self.trace.emit(
                            "product_generated_heartbeat",
                            self.trace.nodes_processed,
                            self.trace.nodes_total,
                            false,
                        );
                        self.trace.next_product_generated_heartbeat = self
                            .trace
                            .next_product_generated_heartbeat
                            .saturating_add(self.trace.product_heartbeat_interval);
                    }
                }
                if let Some(rewritten) = direct_rewrite {
                    if self.trace.active {
                        self.trace.product_rewrites = self.trace.product_rewrites.saturating_add(1);
                    }
                    for (rewritten_monomial, rewritten_coefficient) in rewritten.into_iter().rev() {
                        worklist.push_front((
                            rewritten_monomial,
                            coefficient.clone() * rewritten_coefficient,
                        ));
                    }
                } else {
                    let product = self.product_monomials(*left_id, *right_id)?;
                    worklist.push_back((product, coefficient));
                }
                self.watchdog_record_product_generated(
                    true,
                    u64::try_from(worklist.len()).unwrap_or(u64::MAX),
                    left.exact_terms
                        .keys()
                        .chain(right.exact_terms.keys())
                        .chain(worklist.iter().map(|(id, _)| id))
                        .chain(terms.keys())
                        .copied(),
                );
                if self.trace.active {
                    self.trace.product_current_queue =
                        u64::try_from(worklist.len()).unwrap_or(u64::MAX);
                    self.trace.product_enqueued = self.trace.product_enqueued.saturating_add(1);
                    self.trace.product_peak_queue = self
                        .trace
                        .product_peak_queue
                        .max(u64::try_from(worklist.len()).unwrap_or(u64::MAX));
                }
                // Drain each completed cartesian pair before generating the next one. The same
                // rewrite queue remains authoritative, but its live size now follows one pair's
                // recursive splice instead of the full product cardinality.
                self.drain_product_worklist(left, right, terms, &mut worklist)?;
            }
        }
        if self.watchdog.is_some() {
            let generated = self.watchdog_product_generated;
            let enqueued = self.watchdog_product_enqueued;
            let generation_current = self.watchdog_product_generation_current;
            let enqueued_current = self.watchdog_product_enqueued_current;
            let queue_current = u64::try_from(worklist.len()).unwrap_or(u64::MAX);
            let output_current = u64::try_from(terms.len()).unwrap_or(u64::MAX);
            self.watchdog_update(|progress| {
                progress.phase = DiagnosticPhase::ProductGenerationEnd;
                progress.product_generated = generated;
                progress.product_enqueued = enqueued;
                progress.product_generation_current = generation_current;
                progress.product_enqueued_current = enqueued_current;
                progress.product_queue_current = queue_current;
                progress.product_output_current = output_current;
            });
            // Up to five record-breaking large products are sampled as complete GenEnd/End
            // pairs. This keeps the latest OOM-dominant product useful while reserving one final
            // terminal/error sample under the global cap of twelve.
            if self.admits_large_product_census_pair(generation_current) {
                self.current_large_product_sampled = true;
                self.largest_sampled_product_planned = generation_current;
                self.sample_owner_census(
                    OwnerCensusReason::LargeProductGenerationEnd,
                    left.exact_terms
                        .keys()
                        .chain(right.exact_terms.keys())
                        .chain(worklist.iter().map(|(id, _)| id))
                        .chain(terms.keys())
                        .copied(),
                );
            }
        }
        self.drain_product_worklist(left, right, terms, &mut worklist)?;
        if self.watchdog.is_some() {
            self.watchdog_product_queue_current = 0;
            self.watchdog_product_output_current = u64::try_from(terms.len()).unwrap_or(u64::MAX);
            let output_current = self.watchdog_product_output_current;
            self.watchdog_update(|progress| {
                progress.phase = DiagnosticPhase::ProductEnd;
                progress.product_queue_current = 0;
                progress.product_output_current = output_current;
            });
            if self.current_large_product_sampled {
                self.sample_owner_census(
                    OwnerCensusReason::LargeProductEnd,
                    left.exact_terms
                        .keys()
                        .chain(right.exact_terms.keys())
                        .chain(terms.keys())
                        .copied(),
                );
                self.large_product_pairs_sampled =
                    self.large_product_pairs_sampled.saturating_add(1);
                self.current_large_product_sampled = false;
            }
        }
        if self.trace.active {
            self.trace.product_current_queue = 0;
            self.trace.product_current_output = u64::try_from(terms.len()).unwrap_or(u64::MAX);
        }
        Ok(())
    }

    fn drain_product_worklist(
        &mut self,
        left: &PolynomialNF,
        right: &PolynomialNF,
        terms: &mut BTreeMap<MonomialId, BigInt>,
        worklist: &mut VecDeque<(MonomialId, BigInt)>,
    ) -> Result<(), NormalizeError> {
        while let Some((monomial, coefficient)) = worklist.pop_front() {
            let queue_current = u64::try_from(worklist.len()).unwrap_or(u64::MAX);
            let output_current = u64::try_from(terms.len()).unwrap_or(u64::MAX);
            self.watchdog_record_product_processed(
                queue_current,
                output_current,
                std::iter::once(monomial)
                    .chain(left.exact_terms.keys().copied())
                    .chain(right.exact_terms.keys().copied())
                    .chain(worklist.iter().map(|(id, _)| *id))
                    .chain(terms.keys().copied()),
            );
            if self.trace.active {
                self.trace.product_current_queue = queue_current;
                self.trace.product_current_output = output_current;
                self.trace.product_processed = self.trace.product_processed.saturating_add(1);
                if self.trace.product_processed >= self.trace.next_product_processed_heartbeat {
                    self.trace.last_product_heartbeat_operator = self.trace.current_operator;
                    self.trace.product_heartbeat_saw_matrix_multiply |=
                        self.trace.current_operator == "multiply";
                    self.refresh_owner_diagnostics();
                    self.trace.emit(
                        "product_processed_heartbeat",
                        self.trace.nodes_processed,
                        self.trace.nodes_total,
                        false,
                    );
                    self.trace.next_product_processed_heartbeat = self
                        .trace
                        .next_product_processed_heartbeat
                        .saturating_add(self.trace.product_heartbeat_interval);
                }
            }
            if coefficient.is_zero() {
                continue;
            }
            let Some(rewritten) = self.rewrite_gadget_decomposition(monomial)? else {
                merge_term(terms, monomial, coefficient);
                if self.trace.active {
                    self.trace.product_current_output =
                        u64::try_from(terms.len()).unwrap_or(u64::MAX);
                }
                continue;
            };
            if self.trace.active {
                self.trace.product_rewrites = self.trace.product_rewrites.saturating_add(1);
            }
            // Process every newly spliced NF term through the same deterministic queue. This
            // closes multiple adjacent gadget/decomposition pairs without ever reifying `A` as
            // an opaque raw expression factor.
            for (rewritten_monomial, rewritten_coefficient) in
                rewritten.exact_terms.into_iter().rev()
            {
                worklist
                    .push_front((rewritten_monomial, coefficient.clone() * rewritten_coefficient));
                if self.trace.active {
                    self.trace.product_enqueued = self.trace.product_enqueued.saturating_add(1);
                    self.trace.product_peak_queue = self
                        .trace
                        .product_peak_queue
                        .max(u64::try_from(worklist.len()).unwrap_or(u64::MAX));
                    self.trace.product_current_queue =
                        u64::try_from(worklist.len()).unwrap_or(u64::MAX);
                }
            }
        }
        Ok(())
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
            let Some(terms) =
                self.splice_gadget_decomposition(&central_factors, &ordered_factors, index, input)?
            else {
                return Ok(None);
            };
            return Ok(Some(PolynomialNF {
                exact_terms: terms,
                bounded_summary: BoundedSummary::missing(),
            }));
        }
        Ok(None)
    }

    /// Rewrite a gadget/decomposition pair which lies exactly across a product boundary without
    /// first interning the transient concatenated descriptor. This is the same typed splice used
    /// by `rewrite_gadget_decomposition`; non-matching pairs retain the ordinary product path.
    fn rewrite_gadget_product_pair(
        &mut self,
        left: MonomialId,
        right: MonomialId,
    ) -> Result<Option<BTreeMap<MonomialId, BigInt>>, NormalizeError> {
        let (mut central, mut ordered, boundary) = {
            let left = self.monomials.descriptor(left)?;
            let right = self.monomials.descriptor(right)?;
            let Some(gadget) = left.ordered_factors.last().copied() else {
                return Ok(None);
            };
            let Some(decomposition) = right.ordered_factors.first().copied() else {
                return Ok(None);
            };
            let Some(input) = self.authorized_gadget_pair_input(gadget, decomposition)? else {
                return Ok(None);
            };
            let central_len = left
                .central_factors
                .len()
                .checked_add(right.central_factors.len())
                .ok_or(MonomialError::ArenaExhausted)?;
            let mut central = Vec::new();
            central.try_reserve_exact(central_len).map_err(|_| MonomialError::ArenaExhausted)?;
            central.extend_from_slice(&left.central_factors);
            central.extend_from_slice(&right.central_factors);
            central.sort_unstable();
            let ordered_len = left
                .ordered_factors
                .len()
                .checked_add(right.ordered_factors.len())
                .ok_or(MonomialError::ArenaExhausted)?;
            let mut ordered = Vec::new();
            ordered.try_reserve_exact(ordered_len).map_err(|_| MonomialError::ArenaExhausted)?;
            ordered.extend_from_slice(&left.ordered_factors);
            ordered.extend_from_slice(&right.ordered_factors);
            (central, ordered, (left.ordered_factors.len() - 1, input))
        };
        self.splice_gadget_decomposition(&mut central, &mut ordered, boundary.0, boundary.1)
    }

    fn splice_gadget_decomposition(
        &mut self,
        central_factors: &[ScopedExprId],
        ordered_factors: &[ScopedExprId],
        index: usize,
        input: ExprId,
    ) -> Result<Option<BTreeMap<MonomialId, BigInt>>, NormalizeError> {
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
        for (input_monomial, input_coefficient) in &input_nf.exact_terms {
            let mut replacement = *input_monomial;
            if let Some(left) = left {
                replacement = self.monomials.combine_interned(self.scope, left, replacement)?;
            }
            if let Some(suffix) = suffix {
                replacement = self.monomials.combine_interned(self.scope, replacement, suffix)?;
            }
            merge_term(&mut terms, replacement, input_coefficient.clone());
        }
        Ok(Some(terms))
    }

    fn authorized_gadget_pair_input(
        &self,
        gadget: ScopedExprId,
        decomposition: ScopedExprId,
    ) -> Result<Option<ExprId>, NormalizeError> {
        let decomposition_node = self.expressions.node(decomposition.expression())?;
        let ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
            base,
            small,
            digit_count,
            ..
        }) = &decomposition_node.operator
        else {
            return Ok(None);
        };
        let gadget_node = self.expressions.node(gadget.expression())?;
        let Some(super::arena::MatrixConstantKind::Gadget {
            base: gadget_base,
            small: gadget_small,
        }) = gadget_node.operator.source_matrix_constant()
        else {
            return Ok(None);
        };
        if gadget_base != base || gadget_small != small {
            return Ok(None);
        }
        let Some(input) = decomposition_node.inputs.first().copied() else {
            return Ok(None);
        };
        let matrix_type = |expression| -> Result<Option<&ResolvedMatrixType>, NormalizeError> {
            Ok(match self.expressions.value_type(expression)? {
                ResolvedValueType::Matrix(matrix) => Some(matrix),
                _ => None,
            })
        };
        let (Some(input_type), Some(gadget_type), Some(decomposition_type), Some(output_type)) = (
            matrix_type(input)?,
            matrix_type(gadget.expression())?,
            matrix_type(decomposition.expression())?,
            matrix_type(input)?,
        ) else {
            return Ok(None);
        };
        let layout = |expression| match self.facts.facts(expression) {
            Ok(ValueFacts::Matrix(facts)) => Some(facts.metadata.layout.clone()),
            _ => None,
        };
        let Some(registry) = self.gadget_recompositions else {
            return Ok(None);
        };
        Ok(registry
            .allows(
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
            )
            .then_some(input))
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
    fn rewrite_closed_relations(
        &mut self,
        normal_form: &mut PolynomialNF,
    ) -> Result<bool, NormalizeError> {
        let has_closed_relations = self
            .relations
            .map(RelationRegistry::has_closed_relations)
            .transpose()?
            .unwrap_or(false);
        if self.watchdog.is_none() {
            return self.rewrite_closed_relations_impl(normal_form, None, has_closed_relations);
        }
        let mut diagnostic = RelationClosureDiagnostic {
            counters: self.watchdog_relation_closure,
            outcomes: HashMap::new(),
        };
        diagnostic.counters.closed_relations_present |= has_closed_relations;
        diagnostic.counters.closures_started =
            diagnostic.counters.closures_started.saturating_add(1);
        diagnostic.counters.active_depth = diagnostic.counters.active_depth.saturating_add(1);
        self.watchdog_publish_relation_closure_with_active(
            DiagnosticPhase::RelationClosure,
            diagnostic.counters,
            normal_form.exact_terms.keys().copied(),
        );
        let result = self.rewrite_closed_relations_impl(
            normal_form,
            Some(&mut diagnostic),
            has_closed_relations,
        );
        diagnostic.counters.active_depth = diagnostic.counters.active_depth.saturating_sub(1);
        if result.is_ok() {
            diagnostic.counters.closures_completed =
                diagnostic.counters.closures_completed.saturating_add(1);
        } else {
            diagnostic.counters.closures_errored =
                diagnostic.counters.closures_errored.saturating_add(1);
        }
        self.watchdog_publish_relation_closure_with_active(
            if result.is_ok() { DiagnosticPhase::RelationClosure } else { DiagnosticPhase::Error },
            diagnostic.counters,
            normal_form.exact_terms.keys().copied(),
        );
        result
    }

    fn rewrite_closed_relations_impl(
        &mut self,
        normal_form: &mut PolynomialNF,
        mut diagnostic: Option<&mut RelationClosureDiagnostic>,
        has_closed_relations: bool,
    ) -> Result<bool, NormalizeError> {
        let setup_started = self.watchdog_timing_start();
        let initial = std::mem::take(&mut normal_form.exact_terms);
        let mut worklist = initial.into_iter().collect::<VecDeque<_>>();
        let mut result = BTreeMap::new();
        let mut changed = false;
        self.watchdog_record_timing(setup_started, |timings| &mut timings.closure_setup);
        if let Some(diagnostic) = diagnostic.as_deref_mut() {
            let initial = u64::try_from(worklist.len()).unwrap_or(u64::MAX);
            diagnostic.counters.initial_terms =
                diagnostic.counters.initial_terms.saturating_add(initial);
            diagnostic.counters.queue_peak = diagnostic.counters.queue_peak.max(initial);
        }
        if self.trace.relation_trace_selected() {
            self.trace.relation_initial = u64::try_from(worklist.len()).unwrap_or(u64::MAX);
            self.trace.relation_worklist = self.trace.relation_initial;
            self.trace.relation_peak_worklist = self.trace.relation_initial;
        }
        while let Some((monomial, coefficient)) = worklist.pop_front() {
            let diagnostic_before = diagnostic.as_deref().map(|diagnostic| diagnostic.counters);
            self.watchdog_record_relation_processed(
                std::iter::once(monomial)
                    .chain(worklist.iter().map(|(id, _)| *id))
                    .chain(result.keys().copied()),
            );
            self.trace.record_relation_processed(worklist.len(), result.len());
            if let Some(diagnostic) = diagnostic.as_deref_mut() {
                diagnostic.counters.dequeued = diagnostic.counters.dequeued.saturating_add(1);
            }
            if coefficient.is_zero() {
                if let Some(diagnostic) = diagnostic.as_deref_mut() {
                    diagnostic.counters.zero_skipped =
                        diagnostic.counters.zero_skipped.saturating_add(1);
                }
                if let (Some(before), Some(diagnostic)) = (diagnostic_before, diagnostic.as_deref())
                {
                    self.watchdog_maybe_publish_relation_progress(before, diagnostic.counters);
                }
                continue;
            }
            let descriptor_and_gadget_started = self.watchdog_timing_start();
            if let Some(diagnostic) = diagnostic.as_deref_mut() {
                diagnostic.counters.nonzero_dequeued =
                    diagnostic.counters.nonzero_dequeued.saturating_add(1);
            }
            let descriptor_shape = diagnostic.as_deref().map(|_| {
                self.monomials.descriptor(monomial).map(|descriptor| {
                    (
                        u64::try_from(descriptor.central_factors.len()).unwrap_or(u64::MAX),
                        u64::try_from(descriptor.ordered_factors.len()).unwrap_or(u64::MAX),
                    )
                })
            });
            let descriptor_shape = match descriptor_shape.transpose() {
                Ok(shape) => shape,
                Err(error) => {
                    self.watchdog_record_timing(descriptor_and_gadget_started, |timings| {
                        &mut timings.descriptor_and_gadget
                    });
                    return Err(error.into());
                }
            };
            if let (Some(diagnostic), Some((central, ordered))) =
                (diagnostic.as_deref_mut(), descriptor_shape)
            {
                diagnostic.counters.central_factors_total =
                    diagnostic.counters.central_factors_total.saturating_add(central);
                diagnostic.counters.central_factors_max =
                    diagnostic.counters.central_factors_max.max(central);
                diagnostic.counters.ordered_factors_total =
                    diagnostic.counters.ordered_factors_total.saturating_add(ordered);
                diagnostic.counters.ordered_factors_max =
                    diagnostic.counters.ordered_factors_max.max(ordered);
            }
            // Relation RHS splices recombine prefix, canonical RHS, and suffix words; a gadget
            // factor ending the prefix then sits adjacent to a decomposition opening the suffix.
            // Recomposition otherwise runs only under `product_nf`, so close those pairs here or
            // the spliced word can never cancel against its ordinarily-evaluated counterpart.
            if let Some(diagnostic) = diagnostic.as_deref_mut() {
                diagnostic.counters.gadget_attempts =
                    diagnostic.counters.gadget_attempts.saturating_add(1);
            }
            let gadget_rewrite = self.rewrite_gadget_decomposition(monomial);
            self.watchdog_record_timing(descriptor_and_gadget_started, |timings| {
                &mut timings.descriptor_and_gadget
            });
            if gadget_rewrite.is_err() {
                record_relation_outcome(
                    diagnostic.as_deref_mut(),
                    monomial,
                    RelationOutcomeKind::Error,
                );
            }
            if let Some(rewritten) = gadget_rewrite? {
                if let Some(diagnostic) = diagnostic.as_deref_mut() {
                    diagnostic.counters.gadget_matches =
                        diagnostic.counters.gadget_matches.saturating_add(1);
                    record_relation_outcome(
                        Some(diagnostic),
                        monomial,
                        RelationOutcomeKind::Gadget,
                    );
                    let output_terms =
                        u64::try_from(rewritten.exact_terms.len()).unwrap_or(u64::MAX);
                    diagnostic.counters.gadget_output_terms_total =
                        diagnostic.counters.gadget_output_terms_total.saturating_add(output_terms);
                    diagnostic.counters.gadget_output_terms_max =
                        diagnostic.counters.gadget_output_terms_max.max(output_terms);
                }
                changed = true;
                if self.trace.relation_trace_selected() {
                    self.trace.relation_rewrites = self.trace.relation_rewrites.saturating_add(1);
                }
                for (rewritten_monomial, rewritten_coefficient) in
                    rewritten.exact_terms.into_iter().rev()
                {
                    worklist.push_front((
                        rewritten_monomial,
                        coefficient.clone() * rewritten_coefficient,
                    ));
                    if let Some(diagnostic) = diagnostic.as_deref_mut() {
                        diagnostic.counters.enqueued =
                            diagnostic.counters.enqueued.saturating_add(1);
                        diagnostic.counters.queue_peak = diagnostic
                            .counters
                            .queue_peak
                            .max(u64::try_from(worklist.len()).unwrap_or(u64::MAX));
                    }
                    if self.trace.relation_trace_selected() {
                        self.trace.relation_enqueues =
                            self.trace.relation_enqueues.saturating_add(1);
                        self.trace.relation_peak_worklist = self
                            .trace
                            .relation_peak_worklist
                            .max(u64::try_from(worklist.len()).unwrap_or(u64::MAX));
                    }
                }
                if let (Some(before), Some(diagnostic)) = (diagnostic_before, diagnostic.as_deref())
                {
                    self.watchdog_maybe_publish_relation_progress(before, diagnostic.counters);
                }
                continue;
            }
            self.counters.relation_candidates = self.counters.relation_candidates.saturating_add(1);
            let relation_match =
                self.find_relation_match(monomial, has_closed_relations, diagnostic.as_deref_mut());
            if relation_match.is_err() {
                record_relation_outcome(
                    diagnostic.as_deref_mut(),
                    monomial,
                    RelationOutcomeKind::Error,
                );
            }
            let Some(relation_match) = relation_match? else {
                let no_match_started = self.watchdog_timing_start();
                if let Some(diagnostic) = diagnostic.as_deref_mut() {
                    diagnostic.counters.no_matches =
                        diagnostic.counters.no_matches.saturating_add(1);
                    record_relation_outcome(
                        Some(diagnostic),
                        monomial,
                        RelationOutcomeKind::NoMatch,
                    );
                }
                merge_term(&mut result, monomial, coefficient);
                if let Some(diagnostic) = diagnostic.as_deref_mut() {
                    diagnostic.counters.result_terms =
                        u64::try_from(result.len()).unwrap_or(u64::MAX);
                }
                if let (Some(before), Some(diagnostic)) = (diagnostic_before, diagnostic.as_deref())
                {
                    self.watchdog_maybe_publish_relation_progress(before, diagnostic.counters);
                }
                self.watchdog_record_timing(no_match_started, |timings| {
                    &mut timings.no_match_result_merge
                });
                continue;
            };
            record_relation_outcome(diagnostic.as_deref_mut(), monomial, relation_match.kind);
            changed = true;
            if self.trace.relation_trace_selected() {
                self.trace.relation_rewrites = self.trace.relation_rewrites.saturating_add(1);
            }
            self.counters.relation_applied = self.counters.relation_applied.saturating_add(1);
            let rhs_fetch_started = self.watchdog_timing_start();
            let rhs_parts = (|| -> Result<_, NormalizeError> {
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
                Ok((rhs, left, suffix))
            })();
            self.watchdog_record_timing(rhs_fetch_started, |timings| {
                &mut timings.rhs_fetch_prefix_suffix
            });
            let (rhs, left, suffix) = rhs_parts?;
            let rhs_recombine_started = self.watchdog_timing_start();
            let recombine_result = (|| -> Result<(), NormalizeError> {
                let mut recombined = Vec::new();
                if let Some(diagnostic) = diagnostic.as_deref_mut() {
                    diagnostic.counters.rhs_splices =
                        diagnostic.counters.rhs_splices.saturating_add(1);
                    let rhs_terms = u64::try_from(rhs.exact_terms.len()).unwrap_or(u64::MAX);
                    diagnostic.counters.rhs_terms_total =
                        diagnostic.counters.rhs_terms_total.saturating_add(rhs_terms);
                    diagnostic.counters.rhs_terms_max =
                        diagnostic.counters.rhs_terms_max.max(rhs_terms);
                }
                for (rhs_monomial, rhs_coefficient) in &rhs.exact_terms {
                    let mut combined = *rhs_monomial;
                    if let Some(left) = left {
                        combined = self.monomials.combine_interned(self.scope, left, combined)?;
                        if let Some(diagnostic) = diagnostic.as_deref_mut() {
                            diagnostic.counters.monomial_combines =
                                diagnostic.counters.monomial_combines.saturating_add(1);
                            diagnostic.counters.prefix_combines =
                                diagnostic.counters.prefix_combines.saturating_add(1);
                        }
                    }
                    if let Some(suffix) = suffix {
                        combined = self.monomials.combine_interned(self.scope, combined, suffix)?;
                        if let Some(diagnostic) = diagnostic.as_deref_mut() {
                            diagnostic.counters.monomial_combines =
                                diagnostic.counters.monomial_combines.saturating_add(1);
                            diagnostic.counters.suffix_combines =
                                diagnostic.counters.suffix_combines.saturating_add(1);
                        }
                    }
                    recombined.push((combined, &coefficient * rhs_coefficient));
                }
                for term in recombined.into_iter().rev() {
                    worklist.push_front(term);
                    if let Some(diagnostic) = diagnostic.as_deref_mut() {
                        diagnostic.counters.enqueued =
                            diagnostic.counters.enqueued.saturating_add(1);
                        diagnostic.counters.rhs_terms_enqueued =
                            diagnostic.counters.rhs_terms_enqueued.saturating_add(1);
                        diagnostic.counters.queue_peak = diagnostic
                            .counters
                            .queue_peak
                            .max(u64::try_from(worklist.len()).unwrap_or(u64::MAX));
                    }
                    if self.trace.relation_trace_selected() {
                        self.trace.relation_enqueues =
                            self.trace.relation_enqueues.saturating_add(1);
                        self.trace.relation_peak_worklist = self
                            .trace
                            .relation_peak_worklist
                            .max(u64::try_from(worklist.len()).unwrap_or(u64::MAX));
                    }
                }
                if let (Some(before), Some(diagnostic)) = (diagnostic_before, diagnostic.as_deref())
                {
                    self.watchdog_maybe_publish_relation_progress(before, diagnostic.counters);
                }
                Ok(())
            })();
            self.watchdog_record_timing(rhs_recombine_started, |timings| {
                &mut timings.rhs_recombine_enqueue
            });
            recombine_result?;
        }
        let final_assignment_started = self.watchdog_timing_start();
        normal_form.exact_terms = result;
        if let Some(diagnostic) = diagnostic.as_deref_mut() {
            let final_terms = u64::try_from(normal_form.exact_terms.len()).unwrap_or(u64::MAX);
            diagnostic.counters.result_terms = final_terms;
            diagnostic.counters.final_terms = final_terms;
        }
        if self.trace.relation_trace_selected() {
            self.trace.relation_worklist = 0;
            self.trace.relation_result =
                u64::try_from(normal_form.exact_terms.len()).unwrap_or(u64::MAX);
        }
        if normal_form.exact_terms.is_empty() {
            normal_form.bounded_summary = BoundedSummary::known(CoefficientBound::ExactZero);
        }
        self.watchdog_record_timing(final_assignment_started, |timings| {
            &mut timings.closure_final_assignment
        });
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
        has_closed_relations: bool,
        mut diagnostic: Option<&mut RelationClosureDiagnostic>,
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
        let closed_search_started = self.watchdog_timing_start();
        if has_closed_relations {
            // Closed and universal relations share the same ordered-subword matcher. The
            // whole-term lookup remains a fast path, but it is not the semantic boundary of
            // relation use.
            for candidate_layout in [layout.clone(), None] {
                if let Some(diagnostic) = diagnostic.as_deref_mut() {
                    diagnostic.counters.whole_closed_probes =
                        diagnostic.counters.whole_closed_probes.saturating_add(1);
                }
                let lhs = CanonicalLhsKey { layout: candidate_layout, monomial };
                if let Some(diagnostic) = diagnostic.as_deref_mut() {
                    diagnostic.counters.whole_closed_resolves =
                        diagnostic.counters.whole_closed_resolves.saturating_add(1);
                }
                let resolution = relations.resolve_closed(&lhs);
                if matches!(resolution, Err(RelationRegistryError::Ambiguous { .. })) {
                    if let Some(diagnostic) = diagnostic.as_deref_mut() {
                        diagnostic.counters.whole_closed_ambiguities =
                            diagnostic.counters.whole_closed_ambiguities.saturating_add(1);
                        diagnostic.counters.match_errors =
                            diagnostic.counters.match_errors.saturating_add(1);
                    }
                }
                let resolution = match resolution {
                    Ok(resolution) => resolution,
                    Err(error) => {
                        self.watchdog_record_timing(closed_search_started, |timings| {
                            &mut timings.closed_search
                        });
                        return Err(error.into());
                    }
                };
                if let RelationResolution::Rewrite(rhs) = resolution {
                    if let Some(diagnostic) = diagnostic.as_deref_mut() {
                        diagnostic.counters.whole_closed_matches =
                            diagnostic.counters.whole_closed_matches.saturating_add(1);
                    }
                    let matched = RelationMatch {
                        kind: RelationOutcomeKind::WholeClosed,
                        prefix: Vec::new(),
                        suffix: Vec::new(),
                        remaining_central: Vec::new(),
                        rhs,
                    };
                    self.watchdog_record_timing(closed_search_started, |timings| {
                        &mut timings.closed_search
                    });
                    return Ok(Some(matched));
                }
            }
            let closed_subword =
                self.find_closed_subword_match(&central, &ordered, diagnostic.as_deref_mut());
            self.watchdog_record_timing(closed_search_started, |timings| {
                &mut timings.closed_search
            });
            if let Some(result) = closed_subword? {
                return Ok(Some(result));
            }
        } else {
            self.watchdog_record_timing(closed_search_started, |timings| {
                &mut timings.closed_search
            });
        }
        let universal_started = self.watchdog_timing_start();
        let result = self.find_universal_subword_match(&central, &ordered, diagnostic);
        self.watchdog_record_timing(universal_started, |timings| {
            &mut timings.universal_search_total
        });
        result
    }

    fn find_closed_subword_match(
        &mut self,
        central: &[ScopedExprId],
        ordered: &[ScopedExprId],
        mut diagnostic: Option<&mut RelationClosureDiagnostic>,
    ) -> Result<Option<RelationMatch>, NormalizeError> {
        let Some(relations) = self.relations else { return Ok(None) };
        // Leftmost boundary wins. At one boundary, try the longest word first so a registered
        // `B * X * K` relation is not shadowed by a shorter `X * K` relation.
        for start in 0..=ordered.len() {
            for width in (1..=ordered.len() - start).rev() {
                self.trace.record_relation_probe(true);
                if let Some(diagnostic) = diagnostic.as_deref_mut() {
                    let before = diagnostic.counters.closed_window_probes;
                    diagnostic.counters.closed_window_probes =
                        diagnostic.counters.closed_window_probes.saturating_add(1);
                    let _published = Self::watchdog_publish_relation_matcher_counter(
                        self.watchdog.as_ref(),
                        &mut self.watchdog_relation_closure,
                        before,
                        diagnostic.counters.closed_window_probes,
                        diagnostic.counters,
                    );
                    #[cfg(test)]
                    if _published {
                        self.watchdog_hot_publish_count =
                            self.watchdog_hot_publish_count.saturating_add(1);
                        if let Some(observer) = self.relation_matcher_publish_observer.as_mut() {
                            observer(diagnostic.counters);
                        }
                    }
                }
                let Some(candidate) = self.monomials.find_interned(
                    self.scope,
                    &[],
                    &ordered[start..start + width],
                )?
                else {
                    continue
                };
                if let Some(diagnostic) = diagnostic.as_deref_mut() {
                    diagnostic.counters.closed_window_interned_hits =
                        diagnostic.counters.closed_window_interned_hits.saturating_add(1);
                }
                let remaining_central = central.to_vec();
                let candidate_layout = ordered[start..start + width].first().and_then(|factor| {
                    match self.facts.facts(factor.expression()) {
                        Ok(ValueFacts::Matrix(facts)) => Some(facts.metadata.layout.clone()),
                        _ => None,
                    }
                });
                for candidate_layout in [candidate_layout, None] {
                    if let Some(diagnostic) = diagnostic.as_deref_mut() {
                        diagnostic.counters.closed_window_resolves =
                            diagnostic.counters.closed_window_resolves.saturating_add(1);
                    }
                    let lhs = CanonicalLhsKey { layout: candidate_layout, monomial: candidate };
                    let resolution = relations.resolve_closed(&lhs);
                    if matches!(resolution, Err(RelationRegistryError::Ambiguous { .. })) {
                        if let Some(diagnostic) = diagnostic.as_deref_mut() {
                            diagnostic.counters.closed_window_ambiguities =
                                diagnostic.counters.closed_window_ambiguities.saturating_add(1);
                            diagnostic.counters.match_errors =
                                diagnostic.counters.match_errors.saturating_add(1);
                        }
                    }
                    if let RelationResolution::Rewrite(rhs) = resolution? {
                        if let Some(diagnostic) = diagnostic.as_deref_mut() {
                            diagnostic.counters.closed_window_matches =
                                diagnostic.counters.closed_window_matches.saturating_add(1);
                            diagnostic.counters.closed_subword_matches =
                                diagnostic.counters.closed_subword_matches.saturating_add(1);
                        }
                        return Ok(Some(RelationMatch {
                            kind: RelationOutcomeKind::ClosedWindow,
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
        mut diagnostic: Option<&mut RelationClosureDiagnostic>,
    ) -> Result<Option<RelationMatch>, NormalizeError> {
        let Some(relations) = self.relations else { return Ok(None) };
        let mut candidates = BTreeMap::<(usize, usize), BTreeSet<_>>::new();
        for (k_position, &k_factor) in ordered.iter().enumerate() {
            let factor_dispatch_started = self.watchdog_timing_start();
            self.trace.record_relation_probe(false);
            if let Some(diagnostic) = diagnostic.as_deref_mut() {
                let before = diagnostic.counters.universal_probes;
                diagnostic.counters.universal_probes =
                    diagnostic.counters.universal_probes.saturating_add(1);
                let _published = Self::watchdog_publish_relation_matcher_counter(
                    self.watchdog.as_ref(),
                    &mut self.watchdog_relation_closure,
                    before,
                    diagnostic.counters.universal_probes,
                    diagnostic.counters,
                );
                #[cfg(test)]
                if _published {
                    self.watchdog_hot_publish_count =
                        self.watchdog_hot_publish_count.saturating_add(1);
                    if let Some(observer) = self.relation_matcher_publish_observer.as_mut() {
                        observer(diagnostic.counters);
                    }
                }
            }
            let factor_dispatch = (|| -> Result<_, NormalizeError> {
                let node = self.expressions.node(k_factor.expression())?;
                let ValueOperator::ProgramCall { program } = node.operator else { return Ok(None) };
                let unary = node.inputs.len() == 1;
                let dispatch = relations.dispatch_for_preimage_program(program);
                if matches!(dispatch, Err(RelationRegistryError::AmbiguousPreimageDispatch)) {
                    if let Some(diagnostic) = diagnostic.as_deref_mut() {
                        diagnostic.counters.universal_ambiguities =
                            diagnostic.counters.universal_ambiguities.saturating_add(1);
                        diagnostic.counters.match_errors =
                            diagnostic.counters.match_errors.saturating_add(1);
                    }
                }
                Ok(dispatch?.map(|dispatch| (unary, dispatch)))
            })();
            self.watchdog_record_timing(factor_dispatch_started, |timings| {
                &mut timings.universal_factor_dispatch
            });
            let Some((unary, dispatch)) = factor_dispatch? else {
                continue;
            };
            if let Some(diagnostic) = diagnostic.as_deref_mut() {
                let before = diagnostic.counters.universal_dispatch_hits;
                diagnostic.counters.universal_dispatch_hits =
                    diagnostic.counters.universal_dispatch_hits.saturating_add(1);
                let _published = Self::watchdog_publish_relation_matcher_counter(
                    self.watchdog.as_ref(),
                    &mut self.watchdog_relation_closure,
                    before,
                    diagnostic.counters.universal_dispatch_hits,
                    diagnostic.counters,
                );
                #[cfg(test)]
                if _published {
                    self.watchdog_hot_publish_count =
                        self.watchdog_hot_publish_count.saturating_add(1);
                    if let Some(observer) = self.relation_matcher_publish_observer.as_mut() {
                        observer(diagnostic.counters);
                    }
                }
            }
            let selector_range_started = self.watchdog_timing_start();
            let selector_range = (|| -> Result<_, NormalizeError> {
                if !unary {
                    return Ok(None)
                }
                let index = self.expressions.scoped_only_input(k_factor)?;
                if index.program() != self.scope {
                    return Err(ArenaError::ScopeMismatch {
                        expected: self.scope,
                        actual: index.program(),
                    }
                    .into())
                }
                Ok(self.universal_index_range(index)?.map(|range| (index, range)))
            })();
            self.watchdog_record_timing(selector_range_started, |timings| {
                &mut timings.universal_selector_range
            });
            let Some((index, index_range)) = selector_range? else { continue };
            // Universal specialization may run nested root normalizations. Flush this closure's
            // local work counters without taking the watchdog mutex, then resume from the shared
            // outer-session counters so nested work is never overwritten on unwind.
            if let Some(diagnostic) = diagnostic.as_deref_mut() {
                let before = diagnostic.counters.universal_specializations;
                diagnostic.counters.universal_specializations =
                    diagnostic.counters.universal_specializations.saturating_add(1);
                let _published = Self::watchdog_publish_relation_matcher_counter(
                    self.watchdog.as_ref(),
                    &mut self.watchdog_relation_closure,
                    before,
                    diagnostic.counters.universal_specializations,
                    diagnostic.counters,
                );
                #[cfg(test)]
                if _published {
                    self.watchdog_hot_publish_count =
                        self.watchdog_hot_publish_count.saturating_add(1);
                    if let Some(observer) = self.relation_matcher_publish_observer.as_mut() {
                        observer(diagnostic.counters);
                    }
                }
                self.watchdog_relation_closure = diagnostic.counters;
            }
            let specialized_started = self.watchdog_timing_start();
            let specialized = self.specialized_universal_cached(dispatch, index, index_range);
            self.watchdog_record_timing(specialized_started, |timings| {
                &mut timings.universal_specialized_cached
            });
            if let Some(diagnostic) = diagnostic.as_deref_mut() {
                diagnostic.counters = self.watchdog_relation_closure;
            }
            let specialized = specialized?;
            let lhs_layout_span_started = self.watchdog_timing_start();
            let lhs_layout_span = (|| -> Result<(), NormalizeError> {
                for (lhs, rhs_candidates) in specialized {
                    if let Some(diagnostic) = diagnostic.as_deref_mut() {
                        let before = diagnostic.counters.universal_lhs_candidates;
                        diagnostic.counters.universal_lhs_candidates =
                            diagnostic.counters.universal_lhs_candidates.saturating_add(1);
                        let _published = Self::watchdog_publish_relation_matcher_counter(
                            self.watchdog.as_ref(),
                            &mut self.watchdog_relation_closure,
                            before,
                            diagnostic.counters.universal_lhs_candidates,
                            diagnostic.counters,
                        );
                        #[cfg(test)]
                        if _published {
                            self.watchdog_hot_publish_count =
                                self.watchdog_hot_publish_count.saturating_add(1);
                            if let Some(observer) = self.relation_matcher_publish_observer.as_mut()
                            {
                                observer(diagnostic.counters);
                            }
                        }
                    }
                    let descriptor = self.monomials.descriptor(lhs.monomial)?;
                    // Universal preimage relations consume an adjacent ordered word. A relation
                    // whose LHS is central-only has no lexical boundary and is deliberately not
                    // dispatched here.
                    if descriptor.ordered_factors.is_empty() ||
                        !descriptor.central_factors.is_empty()
                    {
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
                                Ok(ValueFacts::Matrix(facts)) => {
                                    Some(facts.metadata.layout.clone())
                                }
                                _ => None,
                            });
                        if lhs.layout != candidate_layout {
                            continue;
                        }
                    }
                    let mut lhs_k_positions =
                        descriptor.ordered_factors.iter().enumerate().filter_map(
                            |(position, factor)| (*factor == k_factor).then_some(position),
                        );
                    while let Some(lhs_k_position) = lhs_k_positions.next() {
                        let ordered_len = descriptor.ordered_factors.len();
                        let Some(start) = k_position.checked_sub(lhs_k_position) else { continue };
                        let Some(end) = start.checked_add(ordered_len) else { continue };
                        if end > ordered.len() ||
                            ordered[start..end] != descriptor.ordered_factors[..]
                        {
                            continue;
                        }
                        if remove_central_subword(central, &descriptor.central_factors).is_none() {
                            continue
                        }
                        // Universal matching is selected globally, not by K occurrence or
                        // registration/map iteration order.  All universal LHSes have an empty
                        // central word, so the remaining central factors are identical for every
                        // candidate in this term; retain the computed proof only after selecting
                        // the winning span below.
                        candidates
                            .entry((start, end))
                            .or_default()
                            .extend(rhs_candidates.iter().copied());
                        if let Some(diagnostic) = diagnostic.as_deref_mut() {
                            let before = diagnostic.counters.universal_span_candidates;
                            diagnostic.counters.universal_span_candidates =
                                diagnostic.counters.universal_span_candidates.saturating_add(1);
                            let _published = Self::watchdog_publish_relation_matcher_counter(
                                self.watchdog.as_ref(),
                                &mut self.watchdog_relation_closure,
                                before,
                                diagnostic.counters.universal_span_candidates,
                                diagnostic.counters,
                            );
                            #[cfg(test)]
                            if _published {
                                self.watchdog_hot_publish_count =
                                    self.watchdog_hot_publish_count.saturating_add(1);
                                if let Some(observer) =
                                    self.relation_matcher_publish_observer.as_mut()
                                {
                                    observer(diagnostic.counters);
                                }
                            }
                        }
                    }
                }
                Ok(())
            })();
            self.watchdog_record_timing(lhs_layout_span_started, |timings| {
                &mut timings.universal_lhs_layout_span
            });
            lhs_layout_span?;
        }
        let global_selection_started = self.watchdog_timing_start();
        let Some(((start, end), rhs_candidates)) = candidates.into_iter().min_by(
            |((left_start, left_end), _), ((right_start, right_end), _)| {
                left_start.cmp(right_start).then_with(|| {
                    right_end
                        .saturating_sub(*right_start)
                        .cmp(&left_end.saturating_sub(*left_start))
                })
            },
        ) else {
            self.watchdog_record_timing(global_selection_started, |timings| {
                &mut timings.universal_global_selection
            });
            return Ok(None);
        };
        let resolution = super::relation::resolve_candidates(Some(&rhs_candidates));
        if matches!(resolution, Err(RelationRegistryError::Ambiguous { .. })) {
            if let Some(diagnostic) = diagnostic.as_deref_mut() {
                diagnostic.counters.universal_ambiguities =
                    diagnostic.counters.universal_ambiguities.saturating_add(1);
                diagnostic.counters.match_errors =
                    diagnostic.counters.match_errors.saturating_add(1);
            }
        }
        let resolution = match resolution {
            Ok(resolution) => resolution,
            Err(error) => {
                self.watchdog_record_timing(global_selection_started, |timings| {
                    &mut timings.universal_global_selection
                });
                return Err(error.into());
            }
        };
        let super::relation::RelationResolution::Rewrite(rhs) = resolution else {
            self.watchdog_record_timing(global_selection_started, |timings| {
                &mut timings.universal_global_selection
            });
            return Ok(None);
        };
        if let Some(diagnostic) = diagnostic.as_deref_mut() {
            diagnostic.counters.universal_matches =
                diagnostic.counters.universal_matches.saturating_add(1);
            diagnostic.counters.universal_rewrites =
                diagnostic.counters.universal_rewrites.saturating_add(1);
        }
        let matched = RelationMatch {
            kind: RelationOutcomeKind::Universal,
            prefix: ordered[..start].to_vec(),
            suffix: ordered[end..].to_vec(),
            remaining_central: central.to_vec(),
            rhs,
        };
        self.watchdog_record_timing(global_selection_started, |timings| {
            &mut timings.universal_global_selection
        });
        Ok(Some(matched))
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
        if self.watchdog.is_none() {
            let relations =
                self.relations.ok_or(NormalizeError::Relation(RelationRegistryError::NotFrozen))?;
            let generation = relations.frozen_generation()?;
            let key = RuntimeSpecializationKey { dispatch: dispatch.clone(), index, generation };
            if let Some(cached) =
                self.normalization.as_deref().and_then(|cache| cache.runtime_get(&key)).cloned()
            {
                return Ok(cached);
            }
            let specialized = self.specialize_universal(
                dispatch,
                index,
                index_range,
                SpecializationKind::Ordinary,
            )?;
            self.normalization
                .as_deref_mut()
                .ok_or(NormalizeError::Relation(RelationRegistryError::InvalidCanonicalRhs))?
                .runtime_insert(key, specialized.clone());
            return Ok(specialized);
        }
        let key_lookup_started = self.watchdog_timing_start();
        let key_lookup = (|| -> Result<_, NormalizeError> {
            let relations =
                self.relations.ok_or(NormalizeError::Relation(RelationRegistryError::NotFrozen))?;
            let generation = relations.frozen_generation()?;
            let key = RuntimeSpecializationKey { dispatch: dispatch.clone(), index, generation };
            let cache_hit = self
                .normalization
                .as_deref()
                .is_some_and(|cache| cache.runtime_get(&key).is_some());
            Ok((key, cache_hit))
        })();
        self.watchdog_record_timing(key_lookup_started, |timings| &mut timings.cached_key_lookup);
        let (key, cache_hit) = key_lookup?;
        if cache_hit {
            let hit_clone_started = self.watchdog_timing_start();
            let hit_clone = (|| -> Result<_, NormalizeError> {
                let cached =
                    self.normalization.as_deref().and_then(|cache| cache.runtime_get(&key)).ok_or(
                        NormalizeError::Relation(RelationRegistryError::InvalidCanonicalRhs),
                    )?;
                let returned_entries = cached.values().fold(0_u64, |total, entries| {
                    total.saturating_add(u64::try_from(entries.len()).unwrap_or(u64::MAX))
                });
                Ok((cached.clone(), returned_entries))
            })();
            self.watchdog_record_timing(hit_clone_started, |timings| &mut timings.cached_hit_clone);
            let (cached, returned_entries) = hit_clone?;
            if self.watchdog.is_some() {
                self.watchdog_timings.cached_hit_returned_entries_total = self
                    .watchdog_timings
                    .cached_hit_returned_entries_total
                    .saturating_add(returned_entries);
                self.watchdog_timings.cached_hit_returned_entries_max =
                    self.watchdog_timings.cached_hit_returned_entries_max.max(returned_entries);
            }
            self.watchdog_record_specialization(DiagnosticPhase::RuntimeLookup, |counters| {
                counters.runtime_lookup_hits = counters.runtime_lookup_hits.saturating_add(1);
            });
            return Ok(cached);
        }
        self.watchdog_record_specialization(DiagnosticPhase::RuntimeLookup, |counters| {
            counters.runtime_lookup_misses = counters.runtime_lookup_misses.saturating_add(1);
        });
        let miss_started = self.watchdog_timing_start();
        let specialized =
            self.specialize_universal(dispatch, index, index_range, SpecializationKind::Ordinary);
        self.watchdog_record_timing(miss_started, |timings| &mut timings.cached_miss_specialize);
        let specialized = specialized?;
        let insert_clone_started = self.watchdog_timing_start();
        let insert_clone = (|| -> Result<_, NormalizeError> {
            self.normalization
                .as_deref_mut()
                .ok_or(NormalizeError::Relation(RelationRegistryError::InvalidCanonicalRhs))?
                .runtime_insert(key, specialized.clone());
            Ok(specialized)
        })();
        self.watchdog_record_timing(insert_clone_started, |timings| {
            &mut timings.cached_insert_return_clone
        });
        insert_clone
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

fn record_relation_outcome(
    diagnostic: Option<&mut RelationClosureDiagnostic>,
    monomial: MonomialId,
    outcome: RelationOutcomeKind,
) {
    let Some(diagnostic) = diagnostic else { return };
    match diagnostic.outcomes.get(&monomial).copied() {
        None => {
            diagnostic.outcomes.insert(monomial, outcome);
        }
        Some(existing) if existing == outcome => {
            diagnostic.counters.duplicate_same_outcome =
                diagnostic.counters.duplicate_same_outcome.saturating_add(1);
        }
        Some(_) => {
            // Keep the first result as the authoritative observation. Diagnostics never replace
            // or consult it to decide execution; a changed classification is merely reported.
            diagnostic.counters.duplicate_changed_outcome =
                diagnostic.counters.duplicate_changed_outcome.saturating_add(1);
        }
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

fn with_summary_arc(
    normal_form: Arc<PolynomialNF>,
    bound: NumericContract<CoefficientBound>,
) -> Arc<PolynomialNF> {
    match Arc::try_unwrap(normal_form) {
        Ok(normal_form) => Arc::new(with_summary(normal_form, bound)),
        Err(normal_form) => Arc::new(with_summary(normal_form.as_ref().clone(), bound)),
    }
}

fn synchronize_materialized_value(
    mut value: AnalyzedValue,
    normal_form: Arc<PolynomialNF>,
) -> AnalyzedValue {
    let normal_form = with_summary_arc(normal_form, value.coefficient_bound.clone());
    value.coefficient_bound = normal_form.bounded_summary.coefficient_bound.clone();
    value.exact_nf = Some(normal_form);
    value
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

    #[test]
    fn four_class_census_uses_bound_authority_and_deduplicates_global_payload() {
        let mut expressions = ExprArena::new();
        let finite = gaussian_factor(&mut expressions, matrix_type(), 80_001, 3);
        let large = source_with(&mut expressions, matrix_type(), 80_002);
        let missing =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[finite, finite]).unwrap();
        let root =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[missing, large]).unwrap();
        let mut programs = ProgramArena::new();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let scope = semantic.program();
        let mut ids = BTreeMap::new();
        for expression in [finite, large, missing] {
            let scoped = programs.scoped(&expressions, scope, expression).unwrap();
            ids.insert(
                expression,
                monomials.intern(&expressions, &programs, &[], &[scoped]).unwrap(),
            );
        }
        let shared = Arc::new(PolynomialNF {
            exact_terms: BTreeMap::from([
                (ids[&finite], BigInt::from(1_u8)),
                (ids[&large], BigInt::from(1_u8)),
                (ids[&missing], BigInt::from(1_u8)),
            ]),
            bounded_summary: BoundedSummary::missing(),
        });
        let normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let census = normalizer
            .diagnostic_four_class_census(&[DiagnosticExactNf {
                normal_form: Arc::clone(&shared),
                ordinal: 0,
                under_product: false,
            }])
            .unwrap();
        assert_eq!(census.finite_no_relation.unique_monomials, 1);
        assert_eq!(census.finite_no_relation.term_refs, 1);
        assert_eq!(census.missing.unique_monomials, 1);
        assert_eq!(census.missing.term_refs, 1);
        assert_eq!(census.large.unique_monomials, 1);
        assert_eq!(census.large.term_refs, 1);
        assert_eq!(census.finite_relation_frontier, DiagnosticClassStats::default());
        assert_eq!(census.frontier_unique_union, census.frontier_reason_unique_union);
        assert_eq!(census.top_len, 1);
        assert_eq!(census.top[0].finite_no_relation_refs, 1);
        assert_eq!(census.top[0].missing_refs, 1);
        assert_eq!(census.top[0].large_refs, 1);
        assert_eq!(
            census.finite_no_relation.payload_lower_bound_bytes,
            normalizer.monomials.descriptor_payload_lower_bound_bytes(ids[&finite]).unwrap()
        );
    }

    #[test]
    fn four_class_census_closed_blanket_exactly_covers_finite_frontier_and_rejects_mutable_authority()
     {
        let mut expressions = ExprArena::new();
        let b = gaussian_factor(&mut expressions, matrix_type(), 80_011, 3);
        let k = gaussian_factor(&mut expressions, matrix_type(), 80_012, 5);
        let p = gaussian_factor(&mut expressions, matrix_type(), 80_013, 7);
        let bk = product(&mut expressions, &[b, k]);
        let mut programs = ProgramArena::new();
        let (facts, mut monomials, _) = setup(&mut expressions, &mut programs, bk);
        let (relations, mut cache, lhs) = register_test_closed_relation(
            &mut expressions,
            &programs,
            &facts,
            &mut monomials,
            bk,
            p,
            k,
            b,
        );
        let nf = Arc::new(PolynomialNF {
            exact_terms: BTreeMap::from([(lhs, BigInt::from(1_u8))]),
            bounded_summary: BoundedSummary::missing(),
        });
        let normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_relations(&relations, &mut cache);
        let census = normalizer
            .diagnostic_four_class_census(&[DiagnosticExactNf {
                normal_form: nf,
                ordinal: 0,
                under_product: false,
            }])
            .unwrap();
        assert_eq!(census.finite_relation_frontier.unique_monomials, 1);
        assert_eq!(census.closed_blanket.unique_monomials, 1);
        assert_eq!(census.frontier_unique_union, 1);
        assert_eq!(census.frontier_reason_unique_union, 1);
        assert_eq!(
            census.finite_relation_frontier.term_refs,
            census.frontier_reason_term_ref_union
        );
        assert_eq!(
            census.finite_relation_frontier.payload_lower_bound_bytes,
            census.frontier_reason_payload_union
        );

        let mutable = RelationRegistry::new();
        let mut mutable_cache = NormalizationCache::new();
        let normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_relations(&mutable, &mut mutable_cache);
        assert_eq!(
            normalizer.diagnostic_four_class_census(&[]),
            Err(NormalizeError::Relation(RelationRegistryError::NotFrozen))
        );
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
    fn forced_monomial_gc_reports_three_sweeps_and_exact_watchdog_terminal_totals() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let left = gaussian_factor(&mut expressions, matrix_type(), 81_011, 3);
        let right = gaussian_factor(&mut expressions, matrix_type(), 81_012, 5);
        let root = product(&mut expressions, &[left, right]);
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);

        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_watchdog_override(true, Duration::from_secs(60));
        normalizer.monomial_gc_allocation_threshold_bytes = 1;
        let value = normalizer.normalize(semantic).unwrap();
        let gc = normalizer.gc_counters;
        assert_eq!(gc.sweep_count, 3);
        assert_eq!(gc.last_sweep_node, 3);
        assert_eq!(gc.last_high_water_slots, 3);
        assert_eq!(gc.last_occupied_slots, 1);
        assert_eq!(gc.last_reclaimed_slots, 2);
        assert_eq!(gc.cumulative_reclaimed_slots, 2);
        assert_eq!(gc.cumulative_reclaimed_payload_bytes, gc.last_reclaimed_payload_bytes);
        assert!(gc.last_reclaimed_payload_bytes > 0);
        assert!(gc.last_allocated_payload_before_bytes > 0);
        assert_eq!(gc.last_bucket_entries, 1);
        assert!(gc.sweep_total_ns >= gc.sweep_max_ns);
        assert!(gc.sweep_max_ns >= gc.sweep_last_ns);
        assert!(gc.value_cache_top8_len > 0, "ordinary watchdog owner telemetry stays enabled");
        assert_eq!(gc.exact_plan_four_class, DiagnosticFourClassCensus::default());
        assert_eq!(gc.four_class_total_ns, 0);
        let terminal = normalizer.last_watchdog_snapshots.last().copied().unwrap();
        assert_eq!(terminal.gc, gc);
        let live = *value.exact_nf.as_ref().unwrap().exact_terms.keys().next().unwrap();
        assert!(normalizer.monomials.descriptor(live).is_ok());
    }

    #[test]
    fn forced_monomial_gc_watchdog_on_off_preserves_nf_bound_and_cache_fingerprint() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let left = gaussian_factor(&mut expressions, matrix_type(), 81_021, 3);
        let right = gaussian_factor(&mut expressions, matrix_type(), 81_022, 5);
        let root = product(&mut expressions, &[left, right]);
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let mut relations = RelationRegistry::new();
        relations.freeze();
        let mut cache = NormalizationCache::new();

        let (off, off_counters, off_gc) = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                    .unwrap()
                    .with_relations(&relations, &mut cache)
                    .with_watchdog_override(false, Duration::from_secs(60));
            normalizer.monomial_gc_allocation_threshold_bytes = 1;
            let value = normalizer.normalize(semantic).unwrap();
            (value, normalizer.counters(), normalizer.gc_counters)
        };
        assert_eq!(off_gc.value_cache_top8_len, 0);
        assert_eq!(off_gc.four_class_total_ns, 0);
        assert_eq!(off_gc.four_class_max_ns, 0);
        assert_eq!(off_gc.four_class_last_ns, 0);
        let fingerprint = cache.canonical_state_fingerprint();
        let (on, on_counters, on_gc) = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                    .unwrap()
                    .with_relations(&relations, &mut cache)
                    .with_watchdog_override(true, Duration::from_secs(60));
            normalizer.four_class_census_enabled = true;
            normalizer.monomial_gc_allocation_threshold_bytes = 1;
            let value = normalizer.normalize(semantic).unwrap();
            (value, normalizer.counters(), normalizer.gc_counters)
        };
        assert!(on_gc.sweep_count > 0);
        assert!(on_gc.value_cache_top8_len > 0);
        assert!(on_gc.four_class_total_ns >= on_gc.four_class_max_ns);
        assert!(on_gc.four_class_max_ns >= on_gc.four_class_last_ns);
        assert_eq!(off.semantic, on.semantic);
        assert_eq!(off.exact_nf, on.exact_nf);
        assert_eq!(off.coefficient_bound, on.coefficient_bound);
        assert_eq!(off_counters, on_counters);
        assert_eq!(cache.canonical_state_fingerprint(), fingerprint);
    }

    #[test]
    fn forced_monomial_gc_keeps_left_deep_fresh_factor_chain_occupied_set_bounded() {
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
        let final_id = *value.exact_nf.as_ref().unwrap().exact_terms.keys().next().unwrap();
        assert_eq!(monomials.descriptor(final_id).unwrap().ordered_factors.len(), 128);
        assert_eq!(monomials.occupied_len(), 1);
        assert!(monomials.len() > 128, "slot high-water remains monotonic while payload is swept");
        assert_eq!(monomials.allocated_payload_since_sweep(), 0);
    }

    #[test]
    fn owner_census_tracks_cache_last_use_and_gadget_hold_lifecycles() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let root = source(&mut expressions);
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let monomial = monomials.intern(&expressions, &programs, &[], &[semantic]).unwrap();
        let normal_form = Arc::new(PolynomialNF {
            exact_terms: BTreeMap::from([(monomial, BigInt::from(1))]),
            bounded_summary: BoundedSummary::missing(),
        });
        let analyzed = Arc::new(AnalyzedValue {
            semantic,
            exact_nf: Some(Arc::clone(&normal_form)),
            coefficient_bound: NumericContract::Missing,
        });
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();

        normalizer.insert_value_cache(semantic.expression(), Arc::clone(&analyzed));
        normalizer.remaining_uses.insert(semantic.expression(), 1);
        let held = normalizer.owner_census();
        assert_eq!(held.cache_entries, 1);
        assert_eq!(held.cache_exact_terms, 1);
        assert_eq!(held.cache_exact_terms_peak, 1);
        assert_eq!(held.cache_largest_nf_terms_seen, 1);
        assert_eq!(held.monomial_retained_descriptor_slots, 1);
        assert_eq!(held.monomial_reachable_descriptor_slots, 1);

        normalizer.child_value(semantic.expression()).unwrap();
        let released = normalizer.owner_census();
        assert_eq!(released.cache_entries, 0);
        assert_eq!(released.cache_exact_terms, 0);
        assert_eq!(released.cache_exact_terms_peak, 1);
        assert_eq!(released.monomial_reachable_descriptor_slots, 0);
        assert_eq!(released.monomial_unreachable_descriptor_slots, 1);

        normalizer.insert_gadget_hold(semantic.expression(), Arc::clone(&normal_form));
        normalizer.insert_gadget_hold(semantic.expression(), Arc::clone(&normal_form));
        let gadget = normalizer.owner_census();
        assert_eq!(gadget.gadget_entries, 1);
        assert_eq!(gadget.gadget_exact_terms, 1);
        assert_eq!(gadget.gadget_exact_terms_peak, 1);
        assert_eq!(gadget.gadget_largest_nf_terms_seen, 1);
        normalizer.insert_value_cache(semantic.expression(), analyzed);
        let shared = normalizer.owner_census();
        assert_eq!(shared.cache_exact_terms, 1);
        assert_eq!(shared.gadget_exact_terms, 1);
        assert_eq!(shared.monomial_reachable_descriptor_slots, 1);
        normalizer.clear_gadget_holds();
        let cleared = normalizer.owner_census();
        assert_eq!(cleared.gadget_exact_terms, 0);
        assert_eq!(cleared.cache_exact_terms, 1);
    }

    #[test]
    fn monomial_gc_preserves_cache_gadget_and_canonical_rhs_owners() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let root = source(&mut expressions);
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let cache_id = monomials.intern(&expressions, &programs, &[], &[semantic]).unwrap();
        let gadget_id =
            monomials.intern(&expressions, &programs, &[], &[semantic, semantic]).unwrap();
        let canonical_id = monomials
            .intern(&expressions, &programs, &[], &[semantic, semantic, semantic])
            .unwrap();
        let runtime_id = monomials
            .intern(&expressions, &programs, &[], &[semantic, semantic, semantic, semantic])
            .unwrap();
        let closed_id = monomials
            .intern(
                &expressions,
                &programs,
                &[],
                &[semantic, semantic, semantic, semantic, semantic],
            )
            .unwrap();
        let suspended_id = monomials
            .intern(
                &expressions,
                &programs,
                &[],
                &[semantic, semantic, semantic, semantic, semantic, semantic],
            )
            .unwrap();
        let unowned_id = monomials
            .intern(
                &expressions,
                &programs,
                &[],
                &[semantic, semantic, semantic, semantic, semantic, semantic, semantic],
            )
            .unwrap();
        let nf = |id| {
            Arc::new(PolynomialNF {
                exact_terms: BTreeMap::from([(id, BigInt::from(1))]),
                bounded_summary: BoundedSummary::missing(),
            })
        };
        let mut canonical = NormalizationCache::new();
        let rhs = canonical.intern_arc(nf(canonical_id)).unwrap();
        let family = programs
            .generated_family(
                &mut expressions,
                ProgramSignature {
                    inputs: Box::new([ProgramInput {
                        value_type: ResolvedValueType::Int,
                        trusted_index_range: Some(TrustedIndexRange {
                            minimum: 0,
                            maximum_exclusive: 1,
                        }),
                    }]),
                    output: ResolvedValueType::Matrix(matrix_type()),
                },
                root,
            )
            .unwrap();
        let dispatch = UniversalDispatchKey {
            preimage_family: family,
            preimage_source: SamplerSourceContract { expression: root },
            matrix_type: matrix_type(),
            trapdoor_source: TrapdoorSourceContract { expression: root },
        };
        let mut relations = RelationRegistry::new();
        relations
            .register_closed(
                CanonicalLhsKey { layout: None, monomial: closed_id },
                rhs,
                &closed_relation_authority(&matrix_type(), root, root),
            )
            .unwrap();
        let generation = relations.freeze();
        canonical.runtime_insert(
            RuntimeSpecializationKey { dispatch, index: semantic, generation },
            BTreeMap::from([(
                CanonicalLhsKey { layout: None, monomial: runtime_id },
                BTreeSet::from([rhs]),
            )]),
        );
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_relations(&relations, &mut canonical);
        normalizer.protected_monomial_prefix = 0;
        normalizer.monomial_gc_allocation_threshold_bytes = 1;
        normalizer.normalization_depth = 1;
        normalizer.insert_value_cache(
            semantic.expression(),
            Arc::new(AnalyzedValue {
                semantic,
                exact_nf: Some(nf(cache_id)),
                coefficient_bound: NumericContract::Missing,
            }),
        );
        normalizer.insert_gadget_hold(semantic.expression(), nf(gadget_id));
        normalizer.suspended_owner_roots.push(suspended_id);
        normalizer.sweep_monomials_at_node_commit().unwrap();
        assert_eq!(normalizer.gc_counters.last_protected_prefix.descriptor_slots, 0);
        assert_eq!(normalizer.gc_counters.last_value_cache.descriptor_slots, 1);
        assert_eq!(normalizer.gc_counters.last_gadget.descriptor_slots, 1);
        assert_eq!(normalizer.gc_counters.last_canonical_runtime.descriptor_slots, 2);
        assert_eq!(normalizer.gc_counters.last_closed.descriptor_slots, 1);
        assert_eq!(normalizer.gc_counters.last_suspended.descriptor_slots, 1);
        assert_eq!(normalizer.gc_counters.last_occupied_slots, 6);
        assert_eq!(normalizer.gc_counters.last_reclaimed_slots, 1);
        assert_eq!(normalizer.gc_counters.value_cache_top8_len, 0);
        assert!(normalizer.monomials.descriptor(cache_id).is_ok());
        assert!(normalizer.monomials.descriptor(gadget_id).is_ok());
        assert!(normalizer.monomials.descriptor(canonical_id).is_ok());
        assert!(normalizer.monomials.descriptor(runtime_id).is_ok());
        assert!(normalizer.monomials.descriptor(closed_id).is_ok());
        assert!(normalizer.monomials.descriptor(suspended_id).is_ok());
        assert!(matches!(
            normalizer.monomials.descriptor(unowned_id),
            Err(MonomialError::CollectedMonomialId { .. })
        ));
        assert_eq!(normalizer.monomials.occupied_len(), 6);
    }

    #[test]
    fn normalization_operator_category_names_every_matrix_operation() {
        let output = matrix_type();
        let layout = MatrixLayout::row_major(output.rows, output.columns);
        let cases = [
            (ValueOperator::Matrix(MatrixOperation::Add), "add"),
            (ValueOperator::Matrix(MatrixOperation::Subtract), "subtract"),
            (ValueOperator::Matrix(MatrixOperation::Multiply), "multiply"),
            (ValueOperator::Matrix(MatrixOperation::Negate), "negate"),
            (ValueOperator::Matrix(MatrixOperation::Scale), "scale"),
            (ValueOperator::Matrix(MatrixOperation::Transpose), "transpose"),
            (
                ValueOperator::Matrix(MatrixOperation::Slice {
                    row_start: 0,
                    row_end_exclusive: output.rows,
                    column_start: 0,
                    column_end_exclusive: output.columns,
                    layout: layout.clone(),
                }),
                "slice",
            ),
            (
                ValueOperator::Matrix(MatrixOperation::IndexedSlice {
                    output: output.clone(),
                    layout: layout.clone(),
                }),
                "indexed_slice",
            ),
            (
                ValueOperator::Matrix(MatrixOperation::View {
                    output: output.clone(),
                    layout: layout.clone(),
                }),
                "view",
            ),
            (
                ValueOperator::Matrix(MatrixOperation::Concat {
                    axis: 0,
                    output: output.clone(),
                    layout: layout.clone(),
                }),
                "concat",
            ),
            (
                ValueOperator::Matrix(MatrixOperation::Tensor {
                    output: output.clone(),
                    left_layout: layout.clone(),
                    right_layout: layout.clone(),
                    output_layout: layout.clone(),
                }),
                "tensor",
            ),
            (
                ValueOperator::Matrix(MatrixOperation::CrtRecompose {
                    plaintext_moduli: vec![BigUint::from(17_u8)].into_boxed_slice(),
                    reconstruction_coefficients: vec![BigInt::from(1_u8)].into_boxed_slice(),
                    output: output.clone(),
                }),
                "crt_recompose",
            ),
            (
                ValueOperator::Matrix(MatrixOperation::ExtractCoefficient { row: 0, column: 0 }),
                "extract_coefficient",
            ),
            (
                ValueOperator::Matrix(MatrixOperation::LiftConstantPolynomial {
                    output,
                    coefficient_bits: 1,
                }),
                "lift_constant_polynomial",
            ),
        ];
        for (operator, expected) in cases {
            assert_eq!(normalization_operator_category(&operator), expected);
        }
    }

    #[test]
    fn monomial_gc_watchdog_ranks_value_cache_top8_stably_without_semantic_payloads() {
        let mut expressions = ExprArena::new();
        let cache_expressions = (0_u64..10)
            .map(|event| source_with(&mut expressions, matrix_type(), 83_100 + event))
            .collect::<Vec<_>>();
        let mut programs = ProgramArena::new();
        let (facts, mut monomials, semantic) =
            setup(&mut expressions, &mut programs, cache_expressions[0]);
        let ids = (1..=9)
            .map(|count| {
                monomials.intern(&expressions, &programs, &[], &vec![semantic; count]).unwrap()
            })
            .collect::<Vec<_>>();
        let term_counts = [1_usize, 9, 3, 8, 8, 5, 4, 2, 7, 6];
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        normalizer.monomial_gc_allocation_threshold_bytes = 1;
        normalizer.normalization_depth = 1;
        normalizer.watchdog = DiagnosticWatchdog::start(1, Duration::from_secs(60));
        for (index, (&expression, &term_count)) in
            cache_expressions.iter().zip(term_counts.iter()).enumerate()
        {
            let exact_terms =
                ids[..term_count].iter().copied().map(|id| (id, BigInt::from(1))).collect();
            normalizer.insert_value_cache(
                expression,
                Arc::new(AnalyzedValue {
                    semantic,
                    exact_nf: Some(Arc::new(PolynomialNF {
                        exact_terms,
                        bounded_summary: BoundedSummary::missing(),
                    })),
                    coefficient_bound: NumericContract::Missing,
                }),
            );
            normalizer.remaining_uses.insert(expression, index + 1);
        }

        normalizer.sweep_monomials_at_node_commit().unwrap();
        let gc = normalizer.gc_counters;
        assert_eq!(gc.value_cache_entries, 10);
        assert_eq!(gc.value_cache_exact_term_refs, 53);
        assert_eq!(gc.value_cache_top8_len, 8);
        assert_eq!(gc.value_cache_top8_exact_term_refs, 50);
        let ranked = &gc.value_cache_top8;
        assert_eq!(ranked.map(|entry| entry.term_count), [9, 8, 8, 7, 6, 5, 4, 3]);
        assert!(ranked[1].expression_slot < ranked[2].expression_slot);
        for entry in ranked {
            assert_eq!(entry.operator_category, "sample");
            let position = cache_expressions
                .iter()
                .position(|expression| u64::from(expression.slot()) == entry.expression_slot)
                .unwrap();
            assert_eq!(entry.remaining_uses, u64::try_from(position + 1).unwrap());
        }
        let progress = *normalizer
            .watchdog
            .as_ref()
            .unwrap()
            .shared
            .progress
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(progress.gc, gc);
        normalizer.watchdog.as_mut().unwrap().finish(false);
    }

    #[test]
    fn value_cache_top8_reports_selected_producer_input_nf_sizes() {
        let mut expressions = ExprArena::new();
        let left = source_with(&mut expressions, matrix_type(), 83_200);
        let right = source_with(&mut expressions, matrix_type(), 83_201);
        let sum =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[left, right]).unwrap();
        let mut programs = ProgramArena::new();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, sum);
        let ids = (1..=9)
            .map(|count| {
                monomials.intern(&expressions, &programs, &[], &vec![semantic; count]).unwrap()
            })
            .collect::<Vec<_>>();
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        for (expression, term_count) in [(left, 2_usize), (right, 3), (sum, 9)] {
            normalizer.insert_value_cache(
                expression,
                Arc::new(AnalyzedValue {
                    semantic,
                    exact_nf: Some(Arc::new(PolynomialNF {
                        exact_terms: ids[..term_count]
                            .iter()
                            .copied()
                            .map(|id| (id, BigInt::from(1_u8)))
                            .collect(),
                        bounded_summary: BoundedSummary::missing(),
                    })),
                    coefficient_bound: NumericContract::Missing,
                }),
            );
        }

        let top8 = normalizer.diagnostic_value_cache_top8();
        assert_eq!(top8.len, 3);
        let producer = top8.top[0];
        assert_eq!(producer.expression_slot, u64::from(sum.slot()));
        assert_eq!(producer.operator_category, "add");
        assert_eq!(producer.producer_input_count, 2);
        assert_eq!(producer.cached_input_exact_term_refs_sum, 5);
        assert_eq!(producer.cached_input_exact_term_refs_max, 3);
        for source in &top8.top[1..3] {
            assert_eq!(source.producer_input_count, 0);
            assert_eq!(source.cached_input_exact_term_refs_sum, 0);
            assert_eq!(source.cached_input_exact_term_refs_max, 0);
        }
    }

    #[test]
    fn product_deferral_diagnostic_classifies_every_rejection_without_semantic_branching() {
        let mut expressions = ExprArena::new();
        let ordinary = matrix_type();
        let scalar =
            ResolvedMatrixType::new(ordinary.modulus.clone(), ordinary.ring_dimension, 1, 1)
                .unwrap();
        let mut event = 84_000_u64;
        let mut source_pair = |expressions: &mut ExprArena| {
            event += 2;
            (
                source_with(expressions, ordinary.clone(), event - 1),
                source_with(expressions, ordinary.clone(), event),
            )
        };
        let make_product = |expressions: &mut ExprArena, left, right| {
            expressions.intern_matrix_transform(MatrixOperation::Multiply, &[left, right]).unwrap()
        };
        let (eligible_left, eligible_right) = source_pair(&mut expressions);
        let eligible = make_product(&mut expressions, eligible_left, eligible_right);
        let (nonadd_left, nonadd_right) = source_pair(&mut expressions);
        let nonadd = make_product(&mut expressions, nonadd_left, nonadd_right);
        let (structural_left, structural_right) = source_pair(&mut expressions);
        let structural = make_product(&mut expressions, structural_left, structural_right);
        let gadget = matrix_source(
            &mut expressions,
            "diagnostic-gadget",
            ordinary.clone(),
            Some((2, false)),
        );
        let gadget_right = source_with(&mut expressions, ordinary.clone(), 84_100);
        let gadget_product = make_product(&mut expressions, gadget, gadget_right);
        let (root_left, root_right) = source_pair(&mut expressions);
        let root_product = make_product(&mut expressions, root_left, root_right);
        let (missing_left, missing_right) = source_pair(&mut expressions);
        let missing = make_product(&mut expressions, missing_left, missing_right);
        let scalar_left = source_with(&mut expressions, scalar.clone(), 84_200);
        let scalar_right = source_with(&mut expressions, scalar.clone(), 84_201);
        let matrix_left = source_with(&mut expressions, ordinary.clone(), 84_202);
        let matrix_right = source_with(&mut expressions, ordinary, 84_203);
        let left_scalar = make_product(&mut expressions, scalar_left, matrix_right);
        let right_scalar = make_product(&mut expressions, matrix_left, scalar_right);
        let both_scalar = make_product(&mut expressions, scalar_left, scalar_right);

        let mut programs = ProgramArena::new();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root_product);
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let exact_zero = Arc::new(PolynomialNF::zero());
        for input in [eligible_left, eligible_right, gadget, gadget_right] {
            normalizer.insert_value_cache(
                input,
                Arc::new(AnalyzedValue {
                    semantic,
                    exact_nf: Some(Arc::clone(&exact_zero)),
                    coefficient_bound: NumericContract::Known(CoefficientBound::ExactZero),
                }),
            );
        }
        normalizer
            .diagnostic_product_consumers
            .insert(eligible, DiagnosticProductConsumerCounts { add_sub: 1, ..Default::default() });
        normalizer
            .diagnostic_product_consumers
            .insert(nonadd, DiagnosticProductConsumerCounts { multiply: 1, ..Default::default() });
        normalizer.diagnostic_product_consumers.insert(
            structural,
            DiagnosticProductConsumerCounts { structural: 1, ..Default::default() },
        );
        normalizer.diagnostic_product_root = Some(root_product);
        let no_exact = Arc::new(AnalyzedValue {
            semantic,
            exact_nf: None,
            coefficient_bound: NumericContract::Missing,
        });
        for expression in [
            eligible,
            nonadd,
            structural,
            gadget_product,
            root_product,
            missing,
            left_scalar,
            right_scalar,
            both_scalar,
        ] {
            let node = normalizer.expressions.node_arc(expression).unwrap();
            let children = node
                .inputs
                .iter()
                .map(|input| {
                    normalizer.cache.get(input).cloned().unwrap_or_else(|| Arc::clone(&no_exact))
                })
                .collect::<Vec<_>>();
            let snapshot = normalizer.diagnostic_product_evaluation_snapshot(&node, &children);
            normalizer.diagnostic_product_evaluations.insert(expression, snapshot);
        }
        // Classification must be based on the immutable evaluation snapshots, not on whichever
        // operand values happen to remain in the last-use cache when a later sweep reports top8.
        normalizer.clear_value_cache();

        let classify = |normalizer: &Normalizer<'_>, expression| {
            let node = normalizer.expressions.node(expression).unwrap();
            let mut entry = DiagnosticValueCacheTopEntry::default();
            normalizer.populate_product_deferral_diagnostic(expression, node, &mut entry);
            entry
        };
        assert_eq!(classify(&normalizer, eligible).multiply_deferral_rejection, "eligible");
        assert_eq!(
            classify(&normalizer, nonadd).multiply_deferral_rejection,
            "non_additive_consumer"
        );
        assert_eq!(
            classify(&normalizer, structural).multiply_deferral_rejection,
            "structural_hold"
        );
        assert_eq!(
            classify(&normalizer, gadget_product).multiply_deferral_rejection,
            "gadget_boundary"
        );
        assert_eq!(classify(&normalizer, root_product).multiply_deferral_rejection, "root");
        assert_eq!(
            classify(&normalizer, missing).multiply_deferral_rejection,
            "missing_exact_operand"
        );
        assert_eq!(
            classify(&normalizer, left_scalar).multiply_scalar_classification,
            "scalar_left"
        );
        assert_eq!(
            classify(&normalizer, right_scalar).multiply_scalar_classification,
            "scalar_right"
        );
        let both = classify(&normalizer, both_scalar);
        assert_eq!(both.multiply_scalar_classification, "scalar_both");
        for scalar_product in [left_scalar, right_scalar, both_scalar] {
            assert_eq!(
                classify(&normalizer, scalar_product).multiply_deferral_rejection,
                "scalar_shape"
            );
        }
    }

    #[test]
    fn product_deferral_diagnostic_is_published_in_top8_with_additive_leaf_identity() {
        let mut expressions = ExprArena::new();
        let left = source_with(&mut expressions, matrix_type(), 84_300);
        let right = source_with(&mut expressions, matrix_type(), 84_301);
        let product =
            expressions.intern_matrix_transform(MatrixOperation::Multiply, &[left, right]).unwrap();
        let other = source_with(&mut expressions, matrix_type(), 84_302);
        let root =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[product, other]).unwrap();
        let mut programs = ProgramArena::new();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let ids = (1..=3)
            .map(|count| {
                monomials.intern(&expressions, &programs, &[], &vec![semantic; count]).unwrap()
            })
            .collect::<Vec<_>>();
        let nf = |terms: usize| {
            Arc::new(PolynomialNF {
                exact_terms: ids[..terms]
                    .iter()
                    .copied()
                    .map(|id| (id, BigInt::from(1_u8)))
                    .collect(),
                bounded_summary: BoundedSummary::missing(),
            })
        };
        let product_nf = nf(3);
        let other_nf = nf(1);
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        for (expression, normal_form) in [
            (left, nf(1)),
            (right, nf(1)),
            (product, Arc::clone(&product_nf)),
            (other, Arc::clone(&other_nf)),
        ] {
            normalizer.insert_value_cache(
                expression,
                Arc::new(AnalyzedValue {
                    semantic,
                    exact_nf: Some(normal_form),
                    coefficient_bound: NumericContract::Missing,
                }),
            );
        }
        normalizer
            .diagnostic_product_consumers
            .insert(product, DiagnosticProductConsumerCounts { add_sub: 1, ..Default::default() });
        let product_node = normalizer.expressions.node_arc(product).unwrap();
        let product_children = product_node
            .inputs
            .iter()
            .map(|input| normalizer.cache.get(input).cloned().unwrap())
            .collect::<Vec<_>>();
        let product_snapshot =
            normalizer.diagnostic_product_evaluation_snapshot(&product_node, &product_children);
        normalizer.diagnostic_product_evaluations.insert(product, product_snapshot);
        let product_leaf =
            normalizer.materialized_exact_state(product, Arc::clone(&product_nf)).unwrap();
        let other_leaf = normalizer.materialized_exact_state(other, other_nf).unwrap();
        let plan = normalizer.new_additive_plan(root, product_leaf, other_leaf, false).unwrap();
        normalizer.exact_plans.insert(root, plan);
        normalizer.remove_value_cache(left);
        normalizer.remove_value_cache(right);

        let top8 = normalizer.diagnostic_value_cache_top8();
        let product = top8.top[..usize::from(top8.len)]
            .iter()
            .find(|entry| entry.expression_slot == u64::from(product.slot()))
            .copied()
            .unwrap();
        assert_eq!(product.multiply_scalar_classification, "ordinary");
        assert_eq!(
            (
                product.multiply_left_rows,
                product.multiply_left_columns,
                product.multiply_right_rows,
                product.multiply_right_columns,
            ),
            (2, 2, 2, 2)
        );
        assert_eq!(product.multiply_add_sub_consumers, 1);
        assert_eq!(product.multiply_consumers, 0);
        assert_eq!(product.multiply_deferral_rejection, "eligible");
        assert!(product.additive_materialized_leaf);
        assert_eq!(product.cached_input_exact_term_refs_sum, 0);
        assert_eq!(product.cached_input_exact_term_refs_max, 0);
    }

    #[test]
    fn product_evaluation_snapshot_survives_operand_last_use_for_later_classification() {
        let mut expressions = ExprArena::new();
        let left = source_with(&mut expressions, matrix_type(), 84_310);
        let right = source_with(&mut expressions, matrix_type(), 84_311);
        let product =
            expressions.intern_matrix_transform(MatrixOperation::Multiply, &[left, right]).unwrap();
        let other = source_with(&mut expressions, matrix_type(), 84_312);
        let root =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[product, other]).unwrap();
        let mut programs = ProgramArena::new();
        let (facts, mut monomials, root_semantic) = setup(&mut expressions, &mut programs, root);
        let left_semantic = programs.scoped(&expressions, root_semantic.program(), left).unwrap();
        let right_semantic = programs.scoped(&expressions, root_semantic.program(), right).unwrap();
        let left_id = monomials.intern(&expressions, &programs, &[], &[left_semantic]).unwrap();
        let right_id = monomials.intern(&expressions, &programs, &[], &[right_semantic]).unwrap();
        let atom_value = |semantic, monomial| {
            Arc::new(AnalyzedValue {
                semantic,
                exact_nf: Some(Arc::new(PolynomialNF {
                    exact_terms: BTreeMap::from([(monomial, BigInt::from(1_u8))]),
                    bounded_summary: BoundedSummary::missing(),
                })),
                coefficient_bound: NumericContract::Missing,
            })
        };
        let mut proof =
            expressions.scope_proof(root_semantic.program(), root_semantic.expression()).unwrap();
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        normalizer.watchdog = DiagnosticWatchdog::start(23, Duration::from_secs(60));
        normalizer.insert_value_cache(left, atom_value(left_semantic, left_id));
        normalizer.insert_value_cache(right, atom_value(right_semantic, right_id));
        normalizer.remaining_uses.insert(left, 1);
        normalizer.remaining_uses.insert(right, 1);
        normalizer
            .diagnostic_product_consumers
            .insert(product, DiagnosticProductConsumerCounts { add_sub: 1, ..Default::default() });

        let product_node = normalizer.expressions.node_arc(product).unwrap();
        let evaluated =
            normalizer.evaluate_node(&mut proof, product, product_node.as_ref()).unwrap();
        assert!(evaluated.exact_nf.is_some());
        assert!(!normalizer.cache.contains_key(&left));
        assert!(!normalizer.cache.contains_key(&right));

        let mut entry = DiagnosticValueCacheTopEntry::default();
        normalizer.populate_product_deferral_diagnostic(product, &product_node, &mut entry);
        assert_eq!(entry.multiply_deferral_rejection, "eligible");
        normalizer
            .diagnostic_product_consumers
            .insert(product, DiagnosticProductConsumerCounts { multiply: 1, ..Default::default() });
        normalizer.populate_product_deferral_diagnostic(product, &product_node, &mut entry);
        assert_eq!(entry.multiply_deferral_rejection, "non_additive_consumer");
        normalizer.watchdog.as_mut().unwrap().finish(false);
    }

    #[test]
    fn wrapped_gadget_endpoint_snapshot_survives_operand_last_use() {
        let mut expressions = ExprArena::new();
        let input_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 1).unwrap();
        let decomposition_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 3, 1).unwrap();
        let gadget_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 3).unwrap();
        let (gadget, decomposition, input) = gadget_product(
            &mut expressions,
            false,
            3,
            gadget_type.clone(),
            decomposition_type.clone(),
            input_type.clone(),
            Some((2, false)),
        );
        let gadget_view = expressions
            .intern_matrix_transform(
                MatrixOperation::View {
                    output: gadget_type.clone(),
                    layout: MatrixLayout::row_major(gadget_type.rows, gadget_type.columns),
                },
                &[gadget],
            )
            .unwrap();
        let decomposition_view = expressions
            .intern_matrix_transform(
                MatrixOperation::View {
                    output: decomposition_type.clone(),
                    layout: MatrixLayout::row_major(
                        decomposition_type.rows,
                        decomposition_type.columns,
                    ),
                },
                &[decomposition],
            )
            .unwrap();
        let product = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[gadget_view, decomposition_view])
            .unwrap();
        let registry =
            recomposition_registry(gadget_type, decomposition_type, input_type, false, 3);
        let mut programs = ProgramArena::new();
        let (mut facts, mut monomials, product_semantic) =
            setup(&mut expressions, &mut programs, product);
        for expression in [gadget, decomposition, input] {
            insert_matrix_layout_fact(&expressions, &mut facts, expression, false);
        }
        let gadget_semantic =
            programs.scoped(&expressions, product_semantic.program(), gadget).unwrap();
        let decomposition_semantic =
            programs.scoped(&expressions, product_semantic.program(), decomposition).unwrap();
        let gadget_id = monomials.intern(&expressions, &programs, &[], &[gadget_semantic]).unwrap();
        let decomposition_id =
            monomials.intern(&expressions, &programs, &[], &[decomposition_semantic]).unwrap();
        let atom_value = |semantic, monomial| {
            Arc::new(AnalyzedValue {
                semantic,
                exact_nf: Some(Arc::new(PolynomialNF {
                    exact_terms: BTreeMap::from([(monomial, BigInt::from(1_u8))]),
                    bounded_summary: BoundedSummary::missing(),
                })),
                coefficient_bound: NumericContract::Missing,
            })
        };
        let mut proof = expressions
            .scope_proof(product_semantic.program(), product_semantic.expression())
            .unwrap();
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_gadget_recompositions(&registry);
        normalizer.watchdog = DiagnosticWatchdog::start(24, Duration::from_secs(60));
        normalizer.insert_value_cache(gadget, atom_value(gadget_semantic, gadget_id));
        normalizer.insert_value_cache(
            decomposition,
            atom_value(decomposition_semantic, decomposition_id),
        );
        for (source, view) in [(gadget, gadget_view), (decomposition, decomposition_view)] {
            normalizer.remaining_uses.insert(source, 1);
            let node = normalizer.expressions.node_arc(view).unwrap();
            let value = normalizer.evaluate_node(&mut proof, view, node.as_ref()).unwrap();
            normalizer.insert_value_cache(view, Arc::new(value));
        }
        normalizer.remaining_uses.insert(gadget_view, 1);
        normalizer.remaining_uses.insert(decomposition_view, 1);
        let product_node = normalizer.expressions.node_arc(product).unwrap();
        let evaluated =
            normalizer.evaluate_node(&mut proof, product, product_node.as_ref()).unwrap();
        assert!(evaluated.exact_nf.is_some());
        assert!(!normalizer.cache.contains_key(&gadget_view));
        assert!(!normalizer.cache.contains_key(&decomposition_view));

        let mut entry = DiagnosticValueCacheTopEntry::default();
        normalizer.populate_product_deferral_diagnostic(product, &product_node, &mut entry);
        assert_eq!(entry.multiply_deferral_rejection, "gadget_boundary");
        normalizer.watchdog.as_mut().unwrap().finish(false);
    }

    #[test]
    fn product_consumer_diagnostic_is_collected_from_real_reverse_edges_only_with_watchdog() {
        let mut expressions = ExprArena::new();
        let left = source_with(&mut expressions, matrix_type(), 84_400);
        let right = source_with(&mut expressions, matrix_type(), 84_401);
        let product =
            expressions.intern_matrix_transform(MatrixOperation::Multiply, &[left, right]).unwrap();
        let add =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[product, left]).unwrap();
        let multiply = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[product, right])
            .unwrap();
        let root =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[add, multiply]).unwrap();
        let mut programs = ProgramArena::new();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        normalizer.watchdog = DiagnosticWatchdog::start(22, Duration::from_secs(60));
        normalizer.compute_use_counts(root).unwrap();
        assert_eq!(
            normalizer.diagnostic_product_consumers.get(&product),
            Some(&DiagnosticProductConsumerCounts {
                add_sub: 1,
                multiply: 1,
                structural: 0,
                root_other: 0,
            })
        );
        assert_eq!(normalizer.diagnostic_product_root, Some(root));
        normalizer.watchdog.as_mut().unwrap().finish(false);

        normalizer.watchdog = None;
        normalizer.diagnostic_product_consumers.clear();
        normalizer.diagnostic_product_root = None;
        normalizer.remaining_uses.clear();
        normalizer.compute_use_counts(root).unwrap();
        assert!(normalizer.diagnostic_product_consumers.is_empty());
        assert_eq!(normalizer.diagnostic_product_root, None);
        normalizer.diagnostic_product_evaluations.insert(
            product,
            DiagnosticProductEvaluationSnapshot {
                had_left_exact: true,
                had_right_exact: true,
                ..DiagnosticProductEvaluationSnapshot::default()
            },
        );
        normalizer.diagnostic_product_root = Some(product);
        normalizer.normalize(semantic).unwrap();
        assert!(normalizer.diagnostic_product_evaluations.is_empty());
        assert_eq!(normalizer.diagnostic_product_root, None);
    }

    #[test]
    fn monomial_gc_reports_value_cache_before_exact_plan_without_double_attribution() {
        let mut expressions = ExprArena::new();
        let root = source_with(&mut expressions, matrix_type(), 84_500);
        let mut programs = ProgramArena::new();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let cache_id = monomials.intern(&expressions, &programs, &[], &[semantic]).unwrap();
        let plan_id =
            monomials.intern(&expressions, &programs, &[], &[semantic, semantic]).unwrap();
        let dead_id = monomials
            .intern(&expressions, &programs, &[], &[semantic, semantic, semantic])
            .unwrap();
        let nf = |id| {
            Arc::new(PolynomialNF {
                exact_terms: BTreeMap::from([(id, BigInt::from(1_u8))]),
                bounded_summary: BoundedSummary::missing(),
            })
        };
        let cache_nf = nf(cache_id);
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        normalizer.protected_monomial_prefix = 0;
        normalizer.normalization_depth = 1;
        normalizer.monomial_gc_allocation_threshold_bytes = 0;
        normalizer.insert_value_cache(
            root,
            Arc::new(AnalyzedValue {
                semantic,
                exact_nf: Some(Arc::clone(&cache_nf)),
                coefficient_bound: NumericContract::Missing,
            }),
        );
        let left = normalizer.materialized_exact_state(root, cache_nf).unwrap();
        let right = normalizer.materialized_exact_state(root, nf(plan_id)).unwrap();
        let plan = normalizer.new_additive_plan(root, left, right, false).unwrap();
        normalizer.exact_plans.insert(root, plan);
        normalizer.sweep_monomials_at_node_commit().unwrap();
        assert_eq!(normalizer.gc_counters.last_value_cache.descriptor_slots, 1);
        assert_eq!(normalizer.gc_counters.last_exact_plan.descriptor_slots, 1);
        assert_eq!(normalizer.gc_counters.last_occupied_slots, 2);
        assert_eq!(normalizer.gc_counters.last_reclaimed_slots, 1);
        assert!(matches!(
            normalizer.monomials.descriptor(dead_id),
            Err(MonomialError::CollectedMonomialId { .. })
        ));
    }

    #[test]
    fn monomial_gc_projects_global_relation_roots_to_the_local_arena() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let root_a = source_with(&mut expressions, matrix_type(), 83_001);
        let (_, mut arena_a, semantic_a) = setup(&mut expressions, &mut programs, root_a);
        let canonical_a =
            arena_a.intern(&expressions, &programs, &[], &[semantic_a, semantic_a]).unwrap();
        let runtime_a = arena_a
            .intern(&expressions, &programs, &[], &[semantic_a, semantic_a, semantic_a])
            .unwrap();
        let closed_a = arena_a
            .intern(&expressions, &programs, &[], &[semantic_a, semantic_a, semantic_a, semantic_a])
            .unwrap();
        let rhs_nf = Arc::new(PolynomialNF {
            exact_terms: BTreeMap::from([(canonical_a, BigInt::from(1))]),
            bounded_summary: BoundedSummary::missing(),
        });
        let mut cache = NormalizationCache::new();
        let rhs = cache.intern_arc(rhs_nf).unwrap();
        let family_a = programs
            .generated_family(
                &mut expressions,
                ProgramSignature {
                    inputs: Box::new([ProgramInput {
                        value_type: ResolvedValueType::Int,
                        trusted_index_range: Some(TrustedIndexRange {
                            minimum: 0,
                            maximum_exclusive: 1,
                        }),
                    }]),
                    output: ResolvedValueType::Matrix(matrix_type()),
                },
                root_a,
            )
            .unwrap();
        let dispatch = UniversalDispatchKey {
            preimage_family: family_a,
            preimage_source: SamplerSourceContract { expression: root_a },
            matrix_type: matrix_type(),
            trapdoor_source: TrapdoorSourceContract { expression: root_a },
        };
        let mut relations = RelationRegistry::new();
        relations
            .register_closed(
                CanonicalLhsKey { layout: None, monomial: closed_a },
                rhs,
                &closed_relation_authority(&matrix_type(), root_a, root_a),
            )
            .unwrap();
        let generation = relations.freeze();
        cache.runtime_insert(
            RuntimeSpecializationKey { dispatch, index: semantic_a, generation },
            BTreeMap::from([(
                CanonicalLhsKey { layout: None, monomial: runtime_a },
                BTreeSet::from([rhs]),
            )]),
        );

        let root_b = source_with(&mut expressions, matrix_type(), 83_002);
        let (facts_b, mut arena_b, semantic_b) = setup(&mut expressions, &mut programs, root_b);
        let local_live = arena_b.intern(&expressions, &programs, &[], &[semantic_b]).unwrap();
        let local_dead =
            arena_b.intern(&expressions, &programs, &[], &[semantic_b, semantic_b]).unwrap();
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts_b, &mut arena_b)
            .unwrap()
            .with_relations(&relations, &mut cache);
        normalizer.protected_monomial_prefix = 0;
        normalizer.monomial_gc_allocation_threshold_bytes = 1;
        normalizer.normalization_depth = 1;
        normalizer.insert_value_cache(
            semantic_b.expression(),
            Arc::new(AnalyzedValue {
                semantic: semantic_b,
                exact_nf: Some(Arc::new(PolynomialNF {
                    exact_terms: BTreeMap::from([(local_live, BigInt::from(1))]),
                    bounded_summary: BoundedSummary::missing(),
                })),
                coefficient_bound: NumericContract::Missing,
            }),
        );
        normalizer.sweep_monomials_at_node_commit().unwrap();
        assert!(normalizer.monomials.descriptor(local_live).is_ok());
        assert!(matches!(
            normalizer.monomials.descriptor(local_dead),
            Err(MonomialError::CollectedMonomialId { .. })
        ));
        assert!(arena_a.descriptor(canonical_a).is_ok());
        assert!(arena_a.descriptor(runtime_a).is_ok());
        assert!(arena_a.descriptor(closed_a).is_ok());
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
                    bounded_summary: BoundedSummary::missing(),
                })),
                coefficient_bound: NumericContract::Missing,
            }),
        );
        assert!(matches!(
            normalizer.sweep_monomials_at_node_commit(),
            Err(NormalizeError::Monomial(MonomialError::InvalidMonomialId { .. }))
        ));
        assert_eq!(normalizer.gc_counters, DiagnosticGcCounters::default());
        assert_eq!(normalizer.monomials.occupied_len(), occupied_before);
        assert!(normalizer.monomials.descriptor(local_live).is_ok());
        assert!(normalizer.monomials.descriptor(local_dead).is_ok());
    }

    #[test]
    fn monomial_gc_rejects_tombstoned_local_cache_root_without_telemetry_update() {
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
                    bounded_summary: BoundedSummary::missing(),
                })),
                coefficient_bound: NumericContract::Missing,
            }),
        );
        assert!(matches!(
            normalizer.sweep_monomials_at_node_commit(),
            Err(NormalizeError::Monomial(MonomialError::CollectedMonomialId { .. }))
        ));
        assert_eq!(normalizer.gc_counters, DiagnosticGcCounters::default());
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
                    bounded_summary: BoundedSummary::missing(),
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
    fn owner_census_restores_outer_live_counts_and_retains_nested_peaks() {
        let saved = NormalizerOwnerCounters {
            cache_exact_terms: 3,
            cache_exact_terms_peak: 5,
            cache_largest_nf_terms_seen: 4,
            gadget_exact_terms: 2,
            gadget_exact_terms_peak: 2,
            gadget_largest_nf_terms_seen: 2,
        };
        let nested = NormalizerOwnerCounters {
            cache_exact_terms: 7,
            cache_exact_terms_peak: 11,
            cache_largest_nf_terms_seen: 9,
            gadget_exact_terms: 6,
            gadget_exact_terms_peak: 8,
            gadget_largest_nf_terms_seen: 7,
        };
        let restored = Normalizer::restored_owner_counters(saved, nested);
        assert_eq!(restored.cache_exact_terms, 3);
        assert_eq!(restored.cache_exact_terms_peak, 11);
        assert_eq!(restored.cache_largest_nf_terms_seen, 9);
        assert_eq!(restored.gadget_exact_terms, 2);
        assert_eq!(restored.gadget_exact_terms_peak, 8);
        assert_eq!(restored.gadget_largest_nf_terms_seen, 7);
    }

    #[test]
    fn owner_census_marks_rolled_back_canonical_rhs_as_historical_unreachable() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let root = source_with(&mut expressions, matrix_type(), 7_081);
        let (_, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let monomial = monomials.intern(&expressions, &programs, &[], &[semantic]).unwrap();
        let rhs = PolynomialNF {
            exact_terms: BTreeMap::from([(monomial, BigInt::from(1))]),
            bounded_summary: BoundedSummary::missing(),
        };
        let mut cache = NormalizationCache::new();
        let checkpoint = cache.checkpoint();
        cache.intern(rhs).unwrap();
        let reachable = monomials.owner_census(cache.monomial_roots());
        assert_eq!(reachable.reachable_descriptor_slots, 1);
        assert_eq!(reachable.unreachable_descriptor_slots, 0);

        cache.rollback(checkpoint);
        let rolled_back = monomials.owner_census(cache.monomial_roots());
        assert_eq!(rolled_back.retained_descriptor_slots, 1);
        assert_eq!(rolled_back.reachable_descriptor_slots, 0);
        assert_eq!(rolled_back.unreachable_descriptor_slots, 1);
        assert_eq!(cache.owner_census().canonical_rhs_exact_terms_peak, 1);
    }

    #[test]
    fn suspended_owner_roots_keep_outer_nested_and_shared_ids_in_one_unique_union() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let root = source_with(&mut expressions, matrix_type(), 7_082);
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let outer = monomials.intern(&expressions, &programs, &[], &[semantic]).unwrap();
        let shared = monomials.combine_interned(semantic.program(), outer, outer).unwrap();
        let nested = monomials.combine_interned(semantic.program(), shared, outer).unwrap();
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let checkpoint = normalizer.suspended_owner_roots.len();
        normalizer.suspended_owner_roots.extend([outer, nested, shared, shared]);
        let nested_census = normalizer.owner_census_with_active([outer, shared]);
        assert_eq!(nested_census.monomial_reachable_descriptor_slots, 3);

        normalizer.suspended_owner_roots.truncate(checkpoint);
        let restored = normalizer.owner_census_with_active([outer, shared]);
        assert_eq!(restored.monomial_reachable_descriptor_slots, 2);
        assert_eq!(restored.monomial_unreachable_descriptor_slots, 1);
        assert!(normalizer.suspended_owner_roots.is_empty());
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
        assert_eq!(normal_form.exact_terms.len(), atoms.len());
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
        assert_eq!(retained, atoms.into_iter().collect());
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
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                    .unwrap()
                    .with_watchdog_override(true, Duration::from_secs(60));
            let value = normalizer.normalize_specialized_root(transformed.expression()).unwrap();
            assert!(normalizer.suspended_owner_roots.is_empty());
            value
        };
        assert_eq!(expressions.scope_proof_build_count(), 1);
        assert_eq!(specialized.semantic, public.semantic);
        assert_eq!(specialized.exact_nf, public.exact_nf);
        assert_eq!(specialized.coefficient_bound, public.coefficient_bound);
    }

    #[test]
    fn focused_subphase_trace_is_ordered_bounded_and_disabled_by_default() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let matrix = matrix_type();
        let constant = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(BigInt::from(1_u8))), Box::new([]))
            .unwrap();
        let lifted = expressions
            .intern_matrix_transform(
                MatrixOperation::LiftConstantPolynomial {
                    output: matrix.clone(),
                    coefficient_bits: 1,
                },
                &[constant],
            )
            .unwrap();
        let atom = gaussian_factor(&mut expressions, matrix, 704, 3);
        let root =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[atom, lifted]).unwrap();
        let focused_slot = u64::from(atom.slot());
        assert!(focused_slot > 0);
        let expected_tail = [
            (u64::from(lifted.slot()), "lift_constant_polynomial", 3),
            (focused_slot, "sample", 2),
            (u64::from(root.slot()), "add", 1),
        ];
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);

        let mut focused = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_trace_focus_call_override(Some(1))
            .with_trace_focus_expression_slot_override(Some(focused_slot))
            .with_trace_focus_tail_nodes_override(Some(3));
        focused.normalize_with_trace(semantic).unwrap();
        assert_eq!(focused.trace.focus_normalization_call, Some(1));
        assert_eq!(focused.trace.focus_expression_slot, Some(focused_slot));
        assert_eq!(focused.trace.focus_tail_nodes, Some(3));
        assert_eq!(focused.trace.node_start_history, expected_tail);
        assert_eq!(focused.trace.current_normalization_call, 1);
        assert_eq!(
            focused.trace.subphase_history,
            [
                "evaluate_matrix:bound",
                "evaluate_matrix:exact",
                "atom:scope_validate",
                "atom:monomial_intern",
                "atom:term_insert",
                "evaluate_node:relation_rewrite",
                "evaluate_node:zero_check",
                "evaluate_node:complete",
            ]
        );
        assert_eq!(focused.trace.subphase_lines_emitted, NORMALIZATION_TRACE_SUBPHASE_LINE_BUDGET);
        assert_eq!(
            usize::from(focused.trace.lines_emitted),
            2 + focused.trace.node_start_history.len() +
                usize::from(focused.trace.subphase_lines_emitted) +
                usize::from(focused.trace.post_lines_emitted)
        );
        assert!(focused.trace.post_lines_emitted <= NORMALIZATION_TRACE_POST_LINE_BUDGET);
        assert!(focused.trace.lines_emitted <= NORMALIZATION_TRACE_LINE_BUDGET);
        assert!(focused.trace.terminal_emitted);
        assert_eq!(focused.trace.current_subphase, "post:complete");
        drop(focused);

        let mut mismatched = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_trace_focus_call_override(Some(2))
            .with_trace_focus_expression_slot_override(Some(focused_slot))
            .with_trace_focus_tail_nodes_override(Some(3));
        mismatched.normalize_with_trace(semantic).unwrap();
        assert_eq!(mismatched.trace.focus_normalization_call, Some(2));
        assert_eq!(mismatched.trace.subphase_lines_emitted, 0);
        assert!(mismatched.trace.subphase_history.is_empty());
        assert!(mismatched.trace.node_start_history.is_empty());
        assert_eq!(mismatched.trace.lines_emitted, 2);
        drop(mismatched);

        let mut ordinary =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        ordinary.normalize(semantic).unwrap();
        assert_eq!(ordinary.trace.focus_normalization_call, None);
        assert_eq!(ordinary.trace.focus_expression_slot, None);
        assert_eq!(ordinary.trace.focus_tail_nodes, None);
        assert_eq!(ordinary.trace.subphase_lines_emitted, 0);
        assert!(ordinary.trace.subphase_history.is_empty());
        assert!(ordinary.trace.node_start_history.is_empty());
        assert_eq!(ordinary.trace.lines_emitted, 0);
        assert_eq!(normalization_trace_positive_u64("4078"), Some(4078));
        assert_eq!(normalization_trace_positive_u64("0"), None);
        assert_eq!(normalization_trace_expression_slot("0"), Some(0));
        assert_eq!(normalization_trace_expression_slot("4078"), Some(4078));
        for invalid in ["", "-1", "not-a-slot", "18446744073709551616"] {
            assert_eq!(normalization_trace_positive_u64(invalid), None);
            assert_eq!(normalization_trace_expression_slot(invalid), None);
        }
    }

    #[test]
    fn focused_caller_reservation_survives_noncritical_trace_saturation() {
        let mut trace = NormalizationTrace::new();
        trace.focus_normalization_call = Some(2);
        trace.focus_tail_nodes = Some(u64::MAX);
        trace.activate(1, 1, 0);
        trace.record_nested_normalization(1, 2);
        for _ in 0..16 {
            trace.enter_subphase("fixture:subphase");
            trace.enter_postphase("fixture:post");
            trace.emit_node_start(1);
            trace.emit("fixture_noncritical", 0, 1, false);
        }
        assert_eq!(
            trace.lines_emitted,
            NORMALIZATION_TRACE_LINE_BUDGET - 1 - NORMALIZATION_TRACE_CRITICAL_CALLER_RESERVE
        );
        assert_eq!(
            trace.critical_caller_lines_reserved,
            NORMALIZATION_TRACE_CRITICAL_CALLER_RESERVE
        );

        trace.record_completed_invocation(2);
        trace.current_normalization_call = 1;
        trace.enter_caller_phase("caller:nested_return", true);
        trace.enter_caller_phase("caller:bounds_merge_start", true);
        trace.enter_caller_phase("caller:bounds_merge_end", true);
        assert!(trace.claim_next_specialized_root());
        trace.enter_next_root_phase("next_root:preproof_start", true);
        trace.enter_next_root_phase("next_root:preproof_end", true);
        trace.enter_next_root_phase("next_root:normalize_proof_start", true);
        trace.enter_next_root_phase("next_root:normalize_proof_end", true);
        assert_eq!(
            trace.caller_history,
            [
                "caller:nested_return",
                "caller:bounds_merge_start",
                "caller:bounds_merge_end",
                "next_root:preproof_start",
                "next_root:preproof_end",
                "next_root:normalize_proof_start",
                "next_root:normalize_proof_end",
            ]
        );
        assert_eq!(trace.critical_caller_lines_reserved, 0);
        assert_eq!(trace.lines_emitted, NORMALIZATION_TRACE_LINE_BUDGET - 1);
        assert!(trace.emit("normalize_end", 1, 1, true));
        assert_eq!(trace.lines_emitted, NORMALIZATION_TRACE_LINE_BUDGET);
        assert!(trace.terminal_emitted);
    }

    #[test]
    fn watchdog_interval_parser_defaults_rejects_zero_and_clamps_large_values() {
        assert_eq!(
            normalization_watchdog_interval_from_value(None),
            NORMALIZATION_WATCHDOG_INTERVAL
        );
        for invalid in [Some(""), Some("0"), Some("-1"), Some("not-a-number")] {
            assert_eq!(
                normalization_watchdog_interval_from_value(invalid),
                NORMALIZATION_WATCHDOG_INTERVAL
            );
        }
        assert_eq!(normalization_watchdog_interval_from_value(Some("45")), Duration::from_secs(45));
        assert_eq!(
            normalization_watchdog_interval_from_value(Some("18446744073709551615")),
            Duration::from_secs(NORMALIZATION_WATCHDOG_MAX_INTERVAL_SECS)
        );
    }

    #[test]
    fn four_class_census_parser_is_explicit_opt_in() {
        for disabled in [None, Some(""), Some("0"), Some("false"), Some("False"), Some("no")] {
            assert!(!normalization_four_class_census_enabled_from_value(disabled));
        }
        for enabled in [Some("1"), Some("true"), Some("TRUE"), Some("yes"), Some("YES")] {
            assert!(normalization_four_class_census_enabled_from_value(enabled));
        }
    }

    #[test]
    fn watchdog_caps_lines_and_joins_after_progress_and_barrier_stall() {
        let mut watchdog = DiagnosticWatchdog::start(7, Duration::from_millis(1)).unwrap();
        watchdog.update(|progress| {
            progress.phase = DiagnosticPhase::NodeWalk;
            progress.expression_slot = 41;
            progress.operator = "multiply";
            progress.nodes_done = 5;
            progress.nodes_total = 10;
        });
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let release = Arc::clone(&barrier);
        let helper = thread::spawn(move || {
            thread::sleep(Duration::from_millis(8));
            release.wait();
        });
        barrier.wait();
        helper.join().unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        while watchdog.shared.events.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).len() <
            19 &&
            Instant::now() < deadline
        {
            thread::yield_now();
        }
        let shared = Arc::clone(&watchdog.shared);
        watchdog.finish(false);
        let events = shared.events.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone();
        assert_eq!(events.len(), 20);
        assert_eq!(events.first(), Some(&"watchdog_initial"));
        assert_eq!(events.last(), Some(&"watchdog_terminal"));
        assert_eq!(events.iter().filter(|event| **event == "watchdog_snapshot").count(), 18);
        let snapshots =
            shared.snapshots.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone();
        assert!(snapshots.iter().any(|snapshot| snapshot.phase == DiagnosticPhase::NodeWalk));
        for snapshot in
            snapshots.iter().filter(|snapshot| snapshot.phase == DiagnosticPhase::NodeWalk)
        {
            assert_eq!(
                (
                    snapshot.expression_slot,
                    snapshot.operator,
                    snapshot.nodes_done,
                    snapshot.nodes_total
                ),
                (41, "multiply", 5, 10)
            );
        }
        drop(watchdog);
        assert_eq!(Arc::strong_count(&shared), 1);
    }

    #[test]
    fn watchdog_quick_finish_never_enters_the_long_startup_timeout() {
        for error in [false, true] {
            let started = Instant::now();
            let mut watchdog = DiagnosticWatchdog::start(8, Duration::from_secs(60)).unwrap();
            watchdog.finish(error);
            assert!(
                started.elapsed() < Duration::from_secs(2),
                "quick finish must not wait for the reporter interval"
            );
            let events = watchdog
                .shared
                .events
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            assert_eq!(events, ["watchdog_initial", "watchdog_terminal"]);
            assert!(events.len() <= 32);
            let snapshots = watchdog
                .shared
                .snapshots
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            assert_eq!(snapshots.len(), 2);
            assert_eq!(
                snapshots.last().map(|snapshot| snapshot.phase),
                Some(if error { DiagnosticPhase::Error } else { DiagnosticPhase::CallReturn })
            );
        }
    }

    #[test]
    fn watchdog_restores_depth_three_parent_and_preserves_last_completed() {
        let mut watchdog = DiagnosticWatchdog::start(9, Duration::from_secs(60)).unwrap();
        let outer = watchdog.enter_call(1);
        watchdog.update(|progress| progress.phase = DiagnosticPhase::NodeWalk);
        let middle = watchdog.enter_call(2);
        watchdog.update(|progress| progress.phase = DiagnosticPhase::EvaluateNode);
        let inner = watchdog.enter_call(3);
        watchdog.complete_call(inner, false);
        let after_inner =
            *watchdog.shared.progress.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(after_inner.current_call, 2);
        assert_eq!(after_inner.last_completed, 3);
        assert_eq!(after_inner.depth, 2);
        assert_eq!(after_inner.phase, DiagnosticPhase::EvaluateNode);
        watchdog.complete_call(middle, false);
        let after_middle =
            *watchdog.shared.progress.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(after_middle.current_call, 1);
        assert_eq!(after_middle.last_completed, 2);
        assert_eq!(after_middle.depth, 1);
        assert_eq!(after_middle.phase, DiagnosticPhase::NodeWalk);
        watchdog.complete_call(outer, false);
        watchdog.finish(false);
    }

    #[test]
    fn watchdog_lexical_error_finishes_with_error_terminal() {
        let mut watchdog = DiagnosticWatchdog::start(10, Duration::from_secs(60)).unwrap();
        let parent = watchdog.enter_call(1);
        watchdog.update(|progress| progress.phase = DiagnosticPhase::ScopeProof);
        watchdog.complete_call(parent, true);
        watchdog.finish(true);
        let snapshots = watchdog
            .shared
            .snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        assert_eq!(snapshots.last().map(|snapshot| snapshot.phase), Some(DiagnosticPhase::Error));
        assert_eq!(
            watchdog
                .shared
                .events
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .last()
                .copied(),
            Some("watchdog_terminal")
        );
    }

    #[test]
    fn watchdog_opt_in_preserves_normal_form_bound_and_identity() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let root = gaussian_factor(&mut expressions, matrix_type(), 707, 3);
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let (off, off_counters, off_trace_lines) = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                    .unwrap()
                    .with_watchdog_override(false, Duration::from_millis(1));
            let value = normalizer.normalize_with_trace(semantic).unwrap();
            assert!(normalizer.last_watchdog_events.is_empty());
            (value, normalizer.counters(), normalizer.trace.lines_emitted)
        };
        let (on, on_counters, on_trace_lines) = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                    .unwrap()
                    .with_watchdog_override(true, Duration::from_millis(1));
            let value = normalizer.normalize_with_trace(semantic).unwrap();
            assert_eq!(normalizer.last_watchdog_events.first(), Some(&"watchdog_initial"));
            assert_eq!(normalizer.last_watchdog_events.last(), Some(&"watchdog_terminal"));
            assert!(normalizer.last_watchdog_events.len() <= 32);
            assert!(normalizer.last_watchdog_events.iter().all(|event| {
                matches!(
                    *event,
                    "watchdog_initial" |
                        "watchdog_snapshot" |
                        "watchdog_owner_sample" |
                        "watchdog_terminal"
                )
            }));
            let terminal = normalizer.last_watchdog_snapshots.last().copied().unwrap();
            assert_eq!(terminal.owners.monomial_retained_descriptor_slots, 1);
            assert_eq!(terminal.owners.monomial_reachable_descriptor_slots, 1);
            assert_eq!(terminal.owners.cache_entries, 0);
            assert_eq!(terminal.owners.cache_exact_terms, 0);
            assert_eq!(terminal.owners.cache_exact_terms_peak, 1);
            (value, normalizer.counters(), normalizer.trace.lines_emitted)
        };
        assert!(off_trace_lines > 0);
        assert_eq!(on_trace_lines, 0);
        assert_eq!(off.semantic, on.semantic);
        assert_eq!(off.coefficient_bound, on.coefficient_bound);
        assert_eq!(off.exact_nf, on.exact_nf);
        assert_eq!(off_counters, on_counters);
    }

    #[test]
    fn materialization_origin_is_watchdog_only_and_preserves_exact_result() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let x = source_with(&mut expressions, matrix_type(), 70_701);
        let y = source_with(&mut expressions, matrix_type(), 70_702);
        let sum = expressions.intern_matrix_transform(MatrixOperation::Add, &[x, y]).unwrap();
        let root = expressions.intern_matrix_transform(MatrixOperation::Negate, &[sum]).unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let (off, off_counters) = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                    .unwrap()
                    .with_watchdog_override(false, Duration::from_secs(60));
            let value = normalizer.normalize(semantic).unwrap();
            assert!(normalizer.diagnostic_materialization_origins.is_empty());
            (value, normalizer.counters())
        };
        let (on, on_counters, origin) = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                    .unwrap()
                    .with_watchdog_override(true, Duration::from_secs(60));
            let value = normalizer.normalize(semantic).unwrap();
            let origin = normalizer.diagnostic_materialization_origins.get(&root).copied().unwrap();
            (value, normalizer.counters(), origin)
        };
        assert_eq!(origin.producer, Some(sum));
        assert_eq!(origin.producer_operator, "add");
        assert_eq!(origin.reason, DiagnosticMaterializationReason::NonAddConsumer);
        assert_eq!(origin.consumer, Some(root));
        assert_eq!(origin.consumer_operator, "negate");
        assert_eq!(origin.consumer_category, "structural");
        assert_eq!(origin.remaining_uses, 0);
        assert_eq!(origin.scalar_classification, "not_multiply");
        assert_eq!(origin.forced_input_count, 1);
        assert_eq!(origin.forced_terms_sum, 2);
        assert_eq!(origin.forced_terms_max, 2);
        assert_eq!(origin.retained_term_count, 2);
        assert_eq!(off.semantic, on.semantic);
        assert_eq!(off.coefficient_bound, on.coefficient_bound);
        assert_eq!(off.exact_nf, on.exact_nf);
        assert_eq!(off_counters, on_counters);
    }

    #[test]
    fn ordinary_normalize_without_watchdog_or_trace_runs_zero_owner_censuses() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let root = source_with(&mut expressions, matrix_type(), 7_083);
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_watchdog_override(false, Duration::from_millis(1));
        normalizer.normalize(semantic).unwrap();
        assert_eq!(normalizer.owner_census_samples, 0);
        assert_eq!(normalizer.owner_census_seq, 0);
    }

    #[test]
    fn watchdog_hot_loop_publication_is_sublinear_and_bounded() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let root = source_with(&mut expressions, matrix_type(), 708);
        let (facts, mut monomials, _) = setup(&mut expressions, &mut programs, root);
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        normalizer.watchdog = DiagnosticWatchdog::start(11, Duration::from_secs(60));
        for _ in 0..1_000_000 {
            normalizer.watchdog_record_product_processed(0, 0, std::iter::empty());
            normalizer.watchdog_record_relation_processed(std::iter::empty());
        }
        assert_eq!(normalizer.watchdog_product_processed, 1_000_000);
        assert_eq!(normalizer.watchdog_relation_processed, 1_000_000);
        assert!(normalizer.watchdog_hot_publish_count <= 20);
        let progress = *normalizer
            .watchdog
            .as_ref()
            .unwrap()
            .shared
            .progress
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert!(progress.product_processed >= 524_288);
        assert!(progress.relation_processed >= 524_288);
        normalizer.watchdog.as_mut().unwrap().finish(false);
        normalizer.watchdog = None;
    }

    #[test]
    fn watchdog_cartesian_generation_publishes_before_drain_with_exact_active_owner_union() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let root = source_with(&mut expressions, matrix_type(), 7_080);
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let monomial = monomials.intern(&expressions, &programs, &[], &[semantic]).unwrap();
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        normalizer.watchdog = DiagnosticWatchdog::start(12, Duration::from_secs(60));

        for queue in 1_u64..=1_024 {
            // This directly models the eager Cartesian producer before the drain loop starts.
            normalizer.watchdog_record_product_generated(true, queue, [monomial, monomial]);
        }
        let progress = *normalizer
            .watchdog
            .as_ref()
            .unwrap()
            .shared
            .progress
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(progress.phase, DiagnosticPhase::ProductGeneration);
        assert_eq!(progress.product_generation_current, 1_024);
        assert_eq!(progress.product_enqueued_current, 1_024);
        assert_eq!(progress.product_queue_current, 1_024);
        assert_eq!(progress.product_output_current, 0);
        assert_eq!(progress.owner_census_seq, 0);
        // Small-product counter heartbeats do not run the O(retained) census. A designated
        // generation-end sample carries the active queue union and its coherence metadata.
        normalizer
            .watchdog_update(|progress| progress.phase = DiagnosticPhase::ProductGenerationEnd);
        normalizer.sample_owner_census(
            OwnerCensusReason::LargeProductGenerationEnd,
            [monomial, monomial],
        );
        let progress = *normalizer
            .watchdog
            .as_ref()
            .unwrap()
            .shared
            .progress
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(progress.owners.monomial_retained_descriptor_slots, 1);
        assert_eq!(progress.owners.monomial_reachable_descriptor_slots, 1);
        assert_eq!(progress.owners.monomial_unreachable_descriptor_slots, 0);
        assert_eq!(progress.owner_census_seq, 1);
        assert_eq!(progress.owner_census_phase, DiagnosticPhase::ProductGenerationEnd);
        assert_eq!(
            progress.owner_census_reason,
            Some(OwnerCensusReason::LargeProductGenerationEnd)
        );
        assert_eq!(normalizer.owner_census_samples, 1);
        normalizer.watchdog_update(|progress| {
            progress.phase = DiagnosticPhase::ProductEnd;
            progress.product_queue_current = 0;
            progress.product_output_current = 7;
        });
        normalizer.sample_owner_census(OwnerCensusReason::LargeProductEnd, [monomial]);
        normalizer.sample_owner_census(OwnerCensusReason::OuterTerminal, [monomial]);
        let terminal = *normalizer
            .watchdog
            .as_ref()
            .unwrap()
            .shared
            .progress
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(terminal.owner_census_seq, 3);
        assert_eq!(terminal.owner_census_reason, Some(OwnerCensusReason::OuterTerminal));
        assert_eq!(normalizer.owner_census_samples, 3);
        let shared = Arc::clone(&normalizer.watchdog.as_ref().unwrap().shared);
        normalizer.watchdog.as_mut().unwrap().finish(false);
        normalizer.watchdog = None;
        let owner_samples =
            shared.owner_samples.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone();
        assert_eq!(owner_samples.len(), 3);
        assert_eq!(
            (
                owner_samples[0].seq,
                owner_samples[0].product_queue_current,
                owner_samples[0].product_output_current
            ),
            (1, 1_024, 0)
        );
        assert_eq!(
            (
                owner_samples[1].seq,
                owner_samples[1].product_queue_current,
                owner_samples[1].product_output_current
            ),
            (2, 0, 7)
        );
        assert_eq!(owner_samples[0].reason, OwnerCensusReason::LargeProductGenerationEnd);
        assert_eq!(owner_samples[1].reason, OwnerCensusReason::LargeProductEnd);
        assert_eq!(owner_samples[2].reason, OwnerCensusReason::OuterTerminal);
        assert_eq!(owner_samples[0].phase, DiagnosticPhase::ProductGenerationEnd);
        assert_eq!(owner_samples[1].phase, DiagnosticPhase::ProductEnd);
        assert_eq!(
            (owner_samples[0].product_generated, owner_samples[0].product_enqueued),
            (1_024, 1_024)
        );
        assert_eq!(owner_samples[0].owners.monomial_reachable_descriptor_slots, 1);
        let events = shared.events.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone();
        let snapshots =
            shared.snapshots.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone();
        let emitted_owner_samples = events
            .iter()
            .zip(&snapshots)
            .filter_map(|(event, snapshot)| (*event == "watchdog_owner_sample").then_some(snapshot))
            .collect::<Vec<_>>();
        assert_eq!(emitted_owner_samples.len(), 3);
        assert_eq!(
            (
                emitted_owner_samples[0].owner_census_seq,
                emitted_owner_samples[0].product_queue_current,
                emitted_owner_samples[0].product_output_current
            ),
            (1, 1_024, 0)
        );
        assert_eq!(
            (
                emitted_owner_samples[1].owner_census_seq,
                emitted_owner_samples[1].product_queue_current,
                emitted_owner_samples[1].product_output_current
            ),
            (2, 0, 7)
        );
        // One publication per shared first/power-of-two boundary, not per generated pair.
        assert!(normalizer.watchdog_hot_publish_count <= 11);
    }

    #[test]
    fn owner_census_scheduler_preserves_late_strictly_larger_product_pair() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let root = source_with(&mut expressions, matrix_type(), 7_081);
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let monomial = monomials.intern(&expressions, &programs, &[], &[semantic]).unwrap();
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        normalizer.watchdog = DiagnosticWatchdog::start(14, Duration::from_secs(60));
        normalizer.next_retained_census_milestone = 1 << 20;

        // Only the first of five equal early products consumes a pair. Equal planned sizes do not
        // displace a later, strictly larger OOM-dominant product.
        for product_call_id in 1_u64..=5 {
            normalizer.watchdog_product_call_id = product_call_id;
            if normalizer.admits_large_product_census_pair(100_000) {
                normalizer.largest_sampled_product_planned = 100_000;
                normalizer.watchdog_update(|progress| {
                    progress.phase = DiagnosticPhase::ProductGenerationEnd;
                    progress.product_generation_current = 100_000;
                    progress.product_enqueued_current = 100_000;
                    progress.product_queue_current = 100_000;
                    progress.product_output_current = 0;
                });
                normalizer
                    .sample_owner_census(OwnerCensusReason::LargeProductGenerationEnd, [monomial]);
                normalizer.watchdog_update(|progress| {
                    progress.phase = DiagnosticPhase::ProductEnd;
                    progress.product_queue_current = 0;
                    progress.product_output_current = 1;
                });
                normalizer.sample_owner_census(OwnerCensusReason::LargeProductEnd, [monomial]);
                normalizer.large_product_pairs_sampled =
                    normalizer.large_product_pairs_sampled.saturating_add(1);
            }
        }
        assert_eq!(normalizer.large_product_pairs_sampled, 1);

        // Model the authoritative retained-arena geometric crossings without allocating millions
        // of descriptors in this scheduler fixture.
        for retained in [1_u64 << 20, 1_u64 << 21, 1_u64 << 22] {
            assert!(normalizer.admits_retained_arena_census(retained));
            normalizer.watchdog_update(|progress| {
                progress.phase = DiagnosticPhase::ProductGeneration;
            });
            normalizer.sample_owner_census(OwnerCensusReason::RetainedArenaMilestone, [monomial]);
        }

        normalizer.watchdog_product_call_id = 6;
        assert!(normalizer.admits_large_product_census_pair(200_000));
        normalizer.largest_sampled_product_planned = 200_000;
        normalizer.watchdog_update(|progress| {
            progress.phase = DiagnosticPhase::ProductGenerationEnd;
            progress.product_generation_current = 200_000;
            progress.product_enqueued_current = 200_000;
            progress.product_queue_current = 200_000;
            progress.product_output_current = 0;
        });
        normalizer.sample_owner_census(OwnerCensusReason::LargeProductGenerationEnd, [monomial]);
        normalizer.watchdog_update(|progress| {
            progress.phase = DiagnosticPhase::ProductEnd;
            progress.product_queue_current = 0;
            progress.product_output_current = 2;
        });
        normalizer.sample_owner_census(OwnerCensusReason::LargeProductEnd, [monomial]);
        normalizer.large_product_pairs_sampled =
            normalizer.large_product_pairs_sampled.saturating_add(1);
        normalizer.sample_owner_census(OwnerCensusReason::OuterTerminal, [monomial]);

        let shared = Arc::clone(&normalizer.watchdog.as_ref().unwrap().shared);
        normalizer.watchdog.as_mut().unwrap().finish(false);
        normalizer.watchdog = None;
        let samples =
            shared.owner_samples.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone();
        assert_eq!(samples.len(), 8);
        assert_eq!(samples.iter().filter(|sample| sample.product_call_id <= 5).count(), 5);
        assert_eq!(samples[5].reason, OwnerCensusReason::LargeProductGenerationEnd);
        assert_eq!(samples[5].product_call_id, 6);
        assert_eq!(
            (samples[5].product_queue_current, samples[5].product_output_current),
            (200_000, 0)
        );
        assert_eq!(samples[6].reason, OwnerCensusReason::LargeProductEnd);
        assert_eq!((samples[6].product_queue_current, samples[6].product_output_current), (0, 2));
        assert_eq!(samples[7].reason, OwnerCensusReason::OuterTerminal);
        assert_eq!(
            samples.iter().map(|sample| sample.seq).collect::<Vec<_>>(),
            (1..=8).collect::<Vec<_>>()
        );
        let events = shared.events.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone();
        assert_eq!(events.iter().filter(|event| **event == "watchdog_owner_sample").count(), 8);
        assert_eq!(events.iter().filter(|event| **event == "watchdog_terminal").count(), 1);
        assert!(events.len() <= 32);
        assert!(normalizer.owner_census_samples <= 11);
    }

    #[test]
    fn specialization_watchdog_publication_is_sublinear_and_terminal_is_exact() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let root = source_with(&mut expressions, matrix_type(), 709);
        let (facts, mut monomials, _) = setup(&mut expressions, &mut programs, root);
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        normalizer.watchdog = DiagnosticWatchdog::start(13, Duration::from_secs(60));
        for _ in 0..1_000_000 {
            normalizer.watchdog_record_specialization(DiagnosticPhase::Registration, |counters| {
                counters.registrations_started = counters.registrations_started.saturating_add(1);
            });
        }
        assert_eq!(normalizer.watchdog_specialization.registrations_started, 1_000_000);
        assert!(normalizer.watchdog_hot_publish_count <= 20);
        let exact = normalizer.watchdog_specialization;
        normalizer.watchdog_update(|progress| progress.specialization = exact);
        normalizer.watchdog.as_mut().unwrap().finish(false);
        let terminal = normalizer
            .watchdog
            .as_ref()
            .unwrap()
            .shared
            .snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .last()
            .copied()
            .unwrap();
        assert_eq!(terminal.specialization, exact);
        normalizer.watchdog = None;
    }

    #[test]
    fn relation_outcome_observations_are_coefficient_free_and_closure_local() {
        let arena = ArenaToken::fresh();
        let same = MonomialId::new(arena, 0);
        let outcomes = [
            RelationOutcomeKind::Gadget,
            RelationOutcomeKind::WholeClosed,
            RelationOutcomeKind::ClosedWindow,
            RelationOutcomeKind::Universal,
            RelationOutcomeKind::NoMatch,
            RelationOutcomeKind::Error,
        ];
        let mut first = RelationClosureDiagnostic {
            counters: DiagnosticRelationCounters::default(),
            outcomes: HashMap::new(),
        };
        // Coefficients are deliberately absent from the diagnostic key. Two dequeues of the same
        // arena-qualified monomial with different coefficients retain one authoritative outcome.
        for (monomial, _coefficient) in [(same, BigInt::from(3_u8)), (same, BigInt::from(-5_i8))] {
            record_relation_outcome(Some(&mut first), monomial, RelationOutcomeKind::WholeClosed);
        }
        assert_eq!(first.outcomes.len(), 1);
        assert_eq!(first.counters.duplicate_same_outcome, 1);
        record_relation_outcome(Some(&mut first), same, RelationOutcomeKind::Universal);
        assert_eq!(first.counters.duplicate_changed_outcome, 1);
        assert_eq!(first.outcomes.get(&same), Some(&RelationOutcomeKind::WholeClosed));

        for (slot, outcome) in outcomes.into_iter().enumerate() {
            record_relation_outcome(
                Some(&mut first),
                MonomialId::new(arena, u32::try_from(slot + 1).unwrap()),
                outcome,
            );
        }
        assert_eq!(first.outcomes.len(), outcomes.len() + 1);

        let mut second = RelationClosureDiagnostic {
            counters: DiagnosticRelationCounters::default(),
            outcomes: HashMap::new(),
        };
        record_relation_outcome(Some(&mut second), same, RelationOutcomeKind::Universal);
        assert_eq!(second.outcomes.get(&same), Some(&RelationOutcomeKind::Universal));
        assert_eq!(second.counters.duplicate_same_outcome, 0);
        assert_eq!(second.counters.duplicate_changed_outcome, 0);
    }

    #[test]
    fn relation_closure_watchdog_is_diagnostic_only_and_terminal_exact() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let root = source_with(&mut expressions, matrix_type(), 712);
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let mut relations = RelationRegistry::new();
        relations.freeze();
        let mut cache = NormalizationCache::new();

        let (off, off_timings) = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                    .unwrap()
                    .with_relations(&relations, &mut cache)
                    .with_watchdog_override(false, Duration::from_secs(60));
            let value = normalizer.normalize(semantic).unwrap();
            (value, normalizer.watchdog_timings)
        };
        assert_eq!(off_timings, DiagnosticTimings::default());
        let fingerprint = cache.canonical_state_fingerprint();
        let on = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                    .unwrap()
                    .with_relations(&relations, &mut cache)
                    .with_watchdog_override(true, Duration::from_secs(60));
            let value = normalizer.normalize(semantic).unwrap();
            let terminal = normalizer.last_watchdog_snapshots.last().copied().unwrap();
            assert_eq!(terminal.relation_closure.closures_started, 1);
            assert_eq!(terminal.relation_closure.closures_completed, 1);
            assert_eq!(terminal.relation_closure.closures_errored, 0);
            assert_eq!(terminal.relation_closure.active_depth, 0);
            assert!(!terminal.relation_closure.closed_relations_present);
            assert_eq!(terminal.relation_closure.initial_terms, 1);
            assert_eq!(terminal.relation_closure.dequeued, 1);
            assert_eq!(terminal.relation_closure.zero_skipped, 0);
            assert_eq!(terminal.relation_closure.nonzero_dequeued, 1);
            assert_eq!(terminal.relation_closure.enqueued, 0);
            assert_eq!(terminal.relation_closure.queue_peak, 1);
            assert_eq!(terminal.relation_closure.gadget_attempts, 1);
            assert_eq!(terminal.relation_closure.gadget_matches, 0);
            assert_eq!(terminal.relation_closure.whole_closed_probes, 0);
            assert_eq!(terminal.relation_closure.whole_closed_resolves, 0);
            assert_eq!(terminal.relation_closure.closed_window_probes, 0);
            assert_eq!(terminal.relation_closure.closed_window_interned_hits, 0);
            assert_eq!(terminal.relation_closure.closed_window_resolves, 0);
            assert_eq!(terminal.relation_closure.universal_probes, 1);
            assert_eq!(terminal.relation_closure.no_matches, 1);
            assert_eq!(terminal.relation_closure.match_errors, 0);
            assert_eq!(terminal.relation_closure.result_terms, 1);
            assert_eq!(terminal.relation_closure.final_terms, 1);
            for counter in [
                terminal.timings.closure_setup,
                terminal.timings.descriptor_and_gadget,
                terminal.timings.closed_search,
                terminal.timings.universal_search_total,
                terminal.timings.no_match_result_merge,
                terminal.timings.closure_final_assignment,
                terminal.timings.outer_scope_proof,
                terminal.timings.outer_use_counts,
                terminal.timings.outer_relation_rebound,
                terminal.timings.outer_bound_fold,
            ] {
                assert_eq!(counter.calls, 1);
                assert!(counter.total_ns >= counter.max_ns);
            }
            assert_eq!(terminal.timings.rhs_fetch_prefix_suffix.calls, 0);
            assert_eq!(terminal.timings.rhs_recombine_enqueue.calls, 0);
            assert_eq!(
                terminal.timings.universal_factor_dispatch.calls,
                terminal.relation_closure.universal_probes,
                "every inspected factor, including a non-call, has one complete dispatch timing"
            );
            // Entry/exit and the outer worklist publication are joined by the first universal
            // threshold. The empty closed authority emits no closed-search publication.
            assert!(normalizer.watchdog_hot_publish_count <= 8);
            assert!(normalizer.last_watchdog_events.len() <= 32);
            value
        };
        assert_eq!(off.semantic, on.semantic);
        assert_eq!(off.exact_nf, on.exact_nf);
        assert_eq!(off.coefficient_bound, on.coefficient_bound);
        assert_eq!(cache.canonical_state_fingerprint(), fingerprint);
    }

    #[test]
    fn relation_matcher_publishes_power_of_two_progress_before_return() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let factors = (0..16)
            .map(|offset| source_with(&mut expressions, matrix_type(), 760 + offset))
            .collect::<Vec<_>>();
        let unrelated_lhs = product(&mut expressions, &[factors[0], factors[0]]);
        let root = product(&mut expressions, &factors);
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let mut normal_form = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
            normalizer.fold_final_no_match = false;
            (*normalizer.normalize(semantic).unwrap().exact_nf.unwrap()).clone()
        };
        let expected = normal_form.clone();
        let mut relations = RelationRegistry::new();
        let mut cache = NormalizationCache::new();
        register_test_closed_relation_into(
            &mut expressions,
            &programs,
            &facts,
            &mut monomials,
            &mut relations,
            &mut cache,
            unrelated_lhs,
            factors[0],
            factors[0],
            factors[1],
        );
        relations.freeze();
        let observed = Arc::new(Mutex::new(Vec::new()));
        let observer = Arc::clone(&observed);
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_relations(&relations, &mut cache);
        normalizer.fold_final_no_match = false;
        normalizer.watchdog = DiagnosticWatchdog::start(14, Duration::from_secs(60));
        normalizer.relation_matcher_publish_observer = Some(Box::new(move |counters| {
            observer.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push(counters);
        }));

        assert!(!normalizer.rewrite_closed_relations(&mut normal_form).unwrap());
        assert_eq!(normal_form, expected);
        let observed = observed.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone();
        for threshold in [1, 2, 4, 8, 16, 32, 64, 128] {
            assert!(observed.iter().any(|snapshot| snapshot.closed_window_probes == threshold));
        }
        for threshold in [1, 2, 4, 8, 16] {
            assert!(observed.iter().any(|snapshot| snapshot.universal_probes == threshold));
        }
        assert!(observed.iter().all(|snapshot| snapshot.final_terms == 0));
        let progress = *normalizer
            .watchdog
            .as_ref()
            .unwrap()
            .shared
            .progress
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(progress.relation_closure.closed_window_probes, 136);
        assert_eq!(progress.relation_closure.universal_probes, 16);
        assert_eq!(progress.relation_closure.final_terms, 1);
        normalizer.watchdog.as_mut().unwrap().finish(false);
        normalizer.watchdog = None;
    }

    #[test]
    fn focused_caller_uses_exact_completed_invocation_across_depth_three() {
        let mut trace = NormalizationTrace::new();
        trace.focus_normalization_call = Some(3);
        trace.activate(1, 1, 0);
        trace.record_nested_normalization(1, 2);
        assert_eq!(trace.current_normalization_call, 2);
        trace.record_nested_normalization(1, 3);
        assert_eq!(trace.current_normalization_call, 3);
        assert!(trace.focused_invocation_selected());

        trace.record_completed_invocation(trace.current_normalization_call);
        trace.current_normalization_call = 2;
        trace.enter_caller_phase("caller:nested_return", true);
        assert_eq!(trace.caller_history, ["caller:nested_return"]);
        assert!(trace.next_specialized_root_armed);

        trace.record_completed_invocation(trace.current_normalization_call);
        trace.current_normalization_call = 1;
        trace.enter_caller_phase("caller:bounds_merge_start", true);
        assert_eq!(trace.caller_history, ["caller:nested_return"]);
        assert!(trace.next_specialized_root_armed);
        assert!(trace.claim_next_specialized_root());
        assert!(!trace.claim_next_specialized_root());
    }

    #[test]
    fn next_root_preproof_end_is_absent_after_early_proof_error() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let root = source_with(&mut expressions, matrix_type(), 706);
        let invalid_root = expressions.intern_argument(99, ResolvedValueType::Int).unwrap();
        let (facts, mut monomials, _) = setup(&mut expressions, &mut programs, root);
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        normalizer.trace.focus_normalization_call = Some(2);
        normalizer.trace.activate(1, 1, normalizer.monomials.len());
        normalizer.trace.record_completed_invocation(2);
        assert!(normalizer.normalize_specialized_root(invalid_root).is_err());
        assert!(normalizer.suspended_owner_roots.is_empty());
        // `preproof_end` is lexically after the fallible proof and cannot be replayed on error.
        assert_eq!(normalizer.trace.caller_history, ["next_root:preproof_start"]);
        assert!(!normalizer.trace.next_specialized_root_armed);
    }

    #[test]
    fn focused_subphase_trace_accepts_expression_slot_zero() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let atom = gaussian_factor(&mut expressions, matrix_type(), 705, 3);
        assert_eq!(atom.slot(), 0);
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, atom);
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_trace_focus_call_override(Some(1))
            .with_trace_focus_expression_slot_override(Some(0));
        normalizer.normalize_with_trace(semantic).unwrap();
        assert_eq!(normalizer.trace.focus_expression_slot, Some(0));
        assert_eq!(normalizer.trace.subphase_lines_emitted, 8);
        assert_eq!(normalizer.trace.subphase_history.len(), 8);
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
    fn four_class_frontier_distinguishes_current_and_future_typed_gadget_authority() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let input_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 1).unwrap();
        let decomposition_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 3, 1).unwrap();
        let gadget_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 3).unwrap();
        let (gadget, decomposition, input) = gadget_product(
            &mut expressions,
            false,
            3,
            gadget_type.clone(),
            decomposition_type.clone(),
            input_type.clone(),
            Some((2, false)),
        );
        let root = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[gadget, decomposition])
            .unwrap();
        let (mut facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        for expression in [gadget, decomposition, input] {
            insert_matrix_bound(&mut facts, &expressions, expression, 3);
        }
        let scope = semantic.program();
        let scoped = |expression| programs.scoped(&expressions, scope, expression).unwrap();
        let gadget_only =
            monomials.intern(&expressions, &programs, &[], &[scoped(gadget)]).unwrap();
        let pair = monomials
            .intern(&expressions, &programs, &[], &[scoped(gadget), scoped(decomposition)])
            .unwrap();
        let registry =
            recomposition_registry(gadget_type, decomposition_type, input_type, false, 3);
        let normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_gadget_recompositions(&registry);
        let future =
            normalizer.diagnostic_frontier_reasons(gadget_only, true, false, false).unwrap();
        assert_eq!(future, FRONTIER_FUTURE_TYPED_GADGET);
        let overlapping_future =
            normalizer.diagnostic_frontier_reasons(gadget_only, true, false, true).unwrap();
        assert_eq!(
            overlapping_future,
            FRONTIER_FUTURE_TYPED_GADGET | FRONTIER_FUTURE_UNIVERSAL_BLANKET
        );
        let current = normalizer.diagnostic_frontier_reasons(pair, false, false, false).unwrap();
        assert_eq!(current, FRONTIER_CURRENT_AUTHORIZED_GADGET);
        let no_future =
            normalizer.diagnostic_frontier_reasons(gadget_only, false, false, false).unwrap();
        assert_eq!(no_future, 0);
        let leaf = || {
            Arc::new(PolynomialNF {
                exact_terms: BTreeMap::from([(gadget_only, BigInt::from(1_u8))]),
                bounded_summary: BoundedSummary::missing(),
            })
        };
        let joined = normalizer
            .diagnostic_four_class_census(&[
                DiagnosticExactNf { normal_form: leaf(), ordinal: 0, under_product: false },
                DiagnosticExactNf { normal_form: leaf(), ordinal: 1, under_product: true },
            ])
            .unwrap();
        assert_eq!(joined.finite_relation_frontier.unique_monomials, 1);
        assert_eq!(joined.finite_relation_frontier.term_refs, 1);
        assert_eq!(joined.finite_no_relation.unique_monomials, 0);
        assert_eq!(joined.finite_no_relation.term_refs, 1);
        assert_eq!(joined.future_typed_gadget.unique_monomials, 1);
        assert_eq!(joined.future_typed_gadget.term_refs, 1);
        assert_eq!(joined.frontier_reason_term_ref_union, 1);
        assert_eq!(joined.top_len, 2);
        assert_eq!(joined.top[0].finite_no_relation_refs, 1);
        assert_eq!(joined.top[1].finite_relation_frontier_refs, 1);
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
    fn deferred_gadget_product_matches_eager_production_shape_and_bound() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let input_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 40).unwrap();
        let decomposition_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 40, 40).unwrap();
        let gadget_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 40).unwrap();
        let (gadget, decomposition, input) = gadget_product(
            &mut expressions,
            false,
            40,
            gadget_type.clone(),
            decomposition_type.clone(),
            input_type.clone(),
            Some((2, false)),
        );
        let product = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[gadget, decomposition])
            .unwrap();
        let zero_source =
            matrix_source(&mut expressions, "deferred-zero", input_type.clone(), None);
        let zero = expressions
            .intern_matrix_transform(MatrixOperation::Subtract, &[zero_source, zero_source])
            .unwrap();
        let root =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[product, zero]).unwrap();
        let registry =
            recomposition_registry(gadget_type, decomposition_type, input_type, false, 40);
        let (mut facts, mut monomials, root_semantic) =
            setup(&mut expressions, &mut programs, root);
        mark_scalar_sources_constant(&expressions, &mut facts, root);
        let product_semantic = programs.scoped(&expressions, monomials.scope(), product).unwrap();

        let eager = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                    .unwrap()
                    .with_gadget_recompositions(&registry);
            let value = normalizer.normalize(product_semantic).unwrap();
            assert_eq!(normalizer.next_exact_plan_id, 0, "a root product is always eager");
            value
        };
        let deferred = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                    .unwrap()
                    .with_gadget_recompositions(&registry);
            normalizer.monomial_gc_allocation_threshold_bytes = 0;
            let value = normalizer.normalize(root_semantic).unwrap();
            assert!(normalizer.next_exact_plan_id >= 3);
            assert_eq!(normalizer.exact_plan_materializations, 1);
            assert!(normalizer.gc_counters.sweep_count > 0);
            assert_eq!(normalizer.gadget_product_counters.plans_created, 1);
            assert_eq!(normalizer.gadget_product_counters.streamed_executions, 1);
            assert_eq!(normalizer.gadget_product_counters.planned_pairs, 1);
            assert_eq!(normalizer.gadget_product_counters.max_streamed_output_terms, 1);
            value
        };

        assert_eq!(deferred.exact_nf, eager.exact_nf);
        assert_eq!(deferred.coefficient_bound, eager.coefficient_bound);
        assert_eq!(
            deferred.exact_nf.as_ref().unwrap().bounded_summary.coefficient_bound,
            deferred.coefficient_bound
        );
        let exact = deferred.exact_nf.as_ref().unwrap();
        assert_eq!(exact.term_count(), 1);
        let input_semantic = programs.scoped(&expressions, root_semantic.program(), input).unwrap();
        let descriptor = monomials.descriptor(*exact.exact_terms.keys().next().unwrap()).unwrap();
        assert_eq!(descriptor.ordered_factors.as_ref(), &[input_semantic]);
    }

    #[test]
    fn deferred_small_gadget_product_preserves_multiterm_prefix_suffix_semantics() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let modulus = BigUint::from(17_u8);
        let scalar_type = ResolvedMatrixType::new(modulus.clone(), 1, 1, 1).unwrap();
        let prefix_type = ResolvedMatrixType::new(modulus.clone(), 1, 2, 2).unwrap();
        let input_type = ResolvedMatrixType::new(modulus.clone(), 1, 2, 4).unwrap();
        let gadget_type = ResolvedMatrixType::new(modulus.clone(), 1, 2, 8).unwrap();
        let decomposition_type = ResolvedMatrixType::new(modulus.clone(), 1, 8, 4).unwrap();
        let suffix_type = ResolvedMatrixType::new(modulus, 1, 4, 4).unwrap();
        let scalar = matrix_source(&mut expressions, "deferred-central", scalar_type, None);
        let prefix = matrix_source(&mut expressions, "deferred-prefix", prefix_type, None);
        let gadget = matrix_source(
            &mut expressions,
            "deferred-small-gadget",
            gadget_type.clone(),
            Some((2, true)),
        );
        let a0 = matrix_source(&mut expressions, "deferred-a0", input_type.clone(), None);
        let a1 = matrix_source(&mut expressions, "deferred-a1", input_type.clone(), None);
        let input = expressions.intern_matrix_transform(MatrixOperation::Add, &[a0, a1]).unwrap();
        let decomposition = expressions
            .intern(
                ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
                    output: decomposition_type.clone(),
                    base: 2,
                    small: true,
                    digit_count: 4,
                }),
                Box::new([input]),
            )
            .unwrap();
        let suffix = matrix_source(&mut expressions, "deferred-suffix", suffix_type, None);
        let prefixed = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[prefix, gadget])
            .unwrap();
        let left = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[scalar, prefixed])
            .unwrap();
        let right = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[decomposition, suffix])
            .unwrap();
        let product =
            expressions.intern_matrix_transform(MatrixOperation::Multiply, &[left, right]).unwrap();
        let zero_source =
            matrix_source(&mut expressions, "deferred-multiterm-zero", input_type.clone(), None);
        let zero = expressions
            .intern_matrix_transform(MatrixOperation::Subtract, &[zero_source, zero_source])
            .unwrap();
        let root =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[product, zero]).unwrap();
        let registry = recomposition_registry(gadget_type, decomposition_type, input_type, true, 4);
        let (mut facts, mut monomials, root_semantic) =
            setup(&mut expressions, &mut programs, root);
        mark_scalar_sources_constant(&expressions, &mut facts, root);
        let product_semantic = programs.scoped(&expressions, monomials.scope(), product).unwrap();
        let eager = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_gadget_recompositions(&registry)
            .normalize(product_semantic)
            .unwrap();
        let (deferred, product_counters, gadget_counters) = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                    .unwrap()
                    .with_gadget_recompositions(&registry);
            let value = normalizer.normalize(root_semantic).unwrap();
            (value, normalizer.product_plan_counters, normalizer.gadget_product_counters)
        };
        let descriptor_multiset = |nf: &PolynomialNF, monomials: &MonomialArena| {
            nf.exact_terms
                .iter()
                .map(|(id, coefficient)| {
                    let descriptor = monomials.descriptor(*id).unwrap();
                    (
                        descriptor.central_factors.to_vec(),
                        descriptor.ordered_factors.to_vec(),
                        coefficient.clone(),
                    )
                })
                .collect::<BTreeSet<_>>()
        };
        assert_eq!(
            descriptor_multiset(eager.exact_nf.as_ref().unwrap(), &monomials),
            descriptor_multiset(deferred.exact_nf.as_ref().unwrap(), &monomials)
        );
        assert_eq!(eager.coefficient_bound, deferred.coefficient_bound);
        assert_eq!(deferred.exact_nf.as_ref().unwrap().term_count(), 2);
        assert_eq!(gadget_counters, GadgetProductPlanCounters::default());
        assert_eq!(product_counters.plans_created, 2);
        assert_eq!(product_counters.typed_candidate_plans, 2);
        assert_eq!(product_counters.typed_direct_executions, 1);
        assert_eq!(product_counters.typed_pair_attempts, 1);
        assert_eq!(product_counters.typed_pair_matches, 1);
        assert_eq!(product_counters.typed_pair_ordinary_fallbacks, 0);
        assert_eq!(product_counters.streamed_executions, 1);
        assert_eq!(product_counters.typed_standalone_materializations, 1);
    }

    #[test]
    fn shared_deferred_gadget_product_zero_weight_skips_execution() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let input_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 4).unwrap();
        let decomposition_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 4, 4).unwrap();
        let gadget_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 4).unwrap();
        let (gadget, decomposition, _) = gadget_product(
            &mut expressions,
            false,
            4,
            gadget_type.clone(),
            decomposition_type.clone(),
            input_type.clone(),
            Some((2, false)),
        );
        let product = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[gadget, decomposition])
            .unwrap();
        let root = expressions
            .intern_matrix_transform(MatrixOperation::Subtract, &[product, product])
            .unwrap();
        let registry =
            recomposition_registry(gadget_type, decomposition_type, input_type, false, 4);
        let (mut facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        mark_scalar_sources_constant(&expressions, &mut facts, root);
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_gadget_recompositions(&registry);
        let value = normalizer.normalize(semantic).unwrap();
        assert!(value.exact_nf.as_ref().unwrap().is_zero());
        assert_eq!(value.coefficient_bound, NumericContract::Known(CoefficientBound::ExactZero));
        assert_eq!(normalizer.next_exact_plan_id, 2, "one product plan and one subtraction plan");
        assert_eq!(normalizer.exact_plan_materializations, 1);
        assert_eq!(normalizer.gadget_product_counters.plans_created, 1);
        assert_eq!(normalizer.gadget_product_counters.zero_weight_skips, 1);
        assert_eq!(normalizer.gadget_product_counters.streamed_executions, 0);
        assert_eq!(normalizer.gadget_product_counters.planned_pairs, 0);
    }

    #[test]
    fn shared_deferred_gadget_product_streams_one_weighted_execution_without_boundary_descriptor() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let input_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 4).unwrap();
        let decomposition_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 4, 4).unwrap();
        let gadget_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 4).unwrap();
        let (gadget, decomposition, input) = gadget_product(
            &mut expressions,
            false,
            4,
            gadget_type.clone(),
            decomposition_type.clone(),
            input_type.clone(),
            Some((2, false)),
        );
        let product = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[gadget, decomposition])
            .unwrap();
        let root =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[product, product]).unwrap();
        let registry =
            recomposition_registry(gadget_type, decomposition_type, input_type, false, 4);
        let (mut facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        mark_scalar_sources_constant(&expressions, &mut facts, root);
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_gadget_recompositions(&registry);
        normalizer.monomial_gc_allocation_threshold_bytes = u64::MAX;
        let value = normalizer.normalize(semantic).unwrap();
        let exact = value.exact_nf.as_ref().unwrap();
        assert_eq!(exact.term_count(), 1);
        assert_eq!(exact.exact_terms.values().next(), Some(&BigInt::from(2_u8)));
        assert_eq!(normalizer.gadget_product_counters.plans_created, 1);
        assert_eq!(normalizer.gadget_product_counters.streamed_executions, 1);
        assert_eq!(normalizer.gadget_product_counters.zero_weight_skips, 0);
        assert_eq!(normalizer.gadget_product_counters.planned_pairs, 1);
        let scope = semantic.program();
        let gadget = programs.scoped(&expressions, scope, gadget).unwrap();
        let decomposition = programs.scoped(&expressions, scope, decomposition).unwrap();
        let input = programs.scoped(&expressions, scope, input).unwrap();
        let output = monomials.descriptor(*exact.exact_terms.keys().next().unwrap()).unwrap();
        assert_eq!(output.ordered_factors.as_ref(), &[input]);
        for slot in 0..monomials.len() {
            let Ok(slot) = u32::try_from(slot) else { panic!("fixture arena exceeded u32") };
            let id = MonomialId::new(monomials.token(), slot);
            let Ok(descriptor) = monomials.descriptor(id) else { continue };
            assert_ne!(
                descriptor.ordered_factors.as_ref(),
                &[gadget, decomposition],
                "the direct boundary must not intern a transient G*D descriptor"
            );
        }
    }

    #[test]
    fn gadget_product_deferral_requires_additive_only_nonroot_consumers() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let input_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 4).unwrap();
        let decomposition_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 4, 4).unwrap();
        let gadget_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 4).unwrap();
        let (gadget, decomposition, _) = gadget_product(
            &mut expressions,
            false,
            4,
            gadget_type.clone(),
            decomposition_type.clone(),
            input_type.clone(),
            Some((2, false)),
        );
        let candidate = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[gadget, decomposition])
            .unwrap();
        let right_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 4, 4).unwrap();
        let right = matrix_source(&mut expressions, "non-additive-consumer", right_type, None);
        let root = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[candidate, right])
            .unwrap();
        let registry =
            recomposition_registry(gadget_type, decomposition_type, input_type, false, 4);
        let (mut facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        mark_scalar_sources_constant(&expressions, &mut facts, root);
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_gadget_recompositions(&registry);
        let value = normalizer.normalize(semantic).unwrap();
        assert!(value.exact_nf.is_some());
        assert_eq!(normalizer.next_exact_plan_id, 0);
    }

    #[test]
    fn typed_gadget_candidate_requires_a_frozen_registry_at_plan_construction() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let input_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 4).unwrap();
        let decomposition_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 4, 4).unwrap();
        let gadget_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 4).unwrap();
        let (gadget, decomposition, _) = gadget_product(
            &mut expressions,
            false,
            4,
            gadget_type,
            decomposition_type,
            input_type.clone(),
            Some((2, false)),
        );
        let product = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[gadget, decomposition])
            .unwrap();
        let zero_source = matrix_source(&mut expressions, "no-registry-zero", input_type, None);
        let zero = expressions
            .intern_matrix_transform(MatrixOperation::Subtract, &[zero_source, zero_source])
            .unwrap();
        let root =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[product, zero]).unwrap();
        let (mut facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        mark_scalar_sources_constant(&expressions, &mut facts, root);
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let value = normalizer.normalize(semantic).unwrap();
        assert!(value.exact_nf.is_some());
        assert_eq!(normalizer.product_plan_counters, ProductPlanCounters::default());
        assert_eq!(normalizer.gadget_product_counters, GadgetProductPlanCounters::default());
    }

    #[test]
    fn gadget_product_deferral_rejects_incomplete_or_mismatched_semantic_authority() {
        #[derive(Clone, Copy, Debug)]
        enum Rejection {
            WrongSource,
            WrongBase,
            WrongSmall,
            WrongDigit,
            WrongType,
            WrongLayout,
            Unfrozen,
        }

        for rejection in [
            Rejection::WrongSource,
            Rejection::WrongBase,
            Rejection::WrongSmall,
            Rejection::WrongDigit,
            Rejection::WrongType,
            Rejection::WrongLayout,
            Rejection::Unfrozen,
        ] {
            let mut expressions = ExprArena::new();
            let mut programs = ProgramArena::new();
            let modulus = BigUint::from(17_u8);
            let input_type = ResolvedMatrixType::new(modulus.clone(), 1, 1, 4).unwrap();
            let decomposition_type = ResolvedMatrixType::new(modulus.clone(), 1, 4, 4).unwrap();
            let gadget_type = ResolvedMatrixType::new(modulus, 1, 1, 4).unwrap();
            let gadget_constant = match rejection {
                Rejection::WrongSource => None,
                Rejection::WrongBase => Some((3, false)),
                Rejection::WrongSmall => Some((2, true)),
                _ => Some((2, false)),
            };
            let (gadget, decomposition, _) = gadget_product(
                &mut expressions,
                false,
                4,
                gadget_type.clone(),
                decomposition_type.clone(),
                input_type.clone(),
                gadget_constant,
            );
            let product = expressions
                .intern_matrix_transform(MatrixOperation::Multiply, &[gadget, decomposition])
                .unwrap();
            let zero_source = matrix_source(
                &mut expressions,
                "rejected-gadget-product-zero",
                input_type.clone(),
                None,
            );
            let zero = expressions
                .intern_matrix_transform(MatrixOperation::Subtract, &[zero_source, zero_source])
                .unwrap();
            let root = expressions
                .intern_matrix_transform(MatrixOperation::Add, &[product, zero])
                .unwrap();

            let rule_modulus = if matches!(rejection, Rejection::WrongType) {
                BigUint::from(19_u8)
            } else {
                BigUint::from(17_u8)
            };
            let rule_input = ResolvedMatrixType::new(rule_modulus.clone(), 1, 1, 4).unwrap();
            let rule_digits = if matches!(rejection, Rejection::WrongDigit) { 3 } else { 4 };
            let rule_decomposition = ResolvedMatrixType::new(
                rule_modulus.clone(),
                1,
                usize::try_from(rule_digits).unwrap(),
                4,
            )
            .unwrap();
            let rule_gadget =
                ResolvedMatrixType::new(rule_modulus, 1, 1, usize::try_from(rule_digits).unwrap())
                    .unwrap();
            let mut registry = GadgetRecompositionRegistry::new();
            registry
                .register(GadgetRecompositionRule {
                    base: 2,
                    small: false,
                    digit_count: rule_digits,
                    gadget_layout: (!matches!(rejection, Rejection::WrongLayout))
                        .then(|| MatrixLayout::row_major(1, usize::try_from(rule_digits).unwrap())),
                    decomposition_layout: (!matches!(rejection, Rejection::WrongLayout))
                        .then(|| MatrixLayout::row_major(usize::try_from(rule_digits).unwrap(), 4)),
                    input_layout: (!matches!(rejection, Rejection::WrongLayout))
                        .then(|| MatrixLayout::row_major(1, 4)),
                    output_type: rule_input.clone(),
                    gadget_type: rule_gadget,
                    decomposition_type: rule_decomposition,
                    input_type: rule_input,
                })
                .unwrap();
            if !matches!(rejection, Rejection::Unfrozen) {
                registry.freeze();
            }
            let (mut facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
            mark_scalar_sources_constant(&expressions, &mut facts, root);
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                    .unwrap()
                    .with_gadget_recompositions(&registry);
            let value = normalizer.normalize(semantic).unwrap();
            assert!(value.exact_nf.is_some(), "{rejection:?} must fall back eagerly");
            assert_eq!(
                normalizer.gadget_product_counters.plans_created, 0,
                "{rejection:?} must not authorize a deferred product"
            );
            assert_eq!(normalizer.gadget_product_counters.streamed_executions, 0);
            if matches!(rejection, Rejection::Unfrozen) {
                assert_eq!(normalizer.product_plan_counters.plans_created, 0);
            } else {
                assert_eq!(normalizer.product_plan_counters.plans_created, 1, "{rejection:?}");
                assert_eq!(
                    normalizer.product_plan_counters.typed_candidate_plans, 1,
                    "{rejection:?}"
                );
                assert_eq!(normalizer.product_plan_counters.typed_direct_executions, 0);
                assert_eq!(normalizer.product_plan_counters.typed_pair_attempts, 0);
                assert_eq!(normalizer.product_plan_counters.typed_pair_matches, 0);
                assert_eq!(normalizer.product_plan_counters.typed_pair_ordinary_fallbacks, 0);
            }
        }

        let valid = recomposition_registry(
            ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 4).unwrap(),
            ResolvedMatrixType::new(BigUint::from(17_u8), 1, 4, 4).unwrap(),
            ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 4).unwrap(),
            false,
            4,
        );
        assert!(!valid.allows(
            2,
            false,
            3,
            &ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 4).unwrap(),
            &ResolvedMatrixType::new(BigUint::from(17_u8), 1, 4, 4).unwrap(),
            &ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 4).unwrap(),
            &ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 4).unwrap(),
            Some(&MatrixLayout::row_major(1, 4)),
            Some(&MatrixLayout::row_major(4, 4)),
            Some(&MatrixLayout::row_major(1, 4)),
        ));
    }

    #[test]
    fn gadget_product_deferral_rejects_hash_and_reversed_order() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let matrix = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 4, 4).unwrap();
        let gadget =
            matrix_source(&mut expressions, "reversed-gadget", matrix.clone(), Some((2, false)));
        let input = matrix_source(&mut expressions, "reversed-input", matrix.clone(), None);
        let decomposition = expressions
            .intern(
                ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
                    output: matrix.clone(),
                    base: 2,
                    small: false,
                    digit_count: 1,
                }),
                Box::new([input]),
            )
            .unwrap();
        let hash = expressions
            .intern(
                ValueOperator::Sampler {
                    event: super::super::arena::SampleEventId(90_201),
                    operation: SamplerOperation::Hash {
                        output: matrix.clone(),
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
        let reversed = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[decomposition, gadget])
            .unwrap();
        let hash_product = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[gadget, hash])
            .unwrap();
        let sum = expressions
            .intern_matrix_transform(MatrixOperation::Add, &[reversed, hash_product])
            .unwrap();
        let zero_source = matrix_source(&mut expressions, "hash-order-zero", matrix.clone(), None);
        let zero = expressions
            .intern_matrix_transform(MatrixOperation::Subtract, &[zero_source, zero_source])
            .unwrap();
        let root = expressions.intern_matrix_transform(MatrixOperation::Add, &[sum, zero]).unwrap();
        let registry = recomposition_registry(matrix.clone(), matrix.clone(), matrix, false, 1);
        let (mut facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        mark_scalar_sources_constant(&expressions, &mut facts, root);
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_gadget_recompositions(&registry);
        let value = normalizer.normalize(semantic).unwrap();
        assert!(value.exact_nf.is_some());
        assert_eq!(normalizer.gadget_product_counters.plans_created, 0);
        assert_eq!(normalizer.gadget_product_counters.streamed_executions, 0);
    }

    #[test]
    fn gadget_product_owner_telemetry_separates_plan_and_operand_roots() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let input_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 4).unwrap();
        let decomposition_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 4, 4).unwrap();
        let gadget_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 4).unwrap();
        let (gadget, decomposition, _) = gadget_product(
            &mut expressions,
            false,
            4,
            gadget_type.clone(),
            decomposition_type.clone(),
            input_type.clone(),
            Some((2, false)),
        );
        let product = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[gadget, decomposition])
            .unwrap();
        let zero = matrix_source(&mut expressions, "telemetry-zero", input_type.clone(), None);
        let root =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[product, zero]).unwrap();
        let registry =
            recomposition_registry(gadget_type, decomposition_type, input_type, false, 4);
        let (mut facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        mark_scalar_sources_constant(&expressions, &mut facts, root);
        let normalize_operand =
            |expression: ExprId, expressions: &mut ExprArena, monomials: &mut MonomialArena| {
                let scoped = programs.scoped(expressions, semantic.program(), expression).unwrap();
                Normalizer::new(expressions, &programs, &facts, monomials)
                    .unwrap()
                    .normalize(scoped)
                    .unwrap()
                    .exact_nf
                    .unwrap()
            };
        let left = normalize_operand(gadget, &mut expressions, &mut monomials);
        let right = normalize_operand(decomposition, &mut expressions, &mut monomials);
        let gadget_semantic = programs.scoped(&expressions, semantic.program(), gadget).unwrap();
        let decomposition_semantic =
            programs.scoped(&expressions, semantic.program(), decomposition).unwrap();
        let dead = monomials
            .intern(&expressions, &programs, &[], &[gadget_semantic, decomposition_semantic])
            .unwrap();
        let node = expressions.node_arc(product).unwrap();
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_gadget_recompositions(&registry);
        normalizer.relation_rewriting_enabled = false;
        normalizer.deferred_gadget_products.insert(product);
        let plan = normalizer
            .new_gadget_product_plan(product, node.as_ref(), left, right)
            .unwrap()
            .unwrap();
        normalizer.gadget_product_plans.insert(product, plan);
        let owners = normalizer.owner_census();
        assert_eq!(owners.additive_plan_nodes, 0);
        assert_eq!(owners.gadget_product_plan_nodes, 1);
        assert_eq!(owners.additive_unique_leaf_refs, 0);
        assert_eq!(owners.gadget_product_unique_operand_refs, 2);
        assert_eq!(owners.gadget_product_operand_exact_term_refs, 2);
        assert_eq!(owners.gadget_product_largest_operand_exact_terms, 1);
        assert_eq!(owners.gadget_product_plans_created, 1);
        assert_eq!(owners.gadget_product_streamed_executions, 0);
        assert_eq!(owners.gadget_product_zero_weight_skips, 0);
        assert_eq!(owners.gadget_product_standalone_materializations, 0);
        assert_eq!(owners.gadget_product_planned_pairs, 0);
        assert_eq!(owners.gadget_product_max_streamed_output_terms, 0);
        let operand_roots = normalizer
            .gadget_product_plans
            .get(&product)
            .unwrap()
            .left
            .exact_terms
            .keys()
            .chain(normalizer.gadget_product_plans.get(&product).unwrap().right.exact_terms.keys())
            .copied()
            .collect::<Vec<_>>();
        normalizer.protected_monomial_prefix = 0;
        normalizer.normalization_depth = 1;
        normalizer.monomial_gc_allocation_threshold_bytes = 0;
        normalizer.sweep_monomials_at_node_commit().unwrap();
        for operand in operand_roots {
            normalizer.monomials.descriptor(operand).unwrap();
        }
        assert!(matches!(
            normalizer.monomials.descriptor(dead),
            Err(MonomialError::CollectedMonomialId { .. })
        ));
    }

    #[test]
    fn gadget_product_pending_operands_fail_before_any_weighted_execution() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let left_expression = source_with(&mut expressions, matrix_type(), 90_301);
        let right_expression = source_with(&mut expressions, matrix_type(), 90_302);
        let product = expressions
            .intern_matrix_transform(
                MatrixOperation::Multiply,
                &[left_expression, right_expression],
            )
            .unwrap();
        let root =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[product, product]).unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let left_scoped =
            programs.scoped(&expressions, semantic.program(), left_expression).unwrap();
        let right_scoped =
            programs.scoped(&expressions, semantic.program(), right_expression).unwrap();
        let left = monomials.intern(&expressions, &programs, &[], &[left_scoped]).unwrap();
        let right = monomials.intern(&expressions, &programs, &[], &[right_scoped]).unwrap();
        let tombstone =
            monomials.intern(&expressions, &programs, &[], &[left_scoped, right_scoped]).unwrap();
        monomials.sweep(0, [left, right]).unwrap();
        let mut foreign_arena =
            MonomialArena::new(&expressions, &programs, semantic.program()).unwrap();
        let foreign = foreign_arena.intern(&expressions, &programs, &[], &[left_scoped]).unwrap();
        let leaf = |monomial| {
            Arc::new(PolynomialNF {
                exact_terms: BTreeMap::from([(monomial, BigInt::from(1_u8))]),
                bounded_summary: BoundedSummary::missing(),
            })
        };
        let left_type = match expressions.value_type(left_expression).unwrap() {
            ResolvedValueType::Matrix(matrix) => matrix.clone(),
            _ => panic!("fixture operand must be a matrix"),
        };
        let right_type = match expressions.value_type(right_expression).unwrap() {
            ResolvedValueType::Matrix(matrix) => matrix.clone(),
            _ => panic!("fixture operand must be a matrix"),
        };
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let product_authority = normalizer.exact_plan_authority(product).unwrap();
        let root_authority = normalizer.exact_plan_authority(root).unwrap();
        for invalid in [foreign, tombstone] {
            let valid_plan = Arc::new(GadgetProductExactPlan {
                id: 1,
                authority: product_authority.clone(),
                expression: product,
                left_expression,
                right_expression,
                left_type: left_type.clone(),
                right_type: right_type.clone(),
                left: leaf(left),
                right: leaf(right),
            });
            let invalid_plan = Arc::new(GadgetProductExactPlan {
                id: 2,
                authority: product_authority.clone(),
                expression: product,
                left_expression,
                right_expression,
                left_type: left_type.clone(),
                right_type: right_type.clone(),
                left: leaf(left),
                right: leaf(invalid),
            });
            let state = NodeExactState::Additive(Arc::new(AdditiveExactPlan {
                id: 3,
                authority: root_authority.clone(),
                left: NodeExactState::GadgetProduct(valid_plan),
                right: NodeExactState::GadgetProduct(invalid_plan),
                subtract_right: false,
            }));
            let high_water = normalizer.monomials.len();
            let occupied = normalizer.monomials.occupied_len();
            let counters = normalizer.gadget_product_counters;
            let error = normalizer.materialize_exact_state(&state).unwrap_err();
            assert!(matches!(
                error,
                NormalizeError::Monomial(
                    MonomialError::InvalidMonomialId { .. } |
                        MonomialError::CollectedMonomialId { .. }
                )
            ));
            assert_eq!(normalizer.monomials.len(), high_water);
            assert_eq!(normalizer.monomials.occupied_len(), occupied);
            assert_eq!(normalizer.gadget_product_counters, counters);
        }
    }

    #[test]
    fn compositional_product_prevalidates_authority_and_all_pending_leaves() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let left_expression = source_with(&mut expressions, matrix_type(), 90_311);
        let right_expression = source_with(&mut expressions, matrix_type(), 90_312);
        let product = expressions
            .intern_matrix_transform(
                MatrixOperation::Multiply,
                &[left_expression, right_expression],
            )
            .unwrap();
        let root =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[product, product]).unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let left_scoped =
            programs.scoped(&expressions, semantic.program(), left_expression).unwrap();
        let right_scoped =
            programs.scoped(&expressions, semantic.program(), right_expression).unwrap();
        let left = monomials.intern(&expressions, &programs, &[], &[left_scoped]).unwrap();
        let right = monomials.intern(&expressions, &programs, &[], &[right_scoped]).unwrap();
        let tombstone =
            monomials.intern(&expressions, &programs, &[], &[left_scoped, right_scoped]).unwrap();
        monomials.sweep(0, [left, right]).unwrap();
        let mut foreign_arena =
            MonomialArena::new(&expressions, &programs, semantic.program()).unwrap();
        let foreign = foreign_arena.intern(&expressions, &programs, &[], &[left_scoped]).unwrap();
        let leaf = |authority: ExactPlanAuthority, monomial| NodeExactState::Materialized {
            authority,
            normal_form: Arc::new(PolynomialNF {
                exact_terms: BTreeMap::from([(monomial, BigInt::from(1_u8))]),
                bounded_summary: BoundedSummary::missing(),
            }),
        };
        let registry =
            recomposition_registry(matrix_type(), matrix_type(), matrix_type(), false, 1);
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_gadget_recompositions(&registry);
        let product_authority = normalizer.exact_plan_authority(product).unwrap();
        let left_authority = normalizer.exact_plan_authority(left_expression).unwrap();
        let right_authority = normalizer.exact_plan_authority(right_expression).unwrap();
        let left_type = left_authority.matrix_type.clone();
        let right_type = right_authority.matrix_type.clone();
        for invalid in [foreign, tombstone] {
            for mode in [ProductMode::Ordinary, ProductMode::TypedGadgetCandidate] {
                let state = NodeExactState::Product(Arc::new(ProductExactPlan {
                    id: 1,
                    authority: product_authority.clone(),
                    expression: product,
                    left_expression,
                    right_expression,
                    left_type: left_type.clone(),
                    right_type: right_type.clone(),
                    mode,
                    left: leaf(left_authority.clone(), left),
                    right: leaf(right_authority.clone(), invalid),
                }));
                let high_water = normalizer.monomials.len();
                let occupied = normalizer.monomials.occupied_len();
                let counters = normalizer.product_plan_counters;
                assert!(matches!(
                    normalizer.materialize_exact_state(&state),
                    Err(NormalizeError::Monomial(
                        MonomialError::InvalidMonomialId { .. } |
                            MonomialError::CollectedMonomialId { .. }
                    ))
                ));
                assert_eq!(normalizer.monomials.len(), high_water);
                assert_eq!(normalizer.monomials.occupied_len(), occupied);
                assert_eq!(normalizer.product_plan_counters, counters);
            }
        }

        let mut foreign_authority = product_authority.clone();
        foreign_authority.expressions = ArenaToken::fresh();
        let foreign_state = NodeExactState::Product(Arc::new(ProductExactPlan {
            id: 1,
            authority: foreign_authority,
            expression: product,
            left_expression,
            right_expression,
            left_type,
            right_type,
            mode: ProductMode::Ordinary,
            left: leaf(left_authority, left),
            right: leaf(right_authority, right),
        }));
        let high_water = normalizer.monomials.len();
        let counters = normalizer.product_plan_counters;
        assert!(matches!(
            normalizer.materialize_exact_state(&foreign_state),
            Err(NormalizeError::InvalidExactPlan { reason: "foreign exact authority" })
        ));
        assert_eq!(normalizer.monomials.len(), high_water);
        assert_eq!(normalizer.product_plan_counters, counters);
    }

    #[test]
    fn compositional_product_gc_roots_pending_leaves_and_releases_old_add_arcs() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let matrix = matrix_type();
        let a = source_with(&mut expressions, matrix.clone(), 90_321);
        let b = source_with(&mut expressions, matrix.clone(), 90_322);
        let c = source_with(&mut expressions, matrix, 90_323);
        let sum = expressions.intern_matrix_transform(MatrixOperation::Add, &[a, b]).unwrap();
        let product =
            expressions.intern_matrix_transform(MatrixOperation::Multiply, &[sum, c]).unwrap();
        let root =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[product, product]).unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let scoped =
            |expression| programs.scoped(&expressions, semantic.program(), expression).unwrap();
        let a_id = monomials.intern(&expressions, &programs, &[], &[scoped(a)]).unwrap();
        let b_id = monomials.intern(&expressions, &programs, &[], &[scoped(b)]).unwrap();
        let c_id = monomials.intern(&expressions, &programs, &[], &[scoped(c)]).unwrap();
        let dead = monomials
            .intern(&expressions, &programs, &[], &[scoped(a), scoped(c), scoped(b)])
            .unwrap();
        let nf = |id| {
            Arc::new(PolynomialNF {
                exact_terms: BTreeMap::from([(id, BigInt::from(1_u8))]),
                bounded_summary: BoundedSummary::missing(),
            })
        };
        let a_nf = nf(a_id);
        let old_add_leaf = Arc::downgrade(&a_nf);
        let node = expressions.node_arc(product).unwrap();
        let proof = expressions.scope_proof(semantic.program(), root).unwrap();
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let a_state = normalizer.materialized_exact_state(a, Arc::clone(&a_nf)).unwrap();
        let b_state = normalizer.materialized_exact_state(b, nf(b_id)).unwrap();
        let sum_plan = normalizer.new_additive_plan(sum, a_state, b_state, false).unwrap();
        let c_state = normalizer.materialized_exact_state(c, nf(c_id)).unwrap();
        normalizer.relation_rewriting_enabled = false;
        normalizer.deferred_products.insert(product);
        let product_plan = normalizer
            .new_product_plan(
                &proof,
                product,
                node.as_ref(),
                NodeExactState::Additive(sum_plan),
                c_state,
            )
            .unwrap()
            .unwrap();
        normalizer.product_plans.insert(product, Arc::clone(&product_plan));
        drop(a_nf);
        assert!(old_add_leaf.upgrade().is_some());

        normalizer.protected_monomial_prefix = 0;
        normalizer.normalization_depth = 1;
        normalizer.monomial_gc_allocation_threshold_bytes = 0;
        normalizer.sweep_monomials_at_node_commit().unwrap();
        for live in [a_id, b_id, c_id] {
            normalizer.monomials.descriptor(live).unwrap();
        }
        assert!(matches!(
            normalizer.monomials.descriptor(dead),
            Err(MonomialError::CollectedMonomialId { .. })
        ));
        let result = normalizer
            .materialize_exact_state(&NodeExactState::Product(Arc::clone(&product_plan)))
            .unwrap();
        assert_eq!(result.term_count(), 2);
        normalizer.product_plans.clear();
        drop(product_plan);
        drop(normalizer);
        assert!(old_add_leaf.upgrade().is_none());
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
    fn deferred_scalar_action_matches_eager_on_both_sides_and_streams() {
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
            let (deferred, counters) = {
                let mut normalizer =
                    Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
                let value = normalizer.normalize(root_semantic).unwrap();
                (value, normalizer.product_plan_counters)
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
            assert_eq!(counters.scalar_action_plans_created, 1);
            assert_eq!(counters.scalar_action_streamed_executions, 1);
            assert_eq!(counters.scalar_action_standalone_materializations, 0);
            assert_eq!(counters.scalar_action_zero_weight_skips, 0);
        }
    }

    #[test]
    fn deferred_scalar_actions_share_one_additive_matrix_output_and_cancel() {
        for subtract_last in [false, true] {
            let mut expressions = ExprArena::new();
            let mut programs = ProgramArena::new();
            let scalar_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap();
            let matrix_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 2).unwrap();
            let a = matrix_source(
                &mut expressions,
                "shared-scalar-action-a",
                matrix_type.clone(),
                None,
            );
            let b = matrix_source(&mut expressions, "shared-scalar-action-b", matrix_type, None);
            let shared =
                expressions.intern_matrix_transform(MatrixOperation::Add, &[a, b]).unwrap();
            let actions = (0..4)
                .map(|index| {
                    let scalar = matrix_source(
                        &mut expressions,
                        &format!("shared-scalar-action-scalar-{index}"),
                        scalar_type.clone(),
                        None,
                    );
                    expressions
                        .intern_matrix_transform(MatrixOperation::Multiply, &[shared, scalar])
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let left = expressions
                .intern_matrix_transform(MatrixOperation::Add, &[actions[0], actions[1]])
                .unwrap();
            let right = expressions
                .intern_matrix_transform(MatrixOperation::Add, &[actions[2], actions[3]])
                .unwrap();
            let root = if subtract_last {
                expressions
                    .intern_matrix_transform(MatrixOperation::Subtract, &[actions[0], actions[0]])
                    .unwrap()
            } else {
                expressions.intern_matrix_transform(MatrixOperation::Add, &[left, right]).unwrap()
            };
            let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
            let value = normalizer.normalize(semantic).unwrap();
            assert_eq!(
                normalizer.product_plan_counters.scalar_action_plans_created,
                if subtract_last { 1 } else { 4 }
            );
            assert_eq!(
                normalizer.product_plan_counters.scalar_action_streamed_executions,
                if subtract_last { 0 } else { 4 }
            );
            assert_eq!(
                normalizer.product_plan_counters.scalar_action_standalone_materializations,
                0
            );
            assert_eq!(
                normalizer.product_plan_counters.scalar_action_zero_weight_skips,
                u64::from(subtract_last)
            );
            assert!(normalizer.exact_plan_materializations <= 2);
            let exact = value.exact_nf.unwrap();
            if subtract_last {
                assert!(exact.is_zero());
            } else {
                assert_eq!(exact.term_count(), 8);
                assert!(
                    exact
                        .exact_terms
                        .values()
                        .all(|coefficient| coefficient == &BigInt::from(1_u8))
                );
            }
        }
    }

    #[test]
    fn deferred_multi_term_scalar_actions_match_eager_on_both_sides() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let scalar_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap();
        let matrix_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 2).unwrap();
        let a = matrix_source(&mut expressions, "multi-scalar-a", matrix_type.clone(), None);
        let b = matrix_source(&mut expressions, "multi-scalar-b", matrix_type, None);
        let shared = expressions.intern_matrix_transform(MatrixOperation::Add, &[a, b]).unwrap();
        let s1 = matrix_source(&mut expressions, "multi-scalar-s1", scalar_type.clone(), None);
        let s2 = matrix_source(&mut expressions, "multi-scalar-s2", scalar_type.clone(), None);
        let s3 = matrix_source(&mut expressions, "multi-scalar-s3", scalar_type, None);
        let difference =
            expressions.intern_matrix_transform(MatrixOperation::Subtract, &[s1, s2]).unwrap();
        let sum = expressions.intern_matrix_transform(MatrixOperation::Add, &[s2, s3]).unwrap();
        let cancelled =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[difference, sum]).unwrap();
        let scalar =
            expressions.intern_matrix_transform(MatrixOperation::Negate, &[cancelled]).unwrap();
        let left = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[scalar, shared])
            .unwrap();
        let right = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[shared, scalar])
            .unwrap();
        let action_sum =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[left, right]).unwrap();
        let action_difference =
            expressions.intern_matrix_transform(MatrixOperation::Subtract, &[left, right]).unwrap();
        let root = expressions
            .intern_matrix_transform(MatrixOperation::Add, &[action_sum, action_difference])
            .unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let scope = semantic.program();
        let scoped = |expression| programs.scoped(&expressions, scope, expression).unwrap();
        let left_semantic = scoped(left);
        let right_semantic = scoped(right);
        let sum_semantic = scoped(action_sum);
        let difference_semantic = scoped(action_difference);

        let eager_left = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(left_semantic)
            .unwrap();
        let eager_right = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(right_semantic)
            .unwrap();
        assert_eq!(
            descriptor_coefficient_multiset(eager_left.exact_nf.as_ref().unwrap(), &monomials),
            descriptor_coefficient_multiset(eager_right.exact_nf.as_ref().unwrap(), &monomials)
        );
        let eager_sum = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
            normalizer
                .add_nf(
                    eager_left.exact_nf.as_ref().unwrap(),
                    eager_right.exact_nf.as_ref().unwrap(),
                    false,
                )
                .unwrap()
        };
        assert_eq!(eager_left.exact_nf.as_ref().unwrap().term_count(), 4);
        assert_eq!(eager_sum.term_count(), 4);
        assert!(eager_sum.exact_terms.values().all(|coefficient| coefficient == &BigInt::from(-2)));

        let (deferred_sum, sum_counters) = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
            let value = normalizer.normalize(sum_semantic).unwrap();
            (value, normalizer.product_plan_counters)
        };
        assert_eq!(eager_left.coefficient_bound, eager_right.coefficient_bound);
        assert_eq!(
            eager_sum.bounded_summary,
            deferred_sum.exact_nf.as_ref().unwrap().bounded_summary
        );
        assert_eq!(
            descriptor_coefficient_multiset(&eager_sum, &monomials),
            descriptor_coefficient_multiset(deferred_sum.exact_nf.as_ref().unwrap(), &monomials)
        );
        assert_eq!(sum_counters.scalar_action_plans_created, 2);
        assert_eq!(sum_counters.scalar_action_streamed_executions, 2);

        let (deferred_difference, difference_counters) = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
            let value = normalizer.normalize(difference_semantic).unwrap();
            (value, normalizer.product_plan_counters)
        };
        assert!(deferred_difference.exact_nf.unwrap().is_zero());
        assert_eq!(difference_counters.scalar_action_plans_created, 2);
        assert_eq!(difference_counters.scalar_action_streamed_executions, 2);
    }

    #[test]
    fn product_use_counts_include_each_real_additive_output_destination() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let scalar_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap();
        let matrix_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 2).unwrap();
        let a = matrix_source(&mut expressions, "use-count-a", matrix_type.clone(), None);
        let b = matrix_source(&mut expressions, "use-count-b", matrix_type.clone(), None);
        let x = matrix_source(&mut expressions, "use-count-x", matrix_type, None);
        let scalar = matrix_source(&mut expressions, "use-count-scalar", scalar_type, None);
        let product =
            expressions.intern_matrix_transform(MatrixOperation::Multiply, &[a, b]).unwrap();
        let shared =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[product, x]).unwrap();
        let scalar_action = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[shared, scalar])
            .unwrap();
        let root = expressions
            .intern_matrix_transform(MatrixOperation::Add, &[product, scalar_action])
            .unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let proof = expressions.scope_proof(semantic.program(), root).unwrap();
        let scoped =
            |expression| programs.scoped(&expressions, semantic.program(), expression).unwrap();
        let a_id = monomials.intern(&expressions, &programs, &[], &[scoped(a)]).unwrap();
        let b_id = monomials.intern(&expressions, &programs, &[], &[scoped(b)]).unwrap();
        let x_id = monomials.intern(&expressions, &programs, &[], &[scoped(x)]).unwrap();
        let scalar_semantic = scoped(scalar);
        let scalar_id = monomials
            .intern_with_proof(&expressions, &programs, &proof, &[scalar_semantic], &[])
            .unwrap();
        let dead = monomials
            .intern(&expressions, &programs, &[], &[scoped(a), scoped(x), scoped(b)])
            .unwrap();
        let scalar_action_semantic = scoped(scalar_action);
        let nf = |id| {
            Arc::new(PolynomialNF {
                exact_terms: BTreeMap::from([(id, BigInt::from(1_u8))]),
                bounded_summary: BoundedSummary::missing(),
            })
        };
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let a_state = normalizer.materialized_exact_state(a, nf(a_id)).unwrap();
        let b_state = normalizer.materialized_exact_state(b, nf(b_id)).unwrap();
        let product_plan = Arc::new(ProductExactPlan {
            id: 1,
            authority: normalizer.exact_plan_authority(product).unwrap(),
            expression: product,
            left_expression: a,
            right_expression: b,
            left_type: Normalizer::node_exact_authority(&a_state).matrix_type.clone(),
            right_type: Normalizer::node_exact_authority(&b_state).matrix_type.clone(),
            mode: ProductMode::Ordinary,
            left: a_state,
            right: b_state,
        });
        let x_state = normalizer.materialized_exact_state(x, nf(x_id)).unwrap();
        let shared_plan = Arc::new(AdditiveExactPlan {
            id: 2,
            authority: normalizer.exact_plan_authority(shared).unwrap(),
            left: NodeExactState::Product(Arc::clone(&product_plan)),
            right: x_state,
            subtract_right: false,
        });
        let scalar_nf = nf(scalar_id);
        let scalar_state =
            normalizer.materialized_exact_state(scalar, Arc::clone(&scalar_nf)).unwrap();
        let scalar_plan = Arc::new(ProductExactPlan {
            id: 3,
            authority: normalizer.exact_plan_authority(scalar_action).unwrap(),
            expression: scalar_action,
            left_expression: shared,
            right_expression: scalar,
            left_type: normalizer.exact_plan_authority(shared).unwrap().matrix_type.clone(),
            right_type: normalizer.exact_plan_authority(scalar).unwrap().matrix_type.clone(),
            mode: ProductMode::ScalarAction(ScalarActionExactPlan {
                scalar_on_left: false,
                scalar_expression: scalar,
                matrix_expression: shared,
                scalar_type: normalizer.exact_plan_authority(scalar).unwrap().matrix_type.clone(),
                matrix_type: normalizer.exact_plan_authority(shared).unwrap().matrix_type.clone(),
                centralized_scalar: scalar_nf,
            }),
            left: NodeExactState::Additive(shared_plan),
            right: scalar_state,
        });
        normalizer.product_plans.insert(scalar_action, Arc::clone(&scalar_plan));
        normalizer.insert_value_cache(
            scalar_action,
            Arc::new(AnalyzedValue {
                semantic: scalar_action_semantic,
                exact_nf: None,
                coefficient_bound: NumericContract::Missing,
            }),
        );
        normalizer.protected_monomial_prefix = 0;
        normalizer.normalization_depth = 1;
        normalizer.monomial_gc_allocation_threshold_bytes = 0;
        normalizer.sweep_monomials_at_node_commit().unwrap();
        normalizer.normalization_depth = 0;
        for live in [a_id, b_id, x_id, scalar_id] {
            normalizer.monomials.descriptor(live).unwrap();
        }
        let centralized_scalar = normalizer.monomials.descriptor(scalar_id).unwrap();
        assert_eq!(centralized_scalar.central_factors.as_ref(), &[scalar_semantic]);
        assert!(centralized_scalar.ordered_factors.is_empty());
        assert!(matches!(
            normalizer.monomials.descriptor(dead),
            Err(MonomialError::CollectedMonomialId { .. })
        ));
        normalizer.clear_value_cache();
        let root_state = NodeExactState::Additive(Arc::new(AdditiveExactPlan {
            id: 4,
            authority: normalizer.exact_plan_authority(root).unwrap(),
            left: NodeExactState::Product(product_plan),
            right: NodeExactState::Product(scalar_plan),
            subtract_right: false,
        }));
        let boundaries = Normalizer::product_operand_additives(&root_state);
        let counts = normalizer.product_plan_use_counts(&root_state, &boundaries).unwrap();
        assert_eq!(counts.get(&1), Some(&2));
        assert_eq!(counts.get(&3), Some(&1));
        assert_eq!(normalizer.materialize_exact_state(&root_state).unwrap().term_count(), 3);
    }

    #[test]
    fn scalar_action_plan_rejects_invalid_leaves_and_authority_before_mutation() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let scalar_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap();
        let matrix_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 2).unwrap();
        let matrix = matrix_source(
            &mut expressions,
            "invalid-scalar-action-matrix",
            matrix_type.clone(),
            None,
        );
        let scalar = matrix_source(
            &mut expressions,
            "invalid-scalar-action-scalar",
            scalar_type.clone(),
            None,
        );
        let action = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[matrix, scalar])
            .unwrap();
        let root =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[action, matrix]).unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let matrix_semantic = programs.scoped(&expressions, semantic.program(), matrix).unwrap();
        let scalar_semantic = programs.scoped(&expressions, semantic.program(), scalar).unwrap();
        let matrix_id = monomials.intern(&expressions, &programs, &[], &[matrix_semantic]).unwrap();
        let scalar_id = monomials.intern(&expressions, &programs, &[scalar_semantic], &[]).unwrap();
        let tombstone = monomials
            .intern(&expressions, &programs, &[], &[matrix_semantic, matrix_semantic])
            .unwrap();
        monomials.sweep(0, [matrix_id, scalar_id]).unwrap();
        let mut foreign_arena =
            MonomialArena::new(&expressions, &programs, semantic.program()).unwrap();
        let foreign =
            foreign_arena.intern(&expressions, &programs, &[], &[matrix_semantic]).unwrap();
        let nf = |monomial| {
            Arc::new(PolynomialNF {
                exact_terms: BTreeMap::from([(monomial, BigInt::from(1_u8))]),
                bounded_summary: BoundedSummary::missing(),
            })
        };
        let matrix_nf = nf(matrix_id);
        let scalar_nf = nf(scalar_id);
        let scalar_nf_equal_but_distinct = nf(scalar_id);
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let action_authority = normalizer.exact_plan_authority(action).unwrap();
        let matrix_authority = normalizer.exact_plan_authority(matrix).unwrap();
        let scalar_authority = normalizer.exact_plan_authority(scalar).unwrap();
        let materialized = |authority: ExactPlanAuthority, normal_form: Arc<PolynomialNF>| {
            NodeExactState::Materialized { authority, normal_form }
        };
        let plan = |matrix_normal_form: Arc<PolynomialNF>,
                    scalar_state_normal_form: Arc<PolynomialNF>,
                    centralized_scalar: Arc<PolynomialNF>,
                    scalar_on_left: bool,
                    mode_scalar_type: ResolvedMatrixType,
                    mode_matrix_type: ResolvedMatrixType| {
            NodeExactState::Product(Arc::new(ProductExactPlan {
                id: 1,
                authority: action_authority.clone(),
                expression: action,
                left_expression: matrix,
                right_expression: scalar,
                left_type: matrix_type.clone(),
                right_type: scalar_type.clone(),
                mode: ProductMode::ScalarAction(ScalarActionExactPlan {
                    scalar_on_left,
                    scalar_expression: scalar,
                    matrix_expression: matrix,
                    scalar_type: mode_scalar_type,
                    matrix_type: mode_matrix_type,
                    centralized_scalar,
                }),
                left: materialized(matrix_authority.clone(), matrix_normal_form),
                right: materialized(scalar_authority.clone(), scalar_state_normal_form),
            }))
        };
        let invalid_states = [
            plan(
                nf(foreign),
                Arc::clone(&scalar_nf),
                Arc::clone(&scalar_nf),
                false,
                scalar_type.clone(),
                matrix_type.clone(),
            ),
            plan(
                nf(tombstone),
                Arc::clone(&scalar_nf),
                Arc::clone(&scalar_nf),
                false,
                scalar_type.clone(),
                matrix_type.clone(),
            ),
            plan(
                Arc::clone(&matrix_nf),
                Arc::clone(&scalar_nf),
                scalar_nf_equal_but_distinct,
                false,
                scalar_type.clone(),
                matrix_type.clone(),
            ),
            plan(
                Arc::clone(&matrix_nf),
                Arc::clone(&scalar_nf),
                Arc::clone(&scalar_nf),
                true,
                scalar_type.clone(),
                matrix_type.clone(),
            ),
            plan(
                matrix_nf,
                Arc::clone(&scalar_nf),
                scalar_nf,
                false,
                matrix_type.clone(),
                scalar_type.clone(),
            ),
        ];
        for state in invalid_states {
            let high_water = normalizer.monomials.len();
            let occupied = normalizer.monomials.occupied_len();
            let counters = normalizer.product_plan_counters;
            assert!(normalizer.materialize_exact_state(&state).is_err());
            assert_eq!(normalizer.monomials.len(), high_water);
            assert_eq!(normalizer.monomials.occupied_len(), occupied);
            assert_eq!(normalizer.product_plan_counters, counters);
        }
    }

    #[test]
    fn deferred_scalar_action_keeps_both_scalar_and_composite_scalar_eager() {
        let run = |composite_scalar: bool| {
            let mut expressions = ExprArena::new();
            let mut programs = ProgramArena::new();
            let scalar_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap();
            let matrix_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 2).unwrap();
            let scalar = if composite_scalar {
                let row = matrix_source(
                    &mut expressions,
                    "deferred-composite-scalar-row",
                    ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 2).unwrap(),
                    None,
                );
                let column = matrix_source(
                    &mut expressions,
                    "deferred-composite-scalar-column",
                    ResolvedMatrixType::new(BigUint::from(17_u8), 4, 2, 1).unwrap(),
                    None,
                );
                expressions
                    .intern_matrix_transform(MatrixOperation::Multiply, &[row, column])
                    .unwrap()
            } else {
                matrix_source(
                    &mut expressions,
                    "deferred-both-scalar-left",
                    scalar_type.clone(),
                    None,
                )
            };
            let other = matrix_source(
                &mut expressions,
                "deferred-scalar-action-other",
                if composite_scalar { matrix_type.clone() } else { scalar_type.clone() },
                None,
            );
            let action = expressions
                .intern_matrix_transform(MatrixOperation::Multiply, &[other, scalar])
                .unwrap();
            let zero_source = matrix_source(
                &mut expressions,
                "deferred-scalar-action-fallback-zero",
                if composite_scalar { matrix_type } else { scalar_type },
                None,
            );
            let zero = expressions
                .intern_matrix_transform(MatrixOperation::Subtract, &[zero_source, zero_source])
                .unwrap();
            let root =
                expressions.intern_matrix_transform(MatrixOperation::Add, &[action, zero]).unwrap();
            let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
            let value = normalizer.normalize(semantic).unwrap();
            assert_eq!(normalizer.product_plan_counters.scalar_action_plans_created, 0);
            assert_eq!(normalizer.product_plan_counters.scalar_action_streamed_executions, 0);
            assert_eq!(value.exact_nf.unwrap().term_count(), 1);
        };
        run(false);
        run(true);
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
    fn persistent_additive_plan_materializes_shared_fanout_once() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let left = source_with(&mut expressions, matrix_type(), 10_901);
        let right = source_with(&mut expressions, matrix_type(), 10_902);
        let mut root =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[left, right]).unwrap();
        for _ in 0..16 {
            root =
                expressions.intern_matrix_transform(MatrixOperation::Add, &[root, root]).unwrap();
        }
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let value = normalizer.normalize(semantic).unwrap();
        let exact = value.exact_nf.unwrap();
        assert_eq!(exact.exact_terms.len(), 2);
        assert!(
            exact.exact_terms.values().all(|coefficient| coefficient == &BigInt::from(1_u64 << 16))
        );
        assert_eq!(normalizer.exact_plan_materializations, 1);
        assert_eq!(normalizer.exact_plan_materialization_output_terms_total, 2);
        assert_eq!(normalizer.exact_plan_materialization_output_terms_max, 2);
    }

    #[test]
    fn persistent_additive_plan_merges_distinct_sibling_fanouts() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let x = source_with(&mut expressions, matrix_type(), 10_912);
        let y = source_with(&mut expressions, matrix_type(), 10_913);
        let mut left = expressions.intern_matrix_transform(MatrixOperation::Add, &[x, y]).unwrap();
        let mut right =
            expressions.intern_matrix_transform(MatrixOperation::Subtract, &[x, y]).unwrap();
        for _ in 0..16 {
            left =
                expressions.intern_matrix_transform(MatrixOperation::Add, &[left, left]).unwrap();
            right =
                expressions.intern_matrix_transform(MatrixOperation::Add, &[right, right]).unwrap();
        }
        let root =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[left, right]).unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let value = normalizer.normalize(semantic).unwrap();
        let exact = value.exact_nf.unwrap();
        assert_eq!(exact.exact_terms.len(), 1);
        assert_eq!(exact.exact_terms.values().next(), Some(&BigInt::from(1_u64 << 17)));
        assert_eq!(normalizer.exact_plan_materializations, 1);
    }

    #[test]
    fn persistent_additive_plan_cancels_and_materializes_before_multiply() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let x = source_with(&mut expressions, matrix_type(), 10_903);
        let y = source_with(&mut expressions, matrix_type(), 10_904);
        let sum = expressions.intern_matrix_transform(MatrixOperation::Add, &[x, y]).unwrap();
        let cancelled =
            expressions.intern_matrix_transform(MatrixOperation::Subtract, &[sum, sum]).unwrap();
        let root = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[cancelled, x])
            .unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let cancelled_semantic =
            programs.scoped(&expressions, monomials.scope(), cancelled).unwrap();
        let cancelled_value = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(cancelled_semantic)
            .unwrap();
        assert_eq!(
            cancelled_value.coefficient_bound,
            NumericContract::Known(CoefficientBound::ExactZero)
        );
        assert_eq!(
            cancelled_value.exact_nf.as_ref().unwrap().bounded_summary.coefficient_bound,
            cancelled_value.coefficient_bound
        );
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let value = normalizer.normalize(semantic).unwrap();
        assert!(value.exact_nf.unwrap().is_zero());
        assert_eq!(value.coefficient_bound, NumericContract::Known(CoefficientBound::ExactZero));
        assert_eq!(normalizer.exact_plan_materializations, 1);
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
    fn compositional_product_matches_eager_additive_operands_and_streams_once() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let matrix = matrix_type();
        let a = source_with(&mut expressions, matrix.clone(), 31_001);
        let b = source_with(&mut expressions, matrix.clone(), 31_002);
        let c = source_with(&mut expressions, matrix.clone(), 31_003);
        let d = source_with(&mut expressions, matrix.clone(), 31_004);
        let left = expressions.intern_matrix_transform(MatrixOperation::Add, &[a, b]).unwrap();
        let right =
            expressions.intern_matrix_transform(MatrixOperation::Subtract, &[c, d]).unwrap();
        let product =
            expressions.intern_matrix_transform(MatrixOperation::Multiply, &[left, right]).unwrap();
        let zero_source = source_with(&mut expressions, matrix, 31_005);
        let zero = expressions
            .intern_matrix_transform(MatrixOperation::Subtract, &[zero_source, zero_source])
            .unwrap();
        let root =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[product, zero]).unwrap();
        let (facts, mut monomials, root_semantic) = setup(&mut expressions, &mut programs, root);
        let product_semantic = programs.scoped(&expressions, monomials.scope(), product).unwrap();
        let eager = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .normalize(product_semantic)
            .unwrap();
        let (deferred, counters) = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
            let value = normalizer.normalize(root_semantic).unwrap();
            (value, normalizer.product_plan_counters)
        };
        assert_eq!(eager.coefficient_bound, deferred.coefficient_bound);
        assert_eq!(
            descriptor_coefficient_multiset(eager.exact_nf.as_ref().unwrap(), &monomials),
            descriptor_coefficient_multiset(deferred.exact_nf.as_ref().unwrap(), &monomials)
        );
        assert_eq!(counters.plans_created, 1);
        assert_eq!(counters.streamed_executions, 1);
        assert_eq!(counters.standalone_materializations, 0);
    }

    #[test]
    fn compositional_product_chain_keeps_intermediate_out_of_the_node_cache() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let matrix = matrix_type();
        let a = source_with(&mut expressions, matrix.clone(), 31_011);
        let b = source_with(&mut expressions, matrix.clone(), 31_012);
        let c = source_with(&mut expressions, matrix.clone(), 31_013);
        let d = source_with(&mut expressions, matrix.clone(), 31_014);
        let sum = expressions.intern_matrix_transform(MatrixOperation::Add, &[a, b]).unwrap();
        let first =
            expressions.intern_matrix_transform(MatrixOperation::Multiply, &[sum, c]).unwrap();
        let second =
            expressions.intern_matrix_transform(MatrixOperation::Multiply, &[first, d]).unwrap();
        let zero = expressions.intern_matrix_transform(MatrixOperation::Subtract, &[a, a]).unwrap();
        let root =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[second, zero]).unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let value = normalizer.normalize(semantic).unwrap();
        assert_eq!(value.exact_nf.as_ref().unwrap().term_count(), 2);
        assert_eq!(normalizer.product_plan_counters.plans_created, 2);
        assert_eq!(normalizer.product_plan_counters.streamed_executions, 1);
        assert_eq!(normalizer.product_plan_counters.standalone_materializations, 1);
        assert!(normalizer.product_plans.is_empty());
        assert!(normalizer.cache.is_empty());
    }

    #[test]
    fn compositional_product_deep_chain_is_iterative_and_eligibility_is_linear() {
        const DEPTH: usize = 4_096;
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let matrix = matrix_type();
        let first = source_with(&mut expressions, matrix.clone(), 31_100);
        let mut chain = first;
        let mut products = Vec::with_capacity(DEPTH);
        for offset in 0..DEPTH {
            let factor = source_with(
                &mut expressions,
                matrix.clone(),
                31_101_u64.checked_add(u64::try_from(offset).unwrap()).unwrap(),
            );
            chain = expressions
                .intern_matrix_transform(MatrixOperation::Multiply, &[chain, factor])
                .unwrap();
            products.push(chain);
        }
        let zero = expressions
            .intern_matrix_transform(MatrixOperation::Subtract, &[first, first])
            .unwrap();
        let root =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[chain, zero]).unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();

        let candidates = products.iter().copied().collect::<BTreeSet<_>>();
        let mut consumers = BTreeMap::<ExprId, BTreeSet<ExprId>>::new();
        for pair in products.windows(2) {
            consumers.entry(pair[0]).or_default().insert(pair[1]);
        }
        consumers.entry(*products.last().unwrap()).or_default().insert(root);
        let (eligible, stats) =
            normalizer.propagate_product_eligibility(&candidates, &consumers).unwrap();
        assert_eq!(eligible, candidates);
        assert_eq!(stats.candidates, DEPTH);
        assert_eq!(stats.consumer_edges, DEPTH);
        assert_eq!(stats.queue_pops, DEPTH);

        let value = normalizer.normalize(semantic).unwrap();
        assert_eq!(value.exact_nf.as_ref().unwrap().term_count(), 1);
        assert_eq!(normalizer.product_plan_counters.plans_created, DEPTH as u64);
        assert_eq!(normalizer.product_plan_counters.streamed_executions, 1);
        assert_eq!(normalizer.product_plan_counters.standalone_materializations, DEPTH as u64 - 1);
    }

    #[test]
    fn compositional_product_diamond_executes_each_shared_product_once() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let matrix = matrix_type();
        let a = source_with(&mut expressions, matrix.clone(), 31_201);
        let b = source_with(&mut expressions, matrix.clone(), 31_202);
        let c = source_with(&mut expressions, matrix.clone(), 31_203);
        let d = source_with(&mut expressions, matrix, 31_204);
        let shared =
            expressions.intern_matrix_transform(MatrixOperation::Multiply, &[a, b]).unwrap();
        let left =
            expressions.intern_matrix_transform(MatrixOperation::Multiply, &[shared, c]).unwrap();
        let right =
            expressions.intern_matrix_transform(MatrixOperation::Multiply, &[shared, d]).unwrap();
        let root =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[left, right]).unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let value = normalizer.normalize(semantic).unwrap();
        assert_eq!(value.exact_nf.as_ref().unwrap().term_count(), 2);
        assert_eq!(normalizer.product_plan_counters.plans_created, 3);
        assert_eq!(normalizer.product_plan_counters.streamed_executions, 2);
        assert_eq!(normalizer.product_plan_counters.standalone_materializations, 1);
    }

    #[test]
    fn compositional_product_canceled_subtrees_release_dependencies_without_execution() {
        for keep_child in [false, true] {
            let mut expressions = ExprArena::new();
            let mut programs = ProgramArena::new();
            let matrix = matrix_type();
            let a = source_with(&mut expressions, matrix.clone(), 31_301);
            let b = source_with(&mut expressions, matrix.clone(), 31_302);
            let c = source_with(&mut expressions, matrix, 31_303);
            let child =
                expressions.intern_matrix_transform(MatrixOperation::Multiply, &[a, b]).unwrap();
            let shared_additive =
                expressions.intern_matrix_transform(MatrixOperation::Add, &[child, c]).unwrap();
            let parent = expressions
                .intern_matrix_transform(MatrixOperation::Multiply, &[shared_additive, c])
                .unwrap();
            let canceled = expressions
                .intern_matrix_transform(MatrixOperation::Subtract, &[parent, parent])
                .unwrap();
            let root = if keep_child {
                expressions
                    .intern_matrix_transform(MatrixOperation::Add, &[canceled, shared_additive])
                    .unwrap()
            } else {
                canceled
            };
            let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
            let value = normalizer.normalize(semantic).unwrap();
            assert!(normalizer.product_plans.is_empty());
            assert!(normalizer.cache.is_empty());
            if keep_child {
                assert_eq!(value.exact_nf.as_ref().unwrap().term_count(), 2);
                assert_eq!(
                    normalizer.product_plan_counters.streamed_executions +
                        normalizer.product_plan_counters.standalone_materializations,
                    1
                );
            } else {
                assert!(value.exact_nf.as_ref().unwrap().is_zero());
                assert_eq!(normalizer.product_plan_counters.streamed_executions, 0);
                assert_eq!(normalizer.product_plan_counters.standalone_materializations, 0);
            }
        }
    }

    #[test]
    fn compositional_product_shared_additive_operand_consumes_product_edges_once() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let matrix = matrix_type();
        let a = source_with(&mut expressions, matrix.clone(), 31_401);
        let b = source_with(&mut expressions, matrix.clone(), 31_402);
        let c = source_with(&mut expressions, matrix.clone(), 31_403);
        let d = source_with(&mut expressions, matrix, 31_404);
        let product =
            expressions.intern_matrix_transform(MatrixOperation::Multiply, &[a, b]).unwrap();
        let shared_additive =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[product, c]).unwrap();
        let parent = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[shared_additive, d])
            .unwrap();
        let root = expressions
            .intern_matrix_transform(MatrixOperation::Add, &[parent, shared_additive])
            .unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let value = normalizer.normalize(semantic).unwrap();
        assert_eq!(value.exact_nf.as_ref().unwrap().term_count(), 4);
        assert_eq!(normalizer.product_plan_counters.plans_created, 2);
        assert_eq!(normalizer.product_plan_counters.streamed_executions, 2);
        assert_eq!(normalizer.product_plan_counters.standalone_materializations, 0);
    }

    #[test]
    fn compositional_product_aggregates_shared_weights_and_skips_exact_zero() {
        for (subtract, expected_executions, expected_zero_skips, expected_coefficient) in
            [(false, 1, 0, Some(BigInt::from(2_u8))), (true, 0, 1, None)]
        {
            let mut expressions = ExprArena::new();
            let mut programs = ProgramArena::new();
            let matrix = matrix_type();
            let a = source_with(&mut expressions, matrix.clone(), 31_021);
            let b = source_with(&mut expressions, matrix, 31_022);
            let product =
                expressions.intern_matrix_transform(MatrixOperation::Multiply, &[a, b]).unwrap();
            let operation = if subtract { MatrixOperation::Subtract } else { MatrixOperation::Add };
            let root = expressions.intern_matrix_transform(operation, &[product, product]).unwrap();
            let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
            let value = normalizer.normalize(semantic).unwrap();
            let exact = value.exact_nf.unwrap();
            assert_eq!(normalizer.product_plan_counters.plans_created, 1);
            assert_eq!(normalizer.product_plan_counters.streamed_executions, expected_executions);
            assert_eq!(normalizer.product_plan_counters.zero_weight_skips, expected_zero_skips);
            match expected_coefficient {
                Some(coefficient) => {
                    assert_eq!(exact.exact_terms.len(), 1);
                    assert_eq!(exact.exact_terms.values().next(), Some(&coefficient));
                }
                None => assert!(exact.is_zero()),
            }
        }
    }

    #[test]
    fn compositional_product_nests_the_typed_gadget_plan_without_semantic_drift() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let matrix = ResolvedMatrixType::new(BigUint::from(17_u8), 2, 2, 2).unwrap();
        let (gadget, decomposition, _) = gadget_product(
            &mut expressions,
            false,
            1,
            matrix.clone(),
            matrix.clone(),
            matrix.clone(),
            Some((2, false)),
        );
        let recomposed = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[gadget, decomposition])
            .unwrap();
        let suffix = matrix_source(&mut expressions, "nested-product-suffix", matrix.clone(), None);
        let product = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[recomposed, suffix])
            .unwrap();
        let zero_source = matrix_source(&mut expressions, "nested-product-zero", matrix, None);
        let zero = expressions
            .intern_matrix_transform(MatrixOperation::Subtract, &[zero_source, zero_source])
            .unwrap();
        let root =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[product, zero]).unwrap();
        let registry = recomposition_registry(
            ResolvedMatrixType::new(BigUint::from(17_u8), 2, 2, 2).unwrap(),
            ResolvedMatrixType::new(BigUint::from(17_u8), 2, 2, 2).unwrap(),
            ResolvedMatrixType::new(BigUint::from(17_u8), 2, 2, 2).unwrap(),
            false,
            1,
        );
        let (mut facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        mark_scalar_sources_constant(&expressions, &mut facts, root);
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_gadget_recompositions(&registry);
        let value = normalizer.normalize(semantic).unwrap();
        assert_eq!(value.exact_nf.as_ref().unwrap().term_count(), 1);
        assert_eq!(normalizer.gadget_product_counters.plans_created, 1);
        assert_eq!(normalizer.product_plan_counters.plans_created, 1);
        assert_eq!(normalizer.product_plan_counters.streamed_executions, 1);
    }

    #[test]
    fn compositional_typed_gadget_candidate_reclassifies_materialized_endpoints_exactly() {
        #[derive(Clone, Copy, Debug)]
        enum Case {
            Authorized,
            Mixed,
            Canceled,
        }

        for case in [Case::Authorized, Case::Mixed, Case::Canceled] {
            let mut expressions = ExprArena::new();
            let mut programs = ProgramArena::new();
            let modulus = BigUint::from(17_u8);
            let input_type = ResolvedMatrixType::new(modulus.clone(), 1, 1, 4).unwrap();
            let decomposition_type = ResolvedMatrixType::new(modulus.clone(), 1, 4, 4).unwrap();
            let gadget_type = ResolvedMatrixType::new(modulus, 1, 1, 4).unwrap();
            let (gadget, decomposition, _) = gadget_product(
                &mut expressions,
                false,
                4,
                gadget_type.clone(),
                decomposition_type.clone(),
                input_type.clone(),
                Some((2, false)),
            );
            let other = match case {
                Case::Authorized => matrix_source(
                    &mut expressions,
                    "compositional-authorized-gadget",
                    gadget_type.clone(),
                    Some((2, false)),
                ),
                Case::Mixed => matrix_source(
                    &mut expressions,
                    "compositional-ordinary-prefix",
                    gadget_type.clone(),
                    None,
                ),
                Case::Canceled => gadget,
            };
            let additive_operation = if matches!(case, Case::Canceled) {
                MatrixOperation::Subtract
            } else {
                MatrixOperation::Add
            };
            let left =
                expressions.intern_matrix_transform(additive_operation, &[gadget, other]).unwrap();
            let product = expressions
                .intern_matrix_transform(MatrixOperation::Multiply, &[left, decomposition])
                .unwrap();
            let zero_source = matrix_source(
                &mut expressions,
                "compositional-gadget-zero",
                input_type.clone(),
                None,
            );
            let zero = expressions
                .intern_matrix_transform(MatrixOperation::Subtract, &[zero_source, zero_source])
                .unwrap();
            let root = expressions
                .intern_matrix_transform(MatrixOperation::Add, &[product, zero])
                .unwrap();
            let registry =
                recomposition_registry(gadget_type, decomposition_type, input_type, false, 4);
            let (mut facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
            mark_scalar_sources_constant(&expressions, &mut facts, root);
            let product_semantic =
                programs.scoped(&expressions, semantic.program(), product).unwrap();
            let eager = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                .unwrap()
                .with_gadget_recompositions(&registry)
                .normalize(product_semantic)
                .unwrap();
            let (deferred, product_counters, gadget_counters, product_origin) = {
                let mut normalizer =
                    Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                        .unwrap()
                        .with_gadget_recompositions(&registry)
                        .with_watchdog_override(true, Duration::from_secs(60));
                let value = normalizer.normalize(semantic).unwrap();
                (
                    value,
                    normalizer.product_plan_counters,
                    normalizer.gadget_product_counters,
                    normalizer.diagnostic_materialization_origins.get(&product).copied(),
                )
            };
            assert_eq!(
                descriptor_coefficient_multiset(eager.exact_nf.as_ref().unwrap(), &monomials),
                descriptor_coefficient_multiset(deferred.exact_nf.as_ref().unwrap(), &monomials),
                "{case:?}"
            );
            assert_eq!(eager.coefficient_bound, deferred.coefficient_bound, "{case:?}");
            assert_eq!(product_counters.plans_created, 1, "{case:?}");
            assert_eq!(product_counters.typed_candidate_plans, 1, "{case:?}");
            assert_eq!(product_counters.streamed_executions, 1, "{case:?}");
            assert_eq!(product_counters.typed_standalone_materializations, 0, "{case:?}");
            assert_eq!(gadget_counters, GadgetProductPlanCounters::default(), "{case:?}");
            assert_eq!(product_origin, None, "{case:?} must not force a NonAdd child boundary");
            match case {
                Case::Authorized => {
                    assert_eq!(product_counters.typed_direct_executions, 1);
                    assert_eq!(product_counters.typed_pair_attempts, 2);
                    assert_eq!(product_counters.typed_pair_matches, 2);
                    assert_eq!(product_counters.typed_pair_ordinary_fallbacks, 0);
                }
                Case::Mixed => {
                    assert_eq!(product_counters.typed_direct_executions, 1);
                    assert_eq!(product_counters.typed_pair_attempts, 2);
                    assert_eq!(product_counters.typed_pair_matches, 1);
                    assert_eq!(product_counters.typed_pair_ordinary_fallbacks, 1);
                }
                Case::Canceled => {
                    assert_eq!(product_counters.typed_direct_executions, 0);
                    assert_eq!(product_counters.typed_pair_attempts, 0);
                    assert_eq!(product_counters.typed_pair_matches, 0);
                    assert_eq!(product_counters.typed_pair_ordinary_fallbacks, 0);
                }
            }
        }
    }

    #[test]
    fn compositional_product_keeps_nonadd_eager_defers_scalar_add_child_and_materializes_relations()
    {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let matrix = matrix_type();
        let a = source_with(&mut expressions, matrix.clone(), 31_031);
        let b = source_with(&mut expressions, matrix.clone(), 31_032);
        let product =
            expressions.intern_matrix_transform(MatrixOperation::Multiply, &[a, b]).unwrap();
        let negated =
            expressions.intern_matrix_transform(MatrixOperation::Negate, &[product]).unwrap();
        let scalar = matrix_source(
            &mut expressions,
            "ordinary-product-scalar-boundary",
            ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap(),
            None,
        );
        let scaled =
            expressions.intern_matrix_transform(MatrixOperation::Multiply, &[scalar, a]).unwrap();
        let scalar_root =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[scaled, scaled]).unwrap();
        let relation_root =
            expressions.intern_matrix_transform(MatrixOperation::Add, &[product, product]).unwrap();
        let combined = expressions
            .intern_matrix_transform(MatrixOperation::Add, &[negated, scalar_root])
            .and_then(|combined| {
                expressions
                    .intern_matrix_transform(MatrixOperation::Add, &[combined, relation_root])
            })
            .unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, combined);
        let negated_semantic = programs.scoped(&expressions, semantic.program(), negated).unwrap();
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        normalizer.normalize(negated_semantic).unwrap();
        assert_eq!(normalizer.product_plan_counters.plans_created, 0);

        let scalar_semantic =
            programs.scoped(&expressions, semantic.program(), scalar_root).unwrap();
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        normalizer.normalize(scalar_semantic).unwrap();
        assert_eq!(normalizer.product_plan_counters.plans_created, 1);
        assert_eq!(normalizer.product_plan_counters.scalar_action_plans_created, 1);
        assert_eq!(normalizer.product_plan_counters.scalar_action_streamed_executions, 1);
        let relation_semantic =
            programs.scoped(&expressions, semantic.program(), relation_root).unwrap();
        let (relations, mut normalization_cache, _) = register_test_closed_relation(
            &mut expressions,
            &programs,
            &facts,
            &mut monomials,
            a,
            b,
            a,
            b,
        );
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_relations(&relations, &mut normalization_cache);
        let relation_value = normalizer.normalize(relation_semantic).unwrap();
        assert!(relation_value.exact_nf.is_some());
        assert_eq!(normalizer.product_plan_counters.plans_created, 1);
        assert!(normalizer.product_plans.is_empty());
    }

    #[test]
    fn persistent_additive_plan_rejects_type_mismatch_before_materialization() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let source = source(&mut expressions);
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, source);
        let foreign_expression_token = ExprArena::new().token();
        let foreign_monomial_token =
            MonomialArena::new(&expressions, &programs, monomials.scope()).unwrap().token();
        let mut frozen_relations = RelationRegistry::new();
        let foreign_relation_generation = frozen_relations.freeze();
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        let normal_form = Arc::new(PolynomialNF::zero());
        let left = normalizer
            .materialized_exact_state(semantic.expression(), Arc::clone(&normal_form))
            .unwrap();
        let mut wrong = normalizer.exact_plan_authority(semantic.expression()).unwrap();
        wrong.matrix_type = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 1).unwrap();
        let right = NodeExactState::Materialized { authority: wrong, normal_form };
        assert!(matches!(
            normalizer.new_additive_plan(semantic.expression(), left, right, false),
            Err(NormalizeError::InvalidExactPlan { reason: "additive type mismatch" })
        ));
        let next_id = normalizer.next_exact_plan_id;
        let valid = normalizer.exact_plan_authority(semantic.expression()).unwrap();
        let authorities = [
            ExactPlanAuthority { expressions: foreign_expression_token, ..valid.clone() },
            ExactPlanAuthority { monomials: foreign_monomial_token, ..valid.clone() },
            ExactPlanAuthority {
                scope: ValueProgramId::new(ArenaToken::fresh(), 0),
                ..valid.clone()
            },
            ExactPlanAuthority { relations: Some(foreign_relation_generation), ..valid.clone() },
        ];
        for authority in authorities {
            let left = NodeExactState::Materialized {
                authority,
                normal_form: Arc::new(PolynomialNF::zero()),
            };
            let right = normalizer
                .materialized_exact_state(semantic.expression(), Arc::new(PolynomialNF::zero()))
                .unwrap();
            assert!(matches!(
                normalizer.new_additive_plan(semantic.expression(), left, right, false),
                Err(NormalizeError::InvalidExactPlan { .. })
            ));
            assert_eq!(normalizer.next_exact_plan_id, next_id);
            assert!(normalizer.exact_plans.is_empty());
        }
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
    fn additive_plan_gc_rejects_foreign_and_tombstoned_leaf_ids_before_mutation() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let foreign_root = source_with(&mut expressions, matrix_type(), 10_909);
        let (_, mut foreign_arena, foreign_semantic) =
            setup(&mut expressions, &mut programs, foreign_root);
        let foreign =
            foreign_arena.intern(&expressions, &programs, &[], &[foreign_semantic]).unwrap();

        let x = source_with(&mut expressions, matrix_type(), 10_910);
        let y = source_with(&mut expressions, matrix_type(), 10_911);
        let root = expressions.intern_matrix_transform(MatrixOperation::Add, &[x, y]).unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let local = monomials.intern(&expressions, &programs, &[], &[semantic]).unwrap();
        let tombstone =
            monomials.intern(&expressions, &programs, &[], &[semantic, semantic]).unwrap();
        monomials.sweep(0, [local]).unwrap();
        let occupied_before = monomials.occupied_len();
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        normalizer.protected_monomial_prefix = 0;
        normalizer.monomial_gc_allocation_threshold_bytes = 0;
        normalizer.normalization_depth = 1;
        for (invalid, expected_collected) in [(foreign, false), (tombstone, true)] {
            let invalid_nf = Arc::new(PolynomialNF {
                exact_terms: BTreeMap::from([(invalid, BigInt::from(1_u8))]),
                bounded_summary: BoundedSummary::missing(),
            });
            let left = normalizer.materialized_exact_state(x, invalid_nf).unwrap();
            let right =
                normalizer.materialized_exact_state(y, Arc::new(PolynomialNF::zero())).unwrap();
            let plan = normalizer.new_additive_plan(root, left, right, false).unwrap();
            normalizer.exact_plans.insert(root, plan);
            normalizer.insert_value_cache(
                root,
                Arc::new(AnalyzedValue {
                    semantic,
                    exact_nf: None,
                    coefficient_bound: NumericContract::Missing,
                }),
            );
            let error = normalizer.sweep_monomials_at_node_commit().unwrap_err();
            assert!(if expected_collected {
                matches!(error, NormalizeError::Monomial(MonomialError::CollectedMonomialId { .. }))
            } else {
                matches!(error, NormalizeError::Monomial(MonomialError::InvalidMonomialId { .. }))
            });
            assert_eq!(normalizer.monomials.occupied_len(), occupied_before);
            assert_eq!(normalizer.gc_counters, DiagnosticGcCounters::default());
            normalizer.clear_value_cache();
        }
    }

    #[test]
    fn additive_owner_telemetry_counts_unique_plan_nodes_and_leaf_arcs() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let x = source_with(&mut expressions, matrix_type(), 10_907);
        let y = source_with(&mut expressions, matrix_type(), 10_908);
        let sum = expressions.intern_matrix_transform(MatrixOperation::Add, &[x, y]).unwrap();
        let root = expressions.intern_matrix_transform(MatrixOperation::Add, &[sum, sum]).unwrap();
        let (facts, mut monomials, _) = setup(&mut expressions, &mut programs, root);
        let x_semantic = programs.scoped(&expressions, monomials.scope(), x).unwrap();
        let y_semantic = programs.scoped(&expressions, monomials.scope(), y).unwrap();
        let x_monomial = monomials.intern(&expressions, &programs, &[], &[x_semantic]).unwrap();
        let y_monomial = monomials.intern(&expressions, &programs, &[], &[y_semantic]).unwrap();
        let leaf = |monomial| {
            Arc::new(PolynomialNF {
                exact_terms: BTreeMap::from([(monomial, BigInt::from(1_u8))]),
                bounded_summary: BoundedSummary::missing(),
            })
        };
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        normalizer.watchdog = DiagnosticWatchdog::start(19, Duration::from_secs(60));
        normalizer.four_class_census_enabled = true;
        let x_state = normalizer.materialized_exact_state(x, leaf(x_monomial)).unwrap();
        let y_state = normalizer.materialized_exact_state(y, leaf(y_monomial)).unwrap();
        let sum_plan = normalizer.new_additive_plan(sum, x_state, y_state, false).unwrap();
        let root_plan = normalizer
            .new_additive_plan(
                root,
                NodeExactState::Additive(Arc::clone(&sum_plan)),
                NodeExactState::Additive(sum_plan),
                false,
            )
            .unwrap();
        let materialized = normalizer
            .materialize_exact_state(&NodeExactState::Additive(Arc::clone(&root_plan)))
            .unwrap();
        assert_eq!(materialized.term_count(), 2);
        normalizer.exact_plans.insert(root, root_plan);
        normalizer.normalization_depth = 1;
        normalizer.monomial_gc_allocation_threshold_bytes = 0;
        normalizer.sweep_monomials_at_node_commit().unwrap();
        normalizer.normalization_depth = 0;
        assert_eq!(normalizer.gc_counters.materialized_leaf_top8.len, 2);
        assert_eq!(normalizer.gc_counters.materialized_leaf_top8.exact_term_refs, 2);
        assert_eq!(normalizer.gc_counters.materialized_leaf_top8.top[0].producer, Some(x));
        assert_eq!(normalizer.gc_counters.materialized_leaf_top8.top[1].producer, Some(y));
        let classes = normalizer.gc_counters.exact_plan_four_class;
        assert_eq!(classes.large.unique_monomials, 2);
        assert_eq!(classes.large.term_refs, 2);
        assert_eq!(classes.top_len, 2);
        assert_eq!(classes.frontier_unique_union, classes.frontier_reason_unique_union);
        assert!(
            normalizer.gc_counters.four_class_total_ns >= normalizer.gc_counters.four_class_max_ns
        );
        assert!(
            normalizer.gc_counters.four_class_max_ns >= normalizer.gc_counters.four_class_last_ns
        );
        let owners = normalizer.owner_census();
        assert_eq!(owners.additive_plan_nodes, 2);
        assert_eq!(owners.additive_unique_leaf_refs, 2);
        assert_eq!(owners.additive_unique_leaf_exact_term_refs, 2);
        assert_eq!(owners.additive_largest_leaf_exact_terms, 1);
        assert_eq!(owners.additive_materializations, 1);
        assert_eq!(owners.additive_materialization_output_terms_total, 2);
        assert_eq!(owners.additive_materialization_output_terms_max, 2);

        normalizer.sample_owner_census(OwnerCensusReason::OuterTerminal, std::iter::empty());
        let shared = Arc::clone(&normalizer.watchdog.as_ref().unwrap().shared);
        normalizer.watchdog.as_mut().unwrap().finish(false);
        normalizer.watchdog = None;
        assert_eq!(
            normalizer.exact_plan_diagnostic_shape().8,
            DiagnosticMaterializedLeafTop8::default()
        );
        let events = shared.events.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone();
        let snapshots =
            shared.snapshots.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone();
        let captured = events
            .iter()
            .zip(&snapshots)
            .find_map(|(event, snapshot)| (*event == "watchdog_owner_sample").then_some(snapshot))
            .unwrap();
        assert_eq!(captured.owners.additive_plan_nodes, 2);
        assert_eq!(captured.owners.additive_unique_leaf_refs, 2);
        assert_eq!(captured.owners.additive_unique_leaf_exact_term_refs, 2);
        assert_eq!(captured.owners.additive_largest_leaf_exact_terms, 1);
        assert_eq!(captured.owners.additive_materializations, 1);
        assert_eq!(captured.owners.additive_materialization_output_terms_total, 2);
        assert_eq!(captured.owners.additive_materialization_output_terms_max, 2);
        assert_eq!(captured.owners.materialized_leaf_top8.len, 2);
        assert_eq!(captured.owners.materialized_leaf_top8.exact_term_refs, 2);
    }

    #[test]
    fn materialized_leaf_top8_keeps_the_dominant_eight_with_stable_ties() {
        let arena = ArenaToken::fresh();
        let mut top8 = DiagnosticMaterializedLeafTop8::default();
        for slot in (0_u32..10).rev() {
            let terms = if slot == 9 { 884_118 } else { u64::from(slot + 1) };
            top8.observe(
                DiagnosticMaterializedLeafOrigin {
                    producer: Some(ExprId::new(arena, slot)),
                    producer_operator: "add",
                    reason: DiagnosticMaterializationReason::NonAddConsumer,
                    consumer: Some(ExprId::new(arena, 100 + slot)),
                    consumer_operator: "multiply",
                    consumer_category: "multiply",
                    remaining_uses: u64::from(slot),
                    scalar_classification: "ordinary",
                    forced_input_count: 1,
                    forced_terms_sum: terms,
                    forced_terms_max: terms,
                    retained_term_count: 0,
                },
                terms,
            );
        }
        assert_eq!(top8.len, 8);
        assert_eq!(top8.exact_term_refs, 884_163);
        assert_eq!(top8.top[0].producer.map(ExprId::slot), Some(9));
        assert_eq!(top8.top[0].retained_term_count, 884_118);
        assert_eq!(top8.top[7].producer.map(ExprId::slot), Some(2));

        let tied = DiagnosticMaterializedLeafOrigin {
            producer: Some(ExprId::new(arena, 1)),
            retained_term_count: 0,
            ..DiagnosticMaterializedLeafOrigin::default()
        };
        let mut ties = DiagnosticMaterializedLeafTop8::default();
        ties.observe(tied, 7);
        ties.observe(
            DiagnosticMaterializedLeafOrigin { producer: Some(ExprId::new(arena, 0)), ..tied },
            7,
        );
        assert_eq!(ties.top[0].producer.map(ExprId::slot), Some(0));
        assert_eq!(ties.top[1].producer.map(ExprId::slot), Some(1));
    }

    #[test]
    fn materialization_origin_aggregates_two_forced_children_for_one_consumer() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let x = source_with(&mut expressions, matrix_type(), 71_001);
        let y = source_with(&mut expressions, matrix_type(), 71_002);
        let u = source_with(&mut expressions, matrix_type(), 71_003);
        let v = source_with(&mut expressions, matrix_type(), 71_004);
        let left_add = expressions.intern_matrix_transform(MatrixOperation::Add, &[x, y]).unwrap();
        let right_add = expressions.intern_matrix_transform(MatrixOperation::Add, &[u, v]).unwrap();
        let consumer = expressions
            .intern_matrix_transform(MatrixOperation::Multiply, &[left_add, right_add])
            .unwrap();
        let root = expressions
            .intern_matrix_transform(MatrixOperation::Add, &[consumer, consumer])
            .unwrap();
        let (facts, mut monomials, _) = setup(&mut expressions, &mut programs, root);
        let mut ids = BTreeMap::new();
        for expression in [x, y, u, v] {
            let scoped = programs.scoped(&expressions, monomials.scope(), expression).unwrap();
            ids.insert(
                expression,
                monomials.intern(&expressions, &programs, &[], &[scoped]).unwrap(),
            );
        }
        let leaf = |expression| {
            Arc::new(PolynomialNF {
                exact_terms: BTreeMap::from([(ids[&expression], BigInt::from(1_u8))]),
                bounded_summary: BoundedSummary::missing(),
            })
        };
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        normalizer.watchdog = DiagnosticWatchdog::start(20, Duration::from_secs(60));
        normalizer.remaining_uses.insert(left_add, 1);
        normalizer.remaining_uses.insert(right_add, 3);
        let x_state = normalizer.materialized_exact_state(x, leaf(x)).unwrap();
        let y_state = normalizer.materialized_exact_state(y, leaf(y)).unwrap();
        let left_plan = normalizer.new_additive_plan(left_add, x_state, y_state, false).unwrap();
        let u_state = normalizer.materialized_exact_state(u, leaf(u)).unwrap();
        let v_state = normalizer.materialized_exact_state(v, leaf(v)).unwrap();
        let right_plan = normalizer.new_additive_plan(right_add, u_state, v_state, false).unwrap();
        let left_nf = normalizer
            .materialize_exact_state_for(
                &NodeExactState::Additive(left_plan),
                left_add,
                DiagnosticMaterializationReason::NonAddConsumer,
                Some(consumer),
            )
            .unwrap();
        let right_nf = normalizer
            .materialize_exact_state_for(
                &NodeExactState::Additive(right_plan),
                right_add,
                DiagnosticMaterializationReason::NonAddConsumer,
                Some(consumer),
            )
            .unwrap();
        assert_eq!(left_nf.term_count(), 2);
        assert_eq!(right_nf.term_count(), 2);
        let aggregated = normalizer.diagnostic_materialization_origins[&consumer];
        assert_eq!(aggregated.producer, None);
        assert_eq!(aggregated.producer_operator, "multiple");
        assert_eq!(aggregated.consumer, Some(consumer));
        assert_eq!(aggregated.consumer_operator, "multiply");
        assert_eq!(aggregated.consumer_category, "multiply");
        assert_eq!(aggregated.scalar_classification, "ordinary");
        assert_eq!(aggregated.remaining_uses, 3);
        assert_eq!(aggregated.forced_input_count, 2);
        assert_eq!(aggregated.forced_terms_sum, 4);
        assert_eq!(aggregated.forced_terms_max, 2);

        let consumer_nf = Arc::new(PolynomialNF {
            exact_terms: BTreeMap::from([
                (ids[&x], BigInt::from(1_u8)),
                (ids[&y], BigInt::from(1_u8)),
                (ids[&u], BigInt::from(1_u8)),
            ]),
            bounded_summary: BoundedSummary::missing(),
        });
        let consumer_state =
            normalizer.materialized_exact_state(consumer, Arc::clone(&consumer_nf)).unwrap();
        let root_plan = normalizer
            .new_additive_plan(root, consumer_state.clone(), consumer_state, false)
            .unwrap();
        normalizer.exact_plans.insert(root, root_plan);
        let shape = normalizer.exact_plan_diagnostic_shape();
        let top8 = shape.8;
        assert_eq!(top8.len, 1);
        assert_eq!(top8.exact_term_refs, 3);
        assert_eq!(top8.top[0].producer, None);
        assert_eq!(top8.top[0].producer_operator, "multiple");
        assert_eq!(top8.top[0].forced_input_count, 2);
        assert_eq!(top8.top[0].forced_terms_sum, 4);
        assert_eq!(top8.top[0].forced_terms_max, 2);
        assert_eq!(top8.top[0].retained_term_count, 3);
        normalizer.watchdog.as_mut().unwrap().finish(false);
        normalizer.watchdog = None;
    }

    #[test]
    fn materialized_leaf_origin_does_not_outlive_its_nf_allocation() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let source = source_with(&mut expressions, matrix_type(), 71_010);
        let (facts, mut monomials, _) = setup(&mut expressions, &mut programs, source);
        let mut normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials).unwrap();
        normalizer.watchdog = DiagnosticWatchdog::start(21, Duration::from_secs(60));
        let normal_form = Arc::new(PolynomialNF::zero());
        let key = Arc::as_ptr(&normal_form) as usize;
        let state = normalizer.materialized_exact_state(source, Arc::clone(&normal_form)).unwrap();
        let weak = normalizer.diagnostic_materialized_leaf_origins[&key].normal_form.clone();
        assert!(normalizer.diagnostic_materialized_leaf_origin(&normal_form).is_some());
        drop(state);
        drop(normal_form);
        assert!(weak.upgrade().is_none());
        assert!(
            normalizer.diagnostic_materialized_leaf_origins[&key].normal_form.upgrade().is_none()
        );
        normalizer.watchdog.as_mut().unwrap().finish(false);
        normalizer.watchdog = None;
    }

    #[test]
    fn nested_specialized_additive_plan_restores_outer_state_on_success_and_error() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let x = source_with(&mut expressions, matrix_type(), 10_917);
        let y = source_with(&mut expressions, matrix_type(), 10_918);
        let nested = expressions.intern_matrix_transform(MatrixOperation::Add, &[x, y]).unwrap();
        let b = source_with(&mut expressions, matrix_type(), 10_919);
        let k = source_with(&mut expressions, matrix_type(), 10_920);
        let first_rhs = source_with(&mut expressions, matrix_type(), 10_921);
        let second_rhs = source_with(&mut expressions, matrix_type(), 10_922);
        let ambiguous_root = product(&mut expressions, &[b, k]);
        let root = expressions
            .intern_matrix_transform(MatrixOperation::Add, &[nested, ambiguous_root])
            .unwrap();
        let (facts, mut monomials, _) = setup(&mut expressions, &mut programs, root);
        let mut relations = RelationRegistry::new();
        let mut normalization_cache = NormalizationCache::new();
        for rhs in [first_rhs, second_rhs] {
            register_test_closed_relation_into(
                &mut expressions,
                &programs,
                &facts,
                &mut monomials,
                &mut relations,
                &mut normalization_cache,
                ambiguous_root,
                rhs,
                k,
                b,
            );
        }
        relations.freeze();
        let normalization_fingerprint = normalization_cache.canonical_state_fingerprint();
        let nested_semantic = programs.scoped(&expressions, monomials.scope(), nested).unwrap();
        let matrix = match expressions.value_type(ambiguous_root).unwrap() {
            ResolvedValueType::Matrix(matrix) => matrix.clone(),
            _ => panic!("fixture product must be a matrix"),
        };
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_relations(&relations, &mut normalization_cache)
            .with_watchdog_override(true, Duration::from_secs(60));
        let outer_product_consumers = BTreeMap::from([(
            ambiguous_root,
            DiagnosticProductConsumerCounts { add_sub: 1, ..Default::default() },
        )]);
        let outer_product_evaluations = BTreeMap::from([(
            ambiguous_root,
            DiagnosticProductEvaluationSnapshot {
                had_left_exact: true,
                had_right_exact: true,
                scalar_classification: "ordinary",
                left_rows: 2,
                left_columns: 2,
                right_rows: 2,
                right_columns: 2,
                ..DiagnosticProductEvaluationSnapshot::default()
            },
        )]);
        normalizer.diagnostic_product_consumers = outer_product_consumers.clone();
        normalizer.diagnostic_product_evaluations = outer_product_evaluations.clone();
        normalizer.diagnostic_product_root = Some(ambiguous_root);
        let outer_materialization_origins = BTreeMap::from([(
            root,
            DiagnosticMaterializedLeafOrigin {
                producer: Some(nested),
                producer_operator: "add",
                reason: DiagnosticMaterializationReason::NonAddConsumer,
                consumer: Some(root),
                consumer_operator: "add",
                consumer_category: "add_sub",
                remaining_uses: 7,
                scalar_classification: "not_multiply",
                forced_input_count: 1,
                forced_terms_sum: 2,
                forced_terms_max: 2,
                retained_term_count: 2,
            },
        )]);
        normalizer.diagnostic_materialization_origins = outer_materialization_origins.clone();
        let outer_origin_nf = Arc::new(PolynomialNF::zero());
        normalizer.diagnostic_materialized_leaf_origins.insert(
            Arc::as_ptr(&outer_origin_nf) as usize,
            DiagnosticMaterializedLeafAttachment {
                normal_form: Arc::downgrade(&outer_origin_nf),
                origin: outer_materialization_origins[&root],
            },
        );
        let outer_gadget_product_plan = Arc::new(GadgetProductExactPlan {
            id: 91_001,
            authority: normalizer.exact_plan_authority(ambiguous_root).unwrap(),
            expression: ambiguous_root,
            left_expression: b,
            right_expression: k,
            left_type: matrix.clone(),
            right_type: matrix,
            left: Arc::new(PolynomialNF::zero()),
            right: Arc::new(PolynomialNF::zero()),
        });
        normalizer
            .gadget_product_plans
            .insert(ambiguous_root, Arc::clone(&outer_gadget_product_plan));
        let left = normalizer.materialized_exact_state(x, Arc::new(PolynomialNF::zero())).unwrap();
        let right = normalizer.materialized_exact_state(y, Arc::new(PolynomialNF::zero())).unwrap();
        let nested_plan = normalizer.new_additive_plan(nested, left, right, false).unwrap();
        let nested_value = normalizer.normalize_specialized_root(nested).unwrap();
        assert!(nested_value.exact_nf.is_some());
        assert!(Arc::ptr_eq(
            normalizer.gadget_product_plans.get(&ambiguous_root).unwrap(),
            &outer_gadget_product_plan
        ));
        assert_eq!(normalizer.diagnostic_product_consumers, outer_product_consumers);
        assert_eq!(normalizer.diagnostic_product_evaluations, outer_product_evaluations);
        assert_eq!(normalizer.diagnostic_product_root, Some(ambiguous_root));
        let mut restored_materialization_origins = outer_materialization_origins.clone();
        let specialized_origin =
            normalizer.diagnostic_materialization_origins.get(&nested).copied().unwrap();
        assert_eq!(specialized_origin.producer, Some(nested));
        assert_eq!(specialized_origin.reason, DiagnosticMaterializationReason::SpecializedReturn);
        restored_materialization_origins.insert(nested, specialized_origin);
        assert_eq!(normalizer.diagnostic_materialization_origins, restored_materialization_origins);
        assert_eq!(
            normalizer.diagnostic_materialized_leaf_origin(&outer_origin_nf),
            Some(outer_materialization_origins[&root])
        );

        normalizer.clear_value_cache();
        normalizer.exact_plans.clear();
        normalizer.gadget_product_plans.clear();
        normalizer.expression_bounds.clear();
        normalizer.remaining_uses.clear();
        normalizer.clear_gadget_holds();
        normalizer.exact_plans.insert(nested, Arc::clone(&nested_plan));
        normalizer
            .gadget_product_plans
            .insert(ambiguous_root, Arc::clone(&outer_gadget_product_plan));
        let outer_cache_value = Arc::new(AnalyzedValue {
            semantic: nested_semantic,
            exact_nf: None,
            coefficient_bound: NumericContract::Missing,
        });
        normalizer.insert_value_cache(nested, Arc::clone(&outer_cache_value));
        normalizer.expression_bounds.insert(root, NumericContract::Missing);
        normalizer.remaining_uses.insert(root, 7);
        let gadget_nf = Arc::new(PolynomialNF::zero());
        normalizer.insert_gadget_hold(root, Arc::clone(&gadget_nf));
        let saved_bounds = normalizer.expression_bounds.clone();
        let saved_uses = normalizer.remaining_uses.clone();

        let error = normalizer.normalize_specialized_root(ambiguous_root).unwrap_err();
        assert!(matches!(error, NormalizeError::Relation(RelationRegistryError::Ambiguous { .. })));
        assert!(Arc::ptr_eq(normalizer.exact_plans.get(&nested).unwrap(), &nested_plan));
        assert!(Arc::ptr_eq(
            normalizer.gadget_product_plans.get(&ambiguous_root).unwrap(),
            &outer_gadget_product_plan
        ));
        assert!(Arc::ptr_eq(normalizer.cache.get(&nested).unwrap(), &outer_cache_value));
        assert_eq!(normalizer.expression_bounds, saved_bounds);
        assert_eq!(normalizer.remaining_uses, saved_uses);
        assert!(Arc::ptr_eq(normalizer.gadget_input_nfs.get(&root).unwrap(), &gadget_nf));
        assert_eq!(normalizer.exact_plans.len(), 1);
        assert_eq!(normalizer.gadget_product_plans.len(), 1);
        assert_eq!(normalizer.cache.len(), 1);
        assert_eq!(normalizer.gadget_input_nfs.len(), 1);
        assert!(normalizer.suspended_owner_roots.is_empty());
        assert_eq!(normalizer.diagnostic_product_consumers, outer_product_consumers);
        assert_eq!(normalizer.diagnostic_product_evaluations, outer_product_evaluations);
        assert_eq!(normalizer.diagnostic_product_root, Some(ambiguous_root));
        assert_eq!(normalizer.diagnostic_materialization_origins, restored_materialization_origins);
        assert_eq!(
            normalizer.diagnostic_materialized_leaf_origin(&outer_origin_nf),
            Some(outer_materialization_origins[&root])
        );
        drop(normalizer);
        assert_eq!(normalization_cache.canonical_state_fingerprint(), normalization_fingerprint);
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
        assert!(!normalizer.trace.active);
        assert_eq!(normalizer.trace.lines_emitted, 0);
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
    fn additive_plan_materializes_before_closed_relation_boundary_without_cache_drift() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let b = source_with(&mut expressions, matrix_type(), 10_914);
        let k = source_with(&mut expressions, matrix_type(), 10_915);
        let p = source_with(&mut expressions, matrix_type(), 10_916);
        let bk = product(&mut expressions, &[b, k]);
        let root = expressions.intern_matrix_transform(MatrixOperation::Add, &[bk, bk]).unwrap();
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
        let p_proof = expressions.scope_proof(monomials.scope(), p).unwrap();
        let p_scoped = expressions.scoped_from_proof(&p_proof, p).unwrap();
        let p_id = {
            let value = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                .unwrap()
                .normalize(p_scoped)
                .unwrap();
            *value.exact_nf.unwrap().exact_terms.keys().next().unwrap()
        };
        let fingerprint = cache.canonical_state_fingerprint();
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_relations(&relations, &mut cache);
        normalizer.fold_final_no_match = false;
        let value = normalizer.normalize(semantic).unwrap();
        assert_eq!(
            value.exact_nf.unwrap().exact_terms,
            BTreeMap::from([(p_id, BigInt::from(2_u8))])
        );
        drop(normalizer);
        assert_eq!(cache.canonical_state_fingerprint(), fingerprint);
    }

    #[test]
    fn closed_relation_watchdog_distinguishes_whole_and_window_matches() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let b = source_with(&mut expressions, matrix_type(), 732);
        let k = source_with(&mut expressions, matrix_type(), 733);
        let p = source_with(&mut expressions, matrix_type(), 734);
        let prefix = source_with(&mut expressions, matrix_type(), 735);
        let suffix = source_with(&mut expressions, matrix_type(), 736);
        let bk = product(&mut expressions, &[b, k]);
        let whole_root = bk;
        let (whole_facts, mut whole_monomials, whole_semantic) =
            setup(&mut expressions, &mut programs, whole_root);
        let (whole_relations, mut whole_cache, _) = register_test_closed_relation(
            &mut expressions,
            &programs,
            &whole_facts,
            &mut whole_monomials,
            bk,
            p,
            k,
            b,
        );
        let whole_off = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &whole_facts, &mut whole_monomials)
                    .unwrap()
                    .with_relations(&whole_relations, &mut whole_cache)
                    .with_watchdog_override(false, Duration::from_secs(60));
            normalizer.fold_final_no_match = false;
            normalizer.normalize(whole_semantic).unwrap()
        };
        let whole_fingerprint = whole_cache.canonical_state_fingerprint();
        let whole_on = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &whole_facts, &mut whole_monomials)
                    .unwrap()
                    .with_relations(&whole_relations, &mut whole_cache)
                    .with_watchdog_override(true, Duration::from_secs(60));
            normalizer.fold_final_no_match = false;
            let value = normalizer.normalize(whole_semantic).unwrap();
            let counters =
                normalizer.last_watchdog_snapshots.last().copied().unwrap().relation_closure;
            assert_eq!(counters.closures_started, 1);
            assert_eq!(counters.closures_completed, 1);
            assert_eq!(counters.active_depth, 0);
            assert!(counters.closed_relations_present);
            assert_eq!(counters.whole_closed_matches, 1);
            assert_eq!(counters.closed_window_matches, 0);
            assert_eq!(counters.closed_subword_matches, 0);
            assert_eq!(counters.rhs_splices, 1);
            assert_eq!(counters.rhs_terms_total, 1);
            assert_eq!(counters.rhs_terms_max, 1);
            assert_eq!(counters.rhs_terms_enqueued, 1);
            value
        };
        assert_eq!(whole_on.semantic, whole_off.semantic);
        assert_eq!(whole_on.exact_nf, whole_off.exact_nf);
        assert_eq!(whole_on.coefficient_bound, whole_off.coefficient_bound);
        assert_eq!(whole_cache.canonical_state_fingerprint(), whole_fingerprint);

        let window_root = product(&mut expressions, &[prefix, b, k, suffix]);
        let (window_facts, mut window_monomials, window_semantic) =
            setup(&mut expressions, &mut programs, window_root);
        let (window_relations, mut window_cache, _) = register_test_closed_relation(
            &mut expressions,
            &programs,
            &window_facts,
            &mut window_monomials,
            bk,
            p,
            k,
            b,
        );
        let window_off = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &window_facts, &mut window_monomials)
                    .unwrap()
                    .with_relations(&window_relations, &mut window_cache)
                    .with_watchdog_override(false, Duration::from_secs(60));
            normalizer.fold_final_no_match = false;
            normalizer.normalize(window_semantic).unwrap()
        };
        let window_fingerprint = window_cache.canonical_state_fingerprint();
        let window_on = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &window_facts, &mut window_monomials)
                    .unwrap()
                    .with_relations(&window_relations, &mut window_cache)
                    .with_watchdog_override(true, Duration::from_secs(60));
            normalizer.fold_final_no_match = false;
            let value = normalizer.normalize(window_semantic).unwrap();
            let counters =
                normalizer.last_watchdog_snapshots.last().copied().unwrap().relation_closure;
            assert_eq!(counters.closures_started, 1);
            assert_eq!(counters.closures_completed, 1);
            assert_eq!(counters.active_depth, 0);
            assert_eq!(counters.whole_closed_matches, 0);
            assert_eq!(counters.closed_window_matches, 1);
            assert_eq!(counters.closed_subword_matches, 1);
            assert_eq!(counters.rhs_splices, 1);
            assert_eq!(counters.rhs_terms_total, 1);
            assert_eq!(counters.rhs_terms_max, 1);
            assert_eq!(counters.rhs_terms_enqueued, 1);
            assert_eq!(counters.prefix_combines, 1);
            assert_eq!(counters.suffix_combines, 1);
            assert_eq!(counters.monomial_combines, 2);
            value
        };
        assert_eq!(window_on.semantic, window_off.semantic);
        assert_eq!(window_on.exact_nf, window_off.exact_nf);
        assert_eq!(window_on.coefficient_bound, window_off.coefficient_bound);
        assert_eq!(window_cache.canonical_state_fingerprint(), window_fingerprint);
    }

    #[test]
    fn closed_relation_watchdog_attributes_ambiguity_without_changing_error() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let b = source_with(&mut expressions, matrix_type(), 737);
        let k = source_with(&mut expressions, matrix_type(), 738);
        let first_rhs = source_with(&mut expressions, matrix_type(), 739);
        let second_rhs = source_with(&mut expressions, matrix_type(), 740);
        let bk = product(&mut expressions, &[b, k]);
        // Force the ambiguous closed lookup to occur only after the persistent Add plan reaches
        // the root relation boundary.
        let root = expressions.intern_matrix_transform(MatrixOperation::Add, &[bk, bk]).unwrap();
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
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
            first_rhs,
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
            bk,
            second_rhs,
            k,
            b,
        );
        relations.freeze();
        let off = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                    .unwrap()
                    .with_relations(&relations, &mut cache)
                    .with_watchdog_override(false, Duration::from_secs(60));
            normalizer.normalize(semantic).unwrap_err()
        };
        let fingerprint = cache.canonical_state_fingerprint();
        let (on, terminal) = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                    .unwrap()
                    .with_relations(&relations, &mut cache)
                    .with_watchdog_override(true, Duration::from_secs(60));
            let error = normalizer.normalize(semantic).unwrap_err();
            let terminal = normalizer.last_watchdog_snapshots.last().copied().unwrap();
            (error, terminal)
        };
        assert_eq!(on, off);
        assert!(matches!(on, NormalizeError::Relation(RelationRegistryError::Ambiguous { .. })));
        assert_eq!(terminal.relation_closure.closures_started, 1);
        assert_eq!(terminal.relation_closure.closures_completed, 0);
        assert_eq!(terminal.relation_closure.closures_errored, 1);
        assert_eq!(terminal.relation_closure.active_depth, 0);
        assert_eq!(terminal.relation_closure.whole_closed_ambiguities, 1);
        assert_eq!(terminal.relation_closure.closed_window_ambiguities, 0);
        assert_eq!(terminal.relation_closure.universal_ambiguities, 0);
        assert_eq!(terminal.relation_closure.match_errors, 1);
        assert_eq!(terminal.timings.closed_search.calls, 1);
        assert!(terminal.timings.closed_search.total_ns >= terminal.timings.closed_search.max_ns);
        assert_eq!(cache.canonical_state_fingerprint(), fingerprint);
    }

    #[test]
    fn closed_window_watchdog_attributes_ambiguity_without_changing_error() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let prefix = source_with(&mut expressions, matrix_type(), 741);
        let b = source_with(&mut expressions, matrix_type(), 742);
        let k = source_with(&mut expressions, matrix_type(), 743);
        let suffix = source_with(&mut expressions, matrix_type(), 744);
        let first_rhs = source_with(&mut expressions, matrix_type(), 745);
        let second_rhs = source_with(&mut expressions, matrix_type(), 746);
        let bk = product(&mut expressions, &[b, k]);
        let root = product(&mut expressions, &[prefix, b, k, suffix]);
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, root);
        let mut relations = RelationRegistry::new();
        let mut cache = NormalizationCache::new();
        for rhs in [first_rhs, second_rhs] {
            register_test_closed_relation_into(
                &mut expressions,
                &programs,
                &facts,
                &mut monomials,
                &mut relations,
                &mut cache,
                bk,
                rhs,
                k,
                b,
            );
        }
        relations.freeze();
        let off = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                    .unwrap()
                    .with_relations(&relations, &mut cache)
                    .with_watchdog_override(false, Duration::from_secs(60));
            normalizer.normalize(semantic).unwrap_err()
        };
        let fingerprint = cache.canonical_state_fingerprint();
        let (on, terminal) = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                    .unwrap()
                    .with_relations(&relations, &mut cache)
                    .with_watchdog_override(true, Duration::from_secs(60));
            let error = normalizer.normalize(semantic).unwrap_err();
            let terminal = normalizer.last_watchdog_snapshots.last().copied().unwrap();
            (error, terminal)
        };
        assert_eq!(on, off);
        assert!(matches!(on, NormalizeError::Relation(RelationRegistryError::Ambiguous { .. })));
        assert_eq!(terminal.relation_closure.active_depth, 0);
        assert_eq!(terminal.relation_closure.whole_closed_ambiguities, 0);
        assert_eq!(terminal.relation_closure.closed_window_ambiguities, 1);
        assert_eq!(terminal.relation_closure.universal_ambiguities, 0);
        assert_eq!(terminal.relation_closure.match_errors, 1);
        assert_eq!(cache.canonical_state_fingerprint(), fingerprint);
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
        let alternate_trapdoor_root = expressions
            .intern(
                ValueOperator::Trapdoor(super::super::arena::TrapdoorOperation::Generate {
                    descriptor: "fixture-alternate-trapdoor".into(),
                    parameters: Box::new([]),
                    paired_public_event: SampleEventId(52),
                    paired_public_output_role: "value".to_owned(),
                }),
                Box::new([]),
            )
            .unwrap();
        let alternate_trapdoor_family = programs
            .generated_family_from_body(&mut expressions, domain, alternate_trapdoor_root)
            .unwrap();
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
        let alternate_trapdoor = TrapdoorSourceContract { expression: alternate_trapdoor_root };
        let ambiguous_registration = UniversalRelationRegistration {
            dispatch: UniversalDispatchKey {
                preimage_family: preimage,
                preimage_source: registration.dispatch.preimage_source.clone(),
                matrix_type: matrix_type(),
                trapdoor_source: alternate_trapdoor.clone(),
            },
            lhs: StaticLhsKey {
                domain,
                public_plan: public.program(),
                preimage_plan: preimage.program(),
                trapdoor_plan: alternate_trapdoor_family.program(),
                public_pairing: public.program(),
                layout: None,
                factor_order: FactorOrderContract::ordered_public_preimage(),
                remaining_contracts: Box::new([]),
                validation: RelationValidationAuthority {
                    source: registration.dispatch.preimage_source.clone(),
                    trapdoor_source: alternate_trapdoor,
                    matrix_type: matrix_type(),
                    public_type: ty.clone(),
                    preimage_type: ty.clone(),
                    target_type: ty,
                    trapdoor_type: ResolvedValueType::Trapdoor,
                    layout: None,
                    factor_order: FactorOrderContract::ordered_public_preimage(),
                    domain,
                    index_range: range,
                    gadget: None,
                    decomposition: None,
                },
            },
            target_plan: target.program(),
        };
        let mut relations = RelationRegistry::new();
        relations.register_universal(registration.clone()).unwrap();
        let generation = relations.freeze();
        let mut cache = NormalizationCache::new();
        // Construct the unmatched root before borrowing the expression arena through the
        // normalizer; it is used below for the final relation-remaining regression.
        let unmatched = expressions
            .intern_matrix_transform(MatrixOperation::Add, &[k, ordinary_residual])
            .unwrap();
        let unmatched_proof = expressions.scope_proof(scope, unmatched).unwrap();
        let unmatched = expressions.scoped_from_proof(&unmatched_proof, unmatched).unwrap();
        let alternate_index_expression = expressions
            .intern(
                ValueOperator::Constant(super::super::arena::TypedConstant::int(1)),
                Box::new([]),
            )
            .unwrap();
        let alternate_index = expressions
            .scoped_from_proof(
                &expressions.scope_proof(scope, alternate_index_expression).unwrap(),
                alternate_index_expression,
            )
            .unwrap();
        let mut normalizer = Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
            .unwrap()
            .with_relations(&relations, &mut cache)
            .with_trace_product_heartbeat_interval(1)
            .with_trace_focus_call_override(Some(1));
        normalizer.watchdog = DiagnosticWatchdog::start(12, Duration::from_secs(60));
        normalizer.watchdog_specialization = DiagnosticSpecializationCounters::default();
        // Keep direct resolver calls in one synthetic outer watchdog session. Production enters
        // these paths from an already-active root normalization.
        normalizer.normalization_depth = 1;
        let reached = ReachedUniversalLhs::fixture(dispatch.clone(), index, range, None, lhs);
        assert_eq!(normalizer.normalization.as_deref().unwrap().runtime_entry_count(), 0);
        let canonical_count = normalizer.normalization.as_deref().unwrap().canonical_rhs_count();
        let canonical_fingerprint =
            normalizer.normalization.as_deref().unwrap().canonical_state_fingerprint();
        let proof = ProofReachedUniversalLhs::fixture(
            ReachedUniversalLhs::fixture(dispatch.clone(), index, range, None, lhs),
            generation,
        );
        normalizer.expressions.reset_scope_proof_build_count();
        let owned = normalizer.resolve_universal_proof(proof).unwrap();
        assert_eq!(
            normalizer.expressions.scope_proof_build_count(),
            2,
            "one registration builds one LHS proof and one RHS proof"
        );
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
        assert_eq!(normalizer.watchdog_specialization.runtime_lookup_hits, 0);
        assert_eq!(normalizer.watchdog_specialization.runtime_lookup_misses, 0);
        assert_eq!(normalizer.watchdog_specialization.proof_specializations_started, 3);
        assert_eq!(normalizer.watchdog_specialization.proof_specializations_completed, 2);
        assert_eq!(normalizer.watchdog_specialization.proof_rollbacks_completed, 3);
        assert_eq!(normalizer.watchdog_specialization.registrations_started, 2);
        assert_eq!(normalizer.watchdog_specialization.registrations_completed, 2);
        let saved_normalization = normalizer.normalization.take();
        assert_eq!(
            normalizer.resolve_universal(&reached),
            Err(NormalizeError::Relation(RelationRegistryError::InvalidCanonicalRhs))
        );
        normalizer.normalization = saved_normalization;
        assert_eq!(normalizer.watchdog_specialization.registrations_started, 3);
        assert_eq!(normalizer.watchdog_specialization.registrations_completed, 2);
        assert!(matches!(
            normalizer.resolve_universal(&reached).unwrap(),
            RelationResolution::Rewrite(_)
        ));
        assert_eq!(normalizer.normalization.as_deref().unwrap().runtime_entry_count(), 1);
        assert!(matches!(
            normalizer.resolve_universal(&reached).unwrap(),
            RelationResolution::Rewrite(_)
        ));
        let alternate =
            ReachedUniversalLhs::fixture(dispatch.clone(), alternate_index, range, None, lhs);
        assert!(matches!(
            normalizer.resolve_universal(&alternate).unwrap(),
            RelationResolution::NoMatch
        ));
        assert_eq!(normalizer.watchdog_specialization.runtime_lookup_hits, 1);
        assert_eq!(normalizer.watchdog_specialization.runtime_lookup_misses, 3);
        assert_eq!(normalizer.watchdog_specialization.ordinary_specializations_started, 3);
        assert_eq!(normalizer.watchdog_specialization.ordinary_specializations_completed, 2);
        assert_eq!(normalizer.watchdog_specialization.registrations_started, 5);
        assert_eq!(normalizer.watchdog_specialization.registrations_completed, 4);
        assert!(normalizer.watchdog_specialization.rhs_exact_terms_total > 0);
        assert!(normalizer.watchdog_specialization.rhs_exact_terms_max > 0);
        assert_eq!(
            normalizer.watchdog_specialization.interner_existing +
                normalizer.watchdog_specialization.interner_inserted,
            normalizer.watchdog_specialization.registrations_completed
        );
        let expected_watchdog_specialization = normalizer.watchdog_specialization;
        normalizer.watchdog.as_mut().unwrap().finish(false);
        let terminal = normalizer
            .watchdog
            .as_ref()
            .unwrap()
            .shared
            .snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .last()
            .copied()
            .unwrap();
        assert_eq!(terminal.specialization, expected_watchdog_specialization);
        normalizer.watchdog = None;
        normalizer.normalization_depth = 0;
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
        normalizer.normalization.as_deref_mut().unwrap().clear_runtime();
        let rewritten = normalizer.normalize_with_trace(root).unwrap();
        assert!(rewritten.exact_nf.as_ref().is_some_and(|value| value.term_count() > 0));
        let outer_counters = normalizer.counters();
        assert!(normalizer.trace.active);
        assert!(normalizer.trace.lines_emitted >= 3);
        assert!(normalizer.trace.lines_emitted <= NORMALIZATION_TRACE_LINE_BUDGET);
        assert!(normalizer.trace.terminal_emitted);
        assert_eq!(normalizer.trace.terminal_event, "normalize_end");
        assert!(normalizer.trace.normalization_calls >= 3);
        assert!(normalizer.trace.max_normalization_depth >= 2);
        assert_eq!(normalizer.trace.outer_nodes_total, outer_counters.nodes_total);
        assert!(normalizer.trace.nodes_total > normalizer.trace.outer_nodes_total);
        assert!(normalizer.trace.nodes_processed >= outer_counters.nodes_processed);
        assert!(normalizer.trace.product_generated > 0);
        assert!(normalizer.trace.product_processed > 0);
        assert_eq!(
            normalizer.trace.post_history,
            [
                "post:root_take",
                "post:root_unwrap",
                "post:relation_rewrite",
                "post:relation_rebound",
                "post:fold_bound",
                "post:fold_terms",
                "post:relation_remaining",
                "post:complete",
            ]
        );
        assert_eq!(normalizer.trace.post_lines_emitted, NORMALIZATION_TRACE_POST_LINE_BUDGET);
        assert!(normalizer.trace.root_exact_terms > 0);
        assert!(normalizer.trace.root_sum_ordered_factors > 0);
        assert!(normalizer.trace.relation_initial > 0);
        assert!(normalizer.trace.relation_processed >= normalizer.trace.relation_initial);
        assert!(normalizer.trace.relation_peak_worklist >= normalizer.trace.relation_initial);
        assert!(normalizer.trace.relation_rewrites > 0);
        assert!(normalizer.trace.relation_enqueues > 0);
        assert_eq!(normalizer.trace.relation_closed_window_probes, 0);
        assert!(normalizer.trace.relation_universal_factor_probes > 0);
        assert!(
            normalizer.trace.product_heartbeat_saw_matrix_multiply,
            "trace={:?}",
            normalizer.trace
        );
        assert_eq!(normalizer.normalization_depth, 0);

        // Focus the first specialized normalization itself. Its immediate successor claims the
        // one-shot token and traces both lexical scope-proof calls in place.
        normalizer.normalization.as_deref_mut().unwrap().clear_runtime();
        normalizer.trace_focus_call_override = Some(Some(2));
        normalizer.trace_product_heartbeat_interval = u64::MAX;
        normalizer.normalize_with_trace(root).unwrap();
        assert_eq!(
            normalizer.trace.caller_history,
            [
                "specialized_root:normalize_enter",
                "caller:nested_return",
                "caller:cache_restored",
                "caller:bounds_merge_start",
                "caller:bounds_merge_end",
                "caller:uses_restore_start",
                "caller:uses_restore_end",
                "caller:state_restored",
                "next_root:preproof_start",
                "next_root:preproof_end",
                "next_root:normalize_proof_start",
                "next_root:normalize_proof_end",
            ]
        );
        assert!(normalizer.trace.caller_nested_bounds_len > 0);
        assert!(normalizer.trace.caller_nested_uses_len > 0);
        assert!(normalizer.trace.caller_outer_bounds_len > 0);
        assert!(
            normalizer.trace.caller_after_bounds_len >= normalizer.trace.caller_outer_bounds_len
        );
        assert!(normalizer.trace.lines_emitted <= NORMALIZATION_TRACE_LINE_BUDGET);
        assert!(normalizer.trace.terminal_emitted);

        // A dispatchable K which has no adjacent matching public factor is retained, while the
        // ordinary residual in the same sum is not mislabeled as relation-bearing. This is a
        // final structural diagnostic only: no second universal specialization is performed.
        normalizer.fold_final_no_match = false;
        let unmatched_value = normalizer.normalize(unmatched).unwrap();
        assert_eq!(unmatched_value.exact_nf.as_ref().unwrap().exact_terms.len(), 2);
        assert_eq!(normalizer.counters().relation_remaining, 1);

        // Run the same production-shaped universal rewrite under the watchdog. Universal
        // specialization recursively normalizes registration roots, so this exercises nested
        // relation closures rather than a synthetic counter update. The outer terminal snapshot
        // must retain all nested work and report a fully unwound closure stack.
        normalizer.normalization.as_deref_mut().unwrap().clear_runtime();
        normalizer.watchdog_enabled_override = Some(true);
        normalizer.watchdog_interval_override = Some(Duration::from_secs(60));
        let watched = normalizer.normalize(root).unwrap();
        let terminal = normalizer.last_watchdog_snapshots.last().copied().unwrap();
        assert_eq!(watched.semantic, rewritten.semantic);
        assert_eq!(watched.exact_nf, rewritten.exact_nf);
        assert_eq!(watched.coefficient_bound, rewritten.coefficient_bound);
        assert!(terminal.relation_closure.closures_started > 1);
        assert_eq!(
            terminal.relation_closure.closures_completed,
            terminal.relation_closure.closures_started
        );
        assert_eq!(terminal.relation_closure.closures_errored, 0);
        assert_eq!(terminal.relation_closure.active_depth, 0);
        assert!(!terminal.relation_closure.closed_relations_present);
        assert!(terminal.relation_closure.universal_dispatch_hits > 0);
        assert!(terminal.relation_closure.universal_specializations > 0);
        assert!(terminal.relation_closure.universal_matches > 0);
        assert!(terminal.relation_closure.universal_rewrites > 0);
        assert_eq!(terminal.relation_closure, normalizer.watchdog_relation_closure);
        assert!(terminal.timings.universal_specialized_cached.calls > 0);
        assert_eq!(
            terminal.timings.universal_factor_dispatch.calls,
            terminal.relation_closure.universal_probes,
            "mixed call/non-call factors each contribute exactly one dispatch timing"
        );
        assert!(terminal.timings.cached_key_lookup.calls > 0);
        assert!(terminal.timings.cached_miss_specialize.calls > 0);
        assert!(terminal.timings.specialized_nested_normalize.calls > 0);
        assert!(terminal.timings.specialized_merge_bounds.calls > 0);
        for counter in [
            terminal.timings.universal_specialized_cached,
            terminal.timings.cached_key_lookup,
            terminal.timings.cached_miss_specialize,
            terminal.timings.specialized_nested_normalize,
            terminal.timings.specialized_merge_bounds,
        ] {
            assert!(counter.total_ns >= counter.max_ns);
        }
        assert_eq!(terminal.timings, normalizer.watchdog_timings);

        let warm_runtime_entries =
            normalizer.normalization.as_deref().unwrap().runtime_entry_count();
        let warm_fingerprint =
            normalizer.normalization.as_deref().unwrap().canonical_state_fingerprint();
        normalizer.expressions.reset_scope_proof_build_count();
        let warm = normalizer.normalize(root).unwrap();
        assert_eq!(normalizer.expressions.scope_proof_build_count(), 1);
        assert_eq!(warm.semantic, watched.semantic);
        assert_eq!(warm.exact_nf, watched.exact_nf);
        assert_eq!(warm.coefficient_bound, watched.coefficient_bound);
        assert_eq!(
            normalizer.normalization.as_deref().unwrap().runtime_entry_count(),
            warm_runtime_entries
        );
        assert_eq!(
            normalizer.normalization.as_deref().unwrap().canonical_state_fingerprint(),
            warm_fingerprint
        );

        normalizer.normalization.as_deref_mut().unwrap().clear_runtime();
        normalizer.watchdog_timings = DiagnosticTimings::default();
        normalizer.watchdog = DiagnosticWatchdog::start(15, Duration::from_secs(60));
        normalizer.normalization_depth = 1;
        let first_cached =
            normalizer.specialized_universal_cached(&dispatch, index, range).unwrap();
        let second_cached =
            normalizer.specialized_universal_cached(&dispatch, index, range).unwrap();
        assert_eq!(first_cached, second_cached);
        assert_eq!(normalizer.watchdog_timings.cached_key_lookup.calls, 2);
        assert_eq!(normalizer.watchdog_timings.cached_miss_specialize.calls, 1);
        assert_eq!(normalizer.watchdog_timings.cached_insert_return_clone.calls, 1);
        assert_eq!(normalizer.watchdog_timings.cached_hit_clone.calls, 1);
        assert!(normalizer.watchdog_timings.cached_hit_returned_entries_total > 0);
        assert_eq!(
            normalizer.watchdog_timings.cached_hit_returned_entries_total,
            normalizer.watchdog_timings.cached_hit_returned_entries_max
        );
        normalizer.watchdog.as_mut().unwrap().finish(false);
        normalizer.watchdog = None;
        normalizer.normalization_depth = 0;
        drop(normalizer);

        let mut ambiguous_relations = RelationRegistry::new();
        ambiguous_relations.register_universal(registration).unwrap();
        ambiguous_relations.register_universal(ambiguous_registration).unwrap();
        ambiguous_relations.freeze();
        let mut ambiguous_cache = NormalizationCache::new();
        let lhs_factor_expressions = {
            let descriptor = monomials.descriptor(lhs).unwrap();
            descriptor
                .central_factors
                .iter()
                .chain(&descriptor.ordered_factors)
                .map(|factor| factor.expression())
                .collect::<Vec<_>>()
        };
        let mut ambiguous_normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                .unwrap()
                .with_relations(&ambiguous_relations, &mut ambiguous_cache)
                .with_watchdog_override(true, Duration::from_secs(60));
        for expression in lhs_factor_expressions {
            ambiguous_normalizer
                .expression_bounds
                .insert(expression, NumericContract::Known(CoefficientBound::finite(1_u8)));
        }
        let ambiguous_reasons =
            ambiguous_normalizer.diagnostic_frontier_reasons(lhs, false, false, true).unwrap();
        assert_eq!(ambiguous_reasons, FRONTIER_AMBIGUOUS_UNIVERSAL_DISPATCH);
        let cache_entries_before =
            ambiguous_normalizer.normalization.as_deref().unwrap().runtime_entry_count();
        let cache_fingerprint_before =
            ambiguous_normalizer.normalization.as_deref().unwrap().canonical_state_fingerprint();
        let ambiguous_census = ambiguous_normalizer
            .diagnostic_four_class_census(&[DiagnosticExactNf {
                normal_form: Arc::new(PolynomialNF {
                    exact_terms: BTreeMap::from([(lhs, BigInt::from(1_u8))]),
                    bounded_summary: BoundedSummary::missing(),
                }),
                ordinal: 0,
                under_product: false,
            }])
            .unwrap();
        assert_eq!(ambiguous_census.finite_relation_frontier.unique_monomials, 1);
        assert_eq!(ambiguous_census.finite_relation_frontier.term_refs, 1);
        assert_eq!(ambiguous_census.current_exact_preimage, DiagnosticClassStats::default());
        assert_eq!(ambiguous_census.ambiguous_universal_dispatch.unique_monomials, 1);
        assert_eq!(ambiguous_census.ambiguous_universal_dispatch.term_refs, 1);
        assert_eq!(ambiguous_census.top_len, 1);
        assert_eq!(ambiguous_census.top[0].finite_relation_frontier_refs, 1);
        assert_eq!(
            ambiguous_normalizer.normalization.as_deref().unwrap().runtime_entry_count(),
            cache_entries_before
        );
        assert_eq!(
            ambiguous_normalizer.normalization.as_deref().unwrap().canonical_state_fingerprint(),
            cache_fingerprint_before
        );
        let error = ambiguous_normalizer.normalize(root).unwrap_err();
        assert_eq!(
            error,
            NormalizeError::Relation(RelationRegistryError::AmbiguousPreimageDispatch)
        );
        let terminal = ambiguous_normalizer.last_watchdog_snapshots.last().copied().unwrap();
        assert_eq!(terminal.relation_closure.closures_started, 1);
        assert_eq!(terminal.relation_closure.closures_completed, 0);
        assert_eq!(terminal.relation_closure.closures_errored, 1);
        assert_eq!(terminal.relation_closure.active_depth, 0);
        assert_eq!(terminal.relation_closure.universal_ambiguities, 1);
        assert_eq!(terminal.relation_closure.match_errors, 1);
        assert_eq!(
            terminal.timings.universal_factor_dispatch.calls,
            terminal.relation_closure.universal_probes
        );
        assert_eq!(terminal.timings.universal_search_total.calls, 1);
        assert_eq!(terminal.timings.outer_relation_rebound.calls, 1);
        for counter in [
            terminal.timings.universal_factor_dispatch,
            terminal.timings.universal_search_total,
            terminal.timings.outer_relation_rebound,
        ] {
            assert!(counter.total_ns >= counter.max_ns);
        }
        assert_eq!(terminal.timings, ambiguous_normalizer.watchdog_timings);
        assert!(ambiguous_normalizer.last_watchdog_events.len() <= 32);
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
