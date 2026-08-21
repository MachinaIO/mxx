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
    monomial::{MonomialArena, MonomialError, MonomialId, MonomialSweepOwnerReport, TermMap},
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
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    fmt,
    sync::{
        Arc, Mutex,
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
    last_gadget: MonomialSweepOwnerReport,
    last_canonical_runtime: MonomialSweepOwnerReport,
    last_closed: MonomialSweepOwnerReport,
    last_suspended: MonomialSweepOwnerReport,
    value_cache_entries: u64,
    value_cache_exact_term_refs: u64,
    value_cache_top8_exact_term_refs: u64,
    value_cache_top8_len: u8,
    value_cache_top8: [DiagnosticValueCacheTopEntry; 8],
    sweep_total_ns: u64,
    sweep_max_ns: u64,
    sweep_last_ns: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DiagnosticValueCacheTopEntry {
    expression_slot: u64,
    operator_category: &'static str,
    term_count: u64,
    remaining_uses: u64,
    producer_input_count: u64,
    cached_input_exact_term_refs_sum: u64,
    cached_input_exact_term_refs_max: u64,
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
    watchdog_generation: u64,
    watchdog_product_processed: u64,
    watchdog_product_generated: u64,
    watchdog_product_enqueued: u64,
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

    fn diagnostic_value_cache_top8(&self) -> DiagnosticValueCacheTop8 {
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
        }
        result
    }

    fn clear_value_cache(&mut self) {
        self.cache.clear();
        self.owner_counters.cache_exact_terms = 0;
    }

    fn insert_value_cache(&mut self, expression: ExprId, value: Arc<AnalyzedValue>) {
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
                .chain(gadget_roots)
                .chain(canonical_roots)
                .chain(closed_roots)
                .chain(suspended_roots)
                .chain(active),
        );
        let canonical =
            self.normalization.as_deref().map(|cache| cache.owner_census()).unwrap_or_default();
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
            expression_bounds: BTreeMap::new(),
            remaining_uses: BTreeMap::new(),
            gadget_input_nfs: BTreeMap::new(),
            owner_counters: NormalizerOwnerCounters::default(),
            counters: NormalizationCounters::default(),
            trace: NormalizationTrace::new(),
            watchdog: None,
            watchdog_generation: 0,
            watchdog_product_processed: 0,
            watchdog_product_generated: 0,
            watchdog_product_enqueued: 0,
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
        let processed = self.watchdog_product_processed;
        if processed >= 1_024 && processed.is_power_of_two() {
            let _ = active;
            self.watchdog_update(|progress| {
                progress.phase = DiagnosticPhase::ProductDrain;
                progress.product_processed = processed;
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
        self.expression_bounds.clear();
        self.remaining_uses.clear();
        self.clear_gadget_holds();
        self.owner_counters = NormalizerOwnerCounters::default();
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
        }
        let saved_cache = std::mem::take(&mut self.cache);
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
        self.merge_expression_bounds(nested_expression_bounds);
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
            self.remove_value_cache(expression);
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
        self.insert_gadget_hold(expression, normal_form.clone());
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

        let mut reclassify = |normal_form: &PolynomialNF| -> Result<PolynomialNF, NormalizeError> {
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
            Ok(PolynomialNF {
                exact_terms: reclassified_terms,
                bounded_summary: BoundedSummary::missing(),
            })
        };
        if let Some(ordered_product) = ordered_scalar_product {
            return Ok(ScalarActionNormalization::Exact(reclassify(&ordered_product)?));
        }

        let reclassified_left = if left_scalar { reclassify(left)? } else { left.clone() };
        let reclassified_right = if right_scalar { reclassify(right)? } else { right.clone() };
        Ok(ScalarActionNormalization::Exact(self.product_nf(
            scope_proof,
            &reclassified_left,
            &reclassified_right,
        )?))
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
        scope_proof: &ScopeProof,
        left: &PolynomialNF,
        right: &PolynomialNF,
    ) -> Result<PolynomialNF, NormalizeError> {
        if self.watchdog.is_some() {
            self.watchdog_product_call_id = self.watchdog_product_call_id.saturating_add(1);
            self.watchdog_product_generation_current = 0;
            self.watchdog_product_enqueued_current = 0;
            self.watchdog_product_queue_current = 0;
            self.watchdog_product_output_current = 0;
        }
        let planned = u64::try_from(left.exact_terms.len())
            .unwrap_or(u64::MAX)
            .saturating_mul(u64::try_from(right.exact_terms.len()).unwrap_or(u64::MAX));
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
        let result = self.product_nf_body(scope_proof, left, right);
        if self.trace.active {
            if let Ok(output) = &result {
                self.trace.product_max_output_terms = self
                    .trace
                    .product_max_output_terms
                    .max(u64::try_from(output.exact_terms.len()).unwrap_or(u64::MAX));
            }
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

    fn product_nf_body(
        &mut self,
        scope_proof: &ScopeProof,
        left: &PolynomialNF,
        right: &PolynomialNF,
    ) -> Result<PolynomialNF, NormalizeError> {
        let mut worklist = VecDeque::new();
        for (left_id, left_coefficient) in &left.exact_terms {
            for (right_id, right_coefficient) in &right.exact_terms {
                let product = self.product_monomials(scope_proof, *left_id, *right_id)?;
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
                let coefficient = left_coefficient * right_coefficient;
                if coefficient.is_zero() {
                    self.watchdog_record_product_generated(
                        false,
                        u64::try_from(worklist.len()).unwrap_or(u64::MAX),
                        std::iter::once(product)
                            .chain(left.exact_terms.keys().copied())
                            .chain(right.exact_terms.keys().copied())
                            .chain(worklist.iter().map(|(id, _)| *id)),
                    );
                    continue;
                }
                worklist.push_back((product, coefficient));
                self.watchdog_record_product_generated(
                    true,
                    u64::try_from(worklist.len()).unwrap_or(u64::MAX),
                    left.exact_terms
                        .keys()
                        .chain(right.exact_terms.keys())
                        .chain(worklist.iter().map(|(id, _)| id))
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
            }
        }
        if self.watchdog.is_some() {
            let generated = self.watchdog_product_generated;
            let enqueued = self.watchdog_product_enqueued;
            let generation_current = self.watchdog_product_generation_current;
            let enqueued_current = self.watchdog_product_enqueued_current;
            let queue_current = u64::try_from(worklist.len()).unwrap_or(u64::MAX);
            self.watchdog_update(|progress| {
                progress.phase = DiagnosticPhase::ProductGenerationEnd;
                progress.product_generated = generated;
                progress.product_enqueued = enqueued;
                progress.product_generation_current = generation_current;
                progress.product_enqueued_current = enqueued_current;
                progress.product_queue_current = queue_current;
                progress.product_output_current = 0;
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
                        .copied(),
                );
            }
        }
        let mut terms = BTreeMap::new();
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
                merge_term(&mut terms, monomial, coefficient);
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
        let fingerprint = cache.canonical_state_fingerprint();
        let (on, on_counters, on_gc) = {
            let mut normalizer =
                Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                    .unwrap()
                    .with_relations(&relations, &mut cache)
                    .with_watchdog_override(true, Duration::from_secs(60));
            normalizer.monomial_gc_allocation_threshold_bytes = 1;
            let value = normalizer.normalize(semantic).unwrap();
            (value, normalizer.counters(), normalizer.gc_counters)
        };
        assert!(on_gc.sweep_count > 0);
        assert!(on_gc.value_cache_top8_len > 0);
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
        let (facts, mut monomials, semantic) = setup(&mut expressions, &mut programs, bk);
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
        let mut ambiguous_normalizer =
            Normalizer::new(&mut expressions, &programs, &facts, &mut monomials)
                .unwrap()
                .with_relations(&ambiguous_relations, &mut ambiguous_cache)
                .with_watchdog_override(true, Duration::from_secs(60));
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
