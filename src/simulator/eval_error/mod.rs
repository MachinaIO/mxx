use super::{
    SimulatorContext, error_norm::ErrorNorm, poly_matrix_norm::PolyMatrixNorm, poly_norm::PolyNorm,
};
use crate::{
    circuit::{
        BatchedWire, Evaluable, GroupedCallExecutionLayer, PolyCircuit, PolyGateType,
        SubCircuitParamValue, batched_wire_slice_at, batched_wire_slice_len, gate::GateId,
        iter_batched_wire_gates,
    },
    element::PolyElem,
    lookup::{PltEvaluator, PublicLut, commit_eval::compute_padded_len},
    poly::dcrt::poly::DCRTPoly,
    slot_transfer::SlotTransferEvaluator,
    utils::bigdecimal_bits_ceil,
};
use bigdecimal::BigDecimal;
use dashmap::DashMap;
use num_bigint::BigUint;
use num_traits::{FromPrimitive, One};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
        IntoParallelRefMutIterator, ParallelIterator,
    },
    prelude::ParallelSlice,
};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};
use tracing::{debug, info};

use self::layers::ErrorNormExecutionLayer;

const ERROR_NORM_OUTPUT_BATCH_SIZE: usize = 1024;
const ERROR_NORM_EXPR_PAR_BATCH_SIZE: usize = 32;
const ERROR_NORM_CALL_COMMIT_BATCH_SIZE: usize = 128;
const ERROR_NORM_SUMMED_CALL_COMMIT_BATCH_SIZE: usize = 16;
const ERROR_NORM_SUMMED_INNER_REDUCE_BATCH_SIZE: usize = 32;

/// Global cache for sub-circuit summaries.
///
/// `eval_max_error_norm` materializes affine summaries for a sub-circuit once per
/// `(sub-circuit, input plaintext profile)` pair, then reuses that summary across direct calls,
/// summed calls, and nested callers. The value is immutable after construction, so the cache can
/// safely share `Arc<ErrorNormSubCircuitSummary>` across threads.
type ErrorNormSubCircuitSummaryCache =
    DashMap<ErrorNormSubCircuitSummaryCacheKey, Arc<ErrorNormSubCircuitSummary>>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Hash-friendly identity for one `PolyNorm`.
///
/// The summary cache needs a stable key, but `PolyNorm` itself contains shared context and big
/// decimal state. The cache only cares about the simulator context identity and the textual bound,
/// so this key strips the runtime object down to those two stable components.
struct ErrorNormPolyNormKey {
    ctx_id: usize,
    norm: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Cache key for one prepared sub-circuit summary.
///
/// The same sub-circuit body can be reused under many plaintext profiles. This key intentionally
/// does not include parameter bindings because bindings affect only the summary build process for a
/// given profile and are handled by `ErrorNormSummaryBuildKey` while registration is in progress.
struct ErrorNormSubCircuitSummaryCacheKey {
    circuit_key: usize,
    sub_circuit_id: usize,
    input_plaintext_norms: Vec<ErrorNormPolyNormKey>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Cycle-detection key for the recursive summary builder.
///
/// Two summary builds with the same cache key but different binding allocations must still be
/// treated as distinct visits while the recursion stack is active, otherwise a valid nested call
/// graph could be mistaken for a cycle. `binding_sig` captures that call-local identity.
struct ErrorNormSummaryBuildKey {
    cache_key: ErrorNormSubCircuitSummaryCacheKey,
    binding_sig: usize,
}

#[derive(Debug, Clone)]
/// Fully described summary build job for one sub-circuit instance.
///
/// The registry stores these nodes in topological order before any affine expressions are
/// materialized. Later stages walk `topo_order` and build `ErrorNormSubCircuitSummary` values from
/// the leaf nodes upward.
struct ErrorNormSummaryNode {
    circuit: Arc<PolyCircuit<DCRTPoly>>,
    sub_circuit_id: usize,
    input_plaintext_norms: Arc<[PolyNorm]>,
    param_bindings: Arc<[SubCircuitParamValue]>,
    output_plaintext_norms: Arc<[PolyNorm]>,
    direct_call_keys: HashMap<usize, ErrorNormSubCircuitSummaryCacheKey>,
}

#[derive(Debug, Default)]
/// Temporary registry used while discovering nested sub-circuit summaries.
///
/// `nodes` owns the specification for each summary that still needs to be built, and
/// `topo_order` records a leaf-to-root order so `ensure_error_norm_summary_built` can populate the
/// shared cache without revisiting the dependency graph.
struct ErrorNormSummaryRegistry {
    nodes: HashMap<ErrorNormSubCircuitSummaryCacheKey, ErrorNormSummaryNode>,
    topo_order: Vec<ErrorNormSubCircuitSummaryCacheKey>,
}

/// Convert a runtime `PolyNorm` into the reduced cache-key representation used by summary caches.
fn error_norm_poly_norm_key(norm: &PolyNorm) -> ErrorNormPolyNormKey {
    ErrorNormPolyNormKey { ctx_id: Arc::as_ptr(&norm.ctx) as usize, norm: norm.norm.to_string() }
}

/// Build the cache key for one sub-circuit summary request.
///
/// This intentionally captures only the circuit identity plus the caller-observed plaintext
/// profile. If two call sites expose the same input bounds, they can share the exact same affine
/// summary.
fn error_norm_sub_circuit_summary_cache_key(
    circuit_key: usize,
    sub_circuit_id: usize,
    input_plaintext_norms: &[PolyNorm],
) -> ErrorNormSubCircuitSummaryCacheKey {
    ErrorNormSubCircuitSummaryCacheKey {
        circuit_key,
        sub_circuit_id,
        input_plaintext_norms: input_plaintext_norms.iter().map(error_norm_poly_norm_key).collect(),
    }
}

/// Return the largest plaintext-profile bit width in one summary request.
///
/// The builder uses this as a cheap heuristic when choosing grouping strategies for direct and
/// summed sub-circuit calls: similar profiles are likely to share cache entries and have similar
/// substitution costs.
fn error_norm_summary_profile_max_bits(input_plaintext_norms: &[PolyNorm]) -> u64 {
    input_plaintext_norms.iter().map(|norm| bigdecimal_bits_ceil(&norm.norm)).max().unwrap_or(0)
}

/// Fixed Gaussian tail multiplier used when turning simulator noise parameters into hard bounds.
///
/// This session does not change the bound itself; the helper exists only so all extracted files use
/// the exact same constant and the intent remains visible near the error simulator.
fn gaussian_tail_bound_factor() -> BigDecimal {
    BigDecimal::from_f32(6.5).unwrap()
}

/// Resolve the `input_idx`-th input of a direct sub-circuit call that uses a shared-prefix input
/// set.
///
/// Direct-call helpers split inputs into a reusable prefix plus a per-call suffix to reduce
/// cloning. This helper reconstructs the logical input ordering expected by the original monolithic
/// code.
fn resolve_shared_prefix_suffix_input_id(
    shared_prefix: &[BatchedWire],
    suffix: &[BatchedWire],
    input_idx: usize,
) -> GateId {
    let shared_prefix_len = batched_wire_slice_len(shared_prefix);
    if input_idx < shared_prefix_len {
        return batched_wire_slice_at(shared_prefix, input_idx);
    }
    let suffix_idx = input_idx
        .checked_sub(shared_prefix_len)
        .unwrap_or_else(|| panic!("error-norm shared-prefix input index underflow: {input_idx}"));
    if suffix_idx < batched_wire_slice_len(suffix) {
        return batched_wire_slice_at(suffix, suffix_idx);
    }
    panic!(
        "error-norm shared-prefix input index {input_idx} out of range (prefix={}, suffix={})",
        shared_prefix_len,
        batched_wire_slice_len(suffix)
    );
}

/// Resolve one logical input position inside the flattened representation of an input-set.
fn resolve_input_set_gate_id(input_ids: &[BatchedWire], input_idx: usize) -> GateId {
    if input_idx < batched_wire_slice_len(input_ids) {
        return batched_wire_slice_at(input_ids, input_idx);
    }
    panic!(
        "error-norm input-set index {input_idx} out of range (len={})",
        batched_wire_slice_len(input_ids)
    )
}

#[derive(Debug)]
/// Contiguous non-zero slice inside `BatchedGateUseCounts`.
///
/// The simulator stores use counts only for live gates. Grouping them into contiguous batches keeps
/// lookups and removals cheap while avoiding one `HashMap` entry per gate.
struct BatchedGateUseCountBatch {
    range: BatchedWire,
    counts: Box<[u32]>,
}

#[derive(Debug)]
/// Sparse reference-count table for live gate outputs.
///
/// Both the concrete `ErrorNorm` evaluator and the affine summary builder use this to decide when a
/// wire can be dropped from temporary storage. `live_nonzero` tracks how many gates still have a
/// positive consumer count so the caller can log memory pressure cheaply.
struct BatchedGateUseCounts {
    batches: BTreeMap<GateId, BatchedGateUseCountBatch>,
    live_nonzero: usize,
}

impl BatchedGateUseCounts {
    fn from_dense_counts(counts: Vec<u32>) -> Self {
        let mut batches = BTreeMap::new();
        let mut live_nonzero = 0usize;
        let mut batch_start: Option<GateId> = None;
        let mut batch_counts = Vec::new();
        for (gate_idx, count) in counts.into_iter().enumerate() {
            if count == 0 {
                if let Some(start) = batch_start.take() {
                    let range = BatchedWire::from_start_len(start, batch_counts.len());
                    batches.insert(
                        start,
                        BatchedGateUseCountBatch {
                            range,
                            counts: std::mem::take(&mut batch_counts).into_boxed_slice(),
                        },
                    );
                }
                continue;
            }
            live_nonzero += 1;
            if batch_start.is_none() {
                batch_start = Some(GateId(gate_idx));
            }
            batch_counts.push(count);
        }
        if let Some(start) = batch_start {
            let range = BatchedWire::from_start_len(start, batch_counts.len());
            batches.insert(
                start,
                BatchedGateUseCountBatch { range, counts: batch_counts.into_boxed_slice() },
            );
        }
        Self { batches, live_nonzero }
    }

    fn batch_containing(&self, gate_id: GateId) -> Option<&BatchedGateUseCountBatch> {
        self.batches
            .range(..=gate_id)
            .next_back()
            .and_then(|(_, batch)| if gate_id.0 < batch.range.end().0 { Some(batch) } else { None })
    }

    fn batch_containing_mut(&mut self, gate_id: GateId) -> Option<&mut BatchedGateUseCountBatch> {
        self.batches
            .range_mut(..=gate_id)
            .next_back()
            .and_then(|(_, batch)| if gate_id.0 < batch.range.end().0 { Some(batch) } else { None })
    }

    #[inline]
    fn contains(&self, gate_id: GateId) -> bool {
        self.batch_containing(gate_id)
            .map(|batch| batch.counts[gate_id.0 - batch.range.start().0] != 0)
            .unwrap_or(false)
    }

    #[inline]
    fn len(&self) -> usize {
        self.live_nonzero
    }

    fn decrement(&mut self, gate_id: GateId) -> bool {
        let batch = self
            .batch_containing_mut(gate_id)
            .unwrap_or_else(|| panic!("error-norm use count missing storage for gate {gate_id}"));
        let count = &mut batch.counts[gate_id.0 - batch.range.start().0];
        let current = *count;
        if current == 0 {
            return false;
        }
        *count = current - 1;
        if current == 1 {
            self.live_nonzero -= 1;
            return true;
        }
        false
    }

    fn decrement_by(&mut self, gate_id: GateId, amount: u32) -> bool {
        if amount == 0 {
            return false;
        }
        let batch = self
            .batch_containing_mut(gate_id)
            .unwrap_or_else(|| panic!("error-norm use count missing storage for gate {gate_id}"));
        let count = &mut batch.counts[gate_id.0 - batch.range.start().0];
        let current = *count;
        if current == 0 {
            return false;
        }
        assert!(
            amount <= current,
            "error-norm use count underflow for gate {gate_id}: current={current}, amount={amount}"
        );
        *count = current - amount;
        if current == amount {
            self.live_nonzero -= 1;
            return true;
        }
        false
    }
}

type ErrorNormInputGatePositions = HashMap<GateId, usize>;

#[derive(Debug)]
/// One contiguous storage block inside `BatchedValueStore`.
///
/// Entries are indexed by gate id range. Individual slots become `None` once their last consumer is
/// released, but the outer batch stays in place so the store can keep O(log n) lookup behavior.
struct BatchedValueStoreEntry<T> {
    range: BatchedWire,
    values: Box<[Option<Arc<T>>]>,
}

#[derive(Debug)]
/// Sparse store for gate-indexed values that are produced and released in topological order.
///
/// This is the extracted replacement for the ad-hoc vector bookkeeping in the old monolithic file.
/// It stores either concrete `ErrorNorm` values or affine summary expressions, always batched by
/// contiguous gate ranges so insertion during one execution layer stays cheap.
struct BatchedValueStore<T> {
    entries: BTreeMap<GateId, BatchedValueStoreEntry<T>>,
    live_entries: usize,
}

impl<T> Default for BatchedValueStore<T> {
    fn default() -> Self {
        Self { entries: BTreeMap::new(), live_entries: 0 }
    }
}

impl<T> BatchedValueStore<T> {
    fn entry_containing(&self, gate_id: GateId) -> Option<&BatchedValueStoreEntry<T>> {
        self.entries
            .range(..=gate_id)
            .next_back()
            .and_then(|(_, entry)| if gate_id.0 < entry.range.end().0 { Some(entry) } else { None })
    }

    fn entry_containing_mut(&mut self, gate_id: GateId) -> Option<&mut BatchedValueStoreEntry<T>> {
        self.entries
            .range_mut(..=gate_id)
            .next_back()
            .and_then(|(_, entry)| if gate_id.0 < entry.range.end().0 { Some(entry) } else { None })
    }

    fn get(&self, gate_id: GateId) -> Option<Arc<T>> {
        self.entry_containing(gate_id).and_then(|entry| {
            entry.values[gate_id.0 - entry.range.start().0].as_ref().map(Arc::clone)
        })
    }

    fn insert_single(&mut self, gate_id: GateId, value: T) {
        self.insert_batch(BatchedWire::single(gate_id), vec![value]);
    }

    fn insert_batch(&mut self, range: BatchedWire, values: Vec<T>) {
        assert_eq!(
            range.len(),
            values.len(),
            "batched value store range len {} must match value count {}",
            range.len(),
            values.len()
        );
        let values = values
            .into_iter()
            .map(|value| Some(Arc::new(value)))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        self.live_entries += values.len();
        self.entries.insert(range.start(), BatchedValueStoreEntry { range, values });
    }

    fn insert_batch_shared(&mut self, range: BatchedWire, values: Vec<Arc<T>>) {
        assert_eq!(
            range.len(),
            values.len(),
            "batched value store range len {} must match value count {}",
            range.len(),
            values.len()
        );
        let values = values.into_iter().map(Some).collect::<Vec<_>>().into_boxed_slice();
        self.live_entries += values.len();
        self.entries.insert(range.start(), BatchedValueStoreEntry { range, values });
    }

    fn take(&mut self, gate_id: GateId) -> Option<Arc<T>> {
        let entry = self.entry_containing_mut(gate_id)?;
        let slot = &mut entry.values[gate_id.0 - entry.range.start().0];
        let value = slot.take();
        if value.is_some() {
            self.live_entries -= 1;
        }
        value
    }
}

type ErrorNormValueStore = BatchedValueStore<ErrorNorm>;
type ErrorNormWireStore = ErrorNormValueStore;
type ErrorNormMaterializedExprStore = BatchedValueStore<ErrorNormSummaryExpr>;
type ErrorNormExprStore = ErrorNormSummaryGateStore;

pub trait AffinePltEvaluator: PltEvaluator<ErrorNorm> + Sync {
    fn public_lookup_affine(
        &self,
        input: &ErrorNormSummaryExpr,
        plt: &PublicLut<DCRTPoly>,
        gate_id: GateId,
        lut_id: usize,
    ) -> ErrorNormSummaryExpr;
}

pub trait AffineSlotTransferEvaluator: SlotTransferEvaluator<ErrorNorm> + Sync {
    fn slot_transfer_affine(
        &self,
        input: &ErrorNormSummaryExpr,
        src_slots: &[(u32, Option<u32>)],
        gate_id: GateId,
    ) -> ErrorNormSummaryExpr;
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// One deferred transformation applied to an input matrix norm inside an affine summary.
///
/// The summary engine keeps matrix contributions symbolic as long as possible. Instead of eagerly
/// multiplying every input by gadget matrices or scalars, it records those steps here and applies
/// them only when the summary is finally instantiated or evaluated.
enum AffineMatrixTransformOp {
    Scalar(BigDecimal),
    Poly(PolyNorm),
    Matrix(PolyMatrixNorm),
    SplitColsLeft(usize),
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
/// Ordered transform pipeline attached to one affine input term.
///
/// Using an explicit transform chain lets the summary cache share structure across many outputs.
/// Several substitution paths can then reuse the same cached transform prefix instead of rebuilding
/// large matrix products from scratch.
struct AffineMatrixTransform {
    ops: Arc<[AffineMatrixTransformOp]>,
}

impl AffineMatrixTransform {
    fn apply(&self, mut value: PolyMatrixNorm) -> PolyMatrixNorm {
        for op in self.ops.iter() {
            value = match op {
                AffineMatrixTransformOp::Scalar(scalar) => value * scalar,
                AffineMatrixTransformOp::Poly(poly) => value * poly,
                AffineMatrixTransformOp::Matrix(matrix) => value * matrix.clone(),
                AffineMatrixTransformOp::SplitColsLeft(left_col_size) => {
                    value.split_cols(*left_col_size).0
                }
            };
        }
        value
    }

    fn then_op(&self, op: AffineMatrixTransformOp) -> Self {
        let mut ops = self.ops.iter().cloned().collect::<Vec<_>>();
        ops.push(op);
        Self { ops: Arc::from(ops) }
    }

    fn cache_key(&self) -> (usize, usize) {
        (self.ops.as_ptr() as usize, self.ops.len())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// One symbolic dependency on a sub-circuit input.
///
/// `input_idx` selects which summary input is referenced, `transform` describes how its matrix norm
/// should be reshaped, and `coefficient` is the scalar weight applied after the transform.
struct AffineInputTerm {
    input_idx: usize,
    transform: AffineMatrixTransform,
    coefficient: BigDecimal,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
/// Symbolic affine expression for the matrix-norm part of an error bound.
///
/// This is the core abstraction that makes sub-circuit summaries reusable. The plaintext norm can
/// be propagated exactly and cheaply, while the matrix norm is kept as "constant part plus
/// transformed input terms" until actual caller inputs are known.
struct AffineErrorNormExpr {
    const_term: Option<PolyMatrixNorm>,
    input_terms: Box<[AffineInputTerm]>,
}

impl AffineErrorNormExpr {
    fn forwarded_input_source(&self) -> Option<usize> {
        if self.const_term.is_none() &&
            self.input_terms.len() == 1 &&
            self.input_terms[0].transform == AffineMatrixTransform::default() &&
            self.input_terms[0].coefficient == BigDecimal::one()
        {
            Some(self.input_terms[0].input_idx)
        } else {
            None
        }
    }

    fn substitute_term_with_caches<F>(
        term: &AffineInputTerm,
        actual_input_at: &F,
        input_cache: &mut HashMap<usize, AffineErrorNormExpr>,
        transform_cache: &mut HashMap<(usize, usize, usize, usize), AffineMatrixTransform>,
    ) -> Self
    where
        F: Fn(usize) -> Arc<ErrorNormSummaryExpr> + Sync,
    {
        let actual_expr = input_cache
            .entry(term.input_idx)
            .or_insert_with(|| actual_input_at(term.input_idx).matrix_expr.clone());
        Self {
            const_term: actual_expr.const_term.clone().map(|value| {
                let value = term.transform.apply(value);
                if term.coefficient == BigDecimal::one() {
                    value
                } else {
                    value * &term.coefficient
                }
            }),
            input_terms: Self::combine_like_terms(
                actual_expr
                    .input_terms
                    .iter()
                    .map(|actual_term| {
                        let key = {
                            let (left_ptr, left_len) = actual_term.transform.cache_key();
                            let (right_ptr, right_len) = term.transform.cache_key();
                            (left_ptr, left_len, right_ptr, right_len)
                        };
                        let transform = transform_cache
                            .entry(key)
                            .or_insert_with(|| {
                                let mut ops =
                                    actual_term.transform.ops.iter().cloned().collect::<Vec<_>>();
                                ops.extend(term.transform.ops.iter().cloned());
                                AffineMatrixTransform { ops: Arc::from(ops) }
                            })
                            .clone();
                        AffineInputTerm {
                            input_idx: actual_term.input_idx,
                            transform,
                            coefficient: &actual_term.coefficient * &term.coefficient,
                        }
                    })
                    .collect::<Vec<_>>(),
            ),
        }
    }

    fn combine_like_terms(input_terms: Vec<AffineInputTerm>) -> Box<[AffineInputTerm]> {
        let mut grouped = HashMap::<(usize, usize, usize), usize>::new();
        let mut combined = Vec::<AffineInputTerm>::with_capacity(input_terms.len());
        for mut term in input_terms {
            let key = {
                let (ops_ptr, ops_len) = term.transform.cache_key();
                (term.input_idx, ops_ptr, ops_len)
            };
            if let Some(&combined_idx) = grouped.get(&key) {
                combined[combined_idx].coefficient += term.coefficient;
                continue;
            }
            let combined_idx = combined.len();
            term.coefficient = term.coefficient.clone();
            combined.push(term);
            grouped.insert(key, combined_idx);
        }
        combined.into_boxed_slice()
    }

    fn zero() -> Self {
        Self::default()
    }

    fn constant(const_term: PolyMatrixNorm) -> Self {
        Self { const_term: Some(const_term), input_terms: Vec::new().into_boxed_slice() }
    }

    fn input(input_idx: usize) -> Self {
        Self {
            const_term: None,
            input_terms: vec![AffineInputTerm {
                input_idx,
                transform: AffineMatrixTransform::default(),
                coefficient: BigDecimal::one(),
            }]
            .into_boxed_slice(),
        }
    }

    fn add_expr(&self, rhs: &Self) -> Self {
        let const_term = match (&self.const_term, &rhs.const_term) {
            (Some(left), Some(right)) => Some(left + right),
            (Some(left), None) => Some(left.clone()),
            (None, Some(right)) => Some(right.clone()),
            (None, None) => None,
        };
        let mut input_terms = Vec::with_capacity(self.input_terms.len() + rhs.input_terms.len());
        input_terms.extend(self.input_terms.iter().cloned());
        input_terms.extend(rhs.input_terms.iter().cloned());
        Self { const_term, input_terms: Self::combine_like_terms(input_terms) }
    }

    fn add_assign_expr(&mut self, rhs: Self) {
        let left_const = self.const_term.take();
        self.const_term = match (left_const, rhs.const_term) {
            (Some(left), Some(right)) => Some(left + &right),
            (Some(left), None) => Some(left),
            (None, Some(right)) => Some(right),
            (None, None) => None,
        };

        let mut input_terms =
            Vec::with_capacity(self.input_terms.len().saturating_add(rhs.input_terms.len()));
        input_terms.extend(std::mem::take(&mut self.input_terms).into_vec());
        input_terms.extend(rhs.input_terms.into_vec());
        self.input_terms = Self::combine_like_terms(input_terms);
    }

    fn transform_scalar(&self, scalar: &BigDecimal) -> Self {
        if scalar == &BigDecimal::one() {
            return self.clone();
        }
        Self {
            const_term: self.const_term.clone().map(|value| value * scalar),
            input_terms: self
                .input_terms
                .iter()
                .map(|term| AffineInputTerm {
                    input_idx: term.input_idx,
                    transform: term
                        .transform
                        .then_op(AffineMatrixTransformOp::Scalar(scalar.clone())),
                    coefficient: term.coefficient.clone(),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    fn transform_poly(&self, poly: &PolyNorm) -> Self {
        Self {
            const_term: self.const_term.clone().map(|value| value * poly),
            input_terms: self
                .input_terms
                .iter()
                .map(|term| AffineInputTerm {
                    input_idx: term.input_idx,
                    transform: term.transform.then_op(AffineMatrixTransformOp::Poly(poly.clone())),
                    coefficient: term.coefficient.clone(),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    fn transform_matrix(&self, matrix: &PolyMatrixNorm) -> Self {
        Self {
            const_term: self.const_term.clone().map(|value| value * matrix.clone()),
            input_terms: self
                .input_terms
                .iter()
                .map(|term| AffineInputTerm {
                    input_idx: term.input_idx,
                    transform: term
                        .transform
                        .then_op(AffineMatrixTransformOp::Matrix(matrix.clone())),
                    coefficient: term.coefficient.clone(),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    fn split_cols_left(&self, left_col_size: usize) -> Self {
        Self {
            const_term: self.const_term.clone().map(|value| value.split_cols(left_col_size).0),
            input_terms: self
                .input_terms
                .iter()
                .map(|term| AffineInputTerm {
                    input_idx: term.input_idx,
                    transform: term
                        .transform
                        .then_op(AffineMatrixTransformOp::SplitColsLeft(left_col_size)),
                    coefficient: term.coefficient.clone(),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    fn substitute_inputs_with_cached<F>(
        &self,
        actual_input_at: &F,
        cache: &mut HashMap<usize, AffineErrorNormExpr>,
    ) -> Self
    where
        F: Fn(usize) -> Arc<ErrorNormSummaryExpr> + Sync,
    {
        let mut substituted = if let Some(const_term) = &self.const_term {
            Self::constant(const_term.clone())
        } else {
            Self::zero()
        };
        if self.input_terms.len() < ERROR_NORM_EXPR_PAR_BATCH_SIZE {
            let mut transform_cache =
                HashMap::<(usize, usize, usize, usize), AffineMatrixTransform>::new();
            for term in &self.input_terms {
                substituted.add_assign_expr(Self::substitute_term_with_caches(
                    term,
                    actual_input_at,
                    cache,
                    &mut transform_cache,
                ));
            }
            return substituted;
        }
        let partials = self
            .input_terms
            .par_chunks(ERROR_NORM_EXPR_PAR_BATCH_SIZE)
            .map(|term_chunk| {
                let mut local_input_cache = HashMap::<usize, AffineErrorNormExpr>::new();
                let mut transform_cache =
                    HashMap::<(usize, usize, usize, usize), AffineMatrixTransform>::new();
                let mut partial = Self::zero();
                for term in term_chunk {
                    partial.add_assign_expr(Self::substitute_term_with_caches(
                        term,
                        actual_input_at,
                        &mut local_input_cache,
                        &mut transform_cache,
                    ));
                }
                partial
            })
            .collect::<Vec<_>>();
        for partial in partials {
            substituted.add_assign_expr(partial);
        }
        substituted
    }

    fn evaluate_with_cached<F>(
        &self,
        input_matrix_norm_at: &F,
        cache: &mut HashMap<usize, PolyMatrixNorm>,
    ) -> PolyMatrixNorm
    where
        F: Fn(usize) -> PolyMatrixNorm + Sync,
    {
        let mut value = self.const_term.clone().unwrap_or_else(|| {
            if let Some(first_term) = self.input_terms.first() {
                let input = cache
                    .entry(first_term.input_idx)
                    .or_insert_with(|| input_matrix_norm_at(first_term.input_idx))
                    .clone();
                return first_term.transform.apply(input) * BigDecimal::from(0u32);
            }
            panic!("affine error-norm zero expression must not be evaluated directly");
        });
        if self.input_terms.len() < ERROR_NORM_EXPR_PAR_BATCH_SIZE {
            for term in &self.input_terms {
                let input = cache
                    .entry(term.input_idx)
                    .or_insert_with(|| input_matrix_norm_at(term.input_idx))
                    .clone();
                let transformed = term.transform.apply(input);
                value = value + &(transformed * &term.coefficient);
            }
            return value;
        }
        let partials = self
            .input_terms
            .par_chunks(ERROR_NORM_EXPR_PAR_BATCH_SIZE)
            .filter_map(|term_chunk| {
                let mut local_cache = HashMap::<usize, PolyMatrixNorm>::new();
                let mut partial: Option<PolyMatrixNorm> = None;
                for term in term_chunk {
                    let input = local_cache
                        .entry(term.input_idx)
                        .or_insert_with(|| input_matrix_norm_at(term.input_idx))
                        .clone();
                    let transformed = term.transform.apply(input) * &term.coefficient;
                    partial = Some(match partial {
                        Some(current) => current + &transformed,
                        None => transformed,
                    });
                }
                partial
            })
            .collect::<Vec<_>>();
        for partial in partials {
            value = value + &partial;
        }
        value
    }

    fn is_identity_input(&self, input_idx: usize) -> bool {
        self.forwarded_input_source() == Some(input_idx)
    }

    fn remap_input_indices(&self, input_sources: &[usize]) -> Self {
        let remapped_terms = self
            .input_terms
            .iter()
            .map(|term| {
                let remapped_input_idx = *input_sources.get(term.input_idx).unwrap_or_else(|| {
                    panic!(
                        "error-norm summary input index {} out of range during remap",
                        term.input_idx
                    )
                });
                AffineInputTerm {
                    input_idx: remapped_input_idx,
                    transform: term.transform.clone(),
                    coefficient: term.coefficient.clone(),
                }
            })
            .collect::<Vec<_>>();
        Self {
            const_term: self.const_term.clone(),
            input_terms: Self::combine_like_terms(remapped_terms),
        }
    }
}

#[derive(Debug, Clone)]
/// Reusable symbolic error bound for one gate or one sub-circuit output.
///
/// The plaintext component is already concrete for the profiled input bounds. The matrix component
/// stays affine in the caller inputs, which allows `ErrorNormSubCircuitSummary` to substitute or
/// remap many outputs without rerunning the full simulator.
pub struct ErrorNormSummaryExpr {
    plaintext_norm: PolyNorm,
    matrix_expr: AffineErrorNormExpr,
}

impl ErrorNormSummaryExpr {
    fn forwarded_input_source(&self) -> Option<usize> {
        self.matrix_expr.forwarded_input_source()
    }

    fn constant(value: ErrorNorm) -> Self {
        Self {
            plaintext_norm: value.plaintext_norm,
            matrix_expr: AffineErrorNormExpr::constant(value.matrix_norm),
        }
    }

    fn input_with_plaintext_norm(input_idx: usize, plaintext_norm: PolyNorm) -> Self {
        Self { plaintext_norm, matrix_expr: AffineErrorNormExpr::input(input_idx) }
    }

    fn add_bound(&self, rhs: &Self) -> Self {
        Self {
            plaintext_norm: &self.plaintext_norm + &rhs.plaintext_norm,
            matrix_expr: self.matrix_expr.add_expr(&rhs.matrix_expr),
        }
    }

    fn add_assign_bound(&mut self, rhs: Self) {
        self.plaintext_norm = &self.plaintext_norm + &rhs.plaintext_norm;
        self.matrix_expr.add_assign_expr(rhs.matrix_expr);
    }

    fn scale_bound(&self, scalar: &BigDecimal) -> Self {
        if scalar == &BigDecimal::one() {
            return self.clone();
        }
        let scalar_poly = PolyNorm::new(self.plaintext_norm.ctx.clone(), scalar.clone());
        Self {
            plaintext_norm: self.plaintext_norm.clone() * &scalar_poly,
            matrix_expr: self.matrix_expr.transform_scalar(scalar),
        }
    }

    fn mul_bound(&self, rhs: &Self) -> Self {
        Self {
            plaintext_norm: &self.plaintext_norm * &rhs.plaintext_norm,
            matrix_expr: self
                .matrix_expr
                .transform_matrix(&PolyMatrixNorm::gadget_decomposed(
                    self.plaintext_norm.ctx.clone(),
                    self.plaintext_norm.ctx.m_g,
                ))
                .add_expr(&rhs.matrix_expr.transform_poly(&self.plaintext_norm)),
        }
    }

    fn small_scalar_mul_bound(&self, scalar: &[u32]) -> Self {
        let scalar_max = BigDecimal::from(*scalar.iter().max().unwrap());
        let scalar_poly = PolyNorm::new(self.plaintext_norm.ctx.clone(), scalar_max.clone());
        Self {
            plaintext_norm: self.plaintext_norm.clone() * &scalar_poly,
            matrix_expr: self.matrix_expr.transform_scalar(&scalar_max),
        }
    }

    fn large_scalar_mul_bound(&self, scalar: &[BigUint]) -> Self {
        let scalar_max = scalar.iter().max().unwrap().clone();
        let scalar_bd = BigDecimal::from(num_bigint::BigInt::from(scalar_max));
        let scalar_poly = PolyNorm::new(self.plaintext_norm.ctx.clone(), scalar_bd);
        Self {
            plaintext_norm: self.plaintext_norm.clone() * &scalar_poly,
            matrix_expr: self.matrix_expr.transform_matrix(&PolyMatrixNorm::gadget_decomposed(
                self.plaintext_norm.ctx.clone(),
                self.plaintext_norm.ctx.m_g,
            )),
        }
    }

    fn substitute_inputs_with_cached<F>(
        &self,
        actual_input_at: &F,
        cache: &mut HashMap<usize, AffineErrorNormExpr>,
    ) -> Self
    where
        F: Fn(usize) -> Arc<ErrorNormSummaryExpr> + Sync,
    {
        Self {
            plaintext_norm: self.plaintext_norm.clone(),
            matrix_expr: self.matrix_expr.substitute_inputs_with_cached(actual_input_at, cache),
        }
    }

    fn is_identity_input(&self, input_idx: usize) -> bool {
        self.matrix_expr.is_identity_input(input_idx)
    }

    fn remap_input_indices(&self, input_sources: &[usize]) -> Self {
        Self {
            plaintext_norm: self.plaintext_norm.clone(),
            matrix_expr: self.matrix_expr.remap_input_indices(input_sources),
        }
    }
}

#[derive(Debug, Default)]
/// Gate-indexed store used while constructing affine summaries.
///
/// This thin wrapper distinguishes summary-expression storage from concrete `ErrorNorm` wire
/// storage even though both rely on `BatchedValueStore` internally.
struct ErrorNormSummaryGateStore {
    entries: ErrorNormMaterializedExprStore,
}

impl ErrorNormSummaryGateStore {
    fn get(&self, gate_id: GateId) -> Option<Arc<ErrorNormSummaryExpr>> {
        self.entries.get(gate_id)
    }

    fn insert_single(&mut self, gate_id: GateId, value: ErrorNormSummaryExpr) {
        self.entries.insert_single(gate_id, value);
    }

    fn insert_batch_shared(&mut self, range: BatchedWire, values: Vec<Arc<ErrorNormSummaryExpr>>) {
        self.entries.insert_batch_shared(range, values);
    }

    fn take(&mut self, gate_id: GateId) -> Option<Arc<ErrorNormSummaryExpr>> {
        self.entries.take(gate_id)
    }

    fn live_entries(&self) -> usize {
        self.entries.live_entries
    }
}

#[derive(Debug, Clone)]
/// Cached affine summary for all outputs of one sub-circuit under one plaintext profile.
///
/// Each output remains expressed in terms of the sub-circuit inputs. Call sites can either share
/// the cached expressions directly, remap forwarded inputs cheaply, or perform full affine
/// substitution when the actual caller inputs are themselves symbolic.
struct ErrorNormSubCircuitSummary {
    outputs: Box<[Arc<ErrorNormSummaryExpr>]>,
}

impl ErrorNormSubCircuitSummary {
    fn output_len(&self) -> usize {
        self.outputs.len()
    }

    fn output_expr_arc(&self, output_idx: usize) -> Arc<ErrorNormSummaryExpr> {
        self.outputs
            .get(output_idx)
            .map(Arc::clone)
            .unwrap_or_else(|| panic!("error-norm summary output index {output_idx} out of range"))
    }

    fn clone_output_range_arcs(
        &self,
        output_range: std::ops::Range<usize>,
    ) -> Vec<Arc<ErrorNormSummaryExpr>> {
        self.outputs[output_range].iter().map(Arc::clone).collect()
    }

    fn forwarded_input_sources(
        actual_inputs: &[Arc<ErrorNormSummaryExpr>],
    ) -> Option<Box<[usize]>> {
        actual_inputs
            .iter()
            .map(|expr| expr.as_ref().forwarded_input_source())
            .collect::<Option<Vec<_>>>()
            .map(Vec::into_boxed_slice)
    }

    fn can_shallow_share_inputs(actual_inputs: &[Arc<ErrorNormSummaryExpr>]) -> bool {
        actual_inputs
            .iter()
            .enumerate()
            .all(|(input_idx, expr)| expr.as_ref().is_identity_input(input_idx))
    }

    fn remap_output_range_shared(
        &self,
        output_range: std::ops::Range<usize>,
        input_sources: &[usize],
    ) -> Vec<Arc<ErrorNormSummaryExpr>> {
        let outputs = &self.outputs[output_range];
        if outputs.len() < ERROR_NORM_EXPR_PAR_BATCH_SIZE {
            outputs
                .par_iter()
                .map(|expr| Arc::new(expr.as_ref().remap_input_indices(input_sources)))
                .collect()
        } else {
            outputs
                .par_chunks(ERROR_NORM_EXPR_PAR_BATCH_SIZE)
                .map(|output_chunk| {
                    output_chunk
                        .iter()
                        .map(|expr| Arc::new(expr.as_ref().remap_input_indices(input_sources)))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
                .into_iter()
                .flatten()
                .collect()
        }
    }

    fn substitute_output_range_shared(
        &self,
        output_range: std::ops::Range<usize>,
        actual_inputs: &[Arc<ErrorNormSummaryExpr>],
    ) -> Vec<Arc<ErrorNormSummaryExpr>> {
        if Self::can_shallow_share_inputs(actual_inputs) {
            return self.clone_output_range_arcs(output_range);
        }
        if let Some(input_sources) = Self::forwarded_input_sources(actual_inputs) {
            return self.remap_output_range_shared(output_range, input_sources.as_ref());
        }
        self.substitute_output_range_with_input_fn(output_range, &|input_idx| {
            actual_inputs
                .get(input_idx)
                .unwrap_or_else(|| {
                    panic!(
                        "error-norm summary input index {input_idx} out of range during substitution"
                    )
                })
                .clone()
        })
        .into_iter()
        .map(Arc::new)
        .collect()
    }

    fn substitute_output_shared(
        &self,
        output_idx: usize,
        actual_inputs: &[Arc<ErrorNormSummaryExpr>],
    ) -> Arc<ErrorNormSummaryExpr> {
        if Self::can_shallow_share_inputs(actual_inputs) {
            return self.output_expr_arc(output_idx);
        }
        if let Some(input_sources) = Self::forwarded_input_sources(actual_inputs) {
            return Arc::new(
                self.outputs[output_idx].as_ref().remap_input_indices(input_sources.as_ref()),
            );
        }
        Arc::new(self.substitute_output(output_idx, actual_inputs))
    }

    fn substitute_output_range_with_input_fn<F>(
        &self,
        output_range: std::ops::Range<usize>,
        actual_input_at: &F,
    ) -> Vec<ErrorNormSummaryExpr>
    where
        F: Fn(usize) -> Arc<ErrorNormSummaryExpr> + Sync,
    {
        let outputs = &self.outputs[output_range];
        if outputs.len() < ERROR_NORM_EXPR_PAR_BATCH_SIZE {
            outputs
                .par_iter()
                .map(|expr| {
                    let mut direct_substitution_cache =
                        HashMap::<usize, AffineErrorNormExpr>::new();
                    expr.as_ref().substitute_inputs_with_cached(
                        actual_input_at,
                        &mut direct_substitution_cache,
                    )
                })
                .collect()
        } else {
            outputs
                .par_chunks(ERROR_NORM_EXPR_PAR_BATCH_SIZE)
                .map(|output_chunk| {
                    let mut direct_substitution_cache =
                        HashMap::<usize, AffineErrorNormExpr>::new();
                    output_chunk
                        .iter()
                        .map(|expr| {
                            expr.as_ref().substitute_inputs_with_cached(
                                actual_input_at,
                                &mut direct_substitution_cache,
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
                .into_iter()
                .flatten()
                .collect()
        }
    }

    fn substitute_output(
        &self,
        output_idx: usize,
        actual_inputs: &[Arc<ErrorNormSummaryExpr>],
    ) -> ErrorNormSummaryExpr {
        self.substitute_output_range_shared(output_idx..output_idx + 1, actual_inputs)
            .into_iter()
            .next()
            .unwrap_or_else(|| panic!("error-norm summary output index {output_idx} out of range"))
            .as_ref()
            .clone()
    }

    fn evaluate_output_range_with_shared_cache<F>(
        &self,
        output_range: std::ops::Range<usize>,
        input_matrix_norm_at: &F,
    ) -> Vec<ErrorNorm>
    where
        F: Fn(usize) -> PolyMatrixNorm + Sync,
    {
        let outputs = &self.outputs[output_range];
        if outputs.len() < ERROR_NORM_EXPR_PAR_BATCH_SIZE {
            outputs
                .par_iter()
                .map(|expr| {
                    let mut cache = HashMap::<usize, PolyMatrixNorm>::new();
                    ErrorNorm::new(
                        expr.plaintext_norm.clone(),
                        expr.matrix_expr.evaluate_with_cached(input_matrix_norm_at, &mut cache),
                    )
                })
                .collect()
        } else {
            outputs
                .par_chunks(ERROR_NORM_EXPR_PAR_BATCH_SIZE)
                .map(|expr_chunk| {
                    let mut cache = HashMap::<usize, PolyMatrixNorm>::new();
                    expr_chunk
                        .iter()
                        .map(|expr| {
                            ErrorNorm::new(
                                expr.plaintext_norm.clone(),
                                expr.matrix_expr
                                    .evaluate_with_cached(input_matrix_norm_at, &mut cache),
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
                .into_iter()
                .flatten()
                .collect()
        }
    }
}

/// Multiply every summary expression in-place by the multiplicity of a summed sub-circuit group.
///
/// Summed sub-circuit calls can collapse many identical summaries into one representative summary
/// plus a count. Scaling in-place keeps sharing for the common expression structure while adjusting
/// the bound exactly as the original summed evaluation did.
fn scale_summary_expr_batch(exprs: &mut [Arc<ErrorNormSummaryExpr>], multiplier: usize) {
    if multiplier <= 1 {
        return;
    }
    let scalar = BigDecimal::from(
        u64::try_from(multiplier)
            .unwrap_or_else(|_| panic!("summary-expression multiplier {multiplier} exceeds u64")),
    );
    exprs.par_iter_mut().for_each(|expr| {
        let scaled = expr.as_ref().scale_bound(&scalar);
        *Arc::make_mut(expr) = scaled;
    });
}

/// Add one batch of output expressions into another batch position-wise.
fn add_summary_expr_batches_in_place(
    left: &mut [Arc<ErrorNormSummaryExpr>],
    right: Vec<Arc<ErrorNormSummaryExpr>>,
) {
    left.par_iter_mut().zip(right.into_par_iter()).for_each(|(left_expr, right_expr)| {
        Arc::make_mut(left_expr).add_assign_bound(right_expr.as_ref().clone());
    });
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// One run inside the run-length encoded plaintext profile representation.
struct PolyNormRun {
    len: usize,
    norm: PolyNorm,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Run-length encoded plaintext profile for summary cache keys.
///
/// Large shared-prefix call groups often repeat the same plaintext bounds many times. Compressing
/// those runs reduces both cache-key allocation cost and temporary cloning during summary
/// preparation, while `materialize` reconstructs the exact flat list when the simulator needs it.
struct CompressedPolyNormRuns {
    total_len: usize,
    runs: Arc<[PolyNormRun]>,
}

impl CompressedPolyNormRuns {
    fn from_slice(norms: &[PolyNorm]) -> Self {
        if norms.is_empty() {
            return Self { total_len: 0, runs: Arc::from([]) };
        }
        let mut runs = Vec::new();
        let mut current = norms[0].clone();
        let mut current_len = 1usize;
        for norm in &norms[1..] {
            if *norm == current {
                current_len += 1;
            } else {
                runs.push(PolyNormRun { len: current_len, norm: current });
                current = norm.clone();
                current_len = 1;
            }
        }
        runs.push(PolyNormRun { len: current_len, norm: current });
        Self { total_len: norms.len(), runs: Arc::from(runs) }
    }

    fn from_vec(norms: Vec<PolyNorm>) -> Self {
        Self::from_slice(&norms)
    }

    fn len(&self) -> usize {
        self.total_len
    }

    fn materialize(&self) -> Vec<PolyNorm> {
        let mut norms = Vec::with_capacity(self.total_len);
        for run in self.runs.iter() {
            norms.extend(std::iter::repeat_n(run.norm.clone(), run.len));
        }
        norms
    }
}

#[derive(Debug, Clone)]
/// Caller-observed plaintext profile for one sub-circuit invocation.
///
/// The `SharedPrefix` variant mirrors the circuit representation used by direct sub-circuit calls:
/// one reusable prefix input-set plus a call-local suffix. Keeping that distinction here lets the
/// summary preparation path reuse compressed prefix bounds across many calls.
enum ErrorNormInputPlaintextProfile {
    Flat(CompressedPolyNormRuns),
    SharedPrefix { prefix_norms: CompressedPolyNormRuns, suffix_norms: CompressedPolyNormRuns },
}

impl ErrorNormInputPlaintextProfile {
    fn flat_from_vec(norms: Vec<PolyNorm>) -> Self {
        Self::Flat(CompressedPolyNormRuns::from_vec(norms))
    }

    fn shared_prefix_from_parts(prefix_norms: &[PolyNorm], suffix_norms: Vec<PolyNorm>) -> Self {
        Self::SharedPrefix {
            prefix_norms: CompressedPolyNormRuns::from_slice(prefix_norms),
            suffix_norms: CompressedPolyNormRuns::from_vec(suffix_norms),
        }
    }

    fn materialize(&self) -> Vec<PolyNorm> {
        match self {
            Self::Flat(norms) => norms.materialize(),
            Self::SharedPrefix { prefix_norms, suffix_norms } => {
                let mut norms = Vec::with_capacity(prefix_norms.len() + suffix_norms.len());
                norms.extend(prefix_norms.materialize());
                norms.extend(suffix_norms.materialize());
                norms
            }
        }
    }
}

#[derive(Debug, Clone)]
/// Fully prepared request passed into summary-cache lookup/build helpers.
///
/// This bundles the call metadata after the surrounding execution layer has already collected and
/// compressed the relevant input plaintext norms.
struct ErrorNormPreparedSubCircuitSummaryRequest {
    sub_circuit_id: usize,
    param_bindings: Arc<[SubCircuitParamValue]>,
    input_plaintext_profile: ErrorNormInputPlaintextProfile,
}

mod engine;
mod evaluators;
mod gates;
mod layers;
mod store;
mod summary;

pub use evaluators::{
    NormBggPolyEncodingSTEvaluator, NormPltCommitEvaluator, NormPltGGH15Evaluator,
    NormPltLWEEvaluator, compute_preimage_norm,
};

#[cfg(test)]
mod tests;
