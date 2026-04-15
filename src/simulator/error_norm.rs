use crate::{
    circuit::{
        BatchedWire, ErrorNormExecutionLayer, Evaluable, PolyCircuit, PolyGateType,
        SubCircuitParamValue, batched_wire_slice_at, batched_wire_slice_len, gate::GateId,
        iter_batched_wire_gates,
    },
    element::PolyElem,
    impl_binop_with_refs,
    lookup::{PltEvaluator, PublicLut, commit_eval::compute_padded_len},
    poly::dcrt::poly::DCRTPoly,
    simulator::{SimulatorContext, poly_matrix_norm::PolyMatrixNorm, poly_norm::PolyNorm},
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
    ops::{Add, Mul, Sub},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};
use tracing::{debug, info};

const ERROR_NORM_OUTPUT_BATCH_SIZE: usize = 1024;
const ERROR_NORM_EXPR_PAR_BATCH_SIZE: usize = 32;
const ERROR_NORM_CALL_COMMIT_BATCH_SIZE: usize = 128;
const ERROR_NORM_SUMMED_CALL_COMMIT_BATCH_SIZE: usize = 16;
const ERROR_NORM_SUMMED_INNER_REDUCE_BATCH_SIZE: usize = 32;

type ErrorNormSubCircuitSummaryCache =
    DashMap<ErrorNormSubCircuitSummaryCacheKey, Arc<ErrorNormSubCircuitSummary>>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ErrorNormPolyNormKey {
    ctx_id: usize,
    norm: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ErrorNormSubCircuitSummaryCacheKey {
    circuit_key: usize,
    sub_circuit_id: usize,
    input_plaintext_norms: Vec<ErrorNormPolyNormKey>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ErrorNormSummaryBuildKey {
    cache_key: ErrorNormSubCircuitSummaryCacheKey,
    binding_sig: usize,
}

#[derive(Debug, Clone)]
struct ErrorNormSummaryNode {
    circuit: Arc<PolyCircuit<DCRTPoly>>,
    sub_circuit_id: usize,
    input_plaintext_norms: Arc<[PolyNorm]>,
    param_bindings: Arc<[SubCircuitParamValue]>,
    output_plaintext_norms: Arc<[PolyNorm]>,
    direct_call_keys: HashMap<usize, ErrorNormSubCircuitSummaryCacheKey>,
}

#[derive(Debug, Default)]
struct ErrorNormSummaryRegistry {
    nodes: HashMap<ErrorNormSubCircuitSummaryCacheKey, ErrorNormSummaryNode>,
    topo_order: Vec<ErrorNormSubCircuitSummaryCacheKey>,
}

fn error_norm_poly_norm_key(norm: &PolyNorm) -> ErrorNormPolyNormKey {
    ErrorNormPolyNormKey { ctx_id: Arc::as_ptr(&norm.ctx) as usize, norm: norm.norm.to_string() }
}

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

fn error_norm_summary_profile_max_bits(input_plaintext_norms: &[PolyNorm]) -> u64 {
    input_plaintext_norms.iter().map(|norm| bigdecimal_bits_ceil(&norm.norm)).max().unwrap_or(0)
}

fn gaussian_tail_bound_factor() -> BigDecimal {
    BigDecimal::from_f32(6.5).unwrap()
}

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
struct BatchedGateUseCountBatch {
    range: BatchedWire,
    counts: Box<[u32]>,
}

#[derive(Debug)]
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
struct BatchedValueStoreEntry<T> {
    range: BatchedWire,
    values: Box<[Option<Arc<T>>]>,
}

#[derive(Debug)]
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

impl PolyCircuit<DCRTPoly> {
    pub fn simulate_max_error_norm<P: AffinePltEvaluator>(
        &self,
        ctx: Arc<SimulatorContext>,
        input_norm_bound: BigDecimal,
        input_size: usize,
        e_init_norm: &BigDecimal,
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
    ) -> Vec<ErrorNorm>
    where
        ErrorNorm: Evaluable<P = DCRTPoly>,
    {
        let one_error = ErrorNorm::new(
            PolyNorm::one(ctx.clone()),
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, e_init_norm.clone(), None),
        );
        let input_error = ErrorNorm::new(
            PolyNorm::new(ctx.clone(), input_norm_bound.clone()),
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, e_init_norm.clone(), None),
        );
        info!("e_init_norm bits {}", bigdecimal_bits_ceil(e_init_norm));
        info!("input_norm_bound bits {}", bigdecimal_bits_ceil(&input_norm_bound));
        let input_errors = vec![input_error; input_size];
        self.eval_max_error_norm(one_error, input_errors, plt_evaluator, slot_transfer_evaluator)
    }

    fn eval_pure_regular_error_norm_gate(
        &self,
        gate_id: GateId,
        wires: &ErrorNormWireStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_values: &[Option<Arc<ErrorNorm>>],
    ) -> ErrorNorm {
        let gate = self.gate(gate_id);
        match &gate.gate_type {
            PolyGateType::Add => {
                let left = self.clone_error_norm_value_for_gate(
                    gate.input_gates[0],
                    wires,
                    input_gate_positions,
                    input_values,
                );
                let right = self.clone_error_norm_value_for_gate(
                    gate.input_gates[1],
                    wires,
                    input_gate_positions,
                    input_values,
                );
                left + &right
            }
            PolyGateType::Sub => {
                let left = self.clone_error_norm_value_for_gate(
                    gate.input_gates[0],
                    wires,
                    input_gate_positions,
                    input_values,
                );
                let right = self.clone_error_norm_value_for_gate(
                    gate.input_gates[1],
                    wires,
                    input_gate_positions,
                    input_values,
                );
                left - &right
            }
            PolyGateType::Mul => {
                let left = self.clone_error_norm_value_for_gate(
                    gate.input_gates[0],
                    wires,
                    input_gate_positions,
                    input_values,
                );
                let right = self.clone_error_norm_value_for_gate(
                    gate.input_gates[1],
                    wires,
                    input_gate_positions,
                    input_values,
                );
                left * &right
            }
            PolyGateType::SmallScalarMul { scalar } => {
                let scalar = scalar.resolve_small_scalar(&[]);
                self.clone_error_norm_value_for_gate(
                    gate.input_gates[0],
                    wires,
                    input_gate_positions,
                    input_values,
                )
                .small_scalar_mul(&(), scalar)
            }
            PolyGateType::LargeScalarMul { scalar } => {
                let scalar = scalar.resolve_large_scalar(&[]);
                self.clone_error_norm_value_for_gate(
                    gate.input_gates[0],
                    wires,
                    input_gate_positions,
                    input_values,
                )
                .large_scalar_mul(&(), scalar)
            }
            _ => panic!("gate {gate_id} is not pure in error-norm evaluation"),
        }
    }

    fn eval_evaluator_regular_error_norm_gate<P: AffinePltEvaluator>(
        &self,
        gate_id: GateId,
        one_error: &ErrorNorm,
        wires: &ErrorNormWireStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_values: &[Option<Arc<ErrorNorm>>],
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
    ) -> ErrorNorm {
        let gate = self.gate(gate_id);
        match &gate.gate_type {
            PolyGateType::SlotTransfer { src_slots } => {
                let src_slots = src_slots.resolve_slot_transfer(&[]);
                let input = self.clone_error_norm_value_for_gate(
                    gate.input_gates[0],
                    wires,
                    input_gate_positions,
                    input_values,
                );
                slot_transfer_evaluator.expect("slot transfer evaluator missing").slot_transfer(
                    &(),
                    &input,
                    src_slots.as_ref(),
                    gate_id,
                )
            }
            PolyGateType::PubLut { lut_id } => {
                let lut_id = lut_id.resolve_public_lookup(&[]);
                let input = self.clone_error_norm_value_for_gate(
                    gate.input_gates[0],
                    wires,
                    input_gate_positions,
                    input_values,
                );
                let lookup = self.lookup_table(lut_id);
                plt_evaluator.expect("public lookup evaluator missing").public_lookup(
                    &(),
                    lookup.as_ref(),
                    one_error,
                    &input,
                    gate_id,
                    lut_id,
                )
            }
            _ => self.eval_pure_regular_error_norm_gate(
                gate_id,
                wires,
                input_gate_positions,
                input_values,
            ),
        }
    }

    fn error_norm_input_gate_positions(
        &self,
        input_gate_ids: &[GateId],
    ) -> ErrorNormInputGatePositions {
        let mut positions = HashMap::with_capacity(input_gate_ids.len());
        for (idx, gate_id) in input_gate_ids.iter().copied().enumerate() {
            positions.insert(gate_id, idx);
        }
        positions
    }

    fn register_error_norm_summary_node<P: AffinePltEvaluator>(
        &self,
        sub_circuit_id: usize,
        input_plaintext_norms: Arc<[PolyNorm]>,
        param_bindings: Arc<[SubCircuitParamValue]>,
        one_error: &ErrorNorm,
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
        registry: &mut ErrorNormSummaryRegistry,
        visiting: &mut HashSet<ErrorNormSummaryBuildKey>,
    ) -> ErrorNormSubCircuitSummaryCacheKey {
        let sub_circuit = self.registered_sub_circuit_ref(sub_circuit_id);
        let cache_key = error_norm_sub_circuit_summary_cache_key(
            Arc::as_ptr(&sub_circuit) as usize,
            sub_circuit_id,
            &input_plaintext_norms,
        );
        let binding_sig = Arc::as_ptr(&param_bindings).cast::<u8>() as usize;
        let build_key = ErrorNormSummaryBuildKey { cache_key: cache_key.clone(), binding_sig };
        if registry.nodes.contains_key(&cache_key) {
            return cache_key;
        }
        if visiting.contains(&build_key) {
            panic!(
                "error-norm summary registration cycle detected sub_circuit_id={} inputs={}",
                sub_circuit_id,
                input_plaintext_norms.len()
            );
        }
        visiting.insert(build_key.clone());
        let (output_plaintext_norms, direct_call_keys) =
            sub_circuit.as_ref().compute_error_norm_plaintext_norms_for_summary(
                &input_plaintext_norms,
                &param_bindings,
                one_error,
                plt_evaluator,
                slot_transfer_evaluator,
                registry,
                visiting,
            );
        let node = ErrorNormSummaryNode {
            circuit: Arc::clone(&sub_circuit),
            sub_circuit_id,
            input_plaintext_norms,
            param_bindings,
            output_plaintext_norms,
            direct_call_keys,
        };
        registry.nodes.insert(cache_key.clone(), node);
        registry.topo_order.push(cache_key.clone());
        visiting.remove(&build_key);
        cache_key
    }

    fn compute_error_norm_plaintext_norms_for_summary<P: AffinePltEvaluator>(
        &self,
        input_plaintext_norms: &[PolyNorm],
        param_bindings: &[SubCircuitParamValue],
        one_error: &ErrorNorm,
        _plt_evaluator: Option<&P>,
        _slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
        registry: &mut ErrorNormSummaryRegistry,
        visiting: &mut HashSet<ErrorNormSummaryBuildKey>,
    ) -> (Arc<[PolyNorm]>, HashMap<usize, ErrorNormSubCircuitSummaryCacheKey>) {
        let input_gate_ids = self.sorted_input_gate_ids();
        assert_eq!(
            input_gate_ids.len(),
            input_plaintext_norms.len(),
            "error-norm summary plaintext profile length must match sub-circuit inputs"
        );
        let norm_ctx = input_plaintext_norms
            .first()
            .map(|norm| norm.ctx.clone())
            .unwrap_or_else(|| one_error.plaintext_norm.ctx.clone());
        let mut plaintext_norms = HashMap::<GateId, PolyNorm>::new();
        for (idx, gate_id) in input_gate_ids.iter().copied().enumerate() {
            plaintext_norms.insert(gate_id, input_plaintext_norms[idx].clone());
        }
        plaintext_norms.entry(GateId(0)).or_insert_with(|| PolyNorm::one(norm_ctx.clone()));
        let mut direct_call_keys = HashMap::<usize, ErrorNormSubCircuitSummaryCacheKey>::new();
        let execution_layers = self.error_norm_execution_layers();
        let mut shared_prefix_plaintext_norms = HashMap::<usize, Arc<[PolyNorm]>>::new();

        for ErrorNormExecutionLayer {
            regular_gate_ids,
            sub_circuit_call_ids,
            summed_sub_circuit_call_ids,
        } in execution_layers
        {
            for gate_id in regular_gate_ids {
                let gate = self.gate(gate_id);
                let norm = match &gate.gate_type {
                    PolyGateType::Add | PolyGateType::Sub => {
                        let left = plaintext_norms
                            .get(&gate.input_gates[0])
                            .unwrap_or_else(|| {
                                panic!("missing plaintext norm for gate {}", gate.input_gates[0])
                            })
                            .clone();
                        let right = plaintext_norms
                            .get(&gate.input_gates[1])
                            .unwrap_or_else(|| {
                                panic!("missing plaintext norm for gate {}", gate.input_gates[1])
                            })
                            .clone();
                        &left + &right
                    }
                    PolyGateType::Mul => {
                        let left = plaintext_norms
                            .get(&gate.input_gates[0])
                            .unwrap_or_else(|| {
                                panic!("missing plaintext norm for gate {}", gate.input_gates[0])
                            })
                            .clone();
                        let right = plaintext_norms
                            .get(&gate.input_gates[1])
                            .unwrap_or_else(|| {
                                panic!("missing plaintext norm for gate {}", gate.input_gates[1])
                            })
                            .clone();
                        &left * &right
                    }
                    PolyGateType::SmallScalarMul { scalar } => {
                        let scalar_max =
                            *scalar.resolve_small_scalar(param_bindings).iter().max().unwrap();
                        let scalar_bd = BigDecimal::from(scalar_max);
                        let scalar_poly = PolyNorm::new(norm_ctx.clone(), scalar_bd);
                        plaintext_norms
                            .get(&gate.input_gates[0])
                            .unwrap_or_else(|| {
                                panic!("missing plaintext norm for gate {}", gate.input_gates[0])
                            })
                            .clone() *
                            &scalar_poly
                    }
                    PolyGateType::LargeScalarMul { scalar } => {
                        let scalar_max = scalar
                            .resolve_large_scalar(param_bindings)
                            .iter()
                            .max()
                            .unwrap()
                            .clone();
                        let scalar_bd = BigDecimal::from(num_bigint::BigInt::from(scalar_max));
                        let scalar_poly = PolyNorm::new(norm_ctx.clone(), scalar_bd);
                        plaintext_norms
                            .get(&gate.input_gates[0])
                            .unwrap_or_else(|| {
                                panic!("missing plaintext norm for gate {}", gate.input_gates[0])
                            })
                            .clone() *
                            &scalar_poly
                    }
                    PolyGateType::SlotTransfer { src_slots } => {
                        let scalar_max = src_slots
                            .resolve_slot_transfer(param_bindings)
                            .iter()
                            .map(|(_, scalar)| u64::from(scalar.unwrap_or(1)))
                            .max()
                            .unwrap_or(1);
                        let scalar_bd = BigDecimal::from(scalar_max);
                        let scalar_poly = PolyNorm::new(norm_ctx.clone(), scalar_bd);
                        plaintext_norms
                            .get(&gate.input_gates[0])
                            .unwrap_or_else(|| {
                                panic!("missing plaintext norm for gate {}", gate.input_gates[0])
                            })
                            .clone() *
                            &scalar_poly
                    }
                    PolyGateType::PubLut { lut_id } => {
                        let lut_id = lut_id.resolve_public_lookup(param_bindings);
                        let plt = self.lookup_table(lut_id);
                        let plaintext_bd = BigDecimal::from(num_bigint::BigInt::from(
                            plt.max_output_row().1.value().clone(),
                        ));
                        PolyNorm::new(norm_ctx.clone(), plaintext_bd)
                    }
                    PolyGateType::Input => {
                        panic!("input gate {gate_id} should already have a plaintext norm")
                    }
                    PolyGateType::SubCircuitOutput { .. } |
                    PolyGateType::SummedSubCircuitOutput { .. } => {
                        unreachable!("subcircuit outputs are handled in subcall phases")
                    }
                };
                plaintext_norms.insert(gate_id, norm);
            }

            for call_id in sub_circuit_call_ids {
                let shared_prefix_set_id = self.sub_circuit_call_shared_prefix_set_id(call_id);
                self.with_sub_circuit_call_by_id(
                    call_id,
                    |sub_id, bindings, _shared_prefix, suffix, output_gate_ids| {
                        let input_plaintext_profile = if let Some(input_set_id) = shared_prefix_set_id {
                            let prefix_norms = shared_prefix_plaintext_norms
                                .entry(input_set_id)
                                .or_insert_with(|| {
                                    let input_ids = self.input_set(input_set_id);
                                    Arc::from(
                                        input_ids
                                            .iter()
                                            .flat_map(|batch| batch.gate_ids())
                                            .map(|input_id| {
                                                plaintext_norms
                                                    .get(&input_id)
                                                    .unwrap_or_else(|| {
                                                        panic!(
                                                            "missing plaintext norm for gate {input_id}"
                                                        )
                                                    })
                                                    .clone()
                                            })
                                            .collect::<Vec<_>>(),
                                    )
                                })
                                .clone();
                            ErrorNormInputPlaintextProfile::shared_prefix_from_parts(
                                prefix_norms.as_ref(),
                                suffix
                                    .iter()
                                    .flat_map(|batch| batch.gate_ids())
                                    .map(|input_id| {
                                        plaintext_norms
                                            .get(&input_id)
                                            .unwrap_or_else(|| panic!("missing plaintext norm for gate {input_id}"))
                                            .clone()
                                    })
                                    .collect::<Vec<_>>(),
                            )
                        } else {
                            ErrorNormInputPlaintextProfile::flat_from_vec(
                                suffix
                                    .iter()
                                    .flat_map(|batch| batch.gate_ids())
                                    .map(|input_id| {
                                        plaintext_norms
                                            .get(&input_id)
                                            .unwrap_or_else(|| panic!("missing plaintext norm for gate {input_id}"))
                                            .clone()
                                    })
                                    .collect::<Vec<_>>(),
                            )
                        };
                        let child_key = self.register_error_norm_summary_node(
                            sub_id,
                            Arc::from(input_plaintext_profile.materialize()),
                            bindings.clone(),
                            one_error,
                            _plt_evaluator,
                            _slot_transfer_evaluator,
                            registry,
                            visiting,
                        );
                        direct_call_keys.insert(call_id, child_key.clone());
                        let child_node = registry
                            .nodes
                            .get(&child_key)
                            .unwrap_or_else(|| panic!("missing child node for call {call_id}"));
                        for (output_idx, output_gate_id) in output_gate_ids.iter().copied().enumerate() {
                            plaintext_norms.insert(
                                output_gate_id,
                                child_node.output_plaintext_norms[output_idx].clone(),
                            );
                        }
                    },
                );
            }

            for summed_call_id in summed_sub_circuit_call_ids {
                let (sub_id, call_input_set_ids, call_binding_set_ids, output_gate_ids) = self
                    .with_summed_sub_circuit_call_by_id(
                        summed_call_id,
                        |sub_id, call_input_set_ids, call_binding_set_ids, output_gate_ids| {
                            (
                                sub_id,
                                call_input_set_ids.to_vec(),
                                call_binding_set_ids.to_vec(),
                                output_gate_ids.to_vec(),
                            )
                        },
                    );
                let mut output_accum: Vec<Option<PolyNorm>> = vec![None; output_gate_ids.len()];
                for (input_set_id, binding_set_id) in
                    call_input_set_ids.into_iter().zip(call_binding_set_ids.into_iter())
                {
                    let input_ids = self.input_set(input_set_id);
                    let child_inputs = input_ids
                        .iter()
                        .flat_map(|batch| batch.gate_ids())
                        .map(|input_id| {
                            plaintext_norms
                                .get(&input_id)
                                .unwrap_or_else(|| {
                                    panic!("missing plaintext norm for gate {input_id}")
                                })
                                .clone()
                        })
                        .collect::<Vec<_>>();
                    let child_key = self.register_error_norm_summary_node(
                        sub_id,
                        Arc::from(child_inputs),
                        self.binding_set(binding_set_id),
                        one_error,
                        _plt_evaluator,
                        _slot_transfer_evaluator,
                        registry,
                        visiting,
                    );
                    let child_node = registry.nodes.get(&child_key).unwrap_or_else(|| {
                        panic!("missing child node for summed call {summed_call_id}")
                    });
                    for (idx, output_norm) in child_node.output_plaintext_norms.iter().enumerate() {
                        output_accum[idx] = Some(match output_accum[idx].take() {
                            Some(acc) => &acc + output_norm,
                            None => output_norm.clone(),
                        });
                    }
                }
                for (idx, output_gate_id) in output_gate_ids.into_iter().enumerate() {
                    let norm = output_accum[idx].clone().unwrap_or_else(|| {
                        panic!("summed sub-circuit call {summed_call_id} has no inner calls")
                    });
                    plaintext_norms.insert(output_gate_id, norm);
                }
            }
        }

        let output_gate_ids = self.output_gate_ids();
        let output_plaintext_norms = Arc::from(
            output_gate_ids
                .iter()
                .copied()
                .map(|gate_id| {
                    plaintext_norms
                        .get(&gate_id)
                        .unwrap_or_else(|| {
                            panic!("missing plaintext norm for output gate {gate_id}")
                        })
                        .clone()
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        (output_plaintext_norms, direct_call_keys)
    }

    fn build_sub_circuit_summary_from_node<P: AffinePltEvaluator>(
        &self,
        node: &ErrorNormSummaryNode,
        one_error: &ErrorNorm,
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
        summary_cache: &ErrorNormSubCircuitSummaryCache,
    ) -> ErrorNormSubCircuitSummary {
        let input_plaintext_norms = node.input_plaintext_norms.as_ref();
        let param_bindings = node.param_bindings.as_ref();
        let input_gate_ids = self.sorted_input_gate_ids();
        assert_eq!(
            input_gate_ids.len(),
            input_plaintext_norms.len(),
            "error-norm summary plaintext profile length must match sub-circuit inputs"
        );
        let output_gate_ids = self.output_gate_ids();
        let output_gate_ids_set = output_gate_ids.iter().copied().collect::<HashSet<_>>();
        let log_large_summary = self.num_gates() >= 100_000 || output_gate_ids.len() >= 100_000;
        // debug!(
        //     "error-norm summary build start sub_circuit_id={} inputs={} outputs={} gates={}
        // large_summary={}",     node.sub_circuit_id,
        //     input_gate_ids.len(),
        //     output_gate_ids.len(),
        //     self.num_gates(),
        //     log_large_summary
        // );
        let mut output_positions = HashMap::<GateId, Vec<usize>>::new();
        for (idx, gate_id) in output_gate_ids.iter().copied().enumerate() {
            output_positions.entry(gate_id).or_default().push(idx);
        }
        let mut final_output_exprs: Vec<Option<Arc<ErrorNormSummaryExpr>>> =
            vec![None; output_gate_ids.len()];
        let input_gate_positions = self.error_norm_input_gate_positions(&input_gate_ids);
        let mut gate_exprs = ErrorNormExprStore::default();
        gate_exprs.insert_single(GateId(0), ErrorNormSummaryExpr::constant(one_error.clone()));
        let execution_layers = self.error_norm_execution_layers();
        // debug!(
        //     "error-norm summary execution layers ready sub_circuit_id={} levels={}
        // elapsed_ms={}",     node.sub_circuit_id,
        //     execution_layers.len(),
        //     execution_layers_start.elapsed().as_millis()
        // );
        let mut remaining_use_count = self.error_norm_remaining_use_count(&execution_layers);
        // debug!(
        //     "error-norm summary remaining-use-count ready sub_circuit_id={} entries={}
        // elapsed_ms={}",     node.sub_circuit_id,
        //     remaining_use_count.len(),
        //     remaining_use_count_start.elapsed().as_millis()
        // );
        for (
            level_idx,
            ErrorNormExecutionLayer {
                regular_gate_ids,
                sub_circuit_call_ids,
                summed_sub_circuit_call_ids,
            },
        ) in execution_layers.into_iter().enumerate()
        {
            let (pure_regular_gate_ids, evaluator_regular_gate_ids) = regular_gate_ids
                .into_par_iter()
                .fold(
                    || (Vec::new(), Vec::new()),
                    |mut acc, gate_id| {
                        match &self.gate(gate_id).gate_type {
                            PolyGateType::Input => {
                                panic!(
                                    "input gate {gate_id} should already be present in the error-norm summary"
                                )
                            }
                            PolyGateType::Add |
                            PolyGateType::Sub |
                            PolyGateType::Mul |
                            PolyGateType::SmallScalarMul { .. } |
                            PolyGateType::LargeScalarMul { .. } => acc.0.push(gate_id),
                            PolyGateType::SlotTransfer { .. } | PolyGateType::PubLut { .. } => {
                                acc.1.push(gate_id)
                            }
                            PolyGateType::SubCircuitOutput { .. } |
                            PolyGateType::SummedSubCircuitOutput { .. } => {
                                unreachable!(
                                    "subcircuit outputs should not appear in regular execution layers"
                                )
                            }
                        }
                        acc
                    },
                )
                .reduce(
                    || (Vec::new(), Vec::new()),
                    |mut left, mut right| {
                        left.0.append(&mut right.0);
                        left.1.append(&mut right.1);
                        left
                    },
                );
            // debug!(
            //     "error-norm summary level ready sub_circuit_id={} level={} regular={} pure_regular={} evaluator_regular={} sub_calls={} summed_sub_calls={} elapsed_ms=0",
            //     node.sub_circuit_id,
            //     level_idx,
            //     pure_regular_gate_ids.len() + evaluator_regular_gate_ids.len(),
            //     pure_regular_gate_ids.len(),
            //     evaluator_regular_gate_ids.len(),
            //     sub_circuit_call_ids.len(),
            //     summed_sub_circuit_call_ids.len(),
            // );

            let pure_regular_exprs = pure_regular_gate_ids
                .iter()
                .map(|gate_id| {
                    (
                        *gate_id,
                        self.build_pure_regular_error_norm_expr(
                            *gate_id,
                            &gate_exprs,
                            &input_gate_positions,
                            input_plaintext_norms,
                            one_error,
                            param_bindings,
                            plt_evaluator,
                            slot_transfer_evaluator,
                            summary_cache,
                            node,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            let evaluator_regular_exprs = evaluator_regular_gate_ids
                .iter()
                .map(|gate_id| {
                    (
                        *gate_id,
                        self.build_evaluator_regular_error_norm_expr(
                            *gate_id,
                            &gate_exprs,
                            &input_gate_positions,
                            input_plaintext_norms,
                            one_error,
                            param_bindings,
                            plt_evaluator,
                            slot_transfer_evaluator,
                            summary_cache,
                            node,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            // debug!(
            //     "error-norm summary level evaluator regular finished sub_circuit_id={} level={}
            // evaluator_regular={} elapsed_ms={}",     node.sub_circuit_id,
            //     level_idx,
            //     evaluator_regular_gate_ids.len(),
            //     evaluator_regular_start.elapsed().as_millis()
            // );

            self.store_error_norm_summary_expr_batch(
                pure_regular_exprs.into_iter().chain(evaluator_regular_exprs.into_iter()),
                &output_positions,
                &remaining_use_count,
                &mut gate_exprs,
                &mut final_output_exprs,
            );
            for gate_id in pure_regular_gate_ids.iter().chain(evaluator_regular_gate_ids.iter()) {
                let gate = self.gate(*gate_id);
                self.release_error_norm_summary_inputs(
                    gate.input_gates.iter().copied(),
                    &output_gate_ids_set,
                    &mut remaining_use_count,
                    &mut gate_exprs,
                );
            }

            let prepare_sub_calls_start = Instant::now();
            let mut prepared_sub_call_count = 0usize;
            for call_id_chunk in sub_circuit_call_ids.chunks(ERROR_NORM_CALL_COMMIT_BATCH_SIZE) {
                let prepared_sub_circuit_calls = call_id_chunk
                    .iter()
                    .map(|call_id| {
                        let child_key = node.direct_call_keys.get(call_id).unwrap_or_else(|| {
                            panic!(
                                "missing direct call key for call_id {} in sub_circuit_id {}",
                                call_id, node.sub_circuit_id
                            )
                        });
                        let summary = summary_cache.get(child_key).unwrap_or_else(|| {
                            panic!(
                                "missing summary for child call in sub_circuit_id {}",
                                node.sub_circuit_id
                            )
                        });
                        (*call_id, Arc::clone(summary.value()))
                    })
                    .collect::<Vec<_>>();
                let mut process_prepared_sub_circuit_calls =
                    |prepared_sub_circuit_calls: &[(usize, Arc<ErrorNormSubCircuitSummary>)],
                     call_batch_size: usize| {
                        enum PreparedSummaryOutputs {
                            Shared {
                                output_range: BatchedWire,
                                outputs: Vec<Arc<ErrorNormSummaryExpr>>,
                                finished_call: bool,
                                release_counts: Option<Vec<(GateId, u32)>>,
                            },
                        }

                        prepared_sub_call_count += prepared_sub_circuit_calls.len();
                        for prepared_chunk in prepared_sub_circuit_calls.chunks(call_batch_size) {
                            let max_output_len = prepared_chunk
                                .iter()
                                .map(|(_, summary)| summary.output_len())
                                .max()
                                .unwrap_or(0);
                            let use_whole_call_shared_cache = prepared_chunk.len() <= 4 &&
                                max_output_len <= ERROR_NORM_OUTPUT_BATCH_SIZE * 4;
                            for chunk_start in
                                (0..max_output_len).step_by(ERROR_NORM_OUTPUT_BATCH_SIZE)
                            {
                                let prepared_outputs = prepared_chunk
                                .par_iter()
                                .filter_map(|(call_id, summary)| {
                                    if chunk_start >= summary.output_len() {
                                        return None;
                                    }
                                    self.with_sub_circuit_call_by_id(
                                        *call_id,
                                        |actual_sub_circuit_id,
                                         _param_bindings,
                                         shared_prefix,
                                         suffix,
                                         output_gate_ids| {
                                            let _ = actual_sub_circuit_id;
                                            assert_eq!(
                                                output_gate_ids.len(),
                                                summary.output_len(),
                                                "error-norm summary sub-circuit output count mismatch for call {call_id}"
                                            );
                                            let output_range =
                                                BatchedWire::from_batches(output_gate_ids.iter().copied());
                                            let (output_slice, finished_call) = if
                                                use_whole_call_shared_cache
                                            {
                                                (output_range, true)
                                            } else {
                                                let chunk_end =
                                                    (chunk_start + ERROR_NORM_OUTPUT_BATCH_SIZE)
                                                        .min(output_range.len());
                                                (
                                                    output_range.slice(chunk_start..chunk_end),
                                                    chunk_end == output_range.len(),
                                                )
                                            };
                                            let actual_inputs = Arc::<[Arc<ErrorNormSummaryExpr>]>::from(
                                                self.collect_error_norm_summary_exprs_for_inputs_direct(
                                                    shared_prefix,
                                                    suffix,
                                                    &gate_exprs,
                                                    &input_gate_positions,
                                                    input_plaintext_norms,
                                                    one_error,
                                                    plt_evaluator,
                                                    slot_transfer_evaluator,
                                                    summary_cache,
                                                    node,
                                                ),
                                            );
                                            let release_counts = if finished_call {
                                                // debug!(
                                                //     "error-norm summary direct-sub-call release-count collect start sub_circuit_id={} level={} call_id={} output_chunk_start={} output_chunk_end={} shared_prefix_batches={} suffix_batches={}",
                                                //     node.sub_circuit_id,
                                                //     level_idx,
                                                //     call_id,
                                                //     output_slice.start().0,
                                                //     output_slice.end().0,
                                                //     shared_prefix.len(),
                                                //     suffix.len(),
                                                // );
                                                let release_counts = self
                                                    .collect_error_norm_summary_release_counts_for_direct_inputs(
                                                        shared_prefix,
                                                        suffix,
                                                        &output_gate_ids_set,
                                                        &remaining_use_count,
                                                    );
                                                // debug!(
                                                //     "error-norm summary direct-sub-call release-count collect finished sub_circuit_id={} level={} call_id={} output_chunk_start={} output_chunk_end={} release_gates={} elapsed_ms={}",
                                                //     node.sub_circuit_id,
                                                //     level_idx,
                                                //     call_id,
                                                //     output_slice.start().0,
                                                //     output_slice.end().0,
                                                //     release_counts.len(),
                                                //     release_collect_start.elapsed().as_millis(),
                                                // );
                                                Some(release_counts)
                                            } else {
                                                None
                                            };
                                            // debug!(
                                            //     "error-norm summary direct-sub-call substitute start sub_circuit_id={} level={} call_id={} output_chunk_start={} output_chunk_end={} output_len={} actual_inputs={} whole_call_cache={} share_fast_path={} expr_batch_size={}",
                                            //     node.sub_circuit_id,
                                            //     level_idx,
                                            //     call_id,
                                            //     output_slice.start().0,
                                            //     output_slice.end().0,
                                            //     output_slice.len(),
                                            //     actual_inputs.len(),
                                            //     use_whole_call_shared_cache,
                                            //     output_slice.len() >= ERROR_NORM_FORWARD_OUTPUT_MIN_LEN &&
                                            //         actual_inputs.len() <=
                                            //             output_slice.len() *
                                            //                 ERROR_NORM_FORWARD_INPUT_OUTPUT_RATIO_LIMIT,
                                            //     ERROR_NORM_EXPR_PAR_BATCH_SIZE,
                                            // );
                                            let outputs = if use_whole_call_shared_cache {
                                                summary.substitute_output_range_shared(
                                                    0..output_range.len(),
                                                    actual_inputs.as_ref(),
                                                )
                                            } else {
                                                summary.substitute_output_range_shared(
                                                    chunk_start..chunk_start + output_slice.len(),
                                                    actual_inputs.as_ref(),
                                                )
                                            };
                                            // debug!(
                                            //     "error-norm summary direct-sub-call substitute finished sub_circuit_id={} level={} call_id={} output_chunk_start={} output_chunk_end={} output_len={} actual_inputs={} elapsed_ms={}",
                                            //     node.sub_circuit_id,
                                            //     level_idx,
                                            //     call_id,
                                            //     output_slice.start().0,
                                            //     output_slice.end().0,
                                            //     output_slice.len(),
                                            //     actual_inputs.len(),
                                            //     substitute_start.elapsed().as_millis(),
                                            // );
                                            Some(PreparedSummaryOutputs::Shared {
                                                output_range: output_slice,
                                                outputs,
                                                finished_call,
                                                release_counts,
                                            })
                                        },
                                    )
                                })
                                .collect::<Vec<_>>();
                                for prepared_output in prepared_outputs {
                                    match prepared_output {
                                        PreparedSummaryOutputs::Shared {
                                            output_range,
                                            outputs,
                                            finished_call,
                                            release_counts,
                                        } => {
                                            // debug!(
                                            //     "error-norm summary direct-sub-call store start
                                            // sub_circuit_id={} level={} call_id={}
                                            // output_chunk_start={} output_chunk_end={}
                                            // output_len={} finished_call={} mode=shared",
                                            //     node.sub_circuit_id,
                                            //     level_idx,
                                            //     call_id,
                                            //     output_range.start().0,
                                            //     output_range.end().0,
                                            //     output_range.len(),
                                            //     finished_call,
                                            // );
                                            self.store_error_norm_summary_expr_batch_shared(
                                                output_range.gate_ids().zip(outputs.into_iter()),
                                                &output_positions,
                                                &remaining_use_count,
                                                &mut gate_exprs,
                                                &mut final_output_exprs,
                                            );
                                            // debug!(
                                            //     "error-norm summary direct-sub-call store
                                            // finished sub_circuit_id={} level={} call_id={}
                                            // output_chunk_start={} output_chunk_end={}
                                            // gate_expr_entries={} elapsed_ms={} mode=shared",
                                            //     node.sub_circuit_id,
                                            //     level_idx,
                                            //     call_id,
                                            //     output_range.start().0,
                                            //     output_range.end().0,
                                            //     gate_exprs.live_entries(),
                                            //     store_start.elapsed().as_millis(),
                                            // );
                                            if finished_call {
                                                let release_counts = release_counts.expect(
                                                    "error-norm direct sub-circuit finished call must provide release counts",
                                                );
                                                // debug!(
                                                //     "error-norm summary direct-sub-call release
                                                // commit start sub_circuit_id={} level={}
                                                // call_id={} release_gates={} release_total={}
                                                // remaining_use_entries={} mode=shared",
                                                //     node.sub_circuit_id,
                                                //     level_idx,
                                                //     call_id,
                                                //     release_gate_count,
                                                //     release_total_count,
                                                //     remaining_use_count.len(),
                                                // );
                                                self.release_error_norm_summary_inputs_batched(
                                                    release_counts,
                                                    &mut remaining_use_count,
                                                    &mut gate_exprs,
                                                );
                                                // debug!(
                                                //     "error-norm summary direct-sub-call release
                                                // commit finished sub_circuit_id={} level={}
                                                // call_id={} release_gates={} release_total={}
                                                // remaining_use_entries={} gate_expr_entries={}
                                                // elapsed_ms={} mode=shared",
                                                //     node.sub_circuit_id,
                                                //     level_idx,
                                                //     call_id,
                                                //     release_gate_count,
                                                //     release_total_count,
                                                //     remaining_use_count.len(),
                                                //     gate_exprs.live_entries(),
                                                //     release_start.elapsed().as_millis(),
                                                // );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    };
                process_prepared_sub_circuit_calls(
                    &prepared_sub_circuit_calls,
                    ERROR_NORM_CALL_COMMIT_BATCH_SIZE,
                );
                if log_large_summary &&
                    (prepared_sub_call_count % (ERROR_NORM_CALL_COMMIT_BATCH_SIZE * 8) == 0 ||
                        prepared_sub_call_count == sub_circuit_call_ids.len())
                {
                    debug!(
                        "error-norm summary sub-call commit batch finished sub_circuit_id={} level={} processed_calls={} total_calls={} elapsed_ms={}",
                        node.sub_circuit_id,
                        level_idx,
                        prepared_sub_call_count,
                        sub_circuit_call_ids.len(),
                        prepare_sub_calls_start.elapsed().as_millis()
                    );
                }
            }
            // debug!(
            //     "error-norm summary level sub-call preparation finished sub_circuit_id={}
            // level={} sub_calls={} elapsed_ms={}",     node.sub_circuit_id,
            //     level_idx,
            //     prepared_sub_call_count,
            //     prepare_sub_calls_start.elapsed().as_millis()
            // );

            let prepare_summed_sub_calls_start = Instant::now();
            let mut prepared_summed_call_count = 0usize;
            for summed_call_chunk in
                summed_sub_circuit_call_ids.chunks(ERROR_NORM_SUMMED_CALL_COMMIT_BATCH_SIZE)
            {
                for summed_call_id in summed_call_chunk {
                    debug!(
                        "error-norm summary level summed-sub-call build start sub_circuit_id={} level={} summed_call_id={} inner_requests={} processed_summed_calls_before={} elapsed_ms={}",
                        node.sub_circuit_id,
                        level_idx,
                        summed_call_id,
                        self.with_summed_sub_circuit_call_by_id(
                            *summed_call_id,
                            |_sub_circuit_id,
                             call_input_set_ids,
                             _call_binding_set_ids,
                             _output_gate_ids| {
                                call_input_set_ids.len()
                            },
                        ),
                        prepared_summed_call_count,
                        prepare_summed_sub_calls_start.elapsed().as_millis(),
                    );
                    prepared_summed_call_count += 1;
                    self.with_summed_sub_circuit_call_by_id(
                        *summed_call_id,
                        |actual_sub_circuit_id,
                         call_input_set_ids,
                         call_binding_set_ids,
                         output_gate_ids| {
                            let output_range_full =
                                BatchedWire::from_batches(output_gate_ids.iter().copied());
                            let inner_total = call_input_set_ids.len();
                            let grouped_summaries =
                                self.build_grouped_summed_sub_circuit_summaries_direct(
                                    actual_sub_circuit_id,
                                    call_input_set_ids,
                                    call_binding_set_ids,
                                    &gate_exprs,
                                    &input_gate_positions,
                                    input_plaintext_norms,
                                    one_error,
                                    plt_evaluator,
                                    slot_transfer_evaluator,
                                    summary_cache,
                                    node,
                                );
                            for chunk_start in
                                (0..output_range_full.len()).step_by(ERROR_NORM_OUTPUT_BATCH_SIZE)
                            {
                                let chunk_end = (chunk_start + ERROR_NORM_OUTPUT_BATCH_SIZE)
                                    .min(output_range_full.len());
                                let output_range =
                                    output_range_full.slice(chunk_start..chunk_end);
                                let output_len = output_range.len();
                                let finished_call = chunk_end == output_range_full.len();
                                let total_inner_chunks = grouped_summaries.len();
                                debug!(
                                    "error-norm summary summed-sub-call reduce start sub_circuit_id={} level={} summed_call_id={} output_chunk_start={} output_chunk_end={} inner_summaries={} inner_chunk_total={}",
                                    node.sub_circuit_id,
                                    level_idx,
                                    summed_call_id,
                                    chunk_start,
                                    chunk_end,
                                    inner_total,
                                    total_inner_chunks,
                                );
                                let outputs = grouped_summaries
                                    .par_iter()
                                    .map(|(summary, input_set_counts)| {
                                        assert_eq!(
                                            output_gate_ids.len(),
                                            summary.output_len(),
                                            "error-norm summary summed sub-circuit output count mismatch for call {summed_call_id}"
                                        );
                                        input_set_counts
                                            .par_iter()
                                            .map(|(input_set_id, call_count)| {
                                                let input_ids = self.input_set(*input_set_id);
                                                let actual_inputs = self
                                                    .collect_error_norm_summary_exprs_for_input_set_direct(
                                                        input_ids.as_ref(),
                                                        &gate_exprs,
                                                        &input_gate_positions,
                                                        input_plaintext_norms,
                                                        one_error,
                                                        plt_evaluator,
                                                        slot_transfer_evaluator,
                                                        summary_cache,
                                                        node,
                                                    );
                                                let mut outputs = summary
                                                    .substitute_output_range_shared(
                                                        chunk_start..chunk_end,
                                                        actual_inputs.as_ref(),
                                                    );
                                                scale_summary_expr_batch(&mut outputs, *call_count);
                                                outputs
                                            })
                                            .reduce_with(|mut left, right| {
                                                add_summary_expr_batches_in_place(&mut left, right);
                                                left
                                            })
                                            .unwrap_or_else(|| {
                                                panic!(
                                                    "summed sub-circuit call requires at least one inner call"
                                                )
                                            })
                                    })
                                    .reduce_with(|mut left, right| {
                                        add_summary_expr_batches_in_place(&mut left, right);
                                        left
                                    })
                                    .unwrap_or_else(|| {
                                    panic!(
                                        "summed sub-circuit call {} has no inner summaries",
                                        summed_call_id
                                    )
                                });
                                let store_start = Instant::now();
                                debug!(
                                    "error-norm summary summed-sub-call store start sub_circuit_id={} level={} summed_call_id={} output_chunk_start={} output_chunk_end={} output_len={} finished_call={}",
                                    node.sub_circuit_id,
                                    level_idx,
                                    summed_call_id,
                                    output_range.start().0,
                                    output_range.end().0,
                                    output_len,
                                    finished_call,
                                );
                                self.store_error_norm_summary_expr_batch_shared(
                                    output_range.gate_ids().zip(outputs.into_iter()),
                                    &output_positions,
                                    &remaining_use_count,
                                    &mut gate_exprs,
                                    &mut final_output_exprs,
                                );
                                debug!(
                                    "error-norm summary summed-sub-call store finished sub_circuit_id={} level={} summed_call_id={} output_chunk_start={} output_chunk_end={} gate_expr_entries={} elapsed_ms={}",
                                    node.sub_circuit_id,
                                    level_idx,
                                    summed_call_id,
                                    output_range.start().0,
                                    output_range.end().0,
                                    gate_exprs.live_entries(),
                                    store_start.elapsed().as_millis(),
                                );
                            }
                        },
                    );
                }
            }
            // debug!(
            //     "error-norm summary level summed-sub-call preparation finished sub_circuit_id={}
            // level={} summed_sub_calls={} elapsed_ms={}",     node.sub_circuit_id,
            //     level_idx,
            //     prepared_summed_call_count,
            //     prepare_summed_sub_calls_start.elapsed().as_millis(),
            // );

            // debug!(
            //     "error-norm summary level finished sub_circuit_id={} level={} gate_exprs={}
            // remaining_use_entries={} total_elapsed_ms={}",     node.sub_circuit_id,
            //     level_idx,
            //     gate_exprs.live_entries(),
            //     remaining_use_count.len(),
            //     level_start.elapsed().as_millis()
            // );
        }

        // debug!(
        //     "error-norm summary final outputs start sub_circuit_id={} outputs={} unresolved={}",
        //     node.sub_circuit_id,
        //     output_gate_ids.len(),
        //     final_output_exprs.iter().filter(|expr| expr.is_none()).count()
        // );
        let outputs = final_output_exprs
            .into_par_iter()
            .zip(output_gate_ids.par_iter().copied())
            .map(|(output_expr, gate_id)| match output_expr {
                Some(output) => output,
                None => self.clone_error_norm_summary_expr_for_summary_gate_direct(
                    gate_id,
                    &gate_exprs,
                    &input_gate_positions,
                    input_plaintext_norms,
                    one_error,
                    plt_evaluator,
                    slot_transfer_evaluator,
                    summary_cache,
                    node,
                ),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        // debug!(
        //     "error-norm summary build finished sub_circuit_id={} outputs={} elapsed_ms={}",
        //     node.sub_circuit_id,
        //     output_gate_ids.len(),
        //     final_output_start.elapsed().as_millis(),
        // );
        ErrorNormSubCircuitSummary { outputs }
    }

    fn build_prepared_error_norm_sub_circuit_summaries<P: AffinePltEvaluator>(
        &self,
        requests: Vec<ErrorNormPreparedSubCircuitSummaryRequest>,
        one_error: &ErrorNorm,
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
        summary_cache: &ErrorNormSubCircuitSummaryCache,
    ) -> Vec<Arc<ErrorNormSubCircuitSummary>> {
        if requests.is_empty() {
            return Vec::new();
        }

        let mut unique_request_index_by_key =
            HashMap::<ErrorNormSubCircuitSummaryCacheKey, usize>::new();
        let mut request_unique_indices = Vec::with_capacity(requests.len());
        let mut unique_requests = Vec::<(usize, Arc<[SubCircuitParamValue]>, Vec<PolyNorm>)>::new();
        for request in requests {
            let materialized_input_plaintext_norms = request.input_plaintext_profile.materialize();
            let sub_circuit = self.registered_sub_circuit_ref(request.sub_circuit_id);
            let cache_key = error_norm_sub_circuit_summary_cache_key(
                Arc::as_ptr(&sub_circuit) as usize,
                request.sub_circuit_id,
                &materialized_input_plaintext_norms,
            );
            let unique_idx =
                if let Some(&existing_idx) = unique_request_index_by_key.get(&cache_key) {
                    existing_idx
                } else {
                    let next_idx = unique_requests.len();
                    unique_request_index_by_key.insert(cache_key, next_idx);
                    unique_requests.push((
                        request.sub_circuit_id,
                        request.param_bindings,
                        materialized_input_plaintext_norms,
                    ));
                    next_idx
                };
            request_unique_indices.push(unique_idx);
        }

        let _total_requests = request_unique_indices.len();
        let _unique_request_count = unique_requests.len();
        // debug!(
        //     "error-norm sub-circuit summary request batch dedup finished requests={}
        // unique_requests={} duplicate_requests={} cached_unique_requests={}
        // uncached_unique_requests={} dedup_elapsed_ms={} cache_entries={}",
        //     total_requests,
        //     unique_request_count,
        //     duplicate_request_count,
        //     cached_unique_count,
        //     unique_request_count.saturating_sub(cached_unique_count),
        //     dedup_start.elapsed().as_millis(),
        //     summary_cache.len(),
        // );

        let build_progress = Arc::new(AtomicUsize::new(0));
        let mut registry = ErrorNormSummaryRegistry::default();
        let mut visiting = HashSet::<ErrorNormSummaryBuildKey>::new();

        for (_unique_idx, (sub_circuit_id, param_bindings, materialized_input_plaintext_norms)) in
            unique_requests.iter().enumerate()
        {
            let _profile_inputs = materialized_input_plaintext_norms.len();
            let _profile_max_bits =
                error_norm_summary_profile_max_bits(materialized_input_plaintext_norms);
            let sub_circuit = self.registered_sub_circuit_ref(*sub_circuit_id);
            let cache_key = error_norm_sub_circuit_summary_cache_key(
                Arc::as_ptr(&sub_circuit) as usize,
                *sub_circuit_id,
                materialized_input_plaintext_norms,
            );
            if summary_cache.contains_key(&cache_key) {
                // debug!(
                //     "error-norm sub-circuit summary unique build skip (cached) unique_idx={}
                // unique_total={} sub_circuit_id={} inputs={} max_input_bits={}",
                //     unique_idx + 1,
                //     unique_request_count,
                //     sub_circuit_id,
                //     profile_inputs,
                //     profile_max_bits,
                // );
                continue;
            }
            // debug!(
            //     "error-norm sub-circuit summary unique build start unique_idx={} unique_total={}
            // sub_circuit_id={} inputs={} max_input_bits={}",     unique_idx + 1,
            //     unique_request_count,
            //     sub_circuit_id,
            //     profile_inputs,
            //     profile_max_bits,
            // );
            self.register_error_norm_summary_node(
                *sub_circuit_id,
                Arc::from(materialized_input_plaintext_norms.clone()),
                Arc::clone(param_bindings),
                one_error,
                plt_evaluator,
                slot_transfer_evaluator,
                &mut registry,
                &mut visiting,
            );
            let _completed = build_progress.fetch_add(1, Ordering::Relaxed) + 1;
            // debug!(
            //     "error-norm sub-circuit summary unique build registered completed={}
            // unique_total={} sub_circuit_id={} inputs={} max_input_bits={} elapsed_ms={}",
            //     completed,
            //     unique_request_count,
            //     sub_circuit_id,
            //     profile_inputs,
            //     profile_max_bits,
            //     unique_build_start.elapsed().as_millis(),
            // );
        }

        request_unique_indices
            .into_iter()
            .map(|unique_idx| {
                let (sub_circuit_id, _param_bindings, materialized_input_plaintext_norms) =
                    &unique_requests[unique_idx];
                let sub_circuit = self.registered_sub_circuit_ref(*sub_circuit_id);
                let cache_key = error_norm_sub_circuit_summary_cache_key(
                    Arc::as_ptr(&sub_circuit) as usize,
                    *sub_circuit_id,
                    materialized_input_plaintext_norms,
                );
                self.ensure_error_norm_summary_built(
                    &cache_key,
                    &registry,
                    one_error,
                    plt_evaluator,
                    slot_transfer_evaluator,
                    summary_cache,
                )
            })
            .collect()
    }

    fn build_grouped_summed_sub_circuit_summaries_direct<P: AffinePltEvaluator>(
        &self,
        sub_circuit_id: usize,
        call_input_set_ids: &[usize],
        call_binding_set_ids: &[usize],
        gate_exprs: &ErrorNormExprStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_plaintext_norms: &[PolyNorm],
        one_error: &ErrorNorm,
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
        summary_cache: &ErrorNormSubCircuitSummaryCache,
        node: &ErrorNormSummaryNode,
    ) -> Vec<(Arc<ErrorNormSubCircuitSummary>, Vec<(usize, usize)>)> {
        if call_input_set_ids.is_empty() {
            return Vec::new();
        }
        assert_eq!(
            call_input_set_ids.len(),
            call_binding_set_ids.len(),
            "summed sub-circuit grouping requires matching input and binding counts"
        );
        let sub_circuit = self.registered_sub_circuit_ref(sub_circuit_id);
        let circuit_key = Arc::as_ptr(&sub_circuit) as usize;
        let mut grouped_idx_by_key = HashMap::<ErrorNormSubCircuitSummaryCacheKey, usize>::new();
        let mut grouped_requests = Vec::<ErrorNormPreparedSubCircuitSummaryRequest>::new();
        let mut grouped_input_set_counts = Vec::<HashMap<usize, usize>>::new();
        let grouped_entries = call_input_set_ids
            .par_iter()
            .zip(call_binding_set_ids.par_iter())
            .map(|(&input_set_id, &binding_set_id)| {
                let input_ids = self.input_set(input_set_id);
                let input_plaintext_profile = self
                    .collect_error_norm_summary_plaintext_norms_for_input_set_direct(
                        input_ids.as_ref(),
                        gate_exprs,
                        input_gate_positions,
                        input_plaintext_norms,
                        one_error,
                        plt_evaluator,
                        slot_transfer_evaluator,
                        summary_cache,
                        node,
                    );
                let cache_key = error_norm_sub_circuit_summary_cache_key(
                    circuit_key,
                    sub_circuit_id,
                    &input_plaintext_profile,
                );
                (
                    cache_key,
                    binding_set_id,
                    input_set_id,
                    ErrorNormInputPlaintextProfile::flat_from_vec(input_plaintext_profile),
                )
            })
            .collect::<Vec<_>>();
        for (cache_key, binding_set_id, input_set_id, input_plaintext_profile) in grouped_entries {
            let grouped_idx = if let Some(&existing_idx) = grouped_idx_by_key.get(&cache_key) {
                existing_idx
            } else {
                let next_idx = grouped_requests.len();
                grouped_idx_by_key.insert(cache_key, next_idx);
                grouped_requests.push(ErrorNormPreparedSubCircuitSummaryRequest {
                    sub_circuit_id,
                    param_bindings: self.binding_set(binding_set_id),
                    input_plaintext_profile,
                });
                grouped_input_set_counts.push(HashMap::new());
                next_idx
            };
            *grouped_input_set_counts[grouped_idx].entry(input_set_id).or_insert(0) += 1;
        }

        self.build_prepared_error_norm_sub_circuit_summaries(
            grouped_requests,
            one_error,
            plt_evaluator,
            slot_transfer_evaluator,
            summary_cache,
        )
        .into_iter()
        .zip(grouped_input_set_counts)
        .map(|(summary, input_set_counts)| (summary, input_set_counts.into_iter().collect()))
        .collect()
    }

    fn ensure_error_norm_summary_built<P: AffinePltEvaluator>(
        &self,
        cache_key: &ErrorNormSubCircuitSummaryCacheKey,
        registry: &ErrorNormSummaryRegistry,
        one_error: &ErrorNorm,
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
        summary_cache: &ErrorNormSubCircuitSummaryCache,
    ) -> Arc<ErrorNormSubCircuitSummary> {
        if let Some(summary) = summary_cache.get(cache_key) {
            return Arc::clone(summary.value());
        }
        let node = registry
            .nodes
            .get(cache_key)
            .unwrap_or_else(|| panic!("summary node missing for cache key"));
        for child_key in node.direct_call_keys.values() {
            self.ensure_error_norm_summary_built(
                child_key,
                registry,
                one_error,
                plt_evaluator,
                slot_transfer_evaluator,
                summary_cache,
            );
        }
        let node_circuit = Arc::clone(&node.circuit);
        let summary = Arc::new(node_circuit.as_ref().build_sub_circuit_summary_from_node(
            node,
            one_error,
            plt_evaluator,
            slot_transfer_evaluator,
            summary_cache,
        ));
        summary_cache.insert(cache_key.clone(), Arc::clone(&summary));
        summary
    }

    fn clone_error_norm_summary_expr_for_summary_gate_direct<P: AffinePltEvaluator>(
        &self,
        gate_id: GateId,
        gate_exprs: &ErrorNormExprStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_plaintext_norms: &[PolyNorm],
        one_error: &ErrorNorm,
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
        summary_cache: &ErrorNormSubCircuitSummaryCache,
        node: &ErrorNormSummaryNode,
    ) -> Arc<ErrorNormSummaryExpr> {
        if let Some(output) = gate_exprs.get(gate_id) {
            return output;
        }
        if let Some(&input_idx) = input_gate_positions.get(&gate_id) {
            return Arc::new(ErrorNormSummaryExpr::input_with_plaintext_norm(
                input_idx,
                input_plaintext_norms[input_idx].clone(),
            ));
        }
        match self.gate(gate_id).gate_type.clone() {
            PolyGateType::SubCircuitOutput { call_id, output_idx, .. } => self
                .with_sub_circuit_call_by_id(
                    call_id,
                    |_sub_circuit_id, _param_bindings, shared_prefix, suffix, _output_gate_ids| {
                        let child_key = node.direct_call_keys.get(&call_id).unwrap_or_else(|| {
                            panic!(
                                "missing direct call key for call_id {} in sub_circuit_id {}",
                                call_id, node.sub_circuit_id
                            )
                        });
                        let summary = summary_cache
                            .get(child_key)
                            .unwrap_or_else(|| {
                                panic!(
                                    "missing summary for direct call {} in sub_circuit_id {}",
                                    call_id, node.sub_circuit_id
                                )
                            })
                            .value()
                            .clone();
                        let actual_inputs =
                            self.collect_error_norm_summary_exprs_for_inputs_direct(
                                shared_prefix,
                                suffix,
                                gate_exprs,
                                input_gate_positions,
                                input_plaintext_norms,
                                one_error,
                                plt_evaluator,
                                slot_transfer_evaluator,
                                summary_cache,
                                node,
                            );
                        summary.substitute_output_shared(output_idx, &actual_inputs)
                    },
                ),
            PolyGateType::SummedSubCircuitOutput { summed_call_id, output_idx, .. } => self
                .with_summed_sub_circuit_call_by_id(
                    summed_call_id,
                    |sub_circuit_id, call_input_set_ids, call_binding_set_ids, _output_gate_ids| {
                        let accumulated = self
                            .build_grouped_summed_sub_circuit_summaries_direct(
                                sub_circuit_id,
                                call_input_set_ids,
                                call_binding_set_ids,
                                gate_exprs,
                                input_gate_positions,
                                input_plaintext_norms,
                                one_error,
                                plt_evaluator,
                                slot_transfer_evaluator,
                                summary_cache,
                                node,
                            )
                            .into_par_iter()
                            .map(|(summary, input_set_counts)| {
                                input_set_counts
                                    .into_par_iter()
                                    .map(|(input_set_id, call_count)| {
                                        let input_ids = self.input_set(input_set_id);
                                        let actual_inputs = self
                                            .collect_error_norm_summary_exprs_for_input_set_direct(
                                                input_ids.as_ref(),
                                                gate_exprs,
                                                input_gate_positions,
                                                input_plaintext_norms,
                                                one_error,
                                                plt_evaluator,
                                                slot_transfer_evaluator,
                                                summary_cache,
                                                node,
                                            );
                                        let output =
                                            summary.substitute_output_shared(output_idx, &actual_inputs);
                                        if call_count > 1 {
                                            output.as_ref().scale_bound(&BigDecimal::from(
                                                u64::try_from(call_count).unwrap_or_else(|_| {
                                                    panic!(
                                                        "summary-expression multiplier {} exceeds u64",
                                                        call_count
                                                    )
                                                }),
                                            ))
                                        } else {
                                            output.as_ref().clone()
                                        }
                                    })
                                    .reduce_with(|mut left, right| {
                                        left.add_assign_bound(right);
                                        left
                                    })
                                    .unwrap_or_else(|| {
                                        panic!(
                                            "error-norm summary summed sub-circuit call {} has no inner calls",
                                            summed_call_id
                                        )
                                    })
                            })
                            .reduce_with(|mut left, right| {
                                left.add_assign_bound(right);
                                left
                            });
                        Arc::new(accumulated.unwrap_or_else(|| {
                            panic!(
                                "error-norm summary summed sub-circuit call {} has no inner calls",
                                summed_call_id
                            )
                        }))
                    },
                ),
            _ => panic!("error-norm expression missing for gate {gate_id}"),
        }
    }

    fn collect_error_norm_summary_exprs_for_inputs_direct<P: AffinePltEvaluator>(
        &self,
        shared_prefix: &[BatchedWire],
        suffix: &[BatchedWire],
        gate_exprs: &ErrorNormExprStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_plaintext_norms: &[PolyNorm],
        one_error: &ErrorNorm,
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
        summary_cache: &ErrorNormSubCircuitSummaryCache,
        node: &ErrorNormSummaryNode,
    ) -> Vec<Arc<ErrorNormSummaryExpr>> {
        let mut prefix_exprs = shared_prefix
            .iter()
            .flat_map(|batch| batch.gate_ids())
            .map(|input_id| {
                self.clone_error_norm_summary_expr_for_summary_gate_direct(
                    input_id,
                    gate_exprs,
                    input_gate_positions,
                    input_plaintext_norms,
                    one_error,
                    plt_evaluator,
                    slot_transfer_evaluator,
                    summary_cache,
                    node,
                )
            })
            .collect::<Vec<_>>();
        let suffix_exprs = suffix
            .iter()
            .flat_map(|batch| batch.gate_ids())
            .map(|input_id| {
                self.clone_error_norm_summary_expr_for_summary_gate_direct(
                    input_id,
                    gate_exprs,
                    input_gate_positions,
                    input_plaintext_norms,
                    one_error,
                    plt_evaluator,
                    slot_transfer_evaluator,
                    summary_cache,
                    node,
                )
            })
            .collect::<Vec<_>>();
        prefix_exprs.extend(suffix_exprs);
        prefix_exprs
    }

    fn collect_error_norm_summary_exprs_for_input_set_direct<P: AffinePltEvaluator>(
        &self,
        input_ids: &[BatchedWire],
        gate_exprs: &ErrorNormExprStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_plaintext_norms: &[PolyNorm],
        one_error: &ErrorNorm,
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
        summary_cache: &ErrorNormSubCircuitSummaryCache,
        node: &ErrorNormSummaryNode,
    ) -> Vec<Arc<ErrorNormSummaryExpr>> {
        input_ids
            .iter()
            .flat_map(|batch| batch.gate_ids())
            .map(|input_id| {
                self.clone_error_norm_summary_expr_for_summary_gate_direct(
                    input_id,
                    gate_exprs,
                    input_gate_positions,
                    input_plaintext_norms,
                    one_error,
                    plt_evaluator,
                    slot_transfer_evaluator,
                    summary_cache,
                    node,
                )
            })
            .collect::<Vec<_>>()
    }

    fn clone_error_norm_value_for_gate(
        &self,
        gate_id: GateId,
        wires: &ErrorNormWireStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_values: &[Option<Arc<ErrorNorm>>],
    ) -> ErrorNorm {
        if let Some(value) = wires.get(gate_id) {
            return value.as_ref().clone();
        }
        if let Some(&input_idx) = input_gate_positions.get(&gate_id) {
            return input_values[input_idx]
                .as_ref()
                .unwrap_or_else(|| panic!("error-norm value missing for input gate {gate_id}"))
                .as_ref()
                .clone();
        }
        panic!("error-norm value missing for gate {gate_id}")
    }

    fn clone_error_norm_matrix_norm_for_gate(
        &self,
        gate_id: GateId,
        wires: &ErrorNormWireStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_values: &[Option<Arc<ErrorNorm>>],
    ) -> PolyMatrixNorm {
        if let Some(value) = wires.get(gate_id) {
            return value.matrix_norm.clone();
        }
        if let Some(&input_idx) = input_gate_positions.get(&gate_id) {
            return input_values[input_idx]
                .as_ref()
                .unwrap_or_else(|| {
                    panic!("error-norm matrix norm missing for input gate {gate_id}")
                })
                .matrix_norm
                .clone();
        }
        panic!("error-norm matrix norm missing for gate {gate_id}")
    }

    fn clone_error_norm_plaintext_norm_for_value_gate(
        &self,
        gate_id: GateId,
        wires: &ErrorNormWireStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_values: &[Option<Arc<ErrorNorm>>],
    ) -> PolyNorm {
        if let Some(value) = wires.get(gate_id) {
            return value.plaintext_norm.clone();
        }
        if let Some(&input_idx) = input_gate_positions.get(&gate_id) {
            return input_values[input_idx]
                .as_ref()
                .unwrap_or_else(|| panic!("error-norm value missing for input gate {gate_id}"))
                .plaintext_norm
                .clone();
        }
        panic!("error-norm value missing for gate {gate_id}")
    }

    fn collect_error_norm_plaintext_norms_for_value_input_set(
        &self,
        input_ids: &[BatchedWire],
        wires: &ErrorNormWireStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_values: &[Option<Arc<ErrorNorm>>],
    ) -> Vec<PolyNorm> {
        input_ids
            .par_iter()
            .flat_map_iter(|batch| batch.gate_ids())
            .map(|input_id| {
                self.clone_error_norm_plaintext_norm_for_value_gate(
                    input_id,
                    wires,
                    input_gate_positions,
                    input_values,
                )
            })
            .collect::<Vec<_>>()
    }

    fn eval_max_error_norm<P: AffinePltEvaluator>(
        &self,
        one_error: ErrorNorm,
        inputs: Vec<ErrorNorm>,
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
    ) -> Vec<ErrorNorm> {
        let input_gate_ids = self.sorted_input_gate_ids();
        assert_eq!(
            input_gate_ids.len(),
            inputs.len(),
            "number of provided inputs must match circuit inputs"
        );
        let input_gate_positions = self.error_norm_input_gate_positions(&input_gate_ids);

        let output_gate_ids = self.output_gate_ids();
        let output_gate_ids_set = output_gate_ids.iter().copied().collect::<HashSet<_>>();
        let mut input_values =
            inputs.into_iter().map(|value| Some(Arc::new(value))).collect::<Vec<_>>();
        let mut wires = ErrorNormWireStore::default();
        wires.insert_single(GateId(0), one_error.clone());

        let execution_layers_start = Instant::now();
        let execution_layers = self.error_norm_execution_layers();
        debug!(
            "error-norm eval execution layers ready levels={} elapsed_ms={}",
            execution_layers.len(),
            execution_layers_start.elapsed().as_millis()
        );
        let remaining_use_count_start = Instant::now();
        let mut remaining_use_count = self.error_norm_remaining_use_count(&execution_layers);
        debug!(
            "error-norm eval remaining-use-count ready entries={} elapsed_ms={}",
            remaining_use_count.len(),
            remaining_use_count_start.elapsed().as_millis()
        );
        for (
            level_idx,
            ErrorNormExecutionLayer {
                regular_gate_ids,
                sub_circuit_call_ids,
                summed_sub_circuit_call_ids,
            },
        ) in execution_layers.into_iter().enumerate()
        {
            let (pure_regular_gate_ids, evaluator_regular_gate_ids) = regular_gate_ids
                .into_par_iter()
                .fold(
                    || (Vec::new(), Vec::new()),
                    |mut acc, gate_id| {
                        match &self.gate(gate_id).gate_type {
                            PolyGateType::Input => {
                                panic!("input gate {gate_id} should already be preloaded");
                            }
                            PolyGateType::Add |
                            PolyGateType::Sub |
                            PolyGateType::Mul |
                            PolyGateType::SmallScalarMul { .. } |
                            PolyGateType::LargeScalarMul { .. } => acc.0.push(gate_id),
                            PolyGateType::SlotTransfer { .. } | PolyGateType::PubLut { .. } => {
                                acc.1.push(gate_id)
                            }
                            PolyGateType::SubCircuitOutput { .. } |
                            PolyGateType::SummedSubCircuitOutput { .. } => {
                                unreachable!(
                                    "subcircuit outputs should not appear in regular execution layers"
                                )
                            }
                        }
                        acc
                    },
                )
                .reduce(
                    || (Vec::new(), Vec::new()),
                    |mut left, mut right| {
                        left.0.append(&mut right.0);
                        left.1.append(&mut right.1);
                        left
                    },
                );
            // debug!(
            //     "error-norm eval level ready level={} regular={} pure_regular={}
            // evaluator_regular={} sub_calls={} summed_sub_calls={} elapsed_ms=0",
            //     level_idx,
            //     pure_regular_gate_ids.len() + evaluator_regular_gate_ids.len(),
            //     pure_regular_gate_ids.len(),
            //     evaluator_regular_gate_ids.len(),
            //     sub_circuit_call_ids.len(),
            //     summed_sub_circuit_call_ids.len(),
            // );

            let pure_regular_results = pure_regular_gate_ids
                .par_iter()
                .map(|gate_id| {
                    (
                        *gate_id,
                        self.eval_pure_regular_error_norm_gate(
                            *gate_id,
                            &wires,
                            &input_gate_positions,
                            &input_values,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            let evaluator_regular_results = evaluator_regular_gate_ids
                .par_iter()
                .map(|gate_id| {
                    (
                        *gate_id,
                        self.eval_evaluator_regular_error_norm_gate(
                            *gate_id,
                            &one_error,
                            &wires,
                            &input_gate_positions,
                            &input_values,
                            plt_evaluator,
                            slot_transfer_evaluator,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            // debug!(
            //     "error-norm eval level evaluator regular finished level={} evaluator_regular={}
            // elapsed_ms={}",     level_idx,
            //     evaluator_regular_gate_ids.len(),
            //     evaluator_regular_start.elapsed().as_millis()
            // );

            self.store_error_norm_value_pairs(
                pure_regular_results.into_iter().chain(evaluator_regular_results.into_iter()),
                &output_gate_ids_set,
                &remaining_use_count,
                &mut wires,
            );
            for gate_id in pure_regular_gate_ids.iter().chain(evaluator_regular_gate_ids.iter()) {
                let gate = self.gate(*gate_id);
                self.release_error_norm_inputs(
                    gate.input_gates.iter().copied(),
                    &output_gate_ids_set,
                    &mut remaining_use_count,
                    &mut wires,
                    &input_gate_positions,
                    &mut input_values,
                );
            }

            let prepare_sub_calls_start = Instant::now();
            let mut prepared_sub_call_count = 0usize;
            for call_id_chunk in sub_circuit_call_ids.chunks(ERROR_NORM_CALL_COMMIT_BATCH_SIZE) {
                let mut shared_prefix_plaintext_norms = HashMap::<usize, Arc<[PolyNorm]>>::new();
                let prepared_profiles = call_id_chunk
                    .par_iter()
                    .map(|call_id| {
                        let shared_prefix_set_id =
                            self.sub_circuit_call_shared_prefix_set_id(*call_id);
                        self.with_sub_circuit_call_by_id(
                            *call_id,
                            |sub_circuit_id, param_bindings, _shared_prefix, suffix, _| {
                                let suffix_plaintext_norms = suffix
                                    .par_iter()
                                    .flat_map_iter(|batch| batch.gate_ids())
                                    .map(|input_id| {
                                        self.clone_error_norm_plaintext_norm_for_value_gate(
                                            input_id,
                                            &wires,
                                            &input_gate_positions,
                                            &input_values,
                                        )
                                    })
                                    .collect::<Vec<_>>();
                                (
                                    *call_id,
                                    sub_circuit_id,
                                    param_bindings,
                                    shared_prefix_set_id,
                                    suffix_plaintext_norms,
                                )
                            },
                        )
                    })
                    .collect::<Vec<_>>();
                let prepared_requests = prepared_profiles
                    .into_iter()
                    .map(
                        |(
                            call_id,
                            sub_circuit_id,
                            param_bindings,
                            shared_prefix_set_id,
                            suffix_plaintext_norms,
                        )| {
                            let input_plaintext_profile = if let Some(input_set_id) =
                                shared_prefix_set_id
                            {
                                let prefix_norms = shared_prefix_plaintext_norms
                                    .entry(input_set_id)
                                    .or_insert_with(|| {
                                        Arc::from(
                                            self.collect_error_norm_plaintext_norms_for_value_input_set(
                                                self.input_set(input_set_id).as_ref(),
                                                &wires,
                                                &input_gate_positions,
                                                &input_values,
                                            ),
                                        )
                                    })
                                    .clone();
                                ErrorNormInputPlaintextProfile::shared_prefix_from_parts(
                                    prefix_norms.as_ref(),
                                    suffix_plaintext_norms,
                                )
                            } else {
                                ErrorNormInputPlaintextProfile::flat_from_vec(
                                    suffix_plaintext_norms,
                                )
                            };
                            (
                                call_id,
                                ErrorNormPreparedSubCircuitSummaryRequest {
                                    sub_circuit_id,
                                    param_bindings,
                                    input_plaintext_profile,
                                },
                            )
                        },
                    )
                    .collect::<Vec<_>>();
                let processed_calls_before = prepared_sub_call_count;
                debug!(
                    "error-norm eval level sub-call batch build start level={} batch_calls={} processed_calls_before={} elapsed_ms={}",
                    level_idx,
                    prepared_requests.len(),
                    processed_calls_before,
                    prepare_sub_calls_start.elapsed().as_millis(),
                );
                let mut process_prepared_sub_circuit_calls =
                    |prepared_sub_circuit_calls: &[(usize, Arc<ErrorNormSubCircuitSummary>)],
                     call_batch_size: usize| {
                        prepared_sub_call_count += prepared_sub_circuit_calls.len();
                        for prepared_chunk in prepared_sub_circuit_calls.chunks(call_batch_size) {
                            let max_output_len = prepared_chunk
                                .iter()
                                .map(|(_, summary)| summary.output_len())
                                .max()
                                .unwrap_or(0);
                            // debug!(
                            //     "error-norm eval level sub-call output batch start level={}
                            // batch_calls={} processed_calls_before={} processed_calls_after={}
                            // output_max_len={} elapsed_ms={}",
                            //     level_idx,
                            //     prepared_chunk.len(),
                            //     processed_calls_before,
                            //     processed_calls_before + prepared_chunk.len(),
                            //     max_output_len,
                            //     prepare_sub_calls_start.elapsed().as_millis(),
                            // );
                            for chunk_start in
                                (0..max_output_len).step_by(ERROR_NORM_OUTPUT_BATCH_SIZE)
                            {
                                let prepared_outputs = prepared_chunk
                                    .par_iter()
                                    .filter_map(|(call_id, summary)| {
                                        if chunk_start >= summary.output_len() {
                                            return None;
                                        }
                                        self.with_sub_circuit_call_by_id(
                                            *call_id,
                                            |actual_sub_circuit_id,
                                             _param_bindings,
                                             shared_prefix,
                                             suffix,
                                             output_gate_ids| {
                                                let _ = actual_sub_circuit_id;
                                                assert_eq!(
                                                    output_gate_ids.len(),
                                                    summary.output_len(),
                                                    "error-norm sub-circuit output count mismatch for call {call_id}"
                                                );
                                                let output_range =
                                                    BatchedWire::from_batches(output_gate_ids.iter().copied());
                                                let chunk_end =
                                                    (chunk_start + ERROR_NORM_OUTPUT_BATCH_SIZE)
                                                        .min(output_range.len());
                                        // debug!(
                                        //     "error-norm eval sub-call output-eval start level={} call_id={} output_chunk_start={} output_chunk_end={} output_len={} summary_outputs={} expr_batch_size={}",
                                        //     level_idx,
                                        //     call_id,
                                        //     chunk_start,
                                        //     chunk_end,
                                        //     chunk_end - chunk_start,
                                        //     summary.output_len(),
                                        //     ERROR_NORM_EXPR_PAR_BATCH_SIZE,
                                        // );
                                        let outputs = summary.evaluate_output_range_with_shared_cache(
                                            chunk_start..chunk_end,
                                            &|input_idx| {
                                                let input_id =
                                                    resolve_shared_prefix_suffix_input_id(
                                                        shared_prefix,
                                                        suffix,
                                                        input_idx,
                                                    );
                                                self.clone_error_norm_matrix_norm_for_gate(
                                                    input_id,
                                                    &wires,
                                                    &input_gate_positions,
                                                    &input_values,
                                                )
                                            },
                                        );
                                        // debug!(
                                        //     "error-norm eval sub-call output-eval finished level={} call_id={} output_chunk_start={} output_chunk_end={} output_len={} elapsed_ms={}",
                                        //     level_idx,
                                        //     call_id,
                                        //     chunk_start,
                                        //     chunk_end,
                                        //     chunk_end - chunk_start,
                                        //     eval_start.elapsed().as_millis(),
                                        // );
                                        Some((
                                            *call_id,
                                            output_range.slice(chunk_start..chunk_end),
                                                    outputs,
                                                    chunk_end == output_range.len(),
                                                ))
                                            },
                                        )
                                    })
                                    .collect::<Vec<_>>();
                                for (call_id, output_range, outputs, finished_call) in
                                    prepared_outputs
                                {
                                    self.store_error_norm_value_batch(
                                        output_range,
                                        outputs,
                                        &output_gate_ids_set,
                                        &remaining_use_count,
                                        &mut wires,
                                    );
                                    if finished_call {
                                        self.with_sub_circuit_call_inputs_by_id(
                                            call_id,
                                            |shared_prefix, suffix| {
                                                self.release_error_norm_inputs(
                                                    iter_batched_wire_gates(shared_prefix),
                                                    &output_gate_ids_set,
                                                    &mut remaining_use_count,
                                                    &mut wires,
                                                    &input_gate_positions,
                                                    &mut input_values,
                                                );
                                                self.release_error_norm_inputs(
                                                    iter_batched_wire_gates(suffix),
                                                    &output_gate_ids_set,
                                                    &mut remaining_use_count,
                                                    &mut wires,
                                                    &input_gate_positions,
                                                    &mut input_values,
                                                );
                                            },
                                        );
                                    }
                                }
                            }
                        }
                    };
                let prepared_call_ids =
                    prepared_requests.iter().map(|(call_id, _)| *call_id).collect::<Vec<_>>();
                let summary_cache = ErrorNormSubCircuitSummaryCache::new();
                let prepared_sub_circuit_calls = prepared_call_ids
                    .into_iter()
                    .zip(
                        self.build_prepared_error_norm_sub_circuit_summaries(
                            prepared_requests
                                .into_iter()
                                .map(|(_, request)| request)
                                .collect::<Vec<_>>(),
                            &one_error,
                            plt_evaluator,
                            slot_transfer_evaluator,
                            &summary_cache,
                        ),
                    )
                    .collect::<Vec<_>>();
                debug!(
                    "error-norm eval level sub-call batch build finished level={} batch_calls={} processed_calls_before_commit={} elapsed_ms={}",
                    level_idx,
                    prepared_sub_circuit_calls.len(),
                    processed_calls_before,
                    prepare_sub_calls_start.elapsed().as_millis(),
                );
                process_prepared_sub_circuit_calls(
                    &prepared_sub_circuit_calls,
                    ERROR_NORM_CALL_COMMIT_BATCH_SIZE,
                );
            }
            // debug!(
            //     "error-norm eval level sub-call preparation finished level={} sub_calls={}
            // elapsed_ms={}",     level_idx,
            //     prepared_sub_call_count,
            //     prepare_sub_calls_start.elapsed().as_millis()
            // );

            let mut prepared_summed_call_count = 0usize;
            for summed_call_chunk in
                summed_sub_circuit_call_ids.chunks(ERROR_NORM_SUMMED_CALL_COMMIT_BATCH_SIZE)
            {
                for summed_call_id in summed_call_chunk {
                    self.with_summed_sub_circuit_call_by_id(
                        *summed_call_id,
                        |sub_circuit_id, call_input_set_ids, call_binding_set_ids, output_gate_ids| {
                            prepared_summed_call_count += 1;
                            let output_range =
                                BatchedWire::from_batches(output_gate_ids.iter().copied());
                            for chunk_start in
                                (0..output_range.len()).step_by(ERROR_NORM_OUTPUT_BATCH_SIZE)
                            {
                                let chunk_end = (chunk_start + ERROR_NORM_OUTPUT_BATCH_SIZE)
                                    .min(output_range.len());
                                let mut chunk_values: Option<Vec<ErrorNorm>> = None;
                                for batch_start in
                                    (0..call_input_set_ids.len()).step_by(ERROR_NORM_SUMMED_INNER_REDUCE_BATCH_SIZE)
                                {
                                    let batch_end = (batch_start +
                                        ERROR_NORM_SUMMED_INNER_REDUCE_BATCH_SIZE)
                                        .min(call_input_set_ids.len());
                                    let batch_requests = call_input_set_ids[batch_start..batch_end]
                                        .iter()
                                        .copied()
                                        .zip(call_binding_set_ids[batch_start..batch_end].iter().copied())
                                        .map(|(input_set_id, binding_set_id)| {
                                            let input_plaintext_norms = self
                                                .collect_error_norm_plaintext_norms_for_value_input_set(
                                                    self.input_set(input_set_id).as_ref(),
                                                    &wires,
                                                    &input_gate_positions,
                                                    &input_values,
                                                );
                                            ErrorNormPreparedSubCircuitSummaryRequest {
                                                sub_circuit_id,
                                                param_bindings: self.binding_set(binding_set_id),
                                                input_plaintext_profile:
                                                    ErrorNormInputPlaintextProfile::flat_from_vec(
                                                        input_plaintext_norms,
                                                    ),
                                            }
                                        })
                                        .collect::<Vec<_>>();
                                    let batch_summary_cache = ErrorNormSubCircuitSummaryCache::new();
                                    let batch_summaries =
                                        self.build_prepared_error_norm_sub_circuit_summaries(
                                            batch_requests,
                                            &one_error,
                                            plt_evaluator,
                                            slot_transfer_evaluator,
                                            &batch_summary_cache,
                                        );
                                    if batch_start == 0 {
                                        let first_summary = batch_summaries.first().unwrap_or_else(|| {
                                            panic!(
                                                "summed sub-circuit call requires at least one inner call"
                                            )
                                        });
                                        assert_eq!(
                                            output_gate_ids.len(),
                                            first_summary.output_len(),
                                            "error-norm summed sub-circuit output count mismatch for call {summed_call_id}"
                                        );
                                    }
                                    let partial_values = call_input_set_ids[batch_start..batch_end]
                                        .par_iter()
                                        .copied()
                                        .zip(batch_summaries.into_par_iter())
                                        .map(|(input_set_id, summary)| {
                                            let input_ids = self.input_set(input_set_id);
                                            summary.evaluate_output_range_with_shared_cache(
                                                chunk_start..chunk_end,
                                                &|input_idx| {
                                                    let input_id = resolve_input_set_gate_id(
                                                        input_ids.as_ref(),
                                                        input_idx,
                                                    );
                                                    self.clone_error_norm_matrix_norm_for_gate(
                                                        input_id,
                                                        &wires,
                                                        &input_gate_positions,
                                                        &input_values,
                                                    )
                                                },
                                            )
                                        })
                                        .reduce_with(|mut left, right| {
                                            for (left_value, right_value) in left.iter_mut().zip(right)
                                            {
                                                *left_value = &*left_value + &right_value;
                                            }
                                            left
                                        })
                                        .expect(
                                            "summed sub-circuit call requires at least one inner call",
                                        );
                                    if let Some(accumulated_values) = chunk_values.as_mut() {
                                        for (left_value, right_value) in
                                            accumulated_values.iter_mut().zip(partial_values)
                                        {
                                            *left_value = &*left_value + &right_value;
                                        }
                                    } else {
                                        chunk_values = Some(partial_values);
                                    }
                                }
                                let chunk_values = chunk_values.unwrap_or_else(|| {
                                    panic!("summed sub-circuit call requires at least one inner call")
                                });
                                let output_chunk_range = output_range.slice(chunk_start..chunk_end);
                                let finished_call = chunk_end == output_range.len();
                                self.store_error_norm_value_batch(
                                    output_chunk_range,
                                    chunk_values,
                                    &output_gate_ids_set,
                                    &remaining_use_count,
                                    &mut wires,
                                );
                                if finished_call {
                                    for input_set_id in call_input_set_ids {
                                        let input_ids = self.input_set(*input_set_id);
                                        self.release_error_norm_inputs(
                                            iter_batched_wire_gates(input_ids.as_ref()),
                                            &output_gate_ids_set,
                                            &mut remaining_use_count,
                                            &mut wires,
                                            &input_gate_positions,
                                            &mut input_values,
                                        );
                                    }
                                }
                            }
                        },
                    );
                }
            }
            // debug!(
            //     "error-norm eval level summed-sub-call preparation finished level={}
            // summed_sub_calls={} elapsed_ms={}",     level_idx,
            //     prepared_summed_call_count,
            //     prepare_summed_sub_calls_start.elapsed().as_millis()
            // );
            // debug!(
            //     "error-norm eval level finished level={} remaining_wires={} total_elapsed_ms={}",
            //     level_idx,
            //     wires.live_entries,
            //     level_start.elapsed().as_millis()
            // );
        }
        debug!(
            "error-norm eval execution finished outputs={} elapsed_ms={}",
            output_gate_ids.len(),
            execution_layers_start.elapsed().as_millis()
        );

        output_gate_ids
            .par_iter()
            .copied()
            .map(|gate_id| {
                self.clone_error_norm_value_for_gate(
                    gate_id,
                    &wires,
                    &input_gate_positions,
                    &input_values,
                )
            })
            .collect()
    }

    fn collect_error_norm_summary_plaintext_norms_for_input_set_direct<P: AffinePltEvaluator>(
        &self,
        input_ids: &[BatchedWire],
        gate_exprs: &ErrorNormExprStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_plaintext_norms: &[PolyNorm],
        one_error: &ErrorNorm,
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
        summary_cache: &ErrorNormSubCircuitSummaryCache,
        node: &ErrorNormSummaryNode,
    ) -> Vec<PolyNorm> {
        input_ids
            .par_iter()
            .map(|batch| {
                batch
                    .gate_ids()
                    .map(|input_id| {
                        self.clone_error_norm_summary_expr_for_summary_gate_direct(
                            input_id,
                            gate_exprs,
                            input_gate_positions,
                            input_plaintext_norms,
                            one_error,
                            plt_evaluator,
                            slot_transfer_evaluator,
                            summary_cache,
                            node,
                        )
                        .plaintext_norm
                        .clone()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .collect()
    }

    fn build_pure_regular_error_norm_expr(
        &self,
        gate_id: GateId,
        gate_exprs: &ErrorNormExprStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_plaintext_norms: &[PolyNorm],
        one_error: &ErrorNorm,
        param_bindings: &[SubCircuitParamValue],
        plt_evaluator: Option<&impl AffinePltEvaluator>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
        summary_cache: &ErrorNormSubCircuitSummaryCache,
        node: &ErrorNormSummaryNode,
    ) -> ErrorNormSummaryExpr {
        let gate = self.gate(gate_id);
        match &gate.gate_type {
            PolyGateType::Add | PolyGateType::Sub => {
                let left = self.clone_error_norm_summary_expr_for_summary_gate_direct(
                    gate.input_gates[0],
                    gate_exprs,
                    input_gate_positions,
                    input_plaintext_norms,
                    one_error,
                    plt_evaluator,
                    slot_transfer_evaluator,
                    summary_cache,
                    node,
                );
                let right = self.clone_error_norm_summary_expr_for_summary_gate_direct(
                    gate.input_gates[1],
                    gate_exprs,
                    input_gate_positions,
                    input_plaintext_norms,
                    one_error,
                    plt_evaluator,
                    slot_transfer_evaluator,
                    summary_cache,
                    node,
                );
                left.as_ref().add_bound(right.as_ref())
            }
            PolyGateType::Mul => {
                let left = self.clone_error_norm_summary_expr_for_summary_gate_direct(
                    gate.input_gates[0],
                    gate_exprs,
                    input_gate_positions,
                    input_plaintext_norms,
                    one_error,
                    plt_evaluator,
                    slot_transfer_evaluator,
                    summary_cache,
                    node,
                );
                let right = self.clone_error_norm_summary_expr_for_summary_gate_direct(
                    gate.input_gates[1],
                    gate_exprs,
                    input_gate_positions,
                    input_plaintext_norms,
                    one_error,
                    plt_evaluator,
                    slot_transfer_evaluator,
                    summary_cache,
                    node,
                );
                left.as_ref().mul_bound(right.as_ref())
            }
            PolyGateType::SmallScalarMul { scalar } => {
                let scalar = scalar.resolve_small_scalar(param_bindings);
                let input = self.clone_error_norm_summary_expr_for_summary_gate_direct(
                    gate.input_gates[0],
                    gate_exprs,
                    input_gate_positions,
                    input_plaintext_norms,
                    one_error,
                    plt_evaluator,
                    slot_transfer_evaluator,
                    summary_cache,
                    node,
                );
                input.as_ref().small_scalar_mul_bound(scalar)
            }
            PolyGateType::LargeScalarMul { scalar } => {
                let scalar = scalar.resolve_large_scalar(param_bindings);
                let input = self.clone_error_norm_summary_expr_for_summary_gate_direct(
                    gate.input_gates[0],
                    gate_exprs,
                    input_gate_positions,
                    input_plaintext_norms,
                    one_error,
                    plt_evaluator,
                    slot_transfer_evaluator,
                    summary_cache,
                    node,
                );
                input.as_ref().large_scalar_mul_bound(scalar)
            }
            _ => panic!("gate {gate_id} is not pure in error-norm summary eval"),
        }
    }

    fn build_evaluator_regular_error_norm_expr<P: AffinePltEvaluator>(
        &self,
        gate_id: GateId,
        gate_exprs: &ErrorNormExprStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_plaintext_norms: &[PolyNorm],
        one_error: &ErrorNorm,
        param_bindings: &[SubCircuitParamValue],
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
        summary_cache: &ErrorNormSubCircuitSummaryCache,
        node: &ErrorNormSummaryNode,
    ) -> ErrorNormSummaryExpr {
        let gate = self.gate(gate_id);
        match &gate.gate_type {
            PolyGateType::SlotTransfer { src_slots } => {
                let src_slots = src_slots.resolve_slot_transfer(param_bindings);
                let input = self.clone_error_norm_summary_expr_for_summary_gate_direct(
                    gate.input_gates[0],
                    gate_exprs,
                    input_gate_positions,
                    input_plaintext_norms,
                    one_error,
                    plt_evaluator,
                    slot_transfer_evaluator,
                    summary_cache,
                    node,
                );
                slot_transfer_evaluator
                    .expect("slot transfer evaluator missing")
                    .slot_transfer_affine(input.as_ref(), src_slots.as_ref(), gate_id)
            }
            PolyGateType::PubLut { lut_id } => {
                let lut_id = lut_id.resolve_public_lookup(param_bindings);
                let input = self.clone_error_norm_summary_expr_for_summary_gate_direct(
                    gate.input_gates[0],
                    gate_exprs,
                    input_gate_positions,
                    input_plaintext_norms,
                    one_error,
                    plt_evaluator,
                    slot_transfer_evaluator,
                    summary_cache,
                    node,
                );
                let lookup = self.lookup_table(lut_id);
                plt_evaluator.expect("public lookup evaluator missing").public_lookup_affine(
                    input.as_ref(),
                    lookup.as_ref(),
                    gate_id,
                    lut_id,
                )
            }
            _ => self.build_pure_regular_error_norm_expr(
                gate_id,
                gate_exprs,
                input_gate_positions,
                input_plaintext_norms,
                one_error,
                param_bindings,
                plt_evaluator,
                slot_transfer_evaluator,
                summary_cache,
                node,
            ),
        }
    }

    fn error_norm_remaining_use_count(
        &self,
        execution_layers: &[ErrorNormExecutionLayer],
    ) -> BatchedGateUseCounts {
        let mut counts = vec![0u32; self.num_gates() + 1];
        let mut increment = |gate_id: GateId| {
            counts[gate_id.0] += 1;
        };
        for layer in execution_layers {
            for gate_id in &layer.regular_gate_ids {
                let gate = self.gate(*gate_id);
                for input_id in gate.input_gates.iter().copied() {
                    increment(input_id);
                }
            }
            for call_id in &layer.sub_circuit_call_ids {
                self.with_sub_circuit_call_inputs_by_id(*call_id, |shared_prefix, suffix| {
                    for input_id in iter_batched_wire_gates(shared_prefix) {
                        increment(input_id);
                    }
                    for input_id in iter_batched_wire_gates(suffix) {
                        increment(input_id);
                    }
                });
            }
            for summed_call_id in &layer.summed_sub_circuit_call_ids {
                self.for_each_summed_sub_circuit_call_input(*summed_call_id, |input_id| {
                    increment(input_id);
                });
            }
        }
        BatchedGateUseCounts::from_dense_counts(counts)
    }

    fn release_error_norm_inputs<I>(
        &self,
        input_ids: I,
        output_gate_ids_set: &HashSet<GateId>,
        remaining_use_count: &mut BatchedGateUseCounts,
        wires: &mut ErrorNormWireStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_values: &mut [Option<Arc<ErrorNorm>>],
    ) where
        I: IntoIterator<Item = GateId>,
    {
        for input_id in input_ids {
            if output_gate_ids_set.contains(&input_id) {
                continue;
            }
            if !remaining_use_count.contains(input_id) {
                continue;
            }
            if remaining_use_count.decrement(input_id) {
                if let Some(&input_idx) = input_gate_positions.get(&input_id) {
                    input_values[input_idx] = None;
                } else {
                    let _ = wires.take(input_id);
                }
            }
        }
    }

    fn store_error_norm_value_batch(
        &self,
        gate_range: BatchedWire,
        values: Vec<ErrorNorm>,
        output_gate_ids_set: &HashSet<GateId>,
        remaining_use_count: &BatchedGateUseCounts,
        wires: &mut ErrorNormWireStore,
    ) {
        let mut keep_start: Option<GateId> = None;
        let mut keep_values = Vec::new();
        let flush = |keep_start: &mut Option<GateId>,
                     keep_values: &mut Vec<ErrorNorm>,
                     wires: &mut ErrorNormWireStore| {
            if let Some(start) = keep_start.take() {
                let range = BatchedWire::from_start_len(start, keep_values.len());
                wires.insert_batch(range, std::mem::take(keep_values));
            }
        };
        for (offset, value) in values.into_iter().enumerate() {
            let gate_id = GateId(gate_range.start().0 + offset);
            let keep_in_wires =
                remaining_use_count.contains(gate_id) || output_gate_ids_set.contains(&gate_id);
            if keep_in_wires {
                if keep_start.is_none() {
                    keep_start = Some(gate_id);
                }
                keep_values.push(value);
            } else {
                flush(&mut keep_start, &mut keep_values, wires);
            }
        }
        flush(&mut keep_start, &mut keep_values, wires);
    }

    fn store_error_norm_value_pairs<I>(
        &self,
        gate_values: I,
        output_gate_ids_set: &HashSet<GateId>,
        remaining_use_count: &BatchedGateUseCounts,
        wires: &mut ErrorNormWireStore,
    ) where
        I: IntoIterator<Item = (GateId, ErrorNorm)>,
    {
        let mut keep_start: Option<GateId> = None;
        let mut keep_values = Vec::new();
        let flush = |keep_start: &mut Option<GateId>,
                     keep_values: &mut Vec<ErrorNorm>,
                     wires: &mut ErrorNormWireStore| {
            if let Some(start) = keep_start.take() {
                let range = BatchedWire::from_start_len(start, keep_values.len());
                wires.insert_batch(range, std::mem::take(keep_values));
            }
        };
        for (gate_id, value) in gate_values {
            let keep_in_wires =
                remaining_use_count.contains(gate_id) || output_gate_ids_set.contains(&gate_id);
            if keep_in_wires {
                match keep_start {
                    Some(start) if gate_id.0 != start.0 + keep_values.len() => {
                        flush(&mut keep_start, &mut keep_values, wires);
                        keep_start = Some(gate_id);
                    }
                    None => keep_start = Some(gate_id),
                    _ => {}
                }
                keep_values.push(value);
            } else {
                flush(&mut keep_start, &mut keep_values, wires);
            }
        }
        flush(&mut keep_start, &mut keep_values, wires);
    }

    fn release_error_norm_summary_inputs<I>(
        &self,
        input_ids: I,
        output_gate_ids_set: &HashSet<GateId>,
        remaining_use_count: &mut BatchedGateUseCounts,
        gate_exprs: &mut ErrorNormExprStore,
    ) where
        I: IntoIterator<Item = GateId>,
    {
        for input_id in input_ids {
            if output_gate_ids_set.contains(&input_id) {
                continue;
            }
            if !remaining_use_count.contains(input_id) {
                continue;
            }
            if remaining_use_count.decrement(input_id) {
                let _ = gate_exprs.take(input_id);
            }
        }
    }

    fn collect_error_norm_summary_release_counts_for_direct_inputs(
        &self,
        shared_prefix: &[BatchedWire],
        suffix: &[BatchedWire],
        output_gate_ids_set: &HashSet<GateId>,
        remaining_use_count: &BatchedGateUseCounts,
    ) -> Vec<(GateId, u32)> {
        let mut release_counts = HashMap::<GateId, u32>::new();
        for input_id in
            iter_batched_wire_gates(shared_prefix).chain(iter_batched_wire_gates(suffix))
        {
            if output_gate_ids_set.contains(&input_id) {
                continue;
            }
            if !remaining_use_count.contains(input_id) {
                continue;
            }
            *release_counts.entry(input_id).or_insert(0) += 1;
        }
        let mut release_counts = release_counts.into_iter().collect::<Vec<_>>();
        release_counts.sort_unstable_by_key(|(gate_id, _)| *gate_id);
        release_counts
    }

    fn release_error_norm_summary_inputs_batched<I>(
        &self,
        release_counts: I,
        remaining_use_count: &mut BatchedGateUseCounts,
        gate_exprs: &mut ErrorNormExprStore,
    ) where
        I: IntoIterator<Item = (GateId, u32)>,
    {
        for (gate_id, release_count) in release_counts {
            if remaining_use_count.decrement_by(gate_id, release_count) {
                let _ = gate_exprs.take(gate_id);
            }
        }
    }

    fn store_error_norm_summary_expr_batch<I>(
        &self,
        gate_expr_pairs: I,
        output_positions: &HashMap<GateId, Vec<usize>>,
        remaining_use_count: &BatchedGateUseCounts,
        gate_exprs: &mut ErrorNormExprStore,
        final_output_exprs: &mut [Option<Arc<ErrorNormSummaryExpr>>],
    ) where
        I: IntoIterator<Item = (GateId, ErrorNormSummaryExpr)>,
    {
        let mut keep_start: Option<GateId> = None;
        let mut keep_exprs = Vec::new();
        let flush = |keep_start: &mut Option<GateId>,
                     keep_exprs: &mut Vec<Arc<ErrorNormSummaryExpr>>,
                     gate_exprs: &mut ErrorNormExprStore| {
            if let Some(start) = keep_start.take() {
                let range = BatchedWire::from_start_len(start, keep_exprs.len());
                gate_exprs.insert_batch_shared(range, std::mem::take(keep_exprs));
            }
        };
        for (gate_id, expr) in gate_expr_pairs {
            let expr = Arc::new(expr);
            let output_idxs = output_positions.get(&gate_id);
            let is_output = output_idxs.is_some();
            let keep_in_gate_exprs = remaining_use_count.contains(gate_id) || is_output;
            if keep_in_gate_exprs {
                match keep_start {
                    Some(start) if gate_id.0 != start.0 + keep_exprs.len() => {
                        flush(&mut keep_start, &mut keep_exprs, gate_exprs);
                        keep_start = Some(gate_id);
                    }
                    None => keep_start = Some(gate_id),
                    _ => {}
                }
                keep_exprs.push(expr.clone());
            } else {
                flush(&mut keep_start, &mut keep_exprs, gate_exprs);
            }
            if let Some(output_idxs) = output_idxs {
                for &output_idx in output_idxs {
                    final_output_exprs[output_idx] = Some(expr.clone());
                }
            }
        }
        flush(&mut keep_start, &mut keep_exprs, gate_exprs);
    }

    fn store_error_norm_summary_expr_batch_shared<I>(
        &self,
        gate_expr_pairs: I,
        output_positions: &HashMap<GateId, Vec<usize>>,
        remaining_use_count: &BatchedGateUseCounts,
        gate_exprs: &mut ErrorNormExprStore,
        final_output_exprs: &mut [Option<Arc<ErrorNormSummaryExpr>>],
    ) where
        I: IntoIterator<Item = (GateId, Arc<ErrorNormSummaryExpr>)>,
    {
        let mut keep_start: Option<GateId> = None;
        let mut keep_exprs = Vec::new();
        let flush = |keep_start: &mut Option<GateId>,
                     keep_exprs: &mut Vec<Arc<ErrorNormSummaryExpr>>,
                     gate_exprs: &mut ErrorNormExprStore| {
            if let Some(start) = keep_start.take() {
                let range = BatchedWire::from_start_len(start, keep_exprs.len());
                gate_exprs.insert_batch_shared(range, std::mem::take(keep_exprs));
            }
        };
        for (gate_id, expr) in gate_expr_pairs {
            let output_idxs = output_positions.get(&gate_id);
            let is_output = output_idxs.is_some();
            let keep_in_gate_exprs = remaining_use_count.contains(gate_id) || is_output;
            if keep_in_gate_exprs {
                match keep_start {
                    Some(start) if gate_id.0 != start.0 + keep_exprs.len() => {
                        flush(&mut keep_start, &mut keep_exprs, gate_exprs);
                        keep_start = Some(gate_id);
                    }
                    None => keep_start = Some(gate_id),
                    _ => {}
                }
                keep_exprs.push(expr.clone());
            } else {
                flush(&mut keep_start, &mut keep_exprs, gate_exprs);
            }
            if let Some(output_idxs) = output_idxs {
                for &output_idx in output_idxs {
                    final_output_exprs[output_idx] = Some(expr.clone());
                }
            }
        }
        flush(&mut keep_start, &mut keep_exprs, gate_exprs);
    }
}

// Note: h_norm and plaintext_norm computed here can be larger than the modulus `q`.
// In such a case, the error after circuit evaluation could be too large.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorNorm {
    pub plaintext_norm: PolyNorm,
    pub matrix_norm: PolyMatrixNorm,
}

impl ErrorNorm {
    pub fn new(plaintext_norm: PolyNorm, matrix_norm: PolyMatrixNorm) -> Self {
        debug_assert_eq!(plaintext_norm.ctx, matrix_norm.clone_ctx());
        Self { plaintext_norm, matrix_norm }
    }

    #[inline]
    pub fn ctx(&self) -> &SimulatorContext {
        &self.plaintext_norm.ctx
    }
    #[inline]
    pub fn clone_ctx(&self) -> Arc<SimulatorContext> {
        self.plaintext_norm.ctx.clone()
    }
}

impl_binop_with_refs!(ErrorNorm => Add::add(self, rhs: &ErrorNorm) -> ErrorNorm {
    debug_assert_eq!(self.ctx(), rhs.ctx());
    ErrorNorm {
        plaintext_norm: &self.plaintext_norm + &rhs.plaintext_norm,
        matrix_norm: &self.matrix_norm + &rhs.matrix_norm
    }
});

// Note: norm of the subtraction result is bounded by a sum of the norms of the input matrices,
// i.e., |A-B| < |A| + |B|
impl_binop_with_refs!(ErrorNorm => Sub::sub(self, rhs: &ErrorNorm) -> ErrorNorm {
    debug_assert_eq!(self.ctx(), rhs.ctx());
    ErrorNorm {
        plaintext_norm: &self.plaintext_norm + &rhs.plaintext_norm,
        matrix_norm: &self.matrix_norm + &rhs.matrix_norm
    }
});

impl_binop_with_refs!(ErrorNorm => Mul::mul(self, rhs: &ErrorNorm) -> ErrorNorm {
    debug_assert_eq!(self.ctx(), rhs.ctx());
    ErrorNorm {
        plaintext_norm: &self.plaintext_norm * &rhs.plaintext_norm,
        matrix_norm: &self.matrix_norm * PolyMatrixNorm::gadget_decomposed(self.clone_ctx(), self.ctx().m_g) + rhs.matrix_norm.clone() * &self.plaintext_norm
    }
});

impl Evaluable for ErrorNorm {
    type Params = ();
    type P = DCRTPoly;
    type Compact = ErrorNorm;

    fn to_compact(self) -> Self::Compact {
        self
    }

    fn from_compact(_: &Self::Params, compact: &Self::Compact) -> Self {
        compact.clone()
    }

    fn small_scalar_mul(&self, _: &Self::Params, scalar: &[u32]) -> Self {
        let scalar_max = BigDecimal::from(*scalar.iter().max().unwrap());
        let scalar_poly = PolyNorm::new(self.clone_ctx(), scalar_max);
        ErrorNorm {
            matrix_norm: self.matrix_norm.clone() * &scalar_poly,
            plaintext_norm: self.plaintext_norm.clone() * &scalar_poly,
        }
    }

    fn large_scalar_mul(&self, _: &Self::Params, scalar: &[BigUint]) -> Self {
        let scalar_max = scalar.iter().max().unwrap().clone();
        let scalar_bd = BigDecimal::from(num_bigint::BigInt::from(scalar_max));
        let scalar_poly = PolyNorm::new(self.clone_ctx(), scalar_bd);
        ErrorNorm {
            matrix_norm: self.matrix_norm.clone() *
                PolyMatrixNorm::gadget_decomposed(self.clone_ctx(), self.ctx().m_g),
            plaintext_norm: self.plaintext_norm.clone() * &scalar_poly,
        }
    }
}

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
enum AffineMatrixTransformOp {
    Scalar(BigDecimal),
    Poly(PolyNorm),
    Matrix(PolyMatrixNorm),
    SplitColsLeft(usize),
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
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
struct AffineInputTerm {
    input_idx: usize,
    transform: AffineMatrixTransform,
    coefficient: BigDecimal,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
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

fn add_summary_expr_batches_in_place(
    left: &mut [Arc<ErrorNormSummaryExpr>],
    right: Vec<Arc<ErrorNormSummaryExpr>>,
) {
    left.par_iter_mut().zip(right.into_par_iter()).for_each(|(left_expr, right_expr)| {
        Arc::make_mut(left_expr).add_assign_bound(right_expr.as_ref().clone());
    });
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PolyNormRun {
    len: usize,
    norm: PolyNorm,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
struct ErrorNormPreparedSubCircuitSummaryRequest {
    sub_circuit_id: usize,
    param_bindings: Arc<[SubCircuitParamValue]>,
    input_plaintext_profile: ErrorNormInputPlaintextProfile,
}

#[derive(Debug, Clone)]
pub struct NormBggPolyEncodingSTEvaluator {
    pub const_term: PolyMatrixNorm,
    pub transfer_plaintext_multiplier: PolyMatrixNorm,
    pub input_vector_multiplier: PolyMatrixNorm,
}

impl NormBggPolyEncodingSTEvaluator {
    pub fn new(
        ctx: Arc<SimulatorContext>,
        e_b0_sigma: f64,
        e_mat_sigma: &BigDecimal,
        secret_sigma: Option<BigDecimal>,
    ) -> Self {
        let c_b0_error_norm = PolyMatrixNorm::sample_gauss(
            ctx.clone(),
            1,
            ctx.m_b,
            BigDecimal::from_f64(e_b0_sigma).expect("e_b0_sigma must be finite"),
        );

        let b0_preimage_norm =
            compute_preimage_norm(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, Some(1));
        info!(
            "{}",
            format!(
                "BGG poly-encoding slot-transfer preimage norm bits {}",
                bigdecimal_bits_ceil(&b0_preimage_norm)
            )
        );

        let matrix_norm_bits = |m: &PolyMatrixNorm| bigdecimal_bits_ceil(&m.poly_norm.norm);
        let log_matrix_norm_bits = |name: &str, m: &PolyMatrixNorm| {
            debug!(
                "NormBggPolyEncodingSTEvaluator::new {} matrix norm bits {}",
                name,
                matrix_norm_bits(m)
            );
        };
        log_matrix_norm_bits("c_b0_error_norm", &c_b0_error_norm);
        let s_vec = PolyMatrixNorm::new(
            ctx.clone(),
            1,
            ctx.secret_size,
            secret_sigma.unwrap_or(BigDecimal::one()),
            None,
        );
        log_matrix_norm_bits("s_vec", &s_vec);

        // `c_b0 * gate_preimage` with `B0 * gate_preimage = target + error`.
        let gate_preimage =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, ctx.m_g, b0_preimage_norm.clone(), None);
        log_matrix_norm_bits("gate_preimage", &gate_preimage);
        let gaussian_bound = gaussian_tail_bound_factor();
        let gate_target_error = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size,
            ctx.m_g,
            e_mat_sigma * &gaussian_bound,
            None,
        );
        log_matrix_norm_bits("gate_target_error", &gate_target_error);
        let gate_target_error_term = s_vec.clone() * &gate_target_error;
        log_matrix_norm_bits("gate_target_error_term", &gate_target_error_term);
        let c_b0_gate_term = c_b0_error_norm.clone() * &gate_preimage;
        log_matrix_norm_bits("c_b0_gate_term", &c_b0_gate_term);
        let const_term = &gate_target_error_term + &c_b0_gate_term;
        log_matrix_norm_bits("const_term", &const_term);

        // `((c_b0 * slot_preimage_b0) * slot_preimage_b1) * plaintext`.
        let slot_preimage_b0 =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, 2 * ctx.m_b, b0_preimage_norm.clone(), None);
        log_matrix_norm_bits("slot_preimage_b0", &slot_preimage_b0);
        let b1_preimage_norm =
            compute_preimage_norm(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, Some(2));
        // `preimage_b1` targets the `B1` basis, whose trapdoor size is `2 * secret_size`.
        let slot_preimage_b1 =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b * 2, ctx.m_g, b1_preimage_norm.clone(), None);
        log_matrix_norm_bits("slot_preimage_b1", &slot_preimage_b1);
        let slot_preimage_b0_target_error = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size,
            ctx.m_b * 2,
            e_mat_sigma * &gaussian_bound,
            None,
        );
        log_matrix_norm_bits("slot_preimage_b0_target_error", &slot_preimage_b0_target_error);
        let slot_preimage_b1_target_error = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size * 2,
            ctx.m_g,
            e_mat_sigma * &gaussian_bound,
            None,
        );
        log_matrix_norm_bits("slot_preimage_b1_target_error", &slot_preimage_b1_target_error);
        let slot_secret_and_identity = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size,
            ctx.secret_size * 2,
            BigDecimal::one(),
            None,
        );
        log_matrix_norm_bits("slot_secret_and_identity", &slot_secret_and_identity);
        let slot_stage1_error_term =
            s_vec.clone() * slot_secret_and_identity * slot_preimage_b1_target_error;
        log_matrix_norm_bits("slot_stage1_error_term", &slot_stage1_error_term);
        let slot_stage0_error_term =
            s_vec.clone() * slot_preimage_b0_target_error * slot_preimage_b1.clone();
        log_matrix_norm_bits("slot_stage0_error_term", &slot_stage0_error_term);
        let c_b0_transfer_term = c_b0_error_norm * slot_preimage_b0 * slot_preimage_b1;
        log_matrix_norm_bits("c_b0_transfer_term", &c_b0_transfer_term);
        let transfer_plaintext_multiplier = slot_stage1_error_term.clone() +
            slot_stage0_error_term.clone() +
            c_b0_transfer_term.clone();
        log_matrix_norm_bits("transfer_plaintext_multiplier", &transfer_plaintext_multiplier);

        // `input_vector * slot_a.decompose()`.
        let input_vector_multiplier = PolyMatrixNorm::gadget_decomposed(ctx.clone(), ctx.m_g);
        log_matrix_norm_bits("input_vector_multiplier", &input_vector_multiplier);

        info!("BGG poly-encoding slot-transfer const term bits {}", matrix_norm_bits(&const_term));
        info!(
            "BGG poly-encoding slot-transfer plaintext multiplier bits {}",
            matrix_norm_bits(&transfer_plaintext_multiplier)
        );
        info!(
            "BGG poly-encoding slot-transfer input multiplier bits {}",
            matrix_norm_bits(&input_vector_multiplier)
        );

        Self { const_term, transfer_plaintext_multiplier, input_vector_multiplier }
    }
}

impl SlotTransferEvaluator<ErrorNorm> for NormBggPolyEncodingSTEvaluator {
    fn slot_transfer(
        &self,
        _: &(),
        input: &ErrorNorm,
        src_slots: &[(u32, Option<u32>)],
        _: GateId,
    ) -> ErrorNorm {
        let scalar_max =
            src_slots.iter().map(|(_, scalar)| u64::from(scalar.unwrap_or(1))).max().unwrap_or(1);
        let scalar_bd = BigDecimal::from(scalar_max);
        let plaintext_norm = input.plaintext_norm.clone() * &scalar_bd;
        let input_vector_term =
            (input.matrix_norm.clone() * &self.input_vector_multiplier) * &scalar_bd;
        let transfer_plaintext_term = self.transfer_plaintext_multiplier.clone() * &plaintext_norm;
        let matrix_norm = &self.const_term + &input_vector_term + &transfer_plaintext_term;
        ErrorNorm { matrix_norm, plaintext_norm }
    }
}

impl AffineSlotTransferEvaluator for NormBggPolyEncodingSTEvaluator {
    fn slot_transfer_affine(
        &self,
        input: &ErrorNormSummaryExpr,
        src_slots: &[(u32, Option<u32>)],
        _: GateId,
    ) -> ErrorNormSummaryExpr {
        let scalar_max =
            src_slots.iter().map(|(_, scalar)| u64::from(scalar.unwrap_or(1))).max().unwrap_or(1);
        let scalar_bd = BigDecimal::from(scalar_max);
        let plaintext_norm = input.plaintext_norm.clone() * &scalar_bd;
        let matrix_expr = input
            .matrix_expr
            .transform_matrix(&self.input_vector_multiplier)
            .transform_scalar(&scalar_bd)
            .add_expr(&AffineErrorNormExpr::constant(
                &self.const_term + &(self.transfer_plaintext_multiplier.clone() * &plaintext_norm),
            ));
        ErrorNormSummaryExpr { plaintext_norm, matrix_expr }
    }
}

#[derive(Debug, Clone)]
pub struct NormPltLWEEvaluator {
    pub e_b_times_preimage: PolyMatrixNorm,
    pub preimage_lower: PolyMatrixNorm,
}

impl NormPltLWEEvaluator {
    pub fn new(ctx: Arc<SimulatorContext>, e_b_sigma: &BigDecimal) -> Self {
        let norm = compute_preimage_norm(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, None);
        let norm_bits = bigdecimal_bits_ceil(&norm);
        info!("{}", format!("preimage norm bits {}", norm_bits));
        let e_b_init = PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_b, e_b_sigma * 6, None);
        let e_b_times_preimage =
            &e_b_init * &PolyMatrixNorm::new(ctx.clone(), ctx.m_b, ctx.m_g, norm.clone(), None);
        let preimage_lower = PolyMatrixNorm::new(ctx.clone(), ctx.m_g, ctx.m_g, norm.clone(), None);
        info!(
            "LWE PLT const term norm bits {}",
            bigdecimal_bits_ceil(&e_b_times_preimage.poly_norm.norm)
        );
        info!(
            "LWE PLT e_input multiplier norm bits {}",
            bigdecimal_bits_ceil(&preimage_lower.poly_norm.norm)
        );
        Self { e_b_times_preimage, preimage_lower }
    }
}

impl PltEvaluator<ErrorNorm> for NormPltLWEEvaluator {
    fn public_lookup(
        &self,
        _: &(),
        plt: &PublicLut<DCRTPoly>,
        _: &ErrorNorm,
        input: &ErrorNorm,
        _: GateId,
        _: usize,
    ) -> ErrorNorm {
        let matrix_norm = &self.e_b_times_preimage + (&input.matrix_norm * &self.preimage_lower);
        let plaintext_bd =
            BigDecimal::from(num_bigint::BigInt::from(plt.max_output_row().1.value().clone()));
        let plaintext_norm = PolyNorm::new(input.clone_ctx(), plaintext_bd);
        ErrorNorm { matrix_norm, plaintext_norm }
    }
}

impl AffinePltEvaluator for NormPltLWEEvaluator {
    fn public_lookup_affine(
        &self,
        input: &ErrorNormSummaryExpr,
        plt: &PublicLut<DCRTPoly>,
        _: GateId,
        _: usize,
    ) -> ErrorNormSummaryExpr {
        let plaintext_bd =
            BigDecimal::from(num_bigint::BigInt::from(plt.max_output_row().1.value().clone()));
        let plaintext_norm = PolyNorm::new(input.plaintext_norm.ctx.clone(), plaintext_bd);
        let matrix_expr = input
            .matrix_expr
            .transform_matrix(&self.preimage_lower)
            .add_expr(&AffineErrorNormExpr::constant(self.e_b_times_preimage.clone()));
        ErrorNormSummaryExpr { plaintext_norm, matrix_expr }
    }
}
#[derive(Debug, Clone)]
pub struct NormPltGGH15Evaluator {
    pub const_term: PolyMatrixNorm,
    pub e_input_multiplier: PolyMatrixNorm,
}

impl NormPltGGH15Evaluator {
    pub fn new(
        ctx: Arc<SimulatorContext>,
        e_b_sigma: &BigDecimal,
        e_mat_sigma: &BigDecimal,
        secret_sigma: Option<BigDecimal>,
    ) -> Self {
        let dump_const_term_breakdown = std::env::var("MXX_SIM_GGH15_CONST_TERM_BREAKDOWN")
            .ok()
            .map(|raw| matches!(raw.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);
        let matrix_norm_bits = |m: &PolyMatrixNorm| bigdecimal_bits_ceil(&m.poly_norm.norm);
        let gaussian_bound = gaussian_tail_bound_factor();

        let preimage_norm =
            compute_preimage_norm(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, None);
        info!("{}", format!("preimage norm bits {}", bigdecimal_bits_ceil(&preimage_norm)));
        let e_b_init =
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_b, e_b_sigma * &gaussian_bound, None);
        let s_vec = PolyMatrixNorm::new(
            ctx.clone(),
            1,
            ctx.secret_size,
            secret_sigma.unwrap_or(BigDecimal::one()),
            None,
        );
        // Corresponds to `preimage_gate1` sampled in `sample_gate_preimages_batch` stage1
        // from target `S_g * B1 + error` (B1 now has size d, so this is m_b x m_b).
        let preimage_gate1_from_b0 =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, ctx.m_b, preimage_norm.clone(), None);
        // Corresponds to stage1 Gaussian `error` in target `S_g * B1 + error`.
        let stage1_target_error = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size,
            ctx.m_b,
            e_mat_sigma * &gaussian_bound,
            None,
        );
        let gate1_from_eb = e_b_init.clone() * &preimage_gate1_from_b0;
        let gate1_from_s = s_vec.clone() * &stage1_target_error;
        // Corresponds to the error part of `c_b0 * preimage_gate1`.
        let gate1_error_total = &gate1_from_eb + &gate1_from_s;
        let gate1_total_bits = matrix_norm_bits(&gate1_error_total);
        let gate1_from_eb_bits = matrix_norm_bits(&gate1_from_eb);
        let gate1_from_s_bits = matrix_norm_bits(&gate1_from_s);

        // Corresponds to `gy.decompose()` in `public_lookup`.
        let gy_decomposed = PolyMatrixNorm::gadget_decomposed(ctx.clone(), ctx.m_g);
        // Corresponds to `v_idx` in `public_lookup`.
        let v_idx = PolyMatrixNorm::gadget_decomposed(ctx.clone(), ctx.m_g);
        // Corresponds to the vertically stacked
        // `small_decomposed_identity_chunk_from_scalar(...)` blocks used in vx accumulation.
        let small_decomposed_identity_chunks = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.m_g * ctx.log_base_q_small,
            ctx.m_g,
            ctx.base.clone() - BigDecimal::from(1u64),
            Some((ctx.m_g - 1) * ctx.log_base_q_small),
        );
        // Corresponds to `(small_decomposed_identity_chunk_from_scalar * v_idx)` in
        // `public_lookup`.
        let small_times_v = small_decomposed_identity_chunks * &v_idx;

        // Corresponds to `preimage_gate2_identity` (B0 preimage for identity/out term).
        let preimage_gate2_identity_from_b0 =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, ctx.m_g, preimage_norm.clone(), None);
        // Corresponds to `preimage_gate2_gy` (B0 preimage for gy term).
        let preimage_gate2_gy_from_b0 =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, ctx.m_g, preimage_norm.clone(), None);
        // Corresponds to `preimage_gate2_v` (B0 preimage for v_idx term).
        let preimage_gate2_v_from_b0 =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, ctx.m_g, preimage_norm.clone(), None);
        // Corresponds to concatenated `preimage_gate2_vx_chunk` blocks.
        let preimage_gate2_vx_from_b0 = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.m_b,
            ctx.m_g * ctx.log_base_q_small,
            preimage_norm.clone(),
            None,
        );
        // Corresponds to Gaussian `error` added in stage2 target
        // `S_g * w_block_identity + out_matrix + error`.
        let stage2_identity_target_error = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size,
            ctx.m_g,
            e_mat_sigma * &gaussian_bound,
            None,
        );
        // Corresponds to Gaussian `error` added in stage3 target
        // `S_g * w_block_gy - gadget + error`.
        let stage3_gy_target_error = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size,
            ctx.m_g,
            e_mat_sigma * &gaussian_bound,
            None,
        );
        // Corresponds to Gaussian `error` added in stage4 target
        // `S_g * w_block_v - (input_matrix * u_g_decomposed) + error`.
        let stage4_v_target_error = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size,
            ctx.m_g,
            e_mat_sigma * &gaussian_bound,
            None,
        );
        // Corresponds to Gaussian `error` added in stage5 target
        // `S_g * w_block_vx + (u_g_matrix * gadget_small) + error`.
        let stage5_vx_target_error = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size,
            ctx.m_g * ctx.log_base_q_small,
            e_mat_sigma * &gaussian_bound,
            None,
        );

        let gate2_identity_from_eb = e_b_init.clone() * &preimage_gate2_identity_from_b0;
        let gate2_identity_from_s = s_vec.clone() * &stage2_identity_target_error;
        let gate2_identity_total = &gate2_identity_from_eb + &gate2_identity_from_s;

        let gate2_gy_from_eb = e_b_init.clone() * &preimage_gate2_gy_from_b0;
        let gate2_gy_from_s = s_vec.clone() * &stage3_gy_target_error;
        let gate2_gy_total = &gate2_gy_from_eb + &gate2_gy_from_s;

        let gate2_v_from_eb = e_b_init.clone() * &preimage_gate2_v_from_b0;
        let gate2_v_from_s = s_vec.clone() * &stage4_v_target_error;
        let gate2_v_total = &gate2_v_from_eb + &gate2_v_from_s;

        let gate2_vx_from_eb = e_b_init.clone() * &preimage_gate2_vx_from_b0;
        let gate2_vx_from_s = s_vec.clone() * &stage5_vx_target_error;
        let gate2_vx_total = &gate2_vx_from_eb + &gate2_vx_from_s;

        // Corresponds to
        // `c_b0 * (preimage_gate2_gy * gy_decomposed + preimage_gate2_v * v_idx + vx_product_acc *
        // v_idx)`.
        let const_term_gate2_gy_total = gate2_gy_total.clone() * gy_decomposed.clone();
        let const_term_gate2_v_total = gate2_v_total.clone() * v_idx.clone();
        let const_term_gate2_vx_total = gate2_vx_total.clone() * small_times_v.clone();
        let mut const_term_gate2_t_total = const_term_gate2_gy_total.clone();
        const_term_gate2_t_total += const_term_gate2_v_total.clone();
        const_term_gate2_t_total += const_term_gate2_vx_total.clone();
        // Corresponds to `c_b0 * preimage_gate2_identity`.
        let const_term_gate2_identity_total = gate2_identity_total.clone();

        // Corresponds to the stored `preimage_lut` loaded in `public_lookup`.
        // In the GGH15 public-key evaluator, `sample_lut_preimages` already samples this matrix
        // from a target that includes identity + gy + v + vx components, and
        // `public_lookup` subtracts `preimage_gate1 * preimage_lut` without additional
        // multipliers.
        let preimage_lut_total =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, ctx.m_g, preimage_norm.clone(), None);
        // Corresponds to subtraction term
        // `c_b0 * (preimage_gate1 * preimage_lut)` in `public_lookup`.
        let const_term_lut_subtraction_total = gate1_error_total.clone() * preimage_lut_total;

        let mut const_term = const_term_gate2_identity_total.clone();
        const_term += const_term_gate2_t_total.clone();
        const_term += const_term_lut_subtraction_total.clone();
        info!(
            "{}",
            format!(
                "GGH15 PLT const term norm bits {}",
                bigdecimal_bits_ceil(&const_term.poly_norm.norm)
            )
        );

        if dump_const_term_breakdown {
            info!(
                "GGH15 const term breakdown bits: gate1_total={} gate1_from_eb={} gate1_from_s={} gate2_identity_total={} gate2_identity_from_eb={} gate2_identity_from_s={} gate2_gy_total={} gate2_gy_from_eb={} gate2_gy_from_s={} gate2_v_total={} gate2_v_from_eb={} gate2_v_from_s={} gate2_vx_total={} gate2_vx_from_eb={} gate2_vx_from_s={} term_gate2_identity={} term_gate2_gy={} term_gate2_v={} term_gate2_vx={} term_gate2_t={} term_lut_subtraction={} const_total={}",
                gate1_total_bits,
                gate1_from_eb_bits,
                gate1_from_s_bits,
                matrix_norm_bits(&gate2_identity_total),
                matrix_norm_bits(&gate2_identity_from_eb),
                matrix_norm_bits(&gate2_identity_from_s),
                matrix_norm_bits(&gate2_gy_total),
                matrix_norm_bits(&gate2_gy_from_eb),
                matrix_norm_bits(&gate2_gy_from_s),
                matrix_norm_bits(&gate2_v_total),
                matrix_norm_bits(&gate2_v_from_eb),
                matrix_norm_bits(&gate2_v_from_s),
                matrix_norm_bits(&gate2_vx_total),
                matrix_norm_bits(&gate2_vx_from_eb),
                matrix_norm_bits(&gate2_vx_from_s),
                matrix_norm_bits(&const_term_gate2_identity_total),
                matrix_norm_bits(&const_term_gate2_gy_total),
                matrix_norm_bits(&const_term_gate2_v_total),
                matrix_norm_bits(&const_term_gate2_vx_total),
                matrix_norm_bits(&const_term_gate2_t_total),
                matrix_norm_bits(&const_term_lut_subtraction_total),
                matrix_norm_bits(&const_term)
            );
        }

        // Corresponds to `input.vector * u_g_decomposed * v_idx` in `public_lookup`.
        let e_input_multiplier = PolyMatrixNorm::gadget_decomposed(ctx.clone(), ctx.m_g) * &v_idx;
        info!(
            "{}",
            format!(
                "GGH15 PLT e_input multiplier norm bits {}",
                bigdecimal_bits_ceil(&e_input_multiplier.poly_norm.norm)
            )
        );

        Self { const_term, e_input_multiplier }
    }
}

impl PltEvaluator<ErrorNorm> for NormPltGGH15Evaluator {
    fn public_lookup(
        &self,
        _: &(),
        plt: &PublicLut<DCRTPoly>,
        _: &ErrorNorm,
        input: &ErrorNorm,
        _: GateId,
        _: usize,
    ) -> ErrorNorm {
        let plaintext_bd =
            BigDecimal::from(num_bigint::BigInt::from(plt.max_output_row().1.value().clone()));
        let plaintext_norm = PolyNorm::new(input.clone_ctx(), plaintext_bd);
        let matrix_norm = &self.const_term + &input.matrix_norm * &self.e_input_multiplier;
        ErrorNorm { matrix_norm, plaintext_norm }
    }
}

impl AffinePltEvaluator for NormPltGGH15Evaluator {
    fn public_lookup_affine(
        &self,
        input: &ErrorNormSummaryExpr,
        plt: &PublicLut<DCRTPoly>,
        _: GateId,
        _: usize,
    ) -> ErrorNormSummaryExpr {
        let plaintext_bd =
            BigDecimal::from(num_bigint::BigInt::from(plt.max_output_row().1.value().clone()));
        let plaintext_norm = PolyNorm::new(input.plaintext_norm.ctx.clone(), plaintext_bd);
        let matrix_expr = input
            .matrix_expr
            .transform_matrix(&self.e_input_multiplier)
            .add_expr(&AffineErrorNormExpr::constant(self.const_term.clone()));
        ErrorNormSummaryExpr { plaintext_norm, matrix_expr }
    }
}

#[derive(Debug, Clone)]
pub struct NormPltCommitEvaluator {
    pub lut_term: PolyMatrixNorm,
}

impl NormPltCommitEvaluator {
    pub fn new(
        ctx: Arc<SimulatorContext>,
        error_sigma: &BigDecimal,
        tree_base: usize,
        circuit: &PolyCircuit<DCRTPoly>,
    ) -> Self {
        let lut_vector_len = circuit.lut_vector_len_with_subcircuits();
        let padded_len = compute_padded_len(tree_base, lut_vector_len);
        debug!(
            "NormPltCommitEvaluator padded_len={} lut_vector_len={}",
            padded_len, lut_vector_len
        );
        let preimage_norm =
            compute_preimage_norm(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, None);
        let t_bottom = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.m_b,
            tree_base * ctx.m_b * ctx.m_g * ctx.m_g,
            preimage_norm.clone(),
            None,
        );
        let j_mat = PolyMatrixNorm::new(
            ctx.clone(),
            t_bottom.ncol,
            ctx.m_b * ctx.log_base_q,
            ctx.base.clone() - BigDecimal::from(1u64),
            None,
        );
        let verifier_base = t_bottom * &j_mat;
        let verifier_norm = verifier_base *
            PolyMatrixNorm::gadget_decomposed_with_secret_size(ctx.clone(), ctx.m_b, ctx.m_b);
        let t_top = PolyMatrixNorm::new(
            ctx.clone(),
            tree_base * ctx.m_b * ctx.m_b * ctx.m_g,
            tree_base * ctx.m_b * ctx.m_g * ctx.m_g,
            preimage_norm.clone(),
            None,
        );
        let t_top_j_mat = &t_top * &j_mat;
        let msg_tensor_identity = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.m_b,
            t_top.nrow,
            ctx.base.clone() - BigDecimal::from(1u64),
            Some(ctx.m_b - 1),
        );
        let opening_base = &msg_tensor_identity * t_top_j_mat;
        let j_mat_last = PolyMatrixNorm::new(
            ctx.clone(),
            tree_base * ctx.m_b * ctx.m_g * ctx.m_g,
            ctx.m_b,
            ctx.base.clone() - BigDecimal::from(1u64),
            Some(tree_base * ctx.m_b * ctx.m_g * ctx.m_g - ctx.m_b),
        );
        let opening_base_last = &msg_tensor_identity * &t_top * &j_mat_last;
        let log_tree_base_len = {
            let mut padded_len = padded_len;
            let mut depth = 0;
            while padded_len > 1 {
                debug_assert!(padded_len % tree_base == 0);
                padded_len /= tree_base;
                depth += 1;
            }
            depth
        };
        let opening_norm = {
            let lhs = opening_base *
                PolyMatrixNorm::gadget_decomposed_with_secret_size(ctx.clone(), ctx.m_b, ctx.m_b) *
                (log_tree_base_len - 1);
            lhs + opening_base_last
        };

        let gaussian_bound = gaussian_tail_bound_factor();
        let init_error =
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_b, error_sigma * &gaussian_bound, None);
        let preimage =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, verifier_norm.nrow, preimage_norm, None);
        let lut_term = &init_error * preimage * verifier_norm + init_error * opening_norm;
        info!("lut_term norm bits {}", bigdecimal_bits_ceil(&lut_term.poly_norm.norm));
        Self { lut_term }
    }
}

impl PltEvaluator<ErrorNorm> for NormPltCommitEvaluator {
    fn public_lookup(
        &self,
        _: &<ErrorNorm as Evaluable>::Params,
        plt: &PublicLut<<ErrorNorm as Evaluable>::P>,
        _: &ErrorNorm,
        input: &ErrorNorm,
        _: GateId,
        _: usize,
    ) -> ErrorNorm {
        let plaintext_bd =
            BigDecimal::from(num_bigint::BigInt::from(plt.max_output_row().1.value().clone()));
        let plaintext_norm = PolyNorm::new(input.clone_ctx(), plaintext_bd);
        let ctx = input.clone_ctx();
        let m_b = ctx.m_b;
        let m_g = ctx.m_g;
        let matrix_norm =
            &self.lut_term + &input.matrix_norm * PolyMatrixNorm::gadget_decomposed(ctx, m_b);
        // info!("matrix_norm norm bits {}", bigdecimal_bits_ceil(&matrix_norm.poly_norm.norm));
        let (matrix_norm, _) = matrix_norm.split_cols(m_g);
        ErrorNorm { matrix_norm, plaintext_norm }
    }
}

impl AffinePltEvaluator for NormPltCommitEvaluator {
    fn public_lookup_affine(
        &self,
        input: &ErrorNormSummaryExpr,
        plt: &PublicLut<DCRTPoly>,
        _: GateId,
        _: usize,
    ) -> ErrorNormSummaryExpr {
        let ctx = self.lut_term.clone_ctx();
        let m_b = ctx.m_b;
        let m_g = ctx.m_g;
        let plaintext_bd =
            BigDecimal::from(num_bigint::BigInt::from(plt.max_output_row().1.value().clone()));
        let plaintext_norm = PolyNorm::new(input.plaintext_norm.ctx.clone(), plaintext_bd);
        let matrix_expr = input
            .matrix_expr
            .transform_matrix(&PolyMatrixNorm::gadget_decomposed(ctx, m_b))
            .add_expr(&AffineErrorNormExpr::constant(self.lut_term.clone()))
            .split_cols_left(m_g);
        ErrorNormSummaryExpr { plaintext_norm, matrix_expr }
    }
}

pub fn compute_preimage_norm(
    ring_dim_sqrt: &BigDecimal,
    m_g: u64,
    base: &BigDecimal,
    b_nrow: Option<usize>,
) -> BigDecimal {
    let c_0 = BigDecimal::from_f64(1.8).unwrap();
    let c_1 = BigDecimal::from_f64(4.7).unwrap();
    let sigma = BigDecimal::from_f64(4.578).unwrap();
    let two_sqrt = BigDecimal::from(2).sqrt().unwrap();
    let m_g_sqrt = BigDecimal::from(m_g).sqrt().expect("sqrt(m_g) failed");
    let b_nrow = b_nrow.unwrap_or(1);
    let term = BigDecimal::from(b_nrow as u64).sqrt().unwrap() * ring_dim_sqrt.clone() * m_g_sqrt +
        two_sqrt * ring_dim_sqrt +
        c_1;
    let preimage_norm =
        c_0 * BigDecimal::from_f32(6.5).unwrap() * sigma.clone() * ((base + 1) * sigma) * term;
    // let preimage_norm_bits = bigdecimal_bits_ceil(&preimage_norm);
    // info!("{}", format!("preimage norm bits {}", preimage_norm_bits));
    preimage_norm
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        circuit::PolyCircuit,
        lookup::PublicLut,
        poly::{
            Poly,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        simulator::SimulatorContext,
        slot_transfer::SlotTransferEvaluator,
    };
    use bigdecimal::BigDecimal;

    fn make_ctx() -> Arc<SimulatorContext> {
        // secpar_sqrt=50, ring_dim_sqrt=1024, base=32, log_base_q=(128/32)*7 = 28
        Arc::new(SimulatorContext::new(
            BigDecimal::from(1024u64), // ring_dim_sqrt
            BigDecimal::from(32u64),   // base
            2,
            28, // log_base_q
            3,  // log_base_q_small
        ))
    }

    fn simulate_max_error_norm_via_generic_eval_reference<P: AffinePltEvaluator>(
        circuit: &PolyCircuit<DCRTPoly>,
        ctx: Arc<SimulatorContext>,
        input_norm_bound: BigDecimal,
        input_size: usize,
        e_init_norm: &BigDecimal,
        plt_evaluator: Option<&P>,
    ) -> Vec<ErrorNorm> {
        let one_error = ErrorNorm::new(
            PolyNorm::one(ctx.clone()),
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, e_init_norm.clone(), None),
        );
        let input_error = ErrorNorm::new(
            PolyNorm::new(ctx.clone(), input_norm_bound),
            PolyMatrixNorm::new(ctx, 1, one_error.ctx().m_g, e_init_norm.clone(), None),
        );
        circuit.eval(&(), one_error, vec![input_error; input_size], plt_evaluator, None, None)
    }

    const E_B_SIGMA: f64 = 4.0;
    const E_INIT_NORM: u32 = 1 << 14;

    #[test]
    fn test_wire_norm_addition() {
        let ctx = make_ctx();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ins = circuit.input(2).to_vec();
        let out_gid = circuit.add_gate(ins[0], ins[1]);
        circuit.output(vec![out_gid]);
        let input_bound = BigDecimal::from(5u64);

        let out = circuit.simulate_max_error_norm(
            ctx.clone(),
            input_bound.clone(),
            2,
            &BigDecimal::from(E_INIT_NORM),
            None::<&NormPltLWEEvaluator>,
            None,
        );
        assert_eq!(out.len(), 1);
        // Build expected from input wires and add them
        let in_wire = ErrorNorm::new(
            PolyNorm::new(ctx.clone(), input_bound),
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(E_INIT_NORM), None),
        );
        let expected = &in_wire + &in_wire;
        assert_eq!(out[0], expected);
    }

    #[test]
    fn test_wire_norm_subtraction() {
        let ctx = make_ctx();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ins = circuit.input(2).to_vec();
        let out_gid = circuit.sub_gate(ins[0], ins[1]);
        circuit.output(vec![out_gid]);
        let input_bound = BigDecimal::from(5u64);
        let out = circuit.simulate_max_error_norm(
            ctx.clone(),
            input_bound.clone(),
            2,
            &BigDecimal::from(E_INIT_NORM),
            None::<&NormPltLWEEvaluator>,
            None,
        );
        assert_eq!(out.len(), 1);
        let in_wire = ErrorNorm::new(
            PolyNorm::new(ctx.clone(), input_bound),
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(E_INIT_NORM), None),
        );
        let expected = &in_wire - &in_wire; // subtraction bound equals addition bound
        assert_eq!(out[0], expected);
    }

    #[test]
    fn test_wire_norm_multiplication() {
        // ctx: secpar_sqrt=50, ring_dim_sqrt=1024, base=32, log_base_q=28
        let ctx = make_ctx();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ins = circuit.input(2).to_vec();
        let out_gid = circuit.mul_gate(ins[0], ins[1]);
        circuit.output(vec![out_gid]);
        let input_bound = BigDecimal::from(5u64);
        let out = circuit.simulate_max_error_norm(
            ctx.clone(),
            input_bound.clone(),
            2,
            &BigDecimal::from(E_INIT_NORM),
            None::<&NormPltLWEEvaluator>,
            None,
        );
        assert_eq!(out.len(), 1);

        // Build expected = in_wire * in_wire
        let in_wire = ErrorNorm::new(
            PolyNorm::new(ctx.clone(), input_bound),
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(E_INIT_NORM), None),
        );
        let expected = &in_wire * &in_wire;
        assert_eq!(out[0], expected);
    }

    #[test]
    fn test_wire_norm_simulator_multiplication_matches_generic_eval() {
        let ctx = make_ctx();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ins = circuit.input(2).to_vec();
        let out_gid = circuit.mul_gate(ins[0], ins[1]);
        circuit.output(vec![out_gid]);
        let input_bound = BigDecimal::from(5u64);
        let e_init_norm = BigDecimal::from(E_INIT_NORM);
        let out = circuit.simulate_max_error_norm(
            ctx.clone(),
            input_bound.clone(),
            2,
            &e_init_norm,
            None::<&NormPltLWEEvaluator>,
            None,
        );
        let generic = simulate_max_error_norm_via_generic_eval_reference(
            &circuit,
            ctx,
            input_bound,
            2,
            &e_init_norm,
            None::<&NormPltLWEEvaluator>,
        );
        assert_eq!(out, generic);
    }

    #[test]
    fn test_wire_norm_slot_transfer_matches_bgg_poly_encoding_bound() {
        let ctx = make_ctx();
        let e_b0_sigma = 11.0;
        let c_b0_error_norm = PolyMatrixNorm::sample_gauss(
            ctx.clone(),
            1,
            ctx.m_b,
            BigDecimal::from_f64(e_b0_sigma).unwrap(),
        );
        let evaluator = NormBggPolyEncodingSTEvaluator::new(
            ctx.clone(),
            e_b0_sigma,
            &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
            None,
        );
        let input = ErrorNorm::new(
            PolyNorm::new(ctx.clone(), BigDecimal::from(5u64)),
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(7u64), None),
        );
        let src_slots = [(2, None), (0, Some(3)), (1, Some(2))];

        let out = evaluator.slot_transfer(&(), &input, &src_slots, GateId(0));

        let b0_preimage_norm =
            compute_preimage_norm(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, Some(1));
        let s_vec = PolyMatrixNorm::new(ctx.clone(), 1, ctx.secret_size, BigDecimal::one(), None);
        let gate_preimage =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, ctx.m_g, b0_preimage_norm.clone(), None);
        let gate_target_error = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size,
            ctx.m_g,
            BigDecimal::from_f64(E_B_SIGMA * 6.5).unwrap(),
            None,
        );
        let slot_preimage_b0 =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, 2 * ctx.m_b, b0_preimage_norm.clone(), None);
        let b1_preimage_norm =
            compute_preimage_norm(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, Some(2));
        let slot_preimage_b1 =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b * 2, ctx.m_g, b1_preimage_norm.clone(), None);
        let slot_preimage_b0_target_error = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size,
            ctx.m_b * 2,
            BigDecimal::from_f64(E_B_SIGMA * 6.5).unwrap(),
            None,
        );
        let slot_preimage_b1_target_error = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size * 2,
            ctx.m_g,
            BigDecimal::from_f64(E_B_SIGMA * 6.5).unwrap(),
            None,
        );
        let slot_secret_and_identity = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size,
            ctx.secret_size * 2,
            BigDecimal::one(),
            None,
        );
        let scalar_bd = BigDecimal::from(3u64);
        let input_vector_multiplier = PolyMatrixNorm::gadget_decomposed(ctx.clone(), ctx.m_g);
        let plaintext_norm = input.plaintext_norm.clone() * &scalar_bd;
        let const_term =
            s_vec.clone() * &gate_target_error + c_b0_error_norm.clone() * &gate_preimage;
        let transfer_plaintext_multiplier =
            s_vec.clone() * slot_secret_and_identity * slot_preimage_b1_target_error +
                s_vec.clone() * slot_preimage_b0_target_error * slot_preimage_b1.clone() +
                c_b0_error_norm * slot_preimage_b0 * slot_preimage_b1;
        let matrix_norm = const_term +
            (input.matrix_norm.clone() * &input_vector_multiplier) * &scalar_bd +
            transfer_plaintext_multiplier * &plaintext_norm;

        assert_eq!(out, ErrorNorm { plaintext_norm, matrix_norm });
    }

    #[test]
    fn test_wire_norm_slot_transfer_bound_is_independent_of_slot_count() {
        let ctx = make_ctx();
        let e_b0_sigma = 9.0;
        let evaluator = NormBggPolyEncodingSTEvaluator::new(
            ctx.clone(),
            e_b0_sigma,
            &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
            None,
        );
        let input = ErrorNorm::new(
            PolyNorm::new(ctx.clone(), BigDecimal::from(4u64)),
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(6u64), None),
        );

        let out_single = evaluator.slot_transfer(&(), &input, &[(0, Some(2))], GateId(0));
        let out_many = evaluator.slot_transfer(
            &(),
            &input,
            &[(0, Some(2)), (1, Some(2)), (2, Some(2))],
            GateId(1),
        );

        assert_eq!(out_single, out_many);
    }

    #[test]
    fn test_wire_norm_lwe_plt_bounds() {
        // Build a tiny LUT on DCRTPoly where the maximum output coeff is known (e.g., 7)
        let params = DCRTPolyParams::default();
        let plt = PublicLut::<DCRTPoly>::new(
            &params,
            2,
            |params, idx| match idx {
                0 => Some((
                    0,
                    DCRTPoly::from_usize_to_constant(params, 5)
                        .coeffs()
                        .into_iter()
                        .next()
                        .expect("constant-term coefficient must exist"),
                )),
                1 => Some((
                    1,
                    DCRTPoly::from_usize_to_constant(params, 7)
                        .coeffs()
                        .into_iter()
                        .next()
                        .expect("constant-term coefficient must exist"),
                )),
                _ => unreachable!("index out of range for test LUT"),
            },
            None,
        );

        // Circuit: out = PLT(in)
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(1).to_vec();
        let plt_id = circuit.register_public_lookup(plt);
        let out_gate = circuit.public_lookup_gate(inputs[0], plt_id);
        circuit.output(vec![out_gate]);

        let ctx = make_ctx();
        let input_bound = BigDecimal::from(5u64);
        let plt_evaluator =
            NormPltLWEEvaluator::new(ctx.clone(), &BigDecimal::from_f64(E_B_SIGMA).unwrap());
        let out = circuit.simulate_max_error_norm(
            ctx.clone(),
            input_bound.clone(),
            1,
            &BigDecimal::from(E_INIT_NORM),
            Some(&plt_evaluator),
            None,
        );
        assert_eq!(out.len(), 1);
        // Bound must be max output coeff across LUT entries (7)
        assert_eq!(out[0].plaintext_norm.norm, BigDecimal::from(7u64));
    }

    #[test]
    fn test_wire_norm_ggh15_plt_bounds() {
        // Build a tiny LUT on DCRTPoly where the maximum output coeff is known (e.g., 7)
        let params = DCRTPolyParams::default();
        let plt = PublicLut::<DCRTPoly>::new(
            &params,
            2,
            |params, idx| match idx {
                0 => Some((
                    0,
                    DCRTPoly::from_usize_to_constant(params, 5)
                        .coeffs()
                        .into_iter()
                        .next()
                        .expect("constant-term coefficient must exist"),
                )),
                1 => Some((
                    1,
                    DCRTPoly::from_usize_to_constant(params, 7)
                        .coeffs()
                        .into_iter()
                        .next()
                        .expect("constant-term coefficient must exist"),
                )),
                _ => unreachable!("index out of range for test LUT"),
            },
            None,
        );

        // Circuit: out = PLT(in)
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(1).to_vec();
        let plt_id = circuit.register_public_lookup(plt);
        let out_gate = circuit.public_lookup_gate(inputs[0], plt_id);
        circuit.output(vec![out_gate]);

        let ctx = make_ctx();
        let input_bound = BigDecimal::from(5u64);
        let plt_evaluator = NormPltGGH15Evaluator::new(
            ctx.clone(),
            &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
            &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
            None,
        );
        let out = circuit.simulate_max_error_norm(
            ctx.clone(),
            input_bound.clone(),
            1,
            &BigDecimal::from(E_INIT_NORM),
            Some(&plt_evaluator),
            None,
        );
        assert_eq!(out.len(), 1);
        // Bound must be max output coeff across LUT entries (7)
        assert_eq!(out[0].plaintext_norm.norm, BigDecimal::from(7u64));
    }

    #[test]
    fn test_wire_norm_simulator_ggh15_plt_uses_lut_plaintext_bound() {
        let params = DCRTPolyParams::default();
        let plt = PublicLut::<DCRTPoly>::new(
            &params,
            2,
            |params, idx| match idx {
                0 => Some((
                    0,
                    DCRTPoly::from_usize_to_constant(params, 5)
                        .coeffs()
                        .into_iter()
                        .next()
                        .expect("constant-term coefficient must exist"),
                )),
                1 => Some((
                    1,
                    DCRTPoly::from_usize_to_constant(params, 7)
                        .coeffs()
                        .into_iter()
                        .next()
                        .expect("constant-term coefficient must exist"),
                )),
                _ => unreachable!("index out of range for test LUT"),
            },
            None,
        );

        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(1).to_vec();
        let plt_id = circuit.register_public_lookup(plt);
        let out_gate = circuit.public_lookup_gate(inputs[0], plt_id);
        circuit.output(vec![out_gate]);

        let ctx = make_ctx();
        let input_bound = BigDecimal::from(5u64);
        let plt_evaluator = NormPltGGH15Evaluator::new(
            ctx.clone(),
            &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
            &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
            None,
        );
        let out = circuit.simulate_max_error_norm(
            ctx,
            input_bound,
            1,
            &BigDecimal::from(E_INIT_NORM),
            Some(&plt_evaluator),
            None,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].plaintext_norm.norm, BigDecimal::from(7u64));
    }

    #[test]
    fn test_wire_norm_simulator_sub_circuit_matches_generic_eval() {
        let params = DCRTPolyParams::default();
        let plt = PublicLut::<DCRTPoly>::new(
            &params,
            2,
            |params, idx| match idx {
                0 => Some((
                    0,
                    DCRTPoly::from_usize_to_constant(params, 3)
                        .coeffs()
                        .into_iter()
                        .next()
                        .expect("constant-term coefficient must exist"),
                )),
                1 => Some((
                    1,
                    DCRTPoly::from_usize_to_constant(params, 5)
                        .coeffs()
                        .into_iter()
                        .next()
                        .expect("constant-term coefficient must exist"),
                )),
                _ => unreachable!("index out of range for test LUT"),
            },
            None,
        );

        let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
        let sub_inputs = sub_circuit.input(1).to_vec();
        let squared = sub_circuit.mul_gate(sub_inputs[0], sub_inputs[0]);
        let sub_out = sub_circuit.add_gate(squared, sub_inputs[0]);
        sub_circuit.output(vec![sub_out]);

        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(2).to_vec();
        let sub_circuit_id = circuit.register_sub_circuit(sub_circuit);
        let left = circuit.call_sub_circuit(sub_circuit_id, &[inputs[0]]);
        let right = circuit.call_sub_circuit(sub_circuit_id, &[inputs[1]]);
        let summed = circuit.add_gate(left[0], right[0]);
        let plt_id = circuit.register_public_lookup(plt);
        let out = circuit.public_lookup_gate(summed, plt_id);
        circuit.output(vec![out]);

        let ctx = make_ctx();
        let input_bound = BigDecimal::from(13u64);
        let e_init_norm = BigDecimal::from(E_INIT_NORM);
        let plt_evaluator = NormPltGGH15Evaluator::new(
            ctx.clone(),
            &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
            &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
            None,
        );

        let simulated = circuit.simulate_max_error_norm(
            ctx.clone(),
            input_bound.clone(),
            2,
            &e_init_norm,
            Some(&plt_evaluator),
            None,
        );
        let generic = simulate_max_error_norm_via_generic_eval_reference(
            &circuit,
            ctx,
            input_bound,
            2,
            &e_init_norm,
            Some(&plt_evaluator),
        );

        assert_eq!(simulated, generic);
    }

    #[test]
    fn test_wire_norm_simulator_sub_circuit_recomputes_for_new_plaintext_profile() {
        let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
        let sub_inputs = sub_circuit.input(1).to_vec();
        let squared = sub_circuit.mul_gate(sub_inputs[0], sub_inputs[0]);
        let sub_out = sub_circuit.add_gate(squared, sub_inputs[0]);
        sub_circuit.output(vec![sub_out]);

        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(1).to_vec();
        let doubled = circuit.add_gate(inputs[0], inputs[0]);
        let sub_circuit_id = circuit.register_sub_circuit(sub_circuit);
        let left = circuit.call_sub_circuit(sub_circuit_id, &[inputs[0]]);
        let right = circuit.call_sub_circuit(sub_circuit_id, &[doubled]);
        let out = circuit.add_gate(left[0], right[0]);
        circuit.output(vec![out]);

        let ctx = make_ctx();
        let input_bound = BigDecimal::from(13u64);
        let e_init_norm = BigDecimal::from(E_INIT_NORM);

        let simulated = circuit.simulate_max_error_norm(
            ctx.clone(),
            input_bound.clone(),
            1,
            &e_init_norm,
            None::<&NormPltLWEEvaluator>,
            None,
        );
        let generic = simulate_max_error_norm_via_generic_eval_reference(
            &circuit,
            ctx,
            input_bound,
            1,
            &e_init_norm,
            None::<&NormPltLWEEvaluator>,
        );

        assert_eq!(simulated, generic);
    }

    #[test]
    fn test_wire_norm_simulator_nested_sub_circuit_matches_generic_eval() {
        let mut inner_sub_circuit = PolyCircuit::<DCRTPoly>::new();
        let inner_inputs = inner_sub_circuit.input(1).to_vec();
        let inner_out = inner_sub_circuit.add_gate(inner_inputs[0], inner_inputs[0]);
        inner_sub_circuit.output(vec![inner_out]);

        let mut outer_sub_circuit = PolyCircuit::<DCRTPoly>::new();
        let outer_inputs = outer_sub_circuit.input(2).to_vec();
        let inner_sub_circuit_id = outer_sub_circuit.register_sub_circuit(inner_sub_circuit);
        let inner_from_second =
            outer_sub_circuit.call_sub_circuit(inner_sub_circuit_id, &[outer_inputs[1]]);
        let outer_out = outer_sub_circuit.add_gate(outer_inputs[0], inner_from_second[0]);
        outer_sub_circuit.output(vec![outer_out]);

        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(2).to_vec();
        let outer_sub_circuit_id = circuit.register_sub_circuit(outer_sub_circuit);
        let left = circuit.call_sub_circuit(outer_sub_circuit_id, &[inputs[0], inputs[1]]);
        let right = circuit.call_sub_circuit(outer_sub_circuit_id, &[inputs[1], inputs[0]]);
        let out = circuit.add_gate(left[0], right[0]);
        circuit.output(vec![out]);

        let ctx = make_ctx();
        let input_bound = BigDecimal::from(11u64);
        let e_init_norm = BigDecimal::from(E_INIT_NORM);

        let simulated = circuit.simulate_max_error_norm(
            ctx.clone(),
            input_bound.clone(),
            2,
            &e_init_norm,
            None::<&NormPltLWEEvaluator>,
            None,
        );
        let generic = simulate_max_error_norm_via_generic_eval_reference(
            &circuit,
            ctx,
            input_bound,
            2,
            &e_init_norm,
            None::<&NormPltLWEEvaluator>,
        );

        assert_eq!(simulated, generic);
    }

    #[test]
    fn test_wire_norm_commit_plt_bounds() {
        // Build a tiny LUT on DCRTPoly where the maximum output coeff is known (e.g., 7)
        let params = DCRTPolyParams::default();
        let plt = PublicLut::<DCRTPoly>::new(
            &params,
            2,
            |params, idx| match idx {
                0 => Some((
                    0,
                    DCRTPoly::from_usize_to_constant(params, 5)
                        .coeffs()
                        .into_iter()
                        .next()
                        .expect("constant-term coefficient must exist"),
                )),
                1 => Some((
                    1,
                    DCRTPoly::from_usize_to_constant(params, 7)
                        .coeffs()
                        .into_iter()
                        .next()
                        .expect("constant-term coefficient must exist"),
                )),
                _ => unreachable!("index out of range for test LUT"),
            },
            None,
        );

        // Circuit: out = PLT(in)
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(1).to_vec();
        let plt_id = circuit.register_public_lookup(plt);
        let out_gate = circuit.public_lookup_gate(inputs[0], plt_id);
        circuit.output(vec![out_gate]);

        let ctx = make_ctx();
        let input_bound = BigDecimal::from(5u64);
        let tree_base = 2;
        let plt_evaluator = NormPltCommitEvaluator::new(
            ctx.clone(),
            &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
            tree_base,
            &circuit,
        );
        let out = circuit.simulate_max_error_norm(
            ctx.clone(),
            input_bound.clone(),
            1,
            &BigDecimal::from(E_INIT_NORM),
            Some(&plt_evaluator),
            None,
        );
        assert_eq!(out.len(), 1);
        // Bound must be max output coeff across LUT entries (7)
        assert_eq!(out[0].plaintext_norm.norm, BigDecimal::from(7u64));
    }
}
