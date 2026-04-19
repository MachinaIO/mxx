use dashmap::{DashMap, mapref::entry::Entry};
use num_bigint::BigUint;
#[cfg(feature = "gpu")]
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fmt::Debug,
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

#[cfg(feature = "gpu")]
use crate::poly::dcrt::gpu::detected_gpu_device_ids;
use crate::{
    circuit::{
        Evaluable, GateParamSource, PolyGate, PolyGateKind, PolyGateType, SlotTransferSpec,
        SubCircuitParamKind, SubCircuitParamValue, gate::GateId,
    },
    lookup::{PltEvaluator, PublicLut},
    poly::Poly,
    slot_transfer::SlotTransferEvaluator,
};
use tracing::{debug, info};

mod analysis;
mod construction;
mod eval;
mod subcircuits;

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ::serde::Serialize, ::serde::Deserialize)]
pub struct BatchedWire {
    start: GateId,
    end: GateId,
}

impl BatchedWire {
    pub fn new(start: GateId, end: GateId) -> Self {
        assert!(start.0 <= end.0, "BatchedWire start {} must not exceed end {}", start, end);
        Self { start, end }
    }

    pub fn single(gate_id: GateId) -> Self {
        Self::new(gate_id, GateId(gate_id.0 + 1))
    }

    pub fn from_start_len(start: GateId, len: usize) -> Self {
        Self::new(start, GateId(start.0 + len))
    }

    pub fn start(self) -> GateId {
        self.start
    }

    pub fn end(self) -> GateId {
        self.end
    }

    pub fn len(self) -> usize {
        self.end.0 - self.start.0
    }

    pub fn is_empty(self) -> bool {
        self.start == self.end
    }

    pub fn is_single_wire(self) -> bool {
        self.len() == 1
    }

    pub fn as_single_wire(self) -> GateId {
        debug_assert!(
            self.is_single_wire(),
            "expected a single-wire batch, got [{}, {})",
            self.start.0,
            self.end.0
        );
        self.start
    }

    pub fn at(self, idx: usize) -> Self {
        self.slice(idx..idx + 1)
    }

    pub fn slice(self, range: Range<usize>) -> Self {
        assert!(
            range.start <= range.end && range.end <= self.len(),
            "BatchedWire slice [{}, {}) is out of bounds for len {}",
            range.start,
            range.end,
            self.len()
        );
        Self::new(GateId(self.start.0 + range.start), GateId(self.start.0 + range.end))
    }

    pub fn split_at(self, mid: usize) -> (Self, Self) {
        assert!(mid <= self.len(), "BatchedWire split offset {} exceeds len {}", mid, self.len());
        (self.slice(0..mid), self.slice(mid..self.len()))
    }

    pub fn from_batches<I, W>(batches: I) -> Self
    where
        I: IntoIterator<Item = W>,
        W: Into<BatchedWire>,
    {
        let mut iter = batches.into_iter().map(Into::into);
        let first = iter.next().expect("BatchedWire::from_batches requires at least one batch");
        let mut merged = first;
        for batch in iter {
            assert_eq!(
                merged.end(),
                batch.start(),
                "BatchedWire::from_batches requires contiguous batches, got {} then {}",
                merged,
                batch
            );
            merged = Self::new(merged.start(), batch.end());
        }
        merged
    }

    pub fn gate_ids(self) -> impl ExactSizeIterator<Item = GateId> {
        (self.start.0..self.end.0).map(GateId)
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = GateId> {
        self.gate_ids()
    }

    pub fn to_vec(self) -> Vec<GateId> {
        self.gate_ids().collect()
    }
}

impl From<GateId> for BatchedWire {
    fn from(value: GateId) -> Self {
        Self::single(value)
    }
}

impl From<&GateId> for BatchedWire {
    fn from(value: &GateId) -> Self {
        Self::single(*value)
    }
}

impl From<&BatchedWire> for BatchedWire {
    fn from(value: &BatchedWire) -> Self {
        *value
    }
}

impl std::fmt::Display for BatchedWire {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}, {})", self.start.0, self.end.0)
    }
}

pub(crate) fn batched_wire_slice_len(batches: &[BatchedWire]) -> usize {
    batches.iter().copied().map(BatchedWire::len).sum()
}

pub(crate) fn iter_batched_wire_gates(
    batches: &[BatchedWire],
) -> impl Iterator<Item = GateId> + '_ {
    batches.iter().copied().flat_map(BatchedWire::gate_ids)
}

pub(crate) fn batched_wire_slice_at(batches: &[BatchedWire], idx: usize) -> GateId {
    let mut offset = idx;
    for batch in batches {
        if offset < batch.len() {
            return GateId(batch.start().0 + offset);
        }
        offset -= batch.len();
    }
    panic!(
        "batched wire index {idx} out of range for flattened len {}",
        batched_wire_slice_len(batches)
    );
}

#[cfg(feature = "gpu")]
#[derive(Debug)]
enum LoadedGateInputs<E: Evaluable> {
    SkipExisting,
    Unary(E),
    Binary(E, E),
}

#[cfg(feature = "gpu")]
#[derive(Debug)]
struct LoadedGateCtx<E: Evaluable> {
    gate_id: GateId,
    gate: PolyGate,
    shard_idx: usize,
    inputs: LoadedGateInputs<E>,
}

#[cfg(feature = "gpu")]
#[derive(Debug)]
enum ComputedGateValue<E: Evaluable> {
    SkipExisting,
    Value(E),
}

#[cfg(feature = "gpu")]
#[derive(Debug)]
struct ComputedGateCtx<E: Evaluable> {
    gate_id: GateId,
    gate: PolyGate,
    shard_idx: usize,
    value: ComputedGateValue<E>,
}

#[derive(Debug)]
pub(crate) struct LookupRegistry<P: Poly> {
    next_id: AtomicUsize,
    lookups: DashMap<usize, Arc<PublicLut<P>>>,
}

#[derive(Debug)]
pub(crate) struct BindingRegistry {
    next_id: AtomicUsize,
    binding_sets: DashMap<usize, Arc<[SubCircuitParamValue]>>,
    binding_set_index: DashMap<Arc<[SubCircuitParamValue]>, usize>,
}

#[derive(Debug)]
pub(crate) struct InputSetRegistry {
    next_id: AtomicUsize,
    input_sets: DashMap<usize, Arc<[BatchedWire]>>,
    input_set_index: DashMap<Arc<[BatchedWire]>, usize>,
}

impl<P: Poly> LookupRegistry<P> {
    fn new() -> Self {
        Self { next_id: AtomicUsize::new(0), lookups: DashMap::new() }
    }

    fn is_empty(&self) -> bool {
        self.lookups.is_empty()
    }

    fn register(&self, lookup: PublicLut<P>) -> usize {
        let lut_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.lookups.insert(lut_id, Arc::new(lookup));
        lut_id
    }
}

impl BindingRegistry {
    fn new() -> Self {
        Self {
            next_id: AtomicUsize::new(0),
            binding_sets: DashMap::new(),
            binding_set_index: DashMap::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.binding_sets.is_empty()
    }

    fn register_arc(&self, candidate: Arc<[SubCircuitParamValue]>) -> usize {
        if let Some(existing) = self.binding_set_index.get(&candidate) {
            return *existing;
        }
        match self.binding_set_index.entry(candidate.clone()) {
            Entry::Occupied(existing) => *existing.get(),
            Entry::Vacant(vacant) => {
                let binding_set_id = self.next_id.fetch_add(1, Ordering::Relaxed);
                self.binding_sets.insert(binding_set_id, candidate);
                vacant.insert(binding_set_id);
                binding_set_id
            }
        }
    }

    pub(crate) fn register(&self, bindings: &[SubCircuitParamValue]) -> usize {
        self.register_arc(Arc::from(bindings.to_vec()))
    }

    fn get(&self, binding_set_id: usize) -> Arc<[SubCircuitParamValue]> {
        self.binding_sets
            .get(&binding_set_id)
            .unwrap_or_else(|| panic!("binding set {binding_set_id} not found"))
            .clone()
    }
}

impl InputSetRegistry {
    fn new() -> Self {
        Self {
            next_id: AtomicUsize::new(0),
            input_sets: DashMap::new(),
            input_set_index: DashMap::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.input_sets.is_empty()
    }

    fn register_arc(&self, candidate: Arc<[BatchedWire]>) -> usize {
        if let Some(existing) = self.input_set_index.get(&candidate) {
            return *existing;
        }
        match self.input_set_index.entry(candidate.clone()) {
            Entry::Occupied(existing) => *existing.get(),
            Entry::Vacant(vacant) => {
                let input_set_id = self.next_id.fetch_add(1, Ordering::Relaxed);
                self.input_sets.insert(input_set_id, candidate);
                vacant.insert(input_set_id);
                input_set_id
            }
        }
    }

    pub(crate) fn register(&self, input_ids: &[BatchedWire]) -> usize {
        self.register_arc(Arc::from(input_ids.to_vec()))
    }

    fn get(&self, input_set_id: usize) -> Arc<[BatchedWire]> {
        self.input_sets
            .get(&input_set_id)
            .unwrap_or_else(|| panic!("input set {input_set_id} not found"))
            .clone()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SubCircuitCall {
    pub(crate) sub_circuit_id: usize,
    pub(crate) shared_input_prefix_set_id: Option<usize>,
    pub(crate) input_suffix: Vec<BatchedWire>,
    pub(crate) binding_set_id: usize,
    pub(crate) scoped_call_id: usize,
    pub(crate) output_gate_ids: Vec<GateId>,
    pub(crate) num_outputs: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SubCircuitCallInfo {
    pub(crate) sub_circuit_id: usize,
    pub(crate) inputs: Vec<BatchedWire>,
    pub(crate) param_bindings: Arc<[SubCircuitParamValue]>,
    pub(crate) output_gate_ids: Vec<GateId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SummedSubCircuitCall {
    pub(crate) sub_circuit_id: usize,
    pub(crate) call_input_set_ids: Vec<usize>,
    pub(crate) call_binding_set_ids: Vec<usize>,
    pub(crate) scoped_call_ids: Vec<usize>,
    pub(crate) output_gate_ids: Vec<GateId>,
    pub(crate) num_outputs: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SummedSubCircuitCallInfo {
    pub(crate) sub_circuit_id: usize,
    pub(crate) call_inputs: Vec<Vec<BatchedWire>>,
    pub(crate) param_bindings: Vec<Arc<[SubCircuitParamValue]>>,
    pub(crate) output_gate_ids: Vec<GateId>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct GroupedCallExecutionLayer {
    pub(crate) regular_gate_ids: Vec<GateId>,
    pub(crate) sub_circuit_call_ids: Vec<usize>,
    pub(crate) summed_sub_circuit_call_ids: Vec<usize>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct GroupedExecutionPlan {
    pub(crate) layers: Vec<GroupedCallExecutionLayer>,
    pub(crate) reachable_input_gate_ids: Vec<GateId>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct NonFreeDepthContributionVector {
    pub(crate) counts: [u32; 10],
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct NonFreeDepthProfile {
    pub(crate) total_depth: u32,
    pub(crate) contributions: NonFreeDepthContributionVector,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) struct NonFreeDepthCacheKey {
    circuit_key: usize,
    input_profiles: Box<[NonFreeDepthProfile]>,
}

#[derive(Debug, Clone)]
pub struct PolyCircuit<P: Poly> {
    pub(crate) gates: BTreeMap<GateId, PolyGate>,
    pub(crate) print_value: BTreeMap<GateId, String>,
    pub(crate) sub_circuits: BTreeMap<usize, Arc<PolyCircuit<P>>>,
    pub(crate) sub_circuit_calls: BTreeMap<usize, SubCircuitCall>,
    pub(crate) summed_sub_circuit_calls: BTreeMap<usize, SummedSubCircuitCall>,
    pub(crate) sub_circuit_params: Vec<SubCircuitParamKind>,
    pub(crate) output_ids: Vec<GateId>,
    pub(crate) num_input: usize,
    pub(crate) gate_counts: HashMap<PolyGateKind, usize>,
    pub(crate) lookup_registry: Arc<LookupRegistry<P>>,
    pub(crate) binding_registry: Arc<BindingRegistry>,
    pub(crate) input_set_registry: Arc<InputSetRegistry>,
    pub(crate) next_scoped_call_id: usize,
    pub(crate) allow_register_lookup: bool,
}

impl<P: Poly> PartialEq for PolyCircuit<P> {
    fn eq(&self, other: &Self) -> bool {
        self.gates == other.gates &&
            self.print_value == other.print_value &&
            self.sub_circuits == other.sub_circuits &&
            self.sub_circuit_calls == other.sub_circuit_calls &&
            self.summed_sub_circuit_calls == other.summed_sub_circuit_calls &&
            self.sub_circuit_params == other.sub_circuit_params &&
            self.output_ids == other.output_ids &&
            self.gate_counts == other.gate_counts &&
            self.num_input == other.num_input &&
            self.next_scoped_call_id == other.next_scoped_call_id &&
            self.allow_register_lookup == other.allow_register_lookup
    }
}

impl<P: Poly> Eq for PolyCircuit<P> {}

impl<P: Poly> Default for PolyCircuit<P> {
    fn default() -> Self {
        Self::new()
    }
}
