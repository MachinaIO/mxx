pub mod evaluable;
pub mod gate;
pub mod serde;

pub use evaluable::*;
pub use gate::{
    GateParamSource, PolyGate, PolyGateKind, PolyGateType, SlotTransferSpec, SubCircuitParamKind,
    SubCircuitParamValue,
};

use dashmap::{DashMap, mapref::entry::Entry};
use num_bigint::BigUint;
#[cfg(feature = "gpu")]
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fmt::Debug,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

#[cfg(feature = "gpu")]
use crate::poly::dcrt::gpu::detected_gpu_device_ids;
use crate::{
    circuit::gate::GateId,
    lookup::{PltEvaluator, PublicLut},
    poly::Poly,
    slot_transfer::SlotTransferEvaluator,
};
use tracing::{debug, info};

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
    input_sets: DashMap<usize, Arc<[GateId]>>,
    input_set_index: DashMap<Arc<[GateId]>, usize>,
}

static TEMP_SUBCIRCUIT_STORAGE_ID: AtomicUsize = AtomicUsize::new(0);

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

    fn register(&self, bindings: &[SubCircuitParamValue]) -> usize {
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

    fn register_arc(&self, candidate: Arc<[GateId]>) -> usize {
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

    fn register(&self, input_ids: &[GateId]) -> usize {
        self.register_arc(Arc::from(input_ids.to_vec()))
    }

    fn get(&self, input_set_id: usize) -> Arc<[GateId]> {
        self.input_sets
            .get(&input_set_id)
            .unwrap_or_else(|| panic!("input set {input_set_id} not found"))
            .clone()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SubCircuitCall {
    sub_circuit_id: usize,
    shared_input_prefix_set_id: Option<usize>,
    input_suffix: Vec<GateId>,
    binding_set_id: usize,
    scoped_call_id: usize,
    output_gate_ids: Vec<GateId>,
    num_outputs: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SubCircuitCallInfo {
    pub(crate) sub_circuit_id: usize,
    pub(crate) inputs: Vec<GateId>,
    pub(crate) param_bindings: Arc<[SubCircuitParamValue]>,
    pub(crate) output_gate_ids: Vec<GateId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SummedSubCircuitCall {
    sub_circuit_id: usize,
    call_input_set_ids: Vec<usize>,
    call_binding_set_ids: Vec<usize>,
    scoped_call_ids: Vec<usize>,
    output_gate_ids: Vec<GateId>,
    num_outputs: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SummedSubCircuitCallInfo {
    pub(crate) sub_circuit_id: usize,
    pub(crate) call_inputs: Vec<Vec<GateId>>,
    pub(crate) param_bindings: Vec<Arc<[SubCircuitParamValue]>>,
    pub(crate) output_gate_ids: Vec<GateId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StoredSubCircuit<P: Poly> {
    InMemory(Arc<PolyCircuit<P>>),
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct SubCircuitDiskStorage;

impl SubCircuitDiskStorage {
    fn new(_dir: &Path) -> Self {
        Self
    }

    pub(crate) fn temporary(_prefix: &str) -> Self {
        let _ = TEMP_SUBCIRCUIT_STORAGE_ID.fetch_add(1, Ordering::Relaxed);
        Self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CircuitExecutionLayer {
    pub(crate) sub_circuit_call_ids: Vec<usize>,
    pub(crate) summed_sub_circuit_call_ids: Vec<usize>,
    pub(crate) regular_gate_types: Vec<PolyGateType>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ErrorNormExecutionLayer {
    pub(crate) regular_gate_ids: Vec<GateId>,
    pub(crate) sub_circuit_call_ids: Vec<usize>,
    pub(crate) summed_sub_circuit_call_ids: Vec<usize>,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
enum ErrorNormExecutionNodeId {
    Regular(GateId),
    SubCircuitCall(usize),
    SummedSubCircuitCall(usize),
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct NonFreeDepthCacheKey {
    circuit_ptr: usize,
    input_levels: Box<[u32]>,
}

#[derive(Debug, Clone)]
pub struct PolyCircuit<P: Poly> {
    gates: BTreeMap<GateId, PolyGate>,
    print_value: BTreeMap<GateId, String>,
    sub_circuits: BTreeMap<usize, StoredSubCircuit<P>>,
    sub_circuit_calls: BTreeMap<usize, SubCircuitCall>,
    summed_sub_circuit_calls: BTreeMap<usize, SummedSubCircuitCall>,
    sub_circuit_params: Vec<SubCircuitParamKind>,
    output_ids: Vec<GateId>,
    num_input: usize,
    gate_counts: HashMap<PolyGateKind, usize>,
    lookup_registry: Arc<LookupRegistry<P>>,
    binding_registry: Arc<BindingRegistry>,
    input_set_registry: Arc<InputSetRegistry>,
    next_scoped_call_id: usize,
    allow_register_lookup: bool,
    sub_circuit_disk_storage: Option<SubCircuitDiskStorage>,
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

impl<P: Poly> PolyCircuit<P> {
    pub fn new() -> Self {
        let mut gates = BTreeMap::new();
        // Ensure the reserved constant-one gate exists at GateId(0)
        gates.insert(GateId(0), PolyGate::new(GateId(0), PolyGateType::Input, vec![]));
        let mut gate_counts = HashMap::new();
        gate_counts.insert(PolyGateKind::Input, 1);
        let lookup_registry = Arc::new(LookupRegistry::new());
        let binding_registry = Arc::new(BindingRegistry::new());
        let input_set_registry = Arc::new(InputSetRegistry::new());
        Self {
            gates,
            print_value: BTreeMap::new(),
            sub_circuits: BTreeMap::new(),
            sub_circuit_calls: BTreeMap::new(),
            summed_sub_circuit_calls: BTreeMap::new(),
            sub_circuit_params: vec![],
            output_ids: vec![],
            num_input: 0,
            gate_counts,
            lookup_registry,
            binding_registry,
            input_set_registry,
            next_scoped_call_id: 0,
            allow_register_lookup: true,
            sub_circuit_disk_storage: None,
        }
    }

    fn inherit_shared_registries(
        &mut self,
        lookup_registry: Arc<LookupRegistry<P>>,
        binding_registry: Arc<BindingRegistry>,
        input_set_registry: Arc<InputSetRegistry>,
    ) {
        if !Arc::ptr_eq(&self.lookup_registry, &lookup_registry) {
            if !self.lookup_registry.is_empty() {
                panic!("sub-circuit may not register lookup tables");
            }
            self.lookup_registry = Arc::clone(&lookup_registry);
        }
        if !Arc::ptr_eq(&self.binding_registry, &binding_registry) {
            if !self.binding_registry.is_empty() {
                for call in self.sub_circuit_calls.values_mut() {
                    let bindings = self.binding_registry.get(call.binding_set_id);
                    call.binding_set_id = binding_registry.register_arc(bindings);
                }
                for call in self.summed_sub_circuit_calls.values_mut() {
                    call.call_binding_set_ids.iter_mut().for_each(|binding_set_id| {
                        let bindings = self.binding_registry.get(*binding_set_id);
                        *binding_set_id = binding_registry.register_arc(bindings);
                    });
                }
            }
            self.binding_registry = Arc::clone(&binding_registry);
        }
        if !Arc::ptr_eq(&self.input_set_registry, &input_set_registry) {
            if !self.input_set_registry.is_empty() {
                for call in self.sub_circuit_calls.values_mut() {
                    if let Some(input_set_id) = call.shared_input_prefix_set_id.as_mut() {
                        let input_ids = self.input_set_registry.get(*input_set_id);
                        *input_set_id = input_set_registry.register_arc(input_ids);
                    }
                }
                for call in self.summed_sub_circuit_calls.values_mut() {
                    call.call_input_set_ids.iter_mut().for_each(|input_set_id| {
                        let input_ids = self.input_set_registry.get(*input_set_id);
                        *input_set_id = input_set_registry.register_arc(input_ids);
                    });
                }
            }
            self.input_set_registry = Arc::clone(&input_set_registry);
        }
        self.allow_register_lookup = false;
        for sub in self.sub_circuits.values_mut() {
            let StoredSubCircuit::InMemory(sub) = sub;
            if Arc::ptr_eq(&sub.lookup_registry, &lookup_registry) &&
                Arc::ptr_eq(&sub.binding_registry, &binding_registry) &&
                Arc::ptr_eq(&sub.input_set_registry, &input_set_registry)
            {
                continue;
            }
            Arc::make_mut(sub).inherit_shared_registries(
                Arc::clone(&lookup_registry),
                Arc::clone(&binding_registry),
                Arc::clone(&input_set_registry),
            );
        }
    }

    pub(crate) fn inherit_registries_from_parent(&mut self, parent: &Self) {
        self.inherit_shared_registries(
            Arc::clone(&parent.lookup_registry),
            Arc::clone(&parent.binding_registry),
            Arc::clone(&parent.input_set_registry),
        );
    }

    pub(crate) fn inherit_registries(
        &mut self,
        lookup_registry: Arc<LookupRegistry<P>>,
        binding_registry: Arc<BindingRegistry>,
        input_set_registry: Arc<InputSetRegistry>,
    ) {
        self.inherit_shared_registries(lookup_registry, binding_registry, input_set_registry);
    }

    pub(crate) fn cloned_subcircuit_disk_storage(&self) -> Option<SubCircuitDiskStorage> {
        self.sub_circuit_disk_storage.clone()
    }

    pub(crate) fn use_subcircuit_disk_storage(&mut self, storage: SubCircuitDiskStorage) {
        if !self.sub_circuits.is_empty() {
            panic!(
                "disk-backed sub-circuit storage must be configured before registering sub-circuits"
            );
        }
        self.sub_circuit_disk_storage = Some(storage);
    }

    fn with_sub_circuit<R>(&self, circuit_id: usize, f: impl FnOnce(&Self) -> R) -> R {
        let stored = self.sub_circuits.get(&circuit_id).expect("sub-circuit not found");
        match stored {
            StoredSubCircuit::InMemory(sub) => f(sub.as_ref()),
        }
    }

    fn sub_circuit_num_output(&self, circuit_id: usize) -> usize {
        let stored = self.sub_circuits.get(&circuit_id).expect("sub-circuit not found");
        match stored {
            StoredSubCircuit::InMemory(sub) => sub.as_ref().num_output(),
        }
    }

    /// Get number of inputs
    pub fn num_input(&self) -> usize {
        self.num_input
    }

    /// Get number of outputs
    pub fn num_output(&self) -> usize {
        self.output_ids.len()
    }

    /// Get number of sub-circuit parameters
    pub fn num_sub_circuit_params(&self) -> usize {
        self.sub_circuit_params.len()
    }

    /// Get number of gates
    pub fn num_gates(&self) -> usize {
        self.gates.len()
    }

    pub fn gates_in_id_order(&self) -> impl Iterator<Item = (&GateId, &PolyGate)> {
        self.gates.iter()
    }

    pub(crate) fn gate(&self, gate_id: GateId) -> &PolyGate {
        self.gates.get(&gate_id).unwrap_or_else(|| panic!("gate {gate_id} not found"))
    }

    pub(crate) fn sorted_input_gate_ids(&self) -> Vec<GateId> {
        let mut input_gate_ids = self
            .gates
            .iter()
            .filter_map(|(id, gate)| match &gate.gate_type {
                PolyGateType::Input if id.0 != 0 => Some(*id),
                _ => None,
            })
            .collect::<Vec<_>>();
        input_gate_ids.sort_by_key(|gid| gid.0);
        input_gate_ids
    }

    pub(crate) fn output_gate_ids(&self) -> &[GateId] {
        &self.output_ids
    }

    pub(crate) fn lookup_table(&self, lut_id: usize) -> Arc<PublicLut<P>> {
        self.lookup_registry.lookups.get(&lut_id).expect("lookup table missing").clone()
    }

    pub(crate) fn binding_set(&self, binding_set_id: usize) -> Arc<[SubCircuitParamValue]> {
        self.binding_registry.get(binding_set_id)
    }

    pub(crate) fn intern_binding_set(&self, bindings: &[SubCircuitParamValue]) -> usize {
        self.binding_registry.register(bindings)
    }

    pub(crate) fn input_set(&self, input_set_id: usize) -> Arc<[GateId]> {
        self.input_set_registry.get(input_set_id)
    }

    pub(crate) fn intern_input_set(&self, input_ids: &[GateId]) -> usize {
        self.input_set_registry.register(input_ids)
    }

    fn sub_circuit_call_input_len(&self, call: &SubCircuitCall) -> usize {
        call.shared_input_prefix_set_id
            .map(|input_set_id| self.input_set(input_set_id).len())
            .unwrap_or(0) +
            call.input_suffix.len()
    }

    fn with_sub_circuit_call_inputs<T>(
        &self,
        call: &SubCircuitCall,
        f: impl FnOnce(&[GateId], &[GateId]) -> T,
    ) -> T {
        if let Some(input_set_id) = call.shared_input_prefix_set_id {
            let shared_prefix = self.input_set(input_set_id);
            f(shared_prefix.as_ref(), &call.input_suffix)
        } else {
            f(&[], &call.input_suffix)
        }
    }

    pub(crate) fn with_sub_circuit_call_inputs_by_id<T>(
        &self,
        call_id: usize,
        f: impl FnOnce(&[GateId], &[GateId]) -> T,
    ) -> T {
        let call = self.sub_circuit_calls.get(&call_id).expect("sub-circuit call missing");
        self.with_sub_circuit_call_inputs(call, f)
    }

    pub(crate) fn with_sub_circuit_call_by_id<T>(
        &self,
        call_id: usize,
        f: impl FnOnce(usize, Arc<[SubCircuitParamValue]>, &[GateId], &[GateId], &[GateId]) -> T,
    ) -> T {
        let call = self.sub_circuit_calls.get(&call_id).expect("sub-circuit call missing");
        let param_bindings = self.binding_set(call.binding_set_id);
        self.with_sub_circuit_call_inputs(call, |shared_prefix, suffix| {
            f(call.sub_circuit_id, param_bindings, shared_prefix, suffix, &call.output_gate_ids)
        })
    }

    pub(crate) fn sub_circuit_call_shared_prefix_set_id(&self, call_id: usize) -> Option<usize> {
        self.sub_circuit_calls
            .get(&call_id)
            .expect("sub-circuit call missing")
            .shared_input_prefix_set_id
    }

    pub(crate) fn for_each_summed_sub_circuit_call_input(
        &self,
        summed_call_id: usize,
        mut f: impl FnMut(GateId),
    ) {
        let call = self
            .summed_sub_circuit_calls
            .get(&summed_call_id)
            .expect("summed sub-circuit call missing");
        for input_set_id in &call.call_input_set_ids {
            for &input_id in self.input_set(*input_set_id).iter() {
                f(input_id);
            }
        }
    }

    fn collect_sub_circuit_call_inputs(&self, call: &SubCircuitCall) -> Vec<GateId> {
        self.with_sub_circuit_call_inputs(call, |shared_prefix, suffix| {
            let mut inputs = Vec::with_capacity(shared_prefix.len() + suffix.len());
            inputs.extend_from_slice(shared_prefix);
            inputs.extend_from_slice(suffix);
            inputs
        })
    }

    pub(crate) fn sub_circuit_call_info(&self, call_id: usize) -> SubCircuitCallInfo {
        let call = self.sub_circuit_calls.get(&call_id).expect("sub-circuit call missing");
        SubCircuitCallInfo {
            sub_circuit_id: call.sub_circuit_id,
            inputs: self.collect_sub_circuit_call_inputs(call),
            param_bindings: self.binding_set(call.binding_set_id),
            output_gate_ids: call.output_gate_ids.clone(),
        }
    }

    pub(crate) fn summed_sub_circuit_call_info(
        &self,
        summed_call_id: usize,
    ) -> SummedSubCircuitCallInfo {
        let call = self
            .summed_sub_circuit_calls
            .get(&summed_call_id)
            .expect("summed sub-circuit call missing");
        let (call_inputs, param_bindings) = rayon::join(
            || {
                call.call_input_set_ids
                    .iter()
                    .map(|input_set_id| self.input_set(*input_set_id).as_ref().to_vec())
                    .collect::<Vec<_>>()
            },
            || {
                call.call_binding_set_ids
                    .iter()
                    .map(|binding_set_id| self.binding_set(*binding_set_id))
                    .collect::<Vec<_>>()
            },
        );
        SummedSubCircuitCallInfo {
            sub_circuit_id: call.sub_circuit_id,
            call_inputs,
            param_bindings,
            output_gate_ids: call.output_gate_ids.clone(),
        }
    }

    pub(crate) fn with_summed_sub_circuit_call_by_id<T>(
        &self,
        summed_call_id: usize,
        f: impl FnOnce(usize, &[usize], &[usize], &[GateId]) -> T,
    ) -> T {
        let call = self
            .summed_sub_circuit_calls
            .get(&summed_call_id)
            .expect("summed sub-circuit call missing");
        f(
            call.sub_circuit_id,
            &call.call_input_set_ids,
            &call.call_binding_set_ids,
            &call.output_gate_ids,
        )
    }

    pub fn register_sub_circuit_param(&mut self, kind: SubCircuitParamKind) -> usize {
        let param_id = self.sub_circuit_params.len();
        self.sub_circuit_params.push(kind);
        param_id
    }

    fn expect_sub_circuit_param_kind(&self, param_id: usize, expected: SubCircuitParamKind) {
        let actual = self
            .sub_circuit_params
            .get(param_id)
            .copied()
            .unwrap_or_else(|| panic!("sub-circuit parameter {param_id} is out of range"));
        assert_eq!(
            actual, expected,
            "sub-circuit parameter kind mismatch for param {param_id}: expected {:?}, got {:?}",
            expected, actual
        );
    }

    pub fn count_gates_by_type_vec(&self) -> HashMap<PolyGateKind, usize> {
        self.expanded_gate_counts(true)
    }

    fn expanded_gate_counts(&self, include_inputs: bool) -> HashMap<PolyGateKind, usize> {
        let mut counts: HashMap<PolyGateKind, usize> = HashMap::new();
        for gate in self.gates.values() {
            let kind = gate.gate_type.kind();
            if matches!(kind, PolyGateKind::SubCircuitOutput | PolyGateKind::SummedSubCircuitOutput)
            {
                continue;
            }
            if !include_inputs && matches!(kind, PolyGateKind::Input) {
                continue;
            }
            *counts.entry(kind).or_insert(0) += 1;
        }

        let mut call_counts: HashMap<usize, usize> = HashMap::new();
        for call in self.sub_circuit_calls.values() {
            *call_counts.entry(call.sub_circuit_id).or_insert(0) += 1;
        }
        for (sub_id, times) in call_counts {
            let sub_counts = self.with_sub_circuit(sub_id, |sub| sub.expanded_gate_counts(false));
            for (kind, count) in sub_counts {
                *counts.entry(kind).or_insert(0) += count * times;
            }
        }
        for summed_call in self.summed_sub_circuit_calls.values() {
            let times = summed_call.call_input_set_ids.len();
            let sub_counts = self.with_sub_circuit(summed_call.sub_circuit_id, |sub| {
                sub.expanded_gate_counts(false)
            });
            for (kind, count) in sub_counts {
                *counts.entry(kind).or_insert(0) += count * times;
            }
            if times > 0 {
                *counts.entry(PolyGateKind::Add).or_insert(0) +=
                    summed_call.num_outputs * (times - 1);
            }
        }
        counts
    }

    /// Computes the circuit depth excluding Add gates, including sub-circuits.
    ///
    /// Definition:
    /// - Inputs and the reserved constant-one gate contribute 0 to depth.
    /// - Add, Sub, SmallScalarMul gates do not increase depth: level(add) = max(level(inputs)).
    /// - Any other non-input gate increases depth by 1: level(g) = max(level(inputs)) + 1.
    /// - Sub-circuits contribute their internal non-free depth based on the call inputs.
    /// - If there are no outputs, returns 0.
    pub fn non_free_depth(&self) -> usize {
        if self.output_ids.is_empty() {
            return 0;
        }
        let input_levels = vec![0u32; self.num_input()];
        let mut depth_cache = HashMap::<NonFreeDepthCacheKey, Arc<[u32]>>::new();
        let output_levels =
            self.non_free_depths_with_input_levels_cached(&input_levels, &mut depth_cache);
        output_levels.par_iter().copied().max().unwrap_or(0) as usize
    }

    fn non_free_depths_with_input_levels_cached(
        &self,
        input_levels: &[u32],
        depth_cache: &mut HashMap<NonFreeDepthCacheKey, Arc<[u32]>>,
    ) -> Arc<[u32]> {
        if self.output_ids.is_empty() {
            return Arc::from(Vec::<u32>::new());
        }
        debug_assert_eq!(self.num_input(), input_levels.len());
        let cache_key = NonFreeDepthCacheKey {
            circuit_ptr: self as *const Self as usize,
            input_levels: input_levels.to_vec().into_boxed_slice(),
        };
        if let Some(cached) = depth_cache.get(&cache_key) {
            return cached.clone();
        }
        let input_index_by_gate = self
            .sorted_input_gate_ids()
            .into_iter()
            .enumerate()
            .map(|(input_idx, gate_id)| {
                (gate_id, u32::try_from(input_idx).expect("input index overflowed u32"))
            })
            .collect::<HashMap<_, _>>();
        let mut remaining_output_uses_by_call = vec![0u32; self.sub_circuit_calls.len()];
        let mut remaining_output_uses_by_summed_call =
            vec![0u32; self.summed_sub_circuit_calls.len()];
        let output_set = self.output_ids.iter().copied().collect::<HashSet<_>>();
        let mut remaining_use_count = HashMap::<GateId, u32>::new();
        for gate in self.gates.values() {
            if let PolyGateType::SubCircuitOutput { call_id, .. } = gate.gate_type {
                remaining_output_uses_by_call[call_id] += 1;
            } else if let PolyGateType::SummedSubCircuitOutput { summed_call_id, .. } =
                gate.gate_type
            {
                remaining_output_uses_by_summed_call[summed_call_id] += 1;
            }
            self.for_each_gate_dependency_input(gate, |input_id| {
                *remaining_use_count.entry(input_id).or_insert(0) += 1;
            });
        }

        let mut live_gate_levels = HashMap::<GateId, u32>::new();
        let mut call_output_levels =
            (0..self.sub_circuit_calls.len()).map(|_| None).collect::<Vec<Option<Arc<[u32]>>>>();
        let mut summed_call_output_levels = (0..self.summed_sub_circuit_calls.len())
            .map(|_| None)
            .collect::<Vec<Option<Arc<[u32]>>>>();

        // Gates are created in dependency order, so GateId order is already topological.
        for (&gate_id, gate) in self.gates.iter() {
            let gate_level = match &gate.gate_type {
                PolyGateType::Input => {
                    if gate_id == GateId(0) {
                        0
                    } else {
                        let input_idx = input_index_by_gate
                            .get(&gate_id)
                            .copied()
                            .expect("input gate index missing");
                        input_levels[input_idx as usize]
                    }
                }
                PolyGateType::Add | PolyGateType::Sub | PolyGateType::SmallScalarMul { .. } => gate
                    .input_gates
                    .iter()
                    .map(|input_id| {
                        *live_gate_levels.get(input_id).unwrap_or_else(|| {
                            panic!("non-free depth level missing for gate {input_id}")
                        })
                    })
                    .max()
                    .unwrap_or(0),
                PolyGateType::LargeScalarMul { .. } |
                PolyGateType::Mul |
                PolyGateType::PubLut { .. } |
                PolyGateType::SlotTransfer { .. } => {
                    gate.input_gates
                        .iter()
                        .map(|input_id| {
                            *live_gate_levels.get(input_id).unwrap_or_else(|| {
                                panic!("non-free depth level missing for gate {input_id}")
                            })
                        })
                        .max()
                        .unwrap_or(0) +
                        1
                }
                PolyGateType::SubCircuitOutput { call_id, output_idx, .. } => {
                    if call_output_levels[*call_id].is_none() {
                        let call =
                            self.sub_circuit_calls.get(call_id).expect("sub-circuit call missing");
                        let child_input_levels =
                            self.with_sub_circuit_call_inputs(call, |shared_prefix, suffix| {
                                let mut levels =
                                    Vec::with_capacity(shared_prefix.len() + suffix.len());
                                levels.extend(shared_prefix.iter().map(|input_id| {
                                    *live_gate_levels.get(input_id).unwrap_or_else(|| {
                                        panic!("non-free depth level missing for gate {input_id}")
                                    })
                                }));
                                levels.extend(suffix.iter().map(|input_id| {
                                    *live_gate_levels.get(input_id).unwrap_or_else(|| {
                                        panic!("non-free depth level missing for gate {input_id}")
                                    })
                                }));
                                levels
                            });
                        let output_levels = self.with_sub_circuit(call.sub_circuit_id, |sub| {
                            sub.non_free_depths_with_input_levels_cached(
                                &child_input_levels,
                                depth_cache,
                            )
                        });
                        call_output_levels[*call_id] = Some(output_levels);
                    }
                    let output_level = call_output_levels[*call_id]
                        .as_ref()
                        .and_then(|levels| levels.get(*output_idx))
                        .copied()
                        .unwrap_or_else(|| {
                            panic!(
                                "sub-circuit output index {} out of range for call {}",
                                output_idx, call_id
                            )
                        });
                    debug_assert!(remaining_output_uses_by_call[*call_id] > 0);
                    remaining_output_uses_by_call[*call_id] -= 1;
                    if remaining_output_uses_by_call[*call_id] == 0 {
                        call_output_levels[*call_id] = None;
                    }
                    output_level
                }
                PolyGateType::SummedSubCircuitOutput { summed_call_id, output_idx, .. } => {
                    if summed_call_output_levels[*summed_call_id].is_none() {
                        let call = self
                            .summed_sub_circuit_calls
                            .get(summed_call_id)
                            .expect("summed sub-circuit call missing");
                        let mut accumulated = vec![0u32; call.num_outputs];
                        let mut output_levels_by_input_set = HashMap::<usize, Arc<[u32]>>::new();
                        for input_set_id in &call.call_input_set_ids {
                            let output_levels = if let Some(output_levels) =
                                output_levels_by_input_set.get(input_set_id)
                            {
                                output_levels.clone()
                            } else {
                                let child_input_levels = self
                                    .input_set(*input_set_id)
                                    .iter()
                                    .map(|input_id| {
                                        *live_gate_levels.get(input_id).unwrap_or_else(|| {
                                            panic!(
                                                "non-free depth level missing for gate {input_id}"
                                            )
                                        })
                                    })
                                    .collect::<Vec<_>>();
                                let output_levels =
                                    self.with_sub_circuit(call.sub_circuit_id, |sub| {
                                        sub.non_free_depths_with_input_levels_cached(
                                            &child_input_levels,
                                            depth_cache,
                                        )
                                    });
                                output_levels_by_input_set
                                    .insert(*input_set_id, output_levels.clone());
                                output_levels
                            };
                            for (acc, level) in accumulated.iter_mut().zip(output_levels.iter()) {
                                *acc = (*acc).max(*level);
                            }
                        }
                        summed_call_output_levels[*summed_call_id] = Some(Arc::from(accumulated));
                    }
                    let output_level = summed_call_output_levels[*summed_call_id]
                        .as_ref()
                        .and_then(|levels| levels.get(*output_idx))
                        .copied()
                        .unwrap_or_else(|| {
                            panic!(
                                "summed sub-circuit output index {} out of range for call {}",
                                output_idx, summed_call_id
                            )
                        });
                    debug_assert!(remaining_output_uses_by_summed_call[*summed_call_id] > 0);
                    remaining_output_uses_by_summed_call[*summed_call_id] -= 1;
                    if remaining_output_uses_by_summed_call[*summed_call_id] == 0 {
                        summed_call_output_levels[*summed_call_id] = None;
                    }
                    output_level
                }
            };
            let gate_has_future_uses = remaining_use_count.get(&gate_id).copied().unwrap_or(0) > 0;
            if gate_has_future_uses || output_set.contains(&gate_id) {
                live_gate_levels.insert(gate_id, gate_level);
            }
            self.for_each_gate_dependency_input(gate, |input_id| {
                let Some(remaining_uses) = remaining_use_count.get_mut(&input_id) else {
                    return;
                };
                debug_assert!(
                    *remaining_uses > 0,
                    "remaining use counter underflow for gate {input_id}"
                );
                *remaining_uses -= 1;
                if *remaining_uses == 0 && !output_set.contains(&input_id) {
                    live_gate_levels.remove(&input_id);
                }
            });
        }

        let output_levels = Arc::<[u32]>::from(
            self.output_ids
                .iter()
                .map(|output_id| {
                    *live_gate_levels.get(output_id).unwrap_or_else(|| {
                        panic!("non-free depth level missing for output gate {output_id}")
                    })
                })
                .collect::<Vec<_>>(),
        );
        depth_cache.insert(cache_key, output_levels.clone());
        output_levels
    }

    pub fn recompute_gate_counts(&mut self) {
        self.gate_counts.clear();
        for gate in self.gates.values() {
            let kind = gate.gate_type.kind();
            *self.gate_counts.entry(kind).or_insert(0) += 1;
        }
    }

    pub fn print(&mut self, gate_id: GateId, prefix: String) {
        self.print_value.insert(gate_id, prefix);
    }

    pub fn input(&mut self, num_input: usize) -> Vec<GateId> {
        let mut input_gates = Vec::with_capacity(num_input);
        for _ in 0..num_input {
            let next_id = self.gates.len();
            let gid = GateId(next_id);
            self.increment_gate_kind(PolyGateKind::Input);
            self.gates.insert(gid, PolyGate::new(gid, PolyGateType::Input, vec![]));
            input_gates.push(gid);
        }
        self.num_input += num_input;
        input_gates
    }

    pub fn output(&mut self, outputs: Vec<GateId>) {
        #[cfg(debug_assertions)]
        assert_eq!(self.output_ids.len(), 0);
        for gate_id in outputs.into_iter() {
            self.output_ids.push(gate_id);
        }
    }

    pub fn const_zero_gate(&mut self) -> GateId {
        self.not_gate(GateId(0))
    }

    /// index 0 have value 1
    pub fn const_one_gate(&mut self) -> GateId {
        GateId(0)
    }

    pub fn const_minus_one_gate(&mut self) -> GateId {
        let zero = self.const_zero_gate();
        self.sub_gate(zero, GateId(0))
    }

    pub fn and_gate(&mut self, left: GateId, right: GateId) -> GateId {
        self.mul_gate(left, right)
    }

    /// Computes the NOT gate using arithmetic inversion: `1 - x`.
    /// This operation assumes that `x` is restricted to binary values (0 or 1),
    /// meaning it should only be used with polynomials sampled from a bit distribution.
    /// The computation is achieved by subtracting `x` from 1 (i.e., `0 - x + 1`).
    pub fn not_gate(&mut self, input: GateId) -> GateId {
        self.sub_gate(GateId(0), input)
    }

    pub fn or_gate(&mut self, left: GateId, right: GateId) -> GateId {
        let add = self.add_gate(left, right);
        let mul = self.mul_gate(left, right);
        self.sub_gate(add, mul) // A + B - A*B
    }

    /// Computes the NAND gate as `NOT(AND(left, right))`.
    /// This operation follows the same restriction as the NOT gate:
    /// `left` and `right` must be bit distribution (0 or 1)
    pub fn nand_gate(&mut self, left: GateId, right: GateId) -> GateId {
        let and_result = self.and_gate(left, right);
        self.not_gate(and_result) // NOT AND
    }

    /// Computes the NOR gate as `NOT(OR(left, right))`.
    /// This operation follows the same restriction as the NOT gate:
    /// `left` and `right` must be bit distribution (0 or 1)
    pub fn nor_gate(&mut self, left: GateId, right: GateId) -> GateId {
        let or_result = self.or_gate(left, right);
        self.not_gate(or_result) // NOT OR
    }

    pub fn xor_gate(&mut self, left: GateId, right: GateId) -> GateId {
        let two = self.add_gate(GateId(0), GateId(0));
        let mul = self.mul_gate(left, right);
        let two_mul = self.mul_gate(two, mul);
        let add = self.add_gate(left, right);
        self.sub_gate(add, two_mul) // A + B - 2*A*B
    }

    /// Computes the XNOR gate as `NOT(XOR(left, right))`.
    /// This operation follows the same restriction as the NOT gate:
    /// `left` and `right` must be bit distribution (0 or 1)
    pub fn xnor_gate(&mut self, left: GateId, right: GateId) -> GateId {
        let xor_result = self.xor_gate(left, right);
        self.not_gate(xor_result) // NOT XOR
    }

    pub fn const_digits(&mut self, digits: &[u32]) -> GateId {
        let one = self.const_one_gate();
        self.small_scalar_mul(one, digits)
    }

    pub fn const_poly(&mut self, poly: &P) -> GateId {
        let one = self.const_one_gate();
        self.large_scalar_mul(one, &poly.coeffs_biguints())
    }

    pub fn add_gate(&mut self, left_input: GateId, right_input: GateId) -> GateId {
        self.new_gate_generic(vec![left_input, right_input], PolyGateType::Add)
    }

    pub fn sub_gate(&mut self, left_input: GateId, right_input: GateId) -> GateId {
        self.new_gate_generic(vec![left_input, right_input], PolyGateType::Sub)
    }

    pub fn mul_gate(&mut self, left_input: GateId, right_input: GateId) -> GateId {
        self.new_gate_generic(vec![left_input, right_input], PolyGateType::Mul)
    }

    pub fn small_scalar_mul(&mut self, input: GateId, scalar: &[u32]) -> GateId {
        self.new_gate_generic(
            vec![input],
            PolyGateType::SmallScalarMul { scalar: GateParamSource::Const(scalar.to_vec()) },
        )
    }

    pub fn small_scalar_mul_param(&mut self, input: GateId, param_id: usize) -> GateId {
        self.expect_sub_circuit_param_kind(param_id, SubCircuitParamKind::SmallScalarMul);
        self.new_gate_generic(
            vec![input],
            PolyGateType::SmallScalarMul { scalar: GateParamSource::Param(param_id) },
        )
    }

    pub fn large_scalar_mul(&mut self, input: GateId, scalar: &[BigUint]) -> GateId {
        self.new_gate_generic(
            vec![input],
            PolyGateType::LargeScalarMul { scalar: GateParamSource::Const(scalar.to_vec()) },
        )
    }

    pub fn large_scalar_mul_param(&mut self, input: GateId, param_id: usize) -> GateId {
        self.expect_sub_circuit_param_kind(param_id, SubCircuitParamKind::LargeScalarMul);
        self.new_gate_generic(
            vec![input],
            PolyGateType::LargeScalarMul { scalar: GateParamSource::Param(param_id) },
        )
    }

    pub fn poly_scalar_mul(&mut self, input: GateId, scalar: &P) -> GateId {
        self.new_gate_generic(
            vec![input],
            PolyGateType::LargeScalarMul {
                scalar: GateParamSource::Const(scalar.coeffs_biguints()),
            },
        )
    }

    /// Lowers a ring-dimension-normalized rotation into multiplication by the
    /// monomial `x^shift`.
    ///
    /// `shift` must already be reduced modulo the ring dimension.
    pub fn rotate_gate(&mut self, input: GateId, shift: u64) -> GateId {
        let shift = usize::try_from(shift)
            .expect("PolyCircuit::rotate_gate shift does not fit in usize on this platform");
        let mut scalar = vec![0; shift + 1];
        scalar[shift] = 1;
        self.small_scalar_mul(input, &scalar)
    }

    pub fn public_lookup_gate(&mut self, input: GateId, lut_id: usize) -> GateId {
        self.new_gate_generic(
            vec![input],
            PolyGateType::PubLut { lut_id: GateParamSource::Const(lut_id) },
        )
    }

    pub fn public_lookup_gate_param(&mut self, input: GateId, param_id: usize) -> GateId {
        self.expect_sub_circuit_param_kind(param_id, SubCircuitParamKind::PubLut);
        self.new_gate_generic(
            vec![input],
            PolyGateType::PubLut { lut_id: GateParamSource::Param(param_id) },
        )
    }

    pub fn slot_transfer_gate(
        &mut self,
        input: GateId,
        src_slots: &[(u32, Option<u32>)],
    ) -> GateId {
        self.new_gate_generic(
            vec![input],
            PolyGateType::SlotTransfer {
                src_slots: GateParamSource::Const(SlotTransferSpec::explicit(src_slots.to_vec())),
            },
        )
    }

    pub fn slot_transfer_gate_param(&mut self, input: GateId, param_id: usize) -> GateId {
        self.expect_sub_circuit_param_kind(param_id, SubCircuitParamKind::SlotTransfer);
        self.new_gate_generic(
            vec![input],
            PolyGateType::SlotTransfer { src_slots: GateParamSource::Param(param_id) },
        )
    }

    fn new_gate_generic(&mut self, inputs: Vec<GateId>, gate_type: PolyGateType) -> GateId {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.output_ids.len(), 0);
            assert_eq!(inputs.len(), gate_type.num_input());
            for gate_id in inputs.iter() {
                assert!(self.gates.contains_key(gate_id));
            }
        }
        let gate_id = self.gates.len();
        let gate_kind = gate_type.kind();
        self.increment_gate_kind(gate_kind);
        self.gates.insert(GateId(gate_id), PolyGate::new(GateId(gate_id), gate_type, inputs));
        GateId(gate_id)
    }

    fn new_sub_circuit_output_gate(
        &mut self,
        call_id: usize,
        output_idx: usize,
        num_inputs: usize,
    ) -> GateId {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.output_ids.len(), 0);
        }
        let gate_id = GateId(self.gates.len());
        let gate_type = PolyGateType::SubCircuitOutput { call_id, output_idx, num_inputs };
        self.increment_gate_kind(gate_type.kind());
        self.gates.insert(gate_id, PolyGate::new(gate_id, gate_type, Vec::new()));
        gate_id
    }

    fn new_summed_sub_circuit_output_gate(
        &mut self,
        summed_call_id: usize,
        output_idx: usize,
        num_inputs: usize,
    ) -> GateId {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.output_ids.len(), 0);
        }
        let gate_id = GateId(self.gates.len());
        let gate_type =
            PolyGateType::SummedSubCircuitOutput { summed_call_id, output_idx, num_inputs };
        self.increment_gate_kind(gate_type.kind());
        self.gates.insert(gate_id, PolyGate::new(gate_id, gate_type, Vec::new()));
        gate_id
    }

    fn increment_gate_kind(&mut self, kind: PolyGateKind) {
        *self.gate_counts.entry(kind).or_insert(0) += 1;
    }

    fn gate_dependency_input_count(&self, gate: &PolyGate) -> usize {
        match &gate.gate_type {
            PolyGateType::SubCircuitOutput { call_id, .. } => self
                .sub_circuit_calls
                .get(call_id)
                .map(|call| self.sub_circuit_call_input_len(call))
                .expect("sub-circuit call missing"),
            PolyGateType::SummedSubCircuitOutput { summed_call_id, .. } => self
                .summed_sub_circuit_calls
                .get(summed_call_id)
                .expect("summed sub-circuit call missing")
                .call_input_set_ids
                .iter()
                .map(|input_set_id| self.input_set(*input_set_id).len())
                .sum(),
            _ => gate.input_gates.len(),
        }
    }

    fn for_each_gate_dependency_input(&self, gate: &PolyGate, mut f: impl FnMut(GateId)) {
        match &gate.gate_type {
            PolyGateType::SubCircuitOutput { call_id, .. } => {
                let call = self.sub_circuit_calls.get(call_id).expect("sub-circuit call missing");
                self.with_sub_circuit_call_inputs(call, |shared_prefix, suffix| {
                    for &input_id in shared_prefix {
                        f(input_id);
                    }
                    for &input_id in suffix {
                        f(input_id);
                    }
                });
            }
            PolyGateType::SummedSubCircuitOutput { summed_call_id, .. } => {
                let call = self
                    .summed_sub_circuit_calls
                    .get(summed_call_id)
                    .expect("summed sub-circuit call missing");
                for input_set_id in &call.call_input_set_ids {
                    let input_ids = self.input_set(*input_set_id);
                    for &input_id in input_ids.iter() {
                        f(input_id);
                    }
                }
            }
            _ => {
                for &input_id in &gate.input_gates {
                    f(input_id);
                }
            }
        }
    }

    fn error_norm_node_for_gate(&self, gate_id: GateId) -> Option<ErrorNormExecutionNodeId> {
        let gate = self.gate(gate_id);
        match &gate.gate_type {
            PolyGateType::Input => None,
            PolyGateType::SubCircuitOutput { call_id, .. } => {
                Some(ErrorNormExecutionNodeId::SubCircuitCall(*call_id))
            }
            PolyGateType::SummedSubCircuitOutput { summed_call_id, .. } => {
                Some(ErrorNormExecutionNodeId::SummedSubCircuitCall(*summed_call_id))
            }
            _ => Some(ErrorNormExecutionNodeId::Regular(gate_id)),
        }
    }

    fn populate_error_norm_input_set_max_level(
        &self,
        input_set_id: usize,
        node_levels: &mut HashMap<ErrorNormExecutionNodeId, usize>,
        input_set_levels: &mut HashMap<usize, Option<usize>>,
        visiting: &mut HashSet<ErrorNormExecutionNodeId>,
        levels: &mut Vec<ErrorNormExecutionLayer>,
    ) -> Option<usize> {
        if let Some(&max_level) = input_set_levels.get(&input_set_id) {
            return max_level;
        }
        let mut max_dependency_level: Option<usize> = None;
        let mut has_input_dependency = false;
        for &input_id in self.input_set(input_set_id).iter() {
            if let Some(dep_node) = self.error_norm_node_for_gate(input_id) {
                let dep_level = self.populate_error_norm_node_level(
                    dep_node,
                    node_levels,
                    input_set_levels,
                    visiting,
                    levels,
                );
                max_dependency_level = Some(match max_dependency_level {
                    Some(curr) => curr.max(dep_level),
                    None => dep_level,
                });
            } else {
                has_input_dependency = true;
            }
        }
        let max_dependency_level = if has_input_dependency {
            Some(max_dependency_level.map_or(0, |dep_level| dep_level))
        } else {
            max_dependency_level
        };
        input_set_levels.insert(input_set_id, max_dependency_level);
        max_dependency_level
    }

    fn populate_error_norm_node_level(
        &self,
        node: ErrorNormExecutionNodeId,
        node_levels: &mut HashMap<ErrorNormExecutionNodeId, usize>,
        input_set_levels: &mut HashMap<usize, Option<usize>>,
        visiting: &mut HashSet<ErrorNormExecutionNodeId>,
        levels: &mut Vec<ErrorNormExecutionLayer>,
    ) -> usize {
        if let Some(&level) = node_levels.get(&node) {
            return level;
        }
        assert!(
            visiting.insert(node),
            "cycle detected while computing error-norm execution layers"
        );
        let mut max_dependency_level: Option<usize> = None;
        let mut has_input_dependency = false;
        let mut update_dep_level = |dep_level: usize| {
            max_dependency_level = Some(match max_dependency_level {
                Some(curr) => curr.max(dep_level),
                None => dep_level,
            });
        };
        match node {
            ErrorNormExecutionNodeId::Regular(gate_id) => {
                for &input_id in &self.gate(gate_id).input_gates {
                    if let Some(dep_node) = self.error_norm_node_for_gate(input_id) {
                        let dep_level = self.populate_error_norm_node_level(
                            dep_node,
                            node_levels,
                            input_set_levels,
                            visiting,
                            levels,
                        );
                        update_dep_level(dep_level);
                    } else {
                        has_input_dependency = true;
                    }
                }
            }
            ErrorNormExecutionNodeId::SubCircuitCall(call_id) => {
                let call = self.sub_circuit_calls.get(&call_id).expect("sub-circuit call missing");
                if let Some(input_set_id) = call.shared_input_prefix_set_id {
                    if let Some(shared_prefix_level) = self.populate_error_norm_input_set_max_level(
                        input_set_id,
                        node_levels,
                        input_set_levels,
                        visiting,
                        levels,
                    ) {
                        update_dep_level(shared_prefix_level);
                    }
                }
                for &input_id in &call.input_suffix {
                    if let Some(dep_node) = self.error_norm_node_for_gate(input_id) {
                        let dep_level = self.populate_error_norm_node_level(
                            dep_node,
                            node_levels,
                            input_set_levels,
                            visiting,
                            levels,
                        );
                        update_dep_level(dep_level);
                    } else {
                        has_input_dependency = true;
                    }
                }
            }
            ErrorNormExecutionNodeId::SummedSubCircuitCall(summed_call_id) => {
                let call = self
                    .summed_sub_circuit_calls
                    .get(&summed_call_id)
                    .expect("summed sub-circuit call missing");
                for &input_set_id in &call.call_input_set_ids {
                    if let Some(input_set_level) = self.populate_error_norm_input_set_max_level(
                        input_set_id,
                        node_levels,
                        input_set_levels,
                        visiting,
                        levels,
                    ) {
                        update_dep_level(input_set_level);
                    }
                }
            }
        }
        visiting.remove(&node);
        let max_dependency_level = if has_input_dependency {
            Some(max_dependency_level.map_or(0, |dep_level| dep_level))
        } else {
            max_dependency_level
        };
        let level = max_dependency_level.map_or(0, |dep_level| dep_level + 1);
        if levels.len() <= level {
            levels.resize(level + 1, ErrorNormExecutionLayer::default());
        }
        match node {
            ErrorNormExecutionNodeId::Regular(gate_id) => {
                levels[level].regular_gate_ids.push(gate_id)
            }
            ErrorNormExecutionNodeId::SubCircuitCall(call_id) => {
                levels[level].sub_circuit_call_ids.push(call_id)
            }
            ErrorNormExecutionNodeId::SummedSubCircuitCall(summed_call_id) => {
                levels[level].summed_sub_circuit_call_ids.push(summed_call_id)
            }
        }
        node_levels.insert(node, level);
        level
    }

    pub(crate) fn error_norm_execution_layers(&self) -> Vec<ErrorNormExecutionLayer> {
        let mut node_levels: HashMap<ErrorNormExecutionNodeId, usize> = HashMap::new();
        let mut input_set_levels: HashMap<usize, Option<usize>> = HashMap::new();
        let mut levels = Vec::<ErrorNormExecutionLayer>::new();
        let mut visiting = HashSet::new();
        for &output_gate in &self.output_ids {
            let Some(node) = self.error_norm_node_for_gate(output_gate) else {
                continue;
            };
            self.populate_error_norm_node_level(
                node,
                &mut node_levels,
                &mut input_set_levels,
                &mut visiting,
                &mut levels,
            );
        }
        levels
    }

    /// Computes a topological order (as a vector of gate IDs) for all gates that
    /// are needed to evaluate the outputs. This is done via a DFS from each output gate.
    fn topological_order(&self) -> Vec<GateId> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        let mut stack = Vec::new();
        for &output_gate in &self.output_ids {
            if visited.insert(output_gate) {
                stack.push((output_gate, 0));
            }
        }

        while let Some((node, child_idx)) = stack.pop() {
            let gate = self.gates.get(&node).expect("gate not found");
            let dependency_inputs = {
                let mut deps = Vec::with_capacity(self.gate_dependency_input_count(gate));
                self.for_each_gate_dependency_input(gate, |input_id| deps.push(input_id));
                deps
            };

            if child_idx < dependency_inputs.len() {
                stack.push((node, child_idx + 1));
                let child = dependency_inputs[child_idx];
                if visited.insert(child) {
                    stack.push((child, 0));
                }
            } else {
                order.push(node);
            }
        }

        order
    }

    /// Computes a levelized grouping of gate ids.
    /// Input wires (keys 0..=num_input) are assigned level 0.
    /// Each non‐input gate's level is defined as max(levels of its inputs) + 1.
    fn compute_levels(&self) -> Vec<Vec<GateId>> {
        let mut gate_levels: HashMap<GateId, usize> = HashMap::new();
        let mut levels: Vec<Vec<GateId>> = vec![];
        let orders = self.topological_order();
        for gate_id in orders.into_iter() {
            let gate = self.gates.get(&gate_id).expect("gate not found");
            let dependency_count = self.gate_dependency_input_count(gate);
            if dependency_count == 0 {
                // Inputs and consts have no dependencies; place them at level 0
                gate_levels.insert(gate_id, 0);
                if levels.is_empty() {
                    levels.push(vec![]);
                }
                levels[0].push(gate_id);
                continue;
            }
            // Find the maximum level among all input gates, then add 1.
            let mut max_input_level: Option<usize> = None;
            self.for_each_gate_dependency_input(gate, |input_id| {
                let level = gate_levels[&input_id];
                max_input_level = Some(max_input_level.map_or(level, |curr| curr.max(level)));
            });
            let max_input_level =
                max_input_level.expect("gate has dependencies but max() returned None");
            let level = max_input_level + 1;
            gate_levels.insert(gate_id, level);
            if levels.len() <= level {
                levels.resize(level + 1, vec![]);
            }
            levels[level].push(gate_id);
        }
        levels
    }

    pub(crate) fn execution_layers(&self) -> Vec<CircuitExecutionLayer> {
        self.compute_levels()
            .into_iter()
            .map(|level| {
                let mut seen_call_ids = HashSet::new();
                let mut seen_summed_call_ids = HashSet::new();
                let mut sub_circuit_call_ids = Vec::new();
                let mut summed_sub_circuit_call_ids = Vec::new();
                let mut regular_gate_types = Vec::new();
                for gate_id in level {
                    let gate = self.gates.get(&gate_id).expect("gate not found");
                    match &gate.gate_type {
                        PolyGateType::SubCircuitOutput { call_id, .. } => {
                            if seen_call_ids.insert(*call_id) {
                                sub_circuit_call_ids.push(*call_id);
                            }
                        }
                        PolyGateType::SummedSubCircuitOutput { summed_call_id, .. } => {
                            if seen_summed_call_ids.insert(*summed_call_id) {
                                summed_sub_circuit_call_ids.push(*summed_call_id);
                            }
                        }
                        _ => regular_gate_types.push(gate.gate_type.clone()),
                    }
                }
                CircuitExecutionLayer {
                    sub_circuit_call_ids,
                    summed_sub_circuit_call_ids,
                    regular_gate_types,
                }
            })
            .collect()
    }

    pub(crate) fn registered_sub_circuit(&self, circuit_id: usize) -> Self {
        self.with_sub_circuit(circuit_id, Clone::clone)
    }

    pub(crate) fn registered_sub_circuit_ref(&self, circuit_id: usize) -> Arc<Self> {
        let stored = self.sub_circuits.get(&circuit_id).expect("sub-circuit not found");
        match stored {
            StoredSubCircuit::InMemory(sub) => sub.clone(),
        }
    }

    /// Returns the circuit depth defined as the maximum level index among
    /// all gates required to compute the outputs.
    ///
    /// - Inputs and constant-one gate reside at level 0.
    /// - Each non-input gate is assigned level = max(input levels) + 1.
    /// - If there are no outputs, depth is 0.
    pub fn depth(&self) -> usize {
        let levels = self.compute_levels();
        if levels.is_empty() { 0 } else { levels.len() - 1 }
    }

    /// Evaluate the circuit using an iterative approach over a precomputed topological order.
    ///
    /// `helper_lookup` is P_{x_L}
    /// `parallel_gates` overrides the gate-level parallelism limit; `None` preserves the
    /// environment-driven default.
    pub fn eval<E, PE>(
        &self,
        params: &E::Params,
        one: E,
        inputs: Vec<E>,
        plt_evaluator: Option<&PE>,
        slot_transfer_evaluator: Option<&dyn SlotTransferEvaluator<E>>,
        parallel_gates: Option<usize>,
    ) -> Vec<E>
    where
        E: Evaluable<P = P>,
        PE: PltEvaluator<E>,
    {
        let (call_id_base, gate_id_base) = self.eval_gate_id_bases();
        let parallel_gates = crate::env::resolve_circuit_parallel_gates(parallel_gates);
        let one_compact = Arc::new(one.to_compact());
        let one = Arc::new(E::from_compact(params, one_compact.as_ref()));
        let input_compacts =
            inputs.into_iter().map(|input| Arc::new(input.to_compact())).collect::<Vec<_>>();
        let scoped_gate_ids = self.build_scoped_gate_id_map(call_id_base, gate_id_base);
        let outputs = self.eval_scoped(
            params,
            &one,
            one_compact,
            &input_compacts,
            &[],
            plt_evaluator,
            slot_transfer_evaluator,
            0,
            call_id_base,
            gate_id_base,
            &scoped_gate_ids,
            parallel_gates,
        );
        outputs.into_iter().map(|value| E::from_compact(params, value.as_ref())).collect()
    }

    fn eval_gate_id_bases(&self) -> (u128, u128) {
        let max_calls = self.max_sub_circuit_calls();
        let max_gates = self.max_gate_count();
        ((max_calls as u128) + 1, (max_gates as u128) + 1)
    }

    fn max_sub_circuit_calls(&self) -> usize {
        let mut max_calls = self.next_scoped_call_id;
        for sub_id in self.sub_circuits.keys().copied() {
            let sub_max_calls = self.with_sub_circuit(sub_id, |sub| sub.max_sub_circuit_calls());
            max_calls = max_calls.max(sub_max_calls);
        }
        max_calls
    }

    fn max_gate_count(&self) -> usize {
        let mut max_gates = self.gates.len();
        for sub_id in self.sub_circuits.keys().copied() {
            let sub_max_gates = self.with_sub_circuit(sub_id, |sub| sub.max_gate_count());
            max_gates = max_gates.max(sub_max_gates);
        }
        max_gates
    }

    fn collect_scoped_gate_keys(
        &self,
        call_prefix: u128,
        call_id_base: u128,
        gate_id_base: u128,
        scoped_keys: &mut BTreeSet<u128>,
    ) {
        for gate in self.gates.values() {
            match &gate.gate_type {
                PolyGateType::SlotTransfer { .. } | PolyGateType::PubLut { .. } => {
                    let scoped_key = call_prefix
                        .checked_mul(gate_id_base)
                        .and_then(|base| base.checked_add(gate.gate_id.0 as u128))
                        .expect("scoped gate key overflow");
                    scoped_keys.insert(scoped_key);
                }
                _ => {}
            }
        }
        for call in self.sub_circuit_calls.values() {
            let child_prefix = call_prefix
                .checked_mul(call_id_base)
                .and_then(|base| base.checked_add((call.scoped_call_id as u128) + 1))
                .expect("sub-circuit call prefix overflow");
            self.with_sub_circuit(call.sub_circuit_id, |sub| {
                sub.collect_scoped_gate_keys(child_prefix, call_id_base, gate_id_base, scoped_keys);
            });
        }
        for call in self.summed_sub_circuit_calls.values() {
            for scoped_call_id in &call.scoped_call_ids {
                let child_prefix = call_prefix
                    .checked_mul(call_id_base)
                    .and_then(|base| base.checked_add((*scoped_call_id as u128) + 1))
                    .expect("summed sub-circuit call prefix overflow");
                self.with_sub_circuit(call.sub_circuit_id, |sub| {
                    sub.collect_scoped_gate_keys(
                        child_prefix,
                        call_id_base,
                        gate_id_base,
                        scoped_keys,
                    );
                });
            }
        }
    }

    fn build_scoped_gate_id_map(
        &self,
        call_id_base: u128,
        gate_id_base: u128,
    ) -> HashMap<u128, GateId> {
        let mut scoped_keys = BTreeSet::new();
        self.collect_scoped_gate_keys(0, call_id_base, gate_id_base, &mut scoped_keys);

        let mut scoped_gate_ids = HashMap::with_capacity(scoped_keys.len());
        let mut max_direct_gate_id = 0usize;
        let mut overflow_keys = Vec::new();
        for scoped_key in scoped_keys {
            if scoped_key <= usize::MAX as u128 {
                let gate_id = GateId(scoped_key as usize);
                max_direct_gate_id = max_direct_gate_id.max(gate_id.0);
                scoped_gate_ids.insert(scoped_key, gate_id);
            } else {
                overflow_keys.push(scoped_key);
            }
        }

        let mut next_overflow_gate_id =
            max_direct_gate_id.checked_add(1).expect("scoped gate id remap overflow");
        for scoped_key in overflow_keys {
            let gate_id = GateId(next_overflow_gate_id);
            scoped_gate_ids.insert(scoped_key, gate_id);
            next_overflow_gate_id =
                next_overflow_gate_id.checked_add(1).expect("scoped gate id remap overflow");
        }

        scoped_gate_ids
    }

    fn scoped_gate_id(
        scoped_gate_ids: &HashMap<u128, GateId>,
        call_prefix: u128,
        gate_id: GateId,
        gate_id_base: u128,
    ) -> GateId {
        let scoped_key = call_prefix
            .checked_mul(gate_id_base)
            .and_then(|base| base.checked_add(gate_id.0 as u128))
            .expect("scoped gate key overflow");
        *scoped_gate_ids.get(&scoped_key).expect("missing precomputed scoped gate id")
    }

    fn eval_scoped<E, PE>(
        &self,
        params: &E::Params,
        one: &Arc<E>,
        one_compact: Arc<E::Compact>,
        inputs: &[Arc<E::Compact>],
        param_bindings: &[SubCircuitParamValue],
        plt_evaluator: Option<&PE>,
        slot_transfer_evaluator: Option<&dyn SlotTransferEvaluator<E>>,
        call_prefix: u128,
        call_id_base: u128,
        gate_id_base: u128,
        scoped_gate_ids: &HashMap<u128, GateId>,
        parallel_gates: Option<usize>,
    ) -> Vec<Arc<E::Compact>>
    where
        E: Evaluable<P = P>,
        PE: PltEvaluator<E>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.num_input(), inputs.len());
            assert_ne!(self.num_output(), 0);
        }

        let wires: DashMap<GateId, Arc<E::Compact>> = DashMap::new();
        let levels = self.compute_levels();
        debug!("{}", format!("Levels: {levels:?}"));
        debug!("Levels are computed");
        let output_set: HashSet<GateId> = self.output_ids.iter().copied().collect();
        // Count remaining input uses per wire so we can release each input wire
        // immediately after its final consumer gate has been evaluated.
        let use_count_by_gate: HashMap<GateId, usize> = levels
            .par_iter()
            .map(|level| {
                let mut local_count = HashMap::<GateId, usize>::new();
                for gate_id in level {
                    let gate = self.gates.get(gate_id).expect("gate not found");
                    self.for_each_gate_dependency_input(gate, |input_id| {
                        *local_count.entry(input_id).or_insert(0) += 1;
                    });
                }
                local_count
            })
            .reduce(HashMap::new, |mut acc, local| {
                for (gate_id, count) in local {
                    *acc.entry(gate_id).or_insert(0) += count;
                }
                acc
            });
        let remaining_use_count: HashMap<GateId, AtomicUsize> = use_count_by_gate
            .into_iter()
            .map(|(gate_id, count)| (gate_id, AtomicUsize::new(count)))
            .collect();
        debug!("Initialized remaining-use counters for {} wires", remaining_use_count.len());

        wires.insert(GateId(0), one_compact.clone());
        debug!("Constant one gate is set");
        // Collect all input gate IDs excluding the reserved constant-one gate (0)
        let mut input_gate_ids: Vec<GateId> = self
            .gates
            .iter()
            .filter_map(|(id, gate)| match &gate.gate_type {
                PolyGateType::Input if id.0 != 0 => Some(*id),
                _ => None,
            })
            .collect();
        input_gate_ids.sort_by_key(|gid| gid.0);
        debug!("input_gate_ids size {}", input_gate_ids.len());
        assert_eq!(
            input_gate_ids.len(),
            inputs.len(),
            "number of provided inputs must match circuit inputs"
        );
        for (id, input) in input_gate_ids.into_iter().zip(inputs.iter()) {
            wires.insert(id, Arc::clone(input));
            if let Some(prefix) = self.print_value.get(&id) {
                let decoded_input = E::from_compact(params, input.as_ref());
                info!("{}", format!("[{prefix}] Gate ID {id}, {:?}", decoded_input));
            }
        }
        debug!("Input wires are set");

        let use_parallel = parallel_gates.map(|n| n != 1).unwrap_or(true);
        #[cfg(feature = "gpu")]
        let shard_params_and_one: Vec<(E::Params, Arc<E>)> = {
            let mut device_ids = detected_gpu_device_ids();
            if device_ids.is_empty() {
                device_ids.push(0);
            }
            device_ids
                .into_iter()
                .map(|device_id| {
                    let local_params = E::params_for_eval_device(params, device_id);
                    let local_one = Arc::new(E::from_compact(&local_params, one_compact.as_ref()));
                    (local_params, local_one)
                })
                .collect()
        };
        let release_consumed_inputs = |gate: &PolyGate| {
            self.for_each_gate_dependency_input(gate, |input_id| {
                if output_set.contains(&input_id) {
                    return;
                }
                let Some(counter) = remaining_use_count.get(&input_id) else {
                    return;
                };
                let prev = counter.fetch_sub(1, Ordering::AcqRel);
                debug_assert!(prev > 0, "remaining use counter underflow for gate {}", input_id);
                if prev == 1 {
                    wires.remove(&input_id);
                }
            });
        };
        let eval_gate = |gate_id: GateId, eval_params: &E::Params, eval_one: &Arc<E>| {
            debug!("{}", format!("Gate id {gate_id} started"));
            let gate = self.gates.get(&gate_id).expect("gate not found").clone();
            if wires.contains_key(&gate_id) {
                debug!("{}", format!("Gate id {gate_id} already evaluated"));
                release_consumed_inputs(&gate);
                return;
            }
            debug!("Get gate");
            let result: Arc<E::Compact> = match &gate.gate_type {
                PolyGateType::Input => {
                    panic!("Input gate {gate:?} should already be preloaded");
                }
                PolyGateType::Add => {
                    debug!("Add gate start");
                    let left =
                        wires.get(&gate.input_gates[0]).expect("wire missing for Add").clone();
                    let right =
                        wires.get(&gate.input_gates[1]).expect("wire missing for Add").clone();
                    let left = E::from_compact(eval_params, left.as_ref());
                    let right = E::from_compact(eval_params, right.as_ref());
                    let result = left + &right;
                    debug!("Add gate end");
                    Arc::new(result.to_compact())
                }
                PolyGateType::Sub => {
                    debug!("Sub gate start");
                    let left =
                        wires.get(&gate.input_gates[0]).expect("wire missing for Sub").clone();
                    let right =
                        wires.get(&gate.input_gates[1]).expect("wire missing for Sub").clone();
                    let left = E::from_compact(eval_params, left.as_ref());
                    let right = E::from_compact(eval_params, right.as_ref());
                    let result = left - &right;
                    debug!("Sub gate end");
                    Arc::new(result.to_compact())
                }
                PolyGateType::Mul => {
                    debug!("Mul gate start");
                    let left =
                        wires.get(&gate.input_gates[0]).expect("wire missing for Mul").clone();
                    let right =
                        wires.get(&gate.input_gates[1]).expect("wire missing for Mul").clone();
                    let left = E::from_compact(eval_params, left.as_ref());
                    let right = E::from_compact(eval_params, right.as_ref());
                    let result = left * &right;
                    debug!("Mul gate end");
                    Arc::new(result.to_compact())
                }
                PolyGateType::SmallScalarMul { scalar } => {
                    let scalar = scalar.resolve_small_scalar(param_bindings);
                    let input = wires
                        .get(&gate.input_gates[0])
                        .expect("wire missing for LargeScalarMul")
                        .clone();
                    let input = E::from_compact(eval_params, input.as_ref());
                    let result = input.small_scalar_mul(eval_params, scalar);
                    debug!("Large scalar mul gate end");
                    Arc::new(result.to_compact())
                }
                PolyGateType::LargeScalarMul { scalar } => {
                    let scalar = scalar.resolve_large_scalar(param_bindings);
                    let input = wires
                        .get(&gate.input_gates[0])
                        .expect("wire missing for LargeScalarMul")
                        .clone();
                    let input = E::from_compact(eval_params, input.as_ref());
                    let result = input.large_scalar_mul(eval_params, scalar);
                    debug!("Large scalar mul gate end");
                    Arc::new(result.to_compact())
                }
                PolyGateType::SlotTransfer { src_slots } => {
                    let src_slots = src_slots.resolve_slot_transfer(param_bindings);
                    let input = wires
                        .get(&gate.input_gates[0])
                        .expect("wire missing for SlotTransfer")
                        .clone();
                    let input = E::from_compact(eval_params, input.as_ref());
                    let evaluator =
                        slot_transfer_evaluator.expect("slot transfer evaluator missing");
                    let scoped_gate_id =
                        Self::scoped_gate_id(scoped_gate_ids, call_prefix, gate_id, gate_id_base);
                    Arc::new(
                        evaluator
                            .slot_transfer(eval_params, &input, src_slots.as_ref(), scoped_gate_id)
                            .to_compact(),
                    )
                }
                PolyGateType::PubLut { lut_id } => {
                    let lut_id = lut_id.resolve_public_lookup(param_bindings);
                    debug!("Public Lookup gate start");
                    let input = wires
                        .get(&gate.input_gates[0])
                        .expect("wire missing for Public Lookup")
                        .clone();
                    let input = E::from_compact(eval_params, input.as_ref());
                    let scoped_gate_id =
                        Self::scoped_gate_id(scoped_gate_ids, call_prefix, gate_id, gate_id_base);
                    let lookup_guard =
                        self.lookup_registry.lookups.get(&lut_id).expect("lookup table missing");
                    let result =
                        plt_evaluator.expect("public lookup evaluator missing").public_lookup(
                            eval_params,
                            lookup_guard.as_ref(),
                            eval_one,
                            &input,
                            scoped_gate_id,
                            lut_id,
                        );
                    debug!("Public Lookup gate end");
                    Arc::new(result.to_compact())
                }
                PolyGateType::SubCircuitOutput { call_id, output_idx, .. } => {
                    debug!("Sub-circuit call start");
                    let call = self
                        .sub_circuit_calls
                        .get(call_id)
                        .expect("sub-circuit call missing")
                        .clone();
                    let param_bindings = self.binding_set(call.binding_set_id);
                    let child_prefix = call_prefix
                        .checked_mul(call_id_base)
                        .and_then(|base| base.checked_add((call.scoped_call_id as u128) + 1))
                        .expect("sub-circuit call prefix overflow");
                    let sub_inputs =
                        self.with_sub_circuit_call_inputs(&call, |shared_prefix, suffix| {
                            let mut inputs = Vec::with_capacity(shared_prefix.len() + suffix.len());
                            inputs.extend(shared_prefix.iter().map(|id| {
                                wires.get(id).expect("wire missing for sub-circuit").clone()
                            }));
                            inputs.extend(suffix.iter().map(|id| {
                                wires.get(id).expect("wire missing for sub-circuit").clone()
                            }));
                            inputs
                        });
                    let sub_outputs = self.with_sub_circuit(call.sub_circuit_id, |sub_circuit| {
                        sub_circuit.eval_scoped(
                            eval_params,
                            eval_one,
                            one_compact.clone(),
                            &sub_inputs,
                            param_bindings.as_ref(),
                            plt_evaluator,
                            slot_transfer_evaluator,
                            child_prefix,
                            call_id_base,
                            gate_id_base,
                            scoped_gate_ids,
                            parallel_gates,
                        )
                    });
                    if sub_outputs.len() != call.output_gate_ids.len() {
                        panic!("sub-circuit output size mismatch");
                    }
                    for (gate_id, value) in
                        call.output_gate_ids.iter().copied().zip(sub_outputs.iter().cloned())
                    {
                        wires.insert(gate_id, value);
                    }
                    debug!("Sub-circuit call end");
                    sub_outputs[*output_idx].clone()
                }
                PolyGateType::SummedSubCircuitOutput { summed_call_id, output_idx: _, .. } => {
                    debug!("Summed sub-circuit call start");
                    let call = self
                        .summed_sub_circuit_calls
                        .get(summed_call_id)
                        .expect("summed sub-circuit call missing")
                        .clone();
                    let mut accumulated: Option<Vec<E>> = None;
                    for ((input_set_id, binding_set_id), scoped_call_id) in call
                        .call_input_set_ids
                        .iter()
                        .zip(call.call_binding_set_ids.iter())
                        .zip(call.scoped_call_ids.iter())
                    {
                        let bound_params = self.binding_set(*binding_set_id);
                        let child_prefix = call_prefix
                            .checked_mul(call_id_base)
                            .and_then(|base| base.checked_add((*scoped_call_id as u128) + 1))
                            .expect("summed sub-circuit call prefix overflow");
                        let sub_inputs = self
                            .input_set(*input_set_id)
                            .iter()
                            .map(|id| {
                                wires.get(id).expect("wire missing for summed sub-circuit").clone()
                            })
                            .collect::<Vec<_>>();
                        let sub_outputs =
                            self.with_sub_circuit(call.sub_circuit_id, |sub_circuit| {
                                sub_circuit.eval_scoped(
                                    eval_params,
                                    eval_one,
                                    one_compact.clone(),
                                    &sub_inputs,
                                    bound_params.as_ref(),
                                    plt_evaluator,
                                    slot_transfer_evaluator,
                                    child_prefix,
                                    call_id_base,
                                    gate_id_base,
                                    scoped_gate_ids,
                                    parallel_gates,
                                )
                            });
                        let decoded_outputs = sub_outputs
                            .into_iter()
                            .map(|value| E::from_compact(eval_params, value.as_ref()))
                            .collect::<Vec<_>>();
                        match accumulated.as_mut() {
                            Some(current) => {
                                for (acc, output) in
                                    current.iter_mut().zip(decoded_outputs.into_iter())
                                {
                                    *acc = acc.clone() + &output;
                                }
                            }
                            None => accumulated = Some(decoded_outputs),
                        }
                    }
                    let accumulated = accumulated
                        .expect("summed sub-circuit call requires at least one inner call");
                    for (output_gate_id, output) in
                        call.output_gate_ids.iter().copied().zip(accumulated.into_iter())
                    {
                        wires.insert(output_gate_id, Arc::new(output.to_compact()));
                    }
                    debug!("Summed sub-circuit call end");
                    wires
                        .get(&gate_id)
                        .expect("summed sub-circuit output should be populated")
                        .clone()
                }
            };
            if let Some(prefix) = self.print_value.get(&gate_id) {
                let decoded_result = E::from_compact(eval_params, result.as_ref());
                info!("{}", format!("[{prefix}] Gate ID {gate_id}, {:?}", decoded_result));
            }
            wires.insert(gate_id, result);
            release_consumed_inputs(&gate);
            debug!("{}", format!("Gate id {gate_id} finished"));
        };
        #[cfg(feature = "gpu")]
        let load_chunk = |chunk: &[GateId]| -> Vec<LoadedGateCtx<E>> {
            chunk
                .par_iter()
                .enumerate()
                .map(|(slot, gate_id)| {
                    let shard_idx = slot % shard_params_and_one.len();
                    let (eval_params, _) = &shard_params_and_one[shard_idx];
                    let gate_id = *gate_id;
                    let gate = self.gates.get(&gate_id).expect("gate not found").clone();
                    if wires.contains_key(&gate_id) {
                        return LoadedGateCtx {
                            gate_id,
                            gate,
                            shard_idx,
                            inputs: LoadedGateInputs::SkipExisting,
                        };
                    }
                    let inputs = match &gate.gate_type {
                        PolyGateType::Input => LoadedGateInputs::SkipExisting,
                        PolyGateType::Add | PolyGateType::Sub | PolyGateType::Mul => {
                            let left = wires
                                .get(&gate.input_gates[0])
                                .expect("wire missing for binary gate")
                                .clone();
                            let right = wires
                                .get(&gate.input_gates[1])
                                .expect("wire missing for binary gate")
                                .clone();
                            let left = E::from_compact(eval_params, left.as_ref());
                            let right = E::from_compact(eval_params, right.as_ref());
                            LoadedGateInputs::Binary(left, right)
                        }
                        PolyGateType::SmallScalarMul { .. } |
                        PolyGateType::LargeScalarMul { .. } |
                        PolyGateType::SlotTransfer { .. } |
                        PolyGateType::PubLut { .. } => {
                            let input = wires
                                .get(&gate.input_gates[0])
                                .expect("wire missing for unary gate")
                                .clone();
                            let input = E::from_compact(eval_params, input.as_ref());
                            LoadedGateInputs::Unary(input)
                        }
                        PolyGateType::SubCircuitOutput { .. } |
                        PolyGateType::SummedSubCircuitOutput { .. } => {
                            panic!("sub-circuit output gate should not be in regular chunk path");
                        }
                    };
                    LoadedGateCtx { gate_id, gate, shard_idx, inputs }
                })
                .collect()
        };
        #[cfg(feature = "gpu")]
        let compute_chunk = |loaded_chunk: Vec<LoadedGateCtx<E>>| -> Vec<ComputedGateCtx<E>> {
            loaded_chunk
                .into_par_iter()
                .map(|loaded| {
                    let LoadedGateCtx { gate_id, gate, shard_idx, inputs } = loaded;
                    let (eval_params, eval_one) = &shard_params_and_one[shard_idx];
                    let value = match (inputs, &gate.gate_type) {
                        (LoadedGateInputs::SkipExisting, _) => ComputedGateValue::SkipExisting,
                        (LoadedGateInputs::Binary(left, right), PolyGateType::Add) => {
                            ComputedGateValue::Value(left + &right)
                        }
                        (LoadedGateInputs::Binary(left, right), PolyGateType::Sub) => {
                            ComputedGateValue::Value(left - &right)
                        }
                        (LoadedGateInputs::Binary(left, right), PolyGateType::Mul) => {
                            ComputedGateValue::Value(left * &right)
                        }
                        (
                            LoadedGateInputs::Unary(input),
                            PolyGateType::SmallScalarMul { scalar },
                        ) => ComputedGateValue::Value(input.small_scalar_mul(
                            eval_params,
                            scalar.resolve_small_scalar(param_bindings),
                        )),
                        (
                            LoadedGateInputs::Unary(input),
                            PolyGateType::LargeScalarMul { scalar },
                        ) => ComputedGateValue::Value(input.large_scalar_mul(
                            eval_params,
                            scalar.resolve_large_scalar(param_bindings),
                        )),
                        (
                            LoadedGateInputs::Unary(input),
                            PolyGateType::SlotTransfer { src_slots },
                        ) => {
                            let src_slots = src_slots.resolve_slot_transfer(param_bindings);
                            let evaluator =
                                slot_transfer_evaluator.expect("slot transfer evaluator missing");
                            let scoped_gate_id = Self::scoped_gate_id(
                                scoped_gate_ids,
                                call_prefix,
                                gate_id,
                                gate_id_base,
                            );
                            ComputedGateValue::Value(evaluator.slot_transfer(
                                eval_params,
                                &input,
                                src_slots.as_ref(),
                                scoped_gate_id,
                            ))
                        }
                        (LoadedGateInputs::Unary(input), PolyGateType::PubLut { lut_id }) => {
                            let lut_id = lut_id.resolve_public_lookup(param_bindings);
                            let scoped_gate_id = Self::scoped_gate_id(
                                scoped_gate_ids,
                                call_prefix,
                                gate_id,
                                gate_id_base,
                            );
                            let lookup_guard = self
                                .lookup_registry
                                .lookups
                                .get(&lut_id)
                                .expect("lookup table missing");
                            let result = plt_evaluator
                                .expect("public lookup evaluator missing")
                                .public_lookup(
                                    eval_params,
                                    lookup_guard.as_ref(),
                                    eval_one,
                                    &input,
                                    scoped_gate_id,
                                    lut_id,
                                );
                            ComputedGateValue::Value(result)
                        }
                        _ => {
                            panic!("loaded gate inputs do not match gate type for gate {}", gate_id)
                        }
                    };
                    ComputedGateCtx { gate_id, gate, shard_idx, value }
                })
                .collect()
        };
        #[cfg(feature = "gpu")]
        let store_chunk = |computed_chunk: Vec<ComputedGateCtx<E>>| {
            computed_chunk.into_par_iter().for_each(|computed| {
                let ComputedGateCtx { gate_id, gate, shard_idx, value } = computed;
                match value {
                    ComputedGateValue::SkipExisting => {
                        release_consumed_inputs(&gate);
                        debug!("{}", format!("Gate id {gate_id} finished"));
                    }
                    ComputedGateValue::Value(result) => {
                        let (eval_params, _) = &shard_params_and_one[shard_idx];
                        let compact = Arc::new(result.to_compact());
                        if let Some(prefix) = self.print_value.get(&gate_id) {
                            let decoded_result = E::from_compact(eval_params, compact.as_ref());
                            info!(
                                "{}",
                                format!("[{prefix}] Gate ID {gate_id}, {:?}", decoded_result)
                            );
                        }
                        wires.insert(gate_id, compact);
                        release_consumed_inputs(&gate);
                        debug!("{}", format!("Gate id {gate_id} finished"));
                    }
                }
            });
        };
        for (level_idx, level) in levels.iter().enumerate() {
            let lookup_gate_count = level
                .iter()
                .filter(|gate_id| {
                    matches!(
                        self.gates.get(gate_id).expect("gate not found").gate_type,
                        PolyGateType::PubLut { .. }
                    )
                })
                .count();
            debug!(
                "Level {}: gates={}, lookup_gates={}",
                level_idx,
                level.len(),
                lookup_gate_count
            );
            debug!("New level started");
            let mut subcircuit_gates = Vec::new();
            let mut regular_gates = Vec::new();
            for gate_id in level.iter().copied() {
                match self.gates.get(&gate_id).expect("gate not found").gate_type {
                    PolyGateType::SubCircuitOutput { .. } |
                    PolyGateType::SummedSubCircuitOutput { .. } => subcircuit_gates.push(gate_id),
                    _ => regular_gates.push(gate_id),
                }
            }
            if !subcircuit_gates.is_empty() {
                #[cfg(feature = "gpu")]
                {
                    let (eval_params, eval_one) = shard_params_and_one
                        .first()
                        .expect("at least one eval shard context required");
                    subcircuit_gates
                        .iter()
                        .copied()
                        .for_each(|gate_id| eval_gate(gate_id, eval_params, eval_one));
                }
                #[cfg(not(feature = "gpu"))]
                {
                    subcircuit_gates
                        .iter()
                        .copied()
                        .for_each(|gate_id| eval_gate(gate_id, params, one));
                }
                debug!("Evaluated sub-circuit gates in single thread");
            }
            if let Some(chunk_size) = parallel_gates {
                #[cfg(feature = "gpu")]
                {
                    let regular_chunks: Vec<&[GateId]> = regular_gates.chunks(chunk_size).collect();
                    if !regular_chunks.is_empty() {
                        let mut loaded_curr = Some(load_chunk(regular_chunks[0]));
                        let mut computed_prev: Option<Vec<ComputedGateCtx<E>>> = None;
                        for chunk_idx in 0..regular_chunks.len() {
                            let to_store = computed_prev.take();
                            let to_compute =
                                loaded_curr.take().expect("loaded chunk missing in pipeline");
                            let next_chunk = regular_chunks.get(chunk_idx + 1).copied();
                            let ((), (computed_curr, loaded_next)) = rayon::join(
                                || {
                                    if let Some(computed_chunk) = to_store {
                                        store_chunk(computed_chunk);
                                    }
                                },
                                || {
                                    rayon::join(
                                        || compute_chunk(to_compute),
                                        || next_chunk.map(load_chunk),
                                    )
                                },
                            );
                            computed_prev = Some(computed_curr);
                            loaded_curr = loaded_next;
                        }
                        if let Some(last_chunk) = computed_prev.take() {
                            store_chunk(last_chunk);
                        }
                    }
                }
                #[cfg(not(feature = "gpu"))]
                {
                    regular_gates.chunks(chunk_size).for_each(|chunk| {
                        chunk
                            .par_iter()
                            .copied()
                            .for_each(|gate_id| eval_gate(gate_id, params, one));
                    });
                }
            } else if use_parallel {
                regular_gates
                    .par_iter()
                    .copied()
                    .for_each(|gate_id| eval_gate(gate_id, params, one));
                debug!("Evaluated gate in parallel");
            } else {
                regular_gates.iter().copied().for_each(|gate_id| eval_gate(gate_id, params, one));
                debug!("Evaluated gate in single thread");
            }
        }

        let outputs: Vec<Arc<E::Compact>> = if use_parallel {
            if let Some(chunk_size) = parallel_gates {
                let mut out: Vec<Arc<E::Compact>> = Vec::with_capacity(self.output_ids.len());
                for chunk in self.output_ids.chunks(chunk_size) {
                    let mut chunk_out: Vec<Arc<E::Compact>> = chunk
                        .par_iter()
                        .map(|&id| wires.get(&id).expect("output missing").clone())
                        .collect();
                    out.append(&mut chunk_out);
                }
                out
            } else {
                self.output_ids
                    .par_iter()
                    .map(|&id| wires.get(&id).expect("output missing").clone())
                    .collect()
            }
        } else {
            self.output_ids
                .iter()
                .map(|&id| wires.get(&id).expect("output missing").clone())
                .collect()
        };
        debug!("Outputs are collected");
        outputs
    }

    pub fn register_public_lookup(&mut self, public_lookup: PublicLut<P>) -> usize {
        if !self.allow_register_lookup {
            panic!("lookup table registration is only allowed on top-level circuits");
        }
        self.lookup_registry.register(public_lookup)
    }

    pub fn lut_vector_len_with_subcircuits(&self) -> usize {
        self.lut_vector_len_with_subcircuits_and_bindings(&[])
    }

    fn lut_vector_len_with_subcircuits_and_bindings(
        &self,
        param_bindings: &[SubCircuitParamValue],
    ) -> usize {
        let mut total = 0usize;
        for gate in self.gates.values() {
            if let PolyGateType::PubLut { lut_id } = &gate.gate_type {
                let lut_id = lut_id.resolve_public_lookup(param_bindings);
                let lookup =
                    self.lookup_registry.lookups.get(&lut_id).expect("lookup table missing");
                total += lookup.len();
            }
        }
        for call in self.sub_circuit_calls.values() {
            let param_bindings = self.binding_set(call.binding_set_id);
            total += self.with_sub_circuit(call.sub_circuit_id, |sub| {
                sub.lut_vector_len_with_subcircuits_and_bindings(param_bindings.as_ref())
            });
        }
        for call in self.summed_sub_circuit_calls.values() {
            for binding_set_id in &call.call_binding_set_ids {
                let param_bindings = self.binding_set(*binding_set_id);
                total += self.with_sub_circuit(call.sub_circuit_id, |sub| {
                    sub.lut_vector_len_with_subcircuits_and_bindings(param_bindings.as_ref())
                });
            }
        }
        total
    }

    pub fn enable_subcircuits_in_disk(&mut self, dir_path: impl AsRef<Path>) {
        let storage = SubCircuitDiskStorage::new(dir_path.as_ref());
        self.sub_circuit_disk_storage = Some(storage);
    }

    pub fn register_sub_circuit(&mut self, mut sub_circuit: Self) -> usize {
        sub_circuit.inherit_registries_from_parent(self);
        let circuit_id = self.sub_circuits.len();
        self.sub_circuits.insert(circuit_id, StoredSubCircuit::InMemory(Arc::new(sub_circuit)));
        circuit_id
    }

    pub fn register_shared_sub_circuit(&mut self, sub_circuit: Arc<Self>) -> usize {
        let circuit_id = self.sub_circuits.len();
        self.sub_circuits.insert(circuit_id, StoredSubCircuit::InMemory(sub_circuit));
        circuit_id
    }

    pub fn call_sub_circuit(&mut self, circuit_id: usize, inputs: &[GateId]) -> Vec<GateId> {
        self.call_sub_circuit_with_bindings(circuit_id, inputs, &[])
    }

    pub fn call_sub_circuit_with_shared_input_prefix_and_bindings(
        &mut self,
        circuit_id: usize,
        shared_input_prefix_set_id: usize,
        input_suffix: &[GateId],
        param_bindings: &[SubCircuitParamValue],
    ) -> Vec<GateId> {
        self.call_sub_circuit_with_prefix_and_bindings(
            circuit_id,
            Some(shared_input_prefix_set_id),
            input_suffix,
            param_bindings,
        )
    }

    pub fn call_sub_circuit_with_bindings(
        &mut self,
        circuit_id: usize,
        inputs: &[GateId],
        param_bindings: &[SubCircuitParamValue],
    ) -> Vec<GateId> {
        self.call_sub_circuit_with_prefix_and_bindings(circuit_id, None, inputs, param_bindings)
    }

    fn call_sub_circuit_with_prefix_and_bindings(
        &mut self,
        circuit_id: usize,
        shared_input_prefix_set_id: Option<usize>,
        input_suffix: &[GateId],
        param_bindings: &[SubCircuitParamValue],
    ) -> Vec<GateId> {
        let total_num_inputs = shared_input_prefix_set_id
            .map(|input_set_id| self.input_set(input_set_id).len())
            .unwrap_or(0) +
            input_suffix.len();
        #[cfg(debug_assertions)]
        {
            let stored = self.sub_circuits.get(&circuit_id).expect("sub-circuit not found");
            let num_inputs = match stored {
                StoredSubCircuit::InMemory(sub) => sub.num_input(),
            };
            assert_eq!(total_num_inputs, num_inputs);
            let expected_param_kinds = match stored {
                StoredSubCircuit::InMemory(sub) => &sub.sub_circuit_params,
            };
            assert_eq!(param_bindings.len(), expected_param_kinds.len());
            for (param_idx, (binding, expected_kind)) in
                param_bindings.iter().zip(expected_param_kinds.iter()).enumerate()
            {
                assert_eq!(
                    binding.kind(),
                    *expected_kind,
                    "sub-circuit parameter kind mismatch at binding {param_idx}"
                );
            }
        }
        let num_outputs = self.sub_circuit_num_output(circuit_id);
        let call_id = self.sub_circuit_calls.len();
        let binding_set_id = self.binding_registry.register(param_bindings);
        let scoped_call_id = self.next_scoped_call_id;
        self.next_scoped_call_id += 1;
        let mut outputs = Vec::with_capacity(num_outputs);
        for output_idx in 0..num_outputs {
            outputs.push(self.new_sub_circuit_output_gate(call_id, output_idx, total_num_inputs));
        }
        let call = SubCircuitCall {
            sub_circuit_id: circuit_id,
            shared_input_prefix_set_id,
            input_suffix: input_suffix.to_vec(),
            binding_set_id,
            scoped_call_id,
            output_gate_ids: outputs.clone(),
            num_outputs,
        };
        self.sub_circuit_calls.insert(call_id, call);
        outputs
    }

    pub fn call_sub_circuit_sum_many_with_binding_set_ids(
        &mut self,
        circuit_id: usize,
        call_input_set_ids: Vec<usize>,
        call_binding_set_ids: Vec<usize>,
    ) -> Vec<GateId> {
        assert!(
            !call_input_set_ids.is_empty(),
            "summed sub-circuit call requires at least one inner call"
        );
        assert_eq!(
            call_input_set_ids.len(),
            call_binding_set_ids.len(),
            "summed sub-circuit call requires one binding set per inner call"
        );
        #[cfg(debug_assertions)]
        {
            let stored = self.sub_circuits.get(&circuit_id).expect("sub-circuit not found");
            let (num_inputs, expected_param_kinds) = match stored {
                StoredSubCircuit::InMemory(sub) => (sub.num_input(), &sub.sub_circuit_params),
            };
            for (call_idx, (input_set_id, binding_set_id)) in
                call_input_set_ids.iter().zip(call_binding_set_ids.iter()).enumerate()
            {
                let inputs = self.input_set(*input_set_id);
                assert_eq!(
                    inputs.len(),
                    num_inputs,
                    "summed sub-circuit input count mismatch at inner call {call_idx}"
                );
                let bindings = self.binding_set(*binding_set_id);
                assert_eq!(
                    bindings.len(),
                    expected_param_kinds.len(),
                    "summed sub-circuit parameter count mismatch at inner call {call_idx}"
                );
                for (param_idx, (binding, expected_kind)) in
                    bindings.iter().zip(expected_param_kinds.iter()).enumerate()
                {
                    assert_eq!(
                        binding.kind(),
                        *expected_kind,
                        "summed sub-circuit parameter kind mismatch at inner call {call_idx}, binding {param_idx}"
                    );
                }
            }
        }
        let num_outputs = self.sub_circuit_num_output(circuit_id);
        let summed_call_id = self.summed_sub_circuit_calls.len();
        let scoped_call_ids = (0..call_input_set_ids.len())
            .map(|_| {
                let scoped_call_id = self.next_scoped_call_id;
                self.next_scoped_call_id += 1;
                scoped_call_id
            })
            .collect::<Vec<_>>();
        let flattened_num_inputs =
            call_input_set_ids.iter().map(|input_set_id| self.input_set(*input_set_id).len()).sum();
        let output_gate_ids = (0..num_outputs)
            .map(|output_idx| {
                self.new_summed_sub_circuit_output_gate(
                    summed_call_id,
                    output_idx,
                    flattened_num_inputs,
                )
            })
            .collect::<Vec<_>>();
        self.summed_sub_circuit_calls.insert(
            summed_call_id,
            SummedSubCircuitCall {
                sub_circuit_id: circuit_id,
                call_input_set_ids,
                call_binding_set_ids,
                scoped_call_ids,
                output_gate_ids: output_gate_ids.clone(),
                num_outputs,
            },
        );
        output_gate_ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        element::PolyElem,
        lookup::{PltEvaluator, poly::PolyPltEvaluator},
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        rlwe_enc::rlwe_encrypt,
        sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        utils::{create_bit_random_poly, create_random_poly},
    };
    use num_bigint::BigUint;
    fn eval_with_const_one<PE>(
        circuit: &PolyCircuit<DCRTPoly>,
        params: &DCRTPolyParams,
        inputs: &[DCRTPoly],
        plt_evaluator: Option<&PE>,
    ) -> Vec<DCRTPoly>
    where
        PE: PltEvaluator<DCRTPoly>,
    {
        let one = DCRTPoly::const_one(params);
        let eval_inputs = inputs.to_vec();
        circuit.eval(params, one, eval_inputs, plt_evaluator, None, None)
    }

    #[test]
    fn test_eval_add() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();

        // Create input polynomials using UniformSampler
        let poly1 = create_random_poly(&params);
        let poly2 = create_random_poly(&params);

        // Create a circuit with an Add operation
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(2);
        let add_gate = circuit.add_gate(inputs[0], inputs[1]);
        circuit.output(vec![add_gate]);

        // Evaluate the circuit
        let result = eval_with_const_one(
            &circuit,
            &params,
            &[poly1.clone(), poly2.clone()],
            None::<&PolyPltEvaluator>,
        );

        // Expected result: poly1 + poly2
        let expected = poly1 + poly2;

        // Verify the result
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].coeffs(), expected.coeffs());
    }

    #[test]
    fn test_eval_sub() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();

        // Create input polynomials using UniformSampler
        let poly1 = create_random_poly(&params);
        let poly2 = create_random_poly(&params);

        // Create a circuit with a Sub operation
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(2);
        let sub_gate = circuit.sub_gate(inputs[0], inputs[1]);
        circuit.output(vec![sub_gate]);

        // Evaluate the circuit
        let result = eval_with_const_one(
            &circuit,
            &params,
            &[poly1.clone(), poly2.clone()],
            None::<&PolyPltEvaluator>,
        );

        // Expected result: poly1 - poly2
        let expected = poly1 - poly2;

        // Verify the result
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], expected);
    }

    #[test]
    fn test_eval_mul() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();

        // Create input polynomials using UniformSampler
        let poly1 = create_random_poly(&params);
        let poly2 = create_random_poly(&params);

        // Create a circuit with a Mul operation
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(2);
        let mul_gate = circuit.mul_gate(inputs[0], inputs[1]);
        circuit.output(vec![mul_gate]);

        // Evaluate the circuit
        let result = eval_with_const_one(
            &circuit,
            &params,
            &[poly1.clone(), poly2.clone()],
            None::<&PolyPltEvaluator>,
        );

        // Expected result: poly1 * poly2
        let expected = poly1 * poly2;

        // Verify the result
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], expected);
    }

    #[test]
    fn test_rotate_gate_uses_small_scalar_mul() {
        let params = DCRTPolyParams::default();
        let input = create_random_poly(&params);

        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(1);
        let rotated = circuit.rotate_gate(inputs[0], 3);
        circuit.output(vec![rotated]);

        let result = eval_with_const_one(
            &circuit,
            &params,
            std::slice::from_ref(&input),
            None::<&PolyPltEvaluator>,
        );
        let expected = input * DCRTPoly::from_u32s(&params, &[0, 0, 0, 1]);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0], expected);
        assert_eq!(circuit.count_gates_by_type_vec().get(&PolyGateKind::SmallScalarMul), Some(&1));
    }

    #[test]
    fn test_const_digits() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();

        // Create a circuit with a const_bit_poly gate
        let mut circuit = PolyCircuit::new();
        // We need to call input() to initialize the circuit
        circuit.input(1);

        // Define a specific bit pattern
        // This will create a polynomial with coefficients:
        // [1, 0, 1, 1]
        // (where 1 is at positions 0, 2, 3, and 4)
        let digits = vec![1u32, 0u32, 1u32, 1u32];
        let digits_poly_gate = circuit.const_digits(&digits);
        circuit.output(vec![digits_poly_gate]);

        // Evaluate the circuit with any input (it won't be used)
        let dummy_input = create_random_poly(&params);
        let result =
            eval_with_const_one(&circuit, &params, &[dummy_input], None::<&PolyPltEvaluator>);

        // Verify the result
        assert_eq!(result.len(), 1);

        // Check that the coefficients match the bit pattern
        let coeffs = result[0].coeffs();
        for (i, digit) in digits.iter().enumerate() {
            if digit != &0 {
                assert_eq!(
                    coeffs[i].value(),
                    &BigUint::from(1u8),
                    "Coefficient at position {} should be 1",
                    i
                );
            } else {
                assert_eq!(
                    coeffs[i].value(),
                    &BigUint::from(0u8),
                    "Coefficient at position {} should be 0",
                    i
                );
            }
        }

        // Check that remaining coefficients are 0
        for (i, _) in coeffs.iter().enumerate().skip(digits.len()) {
            assert_eq!(
                coeffs[i].value(),
                &BigUint::from(0u8),
                "Coefficient at position {} should be 0",
                i
            );
        }
    }

    #[test]
    fn test_eval_complex() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();

        // Create input polynomials using UniformSampler
        let poly1 = create_random_poly(&params);
        let poly2 = create_random_poly(&params);
        let poly3 = create_random_poly(&params);

        // Create a complex circuit: (poly1 + poly2) - poly3
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(3);

        // poly1 + poly2
        let add_gate = circuit.add_gate(inputs[0], inputs[1]);

        // (poly1 + poly2) - poly3
        let sub_gate = circuit.sub_gate(add_gate, inputs[2]);

        circuit.output(vec![sub_gate]);

        // Evaluate the circuit
        let result = eval_with_const_one(
            &circuit,
            &params,
            &[poly1.clone(), poly2.clone(), poly3.clone()],
            None::<&PolyPltEvaluator>,
        );

        // Expected result: (poly1 + poly2) - poly3
        let expected = (poly1 + poly2) - poly3;

        // Verify the result
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], expected);
    }

    #[test]
    fn test_eval_multiple_outputs() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();

        // Create input polynomials using UniformSampler
        let poly1 = create_random_poly(&params);
        let poly2 = create_random_poly(&params);

        // Create a circuit with multiple outputs
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(2);

        // poly1 + poly2
        let add_gate = circuit.add_gate(inputs[0], inputs[1]);

        // poly1 - poly2
        let sub_gate = circuit.sub_gate(inputs[0], inputs[1]);

        // poly1 * poly2
        let mul_gate = circuit.mul_gate(inputs[0], inputs[1]);

        circuit.output(vec![add_gate, sub_gate, mul_gate]);

        // Evaluate the circuit
        let result = eval_with_const_one(
            &circuit,
            &params,
            &[poly1.clone(), poly2.clone()],
            None::<&PolyPltEvaluator>,
        );

        // Expected results
        let expected_add = poly1.clone() + poly2.clone();
        let expected_sub = poly1.clone() - poly2.clone();
        let expected_mul = poly1 * poly2;

        // Verify the results
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], expected_add);
        assert_eq!(result[1], expected_sub);
        assert_eq!(result[2], expected_mul);
    }

    #[test]
    fn test_multiple_input_calls_with_nonconsecutive_gate_ids() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();

        // Create input polynomials using UniformSampler
        let poly1 = create_random_poly(&params);
        let poly2 = create_random_poly(&params);

        // Create a circuit
        let mut circuit = PolyCircuit::new();

        // First input call: creates GateId(1) as input
        let inputs_first = circuit.input(1);
        assert_eq!(inputs_first.len(), 1);

        // Insert a gate between input calls so next input gate is non-consecutive
        // Use a const-digits gate which introduces a new gate with no inputs
        circuit.const_digits(&[1u32, 0u32, 1u32]);

        // Second input call: creates a new input gate with a higher, non-consecutive GateId
        let inputs_second = circuit.input(1);
        assert_eq!(inputs_second.len(), 1);

        // Ensure non-consecutive input GateIds (there should be a gap)
        assert_ne!(inputs_second[0].0, inputs_first[0].0 + 1);

        // Build a simple circuit that adds the two inputs together
        let add_gate = circuit.add_gate(inputs_first[0], inputs_second[0]);
        circuit.output(vec![add_gate]);

        // Evaluate the circuit: inputs are assigned in ascending input GateId order
        let result = eval_with_const_one(
            &circuit,
            &params,
            &[poly1.clone(), poly2.clone()],
            None::<&PolyPltEvaluator>,
        );

        let expected = poly1 + poly2;
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], expected);
    }

    #[test]
    fn test_eval_deep_complex() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();

        // Create input polynomials using UniformSampler
        let poly1 = create_random_poly(&params);
        let poly2 = create_random_poly(&params);
        let poly3 = create_random_poly(&params);
        let poly4 = create_random_poly(&params);

        // Create a complex circuit with depth = 4
        // Circuit structure:
        // Level 1: a = poly1 + poly2, b = poly3 * poly4, d = poly1 - poly3
        // Level 2: c = a * b
        // Level 3: e = c + d
        // Level 4: f = e * e
        // Output: f
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(4);

        // Level 1
        let a = circuit.add_gate(inputs[0], inputs[1]); // poly1 + poly2
        let b = circuit.mul_gate(inputs[2], inputs[3]); // poly3 * poly4
        let d = circuit.sub_gate(inputs[0], inputs[2]); // poly1 - poly3

        // Level 2
        let c = circuit.mul_gate(a, b); // (poly1 + poly2) * (poly3 * poly4)

        // Level 3
        let e = circuit.add_gate(c, d); // ((poly1 + poly2) * (poly3 * poly4)) + (poly1 - poly3)

        // Level 4
        let f = circuit.mul_gate(e, e); // (((poly1 + poly2) * (poly3 * poly4)) + (poly1 - poly3))^2

        circuit.output(vec![f]);

        // Evaluate the circuit
        let result = eval_with_const_one(
            &circuit,
            &params,
            &[poly1.clone(), poly2.clone(), poly3.clone(), poly4.clone()],
            None::<&PolyPltEvaluator>,
        );

        // Expected result: (((poly1 + poly2) * (poly3 * poly4)) + (poly1 - poly3))^2
        let expected = (((poly1.clone() + poly2.clone()) * (poly3.clone() * poly4.clone())) +
            (poly1.clone() - poly3.clone())) *
            (((poly1.clone() + poly2) * (poly3.clone() * poly4)) + (poly1 - poly3));

        // Verify the result
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], expected);
    }

    #[test]
    fn test_boolean_gate_and() {
        let params = DCRTPolyParams::default();
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(2);
        let and_result = circuit.and_gate(inputs[0], inputs[1]);
        circuit.output(vec![and_result]);
        let poly1 = create_bit_random_poly(&params);
        let poly2 = create_bit_random_poly(&params);
        let result = eval_with_const_one(
            &circuit,
            &params,
            &[poly1.clone(), poly2.clone()],
            None::<&PolyPltEvaluator>,
        );
        let expected = poly1.clone() * poly2;
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].coeffs(), expected.coeffs());
    }

    #[test]
    fn test_boolean_gate_not() {
        let params = DCRTPolyParams::default();
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(1);
        let not_result = circuit.not_gate(inputs[0]);
        circuit.output(vec![not_result]);
        let poly1 = create_bit_random_poly(&params);
        let result = eval_with_const_one(
            &circuit,
            &params,
            std::slice::from_ref(&poly1),
            None::<&PolyPltEvaluator>,
        );
        let expected = DCRTPoly::const_one(&params) - poly1.clone();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].coeffs(), expected.coeffs());
    }

    #[test]
    fn test_boolean_gate_or() {
        let params = DCRTPolyParams::default();
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(2);
        let or_result = circuit.or_gate(inputs[0], inputs[1]);
        circuit.output(vec![or_result]);
        let poly1 = create_bit_random_poly(&params);
        let poly2 = create_bit_random_poly(&params);
        let result = eval_with_const_one(
            &circuit,
            &params,
            &[poly1.clone(), poly2.clone()],
            None::<&PolyPltEvaluator>,
        );
        let expected = (poly1.clone() + poly2.clone()) - (poly1 * poly2);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].coeffs(), expected.coeffs());
    }

    #[test]
    fn test_boolean_gate_nand() {
        let params = DCRTPolyParams::default();
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(2);
        let nand_result = circuit.nand_gate(inputs[0], inputs[1]);
        circuit.output(vec![nand_result]);
        let poly1 = create_bit_random_poly(&params);
        let poly2 = create_bit_random_poly(&params);
        let result = eval_with_const_one(
            &circuit,
            &params,
            &[poly1.clone(), poly2.clone()],
            None::<&PolyPltEvaluator>,
        );
        let expected = DCRTPoly::const_one(&params) - (poly1 * poly2);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].coeffs(), expected.coeffs());
    }

    #[test]
    fn test_boolean_gate_nor() {
        let params = DCRTPolyParams::default();
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(2);
        let nor_result = circuit.nor_gate(inputs[0], inputs[1]); // poly1 AND poly2
        circuit.output(vec![nor_result]);
        let poly1 = create_bit_random_poly(&params);
        let poly2 = create_bit_random_poly(&params);
        let result = eval_with_const_one(
            &circuit,
            &params,
            &[poly1.clone(), poly2.clone()],
            None::<&PolyPltEvaluator>,
        );
        let expected =
            DCRTPoly::const_one(&params) - ((poly1.clone() + poly2.clone()) - (poly1 * poly2));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].coeffs(), expected.coeffs());
    }

    #[test]
    fn test_boolean_gate_xor() {
        let params = DCRTPolyParams::default();
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(2);
        let nor_result = circuit.xor_gate(inputs[0], inputs[1]);
        circuit.output(vec![nor_result]);
        let poly1 = create_bit_random_poly(&params);
        let poly2 = create_bit_random_poly(&params);
        let result = eval_with_const_one(
            &circuit,
            &params,
            &[poly1.clone(), poly2.clone()],
            None::<&PolyPltEvaluator>,
        );
        let expected = (poly1.clone() + poly2.clone()) -
            (DCRTPoly::from_usize_to_constant(&params, 2) * poly1 * poly2);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].coeffs(), expected.coeffs());
    }

    #[test]
    fn test_boolean_gate_xnor() {
        let params = DCRTPolyParams::default();
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(2);
        let xnor_result = circuit.xnor_gate(inputs[0], inputs[1]);
        circuit.output(vec![xnor_result]);
        let poly1 = create_bit_random_poly(&params);
        let poly2 = create_bit_random_poly(&params);
        let result = eval_with_const_one(
            &circuit,
            &params,
            &[poly1.clone(), poly2.clone()],
            None::<&PolyPltEvaluator>,
        );
        let expected = DCRTPoly::const_one(&params) -
            ((poly1.clone() + poly2.clone()) -
                (DCRTPoly::from_usize_to_constant(&params, 2) * poly1 * poly2));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].coeffs(), expected.coeffs());
    }

    #[test]
    fn test_mul_fhe_poly_bits_mul_by_poly_circuit() {
        let mut circuit = PolyCircuit::new();
        let params = DCRTPolyParams::default();
        let sampler = DCRTPolyUniformSampler::new();
        let sigma = 3.0;
        let log_q = params.modulus_bits();

        // encrypt a polynomial m using FHE secret key encryption
        // Generate random message bits
        let m = sampler.sample_poly(&params, &DistType::BitDist);

        // Encrypt the message
        let a = sampler.sample_poly(&params, &DistType::BitDist);
        let t = sampler.sample_poly(&params, &DistType::BitDist);

        let m_mat = DCRTPolyMatrix::from_poly_vec_row(&params, vec![m.clone()]);
        let a_mat = DCRTPolyMatrix::from_poly_vec_row(&params, vec![a.clone()]);
        let t_mat = DCRTPolyMatrix::from_poly_vec_row(&params, vec![t.clone()]);
        let b_mat = rlwe_encrypt(&params, &sampler, &t_mat, &a_mat, &m_mat, sigma);
        let b = b_mat.entry(0, 0);

        // ct = (a, b)
        let a_bits = a.decompose_base(&params);
        let b_bits = b.decompose_base(&params);

        let x = DCRTPoly::const_one(&params);

        let inputs = circuit.input(a_bits.len() + b_bits.len() + 1);
        assert_eq!(inputs.len(), params.modulus_bits() * 2 + 1);

        // Input: ct[bits], x
        // Output: ct[bits] * x
        let x_id = inputs[inputs.len() - 1];
        let output_ids = inputs
            .iter()
            .take(inputs.len() - 1)
            .map(|&input_id| circuit.mul_gate(input_id, x_id))
            .collect();

        circuit.output(output_ids);

        // concatenate decomposed_c0 and decomposed_c1 and x
        let input = [a_bits, b_bits, vec![x.clone()]].concat();
        let result = eval_with_const_one(&circuit, &params, &input, None::<&PolyPltEvaluator>);

        assert_eq!(result.len(), log_q * 2);

        let a_bits_eval = result[..params.modulus_bits()].to_vec();
        let b_bits_eval = result[params.modulus_bits()..].to_vec();

        let a_eval = DCRTPoly::from_decomposed(&params, &a_bits_eval);
        let b_eval = DCRTPoly::from_decomposed(&params, &b_bits_eval);

        assert_eq!(a_eval, &a * &x);
        assert_eq!(b_eval, &b * &x);

        // decrypt the result
        let plaintext = b_eval - (a_eval * t);
        // recover the message bits
        let plaintext_bits = plaintext.extract_bits_with_threshold(&params);
        assert_eq!(plaintext_bits, (m * x).to_bool_vec());
    }

    #[test]
    fn test_register_and_call_sub_circuit() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();

        // Create input polynomials using UniformSampler
        let poly1 = create_random_poly(&params);
        let poly2 = create_random_poly(&params);

        // Create a sub-circuit that performs addition and multiplication
        let mut sub_circuit = PolyCircuit::new();
        let sub_inputs = sub_circuit.input(2);

        // Add operation: poly1 + poly2
        let add_gate = sub_circuit.add_gate(sub_inputs[0], sub_inputs[1]);

        // Mul operation: poly1 * poly2
        let mul_gate = sub_circuit.mul_gate(sub_inputs[0], sub_inputs[1]);

        // Set the outputs of the sub-circuit
        sub_circuit.output(vec![add_gate, mul_gate]);

        // Create the main circuit
        let mut main_circuit = PolyCircuit::new();
        let main_inputs = main_circuit.input(2);

        // Register the sub-circuit and get its ID
        let sub_circuit_id = main_circuit.register_sub_circuit(sub_circuit);

        // Call the sub-circuit with the main circuit's inputs
        let sub_outputs =
            main_circuit.call_sub_circuit(sub_circuit_id, &[main_inputs[0], main_inputs[1]]);

        // Verify we got two outputs from the sub-circuit
        assert_eq!(sub_outputs.len(), 2);

        // Use the sub-circuit outputs for further operations
        // For example, subtract the multiplication result from the addition result
        let final_gate = main_circuit.sub_gate(sub_outputs[0], sub_outputs[1]);

        // Set the output of the main circuit
        main_circuit.output(vec![final_gate]);

        // Evaluate the main circuit
        let result = eval_with_const_one(
            &main_circuit,
            &params,
            &[poly1.clone(), poly2.clone()],
            None::<&PolyPltEvaluator>,
        );

        // Expected result: (poly1 + poly2) - (poly1 * poly2)
        let expected = (poly1.clone() + poly2.clone()) - (poly1 * poly2);

        // Verify the result
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], expected);
    }

    #[test]
    fn test_register_and_call_parameterized_sub_circuit() {
        let params = DCRTPolyParams::default();
        let input_poly = create_random_poly(&params);

        let mut sub_circuit = PolyCircuit::new();
        let scalar_param =
            sub_circuit.register_sub_circuit_param(SubCircuitParamKind::SmallScalarMul);
        let sub_inputs = sub_circuit.input(1);
        let scaled = sub_circuit.small_scalar_mul_param(sub_inputs[0], scalar_param);
        sub_circuit.output(vec![scaled]);

        let mut main_circuit = PolyCircuit::new();
        let main_inputs = main_circuit.input(1);
        let sub_id = main_circuit.register_sub_circuit(sub_circuit);
        let doubled = main_circuit.call_sub_circuit_with_bindings(
            sub_id,
            &[main_inputs[0]],
            &[SubCircuitParamValue::SmallScalarMul(vec![2])],
        );
        let tripled = main_circuit.call_sub_circuit_with_bindings(
            sub_id,
            &[main_inputs[0]],
            &[SubCircuitParamValue::SmallScalarMul(vec![3])],
        );
        main_circuit.output(vec![doubled[0], tripled[0]]);

        let result = eval_with_const_one(
            &main_circuit,
            &params,
            std::slice::from_ref(&input_poly),
            None::<&PolyPltEvaluator>,
        );
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], input_poly.small_scalar_mul(&params, &[2]));
        assert_eq!(result[1], input_poly.small_scalar_mul(&params, &[3]));
        assert_eq!(main_circuit.non_free_depth(), 0);
    }

    #[test]
    fn test_parameterized_sub_circuit_reuses_identical_binding_sets() {
        let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
        let scalar_param =
            sub_circuit.register_sub_circuit_param(SubCircuitParamKind::SmallScalarMul);
        let sub_inputs = sub_circuit.input(1);
        let scaled = sub_circuit.small_scalar_mul_param(sub_inputs[0], scalar_param);
        sub_circuit.output(vec![scaled]);

        let mut main_circuit = PolyCircuit::<DCRTPoly>::new();
        let main_inputs = main_circuit.input(1);
        let sub_id = main_circuit.register_sub_circuit(sub_circuit);

        let _ = main_circuit.call_sub_circuit_with_bindings(
            sub_id,
            &[main_inputs[0]],
            &[SubCircuitParamValue::SmallScalarMul(vec![7])],
        );
        let _ = main_circuit.call_sub_circuit_with_bindings(
            sub_id,
            &[main_inputs[0]],
            &[SubCircuitParamValue::SmallScalarMul(vec![7])],
        );
        let _ = main_circuit.call_sub_circuit_with_bindings(
            sub_id,
            &[main_inputs[0]],
            &[SubCircuitParamValue::SmallScalarMul(vec![9])],
        );

        let binding_set_ids = main_circuit
            .sub_circuit_calls
            .values()
            .map(|call| call.binding_set_id)
            .collect::<Vec<_>>();
        assert_eq!(binding_set_ids.len(), 3);
        assert_eq!(binding_set_ids[0], binding_set_ids[1]);
        assert_ne!(binding_set_ids[0], binding_set_ids[2]);
        assert_eq!(main_circuit.binding_registry.binding_sets.len(), 2);
    }

    #[test]
    fn test_register_sub_circuit_reuses_child_binding_sets_without_duplication() {
        let mut leaf_circuit = PolyCircuit::<DCRTPoly>::new();
        let scalar_param =
            leaf_circuit.register_sub_circuit_param(SubCircuitParamKind::SmallScalarMul);
        let leaf_inputs = leaf_circuit.input(1);
        let scaled = leaf_circuit.small_scalar_mul_param(leaf_inputs[0], scalar_param);
        leaf_circuit.output(vec![scaled]);

        let mut middle_circuit = PolyCircuit::<DCRTPoly>::new();
        let middle_inputs = middle_circuit.input(1);
        let leaf_id = middle_circuit.register_sub_circuit(leaf_circuit);
        let _ = middle_circuit.call_sub_circuit_with_bindings(
            leaf_id,
            &[middle_inputs[0]],
            &[SubCircuitParamValue::SmallScalarMul(vec![11])],
        );
        let _ = middle_circuit.call_sub_circuit_with_bindings(
            leaf_id,
            &[middle_inputs[0]],
            &[SubCircuitParamValue::SmallScalarMul(vec![11])],
        );
        assert_eq!(middle_circuit.binding_registry.binding_sets.len(), 1);

        let mut main_circuit = PolyCircuit::<DCRTPoly>::new();
        let main_inputs = main_circuit.input(1);
        let middle_id = main_circuit.register_sub_circuit(middle_circuit);
        let _ = main_circuit.call_sub_circuit(middle_id, &[main_inputs[0]]);

        assert_eq!(main_circuit.binding_registry.binding_sets.len(), 2);
        let StoredSubCircuit::InMemory(registered_middle) =
            main_circuit.sub_circuits.get(&middle_id).expect("middle circuit missing");
        assert!(Arc::ptr_eq(&registered_middle.binding_registry, &main_circuit.binding_registry));
    }

    #[test]
    fn test_register_and_call_summed_parameterized_sub_circuit() {
        let params = DCRTPolyParams::default();
        let input_poly = create_random_poly(&params);

        let mut sub_circuit = PolyCircuit::new();
        let scalar_param =
            sub_circuit.register_sub_circuit_param(SubCircuitParamKind::SmallScalarMul);
        let sub_inputs = sub_circuit.input(1);
        let scaled = sub_circuit.small_scalar_mul_param(sub_inputs[0], scalar_param);
        sub_circuit.output(vec![scaled]);

        let mut main_circuit = PolyCircuit::new();
        let main_inputs = main_circuit.input(1);
        let sub_id = main_circuit.register_sub_circuit(sub_circuit);
        let double_id =
            main_circuit.intern_binding_set(&[SubCircuitParamValue::SmallScalarMul(vec![2])]);
        let triple_id =
            main_circuit.intern_binding_set(&[SubCircuitParamValue::SmallScalarMul(vec![3])]);
        let input_set_id = main_circuit.intern_input_set(&[main_inputs[0]]);
        let outputs = main_circuit.call_sub_circuit_sum_many_with_binding_set_ids(
            sub_id,
            vec![input_set_id, input_set_id, input_set_id],
            vec![double_id, double_id, triple_id],
        );
        main_circuit.output(outputs.clone());

        let result = eval_with_const_one(
            &main_circuit,
            &params,
            std::slice::from_ref(&input_poly),
            None::<&PolyPltEvaluator>,
        );
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], input_poly.small_scalar_mul(&params, &[7]));
        assert_eq!(main_circuit.non_free_depth(), 0);
        assert_eq!(main_circuit.binding_registry.binding_sets.len(), 2);
        assert_eq!(main_circuit.input_set_registry.input_sets.len(), 1);
        assert_eq!(main_circuit.summed_sub_circuit_calls.len(), 1);
    }

    #[test]
    fn test_summed_sub_circuit_non_free_depth_uses_max_inner_call_depth() {
        let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
        let sub_inputs = sub_circuit.input(2);
        let product = sub_circuit.mul_gate(sub_inputs[0], sub_inputs[1]);
        sub_circuit.output(vec![product]);

        let mut main_circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = main_circuit.input(3);
        let precomputed = main_circuit.mul_gate(inputs[0], inputs[1]);
        let sub_id = main_circuit.register_sub_circuit(sub_circuit);
        let direct_input_set_id = main_circuit.intern_input_set(&[inputs[0], inputs[2]]);
        let precomputed_input_set_id = main_circuit.intern_input_set(&[precomputed, inputs[2]]);
        let outputs = main_circuit.call_sub_circuit_sum_many_with_binding_set_ids(
            sub_id,
            vec![direct_input_set_id, precomputed_input_set_id],
            vec![main_circuit.intern_binding_set(&[]), main_circuit.intern_binding_set(&[])],
        );
        main_circuit.output(outputs);

        assert_eq!(main_circuit.non_free_depth(), 2);
    }

    #[test]
    fn test_nested_sub_circuits_with_disk_api_compatibility() {
        let params = DCRTPolyParams::default();

        let poly1 = create_random_poly(&params);
        let poly2 = create_random_poly(&params);
        let poly3 = create_random_poly(&params);

        let mut inner_circuit = PolyCircuit::new();
        let inner_inputs = inner_circuit.input(2);
        let mul_gate = inner_circuit.mul_gate(inner_inputs[0], inner_inputs[1]);
        inner_circuit.output(vec![mul_gate]);

        let mut middle_circuit = PolyCircuit::new();
        let middle_inputs = middle_circuit.input(3);
        let inner_circuit_id = middle_circuit.register_sub_circuit(inner_circuit);
        let inner_outputs = middle_circuit
            .call_sub_circuit(inner_circuit_id, &[middle_inputs[0], middle_inputs[1]]);
        let add_gate = middle_circuit.add_gate(inner_outputs[0], middle_inputs[2]);
        middle_circuit.output(vec![add_gate]);

        let mut main_circuit = PolyCircuit::new();
        main_circuit.enable_subcircuits_in_disk("unused");
        let main_inputs = main_circuit.input(3);
        let middle_circuit_id = main_circuit.register_sub_circuit(middle_circuit);
        let middle_outputs = main_circuit
            .call_sub_circuit(middle_circuit_id, &[main_inputs[0], main_inputs[1], main_inputs[2]]);
        let square_gate = main_circuit.mul_gate(middle_outputs[0], middle_outputs[0]);
        main_circuit.output(vec![square_gate]);

        assert!(
            main_circuit
                .sub_circuits
                .values()
                .all(|sub| matches!(sub, StoredSubCircuit::InMemory(_)))
        );

        let gate_counts = main_circuit.count_gates_by_type_vec();
        assert_eq!(main_circuit.lut_vector_len_with_subcircuits(), 0);
        assert_eq!(gate_counts.get(&PolyGateKind::Mul).copied().unwrap_or(0), 2);
        assert_eq!(main_circuit.non_free_depth(), 2);

        let result = eval_with_const_one(
            &main_circuit,
            &params,
            &[poly1.clone(), poly2.clone(), poly3.clone()],
            None::<&PolyPltEvaluator>,
        );

        let expected =
            ((poly1.clone() * poly2.clone()) + poly3.clone()) * ((poly1 * poly2) + poly3);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], expected);
    }

    #[test]
    fn test_enable_subcircuits_in_disk_is_noop_for_in_memory_storage() {
        let params = DCRTPolyParams::default();
        let poly1 = create_random_poly(&params);
        let poly2 = create_random_poly(&params);

        let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
        let sub_inputs = sub_circuit.input(2);
        let add_gate = sub_circuit.add_gate(sub_inputs[0], sub_inputs[1]);
        sub_circuit.output(vec![add_gate]);

        let mut main_circuit = PolyCircuit::<DCRTPoly>::new();
        main_circuit.enable_subcircuits_in_disk("unused");
        let main_inputs = main_circuit.input(2);
        let sub_circuit_id = main_circuit.register_sub_circuit(sub_circuit);
        let outputs =
            main_circuit.call_sub_circuit(sub_circuit_id, &[main_inputs[0], main_inputs[1]]);
        main_circuit.output(outputs);

        assert!(
            main_circuit
                .sub_circuits
                .values()
                .all(|sub| matches!(sub, StoredSubCircuit::InMemory(_)))
        );

        let result = eval_with_const_one(
            &main_circuit,
            &params,
            &[poly1.clone(), poly2.clone()],
            None::<&PolyPltEvaluator>,
        );
        assert_eq!(result[0], poly1 + poly2);
    }

    #[test]
    fn test_nested_sub_circuits() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();

        // Create input polynomials
        let poly1 = create_random_poly(&params);
        let poly2 = create_random_poly(&params);
        let poly3 = create_random_poly(&params);

        // Create the innermost sub-circuit that performs multiplication
        let mut inner_circuit = PolyCircuit::new();
        let inner_inputs = inner_circuit.input(2);
        let mul_gate = inner_circuit.mul_gate(inner_inputs[0], inner_inputs[1]);
        inner_circuit.output(vec![mul_gate]);

        // Create a middle sub-circuit that uses the inner sub-circuit
        let mut middle_circuit = PolyCircuit::new();
        let middle_inputs = middle_circuit.input(3);

        // Register the inner circuit
        let inner_circuit_id = middle_circuit.register_sub_circuit(inner_circuit);

        // Call the inner circuit with the first two inputs
        let inner_outputs = middle_circuit
            .call_sub_circuit(inner_circuit_id, &[middle_inputs[0], middle_inputs[1]]);

        // Add the result of the inner circuit with the third input
        let add_gate = middle_circuit.add_gate(inner_outputs[0], middle_inputs[2]);
        middle_circuit.output(vec![add_gate]);

        // Create the main circuit
        let mut main_circuit = PolyCircuit::new();
        let main_inputs = main_circuit.input(3);

        // Register the middle circuit
        let middle_circuit_id = main_circuit.register_sub_circuit(middle_circuit);

        // Call the middle circuit with all inputs
        let middle_outputs = main_circuit
            .call_sub_circuit(middle_circuit_id, &[main_inputs[0], main_inputs[1], main_inputs[2]]);

        let scalar_mul_gate = main_circuit.mul_gate(middle_outputs[0], middle_outputs[0]);

        // Set the output of the main circuit
        main_circuit.output(vec![scalar_mul_gate]);

        // Evaluate the main circuit
        let result = eval_with_const_one(
            &main_circuit,
            &params,
            &[poly1.clone(), poly2.clone(), poly3.clone()],
            None::<&PolyPltEvaluator>,
        );

        // Expected result: ((poly1 * poly2) + poly3)^2
        let expected =
            ((poly1.clone() * poly2.clone()) + poly3.clone()) * ((poly1 * poly2) + poly3);

        // Verify the result
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], expected);
    }

    #[test]
    fn test_const_zero_gate() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();

        // Create a circuit with a const_zero_gate
        let mut circuit = PolyCircuit::new();
        // We need to call input() to initialize the circuit
        circuit.input(1);
        let zero_gate = circuit.const_zero_gate();
        circuit.output(vec![zero_gate]);

        // Evaluate the circuit with any input (it won't be used)
        let dummy_input = create_random_poly(&params);
        let result =
            eval_with_const_one(&circuit, &params, &[dummy_input], None::<&PolyPltEvaluator>);

        // Expected result: 0
        let expected = DCRTPoly::const_zero(&params);

        // Verify the result
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], expected);
    }

    #[test]
    fn test_const_one_gate() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();

        // Create a circuit with a const_one_gate
        let mut circuit = PolyCircuit::new();
        // We need to call input() to initialize the circuit
        circuit.input(1);
        let one_gate = circuit.const_one_gate();
        circuit.output(vec![one_gate]);

        // Evaluate the circuit with any input (it won't be used)
        let dummy_input = create_random_poly(&params);
        let result =
            eval_with_const_one(&circuit, &params, &[dummy_input], None::<&PolyPltEvaluator>);

        // Expected result: 1
        let expected = DCRTPoly::const_one(&params);

        // Verify the result
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], expected);
    }

    #[test]
    fn test_const_minus_one_gate() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();

        // Create a circuit with a const_minus_one_gate
        let mut circuit = PolyCircuit::new();
        // We need to call input() to initialize the circuit
        circuit.input(1);
        let minus_one_gate = circuit.const_minus_one_gate();
        circuit.output(vec![minus_one_gate]);

        // Evaluate the circuit with any input (it won't be used)
        let dummy_input = create_random_poly(&params);
        let result =
            eval_with_const_one(&circuit, &params, &[dummy_input], None::<&PolyPltEvaluator>);

        // Expected result: -1
        // We can compute -1 as 0 - 1
        let expected = DCRTPoly::const_zero(&params) - DCRTPoly::const_one(&params);

        // verify
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], expected);
    }

    #[test]
    fn test_depth_zero_with_direct_input_output() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(1);
        circuit.output(vec![inputs[0]]);
        assert_eq!(circuit.depth(), 0);
    }

    #[test]
    fn test_depth_one_with_add() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(2);
        let add = circuit.add_gate(inputs[0], inputs[1]);
        circuit.output(vec![add]);
        assert_eq!(circuit.depth(), 1);
    }

    #[test]
    fn test_depth_two_with_chain() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(3);
        let add = circuit.add_gate(inputs[0], inputs[1]);
        let mul = circuit.mul_gate(add, inputs[2]);
        circuit.output(vec![mul]);
        assert_eq!(circuit.depth(), 2);
    }

    #[test]
    fn test_non_free_depth_counts_sub_circuit() {
        let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
        let sub_inputs = sub_circuit.input(1);
        let mul1 = sub_circuit.mul_gate(sub_inputs[0], sub_inputs[0]);
        let mul2 = sub_circuit.mul_gate(mul1, sub_inputs[0]);
        sub_circuit.output(vec![mul2]);

        let mut main_circuit = PolyCircuit::<DCRTPoly>::new();
        let main_inputs = main_circuit.input(1);
        let sub_id = main_circuit.register_sub_circuit(sub_circuit);
        let sub_outputs = main_circuit.call_sub_circuit(sub_id, &[main_inputs[0]]);
        main_circuit.output(vec![sub_outputs[0]]);

        assert_eq!(main_circuit.non_free_depth(), 2);
    }

    #[test]
    fn test_non_free_depth_respects_sub_circuit_inputs() {
        let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
        let sub_inputs = sub_circuit.input(2);
        sub_circuit.output(vec![sub_inputs[0]]);

        let mut main_circuit = PolyCircuit::<DCRTPoly>::new();
        let main_inputs = main_circuit.input(3);
        let mul1 = main_circuit.mul_gate(main_inputs[1], main_inputs[2]);
        let mul2 = main_circuit.mul_gate(mul1, main_inputs[1]);

        let sub_id = main_circuit.register_sub_circuit(sub_circuit);
        let sub_outputs = main_circuit.call_sub_circuit(sub_id, &[main_inputs[0], mul2]);
        main_circuit.output(vec![sub_outputs[0]]);

        assert_eq!(main_circuit.non_free_depth(), 0);
    }

    #[test]
    fn test_non_free_depth_counts_slot_transfer_as_non_free() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(1);
        let transferred = circuit.slot_transfer_gate(inputs[0], &[(0, None)]);
        circuit.output(vec![transferred]);

        assert_eq!(circuit.non_free_depth(), 1);
    }

    #[test]
    fn test_non_free_depth_ignores_add_chains() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(4);
        let add1 = circuit.add_gate(inputs[0], inputs[1]);
        let add2 = circuit.add_gate(add1, inputs[2]);
        let mul = circuit.mul_gate(add2, inputs[3]);
        circuit.output(vec![mul]);

        assert_eq!(circuit.non_free_depth(), 1);
    }

    #[test]
    fn test_non_free_depth_handles_multi_output_sub_circuit_call() {
        let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
        let sub_inputs = sub_circuit.input(1);
        let add = sub_circuit.add_gate(sub_inputs[0], sub_inputs[0]);
        let mul = sub_circuit.mul_gate(sub_inputs[0], sub_inputs[0]);
        sub_circuit.output(vec![add, mul]);

        let mut main_circuit = PolyCircuit::<DCRTPoly>::new();
        let main_inputs = main_circuit.input(1);
        let sub_id = main_circuit.register_sub_circuit(sub_circuit);
        let sub_outputs = main_circuit.call_sub_circuit(sub_id, &[main_inputs[0]]);
        let sum = main_circuit.add_gate(sub_outputs[0], sub_outputs[1]);
        main_circuit.output(vec![sum]);

        assert_eq!(main_circuit.non_free_depth(), 1);
    }

    #[test]
    fn test_non_free_depth_handles_repeated_sub_circuit_calls_with_different_input_levels() {
        let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
        let sub_inputs = sub_circuit.input(2);
        let mul = sub_circuit.mul_gate(sub_inputs[0], sub_inputs[1]);
        sub_circuit.output(vec![mul]);

        let mut main_circuit = PolyCircuit::<DCRTPoly>::new();
        let main_inputs = main_circuit.input(3);
        let precomputed = main_circuit.mul_gate(main_inputs[0], main_inputs[1]);

        let sub_id = main_circuit.register_sub_circuit(sub_circuit);
        let direct_call = main_circuit.call_sub_circuit(sub_id, &[main_inputs[0], main_inputs[2]]);
        let nested_call = main_circuit.call_sub_circuit(sub_id, &[precomputed, main_inputs[2]]);
        let output = main_circuit.add_gate(direct_call[0], nested_call[0]);
        main_circuit.output(vec![output]);

        assert_eq!(main_circuit.non_free_depth(), 2);
    }

    #[test]
    fn test_non_free_depth_batches_multiple_ready_sub_circuit_calls() {
        let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
        let sub_inputs = sub_circuit.input(2);
        let mul = sub_circuit.mul_gate(sub_inputs[0], sub_inputs[1]);
        sub_circuit.output(vec![mul]);

        let mut main_circuit = PolyCircuit::<DCRTPoly>::new();
        let main_inputs = main_circuit.input(4);
        let precomputed = main_circuit.mul_gate(main_inputs[0], main_inputs[1]);

        let sub_id = main_circuit.register_sub_circuit(sub_circuit);
        let call1 = main_circuit.call_sub_circuit(sub_id, &[precomputed, main_inputs[2]]);
        let call2 = main_circuit.call_sub_circuit(sub_id, &[precomputed, main_inputs[3]]);
        let output = main_circuit.add_gate(call1[0], call2[0]);
        main_circuit.output(vec![output]);

        assert_eq!(main_circuit.non_free_depth(), 2);
    }
}
