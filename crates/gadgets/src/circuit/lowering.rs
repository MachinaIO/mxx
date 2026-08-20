use super::{
    BatchedWire, GateParamSource, PolyCircuit, PolyGate, PolyGateKind, PolyGateType,
    SlotTransferSpec, SubCircuitInputMaxPlaintextNormRange, SubCircuitParamValue, gate::GateId,
};
use crate::Poly;
use num_bigint::BigUint;
use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap},
    error::Error,
    sync::Arc,
};
use thiserror::Error;

/// Stable structural identity of one circuit-operation invocation.
///
/// `call_path` contains the construction-time scoped call id of every
/// enclosing direct or summed sub-circuit call. `local_gate` is the gate id in
/// the innermost circuit. `occurrence` is zero for ordinary gates and
/// distinguishes multiple accumulation operations emitted by one summed-call
/// placeholder.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct GateInstance<'a> {
    call_path: &'a [usize],
    local_gate: GateId,
    occurrence: usize,
}

impl<'a> GateInstance<'a> {
    fn ordinary(call_path: &'a [usize], local_gate: GateId) -> Self {
        Self { call_path, local_gate, occurrence: 0 }
    }

    fn occurrence(call_path: &'a [usize], local_gate: GateId, occurrence: usize) -> Self {
        Self { call_path, local_gate, occurrence }
    }

    pub fn call_path(self) -> &'a [usize] {
        self.call_path
    }

    pub fn local_gate(self) -> GateId {
        self.local_gate
    }

    pub fn operation_occurrence(self) -> usize {
        self.occurrence
    }
}

/// A BGG-independent lowering interface for translating [`PolyCircuit`] into
/// canonical Graph IR.
///
/// The circuit layer owns traversal, parameter binding, and recursive
/// sub-circuit expansion. Scheme crates implement only the operation-specific
/// wire formulas. This keeps `mxx-gadgets` independent of BGG value types and
/// keeps concrete execution in `mxx-runtime`.
pub trait CircuitLoweringTypes {
    type Wire: Clone;
    type Error: Error + Send + Sync + 'static;

    /// Attaches call-specific compile-time metadata to child inputs before the
    /// child circuit is expanded. Ordinary lowerers leave inputs unchanged.
    fn enter_subcircuit_inputs(
        &mut self,
        inputs: Vec<Self::Wire>,
        _input_max_plaintext_norm_ranges: Option<&[SubCircuitInputMaxPlaintextNormRange]>,
    ) -> Result<Vec<Self::Wire>, Self::Error> {
        Ok(inputs)
    }
}

/// Scheme-specific lowering for public lookup gates.
pub trait PublicLookupLowering<P: Poly>: CircuitLoweringTypes {
    fn public_lookup(
        &mut self,
        circuit: &PolyCircuit<P>,
        lookup_id: usize,
        input: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error>;
}

/// Scheme-specific lowering for slot transfer and reduction gates.
pub trait SlotOperationLowering<P: Poly>: CircuitLoweringTypes {
    fn slot_transfer(
        &mut self,
        input: &Self::Wire,
        source_slots: &[(u32, Option<u32>)],
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error>;

    /// Lowers a cyclic slot rotation.
    ///
    /// Implementors may override this to preserve and exploit the rotation
    /// structure. The default retains the existing behavior by materializing an
    /// explicit slot-transfer mapping.
    fn slot_rotation(
        &mut self,
        input: &Self::Wire,
        offset: u32,
        num_slots: u32,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let materialized = SlotTransferSpec::rotation(
            usize::try_from(offset).expect("rotation offset must fit in usize"),
            usize::try_from(num_slots).expect("rotation slot count must fit in usize"),
        )
        .materialize();
        self.slot_transfer(input, &materialized, gate)
    }

    fn slot_reduce(
        &mut self,
        inputs: &[Self::Wire],
        slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error>;
}

/// Scheme-specific lowering for ordinary arithmetic gates.
pub trait ArithmeticCircuitLowering<P: Poly>: CircuitLoweringTypes {
    fn binary(
        &mut self,
        operation: PolyGateKind,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error>;

    fn small_scalar_mul(
        &mut self,
        input: &Self::Wire,
        scalar: &[u32],
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error>;

    fn large_scalar_mul(
        &mut self,
        input: &Self::Wire,
        scalar: &[BigUint],
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error>;
}

/// Complete lowering used by circuit traversal.
pub trait GraphCircuitLowering<P: Poly>:
    ArithmeticCircuitLowering<P> + PublicLookupLowering<P> + SlotOperationLowering<P>
{
}

impl<P, L> GraphCircuitLowering<P> for L
where
    P: Poly,
    L: ArithmeticCircuitLowering<P> + PublicLookupLowering<P> + SlotOperationLowering<P>,
{
}

/// Graph lowering that can preserve reusable [`PolyCircuit`] sub-circuits as
/// structural subgraphs instead of expanding every call into its parent.
pub trait StructuredCircuitLowering<P: Poly>: GraphCircuitLowering<P> {
    type Subgraph: Clone;

    /// Whether lookup/slot gate identities affect values or artifact names
    /// emitted by this lowerer. Such definitions are kept per invocation;
    /// pure executable lowerers can share them because the core subgraph call
    /// already supplies distinct runtime instantiation paths.
    fn call_site_identity_is_semantic(&self) -> bool {
        true
    }

    fn define_subgraph<F>(
        &mut self,
        name: &str,
        input_examples: &[Self::Wire],
        body: F,
    ) -> Result<Self::Subgraph, CircuitLowerError<Self::Error>>
    where
        F: FnOnce(
            &mut Self,
            Vec<Self::Wire>,
        ) -> Result<Vec<Self::Wire>, CircuitLowerError<Self::Error>>;

    fn call_subgraph(
        &mut self,
        definition: &Self::Subgraph,
        inputs: Vec<Self::Wire>,
    ) -> Result<Vec<Self::Wire>, CircuitLowerError<Self::Error>>;

    /// Calls an audited constant-polynomial canonical-nonnegative LUT subgraph.
    ///
    /// `canonical_input_exclusive_uppers` is aligned with `inputs`; its synthetic constant-one
    /// entry is `None`. A present upper is derived only from a
    /// [`SubCircuitInputMaxPlaintextNormRange`] attached to an audited LUT-bearing call. The LUT
    /// may be in a descendant subgraph; no recursive structural scan is performed. Setting this
    /// existing metadata is the producer's explicit assertion that the covered values are
    /// canonical-nonnegative constant polynomials used by the LUT path. A range registered on the
    /// definition takes precedence over a call-site range. Under that contract, an inclusive norm
    /// `B` means the canonical integer is in `[0, B]`, so the transmitted exclusive upper is
    /// `B + 1`.
    ///
    /// General polynomial calls must not reinterpret `PolyNorm` this way. Lowerers that do not
    /// implement this audited contract reject a nonempty contract rather than silently dropping
    /// it.
    fn call_audited_constant_lut_subgraph(
        &mut self,
        definition: &Self::Subgraph,
        inputs: Vec<Self::Wire>,
        canonical_input_exclusive_uppers: Vec<Option<BigUint>>,
    ) -> Result<Vec<Self::Wire>, CircuitLowerError<Self::Error>> {
        if canonical_input_exclusive_uppers.iter().any(Option::is_some) {
            return Err(CircuitLowerError::GraphStructure(
                "structured lowerer does not implement audited constant-LUT range contracts"
                    .to_owned(),
            ));
        }
        self.call_subgraph(definition, inputs)
    }

    fn call_subgraph_parallel(
        &mut self,
        definition: &Self::Subgraph,
        inputs: Vec<Vec<Self::Wire>>,
    ) -> Result<Vec<Vec<Self::Wire>>, CircuitLowerError<Self::Error>> {
        inputs.into_iter().map(|inputs| self.call_subgraph(definition, inputs)).collect()
    }

    /// Parallel counterpart of [`Self::call_audited_constant_lut_subgraph`].
    fn call_audited_constant_lut_subgraph_parallel(
        &mut self,
        definition: &Self::Subgraph,
        inputs: Vec<Vec<Self::Wire>>,
        canonical_input_exclusive_uppers: Vec<Option<BigUint>>,
    ) -> Result<Vec<Vec<Self::Wire>>, CircuitLowerError<Self::Error>> {
        if canonical_input_exclusive_uppers.iter().any(Option::is_some) {
            return Err(CircuitLowerError::GraphStructure(
                "structured lowerer does not implement audited constant-LUT range contracts"
                    .to_owned(),
            ));
        }
        self.call_subgraph_parallel(definition, inputs)
    }
}

#[derive(Debug, Error)]
pub enum CircuitLowerError<E>
where
    E: Error + Send + Sync + 'static,
{
    #[error("gate {gate} references unavailable input gate {input}")]
    MissingInput { gate: usize, input: usize },
    #[error("gate {gate} has an invalid input arity")]
    InvalidArity { gate: usize },
    #[error("gate {gate} references missing sub-circuit parameter {parameter}")]
    MissingParameter { gate: usize, parameter: usize },
    #[error("gate {gate} sub-circuit parameter {parameter} has kind {actual}, expected {expected}")]
    ParameterKind { gate: usize, parameter: usize, expected: &'static str, actual: &'static str },
    #[error("gate {gate} uses unsupported parameterized public lookup id {parameter}")]
    ParameterizedPublicLookup { gate: usize, parameter: usize },
    #[error("the circuit lowerer received more input bundles than the circuit consumes")]
    ExtraInputs,
    #[error("graph structure error: {0}")]
    GraphStructure(String),
    #[error("gate {gate}: {source}")]
    Operation {
        gate: usize,
        #[source]
        source: E,
    },
}

/// Lowers one complete circuit, including recursively registered and summed
/// sub-circuits, into canonical Graph IR through `lowering`.
pub fn lower_circuit<P, L>(
    circuit: &PolyCircuit<P>,
    one: L::Wire,
    inputs: impl IntoIterator<Item = L::Wire>,
    lowering: &mut L,
) -> Result<Vec<L::Wire>, CircuitLowerError<L::Error>>
where
    P: Poly,
    L: GraphCircuitLowering<P>,
{
    lower_scoped(circuit, one, inputs.into_iter().collect(), &[], &mut Vec::new(), lowering)
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct SubgraphCacheKey {
    circuit_identity: usize,
    bindings: Vec<SubCircuitParamValue>,
    call_path: Option<Vec<usize>>,
}

struct StructuredLoweringState<S> {
    definitions: HashMap<SubgraphCacheKey, S>,
    call_site_identity: HashMap<usize, bool>,
    next_definition: usize,
}

impl<S> Default for StructuredLoweringState<S> {
    fn default() -> Self {
        Self { definitions: HashMap::new(), call_site_identity: HashMap::new(), next_definition: 0 }
    }
}

fn requires_call_site_identity<P: Poly>(
    circuit: &PolyCircuit<P>,
    memo: &mut HashMap<usize, bool>,
) -> bool {
    let circuit_identity = circuit as *const PolyCircuit<P> as usize;
    if let Some(&required) = memo.get(&circuit_identity) {
        return required;
    }
    // Registered sub-circuits form a DAG. Insert a provisional value so an
    // invalid recursive registration cannot make this structural query recurse
    // forever before normal circuit validation reports it.
    memo.insert(circuit_identity, false);
    let required = circuit.gates_in_id_order().any(|(_, gate)| match &gate.gate_type {
        PolyGateType::SlotTransfer { .. } |
        PolyGateType::SlotReduce { .. } |
        PolyGateType::PubLut { .. } => true,
        PolyGateType::SubCircuitOutput { call_id, .. } => {
            let info = circuit.sub_circuit_call_info(*call_id);
            let child = circuit.registered_sub_circuit_ref(info.sub_circuit_id);
            requires_call_site_identity(child.as_ref(), memo)
        }
        PolyGateType::SummedSubCircuitOutput { summed_call_id, .. } => {
            let info = circuit.summed_sub_circuit_call_info(*summed_call_id);
            let child = circuit.registered_sub_circuit_ref(info.sub_circuit_id);
            requires_call_site_identity(child.as_ref(), memo)
        }
        _ => false,
    });
    memo.insert(circuit_identity, required);
    required
}

/// Lowers a complete circuit while retaining registered sub-circuits as
/// reusable structural subgraphs. The implicit constant-one wire is an
/// explicit subgraph argument so no executable value is captured across a
/// structural boundary.
pub fn lower_circuit_structured<P, L>(
    circuit: &PolyCircuit<P>,
    one: L::Wire,
    inputs: impl IntoIterator<Item = L::Wire>,
    lowering: &mut L,
) -> Result<Vec<L::Wire>, CircuitLowerError<L::Error>>
where
    P: Poly,
    L: StructuredCircuitLowering<P>,
{
    lower_scoped_structured(
        circuit,
        one,
        inputs.into_iter().collect(),
        &[],
        &mut Vec::new(),
        lowering,
        &mut StructuredLoweringState::default(),
    )
}

fn structured_definition<P, L>(
    child: Arc<PolyCircuit<P>>,
    one: &L::Wire,
    child_inputs: &[L::Wire],
    bindings: &[SubCircuitParamValue],
    call_path: &[usize],
    lowering: &mut L,
    state: &mut StructuredLoweringState<L::Subgraph>,
) -> Result<L::Subgraph, CircuitLowerError<L::Error>>
where
    P: Poly,
    L: StructuredCircuitLowering<P>,
{
    let call_site_identity = lowering.call_site_identity_is_semantic() &&
        requires_call_site_identity(child.as_ref(), &mut state.call_site_identity);
    let key = SubgraphCacheKey {
        circuit_identity: Arc::as_ptr(&child) as usize,
        bindings: bindings.to_vec(),
        call_path: call_site_identity.then(|| call_path.to_vec()),
    };
    if let Some(definition) = state.definitions.get(&key) {
        return Ok(definition.clone());
    }
    let definition_id = state.next_definition;
    state.next_definition += 1;
    let name = format!("poly-circuit-{definition_id}");
    let definition_inputs =
        std::iter::once(one.clone()).chain(child_inputs.iter().cloned()).collect::<Vec<_>>();
    let bindings = bindings.to_vec();
    let mut definition_path = call_path.to_vec();
    let definition =
        lowering.define_subgraph(&name, &definition_inputs, |lowering, mut placeholders| {
            if placeholders.is_empty() {
                return Err(CircuitLowerError::InvalidArity { gate: 0 });
            }
            let child_one = placeholders.remove(0);
            lower_scoped_structured(
                child.as_ref(),
                child_one,
                placeholders,
                &bindings,
                &mut definition_path,
                lowering,
                state,
            )
        })?;
    state.definitions.insert(key, definition.clone());
    Ok(definition)
}

fn call_structured_child<P, L>(
    child: Arc<PolyCircuit<P>>,
    one: &L::Wire,
    child_inputs: Vec<L::Wire>,
    input_max_plaintext_norm_ranges: Option<&[SubCircuitInputMaxPlaintextNormRange]>,
    bindings: &[SubCircuitParamValue],
    call_path: &[usize],
    lowering: &mut L,
    state: &mut StructuredLoweringState<L::Subgraph>,
) -> Result<Vec<L::Wire>, CircuitLowerError<L::Error>>
where
    P: Poly,
    L: StructuredCircuitLowering<P>,
{
    let definition = structured_definition(
        child.clone(),
        one,
        &child_inputs,
        bindings,
        call_path,
        lowering,
        state,
    )?;
    let inputs = std::iter::once(one.clone()).chain(child_inputs).collect::<Vec<_>>();
    let canonical_input_exclusive_uppers = audited_constant_lut_input_exclusive_uppers(
        child
            .sub_circuit_input_max_plaintext_norm_ranges
            .as_deref()
            .or(input_max_plaintext_norm_ranges),
        inputs.len() - 1,
    );
    lowering.call_audited_constant_lut_subgraph(
        &definition,
        inputs,
        canonical_input_exclusive_uppers,
    )
}

fn call_structured_children_parallel<P, L>(
    child: Arc<PolyCircuit<P>>,
    one: &L::Wire,
    child_inputs: Vec<Vec<L::Wire>>,
    input_max_plaintext_norm_ranges: Option<&[SubCircuitInputMaxPlaintextNormRange]>,
    bindings: &[SubCircuitParamValue],
    call_path: &[usize],
    lowering: &mut L,
    state: &mut StructuredLoweringState<L::Subgraph>,
) -> Result<Vec<Vec<L::Wire>>, CircuitLowerError<L::Error>>
where
    P: Poly,
    L: StructuredCircuitLowering<P>,
{
    let first = child_inputs.first().ok_or(CircuitLowerError::InvalidArity { gate: 0 })?;
    let input_count = first.len();
    let definition =
        structured_definition(child.clone(), one, first, bindings, call_path, lowering, state)?;
    let inputs = child_inputs
        .into_iter()
        .map(|inputs| std::iter::once(one.clone()).chain(inputs).collect())
        .collect();
    let canonical_input_exclusive_uppers = audited_constant_lut_input_exclusive_uppers(
        child
            .sub_circuit_input_max_plaintext_norm_ranges
            .as_deref()
            .or(input_max_plaintext_norm_ranges),
        input_count,
    );
    lowering.call_audited_constant_lut_subgraph_parallel(
        &definition,
        inputs,
        canonical_input_exclusive_uppers,
    )
}

fn audited_constant_lut_input_exclusive_uppers(
    input_max_plaintext_norm_ranges: Option<&[SubCircuitInputMaxPlaintextNormRange]>,
    input_count: usize,
) -> Vec<Option<BigUint>> {
    let mut uppers = vec![None; input_count + 1];
    let Some(ranges) = input_max_plaintext_norm_ranges else {
        return uppers;
    };
    for range in ranges {
        let upper = &range.norm + BigUint::from(1u8);
        for slot in &mut uppers[range.start + 1..range.end + 1] {
            *slot = Some(upper.clone());
        }
    }
    uppers
}

fn lower_scoped_structured<P, L>(
    circuit: &PolyCircuit<P>,
    one: L::Wire,
    inputs: Vec<L::Wire>,
    bindings: &[SubCircuitParamValue],
    call_path: &mut Vec<usize>,
    lowering: &mut L,
    state: &mut StructuredLoweringState<L::Subgraph>,
) -> Result<Vec<L::Wire>, CircuitLowerError<L::Error>>
where
    P: Poly,
    L: StructuredCircuitLowering<P>,
{
    let mut values = BTreeMap::new();
    let mut supplied = inputs.into_iter();
    for (_, gate) in circuit.gates_in_id_order() {
        let gate_id = gate.gate_id.index();
        if values.contains_key(&gate_id) {
            continue;
        }
        let gate_instance = GateInstance::ordinary(call_path, gate.gate_id);
        let value = match &gate.gate_type {
            PolyGateType::Input if gate_id == 0 => one.clone(),
            PolyGateType::Input => supplied
                .next()
                .ok_or(CircuitLowerError::MissingInput { gate: gate_id, input: gate_id })?,
            PolyGateType::Add | PolyGateType::Sub | PolyGateType::Mul => {
                let [lhs, rhs] = lookup_binary(&values, gate)?;
                lowering
                    .binary(gate.gate_type.kind(), lhs, rhs, gate_instance)
                    .map_err(|source| CircuitLowerError::Operation { gate: gate_id, source })?
            }
            PolyGateType::SmallScalarMul { scalar } => {
                let input = lookup_unary(&values, gate)?;
                let scalar = resolve_small_scalar(scalar, bindings, gate_id)?;
                lowering
                    .small_scalar_mul(input, scalar, gate_instance)
                    .map_err(|source| CircuitLowerError::Operation { gate: gate_id, source })?
            }
            PolyGateType::LargeScalarMul { scalar } => {
                let input = lookup_unary(&values, gate)?;
                let scalar = resolve_large_scalar(scalar, bindings, gate_id)?;
                lowering
                    .large_scalar_mul(input, scalar, gate_instance)
                    .map_err(|source| CircuitLowerError::Operation { gate: gate_id, source })?
            }
            PolyGateType::SlotTransfer { src_slots } => {
                let input = lookup_unary(&values, gate)?;
                match resolve_slot_transfer_spec(src_slots, bindings, gate_id)? {
                    SlotTransferSpec::Rotation { diagonal, num_slots } => lowering
                        .slot_rotation(input, *diagonal, *num_slots, gate_instance)
                        .map_err(|source| CircuitLowerError::Operation { gate: gate_id, source })?,
                    spec => {
                        let source_slots = materialize_slot_transfer(spec);
                        lowering.slot_transfer(input, &source_slots, gate_instance).map_err(
                            |source| CircuitLowerError::Operation { gate: gate_id, source },
                        )?
                    }
                }
            }
            PolyGateType::SlotReduce { num_slots, .. } => {
                let inputs = lookup_many(&values, gate)?;
                lowering
                    .slot_reduce(&inputs, *num_slots, gate_instance)
                    .map_err(|source| CircuitLowerError::Operation { gate: gate_id, source })?
            }
            PolyGateType::PubLut { lut_id } => {
                let input = lookup_unary(&values, gate)?;
                let lookup_id = match lut_id {
                    GateParamSource::Const(lookup_id) => *lookup_id,
                    GateParamSource::Param(parameter) => {
                        return Err(CircuitLowerError::ParameterizedPublicLookup {
                            gate: gate_id,
                            parameter: *parameter,
                        });
                    }
                };
                lowering
                    .public_lookup(circuit, lookup_id, input, gate_instance)
                    .map_err(|source| CircuitLowerError::Operation { gate: gate_id, source })?
            }
            PolyGateType::SubCircuitOutput { call_id, .. } => {
                if !lowering.call_site_identity_is_semantic() {
                    let first_call = circuit
                        .sub_circuit_calls
                        .get(call_id)
                        .expect("sub-circuit call missing")
                        .clone();
                    if first_call.shared_input_prefix_set_id.is_some() {
                        let mut child_inputs = Vec::new();
                        let mut call_infos = Vec::new();
                        for (&candidate_id, candidate) in
                            circuit.sub_circuit_calls.range(*call_id..)
                        {
                            if candidate_id != *call_id + call_infos.len() ||
                                candidate.sub_circuit_id != first_call.sub_circuit_id ||
                                candidate.shared_input_prefix_set_id !=
                                    first_call.shared_input_prefix_set_id ||
                                candidate.binding_set_id != first_call.binding_set_id ||
                                candidate.input_suffix.len() != first_call.input_suffix.len() ||
                                candidate.input_max_plaintext_norm_ranges !=
                                    first_call.input_max_plaintext_norm_ranges ||
                                candidate.num_outputs != first_call.num_outputs
                            {
                                break;
                            }

                            let info = circuit.sub_circuit_call_info(candidate_id);
                            if info
                                .inputs
                                .iter()
                                .flat_map(BatchedWire::iter)
                                .any(|input| !values.contains_key(&input.index()))
                            {
                                break;
                            }
                            child_inputs.push(flatten_inputs(&values, &info.inputs, gate_id)?);
                            call_infos.push(info);
                        }
                        if call_infos.len() > 1 {
                            let child =
                                circuit.registered_sub_circuit_ref(first_call.sub_circuit_id);
                            let bindings = circuit.binding_set(first_call.binding_set_id);
                            call_path.push(first_call.scoped_call_id);
                            let outputs = call_structured_children_parallel(
                                child,
                                &one,
                                child_inputs,
                                first_call.input_max_plaintext_norm_ranges.as_deref(),
                                bindings.as_ref(),
                                call_path,
                                lowering,
                                state,
                            );
                            call_path.pop();
                            let outputs = outputs?;
                            if outputs.len() != call_infos.len() {
                                return Err(CircuitLowerError::InvalidArity { gate: gate_id });
                            }
                            for (info, outputs) in call_infos.iter().zip(outputs) {
                                insert_call_outputs(
                                    &mut values,
                                    &info.output_gate_ids,
                                    outputs,
                                    gate_id,
                                )?;
                            }
                            continue;
                        }
                    }
                }
                let info = circuit.sub_circuit_call_info(*call_id);
                let child = circuit.registered_sub_circuit_ref(info.sub_circuit_id);
                let child_inputs = flatten_inputs(&values, &info.inputs, gate_id)?;
                call_path.push(info.scoped_call_id);
                let outputs = call_structured_child(
                    child,
                    &one,
                    child_inputs,
                    info.input_max_plaintext_norm_ranges.as_deref(),
                    info.param_bindings.as_ref(),
                    call_path,
                    lowering,
                    state,
                );
                call_path.pop();
                let outputs = outputs?;
                insert_call_outputs(&mut values, &info.output_gate_ids, outputs, gate_id)?;
                continue;
            }
            PolyGateType::SummedSubCircuitOutput { summed_call_id, .. } => {
                let info = circuit.summed_sub_circuit_call_info(*summed_call_id);
                let child = circuit.registered_sub_circuit_ref(info.sub_circuit_id);
                if info.call_inputs.len() != info.scoped_call_ids.len() ||
                    info.param_bindings.len() != info.scoped_call_ids.len()
                {
                    return Err(CircuitLowerError::InvalidArity { gate: gate_id });
                }
                let mut accumulated: Option<Vec<L::Wire>> = None;
                for (inner_call, ((call_inputs, call_bindings), scoped_call_id)) in info
                    .call_inputs
                    .iter()
                    .zip(info.param_bindings.iter())
                    .zip(info.scoped_call_ids.iter())
                    .enumerate()
                {
                    let child_inputs = flatten_inputs(&values, call_inputs, gate_id)?;
                    call_path.push(*scoped_call_id);
                    let outputs = call_structured_child(
                        child.clone(),
                        &one,
                        child_inputs,
                        info.input_max_plaintext_norm_ranges.as_deref(),
                        call_bindings.as_ref(),
                        call_path,
                        lowering,
                        state,
                    );
                    call_path.pop();
                    let outputs = outputs?;
                    if let Some(current) = accumulated.as_mut() {
                        if current.len() != outputs.len() {
                            return Err(CircuitLowerError::InvalidArity { gate: gate_id });
                        }
                        for ((sum, output), output_gate) in
                            current.iter_mut().zip(outputs).zip(info.output_gate_ids.iter())
                        {
                            *sum = lowering
                                .binary(
                                    PolyGateKind::Add,
                                    sum,
                                    &output,
                                    GateInstance::occurrence(call_path, *output_gate, inner_call),
                                )
                                .map_err(|source| CircuitLowerError::Operation {
                                    gate: gate_id,
                                    source,
                                })?;
                        }
                    } else {
                        accumulated = Some(outputs);
                    }
                }
                let outputs =
                    accumulated.ok_or(CircuitLowerError::InvalidArity { gate: gate_id })?;
                insert_call_outputs(&mut values, &info.output_gate_ids, outputs, gate_id)?;
                continue;
            }
        };
        values.insert(gate_id, value);
    }
    if supplied.next().is_some() {
        return Err(CircuitLowerError::ExtraInputs);
    }
    collect_outputs(circuit, &values)
}

fn lower_scoped<P, L>(
    circuit: &PolyCircuit<P>,
    one: L::Wire,
    inputs: Vec<L::Wire>,
    bindings: &[SubCircuitParamValue],
    call_path: &mut Vec<usize>,
    lowering: &mut L,
) -> Result<Vec<L::Wire>, CircuitLowerError<L::Error>>
where
    P: Poly,
    L: GraphCircuitLowering<P>,
{
    let mut values = BTreeMap::new();
    let mut supplied = inputs.into_iter();
    for (_, gate) in circuit.gates_in_id_order() {
        let gate_id = gate.gate_id.index();
        if values.contains_key(&gate_id) {
            continue;
        }
        let gate_instance = GateInstance::ordinary(call_path, gate.gate_id);
        let value = match &gate.gate_type {
            PolyGateType::Input if gate_id == 0 => one.clone(),
            PolyGateType::Input => supplied
                .next()
                .ok_or(CircuitLowerError::MissingInput { gate: gate_id, input: gate_id })?,
            PolyGateType::Add | PolyGateType::Sub | PolyGateType::Mul => {
                let [lhs, rhs] = lookup_binary(&values, gate)?;
                lowering
                    .binary(gate.gate_type.kind(), lhs, rhs, gate_instance)
                    .map_err(|source| CircuitLowerError::Operation { gate: gate_id, source })?
            }
            PolyGateType::SmallScalarMul { scalar } => {
                let input = lookup_unary(&values, gate)?;
                let scalar = resolve_small_scalar(scalar, bindings, gate_id)?;
                lowering
                    .small_scalar_mul(input, scalar, gate_instance)
                    .map_err(|source| CircuitLowerError::Operation { gate: gate_id, source })?
            }
            PolyGateType::LargeScalarMul { scalar } => {
                let input = lookup_unary(&values, gate)?;
                let scalar = resolve_large_scalar(scalar, bindings, gate_id)?;
                lowering
                    .large_scalar_mul(input, scalar, gate_instance)
                    .map_err(|source| CircuitLowerError::Operation { gate: gate_id, source })?
            }
            PolyGateType::SlotTransfer { src_slots } => {
                let input = lookup_unary(&values, gate)?;
                match resolve_slot_transfer_spec(src_slots, bindings, gate_id)? {
                    SlotTransferSpec::Rotation { diagonal, num_slots } => lowering
                        .slot_rotation(input, *diagonal, *num_slots, gate_instance)
                        .map_err(|source| CircuitLowerError::Operation { gate: gate_id, source })?,
                    spec => {
                        let source_slots = materialize_slot_transfer(spec);
                        lowering.slot_transfer(input, &source_slots, gate_instance).map_err(
                            |source| CircuitLowerError::Operation { gate: gate_id, source },
                        )?
                    }
                }
            }
            PolyGateType::SlotReduce { num_slots, .. } => {
                let inputs = lookup_many(&values, gate)?;
                lowering
                    .slot_reduce(&inputs, *num_slots, gate_instance)
                    .map_err(|source| CircuitLowerError::Operation { gate: gate_id, source })?
            }
            PolyGateType::PubLut { lut_id } => {
                let input = lookup_unary(&values, gate)?;
                let lookup_id = match lut_id {
                    GateParamSource::Const(lookup_id) => *lookup_id,
                    GateParamSource::Param(parameter) => {
                        return Err(CircuitLowerError::ParameterizedPublicLookup {
                            gate: gate_id,
                            parameter: *parameter,
                        });
                    }
                };
                lowering
                    .public_lookup(circuit, lookup_id, input, gate_instance)
                    .map_err(|source| CircuitLowerError::Operation { gate: gate_id, source })?
            }
            PolyGateType::SubCircuitOutput { call_id, .. } => {
                let info = circuit.sub_circuit_call_info(*call_id);
                let child = circuit.registered_sub_circuit_ref(info.sub_circuit_id);
                let child_inputs = flatten_inputs(&values, &info.inputs, gate_id)?;
                let child_inputs = lowering
                    .enter_subcircuit_inputs(
                        child_inputs,
                        child
                            .sub_circuit_input_max_plaintext_norm_ranges
                            .as_deref()
                            .or(info.input_max_plaintext_norm_ranges.as_deref()),
                    )
                    .map_err(|source| CircuitLowerError::Operation { gate: gate_id, source })?;
                call_path.push(info.scoped_call_id);
                let outputs = lower_scoped(
                    child.as_ref(),
                    one.clone(),
                    child_inputs,
                    info.param_bindings.as_ref(),
                    call_path,
                    lowering,
                );
                call_path.pop();
                let outputs = outputs?;
                insert_call_outputs(&mut values, &info.output_gate_ids, outputs, gate_id)?;
                continue;
            }
            PolyGateType::SummedSubCircuitOutput { summed_call_id, .. } => {
                let info = circuit.summed_sub_circuit_call_info(*summed_call_id);
                let child = circuit.registered_sub_circuit_ref(info.sub_circuit_id);
                if info.call_inputs.len() != info.scoped_call_ids.len() ||
                    info.param_bindings.len() != info.scoped_call_ids.len()
                {
                    return Err(CircuitLowerError::InvalidArity { gate: gate_id });
                }
                let mut accumulated: Option<Vec<L::Wire>> = None;
                for (inner_call, ((call_inputs, call_bindings), scoped_call_id)) in info
                    .call_inputs
                    .iter()
                    .zip(info.param_bindings.iter())
                    .zip(info.scoped_call_ids.iter())
                    .enumerate()
                {
                    let child_inputs = flatten_inputs(&values, call_inputs, gate_id)?;
                    let child_inputs = lowering
                        .enter_subcircuit_inputs(
                            child_inputs,
                            child
                                .sub_circuit_input_max_plaintext_norm_ranges
                                .as_deref()
                                .or(info.input_max_plaintext_norm_ranges.as_deref()),
                        )
                        .map_err(|source| CircuitLowerError::Operation { gate: gate_id, source })?;
                    call_path.push(*scoped_call_id);
                    let outputs = lower_scoped(
                        child.as_ref(),
                        one.clone(),
                        child_inputs,
                        call_bindings.as_ref(),
                        call_path,
                        lowering,
                    );
                    call_path.pop();
                    let outputs = outputs?;
                    if let Some(current) = accumulated.as_mut() {
                        if current.len() != outputs.len() {
                            return Err(CircuitLowerError::InvalidArity { gate: gate_id });
                        }
                        for ((sum, output), output_gate) in
                            current.iter_mut().zip(outputs).zip(info.output_gate_ids.iter())
                        {
                            *sum = lowering
                                .binary(
                                    PolyGateKind::Add,
                                    sum,
                                    &output,
                                    GateInstance::occurrence(call_path, *output_gate, inner_call),
                                )
                                .map_err(|source| CircuitLowerError::Operation {
                                    gate: gate_id,
                                    source,
                                })?;
                        }
                    } else {
                        accumulated = Some(outputs);
                    }
                }
                let outputs =
                    accumulated.ok_or(CircuitLowerError::InvalidArity { gate: gate_id })?;
                insert_call_outputs(&mut values, &info.output_gate_ids, outputs, gate_id)?;
                continue;
            }
        };
        values.insert(gate_id, value);
    }
    if supplied.next().is_some() {
        return Err(CircuitLowerError::ExtraInputs);
    }
    collect_outputs(circuit, &values)
}

fn resolve_small_scalar<'a, E>(
    source: &'a GateParamSource<Vec<u32>>,
    bindings: &'a [SubCircuitParamValue],
    gate: usize,
) -> Result<&'a [u32], CircuitLowerError<E>>
where
    E: Error + Send + Sync + 'static,
{
    match source {
        GateParamSource::Const(value) => Ok(value),
        GateParamSource::Param(parameter) => match bindings.get(*parameter) {
            Some(SubCircuitParamValue::SmallScalarMul(value)) => Ok(value),
            Some(actual) => Err(CircuitLowerError::ParameterKind {
                gate,
                parameter: *parameter,
                expected: "small scalar",
                actual: parameter_kind(actual),
            }),
            None => Err(CircuitLowerError::MissingParameter { gate, parameter: *parameter }),
        },
    }
}

fn resolve_large_scalar<'a, E>(
    source: &'a GateParamSource<Vec<BigUint>>,
    bindings: &'a [SubCircuitParamValue],
    gate: usize,
) -> Result<&'a [BigUint], CircuitLowerError<E>>
where
    E: Error + Send + Sync + 'static,
{
    match source {
        GateParamSource::Const(value) => Ok(value),
        GateParamSource::Param(parameter) => match bindings.get(*parameter) {
            Some(SubCircuitParamValue::LargeScalarMul(value)) => Ok(value),
            Some(actual) => Err(CircuitLowerError::ParameterKind {
                gate,
                parameter: *parameter,
                expected: "large scalar",
                actual: parameter_kind(actual),
            }),
            None => Err(CircuitLowerError::MissingParameter { gate, parameter: *parameter }),
        },
    }
}

fn resolve_slot_transfer_spec<'a, E>(
    source: &'a GateParamSource<SlotTransferSpec>,
    bindings: &'a [SubCircuitParamValue],
    gate: usize,
) -> Result<&'a SlotTransferSpec, CircuitLowerError<E>>
where
    E: Error + Send + Sync + 'static,
{
    match source {
        GateParamSource::Const(value) => Ok(value),
        GateParamSource::Param(parameter) => match bindings.get(*parameter) {
            Some(SubCircuitParamValue::SlotTransfer(value)) => Ok(value),
            Some(actual) => Err(CircuitLowerError::ParameterKind {
                gate,
                parameter: *parameter,
                expected: "slot transfer",
                actual: parameter_kind(actual),
            }),
            None => Err(CircuitLowerError::MissingParameter { gate, parameter: *parameter }),
        },
    }
}

fn materialize_slot_transfer(spec: &SlotTransferSpec) -> Cow<'_, [(u32, Option<u32>)]> {
    match spec {
        SlotTransferSpec::Explicit(value) => Cow::Borrowed(value.as_slice()),
        value => Cow::Owned(value.materialize()),
    }
}

fn parameter_kind(value: &SubCircuitParamValue) -> &'static str {
    match value {
        SubCircuitParamValue::SmallScalarMul(_) => "small scalar",
        SubCircuitParamValue::LargeScalarMul(_) => "large scalar",
        SubCircuitParamValue::SlotTransfer(_) => "slot transfer",
    }
}

fn lookup_many<T: Clone, E>(
    values: &BTreeMap<usize, T>,
    gate: &PolyGate,
) -> Result<Vec<T>, CircuitLowerError<E>>
where
    E: Error + Send + Sync + 'static,
{
    gate.input_gates
        .iter()
        .map(|input| {
            values.get(&input.index()).cloned().ok_or(CircuitLowerError::MissingInput {
                gate: gate.gate_id.index(),
                input: input.index(),
            })
        })
        .collect()
}

fn flatten_inputs<T: Clone, E>(
    values: &BTreeMap<usize, T>,
    batches: &[BatchedWire],
    gate: usize,
) -> Result<Vec<T>, CircuitLowerError<E>>
where
    E: Error + Send + Sync + 'static,
{
    batches
        .iter()
        .flat_map(|batch| batch.gate_ids())
        .map(|input| {
            values
                .get(&input.index())
                .cloned()
                .ok_or(CircuitLowerError::MissingInput { gate, input: input.index() })
        })
        .collect()
}

fn insert_call_outputs<T, E>(
    values: &mut BTreeMap<usize, T>,
    output_ids: &[GateId],
    outputs: Vec<T>,
    gate: usize,
) -> Result<(), CircuitLowerError<E>>
where
    E: Error + Send + Sync + 'static,
{
    if output_ids.len() != outputs.len() {
        return Err(CircuitLowerError::InvalidArity { gate });
    }
    values.extend(output_ids.iter().map(|id| id.index()).zip(outputs));
    Ok(())
}

fn lookup_unary<'a, T, E>(
    values: &'a BTreeMap<usize, T>,
    gate: &PolyGate,
) -> Result<&'a T, CircuitLowerError<E>>
where
    E: Error + Send + Sync + 'static,
{
    let [input] = gate.input_gates.as_slice() else {
        return Err(CircuitLowerError::InvalidArity { gate: gate.gate_id.index() });
    };
    values
        .get(&input.index())
        .ok_or(CircuitLowerError::MissingInput { gate: gate.gate_id.index(), input: input.index() })
}

fn lookup_binary<'a, T, E>(
    values: &'a BTreeMap<usize, T>,
    gate: &PolyGate,
) -> Result<[&'a T; 2], CircuitLowerError<E>>
where
    E: Error + Send + Sync + 'static,
{
    let [lhs, rhs] = gate.input_gates.as_slice() else {
        return Err(CircuitLowerError::InvalidArity { gate: gate.gate_id.index() });
    };
    Ok([
        values.get(&lhs.index()).ok_or(CircuitLowerError::MissingInput {
            gate: gate.gate_id.index(),
            input: lhs.index(),
        })?,
        values.get(&rhs.index()).ok_or(CircuitLowerError::MissingInput {
            gate: gate.gate_id.index(),
            input: rhs.index(),
        })?,
    ])
}

fn collect_outputs<P, T, E>(
    circuit: &PolyCircuit<P>,
    values: &BTreeMap<usize, T>,
) -> Result<Vec<T>, CircuitLowerError<E>>
where
    P: Poly,
    T: Clone,
    E: Error + Send + Sync + 'static,
{
    circuit
        .output_gate_ids()
        .iter()
        .map(|id| {
            values
                .get(&id.index())
                .cloned()
                .ok_or(CircuitLowerError::MissingInput { gate: id.index(), input: id.index() })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        circuit::{PolyGateType, SubCircuitParamSpec},
        poly::dcrt::poly::DCRTPoly,
    };
    use std::convert::Infallible;

    struct RecordingLowering {
        operations: Vec<(PolyGateKind, Vec<usize>, usize, usize)>,
        slot_mapping_pointers: Vec<usize>,
        slot_rotations: Vec<(u32, u32)>,
        small_scalars: Vec<Vec<u32>>,
        large_scalars: Vec<Vec<BigUint>>,
        call_site_identity_is_semantic: bool,
        sequential_subgraph_calls: usize,
        parallel_subgraph_batch_sizes: Vec<usize>,
        audited_constant_lut_contracts: Vec<Vec<Option<BigUint>>>,
    }

    impl Default for RecordingLowering {
        fn default() -> Self {
            Self {
                operations: Vec::new(),
                slot_mapping_pointers: Vec::new(),
                slot_rotations: Vec::new(),
                small_scalars: Vec::new(),
                large_scalars: Vec::new(),
                call_site_identity_is_semantic: true,
                sequential_subgraph_calls: 0,
                parallel_subgraph_batch_sizes: Vec::new(),
                audited_constant_lut_contracts: Vec::new(),
            }
        }
    }

    impl RecordingLowering {
        fn record(&mut self, kind: PolyGateKind, gate: GateInstance<'_>) {
            self.operations.push((
                kind,
                gate.call_path().to_vec(),
                gate.local_gate().index(),
                gate.operation_occurrence(),
            ));
        }
    }

    impl CircuitLoweringTypes for RecordingLowering {
        type Wire = usize;
        type Error = Infallible;
    }

    impl ArithmeticCircuitLowering<DCRTPoly> for RecordingLowering {
        fn binary(
            &mut self,
            operation: PolyGateKind,
            lhs: &Self::Wire,
            rhs: &Self::Wire,
            gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            self.record(operation, gate);
            Ok(lhs + rhs)
        }

        fn small_scalar_mul(
            &mut self,
            input: &Self::Wire,
            scalar: &[u32],
            gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            self.record(PolyGateKind::SmallScalarMul, gate);
            self.small_scalars.push(scalar.to_vec());
            Ok(*input)
        }

        fn large_scalar_mul(
            &mut self,
            input: &Self::Wire,
            scalar: &[BigUint],
            gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            self.record(PolyGateKind::LargeScalarMul, gate);
            self.large_scalars.push(scalar.to_vec());
            Ok(*input)
        }
    }

    impl SlotOperationLowering<DCRTPoly> for RecordingLowering {
        fn slot_transfer(
            &mut self,
            input: &Self::Wire,
            source_slots: &[(u32, Option<u32>)],
            gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            self.record(PolyGateKind::SlotTransfer, gate);
            self.slot_mapping_pointers.push(source_slots.as_ptr() as usize);
            Ok(*input)
        }

        fn slot_rotation(
            &mut self,
            input: &Self::Wire,
            offset: u32,
            num_slots: u32,
            gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            self.record(PolyGateKind::SlotTransfer, gate);
            self.slot_rotations.push((offset, num_slots));
            Ok(*input)
        }

        fn slot_reduce(
            &mut self,
            inputs: &[Self::Wire],
            _slot_count: usize,
            gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            self.record(PolyGateKind::SlotReduce, gate);
            Ok(inputs.iter().sum())
        }
    }

    #[derive(Default)]
    struct DefaultRotationLowering {
        mappings: Vec<Vec<(u32, Option<u32>)>>,
    }

    impl CircuitLoweringTypes for DefaultRotationLowering {
        type Wire = usize;
        type Error = Infallible;
    }

    impl SlotOperationLowering<DCRTPoly> for DefaultRotationLowering {
        fn slot_transfer(
            &mut self,
            input: &Self::Wire,
            source_slots: &[(u32, Option<u32>)],
            _gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            self.mappings.push(source_slots.to_vec());
            Ok(*input)
        }

        fn slot_reduce(
            &mut self,
            inputs: &[Self::Wire],
            _slot_count: usize,
            _gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            Ok(inputs.iter().sum())
        }
    }

    impl PublicLookupLowering<DCRTPoly> for RecordingLowering {
        fn public_lookup(
            &mut self,
            _circuit: &PolyCircuit<DCRTPoly>,
            _lookup_id: usize,
            input: &Self::Wire,
            gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            self.record(PolyGateKind::PubLut, gate);
            Ok(*input)
        }
    }

    impl StructuredCircuitLowering<DCRTPoly> for RecordingLowering {
        type Subgraph = Vec<usize>;

        fn call_site_identity_is_semantic(&self) -> bool {
            self.call_site_identity_is_semantic
        }

        fn define_subgraph<F>(
            &mut self,
            _name: &str,
            input_examples: &[Self::Wire],
            body: F,
        ) -> Result<Self::Subgraph, CircuitLowerError<Self::Error>>
        where
            F: FnOnce(
                &mut Self,
                Vec<Self::Wire>,
            ) -> Result<Vec<Self::Wire>, CircuitLowerError<Self::Error>>,
        {
            body(self, input_examples.to_vec())
        }

        fn call_subgraph(
            &mut self,
            definition: &Self::Subgraph,
            _inputs: Vec<Self::Wire>,
        ) -> Result<Vec<Self::Wire>, CircuitLowerError<Self::Error>> {
            self.sequential_subgraph_calls += 1;
            Ok(definition.clone())
        }

        fn call_audited_constant_lut_subgraph(
            &mut self,
            definition: &Self::Subgraph,
            inputs: Vec<Self::Wire>,
            canonical_input_exclusive_uppers: Vec<Option<BigUint>>,
        ) -> Result<Vec<Self::Wire>, CircuitLowerError<Self::Error>> {
            assert_eq!(inputs.len(), canonical_input_exclusive_uppers.len());
            self.sequential_subgraph_calls += 1;
            self.audited_constant_lut_contracts.push(canonical_input_exclusive_uppers);
            Ok(definition.clone())
        }

        fn call_subgraph_parallel(
            &mut self,
            definition: &Self::Subgraph,
            inputs: Vec<Vec<Self::Wire>>,
        ) -> Result<Vec<Vec<Self::Wire>>, CircuitLowerError<Self::Error>> {
            self.parallel_subgraph_batch_sizes.push(inputs.len());
            Ok(inputs.into_iter().map(|_| definition.clone()).collect())
        }

        fn call_audited_constant_lut_subgraph_parallel(
            &mut self,
            definition: &Self::Subgraph,
            inputs: Vec<Vec<Self::Wire>>,
            canonical_input_exclusive_uppers: Vec<Option<BigUint>>,
        ) -> Result<Vec<Vec<Self::Wire>>, CircuitLowerError<Self::Error>> {
            assert!(
                inputs.iter().all(|inputs| inputs.len() == canonical_input_exclusive_uppers.len())
            );
            self.parallel_subgraph_batch_sizes.push(inputs.len());
            self.audited_constant_lut_contracts.push(canonical_input_exclusive_uppers);
            Ok(inputs.into_iter().map(|_| definition.clone()).collect())
        }
    }

    fn repeated_child_circuit() -> (PolyCircuit<DCRTPoly>, usize) {
        let mut child = PolyCircuit::<DCRTPoly>::new();
        let child_input = child.input(1).as_single_wire();
        let transferred = child.slot_transfer_gate(child_input, &[(0, None), (0, Some(3))]);
        child.output([transferred, transferred]);

        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input = circuit.input(1).as_single_wire();
        let child_id = circuit.register_sub_circuit(child);
        let first = circuit.call_sub_circuit(child_id, [input]);
        let second = circuit.call_sub_circuit(child_id, [input]);
        let input_set = circuit.intern_input_set([input]);
        let binding_set = circuit.intern_binding_set(&[]);
        let summed = circuit.call_sub_circuit_sum_many_with_binding_set_ids(
            child_id,
            vec![input_set, input_set],
            vec![binding_set, binding_set],
        );
        circuit.output([first[0], first[1], second[0], second[1], summed[0], summed[1]]);
        (circuit, child_id)
    }

    fn recorded_lowering(circuit: &PolyCircuit<DCRTPoly>) -> RecordingLowering {
        let mut lowering = RecordingLowering::default();
        let outputs =
            lower_circuit(circuit, 1, [7], &mut lowering).expect("recording lowerer is infallible");
        assert_eq!(outputs, vec![7, 7, 7, 7, 14, 14]);
        lowering
    }

    fn recorded_structured_lowering(circuit: &PolyCircuit<DCRTPoly>) -> RecordingLowering {
        let mut lowering = RecordingLowering::default();
        let outputs = lower_circuit_structured(circuit, 1, [7], &mut lowering)
            .expect("recording lowerer is infallible");
        assert_eq!(outputs, vec![7, 7, 7, 7, 14, 14]);
        lowering
    }

    #[test]
    fn repeated_sub_circuit_gate_instances_are_distinct_and_deterministic() {
        let (circuit, child_id) = repeated_child_circuit();
        let first = recorded_lowering(&circuit);
        let second = recorded_lowering(&circuit);
        assert_eq!(first.operations, second.operations);

        let slot_instances = first
            .operations
            .iter()
            .filter(|(kind, ..)| *kind == PolyGateKind::SlotTransfer)
            .collect::<Vec<_>>();
        assert_eq!(slot_instances.len(), 4);
        let distinct_paths = slot_instances
            .iter()
            .map(|(_, path, ..)| path.clone())
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(distinct_paths.len(), 4);
        let distinct_operation_identities = first
            .operations
            .iter()
            .map(|(_, path, gate, occurrence)| (path.clone(), *gate, *occurrence))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            distinct_operation_identities.len(),
            first.operations.len(),
            "every emitted operation needs a distinct structural identity"
        );

        let child = circuit.registered_sub_circuit_ref(child_id);
        let stored_mapping_pointer = child
            .gates_in_id_order()
            .find_map(|(_, gate)| match &gate.gate_type {
                PolyGateType::SlotTransfer {
                    src_slots: GateParamSource::Const(SlotTransferSpec::Explicit(mapping)),
                } => Some(mapping.as_ptr() as usize),
                _ => None,
            })
            .expect("child slot-transfer mapping");
        assert!(
            first.slot_mapping_pointers.iter().all(|pointer| *pointer == stored_mapping_pointer),
            "explicit slot mappings must be borrowed rather than cloned"
        );
    }

    #[test]
    fn rotation_specs_preserve_specialized_dispatch_in_both_traversals() {
        let mut child = PolyCircuit::<DCRTPoly>::new();
        let rotation =
            child.register_sub_circuit_param(SubCircuitParamSpec::SlotTransfer { max_scalar: 1 });
        let input = child.input(1).as_single_wire();
        let rotated = child.slot_transfer_gate_param(input, rotation);
        child.output([rotated]);

        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input = circuit.input(1).as_single_wire();
        let constant = circuit.slot_rotation_gate(input, 5, 4);
        let child_id = circuit.register_sub_circuit(child);
        let parameterized = circuit.call_sub_circuit_with_bindings(
            child_id,
            [constant],
            &[SubCircuitParamValue::SlotTransfer(SlotTransferSpec::rotation(7, 6))],
        );
        circuit.output(parameterized);

        for structured in [false, true] {
            let mut lowering = RecordingLowering::default();
            let outputs = if structured {
                lower_circuit_structured(&circuit, 1usize, [11], &mut lowering)
            } else {
                lower_circuit(&circuit, 1usize, [11], &mut lowering)
            }
            .expect("rotation lowering must be infallible");
            assert_eq!(outputs, vec![11]);
            assert_eq!(lowering.slot_rotations, vec![(5, 4), (7, 6)]);
            assert!(
                lowering.slot_mapping_pointers.is_empty(),
                "rotation specs must not be materialized before specialized dispatch"
            );
        }
    }

    #[test]
    fn default_rotation_lowering_matches_an_explicit_slot_transfer() {
        let expected = SlotTransferSpec::rotation(5, 4).materialize();
        let mut lowering = DefaultRotationLowering::default();
        let input = 17usize;
        let rotated = lowering
            .slot_rotation(&input, 5, 4, GateInstance::ordinary(&[], GateId(1)))
            .expect("default rotation lowering must be infallible");
        let transferred = lowering
            .slot_transfer(&input, &expected, GateInstance::ordinary(&[], GateId(2)))
            .expect("explicit transfer lowering must be infallible");
        assert_eq!(rotated, transferred);
        assert_eq!(lowering.mappings, vec![expected.clone(), expected]);
    }

    #[test]
    fn repeated_lanes_materializes_lane_preserving_broadcast() {
        let spec = SlotTransferSpec::repeated_lanes(1, 3, 2, 2, Some(7));
        assert_eq!(
            spec.materialize(),
            vec![(2, Some(7)), (3, Some(7)), (2, Some(7)), (3, Some(7)), (2, None), (3, None),]
        );
    }

    #[test]
    fn structured_sensitive_subcircuits_preserve_each_invocation_identity() {
        let (circuit, _) = repeated_child_circuit();
        let first = recorded_structured_lowering(&circuit);
        let second = recorded_structured_lowering(&circuit);
        assert_eq!(first.operations, second.operations);

        let slot_instances = first
            .operations
            .iter()
            .filter(|(kind, ..)| *kind == PolyGateKind::SlotTransfer)
            .collect::<Vec<_>>();
        assert_eq!(slot_instances.len(), 4);
        let distinct_paths = slot_instances
            .iter()
            .map(|(_, path, ..)| path.clone())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(distinct_paths.len(), 4);
        assert_eq!(
            first.operations,
            recorded_lowering(&circuit).operations,
            "structured lowering must preserve the flat lowering identities for artifact-sensitive gates"
        );
    }

    #[test]
    fn structured_parallel_calls_stop_before_a_data_dependency() {
        let mut child = PolyCircuit::<DCRTPoly>::new();
        let child_inputs = child.input(2).to_vec();
        let output = child.add_gate(child_inputs[0], child_inputs[1]);
        child.output([output]);

        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(2).to_vec();
        let child_id = circuit.register_sub_circuit(child);
        let shared_prefix = circuit.intern_input_set([inputs[0]]);
        let first = circuit.call_sub_circuit_with_shared_input_prefix_and_bindings(
            child_id,
            shared_prefix,
            [inputs[1]],
            &[],
        );
        let second = circuit.call_sub_circuit_with_shared_input_prefix_and_bindings(
            child_id,
            shared_prefix,
            [first[0]],
            &[],
        );
        circuit.output(second);

        let mut lowering = RecordingLowering {
            call_site_identity_is_semantic: false,
            ..RecordingLowering::default()
        };
        lower_circuit_structured(&circuit, 1usize, [3, 5], &mut lowering)
            .expect("dependent sub-circuit calls must remain sequentially lowerable");
        assert_eq!(lowering.sequential_subgraph_calls, 2);
        assert!(lowering.parallel_subgraph_batch_sizes.is_empty());
    }

    #[test]
    fn structured_independent_calls_share_one_parallel_batch() {
        let mut child = PolyCircuit::<DCRTPoly>::new();
        let child_inputs = child.input(2).to_vec();
        let output = child.add_gate(child_inputs[0], child_inputs[1]);
        child.output([output]);

        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(3).to_vec();
        let child_id = circuit.register_sub_circuit(child);
        let shared_prefix = circuit.intern_input_set([inputs[0]]);
        let first = circuit.call_sub_circuit_with_shared_input_prefix_and_bindings(
            child_id,
            shared_prefix,
            [inputs[1]],
            &[],
        );
        let second = circuit.call_sub_circuit_with_shared_input_prefix_and_bindings(
            child_id,
            shared_prefix,
            [inputs[2]],
            &[],
        );
        circuit.output([first[0], second[0]]);

        let mut lowering = RecordingLowering {
            call_site_identity_is_semantic: false,
            ..RecordingLowering::default()
        };
        lower_circuit_structured(&circuit, 1usize, [3, 5, 7], &mut lowering)
            .expect("independent sub-circuit calls must lower as one parallel batch");
        assert_eq!(lowering.sequential_subgraph_calls, 0);
        assert_eq!(lowering.parallel_subgraph_batch_sizes, vec![2]);
    }

    #[test]
    fn audited_constant_lut_calls_transport_inclusive_norms_as_exclusive_uppers() {
        let mut child = PolyCircuit::<DCRTPoly>::new();
        let input = child.input(2);
        let lookup = child.slot_transfer_gate(input.at(0), &[(0, None)]).as_single_wire();
        child.gates.get_mut(&lookup).expect("lookup placeholder gate").gate_type =
            PolyGateType::PubLut { lut_id: GateParamSource::Const(0) };
        child.output([BatchedWire::single(lookup), input.at(1)]);

        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(2);
        let child_id = circuit.register_sub_circuit(child);
        let outputs = circuit.call_sub_circuit_with_max_plaintext_norms(
            child_id,
            [inputs.at(0), inputs.at(1)],
            [
                SubCircuitInputMaxPlaintextNormRange::new(0, 1, BigUint::from(6u8)),
                SubCircuitInputMaxPlaintextNormRange::new(1, 2, BigUint::from(10u8)),
            ],
        );
        circuit.output(outputs);

        let mut lowering = RecordingLowering::default();
        lower_circuit_structured(&circuit, 1usize, [3, 5], &mut lowering)
            .expect("audited constant-LUT lowering must succeed");
        assert_eq!(
            lowering.audited_constant_lut_contracts,
            vec![vec![None, Some(BigUint::from(7u8)), Some(BigUint::from(11u8))]],
        );
    }

    #[test]
    fn audited_wrapper_contract_reaches_a_descendant_lut_without_recursive_scanning() {
        let mut leaf = PolyCircuit::<DCRTPoly>::new();
        let leaf_input = leaf.input(1);
        let lookup = leaf.slot_transfer_gate(leaf_input, &[(0, None)]).as_single_wire();
        leaf.gates.get_mut(&lookup).expect("lookup placeholder gate").gate_type =
            PolyGateType::PubLut { lut_id: GateParamSource::Const(0) };
        leaf.output([lookup]);

        let mut wrapper = PolyCircuit::<DCRTPoly>::new();
        let wrapper_input = wrapper.input(1);
        let leaf_id = wrapper.register_sub_circuit(leaf);
        let wrapper_output = wrapper.call_sub_circuit(leaf_id, [wrapper_input]);
        wrapper.output(wrapper_output);

        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input = circuit.input(1);
        let child_id = circuit.register_sub_circuit(wrapper);
        let outputs = circuit.call_sub_circuit_with_max_plaintext_norms(
            child_id,
            [input],
            [SubCircuitInputMaxPlaintextNormRange::new(0, 1, BigUint::from(6u8))],
        );
        circuit.output(outputs);

        let mut lowering = RecordingLowering::default();
        lower_circuit_structured(&circuit, 1usize, [3], &mut lowering)
            .expect("audited wrapper-to-leaf LUT lowering must succeed");
        assert!(
            lowering.audited_constant_lut_contracts.contains(&vec![None, Some(BigUint::from(7u8))])
        );
    }

    #[test]
    fn direct_and_summed_subcircuits_resolve_each_scalar_binding() {
        let mut child = PolyCircuit::<DCRTPoly>::new();
        let input = child.input(1).as_single_wire();
        let small =
            child.register_sub_circuit_param(SubCircuitParamSpec::SmallScalarMul { max_scalar: 9 });
        let large = child.register_sub_circuit_param(SubCircuitParamSpec::LargeScalarMul {
            max_scalar: BigUint::from(11u8),
        });
        let scaled = child.small_scalar_mul_param(input, small);
        let scaled = child.large_scalar_mul_param(scaled, large);
        child.output([scaled]);

        let first_bindings = [
            SubCircuitParamValue::SmallScalarMul(vec![2, 3]),
            SubCircuitParamValue::LargeScalarMul(vec![BigUint::from(5u8)]),
        ];
        let second_bindings = [
            SubCircuitParamValue::SmallScalarMul(vec![7]),
            SubCircuitParamValue::LargeScalarMul(vec![BigUint::from(11u8), BigUint::from(1u8)]),
        ];
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(3).to_vec();
        let child_id = circuit.register_sub_circuit(child);
        let direct = circuit.call_sub_circuit_with_bindings(child_id, [inputs[0]], &first_bindings);
        let first_input_set = circuit.intern_input_set([inputs[1]]);
        let second_input_set = circuit.intern_input_set([inputs[2]]);
        let first_binding_set = circuit.intern_binding_set(&first_bindings);
        let second_binding_set = circuit.intern_binding_set(&second_bindings);
        let summed = circuit.call_sub_circuit_sum_many_with_binding_set_ids(
            child_id,
            vec![first_input_set, second_input_set],
            vec![first_binding_set, second_binding_set],
        );
        circuit.output([direct[0], summed[0]]);

        let mut lowering = RecordingLowering::default();
        let outputs = lower_circuit(&circuit, 1usize, [13, 17, 19], &mut lowering)
            .expect("recording lowerer is infallible");
        assert_eq!(outputs, vec![13, 36]);
        assert_eq!(lowering.small_scalars, vec![vec![2, 3], vec![2, 3], vec![7]]);
        assert_eq!(
            lowering.large_scalars,
            vec![
                vec![BigUint::from(5u8)],
                vec![BigUint::from(5u8)],
                vec![BigUint::from(11u8), BigUint::from(1u8)],
            ]
        );
    }

    #[test]
    fn every_gate_category_reaches_its_lowering_trait_operation() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(2).to_vec();
        let add = circuit.add_gate(inputs[0], inputs[1]);
        let sub = circuit.sub_gate(inputs[0], inputs[1]);
        let mul = circuit.mul_gate(inputs[0], inputs[1]);
        let small = circuit.small_scalar_mul(add, &[2, 1]);
        let large = circuit.large_scalar_mul(sub, &[BigUint::from(9u8)]);
        let transferred = circuit.slot_transfer_gate(mul, &[(0, None)]);
        let reduced = circuit.slot_reduce_gate(&[small, large, transferred], 3);
        let lookup = circuit.slot_transfer_gate(reduced, &[(0, None)]).as_single_wire();
        circuit.gates.get_mut(&lookup).expect("lookup placeholder gate").gate_type =
            PolyGateType::PubLut { lut_id: GateParamSource::Const(0) };
        circuit.output([lookup]);

        let mut lowering = RecordingLowering::default();
        let outputs = lower_circuit(&circuit, 1usize, [3, 5], &mut lowering)
            .expect("recording lowerer is infallible");
        assert_eq!(outputs.len(), 1);
        let kinds = lowering
            .operations
            .iter()
            .map(|(kind, ..)| *kind)
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(
            kinds,
            [
                PolyGateKind::Add,
                PolyGateKind::Sub,
                PolyGateKind::Mul,
                PolyGateKind::SmallScalarMul,
                PolyGateKind::LargeScalarMul,
                PolyGateKind::SlotTransfer,
                PolyGateKind::SlotReduce,
                PolyGateKind::PubLut,
            ]
            .into_iter()
            .collect::<std::collections::HashSet<_>>()
        );
    }

    #[test]
    fn lowering_reports_extra_inputs_missing_inputs_and_invalid_arity() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input = circuit.input(1).as_single_wire();
        circuit.output([input]);
        let mut lowering = RecordingLowering::default();
        assert!(matches!(
            lower_circuit(&circuit, 1usize, [7, 11], &mut lowering),
            Err(CircuitLowerError::ExtraInputs)
        ));

        let mut missing = PolyCircuit::<DCRTPoly>::new();
        let inputs = missing.input(2).to_vec();
        let add = missing.add_gate(inputs[0], inputs[1]).as_single_wire();
        missing.gates.get_mut(&add).expect("add gate").input_gates[1] = GateId(999);
        missing.output([add]);
        let mut lowering = RecordingLowering::default();
        assert!(matches!(
            lower_circuit(&missing, 1usize, [7, 11], &mut lowering),
            Err(CircuitLowerError::MissingInput { input: 999, .. })
        ));

        let mut invalid = PolyCircuit::<DCRTPoly>::new();
        let inputs = invalid.input(2).to_vec();
        let add = invalid.add_gate(inputs[0], inputs[1]).as_single_wire();
        invalid.gates.get_mut(&add).expect("add gate").input_gates.pop();
        invalid.output([add]);
        let mut lowering = RecordingLowering::default();
        assert!(matches!(
            lower_circuit(&invalid, 1usize, [7, 11], &mut lowering),
            Err(CircuitLowerError::InvalidArity { gate }) if gate == add.index()
        ));
    }

    #[test]
    fn lowering_reports_missing_wrong_kind_and_unsupported_lookup_parameters() {
        let mut missing = PolyCircuit::<DCRTPoly>::new();
        let parameter = missing
            .register_sub_circuit_param(SubCircuitParamSpec::SmallScalarMul { max_scalar: 7 });
        let input = missing.input(1).as_single_wire();
        let output = missing.small_scalar_mul_param(input, parameter);
        missing.output([output]);
        let mut lowering = RecordingLowering::default();
        assert!(matches!(
            lower_circuit(&missing, 1usize, [5], &mut lowering),
            Err(CircuitLowerError::MissingParameter { parameter: 0, .. })
        ));

        let build_parent = |gate_type: PolyGateType| {
            let mut child = PolyCircuit::<DCRTPoly>::new();
            let parameter = child
                .register_sub_circuit_param(SubCircuitParamSpec::SmallScalarMul { max_scalar: 7 });
            let input = child.input(1).as_single_wire();
            let output = child.small_scalar_mul_param(input, parameter).as_single_wire();
            child.gates.get_mut(&output).expect("parameterized gate").gate_type = gate_type;
            child.output([output]);

            let mut parent = PolyCircuit::<DCRTPoly>::new();
            let input = parent.input(1).as_single_wire();
            let child_id = parent.register_sub_circuit(child);
            let outputs = parent.call_sub_circuit_with_bindings(
                child_id,
                [input],
                &[SubCircuitParamValue::SmallScalarMul(vec![3])],
            );
            parent.output(outputs);
            parent
        };

        let wrong_kind =
            build_parent(PolyGateType::LargeScalarMul { scalar: GateParamSource::Param(0) });
        let mut lowering = RecordingLowering::default();
        assert!(matches!(
            lower_circuit(&wrong_kind, 1usize, [5], &mut lowering),
            Err(CircuitLowerError::ParameterKind { parameter: 0, .. })
        ));

        let parameterized_lookup =
            build_parent(PolyGateType::PubLut { lut_id: GateParamSource::Param(0) });
        let mut lowering = RecordingLowering::default();
        assert!(matches!(
            lower_circuit(&parameterized_lookup, 1usize, [5], &mut lowering),
            Err(CircuitLowerError::ParameterizedPublicLookup { parameter: 0, .. })
        ));
    }
}
