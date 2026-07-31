use super::{
    BatchedWire, GateParamSource, PolyCircuit, PolyGate, PolyGateKind, PolyGateType,
    SlotTransferSpec, SubCircuitParamValue, gate::GateId,
};
use crate::Poly;
use mxx_ir_core::GraphBuilder;
use num_bigint::BigUint;
use std::{borrow::Cow, collections::BTreeMap, error::Error};
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
pub trait GraphCircuitLowering<P: Poly> {
    type Wire: Clone;
    type Error: Error + Send + Sync + 'static;

    fn binary(
        &mut self,
        builder: &mut GraphBuilder,
        operation: PolyGateKind,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error>;

    fn small_scalar_mul(
        &mut self,
        builder: &mut GraphBuilder,
        input: &Self::Wire,
        scalar: &[u32],
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error>;

    fn large_scalar_mul(
        &mut self,
        builder: &mut GraphBuilder,
        input: &Self::Wire,
        scalar: &[BigUint],
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error>;

    fn slot_transfer(
        &mut self,
        builder: &mut GraphBuilder,
        input: &Self::Wire,
        source_slots: &[(u32, Option<u32>)],
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error>;

    fn slot_reduce(
        &mut self,
        builder: &mut GraphBuilder,
        inputs: &[Self::Wire],
        slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error>;

    fn public_lookup(
        &mut self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        lookup_id: usize,
        input: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error>;
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
    builder: &mut GraphBuilder,
    circuit: &PolyCircuit<P>,
    one: L::Wire,
    inputs: impl IntoIterator<Item = L::Wire>,
    lowering: &mut L,
) -> Result<Vec<L::Wire>, CircuitLowerError<L::Error>>
where
    P: Poly,
    L: GraphCircuitLowering<P>,
{
    lower_scoped(
        builder,
        circuit,
        one,
        inputs.into_iter().collect(),
        &[],
        &mut Vec::new(),
        lowering,
    )
}

fn lower_scoped<P, L>(
    builder: &mut GraphBuilder,
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
                    .binary(builder, gate.gate_type.kind(), lhs, rhs, gate_instance)
                    .map_err(|source| CircuitLowerError::Operation { gate: gate_id, source })?
            }
            PolyGateType::SmallScalarMul { scalar } => {
                let input = lookup_unary(&values, gate)?;
                let scalar = resolve_small_scalar(scalar, bindings, gate_id)?;
                lowering
                    .small_scalar_mul(builder, input, scalar, gate_instance)
                    .map_err(|source| CircuitLowerError::Operation { gate: gate_id, source })?
            }
            PolyGateType::LargeScalarMul { scalar } => {
                let input = lookup_unary(&values, gate)?;
                let scalar = resolve_large_scalar(scalar, bindings, gate_id)?;
                lowering
                    .large_scalar_mul(builder, input, scalar, gate_instance)
                    .map_err(|source| CircuitLowerError::Operation { gate: gate_id, source })?
            }
            PolyGateType::SlotTransfer { src_slots } => {
                let input = lookup_unary(&values, gate)?;
                let source_slots = resolve_slot_transfer(src_slots, bindings, gate_id)?;
                lowering
                    .slot_transfer(builder, input, &source_slots, gate_instance)
                    .map_err(|source| CircuitLowerError::Operation { gate: gate_id, source })?
            }
            PolyGateType::SlotReduce { num_slots, .. } => {
                let inputs = lookup_many(&values, gate)?;
                lowering
                    .slot_reduce(builder, &inputs, *num_slots, gate_instance)
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
                    .public_lookup(builder, circuit, lookup_id, input, gate_instance)
                    .map_err(|source| CircuitLowerError::Operation { gate: gate_id, source })?
            }
            PolyGateType::SubCircuitOutput { call_id, .. } => {
                let info = circuit.sub_circuit_call_info(*call_id);
                let child = circuit.registered_sub_circuit_ref(info.sub_circuit_id);
                let child_inputs = flatten_inputs(&values, &info.inputs, gate_id)?;
                call_path.push(info.scoped_call_id);
                let outputs = lower_scoped(
                    builder,
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
                    call_path.push(*scoped_call_id);
                    let outputs = lower_scoped(
                        builder,
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
                                    builder,
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

fn resolve_slot_transfer<'a, E>(
    source: &'a GateParamSource<SlotTransferSpec>,
    bindings: &'a [SubCircuitParamValue],
    gate: usize,
) -> Result<Cow<'a, [(u32, Option<u32>)]>, CircuitLowerError<E>>
where
    E: Error + Send + Sync + 'static,
{
    match source {
        GateParamSource::Const(SlotTransferSpec::Explicit(value)) => {
            Ok(Cow::Borrowed(value.as_slice()))
        }
        GateParamSource::Const(value) => Ok(Cow::Owned(value.materialize())),
        GateParamSource::Param(parameter) => match bindings.get(*parameter) {
            Some(SubCircuitParamValue::SlotTransfer(SlotTransferSpec::Explicit(value))) => {
                Ok(Cow::Borrowed(value.as_slice()))
            }
            Some(SubCircuitParamValue::SlotTransfer(value)) => Ok(Cow::Owned(value.materialize())),
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

    #[derive(Default)]
    struct RecordingLowering {
        operations: Vec<(PolyGateKind, Vec<usize>, usize, usize)>,
        slot_mapping_pointers: Vec<usize>,
        small_scalars: Vec<Vec<u32>>,
        large_scalars: Vec<Vec<BigUint>>,
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

    impl GraphCircuitLowering<DCRTPoly> for RecordingLowering {
        type Wire = usize;
        type Error = Infallible;

        fn binary(
            &mut self,
            _builder: &mut GraphBuilder,
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
            _builder: &mut GraphBuilder,
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
            _builder: &mut GraphBuilder,
            input: &Self::Wire,
            scalar: &[BigUint],
            gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            self.record(PolyGateKind::LargeScalarMul, gate);
            self.large_scalars.push(scalar.to_vec());
            Ok(*input)
        }

        fn slot_transfer(
            &mut self,
            _builder: &mut GraphBuilder,
            input: &Self::Wire,
            source_slots: &[(u32, Option<u32>)],
            gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            self.record(PolyGateKind::SlotTransfer, gate);
            self.slot_mapping_pointers.push(source_slots.as_ptr() as usize);
            Ok(*input)
        }

        fn slot_reduce(
            &mut self,
            _builder: &mut GraphBuilder,
            inputs: &[Self::Wire],
            _slot_count: usize,
            gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            self.record(PolyGateKind::SlotReduce, gate);
            Ok(inputs.iter().sum())
        }

        fn public_lookup(
            &mut self,
            _builder: &mut GraphBuilder,
            _circuit: &PolyCircuit<DCRTPoly>,
            _lookup_id: usize,
            input: &Self::Wire,
            gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            self.record(PolyGateKind::PubLut, gate);
            Ok(*input)
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
        let mut builder = GraphBuilder::new("recording-lowering", Vec::new());
        let mut lowering = RecordingLowering::default();
        let outputs = lower_circuit(&mut builder, circuit, 1, [7], &mut lowering)
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
            .collect::<std::collections::BTreeSet<_>>();
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

        let mut builder = GraphBuilder::new("bound-summed-lowering", Vec::new());
        let mut lowering = RecordingLowering::default();
        let outputs = lower_circuit(&mut builder, &circuit, 1usize, [13, 17, 19], &mut lowering)
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
}
