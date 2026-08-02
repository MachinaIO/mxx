use crate::{
    circuit::{
        BatchedWire, GateParamSource, PolyCircuit, PolyGate, PolyGateType, SubCircuitCall,
        SubCircuitParamSpec, SubCircuitParamValue,
        gate::{GateId, SlotTransferSpec},
    },
    poly::Poly,
};
use num_bigint::BigUint;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json;
use std::{collections::BTreeMap, fs, path::Path, sync::Arc};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SerializablePolyGateType {
    Input,
    SmallScalarMul { scalar: GateParamSource<Vec<u32>> },
    LargeScalarMul { scalar: GateParamSource<Vec<BigUint>> },
    SlotTransfer { src_slots: GateParamSource<SlotTransferSpec> },
    SlotReduce { num_slots: usize, input_count: usize },
    Add,
    Sub,
    Mul,
    PubLut { lut_id: usize },
    SubCircuitOutput { call_id: usize, output_idx: usize, num_inputs: usize },
    SummedSubCircuitOutput { summed_call_id: usize, output_idx: usize, num_inputs: usize },
}

impl SerializablePolyGateType {
    pub fn num_input(&self) -> usize {
        match self {
            SerializablePolyGateType::Input => 0,
            SerializablePolyGateType::SmallScalarMul { .. } |
            SerializablePolyGateType::LargeScalarMul { .. } |
            SerializablePolyGateType::SlotTransfer { .. } |
            SerializablePolyGateType::PubLut { .. } => 1,
            SerializablePolyGateType::SlotReduce { input_count, .. } => *input_count,
            SerializablePolyGateType::SubCircuitOutput { num_inputs, .. } |
            SerializablePolyGateType::SummedSubCircuitOutput { num_inputs, .. } => *num_inputs,
            SerializablePolyGateType::Add |
            SerializablePolyGateType::Sub |
            SerializablePolyGateType::Mul => 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SerializablePolyGate {
    pub gate_id: GateId,
    pub gate_type: SerializablePolyGateType,
    pub input_gates: Vec<GateId>,
}

impl SerializablePolyGate {
    pub fn new(
        gate_id: GateId,
        gate_type: SerializablePolyGateType,
        input_gates: Vec<GateId>,
    ) -> Self {
        Self { gate_id, gate_type, input_gates }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SerializableSubCircuitCall {
    pub sub_circuit_id: usize,
    pub shared_input_prefix: Option<Vec<BatchedWire>>,
    pub input_suffix: Vec<BatchedWire>,
    pub param_bindings: Vec<SubCircuitParamValue>,
    pub input_max_plaintext_norm_ranges:
        Option<Vec<crate::circuit::SubCircuitInputMaxPlaintextNormRange>>,
    pub scoped_call_id: usize,
    pub output_gate_ids: Vec<GateId>,
    pub num_outputs: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SerializableSummedSubCircuitCall {
    pub sub_circuit_id: usize,
    pub call_inputs: Vec<Vec<BatchedWire>>,
    pub param_bindings: Vec<Vec<SubCircuitParamValue>>,
    pub input_max_plaintext_norm_ranges:
        Option<Vec<crate::circuit::SubCircuitInputMaxPlaintextNormRange>>,
    pub scoped_call_ids: Vec<usize>,
    pub output_gate_ids: Vec<GateId>,
    pub num_outputs: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SerializablePolyCircuit {
    gates: BTreeMap<GateId, SerializablePolyGate>,
    sub_circuits: BTreeMap<usize, Box<SerializablePolyCircuit>>,
    sub_circuit_calls: BTreeMap<usize, SerializableSubCircuitCall>,
    summed_sub_circuit_calls: BTreeMap<usize, SerializableSummedSubCircuitCall>,
    sub_circuit_params: Vec<SubCircuitParamSpec>,
    sub_circuit_input_max_plaintext_norm_ranges:
        Option<Vec<crate::circuit::SubCircuitInputMaxPlaintextNormRange>>,
    output_ids: Vec<GateId>,
    num_input: usize,
    next_scoped_call_id: usize,
}

impl SerializablePolyCircuit {
    fn direct_serialized_sub_circuits<P: Poly>(
        circuit: &PolyCircuit<P>,
    ) -> Vec<(usize, Box<SerializablePolyCircuit>)> {
        circuit
            .direct_sub_circuit_ids()
            .into_iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|circuit_id| {
                (
                    circuit_id,
                    Box::new(Self::from_circuit(
                        circuit.registered_sub_circuit_ref(circuit_id).as_ref().clone(),
                    )),
                )
            })
            .collect::<Vec<_>>()
    }

    pub fn new(
        gates: BTreeMap<GateId, SerializablePolyGate>,
        sub_circuits: BTreeMap<usize, Box<SerializablePolyCircuit>>,
        sub_circuit_calls: BTreeMap<usize, SerializableSubCircuitCall>,
        summed_sub_circuit_calls: BTreeMap<usize, SerializableSummedSubCircuitCall>,
        sub_circuit_params: Vec<SubCircuitParamSpec>,
        sub_circuit_input_max_plaintext_norm_ranges: Option<
            Vec<crate::circuit::SubCircuitInputMaxPlaintextNormRange>,
        >,
        output_ids: Vec<GateId>,
        num_input: usize,
        next_scoped_call_id: usize,
    ) -> Self {
        Self {
            gates,
            sub_circuits,
            sub_circuit_calls,
            summed_sub_circuit_calls,
            sub_circuit_params,
            sub_circuit_input_max_plaintext_norm_ranges,
            output_ids,
            num_input,
            next_scoped_call_id,
        }
    }

    pub fn from_circuit<P: Poly>(circuit: PolyCircuit<P>) -> Self {
        let call_entries = circuit
            .sub_circuit_calls
            .iter()
            .map(|(call_id, call)| {
                let (shared_input_prefix, param_bindings) = rayon::join(
                    || {
                        call.shared_input_prefix_set_id
                            .map(|input_set_id| circuit.input_set(input_set_id).as_ref().to_vec())
                    },
                    || circuit.binding_set(call.binding_set_id).as_ref().to_vec(),
                );
                (
                    *call_id,
                    SerializableSubCircuitCall {
                        sub_circuit_id: call.sub_circuit_id,
                        shared_input_prefix,
                        input_suffix: call.input_suffix.clone(),
                        param_bindings,
                        input_max_plaintext_norm_ranges: call
                            .input_max_plaintext_norm_ranges
                            .as_ref()
                            .map(|ranges| ranges.as_ref().to_vec()),
                        scoped_call_id: call.scoped_call_id,
                        output_gate_ids: call.output_gate_ids.clone(),
                        num_outputs: call.num_outputs,
                    },
                )
            })
            .collect::<Vec<_>>();
        let summed_call_entries = circuit
            .summed_sub_circuit_calls
            .iter()
            .map(|(summed_call_id, call)| {
                let (call_inputs, param_bindings) = rayon::join(
                    || {
                        call.call_input_set_ids
                            .iter()
                            .map(|input_set_id| circuit.input_set(*input_set_id).as_ref().to_vec())
                            .collect::<Vec<_>>()
                    },
                    || {
                        call.call_binding_set_ids
                            .iter()
                            .map(|binding_set_id| {
                                circuit.binding_set(*binding_set_id).as_ref().to_vec()
                            })
                            .collect::<Vec<_>>()
                    },
                );
                (
                    *summed_call_id,
                    SerializableSummedSubCircuitCall {
                        sub_circuit_id: call.sub_circuit_id,
                        call_inputs,
                        param_bindings,
                        input_max_plaintext_norm_ranges: call
                            .input_max_plaintext_norm_ranges
                            .as_ref()
                            .map(|ranges| ranges.as_ref().to_vec()),
                        scoped_call_ids: call.scoped_call_ids.clone(),
                        output_gate_ids: call.output_gate_ids.clone(),
                        num_outputs: call.num_outputs,
                    },
                )
            })
            .collect::<Vec<_>>();
        let sub_circuits_vec = Self::direct_serialized_sub_circuits(&circuit);
        let gate_entries = circuit.gates.into_iter().collect::<Vec<_>>();

        let (gates_vec, ((sub_circuits_vec, calls_vec), summed_calls_vec)) = rayon::join(
            || {
                gate_entries
                    .into_par_iter()
                    .map(|(gate_id, gate)| {
                        let gate_type = match gate.gate_type {
                            PolyGateType::Input => SerializablePolyGateType::Input,
                            PolyGateType::SmallScalarMul { scalar } => {
                                SerializablePolyGateType::SmallScalarMul { scalar }
                            }
                            PolyGateType::LargeScalarMul { scalar } => {
                                SerializablePolyGateType::LargeScalarMul { scalar }
                            }
                            PolyGateType::SlotTransfer { src_slots } => {
                                SerializablePolyGateType::SlotTransfer { src_slots }
                            }
                            PolyGateType::SlotReduce { num_slots, input_count } => {
                                SerializablePolyGateType::SlotReduce { num_slots, input_count }
                            }
                            PolyGateType::Add => SerializablePolyGateType::Add,
                            PolyGateType::Sub => SerializablePolyGateType::Sub,
                            PolyGateType::Mul => SerializablePolyGateType::Mul,
                            PolyGateType::PubLut { lut_id } => {
                                let GateParamSource::Const(lut_id) = lut_id else {
                                    panic!("parameterized public lookup ids are not serializable");
                                };
                                SerializablePolyGateType::PubLut { lut_id }
                            }
                            PolyGateType::SubCircuitOutput { call_id, output_idx, num_inputs } => {
                                SerializablePolyGateType::SubCircuitOutput {
                                    call_id,
                                    output_idx,
                                    num_inputs,
                                }
                            }
                            PolyGateType::SummedSubCircuitOutput {
                                summed_call_id,
                                output_idx,
                                num_inputs,
                            } => SerializablePolyGateType::SummedSubCircuitOutput {
                                summed_call_id,
                                output_idx,
                                num_inputs,
                            },
                        };
                        (gate_id, SerializablePolyGate::new(gate_id, gate_type, gate.input_gates))
                    })
                    .collect::<Vec<_>>()
            },
            || {
                rayon::join(
                    || {
                        rayon::join(
                            || sub_circuits_vec,
                            || {
                                call_entries
                                    .into_par_iter()
                                    .map(|(call_id, call)| (call_id, call))
                                    .collect::<Vec<_>>()
                            },
                        )
                    },
                    || {
                        summed_call_entries
                            .into_par_iter()
                            .map(|(call_id, call)| (call_id, call))
                            .collect::<Vec<_>>()
                    },
                )
            },
        );

        Self::new(
            gates_vec.into_iter().collect(),
            sub_circuits_vec.into_iter().collect(),
            calls_vec.into_iter().collect(),
            summed_calls_vec.into_iter().collect(),
            circuit.sub_circuit_params,
            circuit
                .sub_circuit_input_max_plaintext_norm_ranges
                .as_ref()
                .map(|ranges| ranges.as_ref().to_vec()),
            circuit.output_ids,
            circuit.num_input,
            circuit.next_scoped_call_id,
        )
    }

    pub fn to_circuit<P: Poly>(self) -> PolyCircuit<P> {
        let lookup_registry = Arc::new(crate::circuit::poly_circuit::LookupRegistry::new());
        let binding_registry = Arc::new(crate::circuit::poly_circuit::BindingRegistry::new());
        let input_set_registry = Arc::new(crate::circuit::poly_circuit::InputSetRegistry::new());
        let sub_circuit_registry =
            Arc::new(crate::circuit::poly_circuit::SubCircuitRegistry::new());
        self.to_circuit_with_registries(
            lookup_registry,
            binding_registry,
            input_set_registry,
            sub_circuit_registry,
            true,
        )
    }

    fn to_circuit_with_registries<P: Poly>(
        self,
        lookup_registry: Arc<crate::circuit::poly_circuit::LookupRegistry<P>>,
        binding_registry: Arc<crate::circuit::poly_circuit::BindingRegistry>,
        input_set_registry: Arc<crate::circuit::poly_circuit::InputSetRegistry>,
        sub_circuit_registry: Arc<crate::circuit::poly_circuit::SubCircuitRegistry<P>>,
        allow_register_lookup: bool,
    ) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::new();
        circuit.lookup_registry = lookup_registry.clone();
        circuit.binding_registry = binding_registry.clone();
        circuit.input_set_registry = input_set_registry.clone();
        circuit.sub_circuit_registry = sub_circuit_registry.clone();
        circuit.allow_register_lookup = allow_register_lookup;

        let sub_circuit_entries = self.sub_circuits.into_iter().collect::<Vec<_>>();
        let gate_entries = self.gates.into_iter().collect::<Vec<_>>();
        let call_entries = self.sub_circuit_calls.into_iter().collect::<Vec<_>>();
        let summed_call_entries = self.summed_sub_circuit_calls.into_iter().collect::<Vec<_>>();

        let (sub_circuits_vec, (gates_vec, (calls_vec, summed_calls_vec))) = rayon::join(
            || {
                sub_circuit_entries
                    .into_par_iter()
                    .map(|(circuit_id, sub_circuit)| {
                        (
                            circuit_id,
                            Arc::new(sub_circuit.to_circuit_with_registries(
                                lookup_registry.clone(),
                                binding_registry.clone(),
                                input_set_registry.clone(),
                                sub_circuit_registry.clone(),
                                false,
                            )),
                        )
                    })
                    .collect::<Vec<_>>()
            },
            || {
                rayon::join(
                    || {
                        gate_entries
                            .into_par_iter()
                            .map(|(gate_id, sg)| {
                                let gate_type = match sg.gate_type {
                                    SerializablePolyGateType::Input => PolyGateType::Input,
                                    SerializablePolyGateType::SmallScalarMul { scalar } => {
                                        PolyGateType::SmallScalarMul { scalar }
                                    }
                                    SerializablePolyGateType::LargeScalarMul { scalar } => {
                                        PolyGateType::LargeScalarMul { scalar }
                                    }
                                    SerializablePolyGateType::SlotTransfer { src_slots } => {
                                        PolyGateType::SlotTransfer { src_slots }
                                    }
                                    SerializablePolyGateType::SlotReduce {
                                        num_slots,
                                        input_count,
                                    } => PolyGateType::SlotReduce { num_slots, input_count },
                                    SerializablePolyGateType::Add => PolyGateType::Add,
                                    SerializablePolyGateType::Sub => PolyGateType::Sub,
                                    SerializablePolyGateType::Mul => PolyGateType::Mul,
                                    SerializablePolyGateType::PubLut { lut_id } => {
                                        PolyGateType::PubLut {
                                            lut_id: GateParamSource::Const(lut_id),
                                        }
                                    }
                                    SerializablePolyGateType::SubCircuitOutput {
                                        call_id,
                                        output_idx,
                                        num_inputs,
                                    } => PolyGateType::SubCircuitOutput {
                                        call_id,
                                        output_idx,
                                        num_inputs,
                                    },
                                    SerializablePolyGateType::SummedSubCircuitOutput {
                                        summed_call_id,
                                        output_idx,
                                        num_inputs,
                                    } => PolyGateType::SummedSubCircuitOutput {
                                        summed_call_id,
                                        output_idx,
                                        num_inputs,
                                    },
                                };
                                (gate_id, PolyGate::new(gate_id, gate_type, sg.input_gates))
                            })
                            .collect::<Vec<_>>()
                    },
                    || {
                        rayon::join(
                            || {
                                call_entries
                                    .into_par_iter()
                                    .map(|(call_id, call)| (call_id, call))
                                    .collect::<Vec<_>>()
                            },
                            || {
                                summed_call_entries
                                    .into_par_iter()
                                    .map(|(call_id, call)| (call_id, call))
                                    .collect::<Vec<_>>()
                            },
                        )
                    },
                )
            },
        );

        circuit.gates = gates_vec.into_iter().collect();
        for (circuit_id, sub_circuit) in sub_circuits_vec {
            circuit.sub_circuit_registry.register_arc_with_id(circuit_id, sub_circuit);
        }
        circuit.sub_circuit_calls = calls_vec
            .into_iter()
            .map(|(call_id, call)| {
                let (shared_input_prefix_set_id, binding_set_id) = rayon::join(
                    || {
                        call.shared_input_prefix
                            .as_ref()
                            .map(|inputs| circuit.intern_input_set(inputs))
                    },
                    || circuit.binding_registry.register(&call.param_bindings),
                );
                (
                    call_id,
                    SubCircuitCall {
                        sub_circuit_id: call.sub_circuit_id,
                        shared_input_prefix_set_id,
                        input_suffix: call.input_suffix,
                        binding_set_id,
                        input_max_plaintext_norm_ranges: call
                            .input_max_plaintext_norm_ranges
                            .map(Arc::from),
                        scoped_call_id: call.scoped_call_id,
                        output_gate_ids: call.output_gate_ids,
                        num_outputs: call.num_outputs,
                    },
                )
            })
            .collect();
        circuit.summed_sub_circuit_calls = summed_calls_vec
            .into_iter()
            .map(|(summed_call_id, call)| {
                let call_input_set_ids = call
                    .call_inputs
                    .iter()
                    .map(|inputs| circuit.intern_input_set(inputs))
                    .collect::<Vec<_>>();
                let call_binding_set_ids = call
                    .param_bindings
                    .iter()
                    .map(|bindings| circuit.binding_registry.register(bindings))
                    .collect::<Vec<_>>();
                (
                    summed_call_id,
                    crate::circuit::SummedSubCircuitCall {
                        sub_circuit_id: call.sub_circuit_id,
                        call_input_set_ids,
                        call_binding_set_ids,
                        input_max_plaintext_norm_ranges: call
                            .input_max_plaintext_norm_ranges
                            .map(Arc::from),
                        scoped_call_ids: call.scoped_call_ids,
                        output_gate_ids: call.output_gate_ids,
                        num_outputs: call.num_outputs,
                    },
                )
            })
            .collect();
        circuit.sub_circuit_params = self.sub_circuit_params;
        circuit.sub_circuit_input_max_plaintext_norm_ranges =
            self.sub_circuit_input_max_plaintext_norm_ranges.map(Arc::from);
        circuit.num_input = self.num_input;
        circuit.output_ids = self.output_ids;
        circuit.next_scoped_call_id = self.next_scoped_call_id;
        circuit.recompute_gate_counts();
        circuit
    }

    pub fn from_json_str(json_str: &str) -> Self {
        serde_json::from_str(json_str).expect("Failed to deserialize SerializablePolyCircuit")
    }

    pub fn to_json_str(&self) -> String {
        serde_json::to_string(self).expect("Failed to serialize SerializablePolyCircuit")
    }

    pub fn from_json_file(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        let json_str = fs::read_to_string(path).unwrap_or_else(|err| {
            panic!("Failed to read SerializablePolyCircuit from {}: {err}", path.display())
        });
        Self::from_json_str(&json_str)
    }

    pub fn to_json_file(&self, path: impl AsRef<Path>) {
        let path = path.as_ref();
        fs::write(path, self.to_json_str()).unwrap_or_else(|err| {
            panic!("Failed to write SerializablePolyCircuit to {}: {err}", path.display())
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_primitives::poly::dcrt::poly::DCRTPoly;

    fn assert_json_roundtrip(circuit: PolyCircuit<DCRTPoly>) {
        let serialized = SerializablePolyCircuit::from_circuit(circuit.clone());
        let json = serialized.to_json_str();
        let roundtrip = SerializablePolyCircuit::from_json_str(&json).to_circuit::<DCRTPoly>();
        assert_eq!(roundtrip, circuit);
    }

    #[test]
    fn serialization_roundtrip_preserves_arithmetic_and_nonconsecutive_inputs() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(4).to_vec();
        let sum = circuit.add_gate(inputs[0], inputs[2]);
        let product = circuit.mul_gate(sum, inputs[3]);
        circuit.output([product, inputs[1].into()]);
        assert_json_roundtrip(circuit);
    }

    #[test]
    fn serialization_roundtrip_preserves_slot_operations() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(2).to_vec();
        let transferred =
            circuit.slot_transfer_gate(inputs[0], &[(1, None), (0, Some(3)), (1, None)]);
        let reduced = circuit.slot_reduce_gate(&[transferred, inputs[1].into()], 3);
        circuit.output([reduced]);
        assert_json_roundtrip(circuit);
    }

    #[test]
    fn serialization_roundtrip_preserves_parameterized_subcircuits() {
        let mut child = PolyCircuit::<DCRTPoly>::new();
        let scalar =
            child.register_sub_circuit_param(SubCircuitParamSpec::SmallScalarMul { max_scalar: 7 });
        let input = child.input(1).as_single_wire();
        let output = child.small_scalar_mul_param(input, scalar);
        child.output([output]);

        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input = circuit.input(1).as_single_wire();
        let child_id = circuit.register_sub_circuit(child);
        let outputs = circuit.call_sub_circuit_with_bindings(
            child_id,
            [input],
            &[SubCircuitParamValue::SmallScalarMul(vec![5])],
        );
        circuit.output(outputs);
        assert_json_roundtrip(circuit);
    }

    #[test]
    fn serialization_roundtrip_preserves_summed_subcircuits() {
        let mut child = PolyCircuit::<DCRTPoly>::new();
        let scalar =
            child.register_sub_circuit_param(SubCircuitParamSpec::SmallScalarMul { max_scalar: 3 });
        let input = child.input(1).as_single_wire();
        let output = child.small_scalar_mul_param(input, scalar);
        child.output([output]);

        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input = circuit.input(1).as_single_wire();
        let child_id = circuit.register_sub_circuit(child);
        let input_set = circuit.intern_input_set([input]);
        let two = circuit.intern_binding_set(&[SubCircuitParamValue::SmallScalarMul(vec![2])]);
        let three = circuit.intern_binding_set(&[SubCircuitParamValue::SmallScalarMul(vec![3])]);
        let outputs = circuit.call_sub_circuit_sum_many_with_binding_set_ids(
            child_id,
            vec![input_set, input_set],
            vec![two, three],
        );
        circuit.output(outputs);
        assert_json_roundtrip(circuit);
    }
}
