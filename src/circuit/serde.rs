use crate::{
    circuit::{
        GateParamSource, PolyCircuit, PolyGate, PolyGateType, StoredSubCircuit, SubCircuitCall,
        SubCircuitParamKind, SubCircuitParamValue,
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
    Add,
    Sub,
    Mul,
    PubLut { lut_id: GateParamSource<usize> },
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
    pub inputs: Vec<GateId>,
    pub param_bindings: Vec<SubCircuitParamValue>,
    pub scoped_call_id: usize,
    pub output_gate_ids: Vec<GateId>,
    pub num_outputs: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SerializableSummedSubCircuitCall {
    pub sub_circuit_id: usize,
    pub call_inputs: Vec<Vec<GateId>>,
    pub param_bindings: Vec<Vec<SubCircuitParamValue>>,
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
    sub_circuit_params: Vec<SubCircuitParamKind>,
    output_ids: Vec<GateId>,
    num_input: usize,
    next_scoped_call_id: usize,
}

impl SerializablePolyCircuit {
    pub fn new(
        gates: BTreeMap<GateId, SerializablePolyGate>,
        sub_circuits: BTreeMap<usize, Box<SerializablePolyCircuit>>,
        sub_circuit_calls: BTreeMap<usize, SerializableSubCircuitCall>,
        summed_sub_circuit_calls: BTreeMap<usize, SerializableSummedSubCircuitCall>,
        sub_circuit_params: Vec<SubCircuitParamKind>,
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
                let param_bindings = circuit.binding_set(call.binding_set_id).as_ref().to_vec();
                (
                    *call_id,
                    SerializableSubCircuitCall {
                        sub_circuit_id: call.sub_circuit_id,
                        inputs: call.inputs.clone(),
                        param_bindings,
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
                        scoped_call_ids: call.scoped_call_ids.clone(),
                        output_gate_ids: call.output_gate_ids.clone(),
                        num_outputs: call.num_outputs,
                    },
                )
            })
            .collect::<Vec<_>>();
        let gate_entries = circuit.gates.into_iter().collect::<Vec<_>>();
        let sub_circuit_entries = circuit.sub_circuits.into_iter().collect::<Vec<_>>();

        let (gates_vec, (sub_circuits_vec, (calls_vec, summed_calls_vec))) = rayon::join(
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
                            PolyGateType::Add => SerializablePolyGateType::Add,
                            PolyGateType::Sub => SerializablePolyGateType::Sub,
                            PolyGateType::Mul => SerializablePolyGateType::Mul,
                            PolyGateType::PubLut { lut_id } => {
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
                        sub_circuit_entries
                            .into_par_iter()
                            .map(|(circuit_id, sub_circuit)| {
                                let StoredSubCircuit::InMemory(sub_circuit) = sub_circuit;
                                (
                                    circuit_id,
                                    Box::new(Self::from_circuit(sub_circuit.as_ref().clone())),
                                )
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

        Self::new(
            gates_vec.into_iter().collect(),
            sub_circuits_vec.into_iter().collect(),
            calls_vec.into_iter().collect(),
            summed_calls_vec.into_iter().collect(),
            circuit.sub_circuit_params,
            circuit.output_ids,
            circuit.num_input,
            circuit.next_scoped_call_id,
        )
    }

    pub fn to_circuit<P: Poly>(self) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::new();

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
                            StoredSubCircuit::InMemory(Arc::new(sub_circuit.to_circuit::<P>())),
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
                                    SerializablePolyGateType::Add => PolyGateType::Add,
                                    SerializablePolyGateType::Sub => PolyGateType::Sub,
                                    SerializablePolyGateType::Mul => PolyGateType::Mul,
                                    SerializablePolyGateType::PubLut { lut_id } => {
                                        PolyGateType::PubLut { lut_id }
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

        circuit.sub_circuits = sub_circuits_vec.into_iter().collect();
        circuit.gates = gates_vec.into_iter().collect();
        let lookup_registry = circuit.lookup_registry.clone();
        let binding_registry = circuit.binding_registry.clone();
        let input_set_registry = circuit.input_set_registry.clone();
        for sub_circuit in circuit.sub_circuits.values_mut() {
            let StoredSubCircuit::InMemory(sub_circuit) = sub_circuit;
            Arc::make_mut(sub_circuit).inherit_registries(
                lookup_registry.clone(),
                binding_registry.clone(),
                input_set_registry.clone(),
            );
        }
        circuit.sub_circuit_calls = calls_vec
            .into_iter()
            .map(|(call_id, call)| {
                let binding_set_id = circuit.binding_registry.register(&call.param_bindings);
                (
                    call_id,
                    SubCircuitCall {
                        sub_circuit_id: call.sub_circuit_id,
                        inputs: call.inputs,
                        binding_set_id,
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
                        scoped_call_ids: call.scoped_call_ids,
                        output_gate_ids: call.output_gate_ids,
                        num_outputs: call.num_outputs,
                    },
                )
            })
            .collect();
        circuit.sub_circuit_params = self.sub_circuit_params;
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
    use crate::{
        lookup::poly::PolyPltEvaluator,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };

    use super::*;

    #[test]
    fn test_serialization_roundtrip() {
        let mut original_circuit: PolyCircuit<DCRTPoly> = PolyCircuit::new();
        let inputs = original_circuit.input(3);
        let add_gate = original_circuit.add_gate(inputs[0], inputs[1]);
        let sub_gate = original_circuit.sub_gate(add_gate, inputs[2]);
        let mul_gate = original_circuit.mul_gate(inputs[1], inputs[2]);

        let mut sub_circuit: PolyCircuit<_> = PolyCircuit::new();
        let sub_inputs = sub_circuit.input(2);
        let sub_add_gate = sub_circuit.add_gate(sub_inputs[0], sub_inputs[1]);
        let sub_mul_gate = sub_circuit.mul_gate(sub_inputs[0], sub_inputs[1]);
        sub_circuit.output(vec![sub_add_gate, sub_mul_gate]);

        let sub_circuit_id = original_circuit.register_sub_circuit(sub_circuit);
        let sub_outputs =
            original_circuit.call_sub_circuit(sub_circuit_id, &[inputs[0], inputs[1]]);
        let combined_gate = original_circuit.add_gate(sub_gate, sub_outputs[0]);
        original_circuit.output(vec![combined_gate, mul_gate, sub_outputs[1]]);

        let serializable_circuit = SerializablePolyCircuit::from_circuit(original_circuit.clone());
        let roundtrip_circuit = serializable_circuit.to_circuit();
        assert_eq!(roundtrip_circuit, original_circuit);
    }

    #[test]
    fn test_serialization_roundtrip_json() {
        let mut original_circuit: PolyCircuit<DCRTPoly> = PolyCircuit::new();
        let inputs = original_circuit.input(3);
        let add_gate = original_circuit.add_gate(inputs[0], inputs[1]);
        let mul_gate = original_circuit.mul_gate(inputs[1], inputs[2]);

        let mut sub_circuit = PolyCircuit::new();
        let sub_inputs = sub_circuit.input(2);
        let sub_add_gate = sub_circuit.add_gate(sub_inputs[0], sub_inputs[1]);
        let sub_mul_gate = sub_circuit.mul_gate(sub_inputs[0], sub_inputs[1]);
        sub_circuit.output(vec![sub_add_gate, sub_mul_gate]);

        let sub_circuit_id = original_circuit.register_sub_circuit(sub_circuit);
        let sub_outputs =
            original_circuit.call_sub_circuit(sub_circuit_id, &[inputs[0], inputs[1]]);
        let combined_gate = original_circuit.add_gate(add_gate, sub_outputs[0]);
        original_circuit.output(vec![combined_gate, mul_gate, sub_outputs[1]]);

        let serializable_circuit = SerializablePolyCircuit::from_circuit(original_circuit.clone());
        let serializable_circuit_json = serializable_circuit.to_json_str();
        let serializable_circuit =
            SerializablePolyCircuit::from_json_str(&serializable_circuit_json);
        let roundtrip_circuit = serializable_circuit.to_circuit();
        assert_eq!(roundtrip_circuit, original_circuit);
    }

    #[test]
    fn test_serialization_roundtrip_with_nonconsecutive_inputs() {
        let mut circuit: PolyCircuit<DCRTPoly> = PolyCircuit::new();
        let first_inputs = circuit.input(1);
        let _gap_gate = circuit.const_digits(&[1u32, 0u32, 1u32]);
        let second_inputs = circuit.input(1);
        assert_ne!(second_inputs[0].0, first_inputs[0].0 + 1);

        let add = circuit.add_gate(first_inputs[0], second_inputs[0]);
        circuit.output(vec![add]);

        let serializable = SerializablePolyCircuit::from_circuit(circuit.clone());
        let roundtrip = serializable.to_circuit::<DCRTPoly>();

        let params = DCRTPolyParams::default();
        let a = DCRTPoly::const_one(&params);
        let b = DCRTPoly::const_one(&params);
        let one = DCRTPoly::const_one(&params);
        let out1_inputs = vec![a.clone(), b.clone()];
        let out1 = circuit.eval(&params, one, out1_inputs, None::<&PolyPltEvaluator>, None, None);
        let one = DCRTPoly::const_one(&params);
        let out2_inputs = vec![a, b];
        let out2 = roundtrip.eval(&params, one, out2_inputs, None::<&PolyPltEvaluator>, None, None);
        assert_eq!(out1, out2);
    }

    #[test]
    fn test_serialization_roundtrip_with_slot_transfer_gate() {
        let mut circuit: PolyCircuit<DCRTPoly> = PolyCircuit::new();
        let inputs = circuit.input(1);
        let transferred = circuit.slot_transfer_gate(inputs[0], &[(1, None), (0, None), (1, None)]);
        circuit.output(vec![transferred]);

        let serializable = SerializablePolyCircuit::from_circuit(circuit.clone());
        let roundtrip = serializable.to_circuit::<DCRTPoly>();

        assert_eq!(roundtrip, circuit);
    }

    #[test]
    fn test_serialization_roundtrip_with_parameterized_sub_circuit() {
        let mut sub_circuit: PolyCircuit<DCRTPoly> = PolyCircuit::new();
        let scalar_param =
            sub_circuit.register_sub_circuit_param(SubCircuitParamKind::SmallScalarMul);
        let sub_inputs = sub_circuit.input(1);
        let scaled = sub_circuit.small_scalar_mul_param(sub_inputs[0], scalar_param);
        sub_circuit.output(vec![scaled]);

        let mut circuit: PolyCircuit<DCRTPoly> = PolyCircuit::new();
        let inputs = circuit.input(1);
        let sub_id = circuit.register_sub_circuit(sub_circuit);
        let outputs = circuit.call_sub_circuit_with_bindings(
            sub_id,
            &[inputs[0]],
            &[SubCircuitParamValue::SmallScalarMul(vec![5])],
        );
        circuit.output(outputs);

        let serializable = SerializablePolyCircuit::from_circuit(circuit.clone());
        let roundtrip = serializable.to_circuit::<DCRTPoly>();
        assert_eq!(roundtrip, circuit);
    }

    #[test]
    fn test_serialization_roundtrip_with_summed_sub_circuit() {
        let mut sub_circuit: PolyCircuit<DCRTPoly> = PolyCircuit::new();
        let scalar_param =
            sub_circuit.register_sub_circuit_param(SubCircuitParamKind::SmallScalarMul);
        let sub_inputs = sub_circuit.input(1);
        let scaled = sub_circuit.small_scalar_mul_param(sub_inputs[0], scalar_param);
        sub_circuit.output(vec![scaled]);

        let mut circuit: PolyCircuit<DCRTPoly> = PolyCircuit::new();
        let inputs = circuit.input(1);
        let sub_id = circuit.register_sub_circuit(sub_circuit);
        let binding_two =
            circuit.intern_binding_set(&[SubCircuitParamValue::SmallScalarMul(vec![2])]);
        let binding_three =
            circuit.intern_binding_set(&[SubCircuitParamValue::SmallScalarMul(vec![3])]);
        let input_set_id = circuit.intern_input_set(&[inputs[0]]);
        let outputs = circuit.call_sub_circuit_sum_many_with_binding_set_ids(
            sub_id,
            vec![input_set_id, input_set_id],
            vec![binding_two, binding_three],
        );
        circuit.output(outputs);

        let serializable = SerializablePolyCircuit::from_circuit(circuit.clone());
        let roundtrip = serializable.to_circuit::<DCRTPoly>();
        assert_eq!(roundtrip, circuit);
    }
}
