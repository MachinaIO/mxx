use crate::{
    circuit::{
        PolyCircuit, PolyGate,
        gate::{GateId, PolyGateType},
    },
    poly::Poly,
};
use num_bigint::BigUint;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SerializablePolyGateType {
    Input,
    SmallScalarMul { scalar: Vec<u32> },
    LargeScalarMul { scalar: Vec<BigUint> },
    Add,
    Sub,
    Mul,
    Rotate { shift: i32 },
    PubLut { lut_id: usize },
}

impl SerializablePolyGateType {
    pub fn num_input(&self) -> usize {
        match self {
            SerializablePolyGateType::Input => 0,
            SerializablePolyGateType::SmallScalarMul { .. } |
            SerializablePolyGateType::LargeScalarMul { .. } |
            SerializablePolyGateType::Rotate { .. } |
            SerializablePolyGateType::PubLut { .. } => 1,
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
pub struct SerializablePolyCircuit {
    gates: BTreeMap<GateId, SerializablePolyGate>,
    sub_circuits: BTreeMap<usize, SerializablePolyCircuit>,
    output_ids: Vec<GateId>,
    num_input: usize,
}

impl SerializablePolyCircuit {
    pub fn new(
        gates: BTreeMap<GateId, SerializablePolyGate>,
        sub_circuits: BTreeMap<usize, SerializablePolyCircuit>,
        output_ids: Vec<GateId>,
        num_input: usize,
    ) -> Self {
        Self { gates, sub_circuits, output_ids, num_input }
    }

    pub fn from_circuit<P: Poly>(circuit: PolyCircuit<P>) -> Self {
        let mut gates = BTreeMap::new();
        for (gate_id, gate) in circuit.gates.into_iter() {
            let gate_type = match gate.gate_type {
                PolyGateType::Input => SerializablePolyGateType::Input,
                PolyGateType::SmallScalarMul { scalar } => {
                    SerializablePolyGateType::SmallScalarMul { scalar }
                }
                PolyGateType::LargeScalarMul { scalar } => {
                    SerializablePolyGateType::LargeScalarMul { scalar }
                }
                PolyGateType::Add => SerializablePolyGateType::Add,
                PolyGateType::Sub => SerializablePolyGateType::Sub,
                PolyGateType::Mul => SerializablePolyGateType::Mul,
                PolyGateType::Rotate { shift } => SerializablePolyGateType::Rotate { shift },
                PolyGateType::PubLut { lut_id } => SerializablePolyGateType::PubLut { lut_id },
            };
            let serializable_gate = SerializablePolyGate::new(gate_id, gate_type, gate.input_gates);
            gates.insert(gate_id, serializable_gate);
        }

        let mut sub_circuits = BTreeMap::new();
        for (circuit_id, sub_circuit) in circuit.sub_circuits.into_iter() {
            let serializable_sub_circuit = Self::from_circuit(sub_circuit);
            sub_circuits.insert(circuit_id, serializable_sub_circuit);
        }
        Self::new(gates, sub_circuits, circuit.output_ids, circuit.num_input)
    }

    pub fn to_circuit<P: Poly>(self) -> PolyCircuit<P> {
        let mut circuit = PolyCircuit::new();
        // Restore sub-circuits first
        for (_, serializable_sub_circuit) in self.sub_circuits.into_iter() {
            let sub_circuit = serializable_sub_circuit.to_circuit();
            circuit.register_sub_circuit(sub_circuit);
        }
        // Insert gates preserving original GateIds
        let getes_len = self.gates.len();
        for idx in 0..getes_len {
            let sg = self.gates.get(&GateId(idx)).expect("serialized gate id missing").to_owned();
            let gate_type = match sg.gate_type {
                SerializablePolyGateType::Input => PolyGateType::Input,
                SerializablePolyGateType::SmallScalarMul { scalar } => {
                    PolyGateType::SmallScalarMul { scalar }
                }
                SerializablePolyGateType::LargeScalarMul { scalar } => {
                    PolyGateType::LargeScalarMul { scalar }
                }
                SerializablePolyGateType::Add => PolyGateType::Add,
                SerializablePolyGateType::Sub => PolyGateType::Sub,
                SerializablePolyGateType::Mul => PolyGateType::Mul,
                SerializablePolyGateType::Rotate { shift } => PolyGateType::Rotate { shift },
                SerializablePolyGateType::PubLut { lut_id } => PolyGateType::PubLut { lut_id },
            };
            circuit
                .gates
                .insert(GateId(idx), PolyGate::new(GateId(idx), gate_type, sg.input_gates));
        }
        circuit.num_input = self.num_input;
        circuit.output_ids = self.output_ids;
        circuit.recompute_gate_counts();
        circuit
    }

    pub fn from_json_str(json_str: &str) -> Self {
        serde_json::from_str(json_str).expect("Failed to deserialize SerializablePolyCircuit")
    }

    pub fn to_json_str(&self) -> String {
        serde_json::to_string(self).expect("Failed to serialize SerializablePolyCircuit")
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
        // Create a complex circuit with various operations
        let mut original_circuit: PolyCircuit<DCRTPoly> = PolyCircuit::new();

        // Add inputs
        let inputs = original_circuit.input(3);

        // Add various gates
        let add_gate = original_circuit.add_gate(inputs[0], inputs[1]);
        let sub_gate = original_circuit.sub_gate(add_gate, inputs[2]);
        let mul_gate = original_circuit.mul_gate(inputs[1], inputs[2]);

        // Create a sub-circuit
        let mut sub_circuit: PolyCircuit<_> = PolyCircuit::new();
        let sub_inputs = sub_circuit.input(2);
        let sub_add_gate = sub_circuit.add_gate(sub_inputs[0], sub_inputs[1]);
        let sub_mul_gate = sub_circuit.mul_gate(sub_inputs[0], sub_inputs[1]);
        sub_circuit.output(vec![sub_add_gate, sub_mul_gate]);

        // Register the sub-circuit
        let sub_circuit_id = original_circuit.register_sub_circuit(sub_circuit);

        // Call the sub-circuit
        let sub_outputs =
            original_circuit.call_sub_circuit(sub_circuit_id, &[inputs[0], inputs[1]]);

        // Use the sub-circuit outputs
        let combined_gate = original_circuit.add_gate(sub_gate, sub_outputs[0]);

        // Set the output
        original_circuit.output(vec![combined_gate, mul_gate, sub_outputs[1]]);

        // Convert to SerializablePolyCircuit
        let serializable_circuit = SerializablePolyCircuit::from_circuit(original_circuit.clone());

        // Convert back to PolyCircuit
        let roundtrip_circuit = serializable_circuit.to_circuit();

        // Verify that the circuits are identical by directly comparing them
        // This works because PolyCircuit implements the Eq trait
        assert_eq!(roundtrip_circuit, original_circuit);
    }

    #[test]
    fn test_serialization_roundtrip_json() {
        // Create a complex circuit with various operations
        let mut original_circuit: PolyCircuit<DCRTPoly> = PolyCircuit::new();

        // Add inputs
        let inputs = original_circuit.input(3);

        // Add various gates
        let add_gate = original_circuit.add_gate(inputs[0], inputs[1]);
        let mul_gate = original_circuit.mul_gate(inputs[1], inputs[2]);

        // Create a sub-circuit
        let mut sub_circuit = PolyCircuit::new();
        let sub_inputs = sub_circuit.input(2);
        let sub_add_gate = sub_circuit.add_gate(sub_inputs[0], sub_inputs[1]);
        let sub_mul_gate = sub_circuit.mul_gate(sub_inputs[0], sub_inputs[1]);
        sub_circuit.output(vec![sub_add_gate, sub_mul_gate]);

        // Register the sub-circuit
        let sub_circuit_id = original_circuit.register_sub_circuit(sub_circuit);

        // Call the sub-circuit
        let sub_outputs =
            original_circuit.call_sub_circuit(sub_circuit_id, &[inputs[0], inputs[1]]);

        // Use the sub-circuit outputs
        let combined_gate = original_circuit.add_gate(add_gate, sub_outputs[0]);

        // Set the output
        original_circuit.output(vec![combined_gate, mul_gate, sub_outputs[1]]);

        // Convert to SerializablePolyCircuit
        let serializable_circuit = SerializablePolyCircuit::from_circuit(original_circuit.clone());
        let serializable_circuit_json = serializable_circuit.to_json_str();
        println!("{}", serializable_circuit_json);
        // Convert back to PolyCircuit
        let serializable_circuit =
            SerializablePolyCircuit::from_json_str(&serializable_circuit_json);
        let roundtrip_circuit = serializable_circuit.to_circuit();

        // Verify that the circuits are identical by directly comparing them
        // This works because PolyCircuit implements the Eq trait
        assert_eq!(roundtrip_circuit, original_circuit);
    }

    #[test]
    fn test_serialization_roundtrip_with_nonconsecutive_inputs() {
        // Create a circuit where input() is called twice with a gate in between,
        // resulting in non-consecutive input GateIds.
        let mut circuit: PolyCircuit<DCRTPoly> = PolyCircuit::new();
        let first_inputs = circuit.input(1);
        let _gap_gate = circuit.const_digits(&[1u32, 0u32, 1u32]);
        let second_inputs = circuit.input(1);
        // Ensure the input GateIds are not consecutive
        assert_ne!(second_inputs[0].0, first_inputs[0].0 + 1);

        let add = circuit.add_gate(first_inputs[0], second_inputs[0]);
        circuit.output(vec![add]);

        let serializable = SerializablePolyCircuit::from_circuit(circuit.clone());
        let roundtrip = serializable.to_circuit::<DCRTPoly>();

        // Verify behavioral equivalence by evaluating both circuits
        let params = DCRTPolyParams::default();
        let a = DCRTPoly::const_one(&params);
        let b = DCRTPoly::const_one(&params);
        let out1 = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a.clone(), b.clone()],
            None::<&PolyPltEvaluator>,
        );
        let out2 = roundtrip.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b],
            None::<&PolyPltEvaluator>,
        );
        assert_eq!(out1, out2);
    }
}
