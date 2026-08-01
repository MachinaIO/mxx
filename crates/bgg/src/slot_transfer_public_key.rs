//! Public-key pass for cryptographic slot-transfer circuit lowering.

use crate::{BggPublicKeyCompiler, BggPublicKeyWire, CircuitCompileError};
use mxx_dsl::{Bytes, HashTag, Mat};
use mxx_gadgets::{
    Poly,
    circuit::{CircuitLoweringTypes, GateInstance, SlotOperationLowering},
};
use rayon::prelude::*;

#[derive(Clone)]
pub enum BggSlotTransferGateRequest {
    Transfer {
        identity: String,
        input_public_key: Mat,
        output_public_key: Mat,
        source_slots: Vec<(u32, Option<u32>)>,
    },
    Reduce {
        identity: String,
        input_public_keys: Vec<Mat>,
        output_public_key: Mat,
        source_slot_count: usize,
    },
}

#[derive(Clone)]
pub struct BggSlotTransferPublicKeyLowering {
    pub compiler: BggPublicKeyCompiler,
    pub hash_key: Bytes,
    pub public_key_type: mxx_ir_core::types::MatrixType,
    pub configured_slot_count: usize,
    pub requests: Vec<BggSlotTransferGateRequest>,
}

impl BggSlotTransferPublicKeyLowering {
    fn output_public_key(&self, gate: GateInstance<'_>, reduction: bool) -> BggPublicKeyWire {
        let operation = if reduction { "slot_reduce" } else { "slot_transfer" };
        BggPublicKeyWire {
            matrix: self.compiler.ring.hash_matrix(
                self.hash_key.clone(),
                HashTag::from(format!("{operation}_gate_a_out_{}", gate_token(gate)).into_bytes()),
                (self.public_key_type.rows.clone(), self.public_key_type.columns.clone()),
            ),
            reveal_plaintext: true,
        }
    }

    fn valid_type(&self, input: &BggPublicKeyWire) -> bool {
        input.matrix.matrix_type() == &self.public_key_type
    }
}

impl CircuitLoweringTypes for BggSlotTransferPublicKeyLowering {
    type Wire = BggPublicKeyWire;
    type Error = CircuitCompileError;
}

impl<P: Poly> SlotOperationLowering<P> for BggSlotTransferPublicKeyLowering {
    fn slot_transfer(
        &mut self,
        input: &Self::Wire,
        source_slots: &[(u32, Option<u32>)],
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        if !self.valid_type(input) ||
            source_slots.len() > self.configured_slot_count ||
            source_slots
                .par_iter()
                .any(|(source, _)| *source as usize >= self.configured_slot_count)
        {
            return Err(CircuitCompileError::InvalidSlotTransfer {
                gate: gate.local_gate().index(),
            });
        }
        let output = self.output_public_key(gate, false);
        self.requests.push(BggSlotTransferGateRequest::Transfer {
            identity: gate_token(gate),
            input_public_key: input.matrix.clone(),
            output_public_key: output.matrix.clone(),
            source_slots: source_slots.to_vec(),
        });
        Ok(output)
    }

    fn slot_reduce(
        &mut self,
        inputs: &[Self::Wire],
        slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        if inputs.is_empty() ||
            inputs.len() > slot_count ||
            slot_count > self.configured_slot_count ||
            inputs.par_iter().any(|input| !self.valid_type(input))
        {
            return Err(CircuitCompileError::InvalidSlotTransfer {
                gate: gate.local_gate().index(),
            });
        }
        let output = self.output_public_key(gate, true);
        self.requests.push(BggSlotTransferGateRequest::Reduce {
            identity: gate_token(gate),
            input_public_keys: inputs.par_iter().map(|input| input.matrix.clone()).collect(),
            output_public_key: output.matrix.clone(),
            source_slot_count: slot_count,
        });
        Ok(output)
    }
}

pub(crate) fn gate_token(gate: GateInstance<'_>) -> String {
    let mut token = gate.call_path().iter().map(usize::to_string).collect::<Vec<_>>().join("_");
    if !token.is_empty() {
        token.push('_');
    }
    token.push_str(&format!("g{}_o{}", gate.local_gate().index(), gate.operation_occurrence()));
    token
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NoPublicLookup, PolyCircuitCompiler};
    use mxx_dsl::{DslContext, Ring};
    use mxx_gadgets::circuit::PolyCircuit;
    use mxx_ir_core::ParamEnv;
    use mxx_primitives::poly::dcrt::poly::DCRTPoly;

    #[test]
    fn public_key_slot_transfer_lowering_builds_and_symbolically_elaborates() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input_gate = circuit.input(1).as_single_wire();
        let transferred = circuit.slot_transfer_gate(input_gate, &[(1, None), (0, Some(3))]);
        circuit.output([transferred]);

        let ring = Ring::new(257, 8);
        let public_key_type = ring.matrix_type((2, 8));
        let key = |name: &str| BggPublicKeyWire {
            matrix: ring.input(name, (2, 8)),
            reveal_plaintext: true,
        };
        let public_key_compiler =
            BggPublicKeyCompiler { ring: ring.clone(), base: 4.into(), digit_count: 4.into() };
        let mut lowering = BggSlotTransferPublicKeyLowering {
            compiler: public_key_compiler.clone(),
            hash_key: ring.bytes_input("hash-key", 32),
            public_key_type,
            configured_slot_count: 2,
            requests: Vec::new(),
        };
        let mut lookup = NoPublicLookup::default();
        let outputs = PolyCircuitCompiler { public_key: public_key_compiler }
            .compile_public_keys_with_lowerings(
                &circuit,
                key("one"),
                [key("input")],
                &mut lookup,
                &mut lowering,
            )
            .expect("slot-transfer lowering");
        assert_eq!(lowering.requests.len(), 1);
        let built = DslContext::new("slot-transfer-public-key")
            .output("output", outputs[0].matrix.clone())
            .expect("output")
            .build()
            .expect("build");
        let elaborated = built.elaborate(&ParamEnv::default()).expect("symbolic elaboration");
        assert!(elaborated.wire(&elaborated.outputs["output"]).unwrap().expression.is_some());
    }
}
