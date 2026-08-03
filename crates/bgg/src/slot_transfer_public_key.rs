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
mod tests {}
