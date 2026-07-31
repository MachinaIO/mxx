use crate::{AdvancedGateLowering, BggPublicKeyWire, CircuitCompileError};
use mxx_gadgets::{
    Poly,
    circuit::{GateInstance, PolyCircuit},
};
use mxx_ir_core::{GraphBuilder, MatrixWire, WireRef, node::HashVariant};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BggSlotTransferGateRequest {
    Transfer {
        identity: String,
        input_public_key: MatrixWire,
        output_public_key: MatrixWire,
        source_slots: Vec<(u32, Option<u32>)>,
    },
    Reduce {
        identity: String,
        input_public_keys: Vec<MatrixWire>,
        output_public_key: MatrixWire,
        source_slot_count: usize,
    },
}

/// Public-key side of the historical BGG slot-transfer construction.
///
/// The public-key evaluator only derives the destination key and records no
/// online matrix arithmetic. The preimages consumed by polynomial encodings
/// are produced by the separate slot-transfer artifact compiler.
#[derive(Clone, Debug)]
pub struct BggSlotTransferPublicKeyLowering {
    pub hash_key: WireRef,
    pub public_key_type: mxx_ir_core::types::MatrixType,
    pub configured_slot_count: usize,
    pub requests: Vec<BggSlotTransferGateRequest>,
}

impl BggSlotTransferPublicKeyLowering {
    fn output_public_key(
        &self,
        builder: &mut GraphBuilder,
        gate: GateInstance<'_>,
        reduction: bool,
    ) -> BggPublicKeyWire {
        let operation = if reduction { "slot_reduce" } else { "slot_transfer" };
        let tag = format!("{operation}_gate_a_out_{}", gate_token(gate));
        BggPublicKeyWire {
            matrix: builder.hash_sample(
                self.hash_key,
                self.public_key_type.clone(),
                HashVariant::Plain,
                tag.into_bytes(),
                Vec::new(),
                None,
                None,
            ),
            reveal_plaintext: true,
        }
    }
}

impl<P: Poly> AdvancedGateLowering<P, BggPublicKeyWire> for BggSlotTransferPublicKeyLowering {
    fn slot_transfer(
        &mut self,
        builder: &mut GraphBuilder,
        input: &BggPublicKeyWire,
        source_slots: &[(u32, Option<u32>)],
        gate: GateInstance<'_>,
    ) -> Result<BggPublicKeyWire, CircuitCompileError> {
        if input.matrix.matrix_type != self.public_key_type ||
            source_slots.len() > self.configured_slot_count ||
            source_slots.iter().any(|(source, _)| {
                usize::try_from(*source).map_or(true, |source| source >= self.configured_slot_count)
            })
        {
            return Err(CircuitCompileError::InvalidSlotTransfer {
                gate: gate.local_gate().index(),
            });
        }
        let output = self.output_public_key(builder, gate, false);
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
        builder: &mut GraphBuilder,
        inputs: &[BggPublicKeyWire],
        slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<BggPublicKeyWire, CircuitCompileError> {
        if inputs.is_empty() ||
            slot_count == 0 ||
            inputs.len() > slot_count ||
            slot_count > self.configured_slot_count ||
            inputs.iter().any(|input| input.matrix.matrix_type != self.public_key_type)
        {
            return Err(CircuitCompileError::InvalidSlotTransfer {
                gate: gate.local_gate().index(),
            });
        }
        let output = self.output_public_key(builder, gate, true);
        self.requests.push(BggSlotTransferGateRequest::Reduce {
            identity: gate_token(gate),
            input_public_keys: inputs.iter().map(|input| input.matrix.clone()).collect(),
            output_public_key: output.matrix.clone(),
            source_slot_count: slot_count,
        });
        Ok(output)
    }

    fn public_lookup(
        &mut self,
        _builder: &mut GraphBuilder,
        _circuit: &PolyCircuit<P>,
        _lookup_id: usize,
        _input: &BggPublicKeyWire,
        gate: GateInstance<'_>,
    ) -> Result<BggPublicKeyWire, CircuitCompileError> {
        Err(CircuitCompileError::MissingGateContext {
            gate: gate.local_gate().index(),
            kind: "public lookup",
        })
    }
}

pub(crate) fn gate_token(gate: GateInstance<'_>) -> String {
    if gate.call_path().is_empty() && gate.operation_occurrence() == 0 {
        return gate.local_gate().index().to_string();
    }
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
    use crate::{BggPublicKeyCompiler, PolyCircuitCompiler};
    use mxx_gadgets::circuit::PolyCircuit;
    use mxx_ir_core::{IntExpr, node::NodeKind, types::MatrixType};
    use mxx_primitives::poly::dcrt::poly::DCRTPoly;

    fn matrix_type(rows: usize, columns: usize) -> MatrixType {
        MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(8),
            rows: IntExpr::constant(rows),
            columns: IntExpr::constant(columns),
        }
    }

    #[test]
    fn public_key_lowering_preserves_legacy_tags_and_reveal_metadata() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input = circuit.input(1).as_single_wire();
        let transferred = circuit.slot_transfer_gate(input, &[(2, None), (0, Some(3))]);
        let transfer_gate = transferred.as_single_wire();
        let reduced = circuit.slot_reduce_gate(&[transferred], 3);
        let reduce_gate = reduced.as_single_wire();
        circuit.output([reduced]);

        let mut builder = GraphBuilder::new("slot-transfer-public-key", Vec::new());
        let hash_key = builder.bytes_input("hash_key", 32);
        let input = BggPublicKeyWire {
            matrix: builder.input("input", matrix_type(2, 10)),
            reveal_plaintext: false,
        };
        let compiler = PolyCircuitCompiler {
            public_key: BggPublicKeyCompiler {
                base: IntExpr::constant(2),
                decomposed_type: matrix_type(10, 10),
            },
        };
        let mut lowering = BggSlotTransferPublicKeyLowering {
            hash_key,
            public_key_type: matrix_type(2, 10),
            configured_slot_count: 3,
            requests: Vec::new(),
        };
        let outputs = compiler
            .compile_public_keys_with_lowering(
                &mut builder,
                &circuit,
                input.clone(),
                [input],
                &mut lowering,
            )
            .expect("slot-transfer public-key graph");
        assert!(outputs[0].reveal_plaintext);
        assert_eq!(lowering.requests.len(), 2);
        let tags = builder
            .finish()
            .nodes
            .into_iter()
            .filter_map(|node| match node.kind {
                NodeKind::HashSample { tag_prefix, .. } => Some(tag_prefix),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            tags,
            vec![
                format!("slot_transfer_gate_a_out_{transfer_gate}").into_bytes(),
                format!("slot_reduce_gate_a_out_{reduce_gate}").into_bytes(),
            ]
        );
    }

    fn compile_transfer(
        source_slots: &[(u32, Option<u32>)],
        input_type: MatrixType,
    ) -> Result<Vec<BggPublicKeyWire>, CircuitCompileError> {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let circuit_input = circuit.input(1).as_single_wire();
        let transferred = circuit.slot_transfer_gate(circuit_input, source_slots);
        circuit.output([transferred]);
        let mut builder = GraphBuilder::new("slot-transfer-validation", Vec::new());
        let hash_key = builder.bytes_input("hash_key", 32);
        let input = BggPublicKeyWire {
            matrix: builder.input("input", input_type),
            reveal_plaintext: false,
        };
        let compiler = PolyCircuitCompiler {
            public_key: BggPublicKeyCompiler {
                base: IntExpr::constant(2),
                decomposed_type: matrix_type(10, 10),
            },
        };
        let mut lowering = BggSlotTransferPublicKeyLowering {
            hash_key,
            public_key_type: matrix_type(2, 10),
            configured_slot_count: 3,
            requests: Vec::new(),
        };
        compiler.compile_public_keys_with_lowering(
            &mut builder,
            &circuit,
            input.clone(),
            [input],
            &mut lowering,
        )
    }

    #[test]
    fn public_key_lowering_preserves_legacy_slot_transfer_boundaries() {
        assert!(compile_transfer(&[], matrix_type(2, 10)).is_ok());
        assert!(matches!(
            compile_transfer(&[(0, None), (1, None), (2, None), (0, None)], matrix_type(2, 10)),
            Err(CircuitCompileError::InvalidSlotTransfer { .. })
        ));
        assert!(matches!(
            compile_transfer(&[(0, None)], matrix_type(1, 10)),
            Err(CircuitCompileError::InvalidSlotTransfer { .. })
        ));
    }
}
