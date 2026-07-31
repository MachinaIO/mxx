use crate::{
    AdvancedGateLowering, BggPolyEncodingWire, BggPublicKeyWire, BggSlotTransferArtifactCompiler,
    BggSlotTransferGateWires, BggSlotTransferPublicSlotWires, CircuitCompileError,
    slot_transfer_artifact::gate_preimage_name, slot_transfer_public_key::gate_token,
};
use mxx_gadgets::{
    Poly,
    circuit::{GateInstance, PolyCircuit},
};
use mxx_ir_core::{
    GraphBuilder, IntExpr, MatrixFamilyWire, MatrixWire, WireRef,
    node::{ConcatAxis, ConstantMatrix, HashVariant, LoopInputMode, MatrixBinaryOp},
};
use num_bigint::BigInt;

/// Online Graph IR lowering for the historical polynomial BGG+ slot-transfer
/// construction.
///
/// `c_b0` is the evaluator's BGG+ row vector under the preprocessing matrix
/// B0. All remaining matrices are public typed artifacts. The lowering keeps
/// chunked preimages lazy and concatenates only their 1-row products.
#[derive(Clone, Debug)]
pub struct BggPolySlotTransferLowering {
    pub artifact: BggSlotTransferArtifactCompiler,
    pub hash_key: WireRef,
    pub c_b0: MatrixWire,
    pub slots: BggSlotTransferPublicSlotWires,
    pub gates: BggSlotTransferGateWires,
}

impl BggPolySlotTransferLowering {
    fn output_public_key(
        &self,
        builder: &mut GraphBuilder,
        gate: GateInstance<'_>,
        reduction: bool,
    ) -> BggPublicKeyWire {
        let operation = if reduction { "slot_reduce" } else { "slot_transfer" };
        BggPublicKeyWire {
            matrix: builder.hash_sample(
                self.hash_key,
                self.artifact.public_key_type(),
                HashVariant::Plain,
                format!("{operation}_gate_a_out_{}", gate_token(gate)).into_bytes(),
                Vec::new(),
                None,
                None,
            ),
            reveal_plaintext: true,
        }
    }

    fn input_slot_count(&self, input: &BggPolyEncodingWire) -> Option<usize> {
        let IntExpr::Const(count) = &input.vectors.count else {
            return None;
        };
        let count = usize::try_from(count.clone()).ok()?;
        (input.pubkey.matrix.matrix_type == self.artifact.public_key_type() &&
            count <= self.artifact.slot_count &&
            input.vectors.matrix_type ==
                self.artifact.matrix_type(1, self.artifact.gadget_columns()) &&
            input.plaintexts.as_ref().is_some_and(|plaintexts| {
                plaintexts.count == input.vectors.count &&
                    plaintexts.matrix_type == self.artifact.matrix_type(1, 1)
            }) &&
            self.c_b0.matrix_type ==
                self.artifact.matrix_type(1, self.artifact.b0_public_columns()))
        .then_some(count)
    }

    fn gate_families(
        &self,
        reduction: bool,
        identity: &str,
    ) -> Result<Vec<MatrixFamilyWire>, CircuitCompileError> {
        self.artifact
            .chunks(self.artifact.gadget_columns())
            .into_iter()
            .enumerate()
            .map(|(chunk, _)| {
                let name = gate_preimage_name(reduction, identity, chunk);
                self.gates
                    .preimage_chunks
                    .get(&name)
                    .cloned()
                    .ok_or(CircuitCompileError::MissingSlotTransferArtifact { name })
            })
            .collect()
    }

    fn empty_transfer(
        &self,
        builder: &mut GraphBuilder,
        pubkey: BggPublicKeyWire,
        identity: &str,
    ) -> Result<BggPolyEncodingWire, CircuitCompileError> {
        let mut body =
            GraphBuilder::new(format!("bgg-poly-slot-transfer-empty-{identity}"), Vec::new());
        let vector = body.constant_matrix(
            self.artifact.matrix_type(1, self.artifact.gadget_columns()),
            ConstantMatrix::Zero,
        );
        let plaintext = body.constant_matrix(self.artifact.matrix_type(1, 1), ConstantMatrix::Zero);
        body.value_output_wire("0_vector", vector.wire);
        body.value_output_wire("1_plaintext", plaintext.wire);
        let mut outputs = builder.parallel_loop(
            body.finish(),
            IntExpr::constant(0),
            "destination",
            Vec::new(),
            Vec::new(),
            Vec::new(),
            &[vector.matrix_type, plaintext.matrix_type],
        )?;
        Ok(BggPolyEncodingWire {
            vectors: outputs.remove(0),
            pubkey,
            plaintexts: Some(outputs.remove(0)),
        })
    }

    fn product_chunks(
        &self,
        builder: &mut GraphBuilder,
        left: &MatrixWire,
        families: &[MatrixFamilyWire],
        index: WireRef,
        output_columns: usize,
    ) -> MatrixWire {
        let chunks = families
            .iter()
            .map(|family| {
                let right = builder.family_get_dynamic(family, index);
                builder.matrix_binary(
                    MatrixBinaryOp::Multiply,
                    left,
                    &right,
                    mxx_ir_core::types::MatrixType {
                        rows: IntExpr::constant(1),
                        ..family.matrix_type.clone()
                    },
                )
            })
            .collect::<Vec<_>>();
        if chunks.len() == 1 {
            chunks[0].clone()
        } else {
            builder.concat(
                ConcatAxis::Columns,
                &chunks,
                self.artifact.matrix_type(1, output_columns),
            )
        }
    }

    fn transfer(
        &self,
        builder: &mut GraphBuilder,
        input: &BggPolyEncodingWire,
        source_slots: &[(u32, Option<u32>)],
        gate: GateInstance<'_>,
    ) -> Result<BggPolyEncodingWire, CircuitCompileError> {
        let identity = gate_token(gate);
        let pubkey = self.output_public_key(builder, gate, false);
        if source_slots.is_empty() {
            return self.empty_transfer(builder, pubkey, &identity);
        }
        let Some(input_slot_count) = self.input_slot_count(input) else {
            return Err(CircuitCompileError::InvalidSlotTransfer {
                gate: gate.local_gate().index(),
            });
        };
        if source_slots.len() > input_slot_count ||
            source_slots.len() > self.artifact.slot_count ||
            source_slots.iter().any(|(source, _)| *source as usize >= input_slot_count)
        {
            return Err(CircuitCompileError::InvalidSlotTransfer {
                gate: gate.local_gate().index(),
            });
        }
        let gate_families = self.gate_families(false, &identity)?;
        if gate_families.iter().any(|family| family.count != IntExpr::constant(source_slots.len()))
        {
            return Err(CircuitCompileError::InvalidSlotTransfer {
                gate: gate.local_gate().index(),
            });
        }

        let mut body = GraphBuilder::new(format!("bgg-poly-slot-transfer-{identity}"), Vec::new());
        let c_b0 = body.input("000_c_b0", self.c_b0.matrix_type.clone());
        let vectors = body.family_input(
            "001_vectors",
            input.vectors.matrix_type.clone(),
            input.vectors.count.clone(),
        );
        let plaintexts = input.plaintexts.as_ref().expect("validated plaintext family");
        let plaintexts_input = body.family_input(
            "002_plaintexts",
            plaintexts.matrix_type.clone(),
            plaintexts.count.clone(),
        );
        let slot_public_keys = body.family_input(
            "003_slot_public_keys",
            self.slots.public_keys.matrix_type.clone(),
            self.slots.public_keys.count.clone(),
        );
        let mut args =
            vec![self.c_b0.wire, input.vectors.wire, plaintexts.wire, self.slots.public_keys.wire];
        let mut next_input = 4;
        let b0_families = import_body_families(
            &mut body,
            &mut args,
            &mut next_input,
            "slot_b0",
            &self.slots.b0_preimage_chunks,
        );
        let b1_families = import_body_families(
            &mut body,
            &mut args,
            &mut next_input,
            "slot_b1",
            &self.slots.b1_preimage_chunks,
        );
        let gate_families =
            import_body_families(&mut body, &mut args, &mut next_input, "gate", &gate_families);

        let destination = body.evaluate_int(IntExpr::Var("destination".to_owned()));
        let sources =
            source_slots.iter().map(|(source, _)| body.constant_int(*source)).collect::<Vec<_>>();
        let source = body.select_wire(destination, &sources);
        let input_vector = body.family_get_dynamic(&vectors, source);
        let input_plaintext = body.family_get_dynamic(&plaintexts_input, source);
        let output_plaintext = body.constant_coefficient(&input_plaintext, IntExpr::constant(0));
        let slot_public_key = body.family_get_dynamic(&slot_public_keys, destination);
        let decomposed = body.gadget_decompose(
            &slot_public_key,
            self.artifact.gadget_base.clone(),
            self.artifact
                .matrix_type(self.artifact.gadget_columns(), self.artifact.gadget_columns()),
        );
        let c_b1 = self.product_chunks(
            &mut body,
            &c_b0,
            &b0_families,
            source,
            self.artifact.b1_public_columns(),
        );
        let c_transfer = self.product_chunks(
            &mut body,
            &c_b1,
            &b1_families,
            destination,
            self.artifact.gadget_columns(),
        );
        let first = body.matrix_binary(
            MatrixBinaryOp::Multiply,
            &input_vector,
            &decomposed,
            input.vectors.matrix_type.clone(),
        );
        let second = body.matrix_binary(
            MatrixBinaryOp::Multiply,
            &c_transfer,
            &output_plaintext,
            input.vectors.matrix_type.clone(),
        );
        let mut pre_output = body.matrix_binary(
            MatrixBinaryOp::Add,
            &first,
            &second,
            input.vectors.matrix_type.clone(),
        );
        let scalar_branches = source_slots
            .iter()
            .map(|(_, scalar)| {
                body.constant_polynomial(
                    self.artifact.matrix_type(1, 1),
                    [BigInt::from(scalar.unwrap_or(1))],
                )
            })
            .collect::<Vec<_>>();
        let scalar = body.select(destination, &scalar_branches);
        pre_output = body.matrix_binary(
            MatrixBinaryOp::Multiply,
            &pre_output,
            &scalar,
            input.vectors.matrix_type.clone(),
        );
        let output_plaintext = body.matrix_binary(
            MatrixBinaryOp::Multiply,
            &output_plaintext,
            &scalar,
            plaintexts.matrix_type.clone(),
        );
        let c_gate = self.product_chunks(
            &mut body,
            &c_b0,
            &gate_families,
            destination,
            self.artifact.gadget_columns(),
        );
        let output_vector = body.matrix_binary(
            MatrixBinaryOp::Add,
            &c_gate,
            &pre_output,
            input.vectors.matrix_type.clone(),
        );
        body.value_output_wire("0_vector", output_vector.wire);
        body.value_output_wire("1_plaintext", output_plaintext.wire);
        let mut outputs = builder.parallel_loop(
            body.finish(),
            IntExpr::constant(source_slots.len()),
            "destination",
            Vec::new(),
            args.clone(),
            vec![LoopInputMode::Broadcast; args.len()],
            &[output_vector.matrix_type, output_plaintext.matrix_type],
        )?;
        Ok(BggPolyEncodingWire {
            vectors: outputs.remove(0),
            pubkey,
            plaintexts: Some(outputs.remove(0)),
        })
    }

    fn reduce(
        &self,
        builder: &mut GraphBuilder,
        inputs: &[BggPolyEncodingWire],
        source_slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<BggPolyEncodingWire, CircuitCompileError> {
        let input_slot_counts =
            inputs.iter().map(|input| self.input_slot_count(input)).collect::<Vec<_>>();
        let common_input_slot_count = input_slot_counts.first().copied().flatten();
        if inputs.is_empty() ||
            inputs.len() > source_slot_count ||
            source_slot_count == 0 ||
            source_slot_count > self.artifact.slot_count ||
            common_input_slot_count.is_none_or(|count| count < source_slot_count) ||
            input_slot_counts.iter().any(|count| *count != common_input_slot_count)
        {
            return Err(CircuitCompileError::InvalidSlotTransfer {
                gate: gate.local_gate().index(),
            });
        }
        let identity = gate_token(gate);
        let gate_families = self.gate_families(true, &identity)?;
        if gate_families.iter().any(|family| family.count != IntExpr::constant(inputs.len())) {
            return Err(CircuitCompileError::InvalidSlotTransfer {
                gate: gate.local_gate().index(),
            });
        }
        let pubkey = self.output_public_key(builder, gate, true);
        let mut body = GraphBuilder::new(format!("bgg-poly-slot-reduce-{identity}"), Vec::new());
        let c_b0 = body.input("000_c_b0", self.c_b0.matrix_type.clone());
        let slot_public_keys = body.family_input(
            "001_slot_public_keys",
            self.slots.public_keys.matrix_type.clone(),
            self.slots.public_keys.count.clone(),
        );
        let mut args = vec![self.c_b0.wire, self.slots.public_keys.wire];
        let mut next_input = 2;
        let vector_inputs = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                let family = body.family_input(
                    format!("{:03}_vectors_{index}", next_input),
                    input.vectors.matrix_type.clone(),
                    input.vectors.count.clone(),
                );
                next_input += 1;
                args.push(input.vectors.wire);
                family
            })
            .collect::<Vec<_>>();
        let plaintext_inputs = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                let plaintexts = input.plaintexts.as_ref().expect("validated plaintext family");
                let family = body.family_input(
                    format!("{:03}_plaintexts_{index}", next_input),
                    plaintexts.matrix_type.clone(),
                    plaintexts.count.clone(),
                );
                next_input += 1;
                args.push(plaintexts.wire);
                family
            })
            .collect::<Vec<_>>();
        let b0_families = import_body_families(
            &mut body,
            &mut args,
            &mut next_input,
            "slot_b0",
            &self.slots.b0_preimage_chunks,
        );
        let b1_families = import_body_families(
            &mut body,
            &mut args,
            &mut next_input,
            "slot_b1",
            &self.slots.b1_preimage_chunks,
        );
        let gate_families =
            import_body_families(&mut body, &mut args, &mut next_input, "gate", &gate_families);

        let destination = body.evaluate_int(IntExpr::Var("destination".to_owned()));
        let slot_public_key = body.family_get_dynamic(&slot_public_keys, destination);
        let decomposed = body.gadget_decompose(
            &slot_public_key,
            self.artifact.gadget_base.clone(),
            self.artifact
                .matrix_type(self.artifact.gadget_columns(), self.artifact.gadget_columns()),
        );
        let mut pre_output = None;
        let mut output_plaintext = None;
        for source in 0..source_slot_count {
            let source_wire = body.constant_int(source);
            let vector_branches = vector_inputs
                .iter()
                .map(|family| body.family_get_static(family, IntExpr::constant(source)))
                .collect::<Vec<_>>();
            let input_vector = body.select(destination, &vector_branches);
            let plaintext_branches = plaintext_inputs
                .iter()
                .map(|family| body.family_get_static(family, IntExpr::constant(source)))
                .collect::<Vec<_>>();
            let plaintext = body.select(destination, &plaintext_branches);
            let plaintext = body.constant_coefficient(&plaintext, IntExpr::constant(0));
            let c_b1 = self.product_chunks(
                &mut body,
                &c_b0,
                &b0_families,
                source_wire,
                self.artifact.b1_public_columns(),
            );
            let c_transfer = self.product_chunks(
                &mut body,
                &c_b1,
                &b1_families,
                destination,
                self.artifact.gadget_columns(),
            );
            let first = body.matrix_binary(
                MatrixBinaryOp::Multiply,
                &input_vector,
                &decomposed,
                self.artifact.matrix_type(1, self.artifact.gadget_columns()),
            );
            let second = body.matrix_binary(
                MatrixBinaryOp::Multiply,
                &c_transfer,
                &plaintext,
                self.artifact.matrix_type(1, self.artifact.gadget_columns()),
            );
            let term = body.matrix_binary(
                MatrixBinaryOp::Add,
                &first,
                &second,
                self.artifact.matrix_type(1, self.artifact.gadget_columns()),
            );
            let rotation = body.constant_matrix(
                self.artifact.matrix_type(1, 1),
                ConstantMatrix::Rotation { exponent: IntExpr::constant(source) },
            );
            let term = body.matrix_binary(
                MatrixBinaryOp::Multiply,
                &term,
                &rotation,
                self.artifact.matrix_type(1, self.artifact.gadget_columns()),
            );
            pre_output = Some(match pre_output {
                Some(sum) => body.matrix_binary(
                    MatrixBinaryOp::Add,
                    &sum,
                    &term,
                    self.artifact.matrix_type(1, self.artifact.gadget_columns()),
                ),
                None => term,
            });
            let plaintext = body.matrix_binary(
                MatrixBinaryOp::Multiply,
                &plaintext,
                &rotation,
                self.artifact.matrix_type(1, 1),
            );
            output_plaintext = Some(match output_plaintext {
                Some(sum) => body.matrix_binary(
                    MatrixBinaryOp::Add,
                    &sum,
                    &plaintext,
                    self.artifact.matrix_type(1, 1),
                ),
                None => plaintext,
            });
        }
        let c_gate = self.product_chunks(
            &mut body,
            &c_b0,
            &gate_families,
            destination,
            self.artifact.gadget_columns(),
        );
        let output_vector = body.matrix_binary(
            MatrixBinaryOp::Add,
            &c_gate,
            &pre_output.expect("source_slot_count was checked nonzero"),
            self.artifact.matrix_type(1, self.artifact.gadget_columns()),
        );
        let output_plaintext = output_plaintext.expect("source_slot_count was checked nonzero");
        body.value_output_wire("0_vector", output_vector.wire);
        body.value_output_wire("1_plaintext", output_plaintext.wire);
        let mut outputs = builder.parallel_loop(
            body.finish(),
            IntExpr::constant(inputs.len()),
            "destination",
            Vec::new(),
            args.clone(),
            vec![LoopInputMode::Broadcast; args.len()],
            &[output_vector.matrix_type, output_plaintext.matrix_type],
        )?;
        Ok(BggPolyEncodingWire {
            vectors: outputs.remove(0),
            pubkey,
            plaintexts: Some(outputs.remove(0)),
        })
    }
}

impl<P: Poly> AdvancedGateLowering<P, BggPolyEncodingWire> for BggPolySlotTransferLowering {
    fn slot_transfer(
        &mut self,
        builder: &mut GraphBuilder,
        input: &BggPolyEncodingWire,
        source_slots: &[(u32, Option<u32>)],
        gate: GateInstance<'_>,
    ) -> Result<BggPolyEncodingWire, CircuitCompileError> {
        self.transfer(builder, input, source_slots, gate)
    }

    fn slot_reduce(
        &mut self,
        builder: &mut GraphBuilder,
        inputs: &[BggPolyEncodingWire],
        slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<BggPolyEncodingWire, CircuitCompileError> {
        self.reduce(builder, inputs, slot_count, gate)
    }

    fn public_lookup(
        &mut self,
        _builder: &mut GraphBuilder,
        _circuit: &PolyCircuit<P>,
        _lookup_id: usize,
        _input: &BggPolyEncodingWire,
        gate: GateInstance<'_>,
    ) -> Result<BggPolyEncodingWire, CircuitCompileError> {
        Err(CircuitCompileError::MissingGateContext {
            gate: gate.local_gate().index(),
            kind: "public lookup",
        })
    }
}

fn import_body_families(
    body: &mut GraphBuilder,
    args: &mut Vec<WireRef>,
    next_input: &mut usize,
    label: &str,
    families: &[MatrixFamilyWire],
) -> Vec<MatrixFamilyWire> {
    families
        .iter()
        .enumerate()
        .map(|(chunk, family)| {
            let input = body.family_input(
                format!("{:03}_{label}_{chunk}", *next_input),
                family.matrix_type.clone(),
                family.count.clone(),
            );
            *next_input += 1;
            args.push(family.wire);
            input
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BggPublicKeyCompiler, PolyCircuitCompiler};
    use mxx_gadgets::circuit::PolyCircuit;
    use mxx_ir_core::{ParamEnv, artifact::ArtifactConfidentiality, types::MatrixType, validate};
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use mxx_runtime::{
        RuntimeValue,
        artifact::MemoryArtifactStore,
        backend::poly::{CpuDcrtBackend, cpu_backend},
        execute,
        transcript::SamplingMode,
    };
    use std::collections::BTreeMap;

    fn matrix_type(parameters: &DCRTPolyParams, rows: usize, columns: usize) -> MatrixType {
        MatrixType {
            modulus: IntExpr::constant(BigInt::from(parameters.modulus().as_ref().clone())),
            ring_dimension: IntExpr::constant(parameters.ring_dimension()),
            rows: IntExpr::constant(rows),
            columns: IntExpr::constant(columns),
        }
    }

    fn matrix(
        parameters: &DCRTPolyParams,
        rows: usize,
        columns: usize,
        seed: usize,
    ) -> DCRTPolyMatrix {
        DCRTPolyMatrix::from_poly_vec(
            parameters,
            (0..rows)
                .map(|row| {
                    (0..columns)
                        .map(|column| {
                            DCRTPoly::from_usize_to_constant(
                                parameters,
                                seed + row * columns + column + 1,
                            )
                        })
                        .collect()
                })
                .collect(),
        )
    }

    fn concat_columns(mut chunks: Vec<DCRTPolyMatrix>) -> DCRTPolyMatrix {
        let first = chunks.remove(0);
        if chunks.is_empty() {
            first
        } else {
            let refs = chunks.iter().collect::<Vec<_>>();
            first.concat_columns(&refs)
        }
    }

    fn input_family(
        builder: &mut GraphBuilder,
        inputs: &mut BTreeMap<String, RuntimeValue<CpuDcrtBackend>>,
        prefix: &str,
        matrix_type: MatrixType,
        values: &[DCRTPolyMatrix],
    ) -> MatrixFamilyWire {
        let wires = values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let name = format!("{prefix}_{index}");
                inputs.insert(name.clone(), RuntimeValue::matrix(value.clone()));
                builder.input(name, matrix_type.clone())
            })
            .collect::<Vec<_>>();
        builder.family_pack(&wires).expect("nonempty test family")
    }

    #[test]
    fn online_transfer_matches_the_legacy_chunked_formula() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let artifact = BggSlotTransferArtifactCompiler {
            modulus: IntExpr::constant(BigInt::from(parameters.modulus().as_ref().clone())),
            ring_dimension: IntExpr::constant(parameters.ring_dimension()),
            secret_size: 1,
            slot_count: 3,
            digit_count: parameters.modulus_digits(),
            chunk_columns: 2,
            gadget_base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
            trapdoor_sigma: mxx_ir_core::RealExpr::from_f64_exact(4.578).expect("finite sigma"),
            error_sigma: mxx_ir_core::RealExpr::from_f64_exact(0.0).expect("finite sigma"),
        };
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let circuit_input = circuit.input(1).as_single_wire();
        let circuit_input_two = circuit.input(1).as_single_wire();
        let transferred = circuit.slot_transfer_gate(circuit_input, &[(2, None), (0, Some(3))]);
        let gate_identity = transferred.as_single_wire().index().to_string();
        let reduced = circuit.slot_reduce_gate(&[circuit_input, circuit_input_two], 2);
        let reduce_identity = reduced.as_single_wire().index().to_string();
        circuit.output([transferred, reduced]);

        let mut builder = GraphBuilder::new("poly-slot-transfer-online-test", Vec::new());
        let hash_key = builder.bytes_input("hash_key", 32);
        let mut runtime_inputs =
            BTreeMap::from([("hash_key".to_owned(), RuntimeValue::Bytes(vec![0x61; 32]))]);
        let c_b0_value = matrix(&parameters, 1, artifact.b0_public_columns(), 10);
        runtime_inputs.insert("c_b0".to_owned(), RuntimeValue::matrix(c_b0_value.clone()));
        let c_b0 = builder.input("c_b0", matrix_type(&parameters, 1, artifact.b0_public_columns()));
        let vector_values = (0..3)
            .map(|slot| matrix(&parameters, 1, artifact.gadget_columns(), 100 + slot * 20))
            .collect::<Vec<_>>();
        let vectors = input_family(
            &mut builder,
            &mut runtime_inputs,
            "vector",
            matrix_type(&parameters, 1, artifact.gadget_columns()),
            &vector_values,
        );
        let plaintext_values = (0..3)
            .map(|slot| {
                DCRTPolyMatrix::from_poly_vec(
                    &parameters,
                    vec![vec![DCRTPoly::from_u32s(
                        &parameters,
                        &[(5 + slot) as u32, (40 + slot) as u32],
                    )]],
                )
            })
            .collect::<Vec<_>>();
        let plaintexts = input_family(
            &mut builder,
            &mut runtime_inputs,
            "plaintext",
            matrix_type(&parameters, 1, 1),
            &plaintext_values,
        );
        let input_public_key_value = matrix(&parameters, 1, artifact.gadget_columns(), 200);
        runtime_inputs
            .insert("input_public_key".to_owned(), RuntimeValue::matrix(input_public_key_value));
        let input = BggPolyEncodingWire {
            vectors,
            pubkey: BggPublicKeyWire {
                matrix: builder.input("input_public_key", artifact.public_key_type()),
                reveal_plaintext: true,
            },
            plaintexts: Some(plaintexts),
        };
        let vector_values_two = (0..3)
            .map(|slot| matrix(&parameters, 1, artifact.gadget_columns(), 2200 + slot * 20))
            .collect::<Vec<_>>();
        let vectors_two = input_family(
            &mut builder,
            &mut runtime_inputs,
            "vector_two",
            matrix_type(&parameters, 1, artifact.gadget_columns()),
            &vector_values_two,
        );
        let plaintext_values_two = (0..3)
            .map(|slot| {
                DCRTPolyMatrix::from_poly_vec(
                    &parameters,
                    vec![vec![DCRTPoly::from_u32s(
                        &parameters,
                        &[(15 + slot) as u32, (60 + slot) as u32],
                    )]],
                )
            })
            .collect::<Vec<_>>();
        let plaintexts_two = input_family(
            &mut builder,
            &mut runtime_inputs,
            "plaintext_two",
            matrix_type(&parameters, 1, 1),
            &plaintext_values_two,
        );
        runtime_inputs.insert(
            "input_public_key_two".to_owned(),
            RuntimeValue::matrix(matrix(&parameters, 1, artifact.gadget_columns(), 2400)),
        );
        let input_two = BggPolyEncodingWire {
            vectors: vectors_two,
            pubkey: BggPublicKeyWire {
                matrix: builder.input("input_public_key_two", artifact.public_key_type()),
                reveal_plaintext: true,
            },
            plaintexts: Some(plaintexts_two),
        };
        let slot_public_key_values = (0..artifact.slot_count)
            .map(|slot| matrix(&parameters, 1, artifact.gadget_columns(), 300 + slot * 20))
            .collect::<Vec<_>>();
        let slot_public_keys = input_family(
            &mut builder,
            &mut runtime_inputs,
            "slot_public_key",
            artifact.public_key_type(),
            &slot_public_key_values,
        );

        let mut b0_values = Vec::new();
        let b0_preimage_chunks = artifact
            .chunks(artifact.b1_public_columns())
            .into_iter()
            .enumerate()
            .map(|(chunk, columns)| {
                let values = (0..artifact.slot_count)
                    .map(|slot| {
                        matrix(
                            &parameters,
                            artifact.b0_public_columns(),
                            columns.end - columns.start,
                            400 + chunk * 100 + slot * 10,
                        )
                    })
                    .collect::<Vec<_>>();
                let family = input_family(
                    &mut builder,
                    &mut runtime_inputs,
                    &format!("b0_chunk_{chunk}"),
                    artifact.matrix_type(artifact.b0_public_columns(), columns.end - columns.start),
                    &values,
                );
                b0_values.push(values);
                family
            })
            .collect::<Vec<_>>();
        let mut b1_values = Vec::new();
        let b1_preimage_chunks = artifact
            .chunks(artifact.gadget_columns())
            .into_iter()
            .enumerate()
            .map(|(chunk, columns)| {
                let values = (0..artifact.slot_count)
                    .map(|slot| {
                        matrix(
                            &parameters,
                            artifact.b1_public_columns(),
                            columns.end - columns.start,
                            800 + chunk * 100 + slot * 10,
                        )
                    })
                    .collect::<Vec<_>>();
                let family = input_family(
                    &mut builder,
                    &mut runtime_inputs,
                    &format!("b1_chunk_{chunk}"),
                    artifact.matrix_type(artifact.b1_public_columns(), columns.end - columns.start),
                    &values,
                );
                b1_values.push(values);
                family
            })
            .collect::<Vec<_>>();
        let mut gate_values = Vec::new();
        let mut reduce_gate_values = Vec::new();
        let mut gate_preimage_chunks = BTreeMap::new();
        for (chunk, columns) in artifact.chunks(artifact.gadget_columns()).into_iter().enumerate() {
            let values = (0..2)
                .map(|destination| {
                    matrix(
                        &parameters,
                        artifact.b0_public_columns(),
                        columns.end - columns.start,
                        1200 + chunk * 100 + destination * 10,
                    )
                })
                .collect::<Vec<_>>();
            let family = input_family(
                &mut builder,
                &mut runtime_inputs,
                &format!("gate_chunk_{chunk}"),
                artifact.matrix_type(artifact.b0_public_columns(), columns.end - columns.start),
                &values,
            );
            gate_values.push(values);
            gate_preimage_chunks.insert(gate_preimage_name(false, &gate_identity, chunk), family);

            let reduce_values = (0..2)
                .map(|destination| {
                    matrix(
                        &parameters,
                        artifact.b0_public_columns(),
                        columns.end - columns.start,
                        1600 + chunk * 100 + destination * 10,
                    )
                })
                .collect::<Vec<_>>();
            let reduce_family = input_family(
                &mut builder,
                &mut runtime_inputs,
                &format!("reduce_gate_chunk_{chunk}"),
                artifact.matrix_type(artifact.b0_public_columns(), columns.end - columns.start),
                &reduce_values,
            );
            reduce_gate_values.push(reduce_values);
            gate_preimage_chunks
                .insert(gate_preimage_name(true, &reduce_identity, chunk), reduce_family);
        }
        let mut lowering = BggPolySlotTransferLowering {
            artifact: artifact.clone(),
            hash_key,
            c_b0,
            slots: BggSlotTransferPublicSlotWires {
                public_keys: slot_public_keys,
                b0_preimage_chunks,
                b1_preimage_chunks,
            },
            gates: BggSlotTransferGateWires { preimage_chunks: gate_preimage_chunks },
        };
        let compiler = PolyCircuitCompiler {
            public_key: BggPublicKeyCompiler {
                base: artifact.gadget_base.clone(),
                decomposed_type: artifact
                    .matrix_type(artifact.gadget_columns(), artifact.gadget_columns()),
            },
        };
        let mut outputs = compiler
            .compile_poly_encodings_with_lowering(
                &mut builder,
                &circuit,
                input.clone(),
                [input, input_two],
                &mut lowering,
            )
            .expect("slot-transfer lowering");
        let output = outputs.remove(0);
        let reduced_output = outputs.remove(0);
        for destination in 0..2 {
            let vector = builder.family_get_static(&output.vectors, IntExpr::constant(destination));
            let plaintext = builder.family_get_static(
                output.plaintexts.as_ref().expect("revealed output"),
                IntExpr::constant(destination),
            );
            builder.output(
                format!("vector_{destination}"),
                &vector,
                ArtifactConfidentiality::Public,
            );
            builder.output(
                format!("plaintext_{destination}"),
                &plaintext,
                ArtifactConfidentiality::Public,
            );
        }
        for destination in 0..2 {
            let reduced_vector =
                builder.family_get_static(&reduced_output.vectors, IntExpr::constant(destination));
            let reduced_plaintext = builder.family_get_static(
                reduced_output.plaintexts.as_ref().expect("revealed reduction"),
                IntExpr::constant(destination),
            );
            builder.output(
                format!("reduced_vector_{destination}"),
                &reduced_vector,
                ArtifactConfidentiality::Public,
            );
            builder.output(
                format!("reduced_plaintext_{destination}"),
                &reduced_plaintext,
                ArtifactConfidentiality::Public,
            );
        }
        let graph = validate(&builder.finish(), &ParamEnv::default()).expect("valid graph");
        let mut backend = cpu_backend([parameters.clone()]);
        let mut store = MemoryArtifactStore::default();
        let result = execute(&graph, &mut backend, runtime_inputs, &mut store, SamplingMode::Fresh)
            .expect("execution");

        for (destination, (source, scalar)) in [(2usize, 1usize), (0, 3)].into_iter().enumerate() {
            let c_b1 = concat_columns(
                b0_values.iter().map(|chunk| c_b0_value.clone() * &chunk[source]).collect(),
            );
            let c_transfer = concat_columns(
                b1_values.iter().map(|chunk| c_b1.clone() * &chunk[destination]).collect(),
            );
            let plaintext = DCRTPoly::from_usize_to_constant(&parameters, 5 + source);
            let scalar = DCRTPoly::from_usize_to_constant(&parameters, scalar);
            let pre_output = (vector_values[source].clone() *
                &slot_public_key_values[destination].decompose() +
                &(c_transfer * &plaintext)) *
                &scalar;
            let c_gate = concat_columns(
                gate_values.iter().map(|chunk| c_b0_value.clone() * &chunk[destination]).collect(),
            );
            let expected_vector = c_gate + &pre_output;
            let expected_plaintext =
                DCRTPolyMatrix::from_poly_vec(&parameters, vec![vec![plaintext * &scalar]]);
            let RuntimeValue::Matrix(actual_vector) =
                &result.outputs[&format!("vector_{destination}")]
            else {
                panic!("vector output");
            };
            let RuntimeValue::Matrix(actual_plaintext) =
                &result.outputs[&format!("plaintext_{destination}")]
            else {
                panic!("plaintext output");
            };
            assert_eq!(actual_vector.as_ref(), &expected_vector);
            assert_eq!(actual_plaintext.as_ref(), &expected_plaintext);
        }

        for destination in 0..2 {
            let decomposed = slot_public_key_values[destination].decompose();
            let selected_vectors =
                if destination == 0 { &vector_values } else { &vector_values_two };
            let plaintext_offset = if destination == 0 { 5 } else { 15 };
            let mut expected_pre_output =
                DCRTPolyMatrix::zero(&parameters, 1, artifact.gadget_columns());
            let mut expected_reduced_plaintext = DCRTPolyMatrix::zero(&parameters, 1, 1);
            for source in 0..2 {
                let c_b1 = concat_columns(
                    b0_values.iter().map(|chunk| c_b0_value.clone() * &chunk[source]).collect(),
                );
                let c_transfer = concat_columns(
                    b1_values.iter().map(|chunk| c_b1.clone() * &chunk[destination]).collect(),
                );
                let plaintext =
                    DCRTPoly::from_usize_to_constant(&parameters, plaintext_offset + source);
                let rotation = DCRTPoly::const_rotate_poly(&parameters, source);
                let term = (selected_vectors[source].clone() * &decomposed +
                    &(c_transfer * &plaintext)) *
                    &rotation;
                expected_pre_output = expected_pre_output + &term;
                let plaintext =
                    DCRTPolyMatrix::from_poly_vec(&parameters, vec![vec![plaintext * &rotation]]);
                expected_reduced_plaintext = expected_reduced_plaintext + &plaintext;
            }
            let reduce_c_gate = concat_columns(
                reduce_gate_values
                    .iter()
                    .map(|chunk| c_b0_value.clone() * &chunk[destination])
                    .collect(),
            );
            let expected_reduced_vector = reduce_c_gate + &expected_pre_output;
            let RuntimeValue::Matrix(actual_reduced_vector) =
                &result.outputs[&format!("reduced_vector_{destination}")]
            else {
                panic!("reduced vector output");
            };
            let RuntimeValue::Matrix(actual_reduced_plaintext) =
                &result.outputs[&format!("reduced_plaintext_{destination}")]
            else {
                panic!("reduced plaintext output");
            };
            assert_eq!(actual_reduced_vector.as_ref(), &expected_reduced_vector);
            assert_eq!(actual_reduced_plaintext.as_ref(), &expected_reduced_plaintext);
        }
    }
}
