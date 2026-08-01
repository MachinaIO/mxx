//! Online cryptographic slot transfer for polynomial BGG+ encodings.

use crate::{
    BggPolyEncodingCompiler, BggPolyEncodingWire, BggPublicKeyWire,
    BggSlotTransferArtifactCompiler, BggSlotTransferGateWires, BggSlotTransferPublicSlotWires,
    CircuitCompileError, slot_transfer_artifact::gate_preimage_name,
    slot_transfer_public_key::gate_token,
};
use mxx_dsl::{Bytes, Family, HashTag, Mat, Parallel};
use mxx_gadgets::{
    Poly,
    circuit::{CircuitLoweringTypes, GateInstance, SlotOperationLowering},
};
use mxx_ir_core::{IntExpr, node::ConcatAxis};
use rayon::prelude::*;

#[derive(Clone)]
pub struct BggPolySlotTransferLowering {
    pub compiler: BggPolyEncodingCompiler,
    pub artifact: BggSlotTransferArtifactCompiler,
    pub hash_key: Bytes,
    pub c_b0: Mat,
    pub slots: BggSlotTransferPublicSlotWires,
    pub gates: BggSlotTransferGateWires,
}

impl BggPolySlotTransferLowering {
    fn output_public_key(&self, gate: GateInstance<'_>, reduction: bool) -> BggPublicKeyWire {
        let operation = if reduction { "slot_reduce" } else { "slot_transfer" };
        BggPublicKeyWire {
            matrix: self.artifact.ring().hash_matrix(
                self.hash_key.clone(),
                HashTag::from(format!("{operation}_gate_a_out_{}", gate_token(gate)).into_bytes()),
                (self.artifact.secret_size, self.artifact.gadget_columns()),
            ),
            reveal_plaintext: true,
        }
    }

    fn product_chunks(
        &self,
        left: Mat,
        families: &[Family<Mat>],
        index: usize,
    ) -> Result<Mat, CircuitCompileError> {
        let chunks = families
            .iter()
            .map(|family| left.clone() * family.get_static(index))
            .collect::<Vec<_>>();
        chunks
            .first()
            .cloned()
            .map(
                |first| {
                    if chunks.len() == 1 { first } else { Mat::concat(ConcatAxis::Columns, chunks) }
                },
            )
            .ok_or_else(|| {
                CircuitCompileError::Structure("empty slot-transfer chunk list".to_owned())
            })
    }

    fn gate_families(
        &self,
        reduction: bool,
        identity: &str,
    ) -> Result<Vec<Family<Mat>>, CircuitCompileError> {
        self.artifact
            .chunks(self.artifact.gadget_columns())
            .into_par_iter()
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

    fn validate_input(&self, input: &BggPolyEncodingWire) -> Result<(), CircuitCompileError> {
        if input.vectors.element_type() !=
            &self.artifact.matrix_type(1, self.artifact.gadget_columns()) ||
            input.pubkey.matrix.matrix_type() != &self.artifact.public_key_type() ||
            input.plaintexts.as_ref().is_none_or(|values| {
                values.count() != input.vectors.count() ||
                    values.element_type() != &self.artifact.matrix_type(1, 1)
            })
        {
            return Err(CircuitCompileError::Structure(
                "slot-transfer input has incompatible BGG layout".to_owned(),
            ));
        }
        Ok(())
    }

    fn transfer(
        &self,
        input: &BggPolyEncodingWire,
        source_slots: &[(u32, Option<u32>)],
        gate: GateInstance<'_>,
    ) -> Result<BggPolyEncodingWire, CircuitCompileError> {
        self.validate_input(input)?;
        if source_slots.len() > self.artifact.slot_count ||
            source_slots
                .par_iter()
                .any(|(source, _)| *source as usize >= self.artifact.slot_count)
        {
            return Err(CircuitCompileError::InvalidSlotTransfer {
                gate: gate.local_gate().index(),
            });
        }
        let identity = gate_token(gate);
        if source_slots.is_empty() {
            let ring = self.artifact.ring();
            let vector_columns = self.artifact.gadget_columns();
            let vectors = Parallel::range(0).map({
                let ring = ring.clone();
                move |_| ring.zero((1, vector_columns))
            })?;
            let plaintexts = Parallel::range(0).map(move |_| ring.zero((1, 1)))?;
            return Ok(BggPolyEncodingWire {
                vectors,
                pubkey: self.output_public_key(gate, false),
                plaintexts: Some(plaintexts),
            });
        }
        let gate_families = self.gate_families(false, &identity)?;
        let plaintexts = input.plaintexts.as_ref().expect("validated plaintext family");
        let outputs = source_slots
            .iter()
            .enumerate()
            .map(|(destination, (source, scalar))| {
                let source = usize::try_from(*source).expect("u32 fits usize");
                let input_vector = input.vectors.get_static(source);
                let plaintext = plaintexts.get_static(source).constant_coefficient(0);
                let slot_public = self.slots.public_keys.get_static(destination);
                let decomposed = slot_public
                    .decompose(self.artifact.gadget_base.clone(), self.artifact.digit_count)
                    .as_mat();
                let c_b1 =
                    self.product_chunks(self.c_b0.clone(), &self.slots.b0_preimage_chunks, source)?;
                let c_transfer =
                    self.product_chunks(c_b1, &self.slots.b1_preimage_chunks, destination)?;
                let scalar =
                    self.artifact.ring().polynomial([IntExpr::constant(scalar.unwrap_or(1))]);
                let pre_output =
                    (input_vector * decomposed + c_transfer * plaintext.clone()) * scalar.clone();
                let c_gate = self.product_chunks(self.c_b0.clone(), &gate_families, destination)?;
                Ok((c_gate + pre_output, plaintext * scalar))
            })
            .collect::<Result<Vec<_>, CircuitCompileError>>()?;
        let (vectors, output_plaintexts) = outputs.into_iter().unzip();
        Ok(BggPolyEncodingWire {
            vectors: Family::pack(vectors)?,
            pubkey: self.output_public_key(gate, false),
            plaintexts: Some(Family::pack(output_plaintexts)?),
        })
    }

    fn reduce(
        &self,
        inputs: &[BggPolyEncodingWire],
        source_slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<BggPolyEncodingWire, CircuitCompileError> {
        if inputs.is_empty() ||
            inputs.len() > source_slot_count ||
            source_slot_count == 0 ||
            source_slot_count > self.artifact.slot_count
        {
            return Err(CircuitCompileError::InvalidSlotTransfer {
                gate: gate.local_gate().index(),
            });
        }
        inputs.par_iter().try_for_each(|input| self.validate_input(input))?;
        let identity = gate_token(gate);
        let gate_families = self.gate_families(true, &identity)?;
        let ring = self.artifact.ring();
        let outputs = inputs
            .iter()
            .enumerate()
            .map(|(destination, input)| {
                let slot_public = self.slots.public_keys.get_static(destination);
                let decomposed = slot_public
                    .decompose(self.artifact.gadget_base.clone(), self.artifact.digit_count)
                    .as_mat();
                let input_plaintexts =
                    input.plaintexts.as_ref().expect("validated plaintext family");
                let source_terms = (0..source_slot_count)
                    .map(|source| {
                        let plaintext = input_plaintexts.get_static(source).constant_coefficient(0);
                        let c_b1 = self.product_chunks(
                            self.c_b0.clone(),
                            &self.slots.b0_preimage_chunks,
                            source,
                        )?;
                        let c_transfer =
                            self.product_chunks(c_b1, &self.slots.b1_preimage_chunks, destination)?;
                        let rotation = ring.constant(
                            (1, 1),
                            mxx_ir_core::node::ConstantMatrix::Rotation {
                                exponent: IntExpr::constant(source),
                            },
                        );
                        Ok((
                            (input.vectors.get_static(source) * decomposed.clone() +
                                c_transfer * plaintext.clone()) *
                                rotation.clone(),
                            plaintext * rotation,
                        ))
                    })
                    .collect::<Result<Vec<_>, CircuitCompileError>>()?;
                let (pre_output, output_plaintext) = source_terms
                    .into_iter()
                    .reduce(|(left_vector, left_plaintext), (right_vector, right_plaintext)| {
                        (left_vector + right_vector, left_plaintext + right_plaintext)
                    })
                    .expect("validated nonzero source slots");
                let c_gate = self.product_chunks(self.c_b0.clone(), &gate_families, destination)?;
                Ok((c_gate + pre_output, output_plaintext))
            })
            .collect::<Result<Vec<_>, CircuitCompileError>>()?;
        let (vectors, plaintexts) = outputs.into_iter().unzip();
        Ok(BggPolyEncodingWire {
            vectors: Family::pack(vectors)?,
            pubkey: self.output_public_key(gate, true),
            plaintexts: Some(Family::pack(plaintexts)?),
        })
    }
}

impl CircuitLoweringTypes for BggPolySlotTransferLowering {
    type Wire = BggPolyEncodingWire;
    type Error = CircuitCompileError;
}

impl<P: Poly> SlotOperationLowering<P> for BggPolySlotTransferLowering {
    fn slot_transfer(
        &mut self,
        input: &Self::Wire,
        source_slots: &[(u32, Option<u32>)],
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        self.transfer(input, source_slots, gate)
    }

    fn slot_reduce(
        &mut self,
        inputs: &[Self::Wire],
        slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        self.reduce(inputs, slot_count, gate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BggPublicKeyCompiler, BggSlotTransferBaseWires, BggSlotTransferPublicKeyLowering,
        NoPublicLookup, PolyCircuitCompiler,
        test_utils::{execute_graph, matrix_output},
    };
    use mxx_dsl::{DslContext, Ring};
    use mxx_gadgets::circuit::PolyCircuit;
    use mxx_ir_core::{ParamEnv, RealExpr};
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly as ConcretePoly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use mxx_runtime::{RuntimeValue, backend::poly::CpuDcrtBackend};
    use num_bigint::BigInt;
    use std::collections::BTreeMap;

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
            let references = chunks.iter().collect::<Vec<_>>();
            first.concat_columns(&references)
        }
    }

    fn input_family(
        ring: &Ring,
        inputs: &mut BTreeMap<String, RuntimeValue<CpuDcrtBackend>>,
        prefix: &str,
        shape: (usize, usize),
        values: &[DCRTPolyMatrix],
    ) -> Family<Mat> {
        Family::pack(
            values
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let name = format!("{prefix}_{index}");
                    inputs.insert(name.clone(), RuntimeValue::matrix(value.clone()));
                    ring.input(name, shape)
                })
                .collect(),
        )
        .expect("nonempty test family")
    }

    fn range_width(range: &mxx_ir_core::node::IndexRange) -> usize {
        let (IntExpr::Const(start), IntExpr::Const(end)) = (&range.start, &range.end) else {
            panic!("test slot-transfer chunks must be static")
        };
        usize::try_from(end - start).expect("nonnegative static chunk")
    }

    #[test]
    fn online_transfer_and_reduction_match_the_chunked_primitive_formulas() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let ring = Ring::new(
            BigInt::from(parameters.modulus().as_ref().clone()),
            parameters.ring_dimension() as usize,
        );
        let artifact = BggSlotTransferArtifactCompiler {
            modulus: IntExpr::constant(BigInt::from(parameters.modulus().as_ref().clone())),
            ring_dimension: IntExpr::constant(parameters.ring_dimension()),
            secret_size: 1,
            slot_count: 3,
            digit_count: parameters.modulus_digits(),
            chunk_columns: 2,
            gadget_base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
            trapdoor_sigma: RealExpr::from_f64_exact(4.578).expect("finite sigma"),
            error_sigma: RealExpr::from_f64_exact(0.0).expect("finite sigma"),
        };
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let first_gate = circuit.input(1).as_single_wire();
        let second_gate = circuit.input(1).as_single_wire();
        let transferred = circuit.slot_transfer_gate(first_gate, &[(2, None), (0, Some(3))]);
        let transfer_identity = format!("g{}_o0", transferred.as_single_wire().index());
        let reduced = circuit.slot_reduce_gate(&[first_gate, second_gate], 2);
        let reduce_identity = format!("g{}_o0", reduced.as_single_wire().index());
        circuit.output([transferred, reduced]);

        let hash_key = ring.bytes_input("hash_key", 32);
        let mut runtime_inputs =
            BTreeMap::from([("hash_key".to_owned(), RuntimeValue::Bytes(vec![0x61; 32]))]);
        let c_b0_value = matrix(&parameters, 1, artifact.b0_public_columns(), 10);
        runtime_inputs.insert("c_b0".to_owned(), RuntimeValue::matrix(c_b0_value.clone()));
        let c_b0 = ring.input("c_b0", (1, artifact.b0_public_columns()));

        let vector_values = (0..3)
            .map(|slot| matrix(&parameters, 1, artifact.gadget_columns(), 100 + slot * 20))
            .collect::<Vec<_>>();
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
        let input_public_key_value = matrix(&parameters, 1, artifact.gadget_columns(), 200);
        runtime_inputs
            .insert("input_public_key".to_owned(), RuntimeValue::matrix(input_public_key_value));
        let input = BggPolyEncodingWire {
            vectors: input_family(
                &ring,
                &mut runtime_inputs,
                "vector",
                (1, artifact.gadget_columns()),
                &vector_values,
            ),
            pubkey: BggPublicKeyWire {
                matrix: ring.input("input_public_key", (1, artifact.gadget_columns())),
                reveal_plaintext: true,
            },
            plaintexts: Some(input_family(
                &ring,
                &mut runtime_inputs,
                "plaintext",
                (1, 1),
                &plaintext_values,
            )),
        };

        let vector_values_two = (0..3)
            .map(|slot| matrix(&parameters, 1, artifact.gadget_columns(), 2200 + slot * 20))
            .collect::<Vec<_>>();
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
        runtime_inputs.insert(
            "input_public_key_two".to_owned(),
            RuntimeValue::matrix(matrix(&parameters, 1, artifact.gadget_columns(), 2400)),
        );
        let input_two = BggPolyEncodingWire {
            vectors: input_family(
                &ring,
                &mut runtime_inputs,
                "vector_two",
                (1, artifact.gadget_columns()),
                &vector_values_two,
            ),
            pubkey: BggPublicKeyWire {
                matrix: ring.input("input_public_key_two", (1, artifact.gadget_columns())),
                reveal_plaintext: true,
            },
            plaintexts: Some(input_family(
                &ring,
                &mut runtime_inputs,
                "plaintext_two",
                (1, 1),
                &plaintext_values_two,
            )),
        };

        let slot_public_key_values = (0..artifact.slot_count)
            .map(|slot| matrix(&parameters, 1, artifact.gadget_columns(), 300 + slot * 20))
            .collect::<Vec<_>>();
        let slot_public_keys = input_family(
            &ring,
            &mut runtime_inputs,
            "slot_public_key",
            (1, artifact.gadget_columns()),
            &slot_public_key_values,
        );
        let mut b0_values = Vec::new();
        let b0_preimage_chunks = artifact
            .chunks(artifact.b1_public_columns())
            .into_iter()
            .enumerate()
            .map(|(chunk, range)| {
                let width = range_width(&range);
                let values = (0..artifact.slot_count)
                    .map(|slot| {
                        matrix(
                            &parameters,
                            artifact.b0_public_columns(),
                            width,
                            400 + chunk * 100 + slot * 10,
                        )
                    })
                    .collect::<Vec<_>>();
                let family = input_family(
                    &ring,
                    &mut runtime_inputs,
                    &format!("b0_chunk_{chunk}"),
                    (artifact.b0_public_columns(), width),
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
            .map(|(chunk, range)| {
                let width = range_width(&range);
                let values = (0..artifact.slot_count)
                    .map(|slot| {
                        matrix(
                            &parameters,
                            artifact.b1_public_columns(),
                            width,
                            800 + chunk * 100 + slot * 10,
                        )
                    })
                    .collect::<Vec<_>>();
                let family = input_family(
                    &ring,
                    &mut runtime_inputs,
                    &format!("b1_chunk_{chunk}"),
                    (artifact.b1_public_columns(), width),
                    &values,
                );
                b1_values.push(values);
                family
            })
            .collect::<Vec<_>>();
        let mut gate_values = Vec::new();
        let mut reduce_gate_values = Vec::new();
        let mut gate_preimage_chunks = BTreeMap::new();
        for (chunk, range) in artifact.chunks(artifact.gadget_columns()).into_iter().enumerate() {
            let width = range_width(&range);
            let values = (0..2)
                .map(|destination| {
                    matrix(
                        &parameters,
                        artifact.b0_public_columns(),
                        width,
                        1200 + chunk * 100 + destination * 10,
                    )
                })
                .collect::<Vec<_>>();
            let family = input_family(
                &ring,
                &mut runtime_inputs,
                &format!("gate_chunk_{chunk}"),
                (artifact.b0_public_columns(), width),
                &values,
            );
            gate_values.push(values);
            gate_preimage_chunks
                .insert(gate_preimage_name(false, &transfer_identity, chunk), family);

            let values = (0..2)
                .map(|destination| {
                    matrix(
                        &parameters,
                        artifact.b0_public_columns(),
                        width,
                        1600 + chunk * 100 + destination * 10,
                    )
                })
                .collect::<Vec<_>>();
            let family = input_family(
                &ring,
                &mut runtime_inputs,
                &format!("reduce_gate_chunk_{chunk}"),
                (artifact.b0_public_columns(), width),
                &values,
            );
            reduce_gate_values.push(values);
            gate_preimage_chunks.insert(gate_preimage_name(true, &reduce_identity, chunk), family);
        }

        let public_key_compiler = BggPublicKeyCompiler {
            ring: ring.clone(),
            base: artifact.gadget_base.clone(),
            digit_count: artifact.digit_count.into(),
        };
        let circuit_compiler = PolyCircuitCompiler { public_key: public_key_compiler.clone() };
        let mut lowering = BggPolySlotTransferLowering {
            compiler: BggPolyEncodingCompiler { public_key: public_key_compiler },
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
        let mut lookup = NoPublicLookup::default();
        let outputs = circuit_compiler
            .compile_poly_encodings_with_lowerings(
                &circuit,
                input.clone(),
                [input, input_two],
                &mut lookup,
                &mut lowering,
            )
            .expect("slot-transfer lowering");
        let (output, reduced_output) = (&outputs[0], &outputs[1]);
        let mut context = DslContext::new("poly-slot-transfer-online-runtime");
        for destination in 0..2 {
            context = context
                .output(format!("vector_{destination}"), output.vectors.get_static(destination))
                .expect("vector output")
                .output(
                    format!("plaintext_{destination}"),
                    output.plaintexts.as_ref().expect("plaintexts").get_static(destination),
                )
                .expect("plaintext output")
                .output(
                    format!("reduced_vector_{destination}"),
                    reduced_output.vectors.get_static(destination),
                )
                .expect("reduced vector output")
                .output(
                    format!("reduced_plaintext_{destination}"),
                    reduced_output
                        .plaintexts
                        .as_ref()
                        .expect("reduced plaintexts")
                        .get_static(destination),
                )
                .expect("reduced plaintext output");
        }
        let result = execute_graph(
            context.build().expect("runtime graph"),
            parameters.clone(),
            runtime_inputs,
        );

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
            let expected_plaintext =
                DCRTPolyMatrix::from_poly_vec(&parameters, vec![vec![plaintext * &scalar]]);
            assert_eq!(
                matrix_output(&result, &format!("vector_{destination}")),
                &(c_gate + pre_output)
            );
            assert_eq!(
                matrix_output(&result, &format!("plaintext_{destination}")),
                &expected_plaintext
            );
        }

        for destination in 0..2 {
            let decomposed = slot_public_key_values[destination].decompose();
            let selected_vectors =
                if destination == 0 { &vector_values } else { &vector_values_two };
            let plaintext_offset = if destination == 0 { 5 } else { 15 };
            let mut expected_vector =
                DCRTPolyMatrix::zero(&parameters, 1, artifact.gadget_columns());
            let mut expected_plaintext = DCRTPolyMatrix::zero(&parameters, 1, 1);
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
                expected_vector = expected_vector +
                    &((selected_vectors[source].clone() * &decomposed +
                        &(c_transfer * &plaintext)) *
                        &rotation);
                expected_plaintext = expected_plaintext +
                    &DCRTPolyMatrix::from_poly_vec(
                        &parameters,
                        vec![vec![plaintext * &rotation]],
                    );
            }
            let c_gate = concat_columns(
                reduce_gate_values
                    .iter()
                    .map(|chunk| c_b0_value.clone() * &chunk[destination])
                    .collect(),
            );
            assert_eq!(
                matrix_output(&result, &format!("reduced_vector_{destination}")),
                &(c_gate + expected_vector)
            );
            assert_eq!(
                matrix_output(&result, &format!("reduced_plaintext_{destination}")),
                &expected_plaintext
            );
        }
    }

    #[test]
    fn polynomial_slot_transfer_lowering_builds_and_symbolically_elaborates() {
        let artifact = BggSlotTransferArtifactCompiler {
            modulus: 257.into(),
            ring_dimension: 8.into(),
            secret_size: 1,
            slot_count: 2,
            digit_count: 2,
            chunk_columns: 2,
            gadget_base: 4.into(),
            trapdoor_sigma: RealExpr::from_integer(5),
            error_sigma: RealExpr::from_integer(3),
        };
        let ring = Ring::new(257, 8);
        let hash_key = ring.bytes_input("hash-key", 32);
        let base: BggSlotTransferBaseWires = artifact.build_base().expect("base artifacts");
        let slots = artifact.build_slots(hash_key.clone(), &base).expect("slot artifacts");

        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input_gate = circuit.input(1).as_single_wire();
        let transferred = circuit.slot_transfer_gate(input_gate, &[(1, None), (0, Some(2))]);
        circuit.output([transferred]);

        let public_key_type = ring.matrix_type((1, 2));
        let one_public =
            BggPublicKeyWire { matrix: ring.input("one-public", (1, 2)), reveal_plaintext: true };
        let input_public =
            BggPublicKeyWire { matrix: ring.input("input-public", (1, 2)), reveal_plaintext: true };
        let public_compiler =
            BggPublicKeyCompiler { ring: ring.clone(), base: 4.into(), digit_count: 2.into() };
        let mut public_lowering = BggSlotTransferPublicKeyLowering {
            compiler: public_compiler.clone(),
            hash_key: hash_key.clone(),
            public_key_type,
            configured_slot_count: 2,
            requests: Vec::new(),
        };
        let circuit_compiler = PolyCircuitCompiler { public_key: public_compiler.clone() };
        let mut public_lookup = NoPublicLookup::default();
        circuit_compiler
            .compile_public_keys_with_lowerings(
                &circuit,
                one_public.clone(),
                [input_public.clone()],
                &mut public_lookup,
                &mut public_lowering,
            )
            .expect("public-key pass");
        let gates = artifact
            .build_gate_preimages(&base, &slots, &public_lowering.requests)
            .expect("gate artifacts");

        let encoding = BggPolyEncodingWire {
            vectors: ring.input_family("vectors", 2, (1, 2)),
            pubkey: input_public,
            plaintexts: Some(ring.input_family("plaintexts", 2, (1, 1))),
        };
        let one_encoding = BggPolyEncodingWire {
            vectors: ring.input_family("one-vectors", 2, (1, 2)),
            pubkey: one_public,
            plaintexts: Some(ring.input_family("one-plaintexts", 2, (1, 1))),
        };
        let mut lowering = BggPolySlotTransferLowering {
            compiler: BggPolyEncodingCompiler { public_key: public_compiler },
            artifact,
            hash_key,
            c_b0: ring.input("c-b0", (1, 4)),
            slots: BggSlotTransferPublicSlotWires {
                public_keys: slots.public_keys,
                b0_preimage_chunks: slots.b0_preimage_chunks,
                b1_preimage_chunks: slots.b1_preimage_chunks,
            },
            gates,
        };
        let mut encoding_lookup = NoPublicLookup::default();
        let outputs = circuit_compiler
            .compile_poly_encodings_with_lowerings(
                &circuit,
                one_encoding,
                [encoding],
                &mut encoding_lookup,
                &mut lowering,
            )
            .expect("encoding pass");
        let built = DslContext::new("slot-transfer-poly-encoding")
            .family_output("vectors", outputs[0].vectors.clone())
            .expect("vectors")
            .output("public", outputs[0].pubkey.matrix.clone())
            .expect("public")
            .build()
            .expect("build");
        let elaborated = built.elaborate(&ParamEnv::default()).expect("symbolic elaboration");
        assert!(elaborated.wire(&elaborated.outputs["vectors"]).unwrap().family.is_some());
        assert!(elaborated.wire(&elaborated.outputs["public"]).unwrap().expression.is_some());
    }
}
