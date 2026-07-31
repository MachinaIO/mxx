use crate::{BggPublicKeyCompiler, BggPublicKeyWire};
use mxx_ir_core::{
    GraphBuilder, MatrixFamilyWire, MatrixWire, SubgraphBuildError,
    node::{LoopInputMode, MatrixBinaryOp},
};
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BggPolyEncodingWire {
    pub vectors: MatrixFamilyWire,
    pub pubkey: BggPublicKeyWire,
    pub plaintexts: Option<MatrixFamilyWire>,
}

#[derive(Clone, Debug)]
pub struct BggPolyEncodingCompiler {
    pub public_key: BggPublicKeyCompiler,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum PolyEncodingCompileError {
    #[error("BGG+ poly-encoding families must have matching slot counts")]
    SlotCountMismatch,
    #[error("BGG+ poly-encoding multiplication requires the left plaintext family")]
    MissingLeftPlaintext,
    #[error(transparent)]
    Subgraph(#[from] SubgraphBuildError),
}

impl BggPolyEncodingCompiler {
    pub fn add(
        &self,
        builder: &mut GraphBuilder,
        lhs: &BggPolyEncodingWire,
        rhs: &BggPolyEncodingWire,
    ) -> Result<BggPolyEncodingWire, PolyEncodingCompileError> {
        self.componentwise_binary(builder, lhs, rhs, MatrixBinaryOp::Add)
    }

    pub fn sub(
        &self,
        builder: &mut GraphBuilder,
        lhs: &BggPolyEncodingWire,
        rhs: &BggPolyEncodingWire,
    ) -> Result<BggPolyEncodingWire, PolyEncodingCompileError> {
        self.componentwise_binary(builder, lhs, rhs, MatrixBinaryOp::Subtract)
    }

    pub fn mul(
        &self,
        builder: &mut GraphBuilder,
        lhs: &BggPolyEncodingWire,
        rhs: &BggPolyEncodingWire,
    ) -> Result<BggPolyEncodingWire, PolyEncodingCompileError> {
        validate_bundle(lhs)?;
        validate_bundle(rhs)?;
        validate_pair(lhs, rhs)?;
        let Some(lhs_plaintexts) = &lhs.plaintexts else {
            return Err(PolyEncodingCompileError::MissingLeftPlaintext);
        };
        let decomposed_rhs = builder.gadget_decompose(
            &rhs.pubkey.matrix,
            self.public_key.base.clone(),
            self.public_key.decomposed_type.clone(),
        );
        let mut body = GraphBuilder::new(
            format!(
                "bgg-poly-encoding-mul-{}-{}",
                reveal_name(lhs.plaintexts.is_some()),
                reveal_name(rhs.plaintexts.is_some())
            ),
            Vec::new(),
        );
        let lhs_vector = body.input("0_lhs_vector", lhs.vectors.matrix_type.clone());
        let rhs_vector = body.input("1_rhs_vector", rhs.vectors.matrix_type.clone());
        let lhs_plaintext = body.input("2_lhs_plaintext", lhs_plaintexts.matrix_type.clone());
        let decomposed_rhs_input =
            body.preimage_input("3_decomposed_rhs", decomposed_rhs.matrix_type.clone());
        let first = body.matrix_binary(
            MatrixBinaryOp::Multiply,
            &lhs_vector,
            &decomposed_rhs_input,
            lhs.vectors.matrix_type.clone(),
        );
        let second = body.matrix_binary(
            MatrixBinaryOp::Multiply,
            &rhs_vector,
            &lhs_plaintext,
            rhs.vectors.matrix_type.clone(),
        );
        let vector = body.matrix_binary(
            MatrixBinaryOp::Add,
            &first,
            &second,
            lhs.vectors.matrix_type.clone(),
        );
        body.value_output_wire("0_vector", vector.wire);

        let mut args =
            vec![lhs.vectors.wire, rhs.vectors.wire, lhs_plaintexts.wire, decomposed_rhs.wire];
        let mut input_modes = vec![
            LoopInputMode::Zip,
            LoopInputMode::Zip,
            LoopInputMode::Zip,
            LoopInputMode::Broadcast,
        ];
        let mut output_types = vec![lhs.vectors.matrix_type.clone()];
        if let Some(rhs_plaintexts) = &rhs.plaintexts {
            let rhs_plaintext = body.input("4_rhs_plaintext", rhs_plaintexts.matrix_type.clone());
            let plaintext = body.matrix_binary(
                MatrixBinaryOp::Multiply,
                &lhs_plaintext,
                &rhs_plaintext,
                lhs_plaintexts.matrix_type.clone(),
            );
            body.value_output_wire("1_plaintext", plaintext.wire);
            args.push(rhs_plaintexts.wire);
            input_modes.push(LoopInputMode::Zip);
            output_types.push(lhs_plaintexts.matrix_type.clone());
        }
        let mut outputs = builder.parallel_loop(
            body.finish(),
            lhs.vectors.count.clone(),
            "slot",
            Vec::new(),
            args,
            input_modes,
            &output_types,
        )?;
        let vectors = outputs.remove(0);
        let plaintexts = (!outputs.is_empty()).then(|| outputs.remove(0));
        Ok(BggPolyEncodingWire {
            vectors,
            pubkey: self.public_key.mul(builder, &lhs.pubkey, &rhs.pubkey),
            plaintexts,
        })
    }

    pub fn small_scalar_mul(
        &self,
        builder: &mut GraphBuilder,
        input: &BggPolyEncodingWire,
        scalar: &MatrixWire,
    ) -> Result<BggPolyEncodingWire, PolyEncodingCompileError> {
        validate_bundle(input)?;
        self.scalar_mul(builder, input, scalar, None)
    }

    pub fn large_scalar_mul(
        &self,
        builder: &mut GraphBuilder,
        input: &BggPolyEncodingWire,
        scalar: &MatrixWire,
    ) -> Result<BggPolyEncodingWire, PolyEncodingCompileError> {
        validate_bundle(input)?;
        let decomposed = self.public_key.large_scalar_decomposition(builder, &input.pubkey, scalar);
        self.scalar_mul(builder, input, scalar, Some(&decomposed))
    }

    fn componentwise_binary(
        &self,
        builder: &mut GraphBuilder,
        lhs: &BggPolyEncodingWire,
        rhs: &BggPolyEncodingWire,
        operation: MatrixBinaryOp,
    ) -> Result<BggPolyEncodingWire, PolyEncodingCompileError> {
        validate_bundle(lhs)?;
        validate_bundle(rhs)?;
        validate_pair(lhs, rhs)?;
        let operation_name = match operation {
            MatrixBinaryOp::Add => "add",
            MatrixBinaryOp::Subtract => "sub",
            MatrixBinaryOp::Multiply => unreachable!("multiplication has a distinct formula"),
        };
        let mut body = GraphBuilder::new(
            format!(
                "bgg-poly-encoding-{operation_name}-{}-{}",
                reveal_name(lhs.plaintexts.is_some()),
                reveal_name(rhs.plaintexts.is_some())
            ),
            Vec::new(),
        );
        let lhs_vector = body.input("0_lhs_vector", lhs.vectors.matrix_type.clone());
        let rhs_vector = body.input("1_rhs_vector", rhs.vectors.matrix_type.clone());
        let vector = body.matrix_binary(
            operation,
            &lhs_vector,
            &rhs_vector,
            lhs.vectors.matrix_type.clone(),
        );
        body.value_output_wire("0_vector", vector.wire);
        let mut args = vec![lhs.vectors.wire, rhs.vectors.wire];
        let mut input_modes = vec![LoopInputMode::Zip, LoopInputMode::Zip];
        let mut output_types = vec![lhs.vectors.matrix_type.clone()];
        if let (Some(lhs_plaintexts), Some(rhs_plaintexts)) = (&lhs.plaintexts, &rhs.plaintexts) {
            let lhs_plaintext = body.input("2_lhs_plaintext", lhs_plaintexts.matrix_type.clone());
            let rhs_plaintext = body.input("3_rhs_plaintext", rhs_plaintexts.matrix_type.clone());
            let plaintext = body.matrix_binary(
                operation,
                &lhs_plaintext,
                &rhs_plaintext,
                lhs_plaintexts.matrix_type.clone(),
            );
            body.value_output_wire("1_plaintext", plaintext.wire);
            args.extend([lhs_plaintexts.wire, rhs_plaintexts.wire]);
            input_modes.extend([LoopInputMode::Zip, LoopInputMode::Zip]);
            output_types.push(lhs_plaintexts.matrix_type.clone());
        }
        let mut outputs = builder.parallel_loop(
            body.finish(),
            lhs.vectors.count.clone(),
            "slot",
            Vec::new(),
            args,
            input_modes,
            &output_types,
        )?;
        let vectors = outputs.remove(0);
        let plaintexts = (!outputs.is_empty()).then(|| outputs.remove(0));
        let pubkey = match operation {
            MatrixBinaryOp::Add => self.public_key.add(builder, &lhs.pubkey, &rhs.pubkey),
            MatrixBinaryOp::Subtract => self.public_key.sub(builder, &lhs.pubkey, &rhs.pubkey),
            MatrixBinaryOp::Multiply => unreachable!("multiplication has a distinct formula"),
        };
        Ok(BggPolyEncodingWire { vectors, pubkey, plaintexts })
    }

    fn scalar_mul(
        &self,
        builder: &mut GraphBuilder,
        input: &BggPolyEncodingWire,
        scalar: &MatrixWire,
        decomposed_scalar: Option<&MatrixWire>,
    ) -> Result<BggPolyEncodingWire, PolyEncodingCompileError> {
        let operation_name = if decomposed_scalar.is_some() { "large" } else { "small" };
        let mut body = GraphBuilder::new(
            format!(
                "bgg-poly-encoding-{operation_name}-scalar-{}",
                reveal_name(input.plaintexts.is_some())
            ),
            Vec::new(),
        );
        let vector = body.input("0_vector", input.vectors.matrix_type.clone());
        let vector_factor_type = decomposed_scalar
            .map(|decomposed| decomposed.matrix_type.clone())
            .unwrap_or_else(|| scalar.matrix_type.clone());
        let vector_factor = if decomposed_scalar.is_some() {
            body.preimage_input("1_vector_factor", vector_factor_type)
        } else {
            body.input("1_vector_factor", vector_factor_type)
        };
        let output_vector = body.matrix_binary(
            MatrixBinaryOp::Multiply,
            &vector,
            &vector_factor,
            input.vectors.matrix_type.clone(),
        );
        body.value_output_wire("0_vector", output_vector.wire);
        let mut args = vec![input.vectors.wire, decomposed_scalar.unwrap_or(scalar).wire];
        let mut input_modes = vec![LoopInputMode::Zip, LoopInputMode::Broadcast];
        let mut output_types = vec![input.vectors.matrix_type.clone()];
        if let Some(plaintexts) = &input.plaintexts {
            let plaintext = body.input("2_plaintext", plaintexts.matrix_type.clone());
            let plaintext_scalar = body.input("3_plaintext_scalar", scalar.matrix_type.clone());
            let output_plaintext = body.matrix_binary(
                MatrixBinaryOp::Multiply,
                &plaintext,
                &plaintext_scalar,
                plaintexts.matrix_type.clone(),
            );
            body.value_output_wire("1_plaintext", output_plaintext.wire);
            args.extend([plaintexts.wire, scalar.wire]);
            input_modes.extend([LoopInputMode::Zip, LoopInputMode::Broadcast]);
            output_types.push(plaintexts.matrix_type.clone());
        }
        let mut outputs = builder.parallel_loop(
            body.finish(),
            input.vectors.count.clone(),
            "slot",
            Vec::new(),
            args,
            input_modes,
            &output_types,
        )?;
        let vectors = outputs.remove(0);
        let plaintexts = (!outputs.is_empty()).then(|| outputs.remove(0));
        let pubkey = match decomposed_scalar {
            Some(decomposed) => self.public_key.large_scalar_mul_with_decomposition(
                builder,
                &input.pubkey,
                decomposed,
            ),
            None => self.public_key.small_scalar_mul(builder, &input.pubkey, scalar),
        };
        Ok(BggPolyEncodingWire { vectors, pubkey, plaintexts })
    }
}

fn validate_bundle(input: &BggPolyEncodingWire) -> Result<(), PolyEncodingCompileError> {
    if input.plaintexts.as_ref().is_some_and(|plaintexts| plaintexts.count != input.vectors.count) {
        return Err(PolyEncodingCompileError::SlotCountMismatch);
    }
    Ok(())
}

fn validate_pair(
    lhs: &BggPolyEncodingWire,
    rhs: &BggPolyEncodingWire,
) -> Result<(), PolyEncodingCompileError> {
    if lhs.vectors.count != rhs.vectors.count {
        return Err(PolyEncodingCompileError::SlotCountMismatch);
    }
    Ok(())
}

fn reveal_name(revealed: bool) -> &'static str {
    if revealed { "revealed" } else { "hidden" }
}

#[cfg(test)]
mod graph_tests {
    use super::*;
    use mxx_ir_core::{IntExpr, ParamEnv, artifact::ArtifactConfidentiality, types::MatrixType};
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use mxx_runtime::{
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
        transcript::SamplingMode,
    };
    use num_bigint::BigInt;
    use std::collections::BTreeMap;

    fn matrix_type(parameters: &DCRTPolyParams, rows: usize, columns: usize) -> MatrixType {
        MatrixType {
            modulus: IntExpr::constant(BigInt::from(parameters.modulus().as_ref().clone())),
            ring_dimension: IntExpr::constant(parameters.ring_dimension()),
            rows: IntExpr::constant(rows),
            columns: IntExpr::constant(columns),
        }
    }

    fn row(parameters: &DCRTPolyParams, columns: usize, offset: usize) -> DCRTPolyMatrix {
        DCRTPolyMatrix::from_poly_vec_row(
            parameters,
            (0..columns)
                .map(|index| {
                    DCRTPoly::const_rotate_poly(
                        parameters,
                        (index + offset) % parameters.ring_dimension() as usize,
                    )
                })
                .collect(),
        )
    }

    #[test]
    fn add_uses_zip_families_and_matches_primitive_matrix_addition() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let columns = parameters.modulus_digits();
        let vector_type = matrix_type(&parameters, 1, columns);
        let public_key_type = matrix_type(&parameters, 2, columns);
        let plaintext_type = matrix_type(&parameters, 1, 1);
        let mut builder = GraphBuilder::new("poly-add-runtime", Vec::new());
        let lhs_vectors = (0..2)
            .map(|slot| builder.input(format!("lhs_vector_{slot}"), vector_type.clone()))
            .collect::<Vec<_>>();
        let rhs_vectors = (0..2)
            .map(|slot| builder.input(format!("rhs_vector_{slot}"), vector_type.clone()))
            .collect::<Vec<_>>();
        let lhs_plaintexts = (0..2)
            .map(|slot| builder.input(format!("lhs_plaintext_{slot}"), plaintext_type.clone()))
            .collect::<Vec<_>>();
        let rhs_plaintexts = (0..2)
            .map(|slot| builder.input(format!("rhs_plaintext_{slot}"), plaintext_type.clone()))
            .collect::<Vec<_>>();
        let lhs = BggPolyEncodingWire {
            vectors: builder.family_pack(&lhs_vectors).expect("lhs vectors"),
            pubkey: BggPublicKeyWire {
                matrix: builder.input("lhs_pubkey", public_key_type.clone()),
                reveal_plaintext: true,
            },
            plaintexts: Some(builder.family_pack(&lhs_plaintexts).expect("lhs plaintexts")),
        };
        let rhs = BggPolyEncodingWire {
            vectors: builder.family_pack(&rhs_vectors).expect("rhs vectors"),
            pubkey: BggPublicKeyWire {
                matrix: builder.input("rhs_pubkey", public_key_type.clone()),
                reveal_plaintext: true,
            },
            plaintexts: Some(builder.family_pack(&rhs_plaintexts).expect("rhs plaintexts")),
        };
        let compiler = BggPolyEncodingCompiler {
            public_key: BggPublicKeyCompiler {
                base: IntExpr::constant(1u64 << parameters.base_bits()),
                decomposed_type: matrix_type(&parameters, columns, columns),
            },
        };
        let output = compiler.add(&mut builder, &lhs, &rhs).expect("compatible families");
        for slot in 0..2 {
            let vector = builder.family_get_static(&output.vectors, IntExpr::constant(slot));
            let plaintext = builder.family_get_static(
                output.plaintexts.as_ref().expect("revealed plaintexts"),
                IntExpr::constant(slot),
            );
            builder.output(format!("vector_{slot}"), &vector, ArtifactConfidentiality::Public);
            builder.output(
                format!("plaintext_{slot}"),
                &plaintext,
                ArtifactConfidentiality::Public,
            );
        }
        builder.output("pubkey", &output.pubkey.matrix, ArtifactConfidentiality::Public);
        let validated = mxx_ir_core::validate(&builder.finish(), &ParamEnv::default())
            .expect("valid poly-encoding graph");

        let lhs_vector_values = [row(&parameters, columns, 0), row(&parameters, columns, 1)];
        let rhs_vector_values = [row(&parameters, columns, 2), row(&parameters, columns, 3)];
        let lhs_plaintext_values = [row(&parameters, 1, 1), row(&parameters, 1, 2)];
        let rhs_plaintext_values = [row(&parameters, 1, 3), row(&parameters, 1, 4)];
        let lhs_pubkey = DCRTPolyMatrix::from_poly_vec(
            &parameters,
            vec![row(&parameters, columns, 4).get_row(0), row(&parameters, columns, 5).get_row(0)],
        );
        let rhs_pubkey = DCRTPolyMatrix::from_poly_vec(
            &parameters,
            vec![row(&parameters, columns, 6).get_row(0), row(&parameters, columns, 7).get_row(0)],
        );
        let mut inputs = BTreeMap::from([
            ("lhs_pubkey".to_owned(), RuntimeValue::matrix(lhs_pubkey.clone())),
            ("rhs_pubkey".to_owned(), RuntimeValue::matrix(rhs_pubkey.clone())),
        ]);
        for slot in 0..2 {
            inputs.insert(
                format!("lhs_vector_{slot}"),
                RuntimeValue::matrix(lhs_vector_values[slot].clone()),
            );
            inputs.insert(
                format!("rhs_vector_{slot}"),
                RuntimeValue::matrix(rhs_vector_values[slot].clone()),
            );
            inputs.insert(
                format!("lhs_plaintext_{slot}"),
                RuntimeValue::matrix(lhs_plaintext_values[slot].clone()),
            );
            inputs.insert(
                format!("rhs_plaintext_{slot}"),
                RuntimeValue::matrix(rhs_plaintext_values[slot].clone()),
            );
        }
        let mut backend = cpu_backend([parameters]);
        let mut store = MemoryArtifactStore::default();
        let result = execute(&validated, &mut backend, inputs, &mut store, SamplingMode::Fresh)
            .expect("poly-encoding graph execution");
        for slot in 0..2 {
            let RuntimeValue::Matrix(vector) = &result.outputs[&format!("vector_{slot}")] else {
                panic!("vector output");
            };
            let RuntimeValue::Matrix(plaintext) = &result.outputs[&format!("plaintext_{slot}")]
            else {
                panic!("plaintext output");
            };
            assert_eq!(
                vector.as_ref(),
                &(lhs_vector_values[slot].clone() + rhs_vector_values[slot].clone())
            );
            assert_eq!(
                plaintext.as_ref(),
                &(lhs_plaintext_values[slot].clone() + rhs_plaintext_values[slot].clone())
            );
        }
        let RuntimeValue::Matrix(pubkey) = &result.outputs["pubkey"] else {
            panic!("public-key output");
        };
        assert_eq!(pubkey.as_ref(), &(lhs_pubkey + rhs_pubkey));
    }

    #[test]
    fn reveal_combinations_match_the_direct_poly_encoding_contract() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let columns = parameters.modulus_digits();
        let vector_type = matrix_type(&parameters, 1, columns);
        let public_key_type = matrix_type(&parameters, 2, columns);
        let plaintext_type = matrix_type(&parameters, 1, 1);
        let compiler = BggPolyEncodingCompiler {
            public_key: BggPublicKeyCompiler {
                base: IntExpr::constant(1u64 << parameters.base_bits()),
                decomposed_type: matrix_type(&parameters, columns, columns),
            },
        };
        for lhs_revealed in [false, true] {
            for rhs_revealed in [false, true] {
                let mut builder = GraphBuilder::new("poly-encoding-reveal-contract", Vec::new());
                let lhs_vector = builder.input("lhs_vector", vector_type.clone());
                let lhs_plaintext =
                    lhs_revealed.then(|| builder.input("lhs_plaintext", plaintext_type.clone()));
                let lhs = BggPolyEncodingWire {
                    vectors: builder.family_pack(&[lhs_vector]).expect("lhs vector family"),
                    pubkey: BggPublicKeyWire {
                        matrix: builder.input("lhs_pubkey", public_key_type.clone()),
                        reveal_plaintext: lhs_revealed,
                    },
                    plaintexts: lhs_plaintext
                        .map(|value| builder.family_pack(&[value]).expect("lhs plaintext family")),
                };
                let rhs_vector = builder.input("rhs_vector", vector_type.clone());
                let rhs_plaintext =
                    rhs_revealed.then(|| builder.input("rhs_plaintext", plaintext_type.clone()));
                let rhs = BggPolyEncodingWire {
                    vectors: builder.family_pack(&[rhs_vector]).expect("rhs vector family"),
                    pubkey: BggPublicKeyWire {
                        matrix: builder.input("rhs_pubkey", public_key_type.clone()),
                        reveal_plaintext: rhs_revealed,
                    },
                    plaintexts: rhs_plaintext
                        .map(|value| builder.family_pack(&[value]).expect("rhs plaintext family")),
                };
                let expected_reveal = lhs_revealed && rhs_revealed;
                for output in [
                    compiler.add(&mut builder, &lhs, &rhs).expect("addition"),
                    compiler.sub(&mut builder, &lhs, &rhs).expect("subtraction"),
                ] {
                    assert_eq!(output.pubkey.reveal_plaintext, expected_reveal);
                    assert_eq!(output.plaintexts.is_some(), expected_reveal);
                }
                match compiler.mul(&mut builder, &lhs, &rhs) {
                    Ok(output) => {
                        assert!(lhs_revealed);
                        assert_eq!(output.pubkey.reveal_plaintext, expected_reveal);
                        assert_eq!(output.plaintexts.is_some(), expected_reveal);
                    }
                    Err(error) => {
                        assert!(!lhs_revealed);
                        assert_eq!(error, PolyEncodingCompileError::MissingLeftPlaintext);
                    }
                }
            }
        }
    }
}
