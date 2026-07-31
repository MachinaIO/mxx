use crate::{
    BggEncodingCompiler, BggEncodingWire, BggPublicKeyCompiler, BggPublicKeyWire,
    encoding::EncodingCompileError,
};
use mxx_ir_core::{
    GraphBuilder, MatrixFamilyWire, MatrixWire, SubgraphBuildError,
    node::{LoopInputMode, MatrixBinaryOp},
};
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NaiveBggPublicKeyVecWire {
    pub matrices: MatrixFamilyWire,
    pub reveal_plaintext: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NaiveBggEncodingVecWire {
    pub vectors: MatrixFamilyWire,
    pub pubkeys: MatrixFamilyWire,
    pub pubkey_reveal_plaintext: bool,
    pub plaintexts: Option<MatrixFamilyWire>,
}

#[derive(Clone, Debug)]
pub struct NaiveBggVecCompiler {
    pub public_key: BggPublicKeyCompiler,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum NaiveVecCompileError {
    #[error("naive BGG+ vector families must have matching slot counts")]
    SlotCountMismatch,
    #[error(transparent)]
    Encoding(#[from] EncodingCompileError),
    #[error(transparent)]
    Subgraph(#[from] SubgraphBuildError),
}

impl NaiveBggVecCompiler {
    pub fn add_public_keys(
        &self,
        builder: &mut GraphBuilder,
        lhs: &NaiveBggPublicKeyVecWire,
        rhs: &NaiveBggPublicKeyVecWire,
    ) -> Result<NaiveBggPublicKeyVecWire, NaiveVecCompileError> {
        self.public_key_binary(builder, lhs, rhs, MatrixBinaryOp::Add)
    }

    pub fn sub_public_keys(
        &self,
        builder: &mut GraphBuilder,
        lhs: &NaiveBggPublicKeyVecWire,
        rhs: &NaiveBggPublicKeyVecWire,
    ) -> Result<NaiveBggPublicKeyVecWire, NaiveVecCompileError> {
        self.public_key_binary(builder, lhs, rhs, MatrixBinaryOp::Subtract)
    }

    pub fn mul_public_keys(
        &self,
        builder: &mut GraphBuilder,
        lhs: &NaiveBggPublicKeyVecWire,
        rhs: &NaiveBggPublicKeyVecWire,
    ) -> Result<NaiveBggPublicKeyVecWire, NaiveVecCompileError> {
        self.public_key_binary(builder, lhs, rhs, MatrixBinaryOp::Multiply)
    }

    pub fn small_scalar_mul_public_keys(
        &self,
        builder: &mut GraphBuilder,
        input: &NaiveBggPublicKeyVecWire,
        scalar: &MatrixWire,
    ) -> Result<NaiveBggPublicKeyVecWire, NaiveVecCompileError> {
        self.public_key_scalar(builder, input, scalar, false)
    }

    pub fn large_scalar_mul_public_keys(
        &self,
        builder: &mut GraphBuilder,
        input: &NaiveBggPublicKeyVecWire,
        scalar: &MatrixWire,
    ) -> Result<NaiveBggPublicKeyVecWire, NaiveVecCompileError> {
        self.public_key_scalar(builder, input, scalar, true)
    }

    pub fn matrix_mul_public_keys(
        &self,
        builder: &mut GraphBuilder,
        input: &NaiveBggPublicKeyVecWire,
        target: &MatrixWire,
    ) -> Result<NaiveBggPublicKeyVecWire, NaiveVecCompileError> {
        let mut body = GraphBuilder::new("naive-bgg-public-key-matrix-mul", Vec::new());
        let body_input = BggPublicKeyWire {
            matrix: body.input("0_input", input.matrices.matrix_type.clone()),
            reveal_plaintext: input.reveal_plaintext,
        };
        let body_target = body.input("1_target", target.matrix_type.clone());
        let output = self.public_key.matrix_mul(&mut body, &body_input, &body_target);
        body.value_output_wire("0_matrix", output.matrix.wire);
        let mut outputs = builder.parallel_loop(
            body.finish(),
            input.matrices.count.clone(),
            "slot",
            Vec::new(),
            vec![input.matrices.wire, target.wire],
            vec![LoopInputMode::Zip, LoopInputMode::Broadcast],
            std::slice::from_ref(&output.matrix.matrix_type),
        )?;
        Ok(NaiveBggPublicKeyVecWire {
            matrices: outputs.remove(0),
            reveal_plaintext: output.reveal_plaintext,
        })
    }

    pub fn add_encodings(
        &self,
        builder: &mut GraphBuilder,
        lhs: &NaiveBggEncodingVecWire,
        rhs: &NaiveBggEncodingVecWire,
    ) -> Result<NaiveBggEncodingVecWire, NaiveVecCompileError> {
        self.encoding_binary(builder, lhs, rhs, MatrixBinaryOp::Add)
    }

    pub fn sub_encodings(
        &self,
        builder: &mut GraphBuilder,
        lhs: &NaiveBggEncodingVecWire,
        rhs: &NaiveBggEncodingVecWire,
    ) -> Result<NaiveBggEncodingVecWire, NaiveVecCompileError> {
        self.encoding_binary(builder, lhs, rhs, MatrixBinaryOp::Subtract)
    }

    pub fn mul_encodings(
        &self,
        builder: &mut GraphBuilder,
        lhs: &NaiveBggEncodingVecWire,
        rhs: &NaiveBggEncodingVecWire,
    ) -> Result<NaiveBggEncodingVecWire, NaiveVecCompileError> {
        self.encoding_binary(builder, lhs, rhs, MatrixBinaryOp::Multiply)
    }

    pub fn small_scalar_mul_encodings(
        &self,
        builder: &mut GraphBuilder,
        input: &NaiveBggEncodingVecWire,
        scalar: &MatrixWire,
    ) -> Result<NaiveBggEncodingVecWire, NaiveVecCompileError> {
        self.encoding_scalar(builder, input, scalar, false)
    }

    pub fn large_scalar_mul_encodings(
        &self,
        builder: &mut GraphBuilder,
        input: &NaiveBggEncodingVecWire,
        scalar: &MatrixWire,
    ) -> Result<NaiveBggEncodingVecWire, NaiveVecCompileError> {
        self.encoding_scalar(builder, input, scalar, true)
    }

    pub fn matrix_mul_encodings(
        &self,
        builder: &mut GraphBuilder,
        input: &NaiveBggEncodingVecWire,
        target: &MatrixWire,
    ) -> Result<NaiveBggEncodingVecWire, NaiveVecCompileError> {
        validate_encoding(input)?;
        let compiler = BggEncodingCompiler { public_key: self.public_key.clone() };
        let mut body = GraphBuilder::new("naive-bgg-encoding-matrix-mul", Vec::new());
        let body_input = encoding_body_input(&mut body, "input", input);
        let body_target = body.input("3_target", target.matrix_type.clone());
        let output = compiler.matrix_mul(&mut body, &body_input, &body_target);
        body.value_output_wire("0_vector", output.vector.wire);
        body.value_output_wire("1_pubkey", output.pubkey.matrix.wire);
        let mut args = vec![input.vectors.wire, input.pubkeys.wire];
        let mut modes = vec![LoopInputMode::Zip, LoopInputMode::Zip];
        if let Some(plaintexts) = &input.plaintexts {
            args.push(plaintexts.wire);
            modes.push(LoopInputMode::Zip);
        }
        args.push(target.wire);
        modes.push(LoopInputMode::Broadcast);
        let mut outputs = builder.parallel_loop(
            body.finish(),
            input.vectors.count.clone(),
            "slot",
            Vec::new(),
            args,
            modes,
            &[output.vector.matrix_type.clone(), output.pubkey.matrix.matrix_type.clone()],
        )?;
        Ok(NaiveBggEncodingVecWire {
            vectors: outputs.remove(0),
            pubkeys: outputs.remove(0),
            pubkey_reveal_plaintext: output.pubkey.reveal_plaintext,
            plaintexts: None,
        })
    }

    fn public_key_binary(
        &self,
        builder: &mut GraphBuilder,
        lhs: &NaiveBggPublicKeyVecWire,
        rhs: &NaiveBggPublicKeyVecWire,
        operation: MatrixBinaryOp,
    ) -> Result<NaiveBggPublicKeyVecWire, NaiveVecCompileError> {
        validate_family_pair(&lhs.matrices, &rhs.matrices)?;
        let operation_name = operation_name(operation);
        let mut body = GraphBuilder::new(
            format!(
                "naive-bgg-public-key-{operation_name}-{}-{}",
                reveal_name(lhs.reveal_plaintext),
                reveal_name(rhs.reveal_plaintext)
            ),
            Vec::new(),
        );
        let lhs_input = BggPublicKeyWire {
            matrix: body.input("0_lhs", lhs.matrices.matrix_type.clone()),
            reveal_plaintext: lhs.reveal_plaintext,
        };
        let rhs_input = BggPublicKeyWire {
            matrix: body.input("1_rhs", rhs.matrices.matrix_type.clone()),
            reveal_plaintext: rhs.reveal_plaintext,
        };
        let output = match operation {
            MatrixBinaryOp::Add => self.public_key.add(&mut body, &lhs_input, &rhs_input),
            MatrixBinaryOp::Subtract => self.public_key.sub(&mut body, &lhs_input, &rhs_input),
            MatrixBinaryOp::Multiply => self.public_key.mul(&mut body, &lhs_input, &rhs_input),
        };
        body.value_output_wire("0_matrix", output.matrix.wire);
        let mut outputs = builder.parallel_loop(
            body.finish(),
            lhs.matrices.count.clone(),
            "slot",
            Vec::new(),
            vec![lhs.matrices.wire, rhs.matrices.wire],
            vec![LoopInputMode::Zip, LoopInputMode::Zip],
            std::slice::from_ref(&output.matrix.matrix_type),
        )?;
        Ok(NaiveBggPublicKeyVecWire {
            matrices: outputs.remove(0),
            reveal_plaintext: output.reveal_plaintext,
        })
    }

    fn public_key_scalar(
        &self,
        builder: &mut GraphBuilder,
        input: &NaiveBggPublicKeyVecWire,
        scalar: &MatrixWire,
        large: bool,
    ) -> Result<NaiveBggPublicKeyVecWire, NaiveVecCompileError> {
        let mut body = GraphBuilder::new(
            format!(
                "naive-bgg-public-key-{}-scalar-{}",
                if large { "large" } else { "small" },
                reveal_name(input.reveal_plaintext)
            ),
            Vec::new(),
        );
        let body_input = BggPublicKeyWire {
            matrix: body.input("0_input", input.matrices.matrix_type.clone()),
            reveal_plaintext: input.reveal_plaintext,
        };
        let body_scalar = body.input("1_scalar", scalar.matrix_type.clone());
        let output = if large {
            self.public_key.large_scalar_mul(&mut body, &body_input, &body_scalar)
        } else {
            self.public_key.small_scalar_mul(&mut body, &body_input, &body_scalar)
        };
        body.value_output_wire("0_matrix", output.matrix.wire);
        let mut outputs = builder.parallel_loop(
            body.finish(),
            input.matrices.count.clone(),
            "slot",
            Vec::new(),
            vec![input.matrices.wire, scalar.wire],
            vec![LoopInputMode::Zip, LoopInputMode::Broadcast],
            std::slice::from_ref(&output.matrix.matrix_type),
        )?;
        Ok(NaiveBggPublicKeyVecWire {
            matrices: outputs.remove(0),
            reveal_plaintext: output.reveal_plaintext,
        })
    }

    fn encoding_binary(
        &self,
        builder: &mut GraphBuilder,
        lhs: &NaiveBggEncodingVecWire,
        rhs: &NaiveBggEncodingVecWire,
        operation: MatrixBinaryOp,
    ) -> Result<NaiveBggEncodingVecWire, NaiveVecCompileError> {
        validate_encoding(lhs)?;
        validate_encoding(rhs)?;
        validate_family_pair(&lhs.vectors, &rhs.vectors)?;
        let compiler = BggEncodingCompiler { public_key: self.public_key.clone() };
        let mut body = GraphBuilder::new(
            format!(
                "naive-bgg-encoding-{}-{}-{}",
                operation_name(operation),
                reveal_name(lhs.plaintexts.is_some()),
                reveal_name(rhs.plaintexts.is_some())
            ),
            Vec::new(),
        );
        let lhs_input = encoding_body_input(&mut body, "lhs", lhs);
        let rhs_input = encoding_body_input(&mut body, "rhs", rhs);
        let output = match operation {
            MatrixBinaryOp::Add => compiler.add(&mut body, &lhs_input, &rhs_input),
            MatrixBinaryOp::Subtract => compiler.sub(&mut body, &lhs_input, &rhs_input),
            MatrixBinaryOp::Multiply => compiler.mul(&mut body, &lhs_input, &rhs_input),
        }?;
        body.value_output_wire("0_vector", output.vector.wire);
        body.value_output_wire("1_pubkey", output.pubkey.matrix.wire);
        if let Some(plaintext) = &output.plaintext {
            body.value_output_wire("2_plaintext", plaintext.wire);
        }
        let mut args = vec![lhs.vectors.wire, lhs.pubkeys.wire];
        let mut modes = vec![LoopInputMode::Zip, LoopInputMode::Zip];
        if let Some(plaintexts) = &lhs.plaintexts {
            args.push(plaintexts.wire);
            modes.push(LoopInputMode::Zip);
        }
        args.extend([rhs.vectors.wire, rhs.pubkeys.wire]);
        modes.extend([LoopInputMode::Zip, LoopInputMode::Zip]);
        if let Some(plaintexts) = &rhs.plaintexts {
            args.push(plaintexts.wire);
            modes.push(LoopInputMode::Zip);
        }
        let mut output_types =
            vec![output.vector.matrix_type.clone(), output.pubkey.matrix.matrix_type.clone()];
        if let Some(plaintext) = &output.plaintext {
            output_types.push(plaintext.matrix_type.clone());
        }
        let mut outputs = builder.parallel_loop(
            body.finish(),
            lhs.vectors.count.clone(),
            "slot",
            Vec::new(),
            args,
            modes,
            &output_types,
        )?;
        let vectors = outputs.remove(0);
        let pubkeys = outputs.remove(0);
        let plaintexts = (!outputs.is_empty()).then(|| outputs.remove(0));
        Ok(NaiveBggEncodingVecWire {
            vectors,
            pubkeys,
            pubkey_reveal_plaintext: output.pubkey.reveal_plaintext,
            plaintexts,
        })
    }

    fn encoding_scalar(
        &self,
        builder: &mut GraphBuilder,
        input: &NaiveBggEncodingVecWire,
        scalar: &MatrixWire,
        large: bool,
    ) -> Result<NaiveBggEncodingVecWire, NaiveVecCompileError> {
        validate_encoding(input)?;
        let compiler = BggEncodingCompiler { public_key: self.public_key.clone() };
        let mut body = GraphBuilder::new(
            format!(
                "naive-bgg-encoding-{}-scalar-{}",
                if large { "large" } else { "small" },
                reveal_name(input.plaintexts.is_some())
            ),
            Vec::new(),
        );
        let body_input = encoding_body_input(&mut body, "input", input);
        let body_scalar = body.input("3_scalar", scalar.matrix_type.clone());
        let output = if large {
            compiler.large_scalar_mul(&mut body, &body_input, &body_scalar)
        } else {
            compiler.small_scalar_mul(&mut body, &body_input, &body_scalar)
        };
        body.value_output_wire("0_vector", output.vector.wire);
        body.value_output_wire("1_pubkey", output.pubkey.matrix.wire);
        if let Some(plaintext) = &output.plaintext {
            body.value_output_wire("2_plaintext", plaintext.wire);
        }
        let mut args = vec![input.vectors.wire, input.pubkeys.wire];
        let mut modes = vec![LoopInputMode::Zip, LoopInputMode::Zip];
        if let Some(plaintexts) = &input.plaintexts {
            args.push(plaintexts.wire);
            modes.push(LoopInputMode::Zip);
        }
        args.push(scalar.wire);
        modes.push(LoopInputMode::Broadcast);
        let mut output_types =
            vec![output.vector.matrix_type.clone(), output.pubkey.matrix.matrix_type.clone()];
        if let Some(plaintext) = &output.plaintext {
            output_types.push(plaintext.matrix_type.clone());
        }
        let mut outputs = builder.parallel_loop(
            body.finish(),
            input.vectors.count.clone(),
            "slot",
            Vec::new(),
            args,
            modes,
            &output_types,
        )?;
        let vectors = outputs.remove(0);
        let pubkeys = outputs.remove(0);
        let plaintexts = (!outputs.is_empty()).then(|| outputs.remove(0));
        Ok(NaiveBggEncodingVecWire {
            vectors,
            pubkeys,
            pubkey_reveal_plaintext: output.pubkey.reveal_plaintext,
            plaintexts,
        })
    }
}

fn encoding_body_input(
    body: &mut GraphBuilder,
    prefix: &str,
    input: &NaiveBggEncodingVecWire,
) -> BggEncodingWire {
    BggEncodingWire {
        vector: body.input(format!("{prefix}_0_vector"), input.vectors.matrix_type.clone()),
        pubkey: BggPublicKeyWire {
            matrix: body.input(format!("{prefix}_1_pubkey"), input.pubkeys.matrix_type.clone()),
            reveal_plaintext: input.pubkey_reveal_plaintext,
        },
        plaintext: input.plaintexts.as_ref().map(|plaintexts| {
            body.input(format!("{prefix}_2_plaintext"), plaintexts.matrix_type.clone())
        }),
    }
}

fn validate_encoding(input: &NaiveBggEncodingVecWire) -> Result<(), NaiveVecCompileError> {
    validate_family_pair(&input.vectors, &input.pubkeys)?;
    if let Some(plaintexts) = &input.plaintexts {
        validate_family_pair(&input.vectors, plaintexts)?;
    }
    Ok(())
}

fn validate_family_pair(
    lhs: &MatrixFamilyWire,
    rhs: &MatrixFamilyWire,
) -> Result<(), NaiveVecCompileError> {
    if lhs.count != rhs.count {
        return Err(NaiveVecCompileError::SlotCountMismatch);
    }
    Ok(())
}

fn operation_name(operation: MatrixBinaryOp) -> &'static str {
    match operation {
        MatrixBinaryOp::Add => "add",
        MatrixBinaryOp::Subtract => "sub",
        MatrixBinaryOp::Multiply => "mul",
    }
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
    fn encoding_add_zips_every_component_and_matches_primitive_addition() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let columns = parameters.modulus_digits();
        let vector_type = matrix_type(&parameters, 1, columns);
        let public_key_type = matrix_type(&parameters, 2, columns);
        let plaintext_type = matrix_type(&parameters, 1, 1);
        let mut builder = GraphBuilder::new("naive-add-runtime", Vec::new());

        let mut family = |prefix: &str, matrix_type: &MatrixType| {
            let values = (0..2)
                .map(|slot| builder.input(format!("{prefix}_{slot}"), matrix_type.clone()))
                .collect::<Vec<_>>();
            builder.family_pack(&values).expect("homogeneous family")
        };
        let lhs = NaiveBggEncodingVecWire {
            vectors: family("lhs_vector", &vector_type),
            pubkeys: family("lhs_pubkey", &public_key_type),
            pubkey_reveal_plaintext: true,
            plaintexts: Some(family("lhs_plaintext", &plaintext_type)),
        };
        let rhs = NaiveBggEncodingVecWire {
            vectors: family("rhs_vector", &vector_type),
            pubkeys: family("rhs_pubkey", &public_key_type),
            pubkey_reveal_plaintext: true,
            plaintexts: Some(family("rhs_plaintext", &plaintext_type)),
        };
        let compiler = NaiveBggVecCompiler {
            public_key: BggPublicKeyCompiler {
                base: IntExpr::constant(1u64 << parameters.base_bits()),
                decomposed_type: matrix_type(&parameters, columns, columns),
            },
        };
        let output = compiler.add_encodings(&mut builder, &lhs, &rhs).expect("compatible families");
        for slot in 0..2 {
            for (name, output_family) in [
                ("vector", &output.vectors),
                ("pubkey", &output.pubkeys),
                ("plaintext", output.plaintexts.as_ref().expect("revealed plaintexts")),
            ] {
                let value = builder.family_get_static(output_family, IntExpr::constant(slot));
                builder.output(format!("{name}_{slot}"), &value, ArtifactConfidentiality::Public);
            }
        }
        let validated = mxx_ir_core::validate(&builder.finish(), &ParamEnv::default())
            .expect("valid naive-vector graph");

        let mut inputs = BTreeMap::new();
        let mut expected = BTreeMap::new();
        for slot in 0..2 {
            for (component, matrix_type_columns, offset) in [
                ("vector", columns, 0),
                ("pubkey_row_0", columns, 2),
                ("pubkey_row_1", columns, 4),
                ("plaintext", 1, 6),
            ] {
                let lhs_value = row(&parameters, matrix_type_columns, offset + slot);
                let rhs_value = row(&parameters, matrix_type_columns, offset + slot + 1);
                if component.starts_with("pubkey") {
                    continue;
                }
                inputs.insert(
                    format!("lhs_{component}_{slot}"),
                    RuntimeValue::matrix(lhs_value.clone()),
                );
                inputs.insert(
                    format!("rhs_{component}_{slot}"),
                    RuntimeValue::matrix(rhs_value.clone()),
                );
                expected.insert(format!("{component}_{slot}"), lhs_value + rhs_value);
            }
            let lhs_pubkey = DCRTPolyMatrix::from_poly_vec(
                &parameters,
                vec![
                    row(&parameters, columns, 2 + slot).get_row(0),
                    row(&parameters, columns, 4 + slot).get_row(0),
                ],
            );
            let rhs_pubkey = DCRTPolyMatrix::from_poly_vec(
                &parameters,
                vec![
                    row(&parameters, columns, 3 + slot).get_row(0),
                    row(&parameters, columns, 5 + slot).get_row(0),
                ],
            );
            inputs.insert(format!("lhs_pubkey_{slot}"), RuntimeValue::matrix(lhs_pubkey.clone()));
            inputs.insert(format!("rhs_pubkey_{slot}"), RuntimeValue::matrix(rhs_pubkey.clone()));
            expected.insert(format!("pubkey_{slot}"), lhs_pubkey + rhs_pubkey);
        }
        let mut backend = cpu_backend([parameters]);
        let mut store = MemoryArtifactStore::default();
        let result = execute(&validated, &mut backend, inputs, &mut store, SamplingMode::Fresh)
            .expect("naive-vector graph execution");
        for (name, expected) in expected {
            let RuntimeValue::Matrix(actual) = &result.outputs[&name] else {
                panic!("{name} output must be a matrix");
            };
            assert_eq!(actual.as_ref(), &expected);
        }
        assert!(output.pubkey_reveal_plaintext);
        assert!(output.plaintexts.is_some());
    }

    #[test]
    fn arbitrary_matrix_multiplication_matches_primitive_decomposition() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let columns = 2 * parameters.modulus_digits();
        let vector_type = matrix_type(&parameters, 1, columns);
        let public_key_type = matrix_type(&parameters, 2, columns);
        let plaintext_type = matrix_type(&parameters, 1, 1);
        let target_type = matrix_type(&parameters, 2, 1);
        let mut builder = GraphBuilder::new("naive-matrix-mul-runtime", Vec::new());
        let vector_input = builder.input("vector", vector_type);
        let public_key_input = builder.input("public_key", public_key_type);
        let plaintext_input = builder.input("plaintext", plaintext_type);
        let target_input = builder.input("target", target_type);
        let vectors = builder.family_pack(&[vector_input]).expect("vector family");
        let public_keys = builder.family_pack(&[public_key_input]).expect("public-key family");
        let plaintexts = builder.family_pack(&[plaintext_input]).expect("plaintext family");
        let compiler = NaiveBggVecCompiler {
            public_key: BggPublicKeyCompiler {
                base: IntExpr::constant(1u64 << parameters.base_bits()),
                decomposed_type: matrix_type(&parameters, columns, columns),
            },
        };
        let output = compiler
            .matrix_mul_encodings(
                &mut builder,
                &NaiveBggEncodingVecWire {
                    vectors,
                    pubkeys: public_keys,
                    pubkey_reveal_plaintext: true,
                    plaintexts: Some(plaintexts),
                },
                &target_input,
            )
            .expect("compatible matrix multiplication");
        let output_vector = builder.family_get_static(&output.vectors, IntExpr::constant(0));
        let output_public_key = builder.family_get_static(&output.pubkeys, IntExpr::constant(0));
        builder.output("vector", &output_vector, ArtifactConfidentiality::Public);
        builder.output("public_key", &output_public_key, ArtifactConfidentiality::Public);
        let validated = mxx_ir_core::validate(&builder.finish(), &ParamEnv::default())
            .expect("valid matrix-multiplication graph");

        let vector = row(&parameters, columns, 0);
        let public_key = DCRTPolyMatrix::from_poly_vec(
            &parameters,
            vec![row(&parameters, columns, 2).get_row(0), row(&parameters, columns, 4).get_row(0)],
        );
        let plaintext = row(&parameters, 1, 6);
        let target = DCRTPolyMatrix::unit_column_vector(&parameters, 2, 1);
        let expected_vector = vector.mul_decompose(&target);
        let expected_public_key = public_key.mul_decompose(&target);
        let mut backend = cpu_backend([parameters]);
        let mut store = MemoryArtifactStore::default();
        let result = execute(
            &validated,
            &mut backend,
            BTreeMap::from([
                ("vector".to_owned(), RuntimeValue::matrix(vector)),
                ("public_key".to_owned(), RuntimeValue::matrix(public_key)),
                ("plaintext".to_owned(), RuntimeValue::matrix(plaintext)),
                ("target".to_owned(), RuntimeValue::matrix(target)),
            ]),
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("matrix-multiplication execution");
        let RuntimeValue::Matrix(actual_vector) = &result.outputs["vector"] else {
            panic!("vector output must be a matrix");
        };
        let RuntimeValue::Matrix(actual_public_key) = &result.outputs["public_key"] else {
            panic!("public-key output must be a matrix");
        };
        assert_eq!(actual_vector.as_ref(), &expected_vector);
        assert_eq!(actual_public_key.as_ref(), &expected_public_key);
        assert!(output.plaintexts.is_none());
    }
}
