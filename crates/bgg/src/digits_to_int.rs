use crate::{
    BggEncodingCompiler, BggEncodingWire, BggPublicKeyCompiler, BggPublicKeyWire,
    encoding::EncodingCompileError,
};
use mxx_ir_core::{
    GraphBuilder, IntExpr, MatrixWire,
    node::{ConstantMatrix, MatrixBinaryOp},
    types::MatrixType,
};
use thiserror::Error;

/// Graph IR replacement for the legacy `DigitsToInt` evaluator.
#[derive(Clone, Debug)]
pub struct BggDigitsToIntCompiler {
    pub public_key: BggPublicKeyCompiler,
    pub digit_count: usize,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum DigitsToIntCompileError {
    #[error("digits_to_int requires exactly {expected} digits, received {actual}")]
    DigitCount { expected: usize, actual: usize },
    #[error(transparent)]
    Encoding(#[from] EncodingCompileError),
}

impl BggDigitsToIntCompiler {
    pub fn public_keys(
        &self,
        builder: &mut GraphBuilder,
        digits: &[BggPublicKeyWire],
    ) -> Result<BggPublicKeyWire, DigitsToIntCompileError> {
        self.require_digit_count(digits.len())?;
        let terms = digits
            .iter()
            .enumerate()
            .map(|(exponent, digit)| self.public_key_power(builder, digit, exponent))
            .collect::<Vec<_>>();
        let mut sum = terms[0].clone();
        for term in &terms[1..] {
            sum = self.public_key.add(builder, &sum, &term);
        }
        Ok(sum)
    }

    pub fn encodings(
        &self,
        builder: &mut GraphBuilder,
        digits: &[BggEncodingWire],
    ) -> Result<BggEncodingWire, DigitsToIntCompileError> {
        self.require_digit_count(digits.len())?;
        let encoding_compiler = BggEncodingCompiler { public_key: self.public_key.clone() };
        let terms = digits
            .iter()
            .enumerate()
            .map(|(exponent, digit)| self.encoding_power(builder, digit, exponent))
            .collect::<Vec<_>>();
        let mut sum = terms[0].clone();
        for term in &terms[1..] {
            sum = encoding_compiler.add(builder, &sum, &term)?;
        }
        Ok(sum)
    }

    fn require_digit_count(&self, actual: usize) -> Result<(), DigitsToIntCompileError> {
        if self.digit_count == actual && actual > 0 {
            Ok(())
        } else {
            Err(DigitsToIntCompileError::DigitCount { expected: self.digit_count, actual })
        }
    }

    fn public_key_power(
        &self,
        builder: &mut GraphBuilder,
        input: &BggPublicKeyWire,
        exponent: usize,
    ) -> BggPublicKeyWire {
        let decomposition = self.power_decomposition(builder, &input.matrix, exponent);
        let output_type = single_column_type(&input.matrix.matrix_type);
        BggPublicKeyWire {
            matrix: builder.matrix_binary(
                MatrixBinaryOp::Multiply,
                &input.matrix,
                &decomposition,
                output_type,
            ),
            reveal_plaintext: input.reveal_plaintext,
        }
    }

    fn encoding_power(
        &self,
        builder: &mut GraphBuilder,
        input: &BggEncodingWire,
        exponent: usize,
    ) -> BggEncodingWire {
        let decomposition = self.power_decomposition(builder, &input.pubkey.matrix, exponent);
        let vector_type = single_column_type(&input.vector.matrix_type);
        let vector = builder.matrix_binary(
            MatrixBinaryOp::Multiply,
            &input.vector,
            &decomposition,
            vector_type,
        );
        let public_key_type = single_column_type(&input.pubkey.matrix.matrix_type);
        let pubkey = BggPublicKeyWire {
            matrix: builder.matrix_binary(
                MatrixBinaryOp::Multiply,
                &input.pubkey.matrix,
                &decomposition,
                public_key_type,
            ),
            reveal_plaintext: input.pubkey.reveal_plaintext,
        };
        let scalar = builder.constant_matrix(
            scalar_type(&input.pubkey.matrix.matrix_type),
            ConstantMatrix::PowerOfBase {
                base: self.public_key.base.clone(),
                exponent: IntExpr::constant(exponent),
            },
        );
        let plaintext = input.plaintext.as_ref().map(|plaintext| {
            builder.matrix_binary(
                MatrixBinaryOp::Multiply,
                plaintext,
                &scalar,
                plaintext.matrix_type.clone(),
            )
        });
        BggEncodingWire { vector, pubkey, plaintext }
    }

    fn power_decomposition(
        &self,
        builder: &mut GraphBuilder,
        input: &MatrixWire,
        exponent: usize,
    ) -> MatrixWire {
        let scalar_type = scalar_type(&input.matrix_type);
        let scalar = builder.constant_matrix(
            scalar_type.clone(),
            ConstantMatrix::PowerOfBase {
                base: self.public_key.base.clone(),
                exponent: IntExpr::constant(exponent),
            },
        );
        let unit_type = single_column_type(&input.matrix_type);
        let unit = builder.constant_matrix(
            unit_type.clone(),
            ConstantMatrix::UnitColumn {
                index: IntExpr::Sub(
                    Box::new(input.matrix_type.rows.clone()),
                    Box::new(IntExpr::constant(1)),
                )
                .canonicalize(),
            },
        );
        let scaled = builder.matrix_binary(MatrixBinaryOp::Multiply, &unit, &scalar, unit_type);
        let decomposed_type = MatrixType {
            rows: input.matrix_type.columns.clone(),
            columns: IntExpr::constant(1),
            ..input.matrix_type.clone()
        };
        builder.gadget_decompose(&scaled, self.public_key.base.clone(), decomposed_type)
    }
}

fn scalar_type(input: &MatrixType) -> MatrixType {
    MatrixType { rows: IntExpr::constant(1), columns: IntExpr::constant(1), ..input.clone() }
}

fn single_column_type(input: &MatrixType) -> MatrixType {
    MatrixType { columns: IntExpr::constant(1), ..input.clone() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::{ParamEnv, artifact::ArtifactConfidentiality};
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
    fn encoding_digits_match_the_primitive_mul_decompose_formula() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digit_count = parameters.modulus_digits();
        let secret_dimension = 2usize;
        let columns = secret_dimension * digit_count;
        let public_key_type = matrix_type(&parameters, secret_dimension, columns);
        let vector_type = matrix_type(&parameters, 1, columns);
        let plaintext_type = matrix_type(&parameters, 1, 1);
        let base = 1u64 << parameters.base_bits();
        let mut builder = GraphBuilder::new("digits-to-int", Vec::new());
        let digits = (0..digit_count)
            .map(|digit| BggEncodingWire {
                vector: builder.input(format!("vector_{digit}"), vector_type.clone()),
                pubkey: BggPublicKeyWire {
                    matrix: builder.input(format!("pubkey_{digit}"), public_key_type.clone()),
                    reveal_plaintext: true,
                },
                plaintext: Some(
                    builder.input(format!("plaintext_{digit}"), plaintext_type.clone()),
                ),
            })
            .collect::<Vec<_>>();
        let compiler = BggDigitsToIntCompiler {
            public_key: BggPublicKeyCompiler {
                base: IntExpr::constant(base),
                decomposed_type: matrix_type(&parameters, columns, columns),
            },
            digit_count,
        };
        let output = compiler.encodings(&mut builder, &digits).expect("complete digit set");
        builder.output("vector", &output.vector, ArtifactConfidentiality::Public);
        builder.output("pubkey", &output.pubkey.matrix, ArtifactConfidentiality::Public);
        builder.output(
            "plaintext",
            output.plaintext.as_ref().expect("revealed plaintext"),
            ArtifactConfidentiality::Public,
        );
        let validated = mxx_ir_core::validate(&builder.finish(), &ParamEnv::default())
            .expect("valid digits-to-int graph");

        let mut inputs = BTreeMap::new();
        let mut expected_vector = DCRTPolyMatrix::zero(&parameters, 1, 1);
        let mut expected_pubkey = DCRTPolyMatrix::zero(&parameters, secret_dimension, 1);
        let mut expected_plaintext = DCRTPolyMatrix::zero(&parameters, 1, 1);
        for digit in 0..digit_count {
            let vector = row(&parameters, columns, digit);
            let pubkey = DCRTPolyMatrix::from_poly_vec(
                &parameters,
                vec![
                    row(&parameters, columns, digit + 1).get_row(0),
                    row(&parameters, columns, digit + 2).get_row(0),
                ],
            );
            let plaintext = row(&parameters, 1, digit + 3);
            inputs.insert(format!("vector_{digit}"), RuntimeValue::matrix(vector.clone()));
            inputs.insert(format!("pubkey_{digit}"), RuntimeValue::matrix(pubkey.clone()));
            inputs.insert(format!("plaintext_{digit}"), RuntimeValue::matrix(plaintext.clone()));

            let power = DCRTPoly::from_power_of_base_to_constant(&parameters, digit);
            let unit = DCRTPolyMatrix::unit_column_vector(
                &parameters,
                secret_dimension,
                secret_dimension - 1,
            );
            let scaled = unit * power.clone();
            expected_vector = expected_vector + vector.mul_decompose(&scaled);
            expected_pubkey = expected_pubkey + pubkey.mul_decompose(&scaled);
            expected_plaintext = expected_plaintext +
                plaintext * DCRTPolyMatrix::from_poly_vec(&parameters, vec![vec![power]]);
        }
        let mut backend = cpu_backend([parameters]);
        let mut store = MemoryArtifactStore::default();
        let result = execute(&validated, &mut backend, inputs, &mut store, SamplingMode::Fresh)
            .expect("digits-to-int execution");
        for (name, expected) in [
            ("vector", expected_vector),
            ("pubkey", expected_pubkey),
            ("plaintext", expected_plaintext),
        ] {
            let RuntimeValue::Matrix(actual) = &result.outputs[name] else {
                panic!("{name} output must be a matrix");
            };
            assert_eq!(actual.as_ref(), &expected);
        }
    }
}
