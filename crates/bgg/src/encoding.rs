//! Declarative BGG+ encoding graph values.

use crate::{BggPublicKeyCompiler, BggPublicKeyType, BggPublicKeyWire};
use mxx_dsl::{DslError, GraphValue, GraphValueSchema, Mat, MatType, Pending};
use mxx_ir_core::{ValueHandle, WireType};
use thiserror::Error;

#[derive(Clone)]
pub struct BggEncodingWire {
    pub vector: Mat,
    pub pubkey: BggPublicKeyWire,
    pub plaintext: Option<Mat>,
}

#[derive(Clone)]
pub struct BggEncodingType {
    pub vector: MatType,
    pub pubkey: BggPublicKeyType,
    pub plaintext: Option<MatType>,
}

impl GraphValue for BggEncodingWire {
    type Schema = BggEncodingType;

    fn flatten(&self) -> Vec<ValueHandle> {
        let mut values = self.vector.flatten();
        values.extend(self.pubkey.flatten());
        if let Some(plaintext) = &self.plaintext {
            values.extend(plaintext.flatten());
        }
        values
    }

    fn pending(&self) -> Pending {
        Pending::merge(
            std::iter::once(self.vector.pending())
                .chain(std::iter::once(self.pubkey.pending()))
                .chain(self.plaintext.as_ref().map(GraphValue::pending)),
        )
    }

    fn schema(&self) -> Self::Schema {
        BggEncodingType {
            vector: self.vector.schema(),
            pubkey: self.pubkey.schema(),
            plaintext: self.plaintext.as_ref().map(GraphValue::schema),
        }
    }

    fn from_values(
        schema: &Self::Schema,
        values: &[ValueHandle],
        pending: Pending,
    ) -> Result<Self, DslError> {
        let vector_count = schema.vector.wire_types().len();
        let pubkey_count = schema.pubkey.wire_types().len();
        let expected = vector_count + pubkey_count + usize::from(schema.plaintext.is_some());
        if values.len() != expected {
            return Err(DslError::Schema);
        }
        Ok(Self {
            vector: Mat::from_values(&schema.vector, &values[..vector_count], pending.clone())?,
            pubkey: BggPublicKeyWire::from_values(
                &schema.pubkey,
                &values[vector_count..vector_count + pubkey_count],
                pending.clone(),
            )?,
            plaintext: schema
                .plaintext
                .as_ref()
                .map(|ty| Mat::from_values(ty, &values[vector_count + pubkey_count..], pending))
                .transpose()?,
        })
    }
}

impl GraphValueSchema for BggEncodingType {
    type Value = BggEncodingWire;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        BggEncodingWire {
            vector: self.vector.placeholders_from(next),
            pubkey: self.pubkey.placeholders_from(next),
            plaintext: self.plaintext.as_ref().map(|plaintext| plaintext.placeholders_from(next)),
        }
    }

    fn wire_types(&self) -> Vec<WireType> {
        let mut types = self.vector.wire_types();
        types.extend(self.pubkey.wire_types());
        if let Some(plaintext) = &self.plaintext {
            types.extend(plaintext.wire_types());
        }
        types
    }
}

#[derive(Clone)]
pub struct BggEncodingCompiler {
    pub public_key: BggPublicKeyCompiler,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum EncodingCompileError {
    #[error("BGG+ multiplication requires the left operand plaintext")]
    MissingLeftPlaintext,
}

impl BggEncodingCompiler {
    pub fn add(
        &self,
        lhs: &BggEncodingWire,
        rhs: &BggEncodingWire,
    ) -> Result<BggEncodingWire, EncodingCompileError> {
        Ok(BggEncodingWire {
            vector: lhs.vector.clone() + rhs.vector.clone(),
            pubkey: self.public_key.add(&lhs.pubkey, &rhs.pubkey),
            plaintext: binary_plaintext(lhs, rhs, |left, right| left + right),
        })
    }

    pub fn sub(
        &self,
        lhs: &BggEncodingWire,
        rhs: &BggEncodingWire,
    ) -> Result<BggEncodingWire, EncodingCompileError> {
        Ok(BggEncodingWire {
            vector: lhs.vector.clone() - rhs.vector.clone(),
            pubkey: self.public_key.sub(&lhs.pubkey, &rhs.pubkey),
            plaintext: binary_plaintext(lhs, rhs, |left, right| left - right),
        })
    }

    /// Builds `c_L G^-1(A_R) + c_R x_L`, matching the concrete BGG+ formula.
    pub fn mul(
        &self,
        lhs: &BggEncodingWire,
        rhs: &BggEncodingWire,
    ) -> Result<BggEncodingWire, EncodingCompileError> {
        let plaintext = lhs.plaintext.clone().ok_or(EncodingCompileError::MissingLeftPlaintext)?;
        let decomposed_rhs = rhs
            .pubkey
            .matrix
            .clone()
            .decompose(self.public_key.base.clone(), self.public_key.digit_count.clone())
            .as_mat();
        Ok(BggEncodingWire {
            vector: lhs.vector.clone() * decomposed_rhs + rhs.vector.clone() * plaintext,
            pubkey: self.public_key.mul(&lhs.pubkey, &rhs.pubkey),
            plaintext: binary_plaintext(lhs, rhs, |left, right| left * right),
        })
    }

    pub fn small_scalar_mul(&self, input: &BggEncodingWire, scalar: &Mat) -> BggEncodingWire {
        BggEncodingWire {
            vector: input.vector.clone() * scalar.clone(),
            pubkey: self.public_key.small_scalar_mul(&input.pubkey, scalar),
            plaintext: input.plaintext.clone().map(|value| value * scalar.clone()),
        }
    }

    pub fn large_scalar_mul(&self, input: &BggEncodingWire, scalar: &Mat) -> BggEncodingWire {
        let decomposed = self.public_key.large_scalar_decomposition(&input.pubkey, scalar);
        BggEncodingWire {
            vector: input.vector.clone() * decomposed.clone(),
            pubkey: self.public_key.large_scalar_mul_with_decomposition(&input.pubkey, decomposed),
            plaintext: input.plaintext.clone().map(|value| value * scalar.clone()),
        }
    }

    pub fn matrix_mul(&self, input: &BggEncodingWire, target: &Mat) -> BggEncodingWire {
        let decomposed = target
            .clone()
            .decompose(self.public_key.base.clone(), self.public_key.digit_count.clone())
            .as_mat();
        BggEncodingWire {
            vector: input.vector.clone() * decomposed,
            pubkey: self.public_key.matrix_mul(&input.pubkey, target),
            plaintext: None,
        }
    }
}

fn binary_plaintext(
    lhs: &BggEncodingWire,
    rhs: &BggEncodingWire,
    operation: impl FnOnce(Mat, Mat) -> Mat,
) -> Option<Mat> {
    lhs.plaintext.clone().zip(rhs.plaintext.clone()).map(|(lhs, rhs)| operation(lhs, rhs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{execute_graph, matrix_output, row};
    use mxx_dsl::{DslContext, Ring, Subgraph};
    use mxx_ir_core::{ParamEnv, node::NodeKind};
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{PolyParams, dcrt::params::DCRTPolyParams},
    };
    use mxx_runtime::RuntimeValue;
    use num_bigint::BigInt;
    use std::collections::BTreeMap;

    #[test]
    fn repeated_bgg_encoding_schema_defines_a_subgraph() {
        let ring = Ring::new(257, 8);
        let matrix = MatType(ring.matrix_type((1, 1)));
        let encoding = BggEncodingType {
            vector: matrix.clone(),
            pubkey: BggPublicKeyType { matrix: matrix.clone(), reveal_plaintext: true },
            plaintext: Some(matrix),
        };
        Subgraph::<(BggEncodingWire, BggEncodingWire), _>::define(
            "bgg-pair-reverse",
            (encoding.clone(), encoding),
            |(left, right)| (right, left),
        )
        .expect("BGG typed arguments use distinct flattened input names");
    }

    #[test]
    fn reveal_combinations_match_the_encoding_contract() {
        let ring = Ring::new(17, 8);
        let compiler = BggEncodingCompiler {
            public_key: BggPublicKeyCompiler {
                ring: ring.clone(),
                base: 2.into(),
                digit_count: 2.into(),
            },
        };
        for left_revealed in [false, true] {
            for right_revealed in [false, true] {
                let encoding = |prefix: &str, revealed: bool| BggEncodingWire {
                    vector: ring.input(format!("{prefix}-vector"), (1, 4)),
                    pubkey: BggPublicKeyWire {
                        matrix: ring.input(format!("{prefix}-public"), (2, 4)),
                        reveal_plaintext: revealed,
                    },
                    plaintext: revealed.then(|| ring.input(format!("{prefix}-plain"), (1, 1))),
                };
                let left = encoding("left", left_revealed);
                let right = encoding("right", right_revealed);
                let expected = left_revealed && right_revealed;
                for output in
                    [compiler.add(&left, &right).unwrap(), compiler.sub(&left, &right).unwrap()]
                {
                    assert_eq!(output.pubkey.reveal_plaintext, expected);
                    assert_eq!(output.plaintext.is_some(), expected);
                }
                match compiler.mul(&left, &right) {
                    Ok(output) => {
                        assert!(left_revealed);
                        assert_eq!(output.pubkey.reveal_plaintext, expected);
                        assert_eq!(output.plaintext.is_some(), expected);
                    }
                    Err(error) => {
                        assert!(!left_revealed);
                        assert_eq!(error, EncodingCompileError::MissingLeftPlaintext);
                    }
                }
            }
        }
    }

    #[test]
    fn encoding_multiplication_keeps_executable_decompose_multiply_add_and_elaborates() {
        let ring = Ring::new(257, 8);
        let compiler = BggEncodingCompiler {
            public_key: BggPublicKeyCompiler {
                ring: ring.clone(),
                base: 4.into(),
                digit_count: 4.into(),
            },
        };
        let encoding = |prefix: &str| BggEncodingWire {
            vector: ring.input(format!("{prefix}-vector"), (1, 8)),
            pubkey: BggPublicKeyWire {
                matrix: ring.input(format!("{prefix}-public"), (2, 8)),
                reveal_plaintext: true,
            },
            plaintext: Some(ring.input(format!("{prefix}-plaintext"), (1, 1))),
        };
        let product = compiler.mul(&encoding("left"), &encoding("right")).expect("product");
        let built = DslContext::new("bgg-encoding-mul")
            .output("vector", product.vector)
            .expect("vector output")
            .output("public", product.pubkey.matrix)
            .expect("public output")
            .build()
            .expect("build");
        let kinds = built
            .graph
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .map(|node| node.kind())
            .collect::<Vec<_>>();
        assert_eq!(
            kinds.iter().filter(|kind| matches!(kind, NodeKind::GadgetDecompose { .. })).count(),
            2
        );
        assert!(kinds.iter().any(|kind| matches!(kind, NodeKind::MatrixBinary(_))));

        let elaborated = built.elaborate(&ParamEnv::default()).expect("symbolic elaboration");
        let vector = elaborated.wire(&elaborated.outputs["vector"]).expect("vector output");
        assert!(
            elaborated.expressions.get(vector.expression.expect("vector expression")).is_some()
        );
    }

    #[test]
    fn runtime_multiplication_matches_the_bgg_encoding_formula() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digit_count = parameters.modulus_digits();
        let columns = 2 * digit_count;
        let ring = Ring::new(
            BigInt::from(parameters.modulus().as_ref().clone()),
            parameters.ring_dimension() as usize,
        );
        let compiler = BggEncodingCompiler {
            public_key: BggPublicKeyCompiler {
                ring: ring.clone(),
                base: BigInt::from(1u64 << parameters.base_bits()).into(),
                digit_count: digit_count.into(),
            },
        };
        let encoding = |prefix: &str| BggEncodingWire {
            vector: ring.input(format!("{prefix}-vector"), (1, columns)),
            pubkey: BggPublicKeyWire {
                matrix: ring.input(format!("{prefix}-public"), (2, columns)),
                reveal_plaintext: true,
            },
            plaintext: Some(ring.input(format!("{prefix}-plaintext"), (1, 1))),
        };
        let output = compiler.mul(&encoding("lhs"), &encoding("rhs")).unwrap();
        let graph = DslContext::new("bgg-encoding-runtime")
            .output("vector", output.vector)
            .unwrap()
            .output("public", output.pubkey.matrix)
            .unwrap()
            .output("plaintext", output.plaintext.unwrap())
            .unwrap()
            .build()
            .unwrap();

        let lhs_vector = row(&parameters, columns, 0);
        let rhs_vector = row(&parameters, columns, 1);
        let lhs_public = DCRTPolyMatrix::from_poly_vec(
            &parameters,
            vec![row(&parameters, columns, 2).get_row(0), row(&parameters, columns, 3).get_row(0)],
        );
        let rhs_public = DCRTPolyMatrix::from_poly_vec(
            &parameters,
            vec![row(&parameters, columns, 4).get_row(0), row(&parameters, columns, 5).get_row(0)],
        );
        let lhs_plaintext = row(&parameters, 1, 6);
        let rhs_plaintext = row(&parameters, 1, 7);
        let result = execute_graph(
            graph,
            parameters,
            BTreeMap::from([
                ("lhs-vector".to_owned(), RuntimeValue::matrix(lhs_vector.clone())),
                ("rhs-vector".to_owned(), RuntimeValue::matrix(rhs_vector.clone())),
                ("lhs-public".to_owned(), RuntimeValue::matrix(lhs_public.clone())),
                ("rhs-public".to_owned(), RuntimeValue::matrix(rhs_public.clone())),
                ("lhs-plaintext".to_owned(), RuntimeValue::matrix(lhs_plaintext.clone())),
                ("rhs-plaintext".to_owned(), RuntimeValue::matrix(rhs_plaintext.clone())),
            ]),
        );

        let expected_vector =
            lhs_vector.mul_decompose(&rhs_public) + rhs_vector * lhs_plaintext.entry(0, 0);
        assert_eq!(matrix_output(&result, "vector"), &expected_vector);
        assert_eq!(matrix_output(&result, "public"), &lhs_public.mul_decompose(&rhs_public));
        assert_eq!(matrix_output(&result, "plaintext"), &(lhs_plaintext * rhs_plaintext));
    }
}
