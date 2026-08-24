//! Declarative BGG+ public-key graph values.

use crate::{boolean::BggPublicKeyFamily, encoding::BggSamplerLayout};
use mxx_dsl::{
    Bytes, DslError, GraphValue, GraphValueSchema, HashTag, Mat, MatType, Parallel, Pending, Ring,
};
use mxx_ir_core::{IntExpr, ValueHandle};

#[derive(Clone)]
pub struct BggPublicKeyWire {
    pub matrix: Mat,
    pub reveal_plaintext: bool,
}

#[derive(Clone)]
pub struct BggPublicKeyType {
    pub matrix: MatType,
    pub reveal_plaintext: bool,
}

impl GraphValue for BggPublicKeyWire {
    type Schema = BggPublicKeyType;

    fn flatten(&self) -> Vec<ValueHandle> {
        self.matrix.flatten()
    }

    fn pending(&self) -> Pending {
        self.matrix.pending()
    }

    fn schema(&self) -> Self::Schema {
        BggPublicKeyType { matrix: self.matrix.schema(), reveal_plaintext: self.reveal_plaintext }
    }

    fn from_values(
        schema: &Self::Schema,
        values: &[ValueHandle],
        pending: Pending,
    ) -> Result<Self, DslError> {
        Ok(Self {
            matrix: Mat::from_values(&schema.matrix, values, pending)?,
            reveal_plaintext: schema.reveal_plaintext,
        })
    }
}

impl GraphValueSchema for BggPublicKeyType {
    type Value = BggPublicKeyWire;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        BggPublicKeyWire {
            matrix: self.matrix.placeholders_from(next),
            reveal_plaintext: self.reveal_plaintext,
        }
    }

    fn wire_types(&self) -> Vec<mxx_ir_core::types::WireType> {
        self.matrix.wire_types()
    }
}

#[derive(Clone)]
pub struct BggPublicKeyCompiler {
    pub ring: Ring,
    pub base: IntExpr,
    pub digit_count: IntExpr,
}

impl BggPublicKeyCompiler {
    pub fn add(&self, lhs: &BggPublicKeyWire, rhs: &BggPublicKeyWire) -> BggPublicKeyWire {
        BggPublicKeyWire {
            matrix: lhs.matrix.clone() + rhs.matrix.clone(),
            reveal_plaintext: lhs.reveal_plaintext && rhs.reveal_plaintext,
        }
    }

    pub fn sub(&self, lhs: &BggPublicKeyWire, rhs: &BggPublicKeyWire) -> BggPublicKeyWire {
        BggPublicKeyWire {
            matrix: lhs.matrix.clone() - rhs.matrix.clone(),
            reveal_plaintext: lhs.reveal_plaintext && rhs.reveal_plaintext,
        }
    }

    /// Builds `lhs * G^-1(rhs)` directly in the executable core DAG.
    pub fn mul(&self, lhs: &BggPublicKeyWire, rhs: &BggPublicKeyWire) -> BggPublicKeyWire {
        let decomposed = self.decompose(rhs);
        self.mul_with_decomposition(lhs, rhs, decomposed)
    }

    pub(crate) fn decompose(&self, input: &BggPublicKeyWire) -> Mat {
        input.matrix.clone().decompose(self.base.clone(), self.digit_count.clone()).as_mat()
    }

    pub(crate) fn mul_with_decomposition(
        &self,
        lhs: &BggPublicKeyWire,
        rhs: &BggPublicKeyWire,
        decomposed_rhs: Mat,
    ) -> BggPublicKeyWire {
        BggPublicKeyWire {
            matrix: lhs.matrix.clone() * decomposed_rhs,
            reveal_plaintext: lhs.reveal_plaintext && rhs.reveal_plaintext,
        }
    }

    pub fn small_scalar_mul(&self, input: &BggPublicKeyWire, scalar: &Mat) -> BggPublicKeyWire {
        BggPublicKeyWire {
            matrix: input.matrix.clone() * scalar.clone(),
            reveal_plaintext: input.reveal_plaintext,
        }
    }

    pub fn large_scalar_mul(&self, input: &BggPublicKeyWire, scalar: &Mat) -> BggPublicKeyWire {
        let decomposed = self.large_scalar_decomposition(input, scalar);
        self.large_scalar_mul_with_decomposition(input, decomposed)
    }

    pub fn matrix_mul(&self, input: &BggPublicKeyWire, target: &Mat) -> BggPublicKeyWire {
        let decomposed =
            target.clone().decompose(self.base.clone(), self.digit_count.clone()).as_mat();
        BggPublicKeyWire {
            matrix: input.matrix.clone() * decomposed,
            reveal_plaintext: input.reveal_plaintext,
        }
    }

    pub(crate) fn large_scalar_mul_with_decomposition(
        &self,
        input: &BggPublicKeyWire,
        decomposed: Mat,
    ) -> BggPublicKeyWire {
        BggPublicKeyWire {
            matrix: input.matrix.clone() * decomposed,
            reveal_plaintext: input.reveal_plaintext,
        }
    }

    pub(crate) fn large_scalar_decomposition(&self, input: &BggPublicKeyWire, scalar: &Mat) -> Mat {
        let rows = input.matrix.matrix_type().rows.clone();
        let gadget = self.ring.gadget(rows, self.base.clone(), self.digit_count.clone());
        (gadget * scalar.clone()).decompose(self.base.clone(), self.digit_count.clone()).as_mat()
    }
}

#[derive(Clone)]
pub struct BggPublicKeySampler {
    pub layout: BggSamplerLayout,
}

impl BggPublicKeySampler {
    /// Samples a dynamically sized family of ordinary-width public matrices in parallel.
    ///
    /// Every member reveals its plaintext relation. Use [`Self::sample`] when the family size and
    /// reveal policy are both statically known while constructing the graph.
    pub fn sample_family(
        &self,
        hash_key: Bytes,
        tag: impl Into<HashTag>,
        count: impl Into<IntExpr>,
        public_key_columns: impl Into<IntExpr>,
    ) -> Result<BggPublicKeyFamily, DslError> {
        let count = count.into();
        let columns = public_key_columns.into();
        let ring = self.layout.ring();
        let base_tag = tag.into();
        let rows = self.layout.secret_dimension;
        let matrices = Parallel::range(count).map_values(move |index| {
            let mut indexed_tag = base_tag.clone();
            indexed_tag.push(index);
            ring.hash_matrix(hash_key.clone(), indexed_tag, (rows, columns.clone()))
        })?;
        Ok(BggPublicKeyFamily { matrices, reveal_plaintext: true })
    }

    /// Samples ordinary-width public matrices in parallel with deterministic indexed tags.
    pub fn sample(
        &self,
        hash_key: Bytes,
        tag: impl Into<HashTag>,
        reveal_plaintexts: &[bool],
    ) -> Vec<BggPublicKeyWire> {
        let count = reveal_plaintexts.len() + 1;
        let columns = self.layout.public_key_columns();
        let ring = self.layout.ring();
        let base_tag = tag.into();
        let rows = self.layout.secret_dimension;
        let matrices = Parallel::range(count)
            .map_values(move |index| {
                let mut indexed_tag = base_tag.clone();
                indexed_tag.push(index);
                ring.hash_matrix(hash_key.clone(), indexed_tag, (rows, columns))
            })
            .expect("static public-key family layout is valid");
        (0..count)
            .map(|index| BggPublicKeyWire {
                matrix: matrices.get_static(index),
                reveal_plaintext: index == 0 || reveal_plaintexts[index - 1],
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{execute_graph, matrix_output, row};
    use mxx_dsl::DslContext;
    use mxx_ir_core::ParamEnv;
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{PolyParams, dcrt::params::DCRTPolyParams},
    };
    use mxx_runtime::RuntimeValue;
    use num_bigint::BigInt;
    use std::collections::BTreeMap;

    #[test]
    fn multiplication_is_decompose_then_multiply() {
        let ring = Ring::new(17, 8);
        let lhs = BggPublicKeyWire { matrix: ring.input("lhs", (2, 4)), reveal_plaintext: true };
        let rhs = BggPublicKeyWire { matrix: ring.input("rhs", (2, 4)), reveal_plaintext: true };
        let compiler = BggPublicKeyCompiler { ring, base: 2.into(), digit_count: 2.into() };
        let output = compiler.mul(&lhs, &rhs);
        let built = DslContext::new("bgg-public-key-mul")
            .public_output("output", output.matrix)
            .expect("output")
            .build()
            .expect("build");
        mxx_ir_core::validate(&built.graph, &ParamEnv::default()).expect("valid graph");
    }

    #[test]
    fn reveal_metadata_matches_the_public_key_contract() {
        let ring = Ring::new(17, 8);
        let compiler =
            BggPublicKeyCompiler { ring: ring.clone(), base: 2.into(), digit_count: 2.into() };
        for left_revealed in [false, true] {
            for right_revealed in [false, true] {
                let left = BggPublicKeyWire {
                    matrix: ring.input("left", (2, 4)),
                    reveal_plaintext: left_revealed,
                };
                let right = BggPublicKeyWire {
                    matrix: ring.input("right", (2, 4)),
                    reveal_plaintext: right_revealed,
                };
                let expected = left_revealed && right_revealed;
                assert_eq!(compiler.add(&left, &right).reveal_plaintext, expected);
                assert_eq!(compiler.sub(&left, &right).reveal_plaintext, expected);
                assert_eq!(compiler.mul(&left, &right).reveal_plaintext, expected);
            }
        }
        for revealed in [false, true] {
            let input = BggPublicKeyWire {
                matrix: ring.input("input", (2, 4)),
                reveal_plaintext: revealed,
            };
            let scalar = ring.input("scalar", (1, 1));
            assert_eq!(compiler.small_scalar_mul(&input, &scalar).reveal_plaintext, revealed);
            assert_eq!(compiler.large_scalar_mul(&input, &scalar).reveal_plaintext, revealed);
        }
    }

    #[test]
    fn runtime_operations_match_primitive_matrix_formulas() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digit_count = parameters.modulus_digits();
        let columns = 2 * digit_count;
        let ring = Ring::new(
            BigInt::from(parameters.modulus().as_ref().clone()),
            parameters.ring_dimension() as usize,
        );
        let compiler = BggPublicKeyCompiler {
            ring: ring.clone(),
            base: BigInt::from(1u64 << parameters.base_bits()).into(),
            digit_count: digit_count.into(),
        };
        let lhs =
            BggPublicKeyWire { matrix: ring.input("lhs", (2, columns)), reveal_plaintext: true };
        let rhs =
            BggPublicKeyWire { matrix: ring.input("rhs", (2, columns)), reveal_plaintext: true };
        let target = ring.input("target", (2, 1));
        let add = compiler.add(&lhs, &rhs);
        let sub = compiler.sub(&lhs, &rhs);
        let mul = compiler.mul(&lhs, &rhs);
        let matrix_mul = compiler.matrix_mul(&lhs, &target);
        let graph = DslContext::new("bgg-public-key-runtime")
            .output("add", add.matrix)
            .unwrap()
            .output("sub", sub.matrix)
            .unwrap()
            .output("mul", mul.matrix)
            .unwrap()
            .output("matrix-mul", matrix_mul.matrix)
            .unwrap()
            .build()
            .unwrap();

        let lhs_value = DCRTPolyMatrix::from_poly_vec(
            &parameters,
            vec![row(&parameters, columns, 0).get_row(0), row(&parameters, columns, 2).get_row(0)],
        );
        let rhs_value = DCRTPolyMatrix::from_poly_vec(
            &parameters,
            vec![row(&parameters, columns, 1).get_row(0), row(&parameters, columns, 3).get_row(0)],
        );
        let target_value = DCRTPolyMatrix::unit_column_vector(&parameters, 2, 1);
        let result = execute_graph(
            graph,
            parameters,
            BTreeMap::from([
                ("lhs".to_owned(), RuntimeValue::matrix(lhs_value.clone())),
                ("rhs".to_owned(), RuntimeValue::matrix(rhs_value.clone())),
                ("target".to_owned(), RuntimeValue::matrix(target_value.clone())),
            ]),
        );

        assert_eq!(matrix_output(&result, "add"), &(lhs_value.clone() + rhs_value.clone()));
        assert_eq!(matrix_output(&result, "sub"), &(lhs_value.clone() - rhs_value.clone()));
        assert_eq!(matrix_output(&result, "mul"), &lhs_value.mul_decompose(&rhs_value));
        assert_eq!(matrix_output(&result, "matrix-mul"), &lhs_value.mul_decompose(&target_value));
    }
}
