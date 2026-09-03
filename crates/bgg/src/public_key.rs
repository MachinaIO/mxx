//! Declarative BGG+ public-key graph values.

use crate::{boolean::BggPublicKeyFamily, encoding::BggSamplerLayout};
use mxx_dsl::{
    Bytes, DslError, GraphValue, GraphValueSchema, HashTag, Mat, MatType, Parallel, Pending,
    Preimage, Ring,
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
        // Public matrices add in the same coordinates as their encodings:
        // A_out = A_L + A_R, with the reveal bit retained only when both
        // plaintext relations are known.
        BggPublicKeyWire {
            matrix: lhs.matrix.clone() + rhs.matrix.clone(),
            reveal_plaintext: lhs.reveal_plaintext && rhs.reveal_plaintext,
        }
    }

    pub fn sub(&self, lhs: &BggPublicKeyWire, rhs: &BggPublicKeyWire) -> BggPublicKeyWire {
        // Subtraction forms A_out = A_L - A_R component by component; no
        // decomposition is needed because the gadget-column layout is fixed.
        BggPublicKeyWire {
            matrix: lhs.matrix.clone() - rhs.matrix.clone(),
            reveal_plaintext: lhs.reveal_plaintext && rhs.reveal_plaintext,
        }
    }

    /// Builds `lhs * G^-1(rhs)` directly in the executable core DAG.
    pub fn mul(&self, lhs: &BggPublicKeyWire, rhs: &BggPublicKeyWire) -> BggPublicKeyWire {
        let decomposed = rhs.matrix.clone().decompose(self.base.clone(), self.digit_count.clone());
        self.mul_with_decomposition(lhs, rhs, decomposed)
    }

    pub(crate) fn mul_with_decomposition(
        &self,
        lhs: &BggPublicKeyWire,
        rhs: &BggPublicKeyWire,
        decomposed_rhs: Preimage,
    ) -> BggPublicKeyWire {
        // If G K_R = A_R, then A_L K_R is the public-key product A_L G^-1(A_R).
        BggPublicKeyWire {
            matrix: lhs.matrix.clone().mul_small_rhs(decomposed_rhs),
            reveal_plaintext: lhs.reveal_plaintext && rhs.reveal_plaintext,
        }
    }

    pub fn small_scalar_mul(&self, input: &BggPublicKeyWire, scalar: &Mat) -> BggPublicKeyWire {
        // Small-scalar multiplication is the direct relation A_out = A * t;
        // the scalar does not require gadget decomposition.
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
        let decomposed = target.clone().decompose(self.base.clone(), self.digit_count.clone());
        // For an arbitrary target T, this computes A G^-1(T) by the relation
        // G K_T = T.  T is only a consumed matrix target, not a claim that it
        // is itself a canonical gadget encoding.
        BggPublicKeyWire {
            matrix: input.matrix.clone().mul_small_rhs(decomposed),
            reveal_plaintext: input.reveal_plaintext,
        }
    }

    pub(crate) fn large_scalar_mul_with_decomposition(
        &self,
        input: &BggPublicKeyWire,
        decomposed: Preimage,
    ) -> BggPublicKeyWire {
        BggPublicKeyWire {
            matrix: input.matrix.clone().mul_small_rhs(decomposed),
            reveal_plaintext: input.reveal_plaintext,
        }
    }

    pub(crate) fn large_scalar_decomposition(
        &self,
        input: &BggPublicKeyWire,
        scalar: &Mat,
    ) -> Preimage {
        let rows = input.matrix.matrix_type().rows.clone();
        let gadget = self.ring.gadget(rows, self.base.clone(), self.digit_count.clone());
        // A large scalar t must be converted into a gadget-carried target tG.
        // Decomposing `t * G` therefore yields K_t with G K_t = tG, so applying
        // K_t has exactly the intended scalar action on the public matrix.
        (scalar.clone() * gadget).decompose(self.base.clone(), self.digit_count.clone())
    }
}

#[derive(Clone)]
pub struct BggPublicKeySampler {
    pub layout: BggSamplerLayout,
}

impl BggPublicKeySampler {
    /// Samples one packed public matrix and exposes a dynamically sized family of slices.
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

    /// Samples the packed public matrices once and exposes deterministic slices.
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
    use mxx_ir_core::{ParamEnv, node::NodeKind};
    use mxx_primitives::{
        matrix::{PolyMatrix, PolyMatrixSmallRhs, dcrt_poly::DCRTPolyMatrix},
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
        let kinds = built
            .graph
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .map(|node| node.kind())
            .collect::<Vec<_>>();
        assert_eq!(
            kinds.iter().filter(|kind| matches!(kind, NodeKind::MatrixMulSmallRhs)).count(),
            1,
            "public-key multiplication must consume the bounded decomposition directly",
        );
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
        assert_eq!(
            matrix_output(&result, "mul"),
            &lhs_value
                .clone()
                .multiply_small_rhs(&rhs_value.clone().gadget_decompose(false).unwrap())
                .unwrap()
        );
        assert_eq!(
            matrix_output(&result, "matrix-mul"),
            &lhs_value
                .clone()
                .multiply_small_rhs(&target_value.clone().gadget_decompose(false).unwrap())
                .unwrap()
        );
    }
}
