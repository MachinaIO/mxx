use mxx_ir_core::{GraphBuilder, IntExpr, MatrixWire, node::MatrixBinaryOp, types::MatrixType};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BggPublicKeyWire {
    pub matrix: MatrixWire,
    pub reveal_plaintext: bool,
}

#[derive(Clone, Debug)]
pub struct BggPublicKeyCompiler {
    pub base: IntExpr,
    pub decomposed_type: MatrixType,
}

impl BggPublicKeyCompiler {
    pub fn add(
        &self,
        builder: &mut GraphBuilder,
        lhs: &BggPublicKeyWire,
        rhs: &BggPublicKeyWire,
    ) -> BggPublicKeyWire {
        BggPublicKeyWire {
            matrix: builder.matrix_binary(
                MatrixBinaryOp::Add,
                &lhs.matrix,
                &rhs.matrix,
                lhs.matrix.matrix_type.clone(),
            ),
            reveal_plaintext: lhs.reveal_plaintext && rhs.reveal_plaintext,
        }
    }

    pub fn sub(
        &self,
        builder: &mut GraphBuilder,
        lhs: &BggPublicKeyWire,
        rhs: &BggPublicKeyWire,
    ) -> BggPublicKeyWire {
        BggPublicKeyWire {
            matrix: builder.matrix_binary(
                MatrixBinaryOp::Subtract,
                &lhs.matrix,
                &rhs.matrix,
                lhs.matrix.matrix_type.clone(),
            ),
            reveal_plaintext: lhs.reveal_plaintext && rhs.reveal_plaintext,
        }
    }

    /// Reproduces `BggPublicKey::mul`: `lhs.mul_decompose(rhs)`.
    pub fn mul(
        &self,
        builder: &mut GraphBuilder,
        lhs: &BggPublicKeyWire,
        rhs: &BggPublicKeyWire,
    ) -> BggPublicKeyWire {
        let decomposed =
            builder.gadget_decompose(&rhs.matrix, self.base.clone(), self.decomposed_type.clone());
        BggPublicKeyWire {
            matrix: builder.matrix_binary(
                MatrixBinaryOp::Multiply,
                &lhs.matrix,
                &decomposed,
                lhs.matrix.matrix_type.clone(),
            ),
            reveal_plaintext: lhs.reveal_plaintext && rhs.reveal_plaintext,
        }
    }

    pub fn small_scalar_mul(
        &self,
        builder: &mut GraphBuilder,
        input: &BggPublicKeyWire,
        scalar: &MatrixWire,
    ) -> BggPublicKeyWire {
        BggPublicKeyWire {
            matrix: builder.matrix_binary(
                MatrixBinaryOp::Multiply,
                &input.matrix,
                scalar,
                input.matrix.matrix_type.clone(),
            ),
            reveal_plaintext: input.reveal_plaintext,
        }
    }

    /// Reproduces `BggPublicKey::large_scalar_mul`, including its explicit
    /// gadget construction and decomposition.
    pub fn large_scalar_mul(
        &self,
        builder: &mut GraphBuilder,
        input: &BggPublicKeyWire,
        scalar: &MatrixWire,
    ) -> BggPublicKeyWire {
        let decomposed = self.large_scalar_decomposition(builder, input, scalar);
        self.large_scalar_mul_with_decomposition(builder, input, &decomposed)
    }

    /// Reproduces `BggPublicKey::matrix_mul`: `A * G^-1(target)`.
    pub fn matrix_mul(
        &self,
        builder: &mut GraphBuilder,
        input: &BggPublicKeyWire,
        target: &MatrixWire,
    ) -> BggPublicKeyWire {
        let decomposed_type = MatrixType {
            modulus: target.matrix_type.modulus.clone(),
            ring_dimension: target.matrix_type.ring_dimension.clone(),
            rows: input.matrix.matrix_type.columns.clone(),
            columns: target.matrix_type.columns.clone(),
        };
        let decomposed =
            builder.gadget_decompose(target, self.base.clone(), decomposed_type.clone());
        let output_type = MatrixType {
            rows: input.matrix.matrix_type.rows.clone(),
            columns: target.matrix_type.columns.clone(),
            ..input.matrix.matrix_type.clone()
        };
        BggPublicKeyWire {
            matrix: builder.matrix_binary(
                MatrixBinaryOp::Multiply,
                &input.matrix,
                &decomposed,
                output_type,
            ),
            reveal_plaintext: input.reveal_plaintext,
        }
    }

    pub(crate) fn large_scalar_mul_with_decomposition(
        &self,
        builder: &mut GraphBuilder,
        input: &BggPublicKeyWire,
        decomposed: &MatrixWire,
    ) -> BggPublicKeyWire {
        BggPublicKeyWire {
            matrix: builder.matrix_binary(
                MatrixBinaryOp::Multiply,
                &input.matrix,
                decomposed,
                input.matrix.matrix_type.clone(),
            ),
            reveal_plaintext: input.reveal_plaintext,
        }
    }

    pub(crate) fn large_scalar_decomposition(
        &self,
        builder: &mut GraphBuilder,
        input: &BggPublicKeyWire,
        scalar: &MatrixWire,
    ) -> MatrixWire {
        let gadget_type = MatrixType {
            rows: input.matrix.matrix_type.rows.clone(),
            columns: self.decomposed_type.rows.clone(),
            ..input.matrix.matrix_type.clone()
        };
        let gadget = builder.constant_matrix(
            gadget_type.clone(),
            mxx_ir_core::node::ConstantMatrix::Gadget { base: self.base.clone(), small: false },
        );
        let scalar_gadget =
            builder.matrix_binary(MatrixBinaryOp::Multiply, &gadget, scalar, gadget_type.clone());
        let scalar_decomposed_type = MatrixType {
            rows: gadget_type.columns.clone(),
            columns: gadget_type.columns.clone(),
            ..gadget_type
        };
        builder.gadget_decompose(&scalar_gadget, self.base.clone(), scalar_decomposed_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix_type(rows: i64, columns: i64) -> MatrixType {
        MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(8),
            rows: IntExpr::constant(rows),
            columns: IntExpr::constant(columns),
        }
    }

    #[test]
    fn reveal_metadata_matches_direct_public_key_operations() {
        let compiler = BggPublicKeyCompiler {
            base: IntExpr::constant(2),
            decomposed_type: matrix_type(10, 10),
        };
        for lhs_revealed in [false, true] {
            for rhs_revealed in [false, true] {
                let mut builder = GraphBuilder::new("public-key-reveal", Vec::new());
                let lhs = BggPublicKeyWire {
                    matrix: builder.input("lhs", matrix_type(2, 10)),
                    reveal_plaintext: lhs_revealed,
                };
                let rhs = BggPublicKeyWire {
                    matrix: builder.input("rhs", matrix_type(2, 10)),
                    reveal_plaintext: rhs_revealed,
                };
                let expected = lhs_revealed && rhs_revealed;
                assert_eq!(compiler.add(&mut builder, &lhs, &rhs).reveal_plaintext, expected);
                assert_eq!(compiler.sub(&mut builder, &lhs, &rhs).reveal_plaintext, expected);
                assert_eq!(compiler.mul(&mut builder, &lhs, &rhs).reveal_plaintext, expected);
            }
        }

        for revealed in [false, true] {
            let mut builder = GraphBuilder::new("public-key-scalar-reveal", Vec::new());
            let input = BggPublicKeyWire {
                matrix: builder.input("input", matrix_type(2, 10)),
                reveal_plaintext: revealed,
            };
            let scalar = builder.input("scalar", matrix_type(1, 1));
            assert_eq!(
                compiler.small_scalar_mul(&mut builder, &input, &scalar).reveal_plaintext,
                revealed
            );
            assert_eq!(
                compiler.large_scalar_mul(&mut builder, &input, &scalar).reveal_plaintext,
                revealed
            );
        }
    }
}
