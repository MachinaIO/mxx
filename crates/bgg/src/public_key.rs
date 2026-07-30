use crate::{GraphBuilder, MatrixWire};
use mxx_ir_core::{IntExpr, node::MatrixBinaryOp, types::MatrixType};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BggPublicKeyWire {
    pub matrix: MatrixWire,
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
        let decomposed =
            builder.gadget_decompose(&scalar_gadget, self.base.clone(), scalar_decomposed_type);
        BggPublicKeyWire {
            matrix: builder.matrix_binary(
                MatrixBinaryOp::Multiply,
                &input.matrix,
                &decomposed,
                input.matrix.matrix_type.clone(),
            ),
        }
    }
}
