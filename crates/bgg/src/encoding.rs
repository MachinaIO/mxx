use crate::{BggPublicKeyCompiler, BggPublicKeyWire, GraphBuilder, MatrixWire};
use mxx_graph_ir::node::MatrixBinaryOp;
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BggEncodingWire {
    pub vector: MatrixWire,
    pub pubkey: BggPublicKeyWire,
    pub plaintext: Option<MatrixWire>,
}

#[derive(Clone, Debug)]
pub struct BggEncodingCompiler {
    pub public_key: BggPublicKeyCompiler,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum EncodingCompileError {
    #[error("BGG+ operands must either both reveal plaintext or both hide it")]
    PlaintextMismatch,
}

impl BggEncodingCompiler {
    pub fn add(
        &self,
        builder: &mut GraphBuilder,
        lhs: &BggEncodingWire,
        rhs: &BggEncodingWire,
    ) -> Result<BggEncodingWire, EncodingCompileError> {
        Ok(BggEncodingWire {
            vector: builder.matrix_binary(
                MatrixBinaryOp::Add,
                &lhs.vector,
                &rhs.vector,
                lhs.vector.matrix_type.clone(),
            ),
            pubkey: self.public_key.add(builder, &lhs.pubkey, &rhs.pubkey),
            plaintext: componentwise_plaintext(builder, lhs, rhs, MatrixBinaryOp::Add)?,
        })
    }

    pub fn sub(
        &self,
        builder: &mut GraphBuilder,
        lhs: &BggEncodingWire,
        rhs: &BggEncodingWire,
    ) -> Result<BggEncodingWire, EncodingCompileError> {
        Ok(BggEncodingWire {
            vector: builder.matrix_binary(
                MatrixBinaryOp::Subtract,
                &lhs.vector,
                &rhs.vector,
                lhs.vector.matrix_type.clone(),
            ),
            pubkey: self.public_key.sub(builder, &lhs.pubkey, &rhs.pubkey),
            plaintext: componentwise_plaintext(builder, lhs, rhs, MatrixBinaryOp::Subtract)?,
        })
    }

    /// Reproduces `BggEncoding::mul` without changing its operand order:
    /// `vector_L * G^-1(pubkey_R) + vector_R * plaintext_L`.
    pub fn mul(
        &self,
        builder: &mut GraphBuilder,
        lhs: &BggEncodingWire,
        rhs: &BggEncodingWire,
    ) -> Result<BggEncodingWire, EncodingCompileError> {
        let decomposed_rhs = builder.gadget_decompose(
            &rhs.pubkey.matrix,
            self.public_key.base.clone(),
            self.public_key.decomposed_type.clone(),
        );
        let first = builder.matrix_binary(
            MatrixBinaryOp::Multiply,
            &lhs.vector,
            &decomposed_rhs,
            lhs.vector.matrix_type.clone(),
        );
        let second = match &lhs.plaintext {
            Some(plaintext) => builder.matrix_binary(
                MatrixBinaryOp::Multiply,
                &rhs.vector,
                plaintext,
                rhs.vector.matrix_type.clone(),
            ),
            None => return Err(EncodingCompileError::PlaintextMismatch),
        };
        let vector = builder.matrix_binary(
            MatrixBinaryOp::Add,
            &first,
            &second,
            lhs.vector.matrix_type.clone(),
        );
        let plaintext = componentwise_plaintext(builder, lhs, rhs, MatrixBinaryOp::Multiply)?;
        Ok(BggEncodingWire {
            vector,
            pubkey: self.public_key.mul(builder, &lhs.pubkey, &rhs.pubkey),
            plaintext,
        })
    }

    pub fn small_scalar_mul(
        &self,
        builder: &mut GraphBuilder,
        input: &BggEncodingWire,
        scalar: &MatrixWire,
    ) -> BggEncodingWire {
        BggEncodingWire {
            vector: builder.matrix_binary(
                MatrixBinaryOp::Multiply,
                &input.vector,
                scalar,
                input.vector.matrix_type.clone(),
            ),
            pubkey: self.public_key.small_scalar_mul(builder, &input.pubkey, scalar),
            plaintext: input.plaintext.as_ref().map(|plaintext| {
                builder.matrix_binary(
                    MatrixBinaryOp::Multiply,
                    plaintext,
                    scalar,
                    plaintext.matrix_type.clone(),
                )
            }),
        }
    }

    pub fn large_scalar_mul(
        &self,
        builder: &mut GraphBuilder,
        input: &BggEncodingWire,
        scalar: &MatrixWire,
    ) -> BggEncodingWire {
        let gadget_type = mxx_graph_ir::types::MatrixType {
            rows: input.pubkey.matrix.matrix_type.rows.clone(),
            columns: self.public_key.decomposed_type.rows.clone(),
            ..input.pubkey.matrix.matrix_type.clone()
        };
        let gadget = builder.constant_matrix(
            gadget_type.clone(),
            mxx_graph_ir::node::ConstantMatrix::Gadget {
                base: self.public_key.base.clone(),
                small: false,
            },
        );
        let scalar_gadget =
            builder.matrix_binary(MatrixBinaryOp::Multiply, &gadget, scalar, gadget_type.clone());
        let scalar_decomposed_type = mxx_graph_ir::types::MatrixType {
            rows: gadget_type.columns.clone(),
            columns: gadget_type.columns.clone(),
            ..gadget_type
        };
        let decomposed = builder.gadget_decompose(
            &scalar_gadget,
            self.public_key.base.clone(),
            scalar_decomposed_type,
        );
        BggEncodingWire {
            vector: builder.matrix_binary(
                MatrixBinaryOp::Multiply,
                &input.vector,
                &decomposed,
                input.vector.matrix_type.clone(),
            ),
            pubkey: self.public_key.large_scalar_mul(builder, &input.pubkey, scalar),
            plaintext: input.plaintext.as_ref().map(|plaintext| {
                builder.matrix_binary(
                    MatrixBinaryOp::Multiply,
                    plaintext,
                    scalar,
                    plaintext.matrix_type.clone(),
                )
            }),
        }
    }
}

fn componentwise_plaintext(
    builder: &mut GraphBuilder,
    lhs: &BggEncodingWire,
    rhs: &BggEncodingWire,
    operation: MatrixBinaryOp,
) -> Result<Option<MatrixWire>, EncodingCompileError> {
    match (&lhs.plaintext, &rhs.plaintext) {
        (Some(lhs), Some(rhs)) => {
            Ok(Some(builder.matrix_binary(operation, lhs, rhs, lhs.matrix_type.clone())))
        }
        (None, None) => Ok(None),
        _ => Err(EncodingCompileError::PlaintextMismatch),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_graph_ir::{IntExpr, node::NodeKind, types::MatrixType};

    fn matrix_type(rows: i64, columns: i64) -> MatrixType {
        MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(8),
            rows: IntExpr::constant(rows),
            columns: IntExpr::constant(columns),
        }
    }

    #[test]
    fn multiplication_matches_the_existing_encoding_formula() {
        let mut builder = GraphBuilder::new("bgg-mul", Vec::new());
        let lhs = BggEncodingWire {
            vector: builder.input("lhs_vector", matrix_type(1, 10)),
            pubkey: BggPublicKeyWire { matrix: builder.input("lhs_pubkey", matrix_type(2, 10)) },
            plaintext: Some(builder.input("lhs_plaintext", matrix_type(1, 1))),
        };
        let rhs = BggEncodingWire {
            vector: builder.input("rhs_vector", matrix_type(1, 10)),
            pubkey: BggPublicKeyWire { matrix: builder.input("rhs_pubkey", matrix_type(2, 10)) },
            plaintext: Some(builder.input("rhs_plaintext", matrix_type(1, 1))),
        };
        let compiler = BggEncodingCompiler {
            public_key: BggPublicKeyCompiler {
                base: IntExpr::constant(2),
                decomposed_type: matrix_type(10, 10),
            },
        };
        let output = compiler.mul(&mut builder, &lhs, &rhs).expect("compatible bundles");
        builder.output("vector", &output.vector);
        builder.output("pubkey", &output.pubkey.matrix);
        builder.output("plaintext", output.plaintext.as_ref().expect("plaintext"));
        let graph = builder.finish();

        assert_eq!(graph.nodes.len(), 13);
        assert!(matches!(graph.nodes[6].kind, NodeKind::GadgetDecompose { small: false, .. }));
        assert_eq!(graph.nodes[6].args, vec![rhs.pubkey.matrix.wire]);
        assert!(matches!(graph.nodes[7].kind, NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)));
        assert_eq!(graph.nodes[7].args, vec![lhs.vector.wire, graph.nodes[6].into_wire()]);
        assert!(matches!(graph.nodes[8].kind, NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)));
        assert_eq!(
            graph.nodes[8].args,
            vec![rhs.vector.wire, lhs.plaintext.as_ref().expect("plaintext").wire]
        );
        assert!(matches!(graph.nodes[9].kind, NodeKind::MatrixBinary(MatrixBinaryOp::Add)));
        assert!(matches!(graph.nodes[10].kind, NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)));
        assert!(matches!(graph.nodes[11].kind, NodeKind::GadgetDecompose { small: false, .. }));
        assert!(matches!(graph.nodes[12].kind, NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)));
    }

    trait NodeWire {
        fn into_wire(&self) -> mxx_graph_ir::WireRef;
    }

    impl NodeWire for mxx_graph_ir::node::Node {
        fn into_wire(&self) -> mxx_graph_ir::WireRef {
            mxx_graph_ir::WireRef { node: self.id, port: mxx_graph_ir::Port(0) }
        }
    }
}
