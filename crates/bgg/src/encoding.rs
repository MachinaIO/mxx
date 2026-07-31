use crate::{BggPublicKeyCompiler, BggPublicKeyWire};
use mxx_ir_core::{GraphBuilder, MatrixWire, node::MatrixBinaryOp};
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
    #[error("BGG+ multiplication requires the left operand plaintext")]
    MissingLeftPlaintext,
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
            plaintext: componentwise_plaintext(builder, lhs, rhs, MatrixBinaryOp::Add),
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
            plaintext: componentwise_plaintext(builder, lhs, rhs, MatrixBinaryOp::Subtract),
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
            None => return Err(EncodingCompileError::MissingLeftPlaintext),
        };
        let vector = builder.matrix_binary(
            MatrixBinaryOp::Add,
            &first,
            &second,
            lhs.vector.matrix_type.clone(),
        );
        let plaintext = componentwise_plaintext(builder, lhs, rhs, MatrixBinaryOp::Multiply);
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
        let decomposed = self.public_key.large_scalar_decomposition(builder, &input.pubkey, scalar);
        BggEncodingWire {
            vector: builder.matrix_binary(
                MatrixBinaryOp::Multiply,
                &input.vector,
                &decomposed,
                input.vector.matrix_type.clone(),
            ),
            pubkey: self.public_key.large_scalar_mul_with_decomposition(
                builder,
                &input.pubkey,
                &decomposed,
            ),
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

    /// Reproduces `BggEncoding::matrix_mul`. The direct implementation drops
    /// plaintext availability after multiplying by an arbitrary matrix.
    pub fn matrix_mul(
        &self,
        builder: &mut GraphBuilder,
        input: &BggEncodingWire,
        target: &MatrixWire,
    ) -> BggEncodingWire {
        let decomposed_type = mxx_ir_core::types::MatrixType {
            modulus: target.matrix_type.modulus.clone(),
            ring_dimension: target.matrix_type.ring_dimension.clone(),
            rows: input.vector.matrix_type.columns.clone(),
            columns: target.matrix_type.columns.clone(),
        };
        let decomposed =
            builder.gadget_decompose(target, self.public_key.base.clone(), decomposed_type.clone());
        let vector_type = mxx_ir_core::types::MatrixType {
            rows: input.vector.matrix_type.rows.clone(),
            columns: target.matrix_type.columns.clone(),
            ..input.vector.matrix_type.clone()
        };
        BggEncodingWire {
            vector: builder.matrix_binary(
                MatrixBinaryOp::Multiply,
                &input.vector,
                &decomposed,
                vector_type,
            ),
            pubkey: self.public_key.matrix_mul(builder, &input.pubkey, target),
            plaintext: None,
        }
    }
}

fn componentwise_plaintext(
    builder: &mut GraphBuilder,
    lhs: &BggEncodingWire,
    rhs: &BggEncodingWire,
    operation: MatrixBinaryOp,
) -> Option<MatrixWire> {
    match (&lhs.plaintext, &rhs.plaintext) {
        (Some(lhs), Some(rhs)) => {
            Some(builder.matrix_binary(operation, lhs, rhs, lhs.matrix_type.clone()))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::{
        IntExpr, artifact::ArtifactConfidentiality, node::NodeKind, types::MatrixType,
    };

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
            pubkey: BggPublicKeyWire {
                matrix: builder.input("lhs_pubkey", matrix_type(2, 10)),
                reveal_plaintext: true,
            },
            plaintext: Some(builder.input("lhs_plaintext", matrix_type(1, 1))),
        };
        let rhs = BggEncodingWire {
            vector: builder.input("rhs_vector", matrix_type(1, 10)),
            pubkey: BggPublicKeyWire {
                matrix: builder.input("rhs_pubkey", matrix_type(2, 10)),
                reveal_plaintext: true,
            },
            plaintext: Some(builder.input("rhs_plaintext", matrix_type(1, 1))),
        };
        let compiler = BggEncodingCompiler {
            public_key: BggPublicKeyCompiler {
                base: IntExpr::constant(2),
                decomposed_type: matrix_type(10, 10),
            },
        };
        let output = compiler.mul(&mut builder, &lhs, &rhs).expect("compatible bundles");
        builder.output("vector", &output.vector, ArtifactConfidentiality::Public);
        builder.output("pubkey", &output.pubkey.matrix, ArtifactConfidentiality::Public);
        builder.output(
            "plaintext",
            output.plaintext.as_ref().expect("plaintext"),
            ArtifactConfidentiality::Public,
        );
        let graph = builder.finish();

        assert_eq!(graph.nodes.len(), 16);
        assert_eq!(
            graph.nodes.iter().filter(|node| matches!(node.kind, NodeKind::Output { .. })).count(),
            3
        );
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

    #[test]
    fn reveal_combinations_match_the_direct_encoding_contract() {
        let compiler = BggEncodingCompiler {
            public_key: BggPublicKeyCompiler {
                base: IntExpr::constant(2),
                decomposed_type: matrix_type(10, 10),
            },
        };
        for lhs_revealed in [false, true] {
            for rhs_revealed in [false, true] {
                let mut builder = GraphBuilder::new("encoding-reveal-contract", Vec::new());
                let lhs = BggEncodingWire {
                    vector: builder.input("lhs_vector", matrix_type(1, 10)),
                    pubkey: BggPublicKeyWire {
                        matrix: builder.input("lhs_pubkey", matrix_type(2, 10)),
                        reveal_plaintext: lhs_revealed,
                    },
                    plaintext: lhs_revealed
                        .then(|| builder.input("lhs_plaintext", matrix_type(1, 1))),
                };
                let rhs = BggEncodingWire {
                    vector: builder.input("rhs_vector", matrix_type(1, 10)),
                    pubkey: BggPublicKeyWire {
                        matrix: builder.input("rhs_pubkey", matrix_type(2, 10)),
                        reveal_plaintext: rhs_revealed,
                    },
                    plaintext: rhs_revealed
                        .then(|| builder.input("rhs_plaintext", matrix_type(1, 1))),
                };
                let expected_reveal = lhs_revealed && rhs_revealed;
                for output in [
                    compiler.add(&mut builder, &lhs, &rhs).expect("addition"),
                    compiler.sub(&mut builder, &lhs, &rhs).expect("subtraction"),
                ] {
                    assert_eq!(output.pubkey.reveal_plaintext, expected_reveal);
                    assert_eq!(output.plaintext.is_some(), expected_reveal);
                }
                match compiler.mul(&mut builder, &lhs, &rhs) {
                    Ok(output) => {
                        assert!(lhs_revealed);
                        assert_eq!(output.pubkey.reveal_plaintext, expected_reveal);
                        assert_eq!(output.plaintext.is_some(), expected_reveal);
                    }
                    Err(error) => {
                        assert!(!lhs_revealed);
                        assert_eq!(error, EncodingCompileError::MissingLeftPlaintext);
                    }
                }
            }
        }
    }

    trait NodeWire {
        fn into_wire(&self) -> mxx_ir_core::WireRef;
    }

    impl NodeWire for mxx_ir_core::node::Node {
        fn into_wire(&self) -> mxx_ir_core::WireRef {
            mxx_ir_core::WireRef { node: self.id, port: mxx_ir_core::Port(0) }
        }
    }
}
