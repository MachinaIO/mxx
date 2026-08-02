//! Declarative BGG+ encoding graph values.

use crate::{BggPublicKeyCompiler, BggPublicKeyType, BggPublicKeyWire};
use mxx_dsl::{DslError, GraphValue, GraphValueSchema, Mat, MatType, Pending, Ring};
use mxx_ir_core::{
    IntExpr, RealExpr, ValueHandle, WireType,
    node::{ConcatAxis, IndexRange},
};
use rayon::prelude::*;
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BggSamplerLayout {
    pub modulus: IntExpr,
    pub ring_dimension: IntExpr,
    pub secret_dimension: usize,
    pub digit_count: usize,
    pub gadget_base: IntExpr,
}

impl BggSamplerLayout {
    pub fn ring(&self) -> Ring {
        Ring::new(self.modulus.clone(), self.ring_dimension.clone())
    }

    pub fn public_key_columns(&self) -> usize {
        self.secret_dimension
            .checked_mul(self.digit_count)
            .expect("BGG+ public-key column count overflow")
    }
}

#[derive(Debug, Error)]
pub enum BggSampleError {
    #[error("BGG+ sampling requires public_keys.len() == plaintexts.len() + 1")]
    InputCountMismatch,
    #[error("BGG+ sampler received an incompatible matrix type")]
    MatrixTypeMismatch,
    #[error("BGG+ sampler families must have matching slot counts")]
    SlotCountMismatch,
    #[error(transparent)]
    Dsl(#[from] mxx_dsl::DslError),
}

#[derive(Clone)]
pub struct BggEncodingSampler {
    pub layout: BggSamplerLayout,
    pub gaussian_sigma: Option<RealExpr>,
}

impl BggEncodingSampler {
    /// Builds the packed relation `sA - ([1|x_1|...|x_t] tensor sG) + e`, then
    /// exposes its column slices. This preserves the executable dataflow of the
    /// original sampler; the symbolic layer represents Concat and Tensor
    /// directly without changing the runtime formula.
    pub fn sample(
        &self,
        secret: Mat,
        public_keys: &[BggPublicKeyWire],
        plaintexts: &[Mat],
    ) -> Result<Vec<BggEncodingWire>, BggSampleError> {
        if public_keys.len() != plaintexts.len() + 1 {
            return Err(BggSampleError::InputCountMismatch);
        }
        let count = public_keys.len();
        let columns = self.layout.public_key_columns();
        let ring = self.layout.ring();
        let secret_type = ring.matrix_type((1, self.layout.secret_dimension));
        let public_key_type = ring.matrix_type((self.layout.secret_dimension, columns));
        let plaintext_type = ring.matrix_type((1, 1));
        if !same_matrix_type(secret.matrix_type(), &secret_type) ||
            public_keys
                .par_iter()
                .any(|key| !same_matrix_type(key.matrix.matrix_type(), &public_key_type)) ||
            plaintexts
                .par_iter()
                .any(|plaintext| !same_matrix_type(plaintext.matrix_type(), &plaintext_type))
        {
            return Err(BggSampleError::MatrixTypeMismatch);
        }
        let all_public_keys = Mat::concat(
            ConcatAxis::Columns,
            public_keys.iter().map(|key| key.matrix.clone()).collect(),
        );
        let one = ring.identity(1);
        let mut extended_plaintexts = Vec::with_capacity(count);
        extended_plaintexts.push(one);
        extended_plaintexts.extend(plaintexts.iter().cloned());
        let encoded_plaintexts = Mat::concat(ConcatAxis::Columns, extended_plaintexts.clone());
        let gadget = ring.gadget(
            self.layout.secret_dimension,
            self.layout.gadget_base.clone(),
            self.layout.digit_count,
        );
        let packed_vector = secret.clone() * all_public_keys -
            encoded_plaintexts.tensor(secret.clone() * gadget) +
            match &self.gaussian_sigma {
                Some(sigma) => ring.gaussian((1, columns * count), sigma.clone()),
                None => ring.zero((1, columns * count)),
            };
        Ok((0..count)
            .map(|index| BggEncodingWire {
                vector: packed_vector.clone().slice(
                    None,
                    Some(IndexRange {
                        start: (columns * index).into(),
                        end: (columns * (index + 1)).into(),
                    }),
                ),
                pubkey: public_keys[index].clone(),
                plaintext: public_keys[index]
                    .reveal_plaintext
                    .then(|| extended_plaintexts[index].clone()),
            })
            .collect())
    }
}

pub(crate) fn same_matrix_type(
    lhs: &mxx_ir_core::types::MatrixType,
    rhs: &mxx_ir_core::types::MatrixType,
) -> bool {
    lhs.modulus.canonicalize() == rhs.modulus.canonicalize() &&
        lhs.ring_dimension.canonicalize() == rhs.ring_dimension.canonicalize() &&
        lhs.rows.canonicalize() == rhs.rows.canonicalize() &&
        lhs.columns.canonicalize() == rhs.columns.canonicalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BggPublicKeySampler,
        test_utils::{execute_graph, matrix_output, row},
    };
    use mxx_dsl::{DslContext, Ring, Subgraph};
    use mxx_ir_core::{
        ParamEnv,
        node::{ConcatAxis, NodeKind},
    };
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        sampler::{DistType, PolyHashSampler, hash::DCRTPolyHashSampler},
    };
    use mxx_runtime::RuntimeValue;
    use num_bigint::BigInt;
    use std::collections::BTreeMap;

    fn concrete_layout(parameters: &DCRTPolyParams, secret_dimension: usize) -> BggSamplerLayout {
        BggSamplerLayout {
            modulus: IntExpr::constant(BigInt::from(parameters.modulus().as_ref().clone())),
            ring_dimension: IntExpr::constant(parameters.ring_dimension()),
            secret_dimension,
            digit_count: parameters.modulus_digits(),
            gadget_base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
        }
    }
    fn scalar(parameters: &DCRTPolyParams, rotation: usize) -> DCRTPolyMatrix {
        DCRTPolyMatrix::from_poly_vec(
            parameters,
            vec![vec![DCRTPoly::const_rotate_poly(parameters, rotation)]],
        )
    }
    fn secret(parameters: &DCRTPolyParams, dimension: usize) -> DCRTPolyMatrix {
        DCRTPolyMatrix::from_poly_vec_row(
            parameters,
            (0..dimension)
                .map(|index| {
                    DCRTPoly::const_rotate_poly(
                        parameters,
                        index % parameters.ring_dimension() as usize,
                    )
                })
                .collect(),
        )
    }

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
    #[test]
    fn bgg_sampling_elaborates_without_aggregate_atoms() {
        let layout = BggSamplerLayout {
            modulus: 257.into(),
            ring_dimension: 8.into(),
            secret_dimension: 2,
            digit_count: 4,
            gadget_base: 4.into(),
        };
        let ring = layout.ring();
        let public_keys = BggPublicKeySampler { layout: layout.clone() }.sample(
            ring.bytes_input("hash-key", 32),
            b"bgg-test".to_vec(),
            &[true],
        );
        let encodings = BggEncodingSampler { layout, gaussian_sigma: Some(3.into()) }
            .sample(ring.input("secret", (1, 2)), &public_keys, &[ring.input("plaintext", (1, 1))])
            .expect("compatible sampler inputs");
        let built = DslContext::new("bgg-sampling")
            .private_output("constant", encodings[0].vector.clone())
            .expect("constant output")
            .private_output("message", encodings[1].vector.clone())
            .expect("message output")
            .build()
            .expect("build");
        let concat_count = built
            .graph
            .root_scope()
            .nodes()
            .iter()
            .filter(|node| matches!(node.kind(), NodeKind::Concat { axis: ConcatAxis::Columns }))
            .count();
        let tensor_count = built
            .graph
            .root_scope()
            .nodes()
            .iter()
            .filter(|node| matches!(node.kind(), NodeKind::Tensor))
            .count();
        let gaussian_types = built
            .graph
            .root_scope()
            .nodes()
            .iter()
            .filter_map(|node| match node.kind() {
                NodeKind::GaussianSample { matrix_type, .. } => Some(matrix_type),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(concat_count, 2, "packed public keys and packed plaintext row");
        assert_eq!(tensor_count, 1, "one packed plaintext/secret-gadget tensor");
        assert_eq!(gaussian_types.len(), 1, "one packed error sample");
        assert_eq!(gaussian_types[0].columns.canonicalize(), IntExpr::constant(16));
        let elaborated = built.elaborate(&ParamEnv::default()).expect("symbolic elaboration");
        assert_eq!(elaborated.outputs.len(), 2);
        assert!(!elaborated.atoms.is_empty());
        let report = mxx_noise_simulator::simulate(&elaborated).expect("noise simulation");
        for name in ["constant", "message"] {
            let estimate = &report.outputs[name];
            assert!(estimate.has_signal, "{name} retains the BGG signal term");
            assert!(
                estimate.noise.as_ref().is_some_and(|noise| noise.bound > 0),
                "{name} retains sampled error in the noise estimate"
            );
        }
    }
    #[test]
    fn runtime_public_keys_and_encodings_match_the_bgg_sampling_formula() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let layout = concrete_layout(&parameters, 2);
        let key = [23u8; 32];
        let tag = b"bgg-ir-sampler";
        let ring = layout.ring();
        let public_keys = BggPublicKeySampler { layout: layout.clone() }.sample(
            ring.bytes_input("key", key.len()),
            tag.to_vec(),
            &[false, true],
        );
        let encodings = BggEncodingSampler { layout: layout.clone(), gaussian_sigma: None }
            .sample(
                ring.input("secret", (1, layout.secret_dimension)),
                &public_keys,
                &[ring.input("plaintext-0", (1, 1)), ring.input("plaintext-1", (1, 1))],
            )
            .unwrap();
        let mut context = DslContext::new("bgg-sampler-runtime");
        for index in 0..public_keys.len() {
            context = context
                .output(format!("public-{index}"), public_keys[index].matrix.clone())
                .unwrap()
                .output(format!("vector-{index}"), encodings[index].vector.clone())
                .unwrap();
        }
        let graph = context.build().unwrap();

        let secret_value = secret(&parameters, layout.secret_dimension);
        let plaintext_values = [scalar(&parameters, 2), scalar(&parameters, 3)];
        let result = execute_graph(
            graph,
            parameters.clone(),
            BTreeMap::from([
                ("key".to_owned(), RuntimeValue::Bytes(key.to_vec())),
                ("secret".to_owned(), RuntimeValue::matrix(secret_value.clone())),
                ("plaintext-0".to_owned(), RuntimeValue::matrix(plaintext_values[0].clone())),
                ("plaintext-1".to_owned(), RuntimeValue::matrix(plaintext_values[1].clone())),
            ]),
        );

        let packed = DCRTPolyHashSampler::<keccak_asm::Keccak256>::new().sample_hash(
            &parameters,
            key,
            tag,
            layout.secret_dimension,
            layout.public_key_columns() * public_keys.len(),
            DistType::FinRingDist,
        );
        let gadget = DCRTPolyMatrix::gadget_matrix(&parameters, layout.secret_dimension);
        let encoded_plaintexts = DCRTPolyMatrix::from_poly_vec_row(
            &parameters,
            vec![
                DCRTPoly::const_one(&parameters),
                plaintext_values[0].entry(0, 0),
                plaintext_values[1].entry(0, 0),
            ],
        );
        let vectors = secret_value.clone() * packed.clone() -
            encoded_plaintexts.tensor(&(secret_value * gadget));
        for index in 0..public_keys.len() {
            let start = layout.public_key_columns() * index;
            let end = layout.public_key_columns() * (index + 1);
            assert_eq!(
                matrix_output(&result, &format!("public-{index}")),
                &packed.slice_columns(start, end)
            );
            assert_eq!(
                matrix_output(&result, &format!("vector-{index}")),
                &vectors.slice_columns(start, end)
            );
        }
        assert!(encodings[0].plaintext.is_some());
        assert!(encodings[1].plaintext.is_none());
        assert!(encodings[2].plaintext.is_some());
    }
}
