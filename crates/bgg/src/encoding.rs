//! Declarative BGG+ encoding graph values.

use crate::BggPublicKeyWire;
use mxx_dsl::{DslError, GraphValue, GraphValueSchema, Mat, MatType, Pending, Preimage, Ring};
use mxx_ir_core::{
    IntExpr, RealExpr, ValueHandle, WireType,
    node::{ConcatAxis, IndexRange},
};
use rayon::prelude::*;
use thiserror::Error;

/// Worst-case BGG product error for a scalar plaintext with the given coefficient l1 norm.
/// The final term is `x_L * s * (A_R - G D(A_R))`, sharing the actual decomposition residual.
/// For binary plaintexts pass one; no Gaussian independence assumption is made.
pub fn multiplication_error_bound<P: mxx_primitives::poly::PolyParams>(
    params: &P,
    secret_dimension: usize,
    left_error: &num_bigint::BigUint,
    right_error: &num_bigint::BigUint,
    secret_bound: &num_bigint::BigUint,
    plaintext_l1_bound: &num_bigint::BigUint,
) -> num_bigint::BigUint {
    let n = num_bigint::BigUint::from(params.ring_dimension());
    let columns = num_bigint::BigUint::from(secret_dimension) * params.modulus_digits();
    let digit_bound = num_bigint::BigUint::from(1u8) << (params.base_bits() - 1);
    columns * &n * left_error * digit_bound +
        plaintext_l1_bound *
            (right_error +
                n * secret_dimension * secret_bound * params.gadget_error_bound(None))
}

#[derive(Clone)]
pub struct BggEncodingWire {
    pub vector: Mat,
    pub plaintext: Option<Mat>,
}

#[derive(Clone, PartialEq)]
pub struct BggEncodingType {
    pub vector: MatType,
    pub plaintext: Option<MatType>,
}

impl GraphValue for BggEncodingWire {
    type Schema = BggEncodingType;

    fn flatten(&self) -> Vec<ValueHandle> {
        let mut values = self.vector.flatten();
        if let Some(plaintext) = &self.plaintext {
            values.extend(plaintext.flatten());
        }
        values
    }

    fn pending(&self) -> Pending {
        Pending::merge(
            std::iter::once(self.vector.pending())
                .chain(self.plaintext.as_ref().map(GraphValue::pending)),
        )
    }

    fn schema(&self) -> Self::Schema {
        BggEncodingType {
            vector: self.vector.schema(),
            plaintext: self.plaintext.as_ref().map(GraphValue::schema),
        }
    }

    fn from_values(
        schema: &Self::Schema,
        values: &[ValueHandle],
        pending: Pending,
    ) -> Result<Self, DslError> {
        let vector_count = schema.vector.wire_types().len();
        let expected = vector_count + usize::from(schema.plaintext.is_some());
        if values.len() != expected {
            return Err(DslError::Schema);
        }
        Ok(Self {
            vector: Mat::from_values(&schema.vector, &values[..vector_count], pending.clone())?,
            plaintext: schema
                .plaintext
                .as_ref()
                .map(|ty| Mat::from_values(ty, &values[vector_count..], pending))
                .transpose()?,
        })
    }
}

impl GraphValueSchema for BggEncodingType {
    type Value = BggEncodingWire;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        BggEncodingWire {
            vector: self.vector.placeholders_from(next),
            plaintext: self.plaintext.as_ref().map(|plaintext| plaintext.placeholders_from(next)),
        }
    }

    fn wire_types(&self) -> Vec<WireType> {
        let mut types = self.vector.wire_types();
        if let Some(plaintext) = &self.plaintext {
            types.extend(plaintext.wire_types());
        }
        types
    }
}

/// Compiler for online encoding carriers.
///
/// This is intentionally state-free: public-key compilation and gadget
/// decomposition belong to preprocessing.  Online multiplication methods
/// receive the already cached typed `Preimage` explicitly.
#[derive(Clone, Copy, Default)]
pub struct BggEncodingCompiler;

#[derive(Debug, Error, Eq, PartialEq)]
pub enum EncodingCompileError {
    #[error("BGG+ multiplication requires the left operand plaintext")]
    MissingLeftPlaintext,
}

impl BggEncodingCompiler {
    /// Adds encoding carriers.  Public-key arithmetic is deliberately absent:
    /// it belongs to the preprocessing public-key program, not this online
    /// encoding value.
    pub fn add(
        &self,
        lhs: &BggEncodingWire,
        rhs: &BggEncodingWire,
    ) -> Result<BggEncodingWire, EncodingCompileError> {
        Ok(BggEncodingWire {
            vector: &lhs.vector + &rhs.vector,
            plaintext: binary_plaintext(lhs, rhs, |left, right| left + right),
        })
    }

    pub fn sub(
        &self,
        lhs: &BggEncodingWire,
        rhs: &BggEncodingWire,
    ) -> Result<BggEncodingWire, EncodingCompileError> {
        Ok(BggEncodingWire {
            vector: &lhs.vector - &rhs.vector,
            plaintext: binary_plaintext(lhs, rhs, |left, right| left - right),
        })
    }

    /// Builds `c_L K_R + c_R x_L`, where `K_R` is the preprocessing-supplied
    /// typed decomposition of the public RHS.  This is the sole multiplication
    /// input needed by online encoding.
    pub fn mul(
        &self,
        lhs: &BggEncodingWire,
        rhs: &BggEncodingWire,
        decomposed_rhs: Preimage,
    ) -> Result<BggEncodingWire, EncodingCompileError> {
        let plaintext = lhs.plaintext.clone().ok_or(EncodingCompileError::MissingLeftPlaintext)?;
        Ok(BggEncodingWire {
            vector: decomposed_rhs.mul_small_rhs(lhs.vector.clone()) + &rhs.vector * plaintext,
            plaintext: binary_plaintext(lhs, rhs, |left, right| left * right),
        })
    }

    pub fn small_scalar_mul(&self, input: &BggEncodingWire, scalar: &Mat) -> BggEncodingWire {
        BggEncodingWire {
            vector: &input.vector * scalar,
            plaintext: input.plaintext.clone().map(|value| value * scalar),
        }
    }

    /// Applies a preprocessing-supplied scalar decomposition to an encoding.
    pub fn large_scalar_mul(
        &self,
        input: &BggEncodingWire,
        scalar: &Mat,
        decomposed: Preimage,
    ) -> BggEncodingWire {
        BggEncodingWire {
            vector: decomposed.mul_small_rhs(input.vector.clone()),
            plaintext: input.plaintext.clone().map(|value| value * scalar),
        }
    }

    /// Applies a preprocessing-supplied decomposition for an arbitrary target.
    pub fn matrix_mul(&self, input: &BggEncodingWire, decomposed: Preimage) -> BggEncodingWire {
        // This is an explicit right action by an arbitrary target matrix.  Its
        // decomposition is used only to consume the input carrier; it does not
        // assert that the projected target itself is a canonical G encoding.
        BggEncodingWire { vector: decomposed.mul_small_rhs(input.vector.clone()), plaintext: None }
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
    #[error("BGG+ Gaussian sampling requires both a sigma and an explicit coefficient cutoff")]
    MissingGaussianBound,
    #[error(transparent)]
    Dsl(#[from] mxx_dsl::DslError),
}

#[derive(Clone)]
pub struct BggEncodingSampler {
    pub layout: BggSamplerLayout,
    pub gaussian_sigma: Option<RealExpr>,
    pub gaussian_max_coefficient_bound: Option<IntExpr>,
}

impl BggEncodingSampler {
    /// Builds `s_mask A - ([1|x_1|...|x_t] tensor (s_payload G)) + e` and
    /// exposes its column slices. Omitting the payload secret reuses the mask
    /// secret for the ordinary one-secret construction.
    pub fn sample(
        &self,
        mask_secret: Mat,
        payload_secret: Option<Mat>,
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
        let payload_secret = payload_secret.unwrap_or_else(|| mask_secret.clone());
        if !same_matrix_type(mask_secret.matrix_type(), &secret_type) ||
            !same_matrix_type(payload_secret.matrix_type(), &secret_type) ||
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
        let packed_vector = mask_secret * all_public_keys -
            encoded_plaintexts.tensor(payload_secret * gadget) +
            match (&self.gaussian_sigma, &self.gaussian_max_coefficient_bound) {
                (Some(sigma), Some(bound)) => {
                    ring.gaussian((1, columns * count), sigma.clone(), bound.clone())
                }
                (None, None) => ring.zero((1, columns * count)),
                _ => return Err(BggSampleError::MissingGaussianBound),
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
        matrix::{PolyMatrix, PolyMatrixSmallRhs, dcrt_poly::DCRTPolyMatrix},
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
        let encoding = BggEncodingType { vector: matrix.clone(), plaintext: Some(matrix) };
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
        let compiler = BggEncodingCompiler;
        for left_revealed in [false, true] {
            for right_revealed in [false, true] {
                let encoding = |prefix: &str, revealed: bool| BggEncodingWire {
                    vector: ring.input(format!("{prefix}-vector"), (1, 4)),
                    plaintext: revealed.then(|| ring.input(format!("{prefix}-plain"), (1, 1))),
                };
                let left = encoding("left", left_revealed);
                let right = encoding("right", right_revealed);
                let expected = left_revealed && right_revealed;
                for output in
                    [compiler.add(&left, &right).unwrap(), compiler.sub(&left, &right).unwrap()]
                {
                    assert_eq!(output.plaintext.is_some(), expected);
                }
                let decomposition = ring.preimage_input("rhs-decomposition", (4, 4), 2);
                match compiler.mul(&left, &right, decomposition) {
                    Ok(output) => {
                        assert!(left_revealed);
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
        let compiler = BggEncodingCompiler;
        let encoding = |prefix: &str| BggEncodingWire {
            vector: ring.input(format!("{prefix}-vector"), (1, 8)),
            plaintext: Some(ring.input(format!("{prefix}-plaintext"), (1, 1))),
        };
        let decomposition = ring.preimage_input("rhs-decomposition", (8, 8), 2);
        let product =
            compiler.mul(&encoding("left"), &encoding("right"), decomposition).expect("product");
        let built = DslContext::new("bgg-encoding-mul")
            .output("vector", product.vector)
            .expect("vector output")
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
            0
        );
        assert_eq!(
            kinds.iter().filter(|kind| matches!(kind, NodeKind::MatrixMulSmallRhs)).count(),
            1,
            "only the encoding vector consumes the typed decomposition"
        );
        assert!(!kinds.iter().any(|kind| matches!(kind, NodeKind::MatrixScale { .. })));

        built.validate(&ParamEnv::default()).expect("valid executable graph");
    }

    #[test]
    fn runtime_multiplication_matches_the_bgg_encoding_formula() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4, None, None);
        let digit_count = parameters.modulus_digits();
        let columns = 2 * digit_count;
        let ring = Ring::new(
            BigInt::from(parameters.modulus().as_ref().clone()),
            parameters.ring_dimension() as usize,
        );
        let compiler = BggEncodingCompiler;
        let encoding = |prefix: &str| BggEncodingWire {
            vector: ring.input(format!("{prefix}-vector"), (1, columns)),
            plaintext: Some(ring.input(format!("{prefix}-plaintext"), (1, 1))),
        };
        let decomposition = ring.preimage_input("rhs-decomposition", (2 * digit_count, columns), 8);
        let output = compiler.mul(&encoding("lhs"), &encoding("rhs"), decomposition).unwrap();
        let graph = DslContext::new("bgg-encoding-runtime")
            .output("vector", output.vector)
            .unwrap()
            .output("plaintext", output.plaintext.unwrap())
            .unwrap()
            .build()
            .unwrap();

        let lhs_vector = row(&parameters, columns, 0);
        let rhs_vector = row(&parameters, columns, 1);
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
                (
                    "rhs-decomposition".to_owned(),
                    RuntimeValue::small_matrix(
                        rhs_public.clone().gadget_decompose(false, None).unwrap(),
                    ),
                ),
                ("lhs-plaintext".to_owned(), RuntimeValue::matrix(lhs_plaintext.clone())),
                ("rhs-plaintext".to_owned(), RuntimeValue::matrix(rhs_plaintext.clone())),
            ]),
        );

        let expected_vector = lhs_vector
            .clone()
            .multiply_small_rhs(&rhs_public.clone().gadget_decompose(false, None).unwrap())
            .unwrap() +
            rhs_vector * lhs_plaintext.entry(0, 0);
        assert_eq!(matrix_output(&result, "vector"), &expected_vector);
        assert_eq!(matrix_output(&result, "plaintext"), &(lhs_plaintext * rhs_plaintext));
    }

    #[test]
    fn cached_multiplication_keeps_public_projection_lazy() {
        let ring = Ring::new(257, 8);
        let compiler = BggEncodingCompiler;
        let lhs = BggEncodingWire {
            vector: ring.input("cached-lhs-vector", (1, 4)),
            plaintext: Some(ring.input("cached-lhs-plaintext", (1, 1))),
        };
        let rhs = BggEncodingWire {
            vector: ring.input("cached-rhs-vector", (1, 4)),
            plaintext: Some(ring.input("cached-rhs-plaintext", (1, 1))),
        };
        let decomposition = ring.preimage_input("cached-rhs-decomposition", (4, 4), 2);
        let output = compiler.mul(&lhs, &rhs, decomposition).expect("cached artifact");
        let built = DslContext::new("bgg-encoding-cached-mul")
            .output("vector", output.vector)
            .expect("vector output")
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
            0
        );
        assert_eq!(
            kinds.iter().filter(|kind| matches!(kind, NodeKind::MatrixMulSmallRhs)).count(),
            1,
            "only the vector-side cached action is emitted; G*K_out stays lazy"
        );
        built.validate(&ParamEnv::default()).expect("valid cached graph");
    }

    #[test]
    fn runtime_approximate_multiplication_accounts_for_secret_weighted_residual() {
        use mxx_primitives::sampler::{
            PolyUniformSampler, bounds::matrix_within_coefficient_bound,
            uniform::DCRTPolyUniformSampler,
        };
        use num_bigint::BigUint;
        for dropped in [1, 2] {
            let parameters = DCRTPolyParams::new(4, 3, 17, 4, None, Some(dropped));
            let columns = parameters.modulus_digits();
            let layout = concrete_layout(&parameters, 1);
            let ring = layout.ring();
            let compiler = BggEncodingCompiler;
            let encoding = |prefix: &str| BggEncodingWire {
                vector: ring.input(format!("{prefix}-vector"), (1, columns)),
                plaintext: Some(ring.identity(1)),
            };
            let left_public = ring.input("left-public", (1, columns));
            let right_public = ring.input("right-public", (1, columns));
            let decomposition = right_public.clone().decompose(layout.gadget_base.clone(), columns);
            let public = decomposition.clone().mul_small_rhs(left_public);
            let output =
                compiler.mul(&encoding("left"), &encoding("right"), decomposition).unwrap();
            let graph = DslContext::new("approximate-bgg-product")
                .output("vector", output.vector)
                .unwrap()
                .output("public", public)
                .unwrap()
                .build()
                .unwrap();
            let sampler = DCRTPolyUniformSampler::new();
            let left = sampler.sample_uniform(&parameters, 1, columns, DistType::FinRingDist);
            let right = sampler.sample_uniform(&parameters, 1, columns, DistType::FinRingDist);
            let gadget = DCRTPolyMatrix::gadget_matrix(&parameters, 1, None);
            let secret = DCRTPoly::from_usize_to_constant(&parameters, 2);
            let left_ciphertext = (&left - &gadget) * secret.clone();
            let right_ciphertext = (&right - &gadget) * secret.clone();
            let result = execute_graph(
                graph,
                parameters.clone(),
                BTreeMap::from([
                    ("left-vector".into(), RuntimeValue::matrix(left_ciphertext)),
                    ("right-vector".into(), RuntimeValue::matrix(right_ciphertext)),
                    ("left-public".into(), RuntimeValue::matrix(left)),
                    ("right-public".into(), RuntimeValue::matrix(right.clone())),
                ]),
            );
            let ideal = (matrix_output(&result, "public") - &gadget) * secret.clone();
            let actual_error = matrix_output(&result, "vector") - &ideal;
            let residual = &right - &(&gadget * right.decompose());
            assert_eq!(actual_error, residual * secret);
            let bound = multiplication_error_bound(
                &parameters,
                1,
                &BigUint::from(0u8),
                &BigUint::from(0u8),
                &BigUint::from(2u8),
                &BigUint::from(1u8),
            );
            assert!(matrix_within_coefficient_bound(&actual_error, &bound));
        }
    }
    #[test]
    fn bgg_sampling_builds_a_packed_executable_graph() {
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
        let encodings = BggEncodingSampler {
            layout,
            gaussian_sigma: Some(3.into()),
            gaussian_max_coefficient_bound: Some(19.into()),
        }
        .sample(
            ring.input("secret", (1, 2)),
            None,
            &public_keys,
            &[ring.input("plaintext", (1, 1))],
        )
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
        built.validate(&ParamEnv::default()).expect("valid executable graph");
    }
    #[test]
    fn payload_secret_none_reuses_the_mask_secret() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4, None, None);
        let layout = concrete_layout(&parameters, 2);
        let ring = layout.ring();
        let public_keys = BggPublicKeySampler { layout: layout.clone() }.sample(
            ring.bytes_input("key", 32),
            b"bgg-shared-secret".to_vec(),
            &[],
        );
        let sampler = BggEncodingSampler {
            layout,
            gaussian_sigma: None,
            gaussian_max_coefficient_bound: None,
        };
        let shared =
            sampler.sample(ring.input("shared-secret", (1, 2)), None, &public_keys, &[]).unwrap();
        let explicit = sampler
            .sample(
                ring.input("explicit-mask-secret", (1, 2)),
                Some(ring.input("explicit-payload-secret", (1, 2))),
                &public_keys,
                &[],
            )
            .unwrap();
        let graph = DslContext::new("bgg-shared-secret-fallback")
            .output("shared", shared[0].vector.clone())
            .unwrap()
            .output("explicit", explicit[0].vector.clone())
            .unwrap()
            .build()
            .unwrap();
        let secret_value = secret(&parameters, 2);
        let result = execute_graph(
            graph,
            parameters,
            BTreeMap::from([
                ("key".to_owned(), RuntimeValue::Bytes([7u8; 32].to_vec())),
                ("shared-secret".to_owned(), RuntimeValue::matrix(secret_value.clone())),
                ("explicit-mask-secret".to_owned(), RuntimeValue::matrix(secret_value.clone())),
                ("explicit-payload-secret".to_owned(), RuntimeValue::matrix(secret_value)),
            ]),
        );
        assert_eq!(matrix_output(&result, "shared"), matrix_output(&result, "explicit"));
    }

    #[test]
    fn runtime_public_keys_and_encodings_match_the_bgg_sampling_formula() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4, None, None);
        let layout = concrete_layout(&parameters, 2);
        let key = [23u8; 32];
        let tag = b"bgg-ir-sampler";
        let ring = layout.ring();
        let public_keys = BggPublicKeySampler { layout: layout.clone() }.sample(
            ring.bytes_input("key", key.len()),
            tag.to_vec(),
            &[false, true],
        );
        let encodings = BggEncodingSampler {
            layout: layout.clone(),
            gaussian_sigma: None,
            gaussian_max_coefficient_bound: None,
        }
        .sample(
            ring.input("mask-secret", (1, layout.secret_dimension)),
            Some(ring.input("payload-secret", (1, layout.secret_dimension))),
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

        let mask_secret_value = secret(&parameters, layout.secret_dimension);
        let payload_secret_value = DCRTPolyMatrix::from_poly_vec_row(
            &parameters,
            (0..layout.secret_dimension)
                .map(|index| {
                    DCRTPoly::const_rotate_poly(
                        &parameters,
                        (index + 1) % parameters.ring_dimension() as usize,
                    )
                })
                .collect(),
        );
        let plaintext_values = [scalar(&parameters, 2), scalar(&parameters, 3)];
        let result = execute_graph(
            graph,
            parameters.clone(),
            BTreeMap::from([
                ("key".to_owned(), RuntimeValue::Bytes(key.to_vec())),
                ("mask-secret".to_owned(), RuntimeValue::matrix(mask_secret_value.clone())),
                ("payload-secret".to_owned(), RuntimeValue::matrix(payload_secret_value.clone())),
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
        let gadget = DCRTPolyMatrix::gadget_matrix(&parameters, layout.secret_dimension, None);
        let encoded_plaintexts = DCRTPolyMatrix::from_poly_vec_row(
            &parameters,
            vec![
                DCRTPoly::const_one(&parameters),
                plaintext_values[0].entry(0, 0),
                plaintext_values[1].entry(0, 0),
            ],
        );
        let vectors = mask_secret_value * packed.clone() -
            encoded_plaintexts.tensor(&(payload_secret_value * gadget));
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
