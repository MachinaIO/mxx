//! Declarative BGG+ public-key and encoding samplers.

use crate::{
    BggEncodingWire, BggPolyEncodingWire, BggPublicKeyFamily, BggPublicKeyWire,
    NaiveBggEncodingVecWire, NaiveBggPublicKeyVecWire,
};
use mxx_dsl::{Bytes, Family, HashTag, Mat, Parallel, Ring, parallel_zip};
use mxx_ir_core::{
    IntExpr, RealExpr,
    node::{ConcatAxis, IndexRange},
};
use rayon::prelude::*;
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BggSamplerLayout {
    pub modulus: IntExpr,
    pub ring_dimension: IntExpr,
    pub secret_dimension: usize,
    pub digit_count: usize,
    pub gadget_base: IntExpr,
    pub gaussian_max_coefficient_bound: IntExpr,
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

#[derive(Clone)]
pub struct BggPublicKeySampler {
    pub layout: BggSamplerLayout,
}

#[derive(Clone)]
pub struct BggEncodingSampler {
    pub layout: BggSamplerLayout,
    pub gaussian_sigma: Option<RealExpr>,
}

#[derive(Clone)]
pub struct BggPolyEncodingSampler {
    pub layout: BggSamplerLayout,
    pub gaussian_sigma: Option<RealExpr>,
}

#[derive(Clone)]
pub struct NaiveBggPublicKeyVecSampler {
    pub layout: BggSamplerLayout,
    pub slot_count: IntExpr,
}

#[derive(Clone)]
pub struct NaiveBggEncodingVecSampler {
    pub scalar: BggEncodingSampler,
}

#[derive(Clone)]
pub struct BggPolyEncodingSample {
    pub encodings: Vec<BggPolyEncodingWire>,
    pub slot_secret_matrices: Family<Mat>,
}

#[derive(Debug, Error)]
pub enum BggSampleError {
    #[error("BGG+ sampling requires public_keys.len() == plaintexts.len() + 1")]
    InputCountMismatch,
    #[error("BGG+ sampler received an incompatible matrix type")]
    MatrixTypeMismatch,
    #[error("BGG+ polynomial sampler families must have matching slot counts")]
    SlotCountMismatch,
    #[error(transparent)]
    Dsl(#[from] mxx_dsl::DslError),
}

impl BggPolyEncodingSampler {
    pub fn sample(
        &self,
        secret: Mat,
        public_keys: &[BggPublicKeyWire],
        plaintexts: &[Family<Mat>],
        slot_count: IntExpr,
        supplied_slot_secrets: Option<Family<Mat>>,
    ) -> Result<BggPolyEncodingSample, BggSampleError> {
        if public_keys.len() != plaintexts.len() + 1 {
            return Err(BggSampleError::InputCountMismatch);
        }
        let ring = self.layout.ring();
        let secret_size = self.layout.secret_dimension;
        if !same_matrix_type(secret.matrix_type(), &ring.matrix_type((1, secret_size))) ||
            public_keys.par_iter().any(|key| {
                !same_matrix_type(
                    key.matrix.matrix_type(),
                    &ring.matrix_type((secret_size, self.layout.public_key_columns())),
                )
            }) ||
            plaintexts.par_iter().any(|family| {
                family.count() != &slot_count ||
                    !same_matrix_type(family.element_type(), &ring.matrix_type((1, 1)))
            })
        {
            return Err(BggSampleError::MatrixTypeMismatch);
        }
        let (slot_secrets, transformed_secrets) = if let Some(slot_secrets) = supplied_slot_secrets
        {
            if slot_secrets.count() != &slot_count ||
                !same_matrix_type(
                    slot_secrets.element_type(),
                    &ring.matrix_type((secret_size, secret_size)),
                )
            {
                return Err(BggSampleError::SlotCountMismatch);
            }
            let transformed = slot_secrets.clone().parallel_map({
                let secret = secret.clone();
                move |_, slot_secret| secret.clone() * slot_secret
            })?;
            (slot_secrets, transformed)
        } else {
            let (transformed, sampled) = Parallel::range(slot_count.clone()).map_values({
                let ring = ring.clone();
                let secret = secret.clone();
                move |_| {
                    let slot_secret = ring.uniform_interval((secret_size, secret_size), -1, 1);
                    (secret.clone() * slot_secret.clone(), slot_secret)
                }
            })?;
            (sampled, transformed)
        };
        let ones = transformed_secrets.clone().parallel_map({
            let ring = ring.clone();
            move |_, _| ring.identity(1)
        })?;
        let plaintext_rows = plaintexts.iter().cloned().reduce(|left, right| {
            left.parallel_zip(right, |_, left, right| {
                Mat::concat(ConcatAxis::Columns, vec![left, right])
            })
            .expect("validated family counts")
        });
        let encoded_plaintexts = match plaintext_rows {
            Some(rows) => ones.clone().parallel_zip(rows, |_, one, row| {
                Mat::concat(ConcatAxis::Columns, vec![one, row])
            })?,
            None => ones.clone(),
        };
        let packed_public = Mat::concat(
            ConcatAxis::Columns,
            public_keys.iter().map(|key| key.matrix.clone()).collect(),
        );
        let count = public_keys.len();
        let columns = self.layout.public_key_columns();
        let gadget =
            ring.gadget(secret_size, self.layout.gadget_base.clone(), self.layout.digit_count);
        let sigma = self.gaussian_sigma.clone();
        let vector_families = parallel_zip(
            (transformed_secrets, encoded_plaintexts),
            move |_, (secret, encoded_plaintexts)| {
                let packed = secret.clone() * packed_public.clone() -
                    encoded_plaintexts.tensor(secret * gadget.clone()) +
                    match &sigma {
                        Some(sigma) => ring.gaussian(
                            (1, columns * count),
                            sigma.clone(),
                            self.layout.gaussian_max_coefficient_bound.clone(),
                        ),
                        None => ring.zero((1, columns * count)),
                    };
                (0..count)
                    .map(|index| {
                        packed.clone().slice(
                            None,
                            Some(IndexRange {
                                start: (columns * index).into(),
                                end: (columns * (index + 1)).into(),
                            }),
                        )
                    })
                    .collect::<Vec<_>>()
            },
        )?;
        let encodings = vector_families
            .into_iter()
            .enumerate()
            .map(|(index, vectors)| BggPolyEncodingWire {
                vectors,
                pubkey: public_keys[index].clone(),
                plaintexts: public_keys[index]
                    .reveal_plaintext
                    .then(|| if index == 0 { ones.clone() } else { plaintexts[index - 1].clone() }),
            })
            .collect();
        Ok(BggPolyEncodingSample { encodings, slot_secret_matrices: slot_secrets })
    }
}

impl NaiveBggPublicKeyVecSampler {
    pub fn sample(
        &self,
        hash_key: Bytes,
        tag: &[u8],
        reveal_plaintexts: &[bool],
    ) -> Result<Vec<NaiveBggPublicKeyVecWire>, BggSampleError> {
        let outputs = (0..=reveal_plaintexts.len())
            .map(|output| {
                let reveal = output == 0 || reveal_plaintexts[output - 1];
                let packed_count = if output == 0 { 1 } else { 2 };
                let mut prefix = tag.to_vec();
                prefix.extend_from_slice(&(output as u64).to_le_bytes());
                let family = Parallel::range(self.slot_count.clone()).map({
                    let ring = self.layout.ring();
                    let key = hash_key.clone();
                    move |slot| {
                        let mut tag = HashTag::from(prefix.clone());
                        tag.push(slot);
                        let packed = ring.hash_matrix(
                            key.clone(),
                            tag,
                            (
                                self.layout.secret_dimension,
                                self.layout.public_key_columns() * packed_count,
                            ),
                        );
                        if output == 0 {
                            packed
                        } else {
                            packed.slice(
                                None,
                                Some(IndexRange {
                                    start: self.layout.public_key_columns().into(),
                                    end: (self.layout.public_key_columns() * 2).into(),
                                }),
                            )
                        }
                    }
                })?;
                Ok(NaiveBggPublicKeyVecWire { matrices: family, reveal_plaintext: reveal })
            })
            .collect::<Result<Vec<_>, BggSampleError>>()?;
        Ok(outputs)
    }
}

impl NaiveBggEncodingVecSampler {
    pub fn sample(
        &self,
        secret: Mat,
        public_keys: &[NaiveBggPublicKeyVecWire],
        plaintexts: &[Family<Mat>],
    ) -> Result<Vec<NaiveBggEncodingVecWire>, BggSampleError> {
        if public_keys.len() != plaintexts.len() + 1 {
            return Err(BggSampleError::InputCountMismatch);
        }
        let slot_count = public_keys[0].matrices.count().clone();
        if public_keys.par_iter().any(|key| key.matrices.count() != &slot_count) ||
            plaintexts.par_iter().any(|value| value.count() != &slot_count)
        {
            return Err(BggSampleError::SlotCountMismatch);
        }
        let layout = &self.scalar.layout;
        let ring = layout.ring();
        if !same_matrix_type(secret.matrix_type(), &ring.matrix_type((1, layout.secret_dimension))) ||
            public_keys.par_iter().any(|key| {
                !same_matrix_type(
                    key.matrices.element_type(),
                    &ring.matrix_type((layout.secret_dimension, layout.public_key_columns())),
                )
            }) ||
            plaintexts
                .par_iter()
                .any(|value| !same_matrix_type(value.element_type(), &ring.matrix_type((1, 1))))
        {
            return Err(BggSampleError::MatrixTypeMismatch);
        }
        let outputs = (0..public_keys.len())
            .map(|output| {
                let vectors = if output == 0 {
                    public_keys[0].matrices.clone().parallel_map({
                        let sampler = self.scalar.clone();
                        let secret = secret.clone();
                        let reveal = public_keys[0].reveal_plaintext;
                        move |_, one_key| {
                            sampler
                                .sample(
                                    secret.clone(),
                                    &[BggPublicKeyWire {
                                        matrix: one_key,
                                        reveal_plaintext: reveal,
                                    }],
                                    &[],
                                )
                                .expect("validated scalar sampler")
                                .remove(0)
                                .vector
                        }
                    })?
                } else {
                    parallel_zip(
                        (
                            public_keys[0].matrices.clone(),
                            public_keys[output].matrices.clone(),
                            plaintexts[output - 1].clone(),
                        ),
                        {
                            let sampler = self.scalar.clone();
                            let secret = secret.clone();
                            let one_reveal = public_keys[0].reveal_plaintext;
                            let reveal = public_keys[output].reveal_plaintext;
                            move |_, (one_key, key, plaintext)| {
                                sampler
                                    .sample(
                                        secret.clone(),
                                        &[
                                            BggPublicKeyWire {
                                                matrix: one_key,
                                                reveal_plaintext: one_reveal,
                                            },
                                            BggPublicKeyWire {
                                                matrix: key,
                                                reveal_plaintext: reveal,
                                            },
                                        ],
                                        &[plaintext],
                                    )
                                    .expect("validated scalar sampler")
                                    .remove(1)
                                    .vector
                            }
                        },
                    )?
                };
                Ok(NaiveBggEncodingVecWire {
                    vectors,
                    pubkeys: public_keys[output].matrices.clone(),
                    pubkey_reveal_plaintext: public_keys[output].reveal_plaintext,
                    plaintexts: public_keys[output].reveal_plaintext.then(|| {
                        if output == 0 {
                            public_keys[0]
                                .matrices
                                .clone()
                                .parallel_map({
                                    let ring = self.scalar.layout.ring();
                                    move |_, _| ring.identity(1)
                                })
                                .expect("family")
                        } else {
                            plaintexts[output - 1].clone()
                        }
                    }),
                })
            })
            .collect::<Result<Vec<_>, BggSampleError>>()?;
        Ok(outputs)
    }
}

impl BggPublicKeySampler {
    /// Samples one packed public matrix and exposes a dynamically sized family of deterministic
    /// slices with the caller-supplied symbolic column count. Every member reveals its plaintext
    /// relation.
    pub fn sample_family(
        &self,
        hash_key: Bytes,
        tag: impl Into<HashTag>,
        count: impl Into<IntExpr>,
        public_key_columns: impl Into<IntExpr>,
    ) -> Result<BggPublicKeyFamily, mxx_dsl::DslError> {
        let count = count.into();
        let columns = public_key_columns.into();
        let packed = self.layout.ring().hash_matrix(
            hash_key,
            tag,
            (
                IntExpr::constant(self.layout.secret_dimension),
                IntExpr::Mul(Box::new(columns.clone()), Box::new(count.clone())),
            ),
        );
        let matrices = Parallel::range(count).map_values(|index| {
            let start = IntExpr::Mul(Box::new(columns.clone()), Box::new(index.expression()));
            let slice = packed.clone().slice(
                None,
                Some(IndexRange {
                    start: start.clone(),
                    end: IntExpr::Add(Box::new(start), Box::new(columns.clone())),
                }),
            );
            slice
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
        let packed = self.layout.ring().hash_matrix(
            hash_key,
            tag,
            (self.layout.secret_dimension, columns * count),
        );
        (0..count)
            .map(|index| BggPublicKeyWire {
                matrix: packed.clone().slice(
                    None,
                    Some(IndexRange {
                        start: (columns * index).into(),
                        end: (columns * (index + 1)).into(),
                    }),
                ),
                reveal_plaintext: index == 0 || reveal_plaintexts[index - 1],
            })
            .collect()
    }
}

impl BggEncodingSampler {
    /// Builds the packed relation `sA - ([1|x_1|...|x_t] tensor sG) + e`, then
    /// exposes its column slices. This preserves the executable dataflow of the
    /// original sampler. Concat and Tensor remain ordinary executable nodes.
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
                Some(sigma) => ring.gaussian(
                    (1, columns * count),
                    sigma.clone(),
                    self.layout.gaussian_max_coefficient_bound.clone(),
                ),
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

fn same_matrix_type(
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
    use crate::test_utils::{execute_graph, matrix_output};
    use mxx_dsl::DslContext;
    use mxx_ir_core::ParamEnv;
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        sampler::{DistType, PolyHashSampler, hash::DCRTPolyHashSampler},
    };
    use mxx_runtime::{
        RuntimeValue,
        artifact::MemoryArtifactStore,
        backend::poly::cpu_backend,
        execute,
        transcript::{SamplingMode, TranscriptRecorder},
    };
    use num_bigint::BigInt;
    use std::collections::BTreeMap;

    fn concrete_layout(parameters: &DCRTPolyParams, secret_dimension: usize) -> BggSamplerLayout {
        BggSamplerLayout {
            modulus: IntExpr::constant(BigInt::from(parameters.modulus().as_ref().clone())),
            ring_dimension: IntExpr::constant(parameters.ring_dimension()),
            secret_dimension,
            digit_count: parameters.modulus_digits(),
            gadget_base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
            gaussian_max_coefficient_bound: 30.into(),
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
    fn polynomial_and_naive_sampler_entrypoints_build_and_elaborate() {
        let layout = BggSamplerLayout {
            modulus: 257.into(),
            ring_dimension: 8.into(),
            secret_dimension: 1,
            digit_count: 2,
            gadget_base: 4.into(),
            gaussian_max_coefficient_bound: 30.into(),
        };
        let ring = layout.ring();
        let public_keys = BggPublicKeySampler { layout: layout.clone() }.sample(
            ring.bytes_input("packed-key", 32),
            b"packed".to_vec(),
            &[true],
        );
        let public_key_columns = IntExpr::Var("public-key-columns".to_owned());
        let public_key_family = BggPublicKeySampler { layout: layout.clone() }
            .sample_family(
                ring.bytes_input("packed-family-key", 32),
                b"packed-family".to_vec(),
                3,
                public_key_columns.clone(),
            )
            .unwrap();
        assert_eq!(public_key_family.matrices.element_type().columns, public_key_columns);
        let plaintexts = Family::pack(
            (0..2).map(|slot| ring.input(format!("plaintext-{slot}"), (1, 1))).collect(),
        )
        .unwrap();
        let poly =
            BggPolyEncodingSampler { layout: layout.clone(), gaussian_sigma: Some(3.into()) }
                .sample(
                    ring.input("poly-secret", (1, 1)),
                    &public_keys,
                    std::slice::from_ref(&plaintexts),
                    2.into(),
                    None,
                )
                .unwrap();
        let naive_keys =
            NaiveBggPublicKeyVecSampler { layout: layout.clone(), slot_count: 2.into() }
                .sample(ring.bytes_input("naive-key", 32), b"naive", &[true])
                .unwrap();
        let naive = NaiveBggEncodingVecSampler {
            scalar: BggEncodingSampler { layout, gaussian_sigma: Some(3.into()) },
        }
        .sample(ring.input("naive-secret", (1, 1)), &naive_keys, std::slice::from_ref(&plaintexts))
        .unwrap();
        let built = DslContext::new("bgg-family-samplers")
            .int_parameter("public-key-columns")
            .family_output("public-keys", public_key_family.matrices)
            .unwrap()
            .family_output("poly", poly.encodings[1].vectors.clone())
            .unwrap()
            .family_output("poly-secrets", poly.slot_secret_matrices)
            .unwrap()
            .family_output("naive", naive[1].vectors.clone())
            .unwrap()
            .build()
            .unwrap();
        built
            .validate(&ParamEnv {
                integers: BTreeMap::from([("public-key-columns".to_owned(), 5.into())]),
                ..ParamEnv::default()
            })
            .unwrap();
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

    fn supplied_slot_secret_graph(
        layout: &BggSamplerLayout,
        sigma: Option<RealExpr>,
    ) -> mxx_dsl::BuiltGraph {
        let ring = layout.ring();
        let public_keys = [
            BggPublicKeyWire {
                matrix: ring
                    .input("public-0", (layout.secret_dimension, layout.public_key_columns())),
                reveal_plaintext: true,
            },
            BggPublicKeyWire {
                matrix: ring
                    .input("public-1", (layout.secret_dimension, layout.public_key_columns())),
                reveal_plaintext: false,
            },
        ];
        let plaintexts = Family::pack(
            (0..2).map(|slot| ring.input(format!("plaintext-{slot}"), (1, 1))).collect(),
        )
        .unwrap();
        let slot_secrets = Family::pack(
            (0..2)
                .map(|slot| {
                    ring.input(
                        format!("slot-secret-{slot}"),
                        (layout.secret_dimension, layout.secret_dimension),
                    )
                })
                .collect(),
        )
        .unwrap();
        let sample = BggPolyEncodingSampler { layout: layout.clone(), gaussian_sigma: sigma }
            .sample(
                ring.input("secret", (1, layout.secret_dimension)),
                &public_keys,
                &[plaintexts],
                2.into(),
                Some(slot_secrets.clone()),
            )
            .unwrap();
        assert!(sample.encodings[1].plaintexts.is_none());
        let mut context = DslContext::new("bgg-poly-supplied-secret-runtime");
        for slot in 0..2 {
            context = context
                .output(format!("vector-{slot}"), sample.encodings[1].vectors.get_static(slot))
                .unwrap();
        }
        context.build().unwrap()
    }

    #[test]
    fn supplied_slot_secrets_preserve_order_and_zero_sigma_matches_no_error() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let layout = concrete_layout(&parameters, 2);
        let no_error = supplied_slot_secret_graph(&layout, None);
        let zero_error = supplied_slot_secret_graph(&layout, Some(RealExpr::from_integer(0)));
        let secret_value = secret(&parameters, layout.secret_dimension);
        let slot_secret_values = [
            DCRTPolyMatrix::identity(&parameters, layout.secret_dimension, None),
            DCRTPolyMatrix::from_poly_vec(
                &parameters,
                vec![
                    vec![
                        DCRTPoly::const_rotate_poly(&parameters, 1),
                        DCRTPoly::const_rotate_poly(&parameters, 2),
                    ],
                    vec![
                        DCRTPoly::const_rotate_poly(&parameters, 3),
                        DCRTPoly::const_rotate_poly(&parameters, 4),
                    ],
                ],
            ),
        ];
        let public_values = [
            DCRTPolyHashSampler::<keccak_asm::Keccak256>::new().sample_hash(
                &parameters,
                [41u8; 32],
                b"supplied-public-0",
                layout.secret_dimension,
                layout.public_key_columns(),
                DistType::FinRingDist,
            ),
            DCRTPolyHashSampler::<keccak_asm::Keccak256>::new().sample_hash(
                &parameters,
                [41u8; 32],
                b"supplied-public-1",
                layout.secret_dimension,
                layout.public_key_columns(),
                DistType::FinRingDist,
            ),
        ];
        let plaintexts = [scalar(&parameters, 3), scalar(&parameters, 5)];
        let inputs = BTreeMap::from([
            ("secret".to_owned(), RuntimeValue::matrix(secret_value.clone())),
            ("public-0".to_owned(), RuntimeValue::matrix(public_values[0].clone())),
            ("public-1".to_owned(), RuntimeValue::matrix(public_values[1].clone())),
            ("plaintext-0".to_owned(), RuntimeValue::matrix(plaintexts[0].clone())),
            ("plaintext-1".to_owned(), RuntimeValue::matrix(plaintexts[1].clone())),
            ("slot-secret-0".to_owned(), RuntimeValue::matrix(slot_secret_values[0].clone())),
            ("slot-secret-1".to_owned(), RuntimeValue::matrix(slot_secret_values[1].clone())),
        ]);
        let no_error_result = execute_graph(no_error, parameters.clone(), inputs.clone());
        let zero_error_result = execute_graph(zero_error, parameters.clone(), inputs);
        let gadget = DCRTPolyMatrix::gadget_matrix(&parameters, layout.secret_dimension);
        for slot in 0..2 {
            let actual = matrix_output(&no_error_result, &format!("vector-{slot}"));
            assert_eq!(actual, matrix_output(&zero_error_result, &format!("vector-{slot}")));
            let transformed = secret_value.clone() * slot_secret_values[slot].clone();
            let expected = transformed.clone() * public_values[1].clone() -
                plaintexts[slot].clone().tensor(&(transformed * gadget.clone()));
            assert_eq!(actual, &expected);
        }
    }

    #[test]
    fn polynomial_sampler_runtime_uses_slot_secrets_in_the_bgg_formula() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let layout = concrete_layout(&parameters, 2);
        let ring = layout.ring();
        let public_keys = [
            BggPublicKeyWire {
                matrix: ring
                    .input("public-0", (layout.secret_dimension, layout.public_key_columns())),
                reveal_plaintext: true,
            },
            BggPublicKeyWire {
                matrix: ring
                    .input("public-1", (layout.secret_dimension, layout.public_key_columns())),
                reveal_plaintext: true,
            },
        ];
        let plaintext_family = Family::pack(
            (0..3).map(|slot| ring.input(format!("plaintext-{slot}"), (1, 1))).collect(),
        )
        .unwrap();
        let sample = BggPolyEncodingSampler { layout: layout.clone(), gaussian_sigma: None }
            .sample(
                ring.input("secret", (1, layout.secret_dimension)),
                &public_keys,
                &[plaintext_family],
                3.into(),
                None,
            )
            .unwrap();
        let mut context = DslContext::new("bgg-poly-sampler-runtime");
        for slot in 0..3 {
            context = context
                .output(format!("vector-{slot}"), sample.encodings[1].vectors.get_static(slot))
                .unwrap()
                .private_output(
                    format!("slot-secret-{slot}"),
                    sample.slot_secret_matrices.get_static(slot),
                )
                .unwrap();
        }
        let graph = context.build().unwrap();

        let secret_value = secret(&parameters, layout.secret_dimension);
        let public_values = [
            DCRTPolyHashSampler::<keccak_asm::Keccak256>::new().sample_hash(
                &parameters,
                [31u8; 32],
                b"poly-public-0",
                layout.secret_dimension,
                layout.public_key_columns(),
                DistType::FinRingDist,
            ),
            DCRTPolyHashSampler::<keccak_asm::Keccak256>::new().sample_hash(
                &parameters,
                [31u8; 32],
                b"poly-public-1",
                layout.secret_dimension,
                layout.public_key_columns(),
                DistType::FinRingDist,
            ),
        ];
        let plaintexts = (0..3).map(|slot| scalar(&parameters, slot + 1)).collect::<Vec<_>>();
        let mut inputs = BTreeMap::from([
            ("secret".to_owned(), RuntimeValue::matrix(secret_value.clone())),
            ("public-0".to_owned(), RuntimeValue::matrix(public_values[0].clone())),
            ("public-1".to_owned(), RuntimeValue::matrix(public_values[1].clone())),
        ]);
        for (slot, plaintext) in plaintexts.iter().enumerate() {
            inputs.insert(format!("plaintext-{slot}"), RuntimeValue::matrix(plaintext.clone()));
        }
        let result = execute_graph(graph, parameters.clone(), inputs);
        let gadget = DCRTPolyMatrix::gadget_matrix(&parameters, layout.secret_dimension);
        for slot in 0..3 {
            let slot_secret = matrix_output(&result, &format!("slot-secret-{slot}"));
            let transformed = secret_value.clone() * slot_secret.clone();
            let expected = transformed.clone() * public_values[1].clone() -
                plaintexts[slot].clone().tensor(&(transformed * gadget.clone()));
            assert_eq!(matrix_output(&result, &format!("vector-{slot}")), &expected);
        }
    }

    #[test]
    fn naive_sampler_runtime_preserves_tags_and_encoding_formulas() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let layout = concrete_layout(&parameters, 2);
        let key = [47u8; 32];
        let tag = b"naive-bgg-ir";
        let ring = layout.ring();
        let public_keys =
            NaiveBggPublicKeyVecSampler { layout: layout.clone(), slot_count: 2.into() }
                .sample(ring.bytes_input("key", key.len()), tag, &[true])
                .unwrap();
        let plaintexts = Family::pack(
            (0..2).map(|slot| ring.input(format!("plaintext-{slot}"), (1, 1))).collect(),
        )
        .unwrap();
        let encodings = NaiveBggEncodingVecSampler {
            scalar: BggEncodingSampler { layout: layout.clone(), gaussian_sigma: None },
        }
        .sample(ring.input("secret", (1, layout.secret_dimension)), &public_keys, &[plaintexts])
        .unwrap();
        let mut context = DslContext::new("naive-bgg-sampler-runtime");
        for output in 0..public_keys.len() {
            for slot in 0..2 {
                context = context
                    .output(
                        format!("public-{output}-{slot}"),
                        public_keys[output].matrices.get_static(slot),
                    )
                    .unwrap()
                    .output(
                        format!("vector-{output}-{slot}"),
                        encodings[output].vectors.get_static(slot),
                    )
                    .unwrap();
            }
        }
        let graph = context.build().unwrap();
        let secret_value = secret(&parameters, layout.secret_dimension);
        let plaintext_values = [scalar(&parameters, 2), scalar(&parameters, 5)];
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
        let hash_sampler = DCRTPolyHashSampler::<keccak_asm::Keccak256>::new();
        let gadget = DCRTPolyMatrix::gadget_matrix(&parameters, layout.secret_dimension);
        for output in 0..public_keys.len() {
            for slot in 0..2 {
                let mut slot_tag = tag.to_vec();
                slot_tag.extend_from_slice(&(output as u64).to_le_bytes());
                slot_tag.extend_from_slice(&(slot as u64).to_le_bytes());
                let packed_count = if output == 0 { 1 } else { 2 };
                let packed = hash_sampler.sample_hash(
                    &parameters,
                    key,
                    &slot_tag,
                    layout.secret_dimension,
                    layout.public_key_columns() * packed_count,
                    DistType::FinRingDist,
                );
                let expected_public = if output == 0 {
                    packed
                } else {
                    packed
                        .slice_columns(layout.public_key_columns(), 2 * layout.public_key_columns())
                };
                assert_eq!(
                    matrix_output(&result, &format!("public-{output}-{slot}")),
                    &expected_public
                );
                let plaintext = if output == 0 {
                    DCRTPolyMatrix::identity(&parameters, 1, None)
                } else {
                    plaintext_values[slot].clone()
                };
                let expected_vector = secret_value.clone() * expected_public -
                    plaintext.tensor(&(secret_value.clone() * gadget.clone()));
                assert_eq!(
                    matrix_output(&result, &format!("vector-{output}-{slot}")),
                    &expected_vector
                );
            }
        }
        assert!(public_keys.iter().all(|key| key.reveal_plaintext));
        assert!(encodings.iter().all(|encoding| encoding.plaintexts.is_some()));
    }

    #[test]
    fn fresh_slot_secret_draws_replay_at_the_same_parallel_sites() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let layout = concrete_layout(&parameters, 2);
        let ring = layout.ring();
        let sample = BggPolyEncodingSampler { layout: layout.clone(), gaussian_sigma: None }
            .sample(
                ring.input("secret", (1, layout.secret_dimension)),
                &[BggPublicKeyWire {
                    matrix: ring
                        .input("public", (layout.secret_dimension, layout.public_key_columns())),
                    reveal_plaintext: true,
                }],
                &[],
                2.into(),
                None,
            )
            .unwrap();
        let mut context = DslContext::new("bgg-poly-replay");
        for slot in 0..2 {
            context = context
                .output(format!("vector-{slot}"), sample.encodings[0].vectors.get_static(slot))
                .unwrap()
                .private_output(
                    format!("slot-secret-{slot}"),
                    sample.slot_secret_matrices.get_static(slot),
                )
                .unwrap();
        }
        let validated = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let inputs = BTreeMap::from([
            (
                "secret".to_owned(),
                RuntimeValue::matrix(secret(&parameters, layout.secret_dimension)),
            ),
            (
                "public".to_owned(),
                RuntimeValue::matrix(
                    DCRTPolyHashSampler::<keccak_asm::Keccak256>::new().sample_hash(
                        &parameters,
                        [43u8; 32],
                        b"replay-public",
                        layout.secret_dimension,
                        layout.public_key_columns(),
                        DistType::FinRingDist,
                    ),
                ),
            ),
        ]);
        let mut recorder = TranscriptRecorder::default();
        let recorded = execute(
            &validated,
            &mut cpu_backend([parameters.clone()]),
            inputs.clone(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Record(&mut recorder),
        )
        .unwrap();
        assert_eq!(recorder.iter().count(), 2);
        let replayer = recorder.into_replayer();
        let replayed = execute(
            &validated,
            &mut cpu_backend([parameters]),
            inputs,
            &mut MemoryArtifactStore::default(),
            SamplingMode::Replay(&replayer),
        )
        .unwrap();
        for name in recorded.outputs.keys() {
            let RuntimeValue::Matrix(recorded_value) = &recorded.outputs[name] else {
                panic!("{name} recorded output must be a matrix")
            };
            let RuntimeValue::Matrix(replayed_value) = &replayed.outputs[name] else {
                panic!("{name} replayed output must be a matrix")
            };
            assert_eq!(recorded_value, replayed_value, "{name}");
        }
    }

    #[test]
    fn runtime_rejects_a_sampler_gadget_layout_the_backend_cannot_honor() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let mut layout = concrete_layout(&parameters, 2);
        layout.gadget_base = IntExpr::constant(BigInt::from(1u64 << (parameters.base_bits() + 1)));
        let ring = layout.ring();
        let encodings = BggEncodingSampler { layout: layout.clone(), gaussian_sigma: None }
            .sample(
                ring.input("secret", (1, layout.secret_dimension)),
                &[BggPublicKeyWire {
                    matrix: ring
                        .input("public", (layout.secret_dimension, layout.public_key_columns())),
                    reveal_plaintext: true,
                }],
                &[],
            )
            .unwrap();
        let validated = DslContext::new("bgg-invalid-gadget-layout")
            .output("vector", encodings[0].vector.clone())
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let error = match execute(
            &validated,
            &mut cpu_backend([parameters.clone()]),
            BTreeMap::from([
                (
                    "secret".to_owned(),
                    RuntimeValue::matrix(secret(&parameters, layout.secret_dimension)),
                ),
                (
                    "public".to_owned(),
                    RuntimeValue::matrix(DCRTPolyMatrix::zero(
                        &parameters,
                        layout.secret_dimension,
                        layout.public_key_columns(),
                    )),
                ),
            ]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        ) {
            Ok(_) => panic!("mismatched backend gadget layout must be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("does not match backend base"));
    }
}
