//! Per-slot BGG+ vectors represented by indexed DSL families.

use crate::{BggPublicKeyCompiler, BggPublicKeyWire};
use mxx_dsl::{DslError, Family, Mat};
use thiserror::Error;

#[derive(Clone)]
pub struct NaiveBggPublicKeyVecWire {
    pub matrices: Family<Mat>,
    pub reveal_plaintext: bool,
}

#[derive(Clone)]
pub struct NaiveBggEncodingVecWire {
    pub vectors: Family<Mat>,
    pub pubkeys: Family<Mat>,
    pub pubkey_reveal_plaintext: bool,
    pub plaintexts: Option<Family<Mat>>,
}

#[derive(Clone)]
pub struct NaiveBggVecCompiler {
    pub public_key: BggPublicKeyCompiler,
}

#[derive(Debug, Error)]
pub enum NaiveVecCompileError {
    #[error("naive BGG+ vector families must have matching slot counts")]
    SlotCountMismatch,
    #[error("naive BGG+ multiplication requires the left plaintext family")]
    MissingLeftPlaintext,
    #[error(transparent)]
    Dsl(#[from] DslError),
}

impl NaiveBggVecCompiler {
    /// Lifts one scalar family into a per-slot public-key vector.
    ///
    /// This is the public-key counterpart of
    /// [`Self::large_scalar_mul_encoding_families`]. Diamond iO uses the two
    /// methods on the same native Ring-GSW scalar layout during preprocessing
    /// and online evaluation respectively.
    pub fn large_scalar_mul_public_key_families(
        &self,
        input: &NaiveBggPublicKeyVecWire,
        scalars: Family<Mat>,
    ) -> Result<NaiveBggPublicKeyVecWire, NaiveVecCompileError> {
        if input.matrices.count() != scalars.count() {
            return Err(NaiveVecCompileError::SlotCountMismatch);
        }
        let compiler = self.public_key.clone();
        let reveal = input.reveal_plaintext;
        let matrices = input.matrices.clone().parallel_zip(scalars, move |_, matrix, scalar| {
            compiler
                .large_scalar_mul(&BggPublicKeyWire { matrix, reveal_plaintext: reveal }, &scalar)
                .matrix
        })?;
        Ok(NaiveBggPublicKeyVecWire { matrices, reveal_plaintext: reveal })
    }

    /// Lifts one scalar family into a per-slot encoding vector using the same
    /// BGG+ formula as ordinary large-scalar circuit lowering.
    pub fn large_scalar_mul_encoding_families(
        &self,
        input: &NaiveBggEncodingVecWire,
        scalars: Family<Mat>,
    ) -> Result<NaiveBggEncodingVecWire, NaiveVecCompileError> {
        validate_encoding(input)?;
        if input.vectors.count() != scalars.count() {
            return Err(NaiveVecCompileError::SlotCountMismatch);
        }
        let vector_compiler = self.public_key.clone();
        let reveal = input.pubkey_reveal_plaintext;
        let factors =
            input.pubkeys.clone().parallel_zip(scalars.clone(), move |_, matrix, scalar| {
                vector_compiler.large_scalar_decomposition(
                    &BggPublicKeyWire { matrix, reveal_plaintext: reveal },
                    &scalar,
                )
            })?;
        let vectors =
            input.vectors.clone().parallel_zip(factors, |_, vector, factor| vector * factor)?;
        let key_compiler = self.public_key.clone();
        let pubkeys =
            input.pubkeys.clone().parallel_zip(scalars.clone(), move |_, matrix, scalar| {
                key_compiler
                    .large_scalar_mul(
                        &BggPublicKeyWire { matrix, reveal_plaintext: reveal },
                        &scalar,
                    )
                    .matrix
            })?;
        let plaintexts = input
            .plaintexts
            .clone()
            .map(|values| values.parallel_zip(scalars, |_, value, scalar| value * scalar))
            .transpose()?;
        Ok(NaiveBggEncodingVecWire {
            vectors,
            pubkeys,
            pubkey_reveal_plaintext: reveal,
            plaintexts,
        })
    }

    pub fn matrix_mul_public_keys(
        &self,
        input: &NaiveBggPublicKeyVecWire,
        target: &Mat,
    ) -> Result<NaiveBggPublicKeyVecWire, NaiveVecCompileError> {
        let compiler = self.public_key.clone();
        let target = target.clone();
        let reveal = input.reveal_plaintext;
        let matrices = input.matrices.clone().parallel_map(move |_, matrix| {
            compiler
                .matrix_mul(&BggPublicKeyWire { matrix, reveal_plaintext: reveal }, &target)
                .matrix
        })?;
        Ok(NaiveBggPublicKeyVecWire { matrices, reveal_plaintext: reveal })
    }

    pub fn add_public_keys(
        &self,
        lhs: &NaiveBggPublicKeyVecWire,
        rhs: &NaiveBggPublicKeyVecWire,
    ) -> Result<NaiveBggPublicKeyVecWire, NaiveVecCompileError> {
        self.public_key_binary(lhs, rhs, |compiler, left, right| compiler.add(left, right))
    }

    pub fn sub_public_keys(
        &self,
        lhs: &NaiveBggPublicKeyVecWire,
        rhs: &NaiveBggPublicKeyVecWire,
    ) -> Result<NaiveBggPublicKeyVecWire, NaiveVecCompileError> {
        self.public_key_binary(lhs, rhs, |compiler, left, right| compiler.sub(left, right))
    }

    pub fn mul_public_keys(
        &self,
        lhs: &NaiveBggPublicKeyVecWire,
        rhs: &NaiveBggPublicKeyVecWire,
    ) -> Result<NaiveBggPublicKeyVecWire, NaiveVecCompileError> {
        self.public_key_binary(lhs, rhs, |compiler, left, right| compiler.mul(left, right))
    }

    pub fn add_encodings(
        &self,
        lhs: &NaiveBggEncodingVecWire,
        rhs: &NaiveBggEncodingVecWire,
    ) -> Result<NaiveBggEncodingVecWire, NaiveVecCompileError> {
        self.encoding_binary(
            lhs,
            rhs,
            |left, right| left + right,
            |compiler, left, right| compiler.add(left, right),
        )
    }

    pub fn sub_encodings(
        &self,
        lhs: &NaiveBggEncodingVecWire,
        rhs: &NaiveBggEncodingVecWire,
    ) -> Result<NaiveBggEncodingVecWire, NaiveVecCompileError> {
        self.encoding_binary(
            lhs,
            rhs,
            |left, right| left - right,
            |compiler, left, right| compiler.sub(left, right),
        )
    }

    pub fn mul_encodings(
        &self,
        lhs: &NaiveBggEncodingVecWire,
        rhs: &NaiveBggEncodingVecWire,
    ) -> Result<NaiveBggEncodingVecWire, NaiveVecCompileError> {
        validate_encoding_pair(lhs, rhs)?;
        let lhs_plaintexts =
            lhs.plaintexts.clone().ok_or(NaiveVecCompileError::MissingLeftPlaintext)?;
        let base = self.public_key.base.clone();
        let digits = self.public_key.digit_count.clone();
        let decomposed_rhs = rhs.pubkeys.clone().parallel_map(move |_, matrix| {
            matrix.decompose(base.clone(), digits.clone()).as_mat()
        })?;
        let first =
            lhs.vectors.clone().parallel_zip(decomposed_rhs, |_, left, right| left * right)?;
        let second = rhs
            .vectors
            .clone()
            .parallel_zip(lhs_plaintexts, |_, right, plaintext| right * plaintext)?;
        let vectors = first.parallel_zip(second, |_, left, right| left + right)?;
        let pubkeys =
            self.key_family_binary(lhs, rhs, |compiler, left, right| compiler.mul(left, right))?;
        let plaintexts = lhs
            .plaintexts
            .clone()
            .zip(rhs.plaintexts.clone())
            .map(|(left, right)| left.parallel_zip(right, |_, left, right| left * right))
            .transpose()?;
        Ok(NaiveBggEncodingVecWire {
            vectors,
            pubkeys,
            pubkey_reveal_plaintext: lhs.pubkey_reveal_plaintext && rhs.pubkey_reveal_plaintext,
            plaintexts,
        })
    }

    pub fn small_scalar_mul_encodings(
        &self,
        input: &NaiveBggEncodingVecWire,
        scalar: &Mat,
    ) -> Result<NaiveBggEncodingVecWire, NaiveVecCompileError> {
        self.encoding_scalar(input, scalar, false)
    }

    pub fn large_scalar_mul_encodings(
        &self,
        input: &NaiveBggEncodingVecWire,
        scalar: &Mat,
    ) -> Result<NaiveBggEncodingVecWire, NaiveVecCompileError> {
        self.encoding_scalar(input, scalar, true)
    }

    pub fn matrix_mul_encodings(
        &self,
        input: &NaiveBggEncodingVecWire,
        target: &Mat,
    ) -> Result<NaiveBggEncodingVecWire, NaiveVecCompileError> {
        validate_encoding(input)?;
        let base = self.public_key.base.clone();
        let digits = self.public_key.digit_count.clone();
        let decomposed = target.clone().decompose(base, digits).as_mat();
        let vectors =
            input.vectors.clone().parallel_map(move |_, value| value * decomposed.clone())?;
        let key_compiler = self.public_key.clone();
        let target_for_keys = target.clone();
        let pubkeys = input.pubkeys.clone().parallel_map(move |_, matrix| {
            key_compiler
                .matrix_mul(
                    &BggPublicKeyWire { matrix, reveal_plaintext: input.pubkey_reveal_plaintext },
                    &target_for_keys,
                )
                .matrix
        })?;
        Ok(NaiveBggEncodingVecWire {
            vectors,
            pubkeys,
            pubkey_reveal_plaintext: input.pubkey_reveal_plaintext,
            plaintexts: None,
        })
    }

    fn public_key_binary(
        &self,
        lhs: &NaiveBggPublicKeyVecWire,
        rhs: &NaiveBggPublicKeyVecWire,
        operation: impl Fn(
            &BggPublicKeyCompiler,
            &BggPublicKeyWire,
            &BggPublicKeyWire,
        ) -> BggPublicKeyWire
        + Send
        + Sync
        + 'static,
    ) -> Result<NaiveBggPublicKeyVecWire, NaiveVecCompileError> {
        if lhs.matrices.count() != rhs.matrices.count() {
            return Err(NaiveVecCompileError::SlotCountMismatch);
        }
        let compiler = self.public_key.clone();
        let reveal = lhs.reveal_plaintext && rhs.reveal_plaintext;
        let left_reveal = lhs.reveal_plaintext;
        let right_reveal = rhs.reveal_plaintext;
        let matrices =
            lhs.matrices.clone().parallel_zip(rhs.matrices.clone(), move |_, left, right| {
                operation(
                    &compiler,
                    &BggPublicKeyWire { matrix: left, reveal_plaintext: left_reveal },
                    &BggPublicKeyWire { matrix: right, reveal_plaintext: right_reveal },
                )
                .matrix
            })?;
        Ok(NaiveBggPublicKeyVecWire { matrices, reveal_plaintext: reveal })
    }

    fn encoding_binary(
        &self,
        lhs: &NaiveBggEncodingVecWire,
        rhs: &NaiveBggEncodingVecWire,
        vector_op: impl Fn(Mat, Mat) -> Mat + Copy,
        key_op: impl Fn(&BggPublicKeyCompiler, &BggPublicKeyWire, &BggPublicKeyWire) -> BggPublicKeyWire
        + 'static,
    ) -> Result<NaiveBggEncodingVecWire, NaiveVecCompileError> {
        validate_encoding_pair(lhs, rhs)?;
        let vectors = lhs
            .vectors
            .clone()
            .parallel_zip(rhs.vectors.clone(), move |_, left, right| vector_op(left, right))?;
        let pubkeys = self.key_family_binary(lhs, rhs, key_op)?;
        let plaintexts = lhs
            .plaintexts
            .clone()
            .zip(rhs.plaintexts.clone())
            .map(|(left, right)| {
                left.parallel_zip(right, move |_, left, right| vector_op(left, right))
            })
            .transpose()?;
        Ok(NaiveBggEncodingVecWire {
            vectors,
            pubkeys,
            pubkey_reveal_plaintext: lhs.pubkey_reveal_plaintext && rhs.pubkey_reveal_plaintext,
            plaintexts,
        })
    }

    fn key_family_binary(
        &self,
        lhs: &NaiveBggEncodingVecWire,
        rhs: &NaiveBggEncodingVecWire,
        operation: impl Fn(
            &BggPublicKeyCompiler,
            &BggPublicKeyWire,
            &BggPublicKeyWire,
        ) -> BggPublicKeyWire
        + 'static,
    ) -> Result<Family<Mat>, DslError> {
        let compiler = self.public_key.clone();
        let left_reveal = lhs.pubkey_reveal_plaintext;
        let right_reveal = rhs.pubkey_reveal_plaintext;
        lhs.pubkeys.clone().parallel_zip(rhs.pubkeys.clone(), move |_, left, right| {
            operation(
                &compiler,
                &BggPublicKeyWire { matrix: left, reveal_plaintext: left_reveal },
                &BggPublicKeyWire { matrix: right, reveal_plaintext: right_reveal },
            )
            .matrix
        })
    }

    fn encoding_scalar(
        &self,
        input: &NaiveBggEncodingVecWire,
        scalar: &Mat,
        large: bool,
    ) -> Result<NaiveBggEncodingVecWire, NaiveVecCompileError> {
        validate_encoding(input)?;
        let vector_factor = if large {
            let rows = input.pubkeys.clone().parallel_map({
                let compiler = self.public_key.clone();
                let scalar = scalar.clone();
                move |_, matrix| {
                    compiler.large_scalar_decomposition(
                        &BggPublicKeyWire {
                            matrix,
                            reveal_plaintext: input.pubkey_reveal_plaintext,
                        },
                        &scalar,
                    )
                }
            })?;
            Some(rows)
        } else {
            None
        };
        let vectors = match vector_factor {
            Some(factors) => {
                input.vectors.clone().parallel_zip(factors, |_, value, factor| value * factor)?
            }
            None => {
                let scalar = scalar.clone();
                input.vectors.clone().parallel_map(move |_, value| value * scalar.clone())?
            }
        };
        let compiler = self.public_key.clone();
        let scalar_for_keys = scalar.clone();
        let reveal = input.pubkey_reveal_plaintext;
        let pubkeys = input.pubkeys.clone().parallel_map(move |_, matrix| {
            let key = BggPublicKeyWire { matrix, reveal_plaintext: reveal };
            if large {
                compiler.large_scalar_mul(&key, &scalar_for_keys).matrix
            } else {
                compiler.small_scalar_mul(&key, &scalar_for_keys).matrix
            }
        })?;
        let plaintexts = input
            .plaintexts
            .clone()
            .map(|values| {
                let scalar = scalar.clone();
                values.parallel_map(move |_, value| value * scalar.clone())
            })
            .transpose()?;
        Ok(NaiveBggEncodingVecWire {
            vectors,
            pubkeys,
            pubkey_reveal_plaintext: reveal,
            plaintexts,
        })
    }
}

fn validate_encoding(value: &NaiveBggEncodingVecWire) -> Result<(), NaiveVecCompileError> {
    if value.vectors.count() != value.pubkeys.count() ||
        value
            .plaintexts
            .as_ref()
            .is_some_and(|plaintexts| plaintexts.count() != value.vectors.count())
    {
        return Err(NaiveVecCompileError::SlotCountMismatch);
    }
    Ok(())
}

fn validate_encoding_pair(
    lhs: &NaiveBggEncodingVecWire,
    rhs: &NaiveBggEncodingVecWire,
) -> Result<(), NaiveVecCompileError> {
    validate_encoding(lhs)?;
    validate_encoding(rhs)?;
    if lhs.vectors.count() != rhs.vectors.count() {
        return Err(NaiveVecCompileError::SlotCountMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{execute_graph, matrix_output, row};
    use mxx_dsl::{DslContext, Ring};
    use mxx_ir_core::{ParamEnv, node::NodeKind};
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{PolyParams, dcrt::params::DCRTPolyParams},
    };
    use mxx_runtime::RuntimeValue;
    use num_bigint::BigInt;
    use std::collections::BTreeMap;

    #[test]
    fn native_scalar_family_lift_is_symmetric_for_public_keys_and_encodings() {
        let ring = Ring::new(257, 8);
        let compiler = NaiveBggVecCompiler {
            public_key: BggPublicKeyCompiler {
                ring: ring.clone(),
                base: 4.into(),
                digit_count: 4.into(),
            },
        };
        let scalars = ring.input_family("native-scalars", 2, (1, 1));
        let public = NaiveBggPublicKeyVecWire {
            matrices: ring.input_family("public-keys", 2, (1, 4)),
            reveal_plaintext: true,
        };
        let encoding = NaiveBggEncodingVecWire {
            vectors: ring.input_family("encoding-vectors", 2, (1, 4)),
            pubkeys: public.matrices.clone(),
            pubkey_reveal_plaintext: true,
            plaintexts: Some(ring.input_family("plaintexts", 2, (1, 1))),
        };
        let public_output =
            compiler.large_scalar_mul_public_key_families(&public, scalars.clone()).unwrap();
        let encoding_output =
            compiler.large_scalar_mul_encoding_families(&encoding, scalars).unwrap();
        let graph = DslContext::new("native-ring-gsw-bgg-lift")
            .family_output("public", public_output.matrices)
            .unwrap()
            .family_output("vectors", encoding_output.vectors)
            .unwrap()
            .family_output("encoding-public", encoding_output.pubkeys)
            .unwrap()
            .family_output("plaintexts", encoding_output.plaintexts.unwrap())
            .unwrap()
            .build()
            .unwrap();
        graph.validate(&ParamEnv::default()).unwrap();
        graph.elaborate(&ParamEnv::default()).unwrap();
    }

    #[test]
    fn naive_encoding_multiplication_elaborates_all_indexed_families() {
        let ring = Ring::new(257, 8);
        let compiler = NaiveBggVecCompiler {
            public_key: BggPublicKeyCompiler {
                ring: ring.clone(),
                base: 4.into(),
                digit_count: 4.into(),
            },
        };
        let encoding = |prefix: &str| NaiveBggEncodingVecWire {
            vectors: ring.input_family(format!("{prefix}-vectors"), 2, (1, 8)),
            pubkeys: ring.input_family(format!("{prefix}-pubkeys"), 2, (2, 8)),
            pubkey_reveal_plaintext: true,
            plaintexts: Some(ring.input_family(format!("{prefix}-plaintexts"), 2, (1, 1))),
        };
        let product =
            compiler.mul_encodings(&encoding("left"), &encoding("right")).expect("product");
        let built = DslContext::new("naive-bgg-encoding-mul")
            .family_output("vectors", product.vectors)
            .expect("vector family")
            .family_output("pubkeys", product.pubkeys)
            .expect("public-key family")
            .build()
            .expect("build");
        let kinds = built
            .graph
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .map(|node| node.kind())
            .collect::<Vec<_>>();
        assert!(kinds.iter().any(|kind| matches!(kind, NodeKind::ParallelLoop(_))));
        assert!(kinds.iter().any(|kind| matches!(kind, NodeKind::GadgetDecompose { .. })));
        assert!(kinds.iter().any(|kind| matches!(kind, NodeKind::MatrixBinary(_))));

        let elaborated = built.elaborate(&ParamEnv::default()).expect("symbolic elaboration");
        for output in ["vectors", "pubkeys"] {
            let wire = elaborated.wire(&elaborated.outputs[output]).expect("family output");
            assert!(wire.family.is_some());
        }
    }

    #[test]
    fn runtime_addition_zips_every_component_and_matches_primitives() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digit_count = parameters.modulus_digits();
        let columns = 2 * digit_count;
        let ring = Ring::new(
            BigInt::from(parameters.modulus().as_ref().clone()),
            parameters.ring_dimension() as usize,
        );
        let compiler = NaiveBggVecCompiler {
            public_key: BggPublicKeyCompiler {
                ring: ring.clone(),
                base: BigInt::from(1u64 << parameters.base_bits()).into(),
                digit_count: digit_count.into(),
            },
        };
        let encoding = |prefix: &str| NaiveBggEncodingVecWire {
            vectors: Family::pack(
                (0..2)
                    .map(|slot| ring.input(format!("{prefix}-vector-{slot}"), (1, columns)))
                    .collect(),
            )
            .unwrap(),
            pubkeys: Family::pack(
                (0..2)
                    .map(|slot| ring.input(format!("{prefix}-public-{slot}"), (2, columns)))
                    .collect(),
            )
            .unwrap(),
            pubkey_reveal_plaintext: true,
            plaintexts: Some(
                Family::pack(
                    (0..2)
                        .map(|slot| ring.input(format!("{prefix}-plaintext-{slot}"), (1, 1)))
                        .collect(),
                )
                .unwrap(),
            ),
        };
        let output = compiler.add_encodings(&encoding("lhs"), &encoding("rhs")).unwrap();
        let plaintexts = output.plaintexts.unwrap();
        let mut context = DslContext::new("naive-bgg-add-runtime");
        for slot in 0..2 {
            context = context
                .output(format!("vector-{slot}"), output.vectors.get_static(slot))
                .unwrap()
                .output(format!("public-{slot}"), output.pubkeys.get_static(slot))
                .unwrap()
                .output(format!("plaintext-{slot}"), plaintexts.get_static(slot))
                .unwrap();
        }
        let graph = context.build().unwrap();

        let mut inputs = BTreeMap::new();
        let mut expected = BTreeMap::new();
        for slot in 0..2 {
            let lhs_vector = row(&parameters, columns, slot);
            let rhs_vector = row(&parameters, columns, slot + 1);
            let lhs_public = DCRTPolyMatrix::from_poly_vec(
                &parameters,
                vec![
                    row(&parameters, columns, slot + 2).get_row(0),
                    row(&parameters, columns, slot + 4).get_row(0),
                ],
            );
            let rhs_public = DCRTPolyMatrix::from_poly_vec(
                &parameters,
                vec![
                    row(&parameters, columns, slot + 3).get_row(0),
                    row(&parameters, columns, slot + 5).get_row(0),
                ],
            );
            let lhs_plaintext = row(&parameters, 1, slot + 6);
            let rhs_plaintext = row(&parameters, 1, slot + 7);
            for (name, value) in [
                (format!("lhs-vector-{slot}"), lhs_vector.clone()),
                (format!("rhs-vector-{slot}"), rhs_vector.clone()),
                (format!("lhs-public-{slot}"), lhs_public.clone()),
                (format!("rhs-public-{slot}"), rhs_public.clone()),
                (format!("lhs-plaintext-{slot}"), lhs_plaintext.clone()),
                (format!("rhs-plaintext-{slot}"), rhs_plaintext.clone()),
            ] {
                inputs.insert(name, RuntimeValue::matrix(value));
            }
            expected.insert(format!("vector-{slot}"), lhs_vector + rhs_vector);
            expected.insert(format!("public-{slot}"), lhs_public + rhs_public);
            expected.insert(format!("plaintext-{slot}"), lhs_plaintext + rhs_plaintext);
        }
        let result = execute_graph(graph, parameters, inputs);
        for (name, expected) in expected {
            assert_eq!(matrix_output(&result, &name), &expected, "{name}");
        }
    }

    #[test]
    fn runtime_matrix_multiplication_matches_primitive_decomposition() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digit_count = parameters.modulus_digits();
        let columns = 2 * digit_count;
        let ring = Ring::new(
            BigInt::from(parameters.modulus().as_ref().clone()),
            parameters.ring_dimension() as usize,
        );
        let compiler = NaiveBggVecCompiler {
            public_key: BggPublicKeyCompiler {
                ring: ring.clone(),
                base: BigInt::from(1u64 << parameters.base_bits()).into(),
                digit_count: digit_count.into(),
            },
        };
        let input = NaiveBggEncodingVecWire {
            vectors: Family::pack(vec![ring.input("vector", (1, columns))]).unwrap(),
            pubkeys: Family::pack(vec![ring.input("public", (2, columns))]).unwrap(),
            pubkey_reveal_plaintext: true,
            plaintexts: Some(Family::pack(vec![ring.input("plaintext", (1, 1))]).unwrap()),
        };
        let target = ring.input("target", (2, 1));
        let output = compiler.matrix_mul_encodings(&input, &target).unwrap();
        let graph = DslContext::new("naive-bgg-matrix-mul-runtime")
            .output("vector", output.vectors.get_static(0))
            .unwrap()
            .output("public", output.pubkeys.get_static(0))
            .unwrap()
            .build()
            .unwrap();

        let vector = row(&parameters, columns, 0);
        let public = DCRTPolyMatrix::from_poly_vec(
            &parameters,
            vec![row(&parameters, columns, 2).get_row(0), row(&parameters, columns, 4).get_row(0)],
        );
        let plaintext = row(&parameters, 1, 6);
        let target = DCRTPolyMatrix::unit_column_vector(&parameters, 2, 1);
        let result = execute_graph(
            graph,
            parameters,
            BTreeMap::from([
                ("vector".to_owned(), RuntimeValue::matrix(vector.clone())),
                ("public".to_owned(), RuntimeValue::matrix(public.clone())),
                ("plaintext".to_owned(), RuntimeValue::matrix(plaintext)),
                ("target".to_owned(), RuntimeValue::matrix(target.clone())),
            ]),
        );
        assert_eq!(matrix_output(&result, "vector"), &vector.mul_decompose(&target));
        assert_eq!(matrix_output(&result, "public"), &public.mul_decompose(&target));
        assert!(output.plaintexts.is_none());
    }
}
