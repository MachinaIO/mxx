//! Slot-family BGG+ encodings represented by one structural parallel loop.

use crate::{BggPublicKeyCompiler, BggPublicKeyWire};
use mxx_dsl::{DslError, Family, Mat};
use thiserror::Error;

#[derive(Clone)]
pub struct BggPolyEncodingWire {
    pub vectors: Family<Mat>,
    pub pubkey: BggPublicKeyWire,
    pub plaintexts: Option<Family<Mat>>,
}

#[derive(Clone)]
pub struct BggPolyEncodingCompiler {
    pub public_key: BggPublicKeyCompiler,
}

#[derive(Debug, Error)]
pub enum PolyEncodingCompileError {
    #[error("BGG+ poly-encoding families must have matching slot counts")]
    SlotCountMismatch,
    #[error("BGG+ poly-encoding multiplication requires the left plaintext family")]
    MissingLeftPlaintext,
    #[error(transparent)]
    Dsl(#[from] DslError),
}

impl BggPolyEncodingCompiler {
    pub fn add(
        &self,
        lhs: &BggPolyEncodingWire,
        rhs: &BggPolyEncodingWire,
    ) -> Result<BggPolyEncodingWire, PolyEncodingCompileError> {
        self.binary(
            lhs,
            rhs,
            |left, right| left + right,
            |compiler, left, right| compiler.add(left, right),
        )
    }

    pub fn sub(
        &self,
        lhs: &BggPolyEncodingWire,
        rhs: &BggPolyEncodingWire,
    ) -> Result<BggPolyEncodingWire, PolyEncodingCompileError> {
        self.binary(
            lhs,
            rhs,
            |left, right| left - right,
            |compiler, left, right| compiler.sub(left, right),
        )
    }

    pub fn mul(
        &self,
        lhs: &BggPolyEncodingWire,
        rhs: &BggPolyEncodingWire,
    ) -> Result<BggPolyEncodingWire, PolyEncodingCompileError> {
        validate_pair(lhs, rhs)?;
        let lhs_plaintexts =
            lhs.plaintexts.clone().ok_or(PolyEncodingCompileError::MissingLeftPlaintext)?;
        let decomposed_rhs = rhs
            .pubkey
            .matrix
            .clone()
            .decompose(self.public_key.base.clone(), self.public_key.digit_count.clone())
            .as_mat();
        let vectors = lhs.vectors.clone().parallel_zip3(
            rhs.vectors.clone(),
            lhs_plaintexts.clone(),
            move |_, left, right, plaintext| left * decomposed_rhs.clone() + right * plaintext,
        )?;
        let plaintexts = lhs
            .plaintexts
            .clone()
            .zip(rhs.plaintexts.clone())
            .map(|(left, right)| left.parallel_zip(right, |_, left, right| left * right))
            .transpose()?;
        Ok(BggPolyEncodingWire {
            vectors,
            pubkey: self.public_key.mul(&lhs.pubkey, &rhs.pubkey),
            plaintexts,
        })
    }

    pub fn small_scalar_mul(
        &self,
        input: &BggPolyEncodingWire,
        scalar: &Mat,
    ) -> Result<BggPolyEncodingWire, PolyEncodingCompileError> {
        self.scalar_mul(input, scalar, scalar.clone(), false)
    }

    pub fn large_scalar_mul(
        &self,
        input: &BggPolyEncodingWire,
        scalar: &Mat,
    ) -> Result<BggPolyEncodingWire, PolyEncodingCompileError> {
        let decomposed = self.public_key.large_scalar_decomposition(&input.pubkey, scalar);
        self.scalar_mul(input, scalar, decomposed, true)
    }

    fn binary(
        &self,
        lhs: &BggPolyEncodingWire,
        rhs: &BggPolyEncodingWire,
        vector_op: impl FnOnce(Mat, Mat) -> Mat + Copy,
        key_op: impl FnOnce(
            &BggPublicKeyCompiler,
            &BggPublicKeyWire,
            &BggPublicKeyWire,
        ) -> BggPublicKeyWire,
    ) -> Result<BggPolyEncodingWire, PolyEncodingCompileError> {
        validate_pair(lhs, rhs)?;
        let vectors = lhs
            .vectors
            .clone()
            .parallel_zip(rhs.vectors.clone(), move |_, left, right| vector_op(left, right))?;
        let plaintexts = lhs
            .plaintexts
            .clone()
            .zip(rhs.plaintexts.clone())
            .map(|(left, right)| {
                left.parallel_zip(right, move |_, left, right| vector_op(left, right))
            })
            .transpose()?;
        Ok(BggPolyEncodingWire {
            vectors,
            pubkey: key_op(&self.public_key, &lhs.pubkey, &rhs.pubkey),
            plaintexts,
        })
    }

    fn scalar_mul(
        &self,
        input: &BggPolyEncodingWire,
        scalar: &Mat,
        vector_factor: Mat,
        large: bool,
    ) -> Result<BggPolyEncodingWire, PolyEncodingCompileError> {
        let vectors =
            input.vectors.clone().parallel_map(move |_, value| value * vector_factor.clone())?;
        let plaintexts = input
            .plaintexts
            .clone()
            .map(|values| {
                let scalar = scalar.clone();
                values.parallel_map(move |_, value| value * scalar.clone())
            })
            .transpose()?;
        let pubkey = if large {
            self.public_key.large_scalar_mul(&input.pubkey, scalar)
        } else {
            self.public_key.small_scalar_mul(&input.pubkey, scalar)
        };
        Ok(BggPolyEncodingWire { vectors, pubkey, plaintexts })
    }
}

fn validate_pair(
    lhs: &BggPolyEncodingWire,
    rhs: &BggPolyEncodingWire,
) -> Result<(), PolyEncodingCompileError> {
    if lhs.vectors.count() != rhs.vectors.count() ||
        lhs.plaintexts.as_ref().is_some_and(|values| values.count() != lhs.vectors.count()) ||
        rhs.plaintexts.as_ref().is_some_and(|values| values.count() != rhs.vectors.count())
    {
        return Err(PolyEncodingCompileError::SlotCountMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{execute_graph, matrix_output, row};
    use mxx_dsl::{DslContext, Ring};

    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{PolyParams, dcrt::params::DCRTPolyParams},
    };
    use mxx_runtime::RuntimeValue;
    use num_bigint::BigInt;
    use std::collections::BTreeMap;

    #[test]
    fn reveal_combinations_match_the_poly_encoding_contract() {
        let ring = Ring::new(17, 8);
        let compiler = BggPolyEncodingCompiler {
            public_key: BggPublicKeyCompiler {
                ring: ring.clone(),
                base: 2.into(),
                digit_count: 2.into(),
            },
        };
        for left_revealed in [false, true] {
            for right_revealed in [false, true] {
                let encoding = |prefix: &str, revealed: bool| BggPolyEncodingWire {
                    vectors: Family::pack(vec![ring.input(format!("{prefix}-vector"), (1, 4))])
                        .unwrap(),
                    pubkey: BggPublicKeyWire {
                        matrix: ring.input(format!("{prefix}-public"), (2, 4)),
                        reveal_plaintext: revealed,
                    },
                    plaintexts: revealed.then(|| {
                        Family::pack(vec![ring.input(format!("{prefix}-plain"), (1, 1))]).unwrap()
                    }),
                };
                let left = encoding("left", left_revealed);
                let right = encoding("right", right_revealed);
                let expected = left_revealed && right_revealed;
                for output in
                    [compiler.add(&left, &right).unwrap(), compiler.sub(&left, &right).unwrap()]
                {
                    assert_eq!(output.pubkey.reveal_plaintext, expected);
                    assert_eq!(output.plaintexts.is_some(), expected);
                }
                match compiler.mul(&left, &right) {
                    Ok(output) => {
                        assert!(left_revealed);
                        assert_eq!(output.pubkey.reveal_plaintext, expected);
                        assert_eq!(output.plaintexts.is_some(), expected);
                    }
                    Err(error) => {
                        assert!(!left_revealed);
                        assert!(matches!(error, PolyEncodingCompileError::MissingLeftPlaintext));
                    }
                }
            }
        }
    }

    #[test]
    fn runtime_addition_matches_primitive_family_formulas() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digit_count = parameters.modulus_digits();
        let columns = 2 * digit_count;
        let ring = Ring::new(
            BigInt::from(parameters.modulus().as_ref().clone()),
            parameters.ring_dimension() as usize,
        );
        let compiler = BggPolyEncodingCompiler {
            public_key: BggPublicKeyCompiler {
                ring: ring.clone(),
                base: BigInt::from(1u64 << parameters.base_bits()).into(),
                digit_count: digit_count.into(),
            },
        };
        let encoding = |prefix: &str| BggPolyEncodingWire {
            vectors: Family::pack(
                (0..2)
                    .map(|slot| ring.input(format!("{prefix}-vector-{slot}"), (1, columns)))
                    .collect(),
            )
            .unwrap(),
            pubkey: BggPublicKeyWire {
                matrix: ring.input(format!("{prefix}-public"), (2, columns)),
                reveal_plaintext: true,
            },
            plaintexts: Some(
                Family::pack(
                    (0..2)
                        .map(|slot| ring.input(format!("{prefix}-plaintext-{slot}"), (1, 1)))
                        .collect(),
                )
                .unwrap(),
            ),
        };
        let output = compiler.add(&encoding("lhs"), &encoding("rhs")).unwrap();
        let plaintexts = output.plaintexts.unwrap();
        let mut context =
            DslContext::new("bgg-poly-add-runtime").output("public", output.pubkey.matrix).unwrap();
        for slot in 0..2 {
            context = context
                .output(format!("vector-{slot}"), output.vectors.get_static(slot))
                .unwrap()
                .output(format!("plaintext-{slot}"), plaintexts.get_static(slot))
                .unwrap();
        }
        let graph = context.build().unwrap();

        let lhs_vectors = [row(&parameters, columns, 0), row(&parameters, columns, 1)];
        let rhs_vectors = [row(&parameters, columns, 2), row(&parameters, columns, 3)];
        let lhs_plaintexts = [row(&parameters, 1, 4), row(&parameters, 1, 5)];
        let rhs_plaintexts = [row(&parameters, 1, 6), row(&parameters, 1, 7)];
        let lhs_public = DCRTPolyMatrix::from_poly_vec(
            &parameters,
            vec![row(&parameters, columns, 0).get_row(0), row(&parameters, columns, 2).get_row(0)],
        );
        let rhs_public = DCRTPolyMatrix::from_poly_vec(
            &parameters,
            vec![row(&parameters, columns, 1).get_row(0), row(&parameters, columns, 3).get_row(0)],
        );
        let mut inputs = BTreeMap::from([
            ("lhs-public".to_owned(), RuntimeValue::matrix(lhs_public.clone())),
            ("rhs-public".to_owned(), RuntimeValue::matrix(rhs_public.clone())),
        ]);
        for slot in 0..2 {
            inputs.insert(
                format!("lhs-vector-{slot}"),
                RuntimeValue::matrix(lhs_vectors[slot].clone()),
            );
            inputs.insert(
                format!("rhs-vector-{slot}"),
                RuntimeValue::matrix(rhs_vectors[slot].clone()),
            );
            inputs.insert(
                format!("lhs-plaintext-{slot}"),
                RuntimeValue::matrix(lhs_plaintexts[slot].clone()),
            );
            inputs.insert(
                format!("rhs-plaintext-{slot}"),
                RuntimeValue::matrix(rhs_plaintexts[slot].clone()),
            );
        }
        let result = execute_graph(graph, parameters, inputs);
        for slot in 0..2 {
            assert_eq!(
                matrix_output(&result, &format!("vector-{slot}")),
                &(lhs_vectors[slot].clone() + rhs_vectors[slot].clone())
            );
            assert_eq!(
                matrix_output(&result, &format!("plaintext-{slot}")),
                &(lhs_plaintexts[slot].clone() + rhs_plaintexts[slot].clone())
            );
        }
        assert_eq!(matrix_output(&result, "public"), &(lhs_public + rhs_public));
    }
}
