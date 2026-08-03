//! Tall BGG+ encodings with shared public matrices and row-wise evaluation.
//!
//! A wire with diagonal message `X` represents
//! `C_X = S A_X - X S G + E_X`, with one BGG+ row per slot and one public
//! matrix shared by every row.  Rotation artifacts encode cyclic permutation
//! matrices only inside preprocessing; they never enter the ordinary circuit
//! wire type.  For a provisioned permutation `P`, rotation evaluates
//! `C_P G^-1(A_X) + P C_X` followed by multiplication with the inverse
//! permutation encoding.

use crate::{
    BggPublicKeyCompiler, BggPublicKeyWire, BggSamplerLayout,
    tall_rotation_encoding::{TallRotationDirection, TallRotationEncodingWires},
};
use mxx_dsl::{DslError, Family, Mat, Parallel, parallel_zip};
use mxx_ir_core::{
    IntExpr, RealExpr,
    node::{ConcatAxis, IndexRange},
    types::MatrixType,
};
use rayon::prelude::*;
use thiserror::Error;

/// One family of BGG+ rows carrying a diagonal message.
#[derive(Clone)]
pub struct BggTallEncodingWire {
    /// Encoding rows of type `(1, m)`.
    pub rows: Family<Mat>,
    /// Public matrix shared by every row.
    pub pubkey: BggPublicKeyWire,
    /// Known or hidden diagonal plaintext metadata.
    pub plaintext: BggTallPlaintext,
}

/// Plaintext metadata for a tall BGG+ wire.
#[derive(Clone)]
pub enum BggTallPlaintext {
    /// An unrevealed diagonal message.
    Hidden,
    /// One revealed scalar polynomial per diagonal position.
    Diagonal(Family<Mat>),
}

/// Compiler for row-wise tall BGG+ arithmetic.
#[derive(Clone)]
pub struct BggTallEncodingCompiler {
    /// Compiler for the public-matrix side of each operation.
    pub public_key: BggPublicKeyCompiler,
}

/// Sampling result containing all requested tall encodings and slot transforms.
#[derive(Clone)]
pub struct BggTallEncodingSample {
    /// Sampled encodings, in the same order as the supplied public keys.
    pub encodings: Vec<BggTallEncodingWire>,
    /// Matrices `R_i` defining slot secrets `s_i = s R_i`.
    pub slot_secret_matrices: Family<Mat>,
}

/// Sampler for tall BGG+ encodings.
#[derive(Clone)]
pub struct BggTallEncodingSampler {
    /// Shared BGG matrix layout.
    pub layout: BggSamplerLayout,
    /// Optional Gaussian error width; `None` produces exact test encodings.
    pub gaussian_sigma: Option<RealExpr>,
}

/// Errors produced by tall BGG+ compilation and artifact wiring.
#[derive(Debug, Error)]
pub enum TallCompileError {
    /// Input families or matrix types do not match the tall BGG layout.
    #[error("tall BGG+ inputs have incompatible counts or matrix types")]
    InvalidLayout,
    /// A rotation has zero slots or mismatched family counts.
    #[error("tall BGG+ rotation has an invalid slot layout")]
    InvalidRotationLayout,
    /// The same normalized rotation was requested more than once.
    #[error("rotation ({num_slots}, {offset}) is duplicated after normalization")]
    DuplicateRotation { num_slots: u32, offset: u32 },
    /// SIMD multiplication needs the revealed left diagonal.
    #[error("tall BGG+ multiplication requires the left diagonal plaintext")]
    MissingLeftPlaintext,
    /// Rotation needs the revealed diagonal plaintext.
    #[error("tall BGG+ rotation requires a revealed diagonal plaintext")]
    MissingRotationPlaintext,
    /// A directly provisioned rotation pair is unavailable.
    #[error("tall rotation encoding ({num_slots}, {offset}) is unavailable")]
    MissingTallRotationEncoding { num_slots: u32, offset: u32 },
    /// A parameterized rotation is missing or bound to another parameter kind.
    #[error("gate {gate} has an invalid parameterized rotation binding")]
    InvalidRotationParameter { gate: usize },
    /// DSL graph construction failed.
    #[error(transparent)]
    Dsl(#[from] DslError),
}

impl BggTallEncodingCompiler {
    /// Adds two tall encodings row by row.
    pub fn add(
        &self,
        lhs: &BggTallEncodingWire,
        rhs: &BggTallEncodingWire,
    ) -> Result<BggTallEncodingWire, TallCompileError> {
        self.binary(
            lhs,
            rhs,
            |left, right| left + right,
            |compiler, left, right| compiler.add(left, right),
        )
    }

    /// Subtracts two tall encodings row by row.
    pub fn sub(
        &self,
        lhs: &BggTallEncodingWire,
        rhs: &BggTallEncodingWire,
    ) -> Result<BggTallEncodingWire, TallCompileError> {
        self.binary(
            lhs,
            rhs,
            |left, right| left - right,
            |compiler, left, right| compiler.sub(left, right),
        )
    }

    /// Builds `C_L G^-1(A_R) + diag(x_L) C_R` with one shared decomposition.
    pub fn simd_mul(
        &self,
        lhs: &BggTallEncodingWire,
        rhs: &BggTallEncodingWire,
    ) -> Result<BggTallEncodingWire, TallCompileError> {
        validate_pair(lhs, rhs)?;
        let BggTallPlaintext::Diagonal(lhs_plaintexts) = &lhs.plaintext else {
            return Err(TallCompileError::MissingLeftPlaintext);
        };
        let decomposed_rhs = rhs
            .pubkey
            .matrix
            .clone()
            .decompose(self.public_key.base.clone(), self.public_key.digit_count.clone())
            .as_mat();
        let rows = lhs.rows.clone().parallel_zip3(
            rhs.rows.clone(),
            lhs_plaintexts.clone(),
            move |_, left, right, plaintext| left * decomposed_rhs.clone() + right * plaintext,
        )?;
        let plaintext = match &rhs.plaintext {
            BggTallPlaintext::Diagonal(rhs_plaintexts) => BggTallPlaintext::Diagonal(
                lhs_plaintexts
                    .clone()
                    .parallel_zip(rhs_plaintexts.clone(), |_, left, right| left * right)?,
            ),
            BggTallPlaintext::Hidden => BggTallPlaintext::Hidden,
        };
        Ok(BggTallEncodingWire {
            rows,
            pubkey: self.public_key.mul(&lhs.pubkey, &rhs.pubkey),
            plaintext,
        })
    }

    /// Multiplies every row and diagonal plaintext by one small scalar.
    pub fn small_scalar_mul(
        &self,
        input: &BggTallEncodingWire,
        scalar: &Mat,
    ) -> Result<BggTallEncodingWire, TallCompileError> {
        self.scalar_mul(input, scalar, scalar.clone(), false)
    }

    /// Multiplies every row by the gadget decomposition of a large scalar.
    pub fn large_scalar_mul(
        &self,
        input: &BggTallEncodingWire,
        scalar: &Mat,
    ) -> Result<BggTallEncodingWire, TallCompileError> {
        let decomposed = self.public_key.large_scalar_decomposition(&input.pubkey, scalar);
        self.scalar_mul(input, scalar, decomposed, true)
    }

    /// Applies one provisioned cyclic rotation pair in the selected direction.
    pub fn rotate(
        &self,
        input: &BggTallEncodingWire,
        rotation: &TallRotationEncodingWires,
        direction: TallRotationDirection,
    ) -> Result<BggTallEncodingWire, TallCompileError> {
        let BggTallPlaintext::Diagonal(plaintexts) = &input.plaintext else {
            return Err(TallCompileError::MissingRotationPlaintext);
        };
        let num_slots = usize::try_from(rotation.key.num_slots)
            .map_err(|_| TallCompileError::InvalidRotationLayout)?;
        if input.rows.count() != &IntExpr::constant(num_slots) ||
            plaintexts.count() != &IntExpr::constant(num_slots) ||
            rotation.c_forward.count() != &IntExpr::constant(num_slots) ||
            rotation.c_backward.count() != &IntExpr::constant(num_slots)
        {
            return Err(TallCompileError::InvalidRotationLayout);
        }
        let forward = usize::try_from(rotation.key.offset)
            .map_err(|_| TallCompileError::InvalidRotationLayout)?;
        let (offset, a_left, a_right, c_left, c_right) = match direction {
            TallRotationDirection::Forward => (
                forward,
                rotation.a_forward.clone(),
                rotation.a_backward.clone(),
                rotation.c_forward.clone(),
                rotation.c_backward.clone(),
            ),
            TallRotationDirection::Backward => (
                (num_slots - forward) % num_slots,
                rotation.a_backward.clone(),
                rotation.a_forward.clone(),
                rotation.c_backward.clone(),
                rotation.c_forward.clone(),
            ),
        };
        let rotated_rows = rotate_family(&input.rows, offset, num_slots)?;
        let rotated_plaintexts = rotate_family(plaintexts, offset, num_slots)?;
        let rotated_inverse_rows = rotate_family(&c_right, offset, num_slots)?;
        let decomposed_input = input
            .pubkey
            .matrix
            .clone()
            .decompose(self.public_key.base.clone(), self.public_key.digit_count.clone())
            .as_mat();
        let step1 = c_left.parallel_zip(rotated_rows, move |_, permutation, input| {
            permutation * decomposed_input.clone() + input
        })?;
        let decomposed_inverse = a_right
            .clone()
            .decompose(self.public_key.base.clone(), self.public_key.digit_count.clone())
            .as_mat();
        let rows = step1.parallel_zip3(
            rotated_plaintexts.clone(),
            rotated_inverse_rows,
            move |_, intermediate, plaintext, inverse| {
                intermediate * decomposed_inverse.clone() + inverse * plaintext
            },
        )?;
        let first_public = a_left *
            input
                .pubkey
                .matrix
                .clone()
                .decompose(self.public_key.base.clone(), self.public_key.digit_count.clone())
                .as_mat();
        let public_matrix = first_public *
            a_right
                .decompose(self.public_key.base.clone(), self.public_key.digit_count.clone())
                .as_mat();
        Ok(BggTallEncodingWire {
            rows,
            pubkey: BggPublicKeyWire {
                matrix: public_matrix,
                reveal_plaintext: input.pubkey.reveal_plaintext,
            },
            plaintext: BggTallPlaintext::Diagonal(rotated_plaintexts),
        })
    }

    fn binary(
        &self,
        lhs: &BggTallEncodingWire,
        rhs: &BggTallEncodingWire,
        operation: impl FnOnce(Mat, Mat) -> Mat + Copy,
        public_operation: impl FnOnce(
            &BggPublicKeyCompiler,
            &BggPublicKeyWire,
            &BggPublicKeyWire,
        ) -> BggPublicKeyWire,
    ) -> Result<BggTallEncodingWire, TallCompileError> {
        validate_pair(lhs, rhs)?;
        let rows = lhs
            .rows
            .clone()
            .parallel_zip(rhs.rows.clone(), move |_, left, right| operation(left, right))?;
        let plaintext = match (&lhs.plaintext, &rhs.plaintext) {
            (BggTallPlaintext::Diagonal(left), BggTallPlaintext::Diagonal(right)) => {
                BggTallPlaintext::Diagonal(
                    left.clone().parallel_zip(right.clone(), move |_, left, right| {
                        operation(left, right)
                    })?,
                )
            }
            _ => BggTallPlaintext::Hidden,
        };
        Ok(BggTallEncodingWire {
            rows,
            pubkey: public_operation(&self.public_key, &lhs.pubkey, &rhs.pubkey),
            plaintext,
        })
    }

    fn scalar_mul(
        &self,
        input: &BggTallEncodingWire,
        scalar: &Mat,
        row_factor: Mat,
        large: bool,
    ) -> Result<BggTallEncodingWire, TallCompileError> {
        let rows = input.rows.clone().parallel_map(move |_, row| row * row_factor.clone())?;
        let plaintext = match &input.plaintext {
            BggTallPlaintext::Hidden => BggTallPlaintext::Hidden,
            BggTallPlaintext::Diagonal(values) => {
                let scalar = scalar.clone();
                BggTallPlaintext::Diagonal(
                    values.clone().parallel_map(move |_, value| value * scalar.clone())?,
                )
            }
        };
        let pubkey = if large {
            self.public_key.large_scalar_mul(&input.pubkey, scalar)
        } else {
            self.public_key.small_scalar_mul(&input.pubkey, scalar)
        };
        Ok(BggTallEncodingWire { rows, pubkey, plaintext })
    }
}

impl BggTallEncodingWire {
    /// Stacks a compile-time-sized row family into one matrix for tests and export.
    pub fn stack(&self) -> Result<Mat, TallCompileError> {
        let IntExpr::Const(count) = self.rows.count() else {
            return Err(TallCompileError::InvalidLayout);
        };
        let count = usize::try_from(count.clone()).map_err(|_| TallCompileError::InvalidLayout)?;
        if count == 0 {
            return Err(TallCompileError::InvalidLayout);
        }
        Ok(Mat::concat(
            ConcatAxis::Rows,
            (0..count).map(|index| self.rows.get_static(index)).collect(),
        ))
    }
}

impl BggTallEncodingSampler {
    /// Samples slot transforms and packed tall BGG+ rows.
    pub fn sample(
        &self,
        secret: Mat,
        public_keys: &[BggPublicKeyWire],
        plaintexts: &[Family<Mat>],
        slot_count: IntExpr,
        supplied_slot_secrets: Option<Family<Mat>>,
    ) -> Result<BggTallEncodingSample, TallCompileError> {
        if public_keys.len() != plaintexts.len() + 1 {
            return Err(TallCompileError::InvalidLayout);
        }
        let ring = self.layout.ring();
        let secret_size = self.layout.secret_dimension;
        let columns = self.layout.public_key_columns();
        if !same_matrix_type(secret.matrix_type(), &ring.matrix_type((1, secret_size))) ||
            public_keys.par_iter().any(|key| {
                !same_matrix_type(
                    key.matrix.matrix_type(),
                    &ring.matrix_type((secret_size, columns)),
                )
            }) ||
            plaintexts.par_iter().any(|family| {
                family.count() != &slot_count ||
                    !same_matrix_type(family.element_type(), &ring.matrix_type((1, 1)))
            })
        {
            return Err(TallCompileError::InvalidLayout);
        }
        let (slot_secret_matrices, transformed_secrets) =
            if let Some(slot_secrets) = supplied_slot_secrets {
                if slot_secrets.count() != &slot_count ||
                    !same_matrix_type(
                        slot_secrets.element_type(),
                        &ring.matrix_type((secret_size, secret_size)),
                    )
                {
                    return Err(TallCompileError::InvalidLayout);
                }
                let transformed = slot_secrets.clone().parallel_map({
                    let secret = secret.clone();
                    move |_, transform| secret.clone() * transform
                })?;
                (slot_secrets, transformed)
            } else {
                let (transformed, sampled) = Parallel::range(slot_count.clone()).map_values({
                    let ring = ring.clone();
                    let secret = secret.clone();
                    move |_| {
                        let transform = ring.uniform_in((secret_size, secret_size), -1, 1);
                        (secret.clone() * transform.clone(), transform)
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
            .expect("validated tall plaintext counts")
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
        let gadget =
            ring.gadget(secret_size, self.layout.gadget_base.clone(), self.layout.digit_count);
        let sigma = self.gaussian_sigma.clone();
        let row_families = parallel_zip(
            (transformed_secrets, encoded_plaintexts),
            move |_, (slot_secret, encoded)| {
                let packed = slot_secret.clone() * packed_public.clone() -
                    encoded.tensor(slot_secret * gadget.clone()) +
                    match &sigma {
                        Some(sigma) => ring.gaussian((1, columns * count), sigma.clone()),
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
        let encodings = row_families
            .into_iter()
            .enumerate()
            .map(|(index, rows)| BggTallEncodingWire {
                rows,
                pubkey: public_keys[index].clone(),
                plaintext: if public_keys[index].reveal_plaintext {
                    BggTallPlaintext::Diagonal(if index == 0 {
                        ones.clone()
                    } else {
                        plaintexts[index - 1].clone()
                    })
                } else {
                    BggTallPlaintext::Hidden
                },
            })
            .collect();
        Ok(BggTallEncodingSample { encodings, slot_secret_matrices })
    }
}

fn validate_pair(
    lhs: &BggTallEncodingWire,
    rhs: &BggTallEncodingWire,
) -> Result<(), TallCompileError> {
    if lhs.rows.count() != rhs.rows.count() ||
        lhs.rows.element_type() != rhs.rows.element_type() ||
        lhs.pubkey.matrix.matrix_type() != rhs.pubkey.matrix.matrix_type() ||
        plaintext_count_mismatch(lhs) ||
        plaintext_count_mismatch(rhs)
    {
        return Err(TallCompileError::InvalidLayout);
    }
    Ok(())
}

fn plaintext_count_mismatch(wire: &BggTallEncodingWire) -> bool {
    match &wire.plaintext {
        BggTallPlaintext::Hidden => false,
        BggTallPlaintext::Diagonal(values) => {
            let expected = MatrixType {
                modulus: wire.rows.element_type().modulus.clone(),
                ring_dimension: wire.rows.element_type().ring_dimension.clone(),
                rows: IntExpr::constant(1),
                columns: IntExpr::constant(1),
            };
            values.count() != wire.rows.count() ||
                !same_matrix_type(values.element_type(), &expected)
        }
    }
}

pub(crate) fn rotate_family(
    rows: &Family<Mat>,
    offset: usize,
    num_slots: usize,
) -> Result<Family<Mat>, TallCompileError> {
    if num_slots == 0 || rows.count() != &IntExpr::constant(num_slots) {
        return Err(TallCompileError::InvalidRotationLayout);
    }
    let offset = offset % num_slots;
    Ok(Family::pack(
        (0..num_slots)
            .map(|destination| rows.get_static((destination + num_slots - offset) % num_slots))
            .collect(),
    )?)
}

pub(crate) fn same_matrix_type(lhs: &MatrixType, rhs: &MatrixType) -> bool {
    lhs.modulus.canonicalize() == rhs.modulus.canonicalize() &&
        lhs.ring_dimension.canonicalize() == rhs.ring_dimension.canonicalize() &&
        lhs.rows.canonicalize() == rhs.rows.canonicalize() &&
        lhs.columns.canonicalize() == rhs.columns.canonicalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BggSlotTransferArtifactCompiler, BggSlotTransferPublicKeyLowering,
        BggSlotTransferPublicSlotWires, BggTallSlotLowering, BggTallSlotPublicKeyLowering,
        CircuitCompileError, LweLookupArtifactNames, LweLookupArtifacts, LweLookupCompiler,
        LweLookupIdentity, LweLookupInvocation, LweLookupPreprocessingWires,
        LweLookupPublicKeyLowering, LweLookupTable, LweLookupTallEncodingLowering,
        PolyCircuitCompiler, TallRotationEncodingArtifactNames, TallRotationEncodingArtifacts,
        TallRotationEncodingCompiler, TallRotationEncodingKey, required_tall_rotation_encodings,
        tall_rotation_encoding::tall_rotation_public_key_tag,
        test_utils::{execute_graph, matrix_output, row},
    };
    use mxx_dsl::{DslContext, Ring};
    use mxx_gadgets::{
        circuit::{
            CircuitLoweringTypes, GateInstance, PolyCircuit, PublicLookupLowering,
            SlotOperationLowering, SlotTransferSpec, SubCircuitParamSpec, SubCircuitParamValue,
        },
        circuit_gadgets::{
            arith::{NestedRnsPoly, NestedRnsPolyContext},
            conv_mul::negacyclic_conv_mul,
        },
        test_utils::{PolyVec, execute_polyvec_circuit},
    };
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
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
        transcript::SamplingMode,
    };
    use num_bigint::BigInt;
    use std::{
        collections::{BTreeMap, BTreeSet},
        sync::Arc,
    };

    fn concrete_ring(parameters: &DCRTPolyParams) -> Ring {
        Ring::new(
            BigInt::from(parameters.modulus().as_ref().clone()),
            parameters.ring_dimension() as usize,
        )
    }

    fn public_matrix(
        parameters: &DCRTPolyParams,
        rows: usize,
        columns: usize,
        offset: usize,
    ) -> DCRTPolyMatrix {
        DCRTPolyMatrix::from_poly_vec(
            parameters,
            (0..rows)
                .map(|row_index| row(parameters, columns, offset + row_index).get_row(0))
                .collect(),
        )
    }

    struct TestTallRotationLowering {
        compiler: BggTallEncodingCompiler,
        rotation: TallRotationEncodingWires,
    }

    impl CircuitLoweringTypes for TestTallRotationLowering {
        type Wire = BggTallEncodingWire;
        type Error = CircuitCompileError;
    }

    impl SlotOperationLowering<DCRTPoly> for TestTallRotationLowering {
        fn slot_transfer(
            &mut self,
            _input: &Self::Wire,
            _source_slots: &[(u32, Option<u32>)],
            _gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            Err(CircuitCompileError::Structure("unexpected ordinary transfer".to_owned()))
        }

        fn slot_reduce(
            &mut self,
            _inputs: &[Self::Wire],
            _slot_count: usize,
            _gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            Err(CircuitCompileError::Structure("unexpected slot reduction".to_owned()))
        }

        fn slot_rotation(
            &mut self,
            input: &Self::Wire,
            offset: u32,
            num_slots: u32,
            _gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            assert_eq!((offset, num_slots), (1, self.rotation.key.num_slots));
            Ok(self.compiler.rotate(input, &self.rotation, TallRotationDirection::Forward)?)
        }
    }

    #[test]
    fn tall_arithmetic_matches_row_wise_runtime_formulas() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let secret_size = 2;
        let digits = parameters.modulus_digits();
        let columns = secret_size * digits;
        let slots = 3;
        let ring = concrete_ring(&parameters);
        let compiler = BggTallEncodingCompiler {
            public_key: BggPublicKeyCompiler {
                ring: ring.clone(),
                base: BigInt::from(1u64 << parameters.base_bits()).into(),
                digit_count: digits.into(),
            },
        };
        let wire = |prefix: &str| BggTallEncodingWire {
            rows: Family::pack(
                (0..slots)
                    .map(|slot| ring.input(format!("{prefix}-row-{slot}"), (1, columns)))
                    .collect(),
            )
            .unwrap(),
            pubkey: BggPublicKeyWire {
                matrix: ring.input(format!("{prefix}-public"), (secret_size, columns)),
                reveal_plaintext: true,
            },
            plaintext: BggTallPlaintext::Diagonal(
                Family::pack(
                    (0..slots)
                        .map(|slot| ring.input(format!("{prefix}-plain-{slot}"), (1, 1)))
                        .collect(),
                )
                .unwrap(),
            ),
        };
        let left = wire("left");
        let right = wire("right");
        let sum = compiler.add(&left, &right).unwrap();
        let product = compiler.simd_mul(&left, &right).unwrap();
        let mut context = DslContext::new("tall-arithmetic-runtime")
            .output("sum-public", sum.pubkey.matrix)
            .unwrap()
            .output("product-public", product.pubkey.matrix)
            .unwrap();
        let BggTallPlaintext::Diagonal(product_plaintexts) = product.plaintext else {
            panic!("known inputs must keep a diagonal plaintext")
        };
        for slot in 0..slots {
            context = context
                .output(format!("sum-row-{slot}"), sum.rows.get_static(slot))
                .unwrap()
                .output(format!("product-row-{slot}"), product.rows.get_static(slot))
                .unwrap()
                .output(format!("product-plain-{slot}"), product_plaintexts.get_static(slot))
                .unwrap();
        }
        let graph = context.build().unwrap();

        let left_public = public_matrix(&parameters, secret_size, columns, 2);
        let right_public = public_matrix(&parameters, secret_size, columns, 5);
        let left_rows =
            (0..slots).map(|slot| row(&parameters, columns, 8 + slot)).collect::<Vec<_>>();
        let right_rows =
            (0..slots).map(|slot| row(&parameters, columns, 12 + slot)).collect::<Vec<_>>();
        let left_plaintexts =
            (0..slots).map(|slot| row(&parameters, 1, 16 + slot)).collect::<Vec<_>>();
        let right_plaintexts =
            (0..slots).map(|slot| row(&parameters, 1, 20 + slot)).collect::<Vec<_>>();
        let mut inputs = BTreeMap::from([
            ("left-public".to_owned(), RuntimeValue::matrix(left_public.clone())),
            ("right-public".to_owned(), RuntimeValue::matrix(right_public.clone())),
        ]);
        for slot in 0..slots {
            inputs
                .insert(format!("left-row-{slot}"), RuntimeValue::matrix(left_rows[slot].clone()));
            inputs.insert(
                format!("right-row-{slot}"),
                RuntimeValue::matrix(right_rows[slot].clone()),
            );
            inputs.insert(
                format!("left-plain-{slot}"),
                RuntimeValue::matrix(left_plaintexts[slot].clone()),
            );
            inputs.insert(
                format!("right-plain-{slot}"),
                RuntimeValue::matrix(right_plaintexts[slot].clone()),
            );
        }
        let result = execute_graph(graph, parameters, inputs);
        assert_eq!(
            matrix_output(&result, "sum-public"),
            &(left_public.clone() + right_public.clone())
        );
        assert_eq!(
            matrix_output(&result, "product-public"),
            &left_public.mul_decompose(&right_public)
        );
        for slot in 0..slots {
            assert_eq!(
                matrix_output(&result, &format!("sum-row-{slot}")),
                &(left_rows[slot].clone() + right_rows[slot].clone())
            );
            assert_eq!(
                matrix_output(&result, &format!("product-row-{slot}")),
                &(left_rows[slot].mul_decompose(&right_public) +
                    right_rows[slot].clone() * left_plaintexts[slot].entry(0, 0))
            );
            assert_eq!(
                matrix_output(&result, &format!("product-plain-{slot}")),
                &(left_plaintexts[slot].clone() * right_plaintexts[slot].clone())
            );
        }
    }

    #[test]
    fn direct_tall_rotation_encoding_matches_the_two_step_matrix_formula() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let secret_size = 2;
        let digits = parameters.modulus_digits();
        let columns = secret_size * digits;
        let slots = 4;
        let ring = concrete_ring(&parameters);
        let compiler = BggTallEncodingCompiler {
            public_key: BggPublicKeyCompiler {
                ring: ring.clone(),
                base: BigInt::from(1u64 << parameters.base_bits()).into(),
                digit_count: digits.into(),
            },
        };
        let input = BggTallEncodingWire {
            rows: Family::pack(
                (0..slots)
                    .map(|slot| ring.input(format!("input-row-{slot}"), (1, columns)))
                    .collect(),
            )
            .unwrap(),
            pubkey: BggPublicKeyWire {
                matrix: ring.input("input-public", (secret_size, columns)),
                reveal_plaintext: true,
            },
            plaintext: BggTallPlaintext::Diagonal(
                Family::pack(
                    (0..slots)
                        .map(|slot| ring.input(format!("input-plain-{slot}"), (1, 1)))
                        .collect(),
                )
                .unwrap(),
            ),
        };
        let rotation = TallRotationEncodingWires {
            key: TallRotationEncodingKey { num_slots: slots as u32, offset: 1 },
            a_forward: ring.input("a-forward", (secret_size, columns)),
            a_backward: ring.input("a-backward", (secret_size, columns)),
            c_forward: Family::pack(
                (0..slots)
                    .map(|slot| ring.input(format!("c-forward-{slot}"), (1, columns)))
                    .collect(),
            )
            .unwrap(),
            c_backward: Family::pack(
                (0..slots)
                    .map(|slot| ring.input(format!("c-backward-{slot}"), (1, columns)))
                    .collect(),
            )
            .unwrap(),
        };
        let output = compiler.rotate(&input, &rotation, TallRotationDirection::Forward).unwrap();
        let BggTallPlaintext::Diagonal(output_plaintexts) = output.plaintext else {
            panic!("rotation keeps revealed diagonal plaintexts")
        };
        let mut context = DslContext::new("tall-rotation-runtime")
            .output("public", output.pubkey.matrix)
            .unwrap();
        for slot in 0..slots {
            context = context
                .output(format!("row-{slot}"), output.rows.get_static(slot))
                .unwrap()
                .output(format!("plain-{slot}"), output_plaintexts.get_static(slot))
                .unwrap();
        }

        let input_public = public_matrix(&parameters, secret_size, columns, 1);
        let a_forward = public_matrix(&parameters, secret_size, columns, 4);
        let a_backward = public_matrix(&parameters, secret_size, columns, 7);
        let input_rows =
            (0..slots).map(|slot| row(&parameters, columns, 10 + slot)).collect::<Vec<_>>();
        let plaintexts = (0..slots).map(|slot| row(&parameters, 1, 15 + slot)).collect::<Vec<_>>();
        let c_forward =
            (0..slots).map(|slot| row(&parameters, columns, 20 + slot)).collect::<Vec<_>>();
        let c_backward =
            (0..slots).map(|slot| row(&parameters, columns, 25 + slot)).collect::<Vec<_>>();
        let mut inputs = BTreeMap::from([
            ("input-public".to_owned(), RuntimeValue::matrix(input_public.clone())),
            ("a-forward".to_owned(), RuntimeValue::matrix(a_forward.clone())),
            ("a-backward".to_owned(), RuntimeValue::matrix(a_backward.clone())),
        ]);
        for slot in 0..slots {
            inputs.insert(
                format!("input-row-{slot}"),
                RuntimeValue::matrix(input_rows[slot].clone()),
            );
            inputs.insert(
                format!("input-plain-{slot}"),
                RuntimeValue::matrix(plaintexts[slot].clone()),
            );
            inputs
                .insert(format!("c-forward-{slot}"), RuntimeValue::matrix(c_forward[slot].clone()));
            inputs.insert(
                format!("c-backward-{slot}"),
                RuntimeValue::matrix(c_backward[slot].clone()),
            );
        }
        let result = execute_graph(context.build().unwrap(), parameters, inputs);
        assert_eq!(
            matrix_output(&result, "public"),
            &a_forward.mul_decompose(&input_public).mul_decompose(&a_backward)
        );
        for destination in 0..slots {
            let source = (destination + slots - 1) % slots;
            let step1 =
                c_forward[destination].mul_decompose(&input_public) + input_rows[source].clone();
            let expected = step1.mul_decompose(&a_backward) +
                c_backward[source].clone() * plaintexts[source].entry(0, 0);
            assert_eq!(matrix_output(&result, &format!("row-{destination}")), &expected);
            assert_eq!(
                matrix_output(&result, &format!("plain-{destination}")),
                &plaintexts[source]
            );
        }
    }

    #[test]
    fn tall_sampler_uses_supplied_slot_transforms_in_the_bgg_formula() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let secret_size = 2;
        let slots = 3;
        let layout = BggSamplerLayout {
            modulus: BigInt::from(parameters.modulus().as_ref().clone()).into(),
            ring_dimension: (parameters.ring_dimension() as usize).into(),
            secret_dimension: secret_size,
            digit_count: parameters.modulus_digits(),
            gadget_base: BigInt::from(1u64 << parameters.base_bits()).into(),
        };
        let ring = layout.ring();
        let columns = layout.public_key_columns();
        let public_keys = [
            BggPublicKeyWire {
                matrix: ring.input("public-one", (secret_size, columns)),
                reveal_plaintext: true,
            },
            BggPublicKeyWire {
                matrix: ring.input("public-message", (secret_size, columns)),
                reveal_plaintext: true,
            },
        ];
        let plaintexts = Family::pack(
            (0..slots).map(|slot| ring.input(format!("plaintext-{slot}"), (1, 1))).collect(),
        )
        .unwrap();
        let transforms = Family::pack(
            (0..slots)
                .map(|slot| ring.input(format!("transform-{slot}"), (secret_size, secret_size)))
                .collect(),
        )
        .unwrap();
        let sample = BggTallEncodingSampler { layout: layout.clone(), gaussian_sigma: None }
            .sample(
                ring.input("secret", (1, secret_size)),
                &public_keys,
                &[plaintexts],
                slots.into(),
                Some(transforms),
            )
            .unwrap();
        let mut context = DslContext::new("tall-sampler-runtime");
        for slot in 0..slots {
            context = context
                .output(format!("row-{slot}"), sample.encodings[1].rows.get_static(slot))
                .unwrap()
                .private_output(
                    format!("transform-out-{slot}"),
                    sample.slot_secret_matrices.get_static(slot),
                )
                .unwrap();
        }

        let secret = row(&parameters, secret_size, 1);
        let public_one = public_matrix(&parameters, secret_size, columns, 4);
        let public_message = public_matrix(&parameters, secret_size, columns, 7);
        let plaintext_values =
            (0..slots).map(|slot| row(&parameters, 1, 10 + slot)).collect::<Vec<_>>();
        let transform_values = (0..slots)
            .map(|slot| public_matrix(&parameters, secret_size, secret_size, 14 + slot * 2))
            .collect::<Vec<_>>();
        let mut inputs = BTreeMap::from([
            ("secret".to_owned(), RuntimeValue::matrix(secret.clone())),
            ("public-one".to_owned(), RuntimeValue::matrix(public_one)),
            ("public-message".to_owned(), RuntimeValue::matrix(public_message.clone())),
        ]);
        for slot in 0..slots {
            inputs.insert(
                format!("plaintext-{slot}"),
                RuntimeValue::matrix(plaintext_values[slot].clone()),
            );
            inputs.insert(
                format!("transform-{slot}"),
                RuntimeValue::matrix(transform_values[slot].clone()),
            );
        }
        let result = execute_graph(context.build().unwrap(), parameters.clone(), inputs);
        let gadget = DCRTPolyMatrix::gadget_matrix(&parameters, secret_size);
        for slot in 0..slots {
            assert_eq!(
                matrix_output(&result, &format!("transform-out-{slot}")),
                &transform_values[slot]
            );
            let slot_secret = secret.clone() * transform_values[slot].clone();
            let expected = slot_secret.clone() * public_message.clone() -
                plaintext_values[slot].clone().tensor(&(slot_secret * gadget.clone()));
            assert_eq!(matrix_output(&result, &format!("row-{slot}")), &expected);
        }
    }

    #[test]
    fn tall_rotation_encoding_identity_and_artifact_names_are_stable() {
        assert_eq!(TallRotationEncodingKey::normalize(4, 0).unwrap(), None);
        assert_eq!(TallRotationEncodingKey::normalize(4, 4).unwrap(), None);
        assert_eq!(
            TallRotationEncodingKey::normalize(4, 5).unwrap(),
            Some(TallRotationEncodingKey { num_slots: 4, offset: 1 })
        );
        let n4 = TallRotationEncodingArtifactNames::for_key(TallRotationEncodingKey {
            num_slots: 4,
            offset: 1,
        });
        let n8 = TallRotationEncodingArtifactNames::for_key(TallRotationEncodingKey {
            num_slots: 8,
            offset: 1,
        });
        assert_ne!(n4, n8);
    }

    #[test]
    fn tall_rotation_encoding_artifacts_roundtrip_and_match_cross_row_secrets() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let secret_size = 2;
        let slots = 4;
        let digits = parameters.modulus_digits();
        let columns = secret_size * digits;
        let compiler = TallRotationEncodingCompiler {
            modulus: BigInt::from(parameters.modulus().as_ref().clone()).into(),
            ring_dimension: (parameters.ring_dimension() as usize).into(),
            secret_size,
            slot_count: slots,
            gadget_base: BigInt::from(1u64 << parameters.base_bits()).into(),
            digit_count: digits,
            error_sigma: RealExpr::from_integer(0),
        };
        let ring = compiler.ring();
        let transforms = Family::pack(
            (0..slots)
                .map(|slot| ring.input(format!("transform-{slot}"), (secret_size, secret_size)))
                .collect(),
        )
        .unwrap();
        let preprocessing = compiler
            .preprocess(
                ring.bytes_input("hash-key", 32),
                ring.input("secret", (1, secret_size)),
                transforms,
                &[1, 3],
            )
            .unwrap();
        let producer = compiler
            .export_preprocessing(DslContext::new("rotation-producer"), preprocessing)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();

        let secret = row(&parameters, secret_size, 2);
        let transform_values = (0..slots)
            .map(|slot| public_matrix(&parameters, secret_size, secret_size, 5 + slot * 2))
            .collect::<Vec<_>>();
        let hash_key = [0x42; 32];
        let mut producer_inputs = BTreeMap::from([
            ("hash-key".to_owned(), RuntimeValue::Bytes(hash_key.to_vec())),
            ("secret".to_owned(), RuntimeValue::matrix(secret.clone())),
        ]);
        for (slot, transform) in transform_values.iter().enumerate() {
            producer_inputs
                .insert(format!("transform-{slot}"), RuntimeValue::matrix(transform.clone()));
        }
        let mut store = MemoryArtifactStore::default();
        let mut backend = cpu_backend([parameters.clone()]);
        let produced =
            execute(&producer, &mut backend, producer_inputs, &mut store, SamplingMode::Fresh)
                .unwrap();
        let production_id = produced.production_id.expect("artifact production");
        let manifest = store.manifest(&production_id).unwrap().clone();
        assert_eq!(manifest.artifacts.len(), 8);

        let artifacts = TallRotationEncodingArtifacts {
            production_id: production_id.clone(),
            slot_count: slots as u32,
        };
        let mut context = DslContext::new("rotation-consumer");
        for offset in [1, 3] {
            let rotation = compiler.import_artifacts(&artifacts, offset).unwrap().unwrap();
            context = context
                .output(format!("a-forward-{offset}"), rotation.a_forward)
                .unwrap()
                .output(format!("a-backward-{offset}"), rotation.a_backward)
                .unwrap();
            for slot in 0..slots {
                context = context
                    .output(
                        format!("c-forward-{offset}-{slot}"),
                        rotation.c_forward.get_static(slot),
                    )
                    .unwrap()
                    .output(
                        format!("c-backward-{offset}-{slot}"),
                        rotation.c_backward.get_static(slot),
                    )
                    .unwrap();
            }
        }
        let consumer = context
            .build()
            .unwrap()
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production_id, manifest)]),
            )
            .unwrap();
        let consumed =
            execute(&consumer, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .unwrap();

        let hash = DCRTPolyHashSampler::<keccak_asm::Keccak256>::new();
        let gadget = DCRTPolyMatrix::gadget_matrix(&parameters, secret_size);
        let transformed = transform_values
            .iter()
            .map(|transform| secret.clone() * transform.clone())
            .collect::<Vec<_>>();
        for offset in [1usize, 3] {
            let key = TallRotationEncodingKey { num_slots: slots as u32, offset: offset as u32 };
            let forward_tag = format!("bgg_tall_rotation_n{slots}_r{offset}_forward");
            let backward_tag = format!("bgg_tall_rotation_n{slots}_r{offset}_backward");
            let expected_forward = hash.sample_hash(
                &parameters,
                hash_key,
                forward_tag,
                secret_size,
                columns,
                DistType::FinRingDist,
            );
            let expected_backward = hash.sample_hash(
                &parameters,
                hash_key,
                backward_tag,
                secret_size,
                columns,
                DistType::FinRingDist,
            );
            let _ = key;
            assert_eq!(matrix_output(&consumed, &format!("a-forward-{offset}")), &expected_forward);
            assert_eq!(
                matrix_output(&consumed, &format!("a-backward-{offset}")),
                &expected_backward
            );
            for slot in 0..slots {
                let forward_source = (slot + slots - offset) % slots;
                let backward_source = (slot + offset) % slots;
                assert_eq!(
                    matrix_output(&consumed, &format!("c-forward-{offset}-{slot}")),
                    &(transformed[slot].clone() * expected_forward.clone() -
                        transformed[forward_source].clone() * gadget.clone())
                );
                assert_eq!(
                    matrix_output(&consumed, &format!("c-backward-{offset}-{slot}")),
                    &(transformed[slot].clone() * expected_backward.clone() -
                        transformed[backward_source].clone() * gadget.clone())
                );
            }
        }
    }

    #[test]
    fn tall_rotation_encoding_public_key_pass_matches_lookup_input() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let secret_size = 2;
        let slots = 4;
        let digits = parameters.modulus_digits();
        let columns = secret_size * digits;
        let ring = concrete_ring(&parameters);
        let public_compiler = BggPublicKeyCompiler {
            ring: ring.clone(),
            base: BigInt::from(1u64 << parameters.base_bits()).into(),
            digit_count: digits.into(),
        };
        let hash_key = ring.bytes_input("hash-key", 32);
        let key = TallRotationEncodingKey { num_slots: slots as u32, offset: 1 };
        let input_public = BggPublicKeyWire {
            matrix: ring.input("input-public", (secret_size, columns)),
            reveal_plaintext: true,
        };
        let rotation = TallRotationEncodingWires {
            key,
            a_forward: ring.hash_matrix(
                hash_key.clone(),
                tall_rotation_public_key_tag(key, false),
                (secret_size, columns),
            ),
            a_backward: ring.hash_matrix(
                hash_key.clone(),
                tall_rotation_public_key_tag(key, true),
                (secret_size, columns),
            ),
            c_forward: Family::pack((0..slots).map(|_| ring.zero((1, columns))).collect()).unwrap(),
            c_backward: Family::pack((0..slots).map(|_| ring.zero((1, columns))).collect())
                .unwrap(),
        };
        let tall_input = BggTallEncodingWire {
            rows: Family::pack((0..slots).map(|_| ring.zero((1, columns))).collect()).unwrap(),
            pubkey: input_public.clone(),
            plaintext: BggTallPlaintext::Diagonal(
                Family::pack((0..slots).map(|_| ring.zero((1, 1))).collect()).unwrap(),
            ),
        };
        let public_key_type = input_public.matrix.matrix_type().clone();
        let mut public_lowering = BggTallSlotPublicKeyLowering {
            inner: BggSlotTransferPublicKeyLowering {
                compiler: public_compiler.clone(),
                hash_key,
                public_key_type,
                configured_slot_count: slots,
                requests: Vec::new(),
            },
        };
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input_gate = circuit.input(1);
        let rotated = circuit.slot_rotation_gate(input_gate, 1, slots);
        circuit.output([rotated]);
        let circuit_compiler = crate::PolyCircuitCompiler { public_key: public_compiler.clone() };
        let mut no_public_lookup = crate::NoPublicLookup::default();
        let public_output = circuit_compiler
            .compile_public_keys_with_lowerings(
                &circuit,
                BggPublicKeyWire {
                    matrix: ring.zero((secret_size, columns)),
                    reveal_plaintext: true,
                },
                [input_public],
                &mut no_public_lookup,
                &mut public_lowering,
            )
            .unwrap()
            .remove(0);
        assert!(public_lowering.inner.requests.is_empty());
        let mut tall_slots = TestTallRotationLowering {
            compiler: BggTallEncodingCompiler { public_key: public_compiler },
            rotation,
        };
        let mut no_public_lookup = crate::NoPublicLookup::default();
        let tall_output = circuit_compiler
            .compile_tall_encodings_with_lowerings(
                &circuit,
                tall_input.clone(),
                [tall_input],
                &mut no_public_lookup,
                &mut tall_slots,
            )
            .unwrap()
            .remove(0);
        let graph = DslContext::new("rotation-public-key-consistency")
            .output("public-pass", public_output.matrix)
            .unwrap()
            .output("encoding-pass", tall_output.pubkey.matrix)
            .unwrap()
            .build()
            .unwrap();
        let result = execute_graph(
            graph,
            parameters.clone(),
            BTreeMap::from([
                ("hash-key".to_owned(), RuntimeValue::Bytes(vec![0x51; 32])),
                (
                    "input-public".to_owned(),
                    RuntimeValue::matrix(public_matrix(&parameters, secret_size, columns, 3)),
                ),
            ]),
        );
        assert_eq!(matrix_output(&result, "public-pass"), matrix_output(&result, "encoding-pass"));
    }

    #[test]
    fn ordinary_slot_transfer_compiles_tall_rows() {
        let artifact = BggSlotTransferArtifactCompiler {
            modulus: 257.into(),
            ring_dimension: 8.into(),
            secret_size: 1,
            slot_count: 2,
            digit_count: 2,
            chunk_columns: 2,
            gadget_base: 4.into(),
            trapdoor_sigma: RealExpr::from_integer(5),
            error_sigma: RealExpr::from_integer(3),
        };
        let ring = Ring::new(257, 8);
        let hash_key = ring.bytes_input("hash-key", 32);
        let base = artifact.build_base().unwrap();
        let slot_wires = artifact.build_slots(hash_key.clone(), &base).unwrap();
        let public_slots = BggSlotTransferPublicSlotWires {
            public_keys: slot_wires.public_keys.clone(),
            b0_preimage_chunks: slot_wires.b0_preimage_chunks.clone(),
            b1_preimage_chunks: slot_wires.b1_preimage_chunks.clone(),
        };

        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input_gate = circuit.input(1).as_single_wire();
        let transferred = circuit.slot_transfer_gate(input_gate, &[(1, None), (0, Some(2))]);
        circuit.output([transferred]);
        let public_compiler =
            BggPublicKeyCompiler { ring: ring.clone(), base: 4.into(), digit_count: 2.into() };
        let one_public =
            BggPublicKeyWire { matrix: ring.input("one-public", (1, 2)), reveal_plaintext: true };
        let input_public =
            BggPublicKeyWire { matrix: ring.input("input-public", (1, 2)), reveal_plaintext: true };
        let mut public_lowering = BggSlotTransferPublicKeyLowering {
            compiler: public_compiler.clone(),
            hash_key: hash_key.clone(),
            public_key_type: ring.matrix_type((1, 2)),
            configured_slot_count: 2,
            requests: Vec::new(),
        };
        let circuit_compiler = crate::PolyCircuitCompiler { public_key: public_compiler.clone() };
        let mut no_lookup = crate::NoPublicLookup::default();
        circuit_compiler
            .compile_public_keys_with_lowerings(
                &circuit,
                one_public.clone(),
                [input_public.clone()],
                &mut no_lookup,
                &mut public_lowering,
            )
            .unwrap();
        let gate_wires =
            artifact.build_gate_preimages(&base, &slot_wires, &public_lowering.requests).unwrap();
        let input = BggTallEncodingWire {
            rows: ring.input_family("rows", 2, (1, 2)),
            pubkey: input_public,
            plaintext: BggTallPlaintext::Diagonal(ring.input_family("plaintexts", 2, (1, 1))),
        };
        let one = BggTallEncodingWire {
            rows: ring.input_family("one-rows", 2, (1, 2)),
            pubkey: one_public,
            plaintext: BggTallPlaintext::Diagonal(ring.input_family("one-plaintexts", 2, (1, 1))),
        };
        let mut lowering = BggTallSlotLowering {
            compiler: BggTallEncodingCompiler { public_key: public_compiler },
            artifact,
            hash_key,
            c_b0: ring.input("c-b0", (1, 4)),
            slots: public_slots,
            gates: gate_wires,
            rotations: BTreeMap::new(),
        };
        let mut no_lookup = crate::NoPublicLookup::default();
        let output = circuit_compiler
            .compile_tall_encodings_with_lowerings(
                &circuit,
                one,
                [input],
                &mut no_lookup,
                &mut lowering,
            )
            .unwrap()
            .remove(0);
        let built = DslContext::new("tall-slot-transfer")
            .family_output("rows", output.rows)
            .unwrap()
            .output("public", output.pubkey.matrix)
            .unwrap()
            .build()
            .unwrap();
        let elaborated = built.elaborate(&ParamEnv::default()).unwrap();
        assert!(elaborated.wire(&elaborated.outputs["rows"]).unwrap().family.is_some());
        assert!(elaborated.wire(&elaborated.outputs["public"]).unwrap().expression.is_some());
    }

    #[test]
    fn required_tall_rotation_encodings_resolve_subcircuit_parameters() {
        let mut child = PolyCircuit::<DCRTPoly>::new();
        let rotation_param =
            child.register_sub_circuit_param(SubCircuitParamSpec::SlotTransfer { max_scalar: 1 });
        let child_input = child.input(1);
        let child_output = child.slot_transfer_gate_param(child_input, rotation_param);
        child.output([child_output]);

        let mut parent = PolyCircuit::<DCRTPoly>::new();
        let child_id = parent.register_sub_circuit(child);
        let input = parent.input(1);
        let first = parent.call_sub_circuit_with_bindings(
            child_id,
            [input],
            &[SubCircuitParamValue::SlotTransfer(SlotTransferSpec::rotation(5, 4))],
        )[0];
        let second = parent.call_sub_circuit_with_bindings(
            child_id,
            [first],
            &[SubCircuitParamValue::SlotTransfer(SlotTransferSpec::rotation(3, 8))],
        )[0];
        parent.output([second]);

        let required = required_tall_rotation_encodings(&parent).unwrap();
        assert_eq!(
            required,
            BTreeSet::from([
                TallRotationEncodingKey { num_slots: 4, offset: 1 },
                TallRotationEncodingKey { num_slots: 8, offset: 3 },
            ])
        );
    }

    #[derive(Default)]
    struct RecordingPublicLookups {
        identities: Vec<(LweLookupIdentity, usize)>,
    }

    impl CircuitLoweringTypes for RecordingPublicLookups {
        type Wire = BggPublicKeyWire;
        type Error = CircuitCompileError;
    }

    impl<P: mxx_gadgets::Poly> PublicLookupLowering<P> for RecordingPublicLookups {
        fn public_lookup(
            &mut self,
            _circuit: &PolyCircuit<P>,
            lookup_id: usize,
            input: &Self::Wire,
            gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            self.identities.push((
                LweLookupIdentity {
                    call_path: gate.call_path().to_vec(),
                    gate: gate.local_gate().index(),
                    occurrence: gate.operation_occurrence(),
                    lookup: lookup_id,
                    slot: None,
                },
                lookup_id,
            ));
            Ok(input.clone())
        }
    }

    #[derive(Default)]
    struct PassthroughPublicSlots;

    impl CircuitLoweringTypes for PassthroughPublicSlots {
        type Wire = BggPublicKeyWire;
        type Error = CircuitCompileError;
    }

    impl<P: mxx_gadgets::Poly> SlotOperationLowering<P> for PassthroughPublicSlots {
        fn slot_transfer(
            &mut self,
            input: &Self::Wire,
            _source_slots: &[(u32, Option<u32>)],
            _gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            Ok(input.clone())
        }

        fn slot_reduce(
            &mut self,
            inputs: &[Self::Wire],
            _slot_count: usize,
            _gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            Ok(inputs.first().expect("slot reduction input").clone())
        }

        fn slot_rotation(
            &mut self,
            input: &Self::Wire,
            _offset: u32,
            _num_slots: u32,
            _gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            Ok(input.clone())
        }
    }

    fn assert_production_tall_artifacts_cover_circuit(
        circuit: &PolyCircuit<DCRTPoly>,
        parameters: &DCRTPolyParams,
        physical_slots: usize,
    ) -> Vec<Vec<DCRTPolyMatrix>> {
        let ring = concrete_ring(parameters);
        let digits = parameters.modulus_digits();
        let public_key_compiler = BggPublicKeyCompiler {
            ring: ring.clone(),
            base: BigInt::from(1u64 << parameters.base_bits()).into(),
            digit_count: digits.into(),
        };
        let circuit_compiler = PolyCircuitCompiler { public_key: public_key_compiler.clone() };
        let public_key =
            || BggPublicKeyWire { matrix: ring.zero((1, digits)), reveal_plaintext: true };

        let mut recording_lookups = RecordingPublicLookups::default();
        let mut passthrough_slots = PassthroughPublicSlots;
        circuit_compiler
            .compile_public_keys_with_lowerings(
                circuit,
                public_key(),
                (0..circuit.num_input()).map(|_| public_key()),
                &mut recording_lookups,
                &mut passthrough_slots,
            )
            .expect("lookup identity discovery");
        assert!(!recording_lookups.identities.is_empty());

        let lookup_compilers = recording_lookups
            .identities
            .into_iter()
            .map(|(identity, lookup_id)| {
                let table = LweLookupTable::from_public_lut(
                    parameters,
                    circuit.lookup_table(lookup_id).as_ref(),
                )
                .expect("nested-RNS lookup table");
                LweLookupCompiler {
                    identity,
                    table,
                    public_key_type: ring.matrix_type((1, digits)),
                    low_matrix_type: ring.matrix_type((digits, digits)),
                    high_matrix_type: ring.matrix_type((digits + 2, digits)),
                    gadget_base: BigInt::from(1u64 << parameters.base_bits()).into(),
                    digit_count: digits.into(),
                }
            })
            .collect::<Vec<_>>();

        let mut artifact_context = DslContext::new("packed-nested-rns-lookup-artifacts");
        for lookup in &lookup_compilers {
            let wires = LweLookupPreprocessingWires {
                output_public_key: ring.zero((1, digits)),
                low_matrices: Family::pack(
                    (0..lookup.table.len()).map(|_| ring.zero((digits, digits))).collect(),
                )
                .expect("lookup low artifact family"),
                high_matrices: Family::pack(
                    (0..lookup.table.len()).map(|_| ring.zero((digits + 2, digits))).collect(),
                )
                .expect("lookup high artifact family"),
            };
            artifact_context = lookup
                .export_preprocessing(
                    artifact_context,
                    wires,
                    &LweLookupArtifactNames::for_compiler(lookup),
                )
                .expect("lookup artifact outputs");
        }
        let artifact_graph = artifact_context
            .build()
            .expect("lookup artifact graph")
            .validate(&ParamEnv::default())
            .expect("lookup artifact validation");
        let mut store = MemoryArtifactStore::default();
        let mut backend = cpu_backend([parameters.clone()]);
        let artifact_result = execute(
            &artifact_graph,
            &mut backend,
            BTreeMap::new(),
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("lookup artifact production");
        let production_id = artifact_result.production_id.expect("lookup production id");
        let manifest = store.manifest(&production_id).expect("lookup manifest").clone();
        let invocations = lookup_compilers
            .into_iter()
            .map(|lookup| {
                LweLookupInvocation::bind(
                    lookup.clone(),
                    LweLookupArtifacts::for_compiler(production_id.clone(), &lookup),
                    parameters,
                    circuit,
                )
                .expect("nested-RNS lookup invocation")
            })
            .collect::<Vec<_>>();

        let hash_key = ring.bytes_input("packed-artifact-hash-key", 32);
        let mut public_lookup =
            LweLookupPublicKeyLowering::new(invocations.clone()).expect("public lookup lowering");
        let mut public_slots = BggTallSlotPublicKeyLowering {
            inner: BggSlotTransferPublicKeyLowering {
                compiler: public_key_compiler.clone(),
                hash_key: hash_key.clone(),
                public_key_type: ring.matrix_type((1, digits)),
                configured_slot_count: physical_slots,
                requests: Vec::new(),
            },
        };
        circuit_compiler
            .compile_public_keys_with_lowerings(
                circuit,
                public_key(),
                (0..circuit.num_input()).map(|_| public_key()),
                &mut public_lookup,
                &mut public_slots,
            )
            .expect("production public-key lowering");
        assert!(!public_slots.inner.requests.is_empty());

        let artifact = BggSlotTransferArtifactCompiler {
            modulus: BigInt::from(parameters.modulus().as_ref().clone()).into(),
            ring_dimension: (parameters.ring_dimension() as usize).into(),
            secret_size: 1,
            slot_count: physical_slots,
            digit_count: digits,
            chunk_columns: digits.max(1),
            gadget_base: BigInt::from(1u64 << parameters.base_bits()).into(),
            trapdoor_sigma: RealExpr::from_integer(5),
            error_sigma: RealExpr::from_integer(0),
        };
        let base = artifact.build_base().expect("slot base artifacts");
        let slot_wires =
            artifact.build_slots(hash_key.clone(), &base).expect("slot preprocessing artifacts");
        let gate_wires = artifact
            .build_gate_preimages(&base, &slot_wires, &public_slots.inner.requests)
            .expect("per-gate slot artifacts");
        let slots = BggSlotTransferPublicSlotWires {
            public_keys: slot_wires.public_keys,
            b0_preimage_chunks: slot_wires.b0_preimage_chunks,
            b1_preimage_chunks: slot_wires.b1_preimage_chunks,
        };
        let rotations = required_tall_rotation_encodings(circuit)
            .expect("required tall rotations")
            .into_iter()
            .map(|key| {
                let wire = TallRotationEncodingWires {
                    key,
                    a_forward: ring.hash_matrix(
                        hash_key.clone(),
                        tall_rotation_public_key_tag(key, false),
                        (1, digits),
                    ),
                    a_backward: ring.hash_matrix(
                        hash_key.clone(),
                        tall_rotation_public_key_tag(key, true),
                        (1, digits),
                    ),
                    c_forward: Family::pack(
                        (0..physical_slots).map(|_| ring.zero((1, digits))).collect(),
                    )
                    .expect("forward rotation rows"),
                    c_backward: Family::pack(
                        (0..physical_slots).map(|_| ring.zero((1, digits))).collect(),
                    )
                    .expect("backward rotation rows"),
                };
                (key, wire)
            })
            .collect::<BTreeMap<_, _>>();

        let zero_tall = || BggTallEncodingWire {
            rows: Family::pack((0..physical_slots).map(|_| ring.zero((1, digits))).collect())
                .expect("packed rows"),
            pubkey: public_key(),
            plaintext: BggTallPlaintext::Diagonal(
                Family::pack((0..physical_slots).map(|_| ring.zero((1, 1))).collect())
                    .expect("packed plaintexts"),
            ),
        };
        let one = BggTallEncodingWire {
            plaintext: BggTallPlaintext::Diagonal(
                Family::pack((0..physical_slots).map(|_| ring.identity(1)).collect())
                    .expect("packed ones"),
            ),
            ..zero_tall()
        };
        let mut lookup = LweLookupTallEncodingLowering::new(
            invocations,
            Family::pack((0..physical_slots).map(|_| ring.zero((1, digits + 2))).collect())
                .expect("lookup helper rows"),
        )
        .expect("tall lookup lowering");
        let mut slots = BggTallSlotLowering {
            compiler: BggTallEncodingCompiler { public_key: public_key_compiler },
            artifact: artifact.clone(),
            hash_key,
            c_b0: ring.zero((1, artifact.b0_public_columns())),
            slots,
            gates: gate_wires,
            rotations,
        };
        let outputs = circuit_compiler
            .compile_tall_encodings_with_lowerings(
                circuit,
                one,
                (0..circuit.num_input()).map(|_| zero_tall()),
                &mut lookup,
                &mut slots,
            )
            .expect("generated production artifacts must cover every packed gate");
        assert_eq!(outputs.len(), circuit.output_gate_ids().len());

        let mut output_context = DslContext::new("packed-nested-rns-production-tall-runtime");
        for (output_index, output) in outputs.into_iter().enumerate() {
            let BggTallPlaintext::Diagonal(plaintexts) = output.plaintext else {
                panic!("zero nested-RNS inputs keep revealed tall plaintexts")
            };
            for slot in 0..physical_slots {
                output_context = output_context
                    .output(
                        format!("output-{output_index}-slot-{slot}"),
                        plaintexts.get_static(slot),
                    )
                    .expect("unique production tall plaintext output");
            }
        }
        let graph = output_context
            .build()
            .expect("production tall graph")
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production_id, manifest)]),
            )
            .expect("production tall manifest validation");
        let result = execute(
            &graph,
            &mut backend,
            BTreeMap::from([(
                "packed-artifact-hash-key".to_owned(),
                RuntimeValue::Bytes(vec![0x93; 32]),
            )]),
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("production tall runtime");
        (0..circuit.output_gate_ids().len())
            .map(|output_index| {
                (0..physical_slots)
                    .map(|slot| {
                        matrix_output(&result, &format!("output-{output_index}-slot-{slot}"))
                            .clone()
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    #[serial_test::serial]
    fn packed_nested_rns_compiles_through_tall_arithmetic_lookup_and_slot_lowerings() {
        let parameters = DCRTPolyParams::new(2, 2, 12, 6);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let nested =
            Arc::new(NestedRnsPolyContext::setup(&mut circuit, &parameters, 6, 2, 16, false, None));
        let coefficient_slots = 2;
        let left =
            NestedRnsPoly::input(nested.clone(), coefficient_slots, Some(2), None, &mut circuit);
        let right =
            NestedRnsPoly::input(nested.clone(), coefficient_slots, Some(2), None, &mut circuit);
        let ordinary = left.mul(&right, &mut circuit).full_reduce(&mut circuit);
        let convolution =
            negacyclic_conv_mul(&parameters, &mut circuit, &left, &right, coefficient_slots)
                .full_reduce(&mut circuit);
        let ordinary = ordinary.reconstruct(&mut circuit);
        let convolution = convolution.reconstruct(&mut circuit);
        circuit.output([ordinary, convolution]);

        let physical_slots = coefficient_slots * nested.q_moduli_depth;
        let rotations = required_tall_rotation_encodings(&circuit).expect("rotation discovery");
        assert!(rotations.contains(&TallRotationEncodingKey {
            num_slots: physical_slots as u32,
            offset: nested.q_moduli_depth as u32,
        }));
        let outputs =
            assert_production_tall_artifacts_cover_circuit(&circuit, &parameters, physical_slots);

        let zero_polyvec =
            PolyVec((0..physical_slots).map(|_| DCRTPoly::const_zero(&parameters)).collect());
        let plaintext_outputs = execute_polyvec_circuit(
            "packed-nested-rns-plaintext-oracle",
            &parameters,
            &circuit,
            (0..circuit.num_input()).map(|_| zero_polyvec.clone()).collect(),
            physical_slots,
        );
        assert_eq!(plaintext_outputs.len(), outputs.len());
        for (output_index, plaintext_output) in plaintext_outputs.iter().enumerate() {
            for slot in 0..physical_slots {
                assert_eq!(
                    outputs[output_index][slot],
                    DCRTPolyMatrix::from_poly_vec(
                        &parameters,
                        vec![vec![plaintext_output.0[slot].clone()]],
                    ),
                    "tall decoded lane must match the PolyVec plaintext execution"
                );
            }
        }
    }
}
