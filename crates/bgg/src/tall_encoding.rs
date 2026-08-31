//! Tall BGG+ encodings with shared public matrices and row-wise evaluation.
//!
//! A wire with diagonal message `X` represents
//! `C_X = S A_X - X S G + E_X`, with one BGG+ row per slot and one public
//! matrix shared by every row. Rotation preprocessing exports only public
//! matrices; the online graph derives the matching permutation encodings from
//! its one shared secret-row family. For a provisioned permutation `P`, rotation evaluates
//! `C_P G^-1(A_X) + P C_X` followed by multiplication with the inverse
//! permutation encoding.

use crate::{
    BggPublicKeyCompiler, BggPublicKeyWire, BggSamplerLayout,
    tall_rotation_encoding::{TallLinearTransformEncodingWires, TallRotationEncodingKey},
};
use mxx_dsl::{DslError, Family, Int, Mat, Parallel};
use mxx_ir_core::{
    IntExpr, RealExpr,
    node::{ConcatAxis, IndexRange},
    types::MatrixType,
};
use num_bigint::BigUint;
use num_traits::{One, Zero};
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
    /// Compile-time-only exclusive upper bound for the revealed canonical
    /// plaintext. This is meaningful only when `plaintext` is diagonal.
    pub canonical_input_exclusive_upper: Option<BigUint>,
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

/// Sampling result containing all requested tall encodings.
#[derive(Clone)]
pub struct BggTallEncodingSample {
    /// Sampled encodings, in the same order as the supplied public keys.
    pub encodings: Vec<BggTallEncodingWire>,
}

/// Sampler for tall BGG+ encodings.
#[derive(Clone)]
pub struct BggTallEncodingSampler {
    /// Shared BGG matrix layout.
    pub layout: BggSamplerLayout,
    /// Optional Gaussian error width; `None` produces exact test encodings.
    pub gaussian_sigma: Option<RealExpr>,
    /// Explicit hard coefficient cutoff for `gaussian_sigma`.
    pub gaussian_max_coefficient_bound: Option<IntExpr>,
}

/// Errors produced by tall BGG+ compilation and artifact wiring.
#[derive(Debug, Error)]
pub enum TallCompileError {
    /// Input families or matrix types do not match the tall BGG layout.
    #[error("tall BGG+ inputs have incompatible counts or matrix types")]
    InvalidLayout,
    /// A Gaussian sampler needs a bound as well as a width.
    #[error("tall BGG+ Gaussian sampling requires both a sigma and an explicit coefficient cutoff")]
    MissingGaussianBound,
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
    /// Anchor reduction does not match the physical slot layout.
    #[error("tall anchor reduction has an invalid block or lane layout")]
    InvalidAnchorReduceLayout,
    /// One graph requested incompatible anchor-reduction matrices.
    #[error("a Tall graph requires more than one anchor-reduction matrix")]
    ConflictingAnchorReduceEncoding,
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
        // Row-wise addition preserves C_i = s_i A - x_i s_i G + e_i:
        // carriers and diagonal messages are combined at the same slot.
        let mut output = self.binary(
            lhs,
            rhs,
            |left, right| left + right,
            |compiler, left, right| compiler.add(left, right),
        )?;
        output.canonical_input_exclusive_upper = canonical_sum_upper(
            lhs.canonical_input_exclusive_upper.as_ref(),
            rhs.canonical_input_exclusive_upper.as_ref(),
            &output.rows.element_type().modulus,
        );
        Ok(output)
    }

    /// Subtracts two tall encodings row by row.
    pub fn sub(
        &self,
        lhs: &BggTallEncodingWire,
        rhs: &BggTallEncodingWire,
    ) -> Result<BggTallEncodingWire, TallCompileError> {
        // Row-wise subtraction keeps the slot alignment and computes both
        // C_i^L - C_i^R and x_i^L - x_i^R component by component.
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
            .decompose(self.public_key.base.clone(), self.public_key.digit_count.clone());
        // With G K_R = A_R, each slot computes C_i^L K_R + x_i^L C_i^R.
        // The decomposition is consumed on the right, and the correction term
        // cancels the cross term so the result carries x_i^L x_i^R.
        let rows = lhs.rows.clone().parallel_zip3(
            rhs.rows.clone(),
            lhs_plaintexts.clone(),
            move |_, left, right, plaintext| {
                left.mul_decomposed(decomposed_rhs.clone()) + plaintext * right
            },
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
            canonical_input_exclusive_upper: None,
        })
    }

    /// Multiplies every row and diagonal plaintext by one small scalar.
    pub fn small_scalar_mul(
        &self,
        input: &BggTallEncodingWire,
        scalar: &Mat,
    ) -> Result<BggTallEncodingWire, TallCompileError> {
        self.scalar_mul(input, scalar, None, false)
    }

    /// Multiplies every row by the gadget decomposition of a large scalar.
    pub fn large_scalar_mul(
        &self,
        input: &BggTallEncodingWire,
        scalar: &Mat,
    ) -> Result<BggTallEncodingWire, TallCompileError> {
        let decomposed = self.public_key.large_scalar_decomposition(&input.pubkey, scalar);
        self.scalar_mul(input, scalar, Some(decomposed), true)
    }

    /// Applies one provisioned cyclic rotation pair.
    pub fn rotate(
        &self,
        input: &BggTallEncodingWire,
        key: TallRotationEncodingKey,
        transform: &TallLinearTransformEncodingWires,
    ) -> Result<BggTallEncodingWire, TallCompileError> {
        let BggTallPlaintext::Diagonal(plaintexts) = &input.plaintext else {
            return Err(TallCompileError::MissingRotationPlaintext);
        };
        let num_slots =
            usize::try_from(key.num_slots).map_err(|_| TallCompileError::InvalidRotationLayout)?;
        if input.rows.count() != &IntExpr::constant(num_slots) ||
            plaintexts.count() != &IntExpr::constant(num_slots) ||
            transform.left_rows.count() != &IntExpr::constant(num_slots) ||
            transform.right_rows.count() != &IntExpr::constant(num_slots)
        {
            return Err(TallCompileError::InvalidRotationLayout);
        }
        let offset =
            usize::try_from(key.offset).map_err(|_| TallCompileError::InvalidRotationLayout)?;
        let rotated_rows = rotate_family(&input.rows, offset, num_slots)?;
        let rotated_plaintexts = rotate_family(plaintexts, offset, num_slots)?;
        let rotated_right_rows = rotate_family(&transform.right_rows, offset, num_slots)?;
        let decomposed_input = input
            .pubkey
            .matrix
            .clone()
            .decompose(self.public_key.base.clone(), self.public_key.digit_count.clone());
        // The first rotation step is L C_{i+offset} G^-1(A); the rotated
        // source row remains an explicit additive carrier contribution.
        let step1 =
            transform.left_rows.clone().parallel_zip(rotated_rows, move |_, left, input| {
                left.mul_decomposed(decomposed_input.clone()) + input
            })?;
        let decomposed_right = transform
            .right_matrix
            .clone()
            .decompose(self.public_key.base.clone(), self.public_key.digit_count.clone());
        // The second step applies G^-1(R) and adds the rotated plaintext times
        // the right helper row, completing the two-sided transform.
        let rows = step1.parallel_zip3(
            rotated_plaintexts.clone(),
            rotated_right_rows,
            move |_, intermediate, plaintext, right| {
                intermediate.mul_decomposed(decomposed_right.clone()) + plaintext * right
            },
        )?;
        Ok(BggTallEncodingWire {
            rows,
            pubkey: self.linear_transform_public_key(
                &input.pubkey,
                &transform.left_matrix,
                &transform.right_matrix,
            ),
            plaintext: BggTallPlaintext::Diagonal(rotated_plaintexts),
            canonical_input_exclusive_upper: None,
        })
    }

    /// Reduces every repeated CRT-lane block to one anchor row using one fixed
    /// sparse-matrix encoding twice.
    pub fn anchor_reduce(
        &self,
        input: &BggTallEncodingWire,
        num_blocks: u32,
        lane_scalars: &[BigUint],
        transform: &TallLinearTransformEncodingWires,
    ) -> Result<BggTallEncodingWire, TallCompileError> {
        let BggTallPlaintext::Diagonal(plaintexts) = &input.plaintext else {
            return Err(TallCompileError::MissingRotationPlaintext);
        };
        let blocks =
            usize::try_from(num_blocks).map_err(|_| TallCompileError::InvalidAnchorReduceLayout)?;
        let lanes = lane_scalars.len();
        let slots = blocks.checked_mul(lanes).ok_or(TallCompileError::InvalidAnchorReduceLayout)?;
        if lanes == 0 ||
            input.rows.count() != &IntExpr::constant(slots) ||
            plaintexts.count() != &IntExpr::constant(slots) ||
            transform.left_rows.count() != &IntExpr::constant(slots) ||
            transform.right_rows.count() != &IntExpr::constant(slots)
        {
            return Err(TallCompileError::InvalidAnchorReduceLayout);
        }
        let input_lanes = (0..lanes)
            .map(|lane| gather_repeated_lane(&input.rows, blocks, lanes, lane))
            .collect::<Result<Vec<_>, _>>()?;
        let plaintext_lanes = (0..lanes)
            .map(|lane| gather_repeated_lane(plaintexts, blocks, lanes, lane))
            .collect::<Result<Vec<_>, _>>()?;
        let left_helper_lanes = (0..lanes)
            .map(|lane| gather_repeated_lane(&transform.left_rows, blocks, lanes, lane))
            .collect::<Result<Vec<_>, _>>()?;
        let right_helper_lanes = (0..lanes)
            .map(|lane| gather_repeated_lane(&transform.right_rows, blocks, lanes, lane))
            .collect::<Result<Vec<_>, _>>()?;
        if lanes == 1 {
            let scalar = self
                .public_key
                .ring
                .polynomial([IntExpr::constant(num_bigint::BigInt::from(lane_scalars[0].clone()))]);
            return self.large_scalar_mul(
                &BggTallEncodingWire {
                    rows: input_lanes[0].clone(),
                    pubkey: input.pubkey.clone(),
                    plaintext: BggTallPlaintext::Diagonal(plaintext_lanes[0].clone()),
                    canonical_input_exclusive_upper: None,
                },
                &scalar,
            );
        }
        // The helper row is the sum of all non-anchor source lanes; the anchor
        // lane is handled separately as scalar_0 * C_0 below.
        let mut left_times_input_rows = input_lanes[1].clone();
        for lane_rows in input_lanes.iter().skip(2) {
            left_times_input_rows = left_times_input_rows
                .parallel_zip(lane_rows.clone(), |_, left, right| left + right)?;
        }
        // Each lane contributes x_a tensor R_a to the message-weighted helper.
        let mut left_message_times_right_rows = right_helper_lanes[1]
            .clone()
            .parallel_zip(plaintext_lanes[1].clone(), |_, row, plaintext| plaintext.tensor(row))?;
        for lane in 2..lanes {
            left_message_times_right_rows = left_message_times_right_rows.parallel_zip3(
                right_helper_lanes[lane].clone(),
                plaintext_lanes[lane].clone(),
                |_, sum, row, plaintext| sum + plaintext.tensor(row),
            )?;
        }
        let ring = self.public_key.ring.clone();
        let mut helper_plaintexts = plaintext_lanes[1].clone().parallel_map({
            let scalar = ring
                .polynomial([IntExpr::constant(num_bigint::BigInt::from(lane_scalars[1].clone()))]);
            move |_, value| value * scalar.clone()
        })?;
        for lane in 2..lanes {
            let scalar = ring.polynomial([IntExpr::constant(num_bigint::BigInt::from(
                lane_scalars[lane].clone(),
            ))]);
            helper_plaintexts = helper_plaintexts
                .parallel_zip(plaintext_lanes[lane].clone(), move |_, sum, value| {
                    sum + value * scalar.clone()
                })?;
        }
        let helper_output = self.linear_transform(
            input,
            transform,
            left_helper_lanes[0].clone(),
            left_times_input_rows,
            left_message_times_right_rows,
            helper_plaintexts,
        )?;
        let anchor_input = BggTallEncodingWire {
            rows: input_lanes[0].clone(),
            pubkey: input.pubkey.clone(),
            plaintext: BggTallPlaintext::Diagonal(plaintext_lanes[0].clone()),
            canonical_input_exclusive_upper: None,
        };
        let anchor_scalar =
            ring.polynomial([IntExpr::constant(num_bigint::BigInt::from(lane_scalars[0].clone()))]);
        // The anchor contribution uses the large-scalar target scalar_0 * G,
        // preserving the gadget carrier during decomposition.
        let anchor_term = self.large_scalar_mul(&anchor_input, &anchor_scalar)?;
        self.add(&anchor_term, &helper_output)
    }

    fn linear_transform(
        &self,
        input: &BggTallEncodingWire,
        transform: &TallLinearTransformEncodingWires,
        left_rows: Family<Mat>,
        left_times_input_rows: Family<Mat>,
        left_message_times_right_rows: Family<Mat>,
        output_plaintexts: Family<Mat>,
    ) -> Result<BggTallEncodingWire, TallCompileError> {
        let decomposed_input = input
            .pubkey
            .matrix
            .clone()
            .decompose(self.public_key.base.clone(), self.public_key.digit_count.clone());
        // This computes L C G^-1(A) + M: the public relation is consumed on
        // the right while the supplied row term M remains in the output.
        let intermediate = left_rows
            .parallel_zip(left_times_input_rows, move |_, left, input| {
                left.mul_decomposed(decomposed_input.clone()) + input
            })?;
        let decomposed_right = transform
            .right_matrix
            .clone()
            .decompose(self.public_key.base.clone(), self.public_key.digit_count.clone());
        // Apply the right decomposition and add the precomputed plaintext
        // tensor/right-row term to finish the linear transform.
        let rows = intermediate.parallel_zip(
            left_message_times_right_rows,
            move |_, intermediate, right| {
                intermediate.mul_decomposed(decomposed_right.clone()) + right
            },
        )?;
        Ok(BggTallEncodingWire {
            rows,
            pubkey: self.linear_transform_public_key(
                &input.pubkey,
                &transform.left_matrix,
                &transform.right_matrix,
            ),
            plaintext: BggTallPlaintext::Diagonal(output_plaintexts),
            canonical_input_exclusive_upper: None,
        })
    }

    /// Applies the public-key part of a fixed Tall transform `L * X * R`.
    pub fn linear_transform_public_key(
        &self,
        input: &BggPublicKeyWire,
        left_matrix: &Mat,
        right_matrix: &Mat,
    ) -> BggPublicKeyWire {
        let first = left_matrix.clone() *
            input
                .matrix
                .clone()
                .decompose(self.public_key.base.clone(), self.public_key.digit_count.clone())
                .into_preimage_relation()
                .materialize_exact();
        BggPublicKeyWire {
            matrix: first *
                right_matrix
                    .clone()
                    .decompose(self.public_key.base.clone(), self.public_key.digit_count.clone())
                    .into_preimage_relation()
                    .materialize_exact(),
            reveal_plaintext: input.reveal_plaintext,
        }
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
            canonical_input_exclusive_upper: None,
        })
    }

    fn scalar_mul(
        &self,
        input: &BggTallEncodingWire,
        scalar: &Mat,
        row_factor: Option<mxx_dsl::Decomposition>,
        large: bool,
    ) -> Result<BggTallEncodingWire, TallCompileError> {
        // Small scalars use direct t C_i.  Large scalars use K_t satisfying
        // G K_t = tG; revealed diagonal metadata is updated as t x_i in both
        // cases.
        let rows = match row_factor {
            Some(row_factor) => input
                .rows
                .clone()
                .parallel_map(move |_, row| row.mul_decomposed(row_factor.clone()))?,
            None => {
                let scalar = scalar.clone();
                input.rows.clone().parallel_map(move |_, row| scalar.clone() * row)?
            }
        };
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
        Ok(BggTallEncodingWire { rows, pubkey, plaintext, canonical_input_exclusive_upper: None })
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
    /// Samples one revealed diagonal encoding under caller-supplied slot secrets.
    ///
    /// This is the primitive used for public per-row masks.  It deliberately
    /// receives the already-sampled online secret family so a mask cannot
    /// introduce a second Tall secret source or recover the old slot-transfer
    /// construction.
    pub fn sample_diagonal(
        &self,
        secret_rows: Family<Mat>,
        public_key: BggPublicKeyWire,
        plaintexts: Family<Mat>,
    ) -> Result<BggTallEncodingWire, TallCompileError> {
        if self.gaussian_sigma.is_some() != self.gaussian_max_coefficient_bound.is_some() {
            return Err(TallCompileError::MissingGaussianBound);
        }
        let ring = self.layout.ring();
        let secret_size = self.layout.secret_dimension;
        let columns = self.layout.public_key_columns();
        if !public_key.reveal_plaintext ||
            secret_rows.count() != plaintexts.count() ||
            !same_matrix_type(secret_rows.element_type(), &ring.matrix_type((1, secret_size))) ||
            !same_matrix_type(plaintexts.element_type(), &ring.matrix_type((1, 1))) ||
            !same_matrix_type(
                public_key.matrix.matrix_type(),
                &ring.matrix_type((secret_size, columns)),
            )
        {
            return Err(TallCompileError::InvalidLayout);
        }
        let gadget =
            ring.gadget(secret_size, self.layout.gadget_base.clone(), self.layout.digit_count);
        let sigma = self.gaussian_sigma.clone();
        let bound = self.gaussian_max_coefficient_bound.clone();
        let public_matrix = public_key.matrix.clone();
        let rows =
            secret_rows.clone().parallel_zip(plaintexts.clone(), move |_, secret, plaintext| {
                // Each row is C_x = sA - (x tensor s)G + e.  G remains the
                // rightmost carrier factor, including the known-zero case 0*G.
                secret.clone() * public_matrix.clone() - plaintext * (secret * gadget.clone()) +
                    match (&sigma, &bound) {
                        (Some(sigma), Some(bound)) => {
                            ring.gaussian((1, columns), sigma.clone(), bound.clone())
                        }
                        (None, None) => ring.zero((1, columns)),
                        _ => unreachable!("validated Gaussian sampler configuration"),
                    }
            })?;
        Ok(BggTallEncodingWire {
            rows,
            pubkey: public_key,
            plaintext: BggTallPlaintext::Diagonal(plaintexts),
            canonical_input_exclusive_upper: None,
        })
    }

    /// Packs tall BGG+ rows under caller-supplied per-slot secrets.
    ///
    /// The caller owns sampling the fresh `S_i` family.  In particular, this
    /// sampler never derives a slot secret from a long-lived secret or a slot
    /// transform, which keeps the online encoding protocol free of hidden
    /// preprocessing state.
    pub fn sample(
        &self,
        secret_rows: Family<Mat>,
        public_keys: &[BggPublicKeyWire],
        plaintexts: &[Family<Mat>],
        slot_count: IntExpr,
    ) -> Result<BggTallEncodingSample, TallCompileError> {
        if public_keys.len() != plaintexts.len() + 1 {
            return Err(TallCompileError::InvalidLayout);
        }
        if self.gaussian_sigma.is_some() != self.gaussian_max_coefficient_bound.is_some() {
            return Err(TallCompileError::MissingGaussianBound);
        }
        let ring = self.layout.ring();
        let secret_size = self.layout.secret_dimension;
        let columns = self.layout.public_key_columns();
        if secret_rows.count() != &slot_count ||
            !same_matrix_type(secret_rows.element_type(), &ring.matrix_type((1, secret_size))) ||
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
        let ones = secret_rows.clone().parallel_map({
            let ring = ring.clone();
            move |_, _| ring.identity(1)
        })?;
        let count = public_keys.len();
        let gadget =
            ring.gadget(secret_size, self.layout.gadget_base.clone(), self.layout.digit_count);
        let sigma = self.gaussian_sigma.clone();
        let bound = self.gaussian_max_coefficient_bound.clone();
        let public_matrices = public_keys.iter().map(|key| key.matrix.clone()).collect::<Vec<_>>();
        let mut input_families = Vec::with_capacity(count + 1);
        input_families.push(secret_rows);
        input_families.push(ones.clone());
        input_families.extend(plaintexts.iter().cloned());
        let row_families =
            Family::<Mat>::parallel_zip_many_values(input_families, move |_, values| {
                let secret_row = &values[0];
                let packed_error = match (&sigma, &bound) {
                    (Some(sigma), Some(bound)) => {
                        ring.gaussian((1, columns * count), sigma.clone(), bound.clone())
                    }
                    (None, None) => ring.zero((1, columns * count)),
                    _ => unreachable!("validated Gaussian sampler configuration"),
                };
                (0..count)
                    .map(|index| {
                        let plaintext = if index == 0 { &values[1] } else { &values[index + 1] };
                        let error = packed_error.clone().slice(
                            None,
                            Some(IndexRange {
                                start: (columns * index).into(),
                                end: (columns * (index + 1)).into(),
                            }),
                        );
                        // The packed sampler slices only independent error
                        // columns; the signal keeps its complete gadget term.
                        secret_row.clone() * public_matrices[index].clone() -
                            plaintext.clone().tensor(secret_row.clone()) * gadget.clone() +
                            error
                    })
                    .collect::<Vec<_>>()
            })?;
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
                canonical_input_exclusive_upper: None,
            })
            .collect();
        Ok(BggTallEncodingSample { encodings })
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

fn canonical_sum_upper(
    lhs: Option<&BigUint>,
    rhs: Option<&BigUint>,
    modulus: &IntExpr,
) -> Option<BigUint> {
    let (lhs, rhs) = (lhs?, rhs?);
    if lhs.is_zero() || rhs.is_zero() {
        return None;
    }
    let IntExpr::Const(modulus) = modulus else {
        return None;
    };
    let upper = lhs + rhs - BigUint::one();
    (upper <= modulus.to_biguint()?).then_some(upper)
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
    // Keep the permutation as one generated gather. Packing static gets would lower to an
    // explicit family and erase the source expression from checker-visible program bodies.
    let indices = Parallel::range(num_slots).map_values(|destination| {
        destination.as_int().add(Int::constant(num_slots - offset)).rem(Int::constant(num_slots))
    })?;
    Ok(rows.clone().parallel_gather(indices)?)
}

/// Applies the sparse matrix `U` used by anchor reduction to a row family.
/// `U[0,a]=1` and `U[a,0]=lane_scalars[a]` within every block for `a>0`.
pub(crate) fn anchor_matrix_rows(
    rows: &Family<Mat>,
    num_blocks: usize,
    lane_scalars: &[BigUint],
    ring: &mxx_dsl::Ring,
) -> Result<Family<Mat>, TallCompileError> {
    let lanes = lane_scalars.len();
    let slot_count =
        num_blocks.checked_mul(lanes).ok_or(TallCompileError::InvalidAnchorReduceLayout)?;
    if num_blocks == 0 || lanes == 0 || rows.count() != &IntExpr::constant(slot_count) {
        return Err(TallCompileError::InvalidAnchorReduceLayout);
    }
    let source = rows.clone();
    let work = rows.clone();
    let ring = ring.clone();
    let scalars = lane_scalars.to_vec();
    Ok(Family::<Mat>::parallel_zip_many_with_broadcast_values(
        vec![work],
        vec![source],
        move |index, _, families| {
            let lane_count = Int::constant(lanes);
            let flat = index.as_int();
            let block_base = flat.clone().div(lane_count.clone()).mul(lane_count.clone());
            let lane = flat.rem(lane_count);
            let source = &families[0];
            let mut anchor = ring.zero((1, source.element_type().columns.clone()));
            for source_lane in 1..lanes {
                anchor = anchor + source.get(block_base.clone().add(Int::constant(source_lane)));
            }
            let anchor_source = source.get(block_base);
            let mut branches = Vec::with_capacity(lanes);
            branches.push(anchor);
            branches.extend(scalars.iter().skip(1).map(|scalar| {
                anchor_source.clone() *
                    ring.polynomial([IntExpr::constant(num_bigint::BigInt::from(scalar.clone()))])
            }));
            Ok(lane.select(branches)?)
        },
    )?)
}

fn gather_repeated_lane(
    rows: &Family<Mat>,
    num_blocks: usize,
    lanes: usize,
    lane: usize,
) -> Result<Family<Mat>, TallCompileError> {
    if lane >= lanes || rows.count() != &IntExpr::constant(num_blocks.saturating_mul(lanes)) {
        return Err(TallCompileError::InvalidAnchorReduceLayout);
    }
    let indices = Parallel::range(num_blocks).map_values(move |block| {
        block.as_int().mul(Int::constant(lanes)).add(Int::constant(lane))
    })?;
    Ok(rows.clone().parallel_gather(indices)?)
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
        BggTallSlotLowering, BggTallSlotPublicKeyLowering, CircuitCompileError,
        TallLinearTransformEncodingWires, TallLinearTransformPublicWires,
        TallRotationEncodingArtifactNames, TallRotationEncodingArtifacts,
        TallRotationEncodingCompiler, TallRotationEncodingKey, required_tall_rotation_encodings,
        tall_rotation_encoding::tall_rotation_public_key_tag,
        test_utils::{execute_graph, matrix_output, row},
    };
    use mxx_dsl::{BuiltGraph, DslContext, Ring};
    use mxx_gadgets::circuit::{
        CircuitLoweringTypes, GateInstance, PolyCircuit, SlotOperationLowering, SlotTransferSpec,
        SubCircuitParamSpec, SubCircuitParamValue,
    };
    use mxx_ir_core::{ParamEnv, node::NodeKind};
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        sampler::{DistType, PolyHashSampler, hash::DCRTPolyHashSampler},
    };
    use mxx_runtime::{
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
        transcript::SamplingMode,
    };
    use num_bigint::BigInt;
    use std::collections::{BTreeMap, BTreeSet};

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
        key: TallRotationEncodingKey,
        transform: TallLinearTransformEncodingWires,
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

        fn slot_anchor_reduce(
            &mut self,
            _input: &Self::Wire,
            _num_blocks: u32,
            _lane_scalars: &[BigUint],
            _gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            Err(CircuitCompileError::Structure("unexpected anchor reduction".to_owned()))
        }

        fn slot_rotation(
            &mut self,
            input: &Self::Wire,
            offset: u32,
            num_slots: u32,
            _gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            assert_eq!((offset, num_slots), (1, self.key.num_slots));
            Ok(self.compiler.rotate(input, self.key, &self.transform)?)
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
            canonical_input_exclusive_upper: None,
        };
        let left = wire("left");
        let right = wire("right");
        let sum = compiler.add(&left, &right).unwrap();
        let product = compiler.simd_mul(&left, &right).unwrap();
        let mut bounded = left.clone();
        bounded.canonical_input_exclusive_upper = Some(BigUint::from(7u8));
        assert!(
            compiler
                .small_scalar_mul(&bounded, &ring.identity(1))
                .unwrap()
                .canonical_input_exclusive_upper
                .is_none()
        );
        let mut bounded_right = right.clone();
        bounded_right.canonical_input_exclusive_upper = Some(BigUint::from(4u8));
        assert_eq!(
            compiler.add(&bounded, &bounded_right).unwrap().canonical_input_exclusive_upper,
            Some(BigUint::from(10u8)),
            "[0, 7) + [0, 4) has exclusive upper bound 10"
        );
        let IntExpr::Const(modulus) = &left.rows.element_type().modulus else {
            panic!("test ring has a concrete modulus")
        };
        let mut wrapping_left = left.clone();
        wrapping_left.canonical_input_exclusive_upper = modulus.to_biguint();
        let mut wrapping_right = right.clone();
        wrapping_right.canonical_input_exclusive_upper = Some(BigUint::from(2u8));
        assert!(
            compiler
                .add(&wrapping_left, &wrapping_right)
                .unwrap()
                .canonical_input_exclusive_upper
                .is_none(),
            "a sum that reaches the modulus is not a canonical nonwrapping range"
        );
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
            canonical_input_exclusive_upper: None,
        };
        let key = TallRotationEncodingKey { num_slots: slots as u32, offset: 1 };
        let rotation = TallLinearTransformEncodingWires {
            left_matrix: ring.input("a-forward", (secret_size, columns)),
            right_matrix: ring.input("a-backward", (secret_size, columns)),
            left_rows: Family::pack(
                (0..slots)
                    .map(|slot| ring.input(format!("c-forward-{slot}"), (1, columns)))
                    .collect(),
            )
            .unwrap(),
            right_rows: Family::pack(
                (0..slots)
                    .map(|slot| ring.input(format!("c-backward-{slot}"), (1, columns)))
                    .collect(),
            )
            .unwrap(),
        };
        let output = compiler.rotate(&input, key, &rotation).unwrap();
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
        let built = context.build().unwrap();
        assert!(
            !built
                .graph
                .scopes()
                .values()
                .flat_map(|scope| scope.nodes())
                .any(|node| matches!(node.kind(), NodeKind::MatrixMulAccumulate { .. })),
            "baseline Tall rotation must remain explicit Multiply/Add dataflow",
        );
        let result = execute_graph(built, parameters, inputs);
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
    fn tall_sampler_uses_master_secret_rows_in_the_bgg_formula() {
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
        let master_secret_rows = Family::pack(
            (0..slots)
                .map(|slot| ring.input(format!("master-secret-row-{slot}"), (1, secret_size)))
                .collect(),
        )
        .unwrap();
        let sample = BggTallEncodingSampler {
            layout: layout.clone(),
            gaussian_sigma: None,
            gaussian_max_coefficient_bound: None,
        }
        .sample(master_secret_rows, &public_keys, &[plaintexts], slots.into())
        .unwrap();
        let mut context = DslContext::new("tall-sampler-runtime");
        for slot in 0..slots {
            context = context
                .output(format!("row-{slot}"), sample.encodings[1].rows.get_static(slot))
                .unwrap();
        }

        let public_one = public_matrix(&parameters, secret_size, columns, 4);
        let public_message = public_matrix(&parameters, secret_size, columns, 7);
        let plaintext_values =
            (0..slots).map(|slot| row(&parameters, 1, 10 + slot)).collect::<Vec<_>>();
        let secret_row_values =
            (0..slots).map(|slot| row(&parameters, secret_size, 14 + slot * 2)).collect::<Vec<_>>();
        let mut inputs = BTreeMap::from([
            ("public-one".to_owned(), RuntimeValue::matrix(public_one)),
            ("public-message".to_owned(), RuntimeValue::matrix(public_message.clone())),
        ]);
        for slot in 0..slots {
            inputs.insert(
                format!("plaintext-{slot}"),
                RuntimeValue::matrix(plaintext_values[slot].clone()),
            );
            inputs.insert(
                format!("master-secret-row-{slot}"),
                RuntimeValue::matrix(secret_row_values[slot].clone()),
            );
        }
        let result = execute_graph(context.build().unwrap(), parameters.clone(), inputs);
        let gadget = DCRTPolyMatrix::gadget_matrix(&parameters, secret_size);
        for slot in 0..slots {
            let secret_row = secret_row_values[slot].clone();
            let expected = secret_row.clone() * public_message.clone() -
                plaintext_values[slot].clone().tensor(&(secret_row * gadget.clone()));
            assert_eq!(matrix_output(&result, &format!("row-{slot}")), &expected);
        }
    }

    #[test]
    fn tall_sampler_is_blockwise_for_three_keys_and_uses_one_packed_error() {
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
        let public_keys = (0..3)
            .map(|index| BggPublicKeyWire {
                matrix: ring.input(format!("block-public-{index}"), (secret_size, columns)),
                reveal_plaintext: true,
            })
            .collect::<Vec<_>>();
        let plaintexts = (0..2)
            .map(|index| ring.input_family(format!("block-plaintext-{index}"), slots, (1, 1)))
            .collect::<Vec<_>>();
        let secret_rows = ring.input_family("block-secret", slots, (1, secret_size));
        let sample = BggTallEncodingSampler {
            layout: layout.clone(),
            gaussian_sigma: Some(3.into()),
            gaussian_max_coefficient_bound: Some(19.into()),
        }
        .sample(secret_rows, &public_keys, &plaintexts, slots.into())
        .expect("compatible blockwise sampler inputs");

        let mut context = DslContext::new("tall-blockwise-sampler");
        for (index, encoding) in sample.encodings.iter().enumerate() {
            context = context
                .family_output(format!("block-row-{index}"), encoding.rows.clone())
                .expect("family output");
        }
        let built = context.build().expect("build blockwise sampler graph");
        built.validate(&ParamEnv::default()).expect("valid blockwise sampler graph");
        let nodes =
            built.graph.scopes().values().flat_map(|scope| scope.nodes()).collect::<Vec<_>>();
        assert!(!nodes.iter().any(|node| matches!(node.kind(), NodeKind::Concat { .. })));

        let gaussian_nodes = nodes
            .iter()
            .filter(|node| matches!(node.kind(), NodeKind::GaussianSample { .. }))
            .collect::<Vec<_>>();
        assert_eq!(gaussian_nodes.len(), 1, "all blocks share one packed Gaussian");
        let NodeKind::GaussianSample { matrix_type, .. } = gaussian_nodes[0].kind() else {
            unreachable!("filtered Gaussian node")
        };
        assert_eq!(matrix_type.rows.canonicalize(), IntExpr::constant(1));
        assert_eq!(
            matrix_type.columns.canonicalize(),
            IntExpr::constant(columns * public_keys.len())
        );
        let packed_error = gaussian_nodes[0].output(0).expect("Gaussian output");
        let error_slices = nodes
            .iter()
            .filter_map(|node| {
                let NodeKind::Slice { columns: Some(range), rows: None } = node.kind() else {
                    return None;
                };
                (node.arguments().first().map(|argument| argument.node()) ==
                    Some(packed_error.node()))
                .then_some(range)
            })
            .collect::<Vec<_>>();
        assert_eq!(error_slices.len(), public_keys.len(), "each block slices the shared error");
        assert!(nodes.iter().any(|node| matches!(
            node.kind(),
            NodeKind::ConstantMatrix {
                value: mxx_ir_core::node::ConstantMatrix::Gadget { .. },
                ..
            }
        )));
        assert!(
            !nodes.iter().any(|node| {
                matches!(node.kind(), NodeKind::Tensor) &&
                    node.arguments().get(1).is_some_and(|right| {
                        matches!(
                            right.node().kind(),
                            NodeKind::MatrixBinary(mxx_ir_core::node::MatrixBinaryOp::Multiply)
                        )
                    })
            }),
            "the Tall tensor must receive the carrierless secret, before the real final gadget factor"
        );
    }

    #[test]
    fn tall_anchor_reduce_uses_one_helper_for_all_lanes() {
        let ring = Ring::new(257, 8);
        let slots = 4;
        let lane_scalars = vec![BigUint::from(3u8), BigUint::from(5u8)];
        let helper_compiler = TallRotationEncodingCompiler {
            modulus: 257.into(),
            ring_dimension: 8.into(),
            secret_size: 1,
            slot_count: slots,
            gadget_base: 4.into(),
            digit_count: 2,
            error_sigma: 0.into(),
            error_max_coefficient_bound: 0.into(),
        };
        let public = helper_compiler
            .preprocess_anchor_reduce(
                ring.bytes_input("anchor-hash-key", 32),
                2,
                lane_scalars.clone(),
            )
            .unwrap();
        let secret_rows = ring.input_family("anchor-secret", slots, (1, 1));
        let secret_gadget_rows = helper_compiler.secret_gadget_rows(secret_rows.clone()).unwrap();
        let helper = helper_compiler
            .encode_anchor_reduce(&public, 2, &lane_scalars, secret_rows, secret_gadget_rows)
            .unwrap();
        let public_compiler =
            BggPublicKeyCompiler { ring: ring.clone(), base: 4.into(), digit_count: 2.into() };
        let output = BggTallEncodingCompiler { public_key: public_compiler }
            .anchor_reduce(
                &BggTallEncodingWire {
                    rows: ring.input_family("anchor-input-rows", slots, (1, 2)),
                    pubkey: BggPublicKeyWire {
                        matrix: ring.input("anchor-input-public", (1, 2)),
                        reveal_plaintext: true,
                    },
                    plaintext: BggTallPlaintext::Diagonal(ring.input_family(
                        "anchor-input-plaintexts",
                        slots,
                        (1, 1),
                    )),
                    canonical_input_exclusive_upper: None,
                },
                2,
                &lane_scalars,
                &helper,
            )
            .unwrap();
        assert_eq!(output.rows.count(), &IntExpr::constant(2));
        let BggTallPlaintext::Diagonal(plaintexts) = output.plaintext else {
            panic!("anchor reduction keeps the revealed anchor plaintexts")
        };
        DslContext::new("tall-anchor-reduce")
            .family_output("rows", output.rows)
            .unwrap()
            .family_output("plaintexts", plaintexts)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
    }

    #[test]
    fn tall_sampler_blockwise_runtime_matches_identity_and_plaintext_formulas() {
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
        let public_keys = (0..3)
            .map(|index| BggPublicKeyWire {
                matrix: ring.input(format!("formula-public-{index}"), (secret_size, columns)),
                reveal_plaintext: true,
            })
            .collect::<Vec<_>>();
        let plaintexts = (0..2)
            .map(|index| {
                Family::pack(
                    (0..slots)
                        .map(|slot| ring.input(format!("formula-plaintext-{index}-{slot}"), (1, 1)))
                        .collect(),
                )
                .expect("plaintext family")
            })
            .collect::<Vec<_>>();
        let secret_rows = Family::pack(
            (0..slots)
                .map(|slot| ring.input(format!("formula-secret-{slot}"), (1, secret_size)))
                .collect(),
        )
        .expect("secret family");
        let sample = BggTallEncodingSampler {
            layout: layout.clone(),
            gaussian_sigma: None,
            gaussian_max_coefficient_bound: None,
        }
        .sample(secret_rows, &public_keys, &plaintexts, slots.into())
        .expect("compatible formula sampler inputs");

        let mut context = DslContext::new("tall-blockwise-formula");
        for (block, encoding) in sample.encodings.iter().enumerate() {
            for slot in 0..slots {
                context = context
                    .output(format!("formula-row-{block}-{slot}"), encoding.rows.get_static(slot))
                    .expect("row output");
            }
        }
        let built = context.build().expect("build formula graph");
        built.validate(&ParamEnv::default()).expect("valid formula graph");

        let public_values = (0..3)
            .map(|block| public_matrix(&parameters, secret_size, columns, 3 + block * 7))
            .collect::<Vec<_>>();
        let plaintext_values = (0..2)
            .map(|block| {
                (0..slots)
                    .map(|slot| row(&parameters, 1, 30 + block * 7 + slot))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let secret_values =
            (0..slots).map(|slot| row(&parameters, secret_size, 50 + slot * 2)).collect::<Vec<_>>();
        let mut inputs = BTreeMap::new();
        for (block, value) in public_values.iter().enumerate() {
            inputs.insert(format!("formula-public-{block}"), RuntimeValue::matrix(value.clone()));
        }
        for (block, values) in plaintext_values.iter().enumerate() {
            for (slot, value) in values.iter().enumerate() {
                inputs.insert(
                    format!("formula-plaintext-{block}-{slot}"),
                    RuntimeValue::matrix(value.clone()),
                );
            }
        }
        for (slot, value) in secret_values.iter().enumerate() {
            inputs.insert(format!("formula-secret-{slot}"), RuntimeValue::matrix(value.clone()));
        }
        let result = execute_graph(built, parameters.clone(), inputs);
        let gadget = DCRTPolyMatrix::gadget_matrix(&parameters, secret_size);
        let identity = DCRTPolyMatrix::identity(&parameters, 1, None);
        for block in 0..3 {
            for slot in 0..slots {
                let plaintext = if block == 0 {
                    identity.clone()
                } else {
                    plaintext_values[block - 1][slot].clone()
                };
                let expected = secret_values[slot].clone() * public_values[block].clone() -
                    plaintext.tensor(&(secret_values[slot].clone() * gadget.clone()));
                assert_eq!(
                    matrix_output(&result, &format!("formula-row-{block}-{slot}")),
                    &expected,
                    "block {block}, slot {slot} follows the Tall BGG formula",
                );
            }
        }
    }

    #[test]
    fn tall_sampler_supports_one_key_without_plaintext_blocks() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let secret_size = 2;
        let slots = 1;
        let layout = BggSamplerLayout {
            modulus: BigInt::from(parameters.modulus().as_ref().clone()).into(),
            ring_dimension: (parameters.ring_dimension() as usize).into(),
            secret_dimension: secret_size,
            digit_count: parameters.modulus_digits(),
            gadget_base: BigInt::from(1u64 << parameters.base_bits()).into(),
        };
        let ring = layout.ring();
        let columns = layout.public_key_columns();
        let public_key = BggPublicKeyWire {
            matrix: ring.input("single-public", (secret_size, columns)),
            reveal_plaintext: true,
        };
        let sample = BggTallEncodingSampler {
            layout,
            gaussian_sigma: None,
            gaussian_max_coefficient_bound: None,
        }
        .sample(
            ring.input_family("single-secret", slots, (1, secret_size)),
            &[public_key],
            &[],
            slots.into(),
        )
        .expect("one-key sampler input");
        assert_eq!(sample.encodings.len(), 1);
        assert_eq!(sample.encodings[0].rows.count(), &IntExpr::constant(slots));
        let built = DslContext::new("tall-single-block-sampler")
            .family_output("row", sample.encodings[0].rows.clone())
            .expect("family output")
            .build()
            .expect("build single-block sampler graph");
        built.validate(&ParamEnv::default()).expect("valid single-block sampler graph");
        assert!(
            !built
                .graph
                .scopes()
                .values()
                .flat_map(|scope| scope.nodes())
                .any(|node| matches!(node.kind(), NodeKind::Concat { .. }))
        );
        assert_eq!(
            sample.encodings[0].rows.element_type().columns.canonicalize(),
            IntExpr::constant(columns)
        );
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
            error_max_coefficient_bound: 0.into(),
        };
        let ring = compiler.ring();
        let preprocessing = compiler.preprocess(ring.bytes_input("hash-key", 32), &[1, 3]).unwrap();
        let producer = compiler
            .export_preprocessing(DslContext::new("rotation-producer"), preprocessing)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();

        let hash_key = [0x42; 32];
        let producer_inputs =
            BTreeMap::from([("hash-key".to_owned(), RuntimeValue::Bytes(hash_key.to_vec()))]);
        let mut store = MemoryArtifactStore::default();
        let mut backend = cpu_backend([parameters.clone()]);
        let produced =
            execute(&producer, &mut backend, producer_inputs, &mut store, SamplingMode::Fresh)
                .unwrap();
        let production_id = produced.production_id.expect("artifact production");
        let manifest = store.manifest(&production_id).unwrap().clone();
        assert_eq!(manifest.artifacts.len(), 4);

        let artifacts = TallRotationEncodingArtifacts {
            production_id: production_id.clone(),
            slot_count: slots as u32,
        };
        let mut context = DslContext::new("rotation-consumer");
        let secret_rows = Family::pack(
            (0..slots).map(|slot| ring.input(format!("secret-{slot}"), (1, secret_size))).collect(),
        )
        .unwrap();
        for offset in [1, 3] {
            let (key, public) = compiler.import_artifacts(&artifacts, offset).unwrap().unwrap();
            let rotation = compiler.encode_rotation(key, &public, secret_rows.clone()).unwrap();
            context = context
                .output(format!("a-forward-{offset}"), rotation.left_matrix)
                .unwrap()
                .output(format!("a-backward-{offset}"), rotation.right_matrix)
                .unwrap();
            for slot in 0..slots {
                context = context
                    .output(
                        format!("c-forward-{offset}-{slot}"),
                        rotation.left_rows.get_static(slot),
                    )
                    .unwrap()
                    .output(
                        format!("c-backward-{offset}-{slot}"),
                        rotation.right_rows.get_static(slot),
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
        let secret_rows_values =
            (0..slots).map(|slot| row(&parameters, secret_size, 2 + slot * 2)).collect::<Vec<_>>();
        let consumed = execute(
            &consumer,
            &mut backend,
            secret_rows_values
                .iter()
                .enumerate()
                .map(|(slot, value)| {
                    (format!("secret-{slot}"), RuntimeValue::matrix(value.clone()))
                })
                .collect(),
            &mut store,
            SamplingMode::Fresh,
        )
        .unwrap();

        let hash = DCRTPolyHashSampler::<keccak_asm::Keccak256>::new();
        let gadget = DCRTPolyMatrix::gadget_matrix(&parameters, secret_size);
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
                    &(secret_rows_values[slot].clone() * expected_forward.clone() -
                        secret_rows_values[forward_source].clone() * gadget.clone())
                );
                assert_eq!(
                    matrix_output(&consumed, &format!("c-backward-{offset}-{slot}")),
                    &(secret_rows_values[slot].clone() * expected_backward.clone() -
                        secret_rows_values[backward_source].clone() * gadget.clone())
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
        let rotation = TallLinearTransformEncodingWires {
            left_matrix: ring.hash_matrix(
                hash_key.clone(),
                tall_rotation_public_key_tag(key, false),
                (secret_size, columns),
            ),
            right_matrix: ring.hash_matrix(
                hash_key.clone(),
                tall_rotation_public_key_tag(key, true),
                (secret_size, columns),
            ),
            left_rows: Family::pack((0..slots).map(|_| ring.zero((1, columns))).collect()).unwrap(),
            right_rows: Family::pack((0..slots).map(|_| ring.zero((1, columns))).collect())
                .unwrap(),
        };
        let tall_input = BggTallEncodingWire {
            rows: Family::pack((0..slots).map(|_| ring.zero((1, columns))).collect()).unwrap(),
            pubkey: input_public.clone(),
            plaintext: BggTallPlaintext::Diagonal(
                Family::pack((0..slots).map(|_| ring.zero((1, 1))).collect()).unwrap(),
            ),
            canonical_input_exclusive_upper: None,
        };
        let mut public_lowering = BggTallSlotPublicKeyLowering {
            compiler: public_compiler.clone(),
            diagonal_mask_public_key: input_public.clone(),
            configured_slot_count: slots,
            rotations: BTreeMap::from([(
                key,
                TallLinearTransformPublicWires {
                    left_matrix: rotation.left_matrix.clone(),
                    right_matrix: rotation.right_matrix.clone(),
                },
            )]),
            anchor_reduce: None,
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
        let mut tall_slots = TestTallRotationLowering {
            compiler: BggTallEncodingCompiler { public_key: public_compiler },
            key,
            transform: rotation,
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
    fn tall_rotation_reindex_uses_generated_gather_without_family_pack() {
        let ring = Ring::new(257, 8);
        let rows = ring.input_family("rotation-rows", 4, (1, 2));
        let rotated = rotate_family(&rows, 1, 4).expect("generated rotation family");
        let built = DslContext::new("tall-rotation-generated-reindex")
            .family_output("rotated", rotated)
            .unwrap()
            .build()
            .unwrap();
        built.validate(&ParamEnv::default()).expect("valid generated rotation graph");

        let nodes =
            built.graph.scopes().values().flat_map(|scope| scope.nodes()).collect::<Vec<_>>();
        assert!(nodes.iter().any(|node| matches!(node.kind(), NodeKind::ParallelGrid(_))));
        assert!(nodes.iter().any(|node| matches!(node.kind(), NodeKind::FamilyGetDynamic { .. })));
        assert!(!nodes.iter().any(|node| matches!(node.kind(), NodeKind::FamilyPack { .. })));
    }

    #[test]
    fn identity_slot_transfer_uses_the_online_diagonal_mask() {
        let ring = Ring::new(257, 8);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input_gate = circuit.input(1).as_single_wire();
        let transferred = circuit.slot_transfer_gate(input_gate, &[(0, None), (1, Some(2))]);
        circuit.output([transferred]);
        let public_compiler =
            BggPublicKeyCompiler { ring: ring.clone(), base: 4.into(), digit_count: 2.into() };
        let one_public =
            BggPublicKeyWire { matrix: ring.input("one-public", (1, 2)), reveal_plaintext: true };
        let input_public =
            BggPublicKeyWire { matrix: ring.input("input-public", (1, 2)), reveal_plaintext: true };
        let diagonal_mask_public_key =
            BggPublicKeyWire { matrix: ring.input("mask-public", (1, 2)), reveal_plaintext: true };
        let mut public_lowering = BggTallSlotPublicKeyLowering {
            compiler: public_compiler.clone(),
            diagonal_mask_public_key: diagonal_mask_public_key.clone(),
            configured_slot_count: 2,
            rotations: BTreeMap::new(),
            anchor_reduce: None,
        };
        let circuit_compiler = crate::PolyCircuitCompiler { public_key: public_compiler.clone() };
        let mut no_lookup = crate::NoPublicLookup::default();
        let public_output = circuit_compiler
            .compile_public_keys_with_lowerings(
                &circuit,
                one_public.clone(),
                [input_public.clone()],
                &mut no_lookup,
                &mut public_lowering,
            )
            .unwrap()
            .remove(0);
        let input = BggTallEncodingWire {
            rows: ring.input_family("rows", 2, (1, 2)),
            pubkey: input_public,
            plaintext: BggTallPlaintext::Diagonal(ring.input_family("plaintexts", 2, (1, 1))),
            canonical_input_exclusive_upper: Some(BigUint::from(7u8)),
        };
        let one = BggTallEncodingWire {
            rows: ring.input_family("one-rows", 2, (1, 2)),
            pubkey: one_public,
            plaintext: BggTallPlaintext::Diagonal(ring.input_family("one-plaintexts", 2, (1, 1))),
            canonical_input_exclusive_upper: None,
        };
        let mut lowering = BggTallSlotLowering::new(
            BggTallEncodingCompiler { public_key: public_compiler.clone() },
            diagonal_mask_public_key,
            ring.input_family("secret-rows", 2, (1, 1)),
            BggTallEncodingSampler {
                layout: BggSamplerLayout {
                    modulus: 257.into(),
                    ring_dimension: 8.into(),
                    secret_dimension: 1,
                    digit_count: 2,
                    gadget_base: 4.into(),
                },
                gaussian_sigma: None,
                gaussian_max_coefficient_bound: None,
            },
            BTreeMap::new(),
            None,
        );
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
            .output("public", output.pubkey.matrix.clone())
            .unwrap()
            .output("public-key-pass", public_output.matrix.clone())
            .unwrap()
            .build()
            .unwrap();
        built.validate(&ParamEnv::default()).expect("valid executable graph");
        assert_eq!(
            output.pubkey.matrix.matrix_type(),
            public_output.matrix.matrix_type(),
            "online mask multiplication and public-key lowering must agree on the output layout",
        );
    }

    #[test]
    fn compact_identity_lane_mask_graph_scales_with_lanes_not_slots() {
        fn build(slot_count: usize, lanes: usize) -> BuiltGraph {
            assert_eq!(slot_count % lanes, 0);
            let ring = Ring::new(257, 8);
            let mut circuit = PolyCircuit::<DCRTPoly>::new();
            let input_gate = circuit.input(1).as_single_wire();
            let transferred = circuit.slot_identity_repeated_lanes_gate(
                input_gate,
                slot_count / lanes,
                (0..lanes).map(|lane| Some((lane % 3) as u32)).collect(),
            );
            circuit.output([transferred]);
            let public_compiler =
                BggPublicKeyCompiler { ring: ring.clone(), base: 4.into(), digit_count: 2.into() };
            let one_public = BggPublicKeyWire {
                matrix: ring.input("compact-one-public", (1, 2)),
                reveal_plaintext: true,
            };
            let input_public = BggPublicKeyWire {
                matrix: ring.input("compact-input-public", (1, 2)),
                reveal_plaintext: true,
            };
            let diagonal_mask_public_key = BggPublicKeyWire {
                matrix: ring.input("compact-mask-public", (1, 2)),
                reveal_plaintext: true,
            };
            let circuit_compiler =
                crate::PolyCircuitCompiler { public_key: public_compiler.clone() };
            let mut public_lowering = BggTallSlotPublicKeyLowering {
                compiler: public_compiler.clone(),
                diagonal_mask_public_key: diagonal_mask_public_key.clone(),
                configured_slot_count: slot_count,
                rotations: BTreeMap::new(),
                anchor_reduce: None,
            };
            let public_output = circuit_compiler
                .compile_public_keys_with_lowerings(
                    &circuit,
                    one_public.clone(),
                    [input_public.clone()],
                    &mut crate::NoPublicLookup::default(),
                    &mut public_lowering,
                )
                .expect("compact Tall public-key transfer")
                .remove(0);
            let input = BggTallEncodingWire {
                rows: ring.input_family("compact-rows", slot_count, (1, 2)),
                pubkey: input_public,
                plaintext: BggTallPlaintext::Diagonal(ring.input_family(
                    "compact-plaintexts",
                    slot_count,
                    (1, 1),
                )),
                canonical_input_exclusive_upper: None,
            };
            let one = BggTallEncodingWire {
                rows: ring.input_family("compact-one-rows", slot_count, (1, 2)),
                pubkey: one_public,
                plaintext: BggTallPlaintext::Diagonal(ring.input_family(
                    "compact-one-plaintexts",
                    slot_count,
                    (1, 1),
                )),
                canonical_input_exclusive_upper: None,
            };
            let mut lowering = BggTallSlotLowering::new(
                BggTallEncodingCompiler { public_key: public_compiler.clone() },
                diagonal_mask_public_key,
                ring.input_family("compact-secret-rows", slot_count, (1, 1)),
                BggTallEncodingSampler {
                    layout: BggSamplerLayout {
                        modulus: 257.into(),
                        ring_dimension: 8.into(),
                        secret_dimension: 1,
                        digit_count: 2,
                        gadget_base: 4.into(),
                    },
                    gaussian_sigma: None,
                    gaussian_max_coefficient_bound: None,
                },
                BTreeMap::new(),
                None,
            );
            let output = circuit_compiler
                .compile_tall_encodings_with_lowerings(
                    &circuit,
                    one,
                    [input],
                    &mut crate::NoPublicLookup::default(),
                    &mut lowering,
                )
                .expect("compact Tall transfer")
                .remove(0);
            DslContext::new("compact-tall-slot-transfer")
                .family_output("rows", output.rows)
                .unwrap()
                .output("public", public_output.matrix)
                .unwrap()
                .build()
                .unwrap()
        }

        let small = build(8, 4);
        let large = build(1 << 16, 4);
        for graph in [&small, &large] {
            graph.validate(&ParamEnv::default()).expect("valid compact Tall graph");
            let nodes =
                graph.graph.scopes().values().flat_map(|scope| scope.nodes()).collect::<Vec<_>>();
            assert!(!nodes.iter().any(|node| matches!(node.kind(), NodeKind::FamilyPack { .. })));
            assert_eq!(
                nodes
                    .iter()
                    .filter(|node| matches!(node.kind(), NodeKind::ParallelGrid(_)))
                    .count(),
                3,
                "mask generation, Tall sampling, and SIMD multiplication each retain one grid"
            );
        }
        let graph_size = |graph: &BuiltGraph| {
            (
                graph.graph.scopes().len(),
                graph.graph.scopes().values().map(|scope| scope.nodes().len()).sum::<usize>(),
            )
        };
        assert_eq!(graph_size(&small), graph_size(&large));
    }

    #[test]
    fn nonidentity_tall_slot_transfer_fails_closed() {
        let ring = Ring::new(257, 8);
        let compiler =
            BggPublicKeyCompiler { ring: ring.clone(), base: 4.into(), digit_count: 2.into() };
        let key =
            BggPublicKeyWire { matrix: ring.input("input-public", (1, 2)), reveal_plaintext: true };
        let mut slots = BggTallSlotPublicKeyLowering {
            compiler: compiler.clone(),
            diagonal_mask_public_key: BggPublicKeyWire {
                matrix: ring.input("mask-public", (1, 2)),
                reveal_plaintext: true,
            },
            configured_slot_count: 2,
            rotations: BTreeMap::new(),
            anchor_reduce: None,
        };
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input = circuit.input(1).as_single_wire();
        let transferred = circuit.slot_transfer_gate(input, &[(1, None), (0, None)]);
        circuit.output([transferred]);
        let result = crate::PolyCircuitCompiler { public_key: compiler }
            .compile_public_keys_with_lowerings(
                &circuit,
                key.clone(),
                [key],
                &mut crate::NoPublicLookup::default(),
                &mut slots,
            );
        assert!(matches!(
            result,
            Err(CircuitCompileError::Unsupported { feature: "nonidentity Tall slot transfer", .. })
        ));
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
}
