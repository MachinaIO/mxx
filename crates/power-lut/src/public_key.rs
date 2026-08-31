//! Public-key-only Power-LUT projections.
//!
//! Every public operation in this module consumes matrices and public
//! descriptors. It does not require producer-bound vectors, private GSW
//! ciphertexts, sparse support, or private schedules. Private setup code may
//! convert its validated values into these descriptors, but the public-key
//! compiler itself has no private-data dependency.

use std::collections::BTreeMap;

use mxx_bgg::BggPublicKeyWire;

use crate::{
    PowerLutError,
    program::{
        FamilyRange, PowerLutProgram, ProgramBindings, ProgramFamilyRanges, ProgramInputId,
        ProgramLoweringBackend, ProgramWireId, RhsInputId, lower_program,
    },
    rhs::{PowerLutPublicRhsPackage, PowerRhsPackageError},
};
use mxx_dsl::{Bytes, DslError, Family, HashTag, Mat};
use mxx_ir_core::{IntExpr, ParamEnv, node::NodeKind};
use num_traits::ToPrimitive;

/// Public-only descriptor for one fixed-secret automorphism round.
///
/// It contains the public switch projection and mask matrix needed by the
/// public-key compiler. No private vector, GSW ciphertext, or sparse support
/// is retained. Use [`Self::new`] when deriving this descriptor from public
/// setup metadata, or the crate-private conversion when a private setup is
/// already available.
#[derive(Clone)]
pub struct AutomorphismPublicHelper {
    index: usize,
    switch: PowerLutPublicRhsPackage,
    mask: Mat,
}

/// Public companion families for a structural one-hot gate.
///
/// Each parallel family contains one packed `(source_row, target_column)`
/// companion block across all RHS packages. Its columns retain the complete
/// tower-major CRT digit sequence. Together, the families are the public
/// projection of the private packages in canonical row/column order.
/// The family is consumed by a single reusable loop body, so this public
/// representation contains no selector bits, support coordinates, or private
/// GSW material and does not expand the graph with one node per runtime cell.
#[derive(Clone)]
pub struct PublicSelectorFamily {
    companions: Vec<Family<Mat>>,
}

/// Errors raised while deriving public Power-LUT setup material.
#[derive(Debug, thiserror::Error)]
pub enum PowerLutPublicSamplingError {
    #[error(transparent)]
    /// The public companion package has an invalid canonical shape.
    Rhs(#[from] PowerRhsPackageError),
    #[error("invalid public Power-LUT sampler configuration: {0}")]
    /// Setup dimensions or the requested helper width are invalid.
    InvalidConfiguration(&'static str),
}

/// Public-only setup sampler for Power-LUT keys and automorphism helpers.
///
/// Every matrix returned by this type is derived with the existing BGG
/// public-key hash sampler. The API accepts no secret, plaintext, encoding,
/// sparse support, schedule, or selector bit. Private companion vectors,
/// GSW ciphertexts, and errors are sampled only by
/// [`crate::encoding::PowerLutEncodingSampler`] using the same public setup
/// namespace.
#[derive(Clone)]
pub struct PowerLutPublicKeySampler {
    /// BGG dimensions and gadget parameters for the public artifacts.
    pub layout: mxx_bgg::BggSamplerLayout,
}

impl PowerLutPublicKeySampler {
    /// Derives the public companion projection for an explicit RHS package.
    /// This API accepts no RHS payload or ciphertext, and therefore produces
    /// the same matrices for every hidden value prepared under `tag`.
    pub fn sample_rhs_public(
        &self,
        hash_key: Bytes,
        tag: impl Into<HashTag>,
    ) -> Result<PowerLutPublicRhsPackage, PowerLutPublicSamplingError> {
        let columns = self.layout.public_key_columns();
        let sampler = mxx_bgg::BggPublicKeySampler { layout: self.layout.clone() };
        let tag = tag.into();
        let mut by_column = Vec::with_capacity(columns);
        for column in 0..columns {
            let mut column_tag = tag.clone();
            column_tag.push(IntExpr::constant(column));
            let mut keys = sampler.sample(hash_key.clone(), column_tag, &vec![false; columns]);
            by_column.push(keys.drain(1..).map(|key| key.matrix).collect::<Vec<_>>());
        }
        let mut companion_matrices =
            Vec::with_capacity(self.layout.secret_dimension * columns * self.layout.digit_count);
        for row in 0..self.layout.secret_dimension {
            for column in 0..columns {
                let start = row * self.layout.digit_count;
                let end = start + self.layout.digit_count;
                companion_matrices.extend(by_column[column][start..end].iter().cloned());
            }
        }
        Ok(PowerLutPublicRhsPackage::from_sampled_matrices(
            self.layout.secret_dimension,
            columns,
            self.layout.digit_count,
            companion_matrices,
        )?)
    }

    /// Derives one public input-key matrix from the existing BGG sampler.
    pub fn sample_input_key(
        &self,
        hash_key: Bytes,
        tag: impl Into<HashTag>,
    ) -> Result<BggPublicKeyWire, PowerLutPublicSamplingError> {
        let mut keys = mxx_bgg::BggPublicKeySampler { layout: self.layout.clone() }.sample(
            hash_key,
            tag,
            &[false],
        );
        keys.pop()
            .ok_or(PowerLutPublicSamplingError::InvalidConfiguration("input key sample is empty"))
    }

    /// Returns the canonical sign-flag automorphism indices required by a
    /// maximum LUT width. No helper matrix depends on a hidden value.
    pub fn automorphism_helper_indices(
        &self,
        max_lut_width: usize,
    ) -> Result<Vec<usize>, PowerLutPublicSamplingError> {
        let n = self
            .layout
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(PowerLutPublicSamplingError::InvalidConfiguration(
                "ring dimension must be a concrete positive integer",
            ))?;
        if self.layout.secret_dimension == 0 ||
            self.layout.digit_count == 0 ||
            max_lut_width == 0 ||
            !max_lut_width.is_power_of_two() ||
            max_lut_width > n ||
            n % max_lut_width != 0
        {
            return Err(PowerLutPublicSamplingError::InvalidConfiguration(
                "maximum LUT width must be a power of two dividing the ring dimension",
            ));
        }
        Ok((0..max_lut_width.trailing_zeros() as usize)
            .map(|round| (2 * n / (1usize << (round + 1))) + 1)
            .collect())
    }

    /// Samples the public projection of every reusable ClearCoeff helper.
    ///
    /// Companion matrices are generated with distinct domain-separated hash
    /// tags for each automorphism index. The returned helpers contain only
    /// public projections and therefore can be used to derive expected public
    /// keys without knowing the encoding or its plaintext.
    pub fn sample_automorphism_helpers(
        &self,
        hash_key: Bytes,
        tag: impl Into<HashTag>,
        max_lut_width: usize,
    ) -> Result<Vec<AutomorphismPublicHelper>, PowerLutPublicSamplingError> {
        let indices = self.automorphism_helper_indices(max_lut_width)?;
        let mut root_tag = tag.into();
        root_tag.push("power-lut-automorphism");
        let sampler = mxx_bgg::BggPublicKeySampler { layout: self.layout.clone() };
        indices
            .into_iter()
            .map(|index| {
                let switch_tag = canonical_switch_companion_tag(&root_tag, index);
                let switch = self.sample_rhs_public(hash_key.clone(), switch_tag)?;

                let mask_tag = canonical_mask_tag(&root_tag, index);
                let mask =
                    sampler.sample(hash_key.clone(), mask_tag, &[]).into_iter().next().ok_or(
                        PowerLutPublicSamplingError::InvalidConfiguration(
                            "mask key sample is empty",
                        ),
                    )?;
                Ok(AutomorphismPublicHelper::new(index, switch, mask.matrix))
            })
            .collect()
    }
}

/// Builds the same switch-companion hash domain as the private sampler. This
/// duplication is intentional: the public-key module remains usable without
/// importing a secret-bearing encoding module while retaining one canonical
/// setup namespace.
fn canonical_switch_companion_tag(root: &HashTag, index: usize) -> HashTag {
    let mut tag = root.clone();
    tag.push("switch");
    tag.push(IntExpr::constant(index));
    tag.push("companions");
    tag
}

/// Builds the same automorphism-mask hash domain as the private sampler.
fn canonical_mask_tag(root: &HashTag, index: usize) -> HashTag {
    let mut tag = root.clone();
    tag.push("mask");
    tag.push(IntExpr::constant(index));
    tag
}

impl PublicSelectorFamily {
    /// Creates a public selector family from canonical companion families.
    ///
    /// Every family must have the same runtime cardinality; the companion
    /// position represented by each vector is fixed by the artifact schema.
    pub fn new(companions: Vec<Family<Mat>>) -> Result<Self, PowerLutError> {
        let Some(first) = companions.first() else {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        };
        if companions.iter().any(|family| family.count() != first.count()) {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        Ok(Self { companions })
    }

    fn count(&self) -> &mxx_ir_core::IntExpr {
        self.companions[0].count()
    }

    /// Returns the canonical flat public companion order.
    pub(crate) fn flattened(&self) -> Vec<Family<Mat>> {
        self.companions.clone()
    }

    /// Rebuilds public companions from their canonical flat order and checks
    /// arity, family count, and matrix domain before lowering.
    pub(crate) fn from_flattened(flat: Vec<Family<Mat>>) -> Result<Self, PowerLutError> {
        let Some(first) = flat.first() else {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        };
        let first_type = first.element_type();
        if flat.iter().any(|family| {
            family.count() != first.count() ||
                family.element_type().modulus.canonicalize() != first_type.modulus.canonicalize() ||
                family.element_type().ring_dimension.canonicalize() !=
                    first_type.ring_dimension.canonicalize()
        }) {
            return Err(PowerLutError::InvalidSparseLwrBlock);
        }
        Self::new(flat)
    }
}

impl AutomorphismPublicHelper {
    /// Creates a public automorphism descriptor from public setup values.
    pub fn new(index: usize, switch: PowerLutPublicRhsPackage, mask: Mat) -> Self {
        Self { index, switch, mask }
    }

    /// Returns the automorphism index.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Returns the public switch RHS projection.
    pub fn switch(&self) -> &PowerLutPublicRhsPackage {
        &self.switch
    }

    /// Returns the public automorphism mask matrix.
    pub fn mask(&self) -> &Mat {
        &self.mask
    }
}

#[derive(Clone)]
/// Compiler for the public matrix projection of Power-LUT operations.
///
/// This type accepts public matrices, public RHS projections, and validated
/// [`AutomorphismPublicHelper`] values. Its output is intended to have the same DSL
/// shape and arithmetic as the private [`crate::PowerLutEncodingCompiler`] path.
pub struct PowerLutPublicKeyCompiler {
    /// Public BGG parameters used for decomposition and ring operations.
    pub public_key: mxx_bgg::BggPublicKeyCompiler,
}

impl PowerLutPublicKeyCompiler {
    /// Creates a public-only compiler from BGG public parameters.
    pub fn new(public_key: mxx_bgg::BggPublicKeyCompiler) -> Self {
        Self { public_key }
    }

    /// Lowers a validated program from public input keys and public RHS
    /// descriptors only. One-hot families are supplied as public projections
    /// paired with public scalar values; no private encoding, GSW package,
    /// support, or schedule is accepted by this entry point.
    pub fn compile_program(
        &self,
        program: &PowerLutProgram,
        inputs: &BTreeMap<ProgramInputId, BggPublicKeyWire>,
        rhs_inputs: &BTreeMap<RhsInputId, PowerLutPublicRhsPackage>,
        one_hot_selectors: &BTreeMap<crate::program::RhsFamilyId, PublicSelectorFamily>,
        public_values: &BTreeMap<crate::program::PublicValueFamilyId, Family<Mat>>,
        helpers: &[AutomorphismPublicHelper],
    ) -> Result<BTreeMap<ProgramWireId, BggPublicKeyWire>, PowerLutError> {
        let mut ranges = ProgramFamilyRanges::new();
        for (id, family) in one_hot_selectors {
            let range = FamilyRange::full(family.count().clone())
                .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
            ranges.selector(*id, range);
        }
        for (id, family) in public_values {
            let range = FamilyRange::full(family.count().clone())
                .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
            ranges.public_values(*id, range);
        }
        self.compile_program_with_ranges(
            program,
            inputs,
            rhs_inputs,
            one_hot_selectors,
            public_values,
            &ranges,
            helpers,
        )
    }

    /// Lowers a program with explicit contiguous views into flattened public
    /// selector/value families. A bucket can therefore share one family
    /// artifact while selecting only its own non-padding range.
    pub fn compile_program_with_ranges(
        &self,
        program: &PowerLutProgram,
        inputs: &BTreeMap<ProgramInputId, BggPublicKeyWire>,
        rhs_inputs: &BTreeMap<RhsInputId, PowerLutPublicRhsPackage>,
        one_hot_selectors: &BTreeMap<crate::program::RhsFamilyId, PublicSelectorFamily>,
        public_values: &BTreeMap<crate::program::PublicValueFamilyId, Family<Mat>>,
        family_ranges: &ProgramFamilyRanges,
        helpers: &[AutomorphismPublicHelper],
    ) -> Result<BTreeMap<ProgramWireId, BggPublicKeyWire>, PowerLutError> {
        let bindings =
            ProgramBindings::new(inputs, rhs_inputs, one_hot_selectors, public_values, helpers);
        lower_program(program, &bindings, family_ranges, self)
    }
    /// Emits the public projection of one fixed-secret automorphism round.
    ///
    /// Under the BGG convention `c = s A - mu s G + e`, let `A'` be the raw
    /// rotated matrix and `C` the public switch projection. Define
    /// `B = FusePublic(A', C)`. Applying the raw automorphism and then `Fuse`
    /// to the public-key matrix gives
    /// `r = t A' - sigma(mu) s G + e`, where `A'` is the rotated public
    /// matrix and `t` is the transformed payload. The helper mask has the
    /// public equation `h = s D - t G + e`; multiplying it by the BGG
    /// decomposition `D(B)` contributes `s B - t B`. Therefore the returned sum
    ///
    /// `h D(B) + r`
    ///
    /// cancels `-t B + t B` and restores an encoding under the original
    /// secret. The switched public matrix `B` is consumed by that cancellation
    /// and therefore is not added to the output public key: the declared key
    /// is `D * D(B)`, exactly matching the remaining vector term. The
    /// switched matrix `B` is not added separately: it has already been
    /// consumed by the cancellation.
    pub fn automorphism(
        &self,
        input: &Mat,
        helper: &AutomorphismPublicHelper,
    ) -> Result<Mat, PowerLutError> {
        let rotated = input.clone().ring_automorphism(helper.index());
        // `fused` is the public expression for B on the public path. Its
        // payload term is used by the cancellation equation, while the output
        // key retains only the mask projection `D * D(B)`.
        let fused = self.fuse_public(&rotated, helper.switch())?;
        let fused_decomposition = fused
            .clone()
            .decompose(self.public_key.base.clone(), self.public_key.digit_count.clone())
            .as_mat();
        Ok(helper.mask().clone() * fused_decomposition)
    }

    /// Emits the public matrix equation for Fuse using a public RHS projection.
    ///
    /// This uses the crate's CRT-aware packed block algebra. It reads only the
    /// input public matrix and companion projections; private GSW ciphertexts
    /// and encoding vectors are not available on this path.
    pub fn fuse_public(
        &self,
        input: &Mat,
        rhs: &crate::rhs::PowerLutPublicRhsPackage,
    ) -> Result<Mat, PowerLutError> {
        let digits = self
            .public_key
            .digit_count
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|v| v.to_usize())
            .ok_or(PowerLutError::InvalidLut)?;
        let lhs_decomposition =
            input.clone().decompose(self.public_key.base.clone(), digits).as_mat();
        self.fuse_public_with_decomposition(input, &lhs_decomposition, rhs)
    }

    /// Emits the public Fuse expression using an already-built decomposition
    /// of `input`. The caller must provide the complete ordinary decomposition
    /// `D(input)`; this helper validates its ring and shape before using it,
    /// and never slices or decomposes the supplied handle.
    pub(crate) fn fuse_public_with_decomposition(
        &self,
        input: &Mat,
        lhs_decomposition: &Mat,
        rhs: &crate::rhs::PowerLutPublicRhsPackage,
    ) -> Result<Mat, PowerLutError> {
        let digits = self
            .public_key
            .digit_count
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|v| v.to_usize())
            .ok_or(PowerLutError::InvalidLut)?;
        let source_dimension = input
            .matrix_type()
            .rows
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(PowerLutError::InvalidLut)?;
        let target_columns =
            source_dimension.checked_mul(digits).ok_or(PowerLutError::InvalidLut)?;
        let expected_input_type =
            self.public_key.ring.matrix_type((source_dimension, target_columns));
        let input_type = input.matrix_type();
        if input_type.modulus.canonicalize() != expected_input_type.modulus.canonicalize() ||
            input_type.ring_dimension.canonicalize() !=
                expected_input_type.ring_dimension.canonicalize() ||
            input_type.columns.evaluate(&ParamEnv::default()).ok().and_then(|v| v.to_usize()) !=
                Some(target_columns)
        {
            return Err(PowerLutError::InvalidLut);
        }
        let expected_decomposition_type =
            self.public_key.ring.matrix_type((target_columns, target_columns));
        let decomposition_type = lhs_decomposition.matrix_type();
        if decomposition_type.modulus.canonicalize() !=
            expected_decomposition_type.modulus.canonicalize() ||
            decomposition_type.ring_dimension.canonicalize() !=
                expected_decomposition_type.ring_dimension.canonicalize() ||
            decomposition_type
                .rows
                .evaluate(&ParamEnv::default())
                .ok()
                .and_then(|v| v.to_usize()) !=
                Some(target_columns) ||
            decomposition_type
                .columns
                .evaluate(&ParamEnv::default())
                .ok()
                .and_then(|v| v.to_usize()) !=
                Some(target_columns)
        {
            return Err(PowerLutError::InvalidLut);
        }
        let decomposition_node = lhs_decomposition.value_handle().node();
        let NodeKind::MatrixScale { scalar } = decomposition_node.kind() else {
            return Err(PowerLutError::InvalidLut);
        };
        if *scalar != IntExpr::constant(1) {
            return Err(PowerLutError::InvalidLut);
        }
        let Some(decomposition_argument) = decomposition_node.arguments().first() else {
            return Err(PowerLutError::InvalidLut);
        };
        let NodeKind::GadgetDecompose { base, small, digit_count } =
            decomposition_argument.node().kind()
        else {
            return Err(PowerLutError::InvalidLut);
        };
        let Some(gadget_input) = decomposition_argument.node().arguments().first() else {
            return Err(PowerLutError::InvalidLut);
        };
        if gadget_input != input.value_handle() {
            return Err(PowerLutError::InvalidLut);
        }
        if *small ||
            base.canonicalize() != self.public_key.base.canonicalize() ||
            digit_count.canonicalize() != self.public_key.digit_count.canonicalize()
        {
            return Err(PowerLutError::InvalidLut);
        }
        let expected_companions =
            source_dimension.checked_mul(target_columns).ok_or(PowerLutError::InvalidLut)?;
        if rhs.companion_count() != expected_companions {
            return Err(PowerLutError::InvalidLut);
        }
        let expected_block_columns =
            target_columns.checked_mul(digits).ok_or(PowerLutError::InvalidLut)?;
        if rhs.first_companion().and_then(|companion| {
            companion
                .matrix_type()
                .columns
                .evaluate(&ParamEnv::default())
                .ok()
                .and_then(|value| value.to_usize())
        }) != Some(expected_block_columns)
        {
            return Err(PowerLutError::InvalidLut);
        }
        let ring = mxx_dsl::Ring::new(
            input.matrix_type().modulus.clone(),
            input
                .matrix_type()
                .ring_dimension
                .evaluate(&ParamEnv::default())
                .ok()
                .and_then(|value| value.to_usize())
                .ok_or(PowerLutError::InvalidLut)?,
        );
        crate::utils::fuse_columns(
            None,
            lhs_decomposition,
            None,
            source_dimension,
            target_columns,
            digits,
            &ring,
            &self.public_key.base,
            |row, column| rhs.companion_block(row, column, target_columns),
        )
    }
    /// Emits the unnormalised coefficient-sieving (ClearCoeff) projection.
    ///
    /// Round `i` uses `r_i = 2n / 2^(i + 1) + 1`; adding the transformed
    /// state cancels odd quotient residues and doubles the survivors. After
    /// `log2(width)` rounds, only positions divisible by `width` remain, each
    /// with amplitude `width`. The helper indices are checked in this exact
    /// order. Normalisation is deliberately deferred until all LUT branches
    /// have been summed by [`Self::single_input_lut`].
    pub fn clear_coeff(
        &self,
        input: &Mat,
        width: usize,
        helpers: &[AutomorphismPublicHelper],
    ) -> Result<Mat, PowerLutError> {
        let n = input
            .matrix_type()
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|v| v.to_usize())
            .ok_or(PowerLutError::InvalidClearCoeffWidth)?;
        if width == 0 ||
            !width.is_power_of_two() ||
            n % width != 0 ||
            helpers.len() != width.trailing_zeros() as usize
        {
            return Err(PowerLutError::InvalidClearCoeffWidth);
        }
        let mut state = input.clone();
        for (round, helper) in helpers.iter().enumerate() {
            let expected = (2 * n / (1usize << (round + 1))) + 1;
            if helper.index() != expected {
                return Err(PowerLutError::InvalidAutomorphismHelper);
            }
            state = state.clone() + self.automorphism(&state, helper)?;
        }
        Ok(state)
    }
    /// Evaluates a power-of-two table using public rotations and ClearCoeff.
    ///
    /// Branches remain unnormalised while they are accumulated. The final
    /// sum receives one `width^-1` gadget product, matching the encoding path.
    pub fn single_input_lut(
        &self,
        input: &Mat,
        table: &[usize],
        helpers: &[AutomorphismPublicHelper],
    ) -> Result<Mat, PowerLutError> {
        if table.is_empty() || !table.len().is_power_of_two() {
            return Err(PowerLutError::InvalidLut);
        }
        let n = input
            .matrix_type()
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|v| v.to_usize())
            .ok_or(PowerLutError::InvalidLut)?;
        let ring = self.public_key.ring.clone();
        let shifts = Family::pack(
            (0..table.len())
                .map(|candidate| {
                    crate::utils::rotation_power(&ring, (2 * n - candidate % (2 * n)) % (2 * n), n)
                })
                .collect(),
        )
        .map_err(|_| PowerLutError::InvalidLut)?;
        let output_rotations = Family::pack(
            table
                .iter()
                .copied()
                .map(|output| crate::utils::rotation_power(&ring, output, n))
                .collect(),
        )
        .map_err(|_| PowerLutError::InvalidLut)?;
        let input = input.clone();
        let input_for_loop = input.clone();
        let helpers = helpers.to_vec();
        let branches = Family::try_parallel_zip_many_values(
            vec![shifts, output_rotations],
            move |_index, items| {
                let mut items = items.into_iter();
                let shift = items.next().ok_or(DslError::Schema)?;
                let output_rotation = items.next().ok_or(DslError::Schema)?;
                let shifted = input_for_loop.clone() * shift;
                let selected = self
                    .clear_coeff(&shifted, table.len(), &helpers)
                    .map_err(|_| DslError::Schema)?;
                Ok(selected * output_rotation)
            },
        )
        .map_err(|_| PowerLutError::InvalidLut)?;
        let sum = crate::encoding::balanced_sum_family(branches)?;

        // ClearCoeff leaves a factor `table.len()` on surviving coefficients.
        // The conceptual target is `G^-1(width^-1 G)`: the explicit `G`
        // preserves the canonical payload since
        // `G G^-1(width^-1 G) = width^-1 G`. Pass only `width^-1` to the BGG
        // helper because it inserts `G` internally, yielding exactly one
        // decomposition after branch accumulation.
        let modulus = input
            .matrix_type()
            .modulus
            .evaluate(&ParamEnv::default())
            .map_err(|_| PowerLutError::InvalidLut)?;
        let inverse = crate::encoding::modular_inverse(
            &(num_bigint::BigInt::from(table.len()) % &modulus),
            &modulus,
        )
        .ok_or(PowerLutError::InvalidLut)?;
        let scalar = self.public_key.ring.polynomial([inverse.into()]);
        Ok(self
            .public_key
            .large_scalar_mul(&BggPublicKeyWire { matrix: sum, reveal_plaintext: false }, &scalar)
            .matrix)
    }
    /// Fuses a two-input RHS and evaluates the resulting flattened LUT.
    ///
    /// The table entry for inputs `(u, v)` is selected using
    /// `u + lhs_width * v`. These table dimensions describe the logical
    /// exponent domain and are separate from the CRT limb order used inside
    /// Fuse.
    pub fn two_input_lut(
        &self,
        lhs: &Mat,
        rhs: &PowerLutPublicRhsPackage,
        lhs_width: usize,
        rhs_width: usize,
        table: &[usize],
        helpers: &[AutomorphismPublicHelper],
    ) -> Result<Mat, PowerLutError> {
        if table.len() != lhs_width.saturating_mul(rhs_width) {
            return Err(PowerLutError::InvalidLut);
        }
        let out = self.fuse_public(lhs, rhs)?;
        self.single_input_lut(&out, table, helpers)
    }
    /// Computes the public combined refresh target `a_sum_t - (q/q_t) * a_prime`.
    pub fn refresh_combined_target(
        &self,
        a_sum_t: &Mat,
        a_prime: &Mat,
        q: &mxx_ir_core::IntExpr,
        q_t: &mxx_ir_core::IntExpr,
    ) -> Result<Mat, PowerLutError> {
        let q_value = q.evaluate(&ParamEnv::default()).map_err(|_| PowerLutError::InvalidLut)?;
        let qt_value = q_t.evaluate(&ParamEnv::default()).map_err(|_| PowerLutError::InvalidLut)?;
        if qt_value <= 0.into() || &q_value % &qt_value != 0.into() {
            return Err(PowerLutError::InvalidLut);
        }
        Ok(a_sum_t.clone() -
            self.public_key.ring.polynomial([(q_value / qt_value).into()]) * a_prime.clone())
    }
}

impl ProgramLoweringBackend for PowerLutPublicKeyCompiler {
    type Wire = BggPublicKeyWire;
    type Rhs = PowerLutPublicRhsPackage;
    type SelectorFamily = PublicSelectorFamily;
    type PublicValueFamily = Family<Mat>;
    type Helper = AutomorphismPublicHelper;

    fn unary(
        &self,
        input: Self::Wire,
        table: &crate::program::LutTable,
        helpers: &[Self::Helper],
    ) -> Result<Self::Wire, PowerLutError> {
        Ok(BggPublicKeyWire {
            matrix: self.single_input_lut(&input.matrix, table.values(), helpers)?,
            reveal_plaintext: false,
        })
    }

    fn binary(
        &self,
        lhs: Self::Wire,
        rhs: &Self::Rhs,
        table: &crate::program::LutTable,
        helpers: &[Self::Helper],
    ) -> Result<Self::Wire, PowerLutError> {
        self.two_input_lut(
            &lhs.matrix,
            rhs,
            table.input_width(),
            table.rhs_width().expect("shared traversal validates binary LUT"),
            table.values(),
            helpers,
        )
        .map(|matrix| BggPublicKeyWire { matrix, reveal_plaintext: false })
    }

    fn one_hot(
        &self,
        lhs: Self::Wire,
        selectors: &Self::SelectorFamily,
        public_values: &Self::PublicValueFamily,
        selector_range: &FamilyRange,
        public_value_range: &FamilyRange,
        table: &crate::program::LutTable,
        helpers: &[Self::Helper],
    ) -> Result<Self::Wire, PowerLutError> {
        if selectors.count() != public_values.count() ||
            selector_range != public_value_range ||
            !selector_range.is_within(selectors.count())
        {
            return Err(crate::program::ProgramValidationError::WidthMismatch.into());
        }

        // Use a fixed structural loop capacity. The logical range may be
        // shorter for a sparse bucket, but its inactive tail is masked before
        // weighting. This keeps indexed-family wire types independent of the
        // enclosing bucket-loop binder and excludes padding RHS material.
        let capacity = selector_range
            .capacity()
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(PowerLutError::InvalidSparseLwrBlock)?;
        let count = mxx_dsl::Int::evaluate(selector_range.count().clone());
        let start = mxx_dsl::Int::evaluate(selector_range.start().clone());
        let mask_type = public_values.element_type().clone();
        let (safe_indices, active_masks) =
            super::encoding::one_hot_indices_and_masks(capacity, count, start, mask_type)
                .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
        let companions = selectors
            .companions
            .iter()
            .map(|family| {
                family
                    .clone()
                    .parallel_gather(safe_indices.clone())
                    .map_err(|_| PowerLutError::InvalidSparseLwrBlock)
            })
            .collect::<Result<Vec<_>, PowerLutError>>()?;
        let values = public_values
            .clone()
            .parallel_gather(safe_indices)
            .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
        let lhs_matrix = lhs.matrix.clone();
        let compiler = Self { public_key: self.public_key.clone() };
        let mut zipped = companions;
        zipped.push(values);
        zipped.push(active_masks);
        let weighted =
            Family::<Mat>::try_parallel_zip_many_values(zipped, move |_index, mut zipped| {
                let active = zipped.pop().ok_or(DslError::Schema)?;
                let value = zipped.pop().ok_or(DslError::Schema)?;
                let rhs = PowerLutPublicRhsPackage::new(zipped).map_err(|_| DslError::Schema)?;
                let fused =
                    compiler.fuse_public(&lhs_matrix, &rhs).map_err(|_| DslError::Schema)?;
                Ok(fused * (value * active))
            })
            .map_err(|_| PowerLutError::InvalidSparseLwrBlock)?;
        let selected = crate::encoding::balanced_sum_family(weighted)?;
        Ok(BggPublicKeyWire {
            matrix: self.single_input_lut(&selected, table.values(), helpers)?,
            reveal_plaintext: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::{DslContext, Ring};
    use mxx_ir_core::ParamEnv;
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use mxx_runtime::{
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
        transcript::SamplingMode,
    };
    use num_bigint::BigInt;
    use serial_test::serial;
    use std::collections::BTreeMap;

    #[test]
    fn clear_coeff_public_scaling_preserves_rectangular_public_key_shape() {
        let ring = Ring::new(257, 4);
        let compiler = PowerLutPublicKeyCompiler::new(mxx_bgg::BggPublicKeyCompiler {
            ring: ring.clone(),
            base: 2.into(),
            digit_count: 2.into(),
        });
        let output = compiler.clear_coeff(&ring.input("public-key", (2, 4)), 1, &[]).unwrap();
        DslContext::new("power-lut-public-clear-coeff-shape")
            .output("output", output)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
    }

    #[test]
    fn fuse_public_rejects_decomposition_of_another_same_shaped_input() {
        let ring = Ring::new(257, 4);
        let compiler = PowerLutPublicKeyCompiler::new(mxx_bgg::BggPublicKeyCompiler {
            ring: ring.clone(),
            base: 8.into(),
            digit_count: 2.into(),
        });
        let input = ring.input("fuse-identity-input", (1, 2));
        let other = ring.input("fuse-identity-other", (1, 2));
        let rhs =
            PowerLutPublicRhsPackage::new(vec![ring.zero((1, 4)), ring.zero((1, 4))]).unwrap();
        let other_decomposition = other.decompose(8, 2).as_mat();

        assert!(matches!(
            compiler.fuse_public_with_decomposition(&input, &other_decomposition, &rhs),
            Err(PowerLutError::InvalidLut)
        ));
    }

    #[test]
    #[serial(dcrt_runtime)]
    fn refresh_target_equation_runs_with_concrete_crt_values() {
        let parameters = DCRTPolyParams::new(2, 1, 17, 16);
        assert_eq!(parameters.crt_depth(), 1);
        assert_eq!(parameters.crt_bits().div_ceil(parameters.base_bits() as usize), 2);
        let q = BigInt::from(parameters.modulus().as_ref().clone());
        let qt = BigInt::from(parameters.to_crt().0[0]);
        assert_eq!(&q % &qt, BigInt::from(0));
        let ring = Ring::new(q.clone(), parameters.ring_dimension() as usize);
        let compiler = PowerLutPublicKeyCompiler::new(mxx_bgg::BggPublicKeyCompiler {
            ring: ring.clone(),
            base: 65_536.into(),
            digit_count: 2.into(),
        });
        let target = compiler
            .refresh_combined_target(
                &ring.polynomial([3.into()]),
                &ring.polynomial([1.into()]),
                &q.clone().into(),
                &qt.clone().into(),
            )
            .unwrap();
        let graph = DslContext::new("power-lut-refresh-public-runtime")
            .output("target", target)
            .unwrap()
            .build()
            .unwrap();
        let validated = graph.validate(&ParamEnv::default()).unwrap();
        let result = execute(
            &validated,
            &mut cpu_backend([parameters.clone()]),
            BTreeMap::new(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let RuntimeValue::Matrix(actual) = &result.outputs["target"] else { panic!("matrix") };
        let scale = (&q / &qt).to_biguint().unwrap();
        let expected_value =
            (BigInt::from(3) - BigInt::from_biguint(num_bigint::Sign::Plus, scale)) % &q;
        let expected = DCRTPolyMatrix::from_poly_vec(
            &parameters,
            vec![vec![DCRTPoly::from_biguint_to_constant(
                &parameters,
                expected_value.to_biguint().unwrap(),
            )]],
        );
        assert_eq!(actual.as_ref(), &expected);
    }

    #[test]
    fn public_structural_program_uses_only_public_families() {
        let ring = Ring::new(257, 4);
        let compiler = PowerLutPublicKeyCompiler::new(mxx_bgg::BggPublicKeyCompiler {
            ring: ring.clone(),
            base: 2.into(),
            digit_count: 1.into(),
        });
        let mut builder = crate::program::PowerLutProgramBuilder::new();
        let input = builder.input(1).unwrap();
        let selector_family = builder.rhs_family(1).unwrap();
        let public_value_family = builder.public_value_family(1).unwrap();
        let lut = builder.lut(crate::program::LutTable::unary(1, 1, vec![0]).unwrap()).unwrap();
        let output = builder
            .one_hot(builder.input_wire(input).unwrap(), selector_family, public_value_family, lut)
            .unwrap();
        builder.output(output).unwrap();
        let program = builder.build().unwrap();

        let selector = PublicSelectorFamily::new(vec![ring.input_family(
            "public-selector-companions",
            32,
            (1, 1),
        )])
        .unwrap();
        let values = ring.input_family("public-values", 32, (1, 1));
        let mut ranges = crate::program::ProgramFamilyRanges::new();
        let range = crate::program::FamilyRange::full(32).unwrap();
        ranges.selector(selector_family, range.clone());
        ranges.public_values(public_value_family, range);
        let wires = compiler
            .compile_program_with_ranges(
                &program,
                &BTreeMap::from([(
                    input,
                    BggPublicKeyWire {
                        matrix: ring.input("public-input", (1, 1)),
                        reveal_plaintext: false,
                    },
                )]),
                &BTreeMap::new(),
                &BTreeMap::from([(selector_family, selector)]),
                &BTreeMap::from([(public_value_family, values)]),
                &ranges,
                &[],
            )
            .unwrap();
        let graph = DslContext::new("power-lut-public-structural-program")
            .output("result", wires[&output].matrix.clone())
            .unwrap()
            .build()
            .unwrap();
        assert!(graph.graph.root_scope().nodes().iter().all(|node| !matches!(
            node.kind(),
            mxx_ir_core::node::NodeKind::Input {
                artifact: Some(mxx_ir_core::node::ArtifactInput {
                    confidentiality: mxx_ir_core::artifact::ArtifactConfidentiality::Private,
                    ..
                }),
                ..
            }
        )));
    }

    #[test]
    fn public_one_hot_nonuniform_bucket_range_is_structural() {
        let ring = Ring::new(257, 4);
        let compiler = PowerLutPublicKeyCompiler::new(mxx_bgg::BggPublicKeyCompiler {
            ring: ring.clone(),
            base: 2.into(),
            digit_count: 1.into(),
        });
        let mut builder = crate::program::PowerLutProgramBuilder::new();
        let input = builder.input(1).unwrap();
        let selector_family = builder.rhs_family(1).unwrap();
        let public_value_family = builder.public_value_family(1).unwrap();
        let lut = builder.lut(crate::program::LutTable::unary(1, 1, vec![0]).unwrap()).unwrap();
        let selected = builder
            .one_hot(builder.input_wire(input).unwrap(), selector_family, public_value_family, lut)
            .unwrap();
        builder.output(selected).unwrap();
        let program = builder.build().unwrap();

        let family_count = 16;
        let production = mxx_ir_core::artifact::ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([3; 32]),
            execution_nonce: [4; 32],
        };
        let selector = PublicSelectorFamily::new(vec![ring.family_artifact_input(
            production.clone(),
            "nonuniform-public-selector",
            family_count,
            (1, 1),
            mxx_ir_core::artifact::ArtifactConfidentiality::Public,
        )])
        .unwrap();
        let values = ring.family_artifact_input(
            production,
            "nonuniform-public-values",
            family_count,
            (1, 1),
            mxx_ir_core::artifact::ArtifactConfidentiality::Public,
        );
        let mut ranges = crate::program::ProgramFamilyRanges::new();
        let range = crate::program::FamilyRange::bounded(2usize, 3usize, 8).unwrap();
        ranges.selector(selector_family, range.clone());
        ranges.public_values(public_value_family, range);
        let wires = compiler
            .compile_program_with_ranges(
                &program,
                &BTreeMap::from([(
                    input,
                    BggPublicKeyWire {
                        matrix: ring.input("nonuniform-public-input", (1, 1)),
                        reveal_plaintext: false,
                    },
                )]),
                &BTreeMap::new(),
                &BTreeMap::from([(selector_family, selector)]),
                &BTreeMap::from([(public_value_family, values)]),
                &ranges,
                &[],
            )
            .unwrap();
        let graph = DslContext::new("power-lut-nonuniform-public-one-hot")
            .output("result", wires[&selected].matrix.clone())
            .unwrap()
            .build()
            .unwrap();
        let all_nodes =
            graph.graph.scopes().values().flat_map(|scope| scope.nodes()).collect::<Vec<_>>();
        let parallel_loops = graph
            .graph
            .root_scope()
            .nodes()
            .iter()
            .filter(|node| matches!(node.kind(), mxx_ir_core::node::NodeKind::ParallelLoop(_)))
            .count();
        assert!(parallel_loops >= 2, "range gathers and Fuse must remain structural loops");
        assert!(graph.graph.root_scope().nodes().iter().any(|node| matches!(
            node.kind(),
            mxx_ir_core::node::NodeKind::ParallelLoop(spec)
                if spec.count == mxx_ir_core::IntExpr::constant(8)
        )));
        assert!(all_nodes.iter().any(|node| matches!(
            node.kind(),
            mxx_ir_core::node::NodeKind::IntBinary(mxx_ir_core::node::IntBinaryOp::Remainder)
        )));
        assert!(all_nodes.iter().any(|node| matches!(
            node.kind(),
            mxx_ir_core::node::NodeKind::LiftIntegerToConstantPolynomial { .. }
        )));
        let weighted_loop_has_no_singleton_pack = graph
            .graph
            .root_scope()
            .nodes()
            .iter()
            .filter_map(|node| match node.kind() {
                mxx_ir_core::node::NodeKind::ParallelLoop(spec)
                    if spec.count == mxx_ir_core::IntExpr::constant(8) &&
                        spec.input_modes
                            .iter()
                            .position(|mode| {
                                matches!(mode, mxx_ir_core::node::LoopInputMode::Broadcast)
                            })
                            .is_some_and(|first_broadcast| {
                                spec.input_modes[..first_broadcast].iter().all(|mode| {
                                    matches!(mode, mxx_ir_core::node::LoopInputMode::Zip)
                                }) && spec.input_modes[first_broadcast..].iter().all(|mode| {
                                    matches!(mode, mxx_ir_core::node::LoopInputMode::Broadcast)
                                })
                            }) =>
                {
                    graph.graph.root_scope().node_id(node).and_then(|id| {
                        graph
                            .graph
                            .child_scope_id(&mxx_ir_core::graph::FrozenGraphScopeId::Root, id)
                    })
                }
                _ => None,
            })
            .any(|scope_id| {
                graph.graph.scope(&scope_id).unwrap().nodes().iter().all(|node| {
                    !matches!(
                        node.kind(),
                        mxx_ir_core::node::NodeKind::FamilyPack { count }
                            if *count == mxx_ir_core::IntExpr::constant(1)
                    )
                })
            });
        assert!(weighted_loop_has_no_singleton_pack);
        assert!(graph.graph.root_scope().nodes().iter().all(|node| !matches!(
            node.kind(),
            mxx_ir_core::node::NodeKind::Input {
                artifact: Some(mxx_ir_core::node::ArtifactInput {
                    confidentiality: mxx_ir_core::artifact::ArtifactConfidentiality::Private,
                    ..
                }),
                ..
            }
        )));
    }

    #[test]
    fn public_clear_coeff_rejects_helpers_in_the_wrong_round_order() {
        let ring = Ring::new(257, 4);
        let compiler = PowerLutPublicKeyCompiler::new(mxx_bgg::BggPublicKeyCompiler {
            ring: ring.clone(),
            base: 2.into(),
            digit_count: 1.into(),
        });
        let rhs = crate::rhs::PowerLutPublicRhsPackage::new(vec![ring.zero((1, 1))]).unwrap();
        let first = AutomorphismPublicHelper::new(5, rhs.clone(), ring.zero((1, 1)));
        let second = AutomorphismPublicHelper::new(3, rhs, ring.zero((1, 1)));
        let input = ring.input("public-clear-coeff", (1, 1));

        assert!(matches!(
            compiler.clear_coeff(&input, 4, &[second, first]),
            Err(PowerLutError::InvalidAutomorphismHelper)
        ));
    }

    #[test]
    fn public_sampler_derives_only_reusable_hash_domain_material() {
        let layout = mxx_bgg::BggSamplerLayout {
            modulus: 257.into(),
            ring_dimension: 4.into(),
            secret_dimension: 2,
            digit_count: 2,
            gadget_base: 2.into(),
        };
        let ring = layout.ring();
        let sampler = PowerLutPublicKeySampler { layout };
        let hash_key = ring.bytes_input("public-hash-key", 32);
        let input = sampler.sample_input_key(hash_key.clone(), &b"input"[..]).unwrap();
        let helpers = sampler.sample_automorphism_helpers(hash_key, &b"helpers"[..], 4).unwrap();
        assert_eq!(
            helpers.iter().map(AutomorphismPublicHelper::index).collect::<Vec<_>>(),
            vec![5, 3]
        );
        let mut context = DslContext::new("power-lut-public-sampler-shapes")
            .output("input", input.matrix)
            .unwrap();
        for (round, helper) in helpers.iter().enumerate() {
            context = context
                .output(format!("mask-{round}"), helper.mask().clone())
                .unwrap()
                .output(
                    format!("switch-{round}"),
                    helper.switch().companion(0, 0, 4).unwrap().clone(),
                )
                .unwrap();
        }
        context.build().unwrap().validate(&ParamEnv::default()).unwrap();
    }
}
