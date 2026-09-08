//! Scheme-independent sparse-LWR PRF declarations and Program interpretation.
use crate::{
    layout::ExpLayout,
    noise_refresh::pbc::{PbcActiveCellIndex, PbcPublicLayout},
    program::{
        AttributeKnowledge, ExponentLutProgram, ExponentLutProgramBuilder, ExponentLutProgramId,
        FamilyRange, Gate, ProgramGateInstance, ProgramInputId, ProgramLowerError, ProgramStep,
        ProgramValidationError, ProgramWireType, WireId,
    },
    readout::{ExponentLUTReadoutScheme, ValueReadout},
    scheme::ExponentLUTFixedRhsScheme,
};
use mxx_dsl::{Family, GraphValue, GraphValueSchema, Int, Mat};
use serde::{Deserialize, Serialize};

/// Scheme-independent errors in native PRF declarations and bindings.
#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum PrfError {
    #[error("invalid native sparse-LWR profile, block layout or input binding")]
    InvalidSparseLwrBlock,
    #[error(transparent)]
    Program(#[from] ProgramValidationError),
    #[error("PRF artifact binding or transport failed: {0}")]
    Artifact(String),
}

/// Preprocessing-fixed public key and terminal preparation for one PRF.
/// Payload-secret sampling belongs to the backend; the online mask secret
/// is absent from this shared application boundary.
pub struct PrfPublicSetup<S: ExponentLUTReadoutScheme> {
    pub initial: S::PublicKey,
    pub readout_preparation: S::ReadoutPreparation,
    pub readout_public_masks: S::ReadoutPublicMasks,
}

/// Native sparse-LWR profile.
///
/// The ring dimension is derived from the native layout as
/// `n = 2^(m + width_log2)`. Consequently the exponent modulus is
/// `Q = 2^(width_log2 + 1)` and the balanced digit base is `P = 2*p_prime`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct SparseLwrPrfProfile {
    pub layout: ExpLayout,
    pub p_prime: usize,
}

impl SparseLwrPrfProfile {
    /// Validates a deserialized or otherwise externally assembled profile.
    /// Public fields are intentional for declarative manifests, therefore all
    /// consuming boundaries must call this check instead of trusting them.
    pub fn validate(&self) -> Result<(), PrfError> {
        if self.p_prime == 0 {
            return Err(PrfError::InvalidSparseLwrBlock);
        }
        let ring_log2 = self
            .layout
            .stride_log2
            .checked_add(self.layout.width_log2)
            .ok_or(PrfError::InvalidSparseLwrBlock)?;
        let ring_dimension =
            1usize.checked_shl(ring_log2).ok_or(PrfError::InvalidSparseLwrBlock)?;
        self.layout.validate(ring_dimension).map_err(|_| PrfError::InvalidSparseLwrBlock)?;
        let q = 1usize
            .checked_shl(self.layout.width_log2 + 1)
            .ok_or(PrfError::InvalidSparseLwrBlock)?;
        let p = self.p_prime.checked_mul(2).ok_or(PrfError::InvalidSparseLwrBlock)?;
        if p < 2 || p > q || q % p != 0 {
            return Err(PrfError::InvalidSparseLwrBlock);
        }
        Ok(())
    }

    /// Validates a native profile and derives its ring from the full layout.
    pub fn new(layout: ExpLayout, p_prime: usize) -> Result<Self, PrfError> {
        let profile = Self { layout, p_prime };
        profile.validate()?;
        Ok(profile)
    }

    pub const fn native_ring_log2(&self) -> u32 {
        self.layout.stride_log2 + self.layout.width_log2
    }

    pub fn ring_dimension(&self) -> Result<usize, PrfError> {
        1usize.checked_shl(self.native_ring_log2()).ok_or(PrfError::InvalidSparseLwrBlock)
    }

    pub fn q(&self) -> Result<usize, PrfError> {
        1usize.checked_shl(self.layout.width_log2 + 1).ok_or(PrfError::InvalidSparseLwrBlock)
    }

    pub fn p(&self) -> Result<usize, PrfError> {
        self.p_prime.checked_mul(2).ok_or(PrfError::InvalidSparseLwrBlock)
    }

    pub const fn native_width(&self) -> usize {
        1usize << self.layout.width_log2
    }

    /// Returns the canonical sparse-LWR value table
    /// `f(k) = floor(p' * k / n') + 1` for `0 <= k < n'`.
    pub fn canonical_readout(&self) -> Result<ValueReadout, PrfError> {
        let n_prime = self.native_width();
        let values = (0..n_prime)
            .map(|k| self.p_prime.checked_mul(k).map(|value| value / n_prime + 1))
            .collect::<Option<Vec<_>>>()
            .ok_or(PrfError::InvalidSparseLwrBlock)?;
        ValueReadout::from_canonical_values(self.layout, values)
            .map_err(|_| PrfError::InvalidSparseLwrBlock)
    }
}

impl<'de> Deserialize<'de> for SparseLwrPrfProfile {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawProfile {
            layout: ExpLayout,
            p_prime: usize,
        }
        let raw = RawProfile::deserialize(deserializer)?;
        let profile = Self { layout: raw.layout, p_prime: raw.p_prime };
        profile.validate().map_err(serde::de::Error::custom)?;
        Ok(profile)
    }
}

/// A native accumulator plus its value-form readout.
///
/// `initial_input` is the accumulator state initialized by the evaluator to
/// `mono_m(0)=1`; it is represented as a typed program input rather than a
/// label-dependent declaration. Its accumulation is the executed fixed-product
/// chain, shared by preparation, evaluation, and noise tracing.
#[derive(Clone)]
pub struct SparseLwrPrfProgram {
    pub accumulation: ExponentLutProgram,
    pub readout: ValueReadout,
    pub(crate) initial_input: ProgramInputId,
    pub(crate) output: WireId,
    pub(crate) profile: SparseLwrPrfProfile,
    pub(crate) bucket_width: usize,
    pub(crate) block_count: usize,
}

impl std::fmt::Debug for SparseLwrPrfProgram {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SparseLwrPrfProgram")
            .field("id", &self.id())
            .field("profile", &self.profile)
            .field("bucket_width", &self.bucket_width)
            .field("block_count", &self.block_count)
            .field("readout", &self.readout.identity())
            .finish()
    }
}

impl SparseLwrPrfProgram {
    /// Prepare this Program's final readout through the scheme's public
    /// setup capability. The application does not construct decompositions.
    pub fn prepare_readout<S: ExponentLUTReadoutScheme>(
        &self,
        scheme: &S,
        inputs: &crate::scheme::PublicProgramValue<S>,
        preparation: &S::ReadoutPreparation,
    ) -> Result<S::ReadoutArtifacts, S::ReadoutError> {
        scheme.prepare_readout(&self.readout, inputs, preparation)
    }

    /// Public projection of this Program's canonical final value readout.
    pub fn project_readout<S: ExponentLUTReadoutScheme>(
        &self,
        scheme: &S,
        inputs: &crate::scheme::PublicProgramValue<S>,
        artifacts: &S::ReadoutArtifacts,
        masks: &S::ReadoutPublicMasks,
    ) -> Result<S::ReadoutPublicOutput, S::ReadoutError> {
        scheme.project_readout(&self.readout, inputs, artifacts, masks)
    }

    /// Online readout consumes prepared witnesses and online masks through
    /// associated types; application code cannot inspect their representation.
    pub fn evaluate_readout<S: ExponentLUTReadoutScheme>(
        &self,
        scheme: &S,
        inputs: &crate::scheme::EncodingProgramValue<S>,
        artifacts: &S::ReadoutArtifacts,
        masks: &S::ReadoutEncodingMasks,
    ) -> Result<S::ReadoutEncodingOutput, S::ReadoutError> {
        scheme.evaluate_readout(&self.readout, inputs, artifacts, masks)
    }

    /// Canonical active-member ranges shared by ordinary bindings and cached
    /// batched preparation. PBC scheduling does not depend on the backend.
    pub fn active_block_ranges(
        &self,
        layout: &PbcPublicLayout,
    ) -> Result<(usize, Vec<(usize, usize)>), PrfError> {
        validate_profile_program(self.profile, self)?;
        if self.block_count != layout.parameters.bucket_count ||
            self.bucket_width != layout.bucket_width
        {
            return Err(PrfError::InvalidSparseLwrBlock);
        }
        let active =
            PbcActiveCellIndex::build(layout).map_err(|_| PrfError::InvalidSparseLwrBlock)?;
        let ranges = (0..self.block_count)
            .map(|block| {
                active
                    .bucket_iter(block)
                    .and_then(|mut entries| {
                        entries.next().map(|(_, _, offset)| {
                            (offset, active.bucket_active_width(block).unwrap_or_default())
                        })
                    })
                    .ok_or(PrfError::InvalidSparseLwrBlock)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((active.len(), ranges))
    }

    /// Bind a shared selector/label family to the canonical PBC block ranges.
    /// This only constructs structural references; ciphertext preparation is
    /// performed by the scheme when the returned inputs are evaluated.
    pub fn selection_inputs<S: ExponentLUTFixedRhsScheme>(
        &self,
        layout: &PbcPublicLayout,
        selectors: &S::FixedRhs,
        labels: &Family<Mat>,
        id: [u8; 32],
    ) -> Result<Vec<PrfSelectionInputs<S>>, PrfError>
    where
        S::FixedRhs: Clone,
    {
        let (capacity, ranges) = self.active_block_ranges(layout)?;
        ranges
            .into_iter()
            .map(|(start, count)| {
                let active = FamilyRange::bounded(start, count, capacity)
                    .map_err(|_| PrfError::InvalidSparseLwrBlock)?;
                PrfSelectionInputs::new(selectors.clone(), labels, active, id)
                    .map_err(|error| PrfError::Artifact(error.to_string()))
            })
            .collect()
    }

    /// Builds the executed chain: one fixed private product per PBC block.
    pub fn new(
        profile: SparseLwrPrfProfile,
        bucket_width: usize,
        block_count: usize,
    ) -> Result<Self, PrfError> {
        profile.validate()?;
        if bucket_width == 0 || block_count == 0 {
            return Err(PrfError::InvalidSparseLwrBlock);
        }
        let readout = profile.canonical_readout()?;
        let ring_dimension = profile.ring_dimension()?;
        readout.layout.validate(ring_dimension).map_err(|_| PrfError::InvalidSparseLwrBlock)?;

        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(profile.layout, ring_dimension).map_err(PrfError::from)?;
        let initial_input = builder
            .input(
                profile.native_width(),
                ProgramWireType {
                    stride_log2: profile.layout.stride_log2,
                    knowledge: AttributeKnowledge::Private,
                },
            )
            .map_err(PrfError::from)?;
        let mut state = builder.input_wire(initial_input).map_err(PrfError::from)?;
        // Each prepared RHS combines the entire one-hot selector block with
        // its public label monomials. Its payload is a monomial, but all
        // selector errors survive. This is the actual executed gate sequence.
        let width = profile.native_width();
        for _ in 0..block_count {
            let family = builder.rhs_family(width)?;
            let rhs = builder.rhs_input(family, width, width)?;
            state = builder.private_mul(state, rhs)?;
        }
        builder.output(state).map_err(PrfError::from)?;
        let accumulation = builder.build().map_err(PrfError::from)?;
        Ok(Self {
            accumulation,
            readout,
            initial_input,
            output: state,
            profile,
            bucket_width,
            block_count,
        })
    }

    pub const fn profile(&self) -> SparseLwrPrfProfile {
        self.profile
    }

    pub const fn initial_input(&self) -> ProgramInputId {
        self.initial_input
    }

    /// Returns the accumulator's fixed initial native attribute `mono_m(0)`.
    pub fn initial_attribute(&self, ring: &mxx_dsl::Ring) -> Result<mxx_dsl::Mat, PrfError> {
        self.profile
            .layout
            .mono(ring, &num_bigint::BigInt::from(0))
            .map_err(|_| PrfError::InvalidSparseLwrBlock)
    }

    pub const fn output(&self) -> WireId {
        self.output
    }

    pub const fn bucket_width(&self) -> usize {
        self.bucket_width
    }

    pub const fn block_count(&self) -> usize {
        self.block_count
    }

    pub const fn id(&self) -> ExponentLutProgramId {
        self.accumulation.id()
    }

    /// The native accumulator has no reduction or terminal LUT stage.
    pub fn has_only_native_accumulation(&self) -> bool {
        !self
            .accumulation
            .steps()
            .iter()
            .any(|step| matches!(step, ProgramStep::Gate(Gate::Lut { .. })))
    }
}

pub fn validate_profile_program(
    profile: SparseLwrPrfProfile,
    program: &SparseLwrPrfProgram,
) -> Result<(), PrfError> {
    profile.validate()?;
    program.profile.validate()?;
    if program.profile != profile || program.readout.layout != profile.layout {
        return Err(PrfError::InvalidSparseLwrBlock);
    }
    Ok(())
}

/// Fixed public label vectors and encrypted one-hot selectors for one PBC
/// block. Label family members are [output][active cell], with padding absent.
pub struct PrfSelectionInputs<S: ExponentLUTFixedRhsScheme> {
    pub selectors: S::FixedRhs,
    pub labels: Family<Mat>,
    pub active: FamilyRange,
    pub id: [u8; 32],
}

impl<S: ExponentLUTFixedRhsScheme> PrfSelectionInputs<S> {
    /// Bind one block of a label-major public family to its selector range.
    /// `labels` contains [output][selector capacity], whereas the prepared
    /// selection retains only [output][active member]. The construction
    /// contract requires complete rows and a nonempty valid active range.
    /// BFV callers supply labels in this block's RHS ring; BGG may share one
    /// global label family and selector family across blocks.
    pub fn new(
        selectors: S::FixedRhs,
        labels: &Family<Mat>,
        active: FamilyRange,
        id: [u8; 32],
    ) -> Result<Self, mxx_dsl::DslError> {
        let capacity = Int::evaluate(active.capacity().clone());
        let count = Int::evaluate(active.count().clone());
        let outputs = Int::evaluate(labels.count().clone()).div(capacity.clone());
        let labels = mxx_dsl::parallel(outputs.mul(count.clone()).expression()?, |index| {
            let output = index.clone().div(count.clone());
            let member = index.rem(count.clone()).add(Int::evaluate(active.start().clone()));
            Ok(labels.at(output.mul(capacity.clone()).add(member)))
        })?;
        Ok(Self { selectors, labels, active, id })
    }
}

/// Preprocessing output. The application connects the selection steps; the
/// scheme implements each cryptographic step. Output axes remain families.
pub struct PreparedSparseLwrPrf<S: ExponentLUTFixedRhsScheme> {
    program: ExponentLutProgram,
    pub(crate) steps: Vec<S::PreparedFixedProduct>,
}

impl<S: ExponentLUTFixedRhsScheme> Clone for PreparedSparseLwrPrf<S> {
    fn clone(&self) -> Self {
        Self { program: self.program.clone(), steps: self.steps.clone() }
    }
}

/// Ordered fixed-product artifacts authenticated by the preprocessing contract.
/// Backend bindings appear once; step descriptors contain only operation data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreparedSparseLwrPrfDescriptor<B, D> {
    program: ExponentLutProgramId,
    binding: B,
    prefix: String,
    pub(crate) steps: Vec<D>,
}

impl<B, D> PreparedSparseLwrPrfDescriptor<B, D> {
    /// Canonical step order determines names; descriptors cannot rename or
    /// duplicate one step independently of its position in the Program.
    pub(crate) fn artifacts(&self) -> impl Iterator<Item = (String, &D)> {
        self.steps
            .iter()
            .enumerate()
            .map(|(index, schema)| (format!("{}.{index}", self.prefix), schema))
    }
}

impl<B: Serialize, D: Serialize> PreparedSparseLwrPrfDescriptor<B, D> {
    pub fn identity(&self) -> Result<[u8; 32], PrfError> {
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        hash.update(b"exponent-lut-prepared-prf");
        hash.update(serde_json::to_vec(self).map_err(|e| PrfError::Artifact(e.to_string()))?);
        Ok(hash.finalize().into())
    }
}

impl<S: ExponentLUTFixedRhsScheme> PreparedSparseLwrPrf<S> {
    pub fn export(
        &self,
        scheme: &S,
        mut graph: mxx_dsl::DslContext,
        prefix: &str,
    ) -> Result<
        (
            mxx_dsl::DslContext,
            PreparedSparseLwrPrfDescriptor<
                S::FixedProductBinding,
                <S::PreparedFixedProduct as GraphValue>::Schema,
            >,
        ),
        PrfError,
    > {
        let descriptor = PreparedSparseLwrPrfDescriptor {
            program: self.program.id(),
            binding: scheme.fixed_product_binding(),
            prefix: prefix.into(),
            steps: self.steps.iter().map(GraphValue::schema).collect(),
        };
        for ((name, _), step) in descriptor.artifacts().zip(&self.steps) {
            graph = graph
                .public_output(name, step.clone())
                .map_err(|error| PrfError::Artifact(error.to_string()))?;
        }
        Ok((graph, descriptor))
    }

    pub fn import(
        scheme: &S,
        program: SparseLwrPrfProgram,
        descriptor: &PreparedSparseLwrPrfDescriptor<
            S::FixedProductBinding,
            <S::PreparedFixedProduct as GraphValue>::Schema,
        >,
        production: mxx_ir_core::artifact::ProductionId,
        expected_identity: [u8; 32],
    ) -> Result<Self, PrfError> {
        let accumulation = program.accumulation;
        if descriptor.identity()? != expected_identity ||
            descriptor.program != accumulation.id() ||
            descriptor.binding != scheme.fixed_product_binding() ||
            descriptor.steps.len() != program.block_count
        {
            return Err(PrfError::Artifact("prepared PRF identity or schedule mismatch".into()));
        }
        let steps = descriptor
            .artifacts()
            .map(|(name, schema)| {
                schema.artifact_input(
                    production.clone(),
                    name,
                    mxx_ir_core::artifact::ArtifactConfidentiality::Public,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| PrfError::Artifact(error.to_string()))?;
        Ok(Self { program: accumulation, steps })
    }

    /// Return the final public key to preprocessing and retain only operation
    /// caches for online evaluation. The surrounding readout/setup owns any
    /// persisted public output; importing the PRF does not duplicate it.
    pub fn new(
        scheme: &S,
        program: SparseLwrPrfProgram,
        initial: crate::scheme::PublicProgramValue<S>,
        blocks: &[PrfSelectionInputs<S>],
    ) -> Result<(crate::scheme::PublicProgramValue<S>, Self), ProgramLowerError<S::Error>> {
        if blocks.len() != program.block_count {
            return Err(ProgramValidationError::WidthMismatch.into());
        }
        let accumulation = program.accumulation;
        let mut public = initial;
        let mut steps = Vec::with_capacity(blocks.len());
        for (step, gate) in accumulation.steps().iter().enumerate() {
            let ProgramStep::Gate(Gate::MulPrivate { rhs, .. }) = gate else {
                unreachable!("fixed accumulation is a direct-product chain")
            };
            let block = &blocks[rhs.index()];
            let operation = || {
                let rhs = scheme.combine_fixed_rhs(
                    &block.selectors,
                    &block.labels,
                    &block.active,
                    block.id,
                )?;
                scheme.prepare_fixed_product(&public, &rhs, block.id)
            };
            let (output, prepared) = operation().map_err(|source| {
                ProgramLowerError::Operation { gate: ProgramGateInstance::Gate { step }, source }
            })?;
            public = output;
            steps.push(prepared);
        }
        Ok((public, Self { program: accumulation, steps }))
    }

    /// The exact Program executed by public preparation and online evaluation.
    /// Noise bindings and gate identities refer to this direct-product chain.
    pub fn program(&self) -> &ExponentLutProgram {
        &self.program
    }
    pub fn steps(&self) -> &[S::PreparedFixedProduct] {
        &self.steps
    }

    /// Evaluate fixed-product selections using only cached public preparation.
    pub fn evaluate(
        &self,
        scheme: &S,
        mut input: crate::scheme::EncodingProgramValue<S>,
    ) -> Result<crate::scheme::EncodingProgramValue<S>, ProgramLowerError<S::Error>> {
        for (step, gate) in self.program.steps().iter().enumerate() {
            let ProgramStep::Gate(Gate::MulPrivate { rhs, .. }) = gate else {
                unreachable!("fixed accumulation is a direct-product chain")
            };
            input = scheme.evaluate_fixed_product(&input, &self.steps[rhs.index()]).map_err(
                |source| ProgramLowerError::Operation {
                    gate: ProgramGateInstance::Gate { step },
                    source,
                },
            )?;
        }
        Ok(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    #[test]
    fn profile_derives_q_and_p_for_nonbinary_digit_bases() {
        let profile =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 5 }, 8).unwrap();
        assert_eq!(profile.native_width(), 32);
        assert_eq!(profile.q().unwrap(), 64);
        assert_eq!(profile.p().unwrap(), 16);
        let profile =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 6 }, 16).unwrap();
        assert_eq!(profile.q().unwrap(), 128);
        assert_eq!(profile.p().unwrap(), 32);
    }

    #[test]
    fn profile_rejects_nondividing_digit_base() {
        assert!(SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 5 }, 12).is_err());
    }

    #[test]
    fn canonical_sparse_lwr_table_has_full_q_complement_and_uniform_p_values() {
        for (ring_dimension_log2, stride_log2) in
            [(3_u32, 0_u32), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2)]
        {
            let width_log2 = ring_dimension_log2 - stride_log2;
            let n_prime = 1usize << width_log2;
            let p_prime = n_prime / 2;
            let profile =
                SparseLwrPrfProfile::new(ExpLayout { stride_log2, width_log2 }, p_prime).unwrap();
            let readout = profile.canonical_readout().unwrap();
            let expected = (0..n_prime).map(|k| p_prime * k / n_prime + 1).collect::<Vec<_>>();
            let coefficients = readout.canonical_coefficients().unwrap();
            let expected_coefficients =
                expected.iter().map(|value| num_bigint::BigInt::from(*value)).collect::<Vec<_>>();
            assert_eq!(coefficients, expected_coefficients.as_slice());
            for k in 0..n_prime {
                let complement = 1_i64 - expected[k] as i64;
                assert_eq!(expected[k] as i64 + complement, 1);
            }
            for value in 1..=p_prime {
                assert_eq!(expected.iter().filter(|entry| **entry == value).count(), 2);
            }
            assert_eq!(profile.q().unwrap(), 2 * n_prime);
        }
    }

    #[test]
    fn profile_deserialization_rechecks_derived_moduli() {
        let encoded = serde_json::json!({
            "layout": { "stride_log2": 1, "width_log2": 5 },
            "p_prime": 12,
        });
        assert!(serde_json::from_value::<SparseLwrPrfProfile>(encoded).is_err());
    }

    #[test]
    fn native_program_has_one_private_product_per_block_and_no_lut() {
        let profile =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 5 }, 8).unwrap();
        let program = SparseLwrPrfProgram::new(profile, 2, 3).unwrap();
        assert!(program.has_only_native_accumulation());
        assert_eq!(program.accumulation.rhs_inputs().len(), 3);
        assert_eq!(program.accumulation.steps().len(), 3);
        assert_eq!(program.accumulation.gates().len(), 3);
        assert!(
            program.accumulation.gates().iter().all(|gate| matches!(gate, Gate::MulPrivate { .. }))
        );
    }

    #[test]
    fn profile_program_mismatch_is_rejected() {
        let profile =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 5 }, 8).unwrap();
        let program = SparseLwrPrfProgram::new(profile, 2, 1).unwrap();
        let other =
            SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 4 }, 4).unwrap();
        assert!(validate_profile_program(other, &program).is_err());
    }
    #[test]
    fn tiny_native_reference_oracle_covers_every_z_mod_q() {
        let profile =
            SparseLwrPrfProfile::new(crate::layout::ExpLayout { stride_log2: 1, width_log2: 4 }, 2)
                .expect("n=32, Q=32, p'=2 profile");
        assert_eq!(profile.ring_dimension().unwrap(), 32);
        assert_eq!(profile.q().unwrap(), 32);
        assert_eq!(profile.p_prime, 2);
        assert_eq!(profile.layout.stride_log2, 1);
        let coefficients = profile
            .canonical_readout()
            .expect("canonical readout")
            .canonical_coefficients()
            .expect("constant readout table")
            .to_vec();
        assert_eq!(coefficients.len(), 16);

        // Readout represents f(z)-1/2 in doubled centered coordinates.
        // Native negacyclic wrap sends z+n' to the additive inverse, so both
        // halves of Z_Q are covered without an out-of-range table lookup.
        let doubled = (0..32)
            .map(|z| {
                let k = z % 16;
                let value = &coefficients[k] * BigInt::from(2_u8) - BigInt::from(1_u8);
                if z < 16 { value } else { -value }
            })
            .collect::<Vec<_>>();
        for z in 0..16 {
            let expected = BigInt::from(2 * (2 * z / 16 + 1) - 1);
            assert_eq!(doubled[z], expected);
            assert_eq!(doubled[z + 16], -expected);
        }
    }
}
