//! Value-form readout for native mod-Q exponent encodings.
//!
//! A [`ValueReadout`] is the boundary between the exponent-encoded accumulator
//! and its constant ring value.  The table is a scalar ring polynomial whose
//! coefficient at `k` is `f(k)`.  Readout constructs the public ring target
//!
//! ```text
//! T' = sum_k (2 f(k) - 1) * mono_m(-k)
//! ```
//!
//! as a DSL expression.  It deliberately does not lower an encoding, execute
//! a graph, or synthesize a preimage.  The latter is represented by explicit
//! normalization metadata so that preprocessing can provide the typed
//! `Preimage` required by the eventual evaluator.

use crate::{
    layout::{ExpLayout, LayoutError},
    noise_refresh::artifacts::ReadoutCacheImportError,
};
use mxx_bgg::{BggEncodingCompiler, BggEncodingWire, BggPublicKeyCompiler, BggPublicKeyWire};
use mxx_dsl::{Family, Int, Mat, Parallel, ParallelZipTuple, Preimage, Ring};
use mxx_ir_core::{IntExpr, ParamEnv, artifact::ArtifactType, types::ConcreteMatrixType};
use num_bigint::{BigInt, Sign};
use num_traits::{One, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub(crate) fn helper_sigma_for_flat_index(index: IntExpr, branches: Vec<IntExpr>) -> IntExpr {
    // Flattening is label-major: every label visits all helper sigmas.
    // The helper coordinate is a remainder, not an exact integer quotient.
    IntExpr::Select {
        selector: Box::new(IntExpr::Rem(
            Box::new(index),
            Box::new(IntExpr::constant(branches.len())),
        )),
        branches,
    }
}

#[cfg(test)]
#[path = "../../tests/readout_index_tests/test.rs"]
mod readout_index_tests;

/// Errors raised while validating or constructing a value readout.
#[derive(Debug, Eq, PartialEq, Error)]
pub enum ValueReadoutError {
    /// A manifest-backed cache member could not be imported.
    #[error(transparent)]
    Cache(#[from] ReadoutCacheImportError),
    /// The native layout is not valid for the requested concrete ring.
    #[error(transparent)]
    Layout(#[from] LayoutError),
    /// The table must be a scalar ring matrix, not a matrix of encodings.
    #[error("value-readout table must have scalar shape (1, 1)")]
    InvalidTableShape,
    /// An explicit value table must cover the complete native domain.
    #[error("value-readout table must contain exactly n' entries")]
    InvalidTableLength,
    /// The table and target ring have different modulus or ring dimension.
    #[error("value-readout table and target ring have different ring types")]
    RingTypeMismatch,
    /// A required ring parameter could not be evaluated at graph construction.
    #[error("value-readout ring parameters must be concrete")]
    NonConcreteRingParameter,
    /// The scalar normalization denominator overflowed the host index type.
    #[error("value-readout normalization denominator overflows usize")]
    NormalizationDenominatorOverflow,
    /// The normalization scalar has no inverse in the target ring.
    #[error("2*n' is not invertible in the target ring modulus")]
    NonInvertibleNormalization,
    /// A bounded preimage requirement must use a nontrivial gadget base.
    #[error("normalization gadget base must be greater than one")]
    InvalidGadgetBase,
    /// A bounded preimage requirement must have at least one digit.
    #[error("normalization gadget digit count must be positive")]
    InvalidDigitCount,
    /// The trace index set was not the complete native subgroup.
    #[error("value-readout trace must contain each odd native automorphism once")]
    InvalidTraceSet,
    /// A table declaration could not be serialized into its stable identity form.
    #[error("value-readout table declaration could not be canonicalized")]
    Canonicalization,
    /// A normalization preimage bound must be nonnegative.
    #[error("normalization preimage bound must be nonnegative")]
    InvalidPreimageBound,
    /// Normalization requires the canonical gadget declaration for its base and digits.
    #[error(
        "normalization preimage gadget must be a canonical gadget with the requested base and digits"
    )]
    InvalidNormalizationGadget,
    /// A cache preimage does not match the exact source shape, bound, or
    /// decomposition schema declared by the readout operation.
    #[error("readout preimage does not match its declared source shape, bound, or gadget schema")]
    InvalidPreimageSchema,
    /// The trace graph could not be assembled from the supplied typed artifacts.
    #[error(transparent)]
    Graph(#[from] crate::ExponentLutError),
}

/// Stable identity of one graph-bound value readout declaration.
///
/// The table's graph value identity is included so that two readouts with the
/// same layout but different table inputs cannot be silently paired.  This is
/// a declaration identity only; it is not an evaluator commitment or an
/// attestation of a runtime matrix value.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ValueReadoutIdentity([u8; 32]);

impl ValueReadoutIdentity {
    /// Constructs an identity from an already canonicalized declaration
    /// digest.  Artifact producers use this when importing a persisted
    /// readout declaration without retaining the full table graph.
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Returns the raw identity digest.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// The complete single-layer native trace plan.
///
/// `identity_sigma` is exactly the one identity branch.  Every other index is
/// a non-identity helper requirement; no sequential or split trace is encoded
/// by this type.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ValueReadoutTracePlan {
    automorphisms: Vec<usize>,
    identity_sigma: usize,
    non_identity_helpers: Vec<usize>,
}

/// Exact non-matrix schema attached to one cache-required readout preimage.
/// The matrix type is carried by [`Preimage`] itself; these fields bind the
/// gadget decomposition parameters and coefficient bound that produced it.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ReadoutPreimageSchema {
    pub gadget_base: BigInt,
    pub digit_count: usize,
    pub max_coefficient_bound: BigInt,
}

impl ReadoutPreimageSchema {
    pub fn new(
        gadget_base: impl Into<BigInt>,
        digit_count: usize,
        max_coefficient_bound: impl Into<BigInt>,
    ) -> Result<Self, ValueReadoutError> {
        let schema = Self {
            gadget_base: gadget_base.into(),
            digit_count,
            max_coefficient_bound: max_coefficient_bound.into(),
        };
        if schema.gadget_base <= BigInt::one() {
            return Err(ValueReadoutError::InvalidGadgetBase);
        }
        if schema.digit_count == 0 {
            return Err(ValueReadoutError::InvalidDigitCount);
        }
        if schema.max_coefficient_bound < BigInt::zero() {
            return Err(ValueReadoutError::InvalidPreimageBound);
        }
        Ok(schema)
    }
}

/// A non-identity value-readout branch with all deterministic right actions
/// already decomposed.  The two preimages are relation-bearing cache inputs;
/// no branch construction is allowed to call `Mat::decompose`.
#[derive(Clone)]
pub struct ValueReadoutHelper {
    layout: ExpLayout,
    sigma: usize,
    gadget_base: BigInt,
    digit_count: usize,
    max_coefficient_bound: BigInt,
    switch_decomposition: Preimage,
    public_decomposition: Preimage,
    mask: BggEncodingWire,
}

impl ValueReadoutHelper {
    pub fn new(
        layout: ExpLayout,
        sigma: usize,
        gadget_base: BigInt,
        digit_count: usize,
        max_coefficient_bound: BigInt,
        switch_decomposition: Preimage,
        public_decomposition: Preimage,
        mask: BggEncodingWire,
    ) -> Result<Self, ValueReadoutError> {
        if sigma == 0 || sigma % 2 == 0 || sigma == 1 {
            return Err(ValueReadoutError::InvalidTraceSet);
        }
        if gadget_base <= BigInt::one() {
            return Err(ValueReadoutError::InvalidGadgetBase);
        }
        if digit_count == 0 {
            return Err(ValueReadoutError::InvalidDigitCount);
        }
        if max_coefficient_bound < BigInt::zero() {
            return Err(ValueReadoutError::InvalidPreimageBound);
        }
        Ok(Self {
            layout,
            sigma,
            gadget_base,
            digit_count,
            max_coefficient_bound,
            switch_decomposition,
            public_decomposition,
            mask,
        })
    }

    pub fn sigma(&self) -> usize {
        self.sigma
    }

    pub fn layout(&self) -> ExpLayout {
        self.layout
    }
}

/// Public projection of one non-identity value-readout branch. It carries the
/// decomposition of the switched public target and the ordinary public mask;
/// the private switched ciphertext decomposition is not part of this path.
#[derive(Clone)]
pub struct ValueReadoutPublicHelper {
    layout: ExpLayout,
    sigma: usize,
    gadget_base: BigInt,
    digit_count: usize,
    max_coefficient_bound: BigInt,
    public_decomposition: Preimage,
    mask: Mat,
}

impl ValueReadoutPublicHelper {
    pub fn new(
        layout: ExpLayout,
        sigma: usize,
        gadget_base: BigInt,
        digit_count: usize,
        max_coefficient_bound: BigInt,
        public_decomposition: Preimage,
        mask: Mat,
    ) -> Result<Self, ValueReadoutError> {
        if sigma == 0 || sigma % 2 == 0 || sigma == 1 {
            return Err(ValueReadoutError::InvalidTraceSet);
        }
        if gadget_base <= BigInt::one() {
            return Err(ValueReadoutError::InvalidGadgetBase);
        }
        if digit_count == 0 {
            return Err(ValueReadoutError::InvalidDigitCount);
        }
        if max_coefficient_bound < BigInt::zero() {
            return Err(ValueReadoutError::InvalidPreimageBound);
        }
        Ok(Self {
            layout,
            sigma,
            gadget_base,
            digit_count,
            max_coefficient_bound,
            public_decomposition,
            mask,
        })
    }

    pub fn sigma(&self) -> usize {
        self.sigma
    }

    pub fn layout(&self) -> ExpLayout {
        self.layout
    }
}

impl ValueReadoutTracePlan {
    fn from_indices(automorphisms: Vec<usize>) -> Result<Self, ValueReadoutError> {
        let identity_count = automorphisms.iter().filter(|index| **index == 1).count();
        if identity_count != 1 || automorphisms.is_empty() {
            return Err(ValueReadoutError::InvalidTraceSet);
        }
        let mut distinct = automorphisms.clone();
        distinct.sort_unstable();
        distinct.dedup();
        if distinct != automorphisms ||
            automorphisms.iter().any(|index| *index == 0 || index % 2 == 0)
        {
            return Err(ValueReadoutError::InvalidTraceSet);
        }
        let non_identity_helpers =
            automorphisms.iter().copied().filter(|index| *index != 1).collect();
        Ok(Self { automorphisms, identity_sigma: 1, non_identity_helpers })
    }

    /// Returns all trace automorphisms, including identity exactly once.
    pub fn automorphisms(&self) -> &[usize] {
        &self.automorphisms
    }

    /// Returns the identity automorphism (`sigma = 1`).
    pub const fn identity_sigma(&self) -> usize {
        self.identity_sigma
    }

    /// Returns the non-identity helper automorphisms.
    pub fn non_identity_helpers(&self) -> &[usize] {
        &self.non_identity_helpers
    }

    /// Returns the number of trace terms.
    pub fn term_count(&self) -> usize {
        self.automorphisms.len()
    }

    /// Returns the number of plain non-identity helper decompositions.
    pub fn helper_count(&self) -> usize {
        self.non_identity_helpers.len()
    }
}

/// Descriptor for the concrete matrix target used by value-readout normalization.
///
/// The actual target is `(numerator / denominator) * G`, where `G` is supplied
/// by the BGG setup.  Keeping the gadget out of this descriptor prevents this
/// planning layer from pretending to have sampled or executed a preimage.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NormalizationTargetDescriptor {
    /// Exact concrete matrix type of the gadget target.
    pub matrix: ConcreteMatrixType,
    /// Layout from which `n'` and the native trace denominator are derived.
    pub layout: ExpLayout,
    /// Numerator of the normalization scalar; currently one by definition.
    pub numerator: BigInt,
    /// Denominator `2*n'` in the full ring modulus.
    pub denominator: usize,
}

/// Explicit bounded-preimage requirement for normalization.
///
/// This is metadata for a preprocessing artifact request.  It is not a
/// `Preimage`, and construction does not run the sampler or evaluator.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NormalizationPreimageRequirement {
    pub target: NormalizationTargetDescriptor,
    /// This is a relation-bearing bounded artifact, not a sampled witness.
    pub artifact_type: ArtifactType,
    /// The target is `(2*n')^-1 G` for this native layout.
    pub role: NormalizationPreimageRole,
    pub max_coefficient_bound: BigInt,
    pub gadget_base: BigInt,
    pub digit_count: usize,
    /// Stable association to the exact readout declaration.
    pub readout_identity: ValueReadoutIdentity,
}

/// Semantic role of the typed normalization preimage artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum NormalizationPreimageRole {
    InverseNativeTraceGadget,
}

/// Native exponent value-form readout declaration.
///
/// `table` is a scalar ring polynomial with coefficient `k` equal to `f(k)`.
/// Ring-bound declarations check their ring dimension and modulus when a
/// concrete target ring is supplied; canonical native PRF declarations keep
/// only the coefficients and bind them to the target ring during lowering.
#[derive(Clone)]
pub struct ValueReadout {
    pub layout: ExpLayout,
    table: Mat,
    /// Whether the table was bound to a concrete coefficient ring.  Native
    /// PRF declarations are intentionally ring-independent and bind their
    /// canonical coefficients when lowered against the compiler ring.
    ring_bound: bool,
    /// Canonical recursive IR declaration of `table`; it contains the table
    /// values, matrix type, and all graph arguments without allocation identity.
    canonical_table: Vec<u8>,
    canonical_coefficients: Option<Vec<BigInt>>,
}

impl ValueReadout {
    /// Executes readout for every online PRF output without nesting a helper
    /// loop inside a label loop. The label/helper Cartesian product is one
    /// flat structural family, then each consecutive helper segment is
    /// reduced back to one output per label.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_batch_with_preprocessed_family_artifacts(
        &self,
        compiler: &BggEncodingCompiler,
        inputs: Family<Mat>,
        label_count: usize,
        target_decompositions: Family<Preimage>,
        target_schema: ReadoutPreimageSchema,
        helper_sigmas: &[usize],
        switch_decompositions: Family<Preimage>,
        public_decompositions: Family<Preimage>,
        masks: Family<Mat>,
        normalization_decomposition: Preimage,
        normalization_schema: ReadoutPreimageSchema,
    ) -> Result<Family<Mat>, ValueReadoutError> {
        if label_count == 0 || *inputs.count() != IntExpr::constant(label_count) {
            return Err(ValueReadoutError::InvalidTraceSet);
        }
        let input_type = inputs.element_type();
        let ring = Ring::new(input_type.modulus.clone(), input_type.ring_dimension.clone());
        self.validate(&ring)?;
        let ring_dimension = concrete_ring_dimension(&ring)?;
        let plan = self.trace_plan(ring_dimension)?;
        let helper_count = helper_sigmas.len();
        if helper_count == 0 ||
            helper_sigmas != plan.non_identity_helpers() ||
            switch_decompositions.count() != masks.count() ||
            *switch_decompositions.count() != IntExpr::constant(helper_count) ||
            *public_decompositions.count() !=
                IntExpr::constant(
                    label_count
                        .checked_mul(helper_count)
                        .ok_or(ValueReadoutError::InvalidTraceSet)?,
                )
        {
            return Err(ValueReadoutError::InvalidTraceSet);
        }
        if *target_decompositions.count() != IntExpr::constant(label_count) {
            return Err(ValueReadoutError::InvalidTraceSet);
        }
        validate_preimage_ring(&normalization_decomposition, &ring)?;
        if target_schema.gadget_base != normalization_schema.gadget_base ||
            target_schema.digit_count != normalization_schema.digit_count
        {
            return Err(ValueReadoutError::InvalidPreimageSchema);
        }
        validate_encoding_action_preimage_schema_type(
            &target_decompositions.get_static(0),
            input_type,
            &target_schema.gadget_base,
            target_schema.digit_count,
            Some(&target_schema.max_coefficient_bound),
        )?;
        validate_encoding_action_preimage_family_type(
            switch_decompositions.element_type(),
            input_type,
            target_schema.digit_count,
        )?;
        validate_encoding_action_preimage_family_type(
            public_decompositions.element_type(),
            input_type,
            target_schema.digit_count,
        )?;
        validate_family_output_type(masks.element_type(), input_type, &ring)?;

        let target = self.target(&ring)?;
        let target_for_labels = target.clone();
        let label_compiler = compiler.clone();
        let transformed = (inputs, target_decompositions)
            .parallel_zip_tuple(move |_, (vector, target_decomposition)| {
                label_compiler
                    .large_scalar_mul(
                        &BggEncodingWire { vector, plaintext: None },
                        &target_for_labels,
                        target_decomposition,
                    )
                    .vector
            })
            .map_err(|error| {
                ValueReadoutError::Graph(crate::ExponentLutError::OnlinePrfStage {
                    stage: "transform readout input family",
                    detail: error.to_string(),
                })
            })?;
        let flat_count =
            label_count.checked_mul(helper_count).ok_or(ValueReadoutError::InvalidTraceSet)?;
        let helper_indices = Parallel::range(flat_count)
            .map_values(|index| index.as_int().rem(Int::constant(helper_count)))
            .map_err(|error| {
                ValueReadoutError::Graph(crate::ExponentLutError::OnlinePrfStage {
                    stage: "index aligned readout masks",
                    detail: error.to_string(),
                })
            })?;
        let flat_masks = masks.parallel_gather(helper_indices.clone()).map_err(|error| {
            ValueReadoutError::Graph(crate::ExponentLutError::OnlinePrfStage {
                stage: "gather aligned readout masks",
                detail: error.to_string(),
            })
        })?;
        let aligned_public = (flat_masks, public_decompositions)
            .parallel_zip_tuple(|_, (mask, public)| public.mul_small_rhs(mask))
            .map_err(|error| {
                ValueReadoutError::Graph(crate::ExponentLutError::OnlinePrfStage {
                    stage: "align readout public helper family",
                    detail: error.to_string(),
                })
            })?;

        let label_indices = Parallel::range(flat_count)
            .map_values(|index| index.as_int().div(Int::constant(helper_count)))
            .map_err(|error| {
                ValueReadoutError::Graph(crate::ExponentLutError::OnlinePrfStage {
                    stage: "index readout label family",
                    detail: error.to_string(),
                })
            })?;
        let flat_transformed =
            transformed.clone().parallel_gather(label_indices).map_err(|error| {
                ValueReadoutError::Graph(crate::ExponentLutError::OnlinePrfStage {
                    stage: "gather readout labels",
                    detail: error.to_string(),
                })
            })?;
        let flat_aligned = aligned_public;
        let flat_switches = switch_decompositions
            .parallel_gather(
                Parallel::range(flat_count)
                    .map_values(|index| index.as_int().rem(Int::constant(helper_count)))
                    .map_err(|error| {
                        ValueReadoutError::Graph(crate::ExponentLutError::OnlinePrfStage {
                            stage: "index readout switches",
                            detail: error.to_string(),
                        })
                    })?,
            )
            .map_err(|error| {
                ValueReadoutError::Graph(crate::ExponentLutError::OnlinePrfStage {
                    stage: "reindex readout switches",
                    detail: error.to_string(),
                })
            })?;
        let sigma_branches =
            helper_sigmas.iter().copied().map(IntExpr::constant).collect::<Vec<_>>();
        let helper_vectors = (flat_transformed, flat_aligned, flat_switches)
            .parallel_zip_tuple(move |index, (transformed, aligned, switch)| {
                let sigma = helper_sigma_for_flat_index(index.expression(), sigma_branches.clone());
                aligned + switch.mul_small_rhs(transformed.ring_automorphism(sigma))
            })
            .map_err(|error| {
                ValueReadoutError::Graph(crate::ExponentLutError::OnlinePrfStage {
                    stage: "construct flat readout helper family",
                    detail: error.to_string(),
                })
            })?;
        let helper_sums = reduce_matrix_family_segments(helper_vectors, label_count, helper_count)?;
        let traced = (transformed, helper_sums)
            .parallel_zip_tuple(|_, (identity, helpers)| identity + helpers)
            .map_err(|error| {
                ValueReadoutError::Graph(crate::ExponentLutError::OnlinePrfStage {
                    stage: "add readout identity branches",
                    detail: error.to_string(),
                })
            })?;

        let denominator = normalization_denominator(&self.layout, ring_dimension)?;
        let modulus = ring
            .matrix_type((1, 1))
            .modulus
            .evaluate(&ParamEnv::default())
            .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?;
        let inverse = modular_inverse(&(BigInt::from(denominator) % &modulus), &modulus)
            .ok_or(ValueReadoutError::NonInvertibleNormalization)?;
        validate_encoding_action_preimage_schema_type(
            &normalization_decomposition,
            traced.element_type(),
            &normalization_schema.gadget_base,
            normalization_schema.digit_count,
            Some(&normalization_schema.max_coefficient_bound),
        )?;
        let normalization_scalar = ring.polynomial([inverse.into()]);
        let normalization_compiler = compiler.clone();
        traced
            .parallel_map_values(move |_, vector| {
                normalization_compiler
                    .large_scalar_mul(
                        &BggEncodingWire { vector, plaintext: None },
                        &normalization_scalar,
                        normalization_decomposition.clone(),
                    )
                    .vector
            })
            .map_err(|error| {
                ValueReadoutError::Graph(crate::ExponentLutError::OnlinePrfStage {
                    stage: "normalize readout output family",
                    detail: error.to_string(),
                })
            })
    }

    /// Creates a readout declaration after checking the table's structural
    /// scalar shape. Ring-dependent checks are deferred until a concrete ring
    /// is available.
    pub fn new(layout: ExpLayout, table: Mat) -> Result<Self, ValueReadoutError> {
        if !is_scalar_shape(table.matrix_type()) {
            return Err(ValueReadoutError::InvalidTableShape);
        }
        let canonical_table = canonical_table_declaration(&table)?;
        let canonical_coefficients = canonical_polynomial_coefficients(&table);
        Ok(Self { layout, table, ring_bound: true, canonical_table, canonical_coefficients })
    }

    /// Creates a readout from explicit table values and a concrete ring.
    /// Values are placed in the first `n'` polynomial coefficients in canonical
    /// order, which is the table representation consumed by [`Self::target`].
    pub fn from_values(
        layout: ExpLayout,
        ring: &Ring,
        values: Vec<usize>,
    ) -> Result<Self, ValueReadoutError> {
        let ring_dimension = concrete_ring_dimension(ring)?;
        layout.validate(ring_dimension)?;
        let n_prime = native_width(&layout, ring_dimension)?;
        if values.len() != n_prime {
            return Err(ValueReadoutError::InvalidTableLength);
        }
        let table = ring.polynomial(values.into_iter().map(|value| value.into()));
        Self::new(layout, table)
    }

    /// Creates a canonical, ring-independent value table from its complete
    /// native-domain coefficients.  The declaration is later evaluated in
    /// the concrete compiler ring supplied to [`Self::target`].
    pub fn from_canonical_values(
        layout: ExpLayout,
        values: Vec<usize>,
    ) -> Result<Self, ValueReadoutError> {
        let expected = layout.width();
        if values.len() != expected || expected == 0 {
            return Err(ValueReadoutError::InvalidTableLength);
        }
        let ring_dimension =
            layout.stride().checked_mul(expected).ok_or(ValueReadoutError::InvalidTableLength)?;
        layout.validate(ring_dimension)?;
        let coefficients = values.iter().cloned().map(BigInt::from).collect::<Vec<_>>();
        let table =
            Ring::new(1, ring_dimension).polynomial(values.into_iter().map(|value| value.into()));
        let canonical_table =
            serde_json::to_vec(&coefficients).map_err(|_| ValueReadoutError::Canonicalization)?;
        Ok(Self {
            layout,
            table,
            ring_bound: false,
            canonical_table,
            canonical_coefficients: Some(coefficients),
        })
    }

    /// Validates this readout against a concrete target ring.
    pub fn validate(&self, ring: &Ring) -> Result<(), ValueReadoutError> {
        if !is_scalar_shape(self.table.matrix_type()) {
            return Err(ValueReadoutError::InvalidTableShape);
        }
        let ring_type = ring.matrix_type((1, 1));
        if self.ring_bound &&
            (self.table.matrix_type().modulus != ring_type.modulus ||
                self.table.matrix_type().ring_dimension != ring_type.ring_dimension)
        {
            return Err(ValueReadoutError::RingTypeMismatch);
        }
        let ring_dimension = concrete_ring_dimension(ring)?;
        self.layout.validate(ring_dimension)?;
        // This also checks the host-size arithmetic used by all readout terms.
        let _ = native_width(&self.layout, ring_dimension)?;
        Ok(())
    }

    /// Returns the stable declaration identity of this readout.
    pub fn identity(&self) -> ValueReadoutIdentity {
        let mut digest = Sha256::new();
        digest.update(b"mxx-exponent-lut/value-readout/canonical-v2");
        digest.update(serde_json::to_vec(&self.layout).expect("layout is serializable"));
        if self.ring_bound {
            digest.update(
                serde_json::to_vec(self.table.matrix_type()).expect("matrix type is serializable"),
            );
        } else {
            digest.update(b"ring-independent");
        }
        digest.update(&self.canonical_table);
        ValueReadoutIdentity(digest.finalize().into())
    }

    /// Returns coefficients when this declaration is a constant polynomial.
    /// This is useful for audit tooling; the canonical IR remains the identity
    /// source for symbolic table declarations.
    pub fn canonical_coefficients(&self) -> Option<&[BigInt]> {
        self.canonical_coefficients.as_deref()
    }

    /// Convolution gains of the declared integer lift of T', separate from
    /// the decomposition of T'G (not necessarily centered modulo the ring q).
    /// Native signed-monomial accumulator plaintexts only rotate/sign these
    /// coefficients, preserving both norms used for the switch helper error.
    pub(crate) fn target_plaintext_gain(
        &self,
        ring_dimension: usize,
    ) -> Result<crate::noise::RightActionNoiseGain, ValueReadoutError> {
        self.layout.validate(ring_dimension)?;
        let width = native_width(&self.layout, ring_dimension)?;
        let coefficients =
            self.canonical_coefficients().ok_or(ValueReadoutError::Canonicalization)?;
        // Reject polynomial declarations requiring coefficient aliasing. The
        // target reads the first width coefficients, with implicit zero padding.
        if coefficients.len() > ring_dimension {
            return Err(ValueReadoutError::InvalidTableLength);
        }
        // mono(-k) occupies a distinct coefficient for every native k;
        // negacyclic wrapping changes its sign, not its magnitude.
        Ok(crate::noise::RightActionNoiseGain::from_polynomial_coefficients((0..width).map(|k| {
            coefficients.get(k).cloned().unwrap_or_default() * BigInt::from(2u8) - BigInt::one()
        })))
    }

    /// Returns the immutable table declaration.
    pub fn table(&self) -> &Mat {
        &self.table
    }

    /// Constructs `T'` as a pure DSL graph expression.
    ///
    /// Each table coefficient is lifted to a constant ring polynomial after
    /// applying `2*f(k)-1`; the monomial is constructed by `ExpLayout::mono`,
    /// so native negacyclic wrapping is preserved exactly.
    pub fn target(&self, ring: &Ring) -> Result<Mat, ValueReadoutError> {
        self.validate(ring)?;
        let ring_dimension = concrete_ring_dimension(ring)?;
        let n_prime = native_width(&self.layout, ring_dimension)?;
        let scalar_type = ring.matrix_type((1, 1));
        let table = if self.ring_bound {
            self.table.clone()
        } else {
            ring.polynomial(
                self.canonical_coefficients
                    .as_ref()
                    .ok_or(ValueReadoutError::Canonicalization)?
                    .iter()
                    .cloned()
                    .map(Into::into),
            )
        };
        let mut target = ring.zero((1, 1));
        for k in 0..n_prime {
            let coefficient = table
                .clone()
                .extract_coefficient(k)
                .mul(Int::constant(2))
                .sub(Int::constant(1))
                .lift_to_constant_polynomial(scalar_type.clone());
            let monomial = self.layout.mono(ring, &-BigInt::from(k))?;
            target = target + coefficient * monomial;
        }
        Ok(target)
    }

    /// Returns the complete native single-layer trace plan for this ring.
    pub fn trace_plan(
        &self,
        ring_dimension: usize,
    ) -> Result<ValueReadoutTracePlan, ValueReadoutError> {
        self.layout.validate(ring_dimension)?;
        let full_layout = full_trace_layout(&self.layout, ring_dimension)?;
        ValueReadoutTracePlan::from_indices(full_layout.automorphism_indices(ring_dimension)?)
    }

    /// Executes value readout using only explicitly supplied decomposition
    /// artifacts.  The target multiplication is followed by one family trace
    /// containing the identity branch exactly once and one helper branch for
    /// each non-identity automorphism, then by the explicit normalization
    /// preimage.  In particular, this path contains no sampler or fallback
    /// decomposition operation.
    pub fn execute_with_preprocessed_artifacts(
        &self,
        compiler: &BggEncodingCompiler,
        input: &BggEncodingWire,
        target_decomposition: Preimage,
        target_schema: ReadoutPreimageSchema,
        helpers: &[ValueReadoutHelper],
        normalization_decomposition: Preimage,
        normalization_schema: ReadoutPreimageSchema,
    ) -> Result<BggEncodingWire, ValueReadoutError> {
        let matrix_type = input.vector.matrix_type();
        let modulus = matrix_type.modulus.clone();
        let ring = Ring::new(modulus, matrix_type.ring_dimension.clone());
        self.validate(&ring)?;
        let ring_dimension = concrete_ring_dimension(&ring)?;
        let plan = self.trace_plan(ring_dimension)?;
        if helpers.len() != plan.helper_count() ||
            helpers.iter().map(ValueReadoutHelper::sigma).collect::<Vec<_>>() !=
                plan.non_identity_helpers
        {
            return Err(ValueReadoutError::InvalidTraceSet);
        }
        if helpers.iter().any(|helper| helper.layout != self.layout) {
            return Err(ValueReadoutError::InvalidTraceSet);
        }
        validate_preimage_ring(&target_decomposition, &ring)?;
        validate_preimage_ring(&normalization_decomposition, &ring)?;
        let gadget_base = target_schema.gadget_base.clone();
        let digit_count = target_schema.digit_count;
        if target_schema.gadget_base != gadget_base ||
            target_schema.digit_count != digit_count ||
            normalization_schema.gadget_base != gadget_base ||
            normalization_schema.digit_count != digit_count
        {
            return Err(ValueReadoutError::InvalidPreimageSchema);
        }
        validate_encoding_action_preimage_schema(
            &target_decomposition,
            &input.vector,
            &target_schema.gadget_base,
            target_schema.digit_count,
            Some(&target_schema.max_coefficient_bound),
        )?;
        for helper in helpers {
            if helper.gadget_base != gadget_base || helper.digit_count != digit_count {
                return Err(ValueReadoutError::InvalidPreimageSchema);
            }
            validate_encoding_action_preimage_schema(
                &helper.switch_decomposition,
                &input.vector,
                &gadget_base,
                digit_count,
                Some(&helper.max_coefficient_bound),
            )?;
            validate_encoding_action_preimage_schema(
                &helper.public_decomposition,
                &input.vector,
                &gadget_base,
                digit_count,
                Some(&helper.max_coefficient_bound),
            )?;
            validate_matrix_ring(&helper.mask.vector, &ring)?;
            validate_matrix_ring(&helper.mask.vector, &ring)?;
            validate_output_shape(&helper.mask.vector, &input.vector, &ring)?;
            if helper.mask.plaintext.is_some() {
                return Err(ValueReadoutError::InvalidTraceSet);
            }
        }
        // Derive public matrices online from the input relation and typed
        // preimages.  Public output matrices are not preprocessing artifacts.
        let target = self.target(&ring)?;
        let transformed = compiler.large_scalar_mul(input, &target, target_decomposition);
        let mut vectors = Vec::with_capacity(plan.term_count());
        let mut plaintext = transformed.plaintext.clone().map(|value| vec![value]);
        vectors.push(transformed.vector.clone());
        for helper in helpers {
            let raw_vector = transformed.vector.clone().ring_automorphism(helper.sigma);
            let switched_vector = helper.switch_decomposition.clone().mul_small_rhs(raw_vector);
            vectors.push(
                helper.public_decomposition.clone().mul_small_rhs(helper.mask.vector.clone()) +
                    switched_vector,
            );
            if let Some(values) = plaintext.as_mut() {
                values.push(
                    transformed.plaintext.as_ref().unwrap().clone().ring_automorphism(helper.sigma),
                );
            }
        }
        let traced = BggEncodingWire {
            vector: crate::encoding::balanced_sum_family(
                Family::pack(vectors).map_err(|_| crate::ExponentLutError::InvalidLut)?,
            )?,
            plaintext: plaintext.map(|values| {
                crate::encoding::balanced_sum_family(
                    Family::pack(values).expect("validated trace plaintext family"),
                )
                .expect("validated trace plaintext sum")
            }),
        };
        let denominator = normalization_denominator(&self.layout, ring_dimension)?;
        let modulus = ring
            .matrix_type((1, 1))
            .modulus
            .evaluate(&ParamEnv::default())
            .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?;
        let inverse = modular_inverse(&(BigInt::from(denominator) % &modulus), &modulus)
            .ok_or(ValueReadoutError::NonInvertibleNormalization)?;
        validate_encoding_action_preimage_schema(
            &normalization_decomposition,
            &traced.vector,
            &normalization_schema.gadget_base,
            normalization_schema.digit_count,
            Some(&normalization_schema.max_coefficient_bound),
        )?;
        Ok(compiler.large_scalar_mul(
            &traced,
            &ring.polynomial([inverse.into()]),
            normalization_decomposition,
        ))
    }

    /// Executes the public-key projection of value readout with the same
    /// single-layer trace and explicit target/normalization preimages as the
    /// encoding path. Cache-time output-key attachment is kept separate from
    /// this algebraic projection.
    pub fn project_public_with_preprocessed_artifacts(
        &self,
        compiler: &BggPublicKeyCompiler,
        input: &BggPublicKeyWire,
        target_decomposition: Preimage,
        target_schema: ReadoutPreimageSchema,
        helpers: &[ValueReadoutPublicHelper],
        normalization_decomposition: Preimage,
        normalization_schema: ReadoutPreimageSchema,
    ) -> Result<Mat, ValueReadoutError> {
        let ring = compiler.ring.clone();
        self.validate(&ring)?;
        let ring_dimension = concrete_ring_dimension(&ring)?;
        let plan = self.trace_plan(ring_dimension)?;
        if helpers.len() != plan.helper_count() ||
            helpers.iter().map(ValueReadoutPublicHelper::sigma).collect::<Vec<_>>() !=
                plan.non_identity_helpers
        {
            return Err(ValueReadoutError::InvalidTraceSet);
        }
        if helpers.iter().any(|helper| helper.layout != self.layout) {
            return Err(ValueReadoutError::InvalidTraceSet);
        }
        validate_preimage_ring(&target_decomposition, &ring)?;
        validate_preimage_ring(&normalization_decomposition, &ring)?;
        let (gadget_base, digit_count) = public_bgg_schema(compiler)?;
        if target_schema.gadget_base != gadget_base ||
            target_schema.digit_count != digit_count ||
            normalization_schema.gadget_base != gadget_base ||
            normalization_schema.digit_count != digit_count
        {
            return Err(ValueReadoutError::InvalidPreimageSchema);
        }
        validate_preimage_schema(
            &target_decomposition,
            &input.matrix,
            &target_schema.gadget_base,
            target_schema.digit_count,
            Some(&target_schema.max_coefficient_bound),
        )?;
        for helper in helpers {
            if helper.gadget_base != gadget_base || helper.digit_count != digit_count {
                return Err(ValueReadoutError::InvalidPreimageSchema);
            }
            validate_preimage_schema(
                &helper.public_decomposition,
                &input.matrix,
                &gadget_base,
                digit_count,
                Some(&helper.max_coefficient_bound),
            )?;
            validate_output_shape(&helper.mask, &input.matrix, &ring)?;
        }
        // Public projection is the preprocessing graph itself: consume the
        // supplied target decomposition and retain the resulting matrix
        // expression. Cache-time output-key attachment belongs to the
        // encoding path and must not disconnect this public algebra.
        let transformed =
            compiler.large_scalar_mul_with_decomposition(input, target_decomposition).matrix;
        let mut branches = vec![transformed.clone()];
        for helper in helpers {
            // Public projection is exactly the public component of the BGG
            // helper relation: A_mask * D(A_switched). The switched vector
            // term is private ciphertext data and has no public matrix
            // counterpart, so it must not be added here.
            let aligned = helper.public_decomposition.clone().mul_small_rhs(helper.mask.clone());
            branches.push(aligned);
        }
        let traced = crate::encoding::balanced_sum_family(
            Family::pack(branches).map_err(|_| crate::ExponentLutError::InvalidLut)?,
        )?;
        let denominator = normalization_denominator(&self.layout, ring_dimension)?;
        let modulus = ring
            .matrix_type((1, 1))
            .modulus
            .evaluate(&ParamEnv::default())
            .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?;
        let _inverse = modular_inverse(&(BigInt::from(denominator) % &modulus), &modulus)
            .ok_or(ValueReadoutError::NonInvertibleNormalization)?;
        validate_preimage_schema(
            &normalization_decomposition,
            &traced,
            &normalization_schema.gadget_base,
            normalization_schema.digit_count,
            Some(&normalization_schema.max_coefficient_bound),
        )?;
        let normalized = compiler.large_scalar_mul_with_decomposition(
            &BggPublicKeyWire { matrix: traced, reveal_plaintext: false },
            normalization_decomposition,
        );
        Ok(normalized.matrix)
    }

    /// Public-key projection for an output-indexed PRF batch. The public
    /// helper decomposition family is flattened as `[output][helper]`, while
    /// setup-fixed masks and normalization remain shared.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn project_public_batch_with_preprocessed_family_artifacts(
        &self,
        compiler: &BggPublicKeyCompiler,
        inputs: Family<Mat>,
        output_count: usize,
        target_decompositions: Family<Preimage>,
        target_schema: ReadoutPreimageSchema,
        helper_sigmas: &[usize],
        public_decompositions: Family<Preimage>,
        masks: Family<Mat>,
        normalization_decomposition: Preimage,
        normalization_schema: ReadoutPreimageSchema,
    ) -> Result<Family<Mat>, ValueReadoutError> {
        let ring = compiler.ring.clone();
        self.validate(&ring)?;
        let ring_dimension = concrete_ring_dimension(&ring)?;
        let plan = self.trace_plan(ring_dimension)?;
        let helper_count = helper_sigmas.len();
        let flat_count =
            output_count.checked_mul(helper_count).ok_or(ValueReadoutError::InvalidTraceSet)?;
        if output_count == 0 ||
            *inputs.count() != IntExpr::constant(output_count) ||
            helper_sigmas != plan.non_identity_helpers() ||
            *public_decompositions.count() != IntExpr::constant(flat_count) ||
            *masks.count() != IntExpr::constant(helper_count)
        {
            return Err(ValueReadoutError::InvalidTraceSet);
        }
        let (gadget_base, digit_count) = public_bgg_schema(compiler)?;
        if target_schema.gadget_base != gadget_base ||
            target_schema.digit_count != digit_count ||
            normalization_schema.gadget_base != gadget_base ||
            normalization_schema.digit_count != digit_count
        {
            return Err(ValueReadoutError::InvalidPreimageSchema);
        }
        validate_preimage_schema(
            &target_decompositions.get_static(0),
            &inputs.get_static(0),
            &gadget_base,
            digit_count,
            Some(&target_schema.max_coefficient_bound),
        )?;
        validate_family_preimage_shape(
            public_decompositions.element_type(),
            &inputs.get_static(0),
            digit_count,
        )?;
        validate_family_output_shape(masks.element_type(), &inputs.get_static(0), &ring)?;

        if *target_decompositions.count() != IntExpr::constant(output_count) {
            return Err(ValueReadoutError::InvalidTraceSet);
        }
        let transformed = (inputs, target_decompositions)
            .parallel_zip_tuple(|_, (input, target)| target.mul_small_rhs(input))
            .map_err(|_| ValueReadoutError::Graph(crate::ExponentLutError::InvalidLut))?;
        let helper_indices = Parallel::range(flat_count)
            .map_values(|index| index.as_int().rem(Int::constant(helper_count)))
            .map_err(|_| ValueReadoutError::Graph(crate::ExponentLutError::InvalidLut))?;
        let flat_masks = masks
            .parallel_gather(helper_indices)
            .map_err(|_| ValueReadoutError::Graph(crate::ExponentLutError::InvalidLut))?;
        let helper_branches = (public_decompositions, flat_masks)
            .parallel_zip_tuple(|_, (public, mask)| public.mul_small_rhs(mask))
            .map_err(|_| ValueReadoutError::Graph(crate::ExponentLutError::InvalidLut))?;
        let helper_sums =
            reduce_matrix_family_segments(helper_branches, output_count, helper_count)?;
        let traced = (transformed, helper_sums)
            .parallel_zip_tuple(|_, (identity, helpers)| identity + helpers)
            .map_err(|_| ValueReadoutError::Graph(crate::ExponentLutError::InvalidLut))?;
        let denominator = normalization_denominator(&self.layout, ring_dimension)?;
        let modulus = ring
            .matrix_type((1, 1))
            .modulus
            .evaluate(&ParamEnv::default())
            .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?;
        modular_inverse(&(BigInt::from(denominator) % &modulus), &modulus)
            .ok_or(ValueReadoutError::NonInvertibleNormalization)?;
        validate_preimage_schema(
            &normalization_decomposition,
            &traced.get_static(0),
            &normalization_schema.gadget_base,
            normalization_schema.digit_count,
            Some(&normalization_schema.max_coefficient_bound),
        )?;
        traced
            .parallel_map_values(|_, matrix| {
                normalization_decomposition.clone().mul_small_rhs(matrix)
            })
            .map_err(|_| ValueReadoutError::Graph(crate::ExponentLutError::InvalidLut))
    }

    /// Returns the normalization descriptor `(2*n')^-1` for this ring.
    pub fn normalization_target_descriptor(
        &self,
        ring: &Ring,
        gadget: &Mat,
    ) -> Result<NormalizationTargetDescriptor, ValueReadoutError> {
        self.validate(ring)?;
        if gadget.matrix_type().modulus != ring.matrix_type((1, 1)).modulus ||
            gadget.matrix_type().ring_dimension != ring.matrix_type((1, 1)).ring_dimension
        {
            return Err(ValueReadoutError::RingTypeMismatch);
        }
        let matrix = concrete_matrix_type(gadget.matrix_type())?;
        let ring_dimension = concrete_ring_dimension(ring)?;
        let denominator = normalization_denominator(&self.layout, ring_dimension)?;
        Ok(NormalizationTargetDescriptor {
            matrix,
            layout: self.layout,
            numerator: BigInt::one(),
            denominator,
        })
    }

    /// Describes the bounded preimage needed for normalization.
    pub fn normalization_preimage_requirement(
        &self,
        ring: &Ring,
        gadget: Mat,
        max_coefficient_bound: impl Into<BigInt>,
        gadget_base: impl Into<BigInt>,
        digit_count: usize,
    ) -> Result<NormalizationPreimageRequirement, ValueReadoutError> {
        let gadget_base = gadget_base.into();
        if gadget_base <= BigInt::one() {
            return Err(ValueReadoutError::InvalidGadgetBase);
        }
        if digit_count == 0 {
            return Err(ValueReadoutError::InvalidDigitCount);
        }
        let max_coefficient_bound = max_coefficient_bound.into();
        if max_coefficient_bound < BigInt::zero() {
            return Err(ValueReadoutError::InvalidPreimageBound);
        }
        validate_canonical_gadget(&gadget, &gadget_base, digit_count)?;
        let target = self.normalization_target_descriptor(ring, &gadget)?;
        let artifact_type = ArtifactType::Preimage {
            matrix: target.matrix.clone(),
            max_coefficient_bound: max_coefficient_bound.clone(),
        };
        Ok(NormalizationPreimageRequirement {
            target,
            artifact_type,
            role: NormalizationPreimageRole::InverseNativeTraceGadget,
            max_coefficient_bound,
            gadget_base,
            digit_count,
            readout_identity: self.identity(),
        })
    }

    /// Scales a caller-supplied gadget matrix by `(2*n')^-1` in the ring.
    ///
    /// This only constructs the public target expression.  Callers must still
    /// request a typed bounded preimage through the DSL artifact path.
    pub fn normalization_target(&self, ring: &Ring, gadget: Mat) -> Result<Mat, ValueReadoutError> {
        self.validate(ring)?;
        if gadget.matrix_type().modulus != ring.matrix_type((1, 1)).modulus ||
            gadget.matrix_type().ring_dimension != ring.matrix_type((1, 1)).ring_dimension
        {
            return Err(ValueReadoutError::RingTypeMismatch);
        }
        let ring_dimension = concrete_ring_dimension(ring)?;
        let denominator = normalization_denominator(&self.layout, ring_dimension)?;
        let modulus = ring
            .matrix_type((1, 1))
            .modulus
            .evaluate(&ParamEnv::default())
            .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?;
        let inverse = modular_inverse(&(BigInt::from(denominator) % &modulus), &modulus)
            .ok_or(ValueReadoutError::NonInvertibleNormalization)?;
        Ok(ring.polynomial([inverse.into()]) * gadget)
    }
}

fn is_scalar_shape(matrix_type: &mxx_ir_core::types::MatrixType) -> bool {
    matrix_type.rows == mxx_ir_core::IntExpr::constant(1) &&
        matrix_type.columns == mxx_ir_core::IntExpr::constant(1)
}

/// Serializes the table graph recursively, omitting allocation addresses and
/// source locations.  This makes independently rebuilt declarations stable
/// while retaining all table values, types, and argument structure.
fn canonical_table_declaration(table: &Mat) -> Result<Vec<u8>, ValueReadoutError> {
    fn canonical_value(value: &mxx_ir_core::graph::ValueHandle) -> serde_json::Value {
        let node = value.node();
        serde_json::json!({
            "kind": serde_json::to_value(node.kind()).expect("NodeKind is serializable"),
            "output_types": serde_json::to_value(node.output_types())
                .expect("WireType is serializable"),
            "arguments": node.arguments().iter().map(canonical_value).collect::<Vec<_>>(),
            "port": value.port().0,
        })
    }
    serde_json::to_vec(&canonical_value(table.value_handle()))
        .map_err(|_| ValueReadoutError::Canonicalization)
}

fn canonical_polynomial_coefficients(table: &Mat) -> Option<Vec<BigInt>> {
    let mxx_ir_core::node::NodeKind::ConstantMatrix {
        value: mxx_ir_core::node::ConstantMatrix::Polynomial { coefficients },
        ..
    } = table.value_handle().node().kind()
    else {
        return None;
    };
    coefficients
        .iter()
        .map(|coefficient| match coefficient {
            mxx_ir_core::IntExpr::Const(value) => Some(value.clone()),
            _ => None,
        })
        .collect()
}

fn validate_canonical_gadget(
    gadget: &Mat,
    base: &BigInt,
    digit_count: usize,
) -> Result<(), ValueReadoutError> {
    let mxx_ir_core::node::NodeKind::ConstantMatrix {
        matrix_type,
        value: mxx_ir_core::node::ConstantMatrix::Gadget { base: declared_base, small: false },
    } = gadget.value_handle().node().kind()
    else {
        return Err(ValueReadoutError::InvalidNormalizationGadget);
    };
    let mxx_ir_core::IntExpr::Const(declared_base) = declared_base else {
        return Err(ValueReadoutError::InvalidNormalizationGadget);
    };
    if declared_base != base {
        return Err(ValueReadoutError::InvalidNormalizationGadget);
    }
    let concrete = concrete_matrix_type(matrix_type)?;
    if concrete.rows == 0 || concrete.columns != concrete.rows.saturating_mul(digit_count) {
        return Err(ValueReadoutError::InvalidNormalizationGadget);
    }
    Ok(())
}

fn concrete_matrix_type(
    matrix_type: &mxx_ir_core::types::MatrixType,
) -> Result<ConcreteMatrixType, ValueReadoutError> {
    let env = ParamEnv::default();
    let evaluate = |expr: &mxx_ir_core::IntExpr| {
        expr.evaluate(&env).map_err(|_| ValueReadoutError::NonConcreteRingParameter)
    };
    Ok(ConcreteMatrixType {
        modulus: evaluate(&matrix_type.modulus)?,
        ring_dimension: evaluate(&matrix_type.ring_dimension)?
            .to_usize()
            .ok_or(ValueReadoutError::NonConcreteRingParameter)?,
        rows: evaluate(&matrix_type.rows)?
            .to_usize()
            .ok_or(ValueReadoutError::NonConcreteRingParameter)?,
        columns: evaluate(&matrix_type.columns)?
            .to_usize()
            .ok_or(ValueReadoutError::NonConcreteRingParameter)?,
    })
}

fn reduce_matrix_family_segments(
    family: Family<Mat>,
    segment_count: usize,
    segment_size: usize,
) -> Result<Family<Mat>, ValueReadoutError> {
    if segment_count == 0 || segment_size == 0 {
        return Err(ValueReadoutError::InvalidTraceSet);
    }
    crate::encoding::reduce_encoding_family_segments(family, segment_count, segment_size).map_err(
        |error| {
            ValueReadoutError::Graph(crate::ExponentLutError::OnlinePrfStage {
                stage: "reduce readout helper segments",
                detail: error.to_string(),
            })
        },
    )
}

fn validate_preimage_ring(preimage: &Preimage, ring: &Ring) -> Result<(), ValueReadoutError> {
    let expected = ring.matrix_type((1, 1));
    let actual = preimage.matrix_type();
    if actual.modulus != expected.modulus || actual.ring_dimension != expected.ring_dimension {
        return Err(ValueReadoutError::RingTypeMismatch);
    }
    let bound = preimage
        .max_coefficient_bound()
        .evaluate(&ParamEnv::default())
        .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?;
    if bound < BigInt::zero() {
        return Err(ValueReadoutError::InvalidPreimageBound);
    }
    Ok(())
}

fn validate_matrix_ring(matrix: &Mat, ring: &Ring) -> Result<(), ValueReadoutError> {
    let expected = ring.matrix_type((1, 1));
    let actual = matrix.matrix_type();
    if actual.modulus != expected.modulus || actual.ring_dimension != expected.ring_dimension {
        return Err(ValueReadoutError::RingTypeMismatch);
    }
    Ok(())
}

fn validate_family_preimage_shape(
    actual: &mxx_ir_core::types::MatrixType,
    source: &Mat,
    digit_count: usize,
) -> Result<(), ValueReadoutError> {
    let actual = concrete_matrix_type(actual)?;
    let source = concrete_matrix_type(source.matrix_type())?;
    if actual.modulus != source.modulus ||
        actual.ring_dimension != source.ring_dimension ||
        actual.rows != source.rows.saturating_mul(digit_count) ||
        actual.columns != source.columns
    {
        return Err(ValueReadoutError::InvalidPreimageSchema);
    }
    Ok(())
}

fn validate_encoding_action_preimage_family_type(
    actual: &mxx_ir_core::types::MatrixType,
    carrier: &mxx_ir_core::types::MatrixType,
    digit_count: usize,
) -> Result<(), ValueReadoutError> {
    let actual = concrete_matrix_type(actual)?;
    let carrier = concrete_matrix_type(carrier)?;
    if actual.modulus != carrier.modulus ||
        actual.ring_dimension != carrier.ring_dimension ||
        actual.rows != carrier.columns ||
        actual.columns != carrier.columns ||
        digit_count == 0 ||
        carrier.columns % digit_count != 0
    {
        return Err(ValueReadoutError::InvalidPreimageSchema);
    }
    Ok(())
}

fn validate_family_output_shape(
    actual: &mxx_ir_core::types::MatrixType,
    source: &Mat,
    ring: &Ring,
) -> Result<(), ValueReadoutError> {
    validate_family_output_type(actual, source.matrix_type(), ring)
}

fn validate_family_output_type(
    actual: &mxx_ir_core::types::MatrixType,
    source: &mxx_ir_core::types::MatrixType,
    ring: &Ring,
) -> Result<(), ValueReadoutError> {
    let expected_ring = ring.matrix_type((1, 1));
    if actual.modulus != expected_ring.modulus ||
        actual.ring_dimension != expected_ring.ring_dimension
    {
        return Err(ValueReadoutError::RingTypeMismatch);
    }
    let actual = concrete_matrix_type(actual)?;
    let source = concrete_matrix_type(source)?;
    if actual.rows != source.rows || actual.columns != source.columns {
        return Err(ValueReadoutError::InvalidPreimageSchema);
    }
    Ok(())
}

fn validate_output_shape(output: &Mat, source: &Mat, ring: &Ring) -> Result<(), ValueReadoutError> {
    validate_matrix_ring(output, ring)?;
    let expected = concrete_matrix_type(source.matrix_type())?;
    let actual = concrete_matrix_type(output.matrix_type())?;
    if actual.rows != expected.rows || actual.columns != expected.columns {
        return Err(ValueReadoutError::InvalidPreimageSchema);
    }
    Ok(())
}

fn validate_preimage_schema(
    preimage: &Preimage,
    source: &Mat,
    gadget_base: &BigInt,
    digit_count: usize,
    expected_bound: Option<&BigInt>,
) -> Result<(), ValueReadoutError> {
    let actual = concrete_matrix_type(preimage.matrix_type())?;
    let source = concrete_matrix_type(source.matrix_type())?;
    if actual.modulus != source.modulus ||
        actual.ring_dimension != source.ring_dimension ||
        actual.rows != source.rows.saturating_mul(digit_count) ||
        actual.columns != source.columns
    {
        return Err(ValueReadoutError::InvalidPreimageSchema);
    }
    let bound = preimage
        .max_coefficient_bound()
        .evaluate(&ParamEnv::default())
        .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?;
    if bound < BigInt::zero() || expected_bound.is_some_and(|expected| expected != &bound) {
        return Err(ValueReadoutError::InvalidPreimageSchema);
    }
    if gadget_base <= &BigInt::one() || digit_count == 0 {
        return Err(ValueReadoutError::InvalidPreimageSchema);
    }
    Ok(())
}

/// Validates a cached right action consumed by an online encoding carrier.
///
/// A carrier has shape `1 x m_b`, while the cached value is `D(A)` for an
/// underlying public matrix `A` with `m_b` columns.  Consequently the
/// executable `mul_small_rhs` consumes an `m_b x m_b` preimage.  Treating the
/// carrier itself as the decomposition source would incorrectly require only
/// `digit_count x m_b` rows and accept a matrix that cannot be multiplied.
fn validate_encoding_action_preimage_schema(
    preimage: &Preimage,
    carrier: &Mat,
    gadget_base: &BigInt,
    digit_count: usize,
    expected_bound: Option<&BigInt>,
) -> Result<(), ValueReadoutError> {
    validate_encoding_action_preimage_schema_type(
        preimage,
        carrier.matrix_type(),
        gadget_base,
        digit_count,
        expected_bound,
    )
}

fn validate_encoding_action_preimage_schema_type(
    preimage: &Preimage,
    carrier: &mxx_ir_core::types::MatrixType,
    gadget_base: &BigInt,
    digit_count: usize,
    expected_bound: Option<&BigInt>,
) -> Result<(), ValueReadoutError> {
    let actual = concrete_matrix_type(preimage.matrix_type())?;
    let carrier = concrete_matrix_type(carrier)?;
    if actual.modulus != carrier.modulus ||
        actual.ring_dimension != carrier.ring_dimension ||
        actual.rows != carrier.columns ||
        actual.columns != carrier.columns ||
        digit_count == 0 ||
        carrier.columns % digit_count != 0
    {
        return Err(ValueReadoutError::InvalidPreimageSchema);
    }
    let bound = preimage
        .max_coefficient_bound()
        .evaluate(&ParamEnv::default())
        .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?;
    if bound < BigInt::zero() || expected_bound.is_some_and(|expected| expected != &bound) {
        return Err(ValueReadoutError::InvalidPreimageSchema);
    }
    if gadget_base <= &BigInt::one() {
        return Err(ValueReadoutError::InvalidPreimageSchema);
    }
    Ok(())
}

fn public_bgg_schema(
    compiler: &BggPublicKeyCompiler,
) -> Result<(BigInt, usize), ValueReadoutError> {
    let base = compiler
        .base
        .evaluate(&ParamEnv::default())
        .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?;
    let digits = compiler
        .digit_count
        .evaluate(&ParamEnv::default())
        .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?
        .to_usize()
        .ok_or(ValueReadoutError::NonConcreteRingParameter)?;
    Ok((base, digits))
}

fn concrete_ring_dimension(ring: &Ring) -> Result<usize, ValueReadoutError> {
    ring.matrix_type((1, 1))
        .ring_dimension
        .evaluate(&ParamEnv::default())
        .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?
        .to_usize()
        .ok_or(ValueReadoutError::NonConcreteRingParameter)
}

fn native_width(layout: &ExpLayout, ring_dimension: usize) -> Result<usize, ValueReadoutError> {
    layout.validate(ring_dimension)?;
    ring_dimension
        .checked_div(layout.stride())
        .filter(|width| *width > 0)
        .ok_or(ValueReadoutError::NormalizationDenominatorOverflow)
}

fn full_trace_layout(
    layout: &ExpLayout,
    ring_dimension: usize,
) -> Result<ExpLayout, ValueReadoutError> {
    if ring_dimension == 0 || !ring_dimension.is_power_of_two() {
        return Err(LayoutError::InvalidRingDimension(ring_dimension).into());
    }
    let ring_log2 = ring_dimension.trailing_zeros();
    let width_log2 = ring_log2
        .checked_sub(layout.stride_log2)
        .ok_or(LayoutError::StrideOutOfRange { stride_log2: layout.stride_log2, ring_log2 })?;
    let full = ExpLayout { stride_log2: layout.stride_log2, width_log2 };
    full.validate(ring_dimension)?;
    Ok(full)
}

fn normalization_denominator(
    layout: &ExpLayout,
    ring_dimension: usize,
) -> Result<usize, ValueReadoutError> {
    native_width(layout, ring_dimension)?
        .checked_mul(2)
        .ok_or(ValueReadoutError::NormalizationDenominatorOverflow)
}

fn modular_inverse(value: &BigInt, modulus: &BigInt) -> Option<BigInt> {
    if modulus <= &BigInt::one() {
        return None;
    }
    let mut old_r = value % modulus;
    if old_r.sign() == Sign::Minus {
        old_r += modulus;
    }
    let mut r = modulus.clone();
    let mut old_s = BigInt::one();
    let mut s = BigInt::zero();
    while !r.is_zero() {
        let quotient = &old_r / &r;
        let next_r = old_r - &quotient * &r;
        old_r = r;
        r = next_r;
        let next_s = old_s - &quotient * &s;
        old_s = s;
        s = next_s;
    }
    (old_r == BigInt::one()).then(|| {
        let mut result = old_s % modulus;
        if result.sign() == Sign::Minus {
            result += modulus;
        }
        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::{DslContext, Ring};
    use mxx_ir_core::{
        IntExpr,
        node::{ConstantMatrix, NodeKind},
    };

    #[test]
    fn target_plaintext_gain_tracks_table_and_monomial_coefficients() {
        // Native readout requires width >= 2. Check the monomial invariant
        // directly on the same convolution-gain primitive without relaxing it.
        for sign in [-1, 1] {
            assert_eq!(
                crate::noise::RightActionNoiseGain::from_polynomial_coefficients([
                    BigInt::zero(),
                    sign.into(),
                    BigInt::zero()
                ],),
                crate::noise::RightActionNoiseGain::unit()
            );
        }
        for stride_log2 in [0, 1, 3] {
            let readout = ValueReadout::from_canonical_values(
                ExpLayout { stride_log2, width_log2: 2 },
                vec![1, 1, 2, 2],
            )
            .unwrap();
            assert_eq!(
                readout.target_plaintext_gain(4 << stride_log2).unwrap(),
                crate::noise::RightActionNoiseGain { l1: 8u8.into(), l2_squared: 20u8.into() }
            );
        }
        let layout = ExpLayout { stride_log2: 0, width_log2: 2 };
        let ring = Ring::new(97, 4);
        let symbolic = ValueReadout::new(layout, ring.input("table", (1, 1))).unwrap();
        assert!(symbolic.target_plaintext_gain(4).is_err());
        let padded = ValueReadout::new(layout, ring.polynomial([1.into()])).unwrap();
        assert_eq!(
            padded.target_plaintext_gain(4).unwrap(),
            crate::noise::RightActionNoiseGain { l1: 4u8.into(), l2_squared: 4u8.into() }
        );
        let aliased = ValueReadout::new(layout, ring.polynomial(vec![1.into(); 5])).unwrap();
        assert!(aliased.target_plaintext_gain(4).is_err());
    }

    fn independent_automorphisms(ring_dimension: usize, layout: ExpLayout) -> Vec<usize> {
        let modulus = 2 * ring_dimension;
        let mut result = std::collections::BTreeSet::new();
        for subset in 0..layout.width() {
            let mut product = 1usize;
            for bit in 0..layout.width_log2 {
                if subset & (1usize << bit) != 0 {
                    let denominator = 1usize << (layout.stride_log2 + bit + 1);
                    product = product * (modulus / denominator + 1) % modulus;
                }
            }
            result.insert(product);
        }
        result.into_iter().collect()
    }

    fn collect_target_ir(
        value: &mxx_ir_core::graph::ValueHandle,
        extracts: &mut Vec<usize>,
        rotations: &mut Vec<usize>,
        negated_monomials: &mut usize,
    ) {
        let node = value.node();
        match node.kind() {
            NodeKind::ExtractCoefficient { position, .. } => {
                if let IntExpr::Const(position) = position {
                    extracts.push(position.to_usize().unwrap());
                }
            }
            NodeKind::ConstantMatrix { value: ConstantMatrix::Rotation { exponent }, .. } => {
                if let IntExpr::Const(exponent) = exponent {
                    rotations.push(exponent.to_usize().unwrap());
                }
            }
            NodeKind::MatrixNegate => *negated_monomials += 1,
            _ => {}
        }
        for argument in node.arguments() {
            collect_target_ir(argument, extracts, rotations, negated_monomials);
        }
    }

    #[test]
    fn n32_m1_has_exact_j_order_and_one_identity() {
        let ring = Ring::new(97, 32);
        let readout = ValueReadout::from_values(
            ExpLayout { stride_log2: 1, width_log2: 4 },
            &ring,
            vec![0; 16],
        )
        .unwrap();
        let plan = readout.trace_plan(32).unwrap();
        assert_eq!(
            plan.automorphisms(),
            &[1, 3, 5, 7, 9, 11, 15, 17, 21, 25, 27, 45, 51, 55, 61, 63]
        );
        assert_eq!(
            plan.non_identity_helpers(),
            &[3, 5, 7, 9, 11, 15, 17, 21, 25, 27, 45, 51, 55, 61, 63]
        );
        assert_eq!(plan.term_count(), 16);
        assert_eq!(plan.helper_count(), 15);
        assert_eq!(plan.automorphisms().iter().filter(|index| **index == 1).count(), 1);
    }

    #[test]
    fn n65536_m5_has_exact_j_order_and_helper_list() {
        let ring = Ring::new(97, 1usize << 16);
        let readout = ValueReadout::from_values(
            ExpLayout { stride_log2: 5, width_log2: 11 },
            &ring,
            vec![0; 1usize << 11],
        )
        .unwrap();
        let plan = readout.trace_plan(1usize << 16).unwrap();
        let expected =
            independent_automorphisms(1usize << 16, ExpLayout { stride_log2: 5, width_log2: 11 });
        assert_eq!(plan.automorphisms(), expected.as_slice());
        assert_eq!(plan.non_identity_helpers(), &expected[1..]);
        assert_eq!(plan.term_count(), 2048);
        assert_eq!(plan.helper_count(), 2047);
        assert_eq!(plan.automorphisms().iter().filter(|index| **index == 1).count(), 1);
    }

    #[test]
    fn target_has_canonical_coefficients_and_expected_ir_signs() {
        let ring = Ring::new(97, 32);
        let readout = ValueReadout::from_values(
            ExpLayout { stride_log2: 1, width_log2: 4 },
            &ring,
            (0..16).map(|value| value % 2).collect(),
        )
        .unwrap();
        assert_eq!(
            readout.canonical_coefficients().unwrap(),
            (0..16).map(|v| BigInt::from(v % 2)).collect::<Vec<_>>()
        );
        let target = readout.target(&ring).unwrap();
        assert_eq!(target.matrix_type(), &ring.matrix_type((1, 1)));
        let mut extracts = Vec::new();
        let mut rotations = Vec::new();
        let mut negated_monomials = 0;
        collect_target_ir(
            target.value_handle(),
            &mut extracts,
            &mut rotations,
            &mut negated_monomials,
        );
        extracts.sort_unstable();
        assert_eq!(extracts, (0..16).collect::<Vec<_>>());
        assert_eq!(rotations.len(), 16);
        assert_eq!(negated_monomials, 15);
        let expected_signs: Vec<BigInt> =
            (0..16).map(|k| BigInt::from(2 * (k % 2) as i32 - 1)).collect();
        assert_eq!(
            expected_signs,
            vec![
                BigInt::from(-1),
                BigInt::from(1),
                BigInt::from(-1),
                BigInt::from(1),
                BigInt::from(-1),
                BigInt::from(1),
                BigInt::from(-1),
                BigInt::from(1),
                BigInt::from(-1),
                BigInt::from(1),
                BigInt::from(-1),
                BigInt::from(1),
                BigInt::from(-1),
                BigInt::from(1),
                BigInt::from(-1),
                BigInt::from(1),
            ]
        );
    }

    #[test]
    fn normalization_requirement_is_typed_metadata_and_inverse_target() {
        let ring = Ring::new(97, 32);
        let readout = ValueReadout::from_values(
            ExpLayout { stride_log2: 1, width_log2: 4 },
            &ring,
            vec![0; 16],
        )
        .unwrap();
        let gadget = ring.gadget(1, 2, 4);
        let requirement =
            readout.normalization_preimage_requirement(&ring, gadget.clone(), 7, 2, 4).unwrap();
        assert_eq!(requirement.target.denominator, 32);
        assert_eq!(requirement.target.matrix.modulus, BigInt::from(97));
        assert_eq!(requirement.target.matrix.ring_dimension, 32);
        assert_eq!(requirement.target.matrix.rows, 1);
        assert_eq!(requirement.target.matrix.columns, 4);
        assert_eq!(requirement.role, NormalizationPreimageRole::InverseNativeTraceGadget);
        assert_eq!(requirement.max_coefficient_bound, BigInt::from(7));
        assert_eq!(requirement.gadget_base, BigInt::from(2));
        assert_eq!(requirement.digit_count, 4);
        assert_eq!(requirement.readout_identity, readout.identity());
        assert_eq!(modular_inverse(&BigInt::from(32), &BigInt::from(97)), Some(BigInt::from(94)));
        assert_eq!(
            requirement.artifact_type,
            ArtifactType::Preimage {
                matrix: requirement.target.matrix.clone(),
                max_coefficient_bound: BigInt::from(7)
            }
        );
        let target = readout.normalization_target(&ring, ring.identity(1)).unwrap();
        let bytes = canonical_table_declaration(&target).unwrap();
        assert!(String::from_utf8_lossy(&bytes).contains("94"));
        let zero_gadget = ring.zero((1, 4));
        assert!(matches!(
            readout.normalization_preimage_requirement(&ring, zero_gadget, 7, 2, 4),
            Err(ValueReadoutError::InvalidNormalizationGadget)
        ));
    }

    #[test]
    fn m_zero_fixture_matches_legacy_coefficients_and_monomial_signs() {
        let ring = Ring::new(97, 32);
        let readout = ValueReadout::from_values(
            ExpLayout { stride_log2: 0, width_log2: 4 },
            &ring,
            (0..32).map(|k| k % 2).collect(),
        )
        .unwrap();
        assert_eq!(readout.canonical_coefficients().unwrap().len(), 32);
        let target = readout.target(&ring).unwrap();
        let mut extracts = Vec::new();
        let mut rotations = Vec::new();
        let mut negated_monomials = 0;
        collect_target_ir(
            target.value_handle(),
            &mut extracts,
            &mut rotations,
            &mut negated_monomials,
        );
        extracts.sort_unstable();
        assert_eq!(extracts, (0..32).collect::<Vec<_>>());
        assert_eq!(negated_monomials, 31);
        assert!(rotations.contains(&31) && rotations.contains(&1));
    }

    #[test]
    fn stable_identity_is_content_complete_and_layout_typed() {
        let layout = ExpLayout { stride_log2: 1, width_log2: 4 };
        let first = ValueReadout::from_values(layout, &Ring::new(97, 32), vec![0; 16]).unwrap();
        let rebuilt = ValueReadout::from_values(layout, &Ring::new(97, 32), vec![0; 16]).unwrap();
        let changed_table =
            ValueReadout::from_values(layout, &Ring::new(97, 32), vec![1; 16]).unwrap();
        let changed_modulus =
            ValueReadout::from_values(layout, &Ring::new(101, 32), vec![0; 16]).unwrap();
        let changed_layout = ValueReadout::from_values(
            ExpLayout { stride_log2: 0, width_log2: 4 },
            &Ring::new(97, 32),
            vec![0; 32],
        )
        .unwrap();
        assert_eq!(first.identity(), rebuilt.identity());
        assert_ne!(first.identity(), changed_table.identity());
        assert_ne!(first.identity(), changed_modulus.identity());
        assert_ne!(first.identity(), changed_layout.identity());
    }

    #[test]
    fn normalization_rejects_mismatched_and_noninvertible_rings() {
        let ring = Ring::new(97, 32);
        let readout = ValueReadout::from_values(
            ExpLayout { stride_log2: 1, width_log2: 4 },
            &ring,
            vec![0; 16],
        )
        .unwrap();
        assert_eq!(readout.validate(&Ring::new(101, 32)), Err(ValueReadoutError::RingTypeMismatch));
        let bad_gadget = Ring::new(101, 32).gadget(1, 2, 4);
        assert_eq!(
            readout.normalization_target_descriptor(&ring, &bad_gadget),
            Err(ValueReadoutError::RingTypeMismatch)
        );
        let even_ring = Ring::new(32, 32);
        let even_readout = ValueReadout::from_values(
            ExpLayout { stride_log2: 0, width_log2: 4 },
            &even_ring,
            vec![0; 32],
        )
        .unwrap();
        assert!(matches!(
            even_readout.normalization_target(&even_ring, even_ring.identity(1)),
            Err(ValueReadoutError::NonInvertibleNormalization)
        ));
    }

    #[test]
    fn cached_readout_uses_supplied_preimages_and_no_decomposition_nodes() {
        let ring = Ring::new(97, 4);
        let layout = ExpLayout { stride_log2: 0, width_log2: 0 };
        let readout = ValueReadout::from_values(layout, &ring, vec![0, 1, 0, 1]).unwrap();
        let compiler = BggEncodingCompiler;
        let input = BggEncodingWire {
            vector: ring.input("readout-input-vector", (1, 4)),
            plaintext: Some(ring.polynomial([1.into()])),
        };
        let make_helpers = |prefix: &str, rows| {
            [3usize, 5, 7]
                .into_iter()
                .map(|sigma| {
                    ValueReadoutHelper::new(
                        layout,
                        sigma,
                        4.into(),
                        2,
                        7.into(),
                        ring.preimage_input(format!("{prefix}-switch-{sigma}"), (rows, 4), 7),
                        ring.preimage_input(format!("{prefix}-public-{sigma}"), (rows, 4), 7),
                        BggEncodingWire { vector: ring.zero((1, 4)), plaintext: None },
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>()
        };
        let wrong_helpers = make_helpers("readout-wrong", 2);
        assert!(matches!(
            readout.execute_with_preprocessed_artifacts(
                &compiler,
                &input,
                ring.preimage_input("readout-wrong-target", (2, 4), 7),
                ReadoutPreimageSchema::new(4, 2, 7).unwrap(),
                &wrong_helpers,
                ring.preimage_input("readout-wrong-normalization", (2, 4), 7),
                ReadoutPreimageSchema::new(4, 2, 7).unwrap(),
            ),
            Err(ValueReadoutError::InvalidPreimageSchema)
        ));
        let helpers = make_helpers("readout", 4);
        let output = readout
            .execute_with_preprocessed_artifacts(
                &compiler,
                &input,
                ring.preimage_input("readout-target-decomposition", (4, 4), 7),
                ReadoutPreimageSchema::new(4, 2, 7).unwrap(),
                &helpers,
                ring.preimage_input("readout-normalization-decomposition", (4, 4), 7),
                ReadoutPreimageSchema::new(4, 2, 7).unwrap(),
            )
            .unwrap();
        let built = DslContext::new("cached-readout-no-decompose")
            .output("vector", output.vector)
            .unwrap()
            .build()
            .unwrap();
        let nodes =
            built.graph.scopes().values().flat_map(|scope| scope.nodes()).collect::<Vec<_>>();
        assert_eq!(
            nodes
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::GadgetDecompose { .. }))
                .count(),
            0
        );
        assert_eq!(
            nodes.iter().filter(|node| matches!(node.kind(), NodeKind::SequentialLoop(_))).count(),
            0
        );
        built.validate(&ParamEnv::default()).unwrap();
    }

    #[test]
    fn public_readout_uses_masked_switched_decomposition_formula() {
        let ring = Ring::new(97, 4);
        let layout = ExpLayout { stride_log2: 0, width_log2: 0 };
        let readout = ValueReadout::from_values(layout, &ring, vec![0, 1, 0, 1]).unwrap();
        let compiler =
            BggPublicKeyCompiler { ring: ring.clone(), base: 4.into(), digit_count: 2.into() };
        let input = BggPublicKeyWire {
            matrix: ring.input("public-readout-input", (2, 4)),
            reveal_plaintext: false,
        };
        let helpers = [3usize, 5, 7]
            .into_iter()
            .map(|sigma| {
                ValueReadoutPublicHelper::new(
                    layout,
                    sigma,
                    4.into(),
                    2,
                    7.into(),
                    ring.preimage_input(format!("public-readout-decomposition-{sigma}"), (4, 4), 7),
                    ring.input(format!("public-readout-mask-{sigma}"), (2, 4)),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let output = readout
            .project_public_with_preprocessed_artifacts(
                &compiler,
                &input,
                ring.preimage_input("public-readout-target-decomposition", (4, 4), 7),
                ReadoutPreimageSchema::new(4, 2, 7).unwrap(),
                &helpers,
                ring.preimage_input("public-readout-normalization-decomposition", (4, 4), 7),
                ReadoutPreimageSchema::new(4, 2, 7).unwrap(),
            )
            .unwrap();
        let built = DslContext::new("public-readout-trace")
            .output("output", output)
            .unwrap()
            .build()
            .unwrap();
        let nodes =
            built.graph.scopes().values().flat_map(|scope| scope.nodes()).collect::<Vec<_>>();
        assert_eq!(
            nodes.iter().filter(|node| matches!(node.kind(), NodeKind::MatrixMulSmallRhs)).count(),
            helpers.len() + 2,
            "public helper branches must be exactly A_mask * D(A_switched)"
        );
        assert_eq!(
            nodes
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::RingAutomorphism { .. }))
                .count(),
            0,
            "public projection has no switched ciphertext vector term"
        );
        assert_eq!(
            nodes
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::GadgetDecompose { .. }))
                .count(),
            0
        );
        assert_eq!(
            nodes.iter().filter(|node| matches!(node.kind(), NodeKind::SequentialLoop(_))).count(),
            0
        );
        built.validate(&ParamEnv::default()).unwrap();
    }
}
