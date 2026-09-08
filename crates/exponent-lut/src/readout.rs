//! General native value-readout declarations and terminal evaluation.
use crate::{
    layout::{ExpLayout, LayoutError},
    program::ProgramValue,
    scheme::ExponentLUTEncodingScheme,
};
use mxx_dsl::{DslContext, DslError, Family, GraphValue, Int, Mat, Ring};
use mxx_ir_core::{ParamEnv, artifact::ProductionId};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

#[derive(Debug, Eq, PartialEq, Error)]
pub enum ValueReadoutError {
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

    /// The trace index set was not the complete native subgroup.
    #[error("value-readout trace must contain each odd native automorphism once")]
    InvalidTraceSet,
    /// The affine half-offset requires an odd target modulus.
    #[error("two is not invertible in the affine readout modulus")]
    NonInvertibleAffineScale,
    /// A table declaration could not be serialized into its stable identity form.
    #[error("value-readout table declaration could not be canonicalized")]
    Canonicalization,
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

    /// Build weighted rows w*(f-1/2) at the target modulus, before sampling.
    /// Negating a native monomial and adding w/2 gives w*(1-f). The returned
    /// unweighted digit bound covers both f and 1-f; it is distinct from the
    /// modular coefficient bound of the encrypted half-offset table.
    pub fn weighted_centered_table<const C: usize>(
        &self,
        weight: &[Vec<BigInt>; C],
        modulus: &BigInt,
    ) -> Result<(RowLutTable<C>, BigUint), ValueReadoutError>
    where
        [Vec<BigInt>; C]: Serialize,
    {
        let values = self.canonical_coefficients().ok_or(ValueReadoutError::Canonicalization)?;
        let inverse_two = crate::utils::modular_inverse(&2.into(), modulus)
            .ok_or(ValueReadoutError::NonInvertibleAffineScale)?;
        let digit_bound = values
            .iter()
            .flat_map(|value| [value.clone(), BigInt::one() - value])
            .map(|value| value.magnitude().clone())
            .max()
            .unwrap_or_default();
        let table = RowLutTable::new(
            values
                .iter()
                .map(|value| {
                    weight.each_ref().map(|polynomial| {
                        polynomial
                            .iter()
                            .map(|coefficient| {
                                (coefficient * (value * 2u8 - 1u8) * &inverse_two) % modulus
                            })
                            .collect()
                    })
                })
                .collect(),
            Some(self.layout),
        )?;
        Ok((table, digit_bound))
    }

    /// Convolution gains of the declared integer lift of T', separate from
    /// the decomposition of T'G (not necessarily centered modulo the ring q).
    /// Native signed-monomial accumulator plaintexts only rotate/sign these
    /// coefficients, preserving both norms used for the switch helper error.
    pub(crate) fn target_plaintext_gain(
        &self,
        ring_dimension: usize,
    ) -> Result<crate::noise_utils::RightActionNoiseGain, ValueReadoutError> {
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
        Ok(crate::noise_utils::RightActionNoiseGain::from_polynomial_coefficients((0..width).map(
            |k| {
                coefficients.get(k).cloned().unwrap_or_default() * BigInt::from(2u8) - BigInt::one()
            },
        )))
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
}

fn is_scalar_shape(matrix_type: &mxx_ir_core::types::MatrixType) -> bool {
    matrix_type.rows == mxx_ir_core::IntExpr::constant(1) &&
        matrix_type.columns == mxx_ir_core::IntExpr::constant(1)
}

/// Serializes the table graph recursively, omitting allocation addresses and
/// source locations.  This makes independently rebuilt declarations stable
/// while retaining all table values, types, and argument structure.
pub(crate) fn canonical_table_declaration(table: &Mat) -> Result<Vec<u8>, ValueReadoutError> {
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

pub(crate) fn concrete_ring_dimension(ring: &Ring) -> Result<usize, ValueReadoutError> {
    ring.matrix_type((1, 1))
        .ring_dimension
        .evaluate(&ParamEnv::default())
        .map_err(|_| ValueReadoutError::NonConcreteRingParameter)?
        .to_usize()
        .ok_or(ValueReadoutError::NonConcreteRingParameter)
}

pub(crate) fn native_width(
    layout: &ExpLayout,
    ring_dimension: usize,
) -> Result<usize, ValueReadoutError> {
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

/// Readout is a cryptographic step following Program accumulation. Its
/// declaration is common, while carriers, cached witnesses and fresh online
/// masks belong to the scheme. Structural batches must remain structural;
/// callers must not enumerate members to adapt this interface.
///
/// This does not imply a decoder or noise-refresh capability. A scheme may
/// instead support row-valued terminal targets through the terminal trait.
pub trait ExponentLUTReadoutScheme: ExponentLUTEncodingScheme {
    type ReadoutArtifacts;
    /// Public setup material required to produce cached witnesses. This must
    /// not require online mask encodings or any runtime plaintext inspection.
    type ReadoutPreparation;
    type ReadoutPublicMasks;
    type ReadoutEncodingMasks;
    type ReadoutPublicOutput;
    type ReadoutEncodingOutput;
    type ReadoutError: std::error::Error + Send + Sync + 'static;

    fn prepare_readout(
        &self,
        readout: &ValueReadout,
        inputs: &crate::scheme::PublicProgramValue<Self>,
        preparation: &Self::ReadoutPreparation,
    ) -> Result<Self::ReadoutArtifacts, Self::ReadoutError>;

    fn project_readout(
        &self,
        readout: &ValueReadout,
        inputs: &crate::scheme::PublicProgramValue<Self>,
        artifacts: &Self::ReadoutArtifacts,
        masks: &Self::ReadoutPublicMasks,
    ) -> Result<Self::ReadoutPublicOutput, Self::ReadoutError>;

    fn evaluate_readout(
        &self,
        readout: &ValueReadout,
        inputs: &crate::scheme::EncodingProgramValue<Self>,
        artifacts: &Self::ReadoutArtifacts,
        masks: &Self::ReadoutEncodingMasks,
    ) -> Result<Self::ReadoutEncodingOutput, Self::ReadoutError>;
}

/// Terminal outputs are distinct from ordinary Program state. Preparation
/// consumes public material only; evaluation consumes the resulting prepared
/// operations. A scheme keeps outer output rounding inside finalization, so
/// summing raw terms never becomes a sum of independently rounded outputs.
/// Internal helper/public rounding remains part of per-term preparation.
pub trait ExponentLUTTerminalScheme: ExponentLUTEncodingScheme {
    type TerminalMaterial: mxx_dsl::GraphValue;
    type TerminalOnlineHelpers;
    type PreparedTerminal: mxx_dsl::GraphValue;
    type TerminalPublicAccumulator: mxx_dsl::GraphValue;
    type TerminalEncodingAccumulator: mxx_dsl::GraphValue;
    type TerminalPublicOutput;
    type TerminalEncodingOutput;
    type TerminalDescriptor;
    type TerminalBinding;

    /// Field bindings for one prepared term. Batch ordering and transport
    /// traversal belong to `PreparedTerminal`, not the backend.
    fn export_terminal_term(
        &self,
        term: &Self::PreparedTerminal,
        graph: DslContext,
        prefix: &str,
    ) -> Result<(DslContext, Self::TerminalDescriptor), Self::Error>;
    fn import_terminal_term(
        &self,
        descriptor: &Self::TerminalDescriptor,
        production: ProductionId,
        expected: &Self::TerminalBinding,
    ) -> Result<Self::PreparedTerminal, Self::Error>;
    fn export_terminal_output(
        &self,
        public: &Self::TerminalPublicOutput,
        graph: DslContext,
        prefix: &str,
    ) -> Result<DslContext, Self::Error>;
    /// Reopen the already-finalized public output. This must not aggregate
    /// public terms or repeat their public modulus switching.
    fn import_terminal_output(
        &self,
        terms: &[Self::PreparedTerminal],
        production: ProductionId,
        prefix: &str,
    ) -> Result<Self::TerminalPublicOutput, Self::Error>;

    fn prepare_terminal_term(
        &self,
        input: &Self::PublicKey,
        material: &Self::TerminalMaterial,
    ) -> Result<(Self::PreparedTerminal, Self::TerminalPublicAccumulator), Self::Error>;
    fn terminal_encoding_term(
        &self,
        input: &Self::Encoding,
        prepared: &Self::PreparedTerminal,
        helpers: &Self::TerminalOnlineHelpers,
    ) -> Result<Self::TerminalEncodingAccumulator, Self::Error>;
    /// Sum active members at their common input modulus without finalizing.
    /// Member storage is structural; the result represents one logical term.
    fn reduce_terminal_public(
        &self,
        members: mxx_dsl::Family<Self::TerminalPublicAccumulator>,
    ) -> Result<Self::TerminalPublicAccumulator, Self::Error>;
    fn reduce_terminal_encoding(
        &self,
        members: mxx_dsl::Family<Self::TerminalEncodingAccumulator>,
    ) -> Result<Self::TerminalEncodingAccumulator, Self::Error>;

    fn add_terminal_public(
        &self,
        lhs: Self::TerminalPublicAccumulator,
        rhs: Self::TerminalPublicAccumulator,
    ) -> Result<Self::TerminalPublicAccumulator, Self::Error>;
    fn add_terminal_encoding(
        &self,
        lhs: Self::TerminalEncodingAccumulator,
        rhs: Self::TerminalEncodingAccumulator,
    ) -> Result<Self::TerminalEncodingAccumulator, Self::Error>;
    fn finish_terminal_public(
        &self,
        raw: Self::TerminalPublicAccumulator,
    ) -> Result<Self::TerminalPublicOutput, Self::Error>;
    fn finish_terminal_encoding(
        &self,
        raw: Self::TerminalEncodingAccumulator,
        public: &Self::TerminalPublicOutput,
    ) -> Result<Self::TerminalEncodingOutput, Self::Error>;
}

#[derive(Debug, Error)]
pub enum TerminalError<E: std::error::Error + 'static> {
    #[error("terminal preparation requires at least one term")]
    Empty,
    #[error("terminal inputs do not match the prepared term count")]
    InputCount,
    #[error(transparent)]
    Scheme(E),
    #[error(transparent)]
    Graph(#[from] mxx_dsl::DslError),
}

/// Construct one terminal scope. Inputs are local arrays with matching counts;
/// scalars are captures. Independent projection stays inside the same loop,
/// while aggregate projection runs once after raw member reduction.
pub(crate) fn prepare_terminal_value<S: ExponentLUTTerminalScheme, O: GraphValue>(
    scheme: &S,
    input: &crate::scheme::PublicProgramValue<S>,
    material: &ProgramValue<S::TerminalMaterial, Family<S::TerminalMaterial>>,
    aggregate: bool,
    project: impl FnOnce(S::TerminalPublicAccumulator) -> Result<O, S::Error>,
) -> Result<
    (ProgramValue<S::PreparedTerminal, Family<S::PreparedTerminal>>, ProgramValue<O, Family<O>>),
    TerminalError<S::Error>,
> {
    let count = match (input, material) {
        (ProgramValue::Family(values), _) => Some(values.count().clone()),
        (_, ProgramValue::Family(values)) => Some(values.count().clone()),
        _ => None,
    };
    let Some(count) = count else {
        let (cache, raw) = scheme
            .prepare_terminal_term(&input.at(0), &material.at(0))
            .map_err(TerminalError::Scheme)?;
        return Ok((
            ProgramValue::Scalar(cache),
            ProgramValue::Scalar(project(raw).map_err(TerminalError::Scheme)?),
        ));
    };
    let mut error = None;
    if aggregate {
        let values = mxx_dsl::parallel(count, |index| {
            scheme.prepare_terminal_term(&input.at(&index), &material.at(index)).map_err(|source| {
                error = Some(source);
                DslError::Schema
            })
        })
        .map_err(|source| {
            error.map(TerminalError::Scheme).unwrap_or(TerminalError::Graph(source))
        })?;
        let cache = values.field(|value| value.0)?;
        let raw = scheme
            .reduce_terminal_public(values.field(|value| value.1)?)
            .map_err(TerminalError::Scheme)?;
        Ok((
            ProgramValue::Family(cache),
            ProgramValue::Scalar(project(raw).map_err(TerminalError::Scheme)?),
        ))
    } else {
        let values = mxx_dsl::parallel(count, |index| {
            scheme
                .prepare_terminal_term(&input.at(&index), &material.at(index))
                .and_then(|(cache, raw)| project(raw).map(|value| (cache, value)))
                .map_err(|source| {
                    error = Some(source);
                    DslError::Schema
                })
        })
        .map_err(|source| {
            error.map(TerminalError::Scheme).unwrap_or(TerminalError::Graph(source))
        })?;
        Ok((
            ProgramValue::Family(values.field(|value| value.0)?),
            ProgramValue::Family(values.field(|value| value.1)?),
        ))
    }
}

/// Evaluate the matching prepared scope without rebuilding public products or
/// helper lifts. Projection follows the same independent/aggregate ordering.
pub(crate) fn evaluate_terminal_value<S: ExponentLUTTerminalScheme, O: GraphValue>(
    scheme: &S,
    input: &crate::scheme::EncodingProgramValue<S>,
    prepared: &ProgramValue<S::PreparedTerminal, Family<S::PreparedTerminal>>,
    helpers: &S::TerminalOnlineHelpers,
    aggregate: bool,
    project: impl FnOnce(S::TerminalEncodingAccumulator) -> Result<O, S::Error>,
) -> Result<ProgramValue<O, Family<O>>, TerminalError<S::Error>> {
    let ProgramValue::Family(terms) = prepared else {
        let raw = scheme
            .terminal_encoding_term(&input.at(0), &prepared.at(0), helpers)
            .map_err(TerminalError::Scheme)?;
        return Ok(ProgramValue::Scalar(project(raw).map_err(TerminalError::Scheme)?));
    };
    let mut error = None;
    if aggregate {
        let raw = mxx_dsl::parallel(terms.count().clone(), |index| {
            scheme.terminal_encoding_term(&input.at(&index), &terms.at(index), helpers).map_err(
                |source| {
                    error = Some(source);
                    DslError::Schema
                },
            )
        })
        .map_err(|source| {
            error.map(TerminalError::Scheme).unwrap_or(TerminalError::Graph(source))
        })?;
        let raw = scheme.reduce_terminal_encoding(raw).map_err(TerminalError::Scheme)?;
        Ok(ProgramValue::Scalar(project(raw).map_err(TerminalError::Scheme)?))
    } else {
        let values = mxx_dsl::parallel(terms.count().clone(), |index| {
            scheme
                .terminal_encoding_term(&input.at(&index), &terms.at(index), helpers)
                .and_then(project)
                .map_err(|source| {
                    error = Some(source);
                    DslError::Schema
                })
        })
        .map_err(|source| {
            error.map(TerminalError::Scheme).unwrap_or(TerminalError::Graph(source))
        })?;
        Ok(ProgramValue::Family(values))
    }
}

/// One public output and its ordered prepared terms, bound to the scheme
/// instance that produced them. Terms are distinct logical stages, not an
/// enumeration of slots in a structural family. The common layer neither
/// inspects ciphertexts nor assumes their dimensions or modulus representation.
pub struct PreparedTerminal<'a, S: ExponentLUTTerminalScheme> {
    pub(crate) scheme: &'a S,
    pub(crate) terms: Vec<S::PreparedTerminal>,
    pub(crate) public: S::TerminalPublicOutput,
}

/// Ordered terminal artifacts and their cached finalized public output.
#[derive(Clone, Serialize, Deserialize)]
pub struct PreparedTerminalDescriptor<D> {
    prefix: String,
    terms: Vec<D>,
}

impl<'a, S: ExponentLUTTerminalScheme> PreparedTerminal<'a, S> {
    pub fn export(
        &self,
        mut graph: DslContext,
        prefix: &str,
    ) -> Result<(DslContext, PreparedTerminalDescriptor<S::TerminalDescriptor>), S::Error> {
        let mut terms = Vec::with_capacity(self.terms.len());
        for (index, term) in self.terms.iter().enumerate() {
            let (next, descriptor) =
                self.scheme.export_terminal_term(term, graph, &format!("{prefix}.term.{index}"))?;
            graph = next;
            terms.push(descriptor);
        }
        graph =
            self.scheme.export_terminal_output(&self.public, graph, &format!("{prefix}.output"))?;
        Ok((graph, PreparedTerminalDescriptor { prefix: prefix.into(), terms }))
    }

    pub fn import(
        scheme: &'a S,
        descriptor: &PreparedTerminalDescriptor<S::TerminalDescriptor>,
        production: ProductionId,
        expected: &[S::TerminalBinding],
    ) -> Result<Self, TerminalError<S::Error>> {
        if expected.is_empty() {
            return Err(TerminalError::Empty);
        }
        if expected.len() != descriptor.terms.len() {
            return Err(TerminalError::InputCount);
        }
        let terms = descriptor
            .terms
            .iter()
            .zip(expected)
            .map(|(data, binding)| scheme.import_terminal_term(data, production.clone(), binding))
            .collect::<Result<Vec<_>, _>>()
            .map_err(TerminalError::Scheme)?;
        let public = scheme
            .import_terminal_output(&terms, production, &format!("{}.output", descriptor.prefix))
            .map_err(TerminalError::Scheme)?;
        Ok(Self { scheme, terms, public })
    }

    pub fn prepared_terms(&self) -> &[S::PreparedTerminal] {
        &self.terms
    }

    pub fn new(
        scheme: &'a S,
        inputs: &[(&S::PublicKey, &S::TerminalMaterial)],
    ) -> Result<Self, TerminalError<S::Error>> {
        if inputs.is_empty() {
            return Err(TerminalError::Empty);
        }
        let mut terms = Vec::with_capacity(inputs.len());
        let raw = sum_terms(
            inputs.iter().map(|(input, material)| {
                let (prepared, raw) =
                    scheme.prepare_terminal_term(input, material).map_err(TerminalError::Scheme)?;
                terms.push(prepared);
                Ok(raw)
            }),
            |lhs, rhs| scheme.add_terminal_public(lhs, rhs).map_err(TerminalError::Scheme),
        )?;
        let public = scheme.finish_terminal_public(raw).map_err(TerminalError::Scheme)?;
        Ok(Self { scheme, terms, public })
    }

    pub fn public(&self) -> &S::TerminalPublicOutput {
        &self.public
    }

    pub fn evaluate(
        &self,
        inputs: &[(&S::Encoding, &S::TerminalOnlineHelpers)],
    ) -> Result<S::TerminalEncodingOutput, TerminalError<S::Error>> {
        if inputs.len() != self.terms.len() {
            return Err(TerminalError::InputCount);
        }
        let raw = sum_terms(
            inputs.iter().zip(&self.terms).map(|((input, helpers), term)| {
                self.scheme
                    .terminal_encoding_term(input, term, helpers)
                    .map_err(TerminalError::Scheme)
            }),
            |lhs, rhs| self.scheme.add_terminal_encoding(lhs, rhs).map_err(TerminalError::Scheme),
        )?;
        self.scheme.finish_terminal_encoding(raw, &self.public).map_err(TerminalError::Scheme)
    }
}

/// Build a balanced expression without retaining a second full vector of raw
/// carriers. Independent term graphs remain independent at execution; the
/// logarithmic merge stack bounds both addition depth and temporary handles.
pub(crate) fn sum_terms<T, E: std::error::Error + 'static>(
    terms: impl IntoIterator<Item = Result<T, TerminalError<E>>>,
    add: impl Fn(T, T) -> Result<T, TerminalError<E>>,
) -> Result<T, TerminalError<E>> {
    let mut levels = Vec::<Option<T>>::new();
    for term in terms {
        let mut value = term?;
        let mut level = 0;
        while level < levels.len() {
            let Some(left) = levels[level].take() else { break };
            value = add(left, value)?;
            level += 1;
        }
        if level == levels.len() {
            levels.push(Some(value));
        } else {
            levels[level] = Some(value);
        }
    }
    // Smaller final chunks are to the right. Fold from right to left to keep
    // the ordered stage identity unchanged for non-power-of-two term counts.
    let mut result = None;
    for value in levels.into_iter().flatten() {
        result = Some(match result {
            None => value,
            Some(right) => add(value, right)?,
        });
    }
    result.ok_or(TerminalError::Empty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;

    #[test]
    fn test_terminal_reduction_preserves_order_and_logarithmic_depth() {
        for count in 1usize..=33 {
            let (stages, depth) = sum_terms(
                (0..count).map(|index| Ok::<_, TerminalError<Infallible>>((vec![index], 0))),
                |(mut left, left_depth), (right, right_depth)| {
                    left.extend(right);
                    Ok((left, left_depth.max(right_depth) + 1))
                },
            )
            .unwrap();
            assert_eq!(stages, (0..count).collect::<Vec<_>>());
            assert_eq!(depth, count.next_power_of_two().trailing_zeros());
        }
    }
}

/// Public row polynomials indexed by the input exponent. Empty coefficient
/// lists mean zero polynomials; all lists empty represent a zero row. The
/// number of polynomial columns is fixed by the type.
/// The selected-row contract is X^(stride*u) with 0 <= u < width; an encrypted RHS
/// shifts u before lookup. General linear combinations use the linear table
/// extension and require their own output coefficient bound.
#[derive(Clone)]
pub struct RowLutTable<const C: usize> {
    pub(crate) values: Vec<[Vec<BigInt>; C]>,
    pub(crate) layout: ExpLayout,
    pub(crate) id: [u8; 32],
    coefficient_bound: BigUint,
}

impl<const C: usize> RowLutTable<C> {
    pub fn new(
        values: Vec<[Vec<BigInt>; C]>,
        layout: Option<ExpLayout>,
    ) -> Result<Self, ValueReadoutError>
    where
        [Vec<BigInt>; C]: Serialize,
    {
        if !values.len().is_power_of_two() {
            return Err(ValueReadoutError::InvalidTableLength);
        }
        let layout = layout
            .unwrap_or(ExpLayout { stride_log2: 0, width_log2: values.len().trailing_zeros() });
        if layout.width() != values.len() {
            return Err(ValueReadoutError::InvalidTableLength);
        }
        let coefficient_bound = values
            .iter()
            .flatten()
            .flatten()
            .map(|v| v.magnitude())
            .max()
            .cloned()
            .unwrap_or_default();
        let mut hash = Sha256::new();
        hash.update(b"exponent-lut-row-table");
        hash.update(
            serde_json::to_vec(&(layout, &values))
                .map_err(|_| ValueReadoutError::Canonicalization)?,
        );
        Ok(Self { values, layout, id: hash.finalize().into(), coefficient_bound })
    }

    /// Sum sigma(mono(-index))*row[index] using the table's native layout.
    /// Rows are lowered once by the caller and reused across branches. Their
    /// order and cardinality follow this table's construction contract.
    pub(crate) fn sieve_sum<T>(
        &self,
        ring: &Ring,
        sigma: usize,
        rows: &[T],
    ) -> Result<T, crate::ExponentLutError>
    where
        T: mxx_dsl::GraphValue + std::ops::Add<Output = T>,
        Mat: std::ops::Mul<T, Output = T>,
    {
        let terms = rows
            .iter()
            .enumerate()
            .map(|(index, row)| {
                Ok(self
                    .layout
                    .mono(ring, &-BigInt::from(index))
                    .map_err(|_| crate::ExponentLutError::InvalidLut)?
                    .ring_automorphism(sigma) *
                    row.clone())
            })
            .collect::<Result<Vec<_>, crate::ExponentLutError>>()?;
        crate::utils::balanced_sum_family(
            mxx_dsl::Family::pack(terms).map_err(|_| crate::ExponentLutError::InvalidLut)?,
        )
    }

    /// Bound for a selected table row, computed from public constants at setup.
    pub fn coefficient_bound(&self) -> &BigUint {
        &self.coefficient_bound
    }
    pub fn width(&self) -> usize {
        self.values.len()
    }
}
