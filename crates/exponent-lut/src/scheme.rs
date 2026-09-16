//! Scheme-specific values and gate operations with one shared program traversal.
use crate::program::{
    ExponentLutProgram, ExponentProgramLowering, LutId, ProgramBindings, ProgramFamilyBindings,
    ProgramFamilyId, ProgramGateInstance, ProgramLowerError, ProgramLoweringTypes, ProgramStep,
    ProgramValidationError, ProgramValue, RhsId, WireId, lower_body, lower_gate,
};
use mxx_dsl::{Family, GraphValue, GraphValueSchema, Int};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, error::Error};

/// A cryptographic scheme chooses the concrete public keys, encodings and
/// prepared operations. Both interpreters implement the same four gates.
/// Keys and encodings are individual graph values. Program scopes own families
/// and captures; primitive operations never dispatch on a family representation.
pub trait ExponentLUTEncodingScheme {
    type PublicKey: GraphValue;
    type Encoding: GraphValue;
    type PublicRhs: GraphValue;
    type EncodingRhs: GraphValue;
    type PublicHelpers: GraphValue;
    type EncodingHelpers: GraphValue;
    type Cache: Default + GraphValue;
    type NoiseState;
    type Error: Error + Send + Sync + 'static;

    /// Public arithmetic model for one candidate, with the backend's explicit
    /// coordinate convention. It never contains sampled ciphertext coefficients.
    type NoiseModel<'a>
    where
        Self: 'a;
    type NoiseEvaluator<'a>: ExponentProgramLowering<Wire = Self::NoiseState>
    where
        Self: 'a;

    fn noise_evaluator<'a>(
        program: &'a ExponentLutProgram,
        model: Self::NoiseModel<'a>,
        ranges: &'a ProgramFamilyBindings,
    ) -> Self::NoiseEvaluator<'a>
    where
        Self: 'a;

    /// Apply the symbolic sum transfer, retaining shared-source multiplicities.
    fn reduce_noise<'a>(
        evaluator: &mut Self::NoiseEvaluator<'a>,
        value: &Self::NoiseState,
        range: &crate::program::FamilyRange,
        gate: ProgramGateInstance,
    ) -> Result<Self::NoiseState, <Self::NoiseEvaluator<'a> as ProgramLoweringTypes>::Error>
    where
        Self: 'a;

    /// Public encoding dimensions used to size PRF outputs and refresh work.
    /// Returns (polynomial count, coefficients per polynomial); no ciphertext
    /// evaluation or coefficient inspection is required.
    fn encoding_dimensions(
        &self,
        key: &Self::PublicKey,
    ) -> (mxx_ir_core::IntExpr, mxx_ir_core::IntExpr);

    type PublicEvaluator<'a>: ExponentProgramLowering
        + ProgramLoweringTypes<
            Wire = Self::PublicKey,
            Rhs = Self::PublicRhs,
            HelperSet = Self::PublicHelpers,
            Error = Self::Error,
        >
    where
        Self: 'a;
    type EncodingEvaluator<'a>: ExponentProgramLowering
        + ProgramLoweringTypes<
            Wire = Self::Encoding,
            Rhs = Self::EncodingRhs,
            HelperSet = Self::EncodingHelpers,
            Error = Self::Error,
        >
    where
        Self: 'a;

    /// Record newly prepared fields in `cache`. Read-only preparation in
    /// `shared_cache` stays a capture and must not be copied into that output.
    fn public_evaluator<'a>(
        &'a self,
        cache: &'a mut Self::Cache,
        shared_cache: &'a Self::Cache,
        ranges: &'a ProgramFamilyBindings,
    ) -> Self::PublicEvaluator<'a>;
    fn encoding_evaluator<'a>(
        &'a self,
        cache: &'a Self::Cache,
        shared_cache: &'a Self::Cache,
        ranges: &'a ProgramFamilyBindings,
    ) -> Self::EncodingEvaluator<'a>;

    /// Aggregate the additive fields with the common structural reduction and
    /// bind the resulting key metadata once. This is the Program aggregation
    /// boundary, not a family implementation of any primitive gate.
    fn reduce_public(
        &self,
        values: Family<Self::PublicKey>,
        gate: ProgramGateInstance,
        cache: &mut Self::Cache,
    ) -> Result<Self::PublicKey, Self::Error>;

    fn reduce_encoding(
        &self,
        values: Family<Self::Encoding>,
        gate: ProgramGateInstance,
        cache: &Self::Cache,
    ) -> Result<Self::Encoding, Self::Error>;
}

pub type NoiseBindings<'a, S> = ProgramBindings<
    <S as ExponentLUTEncodingScheme>::NoiseState,
    <<S as ExponentLUTEncodingScheme>::NoiseEvaluator<'a> as ProgramLoweringTypes>::Rhs,
    <<S as ExponentLUTEncodingScheme>::NoiseEvaluator<'a> as ProgramLoweringTypes>::HelperSet,
>;

/// The noise interpreter visits the same frozen gates and family instances as
/// public/online evaluation. The returned interpreter owns its recorded transfers.
pub fn trace_noise<'a, S: ExponentLUTEncodingScheme + 'a>(
    program: &'a ExponentLutProgram,
    model: S::NoiseModel<'a>,
    bindings: &NoiseBindings<'a, S>,
    ranges: &'a ProgramFamilyBindings,
) -> Result<
    (
        BTreeMap<WireId, S::NoiseState>,
        S::NoiseEvaluator<'a>,
        crate::noise_utils::NoiseDependencyGraph,
    ),
    ProgramLowerError<<S::NoiseEvaluator<'a> as ProgramLoweringTypes>::Error>,
>
where
    <S::NoiseEvaluator<'a> as ProgramLoweringTypes>::HelperSet: Clone,
{
    let evaluator = S::noise_evaluator(program, model, ranges);
    crate::noise_utils::trace_dependencies(program, bindings, ranges, evaluator, S::reduce_noise)
}

/// Wire, RHS, and helper family storage is paired with the external range it represents.
/// Its packed index is `external - represented.start`; the active scope must
/// be contained in that range by the preparation contract.
pub type PublicBindings<S> = ProgramBindings<
    <S as ExponentLUTEncodingScheme>::PublicKey,
    <S as ExponentLUTEncodingScheme>::PublicRhs,
    ProgramValue<
        <S as ExponentLUTEncodingScheme>::PublicHelpers,
        (Family<<S as ExponentLUTEncodingScheme>::PublicHelpers>, crate::program::FamilyRange),
    >,
    (Family<<S as ExponentLUTEncodingScheme>::PublicKey>, crate::program::FamilyRange),
    (Family<<S as ExponentLUTEncodingScheme>::PublicRhs>, crate::program::FamilyRange),
>;
pub type EncodingBindings<S> = ProgramBindings<
    <S as ExponentLUTEncodingScheme>::Encoding,
    <S as ExponentLUTEncodingScheme>::EncodingRhs,
    ProgramValue<
        <S as ExponentLUTEncodingScheme>::EncodingHelpers,
        (Family<<S as ExponentLUTEncodingScheme>::EncodingHelpers>, crate::program::FamilyRange),
    >,
    (Family<<S as ExponentLUTEncodingScheme>::Encoding>, crate::program::FamilyRange),
    (Family<<S as ExponentLUTEncodingScheme>::EncodingRhs>, crate::program::FamilyRange),
>;

/// Preprocessing owns the scheme and program that produced its cache. Online
/// evaluation cannot substitute a different scheme or a different program.
pub struct PreparedProgram<S: ExponentLUTEncodingScheme> {
    pub(crate) scheme: S,
    pub(crate) program: ExponentLutProgram,
    pub(crate) ranges: ProgramFamilyBindings,
    pub(crate) cache: S::Cache,
    pub(crate) shared_cache: S::Cache,
    pub(crate) family_caches: BTreeMap<ProgramFamilyId, Family<S::Cache>>,
    pub(crate) public_values: BTreeMap<WireId, S::PublicKey>,
}

/// Prepared Program artifact metadata. Schemas retain lexical sharing and key
/// identities; the setup supplies its scheme binding without adding an artifact
/// capability to the cryptographic gate interface.
#[derive(Clone, Serialize, Deserialize)]
pub struct PreparedProgramDescriptor<C, K, B> {
    program: crate::program::ExponentLutProgramId,
    binding: B,
    ranges: ProgramFamilyBindings,
    prefix: String,
    pub(crate) cache: C,
    shared_cache: C,
    family_caches: BTreeMap<ProgramFamilyId, mxx_dsl::FamilyType<C>>,
    public_values: BTreeMap<WireId, K>,
}

impl<C: Serialize, K: Serialize, B: Serialize> PreparedProgramDescriptor<C, K, B> {
    /// The trusted setup stores this identity separately from the descriptor.
    pub fn identity(&self) -> Result<[u8; 32], serde_json::Error> {
        let mut hash = Sha256::new();
        hash.update(b"exponent-lut-prepared-program");
        hash.update(serde_json::to_vec(self)?);
        Ok(hash.finalize().into())
    }
}

/// Build a complete public family scope. The callbacks only resolve scalar
/// bindings and project output fields; Program dispatch and cache lifetime
/// belong here. Every body-local prepared field that is needed online must be
/// part of the returned graph value. Shared preparation remains a capture.
/// The binding callback supplies only the read-only seed cache; the driver
/// owns a separate, initially empty cache for the body's prepared outputs.
pub(crate) fn prepare_family<S: ExponentLUTEncodingScheme, O: mxx_dsl::GraphValue>(
    scheme: &S,
    program: &ExponentLutProgram,
    step: usize,
    count: mxx_ir_core::IntExpr,
    ranges: &ProgramFamilyBindings,
    bind: impl FnOnce(
        mxx_dsl::Int,
    ) -> Result<
        (
            BTreeMap<WireId, S::PublicKey>,
            BTreeMap<RhsId, S::PublicRhs>,
            BTreeMap<LutId, S::PublicHelpers>,
            BTreeMap<ProgramGateInstance, S::PublicHelpers>,
            S::Cache,
        ),
        mxx_dsl::DslError,
    >,
    project: impl FnOnce(S::PublicKey, S::Cache) -> Result<O, mxx_dsl::DslError>,
) -> Result<mxx_dsl::Family<O>, ProgramLowerError<S::Error>> {
    let Some(ProgramStep::Family { id, body, .. }) = program.steps().get(step) else {
        return Err(ProgramValidationError::WidthMismatch.into());
    };
    let mut operation_error = None;
    let output = mxx_dsl::parallel(count, |index| {
        let (wires, rhs, helpers, gate_helpers, shared_cache) = bind(index)?;
        let mut cache = S::Cache::default();
        let value = lower_body(
            program,
            step,
            *id,
            body,
            wires,
            &rhs,
            (&helpers, Some(&gate_helpers)),
            &mut scheme.public_evaluator(&mut cache, &shared_cache, ranges),
        )
        .map_err(|error| {
            operation_error = Some(error);
            mxx_dsl::DslError::Schema
        })?;
        project(value, cache)
    });
    output.map_err(|error| operation_error.unwrap_or_else(|| error.into()))
}

/// Evaluate the same complete body using only its bound online inputs and
/// prepared scalar cache. The binder selects both local and shared preparation
/// at the actual member index; shared fields remain captures. No public
/// preparation is called by this driver.
pub(crate) fn evaluate_family<S: ExponentLUTEncodingScheme, O: mxx_dsl::GraphValue>(
    scheme: &S,
    program: &ExponentLutProgram,
    step: usize,
    count: mxx_ir_core::IntExpr,
    ranges: &ProgramFamilyBindings,
    bind: impl FnOnce(
        mxx_dsl::Int,
    ) -> Result<
        (
            BTreeMap<WireId, S::Encoding>,
            BTreeMap<RhsId, S::EncodingRhs>,
            BTreeMap<LutId, S::EncodingHelpers>,
            BTreeMap<ProgramGateInstance, S::EncodingHelpers>,
            S::Cache,
            S::Cache,
        ),
        mxx_dsl::DslError,
    >,
    project: impl FnOnce(S::Encoding) -> Result<O, mxx_dsl::DslError>,
) -> Result<mxx_dsl::Family<O>, ProgramLowerError<S::Error>> {
    let Some(ProgramStep::Family { id, body, .. }) = program.steps().get(step) else {
        return Err(ProgramValidationError::WidthMismatch.into());
    };
    let mut operation_error = None;
    let output = mxx_dsl::parallel(count, |index| {
        let (wires, rhs, helpers, gate_helpers, cache, shared_cache) = bind(index)?;
        let value = lower_body(
            program,
            step,
            *id,
            body,
            wires,
            &rhs,
            (&helpers, Some(&gate_helpers)),
            &mut scheme.encoding_evaluator(&cache, &shared_cache, ranges),
        )
        .map_err(|error| {
            operation_error = Some(error);
            mxx_dsl::DslError::Schema
        })?;
        project(value)
    });
    output.map_err(|error| operation_error.unwrap_or_else(|| error.into()))
}

impl<S: ExponentLUTEncodingScheme> PreparedProgram<S> {
    /// Persist all caches in their evaluation order, including metadata-only
    /// caches with no graph ports. Binding is supplied by the resolved setup.
    pub fn export<B>(
        &self,
        binding: B,
        graph: mxx_dsl::DslContext,
        prefix: &str,
    ) -> Result<
        (
            mxx_dsl::DslContext,
            PreparedProgramDescriptor<
                <S::Cache as GraphValue>::Schema,
                <S::PublicKey as GraphValue>::Schema,
                B,
            >,
        ),
        mxx_dsl::DslError,
    > {
        let graph = graph.public_output(
            prefix,
            (
                self.cache.clone(),
                self.shared_cache.clone(),
                self.family_caches.clone(),
                self.public_values.clone(),
            ),
        )?;
        Ok((
            graph,
            PreparedProgramDescriptor {
                program: self.program.id(),
                binding,
                ranges: self.ranges.clone(),
                prefix: prefix.to_owned(),
                cache: self.cache.schema(),
                shared_cache: self.shared_cache.schema(),
                family_caches: self.family_caches.schema(),
                public_values: self.public_values.schema(),
            },
        ))
    }

    /// Reopen caches without rerunning public gate preparation. The expected
    /// binding must come from the resolved scheme, never from the descriptor.
    pub fn import<B: Serialize + PartialEq>(
        binding: B,
        scheme: S,
        program: ExponentLutProgram,
        ranges: ProgramFamilyBindings,
        descriptor: &PreparedProgramDescriptor<
            <S::Cache as GraphValue>::Schema,
            <S::PublicKey as GraphValue>::Schema,
            B,
        >,
        production: mxx_ir_core::artifact::ProductionId,
        expected_identity: [u8; 32],
    ) -> Result<Self, ProgramLowerError<S::Error>>
    where
        <S::Cache as GraphValue>::Schema: Serialize,
        <S::PublicKey as GraphValue>::Schema: Serialize,
    {
        if descriptor
            .identity()
            .map_err(|error| ProgramValidationError::Artifact(error.to_string()))? !=
            expected_identity ||
            descriptor.program != program.id() ||
            descriptor.ranges != ranges ||
            descriptor.binding != binding
        {
            return Err(ProgramValidationError::Artifact(
                "prepared Program binding mismatch".into(),
            )
            .into());
        }
        let (cache, shared_cache, family_caches, public_values) = (
            descriptor.cache.clone(),
            descriptor.shared_cache.clone(),
            descriptor.family_caches.clone(),
            descriptor.public_values.clone(),
        )
            .artifact_input(
                production,
                &descriptor.prefix,
                mxx_ir_core::artifact::ArtifactConfidentiality::Public,
            )?;
        Ok(Self { scheme, program, ranges, cache, shared_cache, family_caches, public_values })
    }

    pub fn new(
        scheme: S,
        program: ExponentLutProgram,
        bindings: &PublicBindings<S>,
        ranges: ProgramFamilyBindings,
        shared_cache: Option<S::Cache>,
    ) -> Result<Self, ProgramLowerError<S::Error>> {
        bindings.validate_for_program(&program)?;
        let shared_cache = shared_cache.unwrap_or_default();
        let mut cache = S::Cache::default();
        let mut family_caches = BTreeMap::new();
        let mut public_values = program
            .input_wires()
            .iter()
            .map(|(input, wire)| {
                bindings
                    .inputs
                    .get(input)
                    .cloned()
                    .map(|value| (*wire, value))
                    .ok_or(ProgramValidationError::MissingRuntimeInput(*input))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        for (step, operation) in program.steps().iter().enumerate() {
            match operation {
                ProgramStep::Gate(operation) => {
                    let helpers = indexed_helpers(&bindings.helpers, &Int::constant(0));
                    let gate_helpers = (!bindings.gate_helpers.is_empty())
                        .then(|| indexed_helpers(&bindings.gate_helpers, &Int::constant(0)));
                    let (output, value) = lower_gate(
                        &program,
                        operation,
                        ProgramGateInstance::Gate { step },
                        &public_values,
                        |id| {
                            bindings
                                .rhs_for_gate(ProgramGateInstance::Gate { step }, id)
                                .ok_or(ProgramValidationError::MissingRuntimeRhs(id))
                        },
                        (&helpers, gate_helpers.as_ref()),
                        &mut scheme.public_evaluator(&mut cache, &shared_cache, &ranges),
                    )?;
                    public_values.insert(output, value);
                }
                ProgramStep::Family { id, body, output, .. } => {
                    let range = ranges.range(*id).ok_or(ProgramValidationError::WidthMismatch)?;
                    let prepared = prepare_family(
                        &scheme,
                        &program,
                        step,
                        range.count().clone(),
                        &ranges,
                        |index| {
                            let external = index.add(Int::evaluate(range.start().clone()));
                            let (wires, rhs) = body.bind_inputs(
                                |wire| {
                                    public_values
                                        .get(&wire)
                                        .cloned()
                                        .ok_or(mxx_dsl::DslError::Schema)
                                },
                                |family| {
                                    bindings
                                        .wire_values
                                        .get(&family)
                                        .map(|(values, represented)| {
                                            values.at(external
                                                .clone()
                                                .sub(Int::evaluate(represented.start().clone())))
                                        })
                                        .ok_or(mxx_dsl::DslError::Schema)
                                },
                                |rhs| {
                                    bindings
                                        .rhs_scalar(rhs)
                                        .cloned()
                                        .ok_or(mxx_dsl::DslError::Schema)
                                },
                                |family| {
                                    bindings
                                        .rhs_family_value(family)
                                        .map(|(values, represented)| {
                                            values.at(external
                                                .clone()
                                                .sub(Int::evaluate(represented.start().clone())))
                                        })
                                        .ok_or(mxx_dsl::DslError::Schema)
                                },
                            )?;
                            Ok((
                                wires,
                                rhs,
                                indexed_helpers(&bindings.helpers, &external),
                                indexed_helpers(&bindings.gate_helpers, &external),
                                shared_cache.clone(),
                            ))
                        },
                        |value, local_cache| Ok((value, local_cache)),
                    )?;
                    let members = prepared.field(|value| value.0)?;
                    family_caches.insert(*id, prepared.field(|value| value.1)?);
                    let gate = ProgramGateInstance::FamilyReduction { step };
                    let value = scheme
                        .reduce_public(members, gate, &mut cache)
                        .map_err(|source| ProgramLowerError::Operation { gate, source })?;
                    public_values.insert(*output, value);
                }
            }
        }
        Ok(Self { scheme, program, ranges, cache, shared_cache, family_caches, public_values })
    }

    pub fn public_values(&self) -> &BTreeMap<WireId, S::PublicKey> {
        &self.public_values
    }

    pub fn evaluate(
        &self,
        bindings: &EncodingBindings<S>,
    ) -> Result<BTreeMap<WireId, S::Encoding>, ProgramLowerError<S::Error>> {
        bindings.validate_for_program(&self.program)?;
        let mut values = self
            .program
            .input_wires()
            .iter()
            .map(|(input, wire)| {
                bindings
                    .inputs
                    .get(input)
                    .cloned()
                    .map(|value| (*wire, value))
                    .ok_or(ProgramValidationError::MissingRuntimeInput(*input))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        for (step, operation) in self.program.steps().iter().enumerate() {
            match operation {
                ProgramStep::Gate(operation) => {
                    let helpers = indexed_helpers(&bindings.helpers, &Int::constant(0));
                    let gate_helpers = (!bindings.gate_helpers.is_empty())
                        .then(|| indexed_helpers(&bindings.gate_helpers, &Int::constant(0)));
                    let (output, value) = lower_gate(
                        &self.program,
                        operation,
                        ProgramGateInstance::Gate { step },
                        &values,
                        |id| {
                            bindings
                                .rhs_for_gate(ProgramGateInstance::Gate { step }, id)
                                .ok_or(ProgramValidationError::MissingRuntimeRhs(id))
                        },
                        (&helpers, gate_helpers.as_ref()),
                        &mut self.scheme.encoding_evaluator(
                            &self.cache,
                            &self.shared_cache,
                            &self.ranges,
                        ),
                    )?;
                    values.insert(output, value);
                }
                ProgramStep::Family { id, body, output, .. } => {
                    let range =
                        self.ranges.range(*id).ok_or(ProgramValidationError::WidthMismatch)?;
                    let prepared =
                        self.family_caches.get(id).ok_or(ProgramValidationError::WidthMismatch)?;
                    let members = evaluate_family(
                        &self.scheme,
                        &self.program,
                        step,
                        range.count().clone(),
                        &self.ranges,
                        |index| {
                            let local_cache = prepared.at(&index);
                            let external = index.add(Int::evaluate(range.start().clone()));
                            let (wires, rhs) = body.bind_inputs(
                                |wire| values.get(&wire).cloned().ok_or(mxx_dsl::DslError::Schema),
                                |family| {
                                    bindings
                                        .wire_values
                                        .get(&family)
                                        .map(|(values, represented)| {
                                            values.at(external
                                                .clone()
                                                .sub(Int::evaluate(represented.start().clone())))
                                        })
                                        .ok_or(mxx_dsl::DslError::Schema)
                                },
                                |rhs| {
                                    bindings
                                        .rhs_scalar(rhs)
                                        .cloned()
                                        .ok_or(mxx_dsl::DslError::Schema)
                                },
                                |family| {
                                    bindings
                                        .rhs_family_value(family)
                                        .map(|(values, represented)| {
                                            values.at(external
                                                .clone()
                                                .sub(Int::evaluate(represented.start().clone())))
                                        })
                                        .ok_or(mxx_dsl::DslError::Schema)
                                },
                            )?;
                            Ok((
                                wires,
                                rhs,
                                indexed_helpers(&bindings.helpers, &external),
                                indexed_helpers(&bindings.gate_helpers, &external),
                                local_cache,
                                self.shared_cache.clone(),
                            ))
                        },
                        Ok,
                    )?;
                    let gate = ProgramGateInstance::FamilyReduction { step };
                    let value = self
                        .scheme
                        .reduce_encoding(members, gate, &self.cache)
                        .map_err(|source| ProgramLowerError::Operation { gate, source })?;
                    values.insert(*output, value);
                }
            }
        }
        Ok(values)
    }

    pub fn program(&self) -> &ExponentLutProgram {
        &self.program
    }
    pub fn cache(&self) -> &S::Cache {
        &self.cache
    }
}

fn indexed_helpers<K: Copy + Ord, H: GraphValue>(
    values: &BTreeMap<K, ProgramValue<H, (Family<H>, crate::program::FamilyRange)>>,
    index: &Int,
) -> BTreeMap<K, H> {
    values
        .iter()
        .map(|(id, value)| {
            (
                *id,
                match value {
                    ProgramValue::Scalar(value) => value.clone(),
                    ProgramValue::Family((values, represented)) => {
                        values.at(index.clone().sub(Int::evaluate(represented.start().clone())))
                    }
                },
            )
        })
        .collect()
}

pub type PublicProgramValue<S> = ProgramValue<
    <S as ExponentLUTEncodingScheme>::PublicKey,
    Family<<S as ExponentLUTEncodingScheme>::PublicKey>,
>;
pub type EncodingProgramValue<S> = ProgramValue<
    <S as ExponentLUTEncodingScheme>::Encoding,
    Family<<S as ExponentLUTEncodingScheme>::Encoding>,
>;

/// Public linear combinations of fixed ciphertext operands followed by cached
/// private products. These operations do not depend on PRF labels or one-hot
/// payload semantics; callers supply the required payload-domain contracts.
pub trait ExponentLUTFixedRhsScheme: ExponentLUTEncodingScheme {
    type FixedRhs;
    type PreparedFixedProduct: GraphValue<Schema: serde::Serialize>;

    /// Public configuration and field bindings authenticated with the prepared cache.
    type FixedProductBinding: serde::Serialize + PartialEq;

    fn fixed_product_binding(&self) -> Self::FixedProductBinding;

    fn combine_fixed_rhs(
        &self,
        rhs: &Self::FixedRhs,
        weights: &mxx_dsl::Family<mxx_dsl::Mat>,
        active: &crate::program::FamilyRange,
        id: [u8; 32],
    ) -> Result<Self::FixedRhs, Self::Error>;

    fn prepare_fixed_product(
        &self,
        input: &PublicProgramValue<Self>,
        rhs: &Self::FixedRhs,
        id: [u8; 32],
    ) -> Result<(PublicProgramValue<Self>, Self::PreparedFixedProduct), Self::Error>;

    fn evaluate_fixed_product(
        &self,
        input: &EncodingProgramValue<Self>,
        prepared: &Self::PreparedFixedProduct,
    ) -> Result<EncodingProgramValue<Self>, Self::Error>;
}
