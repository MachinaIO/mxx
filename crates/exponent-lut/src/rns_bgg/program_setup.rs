//! Program-driven setup for scalar RNS-BGG programs.
//!
//! The caller supplies native monomial exponents (including any packing/stride),
//! input levels, secrets, and a fresh setup identity root. Gate levels, sampling
//! domains, RHS packages, and LUT banks are derived from the frozen program.
//! Structural families are deliberately rejected here; the generic typed
//! ProgramBindings API remains the supported family preparation interface.
//! No filesystem, runtime execution, security estimation, or oracle is hidden here.

use super::*;
use crate::{
    program::{
        AttributeKnowledge, Gate, ProgramFamilyId, ProgramInputId, ProgramStep, RhsId, WireId,
    },
    scheme::{PreparedProgram, PreparedProgramDescriptor},
};
use mxx_ir_core::encoding::hash_canonical;
use num_traits::ToPrimitive;

/// A native polynomial exponent: this encodes X^native_exponent, not an
/// unscaled selector. The application owns packing such as `lhs_width * bit`.
#[derive(Clone)]
pub(crate) struct RnsBggProgramInput {
    /// Input key level selected by the caller and checked against parameters.
    pub level: usize,
    /// Native exponent of `X^b`; packing/stride is not applied a second time.
    pub native_exponent: BigInt,
}

/// Caller-owned domain separation. Use a fresh nonce for an independent setup;
/// reuse it only when deliberately reconstructing the same setup's hash sites.
/// The seed input is public hash-sampler entropy, not a payload or mask secret.
pub(crate) struct RnsBggSetupDomain {
    /// Public hash-sampler entropy supplied by the caller.
    pub seed: Bytes,
    /// Domain label identifying the setup role.
    pub label: Vec<u8>,
    /// Fresh setup identity nonce; reuse reconstructs the same sampling sites.
    pub nonce: [u8; 32],
}

/// Scalar plaintext bindings. Every declared input and RHS must appear exactly
/// once. RHS exponents already include the Program's packing/stride.
pub(crate) struct RnsBggProgramSetup {
    /// Exactly the Program's declared scalar inputs.
    pub inputs: BTreeMap<ProgramInputId, RnsBggProgramInput>,
    /// Exactly the Program's declared scalar RHS inputs.
    pub rhs: BTreeMap<RhsId, BigInt>,
    /// Caller-owned seed, label, and nonce used for domain-separated setup.
    pub domain: RnsBggSetupDomain,
}

/// Serialized prepared-Program descriptor used by the authenticated export
/// and import path. Its schema is owned by the prepared graph implementation.
pub(super) type ProgramDescriptor = PreparedProgramDescriptor<
    <RnsBggProgramCache as GraphValue>::Schema,
    <(RnsBggPublicKey, Option<Mat>) as GraphValue>::Schema,
    RnsBggArtifactBinding,
>;

#[derive(Clone, Serialize, Deserialize)]
struct InputDescriptor {
    payload_prefix: String,
    key: [u8; 32],
    level: usize,
}

/// Stable identity metadata for one ordered private-RHS gate use.
#[derive(Clone, Serialize, Deserialize)]
pub(super) struct GateRhsIdentity {
    /// Ordered root or family gate instance consuming this RHS package.
    pub(super) gate: ProgramGateInstance,
    /// Declared RHS identity referenced by the gate.
    pub(super) rhs: RhsId,
    /// Stable sampled package identity, not a secret or ciphertext payload.
    pub(super) identity: [u8; 32],
}

/// Typed export metadata for the prepared Program and every input contribution.
/// The descriptor never contains either secret key or an oracle output.
#[derive(Clone, Serialize, Deserialize)]
pub struct RnsBggProgramDescriptor {
    prepared: ProgramDescriptor,
    inputs: BTreeMap<ProgramInputId, InputDescriptor>,
    rhs: Vec<GateRhsIdentity>,
}

/// Store this trusted setup digest independently of untrusted descriptors.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RnsBggProgramExpectation([u8; 32]);

impl RnsBggProgramDescriptor {
    /// Capture at trusted export time, never from an untrusted imported file.
    pub fn expectation(&self) -> Result<RnsBggProgramExpectation, RnsBggError> {
        Ok(RnsBggProgramExpectation(digest(&("rns-bgg-program-bundle-v1", self))?))
    }
}

/// Public preprocessing and private input contributions, ready to export or
/// evaluate. The mask secret is supplied only at online graph construction.
pub(crate) struct RnsBggPreparedProgram {
    /// Prepared public graph and its authenticated descriptor state.
    pub(super) prepared: PreparedProgram<RnsBggScheme>,
    /// Private input material retained for the caller's online encoding step.
    pub(super) inputs: BTreeMap<ProgramInputId, (RnsBggPublicKey, RnsBggInputMaterial)>,
    /// Ordered RHS identities used to bind deterministic online helper reuse.
    pub(super) rhs: Vec<GateRhsIdentity>,
}

fn digest(value: &impl Serialize) -> Result<[u8; 32], RnsBggError> {
    hash_canonical(value).map_err(|error| RnsBggError::Graph(error.to_string()))
}

fn program_error(error: impl std::fmt::Display) -> RnsBggError {
    RnsBggError::Graph(error.to_string())
}

/// The operation class required by one ordered program gate.  This is purely
/// structural metadata: it never contains a secret, matrix, sampled helper,
/// or runtime artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum GateRequirement {
    /// Level-preserving addition of two wires.
    Add { lhs: WireId, rhs: WireId, output: WireId },
    /// Level-preserving multiplication by a public wire.
    MulPublic { private: WireId, public: WireId, output: WireId },
    /// Private RHS multiplication, which transfers to the next level.
    MulPrivate { lhs: WireId, rhs: RhsId, output: WireId },
    /// Unary LUT helper application, which transfers to the next level.
    Lut { input: WireId, lut_id: LutId, output: WireId },
}

/// One gate together with the scope identity and checked transfer levels used
/// by every setup/interpreter.  `None` means that a family member's level is
/// supplied by its bound runtime value rather than guessed by the host.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GateRequirementRecord {
    /// Root or local-family operation identity, preserving ordered scope.
    pub(crate) instance: ProgramGateInstance,
    /// Structural operation and its wire/RHS/LUT references.
    pub(crate) gate: GateRequirement,
    /// Input level, if statically known from verified metadata.
    pub(crate) input_level: Option<usize>,
    /// Output level after the checked operation transfer.
    pub(crate) output_level: Option<usize>,
    /// Whether the operation consumes a sampled LUT/RHS helper.
    pub(crate) helper_required: bool,
}

/// Structural requirements for one indexed family invocation. This record is
/// metadata only; it does not contain one entry per family member.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FamilyRequirementRecord {
    /// Enclosing Program step containing the family body.
    pub(crate) step: usize,
    /// Structural family identity, separate from local wire IDs.
    pub(crate) id: ProgramFamilyId,
    /// Declared structural capacity; no host member expansion occurs.
    pub(crate) capacity: usize,
    /// Verified active range metadata for this invocation.
    pub(crate) range: Option<FamilyRange>,
    /// Ordered body gate requirements, visited once.
    pub(crate) gates: Vec<GateRequirementRecord>,
    /// Root output wire receiving the family aggregation.
    pub(crate) output: WireId,
}

/// Verified levels for externally supplied indexed family values. The program
/// itself remains the source of family shape and stride facts; this metadata
/// supplies the runtime level facts needed to check linear gates without
/// treating an unknown member level as compatible with every other level.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ProgramFamilyInputMetadata {
    /// Verified levels for externally bound family wire members.
    pub(crate) wire_levels: BTreeMap<crate::program::WireFamilyId, usize>,
    /// Verified levels for externally bound family RHS members.
    pub(crate) rhs_levels: BTreeMap<crate::program::RhsFamilyId, usize>,
}

/// The single structural contract consumed by program-driven setup.  It
/// records declarations, level transfers, ordered gate identities, and family
/// scopes while deliberately excluding all cryptographic values and sampling.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProgramRequirements {
    /// Declaration-to-root-wire mapping.
    pub(crate) input_wires: BTreeMap<ProgramInputId, WireId>,
    /// Caller-provided scalar input levels.
    pub(crate) input_levels: BTreeMap<ProgramInputId, usize>,
    /// Statically known root wire levels.
    pub(crate) wire_levels: BTreeMap<WireId, usize>,
    /// Ordered root gate requirements.
    pub(crate) gates: Vec<GateRequirementRecord>,
    /// Symbolic family body requirements and verified ranges.
    pub(crate) families: Vec<FamilyRequirementRecord>,
    /// Output wires and their statically known levels.
    pub(crate) outputs: Vec<(WireId, Option<usize>)>,
}

fn requirement_gate(gate: &Gate) -> GateRequirement {
    match gate {
        Gate::Add { lhs, rhs, output } => {
            GateRequirement::Add { lhs: *lhs, rhs: *rhs, output: *output }
        }
        Gate::MulPublic { private, public, output } => {
            GateRequirement::MulPublic { private: *private, public: *public, output: *output }
        }
        Gate::MulPrivate { lhs, rhs, output } => {
            GateRequirement::MulPrivate { lhs: *lhs, rhs: *rhs, output: *output }
        }
        Gate::Lut { input, lut_id, output } => {
            GateRequirement::Lut { input: *input, lut_id: *lut_id, output: *output }
        }
    }
}

fn gate_input_level(
    gate: &GateRequirement,
    levels: &BTreeMap<WireId, Option<usize>>,
) -> Option<usize> {
    match gate {
        GateRequirement::Add { lhs, .. } | GateRequirement::MulPublic { private: lhs, .. } => {
            levels.get(lhs).copied().flatten()
        }
        GateRequirement::MulPrivate { lhs, .. } | GateRequirement::Lut { input: lhs, .. } => {
            levels.get(lhs).copied().flatten()
        }
    }
}

fn derive_scope_gates(
    gates: impl IntoIterator<Item = (ProgramGateInstance, Gate)>,
    mut levels: BTreeMap<WireId, Option<usize>>,
) -> Result<(Vec<GateRequirementRecord>, BTreeMap<WireId, Option<usize>>), RnsBggError> {
    let mut requirements = Vec::new();
    for (instance, gate) in gates {
        let required = requirement_gate(&gate);
        let input_level = gate_input_level(&required, &levels);
        let output = match &required {
            GateRequirement::Add { lhs, rhs, output } => {
                if let (Some(lhs), Some(rhs)) =
                    (levels.get(lhs).copied().flatten(), levels.get(rhs).copied().flatten()) &&
                    lhs != rhs
                {
                    return Err(RnsBggError::Context);
                }
                *output
            }
            GateRequirement::MulPublic { private, public, output } => {
                if let (Some(private), Some(public)) =
                    (levels.get(private).copied().flatten(), levels.get(public).copied().flatten()) &&
                    private != public
                {
                    return Err(RnsBggError::Context);
                }
                *output
            }
            GateRequirement::MulPrivate { output, .. } | GateRequirement::Lut { output, .. } => {
                *output
            }
        };
        let output_level = match &required {
            GateRequirement::MulPrivate { .. } | GateRequirement::Lut { .. } => input_level
                .map(|level| level.checked_add(1).ok_or(RnsBggError::Context))
                .transpose()?,
            GateRequirement::Add { lhs, .. } | GateRequirement::MulPublic { private: lhs, .. } => {
                levels.get(lhs).copied().flatten()
            }
        };
        levels.insert(output, output_level);
        requirements.push(GateRequirementRecord {
            instance,
            gate: required,
            input_level,
            output_level,
            helper_required: matches!(gate, Gate::MulPrivate { .. } | Gate::Lut { .. }),
        });
    }
    Ok((requirements, levels))
}

/// Derive the structural contract consumed by program-driven setup.
///
/// The scalar input levels and verified family metadata are caller/runtime
/// facts; the frozen Program remains the source of ordered gates, wire
/// declarations, family capacity, and local scope. The derivation performs no
/// sampling, matrix construction, secret handling, artifact I/O, or backend
/// execution. Missing or extra declarations and family metadata fail closed.
/// Family bodies are walked once in a local namespace, with
/// `FamilyProgram::bind_inputs` providing the root/local binding boundary;
/// members are represented symbolically rather than expanded on the host.
/// Private multiplication and LUT gates transfer to the next logical level,
/// while addition and public multiplication preserve their level. A Program
/// step index is never treated as a cryptographic level index.
pub(crate) fn derive_program_requirements(
    program: &ExponentLutProgram,
    input_levels: &BTreeMap<ProgramInputId, usize>,
    family_metadata: &ProgramFamilyBindings,
    family_inputs: &ProgramFamilyInputMetadata,
) -> Result<ProgramRequirements, RnsBggError> {
    if !input_levels.keys().eq(program.inputs().keys()) {
        return Err(RnsBggError::Parameters("missing or extra Program input"));
    }
    let required_family_ids = program
        .steps()
        .iter()
        .filter_map(|operation| match operation {
            ProgramStep::Family { id, .. } => Some(*id),
            ProgramStep::Gate(_) => None,
        })
        .collect::<std::collections::BTreeSet<_>>();
    let provided_family_ids = family_metadata
        .canonical_entries()
        .into_iter()
        .map(|(id, _)| id)
        .collect::<std::collections::BTreeSet<_>>();
    if provided_family_ids != required_family_ids {
        if required_family_ids.difference(&provided_family_ids).next().is_some() {
            return Err(RnsBggError::Parameters("missing family range metadata"));
        }
        return Err(RnsBggError::Parameters("missing or extra family range metadata"));
    }
    let mut required_wire_families = std::collections::BTreeSet::new();
    let mut required_rhs_families = std::collections::BTreeSet::new();
    for operation in program.steps() {
        if let ProgramStep::Family { body, .. } = operation {
            for source in body.wire_inputs().values() {
                if let crate::program::FamilySource::Member(id) = source {
                    required_wire_families.insert(*id);
                }
            }
            for source in body.rhs_inputs().values() {
                if let crate::program::FamilySource::Member(id) = source {
                    required_rhs_families.insert(*id);
                }
            }
        }
    }
    let provided_wire_families =
        family_inputs.wire_levels.keys().copied().collect::<std::collections::BTreeSet<_>>();
    if provided_wire_families != required_wire_families {
        if required_wire_families.difference(&provided_wire_families).next().is_some() {
            return Err(RnsBggError::Parameters("missing family wire level metadata"));
        }
        return Err(RnsBggError::Parameters("missing or extra family wire level metadata"));
    }
    let provided_rhs_families =
        family_inputs.rhs_levels.keys().copied().collect::<std::collections::BTreeSet<_>>();
    if provided_rhs_families != required_rhs_families {
        if required_rhs_families.difference(&provided_rhs_families).next().is_some() {
            return Err(RnsBggError::Parameters("missing family RHS level metadata"));
        }
        return Err(RnsBggError::Parameters("missing or extra family RHS level metadata"));
    }
    let mut levels: BTreeMap<WireId, Option<usize>> = program
        .input_wires()
        .iter()
        .map(|(input, wire)| (*wire, Some(input_levels[input])))
        .collect();
    let mut gates = Vec::new();
    let mut families = Vec::new();
    for (step, operation) in program.steps().iter().enumerate() {
        match operation {
            ProgramStep::Gate(gate) => {
                let (mut records, next) = derive_scope_gates(
                    [(ProgramGateInstance::Gate { step }, gate.clone())],
                    levels,
                )?;
                levels = next;
                gates.append(&mut records);
            }
            ProgramStep::Family { id, capacity, body, output, .. } => {
                let range = family_metadata
                    .range(*id)
                    .ok_or(RnsBggError::Parameters("missing family range metadata"))?;
                let range_capacity = range
                    .capacity()
                    .evaluate(&mxx_ir_core::ParamEnv::default())
                    .ok()
                    .and_then(|value| value.to_usize())
                    .ok_or(RnsBggError::Parameters("family capacity is not concrete"))?;
                if range_capacity != *capacity {
                    return Err(RnsBggError::Context);
                }
                // Validate the active interval itself before any family body
                // binding or later indexing. This rejects negative/empty or
                // overflowing ranges even when the declared capacity matches.
                validate_family_range(&range, &(*capacity).into())?;
                let (local_levels, local_rhs_levels) =
                    body.bind_inputs(
                        |root| levels.get(&root).copied().ok_or(RnsBggError::Context),
                        |family| {
                            family_inputs.wire_levels.get(&family).copied().map(Some).ok_or(
                                RnsBggError::Parameters("missing family wire level metadata"),
                            )
                        },
                        |rhs| {
                            if program.rhs_inputs().contains_key(&rhs) {
                                Ok(None)
                            } else {
                                Err(RnsBggError::Context)
                            }
                        },
                        |family| {
                            family_inputs
                                .rhs_levels
                                .get(&family)
                                .copied()
                                .map(Some)
                                .ok_or(RnsBggError::Parameters("missing family RHS level metadata"))
                        },
                    )?;
                if local_rhs_levels.len() != body.rhs_inputs().len() {
                    return Err(RnsBggError::Context);
                }
                let family_gates = body.gates().iter().enumerate().map(|(gate, value)| {
                    (ProgramGateInstance::FamilyGate { step, gate, family: *id }, value.clone())
                });
                let (family_gates, local_levels) = derive_scope_gates(family_gates, local_levels)?;
                let output_level = local_levels.get(&body.output()).copied().flatten();
                levels.insert(*output, output_level);
                families.push(FamilyRequirementRecord {
                    step,
                    id: *id,
                    capacity: *capacity,
                    range: Some(range.clone()),
                    gates: family_gates,
                    output: *output,
                });
            }
        }
    }
    let wire_levels: BTreeMap<WireId, usize> =
        levels.into_iter().filter_map(|(wire, level)| level.map(|level| (wire, level))).collect();
    let outputs =
        program.outputs().iter().map(|wire| (*wire, wire_levels.get(wire).copied())).collect();
    Ok(ProgramRequirements {
        input_wires: program.input_wires().clone(),
        input_levels: input_levels.clone(),
        wire_levels,
        gates,
        families,
        outputs,
    })
}

#[cfg(test)]
fn scalar_levels(
    program: &ExponentLutProgram,
    inputs: &BTreeMap<ProgramInputId, usize>,
) -> Result<BTreeMap<WireId, usize>, RnsBggError> {
    if program.steps().iter().any(|step| matches!(step, ProgramStep::Family { .. })) {
        return Err(RnsBggError::Parameters(
            "program-driven setup does not support structural families",
        ));
    }
    Ok(derive_program_requirements(
        program,
        inputs,
        &ProgramFamilyBindings::new(),
        &ProgramFamilyInputMetadata::default(),
    )?
    .wire_levels)
}

impl RnsBggScheme {
    /// Prepare all four scalar gate kinds without caller-created helper maps.
    ///
    /// This is intentionally a scalar-only setup entry point: structural
    /// family steps are rejected rather than silently expanded by the host.
    /// It first derives and validates the structural requirements, then builds
    /// a DSL graph containing sampled setup material. The method constructs
    /// graph values; it does not execute a backend, run a security estimator,
    /// or provide an oracle result. Secrets, setup seed/label/nonce, and native
    /// exponents remain caller-owned. Missing or extra scalar inputs/RHS
    /// bindings and level/stride mismatches are rejected before preparation.
    /// Multiplication/LUT advance one logical level; linear gates preserve it.
    /// The parameters already encode the CRT-drop cadence at each such level.
    /// Repeated RHS uses receive distinct level/instance-specific packages.
    /// A caller auditing average-case noise must likewise bind independent
    /// RHS noise sources per gate (ProgramBindings::gate_rhs), not assume that
    /// repeated plaintext declarations share one sampled ciphertext error.
    pub(crate) fn prepare_program(
        &self,
        program: ExponentLutProgram,
        secret: &RnsBggPayloadSecret,
        setup: RnsBggProgramSetup,
    ) -> Result<RnsBggPreparedProgram, RnsBggError> {
        if program.steps().iter().any(|step| matches!(step, ProgramStep::Family { .. })) {
            return Err(RnsBggError::Parameters(
                "program-driven setup does not support structural families",
            ));
        }
        let input_levels = setup.inputs.iter().map(|(id, input)| (*id, input.level)).collect();
        let requirements = derive_program_requirements(
            &program,
            &input_levels,
            &ProgramFamilyBindings::new(),
            &ProgramFamilyInputMetadata::default(),
        )?;
        let levels = requirements.wire_levels.clone();
        if !setup.rhs.keys().eq(program.rhs_inputs().keys()) {
            return Err(RnsBggError::Parameters("missing or extra Program RHS"));
        }
        let layout = program.layout().ok_or(RnsBggError::Lut)?;
        if layout.exponent.stride_log2 != 0 ||
            layout.ring_dimension_log2 !=
                self.parameters.level(0)?.ring_dimension().trailing_zeros()
        {
            return Err(RnsBggError::Parameters("incompatible Program layout"));
        }
        for level in levels.values() {
            self.parameters.level(*level)?;
        }
        for (id, input) in &setup.inputs {
            let stride = program
                .wire_type(program.input_wires()[id])
                .ok_or(RnsBggError::Context)?
                .stride_log2;
            if &input.native_exponent % (BigInt::one() << stride) != BigInt::zero() {
                return Err(RnsBggError::Parameters("input exponent violates its declared stride"));
            }
        }
        for (id, exponent) in &setup.rhs {
            let stride = program.rhs_exponent_scale_log2(*id).ok_or(RnsBggError::Context)?;
            if exponent % (BigInt::one() << stride) != BigInt::zero() {
                return Err(RnsBggError::Parameters("RHS exponent violates its declared stride"));
            }
        }
        let site = |role: &str, location: usize, level: usize| {
            digest(&(
                "rns-bgg-program-setup-v1",
                &setup.domain.label,
                setup.domain.nonce,
                program.id(),
                self.artifact_binding(),
                role,
                location,
                level,
            ))
        };
        // X^k is already a native exponent. Do not apply the Program stride or
        // binary-table packing a second time here.
        let monomial = |level, exponent: &BigInt| {
            ExpLayout { stride_log2: 0, width_log2: 0 }
                .mono(&self.ring(level)?, exponent)
                .map_err(program_error)
        };
        let mut inputs = BTreeMap::new();
        let mut bindings = crate::scheme::PublicBindings::<Self>::new();
        for (id, input) in &setup.inputs {
            let key_id = site("input-key", id.index(), input.level)?;
            let key = self.sample_input_key(
                input.level,
                setup.domain.seed.clone(),
                key_id.as_slice(),
                key_id,
            )?;
            let plaintext = monomial(input.level, &input.native_exponent)?;
            let material = self.prepare_input(secret, &key, plaintext.clone())?;
            let public = program
                .wire_type(program.input_wires()[id])
                .ok_or(RnsBggError::Context)?
                .knowledge ==
                AttributeKnowledge::Public;
            bindings = bindings.input(*id, (key.clone(), public.then_some(plaintext)));
            inputs.insert(*id, (key, material));
        }
        let mut rhs = Vec::new();
        for (instance, gate) in program.gate_instances() {
            let ProgramGateInstance::Gate { step } = instance else {
                return Err(RnsBggError::Parameters(
                    "program-driven setup does not support structural families",
                ));
            };
            match gate {
                Gate::MulPrivate { lhs, rhs: id, .. } => {
                    let level = levels[lhs];
                    let identity = site("rhs-plaintext", step, level)?;
                    let package = site("rhs-package", step, level)?;
                    let sampled = self.sample_private_rhs(
                        secret,
                        level,
                        RnsBggRhsPlaintext {
                            payload: monomial(level + 1, &setup.rhs[id])?,
                            id: identity,
                        },
                        setup.domain.seed.clone(),
                        package.as_slice(),
                        package,
                    )?;
                    bindings = bindings.gate_rhs(
                        instance,
                        *id,
                        RnsBggPublicRhs::Fixed(self.prepare_private_rhs(&sampled)?),
                    );
                    rhs.push(GateRhsIdentity { gate: instance, rhs: *id, identity });
                }
                Gate::Lut { input, lut_id, .. } => {
                    let level = levels[input];
                    let bank_id = site("lut-bank", step, level)?;
                    let material_id = site("lut-material", step, level)?;
                    let bank = self.sample_mask_bank(
                        level,
                        layout.exponent,
                        setup.domain.seed.clone(),
                        bank_id.as_slice(),
                        bank_id,
                    )?;
                    let material = self.sample_lut_material(
                        secret,
                        program.lut(*lut_id).ok_or(RnsBggError::Lut)?,
                        &bank,
                        setup.domain.seed.clone(),
                        material_id.as_slice(),
                        material_id,
                    )?;
                    bindings = bindings.gate_helper(instance, ProgramValue::Scalar(material));
                }
                Gate::Add { .. } | Gate::MulPublic { .. } => {}
            }
        }
        let prepared = PreparedProgram::new(
            self.clone(),
            program,
            &bindings,
            ProgramFamilyBindings::new(),
            None,
        )
        .map_err(program_error)?;
        // The derived schedule must agree with actual scheme preparation.
        if prepared.public_values().iter().any(|(wire, value)| value.0.level() != levels[wire]) {
            return Err(RnsBggError::Context);
        }
        Ok(RnsBggPreparedProgram { prepared, inputs, rhs })
    }

    /// Restore authenticated setup metadata without resampling public setup.
    ///
    /// The independently retained expectation must come from a trusted export,
    /// not from the descriptor being imported. Import checks the Program's
    /// ordered scalar inputs, RHS identities, and derived levels before it
    /// creates lazy artifact inputs. It restores graph metadata only; it does
    /// not execute a backend, rerun setup sampling, or substitute an oracle for
    /// runtime manifest validation.
    #[cfg(test)]
    pub(crate) fn import_program(
        &self,
        program: ExponentLutProgram,
        descriptor: &RnsBggProgramDescriptor,
        production: ProductionId,
        expected: &RnsBggProgramExpectation,
    ) -> Result<RnsBggPreparedProgram, RnsBggError> {
        if &descriptor.expectation()? != expected {
            return Err(RnsBggError::Context);
        }
        if program.steps().iter().any(|step| matches!(step, ProgramStep::Family { .. })) {
            return Err(RnsBggError::Parameters(
                "program-driven setup does not support structural families",
            ));
        }
        let input_levels = descriptor.inputs.iter().map(|(id, value)| (*id, value.level)).collect();
        let requirements = derive_program_requirements(
            &program,
            &input_levels,
            &ProgramFamilyBindings::new(),
            &ProgramFamilyInputMetadata::default(),
        )?;
        let levels = requirements.wire_levels;
        let prepared = PreparedProgram::import(
            self.fixed_product_binding(),
            self.clone(),
            program,
            ProgramFamilyBindings::new(),
            &descriptor.prepared,
            production.clone(),
            descriptor.prepared.identity().map_err(program_error)?,
        )
        .map_err(program_error)?;
        if !prepared.public_values().keys().eq(levels.keys()) {
            return Err(RnsBggError::Context);
        }
        let expected_rhs: Vec<_> = requirements
            .gates
            .iter()
            .filter_map(|record| match record.gate {
                GateRequirement::MulPrivate { rhs, .. } => Some((record.instance, rhs)),
                _ => None,
            })
            .collect();
        if descriptor.rhs.iter().map(|rhs| (rhs.gate, rhs.rhs)).collect::<Vec<_>>() != expected_rhs
        {
            return Err(RnsBggError::Context);
        }
        let mut inputs = BTreeMap::new();
        for (id, value) in &descriptor.inputs {
            // Reuse the key already imported by the prepared Program; only
            // its private payload contribution has a separate namespace.
            let key = prepared.public_values()[&prepared.program.input_wires()[id]].0.clone();
            if key.id() != &value.key || key.level() != value.level {
                return Err(RnsBggError::Context);
            }
            let ring = self.ring(value.level)?;
            let columns = self.parameters.component_columns(value.level)?;
            let row = RnsBggRow::new([std::array::from_fn(|column| {
                ring.artifact_input(
                    production.clone(),
                    format!("{}.{column}", value.payload_prefix),
                    (1, columns),
                    ArtifactConfidentiality::Private,
                )
            })])?;
            inputs.insert(
                *id,
                (
                    key,
                    RnsBggInputMaterial {
                        row,
                        level: value.level,
                        key_id: value.key,
                        context: self.context,
                    },
                ),
            );
        }
        if prepared
            .public_values()
            .iter()
            .any(|(wire, value)| levels.get(wire) != Some(&value.0.level()))
        {
            return Err(RnsBggError::Context);
        }
        Ok(RnsBggPreparedProgram { prepared, inputs, rhs: descriptor.rhs.clone() })
    }
}

impl RnsBggPreparedProgram {
    /// Append prepared public caches and private input contributions in one
    /// graph-building operation.
    ///
    /// The returned descriptor is export metadata and its expectation should be
    /// retained independently of untrusted storage. This method writes graph
    /// outputs for later artifact handling; it does not execute the backend or
    /// serialize secret keys.
    #[cfg(test)]
    pub fn export(
        &self,
        graph: DslContext,
        prefix: &str,
    ) -> Result<(DslContext, RnsBggProgramDescriptor), RnsBggError> {
        let (mut graph, prepared) =
            self.prepared.export(self.prepared.scheme.fixed_product_binding(), graph, prefix)?;
        let mut inputs = BTreeMap::new();
        for (id, (key, material)) in &self.inputs {
            let payload_prefix = format!("{prefix}.input.{}.payload", id.index());
            for column in 0..2 {
                graph = graph.private_output(
                    format!("{payload_prefix}.{column}"),
                    material.row.entry(0, column).clone(),
                )?;
            }
            inputs.insert(
                *id,
                InputDescriptor { payload_prefix, key: *key.id(), level: key.level() },
            );
        }
        Ok((graph, RnsBggProgramDescriptor { prepared, inputs, rhs: self.rhs.clone() }))
    }

    /// Borrow the prepared public wire values and their optional public rows.
    ///
    /// These values support graph construction and level/identity checks only;
    /// this accessor neither executes the graph nor exposes private input
    /// payloads.
    pub fn public_values(&self) -> &BTreeMap<WireId, (RnsBggPublicKey, Option<Mat>)> {
        self.prepared.public_values()
    }

    /// Build the online encoding graph for the frozen Program.
    ///
    /// Setup-bound inputs are encoded with the caller-owned mask secret and
    /// mask-dependent helpers are bound to the prepared graph. The result is a
    /// DSL graph value; backend execution, readout conversion, noise auditing,
    /// and secret-key oracle checks remain separate runtime operations.
    #[cfg(test)]
    pub fn evaluate(
        &self,
        secret: &RnsBggMaskSecret,
    ) -> Result<BTreeMap<WireId, (RnsBggEncoding, Option<Mat>)>, RnsBggError> {
        let mut bindings = crate::scheme::EncodingBindings::<RnsBggScheme>::new();
        for (id, (key, material)) in &self.inputs {
            let public =
                self.prepared.public_values()[&self.prepared.program.input_wires()[id]].1.clone();
            bindings =
                bindings.input(*id, (self.prepared.scheme.encode(secret, key, material)?, public));
        }
        for rhs in &self.rhs {
            bindings =
                bindings.gate_rhs(rhs.gate, rhs.rhs, RnsBggPublicRhs::Identity(rhs.identity));
        }
        bindings = bindings.gate_helpers(self.prepared.prepare_online_helpers(secret)?);
        self.prepared.evaluate(&bindings).map_err(program_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::{
        ExponentLutProgramBuilder, FamilyProgram, FamilyWireSource, ProgramWireType,
    };

    fn fixture() -> (RnsBggScheme, ExponentLutProgram, ProgramInputId, RhsId) {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|value| value.parse().unwrap())
            .unwrap_or(16);
        let (parameters, _) = RnsBggParameters::plan_fixture(&RnsBggPlanRequest {
            ring_dimension: n,
            widths: vec![2; 4],
            encoding_error_sigma: 4u8.into(),
            crt_bits: 30,
            terminal_depth: 1,
            max_crt_depth: 8,
            terminal_value_bound: 1u8.into(),
            operations_per_drop: 1,
        })
        .unwrap();
        let scheme = RnsBggScheme::new(parameters, rand::random());
        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(ExpLayout { stride_log2: 0, width_log2: 1 }, n as usize).unwrap();
        let input = builder
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let family = builder.rhs_family(1).unwrap();
        let rhs = builder.rhs_input(family, 2, 1).unwrap();
        let table = builder.register_lut(LutTable::unary(2, 2, vec![0, 1]).unwrap()).unwrap();
        let mut wire = builder.input_wire(input).unwrap();
        for _ in 0..2 {
            wire = builder.private_mul(wire, rhs).unwrap();
            wire = builder.lut(wire, table, None).unwrap();
        }
        builder.output(wire).unwrap();
        (scheme, builder.build().unwrap(), input, rhs)
    }

    fn setup(
        scheme: &RnsBggScheme,
        input: ProgramInputId,
        rhs: RhsId,
        nonce: [u8; 32],
    ) -> RnsBggProgramSetup {
        RnsBggProgramSetup {
            inputs: BTreeMap::from([(
                input,
                RnsBggProgramInput { level: 0, native_exponent: 1.into() },
            )]),
            rhs: BTreeMap::from([(rhs, 0.into())]),
            domain: RnsBggSetupDomain {
                seed: scheme.ring(0).unwrap().bytes_input("seed", 32),
                label: b"setup-test".to_vec(),
                nonce,
            },
        }
    }

    fn selection_family_fixture(
        capacity: usize,
    ) -> (
        ExponentLutProgram,
        ProgramInputId,
        crate::program::WireFamilyId,
        crate::program::RhsFamilyId,
    ) {
        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(ExpLayout { stride_log2: 0, width_log2: 1 }, 16).unwrap();
        let input = builder
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let rhs_family = builder.rhs_family(2).unwrap();
        let wire_family = builder.wire_family(2).unwrap();
        let output = builder
            .select(builder.input_wire(input).unwrap(), rhs_family, wire_family, capacity)
            .unwrap();
        builder.output(output).unwrap();
        (builder.build().unwrap(), input, wire_family, rhs_family)
    }

    #[test]
    fn test_program_setup_levels_domains_and_export_import() {
        let (scheme, program, input, rhs) = fixture();
        let secret = scheme.sample_payload_secret().unwrap();
        let nonce = rand::random();
        let requirements = derive_program_requirements(
            &program,
            &BTreeMap::from([(input, 0)]),
            &ProgramFamilyBindings::new(),
            &ProgramFamilyInputMetadata::default(),
        )
        .unwrap();
        assert_eq!(requirements.gates.len(), program.steps().len());
        assert_eq!(requirements.outputs[0].1, Some(4));
        assert!(requirements.gates.iter().enumerate().all(|(index, record)| {
            record.instance == ProgramGateInstance::Gate { step: index }
        }));
        assert_eq!(requirements.gates.iter().filter(|record| record.helper_required).count(), 4);
        let prepared = scheme
            .prepare_program(program.clone(), &secret, setup(&scheme, input, rhs, nonce))
            .unwrap();
        let levels = scalar_levels(&program, &BTreeMap::from([(input, 0)])).unwrap();
        assert_eq!(levels[&program.outputs()[0]], 4);
        assert_ne!(prepared.rhs[0].identity, prepared.rhs[1].identity);
        let first = &prepared.prepared.cache.gates[&ProgramGateInstance::Gate { step: 1 }];
        let second = &prepared.prepared.cache.gates[&ProgramGateInstance::Gate { step: 3 }];
        let (PreparedGate::Lut(first), PreparedGate::Lut(second)) = (first, second) else {
            panic!("LUT cache");
        };
        assert_ne!(first.mask_bank().id, second.mask_bank().id);
        assert_eq!(first.mask_bank().level, 1);
        assert_eq!(second.mask_bank().level, 3);
        let again = scheme
            .prepare_program(program.clone(), &secret, setup(&scheme, input, rhs, nonce))
            .unwrap();
        assert_eq!(prepared.rhs[0].identity, again.rhs[0].identity);
        let independent = scheme
            .prepare_program(program.clone(), &secret, setup(&scheme, input, rhs, rand::random()))
            .unwrap();
        assert_ne!(prepared.rhs[0].identity, independent.rhs[0].identity);

        let (graph, descriptor) = prepared.export(DslContext::new("setup"), "bundle").unwrap();
        let producer = graph.build().unwrap().validate(&ParamEnv::default()).unwrap();
        assert!(producer.source.outputs().keys().all(|name| !name.contains("oracle")));
        assert!(producer.source.outputs().keys().any(|name| name.starts_with("bundle.input.")));
        let expected = descriptor.expectation().unwrap();
        let encoded = serde_json::to_vec(&descriptor).unwrap();
        let decoded: RnsBggProgramDescriptor = serde_json::from_slice(&encoded).unwrap();
        let production = ProductionId {
            spec_hash: mxx_ir_core::encoding::spec_hash(&producer.source, &producer.bindings)
                .unwrap(),
            execution_nonce: rand::random(),
        };
        let reopened = scheme
            .import_program(program.clone(), &decoded, production.clone(), &expected)
            .unwrap();
        let mask = scheme.sample_mask_secret().unwrap();
        let values = reopened.evaluate(&mask).unwrap();
        assert_eq!(values[&program.outputs()[0]].0.level(), 4);
        let manifest =
            mxx_ir_core::artifact::export_validated_manifest(production.clone(), &producer)
                .unwrap();
        let catalog = mxx_ir_core::artifact::ArtifactCatalog::from_manifest(manifest).unwrap();
        DslContext::new("restored-online")
            .output("value", values[&program.outputs()[0]].0.clone())
            .unwrap()
            .build()
            .unwrap()
            .validate_with_catalog(&ParamEnv::default(), &catalog)
            .unwrap();
        let mut changed = decoded.clone();
        changed.rhs[0].identity[0] ^= 1;
        assert!(
            scheme
                .import_program(program.clone(), &changed, production.clone(), &expected)
                .is_err()
        );
        let mut changed = decoded.clone();
        changed.inputs.get_mut(&input).unwrap().level += 1;
        assert!(
            scheme
                .import_program(program.clone(), &changed, production.clone(), &expected)
                .is_err()
        );
        let wrong = RnsBggProgramExpectation(rand::random());
        assert!(scheme.import_program(program, &decoded, production, &wrong).is_err());
    }

    #[test]
    fn test_program_setup_rejects_missing_extra_bindings_and_level_mismatch() {
        let (scheme, program, input, rhs) = fixture();
        let secret = scheme.sample_payload_secret().unwrap();
        let mut missing = setup(&scheme, input, rhs, rand::random());
        missing.inputs.clear();
        assert!(scheme.prepare_program(program.clone(), &secret, missing).is_err());
        let mut extra = setup(&scheme, input, rhs, rand::random());
        extra.rhs.insert(RhsId::from_index(999), 0.into());
        assert!(scheme.prepare_program(program, &secret, extra).is_err());
        for public_mul in [false, true] {
            let mut builder = ExponentLutProgramBuilder::new();
            builder.set_layout(ExpLayout { stride_log2: 0, width_log2: 1 }, 16).unwrap();
            let private = builder
                .input(
                    2,
                    ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private },
                )
                .unwrap();
            let public = builder
                .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Public })
                .unwrap();
            let lhs = builder.input_wire(private).unwrap();
            let rhs = builder.input_wire(public).unwrap();
            let output = if public_mul {
                builder.mul_public(lhs, rhs).unwrap()
            } else {
                builder.add(lhs, rhs).unwrap()
            };
            builder.output(output).unwrap();
            let program = builder.build().unwrap();
            assert!(scalar_levels(&program, &BTreeMap::from([(private, 0), (public, 1)])).is_err());
            let levels =
                scalar_levels(&program, &BTreeMap::from([(private, 0), (public, 0)])).unwrap();
            assert_eq!(levels[&output], 0);
        }
    }

    #[test]
    fn test_program_setup_rejects_structural_family_before_sampling() {
        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(ExpLayout { stride_log2: 0, width_log2: 1 }, 16).unwrap();
        let input = builder
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let rhs = builder.rhs_family(2).unwrap();
        let values = builder.wire_family(2).unwrap();
        let output = builder.select(builder.input_wire(input).unwrap(), rhs, values, 4).unwrap();
        builder.output(output).unwrap();
        let program = builder.build().unwrap();
        let mut family_metadata = ProgramFamilyBindings::new();
        family_metadata
            .bind(ProgramFamilyId::from_index(0), FamilyRange::bounded(2, 2, 4).unwrap());
        let family_inputs = ProgramFamilyInputMetadata {
            wire_levels: BTreeMap::from([(crate::program::WireFamilyId::from_index(0), 1)]),
            rhs_levels: BTreeMap::from([(crate::program::RhsFamilyId::from_index(0), 0)]),
        };
        let requirements = derive_program_requirements(
            &program,
            &BTreeMap::from([(input, 0)]),
            &family_metadata,
            &family_inputs,
        )
        .unwrap();
        assert_eq!(requirements.families.len(), 1);
        assert_eq!(requirements.families[0].capacity, 4);
        assert_eq!(
            requirements.families[0].range,
            family_metadata.range(ProgramFamilyId::from_index(0)).cloned()
        );
        assert!(
            requirements.families[0]
                .gates
                .iter()
                .all(|record| matches!(record.instance, ProgramGateInstance::FamilyGate { .. }))
        );
        assert!(matches!(
            scalar_levels(&program, &BTreeMap::from([(input, 0)])),
            Err(RnsBggError::Parameters(
                "program-driven setup does not support structural families"
            ))
        ));
    }

    #[test]
    fn test_program_requirements_family_bind_inputs_and_metadata() {
        let (program, input, wire_family, rhs_family) = selection_family_fixture(4);
        let family_id = ProgramFamilyId::from_index(0);
        let mut ranges = ProgramFamilyBindings::new();
        ranges.bind(family_id, FamilyRange::bounded(2, 2, 4).unwrap());
        let family_inputs = ProgramFamilyInputMetadata {
            wire_levels: BTreeMap::from([(wire_family, 1)]),
            rhs_levels: BTreeMap::from([(rhs_family, 0)]),
        };
        let requirements = derive_program_requirements(
            &program,
            &BTreeMap::from([(input, 0)]),
            &ranges,
            &family_inputs,
        )
        .unwrap();
        assert_eq!(requirements.input_wires[&input], WireId::from_index(0));
        assert_eq!(requirements.families[0].capacity, 4);
        assert_eq!(requirements.families[0].range.as_ref().unwrap().start(), &2.into());
        assert_eq!(requirements.families[0].gates.len(), 2);
        assert!(requirements.families[0].gates.iter().all(|record| {
            matches!(record.instance, ProgramGateInstance::FamilyGate { family, .. } if family == family_id)
        }));

        let missing_wire = ProgramFamilyInputMetadata {
            wire_levels: BTreeMap::new(),
            rhs_levels: BTreeMap::from([(rhs_family, 0)]),
        };
        assert!(matches!(
            derive_program_requirements(
                &program,
                &BTreeMap::from([(input, 0)]),
                &ranges,
                &missing_wire
            ),
            Err(RnsBggError::Parameters("missing family wire level metadata"))
        ));
        let missing_rhs = ProgramFamilyInputMetadata {
            wire_levels: BTreeMap::from([(wire_family, 1)]),
            rhs_levels: BTreeMap::new(),
        };
        assert!(matches!(
            derive_program_requirements(
                &program,
                &BTreeMap::from([(input, 0)]),
                &ranges,
                &missing_rhs
            ),
            Err(RnsBggError::Parameters("missing family RHS level metadata"))
        ));
        let mut extra_ranges = ranges.clone();
        extra_ranges.bind(ProgramFamilyId::from_index(99), FamilyRange::full(1).unwrap());
        assert!(matches!(
            derive_program_requirements(
                &program,
                &BTreeMap::from([(input, 0)]),
                &extra_ranges,
                &family_inputs
            ),
            Err(RnsBggError::Parameters("missing or extra family range metadata"))
        ));
        let extra_wire = ProgramFamilyInputMetadata {
            wire_levels: BTreeMap::from([
                (wire_family, 1),
                (crate::program::WireFamilyId::from_index(99), 1),
            ]),
            rhs_levels: BTreeMap::from([(rhs_family, 0)]),
        };
        assert!(matches!(
            derive_program_requirements(
                &program,
                &BTreeMap::from([(input, 0)]),
                &ranges,
                &extra_wire
            ),
            Err(RnsBggError::Parameters("missing or extra family wire level metadata"))
        ));
        let extra_rhs = ProgramFamilyInputMetadata {
            wire_levels: BTreeMap::from([(wire_family, 1)]),
            rhs_levels: BTreeMap::from([
                (rhs_family, 0),
                (crate::program::RhsFamilyId::from_index(99), 0),
            ]),
        };
        assert!(matches!(
            derive_program_requirements(
                &program,
                &BTreeMap::from([(input, 0)]),
                &ranges,
                &extra_rhs
            ),
            Err(RnsBggError::Parameters("missing or extra family RHS level metadata"))
        ));
        let mut wrong_range = ProgramFamilyBindings::new();
        wrong_range.bind(family_id, FamilyRange::bounded(3, 2, 5).unwrap());
        assert!(matches!(
            derive_program_requirements(
                &program,
                &BTreeMap::from([(input, 0)]),
                &wrong_range,
                &family_inputs
            ),
            Err(RnsBggError::Context)
        ));

        let mut invalid_range = ProgramFamilyBindings::new();
        invalid_range.bind(family_id, FamilyRange::bounded(3, 2, 4).unwrap());
        assert!(matches!(
            derive_program_requirements(
                &program,
                &BTreeMap::from([(input, 0)]),
                &invalid_range,
                &family_inputs
            ),
            Err(RnsBggError::Context)
        ));

        let (larger, larger_input, larger_wire_family, larger_rhs_family) =
            selection_family_fixture(8);
        let mut larger_ranges = ProgramFamilyBindings::new();
        larger_ranges.bind(family_id, FamilyRange::bounded(0, 8, 8).unwrap());
        let larger_inputs = ProgramFamilyInputMetadata {
            wire_levels: BTreeMap::from([(larger_wire_family, 1)]),
            rhs_levels: BTreeMap::from([(larger_rhs_family, 0)]),
        };
        let larger_requirements = derive_program_requirements(
            &larger,
            &BTreeMap::from([(larger_input, 0)]),
            &larger_ranges,
            &larger_inputs,
        )
        .unwrap();
        assert_eq!(
            larger_requirements.families[0].gates.len(),
            requirements.families[0].gates.len()
        );
    }

    #[test]
    fn test_program_requirements_reject_member_level_mismatch_with_local_collision() {
        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(ExpLayout { stride_log2: 0, width_log2: 1 }, 16).unwrap();
        let first = builder.wire_family(2).unwrap();
        let second = builder.wire_family(2).unwrap();
        let local_lhs = WireId::from_index(0);
        let local_rhs = WireId::from_index(1);
        let local_output = WireId::from_index(2);
        let body = FamilyProgram::new(
            BTreeMap::from([
                (local_lhs, FamilyWireSource::Member(first)),
                (local_rhs, FamilyWireSource::Member(second)),
            ]),
            BTreeMap::new(),
            vec![Gate::Add { lhs: local_lhs, rhs: local_rhs, output: local_output }],
            local_output,
        )
        .unwrap();
        let output = builder.family(2, body).unwrap();
        builder.output(output).unwrap();
        let program = builder.build().unwrap();
        let mut ranges = ProgramFamilyBindings::new();
        ranges.bind(ProgramFamilyId::from_index(0), FamilyRange::bounded(1, 2, 2).unwrap());
        let family_inputs = ProgramFamilyInputMetadata {
            wire_levels: BTreeMap::from([(first, 0), (second, 1)]),
            rhs_levels: BTreeMap::new(),
        };
        assert!(matches!(
            derive_program_requirements(&program, &BTreeMap::new(), &ranges, &family_inputs),
            Err(RnsBggError::Context)
        ));
    }

    #[test]
    fn test_program_setup_nand_preprocessing_checkpoint() {
        use mxx_runtime::{artifact::FileArtifactStore, checkpoint::ArtifactCheckpoint};
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|value| value.parse().unwrap())
            .unwrap_or(16);
        let circuit = crate::NandCircuit::new(2, vec![[0, 1]], 2).unwrap();
        let compiled =
            circuit.compile(ExpLayout { stride_log2: 0, width_log2: 4 }, n as usize, 64).unwrap();
        let (parameters, _) = RnsBggParameters::plan_fixture(&RnsBggPlanRequest {
            ring_dimension: n,
            widths: (0..compiled.rhs_ids().len()).flat_map(|_| [8, 16]).collect(),
            encoding_error_sigma: 4u8.into(),
            crt_bits: 20,
            terminal_depth: 1,
            max_crt_depth: 16,
            terminal_value_bound: 1u8.into(),
            operations_per_drop: 1,
        })
        .unwrap();
        let scheme = RnsBggScheme::new(parameters.clone(), rand::random());
        let secret = scheme.sample_payload_secret().unwrap();
        let assignment = [usize::from(rand::random::<bool>()), usize::from(rand::random::<bool>())];
        let prepared = scheme
            .prepare_program(
                compiled.program().clone(),
                &secret,
                RnsBggProgramSetup {
                    inputs: BTreeMap::from([(
                        compiled.state_input(),
                        RnsBggProgramInput { level: 0, native_exponent: 2.into() },
                    )]),
                    rhs: compiled
                        .rhs_ids()
                        .iter()
                        .copied()
                        .zip(compiled.circuit_input_indices())
                        .map(|(id, input)| (id, BigInt::from(8 * assignment[*input])))
                        .collect(),
                    domain: RnsBggSetupDomain {
                        seed: scheme.ring(0).unwrap().bytes_input("seed", 32),
                        label: b"nand-checkpoint-test".to_vec(),
                        nonce: rand::random(),
                    },
                },
            )
            .unwrap();
        let (mut graph, descriptor) =
            prepared.export(DslContext::new("nand-preprocessing"), "nand").unwrap();
        // Mirror the integration test's explicitly separate oracle extras.
        let level = prepared.public_values()[&compiled.program().outputs()[0]].0.level();
        let payload = scheme.reduce_payload_secret_modulus(&secret, level).unwrap();
        for column in 0..2 {
            graph = graph
                .private_output(format!("oracle.{column}"), payload.entry(0, column).clone())
                .unwrap();
        }
        let graph = graph.build().unwrap().validate(&ParamEnv::default()).unwrap();
        std::fs::create_dir_all("test_data").unwrap();
        let directory = tempfile::tempdir_in("test_data").unwrap();
        let mut store = FileArtifactStore::new(directory.path()).unwrap();
        let mut backend =
            mxx_runtime::backend::poly::cpu_backend(parameters.levels().iter().cloned());
        let result = mxx_runtime::execute_in_session_with_config(
            &graph,
            &mut backend,
            BTreeMap::from([(
                "seed".into(),
                mxx_runtime::RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut store,
            rand::random(),
            mxx_runtime::ExecutionConfig::default(),
        )
        .unwrap();
        let expected = descriptor.expectation().unwrap();
        let checkpoint =
            ArtifactCheckpoint::from_execution(&result, &mut store, descriptor, expected).unwrap();
        let path = directory.path().join("metadata.json");
        checkpoint.write(&path).unwrap();
        drop(result);
        drop(store);
        let (store, checkpoint) = checkpoint.reopen(directory.path(), path).unwrap();
        let restored = scheme
            .import_program(
                compiled.program().clone(),
                checkpoint.descriptor(),
                checkpoint.production().clone(),
                checkpoint.expected_identity(),
            )
            .unwrap();
        let values = restored.evaluate(&scheme.sample_mask_secret().unwrap()).unwrap();
        DslContext::new("nand-restored-online")
            .output("value", values[&compiled.program().outputs()[0]].0.clone())
            .unwrap()
            .build()
            .unwrap()
            .validate_with_catalog(&ParamEnv::default(), checkpoint.catalog())
            .unwrap();
        drop(store);
    }
}
