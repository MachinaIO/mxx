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
    program::{AttributeKnowledge, Gate, ProgramInputId, ProgramStep, RhsId, WireId},
    scheme::{PreparedProgram, PreparedProgramDescriptor},
};
use mxx_ir_core::encoding::hash_canonical;

/// A native polynomial exponent: this encodes X^native_exponent, not an
/// unscaled selector. The application owns packing such as `lhs_width * bit`.
#[derive(Clone)]
pub struct RnsBggProgramInput {
    pub level: usize,
    pub native_exponent: BigInt,
}

/// Caller-owned domain separation. Use a fresh nonce for an independent setup;
/// reuse it only when deliberately reconstructing the same setup's hash sites.
/// The seed input is public hash-sampler entropy, not a payload or mask secret.
pub struct RnsBggSetupDomain {
    pub seed: Bytes,
    pub label: Vec<u8>,
    pub nonce: [u8; 32],
}

/// Scalar plaintext bindings. Every declared input and RHS must appear exactly
/// once. RHS exponents already include the Program's packing/stride.
pub struct RnsBggProgramSetup {
    pub inputs: BTreeMap<ProgramInputId, RnsBggProgramInput>,
    pub rhs: BTreeMap<RhsId, BigInt>,
    pub domain: RnsBggSetupDomain,
}

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

#[derive(Clone, Serialize, Deserialize)]
pub(super) struct GateRhsIdentity {
    pub(super) gate: ProgramGateInstance,
    pub(super) rhs: RhsId,
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
pub struct RnsBggPreparedProgram {
    pub(super) prepared: PreparedProgram<RnsBggScheme>,
    pub(super) inputs: BTreeMap<ProgramInputId, (RnsBggPublicKey, RnsBggInputMaterial)>,
    pub(super) rhs: Vec<GateRhsIdentity>,
}

fn digest(value: &impl Serialize) -> Result<[u8; 32], RnsBggError> {
    hash_canonical(value).map_err(|error| RnsBggError::Graph(error.to_string()))
}

fn program_error(error: impl std::fmt::Display) -> RnsBggError {
    RnsBggError::Graph(error.to_string())
}

fn scalar_levels(
    program: &ExponentLutProgram,
    inputs: &BTreeMap<ProgramInputId, usize>,
) -> Result<BTreeMap<WireId, usize>, RnsBggError> {
    if !inputs.keys().eq(program.inputs().keys()) {
        return Err(RnsBggError::Parameters("missing or extra Program input"));
    }
    let mut levels: BTreeMap<_, _> =
        program.input_wires().iter().map(|(input, wire)| (*wire, inputs[input])).collect();
    for step in program.steps() {
        let ProgramStep::Gate(gate) = step else {
            return Err(RnsBggError::Parameters(
                "program-driven setup does not support structural families",
            ));
        };
        let (output, level) = match gate {
            Gate::MulPrivate { lhs, output, .. } | Gate::Lut { input: lhs, output, .. } => {
                (*output, levels[lhs].checked_add(1).ok_or(RnsBggError::Context)?)
            }
            Gate::Add { lhs, rhs, output } |
            Gate::MulPublic { private: lhs, public: rhs, output } => {
                if levels[lhs] != levels[rhs] {
                    return Err(RnsBggError::Context);
                }
                (*output, levels[lhs])
            }
        };
        levels.insert(output, level);
    }
    Ok(levels)
}

impl RnsBggScheme {
    /// Prepare all four scalar gate kinds without caller-created helper maps.
    /// Multiplication/LUT advance one logical level; linear gates preserve it.
    /// The parameters already encode the CRT-drop cadence at each such level.
    /// Repeated RHS uses receive distinct level/instance-specific packages.
    /// A caller auditing average-case noise must likewise bind independent
    /// RHS noise sources per gate (ProgramBindings::gate_rhs), not assume that
    /// repeated plaintext declarations share one sampled ciphertext error.
    pub fn prepare_program(
        &self,
        program: ExponentLutProgram,
        secret: &RnsBggPayloadSecret,
        setup: RnsBggProgramSetup,
    ) -> Result<RnsBggPreparedProgram, RnsBggError> {
        let levels = scalar_levels(
            &program,
            &setup.inputs.iter().map(|(id, input)| (*id, input.level)).collect(),
        )?;
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
                        Some(program.lut(*lut_id).ok_or(RnsBggError::Lut)?),
                        None,
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

    /// Restore authenticated setup metadata; no public preparation is rerun.
    /// Runtime manifest validation remains an explicit graph validation step.
    pub fn import_program(
        &self,
        program: ExponentLutProgram,
        descriptor: &RnsBggProgramDescriptor,
        production: ProductionId,
        expected: &RnsBggProgramExpectation,
    ) -> Result<RnsBggPreparedProgram, RnsBggError> {
        if &descriptor.expectation()? != expected {
            return Err(RnsBggError::Context);
        }
        let levels = scalar_levels(
            &program,
            &descriptor.inputs.iter().map(|(id, value)| (*id, value.level)).collect(),
        )?;
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
        let expected_rhs: Vec<_> = prepared
            .program
            .gate_instances()
            .filter_map(|(gate, operation)| {
                if let Gate::MulPrivate { rhs, .. } = operation { Some((gate, *rhs)) } else { None }
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
    /// operation. The caller can append explicit test-only outputs before build.
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

    pub fn public_values(&self) -> &BTreeMap<WireId, (RnsBggPublicKey, Option<Mat>)> {
        self.prepared.public_values()
    }

    /// Encode the setup-bound inputs using this online mask secret, prepare
    /// mask-dependent helpers, and evaluate the frozen Program. No secret-key
    /// oracle, readout conversion, or noise/security check is performed here.
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
    use crate::{
        nand_circuit::CompiledNandCircuitDetails,
        program::{ExponentLutProgramBuilder, ProgramWireType},
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

    #[test]
    fn test_program_setup_levels_domains_and_export_import() {
        let (scheme, program, input, rhs) = fixture();
        let secret = scheme.sample_payload_secret().unwrap();
        let nonce = rand::random();
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
        let output = builder.select(builder.input_wire(input).unwrap(), rhs, values, 2).unwrap();
        builder.output(output).unwrap();
        let program = builder.build().unwrap();
        assert!(matches!(
            scalar_levels(&program, &BTreeMap::from([(input, 0)])),
            Err(RnsBggError::Parameters(
                "program-driven setup does not support structural families"
            ))
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
        let payload = scheme.payload_at(&secret, level).unwrap();
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
