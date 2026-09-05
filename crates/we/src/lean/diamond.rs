//! Actual Diamond frozen-IR proof artifacts for the supported circuit topology.
use crate::{
    diamond::DiamondWeCompiler,
    lean::numeric::{NumericCertificateInputs, render_numeric_certificate},
};
use mxx_ir_core::{
    artifact::{ProductionId, SpecHash, export_validated_manifest},
    lean::{ExportOptions, export},
    validate, validate_with_manifests,
};
use mxx_primitives::poly::dcrt::params::DCRTPolyParams;
use mxx_runtime::lean::{export_dcrt_layouts, render_backend_context};
use num_bigint::{BigInt, BigUint};
use std::{
    collections::BTreeMap,
    error::Error,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Clone, Debug)]
pub struct VerifiedDiamondCertificate {
    directory: PathBuf,
    numeric_bound: BigUint,
    radius: BigUint,
}

impl VerifiedDiamondCertificate {
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    pub fn numeric_bound(&self) -> &BigUint {
        &self.numeric_bound
    }

    pub fn radius(&self) -> &BigUint {
        &self.radius
    }
}

/// Check a fresh artifact from this exact candidate. A conservative numeric rejection is
/// distinct from export, unsupported-topology, compiler, and timeout errors. No fallback runs.
pub fn verify_diamond_certificate(
    parameters: &DCRTPolyParams,
    compiler: &DiamondWeCompiler,
    timeout: Duration,
) -> Result<Option<VerifiedDiamondCertificate>, Box<dyn Error>> {
    let directory = tempfile::Builder::new().prefix("mxx-diamond-certificate-").tempdir()?.keep();
    let artifact = export_diamond_certificate(parameters, compiler, &directory)?;
    if !artifact.numeric_pass() {
        tracing::info!(directory = %directory.display(), bound = %artifact.numeric_bound,
            radius = %artifact.radius, "Diamond capped numeric gate rejected candidate");
        return Ok(None);
    }
    crate::lean::check::check_generated_modules(&directory, timeout)
        .map_err(|error| format!("{error}; proof logs retained at {}", directory.display()))?;
    let log = fs::read_to_string(directory.join("Certificate.log"))?;
    let expected = "'DiamondCertificate.correctness' depends on axioms: \
        [propext, Classical.choice, Quot.sound]";
    if log.split_whitespace().collect::<String>() != expected.split_whitespace().collect::<String>()
    {
        return Err(
            format!("unexpected final theorem axiom report at {}", directory.display()).into()
        );
    }
    Ok(Some(VerifiedDiamondCertificate {
        directory,
        numeric_bound: artifact.numeric_bound,
        radius: artifact.radius,
    }))
}

#[derive(Clone, Debug)]
pub struct ExportedDiamondCertificate {
    pub directory: PathBuf,
    pub numeric_bound: BigUint,
    pub radius: BigUint,
}

impl ExportedDiamondCertificate {
    pub fn numeric_pass(&self) -> bool {
        self.numeric_bound < self.radius
    }
}

/// Export the same actual compiler graphs and backend parameters, never a substitute fixture.
/// Ring, gadget, injector, and circuit dimensions come from this compiler's parameters.
pub fn export_diamond_certificate(
    parameters: &DCRTPolyParams,
    compiler: &DiamondWeCompiler,
    directory: &Path,
) -> Result<ExportedDiamondCertificate, Box<dyn Error>> {
    let expected_digit_base =
        u32::try_from(compiler.config.batch_bits).ok().and_then(|bits| 1usize.checked_shl(bits));
    let expected_witness_width =
        compiler.config.input_count.checked_mul(compiler.config.batch_bits);
    if compiler.config.input_count == 0 ||
        compiler.config.batch_bits == 0 ||
        !expected_digit_base.is_some_and(|minimum| compiler.config.digit_base >= minimum) ||
        expected_witness_width != Some(compiler.shape.witness_width) ||
        compiler.shape.depth == 0 ||
        compiler.shape.max_layer_width == 0
    {
        return Err("unsupported Diamond proof topology: requires positive input_count, batch_bits, depth and width, digit_base>=2^batch_bits, and witness_width=input_count*batch_bits".into());
    }
    let layouts = export_dcrt_layouts([parameters])?;
    let layout = &layouts[0];
    if layout.modulus != compiler.config.modulus ||
        layout.ring_dimension as usize != compiler.config.ring_dimension ||
        layout.regular_digit_count != compiler.config.digit_count ||
        (BigInt::from(1u32) << layout.base_bits) != compiler.config.gadget_base
    {
        return Err("Diamond compiler and registered DCRT layout disagree".into());
    }
    fs::create_dir_all(directory)?;
    let protocol = compiler.protocol_decl()?;
    let declaration = protocol.protocol();
    let bindings = compiler.circuit_bindings()?;
    let encryption = declaration
        .stages()
        .iter()
        .find(|stage| stage.id.0 == "encrypt")
        .ok_or("encrypt stage missing")?;
    let producer = validate(&encryption.graph, &bindings)?;
    let placeholder = ProductionId { spec_hash: SpecHash([0; 32]), execution_nonce: [0; 32] };
    let manifests =
        BTreeMap::from([(placeholder.clone(), export_validated_manifest(placeholder, &producer)?)]);
    let backend = render_backend_context(&layouts, "Backend", "DiamondBackend")?;
    fs::write(directory.join("Backend.lean"), backend.source())?;
    fs::write(
        directory.join("DiamondProofParameters.lean"),
        format!(
            "namespace DiamondProofParameters\n\n\
             abbrev q : Nat := {}\n\
             abbrev n : Nat := {}\n\
             abbrev ell : Nat := {}\n\
             abbrev inner : Nat := 2 * (ell + 2)\n\
             abbrev D : Nat := 2 ^ ({} - 1)\n\
             abbrev a : Nat := ell * n * D\n\
             abbrev factor : Nat := 2 * a + 4\n\
             abbrev projection : Nat := inner * n\n\
             abbrev inputCount : Nat := {}\n\
             abbrev batchBits : Nat := {}\n\
             abbrev digitBase : Nat := {}\n\
             abbrev circuitDepth : Nat := {}\n\
             abbrev circuitWidth : Nat := {}\n\
             abbrev instanceWidth : Nat := {}\n\
             abbrev witnessSlots : Nat := inputCount * batchBits\n\
             abbrev stateCount : Nat := 1 + witnessSlots\n\
             abbrev metadataCount : Nat := circuitDepth * circuitWidth\n\
             abbrev basePoolCount : Nat := (inputCount + 1) * stateCount\n\
             abbrev transitionCount : Nat := inputCount * digitBase * stateCount\n\
             abbrev sampleCount : Nat := inputCount * digitBase\n\n\
             end DiamondProofParameters\n",
            layout.modulus,
            layout.ring_dimension,
            layout.regular_digit_count,
            layout.base_bits,
            compiler.config.input_count,
            compiler.config.batch_bits,
            compiler.config.digit_base,
            compiler.shape.depth,
            compiler.shape.max_layer_width,
            compiler.shape.instance_width,
        ),
    )?;
    let mut graphs = declaration
        .stages()
        .iter()
        .map(|stage| (format!("Stage_{}", stage.id.0), &stage.graph))
        .collect::<Vec<_>>();
    graphs.extend(
        declaration
            .bundle
            .requirements
            .iter()
            .enumerate()
            .map(|(index, requirement)| (format!("Requirement_{index}"), &requirement.graph)),
    );
    graphs.push(("Ideal".into(), &declaration.bundle.ideal.graph));
    let mut generated = BTreeMap::new();
    for (name, graph) in graphs {
        let validated = validate_with_manifests(graph, &bindings, &manifests)?;
        let artifact = export(
            &validated,
            &ExportOptions {
                namespace: name.clone(),
                module_name: name.clone(),
                backend_layouts: layouts.iter().map(|layout| layout.exporter_binding()).collect(),
                ..ExportOptions::default()
            },
        )?;
        fs::write(directory.join(format!("{name}.lean")), &artifact.source)?;
        generated.insert(name, artifact);
    }
    let roots = crate::lean::ExportedRoots {
        stages: declaration
            .stages()
            .iter()
            .map(|stage| {
                (
                    stage.id.clone(),
                    generated.remove(&format!("Stage_{}", stage.id.0)).expect("exported stage"),
                )
            })
            .collect(),
        requirements: (0..declaration.bundle.requirements.len())
            .map(|index| {
                generated.remove(&format!("Requirement_{index}")).expect("exported requirement")
            })
            .collect(),
        ideal: generated.remove("Ideal").expect("exported ideal"),
    };
    let claim = crate::lean::assemble_claim(&protocol, &roots, &bindings, &backend)?;
    fs::write(directory.join("Claim.lean"), claim)?;
    fs::write(
        directory.join("DiamondGateProof.lean"),
        include_str!("../../lean/templates/DiamondGateProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondEncryptedGateProof.lean"),
        include_str!("../../lean/templates/DiamondEncryptedGateProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondInjectorProof.lean"),
        include_str!("../../lean/templates/DiamondInjectorProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondSelectorProof.lean"),
        include_str!("../../lean/templates/DiamondSelectorProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondInitialStateProof.lean"),
        include_str!("../../lean/templates/DiamondInitialStateProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondTransitionProof.lean"),
        include_str!("../../lean/templates/DiamondTransitionProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondAccumulatedSecretProof.lean"),
        include_str!("../../lean/templates/DiamondAccumulatedSecretProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondIndexProof.lean"),
        include_str!("../../lean/templates/DiamondIndexProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondInjectorWitness.lean"),
        include_str!("../../lean/templates/DiamondInjectorWitness.lean"),
    )?;
    fs::write(
        directory.join("DiamondLayerProof.lean"),
        include_str!("../../lean/templates/DiamondLayerProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondIntegerInvariant.lean"),
        include_str!("../../lean/templates/DiamondIntegerInvariant.lean"),
    )?;
    fs::write(
        directory.join("DiamondSelectorWitness.lean"),
        include_str!("../../lean/templates/DiamondSelectorWitness.lean"),
    )?;
    fs::write(
        directory.join("DiamondEncodingRowProof.lean"),
        include_str!("../../lean/templates/DiamondEncodingRowProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondInputContractProof.lean"),
        include_str!("../../lean/templates/DiamondInputContractProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondPackingProof.lean"),
        include_str!("../../lean/templates/DiamondPackingProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondDecryptInjectorProof.lean"),
        include_str!("../../lean/templates/DiamondDecryptInjectorProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondWitnessPreimageProof.lean"),
        include_str!("../../lean/templates/DiamondWitnessPreimageProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondBoundedLayerProof.lean"),
        include_str!("../../lean/templates/DiamondBoundedLayerProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondBoundedLoopProof.lean"),
        include_str!("../../lean/templates/DiamondBoundedLoopProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondInjectorLoopProof.lean"),
        include_str!("../../lean/templates/DiamondInjectorLoopProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondBooleanGateProof.lean"),
        include_str!("../../lean/templates/DiamondBooleanGateProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondCircuitLayerProof.lean"),
        include_str!("../../lean/templates/DiamondCircuitLayerProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondCircuitRequirementProof.lean"),
        include_str!("../../lean/templates/DiamondCircuitRequirementProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondCircuitInitialProof.lean"),
        include_str!("../../lean/templates/DiamondCircuitInitialProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondFinalDecryptionProof.lean"),
        include_str!("../../lean/templates/DiamondFinalDecryptionProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondFinalPublicProof.lean"),
        include_str!("../../lean/templates/DiamondFinalPublicProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondFinalEncodingProof.lean"),
        include_str!("../../lean/templates/DiamondFinalEncodingProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondNumericProof.lean"),
        include_str!("../../lean/templates/DiamondNumericProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondClaimInjectorProof.lean"),
        include_str!("../../lean/templates/DiamondClaimInjectorProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondClaimStateProof.lean"),
        include_str!("../../lean/templates/DiamondClaimStateProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondClaimFinalProof.lean"),
        include_str!("../../lean/templates/DiamondClaimFinalProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondClaimCircuitProof.lean"),
        include_str!("../../lean/templates/DiamondClaimCircuitProof.lean"),
    )?;
    fs::write(
        directory.join("DiamondClaimCorrectnessProof.lean"),
        include_str!("../../lean/templates/DiamondClaimCorrectnessProof.lean"),
    )?;
    let q = layout.modulus.to_biguint().ok_or("negative modulus")?;
    if q < BigUint::from(4u32) {
        return Err("decoder modulus must be at least four".into());
    }
    let quarter = &q / 4u32;
    let half = &q / 2u32;
    let cap = [
        quarter.clone(),
        &q - 3u32 * &quarter,
        &half - &quarter + 1u32,
        3u32 * &quarter - &half + 1u32,
    ]
    .into_iter()
    .min()
    .ok_or("empty decoder radius")?;
    let numeric_inputs = NumericCertificateInputs {
        cap,
        n: layout.ring_dimension.into(),
        inner: compiler.config.input_config().state_columns()?.into(),
        ell: layout.regular_digit_count.into(),
        error_bound: compiler
            .config
            .error_max_coefficient_bound
            .to_biguint()
            .ok_or("negative E")?,
        preimage_bound: compiler
            .config
            .preimage_max_coefficient_bound
            .to_biguint()
            .ok_or("negative K")?,
        digit_bound: (BigUint::from(1u32) << layout.base_bits) / 2u32,
        injector_layers: compiler.config.input_count.into(),
        circuit_layers: compiler.shape.depth.into(),
    };
    let numeric = render_numeric_certificate(&numeric_inputs);
    fs::write(directory.join("NumericCertificate.lean"), &numeric.source)?;
    let numeric_gate = "DiamondGeneratedProof.cappedDiamondBound (MxxWe.decoderRadius q) n inner ell \
        GeneratedClaim.stage_0_params.diamond_error_max_coefficient_bound.toNat \
        GeneratedClaim.stage_0_params.diamond_preimage_max_coefficient_bound.toNat D \
        GeneratedClaim.stage_1_params.diamond_input_count.toNat \
        GeneratedClaim.stage_1_params.depth.toNat < MxxWe.decoderRadius q";
    let numeric_proof = format!(
        "by\n  have hvalue : {numeric_gate_left} = {bound} := by\n\
         \x20   norm_num only [GeneratedClaim.stage_0_params, GeneratedClaim.stage_1_params]\n\
         \x20   simp only [Int.toNat]\n\
         \x20   simpa [MxxWe.decoderRadius, MxxWe.quarter, MxxWe.half, q, n, ell, \
         DiamondProofParameters.inner, D] using DiamondNumericCertificate.numeric_bound\n\
         \x20 rw [hvalue]\n\
         \x20 norm_num [MxxWe.decoderRadius, MxxWe.quarter, MxxWe.half, q]",
        numeric_gate_left = numeric_gate.split(" < ").next().ok_or("numeric gate shape")?,
        bound = numeric.bound,
    );
    let certificate = if numeric.bound < numeric_inputs.cap {
        format!(
            "theorem numeric_gate : {numeric_gate} := {numeric_proof}\n\n\
             theorem correctness : GeneratedClaim.CorrectnessClaim :=\n\
               DiamondGeneratedProof.generated_claim_correctness_of_capped_gate numeric_gate\n\n\
             #print axioms correctness\n"
        )
    } else {
        format!("theorem numeric_rejection : ¬ ({numeric_gate}) := {numeric_proof}\n")
    };
    fs::write(
        directory.join("Certificate.lean"),
        format!(
            "import DiamondClaimCorrectnessProof\nimport NumericCertificate\n\n\
             open MxxWe DiamondGeneratedProof DiamondProofParameters\n\n\
             namespace DiamondCertificate\n\n\
             set_option maxRecDepth 8192\n\n\
             {certificate}\nend DiamondCertificate\n"
        ),
    )?;
    fs::write(
        directory.join("DiamondCircuitPublicProof.lean"),
        include_str!("../../lean/templates/DiamondCircuitPublicProof.lean"),
    )?;
    Ok(ExportedDiamondCertificate {
        directory: directory.to_path_buf(),
        numeric_bound: numeric.bound,
        radius: numeric_inputs.cap,
    })
}
