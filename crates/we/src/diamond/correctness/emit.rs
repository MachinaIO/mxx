//! Checked-in/generated Lean file manifests.

use super::{
    super::parameter_search::DiamondSelectedParameters,
    bounds::{BoundData, BoundExpr, BoundParameter, DIAMOND_BOUND_SCHEMA_VERSION},
};
use crate::diamond::{
    DiamondDecryptionSemanticRefs, DiamondDecryptionSiteRefs, DiamondEncryptionSemanticRefs,
    DiamondStructuralSiteRefs,
    graph::{DECRYPTION_STAGE_NAME, DiamondGraphParams, ENCRYPTION_STAGE_NAME},
};
use mxx_bgg::BggTraceRole;
use mxx_gadgets::circuit::BooleanCircuitFamilyParams;
use mxx_ir_core::{
    ConcreteLinkedProgram, ConcreteNodePayload, ConcreteScope, ConcreteSemanticWireRef,
    FrozenValueRef, RenderedLeanProgram, StructuralValueRoute, ValidatedLinkedProgram,
    types::ConcreteWireType,
};
use num_bigint::{BigInt, BigUint};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fmt::Write as _,
    fs, io,
    path::{Component, Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmitMode {
    Write,
    Check,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeneratedLeanFile {
    pub path: PathBuf,
    pub contents: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub struct LeanFileManifest {
    files: BTreeMap<PathBuf, Vec<u8>>,
}

impl LeanFileManifest {
    pub fn new(files: impl IntoIterator<Item = GeneratedLeanFile>) -> Result<Self, ManifestError> {
        let mut manifest = Self::default();
        for file in files {
            validate_relative_path(&file.path)?;
            if manifest.files.insert(file.path, file.contents).is_some() {
                return Err(ManifestError::DuplicatePath);
            }
        }
        Ok(manifest)
    }

    pub fn files(&self) -> impl Iterator<Item = (&Path, &[u8])> {
        self.files.iter().map(|(path, bytes)| (path.as_path(), bytes.as_slice()))
    }

    pub fn get(&self, path: &Path) -> Option<&[u8]> {
        self.files.get(path).map(Vec::as_slice)
    }

    pub fn write(&self, root: &Path) -> Result<(), ManifestError> {
        for (relative, contents) in &self.files {
            let destination = root.join(relative);
            let parent = destination.parent().ok_or(ManifestError::InvalidPath)?;
            fs::create_dir_all(parent)?;
            let temporary = temporary_path(&destination);
            fs::write(&temporary, contents)?;
            if let Err(error) = fs::rename(&temporary, &destination) {
                let _ = fs::remove_file(&temporary);
                return Err(error.into());
            }
        }
        Ok(())
    }

    pub fn check(&self, root: &Path) -> Result<(), ManifestError> {
        for (relative, expected) in &self.files {
            let path = root.join(relative);
            let actual = fs::read(&path).map_err(|error| {
                if error.kind() == io::ErrorKind::NotFound {
                    ManifestError::Stale { path: relative.clone() }
                } else {
                    ManifestError::Io(error)
                }
            })?;
            if actual != *expected {
                return Err(ManifestError::Stale { path: relative.clone() });
            }
        }
        Ok(())
    }

    pub fn emit(&self, root: &Path, mode: EmitMode) -> Result<(), ManifestError> {
        match mode {
            EmitMode::Write => self.write(root),
            EmitMode::Check => self.check(root),
        }
    }
}

#[derive(Debug, Error)]
pub enum ManifestError {
    #[error("generated file path must be relative and contain no parent traversal")]
    InvalidPath,
    #[error("generated file paths must be unique")]
    DuplicatePath,
    #[error("generated file {path:?} differs from the checked-in bytes")]
    Stale { path: PathBuf },
    #[error(transparent)]
    Io(#[from] io::Error),
}

pub fn validate_relative_path(path: &Path) -> Result<(), ManifestError> {
    if path.as_os_str().is_empty() || path.is_absolute() {
        return Err(ManifestError::InvalidPath);
    }
    // Backslash is rejected too, so a manifest is safe when moved between
    // Unix and Windows rather than changing its interpretation.
    if path.to_string_lossy().contains('\\') {
        return Err(ManifestError::InvalidPath);
    }
    for component in path.components() {
        match component {
            Component::Normal(_) => {}
            Component::CurDir |
            Component::ParentDir |
            Component::RootDir |
            Component::Prefix(_) => return Err(ManifestError::InvalidPath),
        }
    }
    Ok(())
}

fn temporary_path(destination: &Path) -> PathBuf {
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
    let name = format!(
        ".{}.{}.{}.tmp",
        destination.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id(),
        stamp
    );
    destination.with_file_name(name)
}

#[derive(Clone, Copy, Debug)]
pub struct DiamondCandidateSemanticRefs<'a> {
    pub encryption: &'a DiamondEncryptionSemanticRefs,
    pub decryption: &'a DiamondDecryptionSemanticRefs,
}

#[derive(Clone, Debug)]
pub struct DiamondLeanClaimRequest<'a> {
    pub linked: &'a ValidatedLinkedProgram,
    pub program: &'a RenderedLeanProgram,
    pub parameters: &'a DiamondSelectedParameters,
    pub bound: &'a BoundData,
    pub refs: DiamondCandidateSemanticRefs<'a>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiamondLeanArtifact {
    pub manifest: LeanFileManifest,
    pub claim_instance_sha256: [u8; 32],
}

impl DiamondLeanArtifact {
    /// Write or byte-check every generated file at `root` according to `mode`.
    pub fn emit(&self, root: &Path, mode: EmitMode) -> Result<(), ManifestError> {
        self.manifest.emit(root, mode)
    }

    /// Return one canonical byte stream for the generated candidate claim.
    pub fn claim_source(&self) -> Option<&[u8]> {
        self.manifest.get(Path::new("MxxGenerated/DiamondCandidate.lean"))
    }
}

#[derive(Debug, Error)]
pub enum DiamondLeanError {
    #[error(
        "Diamond Lean claim generation is not available until the application theorem API is implemented: {0}"
    )]
    Infrastructure(String),
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    #[error("Diamond semantic role is invalid: {0}")]
    SemanticRole(String),
    #[error("Diamond bound data is invalid: {0}")]
    Bound(String),
}

pub fn emit_diamond_lean_correctness(
    request: &DiamondLeanClaimRequest<'_>,
    _mode: EmitMode,
) -> Result<DiamondLeanArtifact, DiamondLeanError> {
    validate_bound(request.bound)?;
    let canonical_bound =
        super::bounds::derive_output_noise_bound(request.linked, request.parameters)
            .map_err(|error| DiamondLeanError::Bound(error.to_string()))?;
    if request.bound != &canonical_bound {
        return Err(DiamondLeanError::Bound(
            "bound does not equal the canonical Diamond recurrence for this program and parameters"
                .to_owned(),
        ));
    }
    let claim = render_candidate(request)?;
    let claim_instance_sha256 = Sha256::digest(&claim).into();
    let mut files = request
        .program
        .modules
        .iter()
        .map(|module| GeneratedLeanFile {
            path: module.relative_path.clone(),
            contents: module.source.as_bytes().to_vec(),
        })
        .collect::<Vec<_>>();
    files.push(GeneratedLeanFile {
        path: PathBuf::from("MxxGenerated/DiamondCandidate.lean"),
        contents: claim,
    });
    let manifest = LeanFileManifest::new(files)?;
    Ok(DiamondLeanArtifact { manifest, claim_instance_sha256 })
}

fn render_candidate(request: &DiamondLeanClaimRequest<'_>) -> Result<Vec<u8>, DiamondLeanError> {
    let encryption = request.refs.encryption;
    let decryption = request.refs.decryption;
    let message = resolve_role(request.linked, "message", &encryption.message)?;
    let encryption_instance =
        resolve_role(request.linked, "encryption instance", &encryption.instance)?;
    let decryption_instance =
        resolve_role(request.linked, "decryption instance", &decryption.instance)?;
    let witness = resolve_role(request.linked, "witness", &decryption.witness)?;
    let noisy = resolve_role(request.linked, "noisy plaintext", &decryption.noisy_plaintext)?;
    let decoded = resolve_role(request.linked, "decoded", &decryption.decoded)?;
    let encryption_stage = require_stage(request.linked, message.stage, ENCRYPTION_STAGE_NAME)?;
    let decryption_stage =
        require_stage(request.linked, decryption_instance.stage, DECRYPTION_STAGE_NAME)?;
    let one_minus_run = resolve_frozen_ref(
        request.linked,
        decryption_stage,
        "decryption one-minus circuit",
        &decryption.sites.one_minus_circuit,
    )?;
    let projected_run = resolve_frozen_ref(
        request.linked,
        decryption_stage,
        "decryption projected difference",
        &decryption.sites.projected_difference,
    )?;
    let k_plus_run = resolve_frozen_ref(
        request.linked,
        decryption_stage,
        "decryption k-plus projection",
        &decryption.sites.k_plus_projection,
    )?;
    let noisy_run = resolve_frozen_ref(
        request.linked,
        decryption_stage,
        "decryption noisy plaintext",
        &decryption.sites.noisy_plaintext,
    )?;
    let decoder_coefficient_run = resolve_frozen_ref(
        request.linked,
        decryption_stage,
        "decoder coefficient",
        &decryption.sites.decoder_coefficient,
    )?;
    let decoder_lower_run = resolve_frozen_ref(
        request.linked,
        decryption_stage,
        "decoder lower comparison",
        &decryption.sites.decoder_lower_comparison,
    )?;
    let decoder_upper_run = resolve_frozen_ref(
        request.linked,
        decryption_stage,
        "decoder upper comparison",
        &decryption.sites.decoder_upper_comparison,
    )?;
    let decoder_lower_int_run = resolve_frozen_ref(
        request.linked,
        decryption_stage,
        "decoder lower bool-to-int",
        &decryption.sites.decoder_lower_bool_to_int,
    )?;
    let decoder_upper_int_run = resolve_frozen_ref(
        request.linked,
        decryption_stage,
        "decoder upper bool-to-int",
        &decryption.sites.decoder_upper_bool_to_int,
    )?;
    let decoder_sum_run = resolve_frozen_ref(
        request.linked,
        decryption_stage,
        "decoder sum",
        &decryption.sites.decoder_sum,
    )?;
    let decoder_decoded_run = resolve_frozen_ref(
        request.linked,
        decryption_stage,
        "decoder equals two",
        &decryption.sites.decoder_equals_two,
    )?;
    let decoder_quarter_run = resolve_frozen_ref(
        request.linked,
        decryption_stage,
        "decoder quarter",
        &decryption.sites.decoder_quarter,
    )?;
    let decoder_three_quarter_run = resolve_frozen_ref(
        request.linked,
        decryption_stage,
        "decoder three-quarter",
        &decryption.sites.decoder_three_quarter,
    )?;
    let decoder_two_run = resolve_frozen_ref(
        request.linked,
        decryption_stage,
        "decoder two",
        &decryption.sites.decoder_two,
    )?;
    let decoder_three_run = resolve_frozen_ref(
        request.linked,
        decryption_stage,
        "decoder three",
        &decryption.sites.decoder_three,
    )?;
    for (label, reference) in
        [("encryption instance", &encryption_instance), ("encryption message", &message)]
    {
        if reference.stage != encryption_stage {
            return Err(DiamondLeanError::SemanticRole(format!(
                "{label} resolved to stage {}, expected encryption stage {encryption_stage}",
                reference.stage
            )));
        }
    }
    for (label, reference) in [
        ("decryption instance", &decryption_instance),
        ("witness", &witness),
        ("noisy plaintext", &noisy),
        ("decoded", &decoded),
    ] {
        if reference.stage != decryption_stage {
            return Err(DiamondLeanError::SemanticRole(format!(
                "{label} resolved to stage {}, expected decryption stage {decryption_stage}",
                reference.stage
            )));
        }
    }
    validate_parameter_bindings(request, encryption_stage, decryption_stage)?;
    require_type("message", &message, |ty| matches!(ty, ConcreteWireType::Bool))?;
    require_type("decoded", &decoded, |ty| matches!(ty, ConcreteWireType::Bool))?;
    let instance_shape = common_int_family_shape(&encryption_instance, &decryption_instance)?;
    let witness_shape = int_family_shape("witness", &witness)?;
    let circuit_shape = &request.parameters.compiler.shape;
    let padded_shape = vec![circuit_shape.max_layer_width];
    if instance_shape != padded_shape {
        return Err(DiamondLeanError::SemanticRole(format!(
            "instance family shape must be {:?}, found {instance_shape:?}",
            padded_shape
        )));
    }
    if witness_shape != padded_shape {
        return Err(DiamondLeanError::SemanticRole(format!(
            "witness family shape must be {:?}, found {witness_shape:?}",
            padded_shape
        )));
    }
    let plaintext = match &noisy.wire_type {
        ConcreteWireType::Matrix(matrix) if matrix.rows == 1 && matrix.columns == 1 => matrix,
        other => {
            return Err(DiamondLeanError::SemanticRole(format!(
                "noisy plaintext must be a 1x1 matrix, found {other:?}"
            )));
        }
    };

    let mut source = format!(
        "import MxxWe.DiamondWE.DecoderTrace\nimport {}\n\nnamespace Mxx.We.Golden.DiamondWE\n\n",
        request.program.root_module
    );
    source.push_str(
        "noncomputable section\n\nabbrev program : Mxx.IR.Program := Mxx.Generated.program\n\n",
    );
    let data = &request.parameters;
    writeln!(source, "abbrev parametersData : Mxx.We.DiamondWE.ParametersData := {{").unwrap();
    writeln!(source, "  modulus := {}", data.modulus).unwrap();
    writeln!(source, "  ringDimension := {}", data.ring_dimension).unwrap();
    writeln!(source, "  inputCount := {}", data.compiler.config.input_count).unwrap();
    writeln!(source, "  digitBase := {}", data.compiler.config.digit_base).unwrap();
    writeln!(source, "  batchBits := {}", data.compiler.config.batch_bits).unwrap();
    writeln!(source, "  gadgetBase := {}", data.compiler.config.gadget_base).unwrap();
    writeln!(source, "  gadgetDigitCount := {}", data.compiler.config.digit_count).unwrap();
    writeln!(source, "  witnessWidth := {}", data.compiler.shape.witness_width).unwrap();
    writeln!(source, "  errorCutoff := {}", data.compiler.config.error_max_coefficient_bound)
        .unwrap();
    writeln!(
        source,
        "  preimageCutoff := {}\n}}",
        data.compiler.config.preimage_max_coefficient_bound
    )
    .unwrap();
    source.push_str("\ndef parameters : Mxx.We.DiamondWE.Parameters := {\n  data := parametersData\n  valid := by\n    norm_num [Mxx.We.DiamondWE.ParametersData.Valid]\n    constructor\n    · norm_num [Mxx.We.DiamondWE.decoderQuarter]\n    · norm_num [Mxx.We.DiamondWE.decoderQuarter]\n    · norm_num [Mxx.We.DiamondWE.decoderQuarter]\n}\n\n");
    writeln!(source, "def circuitShape : Mxx.Gadgets.LayeredBoolCircuitShape := {{").unwrap();
    writeln!(source, "  instanceWidth := {}", circuit_shape.instance_width).unwrap();
    writeln!(source, "  witnessWidth := {}", circuit_shape.witness_width).unwrap();
    writeln!(source, "  depth := {}", circuit_shape.depth).unwrap();
    writeln!(source, "  maxLayerWidth := {}\n}}\n", circuit_shape.max_layer_width).unwrap();

    render_external_input_ref(&mut source, "messageInput", request.linked, &message)?;
    render_external_input_ref(
        &mut source,
        "encryptionInstanceBitsInput",
        request.linked,
        &encryption_instance,
    )?;
    render_external_input_ref(
        &mut source,
        "decryptionInstanceBitsInput",
        request.linked,
        &decryption_instance,
    )?;
    render_external_input_ref(&mut source, "witnessBitsInput", request.linked, &witness)?;
    render_circuit_refs(
        &mut source,
        "encryption",
        request.linked,
        &encryption.circuit,
        circuit_shape,
    )?;
    render_circuit_refs(
        &mut source,
        "decryption",
        request.linked,
        &decryption.circuit,
        circuit_shape,
    )?;
    render_typed_ref(&mut source, "noisyPlaintextOutput", &noisy)?;
    render_typed_ref(&mut source, "decodedOutput", &decoded)?;

    render_encryption_sites(
        &mut source,
        encryption_stage,
        request.linked,
        &encryption.sites,
        request.parameters.compiler.config.input_count,
        request.parameters.compiler.config.batch_bits,
        request.parameters.compiler.config.witness_size().unwrap() + 1,
        request.parameters.compiler.config.digit_base,
        &request.parameters.compiler.config.error_max_coefficient_bound,
    )?;
    render_decryption_sites(
        &mut source,
        decryption_stage,
        request.linked,
        &decryption.sites,
        &request.parameters.modulus,
        request.parameters.compiler.config.input_count,
        request.parameters.compiler.config.batch_bits,
        request.parameters.compiler.config.witness_size().unwrap() + 1,
        request.parameters.compiler.config.digit_base,
    )?;

    render_bound(&mut source, request.bound)?;
    writeln!(source, "  value := {}", request.bound.value).unwrap();
    source.push_str("  evaluated := rfl\n}\n\n");
    writeln!(source, "noncomputable def candidate : Mxx.We.DiamondWE.Candidate := {{").unwrap();
    source.push_str("  program := program\n  parameters := parameters\n");
    source.push_str("  circuitShape := circuitShape\n");
    writeln!(source, "  plaintextMatrixType := {}", render_matrix_type(plaintext)).unwrap();
    source.push_str("  refs := {\n");
    source.push_str("    messageInput := messageInput\n    encryptionInstanceBitsInput := encryptionInstanceBitsInput\n");
    source.push_str("    decryptionInstanceBitsInput := decryptionInstanceBitsInput\n    witnessBitsInput := witnessBitsInput\n");
    source.push_str(
        "    encryptionCircuit := encryptionCircuit\n    decryptionCircuit := decryptionCircuit\n",
    );
    source.push_str("    encryptionCircuitOutput := encryptionCircuitOutput\n    decryptionCircuitOutput := decryptionCircuitOutput\n");
    source.push_str("    noisyPlaintextOutput := noisyPlaintextOutput\n    decodedOutput := decodedOutput\n  }\n");
    source.push_str("  bound := bound\n}\n\n");
    source.push_str(
        "def graphShape : Mxx.We.DiamondWE.Candidate.HasDiamondGraphShape candidate := {\n",
    );
    source.push_str(
        "  encryptionCircuit := encryptionCircuit\n  decryptionCircuit := decryptionCircuit\n",
    );
    source.push_str("  encryptionCircuitOutput := encryptionCircuitOutput\n  decryptionCircuitOutput := decryptionCircuitOutput\n");
    source.push_str("  refs_encryption_eq := rfl\n  refs_decryption_eq := rfl\n");
    source.push_str("  refs_encryption_output_eq := rfl\n  refs_decryption_output_eq := rfl\n  decoded_site_eq := by rfl\n  noisy_plaintext_site_eq := by rfl\n");
    source.push_str(
        "  encryptionSites := encryptionSites\n  decryptionSites := decryptionSites\n  bggTraceTemplate := by decide\n  bggTraceCoverage := by decide\n  bggOutputSelectorEdge := by decide\n}\n\n",
    );
    render_fuse_primitive_runs(
        &mut source,
        &one_minus_run,
        &projected_run,
        &k_plus_run,
        &noisy_run,
    );
    render_decoder_primitive_runs(
        &mut source,
        [
            ("coefficient", &decoder_coefficient_run),
            ("quarter", &decoder_quarter_run),
            ("three", &decoder_three_run),
            ("threeQuarter", &decoder_three_quarter_run),
            ("lower", &decoder_lower_run),
            ("upper", &decoder_upper_run),
            ("lowerInt", &decoder_lower_int_run),
            ("upperInt", &decoder_upper_int_run),
            ("sum", &decoder_sum_run),
            ("two", &decoder_two_run),
            ("decoded", &decoder_decoded_run),
        ],
    );
    source.push_str("end\n\nend Mxx.We.Golden.DiamondWE\n");
    let parameter_match = render_parameter_match(encryption_stage, decryption_stage);
    source = source.replace(
        "\nend\n\nend Mxx.We.Golden.DiamondWE\n",
        &format!(
            "\n{parameter_match}\n\ndef candidateValidity : Mxx.We.DiamondWE.CandidateValidity candidate graphShape := by\n  constructor\n  · exact parametersMatch\n  · rfl\n  · intro parameter\n    rfl\n  · decide\n\nend\n\nend Mxx.We.Golden.DiamondWE\n"
        ),
    );
    Ok(source.into_bytes())
}

fn render_fuse_primitive_runs(
    source: &mut String,
    one_minus: &ConcreteSemanticWireRef,
    projected: &ConcreteSemanticWireRef,
    k_plus: &ConcreteSemanticWireRef,
    noisy: &ConcreteSemanticWireRef,
) {
    let theorem = |site: &ConcreteSemanticWireRef| {
        format!(
            "Mxx.Generated.stage{}_scope{}_node{}ReachedPrimitiveRunFromPublicEval",
            site.stage, site.wire.scope, site.wire.node.0
        )
    };
    writeln!(
        source,
        "theorem fusePrimitiveRuns\n  \
{{oracle : Mxx.Runtime.RuntimeGadgetOracle}}\n  \
(env : Mxx.We.DiamondWE.RuntimeEvalEnv oracle candidate.program.data)\n  \
(trace : Mxx.We.DiamondWE.RuntimeTrace oracle)\n  \
(success : Mxx.IR.eval (Mxx.Runtime.RuntimeBackend oracle) candidate.program env = .ok trace) :\n  \
Nonempty (Mxx.We.DiamondWE.CandidateFusePrimitiveRuns oracle candidate graphShape trace) := by\n  \
let generatedEnv : Mxx.IR.EvalEnv (Mxx.Runtime.RuntimeBackend oracle)\n    \
Mxx.Generated.linkedProgramData := by simpa [candidate, program] using env\n  \
have generatedSuccess : Mxx.IR.eval (Mxx.Runtime.RuntimeBackend oracle)\n    \
Mxx.Generated.program generatedEnv = .ok trace := by\n    \
simpa [candidate, program, generatedEnv] using success\n  \
obtain ⟨oneMinusRun⟩ := {} generatedEnv trace generatedSuccess\n  \
obtain ⟨projectedRun⟩ := {} generatedEnv trace generatedSuccess\n  \
obtain ⟨kPlusRun⟩ := {} generatedEnv trace generatedSuccess\n  \
obtain ⟨noisyRun⟩ := {} generatedEnv trace generatedSuccess\n  \
refine ⟨{{\n    stage := {}\n    scope := {}\n    oneMinusStage := by decide\n    projectedStage := by decide\n    kPlusStage := by decide\n    noisyStage := by decide\n    oneMinusScope := by decide\n    projectedScope := by decide\n    kPlusScope := by decide\n    noisyScope := by decide\n    oneMinusArguments := by decide\n    projectedArguments := by decide\n    kPlusArguments := by decide\n    noisyArguments := by decide\n    oneMinusOutputWire := by decide\n    projectedOutputWire := by decide\n    kPlusOutputWire := by decide\n    oneMinus := ?_\n    projected := ?_\n    kPlus := ?_\n    noisy := ?_\n  }}⟩\n  \
all_goals first\n  \
| simpa [graphShape, decryptionSites, decryptionOneMinusCircuit,\n      \
decryptionProjectedDifference, decryptionKPlusProjection,\n      \
decryptionNoisyPlaintext, Mxx.We.DiamondWE.StoredNodeRef.concreteNode] using oneMinusRun\n  \
| simpa [graphShape, decryptionSites, decryptionOneMinusCircuit,\n      \
decryptionProjectedDifference, decryptionKPlusProjection,\n      \
decryptionNoisyPlaintext, Mxx.We.DiamondWE.StoredNodeRef.concreteNode] using projectedRun\n  \
| simpa [graphShape, decryptionSites, decryptionOneMinusCircuit,\n      \
decryptionProjectedDifference, decryptionKPlusProjection,\n      \
decryptionNoisyPlaintext, Mxx.We.DiamondWE.StoredNodeRef.concreteNode] using kPlusRun\n  \
| simpa [graphShape, decryptionSites, decryptionOneMinusCircuit,\n      \
decryptionProjectedDifference, decryptionKPlusProjection,\n      \
decryptionNoisyPlaintext, Mxx.We.DiamondWE.StoredNodeRef.concreteNode] using noisyRun\n",
        theorem(one_minus),
        theorem(projected),
        theorem(k_plus),
        theorem(noisy),
        one_minus.stage,
        one_minus.wire.scope,
    )
    .unwrap();
}

fn render_decoder_primitive_runs(
    source: &mut String,
    runs: [(&str, &ConcreteSemanticWireRef); 11],
) {
    let coefficient = runs[0].1;
    writeln!(
        source,
        "theorem decoderPrimitiveRuns\n  \
{{oracle : Mxx.Runtime.RuntimeGadgetOracle}}\n  \
(env : Mxx.We.DiamondWE.RuntimeEvalEnv oracle candidate.program.data)\n  \
(trace : Mxx.We.DiamondWE.RuntimeTrace oracle)\n  \
(success : Mxx.IR.eval (Mxx.Runtime.RuntimeBackend oracle) candidate.program env = .ok trace) :\n  \
Nonempty (Mxx.We.DiamondWE.CandidateDecoderPrimitiveRuns oracle candidate graphShape trace) := by\n  \
let generatedEnv : Mxx.IR.EvalEnv (Mxx.Runtime.RuntimeBackend oracle)\n    \
Mxx.Generated.linkedProgramData := by simpa [candidate, program] using env\n  \
have generatedSuccess : Mxx.IR.eval (Mxx.Runtime.RuntimeBackend oracle)\n    \
Mxx.Generated.program generatedEnv = .ok trace := by\n    \
simpa [candidate, program, generatedEnv] using success"
    )
    .unwrap();
    for (name, site) in runs {
        writeln!(
            source,
            "  obtain ⟨{name}Run⟩ := Mxx.Generated.stage{}_scope{}_node{}ReachedPrimitiveRunFromPublicEval \
generatedEnv trace generatedSuccess",
            site.stage, site.wire.scope, site.wire.node.0
        )
        .unwrap();
    }
    writeln!(
        source,
        "  refine ⟨{{\n    stage := {}\n    scope := {}",
        coefficient.stage, coefficient.wire.scope
    )
    .unwrap();
    for field in [
        "coefficientArguments",
        "quarterArguments",
        "threeArguments",
        "threeQuarterArguments",
        "lowerArguments",
        "upperArguments",
        "lowerIntArguments",
        "upperIntArguments",
        "sumArguments",
        "twoArguments",
        "decodedArguments",
        "coefficientOutputs",
        "quarterOutputs",
        "threeOutputs",
        "threeQuarterOutputs",
        "lowerOutputs",
        "upperOutputs",
        "lowerIntOutputs",
        "upperIntOutputs",
        "sumOutputs",
        "twoOutputs",
        "decodedOutputs",
        "decodedOutputWire",
    ] {
        writeln!(source, "    {field} := by decide").unwrap();
    }
    let decoder_definitions = "graphShape, decryptionSites, decryptionDecoderCoefficient, \
decryptionDecoderQuarter, decryptionDecoderThree, decryptionDecoderThreeQuarter, \
decryptionDecoderLowerComparison, decryptionDecoderUpperComparison, \
decryptionDecoderLowerBoolToInt, decryptionDecoderUpperBoolToInt, decryptionDecoderSum, \
decryptionDecoderTwo, decryptionDecoderEqualsTwo, \
Mxx.We.DiamondWE.StoredNodeRef.concreteNode";
    for (name, _) in runs {
        writeln!(source, "    {name} := by simpa [{decoder_definitions}] using {name}Run").unwrap();
    }
    writeln!(
        source,
        "  }}⟩\n\n\
theorem decoderTrace\n  \
{{oracle : Mxx.Runtime.RuntimeGadgetOracle}}\n  \
(env : Mxx.We.DiamondWE.RuntimeEvalEnv oracle candidate.program.data)\n  \
(trace : Mxx.We.DiamondWE.RuntimeTrace oracle)\n  \
(success : Mxx.IR.eval (Mxx.Runtime.RuntimeBackend oracle) candidate.program env = .ok trace)\n  \
{{matrixType : Mxx.IR.MatrixType}}\n  \
(actual : Mxx.We.DiamondWE.RuntimeMatrixValue matrixType)\n  \
(noisyTraced : Mxx.IR.traceValueAt trace (Mxx.IR.occurrenceOf {} #[]\n    \
graphShape.decryptionSites.noisyPlaintext.reference.2.wire) =\n      \
some ⟨.Mxx.IR.WireType.matrix matrixType, actual⟩) :\n  \
∃ coefficient decoded,\n    \
Nonempty (Mxx.We.DiamondWE.DecoderPrimitiveChain oracle trace matrixType actual\n      \
candidate.parameters.modulus coefficient decoded) ∧\n    \
Mxx.IR.traceValueAt trace\n      \
(Mxx.IR.occurrenceOf {} #[] candidate.refs.decodedOutput.wire) =\n        \
some ⟨.Mxx.IR.WireType.bool, decoded⟩ ∧\n    \
decoded = Mxx.We.DiamondWE.decodeInterval candidate.parameters.modulus coefficient := by\n  \
obtain ⟨runs⟩ := decoderPrimitiveRuns env trace success\n  \
exact runs.decoderTrace actual candidate.parameters.valid.1 noisyTraced\n",
        coefficient.stage, coefficient.stage
    )
    .unwrap();
}

fn render_parameter_match(encryption_stage: usize, decryption_stage: usize) -> String {
    format!(
        "def parametersMatch : Mxx.We.DiamondWE.ParametersMatchProgram candidate := by\n  constructor\n  all_goals simp [Mxx.We.DiamondWE.hasDiamondGraphBinding, Mxx.We.DiamondWE.hasConcreteBindingAt, Mxx.We.DiamondWE.bindingValueAt, candidate, parameters, parametersData, circuitShape, program, Mxx.Generated.stage{encryption_stage}, Mxx.Generated.stage{decryption_stage}]"
    )
}

fn resolve_frozen_ref(
    linked: &ValidatedLinkedProgram,
    stage: usize,
    label: &str,
    reference: &FrozenValueRef,
) -> Result<ConcreteSemanticWireRef, DiamondLeanError> {
    let stage_data = linked.stages().get(stage).ok_or_else(|| {
        DiamondLeanError::SemanticRole(format!("{label} refers to missing stage {stage}"))
    })?;
    linked
        .resolve_semantic_wire(stage_data.key(), reference)
        .map_err(|error| DiamondLeanError::SemanticRole(format!("cannot resolve {label}: {error}")))
}

fn structural_site_kind(
    linked: &ValidatedLinkedProgram,
    reference: &ConcreteSemanticWireRef,
    label: &str,
) -> Result<&'static str, DiamondLeanError> {
    let projection = linked
        .semantic_projection()
        .map_err(|error| DiamondLeanError::SemanticRole(error.to_string()))?;
    let stage = projection.stages.get(reference.stage).ok_or_else(|| {
        DiamondLeanError::SemanticRole(format!("{label} refers to missing stage"))
    })?;
    let scope =
        stage.scopes.iter().find(|scope| scope.id == reference.wire.scope).ok_or_else(|| {
            DiamondLeanError::SemanticRole(format!("{label} refers to missing scope"))
        })?;
    let node = scope
        .nodes
        .get(reference.wire.node.0 as usize)
        .ok_or_else(|| DiamondLeanError::SemanticRole(format!("{label} refers to missing node")))?;
    use ConcreteNodePayload as P;
    Ok(match &node.kind {
        P::Input { artifact: None, .. } => "input",
        P::Input { artifact: Some(_), .. } => "artifactInput",
        P::MatrixBinary(mxx_ir_core::node::MatrixBinaryOp::Multiply) => "matrixMultiply",
        P::MatrixBinary(_) => "matrixAddSub",
        P::ApplyPreimage => "applyPreimage",
        P::PreimageSample { .. } => "preimageSample",
        P::FamilyPreimageSample { .. } => "familyPreimageSample",
        P::GadgetDecompose { .. } => "gadgetDecompose",
        P::SequentialLoop(_) => "sequentialLoop",
        P::ParallelGrid(_) => "parallelGrid",
        P::Select { .. } => "select",
        P::FamilyPack { .. } |
        P::FamilyGetStatic { .. } |
        P::FamilyGetDynamic { .. } |
        P::FamilySelectAxis { .. } |
        P::FamilyReindex { .. } |
        P::FamilyGather { .. } => "familyOperation",
        P::ExtractCoefficient { .. } => "coefficientExtraction",
        P::ThresholdDecode { .. } => "thresholdDecode",
        _ => "other",
    })
}

fn render_stored_site(
    source: &mut String,
    name: &str,
    wire_name: &str,
    linked: &ValidatedLinkedProgram,
    stage: usize,
    label: &str,
    value: &FrozenValueRef,
) -> Result<(), DiamondLeanError> {
    let resolved = resolve_frozen_ref(linked, stage, label, value)?;
    render_typed_ref(source, wire_name, &resolved)?;
    let kind = structural_site_kind(linked, &resolved, label)?;
    let payload = render_site_payload(linked, &resolved, label)?;
    writeln!(
        source,
        "def {name} : Mxx.We.DiamondWE.StoredNodeRef program := {{\n  reference := ⟨_, {wire_name}⟩\n  stage := {}\n  scope := {}\n  payload := {payload}\n  kind := .{kind}\n  arguments := (Mxx.We.DiamondWE.nodeArgumentsAt ⟨_, {wire_name}⟩).getD #[]\n  outputs := (Mxx.We.DiamondWE.nodeOutputsAt ⟨_, {wire_name}⟩).getD #[]\n  payload_stored := by rfl\n  kind_stored := by decide\n  arguments_stored := by decide\n  outputs_stored := by decide\n  ownership_stored := by decide\n}}\n",
        resolved.stage,
        resolved.wire.scope
    )
    .unwrap();
    Ok(())
}

fn render_structural_int_expr(value: &mxx_ir_core::linked::ConcreteStructuralIntExpr) -> String {
    use mxx_ir_core::linked::ConcreteStructuralIntExpr as E;
    match value {
        E::Literal(value) => format!("(.literal {value})"),
        E::StructuralSlot(slot) => format!("(.structuralSlot {slot})"),
        E::Add(left, right) => format!(
            "(.add {} {})",
            render_structural_int_expr(left),
            render_structural_int_expr(right)
        ),
        E::Sub(left, right) => format!(
            "(.subtract {} {})",
            render_structural_int_expr(left),
            render_structural_int_expr(right)
        ),
        E::Mul(left, right) => format!(
            "(.multiply {} {})",
            render_structural_int_expr(left),
            render_structural_int_expr(right)
        ),
        E::ExactDivide(left, right) => format!(
            "(.exactDivide {} {})",
            render_structural_int_expr(left),
            render_structural_int_expr(right)
        ),
        E::RoundDivide(left, right) => format!(
            "(.roundDivide {} {})",
            render_structural_int_expr(left),
            render_structural_int_expr(right)
        ),
        E::Log2Ceil(inner) => format!("(.log2Ceil {})", render_structural_int_expr(inner)),
    }
}

fn render_index_map_expr(value: &mxx_ir_core::linked::ConcreteIndexMapExpr) -> String {
    use mxx_ir_core::linked::ConcreteIndexMapExpr as E;
    match value {
        E::Literal(value) => format!("(.literal {value})"),
        E::Axis(axis) => format!("(.axis {axis})"),
        E::StructuralSlot(slot) => format!("(.structuralSlot {slot})"),
        E::Add(left, right) => {
            format!("(.add {} {})", render_index_map_expr(left), render_index_map_expr(right))
        }
        E::Sub(left, right) => {
            format!("(.sub {} {})", render_index_map_expr(left), render_index_map_expr(right))
        }
        E::Mul(left, right) => {
            format!("(.mul {} {})", render_index_map_expr(left), render_index_map_expr(right))
        }
        E::EuclideanDivide(left, right) => {
            format!("(.divide {} {})", render_index_map_expr(left), render_index_map_expr(right))
        }
        E::EuclideanRemainder(left, right) => {
            format!("(.remainder {} {})", render_index_map_expr(left), render_index_map_expr(right))
        }
        E::Equal(left, right) => {
            format!("(.equal {} {})", render_index_map_expr(left), render_index_map_expr(right))
        }
        E::Less(left, right) => {
            format!("(.less {} {})", render_index_map_expr(left), render_index_map_expr(right))
        }
        E::LessEqual(left, right) => {
            format!("(.lessEqual {} {})", render_index_map_expr(left), render_index_map_expr(right))
        }
        E::Log2Ceil(inner) => format!("(.log2Ceil {})", render_index_map_expr(inner)),
        E::Select { selector, branches } => format!(
            "(.select {} #[{}])",
            render_index_map_expr(selector),
            branches.iter().map(render_index_map_expr).collect::<Vec<_>>().join(", ")
        ),
    }
}

fn render_confidentiality(
    confidentiality: mxx_ir_core::artifact::ArtifactConfidentiality,
) -> &'static str {
    match confidentiality {
        mxx_ir_core::artifact::ArtifactConfidentiality::Public => ".Public",
        mxx_ir_core::artifact::ArtifactConfidentiality::Private => ".Private",
    }
}

/// Render the exact payload of a retained node, including operation
/// parameters. Unsupported retained payloads fail closed instead of being
/// silently reduced to an operation class.
fn render_site_payload(
    linked: &ValidatedLinkedProgram,
    reference: &ConcreteSemanticWireRef,
    label: &str,
) -> Result<String, DiamondLeanError> {
    let projection = linked
        .semantic_projection()
        .map_err(|error| DiamondLeanError::SemanticRole(error.to_string()))?;
    let stage = projection.stages.get(reference.stage).ok_or_else(|| {
        DiamondLeanError::SemanticRole(format!("{label} refers to missing stage"))
    })?;
    let scope =
        stage.scopes.iter().find(|scope| scope.id == reference.wire.scope).ok_or_else(|| {
            DiamondLeanError::SemanticRole(format!("{label} refers to missing scope"))
        })?;
    let node = scope
        .nodes
        .get(reference.wire.node.0 as usize)
        .ok_or_else(|| DiamondLeanError::SemanticRole(format!("{label} refers to missing node")))?;
    use ConcreteNodePayload as P;
    Ok(match &node.kind {
        P::ConstantInt(value) => format!(".constantInt {value}"),
        P::EvaluateInt(value) => format!(".evaluateInt {}", render_structural_int_expr(value)),
        P::Input { artifact: None, .. } => format!(".input {}", reference.wire.node.0),
        P::Input { artifact: Some(artifact), .. } => {
            let index = projection
                .artifact_links
                .iter()
                .position(|link| {
                    link.consumer_stage == reference.stage &&
                        link.consumer.scope == reference.wire.scope &&
                        link.consumer.node == reference.wire.node
                })
                .ok_or_else(|| {
                    DiamondLeanError::SemanticRole(format!(
                        "{label} artifact input has no resolved link"
                    ))
                })?;
            let confidentiality = render_confidentiality(artifact.confidentiality);
            format!(
                ".artifactInput {{ index := {index}, name := \"{}\", confidentiality := {confidentiality} }}",
                artifact.name.replace('\\', "\\\\").replace('"', "\\\"")
            )
        }
        P::ApplyPreimage => ".applyPreimage".to_owned(),
        P::MaterializePreimageExact => ".materializePreimageExact".to_owned(),
        P::MatrixBinary(mxx_ir_core::node::MatrixBinaryOp::Multiply) => {
            ".matrixBinary .multiply".to_owned()
        }
        P::MatrixBinary(mxx_ir_core::node::MatrixBinaryOp::Add) => ".matrixBinary .add".to_owned(),
        P::MatrixBinary(mxx_ir_core::node::MatrixBinaryOp::Subtract) => {
            ".matrixBinary .subtract".to_owned()
        }
        P::FamilyGetDynamic { rank } => format!(".familyGetDynamic {rank}"),
        P::FamilySelectAxis { axis } => format!(".familySelectAxis {axis}"),
        P::BoolToInt => ".boolToInt".to_owned(),
        P::IntBinary(mxx_ir_core::node::IntBinaryOp::Add) => ".intBinary .add".to_owned(),
        P::IntBinary(mxx_ir_core::node::IntBinaryOp::Subtract) => ".intBinary .subtract".to_owned(),
        P::IntBinary(mxx_ir_core::node::IntBinaryOp::Multiply) => ".intBinary .multiply".to_owned(),
        P::IntCompare(mxx_ir_core::node::IntCompareOp::LessEqual) => {
            ".intCompare .lessEqual".to_owned()
        }
        P::IntCompare(mxx_ir_core::node::IntCompareOp::Less) => ".intCompare .less".to_owned(),
        P::IntCompare(mxx_ir_core::node::IntCompareOp::Equal) => ".intCompare .equal".to_owned(),
        P::ExtractCoefficient { position, canonical_input_exclusive_upper } => format!(
            ".extractCoefficient {} {}",
            render_structural_int_expr(position),
            canonical_input_exclusive_upper
                .as_ref()
                .map_or("none".to_owned(), |v| format!("some {}", v)),
        ),
        P::ThresholdDecode { plaintext_modulus, length, output_bool } => format!(
            ".thresholdDecode {} {} {}",
            render_structural_int_expr(plaintext_modulus),
            render_structural_int_expr(length),
            output_bool
        ),
        P::GadgetDecompose { base, small, digit_count } => {
            format!(
                ".gadgetDecompose {} {} {}",
                render_structural_int_expr(base),
                small,
                render_structural_int_expr(digit_count)
            )
        }
        P::FamilyReindex { output_shape, map } => format!(
            ".familyReindex #[{}] {{ sourceRank := {}, outputRank := {}, inputIndices := #[{}] }}",
            output_shape.iter().map(render_structural_int_expr).collect::<Vec<_>>().join(", "),
            map.source_rank,
            map.output_rank,
            map.input_indices.iter().map(render_index_map_expr).collect::<Vec<_>>().join(", ")
        ),
        P::FamilyPreimageSample { matrix_type, max_coefficient_bound } => {
            format!(
                ".familyPreimageSample {} {}",
                render_matrix_type(matrix_type),
                render_structural_int_expr(max_coefficient_bound)
            )
        }
        P::PreimageSample { matrix_type, max_coefficient_bound } => format!(
            ".preimageSample {} {}",
            render_matrix_type(matrix_type),
            render_structural_int_expr(max_coefficient_bound)
        ),
        P::GaussianSample { matrix_type, sigma, max_coefficient_bound } => format!(
            ".gaussianSample {} {} {}",
            render_matrix_type(matrix_type),
            render_concrete_real_expr(sigma),
            render_structural_int_expr(max_coefficient_bound)
        ),
        P::ParallelGrid(grid) => {
            let shape =
                grid.shape.iter().map(render_structural_int_expr).collect::<Vec<_>>().join(", ");
            let slots =
                grid.index_slots.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ");
            let bindings = grid
                .bindings
                .iter()
                .map(|(name, value)| format!("(\"{name}\", {})", render_structural_int_expr(value)))
                .collect::<Vec<_>>()
                .join(", ");
            let modes = grid.input_modes.iter().map(|mode| match mode {
                mxx_ir_core::linked::ConcreteGridInputMode::Broadcast => "{ reindex := false, map := none }".to_owned(),
                mxx_ir_core::linked::ConcreteGridInputMode::Reindex { map } => format!("{{ reindex := true, map := some {{ sourceRank := {}, outputRank := {}, inputIndices := #[{}] }} }}", map.source_rank, map.output_rank, map.input_indices.iter().map(render_index_map_expr).collect::<Vec<_>>().join(", ")),
            }).collect::<Vec<_>>().join(", ");
            format!(
                ".parallelGrid {{ child := {}, shape := #[{}], indexSlots := #[{}], bindings := #[{}], inputModes := #[{}] }}",
                node.child_scope.unwrap_or_default(),
                shape,
                slots,
                bindings,
                modes
            )
        }
        P::SequentialLoop(loop_data) => {
            let bindings = loop_data
                .bindings
                .iter()
                .map(|(name, value)| format!("(\"{name}\", {})", render_structural_int_expr(value)))
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                ".sequentialLoop {{ child := {}, count := {}, indexSlot := {}, bindings := #[{}], carriedCount := {} }}",
                node.child_scope.unwrap_or_default(),
                render_structural_int_expr(&loop_data.count),
                loop_data.index_slot,
                bindings,
                loop_data.carried_count
            )
        }
        P::Select { count } => format!(".select {}", render_structural_int_expr(count)),
        other => {
            return Err(DiamondLeanError::SemanticRole(format!(
                "{label} has unsupported retained payload {other:?}"
            )))
        }
    })
}

fn render_concrete_real_expr(expr: &mxx_ir_core::linked::ConcreteRealExpr) -> String {
    use mxx_ir_core::linked::ConcreteRealExpr as E;
    match expr {
        E::Rational(value) => format!(
            "(.literal {{ numerator := {}, denominator := {} }})",
            value.numerator(),
            value.denominator()
        ),
        E::FromInt(value) => format!("(.fromInt {})", render_structural_int_expr(value)),
        E::Add(left, right) => format!(
            "(.add {} {})",
            render_concrete_real_expr(left),
            render_concrete_real_expr(right)
        ),
        E::Sub(left, right) => format!(
            "(.subtract {} {})",
            render_concrete_real_expr(left),
            render_concrete_real_expr(right)
        ),
        E::Mul(left, right) => format!(
            "(.multiply {} {})",
            render_concrete_real_expr(left),
            render_concrete_real_expr(right)
        ),
        E::Div(left, right) => format!(
            "(.divide {} {})",
            render_concrete_real_expr(left),
            render_concrete_real_expr(right)
        ),
        E::Sqrt(value) => format!("(.sqrt {})", render_concrete_real_expr(value)),
    }
}

fn render_frozen_coordinate(
    linked: &ValidatedLinkedProgram,
    stage: usize,
    owner: &FrozenValueRef,
    value: &mxx_ir_core::FrozenStructuralIntExpr,
) -> Result<String, DiamondLeanError> {
    let stage_key = linked
        .stages()
        .get(stage)
        .ok_or_else(|| DiamondLeanError::SemanticRole("missing coordinate stage".to_owned()))?
        .key();
    let concrete = linked
        .close_frozen_structural_expr(stage_key, owner, value)
        .map_err(|error| DiamondLeanError::SemanticRole(error.to_string()))?;
    Ok(render_structural_int_expr(&concrete))
}

fn render_bgg_step(step: mxx_bgg::BggTraceStep) -> &'static str {
    use mxx_bgg::BggTraceStep as S;
    match step {
        S::ZeroPlaintext => "zeroPlaintext",
        S::ZeroVector => "zeroVector",
        S::ZeroPublicKey => "zeroPublicKey",
        S::NotPlaintext => "notPlaintext",
        S::NotVector => "notVector",
        S::NotPublicKey => "notPublicKey",
        S::ProductPublicKeyDecompose => "productPublicKeyDecompose",
        S::ProductPublicKeyMaterialize => "productPublicKeyMaterialize",
        S::ProductPublicKeyMultiply => "productPublicKeyMultiply",
        S::ProductVectorDecompose => "productVectorDecompose",
        S::ProductVectorApplyPreimage => "productVectorApplyPreimage",
        S::ProductVectorMultiply => "productVectorMultiply",
        S::ProductVectorOutput => "productVectorOutput",
        S::ProductPlaintextOutput => "productPlaintextOutput",
        S::SumPlaintext => "sumPlaintext",
        S::SumVector => "sumVector",
        S::SumPublicKey => "sumPublicKey",
        S::TwoProductPublicKey => "twoProductPublicKey",
        S::TwoProductVector => "twoProductVector",
        S::TwoProductPlaintext => "twoProductPlaintext",
        S::XorPlaintext => "xorPlaintext",
        S::XorVector => "xorVector",
        S::XorPublicKey => "xorPublicKey",
        S::CandidateVectorSelect => "candidateVectorSelect",
        S::CandidatePublicKeySelect => "candidatePublicKeySelect",
        S::CandidatePlaintextSelect => "candidatePlaintextSelect",
        S::ActiveVectorSelect => "activeVectorSelect",
        S::ActivePublicKeySelect => "activePublicKeySelect",
        S::ActivePlaintextSelect => "activePlaintextSelect",
        S::LayerOutput => "layerOutput",
    }
}

fn render_bgg_path(
    linked: &ValidatedLinkedProgram,
    stage: usize,
    route: &StructuralValueRoute,
) -> Result<String, DiamondLeanError> {
    let stage_data = linked.stages().get(stage).ok_or_else(|| {
        DiamondLeanError::SemanticRole("missing BGG operand path stage".to_owned())
    })?;
    let scope_index =
        |scope: &mxx_ir_core::FrozenGraphScopeId| {
            stage_data.graph.scopes.keys().position(|candidate| candidate == scope).ok_or_else(
                || DiamondLeanError::SemanticRole("BGG route scope is missing".to_owned()),
            )
        };
    let mut hops = route
        .exits
        .iter()
        .map(|hop| {
            let parent_scope = scope_index(&hop.parent_scope)?;
            Ok(format!(
                ".exit {{ parentScope := {parent_scope}, owner := {}, outputIndex := {} }}",
                hop.owner.0, hop.output_index
            ))
        })
        .collect::<Result<Vec<_>, DiamondLeanError>>()?;
    hops.extend(
        route
            .enters
            .iter()
            .map(|hop| {
                let parent_scope = scope_index(&hop.parent_scope)?;
                Ok(format!(
                    ".enter {{ parentScope := {parent_scope}, owner := {}, inputIndex := {} }}",
                    hop.owner.0, hop.input_index
                ))
            })
            .collect::<Result<Vec<_>, DiamondLeanError>>()?,
    );
    Ok(format!(
        "{{ exits := #[{}], enters := #[{}] }}",
        hops.iter().filter_map(|h| h.strip_prefix(".exit ")).collect::<Vec<_>>().join(", "),
        hops.iter().filter_map(|h| h.strip_prefix(".enter ")).collect::<Vec<_>>().join(", ")
    ))
}

fn render_encryption_sites(
    source: &mut String,
    stage: usize,
    linked: &ValidatedLinkedProgram,
    sites: &DiamondStructuralSiteRefs,
    input_count: usize,
    batch_bits: usize,
    state_count: usize,
    digit_base: usize,
    error_bound: &BigInt,
) -> Result<(), DiamondLeanError> {
    let fields = [
        ("injectorInitial", "injector_initial", &sites.injector_initial),
        ("injectorTransitions", "injector_transitions", &sites.injector_transitions),
        ("injectorFinalTrapdoor", "injector_final_trapdoor", &sites.injector_final_trapdoor),
        ("onePreimage", "one_preimage", &sites.one_preimage),
        ("kPreimage", "k_preimage", &sites.k_preimage),
        ("decoderPreimage", "decoder_preimage", &sites.decoder_preimage),
        ("publicKeys", "public_keys", &sites.public_keys),
        ("witnessPreimages", "witness_preimages", &sites.witness_preimages),
        ("rDecomposition", "r_decomposition", &sites.r_decomposition),
    ];
    for (field, suffix, value) in fields {
        render_stored_site(
            source,
            &format!("encryption{field}"),
            &format!("encryption{field}Wire"),
            linked,
            stage,
            &format!("encryption {suffix}"),
            value,
        )?;
    }
    let target = &sites.injector_target_trace;
    for (field, suffix, value) in [
        ("TargetPublic", "target public", &target.target_public),
        ("TargetGrid", "target grid", &target.target_grid),
        ("TargetReindex", "target reindex", &target.target_reindex),
    ] {
        render_stored_site(
            source,
            &format!("encryptionInjector{field}"),
            &format!("encryptionInjector{field}Wire"),
            linked,
            stage,
            &format!("encryption injector {suffix}"),
            value,
        )?;
    }
    let mut target_entries = Vec::with_capacity(target.entries.len());
    for (index, entry) in target.entries.iter().enumerate() {
        let role = match entry.role {
            mxx_gadgets::input_injector::DiamondInputTargetTraceRole::Selector => "selector",
            mxx_gadgets::input_injector::DiamondInputTargetTraceRole::SelectorProduct => {
                "selectorProduct"
            }
            mxx_gadgets::input_injector::DiamondInputTargetTraceRole::GaussianError => {
                "gaussianError"
            }
            mxx_gadgets::input_injector::DiamondInputTargetTraceRole::TargetAdd => "targetAdd",
        };
        let site = format!("encryptionInjectorTargetTrace{index}");
        render_stored_site(
            source,
            &site,
            &format!("{site}Wire"),
            linked,
            stage,
            &format!("encryption injector target trace {index}"),
            &entry.handle,
        )?;
        let mut operands = Vec::with_capacity(entry.operands.len());
        for (operand_index, operand) in entry.operands.iter().enumerate() {
            let resolved = resolve_frozen_ref(
                linked,
                stage,
                &format!("encryption injector target operand {index}:{operand_index}"),
                operand,
            )?;
            let name = format!("{site}Operand{operand_index}Wire");
            render_typed_ref(source, &name, &resolved)?;
            operands.push(format!("{name}.wire"));
        }
        let tagged = format!("{site}Tagged");
        writeln!(
            source,
            "def {tagged} : Mxx.We.DiamondWE.TaggedInjectorTargetSite program := {{ site := {site}, role := .{role}, operands := #[{}], arguments_eq := by rfl }}",
            operands.join(", ")
        )
        .unwrap();
        target_entries.push(tagged);
    }
    writeln!(
        source,
        "def encryptionInjectorTargetTraceEntries : Array (Mxx.We.DiamondWE.TaggedInjectorTargetSite program) := #[{}]",
        target_entries.join(", ")
    )
    .unwrap();
    writeln!(
        source,
        "def encryptionInjectorTargetTrace : Mxx.We.DiamondWE.InjectorTargetTraceSites program {input_count} {batch_bits} {state_count} {digit_base} {error_bound} := {{"
    )
    .unwrap();
    source.push_str(
        "  targetPublic := encryptionInjectorTargetPublic\n  targetGrid := encryptionInjectorTargetGrid\n  targetReindex := encryptionInjectorTargetReindex\n  entries := encryptionInjectorTargetTraceEntries\n  traceComplete := by decide\n  targetGridPayload := by decide\n  targetReindexPayload := by decide\n  targetGridArgument := by decide\n  entryStages := by decide\n  targetGridChildInput := by decide\n  selectorProductEdges := by decide\n  targetAddEdges := by decide\n  targetAddChildOutput := by decide\n  targetReindexArgument := by decide\n}\n\n",
    );
    let selector = &sites.selector_magnitude_trace;
    for (field, suffix, value) in [
        ("DigitSecrets", "digit secrets", &selector.digit_secrets),
        ("TargetGrid", "target grid", &selector.target_grid),
        ("Loop", "selector loop", &selector.selector_loop.handle),
    ] {
        render_stored_site(
            source,
            &format!("encryptionSelectorMagnitude{field}"),
            &format!("encryptionSelectorMagnitude{field}Wire"),
            linked,
            stage,
            &format!("encryption selector magnitude {suffix}"),
            value,
        )?;
    }
    let mut selector_entries = Vec::with_capacity(selector.entries.len());
    for (index, entry) in selector.entries.iter().enumerate() {
        use mxx_gadgets::input_injector::SelectorMagnitudeTraceRole as Role;
        let role = match entry.role {
            Role::DigitSecretSample => "digitSecretSample",
            Role::SelectedSecret => "selectedSecret",
            Role::RegularDiagonal => "regularDiagonal",
            Role::Identity => "identity",
            Role::KDiagonal => "kDiagonal",
            Role::InitialSelect => "initialSelect",
            Role::BitZero => "bitZero",
            Role::BitIdentity => "bitIdentity",
            Role::BitValueSelect => "bitValueSelect",
            Role::SecretTimesBitValue => "secretTimesBitValue",
            Role::SpecialTop => "specialTop",
            Role::SpecialBottom => "specialBottom",
            Role::SpecialConcat => "specialConcat",
            Role::CarriedVsSpecialSelect => "carriedVsSpecialSelect",
        };
        let site = format!("encryptionSelectorMagnitudeTrace{index}");
        render_stored_site(
            source,
            &site,
            &format!("{site}Wire"),
            linked,
            stage,
            &format!("encryption selector magnitude trace {index}"),
            &entry.handle,
        )?;
        let mut operands = Vec::with_capacity(entry.operands.len());
        for (operand_index, operand) in entry.operands.iter().enumerate() {
            let resolved = resolve_frozen_ref(
                linked,
                stage,
                &format!("encryption selector magnitude operand {index}:{operand_index}"),
                operand,
            )?;
            let name = format!("{site}Operand{operand_index}Wire");
            render_typed_ref(source, &name, &resolved)?;
            operands.push(format!("{name}.wire"));
        }
        let tagged = format!("{site}Tagged");
        writeln!(
            source,
            "def {tagged} : Mxx.We.DiamondWE.TaggedSelectorMagnitudeSite program := {{ site := {site}, role := .{role}, operands := #[{}] }}",
            operands.join(", ")
        )
        .unwrap();
        selector_entries.push(tagged);
    }
    writeln!(
        source,
        "def encryptionSelectorMagnitudeEntries : Array (Mxx.We.DiamondWE.TaggedSelectorMagnitudeSite program) := #[{}]",
        selector_entries.join(", ")
    )
    .unwrap();
    writeln!(
        source,
        "def encryptionSelectorMagnitudeTrace : Mxx.We.DiamondWE.SelectorMagnitudeTraceSites program {input_count} {batch_bits} {digit_base} := {{"
    )
    .unwrap();
    source.push_str(
        "  digitSecrets := encryptionSelectorMagnitudeDigitSecrets\n  targetGrid := encryptionSelectorMagnitudeTargetGrid\n  selectorLoop := encryptionSelectorMagnitudeLoop\n  entries := encryptionSelectorMagnitudeEntries\n  traceComplete := by decide\n  digitGridPayload := by decide\n  selectorLoopPayload := by decide\n  entryStages := by decide\n  operandEdges := by decide\n  digitSampleOutput := by decide\n  selectedSecretFamily := by decide\n  initialCarriedEdge := by decide\n  selectedSecretInvariantEdge := by decide\n  secretMultiplyInvariantEdge := by decide\n  carriedBranchEdge := by decide\n  loopBodyOutput := by decide\n}\n\n",
    );
    writeln!(
        source,
        "def encryptionSites : Mxx.We.DiamondWE.EncryptionGraphSites program {input_count} {batch_bits} {state_count} {digit_base} {error_bound} := {{"
    )
    .unwrap();
    for (field, _, _) in fields {
        writeln!(source, "  {field} := encryption{field}").unwrap();
    }
    source.push_str(
        "  injectorTargetTrace := encryptionInjectorTargetTrace\n  selectorMagnitudeTrace := encryptionSelectorMagnitudeTrace\n  selectorMagnitudeTargetEdge := by decide\n  selectorMagnitudeLoopEdge := by decide\n  injectorTransitionTargetEdge := by decide\n}\n\n",
    );
    Ok(())
}

fn render_decryption_sites(
    source: &mut String,
    stage: usize,
    linked: &ValidatedLinkedProgram,
    sites: &DiamondDecryptionSiteRefs,
    modulus: &BigUint,
    request_input_count: usize,
    request_batch_bits: usize,
    request_state_count: usize,
    request_digit_base: usize,
) -> Result<(), DiamondLeanError> {
    let fields = [
        ("injectorInitial", &sites.injector_initial),
        ("injectorTransitions", &sites.injector_transitions),
        ("injectorStates", &sites.injector_states),
        ("injectorLoopOutput", &sites.injector_loop_output),
        ("injectorBodyOutput", &sites.injector_body_output),
        ("witnessVectors", &sites.witness_vectors),
        ("oneProjection", &sites.one_projection),
        ("kProjection", &sites.k_projection),
        ("decoderProjection", &sites.decoder_projection),
        ("publicKeys", &sites.public_keys),
        ("circuitOutput", &sites.circuit_output),
        ("oneMinusCircuit", &sites.one_minus_circuit),
        ("projectedDifference", &sites.projected_difference),
        ("rDecomposition", &sites.r_decomposition),
        ("kPlusProjection", &sites.k_plus_projection),
        ("noisyPlaintext", &sites.noisy_plaintext),
        ("decoded", &sites.decoded),
        ("decoderCoefficient", &sites.decoder_coefficient),
        ("decoderLowerComparison", &sites.decoder_lower_comparison),
        ("decoderUpperComparison", &sites.decoder_upper_comparison),
        ("decoderLowerBoolToInt", &sites.decoder_lower_bool_to_int),
        ("decoderUpperBoolToInt", &sites.decoder_upper_bool_to_int),
        ("decoderSum", &sites.decoder_sum),
        ("decoderEqualsTwo", &sites.decoder_equals_two),
        ("decoderQuarter", &sites.decoder_quarter),
        ("decoderThreeQuarter", &sites.decoder_three_quarter),
        ("decoderTwo", &sites.decoder_two),
        ("decoderThree", &sites.decoder_three),
    ];
    for (field, value) in fields {
        render_stored_site(
            source,
            &format!("decryption{field}"),
            &format!("decryption{field}Wire"),
            linked,
            stage,
            &format!("decryption {field}"),
            value,
        )?;
    }
    let mut injector_entries = Vec::with_capacity(sites.injector_trace.entries.len());
    for (index, entry) in sites.injector_trace.entries.iter().enumerate() {
        let role = match entry.role {
            mxx_gadgets::input_injector::DiamondInputTraceRole::PackedInputDigits => {
                "packedInputDigits"
            }
            mxx_gadgets::input_injector::DiamondInputTraceRole::SourceStateReindex => {
                "sourceStateReindex"
            }
            mxx_gadgets::input_injector::DiamondInputTraceRole::TransitionReindex => {
                "transitionReindex"
            }
            mxx_gadgets::input_injector::DiamondInputTraceRole::SelectedTransition => {
                "selectedTransition"
            }
            mxx_gadgets::input_injector::DiamondInputTraceRole::BodyApplyPreimage => {
                "bodyApplyPreimage"
            }
            mxx_gadgets::input_injector::DiamondInputTraceRole::CarriedPreviousState => {
                "carriedPreviousState"
            }
            mxx_gadgets::input_injector::DiamondInputTraceRole::NextStateBodyOutput => {
                "nextStateBodyOutput"
            }
        };
        let site = format!("decryptionInjectorTrace{index}");
        render_stored_site(
            source,
            &site,
            &format!("{site}Wire"),
            linked,
            stage,
            &format!("injector trace {index}"),
            &entry.handle,
        )?;
        let mut operands = Vec::new();
        for (operand_index, operand) in entry.operands.iter().enumerate() {
            let resolved = resolve_frozen_ref(linked, stage, "injector operand", operand)?;
            let name = format!("{site}Operand{operand_index}Wire");
            render_typed_ref(source, &name, &resolved)?;
            operands.push(format!("{name}.wire"));
        }
        let coordinate =
            render_frozen_coordinate(linked, stage, &entry.handle, &entry.loop_coordinate)?;
        let tagged = format!("{site}Tagged");
        writeln!(source, "def {tagged} : Mxx.We.DiamondWE.TaggedInjectorSite program := {{ site := {site}, role := .{role}, coordinate := {coordinate}, operands := #[{}], arguments_eq := by rfl }}", operands.join(", ")).unwrap();
        injector_entries.push(tagged);
    }
    writeln!(source, "def decryptionInjectorTraceEntries : Array (Mxx.We.DiamondWE.TaggedInjectorSite program) := #[{}]\n", injector_entries.join(", ")).unwrap();
    for (index, value) in sites.bgg_operations.iter().enumerate() {
        render_stored_site(
            source,
            &format!("decryptionBggOperation{index}"),
            &format!("decryptionBggOperation{index}Wire"),
            linked,
            stage,
            &format!("decryption bgg operation {index}"),
            &value.handle,
        )?;
        let mut operand_names = Vec::new();
        for (operand_index, operand) in value.operands.iter().enumerate() {
            let resolved = resolve_frozen_ref(
                linked,
                stage,
                &format!("bgg operation {index} operand {operand_index}"),
                operand,
            )?;
            let name = format!("decryptionBggOperation{index}Operand{operand_index}Wire");
            render_typed_ref(source, &name, &resolved)?;
            operand_names.push(format!("{name}.wire"));
        }
        let mut source_names = Vec::new();
        for (operand_index, source_value) in value.operand_sources.iter().enumerate() {
            let path = match source_value {
                mxx_bgg::FrozenBggOperandSource::External { path, .. } |
                mxx_bgg::FrozenBggOperandSource::Prior { path, .. } => {
                    render_bgg_path(linked, stage, path)?
                }
            };
            let rendered = match source_value {
                mxx_bgg::FrozenBggOperandSource::External { role, handle, .. } => {
                    let resolved = resolve_frozen_ref(
                        linked,
                        stage,
                        &format!("bgg operation {index} external source {operand_index}"),
                        handle,
                    )?;
                    let name = format!("decryptionBggOperation{index}Source{operand_index}Wire");
                    render_typed_ref(source, &name, &resolved)?;
                    let role = match role {
                        mxx_bgg::BggTraceAnchor::One => "one",
                        mxx_bgg::BggTraceAnchor::Left => "left",
                        mxx_bgg::BggTraceAnchor::Right => "right",
                        mxx_bgg::BggTraceAnchor::Scalar => "scalar",
                        mxx_bgg::BggTraceAnchor::Selector => "selector",
                        mxx_bgg::BggTraceAnchor::Active => "active",
                    };
                    format!(".external .{role} {name}.wire {path}")
                }
                mxx_bgg::FrozenBggOperandSource::Prior { step, .. } => {
                    format!(".prior .{} {path}", render_bgg_step(*step))
                }
            };
            source_names.push(rendered);
        }
        let role = match value.role {
            BggTraceRole::Decomposition => "decomposition",
            BggTraceRole::MaterializePreimageExact => "materializePreimageExact",
            BggTraceRole::ApplyPreimage => "applyPreimage",
            BggTraceRole::MatrixMultiply => "matrixMultiply",
            BggTraceRole::CandidateSelect => "candidateSelect",
            BggTraceRole::ActiveSelect => "activeSelect",
            BggTraceRole::GateOutput => "gateOutput",
        };
        let lane = match value.lane {
            mxx_bgg::BggTraceLane::Vector => "vector",
            mxx_bgg::BggTraceLane::PublicKey => "publicKey",
            mxx_bgg::BggTraceLane::Plaintext => "plaintext",
        };
        let subrole = match value.subrole {
            mxx_bgg::BggTraceSubrole::Decompose => "decompose",
            mxx_bgg::BggTraceSubrole::MaterializeExact => "materializeExact",
            mxx_bgg::BggTraceSubrole::Multiply => "multiply",
            mxx_bgg::BggTraceSubrole::ApplyPreimage => "applyPreimage",
            mxx_bgg::BggTraceSubrole::Select => "select",
            mxx_bgg::BggTraceSubrole::GateOutput => "gateOutput",
        };
        let coordinate = |value: &Option<mxx_ir_core::FrozenStructuralIntExpr>| -> Result<String, DiamondLeanError> {
            value.as_ref().map(|value| render_frozen_coordinate(linked, stage, &sites.bgg_operations[index].handle, value)).transpose().map(|value| value.map_or("none".to_owned(), |value| format!("some {value}")))
        };
        writeln!(source, "def decryptionBggOperation{index}Tagged : Mxx.We.DiamondWE.TaggedBggSite program := {{ site := decryptionBggOperation{index}, step := .{}, role := .{role}, lane := .{lane}, subrole := .{subrole}, layer := {}, gateSlot := {}, candidate := {}, operands := #[{}], operandSources := #[{}], arguments_eq := by decide }}", render_bgg_step(value.step), coordinate(&value.layer)?, coordinate(&value.gate_slot)?, coordinate(&value.candidate)?, operand_names.join(", "), source_names.join(", ")).unwrap();
    }
    source.push_str(&format!(
        "def decryptionSites : Mxx.We.DiamondWE.DecryptionGraphSites program {} {} {} {} {} := {{\n",
        modulus,
        request_input_count,
        request_batch_bits,
        request_state_count,
        request_digit_base
    ));
    for (field, _) in fields {
        writeln!(source, "  {field} := decryption{field}").unwrap();
    }
    source.push_str("  injectorTraceEntries := decryptionInjectorTraceEntries\n  injectorTraceComplete := by decide\n");
    source.push_str("  bggOperations := #[");
    for (index, _value) in sites.bgg_operations.iter().enumerate() {
        if index > 0 {
            source.push_str(", ");
        }
        let name = format!("decryptionBggOperation{index}");
        source.push_str(&format!("{name}Tagged"));
    }
    source.push_str("]\n");
    source.push_str("  bggArgumentsComplete := by decide\n");
    source.push_str("  injectorLoopExactPayload := by decide\n  injectorTraceBodyScope := by decide\n  injectorStatesLoopEdge := by decide\n  injectorBodyApplyEdge := by decide\n  injectorNextBodyEdge := by decide\n  injectorTraceCoordinates := by decide\n  injectorReindexCarriedEdge := by decide\n  injectorSelectionDigitEdge := by decide\n  injectorApplySourceEdge := by decide\n  injectorApplyTransitionEdge := by decide\n  injectorNextSourceEdge := by decide\n  injectorNextTransitionEdge := by decide\n");
    source.push_str("  injectorSourceReindexPayload := by decide\n  injectorTransitionReindexPayload := by decide\n");
    source.push_str("  injectorNextGridPayload := by decide\n");
    source.push_str("  injectorGridChildOutput := by decide\n");
    source.push_str("  injectorSelectionReindexEdge := by decide\n");
    source.push_str("  bggLayerOutputEdge := by decide\n");
    source.push_str("  oneMinusOneEdge := by decide\n  oneMinusCircuitEdge := by decide\n");
    source.push_str("  projectedDifferenceValueEdge := by decide\n  projectedDifferencePreimageEdge := by decide\n");
    source.push_str(
        "  kPlusProjectionKEdge := by decide\n  kPlusProjectionDifferenceEdge := by decide\n",
    );
    source.push_str(
        "  noisyPlaintextDecoderEdge := by decide\n  noisyPlaintextProjectionEdge := by decide\n",
    );
    source.push_str("  decodedDecoderEdge := by decide\n");
    source.push_str("  decoderCoefficientNoiseEdge := by decide\n  decoderLowerQuarterEdge := by decide\n  decoderUpperThreeQuarterEdge := by decide\n  decoderLowerCoefficientEdge := by decide\n  decoderUpperCoefficientEdge := by decide\n  decoderLowerIntEdge := by decide\n  decoderUpperIntEdge := by decide\n  decoderSumLowerEdge := by decide\n  decoderSumUpperEdge := by decide\n  decoderEqualsSumEdge := by decide\n  decoderEqualsTwoEdge := by decide\n  decoderThreeQuarterThreeEdge := by decide\n  decoderThreeQuarterQuarterEdge := by decide\n");
    source.push_str("  injectorLoopIsSequential := by decide\n  injectorLoopPayload := by decide\n  decoderCoefficientIsExtraction := by decide\n  decoderComparisonsAreInteger := by decide\n");
    source.push_str("  decoderCoefficientLiteral := by decide\n  decoderLowerComparisonPayload := by decide\n  decoderUpperComparisonPayload := by decide\n  decoderLowerBoolToIntPayload := by decide\n  decoderUpperBoolToIntPayload := by decide\n  decoderSumPayload := by decide\n  decoderEqualsPayload := by decide\n");
    source.push_str("  decoderLowerLessEqualPayload := by decide\n  decoderUpperLessEqualPayload := by decide\n  decoderEqualsExactPayload := by decide\n  decoderQuarterPayload := by decide\n  decoderThreeQuarterPayload := by decide\n  decoderTwoPayload := by decide\n  decoderThreePayload := by decide\n");
    source.push_str("  circuitOutputPayload := by decide\n  oneMinusCircuitPayload := by decide\n");
    source.push_str("  projectedDifferencePayload := by decide\n  kPlusProjectionPayload := by decide\n  noisyPlaintextPayload := by decide\n");
    source.push_str("}\n\n");
    Ok(())
}

fn resolve_role(
    linked: &ValidatedLinkedProgram,
    label: &str,
    role: &FrozenValueRef,
) -> Result<ConcreteSemanticWireRef, DiamondLeanError> {
    let mut resolved = None;
    for stage in linked.stages() {
        if let Ok(reference) = linked.resolve_semantic_wire(stage.key(), role) {
            if resolved.replace(reference).is_some() {
                return Err(DiamondLeanError::SemanticRole(format!(
                    "{label} resolves in more than one linked stage"
                )));
            }
        }
    }
    resolved.ok_or_else(|| {
        DiamondLeanError::SemanticRole(format!("{label} does not resolve in the linked program"))
    })
}

fn require_stage(
    linked: &ValidatedLinkedProgram,
    stage: usize,
    expected_name: &str,
) -> Result<usize, DiamondLeanError> {
    let actual = linked.stages().get(stage).ok_or_else(|| {
        DiamondLeanError::SemanticRole(format!(
            "resolved stage {stage} is missing from the linked program"
        ))
    })?;
    if actual.key() != expected_name {
        return Err(DiamondLeanError::SemanticRole(format!(
            "resolved stage {stage} is {:?}, expected {expected_name:?}",
            actual.key()
        )));
    }
    Ok(stage)
}

fn validate_parameter_bindings(
    request: &DiamondLeanClaimRequest<'_>,
    encryption_stage: usize,
    decryption_stage: usize,
) -> Result<(), DiamondLeanError> {
    let projection = request
        .linked
        .semantic_projection()
        .map_err(|error| DiamondLeanError::SemanticRole(error.to_string()))?;
    let shape_analysis = request
        .parameters
        .compiler
        .shape
        .analyze()
        .map_err(|error| DiamondLeanError::SemanticRole(error.to_string()))?;
    let config = &request.parameters.compiler.config;
    let expected = BTreeMap::from([
        (
            BooleanCircuitFamilyParams::INSTANCE_WIDTH_PARAMETER.to_owned(),
            BigInt::from(request.parameters.compiler.shape.instance_width),
        ),
        (
            BooleanCircuitFamilyParams::WITNESS_WIDTH_PARAMETER.to_owned(),
            BigInt::from(request.parameters.compiler.shape.witness_width),
        ),
        (
            BooleanCircuitFamilyParams::DEPTH_PARAMETER.to_owned(),
            BigInt::from(shape_analysis.depth),
        ),
        (
            BooleanCircuitFamilyParams::MAX_LAYER_WIDTH_PARAMETER.to_owned(),
            BigInt::from(shape_analysis.maximum_layer_width),
        ),
        (DiamondGraphParams::MODULUS.to_owned(), BigInt::from(request.parameters.modulus.clone())),
        (
            DiamondGraphParams::RING_DIMENSION.to_owned(),
            BigInt::from(request.parameters.ring_dimension),
        ),
        (DiamondGraphParams::INPUT_COUNT.to_owned(), BigInt::from(config.input_count)),
        (DiamondGraphParams::DIGIT_BASE.to_owned(), BigInt::from(config.digit_base)),
        (DiamondGraphParams::BATCH_BITS.to_owned(), BigInt::from(config.batch_bits)),
        (DiamondGraphParams::GADGET_BASE.to_owned(), config.gadget_base.clone()),
        (DiamondGraphParams::DIGIT_COUNT.to_owned(), BigInt::from(config.digit_count)),
        (DiamondGraphParams::ERROR_BOUND.to_owned(), config.error_max_coefficient_bound.clone()),
        (
            DiamondGraphParams::PREIMAGE_BOUND.to_owned(),
            config.preimage_max_coefficient_bound.clone(),
        ),
    ]);
    for stage_index in [encryption_stage, decryption_stage] {
        let stage = projection.stages.get(stage_index).ok_or_else(|| {
            DiamondLeanError::SemanticRole(format!(
                "resolved stage {stage_index} is missing from the concrete projection"
            ))
        })?;
        validate_binding_map(&stage.key, &stage.bindings.integers, &expected)?;
    }
    Ok(())
}

fn validate_binding_map(
    stage_name: &str,
    actual: &BTreeMap<String, BigInt>,
    expected: &BTreeMap<String, BigInt>,
) -> Result<(), DiamondLeanError> {
    if actual != expected {
        return Err(DiamondLeanError::SemanticRole(format!(
            "stage {stage_name:?} parameter bindings differ from the compiler parameters"
        )));
    }
    Ok(())
}

fn validate_bound(bound: &BoundData) -> Result<(), DiamondLeanError> {
    if bound.schema_version != DIAMOND_BOUND_SCHEMA_VERSION {
        return Err(DiamondLeanError::Bound(format!(
            "unsupported schema version {}, expected {}",
            bound.schema_version, DIAMOND_BOUND_SCHEMA_VERSION
        )));
    }
    let evaluated = bound
        .expression
        .evaluate(&bound.environment)
        .map_err(|error| DiamondLeanError::Bound(error.to_string()))?;
    if evaluated != bound.value {
        return Err(DiamondLeanError::Bound(
            "bound value does not match its expression and environment".to_owned(),
        ));
    }
    Ok(())
}

fn render_bound(source: &mut String, bound: &BoundData) -> Result<(), DiamondLeanError> {
    writeln!(source, "def bound : Mxx.We.DiamondWE.BoundData := {{").unwrap();
    writeln!(source, "  -- mxx-bound-schema-version: {}", bound.schema_version).unwrap();
    writeln!(source, "  -- mxx-bound-expression-rust: {:?}", bound.expression).unwrap();
    writeln!(
        source,
        "  expression := {}",
        render_bound_expr(&bound.expression, &bound.environment)?
    )
    .unwrap();
    source.push_str("  environment := fun parameter => match parameter with\n");
    source.push_str("    | .modulus => ");
    writeln!(source, "{}", bound.environment.modulus).unwrap();
    source.push_str("    | .ringDimension => ");
    writeln!(source, "{}", bound.environment.ring_dimension).unwrap();
    source.push_str("    | .stateRows => ");
    writeln!(source, "{}", bound.environment.state_rows).unwrap();
    source.push_str("    | .stateColumns => ");
    writeln!(source, "{}", bound.environment.state_columns).unwrap();
    source.push_str("    | .gadgetColumns => ");
    writeln!(source, "{}", bound.environment.gadget_columns).unwrap();
    source.push_str("    | .errorCoefficientBound => ");
    writeln!(source, "{}", bound.environment.error_coefficient_bound).unwrap();
    source.push_str("    | .preimageCoefficientBound => ");
    writeln!(source, "{}", bound.environment.preimage_coefficient_bound).unwrap();
    source.push_str("    | .gadgetDecompositionBound => ");
    writeln!(source, "{}", bound.environment.gadget_decomposition_bound).unwrap();
    source.push_str("    | .inputSteps => ");
    writeln!(source, "{}", bound.environment.input_steps).unwrap();
    source.push_str("    | .circuitLayers => ");
    writeln!(source, "{}", bound.environment.circuit_layers).unwrap();
    writeln!(
        source,
        "  -- mxx-bound-environment-rust: modulus={}, ring_dimension={}, state_rows={}, state_columns={}, gadget_columns={}, error_coefficient_bound={}, preimage_coefficient_bound={}, gadget_decomposition_bound={}, input_steps={}, circuit_layers={}",
        bound.environment.modulus,
        bound.environment.ring_dimension,
        bound.environment.state_rows,
        bound.environment.state_columns,
        bound.environment.gadget_columns,
        bound.environment.error_coefficient_bound,
        bound.environment.preimage_coefficient_bound,
        bound.environment.gadget_decomposition_bound,
        bound.environment.input_steps,
        bound.environment.circuit_layers,
    )
    .unwrap();
    Ok(())
}

fn render_bound_expr(
    expression: &BoundExpr,
    environment: &super::bounds::BoundEnvironment,
) -> Result<String, DiamondLeanError> {
    Ok(match expression {
        BoundExpr::Literal(value) => format!(".literal {value}"),
        BoundExpr::Parameter(parameter) => format!(".parameter .{}", parameter_name(*parameter)),
        BoundExpr::Add(terms) => render_fold(terms, ".literal 0", ".add", environment)?,
        BoundExpr::Multiply(terms) => render_fold(terms, ".literal 1", ".mul", environment)?,
        BoundExpr::Maximum(terms) => {
            if terms.is_empty() {
                return Err(DiamondLeanError::Bound(
                    "maximum expression must not be empty".to_owned(),
                ));
            }
            render_fold(terms, ".literal 0", ".max", environment)?
        }
        BoundExpr::NegacyclicProduct { left, right } => format!(
            ".mul (.mul (.literal {}) {}) {} /- rust-node: NegacyclicProduct -/",
            environment.ring_dimension,
            render_bound_expr(left, environment)?,
            render_bound_expr(right, environment)?
        ),
        BoundExpr::MatrixProduct { inner_dimension, left, right } => format!(
            ".mul (.mul {} {}) {} /- rust-node: MatrixProduct -/",
            render_bound_expr(inner_dimension, environment)?,
            format!(
                "(.mul (.literal {}) {})",
                environment.ring_dimension,
                render_bound_expr(left, environment)?
            ),
            render_bound_expr(right, environment)?
        ),
    })
}

fn parameter_name(parameter: BoundParameter) -> &'static str {
    match parameter {
        BoundParameter::Modulus => "modulus",
        BoundParameter::RingDimension => "ringDimension",
        BoundParameter::StateRows => "stateRows",
        BoundParameter::StateColumns => "stateColumns",
        BoundParameter::GadgetColumns => "gadgetColumns",
        BoundParameter::ErrorCoefficientBound => "errorCoefficientBound",
        BoundParameter::PreimageCoefficientBound => "preimageCoefficientBound",
        BoundParameter::GadgetDecompositionBound => "gadgetDecompositionBound",
        BoundParameter::InputSteps => "inputSteps",
        BoundParameter::CircuitLayers => "circuitLayers",
    }
}

fn render_fold(
    terms: &[BoundExpr],
    empty: &str,
    operator: &str,
    environment: &super::bounds::BoundEnvironment,
) -> Result<String, DiamondLeanError> {
    let mut rendered = empty.to_owned();
    for term in terms {
        rendered = format!("{operator} ({rendered}) ({})", render_bound_expr(term, environment)?);
    }
    Ok(rendered)
}

fn require_type(
    name: &str,
    reference: &ConcreteSemanticWireRef,
    predicate: impl FnOnce(&ConcreteWireType) -> bool,
) -> Result<(), DiamondLeanError> {
    if predicate(&reference.wire_type) {
        Ok(())
    } else {
        Err(DiamondLeanError::SemanticRole(format!(
            "{name} has incompatible type {:?}",
            reference.wire_type
        )))
    }
}

fn int_family_shape(
    name: &str,
    reference: &ConcreteSemanticWireRef,
) -> Result<Vec<usize>, DiamondLeanError> {
    match &reference.wire_type {
        ConcreteWireType::Family { element, shape }
            if matches!(element.as_ref(), ConcreteWireType::Int) =>
        {
            Ok(shape.clone())
        }
        other => Err(DiamondLeanError::SemanticRole(format!(
            "{name} must be an integer family, found {other:?}"
        ))),
    }
}

fn common_int_family_shape(
    encryption: &ConcreteSemanticWireRef,
    decryption: &ConcreteSemanticWireRef,
) -> Result<Vec<usize>, DiamondLeanError> {
    let encryption_shape = int_family_shape("encryption instance", encryption)?;
    let decryption_shape = int_family_shape("decryption instance", decryption)?;
    if encryption_shape != decryption_shape {
        return Err(DiamondLeanError::SemanticRole(
            "encryption and decryption instance shapes differ".to_owned(),
        ));
    }
    Ok(encryption_shape)
}

fn render_circuit_refs(
    source: &mut String,
    prefix: &str,
    linked: &ValidatedLinkedProgram,
    circuit: &crate::diamond::DiamondCircuitSemanticRefs,
    shape: &mxx_gadgets::circuit::BooleanCircuitShape,
) -> Result<(), DiamondLeanError> {
    let roles = [
        ("ActiveGateCountsInput", &circuit.active_gate_counts),
        ("GateKindsInput", &circuit.gate_kinds),
        ("LeftSourcesInput", &circuit.left_sources),
        ("RightSourcesInput", &circuit.right_sources),
        ("OutputSourceInput", &circuit.output_sources),
        ("CircuitOutputWire", &circuit.evaluated_output),
    ];
    let resolved = roles
        .iter()
        .map(|(suffix, role)| {
            resolve_role(linked, suffix, role).map(|reference| (*suffix, reference))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let flattened = shape.depth.checked_mul(shape.max_layer_width).ok_or_else(|| {
        DiamondLeanError::SemanticRole("circuit flattened gate count overflowed".to_owned())
    })?;
    let expected_shapes =
        [vec![shape.depth], vec![flattened], vec![flattened], vec![flattened], vec![1]];
    for ((suffix, reference), expected) in resolved.iter().take(5).zip(&expected_shapes) {
        let actual = int_family_shape(&format!("{prefix}{suffix}"), reference)?;
        if &actual != expected {
            return Err(DiamondLeanError::SemanticRole(format!(
                "{prefix}{suffix} family shape must be {expected:?}, found {actual:?}"
            )));
        }
    }
    for (suffix, reference) in resolved.iter().take(5) {
        render_external_input_ref(source, &format!("{prefix}{suffix}"), linked, reference)?;
    }
    let (output_suffix, output_reference) = &resolved[5];
    render_typed_ref(source, &format!("{prefix}{output_suffix}"), output_reference)?;
    writeln!(
        source,
        "def {prefix}Circuit : Mxx.We.DiamondWE.BooleanCircuitInputRefs program circuitShape := {{"
    )
    .unwrap();
    writeln!(source, "  activeGateCountsInput := {prefix}ActiveGateCountsInput").unwrap();
    writeln!(source, "  circuitGateKindsInput := {prefix}GateKindsInput").unwrap();
    writeln!(source, "  circuitLeftSourcesInput := {prefix}LeftSourcesInput").unwrap();
    writeln!(source, "  circuitRightSourcesInput := {prefix}RightSourcesInput").unwrap();
    writeln!(source, "  circuitOutputSourceInput := {prefix}OutputSourceInput\n}}\n").unwrap();
    writeln!(
        source,
        "def {prefix}CircuitOutput : Mxx.We.DiamondWE.AnyTypedWireRef program := ⟨_, {prefix}CircuitOutputWire⟩\n"
    )
    .unwrap();
    Ok(())
}

fn external_input_index(
    program: &ConcreteLinkedProgram,
    name: &str,
    reference: &ConcreteSemanticWireRef,
) -> Result<usize, DiamondLeanError> {
    let stage = program.stages.get(reference.stage).ok_or_else(|| {
        DiamondLeanError::SemanticRole(format!("{name} refers to a missing stage"))
    })?;
    if reference.wire.scope != stage.root_scope {
        return Err(DiamondLeanError::SemanticRole(format!(
            "{name} must refer to the stage root scope"
        )));
    }
    root_external_input_index(stage.root_scope, &stage.scopes, name, reference)
}

fn root_external_input_index(
    root_scope: usize,
    scopes: &[ConcreteScope],
    name: &str,
    reference: &ConcreteSemanticWireRef,
) -> Result<usize, DiamondLeanError> {
    let scope = scopes
        .iter()
        .find(|scope| scope.id == root_scope)
        .ok_or_else(|| DiamondLeanError::SemanticRole(format!("{name} root scope is missing")))?;
    let node = scope.nodes.get(reference.wire.node.0 as usize).ok_or_else(|| {
        DiamondLeanError::SemanticRole(format!("{name} refers to a missing root node"))
    })?;
    match &node.kind {
        ConcreteNodePayload::Input { artifact: None, .. } => Ok(reference.wire.node.0 as usize),
        ConcreteNodePayload::Input { artifact: Some(_), .. } => {
            Err(DiamondLeanError::SemanticRole(format!("{name} refers to an artifact input")))
        }
        _ => Err(DiamondLeanError::SemanticRole(format!(
            "{name} must refer to an external input node"
        ))),
    }
}

fn render_external_input_ref(
    source: &mut String,
    name: &str,
    linked: &ValidatedLinkedProgram,
    reference: &ConcreteSemanticWireRef,
) -> Result<(), DiamondLeanError> {
    let projection = linked.semantic_projection().map_err(|error| {
        DiamondLeanError::SemanticRole(format!("cannot inspect {name}: {error}"))
    })?;
    let input_index = external_input_index(&projection, name, reference)?;
    let wire_name = format!("{name}Wire");
    render_typed_ref(source, &wire_name, reference)?;
    let wire_type = render_wire_type(&reference.wire_type)?;
    writeln!(
        source,
        "def {name} : Mxx.We.DiamondWE.TypedExternalInputRef program ({wire_type}) := {{"
    )
    .unwrap();
    writeln!(source, "  reference := {wire_name}").unwrap();
    writeln!(source, "  inputIndex := {input_index}").unwrap();
    source.push_str("  input_stored := by rfl\n}\n\n");
    Ok(())
}

fn render_typed_ref(
    source: &mut String,
    name: &str,
    reference: &ConcreteSemanticWireRef,
) -> Result<(), DiamondLeanError> {
    let wire_type = render_wire_type(&reference.wire_type)?;
    writeln!(source, "def {name} : Mxx.We.DiamondWE.TypedWireRef program ({wire_type}) := {{")
        .unwrap();
    writeln!(source, "  stage := {}", reference.stage).unwrap();
    source.push_str("  stage_valid := by decide\n");
    writeln!(
        source,
        "  wire := {{ scope := {}, node := {}, port := {} }}",
        reference.wire.scope, reference.wire.node.0, reference.wire.port.0
    )
    .unwrap();
    source.push_str("  type_correct := by rfl\n}\n\n");
    Ok(())
}

fn render_wire_type(wire_type: &ConcreteWireType) -> Result<String, DiamondLeanError> {
    Ok(match wire_type {
        ConcreteWireType::ConstantInt => ".constantInt".to_owned(),
        ConcreteWireType::ConstantReal => ".constantReal".to_owned(),
        ConcreteWireType::ConstantBool => ".constantBool".to_owned(),
        ConcreteWireType::Int => ".int".to_owned(),
        ConcreteWireType::Real => ".real".to_owned(),
        ConcreteWireType::Bool => ".bool".to_owned(),
        ConcreteWireType::Bytes { length } => format!(".bytes {length}"),
        ConcreteWireType::Matrix(matrix) => format!(".matrix {}", render_matrix_type(matrix)),
        ConcreteWireType::Trapdoor {
            matrix,
            sigma,
            gadget_base,
            digit_count,
            preimage_max_coefficient_bound,
        } => format!(
            ".trapdoor {{ matrix := {}, sigma := {}, gadgetBase := .literal {}, digitCount := .literal {}, preimageMaxCoefficientBound := .literal {} }}",
            render_matrix_type(matrix),
            render_source_real_expr(sigma),
            gadget_base,
            digit_count,
            preimage_max_coefficient_bound
        ),
        ConcreteWireType::Preimage(matrix) => format!(".preimage {}", render_matrix_type(matrix)),
        ConcreteWireType::Family { element, shape } => {
            format!(".family {} ({})", render_nat_list(shape), render_wire_type(element)?)
        }
        other => {
            return Err(DiamondLeanError::SemanticRole(format!(
                "semantic role uses unsupported Lean wire type {other:?}"
            )));
        }
    })
}

fn render_source_real_expr(expr: &mxx_ir_core::expr::RealExpr) -> String {
    use mxx_ir_core::expr::RealExpr as E;
    match expr {
        E::Rational(value) => format!(
            ".literal {{ numerator := {}, denominator := {} }}",
            value.numerator(),
            value.denominator()
        ),
        E::Var(value) => panic!("unresolved real parameter in closed wire type: {value}"),
        E::FromInt(value) => format!(".fromInt {:?}", value),
        E::Add(a, b) => {
            format!(".add {} {}", render_source_real_expr(a), render_source_real_expr(b))
        }
        E::Sub(a, b) => {
            format!(".subtract {} {}", render_source_real_expr(a), render_source_real_expr(b))
        }
        E::Mul(a, b) => {
            format!(".multiply {} {}", render_source_real_expr(a), render_source_real_expr(b))
        }
        E::Div(a, b) => {
            format!(".divide {} {}", render_source_real_expr(a), render_source_real_expr(b))
        }
        E::Sqrt(a) => format!(".sqrt {}", render_source_real_expr(a)),
    }
}

fn render_matrix_type(matrix: &mxx_ir_core::types::ConcreteMatrixType) -> String {
    format!(
        "{{ modulus := {}, ringDimension := {}, rows := {}, columns := {} }}",
        matrix.modulus, matrix.ring_dimension, matrix.rows, matrix.columns
    )
}

fn render_nat_list(values: &[usize]) -> String {
    format!("[{}]", values.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::{ConcreteNode, ConcreteWireRef, NodeId, Port, types::ConcreteMatrixType};

    fn test_matrix_type() -> ConcreteMatrixType {
        ConcreteMatrixType { modulus: 97u8.into(), ring_dimension: 8, rows: 2, columns: 3 }
    }

    #[test]
    fn family_matrix_wire_type_parenthesizes_the_element() {
        let wire_type = ConcreteWireType::Family {
            element: Box::new(ConcreteWireType::Matrix(test_matrix_type())),
            shape: vec![2, 5],
        };

        assert_eq!(
            render_wire_type(&wire_type).unwrap(),
            ".family [2, 5] (.matrix { modulus := 97, ringDimension := 8, rows := 2, columns := 3 })"
        );
    }

    #[test]
    fn family_preimage_wire_type_parenthesizes_the_element() {
        let wire_type = ConcreteWireType::Family {
            element: Box::new(ConcreteWireType::Preimage(test_matrix_type())),
            shape: vec![7],
        };

        assert_eq!(
            render_wire_type(&wire_type).unwrap(),
            ".family [7] (.preimage { modulus := 97, ringDimension := 8, rows := 2, columns := 3 })"
        );
    }

    #[test]
    fn nested_family_wire_type_parenthesizes_each_element() {
        let wire_type = ConcreteWireType::Family {
            element: Box::new(ConcreteWireType::Family {
                element: Box::new(ConcreteWireType::Matrix(test_matrix_type())),
                shape: vec![3],
            }),
            shape: vec![2],
        };

        assert_eq!(
            render_wire_type(&wire_type).unwrap(),
            ".family [2] (.family [3] (.matrix { modulus := 97, ringDimension := 8, rows := 2, columns := 3 }))"
        );
    }

    #[test]
    fn lean_render_groups_nested_structural_integer_constructors() {
        use mxx_ir_core::linked::ConcreteStructuralIntExpr as E;

        let expression = E::RoundDivide(
            Box::new(E::Sub(Box::new(E::Literal(257.into())), Box::new(E::Literal(2.into())))),
            Box::new(E::Literal(4.into())),
        );

        assert_eq!(
            render_structural_int_expr(&expression),
            "(.roundDivide (.subtract (.literal 257) (.literal 2)) (.literal 4))"
        );
    }

    #[test]
    fn lean_render_uses_index_map_constructor_names() {
        use mxx_ir_core::linked::ConcreteIndexMapExpr as E;

        let left = || Box::new(E::StructuralSlot(0));
        let right = || Box::new(E::Literal(2.into()));
        assert_eq!(
            render_index_map_expr(&E::Sub(left(), right())),
            "(.sub (.structuralSlot 0) (.literal 2))"
        );
        assert_eq!(
            render_index_map_expr(&E::Mul(left(), right())),
            "(.mul (.structuralSlot 0) (.literal 2))"
        );
        assert_eq!(
            render_index_map_expr(&E::EuclideanDivide(left(), right())),
            "(.divide (.structuralSlot 0) (.literal 2))"
        );
        assert_eq!(
            render_index_map_expr(&E::EuclideanRemainder(left(), right())),
            "(.remainder (.structuralSlot 0) (.literal 2))"
        );
    }

    #[test]
    fn lean_render_uses_case_sensitive_confidentiality_constructors() {
        use mxx_ir_core::artifact::ArtifactConfidentiality;

        assert_eq!(render_confidentiality(ArtifactConfidentiality::Public), ".Public");
        assert_eq!(render_confidentiality(ArtifactConfidentiality::Private), ".Private");
    }

    #[test]
    fn traversal_and_absolute_paths_are_rejected() {
        assert!(validate_relative_path(Path::new("../Program.lean")).is_err());
        assert!(validate_relative_path(Path::new("/tmp/Program.lean")).is_err());
        assert!(validate_relative_path(Path::new("a\\b")).is_err());
    }

    #[test]
    fn write_and_check_are_bytewise() {
        let root = tempfile::tempdir().unwrap();
        let manifest = LeanFileManifest::new([GeneratedLeanFile {
            path: PathBuf::from("Program.lean"),
            contents: b"exact bytes".to_vec(),
        }])
        .unwrap();
        manifest.emit(root.path(), EmitMode::Write).unwrap();
        manifest.emit(root.path(), EmitMode::Check).unwrap();
        fs::write(root.path().join("Program.lean"), b"changed").unwrap();
        assert!(matches!(
            manifest.emit(root.path(), EmitMode::Check),
            Err(ManifestError::Stale { .. })
        ));
    }

    #[test]
    fn duplicate_paths_are_rejected() {
        let result = LeanFileManifest::new([
            GeneratedLeanFile { path: PathBuf::from("Claim.lean"), contents: Vec::new() },
            GeneratedLeanFile { path: PathBuf::from("Claim.lean"), contents: Vec::new() },
        ]);
        assert!(matches!(result, Err(ManifestError::DuplicatePath)));
    }

    #[test]
    fn computed_wire_cannot_fill_external_input_role() {
        let scope = ConcreteScope {
            id: 0,
            structural_slots: Vec::new(),
            nodes: vec![ConcreteNode {
                kind: ConcreteNodePayload::ConstantInt(0.into()),
                arguments: Vec::new(),
                outputs: vec![ConcreteWireType::Int],
                child_scope: None,
            }],
            inputs: Vec::new(),
            outputs: Vec::new(),
        };
        let reference = ConcreteSemanticWireRef {
            stage: 0,
            wire: ConcreteWireRef { scope: 0, node: NodeId(0), port: Port(0) },
            wire_type: ConcreteWireType::Int,
        };
        assert!(matches!(
            root_external_input_index(0, &[scope], "mutatedRole", &reference),
            Err(DiamondLeanError::SemanticRole(message))
                if message.contains("external input node")
        ));
    }

    #[test]
    fn bound_rendering_commits_expression_nodes_and_environment() {
        let environment = super::super::bounds::BoundEnvironment {
            modulus: 97u8.into(),
            ring_dimension: 8u8.into(),
            state_rows: 2u8.into(),
            state_columns: 3u8.into(),
            gadget_columns: 2u8.into(),
            error_coefficient_bound: 2u8.into(),
            preimage_coefficient_bound: 5u8.into(),
            gadget_decomposition_bound: 4u8.into(),
            input_steps: 13u8.into(),
            circuit_layers: 11u8.into(),
        };
        let expression = BoundExpr::add([
            BoundExpr::parameter(BoundParameter::Modulus),
            BoundExpr::multiply([
                BoundExpr::parameter(BoundParameter::CircuitLayers),
                BoundExpr::negacyclic_product(BoundExpr::literal(2u8), BoundExpr::literal(3u8)),
            ]),
            BoundExpr::Maximum(vec![
                BoundExpr::parameter(BoundParameter::InputSteps),
                BoundExpr::matrix_product(
                    BoundExpr::parameter(BoundParameter::StateColumns),
                    BoundExpr::literal(2u8),
                    BoundExpr::parameter(BoundParameter::PreimageCoefficientBound),
                ),
            ]),
        ]);
        let value = expression.evaluate(&environment).unwrap();
        let bound = BoundData {
            schema_version: DIAMOND_BOUND_SCHEMA_VERSION,
            expression,
            environment,
            value,
        };
        let mut source = String::new();
        render_bound(&mut source, &bound).unwrap();
        for marker in [
            ".add",
            ".mul",
            ".max",
            "NegacyclicProduct",
            "MatrixProduct",
            "Modulus",
            "CircuitLayers",
            "StateColumns",
            "InputSteps",
            "modulus=97",
            "ring_dimension=8",
            "state_rows=2",
            "state_columns=3",
            "gadget_columns=2",
            "error_coefficient_bound=2",
            "preimage_coefficient_bound=5",
            "gadget_decomposition_bound=4",
            "input_steps=13",
            "circuit_layers=11",
        ] {
            assert!(source.contains(marker), "bound source missing {marker}: {source}");
        }
        assert!(source.contains("expression :="));
        for parameter in
            ["modulus", "circuitLayers", "stateColumns", "inputSteps", "preimageCoefficientBound"]
        {
            assert!(source.contains(&format!(".parameter .{parameter}")));
        }
        assert_eq!(bound.value, 865u32.into());
    }

    #[test]
    fn bound_schema_and_value_mismatches_fail_closed() {
        let environment = super::super::bounds::BoundEnvironment {
            modulus: 97u8.into(),
            ring_dimension: 8u8.into(),
            state_rows: 2u8.into(),
            state_columns: 2u8.into(),
            gadget_columns: 1u8.into(),
            error_coefficient_bound: 1u8.into(),
            preimage_coefficient_bound: 1u8.into(),
            gadget_decomposition_bound: 1u8.into(),
            input_steps: 1u8.into(),
            circuit_layers: 1u8.into(),
        };
        let expression = BoundExpr::literal(3u8);
        let mut bound = BoundData {
            schema_version: DIAMOND_BOUND_SCHEMA_VERSION,
            expression,
            environment,
            value: 3u8.into(),
        };
        bound.value = 4u8.into();
        assert!(matches!(validate_bound(&bound), Err(DiamondLeanError::Bound(message)) if
            message.contains("does not match")));
        bound.value = 3u8.into();
        bound.schema_version += 1;
        assert!(matches!(validate_bound(&bound), Err(DiamondLeanError::Bound(message)) if
            message.contains("schema version")));
    }

    #[test]
    fn parameter_binding_map_rejects_value_name_and_missing_mutations() {
        let expected = BTreeMap::from([
            ("diamond_modulus".to_owned(), BigInt::from(97)),
            ("diamond_ring_dimension".to_owned(), BigInt::from(8)),
        ]);
        validate_binding_map("diamond-we-encryption", &expected, &expected).unwrap();

        let mut wrong_value = expected.clone();
        wrong_value.insert("diamond_modulus".to_owned(), BigInt::from(101));
        assert!(validate_binding_map("diamond-we-encryption", &wrong_value, &expected).is_err());

        let wrong_name = BTreeMap::from([
            ("modulus".to_owned(), BigInt::from(97)),
            ("diamond_ring_dimension".to_owned(), BigInt::from(8)),
        ]);
        assert!(validate_binding_map("diamond-we-encryption", &wrong_name, &expected).is_err());

        let missing = BTreeMap::from([("diamond_modulus".to_owned(), BigInt::from(97))]);
        assert!(validate_binding_map("diamond-we-encryption", &missing, &expected).is_err());
    }

    #[test]
    fn generated_parameter_match_is_anchored_to_both_stage_indices() {
        let source = render_parameter_match(3, 7);
        assert!(source.contains("Mxx.Generated.stage3"));
        assert!(source.contains("Mxx.Generated.stage7"));
    }
}
