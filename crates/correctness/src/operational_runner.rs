//! Fail-closed execution of generated Lean operational checks.

use crate::{EmittedProtocol, GENERATOR_VERSION, emit_lean::EmittedOperationalDecoderTarget};
use mxx_ir_core::{ParamEnv, expr::ExprError};
use num_bigint::BigInt;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::Instant,
};
use tempfile::Builder;
use thiserror::Error;

pub const OPERATIONAL_REPORT_SCHEMA_VERSION: u32 = 5;
const OPERATIONAL_PREPARED_CACHE_FORMAT_VERSION: u32 = 7;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OperationalParameterValue {
    Integer(BigInt),
    Rational { numerator: BigInt, denominator: BigInt },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperationalGadgetLayout {
    pub params_id: String,
    pub ring_dimension: usize,
    pub crt_moduli: Vec<u64>,
    pub crt_bits: usize,
    pub base_bits: usize,
    pub base: BigInt,
    pub regular_digit_count: usize,
    pub small_digit_count: usize,
    pub smallest_crt_modulus: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperationalCheckRequest {
    pub environment: Vec<(String, OperationalParameterValue)>,
    pub layouts: Vec<OperationalGadgetLayout>,
    /// Identifies one closed decoder target emitted with the protocol.  The target, rather than
    /// the caller, supplies the residual, decoder kind, and modulus expressions.
    pub target_id: String,
}

/// The closed checker report.  This is deliberately an exact wire schema: accepting fields from
/// a newer or different checker would weaken the fail-closed report boundary.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationalCheckerReport {
    pub schema_version: u32,
    pub protocol_source_hash: String,
    pub workflow_hash: String,
    pub derivation_hash: String,
    pub toolkit_hash: String,
    pub request_digest: String,
    pub target_id: String,
    pub decoder_kind: String,
    pub noise_bound: String,
    pub plaintext_modulus: String,
    pub ciphertext_modulus: String,
    pub accepted: bool,
    pub rejection: Option<String>,
    pub decode_time_ns: u64,
    pub evaluation_time_ns: u64,
    pub bound_evaluation_time_ns: u64,
    pub expression_node_count: u64,
    pub memo_evaluations: u64,
    pub memo_hits: u64,
    pub memo_misses: u64,
    pub peak_memo_entries: u64,
    pub envelope_logical_branch_count: u64,
    pub envelope_stored_branch_count: u64,
    pub relation_rewrite_count: u64,
    pub transform_cache_hits: u64,
    pub transform_cache_misses: u64,
    pub cartesian_pair_visits: u64,
    pub maximum_polynomial_terms: u64,
}

/// A fail-closed, machine-readable diagnostic emitted when Lean cannot construct an operational
/// report.  Fields are optional because each `OperationalError` constructor carries a different
/// amount of location information; `reason` is always present.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationalFailureDiagnostic {
    #[serde(deserialize_with = "deserialize_present_option")]
    pub target_id: Option<String>,
    #[serde(deserialize_with = "deserialize_present_option")]
    pub stage: Option<String>,
    #[serde(deserialize_with = "deserialize_present_option")]
    pub scope: Option<String>,
    #[serde(deserialize_with = "deserialize_present_option")]
    pub node: Option<u64>,
    #[serde(deserialize_with = "deserialize_present_option")]
    pub wire: Option<String>,
    #[serde(deserialize_with = "deserialize_present_option")]
    pub context: Option<String>,
    #[serde(deserialize_with = "deserialize_present_option")]
    pub operation: Option<String>,
    #[serde(deserialize_with = "deserialize_present_option")]
    pub relation_owner: Option<String>,
    #[serde(deserialize_with = "deserialize_present_option")]
    pub expected_identity: Option<String>,
    #[serde(deserialize_with = "deserialize_present_option")]
    pub actual_identity: Option<String>,
    pub reason: String,
}

fn deserialize_present_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedOperationalChecker {
    module_name: String,
    namespace: String,
    prepared_name: String,
    protocol_source_hash: String,
    workflow_hash: String,
    derivation_hash: String,
    toolkit_hash: String,
    operational_decoder_targets: Vec<EmittedOperationalDecoderTarget>,
    olean_path: PathBuf,
}

impl PreparedOperationalChecker {
    pub fn olean_path(&self) -> &Path {
        &self.olean_path
    }
}

#[derive(Debug, Error)]
pub enum OperationalRunnerError {
    #[error("could not create the temporary Lean checker: {0}")]
    Temporary(#[from] std::io::Error),
    #[error("Lean operational checker failed; stdout: {stdout}; stderr: {stderr}")]
    CheckerFailed {
        stdout: String,
        stderr: String,
        diagnostic: Option<OperationalFailureDiagnostic>,
    },
    #[error("Lean operational checker must emit exactly one nonempty JSON line, got {count}")]
    UnexpectedOutput { count: usize },
    #[error("Lean operational checker emitted {actual} reports, expected {expected}")]
    UnexpectedReportCount { expected: usize, actual: usize },
    #[error("Lean operational checker emitted malformed JSON: {0}")]
    Malformed(#[from] serde_json::Error),
    #[error("unsupported Lean operational report schema {actual}")]
    Schema { actual: u32 },
    #[error("Lean operational report freshness hashes do not match the emitted protocol")]
    Freshness,
    #[error("Lean operational report target id does not match the request")]
    Target,
    #[error("could not evaluate closed operational target {target_id} {field}: {source}")]
    TargetEvaluation {
        target_id: String,
        field: &'static str,
        #[source]
        source: ExprError,
    },
    #[error("Lean operational report decoder or modulus fields do not match the closed target")]
    TargetEcho,
    #[error("Lean operational report request digest does not match the request")]
    RequestDigest,
    #[error("Lean operational report contains an invalid {field}: {value}")]
    InvalidReportField { field: &'static str, value: String },
    #[error(
        "could not compile prepared Lean operational module; stdout: {stdout}; stderr: {stderr}"
    )]
    PreparationFailed { stdout: String, stderr: String },
}

fn lean_string(value: &str) -> String {
    let escaped = value
        .chars()
        .flat_map(|character| match character {
            '\\' => "\\\\".chars().collect::<Vec<_>>(),
            '\"' => "\\\"".chars().collect::<Vec<_>>(),
            '\n' => "\\n".chars().collect::<Vec<_>>(),
            '\r' => "\\r".chars().collect::<Vec<_>>(),
            '\t' => "\\t".chars().collect::<Vec<_>>(),
            character => vec![character],
        })
        .collect::<String>();
    format!("\"{escaped}\"")
}

const OPERATIONAL_REPORT_LEAN_HELPERS: &str = r#"
private def jsonString (value : String) : String := (Lean.Json.str value).compress

private def emitOperationalProgress
    (phase event : String) (targetId detail : Option String) : IO Unit :=
  let targetField := match targetId with
    | none => ""
    | some target => " target_id=" ++ target
  let detailField := match detail with
    | none => ""
    | some value => " detail=" ++ value
  IO.eprintln ("operational_progress phase=" ++ phase ++ " event=" ++ event ++
    targetField ++ detailField)

private def rejectionJson : Option OperationalNoiseRejection -> String
  | none => "null"
  | some rejection => jsonString (reprStr rejection)

private def operationalReportJson
    (schemaVersion : Nat)
    (protocolSourceHash workflowHash derivationHash toolkitHash requestDigest targetId decoderKind : String)
    (noiseBound plaintextModulus ciphertextModulus : Int)
    (report : OperationalNoiseCheckReport)
    (decodeTimeNs evaluationTimeNs boundEvaluationTimeNs : Nat) : String :=
  "{\"schema_version\":" ++ toString schemaVersion ++
  ",\"protocol_source_hash\":" ++ jsonString protocolSourceHash ++
  ",\"workflow_hash\":" ++ jsonString workflowHash ++
  ",\"derivation_hash\":" ++ jsonString derivationHash ++
  ",\"toolkit_hash\":" ++ jsonString toolkitHash ++
  ",\"request_digest\":" ++ jsonString requestDigest ++
  ",\"target_id\":" ++ jsonString targetId ++
  ",\"decoder_kind\":" ++ jsonString decoderKind ++
  ",\"noise_bound\":" ++ jsonString (toString noiseBound) ++
  ",\"plaintext_modulus\":" ++ jsonString (toString plaintextModulus) ++
  ",\"ciphertext_modulus\":" ++ jsonString (toString ciphertextModulus) ++
  ",\"accepted\":" ++ (if report.accepted then "true" else "false") ++
  ",\"rejection\":" ++ rejectionJson report.rejection ++
  ",\"decode_time_ns\":" ++ toString decodeTimeNs ++
  ",\"evaluation_time_ns\":" ++ toString evaluationTimeNs ++
  ",\"bound_evaluation_time_ns\":" ++ toString boundEvaluationTimeNs ++
  ",\"expression_node_count\":" ++ toString report.diagnostics.expressionNodeCount ++
  ",\"memo_evaluations\":" ++ toString report.diagnostics.memoEvaluations ++
  ",\"memo_hits\":" ++ toString report.diagnostics.memoHits ++
  ",\"memo_misses\":" ++ toString report.diagnostics.memoMisses ++
  ",\"peak_memo_entries\":" ++ toString report.diagnostics.peakMemoEntries ++
  ",\"envelope_logical_branch_count\":" ++ toString report.diagnostics.envelopeLogicalBranchCount ++
  ",\"envelope_stored_branch_count\":" ++ toString report.diagnostics.envelopeStoredBranchCount ++
  ",\"relation_rewrite_count\":" ++ toString report.diagnostics.relationRewriteCount ++
  ",\"transform_cache_hits\":" ++ toString report.diagnostics.transformCacheHits ++
  ",\"transform_cache_misses\":" ++ toString report.diagnostics.transformCacheMisses ++
  ",\"cartesian_pair_visits\":" ++ toString report.diagnostics.cartesianPairVisits ++
  ",\"maximum_polynomial_terms\":" ++ toString report.diagnostics.maximumPolynomialTerms ++ "}"

private structure OperationalFailureFields where
  targetId : Option String := none
  stage : Option String := none
  scope : Option String := none
  node : Option Nat := none
  wire : Option String := none
  context : Option String := none
  operation : Option String := none
  relationOwner : Option String := none
  expectedIdentity : Option String := none
  actualIdentity : Option String := none
  reason : String := ""

private def optionJson (value : Option String) : String :=
  match value with
  | none => "null"
  | some value => jsonString value

private def optionNatJson (value : Option Nat) : String :=
  match value with
  | none => "null"
  | some value => toString value

private def operationalFailureFieldsJson (fields : OperationalFailureFields) : String :=
  "{\"target_id\":" ++ optionJson fields.targetId ++
  ",\"stage\":" ++ optionJson fields.stage ++
  ",\"scope\":" ++ optionJson fields.scope ++
  ",\"node\":" ++ optionNatJson fields.node ++
  ",\"wire\":" ++ optionJson fields.wire ++
  ",\"context\":" ++ optionJson fields.context ++
  ",\"operation\":" ++ optionJson fields.operation ++
  ",\"relation_owner\":" ++ optionJson fields.relationOwner ++
  ",\"expected_identity\":" ++ optionJson fields.expectedIdentity ++
  ",\"actual_identity\":" ++ optionJson fields.actualIdentity ++
  ",\"reason\":" ++ jsonString fields.reason ++ "}"

private def operationalFailureFields (error : OperationalError) : OperationalFailureFields :=
  match error with
  | .inScope scope nested =>
      let fields := operationalFailureFields nested
      { fields with scope := fields.scope.or (some (reprStr scope)), reason := reprStr error }
  | .missingOutputType node port =>
      { node := some node, context := some s!"output port {port}",
        operation := some "missing_output_type", reason := reprStr error }
  | .missingOperand node wire =>
      { node := some node, wire := some (reprStr wire), operation := some "missing_operand",
        reason := reprStr error }
  | .operandNotMatrix node wire =>
      { node := some node, wire := some (reprStr wire), operation := some "operand_not_matrix",
        reason := reprStr error }
  | .operandNotInteger node wire =>
      { node := some node, wire := some (reprStr wire), operation := some "operand_not_integer",
        reason := reprStr error }
  | .operandNotBoolean node wire =>
      { node := some node, wire := some (reprStr wire), operation := some "operand_not_boolean",
        reason := reprStr error }
  | .operandNotReal node wire =>
      { node := some node, wire := some (reprStr wire), operation := some "operand_not_real",
        reason := reprStr error }
  | .invalidMatrixParameters node =>
      { node := some node, operation := some "invalid_matrix_parameters", reason := reprStr error }
  | .flat node flatError =>
      { node := some node, context := some (reprStr flatError), operation := some "flat_error",
        reason := reprStr error }
  | .invalidBound node bound =>
      { node := some node, operation := some "invalid_bound", actualIdentity := some (toString bound),
        reason := reprStr error }
  | .missingPreimageCutoff node =>
      { node := some node, operation := some "missing_preimage_cutoff", reason := reprStr error }
  | .preimageCutoffMismatch node =>
      { node := some node, operation := some "preimage_cutoff_mismatch", reason := reprStr error }
  | .invalidCount node count =>
      { node := some node, operation := some "invalid_count", actualIdentity := some (toString count),
        reason := reprStr error }
  | .missingGadgetLayout node =>
      { node := some node, operation := some "missing_gadget_layout", reason := reprStr error }
  | .ambiguousGadgetLayout node =>
      { node := some node, operation := some "ambiguous_gadget_layout", reason := reprStr error }
  | .invalidGadgetLayout node =>
      { node := some node, operation := some "invalid_gadget_layout", reason := reprStr error }
  | .gadgetLayoutMismatch node =>
      { node := some node, operation := some "gadget_layout_mismatch", reason := reprStr error }
  | .missingPublicIdentity node wire =>
      { node := some node, wire := some (reprStr wire), operation := some "missing_public_identity",
        reason := reprStr error }
  | .missingRelation node wire =>
      { node := some node, wire := some (reprStr wire), operation := some "missing_relation",
        reason := reprStr error }
  | .ambiguousRelation node wire =>
      { node := some node, wire := some (reprStr wire), operation := some "ambiguous_relation",
        reason := reprStr error }
  | .unavailableRelation node wire =>
      { node := some node, wire := some (reprStr wire), operation := some "unavailable_relation",
        reason := reprStr error }
  | .malformedRelation node =>
      { node := some node, operation := some "malformed_relation", reason := reprStr error }
  | .missingDefinition name =>
      { context := some name, operation := some "missing_definition", reason := reprStr error }
  | .definitionFuelExhausted =>
      { operation := some "definition_fuel_exhausted", reason := reprStr error }
  | .publicIdentityMismatch node =>
      { node := some node, operation := some "public_identity_mismatch", reason := reprStr error }
  | .childInputMismatch node expected actual =>
      { node := some node, operation := some "child_input_mismatch",
        expectedIdentity := some (toString expected), actualIdentity := some (toString actual),
        reason := reprStr error }
  | .duplicateInputName name =>
      { context := some name, operation := some "duplicate_input_name", reason := reprStr error }
  | .missingInputNode name =>
      { context := some name, operation := some "missing_input_node", reason := reprStr error }
  | .unexpectedInputNode name =>
      { context := some name, operation := some "unexpected_input_node", reason := reprStr error }
  | .missingChildOutput node port =>
      { node := some node, context := some s!"output port {port}",
        operation := some "missing_child_output", reason := reprStr error }
  | .loopInputModeMismatch node argument =>
      { node := some node, context := some s!"argument {argument}",
        operation := some "loop_input_mode_mismatch", reason := reprStr error }
  | .relationBearingCarriedValue scope node slot =>
      { scope := some (reprStr scope), node := some node, relationOwner := some (toString slot),
        operation := some "relation_bearing_carried_value", reason := reprStr error }
  | .sequentialSchemaMismatch scope node slot initialLargeCounts outputLargeCounts =>
      { scope := some (reprStr scope), node := some node, relationOwner := some (toString slot),
        operation := some "sequential_schema_mismatch",
        expectedIdentity := some (reprStr initialLargeCounts),
        actualIdentity := some (reprStr outputLargeCounts), reason := reprStr error }
  | .divisionByZero =>
      { operation := some "division_by_zero", reason := reprStr error }
  | .negativeDenominator value =>
      { operation := some "negative_denominator", actualIdentity := some (toString value),
        reason := reprStr error }
  | .invalidPreviousPath path =>
      { operation := some "invalid_previous_path", context := some (reprStr path), reason := reprStr error }
  | .nonClosedExpression =>
      { operation := some "non_closed_expression", reason := reprStr error }
  | .derivation derivationError =>
      { operation := some "derivation_error", context := some (reprStr derivationError), reason := reprStr error }
  | .unsupportedOutputArity node actual =>
      { node := some node, operation := some "unsupported_output_arity",
        actualIdentity := some (toString actual), reason := reprStr error }
  | .outputTypeMismatch node =>
      { node := some node, operation := some "output_type_mismatch", reason := reprStr error }
  | .missingStageDerivation stage =>
      { stage := some stage, operation := some "missing_stage_derivation", reason := reprStr error }
  | .missingStageResult stage output =>
      { stage := some stage, context := some output, operation := some "missing_stage_result",
        reason := reprStr error }
  | .invalidOperationalDecoderTarget targetId =>
      { targetId := some targetId, operation := some "invalid_operational_decoder_target",
        reason := reprStr error }
  | .emptyOperationalDecoderTargetRegistry =>
      { operation := some "empty_operational_decoder_target_registry", reason := reprStr error }
  | .unknownOperationalDecoderTarget targetId =>
      { targetId := some targetId, operation := some "unknown_operational_decoder_target",
        reason := reprStr error }
  | .duplicateOperationalDecoderTarget targetId =>
      { targetId := some targetId, operation := some "duplicate_operational_decoder_target",
        reason := reprStr error }
  | .missingProtocolContract name =>
      { context := some name, operation := some "missing_protocol_contract", reason := reprStr error }
  | .inputContractMismatch name =>
      { context := some name, operation := some "input_contract_mismatch", reason := reprStr error }
  | .unknownDerivationAttachment ownerNamespace ruleName =>
      { relationOwner := some (ownerNamespace ++ ":" ++ ruleName),
        operation := some "unknown_derivation_attachment", reason := reprStr error }
  | .missingDerivationAttachmentRole ownerNamespace ruleName roleName =>
      { relationOwner := some (ownerNamespace ++ ":" ++ ruleName ++ ":" ++ roleName),
        operation := some "missing_derivation_attachment_role", reason := reprStr error }
  | .invalidDerivationAttachment ownerNamespace ruleName =>
      { relationOwner := some (ownerNamespace ++ ":" ++ ruleName),
        operation := some "invalid_derivation_attachment", reason := reprStr error }
  | .operationalExprTypeMismatch left right =>
      { operation := some "operational_expression_type_mismatch",
        expectedIdentity := some (toString left), actualIdentity := some (toString right),
        reason := reprStr error }
  | .residualContainsLargeTerm node =>
      { node := some node, operation := some "residual_contains_large_term", reason := reprStr error }
  | .incompatibleRelationDomains node leftDomain rightDomain =>
      { node := some node, operation := some "incompatible_relation_domains",
        expectedIdentity := some (toString leftDomain), actualIdentity := some (toString rightDomain),
        reason := reprStr error }
  | .unknownRelationRequirement node expression =>
      { node := some node, operation := some "unknown_relation_requirement",
        actualIdentity := some (toString expression), reason := reprStr error }
  | .unresolvedConcreteStructure node expression =>
      { node := some node, operation := some "unresolved_concrete_structure",
        actualIdentity := some (toString expression), reason := reprStr error }
  | .unsupportedOperationalExpr id =>
      { operation := some "unsupported_operational_expression", actualIdentity := some (toString id),
        reason := reprStr error }
  | .invalidOperationalExprRef id =>
      { operation := some "invalid_operational_expression_reference",
        actualIdentity := some (toString id), reason := reprStr error }
  | .unsupportedNode node =>
      { node := some node, operation := some "unsupported_node", reason := reprStr error }

private def operationalFailureJson
    (targetId stage : Option String) (error : OperationalError) : String :=
  let fields := operationalFailureFields error
  let context :=
    match stage, fields.context with
    | some phase, some context => some (context ++ "; pipeline_phase=" ++ phase)
    | some phase, none => some ("pipeline_phase=" ++ phase)
    | none, context => context
  operationalFailureFieldsJson
    { fields with
      targetId := fields.targetId.or targetId
      stage := fields.stage.or stage
      context }

private def emitOperationalFailure
    (targetId stage : Option String) (error : OperationalError) : IO Unit :=
  IO.eprintln ("operational_failure=" ++ operationalFailureJson targetId stage error)

private def operationalDecoderKindName : OperationalDecoderKind → String
  | .thresholdDecode _ => "threshold_decode"
  | .booleanInterval => "boolean_interval"

private def emitUnexpectedOperationalObligation
    (targetId : String) (targetKind : OperationalDecoderKind)
    (obligations : List OperationalNoiseObligation) : IO Unit :=
  IO.eprintln ("operational_failure=" ++ operationalFailureFieldsJson {
    targetId := some targetId
    stage := some "target_noise_check"
    context := some (reprStr obligations)
    operation := some "unexpected_operational_obligation"
    actualIdentity := some (operationalDecoderKindName targetKind)
    reason := "operational checker returned obligations incompatible with the closed decoder target"
  })

private def emitPreparationFailure
    (error : Sum Mxx.Ir.DecodeError OperationalError) : IO Unit :=
  match error with
  | .inr error => emitOperationalFailure none (some "prepare_workflow") error
  | .inl error =>
      IO.eprintln ("operational_failure={\"target_id\":null,\"stage\":\"prepare_workflow\",\"scope\":null,\"node\":null,\"wire\":null,\"context\":null,\"operation\":\"decode_error\",\"relation_owner\":null,\"expected_identity\":null,\"actual_identity\":null,\"reason\":" ++ jsonString (reprStr error) ++ "}")
"#;

fn operational_lean_arguments() -> [&'static str; 4] {
    ["env", "lean", "-DmaxHeartbeats=0", "--run"]
}

fn stream_and_retain(
    input: impl Read,
    mut emit: impl FnMut(&[u8]) -> std::io::Result<()>,
) -> std::io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut reader = BufReader::new(input);
    let mut line = Vec::new();
    loop {
        line.clear();
        let count = reader.read_until(b'\n', &mut line)?;
        if count == 0 {
            break;
        }
        emit(&line)?;
        bytes.extend_from_slice(&line);
    }
    Ok(bytes)
}

fn lean_version(lean_workspace: &Path) -> Result<String, OperationalRunnerError> {
    eprintln!("operational_progress phase=lean_toolchain event=version_check_start");
    let output = Command::new("lake")
        .args(["env", "lean", "--version"])
        .current_dir(lean_workspace)
        .output()?;
    if !output.status.success() {
        return Err(OperationalRunnerError::PreparationFailed {
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    let version = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    eprintln!(
        "operational_progress phase=lean_toolchain event=version_check_complete version={version}"
    );
    Ok(version)
}

fn prepared_module_source(
    emitted: &EmittedProtocol,
    prepared_name: &str,
    document_paths: &[(String, PathBuf)],
) -> String {
    let namespace = format!("{}.Generated.{}", emitted.module_root, emitted.lean_name);
    let mut raw_loader_source = emitted.operational_raw_ir_template.clone();
    for (name, path) in document_paths {
        let token = format!("__MXX_OPERATIONAL_DOCUMENT_{name}__");
        raw_loader_source = raw_loader_source.replace(&token, &path.to_string_lossy());
    }
    let source = format!(
        "import Mxx.Certificate.OperationalBounds\n{}\nnamespace {}\n\n\
def {} : IO (Except (Sum Mxx.Ir.DecodeError Mxx.Certificate.OperationalError) \
Mxx.Certificate.PreparedOperationalWorkflow) := do\n  \
IO.eprintln \"operational_progress phase=decode_generated_ir_and_prepare_workflow event=workflow_prepare_start\"\n  \
let preparationStarted ← IO.monoNanosNow\n  \
let decoded ← {}_decodedFromRawFiles\n  \
match decoded with\n  \
| .error error => pure (.error (Sum.inl error))\n  \
| .ok decoded =>\n    \
  let protocol := decoded.1\n    \
  let derivations := decoded.2\n    \
  let result := Mxx.Certificate.prepareWorkflowOperational\n      \
    ({{ workflow := protocol.bundle.workflow, inputContract := protocol.bundle.inputContract, \
    operationalDecoderTargets := protocol.bundle.operationalDecoderTargets }} : \
    Mxx.Certificate.OperationalWorkflowSpec)\n      \
    derivations |>.mapError Sum.inr\n    \
  let preparationFinished ← IO.monoNanosNow\n    \
  IO.eprintln (\"operational_progress phase=decode_generated_ir_and_prepare_workflow \
    event=workflow_prepare_complete detail=elapsed_ns=\" ++ \
    toString (preparationFinished - preparationStarted))\n    \
  pure result\n\nend {}\n",
        raw_loader_source, namespace, prepared_name, emitted.lean_name, namespace,
    );
    source.replace(
        "    let preparationFinished ← IO.monoNanosNow",
        "    let forced ← match result with\n    | .ok prepared => some <$> Mxx.Certificate.forcePreparedOperationalWorkflow prepared\n    | .error _ => pure none\n    let preparationFinished ← IO.monoNanosNow",
    ).replace(
        "toString (preparationFinished - preparationStarted))",
        "toString (preparationFinished - preparationStarted) ++ match forced with\n      | none => \"\"\n      | some stats => \"; forced_entries=\" ++ toString stats.entries ++ \"; checksum=\" ++ toString stats.checksum)",
    )
}

pub fn prepare_emitted_operational_checker(
    lean_workspace: &Path,
    emitted: &EmittedProtocol,
) -> Result<PreparedOperationalChecker, OperationalRunnerError> {
    let preparation_started = Instant::now();
    eprintln!(
        "operational_progress phase=prepare_workflow event=start module={} lean_name={}",
        emitted.module_root, emitted.lean_name
    );
    let version = lean_version(lean_workspace)?;
    let key_material = format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n{}",
        OPERATIONAL_PREPARED_CACHE_FORMAT_VERSION,
        GENERATOR_VERSION,
        emitted.freshness.protocol_source_hash,
        emitted.freshness.workflow_hash,
        emitted.derivation_hash,
        emitted.freshness.toolkit_hash,
        version,
    );
    let key = format!("{:x}", Sha256::digest(key_material.as_bytes()));
    let component = format!("C{key}");
    let module_name = format!("MxxOperationalCache.{component}");
    let cache_root = lean_workspace.join(".lake/build/mxx-operational-cache/src");
    let source_dir = cache_root.join("MxxOperationalCache");
    let document_dir = cache_root.join("documents").join(&component);
    let source_path = source_dir.join(format!("{component}.lean"));
    let olean_dir = lean_workspace.join(".lake/build/lib/lean/MxxOperationalCache");
    let olean_path = olean_dir.join(format!("{component}.olean"));
    let namespace = format!("{}.Generated.{}", emitted.module_root, emitted.lean_name);
    let prepared_name = format!("{}_preparedOperational", emitted.lean_name);
    fs::create_dir_all(&source_dir)?;
    fs::create_dir_all(&olean_dir)?;
    fs::create_dir_all(&document_dir)?;
    let document_paths = emitted
        .operational_documents
        .iter()
        .map(|document| {
            (document.name.clone(), document_dir.join(format!("{}.bin", document.name)))
        })
        .collect::<Vec<_>>();
    for (document, (_, path)) in emitted.operational_documents.iter().zip(&document_paths) {
        fs::write(path, &document.bytes)?;
    }
    if !olean_path.is_file() {
        eprintln!(
            "operational_progress phase=prepare_workflow event=cache_miss component={component}"
        );
        fs::write(&source_path, prepared_module_source(emitted, &prepared_name, &document_paths))?;
        let temporary_olean =
            olean_dir.join(format!("{component}.{}.tmp.olean", std::process::id()));
        let compilation_started = Instant::now();
        eprintln!(
            "operational_progress phase=prepare_workflow event=lean_ir_derivation_build_start component={component}"
        );
        let output = Command::new("lake")
            .args(["env", "lean", "-DmaxHeartbeats=0", "-DmaxRecDepth=1000000", "-R"])
            .arg(&cache_root)
            .arg("-o")
            .arg(&temporary_olean)
            .arg(&source_path)
            .current_dir(lean_workspace)
            .output()?;
        if !output.status.success() {
            let _ = fs::remove_file(&temporary_olean);
            return Err(OperationalRunnerError::PreparationFailed {
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }
        fs::rename(temporary_olean, &olean_path)?;
        eprintln!(
            "operational_progress phase=prepare_workflow event=lean_ir_derivation_build_complete component={component} elapsed_ms={}",
            compilation_started.elapsed().as_millis()
        );
    } else {
        eprintln!(
            "operational_progress phase=prepare_workflow event=cache_hit component={component}"
        );
    }
    eprintln!(
        "operational_progress phase=prepare_workflow event=complete component={component} elapsed_ms={}",
        preparation_started.elapsed().as_millis()
    );
    Ok(PreparedOperationalChecker {
        module_name,
        namespace,
        prepared_name,
        protocol_source_hash: emitted.freshness.protocol_source_hash.clone(),
        workflow_hash: emitted.freshness.workflow_hash.clone(),
        derivation_hash: emitted.derivation_hash.clone(),
        toolkit_hash: emitted.freshness.toolkit_hash.clone(),
        operational_decoder_targets: emitted.operational_decoder_targets.clone(),
        olean_path,
    })
}

fn operational_request_digest(request: &OperationalCheckRequest) -> String {
    let environment = request
        .environment
        .iter()
        .map(|(name, value)| match value {
            OperationalParameterValue::Integer(value) => {
                json!({"name": name, "kind": "integer", "value": value.to_string()})
            }
            OperationalParameterValue::Rational { numerator, denominator } => json!({
                "name": name,
                "kind": "rational",
                "numerator": numerator.to_string(),
                "denominator": denominator.to_string(),
            }),
        })
        .collect::<Vec<_>>();
    let layouts = request
        .layouts
        .iter()
        .map(|layout| {
            json!({
                "params_id": layout.params_id,
                "ring_dimension": layout.ring_dimension,
                "crt_moduli": layout.crt_moduli,
                "crt_bits": layout.crt_bits,
                "base_bits": layout.base_bits,
                "base": layout.base.to_string(),
                "regular_digit_count": layout.regular_digit_count,
                "small_digit_count": layout.small_digit_count,
                "smallest_crt_modulus": layout.smallest_crt_modulus,
            })
        })
        .collect::<Vec<_>>();
    let canonical = serde_json::to_vec(&json!({
        "environment": environment,
        "layouts": layouts,
        "target_id": request.target_id,
    }))
    .expect("operational request JSON serialization");
    format!("{:x}", Sha256::digest(canonical))
}

fn lean_request_values(request: &OperationalCheckRequest) -> (String, String) {
    let environment = request
        .environment
        .iter()
        .map(|(name, value)| match value {
            OperationalParameterValue::Integer(value) => {
                format!("(.parameter {}, .integer {})", lean_string(name), value)
            }
            OperationalParameterValue::Rational { numerator, denominator } => format!(
                "(.parameter {}, .rational (({} : Rat) / ({} : Rat)))",
                lean_string(name),
                numerator,
                denominator
            ),
        })
        .collect::<Vec<_>>()
        .join(", ");
    let layouts = request
        .layouts
        .iter()
        .map(|layout| {
            let crt_moduli =
                layout.crt_moduli.iter().map(u64::to_string).collect::<Vec<_>>().join(", ");
            format!(
                "{{ paramsId := {}, ringDimension := {}, crtModuli := [{}], crtBits := {}, \
                 baseBits := {}, base := {}, regularDigitCount := {}, smallDigitCount := {}, \
                 smallestCrtModulus := {} }}",
                lean_string(&layout.params_id),
                layout.ring_dimension,
                crt_moduli,
                layout.crt_bits,
                layout.base_bits,
                layout.base,
                layout.regular_digit_count,
                layout.small_digit_count,
                layout.smallest_crt_modulus,
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    (environment, layouts)
}

fn prepared_checker_source(
    prepared: &PreparedOperationalChecker,
    requests: &[OperationalCheckRequest],
) -> String {
    let mut group_representatives = Vec::<usize>::new();
    let request_groups = requests
        .iter()
        .enumerate()
        .map(|(request_index, request)| {
            if let Some(group) = group_representatives.iter().position(|representative| {
                requests[*representative].environment == request.environment &&
                    requests[*representative].layouts == request.layouts
            }) {
                group
            } else {
                group_representatives.push(request_index);
                group_representatives.len() - 1
            }
        })
        .collect::<Vec<_>>();
    let mut bound_group_representatives = Vec::<usize>::new();
    let request_bound_groups = requests
        .iter()
        .enumerate()
        .map(|(request_index, request)| {
            if let Some(group) = bound_group_representatives.iter().position(|representative| {
                request_groups[*representative] == request_groups[request_index] &&
                    requests[*representative].target_id == request.target_id
            }) {
                group
            } else {
                bound_group_representatives.push(request_index);
                bound_group_representatives.len() - 1
            }
        })
        .collect::<Vec<_>>();
    let output_definitions = group_representatives
        .iter()
        .enumerate()
        .map(|(group, representative)| {
            let (environment, layouts) = lean_request_values(&requests[*representative]);
            format!(
                "private def operationalOutputs_{group} (prepared : PreparedOperationalWorkflow) :=\n\
                   evaluatePreparedWorkflowOperational prepared [{environment}] [{layouts}]"
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let bound_definitions = bound_group_representatives
        .iter()
        .enumerate()
        .map(|(bound_group, representative)| {
            let request = &requests[*representative];
            let (environment, _) = lean_request_values(request);
            format!(
                "private def operationalBound_{bound_group} (prepared : PreparedOperationalWorkflow) \
                 (outputs : List OperationalStageResult) :=\n\
                   operationalTargetNoiseBound prepared outputs {} [{environment}]",
                lean_string(&request.target_id),
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let output_executions = group_representatives
        .iter()
        .enumerate()
        .map(|(group, representative)| {
            let target_id = lean_string(&requests[*representative].target_id);
            format!(
                "emitOperationalProgress \"evaluate_scope\" \"start\" (some {target_id}) \
                   (some \"request_group={group}; request_index={representative}\")\n\
                 let evaluationStarted_{group} ← IO.monoNanosNow\n\
                 let outputs_{group} ← match operationalOutputs_{group} prepared with\n\
                 | .error error => emitOperationalFailure (some {}) (some \"evaluate_scope\") error; return 2\n\
                 | .ok outputs => pure outputs\n\
                 let evaluationFinished_{group} ← IO.monoNanosNow\n\
                 let evaluationTimeNs_{group} := evaluationFinished_{group} - evaluationStarted_{group}\n\
                 emitOperationalProgress \"evaluate_scope\" \"complete\" (some {target_id}) \
                   (some (\"request_group={group}; elapsed_ns=\" ++ toString evaluationTimeNs_{group}))",
                target_id,
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let bound_executions = bound_group_representatives
        .iter()
        .enumerate()
        .map(|(bound_group, representative)| {
            let output_group = request_groups[*representative];
            let target_id = lean_string(&requests[*representative].target_id);
            format!(
                "emitOperationalProgress \"resolve_target_and_evaluate_bound\" \"start\" \
                   (some {target_id}) (some \"bound_group={bound_group}; output_group={output_group}\")\n\
                 let boundEvaluationStarted_{bound_group} ← IO.monoNanosNow\n\
                 let boundResult_{bound_group} ← match \
                   operationalBound_{bound_group} prepared outputs_{output_group} with\n\
                   | .error error => emitOperationalFailure (some {}) (some \"evaluate_decoder_bounds\") error; return 2\n\
                   | .ok result => pure result\n\
                 let boundEvaluationFinished_{bound_group} ← IO.monoNanosNow\n\
                 let boundEvaluationTimeNs_{bound_group} := \
                   boundEvaluationFinished_{bound_group} - boundEvaluationStarted_{bound_group}\n\
                 emitOperationalProgress \"resolve_target_and_evaluate_bound\" \"complete\" \
                   (some {target_id}) (some (\"bound_group={bound_group}; elapsed_ns=\" ++ \
                     toString boundEvaluationTimeNs_{bound_group}))",
                target_id,
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let executions = requests
        .iter()
        .enumerate()
        .map(|(index, request)| {
            let digest = operational_request_digest(request);
            let group = request_groups[index];
            let bound_group = request_bound_groups[index];
            format!(
                "let (target_{index}, ciphertextModulus_{index}, noiseBound_{index}, diagnostics_{index}) := \
                   boundResult_{bound_group}\n\
                 emitOperationalProgress \"target_noise_check\" \"start\" (some target_{index}.targetId) \
                   (some \"request_index={index}\")\n\
                 let report ← match operationalTargetNoiseCheckReportFromBound outputs_{group} \
                   target_{index} ciphertextModulus_{index} noiseBound_{index} diagnostics_{index} [{}] with\n\
                   | .ok report => pure report\n\
                   | .error error => emitOperationalFailure (some target_{index}.targetId) (some \"target_noise_check\") error; return 2\n\
                 match report.obligations, target_{index}.kind with\n\
                     | [.decoderThreshold plaintextModulus ciphertextModulus noiseBound], \
                       .thresholdDecode _ =>\n\
                         emitOperationalProgress \"target_noise_check\" \"complete\" \
                           (some target_{index}.targetId) (some (\"request_index={index}; accepted=\" ++ \
                             toString report.accepted ++ \"; noise_bound=\" ++ toString noiseBound))\n\
                         let json := operationalReportJson operationalReportSchemaVersion {} {} {} {} {} \
                           target_{index}.targetId \"threshold_decode\" noiseBound plaintextModulus ciphertextModulus \
                           report decodeTimeNs evaluationTimeNs_{group} boundEvaluationTimeNs_{bound_group}\n\
                         IO.println json\n\
                     | [.booleanInterval ciphertextModulus noiseBound], .booleanInterval =>\n\
                         emitOperationalProgress \"target_noise_check\" \"complete\" \
                           (some target_{index}.targetId) (some (\"request_index={index}; accepted=\" ++ \
                             toString report.accepted ++ \"; noise_bound=\" ++ toString noiseBound))\n\
                         let json := operationalReportJson operationalReportSchemaVersion {} {} {} {} {} \
                           target_{index}.targetId \"boolean_interval\" noiseBound 2 ciphertextModulus \
                           report decodeTimeNs evaluationTimeNs_{group} boundEvaluationTimeNs_{bound_group}\n\
                         IO.println json\n\
                     | _, _ => emitUnexpectedOperationalObligation target_{index}.targetId target_{index}.kind \
                       report.obligations; return 3",
                lean_request_values(request).0,
                lean_string(&prepared.protocol_source_hash),
                lean_string(&prepared.workflow_hash),
                lean_string(&prepared.derivation_hash),
                lean_string(&prepared.toolkit_hash),
                lean_string(&digest),
                lean_string(&prepared.protocol_source_hash),
                lean_string(&prepared.workflow_hash),
                lean_string(&prepared.derivation_hash),
                lean_string(&prepared.toolkit_hash),
                lean_string(&digest),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "import {}\nopen {}\nopen Mxx.Certificate\n\nprivate def operationalReportSchemaVersion : Nat := {}\n\n{}\n\n{}\n\n{}\n\n\
         def main : IO UInt32 := do\n\
           emitOperationalProgress \"decode_generated_ir_and_prepare_workflow\" \"start\" none none\n\
           let decodeStarted ← IO.monoNanosNow\n\
           let preparedResult ← {}\n\
           match preparedResult with\n\
           | .error error => emitPreparationFailure error; return 2\n\
           | .ok prepared =>\n\
             let decodeFinished ← IO.monoNanosNow\n\
             let decodeTimeNs := decodeFinished - decodeStarted\n\
             emitOperationalProgress \"decode_generated_ir_and_prepare_workflow\" \"complete\" none \
               (some (\"elapsed_ns=\" ++ toString decodeTimeNs))\n\
{}\n{}\nreturn 0\n",
        prepared.module_name,
        prepared.namespace,
        OPERATIONAL_REPORT_SCHEMA_VERSION,
        OPERATIONAL_REPORT_LEAN_HELPERS,
        output_definitions,
        bound_definitions,
        prepared.prepared_name,
        output_executions,
        format!("{bound_executions}\n{executions}"),
    )
}

fn run_operational_checker_reports(
    lean_workspace: &Path,
    source: &str,
    expected: usize,
) -> Result<Vec<OperationalCheckerReport>, OperationalRunnerError> {
    let runner_started = Instant::now();
    eprintln!(
        "operational_progress phase=lean_checker event=source_generation_complete expected_reports={expected} source_bytes={}",
        source.len()
    );
    let file = Builder::new().prefix("mxx-operational-").suffix(".lean").tempfile()?;
    std::fs::write(file.path(), source)?;
    eprintln!(
        "operational_progress phase=lean_checker event=launch path={} expected_reports={expected}",
        file.path().display()
    );
    let mut child = Command::new("lake")
        .args(operational_lean_arguments())
        .arg(file.path())
        .current_dir(lean_workspace)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stdout = child.stdout.take().expect("piped Lean stdout");
    let stderr = child.stderr.take().expect("piped Lean stderr");
    let stdout_reader = thread::spawn(move || {
        stream_and_retain(stdout, |line| {
            print!("{}", String::from_utf8_lossy(line));
            std::io::stdout().flush()
        })
    });
    let stderr_reader = thread::spawn(move || {
        stream_and_retain(stderr, |line| {
            eprint!("{}", String::from_utf8_lossy(line));
            Ok(())
        })
    });
    let status = child.wait()?;
    let stdout = stdout_reader.join().expect("Lean stdout reader panicked")?;
    let stderr = stderr_reader.join().expect("Lean stderr reader panicked")?;
    let stdout = String::from_utf8_lossy(&stdout).into_owned();
    let stderr = String::from_utf8_lossy(&stderr).into_owned();
    eprintln!(
        "operational_progress phase=lean_checker event=process_exit success={} elapsed_ms={}",
        status.success(),
        runner_started.elapsed().as_millis()
    );
    if !status.success() {
        return Err(OperationalRunnerError::CheckerFailed {
            diagnostic: parse_operational_failure_diagnostic(&stderr),
            stdout,
            stderr,
        });
    }
    parse_operational_checker_reports(&stdout, expected)
}

fn parse_operational_failure_diagnostic(stderr: &str) -> Option<OperationalFailureDiagnostic> {
    stderr
        .lines()
        .filter_map(|line| line.strip_prefix("operational_failure="))
        .next_back()
        .and_then(|json| serde_json::from_str(json).ok())
}

fn parse_operational_checker_reports(
    stdout: &str,
    expected: usize,
) -> Result<Vec<OperationalCheckerReport>, OperationalRunnerError> {
    let lines = stdout.lines().map(str::trim).filter(|line| !line.is_empty()).collect::<Vec<_>>();
    if lines.len() != expected {
        return Err(OperationalRunnerError::UnexpectedReportCount { expected, actual: lines.len() });
    }
    let reports = lines
        .into_iter()
        .map(serde_json::from_str)
        .collect::<Result<Vec<OperationalCheckerReport>, _>>()?;
    for report in &reports {
        if report.schema_version != OPERATIONAL_REPORT_SCHEMA_VERSION {
            return Err(OperationalRunnerError::Schema { actual: report.schema_version });
        }
        validate_report_shape(report)?;
    }
    Ok(reports)
}

fn validate_report_shape(report: &OperationalCheckerReport) -> Result<(), OperationalRunnerError> {
    let noise_bound = report.noise_bound.parse::<BigInt>().map_err(|_| {
        OperationalRunnerError::InvalidReportField {
            field: "noise bound",
            value: report.noise_bound.clone(),
        }
    })?;
    if noise_bound < BigInt::from(0) || noise_bound.to_string() != report.noise_bound {
        return Err(OperationalRunnerError::InvalidReportField {
            field: "noise bound",
            value: report.noise_bound.clone(),
        });
    }
    Ok(())
}

pub fn run_operational_checker_source(
    lean_workspace: &Path,
    source: &str,
) -> Result<OperationalCheckerReport, OperationalRunnerError> {
    let mut reports = run_operational_checker_reports(lean_workspace, source, 1)?;
    reports.pop().ok_or(OperationalRunnerError::UnexpectedOutput { count: 0 })
}

fn validate_prepared_report(
    prepared: &PreparedOperationalChecker,
    request: &OperationalCheckRequest,
    report: &OperationalCheckerReport,
) -> Result<(), OperationalRunnerError> {
    validate_report_shape(report)?;
    if report.protocol_source_hash != prepared.protocol_source_hash ||
        report.workflow_hash != prepared.workflow_hash ||
        report.derivation_hash != prepared.derivation_hash ||
        report.toolkit_hash != prepared.toolkit_hash
    {
        return Err(OperationalRunnerError::Freshness);
    }
    if report.request_digest != operational_request_digest(request) {
        return Err(OperationalRunnerError::RequestDigest);
    }
    if report.target_id != request.target_id {
        return Err(OperationalRunnerError::Target);
    }
    let target = prepared
        .operational_decoder_targets
        .iter()
        .find(|target| target.target_id == request.target_id)
        .ok_or(OperationalRunnerError::Target)?;
    let mut environment = ParamEnv::default();
    let mut bound_names = BTreeSet::new();
    for (name, value) in &request.environment {
        if bound_names.insert(name.as_str()) {
            if let OperationalParameterValue::Integer(value) = value {
                environment.integers.insert(name.clone(), value.clone());
            }
        }
    }
    let plaintext_modulus = target.plaintext_modulus.evaluate(&environment).map_err(|source| {
        OperationalRunnerError::TargetEvaluation {
            target_id: target.target_id.clone(),
            field: "plaintext modulus",
            source,
        }
    })?;
    let ciphertext_modulus =
        target.ciphertext_modulus.evaluate(&environment).map_err(|source| {
            OperationalRunnerError::TargetEvaluation {
                target_id: target.target_id.clone(),
                field: "ciphertext modulus",
                source,
            }
        })?;
    if report.decoder_kind != target.decoder_kind.report_name() ||
        report.plaintext_modulus != plaintext_modulus.to_string() ||
        report.ciphertext_modulus != ciphertext_modulus.to_string()
    {
        return Err(OperationalRunnerError::TargetEcho);
    }
    Ok(())
}

pub fn run_prepared_operational_checks(
    lean_workspace: &Path,
    prepared: &PreparedOperationalChecker,
    requests: &[OperationalCheckRequest],
) -> Result<Vec<OperationalCheckerReport>, OperationalRunnerError> {
    eprintln!(
        "operational_progress phase=prepared_checker event=start request_count={} target_ids={}",
        requests.len(),
        requests.iter().map(|request| request.target_id.as_str()).collect::<Vec<_>>().join(",")
    );
    let reports = run_operational_checker_reports(
        lean_workspace,
        &prepared_checker_source(prepared, requests),
        requests.len(),
    )?;
    for (request, report) in requests.iter().zip(&reports) {
        eprintln!(
            "operational_progress phase=prepared_checker event=validate_report_start target_id={}",
            request.target_id
        );
        validate_prepared_report(prepared, request, report)?;
        eprintln!(
            "operational_progress phase=prepared_checker event=validate_report_complete target_id={} accepted={} noise_bound={}",
            request.target_id, report.accepted, report.noise_bound
        );
    }
    eprintln!(
        "operational_progress phase=prepared_checker event=complete request_count={}",
        reports.len()
    );
    Ok(reports)
}

pub fn run_emitted_operational_check(
    lean_workspace: &Path,
    emitted: &EmittedProtocol,
    request: &OperationalCheckRequest,
) -> Result<OperationalCheckerReport, OperationalRunnerError> {
    let prepared = prepare_emitted_operational_checker(lean_workspace, emitted)?;
    let mut reports =
        run_prepared_operational_checks(lean_workspace, &prepared, std::slice::from_ref(request))?;
    reports.pop().ok_or(OperationalRunnerError::UnexpectedOutput { count: 0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{emit_protocol_for, toy_example};

    #[test]
    fn rejects_multiple_json_lines() {
        let output = "{}\n{}\n";
        let lines =
            output.lines().map(str::trim).filter(|line| !line.is_empty()).collect::<Vec<_>>();
        assert_ne!(lines.len(), 1);
    }

    #[test]
    fn rejects_unknown_schema() {
        let report = OperationalCheckerReport {
            schema_version: OPERATIONAL_REPORT_SCHEMA_VERSION + 1,
            protocol_source_hash: String::new(),
            workflow_hash: String::new(),
            derivation_hash: String::new(),
            toolkit_hash: String::new(),
            request_digest: String::new(),
            target_id: String::new(),
            decoder_kind: String::new(),
            noise_bound: "0".to_owned(),
            plaintext_modulus: "2".to_owned(),
            ciphertext_modulus: "17".to_owned(),
            accepted: true,
            rejection: None,
            decode_time_ns: 0,
            evaluation_time_ns: 0,
            bound_evaluation_time_ns: 0,
            expression_node_count: 0,
            memo_evaluations: 0,
            memo_hits: 0,
            memo_misses: 0,
            peak_memo_entries: 0,
            envelope_logical_branch_count: 0,
            envelope_stored_branch_count: 0,
            relation_rewrite_count: 0,
            transform_cache_hits: 0,
            transform_cache_misses: 0,
            cartesian_pair_visits: 0,
            maximum_polynomial_terms: 0,
        };
        assert!(matches!(
            parse_operational_checker_reports(&serde_json::to_string(&report).unwrap(), 1),
            Err(OperationalRunnerError::Schema { actual })
                if actual == OPERATIONAL_REPORT_SCHEMA_VERSION + 1
        ));
    }

    #[test]
    fn report_wire_schema_rejects_unknown_fields_and_non_numeric_metrics() {
        let report = OperationalCheckerReport {
            schema_version: OPERATIONAL_REPORT_SCHEMA_VERSION,
            protocol_source_hash: "protocol".to_owned(),
            workflow_hash: "workflow".to_owned(),
            derivation_hash: "derivation".to_owned(),
            toolkit_hash: "toolkit".to_owned(),
            request_digest: "request".to_owned(),
            target_id: "target".to_owned(),
            decoder_kind: "threshold_decode".to_owned(),
            noise_bound: "0".to_owned(),
            plaintext_modulus: "2".to_owned(),
            ciphertext_modulus: "17".to_owned(),
            accepted: true,
            rejection: None,
            decode_time_ns: 0,
            evaluation_time_ns: 0,
            bound_evaluation_time_ns: 0,
            expression_node_count: 0,
            memo_evaluations: 0,
            memo_hits: 0,
            memo_misses: 0,
            peak_memo_entries: 0,
            envelope_logical_branch_count: 0,
            envelope_stored_branch_count: 0,
            relation_rewrite_count: 0,
            transform_cache_hits: 0,
            transform_cache_misses: 0,
            cartesian_pair_visits: 0,
            maximum_polynomial_terms: 0,
        };
        let mut unknown = serde_json::to_value(&report).unwrap();
        unknown["unrecognized_metric"] = json!(1);
        assert!(matches!(
            parse_operational_checker_reports(&serde_json::to_string(&unknown).unwrap(), 1),
            Err(OperationalRunnerError::Malformed(_))
        ));

        let mut malformed_metric = serde_json::to_value(&report).unwrap();
        malformed_metric["evaluation_time_ns"] = json!("not-a-number");
        assert!(matches!(
            parse_operational_checker_reports(
                &serde_json::to_string(&malformed_metric).unwrap(),
                1
            ),
            Err(OperationalRunnerError::Malformed(_))
        ));
    }

    #[test]
    fn failure_diagnostic_is_strict_and_retains_available_error_context() {
        let diagnostic = r#"operational_failure={"target_id":"target","stage":null,"scope":"scope","node":4,"wire":"{ node := 3, port := 0 }","context":null,"operation":"missing_relation","relation_owner":null,"expected_identity":null,"actual_identity":null,"reason":"missingRelation 4"}"#;
        let parsed = parse_operational_failure_diagnostic(diagnostic).unwrap();
        assert_eq!(parsed.target_id.as_deref(), Some("target"));
        assert_eq!(parsed.scope.as_deref(), Some("scope"));
        assert_eq!(parsed.node, Some(4));
        assert_eq!(parsed.operation.as_deref(), Some("missing_relation"));

        let complete_json = diagnostic.strip_prefix("operational_failure=").unwrap();
        let complete: serde_json::Value = serde_json::from_str(complete_json).unwrap();
        for nullable_key in [
            "target_id",
            "stage",
            "scope",
            "node",
            "wire",
            "context",
            "operation",
            "relation_owner",
            "expected_identity",
            "actual_identity",
        ] {
            let mut missing = complete.clone();
            missing.as_object_mut().unwrap().remove(nullable_key);
            assert!(
                parse_operational_failure_diagnostic(&format!(
                    "operational_failure={}",
                    serde_json::to_string(&missing).unwrap()
                ))
                .is_none(),
                "diagnostic unexpectedly accepted missing key {nullable_key}"
            );
        }

        let earlier_valid_last_malformed =
            format!("{diagnostic}\noperational_failure={{\"reason\":\"malformed last marker\"}}");
        assert!(parse_operational_failure_diagnostic(&earlier_valid_last_malformed).is_none());

        let mut unknown = complete;
        unknown["unknown"] = json!(null);
        assert!(
            parse_operational_failure_diagnostic(&format!(
                "operational_failure={}",
                serde_json::to_string(&unknown).unwrap()
            ))
            .is_none()
        );
    }

    #[test]
    fn diagnostic_source_preserves_specific_error_location_over_pipeline_context() {
        assert!(
            OPERATIONAL_REPORT_LEAN_HELPERS
                .contains("scope := fields.scope.or (some (reprStr scope))")
        );
        assert!(
            OPERATIONAL_REPORT_LEAN_HELPERS.contains("stage := fields.stage.or stage"),
            "missingStageResult must retain its concrete stage under evaluate_decoder_bounds"
        );
        assert!(
            OPERATIONAL_REPORT_LEAN_HELPERS.contains("targetId := fields.targetId.or targetId")
        );
        assert!(OPERATIONAL_REPORT_LEAN_HELPERS.contains("pipeline_phase="));
        assert!(OPERATIONAL_REPORT_LEAN_HELPERS.contains(
            "| .missingStageResult stage output =>\n      { stage := some stage, context := some output"
        ));
    }

    #[test]
    fn report_shape_rejects_noncanonical_or_negative_noise_bound() {
        let mut report = OperationalCheckerReport {
            schema_version: OPERATIONAL_REPORT_SCHEMA_VERSION,
            protocol_source_hash: "protocol".to_owned(),
            workflow_hash: "workflow".to_owned(),
            derivation_hash: "derivation".to_owned(),
            toolkit_hash: "toolkit".to_owned(),
            request_digest: "request".to_owned(),
            target_id: "target".to_owned(),
            decoder_kind: "threshold_decode".to_owned(),
            noise_bound: "0".to_owned(),
            plaintext_modulus: "2".to_owned(),
            ciphertext_modulus: "17".to_owned(),
            accepted: true,
            rejection: None,
            decode_time_ns: 0,
            evaluation_time_ns: 0,
            bound_evaluation_time_ns: 0,
            expression_node_count: 0,
            memo_evaluations: 0,
            memo_hits: 0,
            memo_misses: 0,
            peak_memo_entries: 0,
            envelope_logical_branch_count: 0,
            envelope_stored_branch_count: 0,
            relation_rewrite_count: 0,
            transform_cache_hits: 0,
            transform_cache_misses: 0,
            cartesian_pair_visits: 0,
            maximum_polynomial_terms: 0,
        };
        validate_report_shape(&report).unwrap();
        for noise_bound in ["-1", "01", "+1", "not-a-number"] {
            report.noise_bound = noise_bound.to_owned();
            assert!(matches!(
                validate_report_shape(&report),
                Err(OperationalRunnerError::InvalidReportField { field: "noise bound", .. })
            ));
        }
    }

    #[test]
    fn operational_runner_disables_lean_heartbeat_timeout() {
        assert!(operational_lean_arguments().contains(&"-DmaxHeartbeats=0"));
    }

    #[test]
    fn lean_string_escapes_source_boundaries() {
        assert_eq!(
            lean_string("target\"slash\\newline\ncarriage\rtab\t"),
            "\"target\\\"slash\\\\newline\\ncarriage\\rtab\\t\""
        );
    }

    #[test]
    fn stream_and_retain_emits_lines_and_preserves_exact_output() {
        let input = b"first line\nsecond line without newline";
        let mut emitted = Vec::new();

        let retained = stream_and_retain(input.as_slice(), |line| {
            emitted.push(line.to_vec());
            Ok(())
        })
        .unwrap();

        assert_eq!(
            emitted,
            vec![b"first line\n".to_vec(), b"second line without newline".to_vec()]
        );
        assert_eq!(retained, input);
    }

    #[test]
    fn prepared_checker_reuses_structural_bound_across_threshold_requests() {
        let prepared = PreparedOperationalChecker {
            module_name: "Test.Prepared".to_owned(),
            namespace: "Test.Generated.Protocol".to_owned(),
            prepared_name: "preparedOperational".to_owned(),
            protocol_source_hash: "protocol".to_owned(),
            workflow_hash: "workflow".to_owned(),
            derivation_hash: "derivation".to_owned(),
            toolkit_hash: "toolkit".to_owned(),
            operational_decoder_targets: vec![EmittedOperationalDecoderTarget {
                target_id: "target".to_owned(),
                decoder_kind: crate::emit_lean::EmittedOperationalDecoderKind::ThresholdDecode,
                plaintext_modulus: mxx_ir_core::IntExpr::constant(2),
                ciphertext_modulus: mxx_ir_core::IntExpr::constant(17),
            }],
            olean_path: PathBuf::from("unused.olean"),
        };
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "target".to_owned(),
        };

        let source = prepared_checker_source(&prepared, &[request.clone(), request]);

        assert_eq!(source.matches("operationalBound_0 prepared outputs_0").count(), 1);
        assert!(!source.contains("operationalBound_1"));
        assert_eq!(source.matches("boundResult_0").count(), 3);
        assert!(source.contains("operationalTargetNoiseBound prepared outputs"));
        assert_eq!(source.matches("operationalTargetNoiseCheckReportFromBound").count(), 2);
        assert!(source.contains("boundEvaluationTimeNs_0"));
        assert!(!source.contains("boundEvaluationTimeNs_1"));
        assert!(source.contains(&format!(
            "operationalReportSchemaVersion : Nat := {OPERATIONAL_REPORT_SCHEMA_VERSION}"
        )));
        assert!(source.contains(
            "operational_progress phase=decode_generated_ir_and_prepare_workflow event=start"
        ));
        assert!(source.contains("operational_progress phase=evaluate_scope event=start"));
        assert!(
            source.contains(
                "operational_progress phase=resolve_target_and_evaluate_bound event=start"
            )
        );
        assert!(source.contains("operational_progress phase=target_noise_check event=complete"));
        assert!(!source.contains("operational analysis diagnostics"));
        assert!(source.contains("report.diagnostics.expressionNodeCount"));
        assert!(source.contains("report.diagnostics.relationRewriteCount"));
        assert!(source.contains("Lean.Json.str"));
        assert!(source.contains("target_0.targetId \"threshold_decode\""));
        assert!(source.contains("target_1.targetId \"threshold_decode\""));
        assert!(source.contains("(some \"evaluate_scope\")"));
        assert!(source.contains("(some \"evaluate_decoder_bounds\")"));
        assert!(source.contains("(some \"target_noise_check\")"));
        assert!(source.contains("emitUnexpectedOperationalObligation"));
        assert!(source.contains("residual_contains_large_term"));
    }

    #[test]
    fn prepared_report_must_echo_the_closed_decoder_kind_and_moduli() {
        let prepared = PreparedOperationalChecker {
            module_name: "Test.Prepared".to_owned(),
            namespace: "Test.Generated.Protocol".to_owned(),
            prepared_name: "preparedOperational".to_owned(),
            protocol_source_hash: "protocol".to_owned(),
            workflow_hash: "workflow".to_owned(),
            derivation_hash: "derivation".to_owned(),
            toolkit_hash: "toolkit".to_owned(),
            operational_decoder_targets: vec![EmittedOperationalDecoderTarget {
                target_id: "target".to_owned(),
                decoder_kind: crate::emit_lean::EmittedOperationalDecoderKind::ThresholdDecode,
                plaintext_modulus: mxx_ir_core::IntExpr::Var("p".to_owned()),
                ciphertext_modulus: mxx_ir_core::IntExpr::Var("q".to_owned()),
            }],
            olean_path: PathBuf::from("unused.olean"),
        };
        let request = OperationalCheckRequest {
            environment: vec![
                ("p".to_owned(), OperationalParameterValue::Integer(BigInt::from(3))),
                ("q".to_owned(), OperationalParameterValue::Integer(BigInt::from(257))),
            ],
            layouts: Vec::new(),
            target_id: "target".to_owned(),
        };
        let report = OperationalCheckerReport {
            schema_version: OPERATIONAL_REPORT_SCHEMA_VERSION,
            protocol_source_hash: "protocol".to_owned(),
            workflow_hash: "workflow".to_owned(),
            derivation_hash: "derivation".to_owned(),
            toolkit_hash: "toolkit".to_owned(),
            request_digest: operational_request_digest(&request),
            target_id: "target".to_owned(),
            decoder_kind: "threshold_decode".to_owned(),
            noise_bound: "4".to_owned(),
            plaintext_modulus: "3".to_owned(),
            ciphertext_modulus: "257".to_owned(),
            accepted: true,
            rejection: None,
            decode_time_ns: 0,
            evaluation_time_ns: 0,
            bound_evaluation_time_ns: 0,
            expression_node_count: 0,
            memo_evaluations: 0,
            memo_hits: 0,
            memo_misses: 0,
            peak_memo_entries: 0,
            envelope_logical_branch_count: 0,
            envelope_stored_branch_count: 0,
            relation_rewrite_count: 0,
            transform_cache_hits: 0,
            transform_cache_misses: 0,
            cartesian_pair_visits: 0,
            maximum_polynomial_terms: 0,
        };

        validate_prepared_report(&prepared, &request, &report).unwrap();
        for corrupt in [
            |report: &mut OperationalCheckerReport| {
                report.decoder_kind = "boolean_interval".to_owned();
            },
            |report: &mut OperationalCheckerReport| {
                report.plaintext_modulus = "2".to_owned();
            },
            |report: &mut OperationalCheckerReport| {
                report.ciphertext_modulus = "256".to_owned();
            },
        ] {
            let mut corrupted = report.clone();
            corrupt(&mut corrupted);
            assert!(matches!(
                validate_prepared_report(&prepared, &request, &corrupted),
                Err(OperationalRunnerError::TargetEcho)
            ));
        }

        let mut corrupted = report.clone();
        corrupted.target_id = "different-target".to_owned();
        assert!(matches!(
            validate_prepared_report(&prepared, &request, &corrupted),
            Err(OperationalRunnerError::Target)
        ));

        let mut corrupted = report.clone();
        corrupted.request_digest = "different-request".to_owned();
        assert!(matches!(
            validate_prepared_report(&prepared, &request, &corrupted),
            Err(OperationalRunnerError::RequestDigest)
        ));

        let mut corrupted = report.clone();
        corrupted.protocol_source_hash = "different-protocol".to_owned();
        assert!(matches!(
            validate_prepared_report(&prepared, &request, &corrupted),
            Err(OperationalRunnerError::Freshness)
        ));
    }

    #[test]
    fn emitted_protocol_retains_closed_decoder_echo_metadata() {
        let protocol = toy_example::protocol();
        let emitted = emit_protocol_for(
            "ToyOperationalEchoMetadata",
            &protocol,
            "MxxCorrectness",
            toy_example::PROTOCOL_SOURCE_PATHS,
        )
        .unwrap();
        let target = emitted
            .operational_decoder_targets
            .iter()
            .find(|target| target.target_id == "toy-threshold")
            .expect("Toy threshold target");

        assert_eq!(target.decoder_kind.report_name(), "threshold_decode");
        assert_eq!(
            target.plaintext_modulus.evaluate(&ParamEnv::default()).unwrap(),
            BigInt::from(2)
        );
        assert_eq!(
            target.ciphertext_modulus.evaluate(&ParamEnv::default()).unwrap(),
            BigInt::from(256)
        );
    }

    #[test]
    #[ignore = "invokes the Lean compiler"]
    fn runs_generated_toy_workflow() {
        let protocol = toy_example::protocol();
        let emitted = emit_protocol_for(
            "ToyOperationalRunner",
            &protocol,
            "MxxCorrectness",
            toy_example::PROTOCOL_SOURCE_PATHS,
        )
        .unwrap();
        let request = OperationalCheckRequest {
            environment: vec![(
                "cutoff".to_owned(),
                OperationalParameterValue::Integer(BigInt::from(3)),
            )],
            layouts: Vec::new(),
            target_id: "toy-threshold".to_owned(),
        };
        let lean_workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../lean");
        let prepared = prepare_emitted_operational_checker(&lean_workspace, &emitted).unwrap();
        let modified = fs::metadata(prepared.olean_path()).unwrap().modified().unwrap();
        let cached = prepare_emitted_operational_checker(&lean_workspace, &emitted).unwrap();
        assert_eq!(fs::metadata(cached.olean_path()).unwrap().modified().unwrap(), modified);
        let reports = run_prepared_operational_checks(
            &lean_workspace,
            &prepared,
            &[request.clone(), request],
        )
        .unwrap();
        assert_eq!(reports.len(), 2);
        assert_eq!(reports[0].noise_bound, "3");
        assert!(reports[0].accepted);
        assert!(reports[1].accepted);
        assert_eq!(reports[0].request_digest, reports[1].request_digest);
    }
}
