//! Fail-closed execution of generated Lean operational checks.

use crate::{EmittedProtocol, GENERATOR_VERSION};
use num_bigint::BigInt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
};
use tempfile::Builder;
use thiserror::Error;

pub const OPERATIONAL_REPORT_SCHEMA_VERSION: u32 = 4;
const OPERATIONAL_PREPARED_CACHE_FORMAT_VERSION: u32 = 4;

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
    pub residual_stage: String,
    pub residual_output: String,
    pub plaintext_modulus: BigInt,
    pub ciphertext_modulus: BigInt,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OperationalCheckerReport {
    pub schema_version: u32,
    pub protocol_source_hash: String,
    pub workflow_hash: String,
    pub derivation_hash: String,
    pub toolkit_hash: String,
    pub request_digest: String,
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedOperationalChecker {
    module_name: String,
    namespace: String,
    prepared_name: String,
    protocol_source_hash: String,
    workflow_hash: String,
    derivation_hash: String,
    toolkit_hash: String,
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
    CheckerFailed { stdout: String, stderr: String },
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
    #[error("Lean operational report modulus fields do not match the request")]
    Modulus,
    #[error("Lean operational report request digest does not match the request")]
    RequestDigest,
    #[error(
        "could not compile prepared Lean operational module; stdout: {stdout}; stderr: {stderr}"
    )]
    PreparationFailed { stdout: String, stderr: String },
}

fn lean_string(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('\"', "\\\""))
}

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
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn prepared_module_source(emitted: &EmittedProtocol, prepared_name: &str) -> String {
    let namespace = format!("{}.Generated.{}", emitted.module_root, emitted.lean_name);
    format!(
        "import Mxx.Certificate.OperationalBounds\n{}\nnamespace {}\n\n\
def {} : Except (Sum Mxx.Ir.DecodeError Mxx.Certificate.OperationalError) \
Mxx.Certificate.PreparedOperationalWorkflow := do\n  \
let decoded ← {}_decoded |>.mapError Sum.inl\n  \
let protocol := decoded.1\n  \
let derivations := decoded.2\n  \
Mxx.Certificate.prepareWorkflowOperational\n    \
({{ workflow := protocol.bundle.workflow, inputContract := protocol.bundle.inputContract }} : \
Mxx.Certificate.OperationalWorkflowSpec)\n    \
derivations |>.mapError Sum.inr\n\nend {}\n",
        emitted.ir, namespace, prepared_name, emitted.lean_name, namespace,
    )
}

pub fn prepare_emitted_operational_checker(
    lean_workspace: &Path,
    emitted: &EmittedProtocol,
) -> Result<PreparedOperationalChecker, OperationalRunnerError> {
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
    let source_path = source_dir.join(format!("{component}.lean"));
    let olean_dir = lean_workspace.join(".lake/build/lib/lean/MxxOperationalCache");
    let olean_path = olean_dir.join(format!("{component}.olean"));
    let namespace = format!("{}.Generated.{}", emitted.module_root, emitted.lean_name);
    let prepared_name = format!("{}_preparedOperational", emitted.lean_name);
    fs::create_dir_all(&source_dir)?;
    fs::create_dir_all(&olean_dir)?;
    if !olean_path.is_file() {
        fs::write(&source_path, prepared_module_source(emitted, &prepared_name))?;
        let temporary_olean =
            olean_dir.join(format!("{component}.{}.tmp.olean", std::process::id()));
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
    }
    Ok(PreparedOperationalChecker {
        module_name,
        namespace,
        prepared_name,
        protocol_source_hash: emitted.freshness.protocol_source_hash.clone(),
        workflow_hash: emitted.freshness.workflow_hash.clone(),
        derivation_hash: emitted.derivation_hash.clone(),
        toolkit_hash: emitted.freshness.toolkit_hash.clone(),
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
        "residual_stage": request.residual_stage,
        "residual_output": request.residual_output,
        "plaintext_modulus": request.plaintext_modulus.to_string(),
        "ciphertext_modulus": request.ciphertext_modulus.to_string(),
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
                    requests[*representative].residual_stage == request.residual_stage &&
                    requests[*representative].residual_output == request.residual_output
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
                "private def operationalBound_{bound_group} (outputs : List OperationalStageResult) : \
                 Except OperationalError (Int × OperationalAnalysisDiagnostics) := do\n\
                   let stage <- match outputs.find? (fun result => result.stage == {}) with\n\
                     | some stage => pure stage\n\
                     | none => throw (.missingStageResult {} {})\n\
                   let residual <- match stage.outputs.find? (fun output => output.1 == {}) with\n\
                     | some output => pure output.2\n\
                     | none => throw (.missingStageResult {} {})\n\
                   operationalNoiseBoundForFact stage.facts.arena residual [{environment}]",
                lean_string(&request.residual_stage),
                lean_string(&request.residual_stage),
                lean_string(&request.residual_output),
                lean_string(&request.residual_output),
                lean_string(&request.residual_stage),
                lean_string(&request.residual_output),
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let output_executions = group_representatives
        .iter()
        .enumerate()
        .map(|(group, _)| {
            format!(
                "IO.eprintln \"phase=evaluate_scope group={group} state=start\"\n\
                 let evaluationStarted_{group} ← IO.monoNanosNow\n\
                 let outputs_{group} ← match operationalOutputs_{group} prepared with\n\
                 | .error error => IO.eprintln s!\"operational graph evaluation failed for request \
                   group {group}: {{repr error}}\"; return 2\n\
                 | .ok outputs => pure outputs\n\
                 let evaluationFinished_{group} ← IO.monoNanosNow\n\
                 let evaluationTimeNs_{group} := evaluationFinished_{group} - evaluationStarted_{group}\n\
                 IO.eprintln s!\"phase=evaluate_scope group={group} state=finish \
                   elapsed_ns={{evaluationTimeNs_{group}}}\""
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let bound_executions = bound_group_representatives
        .iter()
        .enumerate()
        .map(|(bound_group, representative)| {
            let output_group = request_groups[*representative];
            format!(
                "IO.eprintln \"phase=evaluate_decoder_bounds group={bound_group} state=start\"\n\
                 let boundEvaluationStarted_{bound_group} ← IO.monoNanosNow\n\
                 let boundResult_{bound_group} ← match \
                   operationalBound_{bound_group} outputs_{output_group} with\n\
                   | .error error => IO.eprintln s!\"operational bound evaluation failed for \
                     bound group {bound_group}: {{repr error}}\"; return 2\n\
                   | .ok result => pure result\n\
                 let boundEvaluationFinished_{bound_group} ← IO.monoNanosNow\n\
                 let boundEvaluationTimeNs_{bound_group} := \
                   boundEvaluationFinished_{bound_group} - boundEvaluationStarted_{bound_group}\n\
                 IO.eprintln s!\"phase=evaluate_decoder_bounds group={bound_group} state=finish \
                   elapsed_ns={{boundEvaluationTimeNs_{bound_group}}}\""
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
                "let (noiseBound_{index}, diagnostics_{index}) := boundResult_{bound_group}\n\
                 let report := decoderNoiseCheckReportFromBound outputs_{group} noiseBound_{index} \
                   diagnostics_{index} {} {}\n\
                 IO.eprintln s!\"operational analysis diagnostics request {index}: \
                   expression_nodes={{report.diagnostics.expressionNodeCount}} \
                   memo_hits={{report.diagnostics.memoHits}} \
                   memo_misses={{report.diagnostics.memoMisses}} \
                   envelope_logical={{report.diagnostics.envelopeLogicalBranchCount}} \
                   envelope_stored={{report.diagnostics.envelopeStoredBranchCount}} \
                   relation_rewrites={{report.diagnostics.relationRewriteCount}} \
                   transform_cache_hits={{report.diagnostics.transformCacheHits}} \
                   transform_cache_misses={{report.diagnostics.transformCacheMisses}} \
                   cartesian_pairs={{report.diagnostics.cartesianPairVisits}} \
                   maximum_polynomial_terms={{report.diagnostics.maximumPolynomialTerms}}\"\n\
                 match report.obligations with\n\
                     | [.decoderThreshold plaintextModulus ciphertextModulus noiseBound] =>\n\
                         let accepted := if report.accepted then \"true\" else \"false\"\n\
                         let json := \"{{\\\"schema_version\\\":4,\\\"protocol_source_hash\\\":\\\"{}\\\",\\\"workflow_hash\\\":\\\"{}\\\",\\\"derivation_hash\\\":\\\"{}\\\",\\\"toolkit_hash\\\":\\\"{}\\\",\\\"request_digest\\\":\\\"{}\\\",\\\"noise_bound\\\":\\\"\" ++ toString noiseBound ++ \"\\\",\\\"plaintext_modulus\\\":\\\"\" ++ toString plaintextModulus ++ \"\\\",\\\"ciphertext_modulus\\\":\\\"\" ++ toString ciphertextModulus ++ \"\\\",\\\"accepted\\\":\" ++ accepted ++ \",\\\"rejection\\\":\" ++ rejectionJson report.rejection ++ \",\\\"decode_time_ns\\\":\" ++ toString decodeTimeNs ++ \",\\\"evaluation_time_ns\\\":\" ++ toString evaluationTimeNs_{group} ++ \",\\\"bound_evaluation_time_ns\\\":\" ++ toString boundEvaluationTimeNs_{bound_group} ++ \",\\\"expression_node_count\\\":\" ++ toString report.diagnostics.expressionNodeCount ++ \",\\\"memo_evaluations\\\":\" ++ toString report.diagnostics.memoEvaluations ++ \",\\\"memo_hits\\\":\" ++ toString report.diagnostics.memoHits ++ \",\\\"memo_misses\\\":\" ++ toString report.diagnostics.memoMisses ++ \",\\\"peak_memo_entries\\\":\" ++ toString report.diagnostics.peakMemoEntries ++ \",\\\"envelope_logical_branch_count\\\":\" ++ toString report.diagnostics.envelopeLogicalBranchCount ++ \",\\\"envelope_stored_branch_count\\\":\" ++ toString report.diagnostics.envelopeStoredBranchCount ++ \",\\\"relation_rewrite_count\\\":\" ++ toString report.diagnostics.relationRewriteCount ++ \",\\\"transform_cache_hits\\\":\" ++ toString report.diagnostics.transformCacheHits ++ \",\\\"transform_cache_misses\\\":\" ++ toString report.diagnostics.transformCacheMisses ++ \",\\\"cartesian_pair_visits\\\":\" ++ toString report.diagnostics.cartesianPairVisits ++ \",\\\"maximum_polynomial_terms\\\":\" ++ toString report.diagnostics.maximumPolynomialTerms ++ \"}}\"\n\
                         IO.println json\n\
                     | _ => IO.eprintln \"operational checker returned an unexpected obligation \
                       set\"; return 3",
                request.plaintext_modulus,
                request.ciphertext_modulus,
                prepared.protocol_source_hash,
                prepared.workflow_hash,
                prepared.derivation_hash,
                prepared.toolkit_hash,
                digest,
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "import {}\nopen {}\nopen Mxx.Certificate\n\n{}\n\n{}\n\n\
         private def rejectionJson : Option OperationalNoiseRejection -> String\n\
           | none => \"null\"\n\
           | some rejection => \"\\\"\" ++ reprStr rejection ++ \"\\\"\"\n\n\
         def main : IO UInt32 := do\n\
           let decodeStarted ← IO.monoNanosNow\n\
           match {} with\n\
           | .error _ => IO.eprintln \"operational graph preparation failed\"; return 2\n\
           | .ok prepared =>\n\
             let decodeFinished ← IO.monoNanosNow\n\
             let decodeTimeNs := decodeFinished - decodeStarted\n\
{}\n{}\nreturn 0\n",
        prepared.module_name,
        prepared.namespace,
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
    let file = Builder::new().prefix("mxx-operational-").suffix(".lean").tempfile()?;
    std::fs::write(file.path(), source)?;
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
    if !status.success() {
        return Err(OperationalRunnerError::CheckerFailed {
            stdout: String::from_utf8_lossy(&stdout).into_owned(),
            stderr: String::from_utf8_lossy(&stderr).into_owned(),
        });
    }
    let stdout = String::from_utf8_lossy(&stdout);
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
    }
    Ok(reports)
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
    if report.plaintext_modulus != request.plaintext_modulus.to_string() ||
        report.ciphertext_modulus != request.ciphertext_modulus.to_string()
    {
        return Err(OperationalRunnerError::Modulus);
    }
    Ok(())
}

pub fn run_prepared_operational_checks(
    lean_workspace: &Path,
    prepared: &PreparedOperationalChecker,
    requests: &[OperationalCheckRequest],
) -> Result<Vec<OperationalCheckerReport>, OperationalRunnerError> {
    let reports = run_operational_checker_reports(
        lean_workspace,
        &prepared_checker_source(prepared, requests),
        requests.len(),
    )?;
    for (request, report) in requests.iter().zip(&reports) {
        validate_prepared_report(prepared, request, report)?;
    }
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
            schema_version: 5,
            protocol_source_hash: String::new(),
            workflow_hash: String::new(),
            derivation_hash: String::new(),
            toolkit_hash: String::new(),
            request_digest: String::new(),
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
        assert_ne!(report.schema_version, OPERATIONAL_REPORT_SCHEMA_VERSION);
    }

    #[test]
    fn operational_runner_disables_lean_heartbeat_timeout() {
        assert!(operational_lean_arguments().contains(&"-DmaxHeartbeats=0"));
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
            olean_path: PathBuf::from("unused.olean"),
        };
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            residual_stage: "evaluate".to_owned(),
            residual_output: "residual".to_owned(),
            plaintext_modulus: BigInt::from(2),
            ciphertext_modulus: BigInt::from(17),
        };

        let mut rejecting_request = request.clone();
        rejecting_request.ciphertext_modulus = BigInt::from(8);
        let source = prepared_checker_source(&prepared, &[request, rejecting_request]);

        assert_eq!(source.matches("operationalBound_0 outputs_0").count(), 1);
        assert!(!source.contains("operationalBound_1"));
        assert_eq!(source.matches("boundResult_0").count(), 3);
        assert!(source.contains("operationalNoiseBoundForFact stage.facts.arena residual"));
        assert_eq!(source.matches("decoderNoiseCheckReportFromBound").count(), 2);
        assert!(source.contains("boundEvaluationTimeNs_0"));
        assert!(!source.contains("boundEvaluationTimeNs_1"));
        assert!(source.contains("report.diagnostics.expressionNodeCount"));
        assert!(source.contains("report.diagnostics.relationRewriteCount"));
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
            residual_stage: "encrypt".to_owned(),
            residual_output: "operational-residual".to_owned(),
            plaintext_modulus: BigInt::from(2),
            ciphertext_modulus: BigInt::from(256),
        };
        let mut rejecting_request = request.clone();
        rejecting_request.ciphertext_modulus = BigInt::from(8);
        let lean_workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../lean");
        let prepared = prepare_emitted_operational_checker(&lean_workspace, &emitted).unwrap();
        let modified = fs::metadata(prepared.olean_path()).unwrap().modified().unwrap();
        let cached = prepare_emitted_operational_checker(&lean_workspace, &emitted).unwrap();
        assert_eq!(fs::metadata(cached.olean_path()).unwrap().modified().unwrap(), modified);
        let reports = run_prepared_operational_checks(
            &lean_workspace,
            &prepared,
            &[request, rejecting_request],
        )
        .unwrap();
        assert_eq!(reports.len(), 2);
        assert_eq!(reports[0].noise_bound, "3");
        assert!(reports[0].accepted);
        assert!(!reports[1].accepted);
        assert_ne!(reports[0].request_digest, reports[1].request_digest);
    }
}
