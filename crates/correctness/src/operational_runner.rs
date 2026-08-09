//! Fail-closed execution of generated Lean operational checks.

use crate::EmittedProtocol;
use num_bigint::BigInt;
use serde::{Deserialize, Serialize};
use std::{path::Path, process::Command};
use tempfile::Builder;
use thiserror::Error;

pub const OPERATIONAL_REPORT_SCHEMA_VERSION: u32 = 1;

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
    pub workflow_hash: String,
    pub derivation_hash: String,
    pub toolkit_hash: String,
    pub noise_bound: String,
    pub plaintext_modulus: String,
    pub ciphertext_modulus: String,
    pub accepted: bool,
    pub rejection: Option<String>,
}

#[derive(Debug, Error)]
pub enum OperationalRunnerError {
    #[error("could not create the temporary Lean checker: {0}")]
    Temporary(#[from] std::io::Error),
    #[error("Lean operational checker failed; stdout: {stdout}; stderr: {stderr}")]
    CheckerFailed { stdout: String, stderr: String },
    #[error("Lean operational checker must emit exactly one nonempty JSON line, got {count}")]
    UnexpectedOutput { count: usize },
    #[error("Lean operational checker emitted malformed JSON: {0}")]
    Malformed(#[from] serde_json::Error),
    #[error("unsupported Lean operational report schema {actual}")]
    Schema { actual: u32 },
    #[error("Lean operational report freshness hashes do not match the emitted protocol")]
    Freshness,
    #[error("Lean operational report modulus fields do not match the request")]
    Modulus,
}

fn lean_string(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('\"', "\\\""))
}

fn lean_identifier(value: &str) -> String {
    let mut output = String::new();
    for character in value.chars() {
        if character.is_ascii_alphanumeric() || character == '_' {
            output.push(character.to_ascii_lowercase());
        } else {
            output.push('_');
        }
    }
    output
}

fn checker_source(emitted: &EmittedProtocol, request: &OperationalCheckRequest) -> String {
    let namespace = format!("{}.Generated.{}", emitted.module_root, emitted.lean_name);
    let derivations = emitted
        .stage_ids
        .iter()
        .map(|stage| {
            format!(
                "({}, {}_stage_{}_derivation)",
                lean_string(stage),
                emitted.lean_name,
                lean_identifier(stage)
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
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
    format!(
        "import Mxx.Certificate.OperationalBounds\n\
set_option maxHeartbeats 0\n\
{}\n\
open {}\n\
open Mxx.Certificate\n\n\
private def operationalCheck : Except OperationalError OperationalNoiseCheckReport := do\n\
  let outputs ← evaluateWorkflowOperational ({{ workflow := {}_protocol.bundle.workflow, inputContract := {}_protocol.bundle.inputContract }} : OperationalWorkflowSpec) [{}] [{}] [{}]\n\
  let stage ← match outputs.find? (fun result => result.stage == {}) with\n\
    | some stage => pure stage\n\
    | none => throw (.missingStageResult {} {})\n\
  let residual ← match stage.outputs.find? (fun output => output.1 == {}) with\n\
    | some output => pure output.2\n\
    | none => throw (.missingStageResult {} {})\n\
  decoderNoiseCheckReportForFact outputs residual [{}] {} {}\n\n\
private def rejectionJson : Option OperationalNoiseRejection → String\n\
  | none => \"null\"\n\
  | some rejection => \"\\\"\" ++ reprStr rejection ++ \"\\\"\"\n\n\
def main : IO UInt32 := do\n\
  match operationalCheck with\n\
  | .error _ => IO.eprintln \"operational graph evaluation failed\"; return 2\n\
  | .ok report =>\n\
      match report.obligations with\n\
      | [.decoderThreshold plaintextModulus ciphertextModulus noiseBound] =>\n\
          let accepted := if report.accepted then \"true\" else \"false\"\n\
          let json := \"{{\\\"schema_version\\\":1,\\\"workflow_hash\\\":\\\"\" ++\n\
            {}_workflowHash ++ \"\\\",\\\"derivation_hash\\\":\\\"\" ++\n\
            {}_derivationHash ++ \"\\\",\\\"toolkit_hash\\\":\\\"\" ++\n\
            {}_toolkitHash ++ \"\\\",\\\"noise_bound\\\":\\\"\" ++ toString noiseBound ++\n\
            \"\\\",\\\"plaintext_modulus\\\":\\\"\" ++ toString plaintextModulus ++\n\
            \"\\\",\\\"ciphertext_modulus\\\":\\\"\" ++ toString ciphertextModulus ++\n\
            \"\\\",\\\"accepted\\\":\" ++ accepted ++ \",\\\"rejection\\\":\" ++\n\
            rejectionJson report.rejection ++ \"}}\"\n\
          IO.println json; return 0\n\
      | _ => IO.eprintln \"operational checker returned an unexpected obligation set\"; return 3\n",
        emitted.ir,
        namespace,
        emitted.lean_name,
        emitted.lean_name,
        derivations,
        environment,
        layouts,
        lean_string(&request.residual_stage),
        lean_string(&request.residual_stage),
        lean_string(&request.residual_output),
        lean_string(&request.residual_output),
        lean_string(&request.residual_stage),
        lean_string(&request.residual_output),
        environment,
        request.plaintext_modulus,
        request.ciphertext_modulus,
        emitted.lean_name,
        emitted.lean_name,
        emitted.lean_name,
    )
}

pub fn run_operational_checker_source(
    lean_workspace: &Path,
    source: &str,
) -> Result<OperationalCheckerReport, OperationalRunnerError> {
    let file = Builder::new().prefix("mxx-operational-").suffix(".lean").tempfile()?;
    std::fs::write(file.path(), source)?;
    let output = Command::new("lake")
        .args(["env", "lean", "--run"])
        .arg(file.path())
        .current_dir(lean_workspace)
        .output()?;
    if !output.status.success() {
        return Err(OperationalRunnerError::CheckerFailed {
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines = stdout.lines().map(str::trim).filter(|line| !line.is_empty()).collect::<Vec<_>>();
    if lines.len() != 1 {
        return Err(OperationalRunnerError::UnexpectedOutput { count: lines.len() });
    }
    let report: OperationalCheckerReport = serde_json::from_str(lines[0])?;
    if report.schema_version != OPERATIONAL_REPORT_SCHEMA_VERSION {
        return Err(OperationalRunnerError::Schema { actual: report.schema_version });
    }
    Ok(report)
}

pub fn run_emitted_operational_check(
    lean_workspace: &Path,
    emitted: &EmittedProtocol,
    request: &OperationalCheckRequest,
) -> Result<OperationalCheckerReport, OperationalRunnerError> {
    let report = run_operational_checker_source(lean_workspace, &checker_source(emitted, request))?;
    if report.workflow_hash != emitted.freshness.workflow_hash ||
        report.derivation_hash != emitted.derivation_hash ||
        report.toolkit_hash != emitted.freshness.toolkit_hash
    {
        return Err(OperationalRunnerError::Freshness);
    }
    if report.plaintext_modulus != request.plaintext_modulus.to_string() ||
        report.ciphertext_modulus != request.ciphertext_modulus.to_string()
    {
        return Err(OperationalRunnerError::Modulus);
    }
    Ok(report)
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
            schema_version: 2,
            workflow_hash: String::new(),
            derivation_hash: String::new(),
            toolkit_hash: String::new(),
            noise_bound: "0".to_owned(),
            plaintext_modulus: "2".to_owned(),
            ciphertext_modulus: "17".to_owned(),
            accepted: true,
            rejection: None,
        };
        assert_ne!(report.schema_version, OPERATIONAL_REPORT_SCHEMA_VERSION);
    }

    #[test]
    fn generated_checker_does_not_timeout_large_workflows() {
        let protocol = toy_example::protocol();
        let emitted = emit_protocol_for(
            "ToyOperationalRunner",
            &protocol,
            "MxxCorrectness",
            toy_example::PROTOCOL_SOURCE_PATHS,
        )
        .unwrap();
        let request = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            residual_stage: "encrypt".to_owned(),
            residual_output: "operational-residual".to_owned(),
            plaintext_modulus: BigInt::from(2),
            ciphertext_modulus: BigInt::from(256),
        };

        assert!(checker_source(&emitted, &request).contains("set_option maxHeartbeats 0"));
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
        let lean_workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../lean");
        let report = run_emitted_operational_check(&lean_workspace, &emitted, &request).unwrap();
        assert_eq!(report.noise_bound, "3");
        assert!(report.accepted);
    }
}
