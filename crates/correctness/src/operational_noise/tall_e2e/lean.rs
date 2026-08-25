mod proof;
mod semantic;
mod statement;
mod statistics;

pub(super) use statistics::measure_owner_claims;

use super::TallSecurity0GeneratedFile;
use crate::operational_noise::{
    certificate_schema::CertificateDocumentV1, simulation::OperationalProofPayload,
};

const NAMESPACE: &str = "Mxx.Certificate.OperationalNoise.TallSecurity0Generated";
const MODULE_ROOT: &str = "Mxx.Certificate.OperationalNoise.TallSecurity0Generated";

pub(super) fn render(
    statement: &CertificateDocumentV1,
    proof: &OperationalProofPayload,
    owner_claim_report_bytes: &[u8],
) -> Result<Vec<TallSecurity0GeneratedFile>, String> {
    let mut files = statement::render(statement)?;
    files.extend(proof::render(statement, proof)?);
    files.extend(semantic::render(statement, proof)?);
    files.push(TallSecurity0GeneratedFile {
        relative_path: "SemanticOwnerStatistics.json".to_owned(),
        bytes: owner_claim_report_bytes.to_vec(),
    });
    files.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    Ok(files)
}

fn generated_file(relative_path: impl Into<String>, source: String) -> TallSecurity0GeneratedFile {
    TallSecurity0GeneratedFile { relative_path: relative_path.into(), bytes: source.into_bytes() }
}

fn quoted(value: &str) -> Result<String, String> {
    serde_json::to_string(value)
        .map_err(|error| format!("Security0 Lean string encoding failed: {error}"))
}

fn list<T>(values: &[T], render: impl Fn(&T) -> Result<String, String>) -> Result<String, String> {
    values
        .iter()
        .map(render)
        .collect::<Result<Vec<_>, _>>()
        .map(|values| format!("[{}]", values.join(", ")))
}

fn option<T>(
    value: Option<&T>,
    render: impl Fn(&T) -> Result<String, String>,
) -> Result<String, String> {
    match value {
        Some(value) => Ok(format!("some ({})", render(value)?)),
        None => Ok("none".to_owned()),
    }
}

fn bool_text(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}
