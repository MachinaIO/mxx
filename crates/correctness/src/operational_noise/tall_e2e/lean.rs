mod closure;
mod proof;
mod semantic;
mod statement;
mod statistics;

pub(super) use statistics::measure_owner_claims;

use super::{TallSecurity0GeneratedFile, TallSecurity0ProfileIdentity};
use crate::operational_noise::{
    certificate_schema::CertificateDocumentV1, simulation::OperationalProofPayload,
};

const NAMESPACE: &str = "Mxx.Certificate.OperationalNoise.TallReachedGenerated";
const MODULE_ROOT: &str = "Mxx.Certificate.OperationalNoise.TallReachedGenerated";

pub(super) fn render(
    statement: &CertificateDocumentV1,
    proof: &OperationalProofPayload,
    owner_claim_report_bytes: &[u8],
    identity: &TallSecurity0ProfileIdentity,
) -> Result<Vec<TallSecurity0GeneratedFile>, String> {
    let semantic_slice = closure::resolve_reached_semantic_slice(statement, proof)?;
    let dependency_closure = closure::collect_reached_final_closure(proof, &semantic_slice)?;
    let mut files = statement::render(statement)?;
    files.extend(proof::render(statement, proof)?);
    files.extend(semantic::render(statement, proof, &semantic_slice)?);
    files.push(TallSecurity0GeneratedFile {
        relative_path: "SemanticOwnerStatistics.json".to_owned(),
        bytes: owner_claim_report_bytes.to_vec(),
    });
    files.push(TallSecurity0GeneratedFile {
        relative_path: "SemanticDependencyClosure.json".to_owned(),
        bytes: dependency_closure.report_bytes()?,
    });
    let profile_namespace =
        format!("Mxx.Certificate.OperationalNoise.Tall{}Generated", identity.profile);
    for file in &mut files {
        if file.relative_path.ends_with(".lean") {
            let source = String::from_utf8(std::mem::take(&mut file.bytes))
                .map_err(|error| format!("generated Lean source is not UTF-8: {error}"))?;
            file.bytes = source.replace(NAMESPACE, &profile_namespace).into_bytes();
        }
    }
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
