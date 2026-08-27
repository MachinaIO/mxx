mod dependency;
mod history;
mod semantics;
mod statement;
mod statistics;

pub(super) use statistics::measure_owner_claims;

use super::{GeneratedLeanFile, LeanArtifactConfig};
use crate::operational_noise::{
    certificate_schema::{CertificateDocumentV1, CertificateIndexUse, CertificateSliceGroup},
    facts::{CoefficientBound, NumericContract},
    simulation::{OperationalProofPayload, ProofPayloadEvent, ProofPayloadValue},
};
use num_bigint::BigUint;
use serde::Serialize;

const NAMESPACE: &str = "Mxx.Certificate.OperationalNoise.Generated";
const MODULE_ROOT: &str = "Mxx.Certificate.OperationalNoise.Generated";

pub(super) fn render(
    statement: &CertificateDocumentV1,
    proof: &OperationalProofPayload,
    owner_claim_report_bytes: &[u8],
    identity: &LeanArtifactConfig,
    ordinary_rust_noise_bound: &BigUint,
    recorder_peak_retained_logical_items: u64,
    proof_projection_peak_retained_logical_items: u64,
) -> Result<Vec<GeneratedLeanFile>, String> {
    let semantic_slice = dependency::resolve_reached_semantic_slice(statement, proof)?;
    let dependency_closure = dependency::collect_reached_final_closure(proof, &semantic_slice)?;
    let final_proof_bound =
        validate_final_bound(proof, dependency_closure.final_end_event, ordinary_rust_noise_bound)?;
    let mut files = statement::render(statement)?;
    files.extend(history::render(statement, proof)?);
    files.extend(semantics::render(statement, proof, &semantic_slice)?);
    files.push(GeneratedLeanFile {
        relative_path: "SemanticOwnerStatistics.json".to_owned(),
        bytes: owner_claim_report_bytes.to_vec(),
    });
    files.push(GeneratedLeanFile {
        relative_path: "SemanticDependencyClosure.json".to_owned(),
        bytes: dependency_closure.report_bytes()?,
    });
    let artifact_namespace = &identity.module_root;
    for file in &mut files {
        if file.relative_path.ends_with(".lean") {
            let source = String::from_utf8(std::mem::take(&mut file.bytes))
                .map_err(|error| format!("generated Lean source is not UTF-8: {error}"))?;
            file.bytes = source.replace(NAMESPACE, artifact_namespace).into_bytes();
        }
    }
    let metrics = CertificateMetrics::from_rendered(
        statement,
        proof,
        &dependency_closure,
        ordinary_rust_noise_bound,
        &final_proof_bound,
        recorder_peak_retained_logical_items,
        proof_projection_peak_retained_logical_items,
        &files,
    )?;
    let metrics_bytes = serde_json::to_vec(&metrics)
        .map_err(|error| format!("certificate metrics encoding failed: {error}"))?;
    files.push(GeneratedLeanFile {
        relative_path: "CertificateMetrics.json".to_owned(),
        bytes: metrics_bytes,
    });
    files.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    Ok(files)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CertificateMetrics {
    schema_id: &'static str,
    schema_version: u32,
    statement: StatementMetrics,
    proof: ProofMetrics,
    raw_lut: RawLutMetrics,
    recorder_peak_retained_logical_items: u64,
    proof_projection_peak_retained_logical_items: u64,
    generated_artifact_file_count_excluding_metrics: u64,
    generated_artifact_byte_total_excluding_metrics: u64,
    reached_dependency_event_count: u64,
    specialization_cache_hit_count: u64,
    ordinary_rust_noise_bound: String,
    final_proof_bound: String,
    final_lean_bound: String,
    bounds_equal: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StatementMetrics {
    expression_rows: u64,
    program_rows: u64,
    source_rows: u64,
    event_rows: u64,
    index_use_rows: u64,
    slice_group_rows: u64,
    n: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ProofMetrics {
    event_count: u64,
    logical_item_count: u64,
    canonical_byte_count: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RawLutMetrics {
    index_use_row_count: u64,
    slice_group_row_count: u64,
    canonical_byte_count: u64,
}

#[derive(Serialize)]
struct RawLutPayload<'a> {
    index_uses: &'a [CertificateIndexUse],
    slice_groups: &'a [CertificateSliceGroup],
}

impl CertificateMetrics {
    #[allow(clippy::too_many_arguments)]
    fn from_rendered(
        statement: &CertificateDocumentV1,
        proof: &OperationalProofPayload,
        dependency_closure: &dependency::DependencyClosure,
        ordinary_rust_noise_bound: &BigUint,
        final_bound: &BigUint,
        recorder_peak_retained_logical_items: u64,
        proof_projection_peak_retained_logical_items: u64,
        files: &[GeneratedLeanFile],
    ) -> Result<Self, String> {
        let expression_rows = cardinality(statement.expressions.len(), "expression rows")?;
        let program_rows = cardinality(statement.programs.len(), "program rows")?;
        let source_rows = cardinality(statement.sources.len(), "source rows")?;
        let event_rows = cardinality(statement.events.len(), "event rows")?;
        let index_use_rows = statement.index_uses.iter().try_fold(
            0_u64,
            |total, use_row| -> Result<u64, String> {
                total
                    .checked_add(cardinality(use_row.rows.len(), "index-use rows")?)
                    .ok_or_else(|| "certificate index-use row count overflow".to_owned())
            },
        )?;
        let slice_group_rows = statement.slice_groups.iter().try_fold(
            0_u64,
            |total, group| -> Result<u64, String> {
                total
                    .checked_add(cardinality(group.rows.len(), "SliceGroup rows")?)
                    .ok_or_else(|| "certificate SliceGroup row count overflow".to_owned())
            },
        )?;
        let n = expression_rows
            .checked_add(program_rows)
            .and_then(|value| value.checked_add(source_rows))
            .and_then(|value| value.checked_add(event_rows))
            .ok_or_else(|| "certificate statement N overflow".to_owned())?;
        let proof_bytes = proof
            .encode_canonical()
            .map_err(|error| format!("proof canonical encoding failed: {error:?}"))?;
        let proof_logical_items = proof
            .logical_items()
            .map_err(|error| format!("proof logical-item count failed: {error:?}"))?;
        let raw_lut_bytes = serde_json::to_vec(&RawLutPayload {
            index_uses: &statement.index_uses,
            slice_groups: &statement.slice_groups,
        })
        .map_err(|error| format!("raw LUT metrics encoding failed: {error}"))?;
        let (artifact_file_count, artifact_byte_total) = artifact_totals(files)?;
        let cache_hits = dependency_closure
            .event_counts
            .get(&dependency::ClosureEventKind::SpecializationCacheHit)
            .copied()
            .unwrap_or(0);
        Ok(Self {
            schema_id: "mxx.operational-noise.certificate-metrics",
            schema_version: 1,
            statement: StatementMetrics {
                expression_rows,
                program_rows,
                source_rows,
                event_rows,
                index_use_rows,
                slice_group_rows,
                n,
            },
            proof: ProofMetrics {
                event_count: cardinality(proof.events.len(), "proof events")?,
                logical_item_count: proof_logical_items,
                canonical_byte_count: cardinality(proof_bytes.len(), "proof canonical bytes")?,
            },
            raw_lut: RawLutMetrics {
                index_use_row_count: index_use_rows,
                slice_group_row_count: slice_group_rows,
                canonical_byte_count: cardinality(raw_lut_bytes.len(), "raw LUT canonical bytes")?,
            },
            recorder_peak_retained_logical_items,
            proof_projection_peak_retained_logical_items,
            generated_artifact_file_count_excluding_metrics: artifact_file_count,
            generated_artifact_byte_total_excluding_metrics: artifact_byte_total,
            reached_dependency_event_count: cardinality(
                dependency_closure.event_ids.len(),
                "reached dependency events",
            )?,
            specialization_cache_hit_count: cache_hits,
            ordinary_rust_noise_bound: ordinary_rust_noise_bound.to_string(),
            final_proof_bound: final_bound.to_string(),
            final_lean_bound: final_bound.to_string(),
            bounds_equal: ordinary_rust_noise_bound == final_bound,
        })
    }
}

fn cardinality(value: usize, label: &str) -> Result<u64, String> {
    u64::try_from(value).map_err(|_| format!("certificate {label} overflow"))
}

fn artifact_totals(files: &[GeneratedLeanFile]) -> Result<(u64, u64), String> {
    let file_count = cardinality(files.len(), "generated artifact files")?;
    let byte_total = files.iter().try_fold(0_u64, |total, file| {
        total
            .checked_add(cardinality(file.bytes.len(), "generated artifact bytes")?)
            .ok_or_else(|| "generated artifact byte total overflow".to_owned())
    })?;
    Ok((file_count, byte_total))
}

fn final_bound_from_typed_events(
    proof: &OperationalProofPayload,
    final_end_event: u64,
) -> Result<BigUint, String> {
    let end_index = usize::try_from(final_end_event)
        .map_err(|_| "final InvocationEnd event index overflows usize".to_owned())?;
    let Some(ProofPayloadEvent::InvocationEnd { root, result: end_result, pre_fold_event }) =
        proof.events.get(end_index)
    else {
        return Err("final dependency event is not an InvocationEnd".to_owned());
    };
    let prefold_index = usize::try_from(*pre_fold_event)
        .map_err(|_| "final PreFold event index overflows usize".to_owned())?;
    let Some(ProofPayloadEvent::PreFoldPolynomial(prefold)) = proof.events.get(prefold_index)
    else {
        return Err("final InvocationEnd does not reference a PreFoldPolynomial".to_owned());
    };
    let result_index = usize::try_from(prefold.result_event)
        .map_err(|_| "final Result event index overflows usize".to_owned())?;
    let Some(ProofPayloadEvent::Result { owner: result_owner, value: result_value }) =
        proof.events.get(result_index)
    else {
        return Err("final PreFoldPolynomial does not reference a Result".to_owned());
    };
    let (
        ProofPayloadValue::Exact { terms: result_terms, summary: result_summary, .. },
        ProofPayloadValue::Exact { terms: end_terms, summary: end_summary, .. },
    ) = (result_value, end_result)
    else {
        return Err("final proof chain must use exact finite summaries".to_owned());
    };
    if root != result_owner ||
        result_terms != &prefold.terms ||
        end_terms != result_terms ||
        result_summary != &prefold.summary ||
        end_summary != result_summary
    {
        return Err("final InvocationEnd, PreFold, and Result references disagree".to_owned());
    }
    match result_summary.coefficient_bound() {
        NumericContract::Known(CoefficientBound::ExactZero) => Ok(BigUint::ZERO),
        NumericContract::Known(CoefficientBound::Finite(bound)) => {
            Ok(bound.maximum_absolute_coefficient.clone())
        }
        NumericContract::Missing | NumericContract::Known(CoefficientBound::Large) => {
            Err("final proof summary is missing or large".to_owned())
        }
    }
}

fn validate_final_bound(
    proof: &OperationalProofPayload,
    final_end_event: u64,
    ordinary_rust_noise_bound: &BigUint,
) -> Result<BigUint, String> {
    let final_bound = final_bound_from_typed_events(proof, final_end_event)?;
    if &final_bound != ordinary_rust_noise_bound {
        return Err(format!(
            "final proof bound {} does not match ordinary Rust noise bound {}",
            final_bound, ordinary_rust_noise_bound
        ));
    }
    Ok(final_bound)
}

fn generated_file(relative_path: impl Into<String>, source: String) -> GeneratedLeanFile {
    GeneratedLeanFile { relative_path: relative_path.into(), bytes: source.into_bytes() }
}

fn quoted(value: &str) -> Result<String, String> {
    serde_json::to_string(value)
        .map_err(|error| format!("certificate Lean string encoding failed: {error}"))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::{
        facts::BoundExpression,
        normal_form::BoundedSummary,
        simulation::{
            ProofPayloadOwner, ProofPayloadPreFoldPolynomial, ProofPayloadScope, ProofPayloadValue,
        },
    };

    fn final_proof(summary: BoundedSummary) -> OperationalProofPayload {
        let owner = ProofPayloadOwner {
            scope: ProofPayloadScope::Closed { root_expression_row: 0 },
            expression_row: 1,
        };
        let value = ProofPayloadValue::Exact {
            terms: Vec::new(),
            coefficient_bound: summary.coefficient_bound(),
            coefficient_producer: 0,
            summary: summary.clone(),
            summary_producer: None,
        };
        OperationalProofPayload {
            events: vec![
                ProofPayloadEvent::Result { owner, value: value.clone() },
                ProofPayloadEvent::PreFoldPolynomial(ProofPayloadPreFoldPolynomial {
                    result_event: 0,
                    terms: Vec::new(),
                    summary: summary.clone(),
                    summary_evidence: None,
                }),
                ProofPayloadEvent::InvocationEnd { root: owner, result: value, pre_fold_event: 1 },
            ],
        }
    }

    #[test]
    fn final_bound_accepts_typed_finite_chain() {
        let proof = final_proof(BoundedSummary::finite(BoundExpression::new(7_u8.into())));
        let bound = validate_final_bound(&proof, 2, &7_u8.into()).expect("matching bound");
        assert_eq!(bound, 7_u8.into());
    }

    #[test]
    fn final_bound_rejects_mismatch() {
        let proof = final_proof(BoundedSummary::finite(BoundExpression::new(7_u8.into())));
        let error = validate_final_bound(&proof, 2, &8_u8.into()).expect_err("mismatching bound");
        assert!(error.contains("does not match ordinary Rust noise bound"));
    }

    #[test]
    fn artifact_totals_are_computed_before_metrics_file() {
        let files = vec![
            GeneratedLeanFile { relative_path: "a".to_owned(), bytes: vec![1, 2] },
            GeneratedLeanFile { relative_path: "b".to_owned(), bytes: vec![3] },
        ];
        assert_eq!(artifact_totals(&files).expect("artifact totals"), (2, 3));
    }
}
