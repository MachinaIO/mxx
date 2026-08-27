//! Proof-free reporting at the operational-noise boundary.
//!
//! This module is deliberately a small boundary adapter.  It owns no expression or family
//! identity and never turns a numeric bound into one.  Root normalization is delegated to the
//! [`CheckerJob`]; the report only classifies the owned result and applies the target's exact
//! acceptance inequality.

use super::{
    OperationalAcceptanceReport, OperationalCounterSnapshot, OperationalSimulationDiagnostics,
    OperationalSimulationReport,
    arena::ResolvedValueType,
    facts::{CoefficientBound, NumericContract},
    g0::{FeasibilitySink, NoFeasibility},
    job::{CheckerJob, ExactTermDiagnostic, JobError, ProofAnalysisResult},
    lower::{ProductionRoot, ProductionRoots},
    normal_form::{AnalyzedValue, NormalizationCounters},
};
use num_bigint::{BigInt, BigUint};
use num_traits::{CheckedMul, Zero};
use std::fmt;

/// The concrete `p` and `q` values consumed by the threshold decoder contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReportTarget {
    pub target_id: String,
    pub plaintext_modulus: BigUint,
    pub ciphertext_modulus: BigUint,
    pub boolean_interval: bool,
}

/// A root's role in a proof-free report.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RootRole {
    Residual,
    Decoder,
}

/// No arena or proof-local identifier is retained in this witness.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RootWitness {
    pub role: RootRole,
    pub exact_term_count: u64,
    pub bound: BoundClass,
    pub exact_terms: Box<[ExactTermDiagnostic]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum BoundClass {
    ExactZero,
    Finite(BigUint),
    Large,
    Missing,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ReportCounters {
    pub occurrences: u64,
    pub samples: u64,
    pub normalization: NormalizationCounters,
}

/// The only information the reporting boundary may retain from a normalized root.
///
/// In particular, this intentionally has no `ScopedExprId`, monomial ID, or proof capability.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AnalyzedRoot {
    pub exact_term_count: u64,
    pub bound: NumericContract<CoefficientBound>,
    pub exact_terms: Box<[ExactTermDiagnostic]>,
}

/// The owned result produced after both roots have been checked.  The conversion intentionally
/// drops witnesses, while retaining their aggregate counters in the existing simulation report.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OperationalReport {
    pub target_id: String,
    pub noise_bound: BigUint,
    pub ciphertext_modulus: BigUint,
    pub accepted: bool,
    pub acceptance: OperationalAcceptanceReport,
    pub diagnostics: OperationalSimulationDiagnostics,
    pub counters: ReportCounters,
    pub residual: RootWitness,
    pub decoder: RootWitness,
}

impl OperationalReport {
    pub(crate) fn counter_snapshot(&self) -> OperationalCounterSnapshot {
        OperationalCounterSnapshot {
            occurrences: self.counters.occurrences,
            samples: self.counters.samples,
            normalization_nodes_processed: self.counters.normalization.nodes_processed,
            normalization_nodes_total: self.counters.normalization.nodes_total,
            normalization_exact_term_count: self.counters.normalization.final_exact_term_count,
            normalization_relation_candidates: self.counters.normalization.relation_candidates,
            normalization_relations_applied: self.counters.normalization.relation_applied,
            normalization_relations_remaining: self.counters.normalization.relation_remaining,
            normalization_bounded_fold_count: self.counters.normalization.bounded_fold_count,
            normalization_peak_cached_values: self.counters.normalization.peak_cached_values,
        }
    }

    pub(crate) fn into_simulation_report(self) -> OperationalSimulationReport {
        let counter_snapshot = self.counter_snapshot();
        OperationalSimulationReport {
            target_id: self.target_id,
            noise_bound: self.noise_bound,
            ciphertext_modulus: self.ciphertext_modulus,
            accepted: self.accepted,
            acceptance: self.acceptance,
            diagnostics: self.diagnostics,
            counter_snapshot,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReportError {
    Job(JobError),
    ScalarRoot { role: RootRole, actual: ResolvedValueType },
    TrapdoorRoot { role: RootRole },
    TupleRoot { role: RootRole, actual: ResolvedValueType },
    ExactResidual { witness: RootWitness },
    KnownLargeResidual { witness: RootWitness },
    MissingResidual { witness: RootWitness },
    NonPositiveModulus { target_id: String },
    ThresholdOverflow,
    BooleanIntervalModulusBelowFour { target_id: String, actual: BigUint },
}

impl fmt::Display for ReportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ReportError {}

impl From<JobError> for ReportError {
    fn from(error: JobError) -> Self {
        Self::Job(error)
    }
}

/// Analyze production roots through the job-owned authority.
///
/// Closed roots are normalized by the job's canonical zero-argument program authority. Family
/// roots are analyzed at the job-owned formal argument and relation registry. In particular,
/// this function never fabricates a scope, calls a family body as an ordinary runtime value, or
/// retains a scoped expression or proof capability in the report.
pub(crate) fn analyze_roots(
    job: &mut CheckerJob,
    roots: &ProductionRoots,
    target: &ReportTarget,
) -> Result<OperationalReport, ReportError> {
    let mut sink = NoFeasibility;
    analyze_roots_with_sink(job, roots, target, &mut sink)
}

pub(crate) fn analyze_roots_with_sink<S: FeasibilitySink>(
    job: &mut CheckerJob,
    roots: &ProductionRoots,
    target: &ReportTarget,
    sink: &mut S,
) -> Result<OperationalReport, ReportError> {
    let residual = analyze_root_with_sink(job, RootRole::Residual, &roots.residual, sink)?;
    let decoder = match &roots.decoder {
        ProductionRoot::Closed(root)
            if matches!(
                job.expressions()
                    .value_type(root.expression())
                    .map_err(|error| ReportError::Job(JobError::Arena(error)))?,
                ResolvedValueType::Bool
            ) =>
        {
            RootAnalysis {
                value: AnalyzedRoot {
                    exact_term_count: 0,
                    bound: NumericContract::Known(CoefficientBound::ExactZero),
                    exact_terms: Box::new([]),
                },
                counters: NormalizationCounters::default(),
            }
        }
        _ => {
            let mut decoder_sink = NoFeasibility;
            analyze_root_with_sink(job, RootRole::Decoder, &roots.decoder, &mut decoder_sink)?
        }
    };
    let counters = ReportCounters {
        occurrences: roots.occurrences,
        samples: roots.samples,
        normalization: add_counters(
            residual.counters,
            decoder.counters,
            residual.value.exact_term_count.saturating_add(decoder.value.exact_term_count),
        ),
    };
    report_analyzed_roots(target.clone(), &residual.value, &decoder.value, counters)
}

struct RootAnalysis {
    value: AnalyzedRoot,
    counters: NormalizationCounters,
}

fn analyze_root(
    job: &mut CheckerJob,
    role: RootRole,
    root: &ProductionRoot,
) -> Result<RootAnalysis, ReportError> {
    let mut sink = NoFeasibility;
    analyze_root_with_sink(job, role, root, &mut sink)
}

fn analyze_root_with_sink(
    job: &mut CheckerJob,
    role: RootRole,
    root: &ProductionRoot,
    sink: &mut impl FeasibilitySink,
) -> Result<RootAnalysis, ReportError> {
    classify_root(job, role, root)?;
    Ok(match root {
        ProductionRoot::Closed(root) => {
            let analysis = job.normalize_closed_root_with_sink(*root, sink)?;
            RootAnalysis {
                value: analyzed_root(&analysis.value, analysis.exact_term_diagnostics),
                counters: analysis.counters,
            }
        }
        ProductionRoot::Family(root) => {
            let result = job.analyze_family_root_with_sink(*root, sink)?;
            RootAnalysis { value: analyzed_family_root(&result), counters: result.counters }
        }
    })
}

fn analyzed_family_root(result: &ProofAnalysisResult) -> AnalyzedRoot {
    AnalyzedRoot {
        exact_term_count: result.exact_term_count,
        bound: result.bounded_summary.coefficient_bound(),
        exact_terms: result.exact_term_diagnostics.clone(),
    }
}

fn add_counters(
    left: NormalizationCounters,
    right: NormalizationCounters,
    final_exact_term_count: u64,
) -> NormalizationCounters {
    NormalizationCounters {
        nodes_processed: left.nodes_processed.saturating_add(right.nodes_processed),
        nodes_total: left.nodes_total.saturating_add(right.nodes_total),
        // Exact terms are final-root witnesses, never an aggregate of intermediate nodes.
        final_exact_term_count,
        remaining_use_releases: left
            .remaining_use_releases
            .saturating_add(right.remaining_use_releases),
        relation_candidates: left.relation_candidates.saturating_add(right.relation_candidates),
        relation_applied: left.relation_applied.saturating_add(right.relation_applied),
        relation_remaining: left.relation_remaining.saturating_add(right.relation_remaining),
        bounded_fold_count: left.bounded_fold_count.saturating_add(right.bounded_fold_count),
        peak_cached_values: left.peak_cached_values.max(right.peak_cached_values),
    }
}

/// Build the final report from already-owned normalizer results.  This is the narrow bridge used
/// by the eventual root transport and keeps acceptance independent from arena IDs and proof
/// lifetimes.
pub(crate) fn report_analyzed_roots(
    target: ReportTarget,
    residual: &AnalyzedRoot,
    decoder: &AnalyzedRoot,
    counters: ReportCounters,
) -> Result<OperationalReport, ReportError> {
    let residual_witness = witness(RootRole::Residual, residual);
    let decoder_witness = witness(RootRole::Decoder, decoder);
    reject_unusable(&residual_witness)?;
    reject_unusable(&decoder_witness)?;
    let noise_bound = bound_value(&residual_witness)?;
    let (accepted, acceptance) = if target.boolean_interval {
        boolean_interval(&target, &noise_bound)?
    } else {
        threshold(&target, &noise_bound)?
    };
    let diagnostics = diagnostics(counters, residual_witness.exact_term_count);
    Ok(OperationalReport {
        target_id: target.target_id,
        noise_bound,
        ciphertext_modulus: target.ciphertext_modulus,
        accepted,
        acceptance,
        diagnostics,
        counters,
        residual: residual_witness,
        decoder: decoder_witness,
    })
}

pub(crate) fn analyzed_root(
    value: &AnalyzedValue,
    exact_terms: Box<[ExactTermDiagnostic]>,
) -> AnalyzedRoot {
    AnalyzedRoot {
        exact_term_count: value
            .exact_nf
            .as_ref()
            .map_or(0, |normal_form| normal_form.exact_terms.len() as u64),
        bound: value.coefficient_bound.clone(),
        exact_terms,
    }
}

fn classify_root(
    job: &CheckerJob,
    role: RootRole,
    root: &ProductionRoot,
) -> Result<(), ReportError> {
    let actual = match root {
        ProductionRoot::Closed(root) => job
            .expressions()
            .value_type(root.expression())
            .map_err(|error| ReportError::Job(JobError::Arena(error)))?
            .clone(),
        ProductionRoot::Family(family) => job
            .programs()
            .family_element_type(*family)
            .map_err(|error| ReportError::Job(JobError::Arena(error)))?,
    };
    match actual {
        ResolvedValueType::Matrix(_) => Ok(()),
        ResolvedValueType::Trapdoor => Err(ReportError::TrapdoorRoot { role }),
        ResolvedValueType::Bool | ResolvedValueType::Int | ResolvedValueType::Real => {
            Err(ReportError::ScalarRoot { role, actual })
        }
        actual => Err(ReportError::TupleRoot { role, actual }),
    }
}

fn witness(role: RootRole, value: &AnalyzedRoot) -> RootWitness {
    RootWitness {
        role,
        exact_term_count: value.exact_term_count,
        bound: bound_class(&value.bound),
        exact_terms: value.exact_terms.clone(),
    }
}

fn bound_class(bound: &NumericContract<CoefficientBound>) -> BoundClass {
    match bound {
        NumericContract::Missing => BoundClass::Missing,
        NumericContract::Known(CoefficientBound::ExactZero) => BoundClass::ExactZero,
        NumericContract::Known(CoefficientBound::Finite(value)) => {
            BoundClass::Finite(value.maximum_absolute_coefficient.clone())
        }
        NumericContract::Known(CoefficientBound::Large) => BoundClass::Large,
    }
}

fn reject_unusable(witness: &RootWitness) -> Result<(), ReportError> {
    if witness.exact_term_count != 0 {
        return Err(ReportError::ExactResidual { witness: witness.clone() });
    }
    match witness.bound {
        BoundClass::ExactZero | BoundClass::Finite(_) => Ok(()),
        BoundClass::Large => Err(ReportError::KnownLargeResidual { witness: witness.clone() }),
        BoundClass::Missing => Err(ReportError::MissingResidual { witness: witness.clone() }),
    }
}

fn bound_value(witness: &RootWitness) -> Result<BigUint, ReportError> {
    match &witness.bound {
        BoundClass::ExactZero => Ok(BigUint::from(0_u8)),
        BoundClass::Finite(value) => Ok(value.clone()),
        BoundClass::Large => Err(ReportError::KnownLargeResidual { witness: witness.clone() }),
        BoundClass::Missing => Err(ReportError::MissingResidual { witness: witness.clone() }),
    }
}

fn threshold(
    target: &ReportTarget,
    noise_bound: &BigUint,
) -> Result<(bool, OperationalAcceptanceReport), ReportError> {
    if target.plaintext_modulus.is_zero() || target.ciphertext_modulus.is_zero() {
        return Err(ReportError::NonPositiveModulus { target_id: target.target_id.clone() });
    }
    let threshold_left = BigUint::from(2_u8)
        .checked_mul(&target.plaintext_modulus)
        .and_then(|value| value.checked_mul(noise_bound))
        .ok_or(ReportError::ThresholdOverflow)?;
    let margin =
        BigInt::from(target.ciphertext_modulus.clone()) - BigInt::from(threshold_left.clone());
    Ok((
        threshold_left < target.ciphertext_modulus,
        OperationalAcceptanceReport::Threshold {
            plaintext_modulus: target.plaintext_modulus.clone(),
            threshold_left,
            margin,
        },
    ))
}

fn boolean_interval(
    target: &ReportTarget,
    noise: &BigUint,
) -> Result<(bool, OperationalAcceptanceReport), ReportError> {
    if target.ciphertext_modulus < BigUint::from(4_u8) {
        return Err(ReportError::BooleanIntervalModulusBelowFour {
            target_id: target.target_id.clone(),
            actual: target.ciphertext_modulus.clone(),
        });
    }
    let q = BigInt::from(target.ciphertext_modulus.clone());
    let quarter = mxx_ir_core::expr::IntExpr::RoundDiv(
        Box::new(mxx_ir_core::expr::IntExpr::constant(&q - BigInt::from(2_u8))),
        Box::new(mxx_ir_core::expr::IntExpr::constant(4_u8)),
    )
    .evaluate(&mxx_ir_core::expr::ParamEnv::default())
    .expect("positive constant RoundDiv denominator");
    let half = &q / BigInt::from(2_u8);
    let noise = BigInt::from(noise.clone());
    let false_lower_margin = &quarter - &noise;
    let false_upper_margin = &q - (BigInt::from(3_u8) * &quarter + &noise);
    let true_lower_margin = &half - (&quarter + &noise);
    let true_upper_margin = BigInt::from(3_u8) * &quarter - (&half + &noise);
    let accepted = false_lower_margin > BigInt::from(0_u8) &&
        false_upper_margin > BigInt::from(0_u8) &&
        true_lower_margin >= BigInt::from(0_u8) &&
        true_upper_margin >= BigInt::from(0_u8);
    Ok((
        accepted,
        OperationalAcceptanceReport::BooleanInterval {
            quarter,
            false_lower_margin,
            false_upper_margin,
            true_lower_margin,
            true_upper_margin,
        },
    ))
}

fn diagnostics(
    counters: ReportCounters,
    final_term_count: u64,
) -> OperationalSimulationDiagnostics {
    OperationalSimulationDiagnostics {
        lowered_term_count: counters.occurrences,
        normalization_node_count: counters.normalization.nodes_processed,
        normalization_node_total: counters.normalization.nodes_total,
        normalization_exact_term_count: final_term_count,
        normalization_relation_count: counters.normalization.relation_candidates,
        normalization_relation_applied: counters.normalization.relation_applied,
        normalization_relation_remaining: counters.normalization.relation_remaining,
        normalization_bounded_fold_count: counters.normalization.bounded_fold_count,
        final_term_count,
        ..OperationalSimulationDiagnostics::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::{
        arena::{
            FamilyDomain, MatrixOperation, ResolvedMatrixType, SemanticFamilySourceIdentity,
            TypedConstant, ValueOperator,
        },
        job::CheckerJob,
        lower::ProductionAdapter,
        program::FamilyValueId,
        protocol::ProtocolPlan,
    };
    use num_bigint::BigInt;
    use std::collections::BTreeMap;

    fn matrix_type() -> ResolvedMatrixType {
        ResolvedMatrixType::new(17_u8.into(), 1, 1, 1).expect("valid test matrix")
    }

    fn zero_matrix(job: &mut CheckerJob) -> super::super::arena::ClosedExprId {
        let scalar = job
            .expressions_mut()
            .intern(ValueOperator::Constant(TypedConstant::int(0)), Box::new([]))
            .expect("zero scalar");
        let matrix = job
            .expressions_mut()
            .intern(
                ValueOperator::Matrix(MatrixOperation::LiftConstantPolynomial {
                    output: matrix_type(),
                    coefficient_bits: 4,
                }),
                Box::new([scalar]),
            )
            .expect("zero matrix");
        job.expressions().close(matrix).expect("closed matrix")
    }

    fn target() -> ReportTarget {
        ReportTarget {
            target_id: "report-test".to_owned(),
            plaintext_modulus: 1_u8.into(),
            ciphertext_modulus: 99_u8.into(),
            boolean_interval: false,
        }
    }

    fn cross_stage_artifact_lineage_protocol(recompute_h: bool) -> crate::ProtocolDecl {
        use crate::{ArtifactBinding, ArtifactName, ProtocolStage, StageId, StageInputName};
        use mxx_dsl::{DslContext, HashTag, Int, Ring, SemanticAnchor};
        use mxx_ir_core::{
            IntExpr,
            artifact::{ArtifactConfidentiality, ProductionId, SpecHash},
            node::ConstantMatrix,
        };

        let ring = Ring::new(256, 1);
        let production_id = ProductionId { spec_hash: SpecHash([0; 32]), execution_nonce: [0; 32] };
        let hash = |key| {
            let mut tag = HashTag::new();
            tag.push("lineage/H");
            tag.push(Int::constant(7));
            tag.push(Int::constant(11));
            ring.hash_matrix(key, tag, (1, 2))
        };
        let gadget = || ring.gadget(1, 4, 2);
        let coefficient = || {
            ring.constant(
                (2, 1),
                ConstantMatrix::Polynomial { coefficients: vec![IntExpr::constant(3)] },
            )
        };
        let key = ring.bytes_input("key", 32);
        let h = hash(key.clone());
        let decomposition = (gadget() * coefficient()).decompose(4, 2).as_mat();
        let p = h.clone() * decomposition;
        let preprocess = DslContext::new("lineage-preprocess")
            .int_parameter("cutoff")
            .public_output("H", h)
            .expect("H output")
            .public_output("P", p)
            .expect("P output")
            .build()
            .expect("preprocess graph");

        let online_key = ring.bytes_input("key", 32);
        let imported_h = ring.artifact_input(
            production_id.clone(),
            "H",
            (1, 2),
            ArtifactConfidentiality::Public,
        );
        let imported_p =
            ring.artifact_input(production_id, "P", (1, 1), ArtifactConfidentiality::Public);
        let recomputed_h = hash(online_key);
        let online_h = if recompute_h { recomputed_h } else { imported_h };
        let online_decomposition = (gadget() * coefficient()).decompose(4, 2).as_mat();
        let residual = imported_p - online_h * online_decomposition;
        let decoded = residual
            .clone()
            .threshold_decode_bools(IntExpr::constant(2), 1)
            .into_iter()
            .next()
            .expect("decoded output")
            .semantic_anchor("lineage-decoded")
            .expect("decoded anchor");
        let online = DslContext::new("lineage-online")
            .int_parameter("cutoff")
            .private_output("operational-residual", residual)
            .expect("residual output")
            .bool_output("decoded", decoded)
            .expect("decoded output")
            .build()
            .expect("online graph");
        let decoder_node = online.graph.outputs()["decoded"].value.node;

        let mut protocol = crate::toy_example::protocol();
        protocol.bundle.workflow.stages = vec![
            ProtocolStage {
                id: StageId("encrypt".to_owned()),
                graph: preprocess.graph,
                semantic_anchors: preprocess.anchors,
                derivation_attachments: preprocess.derivation_attachments,
                bindings: Vec::new(),
            },
            ProtocolStage {
                id: StageId("decrypt".to_owned()),
                graph: online.graph,
                semantic_anchors: online.anchors,
                derivation_attachments: online.derivation_attachments,
                bindings: {
                    let mut bindings = vec![ArtifactBinding {
                        consumer_input: StageInputName("P".to_owned()),
                        producer_stage: StageId("encrypt".to_owned()),
                        producer_output: ArtifactName("P".to_owned()),
                    }];
                    if !recompute_h {
                        bindings.push(ArtifactBinding {
                            consumer_input: StageInputName("H".to_owned()),
                            producer_stage: StageId("encrypt".to_owned()),
                            producer_output: ArtifactName("H".to_owned()),
                        });
                    }
                    bindings
                },
            },
        ];
        if let Some(binding) = protocol
            .bundle
            .input_bindings
            .iter_mut()
            .find(|binding| binding.input == crate::ProtocolInputId::from("message"))
        {
            binding.destinations.retain(|destination| {
                matches!(destination, crate::ProtocolInputDestination::Ideal { .. })
            });
        }
        protocol.bundle.endpoints.entries[0].semantic_anchor = "lineage-decoded".to_owned();
        let target = &mut protocol.bundle.operational_decoder_targets[0];
        target.residual_stage = StageId("decrypt".to_owned());
        target.residual_output = "operational-residual".to_owned();
        target.decoder_stage = StageId("decrypt".to_owned());
        target.decoder_node = decoder_node;
        protocol.bundle.input_contract.inputs.push(crate::InputContractEntry {
            id: crate::ProtocolInputId::from("key"),
            name: "key".to_owned(),
            value: crate::InputValueContract::Bytes { length: IntExpr::constant(32) },
        });
        protocol.bundle.input_bindings.push(crate::ProtocolInputBinding {
            input: crate::ProtocolInputId::from("key"),
            destinations: {
                let mut destinations = vec![crate::ProtocolInputDestination::WorkflowStage {
                    stage: StageId("encrypt".to_owned()),
                    input: StageInputName("key".to_owned()),
                }];
                if recompute_h {
                    destinations.push(crate::ProtocolInputDestination::WorkflowStage {
                        stage: StageId("decrypt".to_owned()),
                        input: StageInputName("key".to_owned()),
                    });
                }
                destinations
            },
        });
        let output_names = protocol
            .bundle
            .workflow
            .stages
            .iter()
            .map(|stage| {
                (stage.id.clone(), stage.graph.outputs().keys().cloned().collect::<Vec<_>>())
            })
            .collect::<Vec<_>>();
        let input_names = protocol
            .bundle
            .workflow
            .stages
            .iter()
            .map(|stage| {
                let inputs = stage
                    .graph
                    .root_scope()
                    .nodes()
                    .iter()
                    .filter_map(|node| match node.kind() {
                        mxx_ir_core::node::NodeKind::Input { name, artifact, .. } => {
                            Some((name.clone(), artifact.is_some()))
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                (stage.id.clone(), inputs, stage.bindings.clone())
            })
            .collect::<Vec<_>>();
        match crate::ProtocolDecl::new(protocol) {
            Ok(protocol) => protocol,
            Err(error) => panic!(
                "cross-stage lineage protocol recompute_h={recompute_h}: {error:?}; outputs={output_names:?}; inputs={input_names:?}"
            ),
        }
    }

    fn value(bound: NumericContract<CoefficientBound>, terms: u64) -> AnalyzedRoot {
        AnalyzedRoot { exact_term_count: terms, bound, exact_terms: Box::new([]) }
    }

    #[test]
    fn strict_threshold_rejects_equality() {
        let target = ReportTarget {
            target_id: "threshold".into(),
            plaintext_modulus: 2_u8.into(),
            ciphertext_modulus: 12_u8.into(),
            boolean_interval: false,
        };
        let residual = value(NumericContract::Known(CoefficientBound::finite(3_u8)), 0);
        let decoder = value(NumericContract::Known(CoefficientBound::ExactZero), 0);
        let report = report_analyzed_roots(target, &residual, &decoder, Default::default())
            .expect("finite roots report");
        assert!(!report.accepted);
        assert!(
            matches!(report.acceptance, OperationalAcceptanceReport::Threshold { margin, .. } if margin.is_zero())
        );
    }

    #[test]
    fn boolean_interval_matches_legacy_mod4_and_boundary_table() {
        for (q, noise, expected) in [
            (16_u64, 3_u64, true),
            (16, 4, false),
            (17, 3, true),
            (17, 4, false),
            (18, 3, true),
            (18, 4, false),
            (19, 3, true),
            (19, 4, false),
            (29, 7, false),
        ] {
            let target = ReportTarget {
                target_id: format!("boolean-{q}-{noise}"),
                plaintext_modulus: 2_u8.into(),
                ciphertext_modulus: q.into(),
                boolean_interval: true,
            };
            let residual = value(NumericContract::Known(CoefficientBound::finite(noise)), 0);
            let decoder = value(NumericContract::Known(CoefficientBound::ExactZero), 0);
            let report = report_analyzed_roots(target, &residual, &decoder, Default::default())
                .expect("boolean interval report");
            assert_eq!(report.accepted, expected, "q={q}, noise={noise}");
            assert!(matches!(
                report.acceptance,
                OperationalAcceptanceReport::BooleanInterval { .. }
            ));
        }
    }

    #[test]
    fn boolean_interval_rejects_modulus_below_four() {
        let target = ReportTarget {
            target_id: "boolean-small".to_owned(),
            plaintext_modulus: 2_u8.into(),
            ciphertext_modulus: 3_u8.into(),
            boolean_interval: true,
        };
        let zero = value(NumericContract::Known(CoefficientBound::ExactZero), 0);
        assert!(matches!(
            report_analyzed_roots(target, &zero, &zero, Default::default()),
            Err(ReportError::BooleanIntervalModulusBelowFour { .. })
        ));
    }

    #[test]
    fn large_and_missing_are_distinct_rejections() {
        let target = ReportTarget {
            target_id: "t".into(),
            plaintext_modulus: 1_u8.into(),
            ciphertext_modulus: 99_u8.into(),
            boolean_interval: false,
        };
        let decoder = value(NumericContract::Known(CoefficientBound::ExactZero), 0);
        let large = value(NumericContract::Known(CoefficientBound::Large), 0);
        let missing = value(NumericContract::Missing, 0);
        assert!(matches!(
            report_analyzed_roots(target.clone(), &large, &decoder, Default::default()),
            Err(ReportError::KnownLargeResidual { .. })
        ));
        assert!(matches!(
            report_analyzed_roots(target, &missing, &decoder, Default::default()),
            Err(ReportError::MissingResidual { .. })
        ));
    }

    #[test]
    fn exact_residual_is_rejected_even_with_a_finite_summary() {
        let target = ReportTarget {
            target_id: "t".into(),
            plaintext_modulus: 1_u8.into(),
            ciphertext_modulus: 99_u8.into(),
            boolean_interval: false,
        };
        let residual = value(NumericContract::Known(CoefficientBound::finite(1_u8)), 1);
        let decoder = value(NumericContract::Known(CoefficientBound::ExactZero), 0);
        assert!(matches!(
            report_analyzed_roots(target, &residual, &decoder, Default::default()),
            Err(ReportError::ExactResidual { .. })
        ));
    }

    #[test]
    fn family_cancellation_a_plus_negative_a_is_exact_zero() {
        let result = ProofAnalysisResult {
            bounded_summary: super::super::normal_form::BoundedSummary::zero(),
            exact_term_count: 0,
            counters: NormalizationCounters::default(),
            exact_term_diagnostics: Box::new([]),
        };
        let family = analyzed_family_root(&result);
        let decoder = value(NumericContract::Known(CoefficientBound::ExactZero), 0);
        let report = report_analyzed_roots(target(), &family, &decoder, Default::default())
            .expect("cancellation report");
        assert!(report.accepted);
        assert_eq!(report.noise_bound, BigUint::ZERO);
    }

    #[test]
    fn counters_and_witnesses_are_deterministic_and_proof_free() {
        let target = ReportTarget {
            target_id: "t".into(),
            plaintext_modulus: 1_u8.into(),
            ciphertext_modulus: 99_u8.into(),
            boolean_interval: false,
        };
        let residual = value(NumericContract::Known(CoefficientBound::finite(2_u8)), 0);
        let decoder = value(NumericContract::Known(CoefficientBound::ExactZero), 0);
        let counters = ReportCounters {
            occurrences: 7,
            samples: 3,
            normalization: NormalizationCounters {
                nodes_processed: 11,
                nodes_total: 13,
                final_exact_term_count: 0,
                remaining_use_releases: 5,
                relation_candidates: 2,
                relation_applied: 1,
                relation_remaining: 0,
                bounded_fold_count: 3,
                peak_cached_values: 4,
            },
        };
        let first = report_analyzed_roots(target.clone(), &residual, &decoder, counters).unwrap();
        let second = report_analyzed_roots(target, &residual, &decoder, counters).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.diagnostics.lowered_term_count, 7);
        assert_eq!(first.diagnostics.normalization_node_count, 11);
        assert_eq!(first.diagnostics.normalization_relation_count, 2);
        assert_eq!(first.diagnostics.normalization_relation_applied, 1);
        assert_eq!(first.diagnostics.normalization_bounded_fold_count, 3);
        assert_eq!(first.residual.bound, BoundClass::Finite(2_u8.into()));
        assert_eq!(first.decoder.bound, BoundClass::ExactZero);
    }

    #[test]
    fn diagnostics_keep_final_root_exact_count_nonzero() {
        let report = diagnostics(ReportCounters::default(), 3);
        assert_eq!(report.normalization_exact_term_count, 3);
        assert_eq!(report.final_term_count, 3);
        assert_eq!(report.normalization_relation_remaining, 0);
    }

    #[test]
    fn real_closed_roots_use_job_owned_normalization() {
        let mut job = CheckerJob::new();
        let token = job.begin_candidate().expect("candidate");
        let root = zero_matrix(&mut job);
        job.finalize_facts(token).expect("finalized facts");
        job.freeze_relations(token).expect("frozen relations");
        let roots = ProductionRoots {
            residual: ProductionRoot::Closed(root),
            decoder: ProductionRoot::Closed(root),
            occurrences: 2,
            samples: 0,
        };

        let report = analyze_roots(&mut job, &roots, &target()).expect("closed report");
        assert!(report.accepted);
        assert_eq!(report.noise_bound, BigUint::ZERO);
        assert_eq!(report.counters.occurrences, 2);
        assert_eq!(report.counters.samples, 0);
    }

    #[test]
    fn real_family_roots_use_job_owned_family_analysis() {
        let mut job = CheckerJob::new();
        let token = job.begin_candidate().expect("candidate");
        let family: FamilyValueId = job
            .with_arena_stores(|expressions, programs, _| {
                programs.source_family(
                    expressions,
                    SemanticFamilySourceIdentity {
                        stable_definition: "report-family".to_owned(),
                        invocation: "0".to_owned(),
                        element_type: ResolvedValueType::Matrix(matrix_type()),
                        domain: FamilyDomain::new(0, 8).expect("family domain"),
                        artifact: None,
                    },
                    None,
                )
            })
            .expect("generated family");
        let argument = job
            .expressions_mut()
            .intern_argument(0, ResolvedValueType::Int)
            .expect("family argument");
        assert!(
            job.expressions()
                .free_arguments(argument)
                .expect("family argument")
                .contains(&(0, ResolvedValueType::Int))
        );
        job.finalize_facts(token).expect("finalized facts");
        job.freeze_relations(token).expect("frozen relations");
        let roots = ProductionRoots {
            residual: ProductionRoot::Family(family),
            decoder: ProductionRoot::Family(family),
            occurrences: 0,
            samples: 0,
        };

        let result = analyze_roots(&mut job, &roots, &target());
        assert!(
            matches!(
                result,
                Err(ReportError::ExactResidual {
                    witness: RootWitness { exact_term_count: 1, .. }
                })
            ),
            "family result: {result:?}"
        );
    }

    #[test]
    fn real_production_roots_reach_the_report_bridge() {
        let protocol = crate::toy_example::protocol();
        let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("toy plan");
        let adapter = ProductionAdapter::new(
            &protocol,
            &plan,
            BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
        )
        .expect("production adapter");
        let (mut job, roots) = adapter.lower().expect("production roots");
        let result = analyze_roots(&mut job, &roots, &target());
        assert_eq!(roots.occurrences, plan.counters().occurrences);
        assert!(roots.samples > 0);
        let ProductionRoot::Closed(residual) = roots.residual else {
            panic!("production residual must be a closed matrix root");
        };
        assert!(matches!(
            job.expressions().value_type(residual.expression()),
            Ok(ResolvedValueType::Matrix(_))
        ));
        match result {
            Ok(report) => {
                assert_eq!(report.counters.occurrences, roots.occurrences);
                assert_eq!(report.counters.samples, roots.samples);
                assert_eq!(report.decoder.bound, BoundClass::ExactZero);
            }
            Err(ReportError::ExactResidual { witness }) |
            Err(ReportError::KnownLargeResidual { witness }) |
            Err(ReportError::MissingResidual { witness }) => {
                assert_eq!(witness.role, RootRole::Residual);
            }
            Err(error) => panic!("unexpected infrastructure/report error: {error:?}"),
        }
    }

    #[test]
    fn cross_stage_artifact_lineage_cancels_deterministic_hash_product() {
        for recompute_h in [false, true] {
            let protocol = cross_stage_artifact_lineage_protocol(recompute_h);
            let plan = ProtocolPlan::build(&protocol, "toy-threshold").expect("lineage plan");
            let adapter = ProductionAdapter::new(
                &protocol,
                &plan,
                BTreeMap::from([("cutoff".to_owned(), BigInt::from(8))]),
            )
            .expect("lineage adapter");
            let (mut job, roots) = adapter.lower().expect("lineage lowering");
            let ProductionRoot::Closed(_) = roots.residual else {
                panic!("lineage residual must be closed")
            };
            let analysis = super::analyze_root(&mut job, RootRole::Residual, &roots.residual)
                .expect("lineage residual analysis");
            assert_eq!(
                analysis.value.exact_term_count, 0,
                "recompute_h={recompute_h}, terms={:?}",
                analysis.value.exact_terms
            );
            assert_eq!(
                analysis.value.bound,
                NumericContract::Known(CoefficientBound::ExactZero),
                "recompute_h={recompute_h}, terms={:?}",
                analysis.value.exact_terms
            );
        }
    }
}
