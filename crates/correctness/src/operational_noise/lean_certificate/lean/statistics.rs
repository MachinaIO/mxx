use crate::operational_noise::{
    facts::{CoefficientBound, NumericContract},
    lean_certificate::OwnerClaimStatistics,
    normal_form::BoundedSummary,
    simulation::{
        OperationalProofPayload, ProofPayloadEvent, ProofPayloadMonomial, ProofPayloadOwner,
        ProofPayloadRule, ProofPayloadScope, ProofPayloadTerm, ProofPayloadValue,
    },
};
use num_bigint::BigInt;
use num_traits::Zero;
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum OwnerClaimInvariantError {
    ContradictoryExactZero { owner: ProofPayloadOwner },
    MultiPayloadFactor { owner: ProofPayloadOwner },
    UnrecognizedFiniteFold { owner: ProofPayloadOwner, event: u64 },
    Structure(String),
}

impl fmt::Display for OwnerClaimInvariantError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContradictoryExactZero { owner } => write!(
                formatter,
                "semantic owner {owner:?} has coefficient-distinct exact-zero claims"
            ),
            Self::MultiPayloadFactor { owner } => {
                write!(formatter, "semantic factor owner {owner:?} has multiple result payloads")
            }
            Self::UnrecognizedFiniteFold { owner, event } => write!(
                formatter,
                "semantic owner {owner:?} has an unrecognized finite fold at event {event}"
            ),
            Self::Structure(reason) => formatter.write_str(reason),
        }
    }
}

impl From<String> for OwnerClaimInvariantError {
    fn from(value: String) -> Self {
        Self::Structure(value)
    }
}

impl From<&'static str> for OwnerClaimInvariantError {
    fn from(value: &'static str) -> Self {
        Self::Structure(value.to_owned())
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum SummaryKey {
    ExactZero,
    Finite(String),
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum CoefficientKey {
    Missing,
    ExactZero,
    Finite(String),
    Large,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum ClaimKey {
    Exact { coefficients: Vec<(ProofPayloadMonomial, BigInt)>, summary: SummaryKey },
    Coefficient(CoefficientKey),
}

#[derive(Clone)]
struct ResultOccurrence {
    event: u64,
    frame_start: u64,
    frame_root: ProofPayloadOwner,
    claim: ClaimKey,
    root_bindings: Vec<Binding>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Binding {
    input_position: u32,
    predecessor: u64,
    source_result: u64,
}

struct Frame {
    root: ProofPayloadOwner,
    start: u64,
    root_bindings: Vec<Binding>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DiagnosticReport {
    schema_id: &'static str,
    schema_version: u32,
    statistics: OwnerClaimStatistics,
    owners: Vec<OwnerRow>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OwnerRow {
    owner: OwnerDto,
    claims: Vec<ClaimRow>,
    exact_zero_consistent: bool,
    factor_present: bool,
    finite_fold_shape: FiniteFoldShape,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum FiniteFoldShape {
    None,
    Direct,
    Sum,
    DirectAndSum,
    Unknown,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ClaimRow {
    event: u64,
    frame_start: u64,
    frame_root: OwnerDto,
    result_kind: &'static str,
    claim_identity: u64,
    summary: Option<SummaryDto>,
    frame_root_predecessor_bindings: Vec<BindingDto>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OwnerDto {
    scope: ScopeDto,
    expression: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
enum ScopeDto {
    Closed { root_expression: u64 },
    Program { program: u64 },
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
enum SummaryDto {
    ExactZero,
    Finite { maximum_absolute_coefficient: String },
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BindingDto {
    input_position: u32,
    predecessor: u64,
    source_result: u64,
}

pub(crate) fn measure_owner_claims(
    proof: &OperationalProofPayload,
) -> Result<(OwnerClaimStatistics, Vec<u8>), OwnerClaimInvariantError> {
    let mut stack = Vec::<Frame>::new();
    let mut occurrences = BTreeMap::<ProofPayloadOwner, Vec<ResultOccurrence>>::new();
    let mut factor_owners = BTreeSet::<ProofPayloadOwner>::new();
    let mut factor_occurrences = 0_u64;

    for (index, event) in proof.events.iter().enumerate() {
        let event_index = u64::try_from(index).map_err(|_| "proof event index overflow")?;
        match event {
            ProofPayloadEvent::InvocationStart { root } => {
                stack.push(Frame { root: *root, start: event_index, root_bindings: Vec::new() })
            }
            ProofPayloadEvent::Predecessor {
                consumer,
                input_position,
                predecessor,
                source_result,
            } => {
                let frame = stack.last_mut().ok_or_else(|| {
                    format!("predecessor event {event_index} has no active frame")
                })?;
                if *consumer == frame.root {
                    frame.root_bindings.push(Binding {
                        input_position: *input_position,
                        predecessor: *predecessor,
                        source_result: *source_result,
                    });
                }
            }
            ProofPayloadEvent::Result { owner, value } => {
                let frame = stack
                    .last()
                    .ok_or_else(|| format!("result event {event_index} has no active frame"))?;
                if let ProofPayloadValue::Exact { terms, .. } = value {
                    collect_term_factors(terms, &mut factor_owners, &mut factor_occurrences)?;
                }
                occurrences.entry(*owner).or_default().push(ResultOccurrence {
                    event: event_index,
                    frame_start: frame.start,
                    frame_root: frame.root,
                    claim: claim_key(value),
                    root_bindings: frame.root_bindings.clone(),
                });
            }
            ProofPayloadEvent::InvocationEnd { root, result, .. } => {
                if let ProofPayloadValue::Exact { terms, .. } = result {
                    collect_term_factors(terms, &mut factor_owners, &mut factor_occurrences)?;
                }
                let frame = stack
                    .pop()
                    .ok_or_else(|| format!("invocation end event {event_index} has no frame"))?;
                if frame.root != *root {
                    return Err(OwnerClaimInvariantError::Structure(format!(
                        "invocation end event {event_index} does not match its active frame"
                    )));
                }
            }
            ProofPayloadEvent::PreFoldPolynomial(value) => {
                collect_term_factors(&value.terms, &mut factor_owners, &mut factor_occurrences)?;
            }
            ProofPayloadEvent::CoefficientMerge(value) => collect_monomial_factors(
                &value.output,
                &mut factor_owners,
                &mut factor_occurrences,
            )?,
            ProofPayloadEvent::AppliedRelation { source_monomial, rule, .. } => {
                collect_monomial_factors(
                    source_monomial,
                    &mut factor_owners,
                    &mut factor_occurrences,
                )?;
                if let crate::operational_noise::simulation::ProofPayloadRelationRule::Universal {
                    lhs,
                    ..
                } = rule
                {
                    collect_monomial_factors(lhs, &mut factor_owners, &mut factor_occurrences)?;
                }
            }
            ProofPayloadEvent::SpecializationComputed { .. } |
            ProofPayloadEvent::SpecializationCacheHit { .. } |
            ProofPayloadEvent::BoundTransfer { .. } |
            ProofPayloadEvent::SurvivorFold(_) => {}
        }
    }
    if !stack.is_empty() {
        return Err(OwnerClaimInvariantError::Structure(
            "proof payload ends with an active frame".to_owned(),
        ));
    }

    let mut statistics = OwnerClaimStatistics {
        result_events: occurrences.values().try_fold(0_u64, |sum, values| {
            sum.checked_add(u64::try_from(values.len()).map_err(|_| "result count overflow")?)
                .ok_or_else(|| "result count overflow".to_owned())
        })?,
        owners: u64::try_from(occurrences.len()).map_err(|_| "owner count overflow")?,
        multi_payload_owners: 0,
        exact_zero_occurrences: 0,
        finite_occurrences: 0,
        factor_occurrences,
        distinct_factor_owners: u64::try_from(factor_owners.len())
            .map_err(|_| "factor owner count overflow")?,
        factor_present_multi_payload_owners: 0,
        direct_fold_occurrences: 0,
        sum_fold_occurrences: 0,
        exact_zero_consistent_owners: 0,
        h2_owners: 0,
        unknown_owners: 0,
    };
    let mut rows = Vec::with_capacity(occurrences.len());
    for (owner, values) in occurrences {
        let identities = values.iter().map(|value| value.claim.clone()).collect::<BTreeSet<_>>();
        let multi = identities.len() > 1;
        let exact_zero_maps = values
            .iter()
            .filter_map(|value| match &value.claim {
                ClaimKey::Exact { coefficients, summary: SummaryKey::ExactZero } => {
                    Some(coefficients.clone())
                }
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        let exact_zero_consistent = exact_zero_maps.len() <= 1;
        let has_exact_zero = !exact_zero_maps.is_empty();
        let factor_present = factor_owners.contains(&owner);
        if factor_present && !exact_zero_consistent {
            return Err(OwnerClaimInvariantError::ContradictoryExactZero { owner });
        }
        if factor_present && multi {
            return Err(OwnerClaimInvariantError::MultiPayloadFactor { owner });
        }
        let mut direct = 0_u64;
        let mut sum = 0_u64;
        if multi {
            statistics.multi_payload_owners += 1;
            for value in &values {
                match &value.claim {
                    ClaimKey::Exact { summary: SummaryKey::ExactZero, .. } => {
                        statistics.exact_zero_occurrences += 1;
                    }
                    ClaimKey::Exact { coefficients, summary: SummaryKey::Finite(_) }
                        if has_exact_zero =>
                    {
                        statistics.finite_occurrences += 1;
                        if !coefficients.is_empty() {
                            return Err(OwnerClaimInvariantError::UnrecognizedFiniteFold {
                                owner,
                                event: value.event,
                            });
                        }
                        match finite_fold_shape(&proof.events, value.event, owner) {
                            FiniteFoldShape::Direct => {
                                direct += 1;
                                statistics.direct_fold_occurrences += 1;
                            }
                            FiniteFoldShape::Sum => {
                                sum += 1;
                                statistics.sum_fold_occurrences += 1;
                            }
                            _ => {
                                return Err(OwnerClaimInvariantError::UnrecognizedFiniteFold {
                                    owner,
                                    event: value.event,
                                });
                            }
                        }
                    }
                    ClaimKey::Coefficient(CoefficientKey::Finite(_)) if has_exact_zero => {
                        return Err(OwnerClaimInvariantError::UnrecognizedFiniteFold {
                            owner,
                            event: value.event,
                        });
                    }
                    _ => {}
                }
            }
            if has_exact_zero && exact_zero_consistent {
                statistics.exact_zero_consistent_owners += 1;
            }
            if factor_present {
                statistics.factor_present_multi_payload_owners += 1;
            }
            if !exact_zero_consistent && factor_present {
                statistics.h2_owners += 1;
            }
        }
        let identity_rows = identities
            .into_iter()
            .enumerate()
            .map(|(ordinal, claim)| (claim, ordinal))
            .collect::<BTreeMap<_, _>>();
        let claims = values
            .into_iter()
            .map(|value| {
                let claim_identity = identity_rows
                    .get_key_value(&value.claim)
                    .map(|(_, ordinal)| *ordinal)
                    .ok_or_else(|| "claim identity is missing".to_owned())?;
                Ok(ClaimRow {
                    event: value.event,
                    frame_start: value.frame_start,
                    frame_root: owner_dto(value.frame_root),
                    result_kind: match value.claim {
                        ClaimKey::Exact { .. } => "exact",
                        ClaimKey::Coefficient(_) => "coefficient",
                    },
                    claim_identity: u64::try_from(claim_identity)
                        .map_err(|_| "claim identity overflow")?,
                    summary: summary_dto(&value.claim),
                    frame_root_predecessor_bindings: value
                        .root_bindings
                        .into_iter()
                        .map(binding_dto)
                        .collect(),
                })
            })
            .collect::<Result<Vec<_>, OwnerClaimInvariantError>>()?;
        let finite_fold_shape = match (direct > 0, sum > 0) {
            (true, true) => FiniteFoldShape::DirectAndSum,
            (true, false) => FiniteFoldShape::Direct,
            (false, true) => FiniteFoldShape::Sum,
            (false, false) => FiniteFoldShape::None,
        };
        rows.push(OwnerRow {
            owner: owner_dto(owner),
            claims,
            exact_zero_consistent,
            factor_present,
            finite_fold_shape,
        });
    }
    let bytes = serde_json::to_vec(&DiagnosticReport {
        schema_id: "mxx.operational-noise.semantic-owner-statistics",
        schema_version: 1,
        statistics: statistics.clone(),
        owners: rows,
    })
    .map_err(|error| OwnerClaimInvariantError::Structure(error.to_string()))?;
    Ok((statistics, bytes))
}

fn claim_key(value: &ProofPayloadValue) -> ClaimKey {
    match value {
        ProofPayloadValue::Exact { terms, summary, .. } => {
            ClaimKey::Exact { coefficients: normalize_terms(terms), summary: summary_key(summary) }
        }
        ProofPayloadValue::Coefficient { bound } => ClaimKey::Coefficient(match bound {
            NumericContract::Missing => CoefficientKey::Missing,
            NumericContract::Known(CoefficientBound::ExactZero) => CoefficientKey::ExactZero,
            NumericContract::Known(CoefficientBound::Finite(value)) => {
                CoefficientKey::Finite(value.maximum_absolute_coefficient.to_string())
            }
            NumericContract::Known(CoefficientBound::Large) => CoefficientKey::Large,
        }),
    }
}

fn normalize_terms(terms: &[ProofPayloadTerm]) -> Vec<(ProofPayloadMonomial, BigInt)> {
    let mut coefficients = BTreeMap::<ProofPayloadMonomial, BigInt>::new();
    for term in terms {
        *coefficients.entry(term.monomial.clone()).or_default() += &term.coefficient;
    }
    coefficients.retain(|_, coefficient| !coefficient.is_zero());
    coefficients.into_iter().collect()
}

fn summary_key(value: &BoundedSummary) -> SummaryKey {
    match value.coefficient_bound() {
        NumericContract::Known(CoefficientBound::ExactZero) => SummaryKey::ExactZero,
        NumericContract::Known(CoefficientBound::Finite(value)) => {
            SummaryKey::Finite(value.maximum_absolute_coefficient.to_string())
        }
        NumericContract::Missing | NumericContract::Known(CoefficientBound::Large) => {
            unreachable!("BoundedSummary contains only exact-zero or finite")
        }
    }
}

fn collect_term_factors(
    terms: &[ProofPayloadTerm],
    owners: &mut BTreeSet<ProofPayloadOwner>,
    occurrences: &mut u64,
) -> Result<(), String> {
    for term in terms {
        collect_monomial_factors(&term.monomial, owners, occurrences)?;
    }
    Ok(())
}

fn collect_monomial_factors(
    monomial: &ProofPayloadMonomial,
    owners: &mut BTreeSet<ProofPayloadOwner>,
    occurrences: &mut u64,
) -> Result<(), String> {
    for owner in monomial.central_factors.iter().chain(&monomial.ordered_factors) {
        owners.insert(*owner);
        *occurrences = occurrences.checked_add(1).ok_or("factor occurrence count overflow")?;
    }
    Ok(())
}

fn finite_fold_shape(
    events: &[ProofPayloadEvent],
    result_event: u64,
    owner: ProofPayloadOwner,
) -> FiniteFoldShape {
    let Ok(index) = usize::try_from(result_event) else {
        return FiniteFoldShape::Unknown;
    };
    if index >= 2 &&
        matches!(
            events.get(index - 1),
            Some(ProofPayloadEvent::SurvivorFold(value)) if value.bound == result_event - 2
        ) &&
        matches!(
            events.get(index - 2),
            Some(ProofPayloadEvent::BoundTransfer {
                owner: transfer_owner,
                rule: ProofPayloadRule::MonomialProduct { .. },
            }) if *transfer_owner == owner
        )
    {
        return FiniteFoldShape::Direct;
    }
    if index >= 2 &&
        let Some(ProofPayloadEvent::SurvivorFold(survivor)) = events.get(index - 2)
    {
        if matches!(
            events.get(index - 1),
            Some(ProofPayloadEvent::BoundTransfer {
                owner: transfer_owner,
                rule: ProofPayloadRule::Sum { inputs },
            }) if *transfer_owner == owner && inputs.iter().any(|input| matches!(
                input,
                crate::operational_noise::simulation::ProofPayloadValueRef::Transfer(event)
                    if *event == survivor.bound
            ))
        ) {
            return FiniteFoldShape::Sum;
        }
    }
    FiniteFoldShape::Unknown
}

fn owner_dto(value: ProofPayloadOwner) -> OwnerDto {
    OwnerDto {
        scope: match value.scope {
            ProofPayloadScope::Closed { root_expression_row } => {
                ScopeDto::Closed { root_expression: root_expression_row }
            }
            ProofPayloadScope::Program { program_row } => {
                ScopeDto::Program { program: program_row }
            }
        },
        expression: value.expression_row,
    }
}

fn summary_dto(value: &ClaimKey) -> Option<SummaryDto> {
    match value {
        ClaimKey::Exact { summary: SummaryKey::ExactZero, .. } => Some(SummaryDto::ExactZero),
        ClaimKey::Exact { summary: SummaryKey::Finite(value), .. } => {
            Some(SummaryDto::Finite { maximum_absolute_coefficient: value.clone() })
        }
        ClaimKey::Coefficient(_) => None,
    }
}

fn binding_dto(value: Binding) -> BindingDto {
    BindingDto {
        input_position: value.input_position,
        predecessor: value.predecessor,
        source_result: value.source_result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::{
        g0::BoundProjection,
        simulation::{
            ProofPayloadCoefficientMerge, ProofPayloadCoefficientMergeSource,
            ProofPayloadPreFoldPolynomial, ProofPayloadRelationRule, ProofPayloadSurvivorFold,
            ProofPayloadTermRef,
        },
    };

    fn owner(expression_row: u64) -> ProofPayloadOwner {
        ProofPayloadOwner { scope: ProofPayloadScope::Program { program_row: 0 }, expression_row }
    }

    fn monomial(expression_row: u64) -> ProofPayloadMonomial {
        ProofPayloadMonomial {
            central_factors: vec![owner(expression_row)],
            ordered_factors: vec![],
        }
    }

    fn term(expression_row: u64, coefficient: i64) -> ProofPayloadTerm {
        ProofPayloadTerm { monomial: monomial(expression_row), coefficient: coefficient.into() }
    }

    fn finite_summary() -> BoundedSummary {
        BoundedSummary::finite(crate::operational_noise::facts::BoundExpression::new(4_u8.into()))
    }

    fn test_exact(terms: Vec<ProofPayloadTerm>, summary: BoundedSummary) -> ProofPayloadValue {
        let coefficient_bound = summary.coefficient_bound();
        let summary_producer =
            (!matches!(coefficient_bound, NumericContract::Known(CoefficientBound::ExactZero)))
                .then_some(0);
        ProofPayloadValue::Exact {
            terms,
            coefficient_bound,
            coefficient_producer: 0,
            summary,
            summary_producer,
        }
    }

    #[test]
    fn owner_claim_statistics_normalize_terms_and_classify_fold_shapes() {
        let root = owner(0);
        let value_owner = owner(1);
        let zero = BoundedSummary::zero();
        let finite = BoundedSummary::finite(crate::operational_noise::facts::BoundExpression::new(
            4_u8.into(),
        ));
        let proof = OperationalProofPayload {
            events: vec![
                ProofPayloadEvent::InvocationStart { root },
                ProofPayloadEvent::Predecessor {
                    consumer: root,
                    input_position: 0,
                    predecessor: 2,
                    source_result: 0,
                },
                ProofPayloadEvent::Result {
                    owner: value_owner,
                    value: test_exact(vec![term(9, 1)], zero.clone()),
                },
                ProofPayloadEvent::Result {
                    owner: value_owner,
                    value: test_exact(vec![term(9, 2), term(9, -1)], zero.clone()),
                },
                ProofPayloadEvent::BoundTransfer {
                    owner: value_owner,
                    rule: ProofPayloadRule::MonomialProduct {
                        monomial: monomial(9),
                        factors: vec![],
                    },
                },
                ProofPayloadEvent::SurvivorFold(ProofPayloadSurvivorFold {
                    coefficient: 1.into(),
                    bound: 4,
                }),
                ProofPayloadEvent::Result {
                    owner: value_owner,
                    value: test_exact(vec![], finite.clone()),
                },
                ProofPayloadEvent::CoefficientMerge(ProofPayloadCoefficientMerge {
                    owner: value_owner,
                    source: ProofPayloadCoefficientMergeSource::Operator {
                        inputs: [
                            ProofPayloadTermRef { value_event: 2, term_ordinal: 0 },
                            ProofPayloadTermRef { value_event: 3, term_ordinal: 0 },
                        ],
                    },
                    output: monomial(10),
                    signed_contribution: 1.into(),
                }),
                ProofPayloadEvent::AppliedRelation {
                    owner: value_owner,
                    source_monomial: monomial(11),
                    outer_coefficient: 1.into(),
                    ordered_start: 0,
                    ordered_end_exclusive: 0,
                    rule: ProofPayloadRelationRule::Universal {
                        computed: 0,
                        lhs: monomial(12),
                        lhs_layout: None,
                        rhs_result: 6,
                    },
                },
                ProofPayloadEvent::PreFoldPolynomial(ProofPayloadPreFoldPolynomial {
                    result_event: 6,
                    terms: vec![term(13, 1)],
                    summary: finite.clone(),
                    summary_evidence: Some(
                        crate::operational_noise::simulation::ProofPayloadValueRef::Result {
                            event: 6,
                            projection: BoundProjection::Summary,
                        },
                    ),
                }),
                ProofPayloadEvent::InvocationEnd {
                    root,
                    result: test_exact(vec![term(14, 1)], finite),
                    pre_fold_event: 9,
                },
            ],
        };

        let (statistics, first) = measure_owner_claims(&proof).expect("measure honest proof");
        let (_, second) = measure_owner_claims(&proof).expect("repeat deterministic measurement");
        assert_eq!(first, second);
        assert_eq!(statistics.result_events, 3);
        assert_eq!(statistics.owners, 1);
        assert_eq!(statistics.multi_payload_owners, 1);
        assert_eq!(statistics.exact_zero_occurrences, 2);
        assert_eq!(statistics.finite_occurrences, 1);
        assert_eq!(statistics.direct_fold_occurrences, 1);
        assert_eq!(statistics.sum_fold_occurrences, 0);
        assert_eq!(statistics.exact_zero_consistent_owners, 1);
        assert_eq!(statistics.factor_occurrences, 8);
        assert_eq!(statistics.distinct_factor_owners, 6);
        assert_eq!(statistics.factor_present_multi_payload_owners, 0);
        assert_eq!(statistics.h2_owners, 0);
        assert_eq!(statistics.unknown_owners, 0);
        let report: serde_json::Value = serde_json::from_slice(&first).expect("diagnostic JSON");
        assert_eq!(report["schemaId"], "mxx.operational-noise.semantic-owner-statistics");
        assert_eq!(report["schemaVersion"], 1);
        assert_eq!(report["owners"][0]["claims"][0]["frameStart"], 0);
        assert_eq!(
            report["owners"][0]["claims"][0]["frameRootPredecessorBindings"][0]["inputPosition"],
            0
        );
        assert_eq!(
            report["owners"][0]["claims"][0]["claimIdentity"],
            report["owners"][0]["claims"][1]["claimIdentity"]
        );
        assert_ne!(
            report["owners"][0]["claims"][0]["claimIdentity"],
            report["owners"][0]["claims"][2]["claimIdentity"]
        );
    }

    #[test]
    fn singleton_finite_claims_remain_event_level_obligations() {
        let root = owner(0);
        let coefficient_owner = owner(1);
        let exact_owner = owner(2);
        let finite = finite_summary();
        let proof = OperationalProofPayload {
            events: vec![
                ProofPayloadEvent::InvocationStart { root },
                ProofPayloadEvent::Result {
                    owner: coefficient_owner,
                    value: ProofPayloadValue::Coefficient {
                        bound: NumericContract::Known(CoefficientBound::Finite(
                            crate::operational_noise::facts::BoundExpression::new(4_u8.into()),
                        )),
                    },
                },
                ProofPayloadEvent::Result {
                    owner: exact_owner,
                    value: test_exact(vec![term(9, 1)], finite.clone()),
                },
                ProofPayloadEvent::PreFoldPolynomial(ProofPayloadPreFoldPolynomial {
                    result_event: 2,
                    terms: vec![],
                    summary: finite.clone(),
                    summary_evidence: None,
                }),
                ProofPayloadEvent::InvocationEnd {
                    root,
                    result: test_exact(vec![], finite),
                    pre_fold_event: 3,
                },
            ],
        };

        let (statistics, _) = measure_owner_claims(&proof).expect("measure singleton claims");
        assert_eq!(statistics.result_events, 2);
        assert_eq!(statistics.owners, 2);
        assert_eq!(statistics.multi_payload_owners, 0);
        assert_eq!(statistics.finite_occurrences, 0);
        assert_eq!(statistics.direct_fold_occurrences, 0);
        assert_eq!(statistics.sum_fold_occurrences, 0);
    }

    #[test]
    fn unrecognized_alternate_finite_claim_is_rejected() {
        let root = owner(0);
        let value_owner = owner(1);
        let zero = BoundedSummary::zero();
        let finite = finite_summary();
        let proof = OperationalProofPayload {
            events: vec![
                ProofPayloadEvent::InvocationStart { root },
                ProofPayloadEvent::Result {
                    owner: value_owner,
                    value: test_exact(vec![term(9, 1)], zero),
                },
                ProofPayloadEvent::Result {
                    owner: value_owner,
                    value: test_exact(vec![], finite.clone()),
                },
                ProofPayloadEvent::PreFoldPolynomial(ProofPayloadPreFoldPolynomial {
                    result_event: 2,
                    terms: vec![],
                    summary: finite.clone(),
                    summary_evidence: None,
                }),
                ProofPayloadEvent::InvocationEnd {
                    root,
                    result: test_exact(vec![], finite),
                    pre_fold_event: 3,
                },
            ],
        };

        assert_eq!(
            measure_owner_claims(&proof),
            Err(OwnerClaimInvariantError::UnrecognizedFiniteFold { owner: value_owner, event: 2 })
        );
    }

    #[test]
    fn contradictory_exact_zero_factor_claims_are_rejected() {
        let root = owner(0);
        let value_owner = owner(1);
        let zero = BoundedSummary::zero();
        let proof = OperationalProofPayload {
            events: vec![
                ProofPayloadEvent::InvocationStart { root },
                ProofPayloadEvent::Result {
                    owner: value_owner,
                    value: test_exact(vec![term(9, 1)], zero.clone()),
                },
                ProofPayloadEvent::Result {
                    owner: value_owner,
                    value: test_exact(vec![term(9, 2)], zero.clone()),
                },
                ProofPayloadEvent::PreFoldPolynomial(ProofPayloadPreFoldPolynomial {
                    result_event: 2,
                    terms: vec![term(1, 1)],
                    summary: zero.clone(),
                    summary_evidence: None,
                }),
                ProofPayloadEvent::InvocationEnd {
                    root,
                    result: test_exact(vec![], zero),
                    pre_fold_event: 3,
                },
            ],
        };

        assert_eq!(
            measure_owner_claims(&proof),
            Err(OwnerClaimInvariantError::ContradictoryExactZero { owner: value_owner })
        );
    }

    #[test]
    fn multi_payload_factor_owner_is_rejected() {
        let root = owner(0);
        let value_owner = owner(1);
        let zero = BoundedSummary::zero();
        let finite = BoundedSummary::finite(crate::operational_noise::facts::BoundExpression::new(
            4_u8.into(),
        ));
        let proof = OperationalProofPayload {
            events: vec![
                ProofPayloadEvent::InvocationStart { root },
                ProofPayloadEvent::Result {
                    owner: value_owner,
                    value: test_exact(vec![term(9, 1)], zero),
                },
                ProofPayloadEvent::BoundTransfer {
                    owner: value_owner,
                    rule: ProofPayloadRule::MonomialProduct {
                        monomial: monomial(9),
                        factors: vec![],
                    },
                },
                ProofPayloadEvent::SurvivorFold(ProofPayloadSurvivorFold {
                    coefficient: 1.into(),
                    bound: 2,
                }),
                ProofPayloadEvent::Result {
                    owner: value_owner,
                    value: test_exact(vec![], finite.clone()),
                },
                ProofPayloadEvent::PreFoldPolynomial(ProofPayloadPreFoldPolynomial {
                    result_event: 4,
                    terms: vec![term(1, 1)],
                    summary: finite.clone(),
                    summary_evidence: None,
                }),
                ProofPayloadEvent::InvocationEnd {
                    root,
                    result: test_exact(vec![], finite),
                    pre_fold_event: 5,
                },
            ],
        };

        assert_eq!(
            measure_owner_claims(&proof),
            Err(OwnerClaimInvariantError::MultiPayloadFactor { owner: value_owner })
        );
    }
}
