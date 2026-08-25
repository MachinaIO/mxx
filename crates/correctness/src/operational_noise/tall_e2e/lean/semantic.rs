use super::{NAMESPACE, generated_file};
use crate::operational_noise::{
    certificate_schema::CertificateDocumentV1,
    g0::{
        CanonicalExpressionDescriptor, CanonicalExpressionOperator, StableMatrixOperation,
        StableOperator,
    },
    simulation::{
        OperationalProofPayload, ProofPayloadCoefficientMergeSource, ProofPayloadEvent,
        ProofPayloadMonomial, ProofPayloadOwner, ProofPayloadRelationRule, ProofPayloadRule,
        ProofPayloadTerm, ProofPayloadValue, ProofPayloadValueRef,
    },
};
use num_traits::Zero;
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
};

const SCHEMA_ID: &str = "mxx.operational-noise.tall-semantic-probe-statistics";
const SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct OwnerDto {
    scope: ScopeDto,
    expression: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
enum ScopeDto {
    Closed { root_expression: u64 },
    Program { program: u64 },
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct NodeStat {
    event: u64,
    owner: OwnerDto,
    term_count: u64,
    max_central_factor_length: u64,
    max_ordered_factor_length: u64,
    max_monomial_factor_length: u64,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProbeStat {
    name: &'static str,
    event: Option<u64>,
    owner: Option<OwnerDto>,
    frame_start: Option<u64>,
    frame_end: Option<u64>,
    score: u64,
    detail: &'static str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StatisticsReport {
    schema_id: &'static str,
    schema_version: u32,
    event_count: u64,
    exact_result_nodes: u64,
    max_term_count: Option<NodeStat>,
    max_monomial_factor_length: Option<NodeStat>,
    probes: Vec<ProbeStat>,
}

#[derive(Clone)]
struct Frame {
    root: ProofPayloadOwner,
    start: u64,
    merge_count: u64,
    add_chain: u64,
    max_add_chain: u64,
    max_add_chain_event: Option<u64>,
    has_prefold: bool,
}

#[derive(Clone)]
struct Selection {
    event: u64,
    owner: ProofPayloadOwner,
    detail: &'static str,
    score: u64,
    frame_start: Option<u64>,
    frame_end: Option<u64>,
}

#[derive(Clone)]
struct FrameSelection {
    start: u64,
    end: u64,
    root: ProofPayloadOwner,
    merge_count: u64,
    detail: &'static str,
}

#[derive(Clone)]
struct ResultRecord {
    event: u64,
    owner: ProofPayloadOwner,
    terms: Vec<ProofPayloadTerm>,
    summary: crate::operational_noise::normal_form::BoundedSummary,
}

#[derive(Clone)]
struct OperationProbe {
    kind: OperationKind,
    inputs: Vec<ResultRecord>,
    output: ResultRecord,
    scalar_left: bool,
    scalar_right: bool,
    raw_work: u64,
    chain_tail: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OperationKind {
    Direct,
    Add,
    Subtract,
    Multiply,
    Tensor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RelationRuleKind {
    Gadget,
    Universal,
}

#[derive(Clone)]
struct RelationProbe {
    event: u64,
    owner: ProofPayloadOwner,
    source: ProofPayloadMonomial,
    lhs: ProofPayloadMonomial,
    outer: num_bigint::BigInt,
    start: u32,
    end: u32,
    accumulator: ResultRecord,
    rhs: ResultRecord,
    output: ResultRecord,
    kind: RelationRuleKind,
}

struct PendingRelation {
    event: u64,
    owner: ProofPayloadOwner,
    source: ProofPayloadMonomial,
    lhs: ProofPayloadMonomial,
    outer: num_bigint::BigInt,
    start: u32,
    end: u32,
    accumulator: ResultRecord,
    rhs: ResultRecord,
    terms: BTreeMap<ProofPayloadMonomial, num_bigint::BigInt>,
    last_merge_event: Option<u64>,
    kind: RelationRuleKind,
}

#[derive(Clone)]
struct BoundProbe {
    root: ResultRecord,
    prefold_terms: Vec<ProofPayloadTerm>,
    prefold_summary: crate::operational_noise::normal_form::BoundedSummary,
    end: ResultRecord,
    survivor_contributions: Vec<String>,
    survivor_bounds: Vec<String>,
}

#[derive(Clone)]
struct ProbeSelection {
    name: &'static str,
    event: u64,
    owner: ProofPayloadOwner,
    score: u64,
    detail: &'static str,
    frame_start: Option<u64>,
    frame_end: Option<u64>,
    long_key: Option<ProofPayloadMonomial>,
    operation: Option<OperationProbe>,
    relations: Vec<RelationProbe>,
    bound: Option<BoundProbe>,
}

impl ProbeStat {
    fn from_probe(probe: &ProbeSelection) -> Self {
        Self {
            name: probe.name,
            event: Some(probe.event),
            owner: Some(owner_dto(probe.owner)),
            frame_start: probe.frame_start,
            frame_end: probe.frame_end,
            score: probe.score,
            detail: probe.detail,
        }
    }
}

fn immediate_frame_map(events: &[ProofPayloadEvent]) -> Result<Vec<Option<u64>>, String> {
    let mut stack = Vec::new();
    let mut frames = vec![None; events.len()];
    for (position, event) in events.iter().enumerate() {
        if matches!(event, ProofPayloadEvent::InvocationStart { .. }) {
            stack.push(u64::try_from(position).map_err(|_| "semantic event index overflow")?);
        }
        frames[position] = stack.last().copied();
        if matches!(event, ProofPayloadEvent::InvocationEnd { .. }) {
            stack.pop();
        }
    }
    if !stack.is_empty() {
        return Err("semantic frame stack not empty".to_owned());
    }
    Ok(frames)
}

impl PayloadIndex {
    fn new(proof: &OperationalProofPayload) -> Result<Self, String> {
        let mut index = Self {
            events: proof.events.clone(),
            immediate_frames: immediate_frame_map(&proof.events)?,
            predecessors: BTreeMap::new(),
            results: Vec::new(),
            by_event: BTreeMap::new(),
            operations: Vec::new(),
            merges: Vec::new(),
            relations: Vec::new(),
            prefolds: Vec::new(),
            ends: Vec::new(),
            survivors: Vec::new(),
        };
        for (position, _event) in proof.events.iter().enumerate() {
            let event = u64::try_from(position).map_err(|_| "semantic event index overflow")?;
            match &proof.events[position] {
                ProofPayloadEvent::Result {
                    owner,
                    value: ProofPayloadValue::Exact { terms, summary },
                } => {
                    let record = ResultRecord {
                        event,
                        owner: *owner,
                        terms: terms.clone(),
                        summary: summary.clone(),
                    };
                    index.by_event.insert(event, record.clone());
                    index.results.push(record);
                }
                ProofPayloadEvent::InvocationEnd {
                    root,
                    result: ProofPayloadValue::Exact { terms, summary },
                    pre_fold_event,
                } => {
                    index.by_event.insert(
                        event,
                        ResultRecord {
                            event,
                            owner: *root,
                            terms: terms.clone(),
                            summary: summary.clone(),
                        },
                    );
                    index.ends.push((
                        event,
                        *root,
                        ProofPayloadValue::Exact { terms: terms.clone(), summary: summary.clone() },
                        *pre_fold_event,
                    ));
                }
                ProofPayloadEvent::Predecessor {
                    consumer, input_position, source_result, ..
                } => {
                    index.predecessors.insert(event, (*consumer, *input_position, *source_result));
                }
                ProofPayloadEvent::BoundTransfer { owner, rule } => {
                    index.operations.push((*owner, rule.clone(), event))
                }
                ProofPayloadEvent::AppliedRelation {
                    owner,
                    source_monomial,
                    outer_coefficient,
                    ordered_start,
                    ordered_end_exclusive,
                    rule,
                } => index.relations.push((
                    event,
                    *owner,
                    source_monomial.clone(),
                    outer_coefficient.clone(),
                    *ordered_start,
                    *ordered_end_exclusive,
                    rule.clone(),
                )),
                ProofPayloadEvent::PreFoldPolynomial(value) => {
                    index.prefolds.push((event, value.clone()))
                }
                ProofPayloadEvent::SurvivorFold(fold) => {
                    index.survivors.push((event, fold.clone()))
                }
                ProofPayloadEvent::CoefficientMerge(merge) => {
                    index.merges.push((event, merge.clone()))
                }
                _ => {}
            }
        }
        Ok(index)
    }

    fn result(&self, event: u64) -> Result<ResultRecord, String> {
        self.by_event
            .get(&event)
            .cloned()
            .ok_or_else(|| format!("payload reference {event} does not identify an exact Result"))
    }

    fn event(&self, event: u64) -> Result<&ProofPayloadEvent, String> {
        self.events
            .get(usize::try_from(event).map_err(|_| "semantic event index overflow")?)
            .ok_or_else(|| format!("semantic event reference {event} is out of range"))
    }

    fn value_ref(
        &self,
        owner: ProofPayloadOwner,
        value: &ProofPayloadValueRef,
    ) -> Result<ResultRecord, String> {
        self.value_ref_seen(owner, value, &mut BTreeSet::new())
    }

    fn value_ref_seen(
        &self,
        owner: ProofPayloadOwner,
        value: &ProofPayloadValueRef,
        seen_transfers: &mut BTreeSet<u64>,
    ) -> Result<ResultRecord, String> {
        match value {
            ProofPayloadValueRef::Result { event, .. } => self.result(*event),
            ProofPayloadValueRef::Predecessor { binding_event, input_position, .. } => {
                let (consumer, position, source_result) = self.predecessors.get(binding_event).ok_or_else(|| {
                    format!("payload predecessor reference {binding_event} is not an exact Predecessor event")
                })?;
                if *consumer != owner || *position != *input_position {
                    return Err(format!(
                        "payload predecessor reference {binding_event} owner/input mismatch"
                    ));
                }
                self.result(*source_result)
            }
            ProofPayloadValueRef::Transfer(event) => {
                if !seen_transfers.insert(*event) {
                    return Err(format!("transfer reference cycle at event {event}"));
                }
                let ProofPayloadEvent::BoundTransfer { owner: transfer_owner, rule } =
                    self.event(*event)?
                else {
                    return Err(format!("transfer reference {event} is not a BoundTransfer event"));
                };
                if *transfer_owner != owner {
                    return Err(format!("transfer reference {event} owner mismatch"));
                }
                match rule {
                    ProofPayloadRule::Identity { input } => {
                        self.value_ref_seen(owner, input, seen_transfers)
                    }
                    ProofPayloadRule::Sum { inputs } if inputs.len() == 1 => {
                        self.value_ref_seen(owner, &inputs[0], seen_transfers)
                    }
                    _ => Err(format!(
                        "transfer reference {event} denotes a bound-only rule without an exact polynomial"
                    )),
                }
            }
        }
    }
}

fn expression_kind(
    statement: &CertificateDocumentV1,
    owner: ProofPayloadOwner,
) -> Option<OperationKind> {
    let row = statement.expressions.get(usize::try_from(owner.expression_row).ok()?)?;
    let CanonicalExpressionDescriptor::Operation {
        operator: CanonicalExpressionOperator::Stable(StableOperator::Matrix { operation }),
        ..
    } = &row.descriptor
    else {
        return None;
    };
    Some(match operation {
        StableMatrixOperation::Add => OperationKind::Add,
        StableMatrixOperation::Subtract => OperationKind::Subtract,
        StableMatrixOperation::Multiply => OperationKind::Multiply,
        StableMatrixOperation::Tensor { .. } => OperationKind::Tensor,
        _ => return None,
    })
}

fn typed_operation_rule(
    index: &PayloadIndex,
    result: &ResultRecord,
    kind: OperationKind,
) -> Result<Option<(u64, [ProofPayloadValueRef; 2], (bool, bool))>, String> {
    let result_frame = index.immediate_frames
        [usize::try_from(result.event).map_err(|_| "semantic event index overflow")?];
    let mut candidates = index
        .operations
        .iter()
        .filter(|(owner, _, event)| {
            *owner == result.owner &&
                *event < result.event &&
                index.immediate_frames[usize::try_from(*event).expect("indexed operation event")] ==
                    result_frame
        })
        .filter_map(|(_, rule, event)| match (kind, rule) {
            (OperationKind::Add | OperationKind::Subtract, ProofPayloadRule::Sum { inputs }) => {
                (inputs.len() == 2)
                    .then(|| (*event, [inputs[0].clone(), inputs[1].clone()], (false, false)))
            }
            (OperationKind::Multiply, ProofPayloadRule::Product { left, right, facts }) => Some((
                *event,
                [left.clone(), right.clone()],
                (facts.left_is_constant_polynomial, facts.right_is_constant_polynomial),
            )),
            (
                OperationKind::Tensor,
                ProofPayloadRule::Tensor {
                    left,
                    right,
                    left_is_constant_polynomial,
                    right_is_constant_polynomial,
                },
            ) => Some((
                *event,
                [left.clone(), right.clone()],
                (*left_is_constant_polynomial, *right_is_constant_polynomial),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    candidates.sort_by_key(|(event, _, _)| *event);
    let Some((event, refs, flags)) = candidates.pop() else {
        return Ok(None);
    };
    Ok(Some((event, refs, flags)))
}

fn op_probe(
    statement: &CertificateDocumentV1,
    index: &PayloadIndex,
    result: &ResultRecord,
    kind: OperationKind,
) -> Result<OperationProbe, String> {
    let Some((rule_event, input_refs, flags)) = typed_operation_rule(index, result, kind)? else {
        return Err(format!(
            "operator Result {} owner {} has no preceding typed {:?} rule",
            result.event,
            owner_text(result.owner),
            kind,
        ));
    };
    let operator_merges = index
        .merges
        .iter()
        .filter(|(event, merge)| {
            *event > rule_event &&
                *event < result.event &&
                merge.owner == result.owner &&
                index.immediate_frames[usize::try_from(*event).expect("indexed merge event")] ==
                    index.immediate_frames
                        [usize::try_from(result.event).expect("indexed result event")] &&
                matches!(merge.source, ProofPayloadCoefficientMergeSource::Operator { .. })
        })
        .collect::<Vec<_>>();
    let typed_inputs = [
        index.value_ref(result.owner, &input_refs[0]),
        index.value_ref(result.owner, &input_refs[1]),
    ];
    let inputs = match typed_inputs {
        [Ok(left), Ok(right)] => vec![left, right],
        [left, right] => {
            let Some((_, merge)) = operator_merges.first() else {
                return Err(left
                    .err()
                    .or_else(|| right.err())
                    .expect("one typed operator input failed"));
            };
            let ProofPayloadCoefficientMergeSource::Operator { inputs: refs } = &merge.source
            else {
                unreachable!()
            };
            vec![index.result(refs[0].value_event)?, index.result(refs[1].value_event)?]
        }
    };
    for (_, merge) in &operator_merges {
        let ProofPayloadCoefficientMergeSource::Operator { inputs: refs } = &merge.source else {
            unreachable!()
        };
        let pair = [refs[0].value_event, refs[1].value_event];
        let expected = [inputs[0].event, inputs[1].event];
        if pair != expected {
            return Err(format!(
                "operator Result {} has inconsistent typed merge input Result refs",
                result.event
            ));
        }
        for (term_ref, input) in refs.iter().zip(&inputs) {
            if usize::try_from(term_ref.term_ordinal)
                .ok()
                .is_none_or(|ordinal| ordinal >= input.terms.len())
            {
                return Err(format!(
                    "operator Result {} has out-of-range typed merge term ref",
                    result.event
                ));
            }
        }
    }
    let raw_work = (inputs[0].terms.len() as u64).saturating_mul(inputs[1].terms.len() as u64);
    let chain_tail = (kind == OperationKind::Add)
        .then(|| operator_merges.iter().map(|(event, _)| *event).max())
        .flatten();
    let _ = statement;
    Ok(OperationProbe {
        kind,
        inputs,
        output: result.clone(),
        scalar_left: flags.0,
        scalar_right: flags.1,
        raw_work,
        chain_tail,
    })
}

fn typed_operator_eligibility(
    index: &PayloadIndex,
    result: &ResultRecord,
    kind: OperationKind,
) -> Result<bool, String> {
    if !matches!(
        result.summary.coefficient_bound(),
        crate::operational_noise::facts::NumericContract::Known(
            crate::operational_noise::facts::CoefficientBound::ExactZero
        )
    ) {
        return Ok(false);
    }
    let Some((_, refs, _)) = typed_operation_rule(index, result, kind)? else {
        return Ok(false);
    };
    for value in refs {
        if let Err(error) = index.value_ref(result.owner, &value) {
            if matches!(value, ProofPayloadValueRef::Transfer(_)) {
                return Ok(false);
            }
            return Err(error);
        }
    }
    Ok(true)
}

fn finalize_relations(
    pending: &mut Vec<PendingRelation>,
    candidates: &mut Vec<RelationProbe>,
) -> Result<(), String> {
    for relation in pending.drain(..) {
        let output_event = relation.last_merge_event.ok_or_else(|| {
            format!(
                "relation application {} has no typed Relation coefficient merge",
                relation.event
            )
        })?;
        let output = ResultRecord {
            event: output_event,
            owner: relation.owner,
            terms: relation
                .terms
                .iter()
                .filter_map(|(monomial, coefficient)| {
                    (!coefficient.is_zero()).then_some(ProofPayloadTerm {
                        monomial: monomial.clone(),
                        coefficient: coefficient.clone(),
                    })
                })
                .collect(),
            summary: relation.accumulator.summary.clone(),
        };
        candidates.push(RelationProbe {
            event: relation.event,
            owner: relation.owner,
            source: relation.source,
            lhs: relation.lhs,
            outer: relation.outer,
            start: relation.start,
            end: relation.end,
            accumulator: relation.accumulator,
            rhs: relation.rhs,
            output,
            kind: relation.kind,
        });
    }
    Ok(())
}

fn relation_candidates(
    index: &PayloadIndex,
    ranges: &[(u64, u64, ProofPayloadOwner, u64)],
) -> Result<Vec<RelationProbe>, String> {
    let mut candidates = Vec::new();
    for (frame_start, frame_end, _, _) in ranges {
        let mut states = BTreeMap::<ProofPayloadOwner, ResultRecord>::new();
        let mut working =
            BTreeMap::<ProofPayloadOwner, BTreeMap<ProofPayloadMonomial, num_bigint::BigInt>>::new(
            );
        let mut pending = Vec::<PendingRelation>::new();
        for event in *frame_start..=*frame_end {
            let is_relation_merge_for_pending = matches!(
                index.event(event)?,
                ProofPayloadEvent::CoefficientMerge(merge)
                    if matches!(merge.source, ProofPayloadCoefficientMergeSource::Relation { application, .. }
                        if pending.iter().any(|relation| relation.event == application && relation.owner == merge.owner))
            );
            if !is_relation_merge_for_pending && !pending.is_empty() {
                finalize_relations(&mut pending, &mut candidates)?;
            }
            let is_immediate = index.immediate_frames
                [usize::try_from(event).map_err(|_| "semantic event index overflow")?] ==
                Some(*frame_start);
            if !is_immediate {
                if let ProofPayloadEvent::CoefficientMerge(merge) = index.event(event)? {
                    if matches!(merge.source, ProofPayloadCoefficientMergeSource::Operator { .. }) {
                        *working
                            .entry(merge.owner)
                            .or_default()
                            .entry(merge.output.clone())
                            .or_default() += &merge.signed_contribution;
                    }
                }
                continue;
            }
            match index.event(event)? {
                ProofPayloadEvent::Result {
                    owner,
                    value: ProofPayloadValue::Exact { terms, summary },
                } => {
                    let result = ResultRecord {
                        event,
                        owner: *owner,
                        terms: terms.clone(),
                        summary: summary.clone(),
                    };
                    states.insert(*owner, result);
                    working.insert(
                        *owner,
                        terms
                            .iter()
                            .map(|term| (term.monomial.clone(), term.coefficient.clone()))
                            .collect(),
                    );
                }
                ProofPayloadEvent::InvocationEnd {
                    root,
                    result: ProofPayloadValue::Exact { terms, summary },
                    ..
                } => {
                    let record = ResultRecord {
                        event,
                        owner: *root,
                        terms: terms.clone(),
                        summary: summary.clone(),
                    };
                    states.insert(*root, record);
                    working.insert(
                        *root,
                        terms
                            .iter()
                            .map(|term| (term.monomial.clone(), term.coefficient.clone()))
                            .collect(),
                    );
                }
                ProofPayloadEvent::AppliedRelation {
                    owner,
                    source_monomial,
                    outer_coefficient,
                    ordered_start,
                    ordered_end_exclusive,
                    rule,
                } => {
                    let accumulator_terms = working.get(owner).cloned().ok_or_else(|| {
                        format!("relation event {event} has no frame-local owner polynomial")
                    })?;
                    let accumulator_summary =
                        states.get(owner).map(|result| result.summary.clone()).unwrap_or_else(
                            crate::operational_noise::normal_form::BoundedSummary::zero,
                        );
                    let accumulator = ResultRecord {
                        event,
                        owner: *owner,
                        terms: accumulator_terms
                            .iter()
                            .filter_map(|(monomial, coefficient)| {
                                (!coefficient.is_zero()).then_some(ProofPayloadTerm {
                                    monomial: monomial.clone(),
                                    coefficient: coefficient.clone(),
                                })
                            })
                            .collect(),
                        summary: accumulator_summary,
                    };
                    if !accumulator.terms.iter().any(|term| term.monomial == *source_monomial) {
                        return Err(format!(
                            "relation event {event} source is absent from frame-local accumulator"
                        ));
                    }
                    let (lhs, rhs_event, kind) = match rule {
                        ProofPayloadRelationRule::Universal { lhs, rhs_result, .. } => {
                            (lhs.clone(), *rhs_result, RelationRuleKind::Universal)
                        }
                        ProofPayloadRelationRule::Gadget {
                            gadget,
                            decomposition,
                            input_result,
                            ..
                        } => (
                            ProofPayloadMonomial {
                                central_factors: Vec::new(),
                                ordered_factors: vec![*gadget, *decomposition],
                            },
                            *input_result,
                            RelationRuleKind::Gadget,
                        ),
                    };
                    let rhs = index.result(rhs_event)?;
                    let mut terms = accumulator
                        .terms
                        .iter()
                        .map(|term| (term.monomial.clone(), term.coefficient.clone()))
                        .collect::<BTreeMap<_, _>>();
                    *terms.entry(source_monomial.clone()).or_default() -= outer_coefficient;
                    pending.push(PendingRelation {
                        event,
                        owner: *owner,
                        source: source_monomial.clone(),
                        lhs,
                        outer: outer_coefficient.clone(),
                        start: *ordered_start,
                        end: *ordered_end_exclusive,
                        accumulator,
                        rhs,
                        terms,
                        last_merge_event: None,
                        kind,
                    });
                }
                ProofPayloadEvent::CoefficientMerge(merge) => {
                    let ProofPayloadCoefficientMergeSource::Relation { application, .. } =
                        merge.source
                    else {
                        *working
                            .entry(merge.owner)
                            .or_default()
                            .entry(merge.output.clone())
                            .or_default() += &merge.signed_contribution;
                        continue;
                    };
                    let Some(relation) = pending.iter_mut().find(|relation| {
                        relation.event == application && relation.owner == merge.owner
                    }) else {
                        return Err(format!(
                            "relation merge event {event} references unknown application {application}"
                        ));
                    };
                    *relation.terms.entry(merge.output.clone()).or_default() +=
                        &merge.signed_contribution;
                    relation.last_merge_event = Some(event);
                }
                _ => {}
            }
        }
        if !pending.is_empty() {
            finalize_relations(&mut pending, &mut candidates).map_err(|error| {
                format!(
                    "frame {frame_start}..{frame_end} has unresolved relation applications: {error}"
                )
            })?;
        }
    }
    Ok(candidates)
}

fn frame_ranges(
    proof: &OperationalProofPayload,
) -> Result<Vec<(u64, u64, ProofPayloadOwner, u64)>, String> {
    let mut stack = Vec::<(u64, ProofPayloadOwner, u64)>::new();
    let mut ranges = Vec::new();
    for (position, event) in proof.events.iter().enumerate() {
        let event_id = u64::try_from(position).map_err(|_| "semantic event index overflow")?;
        match event {
            ProofPayloadEvent::InvocationStart { root } => stack.push((event_id, *root, 0)),
            ProofPayloadEvent::CoefficientMerge(_) => {
                if let Some(frame) = stack.last_mut() {
                    frame.2 += 1;
                }
            }
            ProofPayloadEvent::InvocationEnd { root, .. } => {
                let (start, expected, merges) =
                    stack.pop().ok_or_else(|| format!("semantic frame underflow at {event_id}"))?;
                if expected != *root {
                    return Err(format!("semantic frame root mismatch at {event_id}"));
                }
                ranges.push((start, event_id, *root, merges));
            }
            _ => {}
        }
    }
    if !stack.is_empty() {
        return Err("semantic frame stack not empty".to_owned());
    }
    Ok(ranges)
}

fn build_probes(
    statement: &CertificateDocumentV1,
    proof: &OperationalProofPayload,
    index: &PayloadIndex,
) -> Result<Vec<ProbeSelection>, String> {
    let mut probes = Vec::new();
    let long = proof
        .events
        .iter()
        .enumerate()
        .filter_map(|(position, event)| match event {
            ProofPayloadEvent::CoefficientMerge(merge) => {
                Some((u64::try_from(position).ok()?, merge.owner, merge.output.clone()))
            }
            _ => None,
        })
        .max_by_key(|(event, _, key)| {
            (key.central_factors.len() + key.ordered_factors.len(), std::cmp::Reverse(*event))
        });
    let (long_event, long_owner, long_key) =
        long.ok_or_else(|| "CP3 long-monomial probe has no coefficient merge".to_owned())?;
    probes.push(ProbeSelection {
        name: "long-monomial-merge",
        event: long_event,
        owner: long_owner,
        score: (long_key.central_factors.len() + long_key.ordered_factors.len()) as u64,
        detail: "actual coefficient merge output monomial",
        frame_start: None,
        frame_end: None,
        long_key: Some(long_key),
        operation: None,
        relations: Vec::new(),
        bound: None,
    });

    let ranges = frame_ranges(proof)?;
    let outer = ranges
        .iter()
        .max_by_key(|(_, end, _, merges)| (*merges, std::cmp::Reverse(*end)))
        .ok_or_else(|| "CP3 outer-result probe has no invocation frame".to_owned())?;
    let outer_result = index
        .results
        .iter()
        .filter(|result| {
            result.owner == outer.2 && result.event >= outer.0 && result.event <= outer.1
        })
        .max_by_key(|result| result.event)
        .cloned()
        .ok_or_else(|| {
            format!("8301-merge outer frame {}..{} has no exact root Result", outer.0, outer.1)
        })?;
    let outer_kind = expression_kind(statement, outer_result.owner).ok_or_else(|| {
        format!(
            "outer Result {} owner {} has no matrix Add/Subtract/Multiply/Tensor statement row",
            outer_result.event,
            owner_text(outer_result.owner)
        )
    })?;
    let outer_op = if typed_operator_eligibility(index, &outer_result, outer_kind)? {
        op_probe(statement, index, &outer_result, outer_kind)?
    } else {
        OperationProbe {
            kind: OperationKind::Direct,
            inputs: Vec::new(),
            output: outer_result.clone(),
            scalar_left: false,
            scalar_right: false,
            raw_work: 0,
            chain_tail: None,
        }
    };
    probes.push(ProbeSelection {
        name: "outer-result",
        event: outer_result.event,
        owner: outer_result.owner,
        score: outer.3,
        detail: "actual maximum-merge invocation root Result",
        frame_start: Some(outer.0),
        frame_end: Some(outer.1),
        long_key: None,
        operation: Some(outer_op),
        relations: Vec::new(),
        bound: None,
    });

    let mut add_candidates = index
        .results
        .iter()
        .filter_map(|result| {
            if expression_kind(statement, result.owner) != Some(OperationKind::Add) {
                return None;
            }
            let eligible = match typed_operator_eligibility(index, result, OperationKind::Add) {
                Ok(eligible) => eligible,
                Err(error) => return Some(Err(error)),
            };
            if !eligible {
                return None;
            }
            Some(
                op_probe(statement, index, result, OperationKind::Add).map(|probe| (result, probe)),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    if add_candidates.is_empty() {
        return Err("CP3 add-chain probe has no actual matrix Add Result".to_owned());
    }
    add_candidates.sort_by_key(|(result, operation)| {
        (result.terms.len(), operation.inputs[0].terms.len(), std::cmp::Reverse(result.event))
    });
    let (add_result, add_op) = add_candidates.pop().expect("nonempty add candidates");
    probes.push(ProbeSelection {
        name: "add-chain",
        event: add_result.event,
        owner: add_result.owner,
        score: add_result.terms.len() as u64,
        detail: "actual maximum intermediate Add Result",
        frame_start: None,
        frame_end: None,
        long_key: None,
        operation: Some(add_op),
        relations: Vec::new(),
        bound: None,
    });

    let mut product_candidates = index
        .results
        .iter()
        .filter_map(|result| {
            let kind = expression_kind(statement, result.owner)?;
            if !matches!(kind, OperationKind::Multiply | OperationKind::Tensor) {
                return None;
            }
            let eligible = match typed_operator_eligibility(index, result, kind) {
                Ok(eligible) => eligible,
                Err(error) => return Some(Err(error)),
            };
            if !eligible {
                return None;
            }
            Some(op_probe(statement, index, result, kind).map(|probe| (result, probe)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if product_candidates.is_empty() {
        return Err(
            "CP3 Product/Tensor probe has no actual reached Product/Tensor Result".to_owned()
        );
    }
    product_candidates.sort_by_key(|(result, operation)| {
        (operation.raw_work, result.terms.len(), std::cmp::Reverse(result.event))
    });
    let (product_result, product_op) =
        product_candidates.pop().expect("nonempty product candidates");
    probes.push(ProbeSelection {
        name: "product-tensor",
        event: product_result.event,
        owner: product_result.owner,
        score: product_op.raw_work,
        detail: "actual maximum raw Product/Tensor input work Result",
        frame_start: None,
        frame_end: None,
        long_key: None,
        operation: Some(product_op),
        relations: Vec::new(),
        bound: None,
    });

    let mut relation_probes = relation_candidates(index, &ranges)?;
    relation_probes.sort_by_key(|probe| {
        (
            matches!(probe.kind, RelationRuleKind::Universal),
            probe.output.terms.len(),
            std::cmp::Reverse(probe.event),
        )
    });
    let mut selected_relations = Vec::new();
    for kind in [RelationRuleKind::Gadget, RelationRuleKind::Universal] {
        if let Some(probe) = relation_probes
            .iter()
            .filter(|probe| probe.kind == kind)
            .max_by_key(|probe| (probe.output.terms.len(), std::cmp::Reverse(probe.event)))
        {
            selected_relations.push(probe.clone());
        }
    }
    if selected_relations.is_empty() {
        return Err("CP3 relation probe cannot map a reached AppliedRelation to exact accumulator, RHS Result, and output Result".to_owned());
    }
    probes.push(ProbeSelection {
        name: "relation",
        event: selected_relations[0].event,
        owner: selected_relations[0].owner,
        score: selected_relations
            .iter()
            .map(|probe| probe.output.terms.len() as u64)
            .max()
            .unwrap_or(0),
        detail: "actual maximum reached Gadget and Universal relation source/lhs/rhs/output",
        frame_start: None,
        frame_end: None,
        long_key: None,
        operation: None,
        relations: selected_relations,
        bound: None,
    });

    let mut bound_candidates = Vec::new();
    for (start, end, root, _) in ranges {
        let Some((end_event, end_owner, end_value, pre_fold_event)) =
            index.ends.iter().find(|(event, _, _, _)| *event == end)
        else {
            continue;
        };
        if *end_owner != root {
            return Err(format!("InvocationEnd {end_event} root does not match frame root"));
        }
        let Some((prefold_event, prefold)) =
            index.prefolds.iter().find(|(event, _)| *event == *pre_fold_event)
        else {
            return Err(format!(
                "InvocationEnd {end_event} references missing PreFold {pre_fold_event}"
            ));
        };
        if *prefold_event < start ||
            *prefold_event >= end ||
            index.immediate_frames
                [usize::try_from(*prefold_event).expect("indexed PreFold event")] !=
                Some(start)
        {
            return Err(format!(
                "PreFold {prefold_event} is outside InvocationEnd {end_event} frame"
            ));
        }
        let root_result = match index.event(prefold.result_event)? {
            ProofPayloadEvent::Result { owner, .. } if *owner == root => {
                index.result(prefold.result_event)?
            }
            _ => {
                return Err(format!(
                    "PreFold {prefold_event} does not reference exact root Result {}",
                    prefold.result_event
                ))
            }
        };
        if index.immediate_frames
            [usize::try_from(prefold.result_event).expect("indexed Result event")] !=
            Some(start)
        {
            return Err(format!(
                "Result {} is outside PreFold {prefold_event} frame",
                prefold.result_event
            ));
        }
        let end_result = match end_value {
            ProofPayloadValue::Exact { terms, summary } => ResultRecord {
                event: *end_event,
                owner: *end_owner,
                terms: terms.clone(),
                summary: summary.clone(),
            },
            _ => return Err(format!("InvocationEnd {end_event} does not carry an exact result")),
        };
        let survivors = index
            .survivors
            .iter()
            .filter(|(event, _)| {
                *event >= start &&
                    *event < end &&
                    index.immediate_frames
                        [usize::try_from(*event).expect("indexed survivor event")] ==
                        Some(start)
            })
            .map(|(_, fold)| {
                let contribution = fold.coefficient.magnitude().to_string();
                (contribution, fold.bound.to_string())
            })
            .collect::<Vec<_>>();
        bound_candidates.push((
            start,
            end,
            BoundProbe {
                root: root_result,
                prefold_terms: prefold.terms.clone(),
                prefold_summary: prefold.summary.clone(),
                end: end_result,
                survivor_contributions: survivors
                    .iter()
                    .map(|(actual, _)| actual.clone())
                    .collect(),
                survivor_bounds: survivors.into_iter().map(|(_, bound)| bound).collect(),
            },
        ));
    }
    let has_positive_survivors =
        bound_candidates.iter().any(|(_, _, probe)| !probe.survivor_contributions.is_empty());
    bound_candidates.retain(|(_, _, probe)| {
        !has_positive_survivors || !probe.survivor_contributions.is_empty()
    });
    bound_candidates
        .sort_by_key(|(start, end, probe)| (probe.survivor_contributions.len(), *start, *end));
    let bound_frame = bound_candidates.into_iter().next().ok_or_else(|| {
        "CP3 bound/fold/result probe cannot map PreFold.result_event and InvocationEnd.pre_fold_event to exact root Results".to_owned()
    })?;
    probes.push(ProbeSelection {
        name: "bound-fold-result",
        event: bound_frame.1,
        owner: bound_frame.2.root.owner,
        score: bound_frame.2.prefold_terms.len() as u64,
        detail: "actual root Result to PreFold to InvocationEnd chain",
        frame_start: Some(bound_frame.0),
        frame_end: Some(bound_frame.1),
        long_key: None,
        operation: None,
        relations: Vec::new(),
        bound: Some(bound_frame.2),
    });
    Ok(probes)
}

struct PayloadIndex {
    events: Vec<ProofPayloadEvent>,
    immediate_frames: Vec<Option<u64>>,
    predecessors: BTreeMap<u64, (ProofPayloadOwner, u32, u64)>,
    results: Vec<ResultRecord>,
    by_event: BTreeMap<u64, ResultRecord>,
    operations: Vec<(ProofPayloadOwner, ProofPayloadRule, u64)>,
    merges: Vec<(u64, crate::operational_noise::simulation::ProofPayloadCoefficientMerge)>,
    relations: Vec<(
        u64,
        ProofPayloadOwner,
        ProofPayloadMonomial,
        num_bigint::BigInt,
        u32,
        u32,
        ProofPayloadRelationRule,
    )>,
    prefolds: Vec<(u64, crate::operational_noise::simulation::ProofPayloadPreFoldPolynomial)>,
    ends: Vec<(u64, ProofPayloadOwner, ProofPayloadValue, u64)>,
    survivors: Vec<(u64, crate::operational_noise::simulation::ProofPayloadSurvivorFold)>,
}

pub(super) fn render(
    statement: &CertificateDocumentV1,
    proof: &OperationalProofPayload,
) -> Result<Vec<super::super::TallSecurity0GeneratedFile>, String> {
    let (mut report, _selections, _long_monomial) = measure(proof)?;
    let index = PayloadIndex::new(proof)?;
    let probes = build_probes(statement, proof, &index)?;
    report.probes = probes.iter().map(ProbeStat::from_probe).collect();
    let report_bytes = serde_json::to_vec(&report)
        .map_err(|error| format!("semantic probe statistics encoding failed: {error}"))?;
    let mut files = vec![generated_file(
        "SemanticProbeStatistics.json",
        String::from_utf8(report_bytes).expect("JSON is UTF-8"),
    )];

    let specs = [
        ("Semantic/Semantic000.lean", "Semantic000", "long-monomial-merge"),
        ("Semantic/Semantic001.lean", "Semantic001", "outer-result"),
        ("Semantic/Semantic002.lean", "Semantic002", "add-chain"),
        ("Semantic/Semantic003.lean", "Semantic003", "product-tensor"),
        ("Semantic/Semantic004.lean", "Semantic004", "relation"),
        ("Semantic/Semantic005.lean", "Semantic005", "bound-fold-result"),
    ];
    for (path, module, probe) in specs {
        files.push(generated_file(path, render_probe(module, probe, &probes)));
    }
    let mut index = String::new();
    for (_, module, _) in specs {
        writeln!(index, "import {NAMESPACE}.Semantic.{module}").expect("String write");
    }
    files.push(generated_file("Semantic/Semantic.lean", index));
    Ok(files)
}

fn measure(
    proof: &OperationalProofPayload,
) -> Result<(StatisticsReport, Vec<Selection>, Option<ProofPayloadMonomial>), String> {
    let mut stack = Vec::<Frame>::new();
    let mut max_terms: Option<NodeStat> = None;
    let mut max_factors: Option<NodeStat> = None;
    let mut long_merge: Option<(u64, ProofPayloadOwner, ProofPayloadMonomial)> = None;
    let mut outer_result: Option<FrameSelection> = None;
    let mut add_chain: Option<Selection> = None;
    let mut product_tensor: Option<Selection> = None;
    let mut product_owners = BTreeSet::new();
    let mut relation: Option<Selection> = None;
    let mut relation_score = 0_u64;
    let mut bound_chain: Option<FrameSelection> = None;
    let mut exact_result_nodes = 0_u64;
    let mut selections = Vec::new();

    for (index, _event) in proof.events.iter().enumerate() {
        let event = u64::try_from(index).map_err(|_| "semantic event index overflow")?;
        match event_at(proof, index) {
            ProofPayloadEvent::InvocationStart { root } => stack.push(Frame {
                root: *root,
                start: event,
                merge_count: 0,
                add_chain: 0,
                max_add_chain: 0,
                max_add_chain_event: None,
                has_prefold: false,
            }),
            ProofPayloadEvent::Result { owner, value } => {
                if let ProofPayloadValue::Exact { terms, .. } = value {
                    exact_result_nodes += 1;
                    let node = node_stat(event, *owner, terms);
                    update_max(&mut max_terms, &node, |item| item.term_count);
                    update_max(&mut max_factors, &node, |item| item.max_monomial_factor_length);
                    if product_owners.contains(owner) &&
                        product_tensor.as_ref().is_none_or(|selection| {
                            node.term_count > selection.score ||
                                (node.term_count == selection.score && event < selection.event)
                        })
                    {
                        product_tensor = Some(Selection {
                            event,
                            owner: *owner,
                            detail: "largest reached Product/Tensor result",
                            score: node.term_count,
                            frame_start: stack.last().map(|frame| frame.start),
                            frame_end: None,
                        });
                    }
                }
                if let Some(frame) = stack.last_mut() {
                    frame.add_chain = 0;
                }
            }
            ProofPayloadEvent::CoefficientMerge(merge) => {
                if let Some(frame) = stack.last_mut() {
                    frame.merge_count += 1;
                    match merge.source {
                        crate::operational_noise::simulation::ProofPayloadCoefficientMergeSource::Operator { .. } => {
                            frame.add_chain += 1;
                            if frame.add_chain > frame.max_add_chain {
                                frame.max_add_chain = frame.add_chain;
                                frame.max_add_chain_event = Some(event);
                            }
                        }
                        crate::operational_noise::simulation::ProofPayloadCoefficientMergeSource::Relation { .. } => {
                            frame.add_chain = 0;
                        }
                    }
                }
                let factor_len =
                    merge.output.central_factors.len() + merge.output.ordered_factors.len();
                let replace = long_merge.as_ref().is_none_or(|(old_event, _, old)| {
                    factor_len > old.central_factors.len() + old.ordered_factors.len() ||
                        (factor_len == old.central_factors.len() + old.ordered_factors.len() &&
                            event < *old_event)
                });
                if replace {
                    long_merge = Some((event, merge.owner, merge.output.clone()));
                }
            }
            ProofPayloadEvent::BoundTransfer { owner, rule } => {
                if matches!(
                    rule,
                    ProofPayloadRule::Product { .. } | ProofPayloadRule::Tensor { .. }
                ) {
                    product_owners.insert(*owner);
                }
            }
            ProofPayloadEvent::AppliedRelation { owner, .. } => {
                let score = stack.last().map_or(0, |frame| frame.merge_count);
                if relation.is_none() || score > relation_score {
                    relation_score = score;
                    relation = Some(Selection {
                        event,
                        owner: *owner,
                        detail: "maximum reached relation application",
                        score,
                        frame_start: stack.last().map(|frame| frame.start),
                        frame_end: None,
                    });
                }
            }
            ProofPayloadEvent::PreFoldPolynomial(value) => {
                let _ = value;
                if let Some(frame) = stack.last_mut() {
                    frame.has_prefold = true;
                }
            }
            ProofPayloadEvent::InvocationEnd { root, .. } => {
                let frame = stack
                    .pop()
                    .ok_or_else(|| format!("semantic frame underflow at event {event}"))?;
                if frame.root != *root {
                    return Err(format!("semantic frame root mismatch at event {event}"));
                }
                let end = event;
                if outer_result.as_ref().is_none_or(|old| frame.merge_count > old.merge_count) {
                    outer_result = Some(FrameSelection {
                        start: frame.start,
                        end,
                        root: frame.root,
                        merge_count: frame.merge_count,
                        detail: "maximum merge-count invocation frame",
                    });
                }
                if let Some(chain_event) = frame.max_add_chain_event {
                    if add_chain.as_ref().is_none_or(|selection| {
                        frame.max_add_chain > selection.score ||
                            (frame.max_add_chain == selection.score &&
                                chain_event < selection.event)
                    }) {
                        add_chain = Some(Selection {
                            event: chain_event,
                            owner: frame.root,
                            detail: "maximum consecutive operator-merge chain (chain length is in statistics)",
                            score: frame.max_add_chain,
                            frame_start: Some(frame.start),
                            frame_end: Some(end),
                        });
                    }
                }
                if frame.has_prefold && bound_chain.as_ref().is_none() {
                    bound_chain = Some(FrameSelection {
                        start: frame.start,
                        end,
                        root: frame.root,
                        merge_count: frame.merge_count,
                        detail: "first reached pre-fold to invocation-end chain",
                    });
                }
            }
            _ => {}
        }
    }
    if !stack.is_empty() {
        return Err("semantic statistics found an active invocation frame".to_owned());
    }
    if let Some((event, owner, monomial)) = long_merge.as_ref() {
        selections.push(Selection {
            event: *event,
            owner: *owner,
            detail: "maximum monomial-factor coefficient merge",
            score: monomial.central_factors.len() as u64 + monomial.ordered_factors.len() as u64,
            frame_start: None,
            frame_end: None,
        });
        let _ = monomial;
    }
    if let Some(frame) = outer_result {
        selections.push(Selection {
            event: frame.end,
            owner: frame.root,
            detail: frame.detail,
            score: frame.merge_count,
            frame_start: Some(frame.start),
            frame_end: Some(frame.end),
        });
    }
    if let Some(selection) = add_chain {
        selections.push(selection);
    }
    if let Some(selection) = product_tensor {
        selections.push(selection);
    }
    if let Some(selection) = relation {
        selections.push(selection);
    }
    if let Some(frame) = bound_chain {
        selections.push(Selection {
            event: frame.end,
            owner: frame.root,
            detail: frame.detail,
            score: frame.merge_count,
            frame_start: Some(frame.start),
            frame_end: Some(frame.end),
        });
    }
    let probes = selections
        .iter()
        .map(|selection| ProbeStat {
            name: probe_name(selection.detail),
            event: Some(selection.event),
            owner: Some(owner_dto(selection.owner)),
            frame_start: selection.frame_start,
            frame_end: selection.frame_end,
            score: selection.score,
            detail: selection.detail,
        })
        .collect();
    Ok((
        StatisticsReport {
            schema_id: SCHEMA_ID,
            schema_version: SCHEMA_VERSION,
            event_count: proof.events.len() as u64,
            exact_result_nodes,
            max_term_count: max_terms,
            max_monomial_factor_length: max_factors,
            probes,
        },
        selections,
        long_merge.map(|(_, _, monomial)| monomial),
    ))
}

fn event_at<'a>(proof: &'a OperationalProofPayload, index: usize) -> &'a ProofPayloadEvent {
    &proof.events[index]
}

fn update_max(slot: &mut Option<NodeStat>, node: &NodeStat, key: impl Fn(&NodeStat) -> u64) {
    if slot
        .as_ref()
        .is_none_or(|old| key(node) > key(old) || (key(node) == key(old) && node.event < old.event))
    {
        *slot = Some(node.clone());
    }
}

fn node_stat(event: u64, owner: ProofPayloadOwner, terms: &[ProofPayloadTerm]) -> NodeStat {
    let (central, ordered) = terms.iter().fold((0, 0), |(central, ordered), term| {
        (
            central.max(term.monomial.central_factors.len()),
            ordered.max(term.monomial.ordered_factors.len()),
        )
    });
    NodeStat {
        event,
        owner: owner_dto(owner),
        term_count: terms.len() as u64,
        max_central_factor_length: central as u64,
        max_ordered_factor_length: ordered as u64,
        max_monomial_factor_length: (central + ordered) as u64,
    }
}

fn owner_dto(owner: ProofPayloadOwner) -> OwnerDto {
    let scope = match owner.scope {
        crate::operational_noise::simulation::ProofPayloadScope::Closed { root_expression_row } => {
            ScopeDto::Closed { root_expression: root_expression_row }
        }
        crate::operational_noise::simulation::ProofPayloadScope::Program { program_row } => {
            ScopeDto::Program { program: program_row }
        }
    };
    OwnerDto { scope, expression: owner.expression_row }
}

fn owner_text(owner: ProofPayloadOwner) -> String {
    let scope = match owner.scope {
        crate::operational_noise::simulation::ProofPayloadScope::Closed { root_expression_row } => {
            format!(".closed ⟨{root_expression_row}⟩")
        }
        crate::operational_noise::simulation::ProofPayloadScope::Program { program_row } => {
            format!(".program ⟨{program_row}⟩")
        }
    };
    format!("⟨{scope}, ⟨{}⟩⟩", owner.expression_row)
}

fn monomial_text(monomial: &ProofPayloadMonomial) -> String {
    let central = monomial
        .central_factors
        .iter()
        .map(|owner| owner_text(*owner))
        .collect::<Vec<_>>()
        .join(", ");
    let ordered = monomial
        .ordered_factors
        .iter()
        .map(|owner| owner_text(*owner))
        .collect::<Vec<_>>()
        .join(", ");
    format!("⟨[{central}], [{ordered}]⟩")
}

fn probe_name(detail: &str) -> &'static str {
    if detail.contains("monomial") {
        "long-monomial-merge"
    } else if detail.contains("merge-count") {
        "outer-result"
    } else if detail.contains("operator-merge") {
        "add-chain"
    } else if detail.contains("Product") {
        "product-tensor"
    } else if detail.contains("relation") {
        "relation"
    } else {
        "bound-fold-result"
    }
}

fn term_text(term: &ProofPayloadTerm) -> String {
    let central = term
        .monomial
        .central_factors
        .iter()
        .map(|owner| owner_text(*owner))
        .collect::<Vec<_>>()
        .join(", ");
    let ordered = term
        .monomial
        .ordered_factors
        .iter()
        .map(|owner| owner_text(*owner))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{{ coefficient := ({}), key := {{ centralFactors := [{}], orderedFactors := [{}] }} }}",
        term.coefficient, central, ordered
    )
}

fn terms_text(terms: &[ProofPayloadTerm]) -> String {
    format!("[{}]", terms.iter().map(term_text).collect::<Vec<_>>().join(", "))
}

fn summary_text(summary: &crate::operational_noise::normal_form::BoundedSummary) -> String {
    match summary.coefficient_bound() {
        crate::operational_noise::facts::NumericContract::Known(
            crate::operational_noise::facts::CoefficientBound::ExactZero,
        ) => ".exactZero".to_owned(),
        crate::operational_noise::facts::NumericContract::Known(
            crate::operational_noise::facts::CoefficientBound::Finite(value),
        ) => format!("(.finite {})", value.maximum_absolute_coefficient),
        crate::operational_noise::facts::NumericContract::Known(
            crate::operational_noise::facts::CoefficientBound::Large,
        ) => ".large".to_owned(),
        crate::operational_noise::facts::NumericContract::Missing => ".missing".to_owned(),
    }
}

fn summary_bound_nat_text(
    summary: &crate::operational_noise::normal_form::BoundedSummary,
) -> String {
    match summary.coefficient_bound() {
        crate::operational_noise::facts::NumericContract::Known(
            crate::operational_noise::facts::CoefficientBound::ExactZero,
        ) => "0".to_owned(),
        crate::operational_noise::facts::NumericContract::Known(
            crate::operational_noise::facts::CoefficientBound::Finite(value),
        ) => value.maximum_absolute_coefficient.to_string(),
        crate::operational_noise::facts::NumericContract::Known(
            crate::operational_noise::facts::CoefficientBound::Large,
        ) |
        crate::operational_noise::facts::NumericContract::Missing => "0".to_owned(),
    }
}

fn render_operation(source: &mut String, operation: &OperationProbe) {
    if operation.kind == OperationKind::Direct {
        writeln!(
            source,
            "def output : Polynomial Owner := {}",
            terms_text(&operation.output.terms)
        )
        .expect("String write");
        source.push_str("\ntheorem resultAgreement : CanonicalAgreement output output := by\n  decide +kernel\n");
        source.push_str("\ntheorem resultSound (env : Env Owner) :\n    evalPolynomial env output = evalPolynomial env output := by\n  rfl\n\n");
        return;
    }
    let left = &operation.inputs[0];
    let right = &operation.inputs[1];
    writeln!(source, "def left : Polynomial Owner := {}", terms_text(&left.terms))
        .expect("String write");
    writeln!(source, "def right : Polynomial Owner := {}", terms_text(&right.terms))
        .expect("String write");
    writeln!(source, "def output : Polynomial Owner := {}", terms_text(&operation.output.terms))
        .expect("String write");
    writeln!(source, "def selectedRawWork : Nat := {}", operation.raw_work).expect("String write");
    if let Some(chain_tail) = operation.chain_tail {
        writeln!(source, "def selectedChainTailEvent : Nat := {chain_tail}").expect("String write");
    }
    match operation.kind {
        OperationKind::Direct => unreachable!("direct operation handled above"),
        OperationKind::Add => {
            source.push_str("\ntheorem resultAgreement : CanonicalAgreement output (add left right) := by\n  decide +kernel\n");
            source.push_str("\ntheorem resultSound (env : Env Owner) :\n    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by\n  exact addCanonicalResultSound env left right output resultAgreement\n\n");
        }
        OperationKind::Subtract => {
            source.push_str("\ntheorem resultAgreement : CanonicalAgreement output (subtract left right) := by\n  decide +kernel\n");
            source.push_str("\ntheorem resultSound (env : Env Owner) :\n    evalPolynomial env output = evalPolynomial env left - evalPolynomial env right := by\n  exact subCanonicalResultSound env left right output resultAgreement\n\n");
        }
        OperationKind::Multiply | OperationKind::Tensor => {
            writeln!(
                source,
                "def leftScalar : Bool := {}",
                if operation.scalar_left { "true" } else { "false" }
            )
            .expect("String write");
            writeln!(
                source,
                "def rightScalar : Bool := {}",
                if operation.scalar_right { "true" } else { "false" }
            )
            .expect("String write");
            source.push_str("\ntheorem resultAgreement : CanonicalAgreement output (productPoly left right leftScalar rightScalar) := by\n  decide +kernel\n");
            source.push_str("\ntheorem resultSound (env : Env Owner) :\n    evalPolynomial env output = evalPolynomial env left * evalPolynomial env right := by\n  exact productCanonicalResultSound env left right output leftScalar rightScalar resultAgreement\n\n");
        }
    }
}

fn render_relation(source: &mut String, relation: &RelationProbe) {
    source.push_str("open EventReplay\n");
    writeln!(
        source,
        "def accumulator : Polynomial Owner := {}",
        terms_text(&relation.accumulator.terms)
    )
    .expect("String write");
    writeln!(source, "def sourceKey : MonomialKey Owner := {}", monomial_text(&relation.source))
        .expect("String write");
    writeln!(source, "def lhsKey : MonomialKey Owner := {}", monomial_text(&relation.lhs))
        .expect("String write");
    writeln!(source, "def relationRhs : Polynomial Owner := {}", terms_text(&relation.rhs.terms))
        .expect("String write");
    writeln!(
        source,
        "def relationOutput : Polynomial Owner := {}",
        terms_text(&relation.output.terms)
    )
    .expect("String write");
    let relation_expected = relation_expected_terms(relation);
    writeln!(
        source,
        "def relationExpected : Polynomial Owner := {}",
        terms_text(&relation_expected)
    )
    .expect("String write");
    writeln!(source, "def relationContext0 : MonomialContext Owner := relationContext sourceKey sourceKey.centralFactors {} {}", relation.start, relation.end).expect("String write");
    source
        .push_str("\ntheorem relationShape : relationPoly accumulator sourceKey relationContext0 ");
    writeln!(source, "({}) relationRhs = relationExpected := by rfl\n", relation.outer)
        .expect("String write");
    source.push_str("theorem relationAgreement : CanonicalAgreement relationOutput (relationPoly accumulator sourceKey relationContext0 ");
    writeln!(source, "({}) relationRhs) := by decide +kernel\n", relation.outer)
        .expect("String write");
    source.push_str("theorem relationSound (env : Env Owner)\n    (baseRelation : evalMonomial env lhsKey % Int.ofNat 257 =\n      evalPolynomial env relationRhs % Int.ofNat 257) :\n    evalPolynomial env relationOutput % Int.ofNat 257 =\n      evalPolynomial env accumulator % Int.ofNat 257 := by\n  exact relationCanonicalResultSound 257 env accumulator sourceKey lhsKey\n    sourceKey.centralFactors ");
    writeln!(source, "{} {} ({}) relationRhs relationOutput\n    (by decide +kernel) baseRelation relationAgreement\n", relation.start, relation.end, relation.outer).expect("String write");
}

fn relation_expected_terms(relation: &RelationProbe) -> Vec<ProofPayloadTerm> {
    let mut terms = relation.accumulator.terms.clone();
    terms.push(ProofPayloadTerm {
        coefficient: -&relation.outer,
        monomial: relation.source.clone(),
    });
    let prefix = relation.source.ordered_factors[..relation.start as usize].to_vec();
    let suffix = relation.source.ordered_factors[relation.end as usize..].to_vec();
    terms.extend(relation.rhs.terms.iter().map(|term| {
        ProofPayloadTerm {
            coefficient: &relation.outer * &term.coefficient,
            monomial: ProofPayloadMonomial {
                central_factors: relation
                    .source
                    .central_factors
                    .iter()
                    .chain(term.monomial.central_factors.iter())
                    .copied()
                    .collect(),
                ordered_factors: prefix
                    .iter()
                    .chain(term.monomial.ordered_factors.iter())
                    .chain(suffix.iter())
                    .copied()
                    .collect(),
            },
        }
    }));
    terms
}

fn render_bound(source: &mut String, bound: &BoundProbe) {
    writeln!(source, "def rootTerms : Polynomial Owner := {}", terms_text(&bound.root.terms))
        .expect("String write");
    writeln!(source, "def prefoldTerms : Polynomial Owner := {}", terms_text(&bound.prefold_terms))
        .expect("String write");
    writeln!(source, "def endTerms : Polynomial Owner := {}", terms_text(&bound.end.terms))
        .expect("String write");
    writeln!(source, "def rootSummary : Bound := {}", summary_text(&bound.root_summary()))
        .expect("String write");
    writeln!(source, "def prefoldSummary : Bound := {}", summary_text(&bound.prefold_summary))
        .expect("String write");
    writeln!(source, "def endSummary : Bound := {}", summary_text(&bound.end_summary()))
        .expect("String write");
    writeln!(source, "def rootBound : Nat := {}", summary_bound_nat_text(&bound.root_summary()))
        .expect("String write");
    writeln!(
        source,
        "def prefoldBound : Nat := {}",
        summary_bound_nat_text(&bound.prefold_summary)
    )
    .expect("String write");
    writeln!(
        source,
        "def survivorContributions : List Nat := [{}]",
        bound.survivor_contributions.join(", ")
    )
    .expect("String write");
    writeln!(source, "def survivorBounds : List Nat := [{}]", bound.survivor_bounds.join(", "))
        .expect("String write");
    let mut survivors_proof = String::from("by\n");
    if bound.survivor_contributions.is_empty() {
        survivors_proof.push_str("  exact List.Forall₂.nil\n");
    } else {
        for depth in 0..bound.survivor_contributions.len() {
            let indent = "  ".repeat(depth + 1);
            survivors_proof.push_str(&format!("{indent}constructor\n"));
            survivors_proof.push_str(&format!("{indent}· decide +kernel\n"));
            survivors_proof.push_str(&format!("{indent}·\n"));
        }
        let indent = "  ".repeat(bound.survivor_contributions.len() + 1);
        survivors_proof.push_str(&format!("{indent}exact List.Forall₂.nil\n"));
    }
    source.push_str("\ntheorem prefoldResult : prefoldTerms = rootTerms := by rfl\n\ntheorem prefoldBoundSound : rootBound ≤ prefoldBound := by decide +kernel\n\n");
    writeln!(source, "theorem survivorBoundsSound : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributions survivorBounds :=\n{survivors_proof}").expect("String write");
    source.push_str("\ntheorem prefoldSound :\n  preFoldBound rootBound prefoldBound survivorContributions survivorBounds := by\n  exact (preFoldSound rootTerms prefoldTerms prefoldResult prefoldBoundSound survivorBoundsSound).2\n\ntheorem endResult : endTerms = prefoldTerms := by rfl\n\ntheorem endSummaryResult : endSummary = prefoldSummary := by rfl\n\ntheorem endSound :\n  endTerms = prefoldTerms ∧ endSummary = prefoldSummary := by\n  exact ⟨endResult, endSummaryResult⟩\n\n");
}

impl BoundProbe {
    fn root_summary(&self) -> crate::operational_noise::normal_form::BoundedSummary {
        self.root.summary.clone()
    }
    fn end_summary(&self) -> crate::operational_noise::normal_form::BoundedSummary {
        self.end.summary.clone()
    }
}

fn render_probe(module: &str, probe: &str, probes: &[ProbeSelection]) -> String {
    let mut source = format!(
        "import Mxx.Certificate.OperationalNoise.TallSemantics\n\nset_option autoImplicit false\nset_option relaxedAutoImplicit false\n\nnamespace {NAMESPACE}.Semantic.{module}\n\nopen Mxx.Certificate.OperationalNoise\nopen TallSecurity0ABI\nopen TallSemantics\n\n"
    );
    let selected = probes.iter().find(|selection| selection.name == probe);
    if let Some(selection) = selected {
        writeln!(source, "def selectedEvent : Nat := {}", selection.event).expect("String write");
        writeln!(source, "def selectedOwner : Owner := {}", owner_text(selection.owner))
            .expect("String write");
        writeln!(source, "def selectedDetail : String := {:?}", selection.detail)
            .expect("String write");
        writeln!(source, "def selectedScore : Nat := {}", selection.score).expect("String write");
    }
    match probe {
        "long-monomial-merge" => {
            if let Some(selection) = selected {
                let key = selection.long_key.as_ref().expect("long key");
                writeln!(source, "def selectedKey : MonomialKey Owner := {}", monomial_text(key))
                    .expect("String write");
                source.push_str("theorem kernelLongMonomial (env : Env Owner) :\n  evalMonomial env selectedKey = evalMonomial env selectedKey := by\n  exact evalMonomial_of_key env (left := selectedKey) (right := selectedKey) (List.Perm.refl _) rfl\n\n");
            }
        }
        "outer-result" | "add-chain" | "product-tensor" => {
            if let Some(selection) = selected {
                render_operation(
                    &mut source,
                    selection.operation.as_ref().expect("operation probe"),
                );
                if probe == "add-chain" {
                    source.push_str("namespace AddChainTail\n\n");
                    source.push_str("theorem chainTailResultSound (env : Env Owner) :\n    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by\n  exact resultSound env\n\nend AddChainTail\n");
                }
            }
        }
        "relation" => {
            if let Some(selection) = selected {
                for (ordinal, relation) in selection.relations.iter().enumerate() {
                    writeln!(source, "namespace Relation{ordinal}").expect("String write");
                    render_relation(&mut source, relation);
                    writeln!(source, "end Relation{ordinal}").expect("String write");
                }
            }
        }
        "bound-fold-result" => {
            if let Some(selection) = selected {
                render_bound(&mut source, selection.bound.as_ref().expect("bound probe"));
            }
        }
        _ => {}
    }
    source.push_str(&format!("end {NAMESPACE}.Semantic.{module}\n"));
    source
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::{
        normal_form::BoundedSummary,
        simulation::{
            ProofPayloadCoefficientMerge, ProofPayloadCoefficientMergeSource, ProofPayloadEvent,
            ProofPayloadPreFoldPolynomial, ProofPayloadRelationRule, ProofPayloadRule,
            ProofPayloadScope, ProofPayloadTermRef, ProofPayloadValue, ProofPayloadValueRef,
        },
    };
    use num_bigint::{BigInt, BigUint};

    fn owner(row: u64) -> ProofPayloadOwner {
        ProofPayloadOwner {
            scope: ProofPayloadScope::Closed { root_expression_row: row },
            expression_row: row,
        }
    }

    #[test]
    fn semantic_statistics_are_deterministic_and_keep_owner_identity() {
        let root = owner(7);
        let term = ProofPayloadTerm {
            monomial: ProofPayloadMonomial {
                central_factors: vec![root],
                ordered_factors: vec![root, root],
            },
            coefficient: BigInt::from(1),
        };
        let proof = OperationalProofPayload {
            events: vec![
                ProofPayloadEvent::InvocationStart { root },
                ProofPayloadEvent::Result {
                    owner: root,
                    value: ProofPayloadValue::Exact {
                        terms: vec![term.clone()],
                        summary: BoundedSummary::zero(),
                    },
                },
                ProofPayloadEvent::CoefficientMerge(ProofPayloadCoefficientMerge {
                    owner: root,
                    source: ProofPayloadCoefficientMergeSource::Operator {
                        inputs: [
                            ProofPayloadTermRef { value_event: 1, term_ordinal: 0 },
                            ProofPayloadTermRef { value_event: 1, term_ordinal: 0 },
                        ],
                    },
                    output: term.monomial.clone(),
                    signed_contribution: BigInt::from(1),
                }),
                ProofPayloadEvent::PreFoldPolynomial(ProofPayloadPreFoldPolynomial {
                    result_event: 1,
                    terms: vec![term.clone()],
                    summary: BoundedSummary::zero(),
                    summary_evidence: None,
                }),
                ProofPayloadEvent::InvocationEnd {
                    root,
                    result: ProofPayloadValue::Exact {
                        terms: vec![term.clone()],
                        summary: BoundedSummary::zero(),
                    },
                    pre_fold_event: 3,
                },
            ],
        };
        let (first, _, _) = measure(&proof).expect("first semantic measurement");
        let (second, _, _) = measure(&proof).expect("second semantic measurement");
        assert_eq!(
            serde_json::to_vec(&first).expect("first JSON"),
            serde_json::to_vec(&second).expect("second JSON")
        );
        let max = first.max_monomial_factor_length.expect("factor maximum");
        assert_eq!(max.event, 1);
        assert_eq!(max.term_count, 1);
        assert_eq!(max.max_monomial_factor_length, 3);
        assert_eq!(max.owner.scope, ScopeDto::Closed { root_expression: 7 });
        let source = render_probe(
            "Semantic000",
            "long-monomial-merge",
            &[ProbeSelection {
                name: "long-monomial-merge",
                event: 2,
                owner: root,
                score: 2,
                detail: "actual coefficient merge output monomial",
                frame_start: None,
                frame_end: None,
                long_key: Some(ProofPayloadMonomial {
                    central_factors: vec![root],
                    ordered_factors: vec![root],
                }),
                operation: None,
                relations: Vec::new(),
                bound: None,
            }],
        );
        assert!(source.contains("evalMonomial_of_key"));
        assert!(source.contains(".closed ⟨7⟩"));
        assert!(!source.contains("emptyPolynomial"));
        assert!(!source.contains("CoefficientAgreement"));
        assert!(!source.contains("rcases"));
        assert!(!source.contains("native_decide"));
        assert!(!source.contains(" sorry"));

        let operation = OperationProbe {
            kind: OperationKind::Add,
            inputs: vec![
                ResultRecord {
                    event: 2,
                    owner: root,
                    terms: vec![term.clone()],
                    summary: BoundedSummary::zero(),
                },
                ResultRecord {
                    event: 3,
                    owner: root,
                    terms: vec![term.clone()],
                    summary: BoundedSummary::zero(),
                },
            ],
            output: ResultRecord {
                event: 4,
                owner: root,
                terms: vec![ProofPayloadTerm {
                    monomial: term.monomial.clone(),
                    coefficient: BigInt::from(2),
                }],
                summary: BoundedSummary::zero(),
            },
            scalar_left: false,
            scalar_right: false,
            raw_work: 1,
            chain_tail: Some(2),
        };
        let mut operation_source = String::new();
        render_operation(&mut operation_source, &operation);
        assert!(operation_source.contains("CanonicalAgreement"));
        assert!(operation_source.contains("addCanonicalResultSound"));
        assert!(operation_source.contains("selectedChainTailEvent"));
        assert!(!operation_source.contains("CoefficientAgreement"));
        assert!(!operation_source.contains("expected :"));

        let bound = BoundProbe {
            root: ResultRecord {
                event: 1,
                owner: root,
                terms: vec![],
                summary: BoundedSummary::zero(),
            },
            prefold_terms: vec![],
            prefold_summary: BoundedSummary::zero(),
            end: ResultRecord {
                event: 3,
                owner: root,
                terms: vec![],
                summary: BoundedSummary::zero(),
            },
            survivor_contributions: vec!["1".to_owned(), "2".to_owned()],
            survivor_bounds: vec!["1".to_owned(), "3".to_owned()],
        };
        let mut bound_source = String::new();
        render_bound(&mut bound_source, &bound);
        assert!(bound_source.contains("theorem survivorBoundsSound"));
        assert!(bound_source.contains("List.Forall₂"));
        assert!(bound_source.contains("· decide +kernel"));
    }

    #[test]
    fn relation_snapshot_uses_working_merges_after_stale_result() {
        let root = owner(7);
        let source = ProofPayloadMonomial { central_factors: vec![], ordered_factors: vec![root] };
        let carried = ProofPayloadMonomial { central_factors: vec![root], ordered_factors: vec![] };
        let replacement =
            ProofPayloadMonomial { central_factors: vec![], ordered_factors: vec![root, root] };
        let later = ProofPayloadMonomial {
            central_factors: vec![],
            ordered_factors: vec![root, root, root],
        };
        let term = |monomial: ProofPayloadMonomial, coefficient| ProofPayloadTerm {
            monomial,
            coefficient: BigInt::from(coefficient),
        };
        let proof = OperationalProofPayload {
            events: vec![
                ProofPayloadEvent::InvocationStart { root },
                ProofPayloadEvent::Result {
                    owner: root,
                    value: ProofPayloadValue::Exact {
                        terms: vec![term(source.clone(), 1)],
                        summary: BoundedSummary::zero(),
                    },
                },
                ProofPayloadEvent::CoefficientMerge(ProofPayloadCoefficientMerge {
                    owner: root,
                    source: ProofPayloadCoefficientMergeSource::Operator {
                        inputs: [
                            ProofPayloadTermRef { value_event: 1, term_ordinal: 0 },
                            ProofPayloadTermRef { value_event: 1, term_ordinal: 0 },
                        ],
                    },
                    output: carried.clone(),
                    signed_contribution: BigInt::from(1),
                }),
                ProofPayloadEvent::AppliedRelation {
                    owner: root,
                    source_monomial: source.clone(),
                    outer_coefficient: BigInt::from(1),
                    ordered_start: 0,
                    ordered_end_exclusive: 1,
                    rule: ProofPayloadRelationRule::Gadget {
                        gadget: root,
                        decomposition: root,
                        input: 1,
                        input_result: 1,
                    },
                },
                ProofPayloadEvent::CoefficientMerge(ProofPayloadCoefficientMerge {
                    owner: root,
                    source: ProofPayloadCoefficientMergeSource::Relation {
                        application: 3,
                        source_term_ordinal: 0,
                    },
                    output: replacement.clone(),
                    signed_contribution: BigInt::from(1),
                }),
                ProofPayloadEvent::CoefficientMerge(ProofPayloadCoefficientMerge {
                    owner: root,
                    source: ProofPayloadCoefficientMergeSource::Operator {
                        inputs: [
                            ProofPayloadTermRef { value_event: 1, term_ordinal: 0 },
                            ProofPayloadTermRef { value_event: 1, term_ordinal: 0 },
                        ],
                    },
                    output: later.clone(),
                    signed_contribution: BigInt::from(1),
                }),
                ProofPayloadEvent::Result {
                    owner: root,
                    value: ProofPayloadValue::Exact {
                        terms: vec![
                            term(carried.clone(), 1),
                            term(replacement.clone(), 1),
                            term(later.clone(), 1),
                        ],
                        summary: BoundedSummary::zero(),
                    },
                },
                ProofPayloadEvent::InvocationEnd {
                    root,
                    result: ProofPayloadValue::Exact {
                        terms: vec![
                            term(
                                ProofPayloadMonomial {
                                    central_factors: vec![],
                                    ordered_factors: vec![root, root],
                                },
                                1,
                            ),
                            term(carried.clone(), 1),
                        ],
                        summary: BoundedSummary::zero(),
                    },
                    pre_fold_event: 6,
                },
            ],
        };
        let index = PayloadIndex::new(&proof).expect("payload index");
        let probes = relation_candidates(&index, &[(0, 7, root, 3)]).expect("relation probe");
        assert_eq!(probes.len(), 1);
        assert_eq!(probes[0].output.event, 4);
        assert!(!probes[0].output.terms.iter().any(|term| term.monomial == later));
        assert!(
            probes[0]
                .accumulator
                .terms
                .iter()
                .any(|term| term.monomial.ordered_factors == vec![root])
        );
        assert!(probes[0].accumulator.terms.iter().any(|term| term.monomial == carried));
    }

    #[test]
    fn typed_add_result_without_collision_merges_uses_predecessor_rule_refs() {
        let root = owner(7);
        let left = ProofPayloadMonomial { central_factors: vec![root], ordered_factors: vec![] };
        let right = ProofPayloadMonomial { central_factors: vec![], ordered_factors: vec![root] };
        let term = |monomial: ProofPayloadMonomial| ProofPayloadTerm {
            monomial,
            coefficient: BigInt::from(1),
        };
        let proof = OperationalProofPayload {
            events: vec![
                ProofPayloadEvent::InvocationStart { root },
                ProofPayloadEvent::Result {
                    owner: root,
                    value: ProofPayloadValue::Exact {
                        terms: vec![term(left.clone())],
                        summary: BoundedSummary::zero(),
                    },
                },
                ProofPayloadEvent::Predecessor {
                    consumer: root,
                    input_position: 0,
                    predecessor: 0,
                    source_result: 1,
                },
                ProofPayloadEvent::Result {
                    owner: root,
                    value: ProofPayloadValue::Exact {
                        terms: vec![term(right.clone())],
                        summary: BoundedSummary::zero(),
                    },
                },
                ProofPayloadEvent::Predecessor {
                    consumer: root,
                    input_position: 1,
                    predecessor: 0,
                    source_result: 3,
                },
                ProofPayloadEvent::BoundTransfer {
                    owner: root,
                    rule: ProofPayloadRule::Sum {
                        inputs: vec![
                            ProofPayloadValueRef::Predecessor {
                                binding_event: 2,
                                input_position: 0,
                                projection:
                                    crate::operational_noise::g0::BoundProjection::Coefficient,
                            },
                            ProofPayloadValueRef::Predecessor {
                                binding_event: 4,
                                input_position: 1,
                                projection:
                                    crate::operational_noise::g0::BoundProjection::Coefficient,
                            },
                        ],
                    },
                },
                ProofPayloadEvent::Result {
                    owner: root,
                    value: ProofPayloadValue::Exact {
                        terms: vec![term(left), term(right)],
                        summary: BoundedSummary::zero(),
                    },
                },
                ProofPayloadEvent::InvocationEnd {
                    root,
                    result: ProofPayloadValue::Exact {
                        terms: vec![],
                        summary: BoundedSummary::zero(),
                    },
                    pre_fold_event: 0,
                },
            ],
        };
        let index = PayloadIndex::new(&proof).expect("payload index");
        let result = index.result(6).expect("exact Add result");
        assert!(
            typed_operator_eligibility(&index, &result, OperationKind::Add)
                .expect("typed Add classification")
        );
        let (_, refs, _) = typed_operation_rule(&index, &result, OperationKind::Add)
            .expect("typed Add rule")
            .expect("eligible Add rule");
        assert_eq!(index.value_ref(root, &refs[0]).expect("left ref").event, 1);
        assert_eq!(index.value_ref(root, &refs[1]).expect("right ref").event, 3);
        assert!(index.merges.is_empty());

        let mut folded_proof = proof.clone();
        if let ProofPayloadEvent::Result {
            value: ProofPayloadValue::Exact { summary, .. }, ..
        } = &mut folded_proof.events[6]
        {
            *summary = BoundedSummary::finite(BigUint::from(91_u32).into());
        }
        let folded_index = PayloadIndex::new(&folded_proof).expect("folded payload index");
        let folded_result = folded_index.result(6).expect("folded Add result");
        assert!(
            !typed_operator_eligibility(&folded_index, &folded_result, OperationKind::Add)
                .expect("folded Add classification")
        );
        let transfer_error = match folded_index.value_ref(root, &ProofPayloadValueRef::Transfer(5))
        {
            Ok(_) => panic!("bound-only transfer is not an exact polynomial"),
            Err(error) => error,
        };
        assert!(transfer_error.contains("bound-only"));
    }

    #[test]
    fn bound_chain_uses_typed_end_pre_fold_and_same_frame_result() {
        let root = owner(9);
        let proof = OperationalProofPayload {
            events: vec![
                ProofPayloadEvent::InvocationStart { root },
                ProofPayloadEvent::Result {
                    owner: root,
                    value: ProofPayloadValue::Exact {
                        terms: vec![],
                        summary: BoundedSummary::zero(),
                    },
                },
                ProofPayloadEvent::PreFoldPolynomial(ProofPayloadPreFoldPolynomial {
                    result_event: 1,
                    terms: vec![],
                    summary: BoundedSummary::zero(),
                    summary_evidence: None,
                }),
                ProofPayloadEvent::InvocationEnd {
                    root,
                    result: ProofPayloadValue::Exact {
                        terms: vec![],
                        summary: BoundedSummary::zero(),
                    },
                    pre_fold_event: 2,
                },
            ],
        };
        let index = PayloadIndex::new(&proof).expect("payload index");
        let (end_event, end_owner, _, pre_fold_event) = index.ends[0].clone();
        let (_, prefold) = index
            .prefolds
            .iter()
            .find(|(event, _)| *event == pre_fold_event)
            .expect("typed PreFold reference");
        let result = index.result(prefold.result_event).expect("typed Result reference");
        assert_eq!((end_event, end_owner, pre_fold_event), (3, root, 2));
        assert_eq!(result.owner, root);
        assert_eq!(index.immediate_frames[1], index.immediate_frames[2]);
        assert_eq!(index.immediate_frames[2], index.immediate_frames[3]);
    }
}
