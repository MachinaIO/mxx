use super::{NAMESPACE, generated_file};
use crate::operational_noise::{
    certificate_schema::{CertificateDocumentV1, CertificateResidualRootV1},
    g0::{
        BoundProjection, CanonicalExpressionDescriptor, CanonicalExpressionOperator,
        StableMatrixOperation, StableOperator,
    },
    simulation::{
        OperationalProofPayload, ProofPayloadCoefficientMerge, ProofPayloadCoefficientMergeSource,
        ProofPayloadEvent, ProofPayloadMonomial, ProofPayloadOwner, ProofPayloadRelationRule,
        ProofPayloadRule, ProofPayloadTerm, ProofPayloadValue, ProofPayloadValueRef,
    },
};
use num_bigint::BigUint;
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

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SemanticShardReport {
    schema_id: &'static str,
    schema_version: u32,
    shard_index: u64,
    start_event: u64,
    end_event: u64,
    theorem_count: u64,
    canonical_work: u64,
    operation_count: u64,
    relation_count: u64,
    bound_count: u64,
    raw_semantic_count: u64,
    raw_family_counts: [u64; 6],
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SemanticShardAggregateReport {
    schema_id: &'static str,
    schema_version: u32,
    shard_count: u64,
    theorem_count: u64,
    canonical_work_total: u64,
    canonical_work_max: u64,
    canonical_work_max_event: u64,
    shards: Vec<SemanticShardReport>,
}

#[derive(Clone)]
struct Frame {
    root: ProofPayloadOwner,
    start: u64,
    merge_count: u64,
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
    rule_event: u64,
    input_events: [u64; 2],
    output_event: u64,
    inputs: Vec<ResultRecord>,
    output: ResultRecord,
    scalar_left: bool,
    scalar_right: bool,
    raw_work: u64,
    rule: Option<ProofPayloadRule>,
    composite_relations: Vec<RelationProbe>,
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
    frame_start: u64,
    frame_end: u64,
    source: ProofPayloadMonomial,
    lhs: ProofPayloadMonomial,
    outer: num_bigint::BigInt,
    start: u32,
    end: u32,
    accumulator: ResultRecord,
    rhs: ResultRecord,
    output: ResultRecord,
    kind: RelationRuleKind,
    rule: ProofPayloadRelationRule,
    output_merge: ProofPayloadCoefficientMerge,
    rhs_pre_fold_event: Option<u64>,
}

struct PendingRelation {
    event: u64,
    owner: ProofPayloadOwner,
    frame_start: u64,
    frame_end: u64,
    source: ProofPayloadMonomial,
    lhs: ProofPayloadMonomial,
    outer: num_bigint::BigInt,
    start: u32,
    end: u32,
    accumulator: ResultRecord,
    rhs: ResultRecord,
    terms: BTreeMap<ProofPayloadMonomial, num_bigint::BigInt>,
    last_merge_event: Option<u64>,
    last_merge: Option<ProofPayloadCoefficientMerge>,
    kind: RelationRuleKind,
    rule: ProofPayloadRelationRule,
    rhs_pre_fold_event: Option<u64>,
}

#[derive(Clone)]
struct BoundProbe {
    root_result_event: u64,
    prefold_event: u64,
    end_event: u64,
    survivor_events: Vec<u64>,
    root: ResultRecord,
    prefold_terms: Vec<ProofPayloadTerm>,
    prefold_summary: crate::operational_noise::normal_form::BoundedSummary,
    prefold_evidence: Option<ProofPayloadValueRef>,
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

struct SemanticShard {
    index: u64,
    start: u64,
    end: u64,
    operations: Vec<OperationProbe>,
    relations: Vec<RelationProbe>,
    bounds: Vec<BoundProbe>,
    raw_semantic_count: u64,
    raw_family_counts: [u64; 6],
}

#[derive(Clone)]
enum RightRootNodeKind {
    Operation(OperationProbe),
    Terminal {
        producer_event: u64,
        frame_start: u64,
        rule: ProofPayloadRule,
        term: ProofPayloadTerm,
    },
}

#[derive(Clone)]
struct RightRootNode {
    result: ResultRecord,
    kind: RightRootNodeKind,
}

impl SemanticShard {
    fn canonical_work(&self) -> u64 {
        self.operations
            .iter()
            .map(|operation| operation.raw_work)
            .sum::<u64>()
            .saturating_add(
                self.relations.iter().map(|relation| relation.output.terms.len() as u64).sum(),
            )
            .saturating_add(
                self.bounds.iter().map(|bound| bound.survivor_contributions.len() as u64).sum(),
            )
    }
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
) -> Result<Option<OperationKind>, String> {
    let expression_row = usize::try_from(owner.expression_row).map_err(|_| {
        format!("operator owner {} expression row does not fit platform usize", owner_text(owner))
    })?;
    let row = statement.expressions.get(expression_row).ok_or_else(|| {
        format!(
            "operator owner {} references missing expression row {}",
            owner_text(owner),
            owner.expression_row
        )
    })?;
    let CanonicalExpressionDescriptor::Operation {
        operator: CanonicalExpressionOperator::Stable(StableOperator::Matrix { operation }),
        ..
    } = &row.descriptor
    else {
        return Ok(None);
    };
    Ok(Some(match operation {
        StableMatrixOperation::Add => OperationKind::Add,
        StableMatrixOperation::Subtract => OperationKind::Subtract,
        StableMatrixOperation::Multiply => OperationKind::Multiply,
        StableMatrixOperation::Tensor { .. } => OperationKind::Tensor,
        _ => return Ok(None),
    }))
}

fn typed_operation_rule(
    index: &PayloadIndex,
    result: &ResultRecord,
    kind: OperationKind,
) -> Result<Option<(u64, [ProofPayloadValueRef; 2], (bool, bool), ProofPayloadRule)>, String> {
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
                (inputs.len() == 2).then(|| {
                    (*event, [inputs[0].clone(), inputs[1].clone()], (false, false), rule.clone())
                })
            }
            (OperationKind::Multiply, ProofPayloadRule::Product { left, right, facts: _ }) => {
                Some((*event, [left.clone(), right.clone()], (false, false), rule.clone()))
            }
            (
                OperationKind::Tensor,
                ProofPayloadRule::Tensor {
                    left,
                    right,
                    left_is_constant_polynomial: _,
                    right_is_constant_polynomial: _,
                },
            ) => Some((*event, [left.clone(), right.clone()], (false, false), rule.clone())),
            _ => None,
        })
        .collect::<Vec<_>>();
    candidates.sort_by_key(|(event, _, _, _)| *event);
    if candidates.len() > 1 {
        return Err(format!(
            "operator Result {} owner {} has {} ambiguous typed {:?} rules in its immediate frame",
            result.event,
            owner_text(result.owner),
            candidates.len(),
            kind
        ));
    }
    let Some((event, refs, flags, rule)) = candidates.into_iter().next() else {
        return Ok(None);
    };
    Ok(Some((event, refs, flags, rule)))
}

fn scalar_action_key(key: &ProofPayloadMonomial) -> ProofPayloadMonomial {
    ProofPayloadMonomial {
        central_factors: key
            .central_factors
            .iter()
            .chain(key.ordered_factors.iter())
            .copied()
            .collect(),
        ordered_factors: Vec::new(),
    }
}

fn product_key_for_scalar_flags(
    left: &ProofPayloadMonomial,
    right: &ProofPayloadMonomial,
    left_scalar: bool,
    right_scalar: bool,
) -> ProofPayloadMonomial {
    let (left, right) = if left_scalar && !right_scalar {
        (scalar_action_key(left), right.clone())
    } else if right_scalar && !left_scalar {
        (left.clone(), scalar_action_key(right))
    } else {
        (left.clone(), right.clone())
    };
    ProofPayloadMonomial {
        central_factors: left
            .central_factors
            .iter()
            .chain(right.central_factors.iter())
            .copied()
            .collect(),
        ordered_factors: left
            .ordered_factors
            .iter()
            .chain(right.ordered_factors.iter())
            .copied()
            .collect(),
    }
}

fn canonical_monomial(mut key: ProofPayloadMonomial) -> ProofPayloadMonomial {
    key.central_factors.sort();
    key
}

fn monomial_equivalent(left: &ProofPayloadMonomial, right: &ProofPayloadMonomial) -> bool {
    canonical_monomial(left.clone()) == canonical_monomial(right.clone())
}

fn matching_scalar_flags_from_merges(
    merges: &[(ProofPayloadMonomial, ProofPayloadMonomial, ProofPayloadMonomial)],
) -> Vec<(bool, bool)> {
    let mut candidates = vec![(false, false), (true, false), (false, true), (true, true)];
    for (left, right, output) in merges {
        candidates.retain(|(left_scalar, right_scalar)| {
            monomial_equivalent(
                &product_key_for_scalar_flags(left, right, *left_scalar, *right_scalar),
                output,
            )
        });
    }
    candidates.sort();
    candidates
}

fn product_terms_for_scalar_flags(
    left: &[ProofPayloadTerm],
    right: &[ProofPayloadTerm],
    left_scalar: bool,
    right_scalar: bool,
) -> Vec<ProofPayloadTerm> {
    let mut terms = BTreeMap::<ProofPayloadMonomial, num_bigint::BigInt>::new();
    for left_term in left {
        for right_term in right {
            let key = product_key_for_scalar_flags(
                &left_term.monomial,
                &right_term.monomial,
                left_scalar,
                right_scalar,
            );
            let coefficient = &left_term.coefficient * &right_term.coefficient;
            *terms.entry(canonical_monomial(key)).or_default() += coefficient;
        }
    }
    terms
        .into_iter()
        .filter(|(_, coefficient)| !coefficient.is_zero())
        .map(|(monomial, coefficient)| ProofPayloadTerm { monomial, coefficient })
        .collect()
}

fn resolve_scalar_flags(
    candidates: &[(bool, bool)],
    inputs: &[ResultRecord],
) -> Result<(bool, bool), String> {
    if candidates.is_empty() {
        return Err("typed operator merges have no matching scalar flag placement".to_owned());
    }
    if candidates.len() == 1 {
        return Ok(candidates[0]);
    }
    let expected = candidates
        .iter()
        .map(|(left_scalar, right_scalar)| {
            product_terms_for_scalar_flags(
                &inputs[0].terms,
                &inputs[1].terms,
                *left_scalar,
                *right_scalar,
            )
        })
        .collect::<Vec<_>>();
    if expected.windows(2).all(|pair| pair[0] == pair[1]) {
        Ok(candidates[0])
    } else {
        Err(format!(
            "typed operator merges have ambiguous scalar flag placements {candidates:?} with distinct expected product polynomials"
        ))
    }
}

fn op_probe(
    statement: &CertificateDocumentV1,
    index: &PayloadIndex,
    result: &ResultRecord,
    kind: OperationKind,
) -> Result<OperationProbe, String> {
    let Some((rule_event, input_refs, _flags, rule)) = typed_operation_rule(index, result, kind)?
    else {
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
    let scalar_flags = match kind {
        OperationKind::Multiply | OperationKind::Tensor => {
            if operator_merges.is_empty() {
                return Err(format!(
                    "operator Result {} rule {} has no typed operator coefficient merges for scalar inference",
                    result.event, rule_event
                ));
            }
            let mut merge_events = Vec::new();
            let mut merge_terms = Vec::new();
            for (merge_event, merge) in &operator_merges {
                let ProofPayloadCoefficientMergeSource::Operator { inputs: refs } = &merge.source
                else {
                    unreachable!()
                };
                let left = index.result(refs[0].value_event)?;
                let right = index.result(refs[1].value_event)?;
                let left_term = left
                    .terms
                    .get(usize::try_from(refs[0].term_ordinal).map_err(|_| {
                        format!(
                            "operator Result {} merge {} left term ordinal overflows usize",
                            result.event, merge_event
                        )
                    })?)
                    .ok_or_else(|| {
                        format!(
                            "operator Result {} merge {} left term ordinal is out of range",
                            result.event, merge_event
                        )
                    })?;
                let right_term = right
                    .terms
                    .get(usize::try_from(refs[1].term_ordinal).map_err(|_| {
                        format!(
                            "operator Result {} merge {} right term ordinal overflows usize",
                            result.event, merge_event
                        )
                    })?)
                    .ok_or_else(|| {
                        format!(
                            "operator Result {} merge {} right term ordinal is out of range",
                            result.event, merge_event
                        )
                    })?;
                merge_events.push(*merge_event);
                merge_terms.push((
                    left_term.monomial.clone(),
                    right_term.monomial.clone(),
                    merge.output.clone(),
                ));
            }
            let candidates = matching_scalar_flags_from_merges(&merge_terms);
            resolve_scalar_flags(&candidates, &inputs).map_err(|error| {
                format!(
                    "operator Result {} rule {} typed merges {:?}: {error}",
                    result.event, rule_event, merge_events
                )
            })?
        }
        OperationKind::Add | OperationKind::Subtract | OperationKind::Direct => (false, false),
    };
    let raw_work = (inputs[0].terms.len() as u64).saturating_mul(inputs[1].terms.len() as u64);
    let _ = statement;
    Ok(OperationProbe {
        kind,
        rule_event,
        input_events: [inputs[0].event, inputs[1].event],
        output_event: result.event,
        inputs,
        output: result.clone(),
        scalar_left: scalar_flags.0,
        scalar_right: scalar_flags.1,
        raw_work,
        rule: Some(rule),
        composite_relations: Vec::new(),
    })
}

fn attach_composite_relations(
    mut operation: OperationProbe,
    relations: &[RelationProbe],
) -> Result<OperationProbe, String> {
    let mut attached = relations
        .iter()
        .filter(|relation| {
            relation.owner == operation.output.owner &&
                relation.event > operation.rule_event &&
                relation.event < operation.output_event
        })
        .cloned()
        .collect::<Vec<_>>();
    attached.sort_by_key(|relation| relation.event);
    operation.composite_relations = attached;
    Ok(operation)
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
    let Some((_, refs, _, _)) = typed_operation_rule(index, result, kind)? else {
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

fn select_add_output_index(
    candidates: &[(&ResultRecord, OperationProbe)],
) -> Result<usize, String> {
    candidates
        .iter()
        .enumerate()
        .max_by_key(|(_, (result, operation))| {
            (result.terms.len(), operation.inputs[0].terms.len(), std::cmp::Reverse(result.event))
        })
        .map(|(index, _)| index)
        .ok_or_else(|| "CP3 add probe has no actual matrix Add Result".to_owned())
}

fn select_relation_probes(candidates: Vec<RelationProbe>) -> Result<Vec<RelationProbe>, String> {
    let mut selected = Vec::with_capacity(2);
    for kind in [RelationRuleKind::Gadget, RelationRuleKind::Universal] {
        let Some(probe) = candidates
            .iter()
            .filter(|probe| probe.kind == kind)
            .max_by_key(|probe| (probe.output.terms.len(), std::cmp::Reverse(probe.event)))
            .cloned()
        else {
            let label = match kind {
                RelationRuleKind::Gadget => "Gadget",
                RelationRuleKind::Universal => "Universal",
            };
            return Err(format!(
                "CP3 relation probe requires an actual {label} relation application"
            ));
        };
        selected.push(probe);
    }
    Ok(selected)
}

fn finalize_relations(
    pending: &mut Vec<PendingRelation>,
    candidates: &mut Vec<RelationProbe>,
) -> Result<(), String> {
    for mut relation in pending.drain(..) {
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
            frame_start: relation.frame_start,
            frame_end: relation.frame_end,
            source: relation.source,
            lhs: relation.lhs,
            outer: relation.outer,
            start: relation.start,
            end: relation.end,
            accumulator: relation.accumulator,
            rhs: relation.rhs,
            output,
            kind: relation.kind,
            rule: relation.rule,
            output_merge: relation
                .last_merge
                .take()
                .expect("relation finalization has a typed output merge"),
            rhs_pre_fold_event: relation.rhs_pre_fold_event,
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
                        frame_start: *frame_start,
                        frame_end: *frame_end,
                        source: source_monomial.clone(),
                        lhs,
                        outer: outer_coefficient.clone(),
                        start: *ordered_start,
                        end: *ordered_end_exclusive,
                        accumulator,
                        rhs,
                        terms,
                        last_merge_event: None,
                        last_merge: None,
                        kind,
                        rule: rule.clone(),
                        rhs_pre_fold_event: match index.event(rhs_event)? {
                            ProofPayloadEvent::InvocationEnd { pre_fold_event, .. } => {
                                Some(*pre_fold_event)
                            }
                            _ => None,
                        },
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
                    relation.last_merge = Some(merge.clone());
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

fn bound_candidates(
    index: &PayloadIndex,
    ranges: &[(u64, u64, ProofPayloadOwner, u64)],
) -> Result<Vec<(u64, u64, BoundProbe)>, String> {
    let mut candidates = Vec::new();
    for (start, end, root, _) in ranges {
        let Some((end_event, end_owner, end_value, pre_fold_event)) =
            index.ends.iter().find(|(event, _, _, _)| *event == *end)
        else {
            continue;
        };
        if *end_owner != *root {
            return Err(format!("InvocationEnd {end_event} root does not match frame root"));
        }
        let Some((prefold_event, prefold)) =
            index.prefolds.iter().find(|(event, _)| *event == *pre_fold_event)
        else {
            return Err(format!(
                "InvocationEnd {end_event} references missing PreFold {pre_fold_event}"
            ));
        };
        if *prefold_event < *start ||
            *prefold_event >= *end ||
            index.immediate_frames
                [usize::try_from(*prefold_event).expect("indexed PreFold event")] !=
                Some(*start)
        {
            return Err(format!(
                "PreFold {prefold_event} is outside InvocationEnd {end_event} frame"
            ));
        }
        let root_result = match index.event(prefold.result_event)? {
            ProofPayloadEvent::Result { owner, .. } if *owner == *root => {
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
            Some(*start)
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
                *event >= *start &&
                    *event < *end &&
                    index.immediate_frames
                        [usize::try_from(*event).expect("indexed survivor event")] ==
                        Some(*start)
            })
            .map(|(_, fold)| (fold.coefficient.magnitude().to_string(), fold.bound.to_string()))
            .collect::<Vec<_>>();
        candidates.push((
            *start,
            *end,
            BoundProbe {
                root_result_event: prefold.result_event,
                prefold_event: *prefold_event,
                end_event: *end_event,
                survivor_events: index
                    .survivors
                    .iter()
                    .filter(|(event, _)| {
                        *event >= *start &&
                            *event < *end &&
                            index.immediate_frames
                                [usize::try_from(*event).expect("indexed survivor event")] ==
                                Some(*start)
                    })
                    .map(|(event, _)| *event)
                    .collect(),
                root: root_result,
                prefold_terms: prefold.terms.clone(),
                prefold_summary: prefold.summary.clone(),
                prefold_evidence: prefold.summary_evidence.clone(),
                end: end_result,
                survivor_contributions: survivors
                    .iter()
                    .map(|(actual, _)| actual.clone())
                    .collect(),
                survivor_bounds: survivors.into_iter().map(|(_, bound)| bound).collect(),
            },
        ));
    }
    Ok(candidates)
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
    let outer_kind = expression_kind(statement, outer_result.owner)?.ok_or_else(|| {
        format!(
            "outer Result {} owner {} has no matrix Add/Subtract/Multiply/Tensor statement row",
            outer_result.event,
            owner_text(outer_result.owner)
        )
    })?;
    let outer_semantic_rule = typed_operation_rule(index, &outer_result, outer_kind)?.is_some();
    let outer_eligible = typed_operator_eligibility(index, &outer_result, outer_kind)? ||
        (matches!(outer_kind, OperationKind::Add | OperationKind::Subtract) &&
            outer_semantic_rule);
    let outer_op = if outer_eligible {
        op_probe(statement, index, &outer_result, outer_kind)?
    } else {
        OperationProbe {
            kind: OperationKind::Direct,
            rule_event: 0,
            input_events: [0, 0],
            output_event: outer_result.event,
            inputs: Vec::new(),
            output: outer_result.clone(),
            scalar_left: false,
            scalar_right: false,
            raw_work: 0,
            rule: Some(ProofPayloadRule::Sum {
                inputs: vec![
                    ProofPayloadValueRef::Result {
                        event: 2,
                        projection: BoundProjection::Coefficient,
                    },
                    ProofPayloadValueRef::Result {
                        event: 3,
                        projection: BoundProjection::Coefficient,
                    },
                ],
            }),
            composite_relations: Vec::new(),
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

    let mut add_candidates = Vec::new();
    for result in &index.results {
        let Some(kind) = expression_kind(statement, result.owner)? else {
            continue;
        };
        if kind != OperationKind::Add {
            continue;
        }
        if !typed_operator_eligibility(index, result, kind)? {
            continue;
        }
        add_candidates.push((result, op_probe(statement, index, result, kind)?));
    }
    let add_output_index = select_add_output_index(&add_candidates)?;
    let (add_result, add_op) = add_candidates[add_output_index].clone();
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
    let mut product_candidates = Vec::new();
    for result in &index.results {
        let Some(kind) = expression_kind(statement, result.owner)? else {
            continue;
        };
        if !matches!(kind, OperationKind::Multiply | OperationKind::Tensor) {
            continue;
        }
        if !typed_operator_eligibility(index, result, kind)? {
            continue;
        }
        product_candidates.push((result, op_probe(statement, index, result, kind)?));
    }
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
    let selected_relations = select_relation_probes(relation_probes)?;
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

    let mut all_bound_candidates = bound_candidates(index, &ranges)?;
    let has_positive_survivors =
        all_bound_candidates.iter().any(|(_, _, probe)| !probe.survivor_contributions.is_empty());
    all_bound_candidates.retain(|(_, _, probe)| {
        !has_positive_survivors || !probe.survivor_contributions.is_empty()
    });
    all_bound_candidates
        .sort_by_key(|(start, end, probe)| (probe.survivor_contributions.len(), *start, *end));
    let bound_frame = all_bound_candidates.into_iter().next().ok_or_else(|| {
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

const SEMANTIC_SHARD_CHUNK_SIZE: u64 =
    crate::operational_noise::tall_e2e::SECURITY0_EVENT_CHUNK_SIZE as u64;

fn raw_semantic_shard_counts(index: &PayloadIndex, start: u64, end: u64) -> [u64; 6] {
    let mut counts = [0_u64; 6];
    for event in start..end {
        match &index.events[usize::try_from(event).expect("indexed semantic event")] {
            ProofPayloadEvent::BoundTransfer { .. } => counts[0] += 1,
            ProofPayloadEvent::Result { .. } => counts[1] += 1,
            ProofPayloadEvent::AppliedRelation { .. } => counts[2] += 1,
            ProofPayloadEvent::SurvivorFold(_) => counts[3] += 1,
            ProofPayloadEvent::PreFoldPolynomial(_) => counts[4] += 1,
            ProofPayloadEvent::InvocationEnd { .. } => counts[5] += 1,
            _ => {}
        }
    }
    counts
}

fn raw_semantic_shards(index: &PayloadIndex) -> Result<Vec<(u64, u64, u64, [u64; 6])>, String> {
    let event_count =
        u64::try_from(index.events.len()).map_err(|_| "semantic event count overflow")?;
    let shard_count = event_count.div_ceil(SEMANTIC_SHARD_CHUNK_SIZE);
    Ok((0..shard_count)
        .map(|shard_index| {
            let start = shard_index * SEMANTIC_SHARD_CHUNK_SIZE;
            let end = (start + SEMANTIC_SHARD_CHUNK_SIZE).min(event_count);
            (shard_index, start, end, raw_semantic_shard_counts(index, start, end))
        })
        .collect())
}

fn shard_module_name(index: u64) -> String {
    format!("SemanticShard{index:03}")
}

fn semantic_shard_candidates(
    statement: &CertificateDocumentV1,
    index: &PayloadIndex,
    _ranges: &[(u64, u64, ProofPayloadOwner, u64)],
    relation_probes: &[RelationProbe],
    bound_probes: &[(u64, u64, BoundProbe)],
) -> Result<Vec<SemanticShard>, String> {
    let mut shards = Vec::new();
    for (shard_index, start, end, raw_family_counts) in raw_semantic_shards(index)? {
        let mut operations = Vec::new();
        for result in &index.results {
            if result.event < start || result.event >= end {
                continue;
            }
            let kind = match expression_kind(statement, result.owner)? {
                Some(kind) if kind != OperationKind::Direct => kind,
                _ => continue,
            };
            if !typed_operator_eligibility(index, result, kind)? {
                continue;
            }
            let operation = op_probe(statement, index, result, kind)?;
            let result_frame = index.immediate_frames
                [usize::try_from(result.event).expect("indexed result event")];
            let matching_relations = relation_probes
                .iter()
                .filter(|relation| {
                    relation.owner == result.owner &&
                        relation.event > operation.rule_event &&
                        relation.event < operation.output_event &&
                        relation.frame_start == result_frame.unwrap_or(u64::MAX) &&
                        relation.frame_end >= relation.event
                })
                .cloned()
                .collect::<Vec<_>>();
            operations.push(attach_composite_relations(operation, &matching_relations)?);
        }
        let relations = relation_probes
            .iter()
            .filter(|relation| relation.event >= start && relation.event < end)
            .cloned()
            .collect::<Vec<_>>();
        let bounds = bound_probes
            .iter()
            .filter(|(_, frame_end, _)| *frame_end >= start && *frame_end < end)
            .map(|(_, _, bound)| bound.clone())
            .collect::<Vec<_>>();
        if operations.is_empty() && relations.is_empty() && bounds.is_empty() {
            continue;
        }
        shards.push(SemanticShard {
            index: shard_index,
            start,
            end,
            operations,
            relations,
            bounds,
            raw_semantic_count: raw_family_counts.iter().sum(),
            raw_family_counts,
        });
    }
    if shards.is_empty() {
        return Err("Security0 semantic shards have no supported reached obligations".to_owned());
    }
    Ok(shards)
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
    let modulus = ciphertext_modulus_text(statement)?;
    let (mut report, _selections, _long_monomial) = measure(proof)?;
    let index = PayloadIndex::new(proof)?;
    let _left_root_closure = validate_left_root_closure(proof, &index)?;
    let probes = build_probes(statement, proof, &index)?;
    let ranges = frame_ranges(proof)?;
    let relation_probes = relation_candidates(&index, &ranges)?;
    let all_bound_probes = bound_candidates(&index, &ranges)?;
    let shards =
        semantic_shard_candidates(statement, &index, &ranges, &relation_probes, &all_bound_probes)?;
    report.probes = probes.iter().map(ProbeStat::from_probe).collect();
    let report_bytes = serde_json::to_vec(&report)
        .map_err(|error| format!("semantic probe statistics encoding failed: {error}"))?;
    let mut files = vec![generated_file(
        "SemanticProbeStatistics.json",
        String::from_utf8(report_bytes).expect("JSON is UTF-8"),
    )];
    let (right_root_index, right_root_shards) =
        render_right_root(statement, &index, &relation_probes, &modulus)?;
    files.push(generated_file("Semantic/SemanticRightRoot.lean", right_root_index));
    files.extend(right_root_shards);

    let specs = [
        ("Semantic/Semantic000.lean", "Semantic000", "long-monomial-merge"),
        ("Semantic/Semantic001.lean", "Semantic001", "outer-result"),
        ("Semantic/Semantic002.lean", "Semantic002", "add-chain"),
        ("Semantic/Semantic003.lean", "Semantic003", "product-tensor"),
        ("Semantic/Semantic004.lean", "Semantic004", "relation"),
        ("Semantic/Semantic005.lean", "Semantic005", "bound-fold-result"),
    ];
    for (path, module, probe) in specs {
        files.push(generated_file(path, render_probe(module, probe, &probes, &modulus)));
    }
    let mut shard_reports = Vec::with_capacity(shards.len());
    let mut theorem_count = 0_u64;
    let mut canonical_work_total = 0_u64;
    for shard in &shards {
        let module = shard_module_name(shard.index);
        let (shard_source, emitted_theorem_count) = render_semantic_shard(&module, shard, &modulus);
        files.push(generated_file(format!("Semantic/{module}.lean"), shard_source));
        theorem_count += emitted_theorem_count;
        canonical_work_total += shard.canonical_work();
        shard_reports.push(SemanticShardReport {
            schema_id: SCHEMA_ID,
            schema_version: SCHEMA_VERSION,
            shard_index: shard.index,
            start_event: shard.start,
            end_event: shard.end,
            theorem_count: emitted_theorem_count,
            canonical_work: shard.canonical_work(),
            operation_count: shard.operations.len() as u64,
            relation_count: shard.relations.len() as u64,
            bound_count: shard.bounds.len() as u64,
            raw_semantic_count: shard.raw_semantic_count,
            raw_family_counts: shard.raw_family_counts,
        });
    }
    let max_shard = shards
        .iter()
        .max_by_key(|shard| (shard.canonical_work(), std::cmp::Reverse(shard.index)))
        .expect("nonempty semantic shards");
    let aggregate_report = SemanticShardAggregateReport {
        schema_id: SCHEMA_ID,
        schema_version: SCHEMA_VERSION,
        shard_count: shards.len() as u64,
        theorem_count,
        canonical_work_total,
        canonical_work_max: max_shard.canonical_work(),
        canonical_work_max_event: max_shard.start,
        shards: shard_reports,
    };
    files.push(generated_file(
        "Semantic/SemanticShardStatistics.json",
        serde_json::to_string(&aggregate_report).expect("JSON is UTF-8"),
    ));
    let mut index = String::new();
    index.push_str("import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticRightRoot\n");
    for (_, module, _) in specs {
        writeln!(index, "import {NAMESPACE}.Semantic.{module}").expect("String write");
    }
    for shard in &shards {
        writeln!(index, "import {NAMESPACE}.Semantic.{}", shard_module_name(shard.index))
            .expect("String write");
    }
    files.push(generated_file("Semantic/Semantic.lean", index));
    Ok(files)
}

fn reached_left_bound_rule(rule: &ProofPayloadRule) -> bool {
    use crate::operational_noise::simulation::ProofPayloadAuthority;
    matches!(
        rule,
        ProofPayloadRule::Authority(
            ProofPayloadAuthority::FactStore |
                ProofPayloadAuthority::ProgramFamilyFact |
                ProofPayloadAuthority::Operator |
                ProofPayloadAuthority::RelationPreimageSource { .. }
        ) | ProofPayloadRule::Identity { .. } |
            ProofPayloadRule::Sum { .. } |
            ProofPayloadRule::Scale { .. } |
            ProofPayloadRule::MonomialProduct { .. }
    )
}

fn validate_left_root_closure(
    proof: &OperationalProofPayload,
    index: &PayloadIndex,
) -> Result<Vec<u64>, String> {
    const LEFT_ROOT_EVENT: u64 = 107_402;
    const REACHED_RELATION_COUNT: usize = 1_905;
    let root = index.result(LEFT_ROOT_EVENT)?;
    if !matches!(
        root.summary.coefficient_bound(),
        crate::operational_noise::facts::NumericContract::Known(
            crate::operational_noise::facts::CoefficientBound::Finite(_)
        )
    ) {
        return Err(format!("Security0 left-root Result {LEFT_ROOT_EVENT} is not finite"));
    }
    let event_ids = super::closure::collect_security0_event_closure(proof, LEFT_ROOT_EVENT)?;
    let mut relations = 0_usize;
    let mut reached_rules = BTreeSet::new();
    for event in &event_ids {
        match index.event(*event)? {
            ProofPayloadEvent::AppliedRelation { .. } => relations += 1,
            ProofPayloadEvent::BoundTransfer { rule, .. } => {
                if !reached_left_bound_rule(rule) {
                    return Err(format!(
                        "Security0 left-root closure reaches unsupported bound rule {rule:?} at event {event}"
                    ));
                }
                reached_rules.insert(match rule {
                    ProofPayloadRule::Authority(authority) => match authority {
                        crate::operational_noise::simulation::ProofPayloadAuthority::FactStore => 0,
                        crate::operational_noise::simulation::ProofPayloadAuthority::ProgramFamilyFact => 1,
                        crate::operational_noise::simulation::ProofPayloadAuthority::Operator => 2,
                        crate::operational_noise::simulation::ProofPayloadAuthority::RelationPreimageSource { .. } => 3,
                        crate::operational_noise::simulation::ProofPayloadAuthority::Unavailable => unreachable!(),
                    },
                    ProofPayloadRule::Identity { .. } => 4,
                    ProofPayloadRule::Sum { .. } => 5,
                    ProofPayloadRule::Scale { .. } => 6,
                    ProofPayloadRule::MonomialProduct { .. } => 7,
                    _ => unreachable!(),
                });
            }
            _ => {}
        }
    }
    if relations != REACHED_RELATION_COUNT {
        return Err(format!(
            "Security0 left-root closure reaches {relations} relations, expected {REACHED_RELATION_COUNT}"
        ));
    }
    if reached_rules != BTreeSet::from_iter(0_u8..8) {
        return Err(format!(
            "Security0 left-root closure does not reach exactly the eight supported bound-rule families: {reached_rules:?}"
        ));
    }
    Ok(event_ids)
}

fn ciphertext_modulus_text(statement: &CertificateDocumentV1) -> Result<String, String> {
    BigUint::parse_bytes(statement.ciphertext_modulus.as_bytes(), 10)
        .filter(|value| !value.is_zero())
        .map(|value| value.to_string())
        .ok_or_else(|| {
            format!(
                "semantic renderer requires a positive decimal ciphertext modulus, got {:?}",
                statement.ciphertext_modulus
            )
        })
}

fn measure(
    proof: &OperationalProofPayload,
) -> Result<(StatisticsReport, Vec<Selection>, Option<ProofPayloadMonomial>), String> {
    let mut stack = Vec::<Frame>::new();
    let mut max_terms: Option<NodeStat> = None;
    let mut max_factors: Option<NodeStat> = None;
    let mut long_merge: Option<(u64, ProofPayloadOwner, ProofPayloadMonomial)> = None;
    let mut outer_result: Option<FrameSelection> = None;
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
            ProofPayloadEvent::InvocationStart { root } => {
                stack.push(Frame { root: *root, start: event, merge_count: 0, has_prefold: false })
            }
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
            }
            ProofPayloadEvent::CoefficientMerge(merge) => {
                if let Some(frame) = stack.last_mut() {
                    frame.merge_count += 1;
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

fn raw_term_text(term: &ProofPayloadTerm) -> String {
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
        "{{ coefficient := ({}), monomial := {{ centralFactors := [{}], orderedFactors := [{}] }} }}",
        term.coefficient, central, ordered
    )
}

fn raw_terms_text(terms: &[ProofPayloadTerm]) -> String {
    format!("[{}]", terms.iter().map(raw_term_text).collect::<Vec<_>>().join(", "))
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

fn projection_text(value: &BoundProjection) -> &'static str {
    match value {
        BoundProjection::Coefficient => ".coefficient",
        BoundProjection::Summary => ".summary",
    }
}

fn value_ref_text(value: &ProofPayloadValueRef) -> String {
    match value {
        ProofPayloadValueRef::Predecessor { binding_event, input_position, projection } => {
            format!(".predecessor {input_position} {binding_event} {}", projection_text(projection))
        }
        ProofPayloadValueRef::Result { event, projection } => {
            format!(".result {event} {}", projection_text(projection))
        }
        ProofPayloadValueRef::Transfer(event) => format!(".transfer {event}"),
    }
}

fn rule_text(value: &ProofPayloadRule) -> String {
    match value {
        ProofPayloadRule::Sum { inputs } => {
            format!(".sum [{}]", inputs.iter().map(value_ref_text).collect::<Vec<_>>().join(", "))
        }
        ProofPayloadRule::Product { left, right, facts } => format!(
            ".product ({}) ({}) ⟨{}, {}, {}, {}, {}⟩",
            value_ref_text(left),
            value_ref_text(right),
            if facts.left_is_constant_polynomial { "true" } else { "false" },
            if facts.right_is_constant_polynomial { "true" } else { "false" },
            facts
                .right_known_zero_rows
                .as_ref()
                .map_or_else(|| "none".to_owned(), |v| format!("some {v}")),
            facts.left_support_upper.map_or_else(|| "none".to_owned(), |v| format!("some {v}")),
            facts.right_support_upper.map_or_else(|| "none".to_owned(), |v| format!("some {v}")),
        ),
        ProofPayloadRule::Tensor {
            left,
            right,
            left_is_constant_polynomial,
            right_is_constant_polynomial,
        } => format!(
            ".tensor ({}) ({}) {} {}",
            value_ref_text(left),
            value_ref_text(right),
            if *left_is_constant_polynomial { "true" } else { "false" },
            if *right_is_constant_polynomial { "true" } else { "false" },
        ),
        _ => panic!("typed operation renderer received unsupported rule"),
    }
}

fn relation_rule_text(value: &ProofPayloadRelationRule) -> String {
    match value {
        ProofPayloadRelationRule::Universal { computed, lhs, lhs_layout, rhs_result } => format!(
            ".universal {computed} ({}) ({}) {rhs_result}",
            monomial_text(lhs),
            lhs_layout.as_ref().map_or_else(
                || "none".to_owned(),
                |layout| {
                    format!(
                        "some ⟨\"{}\", {}, {}⟩",
                        layout.name, layout.row_stride, layout.column_stride
                    )
                }
            ),
        ),
        ProofPayloadRelationRule::Gadget { gadget, decomposition, input, input_result } => format!(
            ".gadget ({}) ({}) ⟨{input}⟩ {input_result}",
            owner_text(*gadget),
            owner_text(*decomposition),
        ),
    }
}

fn merge_text(value: &ProofPayloadCoefficientMerge) -> String {
    let source = match &value.source {
        ProofPayloadCoefficientMergeSource::Operator { inputs } => format!(
            ".operator (⟨{}, {}⟩, ⟨{}, {}⟩)",
            inputs[0].value_event,
            inputs[0].term_ordinal,
            inputs[1].value_event,
            inputs[1].term_ordinal,
        ),
        ProofPayloadCoefficientMergeSource::Relation { application, source_term_ordinal } => {
            format!(".relation {application} {source_term_ordinal}")
        }
    };
    format!(
        ".coefficientMerge (⟨{}, {}, {}, ({})⟩)",
        owner_text(value.owner),
        source,
        monomial_text(&value.output),
        value.signed_contribution,
    )
}

fn emit_lookup(source: &mut String, name: &str, event_name: &str, event_value: String) {
    writeln!(
        source,
        "theorem {name} : (history.lookup {event_name}).map AnnotatedEvent.event = some ({event_value}) := by\n  rfl\n"
    )
    .expect("String write");
}

fn render_operation(source: &mut String, operation: &OperationProbe, modulus: &str) {
    if operation.kind == OperationKind::Direct {
        writeln!(
            source,
            "def outputRaw : List Term := {}",
            raw_terms_text(&operation.output.terms)
        )
        .expect("String write");
        writeln!(source, "def output : Polynomial Owner := outputRaw.map Term.toExact")
            .expect("String write");
        writeln!(
            source,
            "def outputSummary : Bound := {}",
            summary_text(&operation.output.summary)
        )
        .expect("String write");
        source.push_str("\ntheorem resultAgreement : CanonicalAgreement output output := by\n  decide +kernel\n");
        source.push_str("\ntheorem resultSound (env : Env Owner) :\n    evalPolynomial env output = evalPolynomial env output := by\n  rfl\n\n");
        return;
    }
    let left = &operation.inputs[0];
    let right = &operation.inputs[1];
    writeln!(source, "def leftRaw : List Term := {}", raw_terms_text(&left.terms))
        .expect("String write");
    writeln!(source, "def rightRaw : List Term := {}", raw_terms_text(&right.terms))
        .expect("String write");
    writeln!(source, "def outputRaw : List Term := {}", raw_terms_text(&operation.output.terms))
        .expect("String write");
    writeln!(source, "def left : Polynomial Owner := leftRaw.map Term.toExact")
        .expect("String write");
    writeln!(source, "def right : Polynomial Owner := rightRaw.map Term.toExact")
        .expect("String write");
    writeln!(source, "def output : Polynomial Owner := outputRaw.map Term.toExact")
        .expect("String write");
    writeln!(source, "def leftOwner : Owner := {}", owner_text(left.owner)).expect("String write");
    writeln!(source, "def rightOwner : Owner := {}", owner_text(right.owner))
        .expect("String write");
    writeln!(source, "def leftSummary : Bound := {}", summary_text(&left.summary))
        .expect("String write");
    writeln!(source, "def rightSummary : Bound := {}", summary_text(&right.summary))
        .expect("String write");
    writeln!(source, "def outputSummary : Bound := {}", summary_text(&operation.output.summary))
        .expect("String write");
    writeln!(source, "def selectedRawWork : Nat := {}", operation.raw_work).expect("String write");
    writeln!(source, "def selectedSumRuleEvent : Nat := {}", operation.rule_event)
        .expect("String write");
    writeln!(source, "def selectedLeftResultEvent : Nat := {}", operation.input_events[0])
        .expect("String write");
    writeln!(source, "def selectedRightResultEvent : Nat := {}", operation.input_events[1])
        .expect("String write");
    writeln!(source, "def selectedResultEvent : Nat := {}", operation.output_event)
        .expect("String write");
    if !operation.composite_relations.is_empty() {
        source.push_str("open EventReplay\n");
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
        source.push_str(
            "def expected0 : Polynomial Owner := productPoly left right leftScalar rightScalar\n",
        );
        for (ordinal, relation) in operation.composite_relations.iter().enumerate() {
            writeln!(
                source,
                "def sourceKey{ordinal} : MonomialKey Owner := {}",
                monomial_text(&relation.source)
            )
            .expect("String write");
            writeln!(
                source,
                "def lhsKey{ordinal} : MonomialKey Owner := {}",
                monomial_text(&relation.lhs)
            )
            .expect("String write");
            writeln!(
                source,
                "def relationRhs{ordinal}Raw : List Term := {}",
                raw_terms_text(&relation.rhs.terms)
            )
            .expect("String write");
            writeln!(
                source,
                "def relationRhs{ordinal} : Polynomial Owner := relationRhs{ordinal}Raw.map Term.toExact"
            )
            .expect("String write");
            writeln!(source, "def relationContext{ordinal} : MonomialContext Owner := relationContext sourceKey{ordinal} sourceKey{ordinal}.centralFactors {} {}", relation.start, relation.end)
                .expect("String write");
            writeln!(source, "def expected{} : Polynomial Owner := relationPoly expected{} sourceKey{} relationContext{} ({}) relationRhs{}", ordinal + 1, ordinal, ordinal, ordinal, relation.outer, ordinal)
                .expect("String write");
        }
        let final_expected = operation.composite_relations.len();
        source.push_str("\ntheorem productAgreement : CanonicalAgreement expected0 (productPoly left right leftScalar rightScalar) := by\n  decide +kernel\n");
        writeln!(source, "\ntheorem resultAgreement : CanonicalAgreement output expected{} := by\n  decide +kernel", final_expected)
            .expect("String write");
        source.push_str("\ntheorem resultSound (env : Env Owner)\n");
        for ordinal in 0..final_expected {
            writeln!(source, "    (baseRelation{ordinal} : evalMonomial env lhsKey{ordinal} % Int.ofNat {modulus} = evalPolynomial env relationRhs{ordinal} % Int.ofNat {modulus})")
                .expect("String write");
        }
        writeln!(source, "    : evalPolynomial env output % Int.ofNat {modulus} =\n      (evalPolynomial env left * evalPolynomial env right) % Int.ofNat {modulus} := by\n  have productSound := productCanonicalResultSound env left right expected0 leftScalar rightScalar productAgreement").expect("String write");
        for (ordinal, relation) in operation.composite_relations.iter().enumerate() {
            writeln!(source, "  have relationSound{ordinal} := relationCanonicalResultSound {modulus} env expected{ordinal} sourceKey{ordinal} lhsKey{ordinal} sourceKey{ordinal}.centralFactors {} {} ({}) relationRhs{ordinal} expected{} (by decide +kernel) baseRelation{ordinal} (by decide +kernel)", relation.start, relation.end, relation.outer, ordinal + 1)
                .expect("String write");
        }
        source.push_str("  have outputSound := canonicalAgreement_eval env output expected");
        writeln!(source, "{} resultAgreement", final_expected).expect("String write");
        source.push_str("  calc\n");
        writeln!(source, "    evalPolynomial env output % Int.ofNat {modulus} = evalPolynomial env expected{} % Int.ofNat {modulus} := by rw [outputSound]", final_expected)
            .expect("String write");
        for ordinal in (0..final_expected).rev() {
            writeln!(
                source,
                "    _ = evalPolynomial env expected{} % Int.ofNat {modulus} := relationSound{ordinal}",
                ordinal
            )
            .expect("String write");
        }
        writeln!(source, "    _ = (evalPolynomial env left * evalPolynomial env right) % Int.ofNat {modulus} := by rw [productSound]\n").expect("String write");
        return;
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
            let left_summary = summary_text(&left.summary);
            let output_summary = summary_text(&operation.output.summary);
            let right_is_exact_zero = matches!(
                right.summary.coefficient_bound(),
                crate::operational_noise::facts::NumericContract::Known(
                    crate::operational_noise::facts::CoefficientBound::ExactZero
                )
            );
            if left_summary == output_summary &&
                left_summary.starts_with("(.finite ") &&
                right_is_exact_zero
            {
                let maximum = summary_bound_nat_text(&left.summary);
                writeln!(
                    source,
                    "theorem resultClaimSound (env : Env Owner) (leftActual rightActual : Int)\n    (leftClaim : ValueClaim.Interprets {modulus} env leftActual (.exact left leftSummary))\n    (rightClaim : ValueClaim.Interprets {modulus} env rightActual (.exact right rightSummary)) :\n    ValueClaim.Interprets {modulus} env (leftActual - rightActual) (.exact output outputSummary) := by\n  have result := exactValueClaim_sub_of_mod_zero {modulus} env leftActual rightActual\n    left right output {maximum}\n    (by simpa [leftSummary] using leftClaim)\n    (exactClaim_mod_zero {modulus} env rightActual right\n      (by simpa [rightSummary] using rightClaim) (by decide))\n    (resultSound env)\n  simpa [outputSummary] using result\n"
                )
                .expect("String write");
            }
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

fn render_operation_bindings(source: &mut String, operation: &OperationProbe) {
    if operation.kind == OperationKind::Direct {
        emit_lookup(
            source,
            "selectedResultAt",
            "selectedEvent",
            ".resultExact selectedOwner outputRaw outputSummary".to_owned(),
        );
        return;
    }
    emit_lookup(
        source,
        "selectedLeftResultAt",
        "selectedLeftResultEvent",
        ".resultExact leftOwner leftRaw leftSummary".to_owned(),
    );
    emit_lookup(
        source,
        "selectedRightResultAt",
        "selectedRightResultEvent",
        ".resultExact rightOwner rightRaw rightSummary".to_owned(),
    );
    emit_lookup(
        source,
        "selectedResultAt",
        "selectedResultEvent",
        ".resultExact selectedOwner outputRaw outputSummary".to_owned(),
    );
    let rule = operation.rule.as_ref().expect("typed operation rule for semantic operation");
    emit_lookup(
        source,
        "selectedRuleAt",
        "selectedSumRuleEvent",
        format!(".boundTransfer selectedOwner ({})", rule_text(rule)),
    );
}

fn render_relation(source: &mut String, relation: &RelationProbe, modulus: &str) {
    source.push_str("open EventReplay\n");
    writeln!(
        source,
        "def accumulatorRaw : List Term := {}",
        raw_terms_text(&relation.accumulator.terms)
    )
    .expect("String write");
    writeln!(source, "def relationRhsRaw : List Term := {}", raw_terms_text(&relation.rhs.terms))
        .expect("String write");
    writeln!(
        source,
        "def relationOutputRaw : List Term := {}",
        raw_terms_text(&relation.output.terms)
    )
    .expect("String write");
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
    writeln!(source, "def relationRhs : Polynomial Owner := relationRhsRaw.map Term.toExact")
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
    writeln!(source, "theorem relationSound (env : Env Owner)\n    (baseRelation : evalMonomial env lhsKey % Int.ofNat {modulus} =\n      evalPolynomial env relationRhs % Int.ofNat {modulus}) :\n    evalPolynomial env relationOutput % Int.ofNat {modulus} =\n      evalPolynomial env accumulator % Int.ofNat {modulus} := by\n  exact relationCanonicalResultSound {modulus} env accumulator sourceKey lhsKey\n    sourceKey.centralFactors {} {} ({}) relationRhs relationOutput\n    (by decide +kernel) baseRelation relationAgreement\n", relation.start, relation.end, relation.outer).expect("String write");
}

fn render_relation_bindings(source: &mut String, relation: &RelationProbe) {
    writeln!(source, "def relationRhsEvent : Nat := {}", relation.rhs.event).expect("String write");
    writeln!(source, "def relationRhsOwner : Owner := {}", owner_text(relation.rhs.owner))
        .expect("String write");
    writeln!(source, "def relationRhsSummary : Bound := {}", summary_text(&relation.rhs.summary))
        .expect("String write");
    writeln!(source, "def relationOutputEvent : Nat := {}", relation.output.event)
        .expect("String write");
    emit_lookup(
        source,
        "selectedRelationAt",
        "selectedEvent",
        format!(
            ".appliedRelation selectedOwner ({}) ({}) {} {} ({})",
            monomial_text(&relation.source),
            relation.outer,
            relation.start,
            relation.end,
            relation_rule_text(&relation.rule),
        ),
    );
    let rhs_event = if let Some(pre_fold_event) = relation.rhs_pre_fold_event {
        format!(
            ".invocationEndExact relationRhsOwner {pre_fold_event} relationRhsRaw relationRhsSummary"
        )
    } else {
        ".resultExact relationRhsOwner relationRhsRaw relationRhsSummary".to_owned()
    };
    emit_lookup(source, "selectedRhsResultAt", "relationRhsEvent", rhs_event);
    emit_lookup(
        source,
        "selectedRelationOutputAt",
        "relationOutputEvent",
        merge_text(&relation.output_merge),
    );
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

fn render_bound(source: &mut String, bound: &BoundProbe, modulus: &str) {
    writeln!(source, "def rootRaw : List Term := {}", raw_terms_text(&bound.root.terms))
        .expect("String write");
    writeln!(source, "def prefoldRaw : List Term := {}", raw_terms_text(&bound.prefold_terms))
        .expect("String write");
    writeln!(source, "def endRaw : List Term := {}", raw_terms_text(&bound.end.terms))
        .expect("String write");
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
    render_survivor_witness(source, bound);
    source.push_str("\ntheorem prefoldResult : prefoldTerms = rootTerms := by rfl\n\ntheorem prefoldBoundSound : rootBound ≤ prefoldBound := by decide +kernel\n\n");
    source.push_str("\ntheorem prefoldSound :\n  preFoldBound rootBound prefoldBound survivorContributions survivorBounds := by\n  exact (preFoldSound rootTerms prefoldTerms prefoldResult prefoldBoundSound survivorBoundsSound).2\n\ntheorem endResult : endTerms = prefoldTerms := by rfl\n\ntheorem endSummaryResult : endSummary = prefoldSummary := by rfl\n\ntheorem endSound :\n  endTerms = prefoldTerms ∧ endSummary = prefoldSummary := by\n  exact ⟨endResult, endSummaryResult⟩\n\n");
    writeln!(
        source,
        "theorem invocationEndClaimSound (env : Env Owner) (actual : Int)\n    (claim : ValueClaim.Interprets {modulus} env actual (.exact rootTerms rootSummary)) :\n    ValueClaim.Interprets {modulus} env actual (.exact endTerms endSummary) := by\n  exact invocationEndSound {modulus} env actual rootTerms endTerms rootSummary endSummary\n    claim endResult endSummaryResult\n"
    )
    .expect("String write");
}

fn render_bound_bindings(source: &mut String, bound: &BoundProbe) {
    emit_lookup(
        source,
        "selectedRootResultAt",
        "rootResultEvent",
        ".resultExact selectedOwner rootRaw rootSummary".to_owned(),
    );
    let evidence = bound
        .prefold_evidence
        .as_ref()
        .map_or_else(|| "none".to_owned(), |value| format!("some ({})", value_ref_text(value)));
    emit_lookup(
        source,
        "selectedPreFoldAt",
        "prefoldEvent",
        format!(".preFoldPolynomial rootResultEvent prefoldRaw prefoldSummary ({evidence})"),
    );
    emit_lookup(
        source,
        "selectedInvocationEndAt",
        "endEvent",
        ".invocationEndExact selectedOwner prefoldEvent endRaw endSummary".to_owned(),
    );
}

fn render_survivor_witness(source: &mut String, bound: &BoundProbe) {
    const CHUNK_SIZE: usize = 16;
    if bound.survivor_contributions.is_empty() {
        source.push_str(
            "def survivorContributions : List Nat := []\ndef survivorBounds : List Nat := []\n",
        );
        source.push_str("theorem survivorBoundsSound : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributions survivorBounds := by\n  exact List.Forall₂.nil\n");
        return;
    }
    let mut nodes = Vec::new();
    for (chunk, (contributions, bounds)) in bound
        .survivor_contributions
        .chunks(CHUNK_SIZE)
        .zip(bound.survivor_bounds.chunks(CHUNK_SIZE))
        .enumerate()
    {
        let contribution_name = format!("survivorContributionsChunk{chunk}");
        let bounds_name = format!("survivorBoundsChunk{chunk}");
        let theorem_name = format!("survivorBoundsSoundChunk{chunk}");
        writeln!(source, "def {contribution_name} : List Nat := [{}]", contributions.join(", "))
            .expect("String write");
        writeln!(source, "def {bounds_name} : List Nat := [{}]", bounds.join(", "))
            .expect("String write");
        let mut proof = String::from("by\n");
        for depth in 0..contributions.len() {
            let indent = "  ".repeat(depth + 1);
            proof.push_str(&format!("{indent}constructor\n"));
            proof.push_str(&format!("{indent}· omega\n"));
            proof.push_str(&format!("{indent}·\n"));
        }
        let indent = "  ".repeat(contributions.len() + 1);
        proof.push_str(&format!("{indent}exact List.Forall₂.nil\n"));
        writeln!(source, "theorem {theorem_name} : List.Forall₂ (fun actual bound => actual ≤ bound) {contribution_name} {bounds_name} :=\n{proof}")
            .expect("String write");
        nodes.push((contribution_name, bounds_name, theorem_name));
    }
    let mut level = 0;
    while nodes.len() > 1 {
        let mut next = Vec::new();
        for (pair, children) in nodes.chunks(2).enumerate() {
            if children.len() == 1 {
                next.push(children[0].clone());
                continue;
            }
            let left = &children[0];
            let right = &children[1];
            let contribution_name = format!("survivorContributionsTree{level}_{pair}");
            let bounds_name = format!("survivorBoundsTree{level}_{pair}");
            let theorem_name = format!("survivorBoundsSoundTree{level}_{pair}");
            writeln!(
                source,
                "def {contribution_name} : List Nat := {left_name} ++ {right_name}",
                left_name = left.0,
                right_name = right.0
            )
            .expect("String write");
            writeln!(
                source,
                "def {bounds_name} : List Nat := {left_name} ++ {right_name}",
                left_name = left.1,
                right_name = right.1
            )
            .expect("String write");
            writeln!(source, "theorem {theorem_name} : List.Forall₂ (fun actual bound => actual ≤ bound) {contribution_name} {bounds_name} := by\n  exact forall₂_append {left_theorem} {right_theorem}", left_theorem = left.2, right_theorem = right.2)
                .expect("String write");
            next.push((contribution_name, bounds_name, theorem_name));
        }
        nodes = next;
        level += 1;
    }
    let root = &nodes[0];
    writeln!(source, "def survivorContributions : List Nat := {}", root.0).expect("String write");
    writeln!(source, "def survivorBounds : List Nat := {}", root.1).expect("String write");
    writeln!(source, "theorem survivorBoundsSound : List.Forall₂ (fun actual bound => actual ≤ bound) survivorContributions survivorBounds := by\n  exact {}", root.2)
        .expect("String write");
}

fn render_semantic_shard(module: &str, shard: &SemanticShard, modulus: &str) -> (String, u64) {
    let mut source = format!(
        "import Mxx.Certificate.OperationalNoise.TallSemantics\n\
         import {NAMESPACE}.Proof.History\n\n\
         set_option autoImplicit false\n\
         set_option relaxedAutoImplicit false\n\n\
         namespace {NAMESPACE}.Semantic.{module}\n\n\
         open Mxx.Certificate.OperationalNoise\n\
         open TallSecurity0ABI\n\
         open TallSemantics\n\n"
    );
    writeln!(source, "def shardIndex : Nat := {}", shard.index).expect("String write");
    writeln!(source, "def shardStartEvent : Nat := {}", shard.start).expect("String write");
    writeln!(source, "def shardEndEvent : Nat := {}", shard.end).expect("String write");
    writeln!(source, "def rawSemanticCount : Nat := {}", shard.raw_semantic_count)
        .expect("String write");
    writeln!(source, "def rawBoundTransferCount : Nat := {}", shard.raw_family_counts[0])
        .expect("String write");
    writeln!(source, "def rawResultCount : Nat := {}", shard.raw_family_counts[1])
        .expect("String write");
    writeln!(source, "def rawRelationCount : Nat := {}", shard.raw_family_counts[2])
        .expect("String write");
    writeln!(source, "def rawSurvivorFoldCount : Nat := {}", shard.raw_family_counts[3])
        .expect("String write");
    writeln!(source, "def rawPreFoldCount : Nat := {}", shard.raw_family_counts[4])
        .expect("String write");
    writeln!(source, "def rawInvocationEndCount : Nat := {}", shard.raw_family_counts[5])
        .expect("String write");
    writeln!(source, "def canonicalWork : Nat := {}", shard.canonical_work())
        .expect("String write");
    for (ordinal, operation) in shard.operations.iter().enumerate() {
        writeln!(source, "\nnamespace Operation{ordinal}").expect("String write");
        writeln!(source, "def selectedEvent : Nat := {}", operation.output_event)
            .expect("String write");
        writeln!(source, "def selectedOwner : Owner := {}", owner_text(operation.output.owner))
            .expect("String write");
        render_operation(&mut source, operation, modulus);
        render_operation_bindings(&mut source, operation);
        writeln!(source, "end Operation{ordinal}").expect("String write");
    }
    for (ordinal, relation) in shard.relations.iter().enumerate() {
        writeln!(source, "\nnamespace Relation{ordinal}").expect("String write");
        writeln!(source, "def selectedEvent : Nat := {}", relation.event).expect("String write");
        writeln!(source, "def selectedOwner : Owner := {}", owner_text(relation.owner))
            .expect("String write");
        render_relation(&mut source, relation, modulus);
        render_relation_bindings(&mut source, relation);
        writeln!(source, "end Relation{ordinal}").expect("String write");
    }
    for (ordinal, bound) in shard.bounds.iter().enumerate() {
        writeln!(source, "\nnamespace Bound{ordinal}").expect("String write");
        writeln!(source, "def selectedEvent : Nat := {}", bound.end.event).expect("String write");
        writeln!(source, "def selectedOwner : Owner := {}", owner_text(bound.end.owner))
            .expect("String write");
        writeln!(source, "def rootResultEvent : Nat := {}", bound.root_result_event)
            .expect("String write");
        writeln!(source, "def prefoldEvent : Nat := {}", bound.prefold_event)
            .expect("String write");
        writeln!(source, "def endEvent : Nat := {}", bound.end_event).expect("String write");
        writeln!(
            source,
            "def survivorEvents : List Nat := [{}]",
            bound.survivor_events.iter().map(u64::to_string).collect::<Vec<_>>().join(", ")
        )
        .expect("String write");
        render_bound(&mut source, bound, modulus);
        render_bound_bindings(&mut source, bound);
        writeln!(source, "end Bound{ordinal}").expect("String write");
    }
    let theorem_count = source.matches("\ntheorem ").count() as u64;
    writeln!(source, "\ndef theoremCount : Nat := {theorem_count}").expect("String write");
    source.push_str(&format!("\nend {NAMESPACE}.Semantic.{module}\n"));
    (source, theorem_count)
}

impl BoundProbe {
    fn root_summary(&self) -> crate::operational_noise::normal_form::BoundedSummary {
        self.root.summary.clone()
    }
    fn end_summary(&self) -> crate::operational_noise::normal_form::BoundedSummary {
        self.end.summary.clone()
    }
}

fn render_probe(module: &str, probe: &str, probes: &[ProbeSelection], modulus: &str) -> String {
    let mut source = format!(
        "import Mxx.Certificate.OperationalNoise.TallSemantics\nimport {NAMESPACE}.Proof.History\n\nset_option autoImplicit false\nset_option relaxedAutoImplicit false\n\nnamespace {NAMESPACE}.Semantic.{module}\n\nopen Mxx.Certificate.OperationalNoise\nopen TallSecurity0ABI\nopen TallSemantics\n\n"
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
                if probe == "add-chain" {
                    source.push_str("namespace AddResult\n\n");
                    render_operation(
                        &mut source,
                        selection.operation.as_ref().expect("operation probe"),
                        modulus,
                    );
                    render_operation_bindings(
                        &mut source,
                        selection.operation.as_ref().expect("operation probe"),
                    );
                    source.push_str("end AddResult\n");
                } else {
                    render_operation(
                        &mut source,
                        selection.operation.as_ref().expect("operation probe"),
                        modulus,
                    );
                    render_operation_bindings(
                        &mut source,
                        selection.operation.as_ref().expect("operation probe"),
                    );
                }
            }
        }
        "relation" => {
            if let Some(selection) = selected {
                for (ordinal, relation) in selection.relations.iter().enumerate() {
                    writeln!(source, "namespace Relation{ordinal}").expect("String write");
                    render_relation(&mut source, relation, modulus);
                    render_relation_bindings(&mut source, relation);
                    writeln!(source, "end Relation{ordinal}").expect("String write");
                }
            }
        }
        "bound-fold-result" => {
            if let Some(selection) = selected {
                render_bound(&mut source, selection.bound.as_ref().expect("bound probe"), modulus);
                render_bound_bindings(&mut source, selection.bound.as_ref().expect("bound probe"));
            }
        }
        _ => {}
    }
    source.push_str(&format!("end {NAMESPACE}.Semantic.{module}\n"));
    source
}

fn reached_terminal_rule(rule: &ProofPayloadRule) -> bool {
    use crate::operational_noise::simulation::ProofPayloadAuthority;
    matches!(
        rule,
        ProofPayloadRule::Authority(
            ProofPayloadAuthority::FactStore |
                ProofPayloadAuthority::ProgramFamilyFact |
                ProofPayloadAuthority::Operator
        ) | ProofPayloadRule::Identity { .. } |
            ProofPayloadRule::Scale { .. }
    )
}

fn right_root_nodes(
    statement: &CertificateDocumentV1,
    index: &PayloadIndex,
    relation_probes: &[RelationProbe],
) -> Result<Vec<RightRootNode>, String> {
    const RIGHT_ROOT_EVENT: u64 = 6275;
    let CertificateResidualRootV1::Family { program, .. } = statement.residual_root else {
        return Err("Security0 right-root semantics requires a family residual root".to_owned());
    };
    let mut pending = vec![RIGHT_ROOT_EVENT];
    let mut nodes = BTreeMap::<u64, RightRootNode>::new();
    while let Some(event) = pending.pop() {
        if nodes.contains_key(&event) {
            continue;
        }
        let result = index.by_event.get(&event).ok_or_else(|| {
            format!("Security0 right-root dependency {event} is not an exact Result")
        })?;
        if !matches!(
            result.summary.coefficient_bound(),
            crate::operational_noise::facts::NumericContract::Known(
                crate::operational_noise::facts::CoefficientBound::ExactZero
            )
        ) {
            return Err(format!("Security0 right-root dependency Result {event} is not exact-zero"));
        }
        if !matches!(
            result.owner.scope,
            crate::operational_noise::simulation::ProofPayloadScope::Program { program_row }
                if program_row == program
        ) {
            return Err(format!(
                "Security0 right-root dependency Result {event} is outside residual program {program}"
            ));
        }
        let kind = match expression_kind(statement, result.owner)? {
            Some(operation_kind) if operation_kind != OperationKind::Direct => {
                let operation = op_probe(statement, index, result, operation_kind)?;
                let result_frame = index.immediate_frames
                    [usize::try_from(event).map_err(|_| "semantic event index overflow")?];
                let matching_relations = relation_probes
                    .iter()
                    .filter(|relation| {
                        relation.owner == result.owner &&
                            relation.event > operation.rule_event &&
                            relation.event < operation.output_event &&
                            relation.frame_start == result_frame.unwrap_or(u64::MAX) &&
                            relation.frame_end >= relation.event
                    })
                    .cloned()
                    .collect::<Vec<_>>();
                let operation = attach_composite_relations(operation, &matching_relations)?;
                pending.extend(operation.input_events);
                RightRootNodeKind::Operation(operation)
            }
            _ => {
                let producer_event = event.checked_sub(1).ok_or_else(|| {
                    format!("terminal Result {event} has no preceding producer event")
                })?;
                let ProofPayloadEvent::BoundTransfer { owner, rule } =
                    index.event(producer_event)?
                else {
                    return Err(format!(
                        "terminal Result {event} is not adjacent to a BoundTransfer producer"
                    ));
                };
                if *owner != result.owner || !reached_terminal_rule(rule) {
                    return Err(format!(
                        "terminal Result {event} has unsupported producer {rule:?}"
                    ));
                }
                let producer_frame = index.immediate_frames[usize::try_from(producer_event)
                    .map_err(|_| "semantic event index overflow")?];
                let result_frame = index.immediate_frames
                    [usize::try_from(event).map_err(|_| "semantic event index overflow")?];
                if producer_frame != result_frame {
                    return Err(format!(
                        "terminal Result {event} and producer {producer_event} have different frames"
                    ));
                }
                let [term] = result.terms.as_slice() else {
                    return Err(format!(
                        "terminal Result {event} is not a singleton exact polynomial"
                    ));
                };
                RightRootNodeKind::Terminal {
                    producer_event,
                    frame_start: result_frame.ok_or_else(|| {
                        format!("terminal Result {event} is outside an invocation frame")
                    })?,
                    rule: rule.clone(),
                    term: term.clone(),
                }
            }
        };
        nodes.insert(event, RightRootNode { result: result.clone(), kind });
    }
    let nodes = nodes.into_values().collect::<Vec<_>>();
    for node in &nodes {
        if let RightRootNodeKind::Operation(operation) = &node.kind {
            for input in operation.input_events {
                if input >= node.result.event ||
                    !nodes.iter().any(|node| node.result.event == input)
                {
                    return Err(format!(
                        "right-root Result {} has non-topological input Result {input}",
                        node.result.event
                    ));
                }
            }
        }
    }
    Ok(nodes)
}

fn reached_terminal_constructor(rule: &ProofPayloadRule) -> Result<String, String> {
    use crate::operational_noise::simulation::ProofPayloadAuthority;
    Ok(match rule {
        ProofPayloadRule::Authority(ProofPayloadAuthority::FactStore) => {
            ".authorityFactStore".to_owned()
        }
        ProofPayloadRule::Authority(ProofPayloadAuthority::ProgramFamilyFact) => {
            ".authorityProgramFamilyFact".to_owned()
        }
        ProofPayloadRule::Authority(ProofPayloadAuthority::Operator) => {
            ".authorityOperator".to_owned()
        }
        ProofPayloadRule::Identity { input } => {
            format!(".identity ({})", value_ref_text(input))
        }
        ProofPayloadRule::Scale { value, scale } => {
            format!(".scale ({}) ({})", value_ref_text(value), terminal_scale_text(scale))
        }
        _ => return Err("unsupported reached terminal rule".to_owned()),
    })
}

fn terminal_scale_text(scale: &crate::operational_noise::simulation::ProofPayloadScale) -> String {
    match scale {
        crate::operational_noise::simulation::ProofPayloadScale::Value(value) => {
            format!(".value ({})", value_ref_text(value))
        }
        crate::operational_noise::simulation::ProofPayloadScale::Magnitude(value) => {
            format!(".magnitude {value}")
        }
    }
}

fn reached_terminal_rule_text(rule: &ProofPayloadRule) -> Result<String, String> {
    use crate::operational_noise::simulation::ProofPayloadAuthority;
    Ok(match rule {
        ProofPayloadRule::Authority(ProofPayloadAuthority::FactStore) => {
            ".authority (.factStore)".to_owned()
        }
        ProofPayloadRule::Authority(ProofPayloadAuthority::ProgramFamilyFact) => {
            ".authority (.programFamilyFact)".to_owned()
        }
        ProofPayloadRule::Authority(ProofPayloadAuthority::Operator) => {
            ".authority (.operator)".to_owned()
        }
        ProofPayloadRule::Identity { input } => {
            format!(".identity ({})", value_ref_text(input))
        }
        ProofPayloadRule::Scale { value, scale } => {
            format!(".scale ({}) ({})", value_ref_text(value), terminal_scale_text(scale))
        }
        _ => return Err("unsupported reached terminal rule".to_owned()),
    })
}

fn render_right_root_operation(
    source: &mut String,
    operation: &OperationProbe,
    left_event: u64,
    right_event: u64,
) {
    writeln!(source, "def leftRaw : List Term := SemanticRightRootResult{left_event}.rawTerms")
        .expect("String write");
    writeln!(source, "def rightRaw : List Term := SemanticRightRootResult{right_event}.rawTerms")
        .expect("String write");
    writeln!(source, "def outputRaw : List Term := {}", raw_terms_text(&operation.output.terms))
        .expect("String write");
    source.push_str(
        "def left : Polynomial Owner := leftRaw.map Term.toExact\n\
         def right : Polynomial Owner := rightRaw.map Term.toExact\n\
         def output : Polynomial Owner := outputRaw.map Term.toExact\n",
    );
    writeln!(source, "def owner : Owner := {}", owner_text(operation.output.owner))
        .expect("String write");
    writeln!(source, "def rawTerms : List Term := outputRaw").expect("String write");
    source.push_str("def summary : Bound := .exactZero\n");
    writeln!(source, "def resultEvent : Nat := {}", operation.output_event).expect("String write");
    match operation.kind {
        OperationKind::Add => {
            source.push_str("theorem resultAgreement : CanonicalAgreement output (add left right) := by\n  decide +kernel\n\ntheorem resultSound (env : Env Owner) :\n    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by\n  exact addCanonicalResultSound env left right output resultAgreement\n\n");
        }
        OperationKind::Subtract => {
            source.push_str("theorem resultAgreement : CanonicalAgreement output (subtract left right) := by\n  decide +kernel\n\ntheorem resultSound (env : Env Owner) :\n    evalPolynomial env output = evalPolynomial env left - evalPolynomial env right := by\n  exact subCanonicalResultSound env left right output resultAgreement\n\n");
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
            source.push_str("theorem resultAgreement : CanonicalAgreement output\n    (productPoly left right leftScalar rightScalar) := by\n  decide +kernel\n\ntheorem resultSound (env : Env Owner) :\n    evalPolynomial env output = evalPolynomial env left * evalPolynomial env right := by\n  exact productCanonicalResultSound env left right output leftScalar rightScalar resultAgreement\n\n");
        }
        OperationKind::Direct => unreachable!("right-root direct operation"),
    }
}

fn right_root_operation_refs(
    operation: &OperationProbe,
) -> Result<[&ProofPayloadValueRef; 2], String> {
    let rule = operation
        .rule
        .as_ref()
        .ok_or_else(|| format!("right-root Result {} has no typed rule", operation.output_event))?;
    match rule {
        ProofPayloadRule::Sum { inputs } => {
            let [left, right] = inputs.as_slice() else {
                return Err(format!(
                    "right-root Result {} sum does not have two inputs",
                    operation.output_event
                ));
            };
            Ok([left, right])
        }
        ProofPayloadRule::Product { left, right, .. } |
        ProofPayloadRule::Tensor { left, right, .. } => Ok([left, right]),
        _ => Err(format!(
            "right-root Result {} has an unsupported operation rule",
            operation.output_event
        )),
    }
}

fn render_right_root_predecessor_premise(
    source: &mut String,
    name: &str,
    owner: ProofPayloadOwner,
    value: &ProofPayloadValueRef,
    expected_source: u64,
    index: &PayloadIndex,
) -> Result<bool, String> {
    match value {
        ProofPayloadValueRef::Result { event, .. } if *event == expected_source => Ok(false),
        ProofPayloadValueRef::Predecessor { binding_event, input_position, .. } => {
            let ProofPayloadEvent::Predecessor {
                consumer,
                input_position: row_position,
                predecessor,
                source_result,
            } = index.event(*binding_event)?
            else {
                return Err(format!(
                    "right-root predecessor reference {binding_event} is not a Predecessor event"
                ));
            };
            if *consumer != owner ||
                row_position != input_position ||
                *source_result != expected_source
            {
                return Err(format!(
                    "right-root predecessor reference {binding_event} does not match its operation input"
                ));
            }
            writeln!(
                source,
                "    ({name} : (history.lookup {binding_event}).map AnnotatedEvent.event =\n      some (.predecessor owner {input_position} {} {source_result}))",
                format!("⟨{predecessor}⟩")
            )
            .expect("String write");
            Ok(true)
        }
        ProofPayloadValueRef::Result { event, .. } => Err(format!(
            "right-root direct Result reference {event} does not match dependency {expected_source}"
        )),
        ProofPayloadValueRef::Transfer(event) => Err(format!(
            "right-root Result {} uses unsupported transfer reference {event}",
            expected_source
        )),
    }
}

fn render_right_root_node(
    source: &mut String,
    node: &RightRootNode,
    index: &PayloadIndex,
    modulus: &str,
) -> Result<(), String> {
    let event = node.result.event;
    writeln!(source, "namespace SemanticRightRootResult{event}\n").expect("String write");
    match &node.kind {
        RightRootNodeKind::Terminal { producer_event, frame_start, rule, term } => {
            writeln!(source, "def owner : Owner := {}", owner_text(node.result.owner))
                .expect("String write");
            writeln!(source, "def rawTerms : List Term := [{}]", raw_term_text(term))
                .expect("String write");
            source.push_str("def summary : Bound := .exactZero\n");
            writeln!(source, "def producerEvent : Nat := {producer_event}").expect("String write");
            writeln!(source, "def resultEvent : Nat := {event}").expect("String write");
            writeln!(
                source,
                "def actual (selector : Nat)\n    (witness : Witness document history (some selector) {modulus}) : Int :=\n  witness.honestTerminalActual resultEvent"
            )
            .expect("String write");
            writeln!(
                source,
                "theorem terminalAt (selector : Nat) (selectorLower : 0 ≤ selector)\n    (selectorUpper : selector < 32) :\n    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by\n  refine ⟨by decide, rightRootOwnerAtSelector214 selector selectorLower selectorUpper _, ?_⟩\n  refine ⟨{}, {frame_start}, {}, ?_, ?_⟩\n  · rfl\n  · rfl",
                reached_terminal_rule_text(rule)?,
                reached_terminal_constructor(rule)?,
            )
            .expect("String write");
            writeln!(
                source,
                "theorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)\n    (selectorUpper : selector < 32)\n    (witness : Witness document history (some selector) {modulus}) :\n    ExactClaimAt history {modulus} witness.env resultEvent owner\n      (actual selector witness) rawTerms summary := by\n  exact terminalExactClaimAt witness\n    (terminalAt selector selectorLower selectorUpper)"
            )
            .expect("String write");
        }
        RightRootNodeKind::Operation(operation) => {
            if !operation.composite_relations.is_empty() {
                return Err(format!(
                    "right-root Result {event} unexpectedly requires {} relation rewrites",
                    operation.composite_relations.len()
                ));
            }
            for input in &operation.inputs {
                if !matches!(
                    input.summary.coefficient_bound(),
                    crate::operational_noise::facts::NumericContract::Known(
                        crate::operational_noise::facts::CoefficientBound::ExactZero
                    )
                ) {
                    return Err(format!(
                        "right-root operation Result {event} input {} is not exact-zero",
                        input.event
                    ));
                }
            }
            let left_event = operation.input_events[0];
            let right_event = operation.input_events[1];
            render_right_root_operation(source, operation, left_event, right_event);
            let operator = match operation.kind {
                OperationKind::Add => "+",
                OperationKind::Subtract => "-",
                OperationKind::Multiply | OperationKind::Tensor => "*",
                OperationKind::Direct => {
                    return Err(format!("right-root Result {event} is a direct operation"));
                }
            };
            writeln!(
                source,
                "def actual (selector : Nat)\n    (witness : Witness document history (some selector) {modulus}) : Int :=\n  SemanticRightRootResult{left_event}.actual selector witness {operator}\n    SemanticRightRootResult{right_event}.actual selector witness"
            )
            .expect("String write");
            let theorem = match operation.kind {
                OperationKind::Add => "exactValueClaim_add_of_mod_zero",
                OperationKind::Subtract => "exactValueClaim_sub_exactZero_of_mod_zero",
                OperationKind::Multiply | OperationKind::Tensor => {
                    "exactValueClaim_product_of_mod_zero"
                }
                OperationKind::Direct => unreachable!(),
            };
            let refs = right_root_operation_refs(operation)?;
            let expression_row = node.result.owner.expression_row;
            let expression_module = expression_row / 256;
            writeln!(
                source,
                "theorem claimOfHistory (selector : Nat)\n    (witness : Witness document history (some selector) {modulus})\n    (expressionAt : document.expressions.lookup {expression_row} =\n      some {NAMESPACE}.Cert.Expression{expression_module:03}.ExpressionRow{expression_row})"
            )
            .expect("String write");
            let left_predecessor = render_right_root_predecessor_premise(
                source,
                "leftPredecessorAt",
                node.result.owner,
                refs[0],
                left_event,
                index,
            )?;
            let right_predecessor = render_right_root_predecessor_premise(
                source,
                "rightPredecessorAt",
                node.result.owner,
                refs[1],
                right_event,
                index,
            )?;
            writeln!(
                source,
                "    (ruleAt : (history.lookup {}).map AnnotatedEvent.event =\n      some (.boundTransfer owner ({})))\n    (leftClaim : ExactClaimAt history {modulus} witness.env\n      SemanticRightRootResult{left_event}.resultEvent\n      SemanticRightRootResult{left_event}.owner\n      (SemanticRightRootResult{left_event}.actual selector witness)\n      SemanticRightRootResult{left_event}.rawTerms\n      SemanticRightRootResult{left_event}.summary)\n    (rightClaim : ExactClaimAt history {modulus} witness.env\n      SemanticRightRootResult{right_event}.resultEvent\n      SemanticRightRootResult{right_event}.owner\n      (SemanticRightRootResult{right_event}.actual selector witness)\n      SemanticRightRootResult{right_event}.rawTerms\n      SemanticRightRootResult{right_event}.summary)\n    (outputAt : (history.lookup resultEvent).map AnnotatedEvent.event =\n      some (.resultExact owner rawTerms summary)) :\n    ExactClaimAt history {modulus} witness.env resultEvent owner\n      (actual selector witness) rawTerms summary := by\n  refine ⟨outputAt, ?_⟩\n  exact {theorem} {modulus} witness.env\n    (SemanticRightRootResult{left_event}.actual selector witness)\n    (SemanticRightRootResult{right_event}.actual selector witness) left right output\n    (by simpa [left, leftRaw, SemanticRightRootResult{left_event}.summary] using leftClaim.claim)\n    (by simpa [right, rightRaw, SemanticRightRootResult{right_event}.summary] using rightClaim.claim)\n    (resultSound witness.env) (by decide)",
                operation.rule_event,
                rule_text(operation.rule.as_ref().expect("typed right-root rule")),
            )
            .expect("String write");
            writeln!(
                source,
                "\ntheorem claimSound (selector : Nat) (selectorLower : 0 ≤ selector)\n    (selectorUpper : selector < 32)\n    (witness : Witness document history (some selector) {modulus}) :\n    ExactClaimAt history {modulus} witness.env resultEvent owner\n      (actual selector witness) rawTerms summary := by\n  apply claimOfHistory selector witness (by rfl){}{} (by rfl)\n  · exact SemanticRightRootResult{left_event}.claimSound\n      selector selectorLower selectorUpper witness\n  · exact SemanticRightRootResult{right_event}.claimSound\n      selector selectorLower selectorUpper witness\n  · rfl",
                if left_predecessor { " (by rfl)" } else { "" },
                if right_predecessor { " (by rfl)" } else { "" },
            )
            .expect("String write");
        }
    }
    writeln!(source, "\nend SemanticRightRootResult{event}\n").expect("String write");
    Ok(())
}

fn render_right_root(
    statement: &CertificateDocumentV1,
    index: &PayloadIndex,
    relation_probes: &[RelationProbe],
    modulus: &str,
) -> Result<(String, Vec<super::super::TallSecurity0GeneratedFile>), String> {
    const CHUNK_SIZE: usize = 16;
    let nodes = right_root_nodes(statement, index, relation_probes)?;
    if nodes.len() != 571 {
        return Err(format!(
            "Security0 right-root closure has {} Results, expected 571",
            nodes.len()
        ));
    }
    let terminal_count =
        nodes.iter().filter(|node| matches!(node.kind, RightRootNodeKind::Terminal { .. })).count();
    if terminal_count != 221 {
        return Err(format!(
            "Security0 right-root closure has {terminal_count} terminals, expected 221"
        ));
    }
    let mut files = Vec::new();
    for (shard_index, shard) in nodes.chunks(CHUNK_SIZE).enumerate() {
        let module = format!("SemanticRightRootShard{shard_index:03}");
        let previous = shard_index
            .checked_sub(1)
            .map(|index| format!("import {NAMESPACE}.Semantic.SemanticRightRootShard{index:03}\n"));
        let mut source = previous.unwrap_or_else(|| {
            format!(
                "import Mxx.Certificate.OperationalNoise.TallSemantics\nimport {NAMESPACE}.Proof.History\n"
            )
        });
        source
            .push_str("\nset_option autoImplicit false\nset_option relaxedAutoImplicit false\n\n");
        writeln!(source, "namespace {NAMESPACE}.Semantic\n").expect("String write");
        source.push_str(
            "open Mxx.Certificate.OperationalNoise\nopen TallSecurity0ABI\nopen TallSemantics\n\n",
        );
        if shard_index == 0 {
            source.push_str(
                "theorem rightRootOwnerAtSelector214 (selector : Nat) (selectorLower : 0 ≤ selector)\n    (selectorUpper : selector < 32) (expression : SchemaV1.ExpressionRef) :\n    ownerAtSelector document (some selector) ⟨.program ⟨214⟩, expression⟩ := by\n  simp [ownerAtSelector, document, selectorLower, selectorUpper]\n\n",
            );
        }
        for node in shard {
            render_right_root_node(&mut source, node, index, modulus)?;
        }
        writeln!(source, "end {NAMESPACE}.Semantic").expect("String write");
        files.push(generated_file(format!("Semantic/{module}.lean"), source));
    }
    let last_shard = nodes.len().div_ceil(CHUNK_SIZE) - 1;
    let mut source = format!(
        "import {NAMESPACE}.Semantic.SemanticRightRootShard{last_shard:03}\n\nset_option autoImplicit false\nset_option relaxedAutoImplicit false\n\nnamespace {NAMESPACE}.Semantic.SemanticRightRoot\n\nopen Mxx.Certificate.OperationalNoise\nopen TallSecurity0ABI\nopen TallSemantics\n\n"
    );
    writeln!(
        source,
        "/-- The generated theorem application for the reached right exact-zero root. -/\ntheorem rightRootClaimSound (selector : Nat) (selectorLower : 0 ≤ selector)\n    (selectorUpper : selector < 32)\n    (witness : Witness document history (some selector) {modulus}) :\n    ExactClaimAt history {modulus} witness.env\n      SemanticRightRootResult6275.resultEvent SemanticRightRootResult6275.owner\n      (SemanticRightRootResult6275.actual selector witness)\n      SemanticRightRootResult6275.rawTerms SemanticRightRootResult6275.summary := by\n  exact SemanticRightRootResult6275.claimSound selector selectorLower selectorUpper witness"
    )
    .expect("String write");
    source.push_str(&format!("\n\nend {NAMESPACE}.Semantic.SemanticRightRoot\n"));
    Ok((source, files))
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
    fn reached_terminal_rules_match_the_fixed_security0_boundary() {
        use crate::operational_noise::simulation::{ProofPayloadAuthority, ProofPayloadScale};

        let value =
            ProofPayloadValueRef::Result { event: 3, projection: BoundProjection::Coefficient };
        let accepted = [
            ProofPayloadRule::Authority(ProofPayloadAuthority::FactStore),
            ProofPayloadRule::Authority(ProofPayloadAuthority::ProgramFamilyFact),
            ProofPayloadRule::Authority(ProofPayloadAuthority::Operator),
            ProofPayloadRule::Identity { input: value.clone() },
            ProofPayloadRule::Scale {
                value: value.clone(),
                scale: ProofPayloadScale::Magnitude(BigUint::from(2_u8)),
            },
        ];
        assert!(accepted.iter().all(reached_terminal_rule));
        assert!(!reached_terminal_rule(&ProofPayloadRule::Sum { inputs: vec![value] }));
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
            "257",
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
            rule_event: 1,
            input_events: [2, 3],
            output_event: 4,
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
            rule: Some(ProofPayloadRule::Sum {
                inputs: vec![
                    ProofPayloadValueRef::Result {
                        event: 2,
                        projection: BoundProjection::Coefficient,
                    },
                    ProofPayloadValueRef::Result {
                        event: 3,
                        projection: BoundProjection::Coefficient,
                    },
                ],
            }),
            composite_relations: Vec::new(),
        };
        let mut operation_source = String::new();
        render_operation(&mut operation_source, &operation, "257");
        assert!(operation_source.contains("CanonicalAgreement"));
        assert!(operation_source.contains("addCanonicalResultSound"));
        assert!(operation_source.contains("selectedSumRuleEvent"));
        assert!(operation_source.contains("selectedLeftResultEvent"));
        assert!(operation_source.contains("selectedRightResultEvent"));
        assert!(operation_source.contains("selectedResultEvent"));
        assert!(!operation_source.contains("CoefficientAgreement"));
        assert!(!operation_source.contains("expected :"));

        let bound = BoundProbe {
            root_result_event: 1,
            prefold_event: 2,
            end_event: 3,
            survivor_events: (0..64).map(|event| event + 4).collect(),
            root: ResultRecord {
                event: 1,
                owner: root,
                terms: vec![],
                summary: BoundedSummary::zero(),
            },
            prefold_terms: vec![],
            prefold_summary: BoundedSummary::zero(),
            prefold_evidence: None,
            end: ResultRecord {
                event: 3,
                owner: root,
                terms: vec![],
                summary: BoundedSummary::zero(),
            },
            survivor_contributions: (1..=64).map(|value| value.to_string()).collect(),
            survivor_bounds: (1..=64).map(|value| (value + 1).to_string()).collect(),
        };
        let mut bound_source = String::new();
        render_bound(&mut bound_source, &bound, "257");
        assert!(bound_source.contains("theorem survivorBoundsSound"));
        assert!(bound_source.contains("List.Forall₂"));
        assert!(bound_source.contains("constructor"));
        assert!(bound_source.contains("· omega"));
        assert!(bound_source.contains("survivorContributionsChunk3"));
        assert!(bound_source.contains("survivorContributionsTree"));
        assert!(bound_source.contains("forall₂_append"));
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
        let (_, refs, _, _) = typed_operation_rule(&index, &result, OperationKind::Add)
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

    fn add_result(root: ProofPayloadOwner, event: u64, term_count: usize) -> ResultRecord {
        let monomial =
            ProofPayloadMonomial { central_factors: vec![root], ordered_factors: vec![] };
        ResultRecord {
            event,
            owner: root,
            terms: (0..term_count)
                .map(|index| ProofPayloadTerm {
                    monomial: ProofPayloadMonomial {
                        central_factors: monomial.central_factors.clone(),
                        ordered_factors: vec![owner(index as u64)],
                    },
                    coefficient: BigInt::from(1),
                })
                .collect(),
            summary: BoundedSummary::zero(),
        }
    }

    fn add_operation(output: ResultRecord) -> OperationProbe {
        let left = add_result(output.owner, output.event - 4, 1);
        let right = add_result(output.owner, output.event - 3, 1);
        let left_event = left.event;
        let right_event = right.event;
        OperationProbe {
            kind: OperationKind::Add,
            rule_event: output.event - 2,
            input_events: [left.event, right.event],
            output_event: output.event,
            inputs: vec![left, right],
            output,
            scalar_left: false,
            scalar_right: false,
            raw_work: 1,
            rule: Some(ProofPayloadRule::Sum {
                inputs: vec![
                    ProofPayloadValueRef::Result {
                        event: left_event,
                        projection: BoundProjection::Coefficient,
                    },
                    ProofPayloadValueRef::Result {
                        event: right_event,
                        projection: BoundProjection::Coefficient,
                    },
                ],
            }),
            composite_relations: Vec::new(),
        }
    }

    #[test]
    fn add_result_render_retains_typed_event_ids() {
        let root = owner(7);
        let output_result = add_result(root, 5327, 3);
        let operation = add_operation(output_result.clone());
        let probes = vec![ProbeSelection {
            name: "add-chain",
            event: output_result.event,
            owner: output_result.owner,
            score: output_result.terms.len() as u64,
            detail: "actual maximum intermediate Add Result",
            frame_start: None,
            frame_end: None,
            long_key: None,
            operation: Some(operation),
            relations: Vec::new(),
            bound: None,
        }];
        let source = render_probe("Semantic002", "add-chain", &probes, "257");
        assert!(source.contains("namespace AddResult"));
        assert!(source.contains("def selectedSumRuleEvent : Nat := 5325"));
        assert!(source.contains("def selectedLeftResultEvent : Nat := 5323"));
        assert!(source.contains("def selectedRightResultEvent : Nat := 5324"));
        assert!(source.contains("def selectedResultEvent : Nat := 5327"));
        assert!(source.contains("addCanonicalResultSound"));
    }

    #[test]
    fn composite_operation_renders_product_then_relation_chain() {
        let root = owner(7);
        let output = add_result(root, 20, 1);
        let mut operation = add_operation(output);
        operation.rule_event = 10;
        operation.output_event = 20;
        let relations = [
            relation_probe(RelationRuleKind::Gadget, 11),
            relation_probe(RelationRuleKind::Gadget, 12),
        ];
        let operation = attach_composite_relations(operation, &relations).expect("two relations");
        let mut source = String::new();
        render_operation(&mut source, &operation, "257");
        assert!(source.contains("def expected1 : Polynomial Owner := relationPoly expected0"));
        assert!(source.contains("def expected2 : Polynomial Owner := relationPoly expected1"));
        assert!(source.contains("relationSound1"));
        assert!(source.contains(
            "have outputSound := canonicalAgreement_eval env output expected2 resultAgreement"
        ));
        assert!(source.contains(":= by rw [outputSound]"));
        assert!(
            source.contains("_ = evalPolynomial env expected0 % Int.ofNat 257 := relationSound0")
        );
        assert!(source.contains("productCanonicalResultSound"));
        assert!(source.contains("relationCanonicalResultSound"));
        assert!(source.contains("evalPolynomial env output % Int.ofNat 257"));
    }

    fn relation_probe(kind: RelationRuleKind, event: u64) -> RelationProbe {
        let root = owner(7);
        let record =
            ResultRecord { event, owner: root, terms: Vec::new(), summary: BoundedSummary::zero() };
        RelationProbe {
            event,
            owner: root,
            frame_start: 0,
            frame_end: 1,
            source: ProofPayloadMonomial {
                central_factors: Vec::new(),
                ordered_factors: vec![root],
            },
            lhs: ProofPayloadMonomial { central_factors: Vec::new(), ordered_factors: vec![root] },
            outer: BigInt::from(1),
            start: 0,
            end: 1,
            accumulator: record.clone(),
            rhs: record.clone(),
            output: record,
            kind,
            rule: match kind {
                RelationRuleKind::Gadget => ProofPayloadRelationRule::Gadget {
                    gadget: root,
                    decomposition: root,
                    input: 0,
                    input_result: event,
                },
                RelationRuleKind::Universal => ProofPayloadRelationRule::Universal {
                    computed: event,
                    lhs: ProofPayloadMonomial {
                        central_factors: Vec::new(),
                        ordered_factors: vec![root],
                    },
                    lhs_layout: None,
                    rhs_result: event,
                },
            },
            output_merge: ProofPayloadCoefficientMerge {
                owner: root,
                source: ProofPayloadCoefficientMergeSource::Relation {
                    application: event,
                    source_term_ordinal: 0,
                },
                output: ProofPayloadMonomial {
                    central_factors: Vec::new(),
                    ordered_factors: vec![root],
                },
                signed_contribution: BigInt::from(1),
            },
            rhs_pre_fold_event: None,
        }
    }

    #[test]
    fn relation_selection_requires_gadget_and_universal() {
        let gadget_error =
            match select_relation_probes(vec![relation_probe(RelationRuleKind::Gadget, 1)]) {
                Ok(_) => panic!("Universal relation must be required"),
                Err(error) => error,
            };
        assert!(gadget_error.contains("Universal"));

        let selected = select_relation_probes(vec![
            relation_probe(RelationRuleKind::Gadget, 1),
            relation_probe(RelationRuleKind::Universal, 2),
        ])
        .expect("both relation kinds");
        assert_eq!(selected.len(), 2);
        assert!(selected.iter().any(|probe| probe.kind == RelationRuleKind::Gadget));
        assert!(selected.iter().any(|probe| probe.kind == RelationRuleKind::Universal));
    }

    #[test]
    fn relation_render_uses_statement_ciphertext_modulus() {
        let modulus = "100418593683253592432016548326729029359133068138294319235841";
        let relation = relation_probe(RelationRuleKind::Universal, 2);
        let mut source = String::new();
        render_relation(&mut source, &relation, modulus);
        assert!(source.contains(&format!("Int.ofNat {modulus}")));
        assert!(!source.contains("Int.ofNat 257"));
    }

    #[test]
    fn invalid_ciphertext_modulus_is_rejected() {
        let mut statement = CertificateDocumentV1 {
            schema_id: "mxx.operational-noise.certificate",
            schema_version: 1,
            plaintext_modulus: "2".to_owned(),
            ciphertext_modulus: "not-a-decimal".to_owned(),
            ring_dimension: 1,
            expressions: Vec::new(),
            programs: Vec::new(),
            sources: Vec::new(),
            events: Vec::new(),
            index_uses: Vec::new(),
            slice_groups: Vec::new(),
            residual_root:
                crate::operational_noise::certificate_schema::CertificateResidualRootV1::Closed {
                    expression: 0,
                },
        };
        let error = ciphertext_modulus_text(&statement).expect_err("invalid modulus must fail");
        assert!(error.contains("positive decimal ciphertext modulus"));
        statement.ciphertext_modulus = "0".to_owned();
        assert!(ciphertext_modulus_text(&statement).is_err());
    }

    #[test]
    fn expression_kind_reports_missing_owner_rows() {
        let statement = CertificateDocumentV1 {
            schema_id: "mxx.operational-noise.certificate",
            schema_version: 1,
            plaintext_modulus: "2".to_owned(),
            ciphertext_modulus: "257".to_owned(),
            ring_dimension: 1,
            expressions: Vec::new(),
            programs: Vec::new(),
            sources: Vec::new(),
            events: Vec::new(),
            index_uses: Vec::new(),
            slice_groups: Vec::new(),
            residual_root:
                crate::operational_noise::certificate_schema::CertificateResidualRootV1::Closed {
                    expression: 0,
                },
        };
        let error = expression_kind(&statement, owner(91)).expect_err("missing row must fail");
        assert!(error.contains("missing expression row 91"));
    }

    #[test]
    fn product_merge_scalar_inference_uses_typed_output_key() {
        let left =
            ProofPayloadMonomial { central_factors: vec![], ordered_factors: vec![owner(5506)] };
        let right =
            ProofPayloadMonomial { central_factors: vec![], ordered_factors: vec![owner(6544)] };
        let output = ProofPayloadMonomial {
            central_factors: vec![owner(5506)],
            ordered_factors: vec![owner(6544)],
        };
        let merge = [(left.clone(), right.clone(), output.clone())];
        let candidates = matching_scalar_flags_from_merges(&merge);
        assert_eq!(candidates, vec![(true, false)]);
        let inputs = vec![
            ResultRecord {
                event: 1,
                owner: owner(5506),
                terms: vec![ProofPayloadTerm { monomial: left, coefficient: BigInt::from(1) }],
                summary: BoundedSummary::zero(),
            },
            ResultRecord {
                event: 2,
                owner: owner(6544),
                terms: vec![ProofPayloadTerm { monomial: right, coefficient: BigInt::from(1) }],
                summary: BoundedSummary::zero(),
            },
        ];
        assert_eq!(resolve_scalar_flags(&candidates, &inputs), Ok((true, false)));

        let central_left =
            ProofPayloadMonomial { central_factors: vec![owner(5506)], ordered_factors: vec![] };
        let central_right =
            ProofPayloadMonomial { central_factors: vec![owner(6544)], ordered_factors: vec![] };
        let central_output = ProofPayloadMonomial {
            central_factors: vec![owner(5506), owner(6544)],
            ordered_factors: vec![],
        };
        let central_merge = [(central_left, central_right, central_output)];
        let central_candidates = matching_scalar_flags_from_merges(&central_merge);
        assert_eq!(central_candidates.len(), 4);
        let central_inputs = vec![
            ResultRecord {
                event: 3,
                owner: owner(5506),
                terms: vec![ProofPayloadTerm {
                    monomial: central_merge[0].0.clone(),
                    coefficient: BigInt::from(1),
                }],
                summary: BoundedSummary::zero(),
            },
            ResultRecord {
                event: 4,
                owner: owner(6544),
                terms: vec![ProofPayloadTerm {
                    monomial: central_merge[0].1.clone(),
                    coefficient: BigInt::from(1),
                }],
                summary: BoundedSummary::zero(),
            },
        ];
        assert_eq!(resolve_scalar_flags(&central_candidates, &central_inputs), Ok((false, false)));
    }

    #[test]
    fn semantic_shard_module_names_are_deterministic() {
        assert_eq!(shard_module_name(0), "SemanticShard000");
        assert_eq!(shard_module_name(59), "SemanticShard059");
        assert_eq!(shard_module_name(1001), "SemanticShard1001");
    }
}
