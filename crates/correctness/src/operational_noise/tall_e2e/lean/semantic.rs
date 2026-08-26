use super::{NAMESPACE, closure::ReachedSemanticSlice, generated_file};
use crate::operational_noise::{
    certificate_schema::{CertificateDocumentV1, CertificateResidualRootV1},
    g0::{
        BoundProjection, CanonicalExpressionDescriptor, CanonicalExpressionOperator,
        StableMatrixOperation, StableOperator,
    },
    simulation::{
        OperationalProofPayload, ProofPayloadAuthority, ProofPayloadCoefficientMerge,
        ProofPayloadCoefficientMergeSource, ProofPayloadEvent, ProofPayloadMonomial,
        ProofPayloadOwner, ProofPayloadRelationRule, ProofPayloadRule, ProofPayloadTerm,
        ProofPayloadValue, ProofPayloadValueRef,
    },
};
use num_bigint::BigUint;
use num_traits::Zero;
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
};

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
    output_merges: Vec<(u64, ProofPayloadCoefficientMerge)>,
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
    output_merges: Vec<(u64, ProofPayloadCoefficientMerge)>,
    last_merge_event: Option<u64>,
    kind: RelationRuleKind,
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

struct LeftRootBoundNode<'a> {
    event: u64,
    owner: ProofPayloadOwner,
    rule: &'a ProofPayloadRule,
}

struct LeftRootMergeNode<'a> {
    event: u64,
    result_event: u64,
    merge: &'a ProofPayloadCoefficientMerge,
}

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
enum LeftMergeGroupKey {
    Operator { frame: u64, owner: ProofPayloadOwner, left_result: u64, right_result: u64 },
    Relation { frame: u64, owner: ProofPayloadOwner, application: u64 },
}

struct LeftRootRenderData<'a> {
    bounds: Vec<LeftRootBoundNode<'a>>,
    merges: Vec<LeftRootMergeNode<'a>>,
    authority_results: BTreeMap<u64, u64>,
}

struct FinalAddRenderData {
    owner: ProofPayloadOwner,
    left: ResultRecord,
    right: ResultRecord,
    merge_events: Vec<u64>,
    result: ResultRecord,
    prefold_event: u64,
    end_event: u64,
}

#[derive(Clone)]
enum LeftClaimStart {
    RightRoot,
    Terminal {
        producer_event: u64,
    },
    Operator {
        first_merge_event: Option<u64>,
        transfer_event: u64,
        kind: OperationKind,
        left_result: u64,
        right_result: u64,
    },
    PriorResult {
        result_event: u64,
    },
}

#[derive(Clone)]
struct LeftClaimNode {
    result: ResultRecord,
    start: LeftClaimStart,
    relations: Vec<u64>,
}

fn left_claim_dependencies(node: &LeftClaimNode) -> Vec<u64> {
    match node.start {
        LeftClaimStart::RightRoot | LeftClaimStart::Terminal { .. } => Vec::new(),
        LeftClaimStart::Operator { left_result, right_result, .. } => {
            if left_result == right_result {
                vec![left_result]
            } else {
                vec![left_result, right_result]
            }
        }
        LeftClaimStart::PriorResult { result_event } => vec![result_event],
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
                    value: ProofPayloadValue::Exact { terms, summary, .. },
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
                    result:
                        ProofPayloadValue::Exact {
                            terms,
                            coefficient_bound,
                            coefficient_producer,
                            summary,
                            summary_producer,
                        },
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
                        ProofPayloadValue::Exact {
                            terms: terms.clone(),
                            coefficient_bound: coefficient_bound.clone(),
                            coefficient_producer: *coefficient_producer,
                            summary: summary.clone(),
                            summary_producer: *summary_producer,
                        },
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
    Ok(OperationProbe {
        kind,
        rule_event,
        input_events: [inputs[0].event, inputs[1].event],
        output_event: result.event,
        inputs,
        output: result.clone(),
        scalar_left: scalar_flags.0,
        scalar_right: scalar_flags.1,
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
            output_merges: relation.output_merges,
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
                    value: ProofPayloadValue::Exact { terms, summary, .. },
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
                    result: ProofPayloadValue::Exact { terms, summary, .. },
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
                        output_merges: Vec::new(),
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
                    relation.output_merges.push((event, merge.clone()));
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
    slice: &ReachedSemanticSlice,
) -> Result<Vec<super::super::TallSecurity0GeneratedFile>, String> {
    let modulus = ciphertext_modulus_text(statement)?;
    let index = PayloadIndex::new(proof)?;
    let mut left_root = collect_left_root_render_data(proof, &index, slice.left_result)?;
    let final_add = collect_final_add_render_data(&index, slice)?;
    for &event in &final_add.merge_events {
        let ProofPayloadEvent::CoefficientMerge(merge) = index.event(event)? else {
            unreachable!("validated final Add merge event");
        };
        left_root.merges.push(LeftRootMergeNode {
            event,
            result_event: final_add.result.event,
            merge,
        });
    }
    let ranges = frame_ranges(proof)?;
    let relation_probes = relation_candidates(&index, &ranges)?;
    let left_claim_nodes = collect_left_claim_nodes(
        statement,
        &index,
        &mut left_root,
        &relation_probes,
        slice.left_result,
        slice.right_result,
    )?;
    let mut files = Vec::new();
    let (right_root_index, right_root_shards) =
        render_right_root(statement, &index, &relation_probes, &modulus, slice.right_result)?;
    files.push(generated_file("Semantic/SemanticRightRoot.lean", right_root_index));
    files.extend(right_root_shards);
    files.extend(render_left_authorities(&index, &left_root, &modulus)?);
    files.extend(render_left_bounds(statement, &index, &left_root, &modulus)?);
    files.extend(render_left_merge_deltas(statement, &index, &left_root, &relation_probes)?);
    files.extend(render_left_claims(
        statement,
        &index,
        &left_root,
        &left_claim_nodes,
        &relation_probes,
        &modulus,
        slice,
    )?);
    files.push(render_final_add(statement, &index, &final_add, &modulus, slice)?);

    files.push(generated_file(
        "Semantic/Semantic.lean",
        format!("import {NAMESPACE}.Semantic.SemanticFinal\n"),
    ));
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
            ProofPayloadRule::MonomialProduct { .. } |
            ProofPayloadRule::Product { .. }
    )
}

fn rule_references_transfer(rule: &ProofPayloadRule, transfer: u64) -> bool {
    let reference_matches = |reference: &ProofPayloadValueRef| matches!(reference, ProofPayloadValueRef::Transfer(event) if *event == transfer);
    match rule {
        ProofPayloadRule::Authority(_) => false,
        ProofPayloadRule::Identity { input } => reference_matches(input),
        ProofPayloadRule::Sum { inputs } |
        ProofPayloadRule::Maximum { inputs } |
        ProofPayloadRule::WeightedSum { inputs } => inputs.iter().any(reference_matches),
        ProofPayloadRule::Scale { value, scale } => {
            reference_matches(value) ||
                matches!(scale, crate::operational_noise::simulation::ProofPayloadScale::Value(reference) if reference_matches(reference))
        }
        ProofPayloadRule::MonomialProduct { factors, .. } => {
            factors.iter().any(|factor| reference_matches(&factor.bound))
        }
        ProofPayloadRule::Product { left, right, .. } |
        ProofPayloadRule::Tensor { left, right, .. } => {
            reference_matches(left) || reference_matches(right)
        }
    }
}

fn reached_bound_references(rule: &ProofPayloadRule) -> Vec<&ProofPayloadValueRef> {
    match rule {
        ProofPayloadRule::Authority(_) => Vec::new(),
        ProofPayloadRule::Identity { input } => vec![input],
        ProofPayloadRule::Sum { inputs } => inputs.iter().collect(),
        ProofPayloadRule::Scale { value, scale } => {
            let mut references = vec![value];
            if let crate::operational_noise::simulation::ProofPayloadScale::Value(reference) = scale
            {
                references.push(reference);
            }
            references
        }
        ProofPayloadRule::MonomialProduct { factors, .. } => {
            factors.iter().map(|factor| &factor.bound).collect()
        }
        ProofPayloadRule::Product { left, right, .. } => vec![left, right],
        _ => Vec::new(),
    }
}

fn reached_bound_reference_producer(
    index: &PayloadIndex,
    consumer: ProofPayloadOwner,
    reference: &ProofPayloadValueRef,
) -> Result<u64, String> {
    if let ProofPayloadValueRef::Transfer(event) = reference {
        return match index.event(*event)? {
            ProofPayloadEvent::BoundTransfer { owner, .. } if owner == &consumer => Ok(*event),
            ProofPayloadEvent::BoundTransfer { .. } => {
                Err(format!("direct transfer reference {event} has a different owner"))
            }
            _ => Err(format!("direct transfer reference {event} is not a BoundTransfer")),
        };
    }
    let projection = match reference {
        ProofPayloadValueRef::Result { projection, .. } |
        ProofPayloadValueRef::Predecessor { projection, .. } => projection,
        ProofPayloadValueRef::Transfer(_) => unreachable!(),
    };
    let result_event = reached_bound_result_event(index, consumer, reference)?;
    match (projection, index.event(result_event)?) {
        (
            BoundProjection::Coefficient,
            ProofPayloadEvent::Result { value: ProofPayloadValue::Coefficient { .. }, .. },
        ) => result_event
            .checked_sub(1)
            .ok_or_else(|| format!("Result {result_event} has no coefficient producer")),
        (
            BoundProjection::Coefficient,
            ProofPayloadEvent::Result {
                value: ProofPayloadValue::Exact { coefficient_producer, .. },
                ..
            },
        ) => Ok(*coefficient_producer),
        (
            BoundProjection::Summary,
            ProofPayloadEvent::Result {
                value: ProofPayloadValue::Exact { summary_producer: Some(producer), .. },
                ..
            },
        ) => Ok(*producer),
        (BoundProjection::Summary, ProofPayloadEvent::Result { .. }) => Err(format!(
            "Security0 reached bound reference {reference:?} has no event-indexed summary producer"
        )),
        (_, _) => unreachable!("reached bound reference identifies a Result"),
    }
}

fn reached_bound_result_event(
    index: &PayloadIndex,
    consumer: ProofPayloadOwner,
    reference: &ProofPayloadValueRef,
) -> Result<u64, String> {
    let event = match reference {
        ProofPayloadValueRef::Result { event, .. } => *event,
        ProofPayloadValueRef::Predecessor { binding_event, input_position, .. } => {
            let (binding_consumer, position, source_result) = index
                .predecessors
                .get(binding_event)
                .ok_or_else(|| format!("bound predecessor {binding_event} is missing"))?;
            if binding_consumer != &consumer || position != input_position {
                return Err(format!(
                    "bound predecessor {binding_event} owner/input does not match its consumer"
                ));
            }
            *source_result
        }
        ProofPayloadValueRef::Transfer(event) => {
            return Err(format!("unsupported direct transfer reference {event}"));
        }
    };
    match index.event(event)? {
        ProofPayloadEvent::Result { .. } => Ok(event),
        _ => Err(format!("payload reference {event} does not identify a Result")),
    }
}

fn collect_left_authority_results(
    index: &PayloadIndex,
    bounds: &[LeftRootBoundNode<'_>],
) -> Result<BTreeMap<u64, u64>, String> {
    let authority_events = bounds
        .iter()
        .filter_map(|bound| {
            matches!(bound.rule, ProofPayloadRule::Authority(_)).then_some(bound.event)
        })
        .collect::<BTreeSet<_>>();
    for bound in bounds {
        if let Some(transfer) = authority_events
            .iter()
            .find(|transfer| rule_references_transfer(bound.rule, **transfer))
        {
            return Err(format!(
                "Security0 reached bound row {} directly consumes authority transfer {transfer}",
                bound.event
            ));
        }
    }
    let mut results = BTreeMap::<u64, Vec<u64>>::new();
    for (position, event) in index.events.iter().enumerate() {
        let result_event =
            u64::try_from(position).map_err(|_| "authority result index overflow")?;
        let (result_owner, producer) = match event {
            ProofPayloadEvent::Result { owner, value: ProofPayloadValue::Coefficient { .. } } => (
                *owner,
                result_event.checked_sub(1).filter(|event| authority_events.contains(event)),
            ),
            ProofPayloadEvent::Result {
                owner,
                value: ProofPayloadValue::Exact { coefficient_producer, .. },
            } => (
                *owner,
                authority_events.contains(coefficient_producer).then_some(*coefficient_producer),
            ),
            _ => continue,
        };
        let Some(producer) = producer else { continue };
        let transfer_frame = index.immediate_frames
            [usize::try_from(producer).map_err(|_| "authority event index overflow")?];
        let result_frame = index.immediate_frames[position];
        if transfer_frame != result_frame {
            return Err(format!(
                "Security0 authority transfer {producer} and Result {} are in different frames",
                result_event
            ));
        }
        let ProofPayloadEvent::BoundTransfer { owner, .. } = index.event(producer)? else {
            unreachable!("authority set contains only bound transfers")
        };
        if *owner != result_owner {
            return Err(format!(
                "Security0 authority transfer {producer} and Result {} have different owners",
                result_event
            ));
        }
        results.entry(producer).or_default().push(result_event);
    }
    authority_events
        .into_iter()
        .map(|event| match results.remove(&event).as_deref() {
            Some([result]) => Ok((event, *result)),
            Some(found) => Err(format!(
                "Security0 authority transfer {event} maps to {} supported Results instead of one",
                found.len()
            )),
            None => Err(format!(
                "Security0 authority transfer {event} has no supported same-frame Result"
            )),
        })
        .collect()
}

fn collect_left_root_render_data<'a>(
    proof: &OperationalProofPayload,
    index: &'a PayloadIndex,
    left_root_event: u64,
) -> Result<LeftRootRenderData<'a>, String> {
    let root = index.result(left_root_event)?;
    if !matches!(
        root.summary.coefficient_bound(),
        crate::operational_noise::facts::NumericContract::Known(
            crate::operational_noise::facts::CoefficientBound::Finite(_)
        )
    ) {
        return Err(format!("reached left-root Result {left_root_event} is not finite"));
    }
    let event_ids = super::closure::collect_security0_event_closure(proof, left_root_event)?;
    let mut relation_events = Vec::new();
    let mut merge_rows = Vec::new();
    let mut exact_results = Vec::new();
    let mut bounds = Vec::new();
    for event in &event_ids {
        match index.event(*event)? {
            ProofPayloadEvent::AppliedRelation { .. } => relation_events.push(*event),
            ProofPayloadEvent::CoefficientMerge(merge) => merge_rows.push((*event, merge)),
            ProofPayloadEvent::Result { value: ProofPayloadValue::Exact { .. }, .. } => {
                exact_results.push(index.result(*event)?)
            }
            ProofPayloadEvent::BoundTransfer { owner, rule } => {
                if !reached_left_bound_rule(rule) {
                    return Err(format!(
                        "Security0 left-root closure reaches unsupported bound rule {rule:?} at event {event}"
                    ));
                }
                bounds.push(LeftRootBoundNode { event: *event, owner: *owner, rule });
            }
            _ => {}
        }
    }
    let merges = merge_rows
        .into_iter()
        .map(|(event, merge)| {
            let frame = index.immediate_frames
                [usize::try_from(event).map_err(|_| "left merge event index overflow")?];
            let result_event = exact_results
                .iter()
                .find(|result| {
                    result.event > event &&
                        result.owner == merge.owner &&
                        index.immediate_frames[usize::try_from(result.event)
                            .expect("indexed left merge Result")] ==
                            frame
                })
                .map(|result| result.event)
                .ok_or_else(|| {
                    format!(
                        "Security0 left coefficient merge {event} has no following same-frame Result"
                    )
                })?;
            Ok(LeftRootMergeNode { event, result_event, merge })
        })
        .collect::<Result<Vec<_>, String>>()?;
    for node in &merges {
        if node.result_event <= node.event ||
            node.merge.owner != index.result(node.result_event)?.owner
        {
            return Err(format!(
                "Security0 left coefficient merge {} is not bound to its following Result {}",
                node.event, node.result_event
            ));
        }
        match &node.merge.source {
            ProofPayloadCoefficientMergeSource::Operator { inputs } => {
                for input in inputs {
                    if input.value_event >= node.event {
                        return Err(format!(
                            "Security0 left operator merge {} has non-prior input Result {}",
                            node.event, input.value_event
                        ));
                    }
                    let result = index.result(input.value_event)?;
                    if usize::try_from(input.term_ordinal)
                        .ok()
                        .is_none_or(|ordinal| ordinal >= result.terms.len())
                    {
                        return Err(format!(
                            "Security0 left operator merge {} has out-of-range term {} for Result {}",
                            node.event, input.term_ordinal, input.value_event
                        ));
                    }
                }
            }
            ProofPayloadCoefficientMergeSource::Relation { application, .. } => {
                if *application >= node.event || !relation_events.contains(application) {
                    return Err(format!(
                        "Security0 left relation merge {} has unavailable application {}",
                        node.event, application
                    ));
                }
            }
        }
    }
    // Expand only the stored Result producers and direct transfer references selected by reached
    // rules; unrelated history remains outside the semantic slice.
    let mut bound_events = bounds.iter().map(|bound| bound.event).collect::<BTreeSet<_>>();
    loop {
        let mut added = Vec::new();
        for event in bound_events.iter().copied().collect::<Vec<_>>() {
            let ProofPayloadEvent::BoundTransfer { owner, rule } = index.event(event)? else {
                unreachable!("left bound event set contains only transfers")
            };
            for reference in reached_bound_references(rule) {
                let producer = reached_bound_reference_producer(index, *owner, reference)?;
                if bound_events.insert(producer) {
                    let ProofPayloadEvent::BoundTransfer { rule, .. } = index.event(producer)?
                    else {
                        return Err(format!(
                            "Security0 reached bound producer {producer} is not a BoundTransfer"
                        ));
                    };
                    if !reached_left_bound_rule(rule) {
                        return Err(format!(
                            "Security0 reached bound producer {producer} has unsupported rule {rule:?}"
                        ));
                    }
                    added.push(producer);
                }
            }
        }
        if added.is_empty() {
            break;
        }
    }
    bounds = bound_events
        .into_iter()
        .map(|event| match index.event(event)? {
            ProofPayloadEvent::BoundTransfer { owner, rule } => {
                Ok(LeftRootBoundNode { event, owner: *owner, rule })
            }
            _ => unreachable!("expanded left bound event set contains only transfers"),
        })
        .collect::<Result<Vec<_>, String>>()?;
    if let Some(bound) = bounds.iter().find(|bound| {
        !reached_left_bound_rule(bound.rule) ||
            !matches!(
                index.event(bound.event),
                Ok(ProofPayloadEvent::BoundTransfer { owner, rule })
                    if *owner == bound.owner && rule == bound.rule
            )
    }) {
        return Err(format!("Security0 left-root bound row {} is inconsistent", bound.event));
    }
    let authority_results = collect_left_authority_results(index, &bounds)?;
    Ok(LeftRootRenderData { bounds, merges, authority_results })
}

fn collect_final_add_render_data(
    index: &PayloadIndex,
    slice: &ReachedSemanticSlice,
) -> Result<FinalAddRenderData, String> {
    let left = index.result(slice.left_result)?;
    let right = index.result(slice.right_result)?;
    let result = index.result(slice.result)?;
    let owner = result.owner;
    for (binding, position, source) in
        [(slice.left_binding, 0, slice.left_result), (slice.right_binding, 1, slice.right_result)]
    {
        if index.predecessors.get(&binding) != Some(&(owner, position, source)) {
            return Err(format!(
                "Security0 final Add predecessor {binding} does not bind input {position} to Result {source}"
            ));
        }
    }
    match index.event(slice.transfer)? {
        ProofPayloadEvent::BoundTransfer {
            owner: transfer_owner,
            rule: ProofPayloadRule::Sum { inputs },
        } if *transfer_owner == owner &&
            inputs.as_ref() ==
                [
                    ProofPayloadValueRef::Predecessor {
                        binding_event: slice.left_binding,
                        input_position: 0,
                        projection: crate::operational_noise::g0::BoundProjection::Coefficient,
                    },
                    ProofPayloadValueRef::Predecessor {
                        binding_event: slice.right_binding,
                        input_position: 1,
                        projection: crate::operational_noise::g0::BoundProjection::Coefficient,
                    },
                ] =>
        {
            ()
        }
        _ => return Err("Security0 final Add transfer row is inconsistent".to_owned()),
    }
    match index.event(slice.result)? {
        ProofPayloadEvent::Result {
            owner: result_owner,
            value:
                ProofPayloadValue::Exact {
                    terms, coefficient_producer, summary, summary_producer, ..
                },
        } if *result_owner == owner &&
            terms.is_empty() &&
            *coefficient_producer == slice.transfer &&
            summary == &result.summary &&
            summary_producer.is_none() => {}
        _ => return Err("Security0 final Add Result row is inconsistent".to_owned()),
    }
    if !matches!(
        result.summary.coefficient_bound(),
        crate::operational_noise::facts::NumericContract::Known(
            crate::operational_noise::facts::CoefficientBound::Finite(_)
        )
    ) {
        return Err("Security0 final Add Result summary is not finite".to_owned());
    }
    let merge_events = (slice.first_merge..slice.result)
        .map(|event| match index.event(event)? {
            ProofPayloadEvent::CoefficientMerge(merge) if merge.owner == owner => Ok(event),
            _ => Err(format!("Security0 final Add merge row {event} is inconsistent")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    match index.event(slice.prefold)? {
        ProofPayloadEvent::PreFoldPolynomial(prefold)
            if prefold.result_event == slice.result &&
                prefold.terms == result.terms &&
                prefold.summary == result.summary &&
                prefold.summary_evidence ==
                    Some(ProofPayloadValueRef::Result {
                        event: slice.result,
                        projection: crate::operational_noise::g0::BoundProjection::Summary,
                    }) => {}
        _ => return Err("Security0 final Add PreFold row is inconsistent".to_owned()),
    }
    match index.event(slice.end)? {
        ProofPayloadEvent::InvocationEnd { root, result: end, pre_fold_event }
            if *root == owner &&
                *pre_fold_event == slice.prefold &&
                matches!(
                    end,
                    ProofPayloadValue::Exact { terms, summary, .. }
                        if terms == &result.terms && summary == &result.summary
                ) => {}
        _ => return Err("Security0 final Add InvocationEnd row is inconsistent".to_owned()),
    }
    Ok(FinalAddRenderData {
        owner,
        left,
        right,
        merge_events,
        result,
        prefold_event: slice.prefold,
        end_event: slice.end,
    })
}

fn add_sub_merge_polynomials(
    kind: OperationKind,
    left: &[ProofPayloadTerm],
    right: &[ProofPayloadTerm],
) -> Result<(Vec<ProofPayloadTerm>, Vec<ProofPayloadTerm>), String> {
    if !matches!(kind, OperationKind::Add | OperationKind::Subtract) {
        return Err("Security0 merge base requested for a non-Add/Sub operation".to_owned());
    }
    let mut base = left
        .iter()
        .map(|term| (term.monomial.clone(), term.coefficient.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut working = base.clone();
    for term in right {
        let contribution =
            if kind == OperationKind::Add { term.coefficient.clone() } else { -&term.coefficient };
        if !working.contains_key(&term.monomial) {
            base.insert(term.monomial.clone(), contribution.clone());
        }
        *working.entry(term.monomial.clone()).or_default() += contribution;
    }
    let to_terms = |terms: BTreeMap<ProofPayloadMonomial, num_bigint::BigInt>| {
        terms
            .into_iter()
            .filter_map(|(monomial, coefficient)| {
                (!coefficient.is_zero()).then_some(ProofPayloadTerm { monomial, coefficient })
            })
            .collect::<Vec<_>>()
    };
    Ok((to_terms(base), to_terms(working)))
}

fn operator_transfer_for_group(
    index: &PayloadIndex,
    owner: ProofPayloadOwner,
    frame: u64,
    first_merge_event: u64,
    kind: OperationKind,
    left_result: u64,
    right_result: u64,
) -> Result<u64, String> {
    let previous_result = index
        .results
        .iter()
        .filter(|result| {
            result.event < first_merge_event &&
                result.owner == owner &&
                index.immediate_frames
                    [usize::try_from(result.event).expect("indexed prior operator Result")] ==
                    Some(frame)
        })
        .map(|result| result.event)
        .max()
        .unwrap_or(frame);
    let mut candidates = Vec::new();
    for (candidate_owner, rule, event) in &index.operations {
        if *candidate_owner != owner || *event <= previous_result || *event >= first_merge_event {
            continue;
        }
        let references = match (kind, rule) {
            (OperationKind::Add | OperationKind::Subtract, ProofPayloadRule::Sum { inputs })
                if inputs.len() == 2 =>
            {
                Some([&inputs[0], &inputs[1]])
            }
            (OperationKind::Multiply, ProofPayloadRule::Product { left, right, .. }) => {
                Some([left, right])
            }
            (OperationKind::Tensor, ProofPayloadRule::Tensor { left, right, .. }) => {
                Some([left, right])
            }
            _ => None,
        };
        let Some(references) = references else { continue };
        let resolved = [
            index.value_ref(owner, references[0]).map(|result| result.event),
            index.value_ref(owner, references[1]).map(|result| result.event),
        ];
        if matches!(resolved, [Ok(left), Ok(right)] if left == left_result && right == right_result)
        {
            candidates.push(*event);
        }
    }
    match candidates.as_slice() {
        [event] => Ok(*event),
        _ => Err(format!(
            "Security0 operator merge {first_merge_event} has {} event-local typed transfer candidates after Result boundary {previous_result}",
            candidates.len()
        )),
    }
}

fn operator_transfer_without_merges(
    index: &PayloadIndex,
    result: &ResultRecord,
    kind: OperationKind,
) -> Result<(u64, u64, u64), String> {
    let coefficient_producer = match index.event(result.event)? {
        ProofPayloadEvent::Result {
            owner,
            value: ProofPayloadValue::Exact { coefficient_producer, .. },
        } if *owner == result.owner => *coefficient_producer,
        _ => {
            return Err(format!(
                "Security0 no-merge Result {} is not the expected exact row",
                result.event
            ));
        }
    };
    let frame = index.immediate_frames
        [usize::try_from(result.event).map_err(|_| "no-merge Result index overflow")?]
    .ok_or_else(|| format!("no-merge Result {} has no immediate frame", result.event))?;
    let previous_result = index
        .results
        .iter()
        .filter(|candidate| {
            candidate.event < result.event &&
                candidate.owner == result.owner &&
                index.immediate_frames
                    [usize::try_from(candidate.event).expect("indexed prior no-merge Result")] ==
                    Some(frame)
        })
        .map(|candidate| candidate.event)
        .max()
        .unwrap_or(frame);
    let mut candidates = Vec::new();
    for (owner, rule, event) in &index.operations {
        if *owner != result.owner ||
            *event != coefficient_producer ||
            *event <= previous_result ||
            *event >= result.event ||
            index.immediate_frames[usize::try_from(*event).expect("indexed no-merge transfer")] !=
                Some(frame)
        {
            continue;
        }
        let references = match (kind, rule) {
            (OperationKind::Add | OperationKind::Subtract, ProofPayloadRule::Sum { inputs })
                if inputs.len() == 2 =>
            {
                Some([&inputs[0], &inputs[1]])
            }
            (OperationKind::Multiply, ProofPayloadRule::Product { left, right, .. }) => {
                Some([left, right])
            }
            _ => None,
        };
        let Some(references) = references else { continue };
        if let (Ok(left), Ok(right)) = (
            index.value_ref(result.owner, references[0]),
            index.value_ref(result.owner, references[1]),
        ) {
            candidates.push((*event, left.event, right.event));
        }
    }
    match candidates.as_slice() {
        [candidate] => Ok(*candidate),
        _ => Err(format!(
            "Security0 no-merge Result {} has {} event-local typed transfer candidates after Result boundary {previous_result}",
            result.event,
            candidates.len()
        )),
    }
}

fn collect_left_claim_nodes<'a>(
    statement: &CertificateDocumentV1,
    index: &'a PayloadIndex,
    data: &mut LeftRootRenderData<'a>,
    relation_probes: &[RelationProbe],
    root_event: u64,
    right_root_event: u64,
) -> Result<Vec<LeftClaimNode>, String> {
    let relation_by_event = relation_probes
        .iter()
        .map(|relation| (relation.event, relation))
        .collect::<BTreeMap<_, _>>();
    let mut grouped = BTreeMap::<u64, BTreeMap<LeftMergeGroupKey, Vec<u64>>>::new();
    for node in &data.merges {
        let frame = index.immediate_frames
            [usize::try_from(node.event).map_err(|_| "left claim merge index overflow")?]
        .ok_or_else(|| format!("left claim merge {} has no immediate frame", node.event))?;
        let key = match &node.merge.source {
            ProofPayloadCoefficientMergeSource::Operator { inputs } => {
                LeftMergeGroupKey::Operator {
                    frame,
                    owner: node.merge.owner,
                    left_result: inputs[0].value_event,
                    right_result: inputs[1].value_event,
                }
            }
            ProofPayloadCoefficientMergeSource::Relation { application, .. } => {
                LeftMergeGroupKey::Relation {
                    frame,
                    owner: node.merge.owner,
                    application: *application,
                }
            }
        };
        grouped.entry(node.result_event).or_default().entry(key).or_default().push(node.event);
    }
    let mut pending = vec![root_event];
    let mut nodes = BTreeMap::<u64, LeftClaimNode>::new();
    while let Some(result_event) = pending.pop() {
        if nodes.contains_key(&result_event) {
            continue;
        }
        let result = index.by_event.get(&result_event).ok_or_else(|| {
            format!("Security0 left claim dependency {result_event} is not an exact Result")
        })?;
        if result_event == right_root_event {
            nodes.insert(
                result_event,
                LeftClaimNode {
                    result: result.clone(),
                    start: LeftClaimStart::RightRoot,
                    relations: Vec::new(),
                },
            );
            continue;
        }
        let mut phases = grouped
            .get(&result_event)
            .map(|groups| groups.iter().map(|(key, rows)| (rows[0], *key)).collect::<Vec<_>>())
            .unwrap_or_default();
        phases.sort_by_key(|(event, _)| *event);
        let (start, relation_start) = match phases.first().copied() {
            Some((
                first_merge_event,
                LeftMergeGroupKey::Operator { frame, owner, left_result, right_result },
            )) => {
                if owner != result.owner {
                    return Err(format!(
                        "Security0 Result {result_event} begins with a foreign operator merge"
                    ));
                }
                let kind = expression_kind(statement, owner)?.ok_or_else(|| {
                    format!(
                        "Security0 Result {result_event} operator owner has no reached operation"
                    )
                })?;
                if kind == OperationKind::Direct {
                    return Err(format!(
                        "Security0 Result {result_event} reaches unsupported {kind:?}"
                    ));
                }
                let transfer_event = operator_transfer_for_group(
                    index,
                    owner,
                    frame,
                    first_merge_event,
                    kind,
                    left_result,
                    right_result,
                )?;
                pending.extend([left_result, right_result]);
                (
                    LeftClaimStart::Operator {
                        first_merge_event: Some(first_merge_event),
                        transfer_event,
                        kind,
                        left_result,
                        right_result,
                    },
                    1,
                )
            }
            Some((_, LeftMergeGroupKey::Relation { frame, owner, application })) => {
                let relation = relation_by_event.get(&application).ok_or_else(|| {
                    format!(
                        "Security0 Result {result_event} begins with unknown relation {application}"
                    )
                })?;
                let prior = index
                    .results
                    .iter()
                    .filter(|candidate| {
                        candidate.event < application &&
                            candidate.owner == owner &&
                            candidate.terms == relation.accumulator.terms &&
                            index.immediate_frames[usize::try_from(candidate.event)
                                .expect("indexed relation accumulator Result")] == Some(frame)
                    })
                    .max_by_key(|candidate| candidate.event)
                    .ok_or_else(|| format!(
                        "Security0 relation {application} has no event-indexed accumulator Result"
                    ))?;
                pending.push(prior.event);
                (LeftClaimStart::PriorResult { result_event: prior.event }, 0)
            }
            None => {
                let frame = index.immediate_frames
                    [usize::try_from(result_event).map_err(|_| "left Result index overflow")?]
                .ok_or_else(|| format!("left Result {result_event} has no immediate frame"))?;
                let previous_result = index
                    .results
                    .iter()
                    .filter(|candidate| {
                        candidate.event < result_event &&
                            candidate.owner == result.owner &&
                            index.immediate_frames[usize::try_from(candidate.event)
                                .expect("indexed prior reached Result")] ==
                                Some(frame)
                    })
                    .map(|candidate| candidate.event)
                    .max()
                    .unwrap_or(frame);
                let existing = data.merges.iter().map(|node| node.event).collect::<BTreeSet<_>>();
                let discovered = index
                    .merges
                    .iter()
                    .filter(|(event, merge)| {
                        *event > previous_result &&
                            *event < result_event &&
                            merge.owner == result.owner &&
                            index.immediate_frames
                                [usize::try_from(*event).expect("indexed reached merge")] ==
                                Some(frame) &&
                            !existing.contains(event)
                    })
                    .collect::<Vec<_>>();
                if !discovered.is_empty() {
                    for (event, merge) in discovered {
                        let key = match &merge.source {
                            ProofPayloadCoefficientMergeSource::Operator { inputs } => {
                                LeftMergeGroupKey::Operator {
                                    frame,
                                    owner: merge.owner,
                                    left_result: inputs[0].value_event,
                                    right_result: inputs[1].value_event,
                                }
                            }
                            ProofPayloadCoefficientMergeSource::Relation {
                                application, ..
                            } => LeftMergeGroupKey::Relation {
                                frame,
                                owner: merge.owner,
                                application: *application,
                            },
                        };
                        grouped
                            .entry(result_event)
                            .or_default()
                            .entry(key)
                            .or_default()
                            .push(*event);
                        data.merges.push(LeftRootMergeNode { event: *event, result_event, merge });
                    }
                    pending.push(result_event);
                    continue;
                }
                if let Some(kind) = expression_kind(statement, result.owner)? &&
                    !matches!(kind, OperationKind::Direct | OperationKind::Tensor)
                {
                    let (transfer_event, left_result, right_result) =
                        operator_transfer_without_merges(index, result, kind)?;
                    pending.extend([left_result, right_result]);
                    nodes.insert(
                        result_event,
                        LeftClaimNode {
                            result: (*result).clone(),
                            start: LeftClaimStart::Operator {
                                first_merge_event: None,
                                transfer_event,
                                kind,
                                left_result,
                                right_result,
                            },
                            relations: Vec::new(),
                        },
                    );
                    continue;
                }
                let producer_event = result_event
                    .checked_sub(1)
                    .ok_or_else(|| "Security0 terminal Result 0 has no producer".to_owned())?;
                match index.event(producer_event)? {
                    ProofPayloadEvent::BoundTransfer { owner, rule }
                        if *owner == result.owner &&
                            reached_terminal_rule(rule) &&
                            matches!(
                                result.summary.coefficient_bound(),
                                crate::operational_noise::facts::NumericContract::Known(
                                    crate::operational_noise::facts::CoefficientBound::ExactZero
                                )
                            ) =>
                    {
                        (LeftClaimStart::Terminal { producer_event }, 0)
                    }
                    event => {
                        return Err(format!(
                            "Security0 left Result {result_event} has no merge phases and unsupported terminal producer {event:?}"
                        ))
                    }
                }
            }
        };
        let mut relations = Vec::new();
        for (_, phase) in phases.into_iter().skip(relation_start) {
            let LeftMergeGroupKey::Relation { application, owner, .. } = phase else {
                return Err(format!(
                    "Security0 Result {result_event} contains a second operator merge phase"
                ));
            };
            if owner != result.owner || !relation_by_event.contains_key(&application) {
                return Err(format!(
                    "Security0 Result {result_event} contains inconsistent relation {application}"
                ));
            }
            relations.push(application);
        }
        nodes.insert(result_event, LeftClaimNode { result: (*result).clone(), start, relations });
    }
    data.merges.sort_by_key(|node| node.event);
    let nodes = nodes.into_values().collect::<Vec<_>>();
    let mut unsupported_transitions = BTreeMap::<String, (usize, String)>::new();
    for node in &nodes {
        let result_is_zero = matches!(
            node.result.summary.coefficient_bound(),
            crate::operational_noise::facts::NumericContract::Known(
                crate::operational_noise::facts::CoefficientBound::ExactZero
            )
        );
        if let LeftClaimStart::Operator {
            first_merge_event: Some(first_merge_event),
            kind,
            left_result,
            right_result,
            ..
        } = node.start
        {
            let left = index.result(left_result)?;
            let right = index.result(right_result)?;
            let input_is_zero = |result: &ResultRecord| {
                matches!(
                    result.summary.coefficient_bound(),
                    crate::operational_noise::facts::NumericContract::Known(
                        crate::operational_noise::facts::CoefficientBound::ExactZero
                    )
                )
            };
            if !input_is_zero(&left) || !input_is_zero(&right) || !result_is_zero {
                let finite_maximum =
                    |result: &ResultRecord| match result.summary.coefficient_bound() {
                        crate::operational_noise::facts::NumericContract::Known(
                            crate::operational_noise::facts::CoefficientBound::Finite(bound),
                        ) => Some(bound.maximum_absolute_coefficient),
                        _ => None,
                    };
                let additive_finite = matches!(kind, OperationKind::Add | OperationKind::Subtract) &&
                    finite_maximum(&left)
                        .zip(finite_maximum(&right))
                        .zip(finite_maximum(&node.result))
                        .is_some_and(|((left, right), output)| left + right == output);
                let reached_product_finite = kind == OperationKind::Multiply &&
                    finite_maximum(&left).is_some() &&
                    input_is_zero(&right) &&
                    finite_maximum(&node.result).is_some();
                if additive_finite || reached_product_finite {
                    continue;
                }
                let category = format!(
                    "{kind:?}: leftZero={}, rightZero={}, outputZero={result_is_zero}",
                    input_is_zero(&left),
                    input_is_zero(&right)
                );
                let detail = format!(
                    "Security0 merged {kind:?} operator Result {} (first merge {first_merge_event}) reaches unsupported summary transition {:?}, {:?} -> {:?}",
                    node.result.event, left.summary, right.summary, node.result.summary
                );
                let entry = unsupported_transitions.entry(category).or_insert((0, detail));
                entry.0 += 1;
            }
        }
    }
    if !unsupported_transitions.is_empty() {
        return Err(unsupported_transitions
            .into_iter()
            .map(|(category, (count, representative))| {
                format!("{category}: {count} Results; representative: {representative}")
            })
            .collect::<Vec<_>>()
            .join("\n"));
    }
    let result_events = nodes.iter().map(|node| node.result.event).collect::<BTreeSet<_>>();
    for node in &nodes {
        for dependency in left_claim_dependencies(node) {
            if dependency >= node.result.event || !result_events.contains(&dependency) {
                return Err(format!(
                    "Security0 left claim Result {} has non-topological dependency {dependency}",
                    node.result.event
                ));
            }
        }
    }
    Ok(nodes)
}

fn exact_result_producers(index: &PayloadIndex, event: u64) -> Result<(u64, Option<u64>), String> {
    match index.event(event)? {
        ProofPayloadEvent::Result {
            value: ProofPayloadValue::Exact { coefficient_producer, summary_producer, .. },
            ..
        } => Ok((*coefficient_producer, *summary_producer)),
        _ => Err(format!("Security0 left claim Result {event} is not exact")),
    }
}

fn predecessor_ref_data(
    index: &PayloadIndex,
    consumer: ProofPayloadOwner,
    reference: &ProofPayloadValueRef,
    expected_result: u64,
) -> Result<(u64, u64, u64), String> {
    let ProofPayloadValueRef::Predecessor { binding_event, input_position, .. } = reference else {
        return Err(format!(
            "Security0 finite operation Result dependency {expected_result} is not predecessor-bound"
        ));
    };
    let ProofPayloadEvent::Predecessor {
        consumer: row_consumer,
        input_position: row_position,
        predecessor,
        source_result,
    } = index.event(*binding_event)?
    else {
        return Err(format!(
            "Security0 predecessor binding {binding_event} is not a Predecessor row"
        ));
    };
    if *row_consumer != consumer ||
        row_position != input_position ||
        *source_result != expected_result
    {
        return Err(format!(
            "Security0 predecessor binding {binding_event} does not bind Result {expected_result}"
        ));
    }
    Ok((*binding_event, u64::from(*input_position), *predecessor))
}

fn render_relation_claim_chain(
    source: &mut String,
    relation_by_event: &BTreeMap<u64, &RelationProbe>,
    node: &LeftClaimNode,
    initial_working: &str,
    modulus: &str,
) -> Result<(), String> {
    let mut reordered = node
        .result
        .terms
        .iter()
        .map(|term| (term.monomial.clone(), term.coefficient.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut reordered_outputs = vec![Vec::new(); node.relations.len()];
    for (ordinal, application) in node.relations.iter().enumerate().rev() {
        let relation = relation_by_event.get(application).ok_or_else(|| {
            format!("Security0 relation continuation {application} has no semantic probe")
        })?;
        reordered_outputs[ordinal] = reordered
            .iter()
            .filter_map(|(monomial, coefficient)| {
                (!coefficient.is_zero()).then_some(ProofPayloadTerm {
                    monomial: monomial.clone(),
                    coefficient: coefficient.clone(),
                })
            })
            .collect();
        for (_, merge) in &relation.output_merges {
            *reordered.entry(merge.output.clone()).or_default() -= &merge.signed_contribution;
        }
        *reordered.entry(relation.source.clone()).or_default() += &relation.outer;
        reordered.retain(|_, coefficient| !coefficient.is_zero());
    }
    let mut previous_claim = "mergeClaim".to_owned();
    let mut previous_working = initial_working.to_owned();
    for (ordinal, application) in node.relations.iter().enumerate() {
        let relation = relation_by_event.get(application).ok_or_else(|| {
            format!("Security0 relation continuation {application} has no semantic probe")
        })?;
        writeln!(source, "theorem relationApplicationAt{ordinal} (selector : Nat)\n    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :\n    RelationApplicationAt document history (some selector) {application} := by\n  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩\n  simp only [selectorMinimum] at selectorLower\n  simp only [selectorMaximum] at selectorUpper\n  simp [ownerAtSelector, document, selectorLower, selectorUpper]")
            .expect("String write");
        writeln!(
            source,
            "def relationWorking{ordinal} : Polynomial Owner := {}",
            terms_text(&reordered_outputs[ordinal])
        )
        .expect("String write");
        writeln!(
            source,
            "def relationRhsRaw{ordinal} : List Term := {}",
            raw_terms_text(&relation.rhs.terms)
        )
        .expect("String write");
        writeln!(source, "def relationBase{ordinal} : Polynomial Owner :=\n  subtract {previous_working}\n    [{{ coefficient := ({}), key := LeftRelationMerge{application}.source }}]", relation.outer)
            .expect("String write");
        writeln!(source, "def relationReconstruction{ordinal} :\n    MergeReconstructionAt history LeftRelationMerge{application}.frameStart\n      LeftRelationMerge{application}.owner (.relation {application}) relationBase{ordinal}\n      relationWorking{ordinal} :=\n  {{ deltas := LeftRelationMerge{application}.deltas\n    rows := LeftRelationMerge{application}.rows\n    agreement := by decide +kernel }}")
            .expect("String write");
        writeln!(source, "theorem relationAgreement{ordinal} :\n    CanonicalAgreement (add relationBase{ordinal} relationReconstruction{ordinal}.deltas)\n      (relationPoly {previous_working} LeftRelationMerge{application}.source\n        (relationContext LeftRelationMerge{application}.source\n          LeftRelationMerge{application}.source.centralFactors {} {}) ({})\n        (relationRhsRaw{ordinal}.map Term.toExact)) := by\n  dsimp [relationReconstruction{ordinal}, relationBase{ordinal}, relationWorking{ordinal},\n    relationRhsRaw{ordinal}, {previous_working}, LeftRelationMerge{application}.deltas,\n    LeftRelationMerge{application}.source]\n  decide +kernel", relation.start, relation.end, relation.outer)
            .expect("String write");
        let theorem = match relation.kind {
            RelationRuleKind::Universal => "universalRelationMergeClaim",
            RelationRuleKind::Gadget => "gadgetRelationMergeClaim",
        };
        writeln!(source, "theorem relationClaim{ordinal} (selector : Nat)\n    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)\n    (witness : Witness document history (some selector) {modulus}) :\n    ValueClaim.Interprets {modulus} witness.env (actual selector witness)\n      (.exact relationWorking{ordinal} summary) := by\n  apply {theorem}\n    (document := document) (history := history) (selector := some selector)\n    (modulus := {modulus}) (witness := witness) (application := {application})\n    (frameStart := {}) (owner := {})\n    (source := {}) (lhs := {})\n    (outerCoefficient := {}) (orderedStart := {}) (orderedEndExclusive := {})\n    (rhsRaw := relationRhsRaw{ordinal})\n    (accumulator := {previous_working}) (working := relationWorking{ordinal})\n    (reconstruction := relationReconstruction{ordinal})\n    (actual := actual selector witness) (summary := summary)\n  · exact relationApplicationAt{ordinal} selector selectorLower selectorUpper\n  · rfl\n  · rfl", relation.frame_start, owner_text(relation.owner), monomial_text(&relation.source), monomial_text(&relation.lhs), relation.outer, relation.start, relation.end)
            .expect("String write");
        if relation.kind == RelationRuleKind::Gadget {
            source.push_str("  · decide +kernel\n");
        }
        writeln!(source, "  · exact {previous_claim} selector selectorLower selectorUpper witness\n  · exact relationAgreement{ordinal}\n  · decide +kernel")
            .expect("String write");
        previous_claim = format!("relationClaim{ordinal}");
        previous_working = format!("relationWorking{ordinal}");
    }
    writeln!(source, "theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)\n    (selectorUpper : selector < selectorMaximum)\n    (witness : Witness document history (some selector) {modulus}) :\n    ExactClaimAt history {modulus} witness.env resultEvent owner\n      (actual selector witness) rawTerms summary := by\n  apply exactClaimAt_of_mergeClaim\n    ({previous_claim} selector selectorLower selectorUpper witness)\n  · decide +kernel\n  · rfl")
        .expect("String write");
    Ok(())
}

fn render_left_claim_node(
    source: &mut String,
    statement: &CertificateDocumentV1,
    index: &PayloadIndex,
    data: &LeftRootRenderData<'_>,
    replayed_bounds: &BTreeMap<
        u64,
        crate::operational_noise::facts::NumericContract<
            crate::operational_noise::facts::CoefficientBound,
        >,
    >,
    relation_by_event: &BTreeMap<u64, &RelationProbe>,
    node: &LeftClaimNode,
    modulus: &str,
) -> Result<(), String> {
    let event = node.result.event;
    writeln!(source, "namespace LeftClaimResult{event}").expect("String write");
    writeln!(source, "def owner : Owner := {}", owner_text(node.result.owner))
        .expect("String write");
    writeln!(source, "def rawTerms : List Term := {}", raw_terms_text(&node.result.terms))
        .expect("String write");
    writeln!(source, "def summary : Bound := {}", summary_text(&node.result.summary))
        .expect("String write");
    writeln!(source, "def resultEvent : Nat := {event}").expect("String write");
    match &node.start {
        LeftClaimStart::RightRoot => {
            source.push_str(&format!(
                "def actual (selector : Nat)\n    (witness : Witness document history (some selector) {modulus}) : Int :=\n  Cert.ResidualLeftResult{}.actual selector witness\n\ntheorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)\n    (selectorUpper : selector < selectorMaximum)\n    (witness : Witness document history (some selector) {modulus}) :\n    ExactClaimAt history {modulus} witness.env resultEvent owner\n      (actual selector witness) rawTerms summary := by\n  simpa [resultEvent, owner, rawTerms, summary, actual,\n    Cert.ResidualLeftResult{}.actual] using\n    SemanticRightRoot.rightRootClaimSound selector selectorLower selectorUpper witness\n",
                event, event
            ));
        }
        LeftClaimStart::Terminal { producer_event } => {
            let frame = index.immediate_frames
                [usize::try_from(event).map_err(|_| "left terminal event index overflow")?]
            .ok_or_else(|| format!("left terminal Result {event} has no frame"))?;
            let ProofPayloadEvent::BoundTransfer { rule, .. } = index.event(*producer_event)?
            else {
                unreachable!("left terminal collector validated BoundTransfer")
            };
            let ProofPayloadEvent::Result {
                value:
                    ProofPayloadValue::Exact {
                        coefficient_bound,
                        coefficient_producer,
                        summary_producer,
                        ..
                    },
                ..
            } = index.event(event)?
            else {
                unreachable!("left claim nodes contain only exact Results");
            };
            if *coefficient_producer != *producer_event || summary_producer.is_some() {
                return Err(format!(
                    "left terminal Result {event} does not use its adjacent producer exclusively"
                ));
            }
            writeln!(source, "def producerEvent : Nat := {producer_event}").expect("String write");
            writeln!(source, "def actual (selector : Nat)\n    (witness : Witness document history (some selector) {modulus}) : Int :=\n  Cert.ResidualLeftResult{event}.actual selector witness").expect("String write");
            writeln!(source, "theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)\n    (selectorUpper : selector < selectorMaximum) :\n    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by\n  refine ⟨by decide, ?_, ?_⟩\n  · simp only [selectorMinimum] at selectorLower\n    simp only [selectorMaximum] at selectorUpper\n    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]\n  · refine ⟨{}, {frame}, {}, {}, ?_, ?_⟩\n    · rfl\n    · rfl", reached_terminal_rule_text(rule)?, recorded_bound_text(coefficient_bound), reached_terminal_constructor(rule)?).expect("String write");
            writeln!(source, "theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)\n    (selectorUpper : selector < selectorMaximum)\n    (witness : Witness document history (some selector) {modulus}) :\n    ExactClaimAt history {modulus} witness.env resultEvent owner\n      (actual selector witness) rawTerms summary := by\n  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)").expect("String write");
        }
        LeftClaimStart::PriorResult { result_event } => {
            let first_relation = *node.relations.first().ok_or_else(|| {
                format!("Security0 Result {event} has a prior accumulator but no relation")
            })?;
            writeln!(source, "def actual (selector : Nat)\n    (witness : Witness document history (some selector) {modulus}) : Int :=\n  Cert.ResidualLeftResult{event}.actual selector witness")
                .expect("String write");
            writeln!(source, "theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)\n    (selectorUpper : selector < selectorMaximum)\n    (witness : Witness document history (some selector) {modulus}) :\n    ValueClaim.Interprets {modulus} witness.env (actual selector witness)\n      (.exact LeftRelationMerge{first_relation}.accumulator summary) := by\n  simpa [actual, summary, LeftRelationMerge{first_relation}.accumulator] using\n    (LeftClaimResult{result_event}.claimSound selector selectorLower selectorUpper witness).claim")
                .expect("String write");
            render_relation_claim_chain(
                source,
                relation_by_event,
                node,
                &format!("LeftRelationMerge{first_relation}.accumulator"),
                modulus,
            )?;
        }
        LeftClaimStart::Operator {
            first_merge_event,
            transfer_event,
            kind,
            left_result,
            right_result,
        } => {
            let left = index.result(*left_result)?;
            let right = index.result(*right_result)?;
            let value_type = owner_value_type_text(statement, node.result.owner)?;
            if *kind == OperationKind::Direct {
                return Err(format!("Security0 left Result {event} reaches Direct"));
            }
            writeln!(source, "def actual (selector : Nat)\n    (witness : Witness document history (some selector) {modulus}) : Int :=\n  Cert.ResidualLeftResult{event}.actual selector witness").expect("String write");
            let left_zero = summary_text(&left.summary) == ".exactZero";
            let right_zero = summary_text(&right.summary) == ".exactZero";
            let output_zero = summary_text(&node.result.summary) == ".exactZero";
            if let Some(first_merge_event) = first_merge_event {
                if !left_zero || !right_zero || !output_zero {
                    if *kind == OperationKind::Multiply {
                        let finite_maximum = |result: &ResultRecord| match result
                            .summary
                            .coefficient_bound()
                        {
                            crate::operational_noise::facts::NumericContract::Known(
                                crate::operational_noise::facts::CoefficientBound::Finite(bound),
                            ) => Some(bound.maximum_absolute_coefficient),
                            _ => None,
                        };
                        let left_maximum = finite_maximum(&left).ok_or_else(|| {
                            format!(
                                "Security0 finite Product Result {event} has non-finite left input"
                            )
                        })?;
                        let (right_coefficient_transfer, right_maximum) =
                            match index.event(*right_result)? {
                            ProofPayloadEvent::Result {
                                value:
                                    ProofPayloadValue::Exact {
                                        coefficient_bound:
                                            crate::operational_noise::facts::NumericContract::Known(
                                                crate::operational_noise::facts::CoefficientBound::Finite(
                                                    bound,
                                                ),
                                            ),
                                        coefficient_producer,
                                        ..
                                    },
                                ..
                            } => (*coefficient_producer, bound.maximum_absolute_coefficient.clone()),
                            _ => {
                                return Err(format!(
                                    "Security0 finite Product Result {event} right Result does not \
                                     carry an authoritative finite coefficient bound"
                                ));
                            }
                        };
                        let right_producer = data
                            .bounds
                            .iter()
                            .find(|bound| bound.event == right_coefficient_transfer)
                            .ok_or_else(|| {
                                format!(
                                    "Security0 finite Product Result {event} right coefficient producer \
                                     {right_coefficient_transfer} is outside the bound closure"
                                )
                            })?;
                        let right_producer_namespace =
                            left_bound_namespace(right_producer.event, right_producer.rule);
                        let right_dependencies = format!("{right_producer_namespace}.bound");
                        let right_producer_maximum = match replayed_bounds
                            .get(&right_producer.event)
                        {
                            Some(crate::operational_noise::facts::NumericContract::Known(
                                crate::operational_noise::facts::CoefficientBound::Finite(bound),
                            )) => bound.maximum_absolute_coefficient.clone(),
                            _ => {
                                return Err(format!(
                                    "Security0 finite Product Result {event} right coefficient \
                                     producer does not replay to a finite bound"
                                ));
                            }
                        };
                        if right_maximum > right_producer_maximum {
                            return Err(format!(
                                "Security0 finite Product Result {event} recorded coefficient bound \
                                 does not refine its stored producer"
                            ));
                        }
                        let ProofPayloadEvent::BoundTransfer {
                            rule:
                                ProofPayloadRule::Product {
                                    left: coefficient_left,
                                    right: coefficient_right,
                                    facts: coefficient_facts,
                                },
                            ..
                        } = index.event(*transfer_event)?
                        else {
                            return Err(format!(
                                "Security0 finite Product Result {event} has no coefficient Product row"
                            ));
                        };
                        let (left_binding, left_position, left_expression) = predecessor_ref_data(
                            index,
                            node.result.owner,
                            coefficient_left,
                            *left_result,
                        )?;
                        let (right_binding, right_position, right_expression) =
                            predecessor_ref_data(
                                index,
                                node.result.owner,
                                coefficient_right,
                                *right_result,
                            )?;
                        let (_, summary_producer) = exact_result_producers(index, event)?;
                        let summary_transfer = summary_producer.ok_or_else(|| {
                            format!(
                                "Security0 finite Product Result {event} has no summary producer"
                            )
                        })?;
                        let summary_node = data
                            .bounds
                            .iter()
                            .find(|bound| bound.event == summary_transfer)
                            .ok_or_else(|| {
                                format!(
                                    "Security0 finite Product Result {event} summary producer \
                                     {summary_transfer} is outside the bound closure"
                                )
                            })?;
                        let ProofPayloadRule::Product {
                            left:
                                ProofPayloadValueRef::Result {
                                    event: summary_left,
                                    projection: BoundProjection::Summary,
                                },
                            right: ProofPayloadValueRef::Transfer(summary_right),
                            facts: summary_facts,
                        } = summary_node.rule
                        else {
                            return Err(format!(
                                "Security0 finite Product Result {event} summary producer has unsupported rule"
                            ));
                        };
                        if *summary_left != *left_result {
                            return Err(format!(
                                "Security0 finite Product Result {event} summary producer references \
                                 unexpected left input"
                            ));
                        }
                        let right_summary_node = data
                            .bounds
                            .iter()
                            .find(|bound| bound.event == *summary_right)
                            .ok_or_else(|| {
                                format!(
                                    "Security0 finite Product Result {event} right summary transfer \
                                     {summary_right} is outside the bound closure"
                                )
                            })?;
                        let right_summary_maximum = match replayed_bounds
                            .get(&right_summary_node.event)
                        {
                            Some(crate::operational_noise::facts::NumericContract::Known(
                                crate::operational_noise::facts::CoefficientBound::Finite(bound),
                            )) => bound.maximum_absolute_coefficient.clone(),
                            _ => {
                                return Err(format!(
                                    "Security0 finite Product Result {event} right summary transfer \
                                     {summary_right} does not replay to a finite bound"
                                ));
                            }
                        };
                        if right_maximum > right_summary_maximum {
                            return Err(format!(
                                "Security0 finite Product Result {event} coefficient maximum exceeds \
                                 its replayed summary-input maximum"
                            ));
                        }
                        let left_input = index.result(*left_result)?;
                        let right_input = index.result(*right_result)?;
                        let merge_terms = data
                            .merges
                            .iter()
                            .filter(|merge| merge.result_event == event)
                            .filter_map(|merge| match &merge.merge.source {
                                ProofPayloadCoefficientMergeSource::Operator { inputs }
                                    if inputs[0].value_event == *left_result &&
                                        inputs[1].value_event == *right_result =>
                                {
                                    let left =
                                        &left_input.terms[usize::try_from(inputs[0].term_ordinal)
                                            .expect("validated finite Product left ordinal")];
                                    let right =
                                        &right_input.terms[usize::try_from(inputs[1].term_ordinal)
                                            .expect("validated finite Product right ordinal")];
                                    Some((
                                        left.monomial.clone(),
                                        right.monomial.clone(),
                                        merge.merge.output.clone(),
                                    ))
                                }
                                _ => None,
                            })
                            .collect::<Vec<_>>();
                        let scalar_flags = resolve_scalar_flags(
                            &matching_scalar_flags_from_merges(&merge_terms),
                            &[left_input, right_input],
                        )?;
                        let (_, _, _, _, _, factor) =
                            reached_product_shape(statement, node.result.owner, summary_facts)?;
                        let coefficient_facts = product_facts_text(coefficient_facts);
                        let summary_facts = product_facts_text(summary_facts);
                        let left_scalar = if scalar_flags.0 { "true" } else { "false" };
                        let right_scalar = if scalar_flags.1 { "true" } else { "false" };
                        writeln!(
                            source,
                            "def computedSummary : Bound :=\n  coeffClassToTallBound\n    (EventReplay.productWithFactor {factor}\n      (.finite ⟨{left_maximum}, by decide⟩)\n      (.finite ⟨{right_summary_maximum}, by decide⟩))"
                        )
                        .expect("String write");
                        writeln!(source, "theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)\n    (selectorUpper : selector < selectorMaximum)\n    (witness : Witness document history (some selector) {modulus}) :\n    ValueClaim.Interprets {modulus} witness.env (actual selector witness)\n      (.exact LeftOperatorMerge{first_merge_event}.working computedSummary) := by\n  apply operatorProductFiniteMergeClaim\n    (document := document) (history := history) (modulus := {modulus})\n    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge{first_merge_event}.frameStart)\n    (owner := owner) (leftOwner := LeftClaimResult{left_result}.owner)\n    (rightOwner := LeftClaimResult{right_result}.owner)\n    (leftResult := {left_result}) (rightResult := {right_result})\n    (leftActual := LeftClaimResult{left_result}.actual selector witness)\n    (rightActual := LeftClaimResult{right_result}.actual selector witness)\n    (leftRaw := LeftClaimResult{left_result}.rawTerms)\n    (rightRaw := LeftClaimResult{right_result}.rawTerms)\n    (working := LeftOperatorMerge{first_merge_event}.working)\n    (leftBinding := {left_binding}) (rightBinding := {right_binding})\n    (leftInputPosition := {left_position}) (rightInputPosition := {right_position})\n    (leftExpression := ⟨{left_expression}⟩) (rightExpression := ⟨{right_expression}⟩)\n    (coefficientTransfer := {transfer_event}) (summaryTransfer := {summary_transfer})\n    (rightCoefficientProducer := {right_coefficient_transfer})\n    (rightSummaryTransfer := {summary_right})\n    (leftMaximum := ⟨{left_maximum}, by decide⟩)\n    (rightProducerMaximum := ⟨{right_producer_maximum}, by decide⟩)\n    (rightRecordedMaximum := {right_maximum})\n    (rightSummaryMaximum := ⟨{right_summary_maximum}, by decide⟩)\n    (leftScalar := {left_scalar}) (rightScalar := {right_scalar}) (factor := {factor})\n    (valueType := {value_type}) (base := LeftOperatorMerge{first_merge_event}.base)\n    (coefficientFacts := {coefficient_facts}) (summaryFacts := {summary_facts})\n    (rightMagnitude := {right_producer_namespace}.actual selector witness)\n    (summaryMagnitude := LeftBound{summary_transfer}.actual selector witness)\n    (reconstruction := LeftOperatorMerge{first_merge_event}.reconstruction)\n    (rightResultAt := by rfl)\n  · rfl\n  · rfl\n  · rfl\n  · rfl\n  · exact LeftClaimResult{left_result}.claimSound selector selectorLower selectorUpper witness\n  · exact LeftClaimResult{right_result}.claimSound selector selectorLower selectorUpper witness\n  · exact .resultExactCoefficient (by rfl)\n      (by dsimp [{right_dependencies}, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,\n        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,\n        RecordedBoundRefines] <;> decide)\n      ({right_producer_namespace}.derived selector witness)\n  · dsimp [RecordedBoundRefines]\n    decide\n  · decide\n  · exact LeftOperatorMerge{first_merge_event}.operationAgreement\n  · exact LeftBound{summary_transfer}.derived selector witness\n  · decide\n  · decide").expect("String write");
                        writeln!(source, "theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)\n    (selectorUpper : selector < selectorMaximum)\n    (witness : Witness document history (some selector) {modulus}) :\n    ValueClaim.Interprets {modulus} witness.env (actual selector witness)\n      (.exact LeftOperatorMerge{first_merge_event}.working summary) := by\n  have claim := computedMergeClaim selector selectorLower selectorUpper witness\n  have summaryEq : computedSummary = summary := by decide\n  simpa only [summaryEq] using claim")
                            .expect("String write");
                        render_relation_claim_chain(
                            source,
                            relation_by_event,
                            node,
                            &format!("LeftOperatorMerge{first_merge_event}.working"),
                            modulus,
                        )?;
                    } else {
                        if !matches!(kind, OperationKind::Add | OperationKind::Subtract) {
                            return Err(format!(
                                "Security0 finite {kind:?} Result {event} awaits its reached theorem ABI"
                            ));
                        }
                        let ProofPayloadEvent::BoundTransfer {
                            rule: ProofPayloadRule::Sum { inputs },
                            ..
                        } = index.event(*transfer_event)?
                        else {
                            return Err(format!(
                                "Security0 finite {kind:?} Result {event} has no Sum row"
                            ));
                        };
                        let [left_ref, right_ref] = inputs.as_slice() else {
                            return Err(format!(
                                "Security0 finite {kind:?} Result {event} Sum is not binary"
                            ));
                        };
                        let (left_binding, left_position, left_expression) =
                            predecessor_ref_data(index, node.result.owner, left_ref, *left_result)?;
                        let (right_binding, right_position, right_expression) =
                            predecessor_ref_data(
                                index,
                                node.result.owner,
                                right_ref,
                                *right_result,
                            )?;
                        let (_, summary_producer) = exact_result_producers(index, event)?;
                        let summary_transfer = summary_producer.ok_or_else(|| {
                            format!(
                                "Security0 finite {kind:?} Result {event} has no summary producer"
                            )
                        })?;
                        let finite_maximum = |result: &ResultRecord| match result
                            .summary
                            .coefficient_bound()
                        {
                            crate::operational_noise::facts::NumericContract::Known(
                                crate::operational_noise::facts::CoefficientBound::Finite(bound),
                            ) => Ok(bound.maximum_absolute_coefficient.clone()),
                            _ => Err(format!(
                                "Security0 finite {kind:?} Result {event} has a non-finite input"
                            )),
                        };
                        let left_maximum = finite_maximum(&left)?;
                        let right_maximum = finite_maximum(&right)?;
                        let theorem = if *kind == OperationKind::Add {
                            "operatorAddFiniteMergeClaimAt"
                        } else {
                            "operatorSubFiniteMergeClaimAt"
                        };
                        let theorem = format!(
                            "{theorem}\n    (document := document) (history := history) \
                             (env := witness.env)"
                        );
                        writeln!(source, "theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)\n    (selectorUpper : selector < selectorMaximum)\n    (witness : Witness document history (some selector) {modulus}) :\n    ExactClaimAt history {modulus} witness.env resultEvent owner\n      (actual selector witness) rawTerms summary := by\n  apply {theorem}\n    (modulus := {modulus}) (frameStart := LeftOperatorMerge{first_merge_event}.frameStart)\n    (resultEvent := resultEvent) (owner := owner)\n    (leftOwner := LeftClaimResult{left_result}.owner)\n    (rightOwner := LeftClaimResult{right_result}.owner)\n    (leftResult := {left_result}) (rightResult := {right_result})\n    (leftActual := LeftClaimResult{left_result}.actual selector witness)\n    (rightActual := LeftClaimResult{right_result}.actual selector witness)\n    (leftRaw := LeftClaimResult{left_result}.rawTerms)\n    (rightRaw := LeftClaimResult{right_result}.rawTerms)\n    (outputRaw := rawTerms) (leftMaximum := {left_maximum})\n    (rightMaximum := {right_maximum}) (valueType := {value_type})\n    (leftBinding := {left_binding}) (rightBinding := {right_binding})\n    (leftInputPosition := {left_position}) (rightInputPosition := {right_position})\n    (leftExpression := ⟨{left_expression}⟩) (rightExpression := ⟨{right_expression}⟩)\n    (coefficientTransfer := {transfer_event}) (summaryTransfer := {summary_transfer})\n    (base := LeftOperatorMerge{first_merge_event}.base)\n    (reconstruction := LeftOperatorMerge{first_merge_event}.reconstruction)\n  · rfl\n  · rfl\n  · rfl\n  · rfl\n  · rfl\n  · exact LeftClaimResult{left_result}.claimSound selector selectorLower selectorUpper witness\n  · exact LeftClaimResult{right_result}.claimSound selector selectorLower selectorUpper witness\n  · exact LeftOperatorMerge{first_merge_event}.operationAgreement\n  · rfl\n  · decide").expect("String write");
                    }
                } else {
                    let theorem = match kind {
                        OperationKind::Add => "operatorAddMergeClaim",
                        OperationKind::Subtract => "operatorSubMergeClaim",
                        OperationKind::Multiply => "operatorProductMergeClaim",
                        OperationKind::Tensor => "operatorTensorMergeClaim",
                        OperationKind::Direct => unreachable!(),
                    };
                    let theorem = format!(
                        "{theorem}\n    (document := document) (history := history) \
                         (env := witness.env)"
                    );
                    let operation = op_probe(index, &node.result, *kind)?;
                    if operation.rule_event != *transfer_event ||
                        operation.input_events != [*left_result, *right_result]
                    {
                        return Err(format!(
                            "Security0 exact-zero Result {event} operation probe changed its selected history rows"
                        ));
                    }
                    let rule_arguments = match operation
                        .rule
                        .as_ref()
                        .expect("reached exact-zero operation has a typed rule")
                    {
                        ProofPayloadRule::Sum { inputs } => {
                            let [left, right] = inputs.as_slice() else {
                                return Err(format!(
                                    "Security0 exact-zero Result {event} Sum is not binary"
                                ));
                            };
                            format!(
                                "    (leftReference := {}) (rightReference := {})\n",
                                value_ref_text(left),
                                value_ref_text(right),
                            )
                        }
                        ProofPayloadRule::Product { left, right, facts } => format!(
                            "    (leftReference := {}) (rightReference := {})\n    \
                             (facts := ⟨{}, {}, {}, {}, {}⟩)\n    \
                             (leftScalar := {}) (rightScalar := {})\n",
                            value_ref_text(left),
                            value_ref_text(right),
                            if facts.left_is_constant_polynomial { "true" } else { "false" },
                            if facts.right_is_constant_polynomial { "true" } else { "false" },
                            facts
                                .right_known_zero_rows
                                .as_ref()
                                .map_or_else(|| "none".to_owned(), |value| format!("some {value}")),
                            facts
                                .left_support_upper
                                .map_or_else(|| "none".to_owned(), |value| format!("some {value}")),
                            facts
                                .right_support_upper
                                .map_or_else(|| "none".to_owned(), |value| format!("some {value}")),
                            if operation.scalar_left { "true" } else { "false" },
                            if operation.scalar_right { "true" } else { "false" },
                        ),
                        ProofPayloadRule::Tensor {
                            left,
                            right,
                            left_is_constant_polynomial,
                            right_is_constant_polynomial,
                        } => {
                            if !left_is_constant_polynomial || *right_is_constant_polynomial {
                                return Err(format!(
                                    "Security0 exact-zero Tensor Result {event} is not the reached scalar shape"
                                ));
                            }
                            format!(
                                "    (leftReference := {}) (rightReference := {})\n",
                                value_ref_text(left),
                                value_ref_text(right),
                            )
                        }
                        _ => {
                            return Err(format!(
                                "Security0 exact-zero Result {event} has an unsupported typed rule"
                            ));
                        }
                    };
                    writeln!(source, "theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)\n    (selectorUpper : selector < selectorMaximum)\n    (witness : Witness document history (some selector) {modulus}) :\n    ValueClaim.Interprets {modulus} witness.env (actual selector witness)\n      (.exact LeftOperatorMerge{first_merge_event}.working .exactZero) := by\n  apply {theorem}\n    (frameStart := LeftOperatorMerge{first_merge_event}.frameStart)\n    (transferEvent := {transfer_event}) (owner := owner)\n    (leftResult := {left_result}) (rightResult := {right_result})\n    (working := LeftOperatorMerge{first_merge_event}.working)\n    (reconstruction := LeftOperatorMerge{first_merge_event}.reconstruction)\n{rule_arguments}  · rfl\n  · rfl\n  · exact LeftClaimResult{left_result}.claimSound selector selectorLower selectorUpper witness\n  · exact LeftClaimResult{right_result}.claimSound selector selectorLower selectorUpper witness\n  · exact LeftOperatorMerge{first_merge_event}.operationAgreement\n  · decide\n\ntheorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)\n    (selectorUpper : selector < selectorMaximum)\n    (witness : Witness document history (some selector) {modulus}) :\n    ExactClaimAt history {modulus} witness.env resultEvent owner\n      (actual selector witness) rawTerms summary := by\n  apply exactClaimAt_of_mergeClaim\n    (mergeClaim selector selectorLower selectorUpper witness)\n  · decide +kernel\n  · rfl").expect("String write");
                }
            } else {
                if !matches!(kind, OperationKind::Add | OperationKind::Subtract) {
                    return Err(format!("Security0 no-merge Result {event} is not reached Add/Sub"));
                }
                let ProofPayloadEvent::BoundTransfer {
                    rule: ProofPayloadRule::Sum { inputs }, ..
                } = index.event(*transfer_event)?
                else {
                    return Err(format!("Security0 no-merge Add Result {event} has no Sum row"));
                };
                let [left_ref, right_ref] = inputs.as_slice() else {
                    return Err(format!("Security0 no-merge Add Result {event} Sum is not binary"));
                };
                let (left_binding, left_position, left_expression) =
                    predecessor_ref_data(index, node.result.owner, left_ref, *left_result)?;
                let (right_binding, right_position, right_expression) =
                    predecessor_ref_data(index, node.result.owner, right_ref, *right_result)?;
                if left_zero && right_zero && output_zero {
                    let theorem = if *kind == OperationKind::Add {
                        "operatorAddNoMergeExactZeroClaimAt"
                    } else {
                        "operatorSubNoMergeExactZeroClaimAt"
                    };
                    let theorem = format!(
                        "{theorem}\n    (document := document) (history := history) \
                         (env := witness.env)"
                    );
                    writeln!(source, "theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)\n    (selectorUpper : selector < selectorMaximum)\n    (witness : Witness document history (some selector) {modulus}) :\n    ExactClaimAt history {modulus} witness.env resultEvent owner\n      (actual selector witness) rawTerms summary := by\n  apply {theorem}\n    (leftBinding := {left_binding}) (rightBinding := {right_binding})\n    (leftInputPosition := {left_position}) (rightInputPosition := {right_position})\n    (leftExpression := ⟨{left_expression}⟩) (rightExpression := ⟨{right_expression}⟩)\n    (transferEvent := {transfer_event})\n  · rfl\n  · rfl\n  · rfl\n  · rfl\n  · exact LeftClaimResult{left_result}.claimSound selector selectorLower selectorUpper witness\n  · exact LeftClaimResult{right_result}.claimSound selector selectorLower selectorUpper witness\n  · rfl\n  · decide +kernel\n  · decide").expect("String write");
                } else if *kind == OperationKind::Add && left_zero && right_zero {
                    let finite_maximum = match node.result.summary.coefficient_bound() {
                        crate::operational_noise::facts::NumericContract::Known(
                            crate::operational_noise::facts::CoefficientBound::Finite(bound),
                        ) => bound.maximum_absolute_coefficient,
                        _ => {
                            return Err(format!(
                                "Security0 survivor-fold Add Result {event} is not finite"
                            ));
                        }
                    };
                    if node.result.terms != left.terms {
                        return Err(format!(
                            "Security0 survivor-fold Add Result {event} does not retain exactly its left terms"
                        ));
                    }
                    let [right_term] = right.terms.as_slice() else {
                        return Err(format!(
                            "Security0 survivor-fold Add Result {event} right input is not a singleton"
                        ));
                    };
                    if right_term.coefficient != 1.into() {
                        return Err(format!(
                            "Security0 survivor-fold Add Result {event} right singleton coefficient is not one"
                        ));
                    }
                    let (right_coefficient_producer, _) =
                        exact_result_producers(index, *right_result)?;
                    let right_producer = data
                        .bounds
                        .iter()
                        .find(|bound| bound.event == right_coefficient_producer)
                        .ok_or_else(|| {
                            format!(
                                "Security0 survivor-fold Add Result {event} right coefficient producer \
                                 {right_coefficient_producer} is outside the bound closure"
                            )
                        })?;
                    let right_producer_namespace =
                        left_bound_namespace(right_producer.event, right_producer.rule);
                    let right_replayed = replayed_bounds
                        .get(&right_coefficient_producer)
                        .ok_or_else(|| {
                            format!(
                                "Security0 survivor-fold Add Result {event} right coefficient producer \
                                 {right_coefficient_producer} was not replayed"
                            )
                        })?;
                    let (_, summary_producer) = exact_result_producers(index, event)?;
                    let survivor_transfer = summary_producer.ok_or_else(|| {
                        format!(
                            "Security0 survivor-fold Add Result {event} has no summary producer"
                        )
                    })?;
                    let survivor_node = data
                        .bounds
                        .iter()
                        .find(|bound| bound.event == survivor_transfer)
                        .ok_or_else(|| {
                            format!(
                                "Security0 survivor-fold Add Result {event} summary producer \
                                 {survivor_transfer} is outside the bound closure"
                            )
                        })?;
                    let ProofPayloadRule::MonomialProduct { monomial, factors } =
                        survivor_node.rule
                    else {
                        return Err(format!(
                            "Security0 survivor-fold Add Result {event} summary producer is not monomial-product"
                        ));
                    };
                    let [
                        crate::operational_noise::simulation::ProofPayloadFactorEvidence {
                            bound:
                                ProofPayloadValueRef::Result {
                                    event: folded_result,
                                    projection: BoundProjection::Coefficient,
                                },
                            is_constant_polynomial: false,
                            support_upper: None,
                        },
                    ] = factors.as_slice()
                    else {
                        return Err(format!(
                            "Security0 survivor-fold Add Result {event} summary producer has unsupported inputs"
                        ));
                    };
                    if *folded_result != *right_result || *monomial != right_term.monomial {
                        return Err(format!(
                            "Security0 survivor-fold Add Result {event} summary producer does not identify its right singleton"
                        ));
                    }
                    let survivor_replayed =
                        replayed_bounds.get(&survivor_transfer).ok_or_else(|| {
                            format!(
                                "Security0 survivor-fold Add Result {event} summary producer \
                             {survivor_transfer} was not replayed"
                            )
                        })?;
                    let expected_finite = crate::operational_noise::facts::NumericContract::Known(
                        crate::operational_noise::facts::CoefficientBound::Finite(
                            crate::operational_noise::facts::BoundExpression {
                                maximum_absolute_coefficient: finite_maximum.clone(),
                            },
                        ),
                    );
                    if right_replayed != &expected_finite || survivor_replayed != &expected_finite {
                        return Err(format!(
                            "Security0 survivor-fold Add Result {event} does not replay to its recorded finite summary"
                        ));
                    }
                    let survivor_event = event.checked_sub(1).ok_or_else(|| {
                        format!("Security0 survivor-fold Add Result {event} has no preceding event")
                    })?;
                    let ProofPayloadEvent::SurvivorFold(fold) = index.event(survivor_event)? else {
                        return Err(format!(
                            "Security0 survivor-fold Add Result {event} is not preceded by SurvivorFold"
                        ));
                    };
                    if fold.coefficient != 1.into() || fold.bound != survivor_transfer {
                        return Err(format!(
                            "Security0 survivor-fold Add Result {event} has inconsistent fold evidence"
                        ));
                    }
                    let frame_start = index.immediate_frames
                        [usize::try_from(event).map_err(|_| "left Result index overflow")?]
                    .ok_or_else(|| format!("Security0 Result {event} has no invocation frame"))?;
                    let survivor_namespace =
                        left_bound_namespace(survivor_node.event, survivor_node.rule);
                    writeln!(source, "theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)\n    (selectorUpper : selector < selectorMaximum)\n    (witness : Witness document history (some selector) {modulus}) :\n    ExactClaimAt history {modulus} witness.env resultEvent owner\n      (actual selector witness) rawTerms summary := by\n  apply operatorAddSingletonSurvivorFoldClaimAt\n    (document := document) (history := history) (modulus := {modulus})\n    (witness := witness) (frameStart := {frame_start})\n    (valueType := {value_type})\n    (coefficientTransfer := {transfer_event}) (survivorTransfer := {survivor_transfer})\n    (survivorEvent := {survivor_event}) (resultEvent := resultEvent)\n    (rightCoefficientProducer := {right_coefficient_producer})\n    (owner := owner) (leftOwner := LeftClaimResult{left_result}.owner)\n    (rightOwner := LeftClaimResult{right_result}.owner)\n    (leftResult := {left_result}) (rightResult := {right_result})\n    (leftBinding := {left_binding}) (rightBinding := {right_binding})\n    (leftInputPosition := {left_position}) (rightInputPosition := {right_position})\n    (leftExpression := ⟨{left_expression}⟩) (rightExpression := ⟨{right_expression}⟩)\n    (leftActual := LeftClaimResult{left_result}.actual selector witness)\n    (rightActual := LeftClaimResult{right_result}.actual selector witness)\n    (leftRaw := LeftClaimResult{left_result}.rawTerms)\n    (survivorMonomial := {}) (maximum := ⟨{finite_maximum}, by decide⟩)\n    (rightMagnitude := {right_producer_namespace}.actual selector witness)\n    (survivorMagnitude := {survivor_namespace}.actual selector witness)\n  · decide +kernel\n  · rfl\n  · rfl\n  · rfl\n  · exact LeftClaimResult{left_result}.claimSound selector selectorLower selectorUpper witness\n  · exact LeftClaimResult{right_result}.claimSound selector selectorLower selectorUpper witness\n  · rfl\n  · exact .resultExactCoefficient (by rfl)\n      (by dsimp [{right_producer_namespace}.bound, RecordedBoundRefines] <;> decide)\n      ({right_producer_namespace}.derived selector witness)\n  · exact {survivor_namespace}.derived selector witness\n  · rfl\n  · rfl\n  · decide", monomial_text(monomial)).expect("String write");
                } else {
                    let theorem = match (*kind, left_zero, right_zero) {
                        (OperationKind::Add, false, false) => "operatorAddNoMergeClaim",
                        (OperationKind::Subtract, false, false) => "operatorSubNoMergeClaim",
                        _ => {
                            return Err(format!(
                                "Security0 finite no-merge Result {event} has an unsupported reached summary transition"
                            ));
                        }
                    };
                    let theorem = format!(
                        "{theorem}\n    (document := document) (history := history) \
                         (env := witness.env)"
                    );
                    let (_, summary_producer) = exact_result_producers(index, event)?;
                    let summary_transfer = summary_producer.ok_or_else(|| {
                        format!("Security0 no-merge Add Result {event} has no summary producer")
                    })?;
                    let frame_start = index.immediate_frames
                        [usize::try_from(event).map_err(|_| "left Result index overflow")?]
                    .ok_or_else(|| format!("Security0 Result {event} has no invocation frame"))?;
                    let finite_maximum =
                        |result: &ResultRecord| match result.summary.coefficient_bound() {
                            crate::operational_noise::facts::NumericContract::Known(
                                crate::operational_noise::facts::CoefficientBound::Finite(bound),
                            ) => Ok(bound.maximum_absolute_coefficient.clone()),
                            _ => Err(format!(
                                "Security0 finite no-merge Result {event} has a non-finite input"
                            )),
                        };
                    let left_maximum = finite_maximum(&left)?;
                    let right_maximum = finite_maximum(&right)?;
                    writeln!(source, "theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)\n    (selectorUpper : selector < selectorMaximum)\n    (witness : Witness document history (some selector) {modulus}) :\n    ExactClaimAt history {modulus} witness.env resultEvent owner\n      (actual selector witness) rawTerms summary := by\n  apply {theorem}\n    (modulus := {modulus}) (frameStart := {frame_start})\n    (resultEvent := resultEvent) (owner := owner)\n    (leftOwner := LeftClaimResult{left_result}.owner)\n    (rightOwner := LeftClaimResult{right_result}.owner)\n    (leftResult := {left_result}) (rightResult := {right_result})\n    (leftActual := LeftClaimResult{left_result}.actual selector witness)\n    (rightActual := LeftClaimResult{right_result}.actual selector witness)\n    (leftRaw := LeftClaimResult{left_result}.rawTerms)\n    (rightRaw := LeftClaimResult{right_result}.rawTerms)\n    (outputRaw := rawTerms) (leftMaximum := {left_maximum})\n    (rightMaximum := {right_maximum}) (valueType := {value_type})\n    (leftBinding := {left_binding}) (rightBinding := {right_binding})\n    (leftInputPosition := {left_position}) (rightInputPosition := {right_position})\n    (leftExpression := ⟨{left_expression}⟩) (rightExpression := ⟨{right_expression}⟩)\n    (transferEvent := {transfer_event}) (summaryTransferEvent := {summary_transfer})\n  · rfl\n  · rfl\n  · rfl\n  · rfl\n  · rfl\n  · exact LeftClaimResult{left_result}.claimSound selector selectorLower selectorUpper witness\n  · exact LeftClaimResult{right_result}.claimSound selector selectorLower selectorUpper witness\n  · rfl\n  · decide +kernel\n  · decide").expect("String write");
                }
            }
        }
    }
    writeln!(source, "end LeftClaimResult{event}\n").expect("String write");
    Ok(())
}

fn render_left_claims(
    statement: &CertificateDocumentV1,
    index: &PayloadIndex,
    data: &LeftRootRenderData<'_>,
    nodes: &[LeftClaimNode],
    relation_probes: &[RelationProbe],
    modulus: &str,
    slice: &ReachedSemanticSlice,
) -> Result<Vec<super::super::TallSecurity0GeneratedFile>, String> {
    const CHUNK_SIZE: usize = 16;
    let replayed_bounds = replay_left_bound_classes(statement, index, data)?;
    let relation_by_event =
        relation_probes.iter().map(|relation| (relation.event, relation)).collect();
    let mut files = Vec::new();
    let shard_by_event = nodes
        .iter()
        .enumerate()
        .map(|(position, node)| (node.result.event, position / CHUNK_SIZE))
        .collect::<BTreeMap<_, _>>();
    for (shard_index, shard) in nodes.chunks(CHUNK_SIZE).enumerate() {
        let residual_module = format!("ResidualLeftShard{shard_index:03}");
        let mut residual = format!(
            "import Mxx.Certificate.OperationalNoise.TallSemantics\nimport {NAMESPACE}.Proof.History\nimport {NAMESPACE}.Cert.ResidualRight\n"
        );
        let residual_dependencies = shard
            .iter()
            .flat_map(left_claim_dependencies)
            .filter_map(|event| shard_by_event.get(&event).copied())
            .filter(|dependency| *dependency != shard_index)
            .collect::<BTreeSet<_>>();
        for dependency in residual_dependencies {
            writeln!(residual, "import {NAMESPACE}.Cert.ResidualLeftShard{dependency:03}")
                .expect("String write");
        }
        residual
            .push_str("\nset_option autoImplicit false\nset_option relaxedAutoImplicit false\n\n");
        writeln!(residual, "namespace {NAMESPACE}.Cert\n").expect("String write");
        residual.push_str(
            "open Mxx.Certificate.OperationalNoise\nopen TallSecurity0ABI\nopen TallSemantics\n\n",
        );
        for node in shard {
            let event = node.result.event;
            writeln!(residual, "namespace ResidualLeftResult{event}").expect("String write");
            match node.start {
                LeftClaimStart::RightRoot => writeln!(residual, "def actual (selector : Nat)\n    (witness : Witness document history (some selector) {modulus}) : Int :=\n  ResidualRightResult{}.actual selector witness", slice.right_result),
                LeftClaimStart::Terminal { .. } => writeln!(residual, "def actual (selector : Nat)\n    (witness : Witness document history (some selector) {modulus}) : Int :=\n  witness.honestTerminalActual {event}"),
                LeftClaimStart::PriorResult { result_event } => writeln!(residual, "def actual (selector : Nat)\n    (witness : Witness document history (some selector) {modulus}) : Int :=\n  ResidualLeftResult{result_event}.actual selector witness"),
                LeftClaimStart::Operator { kind, left_result, right_result, .. } => {
                    let operator = match kind {
                        OperationKind::Add => "+",
                        OperationKind::Subtract => "-",
                        OperationKind::Multiply | OperationKind::Tensor => "*",
                        OperationKind::Direct => return Err(format!("Security0 residual Result {event} reaches Direct")),
                    };
                    writeln!(residual, "def actual (selector : Nat)\n    (witness : Witness document history (some selector) {modulus}) : Int :=\n  ResidualLeftResult{left_result}.actual selector witness {operator}\n    ResidualLeftResult{right_result}.actual selector witness")
                }
            }
            .expect("String write");
            writeln!(residual, "end ResidualLeftResult{event}\n").expect("String write");
        }
        writeln!(residual, "end {NAMESPACE}.Cert").expect("String write");
        files.push(generated_file(format!("Cert/{residual_module}.lean"), residual));

        let module = format!("SemanticLeftClaimShard{shard_index:03}");
        let dependency_shards = shard
            .iter()
            .flat_map(left_claim_dependencies)
            .filter_map(|event| shard_by_event.get(&event).copied())
            .filter(|dependency| *dependency != shard_index)
            .collect::<BTreeSet<_>>();
        let mut source = format!(
            "import Mxx.Certificate.OperationalNoise.TallSemantics\nimport {NAMESPACE}.Proof.History\nimport {NAMESPACE}.Cert.{residual_module}\nimport {NAMESPACE}.Semantic.SemanticRightRoot\nimport {NAMESPACE}.Semantic.SemanticLeftAuthority\nimport {NAMESPACE}.Semantic.SemanticLeftBound\nimport {NAMESPACE}.Semantic.SemanticLeftMergeTree\n"
        );
        for dependency in dependency_shards {
            writeln!(source, "import {NAMESPACE}.Semantic.SemanticLeftClaimShard{dependency:03}")
                .expect("String write");
        }
        source
            .push_str("\nset_option autoImplicit false\nset_option relaxedAutoImplicit false\n\n");
        writeln!(source, "namespace {NAMESPACE}.Semantic\n").expect("String write");
        source.push_str(
            "open Mxx.Certificate.OperationalNoise\nopen TallSecurity0ABI\nopen TallSemantics\n\n",
        );
        for node in shard {
            render_left_claim_node(
                &mut source,
                statement,
                index,
                data,
                &replayed_bounds,
                &relation_by_event,
                node,
                modulus,
            )?;
        }
        writeln!(source, "end {NAMESPACE}.Semantic").expect("String write");
        files.push(generated_file(format!("Semantic/{module}.lean"), source));
    }
    let mut import_level = (0..nodes.len().div_ceil(CHUNK_SIZE))
        .map(|shard| format!("SemanticLeftClaimShard{shard:03}"))
        .collect::<Vec<_>>();
    let mut depth = 0;
    while import_level.len() > 1 {
        let mut next = Vec::with_capacity(import_level.len().div_ceil(CHUNK_SIZE));
        for (position, chunk) in import_level.chunks(CHUNK_SIZE).enumerate() {
            let module = format!("SemanticLeftClaimImport{depth:02}_{position:03}");
            let mut source = String::new();
            for dependency in chunk {
                writeln!(source, "import {NAMESPACE}.Semantic.{dependency}").expect("String write");
            }
            files.push(generated_file(format!("Semantic/{module}.lean"), source));
            next.push(module);
        }
        import_level = next;
        depth += 1;
    }
    let root =
        import_level.first().ok_or_else(|| "Security0 left claim closure is empty".to_owned())?;
    files.push(generated_file(
        "Semantic/SemanticLeftClaim.lean",
        format!(
            "import {NAMESPACE}.Semantic.{root}\n\nset_option autoImplicit false\nset_option relaxedAutoImplicit false\n\nnamespace {NAMESPACE}.Semantic.SemanticLeftClaim\n\nopen Mxx.Certificate.OperationalNoise\nopen TallSecurity0ABI\nopen TallSemantics\n\ntheorem leftRootClaimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)\n    (selectorUpper : selector < selectorMaximum)\n    (witness : Witness document history (some selector) {modulus}) :\n    ExactClaimAt history {modulus} witness.env\n      LeftClaimResult{left}.resultEvent LeftClaimResult{left}.owner\n      (LeftClaimResult{left}.actual selector witness)\n      LeftClaimResult{left}.rawTerms LeftClaimResult{left}.summary := by\n  exact LeftClaimResult{left}.claimSound selector selectorLower selectorUpper witness\n\nend {NAMESPACE}.Semantic.SemanticLeftClaim\n",
            left = slice.left_result,
        ),
    ));
    let mut residual_top = String::new();
    for shard in 0..nodes.len().div_ceil(CHUNK_SIZE) {
        writeln!(residual_top, "import {NAMESPACE}.Cert.ResidualLeftShard{shard:03}")
            .expect("String write");
    }
    residual_top
        .push_str("\nset_option autoImplicit false\nset_option relaxedAutoImplicit false\n\n");
    writeln!(residual_top, "namespace {NAMESPACE}.Cert\n").expect("String write");
    residual_top.push_str(
        "open Mxx.Certificate.OperationalNoise\nopen TallSecurity0ABI\nopen TallSemantics\n\n",
    );
    writeln!(residual_top, "def statementResidual : (selector : Option Nat) →\n    Witness document history selector {modulus} → Int\n  | none, _ => 0\n  | some selector, witness =>\n      ResidualLeftResult{}.actual selector witness -\n        ResidualRightResult{}.actual selector witness\n\nend {NAMESPACE}.Cert", slice.left_result, slice.right_result)
        .expect("String write");
    files.push(generated_file("Cert/StatementResidual.lean", residual_top));
    Ok(files)
}

fn render_final_add(
    statement: &CertificateDocumentV1,
    index: &PayloadIndex,
    data: &FinalAddRenderData,
    modulus: &str,
    slice: &ReachedSemanticSlice,
) -> Result<super::super::TallSecurity0GeneratedFile, String> {
    let left_binding = slice.left_binding;
    let right_binding = slice.right_binding;
    let transfer = slice.transfer;
    let first_merge = slice.first_merge;
    #[allow(non_snake_case)]
    let LEFT_BINDING = left_binding;
    #[allow(non_snake_case)]
    let RIGHT_BINDING = right_binding;
    #[allow(non_snake_case)]
    let TRANSFER = transfer;
    #[allow(non_snake_case)]
    let FIRST_MERGE = first_merge;

    let frame_start = index.immediate_frames
        [usize::try_from(data.result.event).map_err(|_| "final Result event overflow")?]
    .ok_or_else(|| "Security0 final Result is outside an invocation frame".to_owned())?;
    let predecessor = |event, expected_position, expected_result| match index.event(event)? {
        ProofPayloadEvent::Predecessor { consumer, input_position, predecessor, source_result }
            if *consumer == data.owner &&
                *input_position == expected_position &&
                *source_result == expected_result =>
        {
            Ok(*predecessor)
        }
        _ => Err(format!("Security0 final predecessor row {event} is inconsistent")),
    };
    let left_expression = predecessor(slice.left_binding, 0, data.left.event)?;
    let right_expression = predecessor(slice.right_binding, 1, data.right.event)?;
    if expression_kind(statement, data.owner)? != Some(OperationKind::Subtract) {
        return Err("Security0 final operation is not the fixed subtract descriptor".to_owned());
    }
    let value_type = owner_value_type_text(statement, data.owner)?;
    let left_maximum = summary_bound_nat_text(&data.left.summary);
    let result_maximum = summary_bound_nat_text(&data.result.summary);
    if summary_text(&data.left.summary) != format!("(.finite {left_maximum})") ||
        summary_text(&data.right.summary) != ".exactZero" ||
        result_maximum != left_maximum
    {
        return Err("Security0 final Sub does not preserve the finite left summary".to_owned());
    }
    let (coefficient_bound, coefficient_producer, summary_producer) =
        match index.event(data.result.event)? {
            ProofPayloadEvent::Result {
                owner,
                value:
                    ProofPayloadValue::Exact {
                        coefficient_bound,
                        coefficient_producer,
                        summary_producer,
                        ..
                    },
            } if *owner == data.owner => {
                (recorded_bound_text(coefficient_bound), *coefficient_producer, *summary_producer)
            }
            _ => unreachable!("validated final exact Result"),
        };
    if coefficient_producer != transfer || summary_producer.is_some() {
        return Err("Security0 final Result producer metadata is inconsistent".to_owned());
    }
    let (end_coefficient_bound, end_coefficient_producer, end_summary_producer) =
        match index.event(data.end_event)? {
            ProofPayloadEvent::InvocationEnd {
                root,
                result:
                    ProofPayloadValue::Exact {
                        terms,
                        coefficient_bound,
                        coefficient_producer,
                        summary,
                        summary_producer,
                    },
                pre_fold_event,
            } if *root == data.owner &&
                *pre_fold_event == data.prefold_event &&
                terms == &data.result.terms &&
                summary == &data.result.summary =>
            {
                (recorded_bound_text(coefficient_bound), *coefficient_producer, *summary_producer)
            }
            _ => unreachable!("validated final InvocationEnd"),
        };
    let end_summary_producer_text =
        end_summary_producer.map_or_else(|| "none".to_owned(), |event| format!("some {event}"));
    if coefficient_bound != end_coefficient_bound ||
        coefficient_producer != end_coefficient_producer ||
        summary_producer != end_summary_producer
    {
        return Err("Security0 final Result and InvocationEnd producer metadata differ".to_owned());
    }
    let mut source = format!(
        "import {NAMESPACE}.Semantic.SemanticLeftClaim\n\
         import {NAMESPACE}.Semantic.SemanticRightRoot\n\
         import {NAMESPACE}.Semantic.SemanticLeftMergeTree\n\
         import {NAMESPACE}.Cert.StatementResidual\n\
         import {NAMESPACE}.Proof.Proof\n\n\
         set_option autoImplicit false\n\
         set_option relaxedAutoImplicit false\n\n\
         namespace {NAMESPACE}.Semantic.SemanticFinal\n\n\
         open Mxx.Certificate.OperationalNoise\n\
         open TallSecurity0ABI\n\
         open TallSemantics\n\n"
    );
    writeln!(source, "def owner : Owner := {}", owner_text(data.owner)).expect("String write");
    writeln!(source, "def resultEvent : Nat := {}", data.result.event).expect("String write");
    writeln!(source, "def preFoldEvent : Nat := {}", data.prefold_event).expect("String write");
    writeln!(source, "def endEvent : Nat := {}", data.end_event).expect("String write");
    writeln!(source, "def rawTerms : List Term := {}", raw_terms_text(&data.result.terms))
        .expect("String write");
    writeln!(source, "def summary : Bound := .finite {result_maximum}").expect("String write");
    writeln!(source, "def actual (selector : Nat)\n    (witness : Witness document history (some selector) {modulus}) : Int :=\n  Cert.statementResidual (some selector) witness\n")
        .expect("String write");
    writeln!(source, "theorem resultAt : history.lookup resultEvent = some\n    ⟨.resultExact owner rawTerms {end_coefficient_bound} {end_coefficient_producer}\n      summary {end_summary_producer_text}, {frame_start}⟩ := by\n  rfl\n")
        .expect("String write");
    writeln!(source, "theorem resultClaimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)\n    (selectorUpper : selector < selectorMaximum)\n    (witness : Witness document history (some selector) {modulus}) :\n    ExactClaimAt history {modulus} witness.env resultEvent owner\n      (actual selector witness) rawTerms summary := by\n  apply operatorSubFiniteLeftMergeClaimAt\n    (document := document) (history := history) (modulus := {modulus})\n    (frameStart := {frame_start}) (coefficientTransfer := {TRANSFER})\n    (resultEvent := resultEvent) (env := witness.env) (owner := owner)\n    (leftOwner := LeftClaimResult{}.owner)\n    (rightOwner := SemanticRightRootResult{}.owner)\n    (leftResult := {}) (rightResult := {})\n    (leftBinding := {LEFT_BINDING}) (rightBinding := {RIGHT_BINDING})\n    (leftInputPosition := 0) (rightInputPosition := 1)\n    (leftExpression := ⟨{left_expression}⟩) (rightExpression := ⟨{right_expression}⟩)\n    (leftActual := LeftClaimResult{}.actual selector witness)\n    (rightActual := SemanticRightRootResult{}.actual selector witness)\n    (leftRaw := LeftClaimResult{}.rawTerms)\n    (rightRaw := SemanticRightRootResult{}.rawTerms)\n    (outputRaw := rawTerms) (leftMaximum := {left_maximum})\n    (base := LeftOperatorMerge{FIRST_MERGE}.base) (valueType := {value_type})\n    (coefficientBound := {coefficient_bound})\n    (reconstruction := LeftOperatorMerge{FIRST_MERGE}.reconstruction)\n  · rfl\n  · rfl\n  · rfl\n  · rfl\n  · exact SemanticLeftClaim.leftRootClaimSound selector selectorLower selectorUpper witness\n  · exact SemanticRightRoot.rightRootClaimSound selector selectorLower selectorUpper witness\n  · exact LeftOperatorMerge{FIRST_MERGE}.operationAgreement\n  · exact resultAt\n  · decide\n", data.left.event, data.right.event, data.left.event, data.right.event, data.left.event, data.right.event, data.left.event, data.right.event)
        .expect("String write");
    writeln!(source, "theorem preFoldAt : history.lookup preFoldEvent = some\n    ⟨.preFoldPolynomial resultEvent rawTerms summary\n      (some (.result resultEvent .summary)), {frame_start}⟩ := by\n  rfl\n")
        .expect("String write");
    writeln!(source, "theorem invocationEndAt : history.lookup endEvent = some\n    ⟨.invocationEndExact owner preFoldEvent rawTerms {end_coefficient_bound}\n      {end_coefficient_producer} summary {end_summary_producer_text}, {frame_start}⟩ := by\n  rfl\n")
        .expect("String write");
    writeln!(source, "theorem invocationEndClaimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)\n    (selectorUpper : selector < selectorMaximum)\n    (witness : Witness document history (some selector) {modulus}) :\n    ValueClaim.Interprets {modulus} witness.env (actual selector witness)\n      (.exact (rawTerms.map Term.toExact) summary) := by\n  exact invocationEndSound {modulus} witness.env (actual selector witness)\n    (rawTerms.map Term.toExact) (rawTerms.map Term.toExact) summary summary\n    (resultClaimSound selector selectorLower selectorUpper witness).claim rfl rfl\n")
        .expect("String write");
    writeln!(source, "theorem strictBoundSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)\n    (selectorUpper : selector < selectorMaximum)\n    (witness : Witness document history (some selector) {modulus}) :\n    2 * {} * centeredNorm {modulus} (actual selector witness) < {modulus} := by\n  apply finalStrictBound_of_empty_finite_claim {} {modulus} witness.env\n    (actual selector witness) {result_maximum}\n  · simpa [rawTerms, summary] using\n      invocationEndClaimSound selector selectorLower selectorUpper witness\n  · decide\n  · decide\n", statement.plaintext_modulus, statement.plaintext_modulus)
        .expect("String write");
    writeln!(source, "theorem fixedSemanticSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)\n    (selectorUpper : selector < selectorMaximum)\n    (witness : Witness document history (some selector) {modulus}) :\n    history.lookup resultEvent = some\n        ⟨.resultExact owner rawTerms {end_coefficient_bound} {end_coefficient_producer}\n          summary {end_summary_producer_text}, {frame_start}⟩ ∧\n      history.lookup preFoldEvent = some\n        ⟨.preFoldPolynomial resultEvent rawTerms summary\n          (some (.result resultEvent .summary)), {frame_start}⟩ ∧\n      history.lookup endEvent = some\n        ⟨.invocationEndExact owner preFoldEvent rawTerms {end_coefficient_bound}\n          {end_coefficient_producer} summary {end_summary_producer_text}, {frame_start}⟩ ∧\n      ValueClaim.Interprets {modulus} witness.env (actual selector witness)\n        (.exact (rawTerms.map Term.toExact) summary) ∧\n      2 * {} * centeredNorm {modulus} (actual selector witness) < {modulus} := by\n  exact ⟨resultAt, preFoldAt, invocationEndAt,\n    invocationEndClaimSound selector selectorLower selectorUpper witness,\n    strictBoundSound selector selectorLower selectorUpper witness⟩\n", statement.plaintext_modulus)
        .expect("String write");
    writeln!(source, "theorem fixedAcceptance :\n    Security0Accepted document history {} {modulus} ringDimension endEvent preFoldEvent resultEvent\n      {result_maximum} owner rawTerms {end_coefficient_bound} {end_coefficient_producer}\n      summary {end_summary_producer_text} Cert.statementResidual := by\n  refine ⟨proofValid, rfl, rfl, rfl, rfl, rfl, ⟨{frame_start}, resultAt, preFoldAt, invocationEndAt⟩, ?_⟩\n  change ∀ selector, selectorMinimum ≤ selector → selector < selectorMaximum →\n    ∀ witness : Witness document history (some selector) {modulus}, _\n  intro selector selectorLower selectorUpper witness\n  exact ⟨by\n    simpa [actual, Cert.statementResidual] using\n      invocationEndClaimSound selector selectorLower selectorUpper witness, by\n    simpa [actual, Cert.statementResidual] using\n      strictBoundSound selector selectorLower selectorUpper witness⟩\n", statement.plaintext_modulus)
        .expect("String write");
    writeln!(source, "end {NAMESPACE}.Semantic.SemanticFinal").expect("String write");
    Ok(generated_file("Semantic/SemanticFinal.lean", source))
}

fn render_left_merge_deltas(
    statement: &CertificateDocumentV1,
    index: &PayloadIndex,
    data: &LeftRootRenderData<'_>,
    relation_probes: &[RelationProbe],
) -> Result<Vec<super::super::TallSecurity0GeneratedFile>, String> {
    const CHUNK_SIZE: usize = 16;
    let mut files = Vec::new();
    let relation_by_event = relation_probes
        .iter()
        .map(|relation| (relation.event, relation))
        .collect::<BTreeMap<_, _>>();
    for (shard_index, shard) in data.merges.chunks(CHUNK_SIZE).enumerate() {
        let module = format!("SemanticLeftMergeDeltaShard{shard_index:03}");
        let mut source = format!(
            "import Mxx.Certificate.OperationalNoise.TallSemantics\nimport {NAMESPACE}.Proof.History\n"
        );
        source
            .push_str("\nset_option autoImplicit false\nset_option relaxedAutoImplicit false\n\n");
        writeln!(source, "namespace {NAMESPACE}.Semantic\n").expect("String write");
        source.push_str(
            "open Mxx.Certificate.OperationalNoise\nopen TallSecurity0ABI\nopen TallSemantics\nopen EventReplay\n\n",
        );
        for node in shard {
            let event = node.event;
            let frame = index.immediate_frames
                [usize::try_from(event).map_err(|_| "left merge event index overflow")?]
            .ok_or_else(|| format!("left merge event {event} has no immediate frame"))?;
            writeln!(source, "namespace LeftMerge{event}").expect("String write");
            writeln!(source, "def owner : Owner := {}", owner_text(node.merge.owner))
                .expect("String write");
            writeln!(source, "def mergeEvent : Nat := {event}\ndef frameStart : Nat := {frame}")
                .expect("String write");
            let delta = ProofPayloadTerm {
                coefficient: node.merge.signed_contribution.clone(),
                monomial: node.merge.output.clone(),
            };
            writeln!(source, "def delta : ExactTerm Owner := {}", term_text(&delta),)
                .expect("String write");
            match &node.merge.source {
                ProofPayloadCoefficientMergeSource::Operator { inputs } => {
                    let left = index.result(inputs[0].value_event)?;
                    let right = index.result(inputs[1].value_event)?;
                    let left_ordinal = usize::try_from(inputs[0].term_ordinal)
                        .map_err(|_| format!("left merge event {event} term ordinal overflow"))?;
                    let right_ordinal = usize::try_from(inputs[1].term_ordinal)
                        .map_err(|_| format!("left merge event {event} term ordinal overflow"))?;
                    let left_term = left.terms.get(left_ordinal).ok_or_else(|| {
                        format!("left merge event {event} left term is out of range")
                    })?;
                    let right_term = right.terms.get(right_ordinal).ok_or_else(|| {
                        format!("left merge event {event} right term is out of range")
                    })?;
                    writeln!(source, "def leftRaw : List Term := {}", raw_terms_text(&left.terms))
                        .expect("String write");
                    writeln!(
                        source,
                        "def rightRaw : List Term := {}",
                        raw_terms_text(&right.terms)
                    )
                    .expect("String write");
                    writeln!(
                        source,
                        "def group : MergeGroup := .operator {} {}",
                        inputs[0].value_event, inputs[1].value_event
                    )
                    .expect("String write");
                    writeln!(source, "theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by\n  unfold group delta\n  apply MergeDeltaAt.operator (leftResult := {}) (leftOrdinal := {})\n    (rightResult := {}) (rightOrdinal := {}) (leftTerms := leftRaw)\n    (rightTerms := rightRaw) (leftTerm := {}) (rightTerm := {})\n    (output := {}) (signedContribution := ({})) <;> rfl",
                        inputs[0].value_event,
                        inputs[0].term_ordinal,
                        inputs[1].value_event,
                        inputs[1].term_ordinal,
                        raw_term_text(left_term),
                        raw_term_text(right_term),
                        monomial_text(&node.merge.output),
                        node.merge.signed_contribution,
                    ).expect("String write");
                }
                ProofPayloadCoefficientMergeSource::Relation {
                    application,
                    source_term_ordinal,
                } => {
                    let (source_monomial, outer, start, end, rule) = match index
                        .event(*application)?
                    {
                        ProofPayloadEvent::AppliedRelation {
                            owner,
                            source_monomial,
                            outer_coefficient,
                            ordered_start,
                            ordered_end_exclusive,
                            rule,
                        } if *owner == node.merge.owner => (
                            source_monomial,
                            outer_coefficient,
                            ordered_start,
                            ordered_end_exclusive,
                            rule,
                        ),
                        _ => {
                            return Err(format!(
                                "left relation merge event {event} has an inconsistent application {application}"
                            ));
                        }
                    };
                    let rhs_event = match rule {
                        ProofPayloadRelationRule::Universal { rhs_result, .. } => *rhs_result,
                        ProofPayloadRelationRule::Gadget { input_result, .. } => *input_result,
                    };
                    let rhs = index.result(rhs_event)?;
                    let ordinal = usize::try_from(*source_term_ordinal).map_err(|_| {
                        format!("left relation merge event {event} term ordinal overflow")
                    })?;
                    let rhs_term = rhs.terms.get(ordinal).ok_or_else(|| {
                        format!("left relation merge event {event} source term is out of range")
                    })?;
                    writeln!(source, "def rhsRaw : List Term := {}", raw_terms_text(&rhs.terms))
                        .expect("String write");
                    writeln!(source, "def group : MergeGroup := .relation {application}")
                        .expect("String write");
                    writeln!(source, "theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by\n  unfold group delta\n  apply MergeDeltaAt.relation (application := {application}) (rhsResult := {rhs_event})\n    (sourceTermOrdinal := {source_term_ordinal}) (source := {})\n    (outerCoefficient := ({outer})) (orderedStart := {start})\n    (orderedEndExclusive := {end}) (rule := {}) (rhsTerms := rhsRaw)\n    (rhsTerm := {}) (output := {}) (signedContribution := ({})) <;> rfl",
                        monomial_text(source_monomial),
                        relation_rule_text(rule),
                        raw_term_text(rhs_term),
                        monomial_text(&node.merge.output),
                        node.merge.signed_contribution,
                    ).expect("String write");
                }
            }
            writeln!(source, "end LeftMerge{event}\n").expect("String write");
        }
        writeln!(source, "end {NAMESPACE}.Semantic").expect("String write");
        files.push(generated_file(format!("Semantic/{module}.lean"), source));
    }
    let merge_shards = data
        .merges
        .iter()
        .enumerate()
        .map(|(position, node)| (node.event, position / CHUNK_SIZE))
        .collect::<BTreeMap<_, _>>();
    let mut groups = BTreeMap::<LeftMergeGroupKey, Vec<&LeftRootMergeNode<'_>>>::new();
    for node in &data.merges {
        let frame = index.immediate_frames
            [usize::try_from(node.event).map_err(|_| "left merge event index overflow")?]
        .ok_or_else(|| format!("left merge event {} has no immediate frame", node.event))?;
        let key = match &node.merge.source {
            ProofPayloadCoefficientMergeSource::Operator { inputs } => {
                LeftMergeGroupKey::Operator {
                    frame,
                    owner: node.merge.owner,
                    left_result: inputs[0].value_event,
                    right_result: inputs[1].value_event,
                }
            }
            ProofPayloadCoefficientMergeSource::Relation { application, .. } => {
                LeftMergeGroupKey::Relation {
                    frame,
                    owner: node.merge.owner,
                    application: *application,
                }
            }
        };
        groups.entry(key).or_default().push(node);
    }
    let groups = groups.into_iter().collect::<Vec<_>>();
    for (shard_index, shard) in groups.chunks(CHUNK_SIZE).enumerate() {
        let module = format!("SemanticLeftMergeTreeShard{shard_index:03}");
        let mut source = String::new();
        let leaf_shards = shard
            .iter()
            .flat_map(|(_, nodes)| nodes.iter().map(|node| merge_shards[&node.event]))
            .collect::<BTreeSet<_>>();
        for leaf_shard in leaf_shards {
            writeln!(
                source,
                "import {NAMESPACE}.Semantic.SemanticLeftMergeDeltaShard{leaf_shard:03}"
            )
            .expect("String write");
        }
        source
            .push_str("\nset_option autoImplicit false\nset_option relaxedAutoImplicit false\n\n");
        writeln!(source, "namespace {NAMESPACE}.Semantic\n").expect("String write");
        source.push_str(
            "open Mxx.Certificate.OperationalNoise\nopen TallSecurity0ABI\nopen TallSemantics\n\n",
        );
        for (key, nodes) in shard {
            let first_event = nodes[0].event;
            let (namespace, frame, owner, group) = match key {
                LeftMergeGroupKey::Operator { frame, owner, left_result, right_result } => (
                    format!("LeftOperatorMerge{first_event}"),
                    frame,
                    owner,
                    format!(".operator {left_result} {right_result}"),
                ),
                LeftMergeGroupKey::Relation { frame, owner, application } => (
                    format!("LeftRelationMerge{application}"),
                    frame,
                    owner,
                    format!(".relation {application}"),
                ),
            };
            writeln!(source, "namespace {namespace}").expect("String write");
            writeln!(source, "def frameStart : Nat := {frame}").expect("String write");
            writeln!(source, "def owner : Owner := {}", owner_text(*owner)).expect("String write");
            writeln!(source, "def group : MergeGroup := {group}").expect("String write");
            let mut level = Vec::<(String, String)>::new();
            for (position, node) in nodes.iter().enumerate() {
                let deltas = format!("deltas0_{position}");
                let rows = format!("rows0_{position}");
                writeln!(
                    source,
                    "def {deltas} : Polynomial Owner := [LeftMerge{}.delta]",
                    node.event
                )
                .expect("String write");
                writeln!(source, "theorem {rows} : MergeDeltasAt history frameStart owner group {deltas} := by\n  exact .leaf LeftMerge{}.deltaAt", node.event).expect("String write");
                level.push((deltas, rows));
            }
            let mut depth = 1;
            while level.len() > 1 {
                let mut next = Vec::with_capacity(level.len().div_ceil(2));
                for (position, pair) in level.chunks(2).enumerate() {
                    if let [(left_deltas, left_rows), (right_deltas, right_rows)] = pair {
                        let deltas = format!("deltas{depth}_{position}");
                        let rows = format!("rows{depth}_{position}");
                        writeln!(
                            source,
                            "def {deltas} : Polynomial Owner := {left_deltas} ++ {right_deltas}"
                        )
                        .expect("String write");
                        writeln!(source, "theorem {rows} : MergeDeltasAt history frameStart owner group {deltas} := by\n  exact .append {left_rows} {right_rows}").expect("String write");
                        next.push((deltas, rows));
                    } else {
                        next.push(pair[0].clone());
                    }
                }
                level = next;
                depth += 1;
            }
            let (root_deltas, root_rows) = &level[0];
            writeln!(source, "abbrev deltas : Polynomial Owner := {root_deltas}")
                .expect("String write");
            writeln!(
                source,
                "theorem rows : MergeDeltasAt history frameStart owner group deltas := {root_rows}"
            )
            .expect("String write");
            match key {
                LeftMergeGroupKey::Operator { left_result, right_result, .. } => {
                    let left = index.result(*left_result)?;
                    let right = index.result(*right_result)?;
                    let kind = expression_kind(statement, *owner)?.ok_or_else(|| {
                        format!(
                            "Security0 operator merge {first_event} owner has no reached matrix operation"
                        )
                    })?;
                    let (base, working, operation) = match kind {
                        OperationKind::Add | OperationKind::Subtract => {
                            let (base, working) =
                                add_sub_merge_polynomials(kind, &left.terms, &right.terms)?;
                            let operation = if kind == OperationKind::Add {
                                "add left right".to_owned()
                            } else {
                                "subtract left right".to_owned()
                            };
                            (base, working, operation)
                        }
                        OperationKind::Multiply | OperationKind::Tensor => {
                            let merge_terms = nodes
                                .iter()
                                .map(|node| match &node.merge.source {
                                    ProofPayloadCoefficientMergeSource::Operator { inputs } => {
                                        let left_term =
                                            &left.terms[usize::try_from(inputs[0].term_ordinal)
                                                .expect("validated left merge ordinal")];
                                        let right_term =
                                            &right.terms[usize::try_from(inputs[1].term_ordinal)
                                                .expect("validated right merge ordinal")];
                                        (
                                            left_term.monomial.clone(),
                                            right_term.monomial.clone(),
                                            node.merge.output.clone(),
                                        )
                                    }
                                    ProofPayloadCoefficientMergeSource::Relation { .. } => {
                                        unreachable!("operator group contains only operator rows")
                                    }
                                })
                                .collect::<Vec<_>>();
                            let flags = if kind == OperationKind::Tensor {
                                let transfer = operator_transfer_for_group(
                                    index,
                                    *owner,
                                    *frame,
                                    first_event,
                                    kind,
                                    *left_result,
                                    *right_result,
                                )?;
                                match index.event(transfer)? {
                                    ProofPayloadEvent::BoundTransfer {
                                        rule:
                                            ProofPayloadRule::Tensor {
                                                left_is_constant_polynomial: true,
                                                right_is_constant_polynomial: false,
                                                ..
                                            },
                                        ..
                                    } => (true, false),
                                    _ => {
                                        return Err(format!(
                                            "Security0 Tensor merge {first_event} does not use the reached true/false transfer"
                                        ));
                                    }
                                }
                            } else {
                                resolve_scalar_flags(
                                    &matching_scalar_flags_from_merges(&merge_terms),
                                    &[left.clone(), right.clone()],
                                )?
                            };
                            (
                                Vec::new(),
                                product_terms_for_scalar_flags(
                                    &left.terms,
                                    &right.terms,
                                    flags.0,
                                    flags.1,
                                ),
                                format!("productPoly left right {} {}", flags.0, flags.1),
                            )
                        }
                        OperationKind::Direct => {
                            return Err(format!(
                                "Security0 operator merge {first_event} reaches unsupported {kind:?}"
                            ));
                        }
                    };
                    writeln!(source, "def left : Polynomial Owner := {}", terms_text(&left.terms))
                        .expect("String write");
                    writeln!(
                        source,
                        "def right : Polynomial Owner := {}",
                        terms_text(&right.terms)
                    )
                    .expect("String write");
                    writeln!(source, "def base : Polynomial Owner := {}", terms_text(&base))
                        .expect("String write");
                    writeln!(source, "def working : Polynomial Owner := {}", terms_text(&working))
                        .expect("String write");
                    source.push_str("def reconstruction : MergeReconstructionAt history frameStart owner group base working :=\n  { deltas := deltas\n    rows := rows\n    agreement := by decide +kernel }\n");
                    writeln!(source, "theorem operationAgreement : CanonicalAgreement (add base reconstruction.deltas) ({operation}) := by\n  dsimp [reconstruction]\n  decide +kernel")
                        .expect("String write");
                }
                LeftMergeGroupKey::Relation { application, .. } => {
                    let relation = relation_by_event.get(application).ok_or_else(|| {
                        format!(
                            "Security0 relation merge application {application} has no semantic probe"
                        )
                    })?;
                    writeln!(
                        source,
                        "def accumulator : Polynomial Owner := {}",
                        terms_text(&relation.accumulator.terms)
                    )
                    .expect("String write");
                    writeln!(
                        source,
                        "def source : MonomialKey Owner := {}",
                        monomial_text(&relation.source)
                    )
                    .expect("String write");
                    writeln!(
                        source,
                        "def rhs : Polynomial Owner := {}",
                        terms_text(&relation.rhs.terms)
                    )
                    .expect("String write");
                    writeln!(
                        source,
                        "def base : Polynomial Owner := subtract accumulator [{{ coefficient := ({}), key := source }}]",
                        relation.outer
                    )
                    .expect("String write");
                    writeln!(
                        source,
                        "def working : Polynomial Owner := {}",
                        terms_text(&relation.output.terms)
                    )
                    .expect("String write");
                    source.push_str("def reconstruction : MergeReconstructionAt history frameStart owner group base working :=\n  { deltas := deltas\n    rows := rows\n    agreement := by decide +kernel }\n");
                    writeln!(source, "theorem relationAgreement : CanonicalAgreement (add base reconstruction.deltas)\n    (relationPoly accumulator source\n      (relationContext source source.centralFactors {} {}) ({}) rhs) := by\n  dsimp [reconstruction]\n  decide +kernel", relation.start, relation.end, relation.outer)
                        .expect("String write");
                }
            }
            writeln!(source, "end {namespace}\n").expect("String write");
        }
        writeln!(source, "end {NAMESPACE}.Semantic").expect("String write");
        files.push(generated_file(format!("Semantic/{module}.lean"), source));
    }
    let mut import_level = (0..groups.len().div_ceil(CHUNK_SIZE))
        .map(|shard| format!("SemanticLeftMergeTreeShard{shard:03}"))
        .collect::<Vec<_>>();
    let mut depth = 0;
    while import_level.len() > 1 {
        let mut next = Vec::with_capacity(import_level.len().div_ceil(CHUNK_SIZE));
        for (position, chunk) in import_level.chunks(CHUNK_SIZE).enumerate() {
            let module = format!("SemanticLeftMergeTreeImport{depth:02}_{position:03}");
            let mut source = String::new();
            for dependency in chunk {
                writeln!(source, "import {NAMESPACE}.Semantic.{dependency}").expect("String write");
            }
            files.push(generated_file(format!("Semantic/{module}.lean"), source));
            next.push(module);
        }
        import_level = next;
        depth += 1;
    }
    let root =
        import_level.first().ok_or_else(|| "Security0 left merge tree is empty".to_owned())?;
    files.push(generated_file(
        "Semantic/SemanticLeftMergeTree.lean",
        format!("import {NAMESPACE}.Semantic.{root}\n"),
    ));
    Ok(files)
}

fn render_left_authorities(
    index: &PayloadIndex,
    data: &LeftRootRenderData<'_>,
    modulus: &str,
) -> Result<Vec<super::super::TallSecurity0GeneratedFile>, String> {
    const CHUNK_SIZE: usize = 16;
    let authorities = data
        .bounds
        .iter()
        .filter_map(|node| match node.rule {
            ProofPayloadRule::Authority(authority) => Some((node, authority)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut files = Vec::new();
    for (shard_index, shard) in authorities.chunks(CHUNK_SIZE).enumerate() {
        let module = format!("SemanticLeftAuthorityShard{shard_index:03}");
        let mut source = shard_index.checked_sub(1).map_or_else(
            || format!("import Mxx.Certificate.OperationalNoise.TallSemantics\nimport {NAMESPACE}.Proof.History\n"),
            |previous| format!("import {NAMESPACE}.Semantic.SemanticLeftAuthorityShard{previous:03}\n"),
        );
        source
            .push_str("\nset_option autoImplicit false\nset_option relaxedAutoImplicit false\n\n");
        writeln!(source, "namespace {NAMESPACE}.Semantic\n").expect("String write");
        source.push_str("open Mxx.Certificate.OperationalNoise\nopen TallSecurity0ABI\nopen TallSemantics\nopen EventReplay\n\n");
        for (node, authority) in shard {
            let event = node.event;
            let result_event = data.authority_results[&event];
            let frame = index.immediate_frames
                [usize::try_from(event).map_err(|_| "authority event index overflow")?]
            .ok_or_else(|| format!("authority event {event} has no frame"))?;
            let (recorded, constructor) = match index.event(result_event)? {
                ProofPayloadEvent::Result {
                    value: ProofPayloadValue::Coefficient { bound },
                    ..
                } => (bound, ".resultCoefficient (by decide) (by rfl) (by rfl)".to_owned()),
                ProofPayloadEvent::Result {
                    value:
                        ProofPayloadValue::Exact {
                            terms,
                            coefficient_bound,
                            summary,
                            summary_producer,
                            ..
                        },
                    ..
                } => (
                    coefficient_bound,
                    format!(
                        ".resultExact (terms := {}) (recordedCoefficientBound := {}) \
                         (summary := {}) (summaryProducer := {}) (by decide) (by rfl) \
                         (by rfl) (by simp [bound, RecordedBoundRefines])",
                        raw_terms_text(terms),
                        recorded_bound_text(coefficient_bound),
                        summary_text(summary),
                        summary_producer
                            .map_or_else(|| "none".to_owned(), |event| format!("some {event}")),
                    ),
                ),
                _ => unreachable!("authority gate accepted only Results"),
            };
            writeln!(source, "namespace LeftAuthority{event}").expect("String write");
            writeln!(source, "def owner : Owner := {}", owner_text(node.owner))
                .expect("String write");
            writeln!(source, "def authority : Authority := {}", authority_text(authority)?)
                .expect("String write");
            writeln!(source, "def bound : CoeffClass := {}", coeff_class_text(recorded)?)
                .expect("String write");
            writeln!(source, "def producerEvent : Nat := {event}\ndef resultEvent : Nat := {result_event}\ndef frameStart : Nat := {frame}").expect("String write");
            writeln!(source, "theorem leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound := by\n  exact {constructor}").expect("String write");
            writeln!(source, "def actual (selector : Nat)\n    (witness : Witness document history (some selector) {modulus}) : Nat :=\n  witness.authorityMagnitude resultEvent\ntheorem derived (selector : Nat)\n    (witness : Witness document history (some selector) {modulus}) :\n    BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound\n      (actual selector witness) := by\n  exact .authority witness.toAuthorityWitness leaf\nend LeftAuthority{event}\n").expect("String write");
        }
        writeln!(source, "end {NAMESPACE}.Semantic").expect("String write");
        files.push(generated_file(format!("Semantic/{module}.lean"), source));
    }
    let last = authorities.len().div_ceil(CHUNK_SIZE) - 1;
    files.push(generated_file(
        "Semantic/SemanticLeftAuthority.lean",
        format!("import {NAMESPACE}.Semantic.SemanticLeftAuthorityShard{last:03}\n"),
    ));
    Ok(files)
}

fn left_bound_namespace(event: u64, rule: &ProofPayloadRule) -> String {
    if matches!(rule, ProofPayloadRule::Authority(_)) {
        format!("LeftAuthority{event}")
    } else {
        format!("LeftBound{event}")
    }
}

fn left_factor_text(
    factor: &crate::operational_noise::simulation::ProofPayloadFactorEvidence,
) -> String {
    format!(
        "⟨{}, {}, {}⟩",
        value_ref_text(&factor.bound),
        if factor.is_constant_polynomial { "true" } else { "false" },
        factor.support_upper.map_or_else(|| "none".to_owned(), |value| format!("some {value}")),
    )
}

fn left_rule_text(rule: &ProofPayloadRule) -> Result<String, String> {
    Ok(match rule {
        ProofPayloadRule::Identity { input } => {
            format!(".identity ({})", value_ref_text(input))
        }
        ProofPayloadRule::Sum { inputs } => {
            format!(".sum [{}]", inputs.iter().map(value_ref_text).collect::<Vec<_>>().join(", "))
        }
        ProofPayloadRule::Scale { value, scale } => format!(
            ".scale ({}) ({})",
            value_ref_text(value),
            match scale {
                crate::operational_noise::simulation::ProofPayloadScale::Value(reference) => {
                    format!(".value ({})", value_ref_text(reference))
                }
                crate::operational_noise::simulation::ProofPayloadScale::Magnitude(value) => {
                    format!(".magnitude {value}")
                }
            }
        ),
        ProofPayloadRule::MonomialProduct { monomial, factors } => format!(
            ".monomialProduct {} [{}]",
            monomial_text(monomial),
            factors.iter().map(left_factor_text).collect::<Vec<_>>().join(", ")
        ),
        ProofPayloadRule::Product { .. } => rule_text(rule),
        _ => return Err(format!("unsupported reached left bound rule {rule:?}")),
    })
}

fn reached_product_shape(
    statement: &CertificateDocumentV1,
    owner: ProofPayloadOwner,
    facts: &crate::operational_noise::bound::MatrixProductFacts,
) -> Result<(usize, usize, usize, usize, usize, usize), String> {
    let row = statement
        .expressions
        .get(usize::try_from(owner.expression_row).map_err(|_| "product owner row overflow")?)
        .ok_or_else(|| format!("product owner {} has no expression row", owner_text(owner)))?;
    let [left_row, right_row] = row.inputs.as_slice() else {
        return Err(format!(
            "product owner {} does not have two statement inputs",
            owner_text(owner)
        ));
    };
    let matrix = |expression: u64| -> Result<(usize, usize, usize), String> {
        let row = statement
            .expressions
            .get(usize::try_from(expression).map_err(|_| "product input row overflow")?)
            .ok_or_else(|| format!("product input expression {expression} is missing"))?;
        let value_type = match &row.descriptor {
            CanonicalExpressionDescriptor::Operation { value_type, .. } => value_type,
            CanonicalExpressionDescriptor::Source { source } => {
                let source = match source {
                    crate::operational_noise::g0::CanonicalExpressionSource::Direct { source } |
                    crate::operational_noise::g0::CanonicalExpressionSource::Family {
                        source,
                        ..
                    } => *source,
                };
                match statement
                    .sources
                    .get(usize::try_from(source).map_err(|_| "product source row overflow")?)
                    .ok_or_else(|| format!("product source row {source} is missing"))?
                {
                    crate::operational_noise::certificate_schema::CertificateSourceRowV1::Constant {
                        value,
                    } => &value.value_type,
                    crate::operational_noise::certificate_schema::CertificateSourceRowV1::Direct {
                        identity,
                        ..
                    } => &identity.value_type,
                    crate::operational_noise::certificate_schema::CertificateSourceRowV1::Family {
                        identity,
                        ..
                    } => &identity.element_type,
                }
            }
            CanonicalExpressionDescriptor::Event { operator } => {
                let event = match operator {
                    crate::operational_noise::g0::CanonicalEventOperator::Sample { event } |
                    crate::operational_noise::g0::CanonicalEventOperator::Sampler { event } => {
                        event.row
                    }
                    crate::operational_noise::g0::CanonicalEventOperator::GadgetDecompose {
                        events,
                    } => {
                        events
                            .first()
                            .ok_or_else(|| {
                                format!("product event expression {expression} has no event rows")
                            })?
                            .row
                    }
                };
                match statement
                    .events
                    .get(usize::try_from(event).map_err(|_| "product event row overflow")?)
                    .ok_or_else(|| format!("product event row {event} is missing"))?
                {
                    crate::operational_noise::certificate_schema::CertificateEventRowV1::Sample {
                        descriptor,
                        ..
                    } => &descriptor.output_type,
                    crate::operational_noise::certificate_schema::CertificateEventRowV1::Sampler {
                        operation,
                        ..
                    } => match operation {
                        crate::operational_noise::g0::StableSamplerOperation::UniformResidue {
                            output,
                        } |
                        crate::operational_noise::g0::StableSamplerOperation::UniformInterval {
                            output,
                            ..
                        } |
                        crate::operational_noise::g0::StableSamplerOperation::Gaussian {
                            output,
                            ..
                        } |
                        crate::operational_noise::g0::StableSamplerOperation::Hash {
                            output,
                            ..
                        } |
                        crate::operational_noise::g0::StableSamplerOperation::Trapdoor {
                            output,
                            ..
                        } |
                        crate::operational_noise::g0::StableSamplerOperation::Preimage {
                            output,
                            ..
                        } => output,
                    },
                    crate::operational_noise::certificate_schema::CertificateEventRowV1::GadgetDecompose {
                        output,
                        ..
                    } => output,
                }
            }
        };
        let crate::operational_noise::g0::StableValueType::Matrix {
            ring_dimension,
            rows,
            columns,
            ..
        } = value_type
        else {
            return Err(format!("product input expression {expression} is not matrix-typed"));
        };
        Ok((*rows, *columns, *ring_dimension))
    };
    let (left_rows, left_columns, ring_dimension) = matrix(*left_row)?;
    let (right_rows, right_columns, right_ring_dimension) = matrix(*right_row)?;
    if ring_dimension != right_ring_dimension || ring_dimension == 0 {
        return Err(format!("product owner {} has inconsistent ring dimensions", owner_text(owner)));
    }
    let support = |constant: bool, upper: Option<usize>| {
        if constant { 1 } else { upper.unwrap_or(ring_dimension) }
    };
    for upper in [facts.left_support_upper, facts.right_support_upper].into_iter().flatten() {
        if upper > ring_dimension {
            return Err(format!("product owner {} has invalid support upper", owner_text(owner)));
        }
    }
    let left_scalar = left_rows == 1 && left_columns == 1;
    let right_scalar = right_rows == 1 && right_columns == 1;
    let factor = if left_scalar {
        support(facts.left_is_constant_polynomial, facts.left_support_upper)
    } else if right_scalar {
        support(facts.right_is_constant_polynomial, facts.right_support_upper)
    } else {
        if left_columns != right_rows {
            return Err(format!(
                "product owner {} has incompatible matrix inputs",
                owner_text(owner)
            ));
        }
        let zero_rows = facts.right_known_zero_rows.as_ref().map_or(Ok(0_usize), |value| {
            usize::try_from(value).map_err(|_| "product zero-row count overflow".to_owned())
        })?;
        if zero_rows > right_rows {
            return Err(format!("product owner {} has invalid zero rows", owner_text(owner)));
        }
        (left_columns - zero_rows) *
            if facts.left_is_constant_polynomial || facts.right_is_constant_polynomial {
                1
            } else {
                ring_dimension
            }
    };
    Ok((left_rows, left_columns, right_rows, right_columns, ring_dimension, factor))
}

fn left_bound_source<'a>(
    index: &PayloadIndex,
    data: &'a LeftRootRenderData<'_>,
    consumer: ProofPayloadOwner,
    reference: &ProofPayloadValueRef,
) -> Result<(Option<u64>, &'a LeftRootBoundNode<'a>), String> {
    if let ProofPayloadValueRef::Transfer(event) = reference {
        let node = data
            .bounds
            .iter()
            .find(|node| node.event == *event && node.owner == consumer)
            .ok_or_else(|| {
            format!("Security0 direct transfer input {event} is outside the semantic closure")
        })?;
        return Ok((None, node));
    }
    let projection = match reference {
        ProofPayloadValueRef::Result { projection, .. } |
        ProofPayloadValueRef::Predecessor { projection, .. } => projection,
        ProofPayloadValueRef::Transfer(event) => {
            return Err(format!(
                "Security0 left bound renderer reached unsupported direct transfer reference {event}"
            ));
        }
    };
    let result_event = reached_bound_result_event(index, consumer, reference)?;
    let producer = match (projection, index.event(result_event)?) {
        (
            BoundProjection::Coefficient,
            ProofPayloadEvent::Result { value: ProofPayloadValue::Coefficient { .. }, .. },
        ) => result_event
            .checked_sub(1)
            .ok_or_else(|| format!("Result {result_event} has no coefficient producer"))?,
        (
            BoundProjection::Summary,
            ProofPayloadEvent::Result { value: ProofPayloadValue::Coefficient { .. }, .. },
        ) => {
            return Err(format!(
                "Security0 left bound reference {reference:?} selects a ResultCoefficient summary"
            ));
        }
        (
            BoundProjection::Coefficient,
            ProofPayloadEvent::Result {
                value: ProofPayloadValue::Exact { coefficient_producer, .. },
                ..
            },
        ) => *coefficient_producer,
        (
            BoundProjection::Summary,
            ProofPayloadEvent::Result {
                value: ProofPayloadValue::Exact { summary_producer: Some(producer), .. },
                ..
            },
        ) => *producer,
        (
            BoundProjection::Summary,
            ProofPayloadEvent::Result {
                value: ProofPayloadValue::Exact { summary_producer: None, .. },
                ..
            },
        ) => {
            return Err(format!(
                "Security0 left bound reference {reference:?} selects an unresolved ResultExact summary"
            ));
        }
        (_, _) => unreachable!("reached bound reference identifies a Result"),
    };
    let node = data
        .bounds
        .iter()
        .find(|node| node.event == producer)
        .ok_or_else(|| {
            format!(
                "Security0 left bound input Result {} selects producer {producer} outside the left closure",
                result_event
            )
        })?;
    Ok((Some(result_event), node))
}

fn replay_left_bound_classes(
    statement: &CertificateDocumentV1,
    index: &PayloadIndex,
    data: &LeftRootRenderData<'_>,
) -> Result<
    BTreeMap<
        u64,
        crate::operational_noise::facts::NumericContract<
            crate::operational_noise::facts::CoefficientBound,
        >,
    >,
    String,
> {
    use crate::operational_noise::facts::{
        NumericContract, add_bounds, product_bounds, product_bounds_with_factor,
    };

    let mut replayed =
        BTreeMap::<u64, NumericContract<crate::operational_noise::facts::CoefficientBound>>::new();
    for node in &data.bounds {
        let bound = if matches!(node.rule, ProofPayloadRule::Authority(_)) {
            let result_event = data.authority_results.get(&node.event).ok_or_else(|| {
                format!("Security0 authority bound {} has no mapped Result", node.event)
            })?;
            match index.event(*result_event)? {
                ProofPayloadEvent::Result {
                    value: ProofPayloadValue::Coefficient { bound },
                    ..
                } => bound.clone(),
                ProofPayloadEvent::Result {
                    value: ProofPayloadValue::Exact { coefficient_bound, .. },
                    ..
                } => coefficient_bound.clone(),
                _ => unreachable!("authority mapping identifies a Result"),
            }
        } else {
            let inputs = reached_bound_references(node.rule)
                .into_iter()
                .map(|reference| {
                    let (_, producer) = left_bound_source(index, data, node.owner, reference)?;
                    replayed.get(&producer.event).cloned().ok_or_else(|| {
                        format!(
                            "Security0 bound {} depends on unreplayed producer {}",
                            node.event, producer.event
                        )
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            match node.rule {
                ProofPayloadRule::Identity { .. } => inputs[0].clone(),
                ProofPayloadRule::Sum { .. } => add_bounds(&inputs),
                ProofPayloadRule::Scale {
                    scale: crate::operational_noise::simulation::ProofPayloadScale::Magnitude(value),
                    ..
                } => product_bounds_with_factor(&inputs, value),
                ProofPayloadRule::Scale {
                    scale: crate::operational_noise::simulation::ProofPayloadScale::Value(_),
                    ..
                } |
                ProofPayloadRule::MonomialProduct { .. } => product_bounds(&inputs),
                ProofPayloadRule::Product { facts, .. } => {
                    let (_, _, _, _, _, factor) =
                        reached_product_shape(statement, node.owner, facts)?;
                    product_bounds_with_factor(&inputs, &factor.into())
                }
                _ => unreachable!("filtered reached bound replay rule"),
            }
        };
        if matches!(bound, NumericContract::Missing) {
            return Err(format!(
                "Security0 reached bound {} replays to a missing class",
                node.event
            ));
        }
        replayed.insert(node.event, bound);
    }
    Ok(replayed)
}

fn render_left_bound_input(
    source: &mut String,
    index: &PayloadIndex,
    data: &LeftRootRenderData<'_>,
    node: &LeftRootBoundNode<'_>,
    ordinal: usize,
    reference: &ProofPayloadValueRef,
    modulus: &str,
) -> Result<(String, String), String> {
    let (result_event, producer) = left_bound_source(index, data, node.owner, reference)?;
    let producer_namespace = left_bound_namespace(producer.event, producer.rule);
    let name = format!("input{ordinal}");
    if matches!(reference, ProofPayloadValueRef::Transfer(_)) {
        writeln!(
            source,
            "theorem {name} (selector : Nat)\n    (witness : Witness document history (some selector) {modulus}) :\n    BoundInputAt history owner ({})\n      {producer_namespace}.bound ({producer_namespace}.actual selector witness) := by\n  exact .transfer ({producer_namespace}.derived selector witness)\n",
            value_ref_text(reference)
        )
        .expect("String write");
        return Ok((producer_namespace, name));
    }
    let result_event = result_event.expect("non-transfer input has a Result");
    let (raw_terms, result_owner) = match index.event(result_event)? {
        ProofPayloadEvent::Result { owner, value: ProofPayloadValue::Coefficient { .. } } => {
            ("none".to_owned(), owner_text(*owner))
        }
        ProofPayloadEvent::Result { owner, value: ProofPayloadValue::Exact { terms, .. } } => {
            (format!("some ({})", raw_terms_text(terms)), owner_text(*owner))
        }
        _ => unreachable!("reached bound input identifies a Result"),
    };
    let projection = match reference {
        ProofPayloadValueRef::Result { projection, .. } |
        ProofPayloadValueRef::Predecessor { projection, .. } => projection,
        ProofPayloadValueRef::Transfer(_) => unreachable!(),
    };
    let projector = match (projection, index.event(result_event)?) {
        (
            BoundProjection::Coefficient,
            ProofPayloadEvent::Result { value: ProofPayloadValue::Coefficient { .. }, .. },
        ) => format!(
            ".resultCoefficient (by decide) (by rfl) ({producer_namespace}.derived selector witness)"
        ),
        (
            BoundProjection::Coefficient,
            ProofPayloadEvent::Result { value: ProofPayloadValue::Exact { .. }, .. },
        ) => {
            let refines =
                format!("by dsimp [{producer_namespace}.bound, RecordedBoundRefines] <;> decide");
            format!(
                ".resultExactCoefficient (by rfl)\n      ({refines})\n      ({producer_namespace}.derived selector witness)"
            )
        }
        (
            BoundProjection::Summary,
            ProofPayloadEvent::Result { value: ProofPayloadValue::Exact { .. }, .. },
        ) => {
            format!(".resultExactSummary (by rfl) ({producer_namespace}.derived selector witness)")
        }
        _ => unreachable!("validated left bound projection and Result kind"),
    };
    writeln!(
        source,
        "theorem {name} (selector : Nat)\n    (witness : Witness document history (some selector) {modulus}) :\n    BoundInputAt history owner ({})\n      {producer_namespace}.bound ({producer_namespace}.actual selector witness) := by",
        value_ref_text(reference)
    )
    .expect("String write");
    match reference {
        ProofPayloadValueRef::Result { .. } => {
            let constructor = match projection {
                BoundProjection::Coefficient => ".result",
                BoundProjection::Summary => ".resultSummary",
            };
            writeln!(
                source,
                "  refine {constructor} (resultOwner := {result_owner}) \
                 (rawTerms := {raw_terms}) (by decide) ?_\n  exact {projector}\n"
            )
            .expect("String write");
        }
        ProofPayloadValueRef::Predecessor { .. } => {
            writeln!(
                source,
                "  refine .predecessor (rawTerms := {raw_terms}) (by rfl) ?_\n  exact {projector}\n"
            )
            .expect("String write");
        }
        ProofPayloadValueRef::Transfer(_) => unreachable!(),
    }
    Ok((producer_namespace, name))
}

fn render_left_bounds(
    statement: &CertificateDocumentV1,
    index: &PayloadIndex,
    data: &LeftRootRenderData<'_>,
    modulus: &str,
) -> Result<Vec<super::super::TallSecurity0GeneratedFile>, String> {
    const CHUNK_SIZE: usize = 16;
    let replayed = replay_left_bound_classes(statement, index, data)?;
    let nodes = data
        .bounds
        .iter()
        .filter(|node| !matches!(node.rule, ProofPayloadRule::Authority(_)))
        .collect::<Vec<_>>();
    let mut files = Vec::new();
    for (shard_index, shard) in nodes.chunks(CHUNK_SIZE).enumerate() {
        let module = format!("SemanticLeftBoundShard{shard_index:03}");
        let mut source = shard_index.checked_sub(1).map_or_else(
            || format!("import {NAMESPACE}.Semantic.SemanticLeftAuthority\n"),
            |previous| format!("import {NAMESPACE}.Semantic.SemanticLeftBoundShard{previous:03}\n"),
        );
        source
            .push_str("\nset_option autoImplicit false\nset_option relaxedAutoImplicit false\n\n");
        writeln!(source, "namespace {NAMESPACE}.Semantic\n").expect("String write");
        source.push_str("open Mxx.Certificate.OperationalNoise\nopen TallSecurity0ABI\nopen TallSemantics\nopen EventReplay\n\n");
        for node in shard {
            let frame = index.immediate_frames
                [usize::try_from(node.event).map_err(|_| "left bound event index overflow")?]
            .ok_or_else(|| format!("left bound event {} has no frame", node.event))?;
            writeln!(source, "namespace LeftBound{}", node.event).expect("String write");
            writeln!(source, "def owner : Owner := {}", owner_text(node.owner))
                .expect("String write");
            writeln!(
                source,
                "def transferEvent : Nat := {}\ndef frameStart : Nat := {frame}",
                node.event
            )
            .expect("String write");
            writeln!(source, "def rule : BoundRule := {}", left_rule_text(node.rule)?)
                .expect("String write");

            let references = match node.rule {
                ProofPayloadRule::Identity { input } => vec![input],
                ProofPayloadRule::Sum { inputs } => inputs.iter().collect(),
                ProofPayloadRule::Scale { value, scale } => {
                    let mut values = vec![value];
                    if let crate::operational_noise::simulation::ProofPayloadScale::Value(scale) =
                        scale
                    {
                        values.push(scale);
                    }
                    values
                }
                ProofPayloadRule::MonomialProduct { factors, .. } => {
                    factors.iter().map(|factor| &factor.bound).collect()
                }
                ProofPayloadRule::Product { left, right, .. } => vec![left, right],
                _ => unreachable!("filtered reached compositional rule"),
            };
            let inputs = references
                .iter()
                .enumerate()
                .map(|(ordinal, reference)| {
                    render_left_bound_input(
                        &mut source,
                        index,
                        data,
                        node,
                        ordinal,
                        reference,
                        modulus,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let child_bounds = inputs
                .iter()
                .map(|(namespace, _)| format!("{namespace}.bound"))
                .collect::<Vec<_>>();
            let child_actuals = inputs
                .iter()
                .map(|(namespace, _)| format!("{namespace}.actual selector witness"))
                .collect::<Vec<_>>();
            let (bound, actual, proof) = match node.rule {
                ProofPayloadRule::Identity { .. } => (
                    child_bounds[0].clone(),
                    child_actuals[0].clone(),
                    "refine .identity (by rfl) (input0 selector witness)".to_owned(),
                ),
                ProofPayloadRule::Sum { .. } => {
                    let children = (0..inputs.len()).rev().fold(".nil".to_owned(), |tail, ordinal| {
                        format!(".cons (input{ordinal} selector witness) ({tail})")
                    });
                    (
                        format!("addKnownList [{}]", child_bounds.join(", ")),
                        format!("[{}].sum", child_actuals.join(", ")),
                        format!("refine .sum (by rfl) ({children})"),
                    )
                }
                ProofPayloadRule::Scale {
                    scale: crate::operational_noise::simulation::ProofPayloadScale::Magnitude(value),
                    ..
                } => (
                    format!("scaleMagnitude {value} {}", child_bounds[0]),
                    format!("{value} * ({})", child_actuals[0]),
                    "refine .scaleMagnitude (by rfl) (input0 selector witness)".to_owned(),
                ),
                ProofPayloadRule::Scale {
                    scale: crate::operational_noise::simulation::ProofPayloadScale::Value(_),
                    ..
                } => (
                    format!("scaleValue {} {}", child_bounds[0], child_bounds[1]),
                    format!("({}) * ({})", child_actuals[0], child_actuals[1]),
                    "refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)"
                        .to_owned(),
                ),
                ProofPayloadRule::MonomialProduct { factors, .. } => {
                    if factors.is_empty() {
                        return Err(format!(
                            "Security0 reached monomial-product {} has no factors",
                            node.event
                        ));
                    }
                    let tail_bounds = child_bounds[1..].join(", ");
                    let tail_actuals = child_actuals[1..].join(", ");
                    let tail = (1..inputs.len()).rev().fold(".nil".to_owned(), |tail, ordinal| {
                        format!(".cons (.intro (input{ordinal} selector witness)) ({tail})")
                    });
                    (
                        format!("productNonempty {} [{tail_bounds}]", child_bounds[0]),
                        format!("({}) * ([{tail_actuals}].prod)", child_actuals[0]),
                        format!(
                            "refine .monomialProduct (by rfl) (.intro (input0 selector witness)) ({tail})"
                        ),
                    )
                }
                ProofPayloadRule::Product { facts, .. } => {
                    let (left_rows, left_columns, right_rows, right_columns, ring_dimension, factor) =
                        reached_product_shape(statement, node.owner, facts)?;
                    (
                        format!(
                            "productWithFactor {factor} {} {}",
                            child_bounds[0], child_bounds[1]
                        ),
                        format!(
                            "{factor} * ({}) * ({})",
                            child_actuals[0], child_actuals[1]
                        ),
                        format!(
                            "refine .product (leftRows := {left_rows}) (leftColumns := {left_columns}) \
                             (rightRows := {right_rows}) (rightColumns := {right_columns}) \
                             (ringDimension := {ring_dimension}) (factor := {factor}) (by rfl) \
                             (by decide) (input0 selector witness) (input1 selector witness)"
                        ),
                    )
                }
                _ => unreachable!("filtered reached compositional rule"),
            };
            let replayed_bound = coeff_class_text(
                replayed
                    .get(&node.event)
                    .ok_or_else(|| format!("Security0 bound {} was not replayed", node.event))?,
            )?;
            let child_bound_defs = inputs
                .iter()
                .map(|(namespace, _)| format!("{namespace}.bound"))
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(source, "def rawBound : CoeffClass := {bound}").expect("String write");
            writeln!(source, "def bound : CoeffClass := {replayed_bound}").expect("String write");
            writeln!(source, "theorem rawBound_eq_bound : rawBound = bound := by\n  dsimp [rawBound, bound, {child_bound_defs}, addKnownList, addKnown, productWithFactor,\n    scaleMagnitude, scaleValue, productNonempty] <;> decide")
                .expect("String write");
            writeln!(source, "def actual (selector : Nat)\n    (witness : Witness document history (some selector) {modulus}) : Nat := {actual}")
                .expect("String write");
            writeln!(source, "theorem derived (selector : Nat)\n    (witness : Witness document history (some selector) {modulus}) :\n    BoundDerivedAt history transferEvent frameStart owner rule bound\n      (actual selector witness) := by\n  rw [← rawBound_eq_bound]\n  unfold rule rawBound actual\n  {proof}\nend LeftBound{}\n", node.event)
                .expect("String write");
        }
        writeln!(source, "end {NAMESPACE}.Semantic").expect("String write");
        files.push(generated_file(format!("Semantic/{module}.lean"), source));
    }
    let last = nodes.len().div_ceil(CHUNK_SIZE) - 1;
    files.push(generated_file(
        "Semantic/SemanticLeftBound.lean",
        format!("import {NAMESPACE}.Semantic.SemanticLeftBoundShard{last:03}\n"),
    ));
    Ok(files)
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

fn coeff_class_text(
    bound: &crate::operational_noise::facts::NumericContract<
        crate::operational_noise::facts::CoefficientBound,
    >,
) -> Result<String, String> {
    use crate::operational_noise::facts::{CoefficientBound, NumericContract};
    Ok(match bound {
        NumericContract::Known(CoefficientBound::ExactZero) => ".exactZero".to_owned(),
        NumericContract::Known(CoefficientBound::Finite(value)) => {
            format!(".finite ⟨{}, by decide⟩", value.maximum_absolute_coefficient)
        }
        NumericContract::Known(CoefficientBound::Large) => ".large".to_owned(),
        NumericContract::Missing => {
            return Err("reached authority Result has a missing coefficient bound".to_owned());
        }
    })
}

fn recorded_bound_text(
    bound: &crate::operational_noise::facts::NumericContract<
        crate::operational_noise::facts::CoefficientBound,
    >,
) -> String {
    use crate::operational_noise::facts::{CoefficientBound, NumericContract};
    match bound {
        NumericContract::Known(CoefficientBound::ExactZero) => ".exactZero".to_owned(),
        NumericContract::Known(CoefficientBound::Finite(value)) => {
            format!(".finite {}", value.maximum_absolute_coefficient)
        }
        NumericContract::Known(CoefficientBound::Large) => ".large".to_owned(),
        NumericContract::Missing => ".missing".to_owned(),
    }
}

fn authority_text(authority: &ProofPayloadAuthority) -> Result<String, String> {
    Ok(match authority {
        ProofPayloadAuthority::FactStore => ".factStore".to_owned(),
        ProofPayloadAuthority::ProgramFamilyFact => ".programFamilyFact".to_owned(),
        ProofPayloadAuthority::Operator => ".operator".to_owned(),
        ProofPayloadAuthority::RelationPreimageSource { source } => {
            format!(".relationPreimageSource ⟨{source}⟩")
        }
        ProofPayloadAuthority::Unavailable => {
            return Err("reached authority renderer received unavailable authority".to_owned());
        }
    })
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

fn product_facts_text(facts: &crate::operational_noise::bound::MatrixProductFacts) -> String {
    format!(
        "⟨{}, {}, {}, {}, {}⟩",
        if facts.left_is_constant_polynomial { "true" } else { "false" },
        if facts.right_is_constant_polynomial { "true" } else { "false" },
        facts
            .right_known_zero_rows
            .as_ref()
            .map_or_else(|| "none".to_owned(), |value| format!("some {value}")),
        facts.left_support_upper.map_or_else(|| "none".to_owned(), |value| format!("some {value}")),
        facts
            .right_support_upper
            .map_or_else(|| "none".to_owned(), |value| format!("some {value}")),
    )
}

fn stable_value_type_text(
    value: &crate::operational_noise::g0::StableValueType,
) -> Result<String, String> {
    use crate::operational_noise::g0::StableValueType;
    Ok(match value {
        StableValueType::Bool => ".bool".to_owned(),
        StableValueType::Int => ".int".to_owned(),
        StableValueType::Real => ".real".to_owned(),
        StableValueType::Bytes => ".bytes".to_owned(),
        StableValueType::Trapdoor => ".trapdoor".to_owned(),
        StableValueType::Matrix { modulus, ring_dimension, rows, columns } => {
            format!(".matrix {} {ring_dimension} {rows} {columns}", super::quoted(modulus)?)
        }
    })
}

fn owner_value_type_text(
    statement: &CertificateDocumentV1,
    owner: ProofPayloadOwner,
) -> Result<String, String> {
    let row = statement
        .expressions
        .get(usize::try_from(owner.expression_row).map_err(|_| "owner row overflow")?)
        .ok_or_else(|| format!("owner {} has no expression row", owner_text(owner)))?;
    match &row.descriptor {
        CanonicalExpressionDescriptor::Operation { value_type, .. } => {
            stable_value_type_text(value_type)
        }
        _ => Err(format!("operator owner {} is not operation-typed", owner_text(owner))),
    }
}

fn rule_text(value: &ProofPayloadRule) -> String {
    match value {
        ProofPayloadRule::Sum { inputs } => {
            format!(".sum [{}]", inputs.iter().map(value_ref_text).collect::<Vec<_>>().join(", "))
        }
        ProofPayloadRule::Product { left, right, facts } => format!(
            ".product ({}) ({}) {}",
            value_ref_text(left),
            value_ref_text(right),
            product_facts_text(facts),
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
    right_root_event: u64,
) -> Result<Vec<RightRootNode>, String> {
    let CertificateResidualRootV1::Family { program, .. } = statement.residual_root else {
        return Err("Security0 right-root semantics requires a family residual root".to_owned());
    };
    let mut pending = vec![right_root_event];
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
                let operation = op_probe(index, result, operation_kind)?;
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

fn right_root_predecessor_evidence_clause(
    owner: ProofPayloadOwner,
    value: &ProofPayloadValueRef,
    expected_source: u64,
    index: &PayloadIndex,
) -> Result<Option<String>, String> {
    match value {
        ProofPayloadValueRef::Result { event, .. } if *event == expected_source => Ok(None),
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
            Ok(Some(format!(
                "(history.lookup {binding_event}).map AnnotatedEvent.event =\n      some (.predecessor {} {input_position} ⟨{predecessor}⟩ {source_result})",
                owner_text(owner),
            )))
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

fn right_root_operation_evidence(
    node: &RightRootNode,
    operation: &OperationProbe,
    index: &PayloadIndex,
) -> Result<(String, usize), String> {
    let refs = right_root_operation_refs(operation)?;
    let expression_row = node.result.owner.expression_row;
    let expression_module = expression_row / 256;
    let mut clauses = vec![format!(
        "document.expressions.lookup {expression_row} =\n      some {NAMESPACE}.Cert.Expression{expression_module:03}.ExpressionRow{expression_row}"
    )];
    if let Some(clause) = right_root_predecessor_evidence_clause(
        node.result.owner,
        refs[0],
        operation.input_events[0],
        index,
    )? {
        clauses.push(clause);
    }
    if let Some(clause) = right_root_predecessor_evidence_clause(
        node.result.owner,
        refs[1],
        operation.input_events[1],
        index,
    )? {
        clauses.push(clause);
    }
    clauses.push(format!(
        "(history.lookup {}).map AnnotatedEvent.event =\n      some (.boundTransfer {} ({}))",
        operation.rule_event,
        owner_text(node.result.owner),
        rule_text(operation.rule.as_ref().expect("typed right-root rule")),
    ));
    let proof_count = clauses.len();
    Ok((format!("def operationEvidence : Prop :=\n  {}", clauses.join(" ∧\n    ")), proof_count))
}

fn right_root_operation_actual_source(
    event: u64,
    operation: &OperationProbe,
    modulus: &str,
) -> Result<String, String> {
    let operator = match operation.kind {
        OperationKind::Add => "+",
        OperationKind::Subtract => "-",
        OperationKind::Multiply | OperationKind::Tensor => "*",
        OperationKind::Direct => {
            return Err(format!("right residual Result {event} reaches Direct"));
        }
    };
    Ok(format!(
        "def actual (selector : Nat)\n    (witness : Witness document history (some selector) {modulus}) : Int := by\n  letI : Decidable operationEvidence := isTrue operationEvidenceSound\n  exact if operationEvidence then\n    ResidualRightResult{}.actual selector witness {operator}\n      ResidualRightResult{}.actual selector witness\n    else 0",
        operation.input_events[0], operation.input_events[1],
    ))
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
            let ProofPayloadEvent::Result {
                value:
                    ProofPayloadValue::Exact {
                        coefficient_bound,
                        coefficient_producer,
                        summary_producer,
                        ..
                    },
                ..
            } = index.event(event)?
            else {
                unreachable!("right-root nodes contain only exact Results");
            };
            if *coefficient_producer != *producer_event || summary_producer.is_some() {
                return Err(format!(
                    "terminal Result {event} does not use its adjacent producer exclusively"
                ));
            }
            writeln!(source, "def owner : Owner := {}", owner_text(node.result.owner))
                .expect("String write");
            writeln!(source, "def rawTerms : List Term := [{}]", raw_term_text(term))
                .expect("String write");
            source.push_str("def summary : Bound := .exactZero\n");
            writeln!(source, "def producerEvent : Nat := {producer_event}").expect("String write");
            writeln!(source, "def resultEvent : Nat := {event}").expect("String write");
            writeln!(
                source,
                "abbrev actual (selector : Nat)\n    (witness : Witness document history (some selector) {modulus}) : Int :=\n  Cert.ResidualRightResult{event}.actual selector witness"
            )
            .expect("String write");
            writeln!(
                source,
                "theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)\n    (selectorUpper : selector < selectorMaximum) :\n    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by\n  refine ⟨by decide, rightRootOwnerAtSelector selector selectorLower selectorUpper _, ?_⟩\n  refine ⟨{}, {frame_start}, {}, {}, ?_, ?_⟩\n  · rfl\n  · rfl",
                reached_terminal_rule_text(rule)?,
                recorded_bound_text(coefficient_bound),
                reached_terminal_constructor(rule)?,
            )
            .expect("String write");
            writeln!(
                source,
                "theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)\n    (selectorUpper : selector < selectorMaximum)\n    (witness : Witness document history (some selector) {modulus}) :\n    ExactClaimAt history {modulus} witness.env resultEvent owner\n      (actual selector witness) rawTerms summary := by\n  exact terminalExactClaimAt witness\n    (terminalAt selector selectorLower selectorUpper)"
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
            if operation.kind == OperationKind::Direct {
                return Err(format!("right-root Result {event} is a direct operation"));
            }
            writeln!(
                source,
                "abbrev actual (selector : Nat)\n    (witness : Witness document history (some selector) {modulus}) : Int :=\n  Cert.ResidualRightResult{event}.actual selector witness"
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
            let (_, operation_history_arity) =
                right_root_operation_evidence(node, operation, index)?;
            writeln!(
                source,
                "theorem claimOfHistory (selector : Nat)\n    (witness : Witness document history (some selector) {modulus})\n    (operationAt : Cert.ResidualRightResult{event}.operationEvidence)"
            )
            .expect("String write");
            let operation_history_proofs =
                (0..operation_history_arity).map(|_| "by rfl").collect::<Vec<_>>().join(", ");
            writeln!(
                source,
                "\n    (leftClaim : ExactClaimAt history {modulus} witness.env\n      SemanticRightRootResult{left_event}.resultEvent\n      SemanticRightRootResult{left_event}.owner\n      (SemanticRightRootResult{left_event}.actual selector witness)\n      SemanticRightRootResult{left_event}.rawTerms\n      SemanticRightRootResult{left_event}.summary)\n    (rightClaim : ExactClaimAt history {modulus} witness.env\n      SemanticRightRootResult{right_event}.resultEvent\n      SemanticRightRootResult{right_event}.owner\n      (SemanticRightRootResult{right_event}.actual selector witness)\n      SemanticRightRootResult{right_event}.rawTerms\n      SemanticRightRootResult{right_event}.summary)\n    {{frameStart coefficientProducer : Nat}} {{coefficientBound : Bound}}\n    {{summaryProducer : Option Nat}}\n    (outputAt : history.lookup resultEvent = some\n      ⟨.resultExact owner rawTerms coefficientBound coefficientProducer summary summaryProducer,\n        frameStart⟩) :\n    ExactClaimAt history {modulus} witness.env resultEvent owner\n      (actual selector witness) rawTerms summary := by\n  refine ⟨⟨coefficientBound, coefficientProducer, summaryProducer, ?_⟩, ?_⟩\n  · rw [outputAt]\n    rfl\n  · simpa only [actual, Cert.ResidualRightResult{event}.actual, if_pos operationAt,\n      SemanticRightRootResult{left_event}.actual,\n      SemanticRightRootResult{right_event}.actual, rawTerms, outputRaw, output, summary] using\n      {theorem} {modulus} witness.env\n        (SemanticRightRootResult{left_event}.actual selector witness)\n        (SemanticRightRootResult{right_event}.actual selector witness) left right output\n        (by simpa [left, leftRaw, SemanticRightRootResult{left_event}.summary] using leftClaim.claim)\n        (by simpa [right, rightRaw, SemanticRightRootResult{right_event}.summary] using rightClaim.claim)\n        (resultSound witness.env) (by decide)",
            )
            .expect("String write");
            writeln!(
                source,
                "\ntheorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)\n    (selectorUpper : selector < selectorMaximum)\n    (witness : Witness document history (some selector) {modulus}) :\n    ExactClaimAt history {modulus} witness.env resultEvent owner\n      (actual selector witness) rawTerms summary := by\n  apply claimOfHistory selector witness ⟨{operation_history_proofs}⟩\n  · exact SemanticRightRootResult{left_event}.claimSound\n      selector selectorLower selectorUpper witness\n  · exact SemanticRightRootResult{right_event}.claimSound\n      selector selectorLower selectorUpper witness\n  · rfl",
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
    right_root_event: u64,
) -> Result<(String, Vec<super::super::TallSecurity0GeneratedFile>), String> {
    const CHUNK_SIZE: usize = 16;
    let nodes = right_root_nodes(statement, index, relation_probes, right_root_event)?;
    let mut files = Vec::new();
    for (shard_index, shard) in nodes.chunks(CHUNK_SIZE).enumerate() {
        let residual_module = format!("ResidualRightShard{shard_index:03}");
        let previous_residual = shard_index
            .checked_sub(1)
            .map(|index| format!("import {NAMESPACE}.Cert.ResidualRightShard{index:03}\n"));
        let mut residual = previous_residual.unwrap_or_else(|| {
            format!(
                "import Mxx.Certificate.OperationalNoise.TallSemantics\nimport {NAMESPACE}.Proof.History\n"
            )
        });
        residual
            .push_str("\nset_option autoImplicit false\nset_option relaxedAutoImplicit false\n\n");
        writeln!(residual, "namespace {NAMESPACE}.Cert\n").expect("String write");
        residual.push_str(
            "open Mxx.Certificate.OperationalNoise\nopen TallSecurity0ABI\nopen TallSemantics\n\n",
        );
        for node in shard {
            let event = node.result.event;
            writeln!(residual, "namespace ResidualRightResult{event}").expect("String write");
            match &node.kind {
                RightRootNodeKind::Terminal { .. } => writeln!(residual, "def actual (selector : Nat)\n    (witness : Witness document history (some selector) {modulus}) : Int :=\n  witness.honestTerminalActual {event}"),
                RightRootNodeKind::Operation(operation) => {
                    let (operation_evidence, operation_history_arity) =
                        right_root_operation_evidence(node, operation, index)?;
                    writeln!(residual, "{operation_evidence}").expect("String write");
                    let operation_history_proofs = (0..operation_history_arity)
                        .map(|_| "by rfl")
                        .collect::<Vec<_>>()
                        .join(", ");
                    writeln!(
                        residual,
                        "theorem operationEvidenceSound : operationEvidence :=\n  ⟨{operation_history_proofs}⟩"
                    )
                    .expect("String write");
                    writeln!(
                        residual,
                        "{}",
                        right_root_operation_actual_source(event, operation, modulus)?
                    )
                }
            }
            .expect("String write");
            writeln!(residual, "end ResidualRightResult{event}\n").expect("String write");
        }
        writeln!(residual, "end {NAMESPACE}.Cert").expect("String write");
        files.push(generated_file(format!("Cert/{residual_module}.lean"), residual));

        let module = format!("SemanticRightRootShard{shard_index:03}");
        let previous = shard_index
            .checked_sub(1)
            .map(|index| format!("import {NAMESPACE}.Semantic.SemanticRightRootShard{index:03}\n"));
        let mut source = previous.unwrap_or_else(|| {
            format!(
                "import Mxx.Certificate.OperationalNoise.TallSemantics\nimport {NAMESPACE}.Proof.History\n"
            )
        });
        writeln!(source, "import {NAMESPACE}.Cert.{residual_module}").expect("String write");
        source
            .push_str("\nset_option autoImplicit false\nset_option relaxedAutoImplicit false\n\n");
        writeln!(source, "namespace {NAMESPACE}.Semantic\n").expect("String write");
        source.push_str(
            "open Mxx.Certificate.OperationalNoise\nopen TallSecurity0ABI\nopen TallSemantics\n\n",
        );
        if shard_index == 0 {
            let CertificateResidualRootV1::Family { program, .. } = statement.residual_root else {
                unreachable!("right-root collector validated family residual root")
            };
            writeln!(source, "theorem rightRootOwnerAtSelector (selector : Nat) (selectorLower : selectorMinimum ≤ selector)\n    (selectorUpper : selector < selectorMaximum) (expression : SchemaV1.ExpressionRef) :\n    ownerAtSelector document (some selector) ⟨.program ⟨{program}⟩, expression⟩ := by\n  simp only [selectorMinimum] at selectorLower\n  simp only [selectorMaximum] at selectorUpper\n  simp [ownerAtSelector, document, selectorLower, selectorUpper]\n")
                .expect("String write");
        }
        for node in shard {
            render_right_root_node(&mut source, node, index, modulus)?;
        }
        writeln!(source, "end {NAMESPACE}.Semantic").expect("String write");
        files.push(generated_file(format!("Semantic/{module}.lean"), source));
    }
    let last_shard = nodes.len().div_ceil(CHUNK_SIZE) - 1;
    files.push(generated_file(
        "Cert/ResidualRight.lean",
        format!("import {NAMESPACE}.Cert.ResidualRightShard{last_shard:03}\n"),
    ));
    let mut source = format!(
        "import {NAMESPACE}.Semantic.SemanticRightRootShard{last_shard:03}\n\nset_option autoImplicit false\nset_option relaxedAutoImplicit false\n\nnamespace {NAMESPACE}.Semantic.SemanticRightRoot\n\nopen Mxx.Certificate.OperationalNoise\nopen TallSecurity0ABI\nopen TallSemantics\n\n"
    );
    writeln!(
        source,
        "/-- The generated theorem application for the reached right exact-zero root. -/\ntheorem rightRootClaimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)\n    (selectorUpper : selector < selectorMaximum)\n    (witness : Witness document history (some selector) {modulus}) :\n    ExactClaimAt history {modulus} witness.env\n      SemanticRightRootResult{right_root_event}.resultEvent SemanticRightRootResult{right_root_event}.owner\n      (SemanticRightRootResult{right_root_event}.actual selector witness)\n      SemanticRightRootResult{right_root_event}.rawTerms SemanticRightRootResult{right_root_event}.summary := by\n  exact SemanticRightRootResult{right_root_event}.claimSound selector selectorLower selectorUpper witness"
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

    fn test_exact(terms: Vec<ProofPayloadTerm>, summary: BoundedSummary) -> ProofPayloadValue {
        let coefficient_bound = summary.coefficient_bound();
        let summary_producer = (!matches!(
            coefficient_bound,
            crate::operational_noise::facts::NumericContract::Known(
                crate::operational_noise::facts::CoefficientBound::ExactZero
            )
        ))
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
                    value: test_exact(vec![term(source.clone(), 1)], BoundedSummary::zero()),
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
                    value: test_exact(
                        vec![
                            term(carried.clone(), 1),
                            term(replacement.clone(), 1),
                            term(later.clone(), 1),
                        ],
                        BoundedSummary::zero(),
                    ),
                },
                ProofPayloadEvent::InvocationEnd {
                    root,
                    result: test_exact(
                        vec![
                            term(
                                ProofPayloadMonomial {
                                    central_factors: vec![],
                                    ordered_factors: vec![root, root],
                                },
                                1,
                            ),
                            term(carried.clone(), 1),
                        ],
                        BoundedSummary::zero(),
                    ),
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
    fn typed_add_rule_uses_predecessor_result_refs() {
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
                    value: test_exact(vec![term(left.clone())], BoundedSummary::zero()),
                },
                ProofPayloadEvent::Predecessor {
                    consumer: root,
                    input_position: 0,
                    predecessor: 0,
                    source_result: 1,
                },
                ProofPayloadEvent::Result {
                    owner: root,
                    value: test_exact(vec![term(right.clone())], BoundedSummary::zero()),
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
                    value: test_exact(vec![term(left), term(right)], BoundedSummary::zero()),
                },
                ProofPayloadEvent::InvocationEnd {
                    root,
                    result: test_exact(vec![], BoundedSummary::zero()),
                    pre_fold_event: 0,
                },
            ],
        };
        let index = PayloadIndex::new(&proof).expect("payload index");
        let result = index.result(6).expect("exact Add result");
        let (_, refs, _, _) = typed_operation_rule(&index, &result, OperationKind::Add)
            .expect("typed Add rule")
            .expect("eligible Add rule");
        assert_eq!(index.value_ref(root, &refs[0]).expect("left ref").event, 1);
        assert_eq!(index.value_ref(root, &refs[1]).expect("right ref").event, 3);
        assert!(index.merges.is_empty());

        let transfer_error = match index.value_ref(root, &ProofPayloadValueRef::Transfer(5)) {
            Ok(_) => panic!("bound-only transfer is not an exact polynomial"),
            Err(error) => error,
        };
        assert!(transfer_error.contains("bound-only"));

        let ProofPayloadEvent::BoundTransfer { rule, .. } =
            index.event(5).expect("typed Add transfer")
        else {
            panic!("event 5 must be the typed Add transfer");
        };
        let operation = OperationProbe {
            kind: OperationKind::Add,
            rule_event: 5,
            input_events: [1, 3],
            output_event: 6,
            inputs: vec![
                index.result(1).expect("left result"),
                index.result(3).expect("right result"),
            ],
            output: result.clone(),
            scalar_left: false,
            scalar_right: false,
            rule: Some(rule.clone()),
            composite_relations: Vec::new(),
        };
        let node = RightRootNode { result, kind: RightRootNodeKind::Operation(operation.clone()) };
        let (evidence, evidence_arity) = right_root_operation_evidence(&node, &operation, &index)
            .expect("right-root operation evidence");
        assert_eq!(evidence_arity, 4);
        assert!(evidence.contains("document.expressions.lookup 7"));
        assert!(evidence.contains("history.lookup 2"));
        assert!(evidence.contains("history.lookup 4"));
        assert!(evidence.contains("history.lookup 5"));
        let actual = right_root_operation_actual_source(6, &operation, "257")
            .expect("guarded right-root actual");
        assert!(actual.contains("if operationEvidence then"));
        assert!(actual.contains("ResidualRightResult1.actual selector witness +"));

        let mut rendered = String::new();
        render_right_root_node(&mut rendered, &node, &index, "257")
            .expect("right-root operation claim");
        assert!(rendered.contains("operationAt : Cert.ResidualRightResult6.operationEvidence"));
        assert!(rendered.contains("if_pos operationAt"));
        assert!(!rendered.contains("rcases operationAt"));
    }

    #[test]
    fn bound_chain_uses_typed_end_pre_fold_and_same_frame_result() {
        let root = owner(9);
        let proof = OperationalProofPayload {
            events: vec![
                ProofPayloadEvent::InvocationStart { root },
                ProofPayloadEvent::Result {
                    owner: root,
                    value: test_exact(vec![], BoundedSummary::zero()),
                },
                ProofPayloadEvent::PreFoldPolynomial(ProofPayloadPreFoldPolynomial {
                    result_event: 1,
                    terms: vec![],
                    summary: BoundedSummary::zero(),
                    summary_evidence: None,
                }),
                ProofPayloadEvent::InvocationEnd {
                    root,
                    result: test_exact(vec![], BoundedSummary::zero()),
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
}
