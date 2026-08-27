use crate::operational_noise::{
    certificate_schema::{CertificateDocumentV1, CertificateResidualRootV1},
    g0::BoundProjection,
    simulation::{
        OperationalProofPayload, ProofPayloadAuthority, ProofPayloadCoefficientMergeSource,
        ProofPayloadEvent, ProofPayloadOwner, ProofPayloadRelationRule, ProofPayloadRule,
        ProofPayloadScale, ProofPayloadScope, ProofPayloadValue, ProofPayloadValueRef,
    },
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case", tag = "event", content = "detail")]
pub(crate) enum ClosureEventKind {
    InvocationStart,
    Predecessor,
    ResultExact,
    ResultCoefficient,
    InvocationEndExact,
    InvocationEndCoefficient,
    SpecializationComputed,
    SpecializationCacheHit,
    AppliedRelationUniversal,
    AppliedRelationGadget,
    BoundTransfer,
    CoefficientMergeOperator,
    CoefficientMergeRelation,
    PreFoldPolynomial,
    SurvivorFold,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case", tag = "rule", content = "authority")]
pub(crate) enum ClosureBoundRuleKind {
    AuthorityFactStore,
    AuthorityProgramFamilyFact,
    AuthorityOperator,
    AuthorityRelationPreimageSource,
    AuthorityUnavailable,
    Identity,
    Sum,
    Maximum,
    Scale,
    MonomialProduct,
    WeightedSum,
    Product,
    Tensor,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DependencyClosure {
    pub final_end_event: u64,
    pub final_root: ProofPayloadOwner,
    pub event_ids: Vec<u64>,
    pub event_counts: BTreeMap<ClosureEventKind, u64>,
    pub bound_rule_counts: BTreeMap<ClosureBoundRuleKind, u64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ClosureReport<'a> {
    final_end_event: u64,
    final_root: OwnerReport,
    event_ids: &'a [u64],
    event_counts: Vec<EventCountReport>,
    bound_rule_counts: Vec<BoundRuleCountReport>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EventCountReport {
    event: ClosureEventKind,
    count: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BoundRuleCountReport {
    rule: ClosureBoundRuleKind,
    count: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OwnerReport {
    scope: &'static str,
    scope_row: u64,
    expression_row: u64,
}

impl DependencyClosure {
    pub(crate) fn report_bytes(&self) -> Result<Vec<u8>, String> {
        let event_counts = self
            .event_counts
            .iter()
            .map(|(event, count)| EventCountReport { event: *event, count: *count })
            .collect();
        let bound_rule_counts = self
            .bound_rule_counts
            .iter()
            .map(|(rule, count)| BoundRuleCountReport { rule: *rule, count: *count })
            .collect();
        let report = ClosureReport {
            final_end_event: self.final_end_event,
            final_root: OwnerReport::from(self.final_root),
            event_ids: &self.event_ids,
            event_counts,
            bound_rule_counts,
        };
        serde_json::to_vec(&report)
            .map_err(|error| format!("certificate dependency closure encoding failed: {error}"))
    }
}

impl From<ProofPayloadOwner> for OwnerReport {
    fn from(owner: ProofPayloadOwner) -> Self {
        match owner.scope {
            ProofPayloadScope::Closed { root_expression_row } => Self {
                scope: "closed",
                scope_row: root_expression_row,
                expression_row: owner.expression_row,
            },
            ProofPayloadScope::Program { program_row } => Self {
                scope: "program",
                scope_row: program_row,
                expression_row: owner.expression_row,
            },
        }
    }
}

#[derive(Clone, Copy)]
struct FrameInfo {
    start: u64,
    parent: Option<u64>,
}

struct Index<'a> {
    events: &'a [ProofPayloadEvent],
    frames: Vec<Option<FrameInfo>>,
    predecessors: BTreeMap<u64, (ProofPayloadOwner, u32, u64)>,
}

impl<'a> Index<'a> {
    fn new(events: &'a [ProofPayloadEvent]) -> Result<Self, String> {
        let mut stack = Vec::<(u64, ProofPayloadOwner, Option<u64>)>::new();
        let mut frames = vec![None; events.len()];
        let mut predecessors = BTreeMap::new();
        for (position, event) in events.iter().enumerate() {
            let id = event_id(position)?;
            if let ProofPayloadEvent::InvocationStart { root } = event {
                let parent = stack.last().map(|(start, _, _)| *start);
                stack.push((id, *root, parent));
            }
            let Some((start, _, parent)) = stack.last().copied() else {
                return Err(format!("event {id} is outside an invocation frame"));
            };
            frames[position] = Some(FrameInfo { start, parent });
            match event {
                ProofPayloadEvent::Predecessor {
                    consumer,
                    input_position,
                    predecessor,
                    source_result,
                } => {
                    require_prior(*source_result, id, "predecessor source result")?;
                    let source = events.get(event_index(*source_result)?).ok_or_else(|| {
                        format!("predecessor {id} source result {source_result} is missing")
                    })?;
                    if !is_value_event(source) {
                        return Err(format!(
                            "predecessor {id} source result {source_result} is not a value event"
                        ));
                    }
                    if predecessors
                        .insert(id, (*consumer, *input_position, *source_result))
                        .is_some()
                    {
                        return Err(format!("duplicate predecessor event id {id}"));
                    }
                    let _ = predecessor;
                }
                ProofPayloadEvent::InvocationEnd { root, pre_fold_event, .. } => {
                    require_prior(*pre_fold_event, id, "invocation-end pre-fold")?;
                    let Some(prefold_frame) =
                        frames.get(event_index(*pre_fold_event)?).and_then(|frame| *frame)
                    else {
                        return Err(format!(
                            "invocation end {id} references missing pre-fold {pre_fold_event}"
                        ));
                    };
                    if prefold_frame.start != start {
                        return Err(format!(
                            "invocation end {id} pre-fold {pre_fold_event} is outside its frame"
                        ));
                    }
                    let (started, expected, _) = stack
                        .pop()
                        .ok_or_else(|| format!("invocation end {id} has no active frame"))?;
                    if started != start || expected != *root {
                        return Err(format!("invocation end {id} does not close its active frame"));
                    }
                }
                _ => {}
            }
        }
        if !stack.is_empty() {
            return Err("proof payload has an unclosed invocation frame".to_owned());
        }
        Ok(Self { events, frames, predecessors })
    }

    fn event(&self, id: u64) -> Result<&ProofPayloadEvent, String> {
        self.events
            .get(event_index(id)?)
            .ok_or_else(|| format!("event reference {id} is out of range"))
    }

    fn frame(&self, id: u64) -> Result<FrameInfo, String> {
        self.frames
            .get(event_index(id)?)
            .and_then(|frame| *frame)
            .ok_or_else(|| format!("event {id} has no frame"))
    }

    fn require_same_frame(&self, from: u64, to: u64) -> Result<(), String> {
        if self.frame(from)?.start != self.frame(to)?.start {
            return Err(format!("event reference {to} leaves frame of event {from}"));
        }
        Ok(())
    }
}

fn event_id(position: usize) -> Result<u64, String> {
    u64::try_from(position).map_err(|_| "proof event index overflows u64".to_owned())
}

fn event_index(id: u64) -> Result<usize, String> {
    usize::try_from(id).map_err(|_| format!("event reference {id} overflows usize"))
}

fn require_prior(reference: u64, current: u64, label: &str) -> Result<(), String> {
    if reference >= current {
        return Err(format!("{label} {reference} is not prior to event {current}"));
    }
    Ok(())
}

fn is_value_event(event: &ProofPayloadEvent) -> bool {
    matches!(
        event,
        ProofPayloadEvent::Result { .. } |
            ProofPayloadEvent::InvocationEnd { result: ProofPayloadValue::Exact { .. }, .. }
    )
}

fn is_exact_value_event(event: &ProofPayloadEvent) -> bool {
    matches!(
        event,
        ProofPayloadEvent::Result { value: ProofPayloadValue::Exact { .. }, .. } |
            ProofPayloadEvent::InvocationEnd { result: ProofPayloadValue::Exact { .. }, .. }
    )
}

fn event_kind(event: &ProofPayloadEvent) -> ClosureEventKind {
    match event {
        ProofPayloadEvent::InvocationStart { .. } => ClosureEventKind::InvocationStart,
        ProofPayloadEvent::Predecessor { .. } => ClosureEventKind::Predecessor,
        ProofPayloadEvent::Result { value, .. } => match value {
            ProofPayloadValue::Exact { .. } => ClosureEventKind::ResultExact,
            ProofPayloadValue::Coefficient { .. } => ClosureEventKind::ResultCoefficient,
        },
        ProofPayloadEvent::InvocationEnd { result, .. } => match result {
            ProofPayloadValue::Exact { .. } => ClosureEventKind::InvocationEndExact,
            ProofPayloadValue::Coefficient { .. } => ClosureEventKind::InvocationEndCoefficient,
        },
        ProofPayloadEvent::SpecializationComputed { .. } => {
            ClosureEventKind::SpecializationComputed
        }
        ProofPayloadEvent::SpecializationCacheHit { .. } => {
            ClosureEventKind::SpecializationCacheHit
        }
        ProofPayloadEvent::AppliedRelation { rule, .. } => match rule {
            ProofPayloadRelationRule::Universal { .. } => {
                ClosureEventKind::AppliedRelationUniversal
            }
            ProofPayloadRelationRule::Gadget { .. } => ClosureEventKind::AppliedRelationGadget,
        },
        ProofPayloadEvent::BoundTransfer { .. } => ClosureEventKind::BoundTransfer,
        ProofPayloadEvent::CoefficientMerge(merge) => match merge.source {
            ProofPayloadCoefficientMergeSource::Operator { .. } => {
                ClosureEventKind::CoefficientMergeOperator
            }
            ProofPayloadCoefficientMergeSource::Relation { .. } => {
                ClosureEventKind::CoefficientMergeRelation
            }
        },
        ProofPayloadEvent::PreFoldPolynomial(_) => ClosureEventKind::PreFoldPolynomial,
        ProofPayloadEvent::SurvivorFold(_) => ClosureEventKind::SurvivorFold,
    }
}

fn bound_rule_kind(rule: &ProofPayloadRule) -> ClosureBoundRuleKind {
    match rule {
        ProofPayloadRule::Authority(authority) => match authority {
            ProofPayloadAuthority::FactStore => ClosureBoundRuleKind::AuthorityFactStore,
            ProofPayloadAuthority::ProgramFamilyFact => {
                ClosureBoundRuleKind::AuthorityProgramFamilyFact
            }
            ProofPayloadAuthority::Operator => ClosureBoundRuleKind::AuthorityOperator,
            ProofPayloadAuthority::RelationPreimageSource { .. } => {
                ClosureBoundRuleKind::AuthorityRelationPreimageSource
            }
            ProofPayloadAuthority::Unavailable => ClosureBoundRuleKind::AuthorityUnavailable,
        },
        ProofPayloadRule::Identity { .. } => ClosureBoundRuleKind::Identity,
        ProofPayloadRule::Sum { .. } => ClosureBoundRuleKind::Sum,
        ProofPayloadRule::Maximum { .. } => ClosureBoundRuleKind::Maximum,
        ProofPayloadRule::Scale { .. } => ClosureBoundRuleKind::Scale,
        ProofPayloadRule::MonomialProduct { .. } => ClosureBoundRuleKind::MonomialProduct,
        ProofPayloadRule::WeightedSum { .. } => ClosureBoundRuleKind::WeightedSum,
        ProofPayloadRule::Product { .. } => ClosureBoundRuleKind::Product,
        ProofPayloadRule::Tensor { .. } => ClosureBoundRuleKind::Tensor,
    }
}

fn collect_value_ref(
    index: &Index<'_>,
    current: u64,
    owner: ProofPayloadOwner,
    value: &ProofPayloadValueRef,
    work: &mut Vec<u64>,
) -> Result<(), String> {
    match value {
        ProofPayloadValueRef::Result { event, .. } => {
            require_prior(*event, current, "Result reference")?;
            if !is_value_event(index.event(*event)?) {
                return Err(format!("Result reference {event} is not a value event"));
            }
            work.push(*event);
        }
        ProofPayloadValueRef::Predecessor { binding_event, input_position, .. } => {
            require_prior(*binding_event, current, "predecessor reference")?;
            index.require_same_frame(current, *binding_event)?;
            let Some((consumer, position, source_result)) = index.predecessors.get(binding_event)
            else {
                return Err(format!(
                    "predecessor reference {binding_event} is not a Predecessor event"
                ));
            };
            if *consumer != owner || *position != *input_position {
                return Err(format!("predecessor reference {binding_event} owner/input mismatch"));
            }
            work.push(*binding_event);
            collect_value_ref(
                index,
                current,
                owner,
                &ProofPayloadValueRef::Result {
                    event: *source_result,
                    projection: crate::operational_noise::g0::BoundProjection::Coefficient,
                },
                work,
            )?;
        }
        ProofPayloadValueRef::Transfer(event) => {
            require_prior(*event, current, "transfer reference")?;
            index.require_same_frame(current, *event)?;
            let ProofPayloadEvent::BoundTransfer { owner: transfer_owner, rule } =
                index.event(*event)?
            else {
                return Err(format!("transfer reference {event} is not a BoundTransfer event"));
            };
            if *transfer_owner != owner {
                return Err(format!("transfer reference {event} owner mismatch"));
            }
            work.push(*event);
            collect_rule(index, *event, owner, rule, work)?;
        }
    }
    Ok(())
}

fn collect_scale(
    index: &Index<'_>,
    current: u64,
    owner: ProofPayloadOwner,
    scale: &ProofPayloadScale,
    work: &mut Vec<u64>,
) -> Result<(), String> {
    if let ProofPayloadScale::Value(value) = scale {
        collect_value_ref(index, current, owner, value, work)?;
    }
    Ok(())
}

fn collect_rule(
    index: &Index<'_>,
    current: u64,
    owner: ProofPayloadOwner,
    rule: &ProofPayloadRule,
    work: &mut Vec<u64>,
) -> Result<(), String> {
    match rule {
        ProofPayloadRule::Authority(_) => {}
        ProofPayloadRule::Identity { input } => {
            collect_value_ref(index, current, owner, input, work)?
        }
        ProofPayloadRule::Sum { inputs } |
        ProofPayloadRule::Maximum { inputs } |
        ProofPayloadRule::WeightedSum { inputs } => {
            for input in inputs {
                collect_value_ref(index, current, owner, input, work)?;
            }
        }
        ProofPayloadRule::Scale { value, scale } => {
            collect_value_ref(index, current, owner, value, work)?;
            collect_scale(index, current, owner, scale, work)?;
        }
        ProofPayloadRule::MonomialProduct { factors, .. } => {
            for factor in factors {
                collect_value_ref(index, current, owner, &factor.bound, work)?;
            }
        }
        ProofPayloadRule::Product { left, right, .. } |
        ProofPayloadRule::Tensor { left, right, .. } => {
            collect_value_ref(index, current, owner, left, work)?;
            collect_value_ref(index, current, owner, right, work)?;
        }
    }
    Ok(())
}

fn collect_relation(
    index: &Index<'_>,
    current: u64,
    rule: &ProofPayloadRelationRule,
    work: &mut Vec<u64>,
) -> Result<(), String> {
    match rule {
        ProofPayloadRelationRule::Universal { computed, rhs_result, .. } => {
            require_prior(*computed, current, "universal specialization")?;
            require_prior(*rhs_result, current, "universal RHS result")?;
            index.require_same_frame(current, *computed)?;
            if !matches!(index.event(*computed)?, ProofPayloadEvent::SpecializationComputed { .. })
            {
                return Err(format!("universal specialization {computed} is not computed"));
            }
            if !is_exact_value_event(index.event(*rhs_result)?) {
                return Err(format!("universal RHS {rhs_result} is not an exact value event"));
            }
            work.extend([*computed, *rhs_result]);
        }
        ProofPayloadRelationRule::Gadget { input_result, .. } => {
            require_prior(*input_result, current, "gadget input result")?;
            if !is_exact_value_event(index.event(*input_result)?) {
                return Err(format!("gadget input {input_result} is not an exact value event"));
            }
            work.push(*input_result);
        }
    }
    Ok(())
}

fn producer_events(
    index: &Index<'_>,
    result_event: u64,
    owner: ProofPayloadOwner,
) -> Result<Vec<u64>, String> {
    if matches!(
        index.event(result_event)?,
        ProofPayloadEvent::Result { value: ProofPayloadValue::Coefficient { .. }, .. }
    ) {
        let producer = result_event.checked_sub(1).ok_or_else(|| {
            format!("coefficient Result {result_event} has no preceding producer")
        })?;
        index.require_same_frame(result_event, producer)?;
        match index.event(producer)? {
            ProofPayloadEvent::BoundTransfer { owner: transfer_owner, .. }
                if *transfer_owner == owner =>
            {
                return Ok(vec![producer])
            }
            _ => {
                return Err(format!(
                    "coefficient Result {result_event} must immediately follow a same-owner BoundTransfer"
                ));
            }
        }
    }
    let frame = index.frame(result_event)?.start;
    let previous_result = index.events[..event_index(result_event)?]
        .iter()
        .enumerate()
        .rev()
        .filter_map(|(position, event)| {
            let id = u64::try_from(position).ok()?;
            (index.frames[position].is_some_and(|item| item.start == frame) &&
                matches!(event, ProofPayloadEvent::Result { owner: candidate, .. } if *candidate == owner))
                .then_some(id)
        })
        .next()
        .map_or(frame, |event| event.saturating_add(1));
    let mut merges = Vec::new();
    for (position, event) in
        index.events[event_index(previous_result)?..event_index(result_event)?].iter().enumerate()
    {
        let id = previous_result +
            u64::try_from(position).map_err(|_| "producer event index overflow")?;
        if index.frames[event_index(id)?].is_some_and(|item| item.start == frame) {
            if let ProofPayloadEvent::CoefficientMerge(merge) = event {
                if merge.owner == owner {
                    merges.push(id);
                }
            }
        }
    }
    if !merges.is_empty() {
        return Ok(merges);
    }
    let transfers = index.events[event_index(frame)? .. event_index(result_event)?]
        .iter()
        .enumerate()
        .filter_map(|(offset, event)| {
            let id = frame + u64::try_from(offset).ok()?;
            matches!(event, ProofPayloadEvent::BoundTransfer { owner: candidate, .. } if *candidate == owner)
                .then_some(id)
        })
        .collect::<Vec<_>>();
    let Some(transfer) = transfers.last().copied() else {
        return Err(format!(
            "Result {result_event} owner has no producer transfer or coefficient merge"
        ));
    };
    Ok(vec![transfer])
}

fn collect_event(index: &Index<'_>, event_id: u64, work: &mut Vec<u64>) -> Result<(), String> {
    match index.event(event_id)? {
        ProofPayloadEvent::InvocationStart { .. } |
        ProofPayloadEvent::SpecializationComputed { .. } |
        ProofPayloadEvent::SpecializationCacheHit { .. } => {}
        ProofPayloadEvent::Predecessor { source_result, .. } => {
            require_prior(*source_result, event_id, "predecessor source result")?;
            if !is_value_event(index.event(*source_result)?) {
                return Err(format!("predecessor source {source_result} is not a value event"));
            }
            work.push(*source_result);
        }
        ProofPayloadEvent::Result { owner, .. } => {
            work.extend(producer_events(index, event_id, *owner)?);
        }
        ProofPayloadEvent::InvocationEnd { root, pre_fold_event, result } => {
            if !matches!(result, ProofPayloadValue::Exact { .. }) {
                return Err(format!("final closure event {event_id} has a non-exact result"));
            }
            require_prior(*pre_fold_event, event_id, "invocation-end pre-fold")?;
            index.require_same_frame(event_id, *pre_fold_event)?;
            if !matches!(index.event(*pre_fold_event)?, ProofPayloadEvent::PreFoldPolynomial(_)) {
                return Err(format!(
                    "invocation-end pre-fold {pre_fold_event} is not a PreFoldPolynomial"
                ));
            }
            let _ = root;
            work.push(*pre_fold_event);
        }
        ProofPayloadEvent::AppliedRelation { rule, .. } => {
            collect_relation(index, event_id, rule, work)?
        }
        ProofPayloadEvent::BoundTransfer { owner, rule } => {
            collect_rule(index, event_id, *owner, rule, work)?;
        }
        ProofPayloadEvent::CoefficientMerge(merge) => match &merge.source {
            ProofPayloadCoefficientMergeSource::Operator { inputs } => {
                for input in inputs {
                    require_prior(input.value_event, event_id, "operator merge input")?;
                    if !is_exact_value_event(index.event(input.value_event)?) {
                        return Err(format!(
                            "operator merge input {} is not an exact value event",
                            input.value_event
                        ));
                    }
                    work.push(input.value_event);
                }
            }
            ProofPayloadCoefficientMergeSource::Relation { application, .. } => {
                require_prior(*application, event_id, "relation merge application")?;
                if !matches!(index.event(*application)?, ProofPayloadEvent::AppliedRelation { .. })
                {
                    return Err(format!(
                        "relation merge application {application} is not AppliedRelation"
                    ));
                }
                work.push(*application);
            }
        },
        ProofPayloadEvent::PreFoldPolynomial(value) => {
            require_prior(value.result_event, event_id, "pre-fold result")?;
            if !matches!(index.event(value.result_event)?, ProofPayloadEvent::Result { .. }) {
                return Err(format!("pre-fold result {} is not a Result", value.result_event));
            }
            work.push(value.result_event);
            if let Some(evidence) = &value.summary_evidence {
                collect_value_ref(
                    index,
                    event_id,
                    owner_for_value(index, value.result_event)?,
                    evidence,
                    work,
                )?;
            }
        }
        ProofPayloadEvent::SurvivorFold(value) => {
            require_prior(value.bound, event_id, "survivor bound")?;
            if !matches!(
                index.event(value.bound)?,
                ProofPayloadEvent::BoundTransfer { .. } | ProofPayloadEvent::Result { .. }
            ) {
                return Err(format!("survivor bound {} is not a bound event", value.bound));
            }
            work.push(value.bound);
        }
    }
    Ok(())
}

fn owner_for_value(index: &Index<'_>, event: u64) -> Result<ProofPayloadOwner, String> {
    match index.event(event)? {
        ProofPayloadEvent::Result { owner, .. } => Ok(*owner),
        _ => Err(format!("event {event} does not carry a value owner")),
    }
}

fn collect_event_ids(index: &Index<'_>, start_event: u64) -> Result<Vec<u64>, String> {
    index.event(start_event)?;
    let mut pending = vec![start_event];
    let mut reached = BTreeSet::new();
    while let Some(event) = pending.pop() {
        if !reached.insert(event) {
            continue;
        }
        let mut frame = Some(index.frame(event)?);
        while let Some(info) = frame {
            pending.push(info.start);
            frame = info.parent.and_then(|parent| index.frame(parent).ok());
        }
        let mut deps = Vec::new();
        collect_event(index, event, &mut deps)?;
        pending.extend(deps);
    }
    Ok(reached.into_iter().collect())
}

pub(crate) fn collect_event_closure(
    proof: &OperationalProofPayload,
    start_event: u64,
) -> Result<Vec<u64>, String> {
    let index = Index::new(&proof.events)?;
    collect_event_ids(&index, start_event)
}

fn collect(
    proof: &OperationalProofPayload,
    final_end_event: u64,
    final_root: ProofPayloadOwner,
) -> Result<DependencyClosure, String> {
    let index = Index::new(&proof.events)?;
    let final_event = index.event(final_end_event)?;
    if !matches!(final_event, ProofPayloadEvent::InvocationEnd { root, .. } if *root == final_root)
    {
        return Err(format!("event {final_end_event} is not the selected residual InvocationEnd"));
    }
    if index.frame(final_end_event)?.parent.is_some() {
        return Err("final InvocationEnd is nested instead of closing the outer frame".to_owned());
    }
    let event_ids = collect_event_ids(&index, final_end_event)?;
    let mut event_counts = BTreeMap::new();
    let mut bound_rule_counts = BTreeMap::new();
    for event in &event_ids {
        let kind = event_kind(index.event(*event)?);
        *event_counts.entry(kind).or_insert(0) += 1;
        if let ProofPayloadEvent::BoundTransfer { rule, .. } = index.event(*event)? {
            *bound_rule_counts.entry(bound_rule_kind(rule)).or_insert(0) += 1;
        }
    }
    Ok(DependencyClosure {
        final_end_event,
        final_root,
        event_ids,
        event_counts,
        bound_rule_counts,
    })
}

fn exact_event(proof: &OperationalProofPayload, event: u64) -> Result<&ProofPayloadEvent, String> {
    proof.events.get(event_index(event)?).ok_or_else(|| format!("event {event} is missing"))
}

pub(crate) fn collect_residual_closure(
    statement: &CertificateDocumentV1,
    proof: &OperationalProofPayload,
) -> Result<DependencyClosure, String> {
    // The residual-root statement identifies the owner.  For a family, use
    // the program row's root expression and verify that it is exactly the
    // expression named by the owner; for a closed root the expression is
    // already present directly in the root row.  No protocol-specific
    // operation kind is assumed here.
    let root_owner = match statement.residual_root {
        CertificateResidualRootV1::Closed { expression } => ProofPayloadOwner {
            scope: ProofPayloadScope::Closed { root_expression_row: expression },
            expression_row: expression,
        },
        CertificateResidualRootV1::Family { program, domain } => {
            let program_row = statement
                .programs
                .get(usize::try_from(program).map_err(|_| "residual program index overflow")?)
                .ok_or_else(|| format!("residual program {program} is missing"))?;
            let family = program_row
                .family
                .as_ref()
                .ok_or_else(|| format!("residual program {program} is not a family"))?;
            if family.domain != domain {
                return Err(format!(
                    "residual program {program} family domain disagrees with residual root"
                ));
            }
            if family.element_type != program_row.output {
                return Err(format!(
                    "residual program {program} family element type disagrees with program output"
                ));
            }
            statement
                .expressions
                .get(
                    usize::try_from(program_row.root)
                        .map_err(|_| "root expression index overflow")?,
                )
                .ok_or_else(|| {
                    format!("residual root expression {} is missing", program_row.root)
                })?;
            ProofPayloadOwner {
                scope: ProofPayloadScope::Program { program_row: program },
                expression_row: program_row.root,
            }
        }
    };
    let index = Index::new(&proof.events)?;
    let ends = proof
        .events
        .iter()
        .enumerate()
        .filter_map(|(position, event)| match event {
            ProofPayloadEvent::InvocationEnd { root, .. }
                if *root == root_owner &&
                    index.frames[position].is_some_and(|frame| frame.parent.is_none()) =>
            {
                event_id(position).ok()
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let [end] = ends.as_slice() else {
        return Err(format!(
            "residual root has {} outer InvocationEnd rows instead of one",
            ends.len()
        ));
    };
    let ProofPayloadEvent::InvocationEnd { result: end_value, pre_fold_event: prefold, .. } =
        exact_event(proof, *end)?
    else {
        unreachable!("selected invocation end")
    };
    let ProofPayloadEvent::PreFoldPolynomial(prefold_value) = exact_event(proof, *prefold)? else {
        return Err(format!("InvocationEnd {end} references non-PreFold event {prefold}"));
    };
    let result = prefold_value.result_event;
    let ProofPayloadEvent::Result { owner, value: result_value } = exact_event(proof, result)?
    else {
        return Err(format!("PreFold {prefold} references non-Result event {result}"));
    };
    if *owner != root_owner {
        return Err(format!("final Result {result} owner does not match residual root"));
    }
    let ProofPayloadValue::Exact {
        terms, coefficient_producer: _, summary, summary_producer, ..
    } = result_value
    else {
        return Err(format!("final Result {result} is not exact"));
    };
    if !terms.is_empty() || summary_producer.is_some() {
        return Err(format!("final Result {result} has unexpected terms or summary producer"));
    }
    if prefold_value.terms != *terms ||
        prefold_value.summary != *summary ||
        prefold_value.summary_evidence !=
            Some(ProofPayloadValueRef::Result {
                event: result,
                projection: BoundProjection::Summary,
            })
    {
        return Err(format!("PreFold {prefold} does not exactly bind final Result {result}"));
    }
    if end_value != result_value {
        return Err(format!("InvocationEnd {end} metadata differs from final Result {result}"));
    }
    collect(proof, *end, root_owner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::{
        facts::NumericContract,
        normal_form::BoundedSummary,
        simulation::{ProofPayloadMonomial, ProofPayloadTerm, ProofPayloadValue},
    };
    use num_bigint::BigInt;

    fn owner(row: u64) -> ProofPayloadOwner {
        ProofPayloadOwner {
            scope: crate::operational_noise::simulation::ProofPayloadScope::Program {
                program_row: 0,
            },
            expression_row: row,
        }
    }

    fn exact() -> ProofPayloadValue {
        let summary = BoundedSummary::zero();
        ProofPayloadValue::Exact {
            terms: vec![ProofPayloadTerm {
                monomial: ProofPayloadMonomial { central_factors: vec![], ordered_factors: vec![] },
                coefficient: BigInt::from(1),
            }],
            coefficient_bound: summary.coefficient_bound(),
            coefficient_producer: 0,
            summary,
            summary_producer: None,
        }
    }

    fn coefficient() -> ProofPayloadValue {
        ProofPayloadValue::Coefficient {
            bound: NumericContract::Known(
                crate::operational_noise::facts::CoefficientBound::finite(1_u8),
            ),
        }
    }

    #[test]
    fn final_closure_follows_result_merge_and_prefold_chain() {
        let root = owner(3);
        let input = owner(4);
        let proof = OperationalProofPayload {
            events: vec![
                ProofPayloadEvent::InvocationStart { root },
                ProofPayloadEvent::BoundTransfer {
                    owner: input,
                    rule: ProofPayloadRule::Authority(ProofPayloadAuthority::Operator),
                },
                ProofPayloadEvent::Result { owner: input, value: exact() },
                ProofPayloadEvent::CoefficientMerge(
                    crate::operational_noise::simulation::ProofPayloadCoefficientMerge {
                        owner: root,
                        source: ProofPayloadCoefficientMergeSource::Operator {
                            inputs: [
                                crate::operational_noise::simulation::ProofPayloadTermRef {
                                    value_event: 2,
                                    term_ordinal: 0,
                                },
                                crate::operational_noise::simulation::ProofPayloadTermRef {
                                    value_event: 2,
                                    term_ordinal: 0,
                                },
                            ],
                        },
                        output: ProofPayloadMonomial {
                            central_factors: vec![],
                            ordered_factors: vec![],
                        },
                        signed_contribution: BigInt::from(1),
                    },
                ),
                ProofPayloadEvent::Result { owner: root, value: exact() },
                ProofPayloadEvent::PreFoldPolynomial(
                    crate::operational_noise::simulation::ProofPayloadPreFoldPolynomial {
                        result_event: 4,
                        terms: vec![],
                        summary: BoundedSummary::zero(),
                        summary_evidence: Some(ProofPayloadValueRef::Result {
                            event: 4,
                            projection: crate::operational_noise::g0::BoundProjection::Summary,
                        }),
                    },
                ),
                ProofPayloadEvent::InvocationEnd { root, result: exact(), pre_fold_event: 5 },
            ],
        };
        let closure = collect(&proof, 6, root).expect("collect certificate final closure");
        assert_eq!(closure.final_end_event, 6);
        assert_eq!(closure.event_ids, vec![0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(closure.event_counts[&ClosureEventKind::CoefficientMergeOperator], 1);
        assert_eq!(closure.bound_rule_counts[&ClosureBoundRuleKind::AuthorityOperator], 1);
        assert_eq!(
            collect_event_closure(&proof, 4).expect("collect exact Result closure"),
            vec![0, 1, 2, 3, 4]
        );
    }

    #[test]
    fn missing_result_producer_fails_closed() {
        let root = owner(3);
        let proof = OperationalProofPayload {
            events: vec![
                ProofPayloadEvent::InvocationStart { root },
                ProofPayloadEvent::Result { owner: root, value: exact() },
                ProofPayloadEvent::PreFoldPolynomial(
                    crate::operational_noise::simulation::ProofPayloadPreFoldPolynomial {
                        result_event: 1,
                        terms: vec![],
                        summary: BoundedSummary::zero(),
                        summary_evidence: None,
                    },
                ),
                ProofPayloadEvent::InvocationEnd { root, result: exact(), pre_fold_event: 2 },
            ],
        };
        assert!(collect(&proof, 3, root).is_err());
    }

    #[test]
    fn predecessor_can_source_coefficient_result() {
        let root = owner(3);
        let input = owner(4);
        let proof = OperationalProofPayload {
            events: vec![
                ProofPayloadEvent::InvocationStart { root },
                ProofPayloadEvent::BoundTransfer {
                    owner: input,
                    rule: ProofPayloadRule::Authority(ProofPayloadAuthority::Operator),
                },
                ProofPayloadEvent::Result { owner: input, value: coefficient() },
                ProofPayloadEvent::Predecessor {
                    consumer: root,
                    input_position: 0,
                    predecessor: 0,
                    source_result: 2,
                },
                ProofPayloadEvent::BoundTransfer {
                    owner: root,
                    rule: ProofPayloadRule::Authority(ProofPayloadAuthority::Operator),
                },
                ProofPayloadEvent::Result { owner: root, value: exact() },
                ProofPayloadEvent::PreFoldPolynomial(
                    crate::operational_noise::simulation::ProofPayloadPreFoldPolynomial {
                        result_event: 5,
                        terms: vec![],
                        summary: BoundedSummary::zero(),
                        summary_evidence: Some(ProofPayloadValueRef::Predecessor {
                            binding_event: 3,
                            input_position: 0,
                            projection: crate::operational_noise::g0::BoundProjection::Summary,
                        }),
                    },
                ),
                ProofPayloadEvent::InvocationEnd { root, result: exact(), pre_fold_event: 6 },
            ],
        };
        let closure = collect(&proof, 7, root).expect("collect coefficient predecessor closure");
        assert!(closure.event_ids.contains(&2));
        assert_eq!(closure.event_counts[&ClosureEventKind::ResultCoefficient], 1);
    }

    #[test]
    fn report_serializes_sorted_count_rows() {
        let root = owner(3);
        let closure = DependencyClosure {
            final_end_event: 2,
            final_root: root,
            event_ids: vec![0, 1, 2],
            event_counts: BTreeMap::from([(ClosureEventKind::InvocationEndExact, 1)]),
            bound_rule_counts: BTreeMap::from([(ClosureBoundRuleKind::Identity, 1)]),
        };
        let report = String::from_utf8(closure.report_bytes().expect("report JSON")).unwrap();
        assert!(report.contains("\"eventCounts\":[{"));
        assert!(report.contains("\"boundRuleCounts\":[{"));
    }
}
