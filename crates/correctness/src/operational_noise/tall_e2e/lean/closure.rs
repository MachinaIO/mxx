use super::super::super::simulation::{
    OperationalProofPayload, ProofPayloadAuthority, ProofPayloadCoefficientMergeSource,
    ProofPayloadEvent, ProofPayloadOwner, ProofPayloadRelationRule, ProofPayloadRule,
    ProofPayloadScale, ProofPayloadValue, ProofPayloadValueRef,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

const SECURITY0_PROGRAM: u64 = 214;
const SECURITY0_ROOT_EXPRESSION: u64 = 30_220;

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
            .map_err(|error| format!("Security0 dependency closure encoding failed: {error}"))
    }
}

impl From<ProofPayloadOwner> for OwnerReport {
    fn from(owner: ProofPayloadOwner) -> Self {
        match owner.scope {
            super::super::super::simulation::ProofPayloadScope::Closed { root_expression_row } => {
                Self {
                    scope: "closed",
                    scope_row: root_expression_row,
                    expression_row: owner.expression_row,
                }
            }
            super::super::super::simulation::ProofPayloadScope::Program { program_row } => Self {
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

fn event_kind(event: &ProofPayloadEvent) -> ClosureEventKind {
    match event {
        ProofPayloadEvent::InvocationStart { .. } => ClosureEventKind::InvocationStart,
        ProofPayloadEvent::Predecessor { .. } => ClosureEventKind::Predecessor,
        ProofPayloadEvent::Result { value, .. } => match value {
            super::super::super::simulation::ProofPayloadValue::Exact { .. } => {
                ClosureEventKind::ResultExact
            }
            super::super::super::simulation::ProofPayloadValue::Coefficient { .. } => {
                ClosureEventKind::ResultCoefficient
            }
        },
        ProofPayloadEvent::InvocationEnd { result, .. } => match result {
            super::super::super::simulation::ProofPayloadValue::Exact { .. } => {
                ClosureEventKind::InvocationEndExact
            }
            super::super::super::simulation::ProofPayloadValue::Coefficient { .. } => {
                ClosureEventKind::InvocationEndCoefficient
            }
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
            if !is_value_event(index.event(*rhs_result)?) {
                return Err(format!("universal RHS {rhs_result} is not a value event"));
            }
            work.extend([*computed, *rhs_result]);
        }
        ProofPayloadRelationRule::Gadget { input_result, .. } => {
            require_prior(*input_result, current, "gadget input result")?;
            if !is_value_event(index.event(*input_result)?) {
                return Err(format!("gadget input {input_result} is not a value event"));
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
            if !matches!(result, super::super::super::simulation::ProofPayloadValue::Exact { .. }) {
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
                    if !is_value_event(index.event(input.value_event)?) {
                        return Err(format!(
                            "operator merge input {} is not a value event",
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

fn collect(proof: &OperationalProofPayload) -> Result<DependencyClosure, String> {
    let index = Index::new(&proof.events)?;
    let (final_end_event, final_root) = proof
        .events
        .iter()
        .enumerate()
        .rev()
        .find_map(|(position, event)| {
            if let ProofPayloadEvent::InvocationEnd { root, .. } = event {
                Some((event_id(position).ok()?, *root))
            } else {
                None
            }
        })
        .ok_or_else(|| "proof payload has no InvocationEnd".to_owned())?;
    let last_event = event_index(event_id(proof.events.len().saturating_sub(1))?)?;
    let final_event_index = event_index(final_end_event)?;
    if last_event != final_event_index {
        return Err("final InvocationEnd is not the last proof event".to_owned());
    }
    if index.frame(final_end_event)?.parent.is_some() {
        return Err("final InvocationEnd is nested instead of closing the outer frame".to_owned());
    }
    let mut pending = vec![final_end_event];
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
        collect_event(&index, event, &mut deps)?;
        pending.extend(deps);
    }
    let event_ids = reached.into_iter().collect::<Vec<_>>();
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

pub(crate) fn collect_security0_final_closure(
    proof: &OperationalProofPayload,
) -> Result<DependencyClosure, String> {
    let closure = collect(proof)?;
    let expected = ProofPayloadOwner {
        scope: super::super::super::simulation::ProofPayloadScope::Program {
            program_row: SECURITY0_PROGRAM,
        },
        expression_row: SECURITY0_ROOT_EXPRESSION,
    };
    if closure.final_root != expected {
        return Err(format!(
            "Security0 final root {:?} does not match fixed root {:?}",
            closure.final_root, expected
        ));
    }
    Ok(closure)
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
        ProofPayloadValue::Exact {
            terms: vec![ProofPayloadTerm {
                monomial: ProofPayloadMonomial { central_factors: vec![], ordered_factors: vec![] },
                coefficient: BigInt::from(1),
            }],
            summary: BoundedSummary::zero(),
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
        let closure = collect(&proof).expect("collect toy final closure");
        assert_eq!(closure.final_end_event, 6);
        assert_eq!(closure.event_ids, vec![0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(closure.event_counts[&ClosureEventKind::CoefficientMergeOperator], 1);
        assert_eq!(closure.bound_rule_counts[&ClosureBoundRuleKind::AuthorityOperator], 1);
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
        assert!(collect(&proof).is_err());
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
        let closure = collect(&proof).expect("collect coefficient predecessor closure");
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
