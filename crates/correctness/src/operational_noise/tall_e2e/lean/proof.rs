use super::{MODULE_ROOT, NAMESPACE, generated_file, list, option, quoted};
use crate::operational_noise::{
    arena::MatrixLayout,
    bound::MatrixProductFacts,
    certificate_schema::CertificateDocumentV1,
    facts::{CoefficientBound, NumericContract},
    g0::BoundProjection,
    simulation::{
        OperationalProofPayload, ProofPayloadAuthority, ProofPayloadCoefficientMerge,
        ProofPayloadCoefficientMergeSource, ProofPayloadEvent, ProofPayloadFactorEvidence,
        ProofPayloadMonomial, ProofPayloadOwner, ProofPayloadPreFoldPolynomial,
        ProofPayloadRelationRule, ProofPayloadRule, ProofPayloadScale, ProofPayloadScope,
        ProofPayloadTerm, ProofPayloadValue, ProofPayloadValueRef,
    },
};
use std::fmt::Write as _;

const EVENT_LEAF_SIZE: usize = 16;
pub(super) const EVENT_PACKAGE_SIZE: usize = 256;
const REPLAY_CHUNK_SIZE: usize = 4;

#[derive(Clone)]
struct FrameState {
    root: ProofPayloadOwner,
    start: usize,
}

#[derive(Clone)]
struct ExactTermsEvent {
    kind: ExactTermsEventKind,
    terms: String,
}

#[derive(Clone)]
enum ExactTermsEventKind {
    Result(ResultReplayEvent),
    InvocationEnd,
}

#[derive(Clone)]
struct ResultReplayEvent {
    owner: String,
    coefficient_bound: String,
    coefficient_producer: u64,
    summary: String,
    summary_producer: String,
}

pub(super) fn render(
    statement: &CertificateDocumentV1,
    proof: &OperationalProofPayload,
) -> Result<Vec<super::TallSecurity0GeneratedFile>, String> {
    if proof.events.is_empty() {
        return Err("Security0 proof payload is empty".to_owned());
    }
    let mut rendered = Vec::with_capacity(proof.events.len());
    let mut exact_terms_events = Vec::with_capacity(proof.events.len());
    let mut frame_starts = Vec::with_capacity(proof.events.len());
    let mut states = Vec::with_capacity(proof.events.len().div_ceil(REPLAY_CHUNK_SIZE) + 1);
    let mut event_states = Vec::with_capacity(proof.events.len() + 1);
    let mut stack = Vec::<FrameState>::new();
    let initial_state = render_state(0, &stack)?;
    states.push(initial_state.clone());
    event_states.push(initial_state);
    for (index, event) in proof.events.iter().enumerate() {
        let frame_start = if matches!(event, ProofPayloadEvent::InvocationStart { .. }) {
            index
        } else {
            stack.last().map_or(0, |frame| frame.start)
        };
        frame_starts.push(frame_start);
        let exact_terms = match event {
            ProofPayloadEvent::Result {
                owner: result_owner,
                value:
                    ProofPayloadValue::Exact {
                        terms,
                        coefficient_bound: coefficient,
                        coefficient_producer,
                        summary,
                        summary_producer,
                    },
            } => Some(ExactTermsEvent {
                kind: ExactTermsEventKind::Result(ResultReplayEvent {
                    owner: owner(result_owner),
                    coefficient_bound: coefficient_bound(coefficient)?,
                    coefficient_producer: *coefficient_producer,
                    summary: summary_text(summary)?,
                    summary_producer: option(summary_producer.as_ref(), |event| {
                        Ok(event.to_string())
                    })?,
                }),
                terms: terms_text(terms)?,
            }),
            ProofPayloadEvent::InvocationEnd {
                result: ProofPayloadValue::Exact { terms, .. },
                ..
            } => Some(ExactTermsEvent {
                kind: ExactTermsEventKind::InvocationEnd,
                terms: terms_text(terms)?,
            }),
            _ => None,
        };
        let exact_terms_reference = exact_terms.as_ref().map(|_| format!("exact{index}RawTerms"));
        rendered.push(event_text_with_terms(event, exact_terms_reference.as_deref())?);
        exact_terms_events.push(exact_terms);
        update_state(statement, index, event, &mut stack)?;
        event_states.push(render_state(index + 1, &stack)?);
        if (index + 1) % REPLAY_CHUNK_SIZE == 0 || index + 1 == proof.events.len() {
            states.push(event_states[index + 1].clone());
        }
    }
    if !stack.is_empty() {
        return Err("Security0 proof payload ends with an active invocation frame".to_owned());
    }

    let mut files = Vec::new();
    let package_count = proof.events.len().div_ceil(EVENT_PACKAGE_SIZE);
    for package in 0..package_count {
        let start = package * EVENT_PACKAGE_SIZE;
        let end = (start + EVENT_PACKAGE_SIZE).min(proof.events.len());
        files.push(generated_file(
            format!("Proof/Events{package:03}.lean"),
            render_event_package(
                package,
                start,
                end,
                &rendered,
                &exact_terms_events,
                &frame_starts,
            ),
        ));
    }
    files.push(generated_file(
        "Proof/History.lean",
        render_history(proof.events.len(), package_count),
    ));
    for package in 0..package_count {
        let start_event = package * EVENT_PACKAGE_SIZE;
        let end_event = (start_event + EVENT_PACKAGE_SIZE).min(proof.events.len());
        let start_chunk = start_event / REPLAY_CHUNK_SIZE;
        let end_chunk = end_event.div_ceil(REPLAY_CHUNK_SIZE);
        files.push(generated_file(
            format!("Proof/Replay{package:03}.lean"),
            render_replay_package(
                package,
                start_chunk,
                end_chunk,
                &states,
                &event_states,
                &exact_terms_events,
                &frame_starts,
                proof.events.len(),
            ),
        ));
    }
    files.push(generated_file(
        "Proof/Proof.lean",
        render_top(proof.events.len(), package_count, states.len() - 1),
    ));
    Ok(files)
}

fn event_text(event: &ProofPayloadEvent) -> Result<String, String> {
    event_text_with_terms(event, None)
}

fn event_text_with_terms(
    event: &ProofPayloadEvent,
    exact_terms_reference: Option<&str>,
) -> Result<String, String> {
    Ok(match event {
        ProofPayloadEvent::InvocationStart { root } => {
            format!(".invocationStart ({})", owner(root))
        }
        ProofPayloadEvent::Predecessor { consumer, input_position, predecessor, source_result } => {
            format!(
                ".predecessor ({}) {input_position} ⟨{predecessor}⟩ {source_result}",
                owner(consumer)
            )
        }
        ProofPayloadEvent::Result { owner: result_owner, value } => match value {
            ProofPayloadValue::Exact {
                terms,
                coefficient_bound: coefficient,
                coefficient_producer,
                summary,
                summary_producer,
            } => format!(
                ".resultExact ({}) {} {} {coefficient_producer} {} ({})",
                owner(result_owner),
                exact_terms_reference
                    .map_or_else(|| terms_text(terms), |value| Ok(value.to_owned()))?,
                coefficient_bound(coefficient)?,
                summary_text(summary)?,
                option(summary_producer.as_ref(), |event| Ok(event.to_string()))?,
            ),
            ProofPayloadValue::Coefficient { bound } => format!(
                ".resultCoefficient ({}) {}",
                owner(result_owner),
                coefficient_bound(bound)?
            ),
        },
        ProofPayloadEvent::InvocationEnd { root, result, pre_fold_event } => match result {
            ProofPayloadValue::Exact {
                terms,
                coefficient_bound: coefficient,
                coefficient_producer,
                summary,
                summary_producer,
            } => format!(
                ".invocationEndExact ({}) {pre_fold_event} {} {} {coefficient_producer} {} ({})",
                owner(root),
                exact_terms_reference
                    .map_or_else(|| terms_text(terms), |value| Ok(value.to_owned()))?,
                coefficient_bound(coefficient)?,
                summary_text(summary)?,
                option(summary_producer.as_ref(), |event| Ok(event.to_string()))?,
            ),
            ProofPayloadValue::Coefficient { .. } => {
                return Err("unsupported coefficient InvocationEnd in Security0 renderer".to_owned());
            }
        },
        ProofPayloadEvent::SpecializationComputed { owner: event_owner, dispatch, source } => {
            format!(
                ".specializationComputed ({}) ⟨⟨{}⟩, ⟨{}⟩, ⟨{}⟩⟩ ⟨{}, {}⟩",
                owner(event_owner),
                dispatch.preimage_family,
                dispatch.preimage_source,
                dispatch.trapdoor_source,
                source.start,
                source.end
            )
        }
        ProofPayloadEvent::SpecializationCacheHit { .. } => {
            return Err("unsupported specialization cache hit in Security0 renderer".to_owned());
        }
        ProofPayloadEvent::AppliedRelation {
            owner: event_owner,
            source_monomial,
            outer_coefficient,
            ordered_start,
            ordered_end_exclusive,
            rule,
        } => format!(
            ".appliedRelation ({}) ({}) ({outer_coefficient}) {ordered_start} {ordered_end_exclusive} ({})",
            owner(event_owner),
            monomial(source_monomial)?,
            relation_rule(rule)?
        ),
        ProofPayloadEvent::BoundTransfer { owner: event_owner, rule } => {
            format!(".boundTransfer ({}) ({})", owner(event_owner), rule_text(rule)?)
        }
        ProofPayloadEvent::CoefficientMerge(value) => {
            format!(".coefficientMerge ({})", merge(value)?)
        }
        ProofPayloadEvent::PreFoldPolynomial(value) => pre_fold(value)?,
        ProofPayloadEvent::SurvivorFold(value) => {
            format!(".survivorFold ({}) {}", value.coefficient, value.bound)
        }
    })
}

fn owner(value: &ProofPayloadOwner) -> String {
    let scope = match value.scope {
        ProofPayloadScope::Closed { root_expression_row } => {
            format!(".closed ⟨{root_expression_row}⟩")
        }
        ProofPayloadScope::Program { program_row } => format!(".program ⟨{program_row}⟩"),
    };
    format!("⟨{scope}, ⟨{}⟩⟩", value.expression_row)
}

fn monomial(value: &ProofPayloadMonomial) -> Result<String, String> {
    Ok(format!(
        "⟨{}, {}⟩",
        list(&value.central_factors, |item| Ok(owner(item)))?,
        list(&value.ordered_factors, |item| Ok(owner(item)))?,
    ))
}

fn term(value: &ProofPayloadTerm) -> Result<String, String> {
    Ok(format!("⟨{}, ({})⟩", monomial(&value.monomial)?, value.coefficient))
}

fn terms_text(values: &[ProofPayloadTerm]) -> Result<String, String> {
    list(values, term)
}

fn coefficient_bound(value: &NumericContract<CoefficientBound>) -> Result<String, String> {
    Ok(match value {
        NumericContract::Missing => ".missing".to_owned(),
        NumericContract::Known(CoefficientBound::ExactZero) => ".exactZero".to_owned(),
        NumericContract::Known(CoefficientBound::Finite(value)) => {
            format!("(.finite {})", value.maximum_absolute_coefficient)
        }
        NumericContract::Known(CoefficientBound::Large) => ".large".to_owned(),
    })
}

fn summary_text(
    value: &crate::operational_noise::normal_form::BoundedSummary,
) -> Result<String, String> {
    coefficient_bound(&value.coefficient_bound())
}

fn projection(value: &BoundProjection) -> &'static str {
    match value {
        BoundProjection::Coefficient => ".coefficient",
        BoundProjection::Summary => ".summary",
    }
}

fn value_ref(value: &ProofPayloadValueRef) -> String {
    match value {
        ProofPayloadValueRef::Predecessor {
            binding_event,
            input_position,
            projection: selected,
        } => {
            format!(".predecessor {input_position} {binding_event} {}", projection(selected))
        }
        ProofPayloadValueRef::Result { event, projection: selected } => {
            format!(".result {event} {}", projection(selected))
        }
        ProofPayloadValueRef::Transfer(event) => format!(".transfer {event}"),
    }
}

fn relation_rule(value: &ProofPayloadRelationRule) -> Result<String, String> {
    Ok(match value {
        ProofPayloadRelationRule::Universal { computed, lhs, lhs_layout, rhs_result } => format!(
            ".universal {computed} ({}) ({}) {rhs_result}",
            monomial(lhs)?,
            option(lhs_layout.as_ref(), matrix_layout)?
        ),
        ProofPayloadRelationRule::Gadget { gadget, decomposition, input, input_result } => format!(
            ".gadget ({}) ({}) ⟨{input}⟩ {input_result}",
            owner(gadget),
            owner(decomposition)
        ),
    })
}

fn matrix_layout(value: &MatrixLayout) -> Result<String, String> {
    Ok(format!("⟨{}, {}, {}⟩", quoted(&value.name)?, value.row_stride, value.column_stride))
}

fn rule_text(value: &ProofPayloadRule) -> Result<String, String> {
    Ok(match value {
        ProofPayloadRule::Authority(authority) => {
            format!(".authority ({})", authority_text(authority)?)
        }
        ProofPayloadRule::Identity { input } => format!(".identity ({})", value_ref(input)),
        ProofPayloadRule::Sum { inputs } => format!(".sum {}", refs(inputs)?),
        ProofPayloadRule::Scale { value, scale } => {
            format!(".scale ({}) ({})", value_ref(value), scale_text(scale))
        }
        ProofPayloadRule::MonomialProduct { monomial: product, factors } => {
            format!(".monomialProduct ({}) {}", monomial(product)?, list(factors, factor_evidence)?)
        }
        ProofPayloadRule::Product { left, right, facts } => format!(
            ".product ({}) ({}) ({})",
            value_ref(left),
            value_ref(right),
            product_facts(facts)
        ),
        ProofPayloadRule::Tensor {
            left,
            right,
            left_is_constant_polynomial,
            right_is_constant_polynomial,
        } => format!(
            ".tensor ({}) ({}) {} {}",
            value_ref(left),
            value_ref(right),
            super::bool_text(*left_is_constant_polynomial),
            super::bool_text(*right_is_constant_polynomial)
        ),
        ProofPayloadRule::Maximum { .. } => {
            return Err("unsupported Maximum bound rule in Security0 renderer".to_owned());
        }
        ProofPayloadRule::WeightedSum { .. } => {
            return Err("unsupported WeightedSum bound rule in Security0 renderer".to_owned());
        }
    })
}

fn authority_text(value: &ProofPayloadAuthority) -> Result<String, String> {
    Ok(match value {
        ProofPayloadAuthority::FactStore => ".factStore".to_owned(),
        ProofPayloadAuthority::ProgramFamilyFact => ".programFamilyFact".to_owned(),
        ProofPayloadAuthority::Operator => ".operator".to_owned(),
        ProofPayloadAuthority::RelationPreimageSource { source } => {
            format!(".relationPreimageSource ⟨{source}⟩")
        }
        ProofPayloadAuthority::Unavailable => {
            return Err("unsupported unavailable authority in Security0 renderer".to_owned());
        }
    })
}

fn refs(values: &[ProofPayloadValueRef]) -> Result<String, String> {
    list(values, |item| Ok(value_ref(item)))
}

fn scale_text(value: &ProofPayloadScale) -> String {
    match value {
        ProofPayloadScale::Value(value) => format!(".value ({})", value_ref(value)),
        ProofPayloadScale::Magnitude(value) => format!(".magnitude {value}"),
    }
}

fn factor_evidence(value: &ProofPayloadFactorEvidence) -> Result<String, String> {
    Ok(format!(
        "⟨{}, {}, {}⟩",
        value_ref(&value.bound),
        super::bool_text(value.is_constant_polynomial),
        value.support_upper.map_or_else(|| "none".to_owned(), |item| format!("some {item}")),
    ))
}

fn product_facts(value: &MatrixProductFacts) -> String {
    format!(
        "⟨{}, {}, {}, {}, {}⟩",
        super::bool_text(value.left_is_constant_polynomial),
        super::bool_text(value.right_is_constant_polynomial),
        value
            .right_known_zero_rows
            .as_ref()
            .map_or_else(|| "none".to_owned(), |item| format!("some {item}")),
        value.left_support_upper.map_or_else(|| "none".to_owned(), |item| format!("some {item}")),
        value.right_support_upper.map_or_else(|| "none".to_owned(), |item| format!("some {item}")),
    )
}

fn merge(value: &ProofPayloadCoefficientMerge) -> Result<String, String> {
    let source = match &value.source {
        ProofPayloadCoefficientMergeSource::Operator { inputs } => format!(
            ".operator (⟨{}, {}⟩, ⟨{}, {}⟩)",
            inputs[0].value_event,
            inputs[0].term_ordinal,
            inputs[1].value_event,
            inputs[1].term_ordinal
        ),
        ProofPayloadCoefficientMergeSource::Relation { application, source_term_ordinal } => {
            format!(".relation {application} {source_term_ordinal}")
        }
    };
    Ok(format!(
        "⟨{}, {source}, {}, ({})⟩",
        owner(&value.owner),
        monomial(&value.output)?,
        value.signed_contribution
    ))
}

fn pre_fold(value: &ProofPayloadPreFoldPolynomial) -> Result<String, String> {
    let summary_evidence = match value.summary_evidence.as_ref() {
        Some(item) => format!("(some ({}))", value_ref(item)),
        None => "none".to_owned(),
    };
    Ok(format!(
        ".preFoldPolynomial {} {} {} {}",
        value.result_event,
        terms_text(&value.terms)?,
        summary_text(&value.summary)?,
        summary_evidence,
    ))
}

fn update_state(
    statement: &CertificateDocumentV1,
    index: usize,
    event: &ProofPayloadEvent,
    stack: &mut Vec<FrameState>,
) -> Result<(), String> {
    match event {
        ProofPayloadEvent::InvocationStart { root } => {
            statement
                .expressions
                .get(usize::try_from(root.expression_row).map_err(|_| "expression row overflow")?)
                .ok_or_else(|| {
                    format!("Security0 invocation root {} is dangling", root.expression_row)
                })?;
            stack.push(FrameState { root: *root, start: index });
        }
        ProofPayloadEvent::InvocationEnd { root, .. } => {
            let frame = stack.pop().ok_or("Security0 invocation end has no active frame")?;
            if frame.root != *root {
                return Err("Security0 invocation end root does not match active frame".to_owned());
            }
        }
        _ => {}
    }
    Ok(())
}

fn render_state(cursor: usize, stack: &[FrameState]) -> Result<String, String> {
    let frames = stack.iter().rev().map(frame_text).collect::<Result<Vec<_>, _>>()?;
    Ok(format!("⟨{cursor}, [{}]⟩", frames.join(", ")))
}

fn frame_text(value: &FrameState) -> Result<String, String> {
    Ok(format!("⟨{}, {}⟩", owner(&value.root), value.start))
}

fn render_event_package(
    package: usize,
    start: usize,
    end: usize,
    events: &[String],
    exact_terms_events: &[Option<ExactTermsEvent>],
    frame_starts: &[usize],
) -> String {
    let module = format!("Events{package:03}");
    let mut source = format!(
        "import {MODULE_ROOT}.Cert.Cert\n\nset_option autoImplicit false\nset_option relaxedAutoImplicit false\n\nnamespace {NAMESPACE}.Proof.{module}\n\nopen Mxx.Certificate.OperationalNoise\nopen TallSecurity0ABI\n\n"
    );
    for index in start..end {
        if let Some(exact) = &exact_terms_events[index] {
            writeln!(source, "def exact{index}RawTerms : List Term := {}\n", exact.terms)
                .expect("writing to String cannot fail");
            if matches!(exact.kind, ExactTermsEventKind::Result(_)) {
                writeln!(
                    source,
                    "theorem exact{index}RawTermsValid :\n    exact{index}RawTerms.all (fun term => monomialValid document term.monomial) = true := by\n  decide +kernel\n"
                )
                .expect("writing to String cannot fail");
            }
        }
        writeln!(source, "def event{index} : Event := {}\n", events[index])
            .expect("writing to String cannot fail");
    }
    let first_leaf = start / EVENT_LEAF_SIZE;
    let leaf_count = (end - start).div_ceil(EVENT_LEAF_SIZE);
    for local in 0..leaf_count {
        let leaf = first_leaf + local;
        let leaf_start = start + local * EVENT_LEAF_SIZE;
        let leaf_end = (leaf_start + EVENT_LEAF_SIZE).min(end);
        let values = (leaf_start..leaf_end)
            .map(|index| {
                format!("{{ event := event{index}\n    frameStart := {} }}", frame_starts[index])
            })
            .collect::<Vec<_>>()
            .join(",\n  ");
        writeln!(source, "def eventLeaf{leaf} : Array AnnotatedEvent := #[\n  {values}\n]\n")
            .expect("writing to String cannot fail");
    }
    writeln!(source, "end {NAMESPACE}.Proof.{module}").expect("writing to String cannot fail");
    source
}

struct HistoryTreeFact {
    name: String,
    leaf: usize,
    start: usize,
    end: usize,
    height: usize,
    dependencies: Vec<String>,
}

fn render_history(event_count: usize, package_count: usize) -> String {
    let mut source = String::new();
    for package in 0..package_count {
        writeln!(source, "import {MODULE_ROOT}.Proof.Events{package:03}")
            .expect("writing to String cannot fail");
    }
    source.push_str("\nset_option autoImplicit false\nset_option relaxedAutoImplicit false\n\n");
    writeln!(source, "namespace {NAMESPACE}\n").expect("writing to String cannot fail");
    source.push_str("open Mxx.Certificate.OperationalNoise\nopen TallSecurity0ABI\n");
    for package in 0..package_count {
        writeln!(source, "open Proof.Events{package:03}").expect("writing to String cannot fail");
    }
    let leaf_count = event_count.div_ceil(EVENT_LEAF_SIZE);
    let mut facts = Vec::new();
    let root = render_balanced_leaves(0, leaf_count, 0, &mut facts, &mut source);
    for leaf in 0..leaf_count {
        let size = EVENT_LEAF_SIZE.min(event_count - leaf * EVENT_LEAF_SIZE);
        writeln!(
            source,
            "\ntheorem eventLeaf{leaf}Size : eventLeaf{leaf}.size = {size} := by\n  rfl\n"
        )
        .expect("writing to String cannot fail");
    }
    writeln!(
        source,
        "\ndef historyLeaves : RowTable (Array AnnotatedEvent) := {root}\n\ndef history : EventHistory := ⟨historyLeaves, {event_count}⟩\n"
    )
    .expect("writing to String cannot fail");
    render_history_facts(&facts, leaf_count, &mut source);
    writeln!(source, "\nend {NAMESPACE}").expect("writing to String cannot fail");
    source
}

fn render_balanced_leaves(
    start: usize,
    end: usize,
    depth: usize,
    facts: &mut Vec<HistoryTreeFact>,
    out: &mut String,
) -> String {
    if start == end {
        return ".empty".to_owned();
    }
    let middle = (start + end) / 2;
    if depth == 4 {
        let name = format!("historyTree{middle}");
        let dependency_start = facts.len();
        let value = render_balanced_leaves(start, end, 0, facts, out);
        let added = &facts[dependency_start..];
        let nested_names =
            added.iter().flat_map(|fact| fact.dependencies.iter()).collect::<Vec<_>>();
        let dependencies = added
            .iter()
            .filter(|fact| !nested_names.iter().any(|name| *name == &fact.name))
            .map(|fact| fact.name.clone())
            .collect();
        writeln!(out, "def {name} : RowTable (Array AnnotatedEvent) := {value}")
            .expect("writing to String cannot fail");
        facts.push(HistoryTreeFact {
            name: name.clone(),
            leaf: middle,
            start,
            end,
            height: balanced_tree_height(end - start),
            dependencies,
        });
        return name;
    }
    let left = render_balanced_leaves(start, middle, depth + 1, facts, out);
    let right = render_balanced_leaves(middle + 1, end, depth + 1, facts, out);
    format!("(.node {middle} eventLeaf{middle} {left} {right})")
}

fn render_history_facts(facts: &[HistoryTreeFact], leaf_count: usize, out: &mut String) {
    for fact in facts {
        let lower =
            if fact.start == 0 { "none".to_owned() } else { format!("(some {})", fact.start - 1) };
        let upper =
            if fact.end == leaf_count { "none".to_owned() } else { format!("(some {})", fact.end) };
        let dependency_facts = |suffix: &str| {
            fact.dependencies
                .iter()
                .map(|dependency| format!("{dependency}{suffix}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let ordered_facts = dependency_facts("Ordered");
        let balanced_facts = dependency_facts("Balanced");
        let height_facts = dependency_facts("Height");
        let node_count_facts = dependency_facts("NodeCount");
        let all_bool_facts = dependency_facts("AllBool");
        let event_leaf_size_fact = format!("eventLeaf{}Size", fact.leaf);
        writeln!(
            out,
            "\ntheorem {name}Ordered :\n    RowTable.orderedFrom {lower} {upper} {name} = true := by\n  simp [{name}, RowTable.orderedFrom{previous}]",
            previous = if ordered_facts.is_empty() {
                String::new()
            } else {
                format!(", {ordered_facts}")
            },
            name = fact.name,
        )
        .expect("writing to String cannot fail");
        writeln!(
            out,
            "\ntheorem {name}Balanced : {name}.balanced = true := by\n  simp [{name}, RowTable.balanced, RowTable.height{previous}]",
            previous = if balanced_facts.is_empty() && height_facts.is_empty() {
                String::new()
            } else {
                let facts = [balanced_facts.as_str(), height_facts.as_str()]
                    .into_iter()
                    .filter(|facts| !facts.is_empty())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(", {facts}")
            },
            name = fact.name,
        )
        .expect("writing to String cannot fail");
        writeln!(
            out,
            "\ntheorem {name}Height : RowTable.height {name} = {} := by\n  simp [{name}, RowTable.height{previous}]",
            fact.height,
            previous = if height_facts.is_empty() {
                String::new()
            } else {
                format!(", {height_facts}")
            },
            name = fact.name,
        )
        .expect("writing to String cannot fail");
        writeln!(
            out,
            "\ntheorem {name}NodeCount : rowTableNodeCount {name} = {} := by\n  simp [{name}, rowTableNodeCount{previous}]",
            fact.end - fact.start,
            previous = if node_count_facts.is_empty() {
                String::new()
            } else {
                format!(", {node_count_facts}")
            },
            name = fact.name,
        )
        .expect("writing to String cannot fail");
        writeln!(
            out,
            "\ntheorem {name}AllBool :\n    {name}.allBool (fun leaf events =>\n      decide (leaf < history.leafCount) &&\n        decide (events.size = history.expectedLeafSize leaf)) = true := by\n  simp only [{name}, RowTable.allBool]\n  {child_rewrites}\n  simp [history, EventHistory.leafCount, EventHistory.expectedLeafSize, {event_leaf_size_fact}] <;> decide +kernel",
            child_rewrites = if all_bool_facts.is_empty() {
                String::new()
            } else {
                format!("rw [{}]", all_bool_facts)
            },
            event_leaf_size_fact = event_leaf_size_fact,
            name = fact.name,
        )
        .expect("writing to String cannot fail");
    }
    // Each opaque history subtree has already had its own facts checked above.
    // The root rewrites therefore need only the facts for opaque subtrees that
    // occur directly in the root expression.  Passing every descendant fact to
    // every root theorem makes simp re-open the entire history tree and defeats
    // the balanced proof decomposition.
    let nested_names = facts.iter().flat_map(|fact| fact.dependencies.iter()).collect::<Vec<_>>();
    let top_facts = facts
        .iter()
        .filter(|fact| !nested_names.iter().any(|name| *name == &fact.name))
        .collect::<Vec<_>>();
    let top_fact_names = |suffix: &str| {
        top_facts
            .iter()
            .map(|fact| format!("{}{}", fact.name, suffix))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let root_event_leaf_sizes = (0..leaf_count)
        .filter(|leaf| !top_facts.iter().any(|fact| fact.start <= *leaf && *leaf < fact.end))
        .map(|leaf| format!("eventLeaf{leaf}Size"))
        .collect::<Vec<_>>();
    let ordered_facts = top_fact_names("Ordered");
    let balanced_facts = top_fact_names("Balanced");
    let height_facts = top_fact_names("Height");
    let node_count_facts = top_fact_names("NodeCount");
    let all_bool_facts = top_fact_names("AllBool");
    let all_bool_facts = [all_bool_facts, root_event_leaf_sizes.join(", ")]
        .into_iter()
        .filter(|facts| !facts.is_empty())
        .collect::<Vec<_>>()
        .join(", ");
    let suffix = |facts: &str| {
        if facts.is_empty() { String::new() } else { format!(", {facts}") }
    };
    writeln!(
        out,
        "\ntheorem historyLeavesOrdered :\n    RowTable.orderedFrom none none historyLeaves = true := by\n  simp [historyLeaves, RowTable.orderedFrom{ordered}]",
        ordered = suffix(&ordered_facts),
    )
    .expect("writing to String cannot fail");
    writeln!(
        out,
        "\ntheorem historyLeavesBalanced : historyLeaves.balanced = true := by\n  simp [historyLeaves, RowTable.balanced, RowTable.height{balanced}]",
        balanced = suffix(&[balanced_facts.as_str(), height_facts.as_str()]
            .into_iter()
            .filter(|facts| !facts.is_empty())
            .collect::<Vec<_>>()
            .join(", ")),
    )
    .expect("writing to String cannot fail");
    writeln!(
        out,
        "\ntheorem historyLeavesNodeCount : rowTableNodeCount historyLeaves = {} := by\n  simp [historyLeaves, rowTableNodeCount{node_count}]",
        leaf_count,
        node_count = suffix(&node_count_facts),
    )
    .expect("writing to String cannot fail");
    writeln!(
        out,
        "\ntheorem historyLeavesAllBool :\n    historyLeaves.allBool (fun leaf events =>\n      decide (leaf < history.leafCount) &&\n        decide (events.size = history.expectedLeafSize leaf)) = true := by\n  simp only [historyLeaves, RowTable.allBool]\n  {child_rewrites}\n  simp [history, EventHistory.leafCount, EventHistory.expectedLeafSize{event_leaf_sizes}] <;> decide +kernel",
        child_rewrites = if all_bool_facts.is_empty() {
            String::new()
        } else {
            format!("rw [{}]", all_bool_facts)
        },
        event_leaf_sizes = if root_event_leaf_sizes.is_empty() {
            String::new()
        } else {
            format!(", {}", root_event_leaf_sizes.join(", "))
        },
    )
    .expect("writing to String cannot fail");
    out.push_str(
        "\n\ntheorem historyWellFormedFact : history.wellFormed = true := by\n  rw [EventHistory.wellFormed]\n  have historyLeavesEq : history.leaves = historyLeaves := by\n    rfl\n  rw [historyLeavesEq]\n  rw [historyLeavesAllBool]\n  have leavesWellFormed : historyLeaves.wellFormed = true := by\n    rw [RowTable.wellFormed, historyLeavesOrdered, historyLeavesBalanced]\n    rfl\n  rw [leavesWellFormed, historyLeavesNodeCount]\n  simp [history, EventHistory.leafCount, eventLeafSize]\n",
    );
}

fn balanced_tree_height(count: usize) -> usize {
    if count == 0 {
        0
    } else {
        let left = count / 2;
        let right = count - left - 1;
        balanced_tree_height(left).max(balanced_tree_height(right)) + 1
    }
}

fn render_replay_package(
    package: usize,
    start_chunk: usize,
    end_chunk: usize,
    states: &[String],
    event_states: &[String],
    exact_terms_events: &[Option<ExactTermsEvent>],
    frame_starts: &[usize],
    event_count: usize,
) -> String {
    let module = format!("Replay{package:03}");
    let mut source = format!(
        "import {MODULE_ROOT}.Proof.History\n\nset_option autoImplicit false\nset_option relaxedAutoImplicit false\n\nnamespace {NAMESPACE}.Proof.{module}\n\nopen Mxx.Certificate.OperationalNoise\nopen TallSecurity0ABI\nopen {NAMESPACE}\n\n"
    );
    for chunk in start_chunk..=end_chunk {
        writeln!(source, "def replayState{chunk} : ReplayState := {}", states[chunk])
            .expect("writing to String cannot fail");
    }
    source.push('\n');
    for chunk in start_chunk..end_chunk {
        let actual_end = states[chunk + 1]
            .strip_prefix('⟨')
            .and_then(|text| text.split_once(',').map(|pair| pair.0))
            .expect("state renderer always begins with cursor");
        let chunk_start = chunk * REPLAY_CHUNK_SIZE;
        let chunk_end = ((chunk + 1) * REPLAY_CHUNK_SIZE).min(exact_terms_events.len());
        let valid_terms = (chunk_start..chunk_end)
            .filter_map(|event| {
                exact_terms_events[event].as_ref().and_then(|exact| {
                    matches!(exact.kind, ExactTermsEventKind::Result(_)).then_some(event)
                })
            })
            .collect::<Vec<_>>();
        let proof = if valid_terms.is_empty() {
            "by rfl".to_owned()
        } else {
            for event in chunk_start + 1..chunk_end {
                writeln!(
                    source,
                    "def replayEventState{event} : ReplayState := {}",
                    event_states[event]
                )
                .expect("writing to String cannot fail");
            }
            source.push('\n');
            for event in chunk_start..chunk_end {
                let next = event + 1;
                let state = if event == chunk_start {
                    format!("replayState{chunk}")
                } else {
                    format!("replayEventState{event}")
                };
                let next_state = if next == chunk_end {
                    format!("replayState{}", chunk + 1)
                } else {
                    format!("replayEventState{next}")
                };
                if let Some(ExactTermsEvent { kind: ExactTermsEventKind::Result(result), .. }) =
                    &exact_terms_events[event]
                {
                    let event_package = event / EVENT_PACKAGE_SIZE;
                    writeln!(
                        source,
                        "theorem replayStep{event} : stepAt document history {state} = some {next_state} := by\n  apply stepAt_resultExact document history {state}\n    ({}) Proof.Events{event_package:03}.exact{event}RawTerms {} {} {} ({}) {}\n  · rfl\n  · rfl\n  · rw [Proof.Events{event_package:03}.exact{event}RawTermsValid]\n    rfl\n",
                        result.owner,
                        result.coefficient_bound,
                        result.coefficient_producer,
                        result.summary,
                        result.summary_producer,
                        frame_starts[event],
                    )
                    .expect("writing to String cannot fail");
                } else {
                    writeln!(
                        source,
                        "theorem replayStep{event} : stepAt document history {state} = some {next_state} := by\n  rfl\n"
                    )
                    .expect("writing to String cannot fail");
                }
            }
            let mut chunk_proof = format!(
                "by\n    unfold replayRange\n    rw [show replayState{chunk}.cursor = {chunk_start} by rfl]\n    rw [show history.size = {event_count} by rfl]\n    dsimp only\n    rw [show (decide ({chunk_start} ≤ {actual_end}) && decide ({actual_end} ≤ {event_count}) &&\n        decide ({actual_end} - {chunk_start} ≤ 4)) = true by decide]\n    simp only [if_true]\n    unfold replayBlock\n    rw [show replayState{chunk}.cursor = {chunk_start} by rfl]\n    unfold replayBlock.run\n    dsimp only\n    rw [show Nat.min ({actual_end} - {chunk_start}) 4 = {} by decide]\n    simp only [replayBlock.run, Option.bind_eq_bind]",
                chunk_end - chunk_start,
            );
            for event in chunk_start..chunk_end {
                write!(
                    chunk_proof,
                    "\n    rw [replayStep{event}]\n    simp only [Option.bind_some]"
                )
                .expect("writing to String cannot fail");
            }
            chunk_proof.push_str("\n    rfl");
            chunk_proof
        };
        writeln!(
            source,
            "theorem replayChunk{chunk} : ReplayChain document history replayState{chunk} replayState{} :=\n  .chunk {actual_end} ({proof})\n",
            chunk + 1
        )
        .expect("writing to String cannot fail");
    }
    let chain = balanced_chain("replayChunk", start_chunk, end_chunk);
    writeln!(
        source,
        "theorem replayShard{package} : ReplayChain document history replayState{start_chunk} replayState{end_chunk} :=\n  {chain}\n\nend {NAMESPACE}.Proof.{module}"
    )
    .expect("writing to String cannot fail");
    source
}

fn balanced_chain(prefix: &str, start: usize, end: usize) -> String {
    if end - start == 1 {
        return format!("{prefix}{start}");
    }
    let middle = (start + end) / 2;
    format!(
        "(.trans {} {})",
        balanced_chain(prefix, start, middle),
        balanced_chain(prefix, middle, end)
    )
}

fn render_balanced_replay_chain(
    start: usize,
    end: usize,
    depth: usize,
    package_count: usize,
    final_leaf: usize,
    source: &mut String,
) -> String {
    if end - start == 1 {
        return format!("replayShard{start}");
    }
    let middle = (start + end) / 2;
    if depth == 4 {
        let name = format!("replayTree{start}_{end}");
        let value = render_balanced_replay_chain(start, end, 0, package_count, final_leaf, source);
        render_checked_replay_tree(&name, start, end, package_count, final_leaf, &value, source);
        return name;
    }
    let left =
        render_balanced_replay_chain(start, middle, depth + 1, package_count, final_leaf, source);
    let right =
        render_balanced_replay_chain(middle, end, depth + 1, package_count, final_leaf, source);
    format!("(.trans {left} {right})")
}

fn render_checked_replay_tree(
    name: &str,
    start: usize,
    end: usize,
    package_count: usize,
    final_leaf: usize,
    value: &str,
    source: &mut String,
) {
    let start_state = start * (EVENT_PACKAGE_SIZE / REPLAY_CHUNK_SIZE);
    let end_state = if end == package_count {
        final_leaf
    } else {
        end * (EVENT_PACKAGE_SIZE / REPLAY_CHUNK_SIZE)
    };
    let start_state_ref = format!("Proof.Replay{start:03}.replayState{start_state}");
    let end_package = end.min(package_count - 1);
    let end_state_ref = format!("Proof.Replay{end_package:03}.replayState{end_state}");
    writeln!(
        source,
        "theorem {name} : ReplayChain document history {start_state_ref} {end_state_ref} := by\n  exact {value}\n"
    )
    .expect("writing to String cannot fail");
}

fn render_top(event_count: usize, package_count: usize, final_leaf: usize) -> String {
    let mut source = String::new();
    writeln!(source, "import {MODULE_ROOT}.Proof.History").expect("writing to String cannot fail");
    for package in 0..package_count {
        writeln!(source, "import {MODULE_ROOT}.Proof.Replay{package:03}")
            .expect("writing to String cannot fail");
    }
    source.push_str("\nset_option autoImplicit false\nset_option relaxedAutoImplicit false\n\n");
    writeln!(source, "namespace {NAMESPACE}\n").expect("writing to String cannot fail");
    source.push_str("open Mxx.Certificate.OperationalNoise\nopen TallSecurity0ABI\n");
    for package in 0..package_count {
        writeln!(source, "open Proof.Replay{package:03}").expect("writing to String cannot fail");
    }
    let chain = if package_count > 2 {
        let middle = package_count / 2;
        let left =
            render_balanced_replay_chain(0, middle, 1, package_count, final_leaf, &mut source);
        let right = render_balanced_replay_chain(
            middle,
            package_count,
            1,
            package_count,
            final_leaf,
            &mut source,
        );
        let left_name = format!("replayHalf0_{middle}");
        let right_name = format!("replayHalf{middle}_{package_count}");
        render_checked_replay_tree(
            &left_name,
            0,
            middle,
            package_count,
            final_leaf,
            &left,
            &mut source,
        );
        render_checked_replay_tree(
            &right_name,
            middle,
            package_count,
            package_count,
            final_leaf,
            &right,
            &mut source,
        );
        format!("(.trans {left_name} {right_name})")
    } else {
        render_balanced_replay_chain(0, package_count, 0, package_count, final_leaf, &mut source)
    };
    writeln!(
        source,
        "\ntheorem historyWellFormed : history.wellFormed = true := by\n  exact {NAMESPACE}.historyWellFormedFact\n\ntheorem replayChain : ReplayChain document history initialState replayState{final_leaf} := by\n  exact {chain}\n\ntheorem proofValid : Valid document history := by\n  refine ⟨historyWellFormed, replayState{final_leaf}, replayChain, rfl, rfl⟩\n\ntheorem eventCount : history.size = {event_count} := by\n  rfl\n\nend {NAMESPACE}"
    )
    .expect("writing to String cannot fail");
    source
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::{
        certificate_schema::{CertificateExpressionRow, CertificateResidualRootV1},
        g0::{
            CanonicalExpressionDescriptor, CanonicalExpressionOperator, StableOperator,
            StableValueType,
        },
    };

    fn row(inputs: Vec<u64>) -> CertificateExpressionRow {
        CertificateExpressionRow {
            descriptor: CanonicalExpressionDescriptor::Operation {
                operator: CanonicalExpressionOperator::Stable(StableOperator::Argument {
                    position: 0,
                    value_type: StableValueType::Int,
                }),
                value_type: StableValueType::Int,
            },
            inputs,
            program: Some(0),
        }
    }

    fn statement() -> CertificateDocumentV1 {
        CertificateDocumentV1 {
            schema_id: "mxx.operational-noise.certificate",
            schema_version: 1,
            plaintext_modulus: "2".to_owned(),
            ciphertext_modulus: "257".to_owned(),
            ring_dimension: 1,
            expressions: vec![row(vec![]), row(vec![]), row(vec![]), row(vec![1]), row(vec![2])],
            programs: vec![],
            sources: vec![],
            events: vec![],
            index_uses: vec![],
            slice_groups: vec![],
            residual_root: CertificateResidualRootV1::Closed { expression: 0 },
        }
    }

    fn owner(expression_row: u64) -> ProofPayloadOwner {
        ProofPayloadOwner { scope: ProofPayloadScope::Program { program_row: 0 }, expression_row }
    }

    fn result_replay_event() -> ResultReplayEvent {
        ResultReplayEvent {
            owner: "owner3".to_owned(),
            coefficient_bound: ".large".to_owned(),
            coefficient_producer: 2,
            summary: ".exactZero".to_owned(),
            summary_producer: "none".to_owned(),
        }
    }

    #[test]
    fn replay_boundary_retains_only_invocation_depth() {
        let statement = statement();
        let mut stack = Vec::new();
        let events = [
            ProofPayloadEvent::InvocationStart { root: owner(0) },
            ProofPayloadEvent::InvocationStart { root: owner(1) },
        ];
        for (index, event) in events.iter().enumerate() {
            update_state(&statement, index, event, &mut stack).expect("honest invocation state");
        }
        assert_eq!(
            render_state(2, &stack).expect("render compact invocation state"),
            "⟨2, [⟨⟨.program ⟨0⟩, ⟨1⟩⟩, 1⟩, ⟨⟨.program ⟨0⟩, ⟨0⟩⟩, 0⟩]⟩"
        );
    }

    #[test]
    fn predecessor_refs_distinguish_consumers_at_the_same_position_by_binding_event() {
        let first = ProofPayloadValueRef::Predecessor {
            binding_event: 3,
            input_position: 0,
            projection: BoundProjection::Coefficient,
        };
        let second = ProofPayloadValueRef::Predecessor {
            binding_event: 9,
            input_position: 0,
            projection: BoundProjection::Coefficient,
        };
        assert_eq!(value_ref(&first), ".predecessor 0 3 .coefficient");
        assert_eq!(value_ref(&second), ".predecessor 0 9 .coefficient");
    }

    #[test]
    fn event_packages_define_payloads_once_before_ordered_leaf_references() {
        let events = vec![
            ".invocationStart owner0".to_owned(),
            ".resultExact owner1 exact1RawTerms .large 0 .exactZero none".to_owned(),
            ".invocationEndExact owner2 1 exact2RawTerms .large 0 .exactZero none".to_owned(),
        ];
        let source = render_event_package(
            0,
            0,
            events.len(),
            &events,
            &[
                None,
                Some(ExactTermsEvent {
                    kind: ExactTermsEventKind::Result(result_replay_event()),
                    terms: "[]".to_owned(),
                }),
                Some(ExactTermsEvent {
                    kind: ExactTermsEventKind::InvocationEnd,
                    terms: "[]".to_owned(),
                }),
            ],
            &[3, 7, 7],
        );

        assert_eq!(source.matches("def event0 : Event :=").count(), 1);
        assert_eq!(source.matches("def event1 : Event :=").count(), 1);
        assert_eq!(source.matches(".invocationStart owner0").count(), 1);
        assert_eq!(
            source.matches(".resultExact owner1 exact1RawTerms .large 0 .exactZero none").count(),
            1
        );
        assert_eq!(
            source
                .matches(".invocationEndExact owner2 1 exact2RawTerms .large 0 .exactZero none")
                .count(),
            1
        );
        assert!(!source.contains("exact0RawTerms"));
        assert_eq!(source.matches("def exact1RawTerms : List Term := []").count(), 1);
        assert_eq!(source.matches("def exact2RawTerms : List Term := []").count(), 1);
        assert_eq!(source.matches("theorem exact1RawTermsValid").count(), 1);
        assert!(!source.contains("theorem exact2RawTermsValid"));
        let first =
            source.find("{ event := event0\n    frameStart := 3 }").expect("first event reference");
        let second = source
            .find("{ event := event1\n    frameStart := 7 }")
            .expect("second event reference");
        assert!(first < second);
    }

    #[test]
    fn replay_top_uses_checked_depth_four_intermediates_with_exact_endpoints() {
        let source = render_top(32 * EVENT_PACKAGE_SIZE - 3, 32, 32 * 64);

        assert!(source.contains(
            "theorem replayTree0_2 : ReplayChain document history Proof.Replay000.replayState0 Proof.Replay002.replayState128"
        ));
        assert!(source.contains(
            "theorem replayTree30_32 : ReplayChain document history Proof.Replay030.replayState1920 Proof.Replay031.replayState2048"
        ));
        assert!(source.contains(
            "theorem replayHalf0_16 : ReplayChain document history Proof.Replay000.replayState0 Proof.Replay016.replayState1024"
        ));
        assert!(source.contains(
            "theorem replayHalf16_32 : ReplayChain document history Proof.Replay016.replayState1024 Proof.Replay031.replayState2048"
        ));
        assert!(source.contains("exact (.trans replayHalf0_16 replayHalf16_32)"));
        assert!(source.contains("exact (.trans (.trans (.trans replayTree0_2 replayTree2_4)"));
        let tokens = source
            .split(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
            .collect::<Vec<_>>();
        for package in 0..32 {
            assert_eq!(
                tokens.iter().filter(|token| **token == format!("replayShard{package}")).count(),
                1
            );
        }
        assert!(source.find("replayShard0").unwrap() < source.find("replayShard31").unwrap());
    }

    #[test]
    fn pre_fold_links_exact_zero_and_finite_evidence_to_prior_events() {
        let zero = ProofPayloadPreFoldPolynomial {
            result_event: 4,
            terms: vec![],
            summary: crate::operational_noise::normal_form::BoundedSummary::zero(),
            summary_evidence: None,
        };
        assert_eq!(
            pre_fold(&zero).expect("render exact-zero pre-fold"),
            ".preFoldPolynomial 4 [] .exactZero none"
        );

        let finite = ProofPayloadPreFoldPolynomial {
            result_event: 8,
            terms: vec![],
            summary: crate::operational_noise::normal_form::BoundedSummary::finite(
                crate::operational_noise::facts::BoundExpression::new(7_u8.into()),
            ),
            summary_evidence: Some(ProofPayloadValueRef::Result {
                event: 8,
                projection: BoundProjection::Summary,
            }),
        };
        assert_eq!(
            pre_fold(&finite).expect("render finite pre-fold"),
            ".preFoldPolynomial 8 [] (.finite 7) (some (.result 8 .summary))"
        );

        let predecessor = ProofPayloadPreFoldPolynomial {
            result_event: 9,
            terms: vec![],
            summary: crate::operational_noise::normal_form::BoundedSummary::zero(),
            summary_evidence: Some(ProofPayloadValueRef::Predecessor {
                binding_event: 5,
                input_position: 2,
                projection: BoundProjection::Coefficient,
            }),
        };
        assert_eq!(
            pre_fold(&predecessor).expect("render predecessor evidence"),
            ".preFoldPolynomial 9 [] .exactZero (some (.predecessor 2 5 .coefficient))"
        );

        let transfer = ProofPayloadPreFoldPolynomial {
            result_event: 10,
            terms: vec![],
            summary: crate::operational_noise::normal_form::BoundedSummary::zero(),
            summary_evidence: Some(ProofPayloadValueRef::Transfer(6)),
        };
        assert_eq!(
            pre_fold(&transfer).expect("render transfer evidence"),
            ".preFoldPolynomial 10 [] .exactZero (some (.transfer 6))"
        );
    }

    #[test]
    fn finite_bounds_are_parenthesized_in_every_reached_event_position() {
        let finite = crate::operational_noise::normal_form::BoundedSummary::finite(
            crate::operational_noise::facts::BoundExpression::new(7_u8.into()),
        );
        let coefficient = NumericContract::Known(CoefficientBound::Finite(
            crate::operational_noise::facts::BoundExpression::new(7_u8.into()),
        ));
        assert_eq!(coefficient_bound(&coefficient).expect("finite coefficient"), "(.finite 7)");

        let result_coefficient = ProofPayloadEvent::Result {
            owner: owner(3),
            value: ProofPayloadValue::Coefficient { bound: coefficient },
        };
        assert_eq!(
            event_text(&result_coefficient).expect("finite coefficient result"),
            ".resultCoefficient (⟨.program ⟨0⟩, ⟨3⟩⟩) (.finite 7)"
        );

        let result_exact = ProofPayloadEvent::Result {
            owner: owner(3),
            value: ProofPayloadValue::Exact {
                terms: vec![],
                coefficient_bound: finite.coefficient_bound(),
                coefficient_producer: 7,
                summary: finite.clone(),
                summary_producer: Some(8),
            },
        };
        assert_eq!(
            event_text(&result_exact).expect("finite exact result"),
            ".resultExact (⟨.program ⟨0⟩, ⟨3⟩⟩) [] (.finite 7) 7 (.finite 7) (some (8))"
        );

        let invocation_end = ProofPayloadEvent::InvocationEnd {
            root: owner(3),
            result: ProofPayloadValue::Exact {
                terms: vec![],
                coefficient_bound: finite.coefficient_bound(),
                coefficient_producer: 7,
                summary: finite,
                summary_producer: Some(8),
            },
            pre_fold_event: 11,
        };
        assert_eq!(
            event_text(&invocation_end).expect("finite invocation end"),
            ".invocationEndExact (⟨.program ⟨0⟩, ⟨3⟩⟩) 11 [] (.finite 7) 7 (.finite 7) (some (8))"
        );
    }

    #[test]
    fn one_history_leaf_is_replayed_as_four_four_event_chunks() {
        let states = vec![
            "⟨0, []⟩".to_owned(),
            "⟨4, []⟩".to_owned(),
            "⟨8, []⟩".to_owned(),
            "⟨12, []⟩".to_owned(),
            "⟨16, []⟩".to_owned(),
        ];
        let replay = render_replay_package(0, 0, 4, &states, &[], &vec![None; 16], &[], 16);
        assert!(replay.contains(
            "theorem replayChunk0 : ReplayChain document history replayState0 replayState1 :=\n  .chunk 4 (by rfl)"
        ));
        assert!(replay.contains(
            "theorem replayChunk3 : ReplayChain document history replayState3 replayState4 :=\n  .chunk 16 (by rfl)"
        ));
        assert!(replay.contains(
            "(.trans (.trans replayChunk0 replayChunk1) (.trans replayChunk2 replayChunk3))"
        ));
    }

    #[test]
    fn final_partial_replay_chunk_uses_the_exact_event_cursor() {
        let states = vec!["⟨0, []⟩".to_owned(), "⟨4, []⟩".to_owned(), "⟨5, []⟩".to_owned()];
        let replay = render_replay_package(0, 0, 2, &states, &[], &vec![None; 8], &[], 5);
        assert!(replay.contains(".chunk 4 (by rfl)"));
        assert!(replay.contains(".chunk 5 (by rfl)"));
    }

    #[test]
    fn replay_chunk_rewrites_the_authoritative_result_terms_validity() {
        let states = vec!["⟨0, []⟩".to_owned(), "⟨4, []⟩".to_owned()];
        let mut exact = vec![None; 4];
        exact[3] = Some(ExactTermsEvent {
            kind: ExactTermsEventKind::Result(result_replay_event()),
            terms: "[]".to_owned(),
        });
        let event_states = (0..=4).map(|cursor| format!("⟨{cursor}, []⟩")).collect::<Vec<_>>();
        let replay = render_replay_package(0, 0, 1, &states, &event_states, &exact, &[0; 4], 4);
        assert!(replay.contains("apply stepAt_resultExact document history replayEventState3"));
        assert!(replay.contains(
            "theorem replayStep0 : stepAt document history replayState0 = some replayEventState1"
        ));
        assert!(replay.contains(
            "theorem replayStep3 : stepAt document history replayEventState3 = some replayState1"
        ));
        assert!(!replay.contains("def replayEventState0"));
        assert!(!replay.contains("def replayEventState4"));
        assert!(replay.contains("rw [Proof.Events000.exact3RawTermsValid]"));
        assert!(replay.contains("rw [replayStep0]"));
        assert!(replay.contains("rw [replayStep3]"));
        assert!(!replay.contains("rw [Proof.Events000.exact3RawTermsValid]\n    rfl)"));
    }

    #[test]
    fn nested_invocation_frames_are_exact_at_four_event_boundaries() {
        let statement = statement();
        let mut stack = Vec::new();
        let outer = owner(0);
        let inner = owner(1);
        update_state(
            &statement,
            0,
            &ProofPayloadEvent::InvocationStart { root: outer },
            &mut stack,
        )
        .expect("start outer invocation");
        assert_eq!(
            render_state(4, &stack).expect("outer boundary"),
            "⟨4, [⟨⟨.program ⟨0⟩, ⟨0⟩⟩, 0⟩]⟩"
        );
        update_state(
            &statement,
            4,
            &ProofPayloadEvent::InvocationStart { root: inner },
            &mut stack,
        )
        .expect("start inner invocation");
        assert_eq!(
            render_state(8, &stack).expect("nested boundary"),
            "⟨8, [⟨⟨.program ⟨0⟩, ⟨1⟩⟩, 4⟩, ⟨⟨.program ⟨0⟩, ⟨0⟩⟩, 0⟩]⟩"
        );
    }

    #[test]
    fn dense_merge_probe_uses_four_event_states_from_5328_through_5376() {
        let frame = "[⟨⟨.program ⟨214⟩, ⟨30220⟩⟩, 0⟩]";
        let mut states = vec![String::new(); 1345];
        for chunk in 1332..=1344 {
            states[chunk] = format!("⟨{}, {frame}⟩", chunk * REPLAY_CHUNK_SIZE);
        }
        let replay =
            render_replay_package(20, 1332, 1344, &states, &[], &vec![None; 5376], &[], 5376);
        for cursor in (5328..=5376).step_by(REPLAY_CHUNK_SIZE) {
            assert!(replay.contains(&format!("ReplayState := ⟨{cursor}, {frame}⟩")));
        }
        for (chunk, end) in (1332..1344).zip((5332..=5376).step_by(REPLAY_CHUNK_SIZE)) {
            assert!(replay.contains(&format!(
                "theorem replayChunk{chunk} : ReplayChain document history replayState{chunk} replayState{} :=\n  .chunk {end} (by rfl)",
                chunk + 1
            )));
        }
    }
}
