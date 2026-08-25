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
const EVENT_PACKAGE_SIZE: usize = 256;
const REPLAY_CHUNK_SIZE: usize = 4;

#[derive(Clone)]
struct FrameState {
    root: ProofPayloadOwner,
    start: usize,
}

pub(super) fn render(
    statement: &CertificateDocumentV1,
    proof: &OperationalProofPayload,
) -> Result<Vec<super::TallSecurity0GeneratedFile>, String> {
    if proof.events.is_empty() {
        return Err("Security0 proof payload is empty".to_owned());
    }
    let mut rendered = Vec::with_capacity(proof.events.len());
    let mut frame_starts = Vec::with_capacity(proof.events.len());
    let mut states = Vec::with_capacity(proof.events.len().div_ceil(REPLAY_CHUNK_SIZE) + 1);
    let mut stack = Vec::<FrameState>::new();
    states.push(render_state(0, &stack)?);
    for (index, event) in proof.events.iter().enumerate() {
        let frame_start = if matches!(event, ProofPayloadEvent::InvocationStart { .. }) {
            index
        } else {
            stack.last().map_or(0, |frame| frame.start)
        };
        frame_starts.push(frame_start);
        rendered.push(event_text(event)?);
        update_state(statement, index, event, &mut stack)?;
        if (index + 1) % REPLAY_CHUNK_SIZE == 0 || index + 1 == proof.events.len() {
            states.push(render_state(index + 1, &stack)?);
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
            render_event_package(package, start, end, &rendered, &frame_starts),
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
            render_replay_package(package, start_chunk, end_chunk, &states),
        ));
    }
    files.push(generated_file(
        "Proof/Proof.lean",
        render_top(proof.events.len(), package_count, states.len() - 1),
    ));
    Ok(files)
}

fn event_text(event: &ProofPayloadEvent) -> Result<String, String> {
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
                terms_text(terms)?,
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
                terms_text(terms)?,
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
    frame_starts: &[usize],
) -> String {
    let module = format!("Events{package:03}");
    let mut source = format!(
        "import {MODULE_ROOT}.Cert.Cert\n\nset_option autoImplicit false\nset_option relaxedAutoImplicit false\n\nnamespace {NAMESPACE}.Proof.{module}\n\nopen Mxx.Certificate.OperationalNoise\nopen TallSecurity0ABI\n\n"
    );
    let first_leaf = start / EVENT_LEAF_SIZE;
    let leaf_count = (end - start).div_ceil(EVENT_LEAF_SIZE);
    for local in 0..leaf_count {
        let leaf = first_leaf + local;
        let leaf_start = start + local * EVENT_LEAF_SIZE;
        let leaf_end = (leaf_start + EVENT_LEAF_SIZE).min(end);
        let values = (leaf_start..leaf_end)
            .map(|index| {
                format!(
                    "{{ event := {}\n    frameStart := {} }}",
                    events[index], frame_starts[index]
                )
            })
            .collect::<Vec<_>>()
            .join(",\n  ");
        writeln!(source, "def eventLeaf{leaf} : Array AnnotatedEvent := #[\n  {values}\n]\n")
            .expect("writing to String cannot fail");
    }
    writeln!(source, "end {NAMESPACE}.Proof.{module}").expect("writing to String cannot fail");
    source
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
    let root = render_balanced_leaves(0, leaf_count, 0, &mut source);
    writeln!(
        source,
        "\ndef historyLeaves : RowTable (Array AnnotatedEvent) := {root}\n\ndef history : EventHistory := ⟨historyLeaves, {event_count}⟩\n\nend {NAMESPACE}"
    )
    .expect("writing to String cannot fail");
    source
}

fn render_balanced_leaves(start: usize, end: usize, depth: usize, out: &mut String) -> String {
    if start == end {
        return ".empty".to_owned();
    }
    let middle = (start + end) / 2;
    if depth == 4 {
        let name = format!("historyTree{middle}");
        let value = render_balanced_leaves(start, end, 0, out);
        writeln!(out, "def {name} : RowTable (Array AnnotatedEvent) := {value}")
            .expect("writing to String cannot fail");
        return name;
    }
    let left = render_balanced_leaves(start, middle, depth + 1, out);
    let right = render_balanced_leaves(middle + 1, end, depth + 1, out);
    format!("(.node {middle} eventLeaf{middle} {left} {right})")
}

fn render_replay_package(
    package: usize,
    start_chunk: usize,
    end_chunk: usize,
    states: &[String],
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
        writeln!(
            source,
            "theorem replayChunk{chunk} : ReplayChain document history replayState{chunk} replayState{} :=\n  .chunk {actual_end} (by rfl)\n",
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

fn render_top(event_count: usize, package_count: usize, final_leaf: usize) -> String {
    let mut source = String::new();
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
    let chain = balanced_chain("replayShard", 0, package_count);
    writeln!(
        source,
        "\ntheorem historyWellFormed : history.wellFormed = true := by\n  rfl\n\ntheorem replayChain : ReplayChain document history initialState replayState{final_leaf} := by\n  exact {chain}\n\ntheorem proofValid : Valid document history := by\n  refine ⟨historyWellFormed, replayState{final_leaf}, replayChain, rfl, rfl⟩\n\ntheorem eventCount : history.size = {event_count} := by\n  rfl\n\nend {NAMESPACE}"
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
        let replay = render_replay_package(0, 0, 4, &states);
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
        let replay = render_replay_package(0, 0, 2, &states);
        assert!(replay.contains(".chunk 4 (by rfl)"));
        assert!(replay.contains(".chunk 5 (by rfl)"));
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
        let replay = render_replay_package(20, 1332, 1344, &states);
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
