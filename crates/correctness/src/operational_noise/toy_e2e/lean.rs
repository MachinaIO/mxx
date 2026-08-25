use super::{
    ToyAuditAuthority, ToyAuditBound, ToyAuditEvent, ToyAuditMerge, ToyAuditMergeSource,
    ToyAuditMonomial, ToyAuditOwner, ToyAuditProjection, ToyAuditRule, ToyAuditScale,
    ToyAuditScope, ToyAuditTerm, ToyAuditValue, ToyAuditValueRef, ToyGeneratedLean, ToySliceV1,
    ToySourceV1, audit_event,
};
use crate::operational_noise::{
    certificate_schema::{
        CertificateDocumentV1, CertificateEventRowV1, CertificateResidualRootV1,
        CertificateSourceRowV1,
    },
    g0::{
        CanonicalEventOperator, CanonicalExpressionDescriptor, CanonicalExpressionOperator,
        CanonicalExpressionSource, StableConstantValue, StableMatrixOperation, StableOperator,
        StableSamplerOperation, StableTrapdoorOperation, StableValueType,
    },
};

pub(super) fn render(source: &ToySourceV1, slice: &ToySliceV1) -> Result<ToyGeneratedLean, String> {
    Ok(ToyGeneratedLean {
        cert: render_cert(source, &slice.statement)?.into_bytes(),
        proof: render_proof(&slice.events)?.into_bytes(),
    })
}

fn quoted(value: &str) -> Result<String, String> {
    serde_json::to_string(value)
        .map_err(|error| format!("toy Lean string encoding failed: {error}"))
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
        None => Ok("none".into()),
    }
}

fn value_type(value: &StableValueType) -> Result<String, String> {
    Ok(match value {
        StableValueType::Int => ".int".into(),
        StableValueType::Trapdoor => ".trapdoor".into(),
        StableValueType::Matrix { modulus, ring_dimension, rows, columns }
            if modulus == "257" && *ring_dimension == 1 =>
        {
            format!("toyMatrix {rows} {columns}")
        }
        StableValueType::Matrix { .. } => {
            return Err("unexpected matrix type in fixed toy Lean renderer".into());
        }
        _ => return Err("unsupported value type in fixed toy Lean renderer".into()),
    })
}

fn refs(values: &[u64]) -> String {
    format!("[{}]", values.iter().map(|value| format!("⟨{value}⟩")).collect::<Vec<_>>().join(", "))
}

fn expression_descriptor(value: &CanonicalExpressionDescriptor) -> Result<String, String> {
    Ok(match value {
        CanonicalExpressionDescriptor::Source {
            source: CanonicalExpressionSource::Direct { source },
        } => format!(".source (.direct ⟨{source}⟩)"),
        CanonicalExpressionDescriptor::Operation { operator, value_type: ty } => {
            let operator = match operator {
                CanonicalExpressionOperator::Event(CanonicalEventOperator::Sampler { event }) => {
                    format!(".event (.sampler ⟨{}⟩)", event.row)
                }
                CanonicalExpressionOperator::Stable(StableOperator::ProgramCall) => {
                    ".stable .programCall".into()
                }
                CanonicalExpressionOperator::Stable(StableOperator::Matrix { operation }) => {
                    let operation = match operation {
                        StableMatrixOperation::Add => ".add",
                        StableMatrixOperation::Subtract => ".subtract",
                        StableMatrixOperation::Multiply => ".multiply",
                        StableMatrixOperation::Scale => ".scale",
                        _ => {
                            return Err(
                                "unsupported matrix operation in fixed toy Lean renderer".into()
                            );
                        }
                    };
                    format!(".stable (.matrix {operation})")
                }
                CanonicalExpressionOperator::Stable(StableOperator::Trapdoor {
                    operation:
                        StableTrapdoorOperation::Generate {
                            descriptor,
                            parameters,
                            paired_public_event,
                            paired_public_output_role,
                        },
                }) => format!(
                    ".stable (.trapdoor\n            (.generate {} {} ({}) {}))",
                    quoted(descriptor)?,
                    list(parameters, |value| Ok(value.to_string()))?,
                    option(paired_public_event.as_ref(), |event| Ok(format!("⟨{}⟩", event.row)))?,
                    quoted(paired_public_output_role)?
                ),
                _ => return Err("unsupported expression in fixed toy Lean renderer".into()),
            };
            format!(".operation ({operator}) ({})", value_type(ty)?)
        }
        _ => return Err("unsupported expression descriptor in fixed toy Lean renderer".into()),
    })
}

fn observed_node(owner: &impl serde::Serialize) -> Result<u64, String> {
    let value = serde_json::to_value(owner)
        .map_err(|error| format!("toy observed-wire projection failed: {error}"))?;
    let object =
        value.as_object().ok_or_else(|| "toy observed wire is not an object".to_owned())?;
    if object.get("stage").and_then(|value| value.as_str()) != Some("consumer") ||
        object.get("path").and_then(|value| value.as_u64()) != Some(0) ||
        object.get("port").and_then(|value| value.as_u64()) != Some(0) ||
        object
            .get("definition")
            .and_then(|value| value.get("kind"))
            .and_then(|value| value.as_str()) !=
            Some("root")
    {
        return Err("unsupported observed wire in fixed toy Lean renderer".into());
    }
    object
        .get("node")
        .and_then(|value| value.as_u64())
        .ok_or_else(|| "toy observed wire is missing its node".to_owned())
}

fn render_cert(source: &ToySourceV1, document: &CertificateDocumentV1) -> Result<String, String> {
    if !document.index_uses.is_empty() || !document.slice_groups.is_empty() {
        return Err("fixed toy Lean renderer does not support LUT rows".into());
    }
    let mut expressions = Vec::with_capacity(document.expressions.len());
    for row in &document.expressions {
        expressions.push(format!(
            "{{ descriptor := {}\n          inputs := {}\n          program := {} }}",
            expression_descriptor(&row.descriptor)?,
            refs(&row.inputs),
            row.program.map_or_else(|| "none".into(), |row| format!("some ⟨{row}⟩"))
        ));
    }
    let programs = document
        .programs
        .iter()
        .map(|program| {
            let signature = list(&program.signature, |input| {
                let range = option(input.trusted_index_range.as_ref(), |range| {
                    Ok(format!("⟨{}, {}⟩", range.minimum, range.maximum_exclusive))
                })?;
                Ok(format!("⟨{}, {range}⟩", value_type(&input.value_type)?))
            })?;
            let family = option(program.family.as_ref(), |family| {
                if family.artifact.is_some() {
                    return Err("fixed toy Lean renderer does not support program artifacts".into());
                }
                Ok(format!(
                    "⟨⟨{}, {}⟩, {}, {}, none⟩",
                    family.domain.minimum,
                    family.domain.maximum_exclusive,
                    value_type(&family.element_type)?,
                    family.reducible
                ))
            })?;
            Ok(format!(
                "⟨{signature},\n        {},\n        {family},\n        ⟨{}⟩⟩",
                value_type(&program.output)?,
                program.root
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let sources = document
        .sources
        .iter()
        .map(|source| match source {
            CertificateSourceRowV1::Constant { value } => {
                let StableConstantValue::Int { value: integer } = &value.value else {
                    return Err("unsupported source constant in fixed toy Lean renderer".into());
                };
                Ok(format!(
                    ".constant ⟨{}, .int {}⟩",
                    value_type(&value.value_type)?,
                    quoted(integer)?
                ))
            }
            _ => Err("unsupported source row in fixed toy Lean renderer".into()),
        })
        .collect::<Result<Vec<_>, String>>()?;
    let events = document
        .events
        .iter()
        .map(|event| match event {
            CertificateEventRowV1::Sampler { owner, operation, contract: None } => {
                let operation = match operation {
                    StableSamplerOperation::UniformResidue { output } => {
                        format!(".uniformResidue ({})", value_type(output)?)
                    }
                    StableSamplerOperation::Gaussian { output, sigma, max_coefficient_bound } => {
                        format!(
                            ".gaussian ({})\n          {} {}",
                            value_type(output)?,
                            quoted(sigma)?,
                            quoted(max_coefficient_bound)?
                        )
                    }
                    StableSamplerOperation::Trapdoor {
                        output,
                        sigma,
                        gadget_base,
                        digit_count,
                        preimage_max_coefficient_bound,
                    } => format!(
                        ".trapdoor ({})\n          {}\n          {gadget_base} {digit_count} {}",
                        value_type(output)?,
                        quoted(sigma)?,
                        quoted(preimage_max_coefficient_bound)?
                    ),
                    StableSamplerOperation::Preimage { output, max_coefficient_bound } => format!(
                        ".preimage ({}) {}",
                        value_type(output)?,
                        quoted(max_coefficient_bound)?
                    ),
                    _ => {
                        return Err(
                            "unsupported sampler operation in fixed toy Lean renderer".into()
                        );
                    }
                };
                Ok(format!(
                    ".sampler (toyWire {})\n        ({operation}) none",
                    observed_node(owner)?
                ))
            }
            _ => Err("unsupported event row in fixed toy Lean renderer".into()),
        })
        .collect::<Result<Vec<_>, String>>()?;
    let root = match document.residual_root {
        CertificateResidualRootV1::Closed { expression } => expression,
        _ => return Err("unsupported residual root in fixed toy Lean renderer".into()),
    };
    let rows = (0..document.expressions.len())
        .map(|row| format!("⟨{row}⟩"))
        .collect::<Vec<_>>()
        .chunks(7)
        .map(|chunk| chunk.join(", "))
        .collect::<Vec<_>>()
        .join(",\n      ");
    let source_text = format!(
        "{{ schemaId := {}\n    schemaVersion := {}\n    abi := {}\n    rustProjectionVersion := {}\n    leanAbiVersion := {}\n    request := ⟨{}, [], []⟩\n    parameters := ⟨{}, {}, {}, {}, {}, {}, {}, {}, {}, {}⟩ }}",
        quoted(&source.schema_id)?,
        source.schema_version,
        quoted(&source.abi)?,
        quoted(&source.rust_projection_version)?,
        quoted(&source.lean_abi_version)?,
        quoted(&source.request.target_id)?,
        quoted(&source.parameters.plaintext_modulus)?,
        quoted(&source.parameters.ciphertext_modulus)?,
        source.parameters.ring_dimension,
        quoted(&source.parameters.trapdoor_rows)?,
        quoted(&source.parameters.trapdoor_sigma)?,
        quoted(&source.parameters.gadget_base)?,
        quoted(&source.parameters.digit_count)?,
        quoted(&source.parameters.preimage_maximum_absolute_coefficient)?,
        quoted(&source.parameters.gaussian_sigma)?,
        quoted(&source.parameters.gaussian_maximum_absolute_coefficient)?
    );
    Ok(format!(
        "import Mxx.Certificate.OperationalNoise.ToyABI\n\nset_option autoImplicit false\nset_option relaxedAutoImplicit false\n\nnamespace Mxx.Certificate.OperationalNoise.ToyGenerated\n\nopen Mxx.Certificate.OperationalNoise\nopen SchemaV1\nopen ToyABI\n\ndef source : ToySource :=\n  {source_text}\n\ndef document : Document :=\n  {{ schemaId := {}\n    schemaVersion := {}\n    plaintextModulus := {}\n    ciphertextModulus := {}\n    ringDimension := {}\n    expressions :=\n{}\n    programs := {}\n    sources := {}\n    events := {}\n    indexUses := []\n    sliceGroups := []\n    residualRoot := .closed ⟨{root}⟩ }}\n\ndef rows : ToyRows :=\n  {{ expressions := [{rows}]\n    program := ⟨0⟩\n    sources := {}\n    events := {}\n    root := ⟨{root}⟩ }}\n\nend Mxx.Certificate.OperationalNoise.ToyGenerated\n",
        quoted(document.schema_id)?,
        document.schema_version,
        quoted(&document.plaintext_modulus)?,
        quoted(&document.ciphertext_modulus)?,
        document.ring_dimension,
        format!("      [ {} ]", expressions.join(",\n        ")),
        format!("[{}]", programs.join(", ")),
        format!("[{}]", sources.join(", ")),
        format!("[{}]", events.join(",\n      ")),
        refs(&(0..document.sources.len() as u64).collect::<Vec<_>>()),
        refs(&(0..document.events.len() as u64).collect::<Vec<_>>()),
    ))
}

fn owner(value: &ToyAuditOwner) -> String {
    if matches!(value.scope, ToyAuditScope::Closed { root_expression_row: 12 }) {
        return format!("o {}", value.expression_row);
    }
    let scope = match value.scope {
        ToyAuditScope::Closed { root_expression_row } => {
            format!(".closed {root_expression_row}")
        }
        ToyAuditScope::Program { program_row } => format!(".program {program_row}"),
    };
    format!("⟨{scope}, {}⟩", value.expression_row)
}

fn monomial(value: &ToyAuditMonomial) -> String {
    if value.central_factors.is_empty() &&
        value.ordered_factors.iter().all(|owner| {
            matches!(owner.scope, ToyAuditScope::Closed { root_expression_row: 12 })
        })
    {
        return format!(
            "m [{}]",
            value
                .ordered_factors
                .iter()
                .map(|owner| owner.expression_row.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let owners = |values: &[ToyAuditOwner]| {
        format!("[{}]", values.iter().map(owner).collect::<Vec<_>>().join(", "))
    };
    format!("⟨{}, {}⟩", owners(&value.central_factors), owners(&value.ordered_factors))
}

fn summary(value: &ToyAuditBound) -> Result<String, String> {
    Ok(match value {
        ToyAuditBound::ExactZero => ".exactZero".into(),
        ToyAuditBound::Finite { maximum_absolute_coefficient } => {
            format!("(.finite {maximum_absolute_coefficient})")
        }
        _ => return Err("non-finite bound in fixed toy Lean renderer".into()),
    })
}

fn term(value: &ToyAuditTerm) -> String {
    format!("t {} ({})", value.coefficient, monomial(&value.monomial))
}

fn value(value: &ToyAuditValue) -> Result<String, String> {
    Ok(match value {
        ToyAuditValue::Exact { terms, summary: bound } => {
            format!(".exact {} {}", list(terms, |item| Ok(term(item)))?, summary(bound)?)
        }
        ToyAuditValue::Coefficient { bound } => format!(".coefficient {}", summary(bound)?),
    })
}

fn projection(value: &ToyAuditProjection) -> &'static str {
    match value {
        ToyAuditProjection::Coefficient => ".coefficient",
        ToyAuditProjection::Summary => ".summary",
    }
}

fn value_ref(value: &ToyAuditValueRef) -> String {
    match value {
        ToyAuditValueRef::Predecessor { input_position, projection: selected } => {
            format!(".predecessor {input_position} {}", projection(selected))
        }
        ToyAuditValueRef::Result { event, projection: selected } => {
            format!(".result {event} {}", projection(selected))
        }
        ToyAuditValueRef::Transfer { event } => format!(".transfer {event}"),
    }
}

fn rule(value: &ToyAuditRule) -> Result<String, String> {
    Ok(match value {
        ToyAuditRule::Authority { authority } => match authority {
            ToyAuditAuthority::Operator => ".authority .operator".into(),
            ToyAuditAuthority::RelationPreimageSource { source } => {
                format!(".authority (.relationPreimageSource {source})")
            }
        },
        ToyAuditRule::Sum { inputs } => {
            format!(".sum {}", list(inputs, |value| Ok(value_ref(value)))?)
        }
        ToyAuditRule::Scale { value, scale } => {
            let scale = match scale {
                ToyAuditScale::Value { value } => format!(".value ({})", value_ref(value)),
                ToyAuditScale::Magnitude { magnitude } => format!(".magnitude {magnitude}"),
            };
            format!(".scale\n        ({}) ({scale})", value_ref(value))
        }
        ToyAuditRule::MonomialProduct { monomial: product, factors } => format!(
            ".monomialProduct ({}) {}",
            monomial(product),
            list(factors, |factor| Ok(format!(
                "⟨{}, {}, {}⟩",
                value_ref(&factor.bound),
                factor.is_constant_polynomial,
                factor.support_upper.map_or_else(|| "none".into(), |value| format!("some {value}"))
            )))?
        ),
        ToyAuditRule::Product { left, right, facts } => format!(
            ".product ({}) ({})\n        ⟨{}, {}, {}, {}, {}⟩",
            value_ref(left),
            value_ref(right),
            facts.left_is_constant_polynomial,
            facts.right_is_constant_polynomial,
            facts
                .right_known_zero_rows
                .as_ref()
                .map_or_else(|| "none".into(), |value| format!("some {value}")),
            facts.left_support_upper.map_or_else(|| "none".into(), |value| format!("some {value}")),
            facts
                .right_support_upper
                .map_or_else(|| "none".into(), |value| format!("some {value}")),
        ),
    })
}

fn merge(value: &ToyAuditMerge) -> String {
    let source = match &value.source {
        ToyAuditMergeSource::Operator { inputs } => format!(
            ".operator (⟨{}, {}⟩, ⟨{}, {}⟩)",
            inputs[0].value_event,
            inputs[0].term_ordinal,
            inputs[1].value_event,
            inputs[1].term_ordinal
        ),
        ToyAuditMergeSource::Relation { application, source_term_ordinal } => {
            format!(".relation {application} {source_term_ordinal}")
        }
    };
    format!(
        "⟨{}, {source}, {}, {}⟩",
        owner(&value.owner),
        monomial(&value.output),
        value.signed_contribution
    )
}

fn event(item: &ToyAuditEvent) -> Result<String, String> {
    Ok(match item {
        ToyAuditEvent::InvocationStart { root } => format!(".invocationStart ({})", owner(root)),
        ToyAuditEvent::Predecessor { consumer, input_position, predecessor, source_result } => {
            format!(
                ".predecessor ({}) {input_position} {predecessor} {source_result}",
                owner(consumer)
            )
        }
        ToyAuditEvent::Result { owner: result_owner, value: result } => {
            format!(".result ({}) ({})", owner(result_owner), value(result)?)
        }
        ToyAuditEvent::InvocationEnd { root, result } => {
            format!(".invocationEnd ({}) ({})", owner(root), value(result)?)
        }
        ToyAuditEvent::SpecializationComputed { owner: event_owner, dispatch, source } => format!(
            ".specializationComputed ({}) ⟨{}, {}, {}⟩ ⟨{}, {}⟩",
            owner(event_owner),
            dispatch.preimage_family,
            dispatch.preimage_source,
            dispatch.trapdoor_source,
            source.start,
            source.end
        ),
        ToyAuditEvent::AppliedUniversal {
            owner: event_owner,
            source_monomial,
            outer_coefficient,
            ordered_start,
            ordered_end_exclusive,
            computed,
            lhs,
            lhs_layout,
            rhs_result,
        } => {
            let layout = lhs_layout.as_ref().map_or_else(
                || "none".into(),
                |layout| {
                    format!(
                        "some ⟨{}, {}, {}⟩",
                        quoted(&layout.name).expect("layout name serialization"),
                        layout.row_stride,
                        layout.column_stride
                    )
                },
            );
            format!(
                ".appliedUniversal ({}) ({}) {outer_coefficient} {ordered_start} \
                 {ordered_end_exclusive} {computed} ({}) {layout} {rhs_result}",
                owner(event_owner),
                monomial(source_monomial),
                monomial(lhs)
            )
        }
        ToyAuditEvent::BoundTransfer { owner: event_owner, rule: transfer } => {
            format!(".boundTransfer ({}) ({})", owner(event_owner), rule(transfer)?)
        }
        ToyAuditEvent::CoefficientMerge { merge: coefficient_merge } => {
            format!(".coefficientMerge ({})", merge(coefficient_merge))
        }
        ToyAuditEvent::PreFoldPolynomial { terms, summary: bound, summary_evidence } => format!(
            ".preFoldPolynomial {} {} {}",
            list(terms, |item| Ok(term(item)))?,
            summary(bound)?,
            summary_evidence
                .as_ref()
                .map_or_else(|| "none".into(), |value| format!("(some ({}))", value_ref(value)))
        ),
        ToyAuditEvent::SurvivorFold { coefficient, bound } => {
            format!(".survivorFold {coefficient} {bound}")
        }
    })
}

fn render_proof(events: &[super::ToyEventV1]) -> Result<String, String> {
    let events = events.iter().map(audit_event).collect::<Vec<_>>();
    let rendered = events.iter().map(event).collect::<Result<Vec<_>, _>>()?;
    Ok(format!(
        "import Mxx.Certificate.OperationalNoise.ToyGenerated.Cert\n\nset_option autoImplicit false\nset_option relaxedAutoImplicit false\n\nnamespace Mxx.Certificate.OperationalNoise.ToyGenerated\n\nopen ToyABI\n\nprivate def o (row : Nat) : ToyOwner := ⟨.closed 12, row⟩\nprivate def m (rows : List Nat) : ToyMonomial := ⟨[], rows.map o⟩\nprivate def t (coefficient : Int) (monomial : ToyMonomial) : ToyTerm :=\n  ⟨monomial, coefficient⟩\n\ndef events : List ToyEvent :=\n  [ {} ]\n\ntheorem proofValid : ToyValid source document rows events := by\n  refine ⟨rfl, rfl, rfl, rfl, ?_⟩\n  intro index indexBound\n  rfl\n\nend Mxx.Certificate.OperationalNoise.ToyGenerated\n",
        rendered.join(",\n    ")
    ))
}
