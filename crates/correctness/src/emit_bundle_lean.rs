//! Lean transport emission for [`ClosedProtocolBundle`](crate::bundle::ClosedProtocolBundle).
//!
//! Programs are emitted elsewhere. This module only names those definitions and
//! serializes the closed bundle's declared transport data.

use crate::{
    StageId,
    bundle::{
        BundleValidationError, ClosedProtocolBundle, ComparatorSpec, DeclaredBoundExpr,
        EndpointSpecId, InputValueContract, ProtocolInputDestination,
    },
};
use mxx_ir_core::{FrozenGraphScopeId, IntExpr, types::MatrixType};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BundleProgramNames {
    pub stage_programs: BTreeMap<StageId, String>,
    pub ideal_program: String,
    pub requirement_programs: Vec<String>,
    pub comparator_program: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum BundleLeanEmitError {
    #[error(transparent)]
    InvalidBundle(#[from] BundleValidationError),
    #[error("a workflow stage has no caller-supplied Lean program name")]
    MissingStageProgram,
    #[error("caller supplied a Lean program name for a stage outside the workflow")]
    ExtraStageProgram,
    #[error("the number of requirement program names does not match the bundle")]
    RequirementProgramCount,
    #[error("the comparator program name does not match the comparator kind")]
    ComparatorProgramMismatch,
}

/// Emits a complete `Mxx.Certificate.ClosedProtocolBundle` value.
///
/// Strings in `names` are Lean terms naming definitions and are intentionally
/// not quoted. All strings originating in the bundle are quoted as data.
pub fn emit_closed_protocol_bundle(
    bundle: &ClosedProtocolBundle,
    names: &BundleProgramNames,
) -> Result<String, BundleLeanEmitError> {
    bundle.validate()?;
    if names.requirement_programs.len() != bundle.requirements.len() {
        return Err(BundleLeanEmitError::RequirementProgramCount);
    }
    if names.stage_programs.len() != bundle.workflow.stages.len() {
        return Err(
            if bundle
                .workflow
                .stages
                .iter()
                .any(|stage| !names.stage_programs.contains_key(&stage.id))
            {
                BundleLeanEmitError::MissingStageProgram
            } else {
                BundleLeanEmitError::ExtraStageProgram
            },
        );
    }
    if bundle.comparator.program().is_some() != names.comparator_program.is_some() {
        return Err(BundleLeanEmitError::ComparatorProgramMismatch);
    }

    let protocol_sources = bundle
        .input_bindings
        .iter()
        .flat_map(|binding| {
            binding.destinations.iter().filter_map(move |destination| match destination {
                ProtocolInputDestination::WorkflowStage { stage, input } => {
                    Some(((stage.clone(), input.0.clone()), binding.input.0.clone()))
                }
                ProtocolInputDestination::Requirement { .. } |
                ProtocolInputDestination::Ideal { .. } => None,
            })
        })
        .collect::<BTreeMap<_, _>>();

    let stages = bundle
        .workflow
        .stages
        .iter()
        .map(|stage| {
            let program = names
                .stage_programs
                .get(&stage.id)
                .ok_or(BundleLeanEmitError::MissingStageProgram)?;
            let artifact_sources = stage
                .bindings
                .iter()
                .map(|binding| (binding.consumer_input.0.as_str(), binding))
                .collect::<BTreeMap<_, _>>();
            let mut inputs = Vec::new();
            for node in stage.graph.root_scope().nodes() {
                let mxx_ir_core::node::NodeKind::Input { name, .. } = node.kind() else {
                    continue;
                };
                let source = if let Some(binding) = artifact_sources.get(name.as_str()) {
                    format!(
                        ".artifact {} {}",
                        lean_string(&binding.producer_stage.0),
                        lean_string(&binding.producer_output.0)
                    )
                } else {
                    let protocol = protocol_sources
                        .get(&(stage.id.clone(), name.clone()))
                        .expect("validated bundle has a protocol source for every root input");
                    format!(".protocol {}", lean_string(protocol))
                };
                inputs.push(format!("({}, {source})", lean_string(name)));
            }
            Ok(record(&[
                ("id", lean_string(&stage.id.0)),
                ("program", program.clone()),
                ("inputs", list(inputs)),
            ]))
        })
        .collect::<Result<Vec<_>, BundleLeanEmitError>>()?;
    let workflow = record(&[
        ("stages", list(stages)),
        ("entrypoint", lean_string(&bundle.workflow.entrypoint.0)),
    ]);

    let input_contract = record(&[(
        "inputs",
        list(bundle.input_contract.inputs.iter().map(|entry| {
            format!(
                "({}, {}, {})",
                protocol_input_id(&entry.id.0),
                lean_string(&entry.name),
                input_value_contract(&entry.value)
            )
        })),
    )]);
    let input_bindings = list(bundle.input_bindings.iter().map(|binding| {
        record(&[
            ("input", protocol_input_id(&binding.input.0)),
            ("destinations", list(binding.destinations.iter().map(input_destination))),
        ])
    }));

    let comparator_bindings = list(bundle.comparator.endpoints().iter().map(|binding| {
        record(&[
            ("endpoint", endpoint_spec(binding.endpoint).into()),
            ("actualInput", lean_string(&binding.actual_input)),
            ("idealInput", lean_string(&binding.ideal_input)),
            ("resultOutput", lean_string(&binding.result_output)),
            ("failureValue", binding.failure_value.to_string()),
        ])
    }));
    let comparator = match &bundle.comparator {
        ComparatorSpec::Equality { .. } => format!(".equality ({comparator_bindings})"),
        ComparatorSpec::EqualityAfterMap { .. } => format!(
            ".equalityAfterMap {} ({comparator_bindings})",
            names.comparator_program.as_ref().expect("comparator program presence was validated")
        ),
    };

    let endpoints = record(&[(
        "entries",
        list(bundle.endpoints.entries.iter().map(|endpoint| {
            let semantics = match &endpoint.semantics {
                crate::EndpointSemanticBinding::ThresholdDecode => ".thresholdDecode".to_owned(),
                crate::EndpointSemanticBinding::DiamondBoolean {
                    residual_stage,
                    residual_anchor,
                    carrier_stage,
                    carrier_anchor,
                    message,
                } => format!(
                    ".diamondBoolean {} {} {}",
                    semantic_anchor(&residual_stage.0, residual_anchor),
                    semantic_anchor(&carrier_stage.0, carrier_anchor),
                    protocol_input_id(&message.0)
                ),
            };
            record(&[
                ("specification", endpoint_spec(endpoint.spec).into()),
                ("stage", stage_id(&endpoint.stage.0)),
                ("semanticAnchor", semantic_anchor(&endpoint.stage.0, &endpoint.semantic_anchor)),
                ("semantics", semantics),
                ("workflowOutput", lean_string(&endpoint.workflow_output.output)),
                ("idealOutput", lean_string(&endpoint.ideal_output)),
            ])
        })),
    )]);

    let anchor_bindings = list(bundle.workflow.stages.iter().flat_map(|stage| {
        stage.semantic_anchors.iter().map(|(label, wires)| {
            record(&[
                ("anchor", semantic_anchor(&stage.id.0, label)),
                (
                    "wires",
                    list(wires.iter().map(|wire| {
                        core_wire(&stage.id.0, &wire.scope, wire.wire.node.0, wire.wire.port.0)
                    })),
                ),
            ])
        })
    }));

    Ok(record(&[
        ("workflow", workflow),
        ("ideal", names.ideal_program.clone()),
        ("requirements", list(names.requirement_programs.iter().cloned())),
        ("comparator", comparator),
        ("endpoints", endpoints),
        ("anchorBindings", anchor_bindings),
        (
            "endpointSpecs",
            list(bundle.endpoint_specs.iter().map(|value| endpoint_spec(*value).into())),
        ),
        ("inputContract", input_contract),
        ("inputBindings", input_bindings),
        (
            "preconditionSpec",
            record(&[(
                "requirementOutputs",
                list(
                    bundle
                        .precondition_spec
                        .requirement_outputs
                        .iter()
                        .map(|value| lean_string(value)),
                ),
            )]),
        ),
    ]))
}

fn input_value_contract(value: &InputValueContract) -> String {
    match value {
        InputValueContract::MatrixExact { matrix_type: value } => {
            format!(".matrixExact ({})", matrix_type(value))
        }
        InputValueContract::MatrixBounded { matrix_type: value, max_centered_coefficient } => {
            format!(
                ".matrixBounded ({}) ({})",
                matrix_type(value),
                declared_bound(max_centered_coefficient)
            )
        }
        InputValueContract::IntegerRange { lower, upper } => {
            format!(".integerRange ({}) ({})", int_expr(lower), int_expr(upper))
        }
        InputValueContract::Boolean => ".boolean".into(),
        InputValueContract::Bytes { length } => format!(".bytes ({})", int_expr(length)),
        InputValueContract::Family { count, element } => {
            format!(".family ({}) ({})", int_expr(count), input_value_contract(element))
        }
    }
}

fn declared_bound(value: &DeclaredBoundExpr) -> String {
    match value {
        DeclaredBoundExpr::Constant(value) => format!(".constant {value}"),
        DeclaredBoundExpr::Parameter(value) => format!(".parameter ({})", int_expr(value)),
        DeclaredBoundExpr::Add(left, right) => {
            format!(".add ({}) ({})", declared_bound(left), declared_bound(right))
        }
        DeclaredBoundExpr::Multiply(left, right) => {
            format!(".multiply ({}) ({})", declared_bound(left), declared_bound(right))
        }
        DeclaredBoundExpr::Maximum(left, right) => {
            format!(".maximum ({}) ({})", declared_bound(left), declared_bound(right))
        }
        DeclaredBoundExpr::Absolute(value) => format!(".absolute ({})", int_expr(value)),
        DeclaredBoundExpr::FloorDivide { value, positive_divisor } => {
            format!(".floorDivide ({}) {positive_divisor}", declared_bound(value))
        }
        DeclaredBoundExpr::MatrixProduct { ring_dimension, inner_dimension, left, right } => {
            format!(
                ".matrixProduct ({}) ({}) ({}) ({})",
                int_expr(ring_dimension),
                int_expr(inner_dimension),
                declared_bound(left),
                declared_bound(right)
            )
        }
        DeclaredBoundExpr::Minimum(left, right) => {
            format!(".minimum ({}) ({})", declared_bound(left), declared_bound(right))
        }
    }
}

fn input_destination(value: &ProtocolInputDestination) -> String {
    match value {
        ProtocolInputDestination::WorkflowStage { stage, input } => {
            format!(".workflowStage {} {}", stage_id(&stage.0), lean_string(&input.0))
        }
        ProtocolInputDestination::Requirement { requirement, input } => {
            format!(".requirement {requirement} {}", lean_string(input))
        }
        ProtocolInputDestination::Ideal { input } => format!(".ideal {}", lean_string(input)),
    }
}

fn int_expr(value: &IntExpr) -> String {
    match value {
        IntExpr::Const(value) => format!(".constant ({value} : Int)"),
        IntExpr::Var(value) => format!(".parameter {}", lean_string(value)),
        IntExpr::LoopIndex(value) => format!(".loopIndex {value}"),
        IntExpr::Add(left, right) => format!(".add ({}) ({})", int_expr(left), int_expr(right)),
        IntExpr::Sub(left, right) => {
            format!(".subtract ({}) ({})", int_expr(left), int_expr(right))
        }
        IntExpr::Mul(left, right) => {
            format!(".multiply ({}) ({})", int_expr(left), int_expr(right))
        }
        IntExpr::Div(left, right) => {
            format!(".divide ({}) ({})", int_expr(left), int_expr(right))
        }
        IntExpr::RoundDiv(left, right) => {
            format!(".roundDivide ({}) ({})", int_expr(left), int_expr(right))
        }
        IntExpr::Log2Ceil(value) => format!(".log2Ceil ({})", int_expr(value)),
    }
}

fn matrix_type(value: &MatrixType) -> String {
    record(&[
        ("modulus", int_expr(&value.modulus)),
        ("ringDimension", int_expr(&value.ring_dimension)),
        ("rows", int_expr(&value.rows)),
        ("columns", int_expr(&value.columns)),
    ])
}

fn endpoint_spec(value: EndpointSpecId) -> &'static str {
    match value {
        EndpointSpecId::ToyThresholdDecode => ".toyThresholdDecode",
        EndpointSpecId::DiamondBooleanInterval => ".diamondBooleanInterval",
    }
}

fn semantic_anchor(stage: &str, label: &str) -> String {
    record(&[("stage", stage_id(stage)), ("label", lean_string(label))])
}

fn protocol_input_id(value: &str) -> String {
    record(&[("name", lean_string(value))])
}

fn stage_id(value: &str) -> String {
    record(&[("name", lean_string(value))])
}

fn core_wire(stage: &str, scope: &FrozenGraphScopeId, node: u64, port: u32) -> String {
    record(&[
        ("stage", stage_id(stage)),
        ("scope", static_scope(scope)),
        ("node", record(&[("value", node.to_string())])),
        ("port", port.to_string()),
    ])
}

fn static_scope(value: &FrozenGraphScopeId) -> String {
    let path = match value {
        FrozenGraphScopeId::Root => Vec::new(),
        _ => vec![frozen_scope_definition_name(value)],
    };
    record(&[("path", list(path.into_iter().map(|value| lean_string(&value))))])
}

fn frozen_scope_definition_name(value: &FrozenGraphScopeId) -> String {
    match value {
        FrozenGraphScopeId::Root => "__root".to_owned(),
        FrozenGraphScopeId::Subgraph { canonical_name } => format!("subgraph:{canonical_name}"),
        FrozenGraphScopeId::ParallelBody { parent, owner } => {
            format!("parallel:{}:{}", frozen_scope_definition_name(parent), owner.0)
        }
        FrozenGraphScopeId::SequentialBody { parent, owner } => {
            format!("sequential:{}:{}", frozen_scope_definition_name(parent), owner.0)
        }
    }
}

fn lean_string(value: &str) -> String {
    format!(
        "\"{}\"",
        value
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t")
    )
}

fn list(values: impl IntoIterator<Item = String>) -> String {
    format!("[{}]", values.into_iter().collect::<Vec<_>>().join(", "))
}

fn record(fields: &[(impl AsRef<str>, String)]) -> String {
    format!(
        "{{ {} }}",
        fields
            .iter()
            .map(|(name, value)| format!("{} := {value}", name.as_ref()))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::{BigInt, BigUint};

    #[test]
    fn declared_bound_preserves_arbitrary_integers_and_all_shapes() {
        let huge = BigUint::parse_bytes(b"999999999999999999999999999999999", 10).unwrap();
        let bound = DeclaredBoundExpr::Minimum(
            Box::new(DeclaredBoundExpr::Maximum(
                Box::new(DeclaredBoundExpr::Add(
                    Box::new(DeclaredBoundExpr::Constant(huge)),
                    Box::new(DeclaredBoundExpr::Absolute(IntExpr::constant(BigInt::from(-7)))),
                )),
                Box::new(DeclaredBoundExpr::Multiply(
                    Box::new(DeclaredBoundExpr::Parameter(IntExpr::Var("n".into()))),
                    Box::new(DeclaredBoundExpr::FloorDivide {
                        value: Box::new(DeclaredBoundExpr::Constant(5u8.into())),
                        positive_divisor: 2u8.into(),
                    }),
                )),
            )),
            Box::new(DeclaredBoundExpr::MatrixProduct {
                ring_dimension: IntExpr::LoopIndex(3),
                inner_dimension: IntExpr::Log2Ceil(Box::new(IntExpr::constant(8))),
                left: Box::new(DeclaredBoundExpr::Constant(1u8.into())),
                right: Box::new(DeclaredBoundExpr::Constant(2u8.into())),
            }),
        );
        let emitted = declared_bound(&bound);
        assert!(emitted.contains("999999999999999999999999999999999"));
        assert!(emitted.starts_with(".minimum (.maximum (.add"));
        assert!(emitted.contains(".matrixProduct (.loopIndex 3) (.log2Ceil"));
    }

    #[test]
    fn input_destinations_are_explicit() {
        assert_eq!(
            input_destination(&ProtocolInputDestination::WorkflowStage {
                stage: StageId("stage".into()),
                input: crate::StageInputName("message".into()),
            }),
            ".workflowStage { name := \"stage\" } \"message\""
        );
        assert_eq!(
            input_destination(&ProtocolInputDestination::Requirement {
                requirement: 4,
                input: "x".into(),
            }),
            ".requirement 4 \"x\""
        );
        assert_eq!(
            input_destination(&ProtocolInputDestination::Ideal { input: "x".into() }),
            ".ideal \"x\""
        );
    }

    #[test]
    fn nested_scope_identity_is_lossless() {
        let scope = FrozenGraphScopeId::SequentialBody {
            parent: Box::new(FrozenGraphScopeId::ParallelBody {
                parent: Box::new(FrozenGraphScopeId::Subgraph { canonical_name: "callee".into() }),
                owner: mxx_ir_core::NodeId(7),
            }),
            owner: mxx_ir_core::NodeId(11),
        };
        assert_eq!(
            static_scope(&scope),
            "{ path := [\"sequential:parallel:subgraph:callee:7:11\"] }"
        );
    }
}
