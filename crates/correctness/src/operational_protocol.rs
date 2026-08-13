//! Mechanical conversion of executable DSL graphs into an operational-check workflow.

use crate::{
    ArtifactBinding, ArtifactName, ClosedProtocolBundle, ComparatorSpec, EndpointAnchors,
    InputContract, InputContractEntry, InputValueContract, ProtocolDecl, ProtocolInputBinding,
    ProtocolInputDestination, ProtocolInputId, ProtocolPreconditionSpec, ProtocolStage, StageId,
    StageInputName, Workflow,
};
use mxx_dsl::{BuiltGraph, DslContext, IdealSpec};
use mxx_ir_core::{WireType, node::NodeKind};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OperationalProtocolError {
    #[error("operational workflow must contain at least one stage")]
    EmptyWorkflow,
    #[error("operational workflow entrypoint is missing")]
    MissingEntrypoint,
    #[error("protocol input {name} has inconsistent wire types")]
    InputTypeMismatch { name: String },
    #[error("unsupported operational protocol input type: {0:?}")]
    UnsupportedInput(WireType),
    #[error("artifact input {name} has no unique preceding producer output")]
    ArtifactProducer { name: String },
    #[error("could not build the empty operational ideal: {0}")]
    Ideal(String),
    #[error("invalid operational protocol: {0}")]
    Protocol(String),
}

/// Optional semantic metadata for one exact matrix protocol input. Rust attaches these declared
/// facts by input name and does not attempt to infer them from the executable graph.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExactMatrixInputMetadata {
    pub canonical_coefficient_exclusive_upper_bound: Option<mxx_ir_core::IntExpr>,
    pub is_constant_polynomial: bool,
}

fn exact_input_contract(
    wire_type: &WireType,
    metadata: Option<&ExactMatrixInputMetadata>,
) -> Result<InputValueContract, OperationalProtocolError> {
    match wire_type {
        WireType::Matrix(matrix_type) => Ok(InputValueContract::MatrixExact {
            matrix_type: matrix_type.clone(),
            canonical_coefficient_exclusive_upper_bound: metadata
                .and_then(|metadata| metadata.canonical_coefficient_exclusive_upper_bound.clone()),
            is_constant_polynomial: metadata
                .is_some_and(|metadata| metadata.is_constant_polynomial),
        }),
        WireType::Bytes { length } => Ok(InputValueContract::Bytes { length: length.clone() }),
        WireType::Bool | WireType::ConstantBool => Ok(InputValueContract::Boolean),
        WireType::IndexedFamily { element, count } => Ok(InputValueContract::Family {
            count: count.clone(),
            element: Box::new(exact_input_contract(element, metadata)?),
        }),
        unsupported => Err(OperationalProtocolError::UnsupportedInput(unsupported.clone())),
    }
}

/// Builds a closed operational protocol from executable graphs. Artifact bindings are inferred by
/// matching each artifact input to exactly one preceding artifact output with the same name; all
/// ordinary inputs receive exact structural contracts. `complete_bundle` must install the
/// protocol-owned endpoint and decoder-target declarations before this function validates and
/// returns the declaration.
pub fn operational_protocol_from_graphs(
    stages: Vec<(String, &BuiltGraph)>,
    entrypoint: &str,
    exact_matrix_input_metadata: &BTreeMap<String, ExactMatrixInputMetadata>,
    complete_bundle: impl FnOnce(&mut ClosedProtocolBundle),
) -> Result<ProtocolDecl, OperationalProtocolError> {
    if stages.is_empty() {
        return Err(OperationalProtocolError::EmptyWorkflow);
    }
    if !stages.iter().any(|(id, _)| id == entrypoint) {
        return Err(OperationalProtocolError::MissingEntrypoint);
    }
    let mut contracts = BTreeMap::<String, (WireType, Vec<ProtocolInputDestination>)>::new();
    let mut protocol_stages = Vec::new();
    let mut preceding_outputs = Vec::<(StageId, String)>::new();
    for (id, graph) in stages {
        let stage_id = StageId(id);
        let mut bindings = Vec::new();
        for node in graph.graph.root_scope().nodes() {
            let NodeKind::Input { name, wire_type, artifact } = node.kind() else { continue };
            if let Some(artifact) = artifact {
                let candidates = preceding_outputs
                    .iter()
                    .filter(|(_, output)| output == &artifact.artifact_name)
                    .collect::<Vec<_>>();
                let [producer] = candidates.as_slice() else {
                    return Err(OperationalProtocolError::ArtifactProducer { name: name.clone() });
                };
                bindings.push(ArtifactBinding {
                    consumer_input: StageInputName(name.clone()),
                    producer_stage: producer.0.clone(),
                    producer_output: ArtifactName(producer.1.clone()),
                });
                continue;
            }
            let destination = ProtocolInputDestination::WorkflowStage {
                stage: stage_id.clone(),
                input: StageInputName(name.clone()),
            };
            match contracts.get_mut(name) {
                Some((existing, destinations)) => {
                    if existing != wire_type {
                        return Err(OperationalProtocolError::InputTypeMismatch {
                            name: name.clone(),
                        });
                    }
                    destinations.push(destination);
                }
                None => {
                    contracts.insert(name.clone(), (wire_type.clone(), vec![destination]));
                }
            }
        }
        protocol_stages.push(ProtocolStage {
            id: stage_id.clone(),
            graph: graph.graph.clone(),
            semantic_anchors: graph.anchors.clone(),
            derivation_attachments: graph.derivation_attachments.clone(),
            bindings,
        });
        preceding_outputs.extend(
            graph
                .graph
                .outputs()
                .iter()
                .filter(|(_, output)| output.confidentiality.is_some())
                .map(|(name, _)| (stage_id.clone(), name.clone())),
        );
    }
    let mut input_contract = Vec::new();
    let mut input_bindings = Vec::new();
    for (name, (wire_type, destinations)) in contracts {
        let id = ProtocolInputId(name.clone());
        let value = exact_input_contract(&wire_type, exact_matrix_input_metadata.get(&name))?;
        input_contract.push(InputContractEntry { id: id.clone(), name, value });
        input_bindings.push(ProtocolInputBinding { input: id, destinations });
    }
    let ideal = IdealSpec::new(
        DslContext::new("operational-check-ideal")
            .build()
            .map_err(|error| OperationalProtocolError::Ideal(error.to_string()))?,
    )
    .map_err(|error| OperationalProtocolError::Ideal(error.to_string()))?;
    let mut bundle = ClosedProtocolBundle {
        workflow: Workflow { stages: protocol_stages, entrypoint: StageId(entrypoint.to_owned()) },
        ideal,
        requirements: Vec::new(),
        comparator: ComparatorSpec::Equality { endpoints: Vec::new() },
        endpoints: EndpointAnchors::default(),
        operational_decoder_targets: Vec::new(),
        endpoint_specs: Vec::new(),
        input_contract: InputContract { inputs: input_contract },
        input_bindings,
        precondition_spec: ProtocolPreconditionSpec::default(),
    };
    complete_bundle(&mut bundle);
    ProtocolDecl::new(ProtocolDecl { params: Vec::new(), bundle })
        .map_err(|error| OperationalProtocolError::Protocol(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::Ring;
    use mxx_ir_core::IntExpr;

    #[test]
    fn exact_matrix_metadata_is_applied_through_an_indexed_family() {
        let ring = Ring::new(17, 8);
        let metadata = ExactMatrixInputMetadata {
            canonical_coefficient_exclusive_upper_bound: Some(IntExpr::constant(13)),
            is_constant_polynomial: true,
        };
        let wire_type = WireType::IndexedFamily {
            count: IntExpr::constant(4),
            element: Box::new(WireType::Matrix(ring.matrix_type((2, 3)))),
        };

        let contract = exact_input_contract(&wire_type, Some(&metadata)).unwrap();

        let InputValueContract::Family { element, .. } = contract else {
            panic!("indexed family must produce a family contract");
        };
        assert!(matches!(
            element.as_ref(),
            InputValueContract::MatrixExact {
                canonical_coefficient_exclusive_upper_bound: Some(upper),
                is_constant_polynomial: true,
                ..
            } if upper == &IntExpr::constant(13)
        ));
    }

    #[test]
    fn exact_matrix_metadata_defaults_are_conservative() {
        let ring = Ring::new(17, 8);

        let contract =
            exact_input_contract(&WireType::Matrix(ring.matrix_type((1, 1))), None).unwrap();

        assert!(matches!(
            contract,
            InputValueContract::MatrixExact {
                canonical_coefficient_exclusive_upper_bound: None,
                is_constant_polynomial: false,
                ..
            }
        ));
    }
}
