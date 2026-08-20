//! Mechanical conversion of executable DSL graphs into an operational-check workflow.

use crate::{
    ArtifactBinding, ArtifactName, ClosedProtocolBundle, ComparatorSpec, EndpointAnchors,
    InputContract, InputContractEntry, InputValueContract, ProtocolDecl, ProtocolInputBinding,
    ProtocolInputDestination, ProtocolInputId, ProtocolPreconditionSpec, ProtocolStage, StageId,
    StageInputName, TrapdoorContractKind, TrapdoorContractMismatch, TrapdoorContractValue,
    TrapdoorInputContractField, Workflow,
};
use mxx_dsl::{BuiltGraph, DslContext, IdealSpec};
use mxx_ir_core::{WireType, node::NodeKind, types::MatrixType};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Error, Eq, PartialEq)]
pub enum OperationalProtocolError {
    #[error("operational workflow must contain at least one stage")]
    EmptyWorkflow,
    #[error("operational workflow entrypoint is missing")]
    MissingEntrypoint,
    #[error("protocol input {name} has inconsistent wire types")]
    InputTypeMismatch { name: String },
    #[error(
        "trapdoor protocol input {trapdoor_input:?} names missing public input {public_input:?}"
    )]
    MissingTrapdoorPublicInput { trapdoor_input: ProtocolInputId, public_input: ProtocolInputId },
    #[error(
        "trapdoor protocol input {trapdoor_input:?} conflicts with public input {public_input:?}: {mismatch:?}"
    )]
    ConflictingTrapdoorInput {
        trapdoor_input: ProtocolInputId,
        public_input: ProtocolInputId,
        mismatch: TrapdoorContractMismatch,
    },
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

/// The explicit public-matrix association for one protocol input whose value is a trapdoor.
///
/// This contract says nothing about the secret trapdoor contents.  It only identifies the public
/// matrix with which the caller promises the trapdoor is associated.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactTrapdoorInputMetadata {
    pub public_input: ProtocolInputId,
}

fn exact_input_contract(
    wire_type: &WireType,
    matrix_metadata: Option<&ExactMatrixInputMetadata>,
    trapdoor_metadata: Option<&ExactTrapdoorInputMetadata>,
    input: &ProtocolInputId,
) -> Result<InputValueContract, OperationalProtocolError> {
    match wire_type {
        WireType::Matrix(matrix_type) => Ok(InputValueContract::MatrixExact {
            matrix_type: matrix_type.clone(),
            canonical_coefficient_exclusive_upper_bound: matrix_metadata
                .and_then(|metadata| metadata.canonical_coefficient_exclusive_upper_bound.clone()),
            is_constant_polynomial: matrix_metadata
                .is_some_and(|metadata| metadata.is_constant_polynomial),
        }),
        WireType::Trapdoor {
            matrix,
            sigma,
            gadget_base,
            digit_count,
            preimage_max_coefficient_bound,
        } => {
            let metadata = trapdoor_metadata.ok_or_else(|| {
                OperationalProtocolError::MissingTrapdoorPublicInput {
                    trapdoor_input: input.clone(),
                    public_input: ProtocolInputId("<unspecified>".to_owned()),
                }
            })?;
            Ok(InputValueContract::Trapdoor {
                matrix_type: matrix.clone(),
                sigma: sigma.clone(),
                gadget_base: gadget_base.clone(),
                digit_count: digit_count.clone(),
                preimage_max_coefficient_bound: preimage_max_coefficient_bound.clone(),
                public_input: metadata.public_input.clone(),
            })
        }
        WireType::Bytes { length } => Ok(InputValueContract::Bytes { length: length.clone() }),
        WireType::Bool | WireType::ConstantBool => Ok(InputValueContract::Boolean),
        WireType::IndexedFamily { element, count } => Ok(InputValueContract::Family {
            count: count.clone(),
            element: Box::new(exact_input_contract(
                element,
                matrix_metadata,
                trapdoor_metadata,
                input,
            )?),
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
    exact_trapdoor_input_metadata: &BTreeMap<String, ExactTrapdoorInputMetadata>,
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
                        if trapdoor_wire_shape(existing).is_some() ||
                            trapdoor_wire_shape(wire_type).is_some()
                        {
                            let public_input = exact_trapdoor_input_metadata
                                .get(name)
                                .map(|metadata| metadata.public_input.clone())
                                .unwrap_or_else(|| ProtocolInputId("<unspecified>".to_owned()));
                            return Err(conflicting_trapdoor_input(
                                ProtocolInputId(name.clone()),
                                public_input,
                                trapdoor_wire_mismatch(existing, wire_type),
                            ));
                        }
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
    validate_trapdoor_metadata(&contracts, exact_trapdoor_input_metadata)?;
    let mut input_contract = Vec::new();
    let mut input_bindings = Vec::new();
    for (name, (wire_type, destinations)) in contracts {
        let id = ProtocolInputId(name.clone());
        let value = exact_input_contract(
            &wire_type,
            exact_matrix_input_metadata.get(&name),
            exact_trapdoor_input_metadata.get(&name),
            &id,
        )?;
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

fn validate_trapdoor_metadata(
    contracts: &BTreeMap<String, (WireType, Vec<ProtocolInputDestination>)>,
    metadata: &BTreeMap<String, ExactTrapdoorInputMetadata>,
) -> Result<(), OperationalProtocolError> {
    for (name, (wire_type, _)) in contracts {
        let Some(_) = trapdoor_wire_shape(wire_type) else {
            continue;
        };
        let trapdoor_input = ProtocolInputId(name.clone());
        let metadata = metadata.get(name).ok_or_else(|| {
            OperationalProtocolError::MissingTrapdoorPublicInput {
                trapdoor_input: trapdoor_input.clone(),
                public_input: ProtocolInputId("<unspecified>".to_owned()),
            }
        })?;
        if metadata.public_input == trapdoor_input {
            return Err(conflicting_trapdoor_input(
                trapdoor_input,
                metadata.public_input.clone(),
                TrapdoorContractMismatch {
                    field: TrapdoorInputContractField::DistinctPublicInput,
                    expected: TrapdoorContractValue::SameProtocolInput(false),
                    actual: TrapdoorContractValue::SameProtocolInput(true),
                },
            ));
        }
        let Some((public_wire_type, _)) = contracts.get(&metadata.public_input.0) else {
            return Err(OperationalProtocolError::MissingTrapdoorPublicInput {
                trapdoor_input,
                public_input: metadata.public_input.clone(),
            });
        };
        if let Some(mismatch) = trapdoor_public_wire_mismatch(wire_type, public_wire_type) {
            return Err(conflicting_trapdoor_input(
                trapdoor_input,
                metadata.public_input.clone(),
                mismatch,
            ));
        }
    }
    Ok(())
}

fn trapdoor_wire_shape(
    wire_type: &WireType,
) -> Option<(Option<&mxx_ir_core::IntExpr>, &MatrixType)> {
    trapdoor_wire_contract_shape(wire_type).map(|(count, matrix, ..)| (count, matrix))
}

fn trapdoor_wire_contract_shape(
    wire_type: &WireType,
) -> Option<(
    Option<&mxx_ir_core::IntExpr>,
    &MatrixType,
    &mxx_ir_core::RealExpr,
    &mxx_ir_core::IntExpr,
    &mxx_ir_core::IntExpr,
    &mxx_ir_core::IntExpr,
)> {
    match wire_type {
        WireType::Trapdoor {
            matrix,
            sigma,
            gadget_base,
            digit_count,
            preimage_max_coefficient_bound,
        } => Some((None, matrix, sigma, gadget_base, digit_count, preimage_max_coefficient_bound)),
        WireType::IndexedFamily { count, element } => match element.as_ref() {
            WireType::Trapdoor {
                matrix,
                sigma,
                gadget_base,
                digit_count,
                preimage_max_coefficient_bound,
            } => Some((
                Some(count),
                matrix,
                sigma,
                gadget_base,
                digit_count,
                preimage_max_coefficient_bound,
            )),
            _ => None,
        },
        _ => None,
    }
}

fn matrix_wire_shape(wire_type: &WireType) -> Option<(Option<&mxx_ir_core::IntExpr>, &MatrixType)> {
    match wire_type {
        WireType::Matrix(matrix) => Some((None, matrix)),
        WireType::IndexedFamily { count, element } => match element.as_ref() {
            WireType::Matrix(matrix) => Some((Some(count), matrix)),
            _ => None,
        },
        _ => None,
    }
}

fn trapdoor_wire_mismatch(expected: &WireType, actual: &WireType) -> TrapdoorContractMismatch {
    let (Some(expected), Some(actual)) =
        (trapdoor_wire_contract_shape(expected), trapdoor_wire_contract_shape(actual))
    else {
        return TrapdoorContractMismatch {
            field: TrapdoorInputContractField::TrapdoorWireType,
            expected: TrapdoorContractValue::WireType(Some(expected.clone())),
            actual: TrapdoorContractValue::WireType(Some(actual.clone())),
        };
    };
    if expected.0 != actual.0 {
        TrapdoorContractMismatch {
            field: TrapdoorInputContractField::FamilyCount,
            expected: TrapdoorContractValue::FamilyCount(expected.0.cloned()),
            actual: TrapdoorContractValue::FamilyCount(actual.0.cloned()),
        }
    } else if expected.1 != actual.1 {
        TrapdoorContractMismatch {
            field: TrapdoorInputContractField::MatrixType,
            expected: TrapdoorContractValue::MatrixType(Some(expected.1.clone())),
            actual: TrapdoorContractValue::MatrixType(Some(actual.1.clone())),
        }
    } else if expected.2 != actual.2 {
        TrapdoorContractMismatch {
            field: TrapdoorInputContractField::Sigma,
            expected: TrapdoorContractValue::Sigma(Some(expected.2.clone())),
            actual: TrapdoorContractValue::Sigma(Some(actual.2.clone())),
        }
    } else if expected.3 != actual.3 {
        TrapdoorContractMismatch {
            field: TrapdoorInputContractField::GadgetBase,
            expected: TrapdoorContractValue::IntegerExpression(Some(expected.3.clone())),
            actual: TrapdoorContractValue::IntegerExpression(Some(actual.3.clone())),
        }
    } else if expected.4 != actual.4 {
        TrapdoorContractMismatch {
            field: TrapdoorInputContractField::DigitCount,
            expected: TrapdoorContractValue::IntegerExpression(Some(expected.4.clone())),
            actual: TrapdoorContractValue::IntegerExpression(Some(actual.4.clone())),
        }
    } else {
        TrapdoorContractMismatch {
            field: TrapdoorInputContractField::PreimageMaxCoefficientBound,
            expected: TrapdoorContractValue::IntegerExpression(Some(expected.5.clone())),
            actual: TrapdoorContractValue::IntegerExpression(Some(actual.5.clone())),
        }
    }
}

fn trapdoor_public_wire_mismatch(
    trapdoor: &WireType,
    public: &WireType,
) -> Option<TrapdoorContractMismatch> {
    let Some((trapdoor_count, trapdoor_matrix)) = trapdoor_wire_shape(trapdoor) else {
        return Some(TrapdoorContractMismatch {
            field: TrapdoorInputContractField::TrapdoorWireType,
            expected: TrapdoorContractValue::ContractKind(TrapdoorContractKind::Other),
            actual: TrapdoorContractValue::ContractKind(TrapdoorContractKind::Other),
        });
    };
    let Some((public_count, public_matrix)) = matrix_wire_shape(public) else {
        return Some(TrapdoorContractMismatch {
            field: TrapdoorInputContractField::PublicInputKind,
            expected: TrapdoorContractValue::ContractKind(if trapdoor_count.is_some() {
                TrapdoorContractKind::FamilyMatrixExact
            } else {
                TrapdoorContractKind::MatrixExact
            }),
            actual: TrapdoorContractValue::ContractKind(TrapdoorContractKind::Other),
        });
    };
    if trapdoor_count != public_count {
        Some(TrapdoorContractMismatch {
            field: TrapdoorInputContractField::FamilyCount,
            expected: TrapdoorContractValue::FamilyCount(trapdoor_count.cloned()),
            actual: TrapdoorContractValue::FamilyCount(public_count.cloned()),
        })
    } else {
        (trapdoor_matrix != public_matrix).then(|| TrapdoorContractMismatch {
            field: TrapdoorInputContractField::MatrixType,
            expected: TrapdoorContractValue::MatrixType(Some(trapdoor_matrix.clone())),
            actual: TrapdoorContractValue::MatrixType(Some(public_matrix.clone())),
        })
    }
}

fn conflicting_trapdoor_input(
    trapdoor_input: ProtocolInputId,
    public_input: ProtocolInputId,
    mismatch: TrapdoorContractMismatch,
) -> OperationalProtocolError {
    OperationalProtocolError::ConflictingTrapdoorInput { trapdoor_input, public_input, mismatch }
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

        let contract = exact_input_contract(
            &wire_type,
            Some(&metadata),
            None,
            &ProtocolInputId::from("matrix-family"),
        )
        .unwrap();

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

        let contract = exact_input_contract(
            &WireType::Matrix(ring.matrix_type((1, 1))),
            None,
            None,
            &ProtocolInputId::from("matrix"),
        )
        .unwrap();

        assert!(matches!(
            contract,
            InputValueContract::MatrixExact {
                canonical_coefficient_exclusive_upper_bound: None,
                is_constant_polynomial: false,
                ..
            }
        ));
    }

    fn trapdoor_wire(matrix: MatrixType) -> WireType {
        WireType::Trapdoor {
            matrix,
            sigma: mxx_ir_core::RealExpr::from(3),
            gadget_base: IntExpr::constant(2),
            digit_count: IntExpr::constant(4),
            preimage_max_coefficient_bound: IntExpr::constant(9),
        }
    }

    #[test]
    fn trapdoor_input_requires_explicit_public_matrix_metadata() {
        let ring = Ring::new(17, 1);
        let input = ProtocolInputId::from("trapdoor");
        assert!(matches!(
            exact_input_contract(&trapdoor_wire(ring.matrix_type((1, 2))), None, None, &input),
            Err(OperationalProtocolError::MissingTrapdoorPublicInput {
                trapdoor_input,
                public_input,
            }) if trapdoor_input == input && public_input.0 == "<unspecified>"
        ));
    }

    #[test]
    fn trapdoor_metadata_must_name_matching_public_matrix_shape() {
        let ring = Ring::new(17, 1);
        let trapdoor = trapdoor_wire(ring.matrix_type((1, 2)));
        let mut contracts = BTreeMap::new();
        contracts.insert("public".to_owned(), (WireType::Matrix(ring.matrix_type((1, 3))), vec![]));
        contracts.insert("trapdoor".to_owned(), (trapdoor, vec![]));
        let metadata = BTreeMap::from([(
            "trapdoor".to_owned(),
            ExactTrapdoorInputMetadata { public_input: ProtocolInputId::from("public") },
        )]);

        assert_eq!(
            validate_trapdoor_metadata(&contracts, &metadata),
            Err(OperationalProtocolError::ConflictingTrapdoorInput {
                trapdoor_input: ProtocolInputId::from("trapdoor"),
                public_input: ProtocolInputId::from("public"),
                mismatch: TrapdoorContractMismatch {
                    field: TrapdoorInputContractField::MatrixType,
                    expected: TrapdoorContractValue::MatrixType(Some(ring.matrix_type((1, 2)),)),
                    actual: TrapdoorContractValue::MatrixType(Some(ring.matrix_type((1, 3)))),
                },
            })
        );
    }

    #[test]
    fn conflicting_trapdoor_wire_uses_the_typed_contract_owner() {
        let ring = Ring::new(17, 1);
        let expected = trapdoor_wire(ring.matrix_type((1, 2)));
        let actual = WireType::Trapdoor {
            matrix: ring.matrix_type((1, 2)),
            sigma: mxx_ir_core::RealExpr::from(3),
            gadget_base: IntExpr::constant(2),
            digit_count: IntExpr::constant(4),
            preimage_max_coefficient_bound: IntExpr::constant(10),
        };

        assert_eq!(
            conflicting_trapdoor_input(
                ProtocolInputId::from("trapdoor"),
                ProtocolInputId::from("public"),
                trapdoor_wire_mismatch(&expected, &actual),
            ),
            OperationalProtocolError::ConflictingTrapdoorInput {
                trapdoor_input: ProtocolInputId::from("trapdoor"),
                public_input: ProtocolInputId::from("public"),
                mismatch: TrapdoorContractMismatch {
                    field: TrapdoorInputContractField::PreimageMaxCoefficientBound,
                    expected: TrapdoorContractValue::IntegerExpression(Some(IntExpr::constant(9))),
                    actual: TrapdoorContractValue::IntegerExpression(Some(IntExpr::constant(10))),
                },
            }
        );
    }

    #[test]
    fn flat_trapdoor_family_contract_keeps_public_input_identity() {
        let ring = Ring::new(17, 1);
        let input = ProtocolInputId::from("trapdoors");
        let metadata =
            ExactTrapdoorInputMetadata { public_input: ProtocolInputId::from("publics") };
        let contract = exact_input_contract(
            &WireType::IndexedFamily {
                count: IntExpr::constant(3),
                element: Box::new(trapdoor_wire(ring.matrix_type((1, 2)))),
            },
            None,
            Some(&metadata),
            &input,
        )
        .unwrap();
        assert!(matches!(
            contract,
            InputValueContract::Family { count, element }
                if count == IntExpr::constant(3) && matches!(
                    element.as_ref(),
                    InputValueContract::Trapdoor { public_input, .. } if public_input == &metadata.public_input
                )
        ));
    }
}
