#[cfg(test)]
mod tests {
    use crate::{
        Bool, DslContext, Family, IdealSpec, Int, PurePredicateSpec, Ring, SemanticAnchor,
    };
    use mxx_ir_core::{IntExpr, node::IndexRange, protocol::*};

    fn threshold_family_bundle(decoder_uses_residual_lane: bool) -> ClosedProtocolBundle {
        let stage_id = StageId("threshold-family-stage".to_owned());
        let ring = Ring::new(17, 1);
        let residuals =
            Family::pack(vec![ring.zero((1, 1)), ring.zero((1, 1))]).expect("residual family");
        let decoder_source =
            if decoder_uses_residual_lane { residuals.get_static(0) } else { ring.zero((1, 1)) };
        let decoder = decoder_source
            .slice(Some(IndexRange { start: 0.into(), end: 1.into() }), None)
            .threshold_decode_bools(IntExpr::constant(17), 1)
            .into_iter()
            .next()
            .expect("decoder output")
            .semantic_anchor("threshold-family.result")
            .expect("decoder anchor");
        let stage = DslContext::new("threshold-family-stage")
            .family_output("residual", residuals)
            .expect("residual output")
            .bool_output("decoded", decoder)
            .expect("decoded output")
            .build()
            .expect("threshold-family graph");
        let decoder_node = stage.graph.outputs()["decoded"].value.node;
        let ideal = IdealSpec::new(
            DslContext::new("threshold-family-ideal")
                .bool_output("ideal", Bool::constant(false))
                .expect("ideal output")
                .build()
                .expect("ideal graph")
                .graph,
        )
        .expect("pure ideal");
        let endpoint = EndpointSpecId::ToyThresholdDecode;

        ClosedProtocolBundle {
            workflow: Workflow {
                stages: vec![ProtocolStage {
                    id: stage_id.clone(),
                    graph: stage.graph,
                    semantic_anchors: stage.anchors,
                    derivation_attachments: stage.derivation_attachments,
                    bindings: Vec::new(),
                }],
                entrypoint: stage_id.clone(),
            },
            ideal,
            requirements: Vec::new(),
            comparator: ComparatorSpec::Equality {
                endpoints: vec![ComparatorEndpointBinding {
                    endpoint,
                    actual_input: "decoded".to_owned(),
                    ideal_input: "ideal".to_owned(),
                    result_output: "failure".to_owned(),
                    failure_value: true,
                }],
            },
            endpoints: EndpointAnchors {
                entries: vec![EndpointAnchor {
                    spec: endpoint,
                    stage: stage_id.clone(),
                    semantic_anchor: "threshold-family.result".to_owned(),
                    semantics: EndpointSemanticBinding::ThresholdDecode,
                    workflow_output: OutputRef {
                        stage: stage_id.clone(),
                        output: "decoded".to_owned(),
                    },
                    ideal_output: "ideal".to_owned(),
                }],
            },
            operational_decoder_targets: vec![OperationalDecoderTarget {
                target_id: "threshold-family".to_owned(),
                residual_stage: stage_id.clone(),
                residual_output: "residual".to_owned(),
                decoder_stage: stage_id,
                decoder_node,
                kind: OperationalDecoderKind::ThresholdDecode {
                    plaintext_modulus: IntExpr::constant(17),
                },
            }],
            endpoint_specs: vec![endpoint],
            input_contract: InputContract::default(),
            input_bindings: Vec::new(),
            precondition_spec: ProtocolPreconditionSpec::default(),
        }
    }

    fn valid_bundle() -> ClosedProtocolBundle {
        let ring = Ring::new(17, 1);
        let stage_value =
            ring.bool_input("message").semantic_anchor("decoded-result").expect("semantic anchor");
        let stage = DslContext::new("stage")
            .bool_output("result", stage_value)
            .expect("stage output")
            .build()
            .expect("stage graph");
        let residual = ring
            .input("residual", (1, 1))
            .semantic_anchor("interval.residual")
            .expect("residual anchor")
            .semantic_anchor("interval.carrier")
            .expect("carrier anchor");
        let coefficient = residual.clone().extract_coefficient(0);
        let quarter = Int::evaluate(IntExpr::RoundDiv(
            Box::new(IntExpr::Sub(Box::new(IntExpr::constant(17)), Box::new(IntExpr::constant(2)))),
            Box::new(IntExpr::constant(4)),
        ));
        let decoded = quarter
            .clone()
            .less_equal(coefficient.clone())
            .to_int()
            .add(coefficient.less_equal(quarter.mul(Int::constant(3))).to_int())
            .equal(Int::constant(2))
            .semantic_anchor("interval.result")
            .expect("decoder anchor");
        let decoder_stage = DslContext::new("decoder-stage")
            .output("residual", residual)
            .expect("residual output")
            .bool_output("decoded", decoded)
            .expect("decoded output")
            .build()
            .expect("decoder graph");
        let decoder_node = decoder_stage.graph.outputs()["decoded"].value.node;
        let ideal = IdealSpec::new(
            DslContext::new("ideal")
                .bool_output("result", ring.bool_input("message"))
                .expect("ideal output")
                .build()
                .expect("ideal graph")
                .graph,
        )
        .expect("pure ideal");
        let requirement = PurePredicateSpec::new(
            DslContext::new("requirement")
                .bool_output("valid", ring.bool_input("message"))
                .expect("requirement output")
                .build()
                .expect("requirement graph")
                .graph,
        )
        .expect("pure predicate");
        let comparator_actual = ring.bool_input("actual").to_int();
        let comparator_ideal = ring.bool_input("ideal").to_int();
        let comparator = IdealSpec::new(
            DslContext::new("comparator")
                .bool_output("failure", comparator_actual.equal(comparator_ideal))
                .expect("comparator output")
                .build()
                .expect("comparator graph")
                .graph,
        )
        .expect("pure comparator");
        let input = ProtocolInputId::from("message");
        let residual_input = ProtocolInputId::from("residual");
        let decoder_stage_id = StageId("decoder-stage".to_owned());
        let interval_endpoint = EndpointSpecId::DiamondBooleanInterval;

        ClosedProtocolBundle {
            workflow: Workflow {
                stages: vec![
                    ProtocolStage {
                        id: StageId("stage".to_owned()),
                        graph: stage.graph,
                        semantic_anchors: stage.anchors,
                        derivation_attachments: stage.derivation_attachments,
                        bindings: Vec::new(),
                    },
                    ProtocolStage {
                        id: decoder_stage_id.clone(),
                        graph: decoder_stage.graph,
                        semantic_anchors: decoder_stage.anchors,
                        derivation_attachments: decoder_stage.derivation_attachments,
                        bindings: Vec::new(),
                    },
                ],
                entrypoint: StageId("stage".to_owned()),
            },
            ideal,
            requirements: vec![requirement],
            comparator: ComparatorSpec::EqualityAfterMap {
                program: comparator,
                endpoints: vec![ComparatorEndpointBinding {
                    endpoint: interval_endpoint,
                    actual_input: "actual".to_owned(),
                    ideal_input: "ideal".to_owned(),
                    result_output: "failure".to_owned(),
                    failure_value: false,
                }],
            },
            endpoints: EndpointAnchors {
                entries: vec![EndpointAnchor {
                    spec: interval_endpoint,
                    stage: decoder_stage_id.clone(),
                    semantic_anchor: "interval.result".to_owned(),
                    semantics: EndpointSemanticBinding::DiamondBoolean {
                        residual_stage: decoder_stage_id.clone(),
                        residual_anchor: "interval.residual".to_owned(),
                        carrier_stage: decoder_stage_id.clone(),
                        carrier_anchor: "interval.carrier".to_owned(),
                        message: input.clone(),
                    },
                    workflow_output: OutputRef {
                        stage: decoder_stage_id.clone(),
                        output: "decoded".to_owned(),
                    },
                    ideal_output: "result".to_owned(),
                }],
            },
            operational_decoder_targets: vec![OperationalDecoderTarget {
                target_id: "interval".to_owned(),
                residual_stage: decoder_stage_id.clone(),
                residual_output: "residual".to_owned(),
                decoder_stage: decoder_stage_id.clone(),
                decoder_node,
                kind: OperationalDecoderKind::BooleanInterval,
            }],
            endpoint_specs: vec![interval_endpoint],
            input_contract: InputContract {
                inputs: vec![
                    InputContractEntry {
                        id: input.clone(),
                        name: "message".to_owned(),
                        value: InputValueContract::Boolean,
                    },
                    InputContractEntry {
                        id: residual_input.clone(),
                        name: "residual".to_owned(),
                        value: InputValueContract::MatrixExact {
                            matrix_type: ring.matrix_type((1, 1)),
                            canonical_coefficient_exclusive_upper_bound: None,
                            is_constant_polynomial: false,
                        },
                    },
                ],
            },
            input_bindings: vec![
                ProtocolInputBinding {
                    input,
                    destinations: vec![
                        ProtocolInputDestination::WorkflowStage {
                            stage: StageId("stage".to_owned()),
                            input: StageInputName("message".to_owned()),
                        },
                        ProtocolInputDestination::Requirement {
                            requirement: 0,
                            input: "message".to_owned(),
                        },
                        ProtocolInputDestination::Ideal { input: "message".to_owned() },
                    ],
                },
                ProtocolInputBinding {
                    input: residual_input,
                    destinations: vec![ProtocolInputDestination::WorkflowStage {
                        stage: decoder_stage_id,
                        input: StageInputName("residual".to_owned()),
                    }],
                },
            ],
            precondition_spec: ProtocolPreconditionSpec {
                requirement_outputs: vec!["valid".to_owned()],
            },
        }
    }

    fn boolean_interval_bundle(decoder_modulus: IntExpr) -> ClosedProtocolBundle {
        let stage_id = StageId("interval-stage".to_owned());
        let ring = Ring::new(17, 1);
        let matrix_type = ring.matrix_type((1, 1));
        let residual = ring
            .input("residual", (1, 1))
            .semantic_anchor("interval.residual")
            .expect("residual anchor")
            .semantic_anchor("interval.carrier")
            .expect("carrier anchor");
        let coefficient = residual.clone().extract_coefficient(0);
        let quarter = Int::evaluate(IntExpr::RoundDiv(
            Box::new(IntExpr::Sub(Box::new(decoder_modulus), Box::new(IntExpr::constant(2)))),
            Box::new(IntExpr::constant(4)),
        ));
        let decoded = quarter
            .clone()
            .less_equal(coefficient.clone())
            .to_int()
            .add(coefficient.less_equal(quarter.mul(Int::constant(3))).to_int())
            .equal(Int::constant(2))
            .semantic_anchor("interval.result")
            .expect("decoder anchor");
        let stage = DslContext::new("interval-stage")
            .output("residual", residual)
            .expect("residual output")
            .bool_output("decoded", decoded)
            .expect("decoded output")
            .build()
            .expect("interval graph");
        let decoder_node = stage.graph.outputs()["decoded"].value.node;
        let ideal = IdealSpec::new(
            DslContext::new("interval-ideal")
                .bool_output("result", ring.bool_input("message"))
                .expect("ideal output")
                .build()
                .expect("ideal graph")
                .graph,
        )
        .expect("pure ideal");
        let residual_input = ProtocolInputId::from("residual");
        let message_input = ProtocolInputId::from("message");
        let endpoint = EndpointSpecId::DiamondBooleanInterval;

        ClosedProtocolBundle {
            workflow: Workflow {
                stages: vec![ProtocolStage {
                    id: stage_id.clone(),
                    graph: stage.graph,
                    semantic_anchors: stage.anchors,
                    derivation_attachments: stage.derivation_attachments,
                    bindings: Vec::new(),
                }],
                entrypoint: stage_id.clone(),
            },
            ideal,
            requirements: Vec::new(),
            comparator: ComparatorSpec::Equality {
                endpoints: vec![ComparatorEndpointBinding {
                    endpoint,
                    actual_input: "decoded".to_owned(),
                    ideal_input: "result".to_owned(),
                    result_output: "failure".to_owned(),
                    failure_value: true,
                }],
            },
            endpoints: EndpointAnchors {
                entries: vec![EndpointAnchor {
                    spec: endpoint,
                    stage: stage_id.clone(),
                    semantic_anchor: "interval.result".to_owned(),
                    semantics: EndpointSemanticBinding::DiamondBoolean {
                        residual_stage: stage_id.clone(),
                        residual_anchor: "interval.residual".to_owned(),
                        carrier_stage: stage_id.clone(),
                        carrier_anchor: "interval.carrier".to_owned(),
                        message: message_input.clone(),
                    },
                    workflow_output: OutputRef {
                        stage: stage_id.clone(),
                        output: "decoded".to_owned(),
                    },
                    ideal_output: "result".to_owned(),
                }],
            },
            operational_decoder_targets: vec![OperationalDecoderTarget {
                target_id: "boolean-interval".to_owned(),
                residual_stage: stage_id.clone(),
                residual_output: "residual".to_owned(),
                decoder_stage: stage_id.clone(),
                decoder_node,
                kind: OperationalDecoderKind::BooleanInterval,
            }],
            endpoint_specs: vec![endpoint],
            input_contract: InputContract {
                inputs: vec![
                    InputContractEntry {
                        id: residual_input.clone(),
                        name: "residual".to_owned(),
                        value: InputValueContract::MatrixExact {
                            matrix_type,
                            canonical_coefficient_exclusive_upper_bound: None,
                            is_constant_polynomial: false,
                        },
                    },
                    InputContractEntry {
                        id: message_input.clone(),
                        name: "message".to_owned(),
                        value: InputValueContract::Boolean,
                    },
                ],
            },
            input_bindings: vec![
                ProtocolInputBinding {
                    input: residual_input,
                    destinations: vec![ProtocolInputDestination::WorkflowStage {
                        stage: stage_id,
                        input: StageInputName("residual".to_owned()),
                    }],
                },
                ProtocolInputBinding {
                    input: message_input,
                    destinations: vec![ProtocolInputDestination::Ideal {
                        input: "message".to_owned(),
                    }],
                },
            ],
            precondition_spec: ProtocolPreconditionSpec::default(),
        }
    }

    #[test]
    fn valid_closed_bundle_has_total_input_and_endpoint_wiring() {
        assert_eq!(valid_bundle().validate(), Ok(()));
    }

    #[test]
    fn empty_operational_decoder_target_registry_is_rejected() {
        let mut bundle = valid_bundle();
        bundle.operational_decoder_targets.clear();
        assert_eq!(
            bundle.validate(),
            Err(BundleValidationError::EmptyOperationalDecoderTargetRegistry)
        );
    }

    #[test]
    fn boolean_interval_target_rejects_a_forged_interior_modulus() {
        assert_eq!(boolean_interval_bundle(IntExpr::constant(17)).validate(), Ok(()));
        assert_eq!(
            boolean_interval_bundle(IntExpr::constant(19)).validate(),
            Err(BundleValidationError::InvalidOperationalDecoderTarget)
        );
    }

    #[test]
    fn threshold_family_target_requires_decoder_provenance_from_residual_output() {
        assert_eq!(threshold_family_bundle(true).validate(), Ok(()));
        assert_eq!(
            threshold_family_bundle(false).validate(),
            Err(BundleValidationError::InvalidOperationalDecoderTarget)
        );
    }

    #[test]
    fn duplicate_logical_input_id_is_rejected() {
        let mut bundle = valid_bundle();
        bundle.input_contract.inputs.push(bundle.input_contract.inputs[0].clone());
        assert_eq!(bundle.validate(), Err(BundleValidationError::DuplicateInputId));
    }

    #[test]
    fn missing_logical_input_binding_is_rejected() {
        let mut bundle = valid_bundle();
        bundle.input_bindings.clear();
        assert_eq!(bundle.validate(), Err(BundleValidationError::MissingOrDuplicateInputBinding));
    }

    #[test]
    fn destination_type_must_match_the_logical_contract() {
        let mut bundle = valid_bundle();
        bundle.input_contract.inputs[0].value = InputValueContract::IntegerRange {
            lower: IntExpr::constant(0),
            upper: IntExpr::constant(1),
        };
        assert_eq!(bundle.validate(), Err(BundleValidationError::InputContractTypeMismatch));
    }

    #[test]
    fn explicit_large_matrix_contract_matches_only_its_declared_matrix_type() {
        let mut bundle = valid_bundle();
        let matrix_type = match &bundle.input_contract.inputs[1].value {
            InputValueContract::MatrixExact { matrix_type, .. } => matrix_type.clone(),
            contract => panic!("fixture residual must be matrix exact, got {contract:?}"),
        };
        bundle.input_contract.inputs[1].value = InputValueContract::MatrixLarge { matrix_type };
        assert_eq!(bundle.validate(), Ok(()));

        bundle.input_contract.inputs[1].value =
            InputValueContract::MatrixLarge { matrix_type: Ring::new(19, 1).matrix_type((1, 1)) };
        assert_eq!(bundle.validate(), Err(BundleValidationError::InputContractTypeMismatch));
    }

    #[test]
    fn unbound_destination_is_rejected() {
        let mut bundle = valid_bundle();
        bundle.input_bindings[0].destinations.pop();
        assert_eq!(bundle.validate(), Err(BundleValidationError::UnboundInputDestination));
    }

    #[test]
    fn duplicate_destination_across_inputs_is_rejected() {
        let mut bundle = valid_bundle();
        let destination = bundle.input_bindings[0].destinations[0].clone();
        bundle.input_contract.inputs.push(InputContractEntry {
            id: ProtocolInputId::from("other"),
            name: "other".to_owned(),
            value: InputValueContract::Boolean,
        });
        bundle.input_bindings.push(ProtocolInputBinding {
            input: ProtocolInputId::from("other"),
            destinations: vec![destination],
        });
        assert_eq!(bundle.validate(), Err(BundleValidationError::DuplicateInputDestination));
    }

    #[test]
    fn endpoint_cardinality_is_rejected_before_analysis() {
        let mut bundle = valid_bundle();
        bundle.endpoint_specs.push(EndpointSpecId::ToyThresholdDecode);
        assert_eq!(bundle.validate(), Err(BundleValidationError::EndpointCardinalityMismatch));
    }

    #[test]
    fn structural_validation_does_not_claim_endpoint_soundness() {
        let mut bundle = valid_bundle();
        bundle.endpoint_specs.push(EndpointSpecId::ToyThresholdDecode);
        bundle.endpoints.entries.push(EndpointAnchor {
            spec: EndpointSpecId::ToyThresholdDecode,
            stage: StageId("stage".to_owned()),
            semantic_anchor: "decoded-result".to_owned(),
            semantics: EndpointSemanticBinding::ThresholdDecode,
            workflow_output: OutputRef {
                stage: StageId("stage".to_owned()),
                output: "result".to_owned(),
            },
            ideal_output: "result".to_owned(),
        });
        let ComparatorSpec::EqualityAfterMap { endpoints, .. } = &mut bundle.comparator else {
            unreachable!("fixture uses mapped equality")
        };
        endpoints.push(ComparatorEndpointBinding {
            endpoint: EndpointSpecId::ToyThresholdDecode,
            actual_input: "actual".to_owned(),
            ideal_input: "ideal".to_owned(),
            result_output: "failure".to_owned(),
            failure_value: false,
        });
        assert_eq!(bundle.validate(), Ok(()));
    }
}
