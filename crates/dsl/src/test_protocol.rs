//! Small two-stage reference protocol used to test the shared correctness machinery.

use crate::{DslContext, IdealSpec, Ring, SemanticAnchor};
use mxx_ir_core::{
    IntExpr, RealExpr,
    artifact::{ArtifactConfidentiality, ProductionId, SpecHash},
    protocol::{
        ArtifactBinding, ArtifactName, ClosedProtocolBundle, ComparatorEndpointBinding,
        ComparatorSpec, EndpointAnchor, EndpointAnchors, EndpointSemanticBinding, EndpointSpecId,
        InputContract, InputContractEntry, InputValueContract, OperationalDecoderKind,
        OperationalDecoderTarget, OutputRef, ParameterDecl, ParameterKind, ProtocolDecl,
        ProtocolInputBinding, ProtocolInputDestination, ProtocolInputId, ProtocolPreconditionSpec,
        ProtocolStage, StageId, StageInputName, Workflow,
    },
};

pub const DECODED_ENDPOINT: &str = "decoded-endpoint";
pub const RESIDUAL_ANCHOR: &str = "toy.decoder.residual";
pub fn protocol() -> ProtocolDecl {
    let ring = Ring::new(256, 1);
    let message = ring.bool_input("message");
    let selector = message.clone().to_int();
    let zero = ring.zero((1, 1));
    let carrier = ring.polynomial([IntExpr::constant(128)]);
    let encoded = selector
        .select(vec![zero.clone(), carrier.clone()])
        .expect("two equally typed encoding branches");
    let ciphertext = encoded.clone() +
        ring.gaussian((1, 1), RealExpr::from_integer(1), IntExpr::Var("cutoff".to_owned()));
    let residual = (ciphertext.clone() - encoded)
        .semantic_anchor(RESIDUAL_ANCHOR)
        .expect("Toy decoder residual anchor");
    let encrypt = DslContext::new("toy-example-encrypt")
        .int_parameter("cutoff")
        .public_output("ciphertext", ciphertext)
        .expect("unique output")
        .private_output("operational-residual", residual)
        .expect("unique operational residual output")
        .build()
        .expect("toy encryption graph");

    let placeholder = ProductionId { spec_hash: SpecHash([0; 32]), execution_nonce: [0; 32] };
    let ciphertext =
        ring.artifact_input(placeholder, "ciphertext", (1, 1), ArtifactConfidentiality::Public);
    let decoded = ciphertext
        .threshold_decode_bools(IntExpr::constant(2), 1)
        .into_iter()
        .next()
        .expect("one decoded bit")
        .semantic_anchor(DECODED_ENDPOINT)
        .expect("decoded endpoint label");
    let decrypt = DslContext::new("toy-example-decrypt")
        .int_parameter("cutoff")
        .bool_output("decoded", decoded)
        .expect("unique output")
        .build()
        .expect("toy decryption graph");
    let decoder_node = decrypt.graph.outputs()["decoded"].value.node;

    let ideal = IdealSpec::new(
        DslContext::new("toy-example-ideal")
            .int_parameter("cutoff")
            .bool_output("result", ring.bool_input("message"))
            .expect("unique output")
            .build()
            .expect("toy ideal graph")
            .graph,
    )
    .expect("sampler-free ideal");

    let message_id = ProtocolInputId::from("message");
    let endpoint = EndpointSpecId::ToyThresholdDecode;
    ProtocolDecl::new(ProtocolDecl {
        params: vec![ParameterDecl { name: "cutoff".to_owned(), kind: ParameterKind::Dimension }],
        bundle: ClosedProtocolBundle {
            workflow: Workflow {
                stages: vec![
                    ProtocolStage {
                        id: StageId("encrypt".to_owned()),
                        graph: encrypt.graph,
                        semantic_anchors: encrypt.anchors,
                        derivation_attachments: encrypt.derivation_attachments,
                        bindings: Vec::new(),
                    },
                    ProtocolStage {
                        id: StageId("decrypt".to_owned()),
                        graph: decrypt.graph,
                        semantic_anchors: decrypt.anchors,
                        derivation_attachments: decrypt.derivation_attachments,
                        bindings: vec![ArtifactBinding {
                            consumer_input: StageInputName("ciphertext".to_owned()),
                            producer_stage: StageId("encrypt".to_owned()),
                            producer_output: ArtifactName("ciphertext".to_owned()),
                        }],
                    },
                ],
                entrypoint: StageId("decrypt".to_owned()),
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
                    stage: StageId("decrypt".to_owned()),
                    semantic_anchor: DECODED_ENDPOINT.to_owned(),
                    semantics: EndpointSemanticBinding::ThresholdDecode,
                    workflow_output: OutputRef {
                        stage: StageId("decrypt".to_owned()),
                        output: "decoded".to_owned(),
                    },
                    ideal_output: "result".to_owned(),
                }],
            },
            operational_decoder_targets: vec![OperationalDecoderTarget {
                target_id: "toy-threshold".to_owned(),
                residual_stage: StageId("encrypt".to_owned()),
                residual_output: "operational-residual".to_owned(),
                decoder_stage: StageId("decrypt".to_owned()),
                decoder_node,
                kind: OperationalDecoderKind::ThresholdDecode {
                    plaintext_modulus: IntExpr::constant(2),
                },
            }],
            endpoint_specs: vec![endpoint],
            input_contract: InputContract {
                inputs: vec![InputContractEntry {
                    id: message_id.clone(),
                    name: "message".to_owned(),
                    value: InputValueContract::Boolean,
                }],
            },
            input_bindings: vec![ProtocolInputBinding {
                input: message_id,
                destinations: vec![
                    ProtocolInputDestination::WorkflowStage {
                        stage: StageId("encrypt".to_owned()),
                        input: StageInputName("message".to_owned()),
                    },
                    ProtocolInputDestination::Ideal { input: "message".to_owned() },
                ],
            }],
            precondition_spec: ProtocolPreconditionSpec::default(),
        },
    })
    .expect("toy example protocol is valid")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toy_protocol_is_a_closed_bundle_with_a_decoded_endpoint() {
        let protocol = protocol();
        assert_eq!(protocol.bundle.endpoint_specs, vec![EndpointSpecId::ToyThresholdDecode]);
        assert_eq!(protocol.bundle.endpoints.entries[0].semantic_anchor, DECODED_ENDPOINT);
        assert!(matches!(
            protocol.bundle.input_contract.inputs[0].value,
            InputValueContract::Boolean
        ));
        assert!(matches!(protocol.bundle.comparator, ComparatorSpec::Equality { .. }));
    }

    #[test]
    fn direct_comparator_wiring_must_name_the_registered_endpoint_outputs() {
        let mut protocol = protocol();
        let ComparatorSpec::Equality { endpoints } = &mut protocol.bundle.comparator else {
            unreachable!("toy uses direct equality")
        };
        endpoints[0].actual_input = "unrelated".to_owned();
        assert_eq!(
            protocol.bundle.validate(),
            Err(mxx_ir_core::protocol::BundleValidationError::MissingComparatorConnection)
        );
    }

    #[test]
    fn operational_target_plaintext_modulus_must_match_the_executable_decoder() {
        let mut protocol = protocol();
        let OperationalDecoderKind::ThresholdDecode { plaintext_modulus } =
            &mut protocol.bundle.operational_decoder_targets[0].kind
        else {
            unreachable!("toy target is threshold decoding")
        };
        *plaintext_modulus = IntExpr::constant(3);
        assert_eq!(
            protocol.bundle.validate(),
            Err(mxx_ir_core::protocol::BundleValidationError::InvalidOperationalDecoderTarget)
        );
    }
}
