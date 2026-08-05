//! Small two-stage reference protocol used to test the shared correctness machinery.

use crate::{
    ArtifactBinding, ArtifactName, ClosedProtocolBundle, ComparatorEndpointBinding, ComparatorSpec,
    EndpointAnchor, EndpointAnchors, EndpointSemanticBinding, EndpointSpecId, InputContract,
    InputContractEntry, InputValueContract, OutputRef, ParameterDecl, ParameterKind, ProtocolDecl,
    ProtocolInputBinding, ProtocolInputDestination, ProtocolInputId, ProtocolPreconditionSpec,
    ProtocolStage, StageId, StageInputName, Workflow,
};
use mxx_dsl::{DslContext, IdealSpec, Ring, SemanticAnchor};
use mxx_ir_core::{
    IntExpr, RealExpr,
    artifact::{ArtifactConfidentiality, ProductionId, SpecHash},
};

pub const PROTOCOL_NAME: &str = "toy-example";
pub const DECODED_ENDPOINT: &str = "decoded-endpoint";
pub const PROTOCOL_SOURCE_PATHS: &[&str] = &[
    "crates/correctness/Cargo.toml",
    "crates/correctness/examples/emit_correctness.rs",
    "crates/correctness/src",
    "crates/dsl/Cargo.toml",
    "crates/dsl/src",
    "crates/ir-core/Cargo.toml",
    "crates/ir-core/src",
];

pub fn protocol() -> ProtocolDecl {
    let ring = Ring::new(256, 1);
    let message = ring.bool_input("message");
    let encoded = message
        .clone()
        .to_int()
        .select(vec![ring.zero((1, 1)), ring.polynomial([IntExpr::constant(128)])])
        .expect("two equally typed encoding branches");
    let ciphertext = encoded +
        ring.gaussian((1, 1), RealExpr::from_integer(1), IntExpr::Var("cutoff".to_owned()));
    let encrypt = DslContext::new("toy-example-encrypt")
        .int_parameter("cutoff")
        .public_output("ciphertext", ciphertext)
        .expect("unique output")
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

    let ideal = IdealSpec::new(
        DslContext::new("toy-example-ideal")
            .int_parameter("cutoff")
            .bool_output("result", ring.bool_input("message"))
            .expect("unique output")
            .build()
            .expect("toy ideal graph"),
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
                        bindings: Vec::new(),
                    },
                    ProtocolStage {
                        id: StageId("decrypt".to_owned()),
                        graph: decrypt.graph,
                        semantic_anchors: decrypt.anchors,
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
            Err(crate::BundleValidationError::MissingComparatorConnection)
        );
    }
}
