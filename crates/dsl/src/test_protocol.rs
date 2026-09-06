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
    fn test_generic_threshold_protocol_export() {
        use mxx_ir_core::{
            ParamEnv,
            artifact::export_validated_manifest,
            lean::{
                claim::{ClaimBackend, ClaimSemantics},
                protocol::export_claim,
            },
            validate,
        };
        use std::{collections::BTreeMap, fs, path::Path};

        let declaration = protocol();
        let bindings = ParamEnv {
            integers: BTreeMap::from([("cutoff".into(), 3.into())]),
            ..ParamEnv::default()
        };
        let production = ProductionId { spec_hash: SpecHash([0; 32]), execution_nonce: [0; 32] };
        let producer = validate(&declaration.stages()[0].graph, &bindings).unwrap();
        let manifest = export_validated_manifest(production.clone(), &producer).unwrap();
        let manifests = BTreeMap::from([(production, manifest)]);
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/lean_ir_fixtures/threshold_protocol");
        fs::create_dir_all(&directory).unwrap();
        fs::write(directory.join("ThresholdFixture.lean"),
            "import MxxRuntime\nnamespace ThresholdFixture\ndef zeroCenter (_ : Nat) (_ : Bool) : Nat := 0\ndef decoderRadius (q : Nat) : Nat := q / 4\nend ThresholdFixture\n",
        ).unwrap();
        // The residual graph is already ciphertext minus the encoded message. Subtracting a
        // Boolean message center again would change the proposition; its center must be zero.
        let semantics = ClaimSemantics {
            imports: &["ThresholdFixture"],
            hash_model_type: "MxxRuntime.HashModel",
            centered_lift: "Mxx.Primitives.centeredLift",
            message_center: "ThresholdFixture.zeroCenter",
            decoder_radius: "ThresholdFixture.decoderRadius",
        };
        export_claim(
            &declaration,
            &bindings,
            &ClaimBackend {
                module_name: "ThresholdFixture",
                context_name: "ThresholdFixture.backend",
                layouts: &[],
            },
            &semantics,
            &manifests,
            &directory,
        )
        .expect("generic export accepts the validated threshold decoder");

        let source = fs::read_to_string(directory.join("Claim.lean")).unwrap();
        let (premises, conclusion) = source.split_once("def CorrectnessClaim").unwrap();
        assert!(premises.contains("Stage_encrypt.generatedRoot"));
        assert!(premises.contains("Stage_decrypt.generatedRoot"));
        assert!(premises.contains("Ideal.generatedRoot"));
        assert_eq!(premises.matches("(external.input_0)").count(), 2);
        assert!(!premises.contains(".natAbs <"));
        assert!(premises.contains("ThresholdFixture.zeroCenter 256"));
        assert!(!source.contains("MxxWe"));
        assert!(conclusion.contains("Runs hashModel external execution →"));
        assert!(
            conclusion.contains(
                "(observedResidual execution).natAbs < ThresholdFixture.decoderRadius 256"
            )
        );
        assert!(conclusion.contains("execution.«stage_1».2.1 = execution.«ideal»"));
        let decoder = fs::read_to_string(directory.join("Stage_decrypt.lean")).unwrap();
        assert!(decoder.contains("MxxRuntime.thresholdDecode 2 1 0"));
        assert!(decoder.contains("decide (w_1_0_decoded ≠ 0)"));
    }

    #[test]
    fn test_threshold_export_preserves_each_port_and_symbolic_modulus() {
        use mxx_ir_core::{
            ParamEnv,
            lean::{ExportOptions, export},
        };
        use std::{collections::BTreeMap, fs, path::Path};
        let ring = Ring::new(256, 2);
        let input = ring.input("ciphertext", (1, 1));
        let modulus = IntExpr::Var("plaintext_modulus".into());
        let integers = input.clone().threshold_decode_ints(modulus.clone(), 2);
        let booleans = input.threshold_decode_bools(modulus, 2);
        let mut context = DslContext::new("threshold-ports").int_parameter("plaintext_modulus");
        for (index, value) in integers.into_iter().enumerate() {
            context = context.int_output(format!("integer_{index}"), value).unwrap();
        }
        for (index, value) in booleans.into_iter().enumerate() {
            context = context.bool_output(format!("boolean_{index}"), value).unwrap();
        }
        let graph = context.build().unwrap();
        let bindings = ParamEnv {
            integers: BTreeMap::from([("plaintext_modulus".into(), 3.into())]),
            ..ParamEnv::default()
        };
        let validated = graph.validate(&bindings).unwrap();
        let artifact = export(
            &validated,
            &ExportOptions {
                namespace: "ThresholdPorts".into(),
                module_name: "ThresholdPorts".into(),
                ..ExportOptions::default()
            },
        )
        .unwrap();
        assert_eq!(artifact.source.matches("MxxRuntime.thresholdDecode").count(), 4);
        assert_eq!(artifact.source.matches("decide (").count(), 2);
        assert_eq!(artifact.source.matches("params.«plaintext_modulus» 2 0").count(), 2);
        assert_eq!(artifact.source.matches("params.«plaintext_modulus» 2 1").count(), 2);
        assert!(artifact.root.outputs["integer_0"].lean_type == "Int");
        assert!(artifact.root.outputs["boolean_1"].lean_type == "Bool");
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/lean_ir_fixtures/threshold_ports");
        fs::create_dir_all(&directory).unwrap();
        fs::write(directory.join("ThresholdPorts.lean"), artifact.source).unwrap();
    }

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
