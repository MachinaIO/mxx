//! Small two-stage reference protocol used to test the shared correctness machinery.

use crate::{DslContext, IdealSpec, Ring};
use mxx_ir_core::{
    IntExpr, RealExpr,
    artifact::{ArtifactAvailability, ProductionId, SpecHash},
    protocol::{
        ArtifactBinding, ArtifactName, ClosedProtocolBundle, ComparatorEndpointBinding,
        ComparatorSpec, EndpointBinding, EndpointBindings, EndpointSemanticBinding, EndpointSpecId,
        InputContract, InputContractEntry, InputValueContract, OperationalDecoderKind,
        OperationalDecoderTarget, OutputRef, ParameterDecl, ParameterKind, ProtocolDecl,
        ProtocolInputBinding, ProtocolInputDestination, ProtocolInputId, ProtocolPreconditionSpec,
        ProtocolStage, StageId, StageInputName, Workflow,
    },
};
pub fn protocol() -> ProtocolDecl {
    protocol_with_consumer_availability(ArtifactAvailability::Transferred)
        .expect("toy example protocol is valid")
}

fn protocol_with_consumer_availability(
    consumer_availability: ArtifactAvailability,
) -> Result<ProtocolDecl, mxx_ir_core::protocol::ProtocolError> {
    let ring = Ring::from_crt_moduli(vec![257.into()], 1);
    let message = ring.bool_input("message");
    let selector = message.clone().to_int();
    let zero = ring.zero((1, 1));
    let carrier = ring.polynomial([IntExpr::constant(128)]);
    let encoded = crate::select(selector, vec![zero.clone(), carrier.clone()])
        .expect("two equally typed encoding branches");
    let ciphertext = encoded.clone() +
        ring.gaussian((1, 1), RealExpr::from_integer(1), IntExpr::Var("cutoff".to_owned()));
    let residual = ciphertext.clone() - encoded;
    let encrypt = DslContext::new("toy-example-encrypt").int_parameter("cutoff");
    let encrypt = match consumer_availability {
        ArtifactAvailability::Transferred => encrypt
            .transferred_output("ciphertext", ciphertext)
            .expect("unique output")
            .transferred_output("operational-residual", residual)
            .expect("unique operational residual output"),
        ArtifactAvailability::Cached => encrypt
            .cached_output("ciphertext", ciphertext)
            .expect("unique output")
            .transferred_output("operational-residual", residual)
            .expect("unique operational residual output"),
    }
    .build()
    .expect("toy encryption graph");

    let placeholder = ProductionId { spec_hash: SpecHash([0; 32]), execution_nonce: [0; 32] };
    let ciphertext = ring.artifact_input(placeholder, "ciphertext", (1, 1), consumer_availability);
    let decoded = ciphertext
        .threshold_decode_bools(IntExpr::constant(2), 1)
        .into_iter()
        .next()
        .expect("one decoded bit");
    let decrypt = DslContext::new("toy-example-decrypt")
        .int_parameter("cutoff")
        .output("decoded", decoded)
        .expect("unique output")
        .build()
        .expect("toy decryption graph");

    let ideal = IdealSpec::new(
        DslContext::new("toy-example-ideal")
            .int_parameter("cutoff")
            .output("result", ring.bool_input("message"))
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
                        bindings: Vec::new(),
                    },
                    ProtocolStage {
                        id: StageId("decrypt".to_owned()),
                        graph: decrypt.graph,
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
            endpoints: EndpointBindings {
                entries: vec![EndpointBinding {
                    spec: endpoint,
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
                residual: OutputRef {
                    stage: StageId("encrypt".to_owned()),
                    output: "operational-residual".to_owned(),
                },
                endpoint: EndpointSpecId::ToyThresholdDecode,
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
        let producer =
            validate(&declaration.stages()[0].graph, &bindings, crate::test_resolve_basis).unwrap();
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
            crate::test_resolve_basis,
        )
        .expect("generic export accepts the validated threshold decoder");

        let source = fs::read_to_string(directory.join("Claim.lean")).unwrap();
        let (premises, conclusion) = source.split_once("def CorrectnessClaim").unwrap();
        assert!(premises.contains("Stage_encrypt.generatedRoot"));
        assert!(premises.contains("Stage_decrypt.generatedRoot"));
        assert!(premises.contains("Ideal.generatedRoot"));
        assert_eq!(premises.matches("(external.input_0)").count(), 2);
        assert!(!premises.contains(".natAbs <"));
        assert!(premises.contains("ThresholdFixture.zeroCenter 257"));
        assert!(!source.contains("MxxWe"));
        assert!(conclusion.contains("Runs hashModel external execution →"));
        assert!(
            conclusion.contains(
                "(observedResidual execution).natAbs < ThresholdFixture.decoderRadius 257"
            )
        );
        assert!(conclusion.contains("execution.«stage_1» = execution.«ideal»"));
        let decoder = fs::read_to_string(directory.join("Stage_decrypt.lean")).unwrap();
        assert!(decoder.contains("MxxRuntime.thresholdDecode (2) (1) 0"));
        assert!(decoder.contains("decide (w_1_0_decoded ≠ 0)"));
    }

    #[test]
    fn stage_binding_entry_point_matches_protocol_validation() {
        let declaration = protocol();
        assert!(
            mxx_ir_core::protocol::validate_stage_artifact_bindings(declaration.stages()).is_ok()
        );

        let mut duplicate = declaration.stages().to_vec();
        duplicate.push(duplicate[0].clone());
        assert!(matches!(
            mxx_ir_core::protocol::validate_stage_artifact_bindings(&duplicate),
            Err(mxx_ir_core::protocol::ProtocolError::DuplicateStageId)
        ));

        let consumer_only = vec![declaration.stages()[1].clone()];
        assert!(matches!(
            mxx_ir_core::protocol::validate_stage_artifact_bindings(&consumer_only),
            Err(mxx_ir_core::protocol::ProtocolError::MissingProducerStage)
        ));
    }

    #[test]
    fn artifact_availability_is_checked_against_the_producer_semantics() {
        for (availability, expected) in
            [(ArtifactAvailability::Transferred, Ok(())), (ArtifactAvailability::Cached, Ok(()))]
        {
            let actual = protocol_with_consumer_availability(availability).map(|_| ());
            assert_eq!(actual, expected, "consumer declaration: {availability:?}");
        }
    }

    #[test]
    fn sampled_artifacts_are_transferred_but_public_deterministic_cache_is_cached() {
        use mxx_ir_core::node::NodeKind;

        let ring = Ring::from_crt_moduli(vec![257.into()], 8);
        let sampled = ring.sample_trapdoor(1, 1, 2, 3, 4);
        let sampled_public = sampled.public_matrix();
        let sampled_preimage = sampled.sample_preimage(ring.zero((1, 1)), (5, 1));
        let sampled_graph = DslContext::new("sampled-artifact-roles")
            .transferred_output("public", sampled_public)
            .expect("sampled public output")
            .transferred_trapdoor_output("trapdoor", sampled)
            .expect("sampled trapdoor output")
            .transferred_output("preimage", sampled_preimage)
            .expect("sampled preimage output")
            .build()
            .expect("sampled graph");

        // These values contain fresh randomness and therefore require the
        // producer payload at the consumer boundary.
        for name in ["public", "trapdoor", "preimage"] {
            assert_eq!(
                sampled_graph.graph.outputs()[name].availability,
                Some(ArtifactAvailability::Transferred),
                "sampled artifact {name} must be transferred",
            );
        }

        // A cache declaration is reserved for a public deterministic setup
        // keyed by its production metadata.  It is not a substitute for a
        // missing secret/trapdoor recipe.
        let production = ProductionId { spec_hash: SpecHash([17; 32]), execution_nonce: [19; 32] };
        let cached_producer = DslContext::new("deterministic-cache-producer")
            .cached_output("lut", ring.identity(1))
            .expect("cached producer output")
            .build()
            .expect("cached producer graph");
        assert_eq!(
            cached_producer.graph.outputs()["lut"].availability,
            Some(ArtifactAvailability::Cached)
        );
        let cached = ring.artifact_input(
            production,
            "public-deterministic-lut",
            (1, 1),
            ArtifactAvailability::Cached,
        );
        let cached_graph = DslContext::new("deterministic-cache-input")
            .output("lut", cached)
            .expect("cache output")
            .build()
            .expect("cache graph");
        let cached_input = cached_graph
            .graph
            .root_scope()
            .nodes()
            .iter()
            .find_map(|node| match node.kind() {
                NodeKind::Input { artifact: Some(artifact), .. } => Some(artifact.availability),
                _ => None,
            })
            .expect("cached artifact input");
        assert_eq!(cached_input, ArtifactAvailability::Cached);
    }

    #[test]
    fn test_threshold_export_preserves_each_port_and_symbolic_modulus() {
        use mxx_ir_core::{
            ParamEnv,
            lean::{ExportOptions, export},
        };
        use std::{collections::BTreeMap, fs, path::Path};
        let ring = Ring::from_crt_moduli(vec![257.into()], 2);
        let input = ring.input("ciphertext", (1, 1));
        let modulus = IntExpr::Var("plaintext_modulus".into());
        let integers = input.clone().threshold_decode_ints(modulus.clone(), 2);
        let booleans = input.threshold_decode_bools(modulus, 2);
        let mut context = DslContext::new("threshold-ports").int_parameter("plaintext_modulus");
        for (index, value) in integers.into_iter().enumerate() {
            context = context.output(format!("integer_{index}"), value).unwrap();
        }
        for (index, value) in booleans.into_iter().enumerate() {
            context = context.output(format!("boolean_{index}"), value).unwrap();
        }
        let graph = context.build().unwrap();
        let bindings = ParamEnv {
            integers: BTreeMap::from([("plaintext_modulus".into(), 3.into())]),
            ..ParamEnv::default()
        };
        let validated = graph.validate(&bindings, crate::test_resolve_basis).unwrap();
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
        assert_eq!(artifact.source.matches("(params.«plaintext_modulus») (2) 0").count(), 2);
        assert_eq!(artifact.source.matches("(params.«plaintext_modulus») (2) 1").count(), 2);
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
        assert_eq!(
            protocol.bundle.endpoints.entries[0].workflow_output,
            OutputRef { stage: StageId("decrypt".to_owned()), output: "decoded".to_owned() }
        );
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
