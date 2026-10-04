//! Host-only construction of the exact round trips executed by the GPU integration tests.
//! The exporter uses these graphs and linked `ProtocolDecl`s without sampling or execution.

use crate::{BgvCiphertext, BgvParams, FheScheme, TfheParams};
use mxx_backends::poly::PolyParams;
use mxx_dsl::{
    BuiltGraph, DslContext, Family, GraphValue, GraphValueSchema, IdealSpec, Int, IntType, Ring,
    parallel,
};
use mxx_ir_core::{
    Graph,
    protocol::{
        ArtifactBinding, ArtifactName, ClosedProtocolBundle, ComparatorEndpointBinding,
        ComparatorSpec, EndpointBinding, EndpointBindings, EndpointSemanticBinding, EndpointSpecId,
        InputContract, InputContractEntry, InputValueContract, OutputRef, ProtocolDecl,
        ProtocolInputBinding, ProtocolInputDestination, ProtocolInputId, ProtocolPreconditionSpec,
        ProtocolStage, StageId, StageInputName, Workflow,
    },
};

/// Graphs for one TFHE NAND gate and their closed correctness claim.
pub struct TfheRoundTrip {
    pub keygen_graph: BuiltGraph,
    pub encryption_graph: BuiltGraph,
    pub nand_graph: BuiltGraph,
    pub decryption_graph: BuiltGraph,
    pub protocol: ProtocolDecl,
}

/// Builds the same graph templates used at every TFHE GPU execution boundary.
/// Parameter overrides are supplied by the caller through `utils::tfhe_params`.
pub fn tfhe_round_trip(tfhe: &TfheParams) -> TfheRoundTrip {
    let ring = Ring::from_crt_moduli(
        tfhe.common.ring.to_crt().0.into_iter().map(Into::into).collect(),
        tfhe.common.ring.ring_dimension(),
    );
    let keys = tfhe.keygen(&ring.bytes_input("keygen_hash_key", 32)).unwrap();
    let keygen_graph = DslContext::new("tfhe-round-trip-keygen")
        .output("lwe_sk", keys.lwe_secret.clone())
        .unwrap()
        .output("bsk", keys.bootstrapping_key.clone())
        .unwrap()
        .output("ksk", keys.key_switch_key.clone())
        .unwrap()
        .build()
        .unwrap();
    let encryption = DslContext::new("tfhe-round-trip-encryption");
    let ciphertext = tfhe
        .encrypt(
            &encryption.int_family_input("lwe_sk", tfhe.lwe_dimension),
            &encryption.input("message", IntType).unwrap(),
            &ring.bytes_input("encryption_hash_key", 32),
        )
        .unwrap();
    let ciphertext_schema = ciphertext.schema();
    let encryption_graph = encryption.output("ct", ciphertext).unwrap().build().unwrap();
    let nand_graph = {
        let gate = DslContext::new("tfhe-round-trip-nand");
        let nand = tfhe
            .nand(
                &gate.input("left", ciphertext_schema.clone()).unwrap(),
                &gate.input("right", ciphertext_schema.clone()).unwrap(),
                &gate.input("bsk", keys.bootstrapping_key.schema()).unwrap(),
                &gate.input("ksk", keys.key_switch_key.schema()).unwrap(),
            )
            .unwrap();
        gate.output("ct", nand).unwrap().build().unwrap()
    };
    let decryption = DslContext::new("tfhe-round-trip-decryption");
    let bit = tfhe
        .decrypt(
            &decryption.int_family_input("lwe_sk", tfhe.lwe_dimension),
            &decryption.input("ct", ciphertext_schema.clone()).unwrap(),
        )
        .unwrap();
    let decryption_graph = decryption.output("bit", bit).unwrap().build().unwrap();
    let id = |name: &str| StageId(name.to_owned());
    let (keygen, encrypt_left, encrypt_right, nand, decrypt) =
        (id("keygen"), id("encrypt_left"), id("encrypt_right"), id("nand"), id("decrypt"));
    let secret_schema = keys.lwe_secret.schema();
    let stage = |id: &StageId, graph: Graph, bindings: Vec<Vec<ArtifactBinding>>| ProtocolStage {
        id: id.clone(),
        graph,
        bindings: bindings.concat(),
    };
    let encryption_at = |id: &StageId| {
        stage(
            id,
            encryption_graph.graph.clone(),
            vec![link(&secret_schema, "lwe_sk", &keygen, "lwe_sk")],
        )
    };
    let stages = vec![
        stage(&keygen, keygen_graph.graph.clone(), vec![]),
        encryption_at(&encrypt_left),
        encryption_at(&encrypt_right),
        stage(
            &nand,
            nand_graph.graph.clone(),
            vec![
                link(&ciphertext_schema, "left", &encrypt_left, "ct"),
                link(&ciphertext_schema, "right", &encrypt_right, "ct"),
                link(&keys.bootstrapping_key.schema(), "bsk", &keygen, "bsk"),
                link(&keys.key_switch_key.schema(), "ksk", &keygen, "ksk"),
            ],
        ),
        stage(
            &decrypt,
            decryption_graph.graph.clone(),
            vec![
                link(&secret_schema, "lwe_sk", &keygen, "lwe_sk"),
                link(&ciphertext_schema, "ct", &nand, "ct"),
            ],
        ),
    ];
    let protocol = gate_protocol(stages);
    TfheRoundTrip { keygen_graph, encryption_graph, nand_graph, decryption_graph, protocol }
}

/// The bindings feeding every leaf of the consumer input `consumer` from the producer output.
fn link<S: GraphValueSchema>(
    schema: &S,
    consumer: &str,
    producer: &StageId,
    output: &str,
) -> Vec<ArtifactBinding> {
    mxx_dsl::artifact_bindings(schema, consumer, producer, output).unwrap()
}

/// One gate as a closed protocol over the linked `stages`: keygen, the two encryptions, the gate,
/// and decryption. The decrypted bit must equal the ideal `1 - left * right`, except with
/// probability at most `2^-128` over the sampled values.
fn gate_protocol(stages: Vec<ProtocolStage>) -> ProtocolDecl {
    let id = |name: &str| StageId(name.to_owned());
    let (keygen, encrypt_left, encrypt_right, decrypt) =
        (id("keygen"), id("encrypt_left"), id("encrypt_right"), id("decrypt"));
    let ideal = DslContext::new("tfhe-gate-ideal");
    let (left, right): (Int, Int) =
        (ideal.input("left", IntType).unwrap(), ideal.input("right", IntType).unwrap());
    let ideal = IdealSpec::new(
        ideal.output("bit", Int::constant(1).sub(left.mul(right))).unwrap().build().unwrap().graph,
    )
    .unwrap();
    let bit = InputValueContract::IntegerRange { lower: 0.into(), upper: 1.into() };
    let key = InputValueContract::Bytes { length: 32.into() };
    let (contracts, bindings): (Vec<_>, Vec<_>) = [
        ("keygen_hash_key", key.clone(), vec![(&keygen, "keygen_hash_key")], None),
        ("left", bit.clone(), vec![(&encrypt_left, "message")], Some("left")),
        ("left_hash_key", key.clone(), vec![(&encrypt_left, "encryption_hash_key")], None),
        ("right", bit, vec![(&encrypt_right, "message")], Some("right")),
        ("right_hash_key", key, vec![(&encrypt_right, "encryption_hash_key")], None),
    ]
    .into_iter()
    .map(|(name, value, stages, ideal)| {
        let mut destinations = stages
            .into_iter()
            .map(|(stage, input)| ProtocolInputDestination::WorkflowStage {
                stage: stage.clone(),
                input: StageInputName(input.to_owned()),
            })
            .collect::<Vec<_>>();
        destinations
            .extend(ideal.map(|input| ProtocolInputDestination::Ideal { input: input.to_owned() }));
        (
            InputContractEntry { id: ProtocolInputId::from(name), name: name.to_owned(), value },
            ProtocolInputBinding { input: ProtocolInputId::from(name), destinations },
        )
    })
    .unzip();
    // The closed protocol: the workflow output `bit` of the `decrypt` stage is compared with the
    // ideal output of the same name, except with probability at most `2^-128`.
    let endpoint = EndpointSpecId::Exact;
    ProtocolDecl::new(ProtocolDecl {
        params: Vec::new(),
        bindings: Default::default(),
        failure_probability_log2: Some(128),
        bundle: ClosedProtocolBundle {
            workflow: Workflow { stages, entrypoint: decrypt.clone() },
            ideal,
            requirements: Vec::new(),
            comparator: ComparatorSpec::Equality {
                endpoints: vec![ComparatorEndpointBinding {
                    endpoint,
                    actual_input: "bit".to_owned(),
                    ideal_input: "bit".to_owned(),
                    result_output: "failure".to_owned(),
                    failure_value: true,
                }],
            },
            endpoints: EndpointBindings {
                entries: vec![EndpointBinding {
                    spec: endpoint,
                    semantics: EndpointSemanticBinding::Exact,
                    workflow_output: OutputRef { stage: decrypt, output: "bit".to_owned() },
                    ideal_output: "bit".to_owned(),
                }],
            },
            operational_decoder_targets: Vec::new(),
            endpoint_specs: vec![endpoint],
            input_contract: InputContract { inputs: contracts },
            input_bindings: bindings,
            precondition_spec: ProtocolPreconditionSpec::default(),
        },
    })
    .unwrap()
}

/// Graphs for the BGV multiplication round trip and its closed correctness claim.
pub struct BgvRoundTrip {
    pub keygen_graph: BuiltGraph,
    pub encryption_graph_x: BuiltGraph,
    pub encryption_graph_y: BuiltGraph,
    pub multiply_graph: BuiltGraph,
    pub relinearize_graph: BuiltGraph,
    pub modswitch_graph: BuiltGraph,
    pub decryption_graph: BuiltGraph,
    pub protocol: ProtocolDecl,
}

/// Builds the BGV keygen, two encryptions, multiplication, relinearization,
/// modulus switch and decryption graphs, carrying their correction factors and noise bounds.
pub fn bgv_round_trip(bgv: &BgvParams, modswitch_steps: usize) -> BgvRoundTrip {
    let n = bgv.common.ring.ring_dimension() as usize;
    let common = &bgv.common;
    let top = common.ring.crt_depth() - 1;
    let lower_level = top
        .checked_sub(modswitch_steps)
        .expect("modswitch steps must fit within the ciphertext levels");
    let (key_params, key_width) = bgv.key_switch_parameters(top).unwrap();
    let ciphertext_ring = Ring::from_crt_moduli(
        common.ring.to_crt().0.into_iter().map(Into::into).collect(),
        common.ring.ring_dimension(),
    );

    let (sk, pk) = bgv.keygen().unwrap();
    let generated_rk = bgv.relinearization_key(&sk, top).unwrap();
    let keygen_graph = DslContext::new("bgv-round-trip-keygen")
        .output("sk", sk)
        .unwrap()
        .output("pk", pk)
        .unwrap()
        .output("rk", generated_rk)
        .unwrap()
        .build()
        .unwrap();
    let pk_input = ciphertext_ring.input("pk", (2, 1));
    let lhs_template = bgv
        .encrypt(
            &pk_input,
            &DslContext::new("bgv-round-trip-encryption-x").int_family_input("x", n),
        )
        .unwrap();
    let encryption_graph_x = DslContext::new("bgv-round-trip-encryption-x")
        .output("lhs", lhs_template.components.clone())
        .unwrap()
        .build()
        .unwrap();
    let rhs_template = bgv
        .encrypt(
            &pk_input,
            &DslContext::new("bgv-round-trip-encryption-y").int_family_input("y", n),
        )
        .unwrap();
    let encryption_graph_y = DslContext::new("bgv-round-trip-encryption-y")
        .output("rhs", rhs_template.components.clone())
        .unwrap()
        .build()
        .unwrap();
    let lhs = BgvCiphertext { components: ciphertext_ring.input("lhs", (2, 1)), ..lhs_template };
    let rhs = BgvCiphertext { components: ciphertext_ring.input("rhs", (2, 1)), ..rhs_template };
    let key_ring = Ring::from_crt_moduli(
        key_params.to_crt().0.into_iter().map(Into::into).collect(),
        key_params.ring_dimension(),
    );
    let rk = key_ring.input("rk", (2, key_width));
    let quadratic = bgv.mul_unrelinearized(&lhs, &rhs).unwrap();
    let multiply_graph = DslContext::new("bgv-round-trip-multiply")
        .output("quadratic", quadratic.components.clone())
        .unwrap()
        .build()
        .unwrap();
    let quadratic_input = BgvCiphertext {
        components: ciphertext_ring.input("quadratic", (3, 1)),
        correction_factor: quadratic.correction_factor,
        noise_bound: quadratic.noise_bound.clone(),
    };
    let relinearized = bgv.relinearize(&quadratic_input, &rk).unwrap();
    let relinearize_graph = DslContext::new("bgv-round-trip-relinearize")
        .output("relinearized", relinearized.components.clone())
        .unwrap()
        .build()
        .unwrap();
    let relinearized_input = BgvCiphertext {
        components: ciphertext_ring.input("relinearized", (2, 1)),
        correction_factor: relinearized.correction_factor,
        noise_bound: relinearized.noise_bound.clone(),
    };
    let switched = bgv.mod_switch_to(&relinearized_input, lower_level).unwrap();
    let modswitch_graph = DslContext::new("bgv-round-trip-modswitch")
        .output("ct", switched.components.clone())
        .unwrap()
        .build()
        .unwrap();
    let secret = ciphertext_ring.input("sk", (1, 1));
    let lower = common.parameters_at(lower_level).unwrap();
    let lower_ring = Ring::from_crt_moduli(
        lower.to_crt().0.into_iter().map(Into::into).collect(),
        lower.ring_dimension(),
    );
    let imported = BgvCiphertext {
        components: lower_ring.input("ct", (2, 1)),
        correction_factor: switched.correction_factor,
        noise_bound: switched.noise_bound,
    };
    let decryption_graph = DslContext::new("bgv-round-trip-decryption")
        .output("slots", bgv.decrypt(&secret, &imported).unwrap())
        .unwrap()
        .build()
        .unwrap();
    let id = |name: &str| StageId(name.to_owned());
    let (keygen, encrypt_x, encrypt_y, multiply, relinearize, modswitch, decrypt) = (
        id("keygen"),
        id("encrypt_x"),
        id("encrypt_y"),
        id("multiply"),
        id("relinearize"),
        id("modswitch"),
        id("decrypt"),
    );
    let link = |consumer: &str, producer: &StageId, output: &str| ArtifactBinding {
        consumer_input: StageInputName(consumer.to_owned()),
        producer_stage: producer.clone(),
        producer_output: ArtifactName(output.to_owned()),
    };
    let stage = |id: &StageId, graph: Graph, bindings: Vec<ArtifactBinding>| ProtocolStage {
        id: id.clone(),
        graph,
        bindings,
    };
    let stages = vec![
        stage(&keygen, keygen_graph.graph.clone(), vec![]),
        stage(&encrypt_x, encryption_graph_x.graph.clone(), vec![link("pk", &keygen, "pk")]),
        stage(&encrypt_y, encryption_graph_y.graph.clone(), vec![link("pk", &keygen, "pk")]),
        stage(
            &multiply,
            multiply_graph.graph.clone(),
            vec![link("lhs", &encrypt_x, "lhs"), link("rhs", &encrypt_y, "rhs")],
        ),
        stage(
            &relinearize,
            relinearize_graph.graph.clone(),
            vec![link("quadratic", &multiply, "quadratic"), link("rk", &keygen, "rk")],
        ),
        stage(
            &modswitch,
            modswitch_graph.graph.clone(),
            vec![link("relinearized", &relinearize, "relinearized")],
        ),
        stage(
            &decrypt,
            decryption_graph.graph.clone(),
            vec![link("ct", &modswitch, "ct"), link("sk", &keygen, "sk")],
        ),
    ];
    let protocol = round_trip_protocol(bgv, stages);
    BgvRoundTrip {
        keygen_graph,
        encryption_graph_x,
        encryption_graph_y,
        multiply_graph,
        relinearize_graph,
        modswitch_graph,
        decryption_graph,
        protocol,
    }
}

/// The round trip as a closed protocol over the linked `stages`: every execution must decrypt to
/// the slotwise product of the plaintexts modulo `t`.
fn round_trip_protocol(bgv: &BgvParams, stages: Vec<ProtocolStage>) -> ProtocolDecl {
    // The ring dimension, which is also the number of plaintext slots.
    let n = bgv.common.ring.ring_dimension() as usize;
    // Stages are named by `StageId`s; these must match the names given to `stages` by the caller.
    let id = |name: &str| StageId(name.to_owned());

    // The ideal functionality: a separate DSL graph that computes, without any encryption, what the
    // round trip should decrypt to. It reads the same plaintext slot vectors `x` and `y` that the
    // encryption stages read, and outputs `x[i] * y[i] mod t` for every slot `i`.
    let ideal = DslContext::new("bgv-round-trip-ideal");
    let (x, y) = (ideal.int_family_input("x", n), ideal.int_family_input("y", n));
    let t = Int::constant(bgv.plaintext_modulus);
    let product: Family<Int> =
        parallel(n, |slot| Ok(x.at(slot.clone()).mul(y.at(slot)).rem(t.clone()))).unwrap();
    // Its output is named `slots`, like the decryption stage's output it is compared with.
    let ideal =
        IdealSpec::new(ideal.output("slots", product).unwrap().build().unwrap().graph).unwrap();

    // The external inputs of the protocol: the values the caller chooses, as opposed to values
    // computed by an earlier stage. Each one gets
    // - a contract, the assumption the claim makes about it: here, `n` integers in `[0, t - 1]`;
    // - its destinations, the graph inputs it feeds: the encryption stage of the same name and the
    //   ideal graph's input of the same name, so both sides see the same plaintexts.
    let slot = InputValueContract::IntegerRange {
        lower: 0.into(),
        upper: (bgv.plaintext_modulus - 1).into(),
    };
    let (contracts, bindings): (Vec<_>, Vec<_>) = [("x", id("encrypt_x")), ("y", id("encrypt_y"))]
        .into_iter()
        .map(|(name, stage)| {
            (
                InputContractEntry {
                    id: ProtocolInputId::from(name),
                    name: name.to_owned(),
                    value: InputValueContract::Family {
                        count: n.into(),
                        element: Box::new(slot.clone()),
                    },
                },
                ProtocolInputBinding {
                    input: ProtocolInputId::from(name),
                    destinations: vec![
                        ProtocolInputDestination::WorkflowStage {
                            stage,
                            input: StageInputName(name.to_owned()),
                        },
                        ProtocolInputDestination::Ideal { input: name.to_owned() },
                    ],
                },
            )
        })
        .unzip();
    // The closed protocol: the workflow output `slots` of the `decrypt` stage is compared with the
    // ideal output of the same name for every execution.
    let endpoint = EndpointSpecId::Exact;
    let decrypt = id("decrypt");
    ProtocolDecl::new(ProtocolDecl {
        params: Vec::new(),
        bindings: Default::default(),
        failure_probability_log2: None,
        bundle: ClosedProtocolBundle {
            workflow: Workflow { stages, entrypoint: decrypt.clone() },
            ideal,
            requirements: Vec::new(),
            comparator: ComparatorSpec::Equality {
                endpoints: vec![ComparatorEndpointBinding {
                    endpoint,
                    actual_input: "slots".to_owned(),
                    ideal_input: "slots".to_owned(),
                    result_output: "failure".to_owned(),
                    failure_value: true,
                }],
            },
            endpoints: EndpointBindings {
                entries: vec![EndpointBinding {
                    spec: endpoint,
                    semantics: EndpointSemanticBinding::Exact,
                    workflow_output: OutputRef { stage: decrypt, output: "slots".to_owned() },
                    ideal_output: "slots".to_owned(),
                }],
            },
            operational_decoder_targets: Vec::new(),
            endpoint_specs: vec![endpoint],
            input_contract: InputContract { inputs: contracts },
            input_bindings: bindings,
            precondition_spec: ProtocolPreconditionSpec::default(),
        },
    })
    .unwrap()
}
