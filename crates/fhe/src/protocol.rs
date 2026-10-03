//! Closed protocol declarations of the FHE round trips, for correctness-claim export.
//!
//! Each workflow stage is the graph the integration tests execute, with its cross-stage values
//! declared as artifacts. The endpoint is [`EndpointSpecId::CenteredResidual`]: the decrypted
//! value must equal the ideal output, and the decryption phase, exported by the decryption stage
//! next to the decoded value, is the residual whose centered coefficients the claim bounds.
//!
//! The GPU-gated tests `test_export_tfhe_gate_claim` (worst-case TFHE profile) and
//! `test_export_bgv_round_trip_claim` write the generated statement modules and the certificate to
//! `crates/fhe/lean/{tfhe,bgv}/generated`. The handwritten proofs beside them prove each
//! certificate's `GeneratedClaim.CorrectnessClaim`; `lake build` in each package checks them.

use crate::{BgvCiphertext, BgvParams, FheError, FheScheme, LweCiphertext, TfheParams, utils};
use mxx_backends::poly::PolyParams;
use mxx_dsl::{
    DslContext, Family, GraphValue, IdealSpec, Int, Mat, MatType, Ring, artifact_bindings, parallel,
};
use mxx_ir_core::{
    artifact::{ArtifactAvailability, ProductionId, SpecHash},
    protocol::{
        ArtifactBinding, ClosedProtocolBundle, ComparatorEndpointBinding, ComparatorSpec,
        EndpointBinding, EndpointBindings, EndpointSemanticBinding, EndpointSpecId, InputContract,
        InputContractEntry, InputValueContract, OperationalDecoderKind, OperationalDecoderTarget,
        OutputRef, ProtocolDecl, ProtocolInputBinding, ProtocolInputDestination, ProtocolInputId,
        ProtocolPreconditionSpec, ProtocolStage, StageId, StageInputName, Workflow,
    },
};
use num_bigint::BigUint;

/// The placeholder production of each producing stage, distinct per stage so every artifact
/// input names exactly one manifest.
pub fn stage_production(stage: &StageId) -> ProductionId {
    let mut spec_hash = [0; 32];
    for (target, byte) in spec_hash.iter_mut().zip(stage.0.bytes()) {
        *target = byte;
    }
    ProductionId { spec_hash: SpecHash(spec_hash), execution_nonce: [0; 32] }
}

fn ring(parameters: &mxx_backends::poly::dcrt::params::DCRTPolyParams) -> Ring {
    utils::ring(parameters)
}

/// One stage graph together with the artifact bindings of its inputs.
struct Stage {
    id: StageId,
    graph: mxx_ir_core::Graph,
    bindings: Vec<ArtifactBinding>,
}

/// Imports the producer output `name` of `producer` as an artifact of the same name.
fn import<V: GraphValue>(
    context: &DslContext,
    producer: &StageId,
    name: &str,
    schema: V::Schema,
    bindings: &mut Vec<ArtifactBinding>,
) -> Result<V, FheError> {
    bindings.extend(artifact_bindings(&schema, name, producer, name)?);
    Ok(context.artifact_input(
        stage_production(producer),
        name,
        schema,
        ArtifactAvailability::Transferred,
    )?)
}

/// The BGV round trip of `crates/fhe/tests/gpu_bgv.rs`: key generation, two encryptions,
/// an unrelinearized product, relinearization at the top level, `modswitch_steps` level drops,
/// and decryption. The ideal output is the slotwise product modulo `t`. Also returns the static
/// bound on the centered decryption phase that the crate's noise tracking predicts.
pub fn bgv_round_trip_protocol(
    bgv: &BgvParams,
    modswitch_steps: usize,
) -> Result<(ProtocolDecl, BigUint), FheError> {
    let common = &bgv.common;
    let n = common.ring.ring_dimension() as usize;
    let top = common.ring.crt_depth() - 1;
    let lower_level = top.checked_sub(modswitch_steps).ok_or(FheError::LevelMismatch)?;
    let (key_params, key_width) = bgv.key_switch_parameters(top)?;
    let top_ring = ring(&common.ring);
    let key_ring = ring(&key_params);
    let lower_ring = ring(&common.parameters_at(lower_level)?);
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
    let mut stages = Vec::new();

    let (sk, pk) = bgv.keygen()?;
    let rk = bgv.relinearization_key(&sk, top)?;
    stages.push(Stage {
        id: keygen.clone(),
        graph: DslContext::new("bgv-round-trip-keygen")
            .transferred_output("sk", sk)?
            .transferred_output("pk", pk)?
            .transferred_output("rk", rk)?
            .build()?
            .graph,
        bindings: Vec::new(),
    });

    let mut encryption = |stage: &StageId, slots: &str, output: &str| {
        let context = DslContext::new(format!("bgv-round-trip-{}", stage.0));
        let mut bindings = Vec::new();
        let pk: Mat =
            import(&context, &keygen, "pk", MatType(top_ring.matrix_type((2, 1))), &mut bindings)?;
        let ciphertext = bgv.encrypt(&pk, &context.int_family_input(slots, n))?;
        let graph =
            context.transferred_output(output, ciphertext.components.clone())?.build()?.graph;
        stages.push(Stage { id: stage.clone(), graph, bindings });
        Ok::<_, FheError>(ciphertext)
    };
    let fresh = encryption(&encrypt_x, "x", "lhs")?;
    encryption(&encrypt_y, "y", "rhs")?;
    let fresh_at = |components: Mat| BgvCiphertext { components, ..fresh.clone() };

    let context = DslContext::new("bgv-round-trip-multiply");
    let mut bindings = Vec::new();
    let lhs =
        import(&context, &encrypt_x, "lhs", MatType(top_ring.matrix_type((2, 1))), &mut bindings)?;
    let rhs =
        import(&context, &encrypt_y, "rhs", MatType(top_ring.matrix_type((2, 1))), &mut bindings)?;
    let quadratic = bgv.mul_unrelinearized(&fresh_at(lhs), &fresh_at(rhs))?;
    stages.push(Stage {
        id: multiply.clone(),
        graph: context
            .transferred_output("quadratic", quadratic.components.clone())?
            .build()?
            .graph,
        bindings,
    });

    let context = DslContext::new("bgv-round-trip-relinearize");
    let mut bindings = Vec::new();
    let quadratic = BgvCiphertext {
        components: import(
            &context,
            &multiply,
            "quadratic",
            MatType(top_ring.matrix_type((3, 1))),
            &mut bindings,
        )?,
        ..quadratic
    };
    let rk = import(
        &context,
        &keygen,
        "rk",
        MatType(key_ring.matrix_type((2, key_width))),
        &mut bindings,
    )?;
    let relinearized = bgv.relinearize(&quadratic, &rk)?;
    stages.push(Stage {
        id: relinearize.clone(),
        graph: context
            .transferred_output("relinearized", relinearized.components.clone())?
            .build()?
            .graph,
        bindings,
    });

    let context = DslContext::new("bgv-round-trip-modswitch");
    let mut bindings = Vec::new();
    let relinearized = BgvCiphertext {
        components: import(
            &context,
            &relinearize,
            "relinearized",
            MatType(top_ring.matrix_type((2, 1))),
            &mut bindings,
        )?,
        ..relinearized
    };
    let switched = bgv.mod_switch_to(&relinearized, lower_level)?;
    let phase_bound = bgv.phase_bound(&switched);
    stages.push(Stage {
        id: modswitch.clone(),
        graph: context.transferred_output("ct", switched.components.clone())?.build()?.graph,
        bindings,
    });

    let context = DslContext::new("bgv-round-trip-decrypt");
    let mut bindings = Vec::new();
    let switched = BgvCiphertext {
        components: import(
            &context,
            &modswitch,
            "ct",
            MatType(lower_ring.matrix_type((2, 1))),
            &mut bindings,
        )?,
        ..switched
    };
    let sk = import(&context, &keygen, "sk", MatType(top_ring.matrix_type((1, 1))), &mut bindings)?;
    let phase = bgv.decryption_phase(&sk, &switched)?;
    let slots = bgv.decode_phase(&phase, &switched)?;
    stages.push(Stage {
        id: decrypt.clone(),
        graph: context.output("slots", slots)?.output("phase", phase)?.build()?.graph,
        bindings,
    });

    let ideal = DslContext::new("bgv-round-trip-ideal");
    let (x, y) = (ideal.int_family_input("x", n), ideal.int_family_input("y", n));
    let t = Int::constant(bgv.plaintext_modulus);
    let product: Family<Int> =
        parallel(n, |slot| Ok(x.at(slot.clone()).mul(y.at(slot)).rem(t.clone())))?;
    let ideal = IdealSpec::new(ideal.output("slots", product)?.build()?.graph)?;

    let slot = InputValueContract::IntegerRange {
        lower: 0.into(),
        upper: (bgv.plaintext_modulus - 1).into(),
    };
    let inputs = [("x", &encrypt_x), ("y", &encrypt_y)]
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
                            stage: stage.clone(),
                            input: StageInputName(name.to_owned()),
                        },
                        ProtocolInputDestination::Ideal { input: name.to_owned() },
                    ],
                },
            )
        })
        .collect::<Vec<_>>();
    Ok((closed_protocol(stages, decrypt, ideal, "slots", "phase", inputs)?, phase_bound))
}

/// The TFHE gate of `crates/fhe/tests/gpu_tfhe.rs`: key generation, two encryptions of
/// external bits, one bootstrapped NAND gate, and decryption. The ideal output is
/// `1 - left * right`.
pub fn tfhe_nand_protocol(tfhe: &TfheParams) -> Result<ProtocolDecl, FheError> {
    let ring = tfhe.common.ring();
    let id = |name: &str| StageId(name.to_owned());
    let (keygen, encrypt_left, encrypt_right, nand, decrypt) =
        (id("keygen"), id("encrypt_left"), id("encrypt_right"), id("nand"), id("decrypt"));
    let mut stages = Vec::new();

    let keys = tfhe.keygen(&ring.bytes_input("keygen_hash_key", 32))?;
    let (secret_schema, bsk_schema, ksk_schema) =
        (keys.lwe_secret.schema(), keys.bootstrapping_key.schema(), keys.key_switch_key.schema());
    stages.push(Stage {
        id: keygen.clone(),
        graph: DslContext::new("tfhe-gate-keygen")
            .transferred_output("lwe_sk", keys.lwe_secret)?
            .transferred_output("bsk", keys.bootstrapping_key)?
            .transferred_output("ksk", keys.key_switch_key)?
            .build()?
            .graph,
        bindings: Vec::new(),
    });

    let mut ciphertext_schema = None;
    for (stage, message, hash_key, output) in [
        (&encrypt_left, "left", "left_hash_key", "left_ct"),
        (&encrypt_right, "right", "right_hash_key", "right_ct"),
    ] {
        let context = DslContext::new(format!("tfhe-gate-{}", stage.0));
        let mut bindings = Vec::new();
        let secret: Family<Int> =
            import(&context, &keygen, "lwe_sk", secret_schema.clone(), &mut bindings)?;
        let ciphertext = tfhe.encrypt(
            &secret,
            &context.input(message, mxx_dsl::IntType)?,
            &ring.bytes_input(hash_key, 32),
        )?;
        ciphertext_schema = Some(ciphertext.schema());
        let graph = context.transferred_output(output, ciphertext)?.build()?.graph;
        stages.push(Stage { id: stage.clone(), graph, bindings });
    }
    let ciphertext_schema = ciphertext_schema.expect("two encryption stages");

    let context = DslContext::new("tfhe-gate-nand");
    let mut bindings = Vec::new();
    let left: LweCiphertext =
        import(&context, &encrypt_left, "left_ct", ciphertext_schema.clone(), &mut bindings)?;
    let right: LweCiphertext =
        import(&context, &encrypt_right, "right_ct", ciphertext_schema, &mut bindings)?;
    let bsk = import(&context, &keygen, "bsk", bsk_schema, &mut bindings)?;
    let ksk = import(&context, &keygen, "ksk", ksk_schema, &mut bindings)?;
    let gate = tfhe.nand(&left, &right, &bsk, &ksk)?;
    let gate_schema = gate.schema();
    stages.push(Stage {
        id: nand.clone(),
        graph: context.transferred_output("ct", gate)?.build()?.graph,
        bindings,
    });

    let context = DslContext::new("tfhe-gate-decrypt");
    let mut bindings = Vec::new();
    let secret: Family<Int> = import(&context, &keygen, "lwe_sk", secret_schema, &mut bindings)?;
    let gate: LweCiphertext = import(&context, &nand, "ct", gate_schema, &mut bindings)?;
    let phase = tfhe.decryption_phase(&secret, &gate)?;
    let bit = tfhe.decode_phase(phase.clone())?;
    stages.push(Stage {
        id: decrypt.clone(),
        graph: context.output("bit", bit)?.output("phase", phase)?.build()?.graph,
        bindings,
    });

    let ideal = DslContext::new("tfhe-gate-ideal");
    let (left, right): (Int, Int) =
        (ideal.input("left", mxx_dsl::IntType)?, ideal.input("right", mxx_dsl::IntType)?);
    let ideal =
        IdealSpec::new(ideal.output("bit", Int::constant(1).sub(left.mul(right)))?.build()?.graph)?;

    let bit = InputValueContract::IntegerRange { lower: 0.into(), upper: 1.into() };
    let key = InputValueContract::Bytes { length: 32.into() };
    let inputs = [
        ("keygen_hash_key", key.clone(), vec![(&keygen, false)]),
        ("left", bit.clone(), vec![(&encrypt_left, true)]),
        ("left_hash_key", key.clone(), vec![(&encrypt_left, false)]),
        ("right", bit, vec![(&encrypt_right, true)]),
        ("right_hash_key", key, vec![(&encrypt_right, false)]),
    ]
    .into_iter()
    .map(|(name, value, stages)| {
        let mut destinations = Vec::new();
        for (stage, ideal) in stages {
            destinations.push(ProtocolInputDestination::WorkflowStage {
                stage: stage.clone(),
                input: StageInputName(name.to_owned()),
            });
            if ideal {
                destinations.push(ProtocolInputDestination::Ideal { input: name.to_owned() });
            }
        }
        (
            InputContractEntry { id: ProtocolInputId::from(name), name: name.to_owned(), value },
            ProtocolInputBinding { input: ProtocolInputId::from(name), destinations },
        )
    })
    .collect();
    closed_protocol(stages, decrypt, ideal, "bit", "phase", inputs)
}

/// The TFHE claim semantics: the residual is the phase minus the encoded bit `±floor(q/8)`,
/// within `floor(q/8)` so that the sign decoder returns the bit.
pub fn tfhe_semantics(tfhe: &TfheParams) -> String {
    let delta = tfhe.delta();
    format!(
        "import MxxRuntime\n\nnamespace TfheSemantics\n\n\
         /-- The encoded bit: `floor(q/8)` for one, `q - floor(q/8)` for zero. -/\n\
         def messageCenter (q : Nat) (bit : Int) (_ : Fin 1) : Int :=\n  \
         if bit = 1 then {delta} else (q : Int) - {delta}\n\n\
         /-- The sign decoder returns the encoded bit within this distance. -/\n\
         def decoderRadius (_ : Nat) : Nat := {delta}\n\nend TfheSemantics\n"
    )
}

/// Assembles the single centered-residual endpoint shared by both round trips.
fn closed_protocol(
    stages: Vec<Stage>,
    decrypt: StageId,
    ideal: IdealSpec,
    decoded: &str,
    residual: &str,
    inputs: Vec<(InputContractEntry, ProtocolInputBinding)>,
) -> Result<ProtocolDecl, FheError> {
    let endpoint = EndpointSpecId::CenteredResidual;
    let (contracts, bindings): (Vec<_>, Vec<_>) = inputs.into_iter().unzip();
    Ok(ProtocolDecl::new(ProtocolDecl {
        params: Vec::new(),
        bundle: ClosedProtocolBundle {
            workflow: Workflow {
                stages: stages
                    .into_iter()
                    .map(|stage| ProtocolStage {
                        id: stage.id,
                        graph: stage.graph,
                        bindings: stage.bindings,
                    })
                    .collect(),
                entrypoint: decrypt.clone(),
            },
            ideal,
            requirements: Vec::new(),
            comparator: ComparatorSpec::Equality {
                endpoints: vec![ComparatorEndpointBinding {
                    endpoint,
                    actual_input: decoded.to_owned(),
                    ideal_input: decoded.to_owned(),
                    result_output: "failure".to_owned(),
                    failure_value: true,
                }],
            },
            endpoints: EndpointBindings {
                entries: vec![EndpointBinding {
                    spec: endpoint,
                    semantics: EndpointSemanticBinding::Exact,
                    workflow_output: OutputRef {
                        stage: decrypt.clone(),
                        output: decoded.to_owned(),
                    },
                    ideal_output: decoded.to_owned(),
                }],
            },
            operational_decoder_targets: vec![OperationalDecoderTarget {
                target_id: "decryption-phase".to_owned(),
                residual: OutputRef { stage: decrypt, output: residual.to_owned() },
                endpoint,
                kind: OperationalDecoderKind::CenteredResidual,
            }],
            endpoint_specs: vec![endpoint],
            input_contract: InputContract { inputs: contracts },
            input_bindings: bindings,
            precondition_spec: ProtocolPreconditionSpec::default(),
        },
    })?)
}

/// Writes every stage, ideal, and backend module of `protocol`, its linked `Claim.lean`, and the
/// semantics module `semantics` (named `semantics_module`) into `directory`.
pub fn export_claim(
    protocol: &ProtocolDecl,
    runtime_parameters: &[mxx_backends::poly::dcrt::params::DCRTPolyParams],
    semantics_module: &str,
    semantics: &str,
    directory: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use mxx_ir_core::{
        ParamEnv,
        artifact::export_validated_manifest,
        lean::claim::{ClaimBackend, ClaimSemantics},
        validate_with_manifests,
    };
    let bindings = ParamEnv::default();
    // Stages are declared producer before consumer, so each validates against earlier manifests.
    let mut manifests = std::collections::BTreeMap::new();
    for stage in protocol.stages() {
        let validated = validate_with_manifests(&stage.graph, &bindings, &manifests)?;
        let production = stage_production(&stage.id);
        manifests.insert(production.clone(), export_validated_manifest(production, &validated)?);
    }
    let layouts = mxx_backends::lean::export_dcrt_layouts(runtime_parameters)?;
    let backend = mxx_backends::lean::render_backend_context(&layouts, "Backend", "FheBackend")?;
    std::fs::create_dir_all(directory)?;
    std::fs::write(directory.join("Backend.lean"), backend.source())?;
    std::fs::write(directory.join(format!("{semantics_module}.lean")), semantics)?;
    mxx_ir_core::lean::protocol::export_claim(
        protocol,
        &bindings,
        &ClaimBackend {
            module_name: backend.module_name(),
            context_name: backend.context_name(),
            layouts: &backend.exporter_bindings(),
        },
        &ClaimSemantics {
            imports: &[semantics_module],
            hash_model_type: "MxxRuntime.HashModel",
            centered_lift: "Mxx.Primitives.centeredLift",
            message_center: &format!("{semantics_module}.messageCenter"),
            decoder_radius: &format!("{semantics_module}.decoderRadius"),
        },
        &manifests,
        directory,
    )
}

/// The BGV claim semantics: the decryption phase is the residual itself, and the radius is one
/// above the crate's static phase bound, so the claim states that this bound is sound.
pub fn bgv_semantics(ring_dimension: usize, phase_bound: &BigUint) -> String {
    format!(
        "import MxxRuntime\n\nnamespace BgvSemantics\n\n\
         /-- No message is subtracted: the centered decryption phase is the residual. -/\n\
         def messageCenter (_ : Nat) (_ : Fin {ring_dimension} → Int) (_ : Fin {ring_dimension}) : Int := 0\n\n\
         /-- One above the static bound on the centered decryption phase. -/\n\
         def decoderRadius (_ : Nat) : Nat := {}\n\nend BgvSemantics\n",
        phase_bound + 1u8
    )
}

/// The BGV slot permutations as the packed tables the stage modules embed, for the proof.
pub fn bgv_tables(bgv: &BgvParams) -> Result<String, FheError> {
    let (encode, decode) = bgv.slot_tables()?;
    let pack = |values: Vec<usize>| {
        let (width, table) = mxx_ir_core::lean::packed_table(
            &values.into_iter().map(BigUint::from).collect::<Vec<_>>(),
        );
        format!("{width} {table}")
    };
    let field = |name: &str, doc: &str, packed: String| {
        let (width, table) = packed.split_once(' ').expect("width and table");
        format!("/-- {doc} -/\ndef {name}Width : Nat := {width}\n\ndef {name} : Nat := {table}\n\n")
    };
    Ok(format!(
        "namespace BgvTables\n\n{}{}end BgvTables\n",
        field(
            "encode",
            "Encoding reads logical slot `encode[j]` into native evaluation `j`.",
            pack(encode)
        ),
        field(
            "decode",
            "Decoding reads native evaluation `decode[i]` into logical slot `i`.",
            pack(decode)
        ),
    ))
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;

    /// Exports the claim of the worst-case TFHE profile, which `FHE_TEST_TFHE_PROFILE` selects.
    #[test]
    fn test_export_tfhe_gate_claim() {
        // SAFETY: tests in this binary that read the TFHE profile run in this thread only.
        unsafe { std::env::set_var("FHE_TEST_TFHE_PROFILE", "worst-case") };
        let tfhe = utils::tfhe_params();
        let protocol = tfhe_nand_protocol(&tfhe).unwrap();
        let directory =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("lean/tfhe/generated");
        export_claim(
            &protocol,
            &tfhe.runtime_parameters(),
            "TfheSemantics",
            &tfhe_semantics(&tfhe),
            &directory,
        )
        .unwrap();
        std::fs::write(
            directory.join("Certificate.lean"),
            mxx_ir_core::lean::claim::assemble_certificate("TfheProof", "MxxFheTfhe.correctness")
                .unwrap(),
        )
        .unwrap();
        let claim = std::fs::read_to_string(directory.join("Claim.lean")).unwrap();
        assert!(claim.contains("TfheSemantics.decoderRadius"));
    }

    #[test]
    fn test_export_bgv_round_trip_claim() {
        let bgv = utils::bgv_params();
        let (protocol, phase_bound) =
            bgv_round_trip_protocol(&bgv, utils::modswitch_steps()).unwrap();
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("lean/bgv/generated");
        export_claim(
            &protocol,
            &bgv.runtime_parameters().unwrap(),
            "BgvSemantics",
            &bgv_semantics(bgv.common.ring.ring_dimension() as usize, &phase_bound),
            &directory,
        )
        .unwrap();
        std::fs::write(directory.join("BgvTables.lean"), bgv_tables(&bgv).unwrap()).unwrap();
        std::fs::write(
            directory.join("Certificate.lean"),
            mxx_ir_core::lean::claim::assemble_certificate("BgvProof", "MxxFheBgv.correctness")
                .unwrap(),
        )
        .unwrap();
        let claim = std::fs::read_to_string(directory.join("Claim.lean")).unwrap();
        assert!(claim.contains("BgvSemantics.decoderRadius"));
    }
}
