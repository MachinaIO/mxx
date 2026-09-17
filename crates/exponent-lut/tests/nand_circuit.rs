#![cfg(feature = "gpu")]

//! Public RNS-BGG NAND application workflow.

use bigdecimal::BigDecimal;
use mxx_bench_estimator::{
    estimate, gpu::GpuNodeMeasurementBackend, harness::MeasurementHarnessConfig,
};
use mxx_dsl::DslContext;
use mxx_exponent_lut::{
    ExpLayout, NandCircuit, api,
    program::{AttributeKnowledge, InputBinding, Operand, RhsId},
    rns_bgg::{
        RnsBggDerivedWorkload, RnsBggEncoding, RnsBggNoiseBindings, RnsBggNoiseRhs,
        RnsBggNoiseValue, RnsBggParameters, RnsBggPayloadSecret, RnsBggPublicKey, RnsEncryptBuild,
        RnsNoiseBuild, RnsOnlineBuild, RnsPreprocessBuild,
    },
};
use mxx_ir_core::ParamEnv;
use mxx_primitives::poly::{
    PolyParams,
    dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams},
};
use mxx_runtime::{
    ArtifactKey, ArtifactStore, Backend, RuntimeValue,
    artifact::MemoryArtifactStore,
    backend::poly::gpu::{GpuDcrtBackend, gpu_backend_on},
    executor::decode_artifact,
};
use num_bigint::{BigInt, BigUint};
use serial_test::serial as sequential;
use std::path::Path;

const HASH_KEY: &[u8] = b"nand-rns-public-api-v1";

fn rns_parameters() -> (DCRTPolyParams, RnsBggParameters) {
    let source = DCRTPolyParams::new(32, 2, 10, 5, None, None);
    let (moduli, _, _) = source.to_crt();
    let levels = (0..moduli.len())
        .map(|drop| {
            let modulus = moduli[drop..]
                .iter()
                .fold(BigUint::from(1u8), |product, prime| product * BigUint::from(*prime));
            source.select_modulus(&modulus).expect("nested RNS modulus level")
        })
        .collect::<Vec<_>>();
    let parameters = RnsBggParameters::new(levels, BigDecimal::from(1u8), 1)
        .expect("valid RNS-BGG parameter chain");
    (source, parameters)
}

fn gpu_parameters(parameters: &RnsBggParameters) -> Vec<GpuDCRTPolyParams> {
    parameters
        .levels()
        .iter()
        .map(|level| {
            GpuDCRTPolyParams::new(
                level.ring_dimension(),
                level.to_crt().0,
                level.base_bits(),
                None,
            )
        })
        .collect()
}

fn noise_bindings(
    compiled: &mxx_exponent_lut::CompiledNandCircuit,
    parameters: &RnsBggParameters,
) -> RnsBggNoiseBindings {
    let program = compiled.program();
    let private_input =
        RnsBggNoiseValue::public(parameters.encoding_error_bound().clone(), &[BigInt::from(1u8)]);
    let inputs = program
        .encryption_inputs()
        .inputs()
        .keys()
        .map(|wire| (*wire, InputBinding::Scalar(private_input.clone())))
        .collect();
    let rhs = compiled
        .rhs_ids()
        .iter()
        .copied()
        .map(|id: RhsId| {
            (
                id,
                InputBinding::Scalar(RnsBggNoiseRhs {
                    payload_l1: 1u8.into(),
                    payload_l2_squared: 1u8.into(),
                }),
            )
        })
        .collect();
    RnsBggNoiseBindings::new(inputs, rhs)
        .expect("NAND noise bindings match the public declarations")
}

fn checkpoint(path: &Path, name: &str) -> std::path::PathBuf {
    path.join(name)
}

#[test]
#[sequential(dcrt_runtime)]
fn nand_circuit_end_to_end() {
    let circuit = NandCircuit::new(2, vec![[0, 1]], 2).expect("valid NAND circuit");
    let layout = ExpLayout { stride_log2: 0, width_log2: 4 };
    let compiled = circuit.compile(layout, 32, 4).expect("NAND program compiles");
    let (poly_parameters, parameters) = rns_parameters();
    let payload: RnsBggPayloadSecret =
        RnsEncryptBuild::<GpuDcrtBackend>::new(&parameters, [0x4e; 32])
            .sample_payload_secret()
            .expect("payload secret");
    let mask = RnsEncryptBuild::<GpuDcrtBackend>::new(&parameters, [0x4e; 32])
        .sample_mask_secret()
        .expect("mask secret");
    let devices = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids();
    assert!(!devices.is_empty(), "GPU required for NAND round trip");

    for assignment in [[false, false], [false, true], [true, false], [true, true]] {
        let (lhs, rhs) = compiled
            .plaintext_inputs::<mxx_primitives::poly::dcrt::poly::DCRTPoly>(
                &poly_parameters,
                &assignment,
            )
            .expect("NAND plaintext bindings");
        let bindings = noise_bindings(&compiled, &parameters);
        let workload = RnsBggDerivedWorkload::from_program(compiled.program())
            .expect("NAND workload derivation");
        let report = api::noise_simulate(
            compiled.program(),
            RnsNoiseBuild::new(&parameters, compiled.program(), &workload, &bindings)
                .expect("RNS noise builder"),
            |_, outputs| Ok(outputs),
        )
        .expect("NAND noise simulation");
        assert!(!report.wires.is_empty(), "noise simulation must trace the program");
        let output_bound = report
            .wires
            .values()
            .next_back()
            .expect("noise simulation must bound the final NAND output")
            .bound
            .clone();

        let measurement_backends = devices
            .iter()
            .copied()
            .map(|device| (gpu_backend_on(gpu_parameters(&parameters), [device]), device))
            .collect();
        let mut estimator = GpuNodeMeasurementBackend::new(
            measurement_backends,
            MeasurementHarnessConfig {
                warm_up_iterations: 1,
                measured_iterations: 1,
                ..MeasurementHarnessConfig::default()
            },
        );
        let mut backend = gpu_backend_on(gpu_parameters(&parameters), devices.clone());
        let mut store = MemoryArtifactStore::default();
        let directory = tempfile::tempdir().expect("checkpoint directory");
        let preprocess_encrypt = checkpoint(directory.path(), "preprocess-encrypt.json");
        let preprocess_online = checkpoint(directory.path(), "preprocess-online.json");
        let encrypted_checkpoint = checkpoint(directory.path(), "encrypt.json");
        let config = mxx_runtime::ExecutionConfig::default();
        let (preprocess, online) = api::preprocess(
            compiled.program(),
            &ParamEnv::default(),
            &rhs,
            &payload,
            HASH_KEY,
            RnsPreprocessBuild::<GpuDcrtBackend>::new(&parameters, [0x4e; 32]),
            &mut backend,
            &mut store,
            [&preprocess_encrypt, &preprocess_online],
            config.clone(),
            |_, outputs| Ok(outputs),
            |validated| {
                estimate(validated, &mut estimator).expect("estimate preprocess graph");
                estimator.measure_collected().expect("measure preprocess graph");
                let report = estimate(validated, &mut estimator).expect("preprocess estimate");
                assert!(report.total_work_seconds > 0.0);
                Ok(())
            },
        )
        .expect("public preprocessing");
        let encrypted = api::encrypt(
            &compiled.program().encryption_inputs(),
            &ParamEnv::default(),
            &lhs,
            &preprocess,
            &payload,
            &mask,
            RnsEncryptBuild::<GpuDcrtBackend>::new(&parameters, [0x4e; 32]),
            &mut backend,
            api::ArtifactDestination { store: &mut store, checkpoint: &encrypted_checkpoint },
            config.clone(),
            |_| Ok(()),
            |validated| {
                estimate(validated, &mut estimator).expect("estimate encrypt graph");
                estimator.measure_collected().expect("measure encrypt graph");
                let report = estimate(validated, &mut estimator).expect("encrypt estimate");
                assert!(report.total_work_seconds > 0.0);
                Ok(())
            },
        )
        .expect("public encryption");
        let mut result = api::online_eval(
            compiled.program(),
            &ParamEnv::default(),
            &online,
            &encrypted,
            RnsOnlineBuild::new(&parameters, [0x4e; 32]),
            &mut backend,
            &mut store,
            config.clone(),
            |_, outputs| Ok(outputs),
            |validated| {
                estimate(validated, &mut estimator).expect("estimate online graph");
                estimator.measure_collected().expect("measure online graph");
                let report = estimate(validated, &mut estimator).expect("online estimate");
                assert!(report.total_work_seconds > 0.0);
                Ok(())
            },
        )
        .expect("public online evaluation");

        let expected_bit = usize::from(!(assignment[0] && assignment[1]));
        let output_wire = match compiled.program().outputs().last().expect("NAND output") {
            Operand::Scalar(wire) => *wire,
            Operand::Member { .. } => panic!("NAND output must be scalar"),
        };
        let workload = RnsBggDerivedWorkload::from_program(compiled.program())
            .expect("NAND workload derivation");
        let output_level = workload.level(output_wire).expect("NAND output workload level");
        let output_ring = parameters.ring(output_level).expect("NAND output ring");
        let expected = layout
            .mono(&output_ring, &BigInt::from(expected_bit))
            .expect("NAND output exponent monomial");
        assert_eq!(expected.matrix_type(), &output_ring.matrix_type((1, 1)));

        let output_names = result
            .outputs
            .keys()
            .filter(|name| name.starts_with("output-0."))
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(output_names.len(), 2, "online output must expose both encoded row ports");
        let mut online_rows = Vec::new();
        for name in output_names {
            let value = result
                .materialize_output(&name, &mut backend, &mut store)
                .expect("materialize NAND evaluated encoding");
            let RuntimeValue::Matrix(_) = value else {
                panic!("NAND evaluated encoding output must be a matrix");
            };
            online_rows.push(value.clone());
        }

        let manifest = store
            .load_manifest(preprocess.checkpoint().production())
            .expect("load persisted preprocess manifest");
        let public_prefix = format!("public-key/{output_wire:?}");
        let mut public_ports = Vec::new();
        for port in 0..4 {
            let name = format!("{public_prefix}.{port}");
            let descriptor =
                manifest.artifacts.get(&name).expect("persisted public-key port").clone();
            let payload = store
                .load(
                    &ArtifactKey {
                        production: preprocess.checkpoint().production().clone(),
                        name,
                        index: None,
                    },
                    &descriptor,
                )
                .expect("load persisted public-key port");
            let value = decode_artifact(&mut backend, descriptor.artifact_type, payload)
                .expect("materialize persisted public-key port");
            let RuntimeValue::Matrix(_) = value else {
                panic!("persisted public-key port must be a matrix");
            };
            public_ports.push(value);
        }

        assert!(matches!(
            compiled.program().wire_knowledge(output_wire).expect("NAND output knowledge"),
            AttributeKnowledge::PrivatePayload
        ));
        let columns = parameters.component_columns(output_level).unwrap();
        let public_graph_ports = (0..4)
            .map(|port| output_ring.input(format!("executed-public-key-{port}"), (1, columns)))
            .collect::<Vec<_>>();
        let public_key = RnsBggPublicKey::new(
            output_wire,
            (
                (public_graph_ports[0].clone(), public_graph_ports[1].clone()),
                (public_graph_ports[2].clone(), public_graph_ports[3].clone()),
            ),
        );
        let encoding = RnsBggEncoding::new(
            output_wire,
            (
                output_ring.input("executed-encoding-0", (1, columns)),
                output_ring.input("executed-encoding-1", (1, columns)),
            ),
            None,
        );
        let residual = encoding
            .verification_residual(
                &parameters,
                output_level,
                &public_key,
                &payload,
                &mask,
                expected,
            )
            .expect("compute NAND verification residual from executed carriers");
        let residual_graph = DslContext::new("nand-verification-residual")
            .artifact_output("verification-residual.0", residual.0)
            .expect("NAND residual graph port")
            .artifact_output("verification-residual.1", residual.1)
            .expect("NAND residual graph port")
            .build()
            .expect("NAND residual graph");
        let mut residual_runtime = std::collections::BTreeMap::new();
        for (port, value) in public_ports.into_iter().enumerate() {
            residual_runtime.insert(format!("executed-public-key-{port}"), value);
        }
        for (port, value) in online_rows.into_iter().enumerate() {
            residual_runtime.insert(format!("executed-encoding-{port}"), value);
        }
        let residual_graph = api::StageGraph::new(
            residual_graph,
            residual_runtime,
            online.checkpoint().catalog().clone(),
        );
        let mut residual_result = api::execute_graph(
            residual_graph,
            &ParamEnv::default(),
            &mut backend,
            &mut store,
            config.clone(),
            |_| Ok(()),
        )
        .expect("execute NAND verification residual");
        for name in ["verification-residual.0", "verification-residual.1"] {
            let value = residual_result
                .materialize_output(name, &mut backend, &mut store)
                .expect("materialize NAND residual");
            let RuntimeValue::Matrix(matrix) = value else {
                panic!("NAND residual output must be a matrix");
            };
            let decoded = backend
                .threshold_decode(
                    matrix,
                    &BigInt::from(2u8),
                    poly_parameters.ring_dimension() as usize,
                )
                .expect("decode NAND residual");
            assert!(
                decoded.iter().all(|value| value == &BigInt::from(0u8)),
                "NAND assignment {assignment:?} residual decoded nonzero: {decoded:?}"
            );
        }
        assert!(expected_bit <= 1);
        assert!(output_bound.error > 0u8.into());
        assert!(!result.outputs.is_empty(), "online evaluation must produce output");
    }
}
