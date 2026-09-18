#![cfg(feature = "gpu")]

//! Public RNS-BGG NAND application workflow.

use bigdecimal::BigDecimal;
use mxx_bench_estimator::{
    estimate, gpu::GpuNodeMeasurementBackend, harness::MeasurementHarnessConfig,
};
use mxx_exponent_lut::{
    ExpLayout, NandCircuit, api,
    program::{InputBinding, RhsId},
    rns_bgg::{
        RnsBggDerivedWorkload, RnsBggNoiseBindings, RnsBggNoiseRhs, RnsBggNoiseValue,
        RnsBggParameters, RnsBggPayloadSecret, RnsEncryptBuild, RnsNoiseBuild, RnsOnlineBuild,
        RnsPreprocessBuild, verify_output,
    },
};
use mxx_ir_core::ParamEnv;
use mxx_primitives::poly::{
    Poly, PolyParams,
    dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams, poly::DCRTPoly},
};
use mxx_runtime::{
    artifact::MemoryArtifactStore,
    backend::poly::gpu::{GpuDcrtBackend, gpu_backend_on},
};
use num_bigint::{BigInt, BigUint};
use serial_test::serial as sequential;
use std::{env, path::Path, process::Command};

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

fn lattice_security_check(parameters: &DCRTPolyParams, error_sigma: &BigDecimal) {
    if env::var("MXX_EXPONENT_LUT_NAND_SKIP_SAGE").as_deref() == Ok("1") {
        eprintln!("lattice_security_check=skipped");
        return;
    }

    let required = env::var("MXX_EXPONENT_LUT_NAND_SECURITY_BITS")
        .map(|value| {
            value.parse::<u64>().unwrap_or_else(|_| {
                panic!("MXX_EXPONENT_LUT_NAND_SECURITY_BITS must be an unsigned integer")
            })
        })
        .unwrap_or(0);
    let output = Command::new("lattice-estimator-cli")
        .arg(parameters.ring_dimension().to_string())
        .arg(parameters.modulus().to_string())
        .arg("--s-dist")
        .arg(r#"{"name":"Ternary"}"#)
        .arg("--e-dist")
        .arg(format!(r#"{{"name":"DiscreteGaussian","stddev":{error_sigma}}}"#))
        .output()
        .expect("lattice-estimator-cli is required unless MXX_EXPONENT_LUT_NAND_SKIP_SAGE=1");
    assert!(
        output.status.success(),
        "lattice-estimator-cli failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bits = String::from_utf8_lossy(&output.stdout)
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .expect("lattice-estimator-cli returned no security estimate")
        .trim()
        .parse::<u64>()
        .expect("lattice-estimator-cli returned a non-integer security estimate");
    assert!(
        bits >= required,
        "NAND parameters provide {bits} security bits, below the required {required}"
    );
    eprintln!(
        "lattice_security_check=passed security_bits={bits} required_security_bits={required}"
    );
}

fn checkpoint(path: &Path, name: &str) -> std::path::PathBuf {
    path.join(name)
}

#[test]
#[sequential(dcrt_runtime)]
fn nand_circuit_end_to_end() {
    let devices = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids();
    if devices.is_empty() {
        eprintln!("skipping NAND round trip: no CUDA device detected");
        return;
    }
    let circuit = NandCircuit::new(2, vec![[0, 1]], 2).expect("valid NAND circuit");
    let layout = ExpLayout { stride_log2: 0, width_log2: 4 };
    let compiled = circuit.compile(layout, 32, 4).expect("NAND program compiles");
    let (poly_parameters, parameters) = rns_parameters();
    let payload: RnsBggPayloadSecret =
        RnsEncryptBuild::<GpuDcrtBackend>::new(&parameters, [0x4e; 32])
            .sample_payload_secret()
            .expect("payload secret");
    lattice_security_check(&poly_parameters, parameters.encoding_error_sigma());

    for assignment in [[false, false], [false, true], [true, false], [true, true]] {
        let (lhs, rhs) = compiled
            .plaintext_inputs::<mxx_primitives::poly::dcrt::poly::DCRTPoly>(
                &poly_parameters,
                &assignment,
            )
            .expect("NAND plaintext bindings");
        let mask = RnsEncryptBuild::<GpuDcrtBackend>::new(&parameters, rand::random())
            .sample_mask_secret()
            .expect("fresh mask secret");
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
            HASH_KEY,
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
            HASH_KEY,
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

        let expected_bit = circuit.evaluate(&assignment).expect("NAND Boolean evaluation");
        assert!(!result.outputs.is_empty(), "online evaluation must produce output");
        let expected = DCRTPoly::const_rotate_poly(&poly_parameters, usize::from(expected_bit));
        verify_output(
            compiled.program(),
            &mut result,
            0,
            &expected,
            &parameters,
            &payload,
            &mask,
            &mut backend,
            &mut store,
            config,
        )
        .expect("NAND application verification");
        assert!(output_bound.error > 0u8.into());
    }
}
