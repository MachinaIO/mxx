pub mod gpu_utils;
pub mod utils;

use gpu_utils::{centered, input, integers};
use mxx_dsl::{BuiltGraph, DslContext};
use mxx_fhe::{FheCommonParams, FheScheme, RingCiphertext};
use mxx_ir_core::ParamEnv;
use mxx_primitives::poly::{
    PolyParams,
    dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams},
};
use mxx_runtime::{
    ExecutionConfig, MemoryArtifactStore, RuntimeValue,
    backend::poly_gpu::{GpuDcrtBackend, gpu_backend},
    gpu_measurement::{
        GpuPreparationRequest, GpuWarmupMeasurementConfig, PreparedGpuExecution,
        prepare as prepare_gpu,
    },
};
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{Signed, Zero};
use rand::Rng;
use serde_json::json;
use std::{collections::BTreeMap, time::Instant};

fn ring_gsw_ring_parameters(common: &FheCommonParams) -> Vec<DCRTPolyParams> {
    let (primes, _, depth) = common.ring.to_crt();
    let mut rings =
        (0..depth).map(|level| common.parameters_at(level).unwrap()).collect::<Vec<_>>();
    rings.extend(primes.iter().map(|p| common.ring.select_modulus(&BigUint::from(*p)).unwrap()));
    // The first ordered prefix is also the first single-prime correction
    // parameter. Register each logical ring only once: the GPU backend maps
    // parameters by their CRT layout, so duplicate entries make placement
    // ambiguous even when they share a context.
    let mut unique_rings = Vec::with_capacity(rings.len());
    for ring in rings {
        if !unique_rings.contains(&ring) {
            unique_rings.push(ring);
        }
    }
    unique_rings
}

fn ring_gsw_parameters(common: &FheCommonParams) -> Vec<GpuDCRTPolyParams> {
    let rings = ring_gsw_ring_parameters(common);
    let mut parameters: Vec<GpuDCRTPolyParams> = Vec::with_capacity(rings.len());
    for ring in rings {
        let parameter = if let Some(related) = parameters.first() {
            GpuDCRTPolyParams::new_with_gpu(
                ring.ring_dimension(),
                ring.to_crt().0,
                ring.base_bits(),
                related.gpu_ids().to_vec(),
                Some(1),
                Some(related),
                None,
            )
        } else {
            GpuDCRTPolyParams::new(ring.ring_dimension(), ring.to_crt().0, ring.base_bits(), None)
        };
        parameters.push(parameter);
    }
    parameters
}

#[test]
fn test_ring_gsw_parameters_deduplicate_logical_rings() {
    let common = utils::ring_gsw_params().common;
    let rings = ring_gsw_ring_parameters(&common);
    let (primes, _, depth) = common.ring.to_crt();
    assert!(rings.len() < depth + primes.len());
    assert!(rings.iter().enumerate().all(|(index, ring)| { !rings[..index].contains(ring) }));
}

fn prepare_graph(
    graph: BuiltGraph,
    backend: &mut GpuDcrtBackend,
    inputs: &gpu_utils::Inputs,
    parameters: &[GpuDCRTPolyParams],
) -> PreparedGpuExecution {
    let validated = graph.validate(&ParamEnv::default()).expect("valid integration graph");
    // Each preparation starts a new fixed-plan epoch.  The preceding graph
    // may have left a frozen plan and asynchronous releases on this shared
    // backend (keygen, encryption, evaluation, and decryption reuse it).
    backend.clear_frozen_plan();
    assert!(!backend.fixed_dispatch_enabled(), "preparation must start without a frozen plan");
    mxx_runtime::backend::Backend::fence_released_memory(backend)
        .expect("finish prior setup releases");
    let prepared = prepare_gpu(GpuPreparationRequest {
        validated,
        backend,
        inputs,
        parameters,
        default_tile_widths: vec![1, 2, 4, 8],
        implementation_variant: "ring-gsw-production-measured".into(),
        measurement_config: GpuWarmupMeasurementConfig::default(),
        execution_config: ExecutionConfig::default(),
    })
    .expect("production-equivalent GPU warmup");
    prepared.validate_evidence().expect("complete production-equivalent GPU evidence");
    prepared
}

/// Includes production execution, output retrieval and result-event completion.
fn run(
    prepared: &PreparedGpuExecution,
    backend: &mut GpuDcrtBackend,
    inputs: gpu_utils::Inputs,
) -> (gpu_utils::Inputs, f64) {
    let mut store = MemoryArtifactStore::default();
    mxx_runtime::backend::Backend::fence_released_memory(backend)
        .expect("complete prior-iteration GPU cleanup");
    let start = Instant::now();
    let mut result = prepared.run(backend, inputs, &mut store, [0; 32]).expect("GPU execution");
    for name in result.outputs.keys().cloned().collect::<Vec<_>>() {
        if let RuntimeValue::Matrix(matrix) =
            result.materialize_output(&name, backend, &mut store).expect("materialize GPU output")
        {
            matrix.wait_until_ready();
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    result.cleanup_staged(&mut store).unwrap();
    prepared.assert_measurements_unchanged();
    (result.outputs, seconds)
}

#[test]
fn test_gpu_ring_gsw_round_trip() {
    let scheme = utils::ring_gsw_params();
    let common = &scheme.common;
    let n = common.ring.ring_dimension() as usize;
    let ring = mxx_dsl::Ring::new(common.ring.modulus().as_ref().clone(), n);
    let q = common.ring.modulus();
    let context = DslContext::new("integration-ring-gsw-encrypt");
    let secret = ring.input("sk", (1, 1));
    let message = ring.from_coefficients(&context.int_family_input("message", n));
    let bit = ring.from_coefficients(&context.int_family_input("bit", n));
    let ct = scheme.encrypt(&secret, &message).unwrap();
    let gsw = scheme.encrypt_gsw(&secret, &bit).unwrap();
    let encryption = context
        .transferred_output("a", ct.a.clone())
        .unwrap()
        .transferred_output("b", ct.b.clone())
        .unwrap()
        .transferred_output("ga", gsw.a.clone())
        .unwrap()
        .transferred_output("gb", gsw.b.clone())
        .unwrap()
        .build()
        .unwrap();
    let ct = RingCiphertext { a: ring.input("a", (1, 1)), b: ring.input("b", (1, 1)), ..ct };
    let width = 2 * common.ring.modulus_digits();
    let gsw =
        RingCiphertext { a: ring.input("ga", (1, width)), b: ring.input("gb", (1, width)), ..gsw };
    let product = scheme.external_product(&gsw, &ct).unwrap();
    assert!(scheme.can_decrypt(&ct), "fresh correctness bound");
    assert!(
        scheme.can_decrypt(&product),
        "external-product correctness bound: E={}, M={}, delta={}",
        product.noise_bound,
        product.plaintext_bound,
        scheme.scale
    );
    let security = utils::security_report(common, None);
    let parameters = ring_gsw_parameters(common);
    let first_context =
        parameters.first().expect("Ring-GSW parameter construction produced no levels");
    assert!(
        parameters.iter().all(|parameter| parameter.context_execution_identity() ==
            first_context.context_execution_identity()),
        "all Ring-GSW levels must share the first GPU context"
    );
    assert!(
        parameters.iter().all(|parameter| parameter.gpu_ids() == first_context.gpu_ids()),
        "all Ring-GSW levels must share the first GPU placement"
    );
    let mut backend = gpu_backend(parameters.iter().cloned());
    let (keygen_secret, _) = scheme.keygen().unwrap();
    let keygen = prepare_graph(
        DslContext::new("integration-ring-gsw-keygen")
            .transferred_output("sk", keygen_secret)
            .unwrap()
            .build()
            .unwrap(),
        &mut backend,
        &BTreeMap::new(),
        &parameters,
    );
    let (keys, keygen_seconds) = run(&keygen, &mut backend, BTreeMap::new());
    let keygen_report = keygen.report();
    let mut protocol_predicted_seconds = keygen_report.predicted_seconds;
    let mut warmup_steps = vec![json!({
        "step": "keygen",
        "predicted_seconds": keygen_report.predicted_seconds,
        "report": keygen_report,
    })];
    println!("FHE_PROGRESS scheme=ring_gsw stage=keygen seconds={keygen_seconds}");
    let encryption_graph = encryption;
    let evaluator_graph = DslContext::new("integration-ring-gsw-external-product")
        .output("a", product.a.clone())
        .unwrap()
        .output("b", product.b.clone())
        .unwrap()
        .build()
        .unwrap();
    let encryption = prepare_graph(
        encryption_graph,
        &mut backend,
        &BTreeMap::from([
            ("sk".into(), keys["sk"].clone()),
            ("message".into(), input(&vec![0; n])),
            ("bit".into(), input(&vec![0; n])),
        ]),
        &parameters,
    );
    let encryption_report = encryption.report();
    protocol_predicted_seconds += encryption_report.predicted_seconds;
    warmup_steps.push(json!({
        "step": "encryption",
        "predicted_seconds": encryption_report.predicted_seconds,
        "report": encryption_report,
    }));
    let mut evaluator_graph = Some(evaluator_graph);
    let mut evaluator = None;
    let mut reports = Vec::new();
    // Both bits are mandatory; an additional independently sampled bit exercises
    // the requested randomized operand distribution on every invocation.
    for bit_value in [0i64, 1, rand::rng().random_range(0..=1)] {
        let exponent = rand::rng().random_range(0..2 * n);
        let mut message = vec![0i64; n];
        message[exponent % n] = if exponent < n { 1 } else { -1 };
        let mut bit = vec![0i64; n];
        bit[0] = bit_value;
        let (encrypted, encryption_seconds) = run(
            &encryption,
            &mut backend,
            BTreeMap::from([
                ("sk".into(), keys["sk"].clone()),
                ("message".into(), input(&message)),
                ("bit".into(), input(&bit)),
            ]),
        );
        println!(
            "FHE_PROGRESS scheme=ring_gsw stage=encryption bit={bit_value} exponent={exponent} seconds={encryption_seconds}"
        );
        if evaluator.is_none() {
            let prepared = prepare_graph(
                evaluator_graph.take().expect("evaluator graph is prepared once"),
                &mut backend,
                &encrypted,
                &parameters,
            );
            let report = prepared.report();
            protocol_predicted_seconds += report.predicted_seconds;
            warmup_steps.push(json!({
                "step": "external_product",
                "predicted_seconds": report.predicted_seconds,
                "report": report,
            }));
            evaluator = Some(prepared);
        }
        let (evaluated, _) = run(evaluator.as_ref().unwrap(), &mut backend, encrypted.clone());
        let mut stages = Vec::new();
        for (name, metadata, values, multiplier) in [
            ("fresh", &ct, &encrypted, 1i64),
            ("external_product", &product, &evaluated, bit_value),
        ] {
            let imported = RingCiphertext {
                a: ring.input("a", (1, 1)),
                b: ring.input("b", (1, 1)),
                noise_bound: metadata.noise_bound.clone(),
                plaintext_bound: metadata.plaintext_bound.clone(),
            };
            let phase = &imported.b - &secret * &imported.a;
            let decryption = prepare_graph(
                DslContext::new(format!("integration-ring-gsw-check-{name}"))
                    .transferred_output(
                        "decoded",
                        scheme.decrypt(&secret, &imported).unwrap().coefficients(),
                    )
                    .unwrap()
                    .transferred_output("phase", phase.coefficients())
                    .unwrap()
                    .build()
                    .unwrap(),
                &mut backend,
                &BTreeMap::from([
                    ("sk".into(), keys["sk"].clone()),
                    ("a".into(), values["a"].clone()),
                    ("b".into(), values["b"].clone()),
                ]),
                &parameters,
            );
            let decrypt_report = decryption.report();
            protocol_predicted_seconds += decrypt_report.predicted_seconds;
            warmup_steps.push(json!({
                "step": format!("decryption-{name}"),
                "predicted_seconds": decrypt_report.predicted_seconds,
                "report": decrypt_report,
            }));
            let (decoded, _) = run(
                &decryption,
                &mut backend,
                BTreeMap::from([
                    ("sk".into(), keys["sk"].clone()),
                    ("a".into(), values["a"].clone()),
                    ("b".into(), values["b"].clone()),
                ]),
            );
            let expected = message
                .iter()
                .map(|v| BigInt::from(v * multiplier).mod_floor(&BigInt::from(q.as_ref().clone())))
                .collect::<Vec<_>>();
            assert_eq!(
                integers(&decoded, "decoded"),
                expected,
                "all coefficients for bit={bit_value}, exponent={exponent}, stage={name}"
            );
            let mut maximum = BigUint::zero();
            for (value, plain) in integers(&decoded, "phase").iter().zip(&message) {
                let residual = centered(value, q.as_ref()) -
                    BigInt::from(plain * multiplier) * BigInt::from(scheme.scale.clone());
                maximum = maximum.max(residual.abs().to_biguint().unwrap());
            }
            assert!(
                maximum <= metadata.noise_bound,
                "observed {maximum} exceeds bound {} at {name}",
                metadata.noise_bound
            );
            println!(
                "FHE_PROGRESS scheme=ring_gsw stage={name} bit={bit_value} correctness=pass observed_noise={maximum}"
            );
            stages.push(json!({"stage":name,"noise_bound":metadata.noise_bound.to_string(),"plaintext_bound":metadata.plaintext_bound.to_string(),"observed_noise":maximum.to_string()}));
        }
        // The unmeasured evaluation above warms the identical graph and input.
        let expected_bytes =
            ["a", "b"].map(|name| gpu_utils::matrix_bytes(&evaluated[name], &backend));
        let samples = (0..utils::repetitions())
            .map(|sample| {
                let (output, seconds) =
                    run(evaluator.as_ref().unwrap(), &mut backend, encrypted.clone());
                for (name, expected) in ["a", "b"].into_iter().zip(&expected_bytes) {
                    assert!(
                        gpu_utils::matrix_bytes(&output[name], &backend) == *expected,
                        "replay bit {bit_value}, sample {sample}, component {name}"
                    );
                }
                seconds
            })
            .collect::<Vec<_>>();
        println!(
            "FHE_PROGRESS scheme=ring_gsw operation=external_product bit={bit_value} measured_samples={} correctness=pass",
            samples.len()
        );
        reports.push(json!({"bit":bit_value,"monomial_exponent":exponent,"encryption_seconds":encryption_seconds,"stages":stages,"external_product_seconds":samples}));
    }
    utils::write_report(
        "ring_gsw",
        json!({"correct":true,"n":n,"q":common.ring.to_crt().0,"sigma":common.error_sigma,"cutoff":common.error_cutoff.to_string(),"secret_min":common.secret_range.minimum.evaluate(&mxx_ir_core::ParamEnv::default()).unwrap().to_string(),"secret_max":common.secret_range.maximum.evaluate(&mxx_ir_core::ParamEnv::default()).unwrap().to_string(),"security":security,"keygen_seconds":keygen_seconds,"gpu_event_seconds":null,"scale":scheme.scale.to_string(),"base_bits":common.ring.base_bits(),"digits":common.ring.modulus_digits(),"cases":reports,"warmup_steps":warmup_steps,"protocol_predicted_seconds":protocol_predicted_seconds,"timing_contract":"host elapsed production execute + resident output retrieval + output result-event wait; inputs and outputs remain GPU-resident; excludes prior-iteration cleanup, output serialization and transfers, input generation, keygen, encrypt, graph validation, decrypt, diagnostics"}),
    );
}
