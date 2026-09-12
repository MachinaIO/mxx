pub mod gpu_utils;
pub mod utils;

use gpu_utils::{centered, compile, input, integers, run};
use mxx_dsl::DslContext;
use mxx_fhe::{FheScheme, RingCiphertext};
use mxx_primitives::poly::PolyParams;
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{Signed, Zero};
use rand::Rng;
use serde_json::json;
use std::collections::BTreeMap;

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
        .public_output("a", ct.a.clone())
        .unwrap()
        .public_output("b", ct.b.clone())
        .unwrap()
        .public_output("ga", gsw.a.clone())
        .unwrap()
        .public_output("gb", gsw.b.clone())
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
    let mut backend = gpu_utils::backend(common, None);
    let (keygen_secret, _) = scheme.keygen().unwrap();
    let graph = compile(
        DslContext::new("integration-ring-gsw-keygen")
            .private_output("sk", keygen_secret)
            .unwrap()
            .build()
            .unwrap(),
        &mut backend,
    );
    let (keys, keygen_seconds) = run(&graph, &mut backend, BTreeMap::new());
    println!("FHE_PROGRESS scheme=ring_gsw stage=keygen seconds={keygen_seconds}");
    let encryption = compile(encryption, &mut backend);
    let evaluator = compile(
        DslContext::new("integration-ring-gsw-external-product")
            .output("a", product.a.clone())
            .unwrap()
            .output("b", product.b.clone())
            .unwrap()
            .build()
            .unwrap(),
        &mut backend,
    );
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
        let (evaluated, _) = run(&evaluator, &mut backend, encrypted.clone());
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
            let decryption = compile(
                DslContext::new(format!("integration-ring-gsw-check-{name}"))
                    .private_output(
                        "decoded",
                        scheme.decrypt(&secret, &imported).unwrap().coefficients(),
                    )
                    .unwrap()
                    .private_output("phase", phase.coefficients())
                    .unwrap()
                    .build()
                    .unwrap(),
                &mut backend,
            );
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
                let (output, seconds) = run(&evaluator, &mut backend, encrypted.clone());
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
        json!({"correct":true,"n":n,"q":common.ring.to_crt().0,"sigma":common.error_sigma,"cutoff":common.error_cutoff.to_string(),"secret_min":common.secret_range.minimum.evaluate(&mxx_ir_core::ParamEnv::default()).unwrap().to_string(),"secret_max":common.secret_range.maximum.evaluate(&mxx_ir_core::ParamEnv::default()).unwrap().to_string(),"security":security,"keygen_seconds":keygen_seconds,"gpu_event_seconds":null,"scale":scheme.scale.to_string(),"base_bits":common.ring.base_bits(),"digits":common.ring.modulus_digits(),"cases":reports,"timing_contract":"host elapsed production execute + resident output retrieval + output result-event wait; inputs and outputs remain GPU-resident; excludes prior-iteration cleanup, output serialization and transfers, input generation, keygen, encrypt, graph validation, decrypt, diagnostics"}),
    );
}
