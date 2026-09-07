pub mod gpu_utils;
pub mod utils;

use gpu_utils::{centered, compile, input, integers, run};
use mxx_dsl::{DslContext, Mat, Ring};
use mxx_fhe::{BgvCiphertext, FheScheme};
use mxx_ir_core::node::IndexRange;
use mxx_primitives::poly::PolyParams;
use num_bigint::{BigInt, BigUint};
use num_traits::{Signed, Zero};
use rand::Rng;
use serde_json::json;
use std::collections::BTreeMap;

#[test]
fn test_gpu_bgv_round_trip() {
    let bgv = utils::bgv_params();
    let common = &bgv.common;
    let n = common.ring.ring_dimension() as usize;
    let ring = mxx_dsl::Ring::new(common.ring.modulus().as_ref().clone(), n);
    let top = common.ring.crt_depth() - 1;
    let steps = utils::modswitch_steps();
    assert!(steps > 0 && steps <= top, "modswitch needs a nonempty proper suffix of Q");
    let (key_params, key_width) = bgv.key_switch_parameters(top).unwrap();
    let q_primes = common.ring.to_crt().0;
    let p_primes =
        key_params.to_crt().0.into_iter().filter(|p| !q_primes.contains(p)).collect::<Vec<_>>();
    let context = DslContext::new("integration-bgv-encrypt");
    let pk = ring.input("pk", (2, 1));
    let lhs = bgv.encrypt(&pk, &context.int_family_input("x", n)).unwrap();
    let rhs = bgv.encrypt(&pk, &context.int_family_input("y", n)).unwrap();
    let encryption = context
        .public_output("lhs", lhs.components.clone())
        .unwrap()
        .public_output("rhs", rhs.components.clone())
        .unwrap()
        .build()
        .unwrap();
    let lhs = BgvCiphertext { components: ring.input("lhs", (2, 1)), ..lhs };
    let rhs = BgvCiphertext { components: ring.input("rhs", (2, 1)), ..rhs };
    let rk = Ring::new(key_params.modulus().as_ref().clone(), n).input("rk", (2, key_width));
    let quadratic = bgv.mul_unrelinearized(&lhs, &rhs).unwrap();
    let relin = bgv.relinearize(&quadratic, &rk).unwrap();
    let switched = bgv.mod_switch_to(&relin, top - steps).unwrap();
    assert_eq!(quadratic.components.matrix_type().rows, 3.into());
    assert_eq!(relin.components.matrix_type().rows, 2.into());
    assert_eq!(relin.correction_factor, quadratic.correction_factor);
    let lower = common.parameters_at(top - steps).unwrap();
    assert_eq!(
        switched.components.matrix_type().modulus,
        BigInt::from(lower.modulus().as_ref().clone()).into()
    );
    let stages = [
        ("lhs", lhs, top, 2),
        ("rhs", rhs, top, 2),
        ("quadratic", quadratic, top, 3),
        ("relinearized", relin, top, 2),
        ("modswitched", switched, top - steps, 2),
    ];
    for (name, ct, _, _) in &stages {
        assert!(
            bgv.can_decrypt(ct).unwrap(),
            "correctness preflight fails at {name}: E={}",
            ct.noise_bound
        );
    }
    let security = utils::security_report(common, Some(&p_primes));
    let mut backend = gpu_utils::backend(common, Some(&bgv));
    let (sk, pk) = bgv.keygen().unwrap();
    let generated_rk = bgv.relinearization_key(&sk, top).unwrap();
    let keygen = compile(
        DslContext::new("integration-bgv-keygen")
            .private_output("sk", sk)
            .unwrap()
            .public_output("pk", pk)
            .unwrap()
            .public_output("rk", generated_rk)
            .unwrap()
            .build()
            .unwrap(),
        &mut backend,
    );
    let (keys, keygen_seconds) = run(&keygen, &mut backend, BTreeMap::new());
    println!("FHE_PROGRESS scheme=bgv stage=keygen seconds={keygen_seconds}");
    let encryption = compile(encryption, &mut backend);
    let t = bgv.plaintext_modulus;
    let mut x = (0..n).map(|_| rand::rng().random_range(0..t) as i64).collect::<Vec<_>>();
    let mut y = (0..n).map(|_| rand::rng().random_range(0..t) as i64).collect::<Vec<_>>();
    if let Some(manifest) = utils::manifest() {
        let read = |name: &str| {
            manifest[name]
                .as_array()
                .expect("manifest slot array")
                .iter()
                .map(|v| v.as_i64().expect("integer slot"))
                .collect::<Vec<_>>()
        };
        x = read("x");
        y = read("y");
        assert_eq!(x.len(), n);
        assert_eq!(y.len(), n);
        assert!(x.iter().chain(&y).all(|v| *v >= 0 && (*v as u64) < t));
    }
    let (encrypted, encryption_seconds) = run(
        &encryption,
        &mut backend,
        BTreeMap::from([
            ("pk".into(), keys["pk"].clone()),
            ("x".into(), input(&x)),
            ("y".into(), input(&y)),
        ]),
    );
    println!("FHE_PROGRESS scheme=bgv stage=encryption seconds={encryption_seconds}");
    let eval_inputs = BTreeMap::from([
        ("lhs".into(), encrypted["lhs"].clone()),
        ("rhs".into(), encrypted["rhs"].clone()),
        ("rk".into(), keys["rk"].clone()),
    ]);
    let expected_factor =
        q_primes[top - steps + 1..].iter().fold(stages[3].1.correction_factor, |factor, &prime| {
            (u128::from(factor) *
                u128::from(mxx_primitives::utils::mod_inverse(prime % t, t).unwrap()) %
                u128::from(t)) as u64
        });
    assert_eq!(stages[4].1.correction_factor, expected_factor);
    let mut diagnostics = Vec::new();
    let mut stage_values = BTreeMap::new();
    // Separate stage graphs allow all intermediate ciphertexts to be checked;
    // the measured composite graphs below contain only their final output.
    for (name, ct, level, rows) in &stages {
        assert!(
            bgv.can_decrypt(ct).unwrap(),
            "correctness bound fails at {name}: E={}",
            ct.noise_bound
        );
        let graph = compile(
            DslContext::new(format!("integration-bgv-{name}"))
                .output("ct", ct.components.clone())
                .unwrap()
                .build()
                .unwrap(),
            &mut backend,
        );
        let needed = if *name == "lhs" {
            BTreeMap::from([("lhs".into(), encrypted["lhs"].clone())])
        } else if *name == "rhs" {
            BTreeMap::from([("rhs".into(), encrypted["rhs"].clone())])
        } else if *name == "quadratic" {
            BTreeMap::from([
                ("lhs".into(), encrypted["lhs"].clone()),
                ("rhs".into(), encrypted["rhs"].clone()),
            ])
        } else {
            eval_inputs.clone()
        };
        let (evaluated, _) = run(&graph, &mut backend, needed);
        stage_values.insert(*name, evaluated["ct"].clone());
        let p = common.parameters_at(*level).unwrap();
        let ring = Ring::new(p.modulus().as_ref().clone(), n);
        let imported = BgvCiphertext {
            components: ring.input("ct", (*rows, 1)),
            correction_factor: ct.correction_factor,
            noise_bound: ct.noise_bound.clone(),
        };
        let secret = Ring::new(common.ring.modulus().as_ref().clone(), n).input("sk", (1, 1));
        let minus_s = -secret.clone().reduce_modulus(p.modulus().as_ref().clone());
        let row = |m: &Mat, i: usize| {
            m.clone().slice(Some(IndexRange { start: i.into(), end: (i + 1).into() }), None)
        };
        let mut phase = row(&imported.components, 0);
        for i in 1..*rows {
            phase = phase * &minus_s + row(&imported.components, i);
        }
        let decrypt = compile(
            DslContext::new(format!("integration-bgv-check-{name}"))
                .private_output("slots", bgv.decrypt(&secret, &imported).unwrap())
                .unwrap()
                .private_output("phase", phase.coefficients())
                .unwrap()
                .build()
                .unwrap(),
            &mut backend,
        );
        let (decoded, _) = run(
            &decrypt,
            &mut backend,
            BTreeMap::from([
                ("ct".into(), evaluated["ct"].clone()),
                ("sk".into(), keys["sk"].clone()),
            ]),
        );
        let expected = x
            .iter()
            .zip(&y)
            .map(|(&a, &b)| {
                BigInt::from(match *name {
                    "lhs" => a as u64,
                    "rhs" => b as u64,
                    _ => (a as u128 * b as u128 % t as u128) as u64,
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(integers(&decoded, "slots"), expected, "all slots at {name}");
        let mut maximum = BigUint::zero();
        for value in integers(&decoded, "phase") {
            let v = centered(&value, p.modulus().as_ref());
            let representative = centered(&v, &BigUint::from(t));
            let noise = ((v - representative) / BigInt::from(t)).abs().to_biguint().unwrap();
            maximum = maximum.max(noise);
        }
        assert!(maximum <= ct.noise_bound, "phase noise at {name}: {maximum} > {}", ct.noise_bound);
        println!("FHE_PROGRESS scheme=bgv stage={name} correctness=pass observed_noise={maximum}");
        diagnostics.push(json!({"stage":name,"level":level,"components":rows,"correction_factor":ct.correction_factor,"noise_bound":ct.noise_bound.to_string(),"observed_noise":maximum.to_string()}));
    }
    // Compare every sequential level drop with the former centered-correction
    // expression, including exact ciphertext components rather than plaintext alone.
    let imported = BgvCiphertext { components: ring.input("ct", (2, 1)), ..stages[3].1.clone() };
    let scalar = |q: &BigUint, value: BigInt| Ring::new(q.clone(), n).polynomial([value.into()]);
    let mut reference = imported.components.clone();
    for level in (1..=top).rev() {
        let source = common.parameters_at(level).unwrap();
        let dest = common.parameters_at(level - 1).unwrap();
        let p = *source.to_crt().0.last().unwrap();
        let prime = BigUint::from(p);
        let inverse_t = mxx_primitives::utils::mod_inverse(t % p, p).unwrap();
        let correction = (reference.clone().reduce_modulus(p) *
            scalar(&prime, BigInt::from(p - inverse_t)))
        .centered_rebase(dest.modulus().as_ref().clone());
        let inverse_p =
            mxx_primitives::utils::mod_inverse_biguints(&prime, dest.modulus().as_ref()).unwrap();
        reference = (reference.reduce_modulus(dest.modulus().as_ref().clone()) +
            correction * scalar(dest.modulus().as_ref(), BigInt::from(t))) *
            scalar(dest.modulus().as_ref(), BigInt::from(inverse_p));
        let actual = bgv.mod_switch_to(&imported, level - 1).unwrap();
        let graph = compile(
            DslContext::new(format!("integration-bgv-modswitch-reference-{level}"))
                .output("reference", reference.clone())
                .unwrap()
                .output("actual", actual.components)
                .unwrap()
                .build()
                .unwrap(),
            &mut backend,
        );
        let (output, _) = run(
            &graph,
            &mut backend,
            BTreeMap::from([("ct".into(), stage_values["relinearized"].clone())]),
        );
        assert_eq!(
            gpu_utils::matrix_bytes(&output["actual"], &backend),
            gpu_utils::matrix_bytes(&output["reference"], &backend),
            "exact centered correction through level {}",
            level - 1,
        );
    }
    let mut timings = Vec::new();
    for (label, index) in
        [("multiply", 2), ("multiply_relinearize", 3), ("multiply_relinearize_modswitch", 4)]
    {
        let graph = compile(
            DslContext::new(format!("integration-bgv-timing-{label}"))
                .output("ct", stages[index].1.components.clone())
                .unwrap()
                .build()
                .unwrap(),
            &mut backend,
        );
        let expected_bytes = gpu_utils::matrix_bytes(&stage_values[stages[index].0], &backend);
        let (warmup, _) = run(&graph, &mut backend, eval_inputs.clone());
        assert!(
            gpu_utils::matrix_bytes(&warmup["ct"], &backend) == expected_bytes,
            "warmup replay {label}"
        );
        let samples = (0..utils::repetitions())
            .map(|sample| {
                let (output, seconds) = run(&graph, &mut backend, eval_inputs.clone());
                assert!(
                    gpu_utils::matrix_bytes(&output["ct"], &backend) == expected_bytes,
                    "replay {label} sample {sample}"
                );
                seconds
            })
            .collect::<Vec<_>>();
        println!(
            "FHE_PROGRESS scheme=bgv operation={label} measured_samples={} correctness=pass",
            samples.len()
        );
        timings.push(json!({"operation":label,"seconds":samples}));
    }
    for (label, index, rows) in [("relinearize", 2, 3), ("modswitch", 3, 2)] {
        let imported = BgvCiphertext {
            components: ring.input("ct", (rows, 1)),
            correction_factor: stages[index].1.correction_factor,
            noise_bound: stages[index].1.noise_bound.clone(),
        };
        let output = if label == "relinearize" {
            bgv.relinearize(&imported, &rk).unwrap()
        } else {
            bgv.mod_switch_to(&imported, top - steps).unwrap()
        };
        let graph = compile(
            DslContext::new(format!("integration-bgv-timing-{label}"))
                .output("ct", output.components)
                .unwrap()
                .build()
                .unwrap(),
            &mut backend,
        );
        let mut inputs = BTreeMap::from([("ct".into(), stage_values[stages[index].0].clone())]);
        if label == "relinearize" {
            inputs.insert("rk".into(), keys["rk"].clone());
        }
        let expected_bytes = gpu_utils::matrix_bytes(&stage_values[stages[index + 1].0], &backend);
        let (warmup, _) = run(&graph, &mut backend, inputs.clone());
        assert!(
            gpu_utils::matrix_bytes(&warmup["ct"], &backend) == expected_bytes,
            "warmup replay {label}"
        );
        let samples = (0..utils::repetitions())
            .map(|sample| {
                let (output, seconds) = run(&graph, &mut backend, inputs.clone());
                assert!(
                    gpu_utils::matrix_bytes(&output["ct"], &backend) == expected_bytes,
                    "replay {label} sample {sample}"
                );
                seconds
            })
            .collect::<Vec<_>>();
        println!(
            "FHE_PROGRESS scheme=bgv operation={label} measured_samples={} correctness=pass",
            samples.len()
        );
        timings.push(json!({"operation":label,"seconds":samples}));
    }
    utils::write_report(
        "bgv",
        json!({"correct":true,"n":n,"q":q_primes,"p":p_primes,"t":t,"key_columns":key_width,"digit_size":utils::bgv_digit_size(),"gpu_event_seconds":null,"sigma":common.error_sigma,"cutoff":common.error_cutoff.to_string(),"base_bits":common.ring.base_bits(),"security":security,"keygen_seconds":keygen_seconds,"encryption_seconds":encryption_seconds,"stages":diagnostics,"timings":timings,"x":x,"y":y,"modswitch_steps":steps,"timing_contract":"host elapsed production execute + resident output retrieval + output result-event wait; inputs and outputs remain GPU-resident; excludes prior-iteration cleanup, output serialization and transfers, input generation, keygen, encrypt, graph validation, decrypt, diagnostics"}),
    );
}
