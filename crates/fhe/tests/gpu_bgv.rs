//! A minimal BGV round-trip through the production GPU runtime.

use mxx_dsl::{DslContext, Ring};
use mxx_fhe::{
    BgvCiphertext, FheScheme,
    utils::{
        self,
        gpu::{self, input, integers},
    },
};
use mxx_ir_core::ParamEnv;
use mxx_primitives::poly::PolyParams;
use mxx_runtime::{GpuRuntime, MemoryArtifactStore, backend::poly_gpu::gpu_backend};
use num_bigint::BigInt;
use rand::Rng;
use std::{collections::BTreeMap, time::Instant};

/// Record one production execution interval at the `GpuRuntime::execute` return boundary.
/// The runtime contract already waits for the returned resident outputs; this helper deliberately
/// adds no caller-side CUDA synchronization or matrix fence.
fn record_timing(stage: &str, started: Instant, timings: &mut Vec<(String, f64)>) {
    let milliseconds = started.elapsed().as_secs_f64() * 1_000.0;
    println!("BGV_TIMING stage={stage} elapsed_ms={milliseconds:.3}");
    timings.push((stage.to_owned(), milliseconds));
}

#[test]
fn test_gpu_bgv_round_trip() {
    let mut timings = Vec::new();
    let bgv = utils::bgv_params();
    let common = &bgv.common;
    let n = common.ring.ring_dimension() as usize;
    let top = common.ring.crt_depth() - 1;
    let lower_level = top
        .checked_sub(utils::modswitch_steps())
        .expect("modswitch steps must fit within the ciphertext levels");
    let (key_params, key_width) = bgv.key_switch_parameters(top).unwrap();
    let ring = Ring::new(common.ring.modulus().as_ref().clone(), n);

    let plaintext_modulus = bgv.plaintext_modulus;
    let x =
        (0..n).map(|_| rand::rng().random_range(0..plaintext_modulus) as i64).collect::<Vec<_>>();
    let y =
        (0..n).map(|_| rand::rng().random_range(0..plaintext_modulus) as i64).collect::<Vec<_>>();

    let gpu_parameters = gpu::bgv_gpu_parameters(&bgv);
    let backend = gpu_backend(gpu_parameters.iter().cloned());
    let mut runtime = GpuRuntime::new(backend).expect("construct BGV GPU runtime");

    // A DSL program describes values to produce. Validate it, freeze a GPU
    // plan, execute it, and forward its resident outputs to the next program.
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
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let mut keygen_plan = runtime.plan(keygen_graph, &BTreeMap::new()).unwrap();
    let mut keygen_store = MemoryArtifactStore::default();
    let started = Instant::now();
    let keygen_result =
        runtime.execute(&mut keygen_plan, BTreeMap::new(), &mut keygen_store, [0; 32]).unwrap();
    record_timing("keygen", started, &mut timings);
    let keys = keygen_result.outputs;

    // Build two encryption programs so the x and y production execution boundaries are measured
    // independently. Graph construction and plan compilation happen before each timer starts.
    let pk_input = ring.input("pk", (2, 1));
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
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let lhs_inputs = BTreeMap::from([("pk".into(), keys["pk"].clone()), ("x".into(), input(&x))]);
    let mut lhs_plan = runtime.plan(encryption_graph_x, &lhs_inputs).unwrap();
    let mut lhs_store = MemoryArtifactStore::default();
    let started = Instant::now();
    let lhs_result = runtime.execute(&mut lhs_plan, lhs_inputs, &mut lhs_store, [0; 32]).unwrap();
    record_timing("encrypt_x", started, &mut timings);
    let encrypted_lhs = lhs_result.outputs;

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
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let rhs_inputs = BTreeMap::from([("pk".into(), keys["pk"].clone()), ("y".into(), input(&y))]);
    let mut rhs_plan = runtime.plan(encryption_graph_y, &rhs_inputs).unwrap();
    let mut rhs_store = MemoryArtifactStore::default();
    let started = Instant::now();
    let rhs_result = runtime.execute(&mut rhs_plan, rhs_inputs, &mut rhs_store, [0; 32]).unwrap();
    record_timing("encrypt_y", started, &mut timings);
    let encrypted_rhs = rhs_result.outputs;

    // Build separate evaluation programs so each production operation has one measured execute
    // interval. The input artifacts remain GPU-resident across these returned-output boundaries.
    let lhs = BgvCiphertext { components: ring.input("lhs", (2, 1)), ..lhs_template };
    let rhs = BgvCiphertext { components: ring.input("rhs", (2, 1)), ..rhs_template };
    let rk = Ring::new(key_params.modulus().as_ref().clone(), n).input("rk", (2, key_width));
    let quadratic = bgv.mul_unrelinearized(&lhs, &rhs).unwrap();
    let multiply_graph = DslContext::new("bgv-round-trip-multiply")
        .output("quadratic", quadratic.components.clone())
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let multiply_inputs = BTreeMap::from([
        ("lhs".into(), encrypted_lhs["lhs"].clone()),
        ("rhs".into(), encrypted_rhs["rhs"].clone()),
    ]);
    let mut multiply_plan = runtime.plan(multiply_graph, &multiply_inputs).unwrap();
    let mut multiply_store = MemoryArtifactStore::default();
    let started = Instant::now();
    let multiply_result =
        runtime.execute(&mut multiply_plan, multiply_inputs, &mut multiply_store, [0; 32]).unwrap();
    record_timing("multiply", started, &mut timings);

    let quadratic_input = BgvCiphertext {
        components: ring.input("quadratic", (3, 1)),
        correction_factor: quadratic.correction_factor,
        noise_bound: quadratic.noise_bound.clone(),
    };
    let relinearized = bgv.relinearize(&quadratic_input, &rk).unwrap();
    let relinearize_graph = DslContext::new("bgv-round-trip-relinearize")
        .output("relinearized", relinearized.components.clone())
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let relinearize_inputs = BTreeMap::from([
        ("quadratic".into(), multiply_result.outputs["quadratic"].clone()),
        ("rk".into(), keys["rk"].clone()),
    ]);
    let mut relinearize_plan = runtime.plan(relinearize_graph, &relinearize_inputs).unwrap();
    let mut relinearize_store = MemoryArtifactStore::default();
    let started = Instant::now();
    let relinearize_result = runtime
        .execute(&mut relinearize_plan, relinearize_inputs, &mut relinearize_store, [0; 32])
        .unwrap();
    record_timing("relinearize", started, &mut timings);

    let relinearized_input = BgvCiphertext {
        components: ring.input("relinearized", (2, 1)),
        correction_factor: relinearized.correction_factor,
        noise_bound: relinearized.noise_bound.clone(),
    };
    let switched = bgv.mod_switch_to(&relinearized_input, lower_level).unwrap();
    let modswitch_graph = DslContext::new("bgv-round-trip-modswitch")
        .output("ct", switched.components.clone())
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let modswitch_inputs = BTreeMap::from([(
        "relinearized".into(),
        relinearize_result.outputs["relinearized"].clone(),
    )]);
    let mut modswitch_plan = runtime.plan(modswitch_graph, &modswitch_inputs).unwrap();
    let mut modswitch_store = MemoryArtifactStore::default();
    let started = Instant::now();
    let evaluation_result = runtime
        .execute(&mut modswitch_plan, modswitch_inputs, &mut modswitch_store, [0; 32])
        .unwrap();
    record_timing("modswitch", started, &mut timings);

    // Decryption is another DSL program: bind the evaluated ciphertext and
    // secret key, execute it, and inspect the resulting plaintext family.
    let secret = Ring::new(common.ring.modulus().as_ref().clone(), n).input("sk", (1, 1));
    let lower = common.parameters_at(lower_level).unwrap();
    let imported = BgvCiphertext {
        components: Ring::new(lower.modulus().as_ref().clone(), n).input("ct", (2, 1)),
        correction_factor: switched.correction_factor,
        noise_bound: switched.noise_bound,
    };
    let decryption_graph = DslContext::new("bgv-round-trip-decryption")
        .output("slots", bgv.decrypt(&secret, &imported).unwrap())
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let decryption_inputs = BTreeMap::from([
        ("ct".into(), evaluation_result.outputs["ct"].clone()),
        ("sk".into(), keys["sk"].clone()),
    ]);
    let mut decryption_plan = runtime.plan(decryption_graph, &decryption_inputs).unwrap();
    let mut decryption_store = MemoryArtifactStore::default();
    let started = Instant::now();
    let decryption_result = runtime
        .execute(&mut decryption_plan, decryption_inputs, &mut decryption_store, [0; 32])
        .unwrap();
    record_timing("decrypt", started, &mut timings);
    let expected = x
        .iter()
        .zip(&y)
        .map(|(&a, &b)| BigInt::from((a as u128 * b as u128 % plaintext_modulus as u128) as u64))
        .collect::<Vec<_>>();
    assert_eq!(integers(runtime.backend_mut(), &decryption_result.outputs, "slots"), expected);
    let total_ms = timings.iter().map(|(_, milliseconds)| milliseconds).sum::<f64>();
    println!("BGV_TIMING_SUMMARY total_execute_ms={total_ms:.3} stages={}", timings.len());
}
