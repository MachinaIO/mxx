//! A minimal BGV round trip through the production GPU runtime: generate
//! keys, encrypt two plaintexts, multiply, relinearize, modulus switch, and
//! decrypt.
//!
//! The executed graphs, linked as one round trip, also form a closed protocol whose correctness
//! claim the test exports to `crates/fhe/lean/bgv/generated`: every execution decrypts to the
//! slotwise product modulo `t`. `lake build` in `crates/fhe/lean/bgv` checks the handwritten proof
//! of that claim.

use mxx_backends::{GpuRuntime, backend::poly_gpu::gpu_backend};
use mxx_fhe::utils::{
    self,
    gpu::{self, input},
    protocol::{BgvRoundTrip, bgv_round_trip},
};
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
    // Log how many times each primitive operation of every graph runs, and
    // each execute's preparation and run times; `RUST_LOG` overrides this
    // filter.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn,gpu_bgv=info,mxx_backends::gpu_execute=debug".into()),
        )
        .with_test_writer()
        .try_init();
    let mut timings = Vec::new();
    let bgv = utils::bgv_params();
    let n = bgv.common.ring.ring_dimension() as usize;
    let BgvRoundTrip {
        keygen_graph,
        encryption_graph_x,
        encryption_graph_y,
        multiply_graph,
        relinearize_graph,
        modswitch_graph,
        decryption_graph,
        protocol,
    } = bgv_round_trip(&bgv, utils::modswitch_steps());

    let plaintext_modulus = bgv.plaintext_modulus;
    let x =
        (0..n).map(|_| rand::rng().random_range(0..plaintext_modulus) as i64).collect::<Vec<_>>();
    let y =
        (0..n).map(|_| rand::rng().random_range(0..plaintext_modulus) as i64).collect::<Vec<_>>();

    let mut runtime = GpuRuntime::new(gpu_backend(gpu::bgv_gpu_parameters(&bgv)))
        .expect("construct BGV GPU runtime");

    // A DSL program describes values to produce. Freeze it into a GPU plan,
    // execute it, and pass its outputs to the next program as inputs.
    tracing::info!(graph = "keygen", operations = ?keygen_graph.operation_counts().unwrap());
    let mut keygen_plan = runtime.plan(keygen_graph, &BTreeMap::new()).unwrap();
    let started = Instant::now();
    let keys = runtime.execute(&mut keygen_plan, BTreeMap::new()).unwrap();
    record_timing("keygen", started, &mut timings);

    // Build two encryption programs so the x and y production execution boundaries are measured
    // independently. Graph construction and plan compilation happen before each timer starts.
    tracing::info!(graph = "encryption_x", operations = ?encryption_graph_x.operation_counts().unwrap());
    let lhs_inputs = BTreeMap::from([("pk".into(), keys["pk"].clone()), ("x".into(), input(&x))]);
    let mut lhs_plan = runtime.plan(encryption_graph_x, &lhs_inputs).unwrap();
    let started = Instant::now();
    let encrypted_lhs = runtime.execute(&mut lhs_plan, lhs_inputs).unwrap();
    record_timing("encrypt_x", started, &mut timings);

    tracing::info!(graph = "encryption_y", operations = ?encryption_graph_y.operation_counts().unwrap());
    let rhs_inputs = BTreeMap::from([("pk".into(), keys["pk"].clone()), ("y".into(), input(&y))]);
    let mut rhs_plan = runtime.plan(encryption_graph_y, &rhs_inputs).unwrap();
    let started = Instant::now();
    let encrypted_rhs = runtime.execute(&mut rhs_plan, rhs_inputs).unwrap();
    record_timing("encrypt_y", started, &mut timings);

    // Build separate evaluation programs so each production operation has one measured execute
    // interval. The input values remain GPU-resident across these program boundaries.
    tracing::info!(graph = "multiply", operations = ?multiply_graph.operation_counts().unwrap());
    let multiply_inputs = BTreeMap::from([
        ("lhs".into(), encrypted_lhs["lhs"].clone()),
        ("rhs".into(), encrypted_rhs["rhs"].clone()),
    ]);
    let mut multiply_plan = runtime.plan(multiply_graph, &multiply_inputs).unwrap();
    let started = Instant::now();
    let multiplied = runtime.execute(&mut multiply_plan, multiply_inputs).unwrap();
    record_timing("multiply", started, &mut timings);

    tracing::info!(graph = "relinearize", operations = ?relinearize_graph.operation_counts().unwrap());
    let relinearize_inputs = BTreeMap::from([
        ("quadratic".into(), multiplied["quadratic"].clone()),
        ("rk".into(), keys["rk"].clone()),
    ]);
    let mut relinearize_plan = runtime.plan(relinearize_graph, &relinearize_inputs).unwrap();
    let started = Instant::now();
    let relinearized_outputs = runtime.execute(&mut relinearize_plan, relinearize_inputs).unwrap();
    record_timing("relinearize", started, &mut timings);

    tracing::info!(graph = "modswitch", operations = ?modswitch_graph.operation_counts().unwrap());
    let modswitch_inputs =
        BTreeMap::from([("relinearized".into(), relinearized_outputs["relinearized"].clone())]);
    let mut modswitch_plan = runtime.plan(modswitch_graph, &modswitch_inputs).unwrap();
    let started = Instant::now();
    let evaluated = runtime.execute(&mut modswitch_plan, modswitch_inputs).unwrap();
    record_timing("modswitch", started, &mut timings);

    // Decryption is another DSL program: bind the evaluated ciphertext and
    // secret key, execute it, and inspect the resulting plaintext family.
    tracing::info!(graph = "decryption", operations = ?decryption_graph.operation_counts().unwrap());
    let decryption_inputs =
        BTreeMap::from([("ct".into(), evaluated["ct"].clone()), ("sk".into(), keys["sk"].clone())]);
    let mut decryption_plan = runtime.plan(decryption_graph, &decryption_inputs).unwrap();
    let started = Instant::now();
    let decrypted = runtime.execute(&mut decryption_plan, decryption_inputs).unwrap();
    record_timing("decrypt", started, &mut timings);
    let expected = x
        .iter()
        .zip(&y)
        .map(|(&a, &b)| BigInt::from((a as u128 * b as u128 % plaintext_modulus as u128) as u64))
        .collect::<Vec<_>>();
    assert_eq!(runtime.download_integer_family(&decrypted["slots"]).unwrap(), expected);

    // The executed graphs, linked as one round trip, form the closed protocol of the claim. Each
    // stage input fed by an earlier stage's output is linked to it, as the test passes it above.
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("lean/bgv/generated");
    mxx_ir_core::lean::protocol::export(&protocol, &directory).unwrap();

    let total_ms = timings.iter().map(|(_, milliseconds)| milliseconds).sum::<f64>();
    println!("BGV_TIMING_SUMMARY total_execute_ms={total_ms:.3} stages={}", timings.len());

    // Repeat the evaluation on the same plans, whose Graphs have launched
    // before, and log the mean time of each stage. The first repetition
    // follows the decryption and download, so it is discarded as a warmup.
    let repeats = 3;
    let mut repeated = Vec::new();
    for repetition in 0..=repeats {
        if repetition == 1 {
            repeated.clear();
        }
        let multiply_inputs = BTreeMap::from([
            ("lhs".into(), encrypted_lhs["lhs"].clone()),
            ("rhs".into(), encrypted_rhs["rhs"].clone()),
        ]);
        let started = Instant::now();
        let multiplied = runtime.execute(&mut multiply_plan, multiply_inputs).unwrap();
        record_timing("repeat_multiply", started, &mut repeated);
        let relinearize_inputs = BTreeMap::from([
            ("quadratic".into(), multiplied["quadratic"].clone()),
            ("rk".into(), keys["rk"].clone()),
        ]);
        let started = Instant::now();
        let relinearized_outputs =
            runtime.execute(&mut relinearize_plan, relinearize_inputs).unwrap();
        record_timing("repeat_relinearize", started, &mut repeated);
        let modswitch_inputs =
            BTreeMap::from([("relinearized".into(), relinearized_outputs["relinearized"].clone())]);
        let started = Instant::now();
        runtime.execute(&mut modswitch_plan, modswitch_inputs).unwrap();
        record_timing("repeat_modswitch", started, &mut repeated);
    }
    let mean = |stage: &str| {
        repeated.iter().filter(|(name, _)| name == stage).map(|(_, ms)| ms).sum::<f64>() /
            repeats as f64
    };
    let (multiply, relinearize, modswitch) =
        (mean("repeat_multiply"), mean("repeat_relinearize"), mean("repeat_modswitch"));
    println!(
        "BGV_TIMING_SUMMARY stage=eval warmups=1 repeats={repeats} multiply_mean_ms={multiply:.3} \
         relinearize_mean_ms={relinearize:.3} modswitch_mean_ms={modswitch:.3} \
         total_mean_ms={:.3}",
        multiply + relinearize + modswitch
    );
}
