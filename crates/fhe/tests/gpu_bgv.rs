//! BGV through the production GPU runtime, with the graphs of the public-key HE syntax
//! (`mxx_fhe::spec::graphs`): generate keys, encrypt three plaintexts, evaluate the circuit
//! `x₀ · (x₁ · x₂)`, and decrypt. These are the graphs whose correctness and security
//! `crates/fhe/lean/bgv` proves (`cargo run -p mxx-fhe --example export_claims` writes their
//! statements).

use mxx_backends::{GpuRuntime, RuntimeValue, backend::poly_gpu::gpu_backend, poly::PolyParams};
use mxx_dsl::BuiltGraph;
use mxx_fhe::{
    bgv::{
        fixtures::{self, bgv_params, modswitch_steps},
        scheme::BgvScheme,
    },
    spec::{Circuit, graphs, roles},
    utils::gpu::input,
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
fn test_gpu_bgv_circuit() {
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
    let bgv = bgv_params();
    let n = bgv.common.ring.ring_dimension() as usize;
    let scheme = BgvScheme { params: &bgv, modswitch_steps: modswitch_steps() };
    let circuit = Circuit::new(3, vec![[1, 2], [0, 3]], 4).expect("valid circuit");
    let graphs = graphs(&scheme, &circuit).expect("scheme graphs");
    let graph = |name: &str| BuiltGraph {
        graph: graphs.iter().find(|(graph, _)| graph == name).expect("graph").1.clone(),
    };

    let plaintext_modulus = bgv.plaintext_modulus;
    let messages = (0..3)
        .map(|_| {
            (0..n)
                .map(|_| rand::rng().random_range(0..plaintext_modulus) as i64)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut runtime = GpuRuntime::new(gpu_backend(fixtures::gpu_parameters(&bgv)))
        .expect("construct BGV GPU runtime");

    // A DSL program describes values to produce. Freeze it into a GPU plan,
    // execute it, and pass its outputs to the next program as inputs.
    let keygen = graph("Keygen");
    tracing::info!(graph = "keygen", operations = ?keygen.operation_counts().unwrap());
    let mut keygen_plan = runtime.plan(keygen, &BTreeMap::new()).unwrap();
    let started = Instant::now();
    let keys = runtime.execute(&mut keygen_plan, BTreeMap::new()).unwrap();
    record_timing("keygen", started, &mut timings);

    let encryption = graph("Encrypt");
    tracing::info!(graph = "encryption", operations = ?encryption.operation_counts().unwrap());
    let encryption_inputs = |message: &[i64]| {
        BTreeMap::from([
            (roles::PUBLIC_KEY.into(), keys[roles::PUBLIC_KEY].clone()),
            (roles::MESSAGE.into(), input(message)),
        ])
    };
    let mut encryption_plan = runtime.plan(encryption, &encryption_inputs(&messages[0])).unwrap();
    let mut ciphertexts = Vec::new();
    for message in &messages {
        let started = Instant::now();
        let encrypted = runtime.execute(&mut encryption_plan, encryption_inputs(message)).unwrap();
        record_timing("encrypt", started, &mut timings);
        ciphertexts.push(encrypted[roles::CIPHERTEXT].clone());
    }

    // The circuit's evaluation is one program; the ciphertexts stay GPU-resident.
    let evaluation = graph("Eval");
    tracing::info!(graph = "eval", operations = ?evaluation.operation_counts().unwrap());
    let evaluation_inputs = |ciphertexts: &[RuntimeValue]| {
        let mut inputs = ciphertexts
            .iter()
            .enumerate()
            .map(|(index, ciphertext)| (roles::ciphertext(index), ciphertext.clone()))
            .collect::<BTreeMap<_, _>>();
        inputs.insert(roles::EVALUATION_KEY.into(), keys[roles::EVALUATION_KEY].clone());
        inputs
    };
    let mut evaluation_plan = runtime.plan(evaluation, &evaluation_inputs(&ciphertexts)).unwrap();
    let started = Instant::now();
    let evaluated = runtime.execute(&mut evaluation_plan, evaluation_inputs(&ciphertexts)).unwrap();
    record_timing("eval", started, &mut timings);

    // Decryption of the evaluated ciphertext type.
    let decryption = graph("Decrypt_1");
    let decryption_inputs = BTreeMap::from([
        (roles::CIPHERTEXT.into(), evaluated[roles::CIPHERTEXT].clone()),
        (roles::SECRET_KEY.into(), keys[roles::SECRET_KEY].clone()),
    ]);
    let mut decryption_plan = runtime.plan(decryption, &decryption_inputs).unwrap();
    let started = Instant::now();
    let decrypted = runtime.execute(&mut decryption_plan, decryption_inputs).unwrap();
    record_timing("decrypt", started, &mut timings);
    let expected = (0..n)
        .map(|slot| {
            let product = messages.iter().fold(1u128, |product, message| {
                product * message[slot] as u128 % plaintext_modulus as u128
            });
            BigInt::from(product as u64)
        })
        .collect::<Vec<_>>();
    assert_eq!(runtime.download_integer_family(&decrypted[roles::MESSAGE]).unwrap(), expected);

    let total_ms = timings.iter().map(|(_, milliseconds)| milliseconds).sum::<f64>();
    println!("BGV_TIMING_SUMMARY total_execute_ms={total_ms:.3} stages={}", timings.len());

    // Repeat the evaluation on the same plan, whose Graph has launched before, and log its mean
    // time. The first repetition is discarded as a warmup.
    let repeats = 3;
    let mut repeated = Vec::new();
    for repetition in 0..=repeats {
        if repetition == 1 {
            repeated.clear();
        }
        let started = Instant::now();
        runtime.execute(&mut evaluation_plan, evaluation_inputs(&ciphertexts)).unwrap();
        record_timing("repeat_eval", started, &mut repeated);
    }
    let mean = repeated.iter().map(|(_, ms)| ms).sum::<f64>() / repeats as f64;
    println!("BGV_TIMING_SUMMARY stage=eval warmups=1 repeats={repeats} eval_mean_ms={mean:.3}");
}
