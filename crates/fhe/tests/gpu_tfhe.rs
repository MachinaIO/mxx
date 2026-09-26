//! A minimal TFHE round trip through the production GPU runtime: generate
//! keys, encrypt bits, evaluate bootstrapped NAND gates, and decrypt.

use mxx_backends::{GpuRuntime, RuntimeValue, backend::poly_gpu::gpu_backend, poly::PolyParams};
use mxx_dsl::{DslContext, GraphValue, IntType, Ring};
use mxx_fhe::utils::{self, gpu};
use num_bigint::BigInt;
use rand::Rng;
use std::{collections::BTreeMap, time::Instant};

/// Record one production execution interval at the `GpuRuntime::execute` return boundary.
/// The runtime contract already waits for the returned resident outputs; this helper deliberately
/// adds no caller-side CUDA synchronization or matrix fence.
fn record_timing(stage: &str, started: Instant, timings: &mut Vec<(String, f64)>) {
    let milliseconds = started.elapsed().as_secs_f64() * 1_000.0;
    println!("TFHE_TIMING stage={stage} elapsed_ms={milliseconds:.3}");
    timings.push((stage.to_owned(), milliseconds));
}

#[test]
fn test_gpu_tfhe_round_trip() {
    // Log how many times each primitive operation of every graph runs, and
    // each execute's preparation and run times; `RUST_LOG` overrides this
    // filter.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn,gpu_tfhe=info,mxx_backends::gpu_execute=debug".into()),
        )
        .with_test_writer()
        .try_init();
    let mut timings = Vec::new();
    let tfhe = utils::tfhe_params();
    let ring = Ring::from_crt_moduli(
        tfhe.common.ring.to_crt().0.into_iter().map(Into::into).collect(),
        tfhe.common.ring.ring_dimension(),
    );

    let mut runtime = GpuRuntime::new(gpu_backend(gpu::tfhe_gpu_parameters(&tfhe)))
        .expect("construct TFHE GPU runtime");
    // Blind rotations run as one native kernel when it covers the
    // parameters, unless FHE_TEST_TFHE_SUBGRAPH_KERNEL=0 runs their body.
    if std::env::var("FHE_TEST_TFHE_SUBGRAPH_KERNEL").as_deref() != Ok("0") {
        runtime.options_mut().subgraph_kernels.extend(tfhe.gpu_blind_rotation_kernel());
    }

    // Key generation and encryption expand pseudorandom values from a public
    // 32-byte hash key, which the caller draws fresh from an OS CSPRNG.
    let fresh_hash_key = || RuntimeValue::Bytes(rand::rng().random::<[u8; 32]>().to_vec().into());

    // A DSL program describes values to produce. Freeze it into a GPU plan,
    // execute it, and pass its outputs to the next program as inputs. A
    // ciphertext or key is one value, however many arrays it holds.
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
    tracing::info!(graph = "keygen", operations = ?keygen_graph.operation_counts().unwrap());
    let keygen_inputs = BTreeMap::from([("keygen_hash_key".into(), fresh_hash_key())]);
    let mut keygen_plan = runtime.plan(keygen_graph, &keygen_inputs).unwrap();
    let started = Instant::now();
    let key_outputs = runtime.execute(&mut keygen_plan, keygen_inputs).unwrap();
    record_timing("keygen", started, &mut timings);

    // Encryption of one bit under the LWE secret, planned once and executed
    // for every input bit with a fresh hash key.
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
    tracing::info!(graph = "encryption", operations = ?encryption_graph.operation_counts().unwrap());
    let encryption_inputs = |bit: bool| {
        BTreeMap::from([
            ("lwe_sk".into(), key_outputs["lwe_sk"].clone()),
            ("message".into(), RuntimeValue::Int(BigInt::from(u8::from(bit)))),
            ("encryption_hash_key".into(), fresh_hash_key()),
        ])
    };
    let mut encryption_plan = runtime.plan(encryption_graph, &encryption_inputs(false)).unwrap();
    let mut encrypt = |runtime: &mut GpuRuntime, bit: bool, timings: &mut Vec<(String, f64)>| {
        let started = Instant::now();
        let encrypted = runtime.execute(&mut encryption_plan, encryption_inputs(bit)).unwrap();
        record_timing("encrypt", started, timings);
        encrypted["ct"].clone()
    };

    // One bootstrapped NAND gate: it forms the phase Δ - left - right, blind
    // rotates the NAND accumulator by it with the bootstrapping key, extracts
    // the constant coefficient, and key switches back to the LWE secret, so
    // its output is a fresh ciphertext that can feed the next gate.
    let nand_graph = || {
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
    tracing::info!(graph = "nand", operations = ?nand_graph().operation_counts().unwrap());
    let gate_inputs = |left: RuntimeValue, right: RuntimeValue| {
        BTreeMap::from([
            ("left".into(), left),
            ("right".into(), right),
            ("bsk".into(), key_outputs["bsk"].clone()),
            ("ksk".into(), key_outputs["ksk"].clone()),
        ])
    };
    let example = encrypt(&mut runtime, false, &mut timings);
    let mut nand_plan =
        runtime.plan(nand_graph(), &gate_inputs(example.clone(), example.clone())).unwrap();
    // With the blind rotation kernel registered, the same gate planned from
    // the subgraph body must produce the identical ciphertext.
    let kernels = std::mem::take(&mut runtime.options_mut().subgraph_kernels);
    let mut reference_plan = (!kernels.is_empty()).then(|| {
        runtime.plan(nand_graph(), &gate_inputs(example.clone(), example.clone())).unwrap()
    });
    runtime.options_mut().subgraph_kernels = kernels;
    let ciphertext_words = |runtime: &GpuRuntime, value: &RuntimeValue| match value {
        RuntimeValue::Composite(leaves) => leaves
            .iter()
            .map(|leaf| runtime.download_integer_family(leaf).unwrap())
            .collect::<Vec<_>>(),
        other => vec![runtime.download_integer_family(other).unwrap()],
    };

    // Decryption is another DSL program: bind a ciphertext and the LWE
    // secret, execute it, and read the decoded bit.
    let decryption = DslContext::new("tfhe-round-trip-decryption");
    let bit = tfhe
        .decrypt(
            &decryption.int_family_input("lwe_sk", tfhe.lwe_dimension),
            &decryption.input("ct", ciphertext_schema).unwrap(),
        )
        .unwrap();
    let decryption_graph = decryption.output("bit", bit).unwrap().build().unwrap();
    tracing::info!(graph = "decryption", operations = ?decryption_graph.operation_counts().unwrap());
    let decryption_inputs = |ciphertext: RuntimeValue| {
        BTreeMap::from([
            ("lwe_sk".into(), key_outputs["lwe_sk"].clone()),
            ("ct".into(), ciphertext),
        ])
    };
    let mut decryption_plan = runtime.plan(decryption_graph, &decryption_inputs(example)).unwrap();

    // Evaluate the four NAND truth-table cases on fresh encryptions, then
    // chain further gates whose left input is the previous gate's output, so
    // bootstrapped ciphertexts are shown to be usable as gate inputs again.
    let truth_table = [(false, false), (false, true), (true, false), (true, true)];
    let chained_gates = 4;
    let mut previous = None;
    for index in 0..truth_table.len() + chained_gates {
        let (left, left_bit) = match truth_table.get(index) {
            Some(&(left_bit, _)) => (encrypt(&mut runtime, left_bit, &mut timings), left_bit),
            None => previous.take().expect("a chained gate follows a previous gate"),
        };
        let right_bit = truth_table.get(index).map_or_else(|| rand::rng().random(), |&(_, r)| r);
        let right = encrypt(&mut runtime, right_bit, &mut timings);

        let started = Instant::now();
        let output =
            runtime.execute(&mut nand_plan, gate_inputs(left.clone(), right.clone())).unwrap();
        record_timing("nand", started, &mut timings);
        if let Some(reference_plan) = reference_plan.as_mut().filter(|_| index < truth_table.len())
        {
            let reference = runtime.execute(reference_plan, gate_inputs(left, right)).unwrap();
            assert_eq!(
                ciphertext_words(&runtime, &output["ct"]),
                ciphertext_words(&runtime, &reference["ct"]),
                "gate {index}: the blind rotation kernel differs from its subgraph"
            );
        }

        let started = Instant::now();
        let decrypted =
            runtime.execute(&mut decryption_plan, decryption_inputs(output["ct"].clone())).unwrap();
        record_timing("decrypt", started, &mut timings);
        let expected = !(left_bit && right_bit);
        assert_eq!(
            runtime.download_integer_family(&decrypted["bit"]).unwrap(),
            vec![BigInt::from(u8::from(expected))],
            "gate {index}: NAND({left_bit}, {right_bit})"
        );
        previous = Some((output["ct"].clone(), expected));
    }

    let gates = timings.iter().filter(|(stage, _)| stage == "nand").collect::<Vec<_>>();
    let mean_ms =
        gates.iter().map(|(_, milliseconds)| milliseconds).sum::<f64>() / gates.len() as f64;
    println!("TFHE_TIMING_SUMMARY stage=nand samples={} mean_ms={mean_ms:.3}", gates.len());

    // Repeat one gate on the same plan, whose Graph has launched before, on
    // fresh encryptions made outside the timer, and log its mean time. The
    // first repetition is discarded as a warmup.
    let repeats = 3;
    let mut repeated = Vec::new();
    for repetition in 0..=repeats {
        if repetition == 1 {
            repeated.clear();
        }
        let (left_bit, right_bit) = (rand::rng().random::<bool>(), rand::rng().random::<bool>());
        let left = encrypt(&mut runtime, left_bit, &mut Vec::new());
        let right = encrypt(&mut runtime, right_bit, &mut Vec::new());
        let started = Instant::now();
        let output = runtime.execute(&mut nand_plan, gate_inputs(left, right)).unwrap();
        record_timing("repeat_nand", started, &mut repeated);
        let decrypted =
            runtime.execute(&mut decryption_plan, decryption_inputs(output["ct"].clone())).unwrap();
        assert_eq!(
            runtime.download_integer_family(&decrypted["bit"]).unwrap(),
            vec![BigInt::from(u8::from(!(left_bit && right_bit)))],
            "repeated NAND({left_bit}, {right_bit})"
        );
    }
    let mean_ms = repeated.iter().map(|(_, ms)| ms).sum::<f64>() / repeats as f64;
    println!(
        "TFHE_TIMING_SUMMARY stage=eval warmups=1 repeats={repeats} nand_mean_ms={mean_ms:.3}"
    );
}
