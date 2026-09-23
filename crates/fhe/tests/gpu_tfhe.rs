//! Full TFHE NAND bootstrapping through the production GPU runtime.
//!
//! Keys and stage outputs remain resident GPU owners between transient graphs.
//! Only the tiny hash-domain probe and final decrypted bits cross to the host;
//! each graph uses plain outputs and no artifact imports or exports.

use mxx_backends::{
    GpuExecutionPlan, GpuRuntime, MemoryArtifactStore, RuntimeValue,
    backend::poly_gpu::gpu_backend,
    openfhe_guard::gen_modulus_and_warmup,
    poly::dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams},
};
use mxx_dsl::{BuiltGraph, DslContext, GraphValue, GraphValueSchema, HashTag, Int, Ring};
use mxx_fhe::{
    FheCommonParams, TfheParams,
    utils::gpu::{copy_outputs, related_gpu_parameters},
};
use mxx_ir_core::{IntExpr, ParamEnv, ValidatedGraph, node::SampleRange};
use num_bigint::{BigInt, BigUint};
use rand::Rng;
use std::{collections::BTreeMap, env, time::Instant};

/// Resident outputs copied out of their plan, so they outlive its next execute.
type GpuValues = BTreeMap<String, RuntimeValue>;

const LWE_DIMENSION: usize = 1024;
const LWE_MODULUS: u64 = 1u64 << 32;
const LWE_SIGMA: f64 = 32_768.0;
const LWE_ERROR_CUTOFF: u64 = 1u64 << 19; // 16 sigma
const RING_DIMENSION: u32 = 2048;
const RING_MODULI: [u64; 2] = [33_550_337, 33_538_049];
const RING_SIGMA: f64 = 1_048_576.0;
const RING_ERROR_CUTOFF: u64 = 1u64 << 24; // 16 sigma
const GADGET_BASE_BITS: u32 = 4;

/// This fixed profile was evaluated with the lattice-estimator ADPS16 model,
/// estimator revision 2a799b25fb3ee968b4a43f14ac7691f4dc949a15, binary
/// secrets and infinite samples. Its modeled quantum lattice-reduction costs
/// use 0.265β. The minimum reported estimates were 147.275 classical / 135.145
/// quantum bits for LWE and 187.316 classical / 170.940 quantum bits for the
/// ring basis. The KSK uses base 2 with 32 levels. These are estimator outputs,
/// not unconditional quantum security proofs.
/// The test keeps these cryptographic parameters fixed so its correctness run
/// cannot silently stop exercising the estimator-validated profile.
fn tfhe_params() -> TfheParams {
    let ring = DCRTPolyParams::try_new(
        RING_DIMENSION,
        RING_MODULI.len(),
        25,
        GADGET_BASE_BITS,
        Some(RING_MODULI.to_vec()),
        None,
    )
    .expect("estimator-validated exact CRT basis");
    let common = FheCommonParams {
        ring,
        secret_range: SampleRange { minimum: 0.into(), maximum: 1.into() },
        error_sigma: RING_SIGMA,
        error_cutoff: BigUint::from(RING_ERROR_CUTOFF),
    };
    TfheParams::new(
        common,
        LWE_DIMENSION,
        BigUint::from(LWE_MODULUS),
        LWE_SIGMA,
        BigUint::from(LWE_ERROR_CUTOFF),
    )
    .expect("estimator-validated TFHE parameters")
}

fn gpu_parameters(tfhe: &TfheParams) -> Vec<GpuDCRTPolyParams> {
    related_gpu_parameters(tfhe.runtime_parameters())
}

fn fresh_nonce() -> [u8; 32] {
    let mut nonce = [0u8; 32];
    rand::rng().fill(&mut nonce);
    nonce
}

fn fresh_hash_key() -> Vec<u8> {
    let mut key = vec![0u8; 32];
    rand::rng().fill(key.as_mut_slice());
    key
}

fn tfhe_ring() -> Ring {
    Ring::from_crt_moduli(RING_MODULI.into_iter().map(IntExpr::from).collect(), RING_DIMENSION)
}

fn repeated_gate_count() -> usize {
    let count = env::var("FHE_TFHE_REPEATED_GATES")
        .map(|value| value.parse().expect("FHE_TFHE_REPEATED_GATES is an integer"))
        .unwrap_or(4);
    assert!(count > 0, "at least one repeated NAND gate is required");
    count
}

fn named_leaf(base: &str, index: usize, count: usize) -> String {
    if count == 1 { base.to_owned() } else { format!("{base}.{index}") }
}

fn integer_output(runtime: &GpuRuntime, outputs: &GpuValues, name: &str) -> BigInt {
    match &outputs[name] {
        RuntimeValue::Int(value) => value.clone(),
        value => runtime
            .download_integer_family(value)
            .unwrap_or_else(|error| panic!("download integer {name}: {error}"))
            .into_iter()
            .next()
            .expect("one integer output"),
    }
}

/// Exercise the exact integer hash-family primitive with one seed: KSK and
/// ciphertext-a tags produce different samples, and two KSK indices do too.
fn verify_hash_domains(runtime: &mut GpuRuntime, store: &mut MemoryArtifactStore) {
    let mut context = DslContext::new("tfhe-hash-domain-separation");
    let hash_key = tfhe_ring().bytes_input("domain_probe_key", 32);
    let modulus = BigInt::from(LWE_MODULUS);
    let encrypted_a = context.hash_int_family(
        hash_key.clone(),
        HashTag::from(b"tfhe/lwe-encrypt/a/v1".as_slice()),
        4,
        modulus.clone(),
    );
    let ksk_a = context.hash_int_family(
        hash_key,
        HashTag::from(b"tfhe/keygen/ksk-a/v1".as_slice()),
        4,
        modulus,
    );
    context = context.output("encrypt_a", encrypted_a).unwrap().output("ksk_a", ksk_a).unwrap();
    let graph = context
        .build()
        .unwrap()
        .validate(&ParamEnv::default(), gen_modulus_and_warmup)
        .expect("valid hash-domain graph");
    let key_a = fresh_hash_key();
    let planning_inputs = BTreeMap::from([(
        "domain_probe_key".to_owned(),
        RuntimeValue::Bytes(key_a.clone().into()),
    )]);
    let mut plan = runtime.plan(graph, &planning_inputs).expect("plan hash-domain probe");
    let (encrypted_a, ksk_a) = execute_hash_probe(runtime, &mut plan, store, key_a.clone());
    let (encrypted_a_repeat, ksk_a_repeat) =
        execute_hash_probe(runtime, &mut plan, store, key_a.clone());
    let key_b = loop {
        let candidate = fresh_hash_key();
        if candidate != key_a {
            break candidate;
        }
    };
    let (encrypted_b, _) = execute_hash_probe(runtime, &mut plan, store, key_b);

    assert_eq!(encrypted_a, encrypted_a_repeat, "same key and tag reproduce the LWE samples");
    assert_eq!(ksk_a, ksk_a_repeat, "same key and tag reproduce the KSK samples");
    assert_ne!(encrypted_a, encrypted_b, "a rebound fresh key changes the LWE samples");
    assert_ne!(encrypted_a, ksk_a, "LWE encryption and KSK use distinct tag domains");
    assert_ne!(ksk_a[0], ksk_a[1], "KSK hash samples include the family element index");
}

fn execute_hash_probe(
    runtime: &mut GpuRuntime,
    plan: &mut GpuExecutionPlan,
    store: &mut MemoryArtifactStore,
    key: Vec<u8>,
) -> (Vec<BigInt>, Vec<BigInt>) {
    let inputs = BTreeMap::from([("domain_probe_key".to_owned(), RuntimeValue::Bytes(key.into()))]);
    let result =
        runtime.execute(plan, inputs, store, fresh_nonce()).expect("execute hash-domain probe");
    let outputs = copy_outputs(runtime, &result);
    drop(result);
    let encrypted_a = integer_family_output(runtime, &outputs, "encrypt_a");
    let ksk_a = integer_family_output(runtime, &outputs, "ksk_a");
    (encrypted_a, ksk_a)
}

fn integer_family_output(runtime: &GpuRuntime, outputs: &GpuValues, name: &str) -> Vec<BigInt> {
    runtime
        .download_integer_family(&outputs[name])
        .unwrap_or_else(|error| panic!("download integer family {name}: {error}"))
}

/// Bind every flattened leaf of a named graph value from a prior graph output.
/// The schema controls both name expansion and the number of resident owners.
fn bind_output<S: GraphValueSchema>(
    inputs: &mut GpuValues,
    input_name: &str,
    schema: &S,
    outputs: &GpuValues,
    output_name: &str,
) {
    let count = schema.wire_types().len();
    assert!(count > 0, "graph value must have at least one wire");
    for index in 0..count {
        let input_leaf = named_leaf(input_name, index, count);
        let output_leaf = named_leaf(output_name, index, count);
        inputs.insert(input_leaf, outputs[&output_leaf].clone());
    }
}

/// Rename a key-switched result's leaves to the `ciphertext` leaves a fresh
/// encryption has, so it can feed the next gate.
fn as_ciphertext(outputs: GpuValues) -> GpuValues {
    outputs
        .into_iter()
        .map(|(name, value)| (name.replacen("switched", "ciphertext", 1), value))
        .collect()
}

/// One plan per stage. Planning uses the first real input owners; subsequent
/// executions only rebind same-schema values and retain the resident dataflow.
struct TimedStage {
    validated: Option<ValidatedGraph>,
    plan: Option<GpuExecutionPlan>,
    /// A stage may read only some leaves of a bound value, e.g. blind rotation
    /// reads the LWE `a` vector but not `b`; the others are not graph inputs.
    inputs: std::collections::BTreeSet<String>,
}

impl TimedStage {
    fn new(graph: BuiltGraph) -> Self {
        let validated =
            graph.validate(&ParamEnv::default(), gen_modulus_and_warmup).expect("valid TFHE graph");
        let inputs = validated
            .source
            .root_scope()
            .nodes()
            .iter()
            .filter_map(|node| match node.kind() {
                mxx_ir_core::node::NodeKind::Input { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect();
        Self { validated: Some(validated), plan: None, inputs }
    }

    fn declared(&self, mut inputs: GpuValues) -> GpuValues {
        inputs.retain(|name, _| self.inputs.contains(name));
        inputs
    }

    fn execute(
        &mut self,
        runtime: &mut GpuRuntime,
        inputs: GpuValues,
        store: &mut MemoryArtifactStore,
        stage: &'static str,
        gate: usize,
        timings: &mut BTreeMap<&'static str, Vec<f64>>,
    ) -> GpuValues {
        let inputs = self.declared(inputs);
        if self.plan.is_none() {
            let validated = self.validated.take().expect("first execution creates the plan");
            self.plan = Some(runtime.plan(validated, &inputs).expect("plan TFHE GPU stage"));
        }

        // GpuRuntime::execute returns only after its resident output owners are
        // ready, so this interval measures the production stage boundary.
        let started = Instant::now();
        let result = runtime
            .execute(self.plan.as_mut().expect("planned TFHE stage"), inputs, store, fresh_nonce())
            .expect("execute TFHE GPU stage");
        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        println!("TFHE_TIMING stage={stage} gate={gate} elapsed_ms={elapsed_ms:.3}");
        timings.entry(stage).or_default().push(elapsed_ms);
        copy_outputs(runtime, &result)
    }
}

fn decrypt_bit(
    runtime: &mut GpuRuntime,
    stage: &mut TimedStage,
    inputs: GpuValues,
    store: &mut MemoryArtifactStore,
) -> BigInt {
    let inputs = stage.declared(inputs);
    let result = runtime
        .execute(
            stage.plan.as_mut().expect("decryption plan is prepared"),
            inputs,
            store,
            fresh_nonce(),
        )
        .expect("execute TFHE decryption");
    let outputs = copy_outputs(runtime, &result);
    drop(result);
    integer_output(runtime, &outputs, "bit")
}

fn encrypt_bit(
    runtime: &mut GpuRuntime,
    plan: &mut GpuExecutionPlan,
    store: &mut MemoryArtifactStore,
    lwe_secret: &RuntimeValue,
    value: bool,
) -> GpuValues {
    let inputs = BTreeMap::from([
        ("lwe_sk".to_owned(), lwe_secret.clone()),
        ("message".to_owned(), RuntimeValue::Int(BigInt::from(u8::from(value)))),
        ("encryption_hash_key".to_owned(), RuntimeValue::Bytes(fresh_hash_key().into())),
    ]);
    let result =
        runtime.execute(plan, inputs, store, fresh_nonce()).expect("encrypt noisy TFHE input bit");
    copy_outputs(runtime, &result)
}

fn run_nand_gate(
    runtime: &mut GpuRuntime,
    store: &mut MemoryArtifactStore,
    left: &GpuValues,
    right: &GpuValues,
    ciphertext_schema: &impl GraphValueSchema,
    nand_input_schema: &impl GraphValueSchema,
    ring_ciphertext_schema: &impl GraphValueSchema,
    blinded_ciphertext_schema: &impl GraphValueSchema,
    extracted_schema: &impl GraphValueSchema,
    bsk_schema: &impl GraphValueSchema,
    ksk_schema: &impl GraphValueSchema,
    key_outputs: &GpuValues,
    pre_blind_rotation: &mut TimedStage,
    blind_rotation: &mut TimedStage,
    sample_extract: &mut TimedStage,
    key_switch: &mut TimedStage,
    gate: usize,
    timings: &mut BTreeMap<&'static str, Vec<f64>>,
) -> GpuValues {
    let mut pre_inputs = GpuValues::new();
    bind_output(&mut pre_inputs, "left", ciphertext_schema, left, "ciphertext");
    bind_output(&mut pre_inputs, "right", ciphertext_schema, right, "ciphertext");
    let pre_outputs =
        pre_blind_rotation.execute(runtime, pre_inputs, store, "pre_blind_rotation", gate, timings);

    let mut blind_inputs = GpuValues::new();
    bind_output(&mut blind_inputs, "nand_input", nand_input_schema, &pre_outputs, "nand_input");
    bind_output(&mut blind_inputs, "rotated", ring_ciphertext_schema, &pre_outputs, "rotated");
    bind_output(&mut blind_inputs, "bsk", bsk_schema, key_outputs, "bsk");
    let blind_outputs =
        blind_rotation.execute(runtime, blind_inputs, store, "blind_rotation", gate, timings);

    let mut extract_inputs = GpuValues::new();
    bind_output(
        &mut extract_inputs,
        "rotated",
        blinded_ciphertext_schema,
        &blind_outputs,
        "rotated",
    );
    let extract_outputs =
        sample_extract.execute(runtime, extract_inputs, store, "sample_extract", gate, timings);

    let mut switch_inputs = GpuValues::new();
    bind_output(&mut switch_inputs, "extracted", extracted_schema, &extract_outputs, "extracted");
    bind_output(&mut switch_inputs, "ksk", ksk_schema, key_outputs, "ksk");
    key_switch.execute(runtime, switch_inputs, store, "key_switch", gate, timings)
}

#[test]
fn test_gpu_tfhe_hash_int_family_rebind() {
    let tfhe = tfhe_params();
    let mut runtime =
        GpuRuntime::new(gpu_backend(gpu_parameters(&tfhe))).expect("construct TFHE GPU runtime");
    let mut store = MemoryArtifactStore::default();
    verify_hash_domains(&mut runtime, &mut store);
}

#[test]
fn test_gpu_tfhe_noisy_nand_and_repeated_gates() {
    let tfhe = tfhe_params();
    assert_eq!(tfhe.key_switch_base_bits(), 1, "validated KSK uses base 2");
    assert_eq!(tfhe.key_switch_digit_count(), 32, "validated KSK has 32 digits");
    let keygen_hash_key = tfhe_ring().bytes_input("keygen_hash_key", 32);
    let key_handles = tfhe.keygen(&keygen_hash_key).expect("build TFHE key generation graph");
    let keygen_graph_built = DslContext::new("gpu-tfhe-keygen")
        .output("lwe_sk", key_handles.lwe_secret.clone())
        .unwrap()
        .output("ring_sk", key_handles.ring_secret.clone())
        .unwrap()
        .output("bsk", key_handles.bootstrapping_key.clone())
        .unwrap()
        .output("ksk", key_handles.key_switch_key.clone())
        .unwrap()
        .build()
        .unwrap();
    let keygen_graph = keygen_graph_built
        .validate(&ParamEnv::default(), gen_modulus_and_warmup)
        .expect("valid key generation graph");

    let mut runtime =
        GpuRuntime::new(gpu_backend(gpu_parameters(&tfhe))).expect("construct TFHE GPU runtime");
    let mut store = MemoryArtifactStore::default();
    verify_hash_domains(&mut runtime, &mut store);
    let keygen_inputs = BTreeMap::from([(
        "keygen_hash_key".to_owned(),
        RuntimeValue::Bytes(fresh_hash_key().into()),
    )]);
    let mut keygen_plan =
        runtime.plan(keygen_graph, &keygen_inputs).expect("plan TFHE key generation");
    let key_outputs = {
        let result = runtime
            .execute(&mut keygen_plan, keygen_inputs, &mut store, fresh_nonce())
            .expect("generate TFHE keys on GPU");
        copy_outputs(&runtime, &result)
    };
    // The copied keys outlive the key-generation plan's own storage.
    drop(keygen_plan);
    let encryption_context = DslContext::new("gpu-tfhe-encrypt-bit");
    let lwe_secret = encryption_context
        .input("lwe_sk", key_handles.lwe_secret.schema())
        .expect("LWE secret input");
    let message = encryption_context
        .input("message", Int::constant(0).schema())
        .expect("plaintext bit input");
    let encryption_hash_key = tfhe_ring().bytes_input("encryption_hash_key", 32);
    let encrypted_template =
        tfhe.encrypt(&lwe_secret, &message, &encryption_hash_key).expect("build LWE encryption");
    let ciphertext_schema = encrypted_template.schema();
    let encryption_graph = encryption_context
        .output("ciphertext", encrypted_template)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default(), gen_modulus_and_warmup)
        .expect("valid encryption graph");
    let encryption_inputs = BTreeMap::from([
        ("lwe_sk".to_owned(), key_outputs["lwe_sk"].clone()),
        ("message".to_owned(), RuntimeValue::Int(BigInt::from(0u8))),
        ("encryption_hash_key".to_owned(), RuntimeValue::Bytes(fresh_hash_key().into())),
    ]);
    // The encryption graph binds the plaintext bit as a host integer, whose
    // range only this plan declares.
    runtime
        .options_mut()
        .integer_input_ranges
        .insert("message".to_owned(), BigInt::from(0u8)..=BigInt::from(1u8));
    let mut encryption_plan =
        runtime.plan(encryption_graph, &encryption_inputs).expect("plan noisy LWE encryption");
    runtime.options_mut().integer_input_ranges.clear();

    let pre_context = DslContext::new("gpu-tfhe-pre-blind-rotation");
    let left_input = pre_context.input("left", ciphertext_schema.clone()).expect("left LWE input");
    let right_input =
        pre_context.input("right", ciphertext_schema.clone()).expect("right LWE input");
    // Δ − left − right maps the four signed input phases to +3Δ,+Δ,+Δ,−Δ.
    // The sign accumulator therefore keeps mixed inputs away from its LUT edge.
    let nand_input = tfhe.nand_input(&left_input, &right_input).expect("form NAND phase");
    let nand_input_schema = nand_input.schema();
    let accumulator = tfhe.nand_accumulator();
    let rotated =
        tfhe.pre_blind_rotation(&nand_input, &accumulator).expect("build pre-blind rotation");
    let ring_ciphertext_schema = rotated.schema();
    let pre_graph = pre_context
        .output("nand_input", nand_input)
        .unwrap()
        .output("rotated", rotated)
        .unwrap()
        .build()
        .unwrap();
    let mut pre_blind_rotation = TimedStage::new(pre_graph);

    let blind_context = DslContext::new("gpu-tfhe-blind-rotation");
    let blind_nand_input =
        blind_context.input("nand_input", nand_input_schema.clone()).expect("NAND LWE input");
    let blind_rotated = blind_context
        .input("rotated", ring_ciphertext_schema.clone())
        .expect("pre-rotated accumulator input");
    let bsk_input = blind_context
        .input("bsk", key_handles.bootstrapping_key.schema())
        .expect("bootstrapping key input");
    let blinded = tfhe
        .blind_rotation(&blind_rotated, &blind_nand_input, &bsk_input)
        .expect("build blind rotation");
    let blinded_ciphertext_schema = blinded.schema();
    let blind_graph = blind_context.output("rotated", blinded).unwrap().build().unwrap();
    let mut blind_rotation = TimedStage::new(blind_graph);

    let sample_context = DslContext::new("gpu-tfhe-sample-extract");
    let sample_rotated = sample_context
        .input("rotated", blinded_ciphertext_schema.clone())
        .expect("blind-rotated accumulator input");
    let extracted = tfhe.sample_extract(&sample_rotated).expect("build sample extraction");
    let extracted_schema = extracted.schema();
    let sample_graph = sample_context.output("extracted", extracted).unwrap().build().unwrap();
    let mut sample_extract = TimedStage::new(sample_graph);

    let switch_context = DslContext::new("gpu-tfhe-key-switch");
    let switch_extracted =
        switch_context.input("extracted", extracted_schema.clone()).expect("extracted LWE input");
    let ksk_input = switch_context
        .input("ksk", key_handles.key_switch_key.schema())
        .expect("key-switch key input");
    let switched = tfhe.key_switch(&switch_extracted, &ksk_input).expect("build key switching");
    let switched_schema = switched.schema();
    let switch_graph = switch_context.output("switched", switched).unwrap().build().unwrap();
    let mut key_switch = TimedStage::new(switch_graph);

    let decrypt_context = DslContext::new("gpu-tfhe-decrypt-bit");
    let decrypt_secret = decrypt_context
        .input("lwe_sk", key_handles.lwe_secret.schema())
        .expect("decryption secret input");
    let decrypt_ciphertext =
        decrypt_context.input("switched", switched_schema.clone()).expect("switched LWE input");
    let decoded = tfhe.decrypt(&decrypt_secret, &decrypt_ciphertext).expect("build LWE decryption");
    let decrypt_graph = decrypt_context.output("bit", decoded).unwrap().build().unwrap();
    let mut decrypt_stage = TimedStage::new(decrypt_graph);

    let bsk_schema = key_handles.bootstrapping_key.schema();
    let ksk_schema = key_handles.key_switch_key.schema();
    let lwe_secret_schema = key_handles.lwe_secret.schema();
    let mut timings = BTreeMap::<&'static str, Vec<f64>>::new();
    let mut gate_index = 0usize;
    let mut last_truth_table_result = None;
    let mut last_truth_table_plaintext = false;

    for (left_bit, right_bit) in [(false, false), (false, true), (true, false), (true, true)] {
        let left = encrypt_bit(
            &mut runtime,
            &mut encryption_plan,
            &mut store,
            &key_outputs["lwe_sk"],
            left_bit,
        );
        let right = encrypt_bit(
            &mut runtime,
            &mut encryption_plan,
            &mut store,
            &key_outputs["lwe_sk"],
            right_bit,
        );
        let expected = !(left_bit && right_bit);
        let switched_outputs = run_nand_gate(
            &mut runtime,
            &mut store,
            &left,
            &right,
            &ciphertext_schema,
            &nand_input_schema,
            &ring_ciphertext_schema,
            &blinded_ciphertext_schema,
            &extracted_schema,
            &bsk_schema,
            &ksk_schema,
            &key_outputs,
            &mut pre_blind_rotation,
            &mut blind_rotation,
            &mut sample_extract,
            &mut key_switch,
            gate_index,
            &mut timings,
        );
        let mut decrypt_inputs = GpuValues::new();
        bind_output(&mut decrypt_inputs, "lwe_sk", &lwe_secret_schema, &key_outputs, "lwe_sk");
        bind_output(
            &mut decrypt_inputs,
            "switched",
            &switched_schema,
            &switched_outputs,
            "switched",
        );
        if decrypt_stage.plan.is_none() {
            let validated = decrypt_stage.validated.take().expect("first decryption plan");
            let planning_inputs = decrypt_stage.declared(decrypt_inputs.clone());
            decrypt_stage.plan =
                Some(runtime.plan(validated, &planning_inputs).expect("plan TFHE decryption"));
        }
        let actual = decrypt_bit(&mut runtime, &mut decrypt_stage, decrypt_inputs, &mut store);
        assert_eq!(actual, BigInt::from(u8::from(expected)), "NAND({left_bit},{right_bit})");
        if left_bit && right_bit {
            last_truth_table_result = Some(as_ciphertext(switched_outputs));
            last_truth_table_plaintext = expected;
        }
        gate_index += 1;
    }

    // Continue bootstrapping prior outputs and decrypt every result. Runtime
    // wires carry the ciphertext values, not the caller-maintained noise_bound
    // annotation from the graph schema, so this checks actual correctness and
    // does not treat that annotation as a runtime noise certificate.
    let mut current = last_truth_table_result.expect("all four NAND truth-table cases ran");
    let mut current_plaintext = last_truth_table_plaintext;
    for _ in 0..repeated_gate_count() {
        let right_bit = rand::rng().random_range(0..=1) == 1;
        let right = encrypt_bit(
            &mut runtime,
            &mut encryption_plan,
            &mut store,
            &key_outputs["lwe_sk"],
            right_bit,
        );
        let expected = !(current_plaintext && right_bit);
        let switched_outputs = run_nand_gate(
            &mut runtime,
            &mut store,
            &current,
            &right,
            &ciphertext_schema,
            &nand_input_schema,
            &ring_ciphertext_schema,
            &blinded_ciphertext_schema,
            &extracted_schema,
            &bsk_schema,
            &ksk_schema,
            &key_outputs,
            &mut pre_blind_rotation,
            &mut blind_rotation,
            &mut sample_extract,
            &mut key_switch,
            gate_index,
            &mut timings,
        );
        let mut decrypt_inputs = GpuValues::new();
        bind_output(&mut decrypt_inputs, "lwe_sk", &lwe_secret_schema, &key_outputs, "lwe_sk");
        bind_output(
            &mut decrypt_inputs,
            "switched",
            &switched_schema,
            &switched_outputs,
            "switched",
        );
        let actual = decrypt_bit(&mut runtime, &mut decrypt_stage, decrypt_inputs, &mut store);
        assert_eq!(actual, BigInt::from(u8::from(expected)), "repeated NAND gate {gate_index}");
        current = as_ciphertext(switched_outputs);
        current_plaintext = expected;
        gate_index += 1;
    }

    for (stage, elapsed_ms) in timings {
        let mean_ms = elapsed_ms.iter().sum::<f64>() / elapsed_ms.len() as f64;
        println!(
            "TFHE_TIMING_SUMMARY stage={stage} samples={} mean_ms={mean_ms:.3}",
            elapsed_ms.len()
        );
    }
}
