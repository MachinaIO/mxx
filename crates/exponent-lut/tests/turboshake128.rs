#![cfg(feature = "gpu")]

//! The GPU round trip of the counter-mode TurboSHAKE128 label hash.
//!
//! `TurboShake128LabelProgram` hashes a public message with
//! `TurboSHAKE128(T || BE32(i) || M, D = 0x1F)` over public sign encodings and packs the stream
//! into label encodings and chain-bit encodings (`program_gadgets::turboshake128`). The workflow
//! searches the smallest chain modulus that the noise audit accepts at a toy (insecure by
//! default) ring dimension, preprocesses, encrypts and evaluates the Program on the GPU, and
//! checks every output's encoding rows against the payload derived from RustCrypto's
//! `sha3::TurboShake128` (for the full 12 rounds) and against a wrong payload.
//!
//! Variables (prefix `MXX_EXPONENT_LUT_TURBOSHAKE128_`): `ROUNDS` (12), `MESSAGE_BYTES` (32),
//! `LABELS` (1024), `LABEL_BITS` (8), `CHAIN_BITS` (16), `LOG_RING_DIMENSION` (the smallest
//! dimension holding the labels), `CRT_BITS` (60), `BASE_BITS` (`CRT_BITS / 2`),
//! `MAX_CRT_DEPTH` (64), `NOISE_MODEL` (`average`), `ERROR_SIGMA` (4). The parameters are
//! toy parameters: no security estimate is made.
//! With fewer than 12 rounds the expected stream is the reduced-round Keccak-p reference.

use bigdecimal::BigDecimal;
use mxx_backends::{
    GpuRuntime,
    backend::poly_gpu::{GpuDcrtBackend, gpu_backend_on},
    poly::{
        PolyParams,
        dcrt::{
            gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
            params::DCRTPolyParams,
            poly::DCRTPoly,
        },
    },
};
use mxx_exponent_lut::{
    api,
    program::{FamilyRange, InputBinding},
    program_gadgets::turboshake128::{
        DOMAIN_BYTE, RATE_BYTES, TURBOSHAKE_ROUNDS, TurboShake128LabelHash,
        TurboShake128LabelProgram,
    },
    rns_bgg::{
        RhsMode, RnsBggNoiseBindings, RnsBggNoiseFamily, RnsBggNoiseValue, RnsBggParameters,
        RnsBggPlanRequest, RnsEncryptBuild, RnsOnlineEvalBuild, RnsOutputUse, RnsPreprocessBuild,
        verify_output,
    },
    utils::{audit_program_at_crt_depth, env_usize, noise_model},
};
use mxx_ir_core::{IntExpr, ParamEnv};
use num_bigint::{BigInt, BigUint};
use serial_test::serial as sequential;
use sha3::{
    TurboShake128, TurboShake128Core,
    digest::{ExtendableOutput, Update, XofReader},
};
use std::{collections::BTreeMap, time::Instant};

const HASH_KEY: &[u8; 32] = b"turboshake128-rns-public-hashkey";
const TAG: &[u8; 16] = b"mxx-turboshake-t";
const PREFIX: &str = "MXX_EXPONENT_LUT_TURBOSHAKE128_";

fn setting(name: &str, default: usize) -> usize {
    let name = format!("{PREFIX}{name}");
    env_usize(&name, default).unwrap_or_else(|error| panic!("{name}: {error}"))
}

/// `G(M)` from RustCrypto: `TurboSHAKE128(T || BE32(i) || M, D = 0x1F)` per instance,
/// concatenated.
fn rustcrypto_stream(message: &[u8], instances: usize, bits: usize) -> Vec<u8> {
    let mut stream = Vec::new();
    for instance in 0..instances {
        let mut hasher = TurboShake128::from_core(TurboShake128Core::new(DOMAIN_BYTE));
        hasher.update(TAG);
        hasher.update(&(instance as u32).to_be_bytes());
        hasher.update(message);
        let mut block = [0u8; RATE_BYTES];
        hasher.finalize_xof().read(&mut block);
        stream.extend_from_slice(&block);
    }
    stream.truncate(bits.div_ceil(8));
    stream
}

fn build_hash() -> TurboShake128LabelHash {
    let label_bits = setting("LABEL_BITS", 8);
    let ring_log = setting("LOG_RING_DIMENSION", label_bits.saturating_sub(1).max(1));
    TurboShake128LabelHash::new(
        TAG.to_vec(),
        setting("MESSAGE_BYTES", 32),
        setting("LABELS", 1024),
        label_bits as u32,
        setting("CHAIN_BITS", 16),
        1 << ring_log,
        setting("ROUNDS", TURBOSHAKE_ROUNDS),
    )
    .expect("TurboSHAKE128 label-hash declaration")
}

/// Every message bit and the unit are separate encryptions.
fn noise_bindings(program: &TurboShake128LabelProgram, sigma: &BigDecimal) -> RnsBggNoiseBindings {
    let value = RnsBggNoiseValue::sampled_encoding_sigma(
        sigma,
        b"turboshake128-input",
        &[BigInt::from(1u8)],
    )
    .expect("valid encoding sigma");
    RnsBggNoiseBindings::new(
        BTreeMap::from([(
            program.message_input(),
            InputBinding::Family(RnsBggNoiseFamily {
                element: value.independent_family_members(),
                range: FamilyRange::full(IntExpr::constant(program.hash().input_members()))
                    .expect("input family range"),
                max_nonzero_payloads: 1u8.into(),
                row_length: None,
            }),
        )]),
        BTreeMap::new(),
    )
    .expect("noise bindings")
}

/// The smallest CRT depth whose noise audit accepts the Program.
fn search(
    program: &TurboShake128LabelProgram,
    sigma: &BigDecimal,
) -> (RnsBggParameters, DCRTPolyParams) {
    let hash = program.hash();
    let crt_bits = setting("CRT_BITS", 60);
    let base_bits = setting("BASE_BITS", crt_bits / 2);
    let noise = noise_bindings(program, sigma);
    let average = noise_model(PREFIX).expect("noise-model configuration");
    // The simulator's payload envelopes, not the payloads: sums add the envelopes of their
    // terms, so a packed bit grows the envelope 12-fold (`2 (1 + X^c) acc + (1 - X^c) H` with
    // the four chi chains in `H`) and a round 4-fold, while the payloads grow 4-fold and 2-fold.
    let label_bound = BigUint::from(12u8).pow(hash.label_bits());
    let chain_bound = BigUint::from(4u8).pow(hash.rounds() as u32);
    let terminal_bounds = std::iter::repeat_n(label_bound.clone(), hash.labels())
        .chain(std::iter::repeat_n(chain_bound.clone(), hash.chain_bits()))
        .collect::<Vec<_>>();
    let mut failures = Vec::new();
    for crt_depth in 1..=setting("MAX_CRT_DEPTH", 64) {
        let request = RnsBggPlanRequest {
            ring_dimension: hash.ring_dimension() as u32,
            encoding_error_sigma: sigma.clone(),
            crt_bits,
            base_bits: base_bits as u32,
            terminal_depth: 1,
            max_crt_depth: crt_depth,
            terminal_value_bound: label_bound.clone().max(chain_bound.clone()),
            operations_per_drop: 1,
            primes_per_drop: 1,
            rhs_mode: RhsMode::Static,
        };
        let started = Instant::now();
        match audit_program_at_crt_depth(
            &request,
            program.program(),
            &noise,
            &terminal_bounds,
            average.as_ref(),
            0,
        ) {
            Ok((parameters, _)) => {
                let source = parameters.level(0).expect("source level").clone();
                eprintln!(
                    "TurboSHAKE128 parameters accepted: n={} crt_depth={crt_depth} crt_bits={crt_bits} base_bits={base_bits} audit={:.1}s",
                    hash.ring_dimension(),
                    started.elapsed().as_secs_f64()
                );
                return (parameters, source);
            }
            Err(error) => failures.push(format!("crt_depth={crt_depth}: {error}")),
        }
    }
    panic!("no TurboSHAKE128 parameter candidate was accepted: {}", failures.join("; "));
}

fn gpu_backend(parameters: &RnsBggParameters, devices: Vec<i32>) -> GpuDcrtBackend {
    let levels = parameters.distinct_levels().into_iter().map(|parameters| {
        GpuDCRTPolyParams::new(
            parameters.ring_dimension(),
            parameters.to_crt().0,
            parameters.base_bits(),
            None,
        )
    });
    gpu_backend_on(levels, devices)
}

#[test]
#[sequential(dcrt_runtime)]
fn turboshake128_label_hash_round_trip() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
    let devices = detected_gpu_device_ids();
    if devices.is_empty() {
        eprintln!("skipping the TurboSHAKE128 round trip: no CUDA device detected");
        return;
    }
    let sigma = BigDecimal::from(setting("ERROR_SIGMA", 4) as u64);
    let hash = build_hash();
    let program =
        TurboShake128LabelProgram::new(hash.clone()).expect("TurboSHAKE128 label Program");
    let message = (0..hash.message_bytes())
        .map(|index| (index as u8).wrapping_mul(29).wrapping_add(17))
        .collect::<Vec<_>>();
    eprintln!(
        "TurboSHAKE128 label hash: rounds={} |M|={} labels={} label_bits={} chain_bits={} instances={} n={}",
        hash.rounds(),
        hash.message_bytes(),
        hash.labels(),
        hash.label_bits(),
        hash.chain_bits(),
        hash.instances(),
        hash.ring_dimension()
    );
    // The expected stream: RustCrypto for TurboSHAKE128 itself, which the host reference must
    // match.
    let reference = hash.reference_stream(&message);
    let stream = if hash.rounds() == TURBOSHAKE_ROUNDS {
        let stream = rustcrypto_stream(&message, hash.instances(), hash.output_bits());
        assert_eq!(reference, stream, "the host reference is TurboSHAKE128");
        stream
    } else {
        reference
    };
    let (labels, chain) = hash.split_stream(&stream);

    let (parameters, level) = search(&program, &sigma);
    let directory = tempfile::tempdir().expect("artifact directory");
    let mut store = api::StageStorage::File(directory.path().into()).open().expect("stage storage");
    let mut runtime = GpuRuntime::new(gpu_backend(&parameters, devices)).expect("GPU runtime");
    let runtime = &mut runtime;
    let started = Instant::now();
    let payload = RnsEncryptBuild::new(&parameters, [7; 32])
        .sample_payload_secret(runtime, &mut store, |_| Ok(()))
        .expect("payload secret");
    let mask = RnsEncryptBuild::new(&parameters, [7; 32])
        .sample_mask_secret(runtime, &mut store, |_| Ok(()))
        .expect("mask secret");
    let (preprocess, online_eval) = api::preprocess(
        program.program(),
        &ParamEnv::default(),
        &api::PlaintextInputs::<_, DCRTPoly>::new(),
        Some(&payload),
        HASH_KEY,
        RnsPreprocessBuild::new(&parameters, [7; 32]),
        runtime,
        &mut store,
        |_, outputs| Ok(outputs),
        |_| Ok(()),
    )
    .expect("RNS preprocessing");
    eprintln!("preprocessing: {:.1}s", started.elapsed().as_secs_f64());
    let started = Instant::now();
    let encrypted = api::encrypt(
        &program.program().encryption_inputs(),
        &ParamEnv::default(),
        &program.plaintext_inputs::<DCRTPoly>(&level, &message),
        Some(&preprocess),
        &payload,
        Some(&mask),
        HASH_KEY,
        RnsEncryptBuild::new(&parameters, [7; 32]),
        runtime,
        &mut store,
        |_| Ok(()),
        |_| Ok(()),
    )
    .expect("RNS encryption");
    eprintln!("encryption: {:.1}s", started.elapsed().as_secs_f64());
    let started = Instant::now();
    let result = api::online_eval(
        program.program(),
        &ParamEnv::default(),
        &online_eval,
        &encrypted,
        HASH_KEY,
        RnsOnlineEvalBuild::new(&parameters, [7; 32]),
        runtime,
        &mut store,
        |_, outputs| Ok(outputs),
        |_| Ok(()),
    )
    .expect("RNS online evaluation");
    eprintln!("online evaluation: {:.1}s", started.elapsed().as_secs_f64());

    // Every output's rows against its RustCrypto payload, and against the payload of the next
    // label value or the flipped chain bit.
    let started = Instant::now();
    let modulus = 1usize << hash.label_bits();
    let expected = program.expected_outputs::<DCRTPoly>(&level, &labels, &chain);
    let wrong = program.expected_outputs::<DCRTPoly>(
        &level,
        &labels.iter().map(|label| (label + 1) % modulus).collect::<Vec<_>>(),
        &chain.iter().map(|bit| !bit).collect::<Vec<_>>(),
    );
    let mut verify = |index: usize, plaintext: &DCRTPoly| {
        verify_output(
            program.program(),
            &result,
            index,
            plaintext,
            &parameters,
            &payload,
            Some(&mask),
            RnsOutputUse::DirectDecode,
            runtime,
            &mut store,
        )
        .is_ok()
    };
    for index in 0..expected.len() {
        assert!(
            verify(index, &expected[index]),
            "output {index} must encode its TurboSHAKE128 payload"
        );
        assert!(!verify(index, &wrong[index]), "output {index} must reject a wrong payload");
    }
    eprintln!(
        "verified {} outputs against TurboSHAKE128: {:.1}s",
        expected.len(),
        started.elapsed().as_secs_f64()
    );
}
