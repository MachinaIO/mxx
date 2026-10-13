#![cfg(feature = "gpu")]

//! The smudged trapdoor decoder round trip on the GPU runtime.
//!
//! The decoder ([`mxx_exponent_lut::decoder::decoding`]) of the identity Program's private output
//! runs through the RNS-BGG application API in three artifact stages:
//!
//! - preprocessing prepares the source Program and the smudge Program (a sparse-LWR PRF per smudge
//!   member), samples the public `B` with its trapdoor, and publishes `B` and the short preimage
//!   `K` of the summed decoder key `A`; the trapdoor never leaves the stage;
//! - encryption encrypts the source input `X^b`, the smudge PRF inputs, and the unmask hint `b = s
//!   B + e_b`;
//! - online evaluation evaluates the source and smudge Programs, adds the PRF smudge to the source
//!   output, unmasks the sum with `K`, and rounds it to the plaintext modulus.
//!
//! The parameter search tries the ring dimensions, primes per level drop, and CRT depths in order.
//! [`mxx_exponent_lut::utils::select_decoder_at_crt_depth`] plans each candidate with the smudge
//! chain, audits the source under the selected noise model, and picks the smallest smudge digit
//! count whose range `P^d` hides the simulated decoder noise by `STATISTICAL_BITS` and fits the
//! decoding margin. A selected candidate must also reach the lattice security target on both
//! chains. Every stage is planned by the GPU runtime immediately before it executes; the plan's
//! predicted time is logged as the benchmark estimate ("planned Exponent-LUT GPU stage"). The round
//! trip checks that the decoded plaintext is the source input. The stages always execute; an
//! estimate-only run (`MXX_EXPONENT_LUT_ESTIMATE_ONLY=1`) is rejected. The round trip runs the
//! Static RHS lifecycle only.
//!
//! | Variable (`MXX_EXPONENT_LUT_DECODER_` prefix) | Default | Meaning |
//! | --- | --- | --- |
//! | `MIN_LOG_RING_DIMENSION`, `MAX_LOG_RING_DIMENSION` | 5, 12 | Ring-dimension candidates |
//! | `MAX_PRIMES_PER_DROP` | 4 | Primes-per-drop candidates `1..=` |
//! | `MAX_CRT_DEPTH` | 40 | CRT-depth candidates `2..=` |
//! | `CRT_BITS`, `BASE_BITS` | 30, `ceil(CRT_BITS / 2)` | CRT prime and gadget base widths |
//! | `ERROR_SIGMA`, `TRAPDOOR_SIGMA` | 4, 4.578 | Encoding error and trapdoor Gaussian widths |
//! | `SMUDGE_P` | 16 | Output modulus `P` of the smudge PRF |
//! | `STATISTICAL_BITS` | 1 | Hiding margin of the smudge over the decoder noise |
//! | `SECURITY_BITS` | 1 | Required lattice security |
//! | `NOISE_MODEL`, `FAILURE_EXPONENT` | `average`, 100 | Noise model of the source audit |

use bigdecimal::{BigDecimal, ToPrimitive};
use mxx_backends::{
    GpuRuntime,
    backend::poly_gpu::gpu_backend_on,
    lattice_estimator::lattice_security_bits,
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
            poly::DCRTPoly,
        },
    },
};
use mxx_exponent_lut::{
    ExpLayout,
    api::{self, PlaintextInputs, StageExecution},
    decoder::simulation::AverageCaseConfig,
    program::{
        AttributeKnowledge, ExponentLutProgram, ExponentLutProgramBuilder, InputBinding, Operand,
        RhsId, WireId,
    },
    program_gadgets::{
        pbc::{PbcGeneratedKeyLayout, PbcParameters, PbcRootSeed, generate_key_layout},
        prf::SparseLwrPrfProfile,
    },
    rns_bgg::{
        RhsMode, RnsBggNoiseBindings, RnsBggNoiseValue, RnsBggPlanRequest, RnsEncryptBuild,
        RnsOnlineEvalBuild, RnsPreprocessBuild,
    },
    utils::{
        DecoderRequest, DecoderSelection, env_usize, noise_model, ring_dimension_candidates,
        ring_dimension_stride_log2, select_decoder_at_crt_depth,
    },
};
use mxx_ir_core::ParamEnv;
use num_bigint::{BigInt, BigUint};
use num_traits::Zero;
use serial_test::serial as sequential;
use std::{cell::RefCell, collections::BTreeMap, env};

const PREFIX: &str = "MXX_EXPONENT_LUT_DECODER_";
/// The LWR dimension of the smudge PRF.
const LWR_DIMENSION: usize = 4;
/// The input width `n'` of the smudge PRF.
const PRF_WIDTH_LOG2: u32 = 3;

fn setting(name: &str, default: usize) -> usize {
    let name = format!("{PREFIX}{name}");
    env_usize(&name, default).unwrap_or_else(|error| panic!("{name}: {error}"))
}

fn sigma(name: &str, default: &str) -> BigDecimal {
    let name = format!("{PREFIX}{name}");
    let value = env::var(&name)
        .unwrap_or_else(|_| default.to_owned())
        .parse::<BigDecimal>()
        .unwrap_or_else(|error| panic!("{name} must be a decimal: {error}"));
    assert!(value > BigDecimal::zero(), "{name} must be positive");
    value
}

/// The identity Program on one private input, and that input.
fn identity_source() -> (ExponentLutProgram, WireId) {
    let mut builder = ExponentLutProgramBuilder::new();
    let input = builder.input(AttributeKnowledge::PrivatePayload).expect("source input");
    (builder.finish(vec![Operand::Scalar(input)]).expect("identity Program"), input)
}

/// `count` random weight-one PRF keys over the LWR coordinates, in two buckets.
fn smudge_keys(count: usize) -> Vec<PbcGeneratedKeyLayout> {
    let parameters = PbcParameters::custom(LWR_DIMENSION, 1, 2, 2, 64, None);
    (0..count)
        .map(|_| {
            let support = vec![rand::random::<u32>() as usize % LWR_DIMENSION];
            (0..64)
                .find_map(|_| {
                    generate_key_layout(&parameters, PbcRootSeed(rand::random()), &support).ok()
                })
                .expect("PBC key layout")
        })
        .collect()
}

/// The accepted candidate: the decoder selection, the smudge keys of its members, and the
/// trapdoor sigma.
struct Accepted {
    selection: DecoderSelection,
    keys: Vec<PbcGeneratedKeyLayout>,
    trapdoor_sigma: BigDecimal,
}

/// Search the configured candidates for the decoder of `source`'s output `input`.
fn search(source: &ExponentLutProgram, input: WireId) -> Accepted {
    let crt_bits = setting("CRT_BITS", 30);
    let base_bits = setting("BASE_BITS", crt_bits.div_ceil(2)) as u32;
    assert!((1..=31).contains(&base_bits), "the base bits must be in 1..=31");
    let max_primes_per_drop = setting("MAX_PRIMES_PER_DROP", 4);
    let max_crt_depth = setting("MAX_CRT_DEPTH", 40);
    let smudge_p = setting("SMUDGE_P", 16);
    let statistical_bits = setting("STATISTICAL_BITS", 1);
    let required_security = setting("SECURITY_BITS", 1) as u64;
    assert!(required_security > 0, "the security target must be positive");
    let error_sigma = sigma("ERROR_SIGMA", "4");
    let trapdoor_sigma = sigma("TRAPDOOR_SIGMA", "4.578");
    let average: Option<AverageCaseConfig> =
        noise_model(PREFIX).unwrap_or_else(|error| panic!("{error}"));
    let one = [BigInt::from(1u8)];
    let bindings = RnsBggNoiseBindings::from_inputs(
        source,
        &[RnsBggNoiseValue::sampled_encoding_sigma(&error_sigma, b"decoder-source", &one)
            .expect("source noise binding")],
        &[],
    )
    .expect("source noise bindings");
    let candidates = ring_dimension_candidates(
        &format!("{PREFIX}MIN_LOG_RING_DIMENSION"),
        &format!("{PREFIX}MAX_LOG_RING_DIMENSION"),
        5,
        12,
    )
    .expect("ring-dimension configuration");
    let mut failures = Vec::new();
    for ring_dimension in candidates {
        let Some(stride_log2) = ring_dimension_stride_log2(ring_dimension, PRF_WIDTH_LOG2) else {
            failures.push(format!("n={ring_dimension}: below the PRF input width"));
            continue;
        };
        let profile = SparseLwrPrfProfile::new(
            ExpLayout { stride_log2, width_log2: PRF_WIDTH_LOG2 },
            smudge_p,
            LWR_DIMENSION,
        )
        .expect("smudge PRF profile");
        let generated = RefCell::new(BTreeMap::new());
        let layouts = |members: usize| {
            let keys = smudge_keys(members);
            let layouts = keys.iter().map(|key| key.public_layout.clone()).collect();
            generated.borrow_mut().insert(members, keys);
            Ok(layouts)
        };
        let decoder = DecoderRequest {
            output: input,
            profile,
            layouts: &layouts,
            statistical_bits,
            unmask_sigma: trapdoor_sigma.to_f64().expect("finite trapdoor sigma"),
        };
        'dimension: for primes_per_drop in 1..=max_primes_per_drop {
            for crt_depth in 2..=max_crt_depth {
                let request = RnsBggPlanRequest {
                    ring_dimension,
                    encoding_error_sigma: error_sigma.clone(),
                    crt_bits,
                    base_bits,
                    terminal_depth: 1,
                    max_crt_depth: crt_depth,
                    terminal_value_bound: BigUint::from(1u8),
                    operations_per_drop: 1,
                    primes_per_drop,
                    rhs_mode: RhsMode::Static,
                };
                let selection = match select_decoder_at_crt_depth(
                    &request,
                    source,
                    &bindings,
                    &[BigUint::from(1u8)],
                    average.as_ref(),
                    &decoder,
                ) {
                    Ok(selection) => selection,
                    Err(error) => {
                        failures.push(format!(
                            "n={ring_dimension} primes_per_drop={primes_per_drop} \
                             crt_depth={crt_depth}: {error}"
                        ));
                        continue;
                    }
                };
                // The largest moduli of the two chains bound the security of every level.
                let achieved = [&selection.parameters, &selection.smudge_parameters]
                    .into_iter()
                    .map(|parameters| {
                        lattice_security_bits(
                            parameters.levels()[0].ring_dimension(),
                            parameters.levels()[0].modulus().as_ref(),
                            &error_sigma,
                        )
                        .expect("lattice security estimator must be available")
                    })
                    .min()
                    .expect("two chains");
                if achieved < required_security {
                    // More primes only lower the security of this ring dimension.
                    failures.push(format!(
                        "n={ring_dimension} primes_per_drop={primes_per_drop} \
                         crt_depth={crt_depth}: security {achieved} < {required_security}"
                    ));
                    break 'dimension;
                }
                let noise = &selection.noise;
                eprintln!(
                    "decoder parameters accepted: ring_dimension={ring_dimension} \
                     primes_per_drop={primes_per_drop} crt_depth={crt_depth} crt_bits={crt_bits} \
                     base_bits={base_bits} digits={} members={} noise_model={} error_bits={} \
                     smudge_bits={} spacing_bits={} security_bits={achieved}",
                    selection.declaration.digits(),
                    selection.declaration.smudge().members(),
                    if average.is_some() { "average" } else { "worst" },
                    noise.error.bits(),
                    noise.smudge.bits(),
                    noise.spacing.bits(),
                );
                let members = selection.declaration.smudge().members();
                let keys = generated.borrow_mut().remove(&members).expect("smudge keys");
                return Accepted { selection, keys, trapdoor_sigma };
            }
        }
    }
    panic!("no decoder parameter candidate was accepted: {}", failures.join("; "));
}

#[test]
#[sequential(dcrt_runtime)]
fn decoder_round_trip() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
    assert!(
        !env::var("MXX_EXPONENT_LUT_ESTIMATE_ONLY").is_ok_and(|value| value == "1"),
        "the decoder round trip executes every stage; unset MXX_EXPONENT_LUT_ESTIMATE_ONLY"
    );
    let devices = detected_gpu_device_ids();
    if devices.is_empty() {
        eprintln!("skipping the decoder round trip: no CUDA device detected");
        return;
    }
    let (source, input) = identity_source();
    let Accepted { selection, keys, trapdoor_sigma } = search(&source, input);
    let DecoderSelection { parameters, smudge_parameters, declaration, noise, .. } = selection;
    assert!(noise.fits(), "the decoder noise must fit the decoding margin: {noise:?}");

    let smudge_program = declaration.smudge().program();
    let level = parameters.levels()[0].clone();
    let smudge_level = smudge_parameters.levels()[0].clone();
    let selectors = declaration
        .smudge()
        .prf()
        .selector_plaintexts::<DCRTPoly>(&smudge_level, &keys)
        .expect("smudge selector plaintexts");
    let a = (0..LWR_DIMENSION)
        .map(|_| rand::random::<u32>() as usize % (1 << PRF_WIDTH_LOG2))
        .collect::<Vec<_>>();
    let smudge_lhs = declaration
        .smudge()
        .prf()
        .plaintext_inputs::<DCRTPoly>(&smudge_level, &a)
        .expect("smudge PRF inputs");

    // The two chains share their rings; each ring is registered once.
    let gpu_levels = parameters
        .distinct_levels()
        .into_iter()
        .chain(smudge_parameters.distinct_levels())
        .map(|level| (level.ring_dimension(), level.to_crt().0, level.base_bits()))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .map(|(ring_dimension, moduli, base_bits)| {
            GpuDCRTPolyParams::new(ring_dimension, moduli, base_bits, None)
        })
        .collect::<Vec<_>>();
    let mut runtime = GpuRuntime::new(gpu_backend_on(gpu_levels, devices)).expect("GPU runtime");
    let directory = tempfile::tempdir().expect("artifact directory");
    let mut store = api::StageStorage::File(directory.path().into()).open().expect("stage storage");
    let hash_key: [u8; 32] = rand::random();
    let context: [u8; 32] = rand::random();
    let env = ParamEnv::default();
    let no_rhs = PlaintextInputs::<RhsId, DCRTPoly>::new();

    // Preprocessing: the source and smudge public material, `B` with its trapdoor, and the
    // preimage `K` of the decoder key.
    let payload = RnsEncryptBuild::new(&parameters, context)
        .sample_payload_secret(&mut runtime, &mut store, |_| Ok(()))
        .expect("payload secret");
    let mut smudge_preprocess = RnsPreprocessBuild::new(&smudge_parameters, context);
    smudge_preprocess
        .bind_rhs_inputs(
            smudge_program,
            &env,
            &selectors,
            Some(&payload.at_level(&smudge_parameters, 0).expect("smudge payload")),
            &hash_key,
        )
        .expect("smudge preprocessing inputs");
    smudge_preprocess.bind_decoder_smudge(&declaration).expect("smudge preprocessing decoder");
    let (preprocess, online_eval) = api::preprocess(
        &source,
        &env,
        &no_rhs,
        Some(&payload),
        &hash_key,
        RnsPreprocessBuild::new(&parameters, context),
        &mut runtime,
        &mut store,
        |build, outputs| {
            build.apply_decoder(
                0,
                &declaration,
                outputs[0].clone(),
                smudge_preprocess,
                &trapdoor_sigma,
            )?;
            Ok(outputs)
        },
        |_| Ok(()),
    )
    .expect("decoder preprocessing");

    // Encryption: the source input `X^b`, the smudge PRF inputs, and the unmask hint.
    let mask = RnsEncryptBuild::new(&parameters, rand::random())
        .sample_mask_secret(&mut runtime, &mut store, |_| Ok(()))
        .expect("mask secret");
    let mut smudge_encrypt = RnsEncryptBuild::new(&smudge_parameters, context);
    smudge_encrypt
        .bind_lhs_inputs(
            &smudge_program.encryption_inputs(),
            &env,
            &smudge_lhs,
            &payload.at_level(&smudge_parameters, 0).expect("smudge payload"),
            Some(&mask.at_level(&smudge_parameters, 0).expect("smudge mask")),
            &hash_key,
        )
        .expect("smudge encryption inputs");
    smudge_encrypt.bind_decoder_smudge(&declaration).expect("smudge encryption decoder");
    smudge_encrypt.bind_preprocess_artifacts(&preprocess).expect("smudge preprocess artifacts");
    let bit = rand::random::<u32>() as usize % level.ring_dimension() as usize;
    let lhs =
        BTreeMap::from([(input, InputBinding::Scalar(DCRTPoly::const_rotate_poly(&level, bit)))]);
    let encrypted = api::encrypt(
        &source.encryption_inputs(),
        &env,
        &lhs,
        Some(&preprocess),
        &payload,
        Some(&mask),
        &hash_key,
        RnsEncryptBuild::new(&parameters, context),
        &mut runtime,
        &mut store,
        |build| build.bind_decoder(&declaration, smudge_encrypt),
        |_| Ok(()),
    )
    .expect("decoder encryption");

    // Online evaluation: the smudged, unmasked, and rounded source output.
    let mut smudge_online = RnsOnlineEvalBuild::new(&smudge_parameters, context);
    smudge_online
        .bind_decoder_smudge(&declaration, &online_eval, &encrypted, &hash_key)
        .expect("smudge online evaluation");
    let result = api::online_eval(
        &source.without_lut_values(),
        &env,
        &online_eval,
        &encrypted,
        &hash_key,
        RnsOnlineEvalBuild::new(&parameters, context),
        &mut runtime,
        &mut store,
        |build, outputs| {
            build.apply_decoder(0, &declaration, smudge_online)?;
            Ok(outputs)
        },
        |_| Ok(()),
    )
    .expect("decoder online evaluation");
    let plaintext =
        StageExecution::<api::StageStore>::download_matrix(&runtime, &result.outputs["output-0"])
            .expect("decoded plaintext");

    // The decoded plaintext `round((p/q) x) mod p` is the source input.
    let expected = DCRTPoly::const_rotate_poly(&parameters.levels()[1], bit);
    assert_eq!(plaintext.entry(0, 0), expected, "decoded plaintext");
    eprintln!(
        "decoder round trip: decoded X^{bit} with smudge 2^{} over noise 2^{}",
        noise.smudge.bits(),
        noise.error.bits()
    );
}
