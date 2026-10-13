#![cfg(feature = "gpu")]

//! The noise refresh round trip on the GPU runtime.
//!
//! The refresh ([`mxx_exponent_lut::refresh`]) of the identity Program's private output `X^b` runs
//! through the RNS-BGG application API in three artifact stages:
//!
//! - preprocessing prepares the source Program, the fresh-noise child (CBD polynomials of the
//!   sparse-LWR PRF) and one smudge child per decoder, computes every decoded key `K_i` by the
//!   refresh's homomorphic steps, samples one unmask trapdoor `B`, and publishes `B` and one
//!   preimage per decoder;
//! - encryption encrypts the source input, the children's inputs on the hashed PRF label, the one
//!   encoding, and the unmask hint `b = s B + e_b`;
//! - online evaluation combines the one encoding, the target and the fresh noise into `d_i` per
//!   prime, unmasks and rounds every column with its smudge, and recombines the residues into the
//!   refreshed row `y = s A' - Delta X^b G_Q + e'`.
//!
//! The source chain is `[q_0 ... q_(L-1), q_0]` over 30-bit primes for ring dimension 8 and gadget
//! base `2^15`; the children descend to it by `PRIMES_PER_DROP` primes per level. The digit count
//! is the smallest whose simulated refresh noise fits every decoding margin and hides by
//! `STATISTICAL_BITS`. The round trip checks that `y - (s A' - Delta X^b G_Q)` is exactly the fresh
//! CBD samples of the PRF keys on the label (`|e'| <= eta = 1`). It runs the Static RHS lifecycle
//! only.
//!
//! | Variable (`MXX_EXPONENT_LUT_REFRESH_` prefix) | Default | Meaning |
//! | --- | --- | --- |
//! | `PRIMES` | 2 | CRT primes `L` of the refreshed level |
//! | `PRIMES_PER_DROP` | 2 | Primes per drop of the children's chains |
//! | `ERROR_SIGMA`, `TRAPDOOR_SIGMA` | 0, 4.578 | Encoding error and trapdoor Gaussian widths |
//! | `STATISTICAL_BITS` | 1 | Hiding margin of the smudge over each decoder's noise |

use bigdecimal::{BigDecimal, ToPrimitive};
use mxx_backends::{
    GpuRuntime,
    backend::poly_gpu::gpu_backend_on,
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
            params::DCRTPolyParams,
            poly::DCRTPoly,
        },
    },
};
use mxx_exponent_lut::{
    ExpLayout,
    api::{self, PlaintextInputs},
    program::{
        AttributeKnowledge, ExponentLutProgram, ExponentLutProgramBuilder, InputBinding, Operand,
        RhsId, WireId,
    },
    program_gadgets::{
        pbc::{
            PbcGeneratedKeyLayout, PbcParameters, PbcRootSeed, clear_pbc_inner_product,
            generate_key_layout,
        },
        prf::{PrfKeyShape, SparseLwrPrfProfile},
    },
    refresh::RefreshDeclaration,
    rns_bgg::{
        RnsBggNoiseBindings, RnsBggNoiseValue, RnsBggParameters, RnsBggRefreshNoise,
        RnsEncryptBuild, RnsOnlineEvalBuild, RnsPreprocessBuild, refresh_error,
        refresh_fresh_namespace, simulate_refresh_noise,
    },
    utils::env_usize,
};
use mxx_ir_core::ParamEnv;
use num_bigint::{BigInt, BigUint};
use serial_test::serial as sequential;
use std::{collections::BTreeMap, env};

const PREFIX: &str = "MXX_EXPONENT_LUT_REFRESH_";
/// The LWR dimension of every PRF.
const LWR_DIMENSION: usize = 4;

fn setting(name: &str, default: usize) -> usize {
    let name = format!("{PREFIX}{name}");
    env_usize(&name, default).unwrap_or_else(|error| panic!("{name}: {error}"))
}

fn decimal(name: &str, default: &str) -> BigDecimal {
    let name = format!("{PREFIX}{name}");
    env::var(&name)
        .unwrap_or_else(|_| default.to_owned())
        .parse::<BigDecimal>()
        .unwrap_or_else(|error| panic!("{name} must be a decimal: {error}"))
}

/// The identity Program on one private input, and that input.
fn identity_source() -> (ExponentLutProgram, WireId) {
    let mut builder = ExponentLutProgramBuilder::new();
    let input = builder.input(AttributeKnowledge::PrivatePayload).expect("source input");
    (builder.finish(vec![Operand::Scalar(input)]).expect("identity Program"), input)
}

/// `count` random weight-one PRF keys over the LWR coordinates, in two buckets.
fn keys(count: usize) -> Vec<PbcGeneratedKeyLayout> {
    // Every coordinate lies in both buckets, so each bucket has `nu` real cells and a dummy.
    let parameters = PbcParameters::custom(LWR_DIMENSION, 1, 2, 2, 64, Some(LWR_DIMENSION + 1));
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

/// The key shape of `keys`.
fn shape(keys: &[PbcGeneratedKeyLayout]) -> PrfKeyShape {
    let layouts = keys.iter().map(|key| key.public_layout.clone()).collect::<Vec<_>>();
    PrfKeyShape::new(&layouts).expect("PRF key shape")
}

/// The accepted refresh: its declaration, keys, children chains and noise.
struct Accepted {
    declaration: RefreshDeclaration,
    fresh_keys: Vec<PbcGeneratedKeyLayout>,
    smudge_keys: Vec<Vec<PbcGeneratedKeyLayout>>,
    fresh: RnsBggParameters,
    smudge: RnsBggParameters,
    noise: RnsBggRefreshNoise,
}

/// The smallest digit count whose refresh noise fits and hides.
fn search(
    parameters: &RnsBggParameters,
    pool: &DCRTPolyParams,
    source: &ExponentLutProgram,
    input: WireId,
    trapdoor_sigma: f64,
) -> Accepted {
    let profile = SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 2 }, 4, 4)
        .expect("PRF profile");
    let n = profile.ring_dimension().expect("PRF ring dimension");
    let level = &parameters.levels()[0];
    let (primes, columns) = (level.to_crt().0.len(), level.modulus_digits());
    let per_drop = setting("PRIMES_PER_DROP", 2);
    let statistical_bits = setting("STATISTICAL_BITS", 1);
    let one = [BigInt::from(1u8)];
    let bindings = RnsBggNoiseBindings::from_inputs(
        source,
        &[RnsBggNoiseValue::sampled_encoding(parameters, b"refresh-target", &one)],
        &[],
    )
    .expect("source noise bindings");
    for digits in 1..32 {
        let fresh_keys = keys(columns * n);
        let smudge_keys = (0..primes).map(|_| keys(columns * n * digits)).collect::<Vec<_>>();
        let declaration = RefreshDeclaration::new(
            source,
            input,
            profile,
            1,
            primes,
            columns,
            digits,
            shape(&fresh_keys),
            smudge_keys.iter().map(|keys| shape(keys)).collect(),
            b"refresh-round-trip".to_vec(),
        )
        .expect("refresh declaration");
        let fresh = parameters
            .refresh_fresh_parameters(&declaration, source, pool, per_drop)
            .expect("fresh-noise chain");
        let smudge = parameters
            .decoder_smudge_parameters(&declaration.decoders()[0], source, pool, per_drop)
            .expect("smudge chain");
        let smudges = vec![&smudge; declaration.decoders().len()];
        let noise = simulate_refresh_noise(
            parameters,
            None,
            &fresh,
            &smudges,
            source,
            &bindings,
            &declaration,
            0,
            trapdoor_sigma,
            None,
        )
        .expect("refresh noise");
        if noise.fits() && noise.hides(statistical_bits) {
            return Accepted { declaration, fresh_keys, smudge_keys, fresh, smudge, noise };
        }
        assert!(
            &noise.smudge * 2u8 < noise.spacings[0],
            "no digit count fits the decoding margin: {noise:?}"
        );
    }
    panic!("no digit count fits and hides");
}

/// The CBD sample of fresh member `key` on the label `a`: its clear inner product read through
/// the fresh sampler's tables.
fn fresh_sample(declaration: &RefreshDeclaration, key: &PbcGeneratedKeyLayout, a: &[u64]) -> i64 {
    let sampler = declaration.fresh().sampler();
    let q = sampler.prf().profile().q().expect("PRF modulus");
    let z = clear_pbc_inner_product(&key.public_layout, key.private_schedule(), a, q as u64)
        .expect("clear inner product");
    sampler.sample_at(declaration.fresh().program(), z as usize).expect("fresh sample")
}

#[test]
#[sequential(dcrt_runtime)]
fn refresh_round_trip() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
    let devices = detected_gpu_device_ids();
    if devices.is_empty() {
        eprintln!("skipping the refresh round trip: no CUDA device detected");
        return;
    }
    let primes = setting("PRIMES", 2);
    let trapdoor_sigma = decimal("TRAPDOOR_SIGMA", "4.578");
    let pool = DCRTPolyParams::new(8, primes + 10, 30, 15, None, None);
    let prefix = |depth: usize| {
        let modulus = pool.to_crt().0[..depth].iter().map(|prime| BigUint::from(*prime)).product();
        pool.select_modulus(&modulus).expect("CRT prefix")
    };
    let parameters =
        RnsBggParameters::new(vec![prefix(primes), prefix(1)], decimal("ERROR_SIGMA", "0"), 1)
            .expect("source chain");
    let (source, input) = identity_source();
    let Accepted { declaration, fresh_keys, smudge_keys, fresh, smudge, noise } = search(
        &parameters,
        &pool,
        &source,
        input,
        trapdoor_sigma.to_f64().expect("finite trapdoor sigma"),
    );
    eprintln!(
        "refresh parameters accepted: primes={primes} columns={} decoders={} digits={} \
         error_bits={:?} smudge_bits={} spacing_bits={:?}",
        declaration.columns(),
        declaration.decoders().len(),
        declaration.digits(),
        noise.errors.iter().map(BigUint::bits).collect::<Vec<_>>(),
        noise.smudge.bits(),
        noise.spacings.iter().map(BigUint::bits).collect::<Vec<_>>()
    );

    let gpu_levels = parameters
        .distinct_levels()
        .into_iter()
        .chain(fresh.distinct_levels())
        .chain(smudge.distinct_levels())
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
    let fresh_namespace = refresh_fresh_namespace(&declaration);
    let fresh_program = declaration.fresh().program();
    let fresh_level = fresh.levels()[0].clone();
    let smudge_level = smudge.levels()[0].clone();

    // Preprocessing: the children's public material, the decoded keys, `B` and the preimages.
    let payload = RnsEncryptBuild::new(&parameters, context)
        .sample_payload_secret(&mut runtime, &mut store, |_| Ok(()))
        .expect("payload secret");
    let mut fresh_preprocess =
        RnsPreprocessBuild::new(&fresh, context).with_hash_namespace(fresh_namespace);
    let (fresh_rhs, fresh_tables) = declaration
        .fresh()
        .sampler()
        .key_inputs::<DCRTPoly>(&fresh_level, &fresh_keys)
        .expect("fresh selectors and tables");
    fresh_preprocess
        .bind_rhs_inputs(
            fresh_program,
            &env,
            Some((&fresh_rhs, &fresh_tables)),
            Some(&payload.at_level(&fresh, 0).expect("fresh payload")),
            &hash_key,
        )
        .expect("fresh preprocessing inputs");
    let smudge_preprocess = declaration
        .decoders()
        .iter()
        .zip(&smudge_keys)
        .map(|(decoder, keys)| {
            let mut child = RnsPreprocessBuild::new(&smudge, context)
                .with_hash_namespace(*decoder.commitment());
            let (rhs, tables) = decoder
                .smudge()
                .prf()
                .key_inputs::<DCRTPoly>(&smudge_level, keys)
                .expect("smudge selectors and tables");
            child
                .bind_rhs_inputs(
                    decoder.smudge().program(),
                    &env,
                    Some((&rhs, &tables)),
                    Some(&payload.at_level(&smudge, 0).expect("smudge payload")),
                    &hash_key,
                )
                .expect("smudge preprocessing inputs");
            child.bind_decoder_smudge(decoder).expect("smudge preprocessing decoder");
            child
        })
        .collect::<Vec<_>>();
    let no_rhs = PlaintextInputs::<RhsId, DCRTPoly>::new();
    let (preprocess, online_eval) = api::preprocess(
        &source,
        &env,
        &no_rhs,
        &BTreeMap::new(),
        Some(&payload),
        &hash_key,
        RnsPreprocessBuild::new(&parameters, context),
        &mut runtime,
        &mut store,
        |build, outputs| {
            build.apply_refresh(
                0,
                &declaration,
                outputs[0].clone(),
                fresh_preprocess,
                smudge_preprocess,
                &trapdoor_sigma,
            )?;
            Ok(outputs)
        },
        |_| Ok(()),
    )
    .expect("refresh preprocessing");

    // Encryption: the source input `X^b`, the children's inputs on the hashed label, the one
    // encoding and the unmask hint.
    let mask = RnsEncryptBuild::new(&parameters, rand::random())
        .sample_mask_secret(&mut runtime, &mut store, |_| Ok(()))
        .expect("mask secret");
    let label_tag = b"invocation-0";
    let a = declaration.label(hash_key, label_tag).expect("PRF label");
    let child_encryption = |fresh_child: bool,
                            namespace: [u8; 32],
                            program: &ExponentLutProgram,
                            lhs: &PlaintextInputs<WireId, DCRTPoly>| {
        let chain = if fresh_child { &fresh } else { &smudge };
        let mut child = RnsEncryptBuild::new(chain, context).with_hash_namespace(namespace);
        child
            .bind_lhs_inputs(
                &program.encryption_inputs(),
                &env,
                Some(lhs),
                &payload.at_level(chain, 0).expect("child payload"),
                Some(&mask.at_level(chain, 0).expect("child mask")),
                &hash_key,
            )
            .expect("child encryption inputs");
        child.bind_preprocess_artifacts(&preprocess).expect("child preprocess artifacts");
        child
    };
    let fresh_encrypt = child_encryption(
        true,
        fresh_namespace,
        fresh_program,
        &declaration
            .fresh()
            .sampler()
            .plaintext_inputs::<DCRTPoly>(&fresh_level, &a)
            .expect("fresh inputs"),
    );
    let smudge_encrypt = declaration
        .decoders()
        .iter()
        .map(|decoder| {
            let mut child = child_encryption(
                false,
                *decoder.commitment(),
                decoder.smudge().program(),
                &decoder
                    .smudge()
                    .prf()
                    .plaintext_inputs::<DCRTPoly>(&smudge_level, &a)
                    .expect("smudge inputs"),
            );
            child.bind_decoder_smudge(decoder).expect("smudge encryption decoder");
            child
        })
        .collect::<Vec<_>>();
    let level = parameters.levels()[0].clone();
    let bit = rand::random::<u32>() as usize % level.ring_dimension() as usize;
    let lhs =
        BTreeMap::from([(input, InputBinding::Scalar(DCRTPoly::const_rotate_poly(&level, bit)))]);
    let encrypted = api::encrypt(
        &source.encryption_inputs(),
        &env,
        &lhs,
        &preprocess,
        &payload,
        Some(&mask),
        &hash_key,
        RnsEncryptBuild::new(&parameters, context),
        &mut runtime,
        &mut store,
        |build| build.bind_refresh(&declaration, fresh_encrypt, smudge_encrypt),
        |_| Ok(()),
    )
    .expect("refresh encryption");

    // Online evaluation: the refreshed row.
    let child_online =
        |fresh_child: bool,
         namespace: [u8; 32],
         decoder: Option<&mxx_exponent_lut::decoder::decoding::DecoderDeclaration>| {
            let chain = if fresh_child { &fresh } else { &smudge };
            let mut child = RnsOnlineEvalBuild::new(chain, context).with_hash_namespace(namespace);
            child
                .bind_refresh_child(
                    &declaration,
                    decoder,
                    label_tag,
                    &online_eval,
                    &encrypted,
                    &hash_key,
                )
                .expect("refresh child online evaluation");
            child
        };
    let fresh_online = child_online(true, fresh_namespace, None);
    let smudge_online = declaration
        .decoders()
        .iter()
        .map(|decoder| child_online(false, *decoder.commitment(), Some(decoder)))
        .collect::<Vec<_>>();
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
            build.apply_refresh(0, &declaration, fresh_online, smudge_online)?;
            Ok(outputs)
        },
        |_| Ok(()),
    )
    .expect("refresh online evaluation");

    // The refreshed row's error is exactly the fresh CBD noise of the PRF.
    let error = refresh_error(
        &declaration,
        &source,
        &result,
        &parameters,
        &payload,
        Some(&mask),
        &DCRTPoly::const_rotate_poly(&level, bit),
        &hash_key,
        &mut runtime,
        &mut store,
    )
    .expect("refresh error");
    let n = level.ring_dimension() as usize;
    let a = a.iter().map(|value| *value as u64).collect::<Vec<_>>();
    for (column, coefficients) in error.iter().enumerate() {
        let expected = (0..n)
            .map(|k| BigInt::from(fresh_sample(&declaration, &fresh_keys[column * n + k], &a)))
            .collect::<Vec<_>>();
        assert_eq!(coefficients, &expected, "refresh error of column {column}");
    }
    let eta = BigInt::from(declaration.eta());
    assert!(error.iter().flatten().all(|value| value.magnitude() <= eta.magnitude()));
    eprintln!(
        "refresh round trip: X^{bit} refreshed over {} decoders with fresh error at most {eta}",
        declaration.decoders().len()
    );
}
