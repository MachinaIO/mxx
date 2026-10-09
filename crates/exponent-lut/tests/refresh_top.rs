#![cfg(feature = "gpu")]

//! The refresh to the top level on the GPU runtime.
//!
//! The refresh ([`mxx_exponent_lut::refresh::top`]) of an exponent LUT output `x = X^(1 - b)` on
//! the private input `X^b` re-encrypts it as both rows of a level-0 encoding under a new
//! PRF-derived mask `sigma = (sigma_0, 1)`, in the format of a reference chain. It evaluates the
//! source Program on one chain per prime `q` of `Q_0` (each keeps `q` to the plaintext level), and
//! on every chain the two parts (one per component) run the three artifact stages:
//!
//! - preprocessing prepares the source Program on the chain (under the chain's hash namespace), the
//!   mask child (a binary polynomial `sigma_0`), the fresh-noise child, one smudge child per
//!   decoder, the decoded keys, one unmask trapdoor and the preimages;
//! - encryption encrypts the source input on the chain, the children's inputs on the shared hashed
//!   PRF label, the one encoding and the unmask hint;
//! - online evaluation combines `[mask carrier | one encoding]`, the target component and the fresh
//!   noise into the decoded row of `q`, unmasks and rounds every column with its smudge, and
//!   outputs the residue `y_j mod q`.
//!
//! The residues of every chain are recombined modulo `Q_0`, and the round trip checks that
//! `y_j - (sigma A'_(j) - Delta_0 r_j x G^st_0)` is exactly the fresh CBD samples of component
//! `j`'s PRF keys on the label, with `sigma_0` the bits of the mask keys.
//!
//! The reference has `PRIMES` 30-bit primes (ring dimension 8, base `2^15`) at level 0 and one
//! fewer at level 1; the source chains have `[PRIMES, PRIMES - 1, PRIMES - 2]` primes, so the
//! decoding level keeps the output prime and `PRIMES - 2` margin primes. The digit count is the
//! smallest whose simulated noise of every part fits its decoding margin and hides by
//! `STATISTICAL_BITS`. It runs the Static RHS lifecycle only.
//!
//! | Variable (`MXX_EXPONENT_LUT_REFRESH_TOP_` prefix) | Default | Meaning |
//! | --- | --- | --- |
//! | `PRIMES` | 4 | Primes of the reference's level 0 (`>= 3`) |
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
    api::{self, PlaintextInputs, StageArtifacts},
    program::{
        AttributeKnowledge, ExponentLutProgram, ExponentLutProgramBuilder, InputBinding,
        LutOutputRepresentation, LutTable, Operand, RhsId, WireId,
    },
    program_gadgets::{
        pbc::{
            PbcGeneratedKeyLayout, PbcParameters, PbcRootSeed, clear_pbc_inner_product,
            generate_key_layout,
        },
        prf::SparseLwrPrfProfile,
        prf_sampler::PrfSampler,
    },
    refresh::{
        TopReference,
        top::{TopRefreshDeclaration, TopRefreshLayouts},
    },
    rns_bgg::{
        RnsBggMaskSecret, RnsBggNoiseBindings, RnsBggNoiseValue, RnsBggParameters,
        RnsBggPayloadSecret, RnsBggPreprocessEncryptDescriptor, RnsEncryptBuild,
        RnsOnlineEvalBuild, RnsPreprocessBuild, refresh_fresh_namespace, refresh_mask_namespace,
        simulate_refresh_noise, top_refresh_error,
    },
    utils::env_usize,
};
use mxx_ir_core::ParamEnv;
use num_bigint::{BigInt, BigUint};
use serial_test::serial as sequential;
use std::{cell::RefCell, collections::BTreeMap, env};

const PREFIX: &str = "MXX_EXPONENT_LUT_REFRESH_TOP_";
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

/// One private input `X^b` (`b < 2`) mapped by the exponent LUT `X^e -> X^(1 - e)`, that input and
/// the output.
fn lut_source() -> (ExponentLutProgram, WireId, WireId) {
    let mut builder = ExponentLutProgramBuilder::new();
    let input = builder.input(AttributeKnowledge::PrivatePayload).expect("source input");
    let layout = ExpLayout { stride_log2: 0, width_log2: 1 };
    let table = builder
        .register_lut(
            LutTable::unary(2, layout, 2, vec![1, 0], LutOutputRepresentation::Exponent(layout))
                .expect("exponent LUT"),
        )
        .expect("LUT registration");
    let output = builder.lut(Operand::Scalar(input), table).expect("LUT output");
    (builder.finish(vec![Operand::Scalar(output)]).expect("source Program"), input, output)
}

/// `count` random weight-one PRF keys over the LWR coordinates, in two buckets.
fn keys(count: usize) -> Vec<PbcGeneratedKeyLayout> {
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

fn layouts(
    keys: &[PbcGeneratedKeyLayout],
) -> Vec<mxx_exponent_lut::program_gadgets::pbc::PbcPublicLayout> {
    keys.iter().map(|key| key.public_layout.clone()).collect()
}

/// The PRF profile of every child: `n' = 4`, `p' = 2` (`P = 4`, `Q = 8`) in the ring of dimension
/// eight.
fn profile() -> SparseLwrPrfProfile {
    SparseLwrPrfProfile::new(ExpLayout { stride_log2: 1, width_log2: 2 }, 4, LWR_DIMENSION)
        .expect("PRF profile")
}

/// The keys of one top refresh.
struct Keys {
    mask: Vec<PbcGeneratedKeyLayout>,
    fresh: [Vec<PbcGeneratedKeyLayout>; 2],
    smudge: BTreeMap<(usize, usize, usize), Vec<PbcGeneratedKeyLayout>>,
}

fn top_refresh(
    source: &ExponentLutProgram,
    output: WireId,
    reference: &TopReference,
    extension: &[u64],
    digits: usize,
) -> (TopRefreshDeclaration, Keys) {
    let n = 8;
    let columns = reference.columns();
    let primes = reference.top.len();
    let mask = keys(n);
    let fresh = [keys(columns * n), keys(columns * n)];
    let smudge = RefCell::new(BTreeMap::new());
    let mut sample = |chain: usize, component: usize, column: usize| {
        let generated = keys(n * digits);
        let public = layouts(&generated);
        smudge.borrow_mut().insert((chain, component, column), generated);
        public
    };
    let top = TopRefreshDeclaration::new(
        source,
        output,
        profile(),
        1,
        reference.clone(),
        vec![primes, primes - 1, primes - 2],
        extension.to_vec(),
        digits,
        TopRefreshLayouts {
            mask: layouts(&mask),
            fresh: [layouts(&fresh[0]), layouts(&fresh[1])],
            smudge: &mut sample,
        },
        b"top-refresh-round-trip".to_vec(),
    )
    .expect("top refresh declaration");
    (top, Keys { mask, fresh, smudge: smudge.into_inner() })
}

/// The chains of a part: its source chain, pool, and mask, fresh and smudge children.
struct Chains {
    pool: DCRTPolyParams,
    source: RnsBggParameters,
    mask: RnsBggParameters,
    fresh: RnsBggParameters,
    smudge: RnsBggParameters,
}

fn chains(
    top: &TopRefreshDeclaration,
    source: &ExponentLutProgram,
    chain: usize,
    sigma: &BigDecimal,
) -> Chains {
    let (pool, parameters) =
        RnsBggParameters::top_refresh_chain(top, Some(chain), sigma.clone(), 1).expect("chain");
    let part = top.part(chain, 0).expect("part");
    Chains {
        mask: parameters.refresh_mask_parameters(part, source, &pool, 1).expect("mask chain"),
        fresh: parameters.refresh_fresh_parameters(part, source, &pool, 1).expect("fresh chain"),
        smudge: parameters
            .decoder_smudge_parameters(&part.decoders()[0], source, &pool, 1)
            .expect("smudge chain"),
        source: parameters,
        pool,
    }
}

/// Whether every part's simulated noise fits and hides by `bits`.
fn accepted(
    top: &TopRefreshDeclaration,
    source: &ExponentLutProgram,
    sigma: &BigDecimal,
    trapdoor_sigma: f64,
    bits: usize,
) -> bool {
    (0..top.chains()).all(|chain| {
        let chains = chains(top, source, chain, sigma);
        let one = [BigInt::from(1u8)];
        let bindings = RnsBggNoiseBindings::from_inputs(
            source,
            &[RnsBggNoiseValue::sampled_encoding(&chains.source, b"top-target", &one)],
            &[],
        )
        .expect("source noise bindings");
        (0..2).all(|component| {
            let part = top.part(chain, component).expect("part");
            let smudges = vec![&chains.smudge; part.decoders().len()];
            let noise = simulate_refresh_noise(
                &chains.source,
                Some(&chains.mask),
                &chains.fresh,
                &smudges,
                source,
                &bindings,
                part,
                0,
                trapdoor_sigma,
                None,
            )
            .expect("refresh noise");
            noise.fits() && noise.hides(bits)
        })
    })
}

/// The sample of member `key` of `sampler` on the label `a`.
fn sample(
    sampler: &PrfSampler,
    program: &ExponentLutProgram,
    key: &PbcGeneratedKeyLayout,
    a: &[u64],
) -> i64 {
    let q = sampler.prf().profile().q().expect("PRF modulus");
    let z = clear_pbc_inner_product(&key.public_layout, key.private_schedule(), a, q as u64)
        .expect("clear inner product");
    sampler.sample_at(program, z as usize).expect("sample")
}

fn child_preprocess<'p>(
    context: [u8; 32],
    payload: &RnsBggPayloadSecret,
    hash_key: [u8; 32],
    parameters: &'p RnsBggParameters,
    namespace: [u8; 32],
    program: &ExponentLutProgram,
    rhs: PlaintextInputs<RhsId, DCRTPoly>,
) -> RnsPreprocessBuild<'p> {
    let mut child = RnsPreprocessBuild::new(parameters, context).with_hash_namespace(namespace);
    child
        .bind_rhs_inputs(
            program,
            &ParamEnv::default(),
            &rhs,
            Some(&payload.at_level(parameters, 0).expect("child payload")),
            &hash_key,
        )
        .expect("child preprocessing inputs");
    child
}

fn child_encrypt<'p>(
    (context, hash_key, preprocess): (
        [u8; 32],
        [u8; 32],
        &StageArtifacts<RnsBggPreprocessEncryptDescriptor>,
    ),
    (payload, mask): (&RnsBggPayloadSecret, &RnsBggMaskSecret),
    parameters: &'p RnsBggParameters,
    namespace: [u8; 32],
    program: &ExponentLutProgram,
    lhs: PlaintextInputs<WireId, DCRTPoly>,
) -> RnsEncryptBuild<'p> {
    let mut child = RnsEncryptBuild::new(parameters, context).with_hash_namespace(namespace);
    child
        .bind_lhs_inputs(
            &program.encryption_inputs(),
            &ParamEnv::default(),
            &lhs,
            &payload.at_level(parameters, 0).expect("child payload"),
            Some(&mask.at_level(parameters, 0).expect("child mask")),
            &hash_key,
        )
        .expect("child encryption inputs");
    child.bind_preprocess_artifacts(preprocess).expect("child preprocess artifacts");
    child
}

/// One part's preprocessing, encryption and online evaluation on its chain.
#[allow(clippy::too_many_arguments)]
fn run_part(
    top: &TopRefreshDeclaration,
    keys: &Keys,
    source: &ExponentLutProgram,
    chains: &Chains,
    lhs: &PlaintextInputs<WireId, DCRTPoly>,
    (payload, mask): (&RnsBggPayloadSecret, &RnsBggMaskSecret),
    (chain, component): (usize, usize),
    (hash_key, label_tag, trapdoor_sigma): ([u8; 32], &[u8], &BigDecimal),
    runtime: &mut GpuRuntime,
    store: &mut api::StageStore,
) -> api::StageOutputs {
    let part = top.part(chain, component).expect("part");
    let details = part.top().expect("top part");
    let context: [u8; 32] = rand::random();
    let env = ParamEnv::default();
    let mask_level = chains.mask.levels()[0].clone();
    let fresh_level = chains.fresh.levels()[0].clone();
    let smudge_level = chains.smudge.levels()[0].clone();
    let chain_namespace = top.chain_namespace(chain);
    let matrix_namespace = top.matrix_namespace();
    let mask_namespace = refresh_mask_namespace(part);
    let fresh_namespace = refresh_fresh_namespace(part);
    let smudge_keys = |column: usize| &keys.smudge[&(chain, component, column)];

    let mask_preprocess = child_preprocess(
        context,
        payload,
        hash_key,
        &chains.mask,
        mask_namespace,
        details.mask().program(),
        details.mask().sampler().rhs_plaintexts(&mask_level, &keys.mask).expect("mask RHS"),
    );
    let fresh_preprocess = child_preprocess(
        context,
        payload,
        hash_key,
        &chains.fresh,
        fresh_namespace,
        part.fresh().program(),
        part.fresh()
            .sampler()
            .rhs_plaintexts(&fresh_level, &keys.fresh[component])
            .expect("fresh RHS"),
    );
    let smudge_preprocess = part
        .decoders()
        .iter()
        .enumerate()
        .map(|(column, decoder)| {
            let mut child = child_preprocess(
                context,
                payload,
                hash_key,
                &chains.smudge,
                *decoder.commitment(),
                decoder.smudge().program(),
                decoder
                    .smudge()
                    .prf()
                    .selector_plaintexts(&smudge_level, smudge_keys(column))
                    .expect("smudge selectors"),
            );
            child.bind_decoder_smudge(decoder).expect("smudge preprocessing decoder");
            child
        })
        .collect::<Vec<_>>();
    let no_rhs = PlaintextInputs::<RhsId, DCRTPoly>::new();
    let chain_payload = payload.at_level(&chains.source, 0).expect("chain payload");
    let (preprocess, online_eval) = api::preprocess(
        source,
        &env,
        &no_rhs,
        Some(&chain_payload),
        &hash_key,
        RnsPreprocessBuild::new(&chains.source, context)
            .with_hash_namespace(chain_namespace)
            .with_refresh_matrix_key(matrix_namespace, &hash_key),
        runtime,
        store,
        |build, outputs| {
            build.apply_top_refresh(
                0,
                part,
                outputs[0].clone(),
                mask_preprocess,
                fresh_preprocess,
                smudge_preprocess,
                trapdoor_sigma,
            )?;
            Ok(outputs)
        },
        |_| Ok(()),
    )
    .expect("top refresh preprocessing");

    let a = top.label(hash_key, label_tag).expect("PRF label");
    let mask_encrypt = child_encrypt(
        (context, hash_key, &preprocess),
        (payload, mask),
        &chains.mask,
        mask_namespace,
        details.mask().program(),
        details.mask().sampler().plaintext_inputs(&mask_level, &a).expect("mask inputs"),
    );
    let fresh_encrypt = child_encrypt(
        (context, hash_key, &preprocess),
        (payload, mask),
        &chains.fresh,
        fresh_namespace,
        part.fresh().program(),
        part.fresh().sampler().plaintext_inputs(&fresh_level, &a).expect("fresh inputs"),
    );
    let smudge_encrypt = part
        .decoders()
        .iter()
        .map(|decoder| {
            let mut child = child_encrypt(
                (context, hash_key, &preprocess),
                (payload, mask),
                &chains.smudge,
                *decoder.commitment(),
                decoder.smudge().program(),
                decoder.smudge().prf().plaintext_inputs(&smudge_level, &a).expect("smudge inputs"),
            );
            child.bind_decoder_smudge(decoder).expect("smudge encryption decoder");
            child
        })
        .collect::<Vec<_>>();
    let encrypted = api::encrypt(
        &source.encryption_inputs(),
        &env,
        lhs,
        &preprocess,
        &chain_payload,
        Some(&mask.at_level(&chains.source, 0).expect("chain mask")),
        &hash_key,
        RnsEncryptBuild::new(&chains.source, context).with_hash_namespace(chain_namespace),
        runtime,
        store,
        |build| build.bind_top_refresh(part, mask_encrypt, fresh_encrypt, smudge_encrypt),
        |_| Ok(()),
    )
    .expect("top refresh encryption");

    let child_online = |parameters, namespace| {
        RnsOnlineEvalBuild::new(parameters, context).with_hash_namespace(namespace)
    };
    let mut mask_online = child_online(&chains.mask, mask_namespace);
    mask_online
        .bind_refresh_mask_child(part, label_tag, &online_eval, &encrypted, &hash_key)
        .expect("mask online evaluation");
    let mut fresh_online = child_online(&chains.fresh, fresh_namespace);
    fresh_online
        .bind_refresh_child(part, None, label_tag, &online_eval, &encrypted, &hash_key)
        .expect("fresh online evaluation");
    let smudge_online = part
        .decoders()
        .iter()
        .map(|decoder| {
            let mut child = child_online(&chains.smudge, *decoder.commitment());
            child
                .bind_refresh_child(
                    part,
                    Some(decoder),
                    label_tag,
                    &online_eval,
                    &encrypted,
                    &hash_key,
                )
                .expect("smudge online evaluation");
            child
        })
        .collect::<Vec<_>>();
    api::online_eval(
        &source.without_lut_values(),
        &env,
        &online_eval,
        &encrypted,
        &hash_key,
        RnsOnlineEvalBuild::new(&chains.source, context)
            .with_hash_namespace(chain_namespace)
            .with_refresh_matrix_key(matrix_namespace, &hash_key),
        runtime,
        store,
        |build, outputs| {
            build.apply_top_refresh(0, part, mask_online, fresh_online, smudge_online)?;
            Ok(outputs)
        },
        |_| Ok(()),
    )
    .expect("top refresh online evaluation")
}

#[test]
#[sequential(dcrt_runtime)]
fn refresh_top_round_trip() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
    let devices = detected_gpu_device_ids();
    if devices.is_empty() {
        eprintln!("skipping the top refresh round trip: no CUDA device detected");
        return;
    }
    let primes = setting("PRIMES", 4);
    assert!(primes >= 3, "{PREFIX}PRIMES must be at least 3");
    let sigma = decimal("ERROR_SIGMA", "0");
    let trapdoor_sigma = decimal("TRAPDOOR_SIGMA", "4.578");
    let bits = setting("STATISTICAL_BITS", 1);
    let pool = DCRTPolyParams::new(8, primes + 14, 30, 15, None, None);
    let reference = TopReference {
        top: pool.moduli()[..primes].to_vec(),
        next: pool.moduli()[..primes - 1].to_vec(),
        ring_dimension: 8,
        base_bits: 15,
    };
    let extension = pool.moduli()[primes..].to_vec();
    let (source, input, output) = lut_source();
    let trapdoor = trapdoor_sigma.to_f64().expect("finite trapdoor sigma");
    let digits = (1..40)
        .find(|digits| {
            let (top, _) = top_refresh(&source, output, &reference, &extension, *digits);
            accepted(&top, &source, &sigma, trapdoor, bits)
        })
        .expect("a fitting and hiding digit count");
    let (top, keys) = top_refresh(&source, output, &reference, &extension, digits);
    eprintln!(
        "top refresh accepted: chains={} columns={} digits={digits}",
        top.chains(),
        reference.columns()
    );

    let (_, reference_chain) =
        RnsBggParameters::top_refresh_chain(&top, None, sigma.clone(), 1).expect("reference chain");
    let all_chains =
        (0..top.chains()).map(|chain| chains(&top, &source, chain, &sigma)).collect::<Vec<_>>();
    let key =
        |level: &DCRTPolyParams| (level.ring_dimension(), level.to_crt().0, level.base_bits());
    let mut levels = reference_chain.distinct_levels().into_iter().map(key).collect::<Vec<_>>();
    for (chain, chains) in all_chains.iter().enumerate() {
        for parameters in [&chains.source, &chains.mask, &chains.fresh, &chains.smudge] {
            levels.extend(parameters.distinct_levels().into_iter().map(key));
        }
        let prime = top.output_prime(chain).expect("output prime");
        levels.push(key(&chains.pool.select_modulus(&BigUint::from(prime)).expect("output ring")));
    }
    let gpu_levels = levels
        .into_iter()
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
    let label_tag = b"top-invocation-0";
    let context: [u8; 32] = rand::random();
    let payload = RnsEncryptBuild::new(&reference_chain, context)
        .sample_payload_secret(&mut runtime, &mut store, |_| Ok(()))
        .expect("payload secret");
    let mask = RnsEncryptBuild::new(&reference_chain, context)
        .sample_mask_secret(&mut runtime, &mut store, |_| Ok(()))
        .expect("mask secret");
    let bit = rand::random::<u32>() as usize % 2;
    let mut results = Vec::with_capacity(top.chains());
    for (chain, chains) in all_chains.iter().enumerate() {
        let lhs = BTreeMap::from([(
            input,
            InputBinding::Scalar(DCRTPoly::const_rotate_poly(&chains.source.levels()[0], bit)),
        )]);
        let parts = [0, 1].map(|component| {
            run_part(
                &top,
                &keys,
                &source,
                chains,
                &lhs,
                (&payload, &mask),
                (chain, component),
                (hash_key, label_tag, &trapdoor_sigma),
                &mut runtime,
                &mut store,
            )
        });
        eprintln!("top refresh chain {chain} evaluated");
        results.push(parts);
    }

    let level = reference_chain.levels()[0].clone();
    let n = level.ring_dimension() as usize;
    let a = top
        .label(hash_key, label_tag)
        .expect("PRF label")
        .iter()
        .map(|value| *value as u64)
        .collect::<Vec<_>>();
    let mask_program = top.part(0, 0).expect("part").top().expect("top part").mask();
    // The sampler reads `+-h`; the Program adds `h`, so the bit is one exactly for `+h`.
    let sigma_0 = (0..n)
        .map(|k| {
            BigUint::from(u8::from(
                sample(mask_program.sampler(), mask_program.program(), &keys.mask[k], &a) > 0,
            ))
        })
        .collect::<Vec<_>>();
    let sigma_0 = DCRTPoly::from_biguints(&level, &sigma_0);
    let expected = DCRTPoly::const_rotate_poly(&level, 1 - bit);
    let results = results.iter().map(|[first, second]| [first, second]).collect::<Vec<_>>();
    let errors = top_refresh_error(
        &top,
        &results,
        &reference_chain,
        &payload,
        &sigma_0,
        &expected,
        &hash_key,
        &mut runtime,
        &mut store,
    )
    .expect("top refresh error");
    for (component, rows) in errors.iter().enumerate() {
        let part = top.part(0, component).expect("part");
        for (column, coefficients) in rows.iter().enumerate() {
            let expected = (0..n)
                .map(|k| {
                    BigInt::from(sample(
                        part.fresh().sampler(),
                        part.fresh().program(),
                        &keys.fresh[component][column * n + k],
                        &a,
                    ))
                })
                .collect::<Vec<_>>();
            assert_eq!(coefficients, &expected, "component {component}, column {column}");
        }
    }
    eprintln!(
        "top refresh round trip: X^{} re-encrypted over {} chains with fresh error at most 1",
        1 - bit,
        top.chains()
    );
}
