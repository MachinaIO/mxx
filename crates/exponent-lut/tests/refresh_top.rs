#![cfg(feature = "gpu")]

//! The refresh to the top level on the GPU runtime.
//!
//! The refresh ([`mxx_exponent_lut::refresh`]) of an exponent LUT output `x = X^(1 - b)` on
//! the private input `X^b` re-encrypts it as both rows of a level-0 encoding under a new
//! PRF-derived mask `sigma = (sigma_0, 1)`, in the format of a reference chain. It refreshes the
//! first `SLOTS` CRT slots of `Q_0`: it evaluates the source Program on one chain per refreshed
//! prime `q` (each keeps `q` to the plaintext level), and
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
//! The residues of every chain are recombined modulo the product `P` of the refreshed primes, and
//! the round trip checks that `y_j - (sigma A'_(j) - Delta_0 r_j x G^st_0)` modulo `P` is exactly
//! the fresh CBD samples of component `j`'s PRF keys on the label, with `sigma_0` the bits of the
//! mask keys.
//!
//! The reference has `PRIMES` primes of `CRT_BITS` bits (ring dimension `RING_DIMENSION`, base
//! `2^BASE_BITS`) at level 0 and one fewer at level 1; the source chains have
//! `[PRIMES, PRIMES - 1, PRIMES - 2]` primes, so the decoding level keeps the output prime and
//! `PRIMES - 2` margin primes, and the children's chains extend them by `EXTENSION_PRIMES` primes.
//! The digit count is `DIGITS`, or the smallest whose simulated noise of every part fits its
//! decoding margin and hides by `STATISTICAL_BITS` under the average-case model (failure
//! probability `2^-FAILURE_EXPONENT` per accepted bound). `SECURITY_BITS > 0` also requires the
//! estimated security of the encodings on the largest modulus and of the refreshed output to reach
//! that many bits, and `SIMULATE_ONLY=1` stops after these checks. It runs the Static RHS lifecycle
//! only.
//!
//! | Variable (`MXX_EXPONENT_LUT_REFRESH_TOP_` prefix) | Default | Meaning |
//! | --- | --- | --- |
//! | `RING_DIMENSION`, `CRT_BITS`, `BASE_BITS` | 8, 30, 15 | Ring dimension, prime size and gadget base |
//! | `PRIMES` | 4 | Primes of the reference's level 0 (`>= 3`) |
//! | `EXTENSION_PRIMES` | 14 | Primes beyond level 0 for the children's chains |
//! | `SLOTS` | 1 | Refreshed CRT slots, the first primes of level 0 (`1..=PRIMES`) |
//! | `LWR_DIMENSION`, `SUPPORT_WEIGHT` | 4, 1 | Sparse-LWR dimension and key weight of every PRF |
//! | `BUCKET_WIDTH` | 0 | Row width `W` of a PBC layout of weight above one (`0`: [`PbcParameters::default_bucket_width_limit`]) |
//! | `INPUT_MODULUS`, `OUTPUT_MODULUS` | 8, 4 | Sparse-LWR input and output moduli |
//! | `ERROR_SIGMA`, `TRAPDOOR_SIGMA` | 0, 4.578 | Encoding error and trapdoor Gaussian widths |
//! | `DIGITS` | 0 | Smudge digit count (`0`: the smallest in `MIN_DIGITS..=MAX_DIGITS`, 1 and 39) |
//! | `STATISTICAL_BITS` | 1 | Hiding margin of the smudge over each decoder's noise |
//! | `FAILURE_EXPONENT` | 100 | Average-case failure probability exponent of the noise checks |
//! | `SECURITY_BITS` | 0 | Required estimated security (`0`: not checked) |
//! | `SIMULATE_ONLY` | unset | `1`: only the accepted parameters and the checks above |
//! | `NOISE_MODEL` | `worst` | Claim of an export without a recorded set: `worst`, or `average` (failure probability `2^-FAILURE_EXPONENT`, the sampling tape) |
//!
//! The noise and the graphs depend on the key shapes only. The digit search, `SIMULATE_ONLY` and an
//! `EXPORT_ONLY` run without a recorded set generate no PRF keys: they declare each key set by a
//! commitment derived from `KEY_SEED` and its role (see `declared_shape`). A round trip generates
//! its keys (one layout per PRF member) and commits to their digests, as does the export of a
//! recorded set, so that it is the claim its round trip exports. An export binds the children's
//! selector plaintexts, tables and PRF inputs by shape only.
mod common;

use bigdecimal::{BigDecimal, ToPrimitive};
use mxx_backends::{
    GpuRuntime, RuntimeValue,
    authority::CpuExecution,
    backend::poly_gpu::gpu_backend_on,
    matrix::dcrt_poly::DCRTPolyMatrix,
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
            params::DCRTPolyParams,
            poly::DCRTPoly,
        },
    },
};
use mxx_dsl::{DslContext, Family, FamilyType, IdealSpec, Int, IntType};
use mxx_exponent_lut::{
    ExpLayout,
    api::{self, PlaintextInputs, StageArtifacts},
    decoder::simulation::AverageCaseConfig,
    program::{
        AttributeKnowledge, ExponentLutProgram, ExponentLutProgramBuilder, InputBinding,
        LutOutputRepresentation, LutTable, Operand, ProgramData, RhsId, WireId,
    },
    program_gadgets::{
        pbc::{
            PbcGeneratedKeyLayout, PbcParameters, PbcRootSeed, clear_pbc_inner_product,
            generate_key_layout,
        },
        prf::{PrfKeyShape, SparseLwrPrf, SparseLwrPrfProfile},
        prf_sampler::PrfSampler,
    },
    refresh::{RefreshDeclaration, RefreshKeys, TopReference},
    rns_bgg::{
        REFRESH_TERMS_INPUT, RnsBggMaskSecret, RnsBggNoiseBindings, RnsBggNoiseValue,
        RnsBggParameters, RnsBggPayloadSecret, RnsBggPreprocessEncryptDescriptor, RnsEncryptBuild,
        RnsOnlineEvalBuild, RnsPreprocessBuild, data_input_name, decode_refresh,
        plaintext_input_prefix, refresh_error, simulate_refresh_noise,
    },
    utils::{env_usize, refresh_output_security_bits},
};
use mxx_ir_core::{
    Graph, IntExpr, ParamEnv, ValidatedGraph,
    protocol::{
        ClosedProtocolBundle, ComparatorEndpointBinding, ComparatorSpec, EndpointBinding,
        EndpointBindings, EndpointSemanticBinding, EndpointSpecId, InputContract,
        InputContractEntry, InputValueContract, OutputRef, ProtocolDecl, ProtocolInputBinding,
        ProtocolInputDestination, ProtocolPreconditionSpec, PurePredicateSpec, StageId, Workflow,
    },
};
use num_bigint::{BigInt, BigUint};
use rand::{Rng, rngs::StdRng};
use serial_test::serial as sequential;
use std::{cell::RefCell, collections::BTreeMap, env};

const PREFIX: &str = "MXX_EXPONENT_LUT_REFRESH_TOP_";
/// The stage hash key, context and invocation label tag (fixed, so that a recorded set exports
/// the same claim on every run).
const HASH_KEY: &[u8; 32] = b"refresh-top-rns-public-api-key-1";
const CONTEXT: [u8; 32] = *b"refresh-top-rns-public-api-ctx-1";
const LABEL_TAG: &[u8] = b"top-invocation-0";

type ParameterRow = BTreeMap<String, String>;
/// The sparse-LWR PRF of every child: LWR dimension `nu`, key weight `h`, input modulus `Q`,
/// output modulus `P` and, for a weight above one, the PBC row width `W`
/// (`MXX_EXPONENT_LUT_REFRESH_TOP_{LWR_DIMENSION,SUPPORT_WEIGHT,INPUT_MODULUS,OUTPUT_MODULUS,
/// BUCKET_WIDTH}`).
#[derive(Clone, Copy, Debug)]
struct PrfSettings {
    lwr_dimension: usize,
    support_weight: usize,
    input_modulus: usize,
    output_modulus: usize,
    bucket_width: usize,
}

fn prf_settings() -> PrfSettings {
    let settings = PrfSettings {
        lwr_dimension: setting("LWR_DIMENSION", 4),
        support_weight: setting("SUPPORT_WEIGHT", 1),
        input_modulus: setting("INPUT_MODULUS", 8),
        output_modulus: setting("OUTPUT_MODULUS", 4),
        bucket_width: setting("BUCKET_WIDTH", 0),
    };
    assert!(
        settings.support_weight > 0 && settings.support_weight <= settings.lwr_dimension,
        "{PREFIX}SUPPORT_WEIGHT must be in 1..={PREFIX}LWR_DIMENSION"
    );
    assert!(
        settings.input_modulus >= 4 && settings.input_modulus.is_power_of_two(),
        "{PREFIX}INPUT_MODULUS must be a power of two at least 4"
    );
    assert!(
        settings.output_modulus >= 2 &&
            settings.output_modulus % 2 == 0 &&
            settings.input_modulus % settings.output_modulus == 0,
        "{PREFIX}OUTPUT_MODULUS must be even and divide {PREFIX}INPUT_MODULUS"
    );
    settings
}

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

impl PrfSettings {
    /// The PBC parameters of every key: weight one in two buckets (every coordinate in both, so
    /// the row width is `nu + 1`), otherwise the paper's PBC layout of `h + 3` blocks of row width
    /// `BUCKET_WIDTH`.
    fn parameters(&self) -> PbcParameters {
        let nu = self.lwr_dimension;
        if self.support_weight == 1 {
            return PbcParameters::custom(nu, 1, 2, 2, 64, Some(nu + 1));
        }
        let parameters = PbcParameters::paper_evaluation(nu, self.support_weight);
        let width = match self.bucket_width {
            0 => parameters.default_bucket_width_limit(),
            width => width,
        };
        PbcParameters { bucket_width_limit: Some(width), ..parameters }
    }
}

/// `count` PRF keys of weight `h` over the `nu` LWR coordinates on [`PrfSettings::parameters`],
/// drawn from `rng` (layouts wider than the row width are retried).
fn keys(rng: &mut StdRng, count: usize, prf: PrfSettings) -> Vec<PbcGeneratedKeyLayout> {
    let (nu, parameters) = (prf.lwr_dimension, prf.parameters());
    (0..count)
        .map(|_| {
            let support = match prf.support_weight {
                1 => vec![rng.random::<u32>() as usize % nu],
                weight => rand::seq::index::sample(rng, nu, weight).into_vec(),
            };
            (0..64)
                .find_map(|_| {
                    generate_key_layout(&parameters, PbcRootSeed(rng.random()), &support).ok()
                })
                .expect("PBC key layout")
        })
        .collect()
}

/// The key shape of `keys`, which commits to their layouts.
fn shape(keys: &[PbcGeneratedKeyLayout]) -> PrfKeyShape {
    let layouts = keys.iter().map(|key| key.public_layout.clone()).collect::<Vec<_>>();
    PrfKeyShape::new(&layouts).expect("PRF key shape")
}

/// The shape of `members` keys on [`PrfSettings::parameters`] that are never generated, for the
/// digit search and a run that executes no stage (`SIMULATE_ONLY`, or `EXPORT_ONLY` without a
/// recorded set): its key-set commitment and table digest are derived from the key seed `seed`,
/// the role `role` of the keys and the shape, not from layouts, so such a run declares the refresh
/// without materializing one layout per member (`w_0 n d` per smudge child). Its claim has the
/// graphs of a run with generated keys up to these commitments (the hash namespaces derived from
/// them, and the order of the hash-key externals named after them).
fn declared_shape(seed: u64, role: &str, members: usize, prf: PrfSettings) -> PrfKeyShape {
    use sha2::{Digest, Sha256};
    let parameters = prf.parameters();
    let digest = |kind: &[u8]| -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"mxx-exponent-lut/refresh-top-test/declared-keys/v1");
        digest.update(kind);
        digest.update(seed.to_le_bytes());
        digest.update((role.len() as u64).to_le_bytes());
        digest.update(role.as_bytes());
        digest.update((members as u64).to_le_bytes());
        digest.update(serde_json::to_vec(&parameters).expect("PBC parameters encoding"));
        digest.finalize().into()
    };
    PrfKeyShape::committed(parameters.clone(), members, digest(b"key-set"), digest(b"tables"))
        .expect("declared PRF key shape")
}

/// The PRF profile of every child in the ring of dimension `n`: `X^(2n / Q z)` adds exponents
/// modulo `Q`, so `width_log2 = log2 Q - 1` and `stride_log2 = log2 n - width_log2`.
fn profile(n: usize, prf: PrfSettings) -> SparseLwrPrfProfile {
    let width_log2 = prf.input_modulus.ilog2() - 1;
    let stride_log2 = n.ilog2().checked_sub(width_log2).unwrap_or_else(|| {
        panic!("{PREFIX}INPUT_MODULUS {} exceeds twice the ring dimension {n}", prf.input_modulus)
    });
    SparseLwrPrfProfile::new(
        ExpLayout { stride_log2, width_log2 },
        prf.output_modulus,
        prf.lwr_dimension,
    )
    .expect("PRF profile")
}

/// The keys of one top refresh.
struct Keys {
    mask: Vec<PbcGeneratedKeyLayout>,
    fresh: [Vec<PbcGeneratedKeyLayout>; 2],
    /// Every part's smudge keys, `w_0 n d` (column `j` from `j n d`).
    smudge: BTreeMap<(usize, usize), Vec<PbcGeneratedKeyLayout>>,
}

/// The top refresh of `digits` smudge digits with keys drawn from the key seed `seed`, and those
/// keys; with `generate` false, the keys are only declared ([`declared_shape`]).
#[allow(clippy::too_many_arguments)]
fn top_refresh(
    seed: u64,
    generate: bool,
    source: &ExponentLutProgram,
    output: WireId,
    reference: &TopReference,
    slots: usize,
    extension: &[u64],
    digits: usize,
    prf: PrfSettings,
) -> (RefreshDeclaration, Option<Keys>) {
    let n = reference.ring_dimension as usize;
    let columns = reference.columns();
    let primes = reference.top.len();
    let mut rng = <StdRng as rand::SeedableRng>::seed_from_u64(seed);
    let rng = RefCell::new(&mut rng);
    let keys = |role: &str, count: usize| {
        if generate {
            let generated = keys(&mut rng.borrow_mut(), count, prf);
            (shape(&generated), generated)
        } else {
            (declared_shape(seed, role, count, prf), Vec::new())
        }
    };
    let mask = keys("mask", n);
    let fresh = [keys("fresh-0", columns * n), keys("fresh-1", columns * n)];
    let smudge = RefCell::new(BTreeMap::new());
    let mut sample = |chain: usize, component: usize| {
        let (shape, generated) = keys(&format!("smudge-{chain}-{component}"), columns * n * digits);
        smudge.borrow_mut().insert((chain, component), generated);
        shape
    };
    let top = RefreshDeclaration::new(
        source,
        output,
        profile(n, prf),
        1,
        reference.clone(),
        vec![primes, primes - 1, primes - 2],
        (0..slots).collect(),
        extension.to_vec(),
        digits,
        RefreshKeys {
            mask: mask.0.clone(),
            fresh: [fresh[0].0.clone(), fresh[1].0.clone()],
            smudge: &mut sample,
        },
        b"top-refresh-round-trip".to_vec(),
    )
    .expect("top refresh declaration");
    let [fresh_0, fresh_1] = fresh;
    let keys = Keys { mask: mask.1, fresh: [fresh_0.1, fresh_1.1], smudge: smudge.into_inner() };
    (top, generate.then_some(keys))
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
    top: &RefreshDeclaration,
    source: &ExponentLutProgram,
    chain: usize,
    sigma: &BigDecimal,
) -> Chains {
    let (pool, parameters) =
        RnsBggParameters::refresh_chain(top, Some(chain), sigma.clone(), 1).expect("chain");
    let decoder = top.decoder((chain, 0)).expect("decoder");
    Chains {
        mask: parameters.refresh_mask_parameters(top, chain, source, &pool, 1).expect("mask chain"),
        fresh: parameters
            .refresh_fresh_parameters(top, chain, source, &pool, 1)
            .expect("fresh chain"),
        smudge: parameters
            .decoder_smudge_parameters(decoder, source, &pool, 1)
            .expect("smudge chain"),
        source: parameters,
        pool,
    }
}

/// Whether every part's simulated noise fits and hides by `bits` under the average-case noise
/// model with failure probability `2^-failure_exponent` per accepted bound.
fn accepted(
    top: &RefreshDeclaration,
    source: &ExponentLutProgram,
    sigma: &BigDecimal,
    trapdoor_sigma: f64,
    bits: usize,
    failure_exponent: u32,
) -> bool {
    let average = AverageCaseConfig {
        failure_exponent,
        allow_average_acceptance: true,
        ..AverageCaseConfig::default()
    };
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
            let noise = simulate_refresh_noise(
                &chains.source,
                &chains.mask,
                &chains.fresh,
                &chains.smudge,
                source,
                &bindings,
                top,
                (chain, component),
                0,
                trapdoor_sigma,
                Some(&average),
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

/// A child's preprocessing, bound to its RHS plaintexts and coordinate tables `(rhs, data)`, or,
/// without keys, to their shapes only.
fn child_preprocess<'p>(
    context: [u8; 32],
    payload: &RnsBggPayloadSecret,
    hash_key: [u8; 32],
    parameters: &'p RnsBggParameters,
    namespace: [u8; 32],
    program: &ExponentLutProgram,
    values: Option<(PlaintextInputs<RhsId, DCRTPoly>, ProgramData)>,
) -> RnsPreprocessBuild<'p> {
    let (rhs, data) = values.unzip();
    let mut child = RnsPreprocessBuild::new(parameters, context)
        .with_hash_namespace(namespace)
        .with_data(data.unwrap_or_default());
    child
        .bind_rhs_inputs(
            program,
            &ParamEnv::default(),
            rhs.as_ref(),
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
    lhs: Option<PlaintextInputs<WireId, DCRTPoly>>,
) -> RnsEncryptBuild<'p> {
    let mut child = RnsEncryptBuild::new(parameters, context).with_hash_namespace(namespace);
    child
        .bind_lhs_inputs(
            &program.encryption_inputs(),
            &ParamEnv::default(),
            lhs.as_ref(),
            &payload.at_level(parameters, 0).expect("child payload"),
            Some(&mask.at_level(parameters, 0).expect("child mask")),
            &hash_key,
        )
        .expect("child encryption inputs");
    child.bind_preprocess_artifacts(preprocess).expect("child preprocess artifacts");
    child
}

/// One part's preprocessing, encryption and online evaluation on its chain. Without `keys`, the
/// children bind their plaintexts and tables by shape only, so the stages can be recorded but not
/// executed.
#[allow(clippy::too_many_arguments)]
fn run_part(
    top: &RefreshDeclaration,
    keys: Option<&Keys>,
    source: &ExponentLutProgram,
    chains: &Chains,
    lhs: &PlaintextInputs<WireId, DCRTPoly>,
    (payload, mask): (&RnsBggPayloadSecret, &RnsBggMaskSecret),
    (chain, component): (usize, usize),
    (hash_key, label_tag, trapdoor_sigma): ([u8; 32], &[u8], &BigDecimal),
    runtime: &mut impl api::StageExecution<api::StageStore>,
    store: &mut api::StageStore,
    graphs: &StageGraphs,
) -> (api::StageOutputs, [mxx_ir_core::artifact::ProductionId; 2]) {
    let part = (chain, component);
    let name = |stage: &str| format!("{stage}_{chain}_{component}");
    let mask_program = top.mask(chain).expect("mask child");
    let fresh_program = top.fresh(part).expect("fresh child");
    let decoder = top.decoder(part).expect("decoder");
    // One preprocessing session per part.
    let mut context = CONTEXT;
    context[30] = chain as u8;
    context[31] = component as u8;
    let env = ParamEnv::default();
    let mask_level = chains.mask.levels()[0].clone();
    let fresh_level = chains.fresh.levels()[0].clone();
    let smudge_level = chains.smudge.levels()[0].clone();
    let chain_namespace = top.chain_namespace(chain);
    let matrix_namespace = top.matrix_namespace();
    let mask_namespace = top.mask_namespace(part).expect("mask namespace");
    let fresh_namespace = top.fresh_namespace(part).expect("fresh namespace");

    let mask_preprocess = child_preprocess(
        context,
        payload,
        hash_key,
        &chains.mask,
        mask_namespace,
        mask_program.program(),
        keys.map(|keys| {
            mask_program.sampler().key_inputs(&mask_level, &keys.mask).expect("mask RHS")
        }),
    );
    let fresh_preprocess = child_preprocess(
        context,
        payload,
        hash_key,
        &chains.fresh,
        fresh_namespace,
        fresh_program.program(),
        keys.map(|keys| {
            let fresh = &keys.fresh[component];
            fresh_program.sampler().key_inputs(&fresh_level, fresh).expect("fresh RHS")
        }),
    );
    let mut smudge_preprocess = child_preprocess(
        context,
        payload,
        hash_key,
        &chains.smudge,
        *decoder.commitment(),
        decoder.smudge().program(),
        keys.map(|keys| {
            let smudge = &keys.smudge[&(chain, component)];
            decoder.smudge().prf().key_inputs(&smudge_level, smudge).expect("smudge RHS")
        }),
    );
    smudge_preprocess.bind_decoder_smudge(decoder).expect("smudge preprocessing decoder");
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
            build.apply_refresh(
                0,
                top,
                part,
                outputs[0].clone(),
                mask_preprocess,
                fresh_preprocess,
                smudge_preprocess,
                trapdoor_sigma,
            )?;
            Ok(outputs)
        },
        record(graphs, name("preprocess")),
    )
    .expect("top refresh preprocessing");

    let a = top.label(hash_key, label_tag).expect("PRF label");
    let mask_encrypt = child_encrypt(
        (context, hash_key, &preprocess),
        (payload, mask),
        &chains.mask,
        mask_namespace,
        mask_program.program(),
        keys.map(|_| {
            mask_program.sampler().plaintext_inputs(&mask_level, &a).expect("mask inputs")
        }),
    );
    let fresh_encrypt = child_encrypt(
        (context, hash_key, &preprocess),
        (payload, mask),
        &chains.fresh,
        fresh_namespace,
        fresh_program.program(),
        keys.map(|_| {
            fresh_program.sampler().plaintext_inputs(&fresh_level, &a).expect("fresh inputs")
        }),
    );
    let mut smudge_encrypt = child_encrypt(
        (context, hash_key, &preprocess),
        (payload, mask),
        &chains.smudge,
        *decoder.commitment(),
        decoder.smudge().program(),
        keys.map(|_| {
            decoder.smudge().prf().plaintext_inputs(&smudge_level, &a).expect("smudge inputs")
        }),
    );
    smudge_encrypt.bind_decoder_smudge(decoder).expect("smudge encryption decoder");
    let encrypted = api::encrypt(
        &source.encryption_inputs(),
        &env,
        lhs,
        Some(&preprocess),
        &chain_payload,
        Some(&mask.at_level(&chains.source, 0).expect("chain mask")),
        &hash_key,
        RnsEncryptBuild::new(&chains.source, context).with_hash_namespace(chain_namespace),
        runtime,
        store,
        |build| build.bind_refresh(top, part, mask_encrypt, fresh_encrypt, smudge_encrypt),
        record(graphs, name("encrypt")),
    )
    .expect("top refresh encryption");

    let child_online = |parameters, namespace| {
        RnsOnlineEvalBuild::new(parameters, context).with_hash_namespace(namespace)
    };
    let mut mask_online = child_online(&chains.mask, mask_namespace);
    mask_online
        .bind_refresh_mask_child(top, part, label_tag, &online_eval, &encrypted, &hash_key)
        .expect("mask online evaluation");
    let mut fresh_online = child_online(&chains.fresh, fresh_namespace);
    fresh_online
        .bind_refresh_fresh_child(top, part, label_tag, &online_eval, &encrypted, &hash_key)
        .expect("fresh online evaluation");
    let mut smudge_online = child_online(&chains.smudge, *decoder.commitment());
    smudge_online
        .bind_refresh_smudge_child(top, part, label_tag, &online_eval, &encrypted, &hash_key)
        .expect("smudge online evaluation");
    let outputs = api::online_eval(
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
            build.apply_refresh(0, top, part, mask_online, fresh_online, smudge_online)?;
            Ok(outputs)
        },
        record(graphs, name("online_eval")),
    )
    .expect("top refresh online evaluation");
    (outputs, [preprocess.production().clone(), encrypted.production().clone()])
}

/// The recorded parameter set selected by `MXX_EXPONENT_LUT_REFRESH_TOP_PARAMETER_ID`, if any.
fn recorded_parameters() -> Option<(String, ParameterRow)> {
    env::var(format!("{PREFIX}PARAMETER_ID"))
        .ok()
        .map(|id| (id.clone(), common::parameter_row("refresh_top", &id)))
}

/// `MXX_EXPONENT_LUT_REFRESH_TOP_EXPORT_ONLY=1`: build and export the claim without a round trip.
/// The children bind their plaintexts and tables by shape only, and the keys of a set without a
/// recorded row are only declared ([`declared_shape`]).
fn export_only() -> bool {
    env::var(format!("{PREFIX}EXPORT_ONLY")).is_ok_and(|value| value == "1")
}

/// `MXX_EXPONENT_LUT_REFRESH_TOP_SIMULATE_ONLY=1`: only the accepted parameters, their simulated
/// noise and the security gate, with declared keys.
fn simulate_only() -> bool {
    env::var(format!("{PREFIX}SIMULATE_ONLY")).is_ok_and(|value| value == "1")
}

/// `MXX_EXPONENT_LUT_REFRESH_TOP_LEAN_FACTS=1`: also export each module's facts module
/// (`<module>_Facts.lean`: wire constants and named atoms, see
/// `mxx_ir_core::lean::protocol::export`).
fn lean_facts() -> bool {
    env::var(format!("{PREFIX}LEAN_FACTS")).is_ok_and(|value| value == "1")
}

/// A recorded column, or else the setting `name` with `default`.
fn row_setting(row: Option<&ParameterRow>, column: &str, name: &str, default: usize) -> usize {
    row.map_or_else(|| setting(name, default), |row| common::column(row, column))
}

fn row_decimal(row: Option<&ParameterRow>, column: &str, name: &str, default: &str) -> BigDecimal {
    row.map_or_else(|| decimal(name, default), |row| common::column(row, column))
}

/// A stage execution that executes only the secret sampling, on the CPU. Every other stage is
/// validated and recorded by its caller, returns the production its execution would have (which
/// depends only on the validated graph), and yields no outputs.
struct GraphOnly(CpuExecution);

impl api::StageExecution<api::StageStore> for GraphOnly {
    fn produce(
        &mut self,
        validated: ValidatedGraph,
        _: BTreeMap<String, RuntimeValue>,
        _: &mut api::StageStore,
        execution_nonce: [u8; 32],
    ) -> Result<mxx_ir_core::artifact::ProductionId, api::ApiError> {
        let specification =
            mxx_ir_core::encoding::spec_hash(&validated.source, &validated.bindings)
                .expect("stage specification hash");
        Ok(mxx_ir_core::artifact::production_id(specification, execution_nonce))
    }

    fn evaluate(
        &mut self,
        validated: ValidatedGraph,
        inputs: BTreeMap<String, RuntimeValue>,
        store: &mut api::StageStore,
        execution_nonce: [u8; 32],
    ) -> Result<api::StageOutputs, api::ApiError> {
        if validated.source.name() == "rns-bgg-secret" {
            self.0.evaluate(validated, inputs, store, execution_nonce)
        } else {
            Ok(api::StageOutputs { outputs: BTreeMap::new() })
        }
    }

    fn download_matrix(&self, value: &RuntimeValue) -> Result<DCRTPolyMatrix, api::ApiError> {
        api::StageExecution::<api::StageStore>::download_matrix(&self.0, value)
    }

    fn download_integers(&self, value: &RuntimeValue) -> Result<Vec<BigInt>, api::ApiError> {
        api::StageExecution::<api::StageStore>::download_integers(&self.0, value)
    }
}

/// Graphs of the executed stages, recorded for the exported claim.
type StageGraphs = RefCell<BTreeMap<String, Graph>>;

/// A stage observer that records the validated graph of `stage`.
fn record<'a>(
    graphs: &'a StageGraphs,
    stage: String,
) -> impl FnOnce(&ValidatedGraph) -> Result<(), api::ApiError> + 'a {
    move |validated| {
        graphs.borrow_mut().insert(stage, validated.source.clone());
        Ok(())
    }
}

/// The accepted refresh and its chains.
struct Setup {
    source: ExponentLutProgram,
    input: WireId,
    top: RefreshDeclaration,
    /// The generated keys; `None` for a run that executes no stage.
    keys: Option<Keys>,
    reference_chain: RnsBggParameters,
    all_chains: Vec<Chains>,
    trapdoor_sigma: BigDecimal,
}

fn setup(row: Option<&ParameterRow>) -> Setup {
    let primes = row_setting(row, "primes", "PRIMES", 4);
    assert!(primes >= 3, "{PREFIX}PRIMES must be at least 3");
    let slots = row_setting(row, "slots", "SLOTS", 1);
    assert!((1..=primes).contains(&slots), "{PREFIX}SLOTS must be in 1..={PREFIX}PRIMES");
    let sigma = row_decimal(row, "error_sigma", "ERROR_SIGMA", "0");
    let trapdoor_sigma = row_decimal(row, "trapdoor_sigma", "TRAPDOOR_SIGMA", "4.578");
    let bits = setting("STATISTICAL_BITS", 1);
    let failure_exponent = setting("FAILURE_EXPONENT", 100) as u32;
    let seed = row_setting(row, "key_seed", "KEY_SEED", 0) as u64;
    let ring_dimension = setting("RING_DIMENSION", 8);
    assert!(ring_dimension.is_power_of_two(), "{PREFIX}RING_DIMENSION must be a power of two");
    let crt_bits = setting("CRT_BITS", 30);
    let base_bits = setting("BASE_BITS", 15);
    let extension_primes = setting("EXTENSION_PRIMES", 14);
    let prf = prf_settings();
    let pool = DCRTPolyParams::new(
        ring_dimension as u32,
        primes + extension_primes,
        crt_bits,
        base_bits as u32,
        None,
        None,
    );
    let reference = TopReference {
        top: pool.moduli()[..primes].to_vec(),
        next: pool.moduli()[..primes - 1].to_vec(),
        ring_dimension: ring_dimension as u32,
        base_bits: base_bits as u32,
    };
    let extension = pool.moduli()[primes..].to_vec();
    let (source, input, output) = lut_source();
    let trapdoor = trapdoor_sigma.to_f64().expect("finite trapdoor sigma");
    let refresh = |digits: usize, generate: bool| {
        top_refresh(seed, generate, &source, output, &reference, slots, &extension, digits, prf)
    };
    let fits =
        |top: &RefreshDeclaration| accepted(top, &source, &sigma, trapdoor, bits, failure_exponent);
    // A recorded set or `DIGITS` fixes the digit count; otherwise the smallest one in
    // `MIN_DIGITS..=MAX_DIGITS` whose simulated noise fits and hides.
    let digits = match (row, setting("DIGITS", 0)) {
        (Some(row), _) => common::column(row, "digits"),
        (None, 0) => (setting("MIN_DIGITS", 1)..=setting("MAX_DIGITS", 39))
            .find(|digits| fits(&refresh(*digits, false).0))
            .expect("a fitting and hiding digit count"),
        (None, digits) => digits,
    };
    // The noise depends on the key shapes only: the digit search declares them, and only a run
    // that executes the stages, or exports a recorded set (whose claim is then the one its round
    // trip exports), generates the keys.
    let (top, keys) = refresh(digits, !simulate_only() && (!export_only() || row.is_some()));
    assert!(fits(&top), "the digit count {digits} fits and hides");
    let (_, reference_chain) =
        RnsBggParameters::refresh_chain(&top, None, sigma.clone(), 1).expect("reference chain");
    let all_chains =
        (0..top.chains()).map(|chain| chains(&top, &source, chain, &sigma)).collect::<Vec<_>>();
    eprintln!(
        "top refresh accepted: n={ring_dimension} crt_bits={crt_bits} base_bits={base_bits} \
         primes={primes} slots={slots} chains={} columns={} digits={digits} {prf:?} \
         statistical_bits={bits} failure_exponent={failure_exponent}",
        top.chains(),
        reference.columns()
    );
    security_gate(&top, &all_chains, &sigma, setting("SECURITY_BITS", 0) as u64);
    Setup { source, input, top, keys, reference_chain, all_chains, trapdoor_sigma }
}

/// `SECURITY_BITS > 0`: the estimated security of the encodings on the largest modulus of every
/// chain and child (ternary secret, Gaussian error of width `sigma`) and of the refreshed output
/// (binary mask, CBD(1) error, `2 w_0 n` samples over `Q_0`) must reach `target` bits.
fn security_gate(top: &RefreshDeclaration, chains: &[Chains], sigma: &BigDecimal, target: u64) {
    if target == 0 {
        return;
    }
    let largest = chains
        .iter()
        .flat_map(|chains| [&chains.source, &chains.mask, &chains.fresh, &chains.smudge])
        .map(|parameters| parameters.level(0).expect("level 0").clone())
        .max_by_key(|level| level.modulus().bits())
        .expect("a chain");
    let encoding = mxx_backends::lattice_estimator::lattice_security_bits(
        largest.ring_dimension(),
        largest.modulus().as_ref(),
        sigma,
    )
    .expect("encoding security estimate");
    let reference = top.reference();
    let modulus = reference.top.iter().map(|prime| BigUint::from(*prime)).product::<BigUint>();
    let n = reference.ring_dimension;
    let output =
        refresh_output_security_bits(n, &modulus, 1, (2 * reference.columns() * n as usize) as u64)
            .expect("output security estimate");
    eprintln!(
        "top refresh security: largest modulus {} bits ({} primes) encoding={encoding} \
         output={output} target={target}",
        largest.modulus().bits(),
        largest.to_crt().0.len()
    );
    assert!(encoding >= target && output >= target, "the security estimate reaches {target} bits");
}

/// Every part's stages on `runtime`, the decode stage, and, for a recorded set, the export of
/// the claim of the recorded graphs.
fn round_trip(
    setup: Setup,
    runtime: &mut impl api::StageExecution<api::StageStore>,
    mut store: api::StageStore,
    recorded: Option<(String, ParameterRow)>,
) {
    let Setup { source, input, top, keys, reference_chain, all_chains, trapdoor_sigma } = setup;
    let graphs = StageGraphs::default();
    let hash_key = *HASH_KEY;
    let label_tag = LABEL_TAG;
    let payload = RnsEncryptBuild::new(&reference_chain, CONTEXT)
        .sample_payload_secret(runtime, &mut store, record(&graphs, "payload".to_owned()))
        .expect("payload secret");
    let mask = RnsEncryptBuild::new(&reference_chain, CONTEXT)
        .sample_mask_secret(runtime, &mut store, record(&graphs, "mask".to_owned()))
        .expect("mask secret");
    let bit = 0;
    let mut results = Vec::with_capacity(top.chains());
    let mut productions = BTreeMap::new();
    for (chain, chains) in all_chains.iter().enumerate() {
        let lhs = BTreeMap::from([(
            input,
            InputBinding::Scalar(DCRTPoly::const_rotate_poly(&chains.source.levels()[0], bit)),
        )]);
        let parts = [0, 1].map(|component| {
            let (outputs, [preprocess, encrypt]) = run_part(
                &top,
                keys.as_ref().filter(|_| !export_only()),
                &source,
                chains,
                &lhs,
                (&payload, &mask),
                (chain, component),
                (hash_key, label_tag, &trapdoor_sigma),
                runtime,
                &mut store,
                &graphs,
            );
            productions.insert(preprocess, format!("preprocess_{chain}_{component}"));
            productions.insert(encrypt, format!("encrypt_{chain}_{component}"));
            outputs
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
    let mask_program = top.mask(0).expect("mask child");
    // The sampler reads `+-h`; the Program adds `h`, so the bit is one exactly for `+h`.
    // Without keys (an export), the decode graph is only recorded, so any `sigma_0` serves.
    let sigma_0 = match &keys {
        Some(keys) => DCRTPoly::from_biguints(
            &level,
            &(0..n)
                .map(|k| {
                    let bit =
                        sample(mask_program.sampler(), mask_program.program(), &keys.mask[k], &a);
                    BigUint::from(u8::from(bit > 0))
                })
                .collect::<Vec<_>>(),
        ),
        None => DCRTPoly::const_zero(&level),
    };
    let expected = DCRTPoly::const_rotate_poly(&level, 1 - bit);
    let results = results.iter().map(|[first, second]| [first, second]).collect::<Vec<_>>();
    if export_only() {
        let decoded = decode_refresh(
            &top,
            None,
            &reference_chain,
            &payload,
            &sigma_0,
            &expected,
            &hash_key,
            runtime,
            &mut store,
            record(&graphs, "decode".to_owned()),
        );
        assert!(decoded.is_err(), "the export-only decode graph is not executed");
        let directory = export_directory(recorded.as_ref());
        let row = recorded.as_ref().map(|(_, row)| row);
        export_claim(
            &directory,
            row,
            &top,
            input,
            graphs.into_inner(),
            &productions,
            (&payload, &mask),
        );
        return;
    }
    let keys = keys.expect("a round trip generates the keys");
    let errors = refresh_error(
        &top,
        &results,
        &reference_chain,
        &payload,
        &sigma_0,
        &expected,
        &hash_key,
        runtime,
        &mut store,
    )
    .expect("top refresh error");
    for (component, rows) in errors.iter().enumerate() {
        let fresh = top.fresh((0, component)).expect("fresh child");
        for (column, coefficients) in rows.iter().enumerate() {
            let expected = (0..n)
                .map(|k| {
                    BigInt::from(sample(
                        fresh.sampler(),
                        fresh.program(),
                        &keys.fresh[component][column * n + k],
                        &a,
                    ))
                })
                .collect::<Vec<_>>();
            assert_eq!(coefficients, &expected, "component {component}, column {column}");
        }
    }
    let decoded = decode_refresh(
        &top,
        Some(&results),
        &reference_chain,
        &payload,
        &sigma_0,
        &expected,
        &hash_key,
        runtime,
        &mut store,
        record(&graphs, "decode".to_owned()),
    )
    .expect("top refresh decode");
    assert_eq!(decoded, BigInt::from(0), "every refreshed coefficient decodes to zero");
    eprintln!(
        "top refresh round trip: X^{} re-encrypted over {} chains with fresh error at most 1",
        1 - bit,
        top.chains()
    );
    if let Some(recorded) = &recorded {
        let directory = export_directory(Some(recorded));
        let row = Some(&recorded.1);
        export_claim(
            &directory,
            row,
            &top,
            input,
            graphs.into_inner(),
            &productions,
            (&payload, &mask),
        );
    }
}

/// The failure probability exponent of the exported claim, `None` for a worst-case claim: a
/// recorded set's `failure_exponent` unless its `noise_model` is `worst`; without a recorded set,
/// `FAILURE_EXPONENT` when `NOISE_MODEL` is `average`.
fn claim_failure_exponent(row: Option<&ParameterRow>) -> Option<u32> {
    match row {
        Some(row) => {
            (row["noise_model"] != "worst").then(|| common::column(row, "failure_exponent"))
        }
        None => match env::var(format!("{PREFIX}NOISE_MODEL")).as_deref() {
            Ok("average") => Some(setting("FAILURE_EXPONENT", 100) as u32),
            Ok("worst") | Err(_) => None,
            Ok(other) => panic!("{PREFIX}NOISE_MODEL: `worst` or `average`, not `{other}`"),
        },
    }
}

/// The directory receiving the exported claim: `lean/refresh_top/<id>/generated` for a recorded
/// set, else `MXX_EXPONENT_LUT_REFRESH_TOP_EXPORT_DIRECTORY` (an export-only run of the settings
/// without a recorded row, under the noise model `NOISE_MODEL`).
fn export_directory(recorded: Option<&(String, ParameterRow)>) -> std::path::PathBuf {
    match recorded {
        Some((id, _)) => common::generated_directory("refresh_top", id),
        None => env::var(format!("{PREFIX}EXPORT_DIRECTORY"))
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                panic!("export-only mode needs {PREFIX}PARAMETER_ID or {PREFIX}EXPORT_DIRECTORY")
            }),
    }
}

/// Export the claim of the recorded stage graphs to `directory`.
fn export_claim(
    directory: &std::path::Path,
    row: Option<&ParameterRow>,
    top: &RefreshDeclaration,
    input: WireId,
    graphs: BTreeMap<String, Graph>,
    productions: &BTreeMap<mxx_ir_core::artifact::ProductionId, String>,
    secrets: (&RnsBggPayloadSecret, &RnsBggMaskSecret),
) {
    let protocol = top_protocol(top, input, &graphs, productions, secrets, row);
    mxx_ir_core::lean::protocol::export(&protocol, directory, lean_facts())
        .expect("top refresh claim export");
    eprintln!("exported the top refresh correctness claim to {}", directory.display());
}

/// Where the merged stages read each per-part stage input: per-part `(stage, input)` to the
/// merged `(stage, input)`, and the merged inputs that are families over the parts (member `k`
/// the input of part `k`).
#[derive(Default)]
struct MergedInputs {
    targets: BTreeMap<(String, String), (String, String)>,
    families: std::collections::BTreeSet<(String, String)>,
}

/// The protocol's stages with the per-part stages of each phase (`preprocess`, `encrypt`,
/// `online_eval`) replaced by one stage running every part in a parallel loop
/// ([`mxx_ir_core::merge::merge_instances`], part `(chain, component)` the instance at its
/// position in `parts`), and where the merged stages read the per-part inputs. Every input that
/// differs between the parts is one family input over the parts: instance `k` reading an
/// artifact of the producer's instance `k` reads member `k` of the producer's output family, and
/// a plain input is member `k` of a family input (fed by a family external, or, for a linked
/// input, the producer's output family). Stages outside the merged phases read whole output
/// families, so no stage's interface grows with the number of parts.
fn merged_stages(
    stages: Vec<mxx_ir_core::protocol::ProtocolStage>,
    parts: &[(usize, usize)],
    productions: &BTreeMap<mxx_ir_core::artifact::ProductionId, StageId>,
    links: &BTreeMap<String, (StageId, String)>,
) -> (Vec<mxx_ir_core::protocol::ProtocolStage>, MergedInputs) {
    use mxx_ir_core::protocol::{ArtifactBinding, ArtifactName, ProtocolStage, StageInputName};
    const PHASES: [&str; 3] = ["preprocess", "encrypt", "online_eval"];
    // A per-part stage name: its phase and instance.
    let instance_of = |stage: &str| {
        PHASES.iter().find_map(|phase| {
            let rest = stage.strip_prefix(phase)?.strip_prefix('_')?;
            let (chain, component) = rest.split_once('_')?;
            let part = (chain.parse::<usize>().ok()?, component.parse::<usize>().ok()?);
            Some((*phase, parts.iter().position(|candidate| *candidate == part)?))
        })
    };
    let graphs = stages
        .iter()
        .map(|stage| (stage.id.0.clone(), stage.graph.clone()))
        .collect::<BTreeMap<_, _>>();
    // Phases are merged in order; instance `k` reading output `o` of instance `k` of a merged
    // producer reads member `k` of the producer's family named `merged[..].outputs[k][o]`.
    let mut merged: BTreeMap<&str, (Graph, mxx_ir_core::merge::MergeNames)> = BTreeMap::new();
    // The family inputs of the merged stages, by phase and name.
    let family_inputs = RefCell::new(std::collections::BTreeSet::new());
    for phase in PHASES {
        let instances = parts
            .iter()
            .map(|(chain, component)| graphs[&format!("{phase}_{chain}_{component}")].clone())
            .collect::<Vec<_>>();
        let family_input = |kinds: &[&mxx_ir_core::node::NodeKind]| {
            let mxx_ir_core::node::NodeKind::Input { name, artifact, .. } = kinds[0] else {
                return None;
            };
            let Some(artifact) = artifact else {
                // A plain input: member `k` is part `k`'s input.
                family_inputs.borrow_mut().insert((phase, name.clone()));
                return Some((name.clone(), None));
            };
            let mut families = kinds.iter().enumerate().map(|(index, kind)| {
                let mxx_ir_core::node::NodeKind::Input { artifact: Some(artifact), .. } = kind
                else {
                    return None;
                };
                let (producer_phase, instance) =
                    instance_of(&productions.get(&artifact.production_id)?.0)?;
                (instance == index).then_some(())?;
                merged.get(producer_phase)?.1.outputs[index].get(&artifact.artifact_name).cloned()
            });
            let family = families.next()??;
            for other in families {
                if other? != family {
                    return None;
                }
            }
            family_inputs.borrow_mut().insert((phase, name.clone()));
            Some((
                name.clone(),
                Some(mxx_ir_core::node::ArtifactInput {
                    artifact_name: family,
                    ..artifact.clone()
                }),
            ))
        };
        let interface = mxx_ir_core::merge::MergeInterface {
            family_input: &family_input,
            unpacked: &|_| false,
        };
        let result = mxx_ir_core::merge::merge_instances(phase, &instances, &interface)
            .unwrap_or_else(|error| panic!("merging the {phase} stages: {error}"));
        merged.insert(phase, result);
    }
    // A producer (per-part stage, output) as a merged stage and its output family.
    let producer = |stage: &StageId, output: &str| match instance_of(&stage.0) {
        Some((phase, instance)) => (
            StageId(phase.to_owned()),
            merged[phase].1.outputs[instance]
                .get(output)
                .cloned()
                .unwrap_or_else(|| output.to_owned()),
        ),
        None => (stage.clone(), output.to_owned()),
    };
    let mut inputs = MergedInputs::default();
    for (&phase, (_, names)) in &merged {
        for (instance, (chain, component)) in parts.iter().enumerate() {
            for (original, name) in &names.inputs[instance] {
                inputs.targets.insert(
                    (format!("{phase}_{chain}_{component}"), original.clone()),
                    (phase.to_owned(), name.clone()),
                );
            }
        }
    }
    inputs.families = family_inputs
        .into_inner()
        .into_iter()
        .map(|(phase, name)| (phase.to_owned(), name))
        .collect();
    let merged_stage = |phase: &str| {
        let (graph, names) = &merged[phase];
        let bindings = graph
            .root_scope()
            .nodes()
            .iter()
            .filter_map(|node| {
                let mxx_ir_core::node::NodeKind::Input { name, artifact, .. } = node.kind() else {
                    return None;
                };
                let (stage, output) = match artifact {
                    // A family input reads the producer's output family itself.
                    Some(artifact)
                        if inputs.families.contains(&(phase.to_owned(), name.clone())) =>
                    {
                        (
                            StageId(
                                instance_of(&productions[&artifact.production_id].0)?.0.to_owned(),
                            ),
                            artifact.artifact_name.clone(),
                        )
                    }
                    Some(artifact) => {
                        producer(&productions[&artifact.production_id], &artifact.artifact_name)
                    }
                    None => {
                        // The original name of a plain input, in the first instance having it.
                        let original = names
                            .inputs
                            .iter()
                            .find_map(|instance| {
                                instance
                                    .iter()
                                    .find(|(_, merged)| *merged == name)
                                    .map(|(original, _)| original)
                            })
                            .cloned()
                            .unwrap_or_else(|| name.clone());
                        let (stage, output) = links.get(&original)?;
                        producer(stage, output)
                    }
                };
                Some(ArtifactBinding {
                    consumer_input: StageInputName(name.clone()),
                    producer_stage: stage,
                    producer_output: ArtifactName(output),
                })
            })
            .collect();
        ProtocolStage { id: StageId(phase.to_owned()), graph: graph.clone(), bindings }
    };
    let mut result = Vec::new();
    for stage in stages {
        match instance_of(&stage.id.0) {
            // The merged stage takes the place of its first part.
            Some((phase, 0)) => result.push(merged_stage(phase)),
            Some(_) => {}
            None => {
                let bindings = stage
                    .bindings
                    .iter()
                    .map(|binding| {
                        let (producer_stage, output) =
                            producer(&binding.producer_stage, &binding.producer_output.0);
                        ArtifactBinding {
                            consumer_input: binding.consumer_input.clone(),
                            producer_stage,
                            producer_output: ArtifactName(output),
                        }
                    })
                    .collect();
                result.push(ProtocolStage { bindings, ..stage });
            }
        }
    }
    (result, inputs)
}

/// A family external over the parts: member `k` feeds member `k` of every family input in
/// `inputs` and stands for the external `originals[k]` the per-part input of part `k` read.
/// Parts that read one original external read equal members (`representatives[k]` the first
/// part reading `originals[k]`), and a member standing for an original external that other
/// destinations still read equals it; [`top_protocol`]'s requirement ties both (byte strings
/// excepted: every part may use its own hash keys).
struct PartFamily {
    name: String,
    element: InputValueContract,
    originals: Vec<String>,
    representatives: Vec<usize>,
    /// `(k, original)`: member `k` equals the kept external `original`.
    ties: Vec<(usize, String)>,
}

/// The externals of the merged protocol: each external's destinations in the merged stages
/// (deduplicated), with the per-part inputs merged into family inputs fed by one [`PartFamily`]
/// per pattern of original externals over the parts; an original external whose every
/// destination was such a member is dropped.
fn pack_externals(
    externals: Vec<(InputContractEntry, ProtocolInputBinding)>,
    merged: &MergedInputs,
    parts: usize,
) -> (Vec<(InputContractEntry, ProtocolInputBinding)>, Vec<PartFamily>) {
    let instance = |stage: &str| {
        let mut fields = stage.rsplitn(3, '_');
        let component = fields.next()?.parse::<usize>().ok()?;
        let chain = fields.next()?.parse::<usize>().ok()?;
        Some(2 * chain + component)
    };
    // For every family input, the original external feeding each member.
    let mut feeds = BTreeMap::<(String, String), Vec<Option<usize>>>::new();
    let mut kept = Vec::new();
    for (index, (contract, binding)) in externals.iter().enumerate() {
        let mut destinations = Vec::new();
        for destination in &binding.destinations {
            let ProtocolInputDestination::WorkflowStage { stage, input } = destination else {
                destinations.push(destination.clone());
                continue;
            };
            let Some((phase, name)) = merged.targets.get(&(stage.0.clone(), input.0.clone()))
            else {
                destinations.push(destination.clone());
                continue;
            };
            if merged.families.contains(&(phase.clone(), name.clone())) {
                let member = instance(&stage.0).expect("a per-part stage");
                let slot = &mut feeds
                    .entry((phase.clone(), name.clone()))
                    .or_insert_with(|| vec![None; parts])[member];
                assert!(slot.replace(index).is_none(), "{phase}/{name}: one external per member");
            } else {
                let destination = common::to_stage(phase, name);
                if !destinations.contains(&destination) {
                    destinations.push(destination);
                }
            }
        }
        if !destinations.is_empty() {
            kept.push((contract.clone(), ProtocolInputBinding { destinations, ..binding.clone() }));
        }
    }
    let kept_names = kept.iter().map(|(contract, _)| contract.name.clone()).collect::<Vec<_>>();
    // Family inputs fed by one pattern of originals share one family external.
    let mut patterns = BTreeMap::<Vec<usize>, Vec<ProtocolInputDestination>>::new();
    for ((phase, name), members) in feeds {
        let members = members
            .into_iter()
            .map(|member| member.unwrap_or_else(|| panic!("{phase}/{name}: a member is unfed")))
            .collect::<Vec<_>>();
        patterns.entry(members).or_default().push(common::to_stage(&phase, &name));
    }
    let mut families = Vec::new();
    for (members, destinations) in patterns {
        let element = externals[members[0]].0.value.clone();
        assert!(
            members.iter().all(|member| externals[*member].0.value == element),
            "a family external has one element contract"
        );
        let originals =
            members.iter().map(|member| externals[*member].0.name.clone()).collect::<Vec<_>>();
        let representatives = (0..parts)
            .map(|k| members.iter().position(|member| *member == members[k]).expect("member"))
            .collect::<Vec<_>>();
        let ties = (0..parts)
            .filter(|k| representatives[*k] == *k && kept_names.contains(&originals[*k]))
            .map(|k| (k, originals[k].clone()))
            .collect();
        let name = format!("parts_{}", originals[0]);
        kept.push(common::external(
            &name,
            InputValueContract::Family {
                count: IntExpr::constant(parts),
                element: Box::new(element.clone()),
            },
            destinations,
        ));
        families.push(PartFamily { name, element, originals, representatives, ties });
    }
    (kept, families)
}

/// One PRF child of a part: its hash namespace, its PRF, its offset RHS (bound to one), and the
/// external name of its selector families.
struct Child<'a> {
    namespace: [u8; 32],
    prf: &'a SparseLwrPrf,
    offsets: &'a [RhsId],
    selectors: String,
}

/// The executed refresh to the top level as a closed protocol: payload and mask sampling, every
/// part's preprocessing, encryption and online evaluation on its chain, and the decode stage,
/// which counts the coefficients of `y_j - (sigma A'_(j) - Delta_0 r_j x G^st_0)` that decode to
/// one (an error of at least `P / 4`, `P` the product of the refreshed primes); the ideal count
/// is zero. External inputs are the hash
/// keys, the source input `X^b` and the expected `X^(1 - b)`, the label `a` and its monomials,
/// the selector bits and the coordinate tables (the cell coordinates of the key layouts, one per
/// PBC bucket, which every preprocessing stage receives and its online evaluation reads from it)
/// of every PRF (the mask keys shared by every part, the fresh-noise keys of component `j` by
/// every chain, each decoder's own), the unit offsets, and the mask polynomial `sigma_0`. The
/// requirement ties the monomials to `a`, makes every selector row one-hot, the expected exponent
/// the LUT of the input, `sigma_0` the bits of the mask keys on `a` read through the mask tables,
/// and `a` the invocation's label.
///
/// The phases of the parts are merged into one stage each ([`merged_stages`]), and every
/// per-part input is a member of a family external over the parts ([`pack_externals`], e.g. one
/// family of each bucket's smudge selector rows over the parts); the requirement checks these
/// families in loops over the parts, ties the members of parts that read one key set (the mask
/// keys, a component's fresh keys, the label and the source input) and reads the mask keys of
/// part 0. The decode stage reads the family of the parts' residue terms. The stages and the
/// claim thus do not grow with the number of parts or columns.
fn top_protocol(
    top: &RefreshDeclaration,
    input: WireId,
    graphs: &BTreeMap<String, Graph>,
    productions: &BTreeMap<mxx_ir_core::artifact::ProductionId, String>,
    (payload, mask): (&RnsBggPayloadSecret, &RnsBggMaskSecret),
    row: Option<&ParameterRow>,
) -> ProtocolDecl {
    let id = |name: &str| StageId(name.to_owned());
    let productions = productions
        .iter()
        .map(|(production, stage)| (production.clone(), id(stage)))
        .collect::<BTreeMap<_, _>>();
    let secret = |stage: &str| (id(stage), "secret".to_owned());
    let mut links = BTreeMap::from([
        (payload.input_name().expect("sampled payload").to_owned(), secret("payload")),
        (mask.input_name().expect("sampled mask").to_owned(), secret("mask")),
    ]);
    let parts = (0..top.chains()).flat_map(|chain| [(chain, 0), (chain, 1)]).collect::<Vec<_>>();
    // The decode stage reads the family of the parts' residue terms: the merged online
    // evaluation's output family `refresh` (part `(chain, j)` at `2 chain + j`).
    links.insert(REFRESH_TERMS_INPUT.to_owned(), (id("online_eval_0_0"), "refresh".to_owned()));
    // Online evaluation reads each child data family as a resident input named after the
    // preprocessing output holding it.
    for &(chain, component) in &parts {
        let preprocess = id(&format!("preprocess_{chain}_{component}"));
        for name in
            common::inputs_ending_with(&graphs[&format!("online_eval_{chain}_{component}")], "")
        {
            if name.contains("data/DataId(") {
                links.insert(name.clone(), (preprocess.clone(), name));
            }
        }
    }
    let mut names = vec!["payload".to_owned(), "mask".to_owned()];
    for &(chain, component) in &parts {
        for stage in ["preprocess", "encrypt", "online_eval"] {
            names.push(format!("{stage}_{chain}_{component}"));
        }
    }
    names.push("decode".to_owned());

    let range = |lower: i64, upper: i64| InputValueContract::IntegerRange {
        lower: lower.into(),
        upper: upper.into(),
    };
    let family_range = |count: usize, lower: i64, upper: i64| InputValueContract::Family {
        count: IntExpr::constant(count),
        element: Box::new(range(lower, upper)),
    };
    let requirement_input =
        |input: &str| ProtocolInputDestination::Requirement { requirement: 0, input: input.into() };
    let mut externals = Vec::new();

    // Hash keys: one external per derived key name.
    let mut hash_keys = BTreeMap::<String, Vec<ProtocolInputDestination>>::new();
    for name in &names {
        for graph_input in common::inputs_ending_with(&graphs[name], "") {
            if graph_input.starts_with("rns-bgg/hash-key") {
                hash_keys
                    .entry(graph_input.clone())
                    .or_default()
                    .push(common::to_stage(name, &graph_input));
            }
        }
    }
    for (index, destinations) in hash_keys.into_values().enumerate() {
        externals.push(common::external(
            &format!("hash_key_{index}"),
            InputValueContract::Bytes { length: 32.into() },
            destinations,
        ));
    }

    // The source input `X^b` of every chain and the expected `X^(1 - b)` of the decode stage.
    let source_input = |field: &str| {
        parts
            .iter()
            .map(|&(chain, component)| {
                let prefix = plaintext_input_prefix(Some(&top.chain_namespace(chain)), "input");
                common::to_stage(
                    &format!("encrypt_{chain}_{component}"),
                    &format!("{prefix}/input/{input:?}.{field}"),
                )
            })
            .collect::<Vec<_>>()
    };
    externals.push(common::external("x_coefficient", range(1, 1), source_input("coefficient")));
    let mut destinations = source_input("exponent");
    destinations.push(requirement_input("x_exponent"));
    destinations.push(ProtocolInputDestination::Ideal { input: "x_exponent".into() });
    externals.push(common::external("x_exponent", range(0, 1), destinations));
    let n = top.reference().ring_dimension as i64;
    externals.push(common::external(
        "expected_coefficient",
        range(1, 1),
        vec![common::to_stage("decode", "top-refresh-expected.coefficient")],
    ));
    externals.push(common::external(
        "expected_exponent",
        range(0, n - 1),
        vec![
            common::to_stage("decode", "top-refresh-expected.exponent"),
            requirement_input("expected_exponent"),
        ],
    ));
    externals.push(common::external(
        "sigma",
        family_range(n as usize, 0, 1),
        vec![
            common::to_stage("decode", "top-refresh-sigma.coefficients"),
            requirement_input("sigma"),
        ],
    ));

    // The PRF children of every part: inputs (the state `X^0` and the label monomials) and RHS.
    let profile = *top.profile();
    let q = profile.q().expect("PRF modulus");
    let members = profile.lwr_dimension + 1;
    let mut state_coefficient = Vec::new();
    let mut state_exponent = Vec::new();
    let mut label_coefficient = Vec::new();
    let mut label_exponent = Vec::new();
    let mut selectors = BTreeMap::<String, (usize, usize, Vec<ProtocolInputDestination>)>::new();
    let mut tables = BTreeMap::<String, (usize, Vec<ProtocolInputDestination>)>::new();
    let mut offset_coefficient = Vec::new();
    let mut zero_exponents = BTreeMap::<usize, Vec<ProtocolInputDestination>>::new();
    for &(chain, component) in &parts {
        let part = (chain, component);
        let (mask, fresh) = (top.mask(chain).expect("mask"), top.fresh(part).expect("fresh"));
        let decoder = top.decoder(part).expect("decoder");
        let children = [
            Child {
                namespace: top.mask_namespace(part).expect("mask namespace"),
                prf: mask.sampler().prf(),
                offsets: mask.sampler().offset_rhs(),
                selectors: "mask".to_owned(),
            },
            Child {
                namespace: top.fresh_namespace(part).expect("fresh namespace"),
                prf: fresh.sampler().prf(),
                offsets: fresh.sampler().offset_rhs(),
                selectors: format!("fresh_{component}"),
            },
            Child {
                namespace: *decoder.commitment(),
                prf: decoder.smudge().prf(),
                offsets: &[],
                selectors: format!("smudge_{chain}_{component}"),
            },
        ];
        let encrypt = format!("encrypt_{chain}_{component}");
        let preprocess = format!("preprocess_{chain}_{component}");
        for child in &children {
            let prefix = plaintext_input_prefix(Some(&child.namespace), "input");
            let wire = |wire: WireId, field: &str| {
                common::to_stage(&encrypt, &format!("{prefix}/input/{wire:?}.{field}"))
            };
            state_coefficient.push(wire(child.prf.initial_input(), "coefficient"));
            state_exponent.push(wire(child.prf.initial_input(), "exponent"));
            label_coefficient.push(wire(child.prf.input_vector(), "coefficient"));
            label_exponent.push(wire(child.prf.input_vector(), "exponent"));
            let prefix = plaintext_input_prefix(Some(&child.namespace), "rhs");
            let rhs = |rhs: RhsId, field: &str| {
                common::to_stage(&preprocess, &format!("{prefix}/rhs/{rhs:?}.{field}"))
            };
            for (bucket, family) in child.prf.selector_families().iter().enumerate() {
                let row_length = family.row_length.unwrap_or(family.members);
                let entry = selectors
                    .entry(format!("{}_selector_{bucket}", child.selectors))
                    .or_insert_with(|| (family.members, row_length, Vec::new()));
                assert_eq!((entry.0, entry.1), (family.members, row_length));
                entry.2.push(rhs(family.rhs, "coefficient"));
                zero_exponents.entry(family.members).or_default().push(rhs(family.rhs, "exponent"));
            }
            for offset in child.offsets {
                offset_coefficient.push(rhs(*offset, "coefficient"));
                zero_exponents.entry(0).or_default().push(rhs(*offset, "exponent"));
            }
            // Bucket `b`'s coordinate table: the cell coordinates of every key's layout, a row of
            // `W` entries per key, as long as the bucket's selector family.
            let families = child.prf.selector_families();
            for (bucket, table) in child.prf.tables().iter().enumerate() {
                let entry = tables
                    .entry(format!("{}_table_{bucket}", child.selectors))
                    .or_insert_with(|| (families[bucket].members, Vec::new()));
                assert_eq!(entry.0, families[bucket].members);
                entry.1.push(common::to_stage(
                    &preprocess,
                    &data_input_name(Some(&child.namespace), *table),
                ));
            }
        }
    }
    externals.push(common::external("state_coefficient", range(1, 1), state_coefficient));
    externals.push(common::external("state_exponent", range(0, 0), state_exponent));
    label_coefficient.push(requirement_input("a_coefficient"));
    label_exponent.push(requirement_input("a_exponent"));
    externals.push(common::external(
        "a_coefficient",
        family_range(members, -1, 1),
        label_coefficient,
    ));
    externals.push(common::external("a_exponent", family_range(members, 0, n - 1), label_exponent));
    externals.push(common::external(
        "a",
        family_range(members, 0, q as i64 - 1),
        vec![requirement_input("a")],
    ));
    for (name, (count, _, destinations)) in &selectors {
        externals.push(common::external(name, family_range(*count, 0, 1), destinations.clone()));
    }
    if !offset_coefficient.is_empty() {
        externals.push(common::external("offset_coefficient", range(1, 1), offset_coefficient));
    }
    // Every table entry is a member of the label family (`nu` the zero member).
    for (name, (count, destinations)) in &tables {
        externals.push(common::external(
            name,
            family_range(*count, 0, profile.lwr_dimension as i64),
            destinations.clone(),
        ));
    }
    // Every RHS exponent is zero: one external per family size (`0` for a scalar).
    for (count, destinations) in zero_exponents {
        let contract = if count == 0 { range(0, 0) } else { family_range(count, 0, 0) };
        externals.push(common::external(&format!("rhs_exponent_{count}"), contract, destinations));
    }

    // Every per-part input is a member of a family input of the merged stages, fed by a family
    // external over the parts.
    let stages = names
        .iter()
        .map(|name| common::stage(name, &graphs[name], &productions, &links))
        .collect::<Vec<_>>();
    let (stages, merged) = merged_stages(stages, &parts, &productions, &links);
    let (mut externals, families) = pack_externals(externals, &merged, parts.len());

    // The requirement.
    let family = |count: usize| FamilyType {
        element: IntType,
        count: IntExpr::constant(count),
        member_slot: None,
    };
    let requirement = DslContext::new("refresh-top-inputs");
    let a = requirement.input::<Family<Int>>("a", family(members)).unwrap();
    let a_coefficient = requirement.input::<Family<Int>>("a_coefficient", family(members)).unwrap();
    let a_exponent = requirement.input::<Family<Int>>("a_exponent", family(members)).unwrap();
    let stride = profile.layout.stride() as i64;
    // The label is public and fixed by the invocation (the online stage reads its public payload
    // as constants); the last member is the zero member.
    let mut label = top.label(*HASH_KEY, LABEL_TAG).expect("PRF label");
    label.push(0);
    let mut valid = mxx_dsl::iterate(members, Int::constant(0).equal(0), |member, valid| {
        let fixed = mxx_dsl::select(
            member.clone(),
            label.iter().map(|value| Int::constant(*value as i64)).collect(),
        )?;
        let sign = a_coefficient.at(member.clone());
        // `2 e + n (1 - c) = 2 stride a` with `c = ±1`.
        Ok(valid &
            a.at(member.clone()).equal(fixed) &
            sign.clone().mul(sign.clone()).equal(1) &
            a_exponent
                .at(member.clone())
                .mul(2)
                .add(Int::constant(1).sub(sign).mul(n))
                .equal(a.at(member).mul(2 * stride)))
    })
    .unwrap();
    let x_exponent = requirement.input::<Int>("x_exponent", IntType).unwrap();
    let expected_exponent = requirement.input::<Int>("expected_exponent", IntType).unwrap();
    valid = valid & expected_exponent.equal(Int::constant(1).sub(x_exponent.clone()));
    // The family externals over the parts: parts reading one original external read equal
    // members, a member standing for a kept external equals it, and every selector row is
    // one-hot. A pinned element (one admissible value) needs no tie, and byte strings (hash
    // keys) are not tied. Integer families are read as families of members.
    let pinned = |contract: &InputValueContract| match contract {
        InputValueContract::IntegerRange { lower, upper } => lower == upper,
        InputValueContract::Family { element, .. } => matches!(
            element.as_ref(),
            InputValueContract::IntegerRange { lower, upper } if lower == upper
        ),
        _ => false,
    };
    let scalars = BTreeMap::from([("x_exponent", x_exponent)]);
    let label_families = BTreeMap::from([
        ("a_coefficient", a_coefficient.clone()),
        ("a_exponent", a_exponent.clone()),
    ]);
    let parts_count = parts.len();
    let mut part_families = BTreeMap::new();
    for part_family in &families {
        let selector = selectors.get(&part_family.originals[0]).map(|(_, row, _)| *row);
        let tied = part_family.representatives.iter().enumerate().any(|(k, r)| *r != k) ||
            !part_family.ties.is_empty();
        let read = part_family.name.starts_with("parts_mask_");
        let bytes = matches!(part_family.element, InputValueContract::Bytes { .. });
        if bytes || pinned(&part_family.element) || (!tied && selector.is_none() && !read) {
            continue;
        }
        let representative = |member: Int| {
            mxx_dsl::select(
                member,
                part_family.representatives.iter().map(|r| Int::constant(*r as i64)).collect(),
            )
        };
        match &part_family.element {
            InputValueContract::IntegerRange { .. } => {
                let input = requirement
                    .input::<Family<Int>>(part_family.name.clone(), family(parts_count))
                    .unwrap();
                valid = mxx_dsl::iterate(parts_count, valid, |k, valid| {
                    Ok(valid & input.at(k.clone()).equal(input.at(representative(k)?)))
                })
                .unwrap();
                for (k, original) in &part_family.ties {
                    valid = valid &
                        input
                            .at(Int::constant(*k as i64))
                            .equal(scalars[original.as_str()].clone());
                }
            }
            InputValueContract::Family { count, .. } => {
                let count = count.evaluate(&mxx_ir_core::ParamEnv::default()).unwrap();
                let count = count.to_usize().expect("a family count");
                let input = requirement
                    .input::<Family<Family<Int>>>(
                        part_family.name.clone(),
                        FamilyType {
                            element: family(count),
                            count: IntExpr::constant(parts_count),
                            member_slot: None,
                        },
                    )
                    .unwrap();
                if tied {
                    valid = mxx_dsl::iterate(parts_count, valid, |k, valid| {
                        let member = input.at(k.clone());
                        let other = input.at(representative(k)?);
                        mxx_dsl::iterate(count, valid, |entry, valid| {
                            Ok(valid & member.at(entry.clone()).equal(other.at(entry)))
                        })
                    })
                    .unwrap();
                }
                for (k, original) in &part_family.ties {
                    let member = input.at(Int::constant(*k as i64));
                    let original = &label_families[original.as_str()];
                    valid = mxx_dsl::iterate(count, valid, |entry, valid| {
                        Ok(valid & member.at(entry.clone()).equal(original.at(entry)))
                    })
                    .unwrap();
                }
                if let Some(row_length) = selector {
                    valid = mxx_dsl::iterate(parts_count, valid, |k, valid| {
                        let member = input.at(k);
                        mxx_dsl::iterate(count / row_length, valid, |row, valid| {
                            let total =
                                mxx_dsl::iterate(row_length, Int::constant(0), |slot, total| {
                                    Ok(total.add(
                                        member.at(row.clone().mul(row_length as i64).add(slot)),
                                    ))
                                })?;
                            Ok(valid & total.equal(1))
                        })
                    })
                    .unwrap();
                }
                part_families.insert(part_family.originals[0].clone(), input);
            }
            other => panic!("{}: unexpected family element {other:?}", part_family.name),
        }
        let binding = &mut externals
            .iter_mut()
            .find(|(contract, _)| contract.name == part_family.name)
            .expect("the family external")
            .1;
        binding.destinations.push(requirement_input(&part_family.name));
    }
    // `sigma_k` is the bit of mask member `k`: one exactly when `z_k mod Q < Q / 2` for
    // `z_k = sum_b a[coordinate of the selected cell of bucket b]`; slot `f` of bucket `b`'s
    // selector row `k` selects cell `f` of member `k`, whose coordinate is entry `k W + f` of
    // bucket `b`'s mask table (the zero member `v` for a dummy or padding cell). Every part
    // reads the same mask keys (its members are tied), so part 0's stand for all.
    let sigma = requirement.input::<Family<Int>>("sigma", family(n as usize)).unwrap();
    let mask_prf = top.mask(0).expect("mask child").sampler().prf();
    let mask_buckets = (0..mask_prf.tables().len())
        .map(|bucket| {
            let width = selectors[&format!("mask_selector_{bucket}")].1 as i64;
            let first = Int::constant(0);
            (
                width,
                part_families[&format!("mask_selector_{bucket}")].at(first.clone()),
                part_families[&format!("mask_table_{bucket}")].at(first),
            )
        })
        .collect::<Vec<_>>();
    valid = mxx_dsl::iterate(n as usize, valid, |member, valid| {
        let mut z = Int::constant(0);
        for (width, selector, table) in &mask_buckets {
            z = z.add(mxx_dsl::iterate(*width as usize, Int::constant(0), |slot, total| {
                let cell = member.clone().mul(*width).add(slot);
                Ok(total.add(selector.at(cell.clone()).mul(a.at(table.at(cell)))))
            })?);
        }
        let bit = Int::constant(1).sub(z.rem(q as i64).div(q as i64 / 2));
        Ok(valid & sigma.at(member).equal(bit))
    })
    .unwrap();
    let requirement =
        PurePredicateSpec::new(requirement.output("valid", valid).unwrap().build().unwrap().graph)
            .unwrap();
    // The ideal count of failed coefficients is zero (an `Int` read from the input, as the
    // endpoint compares two runtime integers).
    let ideal = DslContext::new("refresh-top-ideal");
    let x = ideal.input::<Int>("x_exponent", IntType).unwrap();
    let ideal = ideal.output("value", x.mul(0)).unwrap().build();
    let ideal = IdealSpec::new(ideal.unwrap().graph).unwrap();

    let (contracts, bindings): (Vec<_>, Vec<_>) = externals.into_iter().unzip();
    let endpoint = EndpointSpecId::Exact;
    let failure_probability_log2 = claim_failure_exponent(row);
    ProtocolDecl::new(ProtocolDecl {
        params: Vec::new(),
        bindings: Default::default(),
        failure_probability_log2,
        bundle: ClosedProtocolBundle {
            workflow: Workflow { stages, entrypoint: id("decode") },
            ideal,
            requirements: vec![requirement],
            comparator: ComparatorSpec::Equality {
                endpoints: vec![ComparatorEndpointBinding {
                    endpoint,
                    actual_input: "decoded-residual".to_owned(),
                    ideal_input: "value".to_owned(),
                    result_output: "failure".to_owned(),
                    failure_value: true,
                }],
            },
            endpoints: EndpointBindings {
                entries: vec![EndpointBinding {
                    spec: endpoint,
                    semantics: EndpointSemanticBinding::Exact,
                    workflow_output: OutputRef {
                        stage: id("decode"),
                        output: "decoded-residual".to_owned(),
                    },
                    ideal_output: "value".to_owned(),
                }],
            },
            operational_decoder_targets: Vec::new(),
            endpoint_specs: vec![endpoint],
            input_contract: InputContract { inputs: contracts },
            input_bindings: bindings,
            precondition_spec: ProtocolPreconditionSpec {
                requirement_outputs: vec!["valid".to_owned()],
            },
        },
    })
    .expect("valid top refresh protocol")
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
    let recorded = recorded_parameters();
    let row = recorded.as_ref().map(|(_, row)| row);
    if simulate_only() {
        // The accepted digit count, its simulated noise and the security gate only.
        setup(row);
        return;
    }
    if export_only() {
        let setup = setup(row);
        let levels = setup.reference_chain.levels().to_vec();
        let directory = tempfile::tempdir().expect("artifact directory");
        let store = api::StageStorage::File(directory.path().into()).open().expect("stage storage");
        let mut execution =
            GraphOnly(CpuExecution::new(mxx_backends::backend::poly::cpu_backend(levels)));
        round_trip(setup, &mut execution, store, recorded);
        return;
    }
    let devices = detected_gpu_device_ids();
    if devices.is_empty() {
        eprintln!("skipping the top refresh round trip: no CUDA device detected");
        return;
    }
    let setup = setup(row);
    let key =
        |level: &DCRTPolyParams| (level.ring_dimension(), level.to_crt().0, level.base_bits());
    let mut levels =
        setup.reference_chain.distinct_levels().into_iter().map(key).collect::<Vec<_>>();
    for (chain, chains) in setup.all_chains.iter().enumerate() {
        for parameters in [&chains.source, &chains.mask, &chains.fresh, &chains.smudge] {
            levels.extend(parameters.distinct_levels().into_iter().map(key));
        }
        let prime = setup.top.output_prime(chain).expect("output prime");
        levels.push(key(&chains.pool.select_modulus(&BigUint::from(prime)).expect("output ring")));
    }
    // The parts' residue terms live in the ring of the product of the refreshed primes.
    let product = setup.top.output_primes().into_iter().map(BigUint::from).product::<BigUint>();
    levels.push(key(&setup.all_chains[0].pool.select_modulus(&product).expect("terms ring")));
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
    let store = api::StageStorage::File(directory.path().into()).open().expect("stage storage");
    round_trip(setup, &mut runtime, store, recorded);
}
