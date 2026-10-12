#![cfg(feature = "gpu")]

//! The refresh to the top level on the GPU runtime.
//!
//! The refresh ([`mxx_exponent_lut::refresh::top`]) of an exponent LUT output `x = X^(1 - b)` on
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
//! The reference has `PRIMES` 30-bit primes (ring dimension 8, base `2^15`) at level 0 and one
//! fewer at level 1; the source chains have `[PRIMES, PRIMES - 1, PRIMES - 2]` primes, so the
//! decoding level keeps the output prime and `PRIMES - 2` margin primes. The digit count is the
//! smallest whose simulated noise of every part fits its decoding margin and hides by
//! `STATISTICAL_BITS`. It runs the Static RHS lifecycle only.
//!
//! | Variable (`MXX_EXPONENT_LUT_REFRESH_TOP_` prefix) | Default | Meaning |
//! | --- | --- | --- |
//! | `PRIMES` | 4 | Primes of the reference's level 0 (`>= 3`) |
//! | `SLOTS` | 1 | Refreshed CRT slots, the first primes of level 0 (`1..=PRIMES`) |
//! | `ERROR_SIGMA`, `TRAPDOOR_SIGMA` | 0, 4.578 | Encoding error and trapdoor Gaussian widths |
//! | `STATISTICAL_BITS` | 1 | Hiding margin of the smudge over each decoder's noise |
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
    program::{
        AttributeKnowledge, ExponentLutProgram, ExponentLutProgramBuilder, InputBinding,
        LutOutputRepresentation, LutTable, Operand, RhsId, WireId,
    },
    program_gadgets::{
        pbc::{
            PbcCell, PbcGeneratedKeyLayout, PbcParameters, PbcRootSeed, clear_pbc_inner_product,
            generate_key_layout,
        },
        prf::{SparseLwrPrf, SparseLwrPrfProfile},
        prf_sampler::PrfSampler,
    },
    refresh::{
        TopReference,
        top::{TopRefreshDeclaration, TopRefreshLayouts},
    },
    rns_bgg::{
        RnsBggMaskSecret, RnsBggNoiseBindings, RnsBggNoiseValue, RnsBggParameters,
        RnsBggPayloadSecret, RnsBggPreprocessEncryptDescriptor, RnsEncryptBuild,
        RnsOnlineEvalBuild, RnsPreprocessBuild, decode_top_refresh, plaintext_input_prefix,
        refresh_fresh_namespace, refresh_mask_namespace, simulate_refresh_noise, top_refresh_error,
    },
    utils::env_usize,
};
use mxx_ir_core::{
    Graph, IntExpr, ParamEnv, ValidatedGraph,
    protocol::{
        ClosedProtocolBundle, ComparatorEndpointBinding, ComparatorSpec, EndpointBinding,
        EndpointBindings, EndpointSemanticBinding, EndpointSpecId, InputContract,
        InputValueContract, OutputRef, ProtocolDecl, ProtocolInputDestination,
        ProtocolPreconditionSpec, PurePredicateSpec, StageId, Workflow,
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

/// `count` weight-one PRF keys over the LWR coordinates, in two buckets, drawn from `rng`.
fn keys(rng: &mut StdRng, count: usize) -> Vec<PbcGeneratedKeyLayout> {
    let parameters = PbcParameters::custom(LWR_DIMENSION, 1, 2, 2, 64, None);
    (0..count)
        .map(|_| {
            let support = vec![rng.random::<u32>() as usize % LWR_DIMENSION];
            (0..64)
                .find_map(|_| {
                    generate_key_layout(&parameters, PbcRootSeed(rng.random()), &support).ok()
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
    rng: &mut StdRng,
    source: &ExponentLutProgram,
    output: WireId,
    reference: &TopReference,
    slots: usize,
    extension: &[u64],
    digits: usize,
) -> (TopRefreshDeclaration, Keys) {
    let n = 8;
    let columns = reference.columns();
    let primes = reference.top.len();
    let mask = keys(rng, n);
    let fresh = [keys(rng, columns * n), keys(rng, columns * n)];
    let smudge = RefCell::new(BTreeMap::new());
    let rng = RefCell::new(rng);
    let mut sample = |chain: usize, component: usize, column: usize| {
        let generated = keys(&mut rng.borrow_mut(), n * digits);
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
        (0..slots).collect(),
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
    runtime: &mut impl api::StageExecution<api::StageStore>,
    store: &mut api::StageStore,
    graphs: &StageGraphs,
) -> (api::StageOutputs, [mxx_ir_core::artifact::ProductionId; 2]) {
    let part = top.part(chain, component).expect("part");
    let name = |stage: &str| format!("{stage}_{chain}_{component}");
    let details = part.top().expect("top part");
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
        record(graphs, name("preprocess")),
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
        Some(&preprocess),
        &chain_payload,
        Some(&mask.at_level(&chains.source, 0).expect("chain mask")),
        &hash_key,
        RnsEncryptBuild::new(&chains.source, context).with_hash_namespace(chain_namespace),
        runtime,
        store,
        |build| build.bind_top_refresh(part, mask_encrypt, fresh_encrypt, smudge_encrypt),
        record(graphs, name("encrypt")),
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
            build.apply_top_refresh(0, part, mask_online, fresh_online, smudge_online)?;
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
fn export_only() -> bool {
    env::var(format!("{PREFIX}EXPORT_ONLY")).is_ok_and(|value| value == "1")
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
    top: TopRefreshDeclaration,
    keys: Keys,
    reference_chain: RnsBggParameters,
    all_chains: Vec<Chains>,
    sigma: BigDecimal,
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
    let seed = row_setting(row, "key_seed", "KEY_SEED", 0) as u64;
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
    let rng = |seed: u64| <StdRng as rand::SeedableRng>::seed_from_u64(seed);
    // A recorded set fixes its digit count; otherwise the smallest one whose simulated noise
    // fits and hides.
    let digits = match row {
        Some(row) => common::column(row, "digits"),
        None => (1..40)
            .find(|digits| {
                let (top, _) = top_refresh(
                    &mut rng(seed),
                    &source,
                    output,
                    &reference,
                    slots,
                    &extension,
                    *digits,
                );
                accepted(&top, &source, &sigma, trapdoor, bits)
            })
            .expect("a fitting and hiding digit count"),
    };
    let (top, keys) =
        top_refresh(&mut rng(seed), &source, output, &reference, slots, &extension, digits);
    assert!(
        row.is_none() || accepted(&top, &source, &sigma, trapdoor, bits),
        "the recorded digit count fits and hides"
    );
    let (_, reference_chain) =
        RnsBggParameters::top_refresh_chain(&top, None, sigma.clone(), 1).expect("reference chain");
    let all_chains =
        (0..top.chains()).map(|chain| chains(&top, &source, chain, &sigma)).collect::<Vec<_>>();
    eprintln!(
        "top refresh accepted: chains={} columns={} digits={digits}",
        top.chains(),
        reference.columns()
    );
    Setup { source, input, top, keys, reference_chain, all_chains, sigma, trapdoor_sigma }
}

/// Every part's stages on `runtime`, the decode stage, and, for a recorded set, the export of
/// the claim of the recorded graphs.
fn round_trip(
    setup: Setup,
    runtime: &mut impl api::StageExecution<api::StageStore>,
    mut store: api::StageStore,
    recorded: Option<(String, ParameterRow)>,
) {
    let Setup { source, input, top, keys, reference_chain, all_chains, trapdoor_sigma, .. } = setup;
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
                &keys,
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
    if export_only() {
        let decoded = decode_top_refresh(
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
        let (id, row) = recorded.expect("export-only mode exports a recorded parameter set");
        export_claim(&id, &row, &top, input, graphs.into_inner(), &productions, (&payload, &mask));
        return;
    }
    let errors = top_refresh_error(
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
    let decoded = decode_top_refresh(
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
    if let Some((id, row)) = recorded {
        export_claim(&id, &row, &top, input, graphs.into_inner(), &productions, (&payload, &mask));
    }
}

/// Export the claim of the recorded stage graphs to `lean/refresh_top/<id>/generated`.
fn export_claim(
    id: &str,
    row: &ParameterRow,
    top: &TopRefreshDeclaration,
    input: WireId,
    graphs: BTreeMap<String, Graph>,
    productions: &BTreeMap<mxx_ir_core::artifact::ProductionId, String>,
    secrets: (&RnsBggPayloadSecret, &RnsBggMaskSecret),
) {
    let directory = common::generated_directory("refresh_top", id);
    let protocol = top_protocol(top, input, &graphs, productions, secrets, row);
    mxx_ir_core::lean::protocol::export(&protocol, &directory, lean_facts())
        .expect("top refresh claim export");
    eprintln!("exported the top refresh correctness claim to {}", directory.display());
}

/// The protocol's stages with the per-part stages of each phase (`preprocess`, `encrypt`,
/// `online_eval`) replaced by one stage running every part in a parallel loop
/// ([`mxx_ir_core::merge::merge_instances`], part `(chain, component)` the instance at its
/// position in `parts`): input destinations, artifact bindings and links name the merged stages
/// and the instances' merged input and output names. Stages pass families between merged phases:
/// instance `k` reading an artifact of the producer's instance `k` reads member `k` of the
/// producer's output family; only outputs read outside the merged stages (the residues) are
/// unpacked per instance.
fn merged_stages(
    stages: Vec<mxx_ir_core::protocol::ProtocolStage>,
    bindings: &mut [mxx_ir_core::protocol::ProtocolInputBinding],
    parts: &[(usize, usize)],
    productions: &BTreeMap<mxx_ir_core::artifact::ProductionId, StageId>,
    links: &BTreeMap<String, (StageId, String)>,
) -> Vec<mxx_ir_core::protocol::ProtocolStage> {
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
    // Outputs of per-part stages read outside the merged stages, by links or bindings.
    let unpacked = links
        .values()
        .map(|(stage, output)| (stage.clone(), output.clone()))
        .chain(stages.iter().filter(|stage| instance_of(&stage.id.0).is_none()).flat_map(|stage| {
            stage
                .bindings
                .iter()
                .map(|binding| (binding.producer_stage.clone(), binding.producer_output.0.clone()))
        }))
        .filter_map(|(stage, output)| Some((instance_of(&stage.0)?.0, output)))
        .collect::<std::collections::BTreeSet<_>>();
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
            let mut families = kinds.iter().enumerate().map(|(index, kind)| {
                let mxx_ir_core::node::NodeKind::Input { artifact: Some(artifact), .. } = kind
                else {
                    return None;
                };
                let (producer_phase, instance) =
                    instance_of(&productions.get(&artifact.production_id)?.0)?;
                (instance == index).then_some(())?;
                let family = merged.get(producer_phase)?.1.outputs[index]
                    .get(&artifact.artifact_name)?
                    .clone();
                Some((family, artifact))
            });
            let (family, artifact) = families.next()??;
            for other in families {
                if other?.0 != family {
                    return None;
                }
            }
            let mxx_ir_core::node::NodeKind::Input { name, .. } = kinds[0] else {
                return None;
            };
            family_inputs.borrow_mut().insert((phase, name.clone()));
            Some((
                name.clone(),
                Some(mxx_ir_core::node::ArtifactInput {
                    artifact_name: family,
                    ..(*artifact).clone()
                }),
            ))
        };
        let is_unpacked = |name: &str| unpacked.contains(&(phase, name.to_owned()));
        let interface = mxx_ir_core::merge::MergeInterface {
            family_input: &family_input,
            unpacked: &is_unpacked,
        };
        let result = mxx_ir_core::merge::merge_instances(phase, &instances, &interface)
            .unwrap_or_else(|error| panic!("merging the {phase} stages: {error}"));
        merged.insert(phase, result);
    }
    // A producer (per-part stage, output) as a merged stage and output.
    let producer = |stage: &StageId, output: &str| match instance_of(&stage.0) {
        Some((phase, instance)) => {
            let names = &merged[phase].1;
            let name = names.unpacked[instance]
                .get(output)
                .or_else(|| names.outputs[instance].get(output))
                .cloned()
                .unwrap_or_else(|| output.to_owned());
            (StageId(phase.to_owned()), name)
        }
        None => (stage.clone(), output.to_owned()),
    };
    for binding in bindings.iter_mut() {
        let mut destinations = Vec::new();
        for destination in binding.destinations.drain(..) {
            let destination = match destination {
                ProtocolInputDestination::WorkflowStage { stage, input } => {
                    match instance_of(&stage.0) {
                        Some((phase, instance)) => ProtocolInputDestination::WorkflowStage {
                            stage: StageId(phase.to_owned()),
                            input: StageInputName(
                                merged[phase].1.inputs[instance]
                                    .get(&input.0)
                                    .cloned()
                                    .unwrap_or(input.0),
                            ),
                        },
                        None => ProtocolInputDestination::WorkflowStage { stage, input },
                    }
                }
                other => other,
            };
            if !destinations.contains(&destination) {
                destinations.push(destination);
            }
        }
        binding.destinations = destinations;
    }
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
                    Some(artifact) if family_inputs.borrow().contains(&(phase, name.clone())) => (
                        StageId(instance_of(&productions[&artifact.production_id].0)?.0.to_owned()),
                        artifact.artifact_name.clone(),
                    ),
                    Some(artifact) => {
                        producer(&productions[&artifact.production_id], &artifact.artifact_name)
                    }
                    None => {
                        // The original name of a plain input, in any instance.
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
    result
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
/// the selector bits of every PRF (the mask keys shared by every part, the fresh-noise keys of
/// component `j` by every chain, each decoder's own), the unit offsets, and the mask polynomial
/// `sigma_0`. The requirement ties the monomials to `a`, makes every selector row one-hot, the
/// expected exponent the LUT of the input, `sigma_0` the bits of the mask keys on `a`, and `a` the
/// invocation's label.
fn top_protocol(
    top: &TopRefreshDeclaration,
    input: WireId,
    graphs: &BTreeMap<String, Graph>,
    productions: &BTreeMap<mxx_ir_core::artifact::ProductionId, String>,
    (payload, mask): (&RnsBggPayloadSecret, &RnsBggMaskSecret),
    row: &ParameterRow,
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
    for &(chain, component) in &parts {
        links.insert(
            format!("residue-{chain}-{component}"),
            (id(&format!("online_eval_{chain}_{component}")), "refresh".to_owned()),
        );
    }
    let mut names = vec!["payload".to_owned(), "mask".to_owned()];
    for &(chain, component) in &parts {
        for stage in ["preprocess", "encrypt", "online_eval"] {
            names.push(format!("{stage}_{chain}_{component}"));
        }
    }
    names.push("decode".to_owned());
    let stages = names
        .iter()
        .map(|name| common::stage(name, &graphs[name], &productions, &links))
        .collect::<Vec<_>>();

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
    let profile = *top.part(0, 0).expect("part").profile();
    let q = profile.q().expect("PRF modulus");
    let members = profile.lwr_dimension + 1;
    let mut state_coefficient = Vec::new();
    let mut state_exponent = Vec::new();
    let mut label_coefficient = Vec::new();
    let mut label_exponent = Vec::new();
    let mut selectors = BTreeMap::<String, (usize, usize, Vec<ProtocolInputDestination>)>::new();
    let mut offset_coefficient = Vec::new();
    let mut zero_exponents = BTreeMap::<usize, Vec<ProtocolInputDestination>>::new();
    for &(chain, component) in &parts {
        let part = top.part(chain, component).expect("part");
        let details = part.top().expect("top part");
        let mut children = vec![
            Child {
                namespace: refresh_mask_namespace(part),
                prf: details.mask().sampler().prf(),
                offsets: details.mask().sampler().offset_rhs(),
                selectors: "mask".to_owned(),
            },
            Child {
                namespace: refresh_fresh_namespace(part),
                prf: part.fresh().sampler().prf(),
                offsets: part.fresh().sampler().offset_rhs(),
                selectors: format!("fresh_{component}"),
            },
        ];
        for (column, decoder) in part.decoders().iter().enumerate() {
            children.push(Child {
                namespace: *decoder.commitment(),
                prf: decoder.smudge().prf(),
                offsets: &[],
                selectors: format!("smudge_{chain}_{component}_{column}"),
            });
        }
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
        let mut destinations = destinations.clone();
        destinations.push(requirement_input(name));
        externals.push(common::external(name, family_range(*count, 0, 1), destinations));
    }
    if !offset_coefficient.is_empty() {
        externals.push(common::external("offset_coefficient", range(1, 1), offset_coefficient));
    }
    // Every RHS exponent is zero: one external per family size (`0` for a scalar).
    for (count, destinations) in zero_exponents {
        let contract = if count == 0 { range(0, 0) } else { family_range(count, 0, 0) };
        externals.push(common::external(&format!("rhs_exponent_{count}"), contract, destinations));
    }

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
    let mut selector_inputs = BTreeMap::new();
    for (name, (count, row_length, _)) in &selectors {
        let selector = requirement.input::<Family<Int>>(name.clone(), family(*count)).unwrap();
        let row_length = *row_length as i64;
        valid = mxx_dsl::iterate(count / row_length as usize, valid, |row, valid| {
            let total = mxx_dsl::iterate(row_length as usize, Int::constant(0), |slot, total| {
                Ok(total.add(selector.at(row.clone().mul(row_length).add(slot))))
            })?;
            Ok(valid & total.equal(1))
        })
        .unwrap();
        selector_inputs.insert(name.clone(), selector);
    }
    let x_exponent = requirement.input::<Int>("x_exponent", IntType).unwrap();
    let expected_exponent = requirement.input::<Int>("expected_exponent", IntType).unwrap();
    valid = valid & expected_exponent.equal(Int::constant(1).sub(x_exponent));
    // `sigma_k` is the bit of mask member `k`: one exactly when `z_k mod Q < Q / 2` for
    // `z_k = sum_b a[coordinate of the selected cell of bucket b]`; slot `f` of bucket `b`'s
    // selector row `k` selects cell `f` of member `k` (the padding coordinate `v` past its cells).
    let sigma = requirement.input::<Family<Int>>("sigma", family(n as usize)).unwrap();
    let mask_prf = top.part(0, 0).expect("part").top().expect("top part").mask().sampler().prf();
    let buckets = mask_prf.layouts()[0].cells.len();
    let tables = (0..buckets)
        .map(|bucket| {
            let width = selectors[&format!("mask_selector_{bucket}")].1;
            let table = mask_prf
                .layouts()
                .iter()
                .flat_map(|layout| {
                    let cells = &layout.cells[bucket];
                    assert!(cells.len() <= width);
                    (0..width).map(move |slot| match cells.get(slot) {
                        Some(PbcCell::Real { coordinate, .. }) => *coordinate as i64,
                        _ => profile.lwr_dimension as i64,
                    })
                })
                .collect::<Vec<_>>();
            (width as i64, table)
        })
        .collect::<Vec<_>>();
    valid = mxx_dsl::iterate(n as usize, valid, |member, valid| {
        let mut z = Int::constant(0);
        for (bucket, (width, table)) in tables.iter().enumerate() {
            let selector = &selector_inputs[&format!("mask_selector_{bucket}")];
            z = z.add(mxx_dsl::iterate(*width as usize, Int::constant(0), |slot, total| {
                let cell = member.clone().mul(*width).add(slot);
                let coordinate = mxx_dsl::select(
                    cell.clone(),
                    table.iter().map(|value| Int::constant(*value)).collect(),
                )?;
                Ok(total.add(selector.at(cell).mul(a.at(coordinate))))
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

    let (contracts, mut bindings): (Vec<_>, Vec<_>) = externals.into_iter().unzip();
    // Every phase of every part is one stage: the parts' graphs merged into one parallel loop.
    let stages = merged_stages(stages, &mut bindings, &parts, &productions, &links);
    let endpoint = EndpointSpecId::Exact;
    let failure_probability_log2 = match row["noise_model"].as_str() {
        "worst" => None,
        _ => Some(common::column(row, "failure_exponent")),
    };
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
    if export_only() {
        assert!(row.is_some(), "export-only mode needs {PREFIX}PARAMETER_ID");
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
