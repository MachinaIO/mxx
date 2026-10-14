#![cfg(feature = "gpu")]

//! The refresh to the top level on the GPU runtime.
//!
//! The refresh ([`mxx_exponent_lut::refresh`]) of an exponent LUT output `x = X^(1 - b)` on
//! the private input `X^b` ([`bit_flip_source`]) re-encrypts it as both rows of a level-0
//! encoding under a new PRF-derived mask `sigma = (sigma_0, 1)`, in the format of a reference
//! chain. It refreshes the first `SLOTS` CRT slots of `Q_0`: [`RnsBggRefresh::run`] evaluates the
//! source Program on one chain per refreshed prime `q` and, on every chain, runs the two parts
//! (one per component) through preprocessing, encryption and online evaluation (see
//! [`mxx_exponent_lut::rns_bgg::RnsBggRefresh`]).
//!
//! The residues of every chain are recombined modulo the product `P` of the refreshed primes, and
//! the round trip checks that `y_j - (sigma A'_(j) - Delta_0 r_j x G^st_0)` modulo `P` is exactly
//! the fresh CBD samples of component `j`'s PRF keys on the label, with `sigma_0` the bits of the
//! mask keys, and that every refreshed coefficient decodes to zero.
//!
//! The reference has `PRIMES` primes of `CRT_BITS` bits (ring dimension `RING_DIMENSION`, base
//! `2^BASE_BITS`) at level 0 and one fewer at level 1; the source chains have
//! `[PRIMES, PRIMES - 1, PRIMES - 2]` primes, so the decoding level keeps the output prime and
//! `PRIMES - 2` margin primes, and the children's chains extend them by `EXTENSION_PRIMES` primes.
//! The digit count is `DIGITS`, or the smallest whose simulated noise of every part fits its
//! decoding margin and hides by `STATISTICAL_BITS` under the average-case model (failure
//! probability `2^-FAILURE_EXPONENT` per accepted bound, [`refresh_noise_accepted`]).
//! `SECURITY_BITS > 0` also requires the estimated security of the encodings on the largest
//! modulus and of the refreshed output ([`RnsBggRefresh::security`]) to reach that many bits, and
//! `SIMULATE_ONLY=1` stops after these checks. It runs the Static RHS lifecycle only.
//!
//! | Variable (`MXX_EXPONENT_LUT_REFRESH_TOP_` prefix) | Default | Meaning |
//! | --- | --- | --- |
//! | `RING_DIMENSION`, `CRT_BITS`, `BASE_BITS` | 8, 30, 15 | Ring dimension, prime size and gadget base |
//! | `PRIMES` | 4 | Primes of the reference's level 0 (`>= 3`) |
//! | `EXTENSION_PRIMES` | 14 | Primes beyond level 0 for the children's chains |
//! | `SLOTS` | 1 | Refreshed CRT slots, the first primes of level 0 (`1..=PRIMES`) |
//! | `LWR_DIMENSION`, `SUPPORT_WEIGHT` | 4, 1 | Sparse-LWR dimension and key weight of every PRF |
//! | `BUCKET_WIDTH` | 0 | Row width `W` of a PBC layout of weight above one (`0`: [`PbcParameters::default_bucket_width_limit`](mxx_exponent_lut::program_gadgets::pbc::PbcParameters::default_bucket_width_limit)) |
//! | `INPUT_MODULUS`, `OUTPUT_MODULUS` | 8, 4 | Sparse-LWR input and output moduli |
//! | `ERROR_SIGMA`, `TRAPDOOR_SIGMA` | 0, 4.578 | Encoding error and trapdoor Gaussian widths |
//! | `DIGITS` | 0 | Smudge digit count (`0`: the smallest in `MIN_DIGITS..=MAX_DIGITS`, 1 and 39) |
//! | `STATISTICAL_BITS` | 1 | Hiding margin of the smudge over each decoder's noise |
//! | `FAILURE_EXPONENT` | 100 | Average-case failure probability exponent of the noise checks |
//! | `SECURITY_BITS` | 0 | Required estimated security (`0`: not checked) |
//! | `SIMULATE_ONLY` | unset | `1`: only the accepted parameters and the checks above |
//! | `NOISE_MODEL` | `worst` | Claim of an export without a recorded set: `worst`, or `average` (failure probability `2^-FAILURE_EXPONENT`, the sampling tape) |
//! | `PARAMETER_ID` | unset | A recorded set: a row of `lean/refresh_top/parameters.csv` |
//!
//! A recorded set fixes every setting it has a column for (the noise model, the failure exponent,
//! the ring, the primes, the slots, the digits, the Gaussian widths, the key seed and the PRF), and
//! exports its claim to `lean/refresh_top/<id>/generated`.
//!
//! The noise and the graphs depend on the key shapes only. The digit search, `SIMULATE_ONLY` and an
//! `EXPORT_ONLY` run without a recorded set generate no PRF keys: they declare each key set by a
//! commitment derived from `KEY_SEED` and its role ([`PrfSettings::declared_shape`]). A round trip
//! generates its keys (one layout per PRF member) and commits to their digests, as does the
//! export of a recorded set, so that it is the claim its round trip exports. An export binds the
//! children's selector plaintexts, tables and PRF inputs by shape only, and states the claim of
//! the recorded graphs ([`RefreshRun::protocol`]).
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
use mxx_exponent_lut::{
    api,
    program::{InputBinding, WireId},
    refresh::{PrfSettings, RefreshDeclaration, TopReference, bit_flip_source},
    rns_bgg::{RefreshRun, RnsBggRefresh, refresh_noise_accepted},
    utils::env_usize,
};
use mxx_ir_core::ValidatedGraph;
use num_bigint::{BigInt, BigUint};
use serial_test::serial as sequential;
use std::{collections::BTreeMap, env};

const PREFIX: &str = "MXX_EXPONENT_LUT_REFRESH_TOP_";
/// The stage hash key, context and invocation label tag (fixed, so that a recorded set exports
/// the same claim on every run).
const HASH_KEY: &[u8; 32] = b"refresh-top-rns-public-api-key-1";
const CONTEXT: [u8; 32] = *b"refresh-top-rns-public-api-ctx-1";
const LABEL_TAG: &[u8] = b"top-invocation-0";
/// The tag of the refresh's hash-sampled matrix `A'`.
const MATRIX_TAG: &[u8] = b"top-refresh-round-trip";

type ParameterRow = BTreeMap<String, String>;

/// The sparse-LWR PRF of every child
/// (`MXX_EXPONENT_LUT_REFRESH_TOP_{LWR_DIMENSION,SUPPORT_WEIGHT,INPUT_MODULUS,OUTPUT_MODULUS,
/// BUCKET_WIDTH}`).
fn prf_settings(row: Option<&ParameterRow>) -> PrfSettings {
    let settings = PrfSettings {
        lwr_dimension: row_setting(row, "lwr_dimension", "LWR_DIMENSION", 4),
        support_weight: row_setting(row, "support_weight", "SUPPORT_WEIGHT", 1),
        input_modulus: row_setting(row, "input_modulus", "INPUT_MODULUS", 8),
        output_modulus: row_setting(row, "output_modulus", "OUTPUT_MODULUS", 4),
        bucket_width: row_setting(row, "bucket_width", "BUCKET_WIDTH", 0),
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

/// The recorded parameter set selected by `MXX_EXPONENT_LUT_REFRESH_TOP_PARAMETER_ID`, if any.
fn recorded_parameters() -> Option<(String, ParameterRow)> {
    env::var(format!("{PREFIX}PARAMETER_ID"))
        .ok()
        .map(|id| (id.clone(), common::parameter_row("refresh_top", &id)))
}

/// `MXX_EXPONENT_LUT_REFRESH_TOP_EXPORT_ONLY=1`: build and export the claim without a round trip.
/// The children bind their plaintexts and tables by shape only, and the keys of a set without a
/// recorded row are only declared ([`PrfSettings::declared_shape`]).
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

/// The accepted refresh of the settings (or of the recorded row `row`) and the source input
/// `X^b`.
fn setup(row: Option<&ParameterRow>) -> (RnsBggRefresh, WireId) {
    let primes = row_setting(row, "primes", "PRIMES", 4);
    assert!(primes >= 3, "{PREFIX}PRIMES must be at least 3");
    let slots = row_setting(row, "slots", "SLOTS", 1);
    assert!((1..=primes).contains(&slots), "{PREFIX}SLOTS must be in 1..={PREFIX}PRIMES");
    let sigma = row_decimal(row, "error_sigma", "ERROR_SIGMA", "0");
    let trapdoor_sigma = row_decimal(row, "trapdoor_sigma", "TRAPDOOR_SIGMA", "4.578");
    let bits = setting("STATISTICAL_BITS", 1);
    let failure_exponent = setting("FAILURE_EXPONENT", 100) as u32;
    let seed = row_setting(row, "key_seed", "KEY_SEED", 0) as u64;
    let ring_dimension = row_setting(row, "ring_dimension", "RING_DIMENSION", 8);
    assert!(ring_dimension.is_power_of_two(), "{PREFIX}RING_DIMENSION must be a power of two");
    let crt_bits = row_setting(row, "crt_bits", "CRT_BITS", 30);
    let base_bits = row_setting(row, "base_bits", "BASE_BITS", 15);
    let extension_primes = row_setting(row, "extension_primes", "EXTENSION_PRIMES", 14);
    let prf = prf_settings(row);
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
    let (source, input, output) = bit_flip_source().expect("source Program");
    let trapdoor = trapdoor_sigma.to_f64().expect("finite trapdoor sigma");
    let refresh = |digits: usize, generate: bool| {
        prf.declare_refresh(
            &source,
            output,
            &reference,
            vec![primes, primes - 1, primes - 2],
            (0..slots).collect(),
            extension.clone(),
            digits,
            (seed, generate),
            MATRIX_TAG.to_vec(),
        )
        .expect("top refresh declaration")
    };
    let fits = |top: &RefreshDeclaration| {
        refresh_noise_accepted(top, &source, &sigma, trapdoor, bits, failure_exponent)
            .expect("refresh noise")
    };
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
    let refresh =
        RnsBggRefresh::new(top, source, keys, &sigma, trapdoor_sigma).expect("refresh chains");
    eprintln!(
        "top refresh accepted: n={ring_dimension} crt_bits={crt_bits} base_bits={base_bits} \
         primes={primes} slots={slots} chains={} columns={} digits={digits} {prf:?} \
         statistical_bits={bits} failure_exponent={failure_exponent}",
        refresh.top.chains(),
        reference.columns()
    );
    security_gate(&refresh, setting("SECURITY_BITS", 0) as u64);
    (refresh, input)
}

/// `SECURITY_BITS > 0`: the estimated security of the encodings and of the refreshed output
/// ([`RnsBggRefresh::security`]) must reach `target` bits.
fn security_gate(refresh: &RnsBggRefresh, target: u64) {
    if target == 0 {
        return;
    }
    let security = refresh.security().expect("security estimate");
    eprintln!(
        "top refresh security: largest modulus {} bits ({} primes) encoding={} output={} \
         target={target}",
        security.largest_modulus_bits,
        security.largest_modulus_primes,
        security.encoding,
        security.output
    );
    assert!(
        security.encoding >= target && security.output >= target,
        "the security estimate reaches {target} bits"
    );
}

/// Every part's stages on `runtime`, the decode stage, and, for a recorded set, the export of
/// the claim of the recorded graphs.
fn round_trip(
    (refresh, input): (RnsBggRefresh, WireId),
    runtime: &mut impl api::StageExecution<api::StageStore>,
    mut store: api::StageStore,
    recorded: Option<(String, ParameterRow)>,
) {
    let bit = 0;
    let lhs = |level: &DCRTPolyParams| {
        BTreeMap::from([(input, InputBinding::Scalar(DCRTPoly::const_rotate_poly(level, bit)))])
    };
    let mut run = refresh
        .run((CONTEXT, *HASH_KEY, LABEL_TAG), export_only(), lhs, runtime, &mut store)
        .expect("top refresh stages");
    let expected = DCRTPoly::const_rotate_poly(&refresh.reference.levels()[0], 1 - bit);
    if export_only() {
        let decoded = run.decode(&refresh, &expected, runtime, &mut store);
        assert!(decoded.is_err(), "the export-only decode graph is not executed");
        let directory = export_directory(recorded.as_ref());
        export_claim(&directory, recorded.as_ref().map(|(_, row)| row), &refresh, &run);
        return;
    }
    let keys = refresh.keys.as_ref().expect("a round trip generates the keys");
    let errors = run.error(&refresh, &expected, runtime, &mut store).expect("top refresh error");
    for (component, rows) in errors.iter().enumerate() {
        let fresh = keys
            .fresh_noise(&refresh.top, component, *HASH_KEY, LABEL_TAG)
            .expect("fresh noise samples");
        for (column, coefficients) in rows.iter().enumerate() {
            let expected = fresh[column].iter().copied().map(BigInt::from).collect::<Vec<_>>();
            assert_eq!(coefficients, &expected, "component {component}, column {column}");
        }
    }
    let decoded = run.decode(&refresh, &expected, runtime, &mut store).expect("top refresh decode");
    assert_eq!(decoded, BigInt::from(0), "every refreshed coefficient decodes to zero");
    eprintln!(
        "top refresh round trip: X^{} re-encrypted over {} chains with fresh error at most 1",
        1 - bit,
        refresh.top.chains()
    );
    if let Some(recorded) = &recorded {
        let directory = export_directory(Some(recorded));
        export_claim(&directory, Some(&recorded.1), &refresh, &run);
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

/// Export the claim of the recorded run `run` of `refresh` to `directory`.
fn export_claim(
    directory: &std::path::Path,
    row: Option<&ParameterRow>,
    refresh: &RnsBggRefresh,
    run: &RefreshRun,
) {
    let protocol =
        run.protocol(refresh, claim_failure_exponent(row)).expect("valid top refresh protocol");
    mxx_ir_core::lean::protocol::export(&protocol, directory, lean_facts())
        .expect("top refresh claim export");
    eprintln!("exported the top refresh correctness claim to {}", directory.display());
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
        let levels = setup.0.reference.levels().to_vec();
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
    let refresh = &setup.0;
    let key =
        |level: &DCRTPolyParams| (level.ring_dimension(), level.to_crt().0, level.base_bits());
    let mut levels = refresh.reference.distinct_levels().into_iter().map(key).collect::<Vec<_>>();
    for (chain, chains) in refresh.chains.iter().enumerate() {
        for parameters in chains.parameters() {
            levels.extend(parameters.distinct_levels().into_iter().map(key));
        }
        let prime = refresh.top.output_prime(chain).expect("output prime");
        levels.push(key(&chains.pool.select_modulus(&BigUint::from(prime)).expect("output ring")));
    }
    // The parts' residue terms live in the ring of the product of the refreshed primes.
    let product = refresh.top.output_primes().into_iter().map(BigUint::from).product::<BigUint>();
    levels.push(key(&refresh.chains[0].pool.select_modulus(&product).expect("terms ring")));
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
