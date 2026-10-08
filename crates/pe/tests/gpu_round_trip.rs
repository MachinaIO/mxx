#![cfg(feature = "gpu")]

//! The predicate encryption round trip on the GPU runtime, at parameters whose noise guarantees
//! correctness and whose Ring-LWE instances reach a lattice security target.
//!
//! The parameter search tries the ring dimensions in order and, for each, the CRT depths through
//! [`PeParameters::select`]: a candidate is accepted when the worst-case simulation of the class's
//! longest program decrypts (`4 E_PE + 2 < q`) and both Ring-LWE instances of
//! [`PeParameters::security_bits`] (the encryption secret and TrapGen) reach the target according
//! to `lattice-estimator-cli`, which must be on `PATH` with its Sage environment. Every cutoff is
//! derived from the tail budget of `STATISTICAL_BITS`.
//!
//! On the accepted parameters, every stage runs on the GPU with file-backed artifacts: setup,
//! one encryption of a random attribute and a random binary message, and keys for two random
//! branching programs of the class's maximal length, one authorized (`F(x) = 0`) and one not. The
//! authorized key must decrypt the message; the unauthorized decryption must complete, and its
//! rounded phase is not constrained.
//!
//! The parameters are checked only for the conditions above: the manuscript's leftover-hash,
//! noise-flooding, and key-width conditions, and the anchor convention of the existing TrapGen,
//! are not enforced (see the `mxx-pe` crate documentation).
//!
//! | Variable (`MXX_PE_ROUND_TRIP_` prefix) | Default | Meaning |
//! | --- | --- | --- |
//! | `ATTRIBUTE_LENGTH`, `ALPHABET_SIZE`, `STATE_WIDTH`, `MAX_STEPS` | 1, 2, 1, 1 | The class |
//! | `MIN_LOG_RING_DIMENSION`, `MAX_LOG_RING_DIMENSION` | 12, 15 | Ring-dimension candidates |
//! | `MAX_CRT_DEPTH` | 12 | CRT-depth candidates `1..=` |
//! | `CRT_BITS`, `BASE_BITS` | 50, `ceil(CRT_BITS / 2)` | CRT prime and gadget base widths |
//! | `ERROR_SIGMA`, `FLOODING_SIGMA`, `PROGRAMMING_SIGMA` | 3.19 | Sigmas of `chi`, `chi_fl`, `Z_t` |
//! | `TRAPDOOR_SIGMA` | 4.578 | TrapGen sigma |
//! | `STATISTICAL_BITS` | 40 | Tail budget of every cutoff |
//! | `SECURITY_BITS` | 128 | Required lattice security of both instances |

use bigdecimal::BigDecimal;
use mxx_backends::{
    GpuRuntime,
    backend::poly_gpu::gpu_backend_on,
    poly::{
        PolyParams,
        dcrt::gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
    },
};
use mxx_exponent_lut::{
    api::StageStorage,
    utils::{env_usize, ring_dimension_candidates},
};
use mxx_pe::{
    BranchingProgram, PeClass, PeNoiseReport, PeParameterRequest, PeParameters, decrypt, encrypt,
    keygen, setup,
};
use serial_test::serial as sequential;
use std::{env, time::Instant};

const PREFIX: &str = "MXX_PE_ROUND_TRIP_";

fn variable(name: &str) -> String {
    format!("{PREFIX}{name}")
}

fn decimal(name: &str, default: &str) -> BigDecimal {
    let name = variable(name);
    env::var(&name)
        .unwrap_or_else(|_| default.to_owned())
        .parse()
        .unwrap_or_else(|error| panic!("{name} must be a decimal: {error}"))
}

fn count(name: &str, default: usize) -> usize {
    env_usize(&variable(name), default).unwrap()
}

/// The first accepted candidate over the ring dimensions, with its noise report.
fn search(class: PeClass) -> (PeParameters, PeNoiseReport) {
    let crt_bits = count("CRT_BITS", 50);
    let base_bits = u32::try_from(count("BASE_BITS", crt_bits.div_ceil(2))).unwrap();
    let max_depth = count("MAX_CRT_DEPTH", 12);
    let security = u64::try_from(count("SECURITY_BITS", 128)).unwrap();
    assert!(security > 0, "{} must be positive", variable("SECURITY_BITS"));
    let trapdoor_sigma = decimal("TRAPDOOR_SIGMA", "4.578").to_string().parse::<f64>().unwrap();
    let mut failures = Vec::new();
    for ring_dimension in ring_dimension_candidates(
        &variable("MIN_LOG_RING_DIMENSION"),
        &variable("MAX_LOG_RING_DIMENSION"),
        12,
        15,
    )
    .unwrap()
    {
        let request = PeParameterRequest {
            class,
            ring_dimension,
            crt_bits,
            base_bits,
            crt_depths: 1..=max_depth,
            trapdoor_sigma,
            error_sigma: decimal("ERROR_SIGMA", "3.19"),
            flooding_sigma: decimal("FLOODING_SIGMA", "3.19"),
            programming_sigma: decimal("PROGRAMMING_SIGMA", "3.19"),
            statistical_bits: u32::try_from(count("STATISTICAL_BITS", 40)).unwrap(),
            lattice_security_bits: Some(security),
        };
        match PeParameters::select(&request) {
            Ok(accepted) => return accepted,
            Err(error) => failures.push(format!("n = {ring_dimension}: {error}")),
        }
    }
    panic!("no parameter candidate was accepted: {}", failures.join("; "));
}

/// A random program of the class's maximal length whose output on `attribute` is `value`:
/// random reads and intermediate tables, and a random final table with its reached entry set.
fn program(class: PeClass, attribute: &[usize], value: usize) -> BranchingProgram {
    let random = |bound: usize| rand::random::<u32>() as usize % bound;
    let entries = class.state_width() * class.alphabet_size();
    let steps = class.max_steps();
    let reads = (0..steps).map(|_| random(class.attribute_length())).collect::<Vec<_>>();
    let mut tables = (0..steps)
        .map(|step| {
            let bound = if step + 1 == steps { 2 } else { class.state_width() };
            (0..entries).map(|_| random(bound)).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut state = 0;
    for (read, table) in reads.iter().zip(&tables).take(steps - 1) {
        state = table[state + class.state_width() * attribute[*read]];
    }
    tables[steps - 1][state + class.state_width() * attribute[reads[steps - 1]]] = value;
    let program = BranchingProgram::new(class, reads, tables).unwrap();
    assert_eq!(program.evaluate(attribute).unwrap(), value);
    program
}

#[test]
#[sequential(dcrt_runtime)]
fn test_gpu_pe_round_trip() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
    let devices = detected_gpu_device_ids();
    assert!(!devices.is_empty(), "the PE round trip needs a CUDA device");
    let class = PeClass::new(
        count("ATTRIBUTE_LENGTH", 1),
        count("ALPHABET_SIZE", 2),
        count("STATE_WIDTH", 1),
        count("MAX_STEPS", 1),
    )
    .unwrap();

    let started = Instant::now();
    let (parameters, report) = search(class);
    assert!(report.decrypts(), "the accepted parameters must guarantee correctness");
    let (encryption_bits, trapdoor_bits) = parameters.security_bits().unwrap();
    let level = parameters.level();
    eprintln!(
        "accepted PE parameters in {:?}: n = {}, CRT {} x {} bits (log2 q = {}), base 2^{}, \
         m = {}, m_B = {}, W = {}, T_max = {}; log2 E_F = {}, log2 E_PE = {}; security \
         {encryption_bits} bits (encryption), {trapdoor_bits} bits (trapdoor)",
        started.elapsed(),
        level.ring_dimension(),
        level.to_crt().0.len(),
        level.crt_bits(),
        report.modulus.bits(),
        level.base_bits(),
        parameters.state_columns(),
        parameters.anchor_columns(),
        class.lookup_width(),
        class.max_steps(),
        report.evaluation.bits(),
        report.decryption.bits(),
    );

    let mut execution = GpuRuntime::new(gpu_backend_on(
        [GpuDCRTPolyParams::new(level.ring_dimension(), level.to_crt().0, level.base_bits(), None)],
        devices,
    ))
    .expect("GPU runtime");
    let directory = tempfile::tempdir().expect("artifact directory");
    let mut store = StageStorage::File(directory.path().into()).open().expect("stage storage");

    let stage = Instant::now();
    let (public_key, master) = setup(&parameters, &mut execution, &mut store).unwrap();
    eprintln!("setup: {:?}", stage.elapsed());
    let attribute = (0..class.attribute_length())
        .map(|_| rand::random::<u32>() as usize % class.alphabet_size())
        .collect::<Vec<_>>();
    let message = (0..level.ring_dimension()).map(|_| rand::random::<bool>()).collect::<Vec<_>>();
    let stage = Instant::now();
    let ciphertext =
        encrypt(&public_key, &attribute, &message, &mut execution, &mut store).unwrap();
    eprintln!("encryption: {:?}", stage.elapsed());
    for value in [0, 1] {
        let predicate = program(class, &attribute, value);
        let stage = Instant::now();
        let key = keygen(&public_key, &master, &predicate, &mut execution, &mut store).unwrap();
        eprintln!("key generation (F(x) = {value}): {:?}", stage.elapsed());
        let stage = Instant::now();
        let decrypted =
            decrypt(&public_key, &key, &ciphertext, &mut execution, &mut store).unwrap();
        eprintln!("decryption (F(x) = {value}): {:?}", stage.elapsed());
        if value == 0 {
            assert_eq!(decrypted, message, "the authorized key must decrypt the message");
        }
    }
}
