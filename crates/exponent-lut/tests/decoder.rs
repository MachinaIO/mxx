#![cfg(feature = "gpu")]

//! The trapdoor decoder round trip on the GPU runtime.
//!
//! The four stages of the trapdoor unmask [`mxx_exponent_lut::decoder::unmask`] run as separate
//! artifact stages, exchanging only published artifacts:
//!
//! - trusted setup samples the public `B` (`2 x 2 (digits + 2)`) and its trapdoor;
//! - preprocessing samples a key `A in R_q^(2 x w)`, republishes `B`, and samples the short
//!   preimage `K` with `B K = A` from the setup's trapdoor;
//! - encryption samples the ternary secret `s = (s_bar, 1)` and publishes the decoder row `b = s B
//!   + e_b` and the encoding `c = s A - Delta mu + e_c` of a message `mu in Z_p^w` (`Delta =
//!   floor(q / p)`);
//! - decoding computes `b K - c = Delta mu + e_b K - e_c`.
//!
//! The parameter search accepts the first ring dimension and CRT depth whose worst-case decoder
//! error `N m_B B_e B_K + B_e` (`m_B = 2 (digits + 2)`, `B_e` the error cutoff, `B_K` the
//! preimage cutoff) leaves `round(p x / q) = mu` with the margin of the reduction remainder, and
//! whose lattice security reaches the target. Every stage is planned by the GPU runtime
//! immediately before it executes; the plan's predicted time is logged as the benchmark estimate
//! ("planned Exponent-LUT GPU stage"). The round trip then checks that every decoded coefficient
//! rounds to the message and that its error stays within the simulated bound. The stages always
//! execute: the plan's I/O trial of the trapdoor import reads the setup's artifact, which an
//! estimate-only run (`MXX_EXPONENT_LUT_ESTIMATE_ONLY=1`) would not have produced.
//!
//! | Variable (`MXX_EXPONENT_LUT_DECODER_` prefix) | Default | Meaning |
//! | --- | --- | --- |
//! | `MIN_LOG_RING_DIMENSION`, `MAX_LOG_RING_DIMENSION` | 5, 12 | Ring-dimension candidates |
//! | `MAX_CRT_DEPTH` | 8 | CRT-depth candidates `1..=` |
//! | `CRT_BITS`, `BASE_BITS` | 40, `ceil(CRT_BITS / 2)` | CRT prime and gadget base widths |
//! | `PLAINTEXT_MODULUS` | 16 | `p` |
//! | `WIDTH` | 2 | Key columns `w` |
//! | `ERROR_SIGMA`, `TRAPDOOR_SIGMA` | 4, 4.578 | Error and trapdoor Gaussian widths |
//! | `SECURITY_BITS` | 1 | Required lattice security |

use bigdecimal::{BigDecimal, ToPrimitive};
use mxx_backends::{
    GpuRuntime,
    backend::poly_gpu::gpu_backend_on,
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
            params::DCRTPolyParams,
        },
    },
    sampler::bounds::{default_preimage_cutoff, hard_cutoff_from_sigma_bound},
};
use mxx_dsl::{DslContext, Mat, Ring};
use mxx_exponent_lut::{
    api::{self, StageArtifacts, StageExecution, StageGraph},
    decoder::unmask::{UNMASK_ROWS, preimage, sample_trapdoor, unmask, unmask_hint},
    utils::{decimal_to_real_expr, env_usize, lattice_security_bits, ring_dimension_candidates},
};
use mxx_ir_core::{IntExpr, ParamEnv, artifact::ArtifactAvailability, node::ConcatAxis};
use num_bigint::{BigInt, BigUint};
use num_traits::{Signed, Zero};
use rand::Rng;
use serial_test::serial as sequential;
use std::{collections::BTreeMap, env};

const PREFIX: &str = "MXX_EXPONENT_LUT_DECODER_";
/// The setup's trapdoor artifact: `.0` is the public `B`, `.1` the trapdoor.
const TRAPDOOR: &str = "decoder/trapdoor";

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

/// The accepted parameters and the bounds the round trip checks against.
struct Decoder {
    parameters: DCRTPolyParams,
    plaintext_modulus: u64,
    width: usize,
    error_sigma: BigDecimal,
    trapdoor_sigma: BigDecimal,
    /// The error cutoff `B_e` of `e_b` and `e_c`.
    error_bound: BigUint,
    /// The preimage cutoff `B_K` of the trapdoor sampler.
    preimage_bound: BigUint,
    /// The worst-case decoder error `N m_B B_e B_K + B_e`.
    decoding_error: BigUint,
}

impl Decoder {
    fn modulus(&self) -> BigUint {
        self.parameters.modulus().as_ref().clone()
    }

    /// `Delta = floor(q / p)`.
    fn spacing(&self) -> BigUint {
        self.modulus() / self.plaintext_modulus
    }

    fn gadget_base(&self) -> u32 {
        1u32 << self.parameters.base_bits()
    }

    fn ring(&self) -> Ring {
        Ring::from_crt_moduli(
            self.parameters.to_crt().0.into_iter().map(Into::into).collect(),
            self.parameters.ring_dimension(),
        )
    }
}

/// Search the configured candidates: the decoder error must leave a correct rounding and the
/// lattice security must reach the target.
fn search() -> Decoder {
    let crt_bits = setting("CRT_BITS", 40);
    let base_bits = setting("BASE_BITS", crt_bits.div_ceil(2)) as u32;
    assert!((1..=31).contains(&base_bits), "the base bits must be in 1..=31");
    let plaintext_modulus = setting("PLAINTEXT_MODULUS", 16) as u64;
    assert!(plaintext_modulus >= 2, "the plaintext modulus must be at least 2");
    let width = setting("WIDTH", 2);
    assert!(width > 0, "the key width must be positive");
    let required_security = setting("SECURITY_BITS", 1) as u64;
    assert!(required_security > 0, "the security target must be positive");
    let max_crt_depth = setting("MAX_CRT_DEPTH", 8);
    let error_sigma = sigma("ERROR_SIGMA", "4");
    let trapdoor_sigma = sigma("TRAPDOOR_SIGMA", "4.578");
    let error_bound = hard_cutoff_from_sigma_bound(&error_sigma);
    let candidates = ring_dimension_candidates(
        &format!("{PREFIX}MIN_LOG_RING_DIMENSION"),
        &format!("{PREFIX}MAX_LOG_RING_DIMENSION"),
        5,
        12,
    )
    .expect("ring-dimension configuration");
    let mut failures = Vec::new();
    for ring_dimension in candidates {
        for crt_depth in 1..=max_crt_depth {
            let parameters =
                DCRTPolyParams::new(ring_dimension, crt_depth, crt_bits, base_bits, None, None);
            let digits = parameters.modulus_digits();
            let preimage_bound = default_preimage_cutoff(
                ring_dimension,
                UNMASK_ROWS,
                digits,
                1u32 << base_bits,
                trapdoor_sigma.to_f64().expect("finite trapdoor sigma"),
            )
            .expect("valid trapdoor sigma");
            let decoding_error = BigUint::from(ring_dimension) *
                BigUint::from(UNMASK_ROWS * (digits + 2)) *
                &error_bound *
                &preimage_bound +
                &error_bound;
            let decoder = Decoder {
                parameters,
                plaintext_modulus,
                width,
                error_sigma: error_sigma.clone(),
                trapdoor_sigma: trapdoor_sigma.clone(),
                error_bound: error_bound.clone(),
                preimage_bound,
                decoding_error,
            };
            // `round(p x / q)` recovers `mu` when the error and the remainder `q - p Delta`
            // stay within half the spacing.
            let remainder = decoder.modulus() - decoder.spacing() * plaintext_modulus;
            if (&decoder.decoding_error + remainder) * 2u8 >= decoder.spacing() {
                failures.push(format!(
                    "n={ring_dimension} crt_depth={crt_depth}: decoder error {} exceeds half of {}",
                    decoder.decoding_error,
                    decoder.spacing()
                ));
                continue;
            }
            let achieved = lattice_security_bits(&decoder.parameters, &error_sigma)
                .expect("lattice security estimator must be available");
            if achieved < required_security {
                failures.push(format!(
                    "n={ring_dimension} crt_depth={crt_depth}: security {achieved} < {required_security}"
                ));
                continue;
            }
            eprintln!(
                "decoder parameters accepted: ring_dimension={ring_dimension} crt_depth={crt_depth} \
                 crt_bits={crt_bits} base_bits={base_bits} p={plaintext_modulus} w={width} \
                 decoder_error={} spacing={} security_bits={achieved}",
                decoder.decoding_error,
                decoder.spacing()
            );
            return decoder;
        }
    }
    panic!("no decoder parameter candidate was accepted: {}", failures.join("; "));
}

/// Validate, plan, execute, and publish one stage on `runtime`.
fn produce(
    graph: DslContext,
    stages: &[&StageArtifacts<&'static str>],
    name: &'static str,
    runtime: &mut GpuRuntime,
    store: &mut api::StageStore,
) -> StageArtifacts<&'static str> {
    let manifests = stages
        .iter()
        .map(|stage| (stage.production().clone(), stage.manifest().clone()))
        .collect::<BTreeMap<_, _>>();
    api::execute_artifacts(
        StageGraph::new(graph.build().expect("stage graph"), BTreeMap::new(), manifests),
        &ParamEnv::default(),
        name,
        rand::random(),
        runtime,
        store,
        |_| Ok(()),
    )
    .unwrap_or_else(|error| panic!("decoder {name} stage: {error}"))
}

/// The polynomial with coefficients `values` in `ring`.
fn polynomial(ring: &Ring, values: &[BigUint]) -> Mat {
    ring.polynomial(values.iter().map(|value| IntExpr::constant(BigInt::from(value.clone()))))
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
    let decoder = search();
    let parameters = &decoder.parameters;
    let ring = decoder.ring();
    let ring_dimension = parameters.ring_dimension() as usize;
    let digits = parameters.modulus_digits();
    let columns = UNMASK_ROWS * (digits + 2);
    let width = decoder.width;
    let trapdoor_sigma = decimal_to_real_expr(&decoder.trapdoor_sigma).expect("trapdoor sigma");
    let error_sigma = decimal_to_real_expr(&decoder.error_sigma).expect("error sigma");
    let error_bound = BigInt::from(decoder.error_bound.clone());
    let preimage_bound = BigInt::from(decoder.preimage_bound.clone());
    let mut runtime = GpuRuntime::new(gpu_backend_on(
        [GpuDCRTPolyParams::new(
            parameters.ring_dimension(),
            parameters.to_crt().0,
            parameters.base_bits(),
            None,
        )],
        devices,
    ))
    .expect("GPU runtime");
    let directory = tempfile::tempdir().expect("artifact directory");
    let mut store = api::StageStorage::File(directory.path().into()).open().expect("stage storage");

    // Trusted setup: `B` and its trapdoor.
    let trapdoor = sample_trapdoor(
        &ring,
        trapdoor_sigma.clone(),
        decoder.gadget_base(),
        digits,
        preimage_bound.clone(),
    );
    let setup = produce(
        DslContext::new("decoder-setup").transferred_output(TRAPDOOR, trapdoor).unwrap(),
        &[],
        "setup",
        &mut runtime,
        &mut store,
    );

    // Preprocessing: the key `A`, the republished `B`, and the preimage `K` of `A`.
    let trapdoor = ring.trapdoor_artifact_input(
        setup.production().clone(),
        format!("{TRAPDOOR}.1"),
        UNMASK_ROWS,
        trapdoor_sigma,
        decoder.gadget_base(),
        digits,
        preimage_bound.clone(),
    );
    let key = ring.uniform_residue((UNMASK_ROWS, width));
    let preprocess = produce(
        DslContext::new("decoder-preprocess")
            .transferred_output("key", key.clone())
            .unwrap()
            .transferred_output("public", trapdoor.public_matrix())
            .unwrap()
            .transferred_output("preimage", preimage(&trapdoor, key))
            .unwrap(),
        &[&setup],
        "preprocess",
        &mut runtime,
        &mut store,
    );

    // Encryption: the decoder row and the encoding of `Delta mu` under the secret `(s_bar, 1)`.
    let mut rng = rand::rng();
    let message = (0..width)
        .map(|_| {
            (0..ring_dimension)
                .map(|_| rng.random_range(0..decoder.plaintext_modulus))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let spacing = decoder.spacing();
    let payload = Mat::concat(
        ConcatAxis::Columns,
        message
            .iter()
            .map(|column| {
                let scaled = column.iter().map(|&mu| &spacing * mu).collect::<Vec<_>>();
                polynomial(&ring, &scaled)
            })
            .collect(),
    );
    let key = ring.artifact_input(
        preprocess.production().clone(),
        "key",
        (UNMASK_ROWS, width),
        ArtifactAvailability::Transferred,
    );
    let public = ring.artifact_input(
        preprocess.production().clone(),
        "public",
        (UNMASK_ROWS, columns),
        ArtifactAvailability::Transferred,
    );
    let secret = Mat::concat(
        ConcatAxis::Columns,
        vec![ring.uniform_interval((1, 1), -1, 1), ring.identity(1)],
    );
    let row = unmask_hint(
        secret.clone(),
        public,
        ring.gaussian((1, columns), error_sigma.clone(), error_bound.clone()),
    );
    let encoding =
        secret * key - payload + ring.gaussian((1, width), error_sigma, error_bound.clone());
    let encryption = produce(
        DslContext::new("decoder-encrypt")
            .transferred_output("row", row)
            .unwrap()
            .transferred_output("encoding", encoding)
            .unwrap(),
        &[&preprocess],
        "encrypt",
        &mut runtime,
        &mut store,
    );

    // Decoding: `b K - c`.
    let preimage = ring.preimage_artifact_input(
        preprocess.production().clone(),
        "preimage",
        (columns, width),
        preimage_bound,
        ArtifactAvailability::Transferred,
    );
    let row = ring.artifact_input(
        encryption.production().clone(),
        "row",
        (1, columns),
        ArtifactAvailability::Transferred,
    );
    let encoding = ring.artifact_input(
        encryption.production().clone(),
        "encoding",
        (1, width),
        ArtifactAvailability::Transferred,
    );
    let manifests = [&preprocess, &encryption]
        .into_iter()
        .map(|stage| (stage.production().clone(), stage.manifest().clone()))
        .collect();
    let graph = DslContext::new("decoder-decode")
        .output("decoded", unmask(row, preimage, encoding))
        .unwrap()
        .build()
        .expect("decoding graph");
    let outputs = api::execute_graph(
        StageGraph::new(graph, BTreeMap::new(), manifests),
        &ParamEnv::default(),
        &mut runtime,
        &mut store,
        |_| Ok(()),
    )
    .expect("decoder decode stage");
    let decoded =
        StageExecution::<api::StageStore>::download_matrix(&runtime, &outputs.outputs["decoded"])
            .expect("decoded payload");

    // Every coefficient rounds to its message, with an error within the simulated bound.
    let modulus = BigInt::from(decoder.modulus());
    let spacing = BigInt::from(spacing);
    let p = BigInt::from(decoder.plaintext_modulus);
    let limit = BigInt::from(decoder.decoding_error.clone());
    let mut largest = BigInt::zero();
    for (column, expected) in message.iter().enumerate() {
        let coefficients = decoded.entry(0, column).coeffs_biguints();
        assert_eq!(coefficients.len(), ring_dimension);
        for (index, (coefficient, &mu)) in coefficients.into_iter().zip(expected).enumerate() {
            let value = BigInt::from(coefficient);
            let rounded = ((&value * &p * 2 + &modulus) / (&modulus * 2)) % &p;
            assert_eq!(
                rounded,
                BigInt::from(mu),
                "column {column}, coefficient {index}: decoded {value} does not round to {mu}"
            );
            let mut error = (&value - &spacing * mu) % &modulus;
            if &error * 2 > modulus {
                error -= &modulus;
            }
            assert!(
                error.abs() <= limit,
                "column {column}, coefficient {index}: error {error} exceeds the bound {limit}"
            );
            largest = largest.max(error.abs());
        }
    }
    eprintln!(
        "decoder round trip: {} coefficients decoded; largest error {largest} of the bound {limit}",
        width * ring_dimension
    );
}
