#![cfg(feature = "gpu")]
//! The parameters and the Lean correctness claim of one Ring-GSW multiplication evaluated as a
//! layered Boolean circuit over BGG+ encodings.
//!
//! The Ring-GSW product of two ciphertexts with lookup-free Montgomery entries
//! (`mxx_khe::circuit_gadgets::fhe::ring_gsw` over one-bit `MontgomeryPoly` limbs) is lowered to
//! a layered Boolean circuit, whose depth and width this test computes without expanding it
//! (`mxx_khe::circuit::boolean_layering_bound`). The circuit is evaluated by the closed protocol of
//! `mxx_khe::bgg::boolean_protocol`: BGG+ encodings `c_i = s (A_i - x_i G) + e_i` of the input
//! bits under `s = (s_bar, 1)`, the layered evaluation, the decoding action
//! `G^(-1)(floor(q / 2) e_2)`, and the rounding of `2 (c' - s A') / q` to the output bit.
//!
//! `test_gpu_ring_gsw_boolean_bgg_parameters` simulates the BGG+ errors of that evaluation
//! (`mxx_khe::noise_simulate`) and selects the smallest ring dimension and CRT depth that decode
//! and whose Ring-LWE estimate (`lattice-estimator-cli`, which must be on `PATH` with its Sage
//! environment) reaches the security target, printing the selection as a parameter row. With
//! `PARAMETER_ID` it instead checks that row of `lean/ring_gsw_bgg/parameters.csv` against the
//! lowered circuit, the simulation, and the estimate, and exports the protocol's average-case
//! correctness claim to `lean/ring_gsw_bgg/<id>/generated`. The protocol's execution is covered by
//! the CPU unit tests of `mxx_khe::bgg::boolean_protocol`: the GPU runtime cannot yet carry a
//! matrix family through a sequential loop, which the layered evaluation needs.
//!
//! Lowering the default Ring-GSW product needs about 40 GiB of host memory.
//!
//! Environment variables (prefix `MXX_KHE_RING_GSW_BGG_`):
//!
//! | Variable | Default | Meaning |
//! | --- | --- | --- |
//! | `FHE_RING_DIMENSION`, `FHE_CRT_BITS`, `FHE_CRT_DEPTH` | 2048, 40, 1 | Ring-GSW ring |
//! | `ERROR_SIGMA` | 3.19 | encoding error width; the cutoff is `floor(6.5 sigma)` |
//! | `CRT_BITS`, `MAX_BASE_BITS`, `MAX_CRT_DEPTH` | 60, 20, 512 | BGG+ CRT width, gadget bases `2^1..2^MAX`, depth limit |
//! | `MIN_LOG_RING_DIMENSION`, `MAX_LOG_RING_DIMENSION` | 16, 21 | BGG+ ring dimensions |
//! | `SECURITY_BITS` | 128 | Ring-LWE security target |
//! | `PARAMETER_ID` | unset | check and export this row of the parameter table |

use bigdecimal::BigDecimal;
use mxx_backends::{
    lattice_estimator::lattice_security_bits,
    poly::{
        PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    sampler::bounds::hard_cutoff_from_sigma_bound,
};
use mxx_ir_core::Rational;
use mxx_khe::{
    bgg::boolean_protocol::{
        BooleanBggProtocolGraphs, BooleanBggProtocolParameters, SECRET_DIMENSION,
        boolean_bgg_protocol, circuit_bindings,
    },
    circuit::{PolyCircuit, boolean_layering_bound},
    circuit_gadgets::{
        arith::{CrtWindow, MontgomeryPoly, MontgomeryPolyContext},
        fhe::ring_gsw::{RingGswCiphertext, RingGswContext},
    },
    noise_simulate::{
        BooleanBggNoise, BooleanBggParameterRequest, BooleanBggSelection, select_boolean_parameters,
    },
};
use num_bigint::{BigInt, BigUint};
use std::{path::PathBuf, str::FromStr, sync::Arc};

const PREFIX: &str = "MXX_KHE_RING_GSW_BGG_";

fn env_string(name: &str) -> Option<String> {
    std::env::var(format!("{PREFIX}{name}")).ok()
}

fn env_usize(name: &str, default: usize) -> usize {
    env_string(name).map_or(default, |value| {
        value.parse().unwrap_or_else(|_| panic!("{PREFIX}{name} must be a nonnegative integer"))
    })
}

fn env_decimal(name: &str, default: &str) -> BigDecimal {
    BigDecimal::from_str(&env_string(name).unwrap_or_else(|| default.to_owned()))
        .unwrap_or_else(|_| panic!("{PREFIX}{name} must be a decimal"))
}

fn lean_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("lean/ring_gsw_bgg")
}

/// The product `C1 G^(-1)(C2)` of two Ring-GSW ciphertexts whose entries are Montgomery
/// polynomials with one-bit limbs over the CRT basis of `parameters`. The inputs are the bits of
/// both ciphertexts and the outputs the bits of the product, entry by entry, then by q-level,
/// coefficient, and limb.
fn ring_gsw_product(parameters: &DCRTPolyParams) -> PolyCircuit<DCRTPoly> {
    let dimension = parameters.ring_dimension() as usize;
    let window = CrtWindow::full(parameters.to_crt().0.len());
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let arithmetic = Arc::new(MontgomeryPolyContext::setup(&mut circuit, parameters, 1, false));
    let context = Arc::new(RingGswContext::from_arith_context(
        &mut circuit,
        parameters,
        dimension,
        arithmetic,
        window,
    ));
    let left = RingGswCiphertext::<DCRTPoly, MontgomeryPoly<DCRTPoly>>::input(
        context.clone(),
        None,
        &mut circuit,
    );
    let right = RingGswCiphertext::input(context, None, &mut circuit);
    let product = left.mul(&right, &mut circuit);
    let outputs = product
        .rows
        .iter()
        .flatten()
        .flat_map(|entry| {
            entry
                .value
                .iter()
                .flatten()
                .flat_map(|residue| residue.limbs.clone())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    circuit.output(outputs);
    circuit
}

/// `value` as an exact rational.
fn rational(value: &BigDecimal) -> Rational {
    let (digits, scale) = value.as_bigint_and_exponent();
    if scale >= 0 {
        Rational::new(digits, BigInt::from(10u8).pow(scale as u32)).unwrap()
    } else {
        Rational::from_integer(digits * BigInt::from(10u8).pow((-scale) as u32))
    }
}

/// One row of `lean/ring_gsw_bgg/parameters.csv`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ParameterRow {
    parameter_id: String,
    fhe_ring_dimension: u32,
    fhe_crt_bits: usize,
    fhe_crt_depth: usize,
    circuit_depth: usize,
    instance_width: u128,
    max_layer_width: u128,
    ring_dimension: u32,
    crt_bits: usize,
    crt_depth: usize,
    base_bits: usize,
    error_sigma: String,
    error_cutoff: u64,
    security_bits: u64,
}

const PARAMETER_HEADER: &str = "parameter_id,fhe_ring_dimension,fhe_crt_bits,fhe_crt_depth,circuit_depth,instance_width,max_layer_width,ring_dimension,crt_bits,crt_depth,base_bits,error_sigma,error_cutoff,security_bits";

impl ParameterRow {
    fn csv(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            self.parameter_id,
            self.fhe_ring_dimension,
            self.fhe_crt_bits,
            self.fhe_crt_depth,
            self.circuit_depth,
            self.instance_width,
            self.max_layer_width,
            self.ring_dimension,
            self.crt_bits,
            self.crt_depth,
            self.base_bits,
            self.error_sigma,
            self.error_cutoff,
            self.security_bits
        )
    }

    fn parse(line: &str) -> Self {
        let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
        assert_eq!(fields.len(), 14, "a parameter row has 14 fields: {line}");
        let number = |index: usize| fields[index].parse::<u128>().expect("a numeric field");
        Self {
            parameter_id: fields[0].to_owned(),
            fhe_ring_dimension: number(1) as u32,
            fhe_crt_bits: number(2) as usize,
            fhe_crt_depth: number(3) as usize,
            circuit_depth: number(4) as usize,
            instance_width: number(5),
            max_layer_width: number(6),
            ring_dimension: number(7) as u32,
            crt_bits: number(8) as usize,
            crt_depth: number(9) as usize,
            base_bits: number(10) as usize,
            error_sigma: fields[11].to_owned(),
            error_cutoff: number(12) as u64,
            security_bits: number(13) as u64,
        }
    }
}

#[test]
fn test_gpu_ring_gsw_boolean_bgg_parameters() {
    let fhe_ring_dimension = env_usize("FHE_RING_DIMENSION", 2048) as u32;
    let fhe_crt_bits = env_usize("FHE_CRT_BITS", 40);
    let fhe_crt_depth = env_usize("FHE_CRT_DEPTH", 1);
    let security_target = env_usize("SECURITY_BITS", 128) as u64;
    let fhe = DCRTPolyParams::new(fhe_ring_dimension, fhe_crt_depth, fhe_crt_bits, 1, None, None);
    let circuit = ring_gsw_product(&fhe);
    let bound = boolean_layering_bound(&circuit);
    drop(circuit);
    println!(
        "RING_GSW_BGG_LAYERING {bound:?} max_layer_width_bound={}",
        bound.max_layer_width_bound()
    );
    let instance_width = bound.input_width as u128;
    let max_layer_width = bound.max_layer_width_bound();

    let recorded = env_string("PARAMETER_ID").map(|parameter_id| {
        let table = std::fs::read_to_string(lean_directory().join("parameters.csv"))
            .expect("the parameter table exists");
        let mut lines = table.lines();
        assert_eq!(lines.next(), Some(PARAMETER_HEADER), "the parameter table header");
        lines
            .map(ParameterRow::parse)
            .find(|row| row.parameter_id == parameter_id)
            .unwrap_or_else(|| panic!("no parameter row {parameter_id}"))
    });
    let row = match recorded {
        Some(row) => {
            assert_eq!(
                (row.fhe_ring_dimension, row.fhe_crt_bits, row.fhe_crt_depth),
                (fhe_ring_dimension, fhe_crt_bits, fhe_crt_depth),
                "the row's Ring-GSW ring is the selected one"
            );
            assert_eq!(
                (row.circuit_depth, row.instance_width, row.max_layer_width),
                (bound.depth, instance_width, max_layer_width),
                "the row records the lowered circuit"
            );
            row
        }
        None => {
            let sigma = env_decimal("ERROR_SIGMA", "3.19");
            let request = BooleanBggParameterRequest {
                depth: bound.depth,
                secret_dimension: SECRET_DIMENSION,
                crt_bits: env_usize("CRT_BITS", 60),
                base_bits: (1..=env_usize("MAX_BASE_BITS", 20)).collect(),
                ring_dimensions: (env_usize("MIN_LOG_RING_DIMENSION", 16)..=
                    env_usize("MAX_LOG_RING_DIMENSION", 21))
                    .map(|log| 1u32 << log)
                    .collect(),
                max_crt_depth: env_usize("MAX_CRT_DEPTH", 512),
                fresh_error_bound: hard_cutoff_from_sigma_bound(&sigma),
                security_bits: security_target,
            };
            let selection: BooleanBggSelection = select_boolean_parameters(&request, |ring_dimension, modulus| {
                let bits = lattice_security_bits(ring_dimension, modulus, &sigma)?;
                println!(
                    "RING_GSW_BGG_ESTIMATE ring_dimension={ring_dimension} log_modulus={} security_bits={bits}",
                    modulus.bits()
                );
                Ok(bits)
            })
            .unwrap();
            ParameterRow {
                parameter_id: format!(
                    "average-gsw{}x{}-n{}-crt{}x{}",
                    fhe_ring_dimension,
                    fhe_crt_bits,
                    selection.noise.ring_dimension,
                    selection.crt_depth(),
                    selection.crt_bits
                ),
                fhe_ring_dimension,
                fhe_crt_bits,
                fhe_crt_depth,
                circuit_depth: bound.depth,
                instance_width,
                max_layer_width,
                ring_dimension: selection.noise.ring_dimension,
                crt_bits: selection.crt_bits,
                crt_depth: selection.crt_depth(),
                base_bits: selection.base_bits,
                error_sigma: sigma.to_string(),
                error_cutoff: u64::try_from(&request.fresh_error_bound).unwrap(),
                security_bits: selection.security_bits,
            }
        }
    };
    println!("RING_GSW_BGG_PARAMETERS {PARAMETER_HEADER}");
    println!("RING_GSW_BGG_PARAMETERS {}", row.csv());

    let moduli =
        mxx_ir_core::generate_crt_basis(row.ring_dimension, row.crt_depth, row.crt_bits).unwrap();
    let modulus: BigUint = moduli.iter().map(|&q| BigUint::from(q)).product();
    let noise = BooleanBggNoise {
        ring_dimension: row.ring_dimension,
        modulus: modulus.clone(),
        secret_dimension: SECRET_DIMENSION,
        digit_count: row.crt_depth * row.crt_bits.div_ceil(row.base_bits),
        gadget_base: BigUint::from(1u8) << row.base_bits,
        fresh_error_bound: BigUint::from(row.error_cutoff),
    };
    let report = noise.simulate(row.circuit_depth);
    println!(
        "RING_GSW_BGG_NOISE log_output_error={} log_modulus={} decrypts={}",
        report.output_error.bits(),
        modulus.bits(),
        report.decrypts()
    );
    assert!(report.decrypts(), "the simulated output error decodes");
    assert!(row.security_bits >= security_target, "the row reaches the security target");
    if env_string("PARAMETER_ID").is_none() {
        return;
    }
    let sigma = BigDecimal::from_str(&row.error_sigma).unwrap();
    assert_eq!(
        hard_cutoff_from_sigma_bound(&sigma),
        BigUint::from(row.error_cutoff),
        "the cutoff is floor(6.5 sigma)"
    );
    assert!(
        lattice_security_bits(row.ring_dimension, &modulus, &sigma).unwrap() >= row.security_bits,
        "the estimate reproduces the recorded security"
    );
    let bgg = BooleanBggProtocolParameters {
        ring: mxx_dsl::Ring::new(row.crt_bits, row.crt_depth, row.ring_dimension),
        base_bits: row.base_bits,
        digit_count: noise.digit_count,
        error: Some((rational(&sigma), BigUint::from(row.error_cutoff))),
    };
    let bindings = circuit_bindings(row.instance_width, row.circuit_depth, row.max_layer_width);
    let graphs = BooleanBggProtocolGraphs::new(&bgg);
    let directory = lean_directory().join(&row.parameter_id).join("generated");
    mxx_ir_core::lean::protocol::export(
        &boolean_bgg_protocol(&graphs, &bgg, &bindings),
        &directory,
    )
    .unwrap();
    println!("RING_GSW_BGG_CLAIM {}", directory.display());
}
