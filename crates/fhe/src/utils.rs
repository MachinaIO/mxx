//! Shared FHE helpers and coefficient-boundary graphs.
//! RNS modulus switching must not call the coefficient-boundary helpers.
use crate::FheError;
#[cfg(feature = "gpu")]
use crate::{BgvHybridParams, BgvParams, FheCommonParams, TfheParams};
use mxx_backends::poly::{PolyParams, dcrt::params::DCRTPolyParams};
#[cfg(test)]
use mxx_dsl::{DslError, select};
use mxx_dsl::{Family, Int, Mat, Ring};
use mxx_ir_core::IntExpr;
#[cfg(feature = "gpu")]
use mxx_ir_core::node::SampleRange;
use num_bigint::BigInt;
#[cfg(any(test, feature = "gpu"))]
use num_bigint::BigUint;

#[cfg(feature = "gpu")]
use num_traits::One;
#[cfg(feature = "gpu")]
use std::env;

// Public modular constants only: no plaintext or ciphertext values are inspected here.
pub(crate) fn pow_mod(mut value: u64, mut exponent: u64, modulus: u64) -> u64 {
    let mut result = 1;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = (u128::from(result) * u128::from(value) % u128::from(modulus)) as u64;
        }
        value = (u128::from(value) * u128::from(value) % u128::from(modulus)) as u64;
        exponent >>= 1;
    }
    result
}

pub(crate) fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    for prime in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if value % prime == 0 {
            return value == prime;
        }
    }
    let shifts = (value - 1).trailing_zeros();
    let odd = (value - 1) >> shifts;
    // Deterministic Miller-Rabin bases covering every unsigned 64-bit integer.
    [2, 325, 9375, 28178, 450775, 9780504, 1795265022].into_iter().all(|base| {
        if base % value == 0 {
            return true;
        }
        let mut x = pow_mod(base % value, odd, value);
        if x == 1 || x == value - 1 {
            return true;
        }
        for _ in 1..shifts {
            x = (u128::from(x) * u128::from(x) % u128::from(value)) as u64;
            if x == value - 1 {
                return true;
            }
        }
        false
    })
}

pub(crate) fn check_family(values: &Family<Int>, count: usize) -> Result<(), FheError> {
    if values.count() != &IntExpr::constant(count) {
        return Err(FheError::ShapeMismatch);
    }
    Ok(())
}

pub(crate) fn check_matrix(
    parameters: &DCRTPolyParams,
    value: &Mat,
    rows: usize,
    columns: usize,
) -> Result<(), FheError> {
    let ty = value.matrix_type();
    if ty.ring != ring(parameters).as_ref().clone() {
        return Err(FheError::LevelMismatch);
    }
    if ty.rows != IntExpr::constant(rows) || ty.columns != IntExpr::constant(columns) {
        return Err(FheError::ShapeMismatch);
    }
    Ok(())
}

pub(crate) fn scalar(parameters: &DCRTPolyParams, value: impl Into<BigInt>) -> Mat {
    ring(parameters).polynomial([IntExpr::constant(value.into())])
}

pub(crate) fn ring(parameters: &DCRTPolyParams) -> Ring {
    Ring::from_crt_moduli(
        parameters.to_crt().0.into_iter().map(IntExpr::from).collect(),
        parameters.ring_dimension(),
    )
}

pub(crate) fn plaintext_ring(modulus: u64, ring_dimension: u32) -> Ring {
    Ring::from_crt_moduli(vec![IntExpr::from(modulus)], ring_dimension)
}

#[cfg(test)]
pub(crate) fn centered(value: Int, modulus: &BigUint) -> Result<Int, DslError> {
    let q = BigInt::from(modulus.clone());
    // select uses index 0 for false and 1 for true; the latter keeps residues
    // at or below q/2, while larger residues are interpreted as value - q.
    let above_half = value.clone().mul(2).less_equal(Int::constant(q.clone()));
    select(above_half.to_int(), vec![value.clone().sub(Int::constant(q)), value])
}

#[cfg(test)]
pub(crate) fn extract(parameters: &DCRTPolyParams, value: &Mat) -> Result<Family<Int>, FheError> {
    check_matrix(parameters, value, 1, 1)?;
    Ok(value.coefficients())
}

#[cfg(feature = "gpu")]
fn integer(name: &str, default: usize) -> usize {
    env::var(name)
        .map(|value| value.parse().unwrap_or_else(|_| panic!("invalid {name}: {value}")))
        .unwrap_or(default)
}

#[cfg(feature = "gpu")]
fn primes(name: &str, defaults: Vec<u64>) -> Vec<u64> {
    env::var(name)
        .map(|value| {
            value
                .split(',')
                .map(|prime| prime.trim().parse().expect("comma-separated CRT primes"))
                .collect()
        })
        .unwrap_or(defaults)
}

#[cfg(feature = "gpu")]
fn common_params(
    n: usize,
    q: Vec<u64>,
    base_bits: usize,
    default_sigma: &str,
    binary: bool,
) -> FheCommonParams {
    let sigma = env::var("FHE_TEST_SIGMA").unwrap_or_else(|_| default_sigma.into());
    let sigma_bound = sigma.parse().expect("positive decimal sigma");
    let error_sigma: f64 = sigma.parse().expect("finite positive sigma");
    assert!(error_sigma.is_finite() && error_sigma > 0.0, "finite positive sigma required");
    let error_cutoff = mxx_backends::sampler::bounds::hard_cutoff_from_sigma_bound(&sigma_bound);
    let binary = match env::var("FHE_TEST_SECRET_DISTRIBUTION").as_deref() {
        Ok("binary") => true,
        Ok("ternary") => false,
        Err(_) => binary,
        _ => panic!("FHE_TEST_SECRET_DISTRIBUTION must be binary or ternary"),
    };
    let bits =
        q.iter().map(|prime| (64 - prime.leading_zeros()) as usize).max().expect("nonempty Q");
    let params = FheCommonParams {
        ring: DCRTPolyParams::try_new(
            n.try_into().expect("ring dimension fits u32"),
            q.len(),
            bits,
            base_bits.try_into().expect("base bits fit u32"),
            Some(q),
            None,
        )
        .expect("valid exact CRT basis"),
        secret_range: SampleRange {
            minimum: (if binary { 0 } else { -1 }).into(),
            maximum: 1.into(),
        },
        error_sigma,
        error_cutoff,
    };
    params.validate().expect("valid FHE parameters");
    params
}

#[cfg(feature = "gpu")]
pub fn bgv_params() -> BgvParams {
    let profile = env::var("FHE_TEST_PROFILE").unwrap_or_else(|_| "bgv-54".into());
    let (q, p, n, t, digit_size, sigma) = match profile.as_str() {
        "bgv-54" => (
            vec![18014398507892737, 18014398508138497, 18014398508400641],
            vec![72057594037616641],
            8192,
            1032193,
            1,
            "3.2",
        ),
        "bgv-36" => (
            vec![68717740033, 68718346241, 68718428161, 68719230977],
            vec![137438773249, 137438822401],
            8192,
            1032193,
            2,
            "3.2",
        ),
        _ => panic!("FHE_TEST_PROFILE must be bgv-54 or bgv-36"),
    };
    BgvParams::new(
        common_params(
            integer("FHE_TEST_RING_DIMENSION", n),
            primes("FHE_TEST_Q_PRIMES", q),
            integer("FHE_TEST_BASE_BITS", 8),
            sigma,
            false,
        ),
        integer("FHE_TEST_PLAINTEXT_MODULUS", t) as u64,
        Some(BgvHybridParams {
            digit_size: integer("FHE_TEST_HYBRID_DIGIT_SIZE", digit_size),
            auxiliary_primes: primes("FHE_TEST_P_PRIMES", p),
        }),
    )
    .expect("valid BGV parameters")
}

/// The TFHE-rs `TFHE_LIB_PARAMETERS` Boolean profile (the original TFHE
/// library set): n = 630, N = 1024, k = 1, LWE noise 2^-15 and ring noise
/// 2^-25 of their moduli, and a base 2^2, 8-digit key switch of the leading
/// 16 bits of q = 2^32. The ring torus 2^32 is replaced by the double-CRT
/// modulus Q = 65537 * 79873 (about 2^32.3) with the same relative noise; its
/// exact per-limb gadget uses two base 2^9 digits per 17-bit limb in place of
/// the profile's three approximate base 2^7 torus digits. `FHE_TEST_TFHE_*`
/// variables override each value.
#[cfg(feature = "gpu")]
pub fn tfhe_params() -> TfheParams {
    let n = integer("FHE_TEST_TFHE_RING_DIMENSION", 1024);
    let base = integer("FHE_TEST_TFHE_BASE_BITS", 9);
    let q = primes("FHE_TEST_TFHE_Q_PRIMES", vec![65_537, 79_873]);
    let ring_sigma = env::var("FHE_TEST_TFHE_RING_SIGMA").unwrap_or_else(|_| "156".into());
    let mut common = common_params(n, q, base, &ring_sigma, true);
    common.error_cutoff = common.error_cutoff.max(ceil_sigma_multiple(&ring_sigma, 16));
    let lwe_sigma = env::var("FHE_TEST_TFHE_LWE_SIGMA").unwrap_or_else(|_| "131072".into());
    let lwe_error_sigma: f64 = lwe_sigma.parse().expect("finite positive LWE sigma");
    assert!(lwe_error_sigma.is_finite() && lwe_error_sigma > 0.0);
    let lwe_error_cutoff = env::var("FHE_TEST_TFHE_LWE_ERROR_CUTOFF")
        .map(|value| value.parse().expect("integer LWE error cutoff"))
        .unwrap_or_else(|_| ceil_sigma_multiple(&lwe_sigma, 16));
    let lwe_dimension = integer("FHE_TEST_TFHE_LWE_DIMENSION", 630);
    let lwe_modulus = env::var("FHE_TEST_TFHE_LWE_MODULUS")
        .map(|value| value.parse().expect("power-of-two LWE modulus"))
        .unwrap_or_else(|_| BigUint::one() << 32usize);
    let key_switch_base_bits = integer("FHE_TEST_TFHE_KS_BASE_BITS", 2);
    let key_switch_digits = integer("FHE_TEST_TFHE_KS_DIGITS", 8);
    TfheParams::new(
        common,
        lwe_dimension,
        lwe_modulus,
        lwe_error_sigma,
        lwe_error_cutoff,
        key_switch_base_bits,
        key_switch_digits,
    )
    .expect("valid TFHE parameters")
}

#[cfg(feature = "gpu")]
fn ceil_sigma_multiple(sigma: &str, multiple: u64) -> BigUint {
    let (mantissa, exponent) = sigma
        .find(['e', 'E'])
        .map(|index| {
            let (mantissa, exponent) = sigma.split_at(index);
            (mantissa, exponent[1..].parse::<i64>().expect("decimal sigma exponent"))
        })
        .unwrap_or((sigma, 0));
    let mantissa = mantissa.strip_prefix('+').unwrap_or(mantissa);
    assert!(!mantissa.starts_with('-'), "sigma must be nonnegative");
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    assert!(whole.bytes().all(|byte| byte.is_ascii_digit()));
    assert!(fraction.bytes().all(|byte| byte.is_ascii_digit()));
    let digits = format!("{whole}{fraction}");
    let mut numerator = digits.parse::<BigUint>().expect("decimal sigma digits");
    numerator *= multiple;
    let scale = i64::try_from(fraction.len()).expect("sigma precision fits i64") - exponent;
    if scale <= 0 {
        let power = u32::try_from(-scale).expect("sigma exponent fits u32");
        numerator * BigUint::from(10u8).pow(power)
    } else {
        let power = u32::try_from(scale).expect("sigma precision fits u32");
        let denominator = BigUint::from(10u8).pow(power);
        let quotient = &numerator / &denominator;
        if &quotient * denominator < numerator { quotient + BigUint::one() } else { quotient }
    }
}

#[cfg(feature = "gpu")]
pub fn modswitch_steps() -> usize {
    let steps = integer("FHE_TEST_MODSWITCH_STEPS", 1);
    assert!(steps > 0, "the round trip must include modulus switching");
    steps
}

/// GPU-only helpers shared by the FHE integration fixtures and GPU unit tests.
/// Keeping these helpers under the public `utils::gpu` namespace makes the
/// integration targets use the crate's single implementation without exposing
/// them as part of the scheme API.
#[cfg(feature = "gpu")]
pub mod gpu {
    use super::*;
    use crate::BgvParams;
    use mxx_backends::{GpuRuntime, RuntimeValue, poly::dcrt::gpu::GpuDCRTPolyParams};
    use std::collections::BTreeMap;

    pub fn bgv_gpu_parameters(bgv: &BgvParams) -> Vec<GpuDCRTPolyParams> {
        related_gpu_parameters(bgv.runtime_parameters().expect("BGV runtime parameters"))
    }

    pub fn tfhe_gpu_parameters(tfhe: &TfheParams) -> Vec<GpuDCRTPolyParams> {
        related_gpu_parameters(tfhe.runtime_parameters())
    }

    /// One GPU context per distinct ordered ring, all sharing the first ring's
    /// execution so a graph can mix their operations on one device.
    pub fn related_gpu_parameters(
        rings: impl IntoIterator<Item = DCRTPolyParams>,
    ) -> Vec<GpuDCRTPolyParams> {
        let rings = rings.into_iter().collect::<Vec<_>>();
        let mut parameters: Vec<GpuDCRTPolyParams> = Vec::with_capacity(rings.len());
        for ring in rings {
            let dimension = ring.ring_dimension();
            let crt_moduli = ring.to_crt().0;
            let parameter = if let Some(related) = parameters.first() {
                GpuDCRTPolyParams::new_with_gpu(
                    dimension,
                    crt_moduli,
                    ring.base_bits(),
                    related.gpu_ids().to_vec(),
                    Some(1),
                    Some(related),
                    None,
                )
            } else {
                GpuDCRTPolyParams::new(dimension, crt_moduli, ring.base_bits(), None)
            };
            if let Some(existing) = parameters.iter().find(|existing| {
                existing.ring_dimension() == dimension &&
                    existing.to_crt().0 == parameter.to_crt().0
            }) {
                assert_eq!(
                    existing, &parameter,
                    "BGV GPU parameters disagree for one exact ordered CRT ring"
                );
                continue;
            }
            parameters.push(parameter);
        }
        parameters
    }

    pub fn input(values: &[i64]) -> RuntimeValue {
        RuntimeValue::integer_values(values.iter().map(|v| BigInt::from(*v)).collect())
    }

    pub fn integers(
        runtime: &GpuRuntime,
        outputs: &BTreeMap<String, RuntimeValue>,
        name: &str,
    ) -> Vec<BigInt> {
        runtime
            .download_integer_family(&outputs[name])
            .unwrap_or_else(|error| panic!("download integer family {name}: {error}"))
    }
}

#[cfg(all(test, not(feature = "gpu")))]
use crate::FheCommonParams;
#[cfg(test)]
use mxx_backends::{
    ExecutionConfig, ExecutionResult, MemoryArtifactStore, RuntimeValue,
    backend::poly::cpu_backend, execute, transcript::SamplingMode,
};
#[cfg(test)]
use mxx_dsl::BuiltGraph;
#[cfg(test)]
use mxx_ir_core::ParamEnv;
#[cfg(all(test, not(feature = "gpu")))]
use mxx_ir_core::node::SampleRange;
#[cfg(test)]
use std::collections::BTreeMap;

// FHE_TEST_RING_DIMENSION/CRT_DEPTH/CRT_BITS/BASE_BITS control the toy ring;
// FHE_TEST_SIGMA/ERROR_CUTOFF control its bounded noise. Defaults are correctness
// fixtures only. Larger dimensions also require compatible SIMD plaintext moduli.
#[cfg(test)]
pub(crate) fn common() -> FheCommonParams {
    let read = |name: &str, default: usize| {
        std::env::var(name)
            .ok()
            .map(|value| value.parse().expect("integer test parameter"))
            .unwrap_or(default)
    };
    let sigma = std::env::var("FHE_TEST_SIGMA").unwrap_or_else(|_| "1.0".into());
    let sigma_bound: bigdecimal::BigDecimal = sigma.parse().expect("Gaussian sigma");
    let error_sigma: f64 = sigma.parse().expect("Gaussian sigma");
    let error_cutoff = std::env::var("FHE_TEST_ERROR_CUTOFF")
        .map(|value| value.parse::<BigUint>().expect("integer error cutoff"))
        .unwrap_or_else(|_| {
            mxx_backends::sampler::bounds::hard_cutoff_from_sigma_bound(&sigma_bound)
        });
    FheCommonParams {
        ring: DCRTPolyParams::new(
            read("FHE_TEST_RING_DIMENSION", 8) as u32,
            read("FHE_TEST_CRT_DEPTH", 3),
            read("FHE_TEST_CRT_BITS", 30),
            read("FHE_TEST_BASE_BITS", 4) as u32,
            None,
            None,
        ),
        secret_range: SampleRange { minimum: (-1).into(), maximum: 1.into() },
        error_sigma,
        error_cutoff,
    }
}

#[cfg(test)]
pub(crate) fn execute_graph(
    graph: BuiltGraph,
    common: &FheCommonParams,
    inputs: BTreeMap<String, RuntimeValue>,
    extra_parameters: &[DCRTPolyParams],
) -> ExecutionResult {
    // Register prefix rings for ciphertext levels and single-prime rings for
    // modswitch corrections; a modulus alone does not specify CRT tower order.
    let (moduli, _, depth) = common.ring.to_crt();
    let mut parameters =
        (0..depth).map(|level| common.parameters_at(level).unwrap()).collect::<Vec<_>>();
    parameters
        .extend(moduli.iter().map(|p| common.ring.select_modulus(&BigUint::from(*p)).unwrap()));
    let validated = graph
        .validate(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
        .expect("valid FHE DSL graph");
    parameters.extend_from_slice(extra_parameters);
    let mut backend = cpu_backend(parameters);
    let mut store = MemoryArtifactStore::default();
    let mut result = execute(
        &validated,
        &mut backend,
        inputs,
        &mut store,
        SamplingMode::Fresh,
        ExecutionConfig::default(),
    )
    .expect("FHE graph execution");
    // Structural families may be lazy staged outputs. Materialize them before
    // dropping the memory store so assertions observe actual runtime values.
    for name in result.outputs.keys().cloned().collect::<Vec<_>>() {
        result.materialize_output(&name, &backend, &mut store).expect("load FHE output");
    }
    result.cleanup_staged(&mut store).expect("clean streamed outputs");
    result
}

#[cfg(test)]
pub(crate) fn int_input(values: &[i64]) -> RuntimeValue {
    RuntimeValue::integer_values(values.iter().map(|v| BigInt::from(*v)).collect())
}

#[cfg(test)]
pub(crate) fn integers(result: &ExecutionResult, name: &str) -> Vec<BigInt> {
    let RuntimeValue::IndexedFamily { values, .. } = &result.outputs[name] else {
        panic!("expected integer family {name}")
    };
    values
        .iter()
        .map(|v| {
            let RuntimeValue::Int(value) = v else { panic!("expected integer") };
            value.clone()
        })
        .collect()
}
