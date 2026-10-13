//! Concrete Ring-LWE security estimates from the external `lattice-estimator-cli`.
//!
//! The estimator must be on `PATH` with its Sage environment. It prints the estimated security
//! in bits on its last nonempty line.

use bigdecimal::{BigDecimal, Zero};
use num_bigint::BigUint;
use std::process::Command;

/// One positive integer estimate for Ring-LWE of dimension `ring_dimension` modulo `modulus` with
/// a uniform ternary secret and a discrete Gaussian error of standard deviation `error_sigma`.
pub fn lattice_security_bits(
    ring_dimension: u32,
    modulus: &BigUint,
    error_sigma: &BigDecimal,
) -> Result<u64, String> {
    lattice_security_bits_with_secret(ring_dimension, modulus, r#"{"name":"Ternary"}"#, error_sigma)
}

/// [`lattice_security_bits`] for a discrete Gaussian secret of standard deviation
/// `secret_sigma`, such as a trapdoor's.
pub fn gaussian_secret_lattice_security_bits(
    ring_dimension: u32,
    modulus: &BigUint,
    secret_sigma: &BigDecimal,
    error_sigma: &BigDecimal,
) -> Result<u64, String> {
    if secret_sigma <= &BigDecimal::zero() {
        return Err("secret sigma must be positive".to_owned());
    }
    lattice_security_bits_with_secret(
        ring_dimension,
        modulus,
        &format!(r#"{{"name":"DiscreteGaussian","stddev":{secret_sigma}}}"#),
        error_sigma,
    )
}

/// One estimate for the secret distribution `secret` (an estimator JSON
/// specification) and a discrete Gaussian error.
fn lattice_security_bits_with_secret(
    ring_dimension: u32,
    modulus: &BigUint,
    secret: &str,
    error_sigma: &BigDecimal,
) -> Result<u64, String> {
    if ring_dimension == 0 || modulus.is_zero() {
        return Err("ring parameters must be positive".to_owned());
    }
    if error_sigma <= &BigDecimal::zero() {
        return Err("error sigma must be positive".to_owned());
    }
    let output = Command::new("lattice-estimator-cli")
        .arg(ring_dimension.to_string())
        .arg(modulus.to_string())
        .arg("--s-dist")
        .arg(secret)
        .arg("--e-dist")
        .arg(format!(r#"{{"name":"DiscreteGaussian","stddev":{error_sigma}}}"#))
        .output()
        .map_err(|error| format!("failed to start lattice-estimator-cli: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "lattice-estimator-cli failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let result = stdout
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .ok_or_else(|| "lattice-estimator-cli returned no security estimate".to_owned())?
        .trim();
    let bits = result.parse::<u64>().map_err(|error| {
        format!("lattice-estimator-cli returned a non-integer security estimate: {error}")
    })?;
    if bits == 0 {
        return Err("lattice-estimator-cli returned a non-positive security estimate".to_owned());
    }
    Ok(bits)
}
