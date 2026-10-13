//! The parameter selection of a layered Boolean BGG+ evaluation.
//!
//! For every ring dimension in increasing order, [`select_boolean_parameters`] finds, for every
//! gadget base, the smallest CRT depth whose simulated output error decodes
//! ([`super::BooleanBggNoiseReport::decrypts`]), keeps the base with the smallest depth (the
//! larger base on a tie, for fewer columns), and accepts the dimension when the Ring-LWE estimate
//! of the resulting modulus reaches the security target. The CRT basis is the generated basis of
//! [`mxx_ir_core::generate_crt_basis`], so a DSL ring `Ring::new(crt_bits, crt_depth, n)` has the
//! same modulus.

use super::{BooleanBggNoise, BooleanBggNoiseReport};
use num_bigint::BigUint;
use rayon::prelude::*;

/// The fixed inputs of the selection.
#[derive(Clone, Debug)]
pub struct BooleanBggParameterRequest {
    /// The number of layers of the evaluated circuit.
    pub depth: usize,
    pub secret_dimension: usize,
    pub crt_bits: usize,
    /// The candidate gadget bases, as base-2 logarithms.
    pub base_bits: Vec<usize>,
    /// The candidate ring dimensions, tried in increasing order.
    pub ring_dimensions: Vec<u32>,
    pub max_crt_depth: usize,
    pub fresh_error_bound: BigUint,
    pub security_bits: u64,
}

/// The selected parameters.
#[derive(Clone, Debug)]
pub struct BooleanBggSelection {
    pub crt_bits: usize,
    pub crt_moduli: Vec<u64>,
    pub base_bits: usize,
    pub noise: BooleanBggNoise,
    pub report: BooleanBggNoiseReport,
    pub security_bits: u64,
}

impl BooleanBggSelection {
    pub fn crt_depth(&self) -> usize {
        self.crt_moduli.len()
    }
}

/// The noise parameters of the first `crt_depth` moduli of `basis` with the base `2^base_bits`.
fn noise_at(
    request: &BooleanBggParameterRequest,
    ring_dimension: u32,
    basis: &[u64],
    crt_depth: usize,
    base_bits: usize,
) -> BooleanBggNoise {
    BooleanBggNoise {
        ring_dimension,
        modulus: basis[..crt_depth].iter().map(|&q| BigUint::from(q)).product(),
        secret_dimension: request.secret_dimension,
        digit_count: crt_depth * request.crt_bits.div_ceil(base_bits),
        gadget_base: BigUint::from(1u8) << base_bits,
        fresh_error_bound: request.fresh_error_bound.clone(),
    }
}

/// Selects the parameters of `request`, estimating the security of a ring dimension and modulus
/// with `security_bits`.
pub fn select_boolean_parameters(
    request: &BooleanBggParameterRequest,
    security_bits: impl Fn(u32, &BigUint) -> Result<u64, String>,
) -> Result<BooleanBggSelection, String> {
    let mut ring_dimensions = request.ring_dimensions.clone();
    ring_dimensions.sort_unstable();
    for ring_dimension in ring_dimensions {
        let basis = mxx_ir_core::generate_crt_basis(
            ring_dimension,
            request.max_crt_depth,
            request.crt_bits,
        )?;
        let candidate = request
            .base_bits
            .par_iter()
            .filter_map(|&base_bits| {
                (1..=request.max_crt_depth).find_map(|crt_depth| {
                    let noise = noise_at(request, ring_dimension, &basis, crt_depth, base_bits);
                    let report = noise.simulate(request.depth);
                    report.decrypts().then_some((crt_depth, base_bits, noise, report))
                })
            })
            .min_by_key(|(crt_depth, base_bits, _, _)| (*crt_depth, std::cmp::Reverse(*base_bits)));
        let Some((crt_depth, base_bits, noise, report)) = candidate else {
            continue;
        };
        let achieved = security_bits(ring_dimension, &noise.modulus)?;
        if achieved >= request.security_bits {
            return Ok(BooleanBggSelection {
                crt_bits: request.crt_bits,
                crt_moduli: basis[..crt_depth].to_vec(),
                base_bits,
                noise,
                report,
                security_bits: achieved,
            });
        }
    }
    Err(format!(
        "no ring dimension of {:?} decodes depth {} within CRT depth {} at {} security bits",
        request.ring_dimensions, request.depth, request.max_crt_depth, request.security_bits
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_takes_the_smallest_secure_dimension_and_decoding_depth() {
        let request = BooleanBggParameterRequest {
            depth: 3,
            secret_dimension: 2,
            crt_bits: 40,
            base_bits: vec![1, 2, 10],
            ring_dimensions: vec![64, 16, 32],
            max_crt_depth: 8,
            fresh_error_bound: BigUint::from(20u8),
            security_bits: 100,
        };
        // A stand-in estimate: only dimensions of at least 32 are secure.
        let selection = select_boolean_parameters(&request, |ring_dimension, _| {
            Ok(if ring_dimension >= 32 { 128 } else { 64 })
        })
        .unwrap();
        assert_eq!(selection.noise.ring_dimension, 32);
        assert!(selection.report.decrypts());
        let depth = selection.crt_depth();
        let basis = mxx_ir_core::generate_crt_basis(32, depth, 40).unwrap();
        assert_eq!(selection.crt_moduli, basis);
        if depth > 1 {
            assert!(
                !noise_at(&request, 32, &basis, depth - 1, selection.base_bits)
                    .simulate(3)
                    .decrypts()
            );
        }
    }
}
