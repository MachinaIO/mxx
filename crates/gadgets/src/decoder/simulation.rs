//! Shared mask-bit search and decode-margin helpers.

use bigdecimal::BigDecimal;
use num_bigint::{BigInt, BigUint, Sign};

/// Decode threshold before subtracting the centered mask range.
///
/// A decoder that rounds to a plaintext modulus `p` has half-cell margin
/// `q / (2 * p)` before the centered mask and accumulated error are applied.
/// Boolean high-bit decode is the special case `p = 2`, giving `q / 4`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeThreshold {
    plaintext_modulus: BigUint,
}

impl DecodeThreshold {
    pub fn new(plaintext_modulus: impl Into<BigUint>) -> Self {
        let plaintext_modulus = plaintext_modulus.into();
        assert!(plaintext_modulus > BigUint::from(1u32), "plaintext modulus must be at least two");
        Self { plaintext_modulus }
    }

    pub fn boolean() -> Self {
        Self::new(2u32)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaskBitSearchResult<S> {
    pub mask_bits: usize,
    pub simulation: S,
}

pub fn biguint_to_decimal(value: &BigUint) -> BigDecimal {
    BigDecimal::from(BigInt::from(value.clone()))
}

pub fn centered_mask_magnitude(bits: usize) -> BigDecimal {
    assert!(bits > 0, "mask bit size must be positive");
    biguint_to_decimal(&(BigUint::from(1u32) << (bits - 1)))
}

pub fn decode_threshold(full_modulus: &BigUint, threshold: &DecodeThreshold) -> BigDecimal {
    biguint_to_decimal(full_modulus) /
        (BigDecimal::from(2u32) * biguint_to_decimal(&threshold.plaintext_modulus))
}

pub fn decode_margin(
    full_modulus: &BigUint,
    mask_bits: usize,
    threshold: &DecodeThreshold,
) -> Option<BigDecimal> {
    let threshold = decode_threshold(full_modulus, threshold);
    let mask = centered_mask_magnitude(mask_bits);
    (threshold > mask).then_some(threshold - mask)
}

pub fn max_safe_mask_bits(
    full_modulus: &BigUint,
    pre_rounding_error: &BigDecimal,
    max_bits: usize,
    threshold: &DecodeThreshold,
) -> Option<usize> {
    let threshold = decode_threshold(full_modulus, threshold);
    let available = threshold - pre_rounding_error;
    max_centered_mask_bits_strictly_below(&available, max_bits)
}

/// Chooses the largest bit-decomposed centered-mask width whose range does not
/// exceed an already-reserved margin.
///
/// If `available` is the range left after reserving correctness or security
/// error, this returns the largest `k <= max_bits` such that
/// `2^(k - 1) <= available`. This is the implementation form of choosing
/// `k = 1 + floor(log2(available))`, capped by `max_bits`.
pub fn max_centered_mask_bits_for_available_range(
    available: &BigDecimal,
    max_bits: usize,
) -> Option<usize> {
    max_centered_mask_bits_at_most(available, max_bits)
}

/// Checks whether an error bound has the requested security margin below a threshold.
pub fn validate_error_bound_security_margin(
    threshold: &BigDecimal,
    error_bound: &BigDecimal,
    security_bit: usize,
) -> bool {
    let security_factor = BigDecimal::from(BigInt::from(BigUint::from(1u32) << security_bit));
    threshold >= &(error_bound * security_factor)
}

pub fn search_mask_bits_with_simulation<S, F, G>(
    full_modulus: &BigUint,
    max_bits: usize,
    threshold: &DecodeThreshold,
    mut simulate: F,
    mut error_bound: G,
) -> Option<MaskBitSearchResult<S>>
where
    S: Clone,
    F: FnMut(usize) -> S,
    G: FnMut(&S) -> BigDecimal,
{
    let zero = BigDecimal::from(0u32);
    let max_candidate = max_mask_bits_after_error_margin(full_modulus, &zero, max_bits, threshold)?;
    let mut low = 1usize;
    let mut high = max_candidate;
    let mut best = None;
    while low <= high {
        let candidate = low + (high - low) / 2;
        let simulation = simulate(candidate);
        let error = error_bound(&simulation);
        let max_valid_bits =
            max_mask_bits_after_error_margin(full_modulus, &error, max_bits, threshold);
        let valid = max_valid_bits.is_some_and(|max_valid| candidate <= max_valid);
        if valid {
            best = Some(MaskBitSearchResult { mask_bits: candidate, simulation });
            low = candidate + 1;
        } else if candidate == 1 {
            break;
        } else {
            high = max_valid_bits.unwrap_or(0).min(candidate - 1);
        }
    }
    best
}

fn max_mask_bits_after_error_margin(
    full_modulus: &BigUint,
    error_margin: &BigDecimal,
    max_bits: usize,
    threshold: &DecodeThreshold,
) -> Option<usize> {
    let available = decode_threshold(full_modulus, threshold) - error_margin;
    max_centered_mask_bits_for_available_range(&available, max_bits)
}

fn max_centered_mask_bits_strictly_below(available: &BigDecimal, max_bits: usize) -> Option<usize> {
    if available <= &BigDecimal::from(1u32) {
        return None;
    }
    let mut bound = floor_positive_decimal(available).to_biguint()?;
    if is_integer_decimal(available) {
        bound -= 1u32;
    }
    if bound < BigUint::from(2u32) {
        return None;
    }
    let max_exponent = (bound.bits() as usize).saturating_sub(1).min(max_bits);
    Some((max_exponent + 1).min(max_bits))
}

fn max_centered_mask_bits_at_most(available: &BigDecimal, max_bits: usize) -> Option<usize> {
    if available < &BigDecimal::from(1u32) {
        return None;
    }
    let bound = floor_positive_decimal(available).to_biguint()?;
    if bound < BigUint::from(1u32) {
        return None;
    }
    let max_exponent = (bound.bits() as usize).saturating_sub(1).min(max_bits);
    Some((max_exponent + 1).min(max_bits))
}

fn floor_positive_decimal(value: &BigDecimal) -> BigInt {
    let (digits, scale) = value.as_bigint_and_exponent();
    if scale <= 0 {
        return digits * BigInt::from(10u32).pow((-scale) as u32);
    }
    let divisor = BigInt::from(10u32).pow(scale as u32);
    let quotient = &digits / &divisor;
    let remainder = &digits % &divisor;
    if digits.sign() == Sign::Minus && remainder != BigInt::from(0u32) {
        quotient - 1u32
    } else {
        quotient
    }
}

fn is_integer_decimal(value: &BigDecimal) -> bool {
    let (digits, scale) = value.as_bigint_and_exponent();
    scale <= 0 || digits % BigInt::from(10u32).pow(scale as u32) == BigInt::from(0u32)
}

#[cfg(test)]
mod tests {
    use bigdecimal::BigDecimal;
    use num_bigint::BigUint;

    use super::{
        DecodeThreshold, centered_mask_magnitude, decode_margin,
        max_centered_mask_bits_for_available_range, max_safe_mask_bits,
        search_mask_bits_with_simulation,
    };

    #[test]
    fn centered_mask_magnitude_uses_top_centered_bit() {
        assert_eq!(centered_mask_magnitude(1), BigDecimal::from(1u32));
        assert_eq!(centered_mask_magnitude(8), BigDecimal::from(128u32));
    }

    #[test]
    fn decode_margin_subtracts_centered_mask_from_quarter_q() {
        let q = BigUint::from(1024u32);
        assert_eq!(
            decode_margin(&q, 8, &DecodeThreshold::boolean()),
            Some(BigDecimal::from(128u32))
        );
    }

    #[test]
    fn max_safe_mask_bits_matches_noise_refresh_threshold_shape() {
        let q = BigUint::from(1024u32);
        let q_max = BigUint::from(8u32);
        let max_bits =
            max_safe_mask_bits(&q, &BigDecimal::from(0u32), 32, &DecodeThreshold::new(q_max))
                .expect("mask bits should fit");
        assert_eq!(max_bits, 6);
    }

    #[test]
    fn max_centered_mask_bits_selects_largest_power_below_available_range() {
        assert_eq!(
            max_centered_mask_bits_for_available_range(&BigDecimal::from(129u32), 32),
            Some(8)
        );
        assert_eq!(
            max_centered_mask_bits_for_available_range(&BigDecimal::from(128u32), 32),
            Some(8)
        );
        assert_eq!(
            max_centered_mask_bits_for_available_range(&BigDecimal::from(1024u32), 5),
            Some(5)
        );
    }

    #[test]
    fn search_mask_bits_uses_simulated_error_bound() {
        let q = BigUint::from(1024u32);
        let result = search_mask_bits_with_simulation(
            &q,
            32,
            &DecodeThreshold::boolean(),
            |bits| bits,
            |bits| BigDecimal::from(*bits as u32),
        )
        .expect("search should find a candidate");
        assert_eq!(result.mask_bits, 8);
    }

    #[test]
    fn search_mask_bits_accepts_exact_power_of_two_available_range() {
        let q = BigUint::from(1024u32);
        let result = search_mask_bits_with_simulation(
            &q,
            32,
            &DecodeThreshold::boolean(),
            |bits| bits,
            |_bits| BigDecimal::from(128u32),
        )
        .expect("search should find an exact-boundary candidate");
        assert_eq!(result.mask_bits, 8);
    }
}
