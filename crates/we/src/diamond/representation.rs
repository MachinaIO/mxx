//! Canonical runtime representation of a Diamond DCRT parameter set.
//!
//! This is a deployment-boundary record.  It deliberately does not claim that a
//! backend implementation is a refinement of the semantic Diamond model.

use super::parameter_search::DiamondSelectedParameters;
use mxx_primitives::poly::PolyParams;
use num_bigint::BigUint;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DcrtRuntimeRepresentation {
    ring_dimension: u32,
    /// The exact ordered CRT basis supplied by the backend parameter object.
    moduli: Vec<u64>,
    base_bits: u32,
    /// Product of `moduli`, derived at construction time.
    modulus: BigUint,
    /// Number of CRT towers, derived from `moduli`.
    depth: usize,
    /// Actual bit width of each ordered CRT modulus.
    actual_modulus_bits: Vec<usize>,
    /// Number of decomposition digits implied by the actual tower widths.
    digit_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum DcrtRuntimeRepresentationError {
    #[error("the DCRT runtime representation has no CRT moduli")]
    EmptyBasis,
    #[error("the DCRT runtime representation has a zero CRT modulus")]
    ZeroModulus,
    #[error("the DCRT decomposition base bits must be non-zero")]
    ZeroBaseBits,
    #[error("the DCRT runtime representation is invalid: {0}")]
    Invalid(String),
}

impl DcrtRuntimeRepresentation {
    pub fn ring_dimension(&self) -> u32 {
        self.ring_dimension
    }
    pub fn moduli(&self) -> &[u64] {
        &self.moduli
    }
    pub fn base_bits(&self) -> u32 {
        self.base_bits
    }
    pub fn modulus(&self) -> &BigUint {
        &self.modulus
    }
    pub fn depth(&self) -> usize {
        self.depth
    }
    pub fn actual_modulus_bits(&self) -> &[usize] {
        &self.actual_modulus_bits
    }
    pub fn digit_count(&self) -> usize {
        self.digit_count
    }

    /// Capture the canonical runtime representation exposed by a typed parameter object.
    pub fn from_params<P: PolyParams>(
        parameters: &P,
    ) -> Result<Self, DcrtRuntimeRepresentationError> {
        let (moduli, _, reported_depth) = parameters.to_crt();
        if moduli.is_empty() {
            return Err(DcrtRuntimeRepresentationError::EmptyBasis);
        }
        if parameters.base_bits() == 0 {
            return Err(DcrtRuntimeRepresentationError::ZeroBaseBits);
        }
        if moduli.iter().any(|modulus| *modulus == 0) {
            return Err(DcrtRuntimeRepresentationError::ZeroModulus);
        }
        if reported_depth != moduli.len() {
            return Err(DcrtRuntimeRepresentationError::Invalid(format!(
                "parameter-reported CRT depth {reported_depth} does not match basis length {}",
                moduli.len()
            )));
        }

        let actual_modulus_bits = moduli
            .iter()
            .map(|modulus| (u64::BITS - modulus.leading_zeros()) as usize)
            .collect::<Vec<_>>();
        let digit_count = actual_modulus_bits
            .iter()
            .try_fold(0usize, |count, bits| {
                count.checked_add(bits.div_ceil(parameters.base_bits() as usize))
            })
            .ok_or_else(|| {
                DcrtRuntimeRepresentationError::Invalid(
                    "decomposition digit count overflowed".to_owned(),
                )
            })?;
        let modulus = moduli
            .iter()
            .fold(BigUint::from(1u8), |product, modulus| product * BigUint::from(*modulus));
        let representation = Self {
            ring_dimension: parameters.ring_dimension(),
            moduli,
            base_bits: parameters.base_bits(),
            modulus,
            depth: reported_depth,
            actual_modulus_bits,
            digit_count,
        };
        representation.validate_canonical()?;
        Ok(representation)
    }

    /// Alias naming the source as a parameter object rather than a generic typed parameter.
    pub fn from_parameters<P: PolyParams>(
        parameters: &P,
    ) -> Result<Self, DcrtRuntimeRepresentationError> {
        Self::from_params(parameters)
    }

    /// Capture the runtime representation and verify all semantic candidate bookkeeping fields.
    pub fn from_selected(
        selected: &DiamondSelectedParameters,
    ) -> Result<Self, DcrtRuntimeRepresentationError> {
        let representation = Self::from_params(&selected.parameters)?;
        representation.validate_selected_fields(
            selected.crt_depth,
            selected.ring_dimension,
            &selected.modulus,
            selected.modulus_bits,
            &selected.compiler.config.modulus,
            selected.compiler.config.ring_dimension,
            selected.compiler.config.digit_count,
            &selected.compiler.config.gadget_base,
        )?;
        if representation.digit_count != selected.parameters.modulus_digits() {
            return Err(DcrtRuntimeRepresentationError::Invalid(
                "derived digit count does not match parameter object".to_owned(),
            ));
        }
        Ok(representation)
    }

    pub fn product_modulus(&self) -> &BigUint {
        &self.modulus
    }

    /// Validate that all derived fields still describe the canonical representation fields.
    pub fn validate_canonical(&self) -> Result<(), DcrtRuntimeRepresentationError> {
        if self.ring_dimension == 0 || !self.ring_dimension.is_power_of_two() {
            return Err(DcrtRuntimeRepresentationError::Invalid(
                "ring dimension must be a non-zero power of two".to_owned(),
            ));
        }
        if self.moduli.is_empty() {
            return Err(DcrtRuntimeRepresentationError::EmptyBasis);
        }
        if self.base_bits == 0 {
            return Err(DcrtRuntimeRepresentationError::ZeroBaseBits);
        }
        if self.moduli.iter().any(|modulus| *modulus == 0) {
            return Err(DcrtRuntimeRepresentationError::ZeroModulus);
        }
        if self.depth != self.moduli.len() {
            return Err(DcrtRuntimeRepresentationError::Invalid(
                "derived CRT depth does not match basis length".to_owned(),
            ));
        }
        let actual_modulus_bits = self
            .moduli
            .iter()
            .map(|modulus| (u64::BITS - modulus.leading_zeros()) as usize)
            .collect::<Vec<_>>();
        let digit_count = actual_modulus_bits
            .iter()
            .try_fold(0usize, |count, bits| {
                count.checked_add(bits.div_ceil(self.base_bits as usize))
            })
            .ok_or_else(|| {
                DcrtRuntimeRepresentationError::Invalid(
                    "decomposition digit count overflowed".to_owned(),
                )
            })?;
        let modulus = self
            .moduli
            .iter()
            .fold(BigUint::from(1u8), |product, modulus| product * BigUint::from(*modulus));
        if self.actual_modulus_bits != actual_modulus_bits {
            return Err(DcrtRuntimeRepresentationError::Invalid(
                "actual CRT modulus widths are not canonical".to_owned(),
            ));
        }
        if self.modulus != modulus {
            return Err(DcrtRuntimeRepresentationError::Invalid(
                "product modulus is not canonical".to_owned(),
            ));
        }
        if self.digit_count != digit_count {
            return Err(DcrtRuntimeRepresentationError::Invalid(
                "derived digit count is not canonical".to_owned(),
            ));
        }
        Ok(())
    }

    /// Validate every representation field against the parameter object, including CRT order.
    pub fn validate_against_params<P: PolyParams>(
        &self,
        parameters: &P,
    ) -> Result<(), DcrtRuntimeRepresentationError> {
        self.validate_canonical()?;
        let expected = Self::from_params(parameters)?;
        if self != &expected {
            return Err(DcrtRuntimeRepresentationError::Invalid(representation_mismatch(
                self, &expected,
            )));
        }
        Ok(())
    }

    pub fn validate_against_parameters<P: PolyParams>(
        &self,
        parameters: &P,
    ) -> Result<(), DcrtRuntimeRepresentationError> {
        self.validate_against_params(parameters)
    }

    pub(crate) fn validate_selected_fields(
        &self,
        crt_depth: usize,
        ring_dimension: u32,
        modulus: &BigUint,
        modulus_bits: usize,
        compiler_modulus: &num_bigint::BigInt,
        compiler_ring_dimension: usize,
        compiler_digit_count: usize,
        compiler_gadget_base: &num_bigint::BigInt,
    ) -> Result<(), DcrtRuntimeRepresentationError> {
        if self.depth != crt_depth {
            return Err(DcrtRuntimeRepresentationError::Invalid(format!(
                "CRT depth {} does not match semantic candidate {crt_depth}",
                self.depth
            )));
        }
        if self.ring_dimension != ring_dimension ||
            self.ring_dimension as usize != compiler_ring_dimension
        {
            return Err(DcrtRuntimeRepresentationError::Invalid(
                "ring dimension does not match semantic candidate".to_owned(),
            ));
        }
        if &self.modulus != modulus ||
            compiler_modulus != &num_bigint::BigInt::from(self.modulus.clone())
        {
            return Err(DcrtRuntimeRepresentationError::Invalid(
                "product modulus does not match semantic candidate".to_owned(),
            ));
        }
        if self.modulus.bits() as usize != modulus_bits {
            return Err(DcrtRuntimeRepresentationError::Invalid(
                "product modulus bit width does not match semantic candidate".to_owned(),
            ));
        }
        let expected_base =
            num_bigint::BigInt::from(1u64.checked_shl(self.base_bits).ok_or_else(|| {
                DcrtRuntimeRepresentationError::Invalid(
                    "base bits cannot be represented as a u64 gadget base".to_owned(),
                )
            })?);
        if compiler_gadget_base != &expected_base {
            return Err(DcrtRuntimeRepresentationError::Invalid(
                "gadget base does not match runtime base bits".to_owned(),
            ));
        }
        if self.digit_count != compiler_digit_count {
            return Err(DcrtRuntimeRepresentationError::Invalid(format!(
                "derived digit count {} does not match semantic candidate {compiler_digit_count}",
                self.digit_count
            )));
        }
        Ok(())
    }
}

fn representation_mismatch(
    actual: &DcrtRuntimeRepresentation,
    expected: &DcrtRuntimeRepresentation,
) -> String {
    if actual.ring_dimension != expected.ring_dimension {
        return "ring dimension does not match parameter object".to_owned();
    }
    if actual.moduli != expected.moduli {
        return "ordered CRT basis does not match parameter object".to_owned();
    }
    if actual.base_bits != expected.base_bits {
        return "base bits do not match parameter object".to_owned();
    }
    if actual.modulus != expected.modulus {
        return "product modulus does not match parameter object".to_owned();
    }
    if actual.actual_modulus_bits != expected.actual_modulus_bits {
        return "actual CRT modulus widths do not match parameter object".to_owned();
    }
    if actual.digit_count != expected.digit_count {
        return "derived digit count does not match parameter object".to_owned();
    }
    "derived CRT representation does not match parameter object".to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_primitives::poly::dcrt::params::DCRTPolyParams;

    fn representation() -> (DCRTPolyParams, DcrtRuntimeRepresentation) {
        let parameters = DCRTPolyParams::new(8, 2, 20, 4);
        let representation = DcrtRuntimeRepresentation::from_params(&parameters).unwrap();
        (parameters, representation)
    }

    #[test]
    fn captures_ordered_basis_and_derived_values() {
        let (parameters, representation) = representation();
        let (moduli, _, _) = parameters.to_crt();
        assert_eq!(representation.moduli, moduli);
        assert_eq!(representation.depth, 2);
        assert_eq!(representation.modulus, parameters.modulus().as_ref().clone());
        assert_eq!(representation.digit_count, parameters.modulus_digits());
    }

    #[test]
    fn reordered_basis_is_rejected() {
        let (parameters, mut representation) = representation();
        representation.moduli.swap(0, 1);
        assert!(representation.validate_against_params(&parameters).is_err());
    }

    #[test]
    fn changed_modulus_is_rejected() {
        let (parameters, mut representation) = representation();
        representation.modulus += BigUint::from(1u8);
        assert!(representation.validate_against_params(&parameters).is_err());
    }

    #[test]
    fn changed_base_is_rejected() {
        let (parameters, mut representation) = representation();
        representation.base_bits += 1;
        assert!(representation.validate_against_params(&parameters).is_err());
    }

    #[test]
    fn changed_ring_dimension_is_rejected() {
        let (parameters, mut representation) = representation();
        representation.ring_dimension *= 2;
        assert!(representation.validate_against_params(&parameters).is_err());
    }
}
