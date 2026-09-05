use mxx_ir_core::types::ConcreteMatrixType;
use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::One;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

/// Which backend gadget layout a generated primitive relation is allowed to use.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum LeanGadgetMode {
    /// Balanced digits from every source CRT tower, copied to every output tower.
    Regular,
    /// Unsigned compact digits from the backend's source-limb layout.
    Small,
}

/// Concrete CRT and gadget data exported from one DCRT parameter set supplied to backend setup.
///
/// `crt_moduli` is the ordered basis returned by `DCRTPolyParams::to_crt`; it is never rebuilt
/// from the full modulus.  The two digit counts are kept separate because regular and compact
/// decomposition have different output layouts.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct LeanRingLayout {
    pub modulus: BigInt,
    pub ring_dimension: u32,
    pub crt_moduli: Vec<u64>,
    pub crt_bits: usize,
    pub crt_depth: usize,
    pub base_bits: u32,
    pub regular_digit_count: usize,
    pub small_digit_count: usize,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum LayoutError {
    #[error("invalid Lean namespace")]
    InvalidNamespace,
    #[error("CRT basis must contain at least one modulus")]
    EmptyCrtBasis,
    #[error("CRT modulus at position {0} must be greater than one")]
    InvalidCrtModulus(usize),
    #[error("CRT moduli are not pairwise coprime at positions {0} and {1}")]
    NonCoprimeCrtModuli(usize, usize),
    #[error("CRT basis product does not equal the registered full modulus")]
    ProductMismatch,
    #[error("CRT depth does not equal the ordered basis length")]
    DepthMismatch,
    #[error("CRT bit width must be positive")]
    InvalidCrtBits,
    #[error("CRT modulus at position {0} exceeds the regular digit capacity")]
    InsufficientDigitCapacity(usize),
    #[error("gadget base bit width must be positive")]
    InvalidBaseBits,
    #[error("ring dimension must be positive")]
    InvalidRingDimension,
    #[error("ring dimension must be a power of two")]
    NonPowerOfTwoRingDimension,
    #[error("regular digit count does not match the CRT basis and base width")]
    RegularDigitCountMismatch,
    #[error(
        "registered layouts disagree for modulus {modulus} and ring dimension {ring_dimension}"
    )]
    ConflictingRingLayout { modulus: BigInt, ring_dimension: u32 },
    #[error("matrix modulus does not match the supplied CRT context")]
    MatrixModulusMismatch,
    #[error("matrix ring dimension does not match the supplied CRT context")]
    MatrixRingDimensionMismatch,
    #[error("declared gadget base does not match the supplied CRT context")]
    GadgetBaseMismatch,
    #[error("declared gadget digit count does not match the selected layout mode")]
    GadgetDigitCountMismatch,
}

/// One generated context and the validated layouts used to generate it.
/// Fields are private so callers cannot disconnect the context from its layout metadata.
pub struct LeanBackendArtifact {
    source: String,
    module_name: String,
    context_name: String,
    layouts: Vec<LeanRingLayout>,
}

impl LeanBackendArtifact {
    pub fn source(&self) -> &str {
        &self.source
    }
    pub fn module_name(&self) -> &str {
        &self.module_name
    }
    pub fn context_name(&self) -> &str {
        &self.context_name
    }
    pub fn exporter_bindings(&self) -> Vec<mxx_ir_core::lean::BackendLayout> {
        self.layouts.iter().map(LeanRingLayout::exporter_binding).collect()
    }
}

/// Emit one fixed backend context from concrete setup data. All arithmetic layout obligations
/// are checked by Lean; the generated code contains no primitive interpretation callbacks.
pub fn render_backend_context(
    layouts: &[LeanRingLayout],
    module_name: &str,
    namespace: &str,
) -> Result<LeanBackendArtifact, LayoutError> {
    if ![module_name, namespace].iter().all(|name| {
        !name.is_empty() &&
            name.split('.').all(|part| {
                !part.is_empty() &&
                    part.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) &&
                    part.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            })
    }) {
        return Err(LayoutError::InvalidNamespace);
    }
    let mut layouts =
        layouts.iter().cloned().map(LeanRingLayout::validate).collect::<Result<Vec<_>, _>>()?;
    layouts.sort();
    for pair in layouts.windows(2) {
        if pair[0].modulus == pair[1].modulus &&
            pair[0].ring_dimension == pair[1].ring_dimension &&
            pair[0] != pair[1]
        {
            return Err(LayoutError::ConflictingRingLayout {
                modulus: pair[0].modulus.clone(),
                ring_dimension: pair[0].ring_dimension,
            });
        }
    }
    layouts.dedup();
    let mut source =
        format!("import MxxRuntime\n\nnamespace {namespace}\n\nnoncomputable section\n\n");
    for (index, layout) in layouts.iter().enumerate() {
        let moduli = layout.crt_moduli.iter().map(u64::to_string).collect::<Vec<_>>().join(", ");
        let base = BigUint::one() << layout.base_bits;
        source.push_str(&format!(
            r#"def moduli{index} : List Nat := [{moduli}]
def layout{index} : MxxRuntime.RegularLayout {q} :=
  {{ crtModuli := moduli{index}
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := {bits}
    base := {base}
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := {digits}
    digits_pos := by norm_num
    capacity := by decide }}

"#,
            q = layout.modulus,
            bits = layout.base_bits,
            digits = layout.small_digit_count
        ));
    }
    source.push_str("def backend : MxxRuntime.BackendContext where\n  regularLayout q n :=\n");
    for (index, layout) in layouts.iter().enumerate() {
        source.push_str(&format!(
            "    if h : q = {} ∧ n = {} then\n      some (h.1.symm ▸ layout{index})\n    else\n",
            layout.modulus, layout.ring_dimension
        ));
    }
    source.push_str(&format!("    none\n\nend\nend {namespace}\n"));
    Ok(LeanBackendArtifact {
        source,
        module_name: module_name.into(),
        context_name: format!("{namespace}.backend"),
        layouts,
    })
}

impl LeanRingLayout {
    pub fn exporter_binding(&self) -> mxx_ir_core::lean::BackendLayout {
        mxx_ir_core::lean::BackendLayout {
            modulus: self.modulus.clone(),
            ring_dimension: self.ring_dimension as usize,
            base: BigInt::one() << self.base_bits,
            regular_digits: self.regular_digit_count,
        }
    }
    /// Export and validate the layout from the exact parameters supplied to backend setup.
    pub fn from_dcrt(parameters: &DCRTPolyParams) -> Result<Self, LayoutError> {
        let modulus = parameters.modulus();
        let (crt_moduli, crt_bits, crt_depth) = parameters.to_crt();
        let layout = Self {
            modulus: BigInt::from_biguint(num_bigint::Sign::Plus, modulus.as_ref().clone()),
            ring_dimension: parameters.ring_dimension(),
            crt_moduli,
            crt_bits,
            crt_depth,
            base_bits: parameters.base_bits(),
            regular_digit_count: parameters.modulus_digits(),
            small_digit_count: 0,
        };
        layout.validate()
    }

    /// Validate a layout and return it with the compact digit count derived from its basis.
    pub fn validate(mut self) -> Result<Self, LayoutError> {
        if self.crt_moduli.is_empty() {
            return Err(LayoutError::EmptyCrtBasis);
        }
        if self.crt_depth != self.crt_moduli.len() {
            return Err(LayoutError::DepthMismatch);
        }
        if self.crt_bits == 0 {
            return Err(LayoutError::InvalidCrtBits);
        }
        if self.base_bits == 0 {
            return Err(LayoutError::InvalidBaseBits);
        }
        if self.ring_dimension == 0 {
            return Err(LayoutError::InvalidRingDimension);
        }
        if !self.ring_dimension.is_power_of_two() {
            return Err(LayoutError::NonPowerOfTwoRingDimension);
        }

        let mut product = BigUint::one();
        for (index, &modulus) in self.crt_moduli.iter().enumerate() {
            if modulus <= 1 {
                return Err(LayoutError::InvalidCrtModulus(index));
            }
            for (other_index, &other) in self.crt_moduli[..index].iter().enumerate() {
                if modulus.gcd(&other) != 1 {
                    return Err(LayoutError::NonCoprimeCrtModuli(other_index, index));
                }
            }
            product *= modulus;
        }
        let expected = self.modulus.to_biguint().ok_or(LayoutError::ProductMismatch)?;
        if product != expected {
            return Err(LayoutError::ProductMismatch);
        }

        let digits_per_tower = self.crt_bits.div_ceil(self.base_bits as usize);
        let capacity_bits = digits_per_tower
            .checked_mul(self.base_bits as usize)
            .ok_or(LayoutError::RegularDigitCountMismatch)?;
        // Every tower is a u64. Avoid constructing an unnecessarily huge power of two.
        if capacity_bits < 64 {
            for (index, &modulus) in self.crt_moduli.iter().enumerate() {
                if modulus > (1u64 << capacity_bits) {
                    return Err(LayoutError::InsufficientDigitCapacity(index));
                }
            }
        }
        let regular_digits = digits_per_tower
            .checked_mul(self.crt_depth)
            .ok_or(LayoutError::RegularDigitCountMismatch)?;
        if self.regular_digit_count != regular_digits {
            return Err(LayoutError::RegularDigitCountMismatch);
        }
        self.small_digit_count = digits_per_tower;
        Ok(self)
    }

    pub fn digit_count(&self, mode: LeanGadgetMode) -> usize {
        match mode {
            LeanGadgetMode::Regular => self.regular_digit_count,
            LeanGadgetMode::Small => self.small_digit_count,
        }
    }

    /// Check one concrete IR matrix and node payload against this backend context.
    pub fn validate_request(
        &self,
        matrix: &ConcreteMatrixType,
        base: &BigInt,
        digit_count: usize,
        mode: LeanGadgetMode,
    ) -> Result<(), LayoutError> {
        if matrix.modulus != self.modulus {
            return Err(LayoutError::MatrixModulusMismatch);
        }
        if matrix.ring_dimension != self.ring_dimension as usize {
            return Err(LayoutError::MatrixRingDimensionMismatch);
        }
        let expected_base = BigInt::from(1u8) << self.base_bits as usize;
        if *base != expected_base {
            return Err(LayoutError::GadgetBaseMismatch);
        }
        if digit_count != self.digit_count(mode) {
            return Err(LayoutError::GadgetDigitCountMismatch);
        }
        Ok(())
    }
}

/// Export a deterministic, key-sorted layout list from concrete backend-setup parameters.
pub fn export_dcrt_layouts<'a>(
    parameters: impl IntoIterator<Item = &'a DCRTPolyParams>,
) -> Result<Vec<LeanRingLayout>, LayoutError> {
    let mut layouts =
        parameters.into_iter().map(LeanRingLayout::from_dcrt).collect::<Result<Vec<_>, _>>()?;
    layouts.sort();
    for pair in layouts.windows(2) {
        if pair[0].modulus == pair[1].modulus &&
            pair[0].ring_dimension == pair[1].ring_dimension &&
            pair[0] != pair[1]
        {
            return Err(LayoutError::ConflictingRingLayout {
                modulus: pair[0].modulus.clone(),
                ring_dimension: pair[0].ring_dimension,
            });
        }
    }
    let mut seen = BTreeSet::new();
    layouts.retain(|layout| seen.insert((layout.modulus.clone(), layout.ring_dimension)));
    Ok(layouts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(moduli: Vec<u64>, crt_bits: usize, base_bits: u32, regular: usize) -> LeanRingLayout {
        let modulus = moduli.iter().fold(BigUint::one(), |value, &part| value * part);
        LeanRingLayout {
            modulus: BigInt::from_biguint(num_bigint::Sign::Plus, modulus),
            ring_dimension: 8,
            crt_moduli: moduli.clone(),
            crt_bits,
            crt_depth: moduli.len(),
            base_bits,
            regular_digit_count: regular,
            small_digit_count: 0,
        }
    }

    #[test]
    fn validates_registered_ordered_basis_and_derives_modes() {
        let validated = layout(vec![17, 19], 8, 4, 4).validate().unwrap();
        assert_eq!(validated.digit_count(LeanGadgetMode::Regular), 4);
        assert_eq!(validated.digit_count(LeanGadgetMode::Small), 2);
    }

    #[test]
    fn rejects_non_coprime_basis() {
        assert_eq!(
            layout(vec![15, 21], 8, 4, 4).validate(),
            Err(LayoutError::NonCoprimeCrtModuli(0, 1))
        );
    }

    #[test]
    fn rejects_insufficient_regular_capacity() {
        assert_eq!(
            layout(vec![17], 4, 4, 1).validate(),
            Err(LayoutError::InsufficientDigitCapacity(0))
        );
    }

    #[test]
    fn backend_context_preserves_order_and_rejects_conflicts() {
        let first = layout(vec![17, 19], 8, 4, 4).validate().unwrap();
        let reversed = layout(vec![19, 17], 8, 4, 4).validate().unwrap();
        let source =
            render_backend_context(&[first.clone(), first.clone()], "Fixture", "Fixture.Backend")
                .unwrap();
        let source = source.source();
        assert_eq!(source.matches("def layout0").count(), 1);
        assert!(!source.contains("def layout1"));
        assert!(source.contains("[17, 19]"));
        assert!(matches!(
            render_backend_context(&[first, reversed], "Fixture", "Fixture"),
            Err(LayoutError::ConflictingRingLayout { .. })
        ));
    }

    #[test]
    fn validates_concrete_ir_request_for_each_mode() {
        let context = layout(vec![17, 19], 8, 4, 4).validate().unwrap();
        let matrix =
            ConcreteMatrixType::scalar(context.modulus.clone(), context.ring_dimension as usize);
        assert!(context.validate_request(&matrix, &16.into(), 4, LeanGadgetMode::Regular).is_ok());
        assert!(context.validate_request(&matrix, &16.into(), 2, LeanGadgetMode::Small).is_ok());
        assert_eq!(
            context.validate_request(&matrix, &16.into(), 3, LeanGadgetMode::Small),
            Err(LayoutError::GadgetDigitCountMismatch)
        );
    }
}
