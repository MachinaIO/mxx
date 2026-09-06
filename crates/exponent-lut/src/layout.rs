//! Native mod-Q exponent layout and strided monomials.
//!
//! A layout fixes the power-of-two stride used by exponent attributes.  The
//! effective exponent modulus is the order of that strided monomial subgroup
//! in the negacyclic ring, rather than an independently configured LUT
//! modulus.

use mxx_dsl::{Mat, Ring};
use mxx_ir_core::{IntExpr, ParamEnv, node::ConstantMatrix};
use num_bigint::{BigInt, Sign};
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

/// Errors returned while checking a native exponent layout or constructing a
/// concrete strided monomial.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum LayoutError {
    #[error("ring dimension {0} is not a positive power of two")]
    InvalidRingDimension(usize),
    #[error("stride log2 {stride_log2} must be smaller than ring log2 {ring_log2}")]
    StrideOutOfRange { stride_log2: u32, ring_log2: u32 },
    #[error("stride log2 {stride_log2} plus width log2 {width_log2} exceeds ring log2 {ring_log2}")]
    WidthOutOfRange { stride_log2: u32, width_log2: u32, ring_log2: u32 },
    #[error("ring dimension expression is not a concrete nonnegative usize")]
    NonConcreteRingDimension,
    #[error("ring dimension {0} is too large to form 2n")]
    RingDimensionOverflow(usize),
    #[error("native automorphism enumeration produced an invalid set")]
    InvalidAutomorphismSet,
    #[error("exponent modulus is not representable for ring dimension {0}")]
    ExponentModulusOverflow(usize),
}

/// Stride and operation-domain width for native mod-Q exponent encoding.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ExpLayout {
    /// `m` in `mono_m(z) = X^(2^m z)`.
    pub stride_log2: u32,
    /// `w`, the log2 size of the operation domain.
    pub width_log2: u32,
}

impl ExpLayout {
    /// Validates the layout against a concrete power-of-two ring dimension.
    pub fn validate(&self, ring_dimension: usize) -> Result<(), LayoutError> {
        if ring_dimension == 0 || !ring_dimension.is_power_of_two() {
            return Err(LayoutError::InvalidRingDimension(ring_dimension));
        }
        let ring_log2 = ring_dimension.trailing_zeros();
        if self.stride_log2 >= ring_log2 {
            return Err(LayoutError::StrideOutOfRange { stride_log2: self.stride_log2, ring_log2 });
        }
        let total_log2 =
            self.stride_log2.checked_add(self.width_log2).ok_or(LayoutError::WidthOutOfRange {
                stride_log2: self.stride_log2,
                width_log2: self.width_log2,
                ring_log2,
            })?;
        if total_log2 > ring_log2 {
            return Err(LayoutError::WidthOutOfRange {
                stride_log2: self.stride_log2,
                width_log2: self.width_log2,
                ring_log2,
            });
        }
        Ok(())
    }

    /// Returns `2^m`.
    pub fn stride(&self) -> usize {
        1usize.checked_shl(self.stride_log2).expect("exponent stride does not fit in usize")
    }

    /// Returns `2^w`, the width of this operation domain.
    pub fn width(&self) -> usize {
        1usize.checked_shl(self.width_log2).expect("exponent width does not fit in usize")
    }

    /// Returns the native exponent modulus `Q = 2n / 2^m`.
    ///
    /// The layout/ring pair is required to be valid; callers that have not
    /// validated it should use [`Self::validate`] first.
    pub fn effective_modulus(&self, ring_dimension: usize) -> usize {
        self.validate(ring_dimension).expect("invalid exponent layout");
        let doubled = ring_dimension.checked_mul(2).expect("2n does not fit in usize");
        doubled / self.stride()
    }

    /// Constructs `mono_m(z) = X^(2^m z)` in `R_q`.
    ///
    /// Exponents are reduced modulo `2n`; the representative in `[n, 2n)`
    /// is emitted as the negation of `X^(representative - n)`, implementing
    /// the relation `X^n = -1`.
    pub fn mono(&self, ring: &Ring, exponent: &BigInt) -> Result<Mat, LayoutError> {
        let ring_dimension = ring
            .matrix_type((1, 1))
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .map_err(|_| LayoutError::NonConcreteRingDimension)?
            .to_usize()
            .ok_or(LayoutError::NonConcreteRingDimension)?;
        self.validate(ring_dimension)?;

        let doubled = ring_dimension
            .checked_mul(2)
            .ok_or(LayoutError::ExponentModulusOverflow(ring_dimension))?;
        let modulus = BigInt::from(doubled);
        let mut reduced = (exponent * BigInt::from(self.stride())) % &modulus;
        if reduced.sign() == Sign::Minus {
            reduced += &modulus;
        }
        let reduced = reduced.to_usize().ok_or(LayoutError::NonConcreteRingDimension)?;
        let (rotation, negate) = if reduced < ring_dimension {
            (reduced, false)
        } else {
            (reduced - ring_dimension, true)
        };
        let monomial = ring
            .constant((1, 1), ConstantMatrix::Rotation { exponent: IntExpr::constant(rotation) });
        Ok(if negate { -monomial } else { monomial })
    }

    /// Enumerates the native single-layer automorphisms `J_{m,w}`.
    ///
    /// The returned order is ascending numeric order.  In particular, when
    /// `m = 0` this is exactly the order of the legacy canonical indices
    /// `1 + j * (2n / 2^w)`.
    pub fn automorphism_indices(&self, ring_dimension: usize) -> Result<Vec<usize>, LayoutError> {
        self.validate(ring_dimension)?;
        let modulus = ring_dimension
            .checked_mul(2)
            .ok_or(LayoutError::RingDimensionOverflow(ring_dimension))?;
        let mut indices = BTreeSet::new();
        let subset_count = self.width();
        for subset in 0..subset_count {
            let mut product = 1u128;
            for bit in 0..self.width_log2 {
                if subset & (1usize << bit) == 0 {
                    continue;
                }
                let i = self.stride_log2 + bit;
                let denominator = 1usize
                    .checked_shl(i + 1)
                    .ok_or(LayoutError::RingDimensionOverflow(ring_dimension))?;
                let factor = modulus / denominator + 1;
                product = (product * factor as u128) % modulus as u128;
            }
            indices.insert(product as usize);
        }

        let expected = subset_count;
        if indices.len() != expected ||
            indices.iter().any(|index| *index == 0 || index % 2 == 0) ||
            !indices.contains(&1)
        {
            return Err(LayoutError::InvalidAutomorphismSet);
        }
        Ok(indices.into_iter().collect())
    }

    /// Selects the canonical automorphism at a structural loop index without
    /// expanding one matrix operation per entry on the host.
    pub(crate) fn automorphism_index_expression(
        &self,
        ring_dimension: usize,
        index: IntExpr,
    ) -> Result<IntExpr, LayoutError> {
        Ok(IntExpr::Select {
            selector: Box::new(index),
            branches: self
                .automorphism_indices(ring_dimension)?
                .into_iter()
                .map(IntExpr::constant)
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{ExpLayout, LayoutError};
    use mxx_dsl::Ring;
    use mxx_ir_core::{IntExpr, ParamEnv, node::ConstantMatrix};
    use num_bigint::BigInt;
    use num_traits::ToPrimitive;

    #[test]
    fn executable_monomial_products_match_exponent_addition_for_m_zero_one_two() {
        use mxx_dsl::DslContext;
        use mxx_ir_core::ParamEnv;
        use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
        use mxx_runtime::{
            RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
            transcript::SamplingMode,
        };
        use std::collections::BTreeMap;

        let parameters = DCRTPolyParams::new(8, 1, 20, 4, None, None);
        let modulus = parameters.modulus().as_ref().clone();
        let ring = Ring::new(
            BigInt::from_biguint(num_bigint::Sign::Plus, modulus),
            parameters.ring_dimension() as usize,
        );
        let mut context = DslContext::new("layout-mono-product-identities");
        let mut names = Vec::new();
        for stride_log2 in 0..=2u32 {
            let layout = ExpLayout { stride_log2, width_log2: 0 };
            for (index, (a, b)) in [(-1i64, 2i64), (3, -4), (7, 5)].into_iter().enumerate() {
                let left = layout.mono(&ring, &BigInt::from(a)).unwrap();
                let right = layout.mono(&ring, &BigInt::from(b)).unwrap();
                let product = left * right;
                let expected = layout.mono(&ring, &BigInt::from(a + b)).unwrap();
                let product_name = format!("product-{stride_log2}-{index}");
                let expected_name = format!("expected-{stride_log2}-{index}");
                context = context.output(&product_name, product).unwrap();
                context = context.output(&expected_name, expected).unwrap();
                names.push((product_name, expected_name));
            }
        }
        let graph = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let result = execute(
            &graph,
            &mut cpu_backend([parameters.clone()]),
            BTreeMap::new(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        for (product_name, expected_name) in names {
            let RuntimeValue::Matrix(product) = &result.outputs[&product_name] else {
                panic!("product matrix")
            };
            let RuntimeValue::Matrix(expected) = &result.outputs[&expected_name] else {
                panic!("expected matrix")
            };
            assert_eq!(product.as_ref(), expected.as_ref(), "{product_name} identity");
        }
    }

    #[test]
    fn validates_stride_and_width_against_ring_dimension() {
        let layout = ExpLayout { stride_log2: 1, width_log2: 3 };
        assert_eq!(layout.validate(32), Ok(()));
        assert_eq!(layout.stride(), 2);
        assert_eq!(layout.width(), 8);
        assert_eq!(layout.effective_modulus(32), 32);
        assert_eq!(
            layout.validate(8),
            Err(LayoutError::WidthOutOfRange { stride_log2: 1, width_log2: 3, ring_log2: 3 })
        );
        assert_eq!(
            (ExpLayout { stride_log2: 4, width_log2: 0 }).validate(16),
            Err(LayoutError::StrideOutOfRange { stride_log2: 4, ring_log2: 4 })
        );
    }

    #[test]
    fn native_modulus_matches_the_ring_and_stride() {
        let n = 1usize << 16;
        assert_eq!((ExpLayout { stride_log2: 6, width_log2: 0 }).effective_modulus(n), 2048);
        assert_eq!((ExpLayout { stride_log2: 5, width_log2: 0 }).effective_modulus(n), 4096);
    }

    #[test]
    fn native_indices_are_distinct_odd_and_include_identity() {
        for layout in [
            ExpLayout { stride_log2: 0, width_log2: 0 },
            ExpLayout { stride_log2: 1, width_log2: 2 },
            ExpLayout { stride_log2: 2, width_log2: 3 },
        ] {
            let indices = layout.automorphism_indices(32).unwrap();
            assert_eq!(indices.len(), layout.width());
            assert_eq!(indices[0], 1);
            assert!(indices.iter().all(|index| index % 2 == 1 && *index < 64));
            let mut distinct = indices.clone();
            distinct.sort_unstable();
            distinct.dedup();
            assert_eq!(distinct, indices);
        }
    }

    #[test]
    fn m_zero_reproduces_legacy_canonical_order_bit_for_bit() {
        let ring_dimension = 64;
        let layout = ExpLayout { stride_log2: 0, width_log2: 4 };
        let expected = (0..16).map(|j| 1 + j * (2 * ring_dimension / 16)).collect::<Vec<_>>();
        assert_eq!(layout.automorphism_indices(ring_dimension).unwrap(), expected);
    }

    #[test]
    fn structural_index_selects_the_exact_canonical_automorphism_order() {
        let layout = ExpLayout { stride_log2: 2, width_log2: 3 };
        let expected = layout.automorphism_indices(32).unwrap();
        let expression = layout.automorphism_index_expression(32, IntExpr::LoopIndex(7)).unwrap();
        for (index, sigma) in expected.into_iter().enumerate() {
            let mut env = ParamEnv::default();
            env.loop_indices.insert(7, index.into());
            assert_eq!(expression.evaluate(&env).unwrap(), sigma.into());
        }
    }

    fn rotation_exponent(mat: &mxx_dsl::Mat) -> (usize, bool) {
        let (exponent, negate) = match mat.value_handle().node().kind() {
            mxx_ir_core::node::NodeKind::ConstantMatrix {
                value: ConstantMatrix::Rotation { exponent },
                ..
            } => (exponent.evaluate(&ParamEnv::default()).unwrap().to_usize().unwrap(), false),
            mxx_ir_core::node::NodeKind::MatrixNegate => {
                let child = mat.value_handle().node().arguments()[0].node();
                match child.kind() {
                    mxx_ir_core::node::NodeKind::ConstantMatrix {
                        value: ConstantMatrix::Rotation { exponent },
                        ..
                    } => {
                        (exponent.evaluate(&ParamEnv::default()).unwrap().to_usize().unwrap(), true)
                    }
                    other => panic!("expected rotation under negation, got {other:?}"),
                }
            }
            other => panic!("expected rotation, got {other:?}"),
        };
        (exponent, negate)
    }

    #[test]
    fn mono_wraps_native_exponents_with_negacyclic_sign() {
        let ring = Ring::new(97, 16);
        let layout = ExpLayout { stride_log2: 1, width_log2: 2 };
        assert_eq!(rotation_exponent(&layout.mono(&ring, &BigInt::from(0)).unwrap()), (0, false));
        assert_eq!(rotation_exponent(&layout.mono(&ring, &BigInt::from(7)).unwrap()), (14, false));
        assert_eq!(rotation_exponent(&layout.mono(&ring, &BigInt::from(8)).unwrap()), (0, true));
        assert_eq!(rotation_exponent(&layout.mono(&ring, &BigInt::from(-1)).unwrap()), (14, true));
    }

    #[test]
    fn mono_rejects_symbolic_ring_dimension() {
        let ring = Ring::new(97, IntExpr::Var("n".to_owned()));
        let layout = ExpLayout { stride_log2: 0, width_log2: 1 };
        assert!(matches!(
            layout.mono(&ring, &BigInt::from(0)),
            Err(LayoutError::NonConcreteRingDimension)
        ));
    }
}
