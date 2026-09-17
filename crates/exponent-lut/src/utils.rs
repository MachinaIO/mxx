//! Scheme-independent LUT polynomials and structural balanced reductions.
//! These functions construct graph expressions; they do not inspect runtime
//! ciphertexts or add validation passes to their evaluation.
use crate::ExponentLutError;
use bigdecimal::BigDecimal;
use mxx_dsl::{Family, GraphValue, Int};
use mxx_ir_core::{ParamEnv, Rational, RealExpr};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use std::{error::Error, fmt};

/// Errors raised while converting a finite decimal into an exact IR real.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DecimalExpressionError {
    ExponentOutOfRange,
    InvalidRational,
}

impl fmt::Display for DecimalExpressionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExponentOutOfRange => formatter.write_str("decimal exponent is out of range"),
            Self::InvalidRational => formatter.write_str("decimal is not a valid rational"),
        }
    }
}

impl Error for DecimalExpressionError {}

/// Convert BigDecimal without floating-point rounding or unchecked exponent
/// arithmetic. BigDecimal stores `integer * 10^(-exponent)`.
pub(crate) fn decimal_to_real_expr(value: &BigDecimal) -> Result<RealExpr, DecimalExpressionError> {
    let (integer, exponent) = value.as_bigint_and_exponent();
    let magnitude = if exponent < 0 {
        exponent.checked_neg().ok_or(DecimalExpressionError::ExponentOutOfRange)?
    } else {
        exponent
    };
    let magnitude =
        u32::try_from(magnitude).map_err(|_| DecimalExpressionError::ExponentOutOfRange)?;
    let power = BigInt::from(10u8).pow(magnitude);
    let (numerator, denominator) =
        if exponent < 0 { (integer * power, BigInt::from(1u8)) } else { (integer, power) };
    Rational::new(numerator, denominator)
        .map(RealExpr::Rational)
        .map_err(|_| DecimalExpressionError::InvalidRational)
}

#[cfg(test)]
mod decimal_tests {
    use super::*;
    use std::str::FromStr;

    fn rational(numerator: i64, denominator: i64) -> RealExpr {
        RealExpr::Rational(Rational::new(numerator.into(), denominator.into()).unwrap())
    }

    #[test]
    fn decimal_expression_conversion_is_exact_and_checks_exponents() {
        assert_eq!(decimal_to_real_expr(&BigDecimal::from(7u8)).unwrap(), rational(7, 1));
        assert_eq!(
            decimal_to_real_expr(&BigDecimal::from_str("0.125").unwrap()).unwrap(),
            rational(1, 8)
        );
        assert_eq!(
            decimal_to_real_expr(&BigDecimal::new(BigInt::from(12), 2)).unwrap(),
            rational(3, 25)
        );
        assert_eq!(
            decimal_to_real_expr(&BigDecimal::new(BigInt::from(12), -2)).unwrap(),
            rational(1200, 1)
        );
        assert_eq!(
            decimal_to_real_expr(&BigDecimal::from_str("4.0").unwrap()).unwrap(),
            decimal_to_real_expr(&BigDecimal::from_str("4.00").unwrap()).unwrap()
        );
        assert_eq!(
            decimal_to_real_expr(&BigDecimal::new(BigInt::from(1u8), i64::MAX)),
            Err(DecimalExpressionError::ExponentOutOfRange)
        );
        assert_eq!(
            decimal_to_real_expr(&BigDecimal::new(BigInt::from(1u8), i64::MIN)),
            Err(DecimalExpressionError::ExponentOutOfRange)
        );
    }
}

/// Reduce a concrete family with the backend's addition operation.
///
/// The utility owns only the structural reduction.  The operation is supplied
/// by the caller so this scheme-independent module does not encode tuple
/// carriers or depend on a concrete evaluator's `Add` implementation.
pub(crate) fn balanced_sum_family<T: GraphValue>(
    family: Family<T>,
    add: impl Fn(T, T) -> T,
) -> Result<T, ExponentLutError> {
    let count = family
        .count()
        .evaluate(&ParamEnv::default())
        .ok()
        .and_then(|value| value.to_usize())
        .ok_or(ExponentLutError::InvalidLut)?;
    Ok(reduce_family_segments(family, 1, count, add)?.at(0))
}

pub(crate) fn reduce_family_segments<T: GraphValue>(
    family: Family<T>,
    segment_count: usize,
    segment_size: usize,
    add: impl Fn(T, T) -> T,
) -> Result<Family<T>, ExponentLutError> {
    if segment_count == 0 ||
        segment_size == 0 ||
        segment_count.checked_mul(segment_size) !=
            Some(
                family
                    .count()
                    .evaluate(&ParamEnv::default())
                    .ok()
                    .and_then(|count| count.to_usize())
                    .ok_or(ExponentLutError::InvalidLut)?,
            )
    {
        return Err(ExponentLutError::InvalidLut);
    }
    reduce_generated_segments(
        segment_count,
        segment_size,
        |segment, member| family.at(segment.mul(Int::constant(segment_size)).add(member)),
        add,
    )
}

/// Reduce generated summands with an ordinary parallel tree. Batching belongs
/// to the producer/consumer loop, rather than to addition itself.
pub(crate) fn reduce_generated_segments<T: GraphValue>(
    segment_count: usize,
    segment_size: usize,
    element: impl Fn(Int, Int) -> T,
    add: impl Fn(T, T) -> T,
) -> Result<Family<T>, ExponentLutError> {
    let mut current = mxx_dsl::parallel(segment_count * segment_size, |index| {
        Ok(element(
            index.clone().div(Int::constant(segment_size)),
            index.rem(Int::constant(segment_size)),
        ))
    })
    .map_err(|_| ExponentLutError::InvalidLut)?;
    let mut width = segment_size;
    while width > 1 {
        let next_width = width.div_ceil(2);
        current = mxx_dsl::parallel(segment_count * next_width, |index| {
            let segment = index.clone().div(Int::constant(next_width));
            let member = index.rem(Int::constant(next_width));
            let base = segment.mul(Int::constant(width));
            let pair = member.clone().mul(Int::constant(2));
            let left = current.at(base.clone().add(pair.clone()));
            let right = current.at(base.add(pair.add(Int::constant(1)).rem(Int::constant(width))));
            if width % 2 == 0 {
                Ok(add(left, right))
            } else {
                let last = member.equal(Int::constant(next_width - 1)).to_int();
                mxx_dsl::select(last, vec![add(left.clone(), right), left])
            }
        })
        .map_err(|_| ExponentLutError::InvalidLut)?;
        width = next_width;
    }
    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::{DslContext, Ring};
    use mxx_ir_core::IntExpr;
    use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
    use mxx_runtime::{
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
        transcript::SamplingMode,
    };
    use std::collections::BTreeMap;
    #[test]
    fn odd_width_segmented_reduction_matches_scalar_addition() {
        let parameters = DCRTPolyParams::new(4, 1, 20, 4, None, None);
        let ring = Ring::new(BigInt::from(parameters.modulus().as_ref().clone()), 4);
        let mut context = DslContext::new("odd-width-segmented-reduction");
        for width in [1usize, 3, 5, 257] {
            let values = (0..2 * width)
                .map(|index| ring.polynomial([(index + 1).into()]))
                .collect::<Vec<_>>();
            let actual = super::reduce_family_segments(
                Family::pack(values.clone()).unwrap(),
                2,
                width,
                |left, right| left + right,
            )
            .unwrap();
            for segment in 0..2 {
                let inputs = &values[segment * width..(segment + 1) * width];
                let expected = inputs.iter().cloned().reduce(|left, right| left + right).unwrap();
                context = context
                    .output(format!("actual-{width}-{segment}"), actual.at(segment))
                    .unwrap()
                    .output(format!("expected-{width}-{segment}"), expected)
                    .unwrap()
                    .output(
                        format!("scalar-{width}-{segment}"),
                        super::balanced_sum_family(
                            Family::pack(inputs.to_vec()).unwrap(),
                            |left, right| left + right,
                        )
                        .unwrap(),
                    )
                    .unwrap();
            }
        }
        let graph = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let result = execute(
            &graph,
            &mut cpu_backend([parameters]),
            BTreeMap::new(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        for width in [1usize, 3, 5, 257] {
            for segment in 0..2 {
                let RuntimeValue::Matrix(expected) =
                    &result.outputs[&format!("expected-{width}-{segment}")]
                else {
                    panic!("expected matrix")
                };
                for prefix in ["actual", "scalar"] {
                    let RuntimeValue::Matrix(actual) =
                        &result.outputs[&format!("{prefix}-{width}-{segment}")]
                    else {
                        panic!("reduced matrix")
                    };
                    assert_eq!(actual.as_ref(), expected.as_ref());
                }
            }
        }
    }
    #[test]
    fn public_refresh_segments_preserve_slot_axis_without_implicit_family_capture() {
        let ring = mxx_dsl::Ring::new(97, 4);
        let routed = ring.input_family("routed-public-prf", 15, (2, 4));
        let reduced = reduce_family_segments(routed, 3, 5, |left, right| left + right).unwrap();
        assert_eq!(reduced.count(), &IntExpr::constant(3));
        assert_eq!(reduced.element_type(), &ring.matrix_type((2, 4)));
        let graph = DslContext::new("public-refresh-segments")
            .output("slot-targets", reduced)
            .unwrap()
            .build()
            .unwrap();
        graph.validate(&ParamEnv::default()).unwrap();
    }
}
