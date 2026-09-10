//! Scheme-independent LUT polynomials and structural balanced reductions.
//! These functions construct graph expressions; they do not inspect runtime
//! ciphertexts or add validation passes to their evaluation.
use crate::ExponentLutError;
use mxx_dsl::{Family, GraphValue, Int, Mat};
use mxx_ir_core::{IntExpr, ParamEnv};
use num_bigint::BigInt;
use num_traits::ToPrimitive;

pub(crate) fn lut_coefficient(
    ring: &mxx_dsl::Ring,
    layout: &crate::layout::ExpLayout,
    sigma: impl Into<IntExpr>,
    table: &[usize],
) -> Result<Mat, ExponentLutError> {
    // The coefficient is exactly
    // `W'^{-1} sum_k mono_m(L(k)) sigma(mono_m(-k))`.
    let width = table.len();
    if width == 0 || !width.is_power_of_two() || width != layout.width() {
        return Err(ExponentLutError::InvalidLut);
    }
    let sigma = sigma.into();
    let modulus = ring
        .zero((1, 1))
        .matrix_type()
        .modulus
        .evaluate(&ParamEnv::default())
        .map_err(|_| ExponentLutError::InvalidLut)?;
    let inverse = modular_inverse(&(BigInt::from(width) % &modulus), &modulus)
        .ok_or(ExponentLutError::InvalidLut)?;
    let mut terms = Vec::with_capacity(width);
    for (k, output) in table.iter().copied().enumerate() {
        let left =
            layout.mono(ring, &BigInt::from(output)).map_err(|_| ExponentLutError::InvalidLut)?;
        let right = layout
            .mono(ring, &-BigInt::from(k))
            .map_err(|_| ExponentLutError::InvalidLut)?
            .ring_automorphism(sigma.clone());
        terms.push(left * right);
    }
    while terms.len() > 1 {
        let mut next = Vec::with_capacity(terms.len().div_ceil(2));
        let mut iter = std::mem::take(&mut terms).into_iter();
        while let Some(left) = iter.next() {
            next.push(match iter.next() {
                Some(right) => left + right,
                None => left,
            });
        }
        terms = next;
    }
    let result = terms.pop().expect("validated nonempty LUT");
    Ok(ring.polynomial([inverse.into()]) * result)
}

pub(crate) fn derive_lut_layout(
    base: &crate::layout::ExpLayout,
    width: usize,
    ring_dimension: usize,
) -> Result<crate::layout::ExpLayout, ExponentLutError> {
    if width == 0 || !width.is_power_of_two() || width > base.width() {
        return Err(ExponentLutError::InvalidLut);
    }
    let layout = crate::layout::ExpLayout {
        stride_log2: base.stride_log2,
        width_log2: width.trailing_zeros(),
    };
    layout.validate(ring_dimension).map_err(|_| ExponentLutError::InvalidLut)?;
    Ok(layout)
}

pub(crate) fn modular_inverse(value: &BigInt, modulus: &BigInt) -> Option<BigInt> {
    use num_traits::{One, Zero};
    let mut old_r = value.clone();
    let mut r = modulus.clone();
    let mut old_s = BigInt::one();
    let mut s = BigInt::zero();
    while !r.is_zero() {
        let quotient = &old_r / &r;
        (old_r, r) = (r.clone(), old_r - &quotient * &r);
        (old_s, s) = (s.clone(), old_s - quotient * &s);
    }
    (old_r == BigInt::one()).then(|| ((old_s % modulus) + modulus) % modulus)
}

/// Maximum selectors or branches in one complete preprocessing batch.
pub(crate) fn preprocess_batch_size() -> Result<usize, mxx_dsl::DslError> {
    std::env::var("MXX_EXPONENT_LUT_PREPROCESS_BATCH_SIZE")
        .ok()
        .map(|value| value.parse::<std::num::NonZeroUsize>())
        .transpose()
        .map_err(|_| mxx_dsl::DslError::Schema)
        .map(|value| value.map_or(256, |value| value.get()))
}

pub(crate) fn balanced_sum_family<T: mxx_dsl::GraphValue + std::ops::Add<Output = T>>(
    family: Family<T>,
) -> Result<T, ExponentLutError> {
    let count = family
        .count()
        .evaluate(&ParamEnv::default())
        .ok()
        .and_then(|value| value.to_usize())
        .ok_or(ExponentLutError::InvalidLut)?;
    Ok(reduce_family_segments(family, 1, count, |left, right| left + right)?.at(0))
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

/// Compare concrete matrix types while allowing equivalent parameter expressions.
pub(crate) fn same_matrix_type(
    left: &mxx_ir_core::types::MatrixType,
    right: &mxx_ir_core::types::MatrixType,
) -> bool {
    [
        (&left.modulus, &right.modulus),
        (&left.ring_dimension, &right.ring_dimension),
        (&left.rows, &right.rows),
        (&left.columns, &right.columns),
    ]
    .into_iter()
    .all(|(left, right)| {
        match (left.evaluate(&ParamEnv::default()), right.evaluate(&ParamEnv::default())) {
            (Ok(left), Ok(right)) => left == right,
            _ => false,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::{DslContext, Ring};
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
                        super::balanced_sum_family(Family::pack(inputs.to_vec()).unwrap()).unwrap(),
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

/// Combine a fixed active RHS family using output-major public weights.
/// The construction contract requires a positive active count and an integral
/// number of weight rows. All terms (including their errors) enter each sum.
pub(crate) fn weighted_family_combinations<T>(
    rhs: &Family<T>,
    weights: &Family<Mat>,
    active: &crate::program::FamilyRange,
) -> Result<Family<T>, mxx_dsl::DslError>
where
    T: GraphValue + std::ops::Add<Output = T>,
    Mat: std::ops::Mul<T, Output = T>,
{
    let count = Int::evaluate(weights.count().clone())
        .div(Int::evaluate(active.count().clone()))
        .expression()?;
    // Resolve concrete setup dimensions once; retain symbolic dimensions for
    // callers preparing a block inside an enclosing structural loop.
    let count = count
        .evaluate(&mxx_ir_core::ParamEnv::default())
        .map(mxx_ir_core::IntExpr::constant)
        .unwrap_or(count);
    mxx_dsl::parallel(count, |output| {
        let start = output.mul(Int::evaluate(active.count().clone()));
        let size = active
            .count()
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|count| count.to_usize())
            .ok_or(mxx_dsl::DslError::Schema)?;
        reduce_generated_segments(
            1,
            size,
            |_, cell| {
                let external = cell.clone().add(Int::evaluate(active.start().clone()));
                weights.at(start.clone().add(cell)) * rhs.at(external)
            },
            |left, right| left + right,
        )
        .map(|family| family.at(0))
        .map_err(|_| mxx_dsl::DslError::Schema)
    })
}
