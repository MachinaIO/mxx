//! Scheme-independent LUT polynomials and structural balanced reductions.
//! These functions construct graph expressions; they do not inspect runtime
//! ciphertexts or add validation passes to their evaluation.
use crate::{
    ExponentLutError,
    decoder::simulation::AverageCaseConfig,
    program::ExponentLutProgram,
    rns_bgg::{
        RhsMode, RnsBggDerivedWorkload, RnsBggNoiseBindings, RnsBggNoiseTrace, RnsBggParameters,
        RnsBggPlanRequest,
    },
};
use bigdecimal::{BigDecimal, Zero};
use mxx_backends::poly::{PolyParams, dcrt::params::DCRTPolyParams};
use mxx_dsl::{Family, GraphValue, Int};
use mxx_ir_core::{ParamEnv, Rational, RealExpr};
use num_bigint::{BigInt, BigUint};
use num_traits::ToPrimitive;
use std::{env, error::Error, fmt};

/// Read an optional unsigned environment setting without accepting malformed values.
pub fn env_usize(name: &str, default: usize) -> Result<usize, String> {
    match env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .map_err(|error| format!("{name} must be a nonnegative integer: {error}")),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(format!("{name} is not valid Unicode: {error}")),
    }
}

/// Read a positive decimal environment setting, using `default` when absent.
pub fn env_positive_decimal(name: &str, default: &str) -> Result<BigDecimal, String> {
    let raw = match env::var(name) {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => default.to_owned(),
        Err(error) => return Err(format!("{name} is not valid Unicode: {error}")),
    };
    let value =
        raw.parse::<BigDecimal>().map_err(|error| format!("{name} must be a decimal: {error}"))?;
    if value <= BigDecimal::zero() {
        return Err(format!("{name} must be positive"));
    }
    Ok(value)
}

/// Return the layout stride exponent when a power-of-two ring is large enough.
pub fn ring_dimension_stride_log2(ring_dimension: u32, minimum_log: u32) -> Option<u32> {
    if ring_dimension == 0 || !ring_dimension.is_power_of_two() {
        return None;
    }
    ring_dimension.trailing_zeros().checked_sub(minimum_log)
}

/// Select the configured noise authority, defaulting to average-case analysis.
pub fn noise_model(prefix: &str) -> Result<Option<AverageCaseConfig>, String> {
    let model_name = format!("{prefix}NOISE_MODEL");
    let model = match env::var(&model_name) {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => "average".to_owned(),
        Err(error) => return Err(format!("{model_name} is not valid Unicode: {error}")),
    };
    match model.as_str() {
        "average" => {
            let failure_name = format!("{prefix}FAILURE_EXPONENT");
            let failure_exponent = env_usize(&failure_name, 100)?;
            if failure_exponent == 0 {
                return Err(format!("{failure_name} must be positive"));
            }
            Ok(Some(AverageCaseConfig {
                failure_exponent: u32::try_from(failure_exponent)
                    .map_err(|_| format!("{failure_name} must fit in u32"))?,
                allow_average_acceptance: true,
                ..AverageCaseConfig::default()
            }))
        }
        "worst" => Ok(None),
        value => Err(format!("{model_name} must be exactly `average` or `worst`, got `{value}`")),
    }
}

/// Select the configured RHS timing mode from `{prefix}RHS_MODE`: `static`
/// (the default) or `dynamic`.
pub fn rhs_mode(prefix: &str) -> Result<RhsMode, String> {
    let name = format!("{prefix}RHS_MODE");
    match env::var(&name).as_deref() {
        Err(env::VarError::NotPresent) | Ok("static") => Ok(RhsMode::Static),
        Ok("dynamic") => Ok(RhsMode::Dynamic),
        Ok(value) => Err(format!("{name} must be exactly `static` or `dynamic`, got `{value}`")),
        Err(error) => Err(format!("{name} is not valid Unicode: {error}")),
    }
}

/// Return inclusive power-of-two ring-dimension candidates.
pub fn ring_dimension_candidates(
    min_name: &str,
    max_name: &str,
    default_min_log: usize,
    default_max_log: usize,
) -> Result<Vec<u32>, String> {
    let min_log = env_usize(min_name, default_min_log)?;
    let max_log = env_usize(max_name, default_max_log)?;
    if min_log > max_log {
        return Err(format!("{min_name} ({min_log}) must not exceed {max_name} ({max_log})"));
    }
    if max_log >= usize::BITS as usize {
        return Err(format!("{max_name} ({max_log}) is too large for this platform"));
    }
    (min_log..=max_log)
        .map(|log| {
            let dimension = 1usize
                .checked_shl(log as u32)
                .ok_or_else(|| format!("ring dimension 2^{log} overflows usize"))?;
            u32::try_from(dimension)
                .map_err(|_| format!("ring dimension 2^{log} does not fit in u32"))
        })
        .collect()
}

/// Audit one exact CRT-depth candidate using the same workload and noise path
/// as the public search API.  Keeping the requested terminal depth unchanged
/// lets callers try deeper source CRT chains without silently changing the
/// configured drop cadence.
///
/// The chain ends at the program's last output level, after coefficient
/// finalization, so a decrypted output carries the terminal relation. A
/// program whose outputs continue requests `successor_levels` further levels
/// past its last LUT level.
pub fn audit_program_at_crt_depth(
    request: &RnsBggPlanRequest,
    program: &ExponentLutProgram,
    bindings: &RnsBggNoiseBindings,
    terminal_value_bounds: &[BigUint],
    average: Option<&AverageCaseConfig>,
    successor_levels: usize,
) -> Result<(RnsBggParameters, RnsBggNoiseTrace), String> {
    let (_, parameters) =
        plan_chain(request, program, successor_levels, request.max_crt_depth, None)?;
    let trace =
        audit_planned(request, &parameters, program, bindings, terminal_value_bounds, average)?;
    Ok((parameters, trace))
}

/// The prime pool of `request` and the chain of `program` on it: logical
/// level `l` keeps the first `planned_crt_depth(l)` primes below `top_depth`,
/// through the program's last output level plus `successor_levels`. With
/// `decoding = Some((level, cap))` every level after `level` keeps at most
/// `cap` primes, so the step out of `level` is a drop.
fn plan_chain(
    request: &RnsBggPlanRequest,
    program: &ExponentLutProgram,
    successor_levels: usize,
    top_depth: usize,
    decoding: Option<(usize, usize)>,
) -> Result<(DCRTPolyParams, RnsBggParameters), String> {
    if request.operations_per_drop == 0 ||
        request.terminal_depth == 0 ||
        request.max_crt_depth < top_depth ||
        top_depth < request.terminal_depth
    {
        return Err("invalid CRT-depth search request".to_owned());
    }
    let workload =
        RnsBggDerivedWorkload::from_program(program).map_err(|error| error.to_string())?;
    let output_levels = program
        .outputs()
        .iter()
        .map(|output| *output.source())
        .map(|wire| {
            let level = workload.level(wire).map_err(|error| error.to_string())?;
            let coefficient = program
                .wire_is_coefficient_output(wire)
                .map_err(|_| "output metadata is invalid".to_owned())?;
            Ok((level, level + usize::from(coefficient)))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let logical_count = output_levels.iter().map(|(level, _)| *level).max().unwrap_or(0);
    let output_last = output_levels.iter().map(|(_, level)| *level).max().unwrap_or(0);
    if request.base_bits == 0 || request.base_bits > 31 {
        return Err("decomposition base bits must be in 1..=31".to_owned());
    }
    let pool = DCRTPolyParams::try_new(
        request.ring_dimension,
        request.max_crt_depth,
        request.crt_bits,
        request.base_bits,
        None,
        None,
    )
    .map_err(|_| "unsupported prime pool".to_owned())?;
    let primes = pool.to_crt().0;
    // A Program may request successors beyond its last LUT level.
    // A Dynamic chain keeps every evaluated carrier in full-gadget form, so
    // it reaches one gadget successor past its normalized outputs.
    let gadget_successor = usize::from(request.rhs_mode == RhsMode::Dynamic);
    let chain_last = logical_count
        .checked_add(successor_levels)
        .ok_or_else(|| "logical level chain overflows".to_owned())?
        .max(output_last + gadget_successor);
    let mut levels = Vec::with_capacity(chain_last + 1);
    for logical_level in 0..=chain_last {
        let mut depth = crate::rns_bgg::planned_crt_depth(
            logical_level,
            request.operations_per_drop,
            request.primes_per_drop,
            top_depth,
            request.terminal_depth,
        )
        .map_err(|error| error.to_string())?;
        if let Some((level, cap)) = decoding &&
            logical_level > level
        {
            depth = depth.min(cap);
        }
        let modulus = primes
            .get(..depth)
            .ok_or_else(|| "CRT-depth candidate exceeds the prime pool".to_owned())?
            .iter()
            .copied()
            .map(BigUint::from)
            .product();
        levels.push(pool.select_modulus(&modulus).ok_or_else(|| "invalid CRT basis".to_owned())?);
    }
    let parameters = RnsBggParameters::new(
        levels,
        request.encoding_error_sigma.clone(),
        request.operations_per_drop,
    )
    .map_err(|error| error.to_string())?
    .with_rhs_mode(request.rhs_mode);
    Ok((pool, parameters))
}

/// Audit `program` on the planned `parameters`, logging the outcome.
fn audit_planned(
    request: &RnsBggPlanRequest,
    parameters: &RnsBggParameters,
    program: &ExponentLutProgram,
    bindings: &RnsBggNoiseBindings,
    terminal_value_bounds: &[BigUint],
    average: Option<&AverageCaseConfig>,
) -> Result<RnsBggNoiseTrace, String> {
    let started = std::time::Instant::now();
    tracing::info!(
        ring_dimension = request.ring_dimension,
        max_crt_depth = request.max_crt_depth,
        primes_per_drop = request.primes_per_drop,
        levels = parameters.levels().len(),
        "auditing parameter candidate"
    );
    let trace = parameters
        .audit_program(program, bindings, terminal_value_bounds, average)
        .map_err(|error| {
            tracing::info!(
                max_crt_depth = request.max_crt_depth,
                primes_per_drop = request.primes_per_drop,
                elapsed_ms = started.elapsed().as_millis() as u64,
                %error,
                "parameter candidate rejected by the noise audit"
            );
            error.to_string()
        })?;
    tracing::info!(
        max_crt_depth = request.max_crt_depth,
        primes_per_drop = request.primes_per_drop,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "parameter candidate passed the noise audit"
    );
    Ok(trace)
}

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
    use crate::test_support::{execute_cpu, explicit_ring, full, primes, ring_of, validate};
    use mxx_backends::poly::dcrt::params::DCRTPolyParams;
    use mxx_dsl::DslContext;
    use mxx_ir_core::IntExpr;
    use std::collections::BTreeMap;
    #[test]
    fn zero_drop_plan_chain_retains_exact_basis_and_successors() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
        let program = builder.finish(vec![crate::program::Operand::Scalar(input)]).unwrap();
        let mut request = RnsBggPlanRequest {
            ring_dimension: 16,
            encoding_error_sigma: 4.into(),
            crt_bits: 20,
            base_bits: 4,
            terminal_depth: 3,
            max_crt_depth: 3,
            terminal_value_bound: 1u8.into(),
            operations_per_drop: 1,
            primes_per_drop: 0,
            rhs_mode: RhsMode::Static,
        };
        let (pool, parameters) = plan_chain(&request, &program, 63, 3, None).unwrap();
        let exact = mxx_ir_core::generate_crt_basis(16, 3, 20).unwrap();
        assert_eq!(pool.to_crt().0, exact);
        assert_eq!(parameters.levels().len(), 64);
        assert!(parameters.levels().iter().all(|level| level.to_crt().0 == exact));
        request.operations_per_drop = 0;
        assert!(plan_chain(&request, &program, 63, 3, None).is_err());
    }

    #[test]
    fn odd_width_segmented_reduction_matches_scalar_addition() {
        let parameters = DCRTPolyParams::new(4, 1, 20, 4, None, None);
        let ring = ring_of(&parameters);
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
        let graph = validate(&context.build().unwrap());
        let result = execute_cpu(&graph, [parameters], BTreeMap::new());
        for width in [1usize, 3, 5, 257] {
            for segment in 0..2 {
                let expected = full(&result.outputs[&format!("expected-{width}-{segment}")]);
                for prefix in ["actual", "scalar"] {
                    let actual = full(&result.outputs[&format!("{prefix}-{width}-{segment}")]);
                    assert_eq!(actual, expected);
                }
            }
        }
    }
    #[test]
    fn public_segments_preserve_slot_axis_without_implicit_family_capture() {
        let ring = explicit_ring(&primes(4, 1, 17), 4);
        let routed = ring.input_family("routed-public", 15, (2, 4));
        let reduced = reduce_family_segments(routed, 3, 5, |left, right| left + right).unwrap();
        assert_eq!(reduced.count(), &IntExpr::constant(3));
        assert_eq!(reduced.element_type(), &ring.matrix_type((2, 4)));
        let graph = DslContext::new("public-segments")
            .output("slot-targets", reduced)
            .unwrap()
            .build()
            .unwrap();
        validate(&graph);
    }
}
