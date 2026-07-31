use crate::serde_support;
use num_bigint::{BigInt, Sign};
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, btree_map::Entry};
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum IntExpr {
    Const(#[serde(with = "serde_support::bigint")] BigInt),
    Var(String),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    RoundDiv(Box<Self>, Box<Self>),
    Log2Ceil(Box<Self>),
}

impl Serialize for IntExpr {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        IntExprRepr::from(self.canonicalize()).serialize(serializer)
    }
}

#[derive(Serialize)]
#[serde(tag = "tag", content = "value")]
enum IntExprRepr {
    Const(String),
    Var(String),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    RoundDiv(Box<Self>, Box<Self>),
    Log2Ceil(Box<Self>),
}

impl From<IntExpr> for IntExprRepr {
    fn from(value: IntExpr) -> Self {
        match value {
            IntExpr::Const(value) => Self::Const(value.to_string()),
            IntExpr::Var(name) => Self::Var(name),
            IntExpr::Add(lhs, rhs) => {
                Self::Add(Box::new(Self::from(*lhs)), Box::new(Self::from(*rhs)))
            }
            IntExpr::Sub(lhs, rhs) => {
                Self::Sub(Box::new(Self::from(*lhs)), Box::new(Self::from(*rhs)))
            }
            IntExpr::Mul(lhs, rhs) => {
                Self::Mul(Box::new(Self::from(*lhs)), Box::new(Self::from(*rhs)))
            }
            IntExpr::Div(lhs, rhs) => {
                Self::Div(Box::new(Self::from(*lhs)), Box::new(Self::from(*rhs)))
            }
            IntExpr::RoundDiv(lhs, rhs) => {
                Self::RoundDiv(Box::new(Self::from(*lhs)), Box::new(Self::from(*rhs)))
            }
            IntExpr::Log2Ceil(value) => Self::Log2Ceil(Box::new(Self::from(*value))),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct Rational {
    #[serde(with = "serde_support::bigint")]
    numerator: BigInt,
    #[serde(with = "serde_support::bigint")]
    denominator: BigInt,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum RealExpr {
    Rational(Rational),
    Var(String),
    FromInt(IntExpr),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    Sqrt(Box<Self>),
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ParamEnv {
    pub integers: BTreeMap<String, BigInt>,
    pub reals: BTreeMap<String, Rational>,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum ExprError {
    #[error("unbound compile variable: {0}")]
    UnboundVariable(String),
    #[error("integer division by zero")]
    DivisionByZero,
    #[error("inexact integer division: {numerator} is not divisible by {denominator}")]
    InexactDivision { numerator: BigInt, denominator: BigInt },
    #[error("RoundDiv denominator must be positive")]
    InvalidRoundDivDenominator,
    #[error("Log2Ceil argument must be at least one")]
    InvalidLog2CeilArgument,
    #[error("rational denominator must be nonzero")]
    InvalidRationalDenominator,
    #[error("a nonnegative real expression evaluated to a negative value")]
    NegativeReal,
    #[error("a floating-point value is not finite")]
    NonFiniteReal,
}

impl IntExpr {
    pub fn constant(value: impl Into<BigInt>) -> Self {
        Self::Const(value.into())
    }

    pub fn evaluate(&self, env: &ParamEnv) -> Result<BigInt, ExprError> {
        match self {
            Self::Const(value) => Ok(value.clone()),
            Self::Var(name) => env
                .integers
                .get(name)
                .cloned()
                .ok_or_else(|| ExprError::UnboundVariable(name.clone())),
            Self::Add(lhs, rhs) => Ok(lhs.evaluate(env)? + rhs.evaluate(env)?),
            Self::Sub(lhs, rhs) => Ok(lhs.evaluate(env)? - rhs.evaluate(env)?),
            Self::Mul(lhs, rhs) => Ok(lhs.evaluate(env)? * rhs.evaluate(env)?),
            Self::Div(lhs, rhs) => {
                let numerator = lhs.evaluate(env)?;
                let denominator = rhs.evaluate(env)?;
                if denominator.is_zero() {
                    return Err(ExprError::DivisionByZero);
                }
                let (quotient, remainder) = numerator.div_rem(&denominator);
                if !remainder.is_zero() {
                    return Err(ExprError::InexactDivision { numerator, denominator });
                }
                Ok(quotient)
            }
            Self::RoundDiv(lhs, rhs) => {
                let numerator = lhs.evaluate(env)?;
                let denominator = rhs.evaluate(env)?;
                if denominator <= BigInt::zero() {
                    return Err(ExprError::InvalidRoundDivDenominator);
                }
                let two = BigInt::from(2);
                Ok((numerator * &two + &denominator).div_floor(&(denominator * two)))
            }
            Self::Log2Ceil(value) => {
                let value = value.evaluate(env)?;
                if value < BigInt::one() {
                    return Err(ExprError::InvalidLog2CeilArgument);
                }
                let value = value.to_biguint().expect("positive value");
                let floor = value.bits() - 1;
                let is_power_of_two = value == (num_bigint::BigUint::one() << floor as usize);
                Ok(BigInt::from(if is_power_of_two { floor } else { floor + 1 }))
            }
        }
    }

    /// Returns the normative polynomial normal form over opaque generators.
    pub fn canonicalize(&self) -> Self {
        Polynomial::from_expr(self).into_expr()
    }

    pub fn contains_variable(&self, variable: &str) -> bool {
        match self {
            Self::Const(_) => false,
            Self::Var(name) => name == variable,
            Self::Add(lhs, rhs) |
            Self::Sub(lhs, rhs) |
            Self::Mul(lhs, rhs) |
            Self::Div(lhs, rhs) |
            Self::RoundDiv(lhs, rhs) => {
                lhs.contains_variable(variable) || rhs.contains_variable(variable)
            }
            Self::Log2Ceil(value) => value.contains_variable(variable),
        }
    }
}

impl Rational {
    pub fn new(numerator: BigInt, denominator: BigInt) -> Result<Self, ExprError> {
        if denominator.is_zero() {
            return Err(ExprError::InvalidRationalDenominator);
        }
        let sign = denominator.sign();
        let numerator = if sign == Sign::Minus { -numerator } else { numerator };
        let denominator = denominator.abs();
        let gcd = numerator.abs().gcd(&denominator);
        Ok(Self { numerator: &numerator / &gcd, denominator: &denominator / &gcd })
    }

    pub fn from_integer(value: BigInt) -> Self {
        Self { numerator: value, denominator: BigInt::one() }
    }

    /// Converts one finite IEEE-754 binary64 value to its exact mathematical
    /// rational value without passing through a decimal or rounded integer.
    pub fn from_f64_exact(value: f64) -> Result<Self, ExprError> {
        if !value.is_finite() {
            return Err(ExprError::NonFiniteReal);
        }
        let bits = value.to_bits();
        let negative = bits >> 63 != 0;
        let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
        let fraction = bits & ((1u64 << 52) - 1);
        if exponent_bits == 0 && fraction == 0 {
            return Ok(Self::from_integer(BigInt::zero()));
        }
        let (significand, exponent) = if exponent_bits == 0 {
            (fraction, -1074)
        } else {
            ((1u64 << 52) | fraction, exponent_bits - 1023 - 52)
        };
        let mut numerator = BigInt::from(significand);
        let mut denominator = BigInt::one();
        if exponent >= 0 {
            numerator <<= exponent as usize;
        } else {
            denominator <<= (-exponent) as usize;
        }
        if negative {
            numerator = -numerator;
        }
        Self::new(numerator, denominator)
    }

    pub fn numerator(&self) -> &BigInt {
        &self.numerator
    }

    pub fn denominator(&self) -> &BigInt {
        &self.denominator
    }

    fn add(&self, rhs: &Self) -> Self {
        Self::new(
            &self.numerator * &rhs.denominator + &rhs.numerator * &self.denominator,
            &self.denominator * &rhs.denominator,
        )
        .expect("nonzero rational denominator")
    }

    fn sub(&self, rhs: &Self) -> Self {
        Self::new(
            &self.numerator * &rhs.denominator - &rhs.numerator * &self.denominator,
            &self.denominator * &rhs.denominator,
        )
        .expect("nonzero rational denominator")
    }

    fn mul(&self, rhs: &Self) -> Self {
        Self::new(&self.numerator * &rhs.numerator, &self.denominator * &rhs.denominator)
            .expect("nonzero rational denominator")
    }

    fn div(&self, rhs: &Self) -> Result<Self, ExprError> {
        Self::new(&self.numerator * &rhs.denominator, &self.denominator * &rhs.numerator)
    }
}

impl RealExpr {
    pub fn from_f64_exact(value: f64) -> Result<Self, ExprError> {
        Ok(Self::Rational(Rational::from_f64_exact(value)?))
    }

    pub fn contains_variable(&self, variable: &str) -> bool {
        match self {
            Self::Rational(_) => false,
            Self::Var(name) => name == variable,
            Self::FromInt(value) => value.contains_variable(variable),
            Self::Add(lhs, rhs) |
            Self::Sub(lhs, rhs) |
            Self::Mul(lhs, rhs) |
            Self::Div(lhs, rhs) => {
                lhs.contains_variable(variable) || rhs.contains_variable(variable)
            }
            Self::Sqrt(value) => value.contains_variable(variable),
        }
    }

    pub fn evaluate_f64(&self, env: &ParamEnv) -> Result<f64, ExprError> {
        let value = match self {
            Self::Rational(value) => {
                let numerator = value.numerator().to_f64().ok_or(ExprError::NegativeReal)?;
                let denominator = value.denominator().to_f64().ok_or(ExprError::NegativeReal)?;
                numerator / denominator
            }
            Self::Var(name) => {
                let value =
                    env.reals.get(name).ok_or_else(|| ExprError::UnboundVariable(name.clone()))?;
                let numerator = value.numerator().to_f64().ok_or(ExprError::NegativeReal)?;
                let denominator = value.denominator().to_f64().ok_or(ExprError::NegativeReal)?;
                numerator / denominator
            }
            Self::FromInt(value) => value.evaluate(env)?.to_f64().ok_or(ExprError::NegativeReal)?,
            Self::Add(lhs, rhs) => lhs.evaluate_f64(env)? + rhs.evaluate_f64(env)?,
            Self::Sub(lhs, rhs) => lhs.evaluate_f64(env)? - rhs.evaluate_f64(env)?,
            Self::Mul(lhs, rhs) => lhs.evaluate_f64(env)? * rhs.evaluate_f64(env)?,
            Self::Div(lhs, rhs) => {
                let denominator = rhs.evaluate_f64(env)?;
                if denominator == 0.0 {
                    return Err(ExprError::DivisionByZero);
                }
                lhs.evaluate_f64(env)? / denominator
            }
            Self::Sqrt(value) => {
                let value = value.evaluate_f64(env)?;
                if value < 0.0 {
                    return Err(ExprError::NegativeReal);
                }
                value.sqrt()
            }
        };
        if value.is_finite() { Ok(value) } else { Err(ExprError::NegativeReal) }
    }

    pub fn evaluate_rational(&self, env: &ParamEnv) -> Result<Rational, ExprError> {
        match self {
            Self::Rational(value) => Ok(value.clone()),
            Self::Var(name) => {
                env.reals.get(name).cloned().ok_or_else(|| ExprError::UnboundVariable(name.clone()))
            }
            Self::FromInt(value) => Ok(Rational::from_integer(value.evaluate(env)?)),
            Self::Add(lhs, rhs) => {
                Ok(lhs.evaluate_rational(env)?.add(&rhs.evaluate_rational(env)?))
            }
            Self::Sub(lhs, rhs) => {
                Ok(lhs.evaluate_rational(env)?.sub(&rhs.evaluate_rational(env)?))
            }
            Self::Mul(lhs, rhs) => {
                Ok(lhs.evaluate_rational(env)?.mul(&rhs.evaluate_rational(env)?))
            }
            Self::Div(lhs, rhs) => lhs.evaluate_rational(env)?.div(&rhs.evaluate_rational(env)?),
            Self::Sqrt(_) => Err(ExprError::InvalidRationalDenominator),
        }
    }

    /// Substitutes every compile-time variable while preserving square roots
    /// as exact symbolic operations. The returned expression is independent
    /// of `env` and is suitable for persisted type descriptors.
    pub fn close(&self, env: &ParamEnv) -> Result<Self, ExprError> {
        if !self.contains_sqrt() {
            return Ok(Self::Rational(self.evaluate_rational(env)?));
        }
        Ok(match self {
            Self::Rational(value) => Self::Rational(value.clone()),
            Self::Var(name) => Self::Rational(
                env.reals
                    .get(name)
                    .cloned()
                    .ok_or_else(|| ExprError::UnboundVariable(name.clone()))?,
            ),
            Self::FromInt(value) => Self::FromInt(IntExpr::constant(value.evaluate(env)?)),
            Self::Add(lhs, rhs) => Self::Add(Box::new(lhs.close(env)?), Box::new(rhs.close(env)?)),
            Self::Sub(lhs, rhs) => Self::Sub(Box::new(lhs.close(env)?), Box::new(rhs.close(env)?)),
            Self::Mul(lhs, rhs) => Self::Mul(Box::new(lhs.close(env)?), Box::new(rhs.close(env)?)),
            Self::Div(lhs, rhs) => Self::Div(Box::new(lhs.close(env)?), Box::new(rhs.close(env)?)),
            Self::Sqrt(value) => Self::Sqrt(Box::new(value.close(env)?)),
        })
    }

    fn contains_sqrt(&self) -> bool {
        match self {
            Self::Sqrt(_) => true,
            Self::Add(lhs, rhs) |
            Self::Sub(lhs, rhs) |
            Self::Mul(lhs, rhs) |
            Self::Div(lhs, rhs) => lhs.contains_sqrt() || rhs.contains_sqrt(),
            Self::Rational(_) | Self::Var(_) | Self::FromInt(_) => false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum Generator {
    Var(String),
    Div(IntExpr, IntExpr),
    RoundDiv(IntExpr, IntExpr),
    Log2Ceil(IntExpr),
}

type Monomial = Vec<Generator>;

#[derive(Clone, Debug, Default)]
struct Polynomial(BTreeMap<Monomial, BigInt>);

impl Polynomial {
    fn from_expr(expr: &IntExpr) -> Self {
        match expr {
            IntExpr::Const(value) => Self::constant(value.clone()),
            IntExpr::Var(name) => Self::generator(Generator::Var(name.clone())),
            IntExpr::Add(lhs, rhs) => Self::from_expr(lhs).add(Self::from_expr(rhs)),
            IntExpr::Sub(lhs, rhs) => Self::from_expr(lhs).sub(Self::from_expr(rhs)),
            IntExpr::Mul(lhs, rhs) => Self::from_expr(lhs).mul(Self::from_expr(rhs)),
            IntExpr::Div(lhs, rhs) => {
                Self::generator(Generator::Div(lhs.canonicalize(), rhs.canonicalize()))
            }
            IntExpr::RoundDiv(lhs, rhs) => {
                Self::generator(Generator::RoundDiv(lhs.canonicalize(), rhs.canonicalize()))
            }
            IntExpr::Log2Ceil(value) => Self::generator(Generator::Log2Ceil(value.canonicalize())),
        }
    }

    fn constant(value: BigInt) -> Self {
        if value.is_zero() { Self::default() } else { Self(BTreeMap::from([(Vec::new(), value)])) }
    }

    fn generator(generator: Generator) -> Self {
        Self(BTreeMap::from([(vec![generator], BigInt::one())]))
    }

    fn add(mut self, rhs: Self) -> Self {
        for (monomial, coefficient) in rhs.0 {
            match self.0.entry(monomial) {
                Entry::Vacant(entry) => {
                    entry.insert(coefficient);
                }
                Entry::Occupied(mut entry) => {
                    *entry.get_mut() += coefficient;
                    if entry.get().is_zero() {
                        entry.remove();
                    }
                }
            }
        }
        self
    }

    fn sub(self, mut rhs: Self) -> Self {
        for coefficient in rhs.0.values_mut() {
            *coefficient = -coefficient.clone();
        }
        self.add(rhs)
    }

    fn mul(self, rhs: Self) -> Self {
        let mut output = Self::default();
        for (lhs_monomial, lhs_coefficient) in self.0 {
            for (rhs_monomial, rhs_coefficient) in &rhs.0 {
                let mut monomial = lhs_monomial.clone();
                monomial.extend(rhs_monomial.iter().cloned());
                monomial.sort();
                let coefficient = &lhs_coefficient * rhs_coefficient;
                output = output.add(Self(BTreeMap::from([(monomial, coefficient)])));
            }
        }
        output
    }

    fn into_expr(self) -> IntExpr {
        let mut terms = self.0.into_iter().map(|(monomial, coefficient)| {
            let mut factors = Vec::new();
            if coefficient != BigInt::one() || monomial.is_empty() {
                factors.push(IntExpr::Const(coefficient));
            }
            factors.extend(monomial.into_iter().map(Generator::into_expr));
            factors
                .into_iter()
                .reduce(|lhs, rhs| IntExpr::Mul(Box::new(lhs), Box::new(rhs)))
                .unwrap_or_else(|| IntExpr::Const(BigInt::one()))
        });
        terms
            .next()
            .map(|first| terms.fold(first, |lhs, rhs| IntExpr::Add(Box::new(lhs), Box::new(rhs))))
            .unwrap_or_else(|| IntExpr::Const(BigInt::zero()))
    }
}

impl Generator {
    fn into_expr(self) -> IntExpr {
        match self {
            Self::Var(name) => IntExpr::Var(name),
            Self::Div(lhs, rhs) => IntExpr::Div(Box::new(lhs), Box::new(rhs)),
            Self::RoundDiv(lhs, rhs) => IntExpr::RoundDiv(Box::new(lhs), Box::new(rhs)),
            Self::Log2Ceil(value) => IntExpr::Log2Ceil(Box::new(value)),
        }
    }
}

pub fn euclidean_div_rem(
    numerator: &BigInt,
    denominator: &BigInt,
) -> Result<(BigInt, BigInt), ExprError> {
    if denominator.is_zero() {
        return Err(ExprError::DivisionByZero);
    }
    Ok(numerator.div_mod_floor(&denominator.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_division_rejects_remainder() {
        let expr = IntExpr::Div(Box::new(IntExpr::constant(5)), Box::new(IntExpr::constant(2)));
        assert!(matches!(
            expr.evaluate(&ParamEnv::default()),
            Err(ExprError::InexactDivision { .. })
        ));
    }

    #[test]
    fn round_div_handles_negative_ties() {
        let expr =
            IntExpr::RoundDiv(Box::new(IntExpr::constant(-3)), Box::new(IntExpr::constant(2)));
        assert_eq!(
            expr.evaluate(&ParamEnv::default()).expect("valid expression"),
            BigInt::from(-1)
        );
    }

    #[test]
    fn canonical_polynomial_distributes_and_sorts() {
        let x = IntExpr::Var("x".to_owned());
        let y = IntExpr::Var("y".to_owned());
        let lhs = IntExpr::Mul(
            Box::new(IntExpr::Add(Box::new(x.clone()), Box::new(y.clone()))),
            Box::new(IntExpr::constant(2)),
        );
        let rhs = IntExpr::Add(
            Box::new(IntExpr::Mul(Box::new(IntExpr::constant(2)), Box::new(y))),
            Box::new(IntExpr::Mul(Box::new(IntExpr::constant(2)), Box::new(x))),
        );
        assert_eq!(lhs.canonicalize(), rhs.canonicalize());
    }

    #[test]
    fn euclidean_remainder_is_nonnegative() {
        let (quotient, remainder) =
            euclidean_div_rem(&BigInt::from(-7), &BigInt::from(3)).expect("nonzero divisor");
        assert_eq!(quotient, BigInt::from(-3));
        assert_eq!(remainder, BigInt::from(2));
    }

    #[test]
    fn closed_real_expression_substitutes_variables_inside_square_root_expressions() {
        let expression = RealExpr::Add(
            Box::new(RealExpr::Var("sigma".to_owned())),
            Box::new(RealExpr::Sqrt(Box::new(RealExpr::FromInt(IntExpr::Var(
                "dimension".to_owned(),
            ))))),
        );
        let env = ParamEnv {
            integers: BTreeMap::from([("dimension".to_owned(), BigInt::from(9))]),
            reals: BTreeMap::from([(
                "sigma".to_owned(),
                Rational::new(BigInt::from(13), BigInt::from(2)).expect("rational"),
            )]),
        };
        let closed = expression.close(&env).expect("closed expression");
        assert!(!closed.contains_variable("sigma"));
        assert!(!closed.contains_variable("dimension"));
        assert_eq!(closed.evaluate_f64(&ParamEnv::default()).expect("closed value"), 9.5);
    }

    #[test]
    fn binary64_conversion_preserves_the_exact_rational_value() {
        let six_and_a_half = Rational::from_f64_exact(6.5).expect("finite value");
        assert_eq!(six_and_a_half.numerator(), &BigInt::from(13));
        assert_eq!(six_and_a_half.denominator(), &BigInt::from(2));

        let sigma = RealExpr::from_f64_exact(4.578).expect("finite sigma");
        assert_eq!(
            sigma.evaluate_f64(&ParamEnv::default()).expect("rational sigma").to_bits(),
            4.578f64.to_bits()
        );

        let minimum_subnormal =
            Rational::from_f64_exact(f64::from_bits(1)).expect("finite subnormal");
        assert_eq!(minimum_subnormal.numerator(), &BigInt::one());
        assert_eq!(minimum_subnormal.denominator(), &(BigInt::one() << 1074usize));
    }

    #[test]
    fn binary64_conversion_rejects_non_finite_values() {
        for value in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert!(matches!(RealExpr::from_f64_exact(value), Err(ExprError::NonFiniteReal)));
        }
    }
}
