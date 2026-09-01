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
    LoopIndex(u32),
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
    LoopIndex(u32),
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
            IntExpr::LoopIndex(slot) => Self::LoopIndex(slot),
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
    #[serde(default)]
    pub loop_indices: BTreeMap<u32, BigInt>,
}

/// A deterministic, typed index program used by rank-N family operations.
///
/// Unlike [`IntExpr`], index programs are normalized structurally rather than
/// as algebraic polynomials. This keeps axis positions and scoped loop slots
/// explicit in the frozen IR while still making serialization deterministic.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum IndexExpr {
    Axis(usize),
    Parameter(String),
    LoopIndex(u32),
    Constant(#[serde(with = "serde_support::bigint")] BigInt),
    Add(Box<Self>, Box<Self>),
    Subtract(Box<Self>, Box<Self>),
    Multiply(Box<Self>, Box<Self>),
    Divide(Box<Self>, Box<Self>),
    Remainder(Box<Self>, Box<Self>),
    Equal(Box<Self>, Box<Self>),
    Less(Box<Self>, Box<Self>),
    LessEqual(Box<Self>, Box<Self>),
    Log2Ceil(Box<Self>),
    Select { selector: Box<Self>, branches: Vec<Self> },
}

impl Serialize for IndexExpr {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.normalize().serialize_inner(serializer)
    }
}

impl IndexExpr {
    pub fn constant(value: impl Into<BigInt>) -> Self {
        Self::Constant(value.into())
    }

    /// Resolves parameters and loop slots, then folds concrete arithmetic.
    pub fn evaluate(&self, env: &ParamEnv) -> Result<BigInt, ExprError> {
        match self {
            Self::Axis(axis) => Ok(BigInt::from(*axis)),
            Self::Parameter(name) => env
                .integers
                .get(name)
                .cloned()
                .ok_or_else(|| ExprError::UnboundVariable(name.clone())),
            Self::LoopIndex(slot) => env
                .loop_indices
                .get(slot)
                .cloned()
                .ok_or_else(|| ExprError::UnboundVariable(format!("loop-index[{slot}]"))),
            Self::Constant(value) => Ok(value.clone()),
            Self::Add(lhs, rhs) => Ok(lhs.evaluate(env)? + rhs.evaluate(env)?),
            Self::Subtract(lhs, rhs) => Ok(lhs.evaluate(env)? - rhs.evaluate(env)?),
            Self::Multiply(lhs, rhs) => Ok(lhs.evaluate(env)? * rhs.evaluate(env)?),
            Self::Divide(lhs, rhs) => {
                let denominator = rhs.evaluate(env)?;
                if denominator.is_zero() {
                    return Err(ExprError::DivisionByZero);
                }
                Ok(lhs.evaluate(env)? / denominator)
            }
            Self::Remainder(lhs, rhs) => {
                let denominator = rhs.evaluate(env)?;
                if denominator.is_zero() {
                    return Err(ExprError::DivisionByZero);
                }
                Ok(lhs.evaluate(env)? % denominator)
            }
            Self::Equal(lhs, rhs) => Ok(BigInt::from(lhs.evaluate(env)? == rhs.evaluate(env)?)),
            Self::Less(lhs, rhs) => Ok(BigInt::from(lhs.evaluate(env)? < rhs.evaluate(env)?)),
            Self::LessEqual(lhs, rhs) => Ok(BigInt::from(lhs.evaluate(env)? <= rhs.evaluate(env)?)),
            Self::Log2Ceil(value) => {
                let value = value.evaluate(env)?;
                let value = value.to_biguint().ok_or_else(|| {
                    ExprError::UnboundVariable("log2ceil argument must be positive".into())
                })?;
                if value.is_zero() {
                    return Err(ExprError::UnboundVariable(
                        "log2ceil argument must be positive".into(),
                    ));
                }
                let floor = value.bits() - 1;
                Ok(BigInt::from(if value == (num_bigint::BigUint::one() << floor as usize) {
                    floor
                } else {
                    floor + 1
                }))
            }
            Self::Select { selector, branches } => {
                let index = selector.evaluate(env)?.to_usize().ok_or_else(|| {
                    ExprError::UnboundVariable("index selector is not a nonnegative usize".into())
                })?;
                branches
                    .get(index)
                    .ok_or_else(|| {
                        ExprError::UnboundVariable("index selector out of range".into())
                    })?
                    .evaluate(env)
            }
        }
    }

    /// Performs only fixed structural normalization and constant folding.
    pub fn normalize(&self) -> Self {
        fn fold(expr: &IndexExpr) -> IndexExpr {
            let result = match expr {
                IndexExpr::Axis(axis) => IndexExpr::Axis(*axis),
                IndexExpr::Parameter(name) => IndexExpr::Parameter(name.clone()),
                IndexExpr::LoopIndex(slot) => IndexExpr::LoopIndex(*slot),
                IndexExpr::Constant(value) => IndexExpr::Constant(value.clone()),
                IndexExpr::Add(lhs, rhs) => binary(lhs, rhs, |a, b| a + b, IndexExpr::Add),
                IndexExpr::Subtract(lhs, rhs) => {
                    binary(lhs, rhs, |a, b| a - b, IndexExpr::Subtract)
                }
                IndexExpr::Multiply(lhs, rhs) => {
                    binary(lhs, rhs, |a, b| a * b, IndexExpr::Multiply)
                }
                IndexExpr::Divide(lhs, rhs) => {
                    binary_checked(lhs, rhs, |a, b| a / b, IndexExpr::Divide)
                }
                IndexExpr::Remainder(lhs, rhs) => {
                    binary_checked(lhs, rhs, |a, b| a % b, IndexExpr::Remainder)
                }
                IndexExpr::Equal(lhs, rhs) => {
                    binary(lhs, rhs, |a, b| BigInt::from(a == b), IndexExpr::Equal)
                }
                IndexExpr::Less(lhs, rhs) => {
                    binary(lhs, rhs, |a, b| BigInt::from(a < b), IndexExpr::Less)
                }
                IndexExpr::LessEqual(lhs, rhs) => {
                    binary(lhs, rhs, |a, b| BigInt::from(a <= b), IndexExpr::LessEqual)
                }
                IndexExpr::Log2Ceil(value) => {
                    let value = fold(value);
                    match &value {
                        IndexExpr::Constant(value) if value > &BigInt::zero() => {
                            let bits = value.to_biguint().expect("positive").bits() - 1;
                            IndexExpr::Constant(BigInt::from(
                                if value.to_biguint().as_ref() ==
                                    Some(&(num_bigint::BigUint::one() << bits as usize))
                                {
                                    bits
                                } else {
                                    bits + 1
                                },
                            ))
                        }
                        _ => IndexExpr::Log2Ceil(Box::new(value)),
                    }
                }
                IndexExpr::Select { selector, branches } => {
                    let selector = fold(selector);
                    let branches = branches.iter().map(fold).collect::<Vec<_>>();
                    match &selector {
                        IndexExpr::Constant(index) => index_to_usize(index)
                            .and_then(|index| branches.get(index))
                            .cloned()
                            .unwrap_or(IndexExpr::Select {
                                selector: Box::new(selector),
                                branches,
                            }),
                        _ => IndexExpr::Select { selector: Box::new(selector), branches },
                    }
                }
            };
            result
        }
        fn binary(
            lhs: &IndexExpr,
            rhs: &IndexExpr,
            operation: impl FnOnce(BigInt, BigInt) -> BigInt,
            build: impl FnOnce(Box<IndexExpr>, Box<IndexExpr>) -> IndexExpr,
        ) -> IndexExpr {
            let lhs = fold(lhs);
            let rhs = fold(rhs);
            match (&lhs, &rhs) {
                (IndexExpr::Constant(lhs), IndexExpr::Constant(rhs)) => {
                    IndexExpr::Constant(operation(lhs.clone(), rhs.clone()))
                }
                _ => build(Box::new(lhs), Box::new(rhs)),
            }
        }
        fn binary_checked(
            lhs: &IndexExpr,
            rhs: &IndexExpr,
            operation: impl FnOnce(BigInt, BigInt) -> BigInt,
            build: impl FnOnce(Box<IndexExpr>, Box<IndexExpr>) -> IndexExpr,
        ) -> IndexExpr {
            let lhs = fold(lhs);
            let rhs = fold(rhs);
            match (&lhs, &rhs) {
                (IndexExpr::Constant(lhs), IndexExpr::Constant(rhs)) if !rhs.is_zero() => {
                    IndexExpr::Constant(operation(lhs.clone(), rhs.clone()))
                }
                _ => build(Box::new(lhs), Box::new(rhs)),
            }
        }
        fn index_to_usize(value: &BigInt) -> Option<usize> {
            value.to_usize()
        }
        fold(self)
    }

    fn serialize_inner<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        #[serde(tag = "tag", content = "value")]
        enum Repr<'a> {
            Axis(usize),
            Parameter(&'a str),
            LoopIndex(u32),
            Constant(String),
            Add(Box<Repr<'a>>, Box<Repr<'a>>),
            Subtract(Box<Repr<'a>>, Box<Repr<'a>>),
            Multiply(Box<Repr<'a>>, Box<Repr<'a>>),
            Divide(Box<Repr<'a>>, Box<Repr<'a>>),
            Remainder(Box<Repr<'a>>, Box<Repr<'a>>),
            Equal(Box<Repr<'a>>, Box<Repr<'a>>),
            Less(Box<Repr<'a>>, Box<Repr<'a>>),
            LessEqual(Box<Repr<'a>>, Box<Repr<'a>>),
            Log2Ceil(Box<Repr<'a>>),
            Select { selector: Box<Repr<'a>>, branches: Vec<Repr<'a>> },
        }
        fn repr<'a>(value: &'a IndexExpr) -> Repr<'a> {
            match value {
                IndexExpr::Axis(axis) => Repr::Axis(*axis),
                IndexExpr::Parameter(name) => Repr::Parameter(name),
                IndexExpr::LoopIndex(slot) => Repr::LoopIndex(*slot),
                IndexExpr::Constant(value) => Repr::Constant(value.to_string()),
                IndexExpr::Add(lhs, rhs) => Repr::Add(Box::new(repr(lhs)), Box::new(repr(rhs))),
                IndexExpr::Subtract(lhs, rhs) => {
                    Repr::Subtract(Box::new(repr(lhs)), Box::new(repr(rhs)))
                }
                IndexExpr::Multiply(lhs, rhs) => {
                    Repr::Multiply(Box::new(repr(lhs)), Box::new(repr(rhs)))
                }
                IndexExpr::Divide(lhs, rhs) => {
                    Repr::Divide(Box::new(repr(lhs)), Box::new(repr(rhs)))
                }
                IndexExpr::Remainder(lhs, rhs) => {
                    Repr::Remainder(Box::new(repr(lhs)), Box::new(repr(rhs)))
                }
                IndexExpr::Equal(lhs, rhs) => Repr::Equal(Box::new(repr(lhs)), Box::new(repr(rhs))),
                IndexExpr::Less(lhs, rhs) => Repr::Less(Box::new(repr(lhs)), Box::new(repr(rhs))),
                IndexExpr::LessEqual(lhs, rhs) => {
                    Repr::LessEqual(Box::new(repr(lhs)), Box::new(repr(rhs)))
                }
                IndexExpr::Log2Ceil(value) => Repr::Log2Ceil(Box::new(repr(value))),
                IndexExpr::Select { selector, branches } => Repr::Select {
                    selector: Box::new(repr(selector)),
                    branches: branches.iter().map(repr).collect(),
                },
            }
        }
        repr(self).serialize(serializer)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum IndexExprConversionError {
    #[error("RoundDiv cannot be represented by IndexExpr")]
    RoundDiv,
}

impl TryFrom<IntExpr> for IndexExpr {
    type Error = IndexExprConversionError;

    fn try_from(value: IntExpr) -> Result<Self, Self::Error> {
        match value {
            IntExpr::Const(value) => Ok(Self::Constant(value)),
            IntExpr::Var(name) => Ok(Self::Parameter(name)),
            IntExpr::LoopIndex(slot) => Ok(Self::LoopIndex(slot)),
            IntExpr::Add(lhs, rhs) => {
                Ok(Self::Add(Box::new((*lhs).try_into()?), Box::new((*rhs).try_into()?)))
            }
            IntExpr::Sub(lhs, rhs) => {
                Ok(Self::Subtract(Box::new((*lhs).try_into()?), Box::new((*rhs).try_into()?)))
            }
            IntExpr::Mul(lhs, rhs) => {
                Ok(Self::Multiply(Box::new((*lhs).try_into()?), Box::new((*rhs).try_into()?)))
            }
            IntExpr::Div(lhs, rhs) => {
                Ok(Self::Divide(Box::new((*lhs).try_into()?), Box::new((*rhs).try_into()?)))
            }
            IntExpr::RoundDiv(_, _) => Err(IndexExprConversionError::RoundDiv),
            IntExpr::Log2Ceil(value) => Ok(Self::Log2Ceil(Box::new((*value).try_into()?))),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Deserialize)]
pub struct IndexMap {
    pub input_indices: Vec<IndexExpr>,
}

impl Serialize for IndexMap {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Repr {
            input_indices: Vec<IndexExpr>,
        }
        Repr { input_indices: self.normalize().input_indices }.serialize(serializer)
    }
}

impl IndexMap {
    pub fn new(input_indices: impl Into<Vec<IndexExpr>>) -> Self {
        Self { input_indices: input_indices.into() }
    }

    pub fn normalize(&self) -> Self {
        Self { input_indices: self.input_indices.iter().map(IndexExpr::normalize).collect() }
    }
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
            Self::LoopIndex(slot) => env
                .loop_indices
                .get(slot)
                .cloned()
                .ok_or_else(|| ExprError::UnboundVariable(format!("loop-index[{slot}]"))),
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
            Self::LoopIndex(_) => false,
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

impl From<i32> for IntExpr {
    fn from(value: i32) -> Self {
        Self::constant(value)
    }
}

impl From<i64> for IntExpr {
    fn from(value: i64) -> Self {
        Self::constant(value)
    }
}

impl From<usize> for IntExpr {
    fn from(value: usize) -> Self {
        Self::constant(value)
    }
}

impl From<BigInt> for IntExpr {
    fn from(value: BigInt) -> Self {
        Self::Const(value)
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
    pub fn from_integer(value: impl Into<BigInt>) -> Self {
        Self::Rational(Rational::from_integer(value.into()))
    }
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

impl From<i32> for RealExpr {
    fn from(value: i32) -> Self {
        Self::from_integer(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum Generator {
    Var(String),
    LoopIndex(u32),
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
            IntExpr::LoopIndex(slot) => Self::generator(Generator::LoopIndex(*slot)),
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
            Self::LoopIndex(slot) => IntExpr::LoopIndex(slot),
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
            loop_indices: BTreeMap::new(),
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

    #[test]
    fn index_map_serialization_is_structural_and_normalized() {
        let map = IndexMap::new(vec![IndexExpr::Add(
            Box::new(IndexExpr::constant(1)),
            Box::new(IndexExpr::constant(2)),
        )]);
        let encoded = serde_json::to_string(&map).expect("index map encoding");
        assert!(encoded.contains("\"tag\":\"Constant\""));
        assert!(encoded.contains("\"value\":\"3\""));
    }

    #[test]
    fn index_expr_resolves_parameters_and_scoped_loop_indices() {
        let expression = IndexExpr::Add(
            Box::new(IndexExpr::Parameter("stride".into())),
            Box::new(IndexExpr::LoopIndex(7)),
        );
        let env = ParamEnv {
            integers: BTreeMap::from([("stride".into(), BigInt::from(3))]),
            reals: BTreeMap::new(),
            loop_indices: BTreeMap::from([(7, BigInt::from(4))]),
        };
        assert_eq!(expression.evaluate(&env).expect("index evaluation"), BigInt::from(7));
    }

    #[test]
    fn round_div_conversion_is_rejected_instead_of_truncated() {
        let expression =
            IntExpr::RoundDiv(Box::new(IntExpr::constant(7)), Box::new(IntExpr::constant(2)));
        assert_eq!(IndexExpr::try_from(expression), Err(IndexExprConversionError::RoundDiv));
    }
}
