use crate::serde_support;
use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, btree_map::Entry};
use thiserror::Error;

/// Symbolic integer arithmetic evaluated with explicit parameter bindings.
///
/// `/` is exact division: evaluation rejects a nonzero remainder or a zero
/// denominator. `%` is floor remainder, whose sign follows the denominator;
/// use [`IntExpr::floor_div`] for the corresponding rounded-down quotient.
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
    FloorDiv(Box<Self>, Box<Self>),
    Rem(Box<Self>, Box<Self>),
    RoundDiv(Box<Self>, Box<Self>),
    Log2Ceil(Box<Self>),
    Select { selector: Box<Self>, branches: Vec<Self> },
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
    FloorDiv(Box<Self>, Box<Self>),
    Rem(Box<Self>, Box<Self>),
    RoundDiv(Box<Self>, Box<Self>),
    Log2Ceil(Box<Self>),
    Select { selector: Box<Self>, branches: Vec<Self> },
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
            IntExpr::FloorDiv(lhs, rhs) => {
                Self::FloorDiv(Box::new(Self::from(*lhs)), Box::new(Self::from(*rhs)))
            }
            IntExpr::Rem(lhs, rhs) => {
                Self::Rem(Box::new(Self::from(*lhs)), Box::new(Self::from(*rhs)))
            }
            IntExpr::RoundDiv(lhs, rhs) => {
                Self::RoundDiv(Box::new(Self::from(*lhs)), Box::new(Self::from(*rhs)))
            }
            IntExpr::Log2Ceil(value) => Self::Log2Ceil(Box::new(Self::from(*value))),
            IntExpr::Select { selector, branches } => Self::Select {
                selector: Box::new(Self::from(*selector)),
                branches: branches.into_iter().map(Self::from).collect(),
            },
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

/// Exact symbolic real arithmetic. Operators retain the existing expression
/// tree; integer operands convert to exact rationals without floating-point rounding.
/// Division by zero is rejected when the expression is evaluated.
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
    /// Exact division, rejecting a nonzero remainder.
    Divide(Box<Self>, Box<Self>),
    FloorDivide(Box<Self>, Box<Self>),
    Remainder(Box<Self>, Box<Self>),
    Equal(Box<Self>, Box<Self>),
    Less(Box<Self>, Box<Self>),
    LessEqual(Box<Self>, Box<Self>),
    Log2Ceil(Box<Self>),
    Select {
        selector: Box<Self>,
        branches: Vec<Self>,
    },
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
                let numerator = lhs.evaluate(env)?;
                let (quotient, remainder) = numerator.div_rem(&denominator);
                if !remainder.is_zero() {
                    return Err(ExprError::InexactDivision { numerator, denominator });
                }
                Ok(quotient)
            }
            Self::FloorDivide(lhs, rhs) => {
                let denominator = rhs.evaluate(env)?;
                if denominator.is_zero() {
                    return Err(ExprError::DivisionByZero);
                }
                Ok(lhs.evaluate(env)?.div_floor(&denominator))
            }
            Self::Remainder(lhs, rhs) => {
                let denominator = rhs.evaluate(env)?;
                if denominator.is_zero() {
                    return Err(ExprError::DivisionByZero);
                }
                Ok(lhs.evaluate(env)?.mod_floor(&denominator))
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
                IndexExpr::Divide(lhs, rhs) => binary_checked(
                    lhs,
                    rhs,
                    |a, b| {
                        let (quotient, remainder) = a.div_rem(&b);
                        remainder.is_zero().then_some(quotient)
                    },
                    IndexExpr::Divide,
                ),
                IndexExpr::FloorDivide(lhs, rhs) => {
                    binary_checked(lhs, rhs, |a, b| Some(a.div_floor(&b)), IndexExpr::FloorDivide)
                }
                IndexExpr::Remainder(lhs, rhs) => {
                    binary_checked(lhs, rhs, |a, b| Some(a.mod_floor(&b)), IndexExpr::Remainder)
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
            operation: impl FnOnce(BigInt, BigInt) -> Option<BigInt>,
            build: impl FnOnce(Box<IndexExpr>, Box<IndexExpr>) -> IndexExpr,
        ) -> IndexExpr {
            let lhs = fold(lhs);
            let rhs = fold(rhs);
            match (&lhs, &rhs) {
                (IndexExpr::Constant(lhs), IndexExpr::Constant(rhs)) if !rhs.is_zero() => {
                    match operation(lhs.clone(), rhs.clone()) {
                        Some(value) => IndexExpr::Constant(value),
                        None => build(
                            Box::new(IndexExpr::Constant(lhs.clone())),
                            Box::new(IndexExpr::Constant(rhs.clone())),
                        ),
                    }
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
            FloorDivide(Box<Repr<'a>>, Box<Repr<'a>>),
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
                IndexExpr::FloorDivide(lhs, rhs) => {
                    Repr::FloorDivide(Box::new(repr(lhs)), Box::new(repr(rhs)))
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
            IntExpr::FloorDiv(lhs, rhs) => {
                Ok(Self::FloorDivide(Box::new((*lhs).try_into()?), Box::new((*rhs).try_into()?)))
            }
            IntExpr::Rem(lhs, rhs) => {
                Ok(Self::Remainder(Box::new((*lhs).try_into()?), Box::new((*rhs).try_into()?)))
            }
            IntExpr::RoundDiv(_, _) => Err(IndexExprConversionError::RoundDiv),
            IntExpr::Log2Ceil(value) => Ok(Self::Log2Ceil(Box::new((*value).try_into()?))),
            IntExpr::Select { selector, branches } => Ok(Self::Select {
                selector: Box::new((*selector).try_into()?),
                branches: branches.into_iter().map(TryInto::try_into).collect::<Result<_, _>>()?,
            }),
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

    /// Constructs division rounded toward negative infinity. A zero denominator
    /// remains an evaluation error, just as with exact `/` and floor `%`.
    pub fn floor_div(&self, denominator: impl Into<Self>) -> Self {
        Self::FloorDiv(Box::new(self.clone()), Box::new(denominator.into())).canonicalize()
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
            Self::FloorDiv(lhs, rhs) => {
                let denominator = rhs.evaluate(env)?;
                if denominator.is_zero() {
                    return Err(ExprError::DivisionByZero);
                }
                Ok(lhs.evaluate(env)?.div_floor(&denominator))
            }
            Self::Rem(lhs, rhs) => {
                let denominator = rhs.evaluate(env)?;
                if denominator.is_zero() {
                    return Err(ExprError::DivisionByZero);
                }
                Ok(lhs.evaluate(env)?.mod_floor(&denominator))
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
            Self::Select { selector, branches } => {
                let index = selector.evaluate(env)?.to_usize().ok_or_else(|| {
                    ExprError::UnboundVariable("integer selector is not a nonnegative usize".into())
                })?;
                branches
                    .get(index)
                    .ok_or_else(|| {
                        ExprError::UnboundVariable("integer selector out of range".into())
                    })?
                    .evaluate(env)
            }
        }
    }

    /// Returns the polynomial normal form over opaque generators, retaining evaluation of
    /// partial operations even when their coefficients cancel. A retained zero product prevents
    /// normalization and serialization from erasing division or other domain errors.
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
            Self::FloorDiv(lhs, rhs) |
            Self::Rem(lhs, rhs) |
            Self::RoundDiv(lhs, rhs) => {
                lhs.contains_variable(variable) || rhs.contains_variable(variable)
            }
            Self::Log2Ceil(value) => value.contains_variable(variable),
            Self::Select { selector, branches } => {
                selector.contains_variable(variable) ||
                    branches.iter().any(|branch| branch.contains_variable(variable))
            }
        }
    }
}

impl From<&IntExpr> for IntExpr {
    fn from(value: &IntExpr) -> Self {
        value.clone()
    }
}

macro_rules! integer_expression_operator {
    ($trait:ident, $method:ident, $variant:ident) => {
        impl<Rhs: Into<IntExpr>> std::ops::$trait<Rhs> for IntExpr {
            type Output = IntExpr;

            fn $method(self, rhs: Rhs) -> Self::Output {
                IntExpr::$variant(Box::new(self), Box::new(rhs.into())).canonicalize()
            }
        }

        impl<Rhs: Into<IntExpr>> std::ops::$trait<Rhs> for &IntExpr {
            type Output = IntExpr;

            fn $method(self, rhs: Rhs) -> Self::Output {
                IntExpr::$variant(Box::new(self.clone()), Box::new(rhs.into())).canonicalize()
            }
        }
    };
}

// Ordinary arithmetic produces the same canonical symbolic polynomial as
// explicit Add/Sub/Mul nodes. Borrowed operands remain available to the caller.
integer_expression_operator!(Add, add, Add);
integer_expression_operator!(Sub, sub, Sub);
integer_expression_operator!(Mul, mul, Mul);
integer_expression_operator!(Div, div, Div);
integer_expression_operator!(Rem, rem, Rem);

impl std::ops::Neg for IntExpr {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::constant(0) - self
    }
}

impl std::ops::Neg for &IntExpr {
    type Output = IntExpr;

    fn neg(self) -> Self::Output {
        IntExpr::constant(0) - self
    }
}

macro_rules! integer_expression_from_primitive {
    ($($primitive:ty),* $(,)?) => {
        $(impl From<$primitive> for IntExpr {
            fn from(value: $primitive) -> Self {
                Self::constant(value)
            }
        })*
    };
}

integer_expression_from_primitive!(i8, i16, i128, isize, u8, u16, u32, u64, u128);

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

impl From<BigUint> for IntExpr {
    fn from(value: BigUint) -> Self {
        Self::Const(value.into())
    }
}

impl From<&BigInt> for IntExpr {
    fn from(value: &BigInt) -> Self {
        Self::Const(value.clone())
    }
}

impl From<&BigUint> for IntExpr {
    fn from(value: &BigUint) -> Self {
        Self::Const(value.clone().into())
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

impl From<&RealExpr> for RealExpr {
    fn from(value: &RealExpr) -> Self {
        value.clone()
    }
}

impl From<IntExpr> for RealExpr {
    fn from(value: IntExpr) -> Self {
        Self::FromInt(value)
    }
}

impl From<&IntExpr> for RealExpr {
    fn from(value: &IntExpr) -> Self {
        Self::FromInt(value.clone())
    }
}

impl From<Rational> for RealExpr {
    fn from(value: Rational) -> Self {
        Self::Rational(value)
    }
}

impl From<&Rational> for RealExpr {
    fn from(value: &Rational) -> Self {
        Self::Rational(value.clone())
    }
}

macro_rules! real_expression_from_integer {
    ($($integer:ty),* $(,)?) => {
        $(impl From<$integer> for RealExpr {
            fn from(value: $integer) -> Self {
                Self::from_integer(value)
            }
        })*
    };
}

// Integers convert exactly. Floating-point callers must continue to opt into
// `from_f64_exact`, which rejects non-finite values and preserves binary64 exactly.
real_expression_from_integer!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, BigInt, BigUint
);

impl From<&BigInt> for RealExpr {
    fn from(value: &BigInt) -> Self {
        Self::from_integer(value.clone())
    }
}

impl From<&BigUint> for RealExpr {
    fn from(value: &BigUint) -> Self {
        Self::from_integer(value.clone())
    }
}

macro_rules! real_expression_operator {
    ($trait:ident, $method:ident, $variant:ident) => {
        impl<Rhs: Into<RealExpr>> std::ops::$trait<Rhs> for RealExpr {
            type Output = RealExpr;

            fn $method(self, rhs: Rhs) -> Self::Output {
                RealExpr::$variant(Box::new(self), Box::new(rhs.into()))
            }
        }

        impl<Rhs: Into<RealExpr>> std::ops::$trait<Rhs> for &RealExpr {
            type Output = RealExpr;

            fn $method(self, rhs: Rhs) -> Self::Output {
                RealExpr::$variant(Box::new(self.clone()), Box::new(rhs.into()))
            }
        }
    };
}

real_expression_operator!(Add, add, Add);
real_expression_operator!(Sub, sub, Sub);
real_expression_operator!(Mul, mul, Mul);
real_expression_operator!(Div, div, Div);

impl std::ops::Neg for RealExpr {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::from_integer(0) - self
    }
}

impl std::ops::Neg for &RealExpr {
    type Output = RealExpr;

    fn neg(self) -> Self::Output {
        RealExpr::from_integer(0) - self
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum Generator {
    Var(String),
    LoopIndex(u32),
    Div(IntExpr, IntExpr),
    FloorDiv(IntExpr, IntExpr),
    Rem(IntExpr, IntExpr),
    RoundDiv(IntExpr, IntExpr),
    Log2Ceil(IntExpr),
    Select { selector: IntExpr, branches: Vec<IntExpr> },
}

type Monomial = Vec<Generator>;

#[derive(Clone, Debug, Default)]
struct Polynomial {
    terms: BTreeMap<Monomial, BigInt>,
    required: BTreeSet<Generator>,
}

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
            IntExpr::FloorDiv(lhs, rhs) => {
                Self::generator(Generator::FloorDiv(lhs.canonicalize(), rhs.canonicalize()))
            }
            IntExpr::Rem(lhs, rhs) => {
                Self::generator(Generator::Rem(lhs.canonicalize(), rhs.canonicalize()))
            }
            IntExpr::RoundDiv(lhs, rhs) => {
                Self::generator(Generator::RoundDiv(lhs.canonicalize(), rhs.canonicalize()))
            }
            IntExpr::Log2Ceil(value) => Self::generator(Generator::Log2Ceil(value.canonicalize())),
            IntExpr::Select { selector, branches } => Self::generator(Generator::Select {
                selector: selector.canonicalize(),
                branches: branches.iter().map(IntExpr::canonicalize).collect(),
            }),
        }
    }

    fn constant(value: BigInt) -> Self {
        if value.is_zero() {
            Self::default()
        } else {
            Self { terms: BTreeMap::from([(Vec::new(), value)]), required: BTreeSet::new() }
        }
    }

    fn generator(generator: Generator) -> Self {
        // Variables are polynomial indeterminates with bindings supplied by the caller. Other
        // generators are partial even with complete bindings, and must still be evaluated if
        // algebraic cancellation removes their numeric contribution. Select stays opaque so
        // only its selected branch is evaluated.
        let required = if matches!(generator, Generator::Var(_) | Generator::LoopIndex(_)) {
            BTreeSet::new()
        } else {
            BTreeSet::from([generator.clone()])
        };
        Self { terms: BTreeMap::from([(vec![generator], BigInt::one())]), required }
    }

    fn add(mut self, rhs: Self) -> Self {
        self.required.extend(rhs.required);
        for (monomial, coefficient) in rhs.terms {
            match self.terms.entry(monomial) {
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
        for coefficient in rhs.terms.values_mut() {
            *coefficient = -coefficient.clone();
        }
        self.add(rhs)
    }

    fn mul(self, rhs: Self) -> Self {
        let mut required = self.required;
        required.extend(rhs.required);
        let mut output = Self { terms: BTreeMap::new(), required };
        for (lhs_monomial, lhs_coefficient) in self.terms {
            for (rhs_monomial, rhs_coefficient) in &rhs.terms {
                let mut monomial = lhs_monomial.clone();
                monomial.extend(rhs_monomial.iter().cloned());
                monomial.sort();
                let coefficient = &lhs_coefficient * rhs_coefficient;
                output = output.add(Self {
                    terms: BTreeMap::from([(monomial, coefficient)]),
                    required: BTreeSet::new(),
                });
            }
        }
        output
    }

    fn into_expr(mut self) -> IntExpr {
        let retained = self.terms.keys().flatten().cloned().collect::<BTreeSet<_>>();
        for generator in self.required.difference(&retained) {
            self.terms.insert(vec![generator.clone()], BigInt::zero());
        }
        let mut terms = self.terms.into_iter().map(|(monomial, coefficient)| {
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
            Self::FloorDiv(lhs, rhs) => IntExpr::FloorDiv(Box::new(lhs), Box::new(rhs)),
            Self::Rem(lhs, rhs) => IntExpr::Rem(Box::new(lhs), Box::new(rhs)),
            Self::RoundDiv(lhs, rhs) => IntExpr::RoundDiv(Box::new(lhs), Box::new(rhs)),
            Self::Log2Ceil(value) => IntExpr::Log2Ceil(Box::new(value)),
            Self::Select { selector, branches } => {
                IntExpr::Select { selector: Box::new(selector), branches }
            }
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
    fn test_integer_division_remainder_and_negation_preserve_existing_semantics() {
        let numerator = IntExpr::Var("numerator".to_owned());
        let denominator = IntExpr::Var("denominator".to_owned());
        let quotient = IntExpr::Div(Box::new(numerator.clone()), Box::new(denominator.clone()));
        let remainder = IntExpr::Rem(Box::new(numerator.clone()), Box::new(denominator.clone()));
        let floor = IntExpr::FloorDiv(Box::new(numerator.clone()), Box::new(denominator.clone()));
        let negative = IntExpr::Sub(Box::new(IntExpr::constant(0)), Box::new(numerator.clone()));
        for (n, d) in [(12, 3), (-12, 3), (12, -3), (-7, 3), (7, -3), (7, 0)] {
            let env = ParamEnv {
                integers: BTreeMap::from([
                    ("numerator".to_owned(), BigInt::from(n)),
                    ("denominator".to_owned(), BigInt::from(d)),
                ]),
                ..ParamEnv::default()
            };
            for expression in
                [&numerator / &denominator, numerator.clone() / denominator.clone(), &numerator / d]
            {
                assert_eq!(expression.evaluate(&env), quotient.evaluate(&env));
            }
            for expression in
                [&numerator % &denominator, numerator.clone() % denominator.clone(), &numerator % d]
            {
                assert_eq!(expression.evaluate(&env), remainder.evaluate(&env));
            }
            assert_eq!(numerator.floor_div(&denominator).evaluate(&env), floor.evaluate(&env));
            assert_eq!((-&numerator).evaluate(&env), negative.evaluate(&env));
            assert_eq!((-numerator.clone()).evaluate(&env), negative.evaluate(&env));
        }
        assert!(matches!(
            (IntExpr::constant(7) / 3).evaluate(&ParamEnv::default()),
            Err(ExprError::InexactDivision { .. })
        ));
        assert_eq!(
            (IntExpr::constant(7) / 0).evaluate(&ParamEnv::default()),
            Err(ExprError::DivisionByZero)
        );
        assert_eq!(
            (IntExpr::constant(7) % 0).evaluate(&ParamEnv::default()),
            Err(ExprError::DivisionByZero)
        );
    }

    #[test]
    fn test_canonicalization_preserves_partial_operation_errors_after_cancellation() {
        let denominator = IntExpr::Var("denominator".to_owned());
        let partials = [
            IntExpr::constant(1) / 0,
            IntExpr::constant(1) / 2,
            IntExpr::constant(1) % 0,
            IntExpr::constant(1).floor_div(0),
            IntExpr::constant(1) / &denominator,
            IntExpr::constant(1) % &denominator,
            IntExpr::constant(1).floor_div(&denominator),
            IntExpr::RoundDiv(Box::new(IntExpr::constant(1)), Box::new(denominator.clone())),
            IntExpr::Log2Ceil(Box::new(denominator.clone())),
        ];
        for divisor in [-1, 0, 1, 2] {
            let env = ParamEnv {
                integers: BTreeMap::from([("denominator".to_owned(), BigInt::from(divisor))]),
                ..ParamEnv::default()
            };
            for partial in &partials {
                let raw = [
                    IntExpr::Sub(Box::new(partial.clone()), Box::new(partial.clone())),
                    IntExpr::Mul(Box::new(partial.clone()), Box::new(IntExpr::constant(0))),
                    IntExpr::Mul(Box::new(IntExpr::constant(0)), Box::new(partial.clone())),
                ];
                let operators = [partial - partial, partial * 0, IntExpr::constant(0) * partial];
                for (raw, operator) in raw.into_iter().zip(operators) {
                    let expected = raw.evaluate(&env);
                    let canonical = raw.canonicalize();
                    assert_eq!(canonical.evaluate(&env), expected);
                    assert_eq!(operator.evaluate(&env), expected);
                    assert_eq!(canonical.canonicalize(), canonical);
                    let serialized = serde_json::to_vec(&raw).unwrap();
                    let decoded: IntExpr = serde_json::from_slice(&serialized).unwrap();
                    assert_eq!(decoded.evaluate(&env), expected);
                }
            }
        }
    }

    #[test]
    fn test_canonicalization_preserves_partial_selector_laziness() {
        let selection = IntExpr::Select {
            selector: Box::new(IntExpr::Var("selector".to_owned())),
            branches: vec![IntExpr::constant(9), IntExpr::constant(1) / 0],
        };
        let raw = IntExpr::Sub(Box::new(selection.clone()), Box::new(selection.clone()));
        let canonical = raw.canonicalize();
        for selector in [-1, 0, 1, 2] {
            let env = ParamEnv {
                integers: BTreeMap::from([("selector".to_owned(), BigInt::from(selector))]),
                ..ParamEnv::default()
            };
            assert_eq!(canonical.evaluate(&env), raw.evaluate(&env));
        }
    }

    #[test]
    fn test_real_operators_preserve_exact_rational_arithmetic() {
        let left = RealExpr::Var("left".to_owned());
        let right = RealExpr::Var("right".to_owned());
        let expected = RealExpr::Div(
            Box::new(RealExpr::Mul(
                Box::new(RealExpr::Sub(
                    Box::new(RealExpr::Add(Box::new(left.clone()), Box::new(right.clone()))),
                    Box::new(RealExpr::from_integer(2)),
                )),
                Box::new(RealExpr::Sub(
                    Box::new(RealExpr::from_integer(0)),
                    Box::new(left.clone()),
                )),
            )),
            Box::new(right.clone()),
        );
        let expressions = [
            ((&left + &right) - 2) * (-&left) / &right,
            ((left.clone() + &right) - 2usize) * (-left.clone()) / right.clone(),
            ((&left + right.clone()) - RealExpr::from_integer(2)) * (-&left) / &right,
        ];
        for (n, d) in [(5, 3), (-7, 2), (0, 1)] {
            let env = ParamEnv {
                reals: BTreeMap::from([
                    ("left".to_owned(), Rational::new(n.into(), d.into()).unwrap()),
                    ("right".to_owned(), Rational::new((-3).into(), 7.into()).unwrap()),
                ]),
                ..ParamEnv::default()
            };
            for expression in &expressions {
                assert_eq!(expression.evaluate_rational(&env), expected.evaluate_rational(&env));
                assert_eq!(expression.evaluate_f64(&env), expected.evaluate_f64(&env));
            }
        }
        let huge = u128::MAX;
        let exact = (RealExpr::from(huge) + 1u64).evaluate_rational(&ParamEnv::default()).unwrap();
        assert_eq!(exact, Rational::from_integer(BigInt::from(huge) + 1));
        let invalid = &left / 0;
        let reference = RealExpr::Div(Box::new(left.clone()), Box::new(RealExpr::from_integer(0)));
        let env = ParamEnv {
            reals: BTreeMap::from([("left".to_owned(), Rational::from_integer(1.into()))]),
            ..ParamEnv::default()
        };
        assert_eq!(invalid.evaluate_rational(&env), reference.evaluate_rational(&env));
        assert_eq!(invalid.evaluate_f64(&env), Err(ExprError::DivisionByZero));
    }

    #[test]
    fn test_arithmetic_operators_preserve_symbolic_evaluation_and_ownership() {
        let start = IntExpr::Var("start".to_owned());
        let columns = IntExpr::Var("columns".to_owned());
        let expected = IntExpr::Sub(
            Box::new(IntExpr::Mul(
                Box::new(IntExpr::Add(Box::new(start.clone()), Box::new(columns.clone()))),
                Box::new(IntExpr::constant(3)),
            )),
            Box::new(columns.clone()),
        );
        let expressions = [
            (&start + &columns) * 3 - &columns,
            (start.clone() + &columns) * 3usize - columns.clone(),
            (&start + columns.clone()) * 3u64 - &columns,
            (start.clone() + columns.clone()) * IntExpr::constant(3) - columns.clone(),
        ];
        for (start_value, columns_value) in [(-7, 2), (0, 0), (11, -3)] {
            let env = ParamEnv {
                integers: BTreeMap::from([
                    ("start".to_owned(), BigInt::from(start_value)),
                    ("columns".to_owned(), BigInt::from(columns_value)),
                ]),
                ..ParamEnv::default()
            };
            let result = expected.evaluate(&env).unwrap();
            for expression in &expressions {
                assert_eq!(expression.evaluate(&env).unwrap(), result);
                assert_eq!(expression, &expected.canonicalize());
            }
        }
        assert_eq!(&start - &start, IntExpr::constant(0));
        assert_eq!(&columns * 1u32, columns);
        assert_eq!(&start + 0i16, start);
    }

    #[test]
    fn index_conversion_preserves_division_and_remainder_semantics() {
        let env = ParamEnv::default();
        for numerator in [-5, -4, -3, 0, 3, 4, 5] {
            for denominator in [-2, 0, 2] {
                let lhs = Box::new(IntExpr::constant(numerator));
                let rhs = Box::new(IntExpr::constant(denominator));
                for expression in [
                    IntExpr::Div(lhs.clone(), rhs.clone()),
                    IntExpr::FloorDiv(lhs.clone(), rhs.clone()),
                    IntExpr::Rem(lhs, rhs),
                ] {
                    let expected = expression.evaluate(&env);
                    let index = IndexExpr::try_from(expression).unwrap();
                    assert_eq!(index.evaluate(&env), expected);
                    assert_eq!(index.normalize().evaluate(&env), expected);
                    let encoded = serde_json::to_vec(&index).unwrap();
                    let decoded: IndexExpr = serde_json::from_slice(&encoded).unwrap();
                    assert_eq!(decoded.evaluate(&env), expected);
                }
            }
        }
    }

    #[test]
    fn symbolic_index_division_preserves_operator_identity_and_errors() {
        let env = ParamEnv {
            loop_indices: BTreeMap::from([(0, BigInt::from(-3))]),
            ..ParamEnv::default()
        };
        let lhs = Box::new(IntExpr::LoopIndex(0));
        let rhs = Box::new(IntExpr::constant(2));
        let exact = IndexExpr::try_from(IntExpr::Div(lhs.clone(), rhs.clone())).unwrap();
        let floor = IndexExpr::try_from(IntExpr::FloorDiv(lhs, rhs)).unwrap();
        assert_ne!(serde_json::to_vec(&exact).unwrap(), serde_json::to_vec(&floor).unwrap());
        for index in [exact, floor] {
            let encoded = serde_json::to_vec(&index).unwrap();
            let decoded: IndexExpr = serde_json::from_slice(&encoded).unwrap();
            assert_eq!(decoded.evaluate(&env), index.evaluate(&env));
        }
    }

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

    #[test]
    fn integer_select_round_trips_and_evaluates_the_selected_branch() {
        let expression = IntExpr::Select {
            selector: Box::new(IntExpr::LoopIndex(7)),
            branches: vec![IntExpr::constant(3), IntExpr::constant(11)],
        };
        let encoded = serde_json::to_vec(&expression).expect("select expression encoding");
        let decoded: IntExpr =
            serde_json::from_slice(&encoded).expect("select expression decoding");
        let env = ParamEnv {
            loop_indices: BTreeMap::from([(7, BigInt::from(1))]),
            ..ParamEnv::default()
        };
        assert_eq!(decoded.evaluate(&env).expect("selected branch"), BigInt::from(11));
    }

    #[test]
    fn integer_select_rejects_an_out_of_range_selector() {
        let expression = IntExpr::Select {
            selector: Box::new(IntExpr::constant(2)),
            branches: vec![IntExpr::constant(3), IntExpr::constant(11)],
        };
        assert!(expression.evaluate(&ParamEnv::default()).is_err());
    }

    #[test]
    fn floor_division_and_remainder_support_structural_index_arithmetic() {
        let numerator = IntExpr::LoopIndex(3);
        let denominator = IntExpr::constant(25);
        let quotient =
            IntExpr::FloorDiv(Box::new(numerator.clone()), Box::new(denominator.clone()));
        let remainder = IntExpr::Rem(Box::new(numerator), Box::new(denominator));
        let env = ParamEnv {
            loop_indices: BTreeMap::from([(3, BigInt::from(63))]),
            ..ParamEnv::default()
        };
        assert_eq!(quotient.evaluate(&env).expect("floor quotient"), BigInt::from(2));
        assert_eq!(remainder.evaluate(&env).expect("remainder"), BigInt::from(13));
        assert_eq!(
            IndexExpr::try_from(quotient).expect("index quotient").evaluate(&env).unwrap(),
            BigInt::from(2)
        );
        assert_eq!(
            IndexExpr::try_from(remainder).expect("index remainder").evaluate(&env).unwrap(),
            BigInt::from(13)
        );
    }
}
