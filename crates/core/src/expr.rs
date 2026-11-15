//! Compile expressions: integers known before a program runs.
//!
//! An [`IntExpr`] is built from constants, named parameters, loop indices, arithmetic,
//! `Log2Ceil`, and `Select`. It has no ring-specific constructors: a DSL resolves those itself
//! before it hands a program to the core. Values are `i128`, and every operation is checked.
//!
//! [`IntExpr::LoopIndex`] refers to the index of an enclosing loop by its slot. Inside a family
//! element type, the slot [`MEMBER_INDEX`] is the member's index. An expression is evaluated in an
//! [`Env`]: the parameter values and the values of the loop indices currently in scope. During
//! validation a loop index has no value; an expression that needs one there fails with
//! [`ExprError::IndexDependent`], which is how the core keeps types independent of loop indices.

use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, ops, sync::Arc};

/// The loop-index slot that denotes a family member's index inside a family element type.
pub const MEMBER_INDEX: u32 = u32::MAX;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntExpr {
    Const(i128),
    Var(String),
    LoopIndex(u32),
    Add(Box<IntExpr>, Box<IntExpr>),
    Sub(Box<IntExpr>, Box<IntExpr>),
    Mul(Box<IntExpr>, Box<IntExpr>),
    /// Division rounding towards negative infinity.
    FloorDiv(Box<IntExpr>, Box<IntExpr>),
    /// The remainder of [`IntExpr::FloorDiv`]: it has the sign of the divisor.
    Rem(Box<IntExpr>, Box<IntExpr>),
    /// `ceil(log2(x))` for `x >= 1`.
    Log2Ceil(Box<IntExpr>),
    /// `branches[selector]`.
    Select {
        selector: Box<IntExpr>,
        branches: Vec<IntExpr>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ExprError {
    #[error("parameter `{0}` has no value")]
    Unbound(String),
    #[error("the expression depends on loop index {0}, which has no value here")]
    IndexDependent(u32),
    #[error("integer overflow")]
    Overflow,
    #[error("division by zero")]
    DivisionByZero,
    #[error("Log2Ceil of {0}, which is not positive")]
    Log2CeilOfNonPositive(i128),
    #[error("selector {selector} is outside 0..{branches}")]
    SelectorOutOfRange { selector: i128, branches: usize },
}

/// Parameter values.
pub type Bindings = BTreeMap<String, i128>;

/// Where an expression is evaluated: parameter values and the loop indices in scope.
#[derive(Clone, Debug, Default)]
pub struct Env {
    pub vars: Arc<Bindings>,
    pub indices: Vec<(u32, i128)>,
}

impl Env {
    pub fn new(vars: Bindings) -> Self {
        Self { vars: Arc::new(vars), indices: Vec::new() }
    }

    /// This environment with loop index `slot` bound to `value`.
    pub fn with_index(&self, slot: u32, value: i128) -> Self {
        let mut indices = self.indices.clone();
        indices.retain(|(s, _)| *s != slot);
        indices.push((slot, value));
        Self { vars: self.vars.clone(), indices }
    }

    /// This environment with more parameters, which shadow existing ones.
    pub fn with_vars(&self, extra: impl IntoIterator<Item = (String, i128)>) -> Self {
        let mut vars = (*self.vars).clone();
        vars.extend(extra);
        Self { vars: Arc::new(vars), indices: self.indices.clone() }
    }

    pub fn index(&self, slot: u32) -> Option<i128> {
        self.indices.iter().find(|(s, _)| *s == slot).map(|(_, v)| *v)
    }
}

/// A closed interval of integers, for bounding expressions over ranges of loop indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Interval {
    pub min: i128,
    pub max: i128,
}

impl Interval {
    pub fn point(value: i128) -> Self {
        Self { min: value, max: value }
    }
}

impl IntExpr {
    pub fn constant(value: impl Into<i128>) -> Self {
        Self::Const(value.into())
    }

    pub fn var(name: impl Into<String>) -> Self {
        Self::Var(name.into())
    }

    pub fn floor_div(self, divisor: impl Into<IntExpr>) -> Self {
        Self::FloorDiv(Box::new(self), Box::new(divisor.into()))
    }

    pub fn floor_rem(self, divisor: impl Into<IntExpr>) -> Self {
        Self::Rem(Box::new(self), Box::new(divisor.into()))
    }

    pub fn log2_ceil(self) -> Self {
        Self::Log2Ceil(Box::new(self))
    }

    pub fn eval(&self, env: &Env) -> Result<i128, ExprError> {
        let binary = |a: &IntExpr, b: &IntExpr| Ok::<_, ExprError>((a.eval(env)?, b.eval(env)?));
        match self {
            Self::Const(v) => Ok(*v),
            Self::Var(name) => {
                env.vars.get(name).copied().ok_or_else(|| ExprError::Unbound(name.clone()))
            }
            Self::LoopIndex(slot) => env.index(*slot).ok_or(ExprError::IndexDependent(*slot)),
            Self::Add(a, b) => {
                let (a, b) = binary(a, b)?;
                a.checked_add(b).ok_or(ExprError::Overflow)
            }
            Self::Sub(a, b) => {
                let (a, b) = binary(a, b)?;
                a.checked_sub(b).ok_or(ExprError::Overflow)
            }
            Self::Mul(a, b) => {
                let (a, b) = binary(a, b)?;
                a.checked_mul(b).ok_or(ExprError::Overflow)
            }
            Self::FloorDiv(a, b) => {
                let (a, b) = binary(a, b)?;
                floor_div(a, b)
            }
            Self::Rem(a, b) => {
                let (a, b) = binary(a, b)?;
                floor_rem(a, b)
            }
            Self::Log2Ceil(a) => log2_ceil(a.eval(env)?),
            Self::Select { selector, branches } => {
                let selector = selector.eval(env)?;
                let branch = usize::try_from(selector)
                    .ok()
                    .and_then(|i| branches.get(i))
                    .ok_or(ExprError::SelectorOutOfRange { selector, branches: branches.len() })?;
                branch.eval(env)
            }
        }
    }

    /// Whether the expression reads loop index `slot`.
    pub fn uses_index(&self, slot: u32) -> bool {
        match self {
            Self::Const(_) | Self::Var(_) => false,
            Self::LoopIndex(s) => *s == slot,
            Self::Add(a, b) |
            Self::Sub(a, b) |
            Self::Mul(a, b) |
            Self::FloorDiv(a, b) |
            Self::Rem(a, b) => a.uses_index(slot) || b.uses_index(slot),
            Self::Log2Ceil(a) => a.uses_index(slot),
            Self::Select { selector, branches } => {
                selector.uses_index(slot) || branches.iter().any(|b| b.uses_index(slot))
            }
        }
    }

    /// Whether the expression reads any loop index other than [`MEMBER_INDEX`].
    pub fn uses_any_index(&self) -> bool {
        match self {
            Self::Const(_) | Self::Var(_) => false,
            Self::LoopIndex(s) => *s != MEMBER_INDEX,
            Self::Add(a, b) |
            Self::Sub(a, b) |
            Self::Mul(a, b) |
            Self::FloorDiv(a, b) |
            Self::Rem(a, b) => a.uses_any_index() || b.uses_any_index(),
            Self::Log2Ceil(a) => a.uses_any_index(),
            Self::Select { selector, branches } => {
                selector.uses_any_index() || branches.iter().any(IntExpr::uses_any_index)
            }
        }
    }

    /// This expression with every `LoopIndex(slot)` replaced by `value`.
    pub fn substitute_index(&self, slot: u32, value: &IntExpr) -> Self {
        let sub = |e: &IntExpr| Box::new(e.substitute_index(slot, value));
        match self {
            Self::LoopIndex(s) if *s == slot => value.clone(),
            Self::Const(_) | Self::Var(_) | Self::LoopIndex(_) => self.clone(),
            Self::Add(a, b) => Self::Add(sub(a), sub(b)),
            Self::Sub(a, b) => Self::Sub(sub(a), sub(b)),
            Self::Mul(a, b) => Self::Mul(sub(a), sub(b)),
            Self::FloorDiv(a, b) => Self::FloorDiv(sub(a), sub(b)),
            Self::Rem(a, b) => Self::Rem(sub(a), sub(b)),
            Self::Log2Ceil(a) => Self::Log2Ceil(sub(a)),
            Self::Select { selector, branches } => Self::Select {
                selector: sub(selector),
                branches: branches.iter().map(|b| b.substitute_index(slot, value)).collect(),
            },
        }
    }

    /// This expression with every parameter in `values` replaced by its value.
    pub fn substitute_vars(&self, values: &BTreeMap<String, IntExpr>) -> Self {
        let sub = |e: &IntExpr| Box::new(e.substitute_vars(values));
        match self {
            Self::Var(name) => values.get(name).cloned().unwrap_or_else(|| self.clone()),
            Self::Const(_) | Self::LoopIndex(_) => self.clone(),
            Self::Add(a, b) => Self::Add(sub(a), sub(b)),
            Self::Sub(a, b) => Self::Sub(sub(a), sub(b)),
            Self::Mul(a, b) => Self::Mul(sub(a), sub(b)),
            Self::FloorDiv(a, b) => Self::FloorDiv(sub(a), sub(b)),
            Self::Rem(a, b) => Self::Rem(sub(a), sub(b)),
            Self::Log2Ceil(a) => Self::Log2Ceil(sub(a)),
            Self::Select { selector, branches } => Self::Select {
                selector: sub(selector),
                branches: branches.iter().map(|b| b.substitute_vars(values)).collect(),
            },
        }
    }

    /// An interval containing every value of the expression when each loop index `slot` ranges
    /// over `ranges[slot]` and other loop indices take their value in `env`.
    pub fn bound(&self, env: &Env, ranges: &[(u32, Interval)]) -> Result<Interval, ExprError> {
        let pair = |a: &IntExpr, b: &IntExpr| {
            Ok::<_, ExprError>((a.bound(env, ranges)?, b.bound(env, ranges)?))
        };
        let corners = |a: Interval,
                       b: Interval,
                       f: &dyn Fn(i128, i128) -> Result<i128, ExprError>| {
            let values = [f(a.min, b.min)?, f(a.min, b.max)?, f(a.max, b.min)?, f(a.max, b.max)?];
            Ok(Interval { min: *values.iter().min().unwrap(), max: *values.iter().max().unwrap() })
        };
        match self {
            Self::LoopIndex(slot) => match ranges.iter().find(|(s, _)| s == slot) {
                Some((_, range)) => Ok(*range),
                None => self.eval(env).map(Interval::point),
            },
            Self::Const(_) | Self::Var(_) => self.eval(env).map(Interval::point),
            Self::Add(a, b) => {
                let (a, b) = pair(a, b)?;
                corners(a, b, &|x, y| x.checked_add(y).ok_or(ExprError::Overflow))
            }
            Self::Sub(a, b) => {
                let (a, b) = pair(a, b)?;
                corners(a, b, &|x, y| x.checked_sub(y).ok_or(ExprError::Overflow))
            }
            Self::Mul(a, b) => {
                let (a, b) = pair(a, b)?;
                corners(a, b, &|x, y| x.checked_mul(y).ok_or(ExprError::Overflow))
            }
            Self::FloorDiv(a, b) | Self::Rem(a, b) => {
                let (x, y) = pair(a, b)?;
                if y.min <= 0 && y.max >= 0 {
                    return Err(ExprError::DivisionByZero);
                }
                if let Self::Rem(..) = self {
                    // The remainder lies between 0 and the divisor, exclusive.
                    return Ok(if y.min > 0 {
                        Interval { min: 0, max: y.max - 1 }
                    } else {
                        Interval { min: y.min + 1, max: 0 }
                    });
                }
                corners(x, y, &floor_div)
            }
            Self::Log2Ceil(a) => {
                let a = a.bound(env, ranges)?;
                Ok(Interval { min: log2_ceil(a.min)?, max: log2_ceil(a.max)? })
            }
            Self::Select { selector, branches } => {
                let s = selector.bound(env, ranges)?;
                let mut result: Option<Interval> = None;
                for i in s.min..=s.max {
                    let branch = usize::try_from(i).ok().and_then(|i| branches.get(i)).ok_or(
                        ExprError::SelectorOutOfRange { selector: i, branches: branches.len() },
                    )?;
                    let b = branch.bound(env, ranges)?;
                    result = Some(match result {
                        Some(r) => Interval { min: r.min.min(b.min), max: r.max.max(b.max) },
                        None => b,
                    });
                }
                Ok(result.expect("a selector interval is never empty"))
            }
        }
    }
}

pub fn floor_div(a: i128, b: i128) -> Result<i128, ExprError> {
    if b == 0 {
        return Err(ExprError::DivisionByZero);
    }
    let q = a.checked_div(b).ok_or(ExprError::Overflow)?;
    Ok(if a % b != 0 && ((a < 0) != (b < 0)) { q - 1 } else { q })
}

pub fn floor_rem(a: i128, b: i128) -> Result<i128, ExprError> {
    if b == 0 {
        return Err(ExprError::DivisionByZero);
    }
    let r = a.checked_rem(b).ok_or(ExprError::Overflow)?;
    Ok(if r != 0 && ((r < 0) != (b < 0)) { r + b } else { r })
}

fn log2_ceil(x: i128) -> Result<i128, ExprError> {
    if x <= 0 {
        return Err(ExprError::Log2CeilOfNonPositive(x));
    }
    Ok(i128::from(128 - (x - 1).leading_zeros()) * (x > 1) as i128)
}

impl From<i128> for IntExpr {
    fn from(value: i128) -> Self {
        Self::Const(value)
    }
}

macro_rules! from_int {
    ($($t:ty),*) => {$(
        impl From<$t> for IntExpr {
            fn from(value: $t) -> Self {
                Self::Const(i128::from(value))
            }
        }
    )*};
}
from_int!(i32, i64, u32, u64);

impl From<usize> for IntExpr {
    fn from(value: usize) -> Self {
        Self::Const(value as i128)
    }
}

impl From<&str> for IntExpr {
    fn from(name: &str) -> Self {
        Self::Var(name.to_owned())
    }
}

macro_rules! binary_op {
    ($trait:ident, $method:ident, $variant:ident) => {
        impl<T: Into<IntExpr>> ops::$trait<T> for IntExpr {
            type Output = IntExpr;
            fn $method(self, rhs: T) -> IntExpr {
                IntExpr::$variant(Box::new(self), Box::new(rhs.into()))
            }
        }
    };
}
binary_op!(Add, add, Add);
binary_op!(Sub, sub, Sub);
binary_op!(Mul, mul, Mul);

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_floor_division_and_remainder_agree() {
        let mut rng = rand::rng();
        for _ in 0..1000 {
            let a: i128 = rng.random_range(-1000..1000);
            let b: i128 = loop {
                let b = rng.random_range(-50..50);
                if b != 0 {
                    break b;
                }
            };
            let q = floor_div(a, b).unwrap();
            let r = floor_rem(a, b).unwrap();
            assert_eq!(q * b + r, a);
            assert!(r == 0 || (r < 0) == (b < 0));
            assert!(r.abs() < b.abs());
        }
    }

    #[test]
    fn test_log2_ceil() {
        for (x, expected) in [(1, 0), (2, 1), (3, 2), (4, 2), (5, 3), (1024, 10), (1025, 11)] {
            assert_eq!(log2_ceil(x).unwrap(), expected);
        }
    }

    #[test]
    fn test_bound_contains_every_value() {
        let mut rng = rand::rng();
        let i = IntExpr::LoopIndex(0);
        let exprs = [
            i.clone() * 3 + 5,
            (i.clone() - 7) * (i.clone() + 2),
            (i.clone() * 5).floor_div(3),
            (i.clone() - 4).floor_rem(6),
            (i.clone() + 1).log2_ceil(),
        ];
        let env = Env::default();
        for expr in exprs {
            let lo: i128 = rng.random_range(0..20);
            let hi = lo + rng.random_range(0..40);
            let bound = expr.bound(&env, &[(0, Interval { min: lo, max: hi })]).unwrap();
            for v in lo..=hi {
                let value = expr.eval(&env.with_index(0, v)).unwrap();
                assert!(
                    bound.min <= value && value <= bound.max,
                    "{expr:?} at {v}: {value} not in {bound:?}"
                );
            }
        }
    }

    #[test]
    fn test_unbound_index_is_reported() {
        let e = IntExpr::LoopIndex(3) + 1;
        assert_eq!(e.eval(&Env::default()), Err(ExprError::IndexDependent(3)));
    }
}
