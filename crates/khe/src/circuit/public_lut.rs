use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Canonical, parameter-independent expression used by a public lookup program.
///
/// The expression is evaluated lazily for the requested input. Keeping the program instead of a
/// materialized table makes circuit serialization and correctness certification independent of the
/// number of lookup entries.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum LutExpr {
    Input,
    Const(BigInt),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    EuclideanMod(Box<Self>, BigUint),
    FloorDiv(Box<Self>, BigUint),
    RoundDiv(Box<Self>, BigUint),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LutInterval {
    pub min: BigInt,
    pub max: BigInt,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum PublicLutError {
    #[error("a public lookup program must contain at least one input")]
    Empty,
    #[error("public lookup input {input} is outside 0..{length}")]
    InputOutOfRange { input: u64, length: u64 },
    #[error("a public lookup divisor must be positive")]
    ZeroDivisor,
}

impl LutExpr {
    pub fn input() -> Self {
        Self::Input
    }

    pub fn constant(value: impl Into<BigInt>) -> Self {
        Self::Const(value.into())
    }

    pub fn add(self, right: Self) -> Self {
        Self::Add(Box::new(self), Box::new(right))
    }

    pub fn sub(self, right: Self) -> Self {
        Self::Sub(Box::new(self), Box::new(right))
    }

    pub fn mul(self, right: Self) -> Self {
        Self::Mul(Box::new(self), Box::new(right))
    }

    pub fn modulo(self, divisor: impl Into<BigUint>) -> Self {
        Self::EuclideanMod(Box::new(self), divisor.into())
    }

    pub fn floor_div(self, divisor: impl Into<BigUint>) -> Self {
        Self::FloorDiv(Box::new(self), divisor.into())
    }

    pub fn round_div(self, divisor: impl Into<BigUint>) -> Self {
        Self::RoundDiv(Box::new(self), divisor.into())
    }

    pub fn evaluate(&self, input: u64) -> Result<BigInt, PublicLutError> {
        match self {
            Self::Input => Ok(BigInt::from(input)),
            Self::Const(value) => Ok(value.clone()),
            Self::Add(left, right) => Ok(left.evaluate(input)? + right.evaluate(input)?),
            Self::Sub(left, right) => Ok(left.evaluate(input)? - right.evaluate(input)?),
            Self::Mul(left, right) => Ok(left.evaluate(input)? * right.evaluate(input)?),
            Self::EuclideanMod(value, divisor) => {
                let divisor = positive_divisor(divisor)?;
                Ok(value.evaluate(input)?.mod_floor(&divisor))
            }
            Self::FloorDiv(value, divisor) => {
                let divisor = positive_divisor(divisor)?;
                Ok(value.evaluate(input)?.div_floor(&divisor))
            }
            Self::RoundDiv(value, divisor) => {
                let divisor = positive_divisor(divisor)?;
                let half: BigInt = &divisor / BigInt::from(2u8);
                Ok((value.evaluate(input)? + half).div_floor(&divisor))
            }
        }
    }

    pub fn interval(&self, input: LutInterval) -> Result<LutInterval, PublicLutError> {
        match self {
            Self::Input => Ok(input),
            Self::Const(value) => Ok(LutInterval { min: value.clone(), max: value.clone() }),
            Self::Add(left, right) => {
                let left = left.interval(input.clone())?;
                let right = right.interval(input)?;
                Ok(LutInterval { min: left.min + right.min, max: left.max + right.max })
            }
            Self::Sub(left, right) => {
                let left = left.interval(input.clone())?;
                let right = right.interval(input)?;
                Ok(LutInterval { min: left.min - right.max, max: left.max - right.min })
            }
            Self::Mul(left, right) => {
                let left = left.interval(input.clone())?;
                let right = right.interval(input)?;
                let products = [
                    &left.min * &right.min,
                    &left.min * &right.max,
                    &left.max * &right.min,
                    &left.max * &right.max,
                ];
                Ok(LutInterval {
                    min: products.iter().min().expect("four products").clone(),
                    max: products.iter().max().expect("four products").clone(),
                })
            }
            Self::EuclideanMod(_, divisor) => {
                let divisor = positive_divisor(divisor)?;
                Ok(LutInterval { min: BigInt::zero(), max: divisor - BigInt::one() })
            }
            Self::FloorDiv(value, divisor) => {
                let divisor = positive_divisor(divisor)?;
                let value = value.interval(input)?;
                Ok(LutInterval {
                    min: value.min.div_floor(&divisor),
                    max: value.max.div_floor(&divisor),
                })
            }
            Self::RoundDiv(value, divisor) => {
                let divisor = positive_divisor(divisor)?;
                let value = value.interval(input)?;
                let half: BigInt = &divisor / BigInt::from(2u8);
                Ok(LutInterval {
                    min: (value.min + &half).div_floor(&divisor),
                    max: (value.max + half).div_floor(&divisor),
                })
            }
        }
    }
}

fn positive_divisor(divisor: &BigUint) -> Result<BigInt, PublicLutError> {
    if divisor.is_zero() {
        Err(PublicLutError::ZeroDivisor)
    } else {
        Ok(BigInt::from(divisor.clone()))
    }
}

/// Compact public lookup program with the fixed row permutation `row(x) = x`.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct PublicLutProgram {
    len: u64,
    value: LutExpr,
}

impl PublicLutProgram {
    pub fn new(len: u64, value: LutExpr) -> Result<Self, PublicLutError> {
        if len == 0 {
            return Err(PublicLutError::Empty);
        }
        let program = Self { len, value };
        program.output_interval()?;
        Ok(program)
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn value(&self) -> &LutExpr {
        &self.value
    }

    pub fn entry(&self, input: u64) -> Result<(u64, BigInt), PublicLutError> {
        if input >= self.len {
            return Err(PublicLutError::InputOutOfRange { input, length: self.len });
        }
        Ok((input, self.value.evaluate(input)?))
    }

    pub fn entries(&self) -> impl Iterator<Item = (u64, (u64, BigInt))> + '_ {
        (0..self.len).map(|input| {
            let (row, value) = self.entry(input).expect("validated lookup expression");
            (input, (row, value))
        })
    }

    pub fn output_interval(&self) -> Result<LutInterval, PublicLutError> {
        self.value.interval(LutInterval { min: BigInt::zero(), max: BigInt::from(self.len - 1) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_program_evaluates_without_materializing_entries() {
        let x = LutExpr::input();
        let program =
            PublicLutProgram::new(16, x.clone().modulo(5u8).mul(x.floor_div(5u8)).modulo(5u8))
                .unwrap();
        assert_eq!(program.entry(13).unwrap(), (13, BigInt::from(1)));
        assert_eq!(
            program.output_interval().unwrap(),
            LutInterval { min: BigInt::zero(), max: BigInt::from(4) }
        );
    }

    #[test]
    fn interval_is_recomputed_from_the_program() {
        let program =
            PublicLutProgram::new(8, LutExpr::input().mul(LutExpr::constant(3)).round_div(2u8))
                .unwrap();
        assert_eq!(
            program.output_interval().unwrap(),
            LutInterval { min: BigInt::zero(), max: BigInt::from(11) }
        );
    }

    #[test]
    fn invalid_programs_are_rejected() {
        assert_eq!(PublicLutProgram::new(0, LutExpr::Input), Err(PublicLutError::Empty));
        assert_eq!(
            PublicLutProgram::new(1, LutExpr::Input.floor_div(0u8)),
            Err(PublicLutError::ZeroDivisor)
        );
    }
}
