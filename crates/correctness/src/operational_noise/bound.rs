//! Deterministic matrix-coefficient bounds for an extracted egg expression.
//!
//! This module never estimates a sampler from sigma or observed values.  The
//! caller supplies closed sampler cutoffs and resolved matrix attributes via
//! [`BoundInput`].  The evaluator owns exactly one memo: the `BTreeMap` from a
//! canonical extracted term to its final [`MatrixBound`].

use super::{
    identity::{AtomicSourceId, Axis, CrtSpecId, MatrixConstantSpecId},
    language::MxxLang,
};
use egg::Id;
use mxx_ir_core::types::ConcreteMatrixType;
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Signed, Zero};
use std::collections::BTreeMap;

/// Whether a matrix has a numeric centered-coefficient bound.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BoundClass {
    ExactZero,
    Bounded { maximum_absolute_coefficient: BigUint },
    Large,
}

impl BoundClass {
    pub fn maximum_absolute_coefficient(&self) -> Option<BigUint> {
        match self {
            Self::ExactZero => Some(BigUint::ZERO),
            Self::Bounded { maximum_absolute_coefficient } => {
                Some(maximum_absolute_coefficient.clone())
            }
            Self::Large => None,
        }
    }

    fn add(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::ExactZero, value) | (value, Self::ExactZero) => value.clone(),
            (
                Self::Bounded { maximum_absolute_coefficient: left },
                Self::Bounded { maximum_absolute_coefficient: right },
            ) => Self::Bounded { maximum_absolute_coefficient: left + right },
            _ => Self::Large,
        }
    }

    fn maximum(classes: impl IntoIterator<Item = Self>) -> Self {
        let mut maximum = BigUint::zero();
        for class in classes {
            match class {
                Self::ExactZero => {}
                Self::Bounded { maximum_absolute_coefficient } => {
                    maximum = maximum.max(maximum_absolute_coefficient);
                }
                Self::Large => return Self::Large,
            }
        }
        if maximum.is_zero() {
            Self::ExactZero
        } else {
            Self::Bounded { maximum_absolute_coefficient: maximum }
        }
    }
}

/// Metadata that has a proof-preserving matrix transfer rule.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatrixMetadata {
    pub is_constant_polynomial: bool,
    pub known_zero_rows: Option<BigUint>,
}

impl MatrixMetadata {
    pub const fn unknown() -> Self {
        Self { is_constant_polynomial: false, known_zero_rows: None }
    }
}

/// A coefficient bound and the concrete matrix shape to which it applies.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatrixBound {
    pub matrix_type: ConcreteMatrixType,
    pub coefficient_class: BoundClass,
    pub metadata: MatrixMetadata,
}

/// The resolved data needed by a matrix constant transfer row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResolvedMatrixConstant {
    Zero,
    Identity,
    UnitRow { index: BigUint },
    UnitColumn { index: BigUint },
    Gadget { base: BigInt, small: bool },
    PowerOfBase { base: BigInt, exponent: BigUint },
    Rotation { exponent: BigInt },
    Polynomial { coefficients: Box<[BigInt]> },
}

/// A bound-side error.  Simulation maps this closed error to the public
/// `BoundError` together with its graph occurrence; no generic margin exists.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BoundEvaluationError {
    MissingExtractedTerm { term: Id },
    NonMatrixTerm { term: Id },
    ExtractedExpressionCycle { term: Id },
    EmptyMatrixOperation { term: Id },
    IncompatibleMatrixProduct { left: ConcreteMatrixType, right: ConcreteMatrixType },
    InvalidKnownZeroRows { known_zero_rows: BigUint, row_count: BigUint },
    InvalidMatrixConstant { term: Id },
    InvalidMatrixScale { term: Id },
    InvalidCrtRecompose { term: Id },
    InvalidPack { term: Id },
    IntegerLimitExceeded { operation: &'static str, value: BigUint },
    IntegerBitLimitExceeded { operation: &'static str, bits: BigUint },
    RecurrenceStepLimitExceeded { limit: BigUint, count: BigUint },
    UnconsumedLargeTerm { term: Id },
}

/// Read-only bridge from extraction/analysis into the bound evaluator.
///
/// It deliberately exposes resolved values rather than caches or provisional
/// bounds.  The evaluator is the sole owner of computed `MatrixBound`s.
pub trait BoundInput {
    fn node(&self, term: Id) -> Option<&MxxLang>;
    fn matrix_type(&self, term: Id) -> Result<ConcreteMatrixType, BoundEvaluationError>;
    fn atom_bound(
        &self,
        source: AtomicSourceId,
        term: Id,
    ) -> Result<MatrixBound, BoundEvaluationError>;
    fn matrix_constant(
        &self,
        spec: MatrixConstantSpecId,
        term: Id,
    ) -> Result<(ConcreteMatrixType, ResolvedMatrixConstant), BoundEvaluationError>;
    fn scalar_maximum_absolute(&self, term: Id) -> Result<BigUint, BoundEvaluationError>;
    fn lift_constant_polynomial_class(
        &self,
        term: Id,
        input: Id,
    ) -> Result<BoundClass, BoundEvaluationError>;
    fn crt_coefficients(
        &self,
        spec: CrtSpecId,
        term: Id,
    ) -> Result<Box<[BigInt]>, BoundEvaluationError>;
    fn validate_integer(
        &self,
        value: &BigUint,
        operation: &'static str,
    ) -> Result<(), BoundEvaluationError>;
    fn validate_integer_bits(
        &self,
        _: &BigUint,
        _: &'static str,
    ) -> Result<(), BoundEvaluationError> {
        Ok(())
    }
    fn reserve_owned_elements(&self, _: usize) -> Result<(), BoundEvaluationError> {
        Ok(())
    }
    fn check_deadline(&self) -> Result<(), BoundEvaluationError> {
        Ok(())
    }
    fn validate_pack(&self, term: Id, bit_count: usize) -> Result<(), BoundEvaluationError>;
}

/// Job-wide controls used by production bound inputs.  The evaluator calls
/// these at every work-stack and transfer boundary; implementations must share
/// their deadline and cumulative allocation accounting with lowering, rewrite,
/// and extraction.  Test inputs may deliberately omit this bridge.
pub trait BoundEvaluationControl {
    fn check_deadline(&self) -> Result<(), BoundEvaluationError>;
    fn reserve_owned_elements(&self, requested: usize) -> Result<(), BoundEvaluationError>;
    fn validate_integer_bits(
        &self,
        value: &BigUint,
        operation: &'static str,
    ) -> Result<(), BoundEvaluationError>;
    fn validate_pack(&self, term: Id, bit_count: usize) -> Result<(), BoundEvaluationError>;
    /// The job-wide recurrence ceiling.  Test inputs may omit it; production
    /// adapters must return the `CheckerLimits` value for the active job.
    fn recurrence_step_limit(&self) -> Option<BigUint> {
        None
    }
}

/// Evaluates one extracted matrix root without recursion.
pub struct BoundEvaluator<'a, I> {
    input: &'a I,
    memo: BTreeMap<Id, MatrixBound>,
}

impl<'a, I: BoundInput> BoundEvaluator<'a, I> {
    pub fn new(input: &'a I) -> Self {
        Self { input, memo: BTreeMap::new() }
    }

    pub fn memo(&self) -> &BTreeMap<Id, MatrixBound> {
        &self.memo
    }

    pub fn evaluate(mut self, root: Id) -> Result<MatrixBound, BoundEvaluationError> {
        enum Work {
            Enter(Id),
            Finish(Id, MxxLang),
        }

        self.input.reserve_owned_elements(1)?;
        let mut work = vec![Work::Enter(root)];
        while let Some(item) = work.pop() {
            self.input.check_deadline()?;
            match item {
                Work::Enter(term) if self.memo.contains_key(&term) => {}
                Work::Enter(term) => {
                    let node = self
                        .input
                        .node(term)
                        .ok_or(BoundEvaluationError::MissingExtractedTerm { term })?
                        .clone();
                    let children = matrix_children(&node, term)?;
                    self.input.reserve_owned_elements(1usize.saturating_add(children.len()))?;
                    work.push(Work::Finish(term, node));
                    for child in children.into_iter().rev() {
                        if !self.memo.contains_key(&child) {
                            work.push(Work::Enter(child));
                        }
                    }
                }
                Work::Finish(term, node) => {
                    if self.memo.contains_key(&term) {
                        continue;
                    }
                    let bound = self.finish(term, &node)?;
                    self.input.reserve_owned_elements(1)?;
                    self.memo.insert(term, bound);
                }
            }
        }
        let bound = self
            .memo
            .remove(&root)
            .ok_or(BoundEvaluationError::ExtractedExpressionCycle { term: root })?;
        if matches!(bound.coefficient_class, BoundClass::Large) {
            return Err(BoundEvaluationError::UnconsumedLargeTerm { term: root });
        }
        Ok(bound)
    }

    fn child(&self, term: Id) -> Result<&MatrixBound, BoundEvaluationError> {
        self.memo.get(&term).ok_or(BoundEvaluationError::ExtractedExpressionCycle { term })
    }

    fn finish(&self, term: Id, node: &MxxLang) -> Result<MatrixBound, BoundEvaluationError> {
        use MxxLang::*;
        match node {
            Atom { source, .. } => self.input.atom_bound(*source, term),
            MatrixConstant(spec) => self.bound_matrix_constant(term, *spec),
            MatrixCanonicalRangeContract { input, .. } => Ok(self.child(input[0])?.clone()),
            HashPlain { .. } => Ok(MatrixBound {
                matrix_type: self.input.matrix_type(term)?,
                coefficient_class: BoundClass::Large,
                metadata: MatrixMetadata::unknown(),
            }),
            MatrixAdd(children) => self.bound_add(term, children),
            MatrixMultiply(children) => self.bound_multiply(term, children),
            MatrixNegate(input) => Ok(self.child(input[0])?.clone()),
            MatrixScale(children) => self.bound_scale(term, children),
            MatrixTranspose(input) => {
                let child = self.child(input[0])?;
                Ok(MatrixBound {
                    matrix_type: self.input.matrix_type(term)?,
                    coefficient_class: child.coefficient_class.clone(),
                    metadata: MatrixMetadata {
                        is_constant_polynomial: child.metadata.is_constant_polynomial,
                        known_zero_rows: None,
                    },
                })
            }
            MatrixSlice { input, .. } => {
                let child = self.child(input[0])?;
                Ok(MatrixBound {
                    matrix_type: self.input.matrix_type(term)?,
                    coefficient_class: child.coefficient_class.clone(),
                    metadata: MatrixMetadata {
                        is_constant_polynomial: child.metadata.is_constant_polynomial,
                        known_zero_rows: None,
                    },
                })
            }
            MatrixTensor(children) => self.bound_tensor(term, children),
            MatrixConcat { axis, inputs } => self.bound_concat(term, *axis, inputs),
            Switch(children) => self.bound_switch(term, children),
            LiftConstantPolynomial { input, .. } => Ok(MatrixBound {
                matrix_type: self.input.matrix_type(term)?,
                coefficient_class: self.input.lift_constant_polynomial_class(term, input[0])?,
                metadata: MatrixMetadata { is_constant_polynomial: true, known_zero_rows: None },
            }),
            CrtRecompose { spec, inputs } => self.bound_crt(term, *spec, inputs),
            PackPolynomialCoefficients { bits, .. } => {
                self.input.validate_pack(term, bits.len())?;
                Ok(MatrixBound {
                    matrix_type: self.input.matrix_type(term)?,
                    coefficient_class: BoundClass::Large,
                    metadata: MatrixMetadata::unknown(),
                })
            }
            IntConst(_) |
            IntParameter(_) |
            IntBinder(_) |
            IntAdd(_) |
            IntSub(_) |
            IntMul(_) |
            IntExactDiv(_) |
            IntEuclideanDiv(_) |
            IntEuclideanRemainder(_) |
            IntRoundDiv(_) |
            IntLog2Ceil(_) |
            BoolConst(_) |
            IntEqual(_) |
            IntLess(_) |
            IntLessEqual(_) |
            BitExtract { .. } |
            BoolToInt(_) |
            RealConst(_) |
            IntToReal(_) |
            RealAdd(_) |
            RealSub(_) |
            RealMul(_) |
            RealDiv(_) |
            RealSqrt(_) |
            ExtractCoefficient(_) => Err(BoundEvaluationError::NonMatrixTerm { term }),
        }
    }

    fn bound_matrix_constant(
        &self,
        term: Id,
        spec: MatrixConstantSpecId,
    ) -> Result<MatrixBound, BoundEvaluationError> {
        let (matrix_type, value) = self.input.matrix_constant(spec, term)?;
        let rows = BigUint::from(matrix_type.rows);
        let metadata = match &value {
            ResolvedMatrixConstant::Zero => {
                MatrixMetadata { is_constant_polynomial: true, known_zero_rows: Some(rows.clone()) }
            }
            ResolvedMatrixConstant::UnitColumn { .. } => MatrixMetadata {
                is_constant_polynomial: true,
                known_zero_rows: (!rows.is_zero()).then(|| rows.clone() - BigUint::one()),
            },
            ResolvedMatrixConstant::Rotation { exponent } => {
                MatrixMetadata { is_constant_polynomial: exponent.is_zero(), known_zero_rows: None }
            }
            ResolvedMatrixConstant::Polynomial { coefficients } => MatrixMetadata {
                is_constant_polynomial: coefficients.iter().skip(1).all(BigInt::is_zero),
                known_zero_rows: None,
            },
            _ => MatrixMetadata { is_constant_polynomial: true, known_zero_rows: None },
        };
        let coefficient_class = match value {
            ResolvedMatrixConstant::Zero => BoundClass::ExactZero,
            ResolvedMatrixConstant::Identity => {
                if matrix_type.rows != matrix_type.columns {
                    return Err(BoundEvaluationError::InvalidMatrixConstant { term });
                }
                BoundClass::Bounded { maximum_absolute_coefficient: BigUint::one() }
            }
            ResolvedMatrixConstant::UnitRow { index } => {
                if matrix_type.rows != 1 || index >= BigUint::from(matrix_type.columns) {
                    return Err(BoundEvaluationError::InvalidMatrixConstant { term });
                }
                BoundClass::Bounded { maximum_absolute_coefficient: BigUint::one() }
            }
            ResolvedMatrixConstant::UnitColumn { index } => {
                if matrix_type.columns != 1 || index >= rows {
                    return Err(BoundEvaluationError::InvalidMatrixConstant { term });
                }
                BoundClass::Bounded { maximum_absolute_coefficient: BigUint::one() }
            }
            ResolvedMatrixConstant::Gadget { base, small } => {
                let absolute = base.abs().to_biguint().unwrap_or_default();
                if small {
                    if absolute < BigUint::from(2_u8) {
                        return Err(BoundEvaluationError::InvalidMatrixConstant { term });
                    }
                    BoundClass::Bounded { maximum_absolute_coefficient: absolute - BigUint::one() }
                } else {
                    BoundClass::Bounded {
                        maximum_absolute_coefficient: (absolute / BigUint::from(2_u8))
                            .max(BigUint::one()),
                    }
                }
            }
            ResolvedMatrixConstant::PowerOfBase { base, exponent } => {
                if matrix_type.rows != 1 || matrix_type.columns != 1 {
                    return Err(BoundEvaluationError::InvalidMatrixConstant { term });
                }
                let absolute = base.abs().to_biguint().unwrap_or_default();
                if absolute > BigUint::one() {
                    let prospective_bits = BigUint::from(absolute.bits()) * &exponent;
                    self.input
                        .validate_integer_bits(&prospective_bits, "power-of-base constant")?;
                }
                let exponent = exponent
                    .try_into()
                    .map_err(|_| BoundEvaluationError::InvalidMatrixConstant { term })?;
                BoundClass::Bounded { maximum_absolute_coefficient: absolute.pow(exponent) }
            }
            ResolvedMatrixConstant::Rotation { exponent } => {
                if matrix_type.rows != 1 || matrix_type.columns != 1 || exponent.is_negative() {
                    return Err(BoundEvaluationError::InvalidMatrixConstant { term });
                }
                BoundClass::Bounded { maximum_absolute_coefficient: BigUint::one() }
            }
            ResolvedMatrixConstant::Polynomial { coefficients } => {
                if matrix_type.rows != 1 || matrix_type.columns != 1 {
                    return Err(BoundEvaluationError::InvalidMatrixConstant { term });
                }
                BoundClass::maximum(coefficients.iter().map(|coefficient| {
                    let maximum_absolute_coefficient =
                        coefficient.abs().to_biguint().unwrap_or_default();
                    if maximum_absolute_coefficient.is_zero() {
                        BoundClass::ExactZero
                    } else {
                        BoundClass::Bounded { maximum_absolute_coefficient }
                    }
                }))
            }
        };
        if let Some(value) = coefficient_class.maximum_absolute_coefficient() {
            self.input.validate_integer(&value, "matrix constant")?;
        }
        Ok(MatrixBound { matrix_type, coefficient_class, metadata })
    }

    fn bound_add(&self, term: Id, children: &[Id]) -> Result<MatrixBound, BoundEvaluationError> {
        if children.is_empty() {
            return Err(BoundEvaluationError::EmptyMatrixOperation { term });
        }
        let matrix_type = self.input.matrix_type(term)?;
        let class = children.iter().try_fold(BoundClass::ExactZero, |accumulator, child| {
            let bound = self.child(*child)?;
            if bound.matrix_type != matrix_type {
                return Err(BoundEvaluationError::IncompatibleMatrixProduct {
                    left: matrix_type.clone(),
                    right: bound.matrix_type.clone(),
                });
            }
            Ok(accumulator.add(&bound.coefficient_class))
        })?;
        if let Some(value) = class.maximum_absolute_coefficient() {
            self.input.validate_integer(&value, "matrix addition")?;
        }
        Ok(MatrixBound {
            matrix_type,
            coefficient_class: class,
            metadata: MatrixMetadata {
                is_constant_polynomial: children.iter().all(|child| {
                    self.child(*child).is_ok_and(|bound| bound.metadata.is_constant_polynomial)
                }),
                known_zero_rows: None,
            },
        })
    }

    fn bound_multiply(
        &self,
        term: Id,
        children: &[Id],
    ) -> Result<MatrixBound, BoundEvaluationError> {
        let Some((&first, rest)) = children.split_first() else {
            return Err(BoundEvaluationError::EmptyMatrixOperation { term });
        };
        let mut bound = self.child(first)?.clone();
        for child in rest {
            bound = product_bound(&bound, self.child(*child)?)?;
            if let Some(value) = bound.coefficient_class.maximum_absolute_coefficient() {
                self.input.validate_integer(&value, "matrix product")?;
            }
        }
        let output = self.input.matrix_type(term)?;
        if bound.matrix_type != output {
            return Err(BoundEvaluationError::IncompatibleMatrixProduct {
                left: bound.matrix_type,
                right: output,
            });
        }
        Ok(bound)
    }

    fn bound_scale(
        &self,
        term: Id,
        children: &[Id; 2],
    ) -> Result<MatrixBound, BoundEvaluationError> {
        let scalar = self.input.scalar_maximum_absolute(children[0])?;
        let matrix = self.child(children[1])?;
        let coefficient_class = match &matrix.coefficient_class {
            BoundClass::Large if scalar.is_zero() => BoundClass::ExactZero,
            BoundClass::ExactZero => BoundClass::ExactZero,
            BoundClass::Bounded { maximum_absolute_coefficient } => {
                let maximum_absolute_coefficient = maximum_absolute_coefficient * scalar;
                self.input.validate_integer(&maximum_absolute_coefficient, "matrix scale")?;
                if maximum_absolute_coefficient.is_zero() {
                    BoundClass::ExactZero
                } else {
                    BoundClass::Bounded { maximum_absolute_coefficient }
                }
            }
            BoundClass::Large => BoundClass::Large,
        };
        let matrix_type = self.input.matrix_type(term)?;
        if matrix.matrix_type != matrix_type {
            return Err(BoundEvaluationError::IncompatibleMatrixProduct {
                left: matrix.matrix_type.clone(),
                right: matrix_type,
            });
        }
        Ok(MatrixBound {
            matrix_type: matrix.matrix_type.clone(),
            coefficient_class,
            metadata: MatrixMetadata {
                is_constant_polynomial: matrix.metadata.is_constant_polynomial,
                known_zero_rows: None,
            },
        })
    }

    fn bound_tensor(
        &self,
        term: Id,
        children: &[Id; 2],
    ) -> Result<MatrixBound, BoundEvaluationError> {
        let left = self.child(children[0])?;
        let right = self.child(children[1])?;
        if left.matrix_type.modulus != right.matrix_type.modulus ||
            left.matrix_type.ring_dimension != right.matrix_type.ring_dimension
        {
            return Err(BoundEvaluationError::IncompatibleMatrixProduct {
                left: left.matrix_type.clone(),
                right: right.matrix_type.clone(),
            });
        }
        let matrix_type = self.input.matrix_type(term)?;
        if matrix_type.modulus != left.matrix_type.modulus ||
            matrix_type.ring_dimension != left.matrix_type.ring_dimension ||
            Some(matrix_type.rows) != left.matrix_type.rows.checked_mul(right.matrix_type.rows) ||
            Some(matrix_type.columns) !=
                left.matrix_type.columns.checked_mul(right.matrix_type.columns)
        {
            return Err(BoundEvaluationError::IncompatibleMatrixProduct {
                left: left.matrix_type.clone(),
                right: matrix_type,
            });
        }
        let ring_factor =
            if left.metadata.is_constant_polynomial || right.metadata.is_constant_polynomial {
                BigUint::one()
            } else {
                BigUint::from(left.matrix_type.ring_dimension)
            };
        let class =
            multiply_classes(&left.coefficient_class, &right.coefficient_class, &ring_factor);
        if let Some(value) = class.maximum_absolute_coefficient() {
            self.input.validate_integer(&value, "matrix tensor")?;
        }
        Ok(MatrixBound {
            matrix_type,
            coefficient_class: class,
            metadata: MatrixMetadata {
                is_constant_polynomial: left.metadata.is_constant_polynomial &&
                    right.metadata.is_constant_polynomial,
                known_zero_rows: None,
            },
        })
    }

    fn bound_concat(
        &self,
        term: Id,
        axis: Axis,
        inputs: &[Id],
    ) -> Result<MatrixBound, BoundEvaluationError> {
        if inputs.is_empty() {
            return Err(BoundEvaluationError::EmptyMatrixOperation { term });
        }
        let matrix_type = self.input.matrix_type(term)?;
        self.input.reserve_owned_elements(inputs.len())?;
        let bounds = inputs.iter().map(|id| self.child(*id)).collect::<Result<Vec<_>, _>>()?;
        let first = bounds[0];
        let mut rows = 0_usize;
        let mut columns = 0_usize;
        for bound in &bounds {
            if bound.matrix_type.modulus != matrix_type.modulus ||
                bound.matrix_type.ring_dimension != matrix_type.ring_dimension
            {
                return Err(BoundEvaluationError::IncompatibleMatrixProduct {
                    left: bound.matrix_type.clone(),
                    right: matrix_type.clone(),
                });
            }
            match axis {
                Axis::Rows if bound.matrix_type.columns == first.matrix_type.columns => {
                    rows = rows
                        .checked_add(bound.matrix_type.rows)
                        .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })?;
                }
                Axis::Columns if bound.matrix_type.rows == first.matrix_type.rows => {
                    columns = columns
                        .checked_add(bound.matrix_type.columns)
                        .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })?;
                }
                Axis::Diagonal => {
                    rows = rows
                        .checked_add(bound.matrix_type.rows)
                        .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })?;
                    columns = columns
                        .checked_add(bound.matrix_type.columns)
                        .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })?;
                }
                _ => {
                    return Err(BoundEvaluationError::IncompatibleMatrixProduct {
                        left: first.matrix_type.clone(),
                        right: bound.matrix_type.clone(),
                    });
                }
            }
        }
        let shape_matches = match axis {
            Axis::Rows => {
                matrix_type.rows == rows && matrix_type.columns == first.matrix_type.columns
            }
            Axis::Columns => {
                matrix_type.rows == first.matrix_type.rows && matrix_type.columns == columns
            }
            Axis::Diagonal => matrix_type.rows == rows && matrix_type.columns == columns,
        };
        if !shape_matches {
            return Err(BoundEvaluationError::IncompatibleMatrixProduct {
                left: first.matrix_type.clone(),
                right: matrix_type,
            });
        }
        let class = BoundClass::maximum(bounds.iter().map(|bound| bound.coefficient_class.clone()));
        if let Some(value) = class.maximum_absolute_coefficient() {
            self.input.validate_integer(&value, "matrix concatenation")?;
        }
        Ok(MatrixBound {
            matrix_type,
            coefficient_class: class,
            metadata: MatrixMetadata {
                is_constant_polynomial: bounds
                    .iter()
                    .all(|bound| bound.metadata.is_constant_polynomial),
                known_zero_rows: None,
            },
        })
    }

    fn bound_switch(&self, term: Id, children: &[Id]) -> Result<MatrixBound, BoundEvaluationError> {
        let Some((_, cases)) = children.split_first() else {
            return Err(BoundEvaluationError::EmptyMatrixOperation { term });
        };
        if cases.is_empty() {
            return Err(BoundEvaluationError::EmptyMatrixOperation { term });
        }
        let matrix_type = self.input.matrix_type(term)?;
        self.input.reserve_owned_elements(cases.len())?;
        let bounds = cases.iter().map(|id| self.child(*id)).collect::<Result<Vec<_>, _>>()?;
        if bounds.iter().any(|bound| bound.matrix_type != matrix_type) {
            return Err(BoundEvaluationError::IncompatibleMatrixProduct {
                left: bounds[0].matrix_type.clone(),
                right: matrix_type,
            });
        }
        let class = BoundClass::maximum(bounds.iter().map(|bound| bound.coefficient_class.clone()));
        let known_zero_rows = bounds
            .iter()
            .map(|bound| bound.metadata.known_zero_rows.clone())
            .collect::<Option<Vec<_>>>()
            .and_then(|values| values.into_iter().min());
        if let Some(value) = class.maximum_absolute_coefficient() {
            self.input.validate_integer(&value, "matrix switch")?;
        }
        Ok(MatrixBound {
            matrix_type,
            coefficient_class: class,
            metadata: MatrixMetadata {
                is_constant_polynomial: bounds
                    .iter()
                    .all(|bound| bound.metadata.is_constant_polynomial),
                known_zero_rows,
            },
        })
    }

    fn bound_crt(
        &self,
        term: Id,
        spec: CrtSpecId,
        inputs: &[Id],
    ) -> Result<MatrixBound, BoundEvaluationError> {
        let coefficients = self.input.crt_coefficients(spec, term)?;
        if coefficients.len() != inputs.len() || inputs.is_empty() {
            return Err(BoundEvaluationError::InvalidCrtRecompose { term });
        }
        let matrix_type = self.input.matrix_type(term)?;
        let mut class = BoundClass::ExactZero;
        let mut all_constant_polynomials = true;
        for (input, coefficient) in inputs.iter().zip(coefficients.iter()) {
            let bound = self.child(*input)?;
            if bound.matrix_type.ring_dimension != matrix_type.ring_dimension ||
                bound.matrix_type.rows != matrix_type.rows ||
                bound.matrix_type.columns != matrix_type.columns
            {
                return Err(BoundEvaluationError::IncompatibleMatrixProduct {
                    left: bound.matrix_type.clone(),
                    right: matrix_type.clone(),
                });
            }
            all_constant_polynomials &= bound.metadata.is_constant_polynomial;
            let factor = coefficient.abs().to_biguint().unwrap_or_default();
            class = class.add(&match &bound.coefficient_class {
                BoundClass::ExactZero => BoundClass::ExactZero,
                BoundClass::Bounded { maximum_absolute_coefficient } => BoundClass::Bounded {
                    maximum_absolute_coefficient: maximum_absolute_coefficient * factor,
                },
                BoundClass::Large => BoundClass::Large,
            });
        }
        if let Some(value) = class.maximum_absolute_coefficient() {
            self.input.validate_integer(&value, "CRT recomposition")?;
        }
        Ok(MatrixBound {
            matrix_type,
            coefficient_class: class,
            metadata: MatrixMetadata {
                is_constant_polynomial: all_constant_polynomials,
                known_zero_rows: None,
            },
        })
    }
}

/// The sole deterministic matrix-product transfer helper.
pub fn product_bound(
    left: &MatrixBound,
    right: &MatrixBound,
) -> Result<MatrixBound, BoundEvaluationError> {
    if left.matrix_type.modulus != right.matrix_type.modulus ||
        left.matrix_type.ring_dimension != right.matrix_type.ring_dimension ||
        left.matrix_type.columns != right.matrix_type.rows
    {
        return Err(BoundEvaluationError::IncompatibleMatrixProduct {
            left: left.matrix_type.clone(),
            right: right.matrix_type.clone(),
        });
    }
    let rows = BigUint::from(right.matrix_type.rows);
    let known_zero_rows = right.metadata.known_zero_rows.clone().unwrap_or_default();
    if known_zero_rows > rows {
        return Err(BoundEvaluationError::InvalidKnownZeroRows { known_zero_rows, row_count: rows });
    }
    let effective_inner = BigUint::from(left.matrix_type.columns) - known_zero_rows;
    let ring_factor =
        if left.metadata.is_constant_polynomial || right.metadata.is_constant_polynomial {
            BigUint::one()
        } else {
            BigUint::from(left.matrix_type.ring_dimension)
        };
    Ok(MatrixBound {
        matrix_type: ConcreteMatrixType {
            modulus: left.matrix_type.modulus.clone(),
            ring_dimension: left.matrix_type.ring_dimension,
            rows: left.matrix_type.rows,
            columns: right.matrix_type.columns,
        },
        coefficient_class: multiply_classes(
            &left.coefficient_class,
            &right.coefficient_class,
            &(effective_inner * ring_factor),
        ),
        metadata: MatrixMetadata {
            is_constant_polynomial: left.metadata.is_constant_polynomial &&
                right.metadata.is_constant_polynomial,
            known_zero_rows: None,
        },
    })
}

fn multiply_classes(left: &BoundClass, right: &BoundClass, factor: &BigUint) -> BoundClass {
    match (left, right) {
        (BoundClass::ExactZero, _) | (_, BoundClass::ExactZero) => BoundClass::ExactZero,
        (
            BoundClass::Bounded { maximum_absolute_coefficient: left },
            BoundClass::Bounded { maximum_absolute_coefficient: right },
        ) => BoundClass::Bounded { maximum_absolute_coefficient: factor * left * right },
        _ => BoundClass::Large,
    }
}

fn matrix_children(node: &MxxLang, term: Id) -> Result<Vec<Id>, BoundEvaluationError> {
    use MxxLang::*;
    match node {
        Atom { .. } |
        MatrixConstant(_) |
        HashPlain { .. } |
        LiftConstantPolynomial { .. } |
        PackPolynomialCoefficients { .. } => Ok(Vec::new()),
        MatrixCanonicalRangeContract { input, .. } |
        MatrixNegate(input) |
        MatrixTranspose(input) => Ok(vec![input[0]]),
        MatrixSlice { input, .. } => Ok(vec![input[0]]),
        MatrixAdd(children) |
        MatrixMultiply(children) |
        MatrixConcat { inputs: children, .. } |
        CrtRecompose { inputs: children, .. } => (!children.is_empty())
            .then(|| children.to_vec())
            .ok_or(BoundEvaluationError::EmptyMatrixOperation { term }),
        MatrixScale(children) => Ok(vec![children[1]]),
        MatrixTensor(children) => Ok(children.to_vec()),
        Switch(children) => {
            let Some((_, cases)) = children.split_first() else {
                return Err(BoundEvaluationError::EmptyMatrixOperation { term });
            };
            (!cases.is_empty())
                .then(|| cases.to_vec())
                .ok_or(BoundEvaluationError::EmptyMatrixOperation { term })
        }
        IntConst(_) |
        IntParameter(_) |
        IntBinder(_) |
        IntAdd(_) |
        IntSub(_) |
        IntMul(_) |
        IntExactDiv(_) |
        IntEuclideanDiv(_) |
        IntEuclideanRemainder(_) |
        IntRoundDiv(_) |
        IntLog2Ceil(_) |
        BoolConst(_) |
        IntEqual(_) |
        IntLess(_) |
        IntLessEqual(_) |
        BitExtract { .. } |
        BoolToInt(_) |
        RealConst(_) |
        IntToReal(_) |
        RealAdd(_) |
        RealSub(_) |
        RealMul(_) |
        RealDiv(_) |
        RealSqrt(_) |
        ExtractCoefficient(_) => Err(BoundEvaluationError::NonMatrixTerm { term }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::identity::Axis;
    use num_bigint::BigInt;
    use std::collections::BTreeMap;

    #[derive(Default)]
    struct Input {
        nodes: BTreeMap<Id, MxxLang>,
        types: BTreeMap<Id, ConcreteMatrixType>,
        atoms: BTreeMap<AtomicSourceId, MatrixBound>,
        constants: BTreeMap<u32, (ConcreteMatrixType, ResolvedMatrixConstant)>,
        crt_coefficients: BTreeMap<u32, Box<[BigInt]>>,
        scalars: BTreeMap<Id, BigUint>,
        maximum_integer_bits: Option<BigUint>,
    }

    impl BoundInput for Input {
        fn node(&self, term: Id) -> Option<&MxxLang> {
            self.nodes.get(&term)
        }
        fn matrix_type(&self, term: Id) -> Result<ConcreteMatrixType, BoundEvaluationError> {
            self.types.get(&term).cloned().ok_or(BoundEvaluationError::NonMatrixTerm { term })
        }
        fn atom_bound(
            &self,
            source: AtomicSourceId,
            _: Id,
        ) -> Result<MatrixBound, BoundEvaluationError> {
            self.atoms
                .get(&source)
                .cloned()
                .ok_or(BoundEvaluationError::InvalidMatrixConstant { term: Id::from(0) })
        }
        fn matrix_constant(
            &self,
            spec: MatrixConstantSpecId,
            term: Id,
        ) -> Result<(ConcreteMatrixType, ResolvedMatrixConstant), BoundEvaluationError> {
            self.constants
                .get(&spec.0)
                .cloned()
                .ok_or(BoundEvaluationError::InvalidMatrixConstant { term })
        }
        fn scalar_maximum_absolute(&self, term: Id) -> Result<BigUint, BoundEvaluationError> {
            Ok(self.scalars.get(&term).cloned().unwrap_or_else(BigUint::one))
        }
        fn lift_constant_polynomial_class(
            &self,
            _: Id,
            _: Id,
        ) -> Result<BoundClass, BoundEvaluationError> {
            Ok(BoundClass::Large)
        }
        fn crt_coefficients(
            &self,
            spec: CrtSpecId,
            term: Id,
        ) -> Result<Box<[BigInt]>, BoundEvaluationError> {
            self.crt_coefficients
                .get(&spec.0)
                .cloned()
                .ok_or(BoundEvaluationError::InvalidCrtRecompose { term })
        }
        fn validate_integer(
            &self,
            _: &BigUint,
            _: &'static str,
        ) -> Result<(), BoundEvaluationError> {
            Ok(())
        }
        fn validate_integer_bits(
            &self,
            bits: &BigUint,
            operation: &'static str,
        ) -> Result<(), BoundEvaluationError> {
            if self.maximum_integer_bits.as_ref().is_some_and(|maximum| bits > maximum) {
                return Err(BoundEvaluationError::IntegerBitLimitExceeded {
                    operation,
                    bits: bits.clone(),
                });
            }
            Ok(())
        }
        fn validate_pack(&self, _: Id, _: usize) -> Result<(), BoundEvaluationError> {
            Ok(())
        }
    }

    fn matrix(rows: usize, columns: usize) -> ConcreteMatrixType {
        ConcreteMatrixType { modulus: BigInt::from(17), ring_dimension: 1, rows, columns }
    }

    fn bounded(matrix_type: ConcreteMatrixType, value: u64) -> MatrixBound {
        MatrixBound {
            matrix_type,
            coefficient_class: BoundClass::Bounded { maximum_absolute_coefficient: value.into() },
            metadata: MatrixMetadata::unknown(),
        }
    }

    #[test]
    fn product_uses_only_proved_zero_rows_and_constant_metadata() {
        let left = bounded(matrix(2, 3), 2);
        let mut right = bounded(matrix(3, 4), 5);
        right.metadata.known_zero_rows = Some(1_u8.into());
        let actual = product_bound(&left, &right).unwrap();
        assert_eq!(
            actual.coefficient_class,
            BoundClass::Bounded { maximum_absolute_coefficient: 20_u8.into() }
        );
    }

    #[test]
    fn large_does_not_receive_a_numeric_fallback() {
        let left = MatrixBound {
            matrix_type: matrix(1, 1),
            coefficient_class: BoundClass::Large,
            metadata: MatrixMetadata::unknown(),
        };
        let right = bounded(matrix(1, 1), 3);
        assert_eq!(product_bound(&left, &right).unwrap().coefficient_class, BoundClass::Large);
    }

    #[test]
    fn iterative_evaluation_memos_shared_matrix_child_once() {
        let atom = Id::from(0);
        let root = Id::from(1);
        let mut input = Input::default();
        input
            .nodes
            .insert(atom, MxxLang::Atom { source: AtomicSourceId(0), indices: Box::new([]) });
        input.nodes.insert(root, MxxLang::MatrixAdd(vec![atom, atom].into_boxed_slice()));
        input.types.insert(atom, matrix(1, 1));
        input.types.insert(root, matrix(1, 1));
        input.atoms.insert(AtomicSourceId(0), bounded(matrix(1, 1), 4));

        let evaluator = BoundEvaluator::new(&input);
        let result = evaluator.evaluate(root).unwrap();
        assert_eq!(
            result.coefficient_class,
            BoundClass::Bounded { maximum_absolute_coefficient: 8_u8.into() }
        );
    }

    #[test]
    fn switch_uses_maximum_not_triangle_sum() {
        let first = Id::from(0);
        let second = Id::from(1);
        let root = Id::from(2);
        let mut input = Input::default();
        input
            .nodes
            .insert(first, MxxLang::Atom { source: AtomicSourceId(0), indices: Box::new([]) });
        input
            .nodes
            .insert(second, MxxLang::Atom { source: AtomicSourceId(1), indices: Box::new([]) });
        input
            .nodes
            .insert(root, MxxLang::Switch(vec![Id::from(9), first, second].into_boxed_slice()));
        for id in [first, second, root] {
            input.types.insert(id, matrix(1, 1));
        }
        input.atoms.insert(AtomicSourceId(0), bounded(matrix(1, 1), 4));
        input.atoms.insert(AtomicSourceId(1), bounded(matrix(1, 1), 7));
        let result = BoundEvaluator::new(&input).evaluate(root).unwrap();
        assert_eq!(
            result.coefficient_class,
            BoundClass::Bounded { maximum_absolute_coefficient: 7_u8.into() }
        );
        let _ = Axis::Rows;
    }

    #[test]
    fn rotation_and_polynomial_constant_metadata_are_exact() {
        let rotation = Id::from(0);
        let polynomial = Id::from(1);
        let root = Id::from(2);
        let mut input = Input::default();
        input.nodes.insert(rotation, MxxLang::MatrixConstant(MatrixConstantSpecId(0)));
        input.nodes.insert(polynomial, MxxLang::MatrixConstant(MatrixConstantSpecId(1)));
        input.nodes.insert(root, MxxLang::MatrixMultiply(vec![rotation, polynomial].into()));
        for id in [rotation, polynomial, root] {
            input.types.insert(id, matrix(1, 1));
        }
        input
            .constants
            .insert(0, (matrix(1, 1), ResolvedMatrixConstant::Rotation { exponent: 1.into() }));
        input.constants.insert(
            1,
            (
                matrix(1, 1),
                ResolvedMatrixConstant::Polynomial {
                    coefficients: vec![3.into(), 2.into()].into(),
                },
            ),
        );

        let result = BoundEvaluator::new(&input).evaluate(root).unwrap();
        assert!(!result.metadata.is_constant_polynomial);
    }

    #[test]
    fn zero_scale_consumes_large_and_final_large_is_rejected() {
        let large = Id::from(0);
        let scaled = Id::from(1);
        let scalar = Id::from(9);
        let mut input = Input::default();
        input.nodes.insert(
            large,
            MxxLang::HashPlain {
                query: super::super::identity::HashQuerySpecId(0),
                arguments: Box::new([]),
            },
        );
        input.nodes.insert(scaled, MxxLang::MatrixScale([scalar, large]));
        input.types.insert(large, matrix(1, 1));
        input.types.insert(scaled, matrix(1, 1));
        input.scalars.insert(scalar, BigUint::zero());

        assert_eq!(
            BoundEvaluator::new(&input).evaluate(scaled).unwrap().coefficient_class,
            BoundClass::ExactZero,
        );
        assert_eq!(
            BoundEvaluator::new(&input).evaluate(large),
            Err(BoundEvaluationError::UnconsumedLargeTerm { term: large }),
        );
    }

    #[test]
    fn power_of_base_is_preflighted_before_pow() {
        let root = Id::from(0);
        let mut input = Input::default();
        input.nodes.insert(root, MxxLang::MatrixConstant(MatrixConstantSpecId(0)));
        input.types.insert(root, matrix(1, 1));
        input.constants.insert(
            0,
            (
                matrix(1, 1),
                ResolvedMatrixConstant::PowerOfBase {
                    base: 2.into(),
                    exponent: 1_000_000_u32.into(),
                },
            ),
        );
        input.maximum_integer_bits = Some(32_u32.into());

        assert_eq!(
            BoundEvaluator::new(&input).evaluate(root),
            Err(BoundEvaluationError::IntegerBitLimitExceeded {
                operation: "power-of-base constant",
                bits: 2_000_000_u32.into(),
            }),
        );
    }

    #[test]
    fn tensor_concat_switch_and_crt_reject_mismatched_shapes() {
        let first = Id::from(0);
        let second = Id::from(1);
        let tensor = Id::from(2);
        let concat = Id::from(3);
        let switched = Id::from(4);
        let crt = Id::from(5);
        let mut input = Input::default();
        input
            .nodes
            .insert(first, MxxLang::Atom { source: AtomicSourceId(0), indices: Box::new([]) });
        input
            .nodes
            .insert(second, MxxLang::Atom { source: AtomicSourceId(1), indices: Box::new([]) });
        input.nodes.insert(tensor, MxxLang::MatrixTensor([first, second]));
        input.nodes.insert(
            concat,
            MxxLang::MatrixConcat { axis: Axis::Rows, inputs: vec![first, second].into() },
        );
        input.nodes.insert(switched, MxxLang::Switch(vec![Id::from(9), first, second].into()));
        input.nodes.insert(
            crt,
            MxxLang::CrtRecompose { spec: CrtSpecId(0), inputs: vec![first, second].into() },
        );
        input.types.insert(first, matrix(1, 1));
        input.types.insert(second, matrix(2, 1));
        input.types.insert(tensor, matrix(1, 1));
        input.types.insert(concat, matrix(3, 1));
        input.types.insert(switched, matrix(1, 1));
        input.types.insert(crt, matrix(1, 1));
        input.atoms.insert(AtomicSourceId(0), bounded(matrix(1, 1), 1));
        input.atoms.insert(AtomicSourceId(1), bounded(matrix(2, 1), 1));
        input.crt_coefficients.insert(0, vec![1.into(), 1.into()].into());

        for root in [tensor, switched, crt] {
            assert!(matches!(
                BoundEvaluator::new(&input).evaluate(root),
                Err(BoundEvaluationError::IncompatibleMatrixProduct { .. }),
            ));
        }
        assert_eq!(BoundEvaluator::new(&input).evaluate(concat).unwrap().matrix_type, matrix(3, 1),);
    }
}
