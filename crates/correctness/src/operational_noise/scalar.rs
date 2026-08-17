//! Typed, job-local scalar storage for operational-noise lowering.
//!
//! Matrix values are owned by [`super::normal_form::ExpressionDag`].  This
//! module owns only scalar/domain values and their facts.  Nodes are inserted
//! bottom-up and are hash-consed by their typed structural identity; facts are
//! deliberately kept outside that identity so a direct-extraction proof can
//! be merged conservatively without changing the semantic key.

use super::{
    error::AnalysisError,
    identity::{
        AtomicSourceId, BinderKey, CanonicalResidueConvention, ResolvedIntBinaryOperation,
        ResolvedIntExpr, ResolvedIntExprArena, ResolvedIntExprArenaNode, ResolvedMatrixType,
        SymbolTables,
    },
};
use mxx_ir_core::{IntExpr, ParamEnv, expr::euclidean_div_rem};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Signed, Zero};
use std::collections::{BTreeMap, HashMap};

/// The closed scalar and boundary sort vocabulary.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ScalarSort {
    Int,
    Bool,
    Real,
    Bytes(ResolvedIntExpr),
    TypedBlob { type_name: String, schema_hash: [u8; 32] },
    Matrix(ResolvedMatrixType),
}

impl ScalarSort {
    pub const fn is_scalar(&self) -> bool {
        matches!(self, Self::Int | Self::Bool | Self::Real)
    }

    pub const fn permits_selector_provenance(&self) -> bool {
        matches!(self, Self::Int | Self::Bool)
    }

    pub const fn permits_scalar_provenance(&self) -> bool {
        self.permits_selector_provenance()
    }
}

/// A closed inclusive integer interval.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct IntegerInterval {
    pub minimum: BigInt,
    pub maximum: BigInt,
}

impl IntegerInterval {
    pub fn new(minimum: BigInt, maximum: BigInt) -> Option<Self> {
        (minimum <= maximum).then_some(Self { minimum, maximum })
    }

    pub fn exact(value: BigInt) -> Self {
        Self { minimum: value.clone(), maximum: value }
    }

    pub fn contains_zero(&self) -> bool {
        self.minimum <= BigInt::zero() && BigInt::zero() <= self.maximum
    }

    pub fn is_exact(&self) -> bool {
        self.minimum == self.maximum
    }

    pub fn add(&self, other: &Self) -> Self {
        Self { minimum: &self.minimum + &other.minimum, maximum: &self.maximum + &other.maximum }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self { minimum: &self.minimum - &other.maximum, maximum: &self.maximum - &other.minimum }
    }

    pub fn mul(&self, other: &Self) -> Self {
        let products = [
            &self.minimum * &other.minimum,
            &self.minimum * &other.maximum,
            &self.maximum * &other.minimum,
            &self.maximum * &other.maximum,
        ];
        Self {
            minimum: products.iter().min().expect("four products").clone(),
            maximum: products.iter().max().expect("four products").clone(),
        }
    }

    pub fn hull(&self, other: &Self) -> Self {
        Self {
            minimum: self.minimum.clone().min(other.minimum.clone()),
            maximum: self.maximum.clone().max(other.maximum.clone()),
        }
    }
}

/// Closed affine or interval domain.  Affine expressions retain owner-aware
/// binders and never enumerate their Cartesian product.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IntegerDomain {
    Exact(BigInt),
    Affine {
        constant: BigInt,
        coefficients: BTreeMap<BinderKey, BigInt>,
        binders: BTreeMap<BinderKey, IntegerInterval>,
    },
    IntervalOnly(IntegerInterval),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntegerDomainError {
    DivisionByZero,
    NonExactDivisor,
    InexactDivision,
    InvalidRoundDivisor,
    InvalidLog2CeilArgument,
    InvalidAffine,
}

impl IntegerDomain {
    pub fn exact(value: impl Into<BigInt>) -> Self {
        Self::Exact(value.into())
    }

    pub fn interval(&self) -> Result<IntegerInterval, IntegerDomainError> {
        match self {
            Self::Exact(value) => Ok(IntegerInterval::exact(value.clone())),
            Self::IntervalOnly(interval) => Ok(interval.clone()),
            Self::Affine { constant, coefficients, binders } => {
                let mut minimum = constant.clone();
                let mut maximum = constant.clone();
                for (binder, coefficient) in coefficients {
                    let interval = binders.get(binder).ok_or(IntegerDomainError::InvalidAffine)?;
                    if coefficient.is_negative() {
                        minimum += coefficient * &interval.maximum;
                        maximum += coefficient * &interval.minimum;
                    } else {
                        minimum += coefficient * &interval.minimum;
                        maximum += coefficient * &interval.maximum;
                    }
                }
                if binders.keys().any(|binder| !coefficients.contains_key(binder)) {
                    return Err(IntegerDomainError::InvalidAffine);
                }
                Ok(IntegerInterval { minimum, maximum })
            }
        }
    }

    pub fn hull(&self, other: &Self) -> Result<Self, IntegerDomainError> {
        if self == other {
            return Ok(self.clone());
        }
        Ok(Self::IntervalOnly(self.interval()?.hull(&other.interval()?)))
    }

    pub fn add(&self, other: &Self) -> Result<Self, IntegerDomainError> {
        self.combine_affine(other, false)
    }

    pub fn sub(&self, other: &Self) -> Result<Self, IntegerDomainError> {
        self.combine_affine(other, true)
    }

    fn combine_affine(&self, other: &Self, subtract: bool) -> Result<Self, IntegerDomainError> {
        let Some((left_constant, left_coefficients, left_binders)) = self.affine_parts() else {
            let left = self.interval()?;
            let right = other.interval()?;
            return Ok(Self::IntervalOnly(if subtract {
                left.sub(&right)
            } else {
                left.add(&right)
            }));
        };
        let Some((right_constant, right_coefficients, right_binders)) = other.affine_parts() else {
            let left = self.interval()?;
            let right = other.interval()?;
            return Ok(Self::IntervalOnly(if subtract {
                left.sub(&right)
            } else {
                left.add(&right)
            }));
        };
        let mut coefficients = left_coefficients;
        let mut binders = left_binders;
        for (binder, coefficient) in right_coefficients {
            let delta = if subtract { -coefficient } else { coefficient };
            let entry = coefficients.entry(binder.clone()).or_insert_with(BigInt::zero);
            *entry += delta;
            if entry.is_zero() {
                coefficients.remove(&binder);
            }
        }
        for (binder, interval) in right_binders {
            match binders.get(&binder) {
                Some(existing) if existing != &interval => {
                    return Err(IntegerDomainError::InvalidAffine)
                }
                Some(_) => {}
                None => {
                    binders.insert(binder, interval);
                }
            }
        }
        binders.retain(|binder, _| coefficients.contains_key(binder));
        Ok(Self::from_affine(
            if subtract { left_constant - right_constant } else { left_constant + right_constant },
            coefficients,
            binders,
        ))
    }

    pub fn mul(&self, other: &Self) -> Result<Self, IntegerDomainError> {
        match (self.exact_value(), other.exact_value()) {
            (Some(left), _) => other.scale(left),
            (_, Some(right)) => self.scale(right),
            _ => Ok(Self::IntervalOnly(self.interval()?.mul(&other.interval()?))),
        }
    }

    pub fn exact_div(&self, divisor: &Self) -> Result<Self, IntegerDomainError> {
        let divisor = divisor.exact_value().ok_or(IntegerDomainError::NonExactDivisor)?;
        if divisor.is_zero() {
            return Err(IntegerDomainError::DivisionByZero);
        }
        match self {
            Self::Exact(value) => exact_quotient(value, divisor).map(Self::Exact),
            Self::Affine { constant, coefficients, binders } => {
                let constant = exact_quotient(constant, divisor)?;
                let mut divided = BTreeMap::new();
                for (binder, coefficient) in coefficients {
                    divided.insert(binder.clone(), exact_quotient(coefficient, divisor)?);
                }
                Ok(Self::from_affine(constant, divided, binders.clone()))
            }
            Self::IntervalOnly(interval) if interval.is_exact() => {
                exact_quotient(&interval.minimum, divisor).map(Self::Exact)
            }
            Self::IntervalOnly(_) => Err(IntegerDomainError::InexactDivision),
        }
    }

    pub fn round_div(&self, divisor: &Self) -> Result<Self, IntegerDomainError> {
        let divisor = divisor.exact_value().ok_or(IntegerDomainError::NonExactDivisor)?;
        if *divisor <= BigInt::zero() {
            return Err(IntegerDomainError::InvalidRoundDivisor);
        }
        let interval = self.interval()?;
        Ok(Self::IntervalOnly(IntegerInterval {
            minimum: evaluate_round_div(&interval.minimum, divisor)?,
            maximum: evaluate_round_div(&interval.maximum, divisor)?,
        }))
    }

    pub fn log2_ceil(&self) -> Result<Self, IntegerDomainError> {
        let interval = self.interval()?;
        if interval.minimum < BigInt::one() {
            return Err(IntegerDomainError::InvalidLog2CeilArgument);
        }
        Ok(Self::IntervalOnly(IntegerInterval {
            minimum: log2_ceil(&interval.minimum),
            maximum: log2_ceil(&interval.maximum),
        }))
    }

    pub fn euclidean_div(&self, divisor: &Self) -> Result<Self, IntegerDomainError> {
        let divisor = divisor.exact_value().ok_or(IntegerDomainError::NonExactDivisor)?;
        if divisor.is_zero() {
            return Err(IntegerDomainError::DivisionByZero);
        }
        let interval = self.interval()?;
        let _ = euclidean_div_rem(&BigInt::zero(), divisor)
            .map_err(|_| IntegerDomainError::DivisionByZero)?;
        let (minimum, _) = euclidean_div_rem(&interval.minimum, divisor)
            .map_err(|_| IntegerDomainError::DivisionByZero)?;
        let (maximum, _) = euclidean_div_rem(&interval.maximum, divisor)
            .map_err(|_| IntegerDomainError::DivisionByZero)?;
        Ok(Self::IntervalOnly(IntegerInterval { minimum, maximum }))
    }

    pub fn euclidean_remainder(&self, divisor: &Self) -> Result<Self, IntegerDomainError> {
        let divisor = divisor.exact_value().ok_or(IntegerDomainError::NonExactDivisor)?;
        if divisor.is_zero() {
            return Err(IntegerDomainError::DivisionByZero);
        }
        if let Some(value) = self.exact_value() {
            let (_, remainder) = euclidean_div_rem(value, divisor)
                .map_err(|_| IntegerDomainError::DivisionByZero)?;
            return Ok(Self::Exact(remainder));
        }
        Ok(Self::IntervalOnly(IntegerInterval {
            minimum: BigInt::zero(),
            maximum: divisor.abs() - BigInt::one(),
        }))
    }

    pub fn exact_value(&self) -> Option<&BigInt> {
        match self {
            Self::Exact(value) => Some(value),
            Self::IntervalOnly(interval) if interval.is_exact() => Some(&interval.minimum),
            Self::Affine { constant, coefficients, .. } if coefficients.is_empty() => {
                Some(constant)
            }
            _ => None,
        }
    }

    fn affine_parts(
        &self,
    ) -> Option<(BigInt, BTreeMap<BinderKey, BigInt>, BTreeMap<BinderKey, IntegerInterval>)> {
        match self {
            Self::Exact(value) => Some((value.clone(), BTreeMap::new(), BTreeMap::new())),
            Self::Affine { constant, coefficients, binders } => {
                Some((constant.clone(), coefficients.clone(), binders.clone()))
            }
            Self::IntervalOnly(_) => None,
        }
    }

    fn scale(&self, scalar: &BigInt) -> Result<Self, IntegerDomainError> {
        match self {
            Self::Exact(value) => Ok(Self::Exact(value * scalar)),
            Self::Affine { constant, coefficients, binders } => Ok(Self::from_affine(
                constant * scalar,
                coefficients
                    .iter()
                    .filter_map(|(binder, coefficient)| {
                        let product = coefficient * scalar;
                        (!product.is_zero()).then_some((binder.clone(), product))
                    })
                    .collect(),
                binders.clone(),
            )),
            Self::IntervalOnly(interval) => {
                Ok(Self::IntervalOnly(interval.mul(&IntegerInterval::exact(scalar.clone()))))
            }
        }
    }

    fn from_affine(
        constant: BigInt,
        coefficients: BTreeMap<BinderKey, BigInt>,
        mut binders: BTreeMap<BinderKey, IntegerInterval>,
    ) -> Self {
        let coefficients = coefficients
            .into_iter()
            .filter(|(_, coefficient)| !coefficient.is_zero())
            .collect::<BTreeMap<_, _>>();
        binders.retain(|binder, _| coefficients.contains_key(binder));
        if coefficients.is_empty() {
            Self::Exact(constant)
        } else {
            Self::Affine { constant, coefficients, binders }
        }
    }
}

fn exact_quotient(value: &BigInt, divisor: &BigInt) -> Result<BigInt, IntegerDomainError> {
    if divisor.is_zero() {
        return Err(IntegerDomainError::DivisionByZero);
    }
    let quotient = value / divisor;
    (value % divisor).is_zero().then_some(quotient).ok_or(IntegerDomainError::InexactDivision)
}

fn evaluate_round_div(value: &BigInt, divisor: &BigInt) -> Result<BigInt, IntegerDomainError> {
    IntExpr::RoundDiv(
        Box::new(IntExpr::constant(value.clone())),
        Box::new(IntExpr::constant(divisor.clone())),
    )
    .evaluate(&ParamEnv::default())
    .map_err(|_| IntegerDomainError::InvalidRoundDivisor)
}

fn log2_ceil(value: &BigInt) -> BigInt {
    let value = value.to_biguint().expect("positive input checked by caller");
    let floor = value.bits() - 1;
    let power = BigUint::one() << usize::try_from(floor).expect("bit count fits usize");
    BigInt::from(if value == power { floor } else { floor + 1 })
}

/// A selector-only scalar remains forbidden from ordinary arithmetic/noise use.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ScalarProvenance {
    Ordinary,
    SelectorOnly,
}

impl ScalarProvenance {
    pub fn merge(self, other: Self) -> Self {
        if matches!(self, Self::SelectorOnly) || matches!(other, Self::SelectorOnly) {
            Self::SelectorOnly
        } else {
            Self::Ordinary
        }
    }

    pub fn runtime_arithmetic(self, other: Self) -> Option<Self> {
        matches!((self, other), (Self::Ordinary, Self::Ordinary)).then_some(Self::Ordinary)
    }

    pub fn euclidean(self, divisor: Self) -> Option<Self> {
        matches!(divisor, Self::Ordinary).then_some(self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectExtractFact {
    pub canonical_upper: Option<BigUint>,
}

/// Facts attached to one scalar entry.  They are not part of the structural
/// key, allowing only the explicitly conservative direct-extraction merge.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScalarFacts {
    pub sort: Result<ScalarSort, AnalysisError>,
    pub integer_domain: Option<IntegerDomain>,
    pub scalar_provenance: Option<ScalarProvenance>,
    pub possible_false: bool,
    pub possible_true: bool,
    pub real_constant_bits: Option<u64>,
    pub canonical_coefficient_exclusive_upper: Option<BigUint>,
    pub canonical_residue_convention: Option<CanonicalResidueConvention>,
    pub direct_extract: Option<DirectExtractFact>,
}

impl ScalarFacts {
    pub fn scalar(
        sort: ScalarSort,
        domain: Option<IntegerDomain>,
        provenance: ScalarProvenance,
    ) -> Self {
        Self {
            sort: Ok(sort),
            integer_domain: domain,
            scalar_provenance: Some(provenance),
            possible_false: false,
            possible_true: false,
            real_constant_bits: None,
            canonical_coefficient_exclusive_upper: None,
            canonical_residue_convention: None,
            direct_extract: None,
        }
    }

    pub fn matrix(sort: ResolvedMatrixType, canonical_upper: Option<BigUint>) -> Self {
        Self {
            sort: Ok(ScalarSort::Matrix(sort)),
            integer_domain: None,
            scalar_provenance: None,
            possible_false: false,
            possible_true: false,
            real_constant_bits: None,
            canonical_coefficient_exclusive_upper: canonical_upper,
            canonical_residue_convention: None,
            direct_extract: None,
        }
    }

    pub fn boolean(
        possible_false: bool,
        possible_true: bool,
        provenance: ScalarProvenance,
    ) -> Self {
        let mut data = Self::scalar(ScalarSort::Bool, None, provenance);
        data.possible_false = possible_false;
        data.possible_true = possible_true;
        data
    }

    fn sort_only(sort: ScalarSort) -> Self {
        Self {
            sort: Ok(sort),
            integer_domain: None,
            scalar_provenance: None,
            possible_false: false,
            possible_true: false,
            real_constant_bits: None,
            canonical_coefficient_exclusive_upper: None,
            canonical_residue_convention: None,
            direct_extract: None,
        }
    }

    pub fn merge_direct_extract(&mut self, from: Self) {
        if self.sort != from.sort || self.scalar_provenance != from.scalar_provenance {
            return;
        }
        self.integer_domain = match (&self.integer_domain, &from.integer_domain) {
            (Some(left), Some(right)) => left.hull(right).ok(),
            _ => None,
        };
        self.possible_false |= from.possible_false;
        self.possible_true |= from.possible_true;
        if self.real_constant_bits != from.real_constant_bits {
            self.real_constant_bits = None;
        }
        match (&mut self.direct_extract, from.direct_extract) {
            (Some(left), Some(right)) => {
                if let (Some(a), Some(b)) = (&left.canonical_upper, &right.canonical_upper) {
                    if b < a {
                        left.canonical_upper = Some(b.clone());
                    }
                } else if left.canonical_upper.is_none() {
                    left.canonical_upper = right.canonical_upper;
                }
            }
            (None, right @ Some(_)) => self.direct_extract = right,
            _ => {}
        }
    }

    /// Pure bottom-up transfer for every scalar operation emitted by Graph IR.
    /// The transfer consumes already computed child facts and never consults a
    /// typed scalar identity or runtime observation.
    pub fn transfer(
        node: &ScalarNode,
        children: &[&ScalarFacts],
        symbols: &SymbolTables,
    ) -> Result<Self, ScalarTransferError> {
        let child = |index: usize| children.get(index).copied().ok_or(ScalarTransferError::Arity);
        let int_binary = |left: &ScalarFacts,
                          right: &ScalarFacts,
                          operation: ScalarOperation|
         -> Result<Self, ScalarTransferError> {
            let left = require_int(left)?;
            let right = require_int(right)?;
            let left_domain = left.integer_domain.as_ref().ok_or(ScalarTransferError::Domain)?;
            let right_domain = right.integer_domain.as_ref().ok_or(ScalarTransferError::Domain)?;
            let domain = match operation {
                ScalarOperation::Add => left_domain.add(right_domain),
                ScalarOperation::Sub => left_domain.sub(right_domain),
                ScalarOperation::Mul => left_domain.mul(right_domain),
                ScalarOperation::ExactDiv => left_domain.exact_div(right_domain),
                ScalarOperation::EuclideanDiv => left_domain.euclidean_div(right_domain),
                ScalarOperation::EuclideanRemainder => {
                    left_domain.euclidean_remainder(right_domain)
                }
                ScalarOperation::RoundDiv => left_domain.round_div(right_domain),
                _ => return Err(ScalarTransferError::Unsupported),
            }
            .map_err(|_| ScalarTransferError::Domain)?;
            let provenance = left
                .scalar_provenance
                .zip(right.scalar_provenance)
                .and_then(|(left, right)| match operation {
                    ScalarOperation::EuclideanDiv | ScalarOperation::EuclideanRemainder => {
                        left.euclidean(right)
                    }
                    _ => left.runtime_arithmetic(right),
                })
                .ok_or(ScalarTransferError::SelectorOnly)?;
            Ok(Self::scalar(ScalarSort::Int, Some(domain), provenance))
        };
        match node {
            ScalarNode::Source { source, .. } => {
                let descriptor = symbols
                    .atomic_sources
                    .get(source.0)
                    .ok_or(ScalarTransferError::MissingSource)?;
                let sort = descriptor.sort.clone();
                match sort {
                    ScalarSort::Int => {
                        let domain = descriptor
                            .integer_domain
                            .as_ref()
                            .and_then(|domain| {
                                IntegerInterval::new(domain.minimum.clone(), domain.maximum.clone())
                            })
                            .ok_or(ScalarTransferError::Domain)?;
                        let mut facts = Self::scalar(
                            ScalarSort::Int,
                            Some(if domain.is_exact() {
                                IntegerDomain::Exact(domain.minimum)
                            } else {
                                IntegerDomain::IntervalOnly(domain)
                            }),
                            ScalarProvenance::Ordinary,
                        );
                        facts.canonical_residue_convention =
                            descriptor.canonical_residue_convention;
                        Ok(facts)
                    }
                    ScalarSort::Bool => {
                        let mut facts = Self::boolean(true, true, ScalarProvenance::Ordinary);
                        facts.canonical_residue_convention =
                            descriptor.canonical_residue_convention;
                        Ok(facts)
                    }
                    ScalarSort::Real | ScalarSort::Bytes(_) | ScalarSort::TypedBlob { .. } => {
                        Ok(Self::sort_only(sort))
                    }
                    ScalarSort::Matrix(_) => Err(ScalarTransferError::NeedsMatrixContract),
                }
            }
            ScalarNode::IntConst(value) => Ok(Self::scalar(
                ScalarSort::Int,
                Some(IntegerDomain::exact(value.clone())),
                ScalarProvenance::Ordinary,
            )),
            ScalarNode::IntParameter(name) => {
                let value = symbols.integer_parameters.get(name).cloned();
                Ok(Self::scalar(
                    ScalarSort::Int,
                    value.map(IntegerDomain::exact),
                    ScalarProvenance::Ordinary,
                ))
            }
            ScalarNode::IntBinder(binder) => {
                let descriptor = symbols
                    .binders
                    .values
                    .iter()
                    .find(|descriptor| &descriptor.key == binder)
                    .ok_or(ScalarTransferError::MissingBinder)?;
                let mut coefficients = BTreeMap::new();
                coefficients.insert(binder.clone(), BigInt::one());
                let mut binders = BTreeMap::new();
                binders.insert(
                    binder.clone(),
                    IntegerInterval {
                        minimum: descriptor.minimum.clone(),
                        maximum: descriptor.maximum.clone(),
                    },
                );
                Ok(Self::scalar(
                    ScalarSort::Int,
                    Some(IntegerDomain::Affine { constant: BigInt::zero(), coefficients, binders }),
                    ScalarProvenance::Ordinary,
                ))
            }
            ScalarNode::IntAdd(_) => int_binary(child(0)?, child(1)?, ScalarOperation::Add),
            ScalarNode::IntSub(_) => int_binary(child(0)?, child(1)?, ScalarOperation::Sub),
            ScalarNode::IntMul(_) => int_binary(child(0)?, child(1)?, ScalarOperation::Mul),
            ScalarNode::IntExactDiv(_) => {
                int_binary(child(0)?, child(1)?, ScalarOperation::ExactDiv)
            }
            ScalarNode::IntEuclideanDiv(_) => {
                int_binary(child(0)?, child(1)?, ScalarOperation::EuclideanDiv)
            }
            ScalarNode::IntEuclideanRemainder(_) => {
                int_binary(child(0)?, child(1)?, ScalarOperation::EuclideanRemainder)
            }
            ScalarNode::IntRoundDiv(_) => {
                int_binary(child(0)?, child(1)?, ScalarOperation::RoundDiv)
            }
            ScalarNode::IntLog2Ceil(_) => {
                let value = require_int(child(0)?)?;
                let domain = value
                    .integer_domain
                    .as_ref()
                    .ok_or(ScalarTransferError::Domain)?
                    .log2_ceil()
                    .map_err(|_| ScalarTransferError::Domain)?;
                Ok(Self::scalar(ScalarSort::Int, Some(domain), ScalarProvenance::Ordinary))
            }
            ScalarNode::BoolConst(value) => {
                Ok(Self::boolean(!*value, *value, ScalarProvenance::Ordinary))
            }
            ScalarNode::IntEqual(_) | ScalarNode::IntLess(_) | ScalarNode::IntLessEqual(_) => {
                let left = require_int(child(0)?)?;
                let right = require_int(child(1)?)?;
                let left = left
                    .integer_domain
                    .as_ref()
                    .ok_or(ScalarTransferError::Domain)?
                    .interval()
                    .map_err(|_| ScalarTransferError::Domain)?;
                let right = right
                    .integer_domain
                    .as_ref()
                    .ok_or(ScalarTransferError::Domain)?
                    .interval()
                    .map_err(|_| ScalarTransferError::Domain)?;
                let (possible_false, possible_true) = match node {
                    ScalarNode::IntEqual(_) => {
                        if left.maximum < right.minimum || right.maximum < left.minimum {
                            (true, false)
                        } else if left.is_exact() &&
                            right.is_exact() &&
                            left.minimum == right.minimum
                        {
                            (false, true)
                        } else {
                            (true, true)
                        }
                    }
                    ScalarNode::IntLess(_) => {
                        if left.maximum < right.minimum {
                            (false, true)
                        } else if left.minimum >= right.maximum {
                            (true, false)
                        } else {
                            (true, true)
                        }
                    }
                    ScalarNode::IntLessEqual(_) => {
                        if left.maximum <= right.minimum {
                            (false, true)
                        } else if left.minimum > right.maximum {
                            (true, false)
                        } else {
                            (true, true)
                        }
                    }
                    _ => unreachable!(),
                };
                Ok(Self::boolean(possible_false, possible_true, ScalarProvenance::Ordinary))
            }
            ScalarNode::BitExtract { .. } => {
                let value = require_int(child(0)?)?;
                let _ = value.scalar_provenance.ok_or(ScalarTransferError::Provenance)?;
                Ok(Self::boolean(true, true, ScalarProvenance::Ordinary))
            }
            ScalarNode::BoolToInt(_) => {
                let value = require_bool(child(0)?)?;
                if value.scalar_provenance != Some(ScalarProvenance::Ordinary) {
                    return Err(ScalarTransferError::SelectorOnly);
                }
                let domain = match (value.possible_false, value.possible_true) {
                    (true, false) => IntegerDomain::Exact(BigInt::zero()),
                    (false, true) => IntegerDomain::Exact(BigInt::one()),
                    (true, true) => IntegerDomain::IntervalOnly(IntegerInterval {
                        minimum: BigInt::zero(),
                        maximum: BigInt::one(),
                    }),
                    (false, false) => return Err(ScalarTransferError::Domain),
                };
                Ok(Self::scalar(ScalarSort::Int, Some(domain), ScalarProvenance::Ordinary))
            }
            ScalarNode::RealConst(bits) => Ok(Self {
                sort: Ok(ScalarSort::Real),
                integer_domain: None,
                scalar_provenance: None,
                possible_false: false,
                possible_true: false,
                real_constant_bits: Some(*bits),
                canonical_coefficient_exclusive_upper: None,
                canonical_residue_convention: None,
                direct_extract: None,
            }),
            ScalarNode::IntToReal(_) => {
                let value = require_int(child(0)?)?;
                if value.scalar_provenance != Some(ScalarProvenance::Ordinary) {
                    return Err(ScalarTransferError::SelectorOnly);
                }
                Ok(Self {
                    sort: Ok(ScalarSort::Real),
                    integer_domain: None,
                    scalar_provenance: None,
                    possible_false: false,
                    possible_true: false,
                    real_constant_bits: None,
                    canonical_coefficient_exclusive_upper: None,
                    canonical_residue_convention: None,
                    direct_extract: None,
                })
            }
            ScalarNode::RealAdd(_) |
            ScalarNode::RealSub(_) |
            ScalarNode::RealMul(_) |
            ScalarNode::RealDiv(_) => {
                let left = require_real(child(0)?)?;
                let right = require_real(child(1)?)?;
                let bits = match (left.real_constant_bits, right.real_constant_bits, node) {
                    (Some(left), Some(right), ScalarNode::RealAdd(_)) => {
                        Some((f64::from_bits(left) + f64::from_bits(right)).to_bits())
                    }
                    (Some(left), Some(right), ScalarNode::RealSub(_)) => {
                        Some((f64::from_bits(left) - f64::from_bits(right)).to_bits())
                    }
                    (Some(left), Some(right), ScalarNode::RealMul(_)) => {
                        Some((f64::from_bits(left) * f64::from_bits(right)).to_bits())
                    }
                    (Some(left), Some(right), ScalarNode::RealDiv(_)) => {
                        Some((f64::from_bits(left) / f64::from_bits(right)).to_bits())
                    }
                    _ => None,
                };
                Ok(Self {
                    sort: Ok(ScalarSort::Real),
                    integer_domain: None,
                    scalar_provenance: None,
                    possible_false: false,
                    possible_true: false,
                    real_constant_bits: bits,
                    canonical_coefficient_exclusive_upper: None,
                    canonical_residue_convention: None,
                    direct_extract: None,
                })
            }
            ScalarNode::RealSqrt(_) => {
                let value = require_real(child(0)?)?;
                let bits =
                    value.real_constant_bits.map(|value| f64::from_bits(value).sqrt().to_bits());
                Ok(Self {
                    sort: Ok(ScalarSort::Real),
                    integer_domain: None,
                    scalar_provenance: None,
                    possible_false: false,
                    possible_true: false,
                    real_constant_bits: bits,
                    canonical_coefficient_exclusive_upper: None,
                    canonical_residue_convention: None,
                    direct_extract: None,
                })
            }
            ScalarNode::Switch(_) => {
                let selector = require_int(child(0)?)?;
                let cases = &children[1..];
                if cases.is_empty() {
                    return Err(ScalarTransferError::Arity);
                }
                let first = cases[0];
                let mut result = first.clone();
                for case in &cases[1..] {
                    if case.sort != first.sort {
                        return Err(ScalarTransferError::Sort);
                    }
                    if let (Some(left), Some(right)) =
                        (&result.integer_domain, &case.integer_domain)
                    {
                        result.integer_domain =
                            Some(left.hull(right).map_err(|_| ScalarTransferError::Domain)?);
                    }
                    result.scalar_provenance = result
                        .scalar_provenance
                        .zip(case.scalar_provenance)
                        .map(|(left, right)| left.merge(right));
                    result.possible_false |= case.possible_false;
                    result.possible_true |= case.possible_true;
                    if result.real_constant_bits != case.real_constant_bits {
                        result.real_constant_bits = None;
                    }
                    result.direct_extract = None;
                }
                let selector_interval = selector
                    .integer_domain
                    .as_ref()
                    .ok_or(ScalarTransferError::Domain)?
                    .interval()
                    .map_err(|_| ScalarTransferError::Domain)?;
                let case_count = BigInt::from(cases.len());
                if selector_interval.minimum < BigInt::zero() ||
                    selector_interval.maximum >= case_count
                {
                    return Err(ScalarTransferError::Domain);
                }
                Ok(result)
            }
            ScalarNode::ExtractCoefficient { .. } => Err(ScalarTransferError::NeedsMatrixContract),
        }
    }
}

/// Computes the closed integer domain of a coefficient extracted from a
/// matrix using the authoritative residue convention and optional exclusive
/// upper premise.
pub fn extract_coefficient_domain(
    matrix: &ResolvedMatrixType,
    modulus: &BigUint,
    authoritative_upper: Option<&BigUint>,
    convention: CanonicalResidueConvention,
) -> Result<IntegerDomain, AnalysisError> {
    if modulus.is_zero() {
        return Err(AnalysisError::UnknownCanonicalResidueRange {
            matrix: ScalarSort::Matrix(matrix.clone()),
        });
    }
    if let Some(upper) = authoritative_upper {
        if convention != CanonicalResidueConvention::Nonnegative ||
            upper.is_zero() ||
            upper > modulus
        {
            return Err(AnalysisError::UnknownCanonicalResidueRange {
                matrix: ScalarSort::Matrix(matrix.clone()),
            });
        }
        return Ok(IntegerDomain::IntervalOnly(IntegerInterval {
            minimum: BigInt::zero(),
            maximum: BigInt::from(upper - BigUint::one()),
        }));
    }
    let (minimum, maximum) = match convention {
        CanonicalResidueConvention::Nonnegative => {
            (BigInt::zero(), BigInt::from(modulus - BigUint::one()))
        }
        CanonicalResidueConvention::Centered => {
            let floor = modulus / BigUint::from(2_u8);
            let ceil = (modulus + BigUint::one()) / BigUint::from(2_u8);
            (-BigInt::from(floor), BigInt::from(ceil) - BigInt::one())
        }
    };
    IntegerInterval::new(minimum, maximum).map(IntegerDomain::IntervalOnly).ok_or(
        AnalysisError::UnknownCanonicalResidueRange { matrix: ScalarSort::Matrix(matrix.clone()) },
    )
}

/// Constructs scalar facts at the sole matrix-to-scalar extraction boundary.
pub fn direct_extract_facts(
    matrix: ResolvedMatrixType,
    modulus: BigUint,
    authoritative_upper: Option<&BigUint>,
) -> Result<ScalarFacts, AnalysisError> {
    let domain = extract_coefficient_domain(
        &matrix,
        &modulus,
        authoritative_upper,
        CanonicalResidueConvention::Nonnegative,
    )?;
    let upper = domain.interval().ok().and_then(|interval| {
        (interval.minimum == BigInt::zero())
            .then(|| (&interval.maximum + BigInt::one()).to_biguint())
            .flatten()
    });
    let mut facts =
        ScalarFacts::scalar(ScalarSort::Int, Some(domain), ScalarProvenance::SelectorOnly);
    facts.direct_extract = Some(DirectExtractFact { canonical_upper: upper });
    Ok(facts)
}

pub(crate) fn resolved_constant(expression: &ResolvedIntExpr) -> Option<BigInt> {
    if let ResolvedIntExpr::Arena(arena) = expression {
        return resolved_arena_constant(arena);
    }
    enum Work<'a> {
        Enter(&'a ResolvedIntExpr),
        Add,
        Sub,
        Mul,
        Div,
        RoundDiv,
        Log2Ceil,
    }
    let mut values = Vec::new();
    let mut work = vec![Work::Enter(expression)];
    while let Some(item) = work.pop() {
        match item {
            Work::Enter(ResolvedIntExpr::Const(value)) => values.push(value.clone()),
            Work::Enter(ResolvedIntExpr::Parameter(_) | ResolvedIntExpr::Binder(_)) |
            Work::Enter(
                ResolvedIntExpr::Source { .. } |
                ResolvedIntExpr::EuclideanDiv(_, _) |
                ResolvedIntExpr::EuclideanRemainder(_, _) |
                ResolvedIntExpr::ExtractMatrixCoefficient { .. } |
                ResolvedIntExpr::Arena(_),
            ) => return None,
            Work::Enter(ResolvedIntExpr::Add(left, right)) => {
                work.extend([Work::Add, Work::Enter(right), Work::Enter(left)])
            }
            Work::Enter(ResolvedIntExpr::Sub(left, right)) => {
                work.extend([Work::Sub, Work::Enter(right), Work::Enter(left)])
            }
            Work::Enter(ResolvedIntExpr::Mul(left, right)) => {
                work.extend([Work::Mul, Work::Enter(right), Work::Enter(left)])
            }
            Work::Enter(ResolvedIntExpr::Div(left, right)) => {
                work.extend([Work::Div, Work::Enter(right), Work::Enter(left)])
            }
            Work::Enter(ResolvedIntExpr::RoundDiv(left, right)) => {
                work.extend([Work::RoundDiv, Work::Enter(right), Work::Enter(left)])
            }
            Work::Enter(ResolvedIntExpr::Log2Ceil(value)) => {
                work.extend([Work::Log2Ceil, Work::Enter(value)])
            }
            Work::Add | Work::Sub | Work::Mul | Work::Div | Work::RoundDiv => {
                let right = values.pop()?;
                let left = values.pop()?;
                let value = match item {
                    Work::Add => left + right,
                    Work::Sub => left - right,
                    Work::Mul => left * right,
                    Work::Div => exact_quotient(&left, &right).ok()?,
                    Work::RoundDiv => evaluate_round_div(&left, &right).ok()?,
                    _ => unreachable!(),
                };
                values.push(value);
            }
            Work::Log2Ceil => {
                let value = values.pop()?;
                (value >= BigInt::one()).then(|| values.push(log2_ceil(&value)))?;
            }
        }
    }
    (values.len() == 1).then(|| values.pop()).flatten()
}

/// Evaluates a descriptor-local identity without materializing a nested tree.
/// This is deliberately a semantic helper for closed-domain consumers; the
/// arena itself remains the canonical exported representation.
fn resolved_arena_constant(arena: &ResolvedIntExprArena) -> Option<BigInt> {
    let mut values = vec![None; arena.nodes.len()];
    for (index, node) in arena.nodes.iter().enumerate() {
        values[index] = match node {
            ResolvedIntExprArenaNode::Const(value) => Some(value.clone()),
            ResolvedIntExprArenaNode::Parameter(_) |
            ResolvedIntExprArenaNode::Binder(_) |
            ResolvedIntExprArenaNode::Source { .. } |
            ResolvedIntExprArenaNode::ExtractMatrixCoefficient { .. } => None,
            ResolvedIntExprArenaNode::Binary { operation, children } => {
                let left = values[children[0] as usize].clone()?;
                let right = values[children[1] as usize].clone()?;
                match operation {
                    ResolvedIntBinaryOperation::Add => Some(left + right),
                    ResolvedIntBinaryOperation::Sub => Some(left - right),
                    ResolvedIntBinaryOperation::Mul => Some(left * right),
                    ResolvedIntBinaryOperation::Div => exact_quotient(&left, &right).ok(),
                    ResolvedIntBinaryOperation::EuclideanDiv => {
                        Some(euclidean_div_rem(&left, &right).ok()?.0)
                    }
                    ResolvedIntBinaryOperation::EuclideanRemainder => {
                        Some(euclidean_div_rem(&left, &right).ok()?.1)
                    }
                    ResolvedIntBinaryOperation::RoundDiv => evaluate_round_div(&left, &right).ok(),
                }
            }
            ResolvedIntExprArenaNode::Log2Ceil(child) => {
                let value = values[*child as usize].clone()?;
                (value >= BigInt::one()).then(|| log2_ceil(&value))
            }
        };
    }
    values.get(arena.root as usize).cloned().flatten()
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum AffineSymbol {
    Parameter(String),
    Binder(BinderKey),
}

#[derive(Debug, Default)]
struct AffineForm {
    constant: BigInt,
    coefficients: BTreeMap<AffineSymbol, BigInt>,
}

impl AffineForm {
    fn add_symbol(&mut self, symbol: AffineSymbol, coefficient: BigInt) {
        use std::collections::btree_map::Entry;
        if coefficient.is_zero() {
            return;
        }
        match self.coefficients.entry(symbol) {
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
}

fn resolved_structurally_equal(left: &ResolvedIntExpr, right: &ResolvedIntExpr) -> bool {
    if let (Some(left), Some(right)) = (
        super::identity::resolved_expr_as_arena(left),
        super::identity::resolved_expr_as_arena(right),
    ) {
        return left == right;
    }
    let mut work = vec![(left, right)];
    while let Some((left, right)) = work.pop() {
        match (left, right) {
            (ResolvedIntExpr::Const(left), ResolvedIntExpr::Const(right)) if left == right => {}
            (ResolvedIntExpr::Parameter(left), ResolvedIntExpr::Parameter(right))
                if left == right => {}
            (ResolvedIntExpr::Binder(left), ResolvedIntExpr::Binder(right)) if left == right => {}
            (ResolvedIntExpr::Add(left_a, left_b), ResolvedIntExpr::Add(right_a, right_b)) |
            (ResolvedIntExpr::Sub(left_a, left_b), ResolvedIntExpr::Sub(right_a, right_b)) |
            (ResolvedIntExpr::Mul(left_a, left_b), ResolvedIntExpr::Mul(right_a, right_b)) |
            (ResolvedIntExpr::Div(left_a, left_b), ResolvedIntExpr::Div(right_a, right_b)) |
            (
                ResolvedIntExpr::RoundDiv(left_a, left_b),
                ResolvedIntExpr::RoundDiv(right_a, right_b),
            ) => {
                work.push((left_b, right_b));
                work.push((left_a, right_a));
            }
            (ResolvedIntExpr::Log2Ceil(left), ResolvedIntExpr::Log2Ceil(right)) => {
                work.push((left, right));
            }
            (
                ResolvedIntExpr::Source { source: left_source, coordinates: left_coordinates },
                ResolvedIntExpr::Source { source: right_source, coordinates: right_coordinates },
            ) if left_source == right_source &&
                left_coordinates.len() == right_coordinates.len() =>
            {
                work.extend(left_coordinates.iter().zip(right_coordinates.iter()));
            }
            (
                ResolvedIntExpr::ExtractMatrixCoefficient {
                    matrix: left_matrix,
                    position: left_position,
                },
                ResolvedIntExpr::ExtractMatrixCoefficient {
                    matrix: right_matrix,
                    position: right_position,
                },
            ) if left_matrix == right_matrix => work.push((left_position, right_position)),
            (ResolvedIntExpr::Arena(left), ResolvedIntExpr::Arena(right)) if left == right => {}
            _ => return false,
        }
    }
    true
}

fn resolved_affine_equal(left: &ResolvedIntExpr, right: &ResolvedIntExpr) -> bool {
    let mut form = AffineForm::default();
    let mut work = vec![(right, -BigInt::one()), (left, BigInt::one())];
    while let Some((expression, scale)) = work.pop() {
        if scale.is_zero() {
            continue;
        }
        match expression {
            ResolvedIntExpr::Const(value) => form.constant += scale * value,
            ResolvedIntExpr::Parameter(parameter) => {
                form.add_symbol(AffineSymbol::Parameter(parameter.clone()), scale);
            }
            ResolvedIntExpr::Binder(binder) => {
                form.add_symbol(AffineSymbol::Binder(binder.clone()), scale);
            }
            ResolvedIntExpr::Add(left, right) => {
                work.push((right, scale.clone()));
                work.push((left, scale));
            }
            ResolvedIntExpr::Sub(left, right) => {
                work.push((right, -scale.clone()));
                work.push((left, scale));
            }
            ResolvedIntExpr::Mul(left, right) => match (&**left, &**right) {
                (_, ResolvedIntExpr::Const(constant)) => work.push((left, scale * constant)),
                (ResolvedIntExpr::Const(constant), _) => work.push((right, scale * constant)),
                _ => return false,
            },
            ResolvedIntExpr::Div(_, _) |
            ResolvedIntExpr::EuclideanDiv(_, _) |
            ResolvedIntExpr::EuclideanRemainder(_, _) |
            ResolvedIntExpr::RoundDiv(_, _) |
            ResolvedIntExpr::Log2Ceil(_) |
            ResolvedIntExpr::Source { .. } |
            ResolvedIntExpr::ExtractMatrixCoefficient { .. } |
            ResolvedIntExpr::Arena(_) => return false,
        }
    }
    form.constant.is_zero() && form.coefficients.is_empty()
}

pub(crate) fn resolved_equal(left: &ResolvedIntExpr, right: &ResolvedIntExpr) -> bool {
    resolved_structurally_equal(left, right) ||
        resolved_constant(left).zip(resolved_constant(right)).is_some_and(|(l, r)| l == r) ||
        resolved_affine_equal(left, right)
}

pub(crate) fn matrix_types_equal(left: &ResolvedMatrixType, right: &ResolvedMatrixType) -> bool {
    resolved_equal(&left.modulus, &right.modulus) &&
        resolved_equal(&left.ring_dimension, &right.ring_dimension) &&
        resolved_equal(&left.rows, &right.rows) &&
        resolved_equal(&left.columns, &right.columns)
}

pub(crate) fn sorts_equal(left: &ScalarSort, right: &ScalarSort) -> bool {
    match (left, right) {
        (ScalarSort::Matrix(left), ScalarSort::Matrix(right)) => matrix_types_equal(left, right),
        (ScalarSort::Bytes(left), ScalarSort::Bytes(right)) => resolved_equal(left, right),
        _ => left == right,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarTransferError {
    Arity,
    Sort,
    Domain,
    Provenance,
    SelectorOnly,
    MissingSource,
    MissingBinder,
    MissingChild,
    NeedsMatrixContract,
    /// The same typed semantic identity was transferred with incompatible
    /// facts.  Facts are not an insertion-order merge authority.
    FactConflict,
    Unsupported,
}

fn require_int(value: &ScalarFacts) -> Result<&ScalarFacts, ScalarTransferError> {
    (value.sort == Ok(ScalarSort::Int)).then_some(value).ok_or(ScalarTransferError::Sort)
}

fn require_bool(value: &ScalarFacts) -> Result<&ScalarFacts, ScalarTransferError> {
    (value.sort == Ok(ScalarSort::Bool)).then_some(value).ok_or(ScalarTransferError::Sort)
}

fn require_real(value: &ScalarFacts) -> Result<&ScalarFacts, ScalarTransferError> {
    (value.sort == Ok(ScalarSort::Real)).then_some(value).ok_or(ScalarTransferError::Sort)
}

fn scalar_children(node: &ScalarNode) -> Box<[ScalarId]> {
    match node {
        ScalarNode::Source { indices, .. } | ScalarNode::Switch(indices) => indices.clone(),
        ScalarNode::IntAdd(children) |
        ScalarNode::IntSub(children) |
        ScalarNode::IntMul(children) |
        ScalarNode::IntExactDiv(children) |
        ScalarNode::IntEuclideanDiv(children) |
        ScalarNode::IntEuclideanRemainder(children) |
        ScalarNode::IntRoundDiv(children) |
        ScalarNode::IntEqual(children) |
        ScalarNode::IntLess(children) |
        ScalarNode::IntLessEqual(children) |
        ScalarNode::RealAdd(children) |
        ScalarNode::RealSub(children) |
        ScalarNode::RealMul(children) |
        ScalarNode::RealDiv(children) => children.to_vec().into_boxed_slice(),
        ScalarNode::IntLog2Ceil(children) |
        ScalarNode::BoolToInt(children) |
        ScalarNode::IntToReal(children) |
        ScalarNode::RealSqrt(children) => children.to_vec().into_boxed_slice(),
        ScalarNode::BitExtract { input, .. } => input.to_vec().into_boxed_slice(),
        ScalarNode::ExtractCoefficient { position, .. } => Box::new([*position]),
        ScalarNode::IntConst(_) |
        ScalarNode::IntParameter(_) |
        ScalarNode::IntBinder(_) |
        ScalarNode::BoolConst(_) |
        ScalarNode::RealConst(_) => Box::new([]),
    }
}

/// Scalar node operations.  All children are scalar IDs and are therefore
/// typed at construction; matrix values can enter only through the explicit
/// coefficient extraction boundary.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ScalarNode {
    Source {
        source: AtomicSourceId,
        indices: Box<[ScalarId]>,
    },
    IntConst(BigInt),
    IntParameter(String),
    IntBinder(BinderKey),
    IntAdd([ScalarId; 2]),
    IntSub([ScalarId; 2]),
    IntMul([ScalarId; 2]),
    IntExactDiv([ScalarId; 2]),
    IntEuclideanDiv([ScalarId; 2]),
    IntEuclideanRemainder([ScalarId; 2]),
    IntRoundDiv([ScalarId; 2]),
    IntLog2Ceil([ScalarId; 1]),
    BoolConst(bool),
    IntEqual([ScalarId; 2]),
    IntLess([ScalarId; 2]),
    IntLessEqual([ScalarId; 2]),
    BitExtract {
        bit: ResolvedIntExpr,
        input: [ScalarId; 1],
    },
    BoolToInt([ScalarId; 1]),
    RealConst(u64),
    IntToReal([ScalarId; 1]),
    RealAdd([ScalarId; 2]),
    RealSub([ScalarId; 2]),
    RealMul([ScalarId; 2]),
    RealDiv([ScalarId; 2]),
    RealSqrt([ScalarId; 1]),
    Switch(Box<[ScalarId]>),
    ExtractCoefficient {
        matrix: super::normal_form::ResolvedMatrixValueIdentity,
        position: ScalarId,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ScalarId(pub u32);

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ScalarIdentityNode {
    Source {
        source: super::identity::AtomicSourceKey,
        coordinates: Box<[ScalarIdentityId]>,
    },
    Const(BigInt),
    Bool(bool),
    Real(u64),
    Parameter(String),
    Binder(BinderKey),
    Unary {
        operation: ScalarOperation,
        input: ScalarIdentityId,
    },
    Binary {
        operation: ScalarOperation,
        left: ScalarIdentityId,
        right: ScalarIdentityId,
    },
    Switch {
        selector: ScalarIdentityId,
        cases: Box<[ScalarIdentityId]>,
    },
    BitExtract {
        bit: ResolvedIntExpr,
        input: ScalarIdentityId,
    },
    ExtractCoefficient {
        matrix: super::normal_form::ResolvedMatrixValueIdentity,
        position: ScalarIdentityId,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ScalarIdentityId(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ScalarOperation {
    Add,
    Sub,
    Mul,
    ExactDiv,
    EuclideanDiv,
    EuclideanRemainder,
    RoundDiv,
    Log2Ceil,
    Equal,
    Less,
    LessEqual,
    BitExtract,
    BoolToInt,
    IntToReal,
    RealAdd,
    RealSub,
    RealMul,
    RealDiv,
    RealSqrt,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScalarEntry {
    pub node: ScalarNode,
    pub identity: ScalarIdentityId,
    pub analysis: ScalarFacts,
}

#[derive(Clone, Debug, Default)]
pub struct ScalarStore {
    entries: Vec<ScalarEntry>,
    by_identity: BTreeMap<ScalarIdentityId, ScalarId>,
    identities: Vec<ScalarIdentityNode>,
    identity_index: BTreeMap<ScalarIdentityNode, ScalarIdentityId>,
}

impl ScalarStore {
    /// different strengths, so the proof upper is joined conservatively by
    /// taking the minimum present upper.  Sort, provenance, and residue
    /// convention remain invariant and cannot be joined away.
    fn merge_repeated_facts(
        existing: &mut ScalarFacts,
        incoming: ScalarFacts,
        allow_direct_extract_join: bool,
    ) -> Result<(), ScalarTransferError> {
        if !allow_direct_extract_join {
            return (*existing == incoming).then_some(()).ok_or(ScalarTransferError::FactConflict);
        }
        if existing.sort != incoming.sort ||
            existing.scalar_provenance != incoming.scalar_provenance ||
            existing.canonical_residue_convention != incoming.canonical_residue_convention
        {
            return Err(ScalarTransferError::FactConflict);
        }
        existing.merge_direct_extract(incoming);
        Ok(())
    }

    /// Inserts one typed scalar node using the transfer table and a shallow
    /// semantic key derived from its typed children.
    pub fn intern_node(
        &mut self,
        node: ScalarNode,
        symbols: &SymbolTables,
    ) -> Result<ScalarId, ScalarTransferError> {
        let child_ids = scalar_children(&node);
        let children = child_ids
            .iter()
            .map(|id| self.facts(*id).ok_or(ScalarTransferError::MissingChild))
            .collect::<Result<Vec<_>, _>>()?;
        let facts = ScalarFacts::transfer(&node, &children, symbols)?;
        let semantic = self.semantic_node(&node, symbols)?;
        let identity = self.intern_identity(semantic);
        if let Some(id) = self.by_identity.get(&identity).copied() {
            let merge = Self::merge_repeated_facts(
                &mut self.entries[id.0 as usize].analysis,
                facts,
                matches!(node, ScalarNode::ExtractCoefficient { .. }),
            );
            merge?;
            return Ok(id);
        }
        let id = ScalarId(self.entries.len() as u32);
        self.by_identity.insert(identity, id);
        self.entries.push(ScalarEntry { node, identity, analysis: facts });
        Ok(id)
    }

    fn semantic_node(
        &self,
        node: &ScalarNode,
        symbols: &SymbolTables,
    ) -> Result<ScalarIdentityNode, ScalarTransferError> {
        let identity = |id: ScalarId| {
            self.get(id).map(|entry| entry.identity).ok_or(ScalarTransferError::MissingChild)
        };
        let binary = |ids: &[ScalarId; 2], operation| {
            Ok(ScalarIdentityNode::Binary {
                operation,
                left: identity(ids[0])?,
                right: identity(ids[1])?,
            })
        };
        let unary = |ids: &[ScalarId; 1], operation| {
            Ok(ScalarIdentityNode::Unary { operation, input: identity(ids[0])? })
        };
        Ok(match node {
            ScalarNode::Source { source, indices } => ScalarIdentityNode::Source {
                source: symbols
                    .atomic_sources
                    .get(source.0)
                    .ok_or(ScalarTransferError::MissingSource)?
                    .key
                    .clone(),
                coordinates: indices
                    .iter()
                    .map(|id| identity(*id))
                    .collect::<Result<Box<_>, _>>()?,
            },
            ScalarNode::IntConst(value) => ScalarIdentityNode::Const(value.clone()),
            ScalarNode::IntParameter(value) => ScalarIdentityNode::Parameter(value.clone()),
            ScalarNode::IntBinder(value) => ScalarIdentityNode::Binder(value.clone()),
            ScalarNode::IntAdd(ids) => binary(ids, ScalarOperation::Add)?,
            ScalarNode::IntSub(ids) => binary(ids, ScalarOperation::Sub)?,
            ScalarNode::IntMul(ids) => binary(ids, ScalarOperation::Mul)?,
            ScalarNode::IntExactDiv(ids) => binary(ids, ScalarOperation::ExactDiv)?,
            ScalarNode::IntEuclideanDiv(ids) => binary(ids, ScalarOperation::EuclideanDiv)?,
            ScalarNode::IntEuclideanRemainder(ids) => {
                binary(ids, ScalarOperation::EuclideanRemainder)?
            }
            ScalarNode::IntRoundDiv(ids) => binary(ids, ScalarOperation::RoundDiv)?,
            ScalarNode::IntLog2Ceil(ids) => unary(ids, ScalarOperation::Log2Ceil)?,
            ScalarNode::BoolConst(value) => ScalarIdentityNode::Bool(*value),
            ScalarNode::IntEqual(ids) => binary(ids, ScalarOperation::Equal)?,
            ScalarNode::IntLess(ids) => binary(ids, ScalarOperation::Less)?,
            ScalarNode::IntLessEqual(ids) => binary(ids, ScalarOperation::LessEqual)?,
            ScalarNode::BitExtract { bit, input } => {
                ScalarIdentityNode::BitExtract { bit: bit.clone(), input: identity(input[0])? }
            }
            ScalarNode::BoolToInt(ids) => unary(ids, ScalarOperation::BoolToInt)?,
            ScalarNode::RealConst(bits) => ScalarIdentityNode::Real(*bits),
            ScalarNode::IntToReal(ids) => unary(ids, ScalarOperation::IntToReal)?,
            ScalarNode::RealAdd(ids) => binary(ids, ScalarOperation::RealAdd)?,
            ScalarNode::RealSub(ids) => binary(ids, ScalarOperation::RealSub)?,
            ScalarNode::RealMul(ids) => binary(ids, ScalarOperation::RealMul)?,
            ScalarNode::RealDiv(ids) => binary(ids, ScalarOperation::RealDiv)?,
            ScalarNode::RealSqrt(ids) => unary(ids, ScalarOperation::RealSqrt)?,
            ScalarNode::Switch(ids) => ScalarIdentityNode::Switch {
                selector: identity(*ids.first().ok_or(ScalarTransferError::Arity)?)?,
                cases: ids[1..].iter().map(|id| identity(*id)).collect::<Result<Box<_>, _>>()?,
            },
            ScalarNode::ExtractCoefficient { matrix, position } => {
                ScalarIdentityNode::ExtractCoefficient {
                    matrix: matrix.clone(),
                    position: identity(*position)?,
                }
            }
        })
    }

    fn intern_identity(&mut self, node: ScalarIdentityNode) -> ScalarIdentityId {
        if let Some(id) = self.identity_index.get(&node).copied() {
            return id;
        }
        let id = ScalarIdentityId(self.identities.len() as u32);
        self.identity_index.insert(node.clone(), id);
        self.identities.push(node);
        id
    }

    pub fn get(&self, id: ScalarId) -> Option<&ScalarEntry> {
        self.entries.get(id.0 as usize)
    }

    pub fn node(&self, id: ScalarId) -> Option<&ScalarNode> {
        self.get(id).map(|entry| &entry.node)
    }

    /// Returns the ordered scalar children of an entry for iterative graph
    /// traversals owned by the lowerer.  The returned slice is detached from
    /// the arena so callers can mutate the store while traversing.
    pub fn children(&self, id: ScalarId) -> Option<Box<[ScalarId]>> {
        self.node(id).map(scalar_children)
    }

    pub fn facts(&self, id: ScalarId) -> Option<&ScalarFacts> {
        self.get(id).map(|entry| &entry.analysis)
    }

    /// Returns the canonical scalar identity as a descriptor-local arena.
    /// The descriptor contains no scalar-store IDs and is safe to drop at
    /// arbitrary expression depth.
    pub fn identity(&self, id: ScalarId) -> Option<ResolvedIntExpr> {
        let root = self.get(id)?.identity;
        let mut order = Vec::new();
        let mut positions = HashMap::new();
        let mut work = vec![(root, false)];
        while let Some((current, expanded)) = work.pop() {
            if positions.contains_key(&current) {
                continue;
            }
            let node = self.identity_node(current)?;
            if !expanded {
                work.push((current, true));
                match node {
                    ScalarIdentityNode::Source { coordinates, .. } => {
                        work.extend(coordinates.iter().rev().map(|id| (*id, false)))
                    }
                    ScalarIdentityNode::Unary { input, .. } => work.push((*input, false)),
                    ScalarIdentityNode::Binary { left, right, .. } => {
                        work.push((*right, false));
                        work.push((*left, false));
                    }
                    ScalarIdentityNode::Switch { selector, cases } => {
                        work.extend(cases.iter().rev().map(|id| (*id, false)));
                        work.push((*selector, false));
                    }
                    ScalarIdentityNode::BitExtract { input, .. } |
                    ScalarIdentityNode::ExtractCoefficient { position: input, .. } => {
                        work.push((*input, false));
                    }
                    ScalarIdentityNode::Const(_) |
                    ScalarIdentityNode::Bool(_) |
                    ScalarIdentityNode::Real(_) |
                    ScalarIdentityNode::Parameter(_) |
                    ScalarIdentityNode::Binder(_) => {}
                }
            } else {
                positions.insert(current, order.len());
                order.push(current);
            }
        }
        self.identity_arena(root, &order, &positions)
    }

    fn identity_arena(
        &self,
        root: ScalarIdentityId,
        order: &[ScalarIdentityId],
        positions: &HashMap<ScalarIdentityId, usize>,
    ) -> Option<ResolvedIntExpr> {
        let child = |id: ScalarIdentityId| u32::try_from(*positions.get(&id)?).ok();
        let nodes = order
            .iter()
            .map(|id| match self.identity_node(*id)? {
                ScalarIdentityNode::Const(value) => {
                    Some(ResolvedIntExprArenaNode::Const(value.clone()))
                }
                ScalarIdentityNode::Parameter(value) => {
                    Some(ResolvedIntExprArenaNode::Parameter(value.clone()))
                }
                ScalarIdentityNode::Binder(value) => {
                    Some(ResolvedIntExprArenaNode::Binder(value.clone()))
                }
                ScalarIdentityNode::Source { source, coordinates } => {
                    Some(ResolvedIntExprArenaNode::Source {
                        source: source.clone(),
                        coordinates: coordinates
                            .iter()
                            .map(|id| child(*id))
                            .collect::<Option<Box<_>>>()?,
                    })
                }
                ScalarIdentityNode::Binary { operation, left, right } => {
                    let operation = match operation {
                        ScalarOperation::Add => ResolvedIntBinaryOperation::Add,
                        ScalarOperation::Sub => ResolvedIntBinaryOperation::Sub,
                        ScalarOperation::Mul => ResolvedIntBinaryOperation::Mul,
                        ScalarOperation::ExactDiv => ResolvedIntBinaryOperation::Div,
                        ScalarOperation::EuclideanDiv => ResolvedIntBinaryOperation::EuclideanDiv,
                        ScalarOperation::EuclideanRemainder => {
                            ResolvedIntBinaryOperation::EuclideanRemainder
                        }
                        ScalarOperation::RoundDiv => ResolvedIntBinaryOperation::RoundDiv,
                        _ => return None,
                    };
                    Some(ResolvedIntExprArenaNode::Binary {
                        operation,
                        children: [child(*left)?, child(*right)?],
                    })
                }
                ScalarIdentityNode::Unary { operation, input } => {
                    if *operation != ScalarOperation::Log2Ceil {
                        return None;
                    }
                    Some(ResolvedIntExprArenaNode::Log2Ceil(child(*input)?))
                }
                ScalarIdentityNode::ExtractCoefficient { matrix, position } => {
                    Some(ResolvedIntExprArenaNode::ExtractMatrixCoefficient {
                        matrix: Box::new(matrix.clone()),
                        position: child(*position)?,
                    })
                }
                ScalarIdentityNode::BitExtract { .. } |
                ScalarIdentityNode::Switch { .. } |
                ScalarIdentityNode::Bool(_) |
                ScalarIdentityNode::Real(_) => return None,
            })
            .collect::<Option<Box<_>>>()?;
        Some(ResolvedIntExpr::Arena(Box::new(ResolvedIntExprArena {
            nodes,
            root: u32::try_from(*positions.get(&root)?).ok()?,
        })))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn identity_len(&self) -> usize {
        self.identities.len()
    }

    pub fn identity_node(&self, id: ScalarIdentityId) -> Option<&ScalarIdentityNode> {
        self.identities.get(id.0 as usize)
    }

    /// Inserts a coefficient extraction at the matrix/scalar boundary. The
    /// position is an existing typed scalar entry; its canonical identity is
    /// read from this arena, so the node and its semantic key cannot diverge.
    pub fn intern_extract(
        &mut self,
        matrix: super::normal_form::ResolvedMatrixValueIdentity,
        position: ScalarId,
        facts: ScalarFacts,
    ) -> Result<ScalarId, ScalarTransferError> {
        let position_identity = self
            .get(position)
            .map(|entry| entry.identity)
            .ok_or(ScalarTransferError::MissingChild)?;
        // The external proof upper is deliberately excluded from this key:
        // the same extraction with two proof strengths is one semantic scalar,
        // and the facts merge conservatively below.
        let extraction_identity = self.intern_identity(ScalarIdentityNode::ExtractCoefficient {
            matrix: matrix.clone(),
            position: position_identity,
        });
        if let Some(id) = self.by_identity.get(&extraction_identity).copied() {
            let merge =
                Self::merge_repeated_facts(&mut self.entries[id.0 as usize].analysis, facts, true);
            merge?;
            return Ok(id);
        }
        let id = ScalarId(self.entries.len() as u32);
        self.by_identity.insert(extraction_identity, id);
        self.entries.push(ScalarEntry {
            node: ScalarNode::ExtractCoefficient { matrix, position },
            identity: extraction_identity,
            analysis: facts,
        });
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ProtocolInputId, operational_noise::identity::AtomicSourceDescriptor};
    use std::thread;

    fn integer(value: i64) -> ScalarFacts {
        ScalarFacts::scalar(
            ScalarSort::Int,
            Some(IntegerDomain::exact(value)),
            ScalarProvenance::Ordinary,
        )
    }

    fn interval(minimum: i64, maximum: i64) -> ScalarFacts {
        ScalarFacts::scalar(
            ScalarSort::Int,
            Some(IntegerDomain::IntervalOnly(IntegerInterval {
                minimum: minimum.into(),
                maximum: maximum.into(),
            })),
            ScalarProvenance::Ordinary,
        )
    }

    #[test]
    fn transfer_covers_integer_domain_operations_and_selector_consumers() {
        let symbols = SymbolTables::default();
        let left = interval(-3, 8);
        let right = integer(2);
        let binary = [
            (ScalarNode::IntAdd([ScalarId(0), ScalarId(1)]), ScalarOperation::Add),
            (ScalarNode::IntSub([ScalarId(0), ScalarId(1)]), ScalarOperation::Sub),
            (ScalarNode::IntMul([ScalarId(0), ScalarId(1)]), ScalarOperation::Mul),
            (
                ScalarNode::IntEuclideanDiv([ScalarId(0), ScalarId(1)]),
                ScalarOperation::EuclideanDiv,
            ),
            (
                ScalarNode::IntEuclideanRemainder([ScalarId(0), ScalarId(1)]),
                ScalarOperation::EuclideanRemainder,
            ),
            (ScalarNode::IntRoundDiv([ScalarId(0), ScalarId(1)]), ScalarOperation::RoundDiv),
        ];
        for (node, _) in binary {
            let result = ScalarFacts::transfer(&node, &[&left, &right], &symbols).unwrap();
            assert_eq!(result.sort, Ok(ScalarSort::Int));
            assert_eq!(result.scalar_provenance, Some(ScalarProvenance::Ordinary));
        }
        let exact = integer(6);
        let divisor = integer(3);
        let result = ScalarFacts::transfer(
            &ScalarNode::IntExactDiv([ScalarId(0), ScalarId(1)]),
            &[&exact, &divisor],
            &symbols,
        )
        .unwrap();
        assert_eq!(
            result.integer_domain.and_then(|domain| domain.exact_value().cloned()),
            Some(2.into())
        );

        let positive = integer(7);
        let log =
            ScalarFacts::transfer(&ScalarNode::IntLog2Ceil([ScalarId(0)]), &[&positive], &symbols)
                .unwrap();
        assert_eq!(
            log.integer_domain.and_then(|domain| domain.exact_value().cloned()),
            Some(3.into())
        );

        let selector = ScalarFacts::scalar(
            ScalarSort::Int,
            Some(IntegerDomain::IntervalOnly(IntegerInterval {
                minimum: 0.into(),
                maximum: 7.into(),
            })),
            ScalarProvenance::SelectorOnly,
        );
        assert_eq!(
            ScalarFacts::transfer(
                &ScalarNode::IntAdd([ScalarId(0), ScalarId(1)]),
                &[&selector, &right],
                &symbols,
            ),
            Err(ScalarTransferError::SelectorOnly)
        );
        let bit = ScalarFacts::transfer(
            &ScalarNode::BitExtract { bit: ResolvedIntExpr::Const(0.into()), input: [ScalarId(0)] },
            &[&selector],
            &symbols,
        )
        .unwrap();
        assert_eq!(bit.sort, Ok(ScalarSort::Bool));
        assert_eq!(
            ScalarFacts::transfer(&ScalarNode::BoolToInt([ScalarId(0)]), &[&bit], &symbols,)
                .unwrap()
                .sort,
            Ok(ScalarSort::Int)
        );
    }

    #[test]
    fn euclidean_transfer_preserves_selector_provenance() {
        let symbols = SymbolTables::default();
        let selector = ScalarFacts::scalar(
            ScalarSort::Int,
            Some(IntegerDomain::IntervalOnly(IntegerInterval {
                minimum: 0.into(),
                maximum: 7.into(),
            })),
            ScalarProvenance::SelectorOnly,
        );
        let divisor = integer(2);
        for node in [
            ScalarNode::IntEuclideanDiv([ScalarId(0), ScalarId(1)]),
            ScalarNode::IntEuclideanRemainder([ScalarId(0), ScalarId(1)]),
        ] {
            let result = ScalarFacts::transfer(&node, &[&selector, &divisor], &symbols).unwrap();
            assert_eq!(result.scalar_provenance, Some(ScalarProvenance::SelectorOnly));
        }
    }

    #[test]
    fn transfer_covers_boolean_comparison_real_and_switch_operations() {
        let symbols = SymbolTables::default();
        let left = integer(2);
        let right = integer(3);
        for node in [
            ScalarNode::IntEqual([ScalarId(0), ScalarId(1)]),
            ScalarNode::IntLess([ScalarId(0), ScalarId(1)]),
            ScalarNode::IntLessEqual([ScalarId(0), ScalarId(1)]),
        ] {
            let result = ScalarFacts::transfer(&node, &[&left, &right], &symbols).unwrap();
            assert_eq!(result.sort, Ok(ScalarSort::Bool));
            assert!(result.possible_false || result.possible_true);
        }
        let first =
            ScalarFacts::transfer(&ScalarNode::RealConst(1.0f64.to_bits()), &[], &symbols).unwrap();
        let second =
            ScalarFacts::transfer(&ScalarNode::RealConst(2.0f64.to_bits()), &[], &symbols).unwrap();
        let sum = ScalarFacts::transfer(
            &ScalarNode::RealAdd([ScalarId(0), ScalarId(1)]),
            &[&first, &second],
            &symbols,
        )
        .unwrap();
        assert_eq!(sum.real_constant_bits, Some(3.0f64.to_bits()));
        let false_case = ScalarFacts::boolean(true, false, ScalarProvenance::Ordinary);
        let true_case = ScalarFacts::boolean(false, true, ScalarProvenance::Ordinary);
        let selector = integer(0);
        let switched = ScalarFacts::transfer(
            &ScalarNode::Switch(vec![ScalarId(0), ScalarId(1), ScalarId(2)].into_boxed_slice()),
            &[&selector, &false_case, &true_case],
            &symbols,
        )
        .unwrap();
        assert_eq!(switched.sort, Ok(ScalarSort::Bool));
        assert!(switched.possible_false && switched.possible_true);
    }

    #[test]
    fn switch_rejects_negative_and_out_of_range_selector_intervals() {
        let symbols = SymbolTables::default();
        let first = ScalarFacts::boolean(true, false, ScalarProvenance::Ordinary);
        let second = ScalarFacts::boolean(false, true, ScalarProvenance::Ordinary);
        let switch =
            || ScalarNode::Switch(vec![ScalarId(0), ScalarId(1), ScalarId(2)].into_boxed_slice());
        let negative = interval(-1, 0);
        assert_eq!(
            ScalarFacts::transfer(&switch(), &[&negative, &first, &second], &symbols),
            Err(ScalarTransferError::Domain)
        );
        let out_of_range = interval(0, 2);
        assert_eq!(
            ScalarFacts::transfer(&switch(), &[&out_of_range, &first, &second], &symbols),
            Err(ScalarTransferError::Domain)
        );
    }

    #[test]
    fn source_and_binder_transfers_use_owner_resolved_symbol_contracts() {
        let mut symbols = SymbolTables::default();
        let source_id = symbols.atomic_sources.intern(AtomicSourceDescriptor {
            key: super::super::identity::AtomicSourceKey::ProtocolInput(ProtocolInputId::from("x")),
            sort: ScalarSort::Int,
            integer_domain: Some(super::super::identity::IntegerSourceDomain {
                minimum: 0.into(),
                maximum: 7.into(),
            }),
            canonical_residue_convention: None,
            relation_role: None,
        });
        let source = ScalarFacts::transfer(
            &ScalarNode::Source { source: AtomicSourceId(source_id), indices: Box::new([]) },
            &[],
            &symbols,
        )
        .unwrap();
        assert_eq!(
            source.integer_domain.and_then(|domain| domain.interval().ok()).unwrap().maximum,
            7.into()
        );
    }

    #[test]
    fn source_transfer_is_sort_specific_and_matrix_is_boundary_only() {
        let mut symbols = SymbolTables::default();
        let (bytes_id, blob_id, matrix_id) = {
            let mut source = |name: &str, sort: ScalarSort| {
                symbols.atomic_sources.intern(AtomicSourceDescriptor {
                    key: super::super::identity::AtomicSourceKey::ProtocolInput(
                        ProtocolInputId::from(name),
                    ),
                    sort,
                    integer_domain: None,
                    canonical_residue_convention: None,
                    relation_role: None,
                })
            };
            let bytes_id = source("bytes", ScalarSort::Bytes(ResolvedIntExpr::Const(4.into())));
            let blob_id = source(
                "blob",
                ScalarSort::TypedBlob { type_name: "TestBlob".to_owned(), schema_hash: [7; 32] },
            );
            let matrix = ResolvedMatrixType {
                modulus: ResolvedIntExpr::Const(17.into()),
                ring_dimension: ResolvedIntExpr::Const(1.into()),
                rows: ResolvedIntExpr::Const(1.into()),
                columns: ResolvedIntExpr::Const(1.into()),
            };
            let matrix_id = source("matrix", ScalarSort::Matrix(matrix));
            (bytes_id, blob_id, matrix_id)
        };
        for (source_id, expected) in [
            (bytes_id, ScalarSort::Bytes(ResolvedIntExpr::Const(4.into()))),
            (
                blob_id,
                ScalarSort::TypedBlob { type_name: "TestBlob".to_owned(), schema_hash: [7; 32] },
            ),
        ] {
            let facts = ScalarFacts::transfer(
                &ScalarNode::Source { source: AtomicSourceId(source_id), indices: Box::new([]) },
                &[],
                &symbols,
            )
            .unwrap();
            assert_eq!(facts.sort, Ok(expected));
            assert_eq!(facts.integer_domain, None);
            assert_eq!(facts.scalar_provenance, None);
        }
        assert_eq!(
            ScalarFacts::transfer(
                &ScalarNode::Source { source: AtomicSourceId(matrix_id), indices: Box::new([]) },
                &[],
                &symbols,
            ),
            Err(ScalarTransferError::NeedsMatrixContract)
        );
    }

    #[test]
    fn repeated_source_identity_rejects_conflicting_descriptor_facts_in_either_order() {
        let source_key = super::super::identity::AtomicSourceKey::ProtocolInput(
            ProtocolInputId::from("same-source"),
        );
        for (first_maximum, second_maximum) in [(7_i64, 9_i64), (9_i64, 7_i64)] {
            let mut symbols = SymbolTables::default();
            let first = symbols.atomic_sources.intern(AtomicSourceDescriptor {
                key: source_key.clone(),
                sort: ScalarSort::Int,
                integer_domain: Some(super::super::identity::IntegerSourceDomain {
                    minimum: 0.into(),
                    maximum: first_maximum.into(),
                }),
                canonical_residue_convention: None,
                relation_role: None,
            });
            let second = symbols.atomic_sources.intern(AtomicSourceDescriptor {
                key: source_key.clone(),
                sort: ScalarSort::Int,
                integer_domain: Some(super::super::identity::IntegerSourceDomain {
                    minimum: 0.into(),
                    maximum: second_maximum.into(),
                }),
                canonical_residue_convention: None,
                relation_role: None,
            });
            let mut store = ScalarStore::default();
            store
                .intern_node(
                    ScalarNode::Source { source: AtomicSourceId(first), indices: Box::new([]) },
                    &symbols,
                )
                .unwrap();
            assert_eq!(
                store.intern_node(
                    ScalarNode::Source { source: AtomicSourceId(second), indices: Box::new([]) },
                    &symbols,
                ),
                Err(ScalarTransferError::FactConflict)
            );
        }
    }

    #[test]
    fn semantic_identity_is_shallow_and_direct_extract_proofs_merge() {
        let mut store = ScalarStore::default();
        let symbols = SymbolTables::default();
        let id = store.intern_node(ScalarNode::IntConst(1.into()), &symbols).unwrap();
        assert!(matches!(store.identity(id), Some(ResolvedIntExpr::Arena(_))));
        assert_eq!(store.len(), 1);

        let position = store.intern_node(ScalarNode::IntConst(2.into()), &symbols).unwrap();
        let matrix = super::super::normal_form::ResolvedMatrixValueIdentity {
            nodes: vec![super::super::normal_form::ResolvedMatrixValueIdentityNode {
                operation: super::super::normal_form::MatrixValueOperation::Atom,
                children: Box::new([]),
                owner: None,
                selector: None,
            }]
            .into_boxed_slice(),
            root: 0,
        };
        let five = BigUint::from(5_u8);
        let matrix_type = super::super::identity::ResolvedMatrixType {
            modulus: ResolvedIntExpr::Const(17.into()),
            ring_dimension: ResolvedIntExpr::Const(1.into()),
            rows: ResolvedIntExpr::Const(1.into()),
            columns: ResolvedIntExpr::Const(1.into()),
        };
        let mut first =
            super::direct_extract_facts(matrix_type.clone(), BigUint::from(17_u8), None).unwrap();
        first.direct_extract = None;
        let first_id = store.intern_extract(matrix.clone(), position, first).unwrap();
        let second =
            super::direct_extract_facts(matrix_type, BigUint::from(17_u8), Some(&five)).unwrap();
        let second_id = store.intern_extract(matrix, position, second).unwrap();
        assert_eq!(first_id, second_id);
        assert_eq!(
            store.get(first_id).unwrap().analysis.direct_extract.as_ref().unwrap().canonical_upper,
            Some(five)
        );
        assert_eq!(store.get(first_id).unwrap().analysis.canonical_residue_convention, None);

        let different_matrix = super::super::normal_form::ResolvedMatrixValueIdentity {
            nodes: vec![super::super::normal_form::ResolvedMatrixValueIdentityNode {
                operation: super::super::normal_form::MatrixValueOperation::Negate,
                children: Box::new([]),
                owner: None,
                selector: None,
            }]
            .into_boxed_slice(),
            root: 0,
        };
        let distinct = super::direct_extract_facts(
            super::super::identity::ResolvedMatrixType {
                modulus: ResolvedIntExpr::Const(17.into()),
                ring_dimension: ResolvedIntExpr::Const(1.into()),
                rows: ResolvedIntExpr::Const(1.into()),
                columns: ResolvedIntExpr::Const(1.into()),
            },
            BigUint::from(17_u8),
            None,
        )
        .unwrap();
        let distinct_id = store.intern_extract(different_matrix, position, distinct).unwrap();
        assert_ne!(first_id, distinct_id);
    }

    #[test]
    fn deep_diamond_identity_interning_uses_controlled_stack() {
        let worker = thread::Builder::new()
            .name("scalar-identity-depth-test".to_owned())
            .stack_size(128 * 1024)
            .spawn(|| {
                let mut store = ScalarStore::default();
                let symbols = SymbolTables::default();
                let one = store.intern_node(ScalarNode::IntConst(1.into()), &symbols).unwrap();
                let mut branch = one;
                for _ in 0..4096 {
                    branch =
                        store.intern_node(ScalarNode::IntAdd([branch, one]), &symbols).unwrap();
                }
                let first =
                    store.intern_node(ScalarNode::IntAdd([branch, branch]), &symbols).unwrap();
                let second =
                    store.intern_node(ScalarNode::IntAdd([branch, branch]), &symbols).unwrap();
                assert_eq!(first, second);
                assert_eq!(store.len(), 4098);
                assert!(store.identity_node(store.get(first).unwrap().identity).is_some());
                let identity = store.identity(first).expect("deep identity descriptor");
                assert!(matches!(identity, ResolvedIntExpr::Arena(_)));
                drop(identity);
            })
            .expect("spawn scalar identity depth test");
        worker.join().expect("scalar identity depth test panicked");
    }

    #[test]
    fn deep_identity_arena_substitutes_owned_binder_without_expansion() {
        let worker = thread::Builder::new()
            .name("scalar-identity-substitution-depth-test".to_owned())
            .stack_size(128 * 1024)
            .spawn(|| {
                use super::super::identity::{
                    BinderKey, OccurrenceScope, ProgramKey, ResolvedIntBinaryOperation,
                    ResolvedIntExprArena, ResolvedIntExprArenaNode,
                };
                let binder = BinderKey {
                    loop_scope: OccurrenceScope {
                        program: ProgramKey::Ideal,
                        definition: mxx_ir_core::FrozenGraphScopeId::Root,
                        path: Box::new([]),
                    },
                    loop_node: mxx_ir_core::NodeId(0),
                    slot: 0,
                };
                let mut nodes = vec![ResolvedIntExprArenaNode::Binder(binder.clone())];
                let mut current = 0_u32;
                for _ in 0..4096 {
                    let next = nodes.len() as u32;
                    nodes.push(ResolvedIntExprArenaNode::Binary {
                        operation: ResolvedIntBinaryOperation::Add,
                        children: [current, current],
                    });
                    current = next;
                }
                let expression = ResolvedIntExpr::Arena(Box::new(ResolvedIntExprArena {
                    nodes: nodes.into_boxed_slice(),
                    root: current,
                }));
                let substituted = super::super::identity::substitute_resolved_int_expr(
                    &expression,
                    &binder,
                    &ResolvedIntExpr::Const(7.into()),
                );
                assert!(matches!(substituted, ResolvedIntExpr::Arena(_)));
                drop(substituted);
                drop(expression);
            })
            .expect("spawn scalar identity substitution depth test");
        worker.join().expect("scalar identity substitution depth test panicked");
    }

    #[test]
    fn arena_substitution_splices_compound_replacement_and_preserves_context() {
        use super::super::identity::{
            BinderKey, OccurrenceScope, ProgramKey, ResolvedIntBinaryOperation,
            ResolvedIntExprArena, ResolvedIntExprArenaNode,
        };
        let binder = BinderKey {
            loop_scope: OccurrenceScope {
                program: ProgramKey::Ideal,
                definition: mxx_ir_core::FrozenGraphScopeId::Root,
                path: Box::new([]),
            },
            loop_node: mxx_ir_core::NodeId(0),
            slot: 0,
        };
        let expression = ResolvedIntExpr::Arena(Box::new(ResolvedIntExprArena {
            nodes: vec![
                ResolvedIntExprArenaNode::Binder(binder.clone()),
                ResolvedIntExprArenaNode::Const(1.into()),
                ResolvedIntExprArenaNode::Binary {
                    operation: ResolvedIntBinaryOperation::Add,
                    children: [0, 1],
                },
            ]
            .into_boxed_slice(),
            root: 2,
        }));
        let replacement = ResolvedIntExpr::Add(
            Box::new(ResolvedIntExpr::Const(2.into())),
            Box::new(ResolvedIntExpr::Const(3.into())),
        );
        let substituted = super::super::identity::substitute_resolved_int_expr(
            &expression,
            &binder,
            &replacement,
        );
        let expected = ResolvedIntExpr::Arena(Box::new(ResolvedIntExprArena {
            nodes: vec![
                ResolvedIntExprArenaNode::Const(2.into()),
                ResolvedIntExprArenaNode::Const(3.into()),
                ResolvedIntExprArenaNode::Binary {
                    operation: ResolvedIntBinaryOperation::Add,
                    children: [0, 1],
                },
                ResolvedIntExprArenaNode::Const(1.into()),
                ResolvedIntExprArenaNode::Binary {
                    operation: ResolvedIntBinaryOperation::Add,
                    children: [2, 3],
                },
            ]
            .into_boxed_slice(),
            root: 4,
        }));
        assert_eq!(substituted, expected);
        assert_eq!(
            super::super::identity::resolved_expr_as_arena(&substituted),
            super::super::identity::resolved_expr_as_arena(&expected)
        );
    }

    #[test]
    fn arena_substitution_handles_extraction_positions_and_canonical_sharing() {
        use super::super::identity::{
            BinderKey, OccurrenceScope, ProgramKey, ResolvedIntBinaryOperation,
            ResolvedIntExprArena, ResolvedIntExprArenaNode,
        };
        let binder = BinderKey {
            loop_scope: OccurrenceScope {
                program: ProgramKey::Ideal,
                definition: mxx_ir_core::FrozenGraphScopeId::Root,
                path: Box::new([]),
            },
            loop_node: mxx_ir_core::NodeId(1),
            slot: 0,
        };
        let matrix = super::super::normal_form::ResolvedMatrixValueIdentity {
            nodes: vec![super::super::normal_form::ResolvedMatrixValueIdentityNode {
                operation: super::super::normal_form::MatrixValueOperation::Atom,
                children: Box::new([]),
                owner: None,
                selector: None,
            }]
            .into_boxed_slice(),
            root: 0,
        };
        let extracted = ResolvedIntExpr::Arena(Box::new(ResolvedIntExprArena {
            nodes: vec![
                ResolvedIntExprArenaNode::Binder(binder.clone()),
                ResolvedIntExprArenaNode::ExtractMatrixCoefficient {
                    matrix: Box::new(matrix),
                    position: 0,
                },
            ]
            .into_boxed_slice(),
            root: 1,
        }));
        let substituted = super::super::identity::substitute_resolved_int_expr(
            &extracted,
            &binder,
            &ResolvedIntExpr::Const(7.into()),
        );
        let ResolvedIntExpr::Arena(substituted) = substituted else {
            panic!("extraction substitution must remain an arena")
        };
        let ResolvedIntExprArenaNode::ExtractMatrixCoefficient { position, .. } =
            &substituted.nodes[substituted.root as usize]
        else {
            panic!("extraction root was not preserved")
        };
        assert!(matches!(
            &substituted.nodes[*position as usize],
            ResolvedIntExprArenaNode::Const(value) if value == &7.into()
        ));

        let binder_plus_one = ResolvedIntExpr::Arena(Box::new(ResolvedIntExprArena {
            nodes: vec![
                ResolvedIntExprArenaNode::Binder(binder.clone()),
                ResolvedIntExprArenaNode::Const(1.into()),
                ResolvedIntExprArenaNode::Binary {
                    operation: ResolvedIntBinaryOperation::Add,
                    children: [0, 1],
                },
            ]
            .into_boxed_slice(),
            root: 2,
        }));
        let one_plus_one = ResolvedIntExpr::Arena(Box::new(ResolvedIntExprArena {
            nodes: vec![
                ResolvedIntExprArenaNode::Const(1.into()),
                ResolvedIntExprArenaNode::Binary {
                    operation: ResolvedIntBinaryOperation::Add,
                    children: [0, 0],
                },
            ]
            .into_boxed_slice(),
            root: 1,
        }));
        let substituted = super::super::identity::substitute_resolved_int_expr(
            &binder_plus_one,
            &binder,
            &ResolvedIntExpr::Const(1.into()),
        );
        assert_eq!(
            super::super::identity::resolved_expr_as_arena(&substituted),
            super::super::identity::resolved_expr_as_arena(&one_plus_one)
        );

        let separately_built = ResolvedIntExpr::Arena(Box::new(ResolvedIntExprArena {
            nodes: vec![
                ResolvedIntExprArenaNode::Const(1.into()),
                ResolvedIntExprArenaNode::Const(1.into()),
                ResolvedIntExprArenaNode::Binary {
                    operation: ResolvedIntBinaryOperation::Add,
                    children: [0, 1],
                },
            ]
            .into_boxed_slice(),
            root: 2,
        }));
        assert_eq!(
            super::super::identity::resolved_expr_as_arena(&separately_built),
            super::super::identity::resolved_expr_as_arena(&one_plus_one)
        );
    }
}
