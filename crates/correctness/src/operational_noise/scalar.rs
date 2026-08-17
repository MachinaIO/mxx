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
        AtomicSourceId, BinderKey, CanonicalResidueConvention, ResolvedIntExpr, ResolvedMatrixType,
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
}

/// Compatibility-free name used by the rest of the checker for a resolved sort.
pub type MxxSort = ScalarSort;

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

/// Semantic key used by the coefficient-extraction bridge while the matrix
/// DAG and scalar arena are joined.  It contains only stable identities, never
/// a runtime scalar ID or a debug rendering.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ScalarExtractKey {
    pub operation: ScalarOperation,
    pub matrix: super::normal_form::MatrixValueIdentityId,
    pub position: ResolvedIntExpr,
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
    /// rewrite class or runtime observation.
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
                let sort = convert_sort(descriptor.sort.clone());
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
                let exact = left.integer_domain.as_ref().and_then(IntegerDomain::exact_value);
                let other = right.integer_domain.as_ref().and_then(IntegerDomain::exact_value);
                let (possible_false, possible_true) = match (exact, other) {
                    (Some(left), Some(right)) => match node {
                        ScalarNode::IntEqual(_) => (left != right, left == right),
                        ScalarNode::IntLess(_) => (left >= right, left < right),
                        ScalarNode::IntLessEqual(_) => (left > right, left <= right),
                        _ => unreachable!(),
                    },
                    _ => (true, true),
                };
                Ok(Self::boolean(possible_false, possible_true, ScalarProvenance::Ordinary))
            }
            ScalarNode::BitExtract { .. } => {
                let value = require_int(child(0)?)?;
                let _ = value.scalar_provenance.ok_or(ScalarTransferError::Provenance)?;
                Ok(Self::scalar(
                    ScalarSort::Bool,
                    Some(IntegerDomain::IntervalOnly(IntegerInterval {
                        minimum: 0.into(),
                        maximum: 1.into(),
                    })),
                    ScalarProvenance::Ordinary,
                ))
            }
            ScalarNode::BoolToInt(_) => {
                let value = require_bool(child(0)?)?;
                if value.scalar_provenance != Some(ScalarProvenance::Ordinary) {
                    return Err(ScalarTransferError::SelectorOnly);
                }
                Ok(Self::scalar(
                    ScalarSort::Int,
                    Some(IntegerDomain::IntervalOnly(IntegerInterval {
                        minimum: 0.into(),
                        maximum: 1.into(),
                    })),
                    ScalarProvenance::Ordinary,
                ))
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
        canonical_exclusive_upper: Option<BigUint>,
        matrix: super::normal_form::MatrixValueIdentityId,
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
        matrix: super::normal_form::MatrixValueIdentityId,
        position: ScalarIdentityId,
        canonical_exclusive_upper: Option<BigUint>,
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
    ExtractCoefficient,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScalarEntry {
    pub node: ScalarNode,
    pub identity: ScalarIdentityId,
    pub identity_expr: ResolvedIntExpr,
    pub analysis: ScalarFacts,
}

#[derive(Clone, Debug, Default)]
pub struct ScalarStore {
    entries: Vec<ScalarEntry>,
    by_identity: BTreeMap<ScalarIdentityId, ScalarId>,
    identities: Vec<ScalarIdentityNode>,
    identity_index: BTreeMap<ScalarIdentityNode, ScalarIdentityId>,
}

fn dispose_resolved_int_expr(root: ResolvedIntExpr) {
    let mut work = vec![root];
    while let Some(expr) = work.pop() {
        match expr {
            ResolvedIntExpr::Source { coordinates, .. } => work.extend(coordinates.into_vec()),
            ResolvedIntExpr::Add(left, right) |
            ResolvedIntExpr::Sub(left, right) |
            ResolvedIntExpr::Mul(left, right) |
            ResolvedIntExpr::Div(left, right) |
            ResolvedIntExpr::EuclideanDiv(left, right) |
            ResolvedIntExpr::EuclideanRemainder(left, right) |
            ResolvedIntExpr::RoundDiv(left, right) => {
                work.push(*left);
                work.push(*right);
            }
            ResolvedIntExpr::Log2Ceil(value) => work.push(*value),
            ResolvedIntExpr::ExtractCoefficient { input, position, .. } => {
                work.push(*input);
                work.push(*position);
            }
            ResolvedIntExpr::Const(_) |
            ResolvedIntExpr::Parameter(_) |
            ResolvedIntExpr::Binder(_) => {}
        }
    }
}

impl Drop for ScalarStore {
    fn drop(&mut self) {
        let entries = std::mem::take(&mut self.entries);
        for entry in entries {
            let ScalarEntry { node, identity, identity_expr, analysis } = entry;
            drop(node);
            let _ = identity;
            drop(analysis);
            dispose_resolved_int_expr(identity_expr);
        }
    }
}

impl ScalarStore {
    /// Inserts one typed scalar node using the transfer table and a shallow
    /// semantic key derived from its typed children.
    pub fn intern_node(
        &mut self,
        node: ScalarNode,
        identity_expr: ResolvedIntExpr,
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
            self.entries[id.0 as usize].analysis.merge_direct_extract(facts);
            dispose_resolved_int_expr(identity_expr);
            return Ok(id);
        }
        let id = ScalarId(self.entries.len() as u32);
        self.by_identity.insert(identity, id);
        self.entries.push(ScalarEntry { node, identity, identity_expr, analysis: facts });
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
            ScalarNode::ExtractCoefficient { matrix, position, .. } => {
                ScalarIdentityNode::ExtractCoefficient {
                    matrix: *matrix,
                    position: identity(*position)?,
                    canonical_exclusive_upper: None,
                }
            }
        })
    }

    pub fn intern(
        &mut self,
        node: ScalarNode,
        identity: ResolvedIntExpr,
        facts: ScalarFacts,
    ) -> ScalarId {
        let identity_id = self.intern_expr_identity(&identity);
        if let Some(id) = self.by_identity.get(&identity_id).copied() {
            self.entries[id.0 as usize].analysis.merge_direct_extract(facts);
            dispose_resolved_int_expr(identity);
            return id;
        }
        let id = ScalarId(self.entries.len() as u32);
        self.by_identity.insert(identity_id, id);
        self.entries.push(ScalarEntry {
            node,
            identity: identity_id,
            identity_expr: identity,
            analysis: facts,
        });
        id
    }

    fn intern_expr_identity(&mut self, value: &ResolvedIntExpr) -> ScalarIdentityId {
        // Resolve the expression graph in explicit postorder.  This keeps the
        // identity arena shallow without making stack depth proportional to a
        // user-controlled expression depth.
        let mut work = vec![(value, false)];
        let mut resolved = HashMap::<usize, ScalarIdentityId>::new();
        while let Some((current, expanded)) = work.pop() {
            let key = current as *const ResolvedIntExpr as usize;
            if resolved.contains_key(&key) {
                continue;
            }
            if !expanded {
                work.push((current, true));
                match current {
                    ResolvedIntExpr::Source { coordinates, .. } => {
                        for coordinate in coordinates.iter().rev() {
                            work.push((coordinate, false));
                        }
                    }
                    ResolvedIntExpr::Add(left, right) |
                    ResolvedIntExpr::Sub(left, right) |
                    ResolvedIntExpr::Mul(left, right) |
                    ResolvedIntExpr::Div(left, right) |
                    ResolvedIntExpr::EuclideanDiv(left, right) |
                    ResolvedIntExpr::EuclideanRemainder(left, right) |
                    ResolvedIntExpr::RoundDiv(left, right) => {
                        work.push((right, false));
                        work.push((left, false));
                    }
                    ResolvedIntExpr::Log2Ceil(input) => work.push((input, false)),
                    ResolvedIntExpr::ExtractCoefficient { input, position, .. } => {
                        work.push((position, false));
                        work.push((input, false));
                    }
                    ResolvedIntExpr::Const(_) |
                    ResolvedIntExpr::Parameter(_) |
                    ResolvedIntExpr::Binder(_) => {}
                }
                continue;
            }
            let child = |child: &ResolvedIntExpr| {
                *resolved
                    .get(&(child as *const ResolvedIntExpr as usize))
                    .expect("postorder child identity")
            };
            let node = match current {
                ResolvedIntExpr::Const(value) => ScalarIdentityNode::Const(value.clone()),
                ResolvedIntExpr::Parameter(value) => ScalarIdentityNode::Parameter(value.clone()),
                ResolvedIntExpr::Binder(value) => ScalarIdentityNode::Binder(value.clone()),
                ResolvedIntExpr::Source { source, coordinates } => ScalarIdentityNode::Source {
                    source: source.clone(),
                    coordinates: coordinates.iter().map(child).collect(),
                },
                ResolvedIntExpr::Add(left, right) => ScalarIdentityNode::Binary {
                    operation: ScalarOperation::Add,
                    left: child(left),
                    right: child(right),
                },
                ResolvedIntExpr::Sub(left, right) => ScalarIdentityNode::Binary {
                    operation: ScalarOperation::Sub,
                    left: child(left),
                    right: child(right),
                },
                ResolvedIntExpr::Mul(left, right) => ScalarIdentityNode::Binary {
                    operation: ScalarOperation::Mul,
                    left: child(left),
                    right: child(right),
                },
                ResolvedIntExpr::Div(left, right) => ScalarIdentityNode::Binary {
                    operation: ScalarOperation::ExactDiv,
                    left: child(left),
                    right: child(right),
                },
                ResolvedIntExpr::EuclideanDiv(left, right) => ScalarIdentityNode::Binary {
                    operation: ScalarOperation::EuclideanDiv,
                    left: child(left),
                    right: child(right),
                },
                ResolvedIntExpr::EuclideanRemainder(left, right) => ScalarIdentityNode::Binary {
                    operation: ScalarOperation::EuclideanRemainder,
                    left: child(left),
                    right: child(right),
                },
                ResolvedIntExpr::RoundDiv(left, right) => ScalarIdentityNode::Binary {
                    operation: ScalarOperation::RoundDiv,
                    left: child(left),
                    right: child(right),
                },
                ResolvedIntExpr::Log2Ceil(input) => ScalarIdentityNode::Unary {
                    operation: ScalarOperation::Log2Ceil,
                    input: child(input),
                },
                ResolvedIntExpr::ExtractCoefficient { position, .. } => {
                    // Matrix identities are carried by the scalar entry's typed node;
                    // extraction proof strength is a fact, not a semantic key.
                    ScalarIdentityNode::ExtractCoefficient {
                        matrix: super::normal_form::MatrixValueIdentityId(0),
                        position: child(position),
                        canonical_exclusive_upper: None,
                    }
                }
            };
            resolved.insert(key, self.intern_identity(node));
        }
        *resolved.get(&(value as *const ResolvedIntExpr as usize)).expect("root identity")
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

    pub fn facts(&self, id: ScalarId) -> Option<&ScalarFacts> {
        self.get(id).map(|entry| &entry.analysis)
    }

    pub fn identity(&self, id: ScalarId) -> Option<&ResolvedIntExpr> {
        self.get(id).map(|entry| &entry.identity_expr)
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

    /// Inserts a coefficient extraction without making the matrix expression
    /// a scalar node.  This is the only temporary bridge needed by the matrix
    /// DAG until the lowerer is migrated to `ScalarNode::ExtractCoefficient`.
    pub fn intern_extract(
        &mut self,
        key: ScalarExtractKey,
        extracted_facts: super::analysis::AnalysisData,
    ) -> ScalarId {
        let integer_domain = convert_integer_domain(extracted_facts.integer_domain);
        let facts = ScalarFacts {
            sort: extracted_facts.sort.clone().map(|sort| match sort {
                super::analysis::MxxSort::Int => ScalarSort::Int,
                super::analysis::MxxSort::Bool => ScalarSort::Bool,
                super::analysis::MxxSort::Real => ScalarSort::Real,
                super::analysis::MxxSort::Bytes(length) => ScalarSort::Bytes(length),
                super::analysis::MxxSort::TypedBlob { type_name, schema_hash } => {
                    ScalarSort::TypedBlob { type_name, schema_hash }
                }
                super::analysis::MxxSort::Matrix(matrix) => ScalarSort::Matrix(matrix),
            }),
            integer_domain,
            scalar_provenance: extracted_facts.scalar_provenance.map(
                |provenance| match provenance {
                    super::analysis::ScalarProvenance::Ordinary => ScalarProvenance::Ordinary,
                    super::analysis::ScalarProvenance::SelectorOnly => {
                        ScalarProvenance::SelectorOnly
                    }
                },
            ),
            possible_false: extracted_facts.possible_false,
            possible_true: extracted_facts.possible_true,
            real_constant_bits: extracted_facts.real_constant_bits,
            canonical_coefficient_exclusive_upper: extracted_facts
                .canonical_coefficient_exclusive_upper,
            canonical_residue_convention: extracted_facts.canonical_residue_convention,
            direct_extract: extracted_facts
                .direct_extract
                .map(|fact| DirectExtractFact { canonical_upper: fact.canonical_upper }),
        };
        let position_identity = self.intern_expr_identity(&key.position);
        // The external proof upper is deliberately excluded from this key:
        // the same extraction with two proof strengths is one semantic scalar,
        // and the facts merge conservatively below.
        let extraction_identity = self.intern_identity(ScalarIdentityNode::ExtractCoefficient {
            matrix: key.matrix,
            position: position_identity,
            canonical_exclusive_upper: None,
        });
        if let Some(id) = self.by_identity.get(&extraction_identity).copied() {
            self.entries[id.0 as usize].analysis.merge_direct_extract(facts);
            return id;
        }
        let position_id = self.intern(
            ScalarNode::IntConst(BigInt::zero()),
            ResolvedIntExpr::Const(BigInt::zero()),
            ScalarFacts::scalar(
                ScalarSort::Int,
                Some(IntegerDomain::exact(0)),
                ScalarProvenance::Ordinary,
            ),
        );
        let id = ScalarId(self.entries.len() as u32);
        self.by_identity.insert(extraction_identity, id);
        self.entries.push(ScalarEntry {
            node: ScalarNode::ExtractCoefficient {
                canonical_exclusive_upper: facts
                    .direct_extract
                    .as_ref()
                    .and_then(|fact| fact.canonical_upper.clone()),
                matrix: key.matrix,
                position: position_id,
            },
            identity: extraction_identity,
            identity_expr: ResolvedIntExpr::ExtractCoefficient {
                input: Box::new(ResolvedIntExpr::Const(BigInt::zero())),
                position: Box::new(key.position),
                canonical_exclusive_upper: None,
            },
            analysis: facts,
        });
        id
    }
}

fn convert_integer_domain(domain: Option<super::analysis::IntegerDomain>) -> Option<IntegerDomain> {
    domain.map(|domain| match domain {
        super::analysis::IntegerDomain::Exact(value) => IntegerDomain::Exact(value),
        super::analysis::IntegerDomain::IntervalOnly(interval) => {
            IntegerDomain::IntervalOnly(IntegerInterval {
                minimum: interval.minimum,
                maximum: interval.maximum,
            })
        }
        super::analysis::IntegerDomain::Affine { constant, coefficients, binders } => {
            IntegerDomain::Affine {
                constant,
                coefficients,
                binders: binders
                    .into_iter()
                    .map(|(binder, interval)| {
                        (
                            binder,
                            IntegerInterval {
                                minimum: interval.minimum,
                                maximum: interval.maximum,
                            },
                        )
                    })
                    .collect(),
            }
        }
    })
}

fn convert_sort(sort: super::analysis::MxxSort) -> ScalarSort {
    match sort {
        super::analysis::MxxSort::Int => ScalarSort::Int,
        super::analysis::MxxSort::Bool => ScalarSort::Bool,
        super::analysis::MxxSort::Real => ScalarSort::Real,
        super::analysis::MxxSort::Bytes(length) => ScalarSort::Bytes(length),
        super::analysis::MxxSort::TypedBlob { type_name, schema_hash } => {
            ScalarSort::TypedBlob { type_name, schema_hash }
        }
        super::analysis::MxxSort::Matrix(matrix) => ScalarSort::Matrix(matrix),
    }
}

/// The direct-extraction domain contract, shared by lowering and scalar tests.
pub fn extract_coefficient_domain(
    matrix: &ResolvedMatrixType,
    modulus: &BigUint,
    authoritative_upper: Option<&BigUint>,
    convention: CanonicalResidueConvention,
) -> Result<IntegerDomain, AnalysisError> {
    if modulus.is_zero() {
        return Err(AnalysisError::UnknownCanonicalResidueRange {
            matrix: super::analysis::MxxSort::Matrix(matrix.clone()),
        });
    }
    if let Some(upper) = authoritative_upper {
        if convention != CanonicalResidueConvention::Nonnegative ||
            upper.is_zero() ||
            upper > modulus
        {
            return Err(AnalysisError::UnknownCanonicalResidueRange {
                matrix: super::analysis::MxxSort::Matrix(matrix.clone()),
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
    Ok(IntegerDomain::IntervalOnly(IntegerInterval { minimum, maximum }))
}

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
    let mut facts = ScalarFacts::scalar(MxxSort::Int, Some(domain), ScalarProvenance::SelectorOnly);
    facts.direct_extract = Some(DirectExtractFact { canonical_upper: upper });
    Ok(facts)
}

/// Compatibility alias for code that only needs the scalar facts record.
pub type AnalysisData = ScalarFacts;

/// The lowerer owns this symbol table directly; this helper keeps construction
/// independent of any rewriting library.
pub fn symbols_default() -> SymbolTables {
    SymbolTables::default()
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
            sort: super::super::analysis::MxxSort::Int,
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
            let mut source = |name: &str, sort: super::super::analysis::MxxSort| {
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
            let bytes_id = source(
                "bytes",
                super::super::analysis::MxxSort::Bytes(ResolvedIntExpr::Const(4.into())),
            );
            let blob_id = source(
                "blob",
                super::super::analysis::MxxSort::TypedBlob {
                    type_name: "TestBlob".to_owned(),
                    schema_hash: [7; 32],
                },
            );
            let matrix = ResolvedMatrixType {
                modulus: ResolvedIntExpr::Const(17.into()),
                ring_dimension: ResolvedIntExpr::Const(1.into()),
                rows: ResolvedIntExpr::Const(1.into()),
                columns: ResolvedIntExpr::Const(1.into()),
            };
            let matrix_id = source("matrix", super::super::analysis::MxxSort::Matrix(matrix));
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
    fn semantic_identity_is_shallow_and_direct_extract_proofs_merge() {
        let mut store = ScalarStore::default();
        let mut identity = ResolvedIntExpr::Const(0.into());
        for _ in 0..2048 {
            identity = ResolvedIntExpr::Add(
                Box::new(identity),
                Box::new(ResolvedIntExpr::Const(1.into())),
            );
        }
        let id = store.intern(ScalarNode::IntConst(1.into()), identity.clone(), integer(1));
        assert_eq!(store.identity(id), Some(&identity));
        assert!(store.identity_len() >= 2049);

        let same = store.intern(
            ScalarNode::IntParameter("not-the-node-key".to_owned()),
            identity,
            integer(9),
        );
        assert_eq!(id, same);
        assert_eq!(store.len(), 1);

        let key = ScalarExtractKey {
            operation: ScalarOperation::ExtractCoefficient,
            matrix: super::super::normal_form::MatrixValueIdentityId(11),
            position: ResolvedIntExpr::Const(2.into()),
        };
        let matrix = super::super::identity::ResolvedMatrixType {
            modulus: ResolvedIntExpr::Const(17.into()),
            ring_dimension: ResolvedIntExpr::Const(1.into()),
            rows: ResolvedIntExpr::Const(1.into()),
            columns: ResolvedIntExpr::Const(1.into()),
        };
        let seven = BigUint::from(7_u8);
        let five = BigUint::from(5_u8);
        let mut first = super::super::analysis::MxxAnalysis::direct_extract_data(
            matrix.clone(),
            BigUint::from(17_u8),
            Some(&seven),
        )
        .unwrap();
        first.direct_extract = Some(super::super::analysis::DirectExtractFact {
            canonical_upper: Some(seven.clone()),
        });
        let first_id = store.intern_extract(key.clone(), first);
        let second = super::super::analysis::MxxAnalysis::direct_extract_data(
            matrix,
            BigUint::from(17_u8),
            Some(&five),
        )
        .unwrap();
        let second_id = store.intern_extract(key, second);
        assert_eq!(first_id, second_id);
        assert_eq!(
            store.get(first_id).unwrap().analysis.direct_extract.as_ref().unwrap().canonical_upper,
            Some(five)
        );
    }

    #[test]
    fn deep_diamond_identity_interning_uses_controlled_stack() {
        let worker = thread::Builder::new()
            .name("scalar-identity-depth-test".to_owned())
            .stack_size(128 * 1024)
            .spawn(|| {
                let mut store = ScalarStore::default();
                let build_branch = || {
                    let mut branch = ResolvedIntExpr::Const(1.into());
                    for _ in 0..4096 {
                        branch = ResolvedIntExpr::Add(
                            Box::new(branch),
                            Box::new(ResolvedIntExpr::Const(1.into())),
                        );
                    }
                    branch
                };
                let diamond =
                    ResolvedIntExpr::Add(Box::new(build_branch()), Box::new(build_branch()));
                let first = store.intern(ScalarNode::IntConst(1.into()), diamond, integer(1));
                let duplicate =
                    ResolvedIntExpr::Add(Box::new(build_branch()), Box::new(build_branch()));
                let second = store.intern(
                    ScalarNode::IntParameter("duplicate-diamond".to_owned()),
                    duplicate,
                    integer(2),
                );
                assert_eq!(first, second);
                assert_eq!(store.len(), 1);
                assert_eq!(store.identity_len(), 4098);
                assert!(store.identity(first).is_some());
            })
            .expect("spawn scalar identity depth test");
        worker.join().expect("scalar identity depth test panicked");
    }
}
