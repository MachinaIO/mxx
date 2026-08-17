//! First operation-table transfer rules for the typed normal form.
//!
//! The operations here are pure NF-to-NF transfers.  They consume the typed
//! owner identities and matrix bounds already produced by lowering; no second
//! expression cache or operation-specific identity registry is introduced.

use super::{
    bound::{
        BoundClass, MatrixBound, MatrixMetadata, PolynomialFacts, ResolvedMatrixConstant,
        gadget_matrix_bound,
    },
    identity::{Axis, CrtSpec, ResolvedIndexRange, ResolvedIntExpr, SliceSpec},
    normal_form::{
        BoundedSummary, CompositeDescriptor, CompositeIdentity, FactorIdentity, FactorOwner,
        HashPlainArgumentIdentity, Monomial, NormalFormError, PolynomialNF, SymbolicFactor,
        centered_residue, monomial_bound, scale_by_multiplicity, summary_from_bound_with_facts,
    },
};
use mxx_ir_core::types::ConcreteMatrixType;
use num_bigint::{BigInt, BigUint};
use num_traits::{ToPrimitive, Zero};
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OperationError {
    NormalForm(NormalFormError),
    InvalidSlice,
    InvalidCrt,
    UnresolvedCoefficient,
    InvalidIntegerDomain,
    SelectorOnlyInteger,
    IntervalScaleOfExactSignal,
    InvalidShape,
    InvalidConcat,
    InvalidPack,
    InvalidHash,
    InvalidConstant,
    InvalidView,
}

impl From<NormalFormError> for OperationError {
    fn from(error: NormalFormError) -> Self {
        Self::NormalForm(error)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct IntegerInterval {
    pub minimum: BigInt,
    pub maximum: BigInt,
    pub selector_only: bool,
    /// Selector provenance that came directly from coefficient extraction.
    /// Ordinary selector arithmetic never fills this field.
    pub direct_extract_upper: Option<BigUint>,
}

impl IntegerInterval {
    pub fn new(minimum: BigInt, maximum: BigInt) -> Result<Self, OperationError> {
        if minimum > maximum {
            return Err(OperationError::InvalidIntegerDomain);
        }
        Ok(Self { minimum, maximum, selector_only: false, direct_extract_upper: None })
    }

    pub fn selector_only(mut self) -> Self {
        self.selector_only = true;
        self
    }

    pub fn selector_only_direct_extract(mut self, upper: BigUint) -> Self {
        self.selector_only = true;
        self.direct_extract_upper = Some(upper);
        self
    }

    fn maximum_absolute(&self) -> num_bigint::BigUint {
        self.minimum.magnitude().max(self.maximum.magnitude()).clone()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ViewSpec {
    Identity,
    Rotation { exponent: BigInt },
    Permutation { indices: Box<[usize]> },
    CoefficientPreserving { view: CoefficientPreservingView },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum CoefficientPreservingView {
    Slice(SliceSpec),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoolBit {
    pub value: PolynomialNF,
    pub identity: FactorIdentity,
    pub maximum: BigUint,
    pub position: usize,
    pub weight: BigUint,
    pub is_bool: bool,
    pub known_zero: bool,
}

/// Typed extension operations used by the next checker integration stage.
pub trait PolynomialNFOperations {
    fn transpose_nf(&self) -> Result<PolynomialNF, OperationError>;
    fn slice_nf(&self, spec: &SliceSpec) -> Result<PolynomialNF, OperationError>;
    fn tensor_nf(&self, other: &PolynomialNF) -> Result<PolynomialNF, OperationError>;
    fn lift_constant_polynomial_nf(
        &self,
        matrix_type: ConcreteMatrixType,
        domain: &IntegerInterval,
    ) -> Result<PolynomialNF, OperationError>;
    fn view_nf(
        &self,
        view: &ViewSpec,
        output_type: ConcreteMatrixType,
    ) -> Result<PolynomialNF, OperationError>;
}

/// Multi-input operation whose spec is shared by all CRT lanes.
pub trait CrtRecompose {
    fn crt_recompose_nf(
        inputs: &[PolynomialNF],
        spec: &CrtSpec,
        output_type: ConcreteMatrixType,
    ) -> Result<PolynomialNF, OperationError>;
}

pub trait AdditionalOperations {
    fn concat_nf(
        inputs: &[PolynomialNF],
        axis: Axis,
        output_type: ConcreteMatrixType,
    ) -> Result<PolynomialNF, OperationError>;
    fn pack_polynomial_coefficients_nf(
        bits: &[BoolBit],
        ring_dimension: usize,
        coefficient_bits: usize,
        output_type: ConcreteMatrixType,
    ) -> Result<PolynomialNF, OperationError>;
    fn hash_plain_nf(
        query: FactorIdentity,
        arguments: &[PolynomialNF],
        output_type: ConcreteMatrixType,
        hard_range: Option<BigUint>,
    ) -> Result<PolynomialNF, OperationError>;
    fn matrix_constant_nf(
        key: FactorIdentity,
        constant: &ResolvedMatrixConstant,
        matrix_type: ConcreteMatrixType,
    ) -> Result<PolynomialNF, OperationError>;
}

impl PolynomialNFOperations for PolynomialNF {
    fn transpose_nf(&self) -> Result<PolynomialNF, OperationError> {
        let mut output = PolynomialNF::zero();
        for term in self.exact_terms().values() {
            let factors = term
                .monomial
                .factors()
                .iter()
                .rev()
                .map(|factor| transpose_factor(factor.clone()))
                .collect::<Result<Vec<_>, _>>()?;
            output.insert(Monomial::from_factors(factors), term.multiplicity.clone())?;
        }
        let summary = transpose_summary(self.bounded_summary())?;
        Ok(PolynomialNF::from_parts(output.exact_terms().clone(), summary))
    }

    fn slice_nf(&self, spec: &SliceSpec) -> Result<PolynomialNF, OperationError> {
        let mut output = PolynomialNF::zero();
        for term in self.exact_terms().values() {
            let bound = monomial_bound(&term.monomial).ok();
            let bound = bound.map(|bound| slice_bound(bound, spec)).transpose()?;
            output.insert(
                Monomial::from_factor(structural_factor(
                    CompositeDescriptor::Slice(spec.clone()),
                    &term.monomial,
                    bound,
                )),
                term.multiplicity.clone(),
            )?;
        }
        let summary = match self.bounded_summary() {
            BoundedSummary::ExactZero => BoundedSummary::ExactZero,
            BoundedSummary::Bounded(bound) => super::normal_form::summary_from_bound_with_facts(
                slice_bound(bound.bound.clone(), spec)?,
                bound.polynomial.clone(),
            ),
        };
        Ok(PolynomialNF::from_parts(output.exact_terms().clone(), summary))
    }

    fn tensor_nf(&self, other: &PolynomialNF) -> Result<PolynomialNF, OperationError> {
        if self.is_exact_zero() || other.is_exact_zero() {
            return Ok(PolynomialNF::zero());
        }
        let mut output = PolynomialNF::zero();
        for left in self.exact_terms().values() {
            for right in other.exact_terms().values() {
                let combined = Monomial::from_factors(
                    left.monomial.factors().iter().chain(right.monomial.factors().iter()).cloned(),
                );
                let bound = match (
                    monomial_bound(&left.monomial).ok(),
                    monomial_bound(&right.monomial).ok(),
                ) {
                    (Some(left), Some(right)) => Some(tensor_bound(&left, &right)?),
                    _ => None,
                };
                output.insert(
                    Monomial::from_factor(structural_factor(
                        CompositeDescriptor::Tensor,
                        &combined,
                        bound,
                    )),
                    &left.multiplicity * &right.multiplicity,
                )?;
            }
        }
        if let BoundedSummary::Bounded(left) = self.bounded_summary() {
            for term in other.exact_terms().values() {
                let right = monomial_bound(&term.monomial)
                    .map_err(|_| NormalFormError::BoundedSummaryMixedWithLarge)?;
                output = output.add(PolynomialNF::bounded(scale_by_multiplicity(
                    tensor_bound(left, &right)?,
                    &term.multiplicity,
                ))?)?;
            }
        }
        if let BoundedSummary::Bounded(right) = other.bounded_summary() {
            for term in self.exact_terms().values() {
                let left = monomial_bound(&term.monomial)
                    .map_err(|_| NormalFormError::BoundedSummaryMixedWithLarge)?;
                output = output.add(PolynomialNF::bounded(scale_by_multiplicity(
                    tensor_bound(&left, right)?,
                    &term.multiplicity,
                ))?)?;
            }
        }
        if let (BoundedSummary::Bounded(left), BoundedSummary::Bounded(right)) =
            (self.bounded_summary(), other.bounded_summary())
        {
            output = output.add(PolynomialNF::bounded(tensor_bound(left, right)?)?)?;
        }
        Ok(output)
    }

    fn lift_constant_polynomial_nf(
        &self,
        matrix_type: ConcreteMatrixType,
        domain: &IntegerInterval,
    ) -> Result<PolynomialNF, OperationError> {
        if domain.selector_only && domain.direct_extract_upper.is_none() {
            return Err(OperationError::SelectorOnlyInteger);
        }
        let cap = domain.maximum_absolute();
        if cap.is_zero() || self.is_exact_zero() {
            return Ok(PolynomialNF::zero());
        }
        let bound = MatrixBound { matrix_type, coefficient_class: BoundClass::bounded(cap) };
        let mut output = PolynomialNF::zero();
        for term in self.exact_terms().values() {
            output.insert(
                Monomial::from_factor(structural_factor(
                    CompositeDescriptor::Generic("lift-constant-polynomial".into()),
                    &term.monomial,
                    Some(bound.clone()),
                )),
                term.multiplicity.clone(),
            )?;
        }
        let summary = match self.bounded_summary() {
            BoundedSummary::ExactZero => BoundedSummary::ExactZero,
            BoundedSummary::Bounded(input) => {
                summary_from_bound_with_facts(bound, input.polynomial.clone())
            }
        };
        Ok(PolynomialNF::from_parts(output.exact_terms().clone(), summary))
    }

    fn view_nf(
        &self,
        view: &ViewSpec,
        output_type: ConcreteMatrixType,
    ) -> Result<PolynomialNF, OperationError> {
        if matches!(view, ViewSpec::Identity) {
            validate_shape(self, &output_type).map_err(|_| OperationError::InvalidView)?;
            return Ok(self.clone());
        }
        if let ViewSpec::Permutation { indices } = view {
            if indices.len() != output_type.rows ||
                indices.iter().copied().collect::<std::collections::BTreeSet<_>>().len() !=
                    indices.len() ||
                indices.iter().any(|index| *index >= output_type.rows)
            {
                return Err(OperationError::InvalidView);
            }
        }
        validate_shape(self, &output_type).map_err(|_| OperationError::InvalidView)?;
        let mut output = PolynomialNF::zero();
        for term in self.exact_terms().values() {
            output.insert(
                Monomial::from_factor(structural_factor(
                    CompositeDescriptor::View(view.clone()),
                    &term.monomial,
                    None,
                )),
                term.multiplicity.clone(),
            )?;
        }
        let summary = match self.bounded_summary() {
            BoundedSummary::ExactZero => BoundedSummary::ExactZero,
            BoundedSummary::Bounded(bound) => {
                let mut bound = bound.clone();
                bound.matrix_type = output_type;
                BoundedSummary::Bounded(bound)
            }
        };
        Ok(PolynomialNF::from_parts(output.exact_terms().clone(), summary))
    }
}

impl CrtRecompose for PolynomialNF {
    fn crt_recompose_nf(
        inputs: &[PolynomialNF],
        spec: &CrtSpec,
        output_type: ConcreteMatrixType,
    ) -> Result<PolynomialNF, OperationError> {
        let coefficients = spec
            .reconstruction_coefficients
            .iter()
            .map(resolve_coefficient)
            .collect::<Result<Vec<_>, _>>()?;
        if inputs.is_empty() || inputs.len() != coefficients.len() {
            return Err(OperationError::InvalidCrt);
        }
        let mut output = PolynomialNF::zero();
        for (input, coefficient) in inputs.iter().zip(coefficients) {
            // Zero is checked before shape or bound inspection.  A zero CRT
            // lane therefore cannot turn a Large input into a false failure.
            if coefficient.is_zero() {
                continue;
            }
            validate_shape(input, &output_type)?;
            for term in input.exact_terms().values() {
                output.insert(term.monomial.clone(), &term.multiplicity * &coefficient)?;
            }
            if let BoundedSummary::Bounded(bound) = input.bounded_summary() {
                let mut converted = bound.clone();
                converted.matrix_type = output_type.clone();
                output = output.add(PolynomialNF::from_parts(
                    Default::default(),
                    summary_from_bound_with_facts(
                        scale_by_multiplicity(converted.bound.clone(), &coefficient),
                        converted.polynomial.clone(),
                    ),
                ))?;
            }
        }
        Ok(output)
    }
}

impl AdditionalOperations for PolynomialNF {
    fn concat_nf(
        inputs: &[PolynomialNF],
        axis: Axis,
        output_type: ConcreteMatrixType,
    ) -> Result<PolynomialNF, OperationError> {
        if inputs.is_empty() {
            return Err(OperationError::InvalidConcat);
        }
        if inputs.iter().any(|input| !input.exact_terms().is_empty()) {
            let mut output = PolynomialNF::zero();
            for (input_index, input) in inputs.iter().enumerate() {
                for term in input.exact_terms().values() {
                    output.insert(
                        Monomial::from_factor(structural_factor(
                            CompositeDescriptor::Concat {
                                axis,
                                arity: inputs.len(),
                                input_index,
                                output_type: output_type.clone(),
                            },
                            &term.monomial,
                            None,
                        )),
                        term.multiplicity.clone(),
                    )?;
                }
            }
            let mut summary = BoundedSummary::ExactZero;
            for input in inputs {
                if let BoundedSummary::Bounded(mut bound) = input.bounded_summary().clone() {
                    bound.matrix_type = output_type.clone();
                    summary = max_summary(summary, BoundedSummary::Bounded(bound))?;
                }
            }
            return Ok(PolynomialNF::from_parts(output.exact_terms().clone(), summary));
        }
        let shapes = inputs.iter().map(shape_of_nf).collect::<Result<Vec<_>, _>>()?;
        let expected = concat_shape(&shapes, axis)?;
        if expected != output_type {
            return Err(OperationError::InvalidConcat);
        }
        // A pointwise Add is the only distributive concat form.  Its inputs
        // must already have the output shape; no selector/cartesian expansion
        // is attempted for structural concatenation.
        if shapes.iter().all(|shape| *shape == output_type) &&
            inputs.iter().all(|input| input.exact_terms().is_empty())
        {
            return inputs
                .iter()
                .cloned()
                .map(Ok)
                .try_fold(PolynomialNF::zero(), |left, right| left.add(right?))
                .map_err(OperationError::from);
        }
        let bound = inputs
            .iter()
            .filter_map(|input| input.bounded_summary().as_matrix_bound())
            .try_fold(None, |current, candidate| max_bound(current, candidate.clone()))?;
        let Some(mut bound) = bound else {
            let first = inputs
                .iter()
                .flat_map(|input| input.exact_terms().values())
                .next()
                .ok_or(OperationError::InvalidConcat)?;
            let factor = structural_factor(
                CompositeDescriptor::Generic("concat".into()),
                &first.monomial,
                None,
            );
            let mut output = PolynomialNF::zero();
            output.insert(Monomial::from_factor(factor), BigInt::from(1))?;
            return Ok(output);
        };
        bound.matrix_type = output_type;
        Ok(PolynomialNF::bounded(bound)?)
    }

    fn pack_polynomial_coefficients_nf(
        bits: &[BoolBit],
        ring_dimension: usize,
        coefficient_bits: usize,
        output_type: ConcreteMatrixType,
    ) -> Result<PolynomialNF, OperationError> {
        if ring_dimension == 0 ||
            coefficient_bits == 0 ||
            bits.len() !=
                ring_dimension
                    .checked_mul(coefficient_bits)
                    .ok_or(OperationError::InvalidPack)? ||
            output_type.rows != 1 ||
            output_type.columns != 1 ||
            output_type.modulus.is_zero()
        {
            return Err(OperationError::InvalidPack);
        }
        if bits.iter().any(|bit| {
            bit.value.exact_terms().values().any(|term| {
                !term.monomial.factors().iter().any(|factor| factor.key == bit.identity)
            })
        }) {
            return Err(OperationError::InvalidPack);
        }
        if bits.iter().any(|bit| {
            !bit.is_bool ||
                bit.maximum > BigUint::from(1_u8) ||
                (bit.known_zero && !bit.maximum.is_zero()) ||
                bit.weight != (BigUint::from(1_u8) << bit.position)
        }) || bits.iter().enumerate().any(|(index, bit)| bit.position != index)
        {
            return Err(OperationError::InvalidPack);
        }
        let modulus = output_type.modulus.to_biguint().ok_or(OperationError::InvalidPack)?;
        let cap = modulus - BigUint::from(1_u8);
        let mut maximum = BigUint::ZERO;
        for coefficient in 0..ring_dimension {
            let mut bound = BigUint::ZERO;
            for bit in 0..coefficient_bits {
                let weight = BigUint::from(1_u8) << bit;
                bound += weight * &bits[coefficient * coefficient_bits + bit].maximum;
            }
            maximum = maximum.max(bound.min(cap.clone()));
        }
        if maximum.is_zero() {
            return Ok(PolynomialNF::zero());
        }
        let bound = MatrixBound {
            matrix_type: output_type,
            coefficient_class: BoundClass::bounded(maximum),
        };
        let combined = bits
            .iter()
            .filter(|bit| !bit.known_zero)
            .map(|bit| bit.identity.clone())
            .collect::<Vec<_>>();
        if !combined.is_empty() {
            let mut output = PolynomialNF::zero();
            output.insert(
                Monomial::from_factor(structural_factor(
                    CompositeDescriptor::Generic("pack-bits".into()),
                    &Monomial::from_factors(
                        combined
                            .into_iter()
                            .map(|key| SymbolicFactor::large_typed(key, bound.matrix_type.clone())),
                    ),
                    Some(bound.clone()),
                )),
                BigInt::from(1),
            )?;
            Ok(output)
        } else {
            Ok(PolynomialNF::bounded(bound)?)
        }
    }

    fn hash_plain_nf(
        query: FactorIdentity,
        arguments: &[PolynomialNF],
        output_type: ConcreteMatrixType,
        hard_range: Option<BigUint>,
    ) -> Result<PolynomialNF, OperationError> {
        if output_type.rows == 0 || output_type.columns == 0 || output_type.modulus.is_zero() {
            return Err(OperationError::InvalidHash);
        }
        let argument_keys = arguments
            .iter()
            .map(|argument| {
                let mut terms = argument.exact_terms().values();
                if let Some(term) = terms.next() {
                    if terms.next().is_some() ||
                        term.multiplicity != BigInt::from(1_u8) ||
                        !argument.bounded_summary().is_exact_zero() ||
                        term.monomial.factors().len() != 1
                    {
                        return Err(OperationError::InvalidHash);
                    }
                    return Ok(HashPlainArgumentIdentity::Exact(
                        term.monomial.factors()[0].key.clone(),
                    ));
                }
                match argument.bounded_summary() {
                    BoundedSummary::ExactZero => Ok(HashPlainArgumentIdentity::ExactZero),
                    BoundedSummary::Bounded(bound) => Ok(HashPlainArgumentIdentity::Bounded {
                        matrix_type: bound.matrix_type.clone(),
                        coefficient_class: bound.coefficient_class.clone(),
                    }),
                }
            })
            .collect::<Result<Box<_>, _>>()?;
        let mut key = query.clone();
        key.owner = FactorOwner::HashPlain { query: Box::new(query), arguments: argument_keys };
        match hard_range {
            Some(range) if range.is_zero() => Ok(PolynomialNF::zero()),
            Some(range) => {
                let ring_dimension = output_type.ring_dimension;
                let matrix_bound = MatrixBound {
                    matrix_type: output_type.clone(),
                    coefficient_class: BoundClass::bounded(range.clone()),
                };
                let factor = SymbolicFactor {
                    key,
                    bound: BoundClass::bounded(range.clone()),
                    relation_live: false,
                    trapdoor: None,
                    matrix_bound: Some(matrix_bound),
                    matrix_type: output_type,
                    polynomial_facts: PolynomialFacts::conservative(ring_dimension),
                    matrix_value_metadata: MatrixMetadata::unknown(),
                    switch: None,
                };
                let mut output = PolynomialNF::zero();
                output.insert(Monomial::from_factor(factor), BigInt::from(1))?;
                Ok(output)
            }
            None => Ok(PolynomialNF::exact_factor_typed(key, output_type)),
        }
    }

    fn matrix_constant_nf(
        key: FactorIdentity,
        constant: &ResolvedMatrixConstant,
        matrix_type: ConcreteMatrixType,
    ) -> Result<PolynomialNF, OperationError> {
        if matrix_type.rows == 0 ||
            matrix_type.columns == 0 ||
            matrix_type.ring_dimension == 0 ||
            matrix_type.modulus.is_zero()
        {
            return Err(OperationError::InvalidConstant);
        }
        let (class, is_constant_polynomial, support_upper) = match constant {
            ResolvedMatrixConstant::Zero => return Ok(PolynomialNF::zero()),
            ResolvedMatrixConstant::Identity => {
                if matrix_type.rows != matrix_type.columns {
                    return Err(OperationError::InvalidConstant);
                }
                (BoundClass::bounded(1_u8.into()), false, None)
            }
            ResolvedMatrixConstant::UnitRow { index } => {
                if matrix_type.rows != 1 || index >= &BigUint::from(matrix_type.columns) {
                    return Err(OperationError::InvalidConstant);
                }
                (BoundClass::bounded(1_u8.into()), false, None)
            }
            ResolvedMatrixConstant::UnitColumn { index } => {
                if matrix_type.columns != 1 || index >= &BigUint::from(matrix_type.rows) {
                    return Err(OperationError::InvalidConstant);
                }
                (BoundClass::bounded(1_u8.into()), false, None)
            }
            ResolvedMatrixConstant::Gadget { base, small } => {
                let class =
                    gadget_matrix_bound(base, *small).ok_or(OperationError::InvalidConstant)?;
                (class, false, None)
            }
            ResolvedMatrixConstant::PowerOfBase { base, exponent } => {
                let exponent = exponent.to_u32().ok_or(OperationError::InvalidConstant)?;
                if base <= &BigInt::from(1_u8) {
                    return Err(OperationError::InvalidConstant);
                }
                (BoundClass::bounded(base.magnitude().pow(exponent)), false, None)
            }
            ResolvedMatrixConstant::Rotation { .. } => {
                (BoundClass::bounded(1_u8.into()), false, None)
            }
            ResolvedMatrixConstant::Polynomial { coefficients } => {
                let coefficients = coefficients
                    .iter()
                    .map(|coefficient| centered_residue(coefficient, &matrix_type.modulus))
                    .collect::<Vec<_>>();
                if coefficients.iter().all(|coefficient| coefficient.is_zero()) {
                    return Ok(PolynomialNF::zero());
                }
                let maximum = coefficients
                    .iter()
                    .map(|coefficient| coefficient.magnitude().clone())
                    .max()
                    .unwrap_or_default();
                let support =
                    coefficients.iter().filter(|coefficient| !coefficient.is_zero()).count();
                if support > matrix_type.ring_dimension {
                    return Err(OperationError::InvalidConstant);
                }
                let support = PolynomialFacts::new(support, matrix_type.ring_dimension)
                    .map_err(|_| OperationError::InvalidConstant)?;
                let is_constant =
                    coefficients.iter().skip(1).all(|coefficient| coefficient.is_zero());
                (BoundClass::bounded(maximum), is_constant, Some(support))
            }
        };
        let scalar_shape = matrix_type.rows == 1 && matrix_type.columns == 1;
        if scalar_shape &&
            matches!(constant, ResolvedMatrixConstant::Polynomial { .. }) &&
            is_constant_polynomial &&
            matches!(
                constant,
                ResolvedMatrixConstant::Polynomial { coefficients }
                    if coefficients.first().is_some_and(|coefficient| {
                        centered_residue(coefficient, &matrix_type.modulus) == BigInt::from(1_u8)
                    })
            )
        {
            return Ok(PolynomialNF::one());
        }
        let key = match constant {
            ResolvedMatrixConstant::Polynomial { coefficients }
                if scalar_shape && is_constant_polynomial && !coefficients.is_empty() =>
            {
                FactorIdentity::constant_polynomial(&matrix_type, &coefficients[0])
            }
            _ => key,
        };
        let support_upper = support_upper.or_else(|| {
            scalar_shape.then(|| PolynomialFacts::new(1, matrix_type.ring_dimension).ok()).flatten()
        });
        let metadata = MatrixMetadata {
            canonical_coefficient_exclusive_upper: None,
            is_constant_polynomial,
            known_zero_rows: None,
            polynomial: support_upper,
        };
        if matches!(class, BoundClass::Large) {
            let factor = SymbolicFactor::large_with_metadata(
                key,
                MatrixBound {
                    matrix_type: matrix_type.clone(),
                    coefficient_class: BoundClass::Large,
                },
                metadata,
            );
            let mut output = PolynomialNF::zero();
            output.insert(Monomial::from_factor(factor), BigInt::from(1))?;
            return Ok(output);
        }
        let bound = MatrixBound { matrix_type: matrix_type.clone(), coefficient_class: class };
        if scalar_shape {
            let factor = SymbolicFactor::bounded_with_metadata(key, bound, metadata)
                .map_err(OperationError::from)?;
            let mut output = PolynomialNF::zero();
            output.insert(Monomial::from_factor(factor), BigInt::from(1))?;
            Ok(output)
        } else {
            Ok(PolynomialNF::bounded(bound)?)
        }
    }
}

fn shape_of_nf(input: &PolynomialNF) -> Result<ConcreteMatrixType, OperationError> {
    if let BoundedSummary::Bounded(bound) = input.bounded_summary() {
        return Ok(bound.matrix_type.clone());
    }
    input
        .exact_terms()
        .values()
        .find_map(|term| monomial_bound(&term.monomial).ok())
        .map(|bound| bound.matrix_type)
        .ok_or(OperationError::InvalidConcat)
}

fn concat_shape(
    shapes: &[ConcreteMatrixType],
    axis: Axis,
) -> Result<ConcreteMatrixType, OperationError> {
    let first = shapes.first().ok_or(OperationError::InvalidConcat)?;
    if shapes
        .iter()
        .any(|shape| shape.modulus != first.modulus || shape.ring_dimension != first.ring_dimension)
    {
        return Err(OperationError::InvalidConcat);
    }
    let rows = match axis {
        Axis::Rows | Axis::Diagonal => shapes
            .iter()
            .try_fold(0_usize, |sum, shape| sum.checked_add(shape.rows))
            .ok_or(OperationError::InvalidConcat)?,
        Axis::Columns => first.rows,
    };
    let columns = match axis {
        Axis::Columns | Axis::Diagonal => shapes
            .iter()
            .try_fold(0_usize, |sum, shape| sum.checked_add(shape.columns))
            .ok_or(OperationError::InvalidConcat)?,
        Axis::Rows => first.columns,
    };
    if matches!(axis, Axis::Rows) && shapes.iter().any(|shape| shape.columns != first.columns) {
        return Err(OperationError::InvalidConcat);
    }
    if matches!(axis, Axis::Columns) && shapes.iter().any(|shape| shape.rows != first.rows) {
        return Err(OperationError::InvalidConcat);
    }
    Ok(ConcreteMatrixType {
        modulus: first.modulus.clone(),
        ring_dimension: first.ring_dimension,
        rows,
        columns,
    })
}

fn max_bound(
    current: Option<MatrixBound>,
    candidate: MatrixBound,
) -> Result<Option<MatrixBound>, OperationError> {
    let Some(mut current) = current else { return Ok(Some(candidate)) };
    if current.matrix_type != candidate.matrix_type {
        return Err(OperationError::InvalidConcat);
    }
    current.coefficient_class = match (current.coefficient_class, candidate.coefficient_class) {
        (BoundClass::ExactZero, value) | (value, BoundClass::ExactZero) => value,
        (
            BoundClass::Bounded { maximum_absolute_coefficient: left },
            BoundClass::Bounded { maximum_absolute_coefficient: right },
        ) => BoundClass::bounded(left.max(right)),
        _ => BoundClass::Large,
    };
    Ok(Some(current))
}

fn max_summary(
    current: BoundedSummary,
    candidate: BoundedSummary,
) -> Result<BoundedSummary, OperationError> {
    match (current, candidate) {
        (BoundedSummary::ExactZero, value) | (value, BoundedSummary::ExactZero) => Ok(value),
        (BoundedSummary::Bounded(mut current), BoundedSummary::Bounded(candidate)) => {
            if current.matrix_type != candidate.matrix_type {
                return Err(OperationError::InvalidConcat);
            }
            current.coefficient_class =
                match (current.coefficient_class.clone(), candidate.coefficient_class.clone()) {
                    (BoundClass::ExactZero, value) | (value, BoundClass::ExactZero) => value,
                    (
                        BoundClass::Bounded { maximum_absolute_coefficient: left },
                        BoundClass::Bounded { maximum_absolute_coefficient: right },
                    ) => BoundClass::bounded(left.max(right)),
                    _ => return Err(OperationError::InvalidConcat),
                };
            current.polynomial.support_upper =
                current.polynomial.support_upper.max(candidate.polynomial.support_upper);
            Ok(BoundedSummary::Bounded(current))
        }
    }
}

fn resolve_coefficient(value: &ResolvedIntExpr) -> Result<BigInt, OperationError> {
    match value {
        ResolvedIntExpr::Const(value) => Ok(value.clone()),
        _ => Err(OperationError::UnresolvedCoefficient),
    }
}

fn transpose_factor(mut factor: SymbolicFactor) -> Result<SymbolicFactor, OperationError> {
    if let FactorOwner::Derived { parent, tag } = &factor.key.owner {
        if tag.as_ref() == b"transpose" {
            factor.key = (**parent).clone();
            if let Some(bound) = factor.matrix_bound.as_mut() {
                std::mem::swap(&mut bound.matrix_type.rows, &mut bound.matrix_type.columns);
                factor.matrix_value_metadata.known_zero_rows = None;
            }
            factor.relation_live = false;
            return Ok(factor);
        }
    }
    factor.key.owner = FactorOwner::Derived {
        parent: Box::new(factor.key.clone()),
        tag: b"transpose".to_vec().into_boxed_slice(),
    };
    if let Some(bound) = factor.matrix_bound.as_mut() {
        std::mem::swap(&mut bound.matrix_type.rows, &mut bound.matrix_type.columns);
        factor.matrix_value_metadata.known_zero_rows = None;
    }
    factor.relation_live = false;
    Ok(factor)
}

fn structural_factor(
    descriptor: CompositeDescriptor,
    monomial: &Monomial,
    bound: Option<MatrixBound>,
) -> SymbolicFactor {
    let factors = monomial.factors();
    let first = factors.first().expect("non-empty structural monomial");
    let mut key = first.key.clone();
    key.owner = FactorOwner::Composite(Box::new(CompositeIdentity {
        descriptor,
        operands: monomial.iter_factors().map(|factor| factor.key.clone()).collect(),
    }));
    let (class, matrix_bound, matrix_type, polynomial_facts) = bound
        .map(|bound| {
            let matrix_type = bound.matrix_type.clone();
            let facts = PolynomialFacts::conservative(matrix_type.ring_dimension);
            (bound.coefficient_class.clone(), Some(bound), matrix_type, facts)
        })
        .unwrap_or_else(|| {
            (BoundClass::Large, None, first.matrix_type.clone(), first.polynomial_facts.clone())
        });
    SymbolicFactor {
        key,
        bound: class,
        relation_live: false,
        trapdoor: None,
        matrix_bound,
        matrix_type,
        polynomial_facts,
        matrix_value_metadata: MatrixMetadata::unknown(),
        switch: None,
    }
}

fn transpose_summary(summary: &BoundedSummary) -> Result<BoundedSummary, OperationError> {
    match summary {
        BoundedSummary::ExactZero => Ok(BoundedSummary::ExactZero),
        BoundedSummary::Bounded(bound) => {
            let mut bound = bound.clone();
            let rows = bound.matrix_type.rows;
            bound.matrix_type.rows = bound.matrix_type.columns;
            bound.matrix_type.columns = rows;
            Ok(summary_from_bound_with_facts(bound.bound, bound.polynomial))
        }
    }
}

fn tensor_bound(left: &MatrixBound, right: &MatrixBound) -> Result<MatrixBound, OperationError> {
    if left.matrix_type.modulus != right.matrix_type.modulus ||
        left.matrix_type.ring_dimension != right.matrix_type.ring_dimension
    {
        return Err(OperationError::InvalidShape);
    }
    let coefficient_class = match (&left.coefficient_class, &right.coefficient_class) {
        (BoundClass::ExactZero, _) | (_, BoundClass::ExactZero) => BoundClass::ExactZero,
        (
            BoundClass::Bounded { maximum_absolute_coefficient: coeff_left },
            BoundClass::Bounded { maximum_absolute_coefficient: coeff_right },
        ) => BoundClass::bounded(
            coeff_left * coeff_right * BigUint::from(left.matrix_type.ring_dimension),
        ),
        _ => BoundClass::Large,
    };
    if matches!(coefficient_class, BoundClass::Large) {
        return Err(OperationError::NormalForm(NormalFormError::LargeBoundCannotBeSummarized));
    }
    Ok(MatrixBound {
        matrix_type: ConcreteMatrixType {
            modulus: left.matrix_type.modulus.clone(),
            ring_dimension: left.matrix_type.ring_dimension,
            rows: left
                .matrix_type
                .rows
                .checked_mul(right.matrix_type.rows)
                .ok_or(OperationError::InvalidShape)?,
            columns: left
                .matrix_type
                .columns
                .checked_mul(right.matrix_type.columns)
                .ok_or(OperationError::InvalidShape)?,
        },
        coefficient_class,
    })
}

fn slice_bound(mut bound: MatrixBound, spec: &SliceSpec) -> Result<MatrixBound, OperationError> {
    if let Some(range) = &spec.rows {
        bound.matrix_type.rows = range_length(range, bound.matrix_type.rows)?;
    }
    if let Some(range) = &spec.columns {
        bound.matrix_type.columns = range_length(range, bound.matrix_type.columns)?;
    }
    Ok(bound)
}

fn range_length(range: &ResolvedIndexRange, extent: usize) -> Result<usize, OperationError> {
    let start = range_value(&range.start)?;
    let end = range_value(&range.end)?;
    if start >= end || end > extent {
        return Err(OperationError::InvalidSlice);
    }
    Ok(end - start)
}

fn range_value(value: &ResolvedIntExpr) -> Result<usize, OperationError> {
    match value {
        ResolvedIntExpr::Const(value) => value.to_usize().ok_or(OperationError::InvalidSlice),
        _ => Err(OperationError::InvalidSlice),
    }
}

fn validate_shape(input: &PolynomialNF, output: &ConcreteMatrixType) -> Result<(), OperationError> {
    if let BoundedSummary::Bounded(bound) = input.bounded_summary() {
        if bound.matrix_type.ring_dimension != output.ring_dimension ||
            bound.matrix_type.rows != output.rows ||
            bound.matrix_type.columns != output.columns
        {
            return Err(OperationError::InvalidCrt);
        }
    }
    for term in input.exact_terms().values() {
        if let Ok(bound) = monomial_bound(&term.monomial) {
            if bound.matrix_type.ring_dimension != output.ring_dimension ||
                bound.matrix_type.rows != output.rows ||
                bound.matrix_type.columns != output.columns
            {
                return Err(OperationError::InvalidCrt);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::{
        identity::{ResolvedIndexRange, ResolvedIntExpr},
        normal_form::{
            ExpressionDag, ExpressionNode, FactorIdentity, FullRelationKey, RelationPattern,
            RelationRegistration, RelationRegistry, SymbolicFactor,
            normal_form_product::{Normalizer, product_bound_only},
        },
    };

    fn matrix_type(rows: usize, columns: usize) -> ConcreteMatrixType {
        ConcreteMatrixType { modulus: 17.into(), ring_dimension: 1, rows, columns }
    }

    fn relation_matrix_bound(value: u64) -> MatrixBound {
        MatrixBound {
            matrix_type: matrix_type(2, 2),
            coefficient_class: BoundClass::bounded(value.into()),
        }
    }

    fn finite(name: &str, rows: usize, columns: usize, bound: u64) -> PolynomialNF {
        PolynomialNF::bounded_factor(
            FactorIdentity::named(name),
            MatrixBound {
                matrix_type: matrix_type(rows, columns),
                coefficient_class: BoundClass::bounded(bound.into()),
            },
        )
        .unwrap()
    }

    fn finite_with_metadata(
        name: &str,
        rows: usize,
        columns: usize,
        bound: u64,
        _metadata: MatrixMetadata,
    ) -> PolynomialNF {
        PolynomialNF::bounded_factor(
            FactorIdentity::named(name),
            MatrixBound {
                matrix_type: matrix_type(rows, columns),
                coefficient_class: BoundClass::bounded(bound.into()),
            },
        )
        .unwrap()
    }

    fn bool_bit(name: &str, position: usize, maximum: u64, known_zero: bool) -> BoolBit {
        let identity = FactorIdentity::named(name);
        BoolBit {
            value: PolynomialNF::exact_factor(identity.clone()),
            identity,
            maximum: maximum.into(),
            position,
            weight: BigUint::from(1_u8) << position,
            is_bool: true,
            known_zero,
        }
    }

    #[test]
    fn transpose_reverses_exact_order_and_double_transpose_removes_view() {
        let left = PolynomialNF::exact_factor(FactorIdentity::named("A"));
        let right = PolynomialNF::exact_factor(FactorIdentity::named("B"));
        let product = product_bound_only(left, right).unwrap();
        let once = product.transpose_nf().unwrap();
        let twice = once.transpose_nf().unwrap();
        assert_eq!(twice, product);
        let once_keys = once
            .exact_terms()
            .values()
            .next()
            .unwrap()
            .monomial
            .factors()
            .iter()
            .map(|factor| factor.key.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            once_keys[0].owner,
            FactorOwner::Derived {
                parent: Box::new(FactorIdentity::named("B")),
                tag: b"transpose".to_vec().into_boxed_slice(),
            }
        );
    }

    #[test]
    fn operation_product_closure_applies_registered_relation() {
        let public = FactorIdentity::named("B");
        let preimage = FactorIdentity::named("K");
        let target = FactorIdentity::named("P");
        let mut dag = ExpressionDag::new();
        let target_term =
            dag.push(ExpressionNode::Atom(SymbolicFactor::large(target.clone()))).unwrap();
        let mut registry = RelationRegistry::default();
        registry
            .register(RelationRegistration {
                pattern: RelationPattern::ordered(public.clone(), preimage.clone()),
                key: FullRelationKey {
                    source: "named".into(),
                    ordered_indices: Box::new([]),
                    public: public.clone(),
                    target: target.clone(),
                    matrix_type: None,
                    layout: None,
                    trapdoor: None,
                    selector: None,
                },
                preimage: preimage.clone(),
                target: target_term,
            })
            .unwrap();
        let left = PolynomialNF::exact_factor(public);
        let right = PolynomialNF::relation_live_factor(preimage, relation_matrix_bound(1)).unwrap();
        let mut normalizer = Normalizer::new(&dag, &registry);
        let result = normalizer.product_and_normalize(left, right).unwrap();
        assert_eq!(result.first_large_witness().unwrap().identity, target);
    }

    #[test]
    fn slice_validates_ranges_and_transfers_shape() {
        let input = finite("A", 4, 5, 3);
        let spec = SliceSpec {
            rows: Some(ResolvedIndexRange {
                start: ResolvedIntExpr::Const(1.into()),
                end: ResolvedIntExpr::Const(3.into()),
            }),
            columns: Some(ResolvedIndexRange {
                start: ResolvedIntExpr::Const(2.into()),
                end: ResolvedIntExpr::Const(5.into()),
            }),
        };
        assert_eq!(
            input.slice_nf(&spec).unwrap().bounded_summary().as_matrix_bound().unwrap().matrix_type,
            matrix_type(2, 3)
        );
        let invalid = SliceSpec {
            rows: Some(ResolvedIndexRange {
                start: ResolvedIntExpr::Const(2.into()),
                end: ResolvedIntExpr::Const(2.into()),
            }),
            columns: None,
        };
        assert_eq!(input.slice_nf(&invalid), Err(OperationError::InvalidSlice));
    }

    #[test]
    fn crt_zero_coefficient_skips_large_lane() {
        let large = PolynomialNF::exact_factor(FactorIdentity::named("large"));
        let finite = finite("noise", 1, 1, 4);
        let spec = CrtSpec {
            plaintext_moduli: vec![
                ResolvedIntExpr::Const(3.into()),
                ResolvedIntExpr::Const(5.into()),
            ]
            .into(),
            reconstruction_coefficients: vec![
                ResolvedIntExpr::Const(0.into()),
                ResolvedIntExpr::Const(2.into()),
            ]
            .into(),
        };
        let result = PolynomialNF::crt_recompose_nf(&[large, finite], &spec, matrix_type(1, 1))
            .unwrap()
            .finish_relation_live()
            .unwrap();
        assert!(result.exact_terms().is_empty());
        assert_eq!(
            result.bounded_summary().as_matrix_bound().unwrap().coefficient_class,
            BoundClass::bounded(8_u64.into())
        );
    }

    #[test]
    fn tensor_multiplies_bounds_and_shapes_without_inner_sum() {
        let left = finite("left", 2, 3, 2);
        let right = finite("right", 4, 5, 3);
        let result = left.tensor_nf(&right).unwrap();
        let bound = result.bounded_summary().as_matrix_bound().unwrap();
        assert_eq!(bound.matrix_type, matrix_type(8, 15));
        assert_eq!(bound.coefficient_class, BoundClass::bounded(6_u64.into()));
    }

    #[test]
    fn lift_marks_constant_polynomial() {
        let input = finite("input", 1, 1, 1);
        let domain = IntegerInterval::new((-2).into(), 3.into()).unwrap();
        let lifted = input
            .lift_constant_polynomial_nf(matrix_type(1, 1), &domain)
            .unwrap()
            .finish_relation_live()
            .unwrap();
        let lifted_bound = lifted.bounded_summary().as_matrix_bound().unwrap();
        assert_eq!(lifted_bound.coefficient_class, BoundClass::bounded(3_u64.into()));
    }

    #[test]
    fn direct_extract_lift_preserves_authoritative_upper() {
        let input = finite("input", 1, 1, 1);
        let domain = IntegerInterval::new(0.into(), 6.into())
            .unwrap()
            .selector_only_direct_extract(7_u8.into());
        let _lifted = input.lift_constant_polynomial_nf(matrix_type(1, 1), &domain).unwrap();
        assert_eq!(
            input.lift_constant_polynomial_nf(
                matrix_type(1, 1),
                &IntegerInterval::new(0.into(), 6.into()).unwrap().selector_only(),
            ),
            Err(OperationError::SelectorOnlyInteger)
        );
    }

    #[test]
    fn coefficient_preserving_views_keep_metadata_but_arithmetic_clears_it() {
        let metadata = MatrixMetadata {
            canonical_coefficient_exclusive_upper: Some(11_u8.into()),
            is_constant_polynomial: true,
            known_zero_rows: Some(1_u8.into()),
            polynomial: None,
        };
        let input = finite_with_metadata("input", 2, 2, 3, metadata);
        let slice = SliceSpec {
            rows: None,
            columns: Some(ResolvedIndexRange {
                start: ResolvedIntExpr::Const(0.into()),
                end: ResolvedIntExpr::Const(1.into()),
            }),
        };
        let sliced = input.slice_nf(&slice).unwrap();
        assert_eq!(
            sliced.bounded_summary().as_matrix_bound().unwrap().matrix_type,
            matrix_type(2, 1)
        );

        let transposed = input.transpose_nf().unwrap();
        assert_eq!(
            transposed.bounded_summary().as_matrix_bound().unwrap().matrix_type,
            matrix_type(2, 2)
        );

        let sum = input.clone().add(finite("other", 2, 2, 2)).unwrap();
        assert!(sum.bounded_summary().as_matrix_bound().is_some());
        let negated = input.negate();
        assert!(negated.bounded_summary().as_matrix_bound().is_some());
    }

    #[test]
    fn concat_uses_reachable_maximum_and_validates_axis_shape() {
        let inputs = [finite("a", 2, 1, 2), finite("b", 2, 1, 3)];
        let output = PolynomialNF::concat_nf(&inputs, Axis::Rows, matrix_type(4, 1)).unwrap();
        assert_eq!(
            output.bounded_summary().as_matrix_bound().unwrap().coefficient_class,
            BoundClass::bounded(3_u64.into())
        );
        assert_eq!(
            PolynomialNF::concat_nf(&inputs, Axis::Rows, matrix_type(3, 1)),
            Err(OperationError::InvalidConcat)
        );
    }

    #[test]
    fn concat_preserves_all_exact_inputs_in_ordered_structural_terms() {
        let inputs = [
            PolynomialNF::exact_factor(FactorIdentity::named("A")),
            PolynomialNF::exact_factor(FactorIdentity::named("B")),
        ];
        let output = PolynomialNF::concat_nf(&inputs, Axis::Rows, matrix_type(4, 1)).unwrap();
        assert_eq!(output.exact_terms().len(), 2);
        assert!(output.first_large_witness().is_some());
    }

    #[test]
    fn pack_uses_bit_weights_and_modulus_cap() {
        let bits = [bool_bit("b0", 0, 1, false), bool_bit("b1", 1, 1, false)];
        let packed =
            PolynomialNF::pack_polynomial_coefficients_nf(&bits, 1, 2, matrix_type(1, 1)).unwrap();
        assert_eq!(
            packed.exact_terms().values().next().unwrap().monomial.factors()[0].bound,
            BoundClass::bounded(3_u64.into())
        );
        let bad_bits = [bool_bit("bad", 0, 2, false), bool_bit("b1", 1, 1, false)];
        assert_eq!(
            PolynomialNF::pack_polynomial_coefficients_nf(&bad_bits, 1, 2, matrix_type(1, 1),),
            Err(OperationError::InvalidPack)
        );
    }

    #[test]
    fn hash_hard_range_is_bounded_and_missing_range_is_large() {
        let bounded = PolynomialNF::hash_plain_nf(
            FactorIdentity::named("query"),
            &[],
            matrix_type(1, 1),
            Some(7_u64.into()),
        )
        .unwrap();
        let bounded_factor =
            bounded.exact_terms().values().next().unwrap().monomial.factors()[0].clone();
        assert_eq!(bounded_factor.bound, BoundClass::bounded(7_u64.into()));
        assert!(
            PolynomialNF::hash_plain_nf(
                FactorIdentity::named("query"),
                &[],
                matrix_type(1, 1),
                None
            )
            .unwrap()
            .first_large_witness()
            .is_some()
        );
    }

    #[test]
    fn plain_hash_identity_preserves_ordered_canonical_scalar_arguments() {
        let query = FactorIdentity::named("query");
        let scalar = |value: i64| {
            PolynomialNF::exact_factor(FactorIdentity::scalar_selector(ResolvedIntExpr::Const(
                value.into(),
            )))
        };
        let first = PolynomialNF::hash_plain_nf(
            query.clone(),
            &[scalar(3), scalar(5)],
            matrix_type(1, 1),
            None,
        )
        .unwrap();
        let equal = PolynomialNF::hash_plain_nf(
            query.clone(),
            &[scalar(3), scalar(5)],
            matrix_type(1, 1),
            None,
        )
        .unwrap();
        let changed = PolynomialNF::hash_plain_nf(
            query.clone(),
            &[scalar(3), scalar(7)],
            matrix_type(1, 1),
            None,
        )
        .unwrap();
        let reordered =
            PolynomialNF::hash_plain_nf(query, &[scalar(5), scalar(3)], matrix_type(1, 1), None)
                .unwrap();
        assert_eq!(first.exact_terms(), equal.exact_terms());
        assert_ne!(first.exact_terms(), changed.exact_terms());
        assert_ne!(first.exact_terms(), reordered.exact_terms());
        let key = first.exact_terms().keys().next().unwrap();
        let key_factors = key.factors();
        let arguments = match &key_factors[0].owner {
            FactorOwner::HashPlain { arguments, .. } => arguments,
            owner => panic!("unexpected plain-hash owner: {owner:?}"),
        };
        assert_eq!(arguments.len(), 2);
        assert_eq!(
            arguments[0],
            HashPlainArgumentIdentity::Exact(FactorIdentity::scalar_selector(
                ResolvedIntExpr::Const(3.into()),
            ))
        );
        assert_eq!(
            arguments[1],
            HashPlainArgumentIdentity::Exact(FactorIdentity::scalar_selector(
                ResolvedIntExpr::Const(5.into()),
            ))
        );
    }

    #[test]
    fn constants_and_views_validate_contracts() {
        let identity = PolynomialNF::matrix_constant_nf(
            FactorIdentity::named("I"),
            &ResolvedMatrixConstant::Identity,
            matrix_type(2, 2),
        )
        .unwrap();
        assert_eq!(
            identity.bounded_summary().as_matrix_bound().unwrap().coefficient_class,
            BoundClass::bounded(1_u64.into())
        );
        let power = PolynomialNF::matrix_constant_nf(
            FactorIdentity::named("pow"),
            &ResolvedMatrixConstant::PowerOfBase { base: 2.into(), exponent: 3_u64.into() },
            matrix_type(1, 1),
        )
        .unwrap();
        let power_bound = power
            .bounded_summary()
            .as_matrix_bound()
            .map(|bound| bound.coefficient_class.clone())
            .or_else(|| {
                power
                    .exact_terms()
                    .values()
                    .next()
                    .and_then(|term| monomial_bound(&term.monomial).ok())
                    .map(|bound| bound.coefficient_class)
            });
        assert_eq!(power_bound, Some(BoundClass::bounded(8_u64.into())));
        let input = finite("view", 2, 2, 2);
        assert_eq!(input.view_nf(&ViewSpec::Identity, matrix_type(2, 2)).unwrap(), input);
        assert!(
            input.view_nf(&ViewSpec::Rotation { exponent: 1.into() }, matrix_type(2, 2)).is_ok()
        );
        assert!(
            input
                .view_nf(
                    &ViewSpec::CoefficientPreserving {
                        view: CoefficientPreservingView::Slice(SliceSpec {
                            rows: None,
                            columns: None
                        }),
                    },
                    matrix_type(2, 2)
                )
                .is_ok()
        );
        assert!(
            input
                .view_nf(&ViewSpec::Permutation { indices: vec![0, 0].into() }, matrix_type(2, 2))
                .is_err()
        );
    }

    #[test]
    fn tensor_includes_ring_factor_and_scalar_keys_are_typed() {
        let left = PolynomialNF::bounded(MatrixBound {
            matrix_type: ConcreteMatrixType {
                modulus: 17.into(),
                ring_dimension: 3,
                rows: 1,
                columns: 1,
            },
            coefficient_class: BoundClass::bounded(2_u64.into()),
        })
        .unwrap();
        let right = PolynomialNF::bounded(MatrixBound {
            matrix_type: ConcreteMatrixType {
                modulus: 17.into(),
                ring_dimension: 3,
                rows: 1,
                columns: 1,
            },
            coefficient_class: BoundClass::bounded(3_u64.into()),
        })
        .unwrap();
        let tensor = left.tensor_nf(&right).unwrap();
        assert_eq!(
            tensor.bounded_summary().as_matrix_bound().unwrap().coefficient_class,
            BoundClass::bounded(18_u64.into())
        );
    }

    #[test]
    fn hash_provenance_and_regular_gadget_metadata_are_preserved() {
        let first = PolynomialNF::hash_plain_nf(
            FactorIdentity::named("query"),
            &[finite("arg", 1, 1, 1)],
            matrix_type(1, 1),
            None,
        )
        .unwrap();
        let second = PolynomialNF::hash_plain_nf(
            FactorIdentity::named("query"),
            &[finite("arg", 1, 1, 2)],
            matrix_type(1, 1),
            None,
        )
        .unwrap();
        assert_eq!(first.exact_terms().keys().next(), second.exact_terms().keys().next());
        let gadget = PolynomialNF::matrix_constant_nf(
            FactorIdentity::named("gadget"),
            &ResolvedMatrixConstant::Gadget { base: 2.into(), small: false },
            matrix_type(1, 1),
        )
        .unwrap();
        let gadget_factors = gadget.exact_terms().values().next().unwrap().monomial.factors();
        let factor = gadget_factors.first().unwrap();
        assert!(matches!(factor.bound, BoundClass::Large));
        assert!(factor.matrix_bound.is_some());
    }

    #[test]
    fn explicit_polynomial_records_support_without_underestimating() {
        let nf = PolynomialNF::matrix_constant_nf(
            FactorIdentity::named("polynomial"),
            &ResolvedMatrixConstant::Polynomial {
                coefficients: vec![1.into(), 0.into(), (-2).into(), 0.into()].into(),
            },
            ConcreteMatrixType { modulus: 17.into(), ring_dimension: 4, rows: 1, columns: 1 },
        )
        .unwrap();
        let factor = nf.exact_terms().values().next().unwrap().monomial.factors()[0].clone();
        assert!(factor.is_central_scalar());
        assert_eq!(factor.matrix_value_metadata.polynomial.unwrap().support_upper, 2);

        let monomial = PolynomialNF::matrix_constant_nf(
            FactorIdentity::named("monomial"),
            &ResolvedMatrixConstant::Polynomial {
                coefficients: vec![0.into(), 2.into(), 0.into(), 0.into()].into(),
            },
            ConcreteMatrixType { modulus: 17.into(), ring_dimension: 4, rows: 1, columns: 1 },
        )
        .unwrap();
        let monomial_factor =
            monomial.exact_terms().values().next().unwrap().monomial.factors()[0].clone();
        assert!(monomial_factor.is_central_scalar());
        assert!(!monomial_factor.matrix_value_metadata.is_constant_polynomial);
        assert_eq!(monomial_factor.polynomial_facts.support_upper, 1);

        let constant = PolynomialNF::matrix_constant_nf(
            FactorIdentity::named("constant"),
            &ResolvedMatrixConstant::Polynomial {
                coefficients: vec![3.into(), 0.into(), 0.into(), 0.into()].into(),
            },
            ConcreteMatrixType { modulus: 17.into(), ring_dimension: 4, rows: 1, columns: 1 },
        )
        .unwrap();
        let constant_factor =
            constant.exact_terms().values().next().unwrap().monomial.factors()[0].clone();
        assert!(constant_factor.matrix_value_metadata.is_constant_polynomial);
        assert_eq!(constant_factor.polynomial_facts.support_upper, 1);
        assert_eq!(
            constant_factor.key,
            FactorIdentity::constant_polynomial(
                &ConcreteMatrixType { modulus: 17.into(), ring_dimension: 4, rows: 1, columns: 1 },
                &3.into(),
            )
        );

        let centered = PolynomialNF::matrix_constant_nf(
            FactorIdentity::named("centered"),
            &ResolvedMatrixConstant::Polynomial {
                coefficients: vec![20.into(), 0.into(), 0.into(), 0.into()].into(),
            },
            ConcreteMatrixType { modulus: 17.into(), ring_dimension: 4, rows: 1, columns: 1 },
        )
        .unwrap();
        let centered_factor =
            centered.exact_terms().values().next().unwrap().monomial.factors()[0].clone();
        assert_eq!(centered_factor.bound, BoundClass::bounded(3_u8.into()));
        assert_eq!(
            centered_factor.key,
            FactorIdentity::constant_polynomial(
                &ConcreteMatrixType { modulus: 17.into(), ring_dimension: 4, rows: 1, columns: 1 },
                &3.into(),
            )
        );
        let centered_zero = PolynomialNF::matrix_constant_nf(
            FactorIdentity::named("centered-zero"),
            &ResolvedMatrixConstant::Polynomial {
                coefficients: vec![17.into(), 0.into(), 0.into(), 0.into()].into(),
            },
            ConcreteMatrixType { modulus: 17.into(), ring_dimension: 4, rows: 1, columns: 1 },
        )
        .unwrap();
        assert!(centered_zero.is_exact_zero());
        let centered_one = PolynomialNF::matrix_constant_nf(
            FactorIdentity::named("centered-one"),
            &ResolvedMatrixConstant::Polynomial {
                coefficients: vec![18.into(), 0.into(), 0.into(), 0.into()].into(),
            },
            ConcreteMatrixType { modulus: 17.into(), ring_dimension: 4, rows: 1, columns: 1 },
        )
        .unwrap();
        assert_eq!(centered_one, PolynomialNF::one());

        let too_wide = PolynomialNF::matrix_constant_nf(
            FactorIdentity::named("too-wide"),
            &ResolvedMatrixConstant::Polynomial {
                coefficients: vec![1.into(), 1.into(), 1.into(), 1.into(), 1.into()].into(),
            },
            ConcreteMatrixType { modulus: 17.into(), ring_dimension: 4, rows: 1, columns: 1 },
        );
        assert_eq!(too_wide, Err(OperationError::InvalidConstant));
    }
}
