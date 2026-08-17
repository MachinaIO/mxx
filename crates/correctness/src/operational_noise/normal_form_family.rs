//! Compact family and sequential-recurrence transfers for `PolynomialNF`.
//!
//! A shared family owns one already-normalized representative and validated
//! binder domains.  Its logical cardinality is metadata; it is never turned
//! into a Cartesian list of cases.  Sequential recurrence evaluation keeps
//! only the current carried state and applies a fixed transition a checked
//! number of times.

use super::{
    analysis::{IntegerDomain, IntegerInterval},
    family::{CoverageBinderDomain, FamilyCoverageError, LoopDomainKey, shared_affine_maximum},
    normal_form::{BoundedSummary, NormalFormError, PolynomialNF, product_bound_only},
};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NormalFormFamilyStorage {
    ExactStored {
        elements: Box<[PolynomialNF]>,
    },
    SharedTemplate {
        domain: LoopDomainKey,
        representative: PolynomialNF,
        binder_domains: Box<[CoverageBinderDomain]>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NormalFormFamily {
    pub storage: NormalFormFamilyStorage,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FamilyNFError {
    Coverage(FamilyCoverageError),
    NormalForm(NormalFormError),
    InvalidCount,
    UnboundedFamily,
    InvalidMaximum,
}

impl From<FamilyCoverageError> for FamilyNFError {
    fn from(error: FamilyCoverageError) -> Self {
        Self::Coverage(error)
    }
}

impl From<NormalFormError> for FamilyNFError {
    fn from(error: NormalFormError) -> Self {
        Self::NormalForm(error)
    }
}

impl NormalFormFamily {
    pub fn validate(&self) -> Result<(), FamilyNFError> {
        match &self.storage {
            NormalFormFamilyStorage::ExactStored { elements } => {
                if elements.is_empty() {
                    return Err(FamilyCoverageError::EmptyExactStorage.into());
                }
            }
            NormalFormFamilyStorage::SharedTemplate { domain, binder_domains, .. } => {
                let mut owner_size = None;
                for binder in binder_domains {
                    if binder.maximum < binder.minimum {
                        return Err(FamilyCoverageError::InvalidBinderDomain {
                            minimum: binder.minimum.clone(),
                            maximum: binder.maximum.clone(),
                        }
                        .into());
                    }
                    let width = (&binder.maximum - &binder.minimum + BigInt::one())
                        .to_biguint()
                        .ok_or(FamilyNFError::InvalidCount)?;
                    if binder.binder == domain.binder {
                        owner_size = Some(width);
                    }
                }
                if owner_size.as_ref() != Some(&domain.logical_count) {
                    return Err(FamilyCoverageError::SharedCountMismatch {
                        count: domain.logical_count.clone(),
                        domain_size: owner_size.unwrap_or_else(BigUint::zero),
                    }
                    .into());
                }
            }
        }
        Ok(())
    }

    pub fn logical_count(&self) -> Result<BigUint, FamilyNFError> {
        self.validate()?;
        Ok(match &self.storage {
            NormalFormFamilyStorage::ExactStored { elements } => BigUint::from(elements.len()),
            NormalFormFamilyStorage::SharedTemplate { domain, .. } => domain.logical_count.clone(),
        })
    }

    /// Static access is physical-only; a shared template has no stored case to
    /// return and therefore cannot be silently enumerated.
    pub fn static_get(&self, index: usize) -> Result<Option<&PolynomialNF>, FamilyNFError> {
        self.validate()?;
        match &self.storage {
            NormalFormFamilyStorage::ExactStored { elements } => {
                elements.get(index).map(Some).ok_or_else(|| {
                    FamilyCoverageError::StaticIndexOutOfRange {
                        index: BigInt::from(index),
                        count: BigUint::from(elements.len()),
                    }
                    .into()
                })
            }
            NormalFormFamilyStorage::SharedTemplate { .. } => Ok(None),
        }
    }

    /// Returns the one representative used for a shared family.  The caller
    /// must bind an owner-aware index; this method never materializes cases.
    pub fn shared_template(
        &self,
    ) -> Result<Option<(&PolynomialNF, &LoopDomainKey, &[CoverageBinderDomain])>, FamilyNFError>
    {
        self.validate()?;
        Ok(match &self.storage {
            NormalFormFamilyStorage::ExactStored { .. } => None,
            NormalFormFamilyStorage::SharedTemplate { domain, representative, binder_domains } => {
                Some((representative, domain, binder_domains))
            }
        })
    }

    /// Normalizes a shared representative once, retaining its validated
    /// domains and logical count unchanged.
    pub fn normalize_shared(&mut self) -> Result<(), FamilyNFError> {
        if let NormalFormFamilyStorage::SharedTemplate { representative, .. } = &mut self.storage {
            *representative = representative.clone().finish_relation_live()?;
        }
        self.validate()
    }

    /// Computes a maximum over stored physical cases, or returns the single
    /// shared representative bound.  No logical family expansion occurs.
    pub fn maximum_bound(&self) -> Result<BoundedSummary, FamilyNFError> {
        self.validate()?;
        match &self.storage {
            NormalFormFamilyStorage::ExactStored { elements } => {
                let mut result = BoundedSummary::ExactZero;
                for element in elements {
                    result = maximum_summary(result, element.validate_bounded_only()?.clone())?;
                }
                Ok(result)
            }
            NormalFormFamilyStorage::SharedTemplate { representative, .. } => {
                Ok(representative.validate_bounded_only()?.clone())
            }
        }
    }

    /// Uses the existing affine endpoint rule for a shared selector domain.
    /// The returned value is a bound on the selector expression, not a family
    /// case list.
    pub fn shared_selector_maximum(
        &self,
        selector_domain: &IntegerDomain,
    ) -> Result<BigInt, FamilyNFError> {
        let Some((_, _, binder_domains)) = self.shared_template()? else {
            return Err(FamilyNFError::Coverage(FamilyCoverageError::StorageMismatch));
        };
        Ok(shared_affine_maximum(selector_domain, binder_domains)?)
    }
}

fn maximum_summary(
    current: BoundedSummary,
    candidate: BoundedSummary,
) -> Result<BoundedSummary, FamilyNFError> {
    match (current, candidate) {
        (BoundedSummary::ExactZero, value) | (value, BoundedSummary::ExactZero) => Ok(value),
        (BoundedSummary::Bounded(mut current), BoundedSummary::Bounded(candidate)) => {
            if current.matrix_type != candidate.matrix_type {
                return Err(FamilyNFError::InvalidMaximum);
            }
            current.coefficient_class =
                match (current.coefficient_class, candidate.coefficient_class) {
                    (super::bound::BoundClass::ExactZero, value) |
                    (value, super::bound::BoundClass::ExactZero) => value,
                    (
                        super::bound::BoundClass::Bounded { maximum_absolute_coefficient: left },
                        super::bound::BoundClass::Bounded { maximum_absolute_coefficient: right },
                    ) => super::bound::BoundClass::bounded(left.max(right)),
                    _ => return Err(FamilyNFError::UnboundedFamily),
                };
            Ok(BoundedSummary::Bounded(current))
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RecurrenceExpr {
    Constant(PolynomialNF),
    Previous(usize),
    Add(Box<Self>, Box<Self>),
    Multiply(Box<Self>, Box<Self>),
    Max(Box<[Self]>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SequentialRecurrence {
    pub initial: Box<[PolynomialNF]>,
    pub transition: Box<[RecurrenceExpr]>,
    pub count: IntegerDomain,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RecurrenceError {
    ArityMismatch { expected: usize, actual: usize },
    PreviousOutOfRange { index: usize, state_size: usize },
    InvalidCount,
    MissingDomain,
    NormalForm(NormalFormError),
}

impl From<NormalFormError> for RecurrenceError {
    fn from(error: NormalFormError) -> Self {
        Self::NormalForm(error)
    }
}

impl SequentialRecurrence {
    pub fn evaluate(&self) -> Result<Box<[PolynomialNF]>, RecurrenceError> {
        self.evaluate_with_product(|left, right| product_bound_only(left, right))
    }

    /// Evaluate with an explicit product policy. The default path is
    /// deliberately bound-only; a semantic caller supplies the job-owned
    /// normalizer so relation targets are handled at the same boundary as
    /// DAG products.
    pub fn evaluate_with_product<F>(
        &self,
        mut product: F,
    ) -> Result<Box<[PolynomialNF]>, RecurrenceError>
    where
        F: FnMut(PolynomialNF, PolynomialNF) -> Result<PolynomialNF, NormalFormError>,
    {
        if self.initial.len() != self.transition.len() {
            return Err(RecurrenceError::ArityMismatch {
                expected: self.initial.len(),
                actual: self.transition.len(),
            });
        }
        let count = recurrence_count(&self.count)?;
        let count = count.to_usize().ok_or(RecurrenceError::InvalidCount)?;
        let mut state = self.initial.to_vec();
        for _ in 0..count {
            let next = self
                .transition
                .iter()
                .map(|expression| evaluate_expression(expression, &state, &mut product))
                .collect::<Result<Vec<_>, _>>()?;
            state = next;
        }
        Ok(state.into_boxed_slice())
    }
}

fn recurrence_count(domain: &IntegerDomain) -> Result<BigUint, RecurrenceError> {
    match domain {
        IntegerDomain::Exact(value) if value.sign() != num_bigint::Sign::Minus => {
            value.to_biguint().ok_or(RecurrenceError::InvalidCount)
        }
        IntegerDomain::IntervalOnly(IntegerInterval { minimum, maximum })
            if minimum == maximum && minimum.sign() != num_bigint::Sign::Minus =>
        {
            minimum.to_biguint().ok_or(RecurrenceError::InvalidCount)
        }
        IntegerDomain::Affine { .. } => Err(RecurrenceError::MissingDomain),
        _ => Err(RecurrenceError::InvalidCount),
    }
}

fn evaluate_expression(
    expression: &RecurrenceExpr,
    state: &[PolynomialNF],
    product: &mut impl FnMut(PolynomialNF, PolynomialNF) -> Result<PolynomialNF, NormalFormError>,
) -> Result<PolynomialNF, RecurrenceError> {
    match expression {
        RecurrenceExpr::Constant(value) => Ok(value.clone()),
        RecurrenceExpr::Previous(index) => state
            .get(*index)
            .cloned()
            .ok_or(RecurrenceError::PreviousOutOfRange { index: *index, state_size: state.len() }),
        RecurrenceExpr::Add(left, right) => Ok(evaluate_expression(left, state, product)?
            .add(evaluate_expression(right, state, product)?)?),
        RecurrenceExpr::Multiply(left, right) => {
            let left = evaluate_expression(left, state, product)?;
            let right = evaluate_expression(right, state, product)?;
            Ok(product(left, right)?)
        }
        RecurrenceExpr::Max(children) => {
            let mut result = BoundedSummary::ExactZero;
            let mut representative = None;
            for child in children.iter() {
                let value = evaluate_expression(child, state, product)?;
                if representative.is_none() {
                    representative = Some(value.clone());
                }
                result = maximum_summary(result, value.validate_bounded_only()?.clone()).map_err(
                    |error| match error {
                        FamilyNFError::NormalForm(error) => RecurrenceError::NormalForm(error),
                        _ => RecurrenceError::NormalForm(NormalFormError::BoundArithmetic),
                    },
                )?;
            }
            match (representative, result) {
                (Some(representative), summary) => {
                    if !representative.exact_terms().is_empty() {
                        return Ok(representative);
                    }
                    Ok(PolynomialNF::from_parts(Default::default(), summary))
                }
                (None, _) => Ok(PolynomialNF::zero()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::{
        bound::{BoundClass, MatrixBound, MatrixMetadata},
        family::CoverageBinderDomain,
        identity::{BinderKey, OccurrenceScope, ProgramKey},
        normal_form::{
            ExpressionDag, ExpressionNode, FactorIdentity, FullRelationKey, RelationRegistration,
            RelationRegistry, SymbolicFactor, normal_form_product::Normalizer,
        },
    };
    use mxx_ir_core::NodeId;

    fn matrix_bound(value: u64) -> MatrixBound {
        MatrixBound {
            matrix_type: mxx_ir_core::types::ConcreteMatrixType {
                modulus: 17.into(),
                ring_dimension: 1,
                rows: 1,
                columns: 1,
            },
            coefficient_class: BoundClass::bounded(value.into()),
            metadata: MatrixMetadata::unknown(),
        }
    }

    fn binder(node: u32) -> BinderKey {
        BinderKey {
            loop_scope: OccurrenceScope {
                program: ProgramKey::Ideal,
                definition: mxx_ir_core::FrozenGraphScopeId::Root,
                path: Box::new([]),
            },
            loop_node: NodeId(node.into()),
            slot: 0,
        }
    }

    #[test]
    fn shared_template_keeps_30720_logical_cases_compact() {
        let owner = binder(1);
        let family = NormalFormFamily {
            storage: NormalFormFamilyStorage::SharedTemplate {
                domain: LoopDomainKey { binder: owner.clone(), logical_count: 30_720_u64.into() },
                representative: PolynomialNF::bounded(matrix_bound(4)).unwrap(),
                binder_domains: vec![CoverageBinderDomain {
                    binder: owner,
                    minimum: 0.into(),
                    maximum: 30_719.into(),
                }]
                .into(),
            },
        };
        assert_eq!(family.logical_count().unwrap(), BigUint::from(30_720_u64));
        assert!(family.shared_template().unwrap().is_some());
        assert_eq!(
            family.maximum_bound().unwrap().as_matrix_bound().unwrap().coefficient_class,
            BoundClass::bounded(4_u64.into())
        );
        let mut normalized = family.clone();
        normalized.normalize_shared().unwrap();
        assert!(normalized.shared_template().unwrap().is_some());
    }

    #[test]
    fn exact_storage_supports_physical_static_access_only() {
        let family = NormalFormFamily {
            storage: NormalFormFamilyStorage::ExactStored {
                elements: vec![PolynomialNF::bounded(matrix_bound(1)).unwrap()].into(),
            },
        };
        assert!(family.static_get(0).unwrap().is_some());
        assert!(matches!(
            family.static_get(1),
            Err(FamilyNFError::Coverage(FamilyCoverageError::StaticIndexOutOfRange { .. }))
        ));
        assert_eq!(family.logical_count().unwrap(), BigUint::from(1_u8));
    }

    #[test]
    fn shared_affine_endpoint_rule_uses_sign_endpoints() {
        let outer = binder(1);
        let inner = binder(2);
        let family = NormalFormFamily {
            storage: NormalFormFamilyStorage::SharedTemplate {
                domain: LoopDomainKey { binder: outer.clone(), logical_count: 5_u64.into() },
                representative: PolynomialNF::bounded(matrix_bound(1)).unwrap(),
                binder_domains: vec![
                    CoverageBinderDomain {
                        binder: outer.clone(),
                        minimum: 0.into(),
                        maximum: 4.into(),
                    },
                    CoverageBinderDomain {
                        binder: inner.clone(),
                        minimum: 1.into(),
                        maximum: 6.into(),
                    },
                ]
                .into(),
            },
        };
        let domain = IntegerDomain::Affine {
            constant: 5.into(),
            coefficients: std::collections::BTreeMap::from([
                (outer, 3.into()),
                (inner, (-2).into()),
            ]),
            binders: std::collections::BTreeMap::from([
                (binder(1), IntegerInterval::new(0.into(), 4.into()).unwrap()),
                (binder(2), IntegerInterval::new(1.into(), 6.into()).unwrap()),
            ]),
        };
        assert_eq!(family.shared_selector_maximum(&domain).unwrap(), BigInt::from(15));
    }

    #[test]
    fn sequential_recurrence_handles_zero_one_and_n_steps_without_history() {
        let initial = PolynomialNF::bounded(matrix_bound(1)).unwrap();
        let transition = RecurrenceExpr::Add(
            Box::new(RecurrenceExpr::Previous(0)),
            Box::new(RecurrenceExpr::Constant(PolynomialNF::bounded(matrix_bound(2)).unwrap())),
        );
        for (count, expected) in [(0_u64, 1_u64), (1, 3), (4, 9)] {
            let recurrence = SequentialRecurrence {
                initial: vec![initial.clone()].into(),
                transition: vec![transition.clone()].into(),
                count: IntegerDomain::Exact(count.into()),
            };
            assert_eq!(
                recurrence.evaluate().unwrap()[0]
                    .bounded_summary()
                    .as_matrix_bound()
                    .unwrap()
                    .coefficient_class,
                BoundClass::bounded(expected.into())
            );
        }
    }

    #[test]
    fn recurrence_rejects_missing_or_negative_count_domain() {
        let recurrence = SequentialRecurrence {
            initial: vec![PolynomialNF::bounded(matrix_bound(1)).unwrap()].into(),
            transition: vec![RecurrenceExpr::Previous(0)].into(),
            count: IntegerDomain::Affine {
                constant: 0.into(),
                coefficients: Default::default(),
                binders: Default::default(),
            },
        };
        assert_eq!(recurrence.evaluate(), Err(RecurrenceError::MissingDomain));
        let negative =
            SequentialRecurrence { count: IntegerDomain::Exact((-1).into()), ..recurrence };
        assert_eq!(negative.evaluate(), Err(RecurrenceError::InvalidCount));
    }

    #[test]
    fn recurrence_multiply_and_max_use_current_state_only() {
        let recurrence = SequentialRecurrence {
            initial: vec![PolynomialNF::bounded(matrix_bound(2)).unwrap()].into(),
            transition: vec![RecurrenceExpr::Max(
                vec![
                    RecurrenceExpr::Multiply(
                        Box::new(RecurrenceExpr::Previous(0)),
                        Box::new(RecurrenceExpr::Constant(
                            PolynomialNF::bounded(matrix_bound(2)).unwrap(),
                        )),
                    ),
                    RecurrenceExpr::Constant(PolynomialNF::bounded(matrix_bound(3)).unwrap()),
                ]
                .into(),
            )]
            .into(),
            count: IntegerDomain::Exact(1.into()),
        };
        let result = recurrence.evaluate().unwrap();
        assert_eq!(
            result[0].bounded_summary().as_matrix_bound().unwrap().coefficient_class,
            BoundClass::bounded(4_u64.into())
        );
    }

    #[test]
    fn recurrence_product_closure_applies_registered_relation() {
        let public = FactorIdentity::named("B");
        let preimage = FactorIdentity::named("K");
        let target = FactorIdentity::named("P");
        let mut dag = ExpressionDag::new();
        let target_term =
            dag.push(ExpressionNode::Atom(SymbolicFactor::large(target.clone()))).unwrap();
        let mut registry = RelationRegistry::default();
        registry
            .register(RelationRegistration {
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
        let k = PolynomialNF::relation_live_factor(preimage, matrix_bound(1)).unwrap();
        let recurrence = SequentialRecurrence {
            initial: vec![PolynomialNF::exact_factor(public)].into(),
            transition: vec![RecurrenceExpr::Multiply(
                Box::new(RecurrenceExpr::Previous(0)),
                Box::new(RecurrenceExpr::Constant(k)),
            )]
            .into(),
            count: IntegerDomain::Exact(1.into()),
        };
        let mut normalizer = Normalizer::new(&dag, &registry);
        let result = recurrence
            .evaluate_with_product(|left, right| normalizer.product_and_normalize(left, right))
            .unwrap();
        assert_eq!(result[0].first_large_witness().unwrap().identity, target);
    }
}
