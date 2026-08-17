//! Compact TermId family helpers and sequential-recurrence transfers.
//!
//! Families retain only physical entries or one shared DAG representative.  A
//! shared logical domain is metadata and is never expanded into a Cartesian
//! list.  Sequential recurrences keep only the current carried state and
//! substitute the complete previous vector simultaneously at each step.

#[cfg(test)]
use super::normal_form::RelationRegistry;
use super::{
    family::{FamilyCoverageError, FamilyLoweringValue},
    identity::{BinderKey, ResolvedIntExpr},
    normal_form::{ExpressionDag, FactorIdentity, NormalFormError, TermId},
    scalar::{IntegerDomain, IntegerInterval, resolved_constant},
};
use num_bigint::{BigInt, BigUint};
use num_traits::ToPrimitive;

/// Matrix-family values are DAG terms, not scalar identifiers.
pub(crate) type MatrixTermFamily = FamilyLoweringValue<TermId>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum FamilyNFError {
    Coverage(FamilyCoverageError),
}

impl From<FamilyCoverageError> for FamilyNFError {
    fn from(error: FamilyCoverageError) -> Self {
        Self::Coverage(error)
    }
}

/// Validate a matrix family before any selector or family operation observes it.
pub(crate) fn validate_matrix_term_family(family: &MatrixTermFamily) -> Result<(), FamilyNFError> {
    family.validate().map_err(FamilyNFError::Coverage)
}

/// Resolve a statically known index from physically stored matrix entries.
/// Shared templates deliberately return `None`: their representative must be
/// instantiated with the owner-aware binder rather than enumerated here.
pub(crate) fn static_matrix_term(
    family: &MatrixTermFamily,
    index: &ResolvedIntExpr,
) -> Result<Option<TermId>, FamilyNFError> {
    let super::family::FamilyCoverageStorage::ExactStored { elements } = &family.storage else {
        return Ok(None);
    };
    let Some(index) = resolved_constant(index) else { return Ok(None) };
    let index = index.clone();
    let Some(index) = index.to_usize() else {
        return Err(FamilyCoverageError::StaticIndexOutOfRange {
            index,
            count: BigUint::from(elements.len()),
        }
        .into());
    };
    elements.get(index).copied().map(Some).ok_or_else(|| {
        FamilyCoverageError::StaticIndexOutOfRange {
            index: BigInt::from(index),
            count: BigUint::from(elements.len()),
        }
        .into()
    })
}

/// A matrix recurrence represented entirely by the job-owned expression DAG.
/// `state_factors` are owner-resolved placeholders installed in the loop body;
/// every transition step substitutes the complete previous vector at once.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TermSequentialRecurrence {
    pub(crate) initial: Box<[TermId]>,
    pub(crate) transition: Box<[TermId]>,
    pub(crate) state_factors: Box<[FactorIdentity]>,
    /// The owner-resolved iteration binder, when the recurrence came from a
    /// graph loop.  Its concrete value is substituted before state substitution.
    pub(crate) iteration_binder: Option<BinderKey>,
    pub(crate) count: IntegerDomain,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum TermRecurrenceError {
    ArityMismatch { expected: usize, actual: usize },
    InvalidCount,
    MissingDomain,
    NormalForm(NormalFormError),
}

impl From<NormalFormError> for TermRecurrenceError {
    fn from(error: NormalFormError) -> Self {
        Self::NormalForm(error)
    }
}

impl TermSequentialRecurrence {
    pub(crate) fn evaluate(
        &self,
        dag: &mut ExpressionDag,
    ) -> Result<Box<[TermId]>, TermRecurrenceError> {
        if self.initial.len() != self.transition.len() ||
            self.initial.len() != self.state_factors.len()
        {
            return Err(TermRecurrenceError::ArityMismatch {
                expected: self.initial.len(),
                actual: self.transition.len().max(self.state_factors.len()),
            });
        }
        let count =
            recurrence_count(&self.count)?.to_usize().ok_or(TermRecurrenceError::InvalidCount)?;
        let mut state = self.initial.to_vec();
        for step in 0..count {
            let mut binder_memo = std::collections::BTreeMap::new();
            let transition = self
                .transition
                .iter()
                .map(|term| {
                    self.iteration_binder.as_ref().map_or(Ok(*term), |binder| {
                        dag.substitute_binder(
                            *term,
                            binder,
                            &ResolvedIntExpr::Const(step.into()),
                            &mut binder_memo,
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let replacements = self
                .state_factors
                .iter()
                .cloned()
                .zip(state.iter().copied())
                .collect::<std::collections::BTreeMap<_, _>>();
            let mut factor_memo = std::collections::BTreeMap::new();
            state = transition
                .iter()
                .map(|term| dag.substitute_factors(*term, &replacements, &mut factor_memo))
                .collect::<Result<Vec<_>, _>>()?;
        }
        Ok(state.into_boxed_slice())
    }

    #[cfg(test)]
    pub(crate) fn evaluate_normalized(
        &self,
        dag: &mut ExpressionDag,
        registry: &RelationRegistry,
    ) -> Result<Box<[super::normal_form::PolynomialNF]>, TermRecurrenceError> {
        self.evaluate(dag)?
            .iter()
            .map(|term| dag.normalize(*term, registry).map_err(Into::into))
            .collect::<Result<Vec<_>, _>>()
            .map(Vec::into_boxed_slice)
    }
}

fn recurrence_count(domain: &IntegerDomain) -> Result<BigUint, TermRecurrenceError> {
    match domain {
        IntegerDomain::Exact(value) if value.sign() != num_bigint::Sign::Minus => {
            value.to_biguint().ok_or(TermRecurrenceError::InvalidCount)
        }
        IntegerDomain::IntervalOnly(IntegerInterval { minimum, maximum, .. })
            if minimum == maximum && minimum.sign() != num_bigint::Sign::Minus =>
        {
            minimum.to_biguint().ok_or(TermRecurrenceError::InvalidCount)
        }
        IntegerDomain::Affine { .. } => Err(TermRecurrenceError::MissingDomain),
        _ => Err(TermRecurrenceError::InvalidCount),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::{
        bound::{BoundClass, MatrixBound},
        family::{CoverageBinderDomain, FamilyCoverageStorage, LoopDomainKey},
        identity::{OccurrenceScope, ProgramKey},
        normal_form::{
            ExpressionNode, FactorKind, FactorOwner, FullRelationKey, RelationPattern,
            RelationRegistration, SymbolicFactor,
        },
        scalar::ScalarSort,
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
        }
    }

    fn relation_matrix_bound(value: u64) -> MatrixBound {
        MatrixBound {
            matrix_type: mxx_ir_core::types::ConcreteMatrixType {
                modulus: 17.into(),
                ring_dimension: 1,
                rows: 2,
                columns: 2,
            },
            coefficient_class: BoundClass::bounded(value.into()),
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
    fn shared_family_validates_count_without_materializing_cases() {
        let owner = binder(1);
        let family = MatrixTermFamily {
            element_type: ScalarSort::Int,
            storage: FamilyCoverageStorage::SharedTemplate {
                domain: LoopDomainKey { binder: owner.clone(), logical_count: 30_720_u64.into() },
                representative: TermId(0),
                binder_domains: vec![CoverageBinderDomain {
                    binder: owner,
                    minimum: 0.into(),
                    maximum: 30_719.into(),
                }]
                .into(),
            },
        };
        validate_matrix_term_family(&family).unwrap();
        assert!(matches!(static_matrix_term(&family, &ResolvedIntExpr::Const(0.into())), Ok(None)));
    }

    #[test]
    fn static_matrix_term_rejects_index_above_stored_count() {
        let family = MatrixTermFamily {
            element_type: ScalarSort::Int,
            storage: FamilyCoverageStorage::ExactStored { elements: vec![TermId(0)].into() },
        };
        assert!(matches!(
            static_matrix_term(&family, &ResolvedIntExpr::Const(1.into())),
            Err(FamilyNFError::Coverage(FamilyCoverageError::StaticIndexOutOfRange { .. }))
        ));
    }

    #[test]
    fn shared_template_substitution_rebuilds_nested_dag_once() {
        let owner = binder(7);
        let mut key = FactorIdentity::named("nested");
        key.coordinates = vec![(owner.clone(), ResolvedIntExpr::Binder(owner.clone()))].into();
        let atom = SymbolicFactor::large(key);
        let mut dag = ExpressionDag::new();
        let leaf = dag.push(ExpressionNode::Atom(atom)).unwrap();
        let product = dag.push(ExpressionNode::Product(vec![leaf, leaf].into())).unwrap();
        let root = dag.push(ExpressionNode::Add(vec![product, leaf].into())).unwrap();
        let replacement = ResolvedIntExpr::Const(13.into());
        let mut memo = std::collections::BTreeMap::new();
        let rebound = dag.substitute_binder(root, &owner, &replacement, &mut memo).unwrap();
        let ExpressionNode::Add(children) = dag.node(rebound).unwrap() else {
            panic!("nested template lost its outer structure")
        };
        let ExpressionNode::Product(product) = dag.node(children[0]).unwrap() else {
            panic!("nested template lost its product")
        };
        for term in [product[0], product[1], children[1]] {
            let ExpressionNode::Atom(factor) = dag.node(term).unwrap() else {
                panic!("nested template did not rebuild an atom")
            };
            assert_eq!(factor.key.coordinates[0].1, replacement);
        }
        assert_eq!(memo.len(), 3, "shared sub-DAG should be substituted once per source node");
    }

    #[test]
    fn stored_dynamic_mapping_does_not_enumerate_unstored_indices() {
        let selector = FactorIdentity::named("selector");
        let mut dag = ExpressionDag::new();
        let first = dag
            .push(ExpressionNode::Atom(SymbolicFactor::large(FactorIdentity::named("first"))))
            .unwrap();
        let second = dag
            .push(ExpressionNode::Atom(SymbolicFactor::large(FactorIdentity::named("second"))))
            .unwrap();
        let dynamic = dag
            .push(ExpressionNode::FamilyGetDynamic {
                selector,
                cases: vec![first, second].into(),
                stored_indices: vec![1_u8.into(), 3_u8.into()].into(),
                domain_upper: 4_u8.into(),
            })
            .unwrap();
        let ExpressionNode::FamilyGetDynamic { cases, .. } = dag.node(dynamic).unwrap() else {
            panic!("dynamic access was not retained as a family barrier")
        };
        assert_eq!(cases.len(), 2);
        let normal = dag.normalize(dynamic, &RelationRegistry::default()).unwrap();
        assert_eq!(normal.exact_terms().len(), 1);
    }

    #[test]
    fn term_recurrence_zero_one_n_and_relation_are_dag_owned() {
        let state_key = FactorIdentity::named("state");
        let public = FactorIdentity::named("B");
        let preimage = FactorIdentity::named("K");
        let target = FactorIdentity::named("P");
        let mut dag = ExpressionDag::new();
        let initial = dag
            .push(ExpressionNode::Atom(
                SymbolicFactor::bounded(FactorIdentity::named("initial"), matrix_bound(1)).unwrap(),
            ))
            .unwrap();
        let state =
            dag.push(ExpressionNode::Atom(SymbolicFactor::large(state_key.clone()))).unwrap();
        let b = dag.push(ExpressionNode::Atom(SymbolicFactor::large(public.clone()))).unwrap();
        let k = dag
            .push(ExpressionNode::Atom(
                SymbolicFactor::relation_live(preimage.clone(), relation_matrix_bound(1)).unwrap(),
            ))
            .unwrap();
        let product = dag.push(ExpressionNode::Product(vec![b, k].into())).unwrap();
        let transition = dag.push(ExpressionNode::Add(vec![state, product].into())).unwrap();
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
                preimage,
                target: target_term,
            })
            .unwrap();
        let recurrence = TermSequentialRecurrence {
            initial: vec![initial].into(),
            transition: vec![transition].into(),
            state_factors: vec![state_key].into(),
            iteration_binder: None,
            count: IntegerDomain::Exact(1.into()),
        };
        let mut one_dag = dag.clone();
        let one = recurrence.evaluate_normalized(&mut one_dag, &registry).unwrap();
        assert_eq!(one[0].first_large_witness().unwrap().identity, target);

        let zero = TermSequentialRecurrence {
            count: IntegerDomain::Exact(0.into()),
            ..recurrence.clone()
        };
        let mut zero_dag = dag.clone();
        assert_eq!(zero.evaluate(&mut zero_dag).unwrap()[0], initial);

        let n = TermSequentialRecurrence { count: IntegerDomain::Exact(3.into()), ..recurrence };
        let mut n_dag = dag;
        assert_ne!(n.evaluate(&mut n_dag).unwrap()[0], initial);
    }

    #[test]
    fn term_recurrence_binds_each_iteration_before_state_substitution() {
        let owner = binder(19);
        let state_key = FactorIdentity::named("x");
        let mut error_key = FactorIdentity::named("e");
        error_key.coordinates =
            vec![(owner.clone(), ResolvedIntExpr::Binder(owner.clone()))].into();
        let mut dag = ExpressionDag::new();
        let initial = dag
            .push(ExpressionNode::Atom(
                SymbolicFactor::bounded(FactorIdentity::named("initial"), matrix_bound(1)).unwrap(),
            ))
            .unwrap();
        let state =
            dag.push(ExpressionNode::Atom(SymbolicFactor::large(state_key.clone()))).unwrap();
        let error =
            dag.push(ExpressionNode::Atom(SymbolicFactor::large(error_key.clone()))).unwrap();
        let negated = dag.push(ExpressionNode::Negate(state)).unwrap();
        let transition = dag.push(ExpressionNode::Add(vec![negated, error].into())).unwrap();
        let recurrence = TermSequentialRecurrence {
            initial: vec![initial].into(),
            transition: vec![transition].into(),
            state_factors: vec![state_key].into(),
            iteration_binder: Some(owner),
            count: IntegerDomain::Exact(2.into()),
        };
        let terms = recurrence.evaluate(&mut dag).unwrap();

        fn collect_error_coordinates(
            dag: &ExpressionDag,
            term: TermId,
            owner: &FactorOwner,
            kind: &FactorKind,
            coordinates: &mut std::collections::BTreeSet<Box<[(BinderKey, ResolvedIntExpr)]>>,
        ) {
            match dag.node(term).unwrap() {
                ExpressionNode::Atom(factor)
                    if &factor.key.owner == owner && &factor.key.kind == kind =>
                {
                    coordinates.insert(factor.key.coordinates.clone());
                }
                ExpressionNode::Add(children) | ExpressionNode::Product(children) => {
                    for child in children {
                        collect_error_coordinates(dag, *child, owner, kind, coordinates);
                    }
                }
                ExpressionNode::Negate(child) => {
                    collect_error_coordinates(dag, *child, owner, kind, coordinates);
                }
                _ => {}
            }
        }
        let mut coordinates = std::collections::BTreeSet::new();
        collect_error_coordinates(
            &dag,
            terms[0],
            &error_key.owner,
            &error_key.kind,
            &mut coordinates,
        );
        assert_eq!(coordinates.len(), 2, "e_0 and e_1 must remain distinct");
    }
}
