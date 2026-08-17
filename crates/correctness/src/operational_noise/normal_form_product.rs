//! The sole ordered-product constructor for the egg-independent normal form.
//!
//! This is deliberately a child of `normal_form`: it may use the storage
//! invariants of `PolynomialNF`, while sibling operation modules cannot reach
//! those fields or a raw product implementation.  Operations use the
//! crate-visible bound-only entry point below; the DAG normalizer uses the
//! parent-visible entry point and performs relation fixed-point processing at
//! its owning normalization context.

use super::{
    BoundClass, BoundedSummary, ExpressionDag, ExpressionNode, FullRelationKey, MatrixBound,
    Monomial, NormalFormError, PolynomialNF, RelationRegistry, TermId, add_summary,
    dag_structure_fingerprint, monomial_bound, scale_by_multiplicity, summary_from_bound,
    switch_normalize,
};
use num_bigint::{BigInt, BigUint};
use std::collections::{BTreeMap, BTreeSet};

/// Owns every piece of state needed for one deterministic normalization job.
/// In particular, relation matching never consults a second product cache or
/// an operation-local registry.
pub(crate) struct Normalizer<'a> {
    dag: &'a ExpressionDag,
    registry: &'a RelationRegistry,
    memo: BTreeMap<TermId, PolynomialNF>,
    visiting: BTreeSet<TermId>,
    relation_stack: Vec<FullRelationKey>,
}

impl<'a> Normalizer<'a> {
    pub(crate) fn new(dag: &'a ExpressionDag, registry: &'a RelationRegistry) -> Self {
        Self {
            dag,
            registry,
            memo: BTreeMap::new(),
            visiting: BTreeSet::new(),
            relation_stack: Vec::new(),
        }
    }

    pub(crate) fn normalize(&mut self, root: TermId) -> Result<PolynomialNF, NormalFormError> {
        self.normalize_term(root)
    }

    fn normalize_term(&mut self, id: TermId) -> Result<PolynomialNF, NormalFormError> {
        if let Some(value) = self.memo.get(&id) {
            return Ok(value.clone());
        }
        if !self.visiting.insert(id) {
            return Err(NormalFormError::CyclicExpression { term: id });
        }
        let result = match self.dag.node(id)?.clone() {
            ExpressionNode::Zero => PolynomialNF::zero(),
            ExpressionNode::Atom(mut factor) => {
                factor.relation_live = self.registry.reaches_preimage(&factor.key);
                if matches!(factor.bound, BoundClass::ExactZero) {
                    PolynomialNF::zero()
                } else {
                    PolynomialNF::from_monomial(Monomial::from_factor(factor))
                }
            }
            ExpressionNode::Add(children) => {
                let mut value = PolynomialNF::zero();
                for child in children {
                    value = value.add(self.normalize_term(child)?)?;
                }
                value
            }
            ExpressionNode::Negate(child) => self.normalize_term(child)?.negate(),
            ExpressionNode::Product(children) => {
                let mut value = PolynomialNF::one();
                for child in children {
                    let child_value = self.normalize_term(child)?;
                    value = self.product_and_normalize(value, child_value)?;
                }
                value
            }
            ExpressionNode::Switch { selector, cases, reachable } |
            ExpressionNode::Select { selector, cases, reachable } => {
                if reachable.is_empty() || reachable.iter().any(|index| *index >= cases.len()) {
                    return Err(NormalFormError::InvalidSwitchReachability);
                }
                let mut seen = BTreeSet::new();
                if reachable.iter().any(|index| !seen.insert(*index)) {
                    return Err(NormalFormError::AmbiguousSwitchMapping);
                }
                let normalized = reachable
                    .iter()
                    .map(|index| self.normalize_term(cases[*index]))
                    .collect::<Result<Vec<_>, _>>()?;
                let fingerprints = reachable
                    .iter()
                    .map(|index| {
                        dag_structure_fingerprint(self.dag, cases[*index], &mut BTreeSet::new())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                switch_normalize(
                    selector,
                    normalized.into_boxed_slice(),
                    reachable.iter().map(|index| BigUint::from(*index)).collect(),
                    fingerprints.into_iter().map(Into::into).collect(),
                )?
            }
            ExpressionNode::FamilyGetStatic { cases, index } => {
                let Some(case) = cases.get(index) else {
                    return Err(NormalFormError::InvalidFamilyIndex);
                };
                self.normalize_term(*case)?
            }
            ExpressionNode::FamilyGetDynamic { selector, cases, stored_indices, domain_upper } => {
                if cases.is_empty() ||
                    stored_indices.len() != cases.len() ||
                    domain_upper > BigUint::from(cases.len()) ||
                    stored_indices.iter().any(|index| index >= &domain_upper) ||
                    stored_indices.iter().collect::<BTreeSet<_>>().len() != stored_indices.len()
                {
                    return Err(NormalFormError::InvalidFamilyDomain);
                }
                let normalized = cases
                    .iter()
                    .map(|case| self.normalize_term(*case))
                    .collect::<Result<Vec<_>, _>>()?;
                let fingerprints = cases
                    .iter()
                    .map(|case| dag_structure_fingerprint(self.dag, *case, &mut BTreeSet::new()))
                    .collect::<Result<Vec<_>, _>>()?;
                switch_normalize(
                    selector,
                    normalized.into_boxed_slice(),
                    stored_indices,
                    fingerprints.into_iter().map(Into::into).collect(),
                )?
            }
        };
        self.visiting.remove(&id);
        self.memo.insert(id, result.clone());
        Ok(result)
    }

    pub(crate) fn product_and_normalize(
        &mut self,
        left: PolynomialNF,
        right: PolynomialNF,
    ) -> Result<PolynomialNF, NormalFormError> {
        let value = product_bound_only(left, right)?;
        let value = self.expose_single_switch_products(value)?;
        self.apply_relations(value)
    }

    /// Distributes only a single selector barrier when an ordered factor is
    /// adjacent to it. This is the finite casewise step that exposes
    /// `B*Switch(s,K_i)` as `Switch(s,B*K_i)`; independent selectors remain
    /// barriers and are never Cartesian-expanded.
    fn expose_single_switch_products(
        &mut self,
        mut value: PolynomialNF,
    ) -> Result<PolynomialNF, NormalFormError> {
        let keys = value.exact_terms.keys().cloned().collect::<Vec<_>>();
        for key in keys {
            let Some(term) = value.exact_terms.remove(&key) else { continue };
            let switch_positions = term
                .monomial
                .factors
                .iter()
                .enumerate()
                .filter_map(|(index, factor)| factor.switch.as_ref().map(|_| index))
                .collect::<Vec<_>>();
            if switch_positions.len() == 1 && term.monomial.factors.len() == 1 {
                let switch = term.monomial.factors[0]
                    .switch
                    .as_ref()
                    .expect("switch position was collected from switch data")
                    .clone();
                let mut cases = Vec::with_capacity(switch.cases.len());
                let mut fingerprints = Vec::with_capacity(switch.cases.len());
                let mut changed_case = false;
                for switch_case in &switch.cases {
                    let exposed = self.expose_single_switch_products(switch_case.clone())?;
                    let case = self.apply_relations(exposed)?;
                    changed_case |= case != *switch_case;
                    fingerprints
                        .push(format!("case={:?}", case.exact_terms.keys()).into_boxed_str());
                    cases.push(case);
                }
                if changed_case {
                    let normalized_switch = switch_normalize(
                        switch.selector.clone(),
                        cases.into_boxed_slice(),
                        switch.case_indices.clone(),
                        fingerprints.into_boxed_slice(),
                    )?;
                    value = value.add(scale_nf(normalized_switch, &term.multiplicity)?)?;
                } else {
                    value.exact_terms.insert(key, term);
                }
                continue;
            }
            if switch_positions.len() != 1 {
                value.exact_terms.insert(key, term);
                continue;
            }
            let switch_position = switch_positions[0];
            let switch = term.monomial.factors[switch_position]
                .switch
                .as_ref()
                .expect("switch position was collected from switch data")
                .clone();
            let mut cases = Vec::with_capacity(switch.cases.len());
            let mut fingerprints = Vec::with_capacity(switch.cases.len());
            for switch_case in &switch.cases {
                let mut case = PolynomialNF::one();
                for factor in &term.monomial.factors[..switch_position] {
                    case = self.product_and_normalize(
                        case,
                        PolynomialNF::from_monomial(Monomial::from_factor(factor.clone())),
                    )?;
                }
                case = self.product_and_normalize(case, switch_case.clone())?;
                for factor in &term.monomial.factors[switch_position + 1..] {
                    case = self.product_and_normalize(
                        case,
                        PolynomialNF::from_monomial(Monomial::from_factor(factor.clone())),
                    )?;
                }
                cases.push(case.clone());
                fingerprints.push(format!("case={:?}", case.exact_terms.keys()).into_boxed_str());
            }
            let normalized_switch = switch_normalize(
                switch.selector.clone(),
                cases.into_boxed_slice(),
                switch.case_indices.clone(),
                fingerprints.into_boxed_slice(),
            )?;
            value = value.add(scale_nf(normalized_switch, &term.multiplicity)?)?;
        }
        Ok(value)
    }

    fn apply_relations(
        &mut self,
        mut value: PolynomialNF,
    ) -> Result<PolynomialNF, NormalFormError> {
        loop {
            let mut changed = false;
            let keys = value.exact_terms.keys().cloned().collect::<Vec<_>>();
            for key in keys {
                let Some(term) = value.exact_terms.remove(&key) else { continue };
                for position in 0..term.monomial.factors.len().saturating_sub(1) {
                    let left = &term.monomial.factors[position];
                    let right = &term.monomial.factors[position + 1];
                    let Some(registration) = self.registry.resolve(left, right)? else { continue };
                    if self.relation_stack.contains(&registration.key) {
                        return Err(NormalFormError::CyclicRelationDependency {
                            key: registration.key.clone(),
                        });
                    }
                    self.relation_stack.push(registration.key.clone());
                    if self.visiting.contains(&registration.target) {
                        return Err(NormalFormError::CyclicRelationDependency {
                            key: registration.key.clone(),
                        });
                    }
                    let target = self.normalize_term(registration.target)?;
                    let prefix = Monomial {
                        factors: term.monomial.factors[..position].to_vec().into_boxed_slice(),
                    };
                    let suffix = Monomial {
                        factors: term.monomial.factors[position + 2..].to_vec().into_boxed_slice(),
                    };
                    let mut expanded = PolynomialNF::zero();
                    for target_term in target.exact_terms.values() {
                        expanded.insert(
                            prefix.concat(&target_term.monomial).concat(&suffix),
                            &term.multiplicity * &target_term.multiplicity,
                        )?;
                    }
                    if let Some(summary) = target.bounded_summary.as_matrix_bound() {
                        expanded.bounded_summary =
                            reconnect_summary(summary, &prefix, &suffix, &term.multiplicity)?;
                    }
                    expanded = self.apply_relations(expanded)?;
                    expanded.fold_finite_non_live_terms()?;
                    self.relation_stack.pop();
                    value = value.add(expanded)?;
                    changed = true;
                    break;
                }
                if !changed {
                    value.exact_terms.insert(key, term);
                } else {
                    break;
                }
            }
            if !changed {
                return Ok(value);
            }
        }
    }
}

fn reconnect_summary(
    target: &MatrixBound,
    prefix: &Monomial,
    suffix: &Monomial,
    multiplicity: &BigInt,
) -> Result<BoundedSummary, NormalFormError> {
    let mut bound = target.clone();
    if !prefix.factors.is_empty() {
        bound = super::product_bound(&monomial_bound(prefix)?, &bound)
            .map_err(NormalFormError::bound)?;
    }
    if !suffix.factors.is_empty() {
        bound = super::product_bound(&bound, &monomial_bound(suffix)?)
            .map_err(NormalFormError::bound)?;
    }
    Ok(summary_from_bound(scale_by_multiplicity(bound, multiplicity)))
}

/// Product used by `ExpressionDag` after all children have been normalized.
///
/// Relation application remains owned by the DAG normalizer because only it
/// has the registry, memo, and active relation-key stack.  This constructor
/// performs the deterministic finite distribution and Switch handling that
/// every product must share.
pub(super) fn product(
    left: PolynomialNF,
    right: PolynomialNF,
) -> Result<PolynomialNF, NormalFormError> {
    product_bound_only(left, right)
}

/// Product entry point for operation/family/recurrence transfers.  The
/// operation layer is intentionally unable to call a `PolynomialNF::product`
/// method: all products pass through this constructor and therefore preserve
/// ordered factors, same-selector casewise combination, and bounded folding.
pub(crate) fn product_bound_only(
    left: PolynomialNF,
    right: PolynomialNF,
) -> Result<PolynomialNF, NormalFormError> {
    if left.is_exact_zero() || right.is_exact_zero() {
        return Ok(PolynomialNF::zero());
    }
    let mut out = PolynomialNF::zero();
    for left_term in left.exact_terms.values() {
        for right_term in right.exact_terms.values() {
            let monomial = left_term.monomial.concat(&right_term.monomial);
            let multiplicity = &left_term.multiplicity * &right_term.multiplicity;
            if let Some(casewise) = super::combine_same_selector_switches(&monomial)? {
                out = out.add(scale_nf(casewise, &multiplicity)?)?;
                continue;
            }
            if monomial.factors().iter().all(|factor| {
                !factor.relation_live &&
                    !matches!(factor.bound, BoundClass::Large) &&
                    factor.switch.is_none()
            }) {
                let bound = monomial_bound(&monomial)?;
                out.bounded_summary = add_summary(
                    out.bounded_summary,
                    BoundedSummary::Bounded(scale_by_multiplicity(bound, &multiplicity)),
                )?;
            } else {
                out.insert(monomial, multiplicity)?;
            }
        }
    }
    if !left.bounded_summary.is_exact_zero() {
        out = out.multiply_summary(&left.bounded_summary, &right.exact_terms)?;
    }
    if !right.bounded_summary.is_exact_zero() {
        out = out.multiply_summary(&right.bounded_summary, &left.exact_terms)?;
    }
    if !left.bounded_summary.is_exact_zero() && !right.bounded_summary.is_exact_zero() {
        out.bounded_summary = add_summary(
            out.bounded_summary,
            product_summary(left.bounded_summary, right.bounded_summary)?,
        )?;
    }
    Ok(out)
}

fn product_summary(
    left: BoundedSummary,
    right: BoundedSummary,
) -> Result<BoundedSummary, NormalFormError> {
    match (left, right) {
        (BoundedSummary::ExactZero, _) | (_, BoundedSummary::ExactZero) => {
            Ok(BoundedSummary::ExactZero)
        }
        (BoundedSummary::Bounded(left), BoundedSummary::Bounded(right)) => {
            Ok(BoundedSummary::Bounded(
                super::product_bound(&left, &right).map_err(NormalFormError::bound)?,
            ))
        }
    }
}

fn scale_nf(
    mut value: PolynomialNF,
    multiplicity: &num_bigint::BigInt,
) -> Result<PolynomialNF, NormalFormError> {
    for term in value.exact_terms.values_mut() {
        term.multiplicity *= multiplicity;
    }
    if let BoundedSummary::Bounded(bound) = &value.bounded_summary {
        value.bounded_summary =
            summary_from_bound(scale_by_multiplicity(bound.clone(), multiplicity));
    }
    Ok(value)
}
