//! The sole ordered-product constructor for the typed normal form.
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
use crate::operational_noise::normal_form_ops::{
    AdditionalOperations, CrtRecompose, PolynomialNFOperations,
};
use num_bigint::{BigInt, BigUint};
use num_traits::Zero;
use std::collections::{BTreeMap, BTreeSet};

/// Work performed by one deterministic normalizer job.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct NormalizationCounters {
    pub(crate) nodes_processed: u64,
    pub(crate) exact_term_count: u64,
    pub(crate) bounded_fold_count: u64,
    pub(crate) relation_candidates: u64,
    pub(crate) relations_applied: u64,
    pub(crate) relations_remaining: u64,
    pub(crate) switch_cases_processed: u64,
}

/// Owns every piece of state needed for one deterministic normalization job.
/// In particular, relation matching never consults a second product cache or
/// an operation-local registry.
pub(crate) struct Normalizer<'a> {
    dag: &'a ExpressionDag,
    registry: &'a RelationRegistry,
    memo: BTreeMap<TermId, PolynomialNF>,
    visiting: BTreeSet<TermId>,
    relation_stack: Vec<FullRelationKey>,
    counters: NormalizationCounters,
}

impl<'a> Normalizer<'a> {
    pub(crate) fn new(dag: &'a ExpressionDag, registry: &'a RelationRegistry) -> Self {
        Self {
            dag,
            registry,
            memo: BTreeMap::new(),
            visiting: BTreeSet::new(),
            relation_stack: Vec::new(),
            counters: NormalizationCounters::default(),
        }
    }

    pub(crate) fn normalize(&mut self, root: TermId) -> Result<PolynomialNF, NormalFormError> {
        self.normalize_dispatch(root)
    }

    fn normalize_dispatch(&mut self, root: TermId) -> Result<PolynomialNF, NormalFormError> {
        if self.requires_iterative_walk(root)? {
            self.normalize_term_iterative(root)
        } else {
            self.normalize_term(root)
        }
    }

    fn requires_iterative_walk(&self, root: TermId) -> Result<bool, NormalFormError> {
        let mut work = vec![(root, 0usize)];
        let mut seen = BTreeSet::new();
        while let Some((id, depth)) = work.pop() {
            if !seen.insert(id) {
                continue;
            }
            if depth >= 512 {
                return Ok(true);
            }
            work.extend(self.dag.node(id)?.children().into_iter().map(|child| (child, depth + 1)));
        }
        Ok(false)
    }

    fn normalize_term_iterative(&mut self, root: TermId) -> Result<PolynomialNF, NormalFormError> {
        enum Visit {
            Enter(TermId),
            Exit(TermId),
        }
        let mut work = vec![Visit::Enter(root)];
        let mut values = BTreeMap::<TermId, PolynomialNF>::new();
        let mut visiting = BTreeSet::<TermId>::new();
        while let Some(visit) = work.pop() {
            let (id, exit) = match visit {
                Visit::Enter(id) => (id, false),
                Visit::Exit(id) => (id, true),
            };
            if self.memo.contains_key(&id) || values.contains_key(&id) {
                continue;
            }
            if !exit {
                if !visiting.insert(id) {
                    return Err(NormalFormError::CyclicExpression { term: id });
                }
                work.push(Visit::Exit(id));
                work.extend(self.dag.node(id)?.children().into_iter().rev().map(Visit::Enter));
                continue;
            }
            let child = |child: TermId,
                         values: &BTreeMap<TermId, PolynomialNF>,
                         memo: &BTreeMap<TermId, PolynomialNF>| {
                values
                    .get(&child)
                    .or_else(|| memo.get(&child))
                    .cloned()
                    .ok_or(NormalFormError::InvalidTermId)
            };
            self.counters.nodes_processed = self.counters.nodes_processed.saturating_add(1);
            let result = match self.dag.node(id)?.clone() {
                ExpressionNode::Zero => PolynomialNF::zero(),
                ExpressionNode::Atom(factor) => self.normalize_atom(factor),
                ExpressionNode::Add(children) => {
                    let mut value = PolynomialNF::zero();
                    for child_id in children {
                        value = value.add(child(child_id, &values, &self.memo)?)?;
                    }
                    value
                }
                ExpressionNode::Negate(child_id) => child(child_id, &values, &self.memo)?.negate(),
                ExpressionNode::Product(children) => {
                    let mut value = PolynomialNF::one();
                    for child_id in children {
                        value = self
                            .product_and_normalize(value, child(child_id, &values, &self.memo)?)?;
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
                    self.counters.switch_cases_processed =
                        self.counters.switch_cases_processed.saturating_add(reachable.len() as u64);
                    let normalized = reachable
                        .iter()
                        .map(|index| child(cases[*index], &values, &self.memo))
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
                        return Err(NormalFormError::InvalidFamilyIndex)
                    };
                    child(*case, &values, &self.memo)?
                }
                ExpressionNode::FamilyGetDynamic {
                    selector,
                    cases,
                    stored_indices,
                    domain_upper,
                } => {
                    if cases.is_empty() ||
                        stored_indices.len() != cases.len() ||
                        domain_upper.is_zero() ||
                        stored_indices.iter().any(|index| index >= &domain_upper) ||
                        stored_indices.iter().collect::<BTreeSet<_>>().len() !=
                            stored_indices.len() ||
                        stored_indices.iter().max().map(|index| index + BigUint::from(1_u8)) !=
                            Some(domain_upper.clone())
                    {
                        return Err(NormalFormError::InvalidFamilyDomain);
                    }
                    self.counters.switch_cases_processed =
                        self.counters.switch_cases_processed.saturating_add(cases.len() as u64);
                    let normalized = cases
                        .iter()
                        .map(|case| child(*case, &values, &self.memo))
                        .collect::<Result<Vec<_>, _>>()?;
                    let fingerprints = cases
                        .iter()
                        .map(|case| {
                            dag_structure_fingerprint(self.dag, *case, &mut BTreeSet::new())
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    switch_normalize(
                        selector,
                        normalized.into_boxed_slice(),
                        stored_indices,
                        fingerprints.into_iter().map(Into::into).collect(),
                    )?
                }
                ExpressionNode::MatrixScale { input, scalar } => child(input, &values, &self.memo)?
                    .matrix_scale_nf(scalar)
                    .map_err(operation_error)?,
                ExpressionNode::Transpose(input) => {
                    child(input, &values, &self.memo)?.transpose_nf().map_err(operation_error)?
                }
                ExpressionNode::Slice { input, spec } => {
                    child(input, &values, &self.memo)?.slice_nf(&spec).map_err(operation_error)?
                }
                ExpressionNode::Tensor { left, right } => child(left, &values, &self.memo)?
                    .tensor_nf(&child(right, &values, &self.memo)?)
                    .map_err(operation_error)?,
                ExpressionNode::LiftConstantPolynomial { input, matrix_type, domain } => {
                    child(input, &values, &self.memo)?
                        .lift_constant_polynomial_nf(matrix_type, &domain)
                        .map_err(operation_error)?
                }
                ExpressionNode::View { input, view, output_type } => {
                    child(input, &values, &self.memo)?
                        .view_nf(&view, output_type)
                        .map_err(operation_error)?
                }
                ExpressionNode::CrtRecompose { inputs, spec, output_type } => {
                    let inputs = inputs
                        .iter()
                        .map(|input| child(*input, &values, &self.memo))
                        .collect::<Result<Vec<_>, _>>()?;
                    PolynomialNF::crt_recompose_nf(&inputs, &spec, output_type)
                        .map_err(operation_error)?
                }
                ExpressionNode::Concat { inputs, axis, output_type } => {
                    let inputs = inputs
                        .iter()
                        .map(|input| child(*input, &values, &self.memo))
                        .collect::<Result<Vec<_>, _>>()?;
                    PolynomialNF::concat_nf(&inputs, axis, output_type).map_err(operation_error)?
                }
            };
            visiting.remove(&id);
            values.insert(id, result);
        }
        values
            .remove(&root)
            .or_else(|| self.memo.get(&root).cloned())
            .ok_or(NormalFormError::InvalidTermId)
    }

    pub(crate) fn normalize_with_counters(
        mut self,
        root: TermId,
    ) -> Result<(PolynomialNF, NormalizationCounters), NormalFormError> {
        let normalized = self.normalize(root)?;
        let normalized =
            normalized.finish_relation_live_counted(&mut self.counters.bounded_fold_count)?;
        self.counters.exact_term_count = normalized.exact_terms().len() as u64;
        self.counters.relations_remaining = self.remaining_relation_candidates(&normalized)?;
        Ok((normalized, self.counters))
    }

    fn normalize_term(&mut self, id: TermId) -> Result<PolynomialNF, NormalFormError> {
        if let Some(value) = self.memo.get(&id) {
            return Ok(value.clone());
        }
        if !self.visiting.insert(id) {
            return Err(NormalFormError::CyclicExpression { term: id });
        }
        self.counters.nodes_processed = self.counters.nodes_processed.saturating_add(1);
        let result = match self.dag.node(id)?.clone() {
            ExpressionNode::Zero => PolynomialNF::zero(),
            ExpressionNode::Atom(factor) => self.normalize_atom(factor),
            ExpressionNode::Add(children) => {
                let mut value = PolynomialNF::zero();
                for child in children {
                    value = value.add(self.normalize_term(child)?)?;
                }
                value
            }
            ExpressionNode::Negate(child) => {
                // `normalize_term` owns a large enum-dispatch frame.  Shared
                // DAG edges are already complete here, so re-entering that
                // frame for a memo hit needlessly consumes stack and makes a
                // shallow shared Add/Negate shape overflow the test thread's
                // fixed stack.  Keep the shared-child fast path explicit.
                let child = self
                    .memo
                    .get(&child)
                    .cloned()
                    .map(Ok)
                    .unwrap_or_else(|| self.normalize_term(child))?;
                child.negate()
            }
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
                self.counters.switch_cases_processed =
                    self.counters.switch_cases_processed.saturating_add(reachable.len() as u64);
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
                    domain_upper.is_zero() ||
                    stored_indices.iter().any(|index| index >= &domain_upper) ||
                    stored_indices.iter().collect::<BTreeSet<_>>().len() != stored_indices.len() ||
                    stored_indices.iter().max().map(|index| index + BigUint::from(1_u8)) !=
                        Some(domain_upper.clone())
                {
                    return Err(NormalFormError::InvalidFamilyDomain);
                }
                self.counters.switch_cases_processed =
                    self.counters.switch_cases_processed.saturating_add(cases.len() as u64);
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
            ExpressionNode::MatrixScale { input, scalar } => {
                self.normalize_term(input)?.matrix_scale_nf(scalar).map_err(operation_error)?
            }
            ExpressionNode::Transpose(input) => {
                self.normalize_term(input)?.transpose_nf().map_err(operation_error)?
            }
            ExpressionNode::Slice { input, spec } => {
                self.normalize_term(input)?.slice_nf(&spec).map_err(operation_error)?
            }
            ExpressionNode::Tensor { left, right } => self
                .normalize_term(left)?
                .tensor_nf(&self.normalize_term(right)?)
                .map_err(operation_error)?,
            ExpressionNode::LiftConstantPolynomial { input, matrix_type, domain } => self
                .normalize_term(input)?
                .lift_constant_polynomial_nf(matrix_type, &domain)
                .map_err(operation_error)?,
            ExpressionNode::View { input, view, output_type } => {
                self.normalize_term(input)?.view_nf(&view, output_type).map_err(operation_error)?
            }
            ExpressionNode::CrtRecompose { inputs, spec, output_type } => {
                let inputs = inputs
                    .iter()
                    .map(|input| self.normalize_term(*input))
                    .collect::<Result<Vec<_>, _>>()?;
                PolynomialNF::crt_recompose_nf(&inputs, &spec, output_type)
                    .map_err(operation_error)?
            }
            ExpressionNode::Concat { inputs, axis, output_type } => {
                let inputs = inputs
                    .iter()
                    .map(|input| self.normalize_term(*input))
                    .collect::<Result<Vec<_>, _>>()?;
                PolynomialNF::concat_nf(&inputs, axis, output_type).map_err(operation_error)?
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
        let value = product_bound_only_counted(left, right, &mut self.counters.bounded_fold_count)?;
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
                    // Count semantic relation candidates, not every adjacent
                    // factor boundary visited by a particular association of
                    // the same product.  The latter would make diagnostics
                    // depend on whether `A*(B*K)` was written as
                    // `(A*B)*K`, even though both normalize to the same
                    // canonical polynomial and perform the same checked
                    // replacement.
                    self.counters.relation_candidates =
                        self.counters.relation_candidates.saturating_add(1);
                    self.counters.relations_applied =
                        self.counters.relations_applied.saturating_add(1);
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
                    let target = match self.memo.get(&registration.target).cloned() {
                        Some(target) => target,
                        None => match self.dag.node(registration.target)?.clone() {
                            ExpressionNode::Atom(factor) => {
                                // Relation targets are commonly atoms.  They
                                // are complete without another recursive DAG
                                // dispatch frame; memoize the same canonical
                                // result and preserve node accounting.
                                self.counters.nodes_processed =
                                    self.counters.nodes_processed.saturating_add(1);
                                let target = self.normalize_atom(factor);
                                self.memo.insert(registration.target, target.clone());
                                target
                            }
                            _ => self.normalize_dispatch(registration.target)?,
                        },
                    };
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
                    expanded.fold_finite_non_live_terms(&mut self.counters.bounded_fold_count)?;
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

    fn normalize_atom(&self, mut factor: super::SymbolicFactor) -> PolynomialNF {
        factor.relation_live = self.registry.reaches_preimage(&factor.key);
        if matches!(factor.bound, BoundClass::ExactZero) {
            PolynomialNF::zero()
        } else {
            PolynomialNF::from_monomial(Monomial::from_factor(factor))
        }
    }

    fn remaining_relation_candidates(
        &self,
        normalized: &PolynomialNF,
    ) -> Result<u64, NormalFormError> {
        let mut remaining = 0_u64;
        for term in normalized.exact_terms().values() {
            for pair in term.monomial.factors().windows(2) {
                if self.registry.resolve(&pair[0], &pair[1])?.is_some() {
                    remaining = remaining.saturating_add(1);
                }
            }
        }
        Ok(remaining)
    }
}

fn operation_error(
    error: crate::operational_noise::normal_form_ops::OperationError,
) -> NormalFormError {
    match error {
        crate::operational_noise::normal_form_ops::OperationError::NormalForm(error) => error,
        _ => NormalFormError::BoundArithmetic,
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
    product_bound_only_inner(left, right, None)
}

fn product_bound_only_counted(
    left: PolynomialNF,
    right: PolynomialNF,
    fold_count: &mut u64,
) -> Result<PolynomialNF, NormalFormError> {
    product_bound_only_inner(left, right, Some(fold_count))
}

fn product_bound_only_inner(
    left: PolynomialNF,
    right: PolynomialNF,
    mut fold_count: Option<&mut u64>,
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
                if let Some(count) = fold_count.as_deref_mut() {
                    *count = count.saturating_add(1);
                }
            } else {
                out.insert(monomial, multiplicity)?;
            }
        }
    }
    if !left.bounded_summary.is_exact_zero() {
        if let Some(count) = fold_count.as_deref_mut() {
            out = out.multiply_summary_counted(&left.bounded_summary, &right.exact_terms, count)?;
        } else {
            let mut ignored_fold_count = 0_u64;
            out = out.multiply_summary_counted(
                &left.bounded_summary,
                &right.exact_terms,
                &mut ignored_fold_count,
            )?;
        }
    }
    if !right.bounded_summary.is_exact_zero() {
        if let Some(count) = fold_count.as_deref_mut() {
            out = out.multiply_summary_counted(&right.bounded_summary, &left.exact_terms, count)?;
        } else {
            let mut ignored_fold_count = 0_u64;
            out = out.multiply_summary_counted(
                &right.bounded_summary,
                &left.exact_terms,
                &mut ignored_fold_count,
            )?;
        }
    }
    if !left.bounded_summary.is_exact_zero() && !right.bounded_summary.is_exact_zero() {
        out.bounded_summary = add_summary(
            out.bounded_summary,
            product_summary(left.bounded_summary, right.bounded_summary)?,
        )?;
        if let Some(count) = fold_count.as_deref_mut() {
            *count = count.saturating_add(1);
        }
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
