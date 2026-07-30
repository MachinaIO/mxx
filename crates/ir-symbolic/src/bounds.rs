use crate::{
    atom::{Atom, AtomClass, AtomId, AtomKind, AtomTable, PreimageRefs},
    checks::{CheckError, multiplication_type},
    term::{Term, TermError, TermList},
    types::ConcreteMatrixType,
    ubound::{UBound, UBoundError},
};
use num_bigint::BigInt;
use num_traits::Signed;
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BoundError {
    #[error(transparent)]
    Terms(#[from] TermError),
    #[error(transparent)]
    Check(#[from] CheckError),
    #[error(transparent)]
    Bound(#[from] UBoundError),
    #[error("term contains non-bounded atoms {atoms:?}: {term:?}")]
    NonBoundedTerm {
        term: Term,
        atoms: Vec<(AtomId, AtomClass)>,
        preimages: Vec<(AtomId, PreimageRefs)>,
    },
}

pub(crate) fn sum_norm(
    terms: &TermList,
    atoms: &AtomTable,
    ring_expansion: &UBound,
) -> Result<UBound, BoundError> {
    sum_norm_recursive(&terms.terms, atoms, ring_expansion)
}

fn sum_norm_recursive(
    terms: &[Term],
    atoms: &AtomTable,
    ring_expansion: &UBound,
) -> Result<UBound, BoundError> {
    let domain = terms
        .iter()
        .flat_map(|term| &term.factors)
        .filter_map(|factor| match &factor.atom {
            AtomId::Indicator { domain, .. } => Some(domain.clone()),
            _ => None,
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .next();
    let Some(domain) = domain else {
        return terms.iter().try_fold(UBound::zero(), |sum, term| {
            Ok(sum.add(&term_norm(term, atoms, ring_expansion)?))
        });
    };

    let mut branches = BTreeMap::<u64, Vec<Term>>::new();
    let mut rest = Vec::new();
    for term in terms {
        let branch = term.factors.iter().find_map(|factor| match &factor.atom {
            AtomId::Indicator { domain: candidate, branch } if candidate == &domain => {
                Some(*branch)
            }
            _ => None,
        });
        if let Some(branch) = branch {
            let stripped = Term {
                coefficient: term.coefficient.clone(),
                factors: term
                    .factors
                    .iter()
                    .filter(|factor| {
                        !matches!(
                            &factor.atom,
                            AtomId::Indicator { domain: candidate, .. } if candidate == &domain
                        )
                    })
                    .cloned()
                    .collect(),
            };
            branches.entry(branch).or_default().push(stripped);
        } else {
            rest.push(term.clone());
        }
    }

    let branch_maximum = branches.values().try_fold(UBound::zero(), |maximum, branch| {
        Ok::<_, BoundError>(UBound::max(
            &maximum,
            &sum_norm_recursive(branch, atoms, ring_expansion)?,
        ))
    })?;
    Ok(branch_maximum.add(&sum_norm_recursive(&rest, atoms, ring_expansion)?))
}

pub(crate) fn term_norm(
    term: &Term,
    atoms: &AtomTable,
    ring_expansion: &UBound,
) -> Result<UBound, BoundError> {
    let non_bounded_atoms = term
        .factors
        .iter()
        .filter_map(|factor| {
            atoms.get(&factor.atom).and_then(|atom| {
                matches!(atom.kind, AtomKind::Large)
                    .then(|| (factor.atom.clone(), atom.class.clone()))
            })
        })
        .collect::<Vec<_>>();
    if !non_bounded_atoms.is_empty() {
        let preimages = term
            .factors
            .iter()
            .filter_map(|factor| {
                atoms
                    .get(&factor.atom)
                    .and_then(|atom| atom.preimage_refs.clone())
                    .map(|preimage| (factor.atom.clone(), preimage))
            })
            .collect();
        return Err(BoundError::NonBoundedTerm {
            term: term.clone(),
            atoms: non_bounded_atoms,
            preimages,
        });
    }
    let coefficient =
        UBound::from_integer(&term.coefficient.abs()).expect("absolute coefficient is nonnegative");
    let mut norm = coefficient;
    let mut previous: Option<ConcreteMatrixType> = None;
    for factor in &term.factors {
        let atom =
            atoms.get(&factor.atom).ok_or_else(|| TermError::MissingAtom(factor.atom.clone()))?;
        let AtomKind::Bounded { norm: factor_norm } = &atom.kind else { unreachable!() };
        let factor_type = apply_view_type(atom.matrix_type.clone(), factor.view.as_ref());
        let factor_norm = effective_factor_norm(factor_norm, atom, factor.view.as_ref())?;
        if let Some(lhs) = &previous {
            norm = norm.mul(ring_expansion);
            if !lhs.is_scalar() && !factor_type.is_scalar() {
                norm = norm.mul(&UBound::from_u64(lhs.columns as u64));
            }
            previous = Some(multiplication_type(lhs, &factor_type)?);
        } else {
            previous = Some(factor_type);
        }
        norm = norm.mul(&factor_norm);
    }
    Ok(norm)
}

fn effective_factor_norm(
    norm: &UBound,
    atom: &Atom,
    view: Option<&crate::term::ViewDescriptor>,
) -> Result<UBound, BoundError> {
    let Some(target) = view.and_then(|view| view.modulus_cast.as_ref()) else {
        return Ok(norm.clone());
    };
    let cap_modulus = atom.matrix_type.modulus.clone().min(target.clone());
    let cap = UBound::from_ratio(&cap_modulus, &BigInt::from(2))?;
    Ok(UBound::min(norm, &cap))
}

fn apply_view_type(
    mut ty: ConcreteMatrixType,
    view: Option<&crate::term::ViewDescriptor>,
) -> ConcreteMatrixType {
    let Some(view) = view else {
        return ty;
    };
    if view.transpose {
        std::mem::swap(&mut ty.rows, &mut ty.columns);
    }
    if let Some(rows) = &view.row_range {
        ty.rows = rows.end.saturating_sub(rows.start);
    }
    if let Some(columns) = &view.column_range {
        ty.columns = columns.end.saturating_sub(columns.start);
    }
    if let Some(modulus) = &view.modulus_cast {
        ty.modulus = modulus.clone();
    }
    ty
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        atom::{DefExpr, SelectionDomain},
        term::Factor,
        types::{NodeId, Port, WireRef},
    };
    use std::collections::BTreeSet;

    fn scalar_type() -> ConcreteMatrixType {
        ConcreteMatrixType::scalar(BigInt::from(17), 1)
    }

    fn domain(node: u64) -> SelectionDomain {
        SelectionDomain {
            index_wire: WireRef { node: NodeId(node), port: Port(0) },
            instantiation_path: Vec::new(),
            count: 2,
            modulus: BigInt::from(17),
            ring_dimension: 1,
        }
    }

    fn bounded_atom(id: AtomId, norm: u64) -> Atom {
        Atom {
            id,
            class: AtomClass::Source,
            kind: AtomKind::Bounded { norm: UBound::from_u64(norm) },
            matrix_type: scalar_type(),
            dependencies: BTreeSet::new(),
            preimage_refs: None,
        }
    }

    fn indicator_atom(domain: &SelectionDomain, branch: u64) -> Atom {
        let id = AtomId::Indicator { domain: domain.clone(), branch };
        Atom {
            id,
            class: AtomClass::Derived {
                definition: DefExpr::Indicator { index_wire: domain.index_wire, branch },
            },
            kind: AtomKind::Bounded { norm: UBound::one() },
            matrix_type: scalar_type(),
            dependencies: BTreeSet::new(),
            preimage_refs: None,
        }
    }

    fn term(coefficient: i64, factors: Vec<AtomId>) -> Term {
        Term {
            coefficient: BigInt::from(coefficient),
            factors: factors.into_iter().map(|atom| Factor { atom, view: None }).collect(),
        }
    }

    #[test]
    fn indicator_sum_uses_branch_max_and_sums_duplicates_within_a_branch() {
        let selection = domain(1);
        let value = AtomId::Local { instantiation_path: Vec::new(), node: NodeId(2), port: 0 };
        let mut atoms = AtomTable::default();
        atoms.insert(bounded_atom(value.clone(), 1));
        for branch in 0..2 {
            atoms.insert(indicator_atom(&selection, branch));
        }
        let terms = TermList {
            terms: vec![
                term(
                    2,
                    vec![AtomId::Indicator { domain: selection.clone(), branch: 0 }, value.clone()],
                ),
                term(
                    3,
                    vec![AtomId::Indicator { domain: selection.clone(), branch: 0 }, value.clone()],
                ),
                term(4, vec![AtomId::Indicator { domain: selection.clone(), branch: 1 }, value]),
            ],
        };
        assert_eq!(sum_norm(&terms, &atoms, &UBound::one()).expect("bounded"), UBound::from_u64(5));
    }

    #[test]
    fn indicator_sum_recurses_across_nested_domains_and_adds_rest() {
        let outer = domain(1);
        let inner = domain(2);
        let value = AtomId::Local { instantiation_path: Vec::new(), node: NodeId(3), port: 0 };
        let rest = AtomId::Local { instantiation_path: Vec::new(), node: NodeId(4), port: 0 };
        let mut atoms = AtomTable::default();
        atoms.insert(bounded_atom(value.clone(), 1));
        atoms.insert(bounded_atom(rest.clone(), 2));
        for selection in [&outer, &inner] {
            for branch in 0..2 {
                atoms.insert(indicator_atom(selection, branch));
            }
        }
        let terms = TermList {
            terms: vec![
                term(
                    2,
                    vec![
                        AtomId::Indicator { domain: outer.clone(), branch: 0 },
                        AtomId::Indicator { domain: inner.clone(), branch: 0 },
                        value.clone(),
                    ],
                ),
                term(
                    7,
                    vec![
                        AtomId::Indicator { domain: outer.clone(), branch: 0 },
                        AtomId::Indicator { domain: inner.clone(), branch: 1 },
                        value.clone(),
                    ],
                ),
                term(
                    5,
                    vec![
                        AtomId::Indicator { domain: outer, branch: 1 },
                        AtomId::Indicator { domain: inner, branch: 0 },
                        value,
                    ],
                ),
                term(1, vec![rest]),
            ],
        };
        assert_eq!(sum_norm(&terms, &atoms, &UBound::one()).expect("bounded"), UBound::from_u64(9));
    }
}
