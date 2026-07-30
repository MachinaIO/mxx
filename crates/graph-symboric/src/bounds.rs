use crate::{
    atom::{Atom, AtomClass, AtomId, AtomKind, AtomTable, PreimageRefs},
    checks::{CheckError, multiplication_type},
    term::{Term, TermError, TermList},
    types::ConcreteMatrixType,
    ubound::{UBound, UBoundError},
};
use num_bigint::BigInt;
use num_traits::Signed;
use std::collections::BTreeMap;
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
    type IndicatorKey = (crate::atom::SelectionDomain, u64);
    let mut ordinary = UBound::zero();
    let mut families = BTreeMap::<crate::atom::SelectionDomain, BTreeMap<u64, UBound>>::new();
    for term in &terms.terms {
        let norm = term_norm(term, atoms, ring_expansion)?;
        let indicators = term
            .factors
            .iter()
            .filter_map(|factor| match &factor.atom {
                AtomId::Indicator { domain, branch } => Some((domain.clone(), *branch)),
                _ => None,
            })
            .collect::<Vec<IndicatorKey>>();
        if indicators.len() == 1 {
            let (domain, branch) = &indicators[0];
            let entry = families
                .entry(domain.clone())
                .or_default()
                .entry(*branch)
                .or_insert_with(UBound::zero);
            *entry = entry.add(&norm);
        } else {
            ordinary = ordinary.add(&norm);
        }
    }
    for branches in families.values() {
        let maximum =
            branches.values().fold(UBound::zero(), |current, norm| UBound::max(&current, norm));
        ordinary = ordinary.add(&maximum);
    }
    Ok(ordinary)
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
