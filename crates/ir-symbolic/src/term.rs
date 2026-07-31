use crate::{
    atom::{AtomId, AtomTable, SelectionDomainRef},
    serde_support,
};
use num_bigint::BigInt;
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, btree_map::Entry};
use thiserror::Error;

pub use mxx_ir_core::node::IndexRange;

#[derive(Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ViewDescriptor {
    pub transpose: bool,
    pub row_range: Option<IndexRange>,
    pub column_range: Option<IndexRange>,
    #[serde(default, with = "optional_bigint")]
    pub modulus_cast: Option<BigInt>,
}

impl ViewDescriptor {
    pub fn is_identity(&self) -> bool {
        !self.transpose &&
            self.row_range.is_none() &&
            self.column_range.is_none() &&
            self.modulus_cast.is_none()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct Factor {
    pub atom: AtomId,
    pub view: Option<ViewDescriptor>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Term {
    #[serde(with = "serde_support::bigint")]
    pub coefficient: BigInt,
    pub factors: Vec<Factor>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TermList {
    pub terms: Vec<Term>,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum TermError {
    #[error("atom is missing from the atom table: {0:?}")]
    MissingAtom(AtomId),
    #[error("matrix shapes are incompatible")]
    ShapeMismatch,
}

impl TermList {
    pub fn atom(atom: AtomId) -> Self {
        Self {
            terms: vec![Term {
                coefficient: BigInt::one(),
                factors: vec![Factor { atom, view: None }],
            }],
        }
    }

    pub fn zero() -> Self {
        Self::default()
    }

    pub fn add(&self, rhs: &Self, atoms: &AtomTable) -> Result<Self, TermError> {
        let mut terms = self.terms.clone();
        terms.extend(rhs.terms.clone());
        Self { terms }.canonicalize(atoms)
    }

    pub fn sub(&self, rhs: &Self, atoms: &AtomTable) -> Result<Self, TermError> {
        let mut terms = self.terms.clone();
        terms.extend(rhs.terms.iter().cloned().map(|mut term| {
            term.coefficient = -term.coefficient;
            term
        }));
        Self { terms }.canonicalize(atoms)
    }

    pub fn negate(&self) -> Self {
        Self {
            terms: self
                .terms
                .iter()
                .cloned()
                .map(|mut term| {
                    term.coefficient = -term.coefficient;
                    term
                })
                .collect(),
        }
    }

    pub fn scale(&self, scalar: &BigInt, atoms: &AtomTable) -> Result<Self, TermError> {
        let terms = self
            .terms
            .iter()
            .cloned()
            .map(|mut term| {
                term.coefficient *= scalar;
                term
            })
            .collect();
        Self { terms }.canonicalize(atoms)
    }

    pub fn multiply(&self, rhs: &Self, atoms: &AtomTable) -> Result<Self, TermError> {
        let mut terms = Vec::with_capacity(self.terms.len().saturating_mul(rhs.terms.len()));
        for lhs_term in &self.terms {
            for rhs_term in &rhs.terms {
                let mut factors = lhs_term.factors.clone();
                factors.extend(rhs_term.factors.clone());
                terms.push(Term {
                    coefficient: &lhs_term.coefficient * &rhs_term.coefficient,
                    factors,
                });
            }
        }
        Self { terms }.canonicalize(atoms)
    }

    pub fn canonicalize(self, atoms: &AtomTable) -> Result<Self, TermError> {
        let mut merged = BTreeMap::<Vec<Factor>, BigInt>::new();
        for term in self.terms {
            if term.coefficient.is_zero() {
                continue;
            }
            let Some(factors) = canonicalize_factors(term.factors, atoms)? else {
                continue;
            };
            match merged.entry(factors) {
                Entry::Vacant(entry) => {
                    entry.insert(term.coefficient);
                }
                Entry::Occupied(mut entry) => {
                    *entry.get_mut() += term.coefficient;
                    if entry.get().is_zero() {
                        entry.remove();
                    }
                }
            }
        }
        Ok(Self {
            terms: merged
                .into_iter()
                .map(|(factors, coefficient)| Term { coefficient, factors })
                .collect(),
        })
    }

    pub fn transpose(&self, atoms: &AtomTable) -> Result<Self, TermError> {
        let terms = self
            .terms
            .iter()
            .cloned()
            .map(|mut term| {
                term.factors.reverse();
                for factor in &mut term.factors {
                    let view = factor.view.get_or_insert_with(ViewDescriptor::default);
                    view.transpose = !view.transpose;
                    if view.is_identity() {
                        factor.view = None;
                    }
                }
                term
            })
            .collect();
        Self { terms }.canonicalize(atoms)
    }

    pub fn slice(
        &self,
        rows: Option<IndexRange>,
        columns: Option<IndexRange>,
        atoms: &AtomTable,
    ) -> Result<Self, TermError> {
        let mut terms = self.terms.clone();
        for term in &mut terms {
            if term.factors.is_empty() {
                continue;
            }
            let non_scalars = term
                .factors
                .iter()
                .enumerate()
                .filter_map(|(index, factor)| {
                    let atom = atoms
                        .get(&factor.atom)
                        .ok_or_else(|| TermError::MissingAtom(factor.atom.clone()));
                    match atom {
                        Ok(atom) if !atom.matrix_type.is_scalar() => Some(Ok(index)),
                        Ok(_) => None,
                        Err(error) => Some(Err(error)),
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            let first = non_scalars.first().copied().unwrap_or(0);
            let last = non_scalars.last().copied().unwrap_or(first);
            if let Some(rows) = &rows {
                let view = term.factors[first].view.get_or_insert_with(ViewDescriptor::default);
                view.row_range = Some(compose_range(view.row_range.as_ref(), rows));
            }
            if let Some(columns) = &columns {
                let view = term.factors[last].view.get_or_insert_with(ViewDescriptor::default);
                view.column_range = Some(compose_range(view.column_range.as_ref(), columns));
            }
        }
        Self { terms }.canonicalize(atoms)
    }

    pub fn cast_bounded_factors(
        &self,
        target_modulus: &BigInt,
        atoms: &AtomTable,
    ) -> Result<Self, TermError> {
        let mut terms = self.terms.clone();
        for term in &mut terms {
            for factor in &mut term.factors {
                let atom = atoms
                    .get(&factor.atom)
                    .ok_or_else(|| TermError::MissingAtom(factor.atom.clone()))?;
                if !atom.is_large() {
                    factor.view.get_or_insert_with(ViewDescriptor::default).modulus_cast =
                        Some(target_modulus.clone());
                }
            }
        }
        Self { terms }.canonicalize(atoms)
    }

    pub fn contains_large(&self, atoms: &AtomTable) -> Result<bool, TermError> {
        for term in &self.terms {
            for factor in &term.factors {
                let atom = atoms
                    .get(&factor.atom)
                    .ok_or_else(|| TermError::MissingAtom(factor.atom.clone()))?;
                if atom.is_large() {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }
}

fn canonicalize_factors(
    factors: Vec<Factor>,
    atoms: &AtomTable,
) -> Result<Option<Vec<Factor>>, TermError> {
    let mut scalars = Vec::new();
    let mut non_scalars = Vec::new();
    let mut indicators = BTreeMap::<SelectionDomainRef, u64>::new();

    for mut factor in factors {
        if factor.view.as_ref().is_some_and(ViewDescriptor::is_identity) {
            factor.view = None;
        }
        let atom =
            atoms.get(&factor.atom).ok_or_else(|| TermError::MissingAtom(factor.atom.clone()))?;
        if atom.matrix_type.is_scalar() {
            if let Some(indicator) = &atom.indicator {
                match indicators.entry(indicator.domain.clone()) {
                    Entry::Vacant(entry) => {
                        entry.insert(indicator.branch);
                        scalars.push(factor);
                    }
                    Entry::Occupied(entry) if *entry.get() == indicator.branch => {}
                    Entry::Occupied(_) => return Ok(None),
                }
            } else {
                scalars.push(factor);
            }
        } else {
            non_scalars.push(factor);
        }
    }
    scalars.sort();
    scalars.extend(non_scalars);
    Ok(Some(scalars))
}

fn compose_range(existing: Option<&IndexRange>, next: &IndexRange) -> IndexRange {
    existing.map_or_else(
        || *next,
        |existing| IndexRange {
            start: existing.start.saturating_add(next.start),
            end: existing.start.saturating_add(next.end).min(existing.end),
        },
    )
}

mod optional_bigint {
    use crate::serde_support;
    use num_bigint::BigInt;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(
        value: &Option<BigInt>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match value {
            Some(value) => {
                #[derive(Serialize)]
                struct Wrapped<'a>(#[serde(with = "serde_support::bigint")] &'a BigInt);
                serializer.serialize_some(&Wrapped(value))
            }
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<BigInt>, D::Error> {
        #[derive(Deserialize)]
        struct Wrapped(#[serde(with = "serde_support::bigint")] BigInt);
        Ok(Option::<Wrapped>::deserialize(deserializer)?.map(|wrapped| wrapped.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        atom::{Atom, AtomClass, AtomKind, SelectionDomain},
        node::ConstantMatrix,
        types::ConcreteMatrixType,
    };
    use std::collections::BTreeSet;

    fn insert_atom(table: &mut AtomTable, id: AtomId, scalar: bool) {
        let indicator = match &id {
            AtomId::Indicator { domain, branch } => Some(crate::atom::IndicatorRole {
                domain: crate::atom::SelectionDomainRef::Local(domain.clone()),
                branch: *branch,
            }),
            _ => None,
        };
        table.insert(Atom {
            id,
            class: AtomClass::Source {
                source: crate::atom::SourceKind::ConstantMatrix { value: ConstantMatrix::Identity },
            },
            kind: AtomKind::Bounded,
            matrix_type: ConcreteMatrixType {
                modulus: BigInt::from(97),
                ring_dimension: 8,
                rows: 1,
                columns: if scalar { 1 } else { 2 },
            },
            dependencies: BTreeSet::new(),
            preimage_refs: None,
            indicator,
        });
    }

    #[test]
    fn scalar_prefix_is_sorted_before_merging() {
        let s1 = AtomId::Constant { kind: "scalar".to_owned(), params: vec!["1".to_owned()] };
        let s2 = AtomId::Constant { kind: "scalar".to_owned(), params: vec!["2".to_owned()] };
        let matrix = AtomId::Constant { kind: "matrix".to_owned(), params: vec![] };
        let mut atoms = AtomTable::default();
        insert_atom(&mut atoms, s1.clone(), true);
        insert_atom(&mut atoms, s2.clone(), true);
        insert_atom(&mut atoms, matrix.clone(), false);
        let list = TermList {
            terms: vec![
                Term {
                    coefficient: BigInt::one(),
                    factors: vec![
                        Factor { atom: s1.clone(), view: None },
                        Factor { atom: s2.clone(), view: None },
                        Factor { atom: matrix.clone(), view: None },
                    ],
                },
                Term {
                    coefficient: -BigInt::one(),
                    factors: vec![
                        Factor { atom: s2, view: None },
                        Factor { atom: s1, view: None },
                        Factor { atom: matrix, view: None },
                    ],
                },
            ],
        };
        assert!(list.canonicalize(&atoms).expect("valid atoms").terms.is_empty());
    }

    #[test]
    fn indicator_algebra_is_domain_scoped() {
        let domain = SelectionDomain {
            index_wire: crate::types::WireRef {
                node: crate::types::NodeId(1),
                port: crate::types::Port(0),
            },
            instantiation_path: Vec::new(),
            count: 2,
            modulus: BigInt::from(97),
            ring_dimension: 8,
        };
        let b0 = AtomId::Indicator { domain: domain.clone(), branch: 0 };
        let b1 = AtomId::Indicator { domain, branch: 1 };
        let mut atoms = AtomTable::default();
        insert_atom(&mut atoms, b0.clone(), true);
        insert_atom(&mut atoms, b1.clone(), true);
        let list = TermList {
            terms: vec![Term {
                coefficient: BigInt::one(),
                factors: vec![Factor { atom: b0, view: None }, Factor { atom: b1, view: None }],
            }],
        };
        assert!(list.canonicalize(&atoms).expect("valid atoms").terms.is_empty());
    }
}
