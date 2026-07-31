use crate::{
    atom::{AtomTable, TargetRef},
    term::{Term, TermError, TermList},
    types::WireRef,
};
use std::collections::BTreeMap;
use thiserror::Error;

pub const DEFAULT_REWRITE_LIMIT: usize = 1024;

pub trait TargetTermLists {
    fn resolve(&self, preimage: &crate::atom::AtomId, target: &TargetRef) -> Option<&TermList>;
}

#[derive(Default)]
pub struct TermListResolver {
    pub local: BTreeMap<WireRef, TermList>,
    pub imported: BTreeMap<TargetRef, TermList>,
}

impl TargetTermLists for TermListResolver {
    fn resolve(&self, _preimage: &crate::atom::AtomId, target: &TargetRef) -> Option<&TermList> {
        match target {
            TargetRef::Local(wire) => self.local.get(wire),
            TargetRef::Imported { .. } | TargetRef::Assumed(_) => self.imported.get(target),
        }
    }
}

#[derive(Debug, Error)]
pub enum RewriteError {
    #[error("preimage target {0:?} could not be resolved")]
    MissingTarget(TargetRef),
    #[error("preimage rewrite did not reach a fixpoint after {0} iterations")]
    IterationLimit(usize),
    #[error(transparent)]
    Terms(#[from] TermError),
}

pub fn rewrite_preimages(
    terms: TermList,
    atoms: &AtomTable,
    targets: &impl TargetTermLists,
) -> Result<TermList, RewriteError> {
    rewrite_preimages_with_limit(terms, atoms, targets, DEFAULT_REWRITE_LIMIT)
}

pub fn rewrite_preimages_with_limit(
    mut terms: TermList,
    atoms: &AtomTable,
    targets: &impl TargetTermLists,
    limit: usize,
) -> Result<TermList, RewriteError> {
    terms = terms.canonicalize(atoms)?;
    for _ in 0..limit {
        let mut changed = false;
        let mut output = Vec::new();
        for term in terms.terms {
            if let Some((position, preimage, target)) = matching_pair(&term, atoms) {
                let replacement = targets
                    .resolve(&preimage, &target)
                    .ok_or_else(|| RewriteError::MissingTarget(target.clone()))?;
                for target_term in &replacement.terms {
                    let mut factors =
                        Vec::with_capacity(term.factors.len() - 2 + target_term.factors.len());
                    factors.extend_from_slice(&term.factors[..position]);
                    factors.extend(target_term.factors.iter().cloned());
                    factors.extend_from_slice(&term.factors[position + 2..]);
                    output.push(Term {
                        coefficient: &term.coefficient * &target_term.coefficient,
                        factors,
                    });
                }
                changed = true;
            } else {
                output.push(term);
            }
        }
        terms = TermList { terms: output }.canonicalize(atoms)?;
        if !changed {
            return Ok(terms);
        }
    }
    Err(RewriteError::IterationLimit(limit))
}

fn matching_pair(
    term: &Term,
    atoms: &AtomTable,
) -> Option<(usize, crate::atom::AtomId, TargetRef)> {
    term.factors.windows(2).enumerate().find_map(|(position, pair)| {
        if pair[0].view.is_some() || pair[1].view.is_some() {
            return None;
        }
        let preimage = atoms.get(&pair[1].atom)?.preimage_refs.as_ref()?;
        (pair[0].atom == preimage.uniform)
            .then(|| (position, pair[1].atom.clone(), preimage.target.clone()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        atom::{
            Atom, AtomClass, AtomId, AtomKind, ManifestAtomId, PreimageRefs, ProductionId,
            SourceKind, SpecHash,
        },
        node::ConstantMatrix,
        term::Factor,
        types::{ConcreteMatrixType, NodeId, Port},
    };
    use num_bigint::BigInt;
    use std::collections::BTreeSet;

    fn local(node: u64) -> AtomId {
        AtomId::Local { instantiation_path: Vec::new(), node: NodeId(node), port: 0 }
    }

    fn atom(id: AtomId, kind: AtomKind, refs: Option<PreimageRefs>) -> Atom {
        Atom {
            id,
            class: AtomClass::Source {
                source: SourceKind::ConstantMatrix { value: ConstantMatrix::Identity },
            },
            kind,
            matrix_type: ConcreteMatrixType {
                modulus: BigInt::from(17),
                ring_dimension: 8,
                rows: 2,
                columns: 2,
            },
            dependencies: BTreeSet::new(),
            preimage_refs: refs,
            indicator: None,
        }
    }

    #[test]
    fn adjacent_uniform_preimage_pair_expands_target() {
        let uniform = local(1);
        let preimage = local(2);
        let target_atom = local(3);
        let target_wire = WireRef { node: NodeId(3), port: Port(0) };
        let mut atoms = AtomTable::default();
        atoms.insert(atom(uniform.clone(), AtomKind::Large, None));
        atoms.insert(atom(
            preimage.clone(),
            AtomKind::Bounded,
            Some(PreimageRefs { uniform: uniform.clone(), target: TargetRef::Local(target_wire) }),
        ));
        atoms.insert(atom(target_atom.clone(), AtomKind::Large, None));
        let mut resolver = TermListResolver::default();
        resolver.local.insert(target_wire, TermList::atom(target_atom.clone()));
        let input = TermList {
            terms: vec![Term {
                coefficient: BigInt::from(2),
                factors: vec![
                    Factor { atom: uniform, view: None },
                    Factor { atom: preimage, view: None },
                ],
            }],
        };
        let output = rewrite_preimages(input, &atoms, &resolver).expect("rewrite succeeds");
        assert_eq!(
            output,
            TermList {
                terms: vec![Term {
                    coefficient: BigInt::from(2),
                    factors: vec![Factor { atom: target_atom, view: None }],
                }],
            }
        );
    }

    #[test]
    fn different_productions_do_not_rewrite() {
        let production =
            |nonce| ProductionId { spec_hash: SpecHash([7; 32]), execution_nonce: [nonce; 32] };
        let a_first = AtomId::Imported {
            production_id: production(1),
            manifest_atom_id: ManifestAtomId([0; 32]),
        };
        let a_second = AtomId::Imported {
            production_id: production(2),
            manifest_atom_id: ManifestAtomId([0; 32]),
        };
        let k_second = AtomId::Imported {
            production_id: production(2),
            manifest_atom_id: ManifestAtomId([1; 32]),
        };
        let target = TargetRef::Imported {
            production_id: production(2),
            term_list_id: crate::atom::TermListId([0; 32]),
        };
        let mut atoms = AtomTable::default();
        atoms.insert(atom(a_first.clone(), AtomKind::Large, None));
        atoms.insert(atom(a_second.clone(), AtomKind::Large, None));
        atoms.insert(atom(
            k_second.clone(),
            AtomKind::Bounded,
            Some(PreimageRefs { uniform: a_second, target }),
        ));
        let input = TermList {
            terms: vec![Term {
                coefficient: BigInt::from(1),
                factors: vec![
                    Factor { atom: a_first, view: None },
                    Factor { atom: k_second, view: None },
                ],
            }],
        };
        let output = rewrite_preimages(input.clone(), &atoms, &TermListResolver::default())
            .expect("nonmatching pair needs no target");
        assert_eq!(output, input);
    }

    #[test]
    fn cyclic_target_hits_the_declared_rewrite_iteration_limit() {
        let uniform = local(1);
        let preimage = local(2);
        let target = WireRef { node: NodeId(3), port: Port(0) };
        let mut atoms = AtomTable::default();
        atoms.insert(atom(uniform.clone(), AtomKind::Large, None));
        atoms.insert(atom(
            preimage.clone(),
            AtomKind::Bounded,
            Some(PreimageRefs { uniform: uniform.clone(), target: TargetRef::Local(target) }),
        ));
        let terms = TermList {
            terms: vec![Term {
                coefficient: BigInt::from(1),
                factors: vec![
                    Factor { atom: uniform.clone(), view: None },
                    Factor { atom: preimage.clone(), view: None },
                ],
            }],
        };
        let resolver = TermListResolver {
            local: BTreeMap::from([(target, terms.clone())]),
            imported: BTreeMap::new(),
        };
        assert!(matches!(
            rewrite_preimages_with_limit(terms, &atoms, &resolver, 2),
            Err(RewriteError::IterationLimit(2))
        ));
    }
}
