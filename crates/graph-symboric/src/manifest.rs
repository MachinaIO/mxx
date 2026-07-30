use crate::{
    atom::{
        Atom, AtomClass, AtomId, AtomKind, AtomTable, ConcatAxis, DefExpr, ManifestAtomId,
        PreimageRefs, ProductionId, SpecHash, TargetRef, TermListId,
    },
    term::{Factor, Term, TermList, ViewDescriptor},
    types::{ConcreteMatrixType, WireId, WireRef},
};
use num_bigint::BigInt;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub ir_version: u32,
    pub production_id: ProductionId,
    pub artifacts: BTreeMap<String, ManifestArtifact>,
    pub atoms: BTreeMap<ManifestAtomId, ManifestAtom>,
    pub term_lists: BTreeMap<TermListId, ManifestTermList>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ManifestArtifact {
    pub wire_type: ConcreteMatrixType,
    pub term_list_id: TermListId,
    pub family: Option<Vec<TermListId>>,
    pub content_hash: Option<[u8; 32]>,
    pub layout: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ManifestAtom {
    pub id: ManifestAtomId,
    pub class: ManifestAtomClass,
    pub kind: AtomKind,
    pub matrix_type: ConcreteMatrixType,
    pub dependencies: BTreeSet<ManifestAtomId>,
    pub preimage_refs: Option<ManifestPreimageRefs>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum ManifestAtomClass {
    Source,
    Derived { definition: ManifestDefExpr },
    Ghost { definition: ManifestDefExpr },
    Opaque,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ManifestPreimageRefs {
    pub uniform: ManifestAtomId,
    pub target: TermListId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum ManifestDefExpr {
    TermList(ManifestTermList),
    Tensor {
        left: ManifestAtomId,
        right: ManifestAtomId,
    },
    Concat {
        inputs: Vec<ManifestAtomId>,
        axis: ConcatAxis,
    },
    Reshape {
        input: ManifestAtomId,
        rows: usize,
        columns: usize,
    },
    ModDownImage {
        source: ManifestAtomId,
        source_modulus: String,
        target_modulus: String,
    },
    ModUpLift {
        source: ManifestAtomId,
        source_modulus: String,
        target_modulus: String,
    },
    Indicator {
        index_wire: WireRef,
        branch: u64,
    },
    ModDownError {
        input: WireRef,
        signal: ManifestTermList,
        source_modulus: String,
        target_modulus: String,
    },
    ModUpError {
        input: WireRef,
        lifted: ManifestTermList,
        source_modulus: String,
        target_modulus: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ManifestFactor {
    pub atom: ManifestAtomId,
    pub view: Option<ViewDescriptor>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ManifestTerm {
    pub coefficient: String,
    pub factors: Vec<ManifestFactor>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ManifestTermList {
    pub terms: Vec<ManifestTerm>,
}

#[derive(Clone, Debug)]
pub struct ExportArtifact {
    pub wire: WireId,
    pub wire_type: ConcreteMatrixType,
    pub family: Option<Vec<WireId>>,
    pub content_hash: Option<[u8; 32]>,
    pub layout: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct ImportedManifest {
    pub atoms: AtomTable,
    pub term_lists: BTreeMap<TargetRef, TermList>,
    pub artifacts: BTreeMap<String, ImportedArtifact>,
}

#[derive(Clone, Debug)]
pub struct ImportedArtifact {
    pub wire_type: ConcreteMatrixType,
    pub terms: TermList,
    pub family: Option<Vec<TermList>>,
}

#[derive(Debug, Error)]
pub enum ManifestError {
    #[error("wire {0:?} has no elaborated term list")]
    MissingWire(WireId),
    #[error("atom {0:?} is absent from the atom table")]
    MissingAtom(AtomId),
    #[error("preimage target {0:?} is not a local wire in the exported graph")]
    NonLocalTarget(TargetRef),
    #[error("manifest atom {0:?} is absent")]
    MissingManifestAtom(ManifestAtomId),
    #[error("manifest term list {0:?} is absent")]
    MissingTermList(TermListId),
    #[error("manifest integer is invalid: {0}")]
    InvalidInteger(String),
    #[error("manifest remapping is not total")]
    IncompleteRemapping,
}

pub fn production_id(spec_hash: SpecHash, execution_nonce: [u8; 32]) -> ProductionId {
    ProductionId { spec_hash, execution_nonce }
}

pub fn export_manifest(
    production_id: ProductionId,
    artifacts: &BTreeMap<String, ExportArtifact>,
    wire_terms: &BTreeMap<WireId, TermList>,
    atoms: &AtomTable,
) -> Result<Manifest, ManifestError> {
    let mut wires = BTreeSet::new();
    let mut atom_ids = BTreeSet::new();
    let mut wire_queue = VecDeque::new();
    let mut atom_queue = VecDeque::new();
    for artifact in artifacts.values() {
        wire_queue.push_back(artifact.wire.clone());
        wire_queue.extend(artifact.family.iter().flatten().cloned());
    }
    while !wire_queue.is_empty() || !atom_queue.is_empty() {
        while let Some(wire) = wire_queue.pop_front() {
            if !wires.insert(wire.clone()) {
                continue;
            }
            let terms =
                wire_terms.get(&wire).ok_or_else(|| ManifestError::MissingWire(wire.clone()))?;
            for atom in referenced_atoms(terms) {
                atom_queue.push_back(atom);
            }
        }
        while let Some(id) = atom_queue.pop_front() {
            if !atom_ids.insert(id.clone()) {
                continue;
            }
            let atom = atoms.get(&id).ok_or_else(|| ManifestError::MissingAtom(id.clone()))?;
            if let Some(preimage) = &atom.preimage_refs {
                atom_queue.push_back(preimage.uniform.clone());
                match &preimage.target {
                    TargetRef::Local(wire) => {
                        let path = match &id {
                            AtomId::Local { instantiation_path, .. } => instantiation_path.clone(),
                            _ => Vec::new(),
                        };
                        wire_queue.push_back(WireId { instantiation_path: path, wire: *wire });
                    }
                    imported => return Err(ManifestError::NonLocalTarget(imported.clone())),
                }
            }
        }
    }

    let atom_map = atom_ids
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, id)| (id, ManifestAtomId(index as u64)))
        .collect::<BTreeMap<_, _>>();
    let wire_map = wires
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, wire)| (wire, TermListId(index as u64)))
        .collect::<BTreeMap<_, _>>();

    let term_lists = wire_map
        .iter()
        .map(|(wire, id)| {
            let terms =
                wire_terms.get(wire).ok_or_else(|| ManifestError::MissingWire(wire.clone()))?;
            Ok((*id, export_terms(terms, &atom_map)?))
        })
        .collect::<Result<_, ManifestError>>()?;

    let exported_atoms = atom_ids
        .iter()
        .map(|id| {
            let atom = atoms.get(id).ok_or_else(|| ManifestError::MissingAtom(id.clone()))?;
            let manifest_id = atom_map[id];
            let definition_is_closed = definition_dependencies(&atom.class)
                .iter()
                .all(|dependency| atom_map.contains_key(dependency));
            let class = if definition_is_closed {
                export_class(&atom.class, &atom_map)?
            } else {
                ManifestAtomClass::Opaque
            };
            let preimage_refs = atom
                .preimage_refs
                .as_ref()
                .map(|preimage| {
                    let uniform = *atom_map
                        .get(&preimage.uniform)
                        .ok_or(ManifestError::IncompleteRemapping)?;
                    let TargetRef::Local(target) = preimage.target else {
                        return Err(ManifestError::NonLocalTarget(preimage.target.clone()));
                    };
                    let instantiation_path = match id {
                        AtomId::Local { instantiation_path, .. } => instantiation_path.clone(),
                        _ => Vec::new(),
                    };
                    let target = *wire_map
                        .get(&WireId { instantiation_path, wire: target })
                        .ok_or(ManifestError::IncompleteRemapping)?;
                    Ok(ManifestPreimageRefs { uniform, target })
                })
                .transpose()?;
            let dependencies = atom
                .dependencies
                .iter()
                .filter_map(|dependency| atom_map.get(dependency).copied())
                .collect();
            Ok((
                manifest_id,
                ManifestAtom {
                    id: manifest_id,
                    class,
                    kind: atom.kind.clone(),
                    matrix_type: atom.matrix_type.clone(),
                    dependencies,
                    preimage_refs,
                },
            ))
        })
        .collect::<Result<_, ManifestError>>()?;

    let exported_artifacts = artifacts
        .iter()
        .map(|(name, artifact)| {
            let term_list_id =
                *wire_map.get(&artifact.wire).ok_or(ManifestError::IncompleteRemapping)?;
            let family = artifact
                .family
                .as_ref()
                .map(|members| {
                    members
                        .iter()
                        .map(|wire| {
                            wire_map.get(wire).copied().ok_or(ManifestError::IncompleteRemapping)
                        })
                        .collect()
                })
                .transpose()?;
            Ok((
                name.clone(),
                ManifestArtifact {
                    wire_type: artifact.wire_type.clone(),
                    term_list_id,
                    family,
                    content_hash: artifact.content_hash,
                    layout: artifact.layout.clone(),
                },
            ))
        })
        .collect::<Result<_, ManifestError>>()?;

    Ok(Manifest {
        ir_version: crate::encoding::IR_VERSION,
        production_id,
        artifacts: exported_artifacts,
        atoms: exported_atoms,
        term_lists,
    })
}

pub fn import_manifest(manifest: &Manifest) -> Result<ImportedManifest, ManifestError> {
    let atom_id = |id: ManifestAtomId| AtomId::Imported {
        production_id: manifest.production_id.clone(),
        manifest_atom_id: id,
    };
    let target_ref = |id: TermListId| TargetRef::Imported {
        production_id: manifest.production_id.clone(),
        term_list_id: id,
    };
    let atom_map = manifest.atoms.keys().map(|id| (*id, atom_id(*id))).collect::<BTreeMap<_, _>>();
    let imported_terms = manifest
        .term_lists
        .iter()
        .map(|(id, terms)| Ok((target_ref(*id), import_terms(terms, &atom_map)?)))
        .collect::<Result<BTreeMap<_, _>, ManifestError>>()?;
    let mut atoms = AtomTable::default();
    for (id, atom) in &manifest.atoms {
        let imported_id = atom_id(*id);
        let class = import_class(&atom.class, &atom_map)?;
        let preimage_refs = atom
            .preimage_refs
            .as_ref()
            .map(|preimage| {
                Ok(PreimageRefs {
                    uniform: atom_map
                        .get(&preimage.uniform)
                        .cloned()
                        .ok_or(ManifestError::MissingManifestAtom(preimage.uniform))?,
                    target: target_ref(preimage.target),
                })
            })
            .transpose()?;
        atoms.insert(Atom {
            id: imported_id,
            class,
            kind: atom.kind.clone(),
            matrix_type: atom.matrix_type.clone(),
            dependencies: atom
                .dependencies
                .iter()
                .map(|dependency| {
                    atom_map
                        .get(dependency)
                        .cloned()
                        .ok_or(ManifestError::MissingManifestAtom(*dependency))
                })
                .collect::<Result<_, _>>()?,
            preimage_refs,
        });
    }
    let artifacts = manifest
        .artifacts
        .iter()
        .map(|(name, artifact)| {
            let terms = imported_terms
                .get(&target_ref(artifact.term_list_id))
                .cloned()
                .ok_or(ManifestError::MissingTermList(artifact.term_list_id))?;
            let family = artifact
                .family
                .as_ref()
                .map(|members| {
                    members
                        .iter()
                        .map(|id| {
                            imported_terms
                                .get(&target_ref(*id))
                                .cloned()
                                .ok_or(ManifestError::MissingTermList(*id))
                        })
                        .collect()
                })
                .transpose()?;
            Ok((
                name.clone(),
                ImportedArtifact { wire_type: artifact.wire_type.clone(), terms, family },
            ))
        })
        .collect::<Result<_, ManifestError>>()?;
    Ok(ImportedManifest { atoms, term_lists: imported_terms, artifacts })
}

fn referenced_atoms(terms: &TermList) -> impl Iterator<Item = AtomId> + '_ {
    terms.terms.iter().flat_map(|term| term.factors.iter().map(|factor| factor.atom.clone()))
}

fn definition_dependencies(class: &AtomClass) -> BTreeSet<AtomId> {
    match class {
        AtomClass::Derived { definition } | AtomClass::Ghost { definition } => {
            def_dependencies(definition)
        }
        AtomClass::Source | AtomClass::OpaqueImported => BTreeSet::new(),
    }
}

fn def_dependencies(definition: &DefExpr) -> BTreeSet<AtomId> {
    match definition {
        DefExpr::TermList(terms) => referenced_atoms(terms).collect(),
        DefExpr::Tensor { left, right } => BTreeSet::from([left.clone(), right.clone()]),
        DefExpr::Concat { inputs, .. } => inputs.iter().cloned().collect(),
        DefExpr::Reshape { input, .. } |
        DefExpr::ModDownImage { source: input, .. } |
        DefExpr::ModUpLift { source: input, .. } => BTreeSet::from([input.clone()]),
        DefExpr::ModDownError { signal, .. } => referenced_atoms(signal).collect(),
        DefExpr::ModUpError { lifted, .. } => referenced_atoms(lifted).collect(),
        DefExpr::Indicator { .. } => BTreeSet::new(),
    }
}

fn export_terms(
    terms: &TermList,
    map: &BTreeMap<AtomId, ManifestAtomId>,
) -> Result<ManifestTermList, ManifestError> {
    Ok(ManifestTermList {
        terms: terms
            .terms
            .iter()
            .map(|term| {
                Ok(ManifestTerm {
                    coefficient: term.coefficient.to_string(),
                    factors: term
                        .factors
                        .iter()
                        .map(|factor| {
                            Ok(ManifestFactor {
                                atom: *map
                                    .get(&factor.atom)
                                    .ok_or(ManifestError::IncompleteRemapping)?,
                                view: factor.view.clone(),
                            })
                        })
                        .collect::<Result<_, ManifestError>>()?,
                })
            })
            .collect::<Result<_, ManifestError>>()?,
    })
}

fn import_terms(
    terms: &ManifestTermList,
    map: &BTreeMap<ManifestAtomId, AtomId>,
) -> Result<TermList, ManifestError> {
    Ok(TermList {
        terms: terms
            .terms
            .iter()
            .map(|term| {
                Ok(Term {
                    coefficient: term
                        .coefficient
                        .parse::<BigInt>()
                        .map_err(|_| ManifestError::InvalidInteger(term.coefficient.clone()))?,
                    factors: term
                        .factors
                        .iter()
                        .map(|factor| {
                            Ok(Factor {
                                atom: map
                                    .get(&factor.atom)
                                    .cloned()
                                    .ok_or(ManifestError::MissingManifestAtom(factor.atom))?,
                                view: factor.view.clone(),
                            })
                        })
                        .collect::<Result<_, ManifestError>>()?,
                })
            })
            .collect::<Result<_, ManifestError>>()?,
    })
}

fn export_class(
    class: &AtomClass,
    map: &BTreeMap<AtomId, ManifestAtomId>,
) -> Result<ManifestAtomClass, ManifestError> {
    Ok(match class {
        AtomClass::Source => ManifestAtomClass::Source,
        AtomClass::Derived { definition } => {
            ManifestAtomClass::Derived { definition: export_definition(definition, map)? }
        }
        AtomClass::Ghost { definition } => {
            ManifestAtomClass::Ghost { definition: export_definition(definition, map)? }
        }
        AtomClass::OpaqueImported => ManifestAtomClass::Opaque,
    })
}

fn import_class(
    class: &ManifestAtomClass,
    map: &BTreeMap<ManifestAtomId, AtomId>,
) -> Result<AtomClass, ManifestError> {
    Ok(match class {
        ManifestAtomClass::Source => AtomClass::Source,
        ManifestAtomClass::Derived { definition } => {
            AtomClass::Derived { definition: import_definition(definition, map)? }
        }
        ManifestAtomClass::Ghost { definition } => {
            AtomClass::Ghost { definition: import_definition(definition, map)? }
        }
        ManifestAtomClass::Opaque => AtomClass::OpaqueImported,
    })
}

fn export_definition(
    definition: &DefExpr,
    map: &BTreeMap<AtomId, ManifestAtomId>,
) -> Result<ManifestDefExpr, ManifestError> {
    let remap = |id: &AtomId| map.get(id).copied().ok_or(ManifestError::IncompleteRemapping);
    Ok(match definition {
        DefExpr::TermList(terms) => ManifestDefExpr::TermList(export_terms(terms, map)?),
        DefExpr::Tensor { left, right } => {
            ManifestDefExpr::Tensor { left: remap(left)?, right: remap(right)? }
        }
        DefExpr::Concat { inputs, axis } => ManifestDefExpr::Concat {
            inputs: inputs.iter().map(remap).collect::<Result<_, _>>()?,
            axis: *axis,
        },
        DefExpr::Reshape { input, rows, columns } => {
            ManifestDefExpr::Reshape { input: remap(input)?, rows: *rows, columns: *columns }
        }
        DefExpr::ModDownImage { source, source_modulus, target_modulus } => {
            ManifestDefExpr::ModDownImage {
                source: remap(source)?,
                source_modulus: source_modulus.to_string(),
                target_modulus: target_modulus.to_string(),
            }
        }
        DefExpr::ModUpLift { source, source_modulus, target_modulus } => {
            ManifestDefExpr::ModUpLift {
                source: remap(source)?,
                source_modulus: source_modulus.to_string(),
                target_modulus: target_modulus.to_string(),
            }
        }
        DefExpr::Indicator { index_wire, branch } => {
            ManifestDefExpr::Indicator { index_wire: *index_wire, branch: *branch }
        }
        DefExpr::ModDownError { input, signal, source_modulus, target_modulus } => {
            ManifestDefExpr::ModDownError {
                input: *input,
                signal: export_terms(signal, map)?,
                source_modulus: source_modulus.to_string(),
                target_modulus: target_modulus.to_string(),
            }
        }
        DefExpr::ModUpError { input, lifted, source_modulus, target_modulus } => {
            ManifestDefExpr::ModUpError {
                input: *input,
                lifted: export_terms(lifted, map)?,
                source_modulus: source_modulus.to_string(),
                target_modulus: target_modulus.to_string(),
            }
        }
    })
}

fn import_definition(
    definition: &ManifestDefExpr,
    map: &BTreeMap<ManifestAtomId, AtomId>,
) -> Result<DefExpr, ManifestError> {
    let remap =
        |id: &ManifestAtomId| map.get(id).cloned().ok_or(ManifestError::MissingManifestAtom(*id));
    let parse = |value: &String| {
        value.parse::<BigInt>().map_err(|_| ManifestError::InvalidInteger(value.clone()))
    };
    Ok(match definition {
        ManifestDefExpr::TermList(terms) => DefExpr::TermList(import_terms(terms, map)?),
        ManifestDefExpr::Tensor { left, right } => {
            DefExpr::Tensor { left: remap(left)?, right: remap(right)? }
        }
        ManifestDefExpr::Concat { inputs, axis } => DefExpr::Concat {
            inputs: inputs.iter().map(remap).collect::<Result<_, _>>()?,
            axis: *axis,
        },
        ManifestDefExpr::Reshape { input, rows, columns } => {
            DefExpr::Reshape { input: remap(input)?, rows: *rows, columns: *columns }
        }
        ManifestDefExpr::ModDownImage { source, source_modulus, target_modulus } => {
            DefExpr::ModDownImage {
                source: remap(source)?,
                source_modulus: parse(source_modulus)?,
                target_modulus: parse(target_modulus)?,
            }
        }
        ManifestDefExpr::ModUpLift { source, source_modulus, target_modulus } => {
            DefExpr::ModUpLift {
                source: remap(source)?,
                source_modulus: parse(source_modulus)?,
                target_modulus: parse(target_modulus)?,
            }
        }
        ManifestDefExpr::Indicator { index_wire, branch } => {
            DefExpr::Indicator { index_wire: *index_wire, branch: *branch }
        }
        ManifestDefExpr::ModDownError { input, signal, source_modulus, target_modulus } => {
            DefExpr::ModDownError {
                input: *input,
                signal: import_terms(signal, map)?,
                source_modulus: parse(source_modulus)?,
                target_modulus: parse(target_modulus)?,
            }
        }
        ManifestDefExpr::ModUpError { input, lifted, source_modulus, target_modulus } => {
            DefExpr::ModUpError {
                input: *input,
                lifted: import_terms(lifted, map)?,
                source_modulus: parse(source_modulus)?,
                target_modulus: parse(target_modulus)?,
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        rewrite::{TermListResolver, rewrite_preimages},
        types::{NodeId, Port},
        ubound::UBound,
    };

    fn local(node: u64) -> AtomId {
        AtomId::Local { instantiation_path: Vec::new(), node: NodeId(node), port: 0 }
    }

    fn matrix_type() -> ConcreteMatrixType {
        ConcreteMatrixType { modulus: BigInt::from(17), ring_dimension: 8, rows: 2, columns: 2 }
    }

    fn source(id: AtomId, kind: AtomKind, refs: Option<PreimageRefs>) -> Atom {
        Atom {
            id,
            class: AtomClass::Source,
            kind,
            matrix_type: matrix_type(),
            dependencies: BTreeSet::new(),
            preimage_refs: refs,
        }
    }

    #[test]
    fn round_trip_preserves_preimage_target_and_rewrite() {
        let a = local(1);
        let k = local(2);
        let target = local(3);
        let output_wire = WireId {
            instantiation_path: Vec::new(),
            wire: WireRef { node: NodeId(4), port: Port(0) },
        };
        let target_wire = WireId {
            instantiation_path: Vec::new(),
            wire: WireRef { node: NodeId(3), port: Port(0) },
        };
        let mut atoms = AtomTable::default();
        atoms.insert(source(a.clone(), AtomKind::Large, None));
        atoms.insert(source(
            k.clone(),
            AtomKind::Bounded { norm: UBound::one() },
            Some(PreimageRefs { uniform: a.clone(), target: TargetRef::Local(target_wire.wire) }),
        ));
        atoms.insert(source(target.clone(), AtomKind::Large, None));
        let output_terms = TermList {
            terms: vec![Term {
                coefficient: BigInt::from(1),
                factors: vec![Factor { atom: a, view: None }, Factor { atom: k, view: None }],
            }],
        };
        let wire_terms = BTreeMap::from([
            (output_wire.clone(), output_terms),
            (target_wire, TermList::atom(target)),
        ]);
        let artifacts = BTreeMap::from([(
            "out".to_owned(),
            ExportArtifact {
                wire: output_wire,
                wire_type: matrix_type(),
                family: None,
                content_hash: None,
                layout: None,
            },
        )]);
        let production = ProductionId { spec_hash: SpecHash([1; 32]), execution_nonce: [2; 32] };
        let manifest =
            export_manifest(production, &artifacts, &wire_terms, &atoms).expect("export");
        let imported = import_manifest(&manifest).expect("import");
        let artifact = &imported.artifacts["out"];
        let resolver =
            TermListResolver { local: BTreeMap::new(), imported: imported.term_lists.clone() };
        let rewritten = rewrite_preimages(artifact.terms.clone(), &imported.atoms, &resolver)
            .expect("imported rewrite");
        assert_eq!(rewritten.terms.len(), 1);
        assert_eq!(rewritten.terms[0].factors.len(), 1);
    }

    #[test]
    fn imported_atoms_do_not_collide_with_same_numbered_consumer_nodes() {
        let producer_atom = local(7);
        let producer_wire = WireId {
            instantiation_path: Vec::new(),
            wire: WireRef { node: NodeId(7), port: Port(0) },
        };
        let mut atoms = AtomTable::default();
        atoms.insert(source(producer_atom.clone(), AtomKind::Large, None));
        let manifest = export_manifest(
            ProductionId { spec_hash: SpecHash([3; 32]), execution_nonce: [4; 32] },
            &BTreeMap::from([(
                "out".to_owned(),
                ExportArtifact {
                    wire: producer_wire.clone(),
                    wire_type: matrix_type(),
                    family: None,
                    content_hash: None,
                    layout: None,
                },
            )]),
            &BTreeMap::from([(producer_wire, TermList::atom(producer_atom))]),
            &atoms,
        )
        .expect("export");
        let imported = import_manifest(&manifest).expect("import");
        let imported_atom = imported.artifacts["out"].terms.terms[0].factors[0].atom.clone();
        let consumer_atom = local(7);
        assert_ne!(imported_atom, consumer_atom);
        assert!(matches!(
            imported_atom,
            AtomId::Imported {
                production_id,
                manifest_atom_id: ManifestAtomId(0),
            } if production_id == manifest.production_id
        ));
    }

    #[test]
    fn conversion_error_manifest_retains_its_symbolic_approximation() {
        let approximation = local(1);
        let error = local(2);
        let input_wire = WireRef { node: NodeId(9), port: Port(0) };
        let output_wire = WireId {
            instantiation_path: Vec::new(),
            wire: WireRef { node: NodeId(2), port: Port(0) },
        };
        let approximation_wire = WireId {
            instantiation_path: Vec::new(),
            wire: WireRef { node: NodeId(1), port: Port(0) },
        };
        let approximation_terms = TermList::atom(approximation.clone());
        let mut atoms = AtomTable::default();
        atoms.insert(source(
            approximation.clone(),
            AtomKind::Bounded { norm: UBound::one() },
            None,
        ));
        atoms.insert(Atom {
            id: error.clone(),
            class: AtomClass::Ghost {
                definition: DefExpr::ModUpError {
                    input: input_wire,
                    lifted: approximation_terms,
                    source_modulus: BigInt::from(5),
                    target_modulus: BigInt::from(17),
                },
            },
            kind: AtomKind::Bounded { norm: UBound::one() },
            matrix_type: matrix_type(),
            dependencies: BTreeSet::from([approximation.clone()]),
            preimage_refs: None,
        });
        let manifest = export_manifest(
            ProductionId { spec_hash: SpecHash([5; 32]), execution_nonce: [6; 32] },
            &BTreeMap::from([
                (
                    "approximation".to_owned(),
                    ExportArtifact {
                        wire: approximation_wire.clone(),
                        wire_type: matrix_type(),
                        family: None,
                        content_hash: None,
                        layout: None,
                    },
                ),
                (
                    "out".to_owned(),
                    ExportArtifact {
                        wire: output_wire.clone(),
                        wire_type: matrix_type(),
                        family: None,
                        content_hash: None,
                        layout: None,
                    },
                ),
            ]),
            &BTreeMap::from([
                (approximation_wire, TermList::atom(approximation)),
                (output_wire, TermList::atom(error)),
            ]),
            &atoms,
        )
        .expect("export");
        assert_eq!(manifest.atoms.len(), 2);
        let imported = import_manifest(&manifest).expect("import");
        let imported_error = &imported.artifacts["out"].terms.terms[0].factors[0].atom;
        let AtomClass::Ghost {
            definition: DefExpr::ModUpError { input, lifted, source_modulus, target_modulus },
        } = &imported.atoms.get(imported_error).expect("imported error atom").class
        else {
            panic!("conversion error definition")
        };
        assert_eq!(*input, input_wire);
        assert_eq!(source_modulus, &BigInt::from(5));
        assert_eq!(target_modulus, &BigInt::from(17));
        assert_eq!(lifted.terms.len(), 1);
        assert!(matches!(lifted.terms[0].factors[0].atom, AtomId::Imported { .. }));
    }

    #[test]
    fn same_spec_artifacts_from_different_productions_do_not_rewrite() {
        let a = local(1);
        let k = local(2);
        let target = local(3);
        let a_wire = WireId {
            instantiation_path: Vec::new(),
            wire: WireRef { node: NodeId(1), port: Port(0) },
        };
        let k_wire = WireId {
            instantiation_path: Vec::new(),
            wire: WireRef { node: NodeId(2), port: Port(0) },
        };
        let target_wire = WireId {
            instantiation_path: Vec::new(),
            wire: WireRef { node: NodeId(3), port: Port(0) },
        };
        let mut atoms = AtomTable::default();
        atoms.insert(source(a.clone(), AtomKind::Large, None));
        atoms.insert(source(
            k.clone(),
            AtomKind::Bounded { norm: UBound::one() },
            Some(PreimageRefs { uniform: a.clone(), target: TargetRef::Local(target_wire.wire) }),
        ));
        atoms.insert(source(target.clone(), AtomKind::Large, None));
        let wire_terms = BTreeMap::from([
            (a_wire.clone(), TermList::atom(a)),
            (k_wire.clone(), TermList::atom(k)),
            (target_wire, TermList::atom(target)),
        ]);
        let artifacts = BTreeMap::from([
            (
                "a".to_owned(),
                ExportArtifact {
                    wire: a_wire,
                    wire_type: matrix_type(),
                    family: None,
                    content_hash: None,
                    layout: None,
                },
            ),
            (
                "k".to_owned(),
                ExportArtifact {
                    wire: k_wire,
                    wire_type: matrix_type(),
                    family: None,
                    content_hash: None,
                    layout: None,
                },
            ),
        ]);
        let first = import_manifest(
            &export_manifest(
                ProductionId { spec_hash: SpecHash([9; 32]), execution_nonce: [1; 32] },
                &artifacts,
                &wire_terms,
                &atoms,
            )
            .expect("first export"),
        )
        .expect("first import");
        let second = import_manifest(
            &export_manifest(
                ProductionId { spec_hash: SpecHash([9; 32]), execution_nonce: [2; 32] },
                &artifacts,
                &wire_terms,
                &atoms,
            )
            .expect("second export"),
        )
        .expect("second import");
        let first_a = first.artifacts["a"].terms.terms[0].factors[0].atom.clone();
        let second_k = second.artifacts["k"].terms.terms[0].factors[0].atom.clone();
        let mut imported_atoms = first.atoms.clone();
        for atom in second.atoms.values() {
            imported_atoms.insert(atom.clone());
        }
        let mut imported_targets = first.term_lists.clone();
        imported_targets.extend(second.term_lists.clone());
        let mixed = TermList {
            terms: vec![Term {
                coefficient: BigInt::from(1),
                factors: vec![
                    Factor { atom: first_a, view: None },
                    Factor { atom: second_k, view: None },
                ],
            }],
        };
        let resolver = TermListResolver { local: BTreeMap::new(), imported: imported_targets };
        let rewritten =
            rewrite_preimages(mixed, &imported_atoms, &resolver).expect("mixed rewrite");
        assert_eq!(rewritten.terms[0].factors.len(), 2);
    }

    #[test]
    fn out_of_closure_lift_source_becomes_opaque_and_stays_large() {
        let source_id = local(10);
        let lift = local(11);
        let output_wire = WireId {
            instantiation_path: Vec::new(),
            wire: WireRef { node: NodeId(11), port: Port(0) },
        };
        let mut atoms = AtomTable::default();
        atoms.insert(source(source_id.clone(), AtomKind::Large, None));
        atoms.insert(Atom {
            id: lift.clone(),
            class: AtomClass::Derived {
                definition: DefExpr::ModUpLift {
                    source: source_id.clone(),
                    source_modulus: BigInt::from(17),
                    target_modulus: BigInt::from(257),
                },
            },
            kind: AtomKind::Large,
            matrix_type: ConcreteMatrixType { modulus: BigInt::from(257), ..matrix_type() },
            dependencies: BTreeSet::from([source_id]),
            preimage_refs: None,
        });
        let artifacts = BTreeMap::from([(
            "lift".to_owned(),
            ExportArtifact {
                wire: output_wire.clone(),
                wire_type: ConcreteMatrixType { modulus: BigInt::from(257), ..matrix_type() },
                family: None,
                content_hash: None,
                layout: None,
            },
        )]);
        let manifest = export_manifest(
            ProductionId { spec_hash: SpecHash([3; 32]), execution_nonce: [4; 32] },
            &artifacts,
            &BTreeMap::from([(output_wire, TermList::atom(lift))]),
            &atoms,
        )
        .expect("export");
        let exported = manifest.atoms.values().next().expect("one atom");
        assert_eq!(exported.class, ManifestAtomClass::Opaque);
        assert_eq!(exported.kind, AtomKind::Large);
    }
}
