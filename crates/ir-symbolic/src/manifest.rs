use crate::{
    atom::{
        AssumedMetadata, Atom, AtomClass, AtomId, AtomTable, DeclaredDependencies,
        DeclaredDependencyRef, ManifestAtomId, PreimageRelation, ProductionId, SelectionDomainRef,
    },
    elaborate::SymbolicFamily,
    expression::{
        ExpressionError, SymbolicExprArena, SymbolicExprId, SymbolicExprNode, SymbolicExprRecord,
    },
};
use mxx_ir_core::types::ConcreteWireType;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const SYMBOLIC_MANIFEST_FORMAT_VERSION: u32 = 6;

/// A self-contained, new-format-only symbolic manifest.
///
/// Expression records are stored in arena order. Every child id therefore
/// names an earlier record, and import can replay the records through the
/// canonical constructors without retaining producer-local arena ids.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub format_version: u32,
    pub production_id: ProductionId,
    pub artifacts: BTreeMap<String, ManifestArtifact>,
    pub atoms: Vec<Atom>,
    pub expressions: Vec<SymbolicExprRecord>,
    pub preimage_relations: Vec<PreimageRelation>,
    pub assumption_digest: Option<[u8; 32]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ManifestArtifact {
    pub wire_type: ConcreteWireType,
    pub expression: Option<SymbolicExprId>,
    pub family: Option<SymbolicFamily>,
    pub content_hash: Option<[u8; 32]>,
    pub layout: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ExportArtifact {
    pub wire_type: ConcreteWireType,
    pub expression: Option<SymbolicExprId>,
    pub family: Option<SymbolicFamily>,
    pub content_hash: Option<[u8; 32]>,
    pub layout: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct ImportedManifest {
    pub artifacts: BTreeMap<String, ManifestArtifact>,
    pub preimage_relations: Vec<PreimageRelation>,
    pub assumption_digest: Option<[u8; 32]>,
}

#[derive(Debug, Error)]
pub enum ManifestError {
    #[error("unsupported symbolic manifest format version: {0}")]
    UnsupportedFormatVersion(u32),
    #[error("symbolic manifest contains an invalid atom identity: {0:?}")]
    InvalidAtomIdentity(AtomId),
    #[error("symbolic manifest references an unavailable expression: {0:?}")]
    InvalidExpression(SymbolicExprId),
    #[error("symbolic manifest contains conflicting atom metadata: {0:?}")]
    ConflictingAtom(AtomId),
    #[error("symbolic manifest contains an unqualified local selection domain")]
    UnqualifiedSelectionDomain,
    #[error("symbolic manifest artifact {name:?} has an invalid symbolic type: {reason}")]
    ArtifactType { name: String, reason: String },
    #[error("symbolic manifest expression is invalid: {0}")]
    Expression(#[from] ExpressionError),
    #[error("symbolic manifest serialization failed: {0}")]
    Serialization(String),
}

pub fn manifest_atom_id(id: &AtomId) -> Result<ManifestAtomId, ManifestError> {
    let bytes =
        serde_json::to_vec(id).map_err(|error| ManifestError::Serialization(error.to_string()))?;
    Ok(ManifestAtomId(Sha256::digest(bytes).into()))
}

pub fn export_manifest(
    production_id: ProductionId,
    artifacts: &BTreeMap<String, ExportArtifact>,
    atoms: &AtomTable,
    expressions: &SymbolicExprArena,
    preimage_relations: &[PreimageRelation],
    assumption_digest: Option<[u8; 32]>,
) -> Result<Manifest, ManifestError> {
    let mut reachable_expressions = BTreeSet::new();
    for (name, artifact) in artifacts {
        validate_artifact_roots(artifact.expression, artifact.family.as_ref(), expressions.len())?;
        validate_artifact_type(
            name,
            &artifact.wire_type,
            artifact.expression,
            artifact.family.as_ref(),
            expressions,
        )?;
        for root in artifact_roots(artifact.expression, artifact.family.as_ref()) {
            collect_reachable_expression(root, expressions, &mut reachable_expressions)?;
        }
    }
    let mut reachable_atoms = atoms_in_expressions(expressions, &reachable_expressions);
    let mut relation_indices = BTreeSet::new();
    loop {
        let mut changed = false;
        for (index, relation) in preimage_relations.iter().enumerate() {
            if reachable_atoms.contains(&relation.left_matrix) &&
                reachable_atoms.contains(&relation.preimage) &&
                relation_indices.insert(index)
            {
                collect_reachable_expression(
                    relation.product,
                    expressions,
                    &mut reachable_expressions,
                )?;
                changed = true;
            }
        }
        let next_atoms = atoms_in_expressions(expressions, &reachable_expressions);
        if next_atoms == reachable_atoms && !changed {
            break;
        }
        reachable_atoms = next_atoms;
    }
    for index in &relation_indices {
        reachable_atoms.insert(preimage_relations[*index].left_matrix.clone());
        reachable_atoms.insert(preimage_relations[*index].preimage.clone());
    }

    let local_ids = reachable_atoms
        .iter()
        .filter(|id| !matches!(id, AtomId::Imported { .. }))
        .map(|id| Ok((id.clone(), manifest_atom_id(id)?)))
        .collect::<Result<BTreeMap<_, _>, ManifestError>>()?;
    let qualify_atom = |id: &AtomId| qualify_atom_id(id, &production_id, &local_ids);

    let mut exported_atoms = reachable_atoms
        .iter()
        .map(|id| {
            let atom =
                atoms.get(id).ok_or_else(|| ManifestError::InvalidAtomIdentity(id.clone()))?;
            remap_atom(atom, &production_id, &qualify_atom)
        })
        .collect::<Result<Vec<_>, _>>()?;
    exported_atoms.sort_by(|left, right| left.id.cmp(&right.id));

    let expression_ids = reachable_expressions.iter().copied().collect::<Vec<_>>();
    let expression_map = expression_ids
        .iter()
        .enumerate()
        .map(|(new, old)| (*old, SymbolicExprId(new as u32)))
        .collect::<BTreeMap<_, _>>();
    let exported_expressions = expression_ids
        .iter()
        .map(|id| {
            let record =
                expressions.get(*id).cloned().ok_or(ManifestError::InvalidExpression(*id))?;
            let record = remap_record_atoms(record, &production_id, &qualify_atom)?;
            remap_record_export_expressions(record, &expression_map)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let exported_relations = relation_indices
        .iter()
        .map(|index| &preimage_relations[*index])
        .map(|relation| {
            Ok(PreimageRelation {
                left_matrix: qualify_atom(&relation.left_matrix)?,
                preimage: qualify_atom(&relation.preimage)?,
                product: *expression_map
                    .get(&relation.product)
                    .ok_or(ManifestError::InvalidExpression(relation.product))?,
            })
        })
        .collect::<Result<Vec<_>, ManifestError>>()?;
    let artifacts = artifacts
        .iter()
        .map(|(name, artifact)| {
            Ok((
                name.clone(),
                ManifestArtifact {
                    wire_type: artifact.wire_type.clone(),
                    expression: artifact
                        .expression
                        .map(|id| remap_export_id(id, &expression_map))
                        .transpose()?,
                    family: artifact
                        .family
                        .as_ref()
                        .map(|family| remap_export_family(family, &expression_map))
                        .transpose()?,
                    content_hash: artifact.content_hash,
                    layout: artifact.layout.clone(),
                },
            ))
        })
        .collect::<Result<_, ManifestError>>()?;

    Ok(Manifest {
        format_version: SYMBOLIC_MANIFEST_FORMAT_VERSION,
        production_id,
        artifacts,
        atoms: exported_atoms,
        expressions: exported_expressions,
        preimage_relations: exported_relations,
        assumption_digest,
    })
}

pub fn import_manifest(
    manifest: &Manifest,
    arena: &mut SymbolicExprArena,
    atoms: &mut AtomTable,
) -> Result<ImportedManifest, ManifestError> {
    if manifest.format_version != SYMBOLIC_MANIFEST_FORMAT_VERSION {
        return Err(ManifestError::UnsupportedFormatVersion(manifest.format_version));
    }
    for atom in &manifest.atoms {
        validate_imported_atom(atom)?;
        if let Some(existing) = atoms.get(&atom.id) {
            if existing != atom {
                return Err(ManifestError::ConflictingAtom(atom.id.clone()));
            }
        } else {
            atoms.insert(atom.clone());
        }
    }

    let mut expression_map = Vec::with_capacity(manifest.expressions.len());
    for record in &manifest.expressions {
        if matches!(
            &record.node,
            SymbolicExprNode::Select { domain: SelectionDomainRef::Local(_), .. }
        ) {
            return Err(ManifestError::UnqualifiedSelectionDomain);
        }
        let remapped = remap_record_expressions(record.clone(), &expression_map)?;
        expression_map.push(arena.replay(remapped, atoms)?);
    }

    let artifacts: BTreeMap<String, ManifestArtifact> = manifest
        .artifacts
        .iter()
        .map(|(name, artifact)| {
            Ok((
                name.clone(),
                ManifestArtifact {
                    wire_type: artifact.wire_type.clone(),
                    expression: artifact
                        .expression
                        .map(|id| remap_expression_id(id, &expression_map))
                        .transpose()?,
                    family: artifact
                        .family
                        .as_ref()
                        .map(|family| remap_family(family, &expression_map))
                        .transpose()?,
                    content_hash: artifact.content_hash,
                    layout: artifact.layout.clone(),
                },
            ))
        })
        .collect::<Result<_, ManifestError>>()?;
    for (name, artifact) in &artifacts {
        validate_artifact_type(
            name,
            &artifact.wire_type,
            artifact.expression,
            artifact.family.as_ref(),
            arena,
        )?;
    }
    let preimage_relations = manifest
        .preimage_relations
        .iter()
        .map(|relation| {
            let left = atoms
                .get(&relation.left_matrix)
                .ok_or_else(|| ManifestError::InvalidAtomIdentity(relation.left_matrix.clone()))?;
            let preimage = atoms
                .get(&relation.preimage)
                .ok_or_else(|| ManifestError::InvalidAtomIdentity(relation.preimage.clone()))?;
            let product = remap_expression_id(relation.product, &expression_map)?;
            let expected =
                mxx_ir_core::checks::multiplication_type(&left.matrix_type, &preimage.matrix_type)
                    .map_err(|_| ManifestError::ArtifactType {
                        name: "preimage relation".to_owned(),
                        reason: "B and K have incompatible matrix types".to_owned(),
                    })?;
            if arena.matrix_type(product)? != &expected {
                return Err(ManifestError::ArtifactType {
                    name: "preimage relation".to_owned(),
                    reason: "B K product type does not match its declared expression".to_owned(),
                });
            }
            Ok(PreimageRelation {
                left_matrix: relation.left_matrix.clone(),
                preimage: relation.preimage.clone(),
                product,
            })
        })
        .collect::<Result<_, ManifestError>>()?;

    Ok(ImportedManifest {
        artifacts,
        preimage_relations,
        assumption_digest: manifest.assumption_digest,
    })
}

fn qualify_atom_id(
    id: &AtomId,
    production: &ProductionId,
    local_ids: &BTreeMap<AtomId, ManifestAtomId>,
) -> Result<AtomId, ManifestError> {
    match id {
        AtomId::Imported { .. } => Ok(id.clone()),
        _ => Ok(AtomId::Imported {
            production_id: production.clone(),
            manifest_atom_id: *local_ids
                .get(id)
                .ok_or_else(|| ManifestError::InvalidAtomIdentity(id.clone()))?,
        }),
    }
}

fn remap_atom(
    atom: &Atom,
    production: &ProductionId,
    map: &impl Fn(&AtomId) -> Result<AtomId, ManifestError>,
) -> Result<Atom, ManifestError> {
    let class = match &atom.class {
        AtomClass::Source { source } => AtomClass::Source { source: source.clone() },
        AtomClass::Assumed { metadata } => AtomClass::Assumed {
            metadata: metadata
                .as_ref()
                .map(|metadata| remap_assumed_metadata(metadata, production)),
        },
    };
    Ok(Atom {
        id: map(&atom.id)?,
        class,
        kind: atom.kind.clone(),
        matrix_type: atom.matrix_type.clone(),
    })
}

fn remap_assumed_metadata(
    metadata: &AssumedMetadata,
    production: &ProductionId,
) -> AssumedMetadata {
    let dependencies = match &metadata.dependencies {
        DeclaredDependencies::Unknown => DeclaredDependencies::Unknown,
        DeclaredDependencies::Known(dependencies) => DeclaredDependencies::Known(
            dependencies
                .iter()
                .map(|dependency| match dependency {
                    DeclaredDependencyRef::Local(label) => DeclaredDependencyRef::Imported {
                        production_id: production.clone(),
                        label: label.clone(),
                    },
                    imported @ DeclaredDependencyRef::Imported { .. } => imported.clone(),
                })
                .collect::<BTreeSet<_>>(),
        ),
    };
    AssumedMetadata {
        norm: metadata.norm.clone(),
        is_const_poly: metadata.is_const_poly,
        zero_rows: metadata.zero_rows,
        dependencies,
        clt_ready: metadata.clt_ready,
    }
}

fn remap_record_atoms(
    mut record: SymbolicExprRecord,
    production: &ProductionId,
    map: &impl Fn(&AtomId) -> Result<AtomId, ManifestError>,
) -> Result<SymbolicExprRecord, ManifestError> {
    match &mut record.node {
        SymbolicExprNode::Atom(atom) => *atom = map(atom)?,
        SymbolicExprNode::Select { domain, .. } => {
            if let SelectionDomainRef::Local(local) = domain {
                *domain = SelectionDomainRef::Imported {
                    production_id: production.clone(),
                    domain: local.clone(),
                };
            }
        }
        _ => {}
    }
    Ok(record)
}

fn remap_record_expressions(
    mut record: SymbolicExprRecord,
    map: &[SymbolicExprId],
) -> Result<SymbolicExprRecord, ManifestError> {
    let remap = |id: &mut SymbolicExprId| -> Result<(), ManifestError> {
        *id = remap_expression_id(*id, map)?;
        Ok(())
    };
    match &mut record.node {
        SymbolicExprNode::Zero | SymbolicExprNode::Atom(_) => {}
        SymbolicExprNode::Add(values) |
        SymbolicExprNode::Mul(values) |
        SymbolicExprNode::Concat { inputs: values, .. } |
        SymbolicExprNode::Select { branches: values, .. } |
        SymbolicExprNode::CrtRecompose { inputs: values, .. } => {
            for value in values {
                remap(value)?;
            }
        }
        SymbolicExprNode::Scale { value, .. } |
        SymbolicExprNode::Transpose(value) |
        SymbolicExprNode::Slice { value, .. } |
        SymbolicExprNode::Reshape { value, .. } |
        SymbolicExprNode::ConstantCoefficient { value, .. } => remap(value)?,
        SymbolicExprNode::Tensor { left, right } => {
            remap(left)?;
            remap(right)?;
        }
    }
    Ok(record)
}

fn remap_expression_id(
    id: SymbolicExprId,
    map: &[SymbolicExprId],
) -> Result<SymbolicExprId, ManifestError> {
    map.get(id.0 as usize).copied().ok_or(ManifestError::InvalidExpression(id))
}

fn remap_family(
    family: &SymbolicFamily,
    map: &[SymbolicExprId],
) -> Result<SymbolicFamily, ManifestError> {
    Ok(match family {
        SymbolicFamily::ExactMembers(members) => SymbolicFamily::ExactMembers(
            members
                .iter()
                .map(|member| remap_expression_id(*member, map))
                .collect::<Result<_, _>>()?,
        ),
        SymbolicFamily::StructuralTemplate { count, template, index_slot } => {
            SymbolicFamily::StructuralTemplate {
                count: *count,
                template: remap_expression_id(*template, map)?,
                index_slot: *index_slot,
            }
        }
    })
}

fn validate_artifact_roots(
    expression: Option<SymbolicExprId>,
    family: Option<&SymbolicFamily>,
    expression_count: usize,
) -> Result<(), ManifestError> {
    let valid = |id: SymbolicExprId| (id.0 as usize) < expression_count;
    if expression.is_some_and(|id| !valid(id)) {
        return Err(ManifestError::InvalidExpression(expression.expect("present")));
    }
    match family {
        Some(SymbolicFamily::ExactMembers(members)) => {
            if let Some(id) = members.iter().copied().find(|id| !valid(*id)) {
                return Err(ManifestError::InvalidExpression(id));
            }
        }
        Some(SymbolicFamily::StructuralTemplate { template, .. }) if !valid(*template) => {
            return Err(ManifestError::InvalidExpression(*template));
        }
        _ => {}
    }
    Ok(())
}

fn validate_artifact_type(
    name: &str,
    wire_type: &ConcreteWireType,
    expression: Option<SymbolicExprId>,
    family: Option<&SymbolicFamily>,
    arena: &SymbolicExprArena,
) -> Result<(), ManifestError> {
    let (expected, expected_count) = match wire_type {
        ConcreteWireType::IndexedFamily { element, count } => (element.matrix_type(), Some(*count)),
        wire_type => (wire_type.matrix_type(), None),
    };
    match (family, expected_count) {
        (Some(SymbolicFamily::ExactMembers(members)), Some(count)) if members.len() != count => {
            return Err(ManifestError::ArtifactType {
                name: name.to_owned(),
                reason: "exact family count does not match the declared wire type".to_owned(),
            });
        }
        (Some(SymbolicFamily::StructuralTemplate { count, .. }), Some(expected))
            if *count != expected =>
        {
            return Err(ManifestError::ArtifactType {
                name: name.to_owned(),
                reason: "structural family count does not match the declared wire type".to_owned(),
            });
        }
        (Some(_), None) | (None, Some(_)) => {
            return Err(ManifestError::ArtifactType {
                name: name.to_owned(),
                reason: "family metadata does not match the declared wire type".to_owned(),
            });
        }
        _ => {}
    }
    let roots = artifact_roots(expression, family);
    let Some(expected) = expected else {
        if roots.is_empty() {
            return Ok(());
        }
        return Err(ManifestError::ArtifactType {
            name: name.to_owned(),
            reason: "non-matrix artifact has a symbolic matrix expression".to_owned(),
        });
    };
    for root in roots {
        if arena.matrix_type(root)? != expected {
            return Err(ManifestError::ArtifactType {
                name: name.to_owned(),
                reason: "expression type does not match the declared artifact type".to_owned(),
            });
        }
    }
    Ok(())
}

fn artifact_roots(
    expression: Option<SymbolicExprId>,
    family: Option<&SymbolicFamily>,
) -> Vec<SymbolicExprId> {
    let mut roots = expression.into_iter().collect::<Vec<_>>();
    match family {
        Some(SymbolicFamily::ExactMembers(members)) => roots.extend(members.iter().copied()),
        Some(SymbolicFamily::StructuralTemplate { template, .. }) => roots.push(*template),
        None => {}
    }
    roots
}

fn collect_reachable_expression(
    id: SymbolicExprId,
    arena: &SymbolicExprArena,
    reachable: &mut BTreeSet<SymbolicExprId>,
) -> Result<(), ManifestError> {
    if !reachable.insert(id) {
        return Ok(());
    }
    let record = arena.get(id).ok_or(ManifestError::InvalidExpression(id))?;
    for child in expression_children(&record.node) {
        collect_reachable_expression(child, arena, reachable)?;
    }
    Ok(())
}

fn atoms_in_expressions(
    arena: &SymbolicExprArena,
    expressions: &BTreeSet<SymbolicExprId>,
) -> BTreeSet<AtomId> {
    expressions
        .iter()
        .filter_map(|id| match &arena.get(*id)?.node {
            SymbolicExprNode::Atom(atom) => Some(atom.clone()),
            _ => None,
        })
        .collect()
}

fn expression_children(node: &SymbolicExprNode) -> Vec<SymbolicExprId> {
    match node {
        SymbolicExprNode::Zero | SymbolicExprNode::Atom(_) => Vec::new(),
        SymbolicExprNode::Add(children) |
        SymbolicExprNode::Mul(children) |
        SymbolicExprNode::Concat { inputs: children, .. } |
        SymbolicExprNode::Select { branches: children, .. } |
        SymbolicExprNode::CrtRecompose { inputs: children, .. } => children.clone(),
        SymbolicExprNode::Scale { value, .. } |
        SymbolicExprNode::Transpose(value) |
        SymbolicExprNode::Slice { value, .. } |
        SymbolicExprNode::Reshape { value, .. } |
        SymbolicExprNode::ConstantCoefficient { value, .. } => vec![*value],
        SymbolicExprNode::Tensor { left, right } => vec![*left, *right],
    }
}

fn remap_record_export_expressions(
    mut record: SymbolicExprRecord,
    map: &BTreeMap<SymbolicExprId, SymbolicExprId>,
) -> Result<SymbolicExprRecord, ManifestError> {
    let remap = |id: &mut SymbolicExprId| -> Result<(), ManifestError> {
        *id = remap_export_id(*id, map)?;
        Ok(())
    };
    match &mut record.node {
        SymbolicExprNode::Zero | SymbolicExprNode::Atom(_) => {}
        SymbolicExprNode::Add(values) |
        SymbolicExprNode::Mul(values) |
        SymbolicExprNode::Concat { inputs: values, .. } |
        SymbolicExprNode::Select { branches: values, .. } |
        SymbolicExprNode::CrtRecompose { inputs: values, .. } => {
            for value in values {
                remap(value)?;
            }
        }
        SymbolicExprNode::Scale { value, .. } |
        SymbolicExprNode::Transpose(value) |
        SymbolicExprNode::Slice { value, .. } |
        SymbolicExprNode::Reshape { value, .. } |
        SymbolicExprNode::ConstantCoefficient { value, .. } => remap(value)?,
        SymbolicExprNode::Tensor { left, right } => {
            remap(left)?;
            remap(right)?;
        }
    }
    Ok(record)
}

fn remap_export_id(
    id: SymbolicExprId,
    map: &BTreeMap<SymbolicExprId, SymbolicExprId>,
) -> Result<SymbolicExprId, ManifestError> {
    map.get(&id).copied().ok_or(ManifestError::InvalidExpression(id))
}

fn remap_export_family(
    family: &SymbolicFamily,
    map: &BTreeMap<SymbolicExprId, SymbolicExprId>,
) -> Result<SymbolicFamily, ManifestError> {
    Ok(match family {
        SymbolicFamily::ExactMembers(members) => SymbolicFamily::ExactMembers(
            members.iter().map(|member| remap_export_id(*member, map)).collect::<Result<_, _>>()?,
        ),
        SymbolicFamily::StructuralTemplate { count, template, index_slot } => {
            SymbolicFamily::StructuralTemplate {
                count: *count,
                template: remap_export_id(*template, map)?,
                index_slot: *index_slot,
            }
        }
    })
}

fn validate_imported_atom(atom: &Atom) -> Result<(), ManifestError> {
    if !matches!(atom.id, AtomId::Imported { .. }) {
        return Err(ManifestError::InvalidAtomIdentity(atom.id.clone()));
    }
    if let AtomClass::Assumed { metadata: Some(metadata) } = &atom.class {
        if let DeclaredDependencies::Known(dependencies) = &metadata.dependencies {
            if let Some(DeclaredDependencyRef::Local(_)) = dependencies
                .iter()
                .find(|dependency| matches!(dependency, DeclaredDependencyRef::Local(_)))
            {
                return Err(ManifestError::InvalidAtomIdentity(atom.id.clone()));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::{AtomKind, SourceKind, SpecHash};
    use mxx_ir_core::{
        FrozenGraphScopeId, ScopedWireRef,
        node::ConstantMatrix,
        types::{ConcreteMatrixType, ConcreteWireType, NodeId, Port, WireRef},
    };
    use num_bigint::BigInt;
    use num_traits::Zero;

    fn ty(modulus: u32) -> ConcreteMatrixType {
        ConcreteMatrixType {
            modulus: BigInt::from(modulus),
            ring_dimension: 8,
            rows: 1,
            columns: 1,
        }
    }

    fn id(node: u64) -> AtomId {
        AtomId::Local(ScopedWireRef {
            scope: FrozenGraphScopeId::Root,
            wire: WireRef { node: NodeId(node), port: Port(0) },
        })
    }

    #[test]
    fn round_trip_replays_tensor_nodes() {
        let mut atoms = AtomTable::default();
        for node in 0..3 {
            let id = id(node);
            atoms.insert(Atom {
                id,
                class: AtomClass::Source {
                    source: SourceKind::ConstantMatrix {
                        value: ConstantMatrix::UnitRow { index: 0.into() },
                    },
                },
                kind: AtomKind::Bounded,
                matrix_type: ty(257),
            });
        }
        let mut arena = SymbolicExprArena::default();
        let left = arena.atom(id(0), &atoms).expect("left");
        let right = arena.atom(id(1), &atoms).expect("right");
        let tensor = arena.tensor(ty(257), left, right).expect("tensor");
        let unused = arena.atom(id(2), &atoms).expect("unreachable atom");
        let artifacts = BTreeMap::from([(
            "tensor".to_owned(),
            ExportArtifact {
                wire_type: ConcreteWireType::Matrix(ty(257)),
                expression: Some(tensor),
                family: None,
                content_hash: None,
                layout: None,
            },
        )]);
        let production_id = ProductionId { spec_hash: SpecHash([3; 32]), execution_nonce: [5; 32] };
        let manifest =
            export_manifest(production_id.clone(), &artifacts, &atoms, &arena, &[], None)
                .expect("export");
        assert_eq!(manifest.format_version, SYMBOLIC_MANIFEST_FORMAT_VERSION);
        assert!(manifest.expressions.len() < arena.len());
        assert_eq!(manifest.atoms.len(), 2);
        assert!(!manifest.expressions.iter().any(|record| {
            matches!(&record.node, SymbolicExprNode::Atom(atom) if atom == &id(2))
        }));
        assert!(arena.get(unused).is_some());

        let mut wrong_version = manifest.clone();
        wrong_version.format_version = 5;
        assert!(matches!(
            import_manifest(
                &wrong_version,
                &mut SymbolicExprArena::default(),
                &mut AtomTable::default(),
            ),
            Err(ManifestError::UnsupportedFormatVersion(_))
        ));

        let mut wrong_artifact_type = manifest.clone();
        wrong_artifact_type.artifacts.get_mut("tensor").unwrap().wire_type =
            ConcreteWireType::Matrix(ty(17));
        assert!(matches!(
            import_manifest(
                &wrong_artifact_type,
                &mut SymbolicExprArena::default(),
                &mut AtomTable::default(),
            ),
            Err(ManifestError::ArtifactType { .. })
        ));

        let mut missing_relation_atom = manifest.clone();
        missing_relation_atom.preimage_relations.push(PreimageRelation {
            left_matrix: AtomId::Imported {
                production_id: production_id.clone(),
                manifest_atom_id: ManifestAtomId([41; 32]),
            },
            preimage: AtomId::Imported {
                production_id: production_id.clone(),
                manifest_atom_id: ManifestAtomId([42; 32]),
            },
            product: missing_relation_atom.artifacts["tensor"].expression.unwrap(),
        });
        assert!(matches!(
            import_manifest(
                &missing_relation_atom,
                &mut SymbolicExprArena::default(),
                &mut AtomTable::default(),
            ),
            Err(ManifestError::InvalidAtomIdentity(_))
        ));

        let mut malformed = manifest.clone();
        let tensor_record = malformed
            .expressions
            .iter_mut()
            .find(|record| matches!(record.node, SymbolicExprNode::Tensor { .. }))
            .expect("tensor record");
        let SymbolicExprNode::Tensor { left, .. } = &mut tensor_record.node else { unreachable!() };
        *left = SymbolicExprId(u32::MAX);
        assert!(matches!(
            import_manifest(
                &malformed,
                &mut SymbolicExprArena::default(),
                &mut AtomTable::default(),
            ),
            Err(ManifestError::InvalidExpression(SymbolicExprId(u32::MAX)))
        ));

        let mut malformed_crt = manifest.clone();
        malformed_crt.expressions.push(SymbolicExprRecord {
            matrix_type: ty(257),
            node: SymbolicExprNode::CrtRecompose {
                inputs: vec![malformed_crt.artifacts["tensor"].expression.unwrap()],
                plaintext_moduli: vec![BigInt::from(257u16)],
                reconstruction_coefficients: vec![BigInt::zero()],
            },
        });
        assert!(matches!(
            import_manifest(
                &malformed_crt,
                &mut SymbolicExprArena::default(),
                &mut AtomTable::default(),
            ),
            Err(ManifestError::Expression(ExpressionError::InvalidStructure))
        ));

        let mut unqualified_select = manifest.clone();
        unqualified_select.expressions.push(SymbolicExprRecord {
            matrix_type: ty(257),
            node: SymbolicExprNode::Select {
                domain: SelectionDomainRef::Local(crate::atom::SelectionDomain {
                    index_wire: ScopedWireRef {
                        scope: FrozenGraphScopeId::Root,
                        wire: WireRef { node: NodeId(91), port: Port(0) },
                    },
                    instantiation_path: Vec::new(),
                    count: 1,
                    modulus: BigInt::from(257u16),
                    ring_dimension: 8,
                }),
                branches: vec![unqualified_select.artifacts["tensor"].expression.unwrap()],
            },
        });
        assert!(matches!(
            import_manifest(
                &unqualified_select,
                &mut SymbolicExprArena::default(),
                &mut AtomTable::default(),
            ),
            Err(ManifestError::UnqualifiedSelectionDomain)
        ));

        let mut malformed_select = manifest.clone();
        malformed_select.expressions.push(SymbolicExprRecord {
            matrix_type: ty(257),
            node: SymbolicExprNode::Select {
                domain: SelectionDomainRef::Imported {
                    production_id: production_id.clone(),
                    domain: crate::atom::SelectionDomain {
                        index_wire: ScopedWireRef {
                            scope: FrozenGraphScopeId::Root,
                            wire: WireRef { node: NodeId(91), port: Port(0) },
                        },
                        instantiation_path: Vec::new(),
                        count: 1,
                        modulus: BigInt::from(17u8),
                        ring_dimension: 8,
                    },
                },
                branches: vec![malformed_select.artifacts["tensor"].expression.unwrap()],
            },
        });
        assert!(matches!(
            import_manifest(
                &malformed_select,
                &mut SymbolicExprArena::default(),
                &mut AtomTable::default(),
            ),
            Err(ManifestError::Expression(ExpressionError::InvalidStructure))
        ));

        let mut imported_atoms = AtomTable::default();
        let mut imported_arena = SymbolicExprArena::default();
        let imported =
            import_manifest(&manifest, &mut imported_arena, &mut imported_atoms).expect("import");
        let node = |name: &str| {
            let root = imported.artifacts[name].expression.expect("root");
            &imported_arena.get(root).expect("record").node
        };
        assert!(matches!(node("tensor"), SymbolicExprNode::Tensor { .. }));
        assert!(imported_atoms.values().all(|atom| matches!(
            &atom.id,
            AtomId::Imported { production_id: origin, .. } if origin == &production_id
        )));

        let reexports = imported
            .artifacts
            .iter()
            .map(|(name, artifact)| {
                (
                    name.clone(),
                    ExportArtifact {
                        wire_type: artifact.wire_type.clone(),
                        expression: artifact.expression,
                        family: artifact.family.clone(),
                        content_hash: artifact.content_hash,
                        layout: artifact.layout.clone(),
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        let reexported = export_manifest(
            ProductionId { spec_hash: SpecHash([7; 32]), execution_nonce: [11; 32] },
            &reexports,
            &imported_atoms,
            &imported_arena,
            &[],
            None,
        )
        .expect("re-export");
        assert!(reexported.atoms.iter().all(|atom| matches!(
            &atom.id,
            AtomId::Imported { production_id: origin, .. } if origin == &production_id
        )));
    }
}
