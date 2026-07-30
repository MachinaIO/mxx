use crate::{
    node::NodeKind,
    types::{ConcreteMatrixType, ConcreteWireType, Port, WireId, WireRef},
    validate::ValidatedGraph,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct SpecHash(pub [u8; 32]);

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ProductionId {
    pub spec_hash: SpecHash,
    pub execution_nonce: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub ir_version: u32,
    pub production_id: ProductionId,
    pub artifacts: BTreeMap<String, ManifestArtifact>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ManifestArtifact {
    pub wire_type: ConcreteMatrixType,
    pub family_count: Option<usize>,
    pub content_hash: Option<[u8; 32]>,
    pub layout: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ExportArtifact {
    pub wire: WireId,
    pub wire_type: ConcreteMatrixType,
    pub family: Option<Vec<WireId>>,
    pub content_hash: Option<[u8; 32]>,
    pub layout: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum ManifestExportError {
    #[error("graph output {name} refers to an unavailable wire")]
    MissingOutput { name: String },
    #[error("graph output {name} is not a matrix artifact")]
    NonMatrixOutput { name: String },
}

pub fn production_id(spec_hash: SpecHash, execution_nonce: [u8; 32]) -> ProductionId {
    ProductionId { spec_hash, execution_nonce }
}

pub fn export_manifest(
    production_id: ProductionId,
    artifacts: &BTreeMap<String, ExportArtifact>,
) -> Manifest {
    let artifacts = artifacts
        .iter()
        .map(|(name, artifact)| {
            (
                name.clone(),
                ManifestArtifact {
                    wire_type: artifact.wire_type.clone(),
                    family_count: artifact.family.as_ref().map(Vec::len),
                    content_hash: artifact.content_hash,
                    layout: artifact.layout.clone(),
                },
            )
        })
        .collect();
    Manifest { ir_version: 1, production_id, artifacts }
}

/// Exports runtime artifact metadata from a validated producer graph.
///
/// Output nodes with multiple ports become indexed artifact families; ordinary
/// matrix outputs become scalar artifacts. Non-matrix outputs are rejected
/// because runtime artifact stores persist matrix payloads only.
pub fn export_validated_manifest(
    production_id: ProductionId,
    graph: &ValidatedGraph,
) -> Result<Manifest, ManifestExportError> {
    let artifacts = graph
        .outputs
        .iter()
        .map(|(name, wire)| {
            let id = WireId { instantiation_path: Vec::new(), wire: *wire };
            let wire_type = graph
                .wires
                .get(&id)
                .ok_or_else(|| ManifestExportError::MissingOutput { name: name.clone() })?;
            let matrix = match wire_type {
                ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                    matrix.clone()
                }
                _ => {
                    return Err(ManifestExportError::NonMatrixOutput { name: name.clone() });
                }
            };
            let family = graph
                .source
                .node(wire.node)
                .and_then(|node| {
                    matches!(node.kind, NodeKind::Output { .. }).then(|| {
                        node.args
                            .iter()
                            .enumerate()
                            .map(|(port, _)| WireId {
                                instantiation_path: Vec::new(),
                                wire: WireRef { node: wire.node, port: Port(port as u32) },
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .filter(|members| members.len() > 1);
            Ok((
                name.clone(),
                ExportArtifact {
                    wire: id,
                    wire_type: matrix,
                    family,
                    content_hash: None,
                    layout: None,
                },
            ))
        })
        .collect::<Result<BTreeMap<_, _>, ManifestExportError>>()?;
    Ok(export_manifest(production_id, &artifacts))
}
