use crate::{
    encoding::IR_VERSION,
    serde_support,
    types::{ConcreteMatrixType, ConcreteWireType, WireId},
    validate::ValidatedGraph,
};
use num_bigint::BigInt;
use num_traits::Signed;
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum ArtifactConfidentiality {
    Public,
    Private,
}

/// Complete validated schema for a compact bounded-coefficient matrix.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ConcreteBoundedMatrixSchema {
    pub matrix: ConcreteMatrixType,
    #[serde(with = "serde_support::bigint")]
    pub max_coefficient_bound: BigInt,
}

/// Artifact semantics carried outside the shared compact matrix owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum SmallMatrixSemanticKind {
    Generic,
    Preimage,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum ArtifactType {
    Matrix(ConcreteMatrixType),
    SmallMatrix {
        matrix: ConcreteMatrixType,
        #[serde(with = "serde_support::bigint")]
        max_coefficient_bound: BigInt,
    },
    Preimage {
        matrix: ConcreteMatrixType,
        #[serde(with = "serde_support::bigint")]
        max_coefficient_bound: BigInt,
    },
    Bytes {
        length: usize,
    },
    Trapdoor {
        matrix: ConcreteMatrixType,
        sigma: crate::expr::RealExpr,
        #[serde(with = "serde_support::bigint")]
        gadget_base: BigInt,
        digit_count: usize,
        #[serde(with = "serde_support::bigint")]
        preimage_max_coefficient_bound: BigInt,
    },
    TypedBlob {
        type_name: String,
        schema_hash: [u8; 32],
    },
}

impl ArtifactType {
    pub fn from_wire_type(wire_type: &ConcreteWireType) -> Option<Self> {
        match wire_type {
            ConcreteWireType::Matrix(matrix) => Some(Self::Matrix(matrix.clone())),
            ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound } => {
                Some(Self::SmallMatrix {
                    matrix: matrix.clone(),
                    max_coefficient_bound: max_coefficient_bound.clone(),
                })
            }
            ConcreteWireType::Preimage { matrix, max_coefficient_bound } => Some(Self::Preimage {
                matrix: matrix.clone(),
                max_coefficient_bound: max_coefficient_bound.clone(),
            }),
            ConcreteWireType::Bytes { length } => Some(Self::Bytes { length: *length }),
            ConcreteWireType::Trapdoor {
                matrix,
                sigma,
                gadget_base,
                digit_count,
                preimage_max_coefficient_bound,
            } => Some(Self::Trapdoor {
                matrix: matrix.clone(),
                sigma: sigma.clone(),
                gadget_base: gadget_base.clone(),
                digit_count: *digit_count,
                preimage_max_coefficient_bound: preimage_max_coefficient_bound.clone(),
            }),
            ConcreteWireType::TypedBlob { type_name, schema_hash } => {
                Some(Self::TypedBlob { type_name: type_name.clone(), schema_hash: *schema_hash })
            }
            ConcreteWireType::ConstantInt |
            ConcreteWireType::ConstantReal |
            ConcreteWireType::ConstantBool |
            ConcreteWireType::Int |
            ConcreteWireType::Real |
            ConcreteWireType::Bool |
            ConcreteWireType::IndexedFamily { .. } => None,
        }
    }

    pub fn bounded_matrix_schema(
        &self,
    ) -> Option<(ConcreteBoundedMatrixSchema, SmallMatrixSemanticKind)> {
        match self {
            Self::SmallMatrix { matrix, max_coefficient_bound } => Some((
                ConcreteBoundedMatrixSchema {
                    matrix: matrix.clone(),
                    max_coefficient_bound: max_coefficient_bound.clone(),
                },
                SmallMatrixSemanticKind::Generic,
            )),
            Self::Preimage { matrix, max_coefficient_bound } => Some((
                ConcreteBoundedMatrixSchema {
                    matrix: matrix.clone(),
                    max_coefficient_bound: max_coefficient_bound.clone(),
                },
                SmallMatrixSemanticKind::Preimage,
            )),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ManifestArtifact {
    pub artifact_type: ArtifactType,
    pub family_count: Option<usize>,
    pub confidentiality: ArtifactConfidentiality,
    pub content_hash: Option<[u8; 32]>,
    pub layout: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum ManifestValidationError {
    #[error("private artifact {name} must not expose a content hash")]
    PrivateContentHash { name: String },
    #[error("bounded artifact {name} has a negative coefficient bound")]
    NegativeCoefficientBound { name: String },
}

pub fn validate_manifest(manifest: &Manifest) -> Result<(), ManifestValidationError> {
    for (name, artifact) in &manifest.artifacts {
        if artifact.confidentiality == ArtifactConfidentiality::Private &&
            artifact.content_hash.is_some()
        {
            return Err(ManifestValidationError::PrivateContentHash { name: name.clone() });
        }
        let bound = match &artifact.artifact_type {
            ArtifactType::SmallMatrix { max_coefficient_bound, .. } |
            ArtifactType::Preimage { max_coefficient_bound, .. } => Some(max_coefficient_bound),
            _ => None,
        };
        if bound.is_some_and(|bound| bound.is_negative()) {
            return Err(ManifestValidationError::NegativeCoefficientBound { name: name.clone() });
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct ExportArtifact {
    pub wire: WireId,
    pub artifact_type: ArtifactType,
    pub family_count: Option<usize>,
    pub confidentiality: ArtifactConfidentiality,
    pub content_hash: Option<[u8; 32]>,
    pub layout: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum ManifestExportError {
    #[error("graph output {name} refers to an unavailable wire")]
    MissingOutput { name: String },
    #[error("graph output {name} is not an artifact-compatible value")]
    UnsupportedOutput { name: String },
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
                    artifact_type: artifact.artifact_type.clone(),
                    family_count: artifact.family_count,
                    confidentiality: artifact.confidentiality,
                    content_hash: match artifact.confidentiality {
                        ArtifactConfidentiality::Public => artifact.content_hash,
                        ArtifactConfidentiality::Private => None,
                    },
                    layout: artifact.layout.clone(),
                },
            )
        })
        .collect();
    Manifest { ir_version: IR_VERSION, production_id, artifacts }
}

/// Exports runtime artifact metadata from a validated producer graph.
///
/// Indexed-family outputs become artifact families; compatible scalar wires
/// become singular artifacts. Every persisted output must be backed by an
/// graph output carrying an explicit confidentiality declaration.
pub fn export_validated_manifest(
    production_id: ProductionId,
    graph: &ValidatedGraph,
) -> Result<Manifest, ManifestExportError> {
    let artifacts = graph
        .source
        .outputs()
        .iter()
        .filter_map(|(name, output)| {
            let confidentiality = output.confidentiality?;
            Some((|| {
                let id = WireId { instantiation_path: Vec::new(), wire: output.value };
                let wire_type = graph
                    .root_scope()
                    .wire_types
                    .get(&output.value)
                    .ok_or_else(|| ManifestExportError::MissingOutput { name: name.clone() })?;
                let (element_type, first_class_family_count) = match wire_type {
                    ConcreteWireType::IndexedFamily { element, count } => {
                        (element.as_ref(), Some(*count))
                    }
                    scalar => (scalar, None),
                };
                let artifact_type = ArtifactType::from_wire_type(element_type)
                    .ok_or_else(|| ManifestExportError::UnsupportedOutput { name: name.clone() })?;
                Ok((
                    name.clone(),
                    ExportArtifact {
                        wire: id,
                        artifact_type,
                        family_count: first_class_family_count,
                        confidentiality,
                        content_hash: None,
                        layout: None,
                    },
                ))
            })())
        })
        .collect::<Result<BTreeMap<_, _>, ManifestExportError>>()?;
    Ok(export_manifest(production_id, &artifacts))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn private_manifest_artifacts_cannot_expose_content_hashes() {
        let manifest = Manifest {
            ir_version: IR_VERSION,
            production_id: ProductionId { spec_hash: SpecHash([1; 32]), execution_nonce: [2; 32] },
            artifacts: BTreeMap::from([(
                "private".to_owned(),
                ManifestArtifact {
                    artifact_type: ArtifactType::Bytes { length: 1 },
                    family_count: None,
                    confidentiality: ArtifactConfidentiality::Private,
                    content_hash: Some([3; 32]),
                    layout: None,
                },
            )]),
        };

        assert!(matches!(
            validate_manifest(&manifest),
            Err(ManifestValidationError::PrivateContentHash { name }) if name == "private"
        ));
    }

    #[test]
    fn bounded_artifact_kinds_remain_distinct() {
        let matrix = ConcreteMatrixType::scalar(BigInt::from(257), 8);
        let small = ConcreteWireType::SmallMatrix {
            matrix: matrix.clone(),
            max_coefficient_bound: BigInt::from(3),
        };
        let preimage = ConcreteWireType::Preimage {
            matrix: matrix.clone(),
            max_coefficient_bound: BigInt::from(3),
        };
        assert_eq!(
            ArtifactType::from_wire_type(&small),
            Some(ArtifactType::SmallMatrix {
                matrix: matrix.clone(),
                max_coefficient_bound: BigInt::from(3),
            })
        );
        assert_eq!(
            ArtifactType::from_wire_type(&preimage),
            Some(ArtifactType::Preimage { matrix, max_coefficient_bound: BigInt::from(3) })
        );
        assert_ne!(ArtifactType::from_wire_type(&small), ArtifactType::from_wire_type(&preimage));

        let schema = ConcreteBoundedMatrixSchema {
            matrix: ConcreteMatrixType::scalar(BigInt::from(257), 8),
            max_coefficient_bound: BigInt::from(3),
        };
        assert_eq!(
            ArtifactType::from_wire_type(&small).unwrap().bounded_matrix_schema(),
            Some((schema.clone(), SmallMatrixSemanticKind::Generic))
        );
        assert_eq!(
            ArtifactType::from_wire_type(&preimage).unwrap().bounded_matrix_schema(),
            Some((schema, SmallMatrixSemanticKind::Preimage))
        );
    }

    #[test]
    fn manifest_rejects_negative_bounded_artifact_bounds() {
        let manifest = Manifest {
            ir_version: IR_VERSION,
            production_id: ProductionId { spec_hash: SpecHash([1; 32]), execution_nonce: [2; 32] },
            artifacts: BTreeMap::from([(
                "negative".to_owned(),
                ManifestArtifact {
                    artifact_type: ArtifactType::SmallMatrix {
                        matrix: ConcreteMatrixType::scalar(BigInt::from(257), 8),
                        max_coefficient_bound: BigInt::from(-1),
                    },
                    family_count: None,
                    confidentiality: ArtifactConfidentiality::Public,
                    content_hash: None,
                    layout: None,
                },
            )]),
        };
        assert!(matches!(
            validate_manifest(&manifest),
            Err(ManifestValidationError::NegativeCoefficientBound { name }) if name == "negative"
        ));
    }
}
