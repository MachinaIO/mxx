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
    /// A bounded compact matrix, persisted in canonical row-major form.
    SmallMatrix {
        matrix: ConcreteMatrixType,
        #[serde(with = "serde_support::bigint")]
        max_coefficient_bound: BigInt,
    },
    /// A bounded relation-bearing compact matrix. This remains distinct from
    /// both ordinary and generic bounded matrices across artifact boundaries.
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
            ConcreteWireType::Family { .. } => None,
        }
    }
}

/// Canonical payload layout for bounded small-RHS artifacts.
pub const SMALL_RHS_ROW_MAJOR_LAYOUT: &str = "small-rhs-row-major";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ManifestArtifact {
    pub artifact_type: ArtifactType,
    pub family_shape: Option<Vec<usize>>,
    pub confidentiality: ArtifactConfidentiality,
    pub content_hash: Option<[u8; 32]>,
    pub layout: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum ManifestValidationError {
    #[error("private artifact {name} must not expose a content hash")]
    PrivateContentHash { name: String },
    #[error("artifact {name} has an empty family shape")]
    EmptyFamilyShape { name: String },
    #[error("artifact {name} family shape product overflows usize")]
    FamilyShapeOverflow { name: String },
    #[error("bounded artifact {name} has a negative coefficient bound")]
    NegativeCoefficientBound { name: String },
    #[error("bounded artifact {name} must use the canonical small-RHS layout")]
    MissingBoundedLayout { name: String },
    #[error("artifact {name} has an invalid small-RHS layout")]
    InvalidLayout { name: String },
    #[error("unbounded artifact {name} must not specify a small-RHS layout")]
    UnexpectedLayout { name: String },
}

pub fn validate_manifest(manifest: &Manifest) -> Result<(), ManifestValidationError> {
    for (name, artifact) in &manifest.artifacts {
        if let Some(shape) = &artifact.family_shape {
            if shape.is_empty() {
                return Err(ManifestValidationError::EmptyFamilyShape { name: name.clone() });
            }
            shape
                .iter()
                .try_fold(1usize, |product, extent| product.checked_mul(*extent))
                .ok_or_else(|| ManifestValidationError::FamilyShapeOverflow {
                    name: name.clone(),
                })?;
        }
        if artifact.confidentiality == ArtifactConfidentiality::Private &&
            artifact.content_hash.is_some()
        {
            return Err(ManifestValidationError::PrivateContentHash { name: name.clone() });
        }
        let bounded = match &artifact.artifact_type {
            ArtifactType::SmallMatrix { max_coefficient_bound, .. } |
            ArtifactType::Preimage { max_coefficient_bound, .. } => {
                if max_coefficient_bound.is_negative() {
                    return Err(ManifestValidationError::NegativeCoefficientBound {
                        name: name.clone(),
                    });
                }
                true
            }
            ArtifactType::Trapdoor { preimage_max_coefficient_bound, .. } => {
                if preimage_max_coefficient_bound.is_negative() {
                    return Err(ManifestValidationError::NegativeCoefficientBound {
                        name: name.clone(),
                    });
                }
                false
            }
            _ => false,
        };
        match (bounded, artifact.layout.as_deref()) {
            (true, Some(layout)) if layout == SMALL_RHS_ROW_MAJOR_LAYOUT => {}
            (true, Some(_)) => {
                return Err(ManifestValidationError::InvalidLayout { name: name.clone() });
            }
            (true, None) => {
                return Err(ManifestValidationError::MissingBoundedLayout { name: name.clone() });
            }
            (false, Some(_)) => {
                return Err(ManifestValidationError::UnexpectedLayout { name: name.clone() });
            }
            (false, None) => {}
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct ExportArtifact {
    pub wire: WireId,
    pub artifact_type: ArtifactType,
    pub family_shape: Option<Vec<usize>>,
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
                    family_shape: artifact.family_shape.clone(),
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
                let (element_type, first_class_family_shape) = match wire_type {
                    ConcreteWireType::Family { element, shape } => {
                        (element.as_ref(), Some(shape.clone()))
                    }
                    scalar => (scalar, None),
                };
                let artifact_type = ArtifactType::from_wire_type(element_type)
                    .ok_or_else(|| ManifestExportError::UnsupportedOutput { name: name.clone() })?;
                let layout = match &artifact_type {
                    ArtifactType::SmallMatrix { .. } | ArtifactType::Preimage { .. } => {
                        Some(SMALL_RHS_ROW_MAJOR_LAYOUT.to_owned())
                    }
                    _ => None,
                };
                Ok((
                    name.clone(),
                    ExportArtifact {
                        wire: id,
                        artifact_type,
                        family_shape: first_class_family_shape,
                        confidentiality,
                        content_hash: None,
                        layout,
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
                    family_shape: None,
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
    fn bounded_artifact_types_preserve_semantic_kind_and_bound() {
        let matrix = ConcreteMatrixType::scalar(17.into(), 8);
        let bound = BigInt::from(3);
        assert_eq!(
            ArtifactType::from_wire_type(&ConcreteWireType::Preimage {
                matrix: matrix.clone(),
                max_coefficient_bound: bound.clone(),
            }),
            Some(ArtifactType::Preimage {
                matrix: matrix.clone(),
                max_coefficient_bound: bound.clone(),
            })
        );
        assert_ne!(
            ArtifactType::from_wire_type(&ConcreteWireType::SmallMatrix {
                matrix: matrix.clone(),
                max_coefficient_bound: bound.clone(),
            }),
            ArtifactType::from_wire_type(&ConcreteWireType::Matrix(matrix))
        );
    }

    #[test]
    fn manifest_requires_canonical_layout_for_bounded_artifacts() {
        let matrix = ConcreteMatrixType::scalar(17.into(), 8);
        let production_id = ProductionId { spec_hash: SpecHash([1; 32]), execution_nonce: [2; 32] };
        let artifact = |bound: BigInt, layout: Option<&str>| Manifest {
            ir_version: IR_VERSION,
            production_id: production_id.clone(),
            artifacts: BTreeMap::from([(
                "rhs".to_owned(),
                ManifestArtifact {
                    artifact_type: ArtifactType::SmallMatrix {
                        matrix: matrix.clone(),
                        max_coefficient_bound: bound,
                    },
                    family_shape: None,
                    confidentiality: ArtifactConfidentiality::Public,
                    content_hash: None,
                    layout: layout.map(str::to_owned),
                },
            )]),
        };
        assert!(matches!(
            validate_manifest(&artifact(3.into(), None)),
            Err(ManifestValidationError::MissingBoundedLayout { .. })
        ));
        assert!(matches!(
            validate_manifest(&artifact(3.into(), Some("legacy"))),
            Err(ManifestValidationError::InvalidLayout { .. })
        ));
        assert!(validate_manifest(&artifact(3.into(), Some(SMALL_RHS_ROW_MAJOR_LAYOUT))).is_ok());
        assert!(matches!(
            validate_manifest(&artifact((-1).into(), Some(SMALL_RHS_ROW_MAJOR_LAYOUT))),
            Err(ManifestValidationError::NegativeCoefficientBound { .. })
        ));
    }
}
