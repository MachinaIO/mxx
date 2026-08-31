//! Power-LUT-owned typed values and artifact boundaries.
//!
//! `mxx-bgg` provides the algebraic encoding wire, while this module owns RHS
//! package validation and public projections. Secret/layout metadata is parsed
//! only while importing independently stored artifacts; runtime packages hold
//! only the GSW matrix and packed companion blocks required by Fuse.

use crate::encoding::{BggEncodingArtifactNames, PowerArtifactImportError};

use mxx_dsl::Mat;
use mxx_ir_core::{
    ParamEnv, artifact::ArtifactConfidentiality, node::ConcatAxis, types::ConcreteMatrixType,
};
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone)]
/// Role-free GSW material and companion encodings for the generic Fuse core.
///
/// The companion list has canonical `(source_row, target_column)` order. Each
/// companion is one horizontal block containing all tower-major CRT limbs for
/// that pair; no caller can observe or reorder individual Fuse limbs. Source
/// and target setup identities are validated only when an artifact is imported.
pub struct PowerRhsPackage {
    gsw_ciphertext: Mat,
    companions: Vec<PowerRhsCompanionBlock>,
}

/// A packed companion relation for one `(source_row, target_column)` pair.
/// Its horizontal columns contain every tower-major CRT digit in order. The
/// block is intentionally not a `BggEncodingWire`: its packed shape spans
/// several ordinary BGG public-key columns and is consumed only by Fuse.
#[derive(Clone)]
pub(crate) struct PowerRhsCompanionBlock {
    pub(crate) vector: Mat,
    pub(crate) public_matrix: Mat,
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Names of the GSW and companion artifacts making up an RHS package.
pub struct PowerRhsPackageArtifactNames {
    /// Private GSW ciphertext artifact.
    pub gsw_ciphertext: String,
    /// Packed companion encodings in canonical row/column order.
    pub companions: Vec<PowerRhsCompanionArtifactName>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Location and artifact names of one packed RHS companion block.
pub struct PowerRhsCompanionArtifactName {
    /// Source row selected by this companion.
    pub source_row: usize,
    /// Target public-key column selected by this companion.
    pub target_column: usize,
    /// Names of the packed companion's private vector and public matrix. The
    /// columns contain all CRT gadget limbs in tower-major order.
    pub encoding: BggEncodingArtifactNames,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ManifestRhsMetadata {
    pub(crate) source: ManifestSecretMetadata,
    pub(crate) target: ManifestSecretMetadata,
}

/// Secret/layout metadata retained solely for fail-closed artifact checks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ManifestSecretMetadata {
    pub(crate) modulus: mxx_ir_core::IntExpr,
    pub(crate) ring_dimension: mxx_ir_core::IntExpr,
    pub(crate) secret_dimension: usize,
    pub(crate) digit_count: usize,
    pub(crate) gadget_base: mxx_ir_core::IntExpr,
    pub(crate) identity: [u8; 32],
}

impl ManifestSecretMetadata {
    pub(crate) fn sampler(&self) -> mxx_bgg::BggSamplerLayout {
        mxx_bgg::BggSamplerLayout {
            modulus: self.modulus.clone(),
            ring_dimension: self.ring_dimension.clone(),
            secret_dimension: self.secret_dimension,
            digit_count: self.digit_count,
            gadget_base: self.gadget_base.clone(),
        }
    }
}

/// Role metadata for an RHS companion encoding artifact.
///
/// The single external-tagged variant intentionally preserves the manifest
/// shape used by earlier artifact writers while keeping this role separate
/// from automorphism and state-encoding roles.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum RhsCompanionArtifactRole {
    /// A packed companion attached to one GSW matrix entry. Its columns hold
    /// all tower-major CRT gadget limbs.
    RhsCompanion {
        /// Name of the companion's parent GSW artifact.
        gsw_artifact: String,
        /// Source matrix row.
        source_row: usize,
        /// Target matrix column.
        target_column: usize,
    },
}

/// Serializes canonical provenance metadata for an RHS package artifact.
#[cfg(test)]
pub(crate) fn power_rhs_artifact_layout<R: Serialize>(
    source: &ManifestSecretMetadata,
    target: &ManifestSecretMetadata,
    role: R,
) -> String {
    #[derive(Serialize)]
    struct ArtifactMetadata<'a, R> {
        source: &'a ManifestSecretMetadata,
        target: &'a ManifestSecretMetadata,
        role: R,
    }
    serde_json::to_string(&ArtifactMetadata {
        source,
        target,
        role: serde_json::to_value(role).expect("Power-LUT RHS role serialization"),
    })
    .expect("Power-LUT RHS metadata serialization")
}

#[derive(Debug, Error, Eq, PartialEq)]
/// Errors raised while constructing or checking generic RHS material.
pub enum PowerRhsPackageError {
    #[error("RHS package has an invalid role or companion count")]
    /// A package has no material or an artifact has a non-canonical companion count.
    InvalidRole,
}

impl PowerRhsPackage {
    /// Imports a package and all of its companions from a validated manifest.
    /// The import checks confidentiality, provenance, role, matrix type, and
    /// canonical companion count before returning the package.
    pub fn artifact_input(
        production_id: mxx_ir_core::artifact::ProductionId,
        manifest: &mxx_ir_core::artifact::Manifest,
        names: PowerRhsPackageArtifactNames,
    ) -> Result<Self, PowerArtifactImportError> {
        if manifest.production_id != production_id {
            return Err(PowerArtifactImportError::ProductionMismatch);
        }
        let artifact = manifest
            .artifacts
            .get(&names.gsw_ciphertext)
            .ok_or(PowerArtifactImportError::MissingArtifact)?;
        if artifact.confidentiality != ArtifactConfidentiality::Private ||
            artifact.family_count.is_some()
        {
            return Err(PowerArtifactImportError::ConfidentialityMismatch);
        }
        let metadata: ManifestRhsMetadata = serde_json::from_str(
            artifact.layout.as_deref().ok_or(PowerArtifactImportError::InvalidMetadata)?,
        )
        .map_err(|_| PowerArtifactImportError::InvalidMetadata)?;
        let source_layout = metadata.source.sampler();
        let target_layout = metadata.target.sampler();
        let layout = &source_layout;
        let ring = layout.ring();
        let modulus = layout
            .modulus
            .evaluate(&ParamEnv::default())
            .map_err(|_| PowerArtifactImportError::MatrixTypeMismatch)?;
        let ring_dimension = layout
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(PowerArtifactImportError::MatrixTypeMismatch)?;
        let target_columns = target_layout.public_key_columns();
        if artifact.artifact_type !=
            mxx_ir_core::artifact::ArtifactType::Matrix(ConcreteMatrixType {
                modulus,
                ring_dimension,
                rows: layout.secret_dimension,
                columns: target_columns,
            })
        {
            return Err(PowerArtifactImportError::MatrixTypeMismatch);
        }
        let gsw = ring.artifact_input(
            production_id.clone(),
            names.gsw_ciphertext.clone(),
            (layout.secret_dimension, target_columns),
            ArtifactConfidentiality::Private,
        );
        let expected_count = layout
            .secret_dimension
            .checked_mul(target_columns)
            .ok_or(PowerArtifactImportError::MatrixTypeMismatch)?;
        if names.companions.len() != expected_count {
            return Err(PowerArtifactImportError::MatrixTypeMismatch);
        }
        let mut unique_names = std::collections::BTreeSet::from([names.gsw_ciphertext.clone()]);
        let packed_columns = target_columns
            .checked_mul(layout.digit_count)
            .ok_or(PowerArtifactImportError::MatrixTypeMismatch)?;
        let mut companions = Vec::with_capacity(names.companions.len());
        for (position_index, position) in names.companions.into_iter().enumerate() {
            let canonical = position.source_row * target_columns + position.target_column;
            if canonical != position_index ||
                position.source_row >= source_layout.secret_dimension ||
                position.target_column >= target_columns ||
                !unique_names.insert(position.encoding.vector.clone()) ||
                !unique_names.insert(position.encoding.public_matrix.clone())
            {
                return Err(PowerArtifactImportError::MatrixTypeMismatch);
            }
            let encoding = crate::encoding::artifact_input_with_columns(
                production_id.clone(),
                manifest,
                position.encoding,
                Some(&RhsCompanionArtifactRole::RhsCompanion {
                    gsw_artifact: names.gsw_ciphertext.clone(),
                    source_row: position.source_row,
                    target_column: position.target_column,
                }),
                Some(packed_columns),
            )
            .map_err(|_| PowerArtifactImportError::MatrixTypeMismatch)?;
            companions.push(PowerRhsCompanionBlock {
                vector: encoding.vector,
                public_matrix: encoding.pubkey.matrix,
            });
        }
        let material =
            Self::new(gsw, companions).map_err(|_| PowerArtifactImportError::MatrixTypeMismatch)?;
        Ok(material)
    }
    pub(crate) fn new(
        gsw_ciphertext: Mat,
        companions: Vec<PowerRhsCompanionBlock>,
    ) -> Result<Self, PowerRhsPackageError> {
        if companions.is_empty() {
            return Err(PowerRhsPackageError::InvalidRole);
        }
        Ok(Self { gsw_ciphertext, companions })
    }
    pub(crate) fn gsw_ciphertext(&self) -> &Mat {
        &self.gsw_ciphertext
    }
    #[cfg(test)]
    pub(crate) fn companion_count(&self) -> usize {
        self.companions.len()
    }
    pub(crate) fn companion_at(&self, index: usize) -> Option<&PowerRhsCompanionBlock> {
        self.companions.get(index)
    }
    pub(crate) fn companion(
        &self,
        source_row: usize,
        target_column: usize,
        target_columns: usize,
    ) -> Option<&PowerRhsCompanionBlock> {
        let index = source_row.checked_mul(target_columns)?.checked_add(target_column)?;
        self.companions.get(index)
    }
    pub(crate) fn companion_block(
        &self,
        source_row: usize,
        target_column: usize,
        target_columns: usize,
    ) -> Option<Mat> {
        self.companion(source_row, target_column, target_columns)
            .map(|companion| companion.vector.clone())
    }
    /// Drops private ciphertext/vector expressions while preserving the exact
    /// public companion matrices and canonical order.
    pub fn public_projection(&self) -> PowerLutPublicRhsPackage {
        PowerLutPublicRhsPackage {
            companions: self.companions.iter().map(|c| c.public_matrix.clone()).collect(),
        }
    }

    /// Compare only the public part of two RHS packages.  The private GSW
    /// ciphertext and companion vectors are intentionally not inspected.
    #[allow(dead_code)]
    pub(crate) fn public_projection_matches(&self, other: &Self) -> bool {
        self.companions.len() == other.companions.len() &&
            self.companions.iter().zip(&other.companions).all(|(left, right)| {
                left.public_matrix.value_handle() == right.public_matrix.value_handle() &&
                    left.public_matrix.matrix_type() == right.public_matrix.matrix_type()
            })
    }
}

#[derive(Clone)]
/// Public-only projection of a [`PowerRhsPackage`].
///
/// This value is sufficient for reproducing the public matrix expression of
/// Fuse. It intentionally contains no private GSW ciphertext or encoding
/// vector, so it can be passed to the public compiler without crossing the
/// private-data boundary. Companion matrices use canonical
/// `(source_row, target_column)` order; each matrix is a packed horizontal
/// block containing all tower-major CRT gadget digits.
pub struct PowerLutPublicRhsPackage {
    companions: Vec<Mat>,
}

impl PowerLutPublicRhsPackage {
    /// Builds a public RHS descriptor from public companion matrices.
    ///
    /// The caller supplies public companion matrices in the canonical flat
    /// order established by the independently validated artifact boundary.
    /// Layout identities are checked during import, not passed to this runtime
    /// descriptor. Private GSW ciphertexts, vectors, and sparse coordinates
    /// are not part of this constructor.
    pub fn new(companions: Vec<Mat>) -> Result<Self, PowerRhsPackageError> {
        if companions.is_empty() {
            return Err(PowerRhsPackageError::InvalidRole);
        }
        Ok(Self { companions })
    }
    /// Packs the independently sampled ordinary BGG columns into the runtime
    /// companion-block layout. This is a setup-only conversion; the resulting
    /// package exposes only `(source_row, target_column)` blocks.
    pub(crate) fn from_sampled_matrices(
        source_dimension: usize,
        target_columns: usize,
        digits: usize,
        raw: Vec<Mat>,
    ) -> Result<Self, PowerRhsPackageError> {
        let expected = source_dimension
            .checked_mul(target_columns)
            .and_then(|count| count.checked_mul(digits))
            .ok_or(PowerRhsPackageError::InvalidRole)?;
        if raw.len() != expected || digits == 0 {
            return Err(PowerRhsPackageError::InvalidRole);
        }
        let mut companions = Vec::with_capacity(source_dimension * target_columns);
        for row in 0..source_dimension {
            for column in 0..target_columns {
                let start = (row * target_columns + column) * digits;
                companions
                    .push(Mat::concat(ConcatAxis::Columns, raw[start..start + digits].to_vec()));
            }
        }
        Self::new(companions)
    }
    pub(crate) fn first_companion(&self) -> Option<&Mat> {
        self.companions.first()
    }
    pub(crate) fn companions(&self) -> &[Mat] {
        &self.companions
    }
    pub(crate) fn companion_count(&self) -> usize {
        self.companions.len()
    }
    pub(crate) fn companion(
        &self,
        source_row: usize,
        target_column: usize,
        target_columns: usize,
    ) -> Option<&Mat> {
        let index = source_row.checked_mul(target_columns)?.checked_add(target_column)?;
        self.companions.get(index)
    }
    pub(crate) fn companion_block(
        &self,
        source_row: usize,
        target_column: usize,
        target_columns: usize,
    ) -> Option<Mat> {
        self.companion(source_row, target_column, target_columns).cloned()
    }
}

#[cfg(test)]
mod runtime_tests {
    use super::*;
    use crate::{encoding::PowerLutEncodingCompiler, public_key::PowerLutPublicKeyCompiler};
    use mxx_bgg::{BggEncodingWire, BggPublicKeyCompiler, BggPublicKeyWire};
    use mxx_dsl::{DslContext, Ring};
    use mxx_ir_core::{ParamEnv, node::NodeKind, types::WireType};
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use mxx_runtime::{
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
        transcript::SamplingMode,
    };
    use num_bigint::BigInt;
    use serial_test::serial;
    use std::collections::BTreeMap;

    fn scalar(parameters: &DCRTPolyParams, value: usize) -> DCRTPolyMatrix {
        DCRTPolyMatrix::from_poly_vec(
            parameters,
            vec![vec![DCRTPoly::from_usize_to_constant(parameters, value)]],
        )
    }

    fn row(parameters: &DCRTPolyParams, width: usize, offset: usize) -> DCRTPolyMatrix {
        DCRTPolyMatrix::from_poly_vec(
            parameters,
            vec![
                (0..width)
                    .map(|index| DCRTPoly::from_usize_to_constant(parameters, offset + index + 1))
                    .collect(),
            ],
        )
    }

    fn matrix(
        parameters: &DCRTPolyParams,
        rows: usize,
        columns: usize,
        offset: usize,
    ) -> DCRTPolyMatrix {
        DCRTPolyMatrix::from_poly_vec(
            parameters,
            (0..rows)
                .map(|row| {
                    (0..columns)
                        .map(|column| {
                            DCRTPoly::from_usize_to_constant(
                                parameters,
                                offset + row * columns + column + 1,
                            )
                        })
                        .collect()
                })
                .collect(),
        )
    }

    #[test]
    #[serial(dcrt_runtime)]
    fn encoding_derived_public_matrix_matches_public_only_graph_on_concrete_values() {
        // One base digit keeps the fixture small while still exercising the
        // executable decomposition and backend layout checks.
        let parameters = DCRTPolyParams::new(4, 1, 17, 17);
        let ring = Ring::new(
            BigInt::from(parameters.modulus().as_ref().clone()),
            parameters.ring_dimension() as usize,
        );
        let compiler = PowerLutEncodingCompiler::from_public_key(BggPublicKeyCompiler {
            ring: ring.clone(),
            base: 131_072.into(),
            digit_count: 1.into(),
        });
        let lhs = BggEncodingWire {
            vector: ring.input("lhs-vector", (1, 1)),
            pubkey: BggPublicKeyWire {
                matrix: ring.input("lhs-public", (1, 1)),
                reveal_plaintext: false,
            },
            plaintext: None,
        };
        let companion = BggEncodingWire {
            vector: ring.input("companion-vector", (1, 1)),
            pubkey: BggPublicKeyWire {
                matrix: ring.input("companion-public", (1, 1)),
                reveal_plaintext: false,
            },
            plaintext: None,
        };
        let rhs = PowerRhsPackage::new(
            ring.input("gsw", (1, 1)),
            vec![PowerRhsCompanionBlock {
                vector: companion.vector,
                public_matrix: companion.pubkey.matrix,
            }],
        )
        .unwrap();
        let encoded = compiler.fuse(&lhs, &rhs).unwrap();
        let public = PowerLutPublicKeyCompiler::new(compiler.bgg.public_key.clone())
            .fuse_public(&lhs.pubkey.matrix, &rhs.public_projection())
            .unwrap();
        let graph = DslContext::new("power-lut-public-runtime")
            .output("encoded", encoded.pubkey.matrix.clone())
            .unwrap()
            .output("public", public)
            .unwrap()
            .build()
            .unwrap();
        let validated = graph.validate(&ParamEnv::default()).unwrap();
        let result = execute(
            &validated,
            &mut cpu_backend([parameters.clone()]),
            BTreeMap::from([
                ("lhs-vector".into(), RuntimeValue::matrix(scalar(&parameters, 3))),
                ("lhs-public".into(), RuntimeValue::matrix(scalar(&parameters, 5))),
                ("companion-vector".into(), RuntimeValue::matrix(scalar(&parameters, 7))),
                ("companion-public".into(), RuntimeValue::matrix(scalar(&parameters, 11))),
                ("gsw".into(), RuntimeValue::matrix(scalar(&parameters, 13))),
            ]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let RuntimeValue::Matrix(encoded) = &result.outputs["encoded"] else { panic!("matrix") };
        let RuntimeValue::Matrix(public) = &result.outputs["public"] else { panic!("matrix") };
        assert_eq!(encoded, public, "encoding and public-only formulas diverged");
    }

    #[test]
    #[serial(dcrt_runtime)]
    fn crt_gadget_fuse_matches_public_projection_for_multiple_towers_and_base_digits() {
        // Two CRT towers and the largest base_bits that still gives exactly
        // two digits per 5-bit tower exercise tower-major order, including the
        // partial final digit of each tower.
        let parameters = DCRTPolyParams::new(2, 2, 5, 4);
        let source_dimension = 2;
        let digits = parameters.modulus_digits();
        let target_columns = source_dimension * digits;
        assert_eq!(digits, 4);
        assert_eq!(parameters.crt_bits().div_ceil(4), 2);
        assert_eq!(digits / parameters.crt_depth(), 2);
        let ring = Ring::new(
            BigInt::from(parameters.modulus().as_ref().clone()),
            parameters.ring_dimension() as usize,
        );
        let compiler = PowerLutEncodingCompiler::from_public_key(BggPublicKeyCompiler {
            ring: ring.clone(),
            base: 16.into(),
            digit_count: digits.into(),
        });
        let lhs = BggEncodingWire {
            vector: ring.input("crt-lhs-vector", (1, target_columns)),
            pubkey: BggPublicKeyWire {
                matrix: ring.input("crt-lhs-public", (source_dimension, target_columns)),
                reveal_plaintext: false,
            },
            plaintext: None,
        };
        let companion_count = source_dimension * target_columns * digits;
        let companions = (0..source_dimension * target_columns)
            .map(|block| {
                let vectors = (0..digits)
                    .map(|digit| {
                        let index = block * digits + digit;
                        ring.input(format!("crt-companion-vector-{index}"), (1, target_columns))
                    })
                    .collect();
                let publics = (0..digits)
                    .map(|digit| {
                        let index = block * digits + digit;
                        ring.input(
                            format!("crt-companion-public-{index}"),
                            (source_dimension, target_columns),
                        )
                    })
                    .collect();
                PowerRhsCompanionBlock {
                    vector: Mat::concat(ConcatAxis::Columns, vectors),
                    public_matrix: Mat::concat(ConcatAxis::Columns, publics),
                }
            })
            .collect::<Vec<_>>();
        let rhs = PowerRhsPackage::new(
            ring.input("crt-gsw", (source_dimension, target_columns)),
            companions,
        )
        .unwrap();
        let encoded = compiler.fuse(&lhs, &rhs).unwrap();
        let public = PowerLutPublicKeyCompiler::new(compiler.bgg.public_key.clone())
            .fuse_public(&lhs.pubkey.matrix, &rhs.public_projection())
            .unwrap();
        let graph = DslContext::new("power-lut-crt-fuse-runtime")
            .output("encoded-vector", encoded.vector)
            .unwrap()
            .output("encoded-public", encoded.pubkey.matrix)
            .unwrap()
            .output("public", public)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let mut inputs = BTreeMap::from([
            ("crt-lhs-vector".into(), RuntimeValue::matrix(row(&parameters, target_columns, 1))),
            (
                "crt-lhs-public".into(),
                RuntimeValue::matrix(matrix(&parameters, source_dimension, target_columns, 2)),
            ),
            (
                "crt-gsw".into(),
                RuntimeValue::matrix(matrix(&parameters, source_dimension, target_columns, 3)),
            ),
        ]);
        for index in 0..companion_count {
            inputs.insert(
                format!("crt-companion-vector-{index}"),
                RuntimeValue::matrix(row(&parameters, target_columns, 10 + index)),
            );
            inputs.insert(
                format!("crt-companion-public-{index}"),
                RuntimeValue::matrix(matrix(
                    &parameters,
                    source_dimension,
                    target_columns,
                    30 + index,
                )),
            );
        }
        let result = execute(
            &graph,
            &mut cpu_backend([parameters.clone()]),
            inputs,
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let RuntimeValue::Matrix(encoded_public) = &result.outputs["encoded-public"] else {
            panic!("encoded public output must be a matrix")
        };
        let RuntimeValue::Matrix(public) = &result.outputs["public"] else {
            panic!("public projection output must be a matrix")
        };
        assert_eq!(encoded_public, public);
    }

    #[test]
    #[serial(dcrt_runtime)]
    fn dcrt_gadget_columns_recompose_after_tower_major_decomposition() {
        // This is the primitive-level identity used by Fuse's routing blocks:
        // each DCRT gadget column is an idempotent CRT coefficient times a
        // local base power, and ordinary decomposition recomposes it exactly.
        let parameters = DCRTPolyParams::new(2, 2, 5, 4);
        let digits = parameters.modulus_digits();
        assert_eq!(parameters.crt_bits().div_ceil(4), 2);
        assert_eq!(digits / parameters.crt_depth(), 2);
        let ring = Ring::new(
            BigInt::from(parameters.modulus().as_ref().clone()),
            parameters.ring_dimension() as usize,
        );
        let gadget = ring.gadget(1, 16, digits);
        let reconstructed = gadget.clone() * gadget.decompose(16, digits).as_mat();
        let graph = DslContext::new("power-lut-dcrt-gadget-recomposition")
            .output("reconstructed", reconstructed)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let result = execute(
            &graph,
            &mut cpu_backend([parameters.clone()]),
            BTreeMap::new(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let RuntimeValue::Matrix(actual) = &result.outputs["reconstructed"] else {
            panic!("reconstructed gadget must be a matrix")
        };
        assert_eq!(
            actual.as_ref(),
            &DCRTPolyMatrix::gadget_matrix(&parameters, 1),
            "tower-major CRT gadget columns must recompose without global powers"
        );
    }

    #[test]
    fn public_fuse_has_no_gadget_decomposition_routing_slices() {
        let ring = Ring::new(257, 4);
        let compiler = PowerLutPublicKeyCompiler::new(BggPublicKeyCompiler {
            ring: ring.clone(),
            base: 8.into(),
            digit_count: 2.into(),
        });
        let input = ring.input("shape-input", (1, 2));
        let rhs =
            PowerLutPublicRhsPackage::new((0..2).map(|_| ring.zero((1, 4))).collect()).unwrap();
        let output = compiler.fuse_public(&input, &rhs).unwrap();
        let graph = DslContext::new("power-lut-fuse-shape")
            .output("result", output)
            .unwrap()
            .build()
            .unwrap();
        let nodes =
            graph.graph.scopes().values().flat_map(|scope| scope.nodes()).collect::<Vec<_>>();
        let decomposition_count = nodes
            .iter()
            .filter(|node| matches!(node.kind(), NodeKind::GadgetDecompose { .. }))
            .count();
        let tensor_nodes =
            nodes.iter().filter(|node| matches!(node.kind(), NodeKind::Tensor)).count();
        assert_eq!(decomposition_count, 1, "public Fuse must decompose the input once");
        assert!(nodes.iter().any(|node| matches!(node.kind(), NodeKind::Slice { .. })));
        assert!(
            !nodes.iter().any(|node| {
                matches!(node.kind(), NodeKind::Slice { .. }) &&
                    node.arguments().iter().any(|argument| {
                        matches!(argument.node().kind(), NodeKind::GadgetDecompose { .. })
                    })
            }),
            "companion slices must not consume D(A)"
        );
        assert_eq!(tensor_nodes, 0, "Fuse must use direct diagonal block algebra without Tensor");
        let column_loops = graph
            .graph
            .root_scope()
            .nodes()
            .iter()
            .filter(|node| {
                matches!(node.kind(), NodeKind::ParallelLoop(spec)
                    if spec.output_mode == mxx_ir_core::node::ParallelOutputMode::CollectColumns)
            })
            .collect::<Vec<_>>();
        assert_eq!(column_loops.len(), 1, "public Fuse has one ordered column sink");
        assert!(matches!(column_loops[0].output_types().first(), Some(WireType::Matrix(matrix))
            if matrix.columns == 2.into()));
    }

    #[test]
    fn packed_public_companions_ignore_private_gsw_material() {
        // Public Fuse receives only the packed companion matrices.  Changing
        // both the GSW expression and private companion vector must therefore
        // leave the derived public matrix expression unchanged.
        let ring = Ring::new(257, 4);
        let public_matrix = ring.input("independence-public", (1, 4));
        let make_rhs = |gsw_name: &str, vector_name: &str| {
            PowerRhsPackage::new(
                ring.input(gsw_name, (1, 2)),
                vec![PowerRhsCompanionBlock {
                    vector: ring.input(vector_name, (1, 4)),
                    public_matrix: public_matrix.clone(),
                }],
            )
            .unwrap()
        };
        let first = make_rhs("independence-gsw-a", "independence-vector-a");
        let second = make_rhs("independence-gsw-b", "independence-vector-b");
        let first_public = first.public_projection();
        let second_public = second.public_projection();
        assert_eq!(first_public.companion_count(), second_public.companion_count());
        assert_eq!(
            first_public.first_companion().unwrap().value_handle(),
            second_public.first_companion().unwrap().value_handle(),
            "packed public companion must be independent of GSW/private randomness"
        );
    }
}
