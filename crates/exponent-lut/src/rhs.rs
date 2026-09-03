//! Setup-fixed RHS material for Exponent-LUT.
//!
//! The setup-fixed construction intentionally has one runtime RHS object: the
//! GSW ciphertext `C`. Older versions stored one BGG encoding for every
//! gadget digit of every entry of `C`. Those companion encodings are needed
//! only when the public matrix must be independent of `C`; they are not part
//! of this setup-fixed interface and are not reconstructed here.
//!
//! A package stores the GSW matrix `C` satisfying `t C = y v G + e_C` for
//! payload secret `t`, target secret `v`, and scalar/input value `y`. Fuse uses
//! `c G^{-1}(C)` and therefore needs concrete C itself; import validates its
//! production, public role, and concrete matrix type against the caller's
//! source/target sampler layouts, without loading private companion encodings.

use crate::encoding::ExponentArtifactImportError;
use mxx_dsl::Mat;
use mxx_ir_core::{
    ParamEnv,
    artifact::{ArtifactConfidentiality, ArtifactType, Manifest, ProductionId},
    types::ConcreteMatrixType,
};
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Artifact names for one setup-fixed RHS ciphertext.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExponentRhsPackageArtifactNames {
    /// The fixed GSW ciphertext `C`.
    pub gsw_ciphertext: String,
}

/// Setup-fixed RHS package consumed by the encoding evaluator.
///
/// `C` is deliberately the only payload. Its value is fixed when the package
/// is prepared, so the public-key projection is allowed to use the same
/// matrix and computes `A * G^{-1}(C)`.
#[derive(Clone)]
pub struct ExponentRhsPackage {
    gsw_ciphertext: Mat,
}

/// Metadata stored by artifact producers so imports can reject a ciphertext
/// prepared for a different source/target BGG layout. These fields are
/// import-only provenance and are not runtime RHS data.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct ManifestRhsMetadata {
    pub(crate) source: ManifestSecretMetadata,
    pub(crate) target: ManifestSecretMetadata,
}

impl ManifestRhsMetadata {
    pub(crate) fn from_layouts(
        source: &mxx_bgg::BggSamplerLayout,
        source_identity: [u8; 32],
        target: &mxx_bgg::BggSamplerLayout,
        target_identity: [u8; 32],
    ) -> Self {
        Self {
            source: ManifestSecretMetadata::from_layout(source, source_identity),
            target: ManifestSecretMetadata::from_layout(target, target_identity),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct ManifestSecretMetadata {
    pub(crate) modulus: mxx_ir_core::IntExpr,
    pub(crate) ring_dimension: mxx_ir_core::IntExpr,
    pub(crate) secret_dimension: usize,
    pub(crate) digit_count: usize,
    pub(crate) gadget_base: mxx_ir_core::IntExpr,
    pub(crate) identity: [u8; 32],
}

impl ManifestSecretMetadata {
    pub(crate) fn from_layout(layout: &mxx_bgg::BggSamplerLayout, identity: [u8; 32]) -> Self {
        Self {
            modulus: layout.modulus.clone(),
            ring_dimension: layout.ring_dimension.clone(),
            secret_dimension: layout.secret_dimension,
            digit_count: layout.digit_count,
            gadget_base: layout.gadget_base.clone(),
            identity,
        }
    }

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

#[derive(Debug, Error, Eq, PartialEq)]
pub enum ExponentRhsPackageError {
    #[error("RHS ciphertext has an invalid shape")]
    InvalidShape,
}

impl ExponentRhsPackage {
    /// Constructs a package from the one fixed GSW ciphertext.
    pub(crate) fn new(gsw_ciphertext: Mat) -> Result<Self, ExponentRhsPackageError> {
        let matrix_type = gsw_ciphertext.matrix_type();
        let rows =
            matrix_type.rows.evaluate(&ParamEnv::default()).ok().and_then(|value| value.to_usize());
        let columns = matrix_type
            .columns
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize());
        if rows != Some(2) || columns.is_none() || columns == Some(0) {
            return Err(ExponentRhsPackageError::InvalidShape);
        }
        Ok(Self { gsw_ciphertext })
    }

    /// Imports the fixed ciphertext after checking its manifest identity and
    /// concrete matrix type. The ciphertext `C` is an ordinary matrix: its
    /// coefficients live in the full ciphertext modulus and therefore it is
    /// not a bounded compact `SmallMatrix` artifact. The bounded object is
    /// the gadget decomposition used by `mul_small_rhs`, which is derived
    /// from this imported ciphertext at the operation boundary.
    pub fn artifact_input(
        production_id: ProductionId,
        manifest: &Manifest,
        names: ExponentRhsPackageArtifactNames,
        expected_source: &mxx_bgg::BggSamplerLayout,
        expected_target: &mxx_bgg::BggSamplerLayout,
    ) -> Result<Self, ExponentArtifactImportError> {
        if manifest.ir_version != mxx_ir_core::encoding::IR_VERSION {
            return Err(ExponentArtifactImportError::InvalidMetadata);
        }
        if manifest.production_id != production_id {
            return Err(ExponentArtifactImportError::ProductionMismatch);
        }
        let artifact = manifest
            .artifacts
            .get(&names.gsw_ciphertext)
            .ok_or(ExponentArtifactImportError::MissingArtifact)?;
        if artifact.confidentiality != ArtifactConfidentiality::Public ||
            artifact.family_shape.is_some()
        {
            return Err(ExponentArtifactImportError::ConfidentialityMismatch);
        }
        mxx_ir_core::artifact::validate_manifest(manifest)
            .map_err(|_| ExponentArtifactImportError::InvalidMetadata)?;
        let modulus = expected_source
            .modulus
            .evaluate(&ParamEnv::default())
            .map_err(|_| ExponentArtifactImportError::MatrixTypeMismatch)?;
        let ring_dimension = expected_source
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(ExponentArtifactImportError::MatrixTypeMismatch)?;
        let target_columns = expected_target
            .secret_dimension
            .checked_mul(expected_target.digit_count)
            .filter(|columns| *columns > 0)
            .ok_or(ExponentArtifactImportError::MatrixTypeMismatch)?;
        let expected = ArtifactType::Matrix(ConcreteMatrixType {
            modulus,
            ring_dimension,
            rows: expected_source.secret_dimension,
            columns: target_columns,
        });
        if artifact.artifact_type != expected {
            return Err(ExponentArtifactImportError::MatrixTypeMismatch);
        }
        let ciphertext = expected_source.ring().artifact_input(
            production_id,
            names.gsw_ciphertext,
            (expected_source.secret_dimension, target_columns),
            ArtifactConfidentiality::Public,
        );
        Self::new(ciphertext).map_err(|_| ExponentArtifactImportError::MatrixTypeMismatch)
    }

    pub(crate) fn gsw_ciphertext(&self) -> &Mat {
        &self.gsw_ciphertext
    }

    /// Returns the public setup-fixed descriptor. The same fixed ciphertext
    /// is intentionally retained because it is an input to public Fuse.
    pub fn public_projection(&self) -> ExponentRhsPackage {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_bgg::{BggPublicKeyCompiler, BggSamplerLayout};
    use mxx_dsl::DslContext;
    use mxx_ir_core::{
        ParamEnv,
        artifact::{ArtifactType, ManifestArtifact, SpecHash},
        node::NodeKind,
    };
    use std::collections::BTreeMap;

    const ARTIFACT_NAME: &str = "fixed-c";

    fn sampler(gadget_base: usize) -> BggSamplerLayout {
        BggSamplerLayout {
            modulus: 97.into(),
            ring_dimension: 4.into(),
            secret_dimension: 2,
            digit_count: 1,
            gadget_base: gadget_base.into(),
        }
    }

    fn manifest_fixture()
    -> (ProductionId, Manifest, ExponentRhsPackageArtifactNames, BggSamplerLayout, BggSamplerLayout)
    {
        let production_id = ProductionId { spec_hash: SpecHash([1; 32]), execution_nonce: [2; 32] };
        let source = sampler(4);
        let target = sampler(8);
        let artifact = ManifestArtifact {
            artifact_type: ArtifactType::Matrix(ConcreteMatrixType {
                modulus: 97.into(),
                ring_dimension: 4,
                rows: source.secret_dimension,
                columns: target.public_key_columns(),
            }),
            family_shape: None,
            confidentiality: ArtifactConfidentiality::Public,
            content_hash: Some([5; 32]),
            layout: None,
        };
        let names = ExponentRhsPackageArtifactNames { gsw_ciphertext: ARTIFACT_NAME.to_owned() };
        let manifest = Manifest {
            ir_version: mxx_ir_core::encoding::IR_VERSION,
            production_id: production_id.clone(),
            artifacts: BTreeMap::from([(ARTIFACT_NAME.to_owned(), artifact)]),
        };
        (production_id, manifest, names, source, target)
    }

    fn import_fixture() -> (ExponentRhsPackage, ProductionId, Manifest) {
        let (production_id, manifest, names, source, target) = manifest_fixture();
        let package = ExponentRhsPackage::artifact_input(
            production_id.clone(),
            &manifest,
            names,
            &source,
            &target,
        )
        .unwrap();
        (package, production_id, manifest)
    }

    #[test]
    fn imports_public_fixed_c_and_preserves_public_artifact_binding() {
        let (package, production_id, _) = import_fixture();
        let NodeKind::Input { artifact: Some(artifact), .. } =
            package.gsw_ciphertext().value_handle().node().kind()
        else {
            panic!("imported fixed C must remain an artifact input")
        };
        assert_eq!(artifact.production_id, production_id);
        assert_eq!(artifact.artifact_name, ARTIFACT_NAME);
        assert_eq!(artifact.confidentiality, ArtifactConfidentiality::Public);
    }

    #[test]
    fn rejects_wrong_source_matrix_schema() {
        let (production_id, manifest, names, _, target) = manifest_fixture();
        let wrong_layout = sampler(16);
        let mut wrong_layout = wrong_layout;
        wrong_layout.ring_dimension = 8.into();
        let wrong_layout = ExponentRhsPackage::artifact_input(
            production_id,
            &manifest,
            names,
            &wrong_layout,
            &target,
        );
        assert!(matches!(wrong_layout, Err(ExponentArtifactImportError::MatrixTypeMismatch)));
    }

    #[test]
    fn rejects_wrong_target_matrix_schema() {
        let (production_id, manifest, names, source, _) = manifest_fixture();
        let wrong_layout = sampler(16);
        let mut wrong_layout = wrong_layout;
        wrong_layout.secret_dimension = 3;
        let wrong_layout = ExponentRhsPackage::artifact_input(
            production_id,
            &manifest,
            names,
            &source,
            &wrong_layout,
        );
        assert!(matches!(wrong_layout, Err(ExponentArtifactImportError::MatrixTypeMismatch)));
    }

    #[test]
    fn rejects_wrong_confidentiality_and_matrix_type() {
        let (production_id, mut manifest, names, source, target) = manifest_fixture();
        manifest.artifacts.get_mut(ARTIFACT_NAME).unwrap().confidentiality =
            ArtifactConfidentiality::Private;
        assert!(matches!(
            ExponentRhsPackage::artifact_input(
                production_id.clone(),
                &manifest,
                names.clone(),
                &source,
                &target,
            ),
            Err(ExponentArtifactImportError::ConfidentialityMismatch)
        ));

        let artifact = manifest.artifacts.get_mut(ARTIFACT_NAME).unwrap();
        artifact.confidentiality = ArtifactConfidentiality::Public;
        artifact.artifact_type = ArtifactType::Matrix(ConcreteMatrixType {
            modulus: 97.into(),
            ring_dimension: 4,
            rows: 2,
            columns: 1,
        });
        assert!(matches!(
            ExponentRhsPackage::artifact_input(production_id, &manifest, names, &source, &target,),
            Err(ExponentArtifactImportError::MatrixTypeMismatch)
        ));
    }

    #[test]
    fn imported_public_fixed_c_feeds_public_compiler_graph() {
        let (package, production_id, manifest) = import_fixture();
        let source = sampler(4);
        let ring = source.ring();
        let compiler = mxx_exponent_lut_public_compiler(&source);
        let fused = compiler.fuse_public(&ring.input("public-a", (2, 2)), &package).unwrap();
        let graph = DslContext::new("rhs-public-import")
            .public_output("fused", fused)
            .unwrap()
            .build()
            .unwrap();
        graph
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production_id, manifest)]),
            )
            .unwrap();
    }

    fn mxx_exponent_lut_public_compiler(
        layout: &BggSamplerLayout,
    ) -> crate::public_key::ExponentLutPublicKeyCompiler {
        crate::public_key::ExponentLutPublicKeyCompiler::new(BggPublicKeyCompiler {
            ring: layout.ring(),
            base: layout.gadget_base.clone(),
            digit_count: layout.digit_count.into(),
        })
    }
}
