use mxx_graph_ir::{
    artifact::{Manifest, ProductionId},
    types::ConcreteMatrixType,
};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct ArtifactKey {
    pub production: ProductionId,
    pub name: String,
    pub index: Option<usize>,
}

pub trait ArtifactStore {
    type Error: std::error::Error + Send + Sync + 'static;

    fn load_matrix(&mut self, key: &ArtifactKey) -> Result<Vec<u8>, Self::Error>;
    fn store_matrix(
        &mut self,
        key: ArtifactKey,
        matrix_type: &ConcreteMatrixType,
        bytes: Vec<u8>,
    ) -> Result<(), Self::Error>;
    fn store_manifest(&mut self, manifest: Manifest) -> Result<(), Self::Error>;
}

#[derive(Clone, Debug, Default)]
pub struct MemoryArtifactStore {
    entries: BTreeMap<ArtifactKey, (ConcreteMatrixType, Vec<u8>)>,
    loads: BTreeMap<ArtifactKey, usize>,
    manifests: BTreeMap<ProductionId, Manifest>,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum MemoryArtifactError {
    #[error("artifact does not exist: {0:?}")]
    Missing(ArtifactKey),
}

impl MemoryArtifactStore {
    pub fn insert(&mut self, key: ArtifactKey, matrix_type: ConcreteMatrixType, bytes: Vec<u8>) {
        self.entries.insert(key, (matrix_type, bytes));
    }

    pub fn load_count(&self, key: &ArtifactKey) -> usize {
        self.loads.get(key).copied().unwrap_or(0)
    }

    pub fn manifest(&self, production: &ProductionId) -> Option<&Manifest> {
        self.manifests.get(production)
    }
}

impl ArtifactStore for MemoryArtifactStore {
    type Error = MemoryArtifactError;

    fn load_matrix(&mut self, key: &ArtifactKey) -> Result<Vec<u8>, Self::Error> {
        let value = self
            .entries
            .get(key)
            .map(|(_, bytes)| bytes.clone())
            .ok_or_else(|| MemoryArtifactError::Missing(key.clone()))?;
        *self.loads.entry(key.clone()).or_default() += 1;
        Ok(value)
    }

    fn store_matrix(
        &mut self,
        key: ArtifactKey,
        matrix_type: &ConcreteMatrixType,
        bytes: Vec<u8>,
    ) -> Result<(), Self::Error> {
        self.entries.insert(key, (matrix_type.clone(), bytes));
        Ok(())
    }

    fn store_manifest(&mut self, manifest: Manifest) -> Result<(), Self::Error> {
        self.manifests.insert(manifest.production_id.clone(), manifest);
        Ok(())
    }
}
