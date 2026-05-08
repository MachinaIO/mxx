use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use crate::matrix::PolyMatrix;

/// Writes decoder artifacts under caller-defined identifiers.
pub trait DecoderArtifactSink {
    fn write_artifact(&mut self, id: &str, bytes: &[u8]);

    fn write_matrix<M: PolyMatrix>(&mut self, id: &str, matrix: &M) {
        self.write_artifact(id, &matrix.to_compact_bytes());
    }
}

/// Reads decoder artifacts produced by a matching sink.
pub trait DecoderArtifactSource {
    fn read_artifact(&self, id: &str) -> Vec<u8>;

    fn read_matrix<M: PolyMatrix>(
        &self,
        params: &<M::P as crate::poly::Poly>::Params,
        id: &str,
    ) -> M {
        M::from_compact_bytes(params, &self.read_artifact(id))
    }
}

/// In-memory artifact store used by unit tests and small functional-key style
/// callers.
#[derive(Debug, Clone, Default)]
pub struct InMemoryDecoderArtifacts {
    artifacts: HashMap<String, Vec<u8>>,
}

impl InMemoryDecoderArtifacts {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn into_artifacts(self) -> HashMap<String, Vec<u8>> {
        self.artifacts
    }
}

impl DecoderArtifactSink for InMemoryDecoderArtifacts {
    fn write_artifact(&mut self, id: &str, bytes: &[u8]) {
        self.artifacts.insert(id.to_owned(), bytes.to_vec());
    }
}

impl DecoderArtifactSource for InMemoryDecoderArtifacts {
    fn read_artifact(&self, id: &str) -> Vec<u8> {
        self.artifacts.get(id).unwrap_or_else(|| panic!("missing decoder artifact {id}")).clone()
    }
}

/// Compact vector-backed artifact store for AKY24-style functional keys.
#[derive(Debug, Clone, Default)]
pub struct VecDecoderArtifacts {
    artifacts: Vec<Vec<u8>>,
}

impl VecDecoderArtifacts {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_matrix<M: PolyMatrix>(&mut self, matrix: &M) {
        self.artifacts.push(matrix.to_compact_bytes());
    }

    pub fn matrix<M: PolyMatrix>(
        &self,
        params: &<M::P as crate::poly::Poly>::Params,
        idx: usize,
    ) -> M {
        M::from_compact_bytes(
            params,
            self.artifacts
                .get(idx)
                .unwrap_or_else(|| panic!("missing decoder artifact index {idx}")),
        )
    }

    pub fn into_vec(self) -> Vec<Vec<u8>> {
        self.artifacts
    }
}

/// Directory-backed artifacts for DiamondIO-style persisted obfuscation state.
#[derive(Debug, Clone)]
pub struct DirectoryDecoderArtifacts {
    dir_path: PathBuf,
    file_prefix: String,
}

impl DirectoryDecoderArtifacts {
    pub fn new(dir_path: impl AsRef<Path>, file_prefix: impl Into<String>) -> Self {
        Self { dir_path: dir_path.as_ref().to_path_buf(), file_prefix: file_prefix.into() }
    }

    fn artifact_path(&self, id: &str) -> PathBuf {
        self.dir_path.join(format!("{}_{}.matrixbin", self.file_prefix, id))
    }
}

impl DecoderArtifactSink for DirectoryDecoderArtifacts {
    fn write_artifact(&mut self, id: &str, bytes: &[u8]) {
        fs::write(self.artifact_path(id), bytes)
            .unwrap_or_else(|err| panic!("failed to write decoder artifact {id}: {err}"));
    }
}

impl DecoderArtifactSource for DirectoryDecoderArtifacts {
    fn read_artifact(&self, id: &str) -> Vec<u8> {
        fs::read(self.artifact_path(id))
            .unwrap_or_else(|err| panic!("failed to read decoder artifact {id}: {err}"))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DecoderArtifactSink, DecoderArtifactSource, DirectoryDecoderArtifacts,
        InMemoryDecoderArtifacts,
    };

    #[test]
    fn in_memory_artifacts_roundtrip_bytes() {
        let mut artifacts = InMemoryDecoderArtifacts::new();
        artifacts.write_artifact("decoder_0", &[1, 2, 3]);
        assert_eq!(artifacts.read_artifact("decoder_0"), vec![1, 2, 3]);
    }

    #[test]
    fn directory_artifacts_roundtrip_bytes() {
        let dir = tempfile::tempdir().expect("temporary decoder artifact directory");
        let mut sink = DirectoryDecoderArtifacts::new(dir.path(), "test_decoder");
        sink.write_artifact("decoder_1", &[4, 5, 6]);
        let source = DirectoryDecoderArtifacts::new(dir.path(), "test_decoder");
        assert_eq!(source.read_artifact("decoder_1"), vec![4, 5, 6]);
    }
}
