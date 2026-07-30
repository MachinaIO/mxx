use mxx_graph_ir::types::{ConcreteMatrixType, InstantiationFrame, NodeId, Port};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct DrawSite {
    pub instantiation_path: Vec<InstantiationFrame>,
    pub node: NodeId,
    pub port: Port,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RecordedValue {
    Matrix { matrix_type: ConcreteMatrixType, bytes: Vec<u8> },
    Trapdoor { matrix_type: ConcreteMatrixType, public_bytes: Vec<u8>, trapdoor_bytes: Vec<u8> },
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TranscriptRecorder {
    entries: BTreeMap<DrawSite, RecordedValue>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TranscriptReplayer {
    entries: BTreeMap<DrawSite, RecordedValue>,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum TranscriptError {
    #[error("transcript draw site is missing: {0:?}")]
    Missing(DrawSite),
    #[error("transcript value kind does not match draw site {0:?}")]
    KindMismatch(DrawSite),
    #[error("transcript draw site was recorded twice: {0:?}")]
    Duplicate(DrawSite),
}

impl TranscriptRecorder {
    pub fn record(&mut self, site: DrawSite, value: RecordedValue) -> Result<(), TranscriptError> {
        if self.entries.insert(site.clone(), value).is_some() {
            Err(TranscriptError::Duplicate(site))
        } else {
            Ok(())
        }
    }

    pub fn into_replayer(self) -> TranscriptReplayer {
        TranscriptReplayer { entries: self.entries }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&DrawSite, &RecordedValue)> {
        self.entries.iter()
    }
}

impl TranscriptReplayer {
    pub fn get(&self, site: &DrawSite) -> Result<&RecordedValue, TranscriptError> {
        self.entries.get(site).ok_or_else(|| TranscriptError::Missing(site.clone()))
    }

    pub fn iter(&self) -> impl Iterator<Item = (&DrawSite, &RecordedValue)> {
        self.entries.iter()
    }
}

pub enum SamplingMode<'a> {
    Fresh,
    Record(&'a mut TranscriptRecorder),
    Replay(&'a TranscriptReplayer),
}
