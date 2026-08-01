use mxx_ir_core::types::{ConcreteMatrixType, InstantiationFrame, NodeId, Port};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct DrawSite {
    pub instantiation_path: Vec<InstantiationFrame>,
    pub node: NodeId,
    pub port: Port,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
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
        match self.entries.entry(site.clone()) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(value);
                Ok(())
            }
            std::collections::btree_map::Entry::Occupied(_) => {
                Err(TranscriptError::Duplicate(site))
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn site(node: u64) -> DrawSite {
        DrawSite { instantiation_path: Vec::new(), node: NodeId(node), port: Port(0) }
    }

    fn value() -> RecordedValue {
        RecordedValue::Matrix {
            matrix_type: ConcreteMatrixType {
                modulus: 17.into(),
                ring_dimension: 8,
                rows: 1,
                columns: 1,
            },
            bytes: vec![1, 2, 3],
        }
    }

    #[test]
    fn recorder_rejects_duplicate_draw_sites_without_replacing_the_original() {
        let mut recorder = TranscriptRecorder::default();
        recorder.record(site(1), value()).expect("first record");
        let replacement = RecordedValue::Matrix {
            matrix_type: ConcreteMatrixType {
                modulus: 17.into(),
                ring_dimension: 8,
                rows: 1,
                columns: 1,
            },
            bytes: vec![9],
        };
        assert_eq!(recorder.record(site(1), replacement), Err(TranscriptError::Duplicate(site(1))));
        assert_eq!(recorder.into_replayer().get(&site(1)).cloned(), Ok(value()));
    }

    #[test]
    fn replayer_reports_the_exact_missing_draw_site() {
        let missing = site(7);
        assert_eq!(
            TranscriptReplayer::default().get(&missing),
            Err(TranscriptError::Missing(missing))
        );
    }
}
