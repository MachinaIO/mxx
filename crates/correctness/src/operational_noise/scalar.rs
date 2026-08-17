//! Job-local typed scalar storage used by the DAG bridge.

use super::{
    analysis::AnalysisData, identity::ResolvedIntExpr, normal_form::MatrixValueIdentityId,
};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ScalarId(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ScalarOperation {
    ExtractCoefficient,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ScalarExtractKey {
    pub operation: ScalarOperation,
    pub matrix: MatrixValueIdentityId,
    pub position: ResolvedIntExpr,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScalarEntry {
    pub key: ScalarExtractKey,
    pub analysis: AnalysisData,
}

#[derive(Clone, Debug, Default)]
pub struct ScalarStore {
    entries: Vec<ScalarEntry>,
    index: BTreeMap<ScalarExtractKey, ScalarId>,
}

impl ScalarStore {
    pub fn intern(&mut self, key: ScalarExtractKey, analysis: AnalysisData) -> ScalarId {
        if let Some(id) = self.index.get(&key).copied() {
            let existing = &mut self.entries[id.0 as usize].analysis;
            let _ = existing.merge_from(analysis);
            return id;
        }
        let id = ScalarId(self.entries.len() as u32);
        self.index.insert(key.clone(), id);
        self.entries.push(ScalarEntry { key, analysis });
        id
    }

    pub fn get(&self, id: ScalarId) -> Option<&ScalarEntry> {
        self.entries.get(id.0 as usize)
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}
