use super::{
    identity::{BinderKey, ResolvedIntExpr, TrapdoorSourceKey},
    normal_form::{
        FactorIdentity, FactorLayoutIdentity, FactorOwner, NormalFormError, SymbolicFactor, TermId,
    },
};
use mxx_ir_core::types::ConcreteMatrixType;
use std::collections::{BTreeMap, BTreeSet};

/// Full identity used for checked one-way `B*K -> P` registration.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FullRelationKey {
    pub source: FactorOwner,
    pub ordered_indices: Box<[(BinderKey, ResolvedIntExpr)]>,
    pub public: FactorIdentity,
    pub target: FactorIdentity,
    pub matrix_type: Option<ConcreteMatrixType>,
    pub layout: Option<FactorLayoutIdentity>,
    pub trapdoor: Option<TrapdoorSourceKey>,
    pub selector: Option<(Box<FactorIdentity>, Box<[num_bigint::BigUint]>)>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelationRegistration {
    pub key: FullRelationKey,
    pub preimage: FactorIdentity,
    pub target: TermId,
}

#[derive(Clone, Debug, Default)]
pub struct RelationRegistry {
    registrations: BTreeMap<(FactorIdentity, FactorIdentity), Vec<RelationRegistration>>,
    preimage_index: BTreeSet<FactorIdentity>,
}

impl RelationRegistry {
    pub(super) fn reaches_preimage(&self, key: &FactorIdentity) -> bool {
        self.preimage_index.contains(&without_trapdoor(key))
    }

    pub fn register(&mut self, registration: RelationRegistration) -> Result<(), NormalFormError> {
        let normalized_preimage = without_trapdoor(&registration.preimage);
        let key = (registration.key.public.clone(), normalized_preimage.clone());
        self.preimage_index.insert(normalized_preimage);
        let entries = self.registrations.entry(key).or_default();
        if let Some(existing) = entries.iter().find(|entry| entry.key == registration.key) {
            if existing.target != registration.target {
                return Err(NormalFormError::ConflictingRelationTarget { key: registration.key });
            }
            return Ok(());
        }
        entries.push(registration);
        entries
            .sort_by(|left, right| left.key.cmp(&right.key).then(left.target.cmp(&right.target)));
        Ok(())
    }

    pub(super) fn resolve(
        &self,
        public: &SymbolicFactor,
        preimage: &SymbolicFactor,
    ) -> Result<Option<&RelationRegistration>, NormalFormError> {
        let key = (public.key.clone(), without_trapdoor(&preimage.key));
        let candidates = self
            .registrations
            .get(&key)
            .into_iter()
            .flat_map(|entries| entries.iter())
            .filter(|entry| {
                entry.key.source == preimage.key.owner &&
                    entry.key.ordered_indices.as_ref() == public.key.coordinates.as_ref() &&
                    entry.key.public == public.key &&
                    entry.key.layout == public.key.layout &&
                    entry.key.selector ==
                        public
                            .key
                            .selector
                            .clone()
                            .map(|selector| (selector, public.key.selector_mapping.clone())) &&
                    entry.key.matrix_type ==
                        public.matrix_bound.as_ref().map(|bound| bound.matrix_type.clone()) &&
                    entry.key.trapdoor == preimage.key.trapdoor
            })
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return Ok(None);
        }
        let distinct = candidates.iter().map(|entry| entry.key.clone()).collect::<BTreeSet<_>>();
        if distinct.len() > 1 {
            let targets = distinct.into_iter().map(|key| key.target).collect();
            return Err(NormalFormError::AmbiguousRelation { keys: targets });
        }
        Ok(candidates.first().copied())
    }
}

fn without_trapdoor(key: &FactorIdentity) -> FactorIdentity {
    let mut key = key.clone();
    key.trapdoor = None;
    key
}
