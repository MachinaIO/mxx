use super::{
    identity::{BinderKey, ResolvedIntExpr, TrapdoorSourceKey},
    normal_form::{
        FactorIdentity, FactorLayoutIdentity, FactorOwner, NormalFormError, SymbolicFactor, TermId,
    },
};
use mxx_ir_core::types::ConcreteMatrixType;
use std::collections::{BTreeMap, BTreeSet};

/// Canonical relation shape: central operands are a sorted multiset and the
/// noncentral operands form one contiguous ordered word.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct RelationPattern {
    pub required_central: Box<[FactorIdentity]>,
    pub ordered_word: Box<[FactorIdentity]>,
}

impl RelationPattern {
    pub fn new(
        required_central: impl IntoIterator<Item = FactorIdentity>,
        ordered_word: impl IntoIterator<Item = FactorIdentity>,
    ) -> Self {
        let mut required_central = required_central.into_iter().collect::<Vec<_>>();
        required_central.sort();
        Self {
            required_central: required_central.into_boxed_slice(),
            ordered_word: ordered_word.into_iter().collect(),
        }
    }

    pub fn ordered(public: FactorIdentity, preimage: FactorIdentity) -> Self {
        Self::new([], [public, preimage])
    }
}

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
    pub pattern: RelationPattern,
}

pub struct RelationMatch<'a> {
    pub registration: &'a RelationRegistration,
    pub pattern: &'a RelationPattern,
    pub ordered_start: usize,
    pub central_indices: Box<[usize]>,
}

#[derive(Clone, Debug, Default)]
pub struct RelationRegistry {
    registrations: BTreeMap<(FactorIdentity, FactorIdentity), Vec<RelationRegistration>>,
    relation_factor_index: BTreeSet<FactorIdentity>,
    pattern_index: BTreeMap<FactorIdentity, BTreeSet<(FactorIdentity, FactorIdentity)>>,
}

impl RelationRegistry {
    pub(super) fn reaches_preimage(&self, key: &FactorIdentity) -> bool {
        self.relation_factor_index.contains(&without_trapdoor(key))
    }

    pub fn register(&mut self, registration: RelationRegistration) -> Result<(), NormalFormError> {
        let normalized_preimage = without_trapdoor(&registration.preimage);
        let key = (registration.key.public.clone(), normalized_preimage.clone());
        let pattern = registration.pattern.clone();
        self.relation_factor_index.insert(normalized_preimage.clone());
        self.relation_factor_index.insert(without_trapdoor(&registration.key.public));
        let entries = self.registrations.entry(key.clone()).or_default();
        if let Some(existing) = entries.iter().find(|entry| entry.key == registration.key) {
            if existing.target != registration.target {
                return Err(NormalFormError::ConflictingRelationTarget { key: registration.key });
            }
            return Ok(());
        }
        entries.push(registration);
        entries
            .sort_by(|left, right| left.key.cmp(&right.key).then(left.target.cmp(&right.target)));
        for factor in pattern.required_central.iter().chain(pattern.ordered_word.iter()) {
            let factor = without_trapdoor(factor);
            self.relation_factor_index.insert(factor.clone());
            self.pattern_index.entry(factor).or_default().insert(key.clone());
        }
        Ok(())
    }

    pub fn register_pattern(
        &mut self,
        pattern: RelationPattern,
        mut registration: RelationRegistration,
    ) -> Result<(), NormalFormError> {
        registration.pattern = pattern;
        self.register(registration)
    }

    pub(super) fn resolve_pattern<'a>(
        &'a self,
        central: &[SymbolicFactor],
        ordered: &[SymbolicFactor],
    ) -> Result<Option<RelationMatch<'a>>, NormalFormError> {
        let mut matches = Vec::new();
        let mut candidate_registration_keys = BTreeSet::new();
        for factor in central.iter().chain(ordered.iter()) {
            if let Some(keys) = self.pattern_index.get(&without_trapdoor(&factor.key)) {
                candidate_registration_keys.extend(keys.iter().cloned());
            }
        }
        for registration_key in candidate_registration_keys {
            let Some(entries) = self.registrations.get(&registration_key) else { continue };
            for entry in entries.iter() {
                let pattern = &entry.pattern;
                let Some(central_indices) = consume_central(pattern, central) else { continue };
                let word_len = pattern.ordered_word.len();
                if word_len > ordered.len() {
                    continue;
                }
                for start in 0..=ordered.len().saturating_sub(word_len) {
                    let candidate = &ordered[start..start + word_len];
                    if candidate.iter().zip(pattern.ordered_word.iter()).all(
                        |(factor, expected)| {
                            without_trapdoor(&factor.key) == without_trapdoor(expected)
                        },
                    ) {
                        let mut consumed_factors =
                            central_indices.iter().filter_map(|index| central.get(*index));
                        let public = candidate
                            .iter()
                            .find(|factor| factor.key == entry.key.public)
                            .or_else(|| {
                                consumed_factors
                                    .clone()
                                    .find(|factor| factor.key == entry.key.public)
                            });
                        let preimage = candidate
                            .iter()
                            .find(|factor| {
                                without_trapdoor(&factor.key) == without_trapdoor(&entry.preimage)
                            })
                            .or_else(|| {
                                consumed_factors.find(|factor| {
                                    if entry.preimage.trapdoor.is_some() {
                                        factor.key == entry.preimage
                                    } else {
                                        without_trapdoor(&factor.key) ==
                                            without_trapdoor(&entry.preimage)
                                    }
                                })
                            });
                        let (Some(public), Some(preimage)) = (public, preimage) else {
                            continue;
                        };
                        if self.matches_registration(entry, public, preimage) {
                            matches.push((start, entry, pattern, central_indices.clone()));
                        }
                    }
                }
            }
        }
        matches.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.key.cmp(&right.1.key)));
        let Some((ordered_start, registration, pattern, central_indices)) = matches.first() else {
            return Ok(None);
        };
        let same_start =
            matches.iter().filter(|candidate| candidate.0 == *ordered_start).collect::<Vec<_>>();
        let distinct =
            same_start.iter().map(|candidate| candidate.1.key.clone()).collect::<BTreeSet<_>>();
        if distinct.len() > 1 {
            return Err(NormalFormError::AmbiguousRelation {
                keys: distinct.into_iter().map(|key| key.target).collect(),
            });
        }
        Ok(Some(RelationMatch {
            registration,
            pattern,
            ordered_start: *ordered_start,
            central_indices: central_indices.clone(),
        }))
    }

    fn matches_registration(
        &self,
        entry: &RelationRegistration,
        public: &SymbolicFactor,
        preimage: &SymbolicFactor,
    ) -> bool {
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
            entry
                .key
                .matrix_type
                .as_ref()
                .is_none_or(|matrix_type| matrix_type == &public.matrix_type) &&
            entry.key.trapdoor == preimage.key.trapdoor
    }
}

fn consume_central(pattern: &RelationPattern, central: &[SymbolicFactor]) -> Option<Box<[usize]>> {
    let mut used = BTreeSet::new();
    let mut indices = Vec::with_capacity(pattern.required_central.len());
    for expected in pattern.required_central.iter() {
        let index = central.iter().enumerate().find_map(|(index, factor)| {
            if used.contains(&index) {
                return None;
            }
            let matches = if expected.trapdoor.is_some() {
                factor.key == *expected
            } else {
                without_trapdoor(&factor.key) == without_trapdoor(expected)
            };
            matches.then_some(index)
        })?;
        used.insert(index);
        indices.push(index);
    }
    Some(indices.into_boxed_slice())
}

fn without_trapdoor(key: &FactorIdentity) -> FactorIdentity {
    let mut key = key.clone();
    key.trapdoor = None;
    key
}
