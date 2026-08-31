//! Opaque identity handles used by abstract state and future relation tables.

use crate::StageId;
use mxx_ir_core::{FrozenGraphScopeId, IndexMap, WireRef};
use num_bigint::BigInt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ValueId(pub(crate) u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct FamilyViewId(pub(crate) u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct SourceId(pub(crate) u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct SelectorId(pub(crate) u32);

/// The structural parameters of a universal gadget source.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct GadgetDescriptor {
    pub modulus: BigInt,
    pub ring_dimension: usize,
    pub rows: usize,
    pub columns: usize,
    pub base: BigInt,
    pub digit_count: usize,
    pub small: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct ValueKey {
    pub stage: StageId,
    pub scope: FrozenGraphScopeId,
    /// Structural call/loop occurrence.  A frozen child scope is shared by
    /// all calls, so `(scope, wire)` alone is not a value identity: the
    /// occurrence coordinate distinguishes each sequential or parallel use.
    pub occurrence: Vec<String>,
    pub wire: WireRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct FamilyViewKey {
    /// Exact planned values making up the view.  The vector is ordered in
    /// row-major coordinate order; no numeric bound participates in identity.
    pub values: Vec<ValueId>,
    pub parents: Vec<FamilyViewId>,
    pub shape: Vec<usize>,
    pub maps: Vec<IndexMap>,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct SelectorKey(pub Vec<ValueId>);

#[derive(Default)]
pub(crate) struct Interners {
    pub values: HashMap<ValueKey, ValueId>,
    pub views: HashMap<FamilyViewKey, FamilyViewId>,
    pub selectors: HashMap<SelectorKey, SelectorId>,
}

impl Interners {
    pub(crate) fn intern_view(
        &mut self,
        values: Vec<ValueId>,
        shape: Vec<usize>,
        maps: &[IndexMap],
    ) -> FamilyViewId {
        let key = FamilyViewKey {
            values,
            parents: Vec::new(),
            shape,
            maps: maps.iter().map(IndexMap::normalize).collect(),
        };
        let next = FamilyViewId(self.views.len() as u32);
        *self.views.entry(key).or_insert(next)
    }

    pub(crate) fn intern_composed_view(
        &mut self,
        parents: Vec<FamilyViewId>,
        shape: Vec<usize>,
        maps: &[IndexMap],
    ) -> FamilyViewId {
        let key = FamilyViewKey {
            values: Vec::new(),
            parents,
            shape,
            maps: maps.iter().map(IndexMap::normalize).collect(),
        };
        let next = FamilyViewId(self.views.len() as u32);
        *self.views.entry(key).or_insert(next)
    }

    pub(crate) fn intern_selector(&mut self, values: Vec<ValueId>) -> SelectorId {
        let next = SelectorId(self.selectors.len() as u32);
        *self.selectors.entry(SelectorKey(values)).or_insert(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::expr::{IndexExpr, IndexMap};

    #[test]
    fn normalized_maps_and_exact_selector_aliases_are_interned() {
        let mut interners = Interners::default();
        let map_a = IndexMap::new(vec![IndexExpr::constant(1)]);
        let map_b = map_a.normalize();
        assert_eq!(
            interners.intern_view(vec![ValueId(1)], vec![2], &[map_a]),
            interners.intern_view(vec![ValueId(1)], vec![2], &[map_b])
        );
        assert_eq!(
            interners.intern_selector(vec![ValueId(2)]),
            interners.intern_selector(vec![ValueId(2)])
        );
        assert_ne!(
            interners.intern_selector(vec![ValueId(3)]),
            interners.intern_selector(vec![ValueId(4)])
        );
    }
}
