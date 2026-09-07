use super::RootBlockAliases;
use mxx_ir_core::{
    ParamEnv, ValidatedGraph,
    artifact::SpecHash,
    encoding,
    graph::NodeHandle,
    types::{ConcreteWireType, WireRef},
};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, VecDeque},
    sync::Arc,
};

struct Entry {
    source: SpecHash,
    bindings: ParamEnv,
    order: Vec<NodeHandle>,
    types: BTreeMap<WireRef, ConcreteWireType>,
    retained: BTreeSet<WireRef>,
    plan: Arc<RootBlockAliases>,
}

thread_local! {
    // Plans contain public graph metadata only. Exact snapshots of the mutable
    // validation fields prevent stale reuse after a caller changes them.
    static PLANS: RefCell<VecDeque<Entry>> = const { RefCell::new(VecDeque::new()) };
}

pub(super) fn get(
    validated: &ValidatedGraph,
    build: impl FnOnce() -> RootBlockAliases,
) -> Arc<RootBlockAliases> {
    let Ok(source) = encoding::spec_hash(&validated.source, &validated.bindings) else {
        return Arc::new(build());
    };
    let scope = validated.root_scope();
    let cached = PLANS.with(|plans| {
        plans
            .borrow()
            .iter()
            .find(|entry| {
                entry.source == source &&
                    entry.bindings == validated.bindings &&
                    entry.order == scope.execution_order &&
                    entry.types == scope.wire_types &&
                    entry.retained == scope.liveness.retained
            })
            .map(|entry| Arc::clone(&entry.plan))
    });
    if let Some(plan) = cached {
        return plan;
    }
    let plan = Arc::new(build());
    PLANS.with(|plans| {
        let mut plans = plans.borrow_mut();
        if plans.len() == 4 {
            plans.pop_front();
        }
        plans.push_back(Entry {
            source,
            bindings: validated.bindings.clone(),
            order: scope.execution_order.clone(),
            types: scope.wire_types.clone(),
            retained: scope.liveness.retained.clone(),
            plan: Arc::clone(&plan),
        });
    });
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::{DslContext, Ring};
    use mxx_ir_core::graph::FrozenGraphScopeId;

    #[test]
    fn root_plan_cache_invalidates_mutable_validation_metadata() {
        let ring = Ring::new(97u64, 8usize);
        let mut graph = DslContext::new("plan-cache-validation-metadata")
            .output("out", ring.input("input", (2, 1)).transpose())
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let initial = get(&graph, RootBlockAliases::default);
        assert!(Arc::ptr_eq(&initial, &get(&graph, RootBlockAliases::default)));
        let hash = encoding::spec_hash(&graph.source, &graph.bindings).unwrap();
        graph.bindings.loop_indices.insert(7, 1.into());
        assert_eq!(hash, encoding::spec_hash(&graph.source, &graph.bindings).unwrap());
        let rebound = get(&graph, RootBlockAliases::default);
        assert!(!Arc::ptr_eq(&initial, &rebound));
        assert!(Arc::ptr_eq(&rebound, &get(&graph, RootBlockAliases::default)));
        let scope = graph.scopes.get_mut(&FrozenGraphScopeId::Root).unwrap();
        let ty = scope
            .wire_types
            .values_mut()
            .find_map(|ty| match ty {
                ConcreteWireType::Matrix(ty) => Some(ty),
                _ => None,
            })
            .unwrap();
        ty.rows += 1;
        let retyped = get(&graph, RootBlockAliases::default);
        assert!(!Arc::ptr_eq(&rebound, &retyped));
        let scope = graph.scopes.get_mut(&FrozenGraphScopeId::Root).unwrap();
        scope.execution_order.swap(0, 1);
        let reordered = get(&graph, RootBlockAliases::default);
        assert!(!Arc::ptr_eq(&retyped, &reordered));
        let scope = graph.scopes.get_mut(&FrozenGraphScopeId::Root).unwrap();
        scope.liveness.retained.insert(*scope.wire_types.keys().next().unwrap());
        assert!(!Arc::ptr_eq(&reordered, &get(&graph, RootBlockAliases::default)));
    }
}
