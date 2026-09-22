//! Immutable preparation-local access to production GPU lowering metadata.
//!
//! The graph borrow prevents mutation while the lowering snapshot is in use.
//! Global cache validation remains authoritative at construction, rather than
//! repeating its whole-graph hash and validation comparisons for every site.

use super::{GpuEffectiveSiteMetadata, RootBlockAliases, root_block_aliases};
use crate::backend::PlannedOperandMetadata;
use mxx_ir_core::{
    ParamEnv, ValidatedGraph, ValidatedScope,
    graph::{FrozenGraphScopeId, GraphScope},
    node::NodeKind,
    types::{ConcreteWireType, NodeId, Port, WireRef},
};
use std::{collections::BTreeMap, sync::Arc};

pub(crate) struct GpuScopeLoweringCache<'a> {
    validated: &'a ValidatedGraph,
    capture_trace: bool,
    lowerings: BTreeMap<FrozenGraphScopeId, GpuScopeLowering<'a>>,
    #[cfg(test)]
    constructions: usize,
}

impl<'a> GpuScopeLoweringCache<'a> {
    pub(crate) fn new(validated: &'a ValidatedGraph, capture_trace: bool) -> Self {
        Self {
            validated,
            capture_trace,
            lowerings: BTreeMap::new(),
            #[cfg(test)]
            constructions: 0,
        }
    }

    pub(crate) fn get(
        &mut self,
        scope_id: &FrozenGraphScopeId,
    ) -> Result<&GpuScopeLowering<'a>, String> {
        if !self.lowerings.contains_key(scope_id) {
            let lowering =
                GpuScopeLowering::new_with_trace(self.validated, scope_id, self.capture_trace)?;
            #[cfg(test)]
            {
                self.constructions = self.constructions.saturating_add(1);
            }
            self.lowerings.insert(scope_id.clone(), lowering);
        }
        Ok(self.lowerings.get(scope_id).expect("inserted GPU lowering"))
    }

    #[cfg(test)]
    pub(crate) fn construction_count(&self) -> usize {
        self.constructions
    }
}

pub(crate) struct GpuScopeLowering<'a> {
    scope: &'a GraphScope,
    checked: &'a ValidatedScope,
    scope_id: FrozenGraphScopeId,
    aliases: Arc<RootBlockAliases>,
}

impl<'a> GpuScopeLowering<'a> {
    pub(crate) fn new(
        validated: &'a ValidatedGraph,
        scope_id: &FrozenGraphScopeId,
    ) -> Result<Self, String> {
        Self::new_with_trace(validated, scope_id, false)
    }

    pub(crate) fn new_with_trace(
        validated: &'a ValidatedGraph,
        scope_id: &FrozenGraphScopeId,
        capture_trace: bool,
    ) -> Result<Self, String> {
        let scope = validated.source.scope(scope_id).ok_or("missing GPU scope")?;
        let checked = validated.scope(scope_id).ok_or("missing GPU scope")?;
        Ok(Self {
            scope,
            checked,
            scope_id: scope_id.clone(),
            aliases: root_block_aliases(validated, scope_id, capture_trace),
        })
    }

    pub(super) fn aliases(&self) -> &Arc<RootBlockAliases> {
        &self.aliases
    }

    pub(super) fn metadata(&self, node: NodeId) -> Result<GpuEffectiveSiteMetadata, String> {
        let checked = self.checked;
        let aliases = Arc::clone(&self.aliases);
        let output_types = if let Some((_, _, groups)) = aliases.compact_products.get(&node) {
            groups
                .iter()
                .map(|aliases| {
                    aliases
                        .first()
                        .and_then(|wire| checked.wire_types.get(wire))
                        .cloned()
                        .ok_or_else(|| "missing fused output type".to_owned())
                })
                .collect::<Result<Vec<_>, _>>()?
        } else if let Some(outputs) = aliases.tensor_row_sum_groups.get(&node) {
            outputs
                .iter()
                .map(|output| {
                    checked
                        .wire_types
                        .get(&WireRef { node: *output, port: Port(0) })
                        .cloned()
                        .ok_or_else(|| "missing tensor row-sum output type".to_owned())
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            checked
                .wire_types
                .range(WireRef { node, port: Port(0) }..=WireRef { node, port: Port(u32::MAX) })
                .map(|(_, ty)| ty.clone())
                .collect()
        };
        #[cfg(feature = "gpu")]
        let effective_identity = aliases.calibration.get(&node).cloned().transpose()?.flatten();
        #[cfg(not(feature = "gpu"))]
        let effective_identity = None;
        Ok(GpuEffectiveSiteMetadata { aliases, output_types, effective_identity })
    }

    pub(crate) fn site_metadata(
        &self,
        node: NodeId,
    ) -> Result<(Vec<ConcreteWireType>, Option<[u8; 32]>), String> {
        let metadata = self.metadata(node)?;
        Ok((metadata.output_types, metadata.effective_identity))
    }

    /// Prove the public/secret association from the validated producer graph,
    /// without observing not-yet-executed capture output allocations.
    pub(crate) fn is_trapdoor_public_projection(&self, public: WireRef, trapdoor: WireRef) -> bool {
        let aliases = self.alias_facts();
        let origin = |mut wire: WireRef| {
            for _ in 0..=aliases.len() {
                match aliases.get(&wire) {
                    Some(source) if *source != wire => wire = *source,
                    _ => return Some(wire),
                }
            }
            None
        };
        let (Some(public), Some(trapdoor)) = (origin(public), origin(trapdoor)) else {
            return false;
        };
        let Some(producer) = self.scope.node(public.node) else {
            return false;
        };
        if public.port != Port(0) {
            return false;
        }
        match producer.kind() {
            NodeKind::TrapdoorSample { .. } => {
                public.node == trapdoor.node && trapdoor.port == Port(1)
            }
            NodeKind::TrapdoorPublic => {
                self.scope
                    .arguments(producer)
                    .and_then(|arguments| arguments.first().copied())
                    .and_then(origin) ==
                    Some(trapdoor)
            }
            _ => false,
        }
    }

    pub(crate) fn alias_facts(&self) -> BTreeMap<WireRef, WireRef> {
        let aliases = &self.aliases;
        let mut facts = BTreeMap::new();
        for pairs in aliases.concats.values() {
            for (output, source) in pairs {
                facts.insert(*output, *source);
            }
        }
        for (_, _, groups) in aliases.compact_products.values() {
            for outputs in groups {
                // Each result block owns a new product allocation. Only duplicate
                // views of that same result may alias; multiplicands never do.
                if let Some((owner, views)) = outputs.split_first() {
                    for view in views {
                        facts.insert(*view, *owner);
                    }
                }
            }
        }
        facts
    }

    #[cfg(any(feature = "gpu", test))]
    pub(crate) fn fused_result_owners(&self) -> BTreeMap<WireRef, Vec<WireRef>> {
        self.aliases
            .compact_products
            .iter()
            .map(|(node, (_, _, groups))| {
                (
                    WireRef { node: *node, port: Port(0) },
                    groups.iter().filter_map(|group| group.first().copied()).collect(),
                )
            })
            .chain(self.aliases.tensor_row_sum_groups.iter().map(|(leader, outputs)| {
                (
                    WireRef { node: *leader, port: Port(0) },
                    outputs.iter().map(|node| WireRef { node: *node, port: Port(0) }).collect(),
                )
            }))
            .collect()
    }

    #[cfg(any(feature = "gpu", test))]
    pub(crate) fn is_fused_follower(&self, node: NodeId) -> bool {
        self.aliases.tensor_row_sum_leaders.get(&node).is_some_and(|leader| *leader != node)
    }

    #[cfg(any(feature = "gpu", test))]
    pub(crate) fn fused_operation(
        &self,
        node: NodeId,
    ) -> Option<crate::gpu_column_policy::FusedWarmupOperation> {
        use crate::gpu_column_policy::FusedWarmupOperation;
        let aliases = &self.aliases;
        if let Some(plan) = aliases.row_sums.get(&node) {
            Some(if plan.tensor_operands.is_some() {
                FusedWarmupOperation::TensorRowSum
            } else {
                FusedWarmupOperation::RowSum
            })
        } else if aliases.compact_products.contains_key(&node) {
            Some(FusedWarmupOperation::CompactProduct)
        } else if aliases.decompositions.contains_key(&node) {
            Some(FusedWarmupOperation::Decompose)
        } else if aliases.adds.contains_key(&node) {
            Some(FusedWarmupOperation::RowBlockAdd)
        } else {
            None
        }
    }

    pub(crate) fn effective_inputs(
        &self,
        node: NodeId,
        bindings: &ParamEnv,
    ) -> Result<crate::backend::GpuEffectiveInputs, String> {
        let scope = self.scope;
        let scope_id = &self.scope_id;
        let metadata = self.metadata(node)?;
        let origins = canonical_gpu_effective_source_wires(scope, node, &metadata.aliases)?;
        // A preimage fed by GadgetTrapdoor is lowered by production to the
        // target-only fixed gadget decomposition path.  The public matrix and
        // trapdoor handle are validation/runtime metadata, not operands of that
        // kernel. Keep the effective request identical to
        // `fixed_gadget_decompose_batch`, otherwise warmup would profile a
        // sampled-trapdoor preimage call and fixed dispatch would receive stale
        // source/layout metadata.
        if matches!(
            scope.node(node).map(|handle| handle.kind()),
            Some(NodeKind::PreimageSample { .. })
        ) && origins
            .get(1)
            .and_then(|wire| scope.node(wire.node))
            .is_some_and(|producer| matches!(producer.kind(), NodeKind::GadgetTrapdoor { .. }))
        {
            let target = *origins.get(2).ok_or("preimage node has no target operand")?;
            let declaration = scope
                .node(target.node)
                .and_then(|producer| producer.output_types().get(target.port.0 as usize))
                .ok_or("missing preimage target declaration")?;
            let ty = mxx_ir_core::concretize_wire_type(declaration, bindings, scope_id, node)
                .map_err(|error| error.to_string())?;
            let matrix = ty.matrix_type();
            let layout = crate::backend::PlannedLayoutMetadata {
                layout_id: None,
                rows: matrix.map_or(0, |matrix| matrix.rows),
                columns: matrix.map_or(0, |matrix| matrix.columns),
                ring_dimension: matrix.map_or(0, |matrix| matrix.ring_dimension),
                representation: format!("{ty:?}"),
            };
            return Ok(crate::backend::GpuEffectiveInputs {
                origins: vec![target],
                source_layouts: vec![layout.clone()],
                operands: vec![PlannedOperandMetadata::RowBlock {
                    logical_operand: 0,
                    block_index: 0,
                    origin: target,
                    layout,
                }],
                row_groups: Vec::new(),
                row_sum_groups: Vec::new(),
            });
        }
        let mut result = crate::backend::GpuEffectiveInputs {
            origins: origins.clone(),
            row_groups: metadata
                .aliases
                .row_sums
                .get(&node)
                .map(|plan| plan.rows.clone())
                .unwrap_or_default(),
            row_sum_groups: metadata
                .aliases
                .tensor_row_sum_leaders
                .get(&node)
                .and_then(|leader| metadata.aliases.tensor_row_sum_groups.get(leader))
                .map(|outputs| {
                    outputs
                        .iter()
                        .map(|output| metadata.aliases.row_sums[output].rows.clone())
                        .collect()
                })
                .unwrap_or_default(),
            ..Default::default()
        };
        let compact = metadata.aliases.compact_products.contains_key(&node);
        let blocks = if compact || metadata.aliases.adds.contains_key(&node) {
            origins.len().saturating_sub(1)
        } else if metadata.aliases.decompositions.contains_key(&node) {
            origins.len()
        } else {
            0
        };
        for (index, origin) in origins.iter().enumerate() {
            let declaration = scope
                .node(origin.node)
                .and_then(|producer| producer.output_types().get(origin.port.0 as usize))
                .ok_or("missing effective input declaration")?;
            let ty = mxx_ir_core::concretize_wire_type(declaration, bindings, scope_id, node)
                .map_err(|error| error.to_string())?;
            let matrix = ty.matrix_type();
            let layout = crate::backend::PlannedLayoutMetadata {
                layout_id: None,
                rows: matrix.map_or(0, |matrix| matrix.rows),
                columns: matrix.map_or(0, |matrix| matrix.columns),
                ring_dimension: matrix.map_or(0, |matrix| matrix.ring_dimension),
                representation: format!("{ty:?}"),
            };
            if index < blocks {
                result.operands.push(PlannedOperandMetadata::RowBlock {
                    logical_operand: 0,
                    block_index: index,
                    origin: *origin,
                    layout: layout.clone(),
                });
            } else if compact {
                result.operands.push(PlannedOperandMetadata::CompactRhs {
                    logical_operand: 1,
                    origin: *origin,
                    layout: layout.clone(),
                });
            }
            if !compact || index < blocks {
                result.source_layouts.push(layout);
            }
        }
        Ok(result)
    }
}

pub(super) fn canonical_gpu_effective_source_wires(
    scope: &GraphScope,
    node: NodeId,
    aliases: &RootBlockAliases,
) -> Result<Vec<WireRef>, String> {
    let concat_arguments = |concat: NodeId| {
        let handle = scope.node(concat).ok_or_else(|| "missing fused concat".to_owned())?;
        scope.arguments(handle).ok_or_else(|| "missing fused concat arguments".to_owned())
    };
    if let Some((concat, rhs, _)) = aliases.compact_products.get(&node) {
        let mut wires = concat_arguments(*concat)?.to_vec();
        wires.push(*rhs);
        return Ok(wires);
    }
    if let Some(concat) = aliases.decompositions.get(&node) {
        return Ok(concat_arguments(*concat)?.to_vec());
    }
    if let Some((concat, right)) = aliases.adds.get(&node) {
        let mut wires = concat_arguments(*concat)?.to_vec();
        wires.push(*right);
        return Ok(wires);
    }
    if let Some(row_sum) = aliases.row_sums.get(&node) {
        return Ok(row_sum
            .tensor_operands
            .map(|operands| operands.to_vec())
            .unwrap_or_else(|| vec![row_sum.source]));
    }
    scope
        .node(node)
        .and_then(|handle| scope.arguments(handle))
        .map(|arguments| arguments.to_vec())
        .ok_or_else(|| "missing GPU node arguments".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::{
        gpu_alias_facts, gpu_effective_inputs, gpu_effective_site_metadata,
        gpu_fused_operation_for_site, gpu_fused_result_owners,
    };
    use mxx_dsl::{DslContext, Int, Mat, Ring, parallel};
    use mxx_ir_core::node::{ConcatAxis, IndexRange};

    #[test]
    fn test_gpu_preimage_public_provenance_rejects_another_trapdoor() {
        let ring = Ring::new(97u64, 8usize);
        let first = ring.sample_trapdoor(1, 1, 2, 7, 1000);
        let second = ring.sample_trapdoor(1, 1, 2, 7, 1000);
        let graph = DslContext::new("preimage-public-provenance")
            .output("first-public", first.public_matrix())
            .unwrap()
            .transferred_trapdoor_output("first-secret", first)
            .unwrap()
            .output("second-public", second.public_matrix())
            .unwrap()
            .transferred_trapdoor_output("second-secret", second)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let lowering = GpuScopeLowering::new(&graph, &FrozenGraphScopeId::Root).unwrap();
        let nodes = lowering
            .scope
            .nodes()
            .iter()
            .enumerate()
            .filter_map(|(index, node)| {
                matches!(node.kind(), NodeKind::TrapdoorSample { .. })
                    .then_some(NodeId(index as u64))
            })
            .collect::<Vec<_>>();
        assert_eq!(nodes.len(), 2);
        let public = WireRef { node: nodes[0], port: Port(0) };
        assert!(
            lowering
                .is_trapdoor_public_projection(public, WireRef { node: nodes[0], port: Port(1) })
        );
        assert!(
            !lowering
                .is_trapdoor_public_projection(public, WireRef { node: nodes[1], port: Port(1) })
        );
        assert!(!lowering.is_trapdoor_public_projection(public, public));
    }

    #[test]
    fn test_gpu_scope_lowering_cache_builds_once_per_scope() {
        let nested = parallel(4usize, |index| Ok(index + Int::constant(1))).unwrap();
        let graph = DslContext::new("snapshot-cache-count")
            .output("out", nested.at(0))
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let root = FrozenGraphScopeId::Root;
        let mut cache = GpuScopeLoweringCache::new(&graph, false);
        let root_nodes = graph.root_scope().execution_order.len();
        // Fixed dispatch asks for the lowering at every node. The execution
        // cache must still construct one immutable snapshot for the scope.
        for _ in 0..root_nodes {
            cache.get(&root).unwrap();
        }
        assert_eq!(cache.construction_count(), 1);
        for scope_id in graph.scopes.keys() {
            cache.get(scope_id).unwrap();
        }
        assert_eq!(cache.construction_count(), graph.scopes.len());
    }

    fn check_scope_snapshots(graph: &ValidatedGraph) {
        for (scope_id, checked) in &graph.scopes {
            let snapshot = GpuScopeLowering::new(graph, scope_id).unwrap();
            assert_eq!(snapshot.alias_facts(), gpu_alias_facts(graph, scope_id));
            assert_eq!(snapshot.fused_result_owners(), gpu_fused_result_owners(graph, scope_id));
            for (index, _) in checked.execution_order.iter().enumerate() {
                let node = NodeId(index as u64);
                assert_eq!(
                    snapshot.site_metadata(node).unwrap(),
                    gpu_effective_site_metadata(graph, scope_id, node).unwrap()
                );
                assert_eq!(
                    snapshot.effective_inputs(node, &graph.bindings).unwrap(),
                    gpu_effective_inputs(graph, scope_id, node, &graph.bindings).unwrap()
                );
                assert_eq!(
                    snapshot.fused_operation(node),
                    gpu_fused_operation_for_site(graph, scope_id, node)
                );
                // Keep the prior full-scan behavior as an independent test
                // oracle for ordinary ports, including non-matrix outputs.
                if !snapshot.aliases.compact_products.contains_key(&node) {
                    let expected = checked
                        .wire_types
                        .iter()
                        .filter(|(wire, _)| wire.node == node)
                        .map(|(_, ty)| ty.clone())
                        .collect::<Vec<_>>();
                    assert_eq!(snapshot.site_metadata(node).unwrap().0, expected);
                }
                assert!(Arc::ptr_eq(&snapshot.aliases, &snapshot.metadata(node).unwrap().aliases));
            }
        }
    }

    #[test]
    fn test_gpu_scope_snapshot_matches_single_site_apis_and_fusions() {
        let ring = Ring::new(97u64, 8usize);
        let left = ring.input("left", (1, 2));
        let right = ring.input("right", (2, 2));
        let blocks = Mat::concat(ConcatAxis::Rows, vec![left.clone(), right.clone()]);
        let reversed = Mat::concat(ConcatAxis::Rows, vec![right, left]);
        let add = DslContext::new("snapshot-row-block-add")
            .output("out", blocks + reversed)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let snapshot = GpuScopeLowering::new(&add, &FrozenGraphScopeId::Root).unwrap();
        assert!(!snapshot.aliases.adds.is_empty(), "exercise fused row-block inputs");
        check_scope_snapshots(&add);

        let tensor = ring.input("tensor-left", (2, 1)).tensor(ring.input("tensor-right", (2, 1)));
        let row = |index: usize| {
            tensor
                .clone()
                .slice(Some(IndexRange { start: index.into(), end: (index + 1).into() }), None)
        };
        let sum = DslContext::new("snapshot-tensor-row-sum")
            .output("out", Mat::concat(ConcatAxis::Rows, vec![row(0), row(1) + row(2), row(3)]))
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        assert!(
            !GpuScopeLowering::new(&sum, &FrozenGraphScopeId::Root)
                .unwrap()
                .aliases
                .row_sums
                .is_empty(),
            "exercise tensor row groups"
        );
        check_scope_snapshots(&sum);

        let input =
            Mat::concat(ConcatAxis::Rows, vec![ring.input("a", (1, 3)), ring.input("b", (2, 3))]);
        let wide =
            Mat::concat(ConcatAxis::Rows, vec![ring.input("x", (1, 6)), ring.input("y", (2, 6))]);
        let product = input.decompose(16, 2).mul_small_rhs(wide);
        let compact = DslContext::new("snapshot-compact-outputs")
            .output(
                "first",
                product.clone().slice(Some(IndexRange { start: 0.into(), end: 1.into() }), None),
            )
            .unwrap()
            .output(
                "second",
                product.slice(Some(IndexRange { start: 1.into(), end: 3.into() }), None),
            )
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let snapshot = GpuScopeLowering::new(&compact, &FrozenGraphScopeId::Root).unwrap();
        assert!(!snapshot.aliases.compact_products.is_empty(), "exercise physical fused ports");
        assert!(!snapshot.aliases.decompositions.is_empty(), "exercise fused decomposition");
        check_scope_snapshots(&compact);

        let nested = parallel(3usize, |index| Ok(index + Int::constant(1))).unwrap();
        let graph = DslContext::new("snapshot-child-scope")
            .output("out", nested)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        assert!(graph.scopes.len() > 1);
        check_scope_snapshots(&graph);
    }

    #[test]
    fn test_gpu_scope_snapshot_survives_global_cache_eviction() {
        let ring = Ring::new(97u64, 8usize);
        let graph = DslContext::new("snapshot-held")
            .output("out", ring.input("input", (1, 1)).transpose())
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let snapshot = GpuScopeLowering::new(&graph, &FrozenGraphScopeId::Root).unwrap();
        let original = Arc::clone(&snapshot.aliases);
        // The global cache has four entries. All later snapshot lookups must
        // keep the original lowering even after its cache entry is evicted.
        for columns in 2..7 {
            let other = DslContext::new("snapshot-eviction")
                .output("out", ring.input("input", (1, columns)).transpose())
                .unwrap()
                .build()
                .unwrap()
                .validate(&ParamEnv::default())
                .unwrap();
            drop(GpuScopeLowering::new(&other, &FrozenGraphScopeId::Root).unwrap());
        }
        for index in 0..graph.root_scope().execution_order.len() {
            assert!(Arc::ptr_eq(
                &original,
                &snapshot.metadata(NodeId(index as u64)).unwrap().aliases,
            ));
        }
        let fresh = GpuScopeLowering::new(&graph, &FrozenGraphScopeId::Root).unwrap();
        assert!(!Arc::ptr_eq(&original, &fresh.aliases), "cache entry was really evicted");
    }

    #[test]
    fn test_gpu_scope_snapshot_revalidates_after_graph_mutation() {
        let ring = Ring::new(97u64, 8usize);
        let joined = Mat::concat(
            ConcatAxis::Rows,
            vec![ring.input("first", (2, 2)), ring.input("second", (2, 2))],
        );
        let mut graph = DslContext::new("snapshot-validation-mutation")
            .output("out", joined.slice(Some(IndexRange { start: 0.into(), end: 2.into() }), None))
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let snapshot = GpuScopeLowering::new(&graph, &FrozenGraphScopeId::Root).unwrap();
        let original = Arc::clone(&snapshot.aliases);
        let concat = *original.concats.keys().next().expect("concat alias");
        drop(snapshot);
        graph
            .scopes
            .get_mut(&FrozenGraphScopeId::Root)
            .unwrap()
            .liveness
            .retained
            .insert(WireRef { node: concat, port: Port(0) });
        let changed = GpuScopeLowering::new(&graph, &FrozenGraphScopeId::Root).unwrap();
        assert!(changed.alias_facts().is_empty());
        assert!(!Arc::ptr_eq(&original, &changed.aliases));
        assert_eq!(
            changed.site_metadata(concat).unwrap(),
            gpu_effective_site_metadata(&graph, &FrozenGraphScopeId::Root, concat).unwrap(),
        );
    }
}
