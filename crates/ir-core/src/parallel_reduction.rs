//! Recognize independent parallel matrix producers consumed only by an ordered sum.
//! The backend and graph residency audit share this exact pattern; arbitrary family
//! consumers are never classified as bounded reductions.
use crate::{
    FrozenGraphScopeId, Graph, IntExpr, ParamEnv,
    node::{MatrixBinaryOp, NodeKind},
    types::{NodeId, Port, WireRef},
};
use num_bigint::BigInt;

/// One independently accumulated matrix port of a parallel member sum.
/// The addition may be the initial root addition or the sequential body's addition.
#[derive(Clone, Debug)]
pub struct ParallelMemberSum {
    pub initial: NodeId,
    pub reads: Vec<NodeId>,
    pub addition_scope: FrozenGraphScopeId,
    pub addition: NodeId,
    pub carried: usize,
}

/// Recognize an ordered matrix sum seeded either with member zero, or with
/// members zero plus one, followed by the remaining members in increasing order.
/// Every sequential argument must be a recognized initial sum or its family;
/// externally returned members and arbitrary family consumers prevent fusion.
pub fn parallel_member_sums(
    graph: &Graph,
    scope_id: &FrozenGraphScopeId,
    loop_id: NodeId,
    ports: usize,
    count: usize,
    env: &ParamEnv,
) -> Option<(NodeId, Vec<Option<ParallelMemberSum>>)> {
    if count < 2 {
        return None;
    }
    let root = graph.scope(scope_id)?;
    let node = |id: NodeId| root.nodes().get(id.0 as usize);
    let output = |id: NodeId| WireRef { node: id, port: Port(0) };
    let consumers = |wire: WireRef| {
        root.nodes()
            .iter()
            .enumerate()
            .flat_map(|(index, consumer)| {
                root.arguments(consumer)
                    .unwrap_or_default()
                    .into_iter()
                    .enumerate()
                    .filter(move |(_, argument)| *argument == wire)
                    .map(move |(position, _)| (NodeId(index as u64), position))
            })
            .collect::<Vec<_>>()
    };
    let returned = |wire: WireRef| {
        root.outputs().contains(&wire) ||
            matches!(scope_id, FrozenGraphScopeId::Root) &&
                graph.outputs().values().any(|output| output.value == wire)
    };
    let evaluates_to = |expression: &IntExpr, value: usize| {
        expression.evaluate(env).is_ok_and(|actual| actual == BigInt::from(value))
    };
    let mut sequential = None;
    let mut summed = vec![None; ports];
    for (port, summed_port) in summed.iter_mut().enumerate() {
        let wire = WireRef { node: loop_id, port: Port(u32::try_from(port).ok()?) };
        let uses = consumers(wire);
        if returned(wire) || !matches!(uses.len(), 2 | 3) {
            continue;
        }
        let member_read = |member: usize| {
            uses.iter().map(|(id, _)| *id).find(|id| node(*id).is_some_and(|read| matches!(read.kind(), NodeKind::FamilyGetStatic { index } if evaluates_to(index, member))))
        };
        let Some(first) = member_read(0) else { continue };
        let Some(&(loop_node, capture)) = uses.iter().find(|(id, _)| {
            node(*id).is_some_and(|use_| matches!(use_.kind(), NodeKind::SequentialLoop(_)))
        }) else {
            continue
        };
        if returned(output(first)) {
            continue;
        }
        let [(first_user, first_position)] = consumers(output(first))[..] else { continue };
        let (initial, reads, prefix, root_add, carried) =
            if first_user == loop_node && uses.len() == 2 {
                (first, vec![first], 1, None, first_position)
            } else {
                let Some(second) = member_read(1) else { continue };
                let [(second_user, _)] = consumers(output(second))[..] else { continue };
                if uses.len() != 3 ||
                    first_user != second_user ||
                    returned(output(second)) ||
                    returned(output(first_user)) ||
                    !node(first_user).is_some_and(|n| {
                        matches!(n.kind(), NodeKind::MatrixBinary(MatrixBinaryOp::Add))
                    })
                {
                    continue;
                }
                let [(carrier, carried)] = consumers(output(first_user))[..] else { continue };
                if carrier != loop_node {
                    continue;
                }
                (first_user, vec![first, second], 2, Some(first_user), carried)
            };
        let Some(NodeKind::SequentialLoop(spec)) = node(loop_node).map(|n| n.kind()) else {
            continue
        };
        if carried >= spec.carried_count ||
            capture < spec.carried_count ||
            !evaluates_to(&spec.count, count - prefix)
        {
            continue;
        }
        let child_id = graph.child_scope_id(scope_id, loop_node)?;
        let child = graph.scope(&child_id)?;
        let (state, family) = (*child.inputs().get(carried)?, *child.inputs().get(capture)?);
        let result = *child.outputs().get(carried)?;
        let child_node = |wire: WireRef| child.nodes().get(wire.node.0 as usize);
        let body_add = child_node(result)?;
        let arguments = child.arguments(body_add)?;
        let read = match (body_add.kind(), arguments.as_slice()) {
            (NodeKind::MatrixBinary(MatrixBinaryOp::Add), [left, right]) if *left == state => {
                *right
            }
            (NodeKind::MatrixBinary(MatrixBinaryOp::Add), [left, right]) if *right == state => {
                *left
            }
            _ => continue,
        };
        let read_node = child_node(read)?;
        let [source, index] = child.arguments(read_node)?[..] else { continue };
        let offset = IntExpr::Add(
            Box::new(IntExpr::LoopIndex(spec.index_slot)),
            Box::new(IntExpr::constant(prefix)),
        )
        .canonicalize();
        let index_node = child_node(index)?;
        let consecutive = match index_node.kind() {
            NodeKind::EvaluateInt(expression) => expression.clone().canonicalize() == offset,
            NodeKind::IntBinary(crate::node::IntBinaryOp::Add) => {
                let kinds = child
                    .arguments(index_node)?
                    .iter()
                    .map(|operand| child_node(*operand).map(|n| n.kind().clone()))
                    .collect::<Option<Vec<_>>>()?;
                let is_index = |kind: &NodeKind| matches!(kind, NodeKind::EvaluateInt(IntExpr::LoopIndex(slot)) if *slot == spec.index_slot);
                let is_offset = |kind: &NodeKind| matches!(kind, NodeKind::ConstantInt(value) if *value == BigInt::from(prefix));
                matches!(kinds.as_slice(), [a, b] if (is_index(a) && is_offset(b)) || (is_offset(a) && is_index(b)))
            }
            _ => false,
        };
        if !matches!(read_node.kind(), NodeKind::FamilyGetDynamic) ||
            source != family ||
            !consecutive
        {
            continue;
        }
        if *sequential.get_or_insert(loop_node) != loop_node {
            return None;
        }
        *summed_port = Some(ParallelMemberSum {
            initial,
            reads,
            addition_scope: if root_add.is_some() { scope_id.clone() } else { child_id },
            addition: root_add.unwrap_or(result.node),
            carried,
        });
    }
    let sequential = sequential?;
    let loop_node = node(sequential)?;
    let NodeKind::SequentialLoop(spec) = loop_node.kind() else { return None };
    let accounted = root.arguments(loop_node)?.iter().enumerate().all(|(position, argument)| {
        if position < spec.carried_count {
            summed
                .iter()
                .flatten()
                .any(|sum| sum.carried == position && *argument == output(sum.initial))
        } else {
            argument.node == loop_id &&
                summed.get(argument.port.0 as usize).is_some_and(Option::is_some)
        }
    });
    let child = graph.scope(&graph.child_scope_id(scope_id, sequential)?)?;
    let plain_body = child.nodes().iter().all(|node| {
        matches!(
            node.kind(),
            NodeKind::Input { .. } |
                NodeKind::EvaluateInt(_) |
                NodeKind::ConstantInt(_) |
                NodeKind::IntBinary(crate::node::IntBinaryOp::Add) |
                NodeKind::FamilyGetDynamic |
                NodeKind::MatrixBinary(MatrixBinaryOp::Add)
        )
    });
    (accounted && plain_body).then_some((sequential, summed))
}

/// Whether this scope or a descendant contains a recognized parallel member sum.
/// Host replay can preserve bounded waves when a sequential ancestor contains one.
pub fn contains_parallel_member_sum(
    graph: &Graph,
    scope_id: &FrozenGraphScopeId,
    env: &ParamEnv,
) -> bool {
    let Some(scope) = graph.scope(scope_id) else { return false };
    scope.nodes().iter().enumerate().any(|(index, node)| {
        let id = NodeId(index as u64);
        let local = if let NodeKind::ParallelLoop(spec) = node.kind() {
            spec.count.evaluate(env).ok().and_then(|n| usize::try_from(n).ok()).is_some_and(
                |count| {
                    parallel_member_sums(graph, scope_id, id, node.output_types().len(), count, env)
                        .is_some()
                },
            )
        } else {
            false
        };
        local ||
            graph
                .child_scope_id(scope_id, id)
                .is_some_and(|child| contains_parallel_member_sum(graph, &child, env))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GraphOutput, NodeHandle, SubgraphHandle, ValueHandle, WireType,
        node::{ParallelLoop, SequentialLoop},
        types::MatrixType,
        with_new_construction_scope,
    };
    use std::collections::BTreeMap;

    fn input(name: &str, ty: &WireType) -> ValueHandle {
        NodeHandle::new(
            NodeKind::Input { name: name.into(), wire_type: ty.clone(), artifact: None },
            vec![],
            vec![ty.clone()],
        )
        .output(0)
        .unwrap()
    }

    fn reduction(
        count: usize,
        prefix: usize,
        return_member: bool,
        wrong_offset: bool,
    ) -> (Graph, NodeId) {
        let matrix = WireType::Matrix(MatrixType {
            ring: crate::ring::test_ring(17, 8),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        });
        let family = WireType::IndexedFamily {
            element: Box::new(matrix.clone()),
            count: IntExpr::constant(count),
        };
        let body = with_new_construction_scope(|scope| {
            let inputs = vec![input("a", &matrix), input("b", &matrix), input("resident", &matrix)];
            SubgraphHandle::new("producer", scope, inputs.clone(), inputs).unwrap()
        });
        let producer = NodeHandle::parallel_loop(
            body,
            vec![input("a", &matrix), input("b", &matrix), input("resident", &matrix)],
            vec![family.clone(); 3],
            ParallelLoop {
                count: IntExpr::constant(count),
                minimum_count: 0,
                index_slot: 0,
                bindings: vec![],
                input_modes: vec![crate::node::LoopInputMode::Broadcast; 3],
            },
        );
        let families = vec![producer.output(0).unwrap(), producer.output(1).unwrap()];
        let read = |family: &ValueHandle, index| {
            NodeHandle::new(
                NodeKind::FamilyGetStatic { index: IntExpr::constant(index) },
                vec![family.clone()],
                vec![matrix.clone()],
            )
            .output(0)
            .unwrap()
        };
        let first = families.iter().map(|f| read(f, 0)).collect::<Vec<_>>();
        let initial = families
            .iter()
            .zip(&first)
            .map(|(f, a)| {
                if prefix == 1 {
                    return a.clone();
                }
                NodeHandle::new(
                    NodeKind::MatrixBinary(MatrixBinaryOp::Add),
                    vec![a.clone(), read(f, 1)],
                    vec![matrix.clone()],
                )
                .output(0)
                .unwrap()
            })
            .collect::<Vec<_>>();
        let body = with_new_construction_scope(|scope| {
            let state = vec![input("s0", &matrix), input("s1", &matrix)];
            let captured = vec![input("f0", &family), input("f1", &family)];
            let index = NodeHandle::new(
                NodeKind::EvaluateInt(IntExpr::Add(
                    Box::new(IntExpr::LoopIndex(1)),
                    Box::new(IntExpr::constant(prefix + usize::from(wrong_offset))),
                )),
                vec![],
                vec![WireType::Int],
            )
            .output(0)
            .unwrap();
            let outputs = state
                .iter()
                .zip(&captured)
                .map(|(s, f)| {
                    let member = NodeHandle::new(
                        NodeKind::FamilyGetDynamic,
                        vec![f.clone(), index.clone()],
                        vec![matrix.clone()],
                    )
                    .output(0)
                    .unwrap();
                    NodeHandle::new(
                        NodeKind::MatrixBinary(MatrixBinaryOp::Add),
                        vec![s.clone(), member],
                        vec![matrix.clone()],
                    )
                    .output(0)
                    .unwrap()
                })
                .collect();
            SubgraphHandle::new("sum", scope, state.into_iter().chain(captured).collect(), outputs)
                .unwrap()
        });
        let sum = NodeHandle::sequential_loop(
            body,
            initial.clone().into_iter().chain(families).collect(),
            vec![matrix; 2],
            SequentialLoop {
                count: IntExpr::constant(count - prefix),
                index_slot: 1,
                bindings: vec![],
                carried_count: 2,
            },
        );
        let mut outputs = BTreeMap::from([
            (
                "sum0".into(),
                GraphOutput {
                    value: if count == prefix {
                        initial[0].clone()
                    } else {
                        sum.output(0).unwrap()
                    },
                    availability: None,
                },
            ),
            (
                "sum1".into(),
                GraphOutput {
                    value: if count == prefix {
                        initial[1].clone()
                    } else {
                        sum.output(1).unwrap()
                    },
                    availability: None,
                },
            ),
        ]);
        outputs.insert(
            "exported".into(),
            GraphOutput {
                value: producer.output(2).unwrap(),
                availability: Some(crate::artifact::ArtifactAvailability::Transferred),
            },
        );
        if return_member {
            outputs.insert(
                "member".into(),
                GraphOutput { value: first[1].clone(), availability: None },
            );
        }
        let (graph, _) =
            Graph::freeze("two-port-reduction", vec![], outputs, vec![], vec![], BTreeMap::new())
                .unwrap();
        let id = graph.root_scope().node_id(&producer).unwrap();
        (graph, id)
    }

    #[test]
    fn test_parallel_member_sums_preserves_two_component_ports() {
        for prefix in [1, 2] {
            let (graph, id) = reduction(6, prefix, false, false);
            let (_, ports) = parallel_member_sums(
                &graph,
                &FrozenGraphScopeId::Root,
                id,
                3,
                6,
                &ParamEnv::default(),
            )
            .expect("both component sums recognized");
            assert_eq!(ports.iter().flatten().map(|p| p.carried).collect::<Vec<_>>(), vec![0, 1]);
            assert!(ports[2].is_none(), "the separately exported family is preserved");
            assert!(ports.iter().flatten().all(|p| p.reads.len() == prefix));
            for (returned, offset) in [(true, false), (false, true)] {
                let (graph, id) = reduction(6, prefix, returned, offset);
                assert!(
                    parallel_member_sums(
                        &graph,
                        &FrozenGraphScopeId::Root,
                        id,
                        2,
                        6,
                        &ParamEnv::default()
                    )
                    .is_none(),
                    "returned members and nonconsecutive reads must prevent fusion"
                );
            }
        }
        // The DSL normalizes iterate(0) to its initial value: no sum loop exists.
        let (graph, id) = reduction(2, 2, false, false);
        assert!(
            parallel_member_sums(&graph, &FrozenGraphScopeId::Root, id, 3, 2, &ParamEnv::default())
                .is_none()
        );
    }
}
