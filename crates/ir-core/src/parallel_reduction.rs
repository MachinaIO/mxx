//! Recognize independent parallel matrix producers consumed only by an ordered sum.
//! The backend and graph residency audit share this exact pattern; arbitrary family
//! consumers are never classified as bounded reductions.
use crate::{
    FrozenGraphScopeId, Graph, IntExpr, ParamEnv,
    node::{MatrixBinaryOp, NodeKind},
    types::{NodeId, Port, WireRef},
};
use num_bigint::BigInt;

/// The sum of every member of some output ports of a parallel
/// loop `loop_id`, in the form a member-by-member sum lowers to: members 0 and
/// 1 read (`FamilyGetStatic`) and added (`MatrixBinary(Add)`), then a
/// sequential loop over the other `count - 2` members whose body adds member
/// `index + 2` (`FamilyGetDynamic`) to its carried state. Returns that
/// sequential loop and, per output port it sums, the two member reads, their
/// sum, and its carried position. Every argument of the sequential loop must
/// be one of these sums or a summed port, and its body may hold nothing else.
pub fn parallel_member_sums(
    graph: &Graph,
    scope_id: &FrozenGraphScopeId,
    loop_id: NodeId,
    ports: usize,
    count: usize,
    env: &ParamEnv,
) -> Option<(NodeId, Vec<Option<(NodeId, NodeId, NodeId, usize)>>)> {
    if count < 2 {
        return None;
    }
    let root = graph.scope(scope_id)?;
    let nodes = root.nodes();
    let node = |id: NodeId| nodes.get(id.0 as usize);
    let output = |id: NodeId| WireRef { node: id, port: Port(0) };
    let consumers = |wire: WireRef| {
        nodes
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
        if returned(wire) || uses.len() != 3 {
            continue;
        }
        let member_read = |member: usize| {
            uses.iter().map(|(id, _)| *id).find(|id| {
                node(*id).is_some_and(|read| {
                    matches!(read.kind(), NodeKind::FamilyGetStatic { index } if evaluates_to(index, member))
                })
            })
        };
        let (Some(first), Some(second)) = (member_read(0), member_read(1)) else { continue };
        let Some(&(loop_node, capture)) = uses.iter().find(|(id, _)| {
            node(*id).is_some_and(|use_| matches!(use_.kind(), NodeKind::SequentialLoop(_)))
        }) else {
            continue;
        };
        let (first_uses, second_uses) = (consumers(output(first)), consumers(output(second)));
        let ([(add, _)], [(other, _)]) = (first_uses.as_slice(), second_uses.as_slice()) else {
            continue;
        };
        if add != other ||
            !node(*add).is_some_and(|add| {
                matches!(add.kind(), NodeKind::MatrixBinary(MatrixBinaryOp::Add))
            }) ||
            [first, second, *add].into_iter().any(|id| returned(output(id)))
        {
            continue;
        }
        let [(carrier, carried)] = consumers(output(*add))[..] else { continue };
        let Some(NodeKind::SequentialLoop(spec)) = node(loop_node).map(|node| node.kind()) else {
            continue;
        };
        if carrier != loop_node ||
            carried >= spec.carried_count ||
            capture < spec.carried_count ||
            !evaluates_to(&spec.count, count - 2)
        {
            continue;
        }
        // The body adds member `index + 2` of the captured family to the state.
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
        let plus_two = IntExpr::Add(
            Box::new(IntExpr::LoopIndex(spec.index_slot)),
            Box::new(IntExpr::constant(2)),
        )
        .canonicalize();
        let index_node = child_node(index)?;
        let loop_index_plus_two = match index_node.kind() {
            NodeKind::EvaluateInt(expression) => expression.clone().canonicalize() == plus_two,
            NodeKind::IntBinary(crate::node::IntBinaryOp::Add) => {
                let operands = child.arguments(index_node)?;
                let kinds = operands
                    .iter()
                    .map(|operand| child_node(*operand).map(|node| node.kind().clone()))
                    .collect::<Option<Vec<_>>>()?;
                let is_index = |kind: &NodeKind| matches!(kind, NodeKind::EvaluateInt(IntExpr::LoopIndex(slot)) if *slot == spec.index_slot);
                let is_two = |kind: &NodeKind| matches!(kind, NodeKind::ConstantInt(value) if *value == BigInt::from(2));
                matches!(kinds.as_slice(), [a, b] if (is_index(a) && is_two(b)) || (is_two(a) && is_index(b)))
            }
            _ => false,
        };
        if !matches!(read_node.kind(), NodeKind::FamilyGetDynamic) ||
            source != family ||
            !loop_index_plus_two
        {
            continue;
        }
        if *sequential.get_or_insert(loop_node) != loop_node {
            return None;
        }
        *summed_port = Some((first, second, *add, carried));
    }
    let sequential = sequential?;
    let loop_node = node(sequential)?;
    let NodeKind::SequentialLoop(spec) = loop_node.kind() else { return None };
    let accounted = root.arguments(loop_node)?.iter().enumerate().all(|(position, argument)| {
        if position < spec.carried_count {
            summed
                .iter()
                .flatten()
                .any(|(_, _, add, carried)| *carried == position && argument.node == *add)
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
