//! A self-contained HTML view of a graph.
//!
//! [`render_html`] draws each scope of a [`Graph`] as a layered dataflow diagram. Hovering a node
//! shows its operation, argument and output types, and construction site; a loop or call node
//! opens its body. With a [`ValidatedGraph`] the types are concrete shapes; otherwise they are the
//! symbolic construction types. With node costs (for example the ones a GPU plan measures while
//! planning) nodes are colored by their predicted share of one execution, and a side panel ranks
//! the bottlenecks. The page has no external dependencies.

use crate::{
    ParamEnv, ValidatedGraph, concretize_wire_type,
    expr::IntExpr,
    graph::{FrozenGraphScopeId, Graph, GraphScope, NodeHandle},
    node::NodeKind,
    ring::{RingExpr, RingRef},
    types::{ConcreteMatrixType, ConcreteWireType, MatrixType, NodeId, Port, WireRef, WireType},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

const TEMPLATE: &str = include_str!("visualize.html");
const DATA_PLACEHOLDER: &str = "/*GRAPH_DATA*/null";
/// Longest operation description shown in a tooltip.
const DETAIL_CHARS: usize = 600;

/// Predicted time of one graph node per execution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodeCost {
    pub scope: FrozenGraphScopeId,
    pub node: NodeId,
    /// Seconds of the operations the node itself runs, over every instance of its scope.
    pub self_seconds: f64,
    /// Seconds including the nodes of the bodies the node runs (loops and calls).
    pub total_seconds: f64,
}

/// Render `graph` as a standalone HTML page. `validated` supplies concrete types and loop counts;
/// `costs` is the predicted seconds of one execution and the per-node costs that divide it.
pub fn render_html(
    graph: &Graph,
    validated: Option<&ValidatedGraph>,
    costs: Option<(f64, &[NodeCost])>,
) -> String {
    let data = graph_data(graph, validated, costs).to_string();
    // A JSON string may contain "</script>"; escaping the slash keeps it inside the script.
    TEMPLATE.replacen(DATA_PLACEHOLDER, &data.replace("</", "<\\/"), 1)
}

fn graph_data(
    graph: &Graph,
    validated: Option<&ValidatedGraph>,
    costs: Option<(f64, &[NodeCost])>,
) -> Value {
    let cost_of = costs
        .map(|(_, nodes)| {
            nodes
                .iter()
                .map(|cost| ((scope_key(&cost.scope), cost.node), cost))
                .collect::<BTreeMap<_, _>>()
        })
        .unwrap_or_default();
    // Every binding each scope is instantiated with: a named subgraph has one
    // per distinct call binding. A validated graph already checked them all.
    let scope_envs = validated.map(|validated| {
        let mut envs = BTreeMap::new();
        crate::validate::collect_scope_bindings(
            graph,
            &FrozenGraphScopeId::Root,
            validated.bindings.clone(),
            &mut envs,
        )
        .expect("a validated graph has scope bindings");
        envs
    });
    let scopes = graph
        .scopes()
        .iter()
        .map(|(id, scope)| {
            let key = scope_key(id);
            let envs = scope_envs.as_ref().and_then(|envs| envs.get(id)).map(Vec::as_slice);
            let nodes = scope
                .nodes()
                .iter()
                .enumerate()
                .map(|(index, node)| {
                    let node_id = NodeId(index as u64);
                    let cost = cost_of.get(&(key.clone(), node_id));
                    node_data(graph, scope, id, node_id, node, envs, cost)
                })
                .collect::<Vec<_>>();
            let outputs = if *id == FrozenGraphScopeId::Root {
                graph
                    .outputs()
                    .iter()
                    .map(|(name, output)| output_data(name, output.value))
                    .collect::<Vec<_>>()
            } else {
                scope
                    .outputs()
                    .iter()
                    .enumerate()
                    .map(|(position, wire)| output_data(&format!("output {position}"), *wire))
                    .collect()
            };
            json!({
                "id": key,
                "label": scope_label(id),
                "parent": scope_parent(id).map(|parent| scope_key(&parent)),
                "nodes": nodes,
                "outputs": outputs,
            })
        })
        .collect::<Vec<_>>();
    let bindings = validated.map(|validated| {
        validated
            .bindings
            .integers
            .iter()
            .map(|(name, value)| (name.clone(), value.to_string()))
            .chain(validated.bindings.reals.iter().map(|(name, value)| {
                let text = if value.denominator() == &1.into() {
                    value.numerator().to_string()
                } else {
                    format!("{}/{}", value.numerator(), value.denominator())
                };
                (name.clone(), text)
            }))
            .collect::<BTreeMap<_, _>>()
    });
    json!({
        "name": graph.name(),
        "bindings": bindings,
        "predicted_seconds": costs.map(|(seconds, _)| seconds),
        "scopes": scopes,
    })
}

fn output_data(name: &str, wire: WireRef) -> Value {
    json!({ "name": name, "node": wire.node.0, "port": wire.port.0 })
}

fn node_data(
    graph: &Graph,
    scope: &GraphScope,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    envs: Option<&[ParamEnv]>,
    cost: Option<&&NodeCost>,
) -> Value {
    let wire_type = |wire: WireRef, symbolic: &WireType| {
        envs.map_or_else(
            || symbolic_type(symbolic),
            |envs| instantiated_type(symbolic, envs, scope_id, wire.node),
        )
    };
    let inputs = node
        .arguments()
        .iter()
        .map(|argument| {
            let wire = scope.wire_ref(argument);
            json!({
                "node": wire.map(|wire| wire.node.0),
                "port": argument.port().0,
                "type": wire.map_or_else(
                    || symbolic_type(argument.wire_type()),
                    |wire| wire_type(wire, argument.wire_type()),
                ),
            })
        })
        .collect::<Vec<_>>();
    let outputs = node
        .output_types()
        .iter()
        .enumerate()
        .map(|(port, symbolic)| {
            wire_type(WireRef { node: node_id, port: Port(port as u32) }, symbolic)
        })
        .collect::<Vec<_>>();
    let detail = format!("{:?}", node.kind());
    let detail = match detail.char_indices().nth(DETAIL_CHARS) {
        Some((end, _)) => format!("{}…", &detail[..end]),
        None => detail,
    };
    json!({
        "id": node_id.0,
        "label": node_label(node.kind(), envs),
        "kind": kind_name(node.kind()),
        "detail": detail,
        "inputs": inputs,
        "outputs": outputs,
        "location": node.source_location().map(|location| {
            format!("{}:{}:{}", location.file, location.line, location.column)
        }),
        "child": graph.child_scope_id(scope_id, node_id).map(|child| scope_key(&child)),
        "self_seconds": cost.map(|cost| cost.self_seconds),
        "total_seconds": cost.map(|cost| cost.total_seconds),
    })
}

fn kind_name(kind: &NodeKind) -> String {
    let debug = format!("{kind:?}");
    debug.split(['(', ' ', '{']).next().unwrap_or_default().to_owned()
}

fn node_label(kind: &NodeKind, envs: Option<&[ParamEnv]>) -> String {
    // A count is shown as a number only when every instantiation agrees.
    let count = |count: &IntExpr| {
        let values = envs
            .unwrap_or_default()
            .iter()
            .map(|env| count.evaluate(env).ok())
            .collect::<Option<BTreeSet<_>>>()
            .unwrap_or_default();
        match (values.len(), values.first()) {
            (1, Some(value)) => value.to_string(),
            _ => int_expr(count),
        }
    };
    match kind {
        NodeKind::Input { name, .. } => format!("Input {name}"),
        NodeKind::SubgraphCall(call) => format!("Call {}", call.definition),
        NodeKind::ParallelLoop(spec) => format!("ParallelLoop ×{}", count(&spec.count)),
        NodeKind::SequentialLoop(spec) => format!("SequentialLoop ×{}", count(&spec.count)),
        NodeKind::MatrixBinary(operation) => format!("Matrix{operation:?}"),
        NodeKind::ConstantInt(value) => format!("Int {value}"),
        NodeKind::ConstantBool(value) => format!("Bool {value}"),
        NodeKind::EvaluateInt(expression) => format!("Int {}", int_expr(expression)),
        _ => kind_name(kind),
    }
}

fn scope_key(id: &FrozenGraphScopeId) -> String {
    match id {
        FrozenGraphScopeId::Root => "root".to_owned(),
        FrozenGraphScopeId::Subgraph { canonical_name } => format!("subgraph:{canonical_name}"),
        FrozenGraphScopeId::ParallelBody { parent, owner } => {
            format!("{}/parallel#{}", scope_key(parent), owner.0)
        }
        FrozenGraphScopeId::SequentialBody { parent, owner } => {
            format!("{}/sequential#{}", scope_key(parent), owner.0)
        }
    }
}

fn scope_label(id: &FrozenGraphScopeId) -> String {
    match id {
        FrozenGraphScopeId::Root => "root".to_owned(),
        FrozenGraphScopeId::Subgraph { canonical_name } => canonical_name.clone(),
        FrozenGraphScopeId::ParallelBody { owner, .. } => format!("parallel body of #{}", owner.0),
        FrozenGraphScopeId::SequentialBody { owner, .. } => {
            format!("sequential body of #{}", owner.0)
        }
    }
}

/// The scope a body is nested in; a subgraph is a definition that any scope may call.
fn scope_parent(id: &FrozenGraphScopeId) -> Option<FrozenGraphScopeId> {
    match id {
        FrozenGraphScopeId::ParallelBody { parent, .. } |
        FrozenGraphScopeId::SequentialBody { parent, .. } => Some(parent.as_ref().clone()),
        FrozenGraphScopeId::Root | FrozenGraphScopeId::Subgraph { .. } => None,
    }
}

/// The concrete type of a wire when every instantiation of its scope gives the
/// same one. Validation resolves a loop body at index zero, so a type that
/// depends on the loop index stays symbolic, as does one that differs between
/// the calls of a named subgraph.
fn instantiated_type(
    symbolic: &WireType,
    envs: &[ParamEnv],
    scope: &FrozenGraphScopeId,
    node: NodeId,
) -> String {
    if uses_loop_index(symbolic) {
        return format!("{} (depends on the loop index)", symbolic_type(symbolic));
    }
    let concrete = envs
        .iter()
        .map(|env| concretize_wire_type(symbolic, env, scope, node).ok())
        .collect::<Option<BTreeSet<_>>>()
        .unwrap_or_default();
    match (concrete.len(), concrete.first()) {
        (1, Some(ty)) => concrete_type(ty),
        (0, _) => symbolic_type(symbolic),
        (count, _) => format!("{} ({count} shapes across calls)", symbolic_type(symbolic)),
    }
}

/// Whether a shape shown by `symbolic_type` depends on a loop index.
fn uses_loop_index(ty: &WireType) -> bool {
    let matrix = |matrix: &MatrixType| {
        matrix.rows.contains_loop_index() ||
            matrix.columns.contains_loop_index() ||
            matrix.ring.contains_loop_index()
    };
    match ty {
        WireType::Matrix(value) => matrix(value),
        WireType::Trapdoor { matrix: value, gadget_base, digit_count, .. } => {
            matrix(value) || gadget_base.contains_loop_index() || digit_count.contains_loop_index()
        }
        WireType::SmallMatrix { matrix: value, max_coefficient_bound, .. } |
        WireType::Preimage { matrix: value, max_coefficient_bound, .. } => {
            matrix(value) || max_coefficient_bound.contains_loop_index()
        }
        WireType::IndexedFamily { element, count } => {
            uses_loop_index(element) || count.contains_loop_index()
        }
        WireType::Bytes { length } => length.contains_loop_index(),
        _ => false,
    }
}

fn concrete_matrix(matrix: &ConcreteMatrixType) -> String {
    format!(
        "{}×{} over N={}, {} CRT limbs ({} bits)",
        matrix.rows,
        matrix.columns,
        matrix.ring.ring_dimension(),
        matrix.ring.crt_depth(),
        matrix.ring.modulus().bits()
    )
}

fn concrete_type(ty: &ConcreteWireType) -> String {
    match ty {
        ConcreteWireType::Matrix(matrix) => format!("Matrix {}", concrete_matrix(matrix)),
        ConcreteWireType::Trapdoor { matrix, digit_count, gadget_base, .. } => format!(
            "Trapdoor {}, base {gadget_base}, {digit_count} digits",
            concrete_matrix(matrix)
        ),
        ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound, .. } => {
            format!("SmallMatrix {}, |c| ≤ {max_coefficient_bound}", concrete_matrix(matrix))
        }
        ConcreteWireType::Preimage { matrix, max_coefficient_bound, .. } => {
            format!("Preimage {}, |c| ≤ {max_coefficient_bound}", concrete_matrix(matrix))
        }
        ConcreteWireType::IndexedFamily { element, count } => {
            format!("Family[{count}] of {}", concrete_type(element))
        }
        ConcreteWireType::Bytes { length } => format!("Bytes[{length}]"),
        ConcreteWireType::TypedBlob { type_name, .. } => format!("TypedBlob {type_name}"),
        scalar => format!("{scalar:?}"),
    }
}

fn symbolic_matrix(matrix: &MatrixType) -> String {
    format!("{}×{} over {}", int_expr(&matrix.rows), int_expr(&matrix.columns), ring(&matrix.ring))
}

fn symbolic_type(ty: &WireType) -> String {
    match ty {
        WireType::Matrix(matrix) => format!("Matrix {}", symbolic_matrix(matrix)),
        WireType::Trapdoor { matrix, digit_count, gadget_base, .. } => format!(
            "Trapdoor {}, base {}, {} digits",
            symbolic_matrix(matrix),
            int_expr(gadget_base),
            int_expr(digit_count)
        ),
        WireType::SmallMatrix { matrix, max_coefficient_bound, .. } => format!(
            "SmallMatrix {}, |c| ≤ {}",
            symbolic_matrix(matrix),
            int_expr(max_coefficient_bound)
        ),
        WireType::Preimage { matrix, max_coefficient_bound, .. } => format!(
            "Preimage {}, |c| ≤ {}",
            symbolic_matrix(matrix),
            int_expr(max_coefficient_bound)
        ),
        WireType::IndexedFamily { element, count } => {
            format!("Family[{}] of {}", int_expr(count), symbolic_type(element))
        }
        WireType::Bytes { length } => format!("Bytes[{}]", int_expr(length)),
        WireType::TypedBlob { type_name, .. } => format!("TypedBlob {type_name}"),
        scalar => format!("{scalar:?}"),
    }
}

fn ring(value: &RingRef) -> String {
    match value.expression() {
        RingExpr::Generated { crt_bits, crt_depth, ring_dimension } => format!(
            "N={ring_dimension}, {} CRT limbs of {} bits",
            int_expr(crt_depth),
            int_expr(crt_bits)
        ),
        RingExpr::Explicit { crt_moduli, ring_dimension } => {
            format!("N={ring_dimension}, {} explicit CRT limbs", crt_moduli.len())
        }
        RingExpr::Slice { source, start, end } => {
            format!("limbs {}..{} of ({})", int_expr(start), int_expr(end), ring(source))
        }
        RingExpr::Select { source, indices } => {
            format!("{} selected limbs of ({})", indices.len(), ring(source))
        }
        RingExpr::Concat { left, right } => format!("({}) ++ ({})", ring(left), ring(right)),
    }
}

fn int_expr(expression: &IntExpr) -> String {
    let binary = |symbol: &str, left: &IntExpr, right: &IntExpr| {
        format!("({} {symbol} {})", int_expr(left), int_expr(right))
    };
    match expression {
        IntExpr::Const(value) => value.to_string(),
        IntExpr::Var(name) => name.clone(),
        IntExpr::LoopIndex(slot) => format!("i{slot}"),
        IntExpr::Add(left, right) => binary("+", left, right),
        IntExpr::Sub(left, right) => binary("-", left, right),
        IntExpr::Mul(left, right) => binary("*", left, right),
        IntExpr::Div(left, right) => binary("/", left, right),
        IntExpr::FloorDiv(left, right) => binary("//", left, right),
        IntExpr::Rem(left, right) => binary("%", left, right),
        IntExpr::RoundDiv(left, right) => binary("round/", left, right),
        IntExpr::Log2Ceil(value) => format!("log2ceil({})", int_expr(value)),
        IntExpr::Select { selector, branches } => format!(
            "select({}; {})",
            int_expr(selector),
            branches.iter().map(int_expr).collect::<Vec<_>>().join(", ")
        ),
        IntExpr::RingModulus(value) => format!("Q({})", ring(value)),
        IntExpr::RingCrtDepth(value) => format!("depth({})", ring(value)),
        IntExpr::RingCrtModulus { ring: value, index } => {
            format!("q[{}]({})", int_expr(index), ring(value))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    #[test]
    fn test_instantiated_type_is_concrete_only_when_every_instantiation_agrees() {
        let ring = RingRef::new(RingExpr::Explicit {
            crt_moduli: vec![IntExpr::from(17)],
            ring_dimension: 8,
        });
        let matrix = |rows: IntExpr| {
            WireType::Matrix(MatrixType { ring: ring.clone(), rows, columns: IntExpr::from(1) })
        };
        let env = |rows: i64| ParamEnv {
            integers: BTreeMap::from([("rows".to_owned(), BigInt::from(rows))]),
            ..ParamEnv::default()
        };
        let by_parameter = matrix(IntExpr::Var("rows".into()));
        let shown = |ty: &WireType, envs: &[ParamEnv]| {
            instantiated_type(ty, envs, &FrozenGraphScopeId::Root, NodeId(0))
        };
        assert!(shown(&by_parameter, &[env(2), env(2)]).starts_with("Matrix 2×1 over N=8"));
        assert_eq!(
            shown(&by_parameter, &[env(2), env(3)]),
            "Matrix rows×1 over N=8, 1 explicit CRT limbs (2 shapes across calls)"
        );
        assert!(
            shown(&matrix(IntExpr::LoopIndex(0)), &[env(2)])
                .ends_with("(depends on the loop index)")
        );
    }
}
