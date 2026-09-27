//! A self-contained HTML view of a graph.
//!
//! [`render_html`] draws each scope of a [`Graph`] as a layered dataflow diagram. Hovering a node
//! shows its operation, argument and output types, and construction site; a loop or call node
//! opens its body. With a [`ValidatedGraph`] the types are concrete shapes; otherwise they are the
//! symbolic construction types. With node costs (for example the ones a GPU plan measures while
//! planning) nodes are colored by their predicted share of one execution, and a side panel ranks
//! the bottlenecks. The page has no external dependencies.

use crate::{
    ValidatedGraph,
    expr::IntExpr,
    graph::{FrozenGraphScopeId, Graph, GraphScope, NodeHandle},
    node::NodeKind,
    ring::{RingExpr, RingRef},
    types::{ConcreteMatrixType, ConcreteWireType, MatrixType, NodeId, Port, WireRef, WireType},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;

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
    let scopes = graph
        .scopes()
        .iter()
        .map(|(id, scope)| {
            let key = scope_key(id);
            let wire_types = validated.and_then(|validated| validated.scope(id));
            let nodes = scope
                .nodes()
                .iter()
                .enumerate()
                .map(|(index, node)| {
                    let node_id = NodeId(index as u64);
                    let cost = cost_of.get(&(key.clone(), node_id));
                    node_data(graph, scope, id, node_id, node, validated, wire_types, cost)
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

#[allow(clippy::too_many_arguments)]
fn node_data(
    graph: &Graph,
    scope: &GraphScope,
    scope_id: &FrozenGraphScopeId,
    node_id: NodeId,
    node: &NodeHandle,
    validated: Option<&ValidatedGraph>,
    wire_types: Option<&crate::ValidatedScope>,
    cost: Option<&&NodeCost>,
) -> Value {
    let wire_type = |wire: WireRef, symbolic: &WireType| {
        wire_types
            .and_then(|scope| scope.wire_types.get(&wire))
            .map_or_else(|| symbolic_type(symbolic), concrete_type)
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
        "label": node_label(node.kind(), validated),
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

fn node_label(kind: &NodeKind, validated: Option<&ValidatedGraph>) -> String {
    let count = |count: &IntExpr| {
        validated
            .and_then(|validated| count.evaluate(&validated.bindings).ok())
            .map_or_else(|| int_expr(count), |count| count.to_string())
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
