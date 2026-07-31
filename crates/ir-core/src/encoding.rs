use crate::{
    artifact::SpecHash,
    expr::{ExprError, ParamEnv, Rational, RealExpr},
    graph::Graph,
    node::NodeKind,
    types::WireType,
};
use serde::Serialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

pub const IR_VERSION: u32 = 3;

#[derive(Debug, Error)]
pub enum EncodingError {
    #[error("canonical JSON serialization failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("canonical expression evaluation failed: {0}")]
    Expression(#[from] ExprError),
}

pub fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>, EncodingError> {
    let value = serde_json::to_value(value)?;
    let canonical = canonicalize_value(value);
    Ok(serde_json::to_vec(&canonical)?)
}

pub fn hash_canonical<T: Serialize>(value: &T) -> Result<[u8; 32], EncodingError> {
    let bytes = canonical_json(value)?;
    Ok(Sha256::digest(bytes).into())
}

pub fn spec_hash(graph: &Graph, bindings: &ParamEnv) -> Result<SpecHash, EncodingError> {
    #[derive(Serialize)]
    struct Payload<'a> {
        ir_version: u32,
        graph: &'a Graph,
        integer_bindings: BTreeMap<&'a str, String>,
        real_bindings: BTreeMap<&'a str, &'a Rational>,
    }
    let mut normalized_graph = graph.clone();
    normalize_graph_reals(&mut normalized_graph, bindings)?;
    let payload = Payload {
        ir_version: IR_VERSION,
        graph: &normalized_graph,
        integer_bindings: bindings
            .integers
            .iter()
            .map(|(name, value)| (name.as_str(), value.to_string()))
            .collect(),
        real_bindings: bindings.reals.iter().map(|(name, value)| (name.as_str(), value)).collect(),
    };
    Ok(SpecHash(hash_canonical(&payload)?))
}

fn normalize_graph_reals(graph: &mut Graph, bindings: &ParamEnv) -> Result<(), ExprError> {
    for expression in graph.real_constants.values_mut() {
        *expression = normalize_real(expression, bindings)?;
    }
    for wire_type in graph.input_types.values_mut() {
        normalize_wire_type(wire_type, bindings)?;
    }
    for node in &mut graph.nodes {
        match &mut node.kind {
            NodeKind::Input { wire_type, .. } => normalize_wire_type(wire_type, bindings)?,
            NodeKind::ConstantReal(expression) |
            NodeKind::GaussianSample { sigma: expression, .. } |
            NodeKind::TrapdoorSample { sigma: expression, .. } => {
                *expression = normalize_real(expression, bindings)?;
            }
            _ => {}
        }
    }
    // A subgraph may declare parameters that are bound only at a call site.
    // Its source expressions therefore remain structural here; each call's
    // binding is already part of the parent graph's canonical payload.
    Ok(())
}

fn normalize_wire_type(wire_type: &mut WireType, bindings: &ParamEnv) -> Result<(), ExprError> {
    match wire_type {
        WireType::Trapdoor { sigma, .. } => {
            *sigma = normalize_real(sigma, bindings)?;
        }
        WireType::IndexedFamily { element, .. } => normalize_wire_type(element, bindings)?,
        _ => {}
    }
    Ok(())
}

fn normalize_real(expression: &RealExpr, bindings: &ParamEnv) -> Result<RealExpr, ExprError> {
    expression.close(bindings)
}

fn canonicalize_value(value: Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.into_iter().map(canonicalize_value).collect()),
        Value::Object(values) => {
            let sorted = values
                .into_iter()
                .map(|(key, value)| (key, canonicalize_value(value)))
                .collect::<BTreeMap<_, _>>();
            Value::Object(sorted.into_iter().collect::<Map<_, _>>())
        }
        scalar => scalar,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;

    #[test]
    fn object_keys_are_sorted_without_whitespace() {
        #[derive(Serialize)]
        struct Unsorted {
            z: u8,
            a: u8,
        }
        let encoded = canonical_json(&Unsorted { z: 1, a: 2 }).expect("serializable");
        assert_eq!(encoded, br#"{"a":2,"z":1}"#);
    }

    #[test]
    fn equivalent_integer_expressions_have_identical_encoding() {
        use crate::expr::IntExpr;
        let x = IntExpr::Var("x".to_owned());
        let lhs = IntExpr::Mul(
            Box::new(IntExpr::Add(Box::new(x.clone()), Box::new(IntExpr::constant(1)))),
            Box::new(IntExpr::constant(2)),
        );
        let rhs = IntExpr::Add(
            Box::new(IntExpr::constant(2)),
            Box::new(IntExpr::Mul(Box::new(IntExpr::constant(2)), Box::new(x))),
        );
        assert_eq!(
            canonical_json(&lhs).expect("serializable"),
            canonical_json(&rhs).expect("serializable")
        );
    }

    #[test]
    fn spec_hash_normalizes_bound_real_expressions() {
        use crate::{
            expr::RealExpr,
            graph::Graph,
            node::{Node, NodeKind},
            types::{NodeId, Port, WireRef},
        };
        use num_bigint::BigInt;

        let graph = |expression| Graph {
            name: "real-normalization".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![Node {
                id: NodeId(1),
                kind: NodeKind::ConstantReal(expression),
                args: Vec::new(),
            }],
            outputs: BTreeMap::from([(
                "out".to_owned(),
                WireRef { node: NodeId(1), port: Port(0) },
            )]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        let half = Rational::new(BigInt::from(1), BigInt::from(2)).expect("nonzero denominator");
        let lhs = graph(RealExpr::Add(
            Box::new(RealExpr::Rational(half.clone())),
            Box::new(RealExpr::Rational(half)),
        ));
        let rhs = graph(RealExpr::Rational(Rational::from_integer(BigInt::from(1))));
        assert_eq!(
            spec_hash(&lhs, &ParamEnv::default()).expect("lhs hash"),
            spec_hash(&rhs, &ParamEnv::default()).expect("rhs hash")
        );
    }
}
