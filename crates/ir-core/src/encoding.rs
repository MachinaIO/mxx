use crate::{
    artifact::SpecHash,
    expr::{ExprError, ParamEnv, Rational},
    graph::Graph,
};
use serde::Serialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

pub const IR_VERSION: u32 = 4;

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
    let payload = Payload {
        ir_version: IR_VERSION,
        graph,
        integer_bindings: bindings
            .integers
            .iter()
            .map(|(name, value)| (name.as_str(), value.to_string()))
            .collect(),
        real_bindings: bindings.reals.iter().map(|(name, value)| (name.as_str(), value)).collect(),
    };
    Ok(SpecHash(hash_canonical(&payload)?))
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
}
