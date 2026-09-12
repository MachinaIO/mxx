use crate::{
    artifact::SpecHash,
    expr::{ExprError, ParamEnv, Rational},
    graph::Graph,
};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::{self, Write},
};
use thiserror::Error;

// Version 9 adds the serialized `IntExpr::Select` variant.  Bumping the wire
// format prevents older runtimes from accepting manifests whose integer
// expressions they cannot decode or evaluate.
pub const IR_VERSION: u32 = 9;

#[derive(Debug, Error)]
pub enum EncodingError {
    #[error("canonical JSON serialization failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("canonical expression evaluation failed: {0}")]
    Expression(#[from] ExprError),
}

pub fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>, EncodingError> {
    let mut value = serde_json::to_value(value)?;
    canonicalize_value(&mut value);
    Ok(serde_json::to_vec(&value)?)
}

pub fn hash_canonical<T: Serialize>(value: &T) -> Result<[u8; 32], EncodingError> {
    let mut value = serde_json::to_value(value)?;
    canonicalize_value(&mut value);
    let mut hasher = Sha256::new();
    serde_json::to_writer(DigestWriter(&mut hasher), &value)?;
    Ok(hasher.finalize().into())
}

/// Hashes one concrete executable graph instantiation.
///
/// Unlike a parameter-independent protocol-declaration hash, this identity
/// deliberately commits to every compile-parameter binding. Artifact
/// `ProductionId`s use this hash so artifacts produced with different
/// concrete dimensions or moduli cannot be interchanged.
pub fn spec_hash(graph: &Graph, bindings: &ParamEnv) -> Result<SpecHash, EncodingError> {
    if let Some((cached_bindings, hash)) = graph.spec_hash_cache().get() {
        if cached_bindings == bindings {
            return Ok(hash.clone());
        }
    }
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
    let hash = SpecHash(hash_canonical(&payload)?);
    // A competing first computation is harmless: both results commit to the
    // exact graph and bindings. No lock or graph serialization on cache hits.
    let _ = graph.spec_hash_cache().set((bindings.clone(), hash.clone()));
    Ok(hash)
}

fn canonicalize_value(value: &mut Value) {
    match value {
        Value::Array(values) => {
            for value in values {
                canonicalize_value(value);
            }
        }
        Value::Object(values) => {
            for value in values.values_mut() {
                canonicalize_value(value);
            }
            values.sort_keys();
        }
        _ => {}
    }
}

struct DigestWriter<'a>(&'a mut Sha256);

impl Write for DigestWriter<'_> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.0.update(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraphOutput, NodeHandle, WireType, node::NodeKind};
    use num_bigint::BigInt;
    use serde::Serialize;
    use serde_json::{Map, json};

    fn tiny_graph() -> Graph {
        let value = NodeHandle::new(
            NodeKind::ConstantInt(BigInt::from(7)),
            Vec::new(),
            vec![WireType::ConstantInt],
        )
        .output(0)
        .expect("constant output");
        Graph::freeze(
            "tiny-golden",
            Vec::new(),
            BTreeMap::from([("result".to_owned(), GraphOutput { value, confidentiality: None })]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .expect("graph freezes")
        .0
    }

    fn reference_canonicalize(value: Value) -> Value {
        match value {
            Value::Array(values) => {
                Value::Array(values.into_iter().map(reference_canonicalize).collect())
            }
            Value::Object(values) => {
                let sorted = values
                    .into_iter()
                    .map(|(key, value)| (key, reference_canonicalize(value)))
                    .collect::<BTreeMap<_, _>>();
                Value::Object(sorted.into_iter().collect::<Map<_, _>>())
            }
            scalar => scalar,
        }
    }

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
    fn in_place_canonicalization_matches_the_previous_nested_contract() {
        let input = json!({
            "z": [{"omega": 0, "alpha": 1}, {"d": [3, {"y": 2, "x": 1}]}],
            "a": {"right": [{"b": false, "a": true}], "left": null}
        });
        let expected = serde_json::to_vec(&reference_canonicalize(input.clone())).unwrap();
        assert_eq!(canonical_json(&input).unwrap(), expected);
    }

    #[test]
    fn streamed_hash_matches_hashing_the_canonical_json_bytes() {
        let input = json!({"z": [3, {"b": 2, "a": 1}], "a": "value"});
        assert_eq!(
            hash_canonical(&input).unwrap(),
            <[u8; 32]>::from(Sha256::digest(canonical_json(&input).unwrap()))
        );
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
    fn concrete_spec_hash_commits_to_parameter_bindings() {
        let value = NodeHandle::new(
            NodeKind::ConstantInt(BigInt::from(0)),
            Vec::new(),
            vec![WireType::ConstantInt],
        )
        .output(0)
        .expect("constant output");
        let graph = Graph::freeze(
            "parameterized",
            Vec::new(),
            BTreeMap::from([("result".to_owned(), GraphOutput { value, confidentiality: None })]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .expect("graph freezes")
        .0;
        let first = ParamEnv {
            integers: BTreeMap::from([("modulus".to_owned(), BigInt::from(17))]),
            ..ParamEnv::default()
        };
        let second = ParamEnv {
            integers: BTreeMap::from([("modulus".to_owned(), BigInt::from(19))]),
            ..ParamEnv::default()
        };

        let first_hash = spec_hash(&graph, &first).unwrap();
        let second_hash = spec_hash(&graph, &second).unwrap();
        assert_ne!(first_hash, second_hash);
        let decoded: Graph = serde_json::from_slice(&canonical_json(&graph).unwrap()).unwrap();
        assert_eq!(second_hash, spec_hash(&decoded, &second).unwrap());
        assert_eq!(first_hash, spec_hash(&graph.clone(), &first).unwrap());
        let mut changed_real = first.clone();
        changed_real.reals.insert("sigma".into(), Rational::new(3.into(), 1.into()).unwrap());
        assert_ne!(first_hash, spec_hash(&graph, &changed_real).unwrap());
        assert_eq!(
            spec_hash(&graph, &changed_real).unwrap(),
            spec_hash(&decoded, &changed_real).unwrap()
        );
    }

    #[test]
    fn tiny_graph_canonical_json_and_spec_hash_match_the_pre_streaming_golden() {
        const CANONICAL_JSON: &[u8] = br#"{"effect_roots":[],"name":"tiny-golden","outputs":{"result":{"confidentiality":null,"value":{"node":0,"port":0}}},"parameters":[],"real_constants":{},"scopes":[{"id":{"tag":"Root"},"scope":{"inputs":[],"nodes":[{"arguments":[],"id":0,"kind":{"tag":"ConstantInt","value":"7"},"output_types":[{"tag":"ConstantInt"}]}],"outputs":[{"node":0,"port":0}]}}]}"#;
        const SPEC_HASH: [u8; 32] = [
            8, 90, 9, 182, 175, 97, 67, 133, 225, 175, 203, 101, 88, 208, 23, 166, 88, 180, 206,
            225, 64, 187, 231, 32, 0, 135, 5, 79, 142, 182, 87, 27,
        ];
        let graph = tiny_graph();

        assert_eq!(canonical_json(&graph).unwrap(), CANONICAL_JSON);
        assert_eq!(spec_hash(&graph, &ParamEnv::default()).unwrap(), SpecHash(SPEC_HASH));

        let clone = graph.clone();
        let roundtrip: Graph =
            serde_json::from_slice(&serde_json::to_vec(&graph).unwrap()).unwrap();
        assert_eq!(
            spec_hash(&graph, &ParamEnv::default()).unwrap(),
            spec_hash(&clone, &ParamEnv::default()).unwrap()
        );
        assert_eq!(
            spec_hash(&graph, &ParamEnv::default()).unwrap(),
            spec_hash(&roundtrip, &ParamEnv::default()).unwrap()
        );
    }
}
