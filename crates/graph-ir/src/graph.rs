use crate::{
    expr::RealExpr,
    node::Node,
    types::{WireRef, WireType},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CompileParameter {
    pub name: String,
    pub kind: CompileParameterKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum CompileParameterKind {
    Integer,
    Real,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    pub name: String,
    pub parameters: Vec<CompileParameter>,
    pub input_types: BTreeMap<String, WireType>,
    pub nodes: Vec<Node>,
    pub outputs: BTreeMap<String, WireRef>,
    pub subgraphs: BTreeMap<String, Box<Graph>>,
    /// Optional exact real constants used by graph generators.
    pub real_constants: BTreeMap<String, RealExpr>,
}

impl Graph {
    pub fn node(&self, id: crate::types::NodeId) -> Option<&Node> {
        self.nodes.iter().find(|node| node.id == id)
    }
}
