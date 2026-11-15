//! Structured binaries: definitions made of leaves and loops.
//!
//! A [`Program`] has named parameters, typed inputs and outputs, and a list of [`Definition`]s;
//! the root definition takes the program's inputs as its parameters. A definition is a list of
//! [`Node`]s, each reading [`Value`]s produced before it, and ends with its outputs.
//!
//! There are two kinds of node. A leaf node runs one leaf ([`LeafRef`]). A [`Loop`] runs its body
//! definition up to `count` times. The body's parameters are the carried values followed by the
//! loop's other inputs. Its outputs are the next carried values, then, for a loop with `until`,
//! one `Bool` stop flag, then the per-iteration outputs. The loop's outputs are the final carried
//! values followed by one family per per-iteration output, member `i` coming from iteration `i`.
//! The loop index is `IntExpr::LoopIndex(index_slot)`, and `bindings` gives the body's own
//! parameters as compile expressions of the caller's.
//!
//! A loop without carried values and without `until` is *independent*
//! ([`Loop::is_independent`]): its iterations may run in any order and concurrently.

use crate::{
    expr::IntExpr,
    leaf::{Builtin, Leaf, LeafRef},
    types::CoreType,
};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DefinitionId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Value {
    /// Parameter `i` of the enclosing definition.
    Param(usize),
    /// Output `port` of node `node` of the enclosing definition.
    Node { node: usize, port: usize },
}

#[derive(Clone, Debug)]
pub struct Loop {
    pub count: IntExpr,
    pub index_slot: u32,
    pub body: DefinitionId,
    pub bindings: Vec<(String, IntExpr)>,
    pub carried: Vec<Value>,
    pub inputs: Vec<Value>,
    pub until: bool,
}

impl Loop {
    pub fn is_independent(&self) -> bool {
        self.carried.is_empty() && !self.until
    }
}

#[derive(Clone, Debug)]
pub enum Node {
    Leaf { leaf: LeafRef, inputs: Vec<Value> },
    Loop(Loop),
}

#[derive(Clone, Debug)]
pub struct Definition {
    pub name: String,
    pub params: Vec<CoreType>,
    pub nodes: Vec<Node>,
    pub outputs: Vec<Value>,
}

#[derive(Clone, Debug)]
pub struct Program {
    pub name: String,
    pub parameters: Vec<String>,
    pub inputs: Vec<(String, CoreType)>,
    pub outputs: Vec<(String, Value)>,
    pub definitions: Vec<Definition>,
    pub root: DefinitionId,
}

impl Program {
    pub fn definition(&self, id: DefinitionId) -> &Definition {
        &self.definitions[id.0]
    }
}

/// Builds one definition.
pub struct DefinitionBuilder {
    definition: Definition,
}

impl DefinitionBuilder {
    pub fn new(name: impl Into<String>, params: Vec<CoreType>) -> Self {
        Self {
            definition: Definition {
                name: name.into(),
                params,
                nodes: Vec::new(),
                outputs: Vec::new(),
            },
        }
    }

    pub fn param(&self, i: usize) -> Value {
        assert!(i < self.definition.params.len(), "parameter {i} out of range");
        Value::Param(i)
    }

    fn push(&mut self, node: Node, ports: usize) -> Vec<Value> {
        let node_index = self.definition.nodes.len();
        self.definition.nodes.push(node);
        (0..ports).map(|port| Value::Node { node: node_index, port }).collect()
    }

    /// Runs a DSL leaf and returns its outputs.
    pub fn leaf(
        &mut self,
        leaf: &Arc<Leaf>,
        arguments: Vec<IntExpr>,
        inputs: Vec<Value>,
    ) -> Vec<Value> {
        let ports = leaf.signature.outputs.len();
        self.push(
            Node::Leaf { leaf: LeafRef::Dsl { leaf: leaf.clone(), arguments }, inputs },
            ports,
        )
    }

    /// Runs a built-in leaf and returns its single output.
    pub fn builtin(&mut self, op: Builtin, inputs: Vec<Value>) -> Value {
        self.push(Node::Leaf { leaf: LeafRef::Builtin(op), inputs }, 1)[0]
    }

    /// Adds a loop. `outputs` is the number of the body's per-iteration outputs.
    pub fn add_loop(&mut self, lp: Loop, outputs: usize) -> Vec<Value> {
        let ports = lp.carried.len() + outputs;
        self.push(Node::Loop(lp), ports)
    }

    pub fn finish(mut self, outputs: Vec<Value>) -> Definition {
        self.definition.outputs = outputs;
        self.definition
    }
}

/// Builds a program from definitions added bottom-up: a loop's body before the loop.
pub struct ProgramBuilder {
    program: Program,
}

impl ProgramBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            program: Program {
                name: name.into(),
                parameters: Vec::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                definitions: Vec::new(),
                root: DefinitionId(0),
            },
        }
    }

    pub fn parameter(mut self, name: impl Into<String>) -> Self {
        self.program.parameters.push(name.into());
        self
    }

    pub fn add_definition(&mut self, definition: Definition) -> DefinitionId {
        self.program.definitions.push(definition);
        DefinitionId(self.program.definitions.len() - 1)
    }

    /// Finishes with `root`, whose parameters are the named inputs.
    pub fn finish(
        mut self,
        input_names: Vec<String>,
        root: Definition,
        output_names: Vec<String>,
    ) -> Program {
        assert_eq!(input_names.len(), root.params.len(), "one name per root parameter");
        assert_eq!(output_names.len(), root.outputs.len(), "one name per root output");
        self.program.inputs = input_names.into_iter().zip(root.params.iter().cloned()).collect();
        self.program.outputs = output_names.into_iter().zip(root.outputs.iter().copied()).collect();
        self.program.root = self.add_definition(root);
        self.program
    }
}
