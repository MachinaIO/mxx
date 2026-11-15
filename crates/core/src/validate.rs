//! Validation: bind parameters, resolve every type, and check the rules of a structured binary.
//!
//! [`validate`] walks the program from its root and builds one [`Instance`] per definition use:
//! the root, and the body of every loop node. Types never depend on a loop index, so one instance
//! of a body serves every iteration. Validation checks:
//!
//! - every parameter has a value, and every type resolves to a concrete one;
//! - every node reads values produced before it, of the types it expects;
//! - a loop's carried values, inputs, and body outputs match its body's signature, and a loop with
//!   `until` has a `Bool` flag, no per-iteration outputs, and at least one iteration (**S1**);
//! - every window, `get`, and member access stays in range for every value of the loop indices in
//!   scope, using interval bounds;
//! - no `query` leaf runs inside an independent loop (**S4**);
//! - definitions do not call themselves, and nested loops use distinct index slots.
//!
//! **S2** holds by construction: an iteration can write only its own values and its own members.

use crate::{
    expr::{Bindings, Env, ExprError, IntExpr, Interval},
    leaf::{Builtin, LeafRef},
    program::{DefinitionId, Loop, Node, Program, Value},
    types::{ConcreteFamily, ConcreteType, CoreType, TypeError},
};
use std::{collections::BTreeMap, sync::Arc};

#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("in `{definition}`, node {node}: {message}")]
    Node { definition: String, node: usize, message: String },
    #[error("in `{definition}`: {message}")]
    Definition { definition: String, message: String },
    #[error("program parameter `{0}` has no value")]
    MissingParameter(String),
}

/// A validated use of one definition.
#[derive(Clone, Debug)]
pub struct Instance {
    pub definition: DefinitionId,
    /// The definition's parameters, without loop indices.
    pub env: Env,
    /// The loop indices in scope and their ranges, innermost last.
    pub ranges: Vec<(u32, Interval)>,
    pub param_types: Vec<ConcreteType>,
    pub node_types: Vec<Vec<ConcreteType>>,
    /// For each node: the loop count and body instance of a loop node.
    pub loops: Vec<Option<LoopInstance>>,
}

#[derive(Clone, Debug)]
pub struct LoopInstance {
    pub count: u64,
    pub body: Box<Instance>,
}

#[derive(Clone, Debug)]
pub struct ValidatedProgram {
    pub program: Arc<Program>,
    pub bindings: Bindings,
    pub root: Instance,
    pub output_types: Vec<ConcreteType>,
}

impl Instance {
    pub fn value_type(&self, value: Value) -> &ConcreteType {
        match value {
            Value::Param(i) => &self.param_types[i],
            Value::Node { node, port } => &self.node_types[node][port],
        }
    }
}

pub fn validate(
    program: Arc<Program>,
    bindings: Bindings,
) -> Result<ValidatedProgram, ValidationError> {
    for name in &program.parameters {
        if !bindings.contains_key(name) {
            return Err(ValidationError::MissingParameter(name.clone()));
        }
    }
    let env = Env::new(bindings.clone());
    let root_def = program.definition(program.root);
    let param_types =
        root_def.params.iter().map(|t| t.resolve(&env)).collect::<Result<Vec<_>, _>>().map_err(
            |e| ValidationError::Definition {
                definition: root_def.name.clone(),
                message: e.to_string(),
            },
        )?;
    let mut checker = Checker { program: &program, stack: Vec::new() };
    let root = checker.instance(program.root, env, Vec::new(), param_types, false)?;
    let output_types = program.outputs.iter().map(|(_, v)| root.value_type(*v).clone()).collect();
    Ok(ValidatedProgram { program, bindings, root, output_types })
}

struct Checker<'a> {
    program: &'a Program,
    stack: Vec<DefinitionId>,
}

impl Checker<'_> {
    fn instance(
        &mut self,
        id: DefinitionId,
        env: Env,
        ranges: Vec<(u32, Interval)>,
        param_types: Vec<ConcreteType>,
        in_independent_loop: bool,
    ) -> Result<Instance, ValidationError> {
        let definition = self.program.definition(id);
        if self.stack.contains(&id) {
            return Err(ValidationError::Definition {
                definition: definition.name.clone(),
                message: "the definition is used inside itself".into(),
            });
        }
        self.stack.push(id);
        let mut instance = Instance {
            definition: id,
            env,
            ranges,
            param_types,
            node_types: Vec::new(),
            loops: Vec::new(),
        };
        for (index, node) in definition.nodes.iter().enumerate() {
            let fail = |message: String| ValidationError::Node {
                definition: definition.name.clone(),
                node: index,
                message,
            };
            let inputs = match node {
                Node::Leaf { inputs, .. } => inputs.clone(),
                Node::Loop(lp) => lp.carried.iter().chain(&lp.inputs).copied().collect(),
            };
            for value in &inputs {
                let exists = match *value {
                    Value::Param(i) => i < instance.param_types.len(),
                    Value::Node { node, port } => {
                        node < index && port < instance.node_types[node].len()
                    }
                };
                if !exists {
                    return Err(fail(format!(
                        "reads {value:?}, which is not produced before this node"
                    )));
                }
            }
            let input_types: Vec<ConcreteType> =
                inputs.iter().map(|v| instance.value_type(*v).clone()).collect();
            let (outputs, loop_instance) = match node {
                Node::Leaf { leaf, .. } => {
                    if in_independent_loop &&
                        matches!(leaf, LeafRef::Builtin(Builtin::Query { .. }))
                    {
                        return Err(fail(
                            "a query leaf may not run inside an independent loop (S4)".into(),
                        ));
                    }
                    (
                        leaf_outputs(leaf, &input_types, &instance.env, &instance.ranges)
                            .map_err(fail)?,
                        None,
                    )
                }
                Node::Loop(lp) => {
                    let (outputs, body) = self
                        .loop_outputs(lp, &input_types, &instance, in_independent_loop)
                        .map_err(|e| match e {
                            LoopError::Message(m) => fail(m),
                            LoopError::Inner(e) => e,
                        })?;
                    (outputs, Some(body))
                }
            };
            instance.node_types.push(outputs);
            instance.loops.push(loop_instance);
        }
        for value in &definition.outputs {
            let exists = match *value {
                Value::Param(i) => i < instance.param_types.len(),
                Value::Node { node, port } => {
                    node < instance.node_types.len() && port < instance.node_types[node].len()
                }
            };
            if !exists {
                return Err(ValidationError::Definition {
                    definition: definition.name.clone(),
                    message: format!("output {value:?} is not produced"),
                });
            }
        }
        self.stack.pop();
        Ok(instance)
    }

    fn loop_outputs(
        &mut self,
        lp: &Loop,
        input_types: &[ConcreteType],
        caller: &Instance,
        in_independent_loop: bool,
    ) -> Result<(Vec<ConcreteType>, LoopInstance), LoopError> {
        let count = eval_static(&lp.count, &caller.env).map_err(LoopError::Message)?;
        let count = u64::try_from(count)
            .map_err(|_| LoopError::Message(format!("negative loop count {count}")))?;
        if lp.until && count == 0 {
            return Err(LoopError::Message("an `until` loop needs at least one iteration".into()));
        }
        if caller.ranges.iter().any(|(slot, _)| *slot == lp.index_slot) {
            return Err(LoopError::Message(format!(
                "index slot {} is already used by an enclosing loop",
                lp.index_slot
            )));
        }
        let bindings = lp
            .bindings
            .iter()
            .map(|(name, expr)| Ok((name.clone(), eval_static(expr, &caller.env)?)))
            .collect::<Result<Vec<_>, String>>()
            .map_err(LoopError::Message)?;
        // The body sees the caller's known indices, and its own when the loop runs once.
        let mut body_env =
            Env { vars: caller.env.vars.clone(), indices: caller.env.indices.clone() }
                .with_vars(bindings);
        if count == 1 {
            body_env = body_env.with_index(lp.index_slot, 0);
        }
        let body_def = self.program.definition(lp.body);
        let body_params = body_def
            .params
            .iter()
            .map(|t| t.resolve(&body_env))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| LoopError::Message(format!("body parameter types: {e}")))?;
        if body_params != input_types {
            return Err(LoopError::Message(format!(
                "the body `{}` takes {body_params:?}, but the loop passes {input_types:?}",
                body_def.name
            )));
        }
        let mut ranges = caller.ranges.clone();
        ranges.push((lp.index_slot, Interval { min: 0, max: count.saturating_sub(1) as i128 }));
        let body = self
            .instance(
                lp.body,
                body_env,
                ranges,
                body_params,
                in_independent_loop || lp.is_independent(),
            )
            .map_err(LoopError::Inner)?;
        let outputs: Vec<ConcreteType> =
            body_def.outputs.iter().map(|v| body.value_type(*v).clone()).collect();
        let carried = lp.carried.len();
        let flag = usize::from(lp.until);
        if outputs.len() < carried + flag || outputs[..carried] != input_types[..carried] {
            return Err(LoopError::Message(
                "the body's first outputs must be the next carried values".into(),
            ));
        }
        if lp.until {
            if outputs[carried] != ConcreteType::Bool {
                return Err(LoopError::Message(
                    "the stop flag of an `until` loop must be a Bool".into(),
                ));
            }
            if outputs.len() > carried + 1 {
                return Err(LoopError::Message(
                    "an `until` loop has no per-iteration outputs".into(),
                ));
            }
        }
        let mut result: Vec<ConcreteType> = outputs[..carried].to_vec();
        result.extend(outputs[carried + flag..].iter().map(|element| {
            ConcreteType::Family(ConcreteFamily::Uniform {
                element: Box::new(element.clone()),
                count,
            })
        }));
        Ok((result, LoopInstance { count, body: Box::new(body) }))
    }
}

enum LoopError {
    Message(String),
    Inner(ValidationError),
}

fn eval_static(expr: &IntExpr, env: &Env) -> Result<i128, String> {
    expr.eval(env).map_err(|e| match e {
        ExprError::IndexDependent(slot) => {
            format!("{expr:?} depends on loop index {slot}, but sizes and counts may not")
        }
        e => e.to_string(),
    })
}

fn resolve(ty: &CoreType, env: &Env) -> Result<ConcreteType, String> {
    ty.resolve(env).map_err(|e| match e {
        TypeError::Expr(ExprError::IndexDependent(slot)) => {
            format!("a type depends on loop index {slot}")
        }
        e => e.to_string(),
    })
}

/// The output types of a leaf with these input types.
fn leaf_outputs(
    leaf: &LeafRef,
    inputs: &[ConcreteType],
    env: &Env,
    ranges: &[(u32, Interval)],
) -> Result<Vec<ConcreteType>, String> {
    let bound = |expr: &IntExpr| expr.bound(env, ranges).map_err(|e| e.to_string());
    let arity = |n: usize| {
        if inputs.len() == n {
            Ok(())
        } else {
            Err(format!("expects {n} inputs, got {}", inputs.len()))
        }
    };
    let buffer_bytes = |ty: &ConcreteType| match ty {
        ConcreteType::Buffer { bytes, .. } => Ok(*bytes),
        other => Err(format!("expects a buffer, got {other:?}")),
    };
    match leaf {
        LeafRef::Dsl { leaf, arguments } => {
            let signature = &leaf.signature;
            if arguments.len() != signature.arguments.len() {
                return Err(format!(
                    "`{}` takes {} arguments, got {}",
                    signature.name,
                    signature.arguments.len(),
                    arguments.len()
                ));
            }
            for (i, argument) in arguments.iter().enumerate() {
                let b = bound(argument)?;
                if b.min < i128::from(i64::MIN) || b.max > i128::from(i64::MAX) {
                    return Err(format!(
                        "argument {i} of `{}` may not fit in 64 bits: {b:?}",
                        signature.name
                    ));
                }
            }
            let substitution: BTreeMap<String, IntExpr> =
                signature.arguments.iter().cloned().zip(arguments.iter().cloned()).collect();
            arity(signature.inputs.len())?;
            for (i, (expected, actual)) in signature.inputs.iter().zip(inputs).enumerate() {
                let expected = resolve(&expected.substitute_vars(&substitution), env)?;
                if &expected != actual {
                    return Err(format!(
                        "input {i} of `{}` is {actual:?}, expected {expected:?}",
                        signature.name
                    ));
                }
            }
            resolve_scratch(&signature.scratch_bytes.substitute_vars(&substitution), env)?;
            signature
                .outputs
                .iter()
                .map(|t| resolve(&t.substitute_vars(&substitution), env))
                .collect()
        }
        LeafRef::Builtin(op) => match op {
            Builtin::Window { offset, runs, length, stride } => {
                arity(1)?;
                let bytes = buffer_bytes(&inputs[0])?;
                let runs = eval_static(runs, env)?;
                let length = eval_static(length, env)?;
                if runs < 0 || length < 0 {
                    return Err("a window has negative size".into());
                }
                if runs > 0 && length > 0 {
                    let last = bound(&(offset.clone() + (stride.clone() * (runs - 1))))?;
                    let first = bound(offset)?;
                    let stride_bound = bound(stride)?;
                    if first.min < 0 ||
                        stride_bound.min < 0 ||
                        last.max + length > i128::from(bytes)
                    {
                        return Err(format!("the window may leave its {bytes}-byte buffer"));
                    }
                }
                Ok(vec![ConcreteType::Buffer { bytes: (runs * length) as u64 }])
            }
            Builtin::Concat => {
                // Any sized value concatenates: buffers, integers, Bools, and families of them
                // (a family's encoding is its members', in order).
                let total = inputs
                    .iter()
                    .map(|ty| ty.size().ok_or_else(|| format!("cannot concatenate {ty:?}")))
                    .sum::<Result<u64, String>>()?;
                Ok(vec![ConcreteType::Buffer { bytes: total }])
            }
            Builtin::Pack => Ok(vec![ConcreteFamily::from_members(inputs.to_vec()).into()]),
            Builtin::Get { index } => {
                arity(1)?;
                let family = inputs[0].family().ok_or("`get` expects a family")?;
                let b = bound(index)?;
                if b.min < 0 || b.max >= i128::from(family.len()) {
                    return Err(format!("member index {b:?} may leave 0..{}", family.len()));
                }
                match family {
                    ConcreteFamily::Uniform { element, .. } => Ok(vec![(**element).clone()]),
                    ConcreteFamily::Dependent(members) if b.min == b.max => {
                        Ok(vec![members[b.min as usize].clone()])
                    }
                    ConcreteFamily::Dependent(_) => {
                        Err("a member of a dependent family needs an index that does not vary"
                            .into())
                    }
                }
            }
            Builtin::GetDynamic => {
                arity(2)?;
                match (&inputs[0], &inputs[1]) {
                    (
                        ConcreteType::Family(ConcreteFamily::Uniform { element, .. }),
                        ConcreteType::Int { .. },
                    ) => Ok(vec![(**element).clone()]),
                    _ => Err("`get_dynamic` expects a uniform family and an Int".into()),
                }
            }
            Builtin::Select => {
                let (selector, candidates) =
                    inputs.split_first().ok_or("`select` expects a selector")?;
                if !matches!(selector, ConcreteType::Int { .. }) {
                    return Err("the selector of `select` must be an Int".into());
                }
                let first = candidates.first().ok_or("`select` expects at least one candidate")?;
                if candidates.iter().any(|c| c != first) {
                    return Err("the candidates of `select` must have one type".into());
                }
                Ok(vec![first.clone()])
            }
            Builtin::Coins { bytes } => {
                arity(0)?;
                let bytes =
                    u64::try_from(eval_static(bytes, env)?).map_err(|_| "negative coin count")?;
                Ok(vec![ConcreteType::Buffer { bytes }])
            }
            Builtin::Constant { ty, bytes } => {
                arity(0)?;
                let ty = ty.resolve(env).map_err(|e| e.to_string())?;
                if ty.size() != Some(bytes.len() as u64) {
                    return Err(format!(
                        "a constant of {} bytes is not of its type {ty:?}",
                        bytes.len()
                    ));
                }
                Ok(vec![ty])
            }
            Builtin::Query { response_bytes } => {
                arity(1)?;
                buffer_bytes(&inputs[0])?;
                let bytes = u64::try_from(eval_static(response_bytes, env)?)
                    .map_err(|_| "negative response size")?;
                Ok(vec![ConcreteType::Buffer { bytes }])
            }
        },
    }
}

fn resolve_scratch(expr: &IntExpr, env: &Env) -> Result<u64, String> {
    let bytes = eval_static(expr, env)?;
    u64::try_from(bytes).map_err(|_| format!("negative scratch size {bytes}"))
}

/// The scratch bytes of a DSL leaf call in `env`.
pub fn scratch_bytes(leaf: &LeafRef, env: &Env) -> u64 {
    match leaf {
        LeafRef::Dsl { leaf, arguments } => {
            let substitution: BTreeMap<String, IntExpr> =
                leaf.signature.arguments.iter().cloned().zip(arguments.iter().cloned()).collect();
            resolve_scratch(&leaf.signature.scratch_bytes.substitute_vars(&substitution), env)
                .expect("validated scratch size")
        }
        LeafRef::Builtin(_) => 0,
    }
}
