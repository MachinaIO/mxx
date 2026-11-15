//! Host execution of a validated program.
//!
//! Every value is held in its byte encoding ([`crate::types`]). DSL leaves run their
//! [`HostKernel`](crate::leaf::HostKernel); the built-in leaves run here. The iterations of an
//! independent loop run concurrently with rayon, in batches of
//! [`host_loop_batch`] iterations.

use crate::{
    env::host_loop_batch,
    expr::Env,
    leaf::{Builtin, HostCall, LeafRef},
    program::{Node, Value},
    types::{ConcreteFamily, ConcreteType},
    validate::{Instance, ValidatedProgram, scratch_bytes},
};
use rand::RngCore;
use rayon::prelude::*;
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum HostError {
    #[error("input {index} has {actual} bytes, expected {expected}")]
    InputSize { index: usize, actual: usize, expected: u64 },
    #[error("expected {expected} inputs, got {actual}")]
    InputCount { expected: usize, actual: usize },
    #[error("leaf `{0}` has no host kernel")]
    NoHostKernel(String),
    #[error("leaf `{0}` resized an output")]
    OutputResized(String),
    #[error("index {index} is outside 0..{len}")]
    IndexOutOfRange { index: i128, len: u64 },
    #[error("a Bool value is {0}, not 0 or 1")]
    BadBool(u64),
    #[error("an `until` loop ran all {0} iterations without stopping")]
    UntilExhausted(u64),
    #[error("the program queries an external party, but none was given")]
    NoQueryHandler,
    #[error("a query response has {actual} bytes, expected {expected}")]
    QueryResponseSize { actual: usize, expected: u64 },
    #[error("expression: {0}")]
    Expr(#[from] crate::expr::ExprError),
}

/// Answers `query` leaves: a request in, a response out.
pub type QueryHandler<'a> = &'a (dyn Fn(&[u8]) -> Vec<u8> + Sync);

type Bytes = Arc<[u8]>;

/// Runs the program on its inputs (in their byte encodings) and returns its outputs.
pub fn run(
    program: &ValidatedProgram,
    inputs: Vec<Vec<u8>>,
    query: Option<QueryHandler<'_>>,
) -> Result<Vec<Vec<u8>>, HostError> {
    let root = &program.root;
    if inputs.len() != root.param_types.len() {
        return Err(HostError::InputCount {
            expected: root.param_types.len(),
            actual: inputs.len(),
        });
    }
    for (index, (input, ty)) in inputs.iter().zip(&root.param_types).enumerate() {
        let expected = ty.size().expect("program inputs have sizes");
        if input.len() as u64 != expected {
            return Err(HostError::InputSize { index, actual: input.len(), expected });
        }
    }
    let executor = Executor { program, query };
    let params: Vec<Bytes> = inputs.into_iter().map(Bytes::from).collect();
    let values = executor.instance(root, &root.env, &params)?;
    Ok(program.program.outputs.iter().map(|(_, v)| lookup(&params, &values, *v).to_vec()).collect())
}

struct Executor<'a> {
    program: &'a ValidatedProgram,
    query: Option<QueryHandler<'a>>,
}

fn lookup<'v>(params: &'v [Bytes], values: &'v [Vec<Bytes>], value: Value) -> &'v Bytes {
    match value {
        Value::Param(i) => &params[i],
        Value::Node { node, port } => &values[node][port],
    }
}

impl Executor<'_> {
    /// Runs one use of a definition and returns the values of all its nodes.
    fn instance(
        &self,
        instance: &Instance,
        env: &Env,
        params: &[Bytes],
    ) -> Result<Vec<Vec<Bytes>>, HostError> {
        let definition = self.program.program.definition(instance.definition);
        let mut values: Vec<Vec<Bytes>> = Vec::with_capacity(definition.nodes.len());
        for (index, node) in definition.nodes.iter().enumerate() {
            let outputs = match node {
                Node::Leaf { leaf, inputs } => {
                    let inputs: Vec<&Bytes> =
                        inputs.iter().map(|v| lookup(params, &values, *v)).collect();
                    let input_types: Vec<&ConcreteType> =
                        node_inputs(node).iter().map(|v| instance.value_type(*v)).collect();
                    self.leaf(leaf, &inputs, &input_types, &instance.node_types[index], env)?
                }
                Node::Loop(lp) => {
                    let lp_instance = instance.loops[index].as_ref().expect("validated loop");
                    let carried: Vec<Bytes> =
                        lp.carried.iter().map(|v| lookup(params, &values, *v).clone()).collect();
                    let inputs: Vec<Bytes> =
                        lp.inputs.iter().map(|v| lookup(params, &values, *v).clone()).collect();
                    let body = &lp_instance.body;
                    let body_env = body.env.clone();
                    let iteration_env = |i: u64| {
                        let mut e = body_env.clone();
                        e.indices = env.indices.clone();
                        e.with_index(lp.index_slot, i128::from(i))
                    };
                    let body_def = self.program.program.definition(body.definition);
                    let outputs_of = |values: &[Vec<Bytes>], params: &[Bytes]| -> Vec<Bytes> {
                        body_def
                            .outputs
                            .iter()
                            .map(|v| lookup(params, values, *v).clone())
                            .collect()
                    };
                    if lp.is_independent() {
                        let mut members: Vec<Vec<Bytes>> =
                            Vec::with_capacity(lp_instance.count as usize);
                        let batch = host_loop_batch() as u64;
                        let mut start = 0;
                        while start < lp_instance.count {
                            let end = (start + batch).min(lp_instance.count);
                            let results = (start..end)
                                .into_par_iter()
                                .map(|i| {
                                    let values = self.instance(body, &iteration_env(i), &inputs)?;
                                    Ok(outputs_of(&values, &inputs))
                                })
                                .collect::<Result<Vec<_>, HostError>>()?;
                            members.extend(results);
                            start = end;
                        }
                        (0..body_def.outputs.len())
                            .map(|port| {
                                Bytes::from(
                                    members
                                        .iter()
                                        .flat_map(|m| m[port].iter().copied())
                                        .collect::<Vec<u8>>(),
                                )
                            })
                            .collect()
                    } else {
                        let mut state = carried;
                        let mut members: Vec<Vec<Bytes>> = Vec::new();
                        let mut stopped = !lp.until;
                        for i in 0..lp_instance.count {
                            let body_params: Vec<Bytes> =
                                state.iter().chain(&inputs).cloned().collect();
                            let values = self.instance(body, &iteration_env(i), &body_params)?;
                            let outputs = outputs_of(&values, &body_params);
                            let carried_count = lp.carried.len();
                            state = outputs[..carried_count].to_vec();
                            if lp.until {
                                if read_bool(&outputs[carried_count])? {
                                    stopped = true;
                                    break;
                                }
                            } else {
                                members.push(outputs[carried_count..].to_vec());
                            }
                        }
                        if !stopped {
                            return Err(HostError::UntilExhausted(lp_instance.count));
                        }
                        let per_iteration =
                            body_def.outputs.len() - lp.carried.len() - usize::from(lp.until);
                        state
                            .into_iter()
                            .chain((0..per_iteration).map(|port| {
                                Bytes::from(
                                    members
                                        .iter()
                                        .flat_map(|m| m[port].iter().copied())
                                        .collect::<Vec<u8>>(),
                                )
                            }))
                            .collect()
                    }
                }
            };
            values.push(outputs);
        }
        Ok(values)
    }

    fn leaf(
        &self,
        leaf: &LeafRef,
        inputs: &[&Bytes],
        input_types: &[&ConcreteType],
        output_types: &[ConcreteType],
        env: &Env,
    ) -> Result<Vec<Bytes>, HostError> {
        let index_in = |index: i128, len: u64| {
            u64::try_from(index)
                .ok()
                .filter(|i| *i < len)
                .ok_or(HostError::IndexOutOfRange { index, len })
        };
        Ok(match leaf {
            LeafRef::Dsl { leaf: dsl, arguments } => {
                let kernel = dsl
                    .host
                    .as_ref()
                    .ok_or_else(|| HostError::NoHostKernel(dsl.signature.name.clone()))?;
                let arguments = arguments
                    .iter()
                    .map(|a| Ok(i64::try_from(a.eval(env)?).expect("validated argument range")))
                    .collect::<Result<Vec<i64>, HostError>>()?;
                let mut outputs: Vec<Vec<u8>> = output_types
                    .iter()
                    .map(|t| vec![0u8; t.size().expect("leaf outputs have sizes") as usize])
                    .collect();
                let sizes: Vec<usize> = outputs.iter().map(Vec::len).collect();
                let mut scratch = vec![0u8; scratch_bytes(leaf, env) as usize];
                let input_slices: Vec<&[u8]> = inputs.iter().map(|b| &b[..]).collect();
                kernel.run(HostCall {
                    arguments: &arguments,
                    inputs: &input_slices,
                    outputs: &mut outputs,
                    scratch: &mut scratch,
                });
                if outputs.iter().map(Vec::len).ne(sizes) {
                    return Err(HostError::OutputResized(dsl.signature.name.clone()));
                }
                outputs.into_iter().map(Bytes::from).collect()
            }
            LeafRef::Builtin(op) => vec![match op {
                Builtin::Window { offset, runs, length, stride } => {
                    let (offset, runs, length, stride) =
                        (offset.eval(env)?, runs.eval(env)?, length.eval(env)?, stride.eval(env)?);
                    let mut out = Vec::with_capacity((runs * length) as usize);
                    for r in 0..runs {
                        let start = (offset + r * stride) as usize;
                        out.extend_from_slice(&inputs[0][start..start + length as usize]);
                    }
                    Bytes::from(out)
                }
                Builtin::Concat | Builtin::Pack => {
                    Bytes::from(inputs.iter().flat_map(|b| b.iter().copied()).collect::<Vec<u8>>())
                }
                Builtin::Get { index } => {
                    let family = input_types[0].family().expect("validated family");
                    let index = index_in(index.eval(env)?, family.len())?;
                    member(inputs[0], family, index)
                }
                Builtin::GetDynamic => {
                    let family = input_types[0].family().expect("validated family");
                    let index = index_in(read_int(inputs[1]), family.len())?;
                    member(inputs[0], family, index)
                }
                Builtin::Select => {
                    let candidates = inputs.len() as u64 - 1;
                    let selector = index_in(read_int(inputs[0]), candidates)?;
                    inputs[1 + selector as usize].clone()
                }
                Builtin::Coins { bytes } => {
                    let mut out = vec![0u8; bytes.eval(env)? as usize];
                    rand::rng().fill_bytes(&mut out);
                    Bytes::from(out)
                }
                Builtin::Constant { bytes, .. } => Bytes::from(bytes.clone()),
                Builtin::Query { response_bytes } => {
                    let query = self.query.ok_or(HostError::NoQueryHandler)?;
                    let response = query(inputs[0]);
                    let expected = response_bytes.eval(env)? as u64;
                    if response.len() as u64 != expected {
                        return Err(HostError::QueryResponseSize {
                            actual: response.len(),
                            expected,
                        });
                    }
                    Bytes::from(response)
                }
            }],
        })
    }
}

fn node_inputs(node: &Node) -> Vec<Value> {
    match node {
        Node::Leaf { inputs, .. } => inputs.clone(),
        Node::Loop(lp) => lp.carried.iter().chain(&lp.inputs).copied().collect(),
    }
}

fn member(bytes: &Bytes, family: &ConcreteFamily, index: u64) -> Bytes {
    let start = family.member_offset(index).expect("validated member") as usize;
    let end = family.member_offset(index + 1).expect("validated member") as usize;
    Bytes::from(&bytes[start..end])
}

/// The value of an `Int` encoding, saturated to the `i128` range.
pub fn read_int(bytes: &[u8]) -> i128 {
    let negative = bytes.last().is_some_and(|b| b & 0x80 != 0);
    let words: Vec<u64> =
        bytes.chunks(8).map(|c| u64::from_le_bytes(c.try_into().expect("whole words"))).collect();
    let fill = if negative { u64::MAX } else { 0 };
    if words.iter().skip(2).any(|w| *w != fill) {
        return if negative { i128::MIN } else { i128::MAX };
    }
    let low = u128::from(words.first().copied().unwrap_or(fill));
    let high = u128::from(words.get(1).copied().unwrap_or(fill));
    (low | (high << 64)) as i128
}

/// The encoding of `value` as an `Int` of `words` words.
pub fn write_int(value: i128, words: u32) -> Vec<u8> {
    let fill: u64 = if value < 0 { u64::MAX } else { 0 };
    (0..words)
        .flat_map(|w| {
            let word = match w {
                0 => value as u64,
                1 => (value >> 64) as u64,
                _ => fill,
            };
            word.to_le_bytes()
        })
        .collect()
}

pub fn read_bool(bytes: &[u8]) -> Result<bool, HostError> {
    match u64::from_le_bytes(bytes.try_into().expect("a Bool is one word")) {
        0 => Ok(false),
        1 => Ok(true),
        other => Err(HostError::BadBool(other)),
    }
}
