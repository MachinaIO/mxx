//! Cost estimation from the structure alone.
//!
//! [`leaf_counts`] gives how many times each leaf runs, in closed form over loop counts: a loop
//! multiplies its body's counts by its count, and an `until` loop is charged its full count, an
//! upper bound. [`call_counts`] refines this by call shape ([`CallKey`]): a DSL leaf with its
//! arguments and input and output sizes, or a built-in with the bytes it moves.
//!
//! [`estimate_host`] extrapolates a whole host run from measurements of single calls: each
//! distinct call shape runs once and twice on zero inputs, the difference is its time per call,
//! and the prediction is the sum over shapes of count times that time. The leaf contract's L1
//! (time depends on arguments and sizes, not on input bytes) makes measuring on zero inputs and
//! scaling by counts valid. An argument that depends on a loop index is measured at index 0.
//!
//! [`value_bytes`] is the storage of every value of every definition use, the memory a run needs
//! when nothing is reused.

use crate::{
    expr::Env,
    leaf::{Builtin, HostCall, LeafRef},
    program::Node,
    validate::{Instance, ValidatedProgram, scratch_bytes},
};
use std::{collections::BTreeMap, time::Instant};

/// The shape of one call: what decides its cost under the leaf contract.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CallKey {
    pub leaf: String,
    /// The arguments of a DSL leaf; empty for a built-in.
    pub arguments: Vec<i64>,
    pub input_bytes: Vec<u64>,
    pub output_bytes: Vec<u64>,
}

/// How many times each call shape runs.
pub fn call_counts(program: &ValidatedProgram) -> BTreeMap<CallKey, u128> {
    let mut counts = BTreeMap::new();
    visit(program, &program.root, 1, &mut |key, times, _, _| {
        *counts.entry(key).or_default() += times
    });
    counts
}

/// Calls `f` with every leaf call's shape, how many times it runs, its leaf, and its definition
/// use.
fn visit(
    program: &ValidatedProgram,
    instance: &Instance,
    times: u128,
    f: &mut impl FnMut(CallKey, u128, &LeafRef, &Instance),
) {
    let definition = program.program.definition(instance.definition);
    for (index, node) in definition.nodes.iter().enumerate() {
        match node {
            Node::Leaf { leaf, inputs } => {
                let arguments = match leaf {
                    LeafRef::Dsl { arguments, .. } => arguments
                        .iter()
                        .map(|a| {
                            let at_zero = a
                                .bound(&instance.env, &zero_ranges(instance))
                                .expect("validated argument");
                            i64::try_from(at_zero.min).expect("validated argument range")
                        })
                        .collect(),
                    LeafRef::Builtin(_) => Vec::new(),
                };
                let key = CallKey {
                    leaf: leaf.name().to_owned(),
                    arguments,
                    input_bytes: inputs
                        .iter()
                        .map(|v| instance.value_type(*v).size().unwrap_or(0))
                        .collect(),
                    output_bytes: instance.node_types[index]
                        .iter()
                        .map(|t| t.size().unwrap_or(0))
                        .collect(),
                };
                f(key, times, leaf, instance);
            }
            Node::Loop(_) => {
                let lp = instance.loops[index].as_ref().expect("validated loop");
                visit(program, &lp.body, times * u128::from(lp.count), f);
            }
        }
    }
}

/// Every loop index in scope pinned to 0.
fn zero_ranges(instance: &Instance) -> Vec<(u32, crate::expr::Interval)> {
    instance.ranges.iter().map(|(slot, _)| (*slot, crate::expr::Interval::point(0))).collect()
}

/// A predicted host run.
#[derive(Clone, Debug)]
pub struct HostEstimate {
    pub seconds: f64,
    /// For each call shape: how many times it runs and its measured seconds per call.
    pub calls: Vec<(CallKey, u128, f64)>,
}

/// Predicts the time of [`crate::host::run`] from measurements of single calls.
pub fn estimate_host(program: &ValidatedProgram) -> HostEstimate {
    let mut sites: BTreeMap<CallKey, (u128, LeafRef, Env)> = BTreeMap::new();
    visit(program, &program.root, 1, &mut |key, times, leaf, instance| {
        let env = zero_ranges(instance)
            .iter()
            .fold(instance.env.clone(), |e, (slot, _)| e.with_index(*slot, 0));
        sites.entry(key).or_insert_with(|| (0, leaf.clone(), env)).0 += times;
    });
    let mut calls = Vec::new();
    let mut seconds = 0.0;
    for (key, (count, leaf, env)) in sites {
        let each = time_per_call(&leaf, &key, &env);
        seconds += each * count as f64;
        calls.push((key, count, each));
    }
    HostEstimate { seconds, calls }
}

/// The seconds of one call: the difference between running it twice and once.
fn time_per_call(leaf: &LeafRef, key: &CallKey, env: &Env) -> f64 {
    let run = |times: usize| {
        let start = Instant::now();
        for _ in 0..times {
            run_once(leaf, key, env);
        }
        start.elapsed().as_secs_f64()
    };
    run(1);
    let (one, two) = (run(1), run(2));
    (two - one).max(one / 2.0).max(0.0)
}

fn run_once(leaf: &LeafRef, key: &CallKey, env: &Env) {
    let inputs: Vec<Vec<u8>> = key.input_bytes.iter().map(|n| vec![0u8; *n as usize]).collect();
    let mut outputs: Vec<Vec<u8>> =
        key.output_bytes.iter().map(|n| vec![0u8; *n as usize]).collect();
    match leaf {
        LeafRef::Dsl { leaf: dsl, .. } => {
            let Some(kernel) = dsl.host.as_ref() else { return };
            let mut scratch = vec![0u8; scratch_bytes(leaf, env) as usize];
            let slices: Vec<&[u8]> = inputs.iter().map(Vec::as_slice).collect();
            kernel.run(HostCall {
                arguments: &key.arguments,
                inputs: &slices,
                outputs: &mut outputs,
                scratch: &mut scratch,
            });
        }
        // A built-in moves its output's bytes.
        LeafRef::Builtin(Builtin::Coins { .. } | Builtin::Query { .. }) => {}
        LeafRef::Builtin(Builtin::Constant { bytes, .. }) => outputs[0].copy_from_slice(bytes),
        LeafRef::Builtin(_) => {
            for (out, input) in outputs.iter_mut().zip(inputs.iter().cycle()) {
                let n = out.len().min(input.len());
                out[..n].copy_from_slice(&input[..n]);
            }
        }
    }
    std::hint::black_box(&outputs);
}

/// How many times each leaf runs, by leaf name.
pub fn leaf_counts(program: &ValidatedProgram) -> BTreeMap<String, u128> {
    let mut counts = BTreeMap::new();
    visit(program, &program.root, 1, &mut |key, times, _, _| {
        *counts.entry(key.leaf).or_default() += times
    });
    counts
}

/// The bytes of every value of every definition use.
pub fn value_bytes(program: &ValidatedProgram) -> u128 {
    fn walk(instance: &Instance) -> u128 {
        let own: u128 =
            instance.node_types.iter().flatten().filter_map(|t| t.size()).map(u128::from).sum();
        own + instance.loops.iter().flatten().map(|lp| walk(&lp.body)).sum::<u128>()
    }
    let inputs: u128 =
        program.root.param_types.iter().filter_map(|t| t.size()).map(u128::from).sum();
    inputs + walk(&program.root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        expr::{Bindings, IntExpr},
        host,
        leaf::{HostCall, Leaf, LeafSignature},
        program::{DefinitionBuilder, Loop, ProgramBuilder},
        types::CoreType,
        validate::validate,
    };
    use std::sync::Arc;

    /// A leaf whose time grows with `n`: it mixes its input `rounds` times into its output.
    fn work_leaf() -> Arc<Leaf> {
        Arc::new(Leaf {
            signature: LeafSignature {
                name: "work".into(),
                arguments: vec!["n".into(), "rounds".into()],
                inputs: vec![CoreType::buffer(8, "n")],
                outputs: vec![CoreType::buffer(8, "n")],
                scratch_bytes: 0.into(),
                in_place: vec![],
                associative: false,
            },
            rv64: None,
            host: Some(Arc::new(|call: HostCall<'_>| {
                let mut total = 0u64;
                for _ in 0..call.arguments[1] {
                    for chunk in call.inputs[0].chunks(8) {
                        total = total
                            .wrapping_add(u64::from_le_bytes(chunk.try_into().unwrap()))
                            .rotate_left(1);
                    }
                }
                call.outputs[0][..8].copy_from_slice(&std::hint::black_box(total).to_le_bytes());
            })),
        })
    }

    #[test]
    fn test_host_estimate_counts_calls_and_predicts_the_run() {
        let n: i128 = std::env::var("MXX_CORE_TEST_ESTIMATE_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);
        let count: i128 = 24;
        let leaf = work_leaf();
        let mut builder = ProgramBuilder::new("estimate").parameter("n");
        let mut body = DefinitionBuilder::new("body", vec![CoreType::buffer(8, "n")]);
        let input = body.param(0);
        let out = body.leaf(&leaf, vec!["n".into(), 64.into()], vec![input])[0];
        let body = builder.add_definition(body.finish(vec![out]));
        let mut root = DefinitionBuilder::new("root", vec![CoreType::buffer(8, "n")]);
        let input = root.param(0);
        // A carried value makes the loop sequential, so the run is the sum of its calls.
        let last = root.add_loop(
            Loop {
                count: IntExpr::from(count),
                index_slot: 0,
                body,
                bindings: vec![],
                carried: vec![input],
                inputs: vec![],
                until: false,
            },
            0,
        )[0];
        let program = builder.finish(vec!["x".into()], root.finish(vec![last]), vec!["y".into()]);
        let program = validate(Arc::new(program), Bindings::from([("n".into(), n)])).unwrap();
        let counts = call_counts(&program);
        let work = counts.iter().find(|(k, _)| k.leaf == "work").unwrap();
        assert_eq!(*work.1, count as u128);
        assert_eq!(work.0.arguments, vec![n as i64, 64]);
        let estimate = estimate_host(&program);
        let start = Instant::now();
        host::run(&program, vec![vec![1u8; 8 * n as usize]], None).unwrap();
        let actual = start.elapsed().as_secs_f64();
        let ratio = estimate.seconds / actual;
        assert!(
            (0.2..5.0).contains(&ratio),
            "predicted {:.3e} s, ran {actual:.3e} s",
            estimate.seconds
        );
    }
}
