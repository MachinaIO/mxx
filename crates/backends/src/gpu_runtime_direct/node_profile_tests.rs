#![cfg(feature = "gpu")]

//! Planning with node profiling predicts each node's share of one execute.

use crate::{
    GpuRuntime, RuntimeValue, backend::poly_gpu::gpu_backend, poly::dcrt::gpu::GpuDCRTPolyParams,
};
use mxx_dsl::{DslContext, HashTag, Ring, parallel};
use mxx_ir_core::{FrozenGraphScopeId, IntExpr, NodeId, ParamEnv, generate_crt_basis};
use std::{collections::BTreeMap, sync::Arc};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).map_or(default, |value| value.parse().expect("a usize test parameter"))
}

#[test]
#[ignore = "requires a CUDA GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_plan_node_costs_divide_the_predicted_time() {
    let ring_dimension = env_usize("MXX_TEST_NODE_PROFILE_RING_DIMENSION", 1024) as u32;
    let columns = env_usize("MXX_TEST_NODE_PROFILE_COLUMNS", 16);
    let lanes = env_usize("MXX_TEST_NODE_PROFILE_LANES", 3);
    let moduli = generate_crt_basis(ring_dimension, 2, 50).unwrap();
    let gpu_params = GpuDCRTPolyParams::new(ring_dimension, moduli.clone(), 10, None);
    let ring =
        Ring::from_crt_moduli(moduli.into_iter().map(IntExpr::from).collect(), ring_dimension);
    let seed = ring.bytes_input("seed", 32);
    let a =
        ring.hash_matrix(seed.clone(), HashTag::from(b"node-profile/a".as_slice()), (1, columns));
    let b = ring.hash_matrix(seed, HashTag::from(b"node-profile/b".as_slice()), (columns, columns));
    let cheap = a.clone() + a.clone();
    let body_b = b.clone();
    let expensive = parallel(lanes, move |_| Ok(&(&a * &body_b) * &body_b)).unwrap();
    let validated = DslContext::new("node-profile")
        .output("cheap", cheap)
        .unwrap()
        .output("expensive", expensive)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let inputs = BTreeMap::from([(
        "seed".to_owned(),
        RuntimeValue::Bytes(Arc::from(rand::random::<[u8; 32]>())),
    )]);
    let mut runtime = GpuRuntime::new(gpu_backend([gpu_params])).unwrap();
    runtime.options_mut().profile_nodes = true;
    let mut plan = runtime.plan(validated, &inputs).unwrap();
    let report = plan.report().clone();
    let cost = |scope: &FrozenGraphScopeId, node: NodeId| {
        report.node_costs.iter().find(|cost| &cost.scope == scope && cost.node == node)
    };
    let root = plan.graph().source.root_scope().clone();
    let root_total = report
        .node_costs
        .iter()
        .filter(|cost| cost.scope == FrozenGraphScopeId::Root)
        .map(|cost| cost.total_seconds)
        .sum::<f64>();
    // Every operation belongs to a node, so the root nodes divide the whole
    // predicted time.
    assert!(
        (root_total - report.predicted_seconds).abs() <= 1e-9 * report.predicted_seconds.max(1.0),
        "root nodes predict {root_total} s of {} s",
        report.predicted_seconds
    );
    let (loop_id, _) = root
        .nodes()
        .iter()
        .enumerate()
        .find(|(_, node)| matches!(node.kind(), mxx_ir_core::node::NodeKind::ParallelLoop(_)))
        .expect("the graph has a parallel loop");
    let loop_id = NodeId(loop_id as u64);
    let body = plan.graph().source.child_scope_id(&FrozenGraphScopeId::Root, loop_id).unwrap();
    let loop_cost = cost(&FrozenGraphScopeId::Root, loop_id).expect("the loop has a cost");
    let body_self = report
        .node_costs
        .iter()
        .filter(|cost| cost.scope == body)
        .map(|cost| cost.self_seconds)
        .sum::<f64>();
    // A loop's total is its own operations plus every body node's own ones.
    assert!(
        (loop_cost.total_seconds - loop_cost.self_seconds - body_self).abs() <=
            1e-9 * loop_cost.total_seconds.max(1.0)
    );
    assert!(body_self > 0.0, "the body multiplications have a predicted time");
    let html = plan.render_html();
    assert!(html.contains("\"predicted_seconds\":"));
    assert!(!html.contains("/*GRAPH_DATA*/"));
    // Profiling leaves the selected plan executable.
    let result = runtime
        .execute(
            &mut plan,
            BTreeMap::from([(
                "seed".to_owned(),
                RuntimeValue::Bytes(Arc::from(rand::random::<[u8; 32]>())),
            )]),
        )
        .unwrap();
    let cheap = runtime.download_matrix(&result["cheap"]).unwrap();
    assert_eq!(cheap.size(), (1, columns));
    assert!(report.node_costs.iter().all(|cost| cost.self_seconds >= 0.0));
}
