//! Exercise exact-division binding guards under both logical loop binders.
use crate::{
    Graph, GraphOutput, IntExpr, NodeHandle, ParamEnv, WireType,
    graph::{SubgraphHandle, with_new_construction_scope},
    lean::{ExportOptions, export},
    node::{NodeKind, ParallelLoop, SequentialLoop},
    validate,
};
use std::collections::BTreeMap;

#[test]
fn export_loop_binding_fixture() {
    let body = with_new_construction_scope(|scope| {
        let value = NodeHandle::new(
            NodeKind::EvaluateInt(IntExpr::Var("k".into())),
            vec![],
            vec![WireType::ConstantInt],
        )
        .output(0)
        .unwrap();
        SubgraphHandle::new("lane", scope, vec![], vec![value]).unwrap()
    });
    let binding =
        vec![("k".into(), IntExpr::Div(Box::new(IntExpr::LoopIndex(0)), Box::new(1.into())))];
    let family = NodeHandle::parallel_loop(
        body,
        vec![],
        vec![WireType::IndexedFamily { element: Box::new(WireType::ConstantInt), count: 3.into() }],
        ParallelLoop {
            count: 3.into(),
            minimum_count: 0,
            index_slot: 0,
            bindings: binding.clone(),
            input_modes: vec![],
        },
    )
    .output(0)
    .unwrap();
    let step = with_new_construction_scope(|scope| {
        let current = NodeHandle::new(
            NodeKind::Input { name: "current".into(), wire_type: WireType::Int, artifact: None },
            vec![],
            vec![WireType::Int],
        )
        .output(0)
        .unwrap();
        let value = NodeHandle::new(
            NodeKind::EvaluateInt(IntExpr::Var("k".into())),
            vec![],
            vec![WireType::ConstantInt],
        )
        .output(0)
        .unwrap();
        let next = NodeHandle::new(
            NodeKind::IntBinary(crate::node::IntBinaryOp::Add),
            vec![current.clone(), value],
            vec![WireType::Int],
        )
        .output(0)
        .unwrap();
        SubgraphHandle::new("step", scope, vec![current], vec![next]).unwrap()
    });
    let initial = NodeHandle::new(
        NodeKind::Input { name: "initial".into(), wire_type: WireType::Int, artifact: None },
        vec![],
        vec![WireType::Int],
    )
    .output(0)
    .unwrap();
    let sequential = NodeHandle::sequential_loop(
        step,
        vec![initial],
        vec![WireType::Int],
        SequentialLoop { count: 3.into(), index_slot: 0, bindings: binding, carried_count: 1 },
    )
    .output(0)
    .unwrap();
    let graph = Graph::freeze(
        "loop-bindings",
        vec![],
        BTreeMap::from([
            ("parallel".into(), GraphOutput { value: family, confidentiality: None }),
            ("sequential".into(), GraphOutput { value: sequential, confidentiality: None }),
        ]),
        vec![],
        vec![],
        BTreeMap::new(),
    )
    .unwrap()
    .0;
    let validated = validate(&graph, &ParamEnv::default()).unwrap();
    let artifact = export(&validated, &ExportOptions::default()).unwrap();
    assert!(artifact.source.contains("∀ i : Fin 3, 1 ≠ 0"));
    assert!(artifact.source.contains("(current next : Int) => 1 ≠ 0"));
    super::write_fixture("loop_binding", artifact.source);
}
