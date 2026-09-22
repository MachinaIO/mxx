#![cfg(feature = "gpu")]

//! Structural coverage for the typed resident/control boundary.
//!
//! These tests deliberately stop after graph validation.  They exercise the
//! same public lowering vocabulary used by capture preparation without
//! allocating a GPU owner or launching a CUDA graph.  Runtime invalid-index
//! behavior is covered by `gpu_control_resident.rs`; this file keeps the
//! compile-time contract cheap to check on every GPU build.

use mxx_dsl::{DslContext, Family, Int, Ring, iterate, parallel, select};
use mxx_ir_core::{
    ParamEnv,
    graph::FrozenGraphScopeId,
    node::NodeKind,
    types::{ConcreteWireType, NodeId, Port},
};
use mxx_runtime::gpu_column_policy::{
    EffectiveGpuOperation, GpuNodeDisposition, effective_gpu_operation_for_types,
    gpu_node_disposition_for_types,
};
use num_bigint::BigInt;

fn matrix_family_graph() -> mxx_ir_core::ValidatedGraph {
    let context = DslContext::new("typed-matrix-family-structure");
    let ring = Ring::new(17, 8);
    let packed = Family::pack(vec![ring.zero((1, 1)), ring.identity(1)]).expect("matrix family");
    let source = ring.input_family("source", 4, (1, 1));
    let indices = context.int_family_input("indices", 1);
    let static_member = packed.at(1);
    let dynamic_member = source.at(indices.at(0));
    let graph = context
        .output("packed", packed)
        .expect("packed output")
        .output("static", static_member)
        .expect("static output")
        .output("dynamic", dynamic_member)
        .expect("dynamic output")
        .build()
        .expect("build matrix-family graph");
    graph.validate(&ParamEnv::default()).expect("validate matrix-family graph")
}

fn root_nodes(
    graph: &mxx_ir_core::ValidatedGraph,
) -> impl Iterator<Item = (NodeId, &mxx_ir_core::NodeHandle)> {
    graph
        .source
        .scope(&mxx_ir_core::graph::FrozenGraphScopeId::Root)
        .expect("root source scope")
        .nodes()
        .iter()
        .enumerate()
        .map(|(index, node)| (NodeId(index as u64), node))
}

fn concrete_outputs(
    graph: &mxx_ir_core::ValidatedGraph,
    node: NodeId,
    output_count: usize,
) -> Vec<ConcreteWireType> {
    (0..output_count)
        .map(|port| {
            graph.root_scope().wire_types
                [&mxx_ir_core::types::WireRef { node, port: Port(port as u32) }]
                .clone()
        })
        .collect()
}

fn concrete_arguments(
    graph: &mxx_ir_core::ValidatedGraph,
    handle: &mxx_ir_core::NodeHandle,
) -> Vec<ConcreteWireType> {
    graph
        .source
        .scope(&mxx_ir_core::graph::FrozenGraphScopeId::Root)
        .expect("root source scope")
        .arguments(handle)
        .expect("local node arguments")
        .into_iter()
        .map(|wire| graph.root_scope().wire_types[&wire].clone())
        .collect::<Vec<_>>()
}

#[test]
fn matrix_family_pack_static_and_dynamic_selection_keep_typed_owners() {
    let graph = matrix_family_graph();
    let mut saw_pack = false;
    let mut saw_static = false;
    let mut saw_dynamic = false;

    for (node, handle) in root_nodes(&graph) {
        let kind = handle.kind();
        let arguments = concrete_arguments(&graph, handle);
        let outputs = concrete_outputs(&graph, node, handle.output_types().len());
        match kind {
            NodeKind::FamilyPack { .. } => {
                let output = outputs.first().expect("packed output type");
                assert!(matches!(output, ConcreteWireType::IndexedFamily { element, .. }
                    if element.matrix_type().is_some()));
                assert!(arguments.iter().all(|ty| ty.matrix_type().is_some()));
                assert_eq!(
                    gpu_node_disposition_for_types(kind, &arguments, &outputs),
                    GpuNodeDisposition::Resident,
                );
                assert_eq!(
                    effective_gpu_operation_for_types(kind, &arguments, &outputs),
                    EffectiveGpuOperation::HostOrControl,
                );
                saw_pack = true;
            }
            NodeKind::FamilyGetStatic { .. } => {
                if !outputs.first().is_some_and(|ty| ty.matrix_type().is_some()) {
                    continue;
                }
                assert!(
                    matches!(arguments.first(), Some(ConcreteWireType::IndexedFamily { element, .. })
                    if element.matrix_type().is_some())
                );
                assert!(outputs.first().is_some_and(|ty| ty.matrix_type().is_some()));
                assert_eq!(
                    gpu_node_disposition_for_types(kind, &arguments, &outputs),
                    GpuNodeDisposition::NativeAlias,
                );
                saw_static = true;
            }
            NodeKind::FamilyGetDynamic => {
                if !outputs.first().is_some_and(|ty| ty.matrix_type().is_some()) {
                    continue;
                }
                assert!(
                    matches!(arguments.first(), Some(ConcreteWireType::IndexedFamily { element, .. })
                    if element.matrix_type().is_some())
                );
                assert!(matches!(arguments.get(1), Some(ConcreteWireType::Int)));
                assert!(outputs.first().is_some_and(|ty| ty.matrix_type().is_some()));
                // Dynamic matrix selection remains a typed resident operation
                // so the index is consumed by the device gather rather than
                // entering a host-side alias path merely because it is an
                // integer.
                assert_ne!(
                    effective_gpu_operation_for_types(kind, &arguments, &outputs),
                    EffectiveGpuOperation::Unsupported,
                );
                saw_dynamic = true;
            }
            _ => {}
        }
    }

    assert!(saw_pack, "matrix family pack missing from graph");
    assert!(saw_static, "static matrix family selection missing from graph");
    assert!(saw_dynamic, "dynamic matrix family selection missing from graph");
}

#[test]
fn matrix_and_wide_integer_carried_state_has_exact_loop_types_and_four_way_swap() {
    let context = DslContext::new("typed-carried-matrix-wide-integer");
    let ring = Ring::new(17, 8);
    let matrices = Family::pack(
        (0..4)
            .map(|index| ring.polynomial([mxx_ir_core::IntExpr::constant(BigInt::from(index + 1))]))
            .collect(),
    )
    .expect("four distinct matrix keys");
    let wide_keys = Family::<Int>::pack(
        (0..4).map(|index| Int::constant((BigInt::from(1u8) << 127usize) + index)).collect(),
    )
    .expect("four distinct wide integer keys");
    let initial = (ring.zero((1, 1)), Int::constant(BigInt::from(0u8)));
    let carried = iterate(4, initial, |index, (_matrix, _key)| {
        Ok((matrices.at(index.clone()), wide_keys.at(index)))
    })
    .expect("mixed carried state");
    let swapped = parallel(4, |index| {
        let selector = index.clone() % 4;
        select(
            selector,
            vec![
                (matrices.at(0), wide_keys.at(0)),
                (matrices.at(1), wide_keys.at(1)),
                (matrices.at(2), wide_keys.at(2)),
                (matrices.at(3), wide_keys.at(3)),
            ],
        )
    })
    .expect("four-way swap");
    let graph = context
        .output("carried", carried)
        .expect("carried output")
        .output("swapped", swapped)
        .expect("swapped output")
        .build()
        .expect("build carried graph")
        .validate(&ParamEnv::default())
        .expect("validate carried graph");

    let root = graph.root_scope();
    let root_source = graph
        .source
        .scope(&mxx_ir_core::graph::FrozenGraphScopeId::Root)
        .expect("root source scope");
    let sequential = root_source
        .nodes()
        .iter()
        .find(|node| matches!(node.kind(), NodeKind::SequentialLoop(_)))
        .expect("sequential carried loop");
    assert_eq!(sequential.output_types().len(), 2);
    let sequential_id = graph
        .source
        .scope(&mxx_ir_core::graph::FrozenGraphScopeId::Root)
        .expect("root scope")
        .node_id(sequential)
        .expect("sequential node id");
    assert!(matches!(
        &root.wire_types[&mxx_ir_core::types::WireRef { node: sequential_id, port: Port(0) }],
        ConcreteWireType::Matrix(_)
    ));
    assert_eq!(
        root.wire_types[&mxx_ir_core::types::WireRef { node: sequential_id, port: Port(1) }],
        ConcreteWireType::Int,
    );
    let sequential_body = FrozenGraphScopeId::SequentialBody {
        parent: Box::new(FrozenGraphScopeId::Root),
        owner: sequential_id,
    };
    let body = graph.source.scope(&sequential_body).expect("sequential body scope");
    let body_validated = graph.scopes.get(&sequential_body).expect("validated body scope");
    assert_eq!(body.outputs().len(), 2, "carried imports/exports retain both fields");
    assert_eq!(
        body.outputs().iter().map(|wire| &body_validated.wire_types[wire]).collect::<Vec<_>>(),
        vec![
            &ConcreteWireType::Matrix(
                root.wire_types
                    [&mxx_ir_core::types::WireRef { node: sequential_id, port: Port(0) }]
                    .matrix_type()
                    .expect("matrix carried type")
                    .clone(),
            ),
            &ConcreteWireType::Int,
        ],
        "loop body exports must match the exact typed carried state",
    );

    let select_count = graph
        .source
        .scopes()
        .values()
        .flat_map(|scope| scope.nodes())
        .filter(|node| matches!(node.kind(), NodeKind::Select { count } if count == &mxx_ir_core::IntExpr::constant(4)))
        .count();
    assert_eq!(select_count, 2, "matrix and wide-key swap each retain four branches");
    assert!(graph.source.scopes().values().flat_map(|scope| scope.nodes()).any(|node| {
        matches!(node.kind(), NodeKind::Select { count } if count == &mxx_ir_core::IntExpr::constant(4)) &&
            node.arguments().len() == 5
    }));
    assert!(graph.source.scopes().values().flat_map(|scope| scope.nodes()).any(|node| {
        matches!(node.kind(), NodeKind::FamilyGetDynamic) &&
            node.output_types()
                .iter()
                .any(|ty| matches!(ty, mxx_ir_core::types::WireType::Matrix(_)))
    }));
}
