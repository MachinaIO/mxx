//! Explicit GPU operation widths shared by FHE correctness fixtures.
use mxx_runtime::backend::poly_gpu::GpuDcrtBackend;
use std::collections::BTreeMap;

pub fn configure_widths(backend: &mut GpuDcrtBackend, graph: &mxx_ir_core::ValidatedGraph) {
    use mxx_runtime::gpu_calibration::{
        GpuColumnWidths, gpu_calibration_operation_identity,
        gpu_operation_is_column_separable_for_types, gpu_row_block_add_operation_identity,
    };
    // Correctness fixtures allocate their complete output widths explicitly.
    // Avoid measuring the shared CUDA pool while another test owns a context,
    // without serializing tests or fabricating a measured calibration profile.
    let mut widths = BTreeMap::<[u8; 32], usize>::new();
    for (id, validated) in &graph.scopes {
        let scope = graph.source.scope(id).unwrap();
        for node in &validated.execution_order {
            let arguments = scope
                .arguments(node)
                .unwrap()
                .iter()
                .map(|wire| validated.wire_types[wire].clone())
                .collect::<Vec<_>>();
            if !gpu_operation_is_column_separable_for_types(node.kind(), &arguments) {
                continue;
            }
            let outputs = (0..node.output_types().len())
                .map(|port| {
                    validated.wire_types
                        [&scope.wire_ref(&node.output(port as u32).unwrap()).unwrap()]
                        .clone()
                })
                .collect::<Vec<_>>();
            let identity = gpu_calibration_operation_identity(
                node.kind(),
                &arguments,
                &outputs,
                &graph.bindings,
            )
            .unwrap();
            let width = outputs
                .iter()
                .filter_map(|ty| ty.matrix_type())
                .map(|ty| ty.columns)
                .max()
                .unwrap_or(1)
                .max(1);
            widths.entry(identity).and_modify(|old| *old = (*old).max(width)).or_insert(width);
            if *id == mxx_ir_core::FrozenGraphScopeId::Root &&
                matches!(
                    node.kind(),
                    mxx_ir_core::node::NodeKind::MatrixBinary(
                        mxx_ir_core::node::MatrixBinaryOp::Add
                    )
                )
            {
                let args = scope.arguments(node).unwrap();
                let left = args[0];
                let producer = scope.node(left.node).unwrap();
                if matches!(
                    producer.kind(),
                    mxx_ir_core::node::NodeKind::Concat {
                        axis: mxx_ir_core::node::ConcatAxis::Rows
                    }
                ) && !validated.liveness.retained.contains(&left) &&
                    !scope.outputs().contains(&left) &&
                    validated
                        .execution_order
                        .iter()
                        .flat_map(|candidate| scope.arguments(candidate).unwrap())
                        .filter(|argument| *argument == left)
                        .count() ==
                        1 &&
                    arguments.len() == 2 &&
                    arguments[0] == arguments[1] &&
                    outputs.first() == arguments.first()
                {
                    let blocks = scope
                        .arguments(producer)
                        .unwrap()
                        .iter()
                        .map(|wire| match &validated.wire_types[wire] {
                            mxx_ir_core::types::ConcreteWireType::Matrix(ty) => Some(ty.clone()),
                            _ => None,
                        })
                        .collect::<Option<Vec<_>>>();
                    if let Some(blocks) = blocks {
                        let fused =
                            gpu_row_block_add_operation_identity(identity, &blocks).unwrap();
                        // Explicit fixture capacity, never a copied measured profile.
                        widths
                            .entry(fused)
                            .and_modify(|old| *old = (*old).max(width))
                            .or_insert(width);
                    }
                }
            }
        }
    }
    for (node, plan) in mxx_runtime::executor::root_row_sum_plans(graph) {
        let validated = graph.root_scope();
        let source = validated.wire_types[&plan.source].matrix_type().unwrap();
        let output = validated.wire_types
            [&mxx_ir_core::types::WireRef { node, port: mxx_ir_core::types::Port(0) }]
            .matrix_type()
            .unwrap();
        let identity = mxx_runtime::gpu_calibration::gpu_sum_rows_operation_identity(
            source, output, &plan.rows,
        )
        .unwrap();
        let width = output.columns.max(1);
        widths.entry(identity).and_modify(|old| *old = (*old).max(width)).or_insert(width);
    }
    for (identity, width) in widths {
        backend.set_column_widths_for_operation(
            identity,
            GpuColumnWidths { gpu0: width, nonzero: Some(width) },
        );
    }
}
