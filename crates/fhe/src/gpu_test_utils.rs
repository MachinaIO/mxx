//! Explicit GPU operation widths shared by FHE correctness fixtures.
use mxx_runtime::backend::poly_gpu::GpuDcrtBackend;
use std::collections::BTreeMap;

pub fn configure_widths(backend: &mut GpuDcrtBackend, graph: &mxx_ir_core::ValidatedGraph) {
    use mxx_runtime::{executor::gpu_effective_site_metadata, gpu_calibration::GpuColumnWidths};
    // Correctness fixtures allocate their complete output widths explicitly.
    // Avoid measuring the shared CUDA pool while another test owns a context,
    // without serializing tests or fabricating a measured calibration profile.
    let mut widths = BTreeMap::<[u8; 32], usize>::new();
    for (scope_id, validated) in &graph.scopes {
        // Consume the executor's effective lowering. Rebuilding identities
        // from logical nodes misses fused row-block, compact-RHS, and row-sum
        // sites and lets dynamic execution start a legacy pilot.
        if *scope_id != mxx_ir_core::FrozenGraphScopeId::Root {
            continue;
        }
        for (index, _node) in validated.execution_order.iter().enumerate() {
            let node = mxx_ir_core::types::NodeId(index as u64);
            let Ok((outputs, Some(identity))) = gpu_effective_site_metadata(graph, scope_id, node)
            else {
                continue;
            };
            let width = outputs
                .iter()
                .filter_map(|ty| ty.matrix_type())
                .map(|ty| ty.columns)
                .max()
                .unwrap_or(1)
                .max(1);
            widths.entry(identity).and_modify(|old| *old = (*old).max(width)).or_insert(width);
        }
    }
    for (node, plan) in mxx_runtime::executor::root_row_sum_plans(graph) {
        let validated = graph.root_scope();
        let source = validated.wire_types[&plan.source].matrix_type().unwrap();
        let output = validated.wire_types
            [&mxx_ir_core::types::WireRef { node, port: mxx_ir_core::types::Port(0) }]
            .matrix_type()
            .unwrap();
        let identity = if let Some([left, right]) = plan.tensor_operands {
            mxx_runtime::gpu_calibration::gpu_tensor_sum_rows_operation_identity(
                validated.wire_types[&left].matrix_type().unwrap(),
                validated.wire_types[&right].matrix_type().unwrap(),
                output,
                &plan.rows,
            )
        } else {
            mxx_runtime::gpu_calibration::gpu_sum_rows_operation_identity(
                source, output, &plan.rows,
            )
        }
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
