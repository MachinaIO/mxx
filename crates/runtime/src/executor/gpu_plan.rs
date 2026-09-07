use super::RootBlockAliases;
use crate::gpu_calibration::{
    gpu_calibration_operation_identity, gpu_operation_is_column_separable,
    gpu_operation_is_column_separable_for_types, gpu_row_block_add_operation_identity,
    gpu_sum_rows_operation_identity, gpu_tensor_sum_rows_operation_identity,
};
use mxx_ir_core::{
    ValidatedGraph,
    graph::FrozenGraphScopeId,
    types::{NodeId, Port, WireRef},
};
use std::collections::BTreeMap;

pub(super) type Operations = BTreeMap<NodeId, Result<Option<[u8; 32]>, String>>;

pub(super) fn prepare(validated: &ValidatedGraph, mut plan: RootBlockAliases) -> RootBlockAliases {
    let scope = validated.source.scope(&FrozenGraphScopeId::Root).expect("validated root");
    let checked = validated.root_scope();
    for (position, handle) in checked.execution_order.iter().enumerate() {
        let id = NodeId(position as u64);
        if plan.slices.contains(&id) ||
            plan.concats.contains_key(&id) ||
            plan.add_concats.contains(&id) ||
            plan.row_sum_interiors.contains(&id)
        {
            continue;
        }
        // Store failures instead of returning them here: dispatch must observe
        // an identity error at its original node, after preceding graph effects.
        let operation = (|| {
            let operation = if gpu_operation_is_column_separable(handle.kind()) {
                let arguments = scope.arguments(handle).expect("validated node arguments");
                let argument_types = arguments
                    .iter()
                    .map(|wire| checked.wire_types[wire].clone())
                    .collect::<Vec<_>>();
                let output_types = (0..handle.output_types().len())
                    .map(|port| {
                        checked.wire_types[&WireRef { node: id, port: Port(port as u32) }].clone()
                    })
                    .collect::<Vec<_>>();
                gpu_operation_is_column_separable_for_types(handle.kind(), &argument_types)
                    .then(|| {
                        gpu_calibration_operation_identity(
                            handle.kind(),
                            &argument_types,
                            &output_types,
                            &validated.bindings,
                        )
                    })
                    .transpose()?
            } else {
                None
            };
            // Preserve ordinary-identity errors before applying a fused identity.
            if let Some(row_sum) = plan.row_sums.get(&id) {
                let source = checked.wire_types[&row_sum.source]
                    .matrix_type()
                    .expect("ordinary row sum source");
                let output = checked.wire_types[&WireRef { node: id, port: Port(0) }]
                    .matrix_type()
                    .expect("ordinary row sum output");
                if let Some([left, right]) = row_sum.tensor_operands {
                    return gpu_tensor_sum_rows_operation_identity(
                        checked.wire_types[&left].matrix_type().expect("tensor left"),
                        checked.wire_types[&right].matrix_type().expect("tensor right"),
                        output,
                        &row_sum.rows,
                    )
                    .map(Some);
                }
                return gpu_sum_rows_operation_identity(source, output, &row_sum.rows).map(Some);
            }
            if let (Some(operation), Some((concat, _))) = (operation, plan.adds.get(&id)) {
                let concat = scope.node(*concat).expect("validated fused concat");
                let blocks = scope
                    .arguments(concat)
                    .expect("validated block arguments")
                    .iter()
                    .map(|wire| {
                        checked.wire_types[wire].matrix_type().expect("ordinary row block").clone()
                    })
                    .collect::<Vec<_>>();
                return gpu_row_block_add_operation_identity(operation, &blocks).map(Some);
            }
            Ok(operation)
        })();
        plan.calibration.insert(id, operation);
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::{DslContext, Mat, Ring};
    use mxx_ir_core::{
        ParamEnv,
        node::{ConcatAxis, IndexRange},
    };

    #[test]
    fn root_cached_calibration_matches_direct_fused_identities() {
        let ring = Ring::new(97u64, 8usize);
        let source = ring.input("source", (4, 2));
        let row = |index: usize| {
            source
                .clone()
                .slice(Some(IndexRange { start: index.into(), end: (index + 1).into() }), None)
        };
        let sums = Mat::concat(ConcatAxis::Rows, vec![row(0), row(1) + row(2), row(3)]);
        let blocks = Mat::concat(
            ConcatAxis::Rows,
            vec![ring.input("first", (1, 2)), ring.input("second", (2, 2))],
        );
        let add = blocks + ring.input("right", (3, 2));
        let tensor = ring.input("lhs", (2, 1)).tensor(ring.input("rhs", (2, 1)));
        let tensor_row = |index: usize| {
            tensor
                .clone()
                .slice(Some(IndexRange { start: index.into(), end: (index + 1).into() }), None)
        };
        let tensor_sum = Mat::concat(
            ConcatAxis::Rows,
            vec![tensor_row(0), tensor_row(1) + tensor_row(2), tensor_row(3)],
        );
        let graph = DslContext::new("cached-fused-calibration")
            .output("tensor_sum", tensor_sum)
            .unwrap()
            .output("sums", sums)
            .unwrap()
            .output("add", add)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let plan = super::super::root_block_aliases(&graph, &FrozenGraphScopeId::Root, false);
        let reused = super::super::root_block_aliases(&graph, &FrozenGraphScopeId::Root, false);
        assert!(std::sync::Arc::ptr_eq(&plan, &reused));
        assert_eq!(plan.row_sums.len(), 2);
        assert_eq!(plan.row_sums.values().filter(|sum| sum.tensor_operands.is_some()).count(), 1);
        assert_eq!(plan.adds.len(), 1);
        let scope = graph.source.scope(&FrozenGraphScopeId::Root).unwrap();
        let checked = graph.root_scope();
        for (id, cached) in &plan.calibration {
            let node = &checked.execution_order[id.0 as usize];
            let arguments = scope.arguments(node).unwrap();
            let inputs =
                arguments.iter().map(|wire| checked.wire_types[wire].clone()).collect::<Vec<_>>();
            let outputs = (0..node.output_types().len())
                .map(|port| {
                    checked.wire_types[&WireRef { node: *id, port: Port(port as u32) }].clone()
                })
                .collect::<Vec<_>>();
            let mut expected = if gpu_operation_is_column_separable(node.kind()) &&
                gpu_operation_is_column_separable_for_types(node.kind(), &inputs)
            {
                Some(
                    gpu_calibration_operation_identity(
                        node.kind(),
                        &inputs,
                        &outputs,
                        &graph.bindings,
                    )
                    .unwrap(),
                )
            } else {
                None
            };
            if let Some(sum) = plan.row_sums.get(id) {
                expected = Some(
                    if let Some([left, right]) = sum.tensor_operands {
                        gpu_tensor_sum_rows_operation_identity(
                            checked.wire_types[&left].matrix_type().unwrap(),
                            checked.wire_types[&right].matrix_type().unwrap(),
                            outputs[0].matrix_type().unwrap(),
                            &sum.rows,
                        )
                    } else {
                        gpu_sum_rows_operation_identity(
                            checked.wire_types[&sum.source].matrix_type().unwrap(),
                            outputs[0].matrix_type().unwrap(),
                            &sum.rows,
                        )
                    }
                    .unwrap(),
                );
            } else if let Some((concat, _)) = plan.adds.get(id) {
                let blocks = scope
                    .arguments(scope.node(*concat).unwrap())
                    .unwrap()
                    .iter()
                    .map(|wire| checked.wire_types[wire].matrix_type().unwrap().clone())
                    .collect::<Vec<_>>();
                expected =
                    Some(gpu_row_block_add_operation_identity(expected.unwrap(), &blocks).unwrap());
            }
            assert_eq!(cached, &Ok(expected));
        }
        assert!(
            super::super::root_block_aliases(&graph, &FrozenGraphScopeId::Root, true)
                .calibration
                .is_empty()
        );
    }
}
