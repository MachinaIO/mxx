use super::*;

/// Builds one independent body and returns its ordered results.
pub fn parallel<T: GraphValue>(
    count: impl Into<IntExpr>,
    body: impl FnOnce(Int) -> Result<T, DslError>,
) -> Result<Family<T>, DslError> {
    let count = count.into().canonicalize();
    let (index_slot, result) = with_loop_index(|index| {
        with_new_construction_scope(|scope| {
            body(Int::evaluate(IntExpr::LoopIndex(index)))
                .and_then(normalize)
                .map(|output| (scope, output))
        })
    });
    let (scope, output) = result?;
    let schema = output.schema();
    let outputs = output.flatten();
    let shared = outputs
        .iter()
        .map(|value| {
            let owner = value.construction_scope();
            owner != scope && owner.is_ancestor_of(&scope)
        })
        .collect::<Vec<_>>();
    let family_schema = FamilyType { element: schema, count: count.clone(), shared };
    if family_schema.shared.iter().all(|shared| *shared) {
        return Family::from_values(&family_schema, &outputs);
    }
    let sealed = SubgraphHandle::seal(
        "parallel-body",
        scope,
        vec![],
        outputs
            .iter()
            .zip(&family_schema.shared)
            .filter_map(|(value, shared)| (!shared).then(|| value.clone()))
            .collect(),
        &[],
        CapturePolicy::Lexical { parallel_index: Some(index_slot) },
    )?;

    let arguments = sealed.captures.iter().map(|capture| capture.outer.clone()).collect();
    let modes = sealed.captures.iter().map(|capture| capture.mode.clone()).collect();
    let types = family_schema
        .wire_types()
        .into_iter()
        .zip(&family_schema.shared)
        .filter_map(|(ty, shared)| (!shared).then_some(ty))
        .collect::<Vec<_>>();
    let node = NodeHandle::parallel_loop(
        sealed.handle,
        arguments,
        types.clone(),
        ParallelLoop { count, minimum_count: 0, index_slot, bindings: vec![], input_modes: modes },
    );
    let mut port = 0u32;
    let values = outputs
        .into_iter()
        .zip(&family_schema.shared)
        .map(|(value, shared)| {
            if *shared {
                value
            } else {
                let value = node.output(port).expect("parallel output field");
                port += 1;
                value
            }
        })
        .collect::<Vec<_>>();
    Family::from_values(&family_schema, &values)
}

/// Repeatedly computes a new state, returning only the final state.
pub fn iterate<S: GraphValue>(
    count: impl Into<IntExpr>,
    initial: S,
    body: impl FnOnce(Int, S) -> Result<S, DslError>,
) -> Result<S, DslError> {
    let count = count.into().canonicalize();
    let schema = initial.schema();
    let types = schema.wire_types();
    if types.is_empty() {
        return Err(DslError::Schema);
    }
    let (index_slot, result) = with_loop_index(|index| {
        with_new_construction_scope(|scope| {
            let state = schema.placeholders();
            let inputs = state.flatten();
            body(Int::evaluate(IntExpr::LoopIndex(index)), state)
                .and_then(normalize)
                .map(|output| (scope, inputs, output))
        })
    });
    let (scope, inputs, output) = result?;
    if output.schema() != schema {
        return Err(DslError::Schema);
    }
    let sealed = SubgraphHandle::seal(
        "iterate-body",
        scope,
        inputs,
        output.flatten(),
        &[],
        CapturePolicy::Lexical { parallel_index: None },
    )?;

    let mut arguments = initial.flatten();
    arguments.extend(sealed.captures.iter().map(|capture| capture.outer.clone()));
    let node = NodeHandle::sequential_loop(
        sealed.handle,
        arguments,
        types.clone(),
        SequentialLoop { count, index_slot, bindings: vec![], carried_count: types.len() },
    );
    let values = (0..types.len())
        .map(|port| node.output(port as u32).expect("iterated state field"))
        .collect::<Vec<_>>();
    S::from_values(&schema, &values)
}

/// Selects one same-schema value, applying the selector to every field together.
/// This is value selection, not a lazy control-flow branch.
pub fn select<T: GraphValue>(selector: impl Into<Int>, candidates: Vec<T>) -> Result<T, DslError> {
    let selector = selector.into();
    let candidates = candidates.into_iter().map(normalize).collect::<Result<Vec<_>, _>>()?;
    let schema = candidates.first().ok_or(DslError::Schema)?.schema();
    let types = schema.wire_types();
    if types.is_empty() || candidates.iter().any(|value| value.schema() != schema) {
        return Err(DslError::Schema);
    }

    let flattened = candidates.iter().map(GraphValue::flatten).collect::<Vec<_>>();
    let values = types
        .into_iter()
        .enumerate()
        .map(|(port, ty)| {
            let mut arguments = vec![selector.value.clone()];
            arguments.extend(flattened.iter().map(|values| values[port].clone()));
            NodeHandle::new(
                NodeKind::Select { count: IntExpr::constant(candidates.len()) },
                arguments,
                vec![ty],
            )
            .output(0)
            .expect("selected field")
        })
        .collect::<Vec<_>>();
    T::from_values(&schema, &values)
}

pub(super) fn normalize<T: GraphValue>(value: T) -> Result<T, DslError> {
    let schema = value.schema();
    let values = value.flatten();
    if values.len() != schema.wire_types().len() {
        return Err(DslError::Schema);
    }
    let normalized = values
        .into_iter()
        .zip(schema.wire_types())
        .map(|(value, expected)| match (value.wire_type(), &expected) {
            (WireType::ConstantInt, WireType::Int) => Ok(Int { value }.add(Int::constant(0)).value),
            (WireType::ConstantBool, WireType::Bool) => {
                Ok(Bool { value }.to_int().equal(Int::constant(1)).value)
            }
            (actual, expected) if actual == expected => Ok(value),
            _ => Err(DslError::Schema),
        })
        .collect::<Result<Vec<_>, _>>()?;
    T::from_values(&schema, &normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn parallel_cache_keeps_captured_fields_shared() {
        let ring = Ring::new(17, 8);
        let captured = ring.uniform_residue((1, 1));
        let cache = parallel(3, |_| Ok((captured.clone(), ring.uniform_residue((1, 1))))).unwrap();
        assert_eq!(cache.schema().shared, vec![true, false]);
        assert_eq!(cache.at(0).0.flatten(), captured.flatten());
        assert_eq!(cache.at(2).0.flatten(), captured.flatten());
        let shared = cache.field(|value| value.0).unwrap();
        assert_eq!(shared.flatten(), captured.flatten());
        assert_eq!(shared.schema().shared, vec![true]);
        let indexed = cache.field(|value| value.1).unwrap();
        assert_eq!(indexed.schema().shared, vec![false]);
        let schema: FamilyType<(MatType, MatType)> =
            serde_json::from_slice(&serde_json::to_vec(&cache.schema()).unwrap()).unwrap();
        let production = ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([1; 32]),
            execution_nonce: [2; 32],
        };
        let imported =
            schema.artifact_input(production, "cache", ArtifactConfidentiality::Public).unwrap();
        assert_eq!(imported.schema(), cache.schema());
        assert_eq!(imported.at(0).0.flatten(), imported.at(2).0.flatten());
        assert!(matches!(imported.flatten()[0].wire_type(), WireType::Matrix(_)));
        assert!(matches!(imported.flatten()[1].wire_type(), WireType::IndexedFamily { .. }));
        let graph =
            DslContext::new("shared-cache").public_output("cache", cache).unwrap().build().unwrap();
        graph.validate(&ParamEnv::default()).unwrap();
        assert_eq!(
            graph
                .graph
                .root_scope()
                .nodes()
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::UniformResidueSample { .. }))
                .count(),
            1
        );
        let loops = graph
            .graph
            .root_scope()
            .nodes()
            .iter()
            .filter(|node| matches!(node.kind(), NodeKind::ParallelLoop(_)))
            .collect::<Vec<_>>();
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].output_types().len(), 1);
    }

    #[test]
    fn parallel_all_captured_fields_need_no_loop() {
        let ring = Ring::new(17, 8);
        let captured = ring.input("captured", (1, 1));
        let family = parallel(7, |_| Ok(captured.clone())).unwrap();
        assert_eq!(family.count(), &IntExpr::constant(7));
        assert_eq!(family.flatten(), captured.flatten());
        let graph =
            DslContext::new("shared-only").output("shared", family).unwrap().build().unwrap();
        graph.validate(&ParamEnv::default()).unwrap();
        assert!(
            !graph
                .graph
                .root_scope()
                .nodes()
                .iter()
                .any(|node| matches!(node.kind(), NodeKind::ParallelLoop(_)))
        );
        // A leaked value from a sibling scope is not a lexical capture.
        let escaped = RefCell::new(None);
        parallel(1, |_| {
            let value = ring.uniform_residue((1, 1));
            *escaped.borrow_mut() = Some(value.clone());
            Ok(value)
        })
        .unwrap();
        assert!(parallel(1, |_| Ok(escaped.borrow().as_ref().unwrap().clone())).is_err());
    }

    #[test]
    fn escaped_binder_cannot_be_rebound_through_bit_or_hash_metadata() {
        let escaped = RefCell::new(None);
        parallel(1, |i| {
            *escaped.borrow_mut() = Some(i);
            Ok(Int::constant(0))
        })
        .unwrap();
        let escaped = escaped.into_inner().unwrap();
        assert!(parallel(1, |_| Int::constant(1).bit(&escaped)).is_err());
        assert!(parallel(1, |_| Int::constant(1).bit(&escaped + 0)).is_err());
        assert!(
            parallel(1, |_| {
                let mut tag = HashTag::new();
                tag.push_decimal(&escaped + 0)?;
                Ok(Int::constant(0))
            })
            .is_err()
        );
        assert!(
            parallel(1, |_| {
                let mut tag = HashTag::new();
                tag.push_decimal(&escaped)?;
                Ok(Int::constant(0))
            })
            .is_err()
        );
        let ring = Ring::new(17, 8);
        let key = ring.bytes_input("key", 32);
        assert!(
            parallel(1, |_| {
                let mut tag = HashTag::new();
                tag.push(escaped);
                Ok(ring.hash_matrix(key, tag, (1, 1)))
            })
            .is_err()
        );
    }

    #[test]
    fn indexed_reads_keep_member_placement_without_retaining_index_arithmetic() {
        for offset in [0, 1] {
            let ring = Ring::new(17, 8);
            let source = ring.input_family("source", 3, (1, 1));
            let output = parallel(2, |i| Ok((source.at(&i + offset), source.at(i)))).unwrap();
            let built =
                DslContext::new("indexed-reads").output("result", output).unwrap().build().unwrap();
            built.validate(&ParamEnv::default()).unwrap();
            let spec = built
                .graph
                .root_scope()
                .nodes()
                .iter()
                .find_map(|node| match node.kind() {
                    NodeKind::ParallelLoop(spec) => Some(spec),
                    _ => None,
                })
                .unwrap();
            assert!(spec.input_modes.contains(&mxx_ir_core::node::LoopInputMode::Zip));
            if offset != 0 {
                assert!(
                    spec.input_modes
                        .contains(&mxx_ir_core::node::LoopInputMode::ZipOffset { offset })
                );
            }
            assert!(
                !built
                    .graph
                    .scopes()
                    .values()
                    .flat_map(|scope| scope.nodes())
                    .any(|node| matches!(node.kind(), NodeKind::FamilyGetDynamic))
            );
        }
    }

    #[test]
    fn ordinary_integer_offset_keeps_member_placement() {
        let ring = Ring::new(17, 8);
        let source = ring.input_family("source", 4, (1, 1));
        let output = parallel(3, |i| Ok(source.at(i + 1))).unwrap();
        let built =
            DslContext::new("offset-members").output("result", output).unwrap().build().unwrap();
        built.validate(&ParamEnv::default()).unwrap();
        let spec = built
            .graph
            .root_scope()
            .nodes()
            .iter()
            .find_map(|node| match node.kind() {
                NodeKind::ParallelLoop(spec) => Some(spec),
                _ => None,
            })
            .unwrap();
        assert_eq!(
            spec.input_modes,
            vec![mxx_ir_core::node::LoopInputMode::ZipOffset { offset: 1 }]
        );
    }

    #[test]
    fn static_index_folding_preserves_runtime_euclidean_division() {
        let ring = Ring::new(17, 8);
        let source = ring.input_family("source", 3, (1, 1));
        let quotient = Int::constant(7) / 3;
        let expected = mxx_ir_core::expr::euclidean_div_rem(&7.into(), &3.into()).unwrap().0;
        assert_eq!(
            quotient.expression().unwrap().evaluate(&ParamEnv::default()).unwrap(),
            expected
        );
        let built = DslContext::new("quotient-index")
            .output("result", source.at(quotient))
            .unwrap()
            .build()
            .unwrap();
        built.validate(&ParamEnv::default()).unwrap();
        assert!((Int::constant(7) / -3).expression().is_err());
        assert!((Int::constant(7) % -3).expression().is_err());
    }
}
