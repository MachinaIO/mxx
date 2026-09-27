use super::*;

/// Builds one independent body and returns its ordered results.
///
/// The closure runs once, in a new construction scope, with its argument bound to the loop
/// index. Outer values the body reads become explicit loop arguments: `family.at(i)` gives each
/// instance one member (and `family.at(i + c)` a shifted one), `artifacts.at(indices.at(i))` of
/// an outer artifact family by an outer integer family gives each instance the member it names
/// (so a backend can read those members ahead), and every other outer value is shared by all
/// instances. A sampler inside the body draws a fresh value per instance, and a
/// zero-count loop samples nothing.
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
    let family_schema = FamilyType { element: schema, count: count.clone() };
    if family_schema.wire_types().is_empty() {
        return Err(DslError::Schema);
    }
    let sealed = SubgraphHandle::seal(
        "parallel-body",
        scope,
        vec![],
        output.flatten(),
        &[],
        CapturePolicy::Lexical { parallel_index: Some(index_slot) },
    )?;

    let arguments = sealed.captures.iter().map(|capture| capture.outer.clone()).collect();
    let modes = sealed.captures.iter().map(|capture| capture.mode.clone()).collect();
    let types = family_schema.wire_types();
    let node = NodeHandle::parallel_loop(
        sealed.handle,
        arguments,
        types.clone(),
        ParallelLoop { count, minimum_count: 0, index_slot, bindings: vec![], input_modes: modes },
    );
    let values = (0..types.len())
        .map(|port| node.output(port as u32).expect("parallel output field"))
        .collect::<Vec<_>>();
    Family::from_values(&family_schema, &values)
}

/// Repeatedly computes a new state, returning only the final state.
///
/// The state keeps one schema across iterations, and a zero count returns the initial state.
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
///
/// A [`Bool`] selects candidate 0 for false and 1 for true; an [`Int`] selects a zero-based
/// candidate.
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
        let ring = Ring::from_crt_moduli(vec![17.into()], 8);
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
            let ring = Ring::from_crt_moduli(vec![17.into()], 8);
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

    fn parallel_input_modes(built: &crate::BuiltGraph) -> Vec<mxx_ir_core::node::LoopInputMode> {
        built
            .graph
            .root_scope()
            .nodes()
            .iter()
            .find_map(|node| match node.kind() {
                NodeKind::ParallelLoop(spec) => Some(spec.input_modes.clone()),
                _ => None,
            })
            .unwrap()
    }

    #[test]
    fn artifact_read_by_an_outer_index_family_is_gathered() {
        use mxx_ir_core::{
            artifact::{ArtifactAvailability, ProductionId, SpecHash},
            node::LoopInputMode,
        };
        let ring = Ring::from_crt_moduli(vec![17.into()], 8);
        let production = ProductionId { spec_hash: SpecHash([3; 32]), execution_nonce: [4; 32] };
        let members = ring.family_artifact_input(
            production,
            "members",
            5,
            (1, 1),
            ArtifactAvailability::Cached,
        );
        let context = DslContext::new("gathered-members");
        let indices = context.int_family_input("indices", 4);
        for offset in [0, 1] {
            let (members, indices) = (members.clone(), indices.clone());
            let output =
                parallel(3, move |i| Ok(members.at(indices.at(&i + offset)) + members.at(i)))
                    .unwrap();
            let built = DslContext::new("gathered-members")
                .output("result", output)
                .unwrap()
                .build()
                .unwrap();
            let modes = parallel_input_modes(&built);
            let index_mode =
                if offset == 0 { LoopInputMode::Zip } else { LoopInputMode::ZipOffset { offset } };
            let index_argument = modes.iter().position(|mode| *mode == index_mode).unwrap();
            assert!(modes.contains(&LoopInputMode::Gather { index_argument }), "{modes:?}");
            // The member read by the instance index stays a plain Zip.
            assert!(modes.contains(&LoopInputMode::Zip), "{modes:?}");
        }
    }

    #[test]
    fn computed_or_resident_family_reads_are_not_gathered() {
        use mxx_ir_core::{
            artifact::{ArtifactAvailability, ProductionId, SpecHash},
            node::LoopInputMode,
        };
        let ring = Ring::from_crt_moduli(vec![17.into()], 8);
        let production = ProductionId { spec_hash: SpecHash([3; 32]), execution_nonce: [4; 32] };
        let members = ring.family_artifact_input(
            production,
            "members",
            5,
            (1, 1),
            ArtifactAvailability::Cached,
        );
        let resident = ring.input_family("resident", 5, (1, 1));
        let context = DslContext::new("not-gathered");
        let indices = context.int_family_input("indices", 3);
        // An index computed in the body depends on more than the instance.
        let computed = {
            let (members, indices) = (members.clone(), indices.clone());
            parallel(3, move |i| Ok(members.at(indices.at(i) + 1))).unwrap()
        };
        let resident_read = parallel(3, move |i| Ok(resident.at(indices.at(i)))).unwrap();
        for (name, output) in [("computed", computed), ("resident", resident_read)] {
            let built = DslContext::new(name).output("result", output).unwrap().build().unwrap();
            let modes = parallel_input_modes(&built);
            assert!(
                !modes.iter().any(|mode| matches!(mode, LoopInputMode::Gather { .. })),
                "{name}: {modes:?}"
            );
        }
    }

    #[test]
    fn ordinary_integer_offset_keeps_member_placement() {
        let ring = Ring::from_crt_moduli(vec![17.into()], 8);
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
        let ring = Ring::from_crt_moduli(vec![17.into()], 8);
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
