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
    // The element schema depends on the member index only if a body type names the loop index.
    let depends = schema
        .wire_types()
        .iter()
        .any(|ty| ty.substitute_loop_index(index_slot, &IntExpr::constant(0)) != *ty);
    let family_schema = FamilyType {
        element: schema,
        count: count.clone(),
        member_slot: depends.then_some(index_slot),
    };
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
    // Outside the body, a type depending on the loop index depends on the member index.
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
///
/// On the GPU, matrix and small-matrix family carries are streamed through private artifact
/// storage with two alternating generations. Read them only member-wise (parallel Zip/Gather
/// lanes or static/dynamic family selection), and return each family from a top-level parallel
/// loop in the body or unchanged. Such a carry cannot be a root graph output or be consumed as
/// a whole family; unsupported uses are GPU planning errors. Integer families stay resident.
/// The CPU releases obsolete staged generations after every iteration, except when another
/// live value, an invariant, or trace capture still references them.
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
    fn parallel_hash_captures_keep_tags_but_not_resident_matrices() {
        let ring = Ring::from_crt_moduli(vec![17.into()], 8);
        let key = ring.bytes_input("key", 32);
        let sample = ring.hash_matrix(key.clone(), b"public-key".as_slice(), (1, 4));
        let other = ring.hash_matrix(key, b"another-key".as_slice(), (1, 4));
        let fresh = ring.uniform_residue((1, 4));
        let outputs =
            parallel(3, |_| Ok((sample.clone(), sample.clone(), fresh.clone(), other.clone())))
                .unwrap();
        let built =
            DslContext::new("hash-captures").output("result", outputs).unwrap().build().unwrap();
        built.validate(&ParamEnv::default()).unwrap();
        let root = built.graph.root_scope();
        assert!(!root.nodes().iter().any(|n| matches!(n.kind(), NodeKind::HashSample { .. })));
        assert_eq!(
            root.nodes()
                .iter()
                .filter(|n| matches!(n.kind(), NodeKind::UniformResidueSample { .. }))
                .count(),
            1
        );
        let body = built
            .graph
            .scopes()
            .values()
            .find(|scope| {
                matches!(scope.id(), mxx_ir_core::graph::FrozenGraphScopeId::ParallelBody { .. })
            })
            .unwrap();
        let hashes = body
            .nodes()
            .iter()
            .filter(|n| matches!(n.kind(), NodeKind::HashSample { .. }))
            .collect::<Vec<_>>();
        let tags = hashes
            .iter()
            .map(|hash| {
                let NodeKind::HashSample { tag_prefix, .. } = hash.kind() else { unreachable!() };
                tag_prefix.as_slice()
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(hashes.len(), 2, "distinct hashes remain distinct; repeated reads share one");
        assert_eq!(
            tags,
            [b"public-key".as_slice(), b"another-key".as_slice()].into_iter().collect()
        );
        assert_eq!(body.outputs()[0], body.outputs()[1]);
    }

    /// Initial state stays outside the loop; immutable hash captures are generated
    /// inside it, while fresh samples retain their single outer sampling site.
    #[test]
    fn sequential_hash_captures_preserve_initial_state_and_fresh_sampling() {
        let ring = Ring::from_crt_moduli(vec![17.into()], 8);
        let key = ring.bytes_input("key", 32);
        let sample = ring.hash_matrix(key.clone(), b"public-key".as_slice(), (1, 4));
        let other = ring.hash_matrix(key, b"another-key".as_slice(), (1, 4));
        let fresh = ring.uniform_residue((1, 4));
        let initial = (sample.clone(), sample.clone(), fresh.clone(), other.clone());
        let outputs = iterate(2, initial, |_, _| {
            Ok((sample.clone(), sample.clone(), fresh.clone(), other.clone()))
        })
        .unwrap();
        let built = DslContext::new("sequential-hash-captures")
            .output("result", outputs)
            .unwrap()
            .build()
            .unwrap();
        built.validate(&ParamEnv::default()).unwrap();
        let root = built.graph.root_scope();
        assert_eq!(
            root.nodes().iter().filter(|n| matches!(n.kind(), NodeKind::HashSample { .. })).count(),
            2
        );
        assert_eq!(
            root.nodes()
                .iter()
                .filter(|n| matches!(n.kind(), NodeKind::UniformResidueSample { .. }))
                .count(),
            1
        );
        let body = built
            .graph
            .scopes()
            .values()
            .find(|scope| {
                matches!(scope.id(), mxx_ir_core::graph::FrozenGraphScopeId::SequentialBody { .. })
            })
            .unwrap();
        let tags = body
            .nodes()
            .iter()
            .filter_map(|node| match node.kind() {
                NodeKind::HashSample { tag_prefix, .. } => Some(tag_prefix.as_slice()),
                _ => None,
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            tags,
            [b"public-key".as_slice(), b"another-key".as_slice()].into_iter().collect()
        );
        assert!(
            !body.nodes().iter().any(|n| matches!(n.kind(), NodeKind::UniformResidueSample { .. }))
        );
        assert_eq!(body.outputs()[0], body.outputs()[1]);
        let loop_node =
            root.nodes().iter().find(|n| matches!(n.kind(), NodeKind::SequentialLoop(_))).unwrap();
        assert_eq!(
            root.arguments(loop_node).unwrap().len(),
            6,
            "four initial carries, one hash key and one fresh matrix capture"
        );
    }

    #[test]
    fn parallel_distinct_hash_captures_preserve_every_domain() {
        let ring = Ring::from_crt_moduli(vec![17.into()], 8);
        let key = ring.bytes_input("key", 32);
        let samples = (0u64..32)
            .map(|tag| ring.hash_matrix(key.clone(), tag.to_le_bytes().to_vec(), (1, 4)))
            .collect::<Vec<_>>();
        let outputs = parallel(3, |_| Family::pack(samples.clone())).unwrap();
        let built = DslContext::new("distinct-hash-domains")
            .output("result", outputs)
            .unwrap()
            .build()
            .unwrap();
        built.validate(&ParamEnv::default()).unwrap();
        let tags = built
            .graph
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .filter_map(|node| match node.kind() {
                NodeKind::HashSample { tag_prefix, .. } => Some(tag_prefix.clone()),
                _ => None,
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(tags, (0u64..32).map(|tag| tag.to_le_bytes().to_vec()).collect());
    }

    #[test]
    fn parallel_hash_capture_does_not_resample_a_parameter_sized_packed_matrix() {
        let ring = Ring::from_crt_moduli(vec![17.into()], 8);
        let sample = ring.hash_matrix(
            ring.bytes_input("key", 32),
            b"packed".as_slice(),
            (IntExpr::constant(1), IntExpr::Var("width".to_owned())),
        );
        let outputs = parallel(3, |_| Ok(sample.clone())).unwrap();
        let built = DslContext::new("packed-hash-capture")
            .int_parameter("width")
            .output("result", outputs)
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(
            built
                .graph
                .root_scope()
                .nodes()
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::HashSample { .. }))
                .count(),
            1
        );
        for scope in built
            .graph
            .scopes()
            .values()
            .filter(|scope| !matches!(scope.id(), mxx_ir_core::graph::FrozenGraphScopeId::Root))
        {
            assert!(
                !scope
                    .nodes()
                    .iter()
                    .any(|node| matches!(node.kind(), NodeKind::HashSample { .. }))
            );
        }
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

    fn multiplications(built: &crate::BuiltGraph) -> u128 {
        built.operation_counts().unwrap().get("MatrixBinary(Multiply)").copied().unwrap_or(0)
    }

    #[test]
    fn index_invariant_loops_are_counted_without_enumeration() {
        let ring = Ring::from_crt_moduli(vec![17.into()], 8);
        let (a, b) = (ring.input("a", (1, 1)), ring.input("b", (1, 1)));
        let output = iterate(1_000_000_000_000u64, a.clone(), |_, state| {
            Ok(parallel(1_000_000_000_000u64, |_| Ok(&state * &b))?.at(0))
        })
        .unwrap();
        let built = DslContext::new("invariant").output("result", output).unwrap().build().unwrap();
        assert_eq!(multiplications(&built), 1_000_000_000_000u128 * 1_000_000_000_000);
    }

    #[test]
    fn empty_loops_count_nothing() {
        let ring = Ring::from_crt_moduli(vec![17.into()], 8);
        let (a, b) = (ring.input("a", (1, 1)), ring.input("b", (1, 1)));
        let output = iterate(0, a, |_, state| Ok(&state * &b)).unwrap();
        let built = DslContext::new("empty").output("result", output).unwrap().build().unwrap();
        assert_eq!(multiplications(&built), 0);
    }

    /// A named input declared inside nested loop bodies is a root input of the graph, read by
    /// the bodies like any outer value.
    #[test]
    fn inputs_declared_in_bodies_are_graph_inputs() {
        let ring = Ring::from_crt_moduli(vec![17.into()], 8);
        let family = parallel(3, |i| {
            iterate(2, ring.zero((1, 1)), |_, sum| {
                Ok(sum + ring.input("declared-in-body", (1, 1)).multiply_monomial(i.clone()))
            })
        })
        .unwrap();
        let built = DslContext::new("body-inputs").output("out", family).unwrap().build().unwrap();
        built.validate(&ParamEnv::default()).unwrap();
        let root_inputs = built
            .graph
            .root_scope()
            .nodes()
            .iter()
            .filter(|node| matches!(node.kind(), NodeKind::Input { name, .. } if name == "declared-in-body"))
            .count();
        assert_eq!(root_inputs, 1);
    }

    /// Leaves move the outputs of a context through a loop and back into a graph.
    #[test]
    fn context_outputs_move_as_leaves() {
        let ring = Ring::from_crt_moduli(vec![17.into()], 8);
        let family = parallel(2, |i| {
            let outputs = DslContext::new("inner")
                .output("value", ring.identity(1).multiply_monomial(i))?
                .into_outputs();
            Ok(outputs.into_iter().map(|(_, leaf, _)| leaf).collect::<Vec<_>>())
        })
        .unwrap();
        let leaf = Leaf::of(family.field(|leaves| leaves[0].clone()).unwrap()).unwrap();
        assert!(matches!(leaf.wire_type(), WireType::IndexedFamily { .. }));
        assert!(leaf.at(Int::constant(1)).is_some());
        DslContext::new("leaves").output("out", leaf).unwrap().build().unwrap();
    }
}
