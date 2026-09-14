use super::*;

/// An ordered collection of values with one common element schema.
/// Composite elements retain their fields; leaf families are an internal representation.
#[derive(Clone)]
pub struct Family<T: GraphValue> {
    pub(super) values: Vec<ValueHandle>,
    pub(super) element_schema: T::Schema,
    pub(super) count: IntExpr,
    pub(super) shared: Vec<bool>,
}

impl<T: GraphValue> Family<T> {
    pub fn count(&self) -> &IntExpr {
        &self.count
    }

    /// Select members in the supplied index order with one structural loop.
    /// Repeated indices preserve source sharing; indices obey the caller's
    /// public range contract rather than requiring a runtime validation pass.
    pub fn gather(self, indices: Family<Int>) -> Result<Self, DslError> {
        parallel(indices.count().clone(), |index| Ok(self.at(indices.at(&index))))
    }

    /// Apply one structural loop to aligned family members. The closure is
    /// lowered once, and composite values retain their individual graph ports.
    pub fn zip_map<U: GraphValue, V: GraphValue>(
        self,
        other: Family<U>,
        f: impl Fn(T, U) -> Result<V, DslError>,
    ) -> Result<Family<V>, DslError> {
        if self.count() != other.count() {
            return Err(DslError::Schema);
        }
        parallel(self.count().clone(), |index| f(self.at(&index), other.at(&index)))
    }

    #[doc(hidden)]
    pub fn value_handle(&self) -> &ValueHandle {
        assert_eq!(self.values.len(), 1, "use GraphValue::flatten for composite families");
        &self.values[0]
    }

    pub fn pack(elements: Vec<T>) -> Result<Self, DslError> {
        let elements =
            elements.into_iter().map(control::normalize).collect::<Result<Vec<_>, _>>()?;
        let first = elements.first().ok_or(DslError::Schema)?;
        let schema = first.schema();
        let types = schema.wire_types();
        if elements.iter().any(|element| element.schema() != schema) {
            return Err(DslError::Schema);
        }
        let count = IntExpr::constant(elements.len());

        let flattened = elements.iter().map(GraphValue::flatten).collect::<Vec<_>>();
        // Packing all ordered members of an existing family is a structural
        // identity. Keep its producer instead of retaining one getter per
        // member and then introducing a large pack node for each field.
        let sources = flattened[0]
            .iter()
            .enumerate()
            .map(|(port, first)| {
                let [source] = first.node().arguments() else { return None };
                let WireType::IndexedFamily { count: source_count, .. } = source.wire_type() else {
                    return None;
                };
                if source_count != &count {
                    return None;
                }
                flattened.iter().enumerate().all(|(index, member)| {
                matches!(member[port].node().kind(), NodeKind::FamilyGetStatic { index: actual }
                    if actual == &IntExpr::constant(index)) &&
                    member[port].node().arguments() == std::slice::from_ref(source)
            }).then(|| source.clone())
            })
            .collect::<Option<Vec<_>>>();
        if let Some(values) = sources {
            return Ok(Self {
                shared: vec![false; types.len()],
                values,
                element_schema: schema,
                count,
            });
        }
        let values = types
            .into_iter()
            .enumerate()
            .map(|(port, element)| {
                let node = NodeHandle::new(
                    NodeKind::FamilyPack { count: count.clone() },
                    flattened.iter().map(|values| values[port].clone()).collect(),
                    vec![WireType::IndexedFamily {
                        element: Box::new(element),
                        count: count.clone(),
                    }],
                );
                node.output(0).expect("packed family field")
            })
            .collect();
        Ok(Self {
            values,
            shared: vec![false; schema.wire_types().len()],
            element_schema: schema,
            count,
        })
    }

    /// Reads one element. Compile-time and runtime indices have the same surface API.
    #[track_caller]
    pub fn at(&self, index: impl Into<Int>) -> T {
        let index = index.into();
        let expression =
            index.compile_expression().filter(|expression| !integer::has_loop_index(expression));

        let values = self
            .values
            .iter()
            .zip(self.element_schema.wire_types())
            .zip(&self.shared)
            .map(|((family, ty), shared)| {
                if *shared {
                    return family.clone();
                }
                let (kind, arguments) = match &expression {
                    Some(expression) => (
                        NodeKind::FamilyGetStatic { index: expression.clone() },
                        vec![family.clone()],
                    ),
                    None => (NodeKind::FamilyGetDynamic, vec![family.clone(), index.value.clone()]),
                };
                NodeHandle::new(kind, arguments, vec![ty]).output(0).expect("family element field")
            })
            .collect::<Vec<_>>();
        T::from_values(&self.element_schema, &values).expect("family element schema")
    }

    /// Projects existing fields without introducing a loop or changing producer identities.
    /// Arithmetic is expressed with `parallel`; this operation accepts structural projections only.
    pub fn field<U: GraphValue>(
        &self,
        project: impl FnOnce(T) -> U,
    ) -> Result<Family<U>, DslError> {
        with_new_construction_scope(|_| {
            let placeholder = self.element_schema.placeholders();
            // Wide cryptographic records can have thousands of ports. Index
            // source handles once instead of rescanning them for every field.
            let mut inputs = std::collections::HashMap::new();
            for (position, input) in placeholder.flatten().into_iter().enumerate() {
                inputs.entry(input).or_insert(position);
            }
            let result = project(placeholder);
            let positions = result
                .flatten()
                .iter()
                .map(|value| inputs.get(value).copied().ok_or(DslError::Schema))
                .collect::<Result<Vec<_>, DslError>>()?;
            Ok(Family {
                values: positions.iter().map(|position| self.values[*position].clone()).collect(),
                shared: positions.iter().map(|position| self.shared[*position]).collect(),
                element_schema: result.schema(),
                count: self.count.clone(),
            })
        })
    }

    pub(super) fn source_input(
        name: String,
        element: T,
        count: IntExpr,
        artifact: Option<ArtifactInput>,
    ) -> Self {
        let element_schema = element.schema();
        let types = element_schema.wire_types();
        let arity = types.len();
        let values = types
            .into_iter()
            .enumerate()
            .map(|(port, element)| {
                let name = if arity == 1 { name.clone() } else { format!("{name}.{port}") };
                let artifact = artifact.clone().map(|mut artifact| {
                    if arity != 1 {
                        artifact.artifact_name = format!("{}.{port}", artifact.artifact_name);
                    }
                    artifact
                });
                let wire_type =
                    WireType::IndexedFamily { element: Box::new(element), count: count.clone() };
                NodeHandle::new(
                    NodeKind::Input { name, wire_type: wire_type.clone(), artifact },
                    vec![],
                    vec![wire_type],
                )
                .output(0)
                .expect("family input field")
            })
            .collect();
        Self { values, shared: vec![false; arity], element_schema, count }
    }
}

impl Family<Mat> {
    pub fn element_type(&self) -> &MatrixType {
        &self.element_schema.0
    }
}

impl Family<Preimage> {
    pub fn element_type(&self) -> &MatrixType {
        &self.element_schema.matrix
    }
    pub fn max_coefficient_bound(&self) -> &IntExpr {
        &self.element_schema.max_coefficient_bound
    }
}

impl Family<Trapdoor> {
    #[doc(hidden)]
    pub fn secret_value_handle(&self) -> &ValueHandle {
        &self.values[1]
    }

    pub fn public_matrices(&self) -> Family<Mat> {
        self.field(|trapdoor| trapdoor.public_matrix()).expect("trapdoor public field")
    }

    pub(super) fn trapdoor_input(
        name: String,
        public: Family<Mat>,
        element_schema: TrapdoorType,
        count: IntExpr,
        artifact: Option<ArtifactInput>,
    ) -> Self {
        let wire_type = WireType::IndexedFamily {
            element: Box::new(element_schema.wire_types()[1].clone()),
            count: count.clone(),
        };
        let secret = NodeHandle::new(
            NodeKind::Input { name, wire_type: wire_type.clone(), artifact },
            vec![],
            vec![wire_type],
        )
        .output(0)
        .expect("trapdoor family secret input");
        Self {
            values: vec![public.values[0].clone(), secret],
            shared: vec![public.shared[0], false],
            element_schema,
            count,
        }
    }
}

impl<T: GraphValue> GraphValue for Family<T> {
    type Schema = FamilyType<T::Schema>;
    fn flatten(&self) -> Vec<ValueHandle> {
        self.values.clone()
    }

    fn schema(&self) -> Self::Schema {
        FamilyType {
            element: self.element_schema.clone(),
            count: self.count.clone(),
            shared: self.shared.clone(),
        }
    }
    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        if values.iter().map(|value| value.wire_type().clone()).collect::<Vec<_>>() !=
            schema.wire_types()
        {
            return Err(DslError::Schema);
        }
        Ok(Self {
            values: values.to_vec(),
            element_schema: schema.element.clone(),
            count: schema.count.clone(),
            shared: schema.shared.clone(),
        })
    }
}

impl<S: GraphValueSchema> GraphValueSchema for FamilyType<S> {
    type Value = Family<S::Value>;
    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        let values = self
            .wire_types()
            .into_iter()
            .map(|wire_type| {
                NodeHandle::new(
                    NodeKind::Input {
                        name: argument_name(next, "family"),
                        wire_type: wire_type.clone(),
                        artifact: None,
                    },
                    vec![],
                    vec![wire_type],
                )
                .output(0)
                .expect("family argument")
            })
            .collect();
        Family {
            values,
            element_schema: self.element.clone(),
            count: self.count.clone(),
            shared: self.shared.clone(),
        }
    }
    fn wire_types(&self) -> Vec<WireType> {
        assert_eq!(self.shared.len(), self.element.wire_types().len(), "family field schema");
        self.element
            .wire_types()
            .into_iter()
            .zip(&self.shared)
            .map(|(element, shared)| {
                if *shared {
                    element
                } else {
                    WireType::IndexedFamily {
                        element: Box::new(element),
                        count: self.count.clone(),
                    }
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
    struct StaticTag(u64);

    impl GraphValue for StaticTag {
        type Schema = Self;
        fn flatten(&self) -> Vec<ValueHandle> {
            Vec::new()
        }
        fn schema(&self) -> Self {
            self.clone()
        }
        fn from_values(schema: &Self, values: &[ValueHandle]) -> Result<Self, DslError> {
            if !values.is_empty() {
                return Err(DslError::Schema);
            }
            Ok(schema.clone())
        }
    }

    impl GraphValueSchema for StaticTag {
        type Value = Self;
        fn placeholders_from(&self, _: &mut usize) -> Self {
            self.clone()
        }
        fn wire_types(&self) -> Vec<WireType> {
            Vec::new()
        }
    }

    #[test]
    fn test_zero_port_families_preserve_metadata_and_count() {
        let tag = StaticTag(17);
        let packed = Family::pack(vec![tag.clone(); 3]).unwrap();
        let repeated = parallel(3, |_| Ok(tag.clone())).unwrap();
        assert_eq!(packed.count(), &IntExpr::constant(3));
        assert_eq!(packed.schema(), repeated.schema());
        assert!(packed.flatten().is_empty());
        assert!(repeated.flatten().is_empty());
        assert_eq!(packed.at(2), tag);
        assert_eq!(repeated.at(2), tag);
        let schema: FamilyType<StaticTag> =
            serde_json::from_slice(&serde_json::to_vec(&packed.schema()).unwrap()).unwrap();
        let restored = Family::<StaticTag>::from_values(&schema, &[]).unwrap();
        assert_eq!(restored.count(), packed.count());
        assert_eq!(restored.at(1), tag);
        assert!(matches!(Family::pack(vec![tag, StaticTag(18)]), Err(DslError::Schema)));
    }

    #[derive(Clone)]
    struct TaggedMatrix {
        matrix: Mat,
        revealed: bool,
    }

    #[derive(Clone, PartialEq)]
    struct TaggedMatrixSchema {
        matrix: MatType,
        revealed: bool,
    }

    impl GraphValue for TaggedMatrix {
        type Schema = TaggedMatrixSchema;
        fn flatten(&self) -> Vec<ValueHandle> {
            self.matrix.flatten()
        }

        fn schema(&self) -> Self::Schema {
            TaggedMatrixSchema { matrix: self.matrix.schema(), revealed: self.revealed }
        }
        fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
            Ok(Self {
                matrix: Mat::from_values(&schema.matrix, values)?,
                revealed: schema.revealed,
            })
        }
    }

    impl GraphValueSchema for TaggedMatrixSchema {
        type Value = TaggedMatrix;
        fn placeholders_from(&self, next: &mut usize) -> Self::Value {
            TaggedMatrix { matrix: self.matrix.placeholders_from(next), revealed: self.revealed }
        }
        fn wire_types(&self) -> Vec<WireType> {
            self.matrix.wire_types()
        }
    }

    #[test]
    fn composite_values_preserve_static_metadata_in_pack_select_and_state() {
        let ring = Ring::new(17, 8);
        let revealed = TaggedMatrix { matrix: ring.input("a", (1, 1)), revealed: true };
        let hidden = TaggedMatrix { matrix: ring.input("b", (1, 1)), revealed: false };
        assert!(matches!(
            Family::pack(vec![revealed.clone(), hidden.clone()]),
            Err(DslError::Schema)
        ));
        assert!(matches!(select(0, vec![revealed.clone(), hidden.clone()]), Err(DslError::Schema)));
        let function =
            Subgraph::define("revealed-only", revealed.schema(), |value| Ok(value)).unwrap();
        assert!(matches!(function.call(hidden.clone()), Err(DslError::Schema)));
        assert!(matches!(
            function.call_with_canonical_input_exclusive_uppers(hidden.clone(), vec![None]),
            Err(DslError::Schema)
        ));
        assert!(matches!(iterate(1, revealed.clone(), |_, _| Ok(hidden)), Err(DslError::Schema)));
        let values = parallel(2, |_| Ok(revealed)).unwrap();
        assert!(values.at(0).revealed);
    }

    #[test]
    fn field_projection_preserves_producer_and_rejects_computation() {
        let ring = Ring::new(17, 8);
        let values =
            parallel(3, |_| Ok((ring.gaussian((1, 1), 1, 4), ring.uniform_residue((1, 1)))))
                .unwrap();
        let repacked = Family::pack((0..3).map(|index| values.at(index)).collect()).unwrap();
        assert_eq!(repacked.flatten(), values.flatten());
        let reversed = Family::pack((0..3).rev().map(|index| values.at(index)).collect()).unwrap();
        assert_ne!(reversed.flatten(), values.flatten());
        let subset = Family::pack(vec![values.at(0), values.at(1)]).unwrap();
        assert_eq!(subset.count(), &IntExpr::constant(2));
        let first = values.field(|value| value.0).unwrap();
        let second = values.field(|value| value.1).unwrap();
        assert_eq!(first.flatten()[0], values.flatten()[0]);
        assert_eq!(second.flatten()[0], values.flatten()[1]);
        let reordered = values.field(|(a, b)| (b, a.clone(), a)).unwrap();
        assert_eq!(
            reordered.flatten(),
            vec![
                values.flatten()[1].clone(),
                values.flatten()[0].clone(),
                values.flatten()[0].clone()
            ]
        );
        assert!(matches!(values.field(|value| value.0 + value.1), Err(DslError::Schema)));
        let built = DslContext::new("field-projection")
            .output("a", first)
            .unwrap()
            .output("b", second)
            .unwrap()
            .build()
            .unwrap();
        built.validate(&ParamEnv::default()).unwrap();
        assert_eq!(
            built
                .graph
                .root_scope()
                .nodes()
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::ParallelLoop(_)))
                .count(),
            1
        );
    }

    #[test]
    fn composite_input_index_and_output_use_one_shared_schema() {
        let ring = Ring::new(17, 8);
        let context = DslContext::new("record-input");
        let schema = FamilyType {
            element: (MatType(ring.matrix_type((1, 1))), IntType, BoolType),
            count: 4.into(),
            shared: vec![false; 3],
        };
        let inputs: Family<(Mat, Int, Bool)> = context.input("records", schema).unwrap();
        let output = parallel(4, |i| Ok(inputs.at(i))).unwrap();
        let built = context.output("records", output).unwrap().build().unwrap();
        built.validate(&ParamEnv::default()).unwrap();
        assert_eq!(
            built.graph.outputs().keys().map(String::as_str).collect::<Vec<_>>(),
            vec!["records.0", "records.1", "records.2"]
        );
        let node = built
            .graph
            .root_scope()
            .nodes()
            .iter()
            .find(|node| matches!(node.kind(), NodeKind::ParallelLoop(_)))
            .unwrap();
        let NodeKind::ParallelLoop(spec) = node.kind() else { unreachable!() };
        assert_eq!(spec.input_modes, vec![mxx_ir_core::node::LoopInputMode::Zip; 3]);
    }

    #[test]
    fn test_prepared_map_leaves_parallel_as_explicit_family_outputs() {
        use std::collections::BTreeMap;

        let ring = Ring::new(17, 8);
        let shared = ring.input("shared", (1, 1));
        let inputs = ring.input_family("members", 4, (1, 1));
        let prepared = parallel(3, |i| {
            let member = inputs.at(i + 1);
            let cache = BTreeMap::from([
                (2usize, (member.clone() + shared.clone(), None::<Mat>)),
                (5usize, (member.clone(), Some(shared.clone()))),
            ]);
            Ok((member, cache, ()))
        })
        .unwrap();
        let schema = prepared.schema();
        // The static preparation phase contributes no runtime artifact port.
        assert_eq!(schema.wire_types().len(), 4);
        let caches = prepared.field(|value| value.1).unwrap();
        let restored: Family<(Mat, BTreeMap<usize, (Mat, Option<Mat>)>, ())> =
            Family::from_values(&schema, &prepared.flatten()).unwrap();
        let consumed = parallel(3, |i| {
            let cache = caches.at(&i);
            let (member, restored_cache, ()) = restored.at(i);
            assert!(cache[&2].1.is_none());
            Ok(cache[&2].0.clone() + restored_cache[&5].1.clone().unwrap() + member)
        })
        .unwrap();
        let built = DslContext::new("prepared-family-cache")
            .output("result", consumed)
            .unwrap()
            .build()
            .unwrap();
        built.validate(&ParamEnv::default()).unwrap();
        assert_eq!(
            built
                .graph
                .root_scope()
                .nodes()
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::ParallelLoop(_)))
                .count(),
            2
        );
    }
}
