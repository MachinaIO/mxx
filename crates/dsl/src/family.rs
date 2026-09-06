use super::*;

/// An ordered collection of values with one common element schema.
/// Composite elements retain their fields; leaf families are an internal representation.
#[derive(Clone)]
pub struct Family<T: GraphValue> {
    pub(super) values: Vec<ValueHandle>,
    pub(super) element_schema: T::Schema,
    pub(super) count: IntExpr,
    pub(super) pending: Pending,
}

impl<T: GraphValue> Family<T> {
    pub fn count(&self) -> &IntExpr {
        &self.count
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
        if types.is_empty() || elements.iter().any(|element| element.schema() != schema) {
            return Err(DslError::Schema);
        }
        let count = IntExpr::constant(elements.len());
        let pending = Pending::merge(elements.iter().map(GraphValue::pending));
        let flattened = elements.iter().map(GraphValue::flatten).collect::<Vec<_>>();
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
        Ok(Self { values, element_schema: schema, count, pending })
    }

    /// Reads one element. Compile-time and runtime indices have the same surface API.
    #[track_caller]
    pub fn at(&self, index: impl Into<Int>) -> T {
        let index = index.into();
        let expression =
            index.compile_expression().filter(|expression| !integer::has_loop_index(expression));
        let pending = Pending::merge([self.pending.clone(), index.pending.clone()]);
        let values = self
            .values
            .iter()
            .zip(self.element_schema.wire_types())
            .map(|(family, ty)| {
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
        T::from_values(&self.element_schema, &values, pending).expect("family element schema")
    }

    /// Projects existing fields without introducing a loop or changing producer identities.
    /// Arithmetic is expressed with `parallel`; this operation accepts structural projections only.
    pub fn field<U: GraphValue>(
        &self,
        project: impl FnOnce(T) -> U,
    ) -> Result<Family<U>, DslError> {
        with_new_construction_scope(|_| {
            let placeholder = self.element_schema.placeholders();
            let inputs = placeholder.flatten();
            let result = project(placeholder);
            let annotations = result.pending();
            if !annotations.semantic_anchors.is_empty() ||
                !annotations.derivation_attachments.is_empty()
            {
                return Err(DslError::Schema);
            }
            let values = result
                .flatten()
                .iter()
                .map(|value| {
                    let position =
                        inputs.iter().position(|input| input == value).ok_or(DslError::Schema)?;
                    Ok(self.values[position].clone())
                })
                .collect::<Result<Vec<_>, DslError>>()?;
            Ok(Family {
                values,
                element_schema: result.schema(),
                count: self.count.clone(),
                pending: self.pending.clone(),
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
        Self { values, element_schema, count, pending: Pending::default() }
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
            element_schema,
            count,
            pending: public.pending,
        }
    }
}

impl<T: GraphValue> GraphValue for Family<T> {
    type Schema = FamilyType<T::Schema>;
    fn flatten(&self) -> Vec<ValueHandle> {
        self.values.clone()
    }
    fn pending(&self) -> Pending {
        self.pending.clone()
    }
    fn schema(&self) -> Self::Schema {
        FamilyType { element: self.element_schema.clone(), count: self.count.clone() }
    }
    fn from_values(
        schema: &Self::Schema,
        values: &[ValueHandle],
        pending: Pending,
    ) -> Result<Self, DslError> {
        if values.iter().map(|value| value.wire_type().clone()).collect::<Vec<_>>() !=
            schema.wire_types()
        {
            return Err(DslError::Schema);
        }
        Ok(Self {
            values: values.to_vec(),
            element_schema: schema.element.clone(),
            count: schema.count.clone(),
            pending,
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
            pending: Pending::default(),
        }
    }
    fn wire_types(&self) -> Vec<WireType> {
        self.element
            .wire_types()
            .into_iter()
            .map(|element| WireType::IndexedFamily {
                element: Box::new(element),
                count: self.count.clone(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        fn pending(&self) -> Pending {
            self.matrix.pending()
        }
        fn schema(&self) -> Self::Schema {
            TaggedMatrixSchema { matrix: self.matrix.schema(), revealed: self.revealed }
        }
        fn from_values(
            schema: &Self::Schema,
            values: &[ValueHandle],
            pending: Pending,
        ) -> Result<Self, DslError> {
            Ok(Self {
                matrix: Mat::from_values(&schema.matrix, values, pending)?,
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
        let function = Subgraph::define("revealed-only", revealed.schema(), |value| value).unwrap();
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
        let first = values.field(|value| value.0).unwrap();
        let second = values.field(|value| value.1).unwrap();
        assert_eq!(first.flatten()[0], values.flatten()[0]);
        assert_eq!(second.flatten()[0], values.flatten()[1]);
        assert!(matches!(values.field(|value| value.0 + value.1), Err(DslError::Schema)));
        assert!(matches!(
            values.field(|value| value.0.semantic_anchor("lost").unwrap()),
            Err(DslError::Schema)
        ));
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
}
