use super::*;

pub trait GraphValue: Clone {
    type Schema: GraphValueSchema<Value = Self>;
    fn flatten(&self) -> Vec<ValueHandle>;

    fn schema(&self) -> Self::Schema;
    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError>;
}

pub trait GraphValueSchema: Clone + PartialEq {
    type Value: GraphValue<Schema = Self>;
    fn placeholders(&self) -> Self::Value {
        self.placeholders_from(&mut 0)
    }
    #[doc(hidden)]
    fn placeholders_from(&self, next: &mut usize) -> Self::Value;
    fn wire_types(&self) -> Vec<WireType>;

    /// Reopen the fields emitted by `DslContext::public_output` or
    /// `private_output`, preserving their exact scalar/family schemas.
    #[track_caller]
    fn artifact_input(
        &self,
        production_id: ProductionId,
        name: impl Into<String>,
        confidentiality: ArtifactConfidentiality,
    ) -> Result<Self::Value, DslError> {
        let name = name.into();
        let types = self.wire_types();
        let arity = types.len();
        let values = types
            .into_iter()
            .enumerate()
            .map(|(port, wire_type)| {
                let name = if arity == 1 { name.clone() } else { format!("{name}.{port}") };
                NodeHandle::new(
                    NodeKind::Input {
                        name: name.clone(),
                        wire_type: wire_type.clone(),
                        artifact: Some(ArtifactInput {
                            production_id: production_id.clone(),
                            artifact_name: name,
                            confidentiality,
                        }),
                    },
                    vec![],
                    vec![wire_type],
                )
                .output(0)
                .expect("schema artifact field")
            })
            .collect::<Vec<_>>();
        Self::Value::from_values(self, &values)
    }
}

pub(super) fn argument_name(next: &mut usize, role: &str) -> String {
    let index = *next;
    *next += 1;
    format!("arg-{index}-{role}")
}

// A statically absent field contributes no graph ports. This lets a record
// retain its preparation phase in its type without inventing a runtime value.
impl GraphValue for () {
    type Schema = ();

    fn flatten(&self) -> Vec<ValueHandle> {
        Vec::new()
    }

    fn schema(&self) -> Self::Schema {}

    fn from_values(_: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        if values.is_empty() { Ok(()) } else { Err(DslError::Schema) }
    }
}

impl GraphValueSchema for () {
    type Value = ();

    fn placeholders_from(&self, _: &mut usize) -> Self::Value {}

    fn wire_types(&self) -> Vec<WireType> {
        Vec::new()
    }
}

impl GraphValue for Mat {
    type Schema = MatType;
    fn flatten(&self) -> Vec<ValueHandle> {
        vec![self.value.clone()]
    }

    fn schema(&self) -> Self::Schema {
        MatType(self.matrix_type.clone())
    }
    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        let [value] = values else { return Err(DslError::Schema) };
        Ok(Mat { value: value.clone(), matrix_type: schema.0.clone() })
    }
}

impl GraphValueSchema for MatType {
    type Value = Mat;
    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        Mat::source_input(argument_name(next, "matrix"), self.0.clone(), None)
    }
    fn wire_types(&self) -> Vec<WireType> {
        vec![WireType::Matrix(self.0.clone())]
    }
}

impl GraphValue for Bytes {
    type Schema = BytesType;

    fn flatten(&self) -> Vec<ValueHandle> {
        vec![self.value.clone()]
    }

    fn schema(&self) -> Self::Schema {
        let WireType::Bytes { length } = self.value.wire_type() else {
            unreachable!("Bytes always wraps a bytes wire")
        };
        BytesType { length: length.clone() }
    }

    fn from_values(_schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        let [value] = values else { return Err(DslError::Schema) };
        Ok(Self { value: value.clone() })
    }
}

impl GraphValue for Int {
    type Schema = IntType;

    fn flatten(&self) -> Vec<ValueHandle> {
        vec![self.value.clone()]
    }

    fn schema(&self) -> Self::Schema {
        IntType
    }

    fn from_values(_schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        let [value] = values else { return Err(DslError::Schema) };
        Ok(Self { value: value.clone() })
    }
}

impl GraphValue for Bool {
    type Schema = BoolType;

    fn flatten(&self) -> Vec<ValueHandle> {
        vec![self.value.clone()]
    }

    fn schema(&self) -> Self::Schema {
        BoolType
    }

    fn from_values(_schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        let [value] = values else { return Err(DslError::Schema) };
        Ok(Self { value: value.clone() })
    }
}

impl GraphValueSchema for BoolType {
    type Value = Bool;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        let node = NodeHandle::new(
            NodeKind::Input {
                name: argument_name(next, "boolean"),
                wire_type: WireType::Bool,
                artifact: None,
            },
            Vec::new(),
            vec![WireType::Bool],
        );
        Bool { value: node.output(0).expect("boolean argument") }
    }

    fn wire_types(&self) -> Vec<WireType> {
        vec![WireType::Bool]
    }
}

impl GraphValueSchema for IntType {
    type Value = Int;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        let node = NodeHandle::new(
            NodeKind::Input {
                name: argument_name(next, "integer"),
                wire_type: WireType::Int,
                artifact: None,
            },
            Vec::new(),
            vec![WireType::Int],
        );
        Int { value: node.output(0).expect("integer argument") }
    }

    fn wire_types(&self) -> Vec<WireType> {
        vec![WireType::Int]
    }
}

impl GraphValueSchema for BytesType {
    type Value = Bytes;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        let wire_type = WireType::Bytes { length: self.length.clone() };
        let node = NodeHandle::new(
            NodeKind::Input {
                name: argument_name(next, "bytes"),
                wire_type: wire_type.clone(),
                artifact: None,
            },
            Vec::new(),
            vec![wire_type],
        );
        Bytes { value: node.output(0).expect("bytes argument") }
    }

    fn wire_types(&self) -> Vec<WireType> {
        vec![WireType::Bytes { length: self.length.clone() }]
    }
}

impl GraphValue for SmallMatrix {
    type Schema = SmallMatrixType;

    fn flatten(&self) -> Vec<ValueHandle> {
        vec![self.value.clone()]
    }

    fn schema(&self) -> Self::Schema {
        SmallMatrixType {
            matrix: self.matrix_type.clone(),
            max_coefficient_bound: self.max_coefficient_bound.clone(),
        }
    }

    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        let [value] = values else { return Err(DslError::Schema) };
        Ok(Self {
            value: value.clone(),
            matrix_type: schema.matrix.clone(),
            max_coefficient_bound: schema.max_coefficient_bound.clone(),
        })
    }
}

impl GraphValueSchema for SmallMatrixType {
    type Value = SmallMatrix;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        SmallMatrix::source_input(
            argument_name(next, "small-matrix"),
            self.matrix.clone(),
            self.max_coefficient_bound.clone(),
            None,
        )
    }

    fn wire_types(&self) -> Vec<WireType> {
        vec![WireType::SmallMatrix {
            matrix: self.matrix.clone(),
            max_coefficient_bound: self.max_coefficient_bound.clone(),
        }]
    }
}

impl GraphValue for Preimage {
    type Schema = PreimageType;

    fn flatten(&self) -> Vec<ValueHandle> {
        vec![self.value.clone()]
    }

    fn schema(&self) -> Self::Schema {
        PreimageType {
            matrix: self.matrix_type.clone(),
            max_coefficient_bound: self.max_coefficient_bound.clone(),
        }
    }

    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        let [value] = values else { return Err(DslError::Schema) };
        Ok(Self {
            value: value.clone(),
            matrix_type: schema.matrix.clone(),
            max_coefficient_bound: schema.max_coefficient_bound.clone(),
        })
    }
}

impl GraphValueSchema for PreimageType {
    type Value = Preimage;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        let wire_type = WireType::Preimage {
            matrix: self.matrix.clone(),
            max_coefficient_bound: self.max_coefficient_bound.clone(),
        };
        let node = NodeHandle::new(
            NodeKind::Input {
                name: argument_name(next, "preimage"),
                wire_type: wire_type.clone(),
                artifact: None,
            },
            Vec::new(),
            vec![wire_type],
        );
        Preimage {
            value: node.output(0).expect("preimage argument"),
            matrix_type: self.matrix.clone(),
            max_coefficient_bound: self.max_coefficient_bound.clone(),
        }
    }

    fn wire_types(&self) -> Vec<WireType> {
        vec![WireType::Preimage {
            matrix: self.matrix.clone(),
            max_coefficient_bound: self.max_coefficient_bound.clone(),
        }]
    }
}

impl GraphValue for Trapdoor {
    type Schema = TrapdoorType;

    fn flatten(&self) -> Vec<ValueHandle> {
        vec![self.public.value.clone(), self.value.clone()]
    }

    fn schema(&self) -> Self::Schema {
        let WireType::Trapdoor {
            sigma,
            gadget_base,
            digit_count,
            preimage_max_coefficient_bound,
            ..
        } = self.value.wire_type()
        else {
            unreachable!("Trapdoor always wraps a trapdoor wire")
        };
        TrapdoorType {
            matrix: self.matrix_type.clone(),
            sigma: sigma.clone(),
            gadget_base: gadget_base.clone(),
            digit_count: digit_count.clone(),
            preimage_max_coefficient_bound: preimage_max_coefficient_bound.clone(),
        }
    }

    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        let [public, value] = values else { return Err(DslError::Schema) };
        Ok(Self {
            public: Mat { value: public.clone(), matrix_type: schema.matrix.clone() },
            value: value.clone(),
            matrix_type: schema.matrix.clone(),
            preimage_max_coefficient_bound: schema.preimage_max_coefficient_bound.clone(),
        })
    }
}

impl GraphValueSchema for TrapdoorType {
    type Value = Trapdoor;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        let public =
            Mat::source_input(argument_name(next, "trapdoor-public"), self.matrix.clone(), None);
        let wire_type = WireType::Trapdoor {
            matrix: self.matrix.clone(),
            sigma: self.sigma.clone(),
            gadget_base: self.gadget_base.clone(),
            digit_count: self.digit_count.clone(),
            preimage_max_coefficient_bound: self.preimage_max_coefficient_bound.clone(),
        };
        let node = NodeHandle::new(
            NodeKind::Input {
                name: argument_name(next, "trapdoor-secret"),
                wire_type: wire_type.clone(),
                artifact: None,
            },
            Vec::new(),
            vec![wire_type],
        );
        Trapdoor {
            public,
            value: node.output(0).expect("trapdoor argument"),
            matrix_type: self.matrix.clone(),
            preimage_max_coefficient_bound: self.preimage_max_coefficient_bound.clone(),
        }
    }

    fn wire_types(&self) -> Vec<WireType> {
        vec![
            WireType::Matrix(self.matrix.clone()),
            WireType::Trapdoor {
                matrix: self.matrix.clone(),
                sigma: self.sigma.clone(),
                gadget_base: self.gadget_base.clone(),
                digit_count: self.digit_count.clone(),
                preimage_max_coefficient_bound: self.preimage_max_coefficient_bound.clone(),
            },
        ]
    }
}

macro_rules! tuple_value {
    ($($value:ident : $field:tt),+ $(,)?) => {
        impl<$($value: GraphValue),+> GraphValue for ($($value,)+) {
            type Schema = ($($value::Schema,)+);
            fn flatten(&self) -> Vec<ValueHandle> {
                let mut values = Vec::new();
                $(values.extend(self.$field.flatten());)+
                values
            }

            fn schema(&self) -> Self::Schema { ($(self.$field.schema(),)+) }
            fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
                let mut offset = 0;
                let result = ($({
                    let count = schema.$field.wire_types().len();
                    let fields = values.get(offset..offset + count).ok_or(DslError::Schema)?;
                    offset += count;
                    $value::from_values(&schema.$field, fields)?
                },)+);
                if offset != values.len() { return Err(DslError::Schema); }
                Ok(result)
            }
        }
        impl<$($value: GraphValueSchema),+> GraphValueSchema for ($($value,)+) {
            type Value = ($($value::Value,)+);
            fn placeholders_from(&self, next: &mut usize) -> Self::Value { ($(self.$field.placeholders_from(next),)+) }
            fn wire_types(&self) -> Vec<WireType> {
                let mut types = Vec::new();
                $(types.extend(self.$field.wire_types());)+
                types
            }
        }
    };
}

tuple_value!(A:0);
tuple_value!(A:0, B:1);
tuple_value!(A:0, B:1, C:2);
tuple_value!(A:0, B:1, C:2, D:3);
tuple_value!(A:0, B:1, C:2, D:3, E:4);
tuple_value!(A:0, B:1, C:2, D:3, E:4, F:5);
tuple_value!(A:0, B:1, C:2, D:3, E:4, F:5, G:6);
tuple_value!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7);
tuple_value!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8);
tuple_value!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9);
tuple_value!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10);
tuple_value!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11);

impl<T: GraphValue> GraphValue for Vec<T> {
    type Schema = Vec<T::Schema>;

    fn flatten(&self) -> Vec<ValueHandle> {
        self.iter().flat_map(GraphValue::flatten).collect()
    }

    fn schema(&self) -> Self::Schema {
        self.iter().map(GraphValue::schema).collect()
    }

    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        let mut offset = 0;
        schema
            .iter()
            .map(|item| {
                let count = item.wire_types().len();
                let result = T::from_values(
                    item,
                    values.get(offset..offset + count).ok_or(DslError::Schema)?,
                )?;
                offset += count;
                Ok(result)
            })
            .collect::<Result<Vec<_>, _>>()
            .and_then(|result| (offset == values.len()).then_some(result).ok_or(DslError::Schema))
    }
}

impl<T: GraphValueSchema> GraphValueSchema for Vec<T> {
    type Value = Vec<T::Value>;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        self.iter().map(|schema| schema.placeholders_from(next)).collect()
    }

    fn wire_types(&self) -> Vec<WireType> {
        self.iter().flat_map(GraphValueSchema::wire_types).collect()
    }
}

// Optional graph fields keep presence in the static schema. No dummy matrix
// or runtime tag is emitted for an absent field inside a composite value.
impl<T: GraphValue> GraphValue for Option<T> {
    type Schema = Option<T::Schema>;

    fn flatten(&self) -> Vec<ValueHandle> {
        self.as_ref().map_or_else(Vec::new, GraphValue::flatten)
    }

    fn schema(&self) -> Self::Schema {
        self.as_ref().map(GraphValue::schema)
    }

    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        match schema {
            Some(schema) => T::from_values(schema, values).map(Some),
            None if values.is_empty() => Ok(None),
            None => Err(DslError::Schema),
        }
    }
}

impl<S: GraphValueSchema> GraphValueSchema for Option<S> {
    type Value = Option<S::Value>;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        self.as_ref().map(|schema| schema.placeholders_from(next))
    }

    fn wire_types(&self) -> Vec<WireType> {
        self.as_ref().map_or_else(Vec::new, GraphValueSchema::wire_types)
    }
}

// Prepared operation maps use their stable keys as schema metadata; their
// graph fields are explicit outputs of the structural body in key order.
impl<K: Clone + Ord, T: GraphValue> GraphValue for std::collections::BTreeMap<K, T> {
    type Schema = std::collections::BTreeMap<K, T::Schema>;

    fn flatten(&self) -> Vec<ValueHandle> {
        self.values().flat_map(GraphValue::flatten).collect()
    }

    fn schema(&self) -> Self::Schema {
        self.iter().map(|(key, value)| (key.clone(), value.schema())).collect()
    }

    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        let mut offset = 0;
        let result = schema
            .iter()
            .map(|(key, schema)| {
                let count = schema.wire_types().len();
                let fields = values.get(offset..offset + count).ok_or(DslError::Schema)?;
                offset += count;
                Ok((key.clone(), T::from_values(schema, fields)?))
            })
            .collect::<Result<Self, DslError>>()?;
        (offset == values.len()).then_some(result).ok_or(DslError::Schema)
    }
}

impl<K: Clone + Ord, S: GraphValueSchema> GraphValueSchema for std::collections::BTreeMap<K, S> {
    type Value = std::collections::BTreeMap<K, S::Value>;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        self.iter().map(|(key, schema)| (key.clone(), schema.placeholders_from(next))).collect()
    }

    fn wire_types(&self) -> Vec<WireType> {
        self.values().flat_map(GraphValueSchema::wire_types).collect()
    }
}
