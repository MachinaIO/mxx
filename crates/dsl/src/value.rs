use super::*;

pub trait GraphValue: Clone {
    type Schema: GraphValueSchema<Value = Self>;
    fn flatten(&self) -> Vec<ValueHandle>;
    fn pending(&self) -> Pending;
    fn schema(&self) -> Self::Schema;
    fn from_values(
        schema: &Self::Schema,
        values: &[ValueHandle],
        pending: Pending,
    ) -> Result<Self, DslError>;
}

pub trait GraphValueSchema: Clone + PartialEq {
    type Value: GraphValue<Schema = Self>;
    fn placeholders(&self) -> Self::Value {
        self.placeholders_from(&mut 0)
    }
    #[doc(hidden)]
    fn placeholders_from(&self, next: &mut usize) -> Self::Value;
    fn wire_types(&self) -> Vec<WireType>;
}

pub(super) fn argument_name(next: &mut usize, role: &str) -> String {
    let index = *next;
    *next += 1;
    format!("arg-{index}-{role}")
}

impl GraphValue for Mat {
    type Schema = MatType;
    fn flatten(&self) -> Vec<ValueHandle> {
        vec![self.value.clone()]
    }
    fn pending(&self) -> Pending {
        self.pending.clone()
    }
    fn schema(&self) -> Self::Schema {
        MatType(self.matrix_type.clone())
    }
    fn from_values(
        schema: &Self::Schema,
        values: &[ValueHandle],
        pending: Pending,
    ) -> Result<Self, DslError> {
        let [value] = values else { return Err(DslError::Schema) };
        Ok(Mat { value: value.clone(), matrix_type: schema.0.clone(), pending })
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

    fn pending(&self) -> Pending {
        self.pending.clone()
    }

    fn schema(&self) -> Self::Schema {
        let WireType::Bytes { length } = self.value.wire_type() else {
            unreachable!("Bytes always wraps a bytes wire")
        };
        BytesType { length: length.clone() }
    }

    fn from_values(
        _schema: &Self::Schema,
        values: &[ValueHandle],
        pending: Pending,
    ) -> Result<Self, DslError> {
        let [value] = values else { return Err(DslError::Schema) };
        Ok(Self { value: value.clone(), pending })
    }
}

impl GraphValue for Int {
    type Schema = IntType;

    fn flatten(&self) -> Vec<ValueHandle> {
        vec![self.value.clone()]
    }

    fn pending(&self) -> Pending {
        self.pending.clone()
    }

    fn schema(&self) -> Self::Schema {
        IntType
    }

    fn from_values(
        _schema: &Self::Schema,
        values: &[ValueHandle],
        pending: Pending,
    ) -> Result<Self, DslError> {
        let [value] = values else { return Err(DslError::Schema) };
        Ok(Self { value: value.clone(), pending })
    }
}

impl GraphValue for Bool {
    type Schema = BoolType;

    fn flatten(&self) -> Vec<ValueHandle> {
        vec![self.value.clone()]
    }

    fn pending(&self) -> Pending {
        self.pending.clone()
    }

    fn schema(&self) -> Self::Schema {
        BoolType
    }

    fn from_values(
        _schema: &Self::Schema,
        values: &[ValueHandle],
        pending: Pending,
    ) -> Result<Self, DslError> {
        let [value] = values else { return Err(DslError::Schema) };
        Ok(Self { value: value.clone(), pending })
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
        Bool { value: node.output(0).expect("boolean argument"), pending: Pending::default() }
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
        Int { value: node.output(0).expect("integer argument"), pending: Pending::default() }
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
        Bytes { value: node.output(0).expect("bytes argument"), pending: Pending::default() }
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

    fn pending(&self) -> Pending {
        self.pending.clone()
    }

    fn schema(&self) -> Self::Schema {
        SmallMatrixType {
            matrix: self.matrix_type.clone(),
            max_coefficient_bound: self.max_coefficient_bound.clone(),
        }
    }

    fn from_values(
        schema: &Self::Schema,
        values: &[ValueHandle],
        pending: Pending,
    ) -> Result<Self, DslError> {
        let [value] = values else { return Err(DslError::Schema) };
        Ok(Self {
            value: value.clone(),
            matrix_type: schema.matrix.clone(),
            max_coefficient_bound: schema.max_coefficient_bound.clone(),
            pending,
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

    fn pending(&self) -> Pending {
        self.pending.clone()
    }

    fn schema(&self) -> Self::Schema {
        PreimageType {
            matrix: self.matrix_type.clone(),
            max_coefficient_bound: self.max_coefficient_bound.clone(),
        }
    }

    fn from_values(
        schema: &Self::Schema,
        values: &[ValueHandle],
        pending: Pending,
    ) -> Result<Self, DslError> {
        let [value] = values else { return Err(DslError::Schema) };
        Ok(Self {
            value: value.clone(),
            matrix_type: schema.matrix.clone(),
            max_coefficient_bound: schema.max_coefficient_bound.clone(),
            pending,
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
            pending: Pending::default(),
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

    fn pending(&self) -> Pending {
        Pending::merge([self.public.pending.clone(), self.pending.clone()])
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

    fn from_values(
        schema: &Self::Schema,
        values: &[ValueHandle],
        pending: Pending,
    ) -> Result<Self, DslError> {
        let [public, value] = values else { return Err(DslError::Schema) };
        Ok(Self {
            public: Mat {
                value: public.clone(),
                matrix_type: schema.matrix.clone(),
                pending: pending.clone(),
            },
            value: value.clone(),
            matrix_type: schema.matrix.clone(),
            preimage_max_coefficient_bound: schema.preimage_max_coefficient_bound.clone(),
            pending,
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
            pending: Pending::default(),
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
            fn pending(&self) -> Pending { Pending::merge([$(self.$field.pending(),)+]) }
            fn schema(&self) -> Self::Schema { ($(self.$field.schema(),)+) }
            fn from_values(schema: &Self::Schema, values: &[ValueHandle], pending: Pending) -> Result<Self, DslError> {
                let mut offset = 0;
                let result = ($({
                    let count = schema.$field.wire_types().len();
                    let fields = values.get(offset..offset + count).ok_or(DslError::Schema)?;
                    offset += count;
                    $value::from_values(&schema.$field, fields, pending.clone())?
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

    fn pending(&self) -> Pending {
        Pending::merge(self.iter().map(GraphValue::pending))
    }

    fn schema(&self) -> Self::Schema {
        self.iter().map(GraphValue::schema).collect()
    }

    fn from_values(
        schema: &Self::Schema,
        values: &[ValueHandle],
        pending: Pending,
    ) -> Result<Self, DslError> {
        let mut offset = 0;
        schema
            .iter()
            .map(|item| {
                let count = item.wire_types().len();
                let result = T::from_values(
                    item,
                    values.get(offset..offset + count).ok_or(DslError::Schema)?,
                    pending.clone(),
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
