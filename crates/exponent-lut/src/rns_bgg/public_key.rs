//! RNS-BGG public preparation and prepared artifact transport.
use super::*;

impl<M: GraphValue> GraphValue for RnsBggPublicKeyFamily<M> {
    type Schema = RnsBggPublicKeyFamily<M::Schema>;

    fn flatten(&self) -> Vec<ValueHandle> {
        self.matrices.flatten()
    }

    fn schema(&self) -> Self::Schema {
        RnsBggPublicKeyFamily {
            matrices: self.matrices.schema(),
            level: self.level,
            id: self.id,
            context: self.context,
            range: self.range.clone(),
        }
    }

    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        Ok(Self {
            matrices: M::from_values(&schema.matrices, values)?,
            level: schema.level,
            id: schema.id,
            context: schema.context,
            range: schema.range.clone(),
        })
    }
}

impl<M: GraphValueSchema> GraphValueSchema for RnsBggPublicKeyFamily<M> {
    type Value = RnsBggPublicKeyFamily<M::Value>;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        RnsBggPublicKeyFamily {
            matrices: self.matrices.placeholders_from(next),
            level: self.level,
            id: self.id,
            context: self.context,
            range: self.range.clone(),
        }
    }

    fn wire_types(&self) -> Vec<WireType> {
        self.matrices.wire_types()
    }
}

impl<M: GraphValue> GraphValue for RnsBggPublicKey<M> {
    type Schema = RnsBggPublicKey<M::Schema>;

    fn flatten(&self) -> Vec<ValueHandle> {
        self.matrix.flatten()
    }

    fn schema(&self) -> Self::Schema {
        RnsBggPublicKey {
            matrix: self.matrix.schema(),
            level: self.level,
            id: self.id,
            context: self.context,
        }
    }

    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        Ok(Self {
            matrix: M::from_values(&schema.matrix, values)?,
            level: schema.level,
            id: schema.id,
            context: schema.context,
        })
    }
}

impl<M: GraphValueSchema> GraphValueSchema for RnsBggPublicKey<M> {
    type Value = RnsBggPublicKey<M::Value>;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        RnsBggPublicKey {
            matrix: self.matrix.placeholders_from(next),
            level: self.level,
            id: self.id,
            context: self.context,
        }
    }

    fn wire_types(&self) -> Vec<WireType> {
        self.matrix.wire_types()
    }
}

impl<C: GraphValue, L: GraphValue> GraphValue for RnsBggPrivateRhs<C, L> {
    type Schema = RnsBggPrivateRhs<C::Schema, L::Schema>;
    fn flatten(&self) -> Vec<ValueHandle> {
        [self.ciphertext.flatten(), self.lifted_ciphertext.flatten()].concat()
    }
    fn schema(&self) -> Self::Schema {
        RnsBggPrivateRhs {
            ciphertext: self.ciphertext.schema(),
            lifted_ciphertext: self.lifted_ciphertext.schema(),
            rhs_id: self.rhs_id,
            level: self.level,
            context: self.context,
            id: self.id,
        }
    }
    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        let (ciphertext, lifted_ciphertext) = <(C, L)>::from_values(
            &(schema.ciphertext.clone(), schema.lifted_ciphertext.clone()),
            values,
        )?;
        Ok(Self {
            ciphertext,
            lifted_ciphertext,
            rhs_id: schema.rhs_id,
            level: schema.level,
            context: schema.context,
            id: schema.id,
        })
    }
}
impl<C: GraphValueSchema, L: GraphValueSchema> GraphValueSchema for RnsBggPrivateRhs<C, L> {
    type Value = RnsBggPrivateRhs<C::Value, L::Value>;
    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        RnsBggPrivateRhs {
            ciphertext: self.ciphertext.placeholders_from(next),
            lifted_ciphertext: self.lifted_ciphertext.placeholders_from(next),
            rhs_id: self.rhs_id,
            level: self.level,
            context: self.context,
            id: self.id,
        }
    }
    fn wire_types(&self) -> Vec<WireType> {
        [self.ciphertext.wire_types(), self.lifted_ciphertext.wire_types()].concat()
    }
}

impl<K: GraphValue, C: GraphValue, D: GraphValue> GraphValue for RnsBggPreparedPrivateMul<K, C, D> {
    type Schema = RnsBggPreparedPrivateMul<K::Schema, C::Schema, D::Schema>;

    fn flatten(&self) -> Vec<ValueHandle> {
        [self.input.flatten(), self.output.flatten(), self.rhs.flatten()]
            .into_iter()
            .flatten()
            .collect()
    }

    fn schema(&self) -> Self::Schema {
        RnsBggPreparedPrivateMul {
            input: self.input.schema(),
            output: self.output.schema(),
            rhs: self.rhs.schema(),
        }
    }

    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        let (input, output, rhs) = <(K, K, RnsBggPrivateRhs<C, D>)>::from_values(
            &(schema.input.clone(), schema.output.clone(), schema.rhs.clone()),
            values,
        )?;
        Ok(Self { input, output, rhs })
    }
}

impl<K: GraphValueSchema, C: GraphValueSchema, D: GraphValueSchema> GraphValueSchema
    for RnsBggPreparedPrivateMul<K, C, D>
{
    type Value = RnsBggPreparedPrivateMul<K::Value, C::Value, D::Value>;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        RnsBggPreparedPrivateMul {
            input: self.input.placeholders_from(next),
            output: self.output.placeholders_from(next),
            rhs: self.rhs.placeholders_from(next),
        }
    }

    fn wire_types(&self) -> Vec<WireType> {
        [self.input.wire_types(), self.output.wire_types(), self.rhs.wire_types()]
            .into_iter()
            .flatten()
            .collect()
    }
}

impl<K: GraphValue, M: GraphValue> GraphValue for RnsBggPreparedLinear<K, M> {
    type Schema = RnsBggPreparedLinear<K::Schema, M::Schema>;

    fn flatten(&self) -> Vec<ValueHandle> {
        [self.lhs.flatten(), self.rhs.flatten(), self.output.flatten(), self.operation.flatten()]
            .into_iter()
            .flatten()
            .collect()
    }

    fn schema(&self) -> Self::Schema {
        RnsBggPreparedLinear {
            lhs: self.lhs.schema(),
            rhs: self.rhs.schema(),
            output: self.output.schema(),
            operation: self.operation.schema(),
        }
    }

    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        let (lhs, rhs, output, operation) = <(K, K, K, LinearOperation<M>)>::from_values(
            &(
                schema.lhs.clone(),
                schema.rhs.clone(),
                schema.output.clone(),
                schema.operation.clone(),
            ),
            values,
        )?;
        Ok(Self { lhs, rhs, output, operation })
    }
}

impl<K: GraphValueSchema, M: GraphValueSchema> GraphValueSchema for RnsBggPreparedLinear<K, M> {
    type Value = RnsBggPreparedLinear<K::Value, M::Value>;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        RnsBggPreparedLinear {
            lhs: self.lhs.placeholders_from(next),
            rhs: self.rhs.placeholders_from(next),
            output: self.output.placeholders_from(next),
            operation: self.operation.placeholders_from(next),
        }
    }

    fn wire_types(&self) -> Vec<WireType> {
        [
            self.lhs.wire_types(),
            self.rhs.wire_types(),
            self.output.wire_types(),
            self.operation.wire_types(),
        ]
        .into_iter()
        .flatten()
        .collect()
    }
}

impl<M: GraphValue, D: GraphValue> GraphValue for PreparedBranch<M, D> {
    type Schema = PreparedBranch<M::Schema, D::Schema>;

    fn flatten(&self) -> Vec<ValueHandle> {
        [
            self.lifted_ciphertext.flatten(),
            self.lifted_intermediate.flatten(),
            self.mask_public.flatten(),
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    fn schema(&self) -> Self::Schema {
        PreparedBranch {
            lifted_ciphertext: self.lifted_ciphertext.schema(),
            lifted_intermediate: self.lifted_intermediate.schema(),
            mask_public: self.mask_public.schema(),
            sigma: self.sigma,
        }
    }

    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        let (lifted_ciphertext, lifted_intermediate, mask_public) = <(D, D, M)>::from_values(
            &(
                schema.lifted_ciphertext.clone(),
                schema.lifted_intermediate.clone(),
                schema.mask_public.clone(),
            ),
            values,
        )?;
        Ok(Self { lifted_ciphertext, lifted_intermediate, mask_public, sigma: schema.sigma })
    }
}

impl<M: GraphValueSchema, D: GraphValueSchema> GraphValueSchema for PreparedBranch<M, D> {
    type Value = PreparedBranch<M::Value, D::Value>;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        PreparedBranch {
            lifted_ciphertext: self.lifted_ciphertext.placeholders_from(next),
            lifted_intermediate: self.lifted_intermediate.placeholders_from(next),
            mask_public: self.mask_public.placeholders_from(next),
            sigma: self.sigma,
        }
    }

    fn wire_types(&self) -> Vec<WireType> {
        [
            self.lifted_ciphertext.wire_types(),
            self.lifted_intermediate.wire_types(),
            self.mask_public.wire_types(),
        ]
        .into_iter()
        .flatten()
        .collect()
    }
}

impl<K: GraphValue, M: GraphValue, D: GraphValue> GraphValue for RnsBggPreparedLut<K, M, D> {
    type Schema = RnsBggPreparedLut<K::Schema, M::Schema, D::Schema>;

    fn flatten(&self) -> Vec<ValueHandle> {
        [self.input.flatten(), self.output.flatten(), self.branches.flatten()]
            .into_iter()
            .flatten()
            .collect()
    }

    fn schema(&self) -> Self::Schema {
        RnsBggPreparedLut {
            input: self.input.schema(),
            output: self.output.schema(),
            branches: self.branches.schema(),
            binding: self.binding.clone(),
        }
    }

    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        let (input, output, branches) = <(K, K, Vec<PreparedBranch<M, D>>)>::from_values(
            &(schema.input.clone(), schema.output.clone(), schema.branches.clone()),
            values,
        )?;
        Ok(Self { input, output, branches, binding: schema.binding.clone() })
    }
}

impl<K: GraphValueSchema, M: GraphValueSchema, D: GraphValueSchema> GraphValueSchema
    for RnsBggPreparedLut<K, M, D>
{
    type Value = RnsBggPreparedLut<K::Value, M::Value, D::Value>;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        RnsBggPreparedLut {
            input: self.input.placeholders_from(next),
            output: self.output.placeholders_from(next),
            branches: self.branches.placeholders_from(next),
            binding: self.binding.clone(),
        }
    }

    fn wire_types(&self) -> Vec<WireType> {
        [self.input.wire_types(), self.output.wire_types(), self.branches.wire_types()]
            .into_iter()
            .flatten()
            .collect()
    }
}

impl<M: GraphValue> GraphValue for LinearOperation<M> {
    type Schema = LinearOperation<M::Schema>;

    fn flatten(&self) -> Vec<ValueHandle> {
        match self {
            Self::Add => Vec::new(),
            Self::MulPublic(multiplier) => multiplier.flatten(),
        }
    }

    fn schema(&self) -> Self::Schema {
        match self {
            Self::Add => LinearOperation::Add,
            Self::MulPublic(multiplier) => LinearOperation::MulPublic(multiplier.schema()),
        }
    }

    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        match schema {
            LinearOperation::Add if values.is_empty() => Ok(Self::Add),
            LinearOperation::Add => Err(DslError::Schema),
            LinearOperation::MulPublic(multiplier) => {
                Ok(Self::MulPublic(M::from_values(multiplier, values)?))
            }
        }
    }
}

impl<M: GraphValueSchema> GraphValueSchema for LinearOperation<M> {
    type Value = LinearOperation<M::Value>;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        match self {
            Self::Add => LinearOperation::Add,
            Self::MulPublic(multiplier) => {
                LinearOperation::MulPublic(multiplier.placeholders_from(next))
            }
        }
    }

    fn wire_types(&self) -> Vec<WireType> {
        match self {
            Self::Add => Vec::new(),
            Self::MulPublic(multiplier) => multiplier.wire_types(),
        }
    }
}

impl<K: GraphValue, M: GraphValue, P: GraphValue, D: GraphValue> GraphValue
    for PreparedGate<K, M, P, D>
{
    type Schema = PreparedGate<K::Schema, M::Schema, P::Schema, D::Schema>;
    fn flatten(&self) -> Vec<ValueHandle> {
        match self {
            Self::MulPrivate(value) => value.flatten(),
            Self::Linear(value) => value.flatten(),
            Self::Lut(value) => value.flatten(),
            Self::Reduction(value) => value.flatten(),
        }
    }
    fn schema(&self) -> Self::Schema {
        match self {
            Self::MulPrivate(value) => PreparedGate::MulPrivate(value.schema()),
            Self::Linear(value) => PreparedGate::Linear(value.schema()),
            Self::Lut(value) => PreparedGate::Lut(value.schema()),
            Self::Reduction(value) => PreparedGate::Reduction(value.schema()),
        }
    }
    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        Ok(match schema {
            PreparedGate::MulPrivate(schema) => {
                Self::MulPrivate(RnsBggPreparedPrivateMul::from_values(schema, values)?)
            }
            PreparedGate::Linear(schema) => {
                Self::Linear(RnsBggPreparedLinear::from_values(schema, values)?)
            }
            PreparedGate::Lut(schema) => Self::Lut(RnsBggPreparedLut::from_values(schema, values)?),
            PreparedGate::Reduction(schema) => Self::Reduction(K::from_values(schema, values)?),
        })
    }
}
impl<K: GraphValueSchema, M: GraphValueSchema, P: GraphValueSchema, D: GraphValueSchema>
    GraphValueSchema for PreparedGate<K, M, P, D>
{
    type Value = PreparedGate<K::Value, M::Value, P::Value, D::Value>;
    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        match self {
            Self::MulPrivate(value) => PreparedGate::MulPrivate(value.placeholders_from(next)),
            Self::Linear(value) => PreparedGate::Linear(value.placeholders_from(next)),
            Self::Lut(value) => PreparedGate::Lut(value.placeholders_from(next)),
            Self::Reduction(value) => PreparedGate::Reduction(value.placeholders_from(next)),
        }
    }
    fn wire_types(&self) -> Vec<WireType> {
        match self {
            Self::MulPrivate(value) => value.wire_types(),
            Self::Linear(value) => value.wire_types(),
            Self::Lut(value) => value.wire_types(),
            Self::Reduction(value) => value.wire_types(),
        }
    }
}
impl<G: GraphValue, P: GraphValue> GraphValue for RnsBggProgramCache<G, P> {
    type Schema = RnsBggProgramCache<G::Schema, P::Schema>;
    fn flatten(&self) -> Vec<ValueHandle> {
        [self.gates.flatten(), self.plaintexts.flatten()].concat()
    }
    fn schema(&self) -> Self::Schema {
        RnsBggProgramCache { gates: self.gates.schema(), plaintexts: self.plaintexts.schema() }
    }
    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        let (gates, plaintexts) =
            <(BTreeMap<ProgramGateInstance, G>, BTreeMap<ProgramGateInstance, P>)>::from_values(
                &(schema.gates.clone(), schema.plaintexts.clone()),
                values,
            )?;
        Ok(Self { gates, plaintexts })
    }
}
impl<G: GraphValueSchema, P: GraphValueSchema> GraphValueSchema for RnsBggProgramCache<G, P> {
    type Value = RnsBggProgramCache<G::Value, P::Value>;
    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        RnsBggProgramCache {
            gates: self.gates.placeholders_from(next),
            plaintexts: self.plaintexts.placeholders_from(next),
        }
    }
    fn wire_types(&self) -> Vec<WireType> {
        [self.gates.wire_types(), self.plaintexts.wire_types()].concat()
    }
}

impl<M: GraphValue> GraphValue for MaskBranch<M> {
    type Schema = MaskBranch<M::Schema>;
    fn flatten(&self) -> Vec<ValueHandle> {
        self.public.flatten()
    }
    fn schema(&self) -> Self::Schema {
        MaskBranch { public: self.public.schema(), sigma: self.sigma }
    }
    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        Ok(Self { public: <M>::from_values(&schema.public, values)?, sigma: schema.sigma })
    }
}
impl<M: GraphValueSchema> GraphValueSchema for MaskBranch<M> {
    type Value = MaskBranch<M::Value>;
    fn wire_types(&self) -> Vec<WireType> {
        self.public.wire_types()
    }
    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        MaskBranch { public: self.public.placeholders_from(next), sigma: self.sigma }
    }
}
impl<M: GraphValue> GraphValue for RnsBggLutMaterial<M> {
    type Schema = RnsBggLutMaterial<M::Schema>;
    fn flatten(&self) -> Vec<ValueHandle> {
        self.branches.flatten()
    }
    fn schema(&self) -> Self::Schema {
        RnsBggLutMaterial {
            branches: self.branches.schema(),
            mask_bank: self.mask_bank,
            id: self.id,
            level: self.level,
            context: self.context,
            table: self.table,
            rhs_id: self.rhs_id,
        }
    }
    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        Ok(Self {
            branches: <Vec<(MaskBranch<M>, M)>>::from_values(&schema.branches, values)?,
            mask_bank: schema.mask_bank,
            id: schema.id,
            level: schema.level,
            context: schema.context,
            table: schema.table,
            rhs_id: schema.rhs_id,
        })
    }
}
impl<M: GraphValueSchema> GraphValueSchema for RnsBggLutMaterial<M> {
    type Value = RnsBggLutMaterial<M::Value>;
    fn wire_types(&self) -> Vec<WireType> {
        self.branches.wire_types()
    }
    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        RnsBggLutMaterial {
            branches: self.branches.placeholders_from(next),
            mask_bank: self.mask_bank,
            id: self.id,
            level: self.level,
            context: self.context,
            table: self.table,
            rhs_id: self.rhs_id,
        }
    }
}
impl<M: GraphValue, D: GraphValue> GraphValue for RnsBggPublicRhs<M, D> {
    type Schema = RnsBggPublicRhs<M::Schema, D::Schema>;
    fn flatten(&self) -> Vec<ValueHandle> {
        match self {
            Self::Fixed(rhs) => rhs.flatten(),
            Self::Identity(_) => vec![],
        }
    }
    fn schema(&self) -> Self::Schema {
        match self {
            Self::Fixed(rhs) => RnsBggPublicRhs::Fixed(rhs.schema()),
            Self::Identity(id) => RnsBggPublicRhs::Identity(*id),
        }
    }
    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        match schema {
            RnsBggPublicRhs::Fixed(rhs) => {
                Ok(Self::Fixed(RnsBggPrivateRhs::from_values(rhs, values)?))
            }
            RnsBggPublicRhs::Identity(id) if values.is_empty() => Ok(Self::Identity(*id)),
            _ => Err(DslError::Schema),
        }
    }
}
impl<M: GraphValueSchema, D: GraphValueSchema> GraphValueSchema for RnsBggPublicRhs<M, D> {
    type Value = RnsBggPublicRhs<M::Value, D::Value>;
    fn wire_types(&self) -> Vec<WireType> {
        match self {
            Self::Fixed(rhs) => rhs.wire_types(),
            Self::Identity(_) => vec![],
        }
    }
    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        match self {
            Self::Fixed(rhs) => RnsBggPublicRhs::Fixed(rhs.placeholders_from(next)),
            Self::Identity(id) => RnsBggPublicRhs::Identity(*id),
        }
    }
}
impl<R: GraphValue> GraphValue for RnsBggOnlineMasks<R> {
    type Schema = RnsBggOnlineMasks<R::Schema>;
    fn flatten(&self) -> Vec<ValueHandle> {
        self.rows.as_ref().flatten()
    }
    fn schema(&self) -> Self::Schema {
        RnsBggOnlineMasks {
            rows: Arc::new(self.rows.as_ref().schema()),
            id: self.id,
            level: self.level,
            context: self.context,
        }
    }
    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        Ok(Self {
            rows: Arc::new(BTreeMap::from_values(schema.rows.as_ref(), values)?),
            id: schema.id,
            level: schema.level,
            context: schema.context,
        })
    }
}
impl<R: GraphValueSchema> GraphValueSchema for RnsBggOnlineMasks<R> {
    type Value = RnsBggOnlineMasks<R::Value>;
    fn wire_types(&self) -> Vec<WireType> {
        self.rows.as_ref().wire_types()
    }
    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        RnsBggOnlineMasks {
            rows: Arc::new(self.rows.as_ref().placeholders_from(next)),
            id: self.id,
            level: self.level,
            context: self.context,
        }
    }
}
impl<R: GraphValue> GraphValue for RnsBggOnlineLutHelpers<R> {
    type Schema = RnsBggOnlineLutHelpers<R::Schema>;
    fn flatten(&self) -> Vec<ValueHandle> {
        self.masks.flatten()
    }
    fn schema(&self) -> Self::Schema {
        RnsBggOnlineLutHelpers { masks: self.masks.schema(), binding: self.binding.clone() }
    }
    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        Ok(Self {
            masks: RnsBggOnlineMasks::from_values(&schema.masks, values)?,
            binding: schema.binding.clone(),
        })
    }
}
impl<R: GraphValueSchema> GraphValueSchema for RnsBggOnlineLutHelpers<R> {
    type Value = RnsBggOnlineLutHelpers<R::Value>;
    fn wire_types(&self) -> Vec<WireType> {
        self.masks.wire_types()
    }
    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        RnsBggOnlineLutHelpers {
            masks: self.masks.placeholders_from(next),
            binding: self.binding.clone(),
        }
    }
}
impl<I: GraphValue, K: GraphValue> GraphValue for RnsBggPreparedReduction<I, K> {
    type Schema = RnsBggPreparedReduction<I::Schema, K::Schema>;
    fn flatten(&self) -> Vec<ValueHandle> {
        [self.input.flatten(), self.output.flatten()].concat()
    }
    fn schema(&self) -> Self::Schema {
        RnsBggPreparedReduction { input: self.input.schema(), output: self.output.schema() }
    }
    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        let (input, output) =
            <(I, K)>::from_values(&(schema.input.clone(), schema.output.clone()), values)?;
        Ok(Self { input, output })
    }
}
impl<I: GraphValueSchema, K: GraphValueSchema> GraphValueSchema for RnsBggPreparedReduction<I, K> {
    type Value = RnsBggPreparedReduction<I::Value, K::Value>;
    fn wire_types(&self) -> Vec<WireType> {
        [self.input.wire_types(), self.output.wire_types()].concat()
    }
    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        RnsBggPreparedReduction {
            input: self.input.placeholders_from(next),
            output: self.output.placeholders_from(next),
        }
    }
}
impl<T: GraphValue> GraphValue for RnsBggTerminalAccumulator<T> {
    type Schema = RnsBggTerminalAccumulator<T::Schema>;
    fn flatten(&self) -> Vec<ValueHandle> {
        self.value.flatten()
    }
    fn schema(&self) -> Self::Schema {
        RnsBggTerminalAccumulator {
            value: self.value.schema(),
            level: self.level,
            context: self.context,
            stages: self.stages.clone(),
        }
    }
    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        Ok(Self {
            value: T::from_values(&schema.value, values)?,
            level: schema.level,
            context: schema.context,
            stages: schema.stages.clone(),
        })
    }
}
impl<T: GraphValueSchema> GraphValueSchema for RnsBggTerminalAccumulator<T> {
    type Value = RnsBggTerminalAccumulator<T::Value>;
    fn wire_types(&self) -> Vec<WireType> {
        self.value.wire_types()
    }
    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        RnsBggTerminalAccumulator {
            value: self.value.placeholders_from(next),
            level: self.level,
            context: self.context,
            stages: self.stages.clone(),
        }
    }
}
impl<M: GraphValue, L: GraphValue> GraphValue for RnsBggTerminalMaterial<M, L> {
    type Schema = RnsBggTerminalMaterial<M::Schema, L::Schema>;
    fn flatten(&self) -> Vec<ValueHandle> {
        [self.material.flatten(), self.lifted_ciphertexts.flatten()].concat()
    }
    fn schema(&self) -> Self::Schema {
        RnsBggTerminalMaterial {
            material: self.material.schema(),
            lifted_ciphertexts: self.lifted_ciphertexts.schema(),
        }
    }
    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        let (material, lifted_ciphertexts) = <(M, L)>::from_values(
            &(schema.material.clone(), schema.lifted_ciphertexts.clone()),
            values,
        )?;
        Ok(Self { material, lifted_ciphertexts })
    }
}
impl<M: GraphValueSchema, L: GraphValueSchema> GraphValueSchema for RnsBggTerminalMaterial<M, L> {
    type Value = RnsBggTerminalMaterial<M::Value, L::Value>;
    fn wire_types(&self) -> Vec<WireType> {
        [self.material.wire_types(), self.lifted_ciphertexts.wire_types()].concat()
    }
    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        RnsBggTerminalMaterial {
            material: self.material.placeholders_from(next),
            lifted_ciphertexts: self.lifted_ciphertexts.placeholders_from(next),
        }
    }
}
impl<K: GraphValue, M: GraphValue, P: GraphValue, D: GraphValue> GraphValue
    for RnsBggPreparedTerminal<K, M, P, D>
{
    type Schema = RnsBggPreparedTerminal<K::Schema, M::Schema, P::Schema, D::Schema>;
    fn flatten(&self) -> Vec<ValueHandle> {
        [self.input.flatten(), self.branches.flatten(), self.raw_public.flatten()].concat()
    }
    fn schema(&self) -> Self::Schema {
        RnsBggPreparedTerminal {
            input: self.input.schema(),
            branches: self.branches.schema(),
            raw_public: self.raw_public.schema(),
            binding: self.binding.clone(),
            id: self.id,
        }
    }
    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        let (input, branches, raw_public) = <(K, Vec<PreparedBranch<M, D>>, P)>::from_values(
            &(schema.input.clone(), schema.branches.clone(), schema.raw_public.clone()),
            values,
        )?;
        Ok(Self { input, branches, raw_public, binding: schema.binding.clone(), id: schema.id })
    }
}
impl<K: GraphValueSchema, M: GraphValueSchema, P: GraphValueSchema, D: GraphValueSchema>
    GraphValueSchema for RnsBggPreparedTerminal<K, M, P, D>
{
    type Value = RnsBggPreparedTerminal<K::Value, M::Value, P::Value, D::Value>;
    fn wire_types(&self) -> Vec<WireType> {
        [self.input.wire_types(), self.branches.wire_types(), self.raw_public.wire_types()].concat()
    }
    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        RnsBggPreparedTerminal {
            input: self.input.placeholders_from(next),
            branches: self.branches.placeholders_from(next),
            raw_public: self.raw_public.placeholders_from(next),
            binding: self.binding.clone(),
            id: self.id,
        }
    }
}
impl RnsBggScheme {
    /// Decompose in the low ring once, then transport bounded digits to the
    /// input ring. The decomposition is independent of the input public key.
    pub fn prepare_private_rhs(
        &self,
        rhs: &RnsBggPrivateRhs,
    ) -> Result<RnsBggPrivateRhs<RnsBggMatrix, RnsBggDigits>, RnsBggError> {
        Ok(RnsBggPrivateRhs {
            ciphertext: rhs.ciphertext.clone(),
            lifted_ciphertext: self.prepare_digits(rhs.level, rhs.ciphertext.clone())?,
            level: rhs.level,
            context: rhs.context,
            id: rhs.id,
            rhs_id: rhs.rhs_id,
        })
    }
    /// Select the active external range before decomposing. Both returned families
    /// use packed local indices; the scope's input range owns the external offset.
    pub fn prepare_private_rhs_family(
        &self,
        rhs: &RnsBggPrivateRhs<RnsBggRhsFamily>,
        range: &FamilyRange,
    ) -> Result<RnsBggPrivateRhs<Family<RnsBggMatrix>, Family<RnsBggDigits>>, RnsBggError> {
        let values = mxx_dsl::parallel(range.count().clone(), |index| {
            let element = RnsBggPrivateRhs {
                ciphertext: rhs.ciphertext.at(index.add(Int::evaluate(range.start().clone())))?,
                lifted_ciphertext: (),
                level: rhs.level,
                context: rhs.context,
                id: rhs.id,
                rhs_id: rhs.rhs_id,
            };
            self.prepare_private_rhs(&element).map_err(|_| DslError::Schema)
        })?;
        Ok(RnsBggPrivateRhs {
            ciphertext: values.field(|value| value.ciphertext)?,
            lifted_ciphertext: values.field(|value| value.lifted_ciphertext)?,
            level: rhs.level,
            context: rhs.context,
            id: rhs.id,
            rhs_id: rhs.rhs_id,
        })
    }

    /// At Q=p*q, prepare MD(A*Dec(C)) once. No internal public rounding,
    /// automorphism bank, or mask-secret-dependent helper enters this operation.
    pub fn prepare_mul_private(
        &self,
        input: &RnsBggPublicKey,
        rhs: &RnsBggPrivateRhs<RnsBggMatrix, RnsBggDigits>,
        output_id: [u8; 32],
    ) -> Result<RnsBggPreparedPrivateMul, RnsBggError> {
        use mxx_primitives::poly::PolyParams;
        self.check_key(input)?;
        if input.level != rhs.level || input.context != rhs.context {
            return Err(RnsBggError::Context);
        }
        let next = num_bigint::BigInt::from(
            self.parameters.level(input.level + 1)?.modulus().as_ref().clone(),
        );
        let output = RnsBggPublicKey {
            matrix: (input.matrix.clone() * rhs.lifted_ciphertext.clone()).block_mod_switch(next),
            level: input.level + 1,
            id: output_id,
            context: self.context,
        };
        Ok(RnsBggPreparedPrivateMul { input: input.clone(), output, rhs: rhs.clone() })
    }
    pub fn prepare_mul_private_family(
        &self,
        input: &RnsBggPublicKeyFamily,
        rhs: &ProgramValue<
            RnsBggPrivateRhs<RnsBggMatrix, RnsBggDigits>,
            RnsBggPrivateRhs<Family<RnsBggMatrix>, Family<RnsBggDigits>>,
        >,
        output_id: [u8; 32],
    ) -> Result<
        RnsBggPreparedPrivateMul<
            RnsBggPublicKeyFamily,
            ProgramValue<RnsBggMatrix, Family<RnsBggMatrix>>,
            ProgramValue<RnsBggDigits, Family<RnsBggDigits>>,
        >,
        RnsBggError,
    > {
        self.check_public_family(input)?;
        let (level, context, id, rhs_id) = match rhs {
            ProgramValue::Scalar(rhs) => (rhs.level, rhs.context, rhs.id, rhs.rhs_id),
            ProgramValue::Family(rhs) => (rhs.level, rhs.context, rhs.id, rhs.rhs_id),
        };
        if input.level != level || input.context != context {
            return Err(RnsBggError::Context);
        }
        let (ciphertext, lifted_ciphertext) = match rhs {
            ProgramValue::Scalar(rhs) => (
                ProgramValue::Scalar(rhs.ciphertext.clone()),
                ProgramValue::Scalar(rhs.lifted_ciphertext.clone()),
            ),
            ProgramValue::Family(rhs) => (
                ProgramValue::Family(rhs.ciphertext.clone()),
                ProgramValue::Family(rhs.lifted_ciphertext.clone()),
            ),
        };
        let rhs = RnsBggPrivateRhs { ciphertext, lifted_ciphertext, level, context, id, rhs_id };
        let matrices = mxx_dsl::parallel(input.range.count().clone(), |index| {
            let input = RnsBggPublicKey {
                matrix: input.matrices.at(&index),
                level: input.level,
                context: input.context,
                id: input.id,
            };
            let rhs = RnsBggPrivateRhs {
                ciphertext: rhs.ciphertext.at(&index),
                lifted_ciphertext: rhs.lifted_ciphertext.at(&index),
                level: rhs.level,
                context: rhs.context,
                id: rhs.id,
                rhs_id: rhs.rhs_id,
            };
            self.prepare_mul_private(&input, &rhs, output_id)
                .map(|prepared| prepared.output.matrix)
                .map_err(|_| DslError::Schema)
        })
        .map_err(|error| RnsBggError::Graph(error.to_string()))?;
        let output = RnsBggPublicKeyFamily {
            matrices: ProgramValue::Family(matrices),
            level: input.level + 1,
            id: output_id,
            context: self.context,
            range: input.range.clone(),
        };
        Ok(RnsBggPreparedPrivateMul { input: input.clone(), output, rhs: rhs.clone() })
    }
    pub fn prepare_add(
        &self,
        lhs: &RnsBggPublicKey,
        rhs: &RnsBggPublicKey,
        output_id: [u8; 32],
    ) -> Result<RnsBggPreparedLinear, RnsBggError> {
        self.check_key(lhs)?;
        self.check_key(rhs)?;
        if lhs.level != rhs.level {
            return Err(RnsBggError::Context);
        }
        Ok(RnsBggPreparedLinear {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            output: RnsBggPublicKey {
                matrix: lhs.matrix.clone() + rhs.matrix.clone(),
                level: lhs.level,
                id: output_id,
                context: self.context,
            },
            operation: LinearOperation::Add,
        })
    }
    /// The multiplier is the known plaintext polynomial of the public operand.
    /// Its key is checked even though its mask does not enter the output formula.
    pub fn prepare_mul_public(
        &self,
        private: &RnsBggPublicKey,
        public: &RnsBggPublicKey,
        multiplier: Mat,
        output_id: [u8; 32],
    ) -> Result<RnsBggPreparedLinear, RnsBggError> {
        self.check_key(private)?;
        self.check_key(public)?;
        if private.level != public.level ||
            multiplier.matrix_type() != &self.ring(private.level)?.matrix_type((1, 1))
        {
            return Err(RnsBggError::Context);
        }
        Ok(RnsBggPreparedLinear {
            lhs: private.clone(),
            rhs: public.clone(),
            output: RnsBggPublicKey {
                matrix: multiplier.clone() * private.matrix.clone(),
                level: private.level,
                id: output_id,
                context: self.context,
            },
            operation: LinearOperation::MulPublic(multiplier),
        })
    }
    pub fn sample_input_key(
        &self,
        level: usize,
        hash_key: Bytes,
        tag: impl Into<HashTag>,
        id: [u8; 32],
    ) -> Result<RnsBggPublicKey, RnsBggError> {
        Ok(RnsBggPublicKey {
            matrix: RnsBggMatrix::hash(
                &self.ring(level)?,
                hash_key,
                tag,
                self.parameters.component_columns(level)?,
            ),
            level,
            id,
            context: self.context,
        })
    }
    pub fn prepare_input(
        &self,
        secret: &RnsBggPayloadSecret,
        key: &RnsBggPublicKey,
        payload: Mat,
    ) -> Result<RnsBggInputMaterial, RnsBggError> {
        self.check_key(key)?;
        if payload.matrix_type() != &self.ring(key.level)?.matrix_type((1, 1)) {
            return Err(RnsBggError::Context);
        }
        let r = self.payload_at(secret, key.level)?;
        let scale = -BigInt::from(self.parameters.scale(key.level)?.clone());
        let gadget = self.state_gadget(key.level)?;
        let row =
            (self.scalar(key.level, scale)? * payload * r).map(|entry| entry * gadget.clone());
        Ok(RnsBggInputMaterial { row, level: key.level, key_id: key.id, context: self.context })
    }
    pub(crate) fn check_key(&self, key: &RnsBggPublicKey) -> Result<(), RnsBggError> {
        if key.context != self.context ||
            key.matrix.matrix_type() !=
                &self
                    .ring(key.level)?
                    .matrix_type((2, 2 * self.parameters.component_columns(key.level)?))
        {
            return Err(RnsBggError::Context);
        }
        Ok(())
    }
    /// Public preprocessing only. `id` is unique to these public matrices and
    /// is retained by artifact import; it is not a ciphertext commitment.
    pub fn sample_mask_bank(
        &self,
        level: usize,
        layout: ExpLayout,
        hash_key: Bytes,
        tag: impl Into<HashTag>,
        id: [u8; 32],
    ) -> Result<RnsBggMaskBank, RnsBggError> {
        let ring = self.ring(level)?;
        self.parameters.level(level + 1)?;
        let sigmas = layout
            .automorphism_indices(self.parameters.level(level)?.ring_dimension() as usize)
            .map_err(|_| RnsBggError::Lut)?;
        let root = tag.into();
        let columns = self.parameters.component_columns(level)?;
        let batch = crate::utils::preprocess_batch_size()?;
        let mut branches = Vec::with_capacity(sigmas.len());
        for start in (0..sigmas.len()).step_by(batch) {
            let len = batch.min(sigmas.len() - start);
            let public = mxx_dsl::parallel(len, |local| {
                let index = local.add(Int::constant(start));
                let sigma = layout
                    .automorphism_index_expression(
                        self.parameters.level(level).unwrap().ring_dimension() as usize,
                        index.expression()?,
                    )
                    .map_err(|_| DslError::Schema)?;
                let mut tag = root.clone();
                tag.push("rns_bgg-mask");
                tag.push(mxx_ir_core::IntExpr::constant(level));
                tag.push(sigma);
                Ok(RnsBggMatrix::hash(&ring, hash_key.clone(), tag, columns))
            })?;
            for local in 0..len {
                branches
                    .push(MaskBranch { sigma: sigmas[start + local], public: public.at(local) });
            }
        }
        Ok(RnsBggMaskBank { id, level, context: self.context, branches })
    }
    pub fn prepare_lut(
        &self,
        input: &RnsBggPublicKey,
        material: &RnsBggLutMaterial,
        output_id: [u8; 32],
    ) -> Result<RnsBggPreparedLut, RnsBggError> {
        if material.table.is_some() && material.rhs_id.is_some() {
            return Err(RnsBggError::Lut);
        }
        let lifted = material
            .branches
            .iter()
            .map(|(_, ciphertext)| self.prepare_digits(input.level, ciphertext.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let (raw_public, branches) = self.prepare_lut_raw(input, material, &lifted)?;
        let next = BigInt::from(self.parameters.level(input.level + 1)?.modulus().as_ref().clone());
        let output = RnsBggPublicKey {
            matrix: raw_public.block_mod_switch(next),
            level: input.level + 1,
            id: output_id,
            context: self.context,
        };
        Ok(RnsBggPreparedLut {
            binding: material.binding(),
            input: input.clone(),
            output,
            branches,
        })
    }
    /// Shared branch preparation; terminal batches retain K before its outer switch.
    pub(super) fn prepare_lut_raw(
        &self,
        input: &RnsBggPublicKey,
        material: &RnsBggLutMaterial,
        lifted_ciphertexts: &[RnsBggDigits],
    ) -> Result<(RnsBggMatrix, Vec<PreparedBranch>), RnsBggError> {
        self.check_key(input)?;
        if input.level != material.level || input.context != material.context {
            return Err(RnsBggError::Context);
        }
        let level = input.level;
        let next = BigInt::from(self.parameters.level(level + 1)?.modulus().as_ref().clone());
        let batch = crate::utils::preprocess_batch_size()?;
        let ring = self.ring(level)?;
        let first = &material.branches[0].1;
        let mut raw_public = RnsBggMatrix::new(std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                ring.zero((1, first.entry(row, column).matrix_type().columns.clone()))
            })
        }))?;
        let mut branches = Vec::with_capacity(material.branches.len());
        for start in (0..material.branches.len()).step_by(batch) {
            let end = (start + batch).min(material.branches.len());
            let members = &material.branches[start..end];
            let lifted = &lifted_ciphertexts[start..end];
            let inputs = Family::pack(
                members
                    .iter()
                    .zip(lifted)
                    .map(|((mask, _), lifted)| (mask.public.clone(), lifted.clone()))
                    .collect(),
            )?;
            let prepared = mxx_dsl::parallel(members.len(), |index| {
                let sigma = IntExpr::Select {
                    selector: Box::new(index.expression()?),
                    branches: members
                        .iter()
                        .map(|(mask, _)| IntExpr::constant(mask.sigma))
                        .collect(),
                };
                let (mask, lifted) = inputs.at(index);
                let intermediate = (input.matrix.clone().ring_automorphism(sigma) * lifted)
                    .block_mod_switch(next.clone());
                let intermediate =
                    self.prepare_digits(level, intermediate).map_err(|_| DslError::Schema)?;
                let public = mask.clone() * intermediate.clone();
                Ok(((mask, intermediate), public))
            })?;
            let sum = crate::utils::balanced_sum_family(prepared.field(|(_, public)| public)?)
                .map_err(|_| DslError::Schema)?;
            raw_public = raw_public + sum;
            let cache = prepared.field(|(cache, _)| cache)?;
            for (index, (mask, _)) in members.iter().enumerate() {
                let (mask_public, lifted_intermediate) = cache.at(index);
                branches.push(PreparedBranch {
                    sigma: mask.sigma,
                    mask_public,
                    lifted_ciphertext: lifted[index].clone(),
                    lifted_intermediate,
                });
            }
        }
        Ok((raw_public, branches))
    }
    pub fn prepare_refresh_recomposition(
        &self,
        output_key: RnsBggPublicKey,
    ) -> Result<RnsBggRefreshRecomposition, RnsBggError> {
        let parameters = self.parameters.level(output_key.level)?;
        let q = BigInt::from(parameters.modulus().as_ref().clone());
        let coefficients = parameters
            .to_crt()
            .0
            .into_iter()
            .map(|prime| {
                let prime = BigInt::from(prime);
                let h = &q / &prime;
                let inverse = crate::utils::modular_inverse(&h, &prime)
                    .ok_or(RnsBggError::Parameters("non-coprime CRT basis"))?;
                self.scalar(output_key.level, h * inverse)
            })
            .collect::<Result<_, _>>()?;
        Ok(RnsBggRefreshRecomposition { output_key, coefficients })
    }
    pub fn export_prepared_terminal(
        &self,
        prepared: &RnsBggPreparedTerminal,
        mut graph: mxx_dsl::DslContext,
        prefix: &str,
    ) -> Result<(mxx_dsl::DslContext, RnsBggPreparedTerminalDescriptor), RnsBggError> {
        self.check_key(&prepared.input)?;
        graph = graph.public_output(prefix, prepared.clone())?;
        Ok((
            graph,
            RnsBggPreparedTerminalDescriptor {
                prepared: RnsBggPreparedLutDescriptor {
                    prefix: prefix.into(),
                    parameters: self.parameter_identity(),
                    binding: prepared.binding.clone(),
                    input_id: prepared.input.id,
                    output_id: prepared.id,
                    sigmas: prepared.branches.iter().map(|branch| branch.sigma).collect(),
                    schema: prepared.schema(),
                },
            },
        ))
    }
    /// Expected identities come from preprocessing. Online s is only needed
    /// later, by prepare_online_masks on the restored public mask bank.
    pub fn import_prepared_terminal(
        &self,
        descriptor: &RnsBggPreparedTerminalDescriptor,
        production: mxx_ir_core::artifact::ProductionId,
        binding: &RnsBggLutBinding,
        input_id: [u8; 32],
        stage_id: [u8; 32],
    ) -> Result<RnsBggPreparedTerminal, RnsBggError> {
        let data = &descriptor.prepared;
        self.check_prepared_descriptor(data, binding, input_id, stage_id)?;
        Ok(data.schema.artifact_input(production, &data.prefix, ArtifactConfidentiality::Public)?)
    }

    pub(super) fn artifact_binding(&self) -> RnsBggArtifactBinding {
        RnsBggArtifactBinding { parameters: self.parameter_identity(), context: self.context }
    }
    pub(super) fn parameter_identity(&self) -> [u8; 32] {
        *self.parameter_identity.get_or_init(|| {
            use sha2::Digest;
            let mut hash = sha2::Sha256::new();
            hash.update(b"rns_bgg-prepared-parameters");
            let sigma = self.parameters.encoding_error_sigma.to_string();
            hash.update((sigma.len() as u64).to_le_bytes());
            hash.update(sigma.as_bytes());
            let bound = self.parameters.encoding_error_bound.to_bytes_be();
            hash.update((bound.len() as u64).to_le_bytes());
            hash.update(bound);
            hash.update((self.parameters.operations_per_drop() as u64).to_le_bytes());
            hash.update((self.parameters.levels().len() as u64).to_le_bytes());
            for level in self.parameters.levels() {
                hash.update(level.ring_dimension().to_le_bytes());
                hash.update(level.base_bits().to_le_bytes());
                let primes = level.to_crt().0;
                hash.update((primes.len() as u64).to_le_bytes());
                for prime in primes {
                    hash.update(prime.to_le_bytes());
                }
            }
            hash.finalize().into()
        })
    }
    pub(super) fn check_public_family(
        &self,
        family: &RnsBggPublicKeyFamily,
    ) -> Result<(), RnsBggError> {
        if family.context != self.context ||
            matches!(&family.matrices, ProgramValue::Family(values) if values.count() != family.range.count()) ||
            family.matrices.element_schema().matrix_type() !=
                self.ring(family.level)?
                    .matrix_type((2, 2 * self.parameters.component_columns(family.level)?))
        {
            return Err(RnsBggError::Context);
        }
        Ok(())
    }
    /// Double-scale note (8.4): combine slot ciphertexts in the low base before
    /// preparing any public intermediates or applying block ModSwitch.
    /// `multipliers` contains active public labels only; `range.start` indexes
    /// the full ciphertext family. Monomial labels and one-hot selectors are
    /// caller contracts, not runtime ciphertext checks. The fresh-error bound
    /// includes every active label's l1 gain, including zero-selector slots.
    /// `rhs_id` identifies this label-specific composed RHS artifact, including
    /// the source material and active range, and is also its material identity.
    pub fn compose_lut_material(
        &self,
        material: &RnsBggLutMaterialFamily,
        multipliers: &Family<Mat>,
        range: &FamilyRange,
        rhs_id: [u8; 32],
    ) -> Result<RnsBggLutMaterial, RnsBggError> {
        if material.binding.context != self.context ||
            material.binding.rhs.is_none() ||
            material.binding.table.is_some() ||
            multipliers.count() != range.count() ||
            multipliers.element_type() !=
                &self.ring(material.binding.level + 1)?.matrix_type((1, 1))
        {
            return Err(RnsBggError::Context);
        }
        if let (Ok(start), Ok(count), Ok(capacity)) = (
            range.start().evaluate(&ParamEnv::default()),
            range.count().evaluate(&ParamEnv::default()),
            material.ciphertexts.count().evaluate(&ParamEnv::default()),
        ) {
            if start < 0.into() || count <= 0.into() || start + count > capacity {
                return Err(RnsBggError::Context);
            }
        }
        let branches = material
            .masks
            .iter()
            .enumerate()
            .map(|(branch, mask)| {
                let count = range
                    .count()
                    .evaluate(&ParamEnv::default())
                    .ok()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or(RnsBggError::Context)?;
                let ciphertext = crate::utils::reduce_generated_segments(
                    1,
                    count,
                    |_, index| {
                        let external = index.clone().add(Int::evaluate(range.start().clone()));
                        multipliers.at(&index).ring_automorphism(mask.sigma) *
                            material.ciphertexts.at(external).0[branch].clone()
                    },
                    |left, right| left + right,
                )
                .map_err(|error| RnsBggError::Graph(error.to_string()))?
                .at(0);
                Ok((mask.clone(), ciphertext))
            })
            .collect::<Result<Vec<_>, RnsBggError>>()?;
        Ok(RnsBggLutMaterial {
            mask_bank: material.binding.mask_bank,
            id: rhs_id,
            level: material.binding.level,
            context: material.binding.context,
            table: material.binding.table,
            rhs_id: Some(rhs_id),
            branches,
        })
    }
    pub fn prepare_lut_family(
        &self,
        input: &RnsBggPublicKeyFamily,
        material: &RnsBggLutMaterialFamily,
        output_id: [u8; 32],
    ) -> Result<RnsBggPreparedLutFamily, RnsBggError> {
        self.check_public_family(input)?;
        if input.level != material.binding.level || input.context != material.binding.context {
            return Err(RnsBggError::Context);
        }
        // Check concrete ranges here; symbolic bounds remain graph-validated.
        if let (Ok(start), Ok(count), Ok(capacity)) = (
            input.range.start().evaluate(&ParamEnv::default()),
            input.range.count().evaluate(&ParamEnv::default()),
            material.ciphertexts.count().evaluate(&ParamEnv::default()),
        ) {
            if start < 0.into() || count <= 0.into() || start + count > capacity {
                return Err(RnsBggError::Context);
            }
        }
        let prepared = mxx_dsl::parallel(input.range.count().clone(), |index| {
            let external = index.clone().add(Int::evaluate(input.range.start().clone()));
            let scalar_material = material.at(external);
            let key = RnsBggPublicKey {
                matrix: input.matrices.at(&index),
                level: input.level,
                id: input.id,
                context: input.context,
            };
            self.prepare_lut(&key, &scalar_material, output_id).map_err(|_| DslError::Schema)
        })?;
        let matrices = prepared.field(|value| value.output.matrix)?;
        Ok(RnsBggPreparedLutFamily {
            input: input.clone(),
            output: RnsBggPublicKeyFamily {
                matrices: ProgramValue::Family(matrices),
                level: input.level + 1,
                id: output_id,
                context: self.context,
                range: input.range.clone(),
            },
            prepared,
        })
    }
    /// A supplied multiplier selects public multiplication; absence selects
    /// addition. The resulting prepared operation fixes the online formula.
    pub fn prepare_linear_family(
        &self,
        lhs: &RnsBggPublicKeyFamily,
        rhs: &RnsBggPublicKeyFamily,
        multiplier: Option<ProgramValue<Mat, Family<Mat>>>,
        output_id: [u8; 32],
    ) -> Result<
        RnsBggPreparedLinear<RnsBggPublicKeyFamily, ProgramValue<Mat, Family<Mat>>>,
        RnsBggError,
    > {
        self.check_public_family(lhs)?;
        self.check_public_family(rhs)?;
        if lhs.level != rhs.level || lhs.range != rhs.range {
            return Err(RnsBggError::Context);
        }
        let scalar_type = self.ring(lhs.level)?.matrix_type((1, 1));
        if multiplier.as_ref().is_some_and(|value| {
            matches!(value, ProgramValue::Family(values) if values.count() != lhs.range.count()) ||
                value.element_schema() != MatType(scalar_type.clone())
        }) {
            return Err(RnsBggError::Context);
        }
        let matrices = mxx_dsl::parallel(lhs.range.count().clone(), |index| {
            let left = lhs.matrices.at(&index);
            Ok(match &multiplier {
                Some(multiplier) => multiplier.at(&index) * left,
                None => left + rhs.matrices.at(&index),
            })
        })
        .map_err(|e| RnsBggError::Graph(e.to_string()))?;
        Ok(RnsBggPreparedLinear {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            operation: multiplier.map_or(LinearOperation::Add, LinearOperation::MulPublic),
            output: RnsBggPublicKeyFamily {
                matrices: ProgramValue::Family(matrices),
                level: lhs.level,
                id: output_id,
                context: self.context,
                range: lhs.range.clone(),
            },
        })
    }
    pub fn prepare_reduce_family(
        &self,
        value: &RnsBggPublicKeyFamily,
        output_id: [u8; 32],
    ) -> Result<RnsBggPreparedReduction, RnsBggError> {
        self.check_public_family(value)?;
        let matrix = match &value.matrices {
            ProgramValue::Scalar(matrix) => {
                self.ring(value.level)?.polynomial([value.range.count().clone()]) * matrix.clone()
            }
            ProgramValue::Family(matrices) => sum(matrices.clone())?,
        };
        Ok(RnsBggPreparedReduction {
            input: value.clone(),
            output: RnsBggPublicKey {
                matrix,
                level: value.level,
                id: output_id,
                context: self.context,
            },
        })
    }
    pub(super) fn check_linear_descriptor(
        &self,
        descriptor: &RnsBggPreparedLinearDescriptor,
        expected: &RnsBggLinearBinding,
    ) -> Result<(), RnsBggError> {
        if descriptor.parameters != self.parameter_identity() ||
            descriptor.binding != *expected ||
            expected.context != self.context
        {
            return Err(RnsBggError::Context);
        }
        self.ring(expected.level)?;
        Ok(())
    }
    pub(super) fn import_linear_key(
        &self,
        descriptor: &RnsBggPreparedLinearDescriptor,
        production: &ProductionId,
        role: &str,
        id: [u8; 32],
    ) -> Result<RnsBggPublicKey, RnsBggError> {
        Ok(RnsBggPublicKey {
            matrix: import_blocks(
                &self.ring(descriptor.binding.level)?,
                production,
                &format!("{}.{role}", descriptor.prefix),
                self.parameters.component_columns(descriptor.binding.level)?,
            ),
            level: descriptor.binding.level,
            id,
            context: self.context,
        })
    }
    pub(super) fn import_linear_key_family(
        &self,
        descriptor: &RnsBggPreparedLinearDescriptor,
        production: &ProductionId,
        role: &str,
        id: [u8; 32],
    ) -> Result<RnsBggPublicKeyFamily, RnsBggError> {
        let range = descriptor.binding.operation.input_range().ok_or(RnsBggError::Context)?;
        let ring = self.ring(descriptor.binding.level)?;
        let schema = BlocksType::<2, 2>(std::array::from_fn(|_| {
            std::array::from_fn(|_| {
                MatType(
                    ring.matrix_type((
                        1,
                        self.parameters
                            .component_columns(descriptor.binding.level)
                            .expect("validated key level"),
                    )),
                )
            })
        }));
        let output_storage = ProgramValue::Family(vec![false; 4]);
        Ok(RnsBggPublicKeyFamily {
            matrices: import_family_value(
                schema,
                range.count().clone(),
                match role {
                    "lhs" => &descriptor.key_storage[0],
                    "rhs" => &descriptor.key_storage[1],
                    _ => &output_storage,
                },
                production.clone(),
                &format!("{}.{role}", descriptor.prefix),
            )?,
            level: descriptor.binding.level,
            id,
            context: self.context,
            range: range.clone(),
        })
    }
    pub fn export_prepared_linear(
        &self,
        prepared: &RnsBggPreparedLinear,
        mut graph: DslContext,
        prefix: &str,
    ) -> Result<(DslContext, RnsBggPreparedLinearDescriptor), RnsBggError> {
        let keys = [("lhs", &prepared.lhs), ("rhs", &prepared.rhs), ("output", &prepared.output)];
        for (_, key) in &keys {
            self.check_key(key)?;
        }
        graph = graph.public_outputs(
            keys.map(|(role, key)| (format!("{prefix}.{role}"), key.matrix.clone())),
        )?;
        let (graph, operation) = prepared.operation.export(graph, prefix, prepared.rhs.id, None)?;
        Ok((
            graph,
            RnsBggPreparedLinearDescriptor {
                key_storage: [ProgramValue::Scalar(()), ProgramValue::Scalar(())],
                multiplier_storage: ProgramValue::Scalar(()),
                prefix: prefix.into(),
                parameters: self.parameter_identity(),
                binding: RnsBggLinearBinding {
                    context: self.context,
                    level: prepared.lhs.level,
                    lhs: prepared.lhs.id,
                    output: prepared.output.id,
                    operation,
                },
            },
        ))
    }
    /// Validate metadata once when constructing the consuming graph. The
    /// production manifest must also be supplied when freezing that graph.
    pub fn import_prepared_linear(
        &self,
        descriptor: &RnsBggPreparedLinearDescriptor,
        production: ProductionId,
        expected: &RnsBggLinearBinding,
    ) -> Result<RnsBggPreparedLinear, RnsBggError> {
        self.check_linear_descriptor(descriptor, expected)?;
        if expected.operation.input_range().is_some() {
            return Err(RnsBggError::Context);
        }
        let (operation, rhs_id) = LinearOperation::import(
            &expected.operation,
            &MatType(self.ring(expected.level)?.matrix_type((1, 1))),
            production.clone(),
            &descriptor.prefix,
        )?;
        Ok(RnsBggPreparedLinear {
            lhs: self.import_linear_key(descriptor, &production, "lhs", expected.lhs)?,
            rhs: self.import_linear_key(descriptor, &production, "rhs", rhs_id)?,
            output: self.import_linear_key(descriptor, &production, "output", expected.output)?,
            operation,
        })
    }
    pub fn export_prepared_linear_family(
        &self,
        prepared: &RnsBggPreparedLinear<RnsBggPublicKeyFamily, ProgramValue<Mat, Family<Mat>>>,
        mut graph: DslContext,
        prefix: &str,
    ) -> Result<(DslContext, RnsBggPreparedLinearDescriptor), RnsBggError> {
        let keys = [("lhs", &prepared.lhs), ("rhs", &prepared.rhs), ("output", &prepared.output)];
        for (_, key) in &keys {
            self.check_public_family(key)?;
        }
        graph = graph.public_outputs(
            keys.map(|(role, key)| (format!("{prefix}.{role}"), key.matrices.clone())),
        )?;
        let (graph, operation) = prepared.operation.export(
            graph,
            prefix,
            prepared.rhs.id,
            Some(prepared.lhs.range.clone()),
        )?;
        Ok((
            graph,
            RnsBggPreparedLinearDescriptor {
                key_storage: [
                    family_storage(&prepared.lhs.matrices),
                    family_storage(&prepared.rhs.matrices),
                ],
                multiplier_storage: match &prepared.operation {
                    LinearOperation::Add => ProgramValue::Scalar(()),
                    LinearOperation::MulPublic(value) => family_storage(value),
                },
                prefix: prefix.into(),
                parameters: self.parameter_identity(),
                binding: RnsBggLinearBinding {
                    context: self.context,
                    level: prepared.lhs.level,
                    lhs: prepared.lhs.id,
                    output: prepared.output.id,
                    operation,
                },
            },
        ))
    }
    pub fn import_prepared_linear_family(
        &self,
        descriptor: &RnsBggPreparedLinearDescriptor,
        production: ProductionId,
        expected: &RnsBggLinearBinding,
    ) -> Result<
        RnsBggPreparedLinear<RnsBggPublicKeyFamily, ProgramValue<Mat, Family<Mat>>>,
        RnsBggError,
    > {
        self.check_linear_descriptor(descriptor, expected)?;
        let range = expected.operation.input_range().ok_or(RnsBggError::Context)?;
        let ring = self.ring(expected.level)?;
        let (operation, rhs_id) = LinearOperation::import(
            &expected.operation,
            &match &descriptor.multiplier_storage {
                ProgramValue::Scalar(()) => ProgramValue::Scalar(MatType(ring.matrix_type((1, 1)))),
                ProgramValue::Family(shared) => ProgramValue::Family(FamilyType {
                    element: MatType(ring.matrix_type((1, 1))),
                    count: range.count().clone(),
                    shared: shared.clone(),
                }),
            },
            production.clone(),
            &descriptor.prefix,
        )?;
        Ok(RnsBggPreparedLinear {
            lhs: self.import_linear_key_family(descriptor, &production, "lhs", expected.lhs)?,
            rhs: self.import_linear_key_family(descriptor, &production, "rhs", rhs_id)?,
            output: self.import_linear_key_family(
                descriptor,
                &production,
                "output",
                expected.output,
            )?,
            operation,
        })
    }
    pub fn export_prepared_reduction(
        &self,
        prepared: &RnsBggPreparedReduction,
        graph: DslContext,
        prefix: &str,
    ) -> Result<(DslContext, RnsBggPreparedLinearDescriptor), RnsBggError> {
        self.check_public_family(&prepared.input)?;
        self.check_key(&prepared.output)?;
        let graph = graph
            .public_output(format!("{prefix}.lhs"), prepared.input.matrices.clone())
            .map_err(|e| RnsBggError::Graph(e.to_string()))?;
        let graph =
            graph.public_output(&format!("{prefix}.output"), (prepared.output.matrix).clone())?;
        Ok((
            graph,
            RnsBggPreparedLinearDescriptor {
                key_storage: [family_storage(&prepared.input.matrices), ProgramValue::Scalar(())],
                multiplier_storage: ProgramValue::Scalar(()),
                prefix: prefix.into(),
                parameters: self.parameter_identity(),
                binding: RnsBggLinearBinding {
                    context: self.context,
                    level: prepared.input.level,
                    lhs: prepared.input.id,
                    output: prepared.output.id,
                    operation: RnsBggLinearOperation::Reduce {
                        range: prepared.input.range.clone(),
                    },
                },
            },
        ))
    }
    pub fn import_prepared_reduction(
        &self,
        descriptor: &RnsBggPreparedLinearDescriptor,
        production: ProductionId,
        expected: &RnsBggLinearBinding,
    ) -> Result<RnsBggPreparedReduction, RnsBggError> {
        self.check_linear_descriptor(descriptor, expected)?;
        if !matches!(expected.operation, RnsBggLinearOperation::Reduce { .. }) {
            return Err(RnsBggError::Context);
        }
        Ok(RnsBggPreparedReduction {
            input: self.import_linear_key_family(descriptor, &production, "lhs", expected.lhs)?,
            output: self.import_linear_key(descriptor, &production, "output", expected.output)?,
        })
    }
    /// Persist the public key and private -p*mu*r contribution separately.
    /// This graph needs no online mask secret and exports neither secret key.
    pub fn export_input_material(
        &self,
        key: &RnsBggPublicKey,
        material: &RnsBggInputMaterial,
        mut graph: DslContext,
        prefix: &str,
    ) -> Result<(DslContext, RnsBggInputMaterialDescriptor), RnsBggError> {
        self.check_key(key)?;
        if material.context != self.context ||
            material.level != key.level ||
            material.key_id != key.id
        {
            return Err(RnsBggError::Context);
        }
        graph = graph.public_output(&format!("{prefix}.key"), (key.matrix).clone())?;
        for column in 0..2 {
            graph = graph
                .private_output(
                    format!("{prefix}.payload.{column}"),
                    material.row.entry(0, column).clone(),
                )
                .map_err(|e| RnsBggError::Graph(e.to_string()))?;
        }
        Ok((
            graph,
            RnsBggInputMaterialDescriptor {
                prefix: prefix.into(),
                scheme: self.artifact_binding(),
                level: key.level,
                key_id: key.id,
            },
        ))
    }
    /// Restore the input pair with private artifact ports for the contribution.
    /// The caller selects the descriptor/production for the intended payload;
    /// the runtime manifest enforces the declared confidentiality and shapes.
    pub fn import_input_material(
        &self,
        descriptor: &RnsBggInputMaterialDescriptor,
        production: ProductionId,
        key_id: [u8; 32],
        level: usize,
    ) -> Result<(RnsBggPublicKey, RnsBggInputMaterial), RnsBggError> {
        if descriptor.scheme != self.artifact_binding() ||
            descriptor.key_id != key_id ||
            descriptor.level != level
        {
            return Err(RnsBggError::Context);
        }
        let ring = self.ring(level)?;
        let columns = self.parameters.component_columns(level)?;
        let key = RnsBggPublicKey {
            matrix: import_blocks(
                &ring,
                &production,
                &format!("{}.key", descriptor.prefix),
                columns,
            ),
            level,
            id: key_id,
            context: self.context,
        };
        let row = crate::rns_bgg::RnsBggRow::new([std::array::from_fn(|column| {
            ring.artifact_input(
                production.clone(),
                format!("{}.payload.{column}", descriptor.prefix),
                (1, columns),
                ArtifactConfidentiality::Private,
            )
        })])
        .map_err(|e| RnsBggError::Graph(e.to_string()))?;
        Ok((key, RnsBggInputMaterial { row, level, key_id, context: self.context }))
    }
    pub(super) fn check_prepared_descriptor<S>(
        &self,
        descriptor: &RnsBggPreparedLutDescriptor<S>,
        binding: &RnsBggLutBinding,
        input_id: [u8; 32],
        output_id: [u8; 32],
    ) -> Result<(), RnsBggError> {
        if descriptor.parameters != self.parameter_identity() ||
            descriptor.binding != *binding ||
            binding.context != self.context ||
            descriptor.input_id != input_id ||
            descriptor.output_id != output_id ||
            descriptor.sigmas.is_empty()
        {
            return Err(RnsBggError::Context);
        }
        let width = descriptor.sigmas.len();
        if !width.is_power_of_two() || (binding.table.is_none() && width != 1) {
            return Err(RnsBggError::Lut);
        }
        // Branch indices come from the selected preprocessing descriptor.
        // Its table binding commits the layout, including terminal strides;
        // regenerating a stride-zero set here would reject valid subrings.
        Ok(())
    }
    /// Persist indexed families without enumerating slots on the host. The
    /// existing runtime streams each polynomial family into its artifact store.
    pub fn export_prepared_lut_family(
        &self,
        prepared: &RnsBggPreparedLutFamily,
        mut graph: DslContext,
        prefix: &str,
    ) -> Result<(DslContext, RnsBggPreparedLutFamilyDescriptor), RnsBggError> {
        self.check_public_family(&prepared.input)?;
        self.check_public_family(&prepared.output)?;
        let schema = prepared.prepared.schema();
        graph = graph.public_output(prefix, prepared.prepared.clone())?;
        let element = prepared.prepared.schema().element;
        Ok((
            graph,
            RnsBggPreparedLutFamilyDescriptor {
                prepared: RnsBggPreparedLutDescriptor {
                    prefix: prefix.into(),
                    parameters: self.parameter_identity(),
                    binding: element.binding,
                    input_id: prepared.input.id,
                    output_id: prepared.output.id,
                    sigmas: element.branches.iter().map(|branch| branch.sigma).collect(),
                    schema,
                },
                range: prepared.input.range.clone(),
            },
        ))
    }
    /// Restore the structural family views. Range and setup contracts are
    /// checked once here, while slot payloads remain runtime-managed artifacts.
    pub fn import_prepared_lut_family(
        &self,
        descriptor: &RnsBggPreparedLutFamilyDescriptor,
        production: ProductionId,
        binding: &RnsBggLutBinding,
        input_id: [u8; 32],
        output_id: [u8; 32],
        range: &FamilyRange,
    ) -> Result<RnsBggPreparedLutFamily, RnsBggError> {
        if binding.table.is_some() && binding.rhs.is_some() {
            return Err(RnsBggError::Lut);
        }
        let data = &descriptor.prepared;
        self.check_prepared_descriptor(data, binding, input_id, output_id)?;
        if descriptor.range != *range {
            return Err(RnsBggError::Context);
        }
        let prepared = data.schema.artifact_input(
            production,
            &data.prefix,
            ArtifactConfidentiality::Public,
        )?;
        let input = RnsBggPublicKeyFamily {
            matrices: ProgramValue::Family(prepared.field(|value| value.input.matrix)?),
            level: binding.level,
            id: input_id,
            context: self.context,
            range: range.clone(),
        };
        let output = RnsBggPublicKeyFamily {
            matrices: ProgramValue::Family(prepared.field(|value| value.output.matrix)?),
            level: binding.level + 1,
            id: output_id,
            context: self.context,
            range: range.clone(),
        };
        Ok(RnsBggPreparedLutFamily { input, output, prepared })
    }
    /// Export existing public helper values. Low-base C/b remain low-base;
    /// no dense concatenation, coefficient extraction, or extra lift is used.
    pub fn export_prepared_lut(
        &self,
        prepared: &RnsBggPreparedLut,
        mut graph: DslContext,
        prefix: &str,
    ) -> Result<(DslContext, RnsBggPreparedLutDescriptor), RnsBggError> {
        self.check_key(&prepared.input)?;
        self.check_key(&prepared.output)?;
        graph = graph.public_output(prefix, prepared.clone())?;
        Ok((
            graph,
            RnsBggPreparedLutDescriptor {
                prefix: prefix.into(),
                parameters: self.parameter_identity(),
                binding: prepared.binding.clone(),
                input_id: prepared.input.id,
                output_id: prepared.output.id,
                sigmas: prepared.branches.iter().map(|branch| branch.sigma).collect(),
                schema: prepared.schema(),
            },
        ))
    }
    /// Expected identities come from the setup contract. Freeze the resulting
    /// graph with the runtime production manifest before executing it. The
    /// imported online graph has no public-key preprocessing multiplications.
    pub fn import_prepared_lut(
        &self,
        descriptor: &RnsBggPreparedLutDescriptor,
        production: ProductionId,
        binding: &RnsBggLutBinding,
        input_id: [u8; 32],
        output_id: [u8; 32],
    ) -> Result<RnsBggPreparedLut, RnsBggError> {
        if binding.table.is_some() && binding.rhs.is_some() {
            return Err(RnsBggError::Lut);
        }
        self.check_prepared_descriptor(descriptor, binding, input_id, output_id)?;
        Ok(descriptor.schema.artifact_input(
            production,
            &descriptor.prefix,
            ArtifactConfidentiality::Public,
        )?)
    }
    pub fn export_prepared_readout(
        &self,
        prepared: &RnsBggPreparedReadout,
        graph: DslContext,
        prefix: &str,
    ) -> Result<(DslContext, RnsBggPreparedReadoutDescriptor), RnsBggError> {
        let (graph, terminal) = match &prepared.terminal {
            RnsBggReadoutTerminal::Scalar { terminal, offset } => {
                let (graph, terminal) =
                    self.export_prepared_terminal(terminal, graph, &format!("{prefix}.terminal"))?;
                let graph = graph.public_output(
                    &format!("{prefix}.output"),
                    (prepared.public.values.scalar().ok_or(RnsBggError::Context)?).clone(),
                )?;
                let graph = graph.public_output(format!("{prefix}.offset"), offset.clone())?;
                (graph, ProgramValue::Scalar(terminal))
            }
            RnsBggReadoutTerminal::Family { terminal, offset } => {
                let (mut graph, terminal) = self.export_prepared_terminal_family(
                    terminal,
                    graph,
                    &format!("{prefix}.terminal"),
                )?;
                if let RnsBggReadoutWeights::Indexed { rows, .. } = &prepared.public.weight {
                    graph = graph
                        .public_output(format!("{prefix}.weights"), rows.clone())
                        .map_err(|e| RnsBggError::Graph(e.to_string()))?;
                }
                let graph = match offset {
                    ProgramValue::Scalar(row) => {
                        graph.public_output(format!("{prefix}.offset"), row.clone())?
                    }
                    ProgramValue::Family(rows) => {
                        graph.public_output(format!("{prefix}.offset"), rows.clone())?
                    }
                };
                (graph, ProgramValue::Family(terminal))
            }
        };
        let weight = match &prepared.public.weight {
            RnsBggReadoutWeights::Shared(weight) => RnsBggReadoutWeights::Shared(weight.clone()),
            RnsBggReadoutWeights::Indexed { coefficient_bound, .. } => {
                RnsBggReadoutWeights::Indexed {
                    rows: (),
                    coefficient_bound: coefficient_bound.clone(),
                }
            }
        };
        Ok((
            graph,
            RnsBggPreparedReadoutDescriptor {
                prefix: prefix.into(),
                identity: prepared.identity,
                terminal,
                weight,
            },
        ))
    }
    /// Restore finalized public outputs without repeating public switching.
    /// Expected bindings come from preprocessing; no online secret is needed.
    pub fn import_prepared_readout(
        &self,
        descriptor: &RnsBggPreparedReadoutDescriptor,
        production: ProductionId,
        readout: &ValueReadout,
        binding: &crate::rns_bgg::RnsBggLutBinding,
        input_id: [u8; 32],
        stage_id: [u8; 32],
    ) -> Result<RnsBggPreparedReadout, RnsBggError> {
        if descriptor.identity != readout.identity() {
            return Err(RnsBggError::Context);
        }
        let data = match &descriptor.terminal {
            ProgramValue::Scalar(data) => data,
            ProgramValue::Family(data) => {
                let terminal = self.import_prepared_terminal_family(
                    data,
                    production.clone(),
                    binding,
                    input_id,
                    stage_id,
                )?;
                let level = terminal.level();
                let ring = self.ring(level)?;
                let columns = terminal.public_matrices().schema().element.matrix_type().columns /
                    IntExpr::constant(2);
                let row_schema = crate::rns_bgg::BlocksType::<1, 2>([[
                    mxx_dsl::MatType(ring.matrix_type((1, columns.clone()))),
                    mxx_dsl::MatType(ring.matrix_type((1, columns.clone()))),
                ]]);
                let weight = match &descriptor.weight {
                    RnsBggReadoutWeights::Shared(weight) => {
                        RnsBggReadoutWeights::Shared(weight.clone())
                    }
                    RnsBggReadoutWeights::Indexed { coefficient_bound, .. } => {
                        let rows = FamilyType {
                            element: row_schema.clone(),
                            shared: vec![false; 2],
                            count: terminal.range().count().clone(),
                        }
                        .artifact_input(
                            production.clone(),
                            &format!("{}.weights", descriptor.prefix),
                            mxx_ir_core::artifact::ArtifactConfidentiality::Public,
                        )?;
                        RnsBggReadoutWeights::Indexed {
                            rows,
                            coefficient_bound: coefficient_bound.clone(),
                        }
                    }
                };
                let offset = if terminal.mode() == RnsBggTerminalFamilyMode::Sum ||
                    matches!(&weight, RnsBggReadoutWeights::Shared(_))
                {
                    ProgramValue::Scalar(crate::rns_bgg::import_blocks(
                        &ring,
                        &production,
                        &format!("{}.offset", descriptor.prefix),
                        columns.clone(),
                    ))
                } else {
                    ProgramValue::Family(
                        FamilyType {
                            element: row_schema,
                            count: terminal.range().count().clone(),
                            shared: vec![false; 2],
                        }
                        .artifact_input(
                            production.clone(),
                            &format!("{}.offset", descriptor.prefix),
                            mxx_ir_core::artifact::ArtifactConfidentiality::Public,
                        )?,
                    )
                };
                let public = RnsBggReadoutPublicOutput {
                    values: ProgramValue::Family((
                        terminal.public_matrices().clone(),
                        terminal.output_range(),
                    )),
                    level,
                    weight,
                };
                return Ok(RnsBggPreparedReadout {
                    identity: descriptor.identity,
                    public,
                    terminal: RnsBggReadoutTerminal::Family { terminal, offset },
                });
            }
        };
        let RnsBggReadoutWeights::Shared(weight) = &descriptor.weight else {
            return Err(RnsBggError::Context);
        };
        let terminal =
            self.import_prepared_terminal(data, production.clone(), binding, input_id, stage_id)?;
        let level = binding.level + 1;
        let ring = self.ring(level)?;
        let public = RnsBggReadoutPublicOutput {
            values: ProgramValue::Scalar(crate::rns_bgg::import_blocks(
                &ring,
                &production,
                &format!("{}.output", descriptor.prefix),
                1,
            )),
            level,
            weight: RnsBggReadoutWeights::Shared(weight.clone()),
        };
        let offset = crate::rns_bgg::import_blocks(
            &ring,
            &production,
            &format!("{}.offset", descriptor.prefix),
            1,
        );
        Ok(RnsBggPreparedReadout {
            identity: descriptor.identity,
            public,
            terminal: RnsBggReadoutTerminal::Scalar { terminal, offset },
        })
    }
    pub fn prepare_terminal(
        &self,
        input: &RnsBggPublicKey,
        material: &RnsBggTerminalMaterial,
    ) -> Result<RnsBggPreparedTerminal, RnsBggError> {
        let (raw_public, branches) =
            self.prepare_lut_raw(input, &material.material, &material.lifted_ciphertexts)?;
        let mut hash = Sha256::new();
        hash.update(b"rns_bgg-terminal-stage");
        hash.update(self.context);
        hash.update(input.id);
        hash.update(material.material.id);
        Ok(RnsBggPreparedTerminal {
            binding: material.material.binding(),
            input: input.clone(),
            id: hash.finalize().into(),
            branches,
            raw_public,
        })
    }
    pub fn finalize_terminal_public(
        &self,
        raw: RnsBggTerminalAccumulator<RnsBggMatrix>,
    ) -> Result<RnsBggTerminalPublicKey, RnsBggError> {
        if raw.context != self.context {
            return Err(RnsBggError::Context);
        }
        let next = BigInt::from(self.parameters.level(raw.level + 1)?.modulus().as_ref().clone());
        Ok(RnsBggTerminalPublicKey {
            matrix: raw.value.block_mod_switch(next),
            level: raw.level + 1,
            context: self.context,
            stages: raw.stages,
        })
    }
    /// Prepare shared or indexed row targets across the active input range.
    /// No member is enumerated on the host; key-dependent work stays structural.
    pub fn prepare_terminal_family(
        &self,
        input: &RnsBggPublicKeyFamily,
        material: &ProgramValue<RnsBggTerminalMaterial, Family<RnsBggTerminalMaterial>>,
        mode: RnsBggTerminalFamilyMode,
        finalize: Option<bool>,
    ) -> Result<RnsBggPreparedTerminalFamily, RnsBggError> {
        self.check_public_family(input)?;
        let finalize = finalize.unwrap_or(true);
        let binding = material.element_schema().material.binding();
        let next = BigInt::from(self.parameters.level(input.level + 1)?.modulus().as_ref().clone());
        let (cache, public) = crate::readout::prepare_terminal_value(
            self,
            &input.clone().into_program()?,
            material,
            mode == RnsBggTerminalFamilyMode::Sum,
            |raw| Ok(if finalize { raw.value.block_mod_switch(next) } else { raw.value }),
        )
        .map_err(|error| match error {
            crate::readout::TerminalError::Scheme(error) => error,
            error => RnsBggError::Graph(error.to_string()),
        })?;
        let ProgramValue::Family(cache) = cache else { return Err(RnsBggError::Context) };
        let public = match public {
            ProgramValue::Scalar(value) => Family::pack(vec![value])?,
            ProgramValue::Family(values) => values,
        };
        let mut hash = Sha256::new();
        hash.update(b"rns_bgg-terminal-stage");
        hash.update(self.context);
        hash.update(input.id);
        hash.update(binding.material_id);
        hash.update([u8::from(finalize)]);
        hash.update([match mode {
            RnsBggTerminalFamilyMode::Independent => 0,
            RnsBggTerminalFamilyMode::Sum => 1,
        }]);
        Ok(RnsBggPreparedTerminalFamily {
            mode,
            finalize,
            input: input.clone(),
            id: hash.finalize().into(),
            cache,
            public,
        })
    }
    pub fn export_prepared_terminal_family(
        &self,
        prepared: &RnsBggPreparedTerminalFamily,
        graph: DslContext,
        prefix: &str,
    ) -> Result<(DslContext, RnsBggPreparedTerminalFamilyDescriptor), RnsBggError> {
        let graph =
            graph.public_output(prefix, (prepared.cache.clone(), prepared.public.clone()))?;
        Ok((
            graph,
            RnsBggPreparedTerminalFamilyDescriptor {
                prefix: prefix.to_owned(),
                parameters: self.parameter_identity(),
                mode: prepared.mode,
                finalize: prepared.finalize,
                id: prepared.id,
                range: prepared.input.range.clone(),
                cache: prepared.cache.schema(),
                public: prepared.public.schema(),
            },
        ))
    }
    /// The descriptor carries actual graph schemas, including captured shared ports.
    pub fn import_prepared_terminal_family(
        &self,
        descriptor: &RnsBggPreparedTerminalFamilyDescriptor,
        production: ProductionId,
        binding: &RnsBggLutBinding,
        input_id: [u8; 32],
        stage_id: [u8; 32],
    ) -> Result<RnsBggPreparedTerminalFamily, RnsBggError> {
        let metadata = &descriptor.cache.element;
        if descriptor.parameters != self.parameter_identity() ||
            metadata.binding != *binding ||
            metadata.input.id != input_id ||
            descriptor.id != stage_id ||
            metadata.input.context != self.context
        {
            return Err(RnsBggError::Context);
        }
        let (cache, public) = (descriptor.cache.clone(), descriptor.public.clone())
            .artifact_input(production, &descriptor.prefix, ArtifactConfidentiality::Public)?;
        let keys = cache.field(|term| (term.input, None))?;
        let mut input = RnsBggPublicKeyFamily::from_program(
            &ProgramValue::Family(keys),
            descriptor.range.count().clone(),
        )?;
        input.range = descriptor.range.clone();
        Ok(RnsBggPreparedTerminalFamily {
            mode: descriptor.mode,
            finalize: descriptor.finalize,
            id: descriptor.id,
            input,
            cache,
            public,
        })
    }
}

impl ExponentProgramLowering for RnsBggPublicEvaluator<'_> {
    fn mul_private(
        &mut self,
        input: &Self::Wire,
        rhs: &Self::Rhs,
        _: Option<&Self::HelperSet>,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, RnsBggError> {
        let (key, _) = input;
        let RnsBggPublicRhs::Fixed(material) = rhs else {
            return Err(RnsBggError::Context);
        };
        let prepared = self.scheme.prepare_mul_private(
            key,
            material,
            output_id(self.scheme.context, b"mul-private", gate, &[key.id, material.id]),
        )?;
        let key = prepared.output_key().clone();
        self.cache.gates.insert(gate, PreparedGate::MulPrivate(prepared));
        Ok((key, None))
    }
    fn mul_public(
        &mut self,
        private: &Self::Wire,
        public: &Self::Wire,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, RnsBggError> {
        let (a, a_plaintext) = private;
        let (b, b_plaintext) = public;
        let multiplier = b_plaintext.clone().ok_or(RnsBggError::Context)?;
        let prepared = self.scheme.prepare_mul_public(
            a,
            b,
            multiplier.clone(),
            output_id(self.scheme.context, b"mul-public", gate, &[a.id, b.id]),
        )?;
        let key = prepared.output_key().clone();
        self.cache.gates.insert(gate, PreparedGate::Linear(prepared));
        let plaintext = a_plaintext.clone().map(|mu| mu * multiplier);
        if let Some(value) = &plaintext {
            self.cache.plaintexts.insert(gate, value.clone());
        }
        Ok((key, plaintext))
    }
    fn add(
        &mut self,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, RnsBggError> {
        let (a, a_plaintext) = lhs;
        let (b, b_plaintext) = rhs;
        let prepared = self.scheme.prepare_add(
            a,
            b,
            output_id(self.scheme.context, b"add", gate, &[a.id, b.id]),
        )?;
        let key = prepared.output_key().clone();
        self.cache.gates.insert(gate, PreparedGate::Linear(prepared));
        let plaintext = a_plaintext.clone().zip(b_plaintext.clone()).map(|(a, b)| a + b);
        if let Some(value) = &plaintext {
            self.cache.plaintexts.insert(gate, value.clone());
        }
        Ok((key, plaintext))
    }
    fn lut(
        &mut self,
        program: &ExponentLutProgram,
        lut: LutId,
        input: &Self::Wire,
        helpers: Option<&Self::HelperSet>,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, RnsBggError> {
        let (input, input_plaintext) = input;
        let helpers = helpers.ok_or(RnsBggError::Context)?;
        let material = helpers;
        let binding = material.binding();
        check_lut_binding(self.scheme, program, lut, input.level, &binding)?;
        let mut identities = vec![input.id, binding.material_id];
        if let Some(table) = binding.table {
            identities.push(table);
        }
        let prepared = self.scheme.prepare_lut(
            input,
            material,
            output_id(self.scheme.context, b"L", gate, &identities),
        )?;
        let key = prepared.output_key().clone();
        let plaintext = input_plaintext
            .as_ref()
            .map(|mu| lut_plaintext(self.scheme, program, lut, key.level, mu))
            .transpose()?;
        self.cache.gates.insert(gate, PreparedGate::Lut(prepared));
        if let Some(value) = &plaintext {
            self.cache.plaintexts.insert(gate, value.clone());
        }
        Ok((key, plaintext))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum RnsBggLinearOperation {
    Add { rhs: [u8; 32], range: Option<FamilyRange> },
    MulPublic { rhs: [u8; 32], range: Option<FamilyRange> },
    Reduce { range: FamilyRange },
}

impl RnsBggLinearOperation {
    pub fn input_range(&self) -> Option<&FamilyRange> {
        match self {
            Self::Add { range, .. } | Self::MulPublic { range, .. } => range.as_ref(),
            Self::Reduce { range } => Some(range),
        }
    }

    pub fn output_range(&self) -> Option<&FamilyRange> {
        match self {
            Self::Add { range, .. } | Self::MulPublic { range, .. } => range.as_ref(),
            Self::Reduce { .. } => None,
        }
    }
}

impl<M: GraphValue> LinearOperation<M> {
    fn import(
        operation: &RnsBggLinearOperation,
        schema: &M::Schema,
        production: ProductionId,
        prefix: &str,
    ) -> Result<(Self, [u8; 32]), RnsBggError> {
        match operation {
            RnsBggLinearOperation::Add { rhs, .. } => Ok((Self::Add, *rhs)),
            RnsBggLinearOperation::MulPublic { rhs, .. } => Ok((
                Self::MulPublic(schema.artifact_input(
                    production,
                    format!("{prefix}.multiplier"),
                    ArtifactConfidentiality::Public,
                )?),
                *rhs,
            )),
            RnsBggLinearOperation::Reduce { .. } => Err(RnsBggError::Context),
        }
    }

    fn export(
        &self,
        graph: DslContext,
        prefix: &str,
        rhs: [u8; 32],
        range: Option<FamilyRange>,
    ) -> Result<(DslContext, RnsBggLinearOperation), RnsBggError> {
        match self {
            Self::Add => Ok((graph, RnsBggLinearOperation::Add { rhs, range })),
            Self::MulPublic(multiplier) => Ok((
                graph.public_output(format!("{prefix}.multiplier"), multiplier.clone())?,
                RnsBggLinearOperation::MulPublic { rhs, range },
            )),
        }
    }
}

/// Expected metadata supplied by setup. The output identity binds the public
/// multiplier, if any, by the same caller contract as public preparation.
/// A range denotes active family members; reduction has a scalar output.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RnsBggLinearBinding {
    pub context: [u8; 32],
    pub level: usize,
    pub lhs: [u8; 32],
    pub output: [u8; 32],
    pub operation: RnsBggLinearOperation,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct RnsBggPreparedLinearDescriptor {
    key_storage: [ProgramValue<(), Vec<bool>>; 2],
    multiplier_storage: ProgramValue<(), Vec<bool>>,
    prefix: String,
    parameters: [u8; 32],
    binding: RnsBggLinearBinding,
}

impl RnsBggPreparedLinearDescriptor {
    pub fn binding(&self) -> &RnsBggLinearBinding {
        &self.binding
    }
}

/// Metadata for a public input key and its private payload contribution.
/// Keep the descriptor bound to the selected production and payload at setup;
/// key identity alone does not distinguish different payloads for that key.
#[derive(Clone, Serialize, Deserialize)]
pub struct RnsBggInputMaterialDescriptor {
    prefix: String,
    scheme: crate::rns_bgg::RnsBggArtifactBinding,
    level: usize,
    key_id: [u8; 32],
}

/// Store this public descriptor alongside the runtime production manifest.
/// Import checks expected setup identities once at graph construction. The
/// runtime manifest supplies artifact shapes/confidentiality and production
/// binding; this descriptor is not a separate authentication protocol.
#[derive(Clone, Serialize, Deserialize)]
pub struct RnsBggPreparedLutDescriptor<S = <RnsBggPreparedLut as GraphValue>::Schema> {
    pub(super) prefix: String,
    pub(super) parameters: [u8; 32],
    pub(super) binding: RnsBggLutBinding,
    pub(super) input_id: [u8; 32],
    pub(super) output_id: [u8; 32],
    pub(super) sigmas: Vec<usize>,
    schema: S,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct RnsBggPreparedLutFamilyDescriptor {
    pub(super) prepared:
        RnsBggPreparedLutDescriptor<<Family<RnsBggPreparedLut> as GraphValue>::Schema>,
    pub(super) range: FamilyRange,
}

#[cfg(test)]
pub(crate) mod terminal_artifact_tests {
    use super::*;
    use crate::rns_bgg::{RnsBggNoiseBound, RnsBggParameters, RnsBggPlanRequest, RnsBggRow};
    use mxx_dsl::DslContext;
    use mxx_ir_core::{ParamEnv, artifact::ArtifactConfidentiality};
    use mxx_primitives::{
        element::PolyElem,
        poly::{Poly, PolyParams},
    };
    use mxx_runtime::{
        Backend, RuntimeValue,
        artifact::{ArtifactStore, FileArtifactStore},
        backend::poly::cpu_backend,
        executor::execute,
        transcript::SamplingMode,
    };
    use num_bigint::{BigInt, BigUint};
    use std::collections::BTreeMap;

    #[test]
    fn test_rns_bgg_fixed_product_schema_preserves_family_bindings() {
        let ring = mxx_dsl::Ring::new(257, 8);
        let matrix = RnsBggMatrix::new(std::array::from_fn(|row| {
            std::array::from_fn(|column| ring.input(format!("key-{row}-{column}"), (1, 1)))
        }))
        .unwrap();
        let context = rand::random();
        let key = RnsBggPublicKeyFamily {
            matrices: ProgramValue::Scalar(matrix.clone()),
            level: 0,
            id: rand::random(),
            context,
            range: FamilyRange::bounded(3, 2, 7).unwrap(),
        };
        let mut output = key.clone();
        output.matrices = ProgramValue::Family(Family::pack(vec![matrix.clone(), matrix]).unwrap());
        output.level = 1;
        output.id = rand::random();
        let prepared = RnsBggPreparedPrivateMul {
            rhs: RnsBggPrivateRhs {
                ciphertext: output.matrices.clone(),
                lifted_ciphertext: key.matrices.clone(),
                level: 0,
                rhs_id: rand::random(),
                id: rand::random(),
                context,
            },
            input: key,
            output,
        };
        let schema = prepared.schema();
        let encoded = serde_json::to_vec(&schema).unwrap();
        let restored = serde_json::from_slice(&encoded).unwrap();
        assert!(schema == restored);
        let producer = DslContext::new("fixed-product-schema-producer")
            .public_output("fixed", prepared)
            .unwrap()
            .build()
            .unwrap();
        let producer = producer.validate(&ParamEnv::default()).unwrap();
        let production = ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash(rand::random()),
            execution_nonce: rand::random(),
        };
        let manifest =
            mxx_ir_core::artifact::export_validated_manifest(production.clone(), &producer)
                .unwrap();
        let imported = restored
            .artifact_input(production.clone(), "fixed", ArtifactConfidentiality::Public)
            .unwrap();
        assert!(imported.schema() == schema);
        let consumer = DslContext::new("fixed-product-schema-consumer")
            .public_output("reopened", imported)
            .unwrap()
            .build()
            .unwrap();
        consumer
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production, manifest)]),
            )
            .unwrap();
    }

    #[test]
    fn test_rns_bgg_terminal_artifacts_reused_with_independent_online_secrets() {
        check_terminal_artifacts(
            |parameters| cpu_backend(parameters.levels().iter().cloned()),
            |matrix, name, q, bound| {
                for polynomial in matrix.block_entries(0..1, 0..1).into_iter().flatten() {
                    for coefficient in polynomial.coeffs() {
                        let absolute = coefficient.value().clone().min(q - coefficient.value());
                        assert!(&absolute <= bound, "{name}: {absolute} > {bound}");
                    }
                }
            },
        );
    }

    pub(crate) fn check_terminal_artifacts<B: Backend>(
        make_backend: impl FnOnce(&RnsBggParameters) -> B,
        check_matrix: impl Fn(&B::Matrix, &str, &BigUint, &BigUint),
    ) {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|value| value.parse().unwrap())
            .unwrap_or(16);
        let values = [
            vec![[vec![], vec![0.into(), 5.into()]], [vec![(-3).into()], vec![]]],
            vec![[vec![2.into()], vec![]], [vec![], vec![0.into(), 0.into(), 7.into()]]],
        ];
        let tables =
            values.clone().map(|values| crate::readout::RowLutTable::new(values, None).unwrap());
        let value_bound = tables[0].coefficient_bound() + tables[1].coefficient_bound();
        let (parameters, _) = RnsBggParameters::plan(&RnsBggPlanRequest {
            ring_dimension: n,
            widths: vec![2],
            slots: vec![1],
            encoding_error_sigma: bigdecimal::BigDecimal::from(2u8),
            crt_bits: 30,
            terminal_depth: 2,
            max_crt_depth: 12,
            terminal_value_bound: value_bound.clone(),
            operations_per_drop: 1,
        })
        .unwrap();
        let scheme = RnsBggScheme::new(parameters.clone(), rand::random());
        let high = scheme.ring(0).unwrap();
        let low = scheme.ring(1).unwrap();
        let seed = high.bytes_input("hash-key", 32);
        let payload_secret = scheme.sample_payload_secret().unwrap();
        let key =
            scheme.sample_input_key(0, seed.clone(), b"key".as_slice(), rand::random()).unwrap();
        let bank = scheme
            .sample_mask_bank(
                0,
                crate::layout::ExpLayout {
                    stride_log2: 0,
                    width_log2: (2 as usize).trailing_zeros(),
                },
                seed.clone(),
                b"bank".as_slice(),
                rand::random(),
            )
            .unwrap();
        use crate::readout::PreparedTerminal;
        let materials = tables
            .iter()
            .enumerate()
            .map(|(stage, table)| {
                scheme
                    .sample_terminal_material(
                        &payload_secret,
                        Some(table),
                        None,
                        None,
                        &bank,
                        seed.clone(),
                        format!("terminal-{stage}").as_bytes(),
                        rand::random(),
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let public_input = (key.clone(), None);
        let prepared = PreparedTerminal::new(
            &scheme,
            &materials.iter().map(|material| (&public_input, material)).collect::<Vec<_>>(),
        )
        .unwrap();
        let expected = prepared
            .prepared_terms()
            .iter()
            .map(|term| (term.binding().clone(), key.id, term.id()))
            .collect::<Vec<_>>();
        let (mut producer, descriptor) =
            prepared.export(DslContext::new("rns_bgg-terminal-preprocessing"), "terminal").unwrap();
        let descriptor: crate::readout::PreparedTerminalDescriptor<
            <RnsBggScheme as ExponentLUTTerminalScheme>::TerminalDescriptor,
        > = serde_json::from_slice(&serde_json::to_vec(&descriptor).unwrap()).unwrap();
        let mut inputs = Vec::new();
        for x in 0..2 {
            let payload = high.polynomial((0..=x).map(|i| BigInt::from(u8::from(i == x)).into()));
            let material = scheme.prepare_input(&payload_secret, &key, payload).unwrap();
            let (graph, descriptor) = scheme
                .export_input_material(&key, &material, producer, &format!("input-{x}"))
                .unwrap();
            producer = graph;
            inputs.push(descriptor);
        }
        let producer = producer.build().unwrap().validate(&ParamEnv::default()).unwrap();
        std::fs::create_dir_all("test_data/rns_bgg_terminal_artifacts").unwrap();
        let directory = tempfile::tempdir_in("test_data/rns_bgg_terminal_artifacts").unwrap();
        let mut store = FileArtifactStore::new(directory.path()).unwrap();
        let mut backend = make_backend(&parameters);
        let produced = execute(
            &producer,
            &mut backend,
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut store,
            SamplingMode::Fresh,
        )
        .unwrap();
        let production = produced.production_id.unwrap();
        drop(store);
        let mut store = FileArtifactStore::open(directory.path()).unwrap();
        let manifest = store.load_manifest(&production).unwrap();
        for (name, artifact) in &manifest.artifacts {
            assert!(!name.contains(".d."), "online helper persisted: {name}");
            let expected = if name.contains(".payload.") {
                ArtifactConfidentiality::Private
            } else {
                ArtifactConfidentiality::Public
            };
            assert_eq!(artifact.confidentiality, expected, "{name}");
        }
        let prepared =
            PreparedTerminal::import(&scheme, &descriptor, production.clone(), &expected).unwrap();
        let mut wrong = expected.clone();
        wrong[0].2[0] ^= 1;
        assert!(
            PreparedTerminal::import(&scheme, &descriptor, production.clone(), &wrong).is_err()
        );
        wrong = expected.clone();
        wrong[0].0.mask_bank[0] ^= 1;
        assert!(
            PreparedTerminal::import(&scheme, &descriptor, production.clone(), &wrong).is_err()
        );
        wrong = expected.clone();
        wrong.reverse();
        assert!(
            PreparedTerminal::import(&scheme, &descriptor, production.clone(), &wrong).is_err()
        );
        assert!(PreparedTerminal::import(&scheme, &descriptor, production.clone(), &[]).is_err());
        let public = prepared.public();
        // Import each artifact port once, then share its graph value across s.
        let inputs = inputs
            .iter()
            .map(|descriptor| {
                scheme.import_input_material(descriptor, production.clone(), key.id, 0).unwrap()
            })
            .collect::<Vec<_>>();
        let bound = RnsBggNoiseBound {
            error: parameters.encoding_error_bound.clone(),
            payload_l1: 1u8.into(),
        };
        let gain = BigUint::from(1u8);
        let noise = parameters
            .terminal_noise_bound(0, [(&bound, 2, &gain), (&bound, 2, &gain)], value_bound)
            .unwrap();
        assert!(parameters.terminal_noise_fits(1, &noise).unwrap());
        let q = parameters.level(1).unwrap().modulus();
        let modulus = BigInt::from(q.as_ref().clone());
        let mut online = DslContext::new("rns_bgg-terminal-independent-online-secrets");
        // No s exists before the preprocessing production is finished/reopened.
        for run in 0..2 {
            let secret = scheme.sample_mask_secret().unwrap();
            let helpers = scheme
                .prepare_online_masks(&secret, &prepared.prepared_terms()[0].mask_bank())
                .unwrap();
            let s = secret.row.clone().reduce_modulus(modulus.clone());
            for (x, (key, material)) in inputs.iter().enumerate() {
                let encoding = scheme.encode(&secret, key, material).unwrap();
                let input = (encoding, None);
                let result = prepared.evaluate(&[(&input, &helpers), (&input, &helpers)]).unwrap();
                let row = |coefficients: &[Vec<BigInt>; 2]| {
                    RnsBggRow::new([coefficients.each_ref().map(|p| {
                        if p.is_empty() {
                            low.zero((1, 1))
                        } else {
                            low.polynomial(p.iter().cloned().map(mxx_ir_core::IntExpr::constant))
                        }
                    })])
                    .unwrap()
                };
                let expected = row(&values[0][x]) + row(&values[1][x]);
                let error = result.row().clone() - s.clone() * public.matrix().clone() - expected;
                online = online.output(format!("error-{run}-{x}"), error).unwrap();
            }
        }
        let online = online
            .build()
            .unwrap()
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production, manifest)]),
            )
            .unwrap();
        assert!(
            online.source.scopes().values().flat_map(|scope| scope.nodes()).all(|node| {
                !matches!(node.kind(), mxx_ir_core::node::NodeKind::GadgetDecompose { .. }) &&
                    !(matches!(node.kind(), mxx_ir_core::node::NodeKind::CenteredExtend { .. }) &&
                        node.output_types()
                            .iter()
                            .any(|ty| matches!(ty, WireType::Preimage { .. })))
            }),
            "online terminal evaluation must reuse persisted bounded ciphertext and intermediate actions"
        );
        let switches = online
            .source
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .filter(|node| {
                matches!(node.kind(), mxx_ir_core::node::NodeKind::BlockModSwitch { .. })
            })
            .count();
        assert_eq!(
            switches,
            2 * 2 * 2,
            "one final row switch per evaluation; all public switches were persisted by preprocessing"
        );
        let results =
            execute(&online, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .unwrap();
        for (name, value) in results.outputs {
            let RuntimeValue::Matrix(matrix) = value else { panic!("polynomial") };
            check_matrix(&matrix, &name, q.as_ref(), &noise.error);
        }
    }
}

#[cfg(test)]
pub(crate) mod program_artifacts_tests {
    use super::*;
    use crate::{
        layout::ExpLayout,
        program::{
            AttributeKnowledge, ExponentLutProgramBuilder, FamilyProgram, FamilyRhsSource,
            FamilyWireSource, Gate, LutTable, ProgramBindings, ProgramFamilyId, ProgramWireType,
            RhsId,
        },
        rns_bgg::{
            RnsBggEncoding, RnsBggParameters, RnsBggPlanRequest, RnsBggRhsPlaintextFamily,
            RnsBggRow,
        },
    };
    use mxx_ir_core::ParamEnv;
    use mxx_runtime::{
        RuntimeValue,
        artifact::{ArtifactStore, FileArtifactStore},
        backend::poly::cpu_backend,
        execute,
        transcript::SamplingMode,
    };

    fn observed_outputs(
        mut graph: DslContext,
        program: &PreparedProgram<RnsBggScheme>,
        online: &BTreeMap<WireId, (RnsBggEncoding, Option<Mat>)>,
    ) -> DslContext {
        for (index, (wire, value)) in online.iter().enumerate() {
            let (value, value_plaintext) = value;
            graph = graph.output(format!("wire-{index}"), value.row.clone()).unwrap();
            if let Some(plaintext) = &value_plaintext {
                graph = graph.output(format!("plain-{index}"), plaintext.clone()).unwrap();
            }
            let (public, public_plaintext) = &program.public_values()[wire];
            graph = graph.output(format!("key-{index}"), public.matrix.clone()).unwrap();
            if let Some(plaintext) = &public_plaintext {
                graph = graph.output(format!("public-plain-{index}"), plaintext.clone()).unwrap();
            }
        }
        graph
    }

    #[test]
    fn test_rns_bgg_program_artifacts_restore_all_gate_kinds_and_public_values() {
        check_program_artifacts(
            |parameters| cpu_backend(parameters.levels().iter().cloned()),
            |actual, expected, name| assert_eq!(actual, expected, "{name}"),
        );
    }

    pub(crate) fn check_program_artifacts<B: mxx_runtime::Backend>(
        make_backend: impl FnOnce(&RnsBggParameters) -> B,
        check_matrix: impl Fn(&B::Matrix, &B::Matrix, &str),
    ) {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|v| v.parse().unwrap())
            .unwrap_or(16);
        let (parameters, _) = RnsBggParameters::plan(&RnsBggPlanRequest {
            ring_dimension: n,
            widths: vec![2],
            slots: vec![4],
            encoding_error_sigma: bigdecimal::BigDecimal::from(2u8),
            crt_bits: 30,
            terminal_depth: 1,
            max_crt_depth: 8,
            terminal_value_bound: 16u8.into(),
            operations_per_drop: 1,
        })
        .unwrap();
        let scheme = RnsBggScheme::new(parameters.clone(), rand::random());
        let high = scheme.ring(0).unwrap();
        let low = scheme.ring(1).unwrap();
        let seed = high.bytes_input("hash-key", 32);
        let secret = scheme.test_secrets().unwrap();
        let private_key = scheme
            .sample_input_key(0, seed.clone(), b"private".as_slice(), rand::random())
            .unwrap();
        let public_key =
            scheme.sample_input_key(0, seed.clone(), b"public".as_slice(), rand::random()).unwrap();
        let member_key =
            scheme.sample_input_key(1, seed.clone(), b"member".as_slice(), rand::random()).unwrap();
        let public_plaintext = high.polynomial([1.into()]);
        let member_plaintext = low.polynomial([2.into(), (-1).into()]);
        let private = scheme
            .encode(
                &secret.mask,
                &private_key,
                &scheme
                    .prepare_input(
                        &secret.payload,
                        &private_key,
                        high.polynomial([0.into(), 1.into()]),
                    )
                    .unwrap(),
            )
            .unwrap();
        let public = scheme
            .encode(
                &secret.mask,
                &public_key,
                &scheme
                    .prepare_input(&secret.payload, &public_key, public_plaintext.clone())
                    .unwrap(),
            )
            .unwrap();
        let member = scheme
            .encode(
                &secret.mask,
                &member_key,
                &scheme
                    .prepare_input(&secret.payload, &member_key, member_plaintext.clone())
                    .unwrap(),
            )
            .unwrap();
        let bank = scheme
            .sample_mask_bank(
                0,
                crate::layout::ExpLayout {
                    stride_log2: 0,
                    width_log2: (2 as usize).trailing_zeros(),
                },
                seed.clone(),
                b"bank".as_slice(),
                rand::random(),
            )
            .unwrap();
        let table = LutTable::unary(2, 2, vec![1, 0]).unwrap();
        let scalar_material = scheme
            .sample_lut_material(
                &secret.payload,
                Some(&table),
                None,
                &bank,
                seed.clone(),
                b"scalar".as_slice(),
                rand::random(),
            )
            .unwrap();
        let rhs_id = rand::random();
        let family_material = scheme
            .sample_private_rhs_family(
                &secret.payload,
                0,
                &RnsBggRhsPlaintextFamily {
                    payloads: mxx_dsl::parallel(4, |_| Ok(low.polynomial([1.into()]))).unwrap(),
                    id: rhs_id,
                },
                seed.clone(),
                b"family".as_slice(),
                rand::random(),
            )
            .unwrap();
        let table_family_material = scheme
            .sample_lut_material_family(
                &secret.payload,
                Some(&table),
                None,
                &bank,
                4.into(),
                seed.clone(),
                b"table-family".as_slice(),
                rand::random(),
            )
            .unwrap();
        let private_material = scheme
            .sample_private_rhs(
                &secret.payload,
                0,
                crate::rns_bgg::RnsBggRhsPlaintext {
                    payload: low.polynomial([1.into()]),
                    id: rhs_id,
                },
                seed.clone(),
                b"scalar-private".as_slice(),
                rand::random(),
            )
            .unwrap();

        let private_material = scheme.prepare_private_rhs(&private_material).unwrap();
        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(ExpLayout { stride_log2: 0, width_log2: 1 }, n as usize).unwrap();
        let private_input = builder
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let public_input = builder
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Public })
            .unwrap();
        let private_wire = builder.input_wire(private_input).unwrap();
        let public_wire = builder.input_wire(public_input).unwrap();
        let added = builder.add(public_wire, public_wire).unwrap();
        let product = builder.mul_public(public_wire, added).unwrap();
        let lut = builder.register_lut(table).unwrap();
        let mapped = builder.lut(private_wire, lut, None).unwrap();
        let rhs_family = builder.rhs_family(2).unwrap();
        let wire_family = builder.wire_family(2).unwrap();
        let w = WireId::from_index;
        let r = RhsId::from_index(0);
        let selected = builder
            .family(
                4,
                FamilyProgram::new(
                    BTreeMap::from([
                        (w(0), FamilyWireSource::Broadcast(private_wire)),
                        (w(1), FamilyWireSource::Member(wire_family)),
                    ]),
                    BTreeMap::from([(r, FamilyRhsSource::Member(rhs_family))]),
                    vec![
                        Gate::MulPrivate { lhs: w(0), rhs: r, output: w(2) },
                        Gate::Add { lhs: w(1), rhs: w(1), output: w(3) },
                        Gate::MulPublic { private: w(2), public: w(3), output: w(4) },
                        Gate::Add { lhs: w(4), rhs: w(4), output: w(5) },
                    ],
                    w(5),
                )
                .unwrap(),
            )
            .unwrap();
        let public_sum = builder
            .family(
                4,
                FamilyProgram::new(
                    BTreeMap::from([(w(0), FamilyWireSource::Member(wire_family))]),
                    BTreeMap::new(),
                    vec![Gate::Add { lhs: w(0), rhs: w(0), output: w(1) }],
                    w(1),
                )
                .unwrap(),
            )
            .unwrap();
        let combined = builder.add(mapped, selected).unwrap();
        let mapped_family = builder
            .family(
                4,
                FamilyProgram::new(
                    BTreeMap::from([(w(0), FamilyWireSource::Broadcast(private_wire))]),
                    BTreeMap::new(),
                    vec![Gate::Lut { input: w(0), lut_id: lut, output: w(1) }],
                    w(1),
                )
                .unwrap(),
            )
            .unwrap();
        let scalar_rhs = builder.rhs_input(rhs_family, 2, 2).unwrap();
        let private_product = builder.private_mul(private_wire, scalar_rhs).unwrap();
        let shared_product = builder
            .family(
                4,
                FamilyProgram::new(
                    BTreeMap::from([(w(0), FamilyWireSource::Broadcast(private_wire))]),
                    BTreeMap::from([(r, FamilyRhsSource::Broadcast(scalar_rhs))]),
                    vec![Gate::MulPrivate { lhs: w(0), rhs: r, output: w(1) }],
                    w(1),
                )
                .unwrap(),
            )
            .unwrap();
        let doubled_private_product = builder.add(private_product, private_product).unwrap();
        for output in [
            product,
            combined,
            public_sum,
            mapped_family,
            private_product,
            shared_product,
            doubled_private_product,
        ] {
            builder.output(output).unwrap();
        }
        let frozen = builder.build().unwrap();
        let range = FamilyRange::bounded(1, 2, 4).unwrap();
        let family_material = scheme.prepare_private_rhs_family(&family_material, &range).unwrap();
        let mut ranges = ProgramFamilyBindings::new();
        for index in 0..4 {
            ranges.bind(ProgramFamilyId::from_index(index), range.clone());
        }
        let scalar_gate = ProgramGateInstance::Gate { step: 2 };
        let family_gate = ProgramGateInstance::FamilyGate {
            step: 3,
            gate: 0,
            family: match &frozen.steps()[3] {
                crate::program::ProgramStep::Family { id, .. } => *id,
                _ => unreachable!(),
            },
        };
        let table_family_gate = ProgramGateInstance::FamilyGate {
            step: 6,
            gate: 0,
            family: match &frozen.steps()[6] {
                crate::program::ProgramStep::Family { id, .. } => *id,
                _ => unreachable!(),
            },
        };
        let private_gate = ProgramGateInstance::Gate { step: 7 };
        let shared_gate = ProgramGateInstance::FamilyGate {
            step: 8,
            gate: 0,
            family: match &frozen.steps()[8] {
                crate::program::ProgramStep::Family { id, .. } => *id,
                _ => unreachable!(),
            },
        };
        let scalar_rhs_bindings =
            BTreeMap::from([(scalar_rhs, RnsBggPublicRhs::Fixed(private_material))]);
        let rhs_bindings = BTreeMap::from([(
            rhs_family,
            (
                mxx_dsl::parallel(2, |index| {
                    Ok(RnsBggPublicRhs::Fixed(RnsBggPrivateRhs {
                        ciphertext: family_material.ciphertext.at(&index),
                        lifted_ciphertext: family_material.lifted_ciphertext.at(&index),
                        rhs_id: family_material.rhs_id,
                        id: family_material.id,
                        context: family_material.context,
                        level: family_material.level,
                    }))
                })
                .unwrap(),
                range.clone(),
            ),
        )]);
        let online_scalar_rhs_bindings =
            BTreeMap::from([(scalar_rhs, RnsBggPublicRhs::Identity(rhs_id))]);
        let online_rhs_bindings = BTreeMap::from([(
            rhs_family,
            (
                mxx_dsl::parallel(2, |_| Ok(RnsBggPublicRhs::Identity(rhs_id))).unwrap(),
                range.clone(),
            ),
        )]);
        let public_helpers = BTreeMap::from([
            (
                table_family_gate,
                ProgramValue::Family((
                    mxx_dsl::parallel(range.count().clone(), |index| {
                        Ok(table_family_material
                            .at(index.add(Int::evaluate(range.start().clone()))))
                    })
                    .unwrap(),
                    range.clone(),
                )),
            ),
            (scalar_gate, ProgramValue::Scalar(scalar_material.clone())),
        ]);
        let public_inputs = BTreeMap::from([
            (private_input, (private_key.clone(), None)),
            (public_input, (public_key.clone(), Some(public_plaintext.clone()))),
        ]);
        let public_members = BTreeMap::from([(
            wire_family,
            (
                mxx_dsl::parallel(2, |_| Ok((member_key.clone(), Some(member_plaintext.clone()))))
                    .unwrap(),
                range.clone(),
            ),
        )]);
        let prepared = PreparedProgram::new(
            scheme.clone(),
            frozen.clone(),
            &ProgramBindings::new(
                &public_inputs,
                &scalar_rhs_bindings,
                &rhs_bindings,
                &public_members,
                &BTreeMap::new(),
                Some(&public_helpers),
            ),
            ranges.clone(),
            None,
        )
        .unwrap();
        let ProgramGateInstance::FamilyGate { family, .. } = shared_gate else { unreachable!() };
        let shared_cache = prepared.family_caches[&family].at(0);
        let PreparedGate::MulPrivate(shared) = &shared_cache.gates[&shared_gate] else {
            panic!("shared private product")
        };
        let other_cache = prepared.family_caches[&family].at(1);
        let PreparedGate::MulPrivate(other) = &other_cache.gates[&shared_gate] else {
            panic!("other shared product")
        };
        assert_eq!(shared.rhs.ciphertext.flatten(), other.rhs.ciphertext.flatten());
        assert_eq!(shared.rhs.lifted_ciphertext.flatten(), other.rhs.lifted_ciphertext.flatten());
        let lift_graph = DslContext::new("rns_bgg-shared-rhs-lift")
            .public_output("lift", shared.rhs.lifted_ciphertext.clone())
            .unwrap()
            .build()
            .unwrap()
            .graph;
        let count_lifts = |scope: &mxx_ir_core::graph::GraphScope| {
            scope
                .nodes()
                .iter()
                .filter(|node| {
                    matches!(node.kind(), mxx_ir_core::node::NodeKind::CenteredExtend { .. })
                })
                .count()
        };
        assert_eq!(
            count_lifts(lift_graph.root_scope()),
            4,
            "one lift per block outside member loops"
        );
        assert_eq!(lift_graph.scopes().values().map(count_lifts).sum::<usize>(), 4);
        let online_inputs = BTreeMap::from([
            (private_input, (private.clone(), None)),
            (public_input, (public.clone(), Some(public_plaintext.clone()))),
        ]);
        let online_members = BTreeMap::from([(
            wire_family,
            (
                mxx_dsl::parallel(2, |_| Ok((member.clone(), Some(member_plaintext.clone()))))
                    .unwrap(),
                range.clone(),
            ),
        )]);
        let mut online_helpers = prepared.prepare_online_helpers(&secret.mask).unwrap();
        assert!(!online_helpers.contains_key(&private_gate));
        assert!(!online_helpers.contains_key(&family_gate));
        let ProgramValue::Scalar(scalar_helpers) = &online_helpers[&scalar_gate] else {
            panic!("shared masks")
        };
        let online_bank = scalar_helpers.masks.clone();
        assert_eq!(online_bank.rows.len(), 2, "explicit scalar and family LUT helper sigmas");
        let ProgramValue::Scalar(family_helpers) = &online_helpers[&table_family_gate] else {
            panic!("shared family masks")
        };
        assert!(
            std::sync::Arc::ptr_eq(&online_bank.rows, &family_helpers.masks.rows),
            "one shared online mask bank, not one sampler per gate"
        );
        let online = prepared
            .evaluate(&ProgramBindings::new(
                &online_inputs,
                &online_scalar_rhs_bindings,
                &online_rhs_bindings,
                &online_members,
                &BTreeMap::new(),
                Some(&online_helpers),
            ))
            .unwrap();
        let (mut producer, descriptor) = prepared
            .export(
                scheme.artifact_binding(),
                DslContext::new("rns_bgg-program-producer"),
                "program",
            )
            .unwrap();
        let identity = descriptor.identity().unwrap();
        for (name, value) in [("private", &private), ("public", &public), ("member", &member)] {
            for column in 0..2 {
                producer = producer
                    .private_output(
                        format!("input-{name}-{column}"),
                        value.row.entry(0, column).clone(),
                    )
                    .unwrap();
            }
        }
        let producer = crate::rns_bgg::encoding::encoding_tests::export_online_masks(
            &online_bank,
            producer,
            "online-mask",
        );
        let producer = observed_outputs(producer, &prepared, &online)
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let mut backend = make_backend(&parameters);
        let directory = tempfile::tempdir().unwrap();
        let mut store = FileArtifactStore::new(directory.path()).unwrap();
        let mut produced = execute(
            &producer,
            &mut backend,
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut store,
            SamplingMode::Fresh,
        )
        .unwrap();
        let wire_name = |wire, column| {
            format!("wire-{}.{}", online.keys().position(|value| *value == wire).unwrap(), column)
        };
        for column in 0..2 {
            let RuntimeValue::Matrix(shared_result) =
                &produced.outputs[&wire_name(shared_product, column)]
            else {
                panic!("shared product component")
            };
            let RuntimeValue::Matrix(expected_result) =
                &produced.outputs[&wire_name(doubled_private_product, column)]
            else {
                panic!("doubled scalar product component")
            };
            check_matrix(
                shared_result,
                expected_result,
                "shared RHS family matches scalar products",
            );
        }
        let production = produced.production_id.clone().unwrap();
        let online_bank = crate::rns_bgg::encoding::encoding_tests::import_online_masks(
            &scheme,
            &online_bank,
            &production,
            "online-mask",
        );
        for helpers in online_helpers.values_mut() {
            let ProgramValue::Scalar(helpers) = helpers else { panic!("shared mask bank") };
            helpers.masks = online_bank.clone();
        }
        drop(store);
        let mut store = FileArtifactStore::new(directory.path()).unwrap();
        let descriptor: crate::scheme::PreparedProgramDescriptor<
            <RnsBggProgramCache as GraphValue>::Schema,
            <(RnsBggPublicKey, Option<Mat>) as GraphValue>::Schema,
            RnsBggArtifactBinding,
        > = serde_json::from_slice(&serde_json::to_vec(&descriptor).unwrap()).unwrap();
        let restored = PreparedProgram::import(
            scheme.artifact_binding(),
            scheme.clone(),
            frozen.clone(),
            ranges.clone(),
            &descriptor,
            production.clone(),
            identity,
        )
        .unwrap();
        let ProgramGateInstance::FamilyGate { family, .. } = shared_gate else { unreachable!() };
        let shared_cache = restored.family_caches[&family].at(0);
        let PreparedGate::MulPrivate(shared) = &shared_cache.gates[&shared_gate] else {
            panic!("restored shared product")
        };
        let other_cache = restored.family_caches[&family].at(1);
        let PreparedGate::MulPrivate(other) = &other_cache.gates[&shared_gate] else {
            panic!("other shared product")
        };
        assert_eq!(shared.rhs.ciphertext.flatten(), other.rhs.ciphertext.flatten());
        assert_eq!(shared.rhs.lifted_ciphertext.flatten(), other.rhs.lifted_ciphertext.flatten());
        let import_input = |name: &str, key: &RnsBggPublicKey| {
            let ring = scheme.ring(key.level).unwrap();
            RnsBggEncoding {
                row: RnsBggRow::new([std::array::from_fn(|column| {
                    ring.artifact_input(
                        production.clone(),
                        format!("input-{name}-{column}"),
                        (1, parameters.component_columns(key.level).unwrap()),
                        ArtifactConfidentiality::Private,
                    )
                })])
                .unwrap(),
                level: key.level,
                key_id: key.id,
                context: scheme.context,
            }
        };
        let online_inputs = BTreeMap::from([
            (private_input, (import_input("private", &private_key), None)),
            (public_input, (import_input("public", &public_key), Some(public_plaintext))),
        ]);
        let imported_member = import_input("member", &member_key);
        let online_members = BTreeMap::from([(
            wire_family,
            (
                mxx_dsl::parallel(2, |_| {
                    Ok((imported_member.clone(), Some(member_plaintext.clone())))
                })
                .unwrap(),
                range.clone(),
            ),
        )]);
        let actual = restored
            .evaluate(&ProgramBindings::new(
                &online_inputs,
                &online_scalar_rhs_bindings,
                &online_rhs_bindings,
                &online_members,
                &BTreeMap::new(),
                Some(&online_helpers),
            ))
            .unwrap();
        let consumer =
            observed_outputs(DslContext::new("rns_bgg-program-consumer"), &restored, &actual)
                .build()
                .unwrap()
                .validate_with_manifests(
                    &ParamEnv::default(),
                    &BTreeMap::from([(
                        production.clone(),
                        store.load_manifest(&production).unwrap(),
                    )]),
                )
                .unwrap();
        let switches = consumer
            .source
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .filter(|node| {
                matches!(node.kind(), mxx_ir_core::node::NodeKind::BlockModSwitch { .. })
            })
            .count();
        assert_eq!(
            switches, 10,
            "scalar/family LUT and indexed/shared direct product, two online polynomials each"
        );
        let mut consumed =
            execute(&consumer, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .unwrap();
        let names = consumed.outputs.keys().cloned().collect::<Vec<_>>();
        for name in names {
            let RuntimeValue::Matrix(actual) =
                consumed.materialize_output(&name, &mut backend, &mut store).unwrap().clone()
            else {
                panic!("polynomial")
            };
            let RuntimeValue::Matrix(expected) =
                produced.materialize_output(&name, &mut backend, &mut store).unwrap().clone()
            else {
                panic!("polynomial")
            };
            check_matrix(actual.as_ref(), expected.as_ref(), &name);
        }
        let mut changed = descriptor.clone();
        changed.cache.gates.pop_first();
        assert!(
            PreparedProgram::import(
                scheme.artifact_binding(),
                scheme.clone(),
                frozen.clone(),
                ranges.clone(),
                &changed,
                production.clone(),
                identity
            )
            .is_err()
        );
        let mut changed_ranges = ranges.clone();
        changed_ranges.bind(ProgramFamilyId::from_index(0), FamilyRange::bounded(0, 2, 4).unwrap());
        assert!(
            PreparedProgram::import(
                scheme.artifact_binding(),
                scheme.clone(),
                frozen.clone(),
                changed_ranges,
                &descriptor,
                production.clone(),
                identity
            )
            .is_err()
        );
        let mut other_program = ExponentLutProgramBuilder::new();
        other_program.set_layout(ExpLayout { stride_log2: 0, width_log2: 1 }, n as usize).unwrap();
        let other_input = other_program
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        other_program.output(other_program.input_wire(other_input).unwrap()).unwrap();
        assert!(
            PreparedProgram::import(
                scheme.artifact_binding(),
                scheme,
                other_program.build().unwrap(),
                ranges.clone(),
                &descriptor,
                production.clone(),
                identity,
            )
            .is_err()
        );
        let other = RnsBggScheme::new(parameters, rand::random());
        assert!(
            PreparedProgram::import(
                other.artifact_binding(),
                other,
                frozen,
                ranges,
                &descriptor,
                production,
                identity
            )
            .is_err()
        );
    }
}

#[cfg(test)]
mod linear_artifact_tests {
    use super::*;
    use crate::rns_bgg::{
        RnsBggEncoding, RnsBggEncodingFamily, RnsBggParameters, RnsBggPlanRequest, RnsBggRow,
    };
    use mxx_ir_core::ParamEnv;
    use mxx_runtime::{
        RuntimeValue,
        artifact::{ArtifactStore, FileArtifactStore},
        backend::poly::cpu_backend,
        execute,
        transcript::SamplingMode,
    };
    use std::collections::BTreeMap;

    #[test]
    fn test_rns_bgg_linear_artifacts_roundtrip_preserves_outputs_and_bindings() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|v| v.parse().unwrap())
            .unwrap_or(16);
        let (parameters, _) = RnsBggParameters::plan(&RnsBggPlanRequest {
            ring_dimension: n,
            widths: vec![2],
            slots: vec![1],
            encoding_error_sigma: bigdecimal::BigDecimal::from(2u8),
            crt_bits: 30,
            terminal_depth: 1,
            max_crt_depth: 8,
            terminal_value_bound: 1u8.into(),
            operations_per_drop: 1,
        })
        .unwrap();
        let scheme = RnsBggScheme::new(parameters.clone(), rand::random());
        let ring = scheme.ring(0).unwrap();
        let key = scheme
            .sample_input_key(
                0,
                ring.bytes_input("hash-key", 32),
                b"key".as_slice(),
                rand::random(),
            )
            .unwrap();
        let secret = scheme.test_secrets().unwrap();
        let input = scheme
            .encode(
                &secret.mask,
                &key,
                &scheme
                    .prepare_input(&secret.payload, &key, ring.uniform_interval((1, 1), -1, 1))
                    .unwrap(),
            )
            .unwrap();
        let multiplier =
            ring.uniform_interval((1, 1), -1, 1) + ring.uniform_interval((1, 1), -1, 1);
        let range = FamilyRange::bounded(1, 2, 4).unwrap();
        let family_key = RnsBggPublicKeyFamily {
            matrices: ProgramValue::Family(
                mxx_dsl::parallel(2, |_| Ok(key.matrix.clone())).unwrap(),
            ),
            level: key.level,
            id: key.id,
            context: scheme.context,
            range: range.clone(),
        };
        let family_input = RnsBggEncodingFamily {
            rows: ProgramValue::Family(mxx_dsl::parallel(2, |_| Ok(input.row.clone())).unwrap()),
            level: 0,
            key_id: key.id,
            context: scheme.context,
            range: range.clone(),
        };
        let mut graph = DslContext::new("rns_bgg-linear-artifact-producer");
        let mut descriptors = Vec::new();
        for (name, is_mul) in [("add", false), ("mul", true)] {
            let prepared = if is_mul {
                scheme.prepare_mul_public(&key, &key, multiplier.clone(), rand::random())
            } else {
                scheme.prepare_add(&key, &key, rand::random())
            }
            .unwrap();
            let expected = scheme.evaluate_linear(&input, &input, &prepared).unwrap();
            let (next, descriptor) = scheme.export_prepared_linear(&prepared, graph, name).unwrap();
            graph = next
                .output(format!("expected-{name}"), expected.row)
                .unwrap()
                .output(format!("expected-{name}-key"), prepared.output_key().matrix.clone())
                .unwrap();
            descriptors.push((name.to_owned(), descriptor));
            let name = format!("family-{name}");
            let prepared = scheme
                .prepare_linear_family(
                    &family_key,
                    &family_key,
                    if is_mul {
                        Some(ProgramValue::Family(
                            mxx_dsl::parallel(2, |_| Ok(multiplier.clone())).unwrap(),
                        ))
                    } else {
                        None
                    },
                    rand::random(),
                )
                .unwrap();
            let expected =
                scheme.evaluate_linear_family(&family_input, &family_input, &prepared).unwrap();
            let (next, descriptor) =
                scheme.export_prepared_linear_family(&prepared, graph, &name).unwrap();
            graph = next;
            for member in 0..2 {
                graph = graph
                    .output(format!("expected-{name}-{member}"), expected.rows.at(member))
                    .unwrap()
                    .output(
                        format!("expected-{name}-{member}-key"),
                        prepared.output_key().matrices.at(member),
                    )
                    .unwrap();
            }
            descriptors.push((name, descriptor));
        }
        let prepared = scheme.prepare_reduce_family(&family_key, rand::random()).unwrap();
        let expected = scheme.evaluate_reduce_family(&family_input, &prepared).unwrap();
        let (next, descriptor) =
            scheme.export_prepared_reduction(&prepared, graph, "reduce").unwrap();
        descriptors.push(("reduce".into(), descriptor));
        graph = next
            .output("expected-reduce", expected.row)
            .unwrap()
            .output("expected-reduce-key", prepared.output_key().matrix.clone())
            .unwrap();
        for column in 0..2 {
            graph = graph
                .private_output(format!("input-{column}"), input.row.entry(0, column).clone())
                .unwrap();
        }
        let graph = graph.build().unwrap().validate(&ParamEnv::default()).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let mut store = FileArtifactStore::new(directory.path()).unwrap();
        let mut backend = cpu_backend(parameters.levels().iter().cloned());
        let mut produced = execute(
            &graph,
            &mut backend,
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut store,
            SamplingMode::Fresh,
        )
        .unwrap();
        let production = produced.production_id.clone().unwrap();
        drop(store);
        let mut store = FileArtifactStore::new(directory.path()).unwrap();
        let input = RnsBggEncoding {
            row: RnsBggRow::new([std::array::from_fn(|column| {
                ring.artifact_input(
                    production.clone(),
                    format!("input-{column}"),
                    (1, parameters.component_columns(0).unwrap()),
                    ArtifactConfidentiality::Private,
                )
            })])
            .unwrap(),
            level: 0,
            key_id: key.id,
            context: scheme.context,
        };
        let family_input = RnsBggEncodingFamily {
            rows: ProgramValue::Family(mxx_dsl::parallel(2, |_| Ok(input.row.clone())).unwrap()),
            level: 0,
            key_id: key.id,
            context: scheme.context,
            range,
        };
        let mut consumer = DslContext::new("rns_bgg-linear-artifact-consumer");
        for (name, original) in descriptors {
            let descriptor: RnsBggPreparedLinearDescriptor =
                serde_json::from_slice(&serde_json::to_vec(&original).unwrap()).unwrap();
            let binding = original.binding();
            if matches!(binding.operation, RnsBggLinearOperation::Reduce { .. }) {
                let prepared = scheme
                    .import_prepared_reduction(&descriptor, production.clone(), binding)
                    .unwrap();
                let actual = scheme.evaluate_reduce_family(&family_input, &prepared).unwrap();
                consumer = consumer
                    .output(format!("expected-{name}"), actual.row)
                    .unwrap()
                    .output(format!("expected-{name}-key"), prepared.output_key().matrix.clone())
                    .unwrap();
                assert!(
                    scheme
                        .import_prepared_linear_family(&descriptor, production.clone(), binding)
                        .is_err()
                );
            } else if binding.operation.input_range().is_some() {
                let prepared = scheme
                    .import_prepared_linear_family(&descriptor, production.clone(), binding)
                    .unwrap();
                let actual =
                    scheme.evaluate_linear_family(&family_input, &family_input, &prepared).unwrap();
                for member in 0..2 {
                    consumer = consumer
                        .output(format!("expected-{name}-{member}"), actual.rows.at(member))
                        .unwrap()
                        .output(
                            format!("expected-{name}-{member}-key"),
                            prepared.output_key().matrices.at(member),
                        )
                        .unwrap();
                }
                let mut wrong = binding.clone();
                match &mut wrong.operation {
                    RnsBggLinearOperation::Add { range, .. } |
                    RnsBggLinearOperation::MulPublic { range, .. } => {
                        *range = Some(FamilyRange::bounded(0, 2, 4).unwrap());
                    }
                    RnsBggLinearOperation::Reduce { .. } => unreachable!("binary family operation"),
                }
                assert!(
                    scheme
                        .import_prepared_linear_family(&descriptor, production.clone(), &wrong)
                        .is_err()
                );
            } else {
                let prepared = scheme
                    .import_prepared_linear(&descriptor, production.clone(), binding)
                    .unwrap();
                let actual = scheme.evaluate_linear(&input, &input, &prepared).unwrap();
                consumer = consumer
                    .output(format!("expected-{name}"), actual.row)
                    .unwrap()
                    .output(format!("expected-{name}-key"), prepared.output_key().matrix.clone())
                    .unwrap();
            }
            let mut wrong = binding.clone();
            wrong.output[0] ^= 1;
            assert!(scheme.check_linear_descriptor(&descriptor, &wrong).is_err());
            wrong = binding.clone();
            wrong.operation = match &binding.operation {
                RnsBggLinearOperation::Add { rhs, range } => {
                    RnsBggLinearOperation::MulPublic { rhs: *rhs, range: range.clone() }
                }
                RnsBggLinearOperation::MulPublic { rhs, range } => {
                    RnsBggLinearOperation::Add { rhs: *rhs, range: range.clone() }
                }
                RnsBggLinearOperation::Reduce { range } => {
                    RnsBggLinearOperation::Add { rhs: binding.lhs, range: Some(range.clone()) }
                }
            };
            assert!(scheme.check_linear_descriptor(&descriptor, &wrong).is_err());
        }
        let consumer = consumer
            .build()
            .unwrap()
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production.clone(), store.load_manifest(&production).unwrap())]),
            )
            .unwrap();
        let mut consumed =
            execute(&consumer, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .unwrap();
        let names = consumed.outputs.keys().cloned().collect::<Vec<_>>();
        for name in names {
            let RuntimeValue::Matrix(actual) =
                consumed.materialize_output(&name, &mut backend, &mut store).unwrap().clone()
            else {
                panic!("polynomial")
            };
            let RuntimeValue::Matrix(expected) =
                produced.materialize_output(&name, &mut backend, &mut store).unwrap().clone()
            else {
                panic!("polynomial")
            };
            assert_eq!(actual.as_ref(), expected.as_ref(), "{name}");
        }
    }
}

#[cfg(test)]
mod lut_artifact_tests {
    use super::*;
    use crate::{
        program::LutTable,
        rns_bgg::{RnsBggEncoding, RnsBggParameters, RnsBggRow},
    };
    use mxx_ir_core::ParamEnv;
    use mxx_runtime::{
        RuntimeValue,
        artifact::{ArtifactStore, FileArtifactStore},
        backend::poly::cpu_backend,
        execute,
        transcript::SamplingMode,
    };
    use std::collections::BTreeMap;

    #[test]
    fn test_rns_bgg_prepared_artifacts_roundtrip_preserves_online_outputs() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|v| v.parse().unwrap())
            .unwrap_or(16);
        let high_parameters = DCRTPolyParams::new(n, 2, 17, 4, None, None);
        let prime = high_parameters.to_crt().0[0];
        let low_parameters = DCRTPolyParams::new(n, 1, 17, 4, Some(vec![prime]), None);
        let parameters =
            RnsBggParameters::new(vec![high_parameters, low_parameters], 2, 1).unwrap();
        let scheme = RnsBggScheme::new(parameters.clone(), rand::random());
        let high = scheme.ring(0).unwrap();
        let seed = high.bytes_input("hash-key", 32);
        let secret = scheme.test_secrets().unwrap();
        let key =
            scheme.sample_input_key(0, seed.clone(), b"input".as_slice(), rand::random()).unwrap();
        let input = scheme
            .encode(
                &secret.mask,
                &key,
                &scheme
                    .prepare_input(&secret.payload, &key, high.polynomial([0.into(), 1.into()]))
                    .unwrap(),
            )
            .unwrap();
        let bank = scheme
            .sample_mask_bank(
                0,
                crate::layout::ExpLayout {
                    stride_log2: 0,
                    width_log2: (2 as usize).trailing_zeros(),
                },
                seed.clone(),
                b"bank".as_slice(),
                rand::random(),
            )
            .unwrap();
        let online_bank = scheme.prepare_online_masks(&secret.mask, &bank).unwrap();
        let table = LutTable::unary(2, 2, vec![1, 0]).unwrap();
        let material = scheme
            .sample_lut_material(
                &secret.payload,
                Some(&table),
                None,
                &bank,
                seed.clone(),
                b"helpers".as_slice(),
                rand::random(),
            )
            .unwrap();
        let prepared = scheme.prepare_lut(&key, &material, rand::random()).unwrap();
        let expected = scheme.evaluate_lut(&input, &prepared, &online_bank).unwrap();
        let (mut graph, descriptor) = scheme
            .export_prepared_lut(&prepared, DslContext::new("rns_bgg-prepared-producer"), "lut")
            .unwrap();
        let range = FamilyRange::bounded(1, 2, 4).unwrap();
        let family_key = RnsBggPublicKeyFamily {
            matrices: ProgramValue::Family(
                mxx_dsl::parallel(2, |_| Ok(key.matrix.clone())).unwrap(),
            ),
            level: 0,
            id: key.id,
            context: scheme.context,
            range: range.clone(),
        };
        let family_input = crate::rns_bgg::RnsBggEncodingFamily {
            rows: ProgramValue::Family(mxx_dsl::parallel(2, |_| Ok(input.row.clone())).unwrap()),
            level: 0,
            key_id: key.id,
            context: scheme.context,
            range: range.clone(),
        };
        let family_material = scheme
            .sample_lut_material_family(
                &secret.payload,
                Some(&table),
                None,
                &bank,
                4.into(),
                seed,
                b"family-helpers".as_slice(),
                rand::random(),
            )
            .unwrap();
        let family_prepared =
            scheme.prepare_lut_family(&family_key, &family_material, rand::random()).unwrap();
        let family_expected =
            scheme.evaluate_lut_family(&family_input, &family_prepared, &online_bank).unwrap();
        let (family_graph, family_descriptor) =
            scheme.export_prepared_lut_family(&family_prepared, graph, "family-lut").unwrap();
        graph = family_graph;
        for member in 0..2 {
            graph = graph
                .output(format!("expected-family-{member}"), family_expected.rows.at(member))
                .unwrap();
        }
        for column in 0..2 {
            graph = graph
                .private_output(format!("input-{column}"), input.row.entry(0, column).clone())
                .unwrap();
        }
        let graph = crate::rns_bgg::encoding::encoding_tests::export_online_masks(
            &online_bank,
            graph,
            "online-mask",
        );
        let graph = graph
            .output("expected", expected.row)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let mut backend = cpu_backend(parameters.levels().iter().cloned());
        let directory = tempfile::tempdir().unwrap();
        let mut store = FileArtifactStore::new(directory.path()).unwrap();
        let produced = execute(
            &graph,
            &mut backend,
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut store,
            SamplingMode::Fresh,
        )
        .unwrap();
        let production = produced.production_id.clone().unwrap();
        let online_bank = crate::rns_bgg::encoding::encoding_tests::import_online_masks(
            &scheme,
            &online_bank,
            &production,
            "online-mask",
        );
        drop(store);
        let mut store = FileArtifactStore::new(directory.path()).unwrap();
        let descriptor: RnsBggPreparedLutDescriptor =
            serde_json::from_slice(&serde_json::to_vec(&descriptor).unwrap()).unwrap();
        let imported = scheme
            .import_prepared_lut(
                &descriptor,
                production.clone(),
                &material.binding(),
                key.id,
                prepared.output.id,
            )
            .unwrap();
        let input = RnsBggEncoding {
            row: RnsBggRow::new([std::array::from_fn(|column| {
                high.artifact_input(
                    production.clone(),
                    format!("input-{column}"),
                    (1, parameters.component_columns(0).unwrap()),
                    ArtifactConfidentiality::Private,
                )
            })])
            .unwrap(),
            level: 0,
            key_id: key.id,
            context: scheme.context,
        };
        let actual = scheme.evaluate_lut(&input, &imported, &online_bank).unwrap();
        let graph = DslContext::new("rns_bgg-prepared-consumer")
            .output("actual", actual.row)
            .unwrap()
            .build()
            .unwrap()
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production.clone(), store.load_manifest(&production).unwrap())]),
            )
            .unwrap();
        // Only the two output polynomials are switched online. Import must
        // not pull the internal b/public-key preprocessing switches back in.
        let switches = graph
            .source
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .filter(|node| {
                matches!(node.kind(), mxx_ir_core::node::NodeKind::BlockModSwitch { .. })
            })
            .count();
        assert_eq!(switches, 2);
        let consumed =
            execute(&graph, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .unwrap();
        for column in 0..2 {
            let RuntimeValue::Matrix(actual) = &consumed.outputs[&format!("actual.{column}")]
            else {
                panic!("polynomial")
            };
            let RuntimeValue::Matrix(expected) = &produced.outputs[&format!("expected.{column}")]
            else {
                panic!("polynomial")
            };
            assert_eq!(actual.as_ref(), expected.as_ref());
        }
        let family_descriptor: RnsBggPreparedLutFamilyDescriptor =
            serde_json::from_slice(&serde_json::to_vec(&family_descriptor).unwrap()).unwrap();
        let family_imported = scheme
            .import_prepared_lut_family(
                &family_descriptor,
                production.clone(),
                family_material.binding(),
                key.id,
                family_prepared.output.id,
                &range,
            )
            .unwrap();
        let family_input = crate::rns_bgg::RnsBggEncodingFamily {
            rows: ProgramValue::Family(mxx_dsl::parallel(2, |_| Ok(input.row.clone())).unwrap()),
            level: 0,
            key_id: key.id,
            context: scheme.context,
            range: range.clone(),
        };
        let result =
            scheme.evaluate_lut_family(&family_input, &family_imported, &online_bank).unwrap();
        let mut consumer = DslContext::new("rns_bgg-prepared-family-consumer");
        for member in 0..2 {
            consumer =
                consumer.output(format!("actual-family-{member}"), result.rows.at(member)).unwrap();
        }
        let consumer = consumer
            .build()
            .unwrap()
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production.clone(), store.load_manifest(&production).unwrap())]),
            )
            .unwrap();
        let family_switches = consumer
            .source
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .filter(|node| {
                matches!(node.kind(), mxx_ir_core::node::NodeKind::BlockModSwitch { .. })
            })
            .count();
        assert_eq!(family_switches, 2, "one final switch per polynomial in the family body");
        let result =
            execute(&consumer, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .unwrap();
        for member in 0..2 {
            for column in 0..2 {
                let RuntimeValue::Matrix(actual) =
                    &result.outputs[&format!("actual-family-{member}.{column}")]
                else {
                    panic!("polynomial")
                };
                let RuntimeValue::Matrix(expected) =
                    &produced.outputs[&format!("expected-family-{member}.{column}")]
                else {
                    panic!("polynomial")
                };
                assert_eq!(actual.as_ref(), expected.as_ref());
            }
        }
        assert!(
            scheme
                .import_prepared_lut_family(
                    &family_descriptor,
                    production.clone(),
                    family_material.binding(),
                    key.id,
                    family_prepared.output.id,
                    &FamilyRange::bounded(0, 2, 4).unwrap()
                )
                .is_err()
        );
        let mut wrong = material.binding();
        wrong.material_id[0] ^= 1;
        assert!(
            scheme
                .import_prepared_lut(
                    &descriptor,
                    production.clone(),
                    &wrong,
                    key.id,
                    prepared.output.id
                )
                .is_err()
        );
        assert!(
            scheme
                .import_prepared_lut(
                    &descriptor,
                    production,
                    &material.binding(),
                    rand::random(),
                    prepared.output.id
                )
                .is_err()
        );
    }
}

#[cfg(test)]
pub(crate) mod readout_tests {
    use super::*;
    use crate::{
        layout::ExpLayout,
        noise_refresh::prf::SparseLwrPrfProfile,
        rns_bgg::{RnsBggNoiseBound, RnsBggParameters, RnsBggPlanRequest},
    };
    use mxx_dsl::DslContext;
    use mxx_ir_core::ParamEnv;
    use mxx_primitives::{element::PolyElem, poly::Poly};
    use mxx_runtime::{
        Backend, RuntimeValue,
        artifact::{ArtifactStore, FileArtifactStore},
        backend::poly::cpu_backend,
        execute,
        transcript::SamplingMode,
    };
    use std::collections::BTreeMap;

    #[test]
    fn test_rns_bgg_affine_readout_covers_full_native_domain() {
        check_cpu_domain(0, [vec![1.into()], vec![]], false, false);
    }

    #[test]
    fn test_rns_bgg_affine_readout_covers_strided_native_domain() {
        check_cpu_domain(1, [vec![1.into()], vec![]], false, false);
    }

    #[test]
    fn test_rns_bgg_weighted_readout_preserves_noise_multiplier() {
        check_cpu_domain(1, [vec![], vec![0.into(), 17.into()]], false, false);
    }

    #[test]
    fn test_rns_bgg_indexed_readout_weights_survive_artifact_reopen() {
        check_cpu_domain(1, [vec![17.into()], vec![]], true, false);
    }

    #[test]
    fn test_rns_bgg_aggregate_readout_survives_artifact_reopen() {
        for indexed in [false, true] {
            check_cpu_domain(2, [vec![17.into()], vec![]], indexed, true);
        }
    }

    fn check_cpu_domain(stride_log2: u32, weight: RnsBggReadoutWeight, indexed: bool, sum: bool) {
        check_full_domain(
            stride_log2,
            weight,
            indexed,
            sum,
            |parameters| cpu_backend(parameters.levels().iter().cloned()),
            |matrix, name, q, bound| {
                for polynomial in matrix.block_entries(0..1, 0..1).into_iter().flatten() {
                    for coefficient in polynomial.coeffs() {
                        let absolute = coefficient.value().clone().min(q - coefficient.value());
                        assert!(&absolute <= bound, "{name}: {absolute} > {bound}");
                    }
                }
            },
        );
    }

    pub(crate) fn check_full_domain<B: Backend>(
        stride_log2: u32,
        weight: RnsBggReadoutWeight,
        indexed: bool,
        sum: bool,
        make_backend: impl FnOnce(&RnsBggParameters) -> B,
        check_matrix: impl Fn(&B::Matrix, &str, &BigUint, &BigUint),
    ) {
        let n: u32 = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|v| v.parse().unwrap())
            .unwrap_or(8);
        let profile = SparseLwrPrfProfile::new(
            ExpLayout { stride_log2, width_log2: n.trailing_zeros() - stride_log2 },
            2,
        )
        .unwrap();
        let width = profile.layout.width();
        let readout = profile.canonical_readout().unwrap();
        let (parameters, _) = RnsBggParameters::plan(&RnsBggPlanRequest {
            ring_dimension: n,
            widths: vec![width],
            slots: vec![1],
            encoding_error_sigma: bigdecimal::BigDecimal::from(2u8),
            crt_bits: 30,
            terminal_depth: 1,
            max_crt_depth: 12,
            terminal_value_bound: BigUint::from(3u8) *
                weight.iter().flatten().map(|x| x.magnitude()).max().unwrap().clone(),
            operations_per_drop: 1,
        })
        .unwrap();
        let scheme = RnsBggScheme::new(parameters.clone(), rand::random());
        let secrets = scheme.test_secrets().unwrap();
        let high = scheme.ring(0).unwrap();
        let low = scheme.ring(1).unwrap();
        let seed = high.bytes_input("hash-key", 32);
        let key =
            scheme.sample_input_key(0, seed.clone(), b"input".as_slice(), rand::random()).unwrap();
        let bank = scheme
            .sample_mask_bank(0, profile.layout, seed.clone(), b"bank".as_slice(), rand::random())
            .unwrap();
        let material = scheme
            .sample_readout_material(
                &secrets.payload,
                &readout,
                &bank,
                seed.clone(),
                b"readout".as_slice(),
                rand::random(),
                Some(RnsBggReadoutWeights::Shared(weight.clone())),
                None,
            )
            .unwrap();
        let public_input = RnsBggPublicProgramValue::Scalar((key.clone(), None));
        let prepared = scheme.prepare_readout(&readout, &public_input, &material).unwrap();
        let (mut producer, descriptor) = scheme
            .export_prepared_readout(
                &prepared,
                DslContext::new("rns_bgg-readout-preprocess"),
                "readout",
            )
            .unwrap();
        let descriptor: RnsBggPreparedReadoutDescriptor =
            serde_json::from_slice(&serde_json::to_vec(&descriptor).unwrap()).unwrap();
        let range = crate::program::FamilyRange::bounded(1, 2 * width, 2 * width + 2).unwrap();
        let family_input = crate::rns_bgg::RnsBggPublicKeyFamily {
            matrices: ProgramValue::Family(
                mxx_dsl::parallel(range.count().clone(), |_| Ok(key.matrix.clone())).unwrap(),
            ),
            level: key.level,
            id: key.id,
            context: key.context,
            range: range.clone(),
        }
        .into_program()
        .unwrap();
        let routed_weights = (0..2 * width)
            .map(|index| {
                let row = readout_offset(&low, &weight).unwrap();
                if index % 2 == 0 {
                    row
                } else {
                    // Swap destinations and rotate one coefficient, preserving the bound.
                    RnsBggRow::new([[
                        low.polynomial([0.into(), (-1).into()]) * row.entry(0, 1).clone(),
                        low.polynomial([0.into(), (-1).into()]) * row.entry(0, 0).clone(),
                    ]])
                    .unwrap()
                }
            })
            .collect::<Vec<_>>();
        let family_material = if indexed || sum {
            let weights = if indexed {
                RnsBggReadoutWeights::Indexed {
                    rows: Family::pack(routed_weights.clone()).unwrap(),
                    coefficient_bound: weight
                        .iter()
                        .flatten()
                        .map(|v| v.magnitude())
                        .max()
                        .cloned()
                        .unwrap_or_default(),
                }
            } else {
                RnsBggReadoutWeights::Shared(weight.clone())
            };
            Some(
                scheme
                    .sample_readout_material(
                        &secrets.payload,
                        &readout,
                        &bank,
                        seed,
                        b"family-readout".as_slice(),
                        rand::random(),
                        Some(weights),
                        Some(if sum {
                            crate::rns_bgg::RnsBggTerminalFamilyMode::Sum
                        } else {
                            crate::rns_bgg::RnsBggTerminalFamilyMode::Independent
                        }),
                    )
                    .unwrap(),
            )
        } else {
            None
        };
        let family_prepared = scheme
            .prepare_readout(&readout, &family_input, family_material.as_ref().unwrap_or(&material))
            .unwrap();
        let family_binding = family_prepared.binding();
        let family_id = family_prepared.id();
        let (next, family_descriptor) =
            scheme.export_prepared_readout(&family_prepared, producer, "family-readout").unwrap();
        producer = next;
        let family_descriptor: RnsBggPreparedReadoutDescriptor =
            serde_json::from_slice(&serde_json::to_vec(&family_descriptor).unwrap()).unwrap();

        let binding = prepared.binding();
        let stage_id = prepared.id();
        let mut input_descriptors = Vec::new();
        for x in 0..2 * width {
            let mu = profile.layout.mono(&high, &BigInt::from(x)).unwrap();
            let material = scheme.prepare_input(&secrets.payload, &key, mu).unwrap();
            let (next, descriptor) = scheme
                .export_input_material(&key, &material, producer, &format!("input-{x}"))
                .unwrap();
            producer = next;
            input_descriptors.push(descriptor);
        }
        let mut backend = make_backend(&parameters);
        std::fs::create_dir_all("test_data/rns_bgg_readout_artifacts").unwrap();
        let directory = tempfile::tempdir_in("test_data/rns_bgg_readout_artifacts").unwrap();
        let mut store = FileArtifactStore::new(directory.path()).unwrap();
        let produced = execute(
            &producer.build().unwrap().validate(&ParamEnv::default()).unwrap(),
            &mut backend,
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut store,
            SamplingMode::Fresh,
        )
        .unwrap();
        let production = produced.production_id.unwrap();
        drop(store);
        let mut store = FileArtifactStore::open(directory.path()).unwrap();
        let prepared = scheme
            .import_prepared_readout(
                &descriptor,
                production.clone(),
                &readout,
                &binding,
                key.id,
                stage_id,
            )
            .unwrap();
        let mut wrong_id = stage_id;
        wrong_id[0] ^= 1;
        assert!(
            scheme
                .import_prepared_readout(
                    &descriptor,
                    production.clone(),
                    &readout,
                    &binding,
                    key.id,
                    wrong_id,
                )
                .is_err()
        );
        let public = scheme.project_readout(&readout, &public_input, &prepared, &()).unwrap();
        let family_prepared = scheme
            .import_prepared_readout(
                &family_descriptor,
                production.clone(),
                &readout,
                &family_binding,
                key.id,
                family_id,
            )
            .unwrap();
        let family_public =
            scheme.project_readout(&readout, &family_input, &family_prepared, &()).unwrap();

        assert!(
            matches!(public.weight(), RnsBggReadoutWeights::Shared(actual) if actual == &weight)
        );
        let masks = scheme.prepare_online_masks(&secrets.mask, &prepared.mask_bank()).unwrap();
        let noise = parameters
            .terminal_noise_bound(
                0,
                [(
                    &RnsBggNoiseBound {
                        error: parameters.encoding_error_bound.clone(),
                        payload_l1: 1u8.into(),
                    },
                    width,
                    &1u8.into(),
                )],
                material.target_bound().clone(),
            )
            .unwrap();
        assert!(parameters.terminal_noise_fits(1, &noise).unwrap());
        let q = parameters.level(1).unwrap().modulus();
        let s = secrets.mask.row.clone().reduce_modulus(BigInt::from(q.as_ref().clone()));
        let mut graph = DslContext::new("rns_bgg-affine-full-native-domain");
        let mut input_rows = Vec::new();
        let mut scalar_rows = Vec::new();
        for (x, descriptor) in input_descriptors.iter().enumerate() {
            let (key, material) =
                scheme.import_input_material(descriptor, production.clone(), key.id, 0).unwrap();
            let input = scheme.encode(&secrets.mask, &key, &material).unwrap();
            input_rows.push(input.row.clone());
            let input = RnsBggEncodingProgramValue::Scalar((input, None));
            let value = scheme.evaluate_readout(&readout, &input, &prepared, &masks).unwrap();
            scalar_rows.push(value.values.scalar().unwrap().clone());
            // The canonical PRF definition, independent of the affine target construction.
            let half_value = BigInt::from(profile.p_prime * (x % width) / width + 1);
            let expected = if x < width { half_value } else { 1u8 - half_value };
            let expected =
                scheme.scalar(1, expected).unwrap() * readout_offset(&low, &weight).unwrap();
            let error = value.values.scalar().unwrap().clone() -
                s.clone() * public.values.scalar().unwrap().clone() -
                expected;
            graph = graph.output(format!("error-{x}"), error).unwrap();
        }
        let family_encodings = crate::rns_bgg::RnsBggEncodingFamily {
            rows: ProgramValue::Family(Family::pack(input_rows).unwrap()),
            level: key.level,
            key_id: key.id,
            context: key.context,
            range,
        }
        .into_program()
        .unwrap();
        let family_value =
            scheme.evaluate_readout(&readout, &family_encodings, &family_prepared, &masks).unwrap();
        let ProgramValue::Family((rows, _)) = family_value.values else { panic!("family") };
        let ProgramValue::Family((matrices, _)) = family_public.values else { panic!("family") };
        if sum {
            assert_eq!(rows.count(), &1.into());
            assert_eq!(matrices.count(), &1.into());
            let expected = (0..2 * width)
                .map(|index| {
                    let half_value = BigInt::from(profile.p_prime * (index % width) / width + 1);
                    let value = if index < width { half_value } else { 1u8 - half_value };
                    let weight_row = if indexed {
                        routed_weights[index].clone()
                    } else {
                        readout_offset(&low, &weight).unwrap()
                    };
                    scheme.scalar(1, value).unwrap() * weight_row
                })
                .reduce(|lhs, rhs| lhs + rhs)
                .unwrap();
            graph = graph
                .output("aggregate-error", rows.at(0) - s.clone() * matrices.at(0) - expected)
                .unwrap();
        } else {
            for (index, row) in scalar_rows.into_iter().enumerate() {
                if indexed {
                    let half_value = BigInt::from(profile.p_prime * (index % width) / width + 1);
                    let value = if index < width { half_value } else { 1u8 - half_value };
                    let expected = scheme.scalar(1, value).unwrap() * routed_weights[index].clone();
                    graph = graph
                        .output(
                            format!("indexed-error-{index}"),
                            rows.at(index) - s.clone() * matrices.at(index) - expected,
                        )
                        .unwrap();
                    continue;
                }
                graph = graph
                    .output(format!("family-match-row-{index}"), rows.at(index) - row)
                    .unwrap();
                graph = graph
                    .output(
                        format!("family-match-public-{index}"),
                        matrices.at(index) - public.values.scalar().unwrap().clone(),
                    )
                    .unwrap();
            }
        }
        let manifest = store.load_manifest(&production).unwrap();
        let graph = graph
            .build()
            .unwrap()
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production, manifest)]),
            )
            .unwrap();
        let switches = graph
            .source
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .filter(|node| {
                matches!(node.kind(), mxx_ir_core::node::NodeKind::BlockModSwitch { .. })
            })
            .count();
        assert_eq!(
            switches,
            4 * width + 2,
            "two blocks per scalar input and one family body; public switches stay in preprocessing"
        );
        let results =
            execute(&graph, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                .unwrap();
        for (name, value) in results.outputs {
            let RuntimeValue::Matrix(matrix) = value else { panic!("matrix") };
            let bound = if name == "aggregate-error" {
                let input_bound = RnsBggNoiseBound {
                    error: parameters.encoding_error_bound.clone(),
                    payload_l1: 1u8.into(),
                };
                let gain = 1u8.into();
                parameters
                    .terminal_noise_bound(
                        0,
                        [(&input_bound, width * (2 * width), &gain)],
                        material.target_bound() * (2 * width),
                    )
                    .unwrap()
                    .error
            } else if name.starts_with("family-match") {
                BigUint::from(0u8)
            } else {
                noise.error.clone()
            };
            check_matrix(&matrix, &name, q.as_ref(), &bound);
        }
    }
}

#[cfg(test)]
pub(crate) mod terminal_family_tests {
    use super::*;
    use crate::{
        program::FamilyRange,
        rns_bgg::{RnsBggParameters, RnsBggPlanRequest},
    };
    use mxx_dsl::DslContext;
    use mxx_ir_core::ParamEnv;
    use mxx_primitives::{element::PolyElem, poly::Poly};
    use mxx_runtime::{
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
        transcript::SamplingMode,
    };
    use num_traits::Zero;
    use std::collections::BTreeMap;

    #[test]
    fn test_rns_bgg_terminal_family_matches_scalar_members() {
        let (graph, parameters) = fixture();
        let result = execute(
            &graph,
            &mut cpu_backend(parameters.levels().iter().cloned()),
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        for (name, value) in result.outputs {
            let RuntimeValue::Matrix(matrix) = value else { panic!("matrix") };
            for polynomial in matrix.block_entries(0..1, 0..1).into_iter().flatten() {
                assert!(polynomial.coeffs().iter().all(|c| c.value().is_zero()), "{name}");
            }
        }
    }

    pub(crate) fn check_indexed_artifacts<B: mxx_runtime::Backend>(
        make_backend: impl Fn(&RnsBggParameters) -> B,
        check_matrix: impl Fn(&B::Matrix, &str, &num_bigint::BigUint, &num_bigint::BigUint),
    ) {
        use mxx_runtime::artifact::{ArtifactStore, FileArtifactStore};
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|v| v.parse().unwrap())
            .unwrap_or(8);
        let (parameters, _) = RnsBggParameters::plan(&RnsBggPlanRequest {
            ring_dimension: n,
            widths: vec![2],
            slots: vec![2],
            encoding_error_sigma: bigdecimal::BigDecimal::from(2u8),
            crt_bits: 30,
            terminal_depth: 2,
            max_crt_depth: 12,
            terminal_value_bound: 18u8.into(),
            operations_per_drop: 1,
        })
        .unwrap();
        let scheme = RnsBggScheme::new(parameters.clone(), rand::random());
        let high = scheme.ring(0).unwrap();
        let low = scheme.ring(1).unwrap();
        let seed = high.bytes_input("hash-key", 32);
        let payload_secret = scheme.sample_payload_secret().unwrap();
        let key =
            scheme.sample_input_key(0, seed.clone(), b"key".as_slice(), rand::random()).unwrap();
        let private_input = scheme
            .prepare_input(&payload_secret, &key, high.polynomial([0.into(), 1.into()]))
            .unwrap();
        let bank = scheme
            .sample_mask_bank(
                0,
                crate::layout::ExpLayout { stride_log2: 0, width_log2: 1 },
                seed.clone(),
                b"bank".as_slice(),
                rand::random(),
            )
            .unwrap();
        let table =
            RowLutTable::new(vec![[vec![], vec![]], [vec![1.into()], vec![]]], None).unwrap();
        let destinations = [
            RnsBggRow::new([[low.identity(1), low.zero((1, 1))]]).unwrap(),
            RnsBggRow::new([[low.zero((1, 1)), low.polynomial([0.into(), 17.into()])]]).unwrap(),
        ];
        let transforms = Family::pack(
            destinations
                .iter()
                .cloned()
                .map(|row| {
                    RnsBggMatrix::from_rows(
                        row,
                        RnsBggRow::new([[low.zero((1, 1)), low.zero((1, 1))]]).unwrap(),
                    )
                })
                .collect(),
        )
        .unwrap();
        let material = scheme
            .sample_terminal_material_family(
                &payload_secret,
                &table,
                &transforms,
                &bank,
                seed,
                b"targets".as_slice(),
                rand::random(),
            )
            .unwrap();
        for index in 0..2 {
            let member = material.at(index);
            for (mask, _) in &member.material.branches {
                let original =
                    bank.branches.iter().find(|branch| branch.sigma == mask.sigma).unwrap();
                assert_eq!(
                    mask.public.flatten(),
                    original.public.flatten(),
                    "indexed terminal material must capture the original shared mask bank"
                );
            }
        }
        let range = FamilyRange::bounded(1, 2, 4).unwrap();
        let input = RnsBggPublicKeyFamily {
            matrices: ProgramValue::Family(
                mxx_dsl::parallel(2, |_| Ok(key.matrix.clone())).unwrap(),
            ),
            level: 0,
            id: key.id,
            context: scheme.context,
            range: range.clone(),
        };
        let (mut producer, input_descriptor) = scheme
            .export_input_material(
                &key,
                &private_input,
                DslContext::new("indexed-terminal-producer"),
                "input",
            )
            .unwrap();
        let mut descriptors = Vec::new();
        for (mode, prefix) in [
            (RnsBggTerminalFamilyMode::Independent, "independent"),
            (RnsBggTerminalFamilyMode::Sum, "sum"),
        ] {
            let prepared = scheme.prepare_terminal_family(&input, &material, mode, None).unwrap();
            let (next, descriptor) =
                scheme.export_prepared_terminal_family(&prepared, producer, prefix).unwrap();
            producer = next;
            let descriptor: RnsBggPreparedTerminalFamilyDescriptor =
                serde_json::from_slice(&serde_json::to_vec(&descriptor).unwrap()).unwrap();
            descriptors.push((descriptor, prepared.binding(), prepared.id()));
        }
        std::fs::create_dir_all("test_data/rns_bgg_indexed_terminal_artifacts").unwrap();
        let directory =
            tempfile::tempdir_in("test_data/rns_bgg_indexed_terminal_artifacts").unwrap();
        let mut store = FileArtifactStore::new(directory.path()).unwrap();
        let mut backend = make_backend(&parameters);
        let produced = execute(
            &producer.build().unwrap().validate(&ParamEnv::default()).unwrap(),
            &mut backend,
            BTreeMap::from([(
                "hash-key".into(),
                RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec()),
            )]),
            &mut store,
            SamplingMode::Fresh,
        )
        .unwrap();
        let production = produced.production_id.unwrap();
        drop(store);
        let mut store = FileArtifactStore::open(directory.path()).unwrap();
        // No online s was constructed in the producer. Both consumers use
        // fresh s and freshly prepared online masks with the same stored t-data.
        for online in 0..2 {
            let mask_secret = scheme.sample_mask_secret().unwrap();
            let (imported_key, imported_input) = scheme
                .import_input_material(&input_descriptor, production.clone(), key.id, 0)
                .unwrap();
            let encoding = scheme.encode(&mask_secret, &imported_key, &imported_input).unwrap();
            let inputs = RnsBggEncodingFamily {
                rows: ProgramValue::Family(
                    mxx_dsl::parallel(2, |_| Ok(encoding.row.clone())).unwrap(),
                ),
                level: 0,
                key_id: key.id,
                context: scheme.context,
                range: range.clone(),
            };
            let q = parameters.level(1).unwrap().modulus();
            let s = mask_secret.row.clone().reduce_modulus(BigInt::from(q.as_ref().clone()));

            let mut consumer = DslContext::new(format!("indexed-terminal-consumer-{online}"));
            for (descriptor, binding, stage_id) in &descriptors {
                let prepared = scheme
                    .import_prepared_terminal_family(
                        descriptor,
                        production.clone(),
                        binding,
                        key.id,
                        *stage_id,
                    )
                    .unwrap();
                let masks =
                    scheme.prepare_online_masks(&mask_secret, &prepared.mask_bank()).unwrap();
                let rows = scheme.evaluate_terminal_family(&inputs, &prepared, &masks).unwrap();
                let expected = match prepared.mode() {
                    RnsBggTerminalFamilyMode::Independent => destinations.to_vec(),
                    RnsBggTerminalFamilyMode::Sum => {
                        vec![destinations[0].clone() + destinations[1].clone()]
                    }
                };
                assert_eq!(rows.count(), &expected.len().into());
                for (index, target) in expected.into_iter().enumerate() {
                    consumer = consumer
                        .output(
                            format!("{:?}-{index}", prepared.mode()),
                            rows.at(index) -
                                s.clone() * prepared.public_matrices().at(index) -
                                target,
                        )
                        .unwrap();
                }
            }
            let graph = consumer
                .build()
                .unwrap()
                .validate_with_manifests(
                    &ParamEnv::default(),
                    &BTreeMap::from([(
                        production.clone(),
                        store.load_manifest(&production).unwrap(),
                    )]),
                )
                .unwrap();
            let input_bound = crate::rns_bgg::RnsBggNoiseBound {
                error: parameters.encoding_error_bound.clone(),
                payload_l1: 1u8.into(),
            };
            let gain = 1u8.into();
            let scalar_noise = parameters
                .terminal_noise_bound(0, [(&input_bound, 2, &gain)], 17u8.into())
                .unwrap();
            let sum_noise = parameters
                .terminal_noise_bound(
                    0,
                    [(&input_bound, 2, &gain), (&input_bound, 2, &gain)],
                    18u8.into(),
                )
                .unwrap();
            let result =
                execute(&graph, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                    .unwrap();
            for (name, value) in result.outputs {
                let RuntimeValue::Matrix(matrix) = value else { panic!("matrix") };
                let bound =
                    if name.starts_with("Sum") { &sum_noise.error } else { &scalar_noise.error };
                check_matrix(&matrix, &name, q.as_ref(), bound);
            }
        }
    }

    #[test]
    fn test_rns_bgg_indexed_terminal_artifacts_reopen_with_new_online_secrets() {
        check_indexed_artifacts(
            |parameters| cpu_backend(parameters.levels().iter().cloned()),
            |matrix, name, q, bound| {
                for polynomial in matrix.block_entries(0..1, 0..1).into_iter().flatten() {
                    for coefficient in polynomial.coeffs() {
                        let absolute = coefficient.value().clone().min(q - coefficient.value());
                        assert!(&absolute <= bound, "{name}: {absolute} > {bound}");
                    }
                }
            },
        );
    }

    pub(crate) fn fixture() -> (mxx_ir_core::ValidatedGraph, RnsBggParameters) {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|s| s.parse().unwrap())
            .unwrap_or(8);
        let (parameters, _) = RnsBggParameters::plan(&RnsBggPlanRequest {
            ring_dimension: n,
            widths: vec![2],
            slots: vec![2],
            encoding_error_sigma: bigdecimal::BigDecimal::from(2u8),
            crt_bits: 30,
            terminal_depth: 1,
            max_crt_depth: 12,
            terminal_value_bound: 3u8.into(),
            operations_per_drop: 1,
        })
        .unwrap();
        let scheme = RnsBggScheme::new(parameters.clone(), rand::random());
        let secrets = scheme.test_secrets().unwrap();
        let ring = scheme.ring(0).unwrap();
        let seed = ring.bytes_input("hash-key", 32);
        let bank = scheme
            .sample_mask_bank(
                0,
                crate::layout::ExpLayout {
                    stride_log2: 0,
                    width_log2: (2 as usize).trailing_zeros(),
                },
                seed.clone(),
                b"bank".as_slice(),
                rand::random(),
            )
            .unwrap();
        let table = RowLutTable::new(
            vec![[vec![2.into()], vec![0.into(), 1.into()]], [vec![(-3).into()], vec![]]],
            None,
        )
        .unwrap();
        let material = scheme
            .sample_terminal_material(
                &secrets.payload,
                Some(&table),
                None,
                None,
                &bank,
                seed.clone(),
                b"table".as_slice(),
                rand::random(),
            )
            .unwrap();
        let masks = scheme.prepare_online_masks(&secrets.mask, &bank).unwrap();
        let mut keys = Vec::new();
        let mut encodings = Vec::new();
        let mut scalar_public = Vec::new();
        let mut scalar_rows = Vec::new();
        let mut aggregate_public: Option<crate::rns_bgg::RnsBggTerminalAccumulator<RnsBggMatrix>> =
            None;
        let mut aggregate_rows: Option<crate::rns_bgg::RnsBggTerminalAccumulator<RnsBggRow>> = None;
        let key_id = rand::random();
        for x in 0..2 {
            let key = scheme
                .sample_input_key(0, seed.clone(), format!("input-{x}").as_bytes(), key_id)
                .unwrap();
            let payload = ring.polynomial((0..=x).map(|i| BigInt::from(u8::from(i == x)).into()));
            let encoding = scheme
                .encode(
                    &secrets.mask,
                    &key,
                    &scheme.prepare_input(&secrets.payload, &key, payload).unwrap(),
                )
                .unwrap();
            let prepared = scheme.prepare_terminal(&key, &material).unwrap();
            let raw_public = prepared.public_accumulator();
            aggregate_public = Some(match aggregate_public {
                Some(sum) => sum.combine(raw_public).unwrap(),
                None => raw_public,
            });
            let raw_row = scheme.evaluate_terminal(&encoding, &prepared, &masks).unwrap();
            aggregate_rows = Some(match aggregate_rows {
                Some(sum) => sum.combine(raw_row).unwrap(),
                None => raw_row,
            });
            let public = scheme.finalize_terminal_public(prepared.public_accumulator()).unwrap();
            let value = scheme
                .finalize_terminal_encoding(
                    scheme.evaluate_terminal(&encoding, &prepared, &masks).unwrap(),
                    &public,
                )
                .unwrap();
            keys.push(key.matrix);
            encodings.push(encoding.row);
            scalar_public.push(public.matrix);
            scalar_rows.push(value.row().clone());
        }
        let range = FamilyRange::bounded(1, 2, 4).unwrap();
        let public_input = RnsBggPublicKeyFamily {
            matrices: ProgramValue::Family(Family::pack(keys).unwrap()),
            level: 0,
            id: key_id,
            context: scheme.context,
            range: range.clone(),
        };
        let encoding_input = RnsBggEncodingFamily {
            rows: ProgramValue::Family(Family::pack(encodings).unwrap()),
            level: 0,
            key_id,
            context: scheme.context,
            range,
        };
        let prepared = scheme
            .prepare_terminal_family(
                &public_input,
                &ProgramValue::Scalar(material.clone()),
                RnsBggTerminalFamilyMode::Independent,
                None,
            )
            .unwrap();
        let rows = scheme.evaluate_terminal_family(&encoding_input, &prepared, &masks).unwrap();
        let aggregate = scheme
            .prepare_terminal_family(
                &public_input,
                &ProgramValue::Scalar(material.clone()),
                RnsBggTerminalFamilyMode::Sum,
                None,
            )
            .unwrap();
        let aggregate_encoding =
            scheme.evaluate_terminal_family(&encoding_input, &aggregate, &masks).unwrap();
        let expected_public = scheme.finalize_terminal_public(aggregate_public.unwrap()).unwrap();
        let expected_row =
            scheme.finalize_terminal_encoding(aggregate_rows.unwrap(), &expected_public).unwrap();
        assert_eq!(aggregate.public_matrices().count(), &1.into());
        assert_eq!(aggregate_encoding.count(), &1.into());
        assert_ne!(prepared.id(), aggregate.id());
        let mut graph = DslContext::new("rns_bgg-terminal-family")
            .output("aggregate-public", aggregate.public_matrices().at(0) - expected_public.matrix)
            .unwrap()
            .output("aggregate-row", aggregate_encoding.at(0) - expected_row.row().clone())
            .unwrap();
        for (index, (public, row)) in scalar_public.into_iter().zip(scalar_rows).enumerate() {
            graph = graph
                .output(format!("public-{index}"), prepared.public_matrices().at(index) - public)
                .unwrap();
            graph = graph.output(format!("row-{index}"), rows.at(index) - row).unwrap();
        }
        let low = scheme.ring(1).unwrap();
        let transforms = Family::pack(vec![
            RnsBggMatrix::new([
                [low.identity(1), low.zero((1, 1))],
                [low.zero((1, 1)), low.identity(1)],
            ])
            .unwrap(),
            RnsBggMatrix::new([
                [low.zero((1, 1)), low.polynomial([0.into(), 17.into()])],
                [low.identity(1), low.zero((1, 1))],
            ])
            .unwrap(),
        ])
        .unwrap();
        let indexed_material = scheme
            .sample_terminal_material_family(
                &secrets.payload,
                &table,
                &transforms,
                &bank,
                seed.clone(),
                b"indexed-targets".as_slice(),
                rand::random(),
            )
            .unwrap();
        let indexed = scheme
            .prepare_terminal_family(
                &public_input,
                &indexed_material,
                RnsBggTerminalFamilyMode::Independent,
                None,
            )
            .unwrap();
        let indexed_rows =
            scheme.evaluate_terminal_family(&encoding_input, &indexed, &masks).unwrap();
        let indexed_sum = scheme
            .prepare_terminal_family(
                &public_input,
                &indexed_material,
                RnsBggTerminalFamilyMode::Sum,
                None,
            )
            .unwrap();
        let indexed_sum_rows =
            scheme.evaluate_terminal_family(&encoding_input, &indexed_sum, &masks).unwrap();
        let mut raw_public_sum: Option<crate::rns_bgg::RnsBggTerminalAccumulator<RnsBggMatrix>> =
            None;
        let mut raw_row_sum: Option<crate::rns_bgg::RnsBggTerminalAccumulator<RnsBggRow>> = None;
        for index in 0..2 {
            let key = RnsBggPublicKey {
                matrix: public_input.matrices.at(index),
                level: 0,
                id: key_id,
                context: scheme.context,
            };
            let encoding = RnsBggEncoding {
                row: encoding_input.rows.at(index),
                level: 0,
                key_id,
                context: scheme.context,
            };
            let scalar = scheme
                .prepare_terminal(&key, &indexed_material.at(mxx_dsl::Int::constant(index)))
                .unwrap();
            let raw_public = scalar.public_accumulator();
            let raw_row = scheme.evaluate_terminal(&encoding, &scalar, &masks).unwrap();
            raw_public_sum = Some(match raw_public_sum {
                Some(sum) => sum.combine(raw_public).unwrap(),
                None => raw_public,
            });
            raw_row_sum = Some(match raw_row_sum {
                Some(sum) => sum.combine(raw_row).unwrap(),
                None => raw_row,
            });
            let public = scheme.finalize_terminal_public(scalar.public_accumulator()).unwrap();
            let row = scheme
                .finalize_terminal_encoding(
                    scheme.evaluate_terminal(&encoding, &scalar, &masks).unwrap(),
                    &public,
                )
                .unwrap();
            graph = graph
                .output(
                    format!("indexed-public-{index}"),
                    indexed.public_matrices().at(index) - public.matrix,
                )
                .unwrap()
                .output(format!("indexed-row-{index}"), indexed_rows.at(index) - row.row().clone())
                .unwrap();
        }
        let public_sum = scheme.finalize_terminal_public(raw_public_sum.unwrap()).unwrap();
        let row_sum = scheme.finalize_terminal_encoding(raw_row_sum.unwrap(), &public_sum).unwrap();
        graph = graph
            .output("indexed-public-sum", indexed_sum.public_matrices().at(0) - public_sum.matrix)
            .unwrap()
            .output("indexed-row-sum", indexed_sum_rows.at(0) - row_sum.row().clone())
            .unwrap();
        (graph.build().unwrap().validate(&ParamEnv::default()).unwrap(), parameters)
    }
}

fn family_storage<T: GraphValue>(
    value: &ProgramValue<T, Family<T>>,
) -> ProgramValue<(), Vec<bool>> {
    match value {
        ProgramValue::Scalar(_) => ProgramValue::Scalar(()),
        ProgramValue::Family(values) => ProgramValue::Family(values.schema().shared),
    }
}

fn import_family_value<T: GraphValue>(
    element: T::Schema,
    count: IntExpr,
    storage: &ProgramValue<(), Vec<bool>>,
    production: ProductionId,
    name: impl Into<String>,
) -> Result<ProgramValue<T, Family<T>>, RnsBggError> {
    let schema = match storage {
        ProgramValue::Scalar(()) => ProgramValue::Scalar(element),
        ProgramValue::Family(shared) => {
            ProgramValue::Family(FamilyType { element, count, shared: shared.clone() })
        }
    };
    Ok(schema.artifact_input(production, name, ArtifactConfidentiality::Public)?)
}
