use super::*;

pub struct Subgraph<I: GraphValue, O: GraphValue> {
    handle: SubgraphHandle,
    input_schema: I::Schema,
    output_schema: O::Schema,
    captures: Vec<ValueHandle>,
    pending: Pending,
}

impl<I: GraphValue, O: GraphValue> Clone for Subgraph<I, O> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            input_schema: self.input_schema.clone(),
            output_schema: self.output_schema.clone(),
            captures: self.captures.clone(),
            pending: self.pending.clone(),
        }
    }
}

impl<I: GraphValue, O: GraphValue> Subgraph<I, O> {
    pub fn define(
        name: impl Into<String>,
        input_schema: I::Schema,
        body: impl FnOnce(I) -> O,
    ) -> Result<Self, DslError> {
        Self::try_define(name, input_schema, |inputs| Ok(body(inputs)))
    }

    pub fn try_define(
        name: impl Into<String>,
        input_schema: I::Schema,
        body: impl FnOnce(I) -> Result<O, DslError>,
    ) -> Result<Self, DslError> {
        let name = name.into();
        let (inputs, output, scope) =
            with_new_construction_scope(|scope| -> Result<_, DslError> {
                let inputs = input_schema.placeholders();
                let output = control::normalize(body(inputs.clone())?)?;
                Ok((inputs, output, scope))
            })?;
        let sealed = SubgraphHandle::seal(
            name,
            scope,
            inputs.flatten(),
            output.flatten(),
            &[],
            CapturePolicy::Lexical { parallel_index: None },
        )?;
        Ok(Self {
            handle: sealed.handle,
            input_schema,
            output_schema: output.schema(),
            captures: sealed.captures.iter().map(|capture| capture.outer.clone()).collect(),
            pending: output.pending().remap(&sealed.remap),
        })
    }

    pub fn call(&self, input: I) -> Result<O, DslError> {
        if input.schema() != self.input_schema {
            return Err(DslError::Schema);
        }
        let flattened = input.flatten();
        let input_count = flattened.len();
        self.call_flattened(flattened, input.pending(), vec![None; input_count])
    }

    /// Calls this subgraph with authoritative canonical coefficient bounds for
    /// its flattened arguments.  `Some(U)` means a constant-polynomial
    /// argument has canonical coefficients in `0..U`; `None` supplies no
    /// such contract.  The vector includes every argument, including a
    /// synthetic constant-one argument when the caller supplies one.
    pub fn call_with_canonical_input_exclusive_uppers(
        &self,
        input: I,
        canonical_input_exclusive_uppers: Vec<Option<BigUint>>,
    ) -> Result<O, DslError> {
        if input.schema() != self.input_schema {
            return Err(DslError::Schema);
        }
        let flattened = input.flatten();
        self.call_flattened(flattened, input.pending(), canonical_input_exclusive_uppers)
    }

    fn call_flattened(
        &self,
        flattened: Vec<ValueHandle>,
        input_pending: Pending,
        canonical_input_exclusive_uppers: Vec<Option<BigUint>>,
    ) -> Result<O, DslError> {
        if canonical_input_exclusive_uppers.len() != flattened.len() {
            return Err(DslError::CanonicalInputUpperCount);
        }
        if canonical_input_exclusive_uppers
            .iter()
            .any(|upper| upper.as_ref().is_some_and(|upper| upper == &BigUint::from(0u8)))
        {
            return Err(DslError::CanonicalInputUpperZero);
        }
        if canonical_input_exclusive_uppers.iter().zip(&flattened).any(|(upper, input)| {
            upper.is_some() && !matches!(input.wire_type(), WireType::Matrix(_))
        }) {
            return Err(DslError::CanonicalInputUpperNonMatrix);
        }
        let mut flattened = flattened;
        let mut canonical_input_exclusive_uppers = canonical_input_exclusive_uppers;
        flattened.extend(self.captures.iter().cloned());
        canonical_input_exclusive_uppers.extend((0..self.captures.len()).map(|_| None));
        let node = NodeHandle::subgraph_call(
            self.handle.clone(),
            flattened,
            Vec::new(),
            canonical_input_exclusive_uppers,
        );
        let values = (0..self.output_schema.wire_types().len())
            .map(|port| node.output(port as u32).expect("subgraph output"))
            .collect::<Vec<_>>();
        O::from_values(
            &self.output_schema,
            &values,
            Pending::merge([input_pending, self.pending.clone()]),
        )
    }
}
