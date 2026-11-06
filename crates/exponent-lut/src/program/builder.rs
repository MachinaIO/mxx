use super::*;

/// Mutable construction session for an immutable program.
///
/// The builder allocates all program identifiers and validates every operation
/// before [`finish`](Self::finish) produces the immutable program. Identifier
/// constructors and the builder's bookkeeping remain private.
///
/// ```
/// use mxx_exponent_lut::program::{AttributeKnowledge, ExponentLutProgramBuilder, Operand};
///
/// let mut builder = ExponentLutProgramBuilder::new();
/// let public = builder.input(AttributeKnowledge::PublicPayload)?;
/// let private = builder.input(AttributeKnowledge::PrivatePayload)?;
/// let output = builder.mul_public(Operand::Scalar(public), Operand::Scalar(private))?;
/// let program = builder.finish(vec![Operand::Scalar(output)])?;
/// assert_eq!(program.outputs(), &[Operand::Scalar(output)]);
/// # Ok::<(), mxx_exponent_lut::program::ProgramValidationError>(())
/// ```
pub struct ExponentLutProgramBuilder {
    pub(crate) inputs: BTreeMap<WireId, ProgramInputDeclaration>,
    pub(crate) rhs_inputs: BTreeMap<RhsId, InputShape>,
    rhs_names: BTreeMap<RhsId, String>,
    data_names: BTreeMap<DataId, String>,
    pub(crate) data: BTreeMap<DataId, DataFamily>,
    pub(crate) luts: Vec<LutTable>,
    pub(crate) bodies: Vec<Body>,
    pub(crate) wire_kinds: BTreeMap<WireId, bool>,
    pub(crate) rhs_kinds: BTreeMap<RhsId, bool>,
    wire_knowledge: BTreeMap<WireId, AttributeKnowledge>,
    pub(crate) next_wire: u32,
    pub(crate) next_rhs: u32,
}

impl ExponentLutProgramBuilder {
    pub fn new() -> Self {
        Self {
            inputs: BTreeMap::new(),
            rhs_inputs: BTreeMap::new(),
            rhs_names: BTreeMap::new(),
            data_names: BTreeMap::new(),
            data: BTreeMap::new(),
            luts: Vec::new(),
            bodies: vec![Body { steps: Vec::new(), outputs: Vec::new() }],
            wire_kinds: BTreeMap::new(),
            rhs_kinds: BTreeMap::new(),
            wire_knowledge: BTreeMap::new(),
            next_wire: 0,
            next_rhs: 0,
        }
    }

    fn wire(&mut self) -> Result<WireId, ProgramValidationError> {
        let id = WireId(self.next_wire);
        self.next_wire =
            self.next_wire.checked_add(1).ok_or(ProgramValidationError::IdExhausted)?;
        if self.wire_kinds.insert(id, false).is_some() {
            return Err(ProgramValidationError::DuplicateWire(id));
        }
        Ok(id)
    }

    fn rhs(&mut self) -> Result<RhsId, ProgramValidationError> {
        let id = RhsId(self.next_rhs);
        self.next_rhs = self.next_rhs.checked_add(1).ok_or(ProgramValidationError::IdExhausted)?;
        if self.rhs_kinds.insert(id, false).is_some() {
            return Err(ProgramValidationError::DuplicateRhs(id));
        }
        Ok(id)
    }

    pub fn input(
        &mut self,
        knowledge: AttributeKnowledge,
    ) -> Result<WireId, ProgramValidationError> {
        if self.bodies.len() != 1 {
            return Err(ProgramValidationError::InputInNestedBody);
        }
        let id = self.wire()?;
        self.inputs.insert(
            id,
            ProgramInputDeclaration {
                shape: InputBinding::Scalar(()),
                knowledge,
                name: None,
                outputs: Vec::new(),
            },
        );
        self.wire_knowledge.insert(id, knowledge);
        Ok(id)
    }

    pub fn input_family(
        &mut self,
        count: IntExpr,
        knowledge: AttributeKnowledge,
    ) -> Result<WireId, ProgramValidationError> {
        if self.bodies.len() != 1 {
            return Err(ProgramValidationError::InputInNestedBody);
        }
        let id = self.wire()?;
        FamilyRange::full(count.clone())?;
        self.wire_kinds.insert(id, true);
        self.inputs.insert(
            id,
            ProgramInputDeclaration {
                shape: InputBinding::Family(count),
                knowledge,
                name: None,
                outputs: Vec::new(),
            },
        );
        self.wire_knowledge.insert(id, knowledge);
        Ok(id)
    }

    pub fn rhs_input(&mut self) -> Result<RhsId, ProgramValidationError> {
        if self.bodies.len() != 1 {
            return Err(ProgramValidationError::InputInNestedBody);
        }
        let id = self.rhs()?;
        self.rhs_inputs.insert(id, InputBinding::Scalar(()));
        Ok(id)
    }

    pub fn rhs_family(&mut self, count: IntExpr) -> Result<RhsId, ProgramValidationError> {
        if self.bodies.len() != 1 {
            return Err(ProgramValidationError::InputInNestedBody);
        }
        FamilyRange::full(count.clone())?;
        let id = self.rhs()?;
        self.rhs_kinds.insert(id, true);
        self.rhs_inputs.insert(id, InputBinding::Family(count));
        Ok(id)
    }

    /// Declare a family of `count` integers supplied at run time, which selects RHS members
    /// ([`Operand::Selected`]) and holds LUT values ([`LutTable::from_data`]). Lowerings read it
    /// as an integer family input; the Program identity covers its count and bound, not its
    /// values. With a `bound`, every entry is below it, and the family may also select members
    /// of a wire family of at least `bound` members. The count may exceed `usize` for graph-only
    /// declarations; lowering its arithmetic uses arbitrary-precision integer expressions.
    pub fn data_family(
        &mut self,
        count: impl TryInto<u128>,
        bound: Option<usize>,
    ) -> Result<DataId, ProgramValidationError> {
        if self.bodies.len() != 1 {
            return Err(ProgramValidationError::InputInNestedBody);
        }
        let count = count.try_into().map_err(|_| ProgramValidationError::InvalidFamilyRange)?;
        if count == 0 || bound == Some(0) {
            return Err(ProgramValidationError::InvalidFamilyRange);
        }
        let id = DataId(
            u32::try_from(self.data.len()).map_err(|_| ProgramValidationError::IdExhausted)?,
        );
        self.data.insert(id, DataFamily { count, bound });
        Ok(id)
    }

    pub fn register_lut(&mut self, table: LutTable) -> Result<LutId, ProgramValidationError> {
        validate_table(&table)?;
        let id =
            LutId(u32::try_from(self.luts.len()).map_err(|_| ProgramValidationError::IdExhausted)?);
        self.luts.push(table);
        Ok(id)
    }

    /// Register a binary convenience table as the unary table consumed by the
    /// Program. Values use index `lhs + lhs_width * rhs`, with the RHS domain
    /// outermost. The explicit input layout is retained in Program metadata.
    pub fn register_binary_lut(
        &mut self,
        lhs_width: usize,
        rhs_width: usize,
        input_layout: ExpLayout,
        output_width: usize,
        values: Vec<usize>,
        output_representation: LutOutputRepresentation,
    ) -> Result<LutId, ProgramValidationError> {
        let input_width =
            lhs_width.checked_mul(rhs_width).ok_or(ProgramValidationError::InvalidLut)?;
        self.register_lut(LutTable::new(
            input_width,
            input_layout,
            output_width,
            values,
            output_representation,
        )?)
    }

    fn operand_wire(&self, operand: &Operand<WireId>) -> Result<bool, ProgramValidationError> {
        match operand {
            Operand::Scalar(id) => {
                self.wire_kinds.get(id).copied().ok_or(ProgramValidationError::UndefinedWire(*id))
            }
            Operand::Member { family, index } => {
                if self.wire_kinds.get(family) != Some(&true) {
                    return Err(ProgramValidationError::ExpectedFamilyWire(*family));
                }
                invalid_loop_slot(index, (self.bodies.len() - 1) as u32)
                    .map_or(Ok(false), |slot| Err(ProgramValidationError::InvalidLoopIndex(slot)))
            }
            Operand::Selected { family, data, position } => {
                if self.wire_kinds.get(family) != Some(&true) {
                    return Err(ProgramValidationError::ExpectedFamilyWire(*family));
                }
                if self.data.get(data).and_then(DataFamily::bound).is_none() {
                    return Err(ProgramValidationError::InvalidStructure(
                        "a wire member is selected by a bounded integer data family".into(),
                    ));
                }
                invalid_loop_slot(position, (self.bodies.len() - 1) as u32)
                    .map_or(Ok(false), |slot| Err(ProgramValidationError::InvalidLoopIndex(slot)))
            }
        }
    }

    fn operand_rhs(&self, operand: &Operand<RhsId>) -> Result<bool, ProgramValidationError> {
        match operand {
            Operand::Scalar(id) => {
                self.rhs_kinds.get(id).copied().ok_or(ProgramValidationError::UndefinedRhs(*id))
            }
            Operand::Member { family, index: position } |
            Operand::Selected { family, position, .. } => {
                if self.rhs_kinds.get(family) != Some(&true) {
                    return Err(ProgramValidationError::ExpectedFamilyRhs(*family));
                }
                if let Operand::Selected { data, .. } = operand &&
                    !self.data.contains_key(data)
                {
                    return Err(ProgramValidationError::InvalidStructure(
                        "undeclared integer data family".into(),
                    ));
                }
                invalid_loop_slot(position, (self.bodies.len() - 1) as u32)
                    .map_or(Ok(false), |slot| Err(ProgramValidationError::InvalidLoopIndex(slot)))
            }
        }
    }

    fn operand_knowledge(
        &self,
        operand: &Operand<WireId>,
    ) -> Result<AttributeKnowledge, ProgramValidationError> {
        let wire = operand.source();
        self.wire_knowledge.get(wire).copied().ok_or(ProgramValidationError::UndefinedWire(*wire))
    }

    fn output(
        &mut self,
        kind: bool,
        knowledge: AttributeKnowledge,
        step: Step,
    ) -> Result<WireId, ProgramValidationError> {
        let id = match &step {
            Step::Add { output, .. } |
            Step::MulPublic { output, .. } |
            Step::MulPrivate { output, .. } |
            Step::Lut { output, .. } |
            Step::MulSmallConstant { output, .. } |
            Step::Parallel { output, .. } |
            Step::Fold { output, .. } |
            Step::Select { output, .. } |
            Step::Sum { output, .. } => *output,
        };
        if !self.wire_kinds.contains_key(&id) {
            return Err(ProgramValidationError::UndefinedWire(id));
        }
        self.wire_kinds.insert(id, kind);
        self.wire_knowledge.insert(id, knowledge);
        self.bodies.last_mut().expect("root body").steps.push(step);
        Ok(id)
    }

    pub fn add(
        &mut self,
        lhs: Operand<WireId>,
        rhs: Operand<WireId>,
    ) -> Result<WireId, ProgramValidationError> {
        if self.operand_wire(&lhs)? || self.operand_wire(&rhs)? {
            return Err(ProgramValidationError::ExpectedScalarWire(WireId(0)));
        }
        let lhs_knowledge = self.operand_knowledge(&lhs)?;
        if lhs_knowledge != self.operand_knowledge(&rhs)? {
            return Err(ProgramValidationError::InvalidStructure(
                "Add operands must have matching knowledge".into(),
            ));
        }
        let output = self.wire()?;
        self.output(false, lhs_knowledge, Step::Add { lhs, rhs, output })
    }

    /// Append a public-left multiplication. The left operand must be public;
    /// the right operand may be public or private, and determines the output
    /// knowledge.
    pub fn mul_public(
        &mut self,
        lhs: Operand<WireId>,
        rhs: Operand<WireId>,
    ) -> Result<WireId, ProgramValidationError> {
        self.mul_public_scaled(lhs, rhs, 0)
    }

    /// Append a public-left multiplication whose public LHS payload carries the public scale
    /// `2^lhs_scale_log2`: the LHS payload must be `2^k` times a signed monomial (`k =
    /// lhs_scale_log2`), which the caller guarantees and nothing checks. The product is
    /// `c_lhs G^-1(2^-k A_rhs) + 2^-k p_lhs c_rhs`, so the output payload is `2^-k p_lhs p_rhs`
    /// and the RHS error is multiplied by the signed monomial `2^-k p_lhs` alone, not by the
    /// scale; the LHS error receives the usual decomposition gain. With `k = 0` this is
    /// [`Self::mul_public`].
    pub fn mul_public_scaled(
        &mut self,
        lhs: Operand<WireId>,
        rhs: Operand<WireId>,
        lhs_scale_log2: u32,
    ) -> Result<WireId, ProgramValidationError> {
        self.mul_public_by_scale(lhs, rhs, PublicScale::constant(lhs_scale_log2))
    }

    /// [`Self::mul_public_scaled`] with the scale exponent `scale.log2`, an integer expression of
    /// the enclosing loop indices and of variables the lowering binds
    /// ([`super::ProgramLoweringContext::variables`]), at most `scale.max`. The caller guarantees
    /// the LHS payload is `2^k` times a signed monomial for every value `k` takes, and that `k`
    /// is never zero unless it is the constant zero.
    pub fn mul_public_by_scale(
        &mut self,
        lhs: Operand<WireId>,
        rhs: Operand<WireId>,
        scale: PublicScale,
    ) -> Result<WireId, ProgramValidationError> {
        if self.operand_wire(&lhs)? || self.operand_wire(&rhs)? {
            return Err(ProgramValidationError::ExpectedScalarWire(WireId(0)));
        }
        if self.operand_knowledge(&lhs)? != AttributeKnowledge::PublicPayload {
            return Err(ProgramValidationError::InvalidStructure(
                "MulPublic requires a public exponent LHS".into(),
            ));
        }
        let rhs_knowledge = self.operand_knowledge(&rhs)?;
        let output = self.wire()?;
        let (lhs_scale_log2, lhs_scale) = match scale.as_constant() {
            Some(log2) => (log2, None),
            None => (scale.max, Some(scale.log2.canonicalize())),
        };
        self.output(
            false,
            rhs_knowledge,
            Step::MulPublic { lhs, rhs, output, lhs_scale_log2, lhs_scale },
        )
    }

    /// Append `input * c(X) X^shift` for the fixed small polynomial `c` with
    /// coefficients `coefficients` (`c_i` at `X^i`, at least one nonzero).
    /// `shift` may read the enclosing loop indices, so a family body can
    /// place each member at its own power of `X`. The output keeps the
    /// input's knowledge and representation; a coefficient bound grows by
    /// `||c||_1`. An exponent input is a signed monomial only when `c` has a
    /// single coefficient of magnitude one, which LUT consumers rely on.
    pub fn mul_small_constant(
        &mut self,
        input: Operand<WireId>,
        coefficients: Vec<i64>,
        shift: IntExpr,
    ) -> Result<WireId, ProgramValidationError> {
        if self.operand_wire(&input)? {
            return Err(ProgramValidationError::ExpectedScalarWire(WireId(0)));
        }
        super::check_small_constant(&coefficients, &shift, (self.bodies.len() - 1) as u32)?;
        let knowledge = self.operand_knowledge(&input)?;
        let output = self.wire()?;
        self.output(false, knowledge, Step::MulSmallConstant { input, coefficients, shift, output })
    }

    pub fn mul_private(
        &mut self,
        input: Operand<WireId>,
        rhs: Operand<RhsId>,
    ) -> Result<WireId, ProgramValidationError> {
        if self.operand_wire(&input)? {
            return Err(ProgramValidationError::ExpectedScalarWire(WireId(0)));
        }
        if self.operand_rhs(&rhs)? {
            return Err(ProgramValidationError::ExpectedScalarRhs(RhsId(0)));
        }
        let output = self.wire()?;
        self.output(
            false,
            AttributeKnowledge::PrivatePayload,
            Step::MulPrivate { input, rhs, output },
        )
    }

    pub fn lut(
        &mut self,
        input: Operand<WireId>,
        table: LutId,
    ) -> Result<WireId, ProgramValidationError> {
        if self.operand_wire(&input)? {
            return Err(ProgramValidationError::ExpectedScalarWire(WireId(0)));
        }
        if table.0 as usize >= self.luts.len() {
            return Err(ProgramValidationError::InvalidLutId(table));
        }
        let input_knowledge = self.operand_knowledge(&input)?;
        let output = self.wire()?;
        self.output(false, input_knowledge, Step::Lut { input, table, output })
    }

    pub fn parallel(
        &mut self,
        range: FamilyRange,
        body: impl FnOnce(&mut Self, IntExpr) -> Result<Operand<WireId>, ProgramValidationError>,
    ) -> Result<WireId, ProgramValidationError> {
        self.parallel_with(range, false, body)
    }

    /// [`Self::parallel`] for a family that is always lowered once as a graph family. A caller
    /// that lowers Program families where their members are read (`lower_program_deferred`, for
    /// sampler families read one block at a time) re-lowers a member at every read; a family
    /// that many members of later families read, such as a round of a hash, must instead be
    /// materialized once. A root materialized family can also be exported to the caller
    /// (`lower_program_deferred_exporting`).
    pub fn parallel_materialized(
        &mut self,
        range: FamilyRange,
        body: impl FnOnce(&mut Self, IntExpr) -> Result<Operand<WireId>, ProgramValidationError>,
    ) -> Result<WireId, ProgramValidationError> {
        self.parallel_with(range, true, body)
    }

    fn parallel_with(
        &mut self,
        range: FamilyRange,
        materialized: bool,
        body: impl FnOnce(&mut Self, IntExpr) -> Result<Operand<WireId>, ProgramValidationError>,
    ) -> Result<WireId, ProgramValidationError> {
        validate_range(&range)?;
        let snapshot = (
            self.luts.len(),
            self.next_wire,
            self.next_rhs,
            self.bodies.len(),
            self.wire_kinds.clone(),
            self.rhs_kinds.clone(),
            self.wire_knowledge.clone(),
        );
        self.bodies.push(Body { steps: Vec::new(), outputs: Vec::new() });
        let depth = (self.bodies.len() - 2) as u32;
        let result = body(self, IntExpr::LoopIndex(depth));
        let result = match result {
            Ok(Operand::Scalar(id)) => id,
            Ok(Operand::Member { family, .. } | Operand::Selected { family, .. }) => {
                self.rollback(snapshot);
                return Err(ProgramValidationError::ExpectedScalarWire(family));
            }
            Err(error) => {
                self.rollback(snapshot);
                return Err(error);
            }
        };
        let mut child = self.bodies.pop().expect("child body");
        child.outputs = vec![Operand::Scalar(result)];
        let child_knowledge = self.operand_knowledge(&Operand::Scalar(result))?;
        // Child-local wires are lexical temporaries. Their IDs remain in the
        // child body, but cannot be captured by a later parent operation.
        self.wire_kinds = snapshot.4.clone();
        self.wire_knowledge = snapshot.6.clone();
        let output = match self.wire() {
            Ok(output) => output,
            Err(error) => {
                self.rollback(snapshot);
                return Err(error);
            }
        };
        self.wire_kinds.insert(output, true);
        self.wire_knowledge.insert(output, child_knowledge);
        self.bodies.last_mut().expect("parent body").steps.push(Step::Parallel {
            range,
            body: Box::new(child),
            output,
            materialized,
        });
        Ok(output)
    }

    /// Append `count` iterations from `initial`: `body` receives the accumulator (the previous
    /// value) and the iteration index `k` and returns the next value, a scalar of the
    /// accumulator's knowledge. Operands the body reads per iteration are chosen with
    /// [`Self::select`]; a chain of `count` steps is then one loop body.
    pub fn fold(
        &mut self,
        count: usize,
        initial: Operand<WireId>,
        body: impl FnOnce(
            &mut Self,
            Operand<WireId>,
            IntExpr,
        ) -> Result<Operand<WireId>, ProgramValidationError>,
    ) -> Result<WireId, ProgramValidationError> {
        if self.operand_wire(&initial)? {
            return Err(ProgramValidationError::ExpectedScalarWire(*initial.source()));
        }
        let knowledge = self.operand_knowledge(&initial)?;
        let snapshot = (
            self.luts.len(),
            self.next_wire,
            self.next_rhs,
            self.bodies.len(),
            self.wire_kinds.clone(),
            self.rhs_kinds.clone(),
            self.wire_knowledge.clone(),
        );
        let accumulator = self.wire()?;
        self.wire_kinds.insert(accumulator, false);
        self.wire_knowledge.insert(accumulator, knowledge);
        self.bodies.push(Body { steps: Vec::new(), outputs: Vec::new() });
        let depth = (self.bodies.len() - 2) as u32;
        let result = match body(self, Operand::Scalar(accumulator), IntExpr::LoopIndex(depth)) {
            Ok(Operand::Scalar(id))
                if self.operand_knowledge(&Operand::Scalar(id)) == Ok(knowledge) =>
            {
                id
            }
            Ok(_) => {
                self.rollback(snapshot);
                return Err(ProgramValidationError::InvalidStructure(
                    "a Fold body returns a scalar of its accumulator's knowledge".into(),
                ));
            }
            Err(error) => {
                self.rollback(snapshot);
                return Err(error);
            }
        };
        let mut child = self.bodies.pop().expect("child body");
        child.outputs = vec![Operand::Scalar(result)];
        self.wire_kinds = snapshot.4.clone();
        self.wire_knowledge = snapshot.6.clone();
        let output = match self.wire() {
            Ok(output) => output,
            Err(error) => {
                self.rollback(snapshot);
                return Err(error);
            }
        };
        self.output(
            false,
            knowledge,
            Step::Fold { count, initial, accumulator, body: Box::new(child), output },
        )
    }

    /// The candidate at index `selector` of `candidates`, scalars of one knowledge and
    /// representation; `selector` may read the enclosing loop indices.
    pub fn select(
        &mut self,
        selector: IntExpr,
        candidates: Vec<Operand<WireId>>,
    ) -> Result<WireId, ProgramValidationError> {
        let first = candidates.first().ok_or_else(|| {
            ProgramValidationError::InvalidStructure("a Select needs a candidate".into())
        })?;
        let knowledge = self.operand_knowledge(first)?;
        for candidate in &candidates {
            if self.operand_wire(candidate)? {
                return Err(ProgramValidationError::ExpectedScalarWire(*candidate.source()));
            }
            if self.operand_knowledge(candidate)? != knowledge {
                return Err(ProgramValidationError::InvalidStructure(
                    "Select candidates share knowledge".into(),
                ));
            }
        }
        let output = self.wire()?;
        self.output(
            false,
            knowledge,
            Step::Select { selector: selector.canonicalize(), candidates, output },
        )
    }

    fn rollback(
        &mut self,
        snapshot: (
            usize,
            u32,
            u32,
            usize,
            BTreeMap<WireId, bool>,
            BTreeMap<RhsId, bool>,
            BTreeMap<WireId, AttributeKnowledge>,
        ),
    ) {
        self.luts.truncate(snapshot.0);
        self.next_wire = snapshot.1;
        self.next_rhs = snapshot.2;
        self.bodies.truncate(snapshot.3);
        self.wire_kinds = snapshot.4;
        self.rhs_kinds = snapshot.5;
        self.wire_knowledge = snapshot.6;
    }

    pub fn sum(&mut self, family: WireId) -> Result<WireId, ProgramValidationError> {
        self.sum_columns(family, 1)
    }

    /// The `1 x columns` row of the block sums of `family` (see
    /// [`super::Step::Sum`]): column `j` sums the members `[j b, (j + 1) b)`
    /// of a family of `columns b` members; one column is [`Self::sum`].
    pub fn sum_columns(
        &mut self,
        family: WireId,
        columns: usize,
    ) -> Result<WireId, ProgramValidationError> {
        if self.wire_kinds.get(&family) != Some(&true) {
            return Err(ProgramValidationError::ExpectedFamilyWire(family));
        }
        let knowledge = self
            .wire_knowledge
            .get(&family)
            .copied()
            .ok_or(ProgramValidationError::UndefinedWire(family))?;
        let output = self.wire()?;
        self.output(false, knowledge, Step::Sum { family, output, columns })
    }

    /// Attach the external/root binding to an existing typed Program input.
    pub fn name_input(
        &mut self,
        input: WireId,
        name: impl Into<String>,
    ) -> Result<(), ProgramValidationError> {
        self.inputs.get_mut(&input).ok_or(ProgramValidationError::UndefinedWire(input))?.name =
            Some(name.into());
        Ok(())
    }
    /// Record the output connections of an input without an application-side output table.
    pub fn bind_outputs(
        &mut self,
        input: WireId,
        outputs: Vec<OutputId>,
    ) -> Result<(), ProgramValidationError> {
        self.inputs.get_mut(&input).ok_or(ProgramValidationError::UndefinedWire(input))?.outputs =
            outputs;
        Ok(())
    }
    /// Name a typed RHS at its external binding boundary. Names must be unique.
    pub fn name_rhs(
        &mut self,
        input: RhsId,
        name: impl Into<String>,
    ) -> Result<(), ProgramValidationError> {
        if !self.rhs_inputs.contains_key(&input) {
            return Err(ProgramValidationError::UndefinedRhs(input));
        }
        self.rhs_names.insert(input, name.into());
        Ok(())
    }
    /// Name a typed integer-data family at its external binding boundary.
    pub fn name_data(
        &mut self,
        input: DataId,
        name: impl Into<String>,
    ) -> Result<(), ProgramValidationError> {
        if !self.data.contains_key(&input) {
            return Err(ProgramValidationError::InvalidFamilyRange);
        }
        self.data_names.insert(input, name.into());
        Ok(())
    }

    /// Declare a root output and return its typed connection handle.
    /// Validation is performed when the complete Program is frozen. Any outputs
    /// supplied to `finish` follow these declarations, preserving existing callers.
    pub fn declare_output(
        &mut self,
        output: Operand<WireId>,
    ) -> Result<OutputId, ProgramValidationError> {
        let id = u32::try_from(self.bodies[0].outputs.len())
            .map_err(|_| ProgramValidationError::IdExhausted)?;
        self.bodies[0].outputs.push(output);
        Ok(OutputId(id))
    }

    pub fn finish(
        mut self,
        outputs: Vec<Operand<WireId>>,
    ) -> Result<ExponentLutProgram, ProgramValidationError> {
        self.bodies[0].outputs.extend(outputs);
        let mut program = ExponentLutProgram {
            id: ExponentLutProgramId::from_digest([0; 32]),
            inputs: self.inputs,
            data: self.data,
            rhs_inputs: self.rhs_inputs,
            rhs_names: self.rhs_names,
            data_names: self.data_names,
            luts: self.luts,
            body: self.bodies.remove(0),
            metadata_cache: std::sync::OnceLock::new(),
        };
        validate_program(&program)?;
        program.id = canonical_program_id(&program);
        Ok(program)
    }
}

fn validate_range(range: &FamilyRange) -> Result<(), ProgramValidationError> {
    super::validate_range(range)
}
fn invalid_loop_slot(expression: &IntExpr, depth: u32) -> Option<u32> {
    super::invalid_loop_slot(expression, depth)
}
