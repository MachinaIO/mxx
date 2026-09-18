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
/// let private = builder.input(AttributeKnowledge::PrivatePayload)?;
/// let public = builder.input(AttributeKnowledge::PublicPayload)?;
/// let output = builder.mul_public(Operand::Scalar(private), Operand::Scalar(public))?;
/// let program = builder.finish(vec![Operand::Scalar(output)])?;
/// assert_eq!(program.outputs(), &[Operand::Scalar(output)]);
/// # Ok::<(), mxx_exponent_lut::program::ProgramValidationError>(())
/// ```
pub struct ExponentLutProgramBuilder {
    pub(crate) inputs: BTreeMap<WireId, ProgramInputDeclaration>,
    pub(crate) rhs_inputs: BTreeMap<RhsId, InputShape>,
    pub(crate) luts: Vec<LutTable>,
    pub(crate) bodies: Vec<Body>,
    pub(crate) wire_kinds: BTreeMap<WireId, bool>,
    pub(crate) rhs_kinds: BTreeMap<RhsId, bool>,
    pub(crate) next_wire: u32,
    pub(crate) next_rhs: u32,
}

impl ExponentLutProgramBuilder {
    pub fn new() -> Self {
        Self {
            inputs: BTreeMap::new(),
            rhs_inputs: BTreeMap::new(),
            luts: Vec::new(),
            bodies: vec![Body { steps: Vec::new(), outputs: Vec::new() }],
            wire_kinds: BTreeMap::new(),
            rhs_kinds: BTreeMap::new(),
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
        self.inputs
            .insert(id, ProgramInputDeclaration { shape: InputBinding::Scalar(()), knowledge });
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
        self.inputs
            .insert(id, ProgramInputDeclaration { shape: InputBinding::Family(count), knowledge });
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
        }
    }

    fn operand_rhs(&self, operand: &Operand<RhsId>) -> Result<bool, ProgramValidationError> {
        match operand {
            Operand::Scalar(id) => {
                self.rhs_kinds.get(id).copied().ok_or(ProgramValidationError::UndefinedRhs(*id))
            }
            Operand::Member { family, index } => {
                if self.rhs_kinds.get(family) != Some(&true) {
                    return Err(ProgramValidationError::ExpectedFamilyRhs(*family));
                }
                invalid_loop_slot(index, (self.bodies.len() - 1) as u32)
                    .map_or(Ok(false), |slot| Err(ProgramValidationError::InvalidLoopIndex(slot)))
            }
        }
    }

    fn output(&mut self, kind: bool, step: Step) -> Result<WireId, ProgramValidationError> {
        let id = match &step {
            Step::Add { output, .. } |
            Step::MulPublic { output, .. } |
            Step::MulPrivate { output, .. } |
            Step::Lut { output, .. } |
            Step::Parallel { output, .. } |
            Step::Sum { output, .. } => *output,
        };
        if !self.wire_kinds.contains_key(&id) {
            return Err(ProgramValidationError::UndefinedWire(id));
        }
        self.wire_kinds.insert(id, kind);
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
        let output = self.wire()?;
        self.output(false, Step::Add { lhs, rhs, output })
    }

    /// Append a public multiplication.  The right operand must be public;
    /// the left operand may have either knowledge class.
    pub fn mul_public(
        &mut self,
        lhs: Operand<WireId>,
        public: Operand<WireId>,
    ) -> Result<WireId, ProgramValidationError> {
        if self.operand_wire(&lhs)? || self.operand_wire(&public)? {
            return Err(ProgramValidationError::ExpectedScalarWire(WireId(0)));
        }
        let output = self.wire()?;
        self.output(false, Step::MulPublic { private: lhs, public, output })
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
        self.output(false, Step::MulPrivate { input, rhs, output })
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
        let output = self.wire()?;
        self.output(false, Step::Lut { input, table, output })
    }

    pub fn parallel(
        &mut self,
        range: FamilyRange,
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
        );
        self.bodies.push(Body { steps: Vec::new(), outputs: Vec::new() });
        let depth = (self.bodies.len() - 2) as u32;
        let result = body(self, IntExpr::LoopIndex(depth));
        let result = match result {
            Ok(Operand::Scalar(id)) => id,
            Ok(Operand::Member { family, .. }) => {
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
        // Child-local wires are lexical temporaries. Their IDs remain in the
        // child body, but cannot be captured by a later parent operation.
        self.wire_kinds = snapshot.4.clone();
        let output = match self.wire() {
            Ok(output) => output,
            Err(error) => {
                self.rollback(snapshot);
                return Err(error);
            }
        };
        self.wire_kinds.insert(output, true);
        self.bodies.last_mut().expect("parent body").steps.push(Step::Parallel {
            range,
            body: Box::new(child),
            output,
        });
        Ok(output)
    }

    fn rollback(
        &mut self,
        snapshot: (usize, u32, u32, usize, BTreeMap<WireId, bool>, BTreeMap<RhsId, bool>),
    ) {
        self.luts.truncate(snapshot.0);
        self.next_wire = snapshot.1;
        self.next_rhs = snapshot.2;
        self.bodies.truncate(snapshot.3);
        self.wire_kinds = snapshot.4;
        self.rhs_kinds = snapshot.5;
    }

    pub fn sum(&mut self, family: WireId) -> Result<WireId, ProgramValidationError> {
        if self.wire_kinds.get(&family) != Some(&true) {
            return Err(ProgramValidationError::ExpectedFamilyWire(family));
        }
        let output = self.wire()?;
        self.output(false, Step::Sum { family, output })
    }

    pub fn finish(
        mut self,
        outputs: Vec<Operand<WireId>>,
    ) -> Result<ExponentLutProgram, ProgramValidationError> {
        self.bodies[0].outputs = outputs;
        let mut program = ExponentLutProgram {
            id: ExponentLutProgramId::from_digest([0; 32]),
            inputs: self.inputs,
            rhs_inputs: self.rhs_inputs,
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
