use super::*;
use std::{error::Error, time::Instant};

/// Lowering progress, reported through `tracing`: long lowerings (a large
/// noise simulation, for instance) show how many steps they have visited.
struct LoweringProgress {
    steps: u64,
    started: Instant,
}

impl LoweringProgress {
    const REPORT_INTERVAL: u64 = 10_000;

    fn step(&mut self, depth: usize) {
        self.steps += 1;
        if self.steps.is_multiple_of(Self::REPORT_INTERVAL) {
            tracing::debug!(
                steps = self.steps,
                depth,
                elapsed_ms = self.started.elapsed().as_millis() as u64,
                "program lowering progress"
            );
        }
    }
}

pub trait ExponentLUTProgramLowering {
    type Wire: Clone;
    type Rhs: Clone;
    type Error: Error + Send + Sync + 'static;

    fn add(&mut self, lhs: &Self::Wire, rhs: &Self::Wire) -> Result<Self::Wire, Self::Error>;
    /// Multiply the public exponent left operand by the exponent right
    /// operand. The right operand may be public or private; the result keeps
    /// the right operand's knowledge. The argument order is part of the
    /// program contract and is intentionally not commutative because prepared
    /// actions and noise accounting attach to the public left operand.
    fn mul_public(&mut self, lhs: &Self::Wire, rhs: &Self::Wire)
    -> Result<Self::Wire, Self::Error>;
    fn mul_private(
        &mut self,
        input: &Self::Wire,
        rhs: &Self::Rhs,
    ) -> Result<Self::Wire, Self::Error>;
    fn lut(&mut self, input: &Self::Wire, table: &LutTable) -> Result<Self::Wire, Self::Error>;
    /// Multiply by the fixed small polynomial `c(X) X^shift` (`c_i` at
    /// `X^i`); `shift` has the enclosing loop indices substituted.
    fn mul_small_constant(
        &mut self,
        input: &Self::Wire,
        coefficients: &[i64],
        shift: &IntExpr,
    ) -> Result<Self::Wire, Self::Error>;
}

#[derive(Clone, Debug)]
pub struct LoweringSite {
    pub output: WireId,
    pub indices: Vec<IntExpr>,
    /// Validated lexical ranges corresponding one-to-one with `indices`.
    pub ranges: Vec<FamilyRange>,
}

impl LoweringSite {
    pub fn family_scope(&self) -> Result<FamilyScope, ProgramValidationError> {
        FamilyScope::new(self.ranges.clone(), self.indices.clone())
    }
}

pub trait ProgramLoweringContext: ExponentLUTProgramLowering {
    fn replace_site(&mut self, site: Option<LoweringSite>) -> Option<LoweringSite>;
}

/// The values bound to a Program's inputs, RHS inputs and runtime tables
/// while it is lowered. `T` is the family execution's table representation.
pub struct Environment<W, R, WF, RF, T> {
    pub(crate) wires: BTreeMap<WireId, InputBinding<W, WF>>,
    pub(crate) rhs: BTreeMap<RhsId, InputBinding<R, RF>>,
    pub(crate) tables: BTreeMap<TableId, T>,
}

enum LoweredValue<W, F> {
    Scalar(W),
    Family(F),
}

impl<W, R, WF, RF, T> Environment<W, R, WF, RF, T> {
    pub fn new() -> Self {
        Self { wires: BTreeMap::new(), rhs: BTreeMap::new(), tables: BTreeMap::new() }
    }
    pub fn bind_wire(
        &mut self,
        id: WireId,
        value: InputBinding<W, WF>,
    ) -> Result<(), ProgramValidationError> {
        if self.wires.insert(id, value).is_some() {
            return Err(ProgramValidationError::DuplicateBinding);
        }
        Ok(())
    }
    pub fn bind_rhs(
        &mut self,
        id: RhsId,
        value: InputBinding<R, RF>,
    ) -> Result<(), ProgramValidationError> {
        if self.rhs.insert(id, value).is_some() {
            return Err(ProgramValidationError::DuplicateBinding);
        }
        Ok(())
    }
    /// Bind the entries of a declared runtime table.
    pub fn bind_table(&mut self, id: TableId, value: T) -> Result<(), ProgramValidationError> {
        if self.tables.insert(id, value).is_some() {
            return Err(ProgramValidationError::DuplicateBinding);
        }
        Ok(())
    }
}

pub trait FamilyExecution<E>
where
    E: ExponentLUTProgramLowering + ProgramLoweringContext,
{
    type WireFamily: Clone;
    type RhsFamily: Clone;
    /// A bound runtime integer table.
    type Table: Clone;
    /// Member `index` of `family`. The evaluator observes the read (the noise
    /// evaluator records the read's payload support).
    fn wire_at(
        &mut self,
        evaluator: &mut E,
        family: &Self::WireFamily,
        index: &IntExpr,
    ) -> Result<E::Wire, ProgramLowerError<E::Error>>;
    /// Member `table[index]` of `family` ([`Operand::Lookup`]): the member
    /// is a runtime value, so an execution that needs the member statically
    /// treats it as unknown.
    fn wire_at_table(
        &mut self,
        evaluator: &mut E,
        family: &Self::WireFamily,
        table: &Self::Table,
        index: &IntExpr,
    ) -> Result<E::Wire, ProgramLowerError<E::Error>>;
    fn rhs_at(
        &mut self,
        family: &Self::RhsFamily,
        index: &IntExpr,
    ) -> Result<E::Rhs, ProgramLowerError<E::Error>>;
    fn parallel(
        &mut self,
        evaluator: &mut E,
        range: &FamilyRange,
        body: impl FnOnce(&mut Self, &mut E, IntExpr) -> Result<E::Wire, ProgramLowerError<E::Error>>,
    ) -> Result<Self::WireFamily, ProgramLowerError<E::Error>>;
    /// The sum of the members of `family`, or with `columns > 1` the row of
    /// its block sums ([`super::Step::Sum`]).
    fn sum(
        &mut self,
        evaluator: &mut E,
        family: &Self::WireFamily,
        columns: usize,
    ) -> Result<E::Wire, ProgramLowerError<E::Error>>;
}

#[derive(Debug, Error)]
pub enum ProgramLowerError<E: Error + Send + Sync + 'static> {
    #[error("program validation failed: {0}")]
    Program(#[from] ProgramValidationError),
    #[error("DSL graph construction failed: {0}")]
    Dsl(#[from] mxx_dsl::DslError),
    #[error("program lowering backend failed: {0}")]
    Backend(#[source] E),
    #[error("unsupported program structure: {reason}")]
    UnsupportedStructure { reason: String },
}

pub(crate) fn lower_program<E, F>(
    program: &ExponentLutProgram,
    evaluator: &mut E,
    families: &mut F,
    environment: &mut Environment<E::Wire, E::Rhs, F::WireFamily, F::RhsFamily, F::Table>,
) -> Result<Vec<E::Wire>, ProgramLowerError<E::Error>>
where
    E: ExponentLUTProgramLowering + ProgramLoweringContext,
    F: FamilyExecution<E>,
{
    validate_program(program)?;
    if program.inputs.keys().any(|id| !environment.wires.contains_key(id)) ||
        program.rhs_inputs.keys().any(|id| !environment.rhs.contains_key(id)) ||
        environment.wires.keys().any(|id| !program.inputs.contains_key(id)) ||
        environment.rhs.keys().any(|id| !program.rhs_inputs.contains_key(id)) ||
        program.tables.keys().any(|id| !environment.tables.contains_key(id)) ||
        environment.tables.keys().any(|id| !program.tables.contains_key(id))
    {
        return Err(ProgramLowerError::Program(ProgramValidationError::MissingBinding));
    }
    for (id, declaration) in &program.inputs {
        let actual_family = matches!(environment.wires.get(id), Some(InputBinding::Family(_)));
        let declared_family = matches!(declaration.shape(), InputBinding::Family(_));
        if actual_family != declared_family {
            return Err(ProgramLowerError::Program(if declared_family {
                ProgramValidationError::ExpectedFamilyWire(*id)
            } else {
                ProgramValidationError::ExpectedScalarWire(*id)
            }));
        }
    }
    for (id, declaration) in &program.rhs_inputs {
        let actual_family = matches!(environment.rhs.get(id), Some(InputBinding::Family(_)));
        let declared_family = matches!(declaration, InputBinding::Family(_));
        if actual_family != declared_family {
            return Err(ProgramLowerError::Program(if declared_family {
                ProgramValidationError::ExpectedFamilyRhs(*id)
            } else {
                ProgramValidationError::ExpectedScalarRhs(*id)
            }));
        }
    }
    let mut indices = Vec::new();
    let mut ranges = Vec::new();
    let mut progress = LoweringProgress { steps: 0, started: Instant::now() };
    tracing::debug!(steps = program.body.steps.len(), "program lowering started");
    let values = lower_body(
        program,
        &program.body,
        evaluator,
        families,
        environment,
        &mut indices,
        &mut ranges,
        &mut progress,
    )?;
    tracing::debug!(
        steps = progress.steps,
        elapsed_ms = progress.started.elapsed().as_millis() as u64,
        "program lowering finished"
    );
    Ok(values)
}

fn lower_body<E, F>(
    program: &ExponentLutProgram,
    body: &Body,
    evaluator: &mut E,
    families: &mut F,
    environment: &mut Environment<E::Wire, E::Rhs, F::WireFamily, F::RhsFamily, F::Table>,
    indices: &mut Vec<IntExpr>,
    ranges: &mut Vec<FamilyRange>,
    progress: &mut LoweringProgress,
) -> Result<Vec<E::Wire>, ProgramLowerError<E::Error>>
where
    E: ExponentLUTProgramLowering + ProgramLoweringContext,
    F: FamilyExecution<E>,
{
    let mut direct = Vec::new();
    let releases = last_reads(body);
    let result = (|| -> Result<Vec<E::Wire>, ProgramLowerError<E::Error>> {
        for (position, step) in body.steps.iter().enumerate() {
            progress.step(indices.len());
            let output = step_output(step);
            let site = LoweringSite { output, indices: indices.clone(), ranges: ranges.clone() };
            let value = match step {
                Step::Add { lhs, rhs, .. } => {
                    let lhs = resolve_wire(lhs, evaluator, families, environment, indices)?;
                    let rhs = resolve_wire(rhs, evaluator, families, environment, indices)?;
                    LoweredValue::Scalar(with_site(evaluator, site, |evaluator| {
                        evaluator.add(&lhs, &rhs).map_err(ProgramLowerError::Backend)
                    })?)
                }
                Step::MulPublic { lhs, rhs, .. } => {
                    let lhs = resolve_wire(lhs, evaluator, families, environment, indices)?;
                    let rhs = resolve_wire(rhs, evaluator, families, environment, indices)?;
                    LoweredValue::Scalar(with_site(evaluator, site, |evaluator| {
                        evaluator.mul_public(&lhs, &rhs).map_err(ProgramLowerError::Backend)
                    })?)
                }
                Step::MulPrivate { input, rhs, .. } => {
                    let input = resolve_wire(input, evaluator, families, environment, indices)?;
                    let rhs = resolve_rhs(rhs, families, environment, indices)?;
                    LoweredValue::Scalar(with_site(evaluator, site, |evaluator| {
                        evaluator.mul_private(&input, &rhs).map_err(ProgramLowerError::Backend)
                    })?)
                }
                Step::Lut { input, table: lut, .. } => {
                    let input = resolve_wire(input, evaluator, families, environment, indices)?;
                    let table = program
                        .luts
                        .get(lut.0 as usize)
                        .ok_or(ProgramValidationError::InvalidLutId(*lut))?;
                    LoweredValue::Scalar(with_site(evaluator, site, |evaluator| {
                        evaluator.lut(&input, table).map_err(ProgramLowerError::Backend)
                    })?)
                }
                Step::MulSmallConstant { input, coefficients, shift, .. } => {
                    let input = resolve_wire(input, evaluator, families, environment, indices)?;
                    let shift = substitute_indices(shift, indices)?;
                    LoweredValue::Scalar(with_site(evaluator, site, |evaluator| {
                        evaluator
                            .mul_small_constant(&input, coefficients, &shift)
                            .map_err(ProgramLowerError::Backend)
                    })?)
                }
                Step::Parallel { range, body, .. } => {
                    let range = FamilyRange::bounded(
                        substitute_indices(range.start(), indices)?,
                        substitute_indices(range.count(), indices)?,
                        substitute_indices(range.capacity(), indices)?,
                    )?;
                    tracing::debug!(
                        depth = indices.len(),
                        body_steps = body.steps.len(),
                        count = ?range.count(),
                        "lowering parallel family body once"
                    );
                    let family =
                        families.parallel(evaluator, &range, |families, evaluator, local| {
                            indices.push(local);
                            ranges.push(range.clone());
                            let result = lower_body(
                                program,
                                body,
                                evaluator,
                                families,
                                environment,
                                indices,
                                ranges,
                                progress,
                            );
                            ranges.pop();
                            indices.pop();
                            let outputs = result?;
                            outputs.into_iter().next().ok_or_else(|| {
                                ProgramLowerError::UnsupportedStructure {
                                    reason: "parallel body has no scalar output".into(),
                                }
                            })
                        })?;
                    LoweredValue::Family(family)
                }
                Step::Sum { family, columns, .. } => {
                    let InputBinding::Family(family) = environment
                        .wires
                        .get(family)
                        .cloned()
                        .ok_or(ProgramValidationError::UndefinedWire(*family))?
                    else {
                        return Err(ProgramLowerError::Program(
                            ProgramValidationError::ExpectedFamilyWire(*family),
                        ));
                    };
                    let value = with_site(evaluator, site, |evaluator| {
                        families.sum(evaluator, &family, *columns)
                    })?;
                    LoweredValue::Scalar(value)
                }
            };
            let scalar = match value {
                LoweredValue::Scalar(value) => InputBinding::Scalar(value),
                LoweredValue::Family(value) => InputBinding::Family(value),
            };
            environment.wires.insert(output, scalar);
            direct.push(output);
            // A body-local wire no later step (nor nested body, nor the body's
            // outputs) reads is released, so a long body holds only its live
            // values.
            for wire in &releases[position] {
                environment.wires.remove(wire);
            }
            if indices.is_empty() {
                tracing::info!(
                    step = position + 1,
                    of = body.steps.len(),
                    visited_steps = progress.steps,
                    elapsed_ms = progress.started.elapsed().as_millis() as u64,
                    "program lowering top-level step finished"
                );
            }
        }
        let outputs = body
            .outputs
            .iter()
            .map(|operand| resolve_wire(operand, evaluator, families, environment, indices))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(outputs)
    })();
    for id in direct {
        environment.wires.remove(&id);
    }
    result
}

/// For each step of `body`, the wires defined by this body whose last read
/// is that step: an operand of it or of any step of its nested bodies. A wire
/// the body's outputs read, or no step reads, is never listed (an unread wire
/// is released at its own step).
fn last_reads(body: &Body) -> Vec<Vec<WireId>> {
    fn reads(step: &Step, into: &mut Vec<WireId>) {
        let operand = |operand: &Operand<WireId>, into: &mut Vec<WireId>| match operand {
            Operand::Scalar(id) |
            Operand::Member { family: id, .. } |
            Operand::Lookup { family: id, .. } => into.push(*id),
        };
        match step {
            Step::Add { lhs, rhs, .. } | Step::MulPublic { lhs, rhs, .. } => {
                operand(lhs, into);
                operand(rhs, into);
            }
            Step::MulPrivate { input, .. } |
            Step::Lut { input, .. } |
            Step::MulSmallConstant { input, .. } => operand(input, into),
            Step::Parallel { body, .. } => {
                for step in &body.steps {
                    reads(step, into);
                }
                for output in &body.outputs {
                    operand(output, into);
                }
            }
            Step::Sum { family, .. } => into.push(*family),
        }
    }
    let defined = body.steps.iter().map(step_output).collect::<BTreeSet<_>>();
    let mut last = BTreeMap::<WireId, usize>::new();
    for (position, step) in body.steps.iter().enumerate() {
        // A wire is defined before any later read, so its own step is its
        // first candidate position.
        last.insert(step_output(step), position);
        let mut read = Vec::new();
        reads(step, &mut read);
        for wire in read {
            if defined.contains(&wire) {
                last.insert(wire, position);
            }
        }
    }
    for output in &body.outputs {
        match output {
            Operand::Scalar(id) |
            Operand::Member { family: id, .. } |
            Operand::Lookup { family: id, .. } => {
                last.remove(id);
            }
        }
    }
    let mut releases = vec![Vec::new(); body.steps.len()];
    for (wire, position) in last {
        releases[position].push(wire);
    }
    releases
}

fn step_output(step: &Step) -> WireId {
    match step {
        Step::Add { output, .. } |
        Step::MulPublic { output, .. } |
        Step::MulPrivate { output, .. } |
        Step::Lut { output, .. } |
        Step::MulSmallConstant { output, .. } |
        Step::Parallel { output, .. } |
        Step::Sum { output, .. } => *output,
    }
}

fn resolve_wire<E, F>(
    operand: &Operand<WireId>,
    evaluator: &mut E,
    families: &mut F,
    environment: &Environment<E::Wire, E::Rhs, F::WireFamily, F::RhsFamily, F::Table>,
    indices: &[IntExpr],
) -> Result<E::Wire, ProgramLowerError<E::Error>>
where
    E: ExponentLUTProgramLowering + ProgramLoweringContext,
    F: FamilyExecution<E>,
{
    match operand {
        Operand::Scalar(id) => match environment.wires.get(id) {
            Some(InputBinding::Scalar(value)) => Ok(value.clone()),
            Some(InputBinding::Family(_)) => {
                Err(ProgramLowerError::Program(ProgramValidationError::ExpectedScalarWire(*id)))
            }
            None => Err(ProgramLowerError::Program(ProgramValidationError::UndefinedWire(*id))),
        },
        Operand::Member { family, index } => match environment.wires.get(family) {
            Some(InputBinding::Family(value)) => {
                families.wire_at(evaluator, value, &substitute_indices(index, indices)?)
            }
            Some(InputBinding::Scalar(_)) => {
                Err(ProgramLowerError::Program(ProgramValidationError::ExpectedFamilyWire(*family)))
            }
            None => Err(ProgramLowerError::Program(ProgramValidationError::UndefinedWire(*family))),
        },
        Operand::Lookup { family, table, index } => {
            let table = environment
                .tables
                .get(table)
                .ok_or(ProgramValidationError::UndefinedTable(*table))?;
            match environment.wires.get(family) {
                Some(InputBinding::Family(value)) => families.wire_at_table(
                    evaluator,
                    value,
                    table,
                    &substitute_indices(index, indices)?,
                ),
                Some(InputBinding::Scalar(_)) => Err(ProgramLowerError::Program(
                    ProgramValidationError::ExpectedFamilyWire(*family),
                )),
                None => {
                    Err(ProgramLowerError::Program(ProgramValidationError::UndefinedWire(*family)))
                }
            }
        }
    }
}

fn resolve_rhs<E, F>(
    operand: &Operand<RhsId>,
    families: &mut F,
    environment: &Environment<E::Wire, E::Rhs, F::WireFamily, F::RhsFamily, F::Table>,
    indices: &[IntExpr],
) -> Result<E::Rhs, ProgramLowerError<E::Error>>
where
    E: ExponentLUTProgramLowering + ProgramLoweringContext,
    F: FamilyExecution<E>,
{
    match operand {
        Operand::Scalar(id) => match environment.rhs.get(id) {
            Some(InputBinding::Scalar(value)) => Ok(value.clone()),
            Some(InputBinding::Family(_)) => {
                Err(ProgramLowerError::Program(ProgramValidationError::ExpectedScalarRhs(*id)))
            }
            None => Err(ProgramLowerError::Program(ProgramValidationError::UndefinedRhs(*id))),
        },
        Operand::Member { family, index } => match environment.rhs.get(family) {
            Some(InputBinding::Family(value)) => {
                families.rhs_at(value, &substitute_indices(index, indices)?)
            }
            Some(InputBinding::Scalar(_)) => {
                Err(ProgramLowerError::Program(ProgramValidationError::ExpectedFamilyRhs(*family)))
            }
            None => Err(ProgramLowerError::Program(ProgramValidationError::UndefinedRhs(*family))),
        },
        Operand::Lookup { .. } => {
            Err(ProgramLowerError::Program(ProgramValidationError::InvalidStructure(
                "an RHS operand cannot be a table lookup".into(),
            )))
        }
    }
}

fn with_site<E, T>(
    evaluator: &mut E,
    site: LoweringSite,
    operation: impl FnOnce(&mut E) -> Result<T, ProgramLowerError<E::Error>>,
) -> Result<T, ProgramLowerError<E::Error>>
where
    E: ExponentLUTProgramLowering + ProgramLoweringContext,
{
    let previous = evaluator.replace_site(Some(site));
    let result = operation(evaluator);
    evaluator.replace_site(previous);
    result
}

pub(crate) fn substitute_indices(
    expression: &IntExpr,
    indices: &[IntExpr],
) -> Result<IntExpr, ProgramValidationError> {
    let result = match expression {
        IntExpr::LoopIndex(slot) => indices
            .get(*slot as usize)
            .cloned()
            .ok_or(ProgramValidationError::InvalidLoopIndex(*slot))?,
        IntExpr::Add(lhs, rhs) => IntExpr::Add(
            Box::new(substitute_indices(lhs, indices)?),
            Box::new(substitute_indices(rhs, indices)?),
        ),
        IntExpr::Sub(lhs, rhs) => IntExpr::Sub(
            Box::new(substitute_indices(lhs, indices)?),
            Box::new(substitute_indices(rhs, indices)?),
        ),
        IntExpr::Mul(lhs, rhs) => IntExpr::Mul(
            Box::new(substitute_indices(lhs, indices)?),
            Box::new(substitute_indices(rhs, indices)?),
        ),
        IntExpr::Div(lhs, rhs) => IntExpr::Div(
            Box::new(substitute_indices(lhs, indices)?),
            Box::new(substitute_indices(rhs, indices)?),
        ),
        IntExpr::FloorDiv(lhs, rhs) => IntExpr::FloorDiv(
            Box::new(substitute_indices(lhs, indices)?),
            Box::new(substitute_indices(rhs, indices)?),
        ),
        IntExpr::Rem(lhs, rhs) => IntExpr::Rem(
            Box::new(substitute_indices(lhs, indices)?),
            Box::new(substitute_indices(rhs, indices)?),
        ),
        IntExpr::RoundDiv(lhs, rhs) => IntExpr::RoundDiv(
            Box::new(substitute_indices(lhs, indices)?),
            Box::new(substitute_indices(rhs, indices)?),
        ),
        IntExpr::Log2Ceil(value) => {
            IntExpr::Log2Ceil(Box::new(substitute_indices(value, indices)?))
        }
        IntExpr::Select { selector, branches } => IntExpr::Select {
            selector: Box::new(substitute_indices(selector, indices)?),
            branches: branches
                .iter()
                .map(|b| substitute_indices(b, indices))
                .collect::<Result<_, _>>()?,
        },
        leaf => leaf.clone(),
    };
    Ok(result.canonicalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::ToPrimitive;

    #[derive(Debug)]
    struct MockError;
    impl std::fmt::Display for MockError {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("mock lowering error")
        }
    }
    impl std::error::Error for MockError {}

    #[derive(Default)]
    struct MockEvaluator {
        site: Option<LoweringSite>,
        rhs: Vec<i32>,
        luts: Vec<Vec<usize>>,
    }

    impl ExponentLUTProgramLowering for MockEvaluator {
        type Wire = i32;
        type Rhs = i32;
        type Error = MockError;

        fn add(&mut self, lhs: &Self::Wire, rhs: &Self::Wire) -> Result<Self::Wire, Self::Error> {
            Ok(lhs + rhs)
        }
        fn mul_public(
            &mut self,
            lhs: &Self::Wire,
            rhs: &Self::Wire,
        ) -> Result<Self::Wire, Self::Error> {
            Ok(lhs * rhs)
        }
        fn mul_private(
            &mut self,
            lhs: &Self::Wire,
            rhs: &Self::Rhs,
        ) -> Result<Self::Wire, Self::Error> {
            self.rhs.push(*rhs);
            Ok(lhs * rhs)
        }
        fn lut(&mut self, input: &Self::Wire, table: &LutTable) -> Result<Self::Wire, Self::Error> {
            let values = table.values().expect("LUT values");
            self.luts.push(values.to_vec());
            Ok(*input + values.iter().sum::<usize>() as i32)
        }
        fn mul_small_constant(
            &mut self,
            input: &Self::Wire,
            coefficients: &[i64],
            _: &IntExpr,
        ) -> Result<Self::Wire, Self::Error> {
            Ok(input * coefficients.iter().sum::<i64>() as i32)
        }
    }

    impl ProgramLoweringContext for MockEvaluator {
        fn replace_site(&mut self, site: Option<LoweringSite>) -> Option<LoweringSite> {
            std::mem::replace(&mut self.site, site)
        }
    }

    struct MockFamilies;
    impl FamilyExecution<MockEvaluator> for MockFamilies {
        type WireFamily = Vec<i32>;
        type RhsFamily = Vec<i32>;
        type Table = Vec<usize>;

        fn wire_at_table(
            &mut self,
            evaluator: &mut MockEvaluator,
            family: &Self::WireFamily,
            table: &Self::Table,
            index: &IntExpr,
        ) -> Result<i32, ProgramLowerError<MockError>> {
            let index = index
                .evaluate(&ParamEnv::default())
                .map_err(|_| ProgramValidationError::MemberOutOfBounds)?;
            let index = index.to_usize().ok_or(ProgramValidationError::MemberOutOfBounds)?;
            let member = table.get(index).ok_or(ProgramValidationError::MemberOutOfBounds)?;
            self.wire_at(evaluator, family, &IntExpr::constant(*member))
        }

        fn wire_at(
            &mut self,
            _evaluator: &mut MockEvaluator,
            family: &Self::WireFamily,
            index: &IntExpr,
        ) -> Result<i32, ProgramLowerError<MockError>> {
            let index = index
                .evaluate(&ParamEnv::default())
                .map_err(|_| ProgramValidationError::MemberOutOfBounds)?;
            let index = index.to_usize().ok_or(ProgramValidationError::MemberOutOfBounds)?;
            family.get(index).copied().ok_or(ProgramValidationError::MemberOutOfBounds.into())
        }
        fn rhs_at(
            &mut self,
            family: &Self::RhsFamily,
            index: &IntExpr,
        ) -> Result<i32, ProgramLowerError<MockError>> {
            let index = index
                .evaluate(&ParamEnv::default())
                .map_err(|_| ProgramValidationError::MemberOutOfBounds)?;
            let index = index.to_usize().ok_or(ProgramValidationError::MemberOutOfBounds)?;
            family.get(index).copied().ok_or(ProgramValidationError::MemberOutOfBounds.into())
        }
        fn parallel(
            &mut self,
            evaluator: &mut MockEvaluator,
            range: &FamilyRange,
            body: impl FnOnce(
                &mut Self,
                &mut MockEvaluator,
                IntExpr,
            ) -> Result<i32, ProgramLowerError<MockError>>,
        ) -> Result<Self::WireFamily, ProgramLowerError<MockError>> {
            let count = range.count().evaluate(&ParamEnv::default()).unwrap().to_usize().unwrap();
            if count != 1 {
                return Err(ProgramLowerError::UnsupportedStructure {
                    reason: "mock family adapter only models singleton ranges".into(),
                });
            }
            let start = range.start().evaluate(&ParamEnv::default()).unwrap();
            Ok(vec![body(self, evaluator, IntExpr::constant(start.to_usize().unwrap()))?])
        }
        fn sum(
            &mut self,
            evaluator: &mut MockEvaluator,
            family: &Self::WireFamily,
            _: usize,
        ) -> Result<i32, ProgramLowerError<MockError>> {
            family.iter().skip(1).try_fold(family[0], |state, value| {
                evaluator.add(&state, value).map_err(ProgramLowerError::Backend)
            })
        }
    }

    #[test]
    fn lower_program_uses_one_environment_and_restores_sites() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input =
            builder.input_family(IntExpr::constant(4), AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_family(IntExpr::constant(4)).unwrap();
        let products = builder
            .parallel(
                FamilyRange::bounded(
                    IntExpr::constant(1),
                    IntExpr::constant(1),
                    IntExpr::constant(1),
                )
                .unwrap(),
                |builder, index| {
                    builder
                        .mul_private(
                            Operand::Member { family: input, index: index.clone() },
                            Operand::Member { family: rhs, index },
                        )
                        .map(Operand::Scalar)
                },
            )
            .unwrap();
        let sum = builder.sum(products).unwrap();
        let program = builder.finish(vec![Operand::Scalar(sum)]).unwrap();

        let mut environment = Environment::<i32, i32, Vec<i32>, Vec<i32>, Vec<usize>>::new();
        environment.bind_wire(input, InputBinding::Family(vec![1, 2, 3, 4])).unwrap();
        environment.bind_rhs(rhs, InputBinding::Family(vec![4, 5, 6, 7])).unwrap();
        let mut evaluator = MockEvaluator::default();
        let mut families = MockFamilies;
        let outputs =
            lower_program(&program, &mut evaluator, &mut families, &mut environment).unwrap();
        assert_eq!(outputs, vec![10]);
        assert!(evaluator.site.is_none());
        assert!(environment.wires.contains_key(&input));
        assert!(!environment.wires.contains_key(&products));
    }

    #[test]
    fn lower_program_resolves_one_rhs_and_lut_table_per_scalar_gate() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_input().unwrap();
        let lut = builder
            .register_lut(
                LutTable::unary(
                    4,
                    ExpLayout { stride_log2: 0, width_log2: 2 },
                    2,
                    vec![0, 1, 1, 0],
                    LutOutputRepresentation::Coefficient,
                )
                .unwrap(),
            )
            .unwrap();
        let product = builder.mul_private(Operand::Scalar(input), Operand::Scalar(rhs)).unwrap();
        let output = builder.lut(Operand::Scalar(product), lut).unwrap();
        let program = builder.finish(vec![Operand::Scalar(output)]).unwrap();

        let mut environment = Environment::<i32, i32, Vec<i32>, Vec<i32>, Vec<usize>>::new();
        environment.bind_wire(input, InputBinding::Scalar(3)).unwrap();
        environment.bind_rhs(rhs, InputBinding::Scalar(5)).unwrap();
        let mut evaluator = MockEvaluator::default();
        let mut families = MockFamilies;
        let outputs =
            lower_program(&program, &mut evaluator, &mut families, &mut environment).unwrap();

        assert_eq!(evaluator.rhs, [5]);
        assert_eq!(evaluator.luts, [vec![0, 1, 1, 0]]);
        assert_eq!(outputs, vec![17]);
        assert!(evaluator.site.is_none());
    }

    /// A lookup reads the member its bound table names; a declared table
    /// must be bound.
    #[test]
    fn lower_program_reads_lookups_through_the_bound_table() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input =
            builder.input_family(IntExpr::constant(4), AttributeKnowledge::PrivatePayload).unwrap();
        let table = builder.table(IntExpr::constant(2), IntExpr::constant(4)).unwrap();
        let output = builder
            .add(
                Operand::Lookup { family: input, table, index: IntExpr::constant(0) },
                Operand::Lookup { family: input, table, index: IntExpr::constant(1) },
            )
            .unwrap();
        let program = builder.finish(vec![Operand::Scalar(output)]).unwrap();

        let environment = || {
            let mut environment = Environment::<i32, i32, Vec<i32>, Vec<i32>, Vec<usize>>::new();
            environment.bind_wire(input, InputBinding::Family(vec![10, 20, 30, 40])).unwrap();
            environment
        };
        let mut bound = environment();
        bound.bind_table(table, vec![3, 1]).unwrap();
        let outputs =
            lower_program(&program, &mut MockEvaluator::default(), &mut MockFamilies, &mut bound)
                .unwrap();
        assert_eq!(outputs, vec![60]);
        assert!(matches!(
            lower_program(
                &program,
                &mut MockEvaluator::default(),
                &mut MockFamilies,
                &mut environment()
            ),
            Err(ProgramLowerError::Program(ProgramValidationError::MissingBinding))
        ));
    }

    #[test]
    fn lower_program_delivers_each_equal_lut_table() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let first_input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let second_input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let table = || {
            LutTable::unary(
                2,
                ExpLayout { stride_log2: 0, width_log2: 1 },
                2,
                vec![0, 1],
                LutOutputRepresentation::Coefficient,
            )
            .unwrap()
        };
        let first_lut = builder.register_lut(table()).unwrap();
        let second_lut = builder.register_lut(table()).unwrap();
        assert_ne!(first_lut, second_lut);
        let first_output = builder.lut(Operand::Scalar(first_input), first_lut).unwrap();
        let second_output = builder.lut(Operand::Scalar(second_input), second_lut).unwrap();
        let program = builder
            .finish(vec![Operand::Scalar(first_output), Operand::Scalar(second_output)])
            .unwrap();

        let mut environment = Environment::<i32, i32, Vec<i32>, Vec<i32>, Vec<usize>>::new();
        environment.bind_wire(first_input, InputBinding::Scalar(0)).unwrap();
        environment.bind_wire(second_input, InputBinding::Scalar(1)).unwrap();
        let mut evaluator = MockEvaluator::default();
        let mut families = MockFamilies;
        let outputs =
            lower_program(&program, &mut evaluator, &mut families, &mut environment).unwrap();

        assert_eq!(evaluator.luts, [vec![0, 1], vec![0, 1]]);
        assert_eq!(outputs, vec![1, 2]);
    }
}
