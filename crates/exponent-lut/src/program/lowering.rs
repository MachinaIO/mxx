use super::*;
use std::{error::Error, sync::Arc, time::Instant};

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

/// The values bound to a Program's inputs and RHS inputs while it is lowered.
/// Its integer data are the family execution's ([`FamilyExecution::wire_selected`]).
pub struct Environment<W, R, WF, RF> {
    pub(crate) wires: BTreeMap<WireId, InputBinding<W, WF>>,
    pub(crate) rhs: BTreeMap<RhsId, InputBinding<R, RF>>,
    /// Families lowered where their members are read ([`lower_program_deferred`]).
    lazy: BTreeMap<WireId, LazyValue<W, WF>>,
}

/// A Program family that [`lower_program_deferred`] lowers where its members are read instead
/// of as one graph loop: its body, its range and the enclosing loop indices and ranges at its
/// definition, and the bindings it reads (a snapshot, since the defining body releases its
/// wires after their last eager read). A member read lowers the body for that member, in the
/// graph scope of the read; a sum of its members is one sequential loop over them, so no family
/// of members is ever a graph value.
struct LazyFamily<W, WF> {
    body: Arc<Body>,
    range: FamilyRange,
    indices: Vec<IntExpr>,
    ranges: Vec<FamilyRange>,
    wires: BTreeMap<WireId, InputBinding<W, WF>>,
    lazy: BTreeMap<WireId, LazyValue<W, WF>>,
}

/// A deferred binding: a lazy family, or a deferred row `Sum` of `columns > 1` columns over a
/// lazy family, whose column blocks are read one at a time ([`DeferredColumns::block`]).
enum LazyValue<W, WF> {
    Family(Arc<LazyFamily<W, WF>>),
    Columns(DeferredColumns<W, WF>),
}

impl<W, WF> Clone for LazyValue<W, WF> {
    fn clone(&self) -> Self {
        match self {
            Self::Family(family) => Self::Family(Arc::clone(family)),
            Self::Columns(columns) => Self::Columns(columns.clone()),
        }
    }
}

/// A deferred Program output: the row `Sum` of `columns` column blocks over a family lowered
/// where it is read. Block `j` is the sum of the family's members `j b .. (j + 1) b` (`b` the
/// members per block), without its placement in column `j` of the row: a composing graph reads
/// each block where it needs it, so neither the family nor the wide row is a graph value.
pub struct DeferredColumns<W, WF> {
    family: Arc<LazyFamily<W, WF>>,
    columns: usize,
    site: LoweringSite,
}

impl<W, WF> Clone for DeferredColumns<W, WF> {
    fn clone(&self) -> Self {
        Self { family: Arc::clone(&self.family), columns: self.columns, site: self.site.clone() }
    }
}

/// A lowered Program output: a value, or a deferred row of column blocks.
pub enum LoweredOutput<W, WF> {
    Value(W),
    Columns(DeferredColumns<W, WF>),
}

impl<W, WF> DeferredColumns<W, WF> {
    /// The number of column blocks.
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// Column block `block` (an index expression of the composing graph) of this row, lowered
    /// in the current graph scope: the sum of its members, one sequential loop over them.
    pub fn block<E, R, RF, F>(
        &self,
        program: &ExponentLutProgram,
        evaluator: &mut E,
        families: &mut F,
        environment: &mut Environment<W, R, WF, RF>,
        block: &IntExpr,
    ) -> Result<W, ProgramLowerError<E::Error>>
    where
        E: ExponentLUTProgramLowering<Wire = W, Rhs = R> + ProgramLoweringContext,
        F: FamilyExecution<E, WireFamily = WF, RhsFamily = RF>,
    {
        let per = block_members(self.family.range.count(), self.columns)?;
        let first = IntExpr::Add(
            Box::new(self.family.range.start().clone()),
            Box::new(IntExpr::Mul(Box::new(block.clone()), Box::new(per.clone()))),
        )
        .canonicalize();
        let mut progress = LoweringProgress { steps: 0, started: Instant::now() };
        lazy_sum(
            program,
            &self.family,
            (first, per),
            self.site.clone(),
            evaluator,
            families,
            environment,
            &mut progress,
        )
    }
}

/// The members per column block of a row `Sum` of `columns` columns over `count` members.
fn block_members<E: Error + Send + Sync + 'static>(
    count: &IntExpr,
    columns: usize,
) -> Result<IntExpr, ProgramLowerError<E>> {
    if columns == 0 {
        return Err(ProgramLowerError::UnsupportedStructure {
            reason: "a row Sum has no columns".into(),
        });
    }
    Ok(IntExpr::FloorDiv(Box::new(count.clone()), Box::new(IntExpr::constant(columns)))
        .canonicalize())
}

enum LoweredValue<W, F> {
    Scalar(W),
    Family(F),
    Lazy(LazyValue<W, F>),
}

impl<W, R, WF, RF> Environment<W, R, WF, RF> {
    pub fn new() -> Self {
        Self { wires: BTreeMap::new(), rhs: BTreeMap::new(), lazy: BTreeMap::new() }
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
}

pub trait FamilyExecution<E>
where
    E: ExponentLUTProgramLowering + ProgramLoweringContext,
{
    type WireFamily: Clone;
    type RhsFamily: Clone;
    /// Member `index` of `family`. The evaluator observes the read (the noise
    /// evaluator records the read's payload support).
    fn wire_at(
        &mut self,
        evaluator: &mut E,
        family: &Self::WireFamily,
        index: &IntExpr,
    ) -> Result<E::Wire, ProgramLowerError<E::Error>>;
    /// The member of `family` whose index is entry `position` of the bounded integer data
    /// family `data` ([`Operand::Selected`]): the member is a runtime value, so an execution
    /// that needs the member statically treats it as unknown.
    fn wire_selected(
        &mut self,
        evaluator: &mut E,
        family: &Self::WireFamily,
        data: DataId,
        position: &IntExpr,
    ) -> Result<E::Wire, ProgramLowerError<E::Error>>;
    fn rhs_at(
        &mut self,
        family: &Self::RhsFamily,
        index: &IntExpr,
    ) -> Result<E::Rhs, ProgramLowerError<E::Error>>;
    /// The member of `family` whose index is entry `position` of the integer data family `data`
    /// ([`Operand::Selected`]).
    fn rhs_selected(
        &mut self,
        family: &Self::RhsFamily,
        data: DataId,
        position: &IntExpr,
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
    /// One sequential loop of `count` iterations from `initial`: iteration `i` (an index
    /// expression) maps the state through `body`. The reduction of a family lowered where its
    /// members are read ([`lower_program_deferred`]); an execution without graph loops does not
    /// lower deferred families.
    fn sequential_sum(
        &mut self,
        _evaluator: &mut E,
        _count: IntExpr,
        _initial: E::Wire,
        _body: impl FnOnce(
            &mut Self,
            &mut E,
            IntExpr,
            E::Wire,
        ) -> Result<E::Wire, ProgramLowerError<E::Error>>,
    ) -> Result<E::Wire, ProgramLowerError<E::Error>> {
        Err(ProgramLowerError::UnsupportedStructure {
            reason: "this family execution does not lower deferred families".into(),
        })
    }
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
    environment: &mut Environment<E::Wire, E::Rhs, F::WireFamily, F::RhsFamily>,
) -> Result<Vec<E::Wire>, ProgramLowerError<E::Error>>
where
    E: ExponentLUTProgramLowering + ProgramLoweringContext,
    F: FamilyExecution<E>,
{
    lower_program_deferred(program, evaluator, families, environment, &BTreeSet::new())?
        .into_iter()
        .map(|output| match output {
            LoweredOutput::Value(value) => Ok(value),
            LoweredOutput::Columns(_) => Err(ProgramLowerError::UnsupportedStructure {
                reason: "an output is deferred without being requested".into(),
            }),
        })
        .collect()
}

/// Lower `program` like [`lower_program`], except that the outputs at the positions `deferred`
/// that are `Sum`s of families (of one or more column blocks) are returned as
/// [`DeferredColumns`], read block by block ([`DeferredColumns::block`]), and that every family
/// is lowered where its members are read: a member read lowers the member, a `Sum` is one
/// sequential loop over the members, so no family is a graph loop's output.
pub(crate) fn lower_program_deferred<E, F>(
    program: &ExponentLutProgram,
    evaluator: &mut E,
    families: &mut F,
    environment: &mut Environment<E::Wire, E::Rhs, F::WireFamily, F::RhsFamily>,
    deferred: &BTreeSet<usize>,
) -> Result<Vec<LoweredOutput<E::Wire, F::WireFamily>>, ProgramLowerError<E::Error>>
where
    E: ExponentLUTProgramLowering + ProgramLoweringContext,
    F: FamilyExecution<E>,
{
    validate_program(program)?;
    if program.inputs.keys().any(|id| !environment.wires.contains_key(id)) ||
        program.rhs_inputs.keys().any(|id| !environment.rhs.contains_key(id)) ||
        environment.wires.keys().any(|id| !program.inputs.contains_key(id)) ||
        environment.rhs.keys().any(|id| !program.rhs_inputs.contains_key(id))
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
    let lazy_steps = deferred_steps(&program.body, deferred);
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
        (&mut indices, &mut ranges),
        &mut progress,
        Defer::Steps(&lazy_steps),
    )?;
    tracing::debug!(
        steps = progress.steps,
        elapsed_ms = progress.started.elapsed().as_millis() as u64,
        "program lowering finished"
    );
    Ok(values)
}

/// Which `Parallel` and row `Sum` steps of a body are lowered where they are read.
#[derive(Clone, Copy)]
enum Defer<'a> {
    /// The root steps whose output only deferred Program outputs read.
    Steps(&'a BTreeSet<WireId>),
    /// Every family of an eagerly lowered nested body.
    None,
    /// Every family of a body lowered for one member of a lazy family.
    All,
}

impl Defer<'_> {
    /// Whether the `Parallel` step of output `wire` is lowered where its members are read.
    fn defers(&self, wire: WireId) -> bool {
        match self {
            Self::Steps(steps) => steps.contains(&wire),
            Self::None => false,
            Self::All => true,
        }
    }

    /// Whether the `Sum` of output `wire` is a deferred Program output, read by blocks.
    fn defers_sum(&self, wire: WireId) -> bool {
        matches!(self, Self::Steps(steps) if steps.contains(&wire))
    }
}

/// The root steps lowered where they are read when the outputs `deferred` are deferred: every
/// `Parallel` step (a member read lowers the member, a `Sum` loops over its members, so no family
/// is a graph value) and the `Sum`s that no output outside `deferred` reads, directly or through
/// other steps (deferred outputs of one or more column blocks).
fn deferred_steps(body: &Body, deferred: &BTreeSet<usize>) -> BTreeSet<WireId> {
    if deferred.is_empty() {
        return BTreeSet::new();
    }
    let mut needed = body
        .outputs
        .iter()
        .enumerate()
        .filter(|(position, _)| !deferred.contains(position))
        .map(|(_, operand)| operand_wire(operand))
        .collect::<BTreeSet<_>>();
    for step in body.steps.iter().rev() {
        if needed.contains(&step_output(step)) {
            let mut read = Vec::new();
            step_reads(step, &mut read);
            needed.extend(read);
        }
    }
    body.steps
        .iter()
        .filter(|step| match step {
            Step::Parallel { .. } => true,
            Step::Sum { output, .. } => !needed.contains(output),
            _ => false,
        })
        .map(step_output)
        .collect()
}

fn operand_wire(operand: &Operand<WireId>) -> WireId {
    match operand {
        Operand::Scalar(id) |
        Operand::Member { family: id, .. } |
        Operand::Selected { family: id, .. } => *id,
    }
}

#[allow(clippy::too_many_arguments)]
fn lower_body<E, F>(
    program: &ExponentLutProgram,
    body: &Body,
    evaluator: &mut E,
    families: &mut F,
    environment: &mut Environment<E::Wire, E::Rhs, F::WireFamily, F::RhsFamily>,
    (indices, ranges): (&mut Vec<IntExpr>, &mut Vec<FamilyRange>),
    progress: &mut LoweringProgress,
    defer: Defer<'_>,
) -> Result<Vec<LoweredOutput<E::Wire, F::WireFamily>>, ProgramLowerError<E::Error>>
where
    E: ExponentLUTProgramLowering + ProgramLoweringContext,
    F: FamilyExecution<E>,
{
    let mut direct = Vec::new();
    let releases = last_reads(body);
    let result =
        (|| -> Result<Vec<LoweredOutput<E::Wire, F::WireFamily>>, ProgramLowerError<E::Error>> {
            for (position, step) in body.steps.iter().enumerate() {
                progress.step(indices.len());
                let output = step_output(step);
                let site =
                    LoweringSite { output, indices: indices.clone(), ranges: ranges.clone() };
                let value = match step {
                    Step::Add { lhs, rhs, .. } => {
                        let lhs = resolve_wire(
                            program,
                            lhs,
                            evaluator,
                            families,
                            environment,
                            indices,
                            progress,
                        )?;
                        let rhs = resolve_wire(
                            program,
                            rhs,
                            evaluator,
                            families,
                            environment,
                            indices,
                            progress,
                        )?;
                        LoweredValue::Scalar(with_site(evaluator, site, |evaluator| {
                            evaluator.add(&lhs, &rhs).map_err(ProgramLowerError::Backend)
                        })?)
                    }
                    Step::MulPublic { lhs, rhs, .. } => {
                        let lhs = resolve_wire(
                            program,
                            lhs,
                            evaluator,
                            families,
                            environment,
                            indices,
                            progress,
                        )?;
                        let rhs = resolve_wire(
                            program,
                            rhs,
                            evaluator,
                            families,
                            environment,
                            indices,
                            progress,
                        )?;
                        LoweredValue::Scalar(with_site(evaluator, site, |evaluator| {
                            evaluator.mul_public(&lhs, &rhs).map_err(ProgramLowerError::Backend)
                        })?)
                    }
                    Step::MulPrivate { input, rhs, .. } => {
                        let input = resolve_wire(
                            program,
                            input,
                            evaluator,
                            families,
                            environment,
                            indices,
                            progress,
                        )?;
                        let rhs = resolve_rhs(rhs, families, environment, indices)?;
                        LoweredValue::Scalar(with_site(evaluator, site, |evaluator| {
                            evaluator.mul_private(&input, &rhs).map_err(ProgramLowerError::Backend)
                        })?)
                    }
                    Step::Lut { input, table: lut, .. } => {
                        let input = resolve_wire(
                            program,
                            input,
                            evaluator,
                            families,
                            environment,
                            indices,
                            progress,
                        )?;
                        let table = program
                            .luts
                            .get(lut.0 as usize)
                            .ok_or(ProgramValidationError::InvalidLutId(*lut))?;
                        LoweredValue::Scalar(with_site(evaluator, site, |evaluator| {
                            evaluator.lut(&input, table).map_err(ProgramLowerError::Backend)
                        })?)
                    }
                    Step::MulSmallConstant { input, coefficients, shift, .. } => {
                        let input = resolve_wire(
                            program,
                            input,
                            evaluator,
                            families,
                            environment,
                            indices,
                            progress,
                        )?;
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
                        if defer.defers(output) {
                            LoweredValue::Lazy(LazyValue::Family(Arc::new(LazyFamily {
                                body: Arc::new(body.as_ref().clone()),
                                range,
                                indices: indices.clone(),
                                ranges: ranges.clone(),
                                wires: environment.wires.clone(),
                                lazy: environment.lazy.clone(),
                            })))
                        } else {
                            tracing::debug!(
                                depth = indices.len(),
                                body_steps = body.steps.len(),
                                count = ?range.count(),
                                "lowering parallel family body once"
                            );
                            let family = families.parallel(
                                evaluator,
                                &range,
                                |families, evaluator, local| {
                                    indices.push(local);
                                    ranges.push(range.clone());
                                    let result = lower_body(
                                        program,
                                        body,
                                        evaluator,
                                        families,
                                        environment,
                                        (&mut *indices, &mut *ranges),
                                        progress,
                                        Defer::None,
                                    );
                                    ranges.pop();
                                    indices.pop();
                                    single_value(result?)
                                },
                            )?;
                            LoweredValue::Family(family)
                        }
                    }
                    Step::Sum { family, columns, .. } => {
                        if let Some(lazy) = environment.lazy.get(family).cloned() {
                            let LazyValue::Family(lazy) = lazy else {
                                return Err(ProgramLowerError::UnsupportedStructure {
                                    reason: "a Sum reads a deferred row".into(),
                                });
                            };
                            if defer.defers_sum(output) {
                                LoweredValue::Lazy(LazyValue::Columns(DeferredColumns {
                                    family: lazy,
                                    columns: *columns,
                                    site,
                                }))
                            } else if *columns == 1 {
                                let first =
                                    (lazy.range.start().clone(), lazy.range.count().clone());
                                LoweredValue::Scalar(lazy_sum(
                                    program,
                                    &lazy,
                                    first,
                                    site,
                                    evaluator,
                                    families,
                                    environment,
                                    progress,
                                )?)
                            } else {
                                return Err(ProgramLowerError::UnsupportedStructure {
                                    reason:
                                        "a row Sum of a deferred family is not a deferred output"
                                            .into(),
                                });
                            }
                        } else {
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
                    }
                };
                match value {
                    LoweredValue::Scalar(value) => {
                        environment.wires.insert(output, InputBinding::Scalar(value));
                    }
                    LoweredValue::Family(value) => {
                        environment.wires.insert(output, InputBinding::Family(value));
                    }
                    LoweredValue::Lazy(value) => {
                        environment.lazy.insert(output, value);
                    }
                }
                direct.push(output);
                // A body-local wire no later step (nor nested body, nor the body's
                // outputs) reads is released, so a long body holds only its live
                // values.
                for wire in &releases[position] {
                    environment.wires.remove(wire);
                    environment.lazy.remove(wire);
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
            body.outputs
                .iter()
                .map(|operand| match (operand, environment.lazy.get(&operand_wire(operand))) {
                    (Operand::Scalar(_), Some(LazyValue::Columns(columns))) => {
                        Ok(LoweredOutput::Columns(columns.clone()))
                    }
                    _ => resolve_wire(
                        program,
                        operand,
                        evaluator,
                        families,
                        environment,
                        indices,
                        progress,
                    )
                    .map(LoweredOutput::Value),
                })
                .collect::<Result<Vec<_>, _>>()
        })();
    for id in direct {
        environment.wires.remove(&id);
        environment.lazy.remove(&id);
    }
    result
}

/// The single value output of a family body.
fn single_value<W, WF, E: Error + Send + Sync + 'static>(
    outputs: Vec<LoweredOutput<W, WF>>,
) -> Result<W, ProgramLowerError<E>> {
    match outputs.into_iter().next() {
        Some(LoweredOutput::Value(value)) => Ok(value),
        Some(LoweredOutput::Columns(_)) => Err(ProgramLowerError::UnsupportedStructure {
            reason: "a family body outputs a deferred row".into(),
        }),
        None => Err(ProgramLowerError::UnsupportedStructure {
            reason: "parallel body has no scalar output".into(),
        }),
    }
}

/// Member `index` (an absolute member index) of the lazy family `family`, lowered in the
/// current graph scope: its body under its definition's bindings and indices, with every
/// family of the body lazy too.
fn instantiate<E, F>(
    program: &ExponentLutProgram,
    family: &LazyFamily<E::Wire, F::WireFamily>,
    index: IntExpr,
    evaluator: &mut E,
    families: &mut F,
    environment: &mut Environment<E::Wire, E::Rhs, F::WireFamily, F::RhsFamily>,
    progress: &mut LoweringProgress,
) -> Result<E::Wire, ProgramLowerError<E::Error>>
where
    E: ExponentLUTProgramLowering + ProgramLoweringContext,
    F: FamilyExecution<E>,
{
    let wires = std::mem::replace(&mut environment.wires, family.wires.clone());
    let lazy = std::mem::replace(&mut environment.lazy, family.lazy.clone());
    let mut indices = family.indices.clone();
    indices.push(index.canonicalize());
    let mut ranges = family.ranges.clone();
    ranges.push(family.range.clone());
    let result = lower_body(
        program,
        &family.body,
        evaluator,
        families,
        environment,
        (&mut indices, &mut ranges),
        progress,
        Defer::All,
    );
    environment.wires = wires;
    environment.lazy = lazy;
    single_value(result?)
}

/// The sum of the `count` members of the lazy family `family` from member `first`, under the
/// site `site` of the reduction: the first member relabelled by the identity operation, then
/// one sequential loop over the rest, whose body lowers each member
/// ([`FamilyExecution::sequential_sum`]).
#[allow(clippy::too_many_arguments)]
fn lazy_sum<E, F>(
    program: &ExponentLutProgram,
    family: &LazyFamily<E::Wire, F::WireFamily>,
    (first, count): (IntExpr, IntExpr),
    site: LoweringSite,
    evaluator: &mut E,
    families: &mut F,
    environment: &mut Environment<E::Wire, E::Rhs, F::WireFamily, F::RhsFamily>,
    progress: &mut LoweringProgress,
) -> Result<E::Wire, ProgramLowerError<E::Error>>
where
    E: ExponentLUTProgramLowering + ProgramLoweringContext,
    F: FamilyExecution<E>,
{
    let member =
        |offset: IntExpr| IntExpr::Add(Box::new(first.clone()), Box::new(offset)).canonicalize();
    let count_value = count.evaluate(&ParamEnv::default()).ok();
    if count_value.as_ref().is_some_and(|count| *count <= 0.into()) {
        return Err(ProgramLowerError::UnsupportedStructure {
            reason: "cannot reduce an empty family".into(),
        });
    }
    let head = instantiate(
        program,
        family,
        member(IntExpr::constant(0)),
        evaluator,
        families,
        environment,
        progress,
    )?;
    if count_value.as_ref().is_some_and(|count| *count == 1.into()) {
        return Ok(head);
    }
    // Label the initial member with the reduction output without changing its value.
    // Lowering a second member outside the loop duplicates every nested lazy body;
    // identity multiplication is a metadata-only operation for RNS graph carriers.
    let initial = with_site(evaluator, site.clone(), |evaluator| {
        evaluator
            .mul_small_constant(&head, &[1], &IntExpr::constant(0))
            .map_err(ProgramLowerError::Backend)
    })?;
    let rest = IntExpr::Sub(Box::new(count), Box::new(IntExpr::constant(1))).canonicalize();
    families.sequential_sum(evaluator, rest, initial, |families, evaluator, index, state| {
        let index = IntExpr::Add(Box::new(index), Box::new(IntExpr::constant(1)));
        let value = instantiate(
            program,
            family,
            member(index),
            evaluator,
            families,
            environment,
            progress,
        )?;
        with_site(evaluator, site, |evaluator| {
            evaluator.add(&state, &value).map_err(ProgramLowerError::Backend)
        })
    })
}

/// For each step of `body`, the wires defined by this body whose last read
/// is that step: an operand of it or of any step of its nested bodies. A wire
/// the body's outputs read, or no step reads, is never listed (an unread wire
/// is released at its own step).
/// Every wire `step` reads: an operand of it or of any step of its nested bodies.
fn step_reads(step: &Step, into: &mut Vec<WireId>) {
    let operand = |operand: &Operand<WireId>, into: &mut Vec<WireId>| match operand {
        Operand::Scalar(id) |
        Operand::Member { family: id, .. } |
        Operand::Selected { family: id, .. } => into.push(*id),
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
                step_reads(step, into);
            }
            for output in &body.outputs {
                operand(output, into);
            }
        }
        Step::Sum { family, .. } => into.push(*family),
    }
}

fn last_reads(body: &Body) -> Vec<Vec<WireId>> {
    let defined = body.steps.iter().map(step_output).collect::<BTreeSet<_>>();
    let mut last = BTreeMap::<WireId, usize>::new();
    for (position, step) in body.steps.iter().enumerate() {
        // A wire is defined before any later read, so its own step is its
        // first candidate position.
        last.insert(step_output(step), position);
        let mut read = Vec::new();
        step_reads(step, &mut read);
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
            Operand::Selected { family: id, .. } => {
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
    program: &ExponentLutProgram,
    operand: &Operand<WireId>,
    evaluator: &mut E,
    families: &mut F,
    environment: &mut Environment<E::Wire, E::Rhs, F::WireFamily, F::RhsFamily>,
    indices: &[IntExpr],
    progress: &mut LoweringProgress,
) -> Result<E::Wire, ProgramLowerError<E::Error>>
where
    E: ExponentLUTProgramLowering + ProgramLoweringContext,
    F: FamilyExecution<E>,
{
    if let Some(lazy) = environment.lazy.get(&operand_wire(operand)).cloned() {
        return match (operand, lazy) {
            (Operand::Member { index, .. }, LazyValue::Family(family)) => {
                let index = substitute_indices(index, indices)?;
                instantiate(program, &family, index, evaluator, families, environment, progress)
            }
            _ => Err(ProgramLowerError::UnsupportedStructure {
                reason: "a deferred family is read other than by a member index".into(),
            }),
        };
    }
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
        Operand::Selected { family, data, position } => match environment.wires.get(family) {
            Some(InputBinding::Family(value)) => families.wire_selected(
                evaluator,
                value,
                *data,
                &substitute_indices(position, indices)?,
            ),
            Some(InputBinding::Scalar(_)) => {
                Err(ProgramLowerError::Program(ProgramValidationError::ExpectedFamilyWire(*family)))
            }
            None => Err(ProgramLowerError::Program(ProgramValidationError::UndefinedWire(*family))),
        },
    }
}

fn resolve_rhs<E, F>(
    operand: &Operand<RhsId>,
    families: &mut F,
    environment: &Environment<E::Wire, E::Rhs, F::WireFamily, F::RhsFamily>,
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
        Operand::Selected { family, data, position } => match environment.rhs.get(family) {
            Some(InputBinding::Family(value)) => {
                families.rhs_selected(value, *data, &substitute_indices(position, indices)?)
            }
            Some(InputBinding::Scalar(_)) => {
                Err(ProgramLowerError::Program(ProgramValidationError::ExpectedFamilyRhs(*family)))
            }
            None => Err(ProgramLowerError::Program(ProgramValidationError::UndefinedRhs(*family))),
        },
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

    /// The family adapter of the mock, with the Program data its selections read.
    #[derive(Default)]
    struct MockFamilies(crate::program::ProgramData);
    impl FamilyExecution<MockEvaluator> for MockFamilies {
        type WireFamily = Vec<i32>;
        type RhsFamily = Vec<i32>;

        fn wire_selected(
            &mut self,
            evaluator: &mut MockEvaluator,
            family: &Self::WireFamily,
            data: DataId,
            position: &IntExpr,
        ) -> Result<i32, ProgramLowerError<MockError>> {
            let position = position
                .evaluate(&ParamEnv::default())
                .map_err(|_| ProgramValidationError::MemberOutOfBounds)?;
            let position = position.to_usize().ok_or(ProgramValidationError::MemberOutOfBounds)?;
            let member = self
                .0
                .values(data)
                .and_then(|values| values.get(position).copied())
                .ok_or(ProgramValidationError::MissingBinding)?;
            self.wire_at(evaluator, family, &IntExpr::constant(member))
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
        fn rhs_selected(
            &mut self,
            _: &Self::RhsFamily,
            _: DataId,
            _: &IntExpr,
        ) -> Result<i32, ProgramLowerError<MockError>> {
            Err(ProgramLowerError::UnsupportedStructure { reason: "mock has no data".into() })
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

        let mut environment = Environment::<i32, i32, Vec<i32>, Vec<i32>>::new();
        environment.bind_wire(input, InputBinding::Family(vec![1, 2, 3, 4])).unwrap();
        environment.bind_rhs(rhs, InputBinding::Family(vec![4, 5, 6, 7])).unwrap();
        let mut evaluator = MockEvaluator::default();
        let mut families = MockFamilies::default();
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

        let mut environment = Environment::<i32, i32, Vec<i32>, Vec<i32>>::new();
        environment.bind_wire(input, InputBinding::Scalar(3)).unwrap();
        environment.bind_rhs(rhs, InputBinding::Scalar(5)).unwrap();
        let mut evaluator = MockEvaluator::default();
        let mut families = MockFamilies::default();
        let outputs =
            lower_program(&program, &mut evaluator, &mut families, &mut environment).unwrap();

        assert_eq!(evaluator.rhs, [5]);
        assert_eq!(evaluator.luts, [vec![0, 1, 1, 0]]);
        assert_eq!(outputs, vec![17]);
        assert!(evaluator.site.is_none());
    }

    /// A wire selection reads the member its bound data name; a declared data
    /// family must be bound.
    #[test]
    fn lower_program_reads_wire_selections_through_the_bound_data() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input =
            builder.input_family(IntExpr::constant(4), AttributeKnowledge::PrivatePayload).unwrap();
        let data = builder.data_family(2, Some(4)).unwrap();
        let output = builder
            .add(
                Operand::Selected { family: input, data, position: IntExpr::constant(0) },
                Operand::Selected { family: input, data, position: IntExpr::constant(1) },
            )
            .unwrap();
        let program = builder.finish(vec![Operand::Scalar(output)]).unwrap();

        let environment = || {
            let mut environment = Environment::<i32, i32, Vec<i32>, Vec<i32>>::new();
            environment.bind_wire(input, InputBinding::Family(vec![10, 20, 30, 40])).unwrap();
            environment
        };
        let outputs = lower_program(
            &program,
            &mut MockEvaluator::default(),
            &mut MockFamilies(crate::program::ProgramData::new().with(data, vec![3, 1])),
            &mut environment(),
        )
        .unwrap();
        assert_eq!(outputs, vec![60]);
        assert!(matches!(
            lower_program(
                &program,
                &mut MockEvaluator::default(),
                &mut MockFamilies::default(),
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

        let mut environment = Environment::<i32, i32, Vec<i32>, Vec<i32>>::new();
        environment.bind_wire(first_input, InputBinding::Scalar(0)).unwrap();
        environment.bind_wire(second_input, InputBinding::Scalar(1)).unwrap();
        let mut evaluator = MockEvaluator::default();
        let mut families = MockFamilies::default();
        let outputs =
            lower_program(&program, &mut evaluator, &mut families, &mut environment).unwrap();

        assert_eq!(evaluator.luts, [vec![0, 1], vec![0, 1]]);
        assert_eq!(outputs, vec![1, 2]);
    }
}
