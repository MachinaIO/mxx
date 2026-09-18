use super::*;
use std::error::Error;

pub trait ExponentLUTProgramLowering {
    type Wire: Clone;
    type Rhs: Clone;
    type Error: Error + Send + Sync + 'static;

    fn add(&mut self, lhs: &Self::Wire, rhs: &Self::Wire) -> Result<Self::Wire, Self::Error>;
    /// Multiply the private/exponent left operand by the public/exponent right
    /// operand. The argument order is part of the program contract and is
    /// intentionally not commutative: prepared actions and noise accounting
    /// attach to the private operand first.
    fn mul_public(
        &mut self,
        private: &Self::Wire,
        public: &Self::Wire,
    ) -> Result<Self::Wire, Self::Error>;
    fn mul_private(
        &mut self,
        input: &Self::Wire,
        rhs: &Self::Rhs,
    ) -> Result<Self::Wire, Self::Error>;
    fn lut(&mut self, input: &Self::Wire, table: &LutTable) -> Result<Self::Wire, Self::Error>;
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

pub struct Environment<W, R, WF, RF> {
    pub(crate) wires: BTreeMap<WireId, InputBinding<W, WF>>,
    pub(crate) rhs: BTreeMap<RhsId, InputBinding<R, RF>>,
}

enum LoweredValue<W, F> {
    Scalar(W),
    Family(F),
}

impl<W, R, WF, RF> Environment<W, R, WF, RF> {
    pub fn new() -> Self {
        Self { wires: BTreeMap::new(), rhs: BTreeMap::new() }
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
    fn wire_at(
        &mut self,
        family: &Self::WireFamily,
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
    fn sum(
        &mut self,
        evaluator: &mut E,
        family: &Self::WireFamily,
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
    environment: &mut Environment<E::Wire, E::Rhs, F::WireFamily, F::RhsFamily>,
) -> Result<Vec<E::Wire>, ProgramLowerError<E::Error>>
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
    let mut indices = Vec::new();
    let mut ranges = Vec::new();
    let values = lower_body(
        program,
        &program.body,
        evaluator,
        families,
        environment,
        &mut indices,
        &mut ranges,
    )?;
    Ok(values)
}

fn lower_body<E, F>(
    program: &ExponentLutProgram,
    body: &Body,
    evaluator: &mut E,
    families: &mut F,
    environment: &mut Environment<E::Wire, E::Rhs, F::WireFamily, F::RhsFamily>,
    indices: &mut Vec<IntExpr>,
    ranges: &mut Vec<FamilyRange>,
) -> Result<Vec<E::Wire>, ProgramLowerError<E::Error>>
where
    E: ExponentLUTProgramLowering + ProgramLoweringContext,
    F: FamilyExecution<E>,
{
    let mut direct = Vec::new();
    let result = (|| -> Result<Vec<E::Wire>, ProgramLowerError<E::Error>> {
        for step in &body.steps {
            let output = step_output(step);
            let site = LoweringSite { output, indices: indices.clone(), ranges: ranges.clone() };
            let value = match step {
                Step::Add { lhs, rhs, .. } => {
                    let lhs = resolve_wire(lhs, families, environment, indices)?;
                    let rhs = resolve_wire(rhs, families, environment, indices)?;
                    LoweredValue::Scalar(with_site(evaluator, site, |evaluator| {
                        evaluator.add(&lhs, &rhs).map_err(ProgramLowerError::Backend)
                    })?)
                }
                Step::MulPublic { private, public, .. } => {
                    let private = resolve_wire(private, families, environment, indices)?;
                    let public = resolve_wire(public, families, environment, indices)?;
                    LoweredValue::Scalar(with_site(evaluator, site, |evaluator| {
                        evaluator.mul_public(&private, &public).map_err(ProgramLowerError::Backend)
                    })?)
                }
                Step::MulPrivate { input, rhs, .. } => {
                    let input = resolve_wire(input, families, environment, indices)?;
                    let rhs = resolve_rhs(rhs, families, environment, indices)?;
                    LoweredValue::Scalar(with_site(evaluator, site, |evaluator| {
                        evaluator.mul_private(&input, &rhs).map_err(ProgramLowerError::Backend)
                    })?)
                }
                Step::Lut { input, table: lut, .. } => {
                    let input = resolve_wire(input, families, environment, indices)?;
                    let table = program
                        .luts
                        .get(lut.0 as usize)
                        .ok_or(ProgramValidationError::InvalidLutId(*lut))?;
                    LoweredValue::Scalar(with_site(evaluator, site, |evaluator| {
                        evaluator.lut(&input, table).map_err(ProgramLowerError::Backend)
                    })?)
                }
                Step::Parallel { range, body, .. } => {
                    let range = FamilyRange::bounded(
                        substitute_indices(range.start(), indices)?,
                        substitute_indices(range.count(), indices)?,
                        substitute_indices(range.capacity(), indices)?,
                    )?;
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
                Step::Sum { family, .. } => {
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
                    let value =
                        with_site(evaluator, site, |evaluator| families.sum(evaluator, &family))?;
                    LoweredValue::Scalar(value)
                }
            };
            let scalar = match value {
                LoweredValue::Scalar(value) => InputBinding::Scalar(value),
                LoweredValue::Family(value) => InputBinding::Family(value),
            };
            environment.wires.insert(output, scalar);
            direct.push(output);
        }
        let outputs = body
            .outputs
            .iter()
            .map(|operand| resolve_wire(operand, families, environment, indices))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(outputs)
    })();
    for id in direct {
        environment.wires.remove(&id);
    }
    result
}

fn step_output(step: &Step) -> WireId {
    match step {
        Step::Add { output, .. } |
        Step::MulPublic { output, .. } |
        Step::MulPrivate { output, .. } |
        Step::Lut { output, .. } |
        Step::Parallel { output, .. } |
        Step::Sum { output, .. } => *output,
    }
}

fn resolve_wire<E, F>(
    operand: &Operand<WireId>,
    families: &mut F,
    environment: &Environment<E::Wire, E::Rhs, F::WireFamily, F::RhsFamily>,
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
                families.wire_at(value, &substitute_indices(index, indices)?)
            }
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
            private: &Self::Wire,
            public: &Self::Wire,
        ) -> Result<Self::Wire, Self::Error> {
            Ok(private * public)
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
            self.luts.push(table.values().to_vec());
            Ok(*input + table.values().iter().sum::<usize>() as i32)
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

        fn wire_at(
            &mut self,
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

        let mut environment = Environment::<i32, i32, Vec<i32>, Vec<i32>>::new();
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
        let mut families = MockFamilies;
        let outputs =
            lower_program(&program, &mut evaluator, &mut families, &mut environment).unwrap();

        assert_eq!(evaluator.luts, [vec![0, 1], vec![0, 1]]);
        assert_eq!(outputs, vec![1, 2]);
    }
}
