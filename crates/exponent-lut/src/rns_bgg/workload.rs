//! The RNS-BGG level workload derived from a Program.
use super::{RhsMode, RnsBggError, RnsBggParameters};
use crate::program::{
    Environment, ExponentLUTProgramLowering, ExponentLutProgram, FamilyExecution, FamilyRange,
    InputBinding, LutId, LutTable, ProgramLowerError, ProgramLoweringContext, RhsId, WireId,
};
use mxx_ir_core::IntExpr;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use super::noise::RnsBggNoiseFamilies;

/// Collects the derived logical level map for a Program.
pub(crate) struct WorkloadCollector<'a> {
    program: &'a ExponentLutProgram,
    parameters: Option<&'a RnsBggParameters>,
    levels: BTreeMap<WireId, usize>,
    rhs_levels: BTreeMap<RhsId, BTreeSet<usize>>,
    luts: BTreeSet<(usize, LutId)>,
    /// For each public Program input, the lowest level at which it is
    /// consumed, or `None` once any use needs it at its own level.
    public_input_uses: BTreeMap<WireId, Option<usize>>,
    current_site: Option<crate::program::LoweringSite>,
}

impl<'a> WorkloadCollector<'a> {
    /// Record one use of `wire`. A public input consumed only as the public
    /// left operand of a product with a private right operand can be encoded
    /// at that product's level, which is set by the right operand alone;
    /// `consumed_at` is then that level. Any other use pins it.
    fn note_use(&mut self, wire: WireId, consumed_at: Option<usize>) {
        if let Some(entry) = self.public_input_uses.get_mut(&wire) {
            *entry = match (*entry, consumed_at) {
                (Some(current), Some(level)) => Some(current.min(level)),
                _ => None,
            };
        }
    }

    fn site(&self) -> Result<&crate::program::LoweringSite, RnsBggError> {
        self.current_site
            .as_ref()
            .ok_or(RnsBggError::Graph("workload lowering site is missing".into()))
    }
}

/// The single workload map retained outside the validated Program.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RnsBggDerivedWorkload {
    program: crate::program::ExponentLutProgramId,
    pub(super) levels: BTreeMap<WireId, usize>,
    /// The consuming levels of each RHS: one level for Static parameters,
    /// any number for Dynamic ones, whose packages are setup material read
    /// at every consuming level.
    rhs_levels: BTreeMap<RhsId, BTreeSet<usize>>,
    /// The input level and identity of every LUT the Program evaluates on a
    /// private payload.
    luts: BTreeSet<(usize, LutId)>,
}

impl RnsBggDerivedWorkload {
    /// Derive logical levels by lowering the Program once, with the Static
    /// rule that every RHS has one consuming level.
    pub fn from_program(program: &ExponentLutProgram) -> Result<Self, RnsBggError> {
        Self::derive(program, None)
    }

    /// Derive logical levels while validating public operand routes against
    /// the physical modulus chain, with the RHS rule of the parameters' mode:
    /// a Dynamic RHS may be read at several logical levels.
    pub fn from_program_with_parameters(
        parameters: &RnsBggParameters,
        program: &ExponentLutProgram,
    ) -> Result<Self, RnsBggError> {
        Self::derive(program, Some(parameters))
    }

    /// Public inputs are encoded at the level where they are first consumed,
    /// so that no stage stores or aligns them at a higher modulus than they
    /// are used at. The first pass finds those levels with every input at
    /// level zero; the second derives the workload with them.
    fn derive<'a>(
        program: &'a ExponentLutProgram,
        parameters: Option<&'a RnsBggParameters>,
    ) -> Result<Self, RnsBggError> {
        let first = Self::derive_with_input_levels(program, parameters, &BTreeMap::new())?;
        let input_levels = first
            .1
            .into_iter()
            .filter_map(|(wire, level)| level.filter(|level| *level > 0).map(|level| (wire, level)))
            .collect::<BTreeMap<_, _>>();
        if input_levels.is_empty() {
            return Ok(first.0);
        }
        Ok(Self::derive_with_input_levels(program, parameters, &input_levels)?.0)
    }

    fn derive_with_input_levels<'a>(
        program: &'a ExponentLutProgram,
        parameters: Option<&'a RnsBggParameters>,
        input_levels: &BTreeMap<WireId, usize>,
    ) -> Result<(Self, BTreeMap<WireId, Option<usize>>), RnsBggError> {
        let input_level = |id: &WireId| input_levels.get(id).copied().unwrap_or(0);
        let mut collector = WorkloadCollector {
            program,
            parameters,
            levels: program.inputs().keys().map(|id| (*id, input_level(id))).collect(),
            rhs_levels: BTreeMap::new(),
            luts: BTreeSet::new(),
            public_input_uses: program
                .inputs()
                .iter()
                .filter(|(_, declaration)| {
                    matches!(
                        declaration.knowledge(),
                        crate::program::AttributeKnowledge::PublicPayload
                    )
                })
                .map(|(id, _)| (*id, Some(usize::MAX)))
                .collect(),
            current_site: None,
        };
        // A public input that is itself an output keeps its own level.
        for output in program.outputs() {
            collector.public_input_uses.insert(*output.source(), None);
        }
        let mut environment = Environment::new();
        for id in program.inputs().keys() {
            match program.inputs().get(id).expect("input declaration").shape() {
                InputBinding::Scalar(()) => {
                    environment.bind_wire(*id, InputBinding::Scalar((*id, input_level(id))))
                }
                InputBinding::Family(count) => environment.bind_wire(
                    *id,
                    InputBinding::Family((
                        *id,
                        input_level(id),
                        FamilyRange::full(count.clone())
                            .map_err(|_| RnsBggError::Binding("invalid family range"))?,
                    )),
                ),
            }
            .map_err(|_| RnsBggError::Binding("invalid workload binding"))?;
        }
        for (id, declaration) in program.rhs_inputs() {
            match declaration {
                InputBinding::Scalar(()) => environment.bind_rhs(*id, InputBinding::Scalar(*id)),
                InputBinding::Family(count) => environment.bind_rhs(
                    *id,
                    InputBinding::Family((
                        *id,
                        FamilyRange::full(count.clone())
                            .map_err(|_| RnsBggError::Binding("invalid RHS family range"))?,
                    )),
                ),
            }
            .map_err(|_| RnsBggError::Binding("invalid workload RHS binding"))?;
        }
        let mut families = RnsBggNoiseFamilies::default();
        crate::program::lower_program(program, &mut collector, &mut families, &mut environment)
            .map_err(|error| match error {
                ProgramLowerError::Backend(error) => error,
                ProgramLowerError::Program(error) => RnsBggError::Graph(error.to_string()),
                ProgramLowerError::Dsl(error) => RnsBggError::Graph(error.to_string()),
                ProgramLowerError::UnsupportedStructure { reason } => RnsBggError::Graph(reason),
            })?;
        // A family wire has the level of the member its body returns, so a
        // member output of a root family has a level.
        for (family, inner) in program.family_body_outputs() {
            if let Some(level) = collector.levels.get(&inner).copied() {
                collector.levels.insert(family, level);
            }
        }
        let uses = collector
            .public_input_uses
            .iter()
            .filter(|(wire, _)| program.inputs().contains_key(wire))
            .map(|(wire, level)| (*wire, level.filter(|level| *level != usize::MAX)))
            .collect();
        Ok((
            Self {
                program: program.id(),
                levels: collector.levels,
                rhs_levels: collector.rhs_levels,
                luts: collector.luts,
            },
            uses,
        ))
    }
    /// Return the logical level of a wire.
    pub fn level(&self, wire: WireId) -> Result<usize, RnsBggError> {
        self.levels
            .get(&wire)
            .copied()
            .ok_or(RnsBggError::Graph("missing wire logical level".into()))
    }
    /// Return the one consuming logical level of a Static RHS ID.
    pub fn rhs_level(&self, rhs: RhsId) -> Result<usize, RnsBggError> {
        let levels = self.rhs_levels(rhs)?;
        match (levels.first(), levels.last()) {
            (Some(first), Some(last)) if first == last => Ok(*first),
            (Some(first), Some(last)) => {
                Err(RnsBggError::RhsLevelMismatch { rhs, expected: *first, actual: *last })
            }
            _ => Err(RnsBggError::Graph("RHS has no consuming level".into())),
        }
    }

    /// Return every consuming logical level of an RHS ID.
    pub fn rhs_levels(&self, rhs: RhsId) -> Result<&BTreeSet<usize>, RnsBggError> {
        self.rhs_levels.get(&rhs).ok_or(RnsBggError::Graph("RHS has no consuming level".into()))
    }
    /// Program identity committed by this workload.
    pub fn program(&self) -> crate::program::ExponentLutProgramId {
        self.program
    }
    /// The input level and identity of every LUT the Program evaluates on a
    /// private payload.
    pub fn luts(&self) -> &BTreeSet<(usize, LutId)> {
        &self.luts
    }
}

impl<'a> ExponentLUTProgramLowering for WorkloadCollector<'a> {
    type Wire = (WireId, usize);
    type Rhs = RhsId;
    type Error = RnsBggError;
    fn add(&mut self, lhs: &Self::Wire, rhs: &Self::Wire) -> Result<Self::Wire, Self::Error> {
        self.note_use(lhs.0, None);
        self.note_use(rhs.0, None);
        if lhs.1 != rhs.1 {
            return Err(RnsBggError::Parameters("add operands have different logical levels"));
        }
        let output = self.site()?.output;
        self.levels.insert(output, lhs.1);
        Ok((output, lhs.1))
    }
    fn mul_public(
        &mut self,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
    ) -> Result<Self::Wire, Self::Error> {
        let output = self.site()?.output;
        let lhs_knowledge = self
            .program
            .wire_knowledge(lhs.0)
            .map_err(|_| RnsBggError::Graph("invalid MulPublic lhs".into()))?;
        let rhs_knowledge = self
            .program
            .wire_knowledge(rhs.0)
            .map_err(|_| RnsBggError::Graph("invalid MulPublic rhs".into()))?;
        if lhs_knowledge != crate::program::AttributeKnowledge::PublicPayload {
            return Err(RnsBggError::Parameters("MulPublic LHS must be public"));
        }
        let private_rhs = rhs_knowledge == crate::program::AttributeKnowledge::PrivatePayload;
        self.note_use(lhs.0, private_rhs.then_some(rhs.1));
        self.note_use(rhs.0, None);
        let target = self.parameters.map_or(Ok(rhs.1), |parameters| {
            parameters.mul_public_target_level(lhs.1, rhs.1, rhs_knowledge)
        })?;
        self.levels.insert(output, target);
        Ok((output, target))
    }
    fn mul_private(
        &mut self,
        input: &Self::Wire,
        rhs: &Self::Rhs,
    ) -> Result<Self::Wire, Self::Error> {
        if self
            .program
            .wire_knowledge(input.0)
            .map_err(|_| RnsBggError::Graph("invalid MulPrivate input metadata".into()))? ==
            crate::program::AttributeKnowledge::PublicPayload
        {
            return Err(RnsBggError::Parameters("RNS-BGG MulPrivate requires a private input"));
        }
        self.note_use(input.0, None);
        let output = self.site()?.output;
        let level = input.1;
        let dynamic =
            self.parameters.is_some_and(|parameters| parameters.rhs_mode() == RhsMode::Dynamic);
        let levels = self.rhs_levels.entry(*rhs).or_default();
        if let Some(&rhs_level) = levels.first() &&
            rhs_level != level &&
            !dynamic
        {
            return Err(RnsBggError::RhsLevelMismatch {
                rhs: *rhs,
                expected: level,
                actual: rhs_level,
            });
        }
        levels.insert(level);
        let next = level.checked_add(1).ok_or(RnsBggError::Parameters("logical level overflow"))?;
        self.levels.insert(output, next);
        Ok((output, next))
    }
    /// A private LUT moves an exponent output to the next level and keeps a
    /// coefficient output (finalized once later). A public LUT ends with one
    /// ModDown in either representation (Static helpers carry the level's
    /// scale), so its output is at the next level.
    fn lut(&mut self, input: &Self::Wire, table: &LutTable) -> Result<Self::Wire, Self::Error> {
        let public = self
            .program
            .wire_knowledge(input.0)
            .map_err(|_| RnsBggError::Graph("invalid LUT input metadata".into()))? ==
            crate::program::AttributeKnowledge::PublicPayload;
        self.note_use(input.0, None);
        let output = self.site()?.output;
        if !public {
            self.luts
                .insert((input.1, self.program.lut_id_at(output).map_err(|_| RnsBggError::Lut)?));
        }
        let next = if table.output_representation().is_coefficient() && !public {
            input.1
        } else {
            input.1.checked_add(1).ok_or(RnsBggError::Parameters("logical level overflow"))?
        };
        self.levels.insert(output, next);
        Ok((output, next))
    }
    fn mul_small_constant(
        &mut self,
        input: &Self::Wire,
        _: &[i64],
        _: &IntExpr,
    ) -> Result<Self::Wire, Self::Error> {
        // A plaintext factor applies to the carrier in place.
        self.note_use(input.0, None);
        let output = self.site()?.output;
        self.levels.insert(output, input.1);
        Ok((output, input.1))
    }
}

impl<'a> ProgramLoweringContext for WorkloadCollector<'a> {
    fn replace_site(
        &mut self,
        site: Option<crate::program::LoweringSite>,
    ) -> Option<crate::program::LoweringSite> {
        std::mem::replace(&mut self.current_site, site)
    }
}

impl<'a> FamilyExecution<WorkloadCollector<'a>> for RnsBggNoiseFamilies {
    type WireFamily = (WireId, usize, FamilyRange);
    type RhsFamily = (RhsId, FamilyRange);

    fn wire_at(
        &mut self,
        _evaluator: &mut WorkloadCollector<'a>,
        family: &Self::WireFamily,
        index: &IntExpr,
    ) -> Result<(WireId, usize), ProgramLowerError<RnsBggError>> {
        if index.clone().canonicalize() != *index {
            return Err(ProgramLowerError::Program(
                crate::program::ProgramValidationError::InvalidStructure(
                    "workload family index is not canonical".into(),
                ),
            ));
        }
        Ok((family.0, family.1))
    }
    fn rhs_at(
        &mut self,
        family: &Self::RhsFamily,
        index: &IntExpr,
    ) -> Result<RhsId, ProgramLowerError<RnsBggError>> {
        if index.clone().canonicalize() != *index {
            return Err(ProgramLowerError::Program(
                crate::program::ProgramValidationError::InvalidStructure(
                    "workload RHS index is not canonical".into(),
                ),
            ));
        }
        Ok(family.0)
    }
    fn rhs_selected(
        &mut self,
        family: &Self::RhsFamily,
        _: crate::program::DataId,
        _: &IntExpr,
    ) -> Result<RhsId, ProgramLowerError<RnsBggError>> {
        Ok(family.0)
    }
    fn parallel(
        &mut self,
        evaluator: &mut WorkloadCollector<'a>,
        range: &FamilyRange,
        body: impl FnOnce(
            &mut Self,
            &mut WorkloadCollector<'a>,
            IntExpr,
        ) -> Result<(WireId, usize), ProgramLowerError<RnsBggError>>,
    ) -> Result<Self::WireFamily, ProgramLowerError<RnsBggError>> {
        let index = IntExpr::Add(Box::new(range.start().clone()), Box::new(IntExpr::LoopIndex(0)))
            .canonicalize();
        let (wire, level) = body(self, evaluator, index)?;
        Ok((wire, level, range.clone()))
    }
    fn sum(
        &mut self,
        evaluator: &mut WorkloadCollector<'a>,
        family: &Self::WireFamily,
    ) -> Result<(WireId, usize), ProgramLowerError<RnsBggError>> {
        let output = evaluator.site().map_err(ProgramLowerError::Backend)?.output;
        evaluator.levels.insert(output, family.1);
        Ok((output, family.1))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use mxx_backends::poly::dcrt::params::DCRTPolyParams;
    use num_bigint::BigUint;

    /// A public input consumed only as the public left operand of products
    /// with private right operands takes the lowest of their levels; one also
    /// consumed any other way stays at level zero.
    #[test]
    fn workload_encodes_public_inputs_at_their_first_consuming_level() {
        use crate::program::{AttributeKnowledge, Operand};
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let state = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let moved = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let pinned = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let rhs = builder.rhs_input().unwrap();
        let second_rhs = builder.rhs_input().unwrap();
        let level_one = builder.mul_private(Operand::Scalar(state), Operand::Scalar(rhs)).unwrap();
        let level_two =
            builder.mul_private(Operand::Scalar(level_one), Operand::Scalar(second_rhs)).unwrap();
        let first = builder.mul_public(Operand::Scalar(moved), Operand::Scalar(level_one)).unwrap();
        let second =
            builder.mul_public(Operand::Scalar(moved), Operand::Scalar(level_two)).unwrap();
        let third =
            builder.mul_public(Operand::Scalar(pinned), Operand::Scalar(level_two)).unwrap();
        let program = builder
            .finish(vec![
                Operand::Scalar(first),
                Operand::Scalar(second),
                Operand::Scalar(third),
                Operand::Scalar(pinned),
            ])
            .unwrap();
        let workload = RnsBggDerivedWorkload::from_program(&program).unwrap();
        assert_eq!(workload.level(state).unwrap(), 0);
        assert_eq!(workload.level(moved).unwrap(), 1);
        assert_eq!(workload.level(pinned).unwrap(), 0);
        assert_eq!(workload.level(first).unwrap(), 1);
        assert_eq!(workload.level(second).unwrap(), 2);
        assert_eq!(workload.level(third).unwrap(), 2);
    }

    #[test]
    fn workload_rejects_one_rhs_at_two_logical_levels_before_sampling() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_input().unwrap();
        let first = builder
            .mul_private(
                crate::program::Operand::Scalar(input),
                crate::program::Operand::Scalar(rhs),
            )
            .unwrap();
        let second = builder
            .mul_private(
                crate::program::Operand::Scalar(first),
                crate::program::Operand::Scalar(rhs),
            )
            .unwrap();
        let program = builder.finish(vec![crate::program::Operand::Scalar(second)]).unwrap();

        assert!(matches!(
            RnsBggDerivedWorkload::from_program(&program),
            Err(RnsBggError::RhsLevelMismatch { rhs: actual, expected: 1, actual: 0 })
                if actual == rhs
        ));
    }

    #[test]
    fn workload_allows_equal_plaintext_through_separate_rhs_ids() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
        let first_rhs = builder.rhs_input().unwrap();
        let second_rhs = builder.rhs_input().unwrap();
        let first = builder
            .mul_private(
                crate::program::Operand::Scalar(input),
                crate::program::Operand::Scalar(first_rhs),
            )
            .unwrap();
        let second = builder
            .mul_private(
                crate::program::Operand::Scalar(first),
                crate::program::Operand::Scalar(second_rhs),
            )
            .unwrap();
        let program = builder.finish(vec![crate::program::Operand::Scalar(second)]).unwrap();
        let workload = RnsBggDerivedWorkload::from_program(&program).unwrap();

        assert_eq!(workload.rhs_level(first_rhs).unwrap(), 0);
        assert_eq!(workload.rhs_level(second_rhs).unwrap(), 1);
    }

    #[test]
    fn workload_allows_same_level_rhs_reuse() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_input().unwrap();
        let left = builder
            .mul_private(
                crate::program::Operand::Scalar(input),
                crate::program::Operand::Scalar(rhs),
            )
            .unwrap();
        let right = builder
            .mul_private(
                crate::program::Operand::Scalar(input),
                crate::program::Operand::Scalar(rhs),
            )
            .unwrap();
        let output = builder
            .add(crate::program::Operand::Scalar(left), crate::program::Operand::Scalar(right))
            .unwrap();
        let program = builder.finish(vec![crate::program::Operand::Scalar(output)]).unwrap();

        assert_eq!(
            RnsBggDerivedWorkload::from_program(&program).unwrap().rhs_level(rhs).unwrap(),
            0
        );
    }

    #[test]
    fn workload_rejects_public_mul_private_and_moves_public_luts_down_a_level() {
        let mut public_mul_builder = crate::program::ExponentLutProgramBuilder::new();
        let public_input =
            public_mul_builder.input(crate::program::AttributeKnowledge::PublicPayload).unwrap();
        let rhs = public_mul_builder.rhs_input().unwrap();
        let output = public_mul_builder
            .mul_private(
                crate::program::Operand::Scalar(public_input),
                crate::program::Operand::Scalar(rhs),
            )
            .unwrap();
        let public_mul_program =
            public_mul_builder.finish(vec![crate::program::Operand::Scalar(output)]).unwrap();
        assert!(RnsBggDerivedWorkload::from_program(&public_mul_program).is_err());

        let mut private_mul_builder = crate::program::ExponentLutProgramBuilder::new();
        let private_input =
            private_mul_builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = private_mul_builder.rhs_input().unwrap();
        let output = private_mul_builder
            .mul_private(
                crate::program::Operand::Scalar(private_input),
                crate::program::Operand::Scalar(rhs),
            )
            .unwrap();
        let private_mul_program =
            private_mul_builder.finish(vec![crate::program::Operand::Scalar(output)]).unwrap();
        assert!(RnsBggDerivedWorkload::from_program(&private_mul_program).is_ok());

        let table = LutTable::unary(
            2,
            crate::program::ExpLayout { stride_log2: 0, width_log2: 1 },
            2,
            vec![0, 1],
            crate::program::LutOutputRepresentation::Exponent(crate::program::ExpLayout {
                stride_log2: 0,
                width_log2: 1,
            }),
        )
        .unwrap();
        let mut public_lut_builder = crate::program::ExponentLutProgramBuilder::new();
        let public_input =
            public_lut_builder.input(crate::program::AttributeKnowledge::PublicPayload).unwrap();
        let lut = public_lut_builder.register_lut(table.clone()).unwrap();
        let output =
            public_lut_builder.lut(crate::program::Operand::Scalar(public_input), lut).unwrap();
        let public_lut_program =
            public_lut_builder.finish(vec![crate::program::Operand::Scalar(output)]).unwrap();
        let workload = RnsBggDerivedWorkload::from_program(&public_lut_program).unwrap();
        assert_eq!(workload.level(output).unwrap(), 1);
        assert!(workload.luts().is_empty(), "a public LUT has no Static switching material");

        let mut private_lut_builder = crate::program::ExponentLutProgramBuilder::new();
        let private_input =
            private_lut_builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
        let lut = private_lut_builder.register_lut(table).unwrap();
        let output =
            private_lut_builder.lut(crate::program::Operand::Scalar(private_input), lut).unwrap();
        let private_lut_program =
            private_lut_builder.finish(vec![crate::program::Operand::Scalar(output)]).unwrap();
        let workload = RnsBggDerivedWorkload::from_program(&private_lut_program).unwrap();
        assert_eq!(workload.level(output).unwrap(), 1);
        assert_eq!(workload.luts().len(), 1);
    }

    #[test]
    fn workload_tracks_nested_family_rhs_consumers_once() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder
            .input_family(IntExpr::constant(4), crate::program::AttributeKnowledge::PrivatePayload)
            .unwrap();
        let rhs = builder.rhs_family(IntExpr::constant(4)).unwrap();
        let outer = builder
            .parallel(FamilyRange::full(IntExpr::constant(2)).unwrap(), |builder, outer_index| {
                let inner = builder.parallel(
                    FamilyRange::full(IntExpr::constant(2)).unwrap(),
                    |builder, inner_index| {
                        let index =
                            IntExpr::Add(Box::new(outer_index.clone()), Box::new(inner_index))
                                .canonicalize();
                        builder
                            .mul_private(
                                crate::program::Operand::Member {
                                    family: input,
                                    index: index.clone(),
                                },
                                crate::program::Operand::Member { family: rhs, index },
                            )
                            .map(crate::program::Operand::Scalar)
                    },
                )?;
                builder.sum(inner).map(crate::program::Operand::Scalar)
            })
            .unwrap();
        let output = builder.sum(outer).unwrap();
        let program = builder.finish(vec![crate::program::Operand::Scalar(output)]).unwrap();

        assert_eq!(
            RnsBggDerivedWorkload::from_program(&program).unwrap().rhs_level(rhs).unwrap(),
            0
        );
    }

    #[test]
    fn workload_distinguishes_logical_levels_when_physical_moduli_are_equal() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_input().unwrap();
        let first = builder
            .mul_private(
                crate::program::Operand::Scalar(input),
                crate::program::Operand::Scalar(rhs),
            )
            .unwrap();
        let second_rhs = builder.rhs_input().unwrap();
        let second = builder
            .mul_private(
                crate::program::Operand::Scalar(first),
                crate::program::Operand::Scalar(second_rhs),
            )
            .unwrap();
        let program = builder.finish(vec![crate::program::Operand::Scalar(second)]).unwrap();
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let parameters =
            RnsBggParameters::new(vec![level.clone(), level.clone(), level], 1, 1).unwrap();
        assert_eq!(parameters.scale(0).unwrap(), &BigUint::from(1u8));
        let workload = RnsBggDerivedWorkload::from_program(&program).unwrap();

        assert_eq!(workload.rhs_level(rhs).unwrap(), 0);
        assert_eq!(workload.rhs_level(second_rhs).unwrap(), 1);
    }

    #[test]
    fn workload_does_not_prepare_unused_rhs_declarations() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_input().unwrap();
        let program = builder.finish(vec![crate::program::Operand::Scalar(input)]).unwrap();
        let workload = RnsBggDerivedWorkload::from_program(&program).unwrap();

        assert!(matches!(workload.rhs_level(rhs), Err(RnsBggError::Graph(_))));
    }
}
