//! Indexed families are ordinary one-argument programs.
//!
//! The low-level
//! [`super::arena::ExprArena`] owns expression identity and validation; this module only adds
//! the family/program view needed by the next migration stage.  In particular, there is no
//! `Switch` node, selector authority, lane enumeration, or family-specific expression arena.

pub use super::arena::{
    ArenaError, ArenaToken, ClosedExprId, ExprArena, ExprId, FamilyDomain, ProgramInput,
    ProgramSignature, ResolvedValueType, SemanticFamilySourceIdentity, TrustedIndexRange,
    ValueOperator, ValueProgram, ValueProgramId,
};
use std::{
    cell::Cell,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    sync::Arc,
};

pub use super::arena::ScopedExprId;
use super::{
    arena::{ArtifactIdentity, ScopeProof},
    facts::{
        CoefficientBound, FactStore, MatrixFacts, NumericContract, PolynomialFacts, ValueFacts,
    },
};

/// A complete indexed family is one finalized value program with an exact, non-empty domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FamilyValueId(ValueProgramId);

impl FamilyValueId {
    pub const fn program(self) -> ValueProgramId {
        self.0
    }

    pub(crate) const fn from_program(program: ValueProgramId) -> Self {
        Self(program)
    }
}

/// An exact selector plan for a family-valued selection.
///
/// A closed selector is independent of the returned family's binder.  A program selector is
/// applied explicitly to that binder through `ProgramCall`; accepting a raw `ExprId` here would
/// let an outer `Argument(0)` silently alias the returned family's `Argument(0)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ProgramSelector {
    program: ValueProgramId,
    output_range: TrustedIndexRange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SelectionSelector {
    Closed(ClosedExprId),
    /// A unary selector program with a program-associated output-range capability. The range is
    /// separate from the program input domain: `i mod k` may consume a family-domain index while
    /// selecting one of `k` branches.
    Program(ProgramSelector),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct FamilyRecord {
    domain: FamilyDomain,
    element_type: ResolvedValueType,
    body: ExprId,
    /// Source/explicit programs are intentionally opaque at access time.  Generated programs
    /// are the only programs eligible for beta reduction.
    reducible: bool,
    artifact: Option<ArtifactIdentity>,
    explicit_matrix_facts: Option<MatrixFacts>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct FamilyKey {
    domain: FamilyDomain,
    element_type: ResolvedValueType,
    body: ExprId,
    reducible: bool,
    artifact: Option<ArtifactIdentity>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct ProgramKey {
    signature: ProgramSignature,
    root: ExprId,
}

type RootValidationFacts = BTreeSet<(u32, ResolvedValueType)>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(crate) enum BetaReason {
    ScalarFamilyGet,
    SelectorRangeExposure,
    MatrixFactObservation,
    GadgetFactFallback,
    ExplicitFamilyFactSummary,
    FamilySelectBranch,
    RelationLiftEndpoint,
    ResidualRoot,
    DecoderRoot,
    JobRelationValidation,
    NormalizerSpecialization,
    Other,
}

pub(crate) const BETA_REASON_COUNT: usize = 12;

impl BetaReason {
    pub(crate) const ALL: [Self; BETA_REASON_COUNT] = [
        Self::ScalarFamilyGet,
        Self::SelectorRangeExposure,
        Self::MatrixFactObservation,
        Self::GadgetFactFallback,
        Self::ExplicitFamilyFactSummary,
        Self::FamilySelectBranch,
        Self::RelationLiftEndpoint,
        Self::ResidualRoot,
        Self::DecoderRoot,
        Self::JobRelationValidation,
        Self::NormalizerSpecialization,
        Self::Other,
    ];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::ScalarFamilyGet => "scalar-family-get",
            Self::SelectorRangeExposure => "selector-range-exposure",
            Self::MatrixFactObservation => "matrix-fact-observation",
            Self::GadgetFactFallback => "gadget-fact-fallback",
            Self::ExplicitFamilyFactSummary => "explicit-family-fact-summary",
            Self::FamilySelectBranch => "family-select-branch",
            Self::RelationLiftEndpoint => "relation-lift-endpoint",
            Self::ResidualRoot => "residual-root",
            Self::DecoderRoot => "decoder-root",
            Self::JobRelationValidation => "job-relation-validation",
            Self::NormalizerSpecialization => "normalizer-specialization",
            Self::Other => "other",
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ProgramDiagnosticCounters {
    pub family_calls: u64,
    pub beta_program_call_sidecar_hits: u64,
    pub beta_program_call_sidecar_misses: u64,
    pub beta_program_call_sidecar_entries: u64,
    pub beta_program_call_early_source_hits: u64,
    pub beta_program_call_early_source_input_skips: u64,
    pub beta_identity_shortcuts: u64,
    pub beta_nodes_visited: u64,
    pub beta_closed_subtrees_reused: u64,
    pub root_validation_facts_hits: u64,
    pub root_validation_facts_misses: u64,
    pub root_validation_facts_entries: u64,
    pub root_validation_nodes_visited: u64,
    pub root_validation_cached_subroot_skips: u64,
    pub beta_reason_misses: [u64; BETA_REASON_COUNT],
    pub beta_reason_visits: [u64; BETA_REASON_COUNT],
    pub beta_reason_expr_allocations: [u64; BETA_REASON_COUNT],
}

/// The single job-local authority for finalized value programs and indexed families.
///
/// Families are views over ordinary one-argument programs.  They do not introduce another
/// expression arena or program-ID namespace; `FamilyValueId` is only a validated view of the
/// canonical `ValueProgramId` stored here.
pub struct ProgramArena {
    token: ArenaToken,
    programs: Vec<ValueProgram>,
    interner: BTreeMap<ProgramKey, u32>,
    families: BTreeMap<FamilyValueId, FamilyRecord>,
    family_intern: BTreeMap<FamilyKey, FamilyValueId>,
    expression_token: Option<ArenaToken>,
    /// Presence records both successful ownership validation and the exact sorted free arguments.
    /// Expression nodes are immutable and this arena binds to one expression authority.
    root_validation_facts: BTreeMap<ExprId, RootValidationFacts>,
    diagnostic_counters_enabled: bool,
    diagnostic_counters: Cell<ProgramDiagnosticCounters>,
}

impl Default for ProgramArena {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgramArena {
    pub fn new() -> Self {
        Self {
            token: ArenaToken::fresh(),
            programs: Vec::new(),
            interner: BTreeMap::new(),
            families: BTreeMap::new(),
            family_intern: BTreeMap::new(),
            expression_token: None,
            root_validation_facts: BTreeMap::new(),
            diagnostic_counters_enabled: tracing::enabled!(
                target: "mxx_correctness::operational_noise",
                tracing::Level::INFO
            ),
            diagnostic_counters: Cell::new(ProgramDiagnosticCounters::default()),
        }
    }

    pub fn token(&self) -> ArenaToken {
        self.token
    }

    pub fn len(&self) -> usize {
        self.programs.len()
    }

    pub(crate) fn diagnostic_counters(&self) -> ProgramDiagnosticCounters {
        self.diagnostic_counters.get()
    }

    #[cfg(test)]
    pub(crate) fn enable_diagnostic_counters(&mut self) {
        self.diagnostic_counters_enabled = true;
    }

    fn update_diagnostic_counters(&self, update: impl FnOnce(&mut ProgramDiagnosticCounters)) {
        if !self.diagnostic_counters_enabled {
            return;
        }
        let mut counters = self.diagnostic_counters.get();
        update(&mut counters);
        self.diagnostic_counters.set(counters);
    }

    pub(crate) fn beta_reason_counters(
        &self,
    ) -> ([u64; BETA_REASON_COUNT], [u64; BETA_REASON_COUNT], [u64; BETA_REASON_COUNT]) {
        let counters = self.diagnostic_counters.get();
        (
            counters.beta_reason_misses,
            counters.beta_reason_visits,
            counters.beta_reason_expr_allocations,
        )
    }

    fn record_beta_miss(&self, reason: BetaReason) {
        self.update_diagnostic_counters(|counters| {
            counters.beta_reason_misses[reason.index()] =
                counters.beta_reason_misses[reason.index()].saturating_add(1);
        });
    }

    fn record_beta_visit(&self, reason: BetaReason, visits: u64) {
        self.update_diagnostic_counters(|counters| {
            counters.beta_reason_visits[reason.index()] =
                counters.beta_reason_visits[reason.index()].saturating_add(visits);
        });
    }

    fn record_expr_allocations(&self, reason: BetaReason, before: usize, after: usize) {
        let allocations = after.saturating_sub(before) as u64;
        self.update_diagnostic_counters(|counters| {
            counters.beta_reason_expr_allocations[reason.index()] =
                counters.beta_reason_expr_allocations[reason.index()].saturating_add(allocations);
        });
    }

    #[cfg(test)]
    fn root_validation_facts_len(&self) -> usize {
        self.root_validation_facts.len()
    }

    #[cfg(test)]
    pub(crate) fn family_scopes(&self) -> Vec<(ValueProgramId, FamilyDomain)> {
        self.families.iter().map(|(family, record)| (family.program(), record.domain)).collect()
    }

    pub fn finalize(
        &mut self,
        expressions: &mut ExprArena,
        signature: ProgramSignature,
        root: ExprId,
    ) -> Result<ValueProgramId, ArenaError> {
        expressions.check_id(root)?;
        self.bind_expressions(expressions)?;
        let new_validation_facts =
            self.validate_root_for_finalization(expressions, &signature, root)?;
        let key = ProgramKey { signature: signature.clone(), root };
        if let Some(slot) = self.interner.get(&key).copied() {
            let id = ValueProgramId::new(self.token, slot);
            expressions.register_program_signature(id, signature)?;
            if let Some(facts) = new_validation_facts {
                self.record_root_validation_facts(root, facts)?;
            }
            return Ok(id);
        }
        let slot =
            u32::try_from(self.programs.len()).map_err(|_| ArenaError::ProgramArenaExhausted)?;
        let id = ValueProgramId::new(self.token, slot);
        self.programs.push(ValueProgram { signature: signature.clone(), root });
        self.interner.insert(key, slot);
        expressions.register_program_signature(id, signature)?;
        if let Some(facts) = new_validation_facts {
            self.record_root_validation_facts(root, facts)?;
        }
        Ok(id)
    }

    pub fn program(&self, id: ValueProgramId) -> Result<&ValueProgram, ArenaError> {
        if id.arena != self.token {
            return Err(ArenaError::ForeignProgram { expected: self.token, actual: id.arena });
        }
        self.programs.get(id.slot as usize).ok_or(ArenaError::InvalidSlot { slot: id.slot })
    }

    pub(crate) fn selector(
        &self,
        expressions: &ExprArena,
        program: ValueProgramId,
    ) -> Result<SelectionSelector, ArenaError> {
        let output_range = self.selector_output_range(expressions, program)?;
        Ok(SelectionSelector::Program(ProgramSelector { program, output_range }))
    }

    /// Derive a selector's result range from its finalized root.  The caller cannot supply a
    /// range: only the supported binder-preserving roots receive a range capability.
    fn selector_output_range(
        &self,
        expressions: &ExprArena,
        program: ValueProgramId,
    ) -> Result<TrustedIndexRange, ArenaError> {
        let program_record = self.program(program)?;
        if program_record.signature.output != ResolvedValueType::Int ||
            program_record.signature.inputs.len() != 1
        {
            return Err(ArenaError::ProgramSignatureMismatch);
        }
        let input = &program_record.signature.inputs[0];
        if input.value_type != ResolvedValueType::Int {
            return Err(ArenaError::ProgramSignatureMismatch);
        }
        let Some(input_range) = input.trusted_index_range else {
            return Err(ArenaError::ProgramSignatureMismatch);
        };
        let root = expressions.node(program_record.root)?;
        let output_range = match &root.operator {
            ValueOperator::Argument { position: 0, value_type }
                if *value_type == ResolvedValueType::Int =>
            {
                input_range
            }
            ValueOperator::Scalar(super::arena::ScalarOperation::Remainder)
                if root.inputs.len() == 2 =>
            {
                if !self.selector_remainder_numerator_is_valid(expressions, root.inputs[0])? {
                    return Err(ArenaError::ProgramSignatureMismatch);
                }
                let divisor = expressions.node(root.inputs[1])?;
                let ValueOperator::Constant(super::arena::TypedConstant {
                    value: super::arena::ConstantValue::Int(value),
                    ..
                }) = &divisor.operator
                else {
                    return Err(ArenaError::ProgramSignatureMismatch);
                };
                let Some(divisor) = num_traits::ToPrimitive::to_u64(value) else {
                    return Err(ArenaError::ProgramSignatureMismatch);
                };
                if divisor == 0 {
                    return Err(ArenaError::ProgramSignatureMismatch);
                }
                TrustedIndexRange { minimum: 0, maximum_exclusive: divisor }
            }
            _ => return Err(ArenaError::ProgramSignatureMismatch),
        };
        output_range.nonempty()?;
        Ok(output_range)
    }

    /// The modulo selector capability is sound for an identity binder or for a direct shifted
    /// binder `(arg + closed-affine) % n`.  The shift is deliberately restricted to closed
    /// integer arithmetic: accepting an opaque source/program call would make the expression's
    /// selector semantics depend on an unproved value rather than on the positive remainder
    /// contract alone.
    fn selector_remainder_numerator_is_valid(
        &self,
        expressions: &ExprArena,
        numerator: ExprId,
    ) -> Result<bool, ArenaError> {
        let node = expressions.node(numerator)?;
        if matches!(
            &node.operator,
            ValueOperator::Argument { position: 0, value_type }
                if *value_type == ResolvedValueType::Int
        ) {
            return Ok(true);
        }
        let ValueOperator::Scalar(super::arena::ScalarOperation::Add) = &node.operator else {
            return Ok(false);
        };
        let [left, right] = node.inputs.as_ref() else {
            return Ok(false);
        };
        Ok((self.is_selector_argument(expressions, *left)? &&
            self.is_closed_affine_integer(expressions, *right)?) ||
            (self.is_selector_argument(expressions, *right)? &&
                self.is_closed_affine_integer(expressions, *left)?))
    }

    fn is_selector_argument(
        &self,
        expressions: &ExprArena,
        expression: ExprId,
    ) -> Result<bool, ArenaError> {
        let node = expressions.node(expression)?;
        Ok(matches!(
            &node.operator,
            ValueOperator::Argument { position: 0, value_type }
                if *value_type == ResolvedValueType::Int
        ))
    }

    fn is_closed_affine_integer(
        &self,
        expressions: &ExprArena,
        expression: ExprId,
    ) -> Result<bool, ArenaError> {
        if expressions.value_type(expression)? != &ResolvedValueType::Int ||
            expressions.close(expression).is_err()
        {
            return Ok(false);
        }
        let node = expressions.node(expression)?;
        match &node.operator {
            ValueOperator::Constant(super::arena::TypedConstant {
                value: super::arena::ConstantValue::Int(_),
                ..
            }) => Ok(true),
            ValueOperator::Scalar(
                super::arena::ScalarOperation::Add |
                super::arena::ScalarOperation::Subtract |
                super::arena::ScalarOperation::Multiply,
            ) if node.inputs.len() == 2 => Ok(self
                .is_closed_affine_integer(expressions, node.inputs[0])? &&
                self.is_closed_affine_integer(expressions, node.inputs[1])?),
            ValueOperator::Scalar(super::arena::ScalarOperation::Negate)
                if node.inputs.len() == 1 =>
            {
                self.is_closed_affine_integer(expressions, node.inputs[0])
            }
            _ => Ok(false),
        }
    }

    pub fn root(
        &self,
        expressions: &ExprArena,
        id: ValueProgramId,
    ) -> Result<ScopedExprId, ArenaError> {
        let program = self.program(id)?;
        self.validate_expression_binding(expressions)?;
        self.validate_program_ownership(expressions, program.root)?;
        let proof = self.scope_proof(expressions, id)?;
        expressions.scoped_from_proof(&proof, program.root)
    }

    pub fn scoped(
        &self,
        expressions: &ExprArena,
        program: ValueProgramId,
        expression: ExprId,
    ) -> Result<ScopedExprId, ArenaError> {
        let value_program = self.program(program)?;
        self.validate_expression_binding(expressions)?;
        self.validate_program_ownership(expressions, expression)?;
        self.validate_free_arguments(expressions, &value_program.signature, expression)?;
        let proof = self.scope_proof(expressions, program)?;
        expressions.scoped_from_proof(&proof, expression)
    }

    pub(crate) fn scope_proof(
        &self,
        expressions: &ExprArena,
        program: ValueProgramId,
    ) -> Result<ScopeProof, ArenaError> {
        let value_program = self.program(program)?;
        self.validate_expression_binding(expressions)?;
        self.validate_program_ownership(expressions, value_program.root)?;
        expressions.scope_proof(program, value_program.root)
    }

    fn validate_free_arguments(
        &self,
        expressions: &ExprArena,
        signature: &ProgramSignature,
        root: ExprId,
    ) -> Result<(), ArenaError> {
        for (position, actual) in expressions.free_arguments(root)? {
            let Some(input) = signature.inputs.get(position as usize) else {
                return Err(ArenaError::FreeArgumentEscapes { position });
            };
            if actual != input.value_type {
                return Err(ArenaError::TypeMismatch {
                    operator: "ProgramSignature".to_owned(),
                    position: position as usize,
                    expected: input.value_type.clone(),
                    actual,
                });
            }
        }
        Ok(())
    }

    fn validate_root_for_finalization(
        &self,
        expressions: &ExprArena,
        signature: &ProgramSignature,
        root: ExprId,
    ) -> Result<Option<RootValidationFacts>, ArenaError> {
        let new_facts = if self.root_validation_facts.get(&root).is_some() {
            self.update_diagnostic_counters(|counters| {
                counters.root_validation_facts_hits =
                    counters.root_validation_facts_hits.saturating_add(1);
            });
            None
        } else {
            self.update_diagnostic_counters(|counters| {
                counters.root_validation_facts_misses =
                    counters.root_validation_facts_misses.saturating_add(1);
            });
            Some(self.collect_root_validation_facts(expressions, root)?)
        };
        let facts = new_facts
            .as_ref()
            .or_else(|| self.root_validation_facts.get(&root))
            .expect("cached or newly computed root validation facts");
        if expressions.value_type(root)? != &signature.output {
            return Err(ArenaError::ProgramOutputMismatch);
        }
        self.validate_free_argument_facts(signature, facts)?;
        Ok(new_facts)
    }

    fn record_root_validation_facts(
        &mut self,
        root: ExprId,
        facts: RootValidationFacts,
    ) -> Result<(), ArenaError> {
        if let Some(existing) = self.root_validation_facts.get(&root) {
            if existing != &facts {
                return Err(ArenaError::ProgramSignatureMismatch);
            }
        } else {
            self.root_validation_facts.insert(root, facts);
        }
        self.update_diagnostic_counters(|counters| {
            counters.root_validation_facts_entries = self.root_validation_facts.len() as u64;
        });
        Ok(())
    }

    fn collect_root_validation_facts(
        &self,
        expressions: &ExprArena,
        root: ExprId,
    ) -> Result<RootValidationFacts, ArenaError> {
        let mut seen = HashSet::new();
        let mut work = vec![root];
        let mut free_arguments = BTreeSet::new();
        let mut nodes_visited = 0_u64;
        let mut cached_subroot_skips = 0_u64;
        while let Some(id) = work.pop() {
            if !seen.insert(id.slot()) {
                continue;
            }
            if let Some(cached_facts) = self.root_validation_facts.get(&id) {
                free_arguments.extend(cached_facts.iter().cloned());
                cached_subroot_skips = cached_subroot_skips.saturating_add(1);
                continue;
            }
            nodes_visited = nodes_visited.saturating_add(1);
            let node = expressions.node(id)?;
            if let ValueOperator::ProgramCall { program } = node.operator {
                if program.arena != self.token {
                    return Err(ArenaError::ForeignProgram {
                        expected: self.token,
                        actual: program.arena,
                    });
                }
                self.program(program)?;
            }
            if let ValueOperator::Argument { position, ref value_type } = node.operator {
                free_arguments.insert((position, value_type.clone()));
            }
            work.extend(node.inputs.iter().copied());
        }
        self.update_diagnostic_counters(|counters| {
            counters.root_validation_nodes_visited =
                counters.root_validation_nodes_visited.saturating_add(nodes_visited);
            counters.root_validation_cached_subroot_skips =
                counters.root_validation_cached_subroot_skips.saturating_add(cached_subroot_skips);
        });
        Ok(free_arguments)
    }

    fn validate_free_argument_facts(
        &self,
        signature: &ProgramSignature,
        free_arguments: &RootValidationFacts,
    ) -> Result<(), ArenaError> {
        for (position, actual) in free_arguments {
            let Some(input) = signature.inputs.get(*position as usize) else {
                return Err(ArenaError::FreeArgumentEscapes { position: *position });
            };
            if actual != &input.value_type {
                return Err(ArenaError::TypeMismatch {
                    operator: "ProgramSignature".to_owned(),
                    position: *position as usize,
                    expected: input.value_type.clone(),
                    actual: actual.clone(),
                });
            }
        }
        Ok(())
    }

    /// A program may call only programs finalized by this exact authority.  The token check is
    /// performed on every reachable call, rather than only on the root, so a foreign callee
    /// cannot hide below an otherwise valid DAG.
    fn validate_program_ownership(
        &self,
        expressions: &ExprArena,
        root: ExprId,
    ) -> Result<(), ArenaError> {
        let mut seen = std::collections::BTreeSet::new();
        let mut work = vec![root];
        while let Some(id) = work.pop() {
            if !seen.insert(id.slot()) {
                continue;
            }
            let node = expressions.node(id)?;
            if let ValueOperator::ProgramCall { program } = node.operator {
                if program.arena != self.token {
                    return Err(ArenaError::ForeignProgram {
                        expected: self.token,
                        actual: program.arena,
                    });
                }
                self.program(program)?;
            }
            work.extend(node.inputs.iter().copied());
        }
        Ok(())
    }

    /// Validate an expression which is scoped to this program but is not necessarily reachable
    /// from the program root. Relation RHS normal forms use this detached path after exposing a
    /// factor that was created in a separate expression branch. The expression arena and every
    /// reachable `ProgramCall` must still belong to this exact program authority, and all free
    /// arguments must match the target program signature.
    pub(crate) fn validate_detached_expression(
        &self,
        expressions: &ExprArena,
        program: ValueProgramId,
        root: ExprId,
    ) -> Result<(), ArenaError> {
        let signature = self.program(program)?.signature.clone();
        self.validate_expression_binding(expressions)?;
        self.validate_program_ownership(expressions, root)?;
        self.validate_free_arguments(expressions, &signature, root)
    }

    /// Build a typed ordinary `ProgramCall`.  Program finalization guarantees that a call can
    /// target only an already finalized program, so the call graph is acyclic by construction.
    pub fn call(
        &self,
        expressions: &mut ExprArena,
        program: ValueProgramId,
        inputs: &[ExprId],
    ) -> Result<ExprId, ArenaError> {
        self.program(program)?;
        self.validate_expression_binding(expressions)?;
        for input in inputs {
            self.validate_program_ownership(expressions, *input)?;
        }
        expressions.intern_slice(ValueOperator::ProgramCall { program }, inputs)
    }

    fn validate_expression_binding(&self, expressions: &ExprArena) -> Result<(), ArenaError> {
        let expected = self.expression_token.ok_or(ArenaError::InvalidScopeProof)?;
        if expected != expressions.token() {
            return Err(ArenaError::ForeignExpression { expected, actual: expressions.token() });
        }
        Ok(())
    }

    fn bind_expressions(&mut self, expressions: &ExprArena) -> Result<(), ArenaError> {
        if let Some(expected) = self.expression_token {
            if expected != expressions.token() {
                return Err(ArenaError::ForeignExpression { expected, actual: expressions.token() });
            }
        } else {
            self.expression_token = Some(expressions.token());
        }
        Ok(())
    }

    /// Construct a family from a finalized one-argument body.  The input range is copied from
    /// the exact signature; callers cannot widen or replace it after construction.
    pub fn generated_family(
        &mut self,
        expressions: &mut ExprArena,
        signature: ProgramSignature,
        body: ExprId,
    ) -> Result<FamilyValueId, ArenaError> {
        let domain = family_signature_domain(&signature)?;
        let element_type = signature.output.clone();
        let program = self.finalize(expressions, signature, body)?;
        self.intern_family(
            expressions,
            program,
            FamilyRecord {
                domain,
                element_type,
                body,
                reducible: true,
                artifact: None,
                explicit_matrix_facts: None,
            },
        )
    }

    /// Convenience constructor for a generated family whose output type is taken from `body`.
    pub fn generated_family_from_body(
        &mut self,
        expressions: &mut ExprArena,
        domain: FamilyDomain,
        body: ExprId,
    ) -> Result<FamilyValueId, ArenaError> {
        let output = expressions.value_type(body)?.clone();
        self.generated_family(expressions, family_signature(domain, output), body)
    }

    /// Build a generated family whose complete body remains an opaque `ProgramCall` at access
    /// sites.  This is used for relation witnesses: the family handle, rather than a synthetic
    /// body-shaped wrapper, is the provenance authority for the preimage factor.  The body is
    /// still beta-reducible explicitly through [`ProgramArena::beta_reduce`] when a registered
    /// relation specializes its plans.
    pub fn opaque_generated_family_from_body(
        &mut self,
        expressions: &mut ExprArena,
        domain: FamilyDomain,
        body: ExprId,
    ) -> Result<FamilyValueId, ArenaError> {
        let output = expressions.value_type(body)?.clone();
        let key = FamilyKey {
            domain,
            element_type: output.clone(),
            body,
            reducible: false,
            artifact: None,
        };
        if let Some(existing) = self.family_intern.get(&key).copied() {
            return Ok(existing);
        }
        let signature = family_signature(domain, output.clone());
        let program = self.finalize_fresh(expressions, signature, body)?;
        self.intern_family(
            expressions,
            program,
            FamilyRecord {
                domain,
                element_type: output,
                body,
                reducible: false,
                artifact: None,
                explicit_matrix_facts: None,
            },
        )
    }

    fn finalize_fresh(
        &mut self,
        expressions: &mut ExprArena,
        signature: ProgramSignature,
        root: ExprId,
    ) -> Result<ValueProgramId, ArenaError> {
        expressions.check_id(root)?;
        self.bind_expressions(expressions)?;
        let new_validation_facts =
            self.validate_root_for_finalization(expressions, &signature, root)?;
        let slot =
            u32::try_from(self.programs.len()).map_err(|_| ArenaError::ProgramArenaExhausted)?;
        let id = ValueProgramId::new(self.token, slot);
        self.programs.push(ValueProgram { signature: signature.clone(), root });
        expressions.register_program_signature(id, signature)?;
        if let Some(facts) = new_validation_facts {
            self.record_root_validation_facts(root, facts)?;
        }
        Ok(id)
    }

    /// Build `lambda i. OpaqueFamilyElement(source)`.  The dynamic index remains an input to the
    /// ordinary `ProgramCall`, so distinct complete index expressions remain distinct without
    /// enumerating the declared domain.  `explicit_matrix_facts` carries the caller-declared
    /// element contract; the opaque body cannot be inspected, so this is the only fact authority
    /// a source family can ever have.
    pub fn source_family(
        &mut self,
        expressions: &mut ExprArena,
        source: SemanticFamilySourceIdentity,
        explicit_matrix_facts: Option<MatrixFacts>,
    ) -> Result<FamilyValueId, ArenaError> {
        if let Some(facts) = &explicit_matrix_facts {
            let declared = ResolvedValueType::Matrix(facts.matrix_type.clone());
            if declared != source.element_type {
                return Err(ArenaError::TypeMismatch {
                    operator: "SourceFamily".to_owned(),
                    position: 0,
                    expected: source.element_type,
                    actual: declared,
                });
            }
        }
        let domain = source.domain.nonempty()?;
        let argument = expressions.intern_argument(0, ResolvedValueType::Int)?;
        let body = expressions.intern(
            ValueOperator::OpaqueFamilyElement { source: source.clone() },
            Box::new([argument]),
        )?;
        let signature = family_signature(domain, source.element_type.clone());
        let program = self.finalize(expressions, signature, body)?;
        self.intern_family(
            expressions,
            program,
            FamilyRecord {
                domain,
                element_type: source.element_type,
                body,
                reducible: false,
                artifact: source.artifact,
                explicit_matrix_facts,
            },
        )
    }

    /// Build `lambda i. ExplicitElement(i, values)`.  The physical values are inputs to one
    /// compact ordinary DAG body; access never expands the alternatives or walks the logical
    /// domain.
    pub fn explicit_family(
        &mut self,
        expressions: &mut ExprArena,
        facts: &FactStore,
        domain: FamilyDomain,
        values: Box<[ExprId]>,
    ) -> Result<FamilyValueId, ArenaError> {
        let domain = domain.nonempty()?;
        let width = domain.maximum_exclusive.checked_sub(domain.minimum).ok_or(
            ArenaError::InvalidRange {
                minimum: domain.minimum,
                maximum_exclusive: domain.maximum_exclusive,
            },
        )?;
        if usize::try_from(width).ok() != Some(values.len()) {
            return Err(ArenaError::InvalidArity {
                operator: "ExplicitFamily".to_owned(),
                expected: usize::try_from(width).unwrap_or(usize::MAX),
                actual: values.len(),
            });
        }
        let Some(first) = values.first().copied() else {
            return Err(ArenaError::EmptyFamilyDomain);
        };
        let element_type = expressions.value_type(first)?.clone();
        for (position, value) in values.iter().copied().enumerate() {
            let actual = expressions.value_type(value)?;
            if actual != &element_type {
                return Err(ArenaError::TypeMismatch {
                    operator: "ExplicitFamily".to_owned(),
                    position,
                    expected: element_type.clone(),
                    actual: actual.clone(),
                });
            }
        }
        // Fact views are derived from each exact compact value under this ProgramArena's
        // authority. Callers cannot pair an unrelated same-typed expression with a value merely
        // to lend it facts. A fact-bearing wrapper remains its own observation root; otherwise
        // reducible generated calls are transitively materialized only for summary inspection.
        let fact_values = if matches!(element_type, ResolvedValueType::Matrix(_)) {
            values
                .iter()
                .map(|value| {
                    if matches!(facts.facts(*value), Ok(ValueFacts::Matrix(_))) {
                        Ok(*value)
                    } else {
                        self.materialize_reducible_generated_calls_with_reason(
                            expressions,
                            *value,
                            BetaReason::ExplicitFamilyFactSummary,
                        )
                    }
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            Vec::new()
        };
        let explicit_matrix_facts =
            self.explicit_matrix_summary(expressions, facts, &fact_values)?;
        let argument = expressions.intern_argument(0, ResolvedValueType::Int)?;
        let mut body_inputs = Vec::with_capacity(values.len() + 1);
        body_inputs.push(argument);
        body_inputs.extend(values.iter().copied());
        let body = expressions.intern(
            ValueOperator::ExplicitElement { domain, element_type: element_type.clone() },
            body_inputs.into_boxed_slice(),
        )?;
        let program =
            self.finalize(expressions, family_signature(domain, element_type.clone()), body)?;
        self.intern_family(
            expressions,
            program,
            FamilyRecord {
                domain,
                element_type,
                body,
                reducible: false,
                artifact: None,
                explicit_matrix_facts,
            },
        )
    }

    /// Summarize explicit matrix branches without enumerating a family domain.  Missing branch
    /// facts are intentionally represented by `None`; an explicit family remains valid, but no
    /// shape-only constant claim may be made for it.
    pub(crate) fn explicit_matrix_summary(
        &self,
        expressions: &ExprArena,
        facts: &FactStore,
        values: &[ExprId],
    ) -> Result<Option<MatrixFacts>, ArenaError> {
        let Some(first) = values.first().copied() else { return Ok(None) };
        let ResolvedValueType::Matrix(matrix_type) = expressions.value_type(first)?.clone() else {
            return Ok(None);
        };
        let mut summaries = Vec::with_capacity(values.len());
        for (position, value) in values.iter().copied().enumerate() {
            if expressions.value_type(value)? != &ResolvedValueType::Matrix(matrix_type.clone()) {
                return Err(ArenaError::TypeMismatch {
                    operator: "ExplicitFamilyFacts".to_owned(),
                    position,
                    expected: ResolvedValueType::Matrix(matrix_type.clone()),
                    actual: expressions.value_type(value)?.clone(),
                });
            }
            let Ok(ValueFacts::Matrix(summary)) = facts.facts(value) else {
                return Ok(None);
            };
            summaries.push(summary);
        }
        Ok(join_matrix_facts(&summaries))
    }

    pub(crate) fn family_matrix_facts(
        &self,
        family: FamilyValueId,
    ) -> Result<Option<&MatrixFacts>, ArenaError> {
        Ok(self.family(family)?.explicit_matrix_facts.as_ref())
    }

    /// Look up the summary owned by the explicit-family callee of one exact ProgramCall.
    ///
    /// This is deliberately a read-only program/family lookup: it does not infer facts from the
    /// call's index or inputs and it never inserts anything into the global FactStore. In
    /// particular, binder-open calls may safely retrieve their producer-owned summary without
    /// pretending that the call itself is a closed value.
    pub(crate) fn program_call_family_matrix_facts(
        &self,
        expressions: &ExprArena,
        expression: ExprId,
    ) -> Result<Option<&MatrixFacts>, ArenaError> {
        let program = match expressions.node(expression)?.operator {
            ValueOperator::ProgramCall { program } => program,
            _ => return Ok(None),
        };
        let Some(family) = self.family_for_program(program) else { return Ok(None) };
        self.family_matrix_facts(family)
    }

    fn join_family_matrix_facts(
        &self,
        families: &[FamilyValueId],
    ) -> Result<Option<MatrixFacts>, ArenaError> {
        let Some(summaries) = families
            .iter()
            .map(|family| self.family(*family).ok()?.explicit_matrix_facts.as_ref())
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        Ok(join_matrix_facts(&summaries))
    }

    /// Access a family at one exact typed index.  The trusted range is mandatory; equal ranges
    /// do not correlate independent expressions, and containment is checked against this exact
    /// family domain.
    pub fn call_family(
        &self,
        expressions: &mut ExprArena,
        facts: &FactStore,
        family: FamilyValueId,
        index: ExprId,
    ) -> Result<ExprId, ArenaError> {
        if !facts.ranges_finalized() {
            return Err(ArenaError::IndexRangeRequired { id: index });
        }
        let index_range = facts
            .trusted_index_range(index)
            .map_err(|_| ArenaError::IndexRangeRequired { id: index })?;
        self.call_family_in_range(expressions, family, index, index_range)
    }

    pub(crate) fn call_family_in_range(
        &self,
        expressions: &mut ExprArena,
        family: FamilyValueId,
        index: ExprId,
        index_range: TrustedIndexRange,
    ) -> Result<ExprId, ArenaError> {
        self.call_family_in_range_with_reason(
            expressions,
            family,
            index,
            index_range,
            BetaReason::Other,
        )
    }

    pub(crate) fn call_family_in_range_with_reason(
        &self,
        expressions: &mut ExprArena,
        family: FamilyValueId,
        index: ExprId,
        index_range: TrustedIndexRange,
        reason: BetaReason,
    ) -> Result<ExprId, ArenaError> {
        let record = self.family(family)?;
        if !record.domain.contains(index_range) {
            return Err(ArenaError::InvalidRange {
                minimum: index_range.minimum,
                maximum_exclusive: index_range.maximum_exclusive,
            });
        }
        if expressions.value_type(index)? != &ResolvedValueType::Int {
            return Err(ArenaError::TypeMismatch {
                operator: "FamilyCall".to_owned(),
                position: 0,
                expected: ResolvedValueType::Int,
                actual: expressions.value_type(index)?.clone(),
            });
        }
        let node_count = expressions.node_count();
        let call = self.call(expressions, family.program(), &[index])?;
        self.record_expr_allocations(reason, node_count, expressions.node_count());
        self.update_diagnostic_counters(|counters| {
            counters.family_calls = counters.family_calls.saturating_add(1);
        });
        if record.reducible {
            self.reduce_validated_family_call_with_reason(expressions, family, index, call, reason)
        } else {
            self.materialize_reducible_generated_calls_with_reason(expressions, call, reason)
        }
    }

    /// Build a validated family call without eagerly expanding a reducible generated family.
    /// This is private composition machinery for binder-open nested parallel bodies. A compact
    /// body containing this call may be finalized as another generated family; callers must fully
    /// materialize it only before semantic exposure to relation validation, family analysis, or
    /// normalization.
    pub(crate) fn call_family_in_range_deferred_generated(
        &self,
        expressions: &mut ExprArena,
        family: FamilyValueId,
        index: ExprId,
        index_range: TrustedIndexRange,
    ) -> Result<ExprId, ArenaError> {
        self.call_family_in_range_deferred_generated_with_reason(
            expressions,
            family,
            index,
            index_range,
            BetaReason::Other,
        )
    }

    pub(crate) fn call_family_in_range_deferred_generated_with_reason(
        &self,
        expressions: &mut ExprArena,
        family: FamilyValueId,
        index: ExprId,
        index_range: TrustedIndexRange,
        reason: BetaReason,
    ) -> Result<ExprId, ArenaError> {
        let record = self.family(family)?;
        if !record.domain.contains(index_range) {
            return Err(ArenaError::InvalidRange {
                minimum: index_range.minimum,
                maximum_exclusive: index_range.maximum_exclusive,
            });
        }
        if expressions.value_type(index)? != &ResolvedValueType::Int {
            return Err(ArenaError::TypeMismatch {
                operator: "FamilyCall".to_owned(),
                position: 0,
                expected: ResolvedValueType::Int,
                actual: expressions.value_type(index)?.clone(),
            });
        }
        let node_count = expressions.node_count();
        let call = self.call(expressions, family.program(), &[index])?;
        self.record_expr_allocations(reason, node_count, expressions.node_count());
        self.update_diagnostic_counters(|counters| {
            counters.family_calls = counters.family_calls.saturating_add(1);
        });
        Ok(call)
    }

    fn reduce_validated_family_call(
        &self,
        expressions: &mut ExprArena,
        family: FamilyValueId,
        index: ExprId,
        call: ExprId,
    ) -> Result<ExprId, ArenaError> {
        self.reduce_validated_family_call_with_reason(
            expressions,
            family,
            index,
            call,
            BetaReason::Other,
        )
    }

    fn reduce_validated_family_call_with_reason(
        &self,
        expressions: &mut ExprArena,
        family: FamilyValueId,
        index: ExprId,
        call: ExprId,
        reason: BetaReason,
    ) -> Result<ExprId, ArenaError> {
        if let Some(reduced) = expressions.program_call_reduction(call)? {
            self.update_diagnostic_counters(|counters| {
                counters.beta_program_call_sidecar_hits =
                    counters.beta_program_call_sidecar_hits.saturating_add(1);
            });
            return Ok(reduced);
        }
        self.update_diagnostic_counters(|counters| {
            counters.beta_program_call_sidecar_misses =
                counters.beta_program_call_sidecar_misses.saturating_add(1);
        });
        self.record_beta_miss(reason);
        let reduced =
            self.beta_reduce_family_call_with_reason(expressions, family, index, call, reason)?;
        let reduced =
            self.materialize_reducible_generated_calls_with_reason(expressions, reduced, reason)?;
        expressions.record_program_call_reduction(call, reduced)?;
        let entries = expressions.program_call_reduction_count() as u64;
        self.update_diagnostic_counters(|counters| {
            counters.beta_program_call_sidecar_entries = entries;
        });
        Ok(reduced)
    }

    pub fn family_domain(&self, family: FamilyValueId) -> Result<FamilyDomain, ArenaError> {
        Ok(self.family(family)?.domain)
    }

    pub fn family_element_type(
        &self,
        family: FamilyValueId,
    ) -> Result<ResolvedValueType, ArenaError> {
        Ok(self.family(family)?.element_type.clone())
    }

    pub fn family_body(&self, family: FamilyValueId) -> Result<ExprId, ArenaError> {
        Ok(self.family(family)?.body)
    }

    pub(crate) fn family_is_reducible(&self, family: FamilyValueId) -> Result<bool, ArenaError> {
        Ok(self.family(family)?.reducible)
    }

    pub(crate) fn program_signature(
        &self,
        program: ValueProgramId,
    ) -> Result<&ProgramSignature, ArenaError> {
        Ok(&self.program(program)?.signature)
    }

    /// Return the family view for an already-finalized program, if that program is an indexed
    /// family in this arena.  The caller must not synthesize a new family around a program call:
    /// preserving this exact handle is what keeps relation provenance attached to the producer.
    pub fn family_for_program(&self, program: ValueProgramId) -> Option<FamilyValueId> {
        let family = FamilyValueId(program);
        self.families.contains_key(&family).then_some(family)
    }

    fn reducible_family_for_program(&self, program: ValueProgramId) -> Option<FamilyValueId> {
        let family = FamilyValueId(program);
        self.families.get(&family).is_some_and(|record| record.reducible).then_some(family)
    }

    /// Fully expand reducible generated-family calls in one not-yet-finalized composition body.
    /// Opaque/nonreducible callees remain folded, while their input expressions are still fully
    /// materialized before the opaque call is rebuilt.
    pub(crate) fn materialize_reducible_generated_calls(
        &self,
        expressions: &mut ExprArena,
        root: ExprId,
    ) -> Result<ExprId, ArenaError> {
        self.materialize_reducible_generated_calls_with_reason(expressions, root, BetaReason::Other)
    }

    pub(crate) fn materialize_reducible_generated_calls_with_reason(
        &self,
        expressions: &mut ExprArena,
        root: ExprId,
        reason: BetaReason,
    ) -> Result<ExprId, ArenaError> {
        expressions.value_type(root)?;
        let mut memo = HashMap::<ExprId, ExprId>::new();
        let mut work = vec![MaterializeFrame::Visit(root)];
        while let Some(frame) = work.pop() {
            match frame {
                MaterializeFrame::Visit(id) => {
                    if memo.contains_key(&id) {
                        continue;
                    }
                    let node = expressions.node_arc(id)?;
                    if let ValueOperator::ProgramCall { program } = node.operator {
                        if self.reducible_family_for_program(program).is_some() {
                            if let Some(materialized) = expressions.program_call_reduction(id)? {
                                if materialized == id ||
                                    expressions.value_type(materialized)? !=
                                        expressions.value_type(id)?
                                {
                                    return Err(ArenaError::ProgramSignatureMismatch);
                                }
                                self.update_diagnostic_counters(|counters| {
                                    counters.beta_program_call_early_source_hits = counters
                                        .beta_program_call_early_source_hits
                                        .saturating_add(1);
                                    counters.beta_program_call_early_source_input_skips = counters
                                        .beta_program_call_early_source_input_skips
                                        .saturating_add(node.inputs.len() as u64);
                                });
                                memo.insert(id, materialized);
                                continue;
                            }
                        }
                    }
                    work.push(MaterializeFrame::Rebuild { id, node: Arc::clone(&node) });
                    for input in node.inputs.iter().rev() {
                        if !memo.contains_key(input) {
                            work.push(MaterializeFrame::Visit(*input));
                        }
                    }
                }
                MaterializeFrame::Rebuild { id, node } => {
                    if memo.contains_key(&id) {
                        continue;
                    }
                    let inputs = node
                        .inputs
                        .iter()
                        .map(|input| {
                            memo.get(input)
                                .copied()
                                .ok_or(ArenaError::InvalidSlot { slot: input.slot() })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    if let ValueOperator::ProgramCall { program } = node.operator &&
                        let Some(family) = self.reducible_family_for_program(program)
                    {
                        let Some(&index) = inputs.first() else {
                            return Err(ArenaError::ProgramSignatureMismatch);
                        };
                        let node_count = expressions.node_count();
                        let call = expressions.intern(
                            ValueOperator::ProgramCall { program },
                            inputs.into_boxed_slice(),
                        )?;
                        self.record_expr_allocations(reason, node_count, expressions.node_count());
                        if let Some(materialized) = expressions.program_call_reduction(call)? {
                            self.update_diagnostic_counters(|counters| {
                                counters.beta_program_call_sidecar_hits =
                                    counters.beta_program_call_sidecar_hits.saturating_add(1);
                            });
                            memo.insert(id, materialized);
                            continue;
                        }
                        self.update_diagnostic_counters(|counters| {
                            counters.beta_program_call_sidecar_misses =
                                counters.beta_program_call_sidecar_misses.saturating_add(1);
                        });
                        self.record_beta_miss(reason);
                        let reduced = self.beta_reduce_family_call_with_reason(
                            expressions,
                            family,
                            index,
                            call,
                            reason,
                        )?;
                        if reduced == id {
                            return Err(ArenaError::ProgramSignatureMismatch);
                        }
                        work.push(MaterializeFrame::FinishReduced {
                            source: id,
                            reduced,
                            canonical_call: call,
                        });
                        if !memo.contains_key(&reduced) {
                            work.push(MaterializeFrame::Visit(reduced));
                        }
                    } else {
                        let node_count = expressions.node_count();
                        let value =
                            expressions.intern(node.operator.clone(), inputs.into_boxed_slice())?;
                        self.record_expr_allocations(reason, node_count, expressions.node_count());
                        memo.insert(id, value);
                    }
                }
                MaterializeFrame::FinishReduced { source, reduced, canonical_call } => {
                    let value = memo
                        .get(&reduced)
                        .copied()
                        .ok_or(ArenaError::InvalidSlot { slot: reduced.slot() })?;
                    if source != canonical_call {
                        if let Some(existing) = expressions.program_call_reduction(source)? {
                            if existing != value {
                                return Err(ArenaError::ProgramSignatureMismatch);
                            }
                        }
                    }
                    expressions.record_program_call_reduction(canonical_call, value)?;
                    if source != canonical_call {
                        expressions.record_program_call_reduction(source, value)?;
                    }
                    let entries = expressions.program_call_reduction_count() as u64;
                    self.update_diagnostic_counters(|counters| {
                        counters.beta_program_call_sidecar_entries = entries;
                    });
                    memo.insert(source, value);
                }
            }
        }
        let materialized =
            memo.get(&root).copied().ok_or(ArenaError::InvalidSlot { slot: root.slot() })?;
        self.ensure_no_reducible_generated_calls(expressions, materialized)?;
        Ok(materialized)
    }

    pub(crate) fn ensure_no_reducible_generated_calls(
        &self,
        expressions: &ExprArena,
        root: ExprId,
    ) -> Result<(), ArenaError> {
        expressions.value_type(root)?;
        let mut seen = BTreeSet::new();
        let mut work = vec![root];
        while let Some(id) = work.pop() {
            if !seen.insert(id) {
                continue;
            }
            let node = expressions.node(id)?;
            if let ValueOperator::ProgramCall { program } = node.operator {
                if self.reducible_family_for_program(program).is_some() {
                    return Err(ArenaError::ProgramSignatureMismatch);
                }
            }
            work.extend(node.inputs.iter().copied());
        }
        Ok(())
    }

    /// Beta-reduce one validated program application and transitively materialize every
    /// reducible generated-family call exposed by the result. Nonreducible callees remain opaque,
    /// while their already-evaluated input expressions are still rebuilt and checked.
    pub(crate) fn beta_reduce_materialized(
        &self,
        expressions: &mut ExprArena,
        program: ValueProgramId,
        arguments: &[ExprId],
    ) -> Result<ExprId, ArenaError> {
        self.beta_reduce_materialized_with_reason(
            expressions,
            program,
            arguments,
            BetaReason::Other,
        )
    }

    pub(crate) fn beta_reduce_materialized_with_reason(
        &self,
        expressions: &mut ExprArena,
        program: ValueProgramId,
        arguments: &[ExprId],
        reason: BetaReason,
    ) -> Result<ExprId, ArenaError> {
        let reduced = self.beta_reduce_with_reason(expressions, program, arguments, reason)?;
        self.materialize_reducible_generated_calls_with_reason(expressions, reduced, reason)
    }

    /// Validate and issue a scoped view for a materialized expression derived from, but not
    /// necessarily reachable by ordinary edges from, the compact finalized program root.
    pub(crate) fn detached_scoped(
        &self,
        expressions: &ExprArena,
        program: ValueProgramId,
        root: ExprId,
    ) -> Result<ScopedExprId, ArenaError> {
        self.validate_detached_expression(expressions, program, root)?;
        let proof = expressions.scope_proof(program, root)?;
        expressions.scoped_from_proof(&proof, root)
    }

    /// Compose two equal-domain families by constructing one ordinary operator body.
    pub fn pointwise_binary(
        &mut self,
        expressions: &mut ExprArena,
        left: FamilyValueId,
        right: FamilyValueId,
        operation: ValueOperator,
    ) -> Result<FamilyValueId, ArenaError> {
        let domain = self.same_domain(left, right)?;
        let argument = expressions.intern_argument(0, ResolvedValueType::Int)?;
        let range = TrustedIndexRange {
            minimum: domain.minimum,
            maximum_exclusive: domain.maximum_exclusive,
        };
        let left_value = self.call_family_in_range(expressions, left, argument, range)?;
        let right_value = self.call_family_in_range(expressions, right, argument, range)?;
        let body = expressions.intern_slice(operation, &[left_value, right_value])?;
        self.generated_family_from_body(expressions, domain, body)
    }

    pub fn pointwise_unary(
        &mut self,
        expressions: &mut ExprArena,
        family: FamilyValueId,
        operation: ValueOperator,
    ) -> Result<FamilyValueId, ArenaError> {
        let domain = self.family_domain(family)?;
        let argument = expressions.intern_argument(0, ResolvedValueType::Int)?;
        let range = TrustedIndexRange {
            minimum: domain.minimum,
            maximum_exclusive: domain.maximum_exclusive,
        };
        let value = self.call_family_in_range(expressions, family, argument, range)?;
        let body = expressions.intern_slice(operation, &[value])?;
        self.generated_family_from_body(expressions, domain, body)
    }

    /// `reindex(F, h)` derives the exact mapped-index range through the job fact authority.
    /// `h` is an explicit unary program, rather than a raw binder-open expression whose free
    /// argument could be captured accidentally.
    pub fn reindex(
        &mut self,
        expressions: &mut ExprArena,
        facts: &FactStore,
        family: FamilyValueId,
        map_program: ValueProgramId,
    ) -> Result<FamilyValueId, ArenaError> {
        let domain = self.family_domain(family)?;
        let (mapped, output_range) =
            self.instantiate_index_map(expressions, facts, map_program, domain)?;
        if !domain.contains(output_range) {
            return Err(ArenaError::InvalidRange {
                minimum: output_range.minimum,
                maximum_exclusive: output_range.maximum_exclusive,
            });
        }
        let body = self.call_family_in_range(expressions, family, mapped, output_range)?;
        self.generated_family_from_body(expressions, domain, body)
    }

    /// Compose `lambda i. operation(direct(i), offset(mapped(i)))` as one generated program.
    ///
    /// The result domain is exactly the direct family's domain.  `map_program` must be an
    /// existing unary integer program with that exact input domain, and its root's finalized
    /// range authorizes the single offset-family call.  Neither domain is enumerated.
    pub fn zip_offset(
        &mut self,
        expressions: &mut ExprArena,
        facts: &FactStore,
        direct: FamilyValueId,
        offset: FamilyValueId,
        map_program: ValueProgramId,
        operation: ValueOperator,
    ) -> Result<FamilyValueId, ArenaError> {
        let direct_domain = self.family_domain(direct)?;
        let (mapped, mapped_range) =
            self.instantiate_index_map(expressions, facts, map_program, direct_domain)?;
        let offset_domain = self.family_domain(offset)?;
        if !offset_domain.contains(mapped_range) {
            return Err(ArenaError::InvalidRange {
                minimum: mapped_range.minimum,
                maximum_exclusive: mapped_range.maximum_exclusive,
            });
        }

        let argument = expressions.intern_argument(0, ResolvedValueType::Int)?;
        let direct_range = TrustedIndexRange {
            minimum: direct_domain.minimum,
            maximum_exclusive: direct_domain.maximum_exclusive,
        };
        let direct_value =
            self.call_family_in_range(expressions, direct, argument, direct_range)?;
        let offset_value = self.call_family_in_range(expressions, offset, mapped, mapped_range)?;
        let body = expressions.intern_slice(operation, &[direct_value, offset_value])?;
        self.generated_family_from_body(expressions, direct_domain, body)
    }

    fn instantiate_index_map(
        &self,
        expressions: &mut ExprArena,
        facts: &FactStore,
        map_program: ValueProgramId,
        input_domain: FamilyDomain,
    ) -> Result<(ExprId, TrustedIndexRange), ArenaError> {
        let map = self.program(map_program)?;
        let input_range = TrustedIndexRange {
            minimum: input_domain.minimum,
            maximum_exclusive: input_domain.maximum_exclusive,
        };
        if map.signature.inputs.len() != 1 ||
            map.signature.inputs[0].value_type != ResolvedValueType::Int ||
            map.signature.inputs[0].trusted_index_range != Some(input_range) ||
            map.signature.output != ResolvedValueType::Int
        {
            return Err(ArenaError::ProgramSignatureMismatch);
        }
        let fact_root = map.root;
        if !facts.ranges_finalized() {
            return Err(ArenaError::IndexRangeRequired { id: fact_root });
        }
        let output_range = facts
            .trusted_scoped_index_range(map_program, fact_root)
            .map_err(|_| ArenaError::IndexRangeRequired { id: fact_root })?;
        let argument = expressions.intern_argument(0, ResolvedValueType::Int)?;
        let mapped = self.call(expressions, map_program, &[argument])?;
        Ok((mapped, output_range))
    }

    /// Gather a source family through a unary index program. The source and result domains may
    /// differ: the map's trusted input range is the result domain, while its trusted output range
    /// must be contained in the source domain. The generated body contains exactly one mapped
    /// source-family call; no selector or domain enumeration is performed.
    pub fn gather(
        &mut self,
        expressions: &mut ExprArena,
        facts: &FactStore,
        family: FamilyValueId,
        map_program: ValueProgramId,
    ) -> Result<FamilyValueId, ArenaError> {
        let source_domain = self.family_domain(family)?;
        let map = self.program(map_program)?;
        let result_domain = family_signature_domain(&map.signature)?;
        let (mapped, output_range) =
            self.instantiate_index_map(expressions, facts, map_program, result_domain)?;
        if !source_domain.contains(output_range) {
            return Err(ArenaError::InvalidRange {
                minimum: output_range.minimum,
                maximum_exclusive: output_range.maximum_exclusive,
            });
        }
        let body = self.call_family_in_range(expressions, family, mapped, output_range)?;
        self.generated_family_from_body(expressions, result_domain, body)
    }

    /// Zip many equal-domain families with one ordinary n-ary operator.  The operation receives
    /// only same-index program calls; no selector product or family-domain enumeration occurs.
    pub fn zip(
        &mut self,
        expressions: &mut ExprArena,
        families: &[FamilyValueId],
        operation: ValueOperator,
    ) -> Result<FamilyValueId, ArenaError> {
        let Some(&first) = families.first() else {
            return Err(ArenaError::InvalidArity {
                operator: "Zip".to_owned(),
                expected: 1,
                actual: 0,
            });
        };
        let domain = families.iter().try_fold(self.family_domain(first)?, |domain, family| {
            let other = self.family_domain(*family)?;
            if other == domain {
                Ok(domain)
            } else {
                Err(ArenaError::InvalidRange {
                    minimum: other.minimum,
                    maximum_exclusive: other.maximum_exclusive,
                })
            }
        })?;
        let argument = expressions.intern_argument(0, ResolvedValueType::Int)?;
        let range = TrustedIndexRange {
            minimum: domain.minimum,
            maximum_exclusive: domain.maximum_exclusive,
        };
        let mut inputs = Vec::with_capacity(families.len());
        for family in families {
            inputs.push(self.call_family_in_range(expressions, *family, argument, range)?);
        }
        let body = expressions.intern(operation, inputs.into_boxed_slice())?;
        self.generated_family_from_body(expressions, domain, body)
    }

    /// Selection is represented as one ordinary explicit-family marker plus its selected body
    /// calls in the private descriptor.  There is intentionally no `Switch`/variant node and no
    /// Cartesian traversal.  A selector is still an exact integer expression and is validated.
    pub fn select(
        &mut self,
        expressions: &mut ExprArena,
        facts: &FactStore,
        selector: SelectionSelector,
        families: &[FamilyValueId],
    ) -> Result<FamilyValueId, ArenaError> {
        let Some(&first) = families.first() else {
            return Err(ArenaError::InvalidArity {
                operator: "FamilySelection".to_owned(),
                expected: 1,
                actual: 0,
            });
        };
        let family_domain = self.family_domain(first)?;
        let element_type = self.family_element_type(first)?;
        for family in families {
            if self.family_domain(*family)? != family_domain ||
                self.family_element_type(*family)? != element_type
            {
                return Err(ArenaError::IncompatibleMatrixTypes);
            }
        }
        let argument = expressions.intern_argument(0, ResolvedValueType::Int)?;
        let family_range = TrustedIndexRange {
            minimum: family_domain.minimum,
            maximum_exclusive: family_domain.maximum_exclusive,
        };
        let (selector, selector_fact_root, explicit_selector_range) = match selector {
            SelectionSelector::Closed(selector) => {
                let selector = selector.expression();
                (selector, selector, None)
            }
            SelectionSelector::Program(ProgramSelector { program, output_range }) => {
                let selector_program = self.program(program)?;
                if selector_program.signature.inputs.len() != 1 ||
                    selector_program.signature.inputs[0].value_type != ResolvedValueType::Int ||
                    selector_program.signature.inputs[0].trusted_index_range !=
                        Some(family_range) ||
                    selector_program.signature.output != ResolvedValueType::Int
                {
                    return Err(ArenaError::ProgramSignatureMismatch);
                }
                if self.selector_output_range(expressions, program)? != output_range {
                    return Err(ArenaError::ProgramSignatureMismatch);
                }
                let selector_root = selector_program.root;
                let node_count = expressions.node_count();
                let selector = self.call(expressions, program, &[argument])?;
                self.record_expr_allocations(
                    BetaReason::FamilySelectBranch,
                    node_count,
                    expressions.node_count(),
                );
                (selector, selector_root, Some(output_range))
            }
        };
        if expressions.value_type(selector)? != &ResolvedValueType::Int {
            return Err(ArenaError::TypeMismatch {
                operator: "FamilySelection".to_owned(),
                position: 0,
                expected: ResolvedValueType::Int,
                actual: expressions.value_type(selector)?.clone(),
            });
        }
        let selector_range = if let Some(range) = explicit_selector_range {
            range
        } else {
            facts
                .ranges_finalized()
                .then(|| facts.trusted_index_range(selector_fact_root))
                .ok_or(ArenaError::IndexRangeRequired { id: selector_fact_root })?
                .map_err(|_| ArenaError::IndexRangeRequired { id: selector_fact_root })?
        };
        let selector_domain =
            FamilyDomain::new(selector_range.minimum, selector_range.maximum_exclusive)?
                .nonempty()?;
        let width = selector_domain.maximum_exclusive - selector_domain.minimum;
        if usize::try_from(width).ok() != Some(families.len()) {
            return Err(ArenaError::InvalidArity {
                operator: "FamilySelection".to_owned(),
                expected: usize::try_from(width).unwrap_or(usize::MAX),
                actual: families.len(),
            });
        }
        // Keep the selected branches in one ordinary ExplicitElement body.  The complete branch
        // values are DAG inputs and are stored once; this is not a `Switch` node and does not
        // enumerate the selector domain.
        let mut body_inputs = Vec::with_capacity(families.len() + 1);
        body_inputs.push(selector);
        for family in families {
            body_inputs.push(self.call_family_in_range_with_reason(
                expressions,
                *family,
                argument,
                family_range,
                BetaReason::FamilySelectBranch,
            )?);
        }
        let body = expressions.intern(
            ValueOperator::ExplicitElement {
                domain: selector_domain,
                element_type: element_type.clone(),
            },
            body_inputs.into_boxed_slice(),
        )?;
        let explicit_matrix_facts = self.join_family_matrix_facts(families)?;
        let signature = family_signature(family_domain, element_type);
        let program = self.finalize(expressions, signature, body)?;
        let id = self.intern_family(
            expressions,
            program,
            FamilyRecord {
                domain: family_domain,
                element_type: self.family_element_type(first)?,
                body,
                reducible: false,
                artifact: None,
                explicit_matrix_facts,
            },
        )?;
        Ok(id)
    }

    /// Iterative beta reduction for a finalized generated program.  No recursive Rust call stack
    /// or domain traversal is used; repeated subexpressions are memoized by expression slot.
    pub fn beta_reduce(
        &self,
        expressions: &mut ExprArena,
        program: ValueProgramId,
        arguments: &[ExprId],
    ) -> Result<ExprId, ArenaError> {
        self.beta_reduce_with_reason(expressions, program, arguments, BetaReason::Other)
    }

    fn beta_reduce_with_reason(
        &self,
        expressions: &mut ExprArena,
        program: ValueProgramId,
        arguments: &[ExprId],
        reason: BetaReason,
    ) -> Result<ExprId, ArenaError> {
        let value_program = self.program(program)?;
        if arguments.len() != value_program.signature.inputs.len() {
            return Err(ArenaError::InvalidArity {
                operator: "BetaReduce".to_owned(),
                expected: value_program.signature.inputs.len(),
                actual: arguments.len(),
            });
        }
        for (position, (argument, input)) in
            arguments.iter().zip(&value_program.signature.inputs).enumerate()
        {
            let actual = expressions.value_type(*argument)?;
            if actual != &input.value_type {
                return Err(ArenaError::TypeMismatch {
                    operator: "BetaReduce".to_owned(),
                    position,
                    expected: input.value_type.clone(),
                    actual: actual.clone(),
                });
            }
        }
        expressions.value_type(value_program.root)?;
        let identity = arguments.iter().zip(&value_program.signature.inputs).enumerate().try_fold(
            true,
            |identity, (position, (argument, input))| {
                let node = expressions.node(*argument)?;
                Ok::<_, ArenaError>(
                    identity &&
                        matches!(
                            &node.operator,
                            ValueOperator::Argument { position: actual, value_type }
                                if u32::try_from(position).ok() == Some(*actual) &&
                                    value_type == &input.value_type
                        ),
                )
            },
        )?;
        if identity {
            self.update_diagnostic_counters(|counters| {
                counters.beta_identity_shortcuts =
                    counters.beta_identity_shortcuts.saturating_add(1);
            });
            return Ok(value_program.root);
        }
        let nodes_before = expressions.node_count();
        let (reduced, substitution) = substitute_iterative(
            expressions,
            value_program.root,
            arguments,
            self.diagnostic_counters_enabled,
        )?;
        self.update_diagnostic_counters(|counters| {
            counters.beta_nodes_visited =
                counters.beta_nodes_visited.saturating_add(substitution.nodes_visited);
            counters.beta_closed_subtrees_reused = counters
                .beta_closed_subtrees_reused
                .saturating_add(substitution.closed_subtrees_reused);
        });
        self.record_beta_visit(reason, substitution.nodes_visited);
        self.record_expr_allocations(reason, nodes_before, expressions.node_count());
        Ok(reduced)
    }

    fn beta_reduce_family_call(
        &self,
        expressions: &mut ExprArena,
        family: FamilyValueId,
        index: ExprId,
        opaque_call: ExprId,
    ) -> Result<ExprId, ArenaError> {
        self.beta_reduce_family_call_with_reason(
            expressions,
            family,
            index,
            opaque_call,
            BetaReason::Other,
        )
    }

    fn beta_reduce_family_call_with_reason(
        &self,
        expressions: &mut ExprArena,
        family: FamilyValueId,
        index: ExprId,
        opaque_call: ExprId,
        reason: BetaReason,
    ) -> Result<ExprId, ArenaError> {
        let record = self.family(family)?;
        // Generated bodies are reduced from their body, not from the already interned call.  A
        // source/explicit call therefore stays one compact opaque ProgramCall.
        if !record.reducible {
            return Ok(opaque_call);
        }
        self.beta_reduce_with_reason(expressions, family.program(), &[index], reason)
    }

    fn family(&self, family: FamilyValueId) -> Result<&FamilyRecord, ArenaError> {
        if let Some(record) = self.families.get(&family) {
            return Ok(record);
        }
        if family.program().arena != self.token {
            return Err(ArenaError::ForeignProgram {
                expected: self.token,
                actual: family.program().arena,
            });
        }
        Err(ArenaError::UnknownProgram(family.program()))
    }

    fn same_domain(
        &self,
        left: FamilyValueId,
        right: FamilyValueId,
    ) -> Result<FamilyDomain, ArenaError> {
        let left_domain = self.family_domain(left)?;
        let right_domain = self.family_domain(right)?;
        if left_domain != right_domain {
            return Err(ArenaError::InvalidRange {
                minimum: right_domain.minimum,
                maximum_exclusive: right_domain.maximum_exclusive,
            });
        }
        Ok(left_domain)
    }

    fn intern_family(
        &mut self,
        _expressions: &ExprArena,
        program: ValueProgramId,
        record: FamilyRecord,
    ) -> Result<FamilyValueId, ArenaError> {
        let key = FamilyKey {
            domain: record.domain,
            element_type: record.element_type.clone(),
            body: record.body,
            reducible: record.reducible,
            artifact: record.artifact.clone(),
        };
        let id = FamilyValueId(program);
        if let Some(existing) = self.family_intern.get(&key).copied() {
            return Ok(existing);
        }
        self.families.insert(id, record);
        self.family_intern.insert(key, id);
        Ok(id)
    }

    /// The producer-owned export descriptor. Unexported generated/sampler families return None.
    pub fn family_artifact(
        &self,
        family: FamilyValueId,
    ) -> Result<Option<&ArtifactIdentity>, ArenaError> {
        Ok(self.family(family)?.artifact.as_ref())
    }
}

fn join_coefficient_bounds<'a>(
    bounds: impl Iterator<Item = &'a NumericContract<CoefficientBound>>,
) -> NumericContract<CoefficientBound> {
    let mut joined = NumericContract::Known(CoefficientBound::ExactZero);
    for bound in bounds {
        let NumericContract::Known(value) = bound else {
            return NumericContract::Missing;
        };
        if let NumericContract::Known(current) = &joined {
            if value > current {
                joined = NumericContract::Known(value.clone());
            }
        }
    }
    joined
}

fn join_polynomial_facts<'a>(
    facts: impl Iterator<Item = &'a NumericContract<PolynomialFacts>>,
) -> NumericContract<PolynomialFacts> {
    let mut support_upper = 0;
    let mut ring_dimension = None;
    for fact in facts {
        let NumericContract::Known(fact) = fact else {
            return NumericContract::Missing;
        };
        support_upper = support_upper.max(fact.support_upper);
        ring_dimension = Some(fact.ring_dimension);
    }
    let Some(ring_dimension) = ring_dimension else { return NumericContract::Missing };
    NumericContract::Known(
        PolynomialFacts::new(support_upper, ring_dimension)
            .expect("explicit-family polynomial support must fit its matrix ring"),
    )
}

fn join_matrix_facts(summaries: &[&MatrixFacts]) -> Option<MatrixFacts> {
    let first = summaries.first().expect("nonempty explicit matrix summaries");
    if summaries.iter().any(|summary| {
        summary.matrix_type != first.matrix_type || summary.metadata.layout != first.metadata.layout
    }) {
        // The family itself has already enforced one exact output type.  Layout and
        // fact-descriptor disagreements only mean that no common value summary exists;
        // they must not turn a previously valid explicit family into an invalid graph.
        return None;
    }
    let mut metadata = first.metadata.clone();
    metadata.is_constant_polynomial =
        summaries.iter().all(|summary| summary.metadata.is_constant_polynomial);
    metadata.canonical_coefficient_exclusive_upper = common_option(
        summaries
            .iter()
            .map(|summary| summary.metadata.canonical_coefficient_exclusive_upper.clone()),
    );
    metadata.known_zero_rows =
        common_option(summaries.iter().map(|summary| summary.metadata.known_zero_rows));
    let mut result = MatrixFacts::new(first.matrix_type.clone(), metadata);
    result.coefficient_bound =
        join_coefficient_bounds(summaries.iter().map(|summary| &summary.coefficient_bound));
    result.polynomial = join_polynomial_facts(summaries.iter().map(|summary| &summary.polynomial));
    Some(result)
}

fn common_option<T: Clone + Eq>(values: impl Iterator<Item = Option<T>>) -> Option<T> {
    let mut values = values;
    let first = values.next()?;
    values.all(|value| value == first).then_some(first).flatten()
}

fn family_signature(domain: FamilyDomain, output: ResolvedValueType) -> ProgramSignature {
    ProgramSignature {
        inputs: Box::new([ProgramInput {
            value_type: ResolvedValueType::Int,
            trusted_index_range: Some(TrustedIndexRange {
                minimum: domain.minimum,
                maximum_exclusive: domain.maximum_exclusive,
            }),
        }]),
        output,
    }
}

fn family_signature_domain(signature: &ProgramSignature) -> Result<FamilyDomain, ArenaError> {
    if signature.inputs.len() != 1 || signature.inputs[0].value_type != ResolvedValueType::Int {
        return Err(ArenaError::ProgramSignatureMismatch);
    }
    let Some(range) = signature.inputs[0].trusted_index_range else {
        return Err(ArenaError::ProgramSignatureMismatch);
    };
    range.nonempty().map_err(|_| ArenaError::EmptyFamilyDomain)
}

trait NonemptyRange {
    fn nonempty(self) -> Result<FamilyDomain, ArenaError>;
}

impl NonemptyRange for TrustedIndexRange {
    fn nonempty(self) -> Result<FamilyDomain, ArenaError> {
        FamilyDomain::new(self.minimum, self.maximum_exclusive)?.nonempty()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct SubstitutionDiagnostics {
    nodes_visited: u64,
    closed_subtrees_reused: u64,
}

enum SubstituteFrame {
    Visit(ExprId),
    Rebuild { id: ExprId, node: Arc<super::arena::ExprNode> },
}

enum MaterializeFrame {
    Visit(ExprId),
    Rebuild { id: ExprId, node: Arc<super::arena::ExprNode> },
    FinishReduced { source: ExprId, reduced: ExprId, canonical_call: ExprId },
}

fn substitute_iterative(
    expressions: &mut ExprArena,
    root: ExprId,
    arguments: &[ExprId],
    collect_diagnostics: bool,
) -> Result<(ExprId, SubstitutionDiagnostics), ArenaError> {
    expressions.value_type(root)?;
    let mut memo = BTreeMap::<u32, ExprId>::new();
    let mut diagnostics = SubstitutionDiagnostics::default();
    let mut work = vec![SubstituteFrame::Visit(root)];
    while let Some(frame) = work.pop() {
        match frame {
            SubstituteFrame::Visit(id) => {
                if memo.contains_key(&id.slot()) {
                    continue;
                }
                if collect_diagnostics {
                    diagnostics.nodes_visited = diagnostics.nodes_visited.saturating_add(1);
                }
                let (node, closed) = expressions.node_arc_and_closed(id)?;
                if let ValueOperator::Argument { position, .. } = &node.operator {
                    let position = *position;
                    let Some(replacement) = arguments.get(position as usize).copied() else {
                        return Err(ArenaError::FreeArgumentEscapes { position });
                    };
                    memo.insert(id.slot(), replacement);
                    continue;
                }
                if closed {
                    memo.insert(id.slot(), id);
                    if collect_diagnostics {
                        diagnostics.closed_subtrees_reused =
                            diagnostics.closed_subtrees_reused.saturating_add(1);
                    }
                    continue;
                }
                work.push(SubstituteFrame::Rebuild { id, node: Arc::clone(&node) });
                for child in node.inputs.iter().rev() {
                    if !memo.contains_key(&child.slot()) {
                        work.push(SubstituteFrame::Visit(*child));
                    }
                }
            }
            SubstituteFrame::Rebuild { id, node } => {
                if memo.contains_key(&id.slot()) {
                    continue;
                }
                let mut inputs = Vec::with_capacity(node.inputs.len());
                for child in node.inputs.iter() {
                    inputs.push(
                        memo.get(&child.slot())
                            .copied()
                            .ok_or(ArenaError::InvalidSlot { slot: child.slot() })?,
                    );
                }
                let value = expressions.intern(node.operator.clone(), inputs.into_boxed_slice())?;
                memo.insert(id.slot(), value);
            }
        }
    }
    let reduced =
        memo.get(&root.slot()).copied().ok_or(ArenaError::InvalidSlot { slot: root.slot() })?;
    Ok((reduced, diagnostics))
}

#[cfg(test)]
mod tests {
    use super::{
        super::{
            arena::{IndexFunctionDefinitionId, MatrixLayout, ResolvedMatrixType, TypedConstant},
            facts::{
                CoefficientBound, MatrixFacts, MatrixMetadata, NumericContract, PolynomialFacts,
                ValueFacts,
            },
        },
        *,
    };
    fn domain() -> FamilyDomain {
        FamilyDomain::new(0, 4).unwrap()
    }

    fn source(name: &str) -> SemanticFamilySourceIdentity {
        source_with_domain(name, domain())
    }

    fn source_with_domain(
        name: impl Into<String>,
        domain: FamilyDomain,
    ) -> SemanticFamilySourceIdentity {
        SemanticFamilySourceIdentity {
            stable_definition: name.into(),
            invocation: "invocation".to_owned(),
            element_type: ResolvedValueType::Int,
            domain,
            artifact: None,
        }
    }

    fn matrix_type() -> ResolvedMatrixType {
        ResolvedMatrixType::new(num_bigint::BigUint::from(17_u8), 1, 2, 2).unwrap()
    }

    fn matrix_expression(expressions: &mut ExprArena, event: u64) -> ExprId {
        expressions
            .intern(
                ValueOperator::Sampler {
                    event: super::super::arena::SampleEventId(event),
                    operation: super::super::arena::SamplerOperation::UniformResidue {
                        output: matrix_type(),
                    },
                },
                Box::new([]),
            )
            .unwrap()
    }

    fn matrix_facts(bound: Option<u64>, constant: bool, layout: MatrixLayout) -> MatrixFacts {
        let mut facts = MatrixFacts::new(matrix_type(), MatrixMetadata::new(layout));
        facts.metadata.is_constant_polynomial = constant;
        facts.coefficient_bound = match bound {
            Some(value) => NumericContract::Known(CoefficientBound::finite(value)),
            None => NumericContract::Missing,
        };
        facts.polynomial = NumericContract::Known(PolynomialFacts::new(2, 2).unwrap());
        facts
    }

    fn index_map_program(
        programs: &mut ProgramArena,
        expressions: &mut ExprArena,
        input_domain: FamilyDomain,
        root: ExprId,
    ) -> ValueProgramId {
        programs
            .finalize(expressions, family_signature(input_domain, ResolvedValueType::Int), root)
            .unwrap()
    }

    #[test]
    fn substitution_reuses_deep_closed_branch_along_open_spine() {
        const CLOSED_DEPTH: usize = 4_096;
        const OPEN_DEPTH: usize = 256;

        let mut expressions = ExprArena::new();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let one = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(1)), Box::new([]))
            .unwrap();
        let mut closed = one;
        for _ in 0..CLOSED_DEPTH {
            closed = expressions
                .intern_slice(
                    ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                    &[closed, one],
                )
                .unwrap();
        }
        assert!(expressions.is_closed(closed).unwrap());

        let mut root = argument;
        for _ in 0..OPEN_DEPTH {
            root = expressions
                .intern_slice(
                    ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                    &[root, closed],
                )
                .unwrap();
        }
        let replacement = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(7)), Box::new([]))
            .unwrap();

        let (reduced, diagnostics) =
            substitute_iterative(&mut expressions, root, &[replacement], true).unwrap();
        assert_eq!(diagnostics.nodes_visited, OPEN_DEPTH as u64 + 2);
        assert_eq!(diagnostics.closed_subtrees_reused, 1);

        let mut expected = replacement;
        for _ in 0..OPEN_DEPTH {
            expected = expressions
                .intern_slice(
                    ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                    &[expected, closed],
                )
                .unwrap();
        }
        assert_eq!(reduced, expected);
    }

    #[test]
    fn substitution_rebuilds_shared_open_dag_once_with_canonical_ids() {
        let mut expressions = ExprArena::new();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let one = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(1)), Box::new([]))
            .unwrap();
        let shared = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                &[argument, one],
            )
            .unwrap();
        let root = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                &[shared, shared],
            )
            .unwrap();
        let replacement = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(7)), Box::new([]))
            .unwrap();
        let nodes_before = expressions.node_count();

        let (reduced, diagnostics) =
            substitute_iterative(&mut expressions, root, &[replacement], true).unwrap();
        assert_eq!(expressions.node_count() - nodes_before, 2);
        assert_eq!(diagnostics.nodes_visited, 4);
        assert_eq!(diagnostics.closed_subtrees_reused, 1);

        let expected_shared = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                &[replacement, one],
            )
            .unwrap();
        let expected = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                &[expected_shared, expected_shared],
            )
            .unwrap();
        assert_eq!(reduced, expected);
        assert_eq!(expressions.node_count() - nodes_before, 2);
    }

    #[test]
    fn beta_reduce_keeps_arity_before_argument_validation_error_order() {
        let mut expressions = ExprArena::new();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let mut programs = ProgramArena::new();
        let program = programs
            .finalize(
                &mut expressions,
                ProgramSignature {
                    inputs: Box::new([ProgramInput {
                        value_type: ResolvedValueType::Int,
                        trusted_index_range: None,
                    }]),
                    output: ResolvedValueType::Int,
                },
                argument,
            )
            .unwrap();
        let boolean = expressions.intern_argument(0, ResolvedValueType::Bool).unwrap();
        let mut foreign_expressions = ExprArena::new();
        let foreign = foreign_expressions.intern_argument(0, ResolvedValueType::Int).unwrap();

        assert!(matches!(
            programs.beta_reduce(&mut expressions, program, &[foreign, foreign]),
            Err(ArenaError::InvalidArity { expected: 1, actual: 2, .. })
        ));
        assert!(matches!(
            programs.beta_reduce(&mut expressions, program, &[foreign]),
            Err(ArenaError::ForeignExpression { .. })
        ));
        assert!(matches!(
            programs.beta_reduce(&mut expressions, program, &[boolean]),
            Err(ArenaError::TypeMismatch { position: 0, .. })
        ));
    }

    #[test]
    fn canonical_argument_identity_shortcuts_deep_open_dag_without_allocation() {
        const DEPTH: usize = 4_096;

        let mut expressions = ExprArena::new();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let one = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(1)), Box::new([]))
            .unwrap();
        let mut root = argument;
        for _ in 0..DEPTH {
            root = expressions
                .intern_slice(
                    ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                    &[root, one],
                )
                .unwrap();
        }
        let signature = ProgramSignature {
            inputs: Box::new([ProgramInput {
                value_type: ResolvedValueType::Int,
                trusted_index_range: None,
            }]),
            output: ResolvedValueType::Int,
        };
        let mut programs = ProgramArena::new();
        programs.diagnostic_counters_enabled = true;
        let program = programs.finalize(&mut expressions, signature, root).unwrap();
        let nodes_before = expressions.node_count();

        let reduced = programs.beta_reduce(&mut expressions, program, &[argument]).unwrap();
        assert_eq!(reduced, root);
        assert_eq!(expressions.node_count(), nodes_before);
        let diagnostics = programs.diagnostic_counters();
        assert_eq!(diagnostics.beta_identity_shortcuts, 1);
        assert_eq!(diagnostics.beta_nodes_visited, 0);
        assert_eq!(diagnostics.beta_closed_subtrees_reused, 0);
    }

    #[test]
    fn multiargument_identity_requires_each_exact_position_and_type() {
        let mut expressions = ExprArena::new();
        let first = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let second = expressions.intern_argument(1, ResolvedValueType::Int).unwrap();
        let root = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                &[first, second],
            )
            .unwrap();
        let mut programs = ProgramArena::new();
        programs.diagnostic_counters_enabled = true;
        let program = programs
            .finalize(
                &mut expressions,
                ProgramSignature {
                    inputs: Box::new([
                        ProgramInput {
                            value_type: ResolvedValueType::Int,
                            trusted_index_range: None,
                        },
                        ProgramInput {
                            value_type: ResolvedValueType::Int,
                            trusted_index_range: None,
                        },
                    ]),
                    output: ResolvedValueType::Int,
                },
                root,
            )
            .unwrap();

        assert_eq!(
            programs.beta_reduce(&mut expressions, program, &[first, second]).unwrap(),
            root
        );
        assert_eq!(programs.diagnostic_counters().beta_identity_shortcuts, 1);

        let swapped = programs.beta_reduce(&mut expressions, program, &[second, first]).unwrap();
        assert_ne!(swapped, root);
        assert_eq!(programs.diagnostic_counters().beta_identity_shortcuts, 1);

        let one_input_program = programs
            .finalize(
                &mut expressions,
                ProgramSignature {
                    inputs: Box::new([ProgramInput {
                        value_type: ResolvedValueType::Int,
                        trusted_index_range: None,
                    }]),
                    output: ResolvedValueType::Int,
                },
                first,
            )
            .unwrap();
        assert_eq!(
            programs.beta_reduce(&mut expressions, one_input_program, &[second]).unwrap(),
            second
        );
        let constant = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(7)), Box::new([]))
            .unwrap();
        assert_eq!(
            programs.beta_reduce(&mut expressions, one_input_program, &[constant]).unwrap(),
            constant
        );
        assert_eq!(programs.diagnostic_counters().beta_identity_shortcuts, 1);
    }

    #[test]
    fn zero_argument_identity_shortcut_validates_local_root_and_rejects_foreign_arena() {
        let mut expressions = ExprArena::new();
        let root = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(7)), Box::new([]))
            .unwrap();
        let mut programs = ProgramArena::new();
        programs.diagnostic_counters_enabled = true;
        let program = programs
            .finalize(
                &mut expressions,
                ProgramSignature { inputs: Box::new([]), output: ResolvedValueType::Int },
                root,
            )
            .unwrap();

        assert_eq!(programs.beta_reduce(&mut expressions, program, &[]).unwrap(), root);
        let counters_before = programs.diagnostic_counters();
        assert_eq!(counters_before.beta_identity_shortcuts, 1);
        assert_eq!(counters_before.beta_nodes_visited, 0);

        let mut foreign_expressions = ExprArena::new();
        let foreign_argument = foreign_expressions
            .intern(ValueOperator::Constant(TypedConstant::int(0)), Box::new([]))
            .unwrap();
        assert!(matches!(
            programs.beta_reduce(&mut foreign_expressions, program, &[foreign_argument]),
            Err(ArenaError::InvalidArity { expected: 0, actual: 1, .. })
        ));
        assert!(matches!(
            programs.beta_reduce(&mut foreign_expressions, program, &[]),
            Err(ArenaError::ForeignExpression { .. })
        ));
        let counters_after = programs.diagnostic_counters();
        assert_eq!(counters_after.beta_identity_shortcuts, counters_before.beta_identity_shortcuts);
        assert_eq!(counters_after.beta_nodes_visited, counters_before.beta_nodes_visited);
    }

    #[test]
    fn alpha_equivalent_programs_intern_by_argument_position() {
        let mut expressions = ExprArena::new();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let one = expressions
            .intern(
                ValueOperator::Constant(super::super::arena::TypedConstant::int(1)),
                Box::new([]),
            )
            .unwrap();
        let root = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                &[argument, one],
            )
            .unwrap();
        let signature = family_signature(domain(), ResolvedValueType::Int);
        let mut programs = ProgramArena::new();
        let first = programs.finalize(&mut expressions, signature.clone(), root).unwrap();
        let second = programs.finalize(&mut expressions, signature, root).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn successful_root_facts_cache_preserves_interned_and_fresh_program_ids() {
        let mut expressions = ExprArena::new();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let mut programs = ProgramArena::new();
        let unbounded = ProgramSignature {
            inputs: Box::new([ProgramInput {
                value_type: ResolvedValueType::Int,
                trusted_index_range: None,
            }]),
            output: ResolvedValueType::Int,
        };
        let bounded = family_signature(domain(), ResolvedValueType::Int);

        let first = programs.finalize(&mut expressions, unbounded.clone(), argument).unwrap();
        let repeated = programs.finalize(&mut expressions, unbounded, argument).unwrap();
        let different_signature =
            programs.finalize(&mut expressions, bounded.clone(), argument).unwrap();
        let fresh_first =
            programs.finalize_fresh(&mut expressions, bounded.clone(), argument).unwrap();
        let fresh_second = programs.finalize_fresh(&mut expressions, bounded, argument).unwrap();

        assert_eq!(first, repeated);
        assert_ne!(first, different_signature);
        assert_ne!(fresh_first, fresh_second);
        assert_eq!(programs.root_validation_facts_len(), 1);
    }

    #[test]
    fn root_validation_facts_reuse_earliest_root_past_legacy_fifo_bound() {
        const ROOTS: u64 = 65_537;
        let mut expressions = ExprArena::new();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let mut programs = ProgramArena::new();
        programs.enable_diagnostic_counters();
        let signature = ProgramSignature {
            inputs: Box::new([ProgramInput {
                value_type: ResolvedValueType::Int,
                trusted_index_range: None,
            }]),
            output: ResolvedValueType::Int,
        };
        let mut first = None;
        for value in 0..ROOTS {
            let constant = expressions
                .intern(ValueOperator::Constant(TypedConstant::int(value)), Box::new([]))
                .unwrap();
            let root = expressions
                .intern_slice(
                    ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                    &[argument, constant],
                )
                .unwrap();
            let program = programs.finalize(&mut expressions, signature.clone(), root).unwrap();
            if value == 0 {
                first = Some((root, program));
            }
        }
        let (first_root, first_program) = first.unwrap();
        let before = programs.diagnostic_counters();
        let nodes_before = expressions.node_count();
        let repeated = programs.finalize(&mut expressions, signature.clone(), first_root).unwrap();
        let after = programs.diagnostic_counters();
        assert_eq!(repeated, first_program);
        assert_eq!(expressions.node_count(), nodes_before);
        assert_eq!(after.root_validation_facts_hits, before.root_validation_facts_hits + 1);
        assert_eq!(after.root_validation_facts_misses, before.root_validation_facts_misses);
        assert_eq!(after.root_validation_facts_entries, ROOTS);
        assert_eq!(programs.root_validation_facts_len() as u64, ROOTS);

        assert_eq!(
            programs.finalize(
                &mut expressions,
                ProgramSignature {
                    inputs: signature.inputs.clone(),
                    output: ResolvedValueType::Bool,
                },
                first_root,
            ),
            Err(ArenaError::ProgramOutputMismatch)
        );
        assert!(matches!(
            programs.finalize(
                &mut expressions,
                ProgramSignature {
                    inputs: Box::new([ProgramInput {
                        value_type: ResolvedValueType::Bool,
                        trusted_index_range: None,
                    }]),
                    output: ResolvedValueType::Int,
                },
                first_root,
            ),
            Err(ArenaError::TypeMismatch { position: 0, .. })
        ));
        assert_eq!(programs.root_validation_facts_len() as u64, ROOTS);
        assert_eq!(expressions.node_count(), nodes_before);

        let original_facts = programs.root_validation_facts.get(&first_root).unwrap().clone();
        let conflicting_facts = BTreeSet::from([(0, ResolvedValueType::Bool)]);
        assert_eq!(
            programs.record_root_validation_facts(first_root, conflicting_facts),
            Err(ArenaError::ProgramSignatureMismatch)
        );
        assert_eq!(programs.root_validation_facts.get(&first_root), Some(&original_facts));
        assert_eq!(programs.root_validation_facts_len() as u64, ROOTS);
    }

    #[test]
    fn root_validation_reuses_cached_subroots_with_full_collector_facts() {
        const LAYERS: u32 = 32;
        let mut expressions = ExprArena::new();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let one = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(1)), Box::new([]))
            .unwrap();
        let signature = ProgramSignature {
            inputs: Box::new([ProgramInput {
                value_type: ResolvedValueType::Int,
                trusted_index_range: None,
            }]),
            output: ResolvedValueType::Int,
        };
        let mut programs = ProgramArena::new();
        programs.enable_diagnostic_counters();
        programs.finalize(&mut expressions, signature.clone(), one).unwrap();

        let mut root = argument;
        for _ in 0..LAYERS {
            root = expressions
                .intern_slice(
                    ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                    &[root, one],
                )
                .unwrap();
            programs.finalize(&mut expressions, signature.clone(), root).unwrap();
        }

        let optimized_counters = programs.diagnostic_counters();
        let optimized_facts = programs.root_validation_facts.get(&root).cloned().unwrap();
        assert_eq!(optimized_facts, BTreeSet::from([(0, ResolvedValueType::Int)]));
        assert_eq!(optimized_counters.root_validation_nodes_visited, u64::from(LAYERS) + 2);
        assert_eq!(
            optimized_counters.root_validation_cached_subroot_skips,
            u64::from(2 * LAYERS - 1)
        );

        let full_programs = ProgramArena::new();
        let full_facts = full_programs.collect_root_validation_facts(&expressions, root).unwrap();
        assert_eq!(full_facts, optimized_facts);
    }

    #[test]
    fn root_validation_cached_sibling_does_not_hide_foreign_call_or_publish() {
        let mut expressions = ExprArena::new();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let one = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(1)), Box::new([]))
            .unwrap();
        let signature = ProgramSignature {
            inputs: Box::new([ProgramInput {
                value_type: ResolvedValueType::Int,
                trusted_index_range: None,
            }]),
            output: ResolvedValueType::Int,
        };
        let mut programs = ProgramArena::new();
        programs.finalize(&mut expressions, signature.clone(), argument).unwrap();

        let mut foreign_programs = ProgramArena::new();
        let foreign = foreign_programs
            .finalize(
                &mut expressions,
                ProgramSignature { inputs: Box::new([]), output: ResolvedValueType::Int },
                one,
            )
            .unwrap();
        let foreign_call = foreign_programs.call(&mut expressions, foreign, &[]).unwrap();
        let root = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                &[foreign_call, argument],
            )
            .unwrap();
        let entries_before = programs.root_validation_facts_len();

        assert!(matches!(
            programs.finalize(&mut expressions, signature, root),
            Err(ArenaError::ForeignProgram { .. })
        ));
        assert_eq!(programs.root_validation_facts_len(), entries_before);
    }

    #[test]
    fn failed_finalization_keeps_error_priority_and_does_not_cache_root_facts() {
        let mut expressions = ExprArena::new();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let mut programs = ProgramArena::new();
        assert_eq!(
            programs.finalize(
                &mut expressions,
                ProgramSignature { inputs: Box::new([]), output: ResolvedValueType::Bool },
                argument,
            ),
            Err(ArenaError::ProgramOutputMismatch)
        );
        assert_eq!(programs.root_validation_facts_len(), 0);
        assert_eq!(
            programs.finalize(
                &mut expressions,
                ProgramSignature { inputs: Box::new([]), output: ResolvedValueType::Int },
                argument,
            ),
            Err(ArenaError::FreeArgumentEscapes { position: 0 })
        );
        assert_eq!(programs.root_validation_facts_len(), 0);
        programs
            .finalize(
                &mut expressions,
                ProgramSignature {
                    inputs: Box::new([ProgramInput {
                        value_type: ResolvedValueType::Int,
                        trusted_index_range: None,
                    }]),
                    output: ResolvedValueType::Int,
                },
                argument,
            )
            .unwrap();
        assert_eq!(programs.root_validation_facts_len(), 1);
    }

    #[test]
    fn root_facts_cache_is_expression_typed_and_rejects_foreign_authority() {
        let mut expressions = ExprArena::new();
        let int_argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let one = expressions
            .intern(
                ValueOperator::Constant(super::super::arena::TypedConstant::int(1)),
                Box::new([]),
            )
            .unwrap();
        let shared = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                &[int_argument, one],
            )
            .unwrap();
        let int_root = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                &[shared, shared],
            )
            .unwrap();
        let bool_root = expressions.intern_argument(0, ResolvedValueType::Bool).unwrap();
        let mut programs = ProgramArena::new();
        programs
            .finalize(
                &mut expressions,
                ProgramSignature {
                    inputs: Box::new([ProgramInput {
                        value_type: ResolvedValueType::Int,
                        trusted_index_range: None,
                    }]),
                    output: ResolvedValueType::Int,
                },
                int_root,
            )
            .unwrap();
        programs
            .finalize(
                &mut expressions,
                ProgramSignature {
                    inputs: Box::new([ProgramInput {
                        value_type: ResolvedValueType::Bool,
                        trusted_index_range: None,
                    }]),
                    output: ResolvedValueType::Bool,
                },
                bool_root,
            )
            .unwrap();
        assert_eq!(programs.root_validation_facts_len(), 2);

        let mut foreign_expressions = ExprArena::new();
        let foreign_root = foreign_expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        assert!(matches!(
            programs.finalize(
                &mut expressions,
                family_signature(domain(), ResolvedValueType::Int),
                foreign_root,
            ),
            Err(ArenaError::ForeignExpression { .. })
        ));
        assert_eq!(programs.root_validation_facts_len(), 2);

        let mut foreign_programs = ProgramArena::new();
        let foreign_program = foreign_programs
            .finalize(
                &mut expressions,
                ProgramSignature { inputs: Box::new([]), output: ResolvedValueType::Int },
                one,
            )
            .unwrap();
        let foreign_call = foreign_programs.call(&mut expressions, foreign_program, &[]).unwrap();
        assert!(matches!(
            programs.finalize(
                &mut expressions,
                ProgramSignature { inputs: Box::new([]), output: ResolvedValueType::Int },
                foreign_call,
            ),
            Err(ArenaError::ForeignProgram { .. })
        ));
        assert_eq!(programs.root_validation_facts_len(), 2);
    }

    #[test]
    fn source_and_explicit_calls_are_compact_and_typed() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let mut facts = FactStore::new(&expressions);
        let source_family =
            programs.source_family(&mut expressions, source("source"), None).unwrap();
        let index = expressions
            .intern(
                ValueOperator::Constant(super::super::arena::TypedConstant::int(1)),
                Box::new([]),
            )
            .unwrap();
        facts
            .declare_trusted_index_range(
                &expressions,
                index,
                TrustedIndexRange { minimum: 1, maximum_exclusive: 2 },
            )
            .unwrap();
        facts.finalize_ranges();
        let call = programs.call_family(&mut expressions, &facts, source_family, index).unwrap();
        assert!(matches!(
            expressions.node(call).unwrap().operator,
            ValueOperator::ProgramCall { .. }
        ));
        let values = (0..4)
            .map(|value| {
                expressions
                    .intern(
                        ValueOperator::Constant(super::super::arena::TypedConstant::int(value)),
                        Box::new([]),
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let explicit =
            programs.explicit_family(&mut expressions, &facts, domain(), values).unwrap();
        let explicit_call =
            programs.call_family(&mut expressions, &facts, explicit, index).unwrap();
        assert!(matches!(
            expressions.node(explicit_call).unwrap().operator,
            ValueOperator::ProgramCall { .. }
        ));
    }

    #[test]
    fn explicit_family_joins_typed_matrix_facts_without_cross_product() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let mut facts = FactStore::new(&expressions);
        let values =
            (0..3).map(|event| matrix_expression(&mut expressions, event)).collect::<Vec<_>>();
        for (value, bound) in values.iter().copied().zip([2_u64, 5, 1]) {
            facts
                .insert(
                    &expressions,
                    value,
                    ValueFacts::Matrix(matrix_facts(
                        Some(bound),
                        true,
                        MatrixLayout::row_major(1, 2),
                    )),
                )
                .unwrap();
        }
        let family = programs
            .explicit_family(
                &mut expressions,
                &facts,
                FamilyDomain::new(0, values.len() as u64).unwrap(),
                values.clone().into_boxed_slice(),
            )
            .unwrap();
        let summary = programs.family_matrix_facts(family).unwrap().unwrap();
        assert!(summary.metadata.is_constant_polynomial);
        assert_eq!(
            summary.coefficient_bound,
            NumericContract::Known(CoefficientBound::finite(5_u64))
        );
        assert_eq!(summary.polynomial, NumericContract::Known(PolynomialFacts::new(2, 2).unwrap()));
        assert_eq!(
            expressions.node(programs.family_body(family).unwrap()).unwrap().inputs.len(),
            4
        );

        // A non-constant branch removes central eligibility, while an unknown bound removes
        // only the numeric contract. Neither case causes branch or selector expansion.
        let nonconstant = matrix_expression(&mut expressions, 10);
        facts
            .insert(
                &expressions,
                nonconstant,
                ValueFacts::Matrix(matrix_facts(Some(5), false, MatrixLayout::row_major(1, 2))),
            )
            .unwrap();
        let family = programs
            .explicit_family(
                &mut expressions,
                &facts,
                FamilyDomain::new(0, 2).unwrap(),
                vec![values[0], nonconstant].into_boxed_slice(),
            )
            .unwrap();
        let summary = programs.family_matrix_facts(family).unwrap().unwrap();
        assert!(!summary.metadata.is_constant_polynomial);

        let unknown = matrix_expression(&mut expressions, 11);
        facts
            .insert(
                &expressions,
                unknown,
                ValueFacts::Matrix(matrix_facts(None, true, MatrixLayout::row_major(1, 2))),
            )
            .unwrap();
        let family = programs
            .explicit_family(
                &mut expressions,
                &facts,
                FamilyDomain::new(0, 2).unwrap(),
                vec![values[0], unknown].into_boxed_slice(),
            )
            .unwrap();
        assert!(
            programs.family_matrix_facts(family).unwrap().unwrap().coefficient_bound.is_missing()
        );

        let mismatched = matrix_expression(&mut expressions, 12);
        facts
            .insert(
                &expressions,
                mismatched,
                ValueFacts::Matrix(matrix_facts(Some(1), true, MatrixLayout::row_major(2, 1))),
            )
            .unwrap();
        let layout_mismatch = programs
            .explicit_family(
                &mut expressions,
                &facts,
                FamilyDomain::new(0, 2).unwrap(),
                vec![values[0], mismatched].into_boxed_slice(),
            )
            .unwrap();
        assert!(programs.family_matrix_facts(layout_mismatch).unwrap().is_none());

        let large = matrix_expression(&mut expressions, 13);
        let mut large_facts = matrix_facts(Some(1), true, MatrixLayout::row_major(1, 2));
        large_facts.coefficient_bound = NumericContract::Known(CoefficientBound::Large);
        facts.insert(&expressions, large, ValueFacts::Matrix(large_facts)).unwrap();
        let large_family = programs
            .explicit_family(
                &mut expressions,
                &facts,
                FamilyDomain::new(0, 2).unwrap(),
                vec![values[0], large].into_boxed_slice(),
            )
            .unwrap();
        assert_eq!(
            programs.family_matrix_facts(large_family).unwrap().unwrap().coefficient_bound,
            NumericContract::Known(CoefficientBound::Large)
        );
        let opaque = programs.source_family(&mut expressions, source("opaque"), None).unwrap();
        assert!(programs.family_matrix_facts(opaque).unwrap().is_none());
    }

    #[test]
    fn explicit_family_derives_authoritative_fact_views_without_replacing_compact_values() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let mut facts = FactStore::new(&expressions);
        let value = matrix_expression(&mut expressions, 77);
        let expected_facts = matrix_facts(Some(5), true, MatrixLayout::row_major(1, 2));
        facts.insert(&expressions, value, ValueFacts::Matrix(expected_facts.clone())).unwrap();
        let unrelated = matrix_expression(&mut expressions, 78);
        facts
            .insert(
                &expressions,
                unrelated,
                ValueFacts::Matrix(matrix_facts(Some(99), true, MatrixLayout::row_major(1, 2))),
            )
            .unwrap();
        let domain = FamilyDomain::new(0, 1).unwrap();
        let generated =
            programs.generated_family_from_body(&mut expressions, domain, value).unwrap();
        let index = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(0)), Box::new([]))
            .unwrap();
        let compact = programs
            .call_family_in_range_deferred_generated(
                &mut expressions,
                generated,
                index,
                TrustedIndexRange::new(0, 1).unwrap(),
            )
            .unwrap();
        assert_ne!(compact, value);
        let materialized =
            programs.materialize_reducible_generated_calls(&mut expressions, compact).unwrap();
        assert_eq!(materialized, value);

        let packed = programs
            .explicit_family(&mut expressions, &facts, domain, Box::new([compact]))
            .unwrap();
        assert!(programs.family_body(packed).is_ok_and(|body| {
            expressions.node(body).is_ok_and(|node| node.inputs.contains(&compact))
        }));
        assert_eq!(programs.family_matrix_facts(packed).unwrap(), Some(&expected_facts));
    }

    #[test]
    fn binder_open_explicit_call_reads_producer_summary_without_fact_insertion() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let mut facts = FactStore::new(&expressions);
        let value = matrix_expression(&mut expressions, 21);
        let mut value_facts = matrix_facts(Some(3), true, MatrixLayout::row_major(1, 2));
        value_facts.coefficient_bound = NumericContract::Known(CoefficientBound::Large);
        facts.insert(&expressions, value, ValueFacts::Matrix(value_facts)).unwrap();
        let family = programs
            .explicit_family(
                &mut expressions,
                &facts,
                FamilyDomain::new(0, 1).unwrap(),
                vec![value].into_boxed_slice(),
            )
            .unwrap();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let call = programs
            .call_family_in_range(
                &mut expressions,
                family,
                argument,
                TrustedIndexRange::new(0, 1).unwrap(),
            )
            .unwrap();
        let summary =
            programs.program_call_family_matrix_facts(&expressions, call).unwrap().unwrap();
        assert!(summary.metadata.is_constant_polynomial);
        assert_eq!(summary.coefficient_bound, NumericContract::Known(CoefficientBound::Large));

        let opaque_body = expressions
            .intern(
                ValueOperator::OpaqueFamilyElement {
                    source: source_with_domain("opaque-summary", FamilyDomain::new(0, 1).unwrap()),
                },
                Box::new([argument]),
            )
            .unwrap();
        let opaque = programs
            .opaque_generated_family_from_body(
                &mut expressions,
                FamilyDomain::new(0, 1).unwrap(),
                opaque_body,
            )
            .unwrap();
        let opaque_call = programs
            .call_family_in_range(
                &mut expressions,
                opaque,
                argument,
                TrustedIndexRange::new(0, 1).unwrap(),
            )
            .unwrap();
        assert!(
            programs.program_call_family_matrix_facts(&expressions, opaque_call).unwrap().is_none()
        );
    }

    #[test]
    fn selected_explicit_family_summary_stays_compact_for_1024_branches() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let mut facts = FactStore::new(&expressions);
        let value = matrix_expression(&mut expressions, 22);
        facts
            .insert(
                &expressions,
                value,
                ValueFacts::Matrix(matrix_facts(Some(4), true, MatrixLayout::row_major(1, 2))),
            )
            .unwrap();
        let domain = FamilyDomain::new(0, 1024).unwrap();
        let family = programs
            .explicit_family(&mut expressions, &facts, domain, vec![value; 1024].into_boxed_slice())
            .unwrap();
        let selector = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(0)), Box::new([]))
            .unwrap();
        facts
            .declare_trusted_index_range(
                &expressions,
                selector,
                TrustedIndexRange::new(0, 1024).unwrap(),
            )
            .unwrap();
        facts.finalize_ranges();
        let closed_selector = expressions.close(selector).unwrap();
        let selected = programs
            .select(
                &mut expressions,
                &facts,
                SelectionSelector::Closed(closed_selector),
                &[family; 1024],
            )
            .unwrap();
        let body = programs.family_body(selected).unwrap();
        assert_eq!(expressions.node(body).unwrap().inputs.len(), 1025);
        let index = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(1)), Box::new([]))
            .unwrap();
        let call = programs
            .call_family_in_range(
                &mut expressions,
                selected,
                index,
                TrustedIndexRange::new(0, 1024).unwrap(),
            )
            .unwrap();
        assert!(programs.program_call_family_matrix_facts(&expressions, call).unwrap().is_some());
    }

    #[test]
    fn generated_pointwise_reindex_gather_zip_and_selection_do_not_walk_domain() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let mut facts = FactStore::new(&expressions);
        let left = programs.source_family(&mut expressions, source("left"), None).unwrap();
        let right = programs.source_family(&mut expressions, source("right"), None).unwrap();
        let _sum = programs
            .pointwise_binary(
                &mut expressions,
                left,
                right,
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
            )
            .unwrap();
        expressions
            .register_index_definition(super::super::arena::IndexFunctionDefinition {
                id: IndexFunctionDefinitionId(7),
                arity: 1,
                output_type: ResolvedValueType::Int,
            })
            .unwrap();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let mapped = expressions
            .intern(
                ValueOperator::IndexMap {
                    definition: IndexFunctionDefinitionId(7),
                    parameters: Box::new([]),
                },
                Box::new([argument]),
            )
            .unwrap();
        let map_program = index_map_program(&mut programs, &mut expressions, domain(), mapped);
        let selector = expressions
            .intern(
                ValueOperator::Constant(super::super::arena::TypedConstant::int(0)),
                Box::new([]),
            )
            .unwrap();
        let range = TrustedIndexRange { minimum: 0, maximum_exclusive: 4 };
        facts.declare_scoped_trusted_index_range(&expressions, map_program, mapped, range).unwrap();
        facts.declare_trusted_index_range(&expressions, selector, range).unwrap();
        assert!(matches!(
            programs.reindex(&mut expressions, &facts, left, map_program),
            Err(ArenaError::IndexRangeRequired { id }) if id == mapped
        ));
        facts.finalize_ranges();
        let reindexed = programs.reindex(&mut expressions, &facts, left, map_program).unwrap();
        let gathered = programs.gather(&mut expressions, &facts, right, map_program).unwrap();
        let _zipped = programs
            .zip(
                &mut expressions,
                &[reindexed, gathered],
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
            )
            .unwrap();
        let selector = expressions.close(selector).unwrap();
        let selected = programs
            .select(
                &mut expressions,
                &facts,
                SelectionSelector::Closed(selector),
                &[left, right, left, right],
            )
            .unwrap();
        assert_eq!(programs.family_domain(selected).unwrap(), domain());
        assert!(expressions.node(programs.family_body(selected).unwrap()).is_ok());
    }

    #[test]
    fn gather_allows_a_compact_cross_domain_map_without_case_expansion() {
        let source_domain = FamilyDomain::new(0, 8).unwrap();
        let result_domain = FamilyDomain::new(0, 4).unwrap();
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let source_family = programs
            .source_family(
                &mut expressions,
                source_with_domain("cross-domain-source", source_domain),
                None,
            )
            .unwrap();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let mapped = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                &[argument, argument],
            )
            .unwrap();
        let map_program = index_map_program(&mut programs, &mut expressions, result_domain, mapped);
        let mut facts = FactStore::new(&expressions);
        facts
            .declare_scoped_trusted_index_range(
                &expressions,
                map_program,
                mapped,
                TrustedIndexRange { minimum: 0, maximum_exclusive: 8 },
            )
            .unwrap();
        facts.finalize_ranges();

        let gathered =
            programs.gather(&mut expressions, &facts, source_family, map_program).unwrap();
        assert_eq!(programs.family_domain(gathered).unwrap(), result_domain);
        let body = programs.family_body(gathered).unwrap();
        assert!(matches!(
            expressions.node(body).unwrap().operator,
            ValueOperator::ProgramCall { .. }
        ));
        assert!(!matches!(
            expressions.node(body).unwrap().operator,
            ValueOperator::ExplicitElement { .. }
        ));
    }

    #[test]
    fn zip_offset_is_constant_work_for_a_million_element_domain() {
        let million = FamilyDomain::new(0, 1_000_000).unwrap();
        let shifted = FamilyDomain::new(0, 1_000_001).unwrap();
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let direct = programs
            .source_family(&mut expressions, source_with_domain("million-direct", million), None)
            .unwrap();
        let offset = programs
            .source_family(&mut expressions, source_with_domain("million-offset", shifted), None)
            .unwrap();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let one = expressions
            .intern(
                ValueOperator::Constant(super::super::arena::TypedConstant::int(1)),
                Box::new([]),
            )
            .unwrap();
        let mapped = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                &[argument, one],
            )
            .unwrap();
        let map_program = index_map_program(&mut programs, &mut expressions, million, mapped);
        let mut facts = FactStore::new(&expressions);
        facts
            .declare_scoped_trusted_index_range(
                &expressions,
                map_program,
                mapped,
                TrustedIndexRange { minimum: 1, maximum_exclusive: 1_000_001 },
            )
            .unwrap();
        facts.finalize_ranges();

        let nodes_before = expressions.node_count();
        let programs_before = programs.len();
        let zipped = programs
            .zip_offset(
                &mut expressions,
                &facts,
                direct,
                offset,
                map_program,
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
            )
            .unwrap();
        assert_eq!(programs.family_domain(zipped).unwrap(), million);
        assert!(expressions.node_count() - nodes_before <= 4);
        assert_eq!(programs.len() - programs_before, 1);
    }

    #[test]
    fn zip_offset_preserves_map_identity_and_reuses_the_same_descriptor() {
        let direct_domain = FamilyDomain::new(0, 4).unwrap();
        let offset_domain = FamilyDomain::new(0, 6).unwrap();
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let direct = programs
            .source_family(
                &mut expressions,
                source_with_domain("identity-direct", direct_domain),
                None,
            )
            .unwrap();
        let offset = programs
            .source_family(
                &mut expressions,
                source_with_domain("identity-offset", offset_domain),
                None,
            )
            .unwrap();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let one = expressions
            .intern(
                ValueOperator::Constant(super::super::arena::TypedConstant::int(1)),
                Box::new([]),
            )
            .unwrap();
        let two = expressions
            .intern(
                ValueOperator::Constant(super::super::arena::TypedConstant::int(2)),
                Box::new([]),
            )
            .unwrap();
        let mapped_one = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                &[argument, one],
            )
            .unwrap();
        let mapped_two = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                &[argument, two],
            )
            .unwrap();
        let map_one_program =
            index_map_program(&mut programs, &mut expressions, direct_domain, mapped_one);
        let map_two_program =
            index_map_program(&mut programs, &mut expressions, direct_domain, mapped_two);
        let mut facts = FactStore::new(&expressions);
        facts
            .declare_scoped_trusted_index_range(
                &expressions,
                map_one_program,
                mapped_one,
                TrustedIndexRange { minimum: 1, maximum_exclusive: 5 },
            )
            .unwrap();
        facts
            .declare_scoped_trusted_index_range(
                &expressions,
                map_two_program,
                mapped_two,
                TrustedIndexRange { minimum: 2, maximum_exclusive: 6 },
            )
            .unwrap();
        facts.finalize_ranges();
        let operation = ValueOperator::Scalar(super::super::arena::ScalarOperation::Add);
        let first = programs
            .zip_offset(
                &mut expressions,
                &facts,
                direct,
                offset,
                map_one_program,
                operation.clone(),
            )
            .unwrap();
        let reused = programs
            .zip_offset(
                &mut expressions,
                &facts,
                direct,
                offset,
                map_one_program,
                operation.clone(),
            )
            .unwrap();
        let different = programs
            .zip_offset(&mut expressions, &facts, direct, offset, map_two_program, operation)
            .unwrap();
        assert_eq!(first, reused);
        assert_ne!(first, different);
        assert_ne!(programs.family_body(first).unwrap(), programs.family_body(different).unwrap());
    }

    #[test]
    fn deferred_nested_offset_extract_materializes_to_eager_structure_before_finalization() {
        let inner_domain = FamilyDomain::new(0, 5).unwrap();
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let matrix = matrix_expression(&mut expressions, 700);
        let matrix_family =
            programs.generated_family_from_body(&mut expressions, inner_domain, matrix).unwrap();
        let inner_argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let inner_range = TrustedIndexRange { minimum: 0, maximum_exclusive: 5 };

        let eager_matrix = programs
            .call_family_in_range(&mut expressions, matrix_family, inner_argument, inner_range)
            .unwrap();
        let eager_extract = expressions.intern_extract_coefficient(eager_matrix, 0, None).unwrap();
        let deferred_matrix = programs
            .call_family_in_range_deferred_generated(
                &mut expressions,
                matrix_family,
                inner_argument,
                inner_range,
            )
            .unwrap();
        let deferred_extract =
            expressions.intern_extract_coefficient(deferred_matrix, 0, None).unwrap();
        assert_eq!(
            programs.ensure_no_reducible_generated_calls(&expressions, deferred_extract),
            Err(ArenaError::ProgramSignatureMismatch)
        );
        let materialized_extract = programs
            .materialize_reducible_generated_calls(&mut expressions, deferred_extract)
            .unwrap();
        assert_eq!(materialized_extract, eager_extract);
        let coefficient_family = programs
            .generated_family_from_body(&mut expressions, inner_domain, materialized_extract)
            .unwrap();

        let outer_argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let one = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(1)), Box::new([]))
            .unwrap();
        let mapped = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                &[outer_argument, one],
            )
            .unwrap();
        let mapped_range = TrustedIndexRange { minimum: 1, maximum_exclusive: 5 };
        let eager = programs
            .call_family_in_range(&mut expressions, coefficient_family, mapped, mapped_range)
            .unwrap();
        let deferred = programs
            .call_family_in_range_deferred_generated(
                &mut expressions,
                coefficient_family,
                mapped,
                mapped_range,
            )
            .unwrap();
        assert!(matches!(
            expressions.node(deferred).unwrap().operator,
            ValueOperator::ProgramCall { .. }
        ));
        let materialized =
            programs.materialize_reducible_generated_calls(&mut expressions, deferred).unwrap();
        assert_eq!(materialized, eager);
        programs.ensure_no_reducible_generated_calls(&expressions, materialized).unwrap();
    }

    #[test]
    fn deferred_family_call_preserves_validation_errors_and_opaque_identity() {
        let domain = FamilyDomain::new(0, 4).unwrap();
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        programs.enable_diagnostic_counters();
        let matrix = matrix_expression(&mut expressions, 701);
        let reducible =
            programs.generated_family_from_body(&mut expressions, domain, matrix).unwrap();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let range = TrustedIndexRange { minimum: 0, maximum_exclusive: 4 };
        let boolean = expressions.intern_argument(0, ResolvedValueType::Bool).unwrap();
        assert_eq!(
            programs.call_family_in_range_deferred_generated(
                &mut expressions,
                reducible,
                boolean,
                range,
            ),
            programs.call_family_in_range(&mut expressions, reducible, boolean, range)
        );
        let invalid_range = TrustedIndexRange { minimum: 3, maximum_exclusive: 5 };
        assert_eq!(
            programs.call_family_in_range_deferred_generated(
                &mut expressions,
                reducible,
                argument,
                invalid_range,
            ),
            programs.call_family_in_range(&mut expressions, reducible, argument, invalid_range,)
        );

        let opaque =
            programs.opaque_generated_family_from_body(&mut expressions, domain, matrix).unwrap();
        let opaque_call = programs
            .call_family_in_range_deferred_generated(&mut expressions, opaque, argument, range)
            .unwrap();
        expressions.record_program_call_reduction(opaque_call, matrix).unwrap();
        let early_source_hits_before =
            programs.diagnostic_counters().beta_program_call_early_source_hits;
        assert_eq!(
            programs.materialize_reducible_generated_calls(&mut expressions, opaque_call).unwrap(),
            opaque_call
        );
        assert_eq!(
            programs.diagnostic_counters().beta_program_call_early_source_hits,
            early_source_hits_before
        );
        assert!(matches!(
            expressions.node(opaque_call).unwrap().operator,
            ValueOperator::ProgramCall { program } if program == opaque.program()
        ));
        programs.ensure_no_reducible_generated_calls(&expressions, opaque_call).unwrap();
    }

    #[test]
    fn materialization_reuses_source_reduction_before_visiting_inputs() {
        let domain = FamilyDomain::new(0, 8).unwrap();
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        programs.enable_diagnostic_counters();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let one = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(1)), Box::new([]))
            .unwrap();
        let shifted = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                &[argument, one],
            )
            .unwrap();
        let inner = programs.generated_family_from_body(&mut expressions, domain, shifted).unwrap();
        let inner_call = programs
            .call_family_in_range_deferred_generated(
                &mut expressions,
                inner,
                argument,
                TrustedIndexRange { minimum: 0, maximum_exclusive: 8 },
            )
            .unwrap();
        let two = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(2)), Box::new([]))
            .unwrap();
        let outer_body = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                &[argument, two],
            )
            .unwrap();
        let outer =
            programs.generated_family_from_body(&mut expressions, domain, outer_body).unwrap();
        let outer_call = programs
            .call_family_in_range_deferred_generated(
                &mut expressions,
                outer,
                inner_call,
                TrustedIndexRange { minimum: 0, maximum_exclusive: 8 },
            )
            .unwrap();

        let first = programs
            .materialize_reducible_generated_calls_with_reason(
                &mut expressions,
                outer_call,
                BetaReason::ResidualRoot,
            )
            .unwrap();
        let node_count = expressions.node_count();
        let before = programs.diagnostic_counters();
        let second =
            programs.materialize_reducible_generated_calls(&mut expressions, outer_call).unwrap();
        let after = programs.diagnostic_counters();

        assert_eq!(second, first);
        assert_eq!(expressions.node_count(), node_count);
        assert_eq!(after.beta_nodes_visited, before.beta_nodes_visited);
        assert_eq!(after.beta_reason_misses, before.beta_reason_misses);
        assert_eq!(after.beta_reason_visits, before.beta_reason_visits);
        assert_eq!(
            after.beta_program_call_early_source_hits,
            before.beta_program_call_early_source_hits + 1
        );
        assert!(
            after.beta_program_call_early_source_input_skips >
                before.beta_program_call_early_source_input_skips
        );
        assert_eq!(
            after.beta_reason_misses.iter().sum::<u64>(),
            after.beta_program_call_sidecar_misses
        );
        assert_eq!(after.beta_reason_visits.iter().sum::<u64>(), after.beta_nodes_visited);
        assert!(after.beta_reason_misses[BetaReason::ResidualRoot as usize] > 0);
        assert!(after.beta_reason_visits[BetaReason::ResidualRoot as usize] > 0);
        assert!(after.beta_reason_expr_allocations[BetaReason::ResidualRoot as usize] > 0);
        programs.ensure_no_reducible_generated_calls(&expressions, second).unwrap();
    }

    #[test]
    fn materialization_waits_for_late_reducible_family_classification() {
        let domain = FamilyDomain::new(0, 4).unwrap();
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        programs.enable_diagnostic_counters();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let one = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(1)), Box::new([]))
            .unwrap();
        let body = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                &[argument, one],
            )
            .unwrap();
        let program = programs
            .finalize(&mut expressions, family_signature(domain, ResolvedValueType::Int), body)
            .unwrap();
        let call = programs.call(&mut expressions, program, &[argument]).unwrap();
        expressions.record_program_call_reduction(call, body).unwrap();
        let before = programs.diagnostic_counters();
        assert_eq!(
            programs.materialize_reducible_generated_calls(&mut expressions, call).unwrap(),
            call
        );
        let after_opaque = programs.diagnostic_counters();
        assert_eq!(
            after_opaque.beta_program_call_early_source_hits,
            before.beta_program_call_early_source_hits
        );

        let _family = programs.generated_family_from_body(&mut expressions, domain, body).unwrap();
        let materialized =
            programs.materialize_reducible_generated_calls(&mut expressions, call).unwrap();
        assert_eq!(materialized, body);
        assert_eq!(
            programs.diagnostic_counters().beta_program_call_early_source_hits,
            after_opaque.beta_program_call_early_source_hits + 1
        );
    }

    #[test]
    fn materialization_rebuilds_opaque_call_index_without_unfolding_opaque_body() {
        let domain = FamilyDomain::new(0, 8).unwrap();
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let one = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(1)), Box::new([]))
            .unwrap();
        let shifted = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                &[argument, one],
            )
            .unwrap();
        let index_family =
            programs.generated_family_from_body(&mut expressions, domain, shifted).unwrap();
        let deferred_index = programs
            .call_family_in_range_deferred_generated(
                &mut expressions,
                index_family,
                argument,
                TrustedIndexRange { minimum: 0, maximum_exclusive: 7 },
            )
            .unwrap();
        let opaque_body = matrix_expression(&mut expressions, 702);
        let opaque = programs
            .opaque_generated_family_from_body(&mut expressions, domain, opaque_body)
            .unwrap();
        let opaque_call = programs
            .call_family_in_range_deferred_generated(
                &mut expressions,
                opaque,
                deferred_index,
                TrustedIndexRange { minimum: 1, maximum_exclusive: 8 },
            )
            .unwrap();

        let materialized =
            programs.materialize_reducible_generated_calls(&mut expressions, opaque_call).unwrap();
        let node = expressions.node(materialized).unwrap();
        assert!(matches!(
            node.operator,
            ValueOperator::ProgramCall { program } if program == opaque.program()
        ));
        assert_eq!(node.inputs.as_ref(), &[shifted]);
        programs.ensure_no_reducible_generated_calls(&expressions, materialized).unwrap();
        assert_eq!(
            programs
                .call_family_in_range(
                    &mut expressions,
                    opaque,
                    deferred_index,
                    TrustedIndexRange { minimum: 1, maximum_exclusive: 8 },
                )
                .unwrap(),
            materialized
        );
    }

    #[test]
    fn layered_deferred_zip_offsets_grow_linearly_and_materialize_to_eager_result() {
        const LAYERS: u64 = 64;
        struct Run {
            construction_nodes: u64,
            construction_visits: u64,
            exposure_nodes: u64,
            exposure_visits: u64,
            fingerprint: String,
        }

        fn fingerprint(
            programs: &ProgramArena,
            expressions: &ExprArena,
            expression: ExprId,
        ) -> String {
            let node = expressions.node(expression).unwrap();
            let operator = match node.operator {
                ValueOperator::ProgramCall { program } => {
                    let family = programs.family_for_program(program).unwrap();
                    let body = programs.family_body(family).unwrap();
                    format!("OpaqueCall({:?})", expressions.node(body).unwrap().operator)
                }
                ref operator => format!("{operator:?}"),
            };
            let inputs = node
                .inputs
                .iter()
                .map(|input| fingerprint(programs, expressions, *input))
                .collect::<Vec<_>>()
                .join(",");
            format!("{operator}({inputs})")
        }

        fn run(compact: bool) -> Run {
            let mut expressions = ExprArena::new();
            let mut programs = ProgramArena::new();
            programs.diagnostic_counters_enabled = true;
            let base_domain = FamilyDomain::new(0, LAYERS + 2).unwrap();
            let source = SemanticFamilySourceIdentity {
                stable_definition: "layered-matrix-source".to_owned(),
                invocation: "invocation".to_owned(),
                element_type: ResolvedValueType::Matrix(matrix_type()),
                domain: base_domain,
                artifact: None,
            };
            let mut family = programs.source_family(&mut expressions, source, None).unwrap();
            let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
            let one = expressions
                .intern(ValueOperator::Constant(TypedConstant::int(1)), Box::new([]))
                .unwrap();
            let mapped = expressions
                .intern_slice(
                    ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                    &[argument, one],
                )
                .unwrap();
            let construction_nodes_before = expressions.node_count() as u64;
            let construction_visits_before = programs.diagnostic_counters().beta_nodes_visited;
            for depth in 1..=LAYERS {
                let previous_maximum = LAYERS + 3 - depth;
                let range = TrustedIndexRange { minimum: 1, maximum_exclusive: previous_maximum };
                let value = if compact {
                    programs
                        .call_family_in_range_deferred_generated(
                            &mut expressions,
                            family,
                            mapped,
                            range,
                        )
                        .unwrap()
                } else {
                    programs.call_family_in_range(&mut expressions, family, mapped, range).unwrap()
                };
                family = programs
                    .generated_family_from_body(
                        &mut expressions,
                        FamilyDomain::new(0, previous_maximum - 1).unwrap(),
                        value,
                    )
                    .unwrap();
            }
            let construction_nodes = expressions.node_count() as u64 - construction_nodes_before;
            let construction_visits =
                programs.diagnostic_counters().beta_nodes_visited - construction_visits_before;
            let exposure_nodes_before = expressions.node_count() as u64;
            let exposure_visits_before = programs.diagnostic_counters().beta_nodes_visited;
            let result = programs
                .call_family_in_range(
                    &mut expressions,
                    family,
                    argument,
                    TrustedIndexRange {
                        minimum: 0,
                        maximum_exclusive: base_domain.maximum_exclusive - LAYERS,
                    },
                )
                .unwrap();
            let exposure_nodes = expressions.node_count() as u64 - exposure_nodes_before;
            let exposure_visits =
                programs.diagnostic_counters().beta_nodes_visited - exposure_visits_before;
            programs.ensure_no_reducible_generated_calls(&expressions, result).unwrap();
            Run {
                construction_nodes,
                construction_visits,
                exposure_nodes,
                exposure_visits,
                fingerprint: fingerprint(&programs, &expressions, result),
            }
        }

        let eager = run(false);
        let compact = run(true);
        assert!(eager.construction_visits >= LAYERS * (LAYERS + 1) / 2);
        assert_eq!(compact.construction_visits, 0);
        assert!(compact.construction_nodes <= 4 * LAYERS);
        assert!(compact.exposure_visits <= 8 * LAYERS);
        assert!(compact.exposure_nodes <= 8 * LAYERS);
        assert_eq!(compact.fingerprint, eager.fingerprint);
    }

    #[test]
    fn zip_offset_rejects_unfinalized_and_out_of_domain_maps() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let direct =
            programs.source_family(&mut expressions, source("boundary-direct"), None).unwrap();
        let offset =
            programs.source_family(&mut expressions, source("boundary-offset"), None).unwrap();
        let outer_argument = expressions.intern_argument(1, ResolvedValueType::Int).unwrap();
        assert_eq!(
            programs.finalize(
                &mut expressions,
                family_signature(domain(), ResolvedValueType::Int),
                outer_argument,
            ),
            Err(ArenaError::FreeArgumentEscapes { position: 1 })
        );
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let one = expressions
            .intern(
                ValueOperator::Constant(super::super::arena::TypedConstant::int(1)),
                Box::new([]),
            )
            .unwrap();
        let mapped = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                &[argument, one],
            )
            .unwrap();
        let map_program = index_map_program(&mut programs, &mut expressions, domain(), mapped);
        let wrong_domain_program = index_map_program(
            &mut programs,
            &mut expressions,
            FamilyDomain::new(0, 5).unwrap(),
            mapped,
        );
        let mut facts = FactStore::new(&expressions);
        facts
            .declare_scoped_trusted_index_range(
                &expressions,
                map_program,
                mapped,
                TrustedIndexRange { minimum: 1, maximum_exclusive: 5 },
            )
            .unwrap();
        let operation = ValueOperator::Scalar(super::super::arena::ScalarOperation::Add);
        assert_eq!(
            programs.zip_offset(
                &mut expressions,
                &facts,
                direct,
                offset,
                wrong_domain_program,
                operation.clone(),
            ),
            Err(ArenaError::ProgramSignatureMismatch)
        );
        assert_eq!(
            programs.zip_offset(
                &mut expressions,
                &facts,
                direct,
                offset,
                map_program,
                operation.clone(),
            ),
            Err(ArenaError::IndexRangeRequired { id: mapped })
        );
        facts.finalize_ranges();
        assert_eq!(
            programs.zip_offset(&mut expressions, &facts, direct, offset, map_program, operation,),
            Err(ArenaError::InvalidRange { minimum: 1, maximum_exclusive: 5 })
        );
    }

    #[test]
    fn selection_requires_closed_or_explicit_unary_program_selector() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let mut facts = FactStore::new(&expressions);
        let left = programs.source_family(&mut expressions, source("select-left"), None).unwrap();
        let right = programs.source_family(&mut expressions, source("select-right"), None).unwrap();
        let outer_argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        assert!(matches!(
            expressions.close(outer_argument),
            Err(ArenaError::FreeArgumentEscapes { position: 0 })
        ));
        let four = expressions
            .intern(
                ValueOperator::Constant(super::super::arena::TypedConstant::int(4)),
                Box::new([]),
            )
            .unwrap();
        let derived_selector = expressions
            .intern(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Remainder),
                Box::new([outer_argument, four]),
            )
            .unwrap();
        let range = TrustedIndexRange { minimum: 0, maximum_exclusive: 4 };
        let selector_program = programs
            .finalize(
                &mut expressions,
                ProgramSignature {
                    inputs: Box::new([ProgramInput {
                        value_type: ResolvedValueType::Int,
                        trusted_index_range: Some(range),
                    }]),
                    output: ResolvedValueType::Int,
                },
                derived_selector,
            )
            .unwrap();
        facts
            .declare_scoped_trusted_index_range(
                &expressions,
                selector_program,
                derived_selector,
                range,
            )
            .unwrap();
        facts.finalize_ranges();
        let selector_capability = programs.selector(&expressions, selector_program).unwrap();
        let selected = programs
            .select(&mut expressions, &facts, selector_capability, &[left, right, left, right])
            .unwrap();
        let body = expressions.node(programs.family_body(selected).unwrap()).unwrap();
        let selector = expressions.node(body.inputs[0]).unwrap();
        assert_eq!(
            selector.operator,
            ValueOperator::ProgramCall { program: selector_program },
            "the outer argument must be captured by an explicit call, never reused as the new binder"
        );
    }

    #[test]
    fn binder_open_identity_selector_uses_program_range_without_global_fact() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let facts = FactStore::new(&expressions);
        let domain = FamilyDomain::new(0, 2).unwrap();
        let left = programs
            .source_family(&mut expressions, source_with_domain("binder-select-left", domain), None)
            .unwrap();
        let right = programs
            .source_family(
                &mut expressions,
                source_with_domain("binder-select-right", domain),
                None,
            )
            .unwrap();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let range = TrustedIndexRange { minimum: 0, maximum_exclusive: 2 };
        let selector_program = programs
            .finalize(
                &mut expressions,
                ProgramSignature {
                    inputs: Box::new([ProgramInput {
                        value_type: ResolvedValueType::Int,
                        trusted_index_range: Some(range),
                    }]),
                    output: ResolvedValueType::Int,
                },
                argument,
            )
            .unwrap();
        let selector_capability = programs.selector(&expressions, selector_program).unwrap();
        let selected = programs
            .select(&mut expressions, &facts, selector_capability, &[left, right])
            .expect("the unary binder range is authoritative for an identity selector");
        let body = expressions.node(programs.family_body(selected).unwrap()).unwrap();
        assert!(matches!(
            expressions.node(body.inputs[0]).unwrap().operator,
            ValueOperator::ProgramCall { program } if program == selector_program
        ));
    }

    #[test]
    fn binder_open_remainder_selector_carries_a_distinct_output_range() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let facts = FactStore::new(&expressions);
        let input_domain = FamilyDomain::new(0, 8).unwrap();
        let families = (0..4)
            .map(|index| {
                programs.source_family(
                    &mut expressions,
                    source_with_domain(format!("remainder-select-{index}"), input_domain),
                    None,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let divisor = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(4)), Box::new([]))
            .unwrap();
        let selector = expressions
            .intern(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Remainder),
                Box::new([argument, divisor]),
            )
            .unwrap();
        let selector_program = programs
            .finalize(
                &mut expressions,
                ProgramSignature {
                    inputs: Box::new([ProgramInput {
                        value_type: ResolvedValueType::Int,
                        trusted_index_range: Some(TrustedIndexRange {
                            minimum: input_domain.minimum,
                            maximum_exclusive: input_domain.maximum_exclusive,
                        }),
                    }]),
                    output: ResolvedValueType::Int,
                },
                selector,
            )
            .unwrap();
        let selector_capability = programs.selector(&expressions, selector_program).unwrap();
        let selected = programs
            .select(&mut expressions, &facts, selector_capability, &families)
            .expect("a transformed binder selector must retain its output range explicitly");
        let body = expressions.node(programs.family_body(selected).unwrap()).unwrap();
        assert!(matches!(
            expressions.node(body.inputs[0]).unwrap().operator,
            ValueOperator::ProgramCall { program } if program == selector_program
        ));
    }

    #[test]
    fn binder_open_shifted_remainder_selector_accepts_closed_affine_offset() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let facts = FactStore::new(&expressions);
        let input_domain = FamilyDomain::new(0, 8).unwrap();
        let families = (0..4)
            .map(|index| {
                programs.source_family(
                    &mut expressions,
                    source_with_domain(format!("shifted-remainder-select-{index}"), input_domain),
                    None,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let two = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(2)), Box::new([]))
            .unwrap();
        let three = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(3)), Box::new([]))
            .unwrap();
        let affine_offset = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                &[two, three],
            )
            .unwrap();
        let shifted = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                &[argument, affine_offset],
            )
            .unwrap();
        let divisor = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(4)), Box::new([]))
            .unwrap();
        let selector = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Remainder),
                &[shifted, divisor],
            )
            .unwrap();
        let selector_program = programs
            .finalize(
                &mut expressions,
                ProgramSignature {
                    inputs: Box::new([ProgramInput {
                        value_type: ResolvedValueType::Int,
                        trusted_index_range: Some(TrustedIndexRange {
                            minimum: input_domain.minimum,
                            maximum_exclusive: input_domain.maximum_exclusive,
                        }),
                    }]),
                    output: ResolvedValueType::Int,
                },
                selector,
            )
            .unwrap();
        let selector_capability = programs
            .selector(&expressions, selector_program)
            .expect("positive remainder of a shifted binder has the divisor range");
        let selected = programs
            .select(&mut expressions, &facts, selector_capability, &families)
            .expect("the validated shifted selector should select compactly");
        let body = expressions.node(programs.family_body(selected).unwrap()).unwrap();
        assert!(matches!(
            expressions.node(body.inputs[0]).unwrap().operator,
            ValueOperator::ProgramCall { program } if program == selector_program
        ));
    }

    #[test]
    fn shifted_remainder_selector_rejects_foreign_binder_and_unproved_offset() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let foreign_argument = expressions.intern_argument(1, ResolvedValueType::Int).unwrap();
        let one = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(1)), Box::new([]))
            .unwrap();
        let two = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(2)), Box::new([]))
            .unwrap();
        let non_affine_offset = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Divide),
                &[one, two],
            )
            .unwrap();
        let divisor = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(4)), Box::new([]))
            .unwrap();
        let foreign_shifted = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                &[argument, foreign_argument],
            )
            .unwrap();
        let foreign_root = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Remainder),
                &[foreign_shifted, divisor],
            )
            .unwrap();
        let foreign_program = programs
            .finalize(
                &mut expressions,
                ProgramSignature {
                    inputs: Box::new([
                        ProgramInput {
                            value_type: ResolvedValueType::Int,
                            trusted_index_range: Some(TrustedIndexRange {
                                minimum: 0,
                                maximum_exclusive: 8,
                            }),
                        },
                        ProgramInput {
                            value_type: ResolvedValueType::Int,
                            trusted_index_range: Some(TrustedIndexRange {
                                minimum: 0,
                                maximum_exclusive: 8,
                            }),
                        },
                    ]),
                    output: ResolvedValueType::Int,
                },
                foreign_root,
            )
            .unwrap();
        assert_eq!(
            programs.selector(&expressions, foreign_program),
            Err(ArenaError::ProgramSignatureMismatch)
        );

        let shifted = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                &[argument, non_affine_offset],
            )
            .unwrap();
        let root = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Remainder),
                &[shifted, divisor],
            )
            .unwrap();
        let program = programs
            .finalize(
                &mut expressions,
                ProgramSignature {
                    inputs: Box::new([ProgramInput {
                        value_type: ResolvedValueType::Int,
                        trusted_index_range: Some(TrustedIndexRange {
                            minimum: 0,
                            maximum_exclusive: 8,
                        }),
                    }]),
                    output: ResolvedValueType::Int,
                },
                root,
            )
            .unwrap();
        assert_eq!(
            programs.selector(&expressions, program),
            Err(ArenaError::ProgramSignatureMismatch)
        );
    }

    #[test]
    fn selector_rejects_forged_output_range_capability() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let facts = FactStore::new(&expressions);
        let domain = FamilyDomain::new(0, 8).unwrap();
        let families = (0..4)
            .map(|index| {
                programs.source_family(
                    &mut expressions,
                    source_with_domain(format!("forged-selector-{index}"), domain),
                    None,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let divisor = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(4)), Box::new([]))
            .unwrap();
        let root = expressions
            .intern(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Remainder),
                Box::new([argument, divisor]),
            )
            .unwrap();
        let selector_program = programs
            .finalize(
                &mut expressions,
                ProgramSignature {
                    inputs: Box::new([ProgramInput {
                        value_type: ResolvedValueType::Int,
                        trusted_index_range: Some(TrustedIndexRange {
                            minimum: domain.minimum,
                            maximum_exclusive: domain.maximum_exclusive,
                        }),
                    }]),
                    output: ResolvedValueType::Int,
                },
                root,
            )
            .unwrap();
        let forged = SelectionSelector::Program(ProgramSelector {
            program: selector_program,
            output_range: TrustedIndexRange { minimum: 0, maximum_exclusive: 3 },
        });
        assert_eq!(
            programs.select(&mut expressions, &facts, forged, &families),
            Err(ArenaError::ProgramSignatureMismatch)
        );
    }

    #[test]
    fn selection_rejects_program_selector_signature_mismatch() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let selector = expressions
            .intern(
                ValueOperator::Constant(super::super::arena::TypedConstant::int(0)),
                Box::new([]),
            )
            .unwrap();
        let bad_program = programs
            .finalize(
                &mut expressions,
                ProgramSignature { inputs: Box::new([]), output: ResolvedValueType::Int },
                selector,
            )
            .unwrap();
        assert_eq!(
            programs.selector(&expressions, bad_program),
            Err(ArenaError::ProgramSignatureMismatch)
        );
    }

    #[test]
    fn beta_reduction_is_iterative_and_preserves_exact_types() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let mut facts = FactStore::new(&expressions);
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let signature = family_signature(domain(), ResolvedValueType::Int);
        let family = programs.generated_family(&mut expressions, signature, argument).unwrap();
        let index = expressions
            .intern(
                ValueOperator::Constant(super::super::arena::TypedConstant::int(2)),
                Box::new([]),
            )
            .unwrap();
        facts
            .declare_trusted_index_range(
                &expressions,
                index,
                TrustedIndexRange { minimum: 2, maximum_exclusive: 3 },
            )
            .unwrap();
        facts.finalize_ranges();
        let reduced = programs.call_family(&mut expressions, &facts, family, index).unwrap();
        assert_eq!(expressions.value_type(reduced).unwrap(), &ResolvedValueType::Int);
        let repeated = programs.call_family(&mut expressions, &facts, family, index).unwrap();
        assert_eq!(repeated, reduced);
    }

    #[test]
    fn canonical_program_call_reductions_do_not_thrash_past_legacy_fifo_bound() {
        const CALLS: u64 = 65_537;
        let mut expressions = ExprArena::new();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let one = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(1)), Box::new([]))
            .unwrap();
        let body = expressions
            .intern_slice(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                &[argument, one],
            )
            .unwrap();
        let mut programs = ProgramArena::new();
        programs.enable_diagnostic_counters();
        let family = programs
            .generated_family(
                &mut expressions,
                family_signature(FamilyDomain::new(0, CALLS).unwrap(), ResolvedValueType::Int),
                body,
            )
            .unwrap();
        let mut first = None;
        for value in 0..CALLS {
            let index = expressions
                .intern(ValueOperator::Constant(TypedConstant::int(value)), Box::new([]))
                .unwrap();
            let output = programs
                .call_family_in_range(
                    &mut expressions,
                    family,
                    index,
                    TrustedIndexRange::new(value, value + 1).unwrap(),
                )
                .unwrap();
            if value == 0 {
                first = Some((index, output));
            }
        }
        let before = programs.diagnostic_counters();
        let nodes_before = expressions.node_count();
        let (first_index, first_output) = first.unwrap();
        let repeated = programs
            .call_family_in_range(
                &mut expressions,
                family,
                first_index,
                TrustedIndexRange::new(0, 1).unwrap(),
            )
            .unwrap();
        let after = programs.diagnostic_counters();
        assert_eq!(repeated, first_output);
        assert_eq!(expressions.node_count(), nodes_before);
        assert_eq!(after.beta_nodes_visited, before.beta_nodes_visited);
        assert_eq!(after.beta_program_call_sidecar_hits, before.beta_program_call_sidecar_hits + 1);
        assert_eq!(after.beta_program_call_sidecar_misses, CALLS);
        assert_eq!(after.beta_program_call_sidecar_entries, CALLS);

        let call = programs.call(&mut expressions, family.program(), &[first_index]).unwrap();
        assert_eq!(expressions.program_call_reduction(call), Ok(Some(first_output)));
        assert_eq!(
            expressions.record_program_call_reduction(call, call),
            Err(ArenaError::ProgramSignatureMismatch)
        );
        let two = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(2)), Box::new([]))
            .unwrap();
        assert_eq!(
            expressions.record_program_call_reduction(call, two),
            Err(ArenaError::ProgramSignatureMismatch)
        );
        let boolean = expressions
            .intern(ValueOperator::Constant(TypedConstant::bool(true)), Box::new([]))
            .unwrap();
        assert_eq!(
            expressions.record_program_call_reduction(call, boolean),
            Err(ArenaError::ProgramSignatureMismatch)
        );
        assert_eq!(expressions.program_call_reduction(call), Ok(Some(first_output)));
        let mut foreign = ExprArena::new();
        let foreign_one =
            foreign.intern(ValueOperator::Constant(TypedConstant::int(1)), Box::new([])).unwrap();
        assert!(matches!(
            expressions.record_program_call_reduction(call, foreign_one),
            Err(ArenaError::ForeignExpression { .. })
        ));
        assert_eq!(expressions.program_call_reduction(call), Ok(Some(first_output)));
    }

    #[test]
    fn foreign_family_and_out_of_domain_access_fail_closed() {
        let mut expressions = ExprArena::new();
        let mut first = ProgramArena::new();
        let second = ProgramArena::new();
        let facts = FactStore::new(&expressions);
        let family = first.source_family(&mut expressions, source("foreign"), None).unwrap();
        assert!(matches!(second.family_domain(family), Err(ArenaError::ForeignProgram { .. })));
        let index = expressions
            .intern(
                ValueOperator::Constant(super::super::arena::TypedConstant::int(0)),
                Box::new([]),
            )
            .unwrap();
        assert!(matches!(
            first.call_family(&mut expressions, &facts, family, index,),
            Err(ArenaError::IndexRangeRequired { .. })
        ));
    }

    #[test]
    fn independent_program_binders_keep_different_domain_authority() {
        let mut expressions = ExprArena::new();
        let mut programs = ProgramArena::new();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let short = programs
            .generated_family_from_body(
                &mut expressions,
                FamilyDomain::new(0, 3).unwrap(),
                argument,
            )
            .unwrap();
        let long = programs
            .generated_family_from_body(
                &mut expressions,
                FamilyDomain::new(0, 7).unwrap(),
                argument,
            )
            .unwrap();
        assert_ne!(short.program(), long.program());
        let short_value = programs
            .call_family_in_range(
                &mut expressions,
                short,
                argument,
                TrustedIndexRange { minimum: 0, maximum_exclusive: 3 },
            )
            .unwrap();
        let long_value = programs
            .call_family_in_range(
                &mut expressions,
                long,
                argument,
                TrustedIndexRange { minimum: 0, maximum_exclusive: 7 },
            )
            .unwrap();
        assert_eq!(short_value, argument);
        assert_eq!(long_value, argument);
        assert!(matches!(
            programs.call_family_in_range(
                &mut expressions,
                short,
                argument,
                TrustedIndexRange { minimum: 0, maximum_exclusive: 7 },
            ),
            Err(ArenaError::InvalidRange { .. })
        ));
    }
}
