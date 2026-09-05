//! Candidate-local ownership for the operational-noise checker.
//!
//! This module deliberately owns no lowering semantics. A [`CheckerJob`] owns the
//! expression and program arenas, their fact store, scope-local monomial arenas, and the
//! candidate's artifact aliases. Artifact aliases are only a semantic binding table: they point
//! at an already-produced [`FamilyValueId`] and never create a second source, family, or
//! expression identity.

use super::{
    arena::{
        ArenaError, ArtifactIdentity, ClosedExprId, ExprArena, FamilyDomain, ProgramSignature,
        ResolvedMatrixType, ResolvedValueType, ScopedExprId, TrustedIndexRange,
    },
    facts::{FactStore, MatrixFacts, TrapdoorFacts, ValueFacts},
    monomial::{MonomialArena, MonomialError, TermMap},
    normal_form::{
        AnalyzedValue, BoundedSummary, CompactShellPlan, NormalizationCounters, NormalizeError,
        Normalizer,
    },
    program::{BetaReason, FamilyValueId, ProgramArena},
    relation::{
        FrozenGeneration, GadgetRecompositionRegistry, GadgetRecompositionRule, NormalizationCache,
        RelationRegistry, RelationRegistryError, UniversalRelationRegistration,
    },
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    sync::atomic::{AtomicU64, Ordering},
};
use tracing::info;

#[cfg(test)]
use super::{facts::ScalarFacts, monomial::MonomialId};

static NEXT_CANDIDATE_TOKEN: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProofAnalysisResult {
    pub bounded_summary: BoundedSummary,
    pub exact_term_count: u64,
    pub counters: NormalizationCounters,
    pub exact_term_diagnostics: Box<[ExactTermDiagnostic]>,
}

/// A bounded, proof-free description of one residual exact term. It intentionally contains
/// semantic operation labels only; arena IDs, monomial slots, and relation keys never escape.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ExactTermDiagnostic {
    pub coefficient: String,
    pub central_factors: Box<[FactorDiagnostic]>,
    pub ordered_factors: Box<[FactorDiagnostic]>,
    pub relation: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FactorDiagnostic {
    pub class: &'static str,
    pub operation: &'static str,
    /// Bounded semantic classification for nested program calls and decomposition inputs.
    /// Local expression slots are included solely to correlate one bounded diagnostic with its
    /// children inside the same job; they never participate in interning, matching, or proofs.
    pub detail: String,
}

#[derive(Clone, Debug)]
pub struct ClosedRootAnalysis {
    pub value: AnalyzedValue,
    pub counters: NormalizationCounters,
    pub exact_term_diagnostics: Box<[ExactTermDiagnostic]>,
}

/// A capability naming one active candidate in one checker job.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CandidateToken(u64);

impl CandidateToken {
    fn fresh() -> Self {
        Self(NEXT_CANDIDATE_TOKEN.fetch_add(1, Ordering::Relaxed))
    }
}

/// Semantic identity of an artifact binding.  It contains no producer handle; the producer is
/// stored separately by [`ArtifactAliasTable`].
pub type ArtifactBindingIdentity = ArtifactIdentity;

/// Candidate-scoped resource counts.  Counts are observations of the job-owned stores and are
/// reset when a candidate is finalized; they are never used as semantic identity.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CandidateResourceCounters {
    pub expression_nodes: usize,
    pub programs: usize,
    pub facts: usize,
    pub monomials: usize,
    pub artifact_aliases: usize,
}

/// Finalization result for one candidate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg(test)]
pub struct CandidateReport {
    pub token: CandidateToken,
    pub resources: CandidateResourceCounters,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AliasEntry {
    candidate: CandidateToken,
    producer: FamilyValueId,
    binding: ArtifactBindingIdentity,
}

/// A nonsemantic alias table.  It never interns a producer and never constructs a consumer
/// source: every successful lookup returns the exact producer family handle supplied at insert.
#[derive(Clone, Debug, Default)]
pub struct ArtifactAliasTable {
    active: Option<CandidateToken>,
    entries: BTreeMap<String, AliasEntry>,
}

impl ArtifactAliasTable {
    pub fn new() -> Self {
        Self::default()
    }

    fn begin(&mut self, token: CandidateToken) {
        self.active = Some(token);
        self.entries.clear();
    }

    #[cfg(test)]
    fn finish(&mut self, token: CandidateToken) -> Result<usize, JobError> {
        self.require_active(token)?;
        let count = self.entries.len();
        self.entries.clear();
        self.active = None;
        Ok(count)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[cfg(test)]
    fn require_active(&self, token: CandidateToken) -> Result<(), JobError> {
        match self.active {
            Some(active) if active == token => Ok(()),
            Some(active) => {
                Err(JobError::CandidateTokenMismatch { expected: active, actual: token })
            }
            None => Err(JobError::NoActiveCandidate),
        }
    }

    /// Register a producer family under an exact semantic binding.  The family must belong to
    /// `programs`; its domain and element type must exactly match the binding descriptor.
    #[cfg(test)]
    pub fn register(
        &mut self,
        token: CandidateToken,
        programs: &ProgramArena,
        binding: ArtifactBindingIdentity,
        producer: FamilyValueId,
    ) -> Result<(), JobError> {
        self.require_active(token)?;
        validate_binding(programs, &binding, producer)?;
        if let Some(existing) = self.entries.get(&binding.definition) {
            if existing.candidate != token {
                return Err(JobError::CandidateTokenMismatch {
                    expected: token,
                    actual: existing.candidate,
                });
            }
            if existing.producer == producer && existing.binding == binding {
                return Ok(());
            }
            if existing.binding != binding {
                return Err(JobError::ArtifactBindingMismatch {
                    expected: existing.binding.clone(),
                    actual: binding,
                });
            }
            return Err(JobError::ConflictingArtifactAlias { binding });
        }
        self.entries
            .insert(binding.definition.clone(), AliasEntry { candidate: token, producer, binding });
        Ok(())
    }

    /// Resolve an alias without constructing a consumer identity.
    #[cfg(test)]
    pub fn resolve(
        &self,
        token: CandidateToken,
        programs: &ProgramArena,
        binding: &ArtifactBindingIdentity,
    ) -> Result<FamilyValueId, JobError> {
        self.require_active(token)?;
        let entry = self
            .entries
            .get(&binding.definition)
            .ok_or_else(|| JobError::MissingArtifactAlias { binding: binding.clone() })?;
        if entry.candidate != token {
            return Err(JobError::CandidateTokenMismatch {
                expected: token,
                actual: entry.candidate,
            });
        }
        if entry.binding != *binding {
            return Err(JobError::ArtifactBindingMismatch {
                expected: entry.binding.clone(),
                actual: binding.clone(),
            });
        }
        validate_binding(programs, binding, entry.producer)?;
        Ok(entry.producer)
    }
}

#[cfg(test)]
fn validate_binding(
    programs: &ProgramArena,
    binding: &ArtifactBindingIdentity,
    producer: FamilyValueId,
) -> Result<(), JobError> {
    if binding.definition.is_empty() {
        return Err(JobError::MissingArtifactDefinition);
    }
    let domain = binding.domain.ok_or(JobError::MissingArtifactDomain)?;
    let actual_domain =
        programs.family_domain(producer).map_err(|error| map_family_error(error, producer))?;
    if actual_domain != domain {
        return Err(JobError::ArtifactDomainMismatch { expected: domain, actual: actual_domain });
    }
    let actual_type = programs
        .family_element_type(producer)
        .map_err(|error| map_family_error(error, producer))?;
    if actual_type != binding.value_type {
        return Err(JobError::ArtifactTypeMismatch {
            expected: binding.value_type.clone(),
            actual: actual_type,
        });
    }
    if binding.layout.is_empty() {
        return Err(JobError::MissingArtifactLayout);
    }
    let expected = programs
        .family_artifact(producer)
        .map_err(|error| map_family_error(error, producer))?
        .cloned()
        .ok_or(JobError::UnexportedProducer { producer })?;
    if expected != *binding {
        return Err(JobError::ArtifactBindingMismatch { expected, actual: binding.clone() });
    }
    Ok(())
}

#[cfg(test)]
fn map_family_error(error: ArenaError, producer: FamilyValueId) -> JobError {
    match error {
        ArenaError::ForeignProgram { .. } | ArenaError::InvalidSlot { .. } => {
            JobError::ForeignOrInvalidProducer { producer }
        }
        other => JobError::Arena(other),
    }
}

fn bounded_coefficient(value: &num_bigint::BigInt) -> String {
    let text = value.to_string();
    if text.len() <= 48 { text } else { format!("{}...", &text[..45]) }
}

fn bounded_text(value: &str) -> String {
    let mut text = value.chars().take(96).collect::<String>();
    if value.chars().count() > 96 {
        text.push_str("...");
    }
    text.replace(['[', ']', '(', ')', ',', ';'], "_")
}

fn value_type_detail(value_type: &super::arena::ResolvedValueType) -> String {
    match value_type {
        super::arena::ResolvedValueType::Bool => "bool".to_owned(),
        super::arena::ResolvedValueType::Int => "int".to_owned(),
        super::arena::ResolvedValueType::Real => "real".to_owned(),
        super::arena::ResolvedValueType::Bytes => "bytes".to_owned(),
        super::arena::ResolvedValueType::Trapdoor => "trapdoor".to_owned(),
        super::arena::ResolvedValueType::Matrix(matrix) => format_matrix_type(matrix),
    }
}

fn format_matrix_type(matrix: &super::arena::ResolvedMatrixType) -> String {
    format!(
        "matrix(q={},ring={},shape={}x{})",
        matrix.modulus, matrix.ring_dimension, matrix.rows, matrix.columns
    )
}

fn format_layout(layout: &super::arena::MatrixLayout) -> String {
    format!(
        "layout(name={},row_stride={},column_stride={})",
        bounded_text(&layout.name),
        layout.row_stride,
        layout.column_stride
    )
}

fn matrix_operation_name(operation: &super::arena::MatrixOperation) -> String {
    match operation {
        super::arena::MatrixOperation::Add => "add".to_owned(),
        super::arena::MatrixOperation::Subtract => "subtract".to_owned(),
        super::arena::MatrixOperation::Multiply => "multiply".to_owned(),
        super::arena::MatrixOperation::Negate => "negate".to_owned(),
        super::arena::MatrixOperation::Scale => "scale".to_owned(),
        super::arena::MatrixOperation::Transpose => "transpose".to_owned(),
        super::arena::MatrixOperation::Slice {
            row_start,
            row_end_exclusive,
            column_start,
            column_end_exclusive,
            layout,
        } => format!(
            "slice(rows={}..{},columns={}..{},{} )",
            row_start,
            row_end_exclusive,
            column_start,
            column_end_exclusive,
            format_layout(layout)
        ),
        super::arena::MatrixOperation::IndexedSlice { output, layout } => {
            format!(
                "indexed-slice(output={},{} )",
                format_matrix_type(output),
                format_layout(layout)
            )
        }
        #[cfg(test)]
        super::arena::MatrixOperation::View { output, layout } => {
            format!("view(output={},{} )", format_matrix_type(output), format_layout(layout))
        }
        super::arena::MatrixOperation::Concat { axis, output, layout } => format!(
            "concat(axis={},output={},{} )",
            axis,
            format_matrix_type(output),
            format_layout(layout)
        ),
        super::arena::MatrixOperation::Tensor {
            output,
            left_layout,
            right_layout,
            output_layout,
        } => format!(
            "tensor(output={},left={},right={},out={})",
            format_matrix_type(output),
            format_layout(left_layout),
            format_layout(right_layout),
            format_layout(output_layout)
        ),
        super::arena::MatrixOperation::CrtRecompose {
            plaintext_moduli,
            reconstruction_coefficients,
            output,
        } => format!(
            "crt-recompose(moduli={},coefficients={},output={})",
            plaintext_moduli.len(),
            reconstruction_coefficients.len(),
            format_matrix_type(output)
        ),
        super::arena::MatrixOperation::ExtractCoefficient { row, column } => {
            format!("extract(row={},column={})", row, column)
        }
        super::arena::MatrixOperation::LiftConstantPolynomial { output, coefficient_bits } => {
            format!("lift(output={},bits={})", format_matrix_type(output), coefficient_bits)
        }
    }
}

fn format_groups(groups: &[(super::arena::ExprId, usize)]) -> String {
    groups
        .iter()
        .map(|(id, count)| format!("node={}x{}", id.slot(), count))
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
fn hash_variant_name(variant: super::arena::HashVariant) -> &'static str {
    match variant {
        super::arena::HashVariant::Plain => "plain",
        super::arena::HashVariant::Decomposed => "decomposed",
    }
}

/// Scope-indexed monomial stores owned by one checker job.
///
/// A monomial arena is necessarily scoped because its factors are
/// [`ScopedExprId`](super::arena::ScopedExprId) values. The map is an ownership
/// container only: each scope still has exactly one [`MonomialArena`] and the
/// program arena remains the sole authority for program identity.
#[derive(Default)]
pub struct MonomialStores {
    arenas: BTreeMap<super::arena::ValueProgramId, MonomialArena>,
}

impl MonomialStores {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    pub fn arena_count(&self) -> usize {
        self.arenas.len()
    }

    pub fn term_count(&self) -> usize {
        self.arenas.values().map(MonomialArena::occupied_len).sum()
    }

    pub fn get(&self, scope: super::arena::ValueProgramId) -> Option<&MonomialArena> {
        self.arenas.get(&scope)
    }

    fn ensure(
        &mut self,
        expressions: &ExprArena,
        programs: &ProgramArena,
        scope: super::arena::ValueProgramId,
    ) -> Result<&mut MonomialArena, MonomialError> {
        if !self.arenas.contains_key(&scope) {
            let arena = MonomialArena::new(expressions, programs, scope)?;
            self.arenas.insert(scope, arena);
        }
        // The key was either present already or inserted above.
        Ok(self.arenas.get_mut(&scope).expect("monomial arena inserted or present"))
    }
}

/// One candidate-local owner for all Stage 2 stores.
pub struct CheckerJob {
    expressions: ExprArena,
    programs: ProgramArena,
    facts: FactStore,
    monomials: MonomialStores,
    relations: RelationRegistry,
    gadget_recompositions: GadgetRecompositionRegistry,
    normalization: NormalizationCache,
    aliases: ArtifactAliasTable,
    active_candidate: Option<CandidateToken>,
    candidate_baseline: Option<CandidateResourceCounters>,
    frozen_resources: Option<CandidateResourceCounters>,
    compact_shell_plan: Option<CompactShellPlan>,
}

impl Default for CheckerJob {
    fn default() -> Self {
        Self::new()
    }
}

impl CheckerJob {
    pub fn new() -> Self {
        let expressions = ExprArena::new();
        let facts = FactStore::new(&expressions);
        Self {
            expressions,
            programs: ProgramArena::new(),
            facts,
            monomials: MonomialStores::new(),
            relations: RelationRegistry::new(),
            gadget_recompositions: GadgetRecompositionRegistry::new(),
            normalization: NormalizationCache::new(),
            aliases: ArtifactAliasTable::new(),
            active_candidate: None,
            candidate_baseline: None,
            frozen_resources: None,
            compact_shell_plan: None,
        }
    }

    pub fn expressions(&self) -> &ExprArena {
        &self.expressions
    }

    pub fn expressions_mut(&mut self) -> &mut ExprArena {
        &mut self.expressions
    }

    pub fn programs(&self) -> &ProgramArena {
        &self.programs
    }

    #[cfg(test)]
    pub fn programs_mut(&mut self) -> &mut ProgramArena {
        &mut self.programs
    }

    pub fn facts(&self) -> &FactStore {
        &self.facts
    }

    #[cfg(test)]
    pub(crate) fn active_candidate_token(&self) -> CandidateToken {
        self.active_candidate.expect("test job candidate remains active")
    }

    pub fn declare_trusted_range(
        &mut self,
        token: CandidateToken,
        index: super::arena::ExprId,
        range: TrustedIndexRange,
    ) -> Result<(), JobError> {
        self.require_candidate(token)?;
        if self.relations.is_frozen() {
            return Err(JobError::CandidateAlreadyFrozen);
        }
        if self.facts.ranges_finalized() {
            return Err(JobError::FactsAlreadyFinalized);
        }
        self.facts
            .declare_trusted_index_range(&self.expressions, index, range)
            .map_err(JobError::Facts)
    }

    pub fn finalize_facts(&mut self, token: CandidateToken) -> Result<(), JobError> {
        self.require_candidate(token)?;
        if self.relations.is_frozen() {
            return Err(JobError::CandidateAlreadyFrozen);
        }
        if self.facts.ranges_finalized() {
            return Err(JobError::FactsAlreadyFinalized);
        }
        self.facts.finalize_ranges();
        Ok(())
    }

    #[cfg(test)]
    pub fn insert_scalar_facts(
        &mut self,
        token: CandidateToken,
        expression: super::arena::ExprId,
        facts: ScalarFacts,
    ) -> Result<(), JobError> {
        self.insert_value_facts(token, expression, ValueFacts::Scalar(facts))
    }

    pub fn insert_matrix_facts(
        &mut self,
        token: CandidateToken,
        expression: super::arena::ExprId,
        facts: MatrixFacts,
    ) -> Result<(), JobError> {
        self.insert_value_facts(token, expression, ValueFacts::Matrix(facts))
    }

    pub fn insert_trapdoor_facts(
        &mut self,
        token: CandidateToken,
        expression: super::arena::ExprId,
        facts: TrapdoorFacts,
    ) -> Result<(), JobError> {
        self.insert_value_facts(token, expression, ValueFacts::Trapdoor(facts))
    }

    fn insert_value_facts(
        &mut self,
        token: CandidateToken,
        expression: super::arena::ExprId,
        facts: ValueFacts,
    ) -> Result<(), JobError> {
        self.require_candidate(token)?;
        if self.relations.is_frozen() {
            return Err(JobError::CandidateAlreadyFrozen);
        }
        // Value facts are independent of trusted-index-range finalization.  Lowering may
        // discover a typed constant while resolving a reached node after the range prepass;
        // relation freeze remains the final mutation barrier.
        self.facts.insert(&self.expressions, expression, facts).map_err(JobError::Facts)
    }

    #[cfg(test)]
    pub fn monomials(&self) -> &MonomialStores {
        &self.monomials
    }

    #[cfg(test)]
    pub fn relations(&self) -> &RelationRegistry {
        &self.relations
    }

    pub(crate) fn register_gadget_recomposition(
        &mut self,
        token: CandidateToken,
        rule: GadgetRecompositionRule,
    ) -> Result<(), JobError> {
        self.require_candidate(token)?;
        self.gadget_recompositions.register(rule).map_err(JobError::Relation)
    }

    pub(crate) fn gadget_recompositions(&self) -> &GadgetRecompositionRegistry {
        &self.gadget_recompositions
    }

    pub(crate) fn set_compact_shell_plan(
        &mut self,
        plan: CompactShellPlan,
    ) -> Result<(), JobError> {
        if self.relations.is_frozen() {
            return Err(JobError::CandidateAlreadyFrozen);
        }
        self.compact_shell_plan = Some(plan);
        Ok(())
    }

    /// Describe at most a bounded prefix of exact terms for diagnostics. The returned data is
    /// deliberately semantic-only: no arena IDs, monomial slots, relation keys, or full Debug
    /// representations are retained.
    pub(crate) fn exact_term_diagnostics(
        &self,
        scope: super::arena::ValueProgramId,
        terms: &TermMap<num_bigint::BigInt>,
    ) -> Result<Box<[ExactTermDiagnostic]>, JobError> {
        let arena = self.monomials.get(scope).ok_or(JobError::MissingMonomialScope { scope })?;
        let mut diagnostics = Vec::new();
        for (monomial, coefficient) in terms.iter().take(128) {
            let descriptor = arena.descriptor(*monomial).map_err(JobError::Monomial)?;
            let factors = descriptor
                .central_factors
                .iter()
                .chain(descriptor.ordered_factors.iter())
                .copied()
                .take(64)
                .collect::<Vec<_>>();
            let relation = self.relation_status(&factors);
            let central_factors = descriptor
                .central_factors
                .iter()
                .take(32)
                .map(|factor| self.factor_diagnostic(*factor))
                .collect::<Result<Vec<_>, _>>()?;
            let ordered_factors = descriptor
                .ordered_factors
                .iter()
                .take(32)
                .map(|factor| self.factor_diagnostic(*factor))
                .collect::<Result<Vec<_>, _>>()?;
            diagnostics.push(ExactTermDiagnostic {
                coefficient: bounded_coefficient(coefficient),
                central_factors: central_factors.into_boxed_slice(),
                ordered_factors: ordered_factors.into_boxed_slice(),
                relation,
            });
        }
        Ok(diagnostics.into_boxed_slice())
    }

    fn factor_diagnostic(
        &self,
        factor: super::arena::ScopedExprId,
    ) -> Result<FactorDiagnostic, JobError> {
        let operator =
            &self.expressions.node(factor.expression()).map_err(JobError::Arena)?.operator;
        let (class, operation) = match operator {
            super::arena::ValueOperator::Source(_) => ("public", "source"),
            #[cfg(test)]
            super::arena::ValueOperator::Sample { .. } => ("sample", "sample"),
            super::arena::ValueOperator::Sampler { operation, .. } => {
                let operation = match operation {
                    super::arena::SamplerOperation::Preimage { .. } => "preimage",
                    super::arena::SamplerOperation::Trapdoor { .. } => "trapdoor",
                    #[cfg(test)]
                    super::arena::SamplerOperation::Hash { .. } => "hash",
                    super::arena::SamplerOperation::Gaussian { .. } => "gaussian",
                    super::arena::SamplerOperation::UniformResidue { .. } => "uniform-residue",
                    super::arena::SamplerOperation::UniformInterval { .. } => "uniform-interval",
                };
                ("sampler", operation)
            }
            super::arena::ValueOperator::DeterministicHash(_) => ("sampler", "hash"),
            super::arena::ValueOperator::Trapdoor(_) => ("trapdoor", "transform"),
            super::arena::ValueOperator::Transform(operation) => {
                let operation = match operation {
                    super::arena::ValueTransformOperation::GadgetDecompose { .. } => {
                        "gadget-decompose"
                    }
                    super::arena::ValueTransformOperation::PackPolynomialCoefficients {
                        ..
                    } => "pack-polynomial",
                };
                ("transform", operation)
            }
            super::arena::ValueOperator::Matrix(_) => ("matrix", "matrix-op"),
            super::arena::ValueOperator::Scalar(_) => ("scalar", "scalar-op"),
            super::arena::ValueOperator::Argument { .. } => ("binder", "argument"),
            super::arena::ValueOperator::Constant(_) => ("constant", "literal"),
            super::arena::ValueOperator::ProgramCall { .. } => ("program", "call"),
            #[cfg(test)]
            super::arena::ValueOperator::IndexMap { .. } => ("index", "map"),
            super::arena::ValueOperator::OpaqueFamilyElement { .. } => ("family", "opaque-element"),
            super::arena::ValueOperator::ExplicitElement { .. } => ("family", "explicit-element"),
            super::arena::ValueOperator::ExtractCoefficient { .. } => {
                ("scalar", "extract-coefficient")
            }
        };
        Ok(FactorDiagnostic { class, operation, detail: self.factor_detail(factor.expression())? })
    }

    fn factor_detail(&self, expression: super::arena::ExprId) -> Result<String, JobError> {
        self.expression_detail(expression, 0)
    }

    /// Return a bounded semantic description of one expression and, for a `ProgramCall`, its
    /// finalized callee body. This is diagnostic-only: it does not participate in interning or
    /// relation matching and stops before nested program graphs can grow the output.
    fn expression_detail(
        &self,
        expression: super::arena::ExprId,
        depth: usize,
    ) -> Result<String, JobError> {
        if depth >= 4 {
            return Ok("depth-limit".to_owned());
        }
        let node = self.expressions.node(expression).map_err(JobError::Arena)?;
        let detail = match &node.operator {
            super::arena::ValueOperator::Source(source) => {
                let identity = format!(
                    "def={},role={},invocation={}",
                    bounded_text(&source.stable_definition),
                    bounded_text(&source.output_role),
                    bounded_text(&source.invocation),
                );
                match source.matrix_constant.as_ref() {
                    Some(super::arena::MatrixConstantKind::Gadget { base, small }) => {
                        format!(
                            "source(node={},{};gadget(base={base},small={small}))",
                            expression.slot(),
                            identity
                        )
                    }
                    Some(kind) => format!(
                        "source(node={},{};constant={})",
                        expression.slot(),
                        identity,
                        bounded_text(&format!("{kind:?}"))
                    ),
                    None => format!(
                        "source(node={},{};value={})",
                        expression.slot(),
                        identity,
                        value_type_detail(&source.value_type)
                    ),
                }
            }
            super::arena::ValueOperator::Sampler { operation, .. } => match operation {
                #[cfg(test)]
                super::arena::SamplerOperation::Hash { variant, base, digit_count, .. } => format!(
                    "hash(variant={},base={},digits={})",
                    hash_variant_name(*variant),
                    base.map_or_else(|| "none".to_owned(), |value| value.to_string()),
                    digit_count.map_or_else(|| "none".to_owned(), |value| value.to_string())
                ),
                super::arena::SamplerOperation::Preimage { .. } => "preimage".to_owned(),
                super::arena::SamplerOperation::Trapdoor { .. } => "trapdoor".to_owned(),
                super::arena::SamplerOperation::Gaussian { .. } => "gaussian".to_owned(),
                super::arena::SamplerOperation::UniformResidue { .. } => {
                    "uniform-residue".to_owned()
                }
                super::arena::SamplerOperation::UniformInterval { .. } => {
                    "uniform-interval".to_owned()
                }
            },
            super::arena::ValueOperator::DeterministicHash(descriptor) => format!(
                "hash(node={},variant=plain,definition={:?},version={},tag={})",
                expression.slot(),
                descriptor.definition,
                descriptor.version,
                bounded_text(&String::from_utf8_lossy(&descriptor.tag_prefix))
            ),
            super::arena::ValueOperator::Transform(
                super::arena::ValueTransformOperation::GadgetDecompose {
                    base,
                    small,
                    digit_count,
                    ..
                },
            ) => {
                let input = node
                    .inputs
                    .first()
                    .map(|input| self.expression_detail(*input, depth + 1))
                    .transpose()?
                    .unwrap_or_else(|| "missing-input".to_owned());
                format!("decompose(base={base},small={small},digits={digit_count},input={input})")
            }
            super::arena::ValueOperator::Transform(
                super::arena::ValueTransformOperation::PackPolynomialCoefficients { .. },
            ) => "pack-polynomial".to_owned(),
            super::arena::ValueOperator::ProgramCall { program } => {
                let root = self.programs.program(*program).map_err(JobError::Arena)?.root;
                format!(
                    "program-call(node={},body-node={},body={})",
                    expression.slot(),
                    root.slot(),
                    self.expression_detail(root, depth + 1)?
                )
            }
            super::arena::ValueOperator::OpaqueFamilyElement { .. } => "family-element".to_owned(),
            super::arena::ValueOperator::Constant(constant) => {
                format!("constant({:?})", constant.value)
            }
            super::arena::ValueOperator::Matrix(operation) => {
                self.matrix_operation_detail(expression, node, operation, depth)?
            }
            super::arena::ValueOperator::Scalar(operation) => {
                let inputs = node
                    .inputs
                    .iter()
                    .take(3)
                    .map(|input| self.expression_detail(*input, depth + 1))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(",");
                format!("scalar({operation:?})[{inputs}]")
            }
            super::arena::ValueOperator::Argument { position, .. } => {
                format!("arg{position}")
            }
            super::arena::ValueOperator::Trapdoor(_) => "trapdoor-transform".to_owned(),
            #[cfg(test)]
            super::arena::ValueOperator::IndexMap { .. } => "index-map".to_owned(),
            super::arena::ValueOperator::ExplicitElement { .. } => "explicit-element".to_owned(),
            super::arena::ValueOperator::ExtractCoefficient { .. } => {
                "extract-coefficient".to_owned()
            }
            #[cfg(test)]
            super::arena::ValueOperator::Sample { .. } => "sample".to_owned(),
        };
        Ok(detail)
    }

    fn matrix_operation_detail(
        &self,
        expression: super::arena::ExprId,
        node: &super::arena::ExprNode,
        operation: &super::arena::MatrixOperation,
        depth: usize,
    ) -> Result<String, JobError> {
        let output = self
            .expressions
            .value_type(expression)
            .map_err(JobError::Arena)
            .map(value_type_detail)?;
        let mut groups = Vec::<(super::arena::ExprId, usize)>::new();
        for input in node.inputs.iter().take(8) {
            if let Some((_, count)) = groups.iter_mut().find(|(id, _)| id == input) {
                *count += 1;
            } else {
                groups.push((*input, 1));
            }
        }
        let grouping = if node.inputs.len() > 8 {
            format!("{};truncated={}", format_groups(&groups), node.inputs.len() - 8)
        } else {
            format_groups(&groups)
        };
        let mut inputs = node
            .inputs
            .iter()
            .take(4)
            .map(|input| {
                self.expression_detail(*input, depth + 1)
                    .map(|detail| format!("node={}:{}", input.slot(), detail))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if node.inputs.len() > 4 {
            inputs.push(format!("...+{}", node.inputs.len() - 4));
        }
        Ok(format!(
            "matrix(node={},op={},output={},groups=[{}],inputs=[{}])",
            expression.slot(),
            matrix_operation_name(operation),
            output,
            grouping,
            inputs.join(",")
        ))
    }

    fn relation_status(&self, factors: &[super::arena::ScopedExprId]) -> String {
        for factor in factors {
            let Ok(node) = self.expressions.node(factor.expression()) else { continue };
            let super::arena::ValueOperator::ProgramCall { program } = node.operator else {
                continue;
            };
            match self.relations.dispatch_for_preimage_program(program) {
                Ok(Some(_)) => return "live-preimage-candidate".to_owned(),
                Ok(None) => {}
                Err(RelationRegistryError::AmbiguousPreimageDispatch) => {
                    return "candidate-ambiguous".to_owned()
                }
                Err(RelationRegistryError::Validation(_)) => {
                    return "candidate-validation-mismatch".to_owned()
                }
                Err(_) => return "candidate-resolution-failed".to_owned(),
            }
        }
        "no-live-candidate".to_owned()
    }

    /// Run one arena operation with the job's expression/program/fact stores split safely.
    /// Production lowering needs this because family construction validates expressions against
    /// immutable facts while mutating both arenas; exposing the split here avoids a second owner
    /// or unsafe aliasing at the production boundary.
    pub(crate) fn with_arena_stores<R>(
        &mut self,
        operation: impl FnOnce(&mut ExprArena, &mut ProgramArena, &FactStore) -> Result<R, ArenaError>,
    ) -> Result<R, ArenaError> {
        let (expressions, programs, facts) =
            (&mut self.expressions, &mut self.programs, &self.facts);
        operation(expressions, programs, facts)
    }

    /// Call a family while constructing a not-yet-finalized program body. The range is checked
    /// against this particular family immediately; it is deliberately not cached by raw
    /// expression ID because interned expressions (especially `Argument(0)`) can be reused by
    /// independent program scopes with different domains.
    pub(crate) fn call_family_in_program_scope(
        &mut self,
        family: FamilyValueId,
        index: super::arena::ExprId,
        range: TrustedIndexRange,
    ) -> Result<super::arena::ExprId, ArenaError> {
        self.call_family_in_program_scope_with_reason(family, index, range, BetaReason::Other)
    }

    pub(crate) fn call_family_in_program_scope_with_reason(
        &mut self,
        family: FamilyValueId,
        index: super::arena::ExprId,
        range: TrustedIndexRange,
        reason: BetaReason,
    ) -> Result<super::arena::ExprId, ArenaError> {
        let expression = self.programs.call_family_in_range_with_reason(
            &mut self.expressions,
            family,
            index,
            range,
            reason,
        )?;
        if self.expressions.is_closed(expression)? {
            if let Some(facts) = self.programs.family_matrix_facts(family)?.cloned() {
                self.facts
                    .insert(&self.expressions, expression, ValueFacts::Matrix(facts))
                    .map_err(|error| ArenaError::FactTransferRejected {
                        expression,
                        reason: error.to_string(),
                    })?;
            }
        }
        Ok(expression)
    }

    pub(crate) fn call_family_in_program_scope_deferred_generated(
        &mut self,
        family: FamilyValueId,
        index: super::arena::ExprId,
        range: TrustedIndexRange,
    ) -> Result<super::arena::ExprId, ArenaError> {
        self.call_family_in_program_scope_deferred_generated_with_reason(
            family,
            index,
            range,
            BetaReason::Other,
        )
    }

    pub(crate) fn call_family_in_program_scope_deferred_generated_with_reason(
        &mut self,
        family: FamilyValueId,
        index: super::arena::ExprId,
        range: TrustedIndexRange,
        reason: BetaReason,
    ) -> Result<super::arena::ExprId, ArenaError> {
        let expression = self.programs.call_family_in_range_deferred_generated_with_reason(
            &mut self.expressions,
            family,
            index,
            range,
            reason,
        )?;
        // A deferred call has the same value as the eager family access. Preserve the producer's
        // authoritative element summary on closed source/explicit calls; generated reducible
        // families deliberately carry no such summary.
        if self.expressions.is_closed(expression)? {
            if let Some(facts) = self.programs.family_matrix_facts(family)?.cloned() {
                self.facts
                    .insert(&self.expressions, expression, ValueFacts::Matrix(facts))
                    .map_err(|error| ArenaError::FactTransferRejected {
                        expression,
                        reason: error.to_string(),
                    })?;
            }
        }
        Ok(expression)
    }

    #[cfg(test)]
    pub(crate) fn materialize_reducible_generated_calls(
        &mut self,
        root: super::arena::ExprId,
    ) -> Result<super::arena::ExprId, ArenaError> {
        self.materialize_reducible_generated_calls_with_reason(root, BetaReason::Other)
    }

    pub(crate) fn materialize_reducible_generated_calls_with_reason(
        &mut self,
        root: super::arena::ExprId,
        reason: BetaReason,
    ) -> Result<super::arena::ExprId, ArenaError> {
        self.programs.materialize_reducible_generated_calls_with_reason(
            &mut self.expressions,
            root,
            reason,
        )
    }

    pub(crate) fn transfer_explicit_matrix_facts(
        &mut self,
        branches: &[super::arena::ExprId],
        expression: super::arena::ExprId,
    ) -> Result<(), ArenaError> {
        // This helper is also used while lowering a binder-open synthetic Select body. Facts
        // are facts about closed values; the finalized selected family (or an exact family call)
        // is the authority that can receive the summary later. Do not turn a temporary open DAG
        // wrapper into a global fact insertion failure.
        if !self.expressions.is_closed(expression)? {
            return Ok(());
        }
        let summary =
            self.programs.explicit_matrix_summary(&self.expressions, &self.facts, branches)?;
        if let Some(facts) = summary {
            self.facts.insert(&self.expressions, expression, ValueFacts::Matrix(facts)).map_err(
                |error| ArenaError::FactTransferRejected { expression, reason: error.to_string() },
            )?;
        }
        Ok(())
    }

    /// Validate all program/family handles and the concrete domain/type contract before the
    /// authority-only registry receives a universal registration.
    pub fn register_universal_relation(
        &mut self,
        registration: UniversalRelationRegistration,
    ) -> Result<(), JobError> {
        let domain = self
            .programs
            .family_domain(registration.dispatch.preimage_family)
            .map_err(JobError::Arena)?;
        if domain != registration.lhs.domain || domain != registration.lhs.validation.domain {
            return Err(JobError::Relation(RelationRegistryError::InvalidDomain));
        }
        let element_type = self
            .programs
            .family_element_type(registration.dispatch.preimage_family)
            .map_err(JobError::Arena)?;
        if element_type != registration.lhs.validation.preimage_type {
            return Err(JobError::Relation(RelationRegistryError::Validation(
                super::relation::RelationValidationError::TypeMismatch,
            )));
        }
        let expected_plans = [
            (registration.lhs.public_plan, &registration.lhs.validation.public_type),
            (registration.lhs.preimage_plan, &registration.lhs.validation.preimage_type),
            (registration.target_plan, &registration.lhs.validation.target_type),
            (registration.lhs.trapdoor_plan, &registration.lhs.validation.trapdoor_type),
            (registration.lhs.public_pairing, &registration.lhs.validation.public_type),
        ];
        for (plan, expected_output) in expected_plans {
            let program = self.programs.program(plan).map_err(JobError::Arena)?;
            if program.signature.inputs.len() != 1 ||
                program.signature.inputs[0].value_type != ResolvedValueType::Int ||
                program.signature.inputs[0].trusted_index_range !=
                    Some(TrustedIndexRange {
                        minimum: domain.minimum,
                        maximum_exclusive: domain.maximum_exclusive,
                    }) ||
                &program.signature.output != expected_output ||
                self.expressions.value_type(program.root).map_err(JobError::Arena)? !=
                    expected_output
            {
                return Err(JobError::RelationProgramContractMismatch { plan });
            }
        }
        let preimage_root = self
            .programs
            .materialize_reducible_generated_calls_with_reason(
                &mut self.expressions,
                self.programs
                    .program(registration.lhs.preimage_plan)
                    .map_err(JobError::Arena)?
                    .root,
                BetaReason::JobRelationValidation,
            )
            .map_err(JobError::Arena)?;
        if registration.dispatch.preimage_source.expression != preimage_root ||
            registration.lhs.validation.source != registration.dispatch.preimage_source
        {
            return Err(JobError::RelationSourceMismatch);
        }
        let trapdoor_root = self
            .programs
            .materialize_reducible_generated_calls_with_reason(
                &mut self.expressions,
                self.programs
                    .program(registration.lhs.trapdoor_plan)
                    .map_err(JobError::Arena)?
                    .root,
                BetaReason::JobRelationValidation,
            )
            .map_err(JobError::Arena)?;
        if registration.dispatch.trapdoor_source.expression != trapdoor_root ||
            registration.lhs.validation.trapdoor_source != registration.dispatch.trapdoor_source
        {
            return Err(JobError::RelationTrapdoorMismatch);
        }
        let (paired_public_event, paired_public_output_role) =
            match &self.expressions.node(trapdoor_root).map_err(JobError::Arena)?.operator {
                super::arena::ValueOperator::Trapdoor(
                    super::arena::TrapdoorOperation::Generate {
                        paired_public_event,
                        paired_public_output_role,
                        ..
                    },
                ) => (*paired_public_event, paired_public_output_role.clone()),
                _ => return Err(JobError::RelationTrapdoorMismatch),
            };
        match self.facts.facts(trapdoor_root) {
            Ok(ValueFacts::Trapdoor(facts))
                if facts.paired_public_event == paired_public_event &&
                    facts.paired_public_output_role == paired_public_output_role => {}
            _ => return Err(JobError::RelationTrapdoorMismatch),
        }
        if registration.lhs.public_pairing != registration.lhs.public_plan {
            return Err(JobError::RelationPairingMismatch);
        }
        let public_root = self
            .programs
            .materialize_reducible_generated_calls_with_reason(
                &mut self.expressions,
                self.programs
                    .program(registration.lhs.public_pairing)
                    .map_err(JobError::Arena)?
                    .root,
                BetaReason::JobRelationValidation,
            )
            .map_err(JobError::Arena)?;
        let public_identity =
            match &self.expressions.node(public_root).map_err(JobError::Arena)?.operator {
                super::arena::ValueOperator::Source(source) => {
                    source.sample_event.map(|event| (event, source.output_role.as_str()))
                }
                #[cfg(test)]
                super::arena::ValueOperator::Sample { event, .. } => Some((*event, "value")),
                super::arena::ValueOperator::Sampler { event, .. } => Some((*event, "value")),
                _ => None,
            };
        if public_identity != Some((paired_public_event, paired_public_output_role.as_str())) {
            return Err(JobError::RelationPairingMismatch);
        }
        if let Some(expected_layout) = registration.lhs.layout.as_ref() {
            let target_root = self
                .programs
                .materialize_reducible_generated_calls_with_reason(
                    &mut self.expressions,
                    self.programs.program(registration.target_plan).map_err(JobError::Arena)?.root,
                    BetaReason::JobRelationValidation,
                )
                .map_err(JobError::Arena)?;
            for root in [public_root, preimage_root, target_root] {
                if !matches!(self.facts.facts(root),
                    Ok(super::facts::ValueFacts::Matrix(facts)) if &facts.metadata.layout == expected_layout)
                {
                    return Err(JobError::RelationLayoutMismatch);
                }
            }
        }
        self.relations.register_universal(registration).map_err(JobError::Relation)?;
        Ok(())
    }

    pub fn freeze_relations(
        &mut self,
        token: CandidateToken,
    ) -> Result<FrozenGeneration, JobError> {
        self.require_candidate(token)?;
        if self.relations.is_frozen() {
            return Err(JobError::CandidateAlreadyFrozen);
        }
        if !self.facts.ranges_finalized() {
            return Err(JobError::UnfinalizedIndexRanges);
        }
        self.normalization.clear_runtime();
        let generation = self.relations.freeze();
        self.gadget_recompositions.freeze();
        self.frozen_resources = Some(self.current_resource_counters());
        Ok(generation)
    }

    /// Analyze a closed expression without requiring callers to manufacture a scoped handle.
    /// The zero-argument program is canonicalized by the job's existing program authority.
    pub fn normalize_closed_root(
        &mut self,
        root: ClosedExprId,
    ) -> Result<ClosedRootAnalysis, JobError> {
        self.validate_frozen_resources()?;
        let expression = root.expression();
        let output = self.expressions.value_type(expression).map_err(JobError::Arena)?.clone();
        let program = self
            .programs
            .finalize(
                &mut self.expressions,
                ProgramSignature { inputs: Box::new([]), output },
                expression,
            )
            .map_err(JobError::Arena)?;
        let scoped = self.programs.root(&self.expressions, program).map_err(JobError::Arena)?;
        self.frozen_resources = Some(self.current_resource_counters());
        let (value, counters) = self.normalize(scoped)?;
        let exact_term_diagnostics = value.exact_nf.as_ref().map_or_else(
            || Ok::<Box<[ExactTermDiagnostic]>, JobError>(Box::new([])),
            |normal_form| self.exact_term_diagnostics(program, &normal_form.exact_terms),
        )?;
        Ok(ClosedRootAnalysis { value, counters, exact_term_diagnostics })
    }

    /// Analyze a lowering-selected compact residual.  The marker is private to the production
    /// lowering/report boundary; callers cannot request compact interpretation for arbitrary IR.
    pub(crate) fn normalize_compact_closed_root(
        &mut self,
        root: ClosedExprId,
    ) -> Result<ClosedRootAnalysis, JobError> {
        self.validate_frozen_resources()?;
        let expression = root.expression();
        let output = self.expressions.value_type(expression).map_err(JobError::Arena)?.clone();
        let program = self
            .programs
            .finalize(
                &mut self.expressions,
                ProgramSignature { inputs: Box::new([]), output },
                expression,
            )
            .map_err(JobError::Arena)?;
        let scoped = self.programs.root(&self.expressions, program).map_err(JobError::Arena)?;
        self.frozen_resources = Some(self.current_resource_counters());
        let (value, counters) = self.normalize_compact(scoped)?;
        info!(
            target: "mxx_correctness::operational_noise",
            stage = "normalize_compact_root",
            event = "complete",
            nodes_processed = counters.nodes_processed,
            nodes_total = counters.nodes_total,
            final_exact_terms = counters.final_exact_term_count,
            bounded_folds = counters.bounded_fold_count,
            relation_applied = counters.relation_applied,
            relation_remaining = counters.relation_remaining,
            compact_virtual_calls = counters.compact_virtual_calls,
            compact_algebra_nodes = counters.compact_algebra_nodes,
            compact_concrete_shell_nodes = counters.compact_concrete_shell_nodes,
            planned_shell_occurrences = counters.compact_planned_shell_occurrences,
            planned_scalar_occurrences = counters.compact_scalar_consumers,
            shell_holds_unmatched = counters.compact_shell_holds_unmatched,
            scalar_holds_unmatched = counters.compact_scalar_holds_unmatched,
            peak_frames = counters.compact_peak_live_frames,
            peak_values = counters.compact_peak_live_values,
            "compact normalization summary"
        );
        let exact_term_diagnostics = value.exact_nf.as_ref().map_or_else(
            || Ok::<Box<[ExactTermDiagnostic]>, JobError>(Box::new([])),
            |normal_form| self.exact_term_diagnostics(program, &normal_form.exact_terms),
        )?;
        Ok(ClosedRootAnalysis { value, counters, exact_term_diagnostics })
    }

    /// Analyze one finalized family symbolically at its existing formal `Argument(0)`. No lane
    /// is enumerated and no caller-provided scope or reached-LHS capability is accepted.
    pub(crate) fn analyze_compact_family_root(
        &mut self,
        family: FamilyValueId,
    ) -> Result<ProofAnalysisResult, JobError> {
        self.validate_frozen_resources()?;
        let _domain = self.validate_compact_family_authority(family)?;
        let checkpoint = self.normalization.checkpoint();
        let normalized = self.normalize_compact_family(family);
        self.normalization.rollback(checkpoint);
        self.normalization.clear_runtime();
        let (analyzed, counters) = normalized?;
        let Some(exact) = analyzed.exact_nf.as_ref() else {
            return Ok(ProofAnalysisResult {
                bounded_summary: BoundedSummary::from_contract(analyzed.coefficient_bound)
                    .map_err(JobError::Normalize)?,
                exact_term_count: 0,
                counters,
                exact_term_diagnostics: Box::new([]),
            });
        };
        let exact_term_diagnostics =
            self.exact_term_diagnostics(family.program(), &exact.exact_terms)?;
        Ok(ProofAnalysisResult {
            bounded_summary: exact.bounded_summary.clone(),
            exact_term_count: exact.term_count() as u64,
            counters,
            exact_term_diagnostics,
        })
    }

    pub fn analyze_family_root(
        &mut self,
        family: FamilyValueId,
    ) -> Result<ProofAnalysisResult, JobError> {
        self.validate_frozen_resources()?;
        let domain = self.programs.family_domain(family).map_err(JobError::Arena)?;
        let element_type =
            match self.programs.family_element_type(family).map_err(JobError::Arena)? {
                ResolvedValueType::Matrix(element_type) => element_type,
                _ => return Err(JobError::RelationTypeMismatch),
            };
        let expected_range = TrustedIndexRange {
            minimum: domain.minimum,
            maximum_exclusive: domain.maximum_exclusive,
        };
        let program = self.programs.program(family.program()).map_err(JobError::Arena)?;
        if !self.facts.ranges_finalized() ||
            program.signature.inputs.len() != 1 ||
            program.signature.inputs[0].value_type != ResolvedValueType::Int ||
            program.signature.inputs[0].trusted_index_range != Some(expected_range)
        {
            return Err(JobError::Relation(RelationRegistryError::IndexOutOfDomain));
        }
        let compact_root =
            self.programs.root(&self.expressions, family.program()).map_err(JobError::Arena)?;
        let materialized = self
            .materialize_reducible_generated_calls_with_reason(
                compact_root.expression(),
                BetaReason::JobRelationValidation,
            )
            .map_err(JobError::Arena)?;
        let root = self
            .programs
            .detached_scoped(&self.expressions, family.program(), materialized)
            .map_err(JobError::Arena)?;
        self.frozen_resources = Some(self.current_resource_counters());
        self.analyze_family_root_in_context(family, domain, element_type, root)
    }

    fn analyze_family_root_in_context(
        &mut self,
        family: FamilyValueId,
        domain: FamilyDomain,
        element_type: ResolvedMatrixType,
        root: ScopedExprId,
    ) -> Result<ProofAnalysisResult, JobError> {
        if self.programs.family_domain(family).map_err(JobError::Arena)? != domain ||
            root.program() != family.program()
        {
            return Err(JobError::RelationScopeMismatch {
                expected: family.program(),
                actual: root.program(),
            });
        }
        if self.programs.family_element_type(family).map_err(JobError::Arena)? !=
            ResolvedValueType::Matrix(element_type)
        {
            return Err(JobError::RelationTypeMismatch);
        }
        info!(
            target: "mxx_correctness::operational_noise",
            "analyze online family family={family:?} program={:?} domain={domain:?} body={root:?}",
            family.program(),
        );
        self.log_family_program_paths(root.expression(), family.program())?;
        // Family analysis is local: relation specialization may transiently intern a
        // concrete RHS, but must not mutate the frozen canonical registry or retain runtime IDs
        // whose backing slots are rolled back after this analysis.
        let checkpoint = self.normalization.checkpoint();
        let normalized = self.normalize(root);
        self.normalization.rollback(checkpoint);
        self.normalization.clear_runtime();
        let (analyzed, counters) = normalized?;
        let Some(exact) = analyzed.exact_nf.as_ref() else {
            return Ok(ProofAnalysisResult {
                bounded_summary: BoundedSummary::from_contract(analyzed.coefficient_bound)
                    .map_err(JobError::Normalize)?,
                exact_term_count: 0,
                counters,
                exact_term_diagnostics: Box::new([]),
            });
        };
        let scope = family.program();
        let exact_term_diagnostics = self.exact_term_diagnostics(scope, &exact.exact_terms)?;
        Ok(ProofAnalysisResult {
            bounded_summary: exact.bounded_summary.clone(),
            exact_term_count: exact.term_count() as u64,
            counters,
            exact_term_diagnostics,
        })
    }

    /// Emit a bounded producer-path map for a family root.  Exact normalization intentionally
    /// turns transforms into atoms, so the residual monomial only exposes the outer transform;
    /// this diagnostic walks the immutable expression/program DAG before normalization and keeps
    /// the root-to-call path (including transform edges) for the registered K and public plans.
    /// It is diagnostic-only and never participates in relation matching.
    fn log_family_program_paths(
        &self,
        root: super::arena::ExprId,
        root_program: super::arena::ValueProgramId,
    ) -> Result<(), JobError> {
        const MAX_VISITS: usize = 512;
        const MAX_PATH_DEPTH: usize = 32;
        const MAX_HITS: usize = 32;
        const MAX_SUMMARY_ENTRIES: usize = 24;
        let mut stack = vec![(root, Some(root_program), String::from("root"), 0usize)];
        let mut visited = BTreeSet::new();
        let mut op_counts = BTreeMap::<String, usize>::new();
        let mut path_counts = BTreeMap::<String, usize>::new();
        let mut hits = Vec::<(String, super::arena::ValueProgramId, String)>::new();
        let registered_plans = self.relations.universal_plan_roles();
        let mut visits = 0usize;
        let mut program_calls = 0usize;
        while let Some((expression, context, path, depth)) = stack.pop() {
            if visits >= MAX_VISITS || depth > MAX_PATH_DEPTH {
                break;
            }
            let visit_key = (expression, context);
            if !visited.insert(visit_key) {
                continue;
            }
            visits += 1;
            let node = self.expressions.node(expression).map_err(JobError::Arena)?;
            let operation = match &node.operator {
                super::arena::ValueOperator::Transform(operation) => match operation {
                    super::arena::ValueTransformOperation::GadgetDecompose {
                        base,
                        small,
                        digit_count,
                        ..
                    } => {
                        format!("gadget-decompose(base={base},small={small},digits={digit_count})")
                    }
                    super::arena::ValueTransformOperation::PackPolynomialCoefficients {
                        coefficient_bits,
                        ..
                    } => format!("pack-polynomial(bits={coefficient_bits})"),
                },
                super::arena::ValueOperator::Matrix(operation) => matrix_operation_name(operation),
                super::arena::ValueOperator::ProgramCall { .. } => "program-call".to_owned(),
                super::arena::ValueOperator::Source(_) => "source".to_owned(),
                super::arena::ValueOperator::Sampler { operation, .. } => {
                    let name = match operation {
                        super::arena::SamplerOperation::UniformResidue { .. } => "uniform-residue",
                        super::arena::SamplerOperation::UniformInterval { .. } => {
                            "uniform-interval"
                        }
                        super::arena::SamplerOperation::Gaussian { .. } => "gaussian",
                        #[cfg(test)]
                        super::arena::SamplerOperation::Hash { .. } => "hash",
                        super::arena::SamplerOperation::Trapdoor { .. } => "trapdoor",
                        super::arena::SamplerOperation::Preimage { .. } => "preimage",
                    };
                    format!("sampler({name})")
                }
                super::arena::ValueOperator::DeterministicHash(_) => {
                    "deterministic-hash".to_owned()
                }
                #[cfg(test)]
                super::arena::ValueOperator::Sample { .. } => "sample".to_owned(),
                super::arena::ValueOperator::OpaqueFamilyElement { .. } => {
                    "opaque-family-element".to_owned()
                }
                super::arena::ValueOperator::ExplicitElement { .. } => {
                    "explicit-element".to_owned()
                }
                super::arena::ValueOperator::Trapdoor(_) => "trapdoor".to_owned(),
                super::arena::ValueOperator::Argument { .. } => "argument".to_owned(),
                super::arena::ValueOperator::Constant(_) => "constant".to_owned(),
                #[cfg(test)]
                super::arena::ValueOperator::IndexMap { .. } => "index-map".to_owned(),
                super::arena::ValueOperator::ExtractCoefficient { .. } => {
                    "extract-coefficient".to_owned()
                }
                super::arena::ValueOperator::Scalar(_) => "scalar".to_owned(),
            };
            *op_counts.entry(operation.clone()).or_default() += 1;
            if let super::arena::ValueOperator::ProgramCall { program } = node.operator {
                program_calls += 1;
                let mut roles = BTreeSet::<&'static str>::new();
                for (preimage, public, target) in &registered_plans {
                    if *preimage == program {
                        roles.insert("preimage/K");
                    }
                    if *public == program {
                        roles.insert("public/B");
                    }
                    if *target == program {
                        roles.insert("target/P");
                    }
                }
                for role in roles {
                    let key = format!("{role}:{path}");
                    *path_counts.entry(key).or_default() += 1;
                    if hits.len() < MAX_HITS {
                        hits.push((role.to_owned(), program, path.clone()));
                    }
                }
                if let Ok(callee) = self.programs.program(program) {
                    if depth < MAX_PATH_DEPTH {
                        stack.push((
                            callee.root,
                            Some(program),
                            format!("{path}/call-body({program:?})"),
                            depth + 1,
                        ));
                    }
                }
            }
            for (position, input) in node.inputs.iter().enumerate().rev() {
                if depth < MAX_PATH_DEPTH {
                    stack.push((
                        *input,
                        context,
                        format!("{path}/{operation}[input={position}]"),
                        depth + 1,
                    ));
                }
            }
        }
        let op_summary = op_counts
            .iter()
            .take(MAX_SUMMARY_ENTRIES)
            .map(|(operation, count)| format!("{operation}={count}"))
            .collect::<Vec<_>>()
            .join(",");
        let path_summary = path_counts
            .iter()
            .take(MAX_SUMMARY_ENTRIES)
            .map(|(path, count)| format!("{count}x{}", bounded_text(path)))
            .collect::<Vec<_>>()
            .join("|");
        info!(
            target: "mxx_correctness::operational_noise",
            "family path summary root={root:?} root_program={root_program:?} visited={} program_calls={} op_counts={op_summary} path_counts={path_summary} path_entries={} truncated={}",
            visits,
            program_calls,
            path_counts.len(),
            visits >= MAX_VISITS,
        );
        for (role, program, path) in hits {
            info!(
                target: "mxx_correctness::operational_noise",
                "family path hit role={role} program={program:?} path={path}"
            );
        }
        Ok(())
    }

    /// Normalize through the exact job-owned registry and cache. Keeping this orchestration on
    /// the job prevents callers from accidentally pairing one job's arena with another cache.
    pub fn normalize(
        &mut self,
        root: super::arena::ScopedExprId,
    ) -> Result<(AnalyzedValue, NormalizationCounters), JobError> {
        self.validate_frozen_resources()?;
        let scope = root.program();
        let (
            expressions,
            programs,
            facts,
            monomials,
            relations,
            gadget_recompositions,
            normalization,
        ) = (
            &mut self.expressions,
            &self.programs,
            &self.facts,
            &mut self.monomials,
            &self.relations,
            &self.gadget_recompositions,
            &mut self.normalization,
        );
        let monomial_arena =
            monomials.ensure(expressions, programs, scope).map_err(JobError::Monomial)?;
        let mut normalizer = Normalizer::new(expressions, programs, facts, monomial_arena)
            .map_err(JobError::Normalize)?
            .with_relations(relations, normalization)
            .with_gadget_recompositions(gadget_recompositions);
        let value = normalizer.normalize(root).map_err(JobError::Normalize)?;
        let counters = normalizer.counters();
        self.frozen_resources = Some(self.current_resource_counters());
        Ok((value, counters))
    }

    pub(crate) fn normalize_compact(
        &mut self,
        root: super::arena::ScopedExprId,
    ) -> Result<(AnalyzedValue, NormalizationCounters), JobError> {
        self.validate_frozen_resources()?;
        let scope = root.program();
        let (
            expressions,
            programs,
            facts,
            monomials,
            relations,
            gadget_recompositions,
            normalization,
        ) = (
            &mut self.expressions,
            &self.programs,
            &self.facts,
            &mut self.monomials,
            &self.relations,
            &self.gadget_recompositions,
            &mut self.normalization,
        );
        let monomial_arena =
            monomials.ensure(expressions, programs, scope).map_err(JobError::Monomial)?;
        let mut normalizer = Normalizer::new(expressions, programs, facts, monomial_arena)
            .map_err(JobError::Normalize)?
            .with_relations(relations, normalization)
            .with_gadget_recompositions(gadget_recompositions);
        if let Some(plan) = self.compact_shell_plan.clone() {
            normalizer = normalizer.with_compact_shell_plan(plan);
        }
        let value = normalizer.normalize_compact_root(root).map_err(JobError::Normalize)?;
        let counters = normalizer.counters();
        self.frozen_resources = Some(self.current_resource_counters());
        Ok((value, counters))
    }

    pub(crate) fn normalize_compact_family(
        &mut self,
        family: FamilyValueId,
    ) -> Result<(AnalyzedValue, NormalizationCounters), JobError> {
        self.validate_frozen_resources()?;
        let domain = self.validate_compact_family_authority(family)?;
        let scope = family.program();
        let (
            expressions,
            programs,
            facts,
            monomials,
            relations,
            gadget_recompositions,
            normalization,
        ) = (
            &mut self.expressions,
            &self.programs,
            &self.facts,
            &mut self.monomials,
            &self.relations,
            &self.gadget_recompositions,
            &mut self.normalization,
        );
        let monomial_arena =
            monomials.ensure(expressions, programs, scope).map_err(JobError::Monomial)?;
        let mut normalizer = Normalizer::new(expressions, programs, facts, monomial_arena)
            .map_err(JobError::Normalize)?
            .with_relations(relations, normalization)
            .with_gadget_recompositions(gadget_recompositions);
        if let Some(plan) = self.compact_shell_plan.clone() {
            normalizer = normalizer.with_compact_shell_plan(plan);
        }
        let value =
            normalizer.normalize_compact_family_root(family).map_err(JobError::Normalize)?;
        let counters = normalizer.counters();
        info!(
            target: "mxx_correctness::operational_noise",
            stage = "normalize_compact_family",
            event = "complete",
            family_program = ?family.program(),
            domain_minimum = domain.minimum,
            domain_maximum_exclusive = domain.maximum_exclusive,
            nodes_processed = counters.nodes_processed,
            nodes_total = counters.nodes_total,
            final_exact_terms = counters.final_exact_term_count,
            bounded_folds = counters.bounded_fold_count,
            relation_applied = counters.relation_applied,
            relation_remaining = counters.relation_remaining,
            compact_virtual_calls = counters.compact_virtual_calls,
            compact_algebra_nodes = counters.compact_algebra_nodes,
            compact_concrete_shell_nodes = counters.compact_concrete_shell_nodes,
            planned_shell_occurrences = counters.compact_planned_shell_occurrences,
            planned_scalar_occurrences = counters.compact_scalar_consumers,
            shell_holds_unmatched = counters.compact_shell_holds_unmatched,
            scalar_holds_unmatched = counters.compact_scalar_holds_unmatched,
            peak_frames = counters.compact_peak_live_frames,
            peak_values = counters.compact_peak_live_values,
            "compact family normalization summary"
        );
        self.frozen_resources = Some(self.current_resource_counters());
        Ok((value, counters))
    }

    /// Validate the private compact-family authority once at the job boundary.  The family
    /// remains a normal indexed-family value; this only proves that its existing unary Int
    /// signature and trusted range are exactly the family domain before the shared evaluator is
    /// allowed to inspect its body.
    fn validate_compact_family_authority(
        &self,
        family: FamilyValueId,
    ) -> Result<FamilyDomain, JobError> {
        let domain = self.programs.family_domain(family).map_err(JobError::Arena)?;
        let element_type = self.programs.family_element_type(family).map_err(JobError::Arena)?;
        if !matches!(element_type, ResolvedValueType::Matrix(_)) {
            return Err(JobError::RelationTypeMismatch);
        }
        let expected_range = TrustedIndexRange {
            minimum: domain.minimum,
            maximum_exclusive: domain.maximum_exclusive,
        };
        let program = self.programs.program(family.program()).map_err(JobError::Arena)?;
        if !self.programs.family_is_reducible(family).map_err(JobError::Arena)? ||
            !self.facts.ranges_finalized() ||
            program.signature.inputs.len() != 1 ||
            program.signature.inputs[0].value_type != ResolvedValueType::Int ||
            program.signature.inputs[0].trusted_index_range != Some(expected_range) ||
            program.signature.output != element_type
        {
            return Err(JobError::Relation(RelationRegistryError::IndexOutOfDomain));
        }
        Ok(domain)
    }

    /// Return the scope-local monomial arena, creating it after the scope has been finalized.

    /// Intern one monomial through the job-owned scope store.
    #[cfg(test)]
    pub fn intern_monomial(
        &mut self,
        scope: super::arena::ValueProgramId,
        central_factors: &[super::arena::ScopedExprId],
        ordered_factors: &[super::arena::ScopedExprId],
    ) -> Result<MonomialId, JobError> {
        let (expressions, programs, monomials) =
            (&self.expressions, &self.programs, &mut self.monomials);
        let arena = monomials.ensure(expressions, programs, scope).map_err(JobError::Monomial)?;
        arena
            .intern(expressions, programs, central_factors, ordered_factors)
            .map_err(JobError::Monomial)
    }

    #[cfg(test)]
    pub fn artifact_aliases(&self) -> &ArtifactAliasTable {
        &self.aliases
    }

    pub fn begin_candidate(&mut self) -> Result<CandidateToken, JobError> {
        if self.relations.is_frozen() {
            return Err(JobError::CandidateAlreadyFrozen);
        }
        if let Some(token) = self.active_candidate {
            return Err(JobError::CandidateAlreadyActive { token });
        }
        let token = CandidateToken::fresh();
        self.active_candidate = Some(token);
        self.aliases.begin(token);
        self.candidate_baseline = Some(self.current_resource_counters());
        Ok(token)
    }

    #[cfg(test)]
    pub fn register_artifact_alias(
        &mut self,
        token: CandidateToken,
        binding: ArtifactBindingIdentity,
        producer: FamilyValueId,
    ) -> Result<(), JobError> {
        self.require_candidate(token)?;
        if self.relations.is_frozen() {
            return Err(JobError::CandidateAlreadyFrozen);
        }
        self.aliases.register(token, &self.programs, binding, producer)
    }

    #[cfg(test)]
    pub fn resolve_artifact_alias(
        &self,
        token: CandidateToken,
        binding: &ArtifactBindingIdentity,
    ) -> Result<FamilyValueId, JobError> {
        self.require_candidate(token)?;
        self.aliases.resolve(token, &self.programs, binding)
    }

    #[cfg(test)]
    pub fn finalize_candidate(
        &mut self,
        token: CandidateToken,
    ) -> Result<CandidateReport, JobError> {
        self.require_candidate(token)?;
        let aliases = self.aliases.finish(token)?;
        let baseline = self.candidate_baseline.take().ok_or(JobError::NoActiveCandidate)?;
        self.active_candidate = None;
        let current = self.current_resource_counters();
        Ok(CandidateReport {
            token,
            resources: CandidateResourceCounters {
                expression_nodes: current
                    .expression_nodes
                    .saturating_sub(baseline.expression_nodes),
                programs: current.programs.saturating_sub(baseline.programs),
                facts: current.facts.saturating_sub(baseline.facts),
                monomials: current.monomials.saturating_sub(baseline.monomials),
                artifact_aliases: aliases,
            },
        })
    }

    fn current_resource_counters(&self) -> CandidateResourceCounters {
        CandidateResourceCounters {
            expression_nodes: self.expressions.node_count(),
            programs: self.programs.len(),
            facts: self.facts.len(),
            monomials: self.monomials.term_count(),
            artifact_aliases: self.aliases.len(),
        }
    }

    fn require_candidate(&self, token: CandidateToken) -> Result<(), JobError> {
        match self.active_candidate {
            Some(active) if active == token => Ok(()),
            Some(active) => {
                Err(JobError::CandidateTokenMismatch { expected: active, actual: token })
            }
            None => Err(JobError::NoActiveCandidate),
        }
    }

    fn validate_frozen_resources(&self) -> Result<(), JobError> {
        self.relations.frozen_generation().map_err(JobError::Relation)?;
        let expected = self.frozen_resources.ok_or(JobError::MissingFrozenResourceSnapshot)?;
        let actual = self.current_resource_counters();
        if expected != actual {
            return Err(JobError::FrozenResourcesChanged { expected, actual });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum JobError {
    Arena(ArenaError),
    Facts(super::facts::FactError),
    Monomial(MonomialError),
    Relation(RelationRegistryError),
    Normalize(NormalizeError),
    MissingMonomialScope {
        scope: super::arena::ValueProgramId,
    },
    UnfinalizedIndexRanges,
    FactsAlreadyFinalized,
    RelationScopeMismatch {
        expected: super::arena::ValueProgramId,
        actual: super::arena::ValueProgramId,
    },
    RelationSourceMismatch,
    RelationTypeMismatch,
    RelationLayoutMismatch,
    RelationProgramContractMismatch {
        plan: super::arena::ValueProgramId,
    },
    RelationTrapdoorMismatch,
    RelationPairingMismatch,
    CandidateAlreadyFrozen,
    MissingFrozenResourceSnapshot,
    FrozenResourcesChanged {
        expected: CandidateResourceCounters,
        actual: CandidateResourceCounters,
    },
    NoActiveCandidate,
    CandidateAlreadyActive {
        token: CandidateToken,
    },
    CandidateTokenMismatch {
        expected: CandidateToken,
        actual: CandidateToken,
    },
    #[cfg(test)]
    ForeignOrInvalidProducer {
        producer: FamilyValueId,
    },
    #[cfg(test)]
    UnexportedProducer {
        producer: FamilyValueId,
    },
    #[cfg(test)]
    MissingArtifactAlias {
        binding: ArtifactBindingIdentity,
    },
    #[cfg(test)]
    ConflictingArtifactAlias {
        binding: ArtifactBindingIdentity,
    },
    #[cfg(test)]
    MissingArtifactDefinition,
    #[cfg(test)]
    MissingArtifactDomain,
    #[cfg(test)]
    MissingArtifactLayout,
    #[cfg(test)]
    ArtifactDomainMismatch {
        expected: FamilyDomain,
        actual: FamilyDomain,
    },
    #[cfg(test)]
    ArtifactTypeMismatch {
        expected: ResolvedValueType,
        actual: ResolvedValueType,
    },
    #[cfg(test)]
    ArtifactBindingMismatch {
        expected: ArtifactBindingIdentity,
        actual: ArtifactBindingIdentity,
    },
}

impl fmt::Display for JobError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for JobError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::{
        arena::{
            MatrixLayout, ProgramInput, ProgramSignature, SampleEventId, SamplerOperation,
            SemanticFamilySourceIdentity, TypedConstant, ValueOperator,
        },
        program::TrustedIndexRange,
    };
    use num_bigint::BigUint;

    fn matrix_type() -> super::super::arena::ResolvedMatrixType {
        super::super::arena::ResolvedMatrixType::new(BigUint::from(17_u8), 1, 2, 2).unwrap()
    }

    fn binding(domain: FamilyDomain, version: u32, layout: &str) -> ArtifactBindingIdentity {
        ArtifactIdentity {
            definition: "artifact:input".to_owned(),
            version,
            confidentiality: 1,
            value_type: ResolvedValueType::Matrix(matrix_type()),
            layout: layout.to_owned(),
            domain: Some(domain),
        }
    }

    fn producer(
        job: &mut CheckerJob,
        domain: FamilyDomain,
        binding: &ArtifactBindingIdentity,
    ) -> FamilyValueId {
        let body = job
            .expressions_mut()
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(domain.minimum),
                    operation: SamplerOperation::UniformResidue { output: matrix_type() },
                },
                Box::new([]),
            )
            .unwrap();
        let signature = ProgramSignature {
            inputs: Box::new([ProgramInput {
                value_type: ResolvedValueType::Int,
                trusted_index_range: Some(TrustedIndexRange {
                    minimum: domain.minimum,
                    maximum_exclusive: domain.maximum_exclusive,
                }),
            }]),
            output: ResolvedValueType::Matrix(matrix_type()),
        };
        let _ = (signature, body);
        job.programs
            .source_family(
                &mut job.expressions,
                SemanticFamilySourceIdentity {
                    stable_definition: binding.definition.clone(),
                    invocation: "test".to_owned(),
                    element_type: binding.value_type.clone(),
                    domain,
                    artifact: Some(binding.clone()),
                },
                None,
            )
            .unwrap()
    }

    #[test]
    fn exact_explicit_family_call_transfers_only_its_conservative_matrix_summary() {
        let mut job = CheckerJob::new();
        let values = (0..3)
            .map(|event| {
                job.expressions
                    .intern(
                        ValueOperator::Sampler {
                            event: SampleEventId(event),
                            operation: SamplerOperation::UniformResidue { output: matrix_type() },
                        },
                        Box::new([]),
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        for (value, bound) in values.iter().copied().zip([2_u64, 7, 4]) {
            let mut matrix = super::super::facts::MatrixFacts::new(
                matrix_type(),
                super::super::facts::MatrixMetadata::new(MatrixLayout::row_major(1, 2)),
            );
            matrix.metadata.is_constant_polynomial = true;
            matrix.coefficient_bound = super::super::facts::NumericContract::Known(
                super::super::facts::CoefficientBound::finite(bound),
            );
            matrix.polynomial = super::super::facts::NumericContract::Known(
                super::super::facts::PolynomialFacts::new(2, 2).unwrap(),
            );
            job.facts.insert(&job.expressions, value, ValueFacts::Matrix(matrix)).unwrap();
        }
        let family = job
            .programs
            .explicit_family(
                &mut job.expressions,
                &job.facts,
                FamilyDomain::new(0, 3).unwrap(),
                values.into_boxed_slice(),
            )
            .unwrap();
        let index = job
            .expressions
            .intern(ValueOperator::Constant(TypedConstant::int(1)), Box::new([]))
            .unwrap();
        let call = job
            .call_family_in_program_scope(family, index, TrustedIndexRange::new(0, 3).unwrap())
            .unwrap();
        let ValueFacts::Matrix(facts) = job.facts.facts(call).unwrap().clone() else {
            panic!("explicit family call did not receive matrix facts")
        };
        assert!(facts.metadata.is_constant_polynomial);
        assert_eq!(
            facts.coefficient_bound,
            super::super::facts::NumericContract::Known(
                super::super::facts::CoefficientBound::finite(7_u64)
            )
        );

        let deferred_index = job
            .expressions
            .intern(ValueOperator::Constant(TypedConstant::int(2)), Box::new([]))
            .unwrap();
        let deferred = job
            .call_family_in_program_scope_deferred_generated(
                family,
                deferred_index,
                TrustedIndexRange::new(0, 3).unwrap(),
            )
            .unwrap();
        let ValueFacts::Matrix(deferred_facts) = job.facts.facts(deferred).unwrap() else {
            panic!("deferred explicit family call did not receive matrix facts")
        };
        assert_eq!(deferred_facts, &facts);

        let open_index = job.expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let open_call = job
            .call_family_in_program_scope(family, open_index, TrustedIndexRange::new(0, 3).unwrap())
            .unwrap();
        assert!(job.facts.facts(open_call).is_err());

        let source_family = job
            .programs
            .source_family(
                &mut job.expressions,
                SemanticFamilySourceIdentity {
                    stable_definition: "opaque".to_owned(),
                    invocation: "test".to_owned(),
                    element_type: ResolvedValueType::Matrix(matrix_type()),
                    domain: FamilyDomain::new(0, 3).unwrap(),
                    artifact: None,
                },
                None,
            )
            .unwrap();
        let source_call = job
            .call_family_in_program_scope(
                source_family,
                index,
                TrustedIndexRange::new(0, 3).unwrap(),
            )
            .unwrap();
        assert!(job.facts.facts(source_call).is_err());
    }

    #[test]
    fn binder_open_select_wrapper_does_not_insert_global_value_facts() {
        let mut job = CheckerJob::new();
        let branch = |job: &mut CheckerJob, event| {
            job.expressions
                .intern(
                    ValueOperator::Sampler {
                        event: SampleEventId(event),
                        operation: SamplerOperation::UniformResidue { output: matrix_type() },
                    },
                    Box::new([]),
                )
                .unwrap()
        };
        let left = branch(&mut job, 301);
        let right = branch(&mut job, 302);
        for value in [left, right] {
            let mut matrix = super::super::facts::MatrixFacts::new(
                matrix_type(),
                super::super::facts::MatrixMetadata::new(MatrixLayout::row_major(1, 2)),
            );
            matrix.metadata.is_constant_polynomial = true;
            job.facts.insert(&job.expressions, value, ValueFacts::Matrix(matrix)).unwrap();
        }
        let argument = job.expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let wrapper = job
            .expressions
            .intern(
                ValueOperator::ExplicitElement {
                    domain: FamilyDomain::new(0, 2).unwrap(),
                    element_type: ResolvedValueType::Matrix(matrix_type()),
                },
                Box::new([argument, left, right]),
            )
            .unwrap();
        job.transfer_explicit_matrix_facts(&[left, right], wrapper).unwrap();
        assert!(job.facts.facts(wrapper).is_err());
    }

    #[test]
    fn exact_alias_reuses_the_producer_and_candidate_counters_are_local() {
        let mut job = CheckerJob::new();
        let token = job.begin_candidate().unwrap();
        let domain = FamilyDomain::new(0, 4).unwrap();
        let binding = binding(domain, 7, "row-major-2x2");
        let family = producer(&mut job, domain, &binding);
        let factor = job
            .programs()
            .scoped(
                job.expressions(),
                family.program(),
                job.programs().family_body(family).unwrap(),
            )
            .unwrap();
        let monomial = job.intern_monomial(family.program(), &[], &[factor]).unwrap();
        assert_eq!(job.monomials().arena_count(), 1);
        assert_eq!(job.monomials().term_count(), 1);
        assert_eq!(
            job.monomials()
                .get(family.program())
                .unwrap()
                .descriptor(monomial)
                .unwrap()
                .ordered_factors
                .as_ref(),
            &[factor]
        );
        job.register_artifact_alias(token, binding.clone(), family).unwrap();
        assert_eq!(job.resolve_artifact_alias(token, &binding).unwrap(), family);
        let report = job.finalize_candidate(token).unwrap();
        assert_eq!(report.resources.artifact_aliases, 1);
        assert_eq!(report.resources.monomials, 1);
        assert!(job.artifact_aliases().is_empty());
        let next = job.begin_candidate().unwrap();
        assert_eq!(job.artifact_aliases().len(), 0);
        job.finalize_candidate(next).unwrap();
    }

    #[test]
    fn missing_wrong_and_foreign_aliases_fail_closed() {
        let mut job = CheckerJob::new();
        let token = job.begin_candidate().unwrap();
        let domain = FamilyDomain::new(0, 2).unwrap();
        let binding = binding(domain, 1, "layout");
        let family = producer(&mut job, domain, &binding);
        let mut wrong_registration = binding.clone();
        wrong_registration.version += 1;
        assert!(matches!(
            job.register_artifact_alias(token, wrong_registration, family),
            Err(JobError::ArtifactBindingMismatch { .. })
        ));
        assert!(matches!(
            job.resolve_artifact_alias(token, &binding),
            Err(JobError::MissingArtifactAlias { .. })
        ));
        job.register_artifact_alias(token, binding.clone(), family).unwrap();
        let mut wrong_version = binding.clone();
        wrong_version.version += 1;
        assert!(matches!(
            job.resolve_artifact_alias(token, &wrong_version),
            Err(JobError::ArtifactBindingMismatch { .. })
        ));
        let mut wrong_confidentiality = binding.clone();
        wrong_confidentiality.confidentiality = 2;
        assert!(matches!(
            job.resolve_artifact_alias(token, &wrong_confidentiality),
            Err(JobError::ArtifactBindingMismatch { .. })
        ));
        let mut wrong_layout = binding.clone();
        wrong_layout.layout = "different-layout".to_owned();
        assert!(matches!(
            job.resolve_artifact_alias(token, &wrong_layout),
            Err(JobError::ArtifactBindingMismatch { .. })
        ));
        let mut foreign_programs = ProgramArena::new();
        let mut foreign_expressions = ExprArena::new();
        let foreign_body = foreign_expressions
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(9),
                    operation: SamplerOperation::UniformResidue { output: matrix_type() },
                },
                Box::new([]),
            )
            .unwrap();
        let foreign_family = foreign_programs
            .generated_family(
                &mut foreign_expressions,
                ProgramSignature {
                    inputs: Box::new([ProgramInput {
                        value_type: ResolvedValueType::Int,
                        trusted_index_range: Some(TrustedIndexRange {
                            minimum: domain.minimum,
                            maximum_exclusive: domain.maximum_exclusive,
                        }),
                    }]),
                    output: ResolvedValueType::Matrix(matrix_type()),
                },
                foreign_body,
            )
            .unwrap();
        assert!(matches!(
            job.register_artifact_alias(token, binding, foreign_family),
            Err(JobError::ForeignOrInvalidProducer { .. })
        ));
        job.finalize_candidate(token).unwrap();
    }

    #[test]
    fn generated_sampler_family_is_not_exportable_without_producer_descriptor() {
        let mut job = CheckerJob::new();
        let token = job.begin_candidate().unwrap();
        let domain = FamilyDomain::new(0, 2).unwrap();
        let body = job
            .expressions_mut()
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(77),
                    operation: SamplerOperation::UniformResidue { output: matrix_type() },
                },
                Box::new([]),
            )
            .unwrap();
        let family = job
            .programs
            .generated_family(
                &mut job.expressions,
                ProgramSignature {
                    inputs: Box::new([ProgramInput {
                        value_type: ResolvedValueType::Int,
                        trusted_index_range: Some(TrustedIndexRange {
                            minimum: 0,
                            maximum_exclusive: 2,
                        }),
                    }]),
                    output: ResolvedValueType::Matrix(matrix_type()),
                },
                body,
            )
            .unwrap();
        let binding = binding(domain, 1, "layout");
        assert!(matches!(
            job.register_artifact_alias(token, binding, family),
            Err(JobError::UnexportedProducer { .. })
        ));
        job.finalize_candidate(token).unwrap();
    }

    #[test]
    fn domain_type_layout_and_candidate_lifecycle_are_validated() {
        let mut job = CheckerJob::new();
        let token = job.begin_candidate().unwrap();
        let domain = FamilyDomain::new(0, 3).unwrap();
        let expected_binding = binding(domain, 1, "layout");
        let family = producer(&mut job, domain, &expected_binding);
        let mut missing_domain = expected_binding.clone();
        missing_domain.domain = None;
        assert_eq!(
            job.register_artifact_alias(token, missing_domain, family),
            Err(JobError::MissingArtifactDomain)
        );
        let mut wrong_domain = binding(FamilyDomain::new(0, 4).unwrap(), 1, "layout");
        assert!(matches!(
            job.register_artifact_alias(token, wrong_domain.clone(), family),
            Err(JobError::ArtifactDomainMismatch { .. })
        ));
        wrong_domain.domain = Some(domain);
        wrong_domain.value_type = ResolvedValueType::Int;
        assert!(matches!(
            job.register_artifact_alias(token, wrong_domain.clone(), family),
            Err(JobError::ArtifactTypeMismatch { .. })
        ));
        wrong_domain.value_type = ResolvedValueType::Matrix(matrix_type());
        wrong_domain.layout.clear();
        assert_eq!(
            job.register_artifact_alias(token, wrong_domain, family),
            Err(JobError::MissingArtifactLayout)
        );
        let stale = CandidateToken::fresh();
        assert!(matches!(
            job.finalize_candidate(stale),
            Err(JobError::CandidateTokenMismatch { .. })
        ));
        job.finalize_candidate(token).unwrap();
        assert!(matches!(
            job.resolve_artifact_alias(token, &expected_binding),
            Err(JobError::NoActiveCandidate)
        ));
    }

    #[test]
    fn finalized_static_call_can_precede_later_program_scoped_offset_call() {
        let mut job = CheckerJob::new();
        let token = job.begin_candidate().unwrap();
        let domain = FamilyDomain::new(0, 12).unwrap();
        let source = job
            .with_arena_stores(|expressions, programs, _| {
                programs.source_family(
                    expressions,
                    SemanticFamilySourceIdentity {
                        stable_definition: "range-order-source".to_owned(),
                        invocation: "fixture".to_owned(),
                        element_type: ResolvedValueType::Matrix(matrix_type()),
                        domain,
                        artifact: None,
                    },
                    None,
                )
            })
            .unwrap();
        let zero = job
            .expressions
            .intern(
                ValueOperator::Constant(super::super::arena::TypedConstant::int(0)),
                Box::new([]),
            )
            .unwrap();
        job.declare_trusted_range(
            token,
            zero,
            TrustedIndexRange { minimum: 0, maximum_exclusive: 1 },
        )
        .unwrap();
        job.finalize_facts(token).unwrap();
        job.with_arena_stores(|expressions, programs, facts| {
            programs.call_family(expressions, facts, source, zero)
        })
        .unwrap();

        let argument = job.expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let offset = job
            .expressions
            .intern(
                ValueOperator::Constant(super::super::arena::TypedConstant::int(2)),
                Box::new([]),
            )
            .unwrap();
        let mapped = job
            .expressions
            .intern(
                ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                Box::new([argument, offset]),
            )
            .unwrap();
        let body = job
            .call_family_in_program_scope(
                source,
                mapped,
                TrustedIndexRange { minimum: 2, maximum_exclusive: 7 },
            )
            .unwrap();
        let generated = job
            .programs
            .generated_family_from_body(
                &mut job.expressions,
                FamilyDomain::new(0, 5).unwrap(),
                body,
            )
            .unwrap();
        assert_eq!(
            job.programs.family_domain(generated).unwrap(),
            FamilyDomain::new(0, 5).unwrap()
        );
        assert!(job.facts.ranges_finalized());
    }

    #[test]
    fn value_facts_are_allowed_after_ranges_finalize_but_ranges_are_not() {
        let mut job = CheckerJob::new();
        let token = job.begin_candidate().unwrap();
        let index = job
            .expressions_mut()
            .intern(
                ValueOperator::Constant(super::super::arena::TypedConstant::int(2)),
                Box::new([]),
            )
            .unwrap();
        job.declare_trusted_range(token, index, TrustedIndexRange::new(0, 4).unwrap()).unwrap();
        job.finalize_facts(token).unwrap();

        let matrix = matrix_type();
        let value = job
            .expressions_mut()
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(910),
                    operation: SamplerOperation::UniformResidue { output: matrix.clone() },
                },
                Box::new([]),
            )
            .unwrap();
        job.insert_matrix_facts(
            token,
            value,
            MatrixFacts::new(
                matrix,
                super::super::facts::MatrixMetadata::new(MatrixLayout::row_major(2, 2)),
            ),
        )
        .unwrap();
        assert_eq!(
            job.declare_trusted_range(token, index, TrustedIndexRange::new(0, 4).unwrap()),
            Err(JobError::FactsAlreadyFinalized)
        );

        job.freeze_relations(token).unwrap();
        assert_eq!(
            job.insert_scalar_facts(
                token,
                index,
                ScalarFacts::new(ResolvedValueType::Int).unwrap(),
            ),
            Err(JobError::CandidateAlreadyFrozen)
        );
    }

    #[test]
    fn closed_root_analysis_builds_its_scope_internally_and_reuses_it() {
        let mut job = CheckerJob::new();
        let token = job.begin_candidate().unwrap();
        let value = job
            .expressions
            .intern(
                ValueOperator::Constant(super::super::arena::TypedConstant::int(7)),
                Box::new([]),
            )
            .unwrap();
        let closed = job.expressions.close(value).unwrap();
        let exact_value = job
            .expressions
            .intern(
                ValueOperator::Source(super::super::arena::SemanticSourceIdentity {
                    stable_definition: "closed-exact".to_owned(),
                    invocation: "0".to_owned(),
                    sample_event: None,
                    output_role: "value".to_owned(),
                    sampler: None,
                    artifact: None,
                    value_type: ResolvedValueType::Matrix(matrix_type()),
                    coordinates: Box::new([]),
                    matrix_constant: None,
                }),
                Box::new([]),
            )
            .unwrap();
        let exact_closed = job.expressions.close(exact_value).unwrap();
        job.facts.finalize_ranges();
        job.freeze_relations(token).unwrap();
        let programs_before = job.programs.len();
        let first = job.normalize_closed_root(closed).unwrap();
        let exact = job.normalize_closed_root(exact_closed).unwrap();
        let programs_after = job.programs.len();
        let second = job.normalize_closed_root(closed).unwrap();
        assert!(matches!(
            first.value.coefficient_bound,
            super::super::facts::NumericContract::Known(_)
        ));
        assert!(exact.value.exact_nf.as_ref().is_some_and(|normal| normal.term_count() == 1));
        assert_eq!(first.value.coefficient_bound, second.value.coefficient_bound);
        assert_eq!(programs_after, programs_before + 2);
        assert_eq!(job.programs.len(), programs_after);
    }

    #[test]
    fn family_root_analysis_is_constant_work_and_uses_program_scoped_domain() {
        let million = FamilyDomain::new(0, 1_000_000).unwrap();
        let mut job = CheckerJob::new();
        let token = job.begin_candidate().unwrap();
        let matrix_family = job
            .programs
            .source_family(
                &mut job.expressions,
                SemanticFamilySourceIdentity {
                    stable_definition: "million-matrix".to_owned(),
                    invocation: "0".to_owned(),
                    element_type: ResolvedValueType::Matrix(matrix_type()),
                    domain: million,
                    artifact: None,
                },
                None,
            )
            .unwrap();
        let int_family = job
            .programs
            .source_family(
                &mut job.expressions,
                SemanticFamilySourceIdentity {
                    stable_definition: "million-int".to_owned(),
                    invocation: "0".to_owned(),
                    element_type: ResolvedValueType::Int,
                    domain: million,
                    artifact: None,
                },
                None,
            )
            .unwrap();
        let argument = job.expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        // The shared fact table never stores a raw binder-open range; each family carries the
        // authoritative domain in its finalized unary signature.
        assert!(
            job.expressions
                .free_arguments(argument)
                .unwrap()
                .contains(&(0, ResolvedValueType::Int))
        );
        job.facts.finalize_ranges();
        job.freeze_relations(token).unwrap();
        assert_eq!(job.analyze_family_root(int_family), Err(JobError::RelationTypeMismatch));
        let result = job.analyze_family_root(matrix_family).unwrap();
        assert!(result.counters.nodes_total < 16);

        let mut wrong = CheckerJob::new();
        let token = wrong.begin_candidate().unwrap();
        let family = wrong
            .programs
            .source_family(
                &mut wrong.expressions,
                SemanticFamilySourceIdentity {
                    stable_definition: "wrong-domain".to_owned(),
                    invocation: "0".to_owned(),
                    element_type: ResolvedValueType::Matrix(matrix_type()),
                    domain: million,
                    artifact: None,
                },
                None,
            )
            .unwrap();
        let argument = wrong.expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        assert!(
            wrong
                .expressions
                .free_arguments(argument)
                .unwrap()
                .contains(&(0, ResolvedValueType::Int))
        );
        wrong.facts.finalize_ranges();
        wrong.freeze_relations(token).unwrap();
        let scoped = wrong.analyze_family_root(family).unwrap();
        assert_eq!(scoped.exact_term_count, 1);
    }
}
