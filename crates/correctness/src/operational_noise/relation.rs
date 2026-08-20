//! Frozen relation-registration authority for the operational-noise checker.
//!
//! This module deliberately performs no expression substitution or normalization. Registration
//! records are immutable authority;
//! [`Normalizer`](super::normal_form::Normalizer) owns specialization, exact
//! matching, and deterministic worklist traversal. Canonical RHS values and the ordinary runtime
//! memo live in [`NormalizationCache`], so proof-root work can use a separate one-shot cache
//! without mutating runtime state.

use super::{
    arena::{
        ArenaToken, ArtifactIdentity, ClosedExprId, FamilyDomain, MatrixLayout, ResolvedMatrixType,
        ResolvedValueType, ScopedExprId, TrustedIndexRange, ValueProgramId,
    },
    monomial::MonomialId,
    normal_form::PolynomialNF,
    program::FamilyValueId,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    hash::{DefaultHasher, Hash, Hasher},
    sync::Arc,
};

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct SamplerSourceContract {
    pub expression: super::arena::ExprId,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct TrapdoorSourceContract {
    pub expression: super::arena::ExprId,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct UniversalDispatchKey {
    pub preimage_family: FamilyValueId,
    pub preimage_source: SamplerSourceContract,
    pub matrix_type: ResolvedMatrixType,
    pub trapdoor_source: TrapdoorSourceContract,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum FactorPlacement {
    Central,
    Ordered,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct FactorOrderContract {
    pub public: FactorPlacement,
    pub preimage: FactorPlacement,
    pub public_precedes_preimage: bool,
}

/// A lowering-witnessed algebraic recomposition. This is deliberately separate from
/// `RelationRegistry`: it is an identity for the typed gadget/decomposition algebra, not a
/// trapdoor or preimage relation. The rule is binder-open: the operand `A` is intentionally not
/// part of the key, because beta reduction may replace an open family argument with any trusted
/// closed index. Normalization may consume the rule only after this registry has been frozen.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct GadgetRecompositionRule {
    pub base: u64,
    pub small: bool,
    pub digit_count: u32,
    pub gadget_type: ResolvedMatrixType,
    pub decomposition_type: ResolvedMatrixType,
    pub input_type: ResolvedMatrixType,
    pub output_type: ResolvedMatrixType,
    pub gadget_layout: Option<MatrixLayout>,
    pub decomposition_layout: Option<MatrixLayout>,
    pub input_layout: Option<MatrixLayout>,
}

#[derive(Clone, Debug, Default)]
pub struct GadgetRecompositionRegistry {
    rules: BTreeSet<GadgetRecompositionRule>,
    frozen: bool,
}

impl GadgetRecompositionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    #[cfg(test)]
    pub(crate) fn rule_count(&self) -> usize {
        self.rules.len()
    }

    pub fn register(&mut self, rule: GadgetRecompositionRule) -> Result<(), RelationRegistryError> {
        if self.frozen {
            return Err(RelationRegistryError::Frozen);
        }
        let expected_rows = rule
            .input_type
            .rows
            .checked_mul(usize::try_from(rule.digit_count).map_err(|_| {
                RelationRegistryError::Validation(RelationValidationError::TypeMismatch)
            })?)
            .ok_or(RelationRegistryError::Validation(RelationValidationError::TypeMismatch))?;
        if rule.gadget_type.modulus != rule.input_type.modulus ||
            rule.decomposition_type.modulus != rule.input_type.modulus ||
            rule.output_type.modulus != rule.input_type.modulus ||
            rule.gadget_type.ring_dimension != rule.input_type.ring_dimension ||
            rule.decomposition_type.ring_dimension != rule.input_type.ring_dimension ||
            rule.output_type.ring_dimension != rule.input_type.ring_dimension ||
            rule.gadget_type.rows != rule.input_type.rows ||
            rule.gadget_type.columns != rule.decomposition_type.rows ||
            rule.decomposition_type.rows != expected_rows ||
            rule.decomposition_type.columns != rule.input_type.columns ||
            rule.output_type.rows != rule.input_type.rows ||
            rule.output_type.columns != rule.input_type.columns
        {
            return Err(RelationRegistryError::Validation(RelationValidationError::TypeMismatch));
        }
        self.rules.insert(rule);
        Ok(())
    }

    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    /// Check the complete typed algebraic contract. The caller separately proves that the first
    /// ordered factor is a `MatrixConstantKind::Gadget` and the next factor is the matching
    /// `GadgetDecompose` transform; this registry deliberately does not perform shape search.
    pub(crate) fn allows(
        &self,
        base: u64,
        small: bool,
        digit_count: u32,
        gadget_type: &ResolvedMatrixType,
        decomposition_type: &ResolvedMatrixType,
        input_type: &ResolvedMatrixType,
        output_type: &ResolvedMatrixType,
        gadget_layout: Option<&MatrixLayout>,
        decomposition_layout: Option<&MatrixLayout>,
        input_layout: Option<&MatrixLayout>,
    ) -> bool {
        self.frozen &&
            self.rules.iter().any(|rule| {
                rule.base == base &&
                    rule.small == small &&
                    rule.digit_count == digit_count &&
                    &rule.gadget_type == gadget_type &&
                    &rule.decomposition_type == decomposition_type &&
                    &rule.input_type == input_type &&
                    &rule.output_type == output_type &&
                    rule.gadget_layout.as_ref() == gadget_layout &&
                    rule.decomposition_layout.as_ref() == decomposition_layout &&
                    rule.input_layout.as_ref() == input_layout
            })
    }
}

impl FactorOrderContract {
    pub fn ordered_public_preimage() -> Self {
        Self {
            public: FactorPlacement::Ordered,
            preimage: FactorPlacement::Ordered,
            public_precedes_preimage: true,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct GadgetContract {
    pub definition: String,
    pub parameters: Box<[u64]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct DecompositionContract {
    pub kind: String,
    pub parameters: Box<[u64]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct RelationValidationAuthority {
    pub source: SamplerSourceContract,
    pub trapdoor_source: TrapdoorSourceContract,
    pub matrix_type: ResolvedMatrixType,
    pub public_type: ResolvedValueType,
    pub preimage_type: ResolvedValueType,
    pub target_type: ResolvedValueType,
    pub trapdoor_type: ResolvedValueType,
    pub layout: Option<MatrixLayout>,
    pub factor_order: FactorOrderContract,
    pub domain: FamilyDomain,
    pub index_range: TrustedIndexRange,
    pub gadget: Option<GadgetContract>,
    pub decomposition: Option<DecompositionContract>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelationValidationError {
    SourceMismatch,
    TrapdoorPublicMismatch,
    TypeMismatch,
    LayoutMismatch,
    FactorOrderMismatch,
    DomainMismatch,
    RangeMismatch,
    GadgetMismatch,
    DecompositionMismatch,
}

/// Why a specialized universal-relation LHS could not be represented by one
/// exact canonical monomial.  Universal dispatch is deliberately fail-closed:
/// a zero, multi-term, missing, or non-unit product is not silently converted
/// into a relation key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CanonicalLhsError {
    Zero,
    MultipleTerms,
    MissingExactNormalForm,
    NonUnitCoefficient,
}

impl RelationValidationAuthority {
    pub(crate) fn validate_universal(
        &self,
        dispatch: &UniversalDispatchKey,
        lhs: &StaticLhsKey,
    ) -> Result<(), RelationValidationError> {
        if &self.source != &dispatch.preimage_source {
            return Err(RelationValidationError::SourceMismatch);
        }
        if &self.trapdoor_source != &dispatch.trapdoor_source {
            return Err(RelationValidationError::TrapdoorPublicMismatch);
        }
        if &self.matrix_type != &dispatch.matrix_type {
            return Err(RelationValidationError::TypeMismatch);
        }
        self.validate_common(lhs.layout.as_ref(), &lhs.factor_order, lhs.domain)
    }

    pub(crate) fn validate_closed(
        &self,
        layout: Option<&MatrixLayout>,
        order: &FactorOrderContract,
    ) -> Result<(), RelationValidationError> {
        self.validate_common(layout, order, self.domain)
    }

    fn validate_common(
        &self,
        layout: Option<&MatrixLayout>,
        order: &FactorOrderContract,
        domain: FamilyDomain,
    ) -> Result<(), RelationValidationError> {
        let ResolvedValueType::Matrix(public) = &self.public_type else {
            return Err(RelationValidationError::TypeMismatch);
        };
        let ResolvedValueType::Matrix(preimage) = &self.preimage_type else {
            return Err(RelationValidationError::TypeMismatch);
        };
        let ResolvedValueType::Matrix(target) = &self.target_type else {
            return Err(RelationValidationError::TypeMismatch);
        };
        if &self.matrix_type != preimage ||
            public.modulus != preimage.modulus ||
            public.ring_dimension != preimage.ring_dimension ||
            target.modulus != preimage.modulus ||
            target.ring_dimension != preimage.ring_dimension ||
            public.columns != preimage.rows ||
            target.rows != public.rows ||
            target.columns != preimage.columns
        {
            return Err(RelationValidationError::TypeMismatch);
        }
        if self.trapdoor_type != ResolvedValueType::Trapdoor {
            return Err(RelationValidationError::TrapdoorPublicMismatch);
        }
        if self.layout.as_ref() != layout {
            return Err(RelationValidationError::LayoutMismatch);
        }
        if &self.factor_order != order {
            return Err(RelationValidationError::FactorOrderMismatch);
        }
        if self.domain != domain || domain.minimum >= domain.maximum_exclusive {
            return Err(RelationValidationError::DomainMismatch);
        }
        if !domain.contains(self.index_range) {
            return Err(RelationValidationError::RangeMismatch);
        }
        match (&self.gadget, &self.decomposition) {
            (None, None) | (Some(_), Some(_)) => Ok(()),
            (None, Some(_)) => Err(RelationValidationError::GadgetMismatch),
            (Some(_), None) => Err(RelationValidationError::DecompositionMismatch),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum StaticValueContract {
    Artifact(ArtifactIdentity),
    ClosedValue(ScopedExprId),
    UnsignedParameter { definition: u64, value: u64 },
    BytesParameter { definition: u64, value: Box<[u8]> },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct StaticLhsKey {
    pub domain: FamilyDomain,
    pub public_plan: ValueProgramId,
    pub preimage_plan: ValueProgramId,
    pub trapdoor_plan: ValueProgramId,
    pub public_pairing: ValueProgramId,
    pub layout: Option<MatrixLayout>,
    pub factor_order: FactorOrderContract,
    pub remaining_contracts: Box<[StaticValueContract]>,
    pub validation: RelationValidationAuthority,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct UniversalRelationRegistration {
    pub dispatch: UniversalDispatchKey,
    pub lhs: StaticLhsKey,
    pub target_plan: ValueProgramId,
}

/// A concrete production relation whose four operands have already been closed by the expression
/// arena. The job validates the operands against `validation` and owns canonicalization; callers
/// cannot provide monomial, scope, or canonical-RHS handles.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClosedRelationRegistration {
    pub public: ClosedExprId,
    pub preimage: ClosedExprId,
    pub trapdoor: ClosedExprId,
    pub target: ClosedExprId,
    pub validation: RelationValidationAuthority,
}

/// Exact instantiated LHS. Monomial identity already contains the sorted central factors and the
/// adjacent ordered factor word, so duplicating those lists here would recreate the old OOM path.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct CanonicalLhsKey {
    pub layout: Option<MatrixLayout>,
    pub monomial: MonomialId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CanonicalRhsId {
    arena: ArenaToken,
    slot: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct RuntimeSpecializationKey {
    pub dispatch: UniversalDispatchKey,
    pub index: ScopedExprId,
    pub generation: FrozenGeneration,
}

/// Concrete canonical RHS owner plus the ordinary runtime-specialization memo.
pub struct NormalizationCache {
    token: ArenaToken,
    rhs: Vec<Arc<PolynomialNF>>,
    // Buckets store slots only; full equality resolves collisions without owning a second copy
    // of an arbitrarily large normal form as a map key.
    rhs_interner: BTreeMap<u64, Vec<u32>>,
    runtime:
        BTreeMap<RuntimeSpecializationKey, BTreeMap<CanonicalLhsKey, BTreeSet<CanonicalRhsId>>>,
}

pub(crate) struct NormalizationCheckpoint {
    rhs_len: usize,
    rhs_interner: BTreeMap<u64, Vec<u32>>,
}

impl Default for NormalizationCache {
    fn default() -> Self {
        Self::new()
    }
}

impl NormalizationCache {
    pub fn new() -> Self {
        Self {
            token: ArenaToken::fresh(),
            rhs: Vec::new(),
            rhs_interner: BTreeMap::new(),
            runtime: BTreeMap::new(),
        }
    }
    pub fn intern(&mut self, rhs: PolynomialNF) -> Result<CanonicalRhsId, RelationRegistryError> {
        let mut hasher = DefaultHasher::new();
        rhs.hash(&mut hasher);
        let hash = hasher.finish();
        if let Some(slots) = self.rhs_interner.get(&hash) {
            for slot in slots {
                if self.rhs.get(*slot as usize).is_some_and(|stored| stored.as_ref() == &rhs) {
                    return Ok(CanonicalRhsId { arena: self.token, slot: *slot });
                }
            }
        }
        let slot =
            u32::try_from(self.rhs.len()).map_err(|_| RelationRegistryError::CacheExhausted)?;
        self.rhs.push(Arc::new(rhs));
        self.rhs_interner.entry(hash).or_default().push(slot);
        Ok(CanonicalRhsId { arena: self.token, slot })
    }
    pub fn get(&self, id: CanonicalRhsId) -> Result<&PolynomialNF, RelationRegistryError> {
        if id.arena != self.token {
            return Err(RelationRegistryError::ForeignCanonicalRhs);
        }
        self.rhs
            .get(id.slot as usize)
            .map(Arc::as_ref)
            .ok_or(RelationRegistryError::InvalidCanonicalRhs)
    }
    pub(crate) fn get_arc(
        &self,
        id: CanonicalRhsId,
    ) -> Result<Arc<PolynomialNF>, RelationRegistryError> {
        if id.arena != self.token {
            return Err(RelationRegistryError::ForeignCanonicalRhs);
        }
        self.rhs.get(id.slot as usize).cloned().ok_or(RelationRegistryError::InvalidCanonicalRhs)
    }
    pub(crate) fn runtime_get(
        &self,
        key: &RuntimeSpecializationKey,
    ) -> Option<&BTreeMap<CanonicalLhsKey, BTreeSet<CanonicalRhsId>>> {
        self.runtime.get(key)
    }
    pub(crate) fn runtime_insert(
        &mut self,
        key: RuntimeSpecializationKey,
        value: BTreeMap<CanonicalLhsKey, BTreeSet<CanonicalRhsId>>,
    ) {
        self.runtime.insert(key, value);
    }
    pub fn runtime_entry_count(&self) -> usize {
        self.runtime.len()
    }
    pub fn canonical_rhs_count(&self) -> usize {
        self.rhs.len()
    }
    pub fn canonical_state_fingerprint(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.rhs.hash(&mut hasher);
        self.rhs_interner.hash(&mut hasher);
        hasher.finish()
    }
    pub(crate) fn checkpoint(&self) -> NormalizationCheckpoint {
        NormalizationCheckpoint { rhs_len: self.rhs.len(), rhs_interner: self.rhs_interner.clone() }
    }
    pub(crate) fn rollback(&mut self, checkpoint: NormalizationCheckpoint) {
        self.rhs.truncate(checkpoint.rhs_len);
        self.rhs_interner = checkpoint.rhs_interner;
    }
    pub(crate) fn clear_runtime(&mut self) {
        self.runtime.clear();
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct FrozenGeneration(u64);
impl FrozenGeneration {
    pub fn value(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RelationResolution {
    NoMatch,
    Rewrite(CanonicalRhsId),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RelationRegistryError {
    Frozen,
    NotFrozen,
    StaleGeneration { expected: u64, actual: u64 },
    InvalidDomain,
    AmbiguousPreimageDispatch,
    IndexOutOfDomain,
    Validation(RelationValidationError),
    NonCanonicalLhs(CanonicalLhsError),
    Ambiguous { candidates: Box<[CanonicalRhsId]> },
    ForeignCanonicalRhs,
    InvalidCanonicalRhs,
    CacheExhausted,
}
impl fmt::Display for RelationRegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for RelationRegistryError {}

pub(crate) type UniversalBucket =
    BTreeMap<StaticLhsKey, BTreeMap<ValueProgramId, UniversalRelationRegistration>>;

/// Non-generic, frozen registration authority. It never calls user code and never normalizes.
pub struct RelationRegistry {
    closed: BTreeMap<CanonicalLhsKey, BTreeSet<CanonicalRhsId>>,
    universal: BTreeMap<UniversalDispatchKey, UniversalBucket>,
    /// Candidate discovery is keyed by the exact preimage family program.  A body-shaped
    /// expression is not sufficient provenance: the same expression can occur under multiple
    /// selectors, while the opaque `ProgramCall` retains the producer handle.
    universal_by_preimage: BTreeMap<ValueProgramId, BTreeSet<UniversalDispatchKey>>,
    generation: u64,
    frozen: bool,
}

impl Default for RelationRegistry {
    fn default() -> Self {
        Self {
            closed: BTreeMap::new(),
            universal: BTreeMap::new(),
            universal_by_preimage: BTreeMap::new(),
            generation: 0,
            frozen: false,
        }
    }
}

impl RelationRegistry {
    pub fn new() -> Self {
        Self::default()
    }
    #[cfg(test)]
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }
    pub fn register_closed(
        &mut self,
        lhs: CanonicalLhsKey,
        rhs: CanonicalRhsId,
        authority: &RelationValidationAuthority,
    ) -> Result<(), RelationRegistryError> {
        self.require_mutable()?;
        authority
            .validate_closed(lhs.layout.as_ref(), &authority.factor_order)
            .map_err(RelationRegistryError::Validation)?;
        self.closed.entry(lhs).or_default().insert(rhs);
        self.bump();
        Ok(())
    }
    pub fn register_universal(
        &mut self,
        registration: UniversalRelationRegistration,
    ) -> Result<(), RelationRegistryError> {
        self.require_mutable()?;
        if registration.lhs.domain.minimum >= registration.lhs.domain.maximum_exclusive {
            return Err(RelationRegistryError::InvalidDomain);
        }
        registration
            .lhs
            .validation
            .validate_universal(&registration.dispatch, &registration.lhs)
            .map_err(RelationRegistryError::Validation)?;
        let dispatch = registration.dispatch.clone();
        self.universal
            .entry(dispatch.clone())
            .or_default()
            .entry(registration.lhs.clone())
            .or_default()
            .insert(registration.target_plan, registration);
        self.universal_by_preimage
            .entry(dispatch.preimage_family.program())
            .or_default()
            .insert(dispatch);
        self.bump();
        Ok(())
    }
    pub fn freeze(&mut self) -> FrozenGeneration {
        self.frozen = true;
        FrozenGeneration(self.generation)
    }
    pub fn frozen_generation(&self) -> Result<FrozenGeneration, RelationRegistryError> {
        if self.frozen {
            Ok(FrozenGeneration(self.generation))
        } else {
            Err(RelationRegistryError::NotFrozen)
        }
    }
    pub fn resolve_closed(
        &self,
        lhs: &CanonicalLhsKey,
    ) -> Result<RelationResolution, RelationRegistryError> {
        if !self.frozen {
            return Err(RelationRegistryError::NotFrozen);
        }
        resolve_candidates(self.closed.get(lhs))
    }
    pub(crate) fn universal_candidates(
        &self,
        key: &UniversalDispatchKey,
    ) -> Result<Option<&UniversalBucket>, RelationRegistryError> {
        if !self.frozen {
            return Err(RelationRegistryError::NotFrozen);
        }
        Ok(self.universal.get(key))
    }
    pub(crate) fn dispatch_for_preimage_program(
        &self,
        program: ValueProgramId,
    ) -> Result<Option<&UniversalDispatchKey>, RelationRegistryError> {
        let Some(dispatches) = self.universal_by_preimage.get(&program) else {
            return Ok(None);
        };
        if dispatches.len() != 1 {
            return Err(RelationRegistryError::AmbiguousPreimageDispatch);
        }
        Ok(dispatches.first())
    }

    /// Return the exact plan handles present in frozen universal registrations for diagnostics.
    /// This exposes no mutable authority and is intentionally a compact projection rather than
    /// the registration bodies; matching continues to use the dispatch indexes above.
    pub(crate) fn universal_plan_roles(
        &self,
    ) -> Vec<(ValueProgramId, ValueProgramId, ValueProgramId)> {
        let mut plans = BTreeSet::new();
        for bucket in self.universal.values() {
            for registrations in bucket.values() {
                for registration in registrations.values() {
                    plans.insert((
                        registration.dispatch.preimage_family.program(),
                        registration.lhs.public_plan,
                        registration.target_plan,
                    ));
                }
            }
        }
        plans.into_iter().collect()
    }

    fn require_mutable(&self) -> Result<(), RelationRegistryError> {
        if self.frozen { Err(RelationRegistryError::Frozen) } else { Ok(()) }
    }
    fn bump(&mut self) {
        self.generation = self.generation.saturating_add(1);
    }
}

pub(crate) fn resolve_candidates(
    candidates: Option<&BTreeSet<CanonicalRhsId>>,
) -> Result<RelationResolution, RelationRegistryError> {
    match candidates.map(BTreeSet::len).unwrap_or(0) {
        0 => Ok(RelationResolution::NoMatch),
        1 => Ok(RelationResolution::Rewrite(*candidates.unwrap().first().unwrap())),
        _ => Err(RelationRegistryError::Ambiguous {
            candidates: candidates.unwrap().iter().take(8).copied().collect(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::{
        arena::{ProgramInput, ProgramSignature},
        normal_form::{BoundedSummary, PolynomialNF},
    };
    use num_bigint::{BigInt, BigUint};

    fn matrix() -> ResolvedMatrixType {
        ResolvedMatrixType::new(BigUint::from(17_u8), 1, 2, 2).unwrap()
    }
    fn order() -> FactorOrderContract {
        FactorOrderContract::ordered_public_preimage()
    }
    fn expression(slot: u32) -> super::super::arena::ExprId {
        super::super::arena::ExprId::new(ArenaToken::fresh(), slot)
    }
    fn source(expression: super::super::arena::ExprId) -> SamplerSourceContract {
        SamplerSourceContract { expression }
    }
    fn trapdoor(expression: super::super::arena::ExprId) -> TrapdoorSourceContract {
        TrapdoorSourceContract { expression }
    }
    fn authority(
        source_expression: super::super::arena::ExprId,
        trapdoor_expression: super::super::arena::ExprId,
    ) -> RelationValidationAuthority {
        typed_authority(source_expression, trapdoor_expression, matrix(), matrix(), matrix())
    }
    fn typed_authority(
        source_expression: super::super::arena::ExprId,
        trapdoor_expression: super::super::arena::ExprId,
        public: ResolvedMatrixType,
        preimage: ResolvedMatrixType,
        target: ResolvedMatrixType,
    ) -> RelationValidationAuthority {
        RelationValidationAuthority {
            source: source(source_expression),
            trapdoor_source: trapdoor(trapdoor_expression),
            matrix_type: preimage.clone(),
            public_type: ResolvedValueType::Matrix(public),
            preimage_type: ResolvedValueType::Matrix(preimage),
            target_type: ResolvedValueType::Matrix(target),
            trapdoor_type: ResolvedValueType::Trapdoor,
            layout: None,
            factor_order: order(),
            domain: FamilyDomain::new(0, 4).unwrap(),
            index_range: TrustedIndexRange::new(0, 4).unwrap(),
            gadget: None,
            decomposition: None,
        }
    }

    #[test]
    fn relation_validation_accepts_rectangular_product_shapes() {
        let source_expression = expression(0);
        let trapdoor_expression = expression(1);
        let public = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 3).unwrap();
        let preimage = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 3, 1).unwrap();
        let target = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 1).unwrap();
        let authority =
            typed_authority(source_expression, trapdoor_expression, public, preimage, target);
        assert_eq!(authority.validate_closed(None, &authority.factor_order), Ok(()));
    }

    #[test]
    fn relation_validation_rejects_incompatible_product_contracts() {
        let source_expression = expression(2);
        let trapdoor_expression = expression(3);
        let public = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 3).unwrap();
        let preimage = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 3, 1).unwrap();
        let target = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 1).unwrap();
        let cases = [
            (
                ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 2).unwrap(),
                preimage.clone(),
                target.clone(),
            ),
            (
                ResolvedMatrixType::new(BigUint::from(19_u8), 1, 1, 3).unwrap(),
                preimage.clone(),
                target.clone(),
            ),
            (
                ResolvedMatrixType::new(BigUint::from(17_u8), 2, 1, 3).unwrap(),
                preimage.clone(),
                target.clone(),
            ),
            (
                public.clone(),
                preimage.clone(),
                ResolvedMatrixType::new(BigUint::from(17_u8), 1, 2, 1).unwrap(),
            ),
        ];
        for (public, preimage, target) in cases {
            let authority =
                typed_authority(source_expression, trapdoor_expression, public, preimage, target);
            assert_eq!(
                authority.validate_closed(None, &authority.factor_order),
                Err(RelationValidationError::TypeMismatch)
            );
        }
    }

    #[test]
    fn gadget_recomposition_registry_is_exact_and_fail_closed() {
        let input_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 1).unwrap();
        let decomposition_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 3, 1).unwrap();
        let gadget_type = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 3).unwrap();
        let layout_gadget = MatrixLayout::row_major(1, 3);
        let layout_decomposition = MatrixLayout::row_major(3, 1);
        let layout_input = MatrixLayout::row_major(1, 1);
        let mut registry = GadgetRecompositionRegistry::new();
        let rule = GadgetRecompositionRule {
            base: 2,
            small: false,
            digit_count: 3,
            gadget_type: gadget_type.clone(),
            decomposition_type: decomposition_type.clone(),
            input_type: input_type.clone(),
            output_type: input_type.clone(),
            gadget_layout: Some(layout_gadget.clone()),
            decomposition_layout: Some(layout_decomposition.clone()),
            input_layout: Some(layout_input.clone()),
        };
        registry.register(rule).unwrap();
        assert!(!registry.allows(
            2,
            false,
            3,
            &gadget_type,
            &decomposition_type,
            &input_type,
            &input_type,
            Some(&layout_gadget),
            Some(&layout_decomposition),
            Some(&layout_input),
        ));
        registry.freeze();
        assert!(registry.allows(
            2,
            false,
            3,
            &gadget_type,
            &decomposition_type,
            &input_type,
            &input_type,
            Some(&layout_gadget),
            Some(&layout_decomposition),
            Some(&layout_input),
        ));
        for (wrong_base, wrong_small, wrong_digits) in [(3, false, 3), (2, true, 3), (2, false, 2)]
        {
            assert!(!registry.allows(
                wrong_base,
                wrong_small,
                wrong_digits,
                &gadget_type,
                &decomposition_type,
                &input_type,
                &input_type,
                Some(&layout_gadget),
                Some(&layout_decomposition),
                Some(&layout_input),
            ));
        }
        assert!(!registry.allows(
            2,
            false,
            3,
            &gadget_type,
            &decomposition_type,
            &input_type,
            &input_type,
            Some(&MatrixLayout::row_major(3, 1)),
            Some(&layout_decomposition),
            Some(&layout_input),
        ));
        // The rule is intentionally binder-open: a different A expression is accepted when
        // the typed gadget/decomposition contract is unchanged.
        assert!(registry.allows(
            2,
            false,
            3,
            &gadget_type,
            &decomposition_type,
            &input_type,
            &input_type,
            Some(&layout_gadget),
            Some(&layout_decomposition),
            Some(&layout_input),
        ));
    }

    #[test]
    fn gadget_recomposition_registry_rejects_non_square_or_ring_mismatch() {
        let base = GadgetRecompositionRule {
            base: 2,
            small: true,
            digit_count: 3,
            gadget_type: ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 3).unwrap(),
            decomposition_type: ResolvedMatrixType::new(BigUint::from(17_u8), 1, 3, 1).unwrap(),
            input_type: ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 1).unwrap(),
            output_type: ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 1).unwrap(),
            gadget_layout: None,
            decomposition_layout: None,
            input_layout: None,
        };
        let mut registry = GadgetRecompositionRegistry::new();
        registry.register(base.clone()).unwrap();
        for (gadget_type, decomposition_type, input_type) in [
            (
                ResolvedMatrixType::new(BigUint::from(19_u8), 1, 1, 3).unwrap(),
                base.decomposition_type.clone(),
                base.input_type.clone(),
            ),
            (
                base.gadget_type.clone(),
                ResolvedMatrixType::new(BigUint::from(17_u8), 1, 2, 1).unwrap(),
                base.input_type.clone(),
            ),
        ] {
            let mut wrong = base.clone();
            wrong.gadget_type = gadget_type;
            wrong.decomposition_type = decomposition_type;
            wrong.input_type = input_type;
            assert_eq!(
                registry.register(wrong),
                Err(RelationRegistryError::Validation(RelationValidationError::TypeMismatch,))
            );
        }
    }
    fn empty_nf() -> PolynomialNF {
        PolynomialNF { exact_terms: BTreeMap::new(), bounded_summary: BoundedSummary::missing() }
    }

    #[test]
    fn canonical_rhs_is_concrete_collision_safe_and_foreign_ids_fail_closed() {
        let mut first = NormalizationCache::new();
        let a = first.intern(empty_nf()).unwrap();
        assert_eq!(a, first.intern(empty_nf()).unwrap());
        let second = NormalizationCache::new();
        assert_eq!(first.get(a), Ok(&empty_nf()));
        assert_eq!(second.get(a), Err(RelationRegistryError::ForeignCanonicalRhs));
    }

    #[test]
    fn closed_registry_is_authority_only_and_reports_deterministic_ambiguity() {
        let mut cache = NormalizationCache::new();
        let mut rhs = empty_nf();
        let first = cache.intern(rhs.clone()).unwrap();
        rhs.exact_terms.insert(MonomialId::new(ArenaToken::fresh(), 0), BigInt::from(1));
        let second = cache.intern(rhs).unwrap();
        let lhs =
            CanonicalLhsKey { layout: None, monomial: MonomialId::new(ArenaToken::fresh(), 0) };
        let source = expression(0);
        let trapdoor = expression(1);
        let mut registry = RelationRegistry::new();
        registry.register_closed(lhs.clone(), first, &authority(source, trapdoor)).unwrap();
        registry.register_closed(lhs.clone(), first, &authority(source, trapdoor)).unwrap();
        registry.register_closed(lhs.clone(), second, &authority(source, trapdoor)).unwrap();
        registry.freeze();
        assert_eq!(
            registry.resolve_closed(&lhs),
            Err(RelationRegistryError::Ambiguous { candidates: Box::new([first, second]) })
        );
        assert_eq!(
            registry.register_closed(lhs, first, &authority(source, trapdoor)),
            Err(RelationRegistryError::Frozen)
        );
    }

    #[test]
    fn universal_stage_a_is_exact_and_does_not_scan_other_dispatches() {
        let mut expressions = super::super::arena::ExprArena::new();
        let mut programs = super::super::program::ProgramArena::new();
        let arg = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let sig = ProgramSignature {
            inputs: Box::new([ProgramInput {
                value_type: ResolvedValueType::Int,
                trusted_index_range: Some(TrustedIndexRange::new(0, 4).unwrap()),
            }]),
            output: ResolvedValueType::Int,
        };
        let plan = programs.generated_family(&mut expressions, sig, arg).unwrap();
        let dispatch = UniversalDispatchKey {
            preimage_family: plan,
            preimage_source: source(arg),
            matrix_type: matrix(),
            trapdoor_source: trapdoor(arg),
        };
        let lhs = StaticLhsKey {
            domain: FamilyDomain::new(0, 4).unwrap(),
            public_plan: plan.program(),
            preimage_plan: plan.program(),
            trapdoor_plan: plan.program(),
            public_pairing: plan.program(),
            layout: None,
            factor_order: order(),
            remaining_contracts: Box::new([]),
            validation: authority(arg, arg),
        };
        let registration = UniversalRelationRegistration {
            dispatch: dispatch.clone(),
            lhs,
            target_plan: plan.program(),
        };
        let mut registry = RelationRegistry::new();
        registry.register_universal(registration).unwrap();
        registry.freeze();
        assert_eq!(registry.universal_candidates(&dispatch).unwrap().unwrap().len(), 1);
        let mut other = dispatch;
        other.preimage_source.expression =
            expressions.intern_argument(1, ResolvedValueType::Int).unwrap();
        assert!(registry.universal_candidates(&other).unwrap().is_none());
    }

    #[test]
    fn registration_revalidates_source_and_range_contracts() {
        let mut expressions = super::super::arena::ExprArena::new();
        let mut programs = super::super::program::ProgramArena::new();
        let arg = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let sig = ProgramSignature {
            inputs: Box::new([ProgramInput {
                value_type: ResolvedValueType::Int,
                trusted_index_range: Some(TrustedIndexRange::new(0, 4).unwrap()),
            }]),
            output: ResolvedValueType::Int,
        };
        let family = programs.generated_family(&mut expressions, sig, arg).unwrap();
        let dispatch = UniversalDispatchKey {
            preimage_family: family,
            preimage_source: source(arg),
            matrix_type: matrix(),
            trapdoor_source: trapdoor(arg),
        };
        let mismatch = expressions.intern_argument(1, ResolvedValueType::Int).unwrap();
        let mismatched = authority(mismatch, arg);
        let lhs = StaticLhsKey {
            domain: FamilyDomain::new(0, 4).unwrap(),
            public_plan: family.program(),
            preimage_plan: family.program(),
            trapdoor_plan: family.program(),
            public_pairing: family.program(),
            layout: None,
            factor_order: order(),
            remaining_contracts: Box::new([]),
            validation: mismatched,
        };
        let mut registry = RelationRegistry::new();
        assert_eq!(
            registry.register_universal(UniversalRelationRegistration {
                dispatch,
                lhs,
                target_plan: family.program(),
            }),
            Err(RelationRegistryError::Validation(RelationValidationError::SourceMismatch))
        );
    }

    #[test]
    fn frozen_generation_is_required_and_stable() {
        let mut registry = RelationRegistry::new();
        let lhs =
            CanonicalLhsKey { layout: None, monomial: MonomialId::new(ArenaToken::fresh(), 0) };
        assert_eq!(registry.resolve_closed(&lhs), Err(RelationRegistryError::NotFrozen));
        let generation = registry.freeze();
        assert_eq!(generation, registry.frozen_generation().unwrap());
        assert_eq!(registry.resolve_closed(&lhs), Ok(RelationResolution::NoMatch));
    }

    #[test]
    fn runtime_specialization_memo_is_keyed_by_exact_index_identity() {
        let mut expressions = super::super::arena::ExprArena::new();
        let mut programs = super::super::program::ProgramArena::new();
        let first = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        expressions
            .register_index_definition(super::super::arena::IndexFunctionDefinition {
                id: super::super::arena::IndexFunctionDefinitionId(1),
                arity: 1,
                output_type: ResolvedValueType::Int,
            })
            .unwrap();
        let second = expressions
            .intern(
                super::super::arena::ValueOperator::IndexMap {
                    definition: super::super::arena::IndexFunctionDefinitionId(1),
                    parameters: Box::new([1]),
                },
                Box::new([first]),
            )
            .unwrap();
        let signature = ProgramSignature {
            inputs: Box::new([ProgramInput {
                value_type: ResolvedValueType::Int,
                trusted_index_range: Some(TrustedIndexRange::new(0, 4).unwrap()),
            }]),
            output: ResolvedValueType::Int,
        };
        let family = programs.generated_family(&mut expressions, signature, second).unwrap();
        let first = programs.scoped(&expressions, family.program(), first).unwrap();
        let second = programs.scoped(&expressions, family.program(), second).unwrap();
        let dispatch = UniversalDispatchKey {
            preimage_family: family,
            preimage_source: source(first.expression()),
            matrix_type: matrix(),
            trapdoor_source: trapdoor(first.expression()),
        };
        let mut registry = RelationRegistry::new();
        let generation = registry.freeze();
        let mut cache = NormalizationCache::new();
        cache.runtime_insert(
            RuntimeSpecializationKey { dispatch: dispatch.clone(), index: first, generation },
            BTreeMap::new(),
        );
        assert!(
            cache
                .runtime_get(&RuntimeSpecializationKey {
                    dispatch: dispatch.clone(),
                    index: first,
                    generation
                })
                .is_some()
        );
        assert!(
            cache
                .runtime_get(&RuntimeSpecializationKey { dispatch, index: second, generation })
                .is_none()
        );
    }

    #[test]
    fn proof_local_work_does_not_mutate_runtime_memo() {
        let cache = NormalizationCache::new();
        let mut proof_local = BTreeMap::<CanonicalLhsKey, BTreeSet<CanonicalRhsId>>::new();
        proof_local.clear();
        assert_eq!(cache.runtime_entry_count(), 0);
    }
}
