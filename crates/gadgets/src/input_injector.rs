//! BGG-independent input-injection preprocessing shared by Diamond applications.

use mxx_dsl::{
    DslError, Family, FamilyAxisSelection, Int, Mat, Parallel, ProofTraceTransport, Ring,
    Sequential, TrapdoorFamily, parallel_zip_bundle, parallel_zip_bundle_trace,
};
use mxx_ir_core::{
    FreezeMap, FreezeResolveError, FrozenGraphScopeId, FrozenStructuralIntExpr, FrozenValueRef,
    IndexMap, IntExpr, RealExpr, ValueHandle, expr::IndexExpr, node::ConcatAxis,
};
use num_bigint::BigInt;
use thiserror::Error;

pub const DIAMOND_SECRET_DIMENSION: usize = 1;
pub const DIAMOND_PREFIX_DIMENSION: usize = 2;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiamondInputConfig {
    pub modulus: BigInt,
    pub ring_dimension: usize,
    pub input_count: usize,
    pub digit_base: usize,
    pub batch_bits: usize,
    pub gadget_base: BigInt,
    pub digit_count: usize,
    pub trapdoor_sigma: RealExpr,
    pub error_sigma: RealExpr,
    pub error_max_coefficient_bound: BigInt,
    pub preimage_max_coefficient_bound: BigInt,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiamondInputParams {
    pub modulus: IntExpr,
    pub ring_dimension: IntExpr,
    pub input_count: IntExpr,
    pub digit_base: IntExpr,
    pub batch_bits: IntExpr,
    pub gadget_base: IntExpr,
    pub digit_count: IntExpr,
    pub trapdoor_sigma: RealExpr,
    pub error_sigma: RealExpr,
    pub error_max_coefficient_bound: IntExpr,
    pub preimage_max_coefficient_bound: IntExpr,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum DiamondInputConfigError {
    #[error("the Diamond input-injection ring modulus must be positive")]
    InvalidModulus,
    #[error(
        "the Diamond input-injection ring dimension, input count, and digit count must be positive"
    )]
    ZeroDimension,
    #[error("batch_bits must be positive and fit the host digit representation")]
    InvalidBatchBits,
    #[error("digit_base must be at least 2^batch_bits")]
    InvalidDigitBase,
    #[error("the gadget base must be at least two")]
    InvalidGadgetBase,
    #[error("sampler coefficient bounds must be nonnegative")]
    InvalidSamplerBound,
    #[error("a Diamond input-injection layout calculation overflowed")]
    LayoutOverflow,
    #[error("Diamond input-injection transition artifacts do not match the configured layout")]
    InvalidTransitionLayout,
}

#[derive(Debug, Error)]
pub enum DiamondInputPreprocessError {
    #[error(transparent)]
    Config(#[from] DiamondInputConfigError),
    #[error(transparent)]
    Dsl(#[from] DslError),
}

pub struct DiamondInputPreprocessing {
    /// Initial input-injection state family indexed by `state`.
    pub initial: Family<Mat>,
    /// Rectangular transition preimages indexed by `(level, state, digit)`.
    pub transitions: Family<mxx_dsl::Preimage>,
    /// Proof-only identities for the actual target family used by `transitions`.
    pub target_trace: DiamondInputTargetTraceFragment,
    /// Proof-only identities for the selector matrices carried into those targets.
    pub selector_magnitude_trace: SelectorMagnitudeTraceFragment,
    /// Proof-only identities for the actual initial-state construction.
    pub initial_state_trace: DiamondInitialStateTraceFragment,
    /// Trapdoors for the final state bases, returned for application-specific projections.
    pub final_trapdoors: TrapdoorFamily,
}

/// Matrix operations in the initial-state grid. The message conversion lives at the
/// application root and is retained separately; these roles cover only the gadget-owned
/// equation `x = selectedCarrier * initialPublic + selectedError`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiamondInitialStateTraceRole {
    TernarySecret,
    CarrierConcat,
    ZeroCarrier,
    CarrierSelect,
    GaussianError,
    ZeroError,
    ErrorSelect,
    CarrierProduct,
    InitialAdd,
}

#[derive(Clone, Debug)]
pub struct DiamondInitialStateTraceEntry {
    pub role: DiamondInitialStateTraceRole,
    pub handle: ValueHandle,
    pub operands: Vec<ValueHandle>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrozenDiamondInitialStateTraceEntry {
    pub role: DiamondInitialStateTraceRole,
    pub handle: FrozenValueRef,
    pub operands: Vec<FrozenValueRef>,
}

/// Typed family identities for the initial injector state. `base_public` is the complete
/// level-by-state family, `initial_public` fixes level zero, and `group_source` applies the
/// current-source map used by family preimage sampling. They are intentionally distinct.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiamondInitialStateTraceRefs {
    pub base_public: FrozenValueRef,
    pub initial_public: FrozenValueRef,
    pub group_source: FrozenValueRef,
    pub initial_grid: FrozenValueRef,
    pub grid: FrozenDiamondInputTargetGrid,
    pub entries: Vec<FrozenDiamondInitialStateTraceEntry>,
}

#[derive(Clone, Debug)]
pub struct DiamondInitialStateTraceFragment {
    base_public: ValueHandle,
    initial_public: ValueHandle,
    group_source: ValueHandle,
    initial_grid: ValueHandle,
    entries: Vec<DiamondInitialStateTraceEntry>,
    transport: ProofTraceTransport,
}

impl DiamondInitialStateTraceFragment {
    fn new(
        base_public: ValueHandle,
        initial_public: ValueHandle,
        group_source: ValueHandle,
        initial_grid: ValueHandle,
        entries: Vec<DiamondInitialStateTraceEntry>,
        transport: ProofTraceTransport,
    ) -> Result<Self, DslError> {
        let expected = [
            DiamondInitialStateTraceRole::TernarySecret,
            DiamondInitialStateTraceRole::CarrierConcat,
            DiamondInitialStateTraceRole::ZeroCarrier,
            DiamondInitialStateTraceRole::CarrierSelect,
            DiamondInitialStateTraceRole::GaussianError,
            DiamondInitialStateTraceRole::ZeroError,
            DiamondInitialStateTraceRole::ErrorSelect,
            DiamondInitialStateTraceRole::CarrierProduct,
            DiamondInitialStateTraceRole::InitialAdd,
        ];
        if entries.iter().map(|entry| entry.role).ne(expected) {
            return Err(DslError::Schema);
        }
        Ok(Self { base_public, initial_public, group_source, initial_grid, entries, transport })
    }

    pub fn into_retained_values(self) -> Vec<ValueHandle> {
        [self.base_public, self.initial_public, self.group_source, self.initial_grid]
            .into_iter()
            .chain(self.transport.into_retained_values())
            .collect()
    }

    pub fn resolve(
        &self,
        map: &FreezeMap,
    ) -> Result<DiamondInitialStateTraceRefs, FreezeResolveError> {
        let initial_grid = map.resolve_typed(&self.initial_grid)?;
        let (child_scope, index_slot) = match self.initial_grid.node().kind() {
            mxx_ir_core::node::NodeKind::ParallelGrid(payload) => (
                FrozenGraphScopeId::ParallelBody {
                    parent: Box::new(initial_grid.reference().scope.clone()),
                    owner: initial_grid.reference().wire.node,
                },
                *payload.index_slots.first().ok_or(FreezeResolveError::Missing)?,
            ),
            _ => return Err(FreezeResolveError::Missing),
        };
        let entries = self
            .entries
            .iter()
            .map(|entry| {
                Ok(FrozenDiamondInitialStateTraceEntry {
                    role: entry.role,
                    handle: map.resolve_typed(&self.transport.remap_handle(&entry.handle))?,
                    operands: entry
                        .operands
                        .iter()
                        .map(|operand| map.resolve_typed(&self.transport.remap_handle(operand)))
                        .collect::<Result<_, _>>()?,
                })
            })
            .collect::<Result<_, FreezeResolveError>>()?;
        Ok(DiamondInitialStateTraceRefs {
            base_public: map.resolve_typed(&self.base_public)?,
            initial_public: map.resolve_typed(&self.initial_public)?,
            group_source: map.resolve_typed(&self.group_source)?,
            initial_grid,
            grid: FrozenDiamondInputTargetGrid { child_scope, index_slot },
            entries,
        })
    }
}

/// Matrix-producing sites needed to prove that every selector has coefficient
/// infinity norm at most one. Control-only arithmetic is deliberately absent:
/// each select is bounded by its two matrix branches, independently of which
/// Boolean branch is chosen.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SelectorMagnitudeTraceRole {
    DigitSecretSample,
    SelectedSecret,
    RegularDiagonal,
    Identity,
    KDiagonal,
    InitialSelect,
    BitZero,
    BitIdentity,
    BitValueSelect,
    SecretTimesBitValue,
    SpecialTop,
    SpecialBottom,
    SpecialConcat,
    CarriedVsSpecialSelect,
}

#[derive(Clone, Debug)]
pub struct SelectorMagnitudeTraceEntry {
    pub role: SelectorMagnitudeTraceRole,
    pub handle: ValueHandle,
    /// Matrix operands whose magnitude is used by this operation's transfer rule.
    pub operands: Vec<ValueHandle>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrozenSelectorMagnitudeTraceEntry {
    pub role: SelectorMagnitudeTraceRole,
    pub handle: FrozenValueRef,
    pub operands: Vec<FrozenValueRef>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrozenSelectorMagnitudeLoop {
    pub handle: FrozenValueRef,
    pub child_scope: FrozenGraphScopeId,
    pub count: FrozenStructuralIntExpr,
    pub index_slot: u32,
}

/// Typed identities for the two nested structural computations that create a
/// selector: a grid of ternary secrets and a sequential bit scan inside the
/// transition-target grid. All references originate from the executable
/// builders that produced the corresponding runtime values.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SelectorMagnitudeTraceRefs {
    pub digit_secrets: FrozenValueRef,
    pub digit_secret_grid: FrozenDiamondInputTargetGrid,
    pub target_grid: FrozenValueRef,
    pub target_grid_info: FrozenDiamondInputTargetGrid,
    pub selector_loop: FrozenSelectorMagnitudeLoop,
    pub entries: Vec<FrozenSelectorMagnitudeTraceEntry>,
}

#[derive(Clone, Debug)]
pub struct SelectorMagnitudeTraceFragment {
    digit_secrets: ValueHandle,
    target_grid: ValueHandle,
    selector_loop: ValueHandle,
    entries: Vec<SelectorMagnitudeTraceEntry>,
    transport: ProofTraceTransport,
}

impl SelectorMagnitudeTraceFragment {
    fn new(
        digit_secrets: ValueHandle,
        target_grid: ValueHandle,
        selector_loop: ValueHandle,
        entries: Vec<SelectorMagnitudeTraceEntry>,
        transport: ProofTraceTransport,
    ) -> Result<Self, DslError> {
        let expected = [
            SelectorMagnitudeTraceRole::DigitSecretSample,
            SelectorMagnitudeTraceRole::SelectedSecret,
            SelectorMagnitudeTraceRole::RegularDiagonal,
            SelectorMagnitudeTraceRole::Identity,
            SelectorMagnitudeTraceRole::KDiagonal,
            SelectorMagnitudeTraceRole::InitialSelect,
            SelectorMagnitudeTraceRole::BitZero,
            SelectorMagnitudeTraceRole::BitIdentity,
            SelectorMagnitudeTraceRole::BitValueSelect,
            SelectorMagnitudeTraceRole::SecretTimesBitValue,
            SelectorMagnitudeTraceRole::SpecialTop,
            SelectorMagnitudeTraceRole::SpecialBottom,
            SelectorMagnitudeTraceRole::SpecialConcat,
            SelectorMagnitudeTraceRole::CarriedVsSpecialSelect,
        ];
        if entries.iter().map(|entry| entry.role).ne(expected) {
            return Err(DslError::Schema);
        }
        Ok(Self { digit_secrets, target_grid, selector_loop, entries, transport })
    }

    pub fn into_retained_values(self) -> Vec<ValueHandle> {
        self.transport.into_retained_values()
    }

    pub fn resolve(
        &self,
        map: &FreezeMap,
    ) -> Result<SelectorMagnitudeTraceRefs, FreezeResolveError> {
        let digit_secrets = map.resolve_typed(&self.digit_secrets)?;
        let target_grid = map.resolve_typed(&self.target_grid)?;
        let selector_loop_handle = self.transport.remap_handle(&self.selector_loop);
        let selector_loop = map.resolve_typed(&selector_loop_handle)?;
        let grid_info = |handle: &ValueHandle, frozen: &FrozenValueRef| match handle.node().kind() {
            mxx_ir_core::node::NodeKind::ParallelGrid(payload) => {
                Ok(FrozenDiamondInputTargetGrid {
                    child_scope: FrozenGraphScopeId::ParallelBody {
                        parent: Box::new(frozen.reference().scope.clone()),
                        owner: frozen.reference().wire.node,
                    },
                    index_slot: *payload.index_slots.first().ok_or(FreezeResolveError::Missing)?,
                })
            }
            _ => Err(FreezeResolveError::Missing),
        };
        let selector_loop_info = match selector_loop_handle.node().kind() {
            mxx_ir_core::node::NodeKind::SequentialLoop(payload) => {
                let child_scope = FrozenGraphScopeId::SequentialBody {
                    parent: Box::new(selector_loop.reference().scope.clone()),
                    owner: selector_loop.reference().wire.node,
                };
                FrozenSelectorMagnitudeLoop {
                    handle: selector_loop,
                    child_scope,
                    count: map.freeze_structural_expr(payload.count.clone()),
                    index_slot: payload.index_slot,
                }
            }
            _ => return Err(FreezeResolveError::Missing),
        };
        let entries = self
            .entries
            .iter()
            .map(|entry| {
                Ok(FrozenSelectorMagnitudeTraceEntry {
                    role: entry.role,
                    handle: map.resolve_typed(&self.transport.remap_handle(&entry.handle))?,
                    operands: entry
                        .operands
                        .iter()
                        .map(|operand| map.resolve_typed(&self.transport.remap_handle(operand)))
                        .collect::<Result<_, _>>()?,
                })
            })
            .collect::<Result<_, FreezeResolveError>>()?;
        Ok(SelectorMagnitudeTraceRefs {
            digit_secret_grid: grid_info(&self.digit_secrets, &digit_secrets)?,
            target_grid_info: grid_info(&self.target_grid, &target_grid)?,
            digit_secrets,
            target_grid,
            selector_loop: selector_loop_info,
            entries,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiamondInputTargetTraceRole {
    Selector,
    SelectorProduct,
    GaussianError,
    TargetAdd,
}

#[derive(Clone, Debug)]
pub struct DiamondInputTargetTraceEntry {
    pub role: DiamondInputTargetTraceRole,
    pub handle: ValueHandle,
    pub operands: Vec<ValueHandle>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrozenDiamondInputTargetTraceEntry {
    pub role: DiamondInputTargetTraceRole,
    pub handle: FrozenValueRef,
    pub operands: Vec<FrozenValueRef>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrozenDiamondInputTargetGrid {
    pub child_scope: FrozenGraphScopeId,
    pub index_slot: u32,
}

/// Exactly seven typed sites describe one transition target construction:
/// three family-level values and four scalar operations in the target grid.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiamondInputTargetTraceRefs {
    pub target_public: FrozenValueRef,
    pub target_grid: FrozenValueRef,
    pub target_reindex: FrozenValueRef,
    pub entries: Vec<FrozenDiamondInputTargetTraceEntry>,
    pub grid: FrozenDiamondInputTargetGrid,
}

#[derive(Clone, Debug)]
pub struct DiamondInputTargetTraceFragment {
    target_public: ValueHandle,
    target_grid: ValueHandle,
    target_reindex: ValueHandle,
    entries: Vec<DiamondInputTargetTraceEntry>,
    transport: ProofTraceTransport,
}

impl DiamondInputTargetTraceFragment {
    fn new(
        target_public: ValueHandle,
        target_grid: ValueHandle,
        target_reindex: ValueHandle,
        entries: Vec<DiamondInputTargetTraceEntry>,
        transport: ProofTraceTransport,
    ) -> Result<Self, DslError> {
        let expected = [
            DiamondInputTargetTraceRole::Selector,
            DiamondInputTargetTraceRole::SelectorProduct,
            DiamondInputTargetTraceRole::GaussianError,
            DiamondInputTargetTraceRole::TargetAdd,
        ];
        if entries.iter().map(|entry| entry.role).ne(expected) {
            return Err(DslError::Schema);
        }
        Ok(Self { target_public, target_grid, target_reindex, entries, transport })
    }

    pub fn into_retained_values(self) -> Vec<ValueHandle> {
        self.transport.into_retained_values()
    }

    pub fn resolve(
        &self,
        map: &FreezeMap,
    ) -> Result<DiamondInputTargetTraceRefs, FreezeResolveError> {
        let target_grid = map.resolve_typed(&self.target_grid)?;
        let (child_scope, index_slot) = match self.target_grid.node().kind() {
            mxx_ir_core::node::NodeKind::ParallelGrid(payload) => (
                FrozenGraphScopeId::ParallelBody {
                    parent: Box::new(target_grid.reference().scope.clone()),
                    owner: target_grid.reference().wire.node,
                },
                *payload.index_slots.first().ok_or(FreezeResolveError::Missing)?,
            ),
            _ => return Err(FreezeResolveError::Missing),
        };
        let entries = self
            .entries
            .iter()
            .map(|entry| {
                Ok(FrozenDiamondInputTargetTraceEntry {
                    role: entry.role,
                    handle: map.resolve_typed(&self.transport.remap_handle(&entry.handle))?,
                    operands: entry
                        .operands
                        .iter()
                        .map(|operand| map.resolve_typed(&self.transport.remap_handle(operand)))
                        .collect::<Result<_, _>>()?,
                })
            })
            .collect::<Result<Vec<_>, FreezeResolveError>>()?;
        Ok(DiamondInputTargetTraceRefs {
            target_public: map.resolve_typed(&self.target_public)?,
            target_grid,
            target_reindex: map.resolve_typed(&self.target_reindex)?,
            entries,
            grid: FrozenDiamondInputTargetGrid { child_scope, index_slot },
        })
    }
}

/// Online result of applying the input-selected transition matrices.
///
/// `states[0]` is the default `(s, k)` state.  The remaining entries are the
/// bit-specific states in the same order returned by
/// [`DiamondInputConfig::bit_state_index`].
pub struct DiamondInputEvaluation {
    pub states: Family<Mat>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiamondInputTraceRole {
    PackedInputDigits,
    SourceStateReindex,
    TransitionReindex,
    SelectedTransition,
    BodyApplyPreimage,
    CarriedPreviousState,
    NextStateBodyOutput,
}

#[derive(Clone, Debug)]
pub struct DiamondInputTraceEntry {
    pub role: DiamondInputTraceRole,
    pub handle: ValueHandle,
    pub operands: Vec<ValueHandle>,
    pub loop_coordinate: IntExpr,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrozenDiamondInputTraceEntry {
    pub role: DiamondInputTraceRole,
    pub handle: FrozenValueRef,
    pub operands: Vec<FrozenValueRef>,
    pub loop_coordinate: FrozenStructuralIntExpr,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrozenDiamondInputLoop {
    pub handle: FrozenValueRef,
    /// The exact `SequentialLoop.count` expression (also exposed as `range` for
    /// callers that describe the loop as a range).
    pub count: FrozenStructuralIntExpr,
    pub range: FrozenStructuralIntExpr,
    pub child_scope: FrozenGraphScopeId,
    pub index_slot: u32,
    pub carried_input_arity: usize,
    pub carried_output_arity: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiamondInputTraceRefs {
    pub entries: Vec<FrozenDiamondInputTraceEntry>,
    pub loop_info: FrozenDiamondInputLoop,
}

#[derive(Clone, Debug, Default)]
pub struct DiamondInputTraceFragment {
    entries: Vec<DiamondInputTraceEntry>,
    loop_handle: Option<ValueHandle>,
    loop_count: Option<IntExpr>,
    loop_index_slot: Option<u32>,
    transport: ProofTraceTransport,
}

impl DiamondInputTraceFragment {
    pub const SCHEMA_VERSION: u32 = 1;

    pub fn entries(&self) -> &[DiamondInputTraceEntry] {
        &self.entries
    }

    pub fn loop_handle(&self) -> &ValueHandle {
        self.loop_handle.as_ref().expect("input trace loop handle")
    }

    pub fn body_output_handle(&self) -> &ValueHandle {
        self.entries
            .iter()
            .find(|entry| entry.role == DiamondInputTraceRole::BodyApplyPreimage)
            .map(|entry| &entry.handle)
            .expect("input trace body output")
    }

    pub fn into_retained_values(self) -> Vec<ValueHandle> {
        self.transport.into_retained_values()
    }

    pub fn resolve(&self, map: &FreezeMap) -> Result<DiamondInputTraceRefs, FreezeResolveError> {
        let loop_handle = self.loop_handle();
        let loop_ref = map.resolve_typed(loop_handle)?;
        let (count, index_slot, carried_count, output_arity) = match loop_handle.node().kind() {
            mxx_ir_core::node::NodeKind::SequentialLoop(payload) => (
                payload.count.clone(),
                payload.index_slot,
                payload.carried_count,
                loop_handle.node().output_types().len(),
            ),
            _ => return Err(FreezeResolveError::Missing),
        };
        let child_scope = FrozenGraphScopeId::SequentialBody {
            parent: Box::new(loop_ref.reference().scope.clone()),
            owner: loop_ref.reference().wire.node,
        };
        let entries = self
            .entries
            .iter()
            .map(|entry| {
                Ok(FrozenDiamondInputTraceEntry {
                    role: entry.role,
                    handle: map.resolve_typed(&entry.handle)?,
                    operands: entry
                        .operands
                        .iter()
                        .map(|operand| map.resolve_typed(operand))
                        .collect::<Result<_, _>>()?,
                    loop_coordinate: map.freeze_structural_expr(entry.loop_coordinate.clone()),
                })
            })
            .collect::<Result<Vec<_>, FreezeResolveError>>()?;
        Ok(DiamondInputTraceRefs {
            entries,
            loop_info: FrozenDiamondInputLoop {
                handle: loop_ref,
                count: map.freeze_structural_expr(count.clone()),
                range: map.freeze_structural_expr(count),
                child_scope,
                index_slot,
                carried_input_arity: carried_count,
                carried_output_arity: output_arity,
            },
        })
    }

    fn from_transport(
        transport: ProofTraceTransport,
        entries: Vec<DiamondInputTraceEntry>,
        loop_handle: ValueHandle,
        loop_count: IntExpr,
        loop_index_slot: u32,
    ) -> Self {
        let entries = entries
            .into_iter()
            .map(|entry| DiamondInputTraceEntry {
                role: entry.role,
                handle: transport.remap_handle(&entry.handle),
                operands: entry
                    .operands
                    .iter()
                    .map(|operand| transport.remap_handle(operand))
                    .collect(),
                loop_coordinate: entry.loop_coordinate,
            })
            .collect();
        Self {
            entries,
            loop_handle: Some(transport.remap_handle(&loop_handle)),
            loop_count: Some(loop_count),
            loop_index_slot: Some(loop_index_slot),
            transport,
        }
    }

    fn validate_schema(&self) -> Result<(), DslError> {
        let roles = self.entries.iter().map(|entry| entry.role).collect::<Vec<_>>();
        let expected = [
            DiamondInputTraceRole::PackedInputDigits,
            DiamondInputTraceRole::SourceStateReindex,
            DiamondInputTraceRole::TransitionReindex,
            DiamondInputTraceRole::SelectedTransition,
            DiamondInputTraceRole::BodyApplyPreimage,
            DiamondInputTraceRole::CarriedPreviousState,
            DiamondInputTraceRole::NextStateBodyOutput,
        ];
        if roles != expected {
            return Err(DslError::Schema);
        }
        let arities = [2, 1, 1, 2, 2, 0, 2];
        if self.entries.iter().zip(arities).any(|(entry, arity)| entry.operands.len() != arity) {
            return Err(DslError::Schema);
        }
        if self.loop_handle.is_none() || self.loop_count.is_none() || self.loop_index_slot.is_none()
        {
            return Err(DslError::Schema);
        }
        Ok(())
    }
}

pub struct DiamondInputEvaluationWithTrace {
    pub states: Family<Mat>,
    pub trace: DiamondInputTraceFragment,
}

#[derive(Clone)]
pub struct DiamondInputInjector {
    pub params: DiamondInputParams,
}

impl DiamondInputConfig {
    pub fn validate(&self) -> Result<(), DiamondInputConfigError> {
        if self.modulus <= BigInt::from(0) {
            return Err(DiamondInputConfigError::InvalidModulus);
        }
        if self.ring_dimension == 0 || self.input_count == 0 || self.digit_count == 0 {
            return Err(DiamondInputConfigError::ZeroDimension);
        }
        if self.batch_bits == 0 || self.batch_bits >= usize::BITS as usize {
            return Err(DiamondInputConfigError::InvalidBatchBits);
        }
        let required_base = 1usize
            .checked_shl(self.batch_bits as u32)
            .ok_or(DiamondInputConfigError::LayoutOverflow)?;
        if self.digit_base < required_base {
            return Err(DiamondInputConfigError::InvalidDigitBase);
        }
        if self.gadget_base < BigInt::from(2) {
            return Err(DiamondInputConfigError::InvalidGadgetBase);
        }
        if self.error_max_coefficient_bound < BigInt::from(0) ||
            self.preimage_max_coefficient_bound < BigInt::from(0)
        {
            return Err(DiamondInputConfigError::InvalidSamplerBound);
        }
        self.witness_size()?;
        self.state_columns()?;
        Ok(())
    }

    pub fn ring(&self) -> Ring {
        Ring::new(self.modulus.clone(), self.ring_dimension)
    }

    pub fn witness_size(&self) -> Result<usize, DiamondInputConfigError> {
        self.input_count.checked_mul(self.batch_bits).ok_or(DiamondInputConfigError::LayoutOverflow)
    }

    pub fn state_rows(&self) -> usize {
        DIAMOND_PREFIX_DIMENSION * DIAMOND_SECRET_DIMENSION
    }

    pub fn state_columns(&self) -> Result<usize, DiamondInputConfigError> {
        self.state_rows()
            .checked_mul(
                self.digit_count.checked_add(2).ok_or(DiamondInputConfigError::LayoutOverflow)?,
            )
            .ok_or(DiamondInputConfigError::LayoutOverflow)
    }

    pub fn state_count_at_level(&self, level: usize) -> Result<usize, DiamondInputConfigError> {
        level
            .checked_mul(self.batch_bits)
            .and_then(|count| count.checked_add(1))
            .ok_or(DiamondInputConfigError::LayoutOverflow)
    }

    pub fn bit_state_index(
        &self,
        digit_index: usize,
        bit_index: usize,
    ) -> Result<usize, DiamondInputConfigError> {
        digit_index
            .checked_mul(self.batch_bits)
            .and_then(|index| index.checked_add(bit_index))
            .and_then(|index| index.checked_add(1))
            .ok_or(DiamondInputConfigError::LayoutOverflow)
    }

    pub fn gadget_base_expr(&self) -> IntExpr {
        self.gadget_base.clone().into()
    }

    pub fn digit_count_expr(&self) -> IntExpr {
        self.digit_count.into()
    }

    pub fn params(&self) -> DiamondInputParams {
        DiamondInputParams {
            modulus: self.modulus.clone().into(),
            ring_dimension: self.ring_dimension.into(),
            input_count: self.input_count.into(),
            digit_base: self.digit_base.into(),
            batch_bits: self.batch_bits.into(),
            gadget_base: self.gadget_base_expr(),
            digit_count: self.digit_count_expr(),
            trapdoor_sigma: self.trapdoor_sigma.clone(),
            error_sigma: self.error_sigma.clone(),
            error_max_coefficient_bound: self.error_max_coefficient_bound.clone().into(),
            preimage_max_coefficient_bound: self.preimage_max_coefficient_bound.clone().into(),
        }
    }
}

impl DiamondInputParams {
    pub fn ring(&self) -> Ring {
        Ring::new(self.modulus.clone(), self.ring_dimension.clone())
    }

    pub fn witness_size(&self) -> IntExpr {
        IntExpr::Mul(Box::new(self.input_count.clone()), Box::new(self.batch_bits.clone()))
            .canonicalize()
    }

    pub fn state_rows(&self) -> IntExpr {
        IntExpr::constant(DIAMOND_PREFIX_DIMENSION * DIAMOND_SECRET_DIMENSION)
    }

    pub fn state_columns(&self) -> IntExpr {
        IntExpr::Mul(
            Box::new(self.state_rows()),
            Box::new(IntExpr::Add(
                Box::new(self.digit_count.clone()),
                Box::new(IntExpr::constant(2)),
            )),
        )
        .canonicalize()
    }

    pub fn max_state_count(&self) -> IntExpr {
        IntExpr::Add(Box::new(IntExpr::constant(1)), Box::new(self.witness_size())).canonicalize()
    }
}

impl DiamondInputInjector {
    pub fn new(config: DiamondInputConfig) -> Result<Self, DiamondInputConfigError> {
        config.validate()?;
        Ok(Self { params: config.params() })
    }

    pub fn parameterized(params: DiamondInputParams) -> Self {
        Self { params }
    }

    pub fn preprocess(
        &self,
        message: Mat,
    ) -> Result<DiamondInputPreprocessing, DiamondInputPreprocessError> {
        let ring = self.params.ring();
        let state_rows = self.params.state_rows();
        let state_columns = self.params.state_columns();
        let level_count = self.params.input_count.clone();
        let digit_base = self.params.digit_base.clone();
        let batch_bits = self.params.batch_bits.clone();
        let max_state_count = self.params.max_state_count();
        let base_count = IntExpr::Mul(
            Box::new(IntExpr::Add(Box::new(level_count.clone()), Box::new(IntExpr::constant(1)))),
            Box::new(max_state_count.clone()),
        )
        .canonicalize();
        let bases = Parallel::range(base_count).map_values(|_| {
            ring.sample_trapdoor(
                state_rows.clone(),
                self.params.trapdoor_sigma.clone(),
                self.params.gadget_base.clone(),
                self.params.digit_count.clone(),
                self.params.preimage_max_coefficient_bound.clone(),
            )
        })?;

        let bases = bases.reindex(
            vec![
                IntExpr::Add(Box::new(level_count.clone()), Box::new(IntExpr::constant(1))),
                max_state_count.clone(),
            ],
            IndexMap::new([IndexExpr::Add(
                Box::new(IndexExpr::Multiply(
                    Box::new(IndexExpr::Axis(0)),
                    Box::new(IndexExpr::try_from(max_state_count.clone()).expect("state count")),
                )),
                Box::new(IndexExpr::Axis(1)),
            )]),
        )?;
        let source_state = IndexExpr::Select {
            selector: Box::new(IndexExpr::LessEqual(
                Box::new(IndexExpr::Add(
                    Box::new(IndexExpr::Multiply(
                        Box::new(IndexExpr::Axis(0)),
                        Box::new(IndexExpr::try_from(batch_bits.clone()).expect("batch bits")),
                    )),
                    Box::new(IndexExpr::Constant(1.into())),
                )),
                Box::new(IndexExpr::Axis(1)),
            )),
            branches: vec![IndexExpr::Axis(1), IndexExpr::Constant(0.into())],
        };
        let source_map = IndexMap::new([IndexExpr::Axis(0), source_state]);
        let group_trapdoors = bases
            .clone()
            .reindex(vec![level_count.clone(), max_state_count.clone()], source_map)?;
        let digit_secret_sample = std::rc::Rc::new(std::cell::RefCell::new(None));
        let digit_secret_sample_for_body = digit_secret_sample.clone();
        let (digit_secrets, digit_secret_transport) = Parallel::range(IntExpr::Mul(
            Box::new(level_count.clone()),
            Box::new(digit_base.clone()),
        ))
        .map_values_with_trace(|_| {
            let secret = ternary_secret(&ring);
            let sample = secret.value_handle().clone();
            *digit_secret_sample_for_body.borrow_mut() = Some(sample.clone());
            Ok((secret, ProofTraceTransport::select([sample])?))
        })?;
        let digit_secrets_handle = digit_secrets.value_handle().clone();
        drop(digit_secret_sample_for_body);
        let digit_secret_sample = std::rc::Rc::try_unwrap(digit_secret_sample)
            .map_err(|_| DslError::Schema)?
            .into_inner()
            .ok_or(DslError::Schema)?;
        let sigma = self.params.error_sigma.clone();
        let error_bound = self.params.error_max_coefficient_bound.clone();
        let target_count = IntExpr::Mul(
            Box::new(IntExpr::Mul(
                Box::new(level_count.clone()),
                Box::new(max_state_count.clone()),
            )),
            Box::new(digit_base.clone()),
        );
        let state_digit_count = IndexExpr::Multiply(
            Box::new(IndexExpr::try_from(max_state_count.clone()).expect("state count")),
            Box::new(IndexExpr::try_from(digit_base.clone()).expect("digit base")),
        );
        let flat = IndexExpr::Axis(0);
        let target_public = bases.public_matrices().reindex(
            vec![target_count],
            IndexMap::new([
                IndexExpr::Add(
                    Box::new(IndexExpr::Divide(
                        Box::new(flat.clone()),
                        Box::new(state_digit_count.clone()),
                    )),
                    Box::new(IndexExpr::constant(1)),
                ),
                IndexExpr::Divide(
                    Box::new(IndexExpr::Remainder(Box::new(flat), Box::new(state_digit_count))),
                    Box::new(IndexExpr::try_from(digit_base.clone()).expect("digit base")),
                ),
            ]),
        )?;
        let target_public_handle = target_public.value_handle().clone();
        let target_entries = std::rc::Rc::new(std::cell::RefCell::new(None));
        let target_entries_for_body = target_entries.clone();
        let selector_entries = std::rc::Rc::new(std::cell::RefCell::new(None));
        let selector_entries_for_body = selector_entries.clone();
        let selector_loop_handle = std::rc::Rc::new(std::cell::RefCell::new(None));
        let selector_loop_handle_for_body = selector_loop_handle.clone();
        let target_state_count = max_state_count.clone();
        let target_digit_base = digit_base.clone();
        let (target_grid, target_transport) =
            target_public.parallel_map_values_with_trace(|flat, public| {
                let flat = flat.as_int();
                let state_digit_count = Int::evaluate(IntExpr::Mul(
                    Box::new(target_state_count.clone()),
                    Box::new(target_digit_base.clone()),
                ));
                let level = flat.clone().div(state_digit_count.clone());
                let within_level = flat.rem(state_digit_count);
                let digit_base = Int::evaluate(target_digit_base.clone());
                let state = within_level.clone().div(digit_base.clone());
                let digit = within_level.rem(digit_base.clone());
                let secret_index = level.clone().mul(digit_base).add(digit.clone());
                let secret = digit_secrets.get(secret_index);
                let selected_secret_handle = secret.value_handle().clone();
                let first_new = level.mul(Int::evaluate(batch_bits.clone())).add(Int::constant(1));
                let regular = regular_selector(secret.clone());
                let regular_handle = regular.value_handle().clone();
                let k_identity = ring.identity(1);
                let identity_handle = k_identity.value_handle().clone();
                // The selected carrier s is either the regular diagonal
                // secret or the special transition carrier.  The appended
                // identity slot is the k coordinate used by the input state.
                let k = Mat::concat(ConcatAxis::Diagonal, vec![secret.clone(), k_identity]);
                let k_handle = k.value_handle().clone();
                let initial_match = state.clone().equal(Int::constant(0)).to_int();
                let selector =
                    initial_match.select(vec![regular, k]).expect("matching matrix branches");
                let initial_selector_handle = selector.value_handle().clone();
                let scan_entries = std::rc::Rc::new(std::cell::RefCell::new(None));
                let scan_entries_for_body = scan_entries.clone();
                let (selector, scan_transport) = Sequential::range(batch_bits.clone())
                    .scan_with_trace_handles(
                        selector,
                        (digit, (state, (first_new, secret))),
                        |bit, selector, (digit, (state, (first_new, secret)))| {
                            let carried_handle = selector.value_handle().clone();
                            let secret_handle = secret.value_handle().clone();
                            let extracted = digit.clone().bit(bit.expression());
                            let extracted_int = extracted.to_int();
                            let bit_zero_value = ring.zero((1, 1));
                            let bit_zero_handle = bit_zero_value.value_handle().clone();
                            let bit_one_value = ring.identity(1);
                            let bit_identity_handle = bit_one_value.value_handle().clone();
                            let bit_value =
                                extracted_int.select(vec![bit_zero_value, bit_one_value])?;
                            let bit_value_handle = bit_value.value_handle().clone();
                            let special_product = secret.clone() * bit_value;
                            let special_product_handle = special_product.value_handle().clone();
                            let special_top =
                                Mat::concat(ConcatAxis::Columns, vec![secret, special_product]);
                            let special_top_handle = special_top.value_handle().clone();
                            let special_bottom_value = ring.zero((1, 2));
                            let special_bottom_handle = special_bottom_value.value_handle().clone();
                            let special = Mat::concat(
                                ConcatAxis::Rows,
                                vec![special_top, special_bottom_value],
                            );
                            let special_handle = special.value_handle().clone();
                            let expected_state = first_new.add(bit.as_int());
                            let state_match_value = state.equal(expected_state);
                            let state_match_int = state_match_value.to_int();
                            let next = state_match_int.select(vec![selector, special])?;
                            let next_handle = next.value_handle().clone();
                            *scan_entries_for_body.borrow_mut() = Some(vec![
                                SelectorMagnitudeTraceEntry {
                                    role: SelectorMagnitudeTraceRole::BitZero,
                                    handle: bit_zero_handle.clone(),
                                    operands: Vec::new(),
                                },
                                SelectorMagnitudeTraceEntry {
                                    role: SelectorMagnitudeTraceRole::BitIdentity,
                                    handle: bit_identity_handle.clone(),
                                    operands: Vec::new(),
                                },
                                SelectorMagnitudeTraceEntry {
                                    role: SelectorMagnitudeTraceRole::BitValueSelect,
                                    handle: bit_value_handle.clone(),
                                    operands: vec![
                                        bit_zero_handle.clone(),
                                        bit_identity_handle.clone(),
                                    ],
                                },
                                SelectorMagnitudeTraceEntry {
                                    role: SelectorMagnitudeTraceRole::SecretTimesBitValue,
                                    handle: special_product_handle.clone(),
                                    operands: vec![secret_handle.clone(), bit_value_handle.clone()],
                                },
                                SelectorMagnitudeTraceEntry {
                                    role: SelectorMagnitudeTraceRole::SpecialTop,
                                    handle: special_top_handle.clone(),
                                    operands: vec![
                                        secret_handle.clone(),
                                        special_product_handle.clone(),
                                    ],
                                },
                                SelectorMagnitudeTraceEntry {
                                    role: SelectorMagnitudeTraceRole::SpecialBottom,
                                    handle: special_bottom_handle.clone(),
                                    operands: Vec::new(),
                                },
                                SelectorMagnitudeTraceEntry {
                                    role: SelectorMagnitudeTraceRole::SpecialConcat,
                                    handle: special_handle.clone(),
                                    operands: vec![
                                        special_top_handle.clone(),
                                        special_bottom_handle.clone(),
                                    ],
                                },
                                SelectorMagnitudeTraceEntry {
                                    role: SelectorMagnitudeTraceRole::CarriedVsSpecialSelect,
                                    handle: next_handle.clone(),
                                    operands: vec![carried_handle.clone(), special_handle.clone()],
                                },
                            ]);
                            let trace = ProofTraceTransport::select([
                                bit_zero_handle,
                                bit_identity_handle,
                                bit_value_handle,
                                special_product_handle,
                                special_top_handle,
                                special_bottom_handle,
                                special_handle,
                                next_handle,
                            ])?
                            .track_handles([carried_handle, secret_handle])?;
                            Ok((next, trace))
                        },
                    )
                    .expect("selector scan");
                let selector_handle = selector.value_handle().clone();
                *selector_loop_handle_for_body.borrow_mut() = Some(selector_handle.clone());
                drop(scan_entries_for_body);
                let mut all_selector_entries = vec![
                    SelectorMagnitudeTraceEntry {
                        role: SelectorMagnitudeTraceRole::SelectedSecret,
                        handle: selected_secret_handle.clone(),
                        operands: vec![digit_secrets_handle.clone()],
                    },
                    SelectorMagnitudeTraceEntry {
                        role: SelectorMagnitudeTraceRole::RegularDiagonal,
                        handle: regular_handle.clone(),
                        operands: vec![
                            selected_secret_handle.clone(),
                            selected_secret_handle.clone(),
                        ],
                    },
                    SelectorMagnitudeTraceEntry {
                        role: SelectorMagnitudeTraceRole::Identity,
                        handle: identity_handle.clone(),
                        operands: Vec::new(),
                    },
                    SelectorMagnitudeTraceEntry {
                        role: SelectorMagnitudeTraceRole::KDiagonal,
                        handle: k_handle.clone(),
                        operands: vec![selected_secret_handle.clone(), identity_handle.clone()],
                    },
                    SelectorMagnitudeTraceEntry {
                        role: SelectorMagnitudeTraceRole::InitialSelect,
                        handle: initial_selector_handle.clone(),
                        operands: vec![regular_handle.clone(), k_handle.clone()],
                    },
                ];
                all_selector_entries.extend(
                    std::rc::Rc::try_unwrap(scan_entries)
                        .map_err(|_| DslError::Schema)?
                        .into_inner()
                        .ok_or(DslError::Schema)?,
                );
                *selector_entries_for_body.borrow_mut() = Some(all_selector_entries);
                // For a sampled public base B and selector s, this is
                // b = s * B + e_b.  Its trapdoor preimage K satisfies
                // B * K = P + E, hence b * K = s * P + s * E + e_b * K;
                // the same equation covers regular, first-new, and special
                // bit transitions selected above.
                let public_handle = public.value_handle().clone();
                let selector_product_value = selector * public;
                let selector_product_handle = selector_product_value.value_handle().clone();
                let error = ring.gaussian(
                    (state_rows.clone(), state_columns.clone()),
                    sigma.clone(),
                    error_bound.clone(),
                );
                let error_handle = error.value_handle().clone();
                let target = selector_product_value + error;
                let target_handle = target.value_handle().clone();
                *target_entries_for_body.borrow_mut() = Some(vec![
                    DiamondInputTargetTraceEntry {
                        role: DiamondInputTargetTraceRole::Selector,
                        handle: selector_handle.clone(),
                        operands: Vec::new(),
                    },
                    DiamondInputTargetTraceEntry {
                        role: DiamondInputTargetTraceRole::SelectorProduct,
                        handle: selector_product_handle.clone(),
                        operands: vec![selector_handle.clone(), public_handle.clone()],
                    },
                    DiamondInputTargetTraceEntry {
                        role: DiamondInputTargetTraceRole::GaussianError,
                        handle: error_handle.clone(),
                        operands: Vec::new(),
                    },
                    DiamondInputTargetTraceEntry {
                        role: DiamondInputTargetTraceRole::TargetAdd,
                        handle: target_handle.clone(),
                        operands: vec![selector_product_handle.clone(), error_handle.clone()],
                    },
                ]);
                let target_trace = ProofTraceTransport::select([
                    selector_handle,
                    selector_product_handle,
                    error_handle,
                    target_handle,
                ])?
                .track_handles([public_handle])?;
                let initial_trace = ProofTraceTransport::select([
                    selected_secret_handle,
                    regular_handle,
                    identity_handle,
                    k_handle,
                    initial_selector_handle,
                ])?;
                Ok((
                    target,
                    ProofTraceTransport::merge([target_trace, initial_trace, scan_transport]),
                ))
            })?;
        let target_grid_handle = target_grid.value_handle().clone();
        let targets = target_grid.reindex(
            vec![level_count.clone(), max_state_count.clone(), digit_base.clone()],
            IndexMap::new([IndexExpr::Add(
                Box::new(IndexExpr::Multiply(
                    Box::new(IndexExpr::Add(
                        Box::new(IndexExpr::Multiply(
                            Box::new(IndexExpr::Axis(0)),
                            Box::new(
                                IndexExpr::try_from(max_state_count.clone()).expect("state count"),
                            ),
                        )),
                        Box::new(IndexExpr::Axis(1)),
                    )),
                    Box::new(IndexExpr::try_from(digit_base.clone()).expect("digit base")),
                )),
                Box::new(IndexExpr::Axis(2)),
            )]),
        )?;
        let target_reindex_handle = targets.value_handle().clone();
        // The closure borrows this helper clone rather than moving it. Drop the
        // now-unused clone before recovering the single owned trace entry list.
        drop(target_entries_for_body);
        drop(selector_entries_for_body);
        drop(selector_loop_handle_for_body);
        let target_entries = std::rc::Rc::try_unwrap(target_entries)
            .map_err(|_| DslError::Schema)?
            .into_inner()
            .ok_or(DslError::Schema)?;
        let selector_transport_source = target_transport.clone();
        let target_transport = target_transport
            .retain_selected(target_entries.iter().map(|entry| entry.handle.clone()));
        let target_trace = DiamondInputTargetTraceFragment::new(
            target_public_handle,
            target_grid_handle.clone(),
            target_reindex_handle,
            target_entries,
            target_transport,
        )?;
        let mut selector_entries = std::rc::Rc::try_unwrap(selector_entries)
            .map_err(|_| DslError::Schema)?
            .into_inner()
            .ok_or(DslError::Schema)?;
        selector_entries.insert(
            0,
            SelectorMagnitudeTraceEntry {
                role: SelectorMagnitudeTraceRole::DigitSecretSample,
                handle: digit_secret_sample,
                operands: Vec::new(),
            },
        );
        let selector_loop_handle = std::rc::Rc::try_unwrap(selector_loop_handle)
            .map_err(|_| DslError::Schema)?
            .into_inner()
            .ok_or(DslError::Schema)?;
        let selector_transport = selector_transport_source.retain_selected(
            selector_entries
                .iter()
                .map(|entry| entry.handle.clone())
                .chain(std::iter::once(selector_loop_handle.clone())),
        );
        let selector_magnitude_trace = SelectorMagnitudeTraceFragment::new(
            digit_secrets_handle,
            target_grid_handle,
            selector_loop_handle,
            selector_entries,
            ProofTraceTransport::merge([digit_secret_transport, selector_transport]),
        )?;
        let transitions = group_trapdoors
            .sample_preimage_branches(targets, (state_columns.clone(), state_columns.clone()))?;
        let initial_public = bases.public_matrices().reindex(
            vec![max_state_count.clone()],
            IndexMap::new([IndexExpr::constant(0), IndexExpr::Axis(0)]),
        )?;
        let initial = initial_public.parallel_map_values(|state, public| {
            let state = state.as_int();
            let secret = ternary_secret(&ring);
            // The carrier formula is b_state = c_state * B_0 + e_state with
            // c_state = [s | m].  Since is_initial is 1 exactly when state=0,
            // the [zero, c_state] and [zero, e_sample] branches select
            // b_0 = [s | m] * B_0 + e_sample at state zero; every other state
            // selects both zero branches and therefore remains exactly zero.
            let carrier = Mat::concat(ConcatAxis::Columns, vec![secret, message.clone()]);
            let is_initial = state.clone().equal(Int::constant(0)).to_int();
            let zero_carrier = Mat::concat(
                ConcatAxis::Columns,
                vec![ring.zero((1, DIAMOND_SECRET_DIMENSION)), ring.zero((1, 1))],
            );
            let carrier = is_initial
                .clone()
                .select(vec![zero_carrier, carrier])
                .expect("initial carrier branches");
            let sampled_error =
                ring.gaussian((1, state_columns.clone()), sigma.clone(), error_bound.clone());
            let error = is_initial
                .select(vec![ring.zero((1, state_columns.clone())), sampled_error])
                .expect("initial error branches");
            carrier * public + error
        })?;
        // The final trapdoor family is retained for application projections;
        // its public relation is still B*K = P+E, so later multiplication by
        // a selected state realizes the same s*P+s*E+e_b*K equation.
        let final_trapdoors = bases.reindex(
            vec![max_state_count.clone()],
            IndexMap::new([
                IndexExpr::try_from(level_count.clone()).expect("level count index"),
                IndexExpr::Axis(0),
            ]),
        )?;
        Ok(DiamondInputPreprocessing {
            initial,
            transitions,
            target_trace,
            selector_magnitude_trace,
            final_trapdoors,
        })
    }

    /// Applies the preprocessed transition matrices to one packed input.
    ///
    /// The transition layout is exactly the one returned by [`Self::preprocess`]:
    /// `[level][state][digit]`. Selection is represented by the DSL `Select`
    /// node and all independent state transitions at a level are represented by
    /// one rank-one IR parallel grid.
    pub fn evaluate(
        &self,
        initial_state: Family<Mat>,
        input_digits: Family<Int>,
        transitions: Family<mxx_dsl::Preimage>,
    ) -> Result<DiamondInputEvaluation, DiamondInputPreprocessError> {
        let states = self.evaluate_normal(initial_state, input_digits, transitions)?;
        Ok(DiamondInputEvaluation { states })
    }

    /// Evaluates the injector and returns a gadget-owned, freeze-only trace.
    /// The executable graph is the same graph produced by [`Self::evaluate`].
    pub fn evaluate_with_trace(
        &self,
        initial_state: Family<Mat>,
        input_digits: Family<Int>,
        transitions: Family<mxx_dsl::Preimage>,
    ) -> Result<DiamondInputEvaluationWithTrace, DiamondInputPreprocessError> {
        let level_count = self.params.input_count.clone();
        let digit_base = self.params.digit_base.clone();
        let batch_bits = self.params.batch_bits.clone();
        let max_state_count = self.params.max_state_count();
        if input_digits.count().canonicalize() != level_count.canonicalize() ||
            transitions.shape() !=
                &[level_count.clone(), max_state_count.clone(), digit_base.clone()]
        {
            return Err(DiamondInputConfigError::InvalidTransitionLayout.into());
        }
        let trace_specs = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let specs_for_body = trace_specs.clone();
        let (states, body_trace) = Sequential::range(level_count.clone()).scan_with_trace_handles(
            initial_state,
            (input_digits, transitions),
            |level, states, (input_digits, transitions)| {
                let loop_coordinate = level.expression();
                let level = level.as_int();
                let digit = input_digits.get(level.clone());
                let digit_handle = digit.value_handle().clone();
                let packed_inputs_handle = input_digits.value_handle().clone();
                let level_handle = level.value_handle().clone();
                specs_for_body.borrow_mut().push(DiamondInputTraceEntry {
                    role: DiamondInputTraceRole::PackedInputDigits,
                    handle: digit_handle.clone(),
                    operands: vec![packed_inputs_handle, level_handle],
                    loop_coordinate: loop_coordinate.clone(),
                });
                let source_state = IndexExpr::Select {
                    selector: Box::new(IndexExpr::LessEqual(
                        Box::new(IndexExpr::Add(
                            Box::new(IndexExpr::Multiply(
                                Box::new(IndexExpr::LoopIndex(0)),
                                Box::new(
                                    IndexExpr::try_from(batch_bits.clone()).expect("batch bits"),
                                ),
                            )),
                            Box::new(IndexExpr::Constant(1.into())),
                        )),
                        Box::new(IndexExpr::Axis(0)),
                    )),
                    branches: vec![IndexExpr::Axis(0), IndexExpr::Constant(0.into())],
                };
                let carried_handle = states.value_handle().clone();
                let source_states = states
                    .clone()
                    .reindex(vec![max_state_count.clone()], IndexMap::new([source_state]))?;
                let source_states_handle = source_states.value_handle().clone();
                specs_for_body.borrow_mut().push(DiamondInputTraceEntry {
                    role: DiamondInputTraceRole::SourceStateReindex,
                    handle: source_states_handle.clone(),
                    operands: vec![carried_handle.clone()],
                    loop_coordinate: loop_coordinate.clone(),
                });
                let level_transitions = transitions.clone().reindex(
                    vec![max_state_count.clone(), digit_base.clone()],
                    IndexMap::new([
                        IndexExpr::LoopIndex(0),
                        IndexExpr::Axis(0),
                        IndexExpr::Axis(1),
                    ]),
                )?;
                let level_transitions_handle = level_transitions.value_handle().clone();
                specs_for_body.borrow_mut().push(DiamondInputTraceEntry {
                    role: DiamondInputTraceRole::TransitionReindex,
                    handle: level_transitions_handle.clone(),
                    operands: vec![transitions.value_handle().clone()],
                    loop_coordinate: loop_coordinate.clone(),
                });
                let selected_transitions = match level_transitions.select_axis(1, digit.clone())? {
                    FamilyAxisSelection::Family(family) => family,
                    FamilyAxisSelection::Scalar(_) => return Err(DslError::Schema),
                };
                let selected_transitions_handle = selected_transitions.value_handle().clone();
                let selected_digit_handle = digit_handle;
                specs_for_body.borrow_mut().push(DiamondInputTraceEntry {
                    role: DiamondInputTraceRole::SelectedTransition,
                    handle: selected_transitions_handle.clone(),
                    operands: vec![level_transitions_handle, selected_digit_handle],
                    loop_coordinate: loop_coordinate.clone(),
                });
                // Selecting one digit chooses the corresponding preimage K;
                // applying it explicitly computes state * K and consumes the
                // transition relation rather than treating K as metadata.
                let body_specs = specs_for_body.clone();
                let body_loop_coordinate = loop_coordinate.clone();
                let (next, trace) = parallel_zip_bundle_trace(
                    (source_states, selected_transitions),
                    move |_, (source, transition)| {
                        let source_handle = source.value_handle().clone();
                        let transition_handle = transition.value_handle().clone();
                        let value = source.apply_preimage(transition);
                        body_specs.borrow_mut().push(DiamondInputTraceEntry {
                            role: DiamondInputTraceRole::BodyApplyPreimage,
                            handle: value.value_handle().clone(),
                            operands: vec![source_handle, transition_handle],
                            loop_coordinate: body_loop_coordinate.clone(),
                        });
                        Ok((
                            value.clone(),
                            ProofTraceTransport::select([value.value_handle().clone()])?,
                        ))
                    },
                )?;
                let next_handle = next.value_handle().clone();
                specs_for_body.borrow_mut().push(DiamondInputTraceEntry {
                    role: DiamondInputTraceRole::CarriedPreviousState,
                    handle: carried_handle,
                    operands: Vec::new(),
                    loop_coordinate: loop_coordinate.clone(),
                });
                specs_for_body.borrow_mut().push(DiamondInputTraceEntry {
                    role: DiamondInputTraceRole::NextStateBodyOutput,
                    handle: next_handle.clone(),
                    operands: vec![source_states_handle, selected_transitions_handle],
                    loop_coordinate,
                });
                let local_handles = vec![digit.value_handle().clone(), next_handle];
                Ok((
                    next,
                    ProofTraceTransport::merge([
                        trace,
                        ProofTraceTransport::select(local_handles)?,
                    ]),
                ))
            },
        )?;
        let loop_output = states.value_handle().clone();
        let (loop_count, loop_index_slot) = match loop_output.node().kind() {
            mxx_ir_core::node::NodeKind::SequentialLoop(payload) => {
                (payload.count.clone(), payload.index_slot)
            }
            _ => return Err(DslError::Schema.into()),
        };
        let trace = ProofTraceTransport::merge([
            body_trace,
            ProofTraceTransport::select([loop_output.clone()])?,
        ]);
        drop(specs_for_body);
        let entries =
            std::rc::Rc::try_unwrap(trace_specs).map_err(|_| DslError::Schema)?.into_inner();
        let trace = DiamondInputTraceFragment::from_transport(
            trace,
            entries,
            loop_output,
            loop_count,
            loop_index_slot,
        );
        trace.validate_schema()?;
        Ok(DiamondInputEvaluationWithTrace { states, trace })
    }

    fn evaluate_normal(
        &self,
        initial_state: Family<Mat>,
        input_digits: Family<Int>,
        transitions: Family<mxx_dsl::Preimage>,
    ) -> Result<Family<Mat>, DiamondInputPreprocessError> {
        let level_count = self.params.input_count.clone();
        let digit_base = self.params.digit_base.clone();
        let batch_bits = self.params.batch_bits.clone();
        let max_state_count = self.params.max_state_count();
        if input_digits.count().canonicalize() != level_count.canonicalize() ||
            transitions.shape() !=
                &[level_count.clone(), max_state_count.clone(), digit_base.clone()]
        {
            return Err(DiamondInputConfigError::InvalidTransitionLayout.into());
        }
        Sequential::range(level_count)
            .scan(
                initial_state,
                (input_digits, transitions),
                |level, states, (input_digits, transitions)| {
                    let level = level.as_int();
                    let digit = input_digits.get(level.clone());
                    let source_state = IndexExpr::Select {
                        selector: Box::new(IndexExpr::LessEqual(
                            Box::new(IndexExpr::Add(
                                Box::new(IndexExpr::Multiply(
                                    Box::new(IndexExpr::LoopIndex(0)),
                                    Box::new(
                                        IndexExpr::try_from(batch_bits.clone())
                                            .expect("batch bits"),
                                    ),
                                )),
                                Box::new(IndexExpr::Constant(1.into())),
                            )),
                            Box::new(IndexExpr::Axis(0)),
                        )),
                        branches: vec![IndexExpr::Axis(0), IndexExpr::Constant(0.into())],
                    };
                    let source_states = states
                        .reindex(vec![max_state_count.clone()], IndexMap::new([source_state]))?;
                    let level_transitions = transitions.clone().reindex(
                        vec![max_state_count.clone(), digit_base.clone()],
                        IndexMap::new([
                            IndexExpr::LoopIndex(0),
                            IndexExpr::Axis(0),
                            IndexExpr::Axis(1),
                        ]),
                    )?;
                    let selected_transitions = match level_transitions.select_axis(1, digit)? {
                        FamilyAxisSelection::Family(family) => family,
                        FamilyAxisSelection::Scalar(_) => return Err(DslError::Schema),
                    };
                    parallel_zip_bundle(
                        (source_states, selected_transitions),
                        |_, (source, transition)| source.apply_preimage(transition),
                    )
                },
            )
            .map_err(Into::into)
    }
}

fn ternary_secret(ring: &Ring) -> Mat {
    ring.uniform_interval((1, 1), -1, 1)
}

fn regular_selector(secret: Mat) -> Mat {
    Mat::concat(ConcatAxis::Diagonal, vec![secret.clone(), secret])
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::DslContext;
    use mxx_ir_core::{ParamEnv, node::NodeKind};

    fn config() -> DiamondInputConfig {
        DiamondInputConfig {
            modulus: BigInt::from(257),
            ring_dimension: 8,
            input_count: 2,
            digit_base: 2,
            batch_bits: 1,
            gadget_base: BigInt::from(4),
            digit_count: 2,
            trapdoor_sigma: RealExpr::from_integer(4),
            error_sigma: RealExpr::from_integer(3),
            error_max_coefficient_bound: BigInt::from(19),
            preimage_max_coefficient_bound: BigInt::from(64),
        }
    }

    #[test]
    fn preprocessing_builds_p_transitions_and_final_trapdoors() {
        let config = config();
        let ring = config.ring();
        let injector = DiamondInputInjector::new(config).unwrap();
        let preprocessing =
            injector.preprocess(ring.input("message", (1, 1))).expect("preprocessing");
        assert_eq!(
            preprocessing.transitions.shape(),
            &[IntExpr::constant(2), IntExpr::constant(3), IntExpr::constant(2)]
        );
        assert_eq!(preprocessing.final_trapdoors.count(), &IntExpr::constant(3));
        let target_trace = preprocessing.target_trace.clone();
        let selector_trace = preprocessing.selector_magnitude_trace.clone();
        let retained_trace_values = preprocessing
            .target_trace
            .into_retained_values()
            .into_iter()
            .chain(preprocessing.selector_magnitude_trace.into_retained_values())
            .collect::<Vec<_>>();

        let (built, freeze_map) = DslContext::new("diamond-input-preprocessing")
            .output("p", preprocessing.initial.get_static(vec![IndexExpr::Constant(0.into())]))
            .unwrap()
            .preimage_output(
                "transition",
                preprocessing.transitions.get_static(vec![
                    IndexExpr::Constant(1.into()),
                    IndexExpr::Constant(1.into()),
                    IndexExpr::Constant(1.into()),
                ]),
            )
            .unwrap()
            .output(
                "final-public",
                preprocessing
                    .final_trapdoors
                    .public_matrices()
                    .get_static(vec![IndexExpr::Constant(0.into())]),
            )
            .unwrap()
            .build_retaining(retained_trace_values)
            .unwrap();
        let validated = built.validate(&ParamEnv::default()).unwrap();
        let resolved_target_trace = target_trace.resolve(&freeze_map).unwrap();
        assert_eq!(resolved_target_trace.entries.len(), 4);
        assert_eq!(
            resolved_target_trace.entries.iter().map(|entry| entry.role).collect::<Vec<_>>(),
            vec![
                DiamondInputTargetTraceRole::Selector,
                DiamondInputTargetTraceRole::SelectorProduct,
                DiamondInputTargetTraceRole::GaussianError,
                DiamondInputTargetTraceRole::TargetAdd,
            ]
        );
        assert_eq!(resolved_target_trace.entries[1].operands.len(), 2);
        assert_eq!(resolved_target_trace.entries[3].operands.len(), 2);
        assert!(matches!(
            resolved_target_trace.grid.child_scope,
            FrozenGraphScopeId::ParallelBody { .. }
        ));
        let selector_reference = resolved_target_trace.entries[0].handle.reference();
        let selector_node = validated
            .source
            .scope(&selector_reference.scope)
            .and_then(|scope| scope.node(selector_reference.wire.node))
            .expect("retained selector node");
        assert!(matches!(
            selector_node.kind(),
            NodeKind::SequentialLoop(loop_spec)
                if loop_spec.count == IntExpr::constant(1) && loop_spec.carried_count == 1 &&
                    loop_spec.bindings.is_empty()
        ));
        let resolved_selector_trace = selector_trace.resolve(&freeze_map).unwrap();
        assert_eq!(resolved_selector_trace.entries.len(), 14);
        assert_eq!(
            resolved_selector_trace.entries.iter().map(|entry| entry.role).collect::<Vec<_>>(),
            vec![
                SelectorMagnitudeTraceRole::DigitSecretSample,
                SelectorMagnitudeTraceRole::SelectedSecret,
                SelectorMagnitudeTraceRole::RegularDiagonal,
                SelectorMagnitudeTraceRole::Identity,
                SelectorMagnitudeTraceRole::KDiagonal,
                SelectorMagnitudeTraceRole::InitialSelect,
                SelectorMagnitudeTraceRole::BitZero,
                SelectorMagnitudeTraceRole::BitIdentity,
                SelectorMagnitudeTraceRole::BitValueSelect,
                SelectorMagnitudeTraceRole::SecretTimesBitValue,
                SelectorMagnitudeTraceRole::SpecialTop,
                SelectorMagnitudeTraceRole::SpecialBottom,
                SelectorMagnitudeTraceRole::SpecialConcat,
                SelectorMagnitudeTraceRole::CarriedVsSpecialSelect,
            ]
        );
        assert!(matches!(
            resolved_selector_trace.digit_secret_grid.child_scope,
            FrozenGraphScopeId::ParallelBody { .. }
        ));
        assert!(matches!(
            resolved_selector_trace.target_grid_info.child_scope,
            FrozenGraphScopeId::ParallelBody { .. }
        ));
        assert!(matches!(
            resolved_selector_trace.selector_loop.child_scope,
            FrozenGraphScopeId::SequentialBody { .. }
        ));
        let selector_loop_reference = resolved_selector_trace.selector_loop.handle.reference();
        let selector_loop_scope = validated
            .source
            .scope(&selector_loop_reference.scope)
            .expect("selector-loop parent scope");
        let selector_loop_node = selector_loop_scope
            .node(selector_loop_reference.wire.node)
            .expect("selector-loop node");
        let selector_loop_arguments =
            selector_loop_scope.arguments(selector_loop_node).expect("selector-loop arguments");
        assert_eq!(
            selector_loop_arguments[4],
            resolved_selector_trace.entries[1].handle.reference().wire
        );
        let selector_body = validated
            .source
            .scope(&resolved_selector_trace.selector_loop.child_scope)
            .expect("selector-loop body");
        assert_eq!(
            resolved_selector_trace.entries[9].operands[0].reference().wire,
            selector_body.inputs()[4]
        );
        assert_eq!(
            resolved_selector_trace.entries[13].operands[0].reference().wire,
            selector_body.inputs()[0]
        );
        let sample = &resolved_selector_trace.entries[0].handle;
        let sample_node = validated
            .source
            .scope(&sample.reference().scope)
            .and_then(|scope| scope.node(sample.reference().wire.node))
            .expect("retained digit-secret sampler");
        assert!(matches!(
            sample_node.kind(),
            NodeKind::UniformIntervalSample { range, .. }
                if range.minimum == IntExpr::constant(-1) &&
                    range.maximum == IntExpr::constant(1)
        ));
        assert!(
            validated
                .source
                .scopes()
                .values()
                .flat_map(|scope| scope.nodes())
                .any(|node| matches!(node.kind(), NodeKind::FamilyPreimageSample { .. }))
        );
    }

    #[test]
    fn online_evaluation_selects_transitions_and_uses_parallel_state_updates() {
        let config = config();
        let input_count = config.input_count;
        let ring = config.ring();
        let injector = DiamondInputInjector::new(config).unwrap();
        let preprocessing =
            injector.preprocess(ring.input("message", (1, 1))).expect("preprocessing");
        let digits = Family::pack(
            (0..input_count)
                .map(|digit| ring.input(format!("digit-{digit}"), (1, 1)).extract_coefficient(0))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let evaluation = injector
            .evaluate(preprocessing.initial, digits, preprocessing.transitions)
            .expect("online evaluation");
        let graph = DslContext::new("diamond-input-online")
            .output("default-state", evaluation.states.get_static(0))
            .unwrap()
            .output("last-state", evaluation.states.get_static(2))
            .unwrap()
            .build()
            .unwrap();
        let validated = graph.validate(&ParamEnv::default()).unwrap();
        assert!(
            validated
                .source
                .scopes()
                .values()
                .flat_map(|scope| scope.nodes())
                .any(|node| matches!(node.kind(), NodeKind::ParallelGrid(_)))
        );
        assert!(
            validated
                .source
                .scopes()
                .values()
                .flat_map(|scope| scope.nodes())
                .any(|node| matches!(node.kind(), NodeKind::Select { .. }))
        );
    }
}
