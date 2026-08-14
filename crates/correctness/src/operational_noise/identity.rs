//! Stable identities and job-local compact symbols for the operational checker.
//!
//! A frozen `NodeId` is unique only inside one scope definition.  The keys in
//! this module add the top-level program and the concrete call/loop occurrence
//! path, so they remain valid when definitions are reused.  `SymbolTables` is
//! the sole owner of compact job-local IDs; no lowering cache is an identity
//! authority.

use super::analysis::MxxSort;
use crate::{ProtocolInputId, StageId};
#[cfg(test)]
use mxx_ir_core::Port;
use mxx_ir_core::{FrozenGraphScopeId, IntExpr, NodeId, WireRef};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Zero};
use std::{
    collections::{BTreeMap, HashMap, hash_map::Entry},
    hash::Hash,
};

/// The top-level executable program that owns an occurrence.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ProgramKey {
    WorkflowStage(StageId),
    Ideal,
    Requirement(u32),
    Comparator,
}

/// One owner-resolved edge on a path from a top-level program into a scope.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum OccurrenceFrame {
    Call { parent: FrozenGraphScopeId, owner: NodeId },
    ParallelLoop { parent: FrozenGraphScopeId, owner: NodeId },
    SequentialLoop { parent: FrozenGraphScopeId, owner: NodeId },
}

/// One concrete use of a reusable frozen scope definition.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct OccurrenceScope {
    pub program: ProgramKey,
    pub definition: FrozenGraphScopeId,
    pub path: Box<[OccurrenceFrame]>,
}

/// The producer and port of a value in one concrete occurrence scope.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct WireSourceKey {
    pub scope: OccurrenceScope,
    pub wire: WireRef,
}

/// A runtime loop-index binder, identified by its introducing loop owner.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BinderKey {
    pub loop_scope: OccurrenceScope,
    pub loop_node: NodeId,
    pub slot: u32,
}

/// The authoritative domain of one owner-resolved loop binder.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BinderDescriptor {
    pub key: BinderKey,
    pub minimum: BigInt,
    pub maximum: BigInt,
}

/// A graph-generated source together with the binders that introduced its
/// active coordinates.  Coordinate *values* remain children of `Atom`; this
/// key intentionally records only the owners of those coordinates.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct GraphWireSourceKey {
    pub wire: WireSourceKey,
    pub coordinate_binders: Box<[BinderKey]>,
}

/// The source of an ordinary symbolic value.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum AtomicSourceKey {
    ProtocolInput(ProtocolInputId),
    GraphWire(GraphWireSourceKey),
    /// One symbolic carried value in a sequential-loop body.  This is never a
    /// runtime sampler: the recurrence descriptor below binds it to the
    /// previous iteration's state when the bound phase evaluates the loop.
    SequentialState(SequentialStateKey),
    /// The final value of one carried output after a symbolic sequential
    /// recurrence.  Its full transition remains in `SymbolTables`.
    SequentialRecurrence {
        recurrence: SequentialRecurrenceId,
        carried_index: usize,
    },
    Sampler(SamplerDescriptorId),
}

/// A carried-state placeholder is owned by the concrete loop occurrence and
/// its carried position, not by a local body node number.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SequentialStateKey {
    pub loop_scope: OccurrenceScope,
    pub loop_node: NodeId,
    pub carried_index: usize,
}

/// Backend convention used when a coefficient is extracted as an integer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum CanonicalResidueConvention {
    Nonnegative,
    Centered,
}

/// An authoritative closed domain for an external/runtime integer atom.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct IntegerSourceDomain {
    pub minimum: BigInt,
    pub maximum: BigInt,
}

/// Closed sampler role used to construct relation provenance directly on an Atom e-class.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum AtomicRelationRole {
    Preimage,
    GadgetDecomposition,
    DecomposedHash,
    SmallGadgetDecomposition { range_proved: bool },
    SmallDecomposedHash { range_proved: bool },
}

/// The complete descriptor for an `MxxLang::Atom`.
///
/// Egg's `Analysis::make` receives only the compact `AtomicSourceId`; keeping
/// the sort beside its stable key in this one interner is therefore necessary
/// to type the atom without a lowering-side type cache.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct AtomicSourceDescriptor {
    pub key: AtomicSourceKey,
    pub sort: MxxSort,
    pub integer_domain: Option<IntegerSourceDomain>,
    pub canonical_residue_convention: Option<CanonicalResidueConvention>,
    pub relation_role: Option<AtomicRelationRole>,
}

/// The source of a trapdoor descriptor.  Trapdoors are structural lowering
/// values, rather than ordinary egg-language values, but obey the same source
/// identity rule as atoms.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum TrapdoorSourceKey {
    ProtocolInput(ProtocolInputId),
    GraphWire(GraphWireSourceKey),
}

/// An integer expression after every loop-index slot has been resolved to its
/// owning binder.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ResolvedIntExpr {
    Const(BigInt),
    Parameter(String),
    Binder(BinderKey),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    RoundDiv(Box<Self>, Box<Self>),
    Log2Ceil(Box<Self>),
}

impl ResolvedIntExpr {
    /// Replaces only parameter and constant syntax.  A `LoopIndex` has no
    /// owner outside lowering and therefore cannot be converted here.
    pub fn from_closed_expr(value: &IntExpr) -> Option<Self> {
        match value {
            IntExpr::Const(value) => Some(Self::Const(value.clone())),
            IntExpr::Var(name) => Some(Self::Parameter(name.clone())),
            IntExpr::LoopIndex(_) => None,
            IntExpr::Add(left, right) => Some(Self::Add(
                Box::new(Self::from_closed_expr(left)?),
                Box::new(Self::from_closed_expr(right)?),
            )),
            IntExpr::Sub(left, right) => Some(Self::Sub(
                Box::new(Self::from_closed_expr(left)?),
                Box::new(Self::from_closed_expr(right)?),
            )),
            IntExpr::Mul(left, right) => Some(Self::Mul(
                Box::new(Self::from_closed_expr(left)?),
                Box::new(Self::from_closed_expr(right)?),
            )),
            IntExpr::Div(left, right) => Some(Self::Div(
                Box::new(Self::from_closed_expr(left)?),
                Box::new(Self::from_closed_expr(right)?),
            )),
            IntExpr::RoundDiv(left, right) => Some(Self::RoundDiv(
                Box::new(Self::from_closed_expr(left)?),
                Box::new(Self::from_closed_expr(right)?),
            )),
            IntExpr::Log2Ceil(value) => {
                Some(Self::Log2Ceil(Box::new(Self::from_closed_expr(value)?)))
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ResolvedMatrixType {
    pub modulus: ResolvedIntExpr,
    pub ring_dimension: ResolvedIntExpr,
    pub rows: ResolvedIntExpr,
    pub columns: ResolvedIntExpr,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ResolvedIndexRange {
    pub start: ResolvedIntExpr,
    pub end: ResolvedIntExpr,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SliceSpec {
    pub rows: Option<ResolvedIndexRange>,
    pub columns: Option<ResolvedIndexRange>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Axis {
    Rows,
    Columns,
    Diagonal,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CrtSpec {
    pub plaintext_moduli: Box<[ResolvedIntExpr]>,
    pub reconstruction_coefficients: Box<[ResolvedIntExpr]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum HashTagPart {
    Literal(Box<[u8]>),
    BinaryArgument { argument: u16 },
    DecimalArgument { argument: u16 },
    U64LeArgument { argument: u16 },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct HashQuerySpec {
    pub matrix_type: ResolvedMatrixType,
    pub tag_program: Box<[HashTagPart]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum MatrixConstantValue {
    Zero,
    Identity,
    UnitRow { index: ResolvedIntExpr },
    UnitColumn { index: ResolvedIntExpr },
    Gadget { base: ResolvedIntExpr, small: bool },
    PowerOfBase { base: ResolvedIntExpr, exponent: ResolvedIntExpr },
    Rotation { exponent: ResolvedIntExpr },
    Polynomial { coefficients: Box<[ResolvedIntExpr]> },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct MatrixConstantSpec {
    pub matrix_type: ResolvedMatrixType,
    pub value: MatrixConstantValue,
}

impl MatrixConstantSpec {
    /// Computes the reviewed canonical maximum through the sole constant descriptor helper.
    /// Variants whose runtime layout is not encoded by this descriptor return no contract.
    pub fn canonical_coefficient_exclusive_upper(&self) -> Option<BigUint> {
        let ResolvedIntExpr::Const(modulus) = &self.matrix_type.modulus else { return None };
        let modulus = modulus.to_biguint()?;
        if modulus.is_zero() {
            return None;
        }
        let canonical = |value: &BigInt| canonical_nonnegative_residue(value, &modulus);
        let upper = |values: Vec<BigInt>| {
            values
                .iter()
                .map(canonical)
                .max()
                .and_then(|maximum| (maximum + BigInt::one()).to_biguint())
        };
        match &self.value {
            MatrixConstantValue::Zero => Some(BigUint::one()),
            MatrixConstantValue::Identity |
            MatrixConstantValue::UnitRow { .. } |
            MatrixConstantValue::UnitColumn { .. } => upper(vec![0.into(), 1.into()]),
            MatrixConstantValue::Gadget { .. } => None,
            MatrixConstantValue::PowerOfBase { base, exponent } => {
                let (ResolvedIntExpr::Const(base), ResolvedIntExpr::Const(exponent)) =
                    (base, exponent)
                else {
                    return None;
                };
                canonical_power_residue(base, exponent, &modulus)
                    .map(|residue| residue + BigUint::one())
            }
            MatrixConstantValue::Rotation { .. } => Some(modulus),
            MatrixConstantValue::Polynomial { coefficients } => coefficients
                .iter()
                .map(|coefficient| match coefficient {
                    ResolvedIntExpr::Const(value) => Some(value.clone()),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
                .and_then(upper),
        }
    }
}

/// Shared checker-private canonicalization used by constant analysis and, in
/// later stages, by the runtime bound evaluator. Keeping this arithmetic here
/// prevents descriptor analysis from inventing a second residue convention.
pub(crate) fn canonical_nonnegative_residue(value: &BigInt, modulus: &BigUint) -> BigInt {
    let modulus = BigInt::from(modulus.clone());
    ((value % &modulus) + &modulus) % &modulus
}

/// Computes `base^exponent mod modulus` without first allocating `base^exponent`.
///
/// The exponent bit guard is a checker resource boundary. The Stage 7 bound
/// evaluator must use this same helper when it gains matrix-constant support.
pub(crate) fn canonical_power_residue(
    base: &BigInt,
    exponent: &BigInt,
    modulus: &BigUint,
) -> Option<BigUint> {
    const MAX_EXPONENT_BITS: u64 = 4_096;
    let exponent = exponent.to_biguint()?;
    if exponent.bits() > MAX_EXPONENT_BITS || modulus.is_zero() {
        return None;
    }
    let base = canonical_nonnegative_residue(base, modulus).to_biguint()?;
    Some(base.modpow(&exponent, modulus))
}

macro_rules! compact_id {
    ($($name:ident),+ $(,)?) => {$(
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
        pub struct $name(pub u32);
    )+};
}

compact_id!(
    AtomicSourceId,
    BinderId,
    TrapdoorDescriptorId,
    MatrixConstantSpecId,
    SliceSpecId,
    HashQuerySpecId,
    CrtSpecId,
    SequentialRecurrenceId,
    SamplerDescriptorId,
);

/// A source-level sampler record retains the exact operand e-classes for a
/// later relation pass; lowering never guesses a relation from a matrix shape.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum SamplerIdentity {
    Preimage {
        source: GraphWireSourceKey,
        indices: Box<[egg::Id]>,
        public: egg::Id,
        trapdoor: TrapdoorDescriptorId,
        target: egg::Id,
        cutoff: ResolvedIntExpr,
    },
}

/// The deferred, fixed-size transition for one sequential-loop occurrence.
///
/// `initial` and `transition` are e-class terms, so the later bound phase can
/// evaluate a recurrence without replaying graph lowering or unrolling a loop.
/// The state placeholders occurring in `transition` are exactly the keys
/// formed from the same loop occurrence and `0..carried_count`.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct SequentialRecurrenceDescriptor {
    pub loop_scope: OccurrenceScope,
    pub loop_node: NodeId,
    pub count: ResolvedIntExpr,
    pub initial: Box<[egg::Id]>,
    pub transition: Box<[egg::Id]>,
    pub output_types: Box<[ResolvedMatrixType]>,
}

/// A trapdoor is a structural lowering value, not an `MxxLang` node.  Its
/// e-class fields are canonicalized by the caller with `egraph.find` before
/// semantic comparison; this interner only removes raw structural duplicates.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TrapdoorIdentity {
    pub source: TrapdoorSourceKey,
    pub indices: Box<[egg::Id]>,
    pub matrix_type: ResolvedMatrixType,
    pub public: egg::Id,
    pub sigma_bits: u64,
    pub gadget_base: ResolvedIntExpr,
    pub digit_count: ResolvedIntExpr,
    pub preimage_cutoff: ResolvedIntExpr,
}

/// A single job-local, amortized-O(1) stable-value-to-compact-ID map.
///
/// The public fields make the one owner inspectable, but callers should use
/// `intern` so an equal stable value always receives exactly one compact ID.
#[derive(Clone, Debug)]
pub struct Interner<T> {
    pub values: Vec<T>,
    pub by_value: HashMap<T, u32>,
}

impl<T> Default for Interner<T>
where
    T: Eq + Hash,
{
    fn default() -> Self {
        Self { values: Vec::new(), by_value: HashMap::new() }
    }
}

impl<T> Interner<T>
where
    T: Clone + Eq + Hash,
{
    pub fn intern(&mut self, value: T) -> u32 {
        let next = u32::try_from(self.values.len()).expect("too many operational symbols");
        match self.by_value.entry(value.clone()) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                self.values.push(value);
                entry.insert(next);
                next
            }
        }
    }

    pub fn get(&self, id: u32) -> Option<&T> {
        self.values.get(id as usize)
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// The sole owner of job-local identity descriptors for one rewrite job.
///
/// Relation provenance and integer ranges deliberately do not appear here:
/// they are analysis data owned by the e-graph, not a second identity cache.
#[derive(Clone, Debug, Default)]
pub struct SymbolTables {
    pub atomic_sources: Interner<AtomicSourceDescriptor>,
    pub binders: Interner<BinderDescriptor>,
    /// Closed parameter values used only if an intrinsic `IntParameter` node is constructed.
    /// Normal lowering emits `IntConst` after request closure.
    pub integer_parameters: BTreeMap<String, BigInt>,
    pub trapdoors: Interner<TrapdoorIdentity>,
    pub sequential_recurrences: Interner<SequentialRecurrenceDescriptor>,
    pub samplers: Interner<SamplerIdentity>,
    pub matrix_constants: Interner<MatrixConstantSpec>,
    pub slices: Interner<SliceSpec>,
    pub hash_queries: Interner<HashQuerySpec>,
    pub crts: Interner<CrtSpec>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope(program: ProgramKey) -> OccurrenceScope {
        OccurrenceScope { program, definition: FrozenGraphScopeId::Root, path: Box::new([]) }
    }

    #[test]
    fn program_key_keeps_equal_node_numbers_in_separate_stages_distinct() {
        let first = WireSourceKey {
            scope: scope(ProgramKey::WorkflowStage(StageId("encrypt".to_owned()))),
            wire: WireRef { node: NodeId(4), port: Port(0) },
        };
        let second = WireSourceKey {
            scope: scope(ProgramKey::WorkflowStage(StageId("decrypt".to_owned()))),
            wire: WireRef { node: NodeId(4), port: Port(0) },
        };

        assert_ne!(first, second);
    }

    #[test]
    fn call_and_loop_owners_make_occurrences_and_binders_distinct() {
        let definition = FrozenGraphScopeId::Subgraph { canonical_name: "shared".to_owned() };
        let first_scope = OccurrenceScope {
            program: ProgramKey::WorkflowStage(StageId("decode".to_owned())),
            definition: definition.clone(),
            path: Box::new([OccurrenceFrame::Call {
                parent: FrozenGraphScopeId::Root,
                owner: NodeId(20),
            }]),
        };
        let second_scope = OccurrenceScope {
            program: ProgramKey::WorkflowStage(StageId("decode".to_owned())),
            definition,
            path: Box::new([OccurrenceFrame::Call {
                parent: FrozenGraphScopeId::Root,
                owner: NodeId(50),
            }]),
        };
        assert_ne!(first_scope, second_scope);

        let first = BinderKey { loop_scope: first_scope, loop_node: NodeId(7), slot: 0 };
        let second = BinderKey { loop_scope: second_scope, loop_node: NodeId(7), slot: 0 };
        let nested_slot = BinderKey {
            loop_scope: scope(ProgramKey::WorkflowStage(StageId("decode".to_owned()))),
            loop_node: NodeId(7),
            slot: 1,
        };
        assert_ne!(first, second);
        assert_ne!(first, nested_slot);
    }

    #[test]
    fn same_scope_slot_zero_and_one_are_distinct_binders() {
        let loop_scope = scope(ProgramKey::WorkflowStage(StageId("decode".to_owned())));
        let slot_zero = BinderKey { loop_scope: loop_scope.clone(), loop_node: NodeId(7), slot: 0 };
        let slot_one = BinderKey { loop_scope, loop_node: NodeId(7), slot: 1 };

        assert_ne!(slot_zero, slot_one);
    }

    #[test]
    fn different_loop_owners_with_slot_zero_are_distinct_binders() {
        let loop_scope = scope(ProgramKey::WorkflowStage(StageId("decode".to_owned())));
        let outer = BinderKey { loop_scope: loop_scope.clone(), loop_node: NodeId(7), slot: 0 };
        let inner = BinderKey { loop_scope, loop_node: NodeId(19), slot: 0 };

        assert_ne!(outer, inner);
    }

    #[test]
    fn graph_wire_source_keeps_coordinate_owner_without_coordinate_value() {
        let loop_scope = scope(ProgramKey::WorkflowStage(StageId("decode".to_owned())));
        let binder = BinderKey { loop_scope: loop_scope.clone(), loop_node: NodeId(9), slot: 0 };
        let source = GraphWireSourceKey {
            wire: WireSourceKey {
                scope: loop_scope,
                wire: WireRef { node: NodeId(11), port: Port(0) },
            },
            coordinate_binders: Box::new([binder]),
        };
        assert_eq!(source.coordinate_binders.len(), 1);
    }

    #[test]
    fn protocol_input_identity_is_shared_across_programs() {
        let input = ProtocolInputId::from("hash-key");
        assert_eq!(
            AtomicSourceKey::ProtocolInput(input.clone()),
            AtomicSourceKey::ProtocolInput(input)
        );
    }

    #[test]
    fn sequential_state_and_result_keep_loop_owner_and_carried_position() {
        let loop_scope = scope(ProgramKey::WorkflowStage(StageId("decode".to_owned())));
        let first_state = AtomicSourceKey::SequentialState(SequentialStateKey {
            loop_scope: loop_scope.clone(),
            loop_node: NodeId(17),
            carried_index: 0,
        });
        let second_state = AtomicSourceKey::SequentialState(SequentialStateKey {
            loop_scope: loop_scope.clone(),
            loop_node: NodeId(17),
            carried_index: 1,
        });
        let other_loop = AtomicSourceKey::SequentialState(SequentialStateKey {
            loop_scope,
            loop_node: NodeId(18),
            carried_index: 0,
        });
        assert_ne!(first_state, second_state);
        assert_ne!(first_state, other_loop);
        assert_ne!(
            AtomicSourceKey::SequentialRecurrence {
                recurrence: SequentialRecurrenceId(3),
                carried_index: 0,
            },
            AtomicSourceKey::SequentialRecurrence {
                recurrence: SequentialRecurrenceId(3),
                carried_index: 1,
            },
        );
    }

    #[test]
    fn interner_reuses_equal_stable_values() {
        let mut interner = Interner::default();
        let first = interner.intern(AtomicSourceDescriptor {
            key: AtomicSourceKey::ProtocolInput(ProtocolInputId::from("key")),
            sort: MxxSort::Int,
            integer_domain: Some(IntegerSourceDomain { minimum: 0.into(), maximum: 7.into() }),
            canonical_residue_convention: None,
            relation_role: None,
        });
        let second = interner.intern(AtomicSourceDescriptor {
            key: AtomicSourceKey::ProtocolInput(ProtocolInputId::from("key")),
            sort: MxxSort::Int,
            integer_domain: Some(IntegerSourceDomain { minimum: 0.into(), maximum: 7.into() }),
            canonical_residue_convention: None,
            relation_role: None,
        });
        let third = interner.intern(AtomicSourceDescriptor {
            key: AtomicSourceKey::ProtocolInput(ProtocolInputId::from("other")),
            sort: MxxSort::Int,
            integer_domain: Some(IntegerSourceDomain { minimum: 0.into(), maximum: 7.into() }),
            canonical_residue_convention: None,
            relation_role: None,
        });

        assert_eq!(first, second);
        assert_ne!(first, third);
        assert_eq!(interner.len(), 2);
    }

    #[test]
    fn unresolved_loop_slot_cannot_be_mistaken_for_an_owned_expression() {
        assert!(ResolvedIntExpr::from_closed_expr(&IntExpr::LoopIndex(0)).is_none());
        assert_eq!(
            ResolvedIntExpr::from_closed_expr(&IntExpr::constant(3)),
            Some(ResolvedIntExpr::Const(BigInt::from(3)))
        );
    }

    #[test]
    fn power_constant_uses_bounded_modular_exponentiation() {
        let spec = MatrixConstantSpec {
            matrix_type: ResolvedMatrixType {
                modulus: ResolvedIntExpr::Const(17.into()),
                ring_dimension: ResolvedIntExpr::Const(1.into()),
                rows: ResolvedIntExpr::Const(1.into()),
                columns: ResolvedIntExpr::Const(1.into()),
            },
            value: MatrixConstantValue::PowerOfBase {
                base: ResolvedIntExpr::Const(3.into()),
                exponent: ResolvedIntExpr::Const(1_000_000.into()),
            },
        };
        assert_eq!(spec.canonical_coefficient_exclusive_upper(), Some(BigUint::from(2_u8)));

        let oversized = MatrixConstantSpec {
            value: MatrixConstantValue::PowerOfBase {
                base: ResolvedIntExpr::Const(3.into()),
                exponent: ResolvedIntExpr::Const(BigInt::one() << 4_096_usize),
            },
            ..spec
        };
        assert_eq!(oversized.canonical_coefficient_exclusive_upper(), None);
    }

    #[test]
    fn gadget_without_closed_layout_has_no_canonical_upper_contract() {
        let spec = MatrixConstantSpec {
            matrix_type: ResolvedMatrixType {
                modulus: ResolvedIntExpr::Const(17.into()),
                ring_dimension: ResolvedIntExpr::Const(1.into()),
                rows: ResolvedIntExpr::Const(1.into()),
                columns: ResolvedIntExpr::Const(2.into()),
            },
            value: MatrixConstantValue::Gadget {
                base: ResolvedIntExpr::Const(2.into()),
                small: false,
            },
        };
        assert_eq!(spec.canonical_coefficient_exclusive_upper(), None);
    }
}
