//! Monomial interning for the operational-noise arena.
//!
//! A monomial owns its factor lists exactly once.  The index below stores only
//! a structural digest and slots; equality is checked against the immutable
//! descriptor in the arena, so a digest collision can never merge terms.

use super::{
    arena::{
        ArenaToken, ExprArena, ExprId, ResolvedValueType, ScopeProof, ScopedExprId, ValueProgramId,
    },
    program::ProgramArena,
};
use rayon::prelude::*;
use std::{
    collections::{BTreeMap, HashMap, hash_map::DefaultHasher},
    error::Error,
    fmt,
    hash::{Hash, Hasher},
};

/// A compact identity for one interned monomial.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct MonomialId {
    arena: ArenaToken,
    slot: u32,
}

impl MonomialId {
    pub(crate) const fn new(arena: ArenaToken, slot: u32) -> Self {
        Self { arena, slot }
    }

    pub(crate) const fn arena(self) -> ArenaToken {
        self.arena
    }
}

/// A canonical monomial descriptor.  Central factors commute and are sorted;
/// ordered factors retain their non-commutative sequence.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MonomialDescriptor {
    pub central_factors: Box<[ScopedExprId]>,
    pub ordered_factors: Box<[ScopedExprId]>,
}

/// An owned descriptor derived exclusively from already validated monomials in one arena.
/// Its private fields prevent callers from turning raw scoped factors into an interning
/// capability without passing the arena's normal validation boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DerivedMonomialDescriptor {
    descriptor: MonomialDescriptor,
    hash: u64,
}

impl DerivedMonomialDescriptor {
    pub(crate) fn ordered_factors(&self) -> &[ScopedExprId] {
        &self.descriptor.ordered_factors
    }
}

/// The exact-term map deliberately stores IDs, never factor trees.
pub type TermMap<V> = BTreeMap<MonomialId, V>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MonomialError {
    ForeignExpressionArena {
        expected: ArenaToken,
        actual: ArenaToken,
    },
    ForeignScope {
        expected: ValueProgramId,
        actual: ValueProgramId,
    },
    InvalidExpression {
        id: ExprId,
    },
    InvalidScope {
        id: ValueProgramId,
    },
    InvalidMonomialId {
        expected: ArenaToken,
        actual: ArenaToken,
    },
    InvalidSlot {
        slot: u32,
    },
    CollectedMonomialId {
        slot: u32,
    },
    NotMatrix {
        factor: ScopedExprId,
        actual: ResolvedValueType,
    },
    CentralNotScalar {
        factor: ScopedExprId,
        rows: usize,
        columns: usize,
    },
    IncompatibleRing {
        factor: ScopedExprId,
    },
    IncompatibleOrderedShape {
        left: ScopedExprId,
        left_columns: usize,
        right: ScopedExprId,
        right_rows: usize,
    },
    ArenaExhausted,
}

impl fmt::Display for MonomialError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for MonomialError {}

/// Interns immutable monomial descriptors for one expression/program scope.
pub struct MonomialArena {
    token: ArenaToken,
    expression_arena: ArenaToken,
    scope: ValueProgramId,
    // Slots are monotonic identities. Sweeping replaces dead entries with tombstones; slots are
    // never compacted or reused, so surviving IDs retain their arena-qualified ordering.
    descriptors: Vec<Option<MonomialDescriptor>>,
    // Hash buckets contain slots only.  Full descriptor equality below makes
    // this collision-safe without storing a second copy of either factor list.
    buckets: Vec<HashMap<u64, MonomialBucket>>,
    central_factor_entries: u64,
    ordered_factor_entries: u64,
    occupied_descriptor_slots: u64,
    allocated_payload_since_sweep: u64,
}

const PARALLEL_DESCRIPTOR_BATCH_MIN: usize = 256;
const MONOMIAL_BUCKET_SHARDS: usize = 256;

#[derive(Clone, Debug, Eq, PartialEq)]
enum MonomialBucket {
    Single(u32),
    Collision(Vec<u32>),
}

impl MonomialBucket {
    fn slots(&self) -> &[u32] {
        match self {
            Self::Single(slot) => std::slice::from_ref(slot),
            Self::Collision(slots) => slots,
        }
    }

    fn push(&mut self, slot: u32) {
        match self {
            Self::Single(existing) => {
                *self = Self::Collision(vec![*existing, slot]);
            }
            Self::Collision(slots) => slots.push(slot),
        }
    }

    fn retain_slots(&mut self, mut retain: impl FnMut(u32) -> bool) -> bool {
        match self {
            Self::Single(slot) => retain(*slot),
            Self::Collision(slots) => {
                slots.retain(|&slot| retain(slot));
                match slots.as_slice() {
                    [] => false,
                    [slot] => {
                        *self = Self::Single(*slot);
                        true
                    }
                    _ => true,
                }
            }
        }
    }
}

fn bucket_shard(hash: u64) -> usize {
    hash as usize & (MONOMIAL_BUCKET_SHARDS - 1)
}

fn new_bucket_shards() -> Vec<HashMap<u64, MonomialBucket>> {
    (0..MONOMIAL_BUCKET_SHARDS).map(|_| HashMap::new()).collect()
}

fn insert_shard_slot(bucket: &mut HashMap<u64, MonomialBucket>, hash: u64, slot: u32) {
    match bucket.entry(hash) {
        std::collections::hash_map::Entry::Vacant(entry) => {
            entry.insert(MonomialBucket::Single(slot));
        }
        std::collections::hash_map::Entry::Occupied(mut entry) => entry.get_mut().push(slot),
    }
}

fn insert_bucket_slot(buckets: &mut [HashMap<u64, MonomialBucket>], hash: u64, slot: u32) {
    insert_shard_slot(&mut buckets[bucket_shard(hash)], hash, slot);
}

fn bucket_slots(buckets: &[HashMap<u64, MonomialBucket>], hash: u64) -> Option<&MonomialBucket> {
    buckets[bucket_shard(hash)].get(&hash)
}

struct PreparedMonomialDescriptor {
    descriptor: MonomialDescriptor,
    hash: u64,
}

struct PreparedWrappedDescriptor {
    intermediate: Option<PreparedMonomialDescriptor>,
    output: PreparedMonomialDescriptor,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct MonomialSweepReport {
    pub high_water_slots: u64,
    pub occupied_slots: u64,
    pub protected_prefix_occupied_slots: u64,
    pub reclaimed_slots: u64,
    pub reclaimed_payload_lower_bound_bytes: u64,
    pub bucket_entries: u64,
    pub occupied_central_factor_entries: u64,
    pub occupied_ordered_factor_entries: u64,
    pub occupied_factor_payload_lower_bound_bytes: u64,
    pub protected_prefix: MonomialSweepOwnerReport,
    pub value_cache: MonomialSweepOwnerReport,
    pub exact_plan: MonomialSweepOwnerReport,
    pub gadget: MonomialSweepOwnerReport,
    pub canonical_runtime: MonomialSweepOwnerReport,
    pub closed: MonomialSweepOwnerReport,
    pub suspended: MonomialSweepOwnerReport,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct MonomialSweepOwnerReport {
    pub descriptor_slots: u64,
    pub payload_lower_bound_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct MonomialOwnerCensus {
    pub allocated_descriptor_slots: u64,
    pub retained_descriptor_slots: u64,
    pub reclaimed_descriptor_slots: u64,
    pub reachable_descriptor_slots: u64,
    pub reachable_central_factor_entries: u64,
    pub reachable_ordered_factor_entries: u64,
    pub reachable_max_factor_word: u64,
    /// Descriptor and boxed-factor payload only. This deliberately excludes allocator metadata,
    /// `Vec` spare capacity, and `BTreeMap` node overhead, so it is a transparent lower bound.
    pub owned_payload_lower_bound_bytes: u64,
    pub unreachable_descriptor_slots: u64,
    pub unreachable_central_factor_entries: u64,
    pub unreachable_ordered_factor_entries: u64,
    pub unreachable_payload_lower_bound_bytes: u64,
    pub invalid_root_count: u64,
}

impl MonomialArena {
    pub fn new(
        expressions: &ExprArena,
        programs: &ProgramArena,
        scope: ValueProgramId,
    ) -> Result<Self, MonomialError> {
        programs.program(scope).map_err(|_| MonomialError::InvalidScope { id: scope })?;
        let root = programs
            .root(expressions, scope)
            .map_err(|_| MonomialError::InvalidScope { id: scope })?;
        programs
            .scoped(expressions, scope, root.expression())
            .map_err(|_| MonomialError::InvalidExpression { id: root.expression() })?;
        Ok(Self {
            token: ArenaToken::fresh(),
            expression_arena: expressions.token(),
            scope,
            descriptors: Vec::new(),
            buckets: new_bucket_shards(),
            central_factor_entries: 0,
            ordered_factor_entries: 0,
            occupied_descriptor_slots: 0,
            allocated_payload_since_sweep: 0,
        })
    }

    pub fn scope(&self) -> ValueProgramId {
        self.scope
    }

    pub(crate) fn token(&self) -> ArenaToken {
        self.token
    }

    pub fn len(&self) -> usize {
        self.descriptors.len()
    }

    pub(crate) fn occupied_len(&self) -> usize {
        usize::try_from(self.occupied_descriptor_slots).unwrap_or(usize::MAX)
    }

    pub(crate) fn allocated_payload_since_sweep(&self) -> u64 {
        self.allocated_payload_since_sweep
    }

    pub fn intern(
        &mut self,
        expressions: &ExprArena,
        programs: &ProgramArena,
        central_factors: &[ScopedExprId],
        ordered_factors: &[ScopedExprId],
    ) -> Result<MonomialId, MonomialError> {
        self.intern_with_scope(expressions, programs, None, central_factors, ordered_factors)
    }

    pub(crate) fn intern_with_proof(
        &mut self,
        expressions: &ExprArena,
        programs: &ProgramArena,
        scope_proof: &ScopeProof,
        central_factors: &[ScopedExprId],
        ordered_factors: &[ScopedExprId],
    ) -> Result<MonomialId, MonomialError> {
        self.intern_with_scope(
            expressions,
            programs,
            Some(scope_proof),
            central_factors,
            ordered_factors,
        )
    }

    fn intern_with_scope(
        &mut self,
        expressions: &ExprArena,
        programs: &ProgramArena,
        scope_proof: Option<&ScopeProof>,
        central_factors: &[ScopedExprId],
        ordered_factors: &[ScopedExprId],
    ) -> Result<MonomialId, MonomialError> {
        let mut central = central_factors.to_vec();
        central.sort_unstable();
        let central_ring =
            self.validate_factors(expressions, programs, scope_proof, &central, true)?;
        let ordered_ring =
            self.validate_factors(expressions, programs, scope_proof, ordered_factors, false)?;
        if central_ring.is_some() && ordered_ring.is_some() && central_ring != ordered_ring {
            return Err(MonomialError::IncompatibleRing { factor: ordered_factors[0] });
        }

        self.intern_descriptor(MonomialDescriptor {
            central_factors: central.into_boxed_slice(),
            ordered_factors: ordered_factors.to_vec().into_boxed_slice(),
        })
    }

    /// Combine two already validated monomials from this exact arena and program scope.
    ///
    /// This boundary is used after relation rewrites, where both descriptors are authoritative
    /// but need not be reachable from one original expression-root proof. It accepts no raw
    /// scoped factors: central multiplicities are retained and sorted canonically, while ordered
    /// multiplicities are concatenated left-to-right.
    pub(crate) fn combine_interned(
        &mut self,
        scope: ValueProgramId,
        left: MonomialId,
        right: MonomialId,
    ) -> Result<MonomialId, MonomialError> {
        if scope != self.scope {
            return Err(MonomialError::ForeignScope { expected: self.scope, actual: scope });
        }
        let (central_len, ordered_len) = {
            let left = self.descriptor(left)?;
            let right = self.descriptor(right)?;
            (
                left.central_factors
                    .len()
                    .checked_add(right.central_factors.len())
                    .ok_or(MonomialError::ArenaExhausted)?,
                left.ordered_factors
                    .len()
                    .checked_add(right.ordered_factors.len())
                    .ok_or(MonomialError::ArenaExhausted)?,
            )
        };
        let mut central = Vec::new();
        central.try_reserve_exact(central_len).map_err(|_| MonomialError::ArenaExhausted)?;
        let mut ordered = Vec::new();
        ordered.try_reserve_exact(ordered_len).map_err(|_| MonomialError::ArenaExhausted)?;
        {
            let left = self.descriptor(left)?;
            let right = self.descriptor(right)?;
            central.extend_from_slice(&left.central_factors);
            central.extend_from_slice(&right.central_factors);
            ordered.extend_from_slice(&left.ordered_factors);
            ordered.extend_from_slice(&right.ordered_factors);
        }
        central.sort_unstable();
        self.intern_descriptor(MonomialDescriptor {
            central_factors: central.into_boxed_slice(),
            ordered_factors: ordered.into_boxed_slice(),
        })
    }

    pub(crate) fn derive_product(
        &self,
        scope: ValueProgramId,
        left: MonomialId,
        right: MonomialId,
    ) -> Result<DerivedMonomialDescriptor, MonomialError> {
        if scope != self.scope {
            return Err(MonomialError::ForeignScope { expected: self.scope, actual: scope });
        }
        let left = self.descriptor(left)?;
        let right = self.descriptor(right)?;
        Self::derive_from_descriptors([left, right])
    }

    pub(crate) fn derive_gadget_splice(
        &self,
        source: &DerivedMonomialDescriptor,
        index: usize,
        input: MonomialId,
    ) -> Result<DerivedMonomialDescriptor, MonomialError> {
        let input = self.descriptor(input)?;
        let ordered = source.descriptor.ordered_factors.as_ref();
        if index.checked_add(1).is_none_or(|right| right >= ordered.len()) {
            return Err(MonomialError::ArenaExhausted);
        }
        let central_len = source
            .descriptor
            .central_factors
            .len()
            .checked_add(input.central_factors.len())
            .ok_or(MonomialError::ArenaExhausted)?;
        let ordered_len = ordered
            .len()
            .checked_sub(2)
            .and_then(|len| len.checked_add(input.ordered_factors.len()))
            .ok_or(MonomialError::ArenaExhausted)?;
        let mut central = Vec::new();
        central.try_reserve_exact(central_len).map_err(|_| MonomialError::ArenaExhausted)?;
        central.extend_from_slice(&source.descriptor.central_factors);
        central.extend_from_slice(&input.central_factors);
        central.sort_unstable();
        let mut replacement = Vec::new();
        replacement.try_reserve_exact(ordered_len).map_err(|_| MonomialError::ArenaExhausted)?;
        replacement.extend_from_slice(&ordered[..index]);
        replacement.extend_from_slice(&input.ordered_factors);
        replacement.extend_from_slice(&ordered[index + 2..]);
        let descriptor = MonomialDescriptor {
            central_factors: central.into_boxed_slice(),
            ordered_factors: replacement.into_boxed_slice(),
        };
        let hash = structural_hash(&descriptor);
        Ok(DerivedMonomialDescriptor { descriptor, hash })
    }

    pub(crate) fn intern_derived(
        &mut self,
        derived: DerivedMonomialDescriptor,
    ) -> Result<MonomialId, MonomialError> {
        self.intern_prepared_descriptor(PreparedMonomialDescriptor {
            descriptor: derived.descriptor,
            hash: derived.hash,
        })
    }

    fn derive_from_descriptors<'descriptor>(
        descriptors: impl IntoIterator<Item = &'descriptor MonomialDescriptor>,
    ) -> Result<DerivedMonomialDescriptor, MonomialError> {
        let prepared = Self::prepare_descriptor(descriptors)?;
        Ok(DerivedMonomialDescriptor { descriptor: prepared.descriptor, hash: prepared.hash })
    }

    /// Wrap many already-validated input monomials in the same optional prefix and suffix.
    ///
    /// Descriptor construction and structural hashing are read-only and therefore run in
    /// parallel for large splice batches. Interning remains ordered and single-threaded: stable
    /// monotonic IDs, collision checks, and the arena's mutation boundary are unchanged.
    pub(crate) fn combine_interned_wrapped_batch(
        &mut self,
        scope: ValueProgramId,
        prefix: Option<MonomialId>,
        inputs: &[MonomialId],
        suffix: Option<MonomialId>,
    ) -> Result<Vec<MonomialId>, MonomialError> {
        if scope != self.scope {
            return Err(MonomialError::ForeignScope { expected: self.scope, actual: scope });
        }
        if let Some(prefix) = prefix {
            self.descriptor(prefix)?;
        }
        for &input in inputs {
            self.descriptor(input)?;
        }
        if let Some(suffix) = suffix {
            self.descriptor(suffix)?;
        }
        if prefix.is_none() && suffix.is_none() {
            return Ok(inputs.to_vec());
        }

        let prepare = |&input: &MonomialId| self.prepare_wrapped_descriptor(prefix, input, suffix);
        let prepared =
            if inputs.len() >= PARALLEL_DESCRIPTOR_BATCH_MIN && rayon::current_num_threads() > 1 {
                inputs.par_iter().map(prepare).collect::<Result<Vec<_>, _>>()?
            } else {
                inputs.iter().map(prepare).collect::<Result<Vec<_>, _>>()?
            };
        prepared
            .into_iter()
            .map(|prepared| {
                if let Some(intermediate) = prepared.intermediate {
                    self.intern_prepared_descriptor(intermediate)?;
                }
                self.intern_prepared_descriptor(prepared.output)
            })
            .collect()
    }

    fn prepare_wrapped_descriptor(
        &self,
        prefix: Option<MonomialId>,
        input: MonomialId,
        suffix: Option<MonomialId>,
    ) -> Result<PreparedWrappedDescriptor, MonomialError> {
        let prefix = prefix.map(|id| self.descriptor(id)).transpose()?;
        let input = self.descriptor(input)?;
        let suffix = suffix.map(|id| self.descriptor(id)).transpose()?;
        let intermediate =
            prefix.zip(suffix).map(|(prefix, _)| Self::prepare_descriptor([prefix, input]));
        let intermediate = intermediate.transpose()?;
        let output = Self::prepare_descriptor(
            prefix.into_iter().chain(std::iter::once(input)).chain(suffix),
        )?;
        Ok(PreparedWrappedDescriptor { intermediate, output })
    }

    fn prepare_descriptor<'descriptor>(
        descriptors: impl IntoIterator<Item = &'descriptor MonomialDescriptor>,
    ) -> Result<PreparedMonomialDescriptor, MonomialError> {
        let descriptors = descriptors.into_iter().collect::<Vec<_>>();
        let central_len = descriptors.iter().try_fold(0_usize, |len, descriptor| {
            len.checked_add(descriptor.central_factors.len()).ok_or(MonomialError::ArenaExhausted)
        })?;
        let ordered_len = descriptors.iter().try_fold(0_usize, |len, descriptor| {
            len.checked_add(descriptor.ordered_factors.len()).ok_or(MonomialError::ArenaExhausted)
        })?;
        let mut central = Vec::new();
        central.try_reserve_exact(central_len).map_err(|_| MonomialError::ArenaExhausted)?;
        let mut ordered = Vec::new();
        ordered.try_reserve_exact(ordered_len).map_err(|_| MonomialError::ArenaExhausted)?;
        for descriptor in descriptors {
            central.extend_from_slice(&descriptor.central_factors);
            ordered.extend_from_slice(&descriptor.ordered_factors);
        }
        central.sort_unstable();
        let descriptor = MonomialDescriptor {
            central_factors: central.into_boxed_slice(),
            ordered_factors: ordered.into_boxed_slice(),
        };
        let hash = structural_hash(&descriptor);
        Ok(PreparedMonomialDescriptor { descriptor, hash })
    }

    /// Find an exact descriptor which has already been validated and interned in this arena.
    ///
    /// Relation fixed-point matching uses this for RHS-derived subwords which are not necessarily
    /// reachable from the original root proof. The lookup validates the non-forgeable scoped
    /// handles' program and expression-arena tokens, but never creates a descriptor or scoped ID.
    pub(crate) fn find_interned(
        &self,
        scope: ValueProgramId,
        central_factors: &[ScopedExprId],
        ordered_factors: &[ScopedExprId],
    ) -> Result<Option<MonomialId>, MonomialError> {
        if scope != self.scope {
            return Err(MonomialError::ForeignScope { expected: self.scope, actual: scope });
        }
        for factor in central_factors.iter().chain(ordered_factors) {
            if factor.expression().arena() != self.expression_arena {
                return Err(MonomialError::ForeignExpressionArena {
                    expected: self.expression_arena,
                    actual: factor.expression().arena(),
                });
            }
            if factor.program() != scope {
                return Err(MonomialError::ForeignScope {
                    expected: scope,
                    actual: factor.program(),
                });
            }
        }
        let mut central = Vec::new();
        central
            .try_reserve_exact(central_factors.len())
            .map_err(|_| MonomialError::ArenaExhausted)?;
        central.extend_from_slice(central_factors);
        central.sort_unstable();
        let mut ordered = Vec::new();
        ordered
            .try_reserve_exact(ordered_factors.len())
            .map_err(|_| MonomialError::ArenaExhausted)?;
        ordered.extend_from_slice(ordered_factors);
        let descriptor = MonomialDescriptor {
            central_factors: central.into_boxed_slice(),
            ordered_factors: ordered.into_boxed_slice(),
        };
        let hash = structural_hash(&descriptor);
        let Some(slots) = bucket_slots(&self.buckets, hash) else {
            return Ok(None);
        };
        for &slot in slots.slots() {
            if self.descriptors.get(slot as usize).and_then(Option::as_ref) == Some(&descriptor) {
                return Ok(Some(MonomialId::new(self.token, slot)));
            }
        }
        Ok(None)
    }

    fn intern_descriptor(
        &mut self,
        descriptor: MonomialDescriptor,
    ) -> Result<MonomialId, MonomialError> {
        let hash = structural_hash(&descriptor);
        self.intern_prepared_descriptor(PreparedMonomialDescriptor { descriptor, hash })
    }

    fn intern_prepared_descriptor(
        &mut self,
        prepared: PreparedMonomialDescriptor,
    ) -> Result<MonomialId, MonomialError> {
        let PreparedMonomialDescriptor { descriptor, hash } = prepared;
        if let Some(slots) = bucket_slots(&self.buckets, hash) {
            for &slot in slots.slots() {
                let Some(existing) = self.descriptors.get(slot as usize).and_then(Option::as_ref)
                else {
                    continue;
                };
                if existing == &descriptor {
                    return Ok(MonomialId::new(self.token, slot));
                }
            }
        }

        let slot =
            u32::try_from(self.descriptors.len()).map_err(|_| MonomialError::ArenaExhausted)?;
        let central_len = u64::try_from(descriptor.central_factors.len()).unwrap_or(u64::MAX);
        let ordered_len = u64::try_from(descriptor.ordered_factors.len()).unwrap_or(u64::MAX);
        self.central_factor_entries = self.central_factor_entries.saturating_add(central_len);
        self.ordered_factor_entries = self.ordered_factor_entries.saturating_add(ordered_len);
        self.occupied_descriptor_slots = self.occupied_descriptor_slots.saturating_add(1);
        self.allocated_payload_since_sweep = self
            .allocated_payload_since_sweep
            .saturating_add(descriptor_payload_lower_bound_bytes(central_len, ordered_len));
        self.descriptors.push(Some(descriptor));
        insert_bucket_slot(&mut self.buckets, hash, slot);
        Ok(MonomialId::new(self.token, slot))
    }

    pub(crate) fn owner_census(
        &self,
        roots: impl IntoIterator<Item = MonomialId>,
    ) -> MonomialOwnerCensus {
        let allocated_descriptor_slots = u64::try_from(self.descriptors.len()).unwrap_or(u64::MAX);
        let retained_descriptor_slots = self.occupied_descriptor_slots;
        let mut marked = vec![0_u64; self.descriptors.len().div_ceil(64)];
        let mut invalid_root_count = 0_u64;
        for root in roots {
            if root.arena != self.token ||
                self.descriptors.get(root.slot as usize).and_then(Option::as_ref).is_none()
            {
                invalid_root_count = invalid_root_count.saturating_add(1);
                continue;
            }
            let slot = root.slot as usize;
            marked[slot / 64] |= 1_u64 << (slot % 64);
        }
        let mut reachable_descriptor_slots = 0_u64;
        let mut reachable_central = 0_u64;
        let mut reachable_ordered = 0_u64;
        let mut max_factor_word = 0_u64;
        for (slot, descriptor) in self.descriptors.iter().enumerate() {
            let Some(descriptor) = descriptor.as_ref() else { continue };
            if marked.get(slot / 64).is_some_and(|word| word & (1_u64 << (slot % 64)) != 0) {
                reachable_descriptor_slots = reachable_descriptor_slots.saturating_add(1);
                let central = u64::try_from(descriptor.central_factors.len()).unwrap_or(u64::MAX);
                let ordered = u64::try_from(descriptor.ordered_factors.len()).unwrap_or(u64::MAX);
                reachable_central = reachable_central.saturating_add(central);
                reachable_ordered = reachable_ordered.saturating_add(ordered);
                max_factor_word = max_factor_word.max(central.saturating_add(ordered));
            }
        }
        let factor_entries = reachable_central.saturating_add(reachable_ordered);
        let descriptor_bytes = reachable_descriptor_slots.saturating_mul(
            u64::try_from(std::mem::size_of::<MonomialDescriptor>()).unwrap_or(u64::MAX),
        );
        let factor_bytes = factor_entries
            .saturating_mul(u64::try_from(std::mem::size_of::<ScopedExprId>()).unwrap_or(u64::MAX));
        MonomialOwnerCensus {
            allocated_descriptor_slots,
            retained_descriptor_slots,
            reclaimed_descriptor_slots: allocated_descriptor_slots
                .saturating_sub(retained_descriptor_slots),
            reachable_descriptor_slots,
            reachable_central_factor_entries: reachable_central,
            reachable_ordered_factor_entries: reachable_ordered,
            reachable_max_factor_word: max_factor_word,
            owned_payload_lower_bound_bytes: descriptor_bytes.saturating_add(factor_bytes),
            unreachable_descriptor_slots: retained_descriptor_slots
                .saturating_sub(reachable_descriptor_slots),
            unreachable_central_factor_entries: self
                .central_factor_entries
                .saturating_sub(reachable_central),
            unreachable_ordered_factor_entries: self
                .ordered_factor_entries
                .saturating_sub(reachable_ordered),
            unreachable_payload_lower_bound_bytes: retained_descriptor_slots
                .saturating_sub(reachable_descriptor_slots)
                .saturating_mul(
                    u64::try_from(std::mem::size_of::<MonomialDescriptor>()).unwrap_or(u64::MAX),
                )
                .saturating_add(
                    self.central_factor_entries
                        .saturating_add(self.ordered_factor_entries)
                        .saturating_sub(factor_entries)
                        .saturating_mul(
                            u64::try_from(std::mem::size_of::<ScopedExprId>()).unwrap_or(u64::MAX),
                        ),
                ),
            invalid_root_count,
        }
    }

    pub fn descriptor(&self, id: MonomialId) -> Result<&MonomialDescriptor, MonomialError> {
        if id.arena != self.token {
            return Err(MonomialError::InvalidMonomialId { expected: self.token, actual: id.arena });
        }
        self.descriptors
            .get(id.slot as usize)
            .ok_or(MonomialError::InvalidSlot { slot: id.slot })?
            .as_ref()
            .ok_or(MonomialError::CollectedMonomialId { slot: id.slot })
    }

    pub(crate) fn descriptor_payload_lower_bound_bytes(
        &self,
        id: MonomialId,
    ) -> Result<u64, MonomialError> {
        let descriptor = self.descriptor(id)?;
        Ok(descriptor_payload_lower_bound_bytes(
            u64::try_from(descriptor.central_factors.len()).unwrap_or(u64::MAX),
            u64::try_from(descriptor.ordered_factors.len()).unwrap_or(u64::MAX),
        ))
    }

    /// Collect unrooted descriptors without changing any surviving or future slot identity.
    /// Every supplied root is authoritative: foreign, out-of-range, or already-collected roots
    /// fail closed before the arena is mutated.
    #[cfg(test)]
    pub(crate) fn sweep(
        &mut self,
        protected_prefix: usize,
        roots: impl IntoIterator<Item = MonomialId>,
    ) -> Result<MonomialSweepReport, MonomialError> {
        self.sweep_with_owners(
            protected_prefix,
            roots,
            std::iter::empty(),
            std::iter::empty(),
            std::iter::empty(),
            std::iter::empty(),
            std::iter::empty(),
        )
    }

    pub(crate) fn sweep_with_owners(
        &mut self,
        protected_prefix: usize,
        value_cache_roots: impl IntoIterator<Item = MonomialId>,
        exact_plan_roots: impl IntoIterator<Item = MonomialId>,
        gadget_roots: impl IntoIterator<Item = MonomialId>,
        canonical_runtime_roots: impl IntoIterator<Item = MonomialId>,
        closed_roots: impl IntoIterator<Item = MonomialId>,
        suspended_roots: impl IntoIterator<Item = MonomialId>,
    ) -> Result<MonomialSweepReport, MonomialError> {
        let high_water = self.descriptors.len();
        let protected_prefix = protected_prefix.min(high_water);
        let mut marked = vec![0_u64; high_water.div_ceil(64)];
        let mut protected_report = MonomialSweepOwnerReport::default();
        for slot in 0..protected_prefix {
            let Some(descriptor) = self.descriptors[slot].as_ref() else { continue };
            marked[slot / 64] |= 1_u64 << (slot % 64);
            protected_report.descriptor_slots = protected_report.descriptor_slots.saturating_add(1);
            protected_report.payload_lower_bound_bytes = protected_report
                .payload_lower_bound_bytes
                .saturating_add(descriptor_payload_lower_bound_bytes(
                    u64::try_from(descriptor.central_factors.len()).unwrap_or(u64::MAX),
                    u64::try_from(descriptor.ordered_factors.len()).unwrap_or(u64::MAX),
                ));
        }
        let value_cache = self.mark_sweep_owner(&mut marked, value_cache_roots.into_iter())?;
        let exact_plan = self.mark_sweep_owner(&mut marked, exact_plan_roots.into_iter())?;
        let gadget = self.mark_sweep_owner(&mut marked, gadget_roots.into_iter())?;
        let canonical_runtime =
            self.mark_sweep_owner(&mut marked, canonical_runtime_roots.into_iter())?;
        let closed = self.mark_sweep_owner(&mut marked, closed_roots.into_iter())?;
        let suspended = self.mark_sweep_owner(&mut marked, suspended_roots.into_iter())?;

        // Descriptor slots are monotonic and therefore naturally ordered. Even with many
        // tombstones, this contiguous scan is substantially faster than following the hash
        // index's random slot order and needs no proportional temporary slot vector. Reclaiming
        // independent slots is parallel: large factor boxes are dropped by workers, then only
        // aggregate counters are updated at the arena mutation boundary.
        let reclaim_slot = |(slot, entry): (usize, &mut Option<MonomialDescriptor>)| {
            if marked.get(slot / 64).is_some_and(|word| word & (1_u64 << (slot % 64)) != 0) {
                return (0_u64, 0_u64, 0_u64, 0_u64);
            }
            let Some(descriptor) = entry.take() else {
                return (0_u64, 0_u64, 0_u64, 0_u64);
            };
            let central = u64::try_from(descriptor.central_factors.len()).unwrap_or(u64::MAX);
            let ordered = u64::try_from(descriptor.ordered_factors.len()).unwrap_or(u64::MAX);
            (1, central, ordered, descriptor_payload_lower_bound_bytes(central, ordered))
        };
        let combine = |left: (u64, u64, u64, u64), right: (u64, u64, u64, u64)| {
            (
                left.0.saturating_add(right.0),
                left.1.saturating_add(right.1),
                left.2.saturating_add(right.2),
                left.3.saturating_add(right.3),
            )
        };
        let (reclaimed_slots, reclaimed_central, reclaimed_ordered, reclaimed_payload) = if self
            .occupied_descriptor_slots
            as usize >=
            PARALLEL_DESCRIPTOR_BATCH_MIN &&
            rayon::current_num_threads() > 1
        {
            self.descriptors
                .par_iter_mut()
                .enumerate()
                .map(reclaim_slot)
                .reduce(|| (0, 0, 0, 0), combine)
        } else {
            self.descriptors.iter_mut().enumerate().map(reclaim_slot).fold((0, 0, 0, 0), combine)
        };
        self.central_factor_entries = self.central_factor_entries.saturating_sub(reclaimed_central);
        self.ordered_factor_entries = self.ordered_factor_entries.saturating_sub(reclaimed_ordered);
        self.occupied_descriptor_slots =
            self.occupied_descriptor_slots.saturating_sub(reclaimed_slots);

        if self
            .descriptors
            .len()
            .checked_sub(1)
            .is_some_and(|last_slot| u32::try_from(last_slot).is_err())
        {
            return Err(MonomialError::ArenaExhausted);
        }
        // Descriptors are immutable, so their structural hash never changes. Prune the existing
        // authoritative index in place for both dense and sparse sweeps instead of rescanning
        // every surviving factor word and rebuilding fresh HashMaps. This retains the dense
        // descriptor walk's cache locality while eliminating the dominant high-live-set cost.
        // Collision buckets collapse back to their compact representation at one live slot.
        let descriptors = &self.descriptors;
        let prune_shard = |shard: &mut HashMap<u64, MonomialBucket>| {
            shard.retain(|_, bucket| {
                bucket.retain_slots(|slot| {
                    descriptors.get(slot as usize).is_some_and(Option::is_some)
                })
            });
        };
        if self.occupied_descriptor_slots as usize >= PARALLEL_DESCRIPTOR_BATCH_MIN &&
            rayon::current_num_threads() > 1
        {
            self.buckets.par_iter_mut().for_each(prune_shard);
        } else {
            self.buckets.iter_mut().for_each(prune_shard);
        }
        self.allocated_payload_since_sweep = 0;
        Ok(MonomialSweepReport {
            high_water_slots: u64::try_from(high_water).unwrap_or(u64::MAX),
            occupied_slots: self.occupied_descriptor_slots,
            protected_prefix_occupied_slots: protected_report.descriptor_slots,
            reclaimed_slots,
            reclaimed_payload_lower_bound_bytes: reclaimed_payload,
            bucket_entries: self.occupied_descriptor_slots,
            occupied_central_factor_entries: self.central_factor_entries,
            occupied_ordered_factor_entries: self.ordered_factor_entries,
            occupied_factor_payload_lower_bound_bytes: self
                .central_factor_entries
                .saturating_add(self.ordered_factor_entries)
                .saturating_mul(
                    u64::try_from(std::mem::size_of::<ScopedExprId>()).unwrap_or(u64::MAX),
                ),
            protected_prefix: protected_report,
            value_cache,
            exact_plan,
            gadget,
            canonical_runtime,
            closed,
            suspended,
        })
    }

    fn mark_sweep_owner(
        &self,
        marked: &mut [u64],
        roots: impl Iterator<Item = MonomialId>,
    ) -> Result<MonomialSweepOwnerReport, MonomialError> {
        let mut report = MonomialSweepOwnerReport::default();
        for root in roots {
            if root.arena != self.token {
                return Err(MonomialError::InvalidMonomialId {
                    expected: self.token,
                    actual: root.arena,
                });
            }
            let Some(entry) = self.descriptors.get(root.slot as usize) else {
                return Err(MonomialError::InvalidSlot { slot: root.slot });
            };
            let Some(descriptor) = entry.as_ref() else {
                return Err(MonomialError::CollectedMonomialId { slot: root.slot });
            };
            let slot = root.slot as usize;
            let mask = 1_u64 << (slot % 64);
            if marked[slot / 64] & mask != 0 {
                continue;
            }
            marked[slot / 64] |= mask;
            report.descriptor_slots = report.descriptor_slots.saturating_add(1);
            report.payload_lower_bound_bytes = report.payload_lower_bound_bytes.saturating_add(
                descriptor_payload_lower_bound_bytes(
                    u64::try_from(descriptor.central_factors.len()).unwrap_or(u64::MAX),
                    u64::try_from(descriptor.ordered_factors.len()).unwrap_or(u64::MAX),
                ),
            );
        }
        Ok(report)
    }

    fn validate_factors(
        &self,
        expressions: &ExprArena,
        programs: &ProgramArena,
        scope_proof: Option<&ScopeProof>,
        factors: &[ScopedExprId],
        central: bool,
    ) -> Result<Option<(num_bigint::BigUint, usize)>, MonomialError> {
        let mut ring = None;
        let mut previous = None;
        for &factor in factors {
            let program = factor.program();
            if program != self.scope {
                return Err(MonomialError::ForeignScope { expected: self.scope, actual: program });
            }
            let expression = factor.expression();
            if expression.arena() != self.expression_arena {
                return Err(MonomialError::ForeignExpressionArena {
                    expected: self.expression_arena,
                    actual: expression.arena(),
                });
            }
            if let Some(proof) = scope_proof {
                expressions
                    .validate_scoped_from_proof(proof, factor)
                    .map_err(|_| MonomialError::InvalidExpression { id: expression })?;
            } else {
                // Relation RHS terms can expose a factor that was not reachable from the
                // original root. Validate that factor as its own scoped root; this still checks
                // the finalized program signature and all free-argument types, while avoiding
                // the incorrect requirement that every RHS-derived expression be in the root
                // DAG. `programs.scoped` cannot be used here because it intentionally requires
                // root reachability.
                programs
                    .validate_detached_expression(expressions, program, expression)
                    .map_err(|_| MonomialError::InvalidExpression { id: expression })?;
                let factor_proof = expressions
                    .scope_proof(program, expression)
                    .map_err(|_| MonomialError::InvalidExpression { id: expression })?;
                expressions
                    .scoped_from_proof(&factor_proof, expression)
                    .map_err(|_| MonomialError::InvalidExpression { id: expression })?;
            }
            let value_type = expressions
                .value_type(expression)
                .map_err(|_| MonomialError::InvalidExpression { id: expression })?;
            let ResolvedValueType::Matrix(matrix) = value_type else {
                return Err(MonomialError::NotMatrix { factor, actual: value_type.clone() });
            };
            if central && (matrix.rows != 1 || matrix.columns != 1) {
                return Err(MonomialError::CentralNotScalar {
                    factor,
                    rows: matrix.rows,
                    columns: matrix.columns,
                });
            }
            if let Some((modulus, ring_dimension)) = &ring {
                if matrix.modulus != *modulus || matrix.ring_dimension != *ring_dimension {
                    return Err(MonomialError::IncompatibleRing { factor });
                }
            } else {
                ring = Some((matrix.modulus.clone(), matrix.ring_dimension));
            }
            if !central {
                if let Some((left, left_columns)) = previous {
                    if left_columns != matrix.rows {
                        return Err(MonomialError::IncompatibleOrderedShape {
                            left,
                            left_columns,
                            right: factor,
                            right_rows: matrix.rows,
                        });
                    }
                }
                previous = Some((factor, matrix.columns));
            }
        }
        Ok(ring)
    }
}

fn structural_hash(descriptor: &MonomialDescriptor) -> u64 {
    let mut hasher = DefaultHasher::new();
    descriptor.central_factors.len().hash(&mut hasher);
    descriptor.central_factors.hash(&mut hasher);
    descriptor.ordered_factors.len().hash(&mut hasher);
    descriptor.ordered_factors.hash(&mut hasher);
    hasher.finish()
}

fn descriptor_payload_lower_bound_bytes(central: u64, ordered: u64) -> u64 {
    u64::try_from(std::mem::size_of::<MonomialDescriptor>()).unwrap_or(u64::MAX).saturating_add(
        central
            .saturating_add(ordered)
            .saturating_mul(u64::try_from(std::mem::size_of::<ScopedExprId>()).unwrap_or(u64::MAX)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::arena::{
        MatrixOperation, ProgramInput, ProgramSignature, ResolvedMatrixType, ValueOperator,
    };
    use num_bigint::BigUint;

    fn matrix(rows: usize, columns: usize, modulus: u64) -> ResolvedValueType {
        ResolvedValueType::Matrix(
            ResolvedMatrixType::new(BigUint::from(modulus), 1, rows, columns).unwrap(),
        )
    }

    fn fixture()
    -> (ExprArena, ProgramArena, ValueProgramId, ScopedExprId, ScopedExprId, ScopedExprId) {
        let mut expressions = ExprArena::new();
        let one = expressions.intern_argument(0, matrix(1, 1, 17)).unwrap();
        let wide = expressions.intern_argument(1, matrix(2, 2, 17)).unwrap();
        let scalar = expressions.intern_argument(2, ResolvedValueType::Int).unwrap();
        let wide_element = expressions
            .intern(
                ValueOperator::Matrix(MatrixOperation::Slice {
                    row_start: 0,
                    row_end_exclusive: 1,
                    column_start: 0,
                    column_end_exclusive: 1,
                    layout: super::super::arena::MatrixLayout::row_major(1, 1),
                }),
                Box::new([wide]),
            )
            .unwrap();
        let scaled = expressions
            .intern(ValueOperator::Matrix(MatrixOperation::Scale), Box::new([one, scalar]))
            .unwrap();
        let root = expressions
            .intern(ValueOperator::Matrix(MatrixOperation::Add), Box::new([wide_element, scaled]))
            .unwrap();
        let signature = ProgramSignature {
            inputs: Box::new([
                ProgramInput { value_type: matrix(1, 1, 17), trusted_index_range: None },
                ProgramInput { value_type: matrix(2, 2, 17), trusted_index_range: None },
                ProgramInput { value_type: ResolvedValueType::Int, trusted_index_range: None },
            ]),
            output: matrix(1, 1, 17),
        };
        let mut programs = ProgramArena::new();
        let scope = programs.finalize(&mut expressions, signature, root).unwrap();
        let one = programs.scoped(&expressions, scope, one).unwrap();
        let wide = programs.scoped(&expressions, scope, wide).unwrap();
        let scalar = programs.scoped(&expressions, scope, scalar).unwrap();
        (expressions, programs, scope, one, wide, scalar)
    }

    #[test]
    fn detached_rhs_factor_is_validated_without_root_reachability() {
        let (mut expressions, programs, scope, one, _, _) = fixture();
        let detached = expressions
            .intern(ValueOperator::Matrix(MatrixOperation::Negate), Box::new([one.expression()]))
            .unwrap();
        let proof = expressions.scope_proof(scope, detached).unwrap();
        let detached = expressions.scoped_from_proof(&proof, detached).unwrap();
        let mut arena = MonomialArena::new(&expressions, &programs, scope).unwrap();
        assert!(arena.intern(&expressions, &programs, &[], &[detached]).is_ok());
    }

    #[test]
    fn detached_factor_rejects_a_foreign_program_call() {
        let (mut expressions, programs, scope, one, _, _) = fixture();
        let mut foreign_programs = ProgramArena::new();
        let foreign_scope = foreign_programs
            .finalize(
                &mut expressions,
                ProgramSignature {
                    inputs: Box::new([ProgramInput {
                        value_type: matrix(1, 1, 17),
                        trusted_index_range: None,
                    }]),
                    output: matrix(1, 1, 17),
                },
                one.expression(),
            )
            .unwrap();
        let foreign_call = expressions
            .intern(
                ValueOperator::ProgramCall { program: foreign_scope },
                Box::new([one.expression()]),
            )
            .unwrap();
        let proof = expressions.scope_proof(scope, foreign_call).unwrap();
        let foreign_call = expressions.scoped_from_proof(&proof, foreign_call).unwrap();
        let mut arena = MonomialArena::new(&expressions, &programs, scope).unwrap();
        assert!(matches!(
            arena.intern(&expressions, &programs, &[], &[foreign_call]),
            Err(MonomialError::InvalidExpression { id }) if id == foreign_call.expression()
        ));
    }

    #[test]
    fn reuses_canonical_descriptor_and_sorts_central_factors() {
        let (expressions, programs, scope, one, _, _) = fixture();
        let mut arena = MonomialArena::new(&expressions, &programs, scope).unwrap();
        let root = programs.root(&expressions, scope).unwrap();
        let first = arena.intern(&expressions, &programs, &[root, one], &[one]).unwrap();
        let second = arena.intern(&expressions, &programs, &[one, root], &[one]).unwrap();
        assert_eq!(first, second);
        assert_eq!(arena.len(), 1);
        let factors = arena.descriptor(first).unwrap().central_factors.as_ref();
        assert_eq!(factors.len(), 2);
        assert!(factors[0] <= factors[1]);
    }

    #[test]
    fn owner_census_tracks_append_only_descriptor_and_factor_payloads() {
        let (expressions, programs, scope, one, _, _) = fixture();
        let mut arena = MonomialArena::new(&expressions, &programs, scope).unwrap();
        let root = programs.root(&expressions, scope).unwrap();
        let first = arena.intern(&expressions, &programs, &[root, one], &[one]).unwrap();
        // A byte-identical hit owns no additional descriptor or factor payload.
        let hit = arena.intern(&expressions, &programs, &[one, root], &[one]).unwrap();
        let second = arena.intern(&expressions, &programs, &[], &[root, one]).unwrap();
        assert_eq!(first, hit);

        // Repeated references are deliberately non-additive in the exact reachable union.
        let census = arena.owner_census([first, hit, second]);
        assert_eq!(census.retained_descriptor_slots, 2);
        assert_eq!(census.reachable_descriptor_slots, 2);
        assert_eq!(census.reachable_central_factor_entries, 2);
        assert_eq!(census.reachable_ordered_factor_entries, 3);
        assert_eq!(census.reachable_max_factor_word, 3);
        assert_eq!(
            census.owned_payload_lower_bound_bytes,
            2 * u64::try_from(std::mem::size_of::<MonomialDescriptor>()).unwrap() +
                5 * u64::try_from(std::mem::size_of::<ScopedExprId>()).unwrap()
        );

        let released = arena.owner_census(std::iter::empty());
        assert_eq!(released.retained_descriptor_slots, 2);
        assert_eq!(released.reachable_descriptor_slots, 0);
        assert_eq!(released.unreachable_descriptor_slots, 2);
        assert_eq!(released.unreachable_central_factor_entries, 2);
        assert_eq!(released.unreachable_ordered_factor_entries, 3);
        let invalid = arena.owner_census([MonomialId::new(ArenaToken::fresh(), 0)]);
        assert_eq!(invalid.invalid_root_count, 1);
    }

    #[test]
    fn sweep_tombstones_dead_slots_and_reinterns_at_fresh_monotonic_slot() {
        let (expressions, programs, scope, one, _, _) = fixture();
        let mut arena = MonomialArena::new(&expressions, &programs, scope).unwrap();
        let root = programs.root(&expressions, scope).unwrap();
        let live = arena.intern(&expressions, &programs, &[one], &[root]).unwrap();
        let dead = arena.intern(&expressions, &programs, &[], &[root, one]).unwrap();
        assert!(arena.allocated_payload_since_sweep() > 0);

        // A synthetic structural-hash collision exercises full descriptor equality and bucket
        // rebuilding without making pointer/hash identity semantic.
        let live_hash = structural_hash(arena.descriptor(live).unwrap());
        insert_bucket_slot(&mut arena.buckets, live_hash, dead.slot);
        let report = arena.sweep(0, [live]).unwrap();
        assert_eq!(report.high_water_slots, 2);
        assert_eq!(report.occupied_slots, 1);
        assert_eq!(report.reclaimed_slots, 1);
        assert!(report.reclaimed_payload_lower_bound_bytes > 0);
        assert_eq!(arena.occupied_len(), 1);
        let census = arena.owner_census([live]);
        assert_eq!(census.allocated_descriptor_slots, 2);
        assert_eq!(census.retained_descriptor_slots, 1);
        assert_eq!(census.reclaimed_descriptor_slots, 1);
        assert_eq!(arena.allocated_payload_since_sweep(), 0);
        assert!(matches!(
            arena.descriptor(dead),
            Err(MonomialError::CollectedMonomialId { slot }) if slot == dead.slot
        ));
        assert_eq!(arena.descriptor(live).unwrap().central_factors.as_ref(), &[one]);
        assert_eq!(
            arena
                .buckets
                .iter()
                .flat_map(HashMap::values)
                .map(|bucket| bucket.slots().len())
                .sum::<usize>(),
            1
        );

        let reinterned = arena.intern(&expressions, &programs, &[], &[root, one]).unwrap();
        assert_ne!(reinterned, dead);
        assert!(reinterned.slot > dead.slot);
        assert_eq!(arena.len(), 3);
        assert_eq!(arena.occupied_len(), 2);
    }

    #[test]
    fn sweep_prunes_bucket_entries_in_place_and_collapses_collisions() {
        let (expressions, programs, scope, one, _, _) = fixture();
        let mut arena = MonomialArena::new(&expressions, &programs, scope).unwrap();
        let root = programs.root(&expressions, scope).unwrap();
        let ids = (1..=8_usize)
            .map(|width| arena.intern(&expressions, &programs, &[], &vec![root; width]).unwrap())
            .collect::<Vec<_>>();
        arena.sweep(0, [ids[0], ids[1]]).unwrap();
        assert_eq!(arena.len(), 8);
        assert_eq!(arena.occupied_len(), 2);

        let dead = arena.intern(&expressions, &programs, &[one], &[root, one, root]).unwrap();
        let live_hash = structural_hash(arena.descriptor(ids[0]).unwrap());
        insert_bucket_slot(&mut arena.buckets, live_hash, dead.slot);
        let report = arena.sweep(0, [ids[0], ids[1]]).unwrap();
        assert_eq!(report.high_water_slots, 9);
        assert_eq!(report.occupied_slots, 2);
        assert_eq!(report.reclaimed_slots, 1);
        assert!(matches!(
            arena.buckets[bucket_shard(live_hash)].get(&live_hash),
            Some(MonomialBucket::Single(slot)) if *slot == ids[0].slot
        ));
        assert_eq!(
            arena
                .buckets
                .iter()
                .flat_map(HashMap::values)
                .map(|bucket| bucket.slots().len())
                .sum::<usize>(),
            2
        );
        assert!(matches!(
            arena.descriptor(dead),
            Err(MonomialError::CollectedMonomialId { slot }) if slot == dead.slot
        ));
    }

    #[test]
    fn sweep_attributes_unique_roots_in_fixed_owner_precedence() {
        let (expressions, programs, scope, one, _, _) = fixture();
        let mut arena = MonomialArena::new(&expressions, &programs, scope).unwrap();
        let root = programs.root(&expressions, scope).unwrap();
        let protected = arena.intern(&expressions, &programs, &[one], &[]).unwrap();
        let value = arena.intern(&expressions, &programs, &[], &[root]).unwrap();
        let exact = arena.intern(&expressions, &programs, &[], &[root, one]).unwrap();
        let gadget = arena.intern(&expressions, &programs, &[], &[root, one, root]).unwrap();
        let canonical =
            arena.intern(&expressions, &programs, &[], &[root, one, root, one]).unwrap();
        let closed =
            arena.intern(&expressions, &programs, &[], &[root, one, root, one, root]).unwrap();
        let suspended =
            arena.intern(&expressions, &programs, &[], &[root, one, root, one, root, one]).unwrap();
        let dead = arena
            .intern(&expressions, &programs, &[], &[root, root, one, one, root, root])
            .unwrap();

        let report = arena
            .sweep_with_owners(
                1,
                [protected, value],
                [value, exact],
                [exact, gadget],
                [gadget, canonical],
                [canonical, closed],
                [closed, suspended, protected],
            )
            .unwrap();
        assert_eq!(report.protected_prefix.descriptor_slots, 1);
        assert_eq!(report.value_cache.descriptor_slots, 1);
        assert_eq!(report.exact_plan.descriptor_slots, 1);
        assert_eq!(report.gadget.descriptor_slots, 1);
        assert_eq!(report.canonical_runtime.descriptor_slots, 1);
        assert_eq!(report.closed.descriptor_slots, 1);
        assert_eq!(report.suspended.descriptor_slots, 1);
        for owner in [
            report.protected_prefix,
            report.value_cache,
            report.exact_plan,
            report.gadget,
            report.canonical_runtime,
            report.closed,
            report.suspended,
        ] {
            assert!(owner.payload_lower_bound_bytes > 0);
        }
        assert_eq!(report.protected_prefix_occupied_slots, 1);
        assert_eq!(report.high_water_slots, 8);
        assert_eq!(report.occupied_slots, 7);
        assert_eq!(report.reclaimed_slots, 1);
        assert_eq!(report.bucket_entries, 7);
        assert_eq!(report.occupied_central_factor_entries, 1);
        assert_eq!(report.occupied_ordered_factor_entries, 21);
        assert_eq!(
            report.occupied_factor_payload_lower_bound_bytes,
            22 * u64::try_from(std::mem::size_of::<ScopedExprId>()).unwrap()
        );
        assert!(matches!(arena.descriptor(dead), Err(MonomialError::CollectedMonomialId { .. })));
    }

    #[test]
    fn sweep_pins_preexisting_prefix_and_rejects_invalid_roots_before_mutation() {
        let (expressions, programs, scope, one, _, _) = fixture();
        let mut arena = MonomialArena::new(&expressions, &programs, scope).unwrap();
        let root = programs.root(&expressions, scope).unwrap();
        let preexisting = arena.intern(&expressions, &programs, &[one], &[]).unwrap();
        let protected_prefix = arena.len();
        let later = arena.intern(&expressions, &programs, &[], &[root]).unwrap();
        let before = arena.occupied_len();
        assert!(matches!(
            arena.sweep(protected_prefix, [MonomialId::new(ArenaToken::fresh(), later.slot)]),
            Err(MonomialError::InvalidMonomialId { .. })
        ));
        assert_eq!(arena.occupied_len(), before, "failed validation must not partially sweep");

        arena.sweep(protected_prefix, std::iter::empty()).unwrap();
        assert!(arena.descriptor(preexisting).is_ok());
        assert!(matches!(arena.descriptor(later), Err(MonomialError::CollectedMonomialId { .. })));
        assert!(matches!(
            arena.sweep(protected_prefix, [later]),
            Err(MonomialError::CollectedMonomialId { .. })
        ));
        let fresh = arena.intern(&expressions, &programs, &[], &[root, one]).unwrap();
        let occupied_before_tombstone_error = arena.occupied_len();
        assert!(matches!(arena.sweep(0, [later]), Err(MonomialError::CollectedMonomialId { .. })));
        assert_eq!(arena.occupied_len(), occupied_before_tombstone_error);
        assert!(arena.descriptor(fresh).is_ok(), "tombstone validation precedes mutation");
    }

    #[test]
    fn ordered_factors_preserve_noncommutative_sequence() {
        let (expressions, programs, scope, one, wide, _) = fixture();
        let mut arena = MonomialArena::new(&expressions, &programs, scope).unwrap();
        let root = programs.root(&expressions, scope).unwrap();
        let left = arena.intern(&expressions, &programs, &[], &[one, root]).unwrap();
        let right = arena.intern(&expressions, &programs, &[], &[root, one]).unwrap();
        assert_ne!(left, right);
        assert_eq!(arena.descriptor(left).unwrap().ordered_factors.as_ref(), &[one, root]);
        assert_eq!(arena.descriptor(right).unwrap().ordered_factors.as_ref(), &[root, one]);
        assert!(matches!(
            arena.intern(&expressions, &programs, &[], &[one, wide]),
            Err(MonomialError::IncompatibleOrderedShape { .. })
        ));
    }

    #[test]
    fn rejects_foreign_scope_and_non_matrix_central_factor() {
        let (mut expressions, mut programs, scope, one, wide, scalar) = fixture();
        let mut arena = MonomialArena::new(&expressions, &programs, scope).unwrap();
        assert!(matches!(
            arena.intern(&expressions, &programs, &[wide], &[]),
            Err(MonomialError::CentralNotScalar { .. })
        ));
        assert!(matches!(
            arena.intern(&expressions, &programs, &[], &[scalar]),
            Err(MonomialError::NotMatrix { .. })
        ));
        let scope2 = {
            let signature = ProgramSignature {
                inputs: Box::new([
                    ProgramInput { value_type: matrix(1, 1, 17), trusted_index_range: None },
                    ProgramInput { value_type: matrix(2, 2, 17), trusted_index_range: None },
                    ProgramInput { value_type: ResolvedValueType::Int, trusted_index_range: None },
                    ProgramInput { value_type: matrix(1, 1, 17), trusted_index_range: None },
                ]),
                output: matrix(1, 1, 17),
            };
            programs.finalize(&mut expressions, signature, one.expression()).unwrap()
        };
        assert_ne!(scope, scope2);
        let foreign = programs.root(&expressions, scope2).unwrap();
        assert!(matches!(
            arena.intern(&expressions, &programs, &[], &[foreign]),
            Err(MonomialError::ForeignScope { .. })
        ));
    }

    #[test]
    fn combine_interned_matches_same_scope_relation_product_semantics() {
        let (expressions, programs, scope, one, _, _) = fixture();
        let mut arena = MonomialArena::new(&expressions, &programs, scope).unwrap();
        let root = programs.root(&expressions, scope).unwrap();
        let left = arena.intern(&expressions, &programs, &[one], &[root, one]).unwrap();
        let right = arena.intern(&expressions, &programs, &[root, one], &[one]).unwrap();
        let combined = arena.combine_interned(scope, left, right).unwrap();
        let expected =
            arena.intern(&expressions, &programs, &[one, root, one], &[root, one, one]).unwrap();
        assert_eq!(combined, expected);
        let descriptor = arena.descriptor(combined).unwrap();
        assert_eq!(descriptor.central_factors.len(), 3);
        assert!(descriptor.central_factors.windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!(descriptor.ordered_factors.as_ref(), &[root, one, one]);
        let before = arena.len();
        assert_eq!(
            arena.find_interned(scope, &[one, one, root], &[root, one, one]).unwrap(),
            Some(combined)
        );
        assert_eq!(arena.find_interned(scope, &[one, one, root], &[one, root, one]).unwrap(), None);
        assert_eq!(arena.len(), before, "lookup must never intern a missing descriptor");
    }

    #[test]
    fn parallel_wrapped_batch_matches_ordered_sequential_interning() {
        let (expressions, programs, scope, one, _, _) = fixture();
        let root = programs.root(&expressions, scope).unwrap();
        let build_inputs = |arena: &mut MonomialArena| {
            (0..300_usize)
                .map(|index| {
                    let ordered = (0..index + 1)
                        .map(|position| if position % 2 == 0 { root } else { one })
                        .collect::<Vec<_>>();
                    arena.intern(&expressions, &programs, &[], &ordered).unwrap()
                })
                .collect::<Vec<_>>()
        };
        let mut sequential = MonomialArena::new(&expressions, &programs, scope).unwrap();
        let sequential_prefix =
            sequential.intern(&expressions, &programs, &[one], &[root]).unwrap();
        let sequential_suffix = sequential.intern(&expressions, &programs, &[], &[one]).unwrap();
        let sequential_inputs = build_inputs(&mut sequential);
        let sequential_outputs = sequential_inputs
            .iter()
            .map(|&input| {
                let intermediate =
                    sequential.combine_interned(scope, sequential_prefix, input).unwrap();
                sequential.combine_interned(scope, intermediate, sequential_suffix).unwrap()
            })
            .collect::<Vec<_>>();

        let mut parallel = MonomialArena::new(&expressions, &programs, scope).unwrap();
        let parallel_prefix = parallel.intern(&expressions, &programs, &[one], &[root]).unwrap();
        let parallel_suffix = parallel.intern(&expressions, &programs, &[], &[one]).unwrap();
        let parallel_inputs = build_inputs(&mut parallel);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let parallel_outputs = pool
            .install(|| {
                parallel.combine_interned_wrapped_batch(
                    scope,
                    Some(parallel_prefix),
                    &parallel_inputs,
                    Some(parallel_suffix),
                )
            })
            .unwrap();

        assert_eq!(sequential.len(), parallel.len());
        for (sequential_id, parallel_id) in sequential_outputs.into_iter().zip(parallel_outputs) {
            assert_eq!(sequential_id.slot, parallel_id.slot, "commit order must remain stable");
            let sequential_descriptor = sequential.descriptor(sequential_id).unwrap();
            let parallel_descriptor = parallel.descriptor(parallel_id).unwrap();
            assert_eq!(sequential_descriptor, parallel_descriptor);
        }

        let before_invalid = parallel.len();
        let mut invalid_inputs = parallel_inputs;
        invalid_inputs.push(MonomialId::new(ArenaToken::fresh(), 0));
        assert!(matches!(
            parallel.combine_interned_wrapped_batch(
                scope,
                Some(parallel_prefix),
                &invalid_inputs,
                Some(parallel_suffix),
            ),
            Err(MonomialError::InvalidMonomialId { .. })
        ));
        assert_eq!(parallel.len(), before_invalid, "validation must precede batch mutation");
    }

    #[test]
    fn parallel_sweep_rebuild_matches_single_threaded_bucket_index() {
        let (expressions, programs, scope, one, _, _) = fixture();
        let root = programs.root(&expressions, scope).unwrap();
        let build = || {
            let mut arena = MonomialArena::new(&expressions, &programs, scope).unwrap();
            let ids = (0..600_usize)
                .map(|index| {
                    let ordered = (0..index + 1)
                        .map(|position| if (index + position) % 2 == 0 { root } else { one })
                        .collect::<Vec<_>>();
                    arena.intern(&expressions, &programs, &[], &ordered).unwrap()
                })
                .collect::<Vec<_>>();
            (arena, ids)
        };
        let (mut sequential, sequential_ids) = build();
        let (mut parallel, parallel_ids) = build();
        let sequential_roots = sequential_ids.iter().step_by(3).copied().collect::<Vec<_>>();
        let parallel_roots = parallel_ids.iter().step_by(3).copied().collect::<Vec<_>>();

        let sequential_pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let sequential_report =
            sequential_pool.install(|| sequential.sweep(0, sequential_roots)).unwrap();
        let parallel_pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let parallel_report = parallel_pool.install(|| parallel.sweep(0, parallel_roots)).unwrap();

        assert_eq!(sequential_report, parallel_report);
        assert_eq!(sequential.descriptors, parallel.descriptors);
        assert_eq!(sequential.buckets, parallel.buckets);
    }

    #[test]
    fn combine_interned_rejects_different_scope_and_foreign_monomial_arena() {
        let (mut expressions, mut programs, scope, one, _, _) = fixture();
        let mut arena = MonomialArena::new(&expressions, &programs, scope).unwrap();
        let local = arena.intern(&expressions, &programs, &[one], &[]).unwrap();
        let other_scope = programs
            .finalize(
                &mut expressions,
                ProgramSignature {
                    inputs: Box::new([
                        ProgramInput { value_type: matrix(1, 1, 17), trusted_index_range: None },
                        ProgramInput { value_type: matrix(2, 2, 17), trusted_index_range: None },
                        ProgramInput {
                            value_type: ResolvedValueType::Int,
                            trusted_index_range: None,
                        },
                    ]),
                    output: matrix(1, 1, 17),
                },
                one.expression(),
            )
            .unwrap();
        assert!(matches!(
            arena.combine_interned(other_scope, local, local),
            Err(MonomialError::ForeignScope { expected, actual })
                if expected == scope && actual == other_scope
        ));

        let mut foreign = MonomialArena::new(&expressions, &programs, scope).unwrap();
        let foreign_id = foreign.intern(&expressions, &programs, &[one], &[]).unwrap();
        assert!(matches!(
            arena.combine_interned(scope, local, foreign_id),
            Err(MonomialError::InvalidMonomialId { expected, actual })
                if expected == arena.token() && actual == foreign.token()
        ));
        let foreign_scope_factor = programs.root(&expressions, other_scope).unwrap();
        assert!(matches!(
            arena.find_interned(scope, &[foreign_scope_factor], &[]),
            Err(MonomialError::ForeignScope { expected, actual })
                if expected == scope && actual == other_scope
        ));

        let (foreign_expressions, foreign_programs, foreign_scope, foreign_factor, _, _) =
            fixture();
        assert!(matches!(
            arena.find_interned(scope, &[foreign_factor], &[]),
            Err(MonomialError::ForeignExpressionArena { expected, actual })
                if expected == expressions.token() && actual == foreign_expressions.token()
        ));
        assert!(foreign_programs.program(foreign_scope).is_ok());
    }

    #[test]
    fn combine_interned_deep_shared_descriptors_are_iterative_and_reused() {
        let (expressions, programs, scope, one, _, _) = fixture();
        let mut arena = MonomialArena::new(&expressions, &programs, scope).unwrap();
        let factors = vec![one; 50_000];
        let half = arena.intern(&expressions, &programs, &[], &factors).unwrap();
        let combined = arena.combine_interned(scope, half, half).unwrap();
        assert_eq!(arena.descriptor(combined).unwrap().ordered_factors.len(), 100_000);
        assert!(
            arena.descriptor(combined).unwrap().ordered_factors.iter().all(|factor| *factor == one)
        );
        let before = arena.len();
        assert_eq!(arena.combine_interned(scope, half, half).unwrap(), combined);
        assert_eq!(arena.len(), before);
        let full = vec![one; 100_000];
        assert_eq!(arena.find_interned(scope, &[], &full).unwrap(), Some(combined));
        assert_eq!(arena.len(), before);
    }

    #[test]
    fn shared_ids_and_deep_lists_are_stack_safe() {
        let (expressions, programs, scope, one, _, _) = fixture();
        let mut arena = MonomialArena::new(&expressions, &programs, scope).unwrap();
        let factors = vec![one; 100_000];
        let id = arena.intern(&expressions, &programs, &[], &factors).unwrap();
        assert_eq!(arena.descriptor(id).unwrap().ordered_factors.len(), 100_000);
        let again = arena.intern(&expressions, &programs, &[], &factors).unwrap();
        assert_eq!(id, again);
        assert_eq!(arena.len(), 1);
    }
}
