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
use std::{
    collections::{BTreeMap, hash_map::DefaultHasher},
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
#[derive(Debug, Eq, PartialEq)]
pub struct MonomialDescriptor {
    pub central_factors: Box<[ScopedExprId]>,
    pub ordered_factors: Box<[ScopedExprId]>,
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
    buckets: BTreeMap<u64, Vec<u32>>,
    central_factor_entries: u64,
    ordered_factor_entries: u64,
    occupied_descriptor_slots: u64,
    allocated_payload_since_sweep: u64,
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
            buckets: BTreeMap::new(),
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
        let Some(slots) = self.buckets.get(&hash) else {
            return Ok(None);
        };
        for &slot in slots {
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
        if let Some(slots) = self.buckets.get(&hash) {
            for &slot in slots {
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
        self.buckets.entry(hash).or_default().push(slot);
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
        )
    }

    pub(crate) fn sweep_with_owners(
        &mut self,
        protected_prefix: usize,
        value_cache_roots: impl IntoIterator<Item = MonomialId>,
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
        let gadget = self.mark_sweep_owner(&mut marked, gadget_roots.into_iter())?;
        let canonical_runtime =
            self.mark_sweep_owner(&mut marked, canonical_runtime_roots.into_iter())?;
        let closed = self.mark_sweep_owner(&mut marked, closed_roots.into_iter())?;
        let suspended = self.mark_sweep_owner(&mut marked, suspended_roots.into_iter())?;

        let mut reclaimed_slots = 0_u64;
        let mut reclaimed_payload = 0_u64;
        for (slot, entry) in self.descriptors.iter_mut().enumerate() {
            if marked.get(slot / 64).is_some_and(|word| word & (1_u64 << (slot % 64)) != 0) {
                continue;
            }
            let Some(descriptor) = entry.take() else { continue };
            let central = u64::try_from(descriptor.central_factors.len()).unwrap_or(u64::MAX);
            let ordered = u64::try_from(descriptor.ordered_factors.len()).unwrap_or(u64::MAX);
            self.central_factor_entries = self.central_factor_entries.saturating_sub(central);
            self.ordered_factor_entries = self.ordered_factor_entries.saturating_sub(ordered);
            self.occupied_descriptor_slots = self.occupied_descriptor_slots.saturating_sub(1);
            reclaimed_slots = reclaimed_slots.saturating_add(1);
            reclaimed_payload = reclaimed_payload
                .saturating_add(descriptor_payload_lower_bound_bytes(central, ordered));
        }

        self.buckets.clear();
        for (slot, descriptor) in self.descriptors.iter().enumerate() {
            let Some(descriptor) = descriptor.as_ref() else { continue };
            let slot = u32::try_from(slot).map_err(|_| MonomialError::ArenaExhausted)?;
            self.buckets.entry(structural_hash(descriptor)).or_default().push(slot);
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
        arena.buckets.entry(live_hash).or_default().push(dead.slot);
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
        assert_eq!(arena.buckets.values().map(Vec::len).sum::<usize>(), 1);

        let reinterned = arena.intern(&expressions, &programs, &[], &[root, one]).unwrap();
        assert_ne!(reinterned, dead);
        assert!(reinterned.slot > dead.slot);
        assert_eq!(arena.len(), 3);
        assert_eq!(arena.occupied_len(), 2);
    }

    #[test]
    fn sweep_attributes_unique_roots_in_fixed_owner_precedence() {
        let (expressions, programs, scope, one, _, _) = fixture();
        let mut arena = MonomialArena::new(&expressions, &programs, scope).unwrap();
        let root = programs.root(&expressions, scope).unwrap();
        let protected = arena.intern(&expressions, &programs, &[one], &[]).unwrap();
        let value = arena.intern(&expressions, &programs, &[], &[root]).unwrap();
        let gadget = arena.intern(&expressions, &programs, &[], &[root, one]).unwrap();
        let canonical = arena.intern(&expressions, &programs, &[], &[root, one, root]).unwrap();
        let closed = arena.intern(&expressions, &programs, &[], &[root, one, root, one]).unwrap();
        let suspended =
            arena.intern(&expressions, &programs, &[], &[root, one, root, one, root]).unwrap();
        let dead = arena
            .intern(&expressions, &programs, &[], &[root, root, one, one, root, root])
            .unwrap();

        let report = arena
            .sweep_with_owners(
                1,
                [protected, value],
                [value, gadget],
                [gadget, canonical],
                [canonical, closed],
                [closed, suspended, protected],
            )
            .unwrap();
        assert_eq!(report.protected_prefix.descriptor_slots, 1);
        assert_eq!(report.value_cache.descriptor_slots, 1);
        assert_eq!(report.gadget.descriptor_slots, 1);
        assert_eq!(report.canonical_runtime.descriptor_slots, 1);
        assert_eq!(report.closed.descriptor_slots, 1);
        assert_eq!(report.suspended.descriptor_slots, 1);
        for owner in [
            report.protected_prefix,
            report.value_cache,
            report.gadget,
            report.canonical_runtime,
            report.closed,
            report.suspended,
        ] {
            assert!(owner.payload_lower_bound_bytes > 0);
        }
        assert_eq!(report.protected_prefix_occupied_slots, 1);
        assert_eq!(report.high_water_slots, 7);
        assert_eq!(report.occupied_slots, 6);
        assert_eq!(report.reclaimed_slots, 1);
        assert_eq!(report.bucket_entries, 6);
        assert_eq!(report.occupied_central_factor_entries, 1);
        assert_eq!(report.occupied_ordered_factor_entries, 15);
        assert_eq!(
            report.occupied_factor_payload_lower_bound_bytes,
            16 * u64::try_from(std::mem::size_of::<ScopedExprId>()).unwrap()
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
