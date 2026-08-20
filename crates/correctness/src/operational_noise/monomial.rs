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
    descriptors: Vec<MonomialDescriptor>,
    // Hash buckets contain slots only.  Full descriptor equality below makes
    // this collision-safe without storing a second copy of either factor list.
    buckets: BTreeMap<u64, Vec<u32>>,
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
        })
    }

    pub fn scope(&self) -> ValueProgramId {
        self.scope
    }

    #[cfg(test)]
    pub fn token(&self) -> ArenaToken {
        self.token
    }

    pub fn len(&self) -> usize {
        self.descriptors.len()
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
            if self.descriptors.get(slot as usize) == Some(&descriptor) {
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
                let Some(existing) = self.descriptors.get(slot as usize) else {
                    continue;
                };
                if existing == &descriptor {
                    return Ok(MonomialId::new(self.token, slot));
                }
            }
        }

        let slot =
            u32::try_from(self.descriptors.len()).map_err(|_| MonomialError::ArenaExhausted)?;
        self.descriptors.push(descriptor);
        self.buckets.entry(hash).or_default().push(slot);
        Ok(MonomialId::new(self.token, slot))
    }

    pub fn descriptor(&self, id: MonomialId) -> Result<&MonomialDescriptor, MonomialError> {
        if id.arena != self.token {
            return Err(MonomialError::InvalidMonomialId { expected: self.token, actual: id.arena });
        }
        self.descriptors.get(id.slot as usize).ok_or(MonomialError::InvalidSlot { slot: id.slot })
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
