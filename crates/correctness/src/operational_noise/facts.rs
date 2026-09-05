//! Typed facts for the operational-noise arenas.
//!
//! Facts are deliberately stored outside expression identity.  In particular,
//! a finite bound can never make two expressions equal and a missing transfer
//! contract can never be upgraded to `Large` by this module.

use super::arena::{
    ArenaError, ArenaToken, ExprArena, ExprId, MatrixLayout, ResolvedMatrixType, ResolvedValueType,
    TrustedIndexRange, ValueProgramId,
};
use num_bigint::BigUint;
use std::{collections::BTreeMap, fmt};

/// A transfer contract is either absent or fully known.  `Missing` is not a
/// numeric value and must not be interpreted as `Large`.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum NumericContract<T> {
    Missing,
    Known(T),
}

impl<T> NumericContract<T> {
    pub fn is_missing(&self) -> bool {
        matches!(self, Self::Missing)
    }

    pub fn as_known(&self) -> Option<&T> {
        match self {
            Self::Missing => None,
            Self::Known(value) => Some(value),
        }
    }
}

/// A finite coefficient bound remains a value-level expression summary.  It
/// is never used as a semantic identity.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BoundExpression {
    pub maximum_absolute_coefficient: BigUint,
}

impl BoundExpression {
    pub fn new(maximum_absolute_coefficient: BigUint) -> Self {
        Self { maximum_absolute_coefficient }
    }
}

impl From<BigUint> for BoundExpression {
    fn from(value: BigUint) -> Self {
        Self::new(value)
    }
}

/// Exact-zero and exact-large are semantic classes; finite is a sound numeric
/// summary.  `Known(Large)` is only accepted when a caller explicitly supplies
/// that exact residual fact.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum CoefficientBound {
    ExactZero,
    Finite(BoundExpression),
    Large,
}

impl CoefficientBound {
    pub fn finite(value: impl Into<BigUint>) -> Self {
        let value = value.into();
        if value == BigUint::from(0_u8) {
            Self::ExactZero
        } else {
            Self::Finite(BoundExpression::new(value))
        }
    }
}

/// Bounds for polynomial support after typed validation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct PolynomialFacts {
    pub support_upper: usize,
    pub ring_dimension: usize,
}

impl PolynomialFacts {
    pub fn new(support_upper: usize, ring_dimension: usize) -> Result<Self, FactError> {
        if ring_dimension == 0 || support_upper > ring_dimension {
            return Err(FactError::InvalidPolynomialFacts { support_upper, ring_dimension });
        }
        Ok(Self { support_upper, ring_dimension })
    }
}

/// Layout and canonical-residue metadata are facts, not identity fields.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct MatrixMetadata {
    pub layout: MatrixLayout,
    pub canonical_coefficient_exclusive_upper: Option<BigUint>,
    pub known_zero_rows: Option<usize>,
    pub is_constant_polynomial: bool,
}

impl MatrixMetadata {
    pub fn new(layout: MatrixLayout) -> Self {
        Self {
            layout,
            canonical_coefficient_exclusive_upper: None,
            known_zero_rows: None,
            is_constant_polynomial: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg(test)]
pub struct ScalarFacts {
    pub value_type: ResolvedValueType,
    pub coefficient_bound: NumericContract<CoefficientBound>,
}

#[cfg(test)]
impl ScalarFacts {
    pub fn new(value_type: ResolvedValueType) -> Result<Self, FactError> {
        if !matches!(
            value_type,
            ResolvedValueType::Bool |
                ResolvedValueType::Int |
                ResolvedValueType::Real |
                ResolvedValueType::Bytes
        ) {
            return Err(FactError::WrongFactType);
        }
        Ok(Self { value_type, coefficient_bound: NumericContract::Missing })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct MatrixFacts {
    pub matrix_type: ResolvedMatrixType,
    pub coefficient_bound: NumericContract<CoefficientBound>,
    pub polynomial: NumericContract<PolynomialFacts>,
    pub metadata: MatrixMetadata,
}

impl MatrixFacts {
    pub fn new(matrix_type: ResolvedMatrixType, metadata: MatrixMetadata) -> Self {
        Self {
            matrix_type,
            coefficient_bound: NumericContract::Missing,
            polynomial: NumericContract::Missing,
            metadata,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TrapdoorFacts {
    pub coefficient_bound: NumericContract<CoefficientBound>,
    pub descriptor: String,
    pub paired_public_event: super::arena::SampleEventId,
    pub paired_public_output_role: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg(test)]
pub struct IndexFacts {
    pub range: Option<TrustedIndexRange>,
}

/// Facts are typed by the semantic value's resolved type and are never used
/// as a substitute for the expression ID.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ValueFacts {
    #[cfg(test)]
    Scalar(ScalarFacts),
    Matrix(MatrixFacts),
    Trapdoor(TrapdoorFacts),
    #[cfg(test)]
    Index(IndexFacts),
}

impl ValueFacts {
    pub fn value_type(&self) -> ResolvedValueType {
        match self {
            #[cfg(test)]
            Self::Scalar(facts) => facts.value_type.clone(),
            Self::Matrix(facts) => ResolvedValueType::Matrix(facts.matrix_type.clone()),
            Self::Trapdoor(_) => ResolvedValueType::Trapdoor,
            #[cfg(test)]
            Self::Index(_) => ResolvedValueType::Int,
        }
    }

    fn coefficient_bound(&self) -> Option<&NumericContract<CoefficientBound>> {
        match self {
            #[cfg(test)]
            Self::Scalar(facts) => Some(&facts.coefficient_bound),
            Self::Matrix(facts) => Some(&facts.coefficient_bound),
            Self::Trapdoor(facts) => Some(&facts.coefficient_bound),
            #[cfg(test)]
            Self::Index(_) => None,
        }
    }

    #[cfg(test)]
    fn coefficient_bound_mut(&mut self) -> Option<&mut NumericContract<CoefficientBound>> {
        match self {
            Self::Scalar(facts) => Some(&mut facts.coefficient_bound),
            Self::Matrix(facts) => Some(&mut facts.coefficient_bound),
            Self::Trapdoor(facts) => Some(&mut facts.coefficient_bound),
            Self::Index(_) => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FactError {
    ForeignExpression {
        expected: ArenaToken,
        actual: ArenaToken,
    },
    InvalidExpression {
        id: ExprId,
    },
    #[cfg(test)]
    InvalidFactVariant {
        id: ExprId,
        value_type: ResolvedValueType,
    },
    BinderOpenExpression {
        id: ExprId,
    },
    TrapdoorAuthorityMismatch {
        id: ExprId,
    },
    FactTypeMismatch {
        id: ExprId,
        expected: ResolvedValueType,
        actual: ResolvedValueType,
    },
    RangeRequiresInteger {
        id: ExprId,
        actual: ResolvedValueType,
    },
    ConflictingFacts {
        id: ExprId,
        first: String,
        second: String,
    },
    MissingFacts {
        id: ExprId,
    },
    #[cfg(test)]
    WrongFactType,
    InvalidPolynomialFacts {
        support_upper: usize,
        ring_dimension: usize,
    },
    InvalidRange {
        minimum: u64,
        maximum_exclusive: u64,
    },
    ConflictingRange {
        id: ExprId,
        first: TrustedIndexRange,
        second: TrustedIndexRange,
    },
    LateRangeDeclaration {
        id: ExprId,
    },
    IndexRangeRequired {
        id: ExprId,
    },
    #[cfg(test)]
    MissingTransferInput,
}

impl fmt::Display for FactError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for FactError {}

/// A job-local, insertion-order-independent fact table.
pub struct FactStore {
    arena: ArenaToken,
    values: BTreeMap<ExprId, ValueFacts>,
    ranges: BTreeMap<ExprId, TrustedIndexRange>,
    scoped_ranges: BTreeMap<(ValueProgramId, ExprId), TrustedIndexRange>,
    ranges_finalized: bool,
}

impl FactStore {
    pub fn new(expressions: &ExprArena) -> Self {
        Self {
            arena: expressions.token(),
            values: BTreeMap::new(),
            ranges: BTreeMap::new(),
            scoped_ranges: BTreeMap::new(),
            ranges_finalized: false,
        }
    }

    pub fn arena(&self) -> ArenaToken {
        self.arena
    }

    fn check_id(&self, id: ExprId) -> Result<(), FactError> {
        if id.arena() != self.arena {
            return Err(FactError::ForeignExpression { expected: self.arena, actual: id.arena() });
        }
        Ok(())
    }

    fn expression_type<'arena>(
        &self,
        expressions: &'arena ExprArena,
        id: ExprId,
    ) -> Result<&'arena ResolvedValueType, FactError> {
        self.check_id(id)?;
        if expressions.token() != self.arena {
            return Err(FactError::ForeignExpression {
                expected: self.arena,
                actual: expressions.token(),
            });
        }
        expressions.value_type(id).map_err(|error| match error {
            ArenaError::ForeignExpression { expected, actual } => {
                FactError::ForeignExpression { expected, actual }
            }
            ArenaError::InvalidSlot { .. } => FactError::InvalidExpression { id },
            _ => FactError::InvalidExpression { id },
        })
    }

    fn validate_facts(
        &self,
        expressions: &ExprArena,
        id: ExprId,
        facts: &ValueFacts,
    ) -> Result<(), FactError> {
        #[cfg(test)]
        if let ValueFacts::Scalar(scalar) = facts {
            if !matches!(
                &scalar.value_type,
                ResolvedValueType::Bool |
                    ResolvedValueType::Int |
                    ResolvedValueType::Real |
                    ResolvedValueType::Bytes
            ) {
                return Err(FactError::InvalidFactVariant {
                    id,
                    value_type: scalar.value_type.clone(),
                });
            }
        }
        let expected = self.expression_type(expressions, id)?.clone();
        if !expressions
            .free_arguments(id)
            .map_err(|_| FactError::InvalidExpression { id })?
            .is_empty()
        {
            return Err(FactError::BinderOpenExpression { id });
        }
        let actual = facts.value_type();
        if expected != actual {
            return Err(FactError::FactTypeMismatch { id, expected, actual });
        }
        if let ValueFacts::Trapdoor(trapdoor) = facts {
            let node = expressions.node(id).map_err(|_| FactError::InvalidExpression { id })?;
            match &node.operator {
                super::arena::ValueOperator::Trapdoor(
                    super::arena::TrapdoorOperation::Generate {
                        descriptor,
                        paired_public_event,
                        paired_public_output_role,
                        ..
                    },
                ) if descriptor == &trapdoor.descriptor &&
                    paired_public_event == &trapdoor.paired_public_event &&
                    paired_public_output_role == &trapdoor.paired_public_output_role => {}
                _ => return Err(FactError::TrapdoorAuthorityMismatch { id }),
            }
        }
        Ok(())
    }

    pub fn insert(
        &mut self,
        expressions: &ExprArena,
        id: ExprId,
        facts: ValueFacts,
    ) -> Result<(), FactError> {
        self.validate_facts(expressions, id, &facts)?;
        if let Some(existing) = self.values.get(&id) {
            if existing != &facts {
                let (first, second) = canonical_conflict(existing, &facts);
                return Err(FactError::ConflictingFacts { id, first, second });
            }
            return Ok(());
        }
        self.values.insert(id, facts);
        Ok(())
    }

    pub fn facts(&self, id: ExprId) -> Result<&ValueFacts, FactError> {
        self.check_id(id)?;
        self.values.get(&id).ok_or(FactError::MissingFacts { id })
    }

    pub fn declare_trusted_index_range(
        &mut self,
        expressions: &ExprArena,
        id: ExprId,
        range: TrustedIndexRange,
    ) -> Result<(), FactError> {
        let value_type = self.expression_type(expressions, id)?.clone();
        if !expressions
            .free_arguments(id)
            .map_err(|_| FactError::InvalidExpression { id })?
            .is_empty()
        {
            return Err(FactError::BinderOpenExpression { id });
        }
        if value_type != ResolvedValueType::Int {
            return Err(FactError::RangeRequiresInteger { id, actual: value_type });
        }
        if range.minimum > range.maximum_exclusive {
            return Err(FactError::InvalidRange {
                minimum: range.minimum,
                maximum_exclusive: range.maximum_exclusive,
            });
        }
        if self.ranges_finalized {
            return Err(FactError::LateRangeDeclaration { id });
        }
        if let Some(existing) = self.ranges.get(&id) {
            if existing != &range {
                let (first, second) =
                    if existing <= &range { (*existing, range) } else { (range, *existing) };
                return Err(FactError::ConflictingRange { id, first, second });
            }
            return Ok(());
        }
        self.ranges.insert(id, range);
        Ok(())
    }

    /// Declare a range for a binder-open expression under one finalized program scope. Raw
    /// expression IDs are intentionally insufficient authority because interning can reuse the
    /// same `Argument(0)` in independent programs with different domains.
    #[cfg(test)]
    pub fn declare_scoped_trusted_index_range(
        &mut self,
        expressions: &ExprArena,
        scope: ValueProgramId,
        id: ExprId,
        range: TrustedIndexRange,
    ) -> Result<(), FactError> {
        self.check_id(id)?;
        let value_type = self.expression_type(expressions, id)?.clone();
        if value_type != ResolvedValueType::Int {
            return Err(FactError::RangeRequiresInteger { id, actual: value_type });
        }
        if range.minimum > range.maximum_exclusive {
            return Err(FactError::InvalidRange {
                minimum: range.minimum,
                maximum_exclusive: range.maximum_exclusive,
            });
        }
        if self.ranges_finalized {
            return Err(FactError::LateRangeDeclaration { id });
        }
        let key = (scope, id);
        if let Some(existing) = self.scoped_ranges.get(&key) {
            if existing != &range {
                let (first, second) =
                    if existing <= &range { (*existing, range) } else { (range, *existing) };
                return Err(FactError::ConflictingRange { id, first, second });
            }
            return Ok(());
        }
        self.scoped_ranges.insert(key, range);
        Ok(())
    }

    pub fn finalize_ranges(&mut self) {
        self.ranges_finalized = true;
    }

    pub fn ranges_finalized(&self) -> bool {
        self.ranges_finalized
    }

    pub fn trusted_index_range(&self, id: ExprId) -> Result<TrustedIndexRange, FactError> {
        self.check_id(id)?;
        self.ranges.get(&id).copied().ok_or(FactError::IndexRangeRequired { id })
    }

    pub fn trusted_scoped_index_range(
        &self,
        scope: ValueProgramId,
        id: ExprId,
    ) -> Result<TrustedIndexRange, FactError> {
        if id.arena() != self.arena {
            return Err(FactError::ForeignExpression { expected: self.arena, actual: id.arena() });
        }
        self.scoped_ranges.get(&(scope, id)).copied().ok_or(FactError::IndexRangeRequired { id })
    }

    /// Replace only a missing coefficient contract.  Conflicting known facts
    /// are rejected, making repeated insertion deterministic and preserving
    /// exact identity independently of numeric summaries.
    #[cfg(test)]
    pub fn set_coefficient_bound(
        &mut self,
        id: ExprId,
        bound: NumericContract<CoefficientBound>,
    ) -> Result<(), FactError> {
        self.check_id(id)?;
        let facts = self.values.get_mut(&id).ok_or(FactError::MissingFacts { id })?;
        let current = facts.coefficient_bound_mut().ok_or(FactError::WrongFactType)?;
        if matches!(current, NumericContract::Missing) &&
            matches!(&bound, NumericContract::Known(CoefficientBound::Large))
        {
            return Err(FactError::MissingTransferInput);
        }
        merge_numeric_contract(current, bound).map_err(|(first, second)| {
            let (first, second) = canonical_conflict(&first, &second);
            FactError::ConflictingFacts { id, first, second }
        })
    }

    pub fn coefficient_bound(
        &self,
        id: ExprId,
    ) -> Result<&NumericContract<CoefficientBound>, FactError> {
        self.check_id(id)?;
        self.values
            .get(&id)
            .and_then(ValueFacts::coefficient_bound)
            .ok_or(FactError::MissingFacts { id })
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// A transfer cannot produce a known result when any required input
    /// contract is missing.  This is the central Stage 1 guard against the
    /// unsound `Missing -> Large` conversion.
    #[cfg(test)]
    pub fn transfer_bound(
        inputs: &[NumericContract<CoefficientBound>],
        result_if_known: CoefficientBound,
    ) -> NumericContract<CoefficientBound> {
        if inputs.iter().any(NumericContract::is_missing) {
            NumericContract::Missing
        } else {
            NumericContract::Known(result_if_known)
        }
    }
}

#[cfg(test)]
fn merge_numeric_contract<T: Clone + Eq + fmt::Debug>(
    current: &mut NumericContract<T>,
    incoming: NumericContract<T>,
) -> Result<(), (String, String)> {
    match (&*current, incoming) {
        (NumericContract::Missing, NumericContract::Missing) => Ok(()),
        (NumericContract::Missing, NumericContract::Known(value)) => {
            *current = NumericContract::Known(value);
            Ok(())
        }
        (NumericContract::Known(_), NumericContract::Missing) => Ok(()),
        (NumericContract::Known(first), NumericContract::Known(second)) if first == &second => {
            Ok(())
        }
        (NumericContract::Known(first), NumericContract::Known(second)) => {
            Err((format!("Known({first:?})"), format!("Known({second:?})")))
        }
    }
}

fn canonical_conflict<T: fmt::Debug>(first: &T, second: &T) -> (String, String) {
    let first = format!("{first:?}");
    let second = format!("{second:?}");
    if first <= second { (first, second) } else { (second, first) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::arena::{
        ExprArena, SampleDescriptor, SampleEventId, TypedConstant, ValueOperator,
    };

    fn scalar(arena: &mut ExprArena) -> ExprId {
        arena.intern(ValueOperator::Constant(TypedConstant::int(7)), Box::new([])).unwrap()
    }

    fn boolean(arena: &mut ExprArena) -> ExprId {
        arena.intern(ValueOperator::Constant(TypedConstant::bool(true)), Box::new([])).unwrap()
    }

    fn matrix_type() -> ResolvedMatrixType {
        ResolvedMatrixType::new(BigUint::from(17_u8), 8, 2, 2).unwrap()
    }

    fn matrix(arena: &mut ExprArena) -> (ExprId, ResolvedMatrixType) {
        let matrix_type = matrix_type();
        let id = arena
            .intern(
                ValueOperator::Sample {
                    event: SampleEventId(1),
                    descriptor: SampleDescriptor::new(
                        "matrix",
                        ResolvedValueType::Matrix(matrix_type.clone()),
                    ),
                },
                Box::new([]),
            )
            .unwrap();
        (id, matrix_type)
    }

    #[test]
    fn missing_is_distinct_and_never_becomes_large_on_transfer() {
        let result =
            FactStore::transfer_bound(&[NumericContract::Missing], CoefficientBound::Large);
        assert_eq!(result, NumericContract::Missing);
        assert_ne!(result, NumericContract::Known(CoefficientBound::Large));
    }

    #[test]
    fn explicit_large_is_retained_only_as_known_fact() {
        let mut arena = ExprArena::new();
        let id = scalar(&mut arena);
        let mut facts = FactStore::new(&arena);
        // An explicit residual fact remains distinct from a missing transfer
        // contract, while insertion still validates the expression's type.
        let scalar_facts = ScalarFacts {
            value_type: ResolvedValueType::Int,
            coefficient_bound: NumericContract::Known(CoefficientBound::Large),
        };
        facts.insert(&arena, id, ValueFacts::Scalar(scalar_facts)).unwrap();
        assert_eq!(
            facts.coefficient_bound(id).unwrap(),
            &NumericContract::Known(CoefficientBound::Large)
        );
    }

    #[test]
    fn setter_does_not_upgrade_missing_to_large() {
        let mut arena = ExprArena::new();
        let id = scalar(&mut arena);
        let mut facts = FactStore::new(&arena);
        facts
            .insert(
                &arena,
                id,
                ValueFacts::Scalar(ScalarFacts::new(ResolvedValueType::Int).unwrap()),
            )
            .unwrap();
        assert_eq!(
            facts.set_coefficient_bound(id, NumericContract::Known(CoefficientBound::Large)),
            Err(FactError::MissingTransferInput)
        );
    }

    #[test]
    fn conflicting_facts_and_ranges_are_order_independent() {
        let mut arena = ExprArena::new();
        let id = scalar(&mut arena);
        let first = ValueFacts::Scalar(ScalarFacts::new(ResolvedValueType::Int).unwrap());
        let second = ValueFacts::Scalar(ScalarFacts {
            value_type: ResolvedValueType::Int,
            coefficient_bound: NumericContract::Known(CoefficientBound::Large),
        });
        let mut left = FactStore::new(&arena);
        left.insert(&arena, id, first.clone()).unwrap();
        let error_left = left.insert(&arena, id, second.clone()).unwrap_err();
        let mut right = FactStore::new(&arena);
        right.insert(&arena, id, second).unwrap();
        let error_right = right.insert(&arena, id, first).unwrap_err();
        assert_eq!(format!("{error_left:?}"), format!("{error_right:?}"));

        let range_a = TrustedIndexRange::new(0, 4).unwrap();
        let range_b = TrustedIndexRange::new(0, 8).unwrap();
        let mut ranges = FactStore::new(&arena);
        ranges.declare_trusted_index_range(&arena, id, range_a).unwrap();
        assert!(ranges.declare_trusted_index_range(&arena, id, range_b).is_err());
        ranges.finalize_ranges();
        assert!(ranges.declare_trusted_index_range(&arena, id, range_a).is_err());
    }

    #[test]
    fn equal_range_repeats_and_foreign_facts_fail_closed() {
        let mut arena = ExprArena::new();
        let id = scalar(&mut arena);
        let mut facts = FactStore::new(&arena);
        facts
            .declare_trusted_index_range(&arena, id, TrustedIndexRange::new(2, 2).unwrap())
            .unwrap();
        facts
            .declare_trusted_index_range(&arena, id, TrustedIndexRange::new(2, 2).unwrap())
            .unwrap();
        assert_eq!(
            facts.trusted_index_range(id).unwrap(),
            TrustedIndexRange { minimum: 2, maximum_exclusive: 2 }
        );
        let mut foreign_arena = ExprArena::new();
        let foreign = scalar(&mut foreign_arena);
        assert!(matches!(facts.facts(foreign), Err(FactError::ForeignExpression { .. })));
    }

    #[test]
    fn insert_requires_exact_arena_type_for_every_fact_variant() {
        let mut arena = ExprArena::new();
        let int_id = scalar(&mut arena);
        let mut facts = FactStore::new(&arena);

        let wrong_scalar = ScalarFacts::new(ResolvedValueType::Bool).unwrap();
        assert!(matches!(
            facts.insert(&arena, int_id, ValueFacts::Scalar(wrong_scalar)),
            Err(FactError::FactTypeMismatch { .. })
        ));

        let scalar_matrix = ValueFacts::Scalar(ScalarFacts {
            value_type: ResolvedValueType::Matrix(matrix_type()),
            coefficient_bound: NumericContract::Missing,
        });
        assert!(matches!(
            facts.insert(&arena, int_id, scalar_matrix),
            Err(FactError::InvalidFactVariant { .. })
        ));

        let wrong_trapdoor = ValueFacts::Trapdoor(TrapdoorFacts {
            coefficient_bound: NumericContract::Missing,
            descriptor: "trapdoor".to_owned(),
            paired_public_event: super::super::arena::SampleEventId(1),
            paired_public_output_role: "value".to_owned(),
        });
        assert!(matches!(
            facts.insert(&arena, int_id, wrong_trapdoor),
            Err(FactError::FactTypeMismatch { .. })
        ));

        let index = ValueFacts::Index(IndexFacts { range: None });
        facts.insert(&arena, int_id, index).unwrap();
    }

    #[test]
    fn matrix_facts_require_the_exact_matrix_type() {
        let mut arena = ExprArena::new();
        let (id, matrix_type) = matrix(&mut arena);
        let mut facts = FactStore::new(&arena);
        let metadata = MatrixMetadata::new(MatrixLayout::row_major(2, 2));
        let exact = ValueFacts::Matrix(MatrixFacts::new(matrix_type.clone(), metadata.clone()));
        facts.insert(&arena, id, exact).unwrap();

        let different = ResolvedMatrixType::new(BigUint::from(17_u8), 8, 1, 4).unwrap();
        let wrong = ValueFacts::Matrix(MatrixFacts::new(different, metadata));
        assert!(matches!(facts.insert(&arena, id, wrong), Err(FactError::FactTypeMismatch { .. })));
    }

    #[test]
    fn ranges_accept_only_arena_integer_expressions() {
        let mut arena = ExprArena::new();
        let int_id = scalar(&mut arena);
        let bool_id = boolean(&mut arena);
        let range = TrustedIndexRange::new(0, 4).unwrap();
        let mut facts = FactStore::new(&arena);
        facts.declare_trusted_index_range(&arena, int_id, range).unwrap();
        assert!(matches!(
            facts.declare_trusted_index_range(&arena, bool_id, range),
            Err(FactError::RangeRequiresInteger { actual: ResolvedValueType::Bool, .. })
        ));
    }

    #[test]
    fn facts_reject_binder_open_expressions_and_ranges() {
        let mut arena = ExprArena::new();
        let argument = arena.intern_argument(0, ResolvedValueType::Int).unwrap();
        let scalar_facts = ValueFacts::Scalar(ScalarFacts::new(ResolvedValueType::Int).unwrap());
        let mut facts = FactStore::new(&arena);
        assert!(matches!(
            facts.insert(&arena, argument, scalar_facts),
            Err(FactError::BinderOpenExpression { id }) if id == argument
        ));
        assert!(matches!(
            facts.declare_trusted_index_range(
                &arena,
                argument,
                TrustedIndexRange::new(0, 4).unwrap(),
            ),
            Err(FactError::BinderOpenExpression { id }) if id == argument
        ));
        // Closed expressions remain valid in the shared fact table.
        let closed = scalar(&mut arena);
        facts
            .insert(
                &arena,
                closed,
                ValueFacts::Scalar(ScalarFacts::new(ResolvedValueType::Int).unwrap()),
            )
            .unwrap();
    }

    #[test]
    fn checked_operations_reject_foreign_arena_and_invalid_slots() {
        let mut arena = ExprArena::new();
        let id = scalar(&mut arena);
        let mut foreign_arena = ExprArena::new();
        let foreign_id = scalar(&mut foreign_arena);
        let mut facts = FactStore::new(&arena);
        let scalar_facts = ValueFacts::Scalar(ScalarFacts::new(ResolvedValueType::Int).unwrap());
        assert!(matches!(
            facts.insert(&arena, foreign_id, scalar_facts.clone()),
            Err(FactError::ForeignExpression { .. })
        ));
        assert!(matches!(
            facts.insert(&foreign_arena, id, scalar_facts),
            Err(FactError::ForeignExpression { .. })
        ));
        let invalid = ExprId::new(arena.token(), u32::MAX);
        assert!(matches!(
            facts.declare_trusted_index_range(
                &arena,
                invalid,
                TrustedIndexRange::new(0, 1).unwrap()
            ),
            Err(FactError::InvalidExpression { .. })
        ));
    }
}
