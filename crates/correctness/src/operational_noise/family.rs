//! Compact family coverage and numeric sequential-loop recurrence support.
//!
//! This module deliberately owns neither graph-wire memoization nor integer
//! analysis.  [`GraphLowerer`](super::lower::GraphLowerer) supplies one
//! symbolic element at a time, and the caller supplies all resource and error
//! policy through [`RecurrenceControl`].

use super::{
    analysis::IntegerDomain,
    identity::{BinderKey, ResolvedIntExpr},
    language::MxxLang,
    lower::LoweredInt,
};
use egg::{EGraph, Id};
use mxx_ir_core::types::ConcreteMatrixType;
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};
use std::collections::BTreeMap;

/// The owner of a parallel-loop logical count.  Output ports are deliberately
/// absent: sibling outputs of one loop occurrence share this domain.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LoopDomainKey {
    pub binder: BinderKey,
    pub logical_count: BigUint,
}

/// One authoritative interval over which a symbolic representative is valid.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CoverageBinderDomain {
    pub binder: BinderKey,
    pub minimum: BigInt,
    pub maximum: BigInt,
}

/// The only two compact representations of a matrix family.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FamilyCoverageStorage {
    /// Physical element references present in the Graph IR or manifest.
    ExactStored { elements: Box<[Id]> },
    /// One symbolic representative over every binder in `binder_domains`.
    SharedTemplate {
        domain: LoopDomainKey,
        representative: Id,
        binder_domains: Box<[CoverageBinderDomain]>,
    },
}

/// A family residual together with its single concrete matrix schema.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FamilyLoweringValue {
    pub element_type: ConcreteMatrixType,
    pub storage: FamilyCoverageStorage,
}

/// Closed, local family failures.  The lowering boundary maps these to its
/// site-bearing `LowerError`; this module never invents Graph-IR expressions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FamilyCoverageError {
    EmptyExactStorage,
    InvalidBinderDomain { minimum: BigInt, maximum: BigInt },
    SharedCountMismatch { count: BigUint, domain_size: BigUint },
    StaticIndexOutOfRange { index: BigInt, count: BigUint },
    DynamicIndexOutOfRange { minimum: BigInt, maximum: BigInt, count: BigUint },
    MatrixTypeMismatch { expected: ConcreteMatrixType, actual: ConcreteMatrixType },
    StorageMismatch,
    SelectorCaseCountMismatch { expected: usize, actual: usize },
}

impl FamilyLoweringValue {
    /// Validates storage invariants without materializing any logical lane.
    pub fn validate(&self) -> Result<(), FamilyCoverageError> {
        match &self.storage {
            FamilyCoverageStorage::ExactStored { elements } => {
                if elements.is_empty() {
                    return Err(FamilyCoverageError::EmptyExactStorage);
                }
            }
            FamilyCoverageStorage::SharedTemplate { domain, binder_domains, .. } => {
                let mut owner_size = None;
                for binder_domain in binder_domains.iter() {
                    if binder_domain.maximum < binder_domain.minimum {
                        return Err(FamilyCoverageError::InvalidBinderDomain {
                            minimum: binder_domain.minimum.clone(),
                            maximum: binder_domain.maximum.clone(),
                        });
                    }
                    let width = (&binder_domain.maximum - &binder_domain.minimum + BigInt::one())
                        .to_biguint()
                        .expect("validated nonnegative binder-domain width");
                    if binder_domain.binder == domain.binder {
                        owner_size = Some(width);
                    }
                }
                if owner_size.as_ref() != Some(&domain.logical_count) {
                    return Err(FamilyCoverageError::SharedCountMismatch {
                        count: domain.logical_count.clone(),
                        domain_size: owner_size.unwrap_or_else(BigUint::zero),
                    });
                }
            }
        }
        Ok(())
    }

    pub fn exact_elements(&self) -> Option<&[Id]> {
        match &self.storage {
            FamilyCoverageStorage::ExactStored { elements } => Some(elements),
            FamilyCoverageStorage::SharedTemplate { .. } => None,
        }
    }

    pub fn shared_template(&self) -> Option<(&LoopDomainKey, Id, &[CoverageBinderDomain])> {
        match &self.storage {
            FamilyCoverageStorage::ExactStored { .. } => None,
            FamilyCoverageStorage::SharedTemplate { domain, representative, binder_domains } => {
                Some((domain, *representative, binder_domains))
            }
        }
    }
}

/// Validates the analysis-owned integer domain against a family count.
pub fn validate_family_index(
    index: &IntegerDomain,
    count: &BigUint,
) -> Result<(), FamilyCoverageError> {
    let interval = index.interval().map_err(|_| FamilyCoverageError::DynamicIndexOutOfRange {
        minimum: BigInt::from(-1),
        maximum: BigInt::from(-1),
        count: count.clone(),
    })?;
    let upper = BigInt::from(count.clone());
    if interval.minimum < BigInt::zero() || interval.maximum >= upper {
        return Err(FamilyCoverageError::DynamicIndexOutOfRange {
            minimum: interval.minimum,
            maximum: interval.maximum,
            count: count.clone(),
        });
    }
    Ok(())
}

/// Resolves a static physical element.  Only an exact constant identity is a
/// static index; all other values must use [`dynamic_get`].
pub fn static_get(
    family: &FamilyLoweringValue,
    index: &LoweredInt,
) -> Result<Option<Id>, FamilyCoverageError> {
    let Some(ResolvedIntExpr::Const(value)) = index.stable_identity.as_ref() else {
        return Ok(None);
    };
    let FamilyCoverageStorage::ExactStored { elements } = &family.storage else {
        return Ok(None);
    };
    let Some(offset) = value.to_usize() else {
        return Err(FamilyCoverageError::StaticIndexOutOfRange {
            index: value.clone(),
            count: BigUint::from(elements.len()),
        });
    };
    elements.get(offset).copied().map(Some).ok_or_else(|| {
        FamilyCoverageError::StaticIndexOutOfRange {
            index: value.clone(),
            count: BigUint::from(elements.len()),
        }
    })
}

/// Builds one ordered physical `Switch`.  Its work is linear in physical
/// cases and never in a symbolic template's logical count.
pub fn dynamic_get<A>(
    egraph: &mut EGraph<MxxLang, A>,
    family: &FamilyLoweringValue,
    selector: Id,
) -> Result<Id, FamilyCoverageError>
where
    A: egg::Analysis<MxxLang>,
{
    let FamilyCoverageStorage::ExactStored { elements } = &family.storage else {
        return Err(FamilyCoverageError::StorageMismatch);
    };
    if elements.is_empty() {
        return Err(FamilyCoverageError::EmptyExactStorage);
    }
    let mut children = Vec::with_capacity(elements.len() + 1);
    children.push(selector);
    children.extend(elements.iter().copied());
    Ok(egraph.add(MxxLang::Switch(children.into_boxed_slice())))
}

/// Resolves an element without enumerating a shared template.  The lowerer is
/// responsible for binding the same symbolic index into the representative.
pub fn shared_element(
    family: &FamilyLoweringValue,
) -> Result<(Id, &LoopDomainKey, &[CoverageBinderDomain]), FamilyCoverageError> {
    let Some((domain, representative, binders)) = family.shared_template() else {
        return Err(FamilyCoverageError::StorageMismatch);
    };
    Ok((representative, domain, binders))
}

/// Selects a compact family.  Exact storage evaluates only stored references;
/// template storage combines representatives pointwise under one selector.
pub fn select_family<A>(
    egraph: &mut EGraph<MxxLang, A>,
    selector: Id,
    cases: &[FamilyLoweringValue],
) -> Result<FamilyLoweringValue, FamilyCoverageError>
where
    A: egg::Analysis<MxxLang>,
{
    let Some(first) = cases.first() else {
        return Err(FamilyCoverageError::SelectorCaseCountMismatch { expected: 1, actual: 0 });
    };
    first.validate()?;
    for case in &cases[1..] {
        case.validate()?;
        if case.element_type != first.element_type {
            return Err(FamilyCoverageError::MatrixTypeMismatch {
                expected: first.element_type.clone(),
                actual: case.element_type.clone(),
            });
        }
    }

    match &first.storage {
        FamilyCoverageStorage::ExactStored { elements } => {
            let width = elements.len();
            for case in cases {
                match &case.storage {
                    FamilyCoverageStorage::ExactStored { elements } if elements.len() == width => {}
                    FamilyCoverageStorage::ExactStored { elements } => {
                        return Err(FamilyCoverageError::SelectorCaseCountMismatch {
                            expected: width,
                            actual: elements.len(),
                        });
                    }
                    FamilyCoverageStorage::SharedTemplate { .. } => {
                        return Err(FamilyCoverageError::StorageMismatch);
                    }
                }
            }
            let mut selected = Vec::with_capacity(width);
            for lane in 0..width {
                let mut children = Vec::with_capacity(cases.len() + 1);
                children.push(selector);
                for case in cases {
                    let FamilyCoverageStorage::ExactStored { elements } = &case.storage else {
                        unreachable!("storage shape was checked before e-graph mutation");
                    };
                    children.push(elements[lane]);
                }
                selected.push(egraph.add(MxxLang::Switch(children.into_boxed_slice())));
            }
            Ok(FamilyLoweringValue {
                element_type: first.element_type.clone(),
                storage: FamilyCoverageStorage::ExactStored {
                    elements: selected.into_boxed_slice(),
                },
            })
        }
        FamilyCoverageStorage::SharedTemplate { domain, binder_domains, .. } => {
            let mut children = Vec::with_capacity(cases.len() + 1);
            children.push(selector);
            for case in cases {
                let FamilyCoverageStorage::SharedTemplate {
                    domain: case_domain,
                    representative,
                    binder_domains: case_binders,
                } = &case.storage
                else {
                    return Err(FamilyCoverageError::StorageMismatch);
                };
                if case_domain != domain || case_binders != binder_domains {
                    return Err(FamilyCoverageError::StorageMismatch);
                }
                children.push(*representative);
            }
            Ok(FamilyLoweringValue {
                element_type: first.element_type.clone(),
                storage: FamilyCoverageStorage::SharedTemplate {
                    domain: domain.clone(),
                    representative: egraph.add(MxxLang::Switch(children.into_boxed_slice())),
                    binder_domains: binder_domains.clone(),
                },
            })
        }
    }
}

/// A scalar recurrence expression extracted from a fixed-size sequential body.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum RecurrenceExpr {
    Const(BigUint),
    SignedAffineCutoff { constant: BigInt, iteration_coefficient: BigInt },
    Previous(usize),
    Iteration,
    Add(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Max(Box<[Self]>),
}

/// A simultaneous, fixed-state numeric sequential transition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VectorRecurrence {
    pub initial: Box<[BigUint]>,
    pub transition: Box<[RecurrenceExpr]>,
    pub count: BigUint,
}

/// Failures whose site-bearing public error is owned by the caller.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RecurrenceFailure {
    ArityMismatch { expected: usize, actual: usize },
    PreviousOutOfRange { index: usize, state_size: usize },
    NegativeCutoff { cutoff: BigInt },
    StepLimit { limit: BigUint, count: BigUint, transition_nodes: u64 },
    IntegerBits { limit: BigUint, observed: BigUint, operation: &'static str },
}

/// Callbacks into the single simulation-wide deadline and resource owner.
pub struct RecurrenceControl<'a, E> {
    pub recurrence_step_limit: &'a BigUint,
    pub max_integer_bits: &'a BigUint,
    pub check_deadline: &'a mut dyn FnMut() -> Result<(), E>,
    /// Charges every recurrence-owned container to the caller's one job-wide
    /// cumulative allocation budget.  This is deliberately a callback rather
    /// than a recurrence-local counter, so nested loop bodies cannot reset it.
    pub reserve_owned_elements: &'a mut dyn FnMut(usize) -> Result<(), E>,
    pub failure: &'a mut dyn FnMut(RecurrenceFailure) -> E,
}

impl VectorRecurrence {
    /// Evaluates the general O(C*T) path, or the affine O(S^3 log C) fast
    /// path when the whole transition has the required nonnegative form.
    pub fn evaluate<E>(&self, control: &mut RecurrenceControl<'_, E>) -> Result<Box<[BigUint]>, E> {
        if self.initial.len() != self.transition.len() {
            return Err((control.failure)(RecurrenceFailure::ArityMismatch {
                expected: self.initial.len(),
                actual: self.transition.len(),
            }));
        }
        self.validate_values(control, &self.initial, "initial-state")?;
        if self.count.is_zero() {
            return Ok(self.initial.clone());
        }
        if let Some(rows) = self.affine_rows() {
            return self.evaluate_affine(rows, control);
        }
        self.evaluate_general(control)
    }

    fn evaluate_general<E>(
        &self,
        control: &mut RecurrenceControl<'_, E>,
    ) -> Result<Box<[BigUint]>, E> {
        let transition_nodes = self.transition_node_count();
        let work = &self.count * BigUint::from(transition_nodes);
        if work > *control.recurrence_step_limit {
            return Err((control.failure)(RecurrenceFailure::StepLimit {
                limit: control.recurrence_step_limit.clone(),
                count: self.count.clone(),
                transition_nodes,
            }));
        }
        (control.reserve_owned_elements)(self.initial.len())?;
        let mut state = self.initial.to_vec();
        let mut iteration = BigUint::zero();
        while iteration < self.count {
            (control.check_deadline)()?;
            (control.reserve_owned_elements)(self.transition.len())?;
            let mut memo = BTreeMap::new();
            let mut next = Vec::with_capacity(self.transition.len());
            for expression in self.transition.iter() {
                next.push(evaluate_expression(expression, &state, &iteration, &mut memo, control)?);
            }
            state = next;
            iteration += BigUint::one();
        }
        Ok(state.into_boxed_slice())
    }

    fn evaluate_affine<E>(
        &self,
        rows: Vec<AffineRow>,
        control: &mut RecurrenceControl<'_, E>,
    ) -> Result<Box<[BigUint]>, E> {
        let state_size = self.initial.len();
        let dimension = state_size + 2;
        (control.reserve_owned_elements)(dimension.checked_mul(dimension).ok_or_else(|| {
            (control.failure)(RecurrenceFailure::IntegerBits {
                limit: control.max_integer_bits.clone(),
                observed: BigUint::from(usize::BITS),
                operation: "affine recurrence matrix dimension",
            })
        })?)?;
        let mut matrix = vec![vec![BigUint::zero(); dimension]; dimension];
        for (row_index, row) in rows.iter().enumerate() {
            matrix[row_index][..state_size].clone_from_slice(&row.previous);
            matrix[row_index][state_size] = row.iteration.clone();
            matrix[row_index][state_size + 1] = row.constant.clone();
        }
        matrix[state_size][state_size] = BigUint::one();
        matrix[state_size + 1][state_size + 1] = BigUint::one();

        let power = matrix_power(matrix, &self.count, control)?;
        let mut input = self.initial.to_vec();
        input.push(BigUint::zero());
        input.push(BigUint::one());
        let output = matrix_vector_product(&power, &input, control)?;
        Ok(output[..state_size].to_vec().into_boxed_slice())
    }

    fn affine_rows(&self) -> Option<Vec<AffineRow>> {
        self.transition
            .iter()
            .map(|expression| affine_expression(expression, self.initial.len()))
            .collect()
    }

    fn transition_node_count(&self) -> u64 {
        let mut nodes = BTreeMap::new();
        for expression in self.transition.iter() {
            count_nodes(expression, &mut nodes);
        }
        u64::try_from(nodes.len()).unwrap_or(u64::MAX)
    }

    fn validate_values<E>(
        &self,
        control: &mut RecurrenceControl<'_, E>,
        values: &[BigUint],
        operation: &'static str,
    ) -> Result<(), E> {
        for value in values {
            validate_bits(value, operation, control)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct AffineRow {
    constant: BigUint,
    iteration: BigUint,
    previous: Vec<BigUint>,
}

fn affine_expression(expression: &RecurrenceExpr, state_size: usize) -> Option<AffineRow> {
    match expression {
        RecurrenceExpr::Const(value) => Some(AffineRow {
            constant: value.clone(),
            iteration: BigUint::zero(),
            previous: vec![BigUint::zero(); state_size],
        }),
        RecurrenceExpr::SignedAffineCutoff { constant, iteration_coefficient } => Some(AffineRow {
            constant: constant.to_biguint()?,
            iteration: iteration_coefficient.to_biguint()?,
            previous: vec![BigUint::zero(); state_size],
        }),
        RecurrenceExpr::Previous(index) => {
            if *index >= state_size {
                return None;
            }
            let mut previous = vec![BigUint::zero(); state_size];
            previous[*index] = BigUint::one();
            Some(AffineRow { constant: BigUint::zero(), iteration: BigUint::zero(), previous })
        }
        RecurrenceExpr::Iteration => Some(AffineRow {
            constant: BigUint::zero(),
            iteration: BigUint::one(),
            previous: vec![BigUint::zero(); state_size],
        }),
        RecurrenceExpr::Add(left, right) => {
            let mut left = affine_expression(left, state_size)?;
            let right = affine_expression(right, state_size)?;
            left.constant += right.constant;
            left.iteration += right.iteration;
            for (left, right) in left.previous.iter_mut().zip(right.previous) {
                *left += right;
            }
            Some(left)
        }
        RecurrenceExpr::Mul(left, right) => {
            let left = affine_expression(left, state_size)?;
            let right = affine_expression(right, state_size)?;
            if is_constant_row(&left) {
                Some(scale_row(right, &left.constant))
            } else if is_constant_row(&right) {
                Some(scale_row(left, &right.constant))
            } else {
                None
            }
        }
        RecurrenceExpr::Max(_) => None,
    }
}

fn is_constant_row(row: &AffineRow) -> bool {
    row.iteration.is_zero() && row.previous.iter().all(Zero::is_zero)
}

fn scale_row(mut row: AffineRow, scalar: &BigUint) -> AffineRow {
    row.constant *= scalar;
    row.iteration *= scalar;
    for coefficient in &mut row.previous {
        *coefficient *= scalar;
    }
    row
}

fn count_nodes<'a>(expression: &'a RecurrenceExpr, nodes: &mut BTreeMap<&'a RecurrenceExpr, ()>) {
    let mut work = vec![expression];
    while let Some(expression) = work.pop() {
        if nodes.insert(expression, ()).is_some() {
            continue;
        }
        match expression {
            RecurrenceExpr::Add(left, right) | RecurrenceExpr::Mul(left, right) => {
                work.push(left);
                work.push(right);
            }
            RecurrenceExpr::Max(children) => work.extend(children.iter()),
            RecurrenceExpr::Const(_) |
            RecurrenceExpr::SignedAffineCutoff { .. } |
            RecurrenceExpr::Previous(_) |
            RecurrenceExpr::Iteration => {}
        }
    }
}

fn evaluate_expression<E>(
    expression: &RecurrenceExpr,
    state: &[BigUint],
    iteration: &BigUint,
    memo: &mut BTreeMap<RecurrenceExpr, BigUint>,
    control: &mut RecurrenceControl<'_, E>,
) -> Result<BigUint, E> {
    enum Work<'a> {
        Enter(&'a RecurrenceExpr),
        Finish(&'a RecurrenceExpr),
    }
    if let Some(value) = memo.get(expression) {
        return Ok(value.clone());
    }
    let mut scheduled = BTreeMap::<&RecurrenceExpr, ()>::new();
    let mut work = vec![Work::Enter(expression)];
    while let Some(item) = work.pop() {
        (control.check_deadline)()?;
        match item {
            Work::Enter(expression) if memo.contains_key(expression) => {}
            Work::Enter(expression) if scheduled.insert(expression, ()).is_some() => {}
            Work::Enter(expression) => {
                work.push(Work::Finish(expression));
                match expression {
                    RecurrenceExpr::Add(left, right) | RecurrenceExpr::Mul(left, right) => {
                        work.push(Work::Enter(right));
                        work.push(Work::Enter(left));
                    }
                    RecurrenceExpr::Max(children) => {
                        for child in children.iter().rev() {
                            work.push(Work::Enter(child));
                        }
                    }
                    RecurrenceExpr::Const(_) |
                    RecurrenceExpr::SignedAffineCutoff { .. } |
                    RecurrenceExpr::Previous(_) |
                    RecurrenceExpr::Iteration => {}
                }
            }
            Work::Finish(expression) => {
                let child =
                    |expression| memo.get(expression).cloned().expect("postorder recurrence child");
                let value = match expression {
                    RecurrenceExpr::Const(value) => value.clone(),
                    RecurrenceExpr::SignedAffineCutoff { constant, iteration_coefficient } => {
                        let value =
                            constant + iteration_coefficient * BigInt::from(iteration.clone());
                        value.to_biguint().ok_or_else(|| {
                            (control.failure)(RecurrenceFailure::NegativeCutoff { cutoff: value })
                        })?
                    }
                    RecurrenceExpr::Previous(index) => {
                        state.get(*index).cloned().ok_or_else(|| {
                            (control.failure)(RecurrenceFailure::PreviousOutOfRange {
                                index: *index,
                                state_size: state.len(),
                            })
                        })?
                    }
                    RecurrenceExpr::Iteration => iteration.clone(),
                    RecurrenceExpr::Add(left, right) => child(left) + child(right),
                    RecurrenceExpr::Mul(left, right) => child(left) * child(right),
                    RecurrenceExpr::Max(children) => {
                        children.iter().fold(BigUint::zero(), |maximum, child_expression| {
                            maximum.max(child(child_expression))
                        })
                    }
                };
                validate_bits(&value, "recurrence-expression", control)?;
                (control.reserve_owned_elements)(1)?;
                memo.insert(expression.clone(), value);
            }
        }
    }
    memo.remove(expression).ok_or_else(|| {
        (control.failure)(RecurrenceFailure::ArityMismatch { expected: 1, actual: 0 })
    })
}

fn matrix_power<E>(
    mut power: Vec<Vec<BigUint>>,
    exponent: &BigUint,
    control: &mut RecurrenceControl<'_, E>,
) -> Result<Vec<Vec<BigUint>>, E> {
    let dimension = power.len();
    let mut result = identity_matrix(dimension);
    let mut remaining = exponent.clone();
    while !remaining.is_zero() {
        (control.check_deadline)()?;
        if (&remaining & BigUint::one()) == BigUint::one() {
            result = matrix_product(&result, &power, control)?;
        }
        remaining >>= 1_usize;
        if !remaining.is_zero() {
            power = matrix_product(&power, &power, control)?;
        }
    }
    Ok(result)
}

fn identity_matrix(dimension: usize) -> Vec<Vec<BigUint>> {
    let mut matrix = vec![vec![BigUint::zero(); dimension]; dimension];
    for (index, row) in matrix.iter_mut().enumerate() {
        row[index] = BigUint::one();
    }
    matrix
}

fn matrix_product<E>(
    left: &[Vec<BigUint>],
    right: &[Vec<BigUint>],
    control: &mut RecurrenceControl<'_, E>,
) -> Result<Vec<Vec<BigUint>>, E> {
    let dimension = left.len();
    let mut output = vec![vec![BigUint::zero(); dimension]; dimension];
    for row in 0..dimension {
        for column in 0..dimension {
            let mut value = BigUint::zero();
            for inner in 0..dimension {
                (control.check_deadline)()?;
                value += &left[row][inner] * &right[inner][column];
                validate_bits(&value, "affine-matrix-product", control)?;
            }
            output[row][column] = value;
        }
    }
    Ok(output)
}

fn matrix_vector_product<E>(
    matrix: &[Vec<BigUint>],
    vector: &[BigUint],
    control: &mut RecurrenceControl<'_, E>,
) -> Result<Vec<BigUint>, E> {
    let mut output = Vec::with_capacity(matrix.len());
    for row in matrix {
        let mut value = BigUint::zero();
        for (coefficient, input) in row.iter().zip(vector) {
            (control.check_deadline)()?;
            value += coefficient * input;
            validate_bits(&value, "affine-matrix-vector", control)?;
        }
        output.push(value);
    }
    Ok(output)
}

fn validate_bits<E>(
    value: &BigUint,
    operation: &'static str,
    control: &mut RecurrenceControl<'_, E>,
) -> Result<(), E> {
    let observed = BigUint::from(value.bits());
    if observed > *control.max_integer_bits {
        return Err((control.failure)(RecurrenceFailure::IntegerBits {
            limit: control.max_integer_bits.clone(),
            observed,
            operation,
        }));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluate(recurrence: &VectorRecurrence) -> Result<Box<[BigUint]>, RecurrenceFailure> {
        let step_limit = BigUint::from(100_u8);
        let bits = BigUint::from(1_000_u16);
        let mut deadline = || Ok::<(), RecurrenceFailure>(());
        let mut reserve = |_| Ok::<(), RecurrenceFailure>(());
        let mut failure = |failure| failure;
        recurrence.evaluate(&mut RecurrenceControl {
            recurrence_step_limit: &step_limit,
            max_integer_bits: &bits,
            check_deadline: &mut deadline,
            reserve_owned_elements: &mut reserve,
            failure: &mut failure,
        })
    }

    #[test]
    fn general_recurrence_uses_simultaneous_previous_state() {
        let recurrence = VectorRecurrence {
            initial: vec![BigUint::from(2_u8), BigUint::from(5_u8)].into_boxed_slice(),
            transition: vec![RecurrenceExpr::Previous(1), RecurrenceExpr::Previous(0)]
                .into_boxed_slice(),
            count: BigUint::from(3_u8),
        };
        assert_eq!(evaluate(&recurrence).unwrap().as_ref(), &[5_u8.into(), 2_u8.into()]);
    }

    #[test]
    fn affine_fast_path_handles_large_count_without_step_budget() {
        let recurrence = VectorRecurrence {
            initial: vec![BigUint::one()].into_boxed_slice(),
            transition: vec![RecurrenceExpr::Mul(
                Box::new(RecurrenceExpr::Const(BigUint::from(2_u8))),
                Box::new(RecurrenceExpr::Previous(0)),
            )]
            .into_boxed_slice(),
            count: BigUint::from(40_u8),
        };
        assert_eq!(evaluate(&recurrence).unwrap()[0], BigUint::one() << 40_usize);
    }

    #[test]
    fn max_forces_the_general_path_and_enforces_step_limit() {
        let recurrence = VectorRecurrence {
            initial: vec![BigUint::one()].into_boxed_slice(),
            transition: vec![RecurrenceExpr::Max(
                vec![RecurrenceExpr::Previous(0), RecurrenceExpr::Const(BigUint::from(2_u8))]
                    .into_boxed_slice(),
            )]
            .into_boxed_slice(),
            count: BigUint::from(101_u8),
        };
        assert!(matches!(evaluate(&recurrence), Err(RecurrenceFailure::StepLimit { .. })));
    }

    #[test]
    fn recurrence_charges_the_shared_allocation_callback() {
        let recurrence = VectorRecurrence {
            initial: vec![BigUint::one()].into_boxed_slice(),
            transition: vec![RecurrenceExpr::Previous(0)].into_boxed_slice(),
            count: BigUint::one(),
        };
        let step_limit = BigUint::from(10_u8);
        let bits = BigUint::from(1_000_u16);
        let mut deadline = || Ok::<(), RecurrenceFailure>(());
        let mut reserve = |_| {
            Err(RecurrenceFailure::StepLimit {
                limit: BigUint::zero(),
                count: BigUint::zero(),
                transition_nodes: 0,
            })
        };
        let mut failure = |failure| failure;
        assert!(matches!(
            recurrence.evaluate(&mut RecurrenceControl {
                recurrence_step_limit: &step_limit,
                max_integer_bits: &bits,
                check_deadline: &mut deadline,
                reserve_owned_elements: &mut reserve,
                failure: &mut failure,
            }),
            Err(RecurrenceFailure::StepLimit { .. })
        ));
    }
}
