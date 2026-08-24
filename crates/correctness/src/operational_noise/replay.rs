//! Opt-in exact replay for the additive normal-form kernel.
//!
//! The ordinary normalizer does not call this module.  Its API is deliberately bounded to atom,
//! add, subtract, and negate so unsupported production operations fail closed at this boundary.

use super::{
    arena::{ArenaToken, ScopedExprId, ValueProgramId},
    facts::{CoefficientBound, NumericContract},
    monomial::{MonomialArena, MonomialError, MonomialId},
    normal_form::{BoundedSummary, PolynomialNF},
};
use num_bigint::BigInt;
use num_traits::Zero;
use std::{collections::BTreeMap, sync::Arc};
use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplayRule {
    Add,
    Subtract,
    Negate,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CoefficientMerge {
    pub monomial: MonomialId,
    pub left: BigInt,
    pub signed_right: BigInt,
    pub result: BigInt,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReplayEvent {
    pub site: ScopedExprId,
    pub rule: ReplayRule,
    pub predecessors: Box<[Arc<PolynomialNF>]>,
    pub merges: Box<[CoefficientMerge]>,
    pub cancellations: Box<[MonomialId]>,
    pub survivors: Box<[MonomialId]>,
    pub result: Arc<PolynomialNF>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ReplayMetrics {
    pub event_count: usize,
    pub merge_count: usize,
    pub cancellation_count: usize,
    pub survivor_count: usize,
    pub logical_payload_terms: usize,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ReplayTrace {
    pub events: Vec<ReplayEvent>,
    pub metrics: ReplayMetrics,
}

pub(crate) trait ReplaySink {
    const ENABLED: bool;

    fn record(&mut self, build: impl FnOnce() -> ReplayEvent);
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct NoReplay;

impl ReplaySink for NoReplay {
    const ENABLED: bool = false;

    fn record(&mut self, _build: impl FnOnce() -> ReplayEvent) {}
}

impl ReplaySink for ReplayTrace {
    const ENABLED: bool = true;

    fn record(&mut self, build: impl FnOnce() -> ReplayEvent) {
        let event = build();
        self.metrics.event_count = self.metrics.event_count.saturating_add(1);
        self.metrics.merge_count = self.metrics.merge_count.saturating_add(event.merges.len());
        self.metrics.cancellation_count =
            self.metrics.cancellation_count.saturating_add(event.cancellations.len());
        self.metrics.survivor_count =
            self.metrics.survivor_count.saturating_add(event.survivors.len());
        self.metrics.logical_payload_terms =
            self.metrics.logical_payload_terms.saturating_add(event.result.term_count());
        self.events.push(event);
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub(crate) enum ReplayError {
    #[error("replay site belongs to scope {actual:?}, expected {expected:?}")]
    ScopeMismatch { expected: ValueProgramId, actual: ValueProgramId },
    #[error("replay monomial belongs to arena {actual:?}, expected {expected:?}")]
    ArenaMismatch { expected: ArenaToken, actual: ArenaToken },
    #[error("replay monomial is invalid: {0}")]
    Monomial(#[from] MonomialError),
    #[error("replay term has a nonzero coefficient but no factors: {monomial:?}")]
    NonzeroFactorlessTerm { monomial: MonomialId },
    #[error("replay requires an exact predecessor; bounded summary was present")]
    FiniteSummary,
    #[error("replay predecessor is missing")]
    MissingPredecessor,
    #[error("replay result is missing")]
    MissingResult,
    #[error("replay operation is unsupported: {operation}")]
    UnsupportedOperation { operation: &'static str },
}

pub(crate) struct ReplayKernel<'a> {
    monomials: &'a MonomialArena,
    scope: ValueProgramId,
}

impl<'a> ReplayKernel<'a> {
    pub(crate) fn new(monomials: &'a MonomialArena) -> Self {
        Self { monomials, scope: monomials.scope() }
    }

    pub(crate) fn atom<S: ReplaySink>(
        &self,
        site: ScopedExprId,
        value: Option<Arc<PolynomialNF>>,
        _sink: &mut S,
    ) -> Result<Arc<PolynomialNF>, ReplayError> {
        self.check_site(site)?;
        let value = value.ok_or(ReplayError::MissingPredecessor)?;
        self.validate_nf(&value)?;
        Ok(value)
    }

    pub(crate) fn add<S: ReplaySink>(
        &self,
        site: ScopedExprId,
        left: Option<Arc<PolynomialNF>>,
        right: Option<Arc<PolynomialNF>>,
        sink: &mut S,
    ) -> Result<Arc<PolynomialNF>, ReplayError> {
        self.binary(site, ReplayRule::Add, left, right, sink)
    }

    pub(crate) fn subtract<S: ReplaySink>(
        &self,
        site: ScopedExprId,
        left: Option<Arc<PolynomialNF>>,
        right: Option<Arc<PolynomialNF>>,
        sink: &mut S,
    ) -> Result<Arc<PolynomialNF>, ReplayError> {
        self.binary(site, ReplayRule::Subtract, left, right, sink)
    }

    pub(crate) fn negate<S: ReplaySink>(
        &self,
        site: ScopedExprId,
        value: Option<Arc<PolynomialNF>>,
        sink: &mut S,
    ) -> Result<Arc<PolynomialNF>, ReplayError> {
        self.check_site(site)?;
        let predecessor = value.ok_or(ReplayError::MissingPredecessor)?;
        self.validate_nf(&predecessor)?;
        let mut terms = BTreeMap::new();
        for (&monomial, coefficient) in &predecessor.exact_terms {
            let coefficient = -coefficient.clone();
            if !coefficient.is_zero() {
                terms.insert(monomial, coefficient);
            }
        }
        let result =
            Arc::new(PolynomialNF { exact_terms: terms, bounded_summary: BoundedSummary::zero() });
        self.validate_nf(&result)?;
        let survivors = S::ENABLED.then(|| result.exact_terms.keys().copied().collect());
        self.record_event(
            site,
            ReplayRule::Negate,
            || vec![predecessor].into_boxed_slice(),
            None,
            None,
            survivors,
            Some(result),
            sink,
        )
    }

    pub(crate) fn unsupported<S: ReplaySink>(
        &self,
        _site: ScopedExprId,
        _sink: &mut S,
        operation: &'static str,
    ) -> Result<Arc<PolynomialNF>, ReplayError> {
        Err(ReplayError::UnsupportedOperation { operation })
    }

    fn binary<S: ReplaySink>(
        &self,
        site: ScopedExprId,
        rule: ReplayRule,
        left: Option<Arc<PolynomialNF>>,
        right: Option<Arc<PolynomialNF>>,
        sink: &mut S,
    ) -> Result<Arc<PolynomialNF>, ReplayError> {
        self.check_site(site)?;
        let left = left.ok_or(ReplayError::MissingPredecessor)?;
        let right = right.ok_or(ReplayError::MissingPredecessor)?;
        self.validate_nf(&left)?;
        self.validate_nf(&right)?;
        let mut terms = left.exact_terms.clone();
        let mut merges = S::ENABLED.then(Vec::new);
        let mut cancellations = S::ENABLED.then(Vec::new);
        for (&monomial, coefficient) in &right.exact_terms {
            let signed_right = match rule {
                ReplayRule::Add => coefficient.clone(),
                ReplayRule::Subtract => -coefficient.clone(),
                ReplayRule::Negate => unreachable!("negate is unary"),
            };
            if let Some(left) = terms.get(&monomial) {
                let result = left.clone() + signed_right.clone();
                if let Some(merges) = &mut merges {
                    merges.push(CoefficientMerge {
                        monomial,
                        left: left.clone(),
                        signed_right: signed_right.clone(),
                        result: result.clone(),
                    });
                }
                if result.is_zero() {
                    terms.remove(&monomial);
                    if let Some(cancellations) = &mut cancellations {
                        cancellations.push(monomial);
                    }
                } else {
                    terms.insert(monomial, result);
                }
            } else {
                terms.insert(monomial, signed_right);
            }
        }
        let result =
            Arc::new(PolynomialNF { exact_terms: terms, bounded_summary: BoundedSummary::zero() });
        self.validate_nf(&result)?;
        let survivors = S::ENABLED.then(|| result.exact_terms.keys().copied().collect());
        self.record_event(
            site,
            rule,
            || vec![left, right].into_boxed_slice(),
            merges,
            cancellations,
            survivors,
            Some(result),
            sink,
        )
    }

    fn record_event<S: ReplaySink>(
        &self,
        site: ScopedExprId,
        rule: ReplayRule,
        predecessors: impl FnOnce() -> Box<[Arc<PolynomialNF>]>,
        merges: Option<Vec<CoefficientMerge>>,
        cancellations: Option<Vec<MonomialId>>,
        survivors: Option<Vec<MonomialId>>,
        result: Option<Arc<PolynomialNF>>,
        sink: &mut S,
    ) -> Result<Arc<PolynomialNF>, ReplayError> {
        let result = result.ok_or(ReplayError::MissingResult)?;
        self.validate_nf(&result)?;
        if S::ENABLED {
            sink.record(|| ReplayEvent {
                site,
                rule,
                predecessors: predecessors(),
                merges: merges.unwrap_or_default().into_boxed_slice(),
                cancellations: cancellations.unwrap_or_default().into_boxed_slice(),
                survivors: survivors.unwrap_or_default().into_boxed_slice(),
                result: result.clone(),
            });
        }
        Ok(result)
    }

    fn check_site(&self, site: ScopedExprId) -> Result<(), ReplayError> {
        if site.program() != self.scope {
            return Err(ReplayError::ScopeMismatch { expected: self.scope, actual: site.program() });
        }
        Ok(())
    }

    fn validate_nf(&self, value: &PolynomialNF) -> Result<(), ReplayError> {
        if value.bounded_summary.coefficient_bound() !=
            NumericContract::Known(CoefficientBound::ExactZero)
        {
            return Err(ReplayError::FiniteSummary);
        }
        for (&monomial, coefficient) in &value.exact_terms {
            if monomial.arena() != self.monomials.token() {
                return Err(ReplayError::ArenaMismatch {
                    expected: self.monomials.token(),
                    actual: monomial.arena(),
                });
            }
            let descriptor = self.monomials.descriptor(monomial)?;
            for factor in descriptor.central_factors.iter().chain(&descriptor.ordered_factors) {
                if factor.program() != self.scope {
                    return Err(ReplayError::ScopeMismatch {
                        expected: self.scope,
                        actual: factor.program(),
                    });
                }
            }
            if !coefficient.is_zero() &&
                descriptor.central_factors.is_empty() &&
                descriptor.ordered_factors.is_empty()
            {
                return Err(ReplayError::NonzeroFactorlessTerm { monomial });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::{
        arena::{
            ExprArena, MatrixOperation, ProgramSignature, ResolvedMatrixType, ResolvedValueType,
            SampleEventId, SamplerOperation, ValueOperator,
        },
        monomial::MonomialArena,
        program::ProgramArena,
    };
    use num_bigint::BigUint;

    fn fixture() -> (MonomialArena, ScopedExprId, MonomialId, MonomialId, MonomialId) {
        let mut expressions = ExprArena::new();
        let output = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 1).unwrap();
        let first = expressions
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(1),
                    operation: SamplerOperation::UniformResidue { output: output.clone() },
                },
                Box::new([]),
            )
            .unwrap();
        let second = expressions
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(2),
                    operation: SamplerOperation::UniformResidue { output: output.clone() },
                },
                Box::new([]),
            )
            .unwrap();
        let root = expressions
            .intern(ValueOperator::Matrix(MatrixOperation::Add), Box::new([first, second]))
            .unwrap();
        let mut programs = ProgramArena::new();
        let scope = programs
            .finalize(
                &mut expressions,
                ProgramSignature {
                    inputs: Box::new([]),
                    output: ResolvedValueType::Matrix(output),
                },
                root,
            )
            .unwrap();
        let site = programs.scoped(&expressions, scope, first).unwrap();
        let second_site = programs.scoped(&expressions, scope, second).unwrap();
        let mut monomials = MonomialArena::new(&expressions, &programs, scope).unwrap();
        let first_monomial = monomials.intern(&expressions, &programs, &[], &[site]).unwrap();
        let second_monomial =
            monomials.intern(&expressions, &programs, &[], &[second_site]).unwrap();
        let factorless = monomials.intern(&expressions, &programs, &[], &[]).unwrap();
        (monomials, site, first_monomial, second_monomial, factorless)
    }

    fn polynomial(monomial: MonomialId, coefficient: i64) -> Arc<PolynomialNF> {
        Arc::new(PolynomialNF {
            exact_terms: [(monomial, BigInt::from(coefficient))].into_iter().collect(),
            bounded_summary: BoundedSummary::zero(),
        })
    }

    #[test]
    fn exact_add_subtract_cancel_and_record_merge() {
        let (monomials, site, monomial, _, _) = fixture();
        let kernel = ReplayKernel::new(&monomials);
        let mut trace = ReplayTrace::default();
        let result = kernel
            .add(site, Some(polynomial(monomial, 1)), Some(polynomial(monomial, -1)), &mut trace)
            .unwrap();
        assert!(result.is_zero());
        assert_eq!(trace.metrics.cancellation_count, 1);
        assert_eq!(trace.events[0].merges[0].result, BigInt::from(0));
        let result = kernel
            .subtract(
                site,
                Some(polynomial(monomial, 1)),
                Some(polynomial(monomial, 1)),
                &mut trace,
            )
            .unwrap();
        assert!(result.is_zero());
    }

    #[test]
    fn negate_survives_and_disabled_sink_records_nothing() {
        let (monomials, site, monomial, _, _) = fixture();
        let kernel = ReplayKernel::new(&monomials);
        let mut trace = ReplayTrace::default();
        let result = kernel.negate(site, Some(polynomial(monomial, 3)), &mut trace).unwrap();
        assert_eq!(result.exact_terms.get(&monomial), Some(&BigInt::from(-3_i8)));
        assert_eq!(trace.metrics.survivor_count, 1);
        let mut disabled = NoReplay;
        kernel
            .add(site, Some(polynomial(monomial, 1)), Some(polynomial(monomial, 2)), &mut disabled)
            .unwrap();
    }

    #[test]
    fn unequal_ordered_lists_and_factorless_terms_fail_closed() {
        let (monomials, site, first, second, factorless) = fixture();
        let kernel = ReplayKernel::new(&monomials);
        let mut trace = ReplayTrace::default();
        let result = kernel
            .add(site, Some(polynomial(first, 1)), Some(polynomial(second, 1)), &mut trace)
            .unwrap();
        assert_eq!(result.term_count(), 2);
        assert!(trace.events[0].merges.is_empty());
        assert_eq!(trace.metrics.survivor_count, 2);
        let error = kernel.atom(site, Some(polynomial(factorless, 1)), &mut NoReplay).unwrap_err();
        assert!(matches!(error, ReplayError::NonzeroFactorlessTerm { .. }));
        assert!(matches!(
            kernel.unsupported(site, &mut trace, "multiply"),
            Err(ReplayError::UnsupportedOperation { operation: "multiply" })
        ));
    }

    #[test]
    fn missing_predecessor_and_result_fail_closed() {
        let (monomials, site, _, _, _) = fixture();
        let kernel = ReplayKernel::new(&monomials);
        assert!(matches!(
            kernel.atom(site, None, &mut NoReplay),
            Err(ReplayError::MissingPredecessor)
        ));
        let error = kernel.record_event(
            site,
            ReplayRule::Add,
            || Box::new([]),
            None,
            None,
            None,
            None,
            &mut NoReplay,
        );
        assert!(matches!(error, Err(ReplayError::MissingResult)));
    }
}
