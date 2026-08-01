use crate::{
    DependencySet, PolyMatrixNorm, PolyNorm, SimulatorContext, SourceId,
    poly_norm::high_probability_envelope_from_sigma,
};
use bigdecimal::BigDecimal;
use mxx_ir_core::{
    ScopedWireRef,
    expr::{IntExpr, ParamEnv, RealExpr},
    node::ConstantMatrix,
    types::{ConcreteMatrixType, ConcreteWireType},
};
use mxx_ir_symbolic::{
    atom::{
        AssumedMetadata, Atom, AtomClass, AtomId, AtomKind, DeclaredDependencies,
        DeclaredDependencyRef, ExternalSourceKind, SelectionDomainRef, SourceKind,
    },
    elaborate::{DecodeTarget, ElaboratedGraph, SymbolicFamily},
    expression::{IndexRange, SymbolicExprId, SymbolicExprNode},
};
use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
use thiserror::Error;

const METHODOLOGY: &str = "High-probability coefficient envelopes with the existing CLT eligibility rules; not worst-case bounds.";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatrixNoiseReport {
    pub bound: BigDecimal,
    pub rows: usize,
    pub columns: usize,
    pub is_const_poly: bool,
    pub zero_rows: Option<usize>,
    pub dependencies: DependencySet,
    pub clt_ready: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WireNoiseReport {
    pub has_signal: bool,
    pub noise: Option<MatrixNoiseReport>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodeNoiseReport {
    pub target: DecodeTarget,
    pub estimate: WireNoiseReport,
    pub threshold: BigDecimal,
    pub within_threshold: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NoiseReport {
    pub outputs: BTreeMap<String, WireNoiseReport>,
    pub decode_targets: Vec<DecodeNoiseReport>,
    pub methodology: &'static str,
}

#[derive(Debug, Error)]
pub enum SimulationError {
    #[error("wire {0:?} has no symbolic matrix expression")]
    MissingWire(ScopedWireRef),
    #[error("symbolic expression is missing: {0:?}")]
    MissingExpression(SymbolicExprId),
    #[error("atom is missing from the symbolic atom table: {0:?}")]
    MissingAtom(AtomId),
    #[error("bounded external source {kind:?} at {atom:?} has no manifest or assume metadata")]
    UnsupportedExternal { atom: AtomId, kind: ExternalSourceKind },
    #[error("invalid selection: {0}")]
    InvalidSelection(String),
    #[error("invalid numerical metadata at atom {atom:?}: {reason}")]
    InvalidMetadata { atom: AtomId, reason: String },
    #[error("matrix shape mismatch while evaluating symbolic expression")]
    ShapeMismatch,
    #[error("a bounded alternative unexpectedly contains a signal factor")]
    MixedBoundedAlternative,
}

#[derive(Clone)]
struct Estimate {
    signal: bool,
    noise: Option<PolyMatrixNorm>,
}

#[derive(Clone)]
enum AnalysisFactor {
    Signal(ConcreteMatrixType),
    Bounded(PolyMatrixNorm),
    Identity(ConcreteMatrixType),
}

#[derive(Clone)]
struct Alternative {
    coefficient: BigInt,
    factors: Vec<AnalysisFactor>,
}

type Assignment = BTreeMap<SelectionDomainRef, u64>;
type CanonicalAssignment = Vec<(SelectionDomainRef, u64)>;

struct Evaluator<'a> {
    graph: &'a ElaboratedGraph,
    contexts: BTreeMap<usize, Arc<SimulatorContext>>,
    atoms: BTreeMap<AtomId, PolyMatrixNorm>,
    memo: BTreeMap<(SymbolicExprId, CanonicalAssignment), Estimate>,
}

pub fn simulate(graph: &ElaboratedGraph) -> Result<NoiseReport, SimulationError> {
    let mut evaluator = Evaluator {
        graph,
        contexts: BTreeMap::new(),
        atoms: BTreeMap::new(),
        memo: BTreeMap::new(),
    };
    let mut outputs = BTreeMap::new();
    for (name, wire) in &graph.outputs {
        let symbolic =
            evaluator.graph.wire(wire).ok_or_else(|| SimulationError::MissingWire(wire.clone()))?;
        if matches!(symbolic.wire_type, ConcreteWireType::Trapdoor { .. }) {
            continue;
        }
        let estimate = match (symbolic.expression, symbolic.family.as_ref()) {
            (Some(expression), _) => evaluator.eval(expression, &Assignment::new())?,
            (None, Some(SymbolicFamily::ExactMembers(members))) => {
                let mut branches = Vec::with_capacity(members.len());
                for member in members {
                    branches.push(evaluator.eval(*member, &Assignment::new())?);
                }
                join_selection(branches)
            }
            (None, Some(SymbolicFamily::StructuralTemplate { template, .. })) => {
                evaluator.eval(*template, &Assignment::new())?
            }
            (None, None) => continue,
        };
        outputs.insert(name.clone(), estimate.into_report());
    }

    let mut decode_targets = Vec::with_capacity(graph.decode_targets.len());
    for target in &graph.decode_targets {
        let symbolic = evaluator
            .graph
            .wire(&target.input)
            .ok_or_else(|| SimulationError::MissingWire(target.input.clone()))?;
        let expression = symbolic
            .expression
            .ok_or_else(|| SimulationError::MissingWire(target.input.clone()))?;
        let estimate = evaluator.eval(expression, &Assignment::new())?;
        let noise_bound = estimate
            .noise
            .as_ref()
            .map_or_else(BigDecimal::zero, PolyMatrixNorm::maximum_coefficient_bound);
        let modulus = symbolic
            .wire_type
            .matrix_type()
            .ok_or_else(|| SimulationError::MissingWire(target.input.clone()))?
            .modulus
            .clone();
        let threshold = BigDecimal::from(modulus) /
            (BigDecimal::from(2u64) * BigDecimal::from(target.plaintext_modulus.clone()));
        decode_targets.push(DecodeNoiseReport {
            target: target.clone(),
            within_threshold: noise_bound < threshold,
            estimate: estimate.into_report(),
            threshold,
        });
    }
    Ok(NoiseReport { outputs, decode_targets, methodology: METHODOLOGY })
}

impl Estimate {
    fn into_report(self) -> WireNoiseReport {
        WireNoiseReport {
            has_signal: self.signal,
            noise: self.noise.map(|noise| MatrixNoiseReport {
                bound: noise.maximum_coefficient_bound(),
                rows: noise.nrow,
                columns: noise.ncol,
                is_const_poly: noise.poly_norm.is_const_poly,
                zero_rows: noise.zero_rows,
                dependencies: noise.deps,
                clt_ready: noise.clt_ready,
            }),
        }
    }
}

impl Evaluator<'_> {
    fn context(&mut self, ring_dimension: usize) -> Arc<SimulatorContext> {
        self.contexts
            .entry(ring_dimension)
            .or_insert_with(|| {
                Arc::new(SimulatorContext::new(
                    BigDecimal::from(ring_dimension as u64)
                        .sqrt()
                        .expect("positive ring dimension has a square root"),
                    BigDecimal::from(2u64),
                    1,
                    1,
                    1,
                ))
            })
            .clone()
    }

    fn eval(
        &mut self,
        expression: SymbolicExprId,
        assignment: &Assignment,
    ) -> Result<Estimate, SimulationError> {
        let key = (expression, self.canonical_assignment(expression, assignment)?);
        if let Some(estimate) = self.memo.get(&key) {
            return Ok(estimate.clone());
        }
        let estimate = if let Some(domain) = self.first_unassigned_domain(expression, assignment)? {
            let mut branches = Vec::with_capacity(domain.count());
            for branch in 0..domain.count() {
                let mut assigned = assignment.clone();
                assigned.insert(domain.clone(), branch as u64);
                branches.push(self.eval(expression, &assigned)?);
            }
            join_selection(branches)
        } else {
            self.eval_assigned(expression, assignment)?
        };
        self.memo.insert(key, estimate.clone());
        Ok(estimate)
    }

    fn eval_assigned(
        &mut self,
        expression: SymbolicExprId,
        assignment: &Assignment,
    ) -> Result<Estimate, SimulationError> {
        let mut signal = false;
        let mut noise = None;
        self.for_each_alternative(expression, assignment, &mut |evaluator, alternative| {
            if alternative.factors.iter().any(|factor| matches!(factor, AnalysisFactor::Signal(_)))
            {
                signal = true;
                return Ok(());
            }
            let value = evaluator.eval_bounded_alternative(alternative)?;
            noise = Some(match noise.take() {
                Some(current) => normalize_exact_zero(current + value),
                None => value,
            });
            Ok(())
        })?;
        Ok(Estimate { signal, noise: noise.map(normalize_exact_zero) })
    }

    fn for_each_alternative(
        &mut self,
        expression: SymbolicExprId,
        assignment: &Assignment,
        callback: &mut dyn FnMut(&mut Self, Alternative) -> Result<(), SimulationError>,
    ) -> Result<(), SimulationError> {
        let record = self.record(expression)?.clone();
        match record.node {
            SymbolicExprNode::Zero => {
                let context = self.context(record.matrix_type.ring_dimension);
                callback(
                    self,
                    Alternative {
                        coefficient: BigInt::one(),
                        factors: vec![AnalysisFactor::Bounded(exact_zero(
                            record.matrix_type.rows,
                            record.matrix_type.columns,
                            context,
                        ))],
                    },
                )
            }
            SymbolicExprNode::Atom(id) => {
                let atom = self.atom(&id)?.clone();
                let factor = if matches!(atom.kind, AtomKind::Large) {
                    AnalysisFactor::Signal(atom.matrix_type)
                } else if is_identity_atom(&atom) {
                    AnalysisFactor::Identity(atom.matrix_type)
                } else {
                    AnalysisFactor::Bounded(self.eval_atom(&id)?)
                };
                callback(self, Alternative { coefficient: BigInt::one(), factors: vec![factor] })
            }
            SymbolicExprNode::Add(children) => {
                for child in children {
                    self.for_each_alternative(child, assignment, callback)?;
                }
                Ok(())
            }
            SymbolicExprNode::Scale { coefficient, value } => {
                self.for_each_alternative(value, assignment, &mut |evaluator, mut alternative| {
                    alternative.coefficient *= &coefficient;
                    callback(evaluator, alternative)
                })
            }
            SymbolicExprNode::Mul(children) => self.for_each_product_alternative(
                &children,
                0,
                assignment,
                Alternative { coefficient: BigInt::one(), factors: Vec::new() },
                callback,
            ),
            SymbolicExprNode::Tensor { left, right } => {
                self.for_each_alternative(left, assignment, &mut |evaluator, left_alt| {
                    evaluator.for_each_alternative(
                        right,
                        assignment,
                        &mut |evaluator, right_alt| {
                            let coefficient = &left_alt.coefficient * &right_alt.coefficient;
                            let left_signal = left_alt
                                .factors
                                .iter()
                                .any(|factor| matches!(factor, AnalysisFactor::Signal(_)));
                            let right_signal = right_alt
                                .factors
                                .iter()
                                .any(|factor| matches!(factor, AnalysisFactor::Signal(_)));
                            let factor = if left_signal || right_signal {
                                AnalysisFactor::Signal(record.matrix_type.clone())
                            } else {
                                let mut left_alt = left_alt.clone();
                                let mut right_alt = right_alt;
                                left_alt.coefficient = BigInt::one();
                                right_alt.coefficient = BigInt::one();
                                let left = evaluator.eval_bounded_alternative(left_alt)?;
                                let right = evaluator.eval_bounded_alternative(right_alt)?;
                                AnalysisFactor::Bounded(tensor_norm(
                                    left,
                                    right,
                                    &record.matrix_type,
                                ))
                            };
                            callback(evaluator, Alternative { coefficient, factors: vec![factor] })
                        },
                    )
                })
            }
            SymbolicExprNode::Concat { inputs, .. } => {
                let mut child_values = Vec::with_capacity(inputs.len());
                let mut has_signal = false;
                for input in inputs {
                    let input_ty = self.record(input)?.matrix_type.clone();
                    let estimate = self.eval_assigned(input, assignment)?;
                    has_signal |= estimate.signal;
                    child_values.push(estimate.noise.unwrap_or_else(|| {
                        exact_zero(
                            input_ty.rows,
                            input_ty.columns,
                            self.context(input_ty.ring_dimension),
                        )
                    }));
                }
                if has_signal {
                    callback(
                        self,
                        Alternative {
                            coefficient: BigInt::one(),
                            factors: vec![AnalysisFactor::Signal(record.matrix_type.clone())],
                        },
                    )?;
                }
                let context = self.context(record.matrix_type.ring_dimension);
                callback(
                    self,
                    Alternative {
                        coefficient: BigInt::one(),
                        factors: vec![AnalysisFactor::Bounded(join_opaque(
                            child_values,
                            &record.matrix_type,
                            context,
                        ))],
                    },
                )
            }
            SymbolicExprNode::Select { domain, branches } => {
                let branch = assignment.get(&domain).copied().ok_or_else(|| {
                    SimulationError::InvalidSelection("unassigned selection domain".to_owned())
                })?;
                let selected = branches.get(branch as usize).copied().ok_or_else(|| {
                    SimulationError::InvalidSelection("selection branch is out of range".to_owned())
                })?;
                self.for_each_alternative(selected, assignment, callback)
            }
            SymbolicExprNode::Transpose(value) => {
                self.for_each_alternative(value, assignment, &mut |evaluator, mut alternative| {
                    alternative.factors.reverse();
                    for factor in &mut alternative.factors {
                        transpose_factor(factor);
                    }
                    callback(evaluator, alternative)
                })
            }
            SymbolicExprNode::Slice { value, rows, columns } => {
                self.for_each_alternative(value, assignment, &mut |evaluator, mut alternative| {
                    slice_alternative(&mut alternative, rows, columns);
                    callback(evaluator, alternative)
                })
            }
            SymbolicExprNode::Reshape { value, .. } => {
                let estimate = self.eval_assigned(value, assignment)?;
                self.emit_structural_transform(
                    estimate,
                    &record.matrix_type,
                    callback,
                    reshape_norm,
                )
            }
            SymbolicExprNode::ConstantCoefficient { value, .. } => {
                let estimate = self.eval_assigned(value, assignment)?;
                self.emit_structural_transform(
                    estimate,
                    &record.matrix_type,
                    callback,
                    constant_coefficient_norm,
                )
            }
            SymbolicExprNode::CrtRecompose { .. } => callback(
                self,
                Alternative {
                    coefficient: BigInt::one(),
                    factors: vec![AnalysisFactor::Signal(record.matrix_type)],
                },
            ),
        }
    }

    fn for_each_product_alternative(
        &mut self,
        children: &[SymbolicExprId],
        index: usize,
        assignment: &Assignment,
        prefix: Alternative,
        callback: &mut dyn FnMut(&mut Self, Alternative) -> Result<(), SimulationError>,
    ) -> Result<(), SimulationError> {
        if index == children.len() {
            return callback(self, prefix);
        }
        self.for_each_alternative(children[index], assignment, &mut |evaluator, child| {
            let mut combined = prefix.clone();
            combined.coefficient *= child.coefficient;
            combined.factors.extend(child.factors);
            evaluator.for_each_product_alternative(
                children,
                index + 1,
                assignment,
                combined,
                callback,
            )
        })
    }

    fn emit_structural_transform(
        &mut self,
        estimate: Estimate,
        ty: &ConcreteMatrixType,
        callback: &mut dyn FnMut(&mut Self, Alternative) -> Result<(), SimulationError>,
        transform: impl FnOnce(PolyMatrixNorm, &ConcreteMatrixType) -> PolyMatrixNorm,
    ) -> Result<(), SimulationError> {
        if estimate.signal {
            callback(
                self,
                Alternative {
                    coefficient: BigInt::one(),
                    factors: vec![AnalysisFactor::Signal(ty.clone())],
                },
            )?;
        }
        if let Some(noise) = estimate.noise {
            callback(
                self,
                Alternative {
                    coefficient: BigInt::one(),
                    factors: vec![AnalysisFactor::Bounded(transform(noise, ty))],
                },
            )?;
        }
        Ok(())
    }

    fn eval_bounded_alternative(
        &mut self,
        alternative: Alternative,
    ) -> Result<PolyMatrixNorm, SimulationError> {
        if alternative.factors.iter().any(|factor| matches!(factor, AnalysisFactor::Signal(_))) {
            return Err(SimulationError::MixedBoundedAlternative);
        }
        let mut product_shape = None;
        let mut identity_ring_dimension = None;
        for factor in &alternative.factors {
            let shape = factor_shape(factor);
            product_shape = Some(match product_shape {
                Some(current) => multiplication_shape(current, shape)?,
                None => shape,
            });
            if let AnalysisFactor::Identity(ty) = factor {
                identity_ring_dimension = Some(ty.ring_dimension);
            }
        }
        let mut bounded = Vec::new();
        for factor in alternative.factors {
            match factor {
                AnalysisFactor::Bounded(value) => bounded.push(value),
                AnalysisFactor::Identity(_) => {}
                AnalysisFactor::Signal(_) => unreachable!("signal checked above"),
            }
        }
        let coefficient = BigDecimal::from(alternative.coefficient.abs());
        if bounded.is_empty() {
            let (rows, columns) = product_shape.unwrap_or((1, 1));
            let ring_dimension = identity_ring_dimension.unwrap_or(1);
            return Ok(PolyMatrixNorm::from_parts(
                rows,
                columns,
                PolyNorm::constant(self.context(ring_dimension), coefficient),
                None,
                DependencySet::empty(),
                false,
            ));
        }
        let mut value = bounded.remove(0);
        for rhs in bounded {
            value = multiply_factors(value, rhs)?;
        }
        if coefficient != BigDecimal::one() {
            value = value * coefficient;
        }
        let mut value = normalize_exact_zero(value);
        if let Some((rows, columns)) = product_shape &&
            (value.nrow != rows || value.ncol != columns)
        {
            if value.poly_norm.norm.is_zero() {
                return Ok(exact_zero(rows, columns, value.clone_ctx()));
            }
            value.nrow = rows;
            value.ncol = columns;
            value.ncol_sqrt =
                BigDecimal::from(columns as u64).sqrt().expect("positive column count");
            value.zero_rows = None;
        }
        Ok(value)
    }

    fn eval_atom(&mut self, id: &AtomId) -> Result<PolyMatrixNorm, SimulationError> {
        if let Some(value) = self.atoms.get(id) {
            return Ok(value.clone());
        }
        let atom = self.atom(id)?.clone();
        if matches!(atom.kind, AtomKind::Large) {
            return Err(SimulationError::MixedBoundedAlternative);
        }
        let value = normalize_exact_zero(match &atom.class {
            AtomClass::Source { source } => self.eval_source(&atom, source)?,
            AtomClass::Assumed { metadata } => {
                let metadata =
                    metadata.as_ref().ok_or_else(|| SimulationError::InvalidMetadata {
                        atom: id.clone(),
                        reason: "bounded assumed atom has no declared metadata".to_owned(),
                    })?;
                self.eval_assumed(&atom, metadata)?
            }
        });
        self.atoms.insert(id.clone(), value.clone());
        Ok(value)
    }

    fn eval_source(
        &mut self,
        atom: &Atom,
        source: &SourceKind,
    ) -> Result<PolyMatrixNorm, SimulationError> {
        let ty = &atom.matrix_type;
        let ctx = self.context(ty.ring_dimension);
        let stable = DependencySet::singleton(stable_source_id(&atom.id));
        Ok(match source {
            SourceKind::ConstantMatrix { value } => PolyMatrixNorm::from_parts(
                ty.rows,
                ty.columns,
                PolyNorm::constant(ctx, constant_norm(value, ty, &atom.id)?),
                matches!(value, ConstantMatrix::Zero).then_some(ty.rows),
                DependencySet::empty(),
                false,
            ),
            SourceKind::UniformSample { minimum, maximum } => PolyMatrixNorm::from_parts(
                ty.rows,
                ty.columns,
                PolyNorm::new(ctx, BigDecimal::from(minimum.abs().max(maximum.abs()))),
                None,
                stable,
                true,
            ),
            SourceKind::GaussianSample { sigma } => PolyMatrixNorm::from_parts(
                ty.rows,
                ty.columns,
                PolyNorm::sample_gauss(ctx, eval_real(sigma, &atom.id)?),
                None,
                stable,
                true,
            ),
            SourceKind::PreimageSample {
                trapdoor_sigma,
                gadget_base,
                digit_count,
                public_matrix_rows,
                target_block_rows,
                zero_rows,
            } => {
                let tau = eval_real(trapdoor_sigma, &atom.id)?;
                let ring_sqrt = &ctx.ring_dim_sqrt;
                let m_g = public_matrix_rows.saturating_mul(*digit_count);
                let term = sqrt_usize(*target_block_rows) * ring_sqrt * sqrt_usize(m_g) +
                    BigDecimal::from(2u64).sqrt().expect("sqrt(2)") * ring_sqrt +
                    decimal_ratio(47, 10);
                let sigma = decimal_ratio(18, 10) *
                    &tau *
                    (BigDecimal::from(gadget_base.clone()) + BigDecimal::one()) *
                    &tau *
                    term;
                PolyMatrixNorm::from_parts(
                    ty.rows,
                    ty.columns,
                    PolyNorm::new(ctx, high_probability_envelope_from_sigma(&sigma)),
                    *zero_rows,
                    stable,
                    true,
                )
            }
            SourceKind::GadgetDecomposition { base, .. } |
            SourceKind::HashSample { base: Some(base), .. } => {
                let variance = (BigDecimal::from(base * base) + BigDecimal::from(2u64)) /
                    BigDecimal::from(12u64);
                let norm = high_probability_envelope_from_sigma(
                    &variance.sqrt().expect("nonnegative digit variance"),
                );
                let deps = if matches!(source, SourceKind::GadgetDecomposition { .. }) {
                    DependencySet::empty()
                } else {
                    stable
                };
                PolyMatrixNorm::from_parts(
                    ty.rows,
                    ty.columns,
                    PolyNorm::new(ctx, norm),
                    None,
                    deps,
                    false,
                )
            }
            SourceKind::External { kind } => {
                return Err(SimulationError::UnsupportedExternal {
                    atom: atom.id.clone(),
                    kind: kind.clone(),
                });
            }
            SourceKind::TrapdoorUniform { .. } |
            SourceKind::HashSample { base: None, .. } |
            SourceKind::HashTarget { .. } => {
                return Err(SimulationError::MixedBoundedAlternative);
            }
        })
    }

    fn eval_assumed(
        &mut self,
        atom: &Atom,
        metadata: &AssumedMetadata,
    ) -> Result<PolyMatrixNorm, SimulationError> {
        let ctx = self.context(atom.matrix_type.ring_dimension);
        let dependencies = match &metadata.dependencies {
            DeclaredDependencies::Unknown => DependencySet::Unknown,
            DeclaredDependencies::Known(labels) => {
                DependencySet::known(labels.iter().map(stable_declared_id).collect())
            }
        };
        let norm = eval_real(&metadata.norm, &atom.id)?;
        Ok(PolyMatrixNorm::from_parts(
            atom.matrix_type.rows,
            atom.matrix_type.columns,
            if metadata.is_const_poly {
                PolyNorm::constant(ctx, norm)
            } else {
                PolyNorm::new(ctx, norm)
            },
            metadata.zero_rows,
            dependencies,
            metadata.clt_ready,
        ))
    }

    fn first_unassigned_domain(
        &self,
        expression: SymbolicExprId,
        assignment: &Assignment,
    ) -> Result<Option<SelectionDomainRef>, SimulationError> {
        let mut domains = Vec::new();
        self.collect_domains(expression, assignment, &mut domains, &mut BTreeSet::new())?;
        domains.sort();
        domains.dedup();
        Ok(domains.into_iter().next())
    }

    fn canonical_assignment(
        &self,
        expression: SymbolicExprId,
        assignment: &Assignment,
    ) -> Result<CanonicalAssignment, SimulationError> {
        let mut domains = Vec::new();
        self.collect_relevant_domains(expression, assignment, &mut domains, &mut BTreeSet::new())?;
        domains.sort();
        domains.dedup();
        Ok(domains
            .into_iter()
            .filter_map(|domain| assignment.get(&domain).map(|branch| (domain, *branch)))
            .collect())
    }

    fn collect_relevant_domains(
        &self,
        expression: SymbolicExprId,
        assignment: &Assignment,
        domains: &mut Vec<SelectionDomainRef>,
        visited: &mut BTreeSet<SymbolicExprId>,
    ) -> Result<(), SimulationError> {
        if !visited.insert(expression) {
            return Ok(());
        }
        let node = &self.record(expression)?.node;
        match node {
            SymbolicExprNode::Select { domain, branches } => {
                domains.push(domain.clone());
                if let Some(branch) = assignment.get(domain).copied() {
                    let selected = branches.get(branch as usize).ok_or_else(|| {
                        SimulationError::InvalidSelection(
                            "selection branch is out of range".to_owned(),
                        )
                    })?;
                    self.collect_relevant_domains(*selected, assignment, domains, visited)?;
                }
            }
            SymbolicExprNode::Add(children) |
            SymbolicExprNode::Mul(children) |
            SymbolicExprNode::Concat { inputs: children, .. } |
            SymbolicExprNode::CrtRecompose { inputs: children, .. } => {
                for child in children {
                    self.collect_relevant_domains(*child, assignment, domains, visited)?;
                }
            }
            SymbolicExprNode::Scale { value, .. } |
            SymbolicExprNode::Transpose(value) |
            SymbolicExprNode::Slice { value, .. } |
            SymbolicExprNode::Reshape { value, .. } |
            SymbolicExprNode::ConstantCoefficient { value, .. } => {
                self.collect_relevant_domains(*value, assignment, domains, visited)?;
            }
            SymbolicExprNode::Tensor { left, right } => {
                self.collect_relevant_domains(*left, assignment, domains, visited)?;
                self.collect_relevant_domains(*right, assignment, domains, visited)?;
            }
            SymbolicExprNode::Zero | SymbolicExprNode::Atom(_) => {}
        }
        Ok(())
    }

    fn collect_domains(
        &self,
        expression: SymbolicExprId,
        assignment: &Assignment,
        domains: &mut Vec<SelectionDomainRef>,
        visited: &mut BTreeSet<SymbolicExprId>,
    ) -> Result<(), SimulationError> {
        if !visited.insert(expression) {
            return Ok(());
        }
        let node = &self.record(expression)?.node;
        match node {
            SymbolicExprNode::Select { domain, branches } => {
                if let Some(branch) = assignment.get(domain).copied() {
                    let selected = branches.get(branch as usize).ok_or_else(|| {
                        SimulationError::InvalidSelection(
                            "selection branch is out of range".to_owned(),
                        )
                    })?;
                    self.collect_domains(*selected, assignment, domains, visited)?;
                } else {
                    domains.push(domain.clone());
                }
            }
            SymbolicExprNode::Add(children) |
            SymbolicExprNode::Mul(children) |
            SymbolicExprNode::Concat { inputs: children, .. } |
            SymbolicExprNode::CrtRecompose { inputs: children, .. } => {
                for child in children {
                    self.collect_domains(*child, assignment, domains, visited)?;
                }
            }
            SymbolicExprNode::Scale { value, .. } |
            SymbolicExprNode::Transpose(value) |
            SymbolicExprNode::Slice { value, .. } |
            SymbolicExprNode::Reshape { value, .. } |
            SymbolicExprNode::ConstantCoefficient { value, .. } => {
                self.collect_domains(*value, assignment, domains, visited)?;
            }
            SymbolicExprNode::Tensor { left, right } => {
                self.collect_domains(*left, assignment, domains, visited)?;
                self.collect_domains(*right, assignment, domains, visited)?;
            }
            SymbolicExprNode::Zero | SymbolicExprNode::Atom(_) => {}
        }
        Ok(())
    }

    fn record(
        &self,
        expression: SymbolicExprId,
    ) -> Result<&mxx_ir_symbolic::expression::SymbolicExprRecord, SimulationError> {
        self.graph.expressions.get(expression).ok_or(SimulationError::MissingExpression(expression))
    }

    fn atom(&self, id: &AtomId) -> Result<&Atom, SimulationError> {
        self.graph.atoms.get(id).ok_or_else(|| SimulationError::MissingAtom(id.clone()))
    }
}

fn join_selection(branches: Vec<Estimate>) -> Estimate {
    let signal = branches.iter().any(|branch| branch.signal);
    let Some(template) = branches.iter().find_map(|branch| branch.noise.as_ref()).cloned() else {
        return Estimate { signal, noise: None };
    };
    let noises = branches
        .into_iter()
        .map(|branch| {
            branch
                .noise
                .unwrap_or_else(|| exact_zero(template.nrow, template.ncol, template.clone_ctx()))
        })
        .collect::<Vec<_>>();
    Estimate { signal, noise: join_selection_noise(noises) }
}

fn join_selection_noise(values: Vec<PolyMatrixNorm>) -> Option<PolyMatrixNorm> {
    let first = values.first()?.clone();
    let norm = values
        .iter()
        .map(PolyMatrixNorm::maximum_coefficient_bound)
        .max()
        .expect("nonempty values");
    if norm.is_zero() {
        return Some(exact_zero(first.nrow, first.ncol, first.clone_ctx()));
    }
    let dependencies =
        values.iter().fold(DependencySet::empty(), |deps, value| deps.union(&value.deps));
    let is_const_poly = values.iter().all(|value| value.poly_norm.is_const_poly);
    let zero_rows = all_equal(values.iter().map(|value| value.zero_rows)).unwrap_or(None);
    Some(PolyMatrixNorm::from_parts(
        first.nrow,
        first.ncol,
        if is_const_poly {
            PolyNorm::constant(first.clone_ctx(), norm)
        } else {
            PolyNorm::new(first.clone_ctx(), norm)
        },
        zero_rows,
        dependencies,
        false,
    ))
}

fn all_equal<T: Clone + PartialEq>(mut values: impl Iterator<Item = T>) -> Option<T> {
    let first = values.next()?;
    values.all(|value| value == first).then_some(first)
}

fn join_opaque(
    values: Vec<PolyMatrixNorm>,
    ty: &ConcreteMatrixType,
    ctx: Arc<SimulatorContext>,
) -> PolyMatrixNorm {
    let norm = values
        .iter()
        .map(PolyMatrixNorm::maximum_coefficient_bound)
        .max()
        .unwrap_or_else(BigDecimal::zero);
    if norm.is_zero() {
        return exact_zero(ty.rows, ty.columns, ctx);
    }
    let deps = values.iter().fold(DependencySet::empty(), |deps, value| deps.union(&value.deps));
    let is_const_poly = values.iter().all(|value| value.poly_norm.is_const_poly);
    PolyMatrixNorm::from_parts(
        ty.rows,
        ty.columns,
        if is_const_poly { PolyNorm::constant(ctx, norm) } else { PolyNorm::new(ctx, norm) },
        None,
        deps,
        false,
    )
}

fn reshape_norm(value: PolyMatrixNorm, ty: &ConcreteMatrixType) -> PolyMatrixNorm {
    let zero_rows = (value.poly_norm.is_const_poly &&
        value.poly_norm.norm.is_zero() &&
        value.zero_rows == Some(value.nrow))
    .then_some(ty.rows);
    PolyMatrixNorm::from_parts(ty.rows, ty.columns, value.poly_norm, zero_rows, value.deps, false)
}

fn constant_coefficient_norm(value: PolyMatrixNorm, ty: &ConcreteMatrixType) -> PolyMatrixNorm {
    let zero_rows =
        (value.poly_norm.norm.is_zero() && value.zero_rows == Some(value.nrow)).then_some(ty.rows);
    PolyMatrixNorm::from_parts(
        ty.rows,
        ty.columns,
        value.poly_norm.into_constant_poly(),
        zero_rows,
        value.deps,
        false,
    )
}

fn tensor_norm(
    left: PolyMatrixNorm,
    right: PolyMatrixNorm,
    ty: &ConcreteMatrixType,
) -> PolyMatrixNorm {
    let disjoint = left.deps.is_disjoint(&right.deps);
    let use_clt = disjoint && (left.clt_ready || right.clt_ready);
    let contraction = if left.poly_norm.is_const_poly || right.poly_norm.is_const_poly {
        BigDecimal::one()
    } else {
        BigDecimal::from(ty.ring_dimension as u64)
    };
    let scale = if use_clt {
        contraction.sqrt().expect("positive tensor contraction")
    } else {
        contraction
    };
    let norm = scale * &left.poly_norm.norm * &right.poly_norm.norm;
    PolyMatrixNorm::from_parts(
        ty.rows,
        ty.columns,
        if left.poly_norm.is_const_poly && right.poly_norm.is_const_poly {
            PolyNorm::constant(left.clone_ctx(), norm)
        } else {
            PolyNorm::new(left.clone_ctx(), norm)
        },
        None,
        left.deps.union(&right.deps),
        disjoint && left.clt_ready && right.clt_ready,
    )
}

fn multiply_factors(
    lhs: PolyMatrixNorm,
    rhs: PolyMatrixNorm,
) -> Result<PolyMatrixNorm, SimulationError> {
    if lhs.nrow == 1 && lhs.ncol == 1 && !(rhs.nrow == 1 && rhs.ncol == 1) {
        return Ok(scale_matrix(rhs, lhs));
    }
    if rhs.nrow == 1 && rhs.ncol == 1 && !(lhs.nrow == 1 && lhs.ncol == 1) {
        return Ok(scale_matrix(lhs, rhs));
    }
    if lhs.ncol != rhs.nrow {
        return Err(SimulationError::ShapeMismatch);
    }
    Ok(lhs * rhs)
}

fn scale_matrix(matrix: PolyMatrixNorm, scalar: PolyMatrixNorm) -> PolyMatrixNorm {
    let disjoint = matrix.deps.is_disjoint(&scalar.deps);
    let clt_ready = disjoint && matrix.clt_ready && scalar.clt_ready;
    let dependencies = matrix.deps.union(&scalar.deps);
    let mut output = matrix * &scalar.poly_norm;
    output.deps = dependencies;
    output.clt_ready = clt_ready;
    output
}

fn transpose_factor(factor: &mut AnalysisFactor) {
    match factor {
        AnalysisFactor::Signal(ty) | AnalysisFactor::Identity(ty) => {
            std::mem::swap(&mut ty.rows, &mut ty.columns);
        }
        AnalysisFactor::Bounded(value) => {
            std::mem::swap(&mut value.nrow, &mut value.ncol);
            value.ncol_sqrt =
                BigDecimal::from(value.ncol as u64).sqrt().expect("positive column count");
            value.zero_rows = None;
        }
    }
}

fn slice_alternative(
    alternative: &mut Alternative,
    rows: Option<IndexRange>,
    columns: Option<IndexRange>,
) {
    if let Some(index) = alternative.factors.iter().position(|factor| !factor_is_scalar(factor)) {
        if let Some(rows) = rows {
            set_factor_rows(&mut alternative.factors[index], rows.end - rows.start);
        }
    }
    if let Some(index) = alternative.factors.iter().rposition(|factor| !factor_is_scalar(factor)) {
        if let Some(columns) = columns {
            set_factor_columns(&mut alternative.factors[index], columns.end - columns.start);
        }
    }
}

fn set_factor_rows(factor: &mut AnalysisFactor, rows: usize) {
    match factor {
        AnalysisFactor::Signal(ty) | AnalysisFactor::Identity(ty) => ty.rows = rows,
        AnalysisFactor::Bounded(value) => {
            value.nrow = rows;
            value.zero_rows = None;
        }
    }
}

fn set_factor_columns(factor: &mut AnalysisFactor, columns: usize) {
    match factor {
        AnalysisFactor::Signal(ty) | AnalysisFactor::Identity(ty) => ty.columns = columns,
        AnalysisFactor::Bounded(value) => {
            value.ncol = columns;
            value.ncol_sqrt =
                BigDecimal::from(columns as u64).sqrt().expect("positive column count");
        }
    }
}

fn factor_is_scalar(factor: &AnalysisFactor) -> bool {
    match factor {
        AnalysisFactor::Signal(ty) | AnalysisFactor::Identity(ty) => ty.is_scalar(),
        AnalysisFactor::Bounded(value) => value.nrow == 1 && value.ncol == 1,
    }
}

fn factor_shape(factor: &AnalysisFactor) -> (usize, usize) {
    match factor {
        AnalysisFactor::Signal(ty) | AnalysisFactor::Identity(ty) => (ty.rows, ty.columns),
        AnalysisFactor::Bounded(value) => (value.nrow, value.ncol),
    }
}

fn multiplication_shape(
    left: (usize, usize),
    right: (usize, usize),
) -> Result<(usize, usize), SimulationError> {
    if left == (1, 1) {
        Ok(right)
    } else if right == (1, 1) {
        Ok(left)
    } else if left.1 == right.0 {
        Ok((left.0, right.1))
    } else {
        Err(SimulationError::ShapeMismatch)
    }
}

fn exact_zero(nrow: usize, ncol: usize, ctx: Arc<SimulatorContext>) -> PolyMatrixNorm {
    PolyMatrixNorm::from_parts(
        nrow,
        ncol,
        PolyNorm::constant(ctx, BigDecimal::zero()),
        Some(nrow),
        DependencySet::empty(),
        false,
    )
}

fn normalize_exact_zero(value: PolyMatrixNorm) -> PolyMatrixNorm {
    if value.maximum_coefficient_bound().is_zero() {
        exact_zero(value.nrow, value.ncol, value.clone_ctx())
    } else {
        value
    }
}

fn is_identity_atom(atom: &Atom) -> bool {
    matches!(
        atom.class,
        AtomClass::Source {
            source: SourceKind::ConstantMatrix { value: ConstantMatrix::Identity }
        }
    )
}

fn constant_norm(
    value: &ConstantMatrix,
    ty: &ConcreteMatrixType,
    atom: &AtomId,
) -> Result<BigDecimal, SimulationError> {
    Ok(match value {
        ConstantMatrix::Zero => BigDecimal::zero(),
        ConstantMatrix::Identity |
        ConstantMatrix::UnitRow { .. } |
        ConstantMatrix::UnitColumn { .. } |
        ConstantMatrix::Rotation { .. } => BigDecimal::one(),
        ConstantMatrix::Gadget { .. } => {
            BigDecimal::from(ty.modulus.clone()) / BigDecimal::from(2u64)
        }
        ConstantMatrix::PowerOfBase { base, exponent } => {
            let base = eval_int(base, atom)?.abs();
            let exponent = eval_int(exponent, atom)?.to_u32().ok_or_else(|| {
                SimulationError::InvalidMetadata {
                    atom: atom.clone(),
                    reason: "power exponent is not a u32".to_owned(),
                }
            })?;
            BigDecimal::from(base.pow(exponent))
        }
        ConstantMatrix::Polynomial { coefficients } => coefficients
            .iter()
            .map(|coefficient| eval_int(coefficient, atom).map(|value| value.abs()))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .map(BigDecimal::from)
            .unwrap_or_else(BigDecimal::zero),
    })
}

fn eval_real(expression: &RealExpr, atom: &AtomId) -> Result<BigDecimal, SimulationError> {
    let value = eval_real_raw(expression, atom)?;
    if value < BigDecimal::zero() {
        return Err(SimulationError::InvalidMetadata {
            atom: atom.clone(),
            reason: "negative norm metadata".to_owned(),
        });
    }
    Ok(value)
}

fn eval_real_raw(expression: &RealExpr, atom: &AtomId) -> Result<BigDecimal, SimulationError> {
    Ok(match expression {
        RealExpr::Rational(value) => {
            BigDecimal::from(value.numerator().clone()) /
                BigDecimal::from(value.denominator().clone())
        }
        RealExpr::FromInt(value) => BigDecimal::from(eval_int(value, atom)?),
        RealExpr::Add(lhs, rhs) => eval_real_raw(lhs, atom)? + eval_real_raw(rhs, atom)?,
        RealExpr::Sub(lhs, rhs) => eval_real_raw(lhs, atom)? - eval_real_raw(rhs, atom)?,
        RealExpr::Mul(lhs, rhs) => eval_real_raw(lhs, atom)? * eval_real_raw(rhs, atom)?,
        RealExpr::Div(lhs, rhs) => {
            let denominator = eval_real_raw(rhs, atom)?;
            if denominator.is_zero() {
                return Err(SimulationError::InvalidMetadata {
                    atom: atom.clone(),
                    reason: "division by zero".to_owned(),
                });
            }
            eval_real_raw(lhs, atom)? / denominator
        }
        RealExpr::Sqrt(value) => {
            let radicand = eval_real_raw(value, atom)?;
            if radicand < BigDecimal::zero() {
                return Err(SimulationError::InvalidMetadata {
                    atom: atom.clone(),
                    reason: "square root of a negative value".to_owned(),
                });
            }
            radicand.sqrt().ok_or_else(|| SimulationError::InvalidMetadata {
                atom: atom.clone(),
                reason: "square root evaluation failed".to_owned(),
            })?
        }
        RealExpr::Var(name) => {
            return Err(SimulationError::InvalidMetadata {
                atom: atom.clone(),
                reason: format!("unclosed real variable {name}"),
            });
        }
    })
}

fn eval_int(expression: &IntExpr, atom: &AtomId) -> Result<BigInt, SimulationError> {
    expression.evaluate(&ParamEnv::default()).map_err(|error| SimulationError::InvalidMetadata {
        atom: atom.clone(),
        reason: error.to_string(),
    })
}

fn stable_source_id(atom: &AtomId) -> SourceId {
    stable_hash(b"mxx-noise-simulator/source/v1", atom)
}

fn stable_declared_id(label: &DeclaredDependencyRef) -> SourceId {
    stable_hash(b"mxx-noise-simulator/declared-dependency/v1", label)
}

fn stable_hash<T: serde::Serialize>(domain: &[u8], value: &T) -> SourceId {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(serde_json::to_vec(value).expect("serializable stable identity"));
    SourceId(hasher.finalize().into())
}

fn sqrt_usize(value: usize) -> BigDecimal {
    BigDecimal::from(value as u64).sqrt().expect("nonnegative integer square root")
}

fn decimal_ratio(numerator: i64, denominator: i64) -> BigDecimal {
    BigDecimal::from(numerator) / BigDecimal::from(denominator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::{BoundedMetadata, DslContext, Family, Ring, VirtualMat};
    use mxx_ir_core::{
        Graph, GraphOutput, NodeHandle, ValueHandle,
        node::NodeKind,
        types::{MatrixType, WireType},
    };
    use mxx_ir_symbolic::overlay::DeclaredDependencyLabels;
    use std::collections::{BTreeMap, BTreeSet};

    fn output_report(output: mxx_dsl::Mat) -> WireNoiseReport {
        let built = DslContext::new("noise-test")
            .output("output", output)
            .expect("output")
            .build()
            .expect("build");
        let elaborated = built.elaborate(&ParamEnv::default()).expect("elaboration");
        simulate(&elaborated).expect("simulation").outputs.remove("output").expect("report")
    }

    fn bounded_metadata(
        norm: i64,
        labels: &[&str],
        zero_rows: Option<usize>,
        clt_ready: bool,
    ) -> BoundedMetadata {
        BoundedMetadata {
            norm: RealExpr::from_integer(norm),
            is_const_poly: false,
            zero_rows: zero_rows.map(IntExpr::constant),
            dependencies: DeclaredDependencyLabels::Known(
                labels.iter().map(|label| (*label).to_owned()).collect::<BTreeSet<_>>(),
            ),
            clt_ready,
        }
    }

    fn core_matrix_type(rows: usize, columns: usize) -> MatrixType {
        MatrixType {
            modulus: IntExpr::constant(97),
            ring_dimension: IntExpr::constant(8),
            rows: IntExpr::constant(rows),
            columns: IntExpr::constant(columns),
        }
    }

    fn freeze_graph(
        name: &str,
        outputs: impl IntoIterator<Item = (&'static str, ValueHandle)>,
        effects: Vec<ValueHandle>,
    ) -> Graph {
        Graph::freeze(
            name,
            Vec::new(),
            outputs
                .into_iter()
                .map(|(name, value)| {
                    (name.to_owned(), GraphOutput { value, confidentiality: None })
                })
                .collect(),
            Vec::new(),
            effects,
            BTreeMap::new(),
        )
        .expect("freeze graph")
        .0
    }

    #[test]
    fn selection_uses_the_largest_branch_bound_instead_of_the_sum() {
        let ring = Ring::new(257, 8);
        let index = ring.input("index", (1, 1)).extract_coefficient(0);
        let selected = index
            .select(vec![ring.gaussian((1, 1), 2), ring.gaussian((1, 1), 3)])
            .expect("compatible branches");
        let built = DslContext::new("selection-bound")
            .output("selected", selected)
            .expect("output")
            .build()
            .expect("build");
        let elaborated = built.elaborate(&ParamEnv::default()).expect("elaboration");
        let report = simulate(&elaborated).expect("simulation");
        assert_eq!(
            report.outputs["selected"].noise.as_ref().expect("bounded output").bound,
            BigDecimal::from(195u64) / BigDecimal::from(10u64),
        );
    }

    #[test]
    fn static_family_access_uses_the_selected_member_expression() {
        let ring = Ring::new(257, 8);
        let family =
            Family::pack(vec![ring.gaussian((1, 1), 2), ring.gaussian((1, 1), 3)]).expect("family");
        let selected = family.get_static(1);
        let built = DslContext::new("family-member-bound")
            .output("selected", selected)
            .expect("output")
            .family_output("family", family)
            .expect("family output")
            .build()
            .expect("build");
        let elaborated = built.elaborate(&ParamEnv::default()).expect("elaboration");
        let report = simulate(&elaborated).expect("simulation");
        assert_eq!(
            report.outputs["selected"].noise.as_ref().expect("bounded output").bound,
            BigDecimal::from(195u64) / BigDecimal::from(10u64),
        );
        assert_eq!(
            report.outputs["family"].noise.as_ref().expect("bounded family").bound,
            BigDecimal::from(195u64) / BigDecimal::from(10u64),
        );
    }

    #[test]
    fn exact_identity_does_not_change_a_gaussian_bound() {
        let ring = Ring::new(257, 8);
        let report = output_report(ring.identity(2) * ring.gaussian((2, 1), 2));
        assert_eq!(report.noise.expect("bounded output").bound, BigDecimal::from(13u8));
    }

    #[test]
    fn direct_preimage_preserves_the_existing_derived_sigma_formula() {
        let ring = Ring::new(97, 8);
        let trapdoor = ring.sample_trapdoor(1, 3, 2, 1);
        let preimage = trapdoor.sample_preimage(ring.gaussian((1, 1), 1), (3, 1));
        let report = output_report(preimage.as_mat()).noise.expect("preimage noise");
        let ring_sqrt = BigDecimal::from(8u8).sqrt().expect("sqrt(8)");
        let derived_sigma = decimal_ratio(18, 10) *
            BigDecimal::from(3u8) *
            BigDecimal::from(3u8) *
            BigDecimal::from(3u8) *
            (BigDecimal::from(3u8).sqrt().expect("sqrt(3)") * ring_sqrt.clone() +
                BigDecimal::from(2u8).sqrt().expect("sqrt(2)") * ring_sqrt +
                decimal_ratio(47, 10));
        assert_eq!(report.bound, high_probability_envelope_from_sigma(&derived_sigma));
    }

    #[test]
    fn bounded_external_preimage_without_manifest_or_assumption_is_unsupported() {
        let ty = core_matrix_type(1, 1);
        let input = NodeHandle::new(
            NodeKind::Input {
                name: "external".to_owned(),
                wire_type: WireType::Preimage(ty.clone()),
                artifact: None,
            },
            Vec::new(),
            vec![WireType::Preimage(ty)],
        );
        let graph = freeze_graph(
            "external-preimage",
            [("external", input.output(0).expect("input"))],
            Vec::new(),
        );
        let elaborated =
            mxx_ir_symbolic::elaborate(&graph, &ParamEnv::default()).expect("symbolic elaboration");
        assert!(matches!(
            simulate(&elaborated),
            Err(SimulationError::UnsupportedExternal { kind: ExternalSourceKind::Preimage, .. })
        ));
    }

    #[test]
    fn decode_effect_is_simulated_without_a_matrix_output() {
        let ty = core_matrix_type(1, 1);
        let gaussian = NodeHandle::new(
            NodeKind::GaussianSample { matrix_type: ty.clone(), sigma: RealExpr::from_integer(1) },
            Vec::new(),
            vec![WireType::Matrix(ty)],
        );
        let decode = NodeHandle::new(
            NodeKind::ThresholdDecode {
                plaintext_modulus: IntExpr::constant(2),
                length: IntExpr::constant(1),
                output_bool: false,
            },
            vec![gaussian.output(0).expect("gaussian")],
            vec![WireType::Int],
        );
        let graph = freeze_graph(
            "decode-effect",
            std::iter::empty(),
            vec![decode.output(0).expect("decode effect")],
        );
        let elaborated =
            mxx_ir_symbolic::elaborate(&graph, &ParamEnv::default()).expect("symbolic elaboration");
        let report = simulate(&elaborated).expect("simulation");
        assert!(report.outputs.is_empty());
        assert_eq!(report.decode_targets.len(), 1);
        assert_eq!(
            report.decode_targets[0].estimate.noise.as_ref().expect("decode noise").bound,
            decimal_ratio(13, 2)
        );
    }

    #[test]
    fn simulation_does_not_visit_a_non_output_external_preimage() {
        let ty = core_matrix_type(1, 1);
        let external = NodeHandle::new(
            NodeKind::Input {
                name: "unused".to_owned(),
                wire_type: WireType::Preimage(ty.clone()),
                artifact: None,
            },
            Vec::new(),
            vec![WireType::Preimage(ty.clone())],
        );
        let gaussian = NodeHandle::new(
            NodeKind::GaussianSample { matrix_type: ty.clone(), sigma: RealExpr::from_integer(1) },
            Vec::new(),
            vec![WireType::Matrix(ty)],
        );
        let graph = freeze_graph(
            "reachable-output-only",
            [
                ("unused", external.output(0).expect("external")),
                ("output", gaussian.output(0).expect("gaussian")),
            ],
            Vec::new(),
        );
        let mut elaborated =
            mxx_ir_symbolic::elaborate(&graph, &ParamEnv::default()).expect("symbolic elaboration");
        elaborated.outputs.remove("unused");
        let report = simulate(&elaborated).expect("unreferenced external is not evaluated");
        assert_eq!(report.outputs.len(), 1);
    }

    #[test]
    fn exact_zero_structural_transform_keeps_canonical_metadata() {
        let ring = Ring::new(257, 8);
        let report = output_report(ring.zero((1, 4)).reshape(2, 2));
        let noise = report.noise.expect("typed zero");
        assert_eq!(noise.bound, BigDecimal::zero());
        assert_eq!((noise.rows, noise.columns), (2, 2));
        assert!(noise.is_const_poly);
        assert_eq!(noise.zero_rows, Some(2));
        assert_eq!(noise.dependencies, DependencySet::empty());
        assert!(!noise.clt_ready);
    }

    #[test]
    fn identity_preserves_all_bounded_metadata_on_both_sides() {
        let ring = Ring::new(257, 8);
        let ty = ring.matrix_type((2, 2));
        let virtual_value =
            VirtualMat::bounded("bounded", ty, bounded_metadata(7, &["source"], Some(1), true));
        let assumed =
            ring.input("assumed", (2, 2)).assume(virtual_value).expect("well-typed assumption");
        let direct = output_report(assumed.clone()).noise.expect("direct bounded output");
        let left =
            output_report(ring.identity(2) * assumed.clone()).noise.expect("left identity output");
        let right = output_report(assumed * ring.identity(2)).noise.expect("right identity output");
        assert_eq!(left, direct);
        assert_eq!(right, direct);
    }

    #[test]
    fn identity_times_bounded_scalar_preserves_matrix_shape() {
        let ring = Ring::new(257, 8);
        let scalar = ring
            .input("scalar", (1, 1))
            .assume(VirtualMat::bounded(
                "scalar",
                ring.matrix_type((1, 1)),
                bounded_metadata(7, &["source"], None, true),
            ))
            .expect("well-typed scalar assumption");
        let report = output_report(ring.identity(2) * scalar).noise.expect("bounded output");
        assert_eq!((report.rows, report.columns), (2, 2));
        assert_eq!(report.bound, BigDecimal::from(7u8));
    }

    #[test]
    fn simulator_skips_identity_factors_before_norm_multiplication() {
        let ring = Ring::new(257, 8);
        let output = ring
            .input("assumed", (2, 2))
            .assume(VirtualMat::bounded(
                "bounded",
                ring.matrix_type((2, 2)),
                bounded_metadata(7, &["source"], Some(1), true),
            ))
            .expect("well-typed assumption");
        let built = DslContext::new("identity-factor")
            .output("output", output)
            .expect("output")
            .build()
            .expect("build");
        let elaborated = built.elaborate(&ParamEnv::default()).expect("elaboration");
        let expression = elaborated.outputs["output"].clone();
        let symbolic = elaborated.wire(&expression).expect("symbolic output");
        let atom = match &elaborated
            .expressions
            .get(symbolic.expression.expect("expression"))
            .expect("record")
            .node
        {
            SymbolicExprNode::Atom(atom) => atom.clone(),
            node => panic!("expected assumed atom, got {node:?}"),
        };
        let mut evaluator = Evaluator {
            graph: &elaborated,
            contexts: BTreeMap::new(),
            atoms: BTreeMap::new(),
            memo: BTreeMap::new(),
        };
        let bounded = evaluator.eval_atom(&atom).expect("bounded atom");
        let identity_type = ConcreteMatrixType {
            modulus: BigInt::from(257u16),
            ring_dimension: 8,
            rows: 2,
            columns: 2,
        };
        let evaluated = evaluator
            .eval_bounded_alternative(Alternative {
                coefficient: BigInt::one(),
                factors: vec![
                    AnalysisFactor::Identity(identity_type.clone()),
                    AnalysisFactor::Bounded(bounded.clone()),
                    AnalysisFactor::Identity(identity_type),
                ],
            })
            .expect("identity factors");
        assert_eq!(evaluated, bounded);
    }

    #[test]
    fn product_of_sum_keeps_dependency_aware_rules_termwise() {
        let ring = Ring::new(257, 8);
        let ty = ring.matrix_type((2, 2));
        let a = VirtualMat::bounded("a", ty.clone(), bounded_metadata(2, &["x"], None, true));
        let b = VirtualMat::bounded("b", ty.clone(), bounded_metadata(3, &["y"], None, true));
        let c = VirtualMat::bounded("c", ty, bounded_metadata(5, &["x"], None, true));
        let output =
            ring.input("output", (2, 2)).assume((a + b) * c).expect("well-typed assumption");
        let report = output_report(output).noise.expect("bounded output");

        let ring_sqrt = BigDecimal::from(8u8).sqrt().expect("sqrt(8)");
        let contraction = BigDecimal::from(2u8) * &ring_sqrt * &ring_sqrt;
        let ac = &contraction * BigDecimal::from(2u8) * BigDecimal::from(5u8);
        let bc = contraction.sqrt().expect("sqrt contraction") *
            BigDecimal::from(3u8) *
            BigDecimal::from(5u8);
        assert_eq!(report.bound, ac + bc);
        assert!(!report.clt_ready);
        assert!(matches!(report.dependencies, DependencySet::Known(ref ids) if ids.len() == 2));
    }

    #[test]
    fn mul_and_tensor_alternatives_are_yielded_one_at_a_time() {
        let ring = Ring::new(257, 8);
        let bounded = (0..4)
            .map(|index| {
                ring.input(format!("bounded-{index}"), (1, 1))
                    .assume(VirtualMat::bounded(
                        format!("bounded-{index}"),
                        ring.matrix_type((1, 1)),
                        bounded_metadata(index as i64 + 1, &["stream"], None, false),
                    ))
                    .expect("bounded assumption")
            })
            .collect::<Vec<_>>();
        let sum = bounded.iter().cloned().reduce(|left, right| left + right).expect("sum");
        let built = DslContext::new("alternative-streaming")
            .output("mul", sum.clone() * sum.clone())
            .expect("mul")
            .output("tensor", sum.clone().tensor(sum))
            .expect("tensor")
            .build()
            .expect("build");
        let elaborated = built.elaborate(&ParamEnv::default()).expect("elaboration");
        let arena_len = elaborated.expressions.len();
        let mut evaluator = Evaluator {
            graph: &elaborated,
            contexts: BTreeMap::new(),
            atoms: BTreeMap::new(),
            memo: BTreeMap::new(),
        };
        for (output, expected) in [("mul", 16), ("tensor", 16)] {
            let wire = elaborated.wire(&elaborated.outputs[output]).expect("output");
            let mut yielded = 0;
            let mut active = 0;
            let mut high_water = 0;
            evaluator
                .for_each_alternative(
                    wire.expression.expect("expression"),
                    &Assignment::new(),
                    &mut |_, _| {
                        active += 1;
                        high_water = high_water.max(active);
                        yielded += 1;
                        active -= 1;
                        Ok(())
                    },
                )
                .expect("stream alternatives");
            assert_eq!(yielded, expected, "{output}");
            assert_eq!(high_water, 1, "{output}");
            assert_eq!(elaborated.expressions.len(), arena_len, "{output}");
        }
    }

    #[test]
    fn tensor_uses_polynomial_contraction_without_matrix_inner_dimension() {
        let ring = Ring::new(257, 8);
        let report = output_report(ring.gaussian((2, 1), 2).tensor(ring.gaussian((3, 2), 3)));
        let expected = BigDecimal::from(8u8).sqrt().expect("sqrt(8)") *
            BigDecimal::from(13u8) *
            (BigDecimal::from(39u8) / BigDecimal::from(2u8));
        let noise = report.noise.expect("bounded tensor");
        assert_eq!(noise.bound, expected);
        assert_eq!((noise.rows, noise.columns), (6, 2));
        assert_eq!(noise.zero_rows, None);
        assert!(noise.clt_ready);
        assert!(matches!(noise.dependencies, DependencySet::Known(ref ids) if ids.len() == 2));
    }

    #[test]
    fn tensor_applies_each_additive_coefficient_once() {
        let ring = Ring::new(257, 8);
        let ty = ring.matrix_type((1, 1));
        let left = VirtualMat::bounded("left", ty.clone(), bounded_metadata(3, &["x"], None, true));
        let right = VirtualMat::bounded("right", ty, bounded_metadata(5, &["x"], None, true));
        let left = ring.input("left", (1, 1)).assume(left).expect("well-typed assumption");
        let right = ring.input("right", (1, 1)).assume(right).expect("well-typed assumption");
        let report =
            output_report((left.clone() + left).tensor(right)).noise.expect("bounded tensor");
        assert_eq!(report.bound, BigDecimal::from(240u16));
        assert!(!report.clt_ready);
        assert!(matches!(report.dependencies, DependencySet::Known(ref ids) if ids.len() == 1));
    }

    #[test]
    fn tensor_preserves_constant_and_one_ready_clt_rules() {
        let ring = Ring::new(257, 8);
        let scalar_type = ring.matrix_type((1, 1));
        let bounded = ring
            .input("bounded", (1, 1))
            .assume(VirtualMat::bounded(
                "bounded",
                scalar_type.clone(),
                bounded_metadata(5, &["bounded"], None, true),
            ))
            .unwrap();
        let constant =
            output_report(ring.identity(1).tensor(bounded)).noise.expect("constant tensor");
        assert_eq!(constant.bound, BigDecimal::from(5u8));
        assert!(!constant.clt_ready);

        let left = ring
            .input("left", (1, 1))
            .assume(VirtualMat::bounded(
                "left",
                scalar_type.clone(),
                bounded_metadata(2, &["x"], None, false),
            ))
            .unwrap();
        let right = ring
            .input("right", (1, 1))
            .assume(VirtualMat::bounded(
                "right",
                scalar_type,
                bounded_metadata(3, &["y"], None, true),
            ))
            .unwrap();
        let one_ready = output_report(left.tensor(right)).noise.expect("one-ready tensor");
        let expected = BigDecimal::from(8u8).sqrt().expect("sqrt(8)") * BigDecimal::from(6u8);
        assert_eq!(one_ready.bound, expected);
        assert!(!one_ready.clt_ready);
    }

    #[test]
    fn mixed_tensor_keeps_signal_and_bounded_contributions_separate() {
        let ring = Ring::new(257, 8);
        let left = ring.input("left-signal", (1, 1)) + ring.gaussian((1, 1), 2);
        let right = ring.input("right-signal", (1, 1)) + ring.gaussian((1, 1), 3);
        let report = output_report(left.tensor(right));
        assert!(report.has_signal);
        let expected = BigDecimal::from(8u8).sqrt().expect("sqrt(8)") *
            BigDecimal::from(13u8) *
            (BigDecimal::from(39u8) / BigDecimal::from(2u8));
        assert_eq!(report.noise.expect("bounded tensor alternative").bound, expected);
    }

    #[test]
    fn correlated_selections_are_joined_after_the_surrounding_product() {
        let ring = Ring::new(257, 8);
        let index = ring.input("index", (1, 1)).extract_coefficient(0);
        let left = index
            .clone()
            .select(vec![ring.gaussian((1, 1), 1), ring.gaussian((1, 1), 10)])
            .expect("left selection");
        let right = index
            .select(vec![ring.gaussian((1, 1), 10), ring.gaussian((1, 1), 1)])
            .expect("right selection");
        let report = output_report(left * right);
        let expected = BigDecimal::from(8u8).sqrt().expect("sqrt(8)") *
            (BigDecimal::from(13u8) / BigDecimal::from(2u8)) *
            BigDecimal::from(65u8);
        assert_eq!(report.noise.expect("bounded product").bound, expected);
    }
}
