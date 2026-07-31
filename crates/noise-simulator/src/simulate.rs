use crate::{
    DependencySet, PolyMatrixNorm, PolyNorm, SimulatorContext, SourceId,
    poly_norm::high_probability_envelope_from_sigma,
};
use bigdecimal::BigDecimal;
use mxx_ir_symbolic::{
    atom::{
        AssumedMetadata, Atom, AtomClass, AtomId, AtomKind, DeclaredDependencies,
        DeclaredDependencyRef, DefExpr, ExternalSourceKind, SelectionDomainRef, SourceKind,
    },
    elaborate::{DecodeTarget, ElaboratedGraph},
    expr::{IntExpr, ParamEnv, RealExpr},
    node::ConstantMatrix,
    term::{Term, TermList, ViewDescriptor},
    types::{ConcreteMatrixType, WireId},
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
    #[error("wire {0:?} has no symbolic matrix term list")]
    MissingWire(WireId),
    #[error("atom is missing from the symbolic atom table: {0:?}")]
    MissingAtom(AtomId),
    #[error("bounded external source {kind:?} at {atom:?} has no manifest or unfold metadata")]
    UnsupportedExternal { atom: AtomId, kind: ExternalSourceKind },
    #[error("unsupported symbolic definition at atom {atom:?}: {reason}")]
    UnsupportedDefinition { atom: AtomId, reason: String },
    #[error("invalid numerical metadata at atom {atom:?}: {reason}")]
    InvalidMetadata { atom: AtomId, reason: String },
    #[error("matrix shape mismatch while evaluating atom {0:?}")]
    ShapeMismatch(AtomId),
    #[error("signal and noise were combined inside bounded atom {0:?}")]
    MixedBoundedAtom(AtomId),
}

#[derive(Clone)]
struct Estimate {
    signal: bool,
    noise: Option<PolyMatrixNorm>,
}

struct Evaluator<'a> {
    graph: &'a ElaboratedGraph,
    contexts: BTreeMap<usize, Arc<SimulatorContext>>,
    atoms: BTreeMap<AtomId, PolyMatrixNorm>,
    active: BTreeSet<AtomId>,
}

pub fn simulate(graph: &ElaboratedGraph) -> Result<NoiseReport, SimulationError> {
    let mut evaluator = Evaluator {
        graph,
        contexts: BTreeMap::new(),
        atoms: BTreeMap::new(),
        active: BTreeSet::new(),
    };
    let mut outputs = BTreeMap::new();
    for (name, wire) in &graph.outputs {
        let wire = WireId { instantiation_path: Vec::new(), wire: *wire };
        if evaluator.graph.wires.get(&wire).and_then(|wire| wire.terms.as_ref()).is_some() {
            outputs.insert(name.clone(), evaluator.eval_wire(&wire)?.into_report());
        }
    }
    let mut decode_targets = Vec::with_capacity(graph.decode_targets.len());
    for target in &graph.decode_targets {
        let estimate = evaluator.eval_wire(&target.input)?;
        let noise_bound = estimate
            .noise
            .as_ref()
            .map_or_else(BigDecimal::zero, PolyMatrixNorm::maximum_coefficient_bound);
        let modulus = evaluator
            .graph
            .wires
            .get(&target.input)
            .and_then(|wire| wire.wire_type.matrix_type())
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

    fn eval_wire(&mut self, wire: &WireId) -> Result<Estimate, SimulationError> {
        let symbolic_wire =
            self.graph.wires.get(wire).ok_or_else(|| SimulationError::MissingWire(wire.clone()))?;
        let terms = symbolic_wire
            .terms
            .clone()
            .ok_or_else(|| SimulationError::MissingWire(wire.clone()))?;
        if terms.terms.is_empty() {
            let ty = symbolic_wire
                .wire_type
                .matrix_type()
                .ok_or_else(|| SimulationError::MissingWire(wire.clone()))?
                .clone();
            let ctx = self.context(ty.ring_dimension);
            return Ok(Estimate {
                signal: false,
                noise: Some(exact_zero(ty.rows, ty.columns, ctx)),
            });
        }
        self.eval_terms(&terms)
    }

    fn eval_terms(&mut self, terms: &TermList) -> Result<Estimate, SimulationError> {
        let Some(domain) = self.first_indicator_domain(terms)? else {
            return self.eval_terms_without_select(terms);
        };
        let mut branches = BTreeMap::<u64, Vec<Term>>::new();
        let mut rest = Vec::new();
        for term in &terms.terms {
            let mut branch = None;
            let mut factors = Vec::with_capacity(term.factors.len());
            for factor in &term.factors {
                let atom = self.atom(&factor.atom)?;
                if atom.indicator.as_ref().is_some_and(|role| role.domain == domain) {
                    branch = Some(atom.indicator.as_ref().expect("checked").branch);
                } else {
                    factors.push(factor.clone());
                }
            }
            let stripped = Term { coefficient: term.coefficient.clone(), factors };
            if let Some(branch) = branch {
                branches.entry(branch).or_default().push(stripped);
            } else {
                rest.push(stripped);
            }
        }
        let mut branch_estimates = Vec::new();
        for branch in 0..selection_domain_count(&domain) {
            let branch_terms = branches.remove(&branch).unwrap_or_default();
            branch_estimates.push(self.eval_terms(&TermList { terms: branch_terms })?);
        }
        let selected = join_selection(branch_estimates);
        add_estimates(selected, self.eval_terms(&TermList { terms: rest })?)
    }

    fn first_indicator_domain(
        &self,
        terms: &TermList,
    ) -> Result<Option<SelectionDomainRef>, SimulationError> {
        for term in &terms.terms {
            for factor in &term.factors {
                if let Some(role) = &self.atom(&factor.atom)?.indicator {
                    return Ok(Some(role.domain.clone()));
                }
            }
        }
        Ok(None)
    }

    fn eval_terms_without_select(&mut self, terms: &TermList) -> Result<Estimate, SimulationError> {
        let mut signal = false;
        let mut noise = None;
        for term in &terms.terms {
            if self.term_has_signal(term)? {
                signal = true;
                continue;
            }
            let term_noise = self.eval_bounded_term(term, None)?;
            noise = Some(match noise {
                Some(current) => current + term_noise,
                None => term_noise,
            });
        }
        Ok(Estimate { signal, noise: noise.map(normalize_exact_zero) })
    }

    fn term_has_signal(&self, term: &Term) -> Result<bool, SimulationError> {
        for factor in &term.factors {
            if matches!(self.atom(&factor.atom)?.kind, AtomKind::Large) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn eval_bounded_term(
        &mut self,
        term: &Term,
        cap: Option<&BigInt>,
    ) -> Result<PolyMatrixNorm, SimulationError> {
        let mut value = None;
        for factor in &term.factors {
            let atom = self.atom(&factor.atom)?.clone();
            let mut factor_value = if matches!(atom.kind, AtomKind::Large) {
                let modulus =
                    cap.ok_or_else(|| SimulationError::MixedBoundedAtom(atom.id.clone()))?;
                self.capped_large(&atom, modulus)
            } else {
                let mut value = self.eval_atom(&atom.id)?;
                if let Some(modulus) = cap {
                    let half = BigDecimal::from(modulus.clone()) / BigDecimal::from(2u64);
                    if value.poly_norm.norm > half {
                        value.poly_norm.norm = half.clone();
                        value.poly_norm.sigma = half;
                    }
                }
                value
            };
            if let Some(view) = &factor.view {
                apply_view(&mut factor_value, view, &atom.matrix_type.modulus);
            }
            value = Some(match value {
                None => factor_value,
                Some(lhs) => multiply_factors(lhs, factor_value, &atom.id)?,
            });
        }
        let coefficient = BigDecimal::from(term.coefficient.abs());
        let value = match value {
            Some(value) if coefficient == BigDecimal::one() => value,
            Some(value) => value * coefficient,
            None => {
                let ctx = self.context(1);
                PolyMatrixNorm::from_parts(
                    1,
                    1,
                    PolyNorm::constant(ctx, coefficient),
                    None,
                    DependencySet::empty(),
                    false,
                )
            }
        };
        Ok(normalize_exact_zero(value))
    }

    fn eval_atom(&mut self, id: &AtomId) -> Result<PolyMatrixNorm, SimulationError> {
        if let Some(value) = self.atoms.get(id) {
            return Ok(value.clone());
        }
        let atom = self.atom(id)?.clone();
        if matches!(atom.kind, AtomKind::Large) {
            return Err(SimulationError::MixedBoundedAtom(id.clone()));
        }
        if !self.active.insert(id.clone()) {
            return Err(SimulationError::UnsupportedDefinition {
                atom: id.clone(),
                reason: "cyclic atom definition".to_owned(),
            });
        }
        let result = match &atom.class {
            AtomClass::Source { source } => self.eval_source(&atom, source),
            AtomClass::Assumed { metadata } => {
                let metadata =
                    metadata.as_ref().ok_or_else(|| SimulationError::InvalidMetadata {
                        atom: id.clone(),
                        reason: "bounded assumed atom has no declared metadata".to_owned(),
                    })?;
                self.eval_assumed(&atom, metadata)
            }
            AtomClass::Derived { definition } => self.eval_derived(&atom, definition),
        };
        self.active.remove(id);
        let value = normalize_exact_zero(result?);
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
        let value = match source {
            SourceKind::ConstantMatrix { value } => {
                let norm = constant_norm(value, ty, &atom.id)?;
                PolyMatrixNorm::from_parts(
                    ty.rows,
                    ty.columns,
                    PolyNorm::constant(ctx, norm),
                    None,
                    DependencySet::empty(),
                    false,
                )
            }
            SourceKind::UniformSample { minimum, maximum } => {
                let norm = BigDecimal::from(minimum.abs().max(maximum.abs()));
                PolyMatrixNorm::from_parts(
                    ty.rows,
                    ty.columns,
                    PolyNorm::new(ctx, norm),
                    None,
                    stable,
                    true,
                )
            }
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
                let derived_sigma = decimal_ratio(18, 10) *
                    &tau *
                    (BigDecimal::from(gadget_base.clone()) + BigDecimal::one()) *
                    &tau *
                    term;
                PolyMatrixNorm::from_parts(
                    ty.rows,
                    ty.columns,
                    PolyNorm::new(ctx, high_probability_envelope_from_sigma(&derived_sigma)),
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
                return Err(SimulationError::MixedBoundedAtom(atom.id.clone()));
            }
        };
        Ok(value)
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
        Ok(PolyMatrixNorm::from_parts(
            atom.matrix_type.rows,
            atom.matrix_type.columns,
            if metadata.is_const_poly {
                PolyNorm::constant(ctx, eval_real(&metadata.norm, &atom.id)?)
            } else {
                PolyNorm::new(ctx, eval_real(&metadata.norm, &atom.id)?)
            },
            metadata.zero_rows,
            dependencies,
            metadata.clt_ready,
        ))
    }

    fn eval_derived(
        &mut self,
        atom: &Atom,
        definition: &DefExpr,
    ) -> Result<PolyMatrixNorm, SimulationError> {
        match definition {
            DefExpr::TermList(terms) | DefExpr::Fold(terms) => {
                let estimate = self.eval_terms(terms)?;
                self.require_noise(atom, estimate)
            }
            DefExpr::Concat { inputs, .. } => {
                let values = inputs
                    .iter()
                    .map(|input| self.eval_atom(input))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(join_opaque(
                    values,
                    &atom.matrix_type,
                    self.context(atom.matrix_type.ring_dimension),
                ))
            }
            DefExpr::Reshape { input, .. } => {
                let value = self.eval_atom(input)?;
                let zero_rows = (value.poly_norm.is_const_poly &&
                    value.poly_norm.norm.is_zero() &&
                    value.zero_rows == Some(value.nrow))
                .then_some(atom.matrix_type.rows);
                Ok(PolyMatrixNorm::from_parts(
                    atom.matrix_type.rows,
                    atom.matrix_type.columns,
                    value.poly_norm,
                    zero_rows,
                    value.deps,
                    false,
                ))
            }
            DefExpr::ConstantCoefficient { input, .. } => {
                let value = self.eval_atom(input)?;
                let zero_rows = (value.poly_norm.norm.is_zero() &&
                    value.zero_rows == Some(value.nrow))
                .then_some(atom.matrix_type.rows);
                Ok(PolyMatrixNorm::from_parts(
                    atom.matrix_type.rows,
                    atom.matrix_type.columns,
                    value.poly_norm.into_constant_poly(),
                    zero_rows,
                    value.deps,
                    false,
                ))
            }
            DefExpr::Indicator { .. } => Ok(PolyMatrixNorm::from_parts(
                1,
                1,
                PolyNorm::constant(
                    self.context(atom.matrix_type.ring_dimension),
                    BigDecimal::one(),
                ),
                None,
                DependencySet::empty(),
                false,
            )),
            DefExpr::ModDownError { input, source_modulus, target_modulus, .. } => {
                self.eval_mod_down_error(atom, input, source_modulus, target_modulus)
            }
            DefExpr::ModUpError { input, source_modulus, .. } => {
                self.eval_mod_up_error(atom, input, source_modulus)
            }
            DefExpr::Tensor { .. } => Err(SimulationError::UnsupportedDefinition {
                atom: atom.id.clone(),
                reason: "tensor is unsupported in the initial simulator".to_owned(),
            }),
            DefExpr::ModDownImage { .. } |
            DefExpr::ModUpLift { .. } |
            DefExpr::CrtRecompose { .. } => Err(SimulationError::MixedBoundedAtom(atom.id.clone())),
        }
    }

    fn require_noise(
        &mut self,
        atom: &Atom,
        estimate: Estimate,
    ) -> Result<PolyMatrixNorm, SimulationError> {
        if estimate.signal {
            return Err(SimulationError::MixedBoundedAtom(atom.id.clone()));
        }
        let ctx = self.context(atom.matrix_type.ring_dimension);
        Ok(estimate.noise.unwrap_or_else(|| zero_matrix(atom, ctx)))
    }

    fn eval_mod_down_error(
        &mut self,
        atom: &Atom,
        input: &TermList,
        source_modulus: &BigInt,
        target_modulus: &BigInt,
    ) -> Result<PolyMatrixNorm, SimulationError> {
        let mut prefix_rounding = BigDecimal::zero();
        let mut dependencies = DependencySet::empty();
        let mut bounded = Vec::new();
        for term in &input.terms {
            let position = term.factors.iter().position(|factor| {
                self.atom(&factor.atom)
                    .is_ok_and(|factor_atom| matches!(factor_atom.kind, AtomKind::Large))
            });
            if let Some(position) = position {
                let prefix = Term {
                    coefficient: term.coefficient.clone(),
                    factors: term.factors[..position].to_vec(),
                };
                let prefix_norm = self.eval_bounded_term(&prefix, None)?;
                dependencies = dependencies.union(&prefix_norm.deps);
                let mut contribution = prefix_norm.maximum_coefficient_bound() *
                    self.context(atom.matrix_type.ring_dimension).ring_dim_sqrt.clone();
                if let Some(last) = prefix.factors.last() {
                    let last = self.atom(&last.atom)?;
                    if !last.matrix_type.is_scalar() {
                        contribution *= BigDecimal::from(last.matrix_type.columns as u64);
                    }
                }
                prefix_rounding += contribution / BigDecimal::from(2u64);
            } else {
                bounded.push(term.clone());
            }
        }
        let bounded = self.eval_terms(&TermList { terms: bounded })?;
        let bounded = bounded.noise.map_or_else(BigDecimal::zero, |value| {
            dependencies = dependencies.union(&value.deps);
            value.maximum_coefficient_bound()
        });
        let norm = BigDecimal::from(1u64) / BigDecimal::from(2u64) +
            prefix_rounding +
            bounded * BigDecimal::from(target_modulus.clone()) /
                BigDecimal::from(source_modulus.clone());
        Ok(plain_bound_with_deps(
            atom,
            self.context(atom.matrix_type.ring_dimension),
            norm,
            dependencies,
        ))
    }

    fn eval_mod_up_error(
        &mut self,
        atom: &Atom,
        input: &TermList,
        source_modulus: &BigInt,
    ) -> Result<PolyMatrixNorm, SimulationError> {
        let mut integer_norm = BigDecimal::zero();
        let mut dependencies = DependencySet::empty();
        for term in &input.terms {
            let term = self.eval_bounded_term(term, Some(source_modulus))?;
            integer_norm += term.maximum_coefficient_bound();
            dependencies = dependencies.union(&term.deps);
        }
        let norm = integer_norm / BigDecimal::from(source_modulus.clone()) +
            BigDecimal::from(1u64) / BigDecimal::from(2u64);
        Ok(plain_bound_with_deps(
            atom,
            self.context(atom.matrix_type.ring_dimension),
            norm,
            dependencies,
        ))
    }

    fn capped_large(&mut self, atom: &Atom, modulus: &BigInt) -> PolyMatrixNorm {
        PolyMatrixNorm::from_parts(
            atom.matrix_type.rows,
            atom.matrix_type.columns,
            PolyNorm::new(
                self.context(atom.matrix_type.ring_dimension),
                BigDecimal::from(modulus.clone()) / BigDecimal::from(2u64),
            ),
            None,
            DependencySet::Unknown,
            false,
        )
    }

    fn atom(&self, id: &AtomId) -> Result<&Atom, SimulationError> {
        self.graph.atoms.get(id).ok_or_else(|| SimulationError::MissingAtom(id.clone()))
    }
}

fn add_estimates(lhs: Estimate, rhs: Estimate) -> Result<Estimate, SimulationError> {
    let noise = match (lhs.noise, rhs.noise) {
        (Some(lhs), Some(rhs)) => Some(normalize_exact_zero(lhs + rhs)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    };
    Ok(Estimate { signal: lhs.signal || rhs.signal, noise })
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

fn selection_domain_count(domain: &SelectionDomainRef) -> u64 {
    match domain {
        SelectionDomainRef::Local(domain) | SelectionDomainRef::Imported { domain, .. } => {
            domain.count
        }
    }
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
    let zero_rows = values.iter().map(|value| value.zero_rows).all_equal().unwrap_or(None);
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

trait AllEqualOption: Iterator {
    fn all_equal(mut self) -> Option<Self::Item>
    where
        Self: Sized,
        Self::Item: Clone + PartialEq,
    {
        let first = self.next()?;
        if self.all(|value| value == first) { Some(first) } else { None }
    }
}
impl<I: Iterator> AllEqualOption for I {}

fn multiply_factors(
    lhs: PolyMatrixNorm,
    rhs: PolyMatrixNorm,
    atom: &AtomId,
) -> Result<PolyMatrixNorm, SimulationError> {
    if lhs.nrow == 1 && lhs.ncol == 1 && !(rhs.nrow == 1 && rhs.ncol == 1) {
        return Ok(scale_matrix(rhs, lhs));
    }
    if rhs.nrow == 1 && rhs.ncol == 1 && !(lhs.nrow == 1 && lhs.ncol == 1) {
        return Ok(scale_matrix(lhs, rhs));
    }
    if lhs.ncol != rhs.nrow {
        return Err(SimulationError::ShapeMismatch(atom.clone()));
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

fn apply_view(value: &mut PolyMatrixNorm, view: &ViewDescriptor, source_modulus: &BigInt) {
    if let Some(target_modulus) = &view.modulus_cast {
        let cap_modulus = source_modulus.min(target_modulus);
        let cap = BigDecimal::from(cap_modulus.clone()) / BigDecimal::from(2u64);
        if value.poly_norm.norm > cap {
            value.poly_norm.norm = cap.clone();
            value.poly_norm.sigma = cap;
        }
    }
    if view.transpose {
        std::mem::swap(&mut value.nrow, &mut value.ncol);
        value.ncol_sqrt =
            BigDecimal::from(value.ncol as u64).sqrt().expect("positive column count");
        value.zero_rows = None;
    }
    if let Some(rows) = view.row_range {
        value.nrow = rows.end.saturating_sub(rows.start);
        value.zero_rows = None;
    }
    if let Some(columns) = view.column_range {
        value.ncol = columns.end.saturating_sub(columns.start);
        value.ncol_sqrt =
            BigDecimal::from(value.ncol as u64).sqrt().expect("positive column count");
    }
}

fn zero_matrix(atom: &Atom, ctx: Arc<SimulatorContext>) -> PolyMatrixNorm {
    exact_zero(atom.matrix_type.rows, atom.matrix_type.columns, ctx)
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

fn plain_bound_with_deps(
    atom: &Atom,
    ctx: Arc<SimulatorContext>,
    norm: BigDecimal,
    dependencies: DependencySet,
) -> PolyMatrixNorm {
    PolyMatrixNorm::from_parts(
        atom.matrix_type.rows,
        atom.matrix_type.columns,
        PolyNorm::new(ctx, norm),
        None,
        dependencies,
        false,
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
    let value = match expression {
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
    };
    Ok(value)
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
    use mxx_ir_symbolic::{
        elaborate::elaborate,
        expr::{IntExpr, Rational},
        graph::Graph,
        node::{MatrixBinaryOp, Node, NodeKind},
        types::{MatrixType, NodeId, Port, WireRef},
    };

    fn matrix_type(rows: i64, columns: i64) -> MatrixType {
        MatrixType {
            modulus: IntExpr::constant(97),
            ring_dimension: IntExpr::constant(8),
            rows: IntExpr::constant(rows),
            columns: IntExpr::constant(columns),
        }
    }

    fn wire(node: u64, port: u32) -> WireRef {
        WireRef { node: NodeId(node), port: Port(port) }
    }

    fn rational(value: i64) -> RealExpr {
        RealExpr::Rational(Rational::new(BigInt::from(value), BigInt::one()).expect("rational"))
    }

    fn graph(name: &str, nodes: Vec<Node>, output: WireRef) -> Graph {
        Graph {
            name: name.to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes,
            outputs: BTreeMap::from([("out".to_owned(), output)]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        }
    }

    #[test]
    fn gaussian_source_uses_the_existing_six_point_five_sigma_envelope() {
        let graph = graph(
            "gaussian",
            vec![Node {
                id: NodeId(1),
                kind: NodeKind::GaussianSample {
                    matrix_type: matrix_type(2, 3),
                    sigma: rational(4),
                },
                args: Vec::new(),
            }],
            wire(1, 0),
        );
        let elaborated = elaborate(&graph, &ParamEnv::default()).expect("elaboration");
        let report = simulate(&elaborated).expect("simulation");
        assert_eq!(
            report.outputs["out"].noise.as_ref().expect("noise").bound,
            BigDecimal::from(26u64)
        );
    }

    #[test]
    fn constant_coefficient_preserves_bound_and_dependency_as_constant_polynomial() {
        let graph = graph(
            "constant-coefficient",
            vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::GaussianSample {
                        matrix_type: matrix_type(1, 1),
                        sigma: rational(4),
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::ConstantCoefficient { position: IntExpr::constant(2) },
                    args: vec![wire(1, 0)],
                },
            ],
            wire(2, 0),
        );
        let elaborated = elaborate(&graph, &ParamEnv::default()).expect("elaboration");
        let report = simulate(&elaborated).expect("simulation");
        let noise = report.outputs["out"].noise.as_ref().expect("noise");
        assert_eq!(noise.bound, BigDecimal::from(26u64));
        assert!(noise.is_const_poly);
        assert_eq!(
            noise.dependencies,
            DependencySet::singleton(stable_source_id(&AtomId::Local {
                instantiation_path: Vec::new(),
                node: NodeId(1),
                port: 0,
            }))
        );
        assert!(!noise.clt_ready);
    }

    #[test]
    fn selection_uses_branch_max_and_joins_metadata_instead_of_summing_branches() {
        let graph = graph(
            "selection",
            vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::ConstantInt(BigInt::zero()),
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::GaussianSample {
                        matrix_type: matrix_type(1, 1),
                        sigma: rational(2),
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(3),
                    kind: NodeKind::GaussianSample {
                        matrix_type: matrix_type(1, 1),
                        sigma: rational(5),
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(4),
                    kind: NodeKind::Select { count: IntExpr::constant(2) },
                    args: vec![wire(1, 0), wire(2, 0), wire(3, 0)],
                },
            ],
            wire(4, 0),
        );
        let elaborated = elaborate(&graph, &ParamEnv::default()).expect("elaboration");
        let report = simulate(&elaborated).expect("simulation");
        let noise = report.outputs["out"].noise.as_ref().expect("noise");
        assert_eq!(noise.bound, decimal_ratio(65, 2));
        assert!(!noise.clt_ready);
        assert_eq!(
            noise.dependencies,
            DependencySet::singleton(stable_source_id(&AtomId::Local {
                instantiation_path: Vec::new(),
                node: NodeId(2),
                port: 0,
            }))
            .union(&DependencySet::singleton(stable_source_id(&AtomId::Local {
                instantiation_path: Vec::new(),
                node: NodeId(3),
                port: 0,
            })))
        );
    }

    #[test]
    fn preimage_uses_explicit_layout_and_the_existing_derived_sigma_formula() {
        let graph = graph(
            "preimage",
            vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::TrapdoorSample {
                        matrix_type: matrix_type(1, 3),
                        sigma: rational(3),
                        gadget_base: IntExpr::constant(2),
                        digit_count: IntExpr::constant(1),
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::GaussianSample {
                        matrix_type: matrix_type(1, 1),
                        sigma: rational(1),
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(3),
                    kind: NodeKind::PreimageSample { matrix_type: matrix_type(3, 1) },
                    args: vec![wire(1, 1), wire(2, 0)],
                },
            ],
            wire(3, 0),
        );
        let elaborated = elaborate(&graph, &ParamEnv::default()).expect("elaboration");
        let report = simulate(&elaborated).expect("simulation");
        let ring_sqrt = BigDecimal::from(8u64).sqrt().expect("sqrt(8)");
        let derived_sigma = decimal_ratio(18, 10) *
            BigDecimal::from(3u64) *
            BigDecimal::from(3u64) *
            BigDecimal::from(3u64) *
            (ring_sqrt.clone() +
                BigDecimal::from(2u64).sqrt().expect("sqrt(2)") * ring_sqrt +
                decimal_ratio(47, 10));
        assert_eq!(
            report.outputs["out"].noise.as_ref().expect("noise").bound,
            high_probability_envelope_from_sigma(&derived_sigma)
        );
    }

    #[test]
    fn gadget_trapdoor_preimage_uses_balanced_digit_rule() {
        let graph = graph(
            "gadget-preimage",
            vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::GadgetTrapdoor {
                        matrix_type: matrix_type(1, 7),
                        base: IntExpr::constant(2),
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::GaussianSample {
                        matrix_type: matrix_type(1, 1),
                        sigma: rational(1),
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(3),
                    kind: NodeKind::PreimageSample { matrix_type: matrix_type(7, 1) },
                    args: vec![wire(1, 0), wire(2, 0)],
                },
            ],
            wire(3, 0),
        );
        let elaborated = elaborate(&graph, &ParamEnv::default()).expect("elaboration");
        let report = simulate(&elaborated).expect("simulation");
        let digit_sigma =
            (BigDecimal::from(6u64) / BigDecimal::from(12u64)).sqrt().expect("digit sigma");
        assert_eq!(
            report.outputs["out"].noise.as_ref().expect("noise").bound,
            high_probability_envelope_from_sigma(&digit_sigma)
        );
    }

    #[test]
    fn real_expression_allows_negative_intermediate_when_final_norm_is_nonnegative() {
        let atom = AtomId::Virtual { name: "norm".to_owned() };
        let expression = RealExpr::Add(
            Box::new(RealExpr::Sub(
                Box::new(rational(1)),
                Box::new(RealExpr::Sqrt(Box::new(rational(4)))),
            )),
            Box::new(rational(2)),
        );
        assert_eq!(eval_real(&expression, &atom).expect("valid final norm"), BigDecimal::one());
    }

    #[test]
    fn exact_zero_derived_atom_has_constant_empty_dependency_metadata() {
        let graph = graph(
            "zero-reshape",
            vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::ConstantMatrix {
                        matrix_type: matrix_type(1, 1),
                        value: ConstantMatrix::Zero,
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::Reshape {
                        rows: IntExpr::constant(1),
                        columns: IntExpr::constant(1),
                    },
                    args: vec![wire(1, 0)],
                },
            ],
            wire(2, 0),
        );
        let elaborated = elaborate(&graph, &ParamEnv::default()).expect("elaboration");
        let report = simulate(&elaborated).expect("simulation");
        let noise = report.outputs["out"].noise.as_ref().expect("explicit zero");
        assert_eq!(noise.bound, BigDecimal::zero());
        assert!(noise.is_const_poly);
        assert_eq!(noise.zero_rows, Some(1));
        assert_eq!(noise.dependencies, DependencySet::empty());
        assert!(!noise.clt_ready);
    }

    #[test]
    fn direct_zero_and_zero_arithmetic_use_canonical_metadata() {
        for (name, nodes, output) in [
            (
                "direct-zero",
                vec![Node {
                    id: NodeId(1),
                    kind: NodeKind::ConstantMatrix {
                        matrix_type: matrix_type(2, 2),
                        value: ConstantMatrix::Zero,
                    },
                    args: Vec::new(),
                }],
                wire(1, 0),
            ),
            (
                "zero-add",
                vec![
                    Node {
                        id: NodeId(1),
                        kind: NodeKind::ConstantMatrix {
                            matrix_type: matrix_type(2, 2),
                            value: ConstantMatrix::Zero,
                        },
                        args: Vec::new(),
                    },
                    Node {
                        id: NodeId(2),
                        kind: NodeKind::MatrixBinary(MatrixBinaryOp::Add),
                        args: vec![wire(1, 0), wire(1, 0)],
                    },
                ],
                wire(2, 0),
            ),
            (
                "zero-multiply",
                vec![
                    Node {
                        id: NodeId(1),
                        kind: NodeKind::ConstantMatrix {
                            matrix_type: matrix_type(2, 2),
                            value: ConstantMatrix::Zero,
                        },
                        args: Vec::new(),
                    },
                    Node {
                        id: NodeId(2),
                        kind: NodeKind::GaussianSample {
                            matrix_type: matrix_type(2, 2),
                            sigma: rational(3),
                        },
                        args: Vec::new(),
                    },
                    Node {
                        id: NodeId(3),
                        kind: NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
                        args: vec![wire(1, 0), wire(2, 0)],
                    },
                ],
                wire(3, 0),
            ),
        ] {
            let elaborated =
                elaborate(&graph(name, nodes, output), &ParamEnv::default()).expect("elaboration");
            let report = simulate(&elaborated).expect("simulation");
            let noise = report.outputs["out"].noise.as_ref().expect("explicit zero");
            assert_eq!(noise.bound, BigDecimal::zero(), "{name}");
            assert!(noise.is_const_poly, "{name}");
            assert_eq!(noise.zero_rows, Some(2), "{name}");
            assert_eq!(noise.dependencies, DependencySet::empty(), "{name}");
            assert!(!noise.clt_ready, "{name}");
        }
    }

    #[test]
    fn modulus_cast_caps_a_factor_at_half_the_smaller_modulus() {
        let ctx = Arc::new(SimulatorContext::new(
            BigDecimal::from(8u64).sqrt().expect("sqrt(8)"),
            BigDecimal::from(2u64),
            8,
            7,
            1,
        ));
        let mut value = PolyMatrixNorm::from_parts(
            1,
            1,
            PolyNorm::new(ctx, BigDecimal::from(80u64)),
            None,
            DependencySet::Unknown,
            false,
        );
        let view = ViewDescriptor {
            modulus_cast: Some(BigInt::from(101u64)),
            ..ViewDescriptor::default()
        };
        apply_view(&mut value, &view, &BigInt::from(97u64));
        assert_eq!(value.maximum_coefficient_bound(), decimal_ratio(97, 2));
    }

    #[test]
    fn all_zero_opaque_join_uses_canonical_zero_metadata() {
        let ctx = Arc::new(SimulatorContext::new(
            BigDecimal::from(8u64).sqrt().expect("sqrt(8)"),
            BigDecimal::from(2u64),
            8,
            7,
            1,
        ));
        let value = exact_zero(1, 1, ctx.clone());
        let joined = join_opaque(
            vec![value.clone(), value],
            &ConcreteMatrixType {
                modulus: BigInt::from(97u64),
                ring_dimension: 8,
                rows: 2,
                columns: 1,
            },
            ctx,
        );
        assert_eq!(joined.maximum_coefficient_bound(), BigDecimal::zero());
        assert!(joined.poly_norm.is_const_poly);
        assert_eq!(joined.zero_rows, Some(2));
        assert_eq!(joined.deps, DependencySet::empty());
        assert!(!joined.clt_ready);
    }

    #[test]
    fn selection_metadata_includes_an_absent_branch_as_exact_zero() {
        let ctx = Arc::new(SimulatorContext::new(
            BigDecimal::from(8u64).sqrt().expect("sqrt(8)"),
            BigDecimal::from(2u64),
            8,
            7,
            1,
        ));
        let nonzero = PolyMatrixNorm::from_parts(
            2,
            1,
            PolyNorm::new(ctx, BigDecimal::one()),
            Some(1),
            DependencySet::Unknown,
            true,
        );
        let joined = join_selection(vec![
            Estimate { signal: false, noise: Some(nonzero) },
            Estimate { signal: false, noise: None },
        ]);
        let noise = joined.noise.expect("one branch has noise");
        assert_eq!(noise.zero_rows, None);
        assert!(!noise.clt_ready);
    }

    #[test]
    fn bounded_external_input_without_manifest_or_unfold_is_unsupported() {
        let mut graph = graph(
            "external",
            vec![Node {
                id: NodeId(1),
                kind: NodeKind::Input {
                    name: "x".to_owned(),
                    wire_type: mxx_ir_symbolic::types::WireType::Preimage(matrix_type(1, 1)),
                    artifact: None,
                },
                args: Vec::new(),
            }],
            wire(1, 0),
        );
        graph
            .input_types
            .insert("x".to_owned(), mxx_ir_symbolic::types::WireType::Preimage(matrix_type(1, 1)));
        let elaborated = elaborate(&graph, &ParamEnv::default()).expect("elaboration");
        assert!(matches!(
            simulate(&elaborated),
            Err(SimulationError::UnsupportedExternal { kind: ExternalSourceKind::Preimage, .. })
        ));
    }

    #[test]
    fn simulation_visits_decode_inputs_even_when_graph_outputs_are_scalar() {
        let graph = graph(
            "decode",
            vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::GaussianSample {
                        matrix_type: matrix_type(1, 1),
                        sigma: rational(1),
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::ThresholdDecode {
                        plaintext_modulus: IntExpr::constant(2),
                        length: IntExpr::constant(1),
                        output_bool: false,
                    },
                    args: vec![wire(1, 0)],
                },
            ],
            wire(2, 0),
        );
        let elaborated = elaborate(&graph, &ParamEnv::default()).expect("elaboration");
        let report = simulate(&elaborated).expect("simulation");
        assert!(report.outputs.is_empty());
        assert_eq!(report.decode_targets.len(), 1);
        assert_eq!(
            report.decode_targets[0].estimate.noise.as_ref().expect("decode noise").bound,
            decimal_ratio(13, 2)
        );
        assert_eq!(report.decode_targets[0].threshold, decimal_ratio(97, 4));
        assert!(report.decode_targets[0].within_threshold);
    }

    #[test]
    fn simulation_does_not_visit_unreachable_bounded_inputs() {
        let mut graph = graph(
            "target-only",
            vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::Input {
                        name: "unused".to_owned(),
                        wire_type: mxx_ir_symbolic::types::WireType::Preimage(matrix_type(1, 1)),
                        artifact: None,
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::GaussianSample {
                        matrix_type: matrix_type(1, 1),
                        sigma: rational(1),
                    },
                    args: Vec::new(),
                },
            ],
            wire(2, 0),
        );
        graph.input_types.insert(
            "unused".to_owned(),
            mxx_ir_symbolic::types::WireType::Preimage(matrix_type(1, 1)),
        );
        let elaborated = elaborate(&graph, &ParamEnv::default()).expect("elaboration");
        assert!(simulate(&elaborated).is_ok());
    }
}
