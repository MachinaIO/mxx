//! Compact family coverage and numeric sequential-loop recurrence support.
//!
//! This module deliberately owns neither graph-wire memoization nor integer
//! analysis.  [`GraphLowerer`](super::lower::GraphLowerer) supplies one
//! symbolic element at a time.  Numeric recurrence evaluation is exact over
//! the supplied finite count; it has no policy ceiling.

use super::{
    analysis::{IntegerDomain, MxxAnalysis, MxxSort},
    identity::{
        AtomicSourceDescriptor, AtomicSourceId, AtomicSourceKey, BinderId, BinderKey,
        ResolvedIntExpr, SamplerDescriptorId, SamplerIdentity, TrapdoorDescriptorId,
        TrapdoorIdentity,
    },
    language::MxxLang,
    lower::LoweredInt,
};
use egg::{CostFunction, EGraph, Extractor, Id, Language, RecExpr};
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

/// The only two compact representations of a supported operational family.
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

/// A family residual together with its single closed element sort.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FamilyLoweringValue {
    pub element_type: MxxSort,
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
    ElementTypeMismatch { expected: MxxSort, actual: MxxSort },
    StorageMismatch,
    SelectorCaseCountMismatch { expected: usize, actual: usize },
    NonAffineSharedMaximum,
}

/// Maximizes an analysis-owned affine value over the retained closed family domains.
/// The work is linear in binder count and never enumerates their Cartesian product.
pub fn shared_affine_maximum(
    domain: &IntegerDomain,
    binder_domains: &[CoverageBinderDomain],
) -> Result<BigInt, FamilyCoverageError> {
    let IntegerDomain::Affine { constant, coefficients, binders } = domain else {
        return match domain {
            IntegerDomain::Exact(value) => Ok(value.clone()),
            IntegerDomain::IntervalOnly(_) => Err(FamilyCoverageError::NonAffineSharedMaximum),
            IntegerDomain::Affine { .. } => unreachable!(),
        };
    };
    if binders.len() != binder_domains.len() || coefficients.len() != binder_domains.len() {
        return Err(FamilyCoverageError::NonAffineSharedMaximum);
    }
    let mut maximum = constant.clone();
    for retained in binder_domains {
        let Some(interval) = binders.get(&retained.binder) else {
            return Err(FamilyCoverageError::NonAffineSharedMaximum);
        };
        if interval.minimum != retained.minimum || interval.maximum != retained.maximum {
            return Err(FamilyCoverageError::NonAffineSharedMaximum);
        }
        let Some(coefficient) = coefficients.get(&retained.binder) else {
            return Err(FamilyCoverageError::NonAffineSharedMaximum);
        };
        maximum += coefficient *
            if coefficient.sign() == num_bigint::Sign::Minus {
                &retained.minimum
            } else {
                &retained.maximum
            };
    }
    Ok(maximum)
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

/// Instantiates one shared representative by replacing only its owning binder.
/// Other binder nodes are retained, so nested independent domains stay symbolic.
/// Extracts one immutable, semantically valid template snapshot.  A lowering
/// job retains it by canonical root; later unions may make it nonminimal but
/// cannot invalidate an already valid expression.
/// Records every descriptor term reachable from `roots` before materializing
/// anything.  `Extractor::new` computes costs for the whole e-graph, so a
/// descriptor closure must share one extractor rather than construct one per
/// descriptor root.
fn snapshot_shared_template_closure<E>(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    templates: &mut std::collections::HashMap<Id, RecExpr<MxxLang>>,
    roots: impl IntoIterator<Item = Id>,
    progress: &mut dyn FnMut() -> Result<(), E>,
) -> Result<(), E> {
    let mut pending = roots
        .into_iter()
        .map(|root| egraph.find(root))
        .filter(|root| !templates.contains_key(root))
        .collect::<Vec<_>>();
    if pending.is_empty() {
        return Ok(());
    }
    struct ProgressAstSize<'a, E> {
        progress: &'a mut dyn FnMut() -> Result<(), E>,
        failure: std::rc::Rc<std::cell::RefCell<Option<E>>>,
    }
    impl<E> CostFunction<MxxLang> for ProgressAstSize<'_, E> {
        type Cost = usize;
        fn cost<C>(&mut self, enode: &MxxLang, mut costs: C) -> usize
        where
            C: FnMut(Id) -> usize,
        {
            if self.failure.borrow().is_none() &&
                let Err(error) = (self.progress)()
            {
                *self.failure.borrow_mut() = Some(error);
            }
            enode.fold(1, |sum, child| sum.saturating_add(costs(child)))
        }
    }
    let failure = std::rc::Rc::new(std::cell::RefCell::new(None));
    let extractor =
        Extractor::new(egraph, ProgressAstSize { progress, failure: std::rc::Rc::clone(&failure) });

    let mut scheduled = std::collections::HashSet::new();
    while let Some(root) = pending.pop() {
        let root = egraph.find(root);
        if !scheduled.insert(root) || templates.contains_key(&root) {
            continue;
        }
        let (_, expression) = extractor.find_best(root);
        if let Some(error) = failure.borrow_mut().take() {
            return Err(error);
        }
        for node in expression.as_ref() {
            let MxxLang::Atom { source, .. } = node else { continue };
            let Some(AtomicSourceDescriptor { key: AtomicSourceKey::Sampler(id), .. }) =
                egraph.analysis.symbols.atomic_sources.get(source.0)
            else {
                continue;
            };
            let sampler = egraph
                .analysis
                .symbols
                .samplers
                .get(id.0)
                .expect("every sampler Atom has an interned descriptor");
            match sampler {
                SamplerIdentity::Preimage { indices, public, trapdoor, target, .. } => {
                    pending.extend(indices.iter().copied());
                    pending.extend([*public, *target]);
                    let trapdoor = egraph
                        .analysis
                        .symbols
                        .trapdoors
                        .get(trapdoor.0)
                        .expect("every preimage sampler trapdoor is interned");
                    pending.extend(trapdoor.indices.iter().copied());
                    pending.push(trapdoor.public);
                }
                SamplerIdentity::DecomposedHash { indices, public, target, arguments, .. } => {
                    pending.extend(indices.iter().copied());
                    pending.extend([*public, *target]);
                    pending.extend(arguments.iter().copied());
                }
                SamplerIdentity::GadgetDecomposition { indices, public, target, .. } => {
                    pending.extend(indices.iter().copied());
                    pending.extend([*public, *target]);
                }
            }
        }
        templates.insert(root, expression);
    }
    Ok(())
}

pub fn instantiate_shared_element<E>(
    egraph: &mut EGraph<MxxLang, MxxAnalysis>,
    templates: &mut std::collections::HashMap<Id, RecExpr<MxxLang>>,
    representative: Id,
    binder: BinderId,
    replacement: Id,
    progress: &mut dyn FnMut() -> Result<(), E>,
) -> Result<Id, E> {
    // The snapshots are immutable valid expressions.  Complete the entire
    // descriptor reference closure before `egraph.add` can change canonical
    // classes or symbol-table descriptors.
    snapshot_shared_template_closure(egraph, templates, [representative], progress)?;

    enum Work {
        Enter(Id),
        Exit(Id, RecExpr<MxxLang>),
    }
    let mut completed = std::collections::HashMap::<Id, Id>::new();
    let mut scheduled = std::collections::HashSet::new();
    // Descriptor fields retain original e-class handles.  Remember their
    // canonical snapshot handle before materialization so re-interning below
    // never needs an immutable e-graph borrow while mutating symbol tables.
    let mut canonical = std::collections::HashMap::<Id, Id>::new();
    let replacement = egraph.find(replacement);
    completed.insert(replacement, replacement);
    canonical.insert(replacement, replacement);
    let mut work = vec![Work::Enter(representative)];
    while let Some(item) = work.pop() {
        match item {
            Work::Enter(root) => {
                let original = root;
                let root = egraph.find(root);
                canonical.insert(original, root);
                if !scheduled.insert(root) {
                    continue;
                }
                let template = templates
                    .get(&root)
                    .expect("descriptor closure snapshots every scheduled root")
                    .clone();
                let mut references = Vec::new();
                for node in template.as_ref() {
                    let MxxLang::Atom { source, .. } = node else { continue };
                    let Some(AtomicSourceDescriptor { key: AtomicSourceKey::Sampler(id), .. }) =
                        egraph.analysis.symbols.atomic_sources.get(source.0)
                    else {
                        continue
                    };
                    let sampler = egraph
                        .analysis
                        .symbols
                        .samplers
                        .get(id.0)
                        .expect("every sampler Atom has an interned descriptor");
                    match sampler {
                        SamplerIdentity::Preimage { indices, public, trapdoor, target, .. } => {
                            references.extend(indices.iter().copied());
                            references.extend([*public, *target]);
                            let trapdoor = egraph
                                .analysis
                                .symbols
                                .trapdoors
                                .get(trapdoor.0)
                                .expect("every preimage sampler trapdoor is interned");
                            references.extend(trapdoor.indices.iter().copied());
                            references.push(trapdoor.public);
                        }
                        SamplerIdentity::DecomposedHash {
                            indices,
                            public,
                            target,
                            arguments,
                            ..
                        } => {
                            references.extend(indices.iter().copied());
                            references.extend([*public, *target]);
                            references.extend(arguments.iter().copied());
                        }
                        SamplerIdentity::GadgetDecomposition {
                            indices, public, target, ..
                        } => {
                            references.extend(indices.iter().copied());
                            references.extend([*public, *target]);
                        }
                    }
                }
                work.push(Work::Exit(root, template));
                for reference in references.into_iter().rev() {
                    let reference_root = egraph.find(reference);
                    canonical.insert(reference, reference_root);
                    if reference_root != root {
                        work.push(Work::Enter(reference));
                    }
                }
            }
            Work::Exit(root, template) => {
                let mut ids = Vec::with_capacity(template.len());
                for node in template.as_ref() {
                    progress()?;
                    let rebuilt = if matches!(node, MxxLang::IntBinder(candidate) if *candidate == binder)
                    {
                        completed[&egraph.find(replacement)]
                    } else if let MxxLang::Atom { source, .. } = node {
                        let descriptor = egraph
                            .analysis
                            .symbols
                            .atomic_sources
                            .get(source.0)
                            .expect("every Atom source is interned")
                            .clone();
                        let AtomicSourceKey::Sampler(sampler_id) = descriptor.key else {
                            ids.push(
                                egraph.add(
                                    node.clone().map_children(|child| ids[usize::from(child)]),
                                ),
                            );
                            continue;
                        };
                        // Descriptor IDs are e-class IDs recorded by the
                        // same extractor snapshots queued above.  `completed`
                        // is keyed by those canonical snapshot IDs, so this
                        // lookup does not borrow the e-graph while the
                        // interners below are mutated.
                        let remap = |term: Id| completed[&canonical[&term]];
                        let sampler = egraph
                            .analysis
                            .symbols
                            .samplers
                            .get(sampler_id.0)
                            .expect("every sampler Atom has an interned descriptor")
                            .clone();
                        let sampler = match sampler {
                            SamplerIdentity::Preimage {
                                source,
                                indices,
                                public,
                                trapdoor,
                                target,
                                cutoff,
                            } => {
                                let trapdoor = egraph
                                    .analysis
                                    .symbols
                                    .trapdoors
                                    .get(trapdoor.0)
                                    .expect("every preimage sampler trapdoor is interned")
                                    .clone();
                                let trapdoor = TrapdoorIdentity {
                                    indices: trapdoor
                                        .indices
                                        .iter()
                                        .map(|term| remap(*term))
                                        .collect(),
                                    public: remap(trapdoor.public),
                                    ..trapdoor
                                };
                                let trapdoor = TrapdoorDescriptorId(
                                    egraph.analysis.symbols.trapdoors.intern(trapdoor),
                                );
                                SamplerIdentity::Preimage {
                                    source,
                                    indices: indices.iter().map(|term| remap(*term)).collect(),
                                    public: remap(public),
                                    trapdoor,
                                    target: remap(target),
                                    cutoff,
                                }
                            }
                            SamplerIdentity::DecomposedHash {
                                source,
                                indices,
                                public,
                                target,
                                arguments,
                                matrix_type,
                                base,
                                digit_count,
                                small,
                                range_proved,
                            } => SamplerIdentity::DecomposedHash {
                                source,
                                indices: indices.iter().map(|term| remap(*term)).collect(),
                                public: remap(public),
                                target: remap(target),
                                arguments: arguments.iter().map(|term| remap(*term)).collect(),
                                matrix_type,
                                base,
                                digit_count,
                                small,
                                range_proved,
                            },
                            SamplerIdentity::GadgetDecomposition {
                                source,
                                indices,
                                public,
                                target,
                                base,
                                digit_count,
                                small,
                                range_proved,
                            } => SamplerIdentity::GadgetDecomposition {
                                source,
                                indices: indices.iter().map(|term| remap(*term)).collect(),
                                public: remap(public),
                                target: remap(target),
                                base,
                                digit_count,
                                small,
                                range_proved,
                            },
                        };
                        let indices = match &sampler {
                            SamplerIdentity::Preimage { indices, .. } |
                            SamplerIdentity::DecomposedHash { indices, .. } |
                            SamplerIdentity::GadgetDecomposition { indices, .. } => indices.clone(),
                        };
                        let sampler = egraph.analysis.symbols.samplers.intern(sampler);
                        let source =
                            egraph.analysis.symbols.atomic_sources.intern(AtomicSourceDescriptor {
                                key: AtomicSourceKey::Sampler(SamplerDescriptorId(sampler)),
                                ..descriptor
                            });
                        egraph.add(MxxLang::Atom { source: AtomicSourceId(source), indices })
                    } else {
                        egraph.add(node.clone().map_children(|child| ids[usize::from(child)]))
                    };
                    ids.push(rebuilt);
                }
                completed.insert(root, *ids.last().expect("egg extraction is nonempty"));
            }
        }
    }
    Ok(completed[&egraph.find(representative)])
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
            return Err(FamilyCoverageError::ElementTypeMismatch {
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
    SizeOverflow { operation: &'static str },
}

impl VectorRecurrence {
    /// Evaluates the general O(C*T) path, or the affine O(S^3 log C) fast
    /// path when the whole transition has the required nonnegative form.
    pub fn evaluate(&self) -> Result<Box<[BigUint]>, RecurrenceFailure> {
        if self.initial.len() != self.transition.len() {
            return Err(RecurrenceFailure::ArityMismatch {
                expected: self.initial.len(),
                actual: self.transition.len(),
            });
        }
        if self.count.is_zero() {
            return Ok(self.initial.clone());
        }
        if let Some(rows) = self.affine_rows()? {
            return self.evaluate_affine(rows);
        }
        self.evaluate_general()
    }

    fn evaluate_general(&self) -> Result<Box<[BigUint]>, RecurrenceFailure> {
        let mut state = self.initial.to_vec();
        let mut iteration = BigUint::zero();
        while iteration < self.count {
            let mut memo = BTreeMap::new();
            let mut next = Vec::with_capacity(self.transition.len());
            for expression in self.transition.iter() {
                next.push(evaluate_expression(expression, &state, &iteration, &mut memo)?);
            }
            state = next;
            iteration += BigUint::one();
        }
        Ok(state.into_boxed_slice())
    }

    fn evaluate_affine(&self, rows: Vec<AffineRow>) -> Result<Box<[BigUint]>, RecurrenceFailure> {
        let state_size = self.initial.len();
        let dimension = state_size
            .checked_add(2)
            .ok_or(RecurrenceFailure::SizeOverflow { operation: "affine recurrence dimension" })?;
        dimension
            .checked_mul(dimension)
            .ok_or(RecurrenceFailure::SizeOverflow { operation: "affine recurrence matrix" })?;
        let mut matrix = vec![vec![BigUint::zero(); dimension]; dimension];
        for (row_index, row) in rows.iter().enumerate() {
            matrix[row_index][..state_size].clone_from_slice(&row.previous);
            matrix[row_index][state_size] = row.iteration.clone();
            matrix[row_index][state_size + 1] = row.constant.clone();
        }
        matrix[state_size][state_size] = BigUint::one();
        matrix[state_size + 1][state_size + 1] = BigUint::one();

        let power = matrix_power(matrix, &self.count)?;
        let mut input = self.initial.to_vec();
        input.push(BigUint::zero());
        input.push(BigUint::one());
        let output = matrix_vector_product(&power, &input)?;
        Ok(output[..state_size].to_vec().into_boxed_slice())
    }

    fn affine_rows(&self) -> Result<Option<Vec<AffineRow>>, RecurrenceFailure> {
        let row_width = self
            .initial
            .len()
            .checked_add(3)
            .ok_or(RecurrenceFailure::SizeOverflow { operation: "affine row width" })?;
        self.transition
            .len()
            .checked_mul(row_width)
            .ok_or(RecurrenceFailure::SizeOverflow { operation: "affine rows allocation" })?;
        Ok(self
            .transition
            .iter()
            .map(|expression| affine_expression(expression, self.initial.len()))
            .collect())
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

fn evaluate_expression(
    expression: &RecurrenceExpr,
    state: &[BigUint],
    iteration: &BigUint,
    memo: &mut BTreeMap<RecurrenceExpr, BigUint>,
) -> Result<BigUint, RecurrenceFailure> {
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
                        value
                            .to_biguint()
                            .ok_or(RecurrenceFailure::NegativeCutoff { cutoff: value })?
                    }
                    RecurrenceExpr::Previous(index) => {
                        state.get(*index).cloned().ok_or(RecurrenceFailure::PreviousOutOfRange {
                            index: *index,
                            state_size: state.len(),
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
                memo.insert(expression.clone(), value);
            }
        }
    }
    memo.remove(expression).ok_or(RecurrenceFailure::ArityMismatch { expected: 1, actual: 0 })
}

fn matrix_power(
    mut power: Vec<Vec<BigUint>>,
    exponent: &BigUint,
) -> Result<Vec<Vec<BigUint>>, RecurrenceFailure> {
    let dimension = power.len();
    dimension
        .checked_mul(dimension)
        .ok_or(RecurrenceFailure::SizeOverflow { operation: "affine power matrix" })?;
    let mut result = identity_matrix(dimension);
    let mut remaining = exponent.clone();
    while !remaining.is_zero() {
        if (&remaining & BigUint::one()) == BigUint::one() {
            result = matrix_product(&result, &power)?;
        }
        remaining >>= 1_usize;
        if !remaining.is_zero() {
            power = matrix_product(&power, &power)?;
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

fn matrix_product(
    left: &[Vec<BigUint>],
    right: &[Vec<BigUint>],
) -> Result<Vec<Vec<BigUint>>, RecurrenceFailure> {
    let dimension = left.len();
    dimension
        .checked_mul(dimension)
        .ok_or(RecurrenceFailure::SizeOverflow { operation: "affine product matrix" })?;
    let mut output = vec![vec![BigUint::zero(); dimension]; dimension];
    for row in 0..dimension {
        for column in 0..dimension {
            let mut value = BigUint::zero();
            for inner in 0..dimension {
                let product = &left[row][inner] * &right[inner][column];
                value += product;
            }
            output[row][column] = value;
        }
    }
    Ok(output)
}

fn matrix_vector_product(
    matrix: &[Vec<BigUint>],
    vector: &[BigUint],
) -> Result<Vec<BigUint>, RecurrenceFailure> {
    let mut output = Vec::with_capacity(matrix.len());
    for row in matrix {
        let mut value = BigUint::zero();
        for (coefficient, input) in row.iter().zip(vector) {
            let product = coefficient * input;
            value += product;
        }
        output.push(value);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::{
        analysis::MxxAnalysis,
        identity::{
            AtomicRelationRole, AtomicSourceDescriptor, BinderDescriptor, GraphWireSourceKey,
            OccurrenceScope, ProgramKey, ResolvedMatrixType, TrapdoorSourceKey, WireSourceKey,
        },
    };

    fn evaluate(recurrence: &VectorRecurrence) -> Result<Box<[BigUint]>, RecurrenceFailure> {
        recurrence.evaluate()
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
    fn affine_fast_path_handles_large_count() {
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
    fn max_forces_the_general_path_without_a_step_ceiling() {
        let recurrence = VectorRecurrence {
            initial: vec![BigUint::one()].into_boxed_slice(),
            transition: vec![RecurrenceExpr::Max(
                vec![RecurrenceExpr::Previous(0), RecurrenceExpr::Const(BigUint::from(2_u8))]
                    .into_boxed_slice(),
            )]
            .into_boxed_slice(),
            count: BigUint::from(101_u8),
        };
        assert_eq!(evaluate(&recurrence).unwrap().as_ref(), &[BigUint::from(2_u8)]);
    }

    #[test]
    fn shared_template_instantiation_replaces_only_the_owner_binder() {
        let mut analysis = MxxAnalysis::default();
        let scope = crate::operational_noise::identity::OccurrenceScope {
            program: crate::operational_noise::identity::ProgramKey::Ideal,
            definition: mxx_ir_core::FrozenGraphScopeId::Root,
            path: Box::new([]),
        };
        let owner =
            BinderKey { loop_scope: scope.clone(), loop_node: mxx_ir_core::NodeId(1), slot: 0 };
        let outer = BinderKey { loop_scope: scope, loop_node: mxx_ir_core::NodeId(2), slot: 0 };
        let owner_id = BinderId(analysis.symbols.binders.intern(BinderDescriptor {
            key: owner,
            minimum: 0.into(),
            maximum: 7.into(),
        }));
        let outer_id = BinderId(analysis.symbols.binders.intern(BinderDescriptor {
            key: outer,
            minimum: 0.into(),
            maximum: 3.into(),
        }));
        let mut egraph = EGraph::new(analysis);
        let owner_term = egraph.add(MxxLang::IntBinder(owner_id));
        let outer_term = egraph.add(MxxLang::IntBinder(outer_id));
        let representative = egraph.add(MxxLang::IntAdd([owner_term, outer_term]));
        let replacement = egraph.add(MxxLang::IntConst(5.into()));
        let mut templates = std::collections::HashMap::new();

        let mut progress_calls = 0;
        let instantiated = instantiate_shared_element(
            &mut egraph,
            &mut templates,
            representative,
            owner_id,
            replacement,
            &mut || {
                progress_calls += 1;
                Ok::<(), ()>(())
            },
        )
        .unwrap();
        let expression = egraph.id_to_expr(instantiated);

        assert!(expression.iter().any(|node| node == &MxxLang::IntConst(5.into())));
        assert!(expression.iter().any(|node| node == &MxxLang::IntBinder(outer_id)));
        assert!(!expression.iter().any(|node| node == &MxxLang::IntBinder(owner_id)));
        assert!(progress_calls >= 3, "extraction and copying report bounded work");

        for value in 0..1_000 {
            egraph.add(MxxLang::IntConst(value.into()));
        }
        let new_replacement = egraph.add(MxxLang::IntConst(6.into()));
        let mut unrelated_progress_calls = 0;
        instantiate_shared_element(
            &mut egraph,
            &mut templates,
            representative,
            owner_id,
            new_replacement,
            &mut || {
                unrelated_progress_calls += 1;
                Ok::<(), ()>(())
            },
        )
        .unwrap();
        assert!(
            unrelated_progress_calls < progress_calls,
            "a new replacement reuses the representative snapshot without re-extracting the graph"
        );

        let mut rejected = || Err::<(), _>("progress stopped");
        assert_eq!(
            instantiate_shared_element(
                &mut egraph,
                &mut templates,
                representative,
                owner_id,
                replacement,
                &mut rejected,
            ),
            Err("progress stopped")
        );
    }

    #[test]
    fn shared_template_instantiation_reuses_one_nontrivial_replacement_dag() {
        let mut analysis = MxxAnalysis::default();
        let scope = crate::operational_noise::identity::OccurrenceScope {
            program: crate::operational_noise::identity::ProgramKey::Ideal,
            definition: mxx_ir_core::FrozenGraphScopeId::Root,
            path: Box::new([]),
        };
        let owner = BinderKey { loop_scope: scope, loop_node: mxx_ir_core::NodeId(1), slot: 0 };
        let owner_id = BinderId(analysis.symbols.binders.intern(BinderDescriptor {
            key: owner,
            minimum: 0.into(),
            maximum: 7.into(),
        }));
        let mut egraph = EGraph::new(analysis);
        let owner_term = egraph.add(MxxLang::IntBinder(owner_id));
        let representative = egraph.add(MxxLang::IntAdd([owner_term, owner_term]));
        let two = egraph.add(MxxLang::IntConst(2.into()));
        let three = egraph.add(MxxLang::IntConst(3.into()));
        let replacement = egraph.add(MxxLang::IntAdd([two, three]));
        let mut templates = std::collections::HashMap::new();

        let instantiated = instantiate_shared_element(
            &mut egraph,
            &mut templates,
            representative,
            owner_id,
            replacement,
            &mut || Ok::<(), ()>(()),
        )
        .unwrap();
        let expression = egraph.id_to_expr(instantiated);
        assert_eq!(expression.len(), 4, "replacement is appended once, not once per binder");
        let MxxLang::IntAdd([left, right]) = expression[expression.root()] else {
            panic!("instantiated template must retain its outer addition");
        };
        assert_eq!(left, right, "both binder occurrences must use the same replacement root");
    }

    #[test]
    fn shared_sampler_instantiation_reinterns_cross_owner_descriptor_terms() {
        let mut analysis = MxxAnalysis::default();
        let scope = OccurrenceScope {
            program: ProgramKey::Ideal,
            definition: mxx_ir_core::FrozenGraphScopeId::Root,
            path: Box::new([]),
        };
        let owner =
            BinderKey { loop_scope: scope.clone(), loop_node: mxx_ir_core::NodeId(1), slot: 0 };
        let replacement_owner =
            BinderKey { loop_scope: scope.clone(), loop_node: mxx_ir_core::NodeId(2), slot: 0 };
        let owner_id = BinderId(analysis.symbols.binders.intern(BinderDescriptor {
            key: owner.clone(),
            minimum: 0.into(),
            maximum: 7.into(),
        }));
        let replacement_id = BinderId(analysis.symbols.binders.intern(BinderDescriptor {
            key: replacement_owner,
            minimum: 0.into(),
            maximum: 7.into(),
        }));
        let matrix_type = ResolvedMatrixType {
            modulus: ResolvedIntExpr::Const(17.into()),
            ring_dimension: ResolvedIntExpr::Const(1.into()),
            rows: ResolvedIntExpr::Const(1.into()),
            columns: ResolvedIntExpr::Const(1.into()),
        };
        let matrix_source = analysis.symbols.atomic_sources.intern(AtomicSourceDescriptor {
            key: AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from("matrix")),
            sort: MxxSort::Matrix(matrix_type.clone()),
            integer_domain: None,
            canonical_residue_convention: None,
            relation_role: None,
        });
        let mut egraph = EGraph::new(analysis);
        let owner_term = egraph.add(MxxLang::IntBinder(owner_id));
        let replacement = egraph.add(MxxLang::IntBinder(replacement_id));
        let public = egraph
            .add(MxxLang::Atom { source: AtomicSourceId(matrix_source), indices: Box::new([]) });
        let trapdoor =
            TrapdoorDescriptorId(egraph.analysis.symbols.trapdoors.intern(TrapdoorIdentity {
                source: TrapdoorSourceKey::ProtocolInput(crate::ProtocolInputId::from("trapdoor")),
                indices: Box::new([]),
                matrix_type: matrix_type.clone(),
                public,
                sigma_bits: 0,
                gadget_base: ResolvedIntExpr::Const(2.into()),
                digit_count: ResolvedIntExpr::Const(1.into()),
                preimage_cutoff: ResolvedIntExpr::Const(1.into()),
            }));
        let wire = WireSourceKey {
            scope: scope.clone(),
            wire: mxx_ir_core::WireRef { node: mxx_ir_core::NodeId(3), port: mxx_ir_core::Port(0) },
        };
        let sampler = egraph.analysis.symbols.samplers.intern(SamplerIdentity::Preimage {
            source: GraphWireSourceKey { wire, coordinate_binders: vec![owner].into_boxed_slice() },
            indices: vec![owner_term].into_boxed_slice(),
            public,
            trapdoor,
            target: public,
            cutoff: ResolvedIntExpr::Const(1.into()),
        });
        let source = egraph.analysis.symbols.atomic_sources.intern(AtomicSourceDescriptor {
            key: AtomicSourceKey::Sampler(SamplerDescriptorId(sampler)),
            sort: MxxSort::Matrix(matrix_type),
            integer_domain: None,
            canonical_residue_convention: None,
            relation_role: Some(AtomicRelationRole::Preimage),
        });
        let representative = egraph.add(MxxLang::Atom {
            source: AtomicSourceId(source),
            indices: vec![owner_term].into_boxed_slice(),
        });
        let mut templates = std::collections::HashMap::new();
        let instantiated = instantiate_shared_element(
            &mut egraph,
            &mut templates,
            representative,
            owner_id,
            replacement,
            &mut || Ok::<(), ()>(()),
        )
        .unwrap();
        assert!(egraph[instantiated].data.sort.is_ok());
        let MxxLang::Atom { source, indices } = &egraph[instantiated].nodes[0] else {
            panic!("instantiated sampler remains an Atom");
        };
        let AtomicSourceKey::Sampler(sampler) = egraph
            .analysis
            .symbols
            .atomic_sources
            .get(source.0)
            .expect("instantiated source is interned")
            .key
        else {
            panic!("sampler source")
        };
        let SamplerIdentity::Preimage {
            indices: recorded,
            public: recorded_public,
            trapdoor,
            target,
            ..
        } = egraph.analysis.symbols.samplers.get(sampler.0).expect("descriptor is interned")
        else {
            panic!("preimage descriptor")
        };
        assert_eq!(indices.len(), 1);
        assert_eq!(egraph.find(indices[0]), egraph.find(replacement));
        assert_eq!(egraph.find(recorded[0]), egraph.find(indices[0]));
        assert_eq!(egraph.find(*recorded_public), egraph.find(public));
        assert_eq!(egraph.find(*target), egraph.find(public));
        let trapdoor = egraph.analysis.symbols.trapdoors.get(trapdoor.0).unwrap();
        assert_eq!(egraph.find(trapdoor.public), egraph.find(public));
    }

    #[test]
    fn shared_affine_maximum_uses_nested_domain_endpoints_without_product() {
        let scope = crate::operational_noise::identity::OccurrenceScope {
            program: crate::operational_noise::identity::ProgramKey::Ideal,
            definition: mxx_ir_core::FrozenGraphScopeId::Root,
            path: Box::new([]),
        };
        let outer =
            BinderKey { loop_scope: scope.clone(), loop_node: mxx_ir_core::NodeId(1), slot: 0 };
        let inner = BinderKey { loop_scope: scope, loop_node: mxx_ir_core::NodeId(2), slot: 0 };
        let retained = vec![
            CoverageBinderDomain { binder: outer.clone(), minimum: 0.into(), maximum: 4.into() },
            CoverageBinderDomain { binder: inner.clone(), minimum: 1.into(), maximum: 6.into() },
        ];
        let domain = IntegerDomain::Affine {
            constant: 5.into(),
            coefficients: BTreeMap::from([(outer.clone(), 3.into()), (inner.clone(), (-2).into())]),
            binders: BTreeMap::from([
                (
                    outer,
                    crate::operational_noise::analysis::IntegerInterval::new(0.into(), 4.into())
                        .unwrap(),
                ),
                (
                    inner,
                    crate::operational_noise::analysis::IntegerInterval::new(1.into(), 6.into())
                        .unwrap(),
                ),
            ]),
        };
        assert_eq!(shared_affine_maximum(&domain, &retained).unwrap(), BigInt::from(15));
    }
}
