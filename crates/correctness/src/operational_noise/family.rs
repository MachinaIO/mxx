//! Compact family coverage and numeric sequential-loop recurrence support.
//!
//! This module deliberately owns neither graph-wire memoization nor integer
//! analysis.  [`GraphLowerer`](super::lower::GraphLowerer) supplies one
//! symbolic element at a time.  Numeric recurrence evaluation is exact over
//! the supplied finite count; it has no policy ceiling. Runtime Switches may
//! omit only a suffix proved unreachable by the selector's authoritative interval.

use super::{
    analysis::{IntegerDomain, MxxAnalysis, MxxSort},
    identity::{
        AtomicSourceDescriptor, AtomicSourceId, AtomicSourceKey, BinderId, BinderKey,
        ResolvedIntExpr, SamplerDescriptorId, SamplerIdentity, SequentialRecurrenceDescriptor,
        SequentialRecurrenceId, TrapdoorDescriptorId, TrapdoorIdentity,
    },
    language::MxxLang,
    lower::LoweredInt,
};
use egg::{EGraph, Id, Language};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};
use std::collections::BTreeMap;

/// Adds one runtime Switch, retaining every case through the authoritative
/// maximum selector value. Invalid or unavailable selector facts preserve the
/// complete physical case list for the owning validation path.
pub(crate) fn add_runtime_switch(
    egraph: &mut EGraph<MxxLang, MxxAnalysis>,
    selector: Id,
    cases: &[Id],
) -> Id {
    let selector = egraph.find(selector);
    let retained = egraph[selector]
        .data
        .integer_domain
        .as_ref()
        .and_then(|domain| domain.interval().ok())
        .filter(|interval| interval.minimum >= BigInt::zero())
        .and_then(|interval| interval.maximum.to_usize())
        .and_then(|maximum| (maximum < cases.len()).then_some(maximum + 1))
        .unwrap_or(cases.len());
    let mut children = Vec::with_capacity(retained + 1);
    children.push(selector);
    children.extend(cases[..retained].iter().map(|case| egraph.find(*case)));
    egraph.add(MxxLang::Switch(children.into_boxed_slice()))
}

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
pub enum FamilyCoverageStorage<T = Id> {
    /// Physical element references present in the Graph IR or manifest.
    ExactStored { elements: Box<[T]> },
    /// One symbolic representative over every binder in `binder_domains`.
    SharedTemplate {
        domain: LoopDomainKey,
        representative: T,
        binder_domains: Box<[CoverageBinderDomain]>,
    },
}

/// A family residual together with its single closed element sort.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FamilyLoweringValue<T = Id> {
    pub element_type: MxxSort,
    pub storage: FamilyCoverageStorage<T>,
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

impl<T> FamilyLoweringValue<T> {
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

    pub fn exact_elements(&self) -> Option<&[T]> {
        match &self.storage {
            FamilyCoverageStorage::ExactStored { elements } => Some(elements),
            FamilyCoverageStorage::SharedTemplate { .. } => None,
        }
    }

    pub fn shared_template(&self) -> Option<(&LoopDomainKey, &T, &[CoverageBinderDomain])> {
        match &self.storage {
            FamilyCoverageStorage::ExactStored { .. } => None,
            FamilyCoverageStorage::SharedTemplate { domain, representative, binder_domains } => {
                Some((domain, representative, binder_domains))
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
pub fn static_get<T>(
    family: &FamilyLoweringValue<T>,
    index: &LoweredInt,
) -> Result<Option<T>, FamilyCoverageError>
where
    T: Copy,
{
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
pub fn dynamic_get(
    egraph: &mut EGraph<MxxLang, MxxAnalysis>,
    family: &FamilyLoweringValue<Id>,
    selector: Id,
) -> Result<Id, FamilyCoverageError> {
    let FamilyCoverageStorage::ExactStored { elements } = &family.storage else {
        return Err(FamilyCoverageError::StorageMismatch);
    };
    if elements.is_empty() {
        return Err(FamilyCoverageError::EmptyExactStorage);
    }
    Ok(add_runtime_switch(egraph, selector, elements))
}

/// Resolves an element without enumerating a shared template.  The lowerer is
/// responsible for binding the same symbolic index into the representative.
pub fn shared_element<T>(
    family: &FamilyLoweringValue<T>,
) -> Result<(&T, &LoopDomainKey, &[CoverageBinderDomain]), FamilyCoverageError> {
    let Some((domain, representative, binders)) = family.shared_template() else {
        return Err(FamilyCoverageError::StorageMismatch);
    };
    Ok((representative, domain, binders))
}

/// Instantiates one shared representative by replacing only its owning binder.
/// Other binder nodes are retained, so nested independent domains stay symbolic.
pub fn instantiate_shared_element<E>(
    egraph: &mut EGraph<MxxLang, MxxAnalysis>,
    representative: Id,
    binder: BinderId,
    replacement: Id,
    progress: &mut dyn FnMut() -> Result<(), E>,
) -> Result<Id, E> {
    // Snapshot exactly one raw representative per reachable e-class before
    // adding anything. Sampler and trapdoor records hold extra e-class edges,
    // so their complete descriptor closure belongs to the same snapshot.
    let mut nodes = std::collections::HashMap::<Id, MxxLang>::new();
    let mut source_descriptors =
        std::collections::HashMap::<AtomicSourceId, AtomicSourceDescriptor>::new();
    let mut samplers = std::collections::HashMap::<SamplerDescriptorId, SamplerIdentity>::new();
    let mut trapdoors = std::collections::HashMap::<TrapdoorDescriptorId, TrapdoorIdentity>::new();
    let mut recurrences =
        std::collections::HashMap::<SequentialRecurrenceId, SequentialRecurrenceDescriptor>::new();
    let mut pending = vec![representative];
    while let Some(id) = pending.pop() {
        if nodes.contains_key(&id) {
            continue;
        }
        progress()?;
        let node = egraph.id_to_node(id).clone();
        pending.extend(node.children().iter().copied());
        if let MxxLang::Atom { source, .. } = &node {
            let descriptor = egraph
                .analysis
                .symbols
                .atomic_sources
                .get(source.0)
                .expect("every Atom source is interned")
                .clone();
            match &descriptor.key {
                AtomicSourceKey::Sampler(sampler_id) => {
                    let sampler = egraph
                        .analysis
                        .symbols
                        .samplers
                        .get(sampler_id.0)
                        .expect("every sampler Atom has an interned descriptor")
                        .clone();
                    match &sampler {
                        SamplerIdentity::Gaussian { indices, .. } |
                        SamplerIdentity::UniformInterval { indices, .. } => {
                            pending.extend(indices.iter().copied());
                        }
                        SamplerIdentity::Preimage {
                            indices,
                            public,
                            trapdoor: trapdoor_id,
                            target,
                            ..
                        } => {
                            pending.extend(indices.iter().copied());
                            pending.extend([*public, *target]);
                            let trapdoor = egraph
                                .analysis
                                .symbols
                                .trapdoors
                                .get(trapdoor_id.0)
                                .expect("every preimage sampler trapdoor is interned")
                                .clone();
                            pending.extend(trapdoor.indices.iter().copied());
                            pending.push(trapdoor.public);
                            trapdoors.insert(*trapdoor_id, trapdoor);
                        }
                        SamplerIdentity::DecomposedHash {
                            indices,
                            public,
                            target,
                            arguments,
                            ..
                        } => {
                            pending.extend(indices.iter().copied());
                            pending.extend([*public, *target]);
                            pending.extend(arguments.iter().copied());
                        }
                        SamplerIdentity::GadgetDecomposition {
                            indices, public, target, ..
                        } => {
                            pending.extend(indices.iter().copied());
                            pending.extend([*public, *target]);
                        }
                    }
                    samplers.insert(*sampler_id, sampler);
                }
                AtomicSourceKey::SequentialRecurrence { recurrence: recurrence_id, .. } => {
                    let recurrence = egraph
                        .analysis
                        .symbols
                        .sequential_recurrences
                        .get(recurrence_id.0)
                        .expect("every sequential recurrence Atom has an interned descriptor")
                        .clone();
                    pending.extend(recurrence.initial.iter().copied());
                    pending.extend(recurrence.transition.iter().copied());
                    recurrences.insert(*recurrence_id, recurrence);
                }
                _ => {}
            }
            source_descriptors.insert(*source, descriptor);
        }
        nodes.insert(id, node);
    }

    enum Visit {
        Enter(Id),
        Exit(Id),
    }
    // Replacement is already an e-class in this e-graph.  Treat it as an
    // opaque leaf: copying it would both duplicate work and accidentally
    // substitute owner binders that are intentionally inside its value.
    let mut completed = std::collections::HashMap::<Id, Id>::from([(replacement, replacement)]);
    let mut work = vec![Visit::Enter(representative)];
    while let Some(visit) = work.pop() {
        let id = match visit {
            Visit::Enter(id) => {
                if completed.contains_key(&id) {
                    continue;
                }
                if matches!(nodes[&id], MxxLang::IntBinder(candidate) if candidate == binder) {
                    completed.insert(id, replacement);
                    continue;
                }
                work.push(Visit::Exit(id));
                let node = &nodes[&id];
                for child in node.children().iter().rev() {
                    work.push(Visit::Enter(*child));
                }
                if let MxxLang::Atom { source, .. } = node &&
                    let Some(AtomicSourceDescriptor {
                        key: AtomicSourceKey::Sampler(sampler),
                        ..
                    }) = source_descriptors.get(source)
                {
                    let sampler = &samplers[sampler];
                    let mut references = Vec::new();
                    match sampler {
                        SamplerIdentity::Gaussian { indices, .. } |
                        SamplerIdentity::UniformInterval { indices, .. } => {
                            references.extend(indices.iter().copied());
                        }
                        SamplerIdentity::Preimage { indices, public, trapdoor, target, .. } => {
                            references.extend(indices.iter().copied());
                            references.extend([*public, *target]);
                            let trapdoor = &trapdoors[trapdoor];
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
                    for term in references.into_iter().rev() {
                        work.push(Visit::Enter(term));
                    }
                } else if let MxxLang::Atom { source, .. } = node &&
                    let Some(AtomicSourceDescriptor {
                        key: AtomicSourceKey::SequentialRecurrence { recurrence, .. },
                        ..
                    }) = source_descriptors.get(source)
                {
                    let recurrence = &recurrences[recurrence];
                    for term in recurrence.initial.iter().chain(recurrence.transition.iter()).rev()
                    {
                        work.push(Visit::Enter(*term));
                    }
                }
                continue;
            }
            Visit::Exit(id) => id,
        };
        if completed.contains_key(&id) {
            continue;
        }
        progress()?;
        let node = &nodes[&id];
        let remap = |term| completed[&term];
        let rebuilt = if let MxxLang::Atom { source, .. } = node &&
            let Some(descriptor) = source_descriptors.get(source) &&
            let AtomicSourceKey::Sampler(sampler_id) = descriptor.key
        {
            let sampler = match samplers[&sampler_id].clone() {
                SamplerIdentity::Gaussian { source, indices, max_coefficient_bound } => {
                    SamplerIdentity::Gaussian {
                        source,
                        indices: indices.iter().map(|term| remap(*term)).collect(),
                        max_coefficient_bound,
                    }
                }
                SamplerIdentity::UniformInterval { source, indices, minimum, maximum } => {
                    SamplerIdentity::UniformInterval {
                        source,
                        indices: indices.iter().map(|term| remap(*term)).collect(),
                        minimum,
                        maximum,
                    }
                }
                SamplerIdentity::Preimage { source, indices, public, trapdoor, target, cutoff } => {
                    let trapdoor = trapdoors[&trapdoor].clone();
                    let trapdoor = TrapdoorDescriptorId(egraph.analysis.symbols.trapdoors.intern(
                        TrapdoorIdentity {
                            indices: trapdoor.indices.iter().map(|term| remap(*term)).collect(),
                            public: remap(trapdoor.public),
                            ..trapdoor
                        },
                    ));
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
                SamplerIdentity::Gaussian { indices, .. } |
                SamplerIdentity::UniformInterval { indices, .. } |
                SamplerIdentity::Preimage { indices, .. } |
                SamplerIdentity::DecomposedHash { indices, .. } |
                SamplerIdentity::GadgetDecomposition { indices, .. } => indices.clone(),
            };
            let sampler = egraph.analysis.symbols.samplers.intern(sampler);
            let source = egraph.analysis.symbols.atomic_sources.intern(AtomicSourceDescriptor {
                key: AtomicSourceKey::Sampler(SamplerDescriptorId(sampler)),
                ..descriptor.clone()
            });
            egraph.add(MxxLang::Atom { source: AtomicSourceId(source), indices })
        } else if let MxxLang::Atom { source, indices } = node &&
            let Some(descriptor) = source_descriptors.get(source) &&
            let AtomicSourceKey::SequentialRecurrence { recurrence, carried_index } =
                descriptor.key
        {
            let recurrence = recurrences[&recurrence].clone();
            let recurrence =
                SequentialRecurrenceId(egraph.analysis.symbols.sequential_recurrences.intern(
                    SequentialRecurrenceDescriptor {
                        initial: recurrence.initial.iter().map(|term| remap(*term)).collect(),
                        transition: recurrence.transition.iter().map(|term| remap(*term)).collect(),
                        ..recurrence
                    },
                ));
            let source = egraph.analysis.symbols.atomic_sources.intern(AtomicSourceDescriptor {
                key: AtomicSourceKey::SequentialRecurrence { recurrence, carried_index },
                ..descriptor.clone()
            });
            egraph.add(MxxLang::Atom {
                source: AtomicSourceId(source),
                indices: indices.iter().map(|term| remap(*term)).collect(),
            })
        } else if let MxxLang::Switch(children) = node {
            let remapped = children.iter().map(|term| remap(*term)).collect::<Vec<_>>();
            let (selector, cases) =
                remapped.split_first().expect("stored Switch nodes have a selector");
            // A substituted exact selector has the ordinary runtime Switch
            // meaning: select its zero-based case instead of retaining every
            // unreachable branch in the instantiated representative.
            match egraph[egraph.find(*selector)]
                .data
                .integer_domain
                .as_ref()
                .and_then(|domain| match domain {
                    IntegerDomain::Exact(value) => Some(value),
                    IntegerDomain::Affine { .. } | IntegerDomain::IntervalOnly(_) => None,
                })
                .and_then(|value| value.to_usize())
                .and_then(|index| cases.get(index))
            {
                Some(selected) => *selected,
                // An out-of-range exact selector cannot be selected safely.
                // Keep the invalid Switch structural so the owning generic
                // validation path rejects it rather than guessing a branch.
                None => add_runtime_switch(egraph, *selector, cases),
            }
        } else {
            egraph.add(node.clone().map_children(remap))
        };
        completed.insert(id, rebuilt);
    }
    Ok(completed[&representative])
}

/// Selects a compact family.  Exact storage evaluates only stored references;
/// template storage combines representatives pointwise under one selector.
pub fn select_family(
    egraph: &mut EGraph<MxxLang, MxxAnalysis>,
    selector: Id,
    cases: &[FamilyLoweringValue],
) -> Result<FamilyLoweringValue, FamilyCoverageError> {
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
                let mut selected_cases = Vec::with_capacity(cases.len());
                for case in cases {
                    let FamilyCoverageStorage::ExactStored { elements } = &case.storage else {
                        unreachable!("storage shape was checked before e-graph mutation");
                    };
                    selected_cases.push(elements[lane]);
                }
                selected.push(add_runtime_switch(egraph, selector, &selected_cases));
            }
            Ok(FamilyLoweringValue {
                element_type: first.element_type.clone(),
                storage: FamilyCoverageStorage::ExactStored {
                    elements: selected.into_boxed_slice(),
                },
            })
        }
        FamilyCoverageStorage::SharedTemplate { domain, binder_domains, .. } => {
            let mut selected_cases = Vec::with_capacity(cases.len());
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
                selected_cases.push(*representative);
            }
            Ok(FamilyLoweringValue {
                element_type: first.element_type.clone(),
                storage: FamilyCoverageStorage::SharedTemplate {
                    domain: domain.clone(),
                    representative: add_runtime_switch(egraph, selector, &selected_cases),
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
            OccurrenceFrame, OccurrenceScope, ProgramKey, ResolvedMatrixType, TrapdoorSourceKey,
            WireSourceKey,
        },
    };

    fn evaluate(recurrence: &VectorRecurrence) -> Result<Box<[BigUint]>, RecurrenceFailure> {
        recurrence.evaluate()
    }

    fn test_binder(analysis: &mut MxxAnalysis, node: u32) -> BinderId {
        let scope = OccurrenceScope {
            program: ProgramKey::Ideal,
            definition: mxx_ir_core::FrozenGraphScopeId::Root,
            path: Box::new([]),
        };
        BinderId(analysis.symbols.binders.intern(BinderDescriptor {
            key: BinderKey {
                loop_scope: scope,
                loop_node: mxx_ir_core::NodeId(node.into()),
                slot: 0,
            },
            minimum: 0.into(),
            maximum: 7.into(),
        }))
    }

    fn test_integer_selector(
        egraph: &mut EGraph<MxxLang, MxxAnalysis>,
        name: &str,
        minimum: BigInt,
        maximum: BigInt,
    ) -> Id {
        let source = egraph.analysis.symbols.atomic_sources.intern(AtomicSourceDescriptor {
            key: AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from(name)),
            sort: MxxSort::Int,
            integer_domain: Some(super::super::identity::IntegerSourceDomain { minimum, maximum }),
            canonical_residue_convention: None,
            relation_role: None,
        });
        egraph.add(MxxLang::Atom { source: AtomicSourceId(source), indices: Box::new([]) })
    }

    #[test]
    fn runtime_switch_trims_only_an_authoritative_unreachable_suffix() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = test_integer_selector(&mut egraph, "range-01", 0.into(), 1.into());
        let cases = [10, 20, 30, 40].map(|value| egraph.add(MxxLang::IntConst(value.into())));
        let three = add_runtime_switch(&mut egraph, selector, &cases[..3]);
        let four = add_runtime_switch(&mut egraph, selector, &cases);
        assert_eq!(egraph.find(three), egraph.find(four));
        assert!(matches!(egraph.id_to_node(three), MxxLang::Switch(children)
            if children.as_ref() == [selector, cases[0], cases[1]]));
        assert_eq!(
            egraph[egraph.find(three)].data.integer_domain.as_ref().unwrap().interval().unwrap(),
            super::super::analysis::IntegerInterval::new(10.into(), 20.into()).unwrap(),
            "analysis sees exactly the retained reachable cases"
        );

        let shifted = test_integer_selector(&mut egraph, "range-12", 1.into(), 2.into());
        let shifted_switch = add_runtime_switch(&mut egraph, shifted, &cases);
        assert!(matches!(egraph.id_to_node(shifted_switch), MxxLang::Switch(children)
            if children.as_ref() == [shifted, cases[0], cases[1], cases[2]]));
    }

    #[test]
    fn runtime_switch_preserves_cases_without_a_valid_strict_upper_bound() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let cases = [10, 20].map(|value| egraph.add(MxxLang::IntConst(value.into())));
        let selectors = [
            test_integer_selector(&mut egraph, "negative", (-1).into(), 1.into()),
            test_integer_selector(&mut egraph, "at-count", 0.into(), 2.into()),
            test_integer_selector(
                &mut egraph,
                "unconvertible",
                0.into(),
                BigInt::from(BigUint::one() << (usize::BITS + 1)),
            ),
        ];
        for selector in selectors {
            let switched = add_runtime_switch(&mut egraph, selector, &cases);
            assert!(matches!(egraph.id_to_node(switched), MxxLang::Switch(children)
                if children.len() == cases.len() + 1));
        }

        let missing = egraph.analysis.symbols.atomic_sources.intern(AtomicSourceDescriptor {
            key: AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from("missing-domain")),
            sort: MxxSort::Int,
            integer_domain: None,
            canonical_residue_convention: None,
            relation_role: None,
        });
        let missing =
            egraph.add(MxxLang::Atom { source: AtomicSourceId(missing), indices: Box::new([]) });
        let switched = add_runtime_switch(&mut egraph, missing, &cases);
        assert!(matches!(egraph.id_to_node(switched), MxxLang::Switch(children)
            if children.len() == cases.len() + 1));
    }

    #[test]
    fn runtime_switch_is_shared_by_dynamic_get_and_select_family_validation() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = test_integer_selector(&mut egraph, "shared-runtime", 0.into(), 1.into());
        let elements = [10, 20, 30].map(|value| egraph.add(MxxLang::IntConst(value.into())));
        let family = FamilyLoweringValue {
            element_type: MxxSort::Int,
            storage: FamilyCoverageStorage::ExactStored { elements: Box::new(elements) },
        };
        let dynamic = dynamic_get(&mut egraph, &family, selector).unwrap();
        let direct = add_runtime_switch(&mut egraph, selector, &elements);
        assert_eq!(egraph.find(dynamic), egraph.find(direct));

        let invalid = FamilyLoweringValue {
            element_type: MxxSort::Int,
            storage: FamilyCoverageStorage::ExactStored { elements: Box::new([elements[0]]) },
        };
        assert_eq!(
            select_family(&mut egraph, selector, &[family, invalid]),
            Err(FamilyCoverageError::SelectorCaseCountMismatch { expected: 3, actual: 1 }),
            "all original family storage is validated before any suffix can be trimmed"
        );
    }

    #[test]
    fn shared_template_instantiation_selects_exact_switch_cases() {
        for (selector_value, expected) in [(0, 10), (1, 20), (2, 30)] {
            let mut analysis = MxxAnalysis::default();
            let binder = test_binder(&mut analysis, 1);
            let mut egraph = EGraph::new(analysis);
            let selector = egraph.add(MxxLang::IntBinder(binder));
            let cases = [10, 20, 30].map(|value| egraph.add(MxxLang::IntConst(value.into())));
            let representative =
                egraph.add(MxxLang::Switch([selector, cases[0], cases[1], cases[2]].into()));
            let replacement = egraph.add(MxxLang::IntConst(selector_value.into()));
            let instantiated = instantiate_shared_element(
                &mut egraph,
                representative,
                binder,
                replacement,
                &mut || Ok::<(), ()>(()),
            )
            .unwrap();
            let expected = egraph.add(MxxLang::IntConst(expected.into()));
            assert_eq!(egraph.find(instantiated), egraph.find(expected));
        }
    }

    #[test]
    fn shared_template_instantiation_keeps_nonexact_or_invalid_switches_structural() {
        let mut analysis = MxxAnalysis::default();
        let owner = test_binder(&mut analysis, 1);
        let foreign = test_binder(&mut analysis, 2);
        let mut egraph = EGraph::new(analysis);
        let owner_selector = egraph.add(MxxLang::IntBinder(owner));
        let foreign_selector = egraph.add(MxxLang::IntBinder(foreign));
        let first = egraph.add(MxxLang::IntConst(10.into()));
        let second = egraph.add(MxxLang::IntConst(20.into()));
        let nonexact = egraph.add(MxxLang::Switch([foreign_selector, first, second].into()));
        let replacement = egraph.add(MxxLang::IntConst(1.into()));
        let instantiated =
            instantiate_shared_element(&mut egraph, nonexact, owner, replacement, &mut || {
                Ok::<(), ()>(())
            })
            .unwrap();
        assert!(matches!(egraph.id_to_node(instantiated), MxxLang::Switch(_)));

        let bounded_selector =
            test_integer_selector(&mut egraph, "instantiated-range", 0.into(), 1.into());
        let bounded =
            egraph.add(MxxLang::Switch([bounded_selector, first, second, replacement].into()));
        let instantiated =
            instantiate_shared_element(&mut egraph, bounded, owner, replacement, &mut || {
                Ok::<(), ()>(())
            })
            .unwrap();
        let MxxLang::Switch(children) = egraph.id_to_node(instantiated) else {
            panic!("non-exact interval remains a Switch")
        };
        assert_eq!(children.len(), 3);
        assert_eq!(egraph.find(children[0]), egraph.find(bounded_selector));
        assert_eq!(egraph.find(children[1]), egraph.find(first));
        assert_eq!(egraph.find(children[2]), egraph.find(second));

        let invalid = egraph.add(MxxLang::Switch([owner_selector, first, second].into()));
        let out_of_range = egraph.add(MxxLang::IntConst(2.into()));
        let instantiated =
            instantiate_shared_element(&mut egraph, invalid, owner, out_of_range, &mut || {
                Ok::<(), ()>(())
            })
            .unwrap();
        assert!(matches!(egraph.id_to_node(instantiated), MxxLang::Switch(_)));
    }

    #[test]
    fn shared_template_instantiation_selects_nested_switches_without_replacing_foreign_binders() {
        let mut analysis = MxxAnalysis::default();
        let owner = test_binder(&mut analysis, 1);
        let foreign = test_binder(&mut analysis, 2);
        let mut egraph = EGraph::new(analysis);
        let owner_selector = egraph.add(MxxLang::IntBinder(owner));
        let foreign_selector = egraph.add(MxxLang::IntBinder(foreign));
        let first = egraph.add(MxxLang::IntConst(10.into()));
        let second = egraph.add(MxxLang::IntConst(20.into()));
        let inner = egraph.add(MxxLang::Switch([foreign_selector, first, second].into()));
        let representative = egraph.add(MxxLang::Switch([owner_selector, first, inner].into()));
        let replacement = egraph.add(MxxLang::IntConst(1.into()));
        let instantiated = instantiate_shared_element(
            &mut egraph,
            representative,
            owner,
            replacement,
            &mut || Ok::<(), ()>(()),
        )
        .unwrap();
        let expression = egraph.id_to_expr(instantiated);
        assert!(matches!(expression[expression.root()], MxxLang::Switch(_)));
        assert!(expression.iter().any(|node| node == &MxxLang::IntBinder(foreign)));
        assert!(!expression.iter().any(|node| node == &MxxLang::IntBinder(owner)));
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

        let mut progress_calls = 0;
        let instantiated = instantiate_shared_element(
            &mut egraph,
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
            unrelated_progress_calls <= progress_calls,
            "unrelated e-classes do not add instantiation work"
        );

        let mut rejected = || Err::<(), _>("progress stopped");
        assert_eq!(
            instantiate_shared_element(
                &mut egraph,
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

        let instantiated = instantiate_shared_element(
            &mut egraph,
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
        let instantiated = instantiate_shared_element(
            &mut egraph,
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
    fn nested_parallel_sequential_descriptor_uses_the_selected_outer_index() {
        let mut analysis = MxxAnalysis::default();
        let root_scope = OccurrenceScope {
            program: ProgramKey::Ideal,
            definition: mxx_ir_core::FrozenGraphScopeId::Root,
            path: Box::new([]),
        };
        let outer = BinderKey {
            loop_scope: root_scope.clone(),
            loop_node: mxx_ir_core::NodeId(10),
            slot: 0,
        };
        let outer_id = BinderId(analysis.symbols.binders.intern(BinderDescriptor {
            key: outer,
            minimum: 0.into(),
            maximum: 7.into(),
        }));
        let matrix_type = ResolvedMatrixType {
            modulus: ResolvedIntExpr::Const(17.into()),
            ring_dimension: ResolvedIntExpr::Const(1.into()),
            rows: ResolvedIntExpr::Const(1.into()),
            columns: ResolvedIntExpr::Const(1.into()),
        };
        let mut egraph = EGraph::new(analysis);
        let outer_term = egraph.add(MxxLang::IntBinder(outer_id));
        let replacement = egraph.add(MxxLang::IntConst(4.into()));
        let initial = egraph.add(MxxLang::LiftConstantPolynomial {
            matrix_type: matrix_type.clone(),
            input: [outer_term],
        });
        let transition = egraph.add(MxxLang::LiftConstantPolynomial {
            matrix_type: matrix_type.clone(),
            input: [outer_term],
        });
        let sequential_scope = OccurrenceScope {
            program: ProgramKey::Ideal,
            definition: mxx_ir_core::FrozenGraphScopeId::Root,
            path: Box::new([OccurrenceFrame::ParallelLoop {
                parent: mxx_ir_core::FrozenGraphScopeId::Root,
                owner: mxx_ir_core::NodeId(10),
            }]),
        };
        let recurrence = SequentialRecurrenceId(
            egraph.analysis.symbols.sequential_recurrences.intern(SequentialRecurrenceDescriptor {
                loop_scope: sequential_scope,
                loop_node: mxx_ir_core::NodeId(11),
                count: ResolvedIntExpr::Const(3.into()),
                initial: vec![initial].into_boxed_slice(),
                transition: vec![transition].into_boxed_slice(),
                output_types: vec![matrix_type.clone()].into_boxed_slice(),
            }),
        );
        let source = egraph.analysis.symbols.atomic_sources.intern(AtomicSourceDescriptor {
            key: AtomicSourceKey::SequentialRecurrence { recurrence, carried_index: 0 },
            sort: MxxSort::Matrix(matrix_type),
            integer_domain: None,
            canonical_residue_convention: None,
            relation_role: None,
        });
        let representative =
            egraph.add(MxxLang::Atom { source: AtomicSourceId(source), indices: Box::new([]) });

        let instantiated = instantiate_shared_element(
            &mut egraph,
            representative,
            outer_id,
            replacement,
            &mut || Ok::<(), ()>(()),
        )
        .unwrap();
        let MxxLang::Atom { source, .. } = &egraph[instantiated].nodes[0] else {
            panic!("instantiated sequential recurrence remains an Atom");
        };
        let AtomicSourceKey::SequentialRecurrence { recurrence, carried_index } = egraph
            .analysis
            .symbols
            .atomic_sources
            .get(source.0)
            .expect("instantiated source is interned")
            .key
        else {
            panic!("sequential recurrence source");
        };
        assert_eq!(carried_index, 0);
        let descriptor = egraph
            .analysis
            .symbols
            .sequential_recurrences
            .get(recurrence.0)
            .expect("instantiated recurrence is interned");
        for term in descriptor.initial.iter().chain(descriptor.transition.iter()) {
            let MxxLang::LiftConstantPolynomial { input, .. } = egraph.id_to_node(*term) else {
                panic!("recurrence term remains a lifted constant polynomial");
            };
            assert_eq!(egraph.find(input[0]), egraph.find(replacement));
        }
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
