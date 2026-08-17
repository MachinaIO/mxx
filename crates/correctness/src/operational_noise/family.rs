//! Compact family coverage and numeric sequential-loop recurrence support.
//!
//! This module deliberately owns neither graph-wire memoization nor integer
//! analysis.  [`GraphLowerer`](super::lower::GraphLowerer) supplies one
//! symbolic element at a time.  Numeric recurrence evaluation is exact over
//! the supplied finite count; it has no policy ceiling. Runtime Switches may
//! omit only a suffix proved unreachable by the selector's authoritative interval.

use super::{
    identity::{
        AtomicSourceDescriptor, AtomicSourceId, AtomicSourceKey, BinderKey, CanonicalTermIdentity,
        ResolvedIntExpr, SamplerDescriptorId, SamplerIdentity, SymbolTables, TrapdoorDescriptorId,
        TrapdoorIdentity,
    },
    lower::LoweredInt,
    normal_form::FactorIdentity,
    scalar::{IntegerDomain, ScalarId, ScalarNode, ScalarSort, ScalarStore},
};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};
use std::collections::{BTreeMap, HashMap};

/// Adds one runtime Switch, retaining every case through the authoritative
/// maximum selector value. Invalid or unavailable selector facts preserve the
/// complete physical case list for the owning validation path.
pub(crate) fn add_runtime_switch(
    store: &mut ScalarStore,
    symbols: &SymbolTables,
    selector: ScalarId,
    cases: &[ScalarId],
) -> Result<ScalarId, FamilyCoverageError> {
    let retained = store
        .facts(selector)
        .and_then(|facts| facts.integer_domain.as_ref())
        .and_then(|domain| domain.interval().ok())
        .filter(|interval| interval.minimum >= BigInt::zero())
        .and_then(|interval| interval.maximum.to_usize())
        .and_then(|maximum| (maximum < cases.len()).then_some(maximum + 1))
        .unwrap_or(cases.len());
    let mut children = Vec::with_capacity(retained + 1);
    children.push(selector);
    children.extend_from_slice(&cases[..retained]);
    store
        .intern_node(
            ScalarNode::Switch(children.into_boxed_slice()),
            ResolvedIntExpr::Const(BigInt::zero()),
            symbols,
        )
        .map_err(|_| FamilyCoverageError::ScalarConstructionFailed)
}

/// Performs binder substitution over a resolved identity without recursive descent.
fn substitute_resolved_iterative(
    value: &ResolvedIntExpr,
    binder: &BinderKey,
    replacement: &ResolvedIntExpr,
) -> ResolvedIntExpr {
    enum Visit<'a> {
        Enter(&'a ResolvedIntExpr),
        Exit(&'a ResolvedIntExpr),
    }
    let mut completed = HashMap::<usize, ResolvedIntExpr>::new();
    let mut work = vec![Visit::Enter(value)];
    while let Some(visit) = work.pop() {
        let (key, expression) = match visit {
            Visit::Enter(expression) => {
                let key = expression as *const ResolvedIntExpr as usize;
                if completed.contains_key(&key) {
                    continue;
                }
                if let ResolvedIntExpr::Binder(candidate) = expression {
                    completed.insert(
                        key,
                        if candidate == binder { replacement.clone() } else { expression.clone() },
                    );
                    continue;
                }
                work.push(Visit::Exit(expression));
                match expression {
                    ResolvedIntExpr::Source { coordinates, .. } => {
                        work.extend(coordinates.iter().rev().map(Visit::Enter))
                    }
                    ResolvedIntExpr::Add(left, right) |
                    ResolvedIntExpr::Sub(left, right) |
                    ResolvedIntExpr::Mul(left, right) |
                    ResolvedIntExpr::Div(left, right) |
                    ResolvedIntExpr::EuclideanDiv(left, right) |
                    ResolvedIntExpr::EuclideanRemainder(left, right) |
                    ResolvedIntExpr::RoundDiv(left, right) => {
                        work.push(Visit::Enter(right));
                        work.push(Visit::Enter(left));
                    }
                    ResolvedIntExpr::Log2Ceil(input) => work.push(Visit::Enter(input)),
                    ResolvedIntExpr::ExtractCoefficient { input, position, .. } => {
                        work.push(Visit::Enter(position));
                        work.push(Visit::Enter(input));
                    }
                    ResolvedIntExpr::Const(_) | ResolvedIntExpr::Parameter(_) => {}
                    ResolvedIntExpr::Binder(_) => unreachable!(),
                }
                continue;
            }
            Visit::Exit(expression) => (expression as *const ResolvedIntExpr as usize, expression),
        };
        let child = |child: &ResolvedIntExpr| {
            completed
                .get(&(child as *const ResolvedIntExpr as usize))
                .cloned()
                .expect("identity child")
        };
        let rebuilt = match expression {
            ResolvedIntExpr::Source { source, coordinates } => ResolvedIntExpr::Source {
                source: source.clone(),
                coordinates: coordinates.iter().map(child).collect(),
            },
            ResolvedIntExpr::Add(left, right) => {
                ResolvedIntExpr::Add(Box::new(child(left)), Box::new(child(right)))
            }
            ResolvedIntExpr::Sub(left, right) => {
                ResolvedIntExpr::Sub(Box::new(child(left)), Box::new(child(right)))
            }
            ResolvedIntExpr::Mul(left, right) => {
                ResolvedIntExpr::Mul(Box::new(child(left)), Box::new(child(right)))
            }
            ResolvedIntExpr::Div(left, right) => {
                ResolvedIntExpr::Div(Box::new(child(left)), Box::new(child(right)))
            }
            ResolvedIntExpr::EuclideanDiv(left, right) => {
                ResolvedIntExpr::EuclideanDiv(Box::new(child(left)), Box::new(child(right)))
            }
            ResolvedIntExpr::EuclideanRemainder(left, right) => {
                ResolvedIntExpr::EuclideanRemainder(Box::new(child(left)), Box::new(child(right)))
            }
            ResolvedIntExpr::RoundDiv(left, right) => {
                ResolvedIntExpr::RoundDiv(Box::new(child(left)), Box::new(child(right)))
            }
            ResolvedIntExpr::Log2Ceil(input) => ResolvedIntExpr::Log2Ceil(Box::new(child(input))),
            ResolvedIntExpr::ExtractCoefficient { input, position, canonical_exclusive_upper } => {
                ResolvedIntExpr::ExtractCoefficient {
                    input: Box::new(child(input)),
                    position: Box::new(child(position)),
                    canonical_exclusive_upper: canonical_exclusive_upper.clone(),
                }
            }
            ResolvedIntExpr::Const(_) |
            ResolvedIntExpr::Parameter(_) |
            ResolvedIntExpr::Binder(_) => expression.clone(),
        };
        completed.insert(key, rebuilt);
    }
    completed.remove(&(value as *const ResolvedIntExpr as usize)).expect("identity root")
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
pub enum FamilyCoverageStorage<T = ScalarId> {
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
pub struct FamilyLoweringValue<T = ScalarId> {
    pub element_type: ScalarSort,
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
    ElementTypeMismatch { expected: ScalarSort, actual: ScalarSort },
    StorageMismatch,
    ScalarConstructionFailed,
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

/// Builds one ordered physical `Switch`. Its work is linear in stored cases.
pub fn dynamic_get(
    store: &mut ScalarStore,
    symbols: &SymbolTables,
    family: &FamilyLoweringValue<ScalarId>,
    selector: ScalarId,
) -> Result<ScalarId, FamilyCoverageError> {
    let FamilyCoverageStorage::ExactStored { elements } = &family.storage else {
        return Err(FamilyCoverageError::StorageMismatch);
    };
    if elements.is_empty() {
        return Err(FamilyCoverageError::EmptyExactStorage);
    }
    let count = BigUint::from(elements.len());
    let index =
        store.facts(selector).and_then(|facts| facts.integer_domain.as_ref()).ok_or_else(|| {
            FamilyCoverageError::DynamicIndexOutOfRange {
                minimum: BigInt::from(-1),
                maximum: BigInt::from(-1),
                count: count.clone(),
            }
        })?;
    validate_family_index(index, &count)?;
    add_runtime_switch(store, symbols, selector, elements)
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

fn substitute_indices(
    indices: &[ResolvedIntExpr],
    binder: &BinderKey,
    replacement: Option<&ResolvedIntExpr>,
) -> Box<[ResolvedIntExpr]> {
    let Some(replacement) = replacement else {
        return indices.to_vec().into_boxed_slice();
    };
    indices.iter().map(|index| substitute_resolved_iterative(index, binder, replacement)).collect()
}

fn substitute_canonical_identity(
    identity: &CanonicalTermIdentity,
    binder: &BinderKey,
    replacement: Option<&ResolvedIntExpr>,
) -> CanonicalTermIdentity {
    let Some(replacement) = replacement else {
        return identity.clone();
    };
    match identity {
        CanonicalTermIdentity::Factor(factor) => CanonicalTermIdentity::Factor(FactorIdentity {
            coordinates: factor
                .coordinates
                .iter()
                .map(|(owner, value)| {
                    (owner.clone(), substitute_resolved_iterative(value, binder, replacement))
                })
                .collect(),
            ..factor.clone()
        }),
        // A graph source identifies the originating occurrence.  Its
        // coordinate binders are provenance, while the varying coordinate
        // values are carried by `indices` and are substituted separately.
        CanonicalTermIdentity::Source(_) => identity.clone(),
    }
}

fn substitute_sampler_identity(
    sampler: &SamplerIdentity,
    binder: &BinderKey,
    replacement: Option<&ResolvedIntExpr>,
) -> SamplerIdentity {
    let map = |value: &ResolvedIntExpr| {
        replacement.map_or_else(
            || value.clone(),
            |replacement| substitute_resolved_iterative(value, &binder, replacement),
        )
    };
    match sampler {
        SamplerIdentity::Gaussian { source, indices, max_coefficient_bound } => {
            SamplerIdentity::Gaussian {
                source: source.clone(),
                indices: substitute_indices(indices, &binder, replacement),
                max_coefficient_bound: map(max_coefficient_bound),
            }
        }
        SamplerIdentity::UniformInterval { source, indices, minimum, maximum } => {
            SamplerIdentity::UniformInterval {
                source: source.clone(),
                indices: substitute_indices(indices, &binder, replacement),
                minimum: map(minimum),
                maximum: map(maximum),
            }
        }
        SamplerIdentity::Preimage { source, indices, public, trapdoor, target, cutoff } => {
            SamplerIdentity::Preimage {
                source: source.clone(),
                indices: substitute_indices(indices, &binder, replacement),
                public: substitute_canonical_identity(public, &binder, replacement),
                trapdoor: *trapdoor,
                target: substitute_canonical_identity(target, &binder, replacement),
                cutoff: map(cutoff),
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
            source: source.clone(),
            indices: substitute_indices(indices, &binder, replacement),
            public: substitute_canonical_identity(public, &binder, replacement),
            target: substitute_canonical_identity(target, &binder, replacement),
            arguments: arguments
                .iter()
                .map(|argument| substitute_canonical_identity(argument, &binder, replacement))
                .collect(),
            matrix_type: matrix_type.clone(),
            base: map(base),
            digit_count: map(digit_count),
            small: *small,
            range_proved: *range_proved,
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
            source: source.clone(),
            indices: substitute_indices(indices, &binder, replacement),
            public: substitute_canonical_identity(public, &binder, replacement),
            target: substitute_canonical_identity(target, &binder, replacement),
            base: map(base),
            digit_count: map(digit_count),
            small: *small,
            range_proved: *range_proved,
        },
    }
}

fn replace_sampler_trapdoor(
    sampler: SamplerIdentity,
    trapdoor: TrapdoorDescriptorId,
) -> SamplerIdentity {
    let SamplerIdentity::Preimage { source, indices, public, target, cutoff, .. } = sampler else {
        return sampler;
    };
    SamplerIdentity::Preimage { source, indices, public, trapdoor, target, cutoff }
}

fn substitute_trapdoor_identity(
    trapdoor: &TrapdoorIdentity,
    binder: &BinderKey,
    replacement: Option<&ResolvedIntExpr>,
) -> TrapdoorIdentity {
    let map = |value: &ResolvedIntExpr| {
        replacement.map_or_else(
            || value.clone(),
            |replacement| substitute_resolved_iterative(value, binder, replacement),
        )
    };
    TrapdoorIdentity {
        source: trapdoor.source.clone(),
        indices: substitute_indices(&trapdoor.indices, binder, replacement),
        matrix_type: trapdoor.matrix_type.clone(),
        public: substitute_canonical_identity(&trapdoor.public, binder, replacement),
        sigma_bits: trapdoor.sigma_bits,
        gadget_base: map(&trapdoor.gadget_base),
        digit_count: map(&trapdoor.digit_count),
        preimage_cutoff: map(&trapdoor.preimage_cutoff),
    }
}

/// Instantiates one shared representative by replacing only its owning binder.
/// The explicit postorder walk keeps nested independent binders symbolic and
/// never materializes a logical family lane.
pub fn instantiate_shared_element<E>(
    store: &mut ScalarStore,
    symbols: &mut SymbolTables,
    representative: ScalarId,
    binder: &BinderKey,
    replacement: ScalarId,
    stable_replacement: Option<ResolvedIntExpr>,
    progress: &mut dyn FnMut() -> Result<(), E>,
) -> Result<ScalarId, E> {
    let stable_replacement = stable_replacement.or_else(|| store.identity(replacement).cloned());
    let mut nodes = HashMap::<ScalarId, ScalarNode>::new();
    let mut identities = HashMap::<ScalarId, ResolvedIntExpr>::new();
    let mut pending = vec![representative];
    while let Some(id) = pending.pop() {
        if nodes.contains_key(&id) || id == replacement {
            continue;
        }
        progress()?;
        let node = store.node(id).cloned().ok_or_else(|| panic!("missing scalar node"));
        let node = match node {
            Ok(node) => node,
            Err(panic) => panic,
        };
        pending.extend(store.children(id).unwrap_or_default().into_vec());
        identities.insert(
            id,
            store.identity(id).cloned().unwrap_or(ResolvedIntExpr::Const(BigInt::zero())),
        );
        nodes.insert(id, node);
    }
    enum Visit {
        Enter(ScalarId),
        Exit(ScalarId),
    }
    let mut completed = HashMap::from([(replacement, replacement)]);
    let mut work = vec![Visit::Enter(representative)];
    while let Some(visit) = work.pop() {
        let id = match visit {
            Visit::Enter(id) => {
                if completed.contains_key(&id) {
                    continue;
                }
                if matches!(nodes.get(&id), Some(ScalarNode::IntBinder(candidate)) if candidate == binder)
                {
                    completed.insert(id, replacement);
                    continue;
                }
                work.push(Visit::Exit(id));
                for child in store.children(id).unwrap_or_default().iter().rev() {
                    if !completed.contains_key(child) {
                        work.push(Visit::Enter(*child));
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
        let node = nodes.get(&id).expect("snapshotted scalar node");
        let remap = |child: ScalarId| *completed.get(&child).expect("postorder child");
        if let ScalarNode::Switch(children) = node {
            let remapped = children.iter().map(|child| remap(*child)).collect::<Vec<_>>();
            if let Some(index) = store
                .facts(remapped[0])
                .and_then(|facts| facts.integer_domain.as_ref())
                .and_then(|domain| domain.exact_value())
                .and_then(|value| value.to_usize())
                .and_then(|index| remapped.get(index + 1).copied())
            {
                completed.insert(id, index);
                continue;
            }
        }
        let rebuilt = remap_scalar_node(node, &remap, binder, stable_replacement.as_ref(), symbols);
        let identity = stable_replacement.as_ref().map_or_else(
            || identities.get(&id).expect("snapshotted scalar identity").clone(),
            |replacement| {
                substitute_resolved_iterative(
                    identities.get(&id).expect("snapshotted scalar identity"),
                    binder,
                    replacement,
                )
            },
        );
        let identity = match (rebuilt.as_ref(), identity) {
            (
                Some(ScalarNode::Source { source, .. }),
                ResolvedIntExpr::Source { source: old_source, coordinates },
            ) => {
                if let Some(descriptor) = symbols.atomic_sources.get(source.0) {
                    ResolvedIntExpr::Source { source: descriptor.key.clone(), coordinates }
                } else {
                    ResolvedIntExpr::Source { source: old_source, coordinates }
                }
            }
            (_, identity) => identity,
        };
        let rebuilt = match rebuilt {
            Some(node) => store
                .intern_node(node, identity, symbols)
                .map_err(|_| panic!("scalar transfer failed")),
            None => Ok(replacement),
        }?;
        completed.insert(id, rebuilt);
    }
    Ok(*completed.get(&representative).expect("representative completed"))
}

fn remap_scalar_node(
    node: &ScalarNode,
    remap: &impl Fn(ScalarId) -> ScalarId,
    binder: &BinderKey,
    replacement: Option<&ResolvedIntExpr>,
    symbols: &mut SymbolTables,
) -> Option<ScalarNode> {
    let map2 = |ids: &[ScalarId; 2]| [remap(ids[0]), remap(ids[1])];
    let map1 = |ids: &[ScalarId; 1]| [remap(ids[0])];
    Some(match node {
        ScalarNode::Source { source, indices } => ScalarNode::Source {
            source: substitute_source(*source, binder, replacement, symbols),
            indices: indices.iter().map(|id| remap(*id)).collect(),
        },
        ScalarNode::IntConst(value) => ScalarNode::IntConst(value.clone()),
        ScalarNode::IntParameter(value) => ScalarNode::IntParameter(value.clone()),
        ScalarNode::IntBinder(value) => ScalarNode::IntBinder(value.clone()),
        ScalarNode::IntAdd(ids) => ScalarNode::IntAdd(map2(ids)),
        ScalarNode::IntSub(ids) => ScalarNode::IntSub(map2(ids)),
        ScalarNode::IntMul(ids) => ScalarNode::IntMul(map2(ids)),
        ScalarNode::IntExactDiv(ids) => ScalarNode::IntExactDiv(map2(ids)),
        ScalarNode::IntEuclideanDiv(ids) => ScalarNode::IntEuclideanDiv(map2(ids)),
        ScalarNode::IntEuclideanRemainder(ids) => ScalarNode::IntEuclideanRemainder(map2(ids)),
        ScalarNode::IntRoundDiv(ids) => ScalarNode::IntRoundDiv(map2(ids)),
        ScalarNode::IntLog2Ceil(ids) => ScalarNode::IntLog2Ceil(map1(ids)),
        ScalarNode::BoolConst(value) => ScalarNode::BoolConst(*value),
        ScalarNode::IntEqual(ids) => ScalarNode::IntEqual(map2(ids)),
        ScalarNode::IntLess(ids) => ScalarNode::IntLess(map2(ids)),
        ScalarNode::IntLessEqual(ids) => ScalarNode::IntLessEqual(map2(ids)),
        ScalarNode::BitExtract { bit, input } => ScalarNode::BitExtract {
            bit: replacement.map_or_else(
                || bit.clone(),
                |replacement| substitute_resolved_iterative(bit, binder, replacement),
            ),
            input: map1(input),
        },
        ScalarNode::BoolToInt(ids) => ScalarNode::BoolToInt(map1(ids)),
        ScalarNode::RealConst(bits) => ScalarNode::RealConst(*bits),
        ScalarNode::IntToReal(ids) => ScalarNode::IntToReal(map1(ids)),
        ScalarNode::RealAdd(ids) => ScalarNode::RealAdd(map2(ids)),
        ScalarNode::RealSub(ids) => ScalarNode::RealSub(map2(ids)),
        ScalarNode::RealMul(ids) => ScalarNode::RealMul(map2(ids)),
        ScalarNode::RealDiv(ids) => ScalarNode::RealDiv(map2(ids)),
        ScalarNode::RealSqrt(ids) => ScalarNode::RealSqrt(map1(ids)),
        ScalarNode::Switch(ids) => {
            ScalarNode::Switch(ids.iter().map(|id| remap(*id)).collect::<Box<[_]>>())
        }
        ScalarNode::ExtractCoefficient { canonical_exclusive_upper, matrix, position } => {
            ScalarNode::ExtractCoefficient {
                canonical_exclusive_upper: canonical_exclusive_upper.clone(),
                matrix: *matrix,
                position: remap(*position),
            }
        }
    })
}

fn substitute_source(
    source: AtomicSourceId,
    binder: &BinderKey,
    replacement: Option<&ResolvedIntExpr>,
    symbols: &mut SymbolTables,
) -> AtomicSourceId {
    let Some(descriptor) = symbols.atomic_sources.get(source.0).cloned() else {
        return source;
    };
    let AtomicSourceKey::Sampler(sampler_id) = descriptor.key else {
        return source;
    };
    let Some(mut sampler) = symbols.samplers.get(sampler_id.0).cloned() else {
        return source;
    };
    if let SamplerIdentity::Preimage { trapdoor, .. } = &sampler {
        if let Some(descriptor) = symbols.trapdoors.get(trapdoor.0).cloned() {
            let descriptor = substitute_trapdoor_identity(&descriptor, binder, replacement);
            let trapdoor = TrapdoorDescriptorId(symbols.trapdoors.intern(descriptor));
            sampler = replace_sampler_trapdoor(sampler, trapdoor);
        }
    }
    sampler = substitute_sampler_identity(&sampler, binder, replacement);
    let sampler = SamplerDescriptorId(symbols.samplers.intern(sampler));
    AtomicSourceId(
        symbols.atomic_sources.intern(AtomicSourceDescriptor {
            key: AtomicSourceKey::Sampler(sampler),
            ..descriptor
        }),
    )
}

/// Selects a compact scalar family. Exact storage combines each physical lane
/// pointwise; shared templates combine only their one representatives.
pub fn select_family(
    store: &mut ScalarStore,
    symbols: &SymbolTables,
    selector: ScalarId,
    cases: &[FamilyLoweringValue<ScalarId>],
) -> Result<FamilyLoweringValue<ScalarId>, FamilyCoverageError> {
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
                        unreachable!("storage shape was checked before scalar construction");
                    };
                    selected_cases.push(elements[lane]);
                }
                selected.push(add_runtime_switch(store, symbols, selector, &selected_cases)?);
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
                    representative: add_runtime_switch(store, symbols, selector, &selected_cases)?,
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
    use crate::operational_noise::identity::{BinderDescriptor, OccurrenceScope, ProgramKey};

    fn binder(node: u32, slot: u32) -> BinderKey {
        BinderKey {
            loop_scope: OccurrenceScope {
                program: ProgramKey::Ideal,
                definition: mxx_ir_core::FrozenGraphScopeId::Root,
                path: Box::new([]),
            },
            loop_node: mxx_ir_core::NodeId(node.into()),
            slot,
        }
    }

    fn int(store: &mut ScalarStore, symbols: &SymbolTables, value: i64) -> ScalarId {
        store
            .intern_node(
                ScalarNode::IntConst(value.into()),
                ResolvedIntExpr::Const(value.into()),
                symbols,
            )
            .unwrap()
    }

    #[test]
    fn runtime_switch_retains_only_authoritatively_reachable_suffix() {
        let mut store = ScalarStore::default();
        let symbols = SymbolTables::default();
        let selector = int(&mut store, &symbols, 1);
        let cases = [10, 20, 30].map(|value| int(&mut store, &symbols, value));
        let switched = add_runtime_switch(&mut store, &symbols, selector, &cases).unwrap();
        assert!(
            matches!(store.node(switched), Some(ScalarNode::Switch(children)) if children.as_ref() == [selector, cases[0], cases[1]])
        );
    }

    #[test]
    fn dynamic_get_and_select_share_scalar_switch_construction() {
        let mut store = ScalarStore::default();
        let symbols = SymbolTables::default();
        let selector = int(&mut store, &symbols, 0);
        let first = [int(&mut store, &symbols, 10), int(&mut store, &symbols, 11)];
        let second = [int(&mut store, &symbols, 20), int(&mut store, &symbols, 21)];
        let family = FamilyLoweringValue {
            element_type: ScalarSort::Int,
            storage: FamilyCoverageStorage::ExactStored { elements: first.into() },
        };
        let dynamic = dynamic_get(&mut store, &symbols, &family, selector).unwrap();
        let selected = select_family(
            &mut store,
            &symbols,
            selector,
            &[
                family,
                FamilyLoweringValue {
                    element_type: ScalarSort::Int,
                    storage: FamilyCoverageStorage::ExactStored { elements: second.into() },
                },
            ],
        )
        .unwrap();
        let FamilyCoverageStorage::ExactStored { elements } = selected.storage else {
            panic!("exact families remain exact")
        };
        assert_eq!(elements.len(), 2);
        assert!(matches!(store.node(dynamic), Some(ScalarNode::Switch(_))));
        assert!(matches!(store.node(elements[0]), Some(ScalarNode::Switch(_))));
    }

    #[test]
    fn dynamic_get_rejects_negative_and_out_of_range_selectors() {
        let mut store = ScalarStore::default();
        let symbols = SymbolTables::default();
        let cases = [10, 20, 30].map(|value| int(&mut store, &symbols, value));
        let family = FamilyLoweringValue {
            element_type: ScalarSort::Int,
            storage: FamilyCoverageStorage::ExactStored { elements: cases.into() },
        };
        for selector_value in [-1, 3] {
            let selector = int(&mut store, &symbols, selector_value);
            assert!(matches!(
                dynamic_get(&mut store, &symbols, &family, selector),
                Err(FamilyCoverageError::DynamicIndexOutOfRange { .. })
            ));
        }
    }

    #[test]
    fn shared_substitution_replaces_only_the_owner_binder() {
        let mut store = ScalarStore::default();
        let mut symbols = SymbolTables::default();
        let owner = binder(1, 0);
        let foreign = binder(2, 0);
        for key in [owner.clone(), foreign.clone()] {
            symbols.binders.intern(BinderDescriptor { key, minimum: 0.into(), maximum: 7.into() });
        }
        let owner_id = store
            .intern_node(
                ScalarNode::IntBinder(owner.clone()),
                ResolvedIntExpr::Binder(owner.clone()),
                &symbols,
            )
            .unwrap();
        let foreign_id = store
            .intern_node(
                ScalarNode::IntBinder(foreign.clone()),
                ResolvedIntExpr::Binder(foreign.clone()),
                &symbols,
            )
            .unwrap();
        let representative = store
            .intern_node(
                ScalarNode::IntAdd([owner_id, foreign_id]),
                ResolvedIntExpr::Add(
                    Box::new(ResolvedIntExpr::Binder(owner.clone())),
                    Box::new(ResolvedIntExpr::Binder(foreign.clone())),
                ),
                &symbols,
            )
            .unwrap();
        let replacement = int(&mut store, &symbols, 5);
        let instantiated = instantiate_shared_element(
            &mut store,
            &mut symbols,
            representative,
            &owner,
            replacement,
            None,
            &mut || Ok::<(), ()>(()),
        )
        .unwrap();
        let ScalarNode::IntAdd([left, right]) = store.node(instantiated).unwrap() else {
            panic!("shared representative retains its operation")
        };
        assert_eq!(*left, replacement);
        assert_eq!(store.node(*right), Some(&ScalarNode::IntBinder(foreign)));
    }

    #[test]
    fn family_index_rejects_negative_and_count_upper_bound() {
        let count = BigUint::from(3_u8);
        assert!(validate_family_index(&IntegerDomain::exact(2), &count).is_ok());
        assert!(validate_family_index(&IntegerDomain::exact(-1), &count).is_err());
        assert!(validate_family_index(&IntegerDomain::exact(3), &count).is_err());
    }
}
