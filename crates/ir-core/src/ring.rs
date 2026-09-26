use crate::expr::{ExprError, IntExpr, ParamEnv};
use num_bigint::BigInt;
use num_traits::{One, ToPrimitive};
use serde::{Deserialize, Serialize};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

#[derive(Default)]
struct RingSerializationTable {
    ids: BTreeMap<RingRef, usize>,
    entries: Vec<serde_json::Value>,
}

thread_local! {
    static SERIALIZING_RINGS: RefCell<Option<RingSerializationTable>> = const { RefCell::new(None) };
    static DESERIALIZING_RINGS: RefCell<Option<Vec<RingRef>>> = const { RefCell::new(None) };
    static RESOLVED_RINGS: RefCell<Option<BTreeMap<(RingRef, ParamEnv, usize), ConcreteRing>>> = const { RefCell::new(None) };
}

pub(crate) fn with_resolution_cache<T>(work: impl FnOnce() -> T) -> T {
    let started = RESOLVED_RINGS.with(|cache| {
        if cache.borrow().is_some() {
            false
        } else {
            *cache.borrow_mut() = Some(BTreeMap::new());
            true
        }
    });
    struct Clear(bool);
    impl Drop for Clear {
        fn drop(&mut self) {
            if self.0 {
                RESOLVED_RINGS.with(|cache| {
                    cache.borrow_mut().take();
                });
            }
        }
    }
    let clear = Clear(started);
    let result = work();
    drop(clear);
    result
}

pub(crate) fn serialize_with_ring_table<T: Serialize>(
    value: &T,
) -> Result<(serde_json::Value, Vec<serde_json::Value>), serde_json::Error> {
    SERIALIZING_RINGS.with(|table| {
        assert!(table.borrow().is_none(), "nested graph ring-table serialization");
        *table.borrow_mut() = Some(RingSerializationTable::default());
    });
    let result = serde_json::to_value(value);
    let table = SERIALIZING_RINGS
        .with(|state| state.borrow_mut().take().expect("ring table active").entries);
    result.map(|value| (value, table))
}

pub(crate) fn deserialize_with_ring_table<T: serde::de::DeserializeOwned>(
    value: serde_json::Value,
    entries: Vec<serde_json::Value>,
) -> Result<T, serde_json::Error> {
    DESERIALIZING_RINGS.with(|table| {
        assert!(table.borrow().is_none(), "nested graph ring-table deserialization");
        *table.borrow_mut() = Some(Vec::new());
    });
    let result = (|| {
        for entry in entries {
            let ring: RingRef = serde_json::from_value(entry)?;
            DESERIALIZING_RINGS
                .with(|table| table.borrow_mut().as_mut().expect("ring table active").push(ring));
        }
        serde_json::from_value(value)
    })();
    DESERIALIZING_RINGS.with(|table| {
        table.borrow_mut().take();
    });
    result
}

pub(crate) fn ring_table_refs(
    entries: Vec<serde_json::Value>,
) -> Result<Vec<RingRef>, serde_json::Error> {
    let references = serde_json::Value::Array(
        (0..entries.len()).map(|id| serde_json::json!({"$ring": id})).collect(),
    );
    deserialize_with_ring_table(references, entries)
}

pub type ResolveCrtBasis = fn(u32, usize, usize, Option<Vec<u64>>) -> Result<Vec<u64>, String>;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct RingRef(Arc<RingExpr>);

impl Serialize for RingRef {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if SERIALIZING_RINGS.with(|table| table.borrow().is_some()) {
            let existing = SERIALIZING_RINGS.with(|table| {
                table.borrow().as_ref().and_then(|table| table.ids.get(self).copied())
            });
            let id = match existing {
                Some(id) => id,
                None => {
                    let entry = serde_json::to_value(self.expression())
                        .map_err(serde::ser::Error::custom)?;
                    SERIALIZING_RINGS.with(|table| {
                        let mut table = table.borrow_mut();
                        let table = table.as_mut().expect("ring table active");
                        let id = table.entries.len();
                        table.entries.push(entry);
                        table.ids.insert(self.clone(), id);
                        id
                    })
                }
            };
            serde_json::json!({"$ring": id}).serialize(serializer)
        } else {
            self.expression().serialize(serializer)
        }
    }
}

impl<'de> Deserialize<'de> for RingRef {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        if let Some(id) = value.get("$ring").and_then(serde_json::Value::as_u64) {
            return DESERIALIZING_RINGS.with(|table| {
                table
                    .borrow()
                    .as_ref()
                    .and_then(|rings| rings.get(id as usize).cloned())
                    .ok_or_else(|| {
                        serde::de::Error::custom("invalid or forward ring-table reference")
                    })
            });
        }
        serde_json::from_value::<RingExpr>(value).map(Self::new).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum RingExpr {
    Generated { crt_bits: IntExpr, crt_depth: IntExpr, ring_dimension: u32 },
    Explicit { crt_moduli: Vec<IntExpr>, ring_dimension: u32 },
    Slice { source: RingRef, start: IntExpr, end: IntExpr },
    Select { source: RingRef, indices: Vec<IntExpr> },
    Concat { left: RingRef, right: RingRef },
}

impl RingRef {
    pub fn new(expr: RingExpr) -> Self {
        Self(Arc::new(expr))
    }
    pub fn expression(&self) -> &RingExpr {
        &self.0
    }
    /// Whether this ring's ordered CRT basis depends on a loop index.
    pub fn contains_loop_index(&self) -> bool {
        self.expression().contains_loop_index()
    }
    pub fn ring_dimension(&self) -> u32 {
        match self.expression() {
            RingExpr::Generated { ring_dimension, .. } |
            RingExpr::Explicit { ring_dimension, .. } => *ring_dimension,
            RingExpr::Slice { source, .. } | RingExpr::Select { source, .. } => {
                source.ring_dimension()
            }
            RingExpr::Concat { left, .. } => left.ring_dimension(),
        }
    }

    pub fn resolve(
        &self,
        env: &ParamEnv,
        resolve_basis: ResolveCrtBasis,
    ) -> Result<ConcreteRing, ExprError> {
        resolve_ring(self, env, resolve_basis)
    }
}

impl RingExpr {
    pub(crate) fn contains_loop_index(&self) -> bool {
        match self {
            Self::Generated { crt_bits, crt_depth, .. } => {
                crt_bits.contains_loop_index() || crt_depth.contains_loop_index()
            }
            Self::Explicit { crt_moduli, .. } => {
                crt_moduli.iter().any(IntExpr::contains_loop_index)
            }
            Self::Slice { source, start, end } => {
                source.expression().contains_loop_index() ||
                    start.contains_loop_index() ||
                    end.contains_loop_index()
            }
            Self::Select { source, indices } => {
                source.expression().contains_loop_index() ||
                    indices.iter().any(IntExpr::contains_loop_index)
            }
            Self::Concat { left, right } => {
                left.expression().contains_loop_index() || right.expression().contains_loop_index()
            }
        }
    }
    pub(crate) fn contains_variable(&self, variable: &str) -> bool {
        match self {
            Self::Generated { crt_bits, crt_depth, .. } => {
                crt_bits.contains_variable(variable) || crt_depth.contains_variable(variable)
            }
            Self::Explicit { crt_moduli, .. } => {
                crt_moduli.iter().any(|q| q.contains_variable(variable))
            }
            Self::Slice { source, start, end } => {
                source.expression().contains_variable(variable) ||
                    start.contains_variable(variable) ||
                    end.contains_variable(variable)
            }
            Self::Select { source, indices } => {
                source.expression().contains_variable(variable) ||
                    indices.iter().any(|i| i.contains_variable(variable))
            }
            Self::Concat { left, right } => {
                left.expression().contains_variable(variable) ||
                    right.expression().contains_variable(variable)
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ConcreteRing(Arc<ConcreteRingData>);

impl Serialize for ConcreteRing {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ConcreteRing {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        ConcreteRingData::deserialize(deserializer).map(|data| Self(Arc::new(data)))
    }
}

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
struct ConcreteRingData {
    crt_moduli: Box<[u64]>,
    ring_dimension: u32,
}

impl ConcreteRing {
    pub fn crt_moduli(&self) -> &[u64] {
        &self.0.crt_moduli
    }
    pub fn ring_dimension(&self) -> u32 {
        self.0.ring_dimension
    }
    pub fn crt_depth(&self) -> usize {
        self.0.crt_moduli.len()
    }
    pub fn modulus(&self) -> BigInt {
        self.crt_moduli().iter().fold(BigInt::one(), |acc, q| acc * q)
    }

    fn checked(
        moduli: Vec<u64>,
        n: u32,
        resolver: ResolveCrtBasis,
        generated: bool,
    ) -> Result<Self, ExprError> {
        if n == 0 || !n.is_power_of_two() || moduli.is_empty() {
            return Err(ExprError::RingResolution(
                "ring dimension must be a positive power of two and the CRT basis must be nonempty"
                    .into(),
            ));
        }
        let modulus = 2 * u64::from(n);
        let mut seen = BTreeSet::new();
        for &q in &moduli {
            if q <= 2 || q > (1u64 << 60) - 1 || (q - 1) % modulus != 0 || !seen.insert(q) {
                return Err(ExprError::RingResolution(format!(
                    "invalid, repeated, or unsupported CRT modulus {q}"
                )));
            }
        }
        let bits = moduli.iter().map(|q| 64 - q.leading_zeros() as usize).max().unwrap();
        if !generated {
            let checked = resolver(n, moduli.len(), bits, Some(moduli.clone()))
                .map_err(ExprError::RingResolution)?;
            if checked != moduli {
                return Err(ExprError::RingResolution(
                    "CRT resolver changed the explicit basis or its order".into(),
                ));
            }
        }
        Ok(Self(Arc::new(ConcreteRingData {
            crt_moduli: moduli.into_boxed_slice(),
            ring_dimension: n,
        })))
    }
}

pub(crate) fn resolve_ring(
    ring: &RingRef,
    env: &ParamEnv,
    resolver: ResolveCrtBasis,
) -> Result<ConcreteRing, ExprError> {
    with_resolution_cache(|| {
        let key = (ring.clone(), env.clone(), resolver as usize);
        if let Some(hit) = RESOLVED_RINGS
            .with(|cache| cache.borrow().as_ref().and_then(|cache| cache.get(&key).cloned()))
        {
            return Ok(hit);
        }
        let value = resolve_ring_uncached(ring, env, resolver)?;
        RESOLVED_RINGS.with(|cache| {
            cache
                .borrow_mut()
                .as_mut()
                .expect("resolution cache active")
                .insert(key, value.clone());
        });
        Ok(value)
    })
}

fn resolve_ring_uncached(
    ring: &RingRef,
    env: &ParamEnv,
    resolver: ResolveCrtBasis,
) -> Result<ConcreteRing, ExprError> {
    let n = ring.ring_dimension();
    let generated = matches!(ring.expression(), RingExpr::Generated { .. });
    let moduli = match ring.expression() {
        RingExpr::Generated { crt_bits, crt_depth, .. } => {
            let bits = crt_bits
                .evaluate_with_rings(env, resolver)?
                .to_usize()
                .ok_or_else(|| ExprError::RingResolution("CRT bits must fit usize".into()))?;
            let depth = crt_depth
                .evaluate_with_rings(env, resolver)?
                .to_usize()
                .ok_or_else(|| ExprError::RingResolution("CRT depth must fit usize".into()))?;
            if n == 0 || !n.is_power_of_two() || depth == 0 || !(2..=60).contains(&bits) {
                return Err(ExprError::RingResolution(
                    "invalid generated CRT dimension, depth, or bits".into(),
                ));
            }
            let moduli = resolver(n, depth, bits, None).map_err(ExprError::RingResolution)?;
            if moduli.len() != depth ||
                moduli.iter().any(|q| 64 - q.leading_zeros() as usize != bits)
            {
                return Err(ExprError::RingResolution(
                    "CRT generator returned the wrong depth or bit width".into(),
                ));
            }
            moduli
        }
        RingExpr::Explicit { crt_moduli, .. } => crt_moduli
            .iter()
            .map(|q| {
                q.evaluate_with_rings(env, resolver)?
                    .to_u64()
                    .ok_or_else(|| ExprError::RingResolution("CRT modulus must fit u64".into()))
            })
            .collect::<Result<Vec<_>, _>>()?,
        RingExpr::Slice { source, start, end } => {
            let source = resolve_ring(source, env, resolver)?;
            let start = start.evaluate_with_rings(env, resolver)?.to_usize().ok_or_else(|| {
                ExprError::RingResolution("CRT slice start is negative or too large".into())
            })?;
            let end = end.evaluate_with_rings(env, resolver)?.to_usize().ok_or_else(|| {
                ExprError::RingResolution("CRT slice end is negative or too large".into())
            })?;
            if start >= end || end > source.crt_depth() {
                return Err(ExprError::RingResolution("CRT slice is empty or out of range".into()));
            }
            source.crt_moduli()[start..end].to_vec()
        }
        RingExpr::Select { source, indices } => {
            let source = resolve_ring(source, env, resolver)?;
            indices
                .iter()
                .map(|index| {
                    let index =
                        index.evaluate_with_rings(env, resolver)?.to_usize().ok_or_else(|| {
                            ExprError::RingResolution("CRT index is negative or too large".into())
                        })?;
                    source.crt_moduli().get(index).copied().ok_or_else(|| {
                        ExprError::RingResolution("CRT index is out of range".into())
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
        }
        RingExpr::Concat { left, right } => {
            if left.ring_dimension() != right.ring_dimension() {
                return Err(ExprError::RingResolution("CRT concat ring dimensions differ".into()));
            }
            let mut left = resolve_ring(left, env, resolver)?.crt_moduli().to_vec();
            left.extend_from_slice(resolve_ring(right, env, resolver)?.crt_moduli());
            left
        }
    };
    ConcreteRing::checked(moduli, n, resolver, generated)
}

#[cfg(test)]
pub(crate) fn test_resolve_basis(
    n: u32,
    depth: usize,
    bits: usize,
    moduli: Option<Vec<u64>>,
) -> Result<Vec<u64>, String> {
    const PRIMES: &[u64] = &[17, 97, 113, 193, 241, 257, 65537];
    match moduli {
        Some(moduli) => {
            if moduli.len() != depth ||
                moduli
                    .iter()
                    .any(|q| !PRIMES.contains(q) || 64 - q.leading_zeros() as usize > bits) ||
                moduli.iter().map(|q| 64 - q.leading_zeros() as usize).max() != Some(bits)
            {
                return Err("test basis has unsupported or mismatched moduli".into());
            }
            Ok(moduli)
        }
        None => {
            let basis = PRIMES
                .iter()
                .copied()
                .filter(|q| {
                    64 - q.leading_zeros() as usize == bits &&
                        n != 0 &&
                        (q - 1) % (2 * u64::from(n)) == 0
                })
                .take(depth)
                .collect::<Vec<_>>();
            if basis.len() != depth {
                return Err("test generator has insufficient moduli".into());
            }
            Ok(basis)
        }
    }
}

#[cfg(test)]
pub(crate) fn test_ring(modulus: i64, n: u32) -> RingRef {
    let mut remaining = modulus;
    let mut moduli = Vec::new();
    for prime in [17, 97, 113, 193, 241, 257, 65537] {
        if remaining % prime == 0 {
            moduli.push(IntExpr::constant(prime));
            remaining /= prime;
        }
    }
    if remaining != 1 || moduli.is_empty() {
        moduli = vec![IntExpr::constant(modulus)];
    }
    RingRef::new(RingExpr::Explicit { crt_moduli: moduli, ring_dimension: n })
}

#[cfg(test)]
pub(crate) fn test_concrete_ring(modulus: i64, n: u32) -> ConcreteRing {
    test_ring(modulus, n)
        .resolve(&ParamEnv::default(), test_resolve_basis)
        .expect("valid test ring")
}

#[cfg(test)]
pub(crate) fn test_validate(
    graph: &crate::Graph,
    env: &ParamEnv,
) -> Result<crate::ValidatedGraph, crate::ValidationError> {
    crate::validate(graph, env, test_resolve_basis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static GENERATED_CALLS: AtomicUsize = AtomicUsize::new(0);

    fn counting_resolver(
        n: u32,
        depth: usize,
        bits: usize,
        moduli: Option<Vec<u64>>,
    ) -> Result<Vec<u64>, String> {
        if moduli.is_none() {
            GENERATED_CALLS.fetch_add(1, Ordering::Relaxed);
        }
        test_resolve_basis(n, depth, bits, moduli)
    }

    #[test]
    fn test_ring_resolution_memoizes_generated_basis_per_binding() {
        GENERATED_CALLS.store(0, Ordering::Relaxed);
        let ring = RingRef::new(RingExpr::Generated {
            crt_bits: IntExpr::constant(7),
            crt_depth: IntExpr::constant(2),
            ring_dimension: 8,
        });
        with_resolution_cache(|| {
            assert_eq!(
                ring.resolve(&ParamEnv::default(), counting_resolver).unwrap().crt_moduli(),
                &[97, 113]
            );
            assert_eq!(
                IntExpr::RingCrtDepth(ring.clone())
                    .evaluate_with_rings(&ParamEnv::default(), counting_resolver)
                    .unwrap(),
                BigInt::from(2)
            );
        });
        assert_eq!(GENERATED_CALLS.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_ring_properties_preserve_order_and_require_resolution() {
        let source = RingRef::new(RingExpr::Explicit {
            crt_moduli: vec![IntExpr::constant(17), IntExpr::constant(97)],
            ring_dimension: 8,
        });
        let reversed = RingRef::new(RingExpr::Select {
            source: source.clone(),
            indices: vec![IntExpr::constant(1), IntExpr::constant(0)],
        });
        let env = ParamEnv::default();
        assert_eq!(
            IntExpr::RingModulus(source.clone()).evaluate(&env),
            Err(ExprError::RingResolutionRequired)
        );
        assert_eq!(source.resolve(&env, test_resolve_basis).unwrap().crt_moduli(), &[17, 97]);
        assert_eq!(reversed.resolve(&env, test_resolve_basis).unwrap().crt_moduli(), &[97, 17]);
        assert_ne!(
            source.resolve(&env, test_resolve_basis).unwrap(),
            reversed.resolve(&env, test_resolve_basis).unwrap()
        );
        assert_eq!(
            IntExpr::RingModulus(reversed.clone())
                .evaluate_with_rings(&env, test_resolve_basis)
                .unwrap(),
            BigInt::from(17 * 97)
        );
        assert_eq!(
            IntExpr::RingCrtDepth(reversed.clone())
                .evaluate_with_rings(&env, test_resolve_basis)
                .unwrap(),
            BigInt::from(2)
        );
        assert_eq!(
            IntExpr::RingCrtModulus { ring: reversed, index: Box::new(IntExpr::constant(0)) }
                .evaluate_with_rings(&env, test_resolve_basis)
                .unwrap(),
            BigInt::from(97)
        );
        let real = crate::RealExpr::FromInt(IntExpr::RingCrtDepth(source));
        assert_eq!(real.evaluate_f64_with_rings(&env, test_resolve_basis).unwrap(), 2.0);
    }

    #[test]
    fn test_ring_derivations_reject_duplicate_or_invalid_basis() {
        let source = test_ring(17 * 97, 8);
        let selected = RingRef::new(RingExpr::Select {
            source: source.clone(),
            indices: vec![IntExpr::constant(0), IntExpr::constant(0)],
        });
        assert!(selected.resolve(&ParamEnv::default(), test_resolve_basis).is_err());
        let concat =
            RingRef::new(RingExpr::Concat { left: source.clone(), right: test_ring(17, 8) });
        assert!(concat.resolve(&ParamEnv::default(), test_resolve_basis).is_err());
        let empty = RingRef::new(RingExpr::Slice {
            source,
            start: IntExpr::constant(1),
            end: IntExpr::constant(1),
        });
        assert!(empty.resolve(&ParamEnv::default(), test_resolve_basis).is_err());
    }
}
