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
    static RESOLVED_RINGS: RefCell<Option<BTreeMap<(RingRef, ParamEnv), ConcreteRing>>> = const { RefCell::new(None) };
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

/// Generates the ordered CRT basis of `crt_depth` primes `q = 1 mod 2N` that OpenFHE's
/// `ILDCRTParams(2N, crt_depth, crt_bits)` generates: the largest such prime below `2^crt_bits`,
/// then each next smaller one. The first prime must have exactly `crt_bits` bits.
pub fn generate_crt_basis(
    ring_dimension: u32,
    crt_depth: usize,
    crt_bits: usize,
) -> Result<Vec<u64>, String> {
    if ring_dimension == 0 || !ring_dimension.is_power_of_two() || crt_depth == 0 {
        return Err("CRT generation needs a power-of-two dimension and a positive depth".into());
    }
    if !(2..=60).contains(&crt_bits) {
        return Err("CRT width must be in 2..=60".into());
    }
    let step = 2 * u64::from(ring_dimension);
    let top = 1u64 << crt_bits;
    // OpenFHE's `LastPrime` starts at `2^bits + 1 - step` when `step` divides `2^bits`.
    let mut candidate = if top % step == 0 { top + 1 - step } else { 1 };
    let mut basis = Vec::with_capacity(crt_depth);
    while basis.len() < crt_depth {
        if candidate <= 2 {
            return Err(format!(
                "no {crt_depth} primes of at most {crt_bits} bits are 1 mod {step}"
            ));
        }
        if is_prime(candidate) {
            if basis.is_empty() && 64 - candidate.leading_zeros() as usize != crt_bits {
                return Err(format!("no {crt_bits}-bit prime is 1 mod {step}"));
            }
            basis.push(candidate);
        }
        candidate = candidate.saturating_sub(step);
    }
    Ok(basis)
}

/// Deterministic Miller-Rabin test; these bases decide primality for every `u64`.
pub(crate) fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    const BASES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    if let Some(&base) = BASES.iter().find(|&&base| n % base == 0) {
        return n == base;
    }
    let mul = |a: u64, b: u64| (u128::from(a) * u128::from(b) % u128::from(n)) as u64;
    let pow = |mut base: u64, mut exponent: u64| {
        let mut result = 1;
        while exponent > 0 {
            if exponent & 1 == 1 {
                result = mul(result, base);
            }
            base = mul(base, base);
            exponent >>= 1;
        }
        result
    };
    let shift = (n - 1).trailing_zeros();
    let odd = (n - 1) >> shift;
    BASES.iter().all(|&base| {
        let mut x = pow(base, odd);
        if x == 1 || x == n - 1 {
            return true;
        }
        (1..shift).any(|_| {
            x = mul(x, x);
            x == n - 1
        })
    })
}

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

    pub fn resolve(&self, env: &ParamEnv) -> Result<ConcreteRing, ExprError> {
        resolve_ring(self, env)
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

    fn checked(moduli: Vec<u64>, n: u32) -> Result<Self, ExprError> {
        if n == 0 || !n.is_power_of_two() || moduli.is_empty() {
            return Err(ExprError::RingResolution(
                "ring dimension must be a positive power of two and the CRT basis must be nonempty"
                    .into(),
            ));
        }
        let modulus = 2 * u64::from(n);
        let mut seen = BTreeSet::new();
        for &q in &moduli {
            if q <= 2 ||
                q > (1u64 << 60) - 1 ||
                (q - 1) % modulus != 0 ||
                !is_prime(q) ||
                !seen.insert(q)
            {
                return Err(ExprError::RingResolution(format!(
                    "invalid, repeated, or unsupported CRT modulus {q}"
                )));
            }
        }
        Ok(Self(Arc::new(ConcreteRingData {
            crt_moduli: moduli.into_boxed_slice(),
            ring_dimension: n,
        })))
    }
}

pub(crate) fn resolve_ring(ring: &RingRef, env: &ParamEnv) -> Result<ConcreteRing, ExprError> {
    with_resolution_cache(|| {
        let key = (ring.clone(), env.clone());
        if let Some(hit) = RESOLVED_RINGS
            .with(|cache| cache.borrow().as_ref().and_then(|cache| cache.get(&key).cloned()))
        {
            return Ok(hit);
        }
        let value = resolve_ring_uncached(ring, env)?;
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

fn resolve_ring_uncached(ring: &RingRef, env: &ParamEnv) -> Result<ConcreteRing, ExprError> {
    let n = ring.ring_dimension();
    let moduli = match ring.expression() {
        RingExpr::Generated { crt_bits, crt_depth, .. } => {
            let bits = crt_bits
                .evaluate(env)?
                .to_usize()
                .ok_or_else(|| ExprError::RingResolution("CRT bits must fit usize".into()))?;
            let depth = crt_depth
                .evaluate(env)?
                .to_usize()
                .ok_or_else(|| ExprError::RingResolution("CRT depth must fit usize".into()))?;
            if n == 0 || !n.is_power_of_two() || depth == 0 || !(2..=60).contains(&bits) {
                return Err(ExprError::RingResolution(
                    "invalid generated CRT dimension, depth, or bits".into(),
                ));
            }
            let moduli = generate_crt_basis(n, depth, bits).map_err(ExprError::RingResolution)?;
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
                q.evaluate(env)?
                    .to_u64()
                    .ok_or_else(|| ExprError::RingResolution("CRT modulus must fit u64".into()))
            })
            .collect::<Result<Vec<_>, _>>()?,
        RingExpr::Slice { source, start, end } => {
            let source = resolve_ring(source, env)?;
            let start = start.evaluate(env)?.to_usize().ok_or_else(|| {
                ExprError::RingResolution("CRT slice start is negative or too large".into())
            })?;
            let end = end.evaluate(env)?.to_usize().ok_or_else(|| {
                ExprError::RingResolution("CRT slice end is negative or too large".into())
            })?;
            if start >= end || end > source.crt_depth() {
                return Err(ExprError::RingResolution("CRT slice is empty or out of range".into()));
            }
            source.crt_moduli()[start..end].to_vec()
        }
        RingExpr::Select { source, indices } => {
            let source = resolve_ring(source, env)?;
            indices
                .iter()
                .map(|index| {
                    let index = index.evaluate(env)?.to_usize().ok_or_else(|| {
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
            let mut left = resolve_ring(left, env)?.crt_moduli().to_vec();
            left.extend_from_slice(resolve_ring(right, env)?.crt_moduli());
            left
        }
    };
    ConcreteRing::checked(moduli, n)
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
    test_ring(modulus, n).resolve(&ParamEnv::default()).expect("valid test ring")
}

#[cfg(test)]
pub(crate) fn test_validate(
    graph: &crate::Graph,
    env: &ParamEnv,
) -> Result<crate::ValidatedGraph, crate::ValidationError> {
    crate::validate(graph, env)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generated_basis_follows_openfhe_order() {
        // 2^7 + 1 - 16 = 113, then the next smaller prime that is 1 mod 16.
        assert_eq!(generate_crt_basis(8, 2, 7).unwrap(), [113, 97]);
        assert_eq!(generate_crt_basis(8, 1, 5).unwrap(), [17]);
        // The largest 1 mod 16 value below 2^6 is 49, which is not prime, so a 6-bit
        // prime does not exist and the smaller 17 is rejected.
        assert!(generate_crt_basis(8, 1, 6).is_err());
        assert!(generate_crt_basis(8, 2, 5).is_err());
        assert!(generate_crt_basis(64, 1, 5).is_err());
        let ring = RingRef::new(RingExpr::Generated {
            crt_bits: IntExpr::constant(7),
            crt_depth: IntExpr::constant(2),
            ring_dimension: 8,
        });
        assert_eq!(ring.resolve(&ParamEnv::default()).unwrap().crt_moduli(), &[113, 97]);
        assert_eq!(
            IntExpr::RingCrtDepth(ring).evaluate(&ParamEnv::default()).unwrap(),
            BigInt::from(2)
        );
    }

    #[test]
    fn test_primality_matches_trial_division() {
        let trial = |n: u64| n >= 2 && (2..).take_while(|d| d * d <= n).all(|d| n % d != 0);
        assert!((0..20_000).all(|n| is_prime(n) == trial(n)));
        assert!(is_prime((1u64 << 61) - 1));
        assert!(!is_prime(3_215_031_751));
    }

    #[test]
    fn test_ring_properties_preserve_order() {
        let source = RingRef::new(RingExpr::Explicit {
            crt_moduli: vec![IntExpr::constant(17), IntExpr::constant(97)],
            ring_dimension: 8,
        });
        let reversed = RingRef::new(RingExpr::Select {
            source: source.clone(),
            indices: vec![IntExpr::constant(1), IntExpr::constant(0)],
        });
        let env = ParamEnv::default();
        assert_eq!(source.resolve(&env).unwrap().crt_moduli(), &[17, 97]);
        assert_eq!(reversed.resolve(&env).unwrap().crt_moduli(), &[97, 17]);
        assert_ne!(source.resolve(&env).unwrap(), reversed.resolve(&env).unwrap());
        assert_eq!(
            IntExpr::RingModulus(reversed.clone()).evaluate(&env).unwrap(),
            BigInt::from(17 * 97)
        );
        assert_eq!(
            IntExpr::RingCrtDepth(reversed.clone()).evaluate(&env).unwrap(),
            BigInt::from(2)
        );
        assert_eq!(
            IntExpr::RingCrtModulus { ring: reversed, index: Box::new(IntExpr::constant(0)) }
                .evaluate(&env)
                .unwrap(),
            BigInt::from(97)
        );
        let real = crate::RealExpr::FromInt(IntExpr::RingCrtDepth(source));
        assert_eq!(real.evaluate_f64(&env).unwrap(), 2.0);
    }

    #[test]
    fn test_ring_derivations_reject_duplicate_or_invalid_basis() {
        let source = test_ring(17 * 97, 8);
        let selected = RingRef::new(RingExpr::Select {
            source: source.clone(),
            indices: vec![IntExpr::constant(0), IntExpr::constant(0)],
        });
        assert!(selected.resolve(&ParamEnv::default()).is_err());
        let concat =
            RingRef::new(RingExpr::Concat { left: source.clone(), right: test_ring(17, 8) });
        assert!(concat.resolve(&ParamEnv::default()).is_err());
        let empty = RingRef::new(RingExpr::Slice {
            source,
            start: IntExpr::constant(1),
            end: IntExpr::constant(1),
        });
        assert!(empty.resolve(&ParamEnv::default()).is_err());
    }
}
