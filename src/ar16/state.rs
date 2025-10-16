use crate::{
    ar16::{AR16Encoding, AR16PublicKey, advice::AdviceSet},
    circuit::gate::GateId,
    poly::Poly,
};
use std::collections::HashMap;

/// Shared evaluation context caching advice, modulus towers, and memoization tables.
#[derive(Debug)]
pub struct EvalContext<P: Poly> {
    advice_sets: Vec<AdviceSet<P>>,
    pk_cache: HashMap<GateId, AR16PublicKey<P>>,
    encoding_cache: HashMap<GateId, AR16Encoding<P>>,
    gate_levels: HashMap<GateId, usize>,
}

impl<P: Poly> EvalContext<P> {
    pub fn new(advice_sets: Vec<AdviceSet<P>>) -> Self {
        Self {
            advice_sets,
            pk_cache: HashMap::new(),
            encoding_cache: HashMap::new(),
            gate_levels: HashMap::new(),
        }
    }

    pub fn advice_sets(&self) -> &[AdviceSet<P>] {
        &self.advice_sets
    }

    pub fn pk_cache(&self) -> &HashMap<GateId, AR16PublicKey<P>> {
        &self.pk_cache
    }

    pub fn pk_cache_mut(&mut self) -> &mut HashMap<GateId, AR16PublicKey<P>> {
        &mut self.pk_cache
    }

    pub fn encoding_cache(&self) -> &HashMap<GateId, AR16Encoding<P>> {
        &self.encoding_cache
    }

    pub fn encoding_cache_mut(&mut self) -> &mut HashMap<GateId, AR16Encoding<P>> {
        &mut self.encoding_cache
    }

    pub fn advice_for_level(&self, level: usize) -> Option<&AdviceSet<P>> {
        self.advice_sets.get(level)
    }

    pub fn advice_for_level_mut(&mut self, level: usize) -> Option<&mut AdviceSet<P>> {
        self.advice_sets.get_mut(level)
    }

    pub fn gate_level(&self, gate: GateId) -> Option<usize> {
        self.gate_levels.get(&gate).copied()
    }

    pub fn set_gate_level(&mut self, gate: GateId, level: usize) {
        self.gate_levels.insert(gate, level);
    }

    pub fn clear_caches(&mut self) {
        self.pk_cache.clear();
        self.encoding_cache.clear();
        self.gate_levels.clear();
    }
}

impl<P: Poly> Default for EvalContext<P> {
    fn default() -> Self {
        Self {
            advice_sets: Vec::new(),
            pk_cache: HashMap::new(),
            encoding_cache: HashMap::new(),
            gate_levels: HashMap::new(),
        }
    }
}
