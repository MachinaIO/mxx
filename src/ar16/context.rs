use std::collections::HashMap;

use crate::{circuit::gate::GateId, poly::Poly};

use super::{AR16Encoding, AR16PublicKey, Level, advice::Advice};

/// Shared mutable state for AR16 evaluation.
#[derive(Debug)]
pub struct EvalContext<P: Poly> {
    advice: Advice<P>,
    params: P::Params,
    pk_cache: HashMap<GateId, AR16PublicKey<P>>,
    ct_cache: HashMap<GateId, AR16Encoding<P>>,
    times_s_cache: HashMap<(Level, GateId), AR16Encoding<P>>,
    gate_levels: HashMap<GateId, Level>,
}

impl<P: Poly> EvalContext<P> {
    pub fn new(advice: Advice<P>, params: P::Params) -> Self {
        Self {
            advice,
            params,
            pk_cache: HashMap::new(),
            ct_cache: HashMap::new(),
            times_s_cache: HashMap::new(),
            gate_levels: HashMap::new(),
        }
    }

    pub fn advice(&self) -> &Advice<P> {
        &self.advice
    }

    pub fn advice_mut(&mut self) -> &mut Advice<P> {
        &mut self.advice
    }

    pub fn params(&self) -> &P::Params {
        &self.params
    }

    pub fn pk_cache(&self) -> &HashMap<GateId, AR16PublicKey<P>> {
        &self.pk_cache
    }

    pub fn pk_cache_mut(&mut self) -> &mut HashMap<GateId, AR16PublicKey<P>> {
        &mut self.pk_cache
    }

    pub fn encoding_cache(&self) -> &HashMap<GateId, AR16Encoding<P>> {
        &self.ct_cache
    }

    pub fn encoding_cache_mut(&mut self) -> &mut HashMap<GateId, AR16Encoding<P>> {
        &mut self.ct_cache
    }

    pub fn times_s_cache(&self) -> &HashMap<(Level, GateId), AR16Encoding<P>> {
        &self.times_s_cache
    }

    pub fn times_s_cache_mut(&mut self) -> &mut HashMap<(Level, GateId), AR16Encoding<P>> {
        &mut self.times_s_cache
    }

    pub fn gate_level(&self, gate: GateId) -> Option<Level> {
        self.gate_levels.get(&gate).copied()
    }

    pub fn set_gate_level(&mut self, gate: GateId, level: Level) {
        self.gate_levels.insert(gate, level);
    }

    pub fn clear(&mut self) {
        self.pk_cache.clear();
        self.ct_cache.clear();
        self.times_s_cache.clear();
        self.gate_levels.clear();
    }
}
