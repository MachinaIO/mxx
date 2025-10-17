use std::collections::BTreeMap;

use crate::{circuit::gate::GateId, poly::Poly};

use super::{AR16Encoding, Level};

/// Index into advice tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AdviceKey {
    /// Advice corresponding to a gate's output message.
    Msg(GateId),
    /// Advice holding `E_k(s^2)` at level `k`.
    S2(Level),
}

/// Storage for all advice encodings required by AR16 evaluation.
#[derive(Debug, Clone)]
pub struct Advice<P: Poly> {
    c1_x: BTreeMap<GateId, AR16Encoding<P>>,
    c1_s: Option<AR16Encoding<P>>,
    lift: Vec<BTreeMap<AdviceKey, AR16Encoding<P>>>,
    lift_s: Vec<BTreeMap<AdviceKey, AR16Encoding<P>>>,
}

impl<P: Poly> Advice<P> {
    pub fn new() -> Self {
        Self { c1_x: BTreeMap::new(), c1_s: None, lift: Vec::new(), lift_s: Vec::new() }
    }

    pub fn with_capacity(levels: usize) -> Self {
        Self {
            c1_x: BTreeMap::new(),
            c1_s: None,
            lift: vec![BTreeMap::new(); levels],
            lift_s: vec![BTreeMap::new(); levels],
        }
    }
}

impl<P: Poly> Default for Advice<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: Poly> Advice<P> {
    pub fn insert_e1_x(&mut self, gate: GateId, encoding: AR16Encoding<P>) {
        self.c1_x.insert(gate, encoding);
    }

    pub fn insert_e1_s(&mut self, encoding: AR16Encoding<P>) {
        self.c1_s = Some(encoding);
    }

    pub fn e1_x(&self, gate: GateId) -> Option<&AR16Encoding<P>> {
        self.c1_x.get(&gate)
    }

    pub fn e1_s(&self) -> Option<&AR16Encoding<P>> {
        self.c1_s.as_ref()
    }

    pub fn insert_ek(&mut self, level: Level, key: AdviceKey, encoding: AR16Encoding<P>) {
        if level.0 < 2 {
            return;
        }
        let idx = (level.0 - 2) as usize;
        if self.lift.len() <= idx {
            self.lift.resize(idx + 1, BTreeMap::new());
        }
        self.lift[idx].insert(key, encoding);
    }

    pub fn insert_ek_times_s(&mut self, level: Level, key: AdviceKey, encoding: AR16Encoding<P>) {
        if level.0 < 2 {
            return;
        }
        let idx = (level.0 - 2) as usize;
        if self.lift_s.len() <= idx {
            self.lift_s.resize(idx + 1, BTreeMap::new());
        }
        self.lift_s[idx].insert(key, encoding);
    }

    pub fn ek_s2(&self, level: Level) -> Option<&AR16Encoding<P>> {
        if level.0 < 2 {
            return None;
        }
        let idx = (level.0 - 2) as usize;
        self.lift.get(idx)?.get(&AdviceKey::S2(level))
    }

    pub fn ek_of(&self, level: Level, gate: GateId) -> Option<&AR16Encoding<P>> {
        if level.0 == 1 {
            return self.c1_x.get(&gate);
        }
        if level.0 < 2 {
            return None;
        }
        let idx = (level.0 - 2) as usize;
        self.lift.get(idx)?.get(&AdviceKey::Msg(gate))
    }

    pub fn ek_times_s_of(&self, level: Level, gate: GateId) -> Option<&AR16Encoding<P>> {
        if level.0 < 2 {
            return None;
        }
        let idx = (level.0 - 2) as usize;
        self.lift_s.get(idx)?.get(&AdviceKey::Msg(gate))
    }
}
