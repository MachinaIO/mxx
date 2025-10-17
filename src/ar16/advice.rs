use std::collections::BTreeMap;

use crate::{circuit::gate::GateId, poly::Poly};

use super::{AR16Encoding, Level};

/// Index into advice tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AdviceKey {
    /// Advice corresponding to a gate's output message `E_k(c^{k-1}_g)`.
    Msg(GateId),
    /// Advice corresponding to the lifted encoding of `E_{k-1}(c^{k-2}_g · s)`.
    MsgTimes(GateId),
    /// Advice holding `E_k(s^2)` at level `k`.
    S2(Level),
    /// Advice holding `E_k(E_{k-1}(s^2) · s)` at level `k`.
    S2Times(Level),
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

    pub fn ek_of_times(&self, level: Level, gate: GateId) -> Option<&AR16Encoding<P>> {
        if level.0 < 2 {
            return None;
        }
        let idx = (level.0 - 2) as usize;
        self.lift.get(idx)?.get(&AdviceKey::MsgTimes(gate))
    }

    pub fn ek_times_s_of(&self, level: Level, gate: GateId) -> Option<&AR16Encoding<P>> {
        if level.0 < 2 {
            return None;
        }
        let idx = (level.0 - 2) as usize;
        self.lift_s.get(idx)?.get(&AdviceKey::Msg(gate))
    }

    pub fn ek_times_s_of_times(&self, level: Level, gate: GateId) -> Option<&AR16Encoding<P>> {
        if level.0 < 2 {
            return None;
        }
        let idx = (level.0 - 2) as usize;
        self.lift_s.get(idx)?.get(&AdviceKey::MsgTimes(gate))
    }

    pub fn ek_times_s_s2(&self, level: Level) -> Option<&AR16Encoding<P>> {
        if level.0 < 2 {
            return None;
        }
        let idx = (level.0 - 2) as usize;
        self.lift_s.get(idx)?.get(&AdviceKey::S2Times(level))
    }

    /// Populate the level-`k` advice tables using caller-provided lifting closures.
    ///
    /// The `lift` closure must return `E_k(z)` for the supplied `AdviceKey`, while `lift_times`
    /// must return `E_k(z·s)` for the same key. This helper enforces insertion of both linear and
    /// times‑`s` components for every gate required at level `k`, including the `s^2` entry.
    pub fn populate_level_with<F, G>(
        &mut self,
        level: Level,
        gates: &[GateId],
        mut lift: F,
        mut lift_times: G,
    ) where
        F: FnMut(AdviceKey) -> AR16Encoding<P>,
        G: FnMut(AdviceKey) -> AR16Encoding<P>,
    {
        if level.0 < 2 {
            return;
        }
        self.insert_ek(level, AdviceKey::S2(level), lift(AdviceKey::S2(level)));
        self.insert_ek_times_s(
            level,
            AdviceKey::S2Times(level),
            lift_times(AdviceKey::S2Times(level)),
        );
        for &gate in gates {
            self.insert_ek(level, AdviceKey::Msg(gate), lift(AdviceKey::Msg(gate)));
            self.insert_ek_times_s(level, AdviceKey::Msg(gate), lift_times(AdviceKey::Msg(gate)));
            self.insert_ek(level, AdviceKey::MsgTimes(gate), lift(AdviceKey::MsgTimes(gate)));
            self.insert_ek_times_s(
                level,
                AdviceKey::MsgTimes(gate),
                lift_times(AdviceKey::MsgTimes(gate)),
            );
        }
    }
}
