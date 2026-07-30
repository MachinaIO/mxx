use mxx_ir_core::{graph::Graph, types::WireRef};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LivenessSchedule {
    pub last_use: BTreeMap<WireRef, usize>,
    pub outputs: BTreeSet<WireRef>,
}

pub fn analyze(graph: &Graph) -> LivenessSchedule {
    let mut last_use = BTreeMap::new();
    for (position, node) in graph.nodes.iter().enumerate() {
        for argument in &node.args {
            last_use.insert(*argument, position);
        }
    }
    LivenessSchedule { last_use, outputs: graph.outputs.values().copied().collect() }
}
