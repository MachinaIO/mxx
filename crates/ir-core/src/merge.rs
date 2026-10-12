//! Merging structurally identical graphs into one parallel loop.
//!
//! [`merge_instances`] takes graphs that differ only in integer constants and ring moduli (for
//! example one graph per CRT chain of a refresh) and returns one graph whose root runs a
//! [`ParallelLoop`] over the instances: the loop body is the instances' common root, every integer
//! constant (in node payloads, wire types and rings) that differs between instances becomes
//! `IntExpr::Select { selector: LoopIndex(0), branches }`, and the body's types may therefore
//! depend on the loop index (see [`crate::types::MEMBER_INDEX`]). Root inputs that are identical
//! in every instance are passed once. The caller may declare per-instance inputs one family input
//! ([`MergeInterface::family_input`], e.g. instance `k` reading member `k` of a merged producer's
//! family); the other per-instance inputs stay one input per instance (named as in that instance)
//! packed into a family. Every root output is the loop's output family, named as in the first
//! instance; outputs the caller asks for ([`MergeInterface::unpacked`]) are also returned per
//! instance, output `name` of instance `k` as `"{name}@{k}"`. A stage boundary carrying families
//! keeps the merged graph's interface independent of the number of instances.
//!
//! Any other difference (a byte string, a name, a structural field) is a [`MergeError`] naming
//! its location, so the merged graph computes, member by member, exactly what each instance
//! computes.

use crate::{
    Graph, IntExpr, NodeHandle, SubgraphHandle, ValueHandle,
    graph::{FrozenGraphScopeId, GraphOutput, GraphScope, with_new_construction_scope},
    node::{ArtifactInput, LoopInputMode, NodeKind, ParallelLoop},
    ring::{deserialize_with_ring_table, serialize_with_ring_table},
    types::{MEMBER_INDEX, NodeId, Port, WireRef, WireType},
};
use serde_json::Value;
use std::collections::BTreeMap;
use thiserror::Error;

/// Why instances cannot be merged into one graph with a parallel loop over them.
#[derive(Debug, Error)]
pub enum MergeError {
    #[error("no instances to merge")]
    Empty,
    #[error("instances differ in structure at {0}")]
    Structure(String),
    #[error("instances differ in a value that is not an integer constant at {0}")]
    Value(String),
    #[error("instance serialization failed: {0}")]
    Encoding(String),
    #[error("merged graph does not freeze: {0}")]
    Freeze(String),
}

/// How each instance's root input and output names map to the merged graph's.
#[derive(Clone, Debug, Default)]
pub struct MergeNames {
    /// Per instance: original root input name to merged root input name.
    pub inputs: Vec<BTreeMap<String, String>>,
    /// Per instance: original output name to the merged output family's name.
    pub outputs: Vec<BTreeMap<String, String>>,
    /// Per instance: original output name to its unpacked merged output, when unpacked.
    pub unpacked: Vec<BTreeMap<String, String>>,
}

/// The caller's choices for the merged graph's interface.
pub struct MergeInterface<'a> {
    /// Given every instance's input node (all `NodeKind::Input`, not all equal), the name and
    /// artifact of the one family input replacing them, if any: member `k` of that family is
    /// instance `k`'s input. The family's type is the merged input type over the member index.
    pub family_input: &'a dyn Fn(&[&NodeKind]) -> Option<(String, Option<ArtifactInput>)>,
    /// Whether the output `name` (of the first instance) is also returned per instance.
    pub unpacked: &'a dyn Fn(&str) -> bool,
}

/// Merge `instances` (at least one) into one graph named `name`; see the module documentation.
pub fn merge_instances(
    name: &str,
    instances: &[Graph],
    interface: &MergeInterface<'_>,
) -> Result<(Graph, MergeNames), MergeError> {
    let first = instances.first().ok_or(MergeError::Empty)?;
    for instance in instances {
        if instance.parameters() != first.parameters() ||
            instance.real_constants() != first.real_constants()
        {
            return Err(MergeError::Structure("graph parameters".into()));
        }
        if instance.outputs().len() != first.outputs().len() {
            return Err(MergeError::Structure("output count".into()));
        }
        if instance.scopes().keys().ne(first.scopes().keys()) {
            return Err(MergeError::Structure(format!(
                "scopes ({} vs {})",
                first.scopes().len(),
                instance.scopes().len()
            )));
        }
        if !instance.effect_roots().is_empty() {
            return Err(MergeError::Structure("effect roots".into()));
        }
    }
    let merger = Merger { instances };
    let mut names = MergeNames {
        inputs: vec![BTreeMap::new(); instances.len()],
        outputs: vec![BTreeMap::new(); instances.len()],
        unpacked: vec![BTreeMap::new(); instances.len()],
    };
    let input_name = |kind: &NodeKind| match kind {
        NodeKind::Input { name, .. } => name.clone(),
        _ => String::new(),
    };
    let root = FrozenGraphScopeId::Root;
    let root_scopes = merger.scopes(&root)?;
    let count = instances.len();

    // The root inputs: shared when identical, else one per instance packed into a family.
    let input_positions = root_scopes[0]
        .nodes()
        .iter()
        .enumerate()
        .filter(|(_, node)| matches!(node.kind(), NodeKind::Input { .. }))
        .map(|(position, _)| position)
        .collect::<Vec<_>>();
    let mut loop_arguments = Vec::new();
    let mut input_modes = Vec::new();
    // The body's formal input type of every root input.
    let mut body_inputs = Vec::<WireType>::new();
    for &position in &input_positions {
        let kinds =
            root_scopes.iter().map(|scope| scope.nodes()[position].kind()).collect::<Vec<_>>();
        let types = root_scopes
            .iter()
            .map(|scope| scope.nodes()[position].output_types().to_vec())
            .collect::<Vec<_>>();
        let merged_type = merger.merge_types(&types, &format!("root input {position}"))?;
        let [merged_type] = <[WireType; 1]>::try_from(merged_type)
            .map_err(|_| MergeError::Structure(format!("root input {position} arity")))?;
        let family_type = WireType::IndexedFamily {
            element: Box::new(
                merged_type.substitute_loop_index(0, &IntExpr::LoopIndex(MEMBER_INDEX)),
            ),
            count: count.into(),
        };
        body_inputs.push(merged_type);
        if kinds.iter().all(|kind| *kind == kinds[0]) {
            for instance in &mut names.inputs {
                instance.insert(input_name(kinds[0]), input_name(kinds[0]));
            }
            let node = NodeHandle::new(kinds[0].clone(), vec![], types[0].clone());
            loop_arguments.push(node.output(0).expect("input value"));
            input_modes.push(LoopInputMode::Broadcast);
        } else if let Some((family_name, artifact)) = (interface.family_input)(&kinds) {
            for (instance, kind) in names.inputs.iter_mut().zip(&kinds) {
                instance.insert(input_name(kind), family_name.clone());
            }
            let node = NodeHandle::new(
                NodeKind::Input { name: family_name, wire_type: family_type.clone(), artifact },
                vec![],
                vec![family_type],
            );
            loop_arguments.push(node.output(0).expect("family input"));
            input_modes.push(LoopInputMode::Zip);
        } else {
            // An input name several instances use (for one artifact name of different
            // productions) is suffixed by the instance index.
            let members = root_scopes
                .iter()
                .enumerate()
                .map(|(index, scope)| {
                    let node = &scope.nodes()[position];
                    let mut kind = node.kind().clone();
                    if let NodeKind::Input { name, .. } = &mut kind {
                        let original = name.clone();
                        let shared = root_scopes
                            .iter()
                            .filter(|other| input_name(other.nodes()[position].kind()) == *name)
                            .count() >
                            1;
                        if shared {
                            *name = format!("{name}@{index}");
                        }
                        names.inputs[index].insert(original, name.clone());
                    }
                    NodeHandle::new(kind, vec![], node.output_types().to_vec())
                        .output(0)
                        .expect("input value")
                })
                .collect::<Vec<_>>();
            let pack = NodeHandle::new(
                NodeKind::FamilyPack { count: IntExpr::constant(count) },
                members,
                vec![family_type],
            );
            loop_arguments.push(pack.output(0).expect("packed inputs"));
            input_modes.push(LoopInputMode::Zip);
        }
    }

    // The loop body: the instances' common root, with the root inputs as formal inputs.
    // Outputs correspond by wire (their names may name instance-specific artifacts).
    let mut output_wires = Vec::<WireRef>::new();
    for root in first.outputs().values() {
        if !output_wires.contains(&root.value) {
            output_wires.push(root.value);
        }
    }
    let names_of = |instance: &Graph, wire: &WireRef| {
        instance
            .outputs()
            .iter()
            .filter(|(_, root)| root.value == *wire)
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>()
    };
    for instance in instances {
        for wire in &output_wires {
            if names_of(instance, wire).len() != names_of(first, wire).len() {
                return Err(MergeError::Structure(format!("outputs of wire {wire:?}")));
            }
        }
    }
    let body = with_new_construction_scope(|scope| -> Result<_, MergeError> {
        let mut values = BTreeMap::new();
        let mut inputs = Vec::new();
        for (index, (&position, ty)) in input_positions.iter().zip(&body_inputs).enumerate() {
            let value = NodeHandle::new(
                NodeKind::Input {
                    name: format!("instance-input-{index}"),
                    wire_type: ty.clone(),
                    artifact: None,
                },
                vec![],
                vec![ty.clone()],
            )
            .output(0)
            .expect("body input");
            inputs.push(value.clone());
            values.insert(WireRef { node: NodeId(position as u64), port: Port(0) }, value);
        }
        merger.build_nodes(&root, &mut values, 0)?;
        let outputs = output_wires
            .iter()
            .map(|wire| {
                values
                    .get(wire)
                    .cloned()
                    .ok_or_else(|| MergeError::Structure(format!("root output {wire:?}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let types = outputs.iter().map(|value| value.wire_type().clone()).collect::<Vec<_>>();
        let handle = SubgraphHandle::new("instances", scope, inputs, outputs)
            .map_err(|error| MergeError::Freeze(error.to_string()))?;
        Ok((handle, types))
    })?;
    let (body, _) = body;
    // Output types merged from the instances' root types, as the inputs' types are, so that a
    // family a merged consumer reads has the same type as the merged producer's family.
    let body_output_types = output_wires
        .iter()
        .map(|wire| {
            let types = root_scopes
                .iter()
                .map(|scope| {
                    vec![
                        scope.nodes()[wire.node.0 as usize].output_types()[wire.port.0 as usize]
                            .clone(),
                    ]
                })
                .collect::<Vec<_>>();
            let [merged] =
                <[WireType; 1]>::try_from(merger.merge_types(&types, &format!("output {wire:?}"))?)
                    .map_err(|_| MergeError::Structure(format!("output {wire:?} arity")))?;
            Ok(merged)
        })
        .collect::<Result<Vec<_>, MergeError>>()?;
    let family_types = body_output_types
        .iter()
        .map(|ty| WireType::IndexedFamily {
            element: Box::new(ty.substitute_loop_index(0, &IntExpr::LoopIndex(MEMBER_INDEX))),
            count: count.into(),
        })
        .collect::<Vec<_>>();
    let parallel = NodeHandle::parallel_loop(
        body,
        loop_arguments,
        family_types,
        ParallelLoop {
            count: count.into(),
            minimum_count: 0,
            index_slot: 0,
            bindings: vec![],
            input_modes,
        },
    );

    // The loop's output families, named as in the first instance, and the unpacked outputs.
    let mut outputs = BTreeMap::new();
    let mut insert = |name: String, output: GraphOutput| {
        if outputs.insert(name.clone(), output).is_some() {
            return Err(MergeError::Structure(format!("output name {name}")));
        }
        Ok(())
    };
    for (port, wire) in output_wires.iter().enumerate() {
        let family = parallel.output(port as u32).expect("loop output");
        let family_names = names_of(first, wire);
        for family_name in &family_names {
            insert(
                family_name.clone(),
                GraphOutput {
                    value: family.clone(),
                    availability: first.outputs()[family_name].availability.clone(),
                },
            )?;
        }
        for (index, instance) in instances.iter().enumerate() {
            let instance_names = names_of(instance, wire);
            for (name, family_name) in instance_names.iter().zip(&family_names) {
                names.outputs[index].insert(name.clone(), family_name.clone());
                if !(interface.unpacked)(family_name) {
                    continue;
                }
                let member =
                    body_output_types[port].substitute_loop_index(0, &IntExpr::constant(index));
                let value = NodeHandle::new(
                    NodeKind::FamilyGetStatic { index: IntExpr::constant(index) },
                    vec![family.clone()],
                    vec![member],
                )
                .output(0)
                .expect("member value");
                let unique = format!("{family_name}@{index}");
                names.unpacked[index].insert(name.clone(), unique.clone());
                insert(
                    unique,
                    GraphOutput {
                        value,
                        availability: instance.outputs()[name].availability.clone(),
                    },
                )?;
            }
        }
    }
    Graph::freeze(
        name,
        first.parameters().to_vec(),
        outputs,
        vec![],
        vec![],
        first.real_constants().clone(),
    )
    .map(|(graph, _)| (graph, names))
    .map_err(|error| MergeError::Freeze(error.to_string()))
}

struct Merger<'a> {
    instances: &'a [Graph],
}

impl Merger<'_> {
    fn scopes(&self, scope_id: &FrozenGraphScopeId) -> Result<Vec<&GraphScope>, MergeError> {
        let scopes = self
            .instances
            .iter()
            .map(|instance| {
                instance
                    .scope(scope_id)
                    .ok_or_else(|| MergeError::Structure(format!("missing scope {scope_id:?}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for scope in &scopes {
            // Root outputs are ordered by name, which may be instance-specific.
            let sorted = |scope: &GraphScope| {
                let mut outputs = scope.outputs().to_vec();
                if matches!(scope_id, FrozenGraphScopeId::Root) {
                    outputs.sort();
                }
                outputs
            };
            if scope.nodes().len() != scopes[0].nodes().len() ||
                scope.inputs() != scopes[0].inputs() ||
                sorted(scope) != sorted(scopes[0])
            {
                return Err(MergeError::Structure(format!(
                    "scope {scope_id:?}: {} vs {} nodes, {} vs {} inputs, {} vs {} outputs",
                    scopes[0].nodes().len(),
                    scope.nodes().len(),
                    scopes[0].inputs().len(),
                    scope.inputs().len(),
                    scopes[0].outputs().len(),
                    scope.outputs().len()
                )));
            }
        }
        Ok(scopes)
    }

    /// Rebuild the nodes of `scope_id` (whose inputs are already in `values`), loop slots shifted
    /// by `shift + 1` (the merged loop takes slot zero), into the current construction scope.
    fn build_nodes(
        &self,
        scope_id: &FrozenGraphScopeId,
        values: &mut BTreeMap<WireRef, ValueHandle>,
        depth: u32,
    ) -> Result<(), MergeError> {
        let scopes = self.scopes(scope_id)?;
        for position in 0..scopes[0].nodes().len() {
            let wire = |port: u32| WireRef { node: NodeId(position as u64), port: Port(port) };
            if values.contains_key(&wire(0)) {
                continue;
            }
            let location = format!("{scope_id:?} node {position}");
            let nodes = scopes.iter().map(|scope| scopes_node(scope, position)).collect::<Vec<_>>();
            if matches!(nodes[0].kind(), NodeKind::Input { .. }) {
                // A formal input of a child scope is bound by its caller.
                continue;
            }
            // An argument that is a different wire in some instance (for example another row
            // of one value) is a selection by the loop index among every instance's wire.
            let instance_arguments = scopes
                .iter()
                .zip(&nodes)
                .map(|(scope, node)| {
                    scope.arguments(node).ok_or_else(|| MergeError::Structure(location.clone()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if instance_arguments
                .iter()
                .any(|arguments| arguments.len() != instance_arguments[0].len())
            {
                return Err(MergeError::Structure(format!("{location} argument count")));
            }
            let value = |wire: &WireRef| {
                values
                    .get(wire)
                    .cloned()
                    .ok_or_else(|| MergeError::Structure(format!("{location} argument order")))
            };
            let mut arguments = Vec::new();
            for position in 0..instance_arguments[0].len() {
                let wires = instance_arguments
                    .iter()
                    .map(|arguments| arguments[position])
                    .collect::<Vec<_>>();
                if wires.iter().all(|wire| *wire == wires[0]) {
                    arguments.push(value(&wires[0])?);
                    continue;
                }
                let branches = wires.iter().map(value).collect::<Result<Vec<_>, _>>()?;
                if branches.iter().any(|branch| branch.wire_type() != branches[0].wire_type()) {
                    return Err(MergeError::Structure(format!(
                        "{location} selected argument types"
                    )));
                }
                let selector = NodeHandle::new(
                    NodeKind::EvaluateInt(IntExpr::LoopIndex(0)),
                    vec![],
                    vec![WireType::ConstantInt],
                )
                .output(0)
                .expect("instance index");
                let selected_type = branches[0].wire_type().clone();
                let select = NodeHandle::new(
                    NodeKind::Select { count: IntExpr::constant(branches.len()) },
                    std::iter::once(selector).chain(branches).collect(),
                    vec![selected_type],
                );
                arguments.push(select.output(0).expect("selected argument"));
            }
            let kind = self.merge_kinds(
                &nodes.iter().map(|node| node.kind()).collect::<Vec<_>>(),
                &location,
            )?;
            let types = self.merge_types(
                &nodes.iter().map(|node| node.output_types().to_vec()).collect::<Vec<_>>(),
                &location,
            )?;
            let node = match &kind {
                NodeKind::ParallelLoop(spec) => {
                    let body = self.build_child(scope_id, position, depth)?;
                    NodeHandle::parallel_loop(body, arguments, types, spec.clone())
                }
                NodeKind::SequentialLoop(spec) => {
                    let body = self.build_child(scope_id, position, depth)?;
                    NodeHandle::sequential_loop(body, arguments, types, spec.clone())
                }
                NodeKind::SubgraphCall(_) => {
                    return Err(MergeError::Structure(format!("{location} named subgraph call")));
                }
                _ => NodeHandle::new(kind, arguments, types),
            };
            for port in 0..node.output_types().len() {
                values.insert(wire(port as u32), node.output(port as u32).expect("node output"));
            }
        }
        Ok(())
    }

    /// The merged body of the loop at `position` of `scope_id`.
    fn build_child(
        &self,
        scope_id: &FrozenGraphScopeId,
        position: usize,
        depth: u32,
    ) -> Result<SubgraphHandle, MergeError> {
        let child = self.instances[0]
            .child_scope_id(scope_id, NodeId(position as u64))
            .ok_or_else(|| MergeError::Structure(format!("{scope_id:?} node {position} child")))?;
        let scopes = self.scopes(&child)?;
        with_new_construction_scope(|construction| {
            let mut values = BTreeMap::new();
            let mut inputs = Vec::new();
            for input in scopes[0].inputs() {
                let location = format!("{child:?} input {input:?}");
                let nodes = scopes
                    .iter()
                    .map(|scope| scopes_node(scope, input.node.0 as usize))
                    .collect::<Vec<_>>();
                // The caller binds a formal input; its artifact reference (one per instance) is
                // provenance only and is dropped.
                let kinds = nodes
                    .iter()
                    .map(|node| match node.kind() {
                        NodeKind::Input { name, wire_type, .. } => NodeKind::Input {
                            name: name.clone(),
                            wire_type: wire_type.clone(),
                            artifact: None,
                        },
                        kind => kind.clone(),
                    })
                    .collect::<Vec<_>>();
                let kind = self.merge_kinds(&kinds.iter().collect::<Vec<_>>(), &location)?;
                let types = self.merge_types(
                    &nodes.iter().map(|node| node.output_types().to_vec()).collect::<Vec<_>>(),
                    &location,
                )?;
                let node = NodeHandle::new(kind, vec![], types);
                let value = node.output(input.port.0).expect("child input");
                values.insert(*input, value.clone());
                inputs.push(value);
            }
            self.build_nodes(&child, &mut values, depth + 1)?;
            let outputs = scopes[0]
                .outputs()
                .iter()
                .map(|output| {
                    values.get(output).cloned().ok_or_else(|| {
                        MergeError::Structure(format!("{child:?} output {output:?}"))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            SubgraphHandle::new("body", construction, inputs, outputs)
                .map_err(|error| MergeError::Freeze(error.to_string()))
        })
    }

    fn merge_kinds(&self, kinds: &[&NodeKind], location: &str) -> Result<NodeKind, MergeError> {
        let encoded = kinds
            .iter()
            .map(|kind| {
                serialize_with_ring_table(kind)
                    .map_err(|error| MergeError::Encoding(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (value, table) = merge_encoded(encoded, location)?;
        deserialize_with_ring_table(value, table)
            .map_err(|error| MergeError::Encoding(error.to_string()))
    }

    fn merge_types(
        &self,
        types: &[Vec<WireType>],
        location: &str,
    ) -> Result<Vec<WireType>, MergeError> {
        let encoded = types
            .iter()
            .map(|ty| {
                serialize_with_ring_table(ty)
                    .map_err(|error| MergeError::Encoding(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (value, table) = merge_encoded(encoded, &format!("{location} types"))?;
        deserialize_with_ring_table(value, table)
            .map_err(|error| MergeError::Encoding(error.to_string()))
    }
}

fn scopes_node(scope: &GraphScope, position: usize) -> NodeHandle {
    scope.nodes()[position].clone()
}

/// Shift loop slots by one, then merge the values and their ring tables.
fn merge_encoded(
    encoded: Vec<(Value, Vec<Value>)>,
    location: &str,
) -> Result<(Value, Vec<Value>), MergeError> {
    let shifted = encoded
        .into_iter()
        .map(|(value, table)| {
            (shift_slots(value), table.into_iter().map(shift_slots).collect::<Vec<_>>())
        })
        .collect::<Vec<_>>();
    let values = shifted.iter().map(|(value, _)| value).collect::<Vec<_>>();
    let value = merge_values(&values, location)?;
    let tables = shifted.iter().map(|(_, table)| table).collect::<Vec<_>>();
    if tables.iter().any(|table| table.len() != tables[0].len()) {
        return Err(MergeError::Structure(format!("{location} rings")));
    }
    let table = (0..tables[0].len())
        .map(|index| {
            merge_values(
                &tables.iter().map(|table| &table[index]).collect::<Vec<_>>(),
                &format!("{location} ring {index}"),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((value, table))
}

/// Every loop slot one deeper: the merged loop is slot zero. The member index is not a slot.
fn shift_slots(value: Value) -> Value {
    match value {
        Value::Object(mut fields) => {
            let is_loop_index = fields.get("tag").and_then(Value::as_str) == Some("LoopIndex");
            for (key, field) in fields.iter_mut() {
                let shift = (is_loop_index && key == "value") || key == "index_slot";
                *field = match (shift, field.as_u64()) {
                    (true, Some(slot)) if slot != u64::from(MEMBER_INDEX) => Value::from(slot + 1),
                    _ => shift_slots(field.take()),
                };
            }
            Value::Object(fields)
        }
        Value::Array(values) => Value::Array(values.into_iter().map(shift_slots).collect()),
        other => other,
    }
}

/// Equal values stay; integer constants that differ become a selection by the merged loop's
/// index; anything else must agree.
fn merge_values(values: &[&Value], location: &str) -> Result<Value, MergeError> {
    let first = values[0];
    if values.iter().all(|value| *value == first) {
        return Ok(first.clone());
    }
    let constant = |value: &Value| {
        value.get("tag").and_then(Value::as_str) == Some("Const") && value.get("value").is_some()
    };
    if values.iter().all(|value| constant(value)) {
        return Ok(serde_json::json!({
            "tag": "Select",
            "value": {
                "selector": { "tag": "LoopIndex", "value": 0 },
                "branches": values.iter().map(|value| (*value).clone()).collect::<Vec<_>>(),
            }
        }));
    }
    match first {
        Value::Object(fields) => {
            let mut merged = serde_json::Map::new();
            for (key, field) in fields {
                let members = values
                    .iter()
                    .map(|value| {
                        value
                            .get(key)
                            .filter(|_| {
                                value.as_object().is_some_and(|object| object.len() == fields.len())
                            })
                            .ok_or_else(|| MergeError::Structure(format!("{location}.{key}")))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let _ = field;
                merged.insert(key.clone(), merge_values(&members, &format!("{location}.{key}"))?);
            }
            Ok(Value::Object(merged))
        }
        Value::Array(items) => {
            let members = values
                .iter()
                .map(|value| {
                    value
                        .as_array()
                        .filter(|array| array.len() == items.len())
                        .ok_or_else(|| MergeError::Structure(format!("{location}[]")))
                })
                .collect::<Result<Vec<_>, _>>()?;
            (0..items.len())
                .map(|index| {
                    merge_values(
                        &members.iter().map(|array| &array[index]).collect::<Vec<_>>(),
                        &format!("{location}[{index}]"),
                    )
                })
                .collect::<Result<Vec<_>, _>>()
                .map(Value::Array)
        }
        _ => Err(MergeError::Value(format!(
            "{location}: {}",
            values.iter().map(|value| value.to_string()).collect::<Vec<_>>().join(" | ")
        ))),
    }
}
