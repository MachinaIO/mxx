//! Equality of separately evaluated deterministic public mask samplers.
//!
//! Ordinary randomness is not identified across runs. These keys are actual
//! HashSample nodes with equal keys, complete tags and matrix geometry.
use std::collections::BTreeMap;

use super::root_inputs::value_expression;
use mxx_ir_core::{
    FrozenGraphScopeId, ValidatedGraph,
    lean::{BoundaryValue, LeanArtifact},
    node::{HashVariant, LoopInputMode, NodeKind},
    types::{ConcreteWireType, NodeId, Port, WireRef},
};

struct MaskSampler {
    scope: FrozenGraphScopeId,
    output: WireRef,
    hash: WireRef,
    slice: WireRef,
    key: WireRef,
    count: usize,
}

fn mask_sampler(graph: &ValidatedGraph) -> Result<MaskSampler, String> {
    let root = graph.source.root_scope();
    let mut matches = Vec::new();
    for (node_index, node) in root.nodes().iter().enumerate() {
        let NodeKind::ParallelLoop(loop_) = node.kind() else { continue };
        let owner = NodeId(node_index as u64);
        let id =
            FrozenGraphScopeId::ParallelBody { parent: Box::new(FrozenGraphScopeId::Root), owner };
        let scope = &graph.source.scopes()[&id];
        for (port, output) in scope.outputs().iter().enumerate() {
            let output_node = scope.node(output.node).ok_or("missing mask output")?;
            if !matches!(output_node.kind(), NodeKind::Slice { .. }) {
                continue;
            }
            let hash = scope.arguments(output_node).ok_or("missing mask slice input")?[0];
            let hash_node = scope.node(hash.node).ok_or("missing mask packed source")?;
            let NodeKind::HashSample { variant: HashVariant::Plain, tag_prefix, .. } =
                hash_node.kind()
            else {
                continue
            };
            let domain = b"exponent-lut-native-mask-bank-v1";
            if !tag_prefix.windows(domain.len()).any(|part| part == domain) {
                continue;
            }
            let args = scope.arguments(hash_node).ok_or("missing mask hash key")?;
            if args.len() != 1 {
                return Err("mask hash uses unsupported runtime tag operands".into());
            }
            let key_slot = scope
                .inputs()
                .iter()
                .position(|wire| *wire == args[0])
                .ok_or("mask hash key is not captured")?;
            if loop_.input_modes[key_slot] != LoopInputMode::Broadcast {
                return Err("mask hash key is not broadcast".into());
            }
            let key = root.arguments(node).ok_or("missing mask captures")?[key_slot];
            let output_wire = WireRef { node: owner, port: Port(port as u32) };
            let ConcreteWireType::IndexedFamily { count, .. } =
                graph.scopes[&FrozenGraphScopeId::Root].wire_types[&output_wire]
            else {
                return Err("mask sampler is not family".into());
            };
            matches.push(MaskSampler {
                scope: id.clone(),
                output: output_wire,
                hash,
                slice: *output,
                key,
                count,
            });
        }
    }
    if matches.len() != 1 {
        return Err(format!("expected one actual public mask sampler, found {}", matches.len()));
    }
    Ok(matches.remove(0))
}

fn boundary_type(values: &BTreeMap<String, BoundaryValue>, count: usize) -> String {
    let mut types = vec![String::new(); count];
    for value in values.values() {
        types[value.tuple_index] = format!("({})", value.lean_type);
    }
    match count {
        0 => "Unit".into(),
        1 => types.remove(0),
        _ => types.join(" × ") + " × Unit",
    }
}

pub(super) fn root_seeded_keys(
    pregraph: &ValidatedGraph,
    preartifact: &LeanArtifact,
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
) -> Result<String, String> {
    let pre = mask_sampler(pregraph)?;
    let ver = mask_sampler(graph)?;
    if pre.count != ver.count {
        return Err("mask helper counts differ".into());
    }
    let pre_scope = &pregraph.source.scopes()[&pre.scope];
    let ver_scope = &graph.source.scopes()[&ver.scope];
    // Exact matching here is only an early diagnostic. The generated Lean proof
    // still derives both hash and slice equalities from their actual relations.
    if pre_scope.node(pre.hash.node).ok_or("missing pre hash")?.kind() !=
        ver_scope.node(ver.hash.node).ok_or("missing ver hash")?.kind() ||
        pre_scope.node(pre.slice.node).ok_or("missing pre slice")?.kind() !=
            ver_scope.node(ver.slice.node).ok_or("missing ver slice")?.kind() ||
        pregraph.scopes[&pre.scope].wire_types[&pre.hash] !=
            graph.scopes[&ver.scope].wire_types[&ver.hash]
    {
        return Err("PRE/VER public mask hash requests differ".into());
    }
    let pre_shape = &preartifact.scope_proofs[&FrozenGraphScopeId::Root];
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let pre_value = |wire: WireRef| {
        value_expression(
            &format!("w_{}_{}", wire.node.0, wire.port.0),
            &pre_shape.value_expressions,
        )
        .map(|v| v.replace("witness.", "preWitness.").replace("inputs.", "preInputs."))
    };
    let value = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
    };
    let pre_key = pre_value(pre.key)?;
    let key = value(ver.key)?;
    let pre_mask = pre_value(pre.output)?;
    let mask = value(ver.output)?;
    let pre_record = pre_shape.witness_record.as_ref().ok_or("missing PRE witness")?;
    let record = shape.witness_record.as_ref().ok_or("missing VER witness")?;
    let pre_body = pre_shape.body_relation.as_ref().ok_or("missing PRE body")?;
    let body = shape.body_relation.as_ref().ok_or("missing VER body")?;
    let mut source = format!(
        "import {}\nimport {}\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\n\nnamespace ExponentGeneratedSeededKeys\nopen Mxx.Primitives\nabbrev PreInput := {}\nabbrev PreOutput := {}\nabbrev RootInput := {}\nabbrev RootOutput := {}\n\ntheorem readout_mask_keys {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{preParams : {}}} {{params : {}}}\n    {{preInputs : PreInput}} {{preOutputs : PreOutput}} {{inputs : RootInput}} {{outputs : RootOutput}}\n    (preWitness : {pre_record}) (preFacts : {pre_body} backend hashModel preParams preInputs preOutputs preWitness)\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness)\n    (hashKey : ({key}) = ({pre_key})) :\n    ∀ helper : Fin {}, ({mask}) helper = ({pre_mask}) helper := by\n  intro helper\n",
        preartifact.module_name,
        artifact.module_name,
        boundary_type(&preartifact.root.inputs, preartifact.root.input_count),
        boundary_type(&preartifact.root.outputs, preartifact.root.output_count),
        boundary_type(&artifact.root.inputs, artifact.root.input_count),
        boundary_type(&artifact.root.outputs, artifact.root.output_count),
        preartifact.root.parameter_type,
        artifact.root.parameter_type,
        pre.count
    );
    for (prefix, sample, exported, root_shape, scope, family) in [
        ("pre", &pre, preartifact, pre_shape, pre_scope, &pre_mask),
        ("ver", &ver, artifact, shape, ver_scope, &mask),
    ] {
        let child_shape = &exported.scope_proofs[&sample.scope];
        if child_shape.witness_record.is_some() {
            return Err("mask sampler unexpectedly contains structural loops".into());
        }
        let names = child_shape
            .witnesses
            .iter()
            .map(|(name, _)| format!("{prefix}_{name}"))
            .collect::<Vec<_>>()
            .join(", ");
        let root_pos = root_shape.node_constraints[&sample.output.node]
            .clone()
            .last()
            .ok_or("missing root sampler fact")?;
        source.push_str(&format!(
            "  obtain ⟨{names}, {prefix}Facts⟩ := {}{}.1 helper\n",
            if prefix == "pre" { "preFacts" } else { "facts" },
            ".2".repeat(root_pos)
        ));
        for (role, wire) in [("Hash", sample.hash), ("Slice", sample.slice)] {
            let pos = child_shape.node_constraints[&wire.node]
                .clone()
                .last()
                .ok_or("missing sampler constraint")?;
            source.push_str(&format!(
                "  have {prefix}{role} := {prefix}Facts{}.1\n",
                ".2".repeat(pos)
            ));
        }
        let output_path = if scope.outputs().len() == 1 {
            String::new()
        } else {
            format!("{}.1", ".2".repeat(sample.output.port.0 as usize))
        };
        source.push_str(&format!("  have {prefix}Output : ({family}) helper = {prefix}_w_{}_{} := congrArg (fun output => output{output_path}) ({prefix}Facts{})\n",sample.slice.node.0,sample.slice.port.0,".2".repeat(child_shape.constraint_count)));
    }
    source.push_str(&format!("  have sameHash : ver_w_{}_{} = pre_w_{}_{} := by\n    rw [verHash.2.2, preHash.2.2, hashKey]\n  rw [verOutput, preOutput]\n  ext row column\n  obtain ⟨_, _, _, _, _, _, _, _, preSlice⟩ := preSlice\n  obtain ⟨_, _, _, _, _, _, _, _, verSlice⟩ := verSlice\n  rw [verSlice row column (by omega) (by omega), preSlice row column (by omega) (by omega), sameHash]\n\nend ExponentGeneratedSeededKeys\n",ver.hash.node.0,ver.hash.port.0,pre.hash.node.0,pre.hash.port.0));
    Ok(source)
}
