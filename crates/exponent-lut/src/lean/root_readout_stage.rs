//! Complete readout execution expression from the fixed verification witness.
use super::root_inputs::value_expression;
use mxx_ir_core::{
    FrozenGraphScopeId, IntExpr, ParamEnv, ValidatedGraph,
    graph::GraphScope,
    lean::LeanArtifact,
    node::{LoopInputMode, MatrixBinaryOp, NodeKind},
    types::{ConcreteWireType, NodeId, Port, WireRef},
};

pub(super) fn root_readout_stage(
    pregraph: &ValidatedGraph,
    preartifact: &LeanArtifact,
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
) -> Result<String, String> {
    let root = graph.source.root_scope();
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let types = &graph.scopes[&FrozenGraphScopeId::Root].wire_types;
    let body = shape.body_relation.as_ref().ok_or("missing readout root body")?;
    let record = shape.witness_record.as_ref().ok_or("missing readout root witness")?;
    let child_id = |wire: WireRef| FrozenGraphScopeId::ParallelBody {
        parent: Box::new(FrozenGraphScopeId::Root),
        owner: wire.node,
    };
    let child =
        |wire: WireRef| -> Option<&GraphScope> { graph.source.scopes().get(&child_id(wire)) };
    let gather = |wire: WireRef| -> Option<(WireRef, WireRef)> {
        let scope = child(wire)?;
        let output = scope.node(scope.outputs()[0].node)?;
        if !matches!(output.kind(), NodeKind::FamilyGetDynamic) {
            return None;
        }
        let args = scope.arguments(output)?;
        let parents = root.arguments(root.node(wire.node)?)?;
        let map = |arg: WireRef| {
            scope.inputs().iter().position(|input| *input == arg).map(|i| parents[i])
        };
        Some((map(args[0])?, map(args[1])?))
    };
    let value = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
    };
    let fact = |wire: WireRef| -> Result<String, String> {
        let position = shape
            .node_constraints
            .get(&wire.node)
            .and_then(|r| r.clone().last())
            .ok_or("missing readout constraint")?;
        Ok(format!("facts{}.1", ".2".repeat(position)))
    };
    let normalization = artifact
        .root
        .inputs
        .iter()
        .find(|(name, _)| name.ends_with("-readout-normalization-decomposition-0"))
        .ok_or("missing readout normalization input")?
        .1
        .wire;
    let (normalization_node, normalization_args) = root
        .nodes()
        .iter()
        .enumerate()
        .find_map(|(index, node)| {
            let args = root.arguments(node)?;
            (matches!(node.kind(), NodeKind::ParallelLoop(_)) && args.contains(&normalization))
                .then_some((index, args))
        })
        .ok_or("missing normalization operation")?;
    let normalized_wire = WireRef { node: NodeId(normalization_node as u64), port: Port(0) };
    let traced = *normalization_args
        .iter()
        .find(|wire| **wire != normalization)
        .ok_or("missing trace input")?;
    let traced_node = root.node(traced.node).ok_or("missing traced family")?;
    let traced_args = root.arguments(traced_node).ok_or("missing traced operands")?;
    if traced_args.len() != 2 {
        return Err("trace must add identity and helpers".into());
    }
    let ConcreteWireType::IndexedFamily { count: labels, .. } = &types[&traced] else {
        return Err("trace is not family".into());
    };
    let transformed = traced_args[0];
    let mut current = traced_args[1];
    let mut levels = Vec::new();
    loop {
        let Some(scope) = child(current) else {
            break;
        };
        let Some(parents) = root.arguments(root.node(current.node).ok_or("missing reducer node")?)
        else {
            break;
        };
        if parents.len() != 2 || scope.outputs().len() != 1 {
            break;
        }
        let (Some((left_source, left_indices)), Some((right_source, right_indices))) =
            (gather(parents[0]), gather(parents[1]))
        else {
            break;
        };
        if left_source != right_source {
            break;
        }
        let ConcreteWireType::IndexedFamily { count, .. } = &types[&left_source] else {
            break;
        };
        let ConcreteWireType::IndexedFamily { count: next, .. } = &types[&current] else {
            break;
        };
        if *labels == 0 ||
            count % labels != 0 ||
            next % labels != 0 ||
            next / labels != (count / labels).div_ceil(2)
        {
            return Err("non-segmented readout reducer".into());
        }
        levels.push((
            current,
            parents[0],
            parents[1],
            left_indices,
            right_indices,
            left_source,
            count / labels,
            next / labels,
        ));
        current = left_source;
    }
    if levels.is_empty() {
        return Err("missing balanced readout helper sum".into());
    }
    let ConcreteWireType::IndexedFamily { count: total, .. } = &types[&current] else {
        return Err("invalid helper family".into());
    };
    let helpers = total / labels;
    let mut input_types = vec![String::new(); artifact.root.input_count];
    for input in artifact.root.inputs.values() {
        input_types[input.tuple_index] = format!("({})", input.lean_type);
    }
    let input_type = input_types.join(" × ") + " × Unit";
    let mut outputs = vec![String::new(); artifact.root.output_count];
    for output in artifact.root.outputs.values() {
        outputs[output.tuple_index] = format!("({})", output.lean_type);
    }
    let output_type = outputs.join(" × ") + " × Unit";
    let normalized = value(normalized_wire)?;
    let transformed_wire = transformed;
    let transformed = value(transformed)?;
    let original = value(current)?;
    let norm = value(normalization)?;
    let mut source = format!(
        "import {}\nimport ExponentReadoutStage\nimport RootOnlineReadout\nimport RootReadout\nimport RootReadoutOperands\nimport RootReadoutFunction\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 4000000\n\nnamespace ExponentGeneratedReadoutStage\nopen Mxx.Primitives MxxExponentLut\nopen scoped BigOperators\nabbrev RootInput := {input_type}\nabbrev RootOutput := {output_type}\n\nprivate theorem lookup_eq {{α : Type}} {{count : Nat}} {{family : Fin count → α}} {{index : Int}} {{output : α}} (position : Fin count) (hindex : index = (position.val : Int)) (h : MxxRuntime.familyGetDynamic family index output) : output = family position := by\n  obtain ⟨actual, ha, hv⟩ := h\n  have same : actual = position := by apply Fin.ext; omega\n  simpa only [same] using hv\n\ntheorem normalized_execution {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {params}}} {{inputs : RootInput}} {{outputs : RootOutput}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness) :\n    ∀ label : Fin {labels}, ({normalized}) label =\n      (({transformed}) label + ∑ helper : Fin {helpers},\n        ({original}) ⟨label.val * {helpers} + helper.val, by omega⟩) * ({norm}) := by\n  intro label\n",
        artifact.module_name,
        params = artifact.root.parameter_type
    );
    let mut equations = Vec::new();
    for (result, left, right, left_indices, right_indices, previous, width, next_width) in
        levels.iter().rev()
    {
        let count = labels * width;
        let next = labels * next_width;
        for lane in 0..*next_width {
            for (side, gather_wire, indices, position) in [
                ("l", *left, *left_indices, lane * 2),
                ("r", *right, *right_indices, (lane * 2 + 1) % width),
            ] {
                let ir = &artifact.scope_relations[&child_id(indices)];
                let gr = &artifact.scope_relations[&child_id(gather_wire)];
                let index_value = value(indices)?;
                let previous_value = value(*previous)?;
                let gather_value = value(gather_wire)?;
                let ipath = fact(indices)?;
                let gpath = fact(gather_wire)?;
                let id = result.node.0;
                source.push_str(&format!("  have idx_{id}_{lane}_{side} : ({index_value}) ⟨label.val * {next_width} + {lane}, by omega⟩ = ((label.val * {width} + {position} : Nat) : Int) := by\n    have h := {ipath} (⟨label.val * {next_width} + {lane}, by omega⟩ : Fin {next})\n    dsimp only [{ir}, {ir}.constraints_0] at h\n    simp only [Int.ofNat_eq_natCast, Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat] at h ⊢\n    omega\n  have get_{id}_{lane}_{side} : ({gather_value}) ⟨label.val * {next_width} + {lane}, by omega⟩ = ({previous_value}) ⟨label.val * {width} + {position}, by omega⟩ := by\n    have h := {gpath} (⟨label.val * {next_width} + {lane}, by omega⟩ : Fin {next})\n    dsimp only [{gr}, {gr}.constraints_0] at h\n    obtain ⟨sample, hlo, hhi, hg, hout⟩ := h\n    exact hout.trans (lookup_eq (⟨label.val * {width} + {position}, by omega⟩ : Fin {count}) idx_{id}_{lane}_{side} hg)\n"));
            }
            if lane == 0 || (2 * lane + 1) % width == 0 {
                source.push_str(&format!(
                    "  simp only [Nat.add_zero] at get_{id}_{lane}_l get_{id}_{lane}_r\n",
                    id = result.node.0
                ));
            }
            let result_value = value(*result)?;
            let previous_value = value(*previous)?;
            let rpath = fact(*result)?;
            let rr = &artifact.scope_relations[&child_id(*result)];
            let id = result.node.0;
            let left_index = lane * 2;
            let right_index = lane * 2 + 1;
            let rhs = if right_index < *width {
                format!(
                    "({previous_value}) ⟨label.val * {width} + {left_index}, by omega⟩ + ({previous_value}) ⟨label.val * {width} + {right_index}, by omega⟩"
                )
            } else {
                format!("({previous_value}) ⟨label.val * {width} + {left_index}, by omega⟩")
            };
            let eq = format!("step_{id}_{lane}");
            source.push_str(&format!("  have {eq} : ({result_value}) ⟨label.val * {next_width} + {lane}, by omega⟩ = {rhs} := by\n    have h := {rpath} (⟨label.val * {next_width} + {lane}, by omega⟩ : Fin {next})\n"));
            let is_select = matches!(
                child(*result)
                    .and_then(|scope| scope.node(scope.outputs()[0].node))
                    .map(|node| node.kind()),
                Some(NodeKind::Select { .. })
            );
            if is_select {
                source.push_str(&format!("    have modEq : ((label.val * {next_width} + {lane} : Nat) : Int) % {next_width} = {lane} := by\n      simp only [Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat]\n      omega\n    norm_num [{rr}, {rr}.constraints_0, modEq, MxxRuntime.select] at h\n"));
                if right_index < *width {
                    source.push_str(&format!("    simpa only [MxxRuntime.matrixAdd, get_{id}_{lane}_l, get_{id}_{lane}_r, Nat.add_zero] using h\n"));
                } else {
                    source.push_str(&format!("    obtain ⟨position, hp, hv⟩ := h\n    have same : position = (1 : Fin 2) := by apply Fin.ext; exact_mod_cast hp\n    subst position\n    simpa only [List.get_eq_getElem, List.getElem_cons_succ, List.getElem_cons_zero, get_{id}_{lane}_l, Nat.add_zero] using hv\n"));
                }
            } else {
                source.push_str(&format!("    simpa only [{rr}, {rr}.constraints_0, MxxRuntime.matrixAdd, get_{id}_{lane}_l, get_{id}_{lane}_r, Nat.add_zero] using h\n"));
            }
            equations.push(eq);
        }
    }
    source.push_str(&format!("  have normalizedEquation := {} label\n  have tracedEquation := {} label\n  rw [normalizedEquation, tracedEquation]\n  simp only [MxxRuntime.matrixMul, MxxRuntime.matrixAdd]\n  congr 1\n  congr 1\n  have labelEq : label = (⟨label.val * 1 + 0, by omega⟩ : Fin {labels}) := by apply Fin.ext; simp\n  conv_lhs => rw [labelEq]\n  simp only [{}]\n  simp [Fin.sum_univ_succ, add_assoc]\n\nend ExponentGeneratedReadoutStage\n",fact(normalized_wire)?,fact(traced)?,equations.iter().rev().cloned().collect::<Vec<_>>().join(", ")));
    let raw_scope = child(current).ok_or("missing raw readout branch")?;
    let raw_node = root.node(current.node).ok_or("missing raw loop")?;
    let raw_parents = root.arguments(raw_node).ok_or("missing raw loop operands")?;
    let auto_node = raw_scope
        .nodes()
        .iter()
        .find(|node| matches!(node.kind(), NodeKind::RingAutomorphism { .. }))
        .ok_or("missing actual readout automorphism")?;
    let NodeKind::RingAutomorphism { index: IntExpr::Select { branches, .. } } = auto_node.kind()
    else {
        return Err("readout automorphism does not use actual Select".into());
    };
    let indices = branches
        .iter()
        .map(|branch| {
            branch
                .evaluate(&ParamEnv::default())
                .map(|index| format!("({index})"))
                .map_err(|_| "nonconstant online index")
        })
        .collect::<Result<Vec<_>, _>>()?
        .join(", ");
    let parent = |wire: WireRef| {
        raw_scope
            .inputs()
            .iter()
            .position(|input| *input == wire)
            .map(|slot| raw_parents[slot])
            .ok_or("raw branch input not captured")
    };
    let auto_input = raw_scope.arguments(auto_node).ok_or("missing raw auto input")?[0];
    let transformed_gather = parent(auto_input)?;
    let add = raw_scope.node(raw_scope.outputs()[0].node).ok_or("missing branch add")?;
    let add_args = raw_scope.arguments(add).ok_or("missing branch add operands")?;
    let aligned_wire = parent(add_args[0])?;
    let product = raw_scope.node(add_args[1].node).ok_or("missing branch product")?;
    let product_args = raw_scope.arguments(product).ok_or("missing branch product operands")?;
    let switch_gather = parent(product_args[1])?;
    let aligned_args = root
        .arguments(root.node(aligned_wire.node).ok_or("missing aligned loop")?)
        .ok_or("missing aligned inputs")?;
    let mask_gather = aligned_args[0];
    let helper_cache = aligned_args[1];
    let mut proof = String::new();
    let mut gathered_sources = Vec::new();
    for (role, wire, position, count) in [
        ("target", transformed_gather, "label", *labels),
        ("mask", mask_gather, "helper", helpers),
        ("switch", switch_gather, "helper", helpers),
    ] {
        let (source_wire, index_wire) = gather(wire).ok_or("readout branch input must gather")?;
        if role == "target" && source_wire != transformed_wire {
            return Err("raw source differs from identity source".into());
        }
        let source_value = value(source_wire)?;
        let gather_value = value(wire)?;
        let index_value = value(index_wire)?;
        let ir = &artifact.scope_relations[&child_id(index_wire)];
        let ip = fact(index_wire)?;
        let gather_id = wire.node.0;
        proof.push_str(&format!("  have index_{role} : ({index_value}) lane = ({position}.val : Int) := by\n    have h := {ip} lane\n    dsimp only [{ir}, {ir}.constraints_0, lane] at h ⊢\n    simp only [Int.ofNat_eq_natCast, Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat] at h\n    omega\n  obtain ⟨pos_{role}, hp_{role}, hv_{role}⟩ := ExponentGeneratedOnlineReadout.family_gather_{gather_id} witness facts lane\n  have same_{role} : pos_{role} = ({position} : Fin {count}) := by apply Fin.ext; omega\n  have gathered_{role} : ({gather_value}) lane = ({source_value}) {position} := by simpa only [same_{role}] using hv_{role}\n"));
        gathered_sources.push(source_value);
    }
    let mask_actual = &gathered_sources[1];
    let switch_actual = &gathered_sources[2];
    let helper_actual = value(helper_cache)?;
    let matrix = types[&current]
        .matrix_type()
        .or_else(|| match &types[&current] {
            ConcreteWireType::IndexedFamily { element, .. } => element.matrix_type(),
            _ => None,
        })
        .ok_or("invalid readout branch type")?;
    let q = &matrix.modulus;
    let n = matrix.ring_dimension;
    let rows = matrix.rows;
    let columns = matrix.columns;
    source = source.replace("end ExponentGeneratedReadoutStage\n", "");
    source.push_str(&format!("theorem helper_execution {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {params}}} {{inputs : RootInput}} {{outputs : RootOutput}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness) :\n    ∀ label : Fin {labels}, ∀ helper : Fin {helpers}, ∃ raw : ExactMatrix {q} {n} {rows} {columns},\n      MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt (Int.ofNat helper.val) [{indices}])\n        (({transformed}) label) raw ∧\n      ({original}) ⟨label.val * {helpers} + helper.val, by omega⟩ =\n        ({mask_actual}) helper * ({helper_actual}) ⟨label.val * {helpers} + helper.val, by omega⟩ +\n          raw * ({switch_actual}) helper := by\n  intro label helper\n  let lane : Fin {total} := ⟨label.val * {helpers} + helper.val, by omega⟩\n{proof}  obtain ⟨raw, hrun, heq⟩ := ExponentGeneratedOnlineReadout.raw_branch_{raw_id} witness facts lane\n  have hmod : Int.fmod (Int.ofNat lane.val) {helpers} = Int.ofNat helper.val := by\n    rw [Int.fmod_eq_emod_of_nonneg _ (by decide)]\n    dsimp only [lane]\n    simp only [Int.ofNat_eq_natCast, Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat]\n    omega\n  refine ⟨raw, ?_, ?_⟩\n  · simpa only [hmod, gathered_target] using hrun\n  · have haligned := ExponentGeneratedOnlineReadout.family_operation_{aligned_id} witness facts lane\n    simpa only [haligned, gathered_mask, gathered_switch] using heq\n\nend ExponentGeneratedReadoutStage\n",
        params = artifact.root.parameter_type, raw_id = current.node.0, aligned_id = aligned_wire.node.0));
    let preroot = pregraph.source.root_scope();
    let preshape = &preartifact.scope_proofs[&FrozenGraphScopeId::Root];
    let prevalue = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &preshape.value_expressions)
            .map(|text| text.replace("witness.", "preWitness.").replace("inputs.", "preInputs."))
    };
    let preboundary = |suffix: &str| {
        preartifact
            .root
            .outputs
            .iter()
            .find(|(name, _)| name.ends_with(suffix))
            .map(|(_, value)| value.wire)
            .ok_or_else(|| format!("missing pre boundary {suffix}"))
    };
    let target_cache_wire = preboundary("-readout-target-decomposition-0")?;
    let normalization_cache_wire = preboundary("-readout-normalization-decomposition-0")?;
    let switch_cache_wire = preboundary("-readout-helper-switch-family-0")?;
    let helper_cache_wire = preboundary("-readout-helper-public-family-0")?;
    let pretarget = prevalue(target_cache_wire)?;
    let prenorm = prevalue(normalization_cache_wire)?;
    let preswitch = prevalue(switch_cache_wire)?;
    let prehelper = prevalue(helper_cache_wire)?;
    let pre_helper_node = preroot.node(helper_cache_wire.node).ok_or("missing pre helper")?;
    let pre_helper_scope = &pregraph.source.scopes()[&child_id(helper_cache_wire)];
    let NodeKind::GadgetDecompose { base, digit_count, .. } = pre_helper_scope
        .node(pre_helper_scope.outputs()[0].node)
        .ok_or("missing pre decomp")?
        .kind()
    else {
        return Err("missing pre decomp".into());
    };
    let base = base.evaluate(&ParamEnv::default()).map_err(|_| "nonconstant readout base")?;
    let digits =
        digit_count.evaluate(&ParamEnv::default()).map_err(|_| "nonconstant readout digits")?;
    let pre_helper_args = preroot.arguments(pre_helper_node).ok_or("missing pre helper args")?;
    let pre_target_gather_args = preroot
        .arguments(preroot.node(pre_helper_args[0].node).ok_or("missing pre gather")?)
        .ok_or("missing pre gathered inputs")?;
    let pre_transformed = pre_target_gather_args
        .iter()
        .copied()
        .find(|wire| {
            matches!(&pregraph.scopes[&FrozenGraphScopeId::Root].wire_types[wire],
            ConcreteWireType::IndexedFamily { element, .. } if element.matrix_type().is_some())
        })
        .ok_or("missing pre transformed family")?;
    let pre_transformed_args = preroot
        .arguments(preroot.node(pre_transformed.node).ok_or("missing pre transformed")?)
        .ok_or("missing pre transformed inputs")?;
    let prepublic = prevalue(pre_transformed_args[0])?;
    let switch_node = preroot.node(switch_cache_wire.node).ok_or("missing switch cache")?;
    let sampled_wire = preroot.arguments(switch_node).ok_or("missing switch sampler")?[0];
    let sampled_scope = &pregraph.source.scopes()[&child_id(sampled_wire)];
    let sampled_auto = sampled_scope
        .nodes()
        .iter()
        .find(|node| matches!(node.kind(), NodeKind::RingAutomorphism { .. }))
        .ok_or("missing switch source automorphism")?;
    let sampled_input =
        sampled_scope.arguments(sampled_auto).ok_or("missing sampled auto input")?[0];
    let sampled_slot = sampled_scope
        .inputs()
        .iter()
        .position(|wire| *wire == sampled_input)
        .ok_or("switch setup not captured")?;
    let sampled_args = preroot
        .arguments(preroot.node(sampled_wire.node).ok_or("missing sampled loop")?)
        .ok_or("missing sampled captures")?;
    let setup = prevalue(sampled_args[sampled_slot])?;
    let NodeKind::RingAutomorphism { index: IntExpr::Select { branches: full_branches, .. } } =
        sampled_auto.kind()
    else {
        return Err("switch source lacks Select table".into());
    };
    let full_indices = full_branches
        .iter()
        .map(|branch| {
            branch
                .evaluate(&ParamEnv::default())
                .map(|index| format!("({index})"))
                .map_err(|_| "nonconstant switch index")
        })
        .collect::<Result<Vec<_>, _>>()?
        .join(", ");
    let (mask_source_wire, _) = gather(mask_gather).ok_or("missing mask source")?;
    let mask_scope = child(mask_source_wire).ok_or("missing mask sampling scope")?;
    let mask_args = root
        .arguments(root.node(mask_source_wire.node).ok_or("missing mask loop")?)
        .ok_or("missing mask captures")?;
    let node_args = |wire: WireRef| {
        mask_scope
            .node(wire.node)
            .and_then(|node| mask_scope.arguments(node))
            .ok_or("missing mask expression operands")
    };
    let selected = mask_scope.outputs()[mask_source_wire.port.0 as usize];
    let batch = node_args(node_args(selected)?[0])?;
    let ideal = node_args(batch[0])?;
    let public_product = node_args(ideal[0])?;
    let mask_slot = mask_scope
        .inputs()
        .iter()
        .position(|wire| *wire == public_product[0])
        .ok_or("actual mask encoding secret not captured")?;
    let secret = value(mask_args[mask_slot])?;
    let public_bank = node_args(public_product[1])?;
    let key_port = mask_scope
        .outputs()
        .iter()
        .position(|wire| *wire == public_bank[1])
        .ok_or("mask key not exported")?;
    let mask_key = value(WireRef { node: mask_source_wire.node, port: Port(key_port as u32) })?;
    let gaussian_node = mask_scope.node(batch[1].node).ok_or("missing mask Gaussian")?;
    let NodeKind::GaussianSample { max_coefficient_bound, .. } = gaussian_node.kind() else {
        return Err("mask error not Gaussian".into());
    };
    let cutoff =
        max_coefficient_bound.evaluate(&ParamEnv::default()).map_err(|_| "nonconstant cutoff")?;
    let target_args = root
        .arguments(root.node(transformed_wire.node).ok_or("missing target operation")?)
        .ok_or("missing target operation inputs")?;
    let original_source = value(target_args[0])?;
    let online_target = value(target_args[1])?;
    let pre_record = preshape.witness_record.as_ref().ok_or("missing pre witness")?;
    let pre_body = preshape.body_relation.as_ref().ok_or("missing pre body")?;
    let pre_params = &preartifact.root.parameter_type;
    if !n.is_power_of_two() {
        return Err("readout degree is not a power of two".into());
    }
    let log = n.trailing_zeros();
    let width = helpers + 1;
    if !width.is_power_of_two() || n % width != 0 {
        return Err("invalid native trace dimensions".into());
    }
    let width_log = width.trailing_zeros();
    let stride = n / width;
    let gain = format!("({columns} * {n} * (({base} : Int).toNat / 2))");
    source = source.replace("end ExponentGeneratedReadoutStage\n", "");
    source.push_str(&format!(r#"
private theorem helper_indices_odd : ∀ i : Fin {helpers},
    Odd (MxxIR.selectInt (Int.ofNat i.val) [{indices}]).toNat := by decide

/-- Actual linked readout noise, before rewriting the proved trace scalar to
its antiperiodic table value. Cache links are concrete boundary equalities. -/
theorem encoding_transfer {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}
    {{preParams : {pre_params}}} {{preInputs : ExponentGeneratedReadoutOperands.RootInput}}
    {{preOutputs : ExponentGeneratedReadoutOperands.RootOutput}}
    (preWitness : {pre_record})
    (preFacts : {pre_body} backend hashModel preParams preInputs preOutputs preWitness)
    {{params : {params}}} {{inputs : RootInput}} {{outputs : RootOutput}}
    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness)
    (targetLink : ({online_target}) = ({pretarget}))
    (switchLink : ({switch_actual}) = ({preswitch}))
    (helperLink : ({helper_actual}) = ({prehelper}))
    (normalizationLink : ({norm}) = ({prenorm}))
    (gadget : ExactMatrix {q} {n} 2 {columns})
    (gadgetRuns : MxxRuntime.gadgetMatrixRuns backend {base} {digits} gadget)
    (label : Fin {labels}) (message : ExactPoly {q} {n}) (inputBound : Nat)
    (hmonomial : SignedMonomial message)
    (hsource : Approx (({original_source}) label)
      (({secret}) * (({prepublic}) label) - message • (({setup}) * gadget)) inputBound) :
    Approx (({normalized}) label)
      (({secret}) * ((({prepublic}) label * ({pretarget}) label +
          ∑ i : Fin {helpers}, ({mask_key}) i * ({prehelper}) ⟨label.val * {helpers} + i.val, by omega⟩) * ({prenorm})) -
        (ExponentGeneratedReadoutFunction.normalizationScalar *
          (message * ExponentGeneratedReadoutFunction.targetScalar preWitness +
            ∑ i : Fin {helpers}, readoutAutomorphismHom
              (MxxIR.selectInt (Int.ofNat i.val) [{indices}]).toNat (helper_indices_odd i)
                (message * ExponentGeneratedReadoutFunction.targetScalar preWitness))) • (({setup}) * gadget))
      (normalizedTraceBound Finset.univ inputBound {cutoff} {gain} 1 {gain}
        (∑ k, (2 * ExponentGeneratedReadoutFunction.targetCoefficients k - 1).natAbs)
        (fun _ : Fin {helpers} => {gain}) (fun _ : Fin {helpers} => {gain})) := by
  classical
  have oddIndices : ∀ i : Fin {helpers}, Odd (MxxIR.selectInt (Int.ofNat i.val) [{indices}]).toNat := by decide
  have indexEq : ∀ i : Fin {helpers},
      (MxxIR.selectInt (Int.ofNat i.val + 1) [{full_indices}]).toNat =
        (MxxIR.selectInt (Int.ofNat i.val) [{indices}]).toNat := by decide
  have indexCast : ∀ i : Fin {helpers},
      ((MxxIR.selectInt (Int.ofNat i.val) [{indices}]).toNat : Int) =
        MxxIR.selectInt (Int.ofNat i.val) [{indices}] := by decide
  have gadgetFacts := gadgetRuns
  obtain ⟨layout, lookup, _, _, _, _⟩ := gadgetFacts
  obtain ⟨target, ht, _, htg⟩ := ExponentGeneratedReadoutOperands.target_gain preWitness preFacts label
  obtain ⟨normalization, hn, _, hng⟩ := ExponentGeneratedReadoutOperands.normalization_gain preWitness preFacts
  choose action ha hab hag using ExponentGeneratedReadoutOperands.switch_gain preWitness preFacts
  choose helper hh hhb hhg using ExponentGeneratedReadoutOperands.helper_gain preWitness preFacts
  choose switchError oldAction oldHa oldGain hs hsb using
    ExponentGeneratedReadoutRoot.readout_switch_0 preWitness preFacts layout lookup gadget gadgetRuns
  choose maskError hm hmb using
    ExponentGeneratedOnlineReadout.helper_mask_{mask_id} witness facts gadget gadgetRuns
  choose raw hr he using helper_execution witness facts label
  obtain ⟨htl, hplain⟩ := ExponentGeneratedReadoutFunction.target_lift_sum preWitness preFacts
  have htc := ExponentGeneratedReadoutOperands.target_reconstruction preWitness preFacts gadget gadgetRuns label
  have hnc := ExponentGeneratedReadoutOperands.normalization_reconstruction preWitness preFacts gadget gadgetRuns
  have htargetRun := ExponentGeneratedOnlineReadout.family_operation_{target_id} witness facts label
  rw [targetLink, ht] at htargetRun
  have rawRuns (i : Fin {helpers}) : MxxRuntime.ringAutomorphismRuns
      ((MxxIR.selectInt (Int.ofNat i.val) [{indices}]).toNat : Int)
      (({original_source}) label * reduceMatrix {q} {n} {columns} {columns} target) (raw i) := by
    simpa only [indexCast, htargetRun] using hr i
  have result := readout_encoding_transfer (log := {log}) (by decide) Finset.univ
    (({original_source}) label) ({secret}) ({setup}) (({prepublic}) label) gadget message
    (ExponentGeneratedReadoutFunction.targetScalar preWitness)
    ExponentGeneratedReadoutFunction.normalizationScalar ExponentGeneratedReadoutFunction.targetInteger
    target normalization (fun i : Fin {helpers} => (MxxIR.selectInt (Int.ofNat i.val) [{indices}]).toNat)
    oddIndices raw ({mask_actual}) ({mask_key}) action
    (fun i : Fin {helpers} => helper ⟨label.val * {helpers} + i.val, by omega⟩)
    switchError maskError (fun _ => {gain}) (fun _ => {gain})
    hsource hmonomial htl hplain htg hng
    (by simpa only [ht, ExponentGeneratedReadoutFunction.targetScalar] using htc)
    (by simpa only [hn, ExponentGeneratedReadoutFunction.normalizationScalar] using hnc)
    (fun i _ => readout_gadget_automorphism (by decide) (by decide) _ (oddIndices i) gadgetRuns)
    (fun i _ => rawRuns i)
    (fun i _ => by simpa only [indexEq] using hm i)
    (fun i _ => by simpa only [indexEq, ha] using hs i)
    (fun i _ => by
      have h := ExponentGeneratedReadoutOperands.aligned_helpers preWitness preFacts layout lookup gadget gadgetRuns label i
      simpa only [ht, hh, ha] using h)
    (fun i _ => hsb i) (fun i _ => hmb i) (fun i _ => hag i)
    (fun i _ => hhg ⟨label.val * {helpers} + i.val, by omega⟩)
  have outputEq := normalized_execution witness facts label
  rw [normalizationLink, htargetRun] at outputEq
  simp only [he, switchLink, helperLink, ha, hh, hn] at outputEq
  rw [outputEq]
  simpa only [Finset.mem_univ, Finset.sum_const_zero, ht, hn, hh] using result

/-- The actual native readout value on the full signed strided-monomial
input domain. The second half is the evaluator's antiperiodic extension. -/
theorem native_transfer {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}
    {{preParams : {pre_params}}} {{preInputs : ExponentGeneratedReadoutOperands.RootInput}}
    {{preOutputs : ExponentGeneratedReadoutOperands.RootOutput}}
    (preWitness : {pre_record})
    (preFacts : {pre_body} backend hashModel preParams preInputs preOutputs preWitness)
    {{params : {params}}} {{inputs : RootInput}} {{outputs : RootOutput}}
    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness)
    (targetLink : ({online_target}) = ({pretarget}))
    (switchLink : ({switch_actual}) = ({preswitch}))
    (helperLink : ({helper_actual}) = ({prehelper}))
    (normalizationLink : ({norm}) = ({prenorm}))
    (gadget : ExactMatrix {q} {n} 2 {columns})
    (gadgetRuns : MxxRuntime.gadgetMatrixRuns backend {base} {digits} gadget)
    (label : Fin {labels}) (exponent : Fin (2 * {width})) (inputBound : Nat)
    (hsource : Approx (({original_source}) label)
      (({secret}) * (({prepublic}) label) -
        (AdjoinRoot.root (negacyclicModulus {n} (ZMod {q})) ^ ({stride} * exponent.val)) •
          (({setup}) * gadget)) inputBound) :
    Approx (({normalized}) label)
      (({secret}) * ((({prepublic}) label * ({pretarget}) label +
          ∑ i : Fin {helpers}, ({mask_key}) i * ({prehelper}) ⟨label.val * {helpers} + i.val, by omega⟩) * ({prenorm})) -
        (if h : exponent.val < {width} then
          (ExponentGeneratedReadoutFunction.targetCoefficients ⟨exponent.val, h⟩ : ExactPoly {q} {n}) -
            ({width} : ExactPoly {q} {n}) * ExponentGeneratedReadoutFunction.normalizationScalar
        else ({width} : ExactPoly {q} {n}) * ExponentGeneratedReadoutFunction.normalizationScalar -
          (ExponentGeneratedReadoutFunction.targetCoefficients ⟨exponent.val - {width}, by omega⟩ : ExactPoly {q} {n})) •
            (({setup}) * gadget))
      (normalizedTraceBound Finset.univ inputBound {cutoff} {gain} 1 {gain}
        (∑ k, (2 * ExponentGeneratedReadoutFunction.targetCoefficients k - 1).natAbs)
        (fun _ : Fin {helpers} => {gain}) (fun _ : Fin {helpers} => {gain})) := by
  classical
  letI : Fact (1 < {q}) := ⟨by decide⟩
  letI : NeZero {q} := ⟨by decide⟩
  have hmonomial : SignedMonomial
      (AdjoinRoot.root (negacyclicModulus {n} (ZMod {q})) ^ ({stride} * exponent.val)) :=
    ⟨⟨{stride} * exponent.val, by omega⟩, rfl⟩
  have result := encoding_transfer preWitness preFacts witness facts targetLink switchLink helperLink
    normalizationLink gadget gadgetRuns label _ inputBound hmonomial hsource
  have tailIndices : ∀ i : Fin {helpers}, TraceIndices.indices i.succ =
      (MxxIR.selectInt (Int.ofNat i.val) [{indices}]).toNat := by decide
  have identityIndex : TraceIndices.indices (0 : Fin {width}) = 1 := by decide
  have identityHom (x : ExactPoly {q} {n}) : readoutAutomorphismHom 1 (by decide) x = x := by
    conv_lhs => rw [Negacyclic.expansion (by decide) x]
    simp [readoutAutomorphismHom]
    exact (Negacyclic.expansion (by decide) x).symm
  have sumEq :
      (∑ i : Fin {width}, readoutAutomorphismHom (TraceIndices.indices i) (TraceIndices.all_odd i)
        (AdjoinRoot.root (negacyclicModulus {n} (ZMod {q})) ^ ({stride} * exponent.val) *
          ExponentGeneratedReadoutFunction.targetScalar preWitness)) =
      AdjoinRoot.root (negacyclicModulus {n} (ZMod {q})) ^ ({stride} * exponent.val) *
          ExponentGeneratedReadoutFunction.targetScalar preWitness +
        ∑ i : Fin {helpers}, readoutAutomorphismHom
          (MxxIR.selectInt (Int.ofNat i.val) [{indices}]).toNat (helper_indices_odd i)
            (AdjoinRoot.root (negacyclicModulus {n} (ZMod {q})) ^ ({stride} * exponent.val) *
              ExponentGeneratedReadoutFunction.targetScalar preWitness) := by
    rw [Fin.sum_univ_succ]
    simp only [identityIndex, tailIndices]
    rw [identityHom]
  have functional := normalized_nativeTraceTarget (q := {q}) (n := {n}) (stride := {stride}) (log := {width_log})
    ExponentGeneratedReadoutFunction.targetCoefficients exponent.val TraceIndices.indices TraceIndices.all_odd
    TraceIndices.permutation TraceIndices.index_correspondence (by decide) exponent.isLt
    ExponentGeneratedReadoutFunction.normalizationScalar
    (({width} : ExactPoly {q} {n}) * ExponentGeneratedReadoutFunction.normalizationScalar)
    (by convert ExponentGeneratedReadoutFunction.normalization_correct using 1; norm_num)
    (by simpa only [mul_comm] using ExponentGeneratedReadoutFunction.half_correct)
  rw [← ExponentGeneratedReadoutFunction.target_function preWitness preFacts] at functional
  change ExponentGeneratedReadoutFunction.normalizationScalar *
    (∑ i : Fin {width}, readoutAutomorphismHom (TraceIndices.indices i) (TraceIndices.all_odd i)
      (AdjoinRoot.root (negacyclicModulus {n} (ZMod {q})) ^ ({stride} * exponent.val) *
        ExponentGeneratedReadoutFunction.targetScalar preWitness)) = _ at functional
  rw [sumEq] at functional
  rw [functional] at result
  exact result

end ExponentGeneratedReadoutStage
"#, params = artifact.root.parameter_type, mask_id = mask_source_wire.node.0, target_id = transformed_wire.node.0));
    Ok(source)
}

/// The native simulator assigns bounded-matrix plans to these four readout
/// operands. Derive their typed digit bound from the actual decomposition,
/// retaining scalar-gadget reconstruction for target and normalization.
pub(super) fn pre_readout_operands(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
) -> Result<String, String> {
    let root = graph.source.root_scope();
    let root_shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let body = root_shape.body_relation.as_ref().ok_or("missing pre readout body")?;
    let record = root_shape.witness_record.as_ref().ok_or("missing pre readout witness")?;
    let root_types = &graph.scopes[&FrozenGraphScopeId::Root].wire_types;
    let value = |wire: WireRef| {
        value_expression(
            &format!("w_{}_{}", wire.node.0, wire.port.0),
            &root_shape.value_expressions,
        )
    };
    let root_fact = |wire: WireRef| -> Result<String, String> {
        let pos = root_shape
            .node_constraints
            .get(&wire.node)
            .and_then(|r| r.clone().last())
            .ok_or("missing pre readout fact")?;
        Ok(format!("facts{}.1", ".2".repeat(pos)))
    };
    let boundary_type =
        |values: &std::collections::BTreeMap<String, mxx_ir_core::lean::BoundaryValue>,
         count: usize| {
            let mut types = vec![String::new(); count];
            for value in values.values() {
                types[value.tuple_index] = format!("({})", value.lean_type);
            }
            if count == 0 {
                "Unit".into()
            } else if count == 1 {
                types.remove(0)
            } else {
                types.join(" × ") + " × Unit"
            }
        };
    let input_type = boundary_type(&artifact.root.inputs, artifact.root.input_count);
    let output_type = boundary_type(&artifact.root.outputs, artifact.root.output_count);
    let binders = format!(
        "{{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {}}} {{inputs : RootInput}} {{outputs : RootOutput}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness)",
        artifact.root.parameter_type
    );
    let mut source = format!(
        "import {}\nimport ExponentReadoutStage\nimport RootReadoutAlignment\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\n\nnamespace ExponentGeneratedReadoutOperands\nopen Mxx.Primitives MxxExponentLut\nabbrev RootInput := {input_type}\nabbrev RootOutput := {output_type}\n\n",
        artifact.module_name
    );
    let mut found = 0;
    for (name, output) in &artifact.root.outputs {
        let role = if name.ends_with("-readout-target-decomposition-0") {
            "target"
        } else if name.ends_with("-readout-normalization-decomposition-0") {
            "normalization"
        } else if name.ends_with("-readout-helper-switch-family-0") {
            "switch"
        } else if name.ends_with("-readout-helper-public-family-0") {
            "helper"
        } else {
            continue
        };
        let node = root.node(output.wire.node).ok_or("missing readout operand")?;
        let actual = value(output.wire)?;
        let (scope, shape, scope_types, decomp_wire, lanes, proof_start, output_equation) =
            if matches!(node.kind(), NodeKind::ParallelLoop(_)) {
                let id = FrozenGraphScopeId::ParallelBody {
                    parent: Box::new(FrozenGraphScopeId::Root),
                    owner: output.wire.node,
                };
                let scope = &graph.source.scopes()[&id];
                let shape = &artifact.scope_proofs[&id];
                let ConcreteWireType::IndexedFamily { count, .. } = &output.wire_type else {
                    return Err("readout operand loop not family".into())
                };
                if scope.outputs().len() != 1 {
                    return Err("readout decomposition multiple outputs".into());
                }
                let names = shape
                    .witnesses
                    .iter()
                    .map(|(name, _)| name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                let proof = format!(
                    "  intro i\n  obtain ⟨{names}, childFacts⟩ := {} i\n",
                    root_fact(output.wire)?
                );
                (
                    scope,
                    shape,
                    &graph.scopes[&id].wire_types,
                    scope.outputs()[0],
                    Some(*count),
                    proof,
                    format!("childFacts{}", ".2".repeat(shape.constraint_count)),
                )
            } else {
                (root, root_shape, root_types, output.wire, None, String::new(), String::new())
            };
        let decomp = scope.node(decomp_wire.node).ok_or("missing actual decomposition")?;
        let NodeKind::GadgetDecompose { base, digit_count, .. } = decomp.kind() else {
            return Err(format!("{role} readout operand is not actual decomposition"));
        };
        let base = base.evaluate(&ParamEnv::default()).map_err(|_| "nonconstant readout radix")?;
        let digits = digit_count
            .evaluate(&ParamEnv::default())
            .map_err(|_| "nonconstant readout digit count")?;
        let matrix =
            scope_types[&decomp_wire].matrix_type().ok_or("invalid readout action type")?;
        let q = &matrix.modulus;
        let n = matrix.ring_dimension;
        let rows = matrix.rows;
        let columns = matrix.columns;
        let fact_for = |wire: WireRef| -> Result<String, String> {
            let pos = shape
                .node_constraints
                .get(&wire.node)
                .and_then(|r| r.clone().last())
                .ok_or("missing operand child fact")?;
            Ok(format!(
                "{}{}.1",
                if lanes.is_some() { "childFacts" } else { "facts" },
                ".2".repeat(pos)
            ))
        };
        let hd = fact_for(decomp_wire)?;
        let quantifier = lanes.map(|lanes| format!("∀ i : Fin {lanes}, ")).unwrap_or_default();
        let selected =
            if lanes.is_some() { format!("({actual}) i") } else { format!("({actual})") };
        let finish = if lanes.is_some() {
            format!("  exact ⟨action, ({output_equation}).trans ha, hb, hg⟩\n")
        } else {
            "  exact ⟨action, ha, hb, hg⟩\n".into()
        };
        source.push_str(&format!("-- Native bounded-matrix gain for actual {role} operand.\ntheorem {role}_gain {binders} :\n    {quantifier}∃ action : ErrorMatrix {n} {rows} {columns},\n      {selected} = reduceMatrix {q} {n} {rows} {columns} action ∧\n      CoeffBound action (({base} : Int).toNat / 2) ∧\n      ActionGain action ({rows} * {n} * (({base} : Int).toNat / 2)) := by\n{proof_start}  obtain ⟨action, ha, hb, hg⟩ := readout_decomposition_gain (by decide) (by decide) {hd}\n{finish}\n"));
        if role == "target" || role == "normalization" {
            let product_wire = scope
                .arguments(decomp)
                .and_then(|a| a.first().copied())
                .ok_or("missing scalar gadget product")?;
            let product = scope.node(product_wire.node).ok_or("missing scalar gadget operation")?;
            if !matches!(product.kind(), NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)) {
                return Err("readout target is not scalar gadget multiplication".into());
            }
            let product_args = scope.arguments(product).ok_or("missing scalar gadget factors")?;
            let scalar_slot = product_args
                .iter()
                .position(|wire| {
                    scope_types[wire]
                        .matrix_type()
                        .is_some_and(|matrix| matrix.rows == 1 && matrix.columns == 1)
                })
                .ok_or("missing readout scalar")?;
            let scalar_operation = if scalar_slot == 0 {
                "MxxRuntime.matrixMulScalarLeft"
            } else {
                "MxxRuntime.matrixMulScalarRight"
            };
            let scalar_wire = product_args[scalar_slot];
            let gadget_wire = product_args[1 - scalar_slot];
            let scalar = if lanes.is_some() {
                let slot = scope
                    .inputs()
                    .iter()
                    .position(|wire| *wire == scalar_wire)
                    .ok_or("readout scalar must be actual captured input")?;
                let args = root.arguments(node).ok_or("missing scalar capture")?;
                let NodeKind::ParallelLoop(loop_) = node.kind() else { unreachable!() };
                if loop_.input_modes[slot] != LoopInputMode::Broadcast {
                    return Err("readout target scalar is not shared across labels".into());
                }
                value(args[slot])?
            } else {
                value(scalar_wire)?
            };
            let hg = fact_for(gadget_wire)?;
            let other = if lanes.is_some() {
                format!("w_{}_{}", gadget_wire.node.0, gadget_wire.port.0)
            } else {
                value(gadget_wire)?
            };
            let target_rows =
                scope_types[&gadget_wire].matrix_type().ok_or("invalid gadget type")?.rows;
            let rw_output =
                if lanes.is_some() { format!("  rw [{output_equation}]\n") } else { String::new() };
            source.push_str(&format!("theorem {role}_reconstruction {binders}\n    (gadget : ExactMatrix {q} {n} {target_rows} {columns})\n    (gadgetRuns : MxxRuntime.gadgetMatrixRuns backend {base} {digits} gadget) :\n    {quantifier}gadget * {selected} = (({scalar}) 0 0) • gadget := by\n{proof_start}  have same : ({other}) = gadget := gadget_runs_deterministic {hg} gadgetRuns\n{rw_output}  rw [gadget_runs_reconstruct (by decide) (by decide) gadgetRuns {hd}]\n  rw [same]\n  ext row column\n  simp [{scalar_operation}, mul_comm]\n\n"));
        }
        found += 1;
    }
    if found != 4 {
        return Err(format!("expected four native readout operands, found {found}"));
    }
    // Reindex the actual flattened public-helper cache by its label and helper.
    // Both positions are proved from the producer's integer index operations.
    let helper_wire = artifact
        .root
        .outputs
        .iter()
        .find(|(name, _)| name.ends_with("-readout-helper-public-family-0"))
        .ok_or("missing readout helper output")?
        .1
        .wire;
    let helper_node = root.node(helper_wire.node).ok_or("missing helper node")?;
    let helper_args = root.arguments(helper_node).ok_or("missing helper operands")?;
    let child_id = |wire: WireRef| FrozenGraphScopeId::ParallelBody {
        parent: Box::new(FrozenGraphScopeId::Root),
        owner: wire.node,
    };
    let helper_scope = &graph.source.scopes()[&child_id(helper_wire)];
    let automorphism = helper_scope
        .nodes()
        .iter()
        .find(|node| matches!(node.kind(), NodeKind::RingAutomorphism { .. }))
        .ok_or("missing helper automorphism")?;
    let NodeKind::RingAutomorphism { index: IntExpr::Select { branches, .. } } =
        automorphism.kind()
    else {
        return Err("helper automorphism must use actual Select table".into());
    };
    let indices = branches
        .iter()
        .map(|branch| {
            branch
                .evaluate(&ParamEnv::default())
                .map(|index| format!("({index})"))
                .map_err(|_| "nonconstant helper index")
        })
        .collect::<Result<Vec<_>, _>>()?
        .join(", ");
    let gather = |wire: WireRef| -> Result<(WireRef, WireRef), String> {
        let scope = &graph.source.scopes()[&child_id(wire)];
        let get = scope.node(scope.outputs()[0].node).ok_or("missing helper gather")?;
        if !matches!(get.kind(), NodeKind::FamilyGetDynamic) {
            return Err("helper input not gathered".into());
        }
        let args = scope.arguments(get).ok_or("missing gathered arguments")?;
        let parents = root
            .arguments(root.node(wire.node).ok_or("missing gather loop")?)
            .ok_or("missing gather inputs")?;
        let map = |wire: WireRef| {
            scope
                .inputs()
                .iter()
                .position(|input| *input == wire)
                .map(|slot| parents[slot])
                .ok_or("gather operand not captured")
        };
        Ok((map(args[0])?, map(args[1])?))
    };
    let (transformed, label_indices) = gather(helper_args[0])?;
    let (switch, helper_indices) = gather(helper_args[1])?;
    let transformed_args = root
        .arguments(root.node(transformed.node).ok_or("missing transformed key")?)
        .ok_or("missing transformed key operands")?;
    let ConcreteWireType::IndexedFamily { count: total, element, .. } = &root_types[&helper_wire]
    else {
        return Err("helper cache not indexed family".into());
    };
    let ConcreteWireType::IndexedFamily { count: helpers, .. } = &root_types[&switch] else {
        return Err("switch cache not indexed family".into());
    };
    let labels = total / helpers;
    let matrix = element.matrix_type().ok_or("helper cache not matrices")?;
    let q = &matrix.modulus;
    let n = matrix.ring_dimension;
    let columns = matrix.columns;
    let helper_actual = value(helper_wire)?;
    let switch_actual = value(switch)?;
    let public_actual = value(transformed_args[0])?;
    let target_actual = value(transformed_args[1])?;
    let li = value(label_indices)?;
    let hi = value(helper_indices)?;
    let lr = &artifact.scope_relations[&child_id(label_indices)];
    let hr = &artifact.scope_relations[&child_id(helper_indices)];
    let lp = root_fact(label_indices)?;
    let hp = root_fact(helper_indices)?;
    let decomp =
        helper_scope.node(helper_scope.outputs()[0].node).ok_or("missing helper decomp")?;
    let NodeKind::GadgetDecompose { base, digit_count, .. } = decomp.kind() else {
        return Err("missing helper decomp".into());
    };
    let base = base.evaluate(&ParamEnv::default()).map_err(|_| "nonconstant base")?;
    let digits = digit_count.evaluate(&ParamEnv::default()).map_err(|_| "nonconstant digits")?;
    source.push_str(&format!("theorem aligned_helpers {binders}\n    (layout : MxxRuntime.RegularLayout {q})\n    (lookup : backend.regularLayout {q} {n} = some layout)\n    (gadget : ExactMatrix {q} {n} 2 {columns})\n    (gadgetRuns : MxxRuntime.gadgetMatrixRuns backend {base} {digits} gadget) :\n    ∀ label : Fin {labels}, ∀ helper : Fin {helpers},\n      gadget * ({helper_actual}) ⟨label.val * {helpers} + helper.val, by omega⟩ =\n        MxxRuntime.ringAutomorphism (MxxIR.selectInt (Int.ofNat helper.val) [{indices}]).toNat\n          (({public_actual}) label * ({target_actual}) label) * ({switch_actual}) helper := by\n  intro label helper\n  let lane : Fin {total} := ⟨label.val * {helpers} + helper.val, by omega⟩\n  obtain ⟨position, targetPosition, action, hposition, htarget, _, _, heq⟩ :=\n    ExponentGeneratedReadoutAlignment.readout_alignment_0 witness facts layout lookup gadget gadgetRuns lane\n  have hlabel : ({li}) lane = (label.val : Int) := by\n    have h := {lp} lane\n    dsimp only [{lr}, {lr}.constraints_0, lane] at h ⊢\n    simp only [Int.ofNat_eq_natCast, Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat] at h\n    omega\n  have hhelper : ({hi}) lane = (helper.val : Int) := by\n    have h := {hp} lane\n    dsimp only [{hr}, {hr}.constraints_0, lane] at h ⊢\n    simp only [Int.ofNat_eq_natCast, Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat] at h\n    omega\n  have sameLabel : targetPosition = label := by apply Fin.ext; omega\n  have sameHelper : position = helper := by apply Fin.ext; omega\n  have hmod : Int.fmod (Int.ofNat lane.val) {helpers} = Int.ofNat helper.val := by\n    rw [Int.fmod_eq_emod_of_nonneg _ (by decide)]\n    dsimp only [lane]\n    simp only [Int.ofNat_eq_natCast, Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat]\n    omega\n  simpa only [sameLabel, sameHelper, hmod] using heq\n\n"));
    source.push_str("end ExponentGeneratedReadoutOperands\n");
    Ok(source)
}
