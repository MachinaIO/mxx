//! Closed routed readout bounds, using one exponent choice for the entire run.
use super::root_inputs::value_expression;
use crate::noise_refresh::simulation::ExponentLutNoiseSnapshot;
use mxx_ir_core::{
    FrozenGraphScopeId, ValidatedGraph,
    lean::{LeanArtifact, ScopeProofShape},
    node::{LoopInputMode, MatrixBinaryOp, NodeKind},
    types::{ConcreteWireType, WireRef},
};

fn child(wire: WireRef) -> FrozenGraphScopeId {
    FrozenGraphScopeId::ParallelBody {
        parent: Box::new(FrozenGraphScopeId::Root),
        owner: wire.node,
    }
}

fn expression(
    shape: &ScopeProofShape,
    wire: WireRef,
    witness: &str,
    inputs: &str,
) -> Result<String, String> {
    let mut values = shape.value_expressions.clone();
    values.insert("witness".into(), witness.into());
    values.insert("inputs".into(), inputs.into());
    value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &values)
}

pub(super) fn root_refresh_source_proof(
    pre: &ValidatedGraph,
    pa: &LeanArtifact,
    ver: &ValidatedGraph,
    va: &LeanArtifact,
    snapshot: &ExponentLutNoiseSnapshot,
) -> Result<String, String> {
    let root = ver.source.root_scope();
    let ps = &pa.scope_proofs[&FrozenGraphScopeId::Root];
    let vs = &va.scope_proofs[&FrozenGraphScopeId::Root];
    let vt = &ver.scopes[&FrozenGraphScopeId::Root].wire_types;
    let pv = |wire| expression(ps, wire, "pw", "pin");
    let vv = |wire| expression(vs, wire, "vw", "vin");
    let args = |wire: WireRef| {
        root.node(wire.node)
            .and_then(|node| root.arguments(node))
            .ok_or("missing source route operands")
    };
    let refresh = &snapshot.refresh;
    let mut source = String::from(
        "import RootRefreshSource\nimport RootReadoutProof\nimport RootRefreshRouteTargets\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedRefreshSourceProof\nopen Mxx.Primitives MxxRuntime MxxExponentLut GeneratedClaim ExponentGeneratedExecution\nopen scoped BigOperators\n\nprivate theorem replace_actual {q n rows columns bound : Nat}\n    {actual other ideal : ExactMatrix q n rows columns}\n    (equation : actual = other) (close : Approx other ideal bound) : Approx actual ideal bound := by\n  obtain ⟨error, errorEquation, bounded⟩ := close\n  exact ⟨error, equation.trans errorEquation, bounded⟩\n\nnoncomputable def readoutExponent {hashModel : MxxRuntime.HashModel} {external : ExternalInputs} {execution : Execution}\n    (runs : Runs hashModel external execution) (label) :=\n  Classical.choose (ExponentGeneratedReadoutProof.native_readout_closed runs label)\n\n",
    );
    for role in ["mask", "fresh"] {
        let suffix = format!("-{role}-route-k-rhs-0");
        let cache = va
            .root
            .inputs
            .iter()
            .find(|(name, _)| name.ends_with(&suffix))
            .map(|(_, x)| x.wire)
            .ok_or("missing online route cache")?;
        let precache = pa
            .root
            .outputs
            .iter()
            .find(|(name, _)| name.ends_with(&suffix))
            .map(|(_, x)| x.wire)
            .ok_or("missing producer route cache")?;
        let ConcreteWireType::IndexedFamily { count: total, element } = &vt[&cache] else {
            return Err("route cache not family".into())
        };
        if !pre.scopes[&FrozenGraphScopeId::Root].wire_types.contains_key(&precache) {
            return Err("missing producer cache type".into())
        }
        let matrix = element.matrix_type().ok_or("route cache element not matrix")?;
        let q = &matrix.modulus;
        let n = matrix.ring_dimension;
        let columns = matrix.columns;
        let layout = pa
            .backend_layouts
            .iter()
            .position(|layout| {
                layout.modulus == *q &&
                    layout.ring_dimension == n &&
                    2 * layout.regular_digits == columns
            })
            .ok_or("missing source route layout")?;
        let backend_layout = &pa.backend_layouts[layout];
        let gadget_node = root
            .nodes()
            .iter()
            .enumerate()
            .find_map(|(i, node)| {
                let NodeKind::ConstantMatrix {
                    value: mxx_ir_core::node::ConstantMatrix::Gadget { .. },
                    ..
                } = node.kind()
                else {
                    return None
                };
                let wire = WireRef {
                    node: mxx_ir_core::types::NodeId(i as u64),
                    port: mxx_ir_core::types::Port(0),
                };
                let ty = vt[&wire].matrix_type()?;
                (ty.modulus == *q &&
                    ty.ring_dimension == n &&
                    ty.rows == 2 &&
                    ty.columns == columns)
                    .then_some(wire)
            })
            .ok_or("missing source gadget")?;
        let gadget_position = vs.node_constraints[&gadget_node.node]
            .clone()
            .last()
            .ok_or("missing source gadget fact")?;
        let gadget_fact = format!("vf{}.1", ".2".repeat(gadget_position));
        let mut products = root.nodes().iter().enumerate().filter_map(|(node_index, node)| {
            let parents = root.arguments(node)?;
            if !parents.contains(&cache) {
                return None
            }
            let wire = WireRef {
                node: mxx_ir_core::types::NodeId(node_index as u64),
                port: mxx_ir_core::types::Port(0),
            };
            let scope = ver.source.scopes().get(&child(wire))?;
            let output = scope.node(scope.outputs()[0].node)?;
            matches!(
                output.kind(),
                NodeKind::MatrixMulSmallRhs | NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)
            )
            .then_some((wire, scope, parents))
        });
        let (product, product_scope, captures) =
            products.next().ok_or("missing actual routed product")?;
        if products.next().is_some() {
            return Err("ambiguous actual routed product".into())
        }
        let operands = product_scope
            .arguments(
                product_scope
                    .node(product_scope.outputs()[0].node)
                    .ok_or("missing route product output")?,
            )
            .ok_or("missing route product operands")?;
        let input = captures[product_scope
            .inputs()
            .iter()
            .position(|wire| *wire == operands[0])
            .ok_or("route input is not captured")?];
        let actual = vv(product)?;
        let input_value = vv(input)?;
        let online_d = vv(cache)?;
        let pre_d = pv(precache)?;
        let mut offset = 0usize;
        for slot in 0..refresh.slot_count {
            let digits = if role == "mask" {
                refresh.mask_digit_counts[slot]
            } else {
                refresh.fresh_digit_count
            };
            let width = columns * n * digits;
            let flat = format!(
                "⟨{offset} + (component.val * {n} + coefficient.val) * {digits} + digit.val, by clear * - component coefficient digit; omega⟩"
            );
            let label = if role == "mask" {
                format!("{offset} + (component.val * {n} + coefficient.val) * {digits} + digit.val")
            } else {
                let mask_total: usize =
                    refresh.mask_digit_counts.iter().sum::<usize>() * columns * n;
                format!(
                    "{mask_total} + (component.val * {n} + coefficient.val) * {digits} + digit.val"
                )
            };
            source.push_str(&format!("noncomputable def {role}Selection_{slot} {{hashModel : MxxRuntime.HashModel}} {{external : ExternalInputs}} {{execution : Execution}}\n    (runs : Runs hashModel external execution) (_ : Fin 1) (component : Fin {columns}) (coefficient : Fin {n}) (digit : Fin {digits}) :=\n  readoutExponent runs ⟨{label}, by clear * - component coefficient digit; omega⟩\n\n"));
            let scale = if role == "mask" {
                &refresh.switch_moduli[slot]
            } else {
                &refresh.plaintext_moduli[slot]
            };
            let q_nat = refresh.global_modulus.as_ref().ok_or("missing source q")?;
            let scale = q_nat / scale;
            let base = &refresh.base_p;
            let gains = refresh.route_gains.as_ref().ok_or("missing source route gains")?;
            let gain = if role == "mask" { &gains.mask[slot].l1 } else { &gains.fresh[slot].l1 };
            source.push_str(&format!("def {role}Gain_{slot} : Nat := {gain}\n\ntheorem {role}_gain_{slot} :\n    ({n} * ∑ digit : Fin {digits}, sparseColumnGain RefreshBackend.layout{layout} (-({scale} : Int) * ({base} : Int)^digit.val)) = {role}Gain_{slot} := by decide\n\n"));
            let result_type = format!(
                "    let pw := preprocessingWitness runs\n    let vw := verificationWitness runs\n    Approx (∑ component : Fin {columns}, ∑ coefficient : Fin {n}, ∑ digit : Fin {digits},\n      ({actual}) ({flat} : Fin {total}))\n      (∑ component : Fin {columns}, ∑ coefficient : Fin {n}, ∑ digit : Fin {digits},\n        (ExponentGeneratedReadoutProof.mask runs * ExponentGeneratedReadoutProof.readoutPublicKey runs ⟨{label}, by clear * - component coefficient digit; omega⟩ -\n          ExponentGeneratedReadoutProof.nativeValue (readoutExponent runs ⟨{label}, by clear * - component coefficient digit; omega⟩) •\n            (ExponentGeneratedReadoutProof.payload runs * ExponentGeneratedReadoutProof.gadget runs)) * ({pre_d}) ({flat} : Fin {total}))\n      (({n} * ∑ digit : Fin {digits}, sparseColumnGain RefreshBackend.layout{layout} (-({scale} : Int) * ({base} : Int)^digit.val)) *\n        SOURCE_BOUND)"
            );
            let result_type = result_type.replace("SOURCE_BOUND", "sourceBound");
            source.push_str(&format!("private theorem {role}_sum_{slot}_at_bound {{hashModel : MxxRuntime.HashModel}} {{external : ExternalInputs}} {{execution : Execution}}\n    (runs : Runs hashModel external execution) (sourceBound : Nat)\n    (boundEquation : ExponentGeneratedReadoutProof.readoutBound ExponentGeneratedNativeProgramProof.nativeOutputBound = sourceBound) :\n{result_type} := by\n  classical\n  dsimp only\n  let pw := preprocessingWitness runs\n  let vw := verificationWitness runs\n  let pin := preprocessingInputs runs.2.1\n  let vin := verificationInputs runs.2.2.1\n  have pf := preprocessingFacts runs\n  have vf := verificationFacts runs\n  have cacheLink : ({online_d}) = ({pre_d}) := by\n    simp only [vin, verificationInputs]\n    rw [preprocessingOutput runs]\n    rfl\n"));
            // The same actual gather chain fixes every readout label, including repeated fresh
            // inputs.
            source.push_str(&format!("  have inputEquation (component : Fin {columns}) (coefficient : Fin {n}) (digit : Fin {digits}) :\n      ({}) ({flat} : Fin {total}) = ExponentGeneratedReadoutProof.readoutCiphertext runs ⟨{label}, by clear * - component coefficient digit; omega⟩ := by\n    let index0 : Fin {total} := {flat}\n", vv(input)?));
            let mut wire = input;
            let mut depth = 0;
            while let Some(scope) = ver.source.scopes().get(&child(wire)) {
                let output =
                    scope.node(scope.outputs()[0].node).ok_or("missing route gather output")?;
                if !matches!(output.kind(), NodeKind::FamilyGetDynamic) {
                    break
                }
                let operands = scope.arguments(output).ok_or("missing gather operands")?;
                let parents = args(wire)?;
                let family = parents[scope
                    .inputs()
                    .iter()
                    .position(|w| *w == operands[0])
                    .ok_or("gather family not capture")?];
                let indices = parents[scope
                    .inputs()
                    .iter()
                    .position(|w| *w == operands[1])
                    .ok_or("gather index not capture")?];
                let index_scope = &ver.source.scopes()[&child(indices)];
                if !index_scope.inputs().is_empty() {
                    return Err("route index not pure".into())
                }
                let NodeKind::ParallelLoop(spec) =
                    root.node(indices.node).ok_or("missing index loop")?.kind()
                else {
                    return Err("route indices not loop".into())
                };
                if spec.input_modes.iter().any(|mode| *mode != LoopInputMode::Broadcast) {
                    return Err("unexpected route index modes".into())
                }
                let index_shape = &va.scope_proofs[&child(indices)];
                let mut values = index_shape.value_expressions.clone();
                values.insert(format!("i_{}", spec.index_slot), format!("index{depth}.val"));
                let index_output = index_scope.outputs()[0];
                let index_value = value_expression(
                    &format!("w_{}_{}", index_output.node.0, index_output.port.0),
                    &values,
                )?;
                let ConcreteWireType::IndexedFamily { count, .. } = &vt[&family] else {
                    return Err("gather source not family".into())
                };
                source.push_str(&format!("    obtain ⟨bound{depth}, eq{depth}⟩ := ExponentGeneratedRefreshSource.route_gather_{} vw vf index{depth}\n    let index{} : Fin {count} := ⟨({index_value}).toNat, bound{depth}⟩\n",wire.node.0,depth+1));
                wire = family;
                depth += 1;
            }
            if depth == 0 {
                return Err("routed source has no readout gather".into())
            }
            let indices = (0..=depth).map(|i| format!("index{i}")).collect::<Vec<_>>().join(", ");
            source.push_str(&format!("    have same : index{depth} = ⟨{label}, by clear * - component coefficient digit; omega⟩ := by\n      apply Fin.ext\n      dsimp only [{indices}]\n      norm_num only [Int.ofNat_eq_natCast, Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat, Int.toNat_natCast]\n      clear * - component coefficient digit\n      omega\n"));
            for d in (0..depth - 1).rev() {
                source.push_str(&format!(
                    "    have combined{d} := eq{d}.trans {}\n",
                    if d + 1 == depth - 1 {
                        format!("eq{}", d + 1)
                    } else {
                        format!("combined{}", d + 1)
                    }
                ));
            }
            let result = if depth == 1 { "eq0" } else { "combined0" };
            source.push_str(&format!("    change ({}) index0 = ({}) index{depth} at {result}\n    rw [same] at {result}\n    exact {result}\n",vv(input)?,vv(wire)?));
            // Final proof is emitted after exact input lineage, with no source approximation
            // premise.
            let output_projection = pa
                .root
                .outputs
                .values()
                .find(|value| value.wire == precache)
                .ok_or("missing route output projection")?
                .projection
                .strip_prefix("outputs")
                .ok_or("invalid route output projection")?;
            let ideal = format!(
                "ExponentGeneratedReadoutProof.mask runs * ExponentGeneratedReadoutProof.readoutPublicKey runs ⟨{label}, by clear * - component coefficient digit; omega⟩ - ExponentGeneratedReadoutProof.nativeValue (readoutExponent runs ⟨{label}, by clear * - component coefficient digit; omega⟩) • (ExponentGeneratedReadoutProof.payload runs * ExponentGeneratedReadoutProof.gadget runs)"
            );
            source.push_str(&format!("  let routeInput : Fin {columns} → Fin {n} → Fin {digits} → ExactMatrix {q} {n} 1 {columns} := fun component coefficient digit => ({input_value}) ({flat} : Fin {total})\n  let routeIdeal : Fin {columns} → Fin {n} → Fin {digits} → ExactMatrix {q} {n} 1 {columns} := fun component coefficient digit => {ideal}\n  let routeD : Fin {columns} → Fin {n} → Fin {digits} → ExactMatrix {q} {n} {columns} {columns} := fun component coefficient digit => ({pre_d}) ({flat} : Fin {total})\n  have close (component : Fin {columns}) (coefficient : Fin {n}) (digit : Fin {digits}) :\n      Approx (({input_value}) ({flat} : Fin {total})) ({ideal}) sourceBound := by\n    rw [inputEquation component coefficient digit, ← boundEquation]\n    exact Classical.choose_spec (ExponentGeneratedReadoutProof.native_readout_closed runs ⟨{label}, by clear * - component coefficient digit; omega⟩)\n  have outputLink : execution.preprocessing{output_projection} = ({pre_d}) := by\n    rw [preprocessingOutput runs]\n    rfl\n  have reduction (component : Fin {columns}) (coefficient : Fin {n}) (digit : Fin {digits}) :\n      ({pre_d}) ({flat} : Fin {total}) = reduceMatrix {q} {n} {columns} {columns}\n        (ExponentGeneratedRefreshRouteTargets.{role}Action_{slot} component coefficient digit) := by\n    have hg : MxxRuntime.gadgetMatrixRuns RefreshBackend.backend {} {} (ExponentGeneratedReadoutProof.gadget runs) := {gadget_fact}\n    have target := (ExponentGeneratedRefreshRouteTargets.{role}_cache_{slot} pw pf _ hg component coefficient digit).1\n    exact (congrFun outputLink ({flat} : Fin {total})).symm.trans target\n  have inputClose : ∀ c k d, Approx (routeInput c k d) (routeIdeal c k d) sourceBound := close\n  have actionReduction : ∀ c k d, routeD c k d = reduceMatrix {q} {n} {columns} {columns}\n      (regularIntegerDigits RefreshBackend.layout{layout} (routeTarget (-({scale} : Int)*({base} : Int)^d.val) k c)) := reduction\n  have bounded := route_sparse_family_approx (bound := sourceBound) (q := {q}) (n := {n}) (columns := {columns}) (digits := {digits})\n    RefreshBackend.layout{layout} (by decide) (by decide) (fun digit : Fin {digits} => -({scale} : Int)*({base} : Int)^digit.val)\n    routeInput routeIdeal routeD inputClose actionReduction\n  have sumEquation : (∑ component : Fin {columns}, ∑ coefficient : Fin {n}, ∑ digit : Fin {digits}, ({actual}) ({flat} : Fin {total})) =\n      ∑ component : Fin {columns}, ∑ coefficient : Fin {n}, ∑ digit : Fin {digits}, ({input_value}) ({flat} : Fin {total}) * ({pre_d}) ({flat} : Fin {total}) := by\n    apply Finset.sum_congr rfl\n    intro component hc\n    apply Finset.sum_congr rfl\n    intro coefficient hk\n    apply Finset.sum_congr rfl\n    intro digit hd\n    have productEquation : ({actual}) ({flat} : Fin {total}) = ({input_value}) ({flat} : Fin {total}) * ({online_d}) ({flat} : Fin {total}) := ExponentGeneratedRefreshSource.{role}_product_0 vw vf ({flat} : Fin {total})\n    exact productEquation.trans (congrArg (fun action => ({input_value}) ({flat} : Fin {total}) * action) (congrFun cacheLink ({flat} : Fin {total})))\n  exact replace_actual sumEquation bounded\n\n",backend_layout.base,backend_layout.regular_digits));
            let result_type = result_type.replace("sourceBound", "ExponentGeneratedReadoutProof.readoutBound ExponentGeneratedNativeProgramProof.nativeOutputBound");
            source.push_str(&format!("theorem {role}_sum_{slot} {{hashModel : MxxRuntime.HashModel}} {{external : ExternalInputs}} {{execution : Execution}}\n    (runs : Runs hashModel external execution) :\n{result_type} := {role}_sum_{slot}_at_bound runs _ rfl\n\n"));
            offset += width;
        }
    }
    source.push_str("end ExponentGeneratedRefreshSourceProof\n");
    Ok(source)
}
