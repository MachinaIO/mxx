//! The native trace target, reconstructed from the producer's actual coefficient reads.
use std::collections::{BTreeMap, BTreeSet};

use super::root_inputs::value_expression;
use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ValidatedGraph,
    lean::{BoundaryValue, LeanArtifact},
    node::{ConstantMatrix, LoopInputMode, MatrixBinaryOp, NodeKind},
    types::WireRef,
};

pub(super) fn root_readout_function(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
) -> Result<String, String> {
    let root = graph.source.root_scope();
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let types = &graph.scopes[&FrozenGraphScopeId::Root].wire_types;
    let value = |wire: WireRef| {
        value_expression(&format!("w_{}_{}", wire.node.0, wire.port.0), &shape.value_expressions)
    };
    let scalar = |suffix: &str| -> Result<WireRef, String> {
        let output = artifact
            .root
            .outputs
            .iter()
            .find(|(name, _)| name.ends_with(suffix))
            .ok_or_else(|| format!("missing actual readout output {suffix}"))?
            .1
            .wire;
        let node = root.node(output.node).ok_or("missing readout output node")?;
        let id = FrozenGraphScopeId::ParallelBody {
            parent: Box::new(FrozenGraphScopeId::Root),
            owner: output.node,
        };
        let (scope, scope_types, decomposition) =
            if matches!(node.kind(), NodeKind::ParallelLoop(_)) {
                let scope = &graph.source.scopes()[&id];
                if scope.outputs().len() != 1 {
                    return Err("readout decomposition output arity".into());
                }
                (scope, &graph.scopes[&id].wire_types, scope.outputs()[0])
            } else {
                (root, types, output)
            };
        let decomp = scope.node(decomposition.node).ok_or("missing readout decomposition")?;
        if !matches!(decomp.kind(), NodeKind::GadgetDecompose { .. }) {
            return Err("readout output not gadget decomposition".into());
        }
        let product = scope.arguments(decomp).ok_or("missing decomposition operand")?[0];
        let product = scope.node(product.node).ok_or("missing readout product")?;
        if !matches!(product.kind(), NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)) {
            return Err("readout decomposition target not multiplication".into());
        }
        let operands = scope.arguments(product).ok_or("missing readout factors")?;
        let scalar = *operands
            .iter()
            .find(|wire| {
                scope_types[wire].matrix_type().is_some_and(|m| m.rows == 1 && m.columns == 1)
            })
            .ok_or("missing readout scalar")?;
        if let NodeKind::ParallelLoop(loop_) = node.kind() {
            let slot = scope
                .inputs()
                .iter()
                .position(|input| *input == scalar)
                .ok_or("target scalar not captured")?;
            if loop_.input_modes[slot] != LoopInputMode::Broadcast {
                return Err("target scalar not broadcast".into());
            }
            Ok(root.arguments(node).ok_or("missing target captures")?[slot])
        } else {
            Ok(scalar)
        }
    };
    let target = scalar("-readout-target-decomposition-0")?;
    let normalization = scalar("-readout-normalization-decomposition-0")?;
    let matrix = types[&target].matrix_type().ok_or("target not matrix")?;
    let q = &matrix.modulus;
    let n = matrix.ring_dimension;
    let mut pending = vec![target];
    let mut visited = BTreeSet::new();
    let mut coefficients = BTreeMap::new();
    let mut table = None;
    while let Some(wire) = pending.pop() {
        if !visited.insert(wire) {
            continue;
        }
        let node = root.node(wire.node).ok_or("missing target ancestor")?;
        if let NodeKind::ExtractCoefficient { position, .. } = node.kind() {
            let position = position
                .evaluate(&ParamEnv::default())
                .map_err(|_| "dynamic target coefficient")?
                .to_string()
                .parse::<usize>()
                .map_err(|_| "negative target position")?;
            let input = root.arguments(node).ok_or("missing target table")?[0];
            if table.replace(input).is_some_and(|previous| previous != input) {
                return Err("target mixes tables".into());
            }
            if coefficients.insert(position, wire).is_some() {
                return Err("duplicate target coefficient read".into());
            }
        } else if let Some(arguments) = root.arguments(node) {
            pending.extend(arguments);
        }
    }
    let table = table.ok_or("target has no coefficient reads")?;
    let width = coefficients.len();
    if width == 0 || n % width != 0 || coefficients.keys().copied().ne(0..width) {
        return Err("target coefficient positions not native width".into());
    }
    let stride = n / width;
    let boundary = |values: &BTreeMap<String, BoundaryValue>, count: usize| {
        let mut types = vec![String::new(); count];
        for value in values.values() {
            types[value.tuple_index] = format!("({})", value.lean_type);
        }
        match count {
            0 => "Unit".into(),
            1 => types.remove(0),
            _ => types.join(" × ") + " × Unit",
        }
    };
    let body = shape.body_relation.as_ref().ok_or("missing readout producer body")?;
    let record = shape.witness_record.as_ref().ok_or("missing readout producer witness")?;
    let input_type = boundary(&artifact.root.inputs, artifact.root.input_count);
    let output_type = boundary(&artifact.root.outputs, artifact.root.output_count);
    let target_value = value(target)?;
    let table_value = value(table)?;
    let normalization_value = value(normalization)?;
    let NodeKind::ConstantMatrix {
        value: ConstantMatrix::Polynomial { coefficients: inverse_coefficients },
        ..
    } = root.node(normalization.node).ok_or("missing inverse literal")?.kind()
    else {
        return Err("normalization scalar is not literal polynomial".into());
    };
    if inverse_coefficients.len() != 1 {
        return Err("normalization polynomial not constant".into());
    }
    let inverse = inverse_coefficients[0]
        .evaluate(&ParamEnv::default())
        .map_err(|_| "dynamic normalization inverse")?;
    let binders = format!(
        "{{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}\n    {{params : {}}} {{inputs : RootInput}} {{outputs : RootOutput}}\n    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness)",
        artifact.root.parameter_type
    );
    let table_function = format!(
        "(fun k : Fin {width} => (Int.ofNat ((({table_value}) 0 0).coeff ⟨k.val, by omega⟩).val))"
    );
    let positions = (0..width)
        .map(|k| format!("⟨{}, by decide⟩", if k == 0 { 0 } else { n - stride * k }))
        .collect::<Vec<_>>()
        .join(", ");
    let definitions = format!(
        "noncomputable def targetScalar (witness : {record}) : ExactPoly {q} {n} := ({target_value}) 0 0\nnoncomputable def targetCoefficients : Fin {width} → Int := {table_function}\nnoncomputable def normalizationScalar : ExactPoly {q} {n} := ({normalization_value}) 0 0\nnoncomputable def targetInteger : ErrorPoly {n} :=\n  let coefficient : Fin {width} → Int := fun k => 2 * ({table_function}) k - 1\n  let positions : Fin {width} → Fin {n} := ![{positions}]\n  let signed : Fin {width} → Int := fun k => if k.val = 0 then coefficient k else -coefficient k\n  ∑ k, scaledBasis (signed k) (positions k)\n\n"
    );
    let mut source = format!(
        "import {}\nimport TraceIndices\nimport ExponentReadoutFunction\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\n\nnamespace ExponentGeneratedReadoutFunction\nopen Mxx.Primitives MxxExponentLut\nopen scoped BigOperators\nabbrev RootInput := {input_type}\nabbrev RootOutput := {output_type}\n\ntheorem dimensions : TraceIndices.ringDimension = {n} ∧ TraceIndices.width = {width} ∧ TraceIndices.stride = {stride} := by decide\n\n{definitions}theorem target_identity {binders} :\n    ({target_value}) 0 0 = nativeTraceTarget (q := {q}) (n := {n}) (stride := {stride}) {table_function} := by\n",
        artifact.module_name
    );
    let mut rewrites = Vec::new();
    let mut powers = String::new();
    for (position, wire) in coefficients {
        let index = shape
            .node_constraints
            .get(&wire.node)
            .and_then(|range| range.clone().last())
            .ok_or("missing extraction relation")?;
        let coefficient = value(wire)?;
        source.push_str(&format!("  have coefficient_{position} : ({coefficient}) = (Int.ofNat ((({table_value}) 0 0).coeff ⟨{position}, by decide⟩).val) := by\n    obtain ⟨index, hi, hv⟩ := facts{}.1\n    have same : index = (⟨{position}, by decide⟩ : Fin {n}) := by\n      apply Fin.ext\n      change index.val = {position}\n      exact_mod_cast hi\n    simpa only [same] using hv\n", ".2".repeat(index)));
        rewrites.push(format!("coefficient_{position}"));
        let exponent = stride * (2 * width - position);
        let power_start = source.len();
        if position == 0 {
            source.push_str(&format!("  have power_{position} : (AdjoinRoot.root (negacyclicModulus {n} (ZMod {q}))) ^ {exponent} = 1 := by\n    change (AdjoinRoot.root (negacyclicModulus {n} (ZMod {q}))) ^ ({n} + {n}) = 1\n    rw [pow_add, Negacyclic.root_pow_n]\n    ring\n"));
        } else {
            let rotation = exponent - n;
            source.push_str(&format!("  have power_{position} : (AdjoinRoot.root (negacyclicModulus {n} (ZMod {q}))) ^ {exponent} = -(AdjoinRoot.root (negacyclicModulus {n} (ZMod {q}))) ^ {rotation} := by\n    change (AdjoinRoot.root (negacyclicModulus {n} (ZMod {q}))) ^ ({n} + {rotation}) = _\n    rw [pow_add, Negacyclic.root_pow_n]\n    ring\n"));
        }
        powers.push_str(&source[power_start..]);
        rewrites.push(format!("power_{position}"));
    }
    source.push_str(&format!("  simp [nativeTraceTarget, Fin.sum_univ_succ, MxxRuntime.matrixAdd, Matrix.mul_apply, MxxRuntime.matrixNeg, MxxRuntime.liftInteger, MxxRuntime.rotationPolynomial, {}, add_assoc, mul_comm]\n\ntheorem normalization_identity :\n    (({normalization_value}) 0 0) * ({width} * 2 : ExactPoly {q} {n}) = 1 := by\n  have modular : ({inverse} : ZMod {q}) * ({width} * 2) = 1 := by decide\n  have lifted := congrArg (algebraMap (ZMod {q}) (ExactPoly {q} {n})) modular\n  simpa [MxxRuntime.matrixPolynomial] using lifted\n\n",rewrites.join(", ")));
    let power_names = (0..width).map(|k| format!("power_{k}")).collect::<Vec<_>>().join(", ");
    source.push_str(&format!("theorem half_identity :\n    (({width} : ExactPoly {q} {n}) * (({normalization_value}) 0 0)) * 2 = 1 := by\n  calc\n    _ = (({normalization_value}) 0 0) * ({width} * 2) := by ring\n    _ = 1 := normalization_identity\n\ntheorem target_lift_sum {binders} :\n    reducePoly {q} {n} targetInteger = targetScalar witness ∧\n      polynomialL1 targetInteger ≤ ∑ k : Fin {width}, (2 * ({table_function}) k - 1).natAbs := by\n  classical\n  let coefficient : Fin {width} → Int := fun k => 2 * ({table_function}) k - 1\n  let positions : Fin {width} → Fin {n} := ![{positions}]\n  let signed : Fin {width} → Int := fun k => if k.val = 0 then coefficient k else -coefficient k\n  change reducePoly {q} {n} (∑ k, scaledBasis (signed k) (positions k)) = ({target_value}) 0 0 ∧ polynomialL1 (∑ k, scaledBasis (signed k) (positions k)) ≤ _\n  constructor\n  · rw [target_identity witness facts]\n"));
    source.push_str(&powers.lines().map(|line| format!("  {line}\n")).collect::<String>());
    source.push_str(&format!("    simp [reduce_scaled_basis, nativeTraceTarget, Fin.sum_univ_succ, signed, coefficient, positions, {power_names}, add_assoc]\n    ring\n  · refine (target_basis_gain (by decide) signed positions).trans_eq ?_\n    apply Finset.sum_congr rfl\n    intro k hk\n    dsimp only [signed, coefficient]\n    split_ifs <;> simp only [Int.natAbs_neg]\n\n"));
    // The numeric gain is read from the same literal polynomial that feeds the
    // actual ExtractCoefficient nodes, then checked again by Lean coefficient extraction.
    let NodeKind::ConstantMatrix {
        value: ConstantMatrix::Polynomial { coefficients: literals },
        ..
    } = root.node(table.node).ok_or("missing actual table declaration")?.kind()
    else {
        return Err("numeric target gain needs canonical literal table".into());
    };
    let literal_values = literals
        .iter()
        .map(|value| {
            value.evaluate(&ParamEnv::default()).map_err(|_| "dynamic target table literal")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let modulus = num_bigint::BigInt::from(q.clone());
    let canonical_values = (0..width)
        .map(|k| {
            let value = literal_values.get(k).cloned().unwrap_or_default();
            ((value % &modulus) + &modulus) % &modulus
        })
        .collect::<Vec<_>>();
    let gain = crate::noise::RightActionNoiseGain::from_polynomial_coefficients(
        canonical_values.iter().map(|value| value * 2u8 - 1u8),
    )
    .l1;
    let expansion = literal_values.iter().enumerate().map(|(i, a)| format!("((({a} : Int) : ExactPoly {q} {n}) * AdjoinRoot.root (negacyclicModulus {n} (ZMod {q})) ^ (⟨{i}, by decide⟩ : Fin {n}).val)")).collect::<Vec<_>>().join(" + ");
    let expansion = if expansion.is_empty() { "0".to_owned() } else { expansion };
    source.push_str(&format!("theorem target_gain_value :\n    (∑ k : Fin {width}, (2 * ({table_function}) k - 1).natAbs) = {gain} := by\n  letI : Fact (1 < {q}) := ⟨by decide⟩\n  have expansion : ({table_value}) 0 0 = {expansion} := by\n    simp [MxxRuntime.matrixPolynomial]\n    ring\n"));
    let basis_rewrites = literal_values.iter().enumerate().map(|(i,a)| format!("coefficient_integer_basis (q := {q}) (n := {n}) (by decide) ({a}) (⟨{i}, by decide⟩ : Fin {n})")).collect::<Vec<_>>().join(", ");
    let mut table_rewrites = Vec::new();
    for (k, canonical) in canonical_values.iter().enumerate() {
        source.push_str(&format!("  have table_{k} : ((({table_value}) 0 0).coeff (⟨{k}, by decide⟩ : Fin {n})).val = {canonical} := by\n    rw [expansion]\n    simp only [Negacyclic.coeff_add, {basis_rewrites}]\n    decide\n"));
        table_rewrites.push(format!("table_{k}"));
    }
    let expanded_gain = (0..width).rev().fold("0".to_owned(), |tail, k| format!("(2 * Int.ofNat ((({table_value}) 0 0).coeff (⟨{k}, by decide⟩ : Fin {n})).val - 1).natAbs + ({tail})"));
    source.push_str(&format!("  simp only [Fin.sum_univ_succ]\n  change ({expanded_gain}) = {gain}\n  rw [{}]\n  decide\n\ntheorem target_lift {binders} :\n    reducePoly {q} {n} targetInteger = targetScalar witness ∧ polynomialL1 targetInteger ≤ {gain} := by\n  simpa only [target_gain_value] using target_lift_sum witness facts\n\ntheorem target_function {binders} :\n    targetScalar witness = nativeTraceTarget (q := {q}) (n := {n}) (stride := {stride}) targetCoefficients := target_identity witness facts\n\ntheorem normalization_correct : normalizationScalar * ({width} * 2 : ExactPoly {q} {n}) = 1 := normalization_identity\ntheorem half_correct : (({width} : ExactPoly {q} {n}) * normalizationScalar) * 2 = 1 := half_identity\n\nend ExponentGeneratedReadoutFunction\n", table_rewrites.join(", ")));
    Ok(source)
}
