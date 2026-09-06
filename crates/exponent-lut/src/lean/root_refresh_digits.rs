//! Integer native digits and exact positional refresh supports from the actual table.
use super::root_inputs::value_expression;
use crate::noise_refresh::simulation::ExponentLutNoiseSnapshot;
use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ValidatedGraph,
    lean::LeanArtifact,
    node::{ConstantMatrix, LoopInputMode, MatrixBinaryOp, NodeKind},
    types::WireRef,
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn root_refresh_digits(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
    snapshot: &ExponentLutNoiseSnapshot,
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
    let topology =
        snapshot.refresh.transcript_topology.as_ref().ok_or("missing refresh digit topology")?;
    if topology.coefficient_count != n ||
        snapshot.layout.width_log2 as usize >= usize::BITS as usize ||
        (1usize << snapshot.layout.width_log2) != width ||
        topology.mask_digit_counts != snapshot.refresh.mask_digit_counts ||
        topology.fresh_digit_count != snapshot.refresh.fresh_digit_count
    {
        return Err("actual native table and refresh digit topology disagree".into())
    }
    let base = &snapshot.refresh.base_p;
    let columns = topology.component_columns;
    let fresh = snapshot.refresh.fresh_digit_count;
    let slots = snapshot.refresh.mask_digit_counts.len();
    if base < &2u8.into() || base % 2u8 != 0u8.into() || slots == 0 || fresh == 0 {
        return Err("invalid native digit radix/count".into())
    }
    let NodeKind::ConstantMatrix {
        value: ConstantMatrix::Polynomial { coefficients: literals },
        ..
    } = root.node(table.node).ok_or("missing native table")?.kind()
    else {
        return Err("native table is not actual literal polynomial".into())
    };
    let values = literals
        .iter()
        .map(|v| v.evaluate(&ParamEnv::default()).map_err(|_| "dynamic native table coefficient"))
        .collect::<Result<Vec<_>, _>>()?;
    if values.len() > n {
        return Err("native table literal exceeds ring dimension".into())
    }
    let canonical = (0..width)
        .map(|i| {
            let a = values.get(i).cloned().unwrap_or_default();
            ((a % q) + q) % q
        })
        .collect::<Vec<_>>();
    let minimum = num_bigint::BigInt::from(1) - num_bigint::BigInt::from(base / 2u8);
    let maximum = num_bigint::BigInt::from(base / 2u8);
    if canonical.iter().any(|v| v < &minimum || v > &maximum) {
        return Err("actual native table leaves signed refresh digit interval".into())
    }
    let table_value = value(table)?;
    let expansion=values.iter().enumerate().map(|(i,a)|format!("((({a} : Int) : ExactPoly {q} {n}) * AdjoinRoot.root (negacyclicModulus {n} (ZMod {q})) ^ (⟨{i}, by decide⟩ : Fin {n}).val)")).collect::<Vec<_>>().join(" + ");
    let expansion = if expansion.is_empty() { "0".into() } else { expansion };
    let basis=values.iter().enumerate().map(|(i,a)|format!("coefficient_integer_basis (q := {q}) (n := {n}) (by decide) ({a}) (⟨{i}, by decide⟩ : Fin {n})")).collect::<Vec<_>>().join(", ");
    let mask_counts = snapshot
        .refresh
        .mask_digit_counts
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let mut source = format!(
        "import RootReadoutFunction\nimport ExponentRefreshDigits\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedRefreshDigits\nopen Mxx.Primitives MxxExponentLut\nopen scoped BigOperators\n\nabbrev radix : Nat := {base}\nabbrev width : Nat := {width}\nabbrev freshCount : Nat := {fresh}\ndef maskCount : Fin {slots} → Nat := ![{mask_counts}]\nnoncomputable def half : ExactPoly {q} {n} := ({width} : ExactPoly {q} {n}) * ExponentGeneratedReadoutFunction.normalizationScalar\nnoncomputable def digit (e : Fin (2 * {width})) : Int := refreshNativeDigit ExponentGeneratedReadoutFunction.targetCoefficients e\n\n/-- The exact trace expression, definitionally the value used by native_readout_closed. -/\nnoncomputable def nativeValue (e : Fin (2 * {width})) : ExactPoly {q} {n} :=\n  if h : e.val < {width} then (ExponentGeneratedReadoutFunction.targetCoefficients ⟨e.val,h⟩ : ExactPoly {q} {n}) - half\n  else half - (ExponentGeneratedReadoutFunction.targetCoefficients ⟨e.val-{width},by omega⟩ : ExactPoly {q} {n})\n\n"
    );
    source.push_str(&format!("theorem table_range : ∀ i, 1 - (radix / 2 : Nat) ≤ ExponentGeneratedReadoutFunction.targetCoefficients i ∧\n    ExponentGeneratedReadoutFunction.targetCoefficients i ≤ (radix / 2 : Nat) := by\n  letI : Fact (1 < {q}) := ⟨by decide⟩\n  have expansion : ({table_value}) 0 0 = {expansion} := by\n    simp [MxxRuntime.matrixPolynomial]\n    ring\n"));
    for (i, a) in canonical.iter().enumerate() {
        source.push_str(&format!("  have table_{i} : ((({table_value}) 0 0).coeff (⟨{i},by decide⟩ : Fin {n})).val = {a} := by\n    rw [expansion]\n    simp only [Negacyclic.coeff_add, {basis}]\n    decide\n"));
    }
    source.push_str("  intro i\n  fin_cases i\n");
    for i in 0..width {
        source.push_str(&format!("  · dsimp only [ExponentGeneratedReadoutFunction.targetCoefficients]\n    rw [table_{i}]\n    decide\n"));
    }
    source.push_str(&format!(r#"
/-- No existential is introduced: e is exactly the caller's native readout choice. -/
theorem integer_lift (e : Fin (2 * {width})) :
    reducePoly {q} {n} (digit e : ErrorPoly {n}) = nativeValue e + half := by
  exact refresh_native_digit_reduction _ e half ExponentGeneratedReadoutFunction.half_correct

theorem digit_range (e : Fin (2 * {width})) :
    1 - (radix / 2 : Nat) ≤ digit e ∧ digit e ≤ (radix / 2 : Nat) :=
  refresh_native_digit_range radix _ e table_range

noncomputable def freshValue (selection : Fin freshCount → Fin (2 * {width})) : Int :=
  ∑ i, (radix : Int) ^ i.val * digit (selection i)

noncomputable def maskValue (slot : Fin {slots})
    (selection : Fin (maskCount slot) → Fin (2 * {width})) : Int :=
  (∑ i, (radix : Int) ^ i.val * (digit (selection i) - (1 - (radix / 2 : Nat)))) -
    ((radix ^ maskCount slot / 2 : Nat) : Int)

theorem fresh_bound (selection : Fin freshCount → Fin (2 * {width})) :
    (freshValue selection).natAbs ≤ radix ^ freshCount - 1 :=
  refresh_fresh_radix_bound radix (by decide) _ (fun i ↦ digit_range (selection i))

theorem mask_bound (slot : Fin {slots}) (selection : Fin (maskCount slot) → Fin (2 * {width})) :
    (maskValue slot selection).natAbs ≤ radix ^ maskCount slot / 2 :=
  refresh_mask_radix_bound radix (by decide) (by decide) _ (fun i ↦ digit_range (selection i))

noncomputable def freshInteger
    (selection : Fin 1 → Fin {columns} → Fin {n} → Fin freshCount → Fin (2 * {width})) :
    ErrorMatrix {n} 1 {columns} := refreshFreshLift fun row col coefficient ↦ freshValue (selection row col coefficient)

noncomputable def maskInteger (slot : Fin {slots})
    (selection : Fin 1 → Fin {columns} → Fin {n} → Fin (maskCount slot) → Fin (2 * {width})) :
    ErrorMatrix {n} 1 {columns} := refreshFreshLift fun row col coefficient ↦ maskValue slot (selection row col coefficient)

theorem fresh_integer_bound
    (selection : Fin 1 → Fin {columns} → Fin {n} → Fin freshCount → Fin (2 * {width})) :
    CoeffBound (freshInteger selection) (radix ^ freshCount - 1) :=
  refreshFreshLift_bounded (by decide) _ (fun row col coefficient ↦ fresh_bound (selection row col coefficient))

theorem mask_integer_bound (slot : Fin {slots})
    (selection : Fin 1 → Fin {columns} → Fin {n} → Fin (maskCount slot) → Fin (2 * {width})) :
    CoeffBound (maskInteger slot selection) (radix ^ maskCount slot / 2) :=
  refreshFreshLift_bounded (by decide) _ (fun row col coefficient ↦ mask_bound slot (selection row col coefficient))

/-- These are the public_half_offset_vector scalars, before their all-ones
polynomial and route scale are applied. Negative native digits are retained. -/
noncomputable def freshOffset : ExactPoly {q} {n} :=
  (∑ i : Fin freshCount, (radix : ExactPoly {q} {n}) ^ i.val) * half
noncomputable def maskOffset (slot : Fin {slots}) : ExactPoly {q} {n} :=
  (∑ i : Fin (maskCount slot), (radix : ExactPoly {q} {n}) ^ i.val) *
    (half - ((1 - (radix / 2 : Nat) : Int) : ExactPoly {q} {n})) -
      ((radix ^ maskCount slot / 2 : Nat) : ExactPoly {q} {n})

theorem fresh_offset_identity (selection : Fin freshCount → Fin (2 * {width})) :
    (freshValue selection : ExactPoly {q} {n}) =
      (∑ i, (radix : ExactPoly {q} {n}) ^ i.val * nativeValue (selection i)) + freshOffset := by
  have digitCast (e : Fin (2 * {width})) : (digit e : ExactPoly {q} {n}) = nativeValue e + half := by
    simpa using integer_lift e
  simp only [freshValue, Int.cast_sum, Int.cast_mul, Int.cast_pow, Int.cast_natCast, digitCast,
    mul_add, Finset.sum_add_distrib, freshOffset, Finset.sum_mul]

theorem mask_offset_identity (slot : Fin {slots})
    (selection : Fin (maskCount slot) → Fin (2 * {width})) :
    (maskValue slot selection : ExactPoly {q} {n}) =
      (∑ i, (radix : ExactPoly {q} {n}) ^ i.val * nativeValue (selection i)) + maskOffset slot := by
  have digitCast (e : Fin (2 * {width})) : (digit e : ExactPoly {q} {n}) = nativeValue e + half := by
    simpa using integer_lift e
  simp only [maskValue, Int.cast_sub, Int.cast_sum, Int.cast_mul, Int.cast_pow, Int.cast_natCast,
    digitCast, mul_sub, mul_add, Finset.sum_sub_distrib, Finset.sum_add_distrib,
    maskOffset, Finset.sum_mul]
  ring

end ExponentGeneratedRefreshDigits
"#));
    Ok(source)
}
