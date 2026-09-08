//! Exact signed refresh ideals, including every actual public offset coefficient.
use super::root_inputs::value_expression;
use crate::bgg::noise::BggNoiseSnapshot;
use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ValidatedGraph,
    lean::LeanArtifact,
    node::{ConcatAxis, ConstantMatrix, MatrixBinaryOp, NodeKind},
    types::WireRef,
};

pub(super) fn root_refresh_normalize(
    graph: &ValidatedGraph,
    artifact: &LeanArtifact,
    snapshot: &BggNoiseSnapshot,
) -> Result<String, String> {
    let root = graph.source.root_scope();
    let shape = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
    let types = &graph.scopes[&FrozenGraphScopeId::Root].wire_types;
    let value = |w: WireRef| {
        value_expression(&format!("w_{}_{}", w.node.0, w.port.0), &shape.value_expressions)
    };
    let args = |w: WireRef| {
        root.node(w.node).and_then(|n| root.arguments(n)).ok_or("missing normalization operands")
    };
    let crt = root
        .nodes()
        .iter()
        .find(|n| matches!(n.kind(), NodeKind::CrtRecompose { .. }))
        .ok_or("missing normalization CRT")?;
    let NodeKind::CrtRecompose { plaintext_moduli, .. } = crt.kind() else { unreachable!() };
    let levels = root.arguments(crt).ok_or("missing normalization levels")?;
    if levels.len() != snapshot.refresh.mask_digit_counts.len() {
        return Err("normalization slot mismatch".into());
    }
    let mut source = format!(
        "import {}\nimport RootRefreshDigits\nimport ExponentRefreshNormalize\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\nnamespace ExponentGeneratedRefreshNormalize\nopen Mxx.Primitives MxxExponentLut\nopen scoped BigOperators\n\n",
        artifact.module_name
    );
    for (slot, level) in levels.iter().enumerate() {
        let NodeKind::FamilyGetStatic { index } =
            root.node(level.node).ok_or("missing level")?.kind()
        else {
            return Err("normalization level not gather".into())
        };
        let local = index
            .evaluate(&ParamEnv::default())
            .map_err(|_| "dynamic normalization slot")?
            .to_string()
            .parse::<usize>()
            .map_err(|_| "invalid slot")?;
        let family = args(*level)?[0];
        let id = FrozenGraphScopeId::ParallelBody {
            parent: Box::new(FrozenGraphScopeId::Root),
            owner: family.node,
        };
        let group = &graph.source.scopes()[&id];
        let ga = |w: WireRef| {
            group
                .node(w.node)
                .and_then(|n| group.arguments(n))
                .ok_or("missing grouped normalization operands")
        };
        let diff = ga(group.outputs()[0])?;
        if !matches!(
            group.node(diff[0].node).map(|n| n.kind()),
            Some(NodeKind::ModulusSwitch { .. })
        ) {
            return Err("normalization missing switch".into());
        }
        let final_terms = ga(ga(diff[0])?[0])?;
        let first = ga(final_terms[0])?;
        let mut offsets = Vec::new();
        for term in [first[1], final_terms[1]] {
            let capture = group
                .inputs()
                .iter()
                .position(|w| *w == term)
                .ok_or("normalization term not captured")?;
            let pack = args(family)?[capture];
            if !matches!(root.node(pack.node).map(|n| n.kind()), Some(NodeKind::FamilyPack { .. }))
            {
                return Err("normalization term not packed".into());
            }
            let actual = *args(pack)?.get(local).ok_or("normalization slot outside pack")?;
            if !matches!(
                root.node(actual.node).map(|n| n.kind()),
                Some(NodeKind::MatrixBinary(MatrixBinaryOp::Add))
            ) {
                return Err("normalization missing public addition".into());
            }
            offsets.push(args(actual)?[1]);
        }
        let mt = types[&offsets[0]].matrix_type().ok_or("normalization offset not matrix")?;
        let q = &mt.modulus;
        let n = mt.ring_dimension;
        let columns = mt.columns;
        let p = &types[level].matrix_type().ok_or("normalization decoder not matrix")?.modulus;
        let plaintext = plaintext_moduli[slot]
            .evaluate(&ParamEnv::default())
            .map_err(|_| "dynamic plaintext")?;
        let j = q / p;
        let delta = p / &plaintext;
        let mu = q / &plaintext;
        if q % p != 0u8.into() || p % &plaintext != 0u8.into() || &j * &delta != mu {
            return Err("normalization nondivisible scales".into());
        }
        let inverse_two = (q + 1u8) / 2u8;
        for (role, w, scale) in [("mask", offsets[0], &j), ("fresh", offsets[1], &mu)] {
            let offset = if role == "mask" {
                format!("ExponentGeneratedRefreshDigits.maskOffset ⟨{slot}, by decide⟩")
            } else {
                "ExponentGeneratedRefreshDigits.freshOffset".into()
            };
            if !matches!(
                root.node(w.node).map(|n| n.kind()),
                Some(NodeKind::MatrixBinary(MatrixBinaryOp::Multiply))
            ) {
                return Err("offset scale is not multiplication".into());
            }
            let factors = args(w)?;
            if factors.len() != 2 || columns < 2 {
                return Err("offset product or replicated column arity".into());
            }
            let concat = factors[1];
            if !matches!(
                root.node(concat.node).map(|n| n.kind()),
                Some(NodeKind::Concat { axis: ConcatAxis::Columns })
            ) {
                return Err("offset not replicated concatenation".into());
            }
            let pieces = args(concat)?;
            if pieces.len() != columns || pieces.iter().any(|p| *p != pieces[0]) {
                return Err("offset columns not identical".into());
            }
            let base_value = value(pieces[0])?;
            let base_args = args(pieces[0])?;
            let NodeKind::ConstantMatrix {
                value: ConstantMatrix::Polynomial { coefficients }, ..
            } = root.node(base_args[0].node).ok_or("missing offset scalar")?.kind()
            else {
                return Err("offset scalar not polynomial literal".into())
            };
            if coefficients.len() != 1 {
                return Err("offset scalar not constant".into());
            }
            let scalar = coefficients[0]
                .evaluate(&ParamEnv::default())
                .map_err(|_| "dynamic offset scalar")?;
            let constraints = shape
                .node_constraints
                .get(&concat.node)
                .ok_or("offset concat constraints missing")?;
            if constraints.len() != columns - 1 {
                return Err("offset concat chain length".into());
            }
            let count = if role == "mask" {
                snapshot.refresh.mask_digit_counts[slot]
            } else {
                snapshot.refresh.fresh_digit_count
            };
            let count_expr = if role == "mask" {
                format!("ExponentGeneratedRefreshDigits.maskCount ⟨{slot}, by decide⟩")
            } else {
                "ExponentGeneratedRefreshDigits.freshCount".into()
            };
            let base = &snapshot.refresh.base_p;
            let digit_sum = (base
                .pow(u32::try_from(count).map_err(|_| "normalization digit count overflow")?) -
                1u8) /
                (base - 1u8);
            let mut proof = format!(
                "  have numeric : ({inverse_two} : ZMod {q}) * 2 = 1 := by decide\n  have candidate : ({inverse_two} : ExactPoly {q} {n}) * 2 = 1 := by\n    simpa using congrArg (algebraMap (ZMod {q}) (ExactPoly {q} {n})) numeric\n  have halfCast : ExponentGeneratedRefreshDigits.half = ({inverse_two} : ExactPoly {q} {n}) := by\n    calc\n      _ = ExponentGeneratedRefreshDigits.half * (({inverse_two} : ExactPoly {q} {n}) * 2) := by rw [candidate, mul_one]\n      _ = ({inverse_two} : ExactPoly {q} {n}) * (ExponentGeneratedRefreshDigits.half * 2) := by ring\n      _ = _ := by rw [show ExponentGeneratedRefreshDigits.half * 2 = 1 from ExponentGeneratedReadoutFunction.half_correct, mul_one]\n  have integerSum : (∑ d : Fin ({count_expr}), (ExponentGeneratedRefreshDigits.radix : Int) ^ d.val) = {digit_sum} := by decide\n  have polySum : (∑ d : Fin ({count_expr}), (ExponentGeneratedRefreshDigits.radix : ExactPoly {q} {n}) ^ d.val) = ({digit_sum} : ExactPoly {q} {n}) := by\n    simpa only [Int.cast_sum, Int.cast_pow, Int.cast_natCast, Int.cast_ofNat] using congrArg (fun z : Int ↦ (z : ExactPoly {q} {n})) integerSum\n  have offsetScalar : ({offset}) = ({scalar} : ExactPoly {q} {n}) := by\n    simp only [ExponentGeneratedRefreshDigits.{role}Offset, halfCast]\n    rw [polySum]\n    norm_num [ExponentGeneratedRefreshDigits.maskCount, ExponentGeneratedRefreshDigits.freshCount, ExponentGeneratedRefreshDigits.radix, Fin.sum_univ_succ]\n  let common : ExactPoly {q} {n} := ∑ k : Fin {n}, ({offset}) * AdjoinRoot.root (negacyclicModulus {n} (ZMod {q})) ^ k.val\n  have baseColumn : ∀ row col, ({base_value}) row col = common := by\n    intro row col\n    dsimp only [common]\n    rw [offsetScalar]\n    norm_num [MxxRuntime.matrixMulScalarLeft, MxxRuntime.matrixPolynomial, Fin.sum_univ_succ, pow_succ]\n    ring\n"
            );
            proof.push_str(&format!(
                "  have concatFacts := facts{}\n",
                ".2".repeat(constraints.start)
            ));
            for i in 0..constraints.len() {
                let left = if i == 0 { "baseColumn".into() } else { format!("column{}", i - 1) };
                proof.push_str(&format!(
                    "  have column{i} := refresh_concat_constant common (concatFacts{}.1) {left} baseColumn\n",
                    ".2".repeat(i)
                ));
            }
            proof.push_str(&format!("  ext row col\n  simp only [{role}PublicOffset_{slot}, MxxRuntime.matrixMulScalarLeft, column{}, common, refreshOffsetMatrix, Matrix.smul_apply]\n  norm_num [MxxRuntime.matrixPolynomial, MxxIR.exactDiv, zsmul_eq_mul]\n",columns-2));
            source.push_str(&format!("noncomputable def {role}PublicOffset_{slot} (witness : {record}) : ExactMatrix {q} {n} 1 {columns} := {actual}\n\ntheorem {role}_public_offset_{slot} {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}} {{params : {params}}} {{inputs : {inputs}}} {{outputs : {outputs}}} (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness) :\n    {role}PublicOffset_{slot} witness = ({scale} : Int) • refreshOffsetMatrix (q := {q}) (n := {n}) (columns := {columns}) ({offset}) := by\n{proof}\n",params=artifact.root.parameter_type,inputs=artifact.root.input_type,outputs=artifact.root.output_type,record=shape.witness_record.as_ref().ok_or("missing normalization witness")?,body=shape.body_relation.as_ref().ok_or("missing normalization body")?,actual=value(w)?));
        }
        let width = 1usize << snapshot.layout.width_log2;
        let selection = format!(
            "Fin 1 → Fin {columns} → Fin {n} → Fin ExponentGeneratedRefreshDigits.freshCount → Fin (2 * {width})"
        );
        let mask_selection = format!(
            "Fin 1 → Fin {columns} → Fin {n} → Fin (ExponentGeneratedRefreshDigits.maskCount ⟨{slot}, by decide⟩) → Fin (2 * {width})"
        );
        source.push_str(&format!(r#"
noncomputable def freshNative_{slot} (selection : {selection}) : ExactMatrix {q} {n} 1 {columns} :=
  refreshNativeRadixMatrix ExponentGeneratedRefreshDigits.radix ExponentGeneratedRefreshDigits.nativeValue selection
noncomputable def maskNative_{slot} (selection : {mask_selection}) : ExactMatrix {q} {n} 1 {columns} :=
  refreshNativeRadixMatrix ExponentGeneratedRefreshDigits.radix ExponentGeneratedRefreshDigits.nativeValue selection

theorem fresh_assembly_{slot} (selection : {selection}) :
    reduceMatrix {q} {n} 1 {columns} (ExponentGeneratedRefreshDigits.freshInteger selection) =
      freshNative_{slot} selection + refreshOffsetMatrix ExponentGeneratedRefreshDigits.freshOffset := by
  ext row col
  exact refresh_integer_assembly
    (fun row col k ↦ ExponentGeneratedRefreshDigits.freshValue (selection row col k))
    (fun row col k ↦ ∑ d, (ExponentGeneratedRefreshDigits.radix : ExactPoly {q} {n}) ^ d.val * ExponentGeneratedRefreshDigits.nativeValue (selection row col k d))
    ExponentGeneratedRefreshDigits.freshOffset
    (fun row col k ↦ ExponentGeneratedRefreshDigits.fresh_offset_identity (selection row col k)) row col

theorem mask_assembly_{slot} (selection : {mask_selection}) :
    reduceMatrix {q} {n} 1 {columns} (ExponentGeneratedRefreshDigits.maskInteger ⟨{slot}, by decide⟩ selection) =
      maskNative_{slot} selection + refreshOffsetMatrix (ExponentGeneratedRefreshDigits.maskOffset ⟨{slot}, by decide⟩) := by
  ext row col
  exact refresh_integer_assembly
    (fun row col k ↦ ExponentGeneratedRefreshDigits.maskValue ⟨{slot}, by decide⟩ (selection row col k))
    (fun row col k ↦ ∑ d, (ExponentGeneratedRefreshDigits.radix : ExactPoly {q} {n}) ^ d.val * ExponentGeneratedRefreshDigits.nativeValue (selection row col k d))
    (ExponentGeneratedRefreshDigits.maskOffset ⟨{slot}, by decide⟩)
    (fun row col k ↦ ExponentGeneratedRefreshDigits.mask_offset_identity ⟨{slot}, by decide⟩ (selection row col k)) row col

theorem high_ideal_factor_{slot} {{backend : MxxRuntime.BackendContext}} {{hashModel : MxxRuntime.HashModel}}
    {{params : {params}}} {{inputs : {inputs}}} {{outputs : {outputs}}}
    (witness : {record}) (facts : {body} backend hashModel params inputs outputs witness)
    (payload : ErrorMatrix {n} 1 {columns}) (fresh : {selection}) (mask : {mask_selection}) :
    -({mu} : Int) • reduceMatrix {q} {n} 1 {columns} payload +
      ({j} : Int) • maskNative_{slot} mask + maskPublicOffset_{slot} witness +
      ({mu} : Int) • freshNative_{slot} fresh + freshPublicOffset_{slot} witness =
    reduceMatrix {q} {n} 1 {columns} (({j} : Int) • (({delta} : Int) •
      (-payload + ExponentGeneratedRefreshDigits.freshInteger fresh) +
        ExponentGeneratedRefreshDigits.maskInteger ⟨{slot}, by decide⟩ mask)) := by
  rw [mask_public_offset_{slot} witness facts, fresh_public_offset_{slot} witness facts]
  exact refresh_high_ideal_factor ({j} : Int) ({delta} : Int) payload _ _ _ _ _ _
    (fresh_assembly_{slot} fresh) (mask_assembly_{slot} mask)

theorem decoder_ideal_coeff_{slot} (secret : ErrorMatrix {n} 1 2)
    (publicKey : ErrorMatrix {n} 2 {columns}) (payload : ErrorMatrix {n} 1 {columns})
    (fresh : {selection}) (mask : {mask_selection}) (row col coefficient) :
    ((reduceMatrix {p} {n} 1 {columns} (({delta} : Int) •
        (-payload + ExponentGeneratedRefreshDigits.freshInteger fresh) +
          ExponentGeneratedRefreshDigits.maskInteger ⟨{slot}, by decide⟩ mask) +
      reduceMatrix {p} {n} 1 2 secret * (({delta} : Int) • reduceMatrix {p} {n} 2 {columns} publicKey))
      row col).coeff coefficient =
      (({delta} : Int) * ((secret * publicKey - payload + ExponentGeneratedRefreshDigits.freshInteger fresh) row col).coeff coefficient +
        (ExponentGeneratedRefreshDigits.maskInteger ⟨{slot}, by decide⟩ mask row col).coeff coefficient : Int) := by
  exact refresh_decoder_ideal_coeff (by decide) (by decide) _ _ _ _ _ _ _ _ _

"#,params=artifact.root.parameter_type,inputs=artifact.root.input_type,outputs=artifact.root.output_type,record=shape.witness_record.as_ref().ok_or("missing normalization witness")?,body=shape.body_relation.as_ref().ok_or("missing normalization body")?));
    }
    source.push_str("end ExponentGeneratedRefreshNormalize\n");
    Ok(source)
}
