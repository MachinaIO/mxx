import TopRefresh
import ExponentScale
import TopKeyForms
import TopDecompose

/-!
# The target action

The online stage maps the source plaintext `q₂ x g` (`g` the gadget of `q₀ q₁` re-encoded at `Q₁`)
to `Q_q T` by the digits `Y` of `X = c' W` at `q₀ q₁`, where `W` lifts `q₃ q₁ G mod q₀`. Since
`q₁ ∣ c'`, `q₂ c' = Q_q (c' / q₁)`, so `T = (c' / q₁) x W`; and `c' ≡ 1 (mod q₀)` makes
`T ≡ q₃ x G (mod q₀)`.
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh

/-- The target constant `c'` of the action. -/
abbrev actionConst : Int := 339699808396096934

/-- `Q_q = q₁ q₂` at `Q₁`. -/
noncomputable abbrev QqPoly : ExactPoly Q1 N := ((1152919668508929361 : Int) : ExactPoly Q1 N)

theorem actionConst_mul : (q2 : Int) * actionConst = 1152919668508929361 * (actionConst / q1) := by
  norm_num

set_option maxHeartbeats 4000000 in
/-- The source plaintext times the action digits is `Q_q T` with `T = (c' / q₁) x W`. -/
theorem action_mul {g' : ExactMatrix M01 N 1 4} (hg' : gadgetMatrixRuns Backend.backend 32768 4 g')
    {g : ExactMatrix Q1 N 1 4} (hg : centeredRebaseRuns g' g)
    {W : ErrorMatrix N 1 6} {X : ExactMatrix M01 N 1 6}
    (hX : X = reduceMatrix M01 N 1 6 (((actionConst : Int) : ErrorPoly N) • W))
    {Y' : ExactMatrix M01 N 4 6} (hY' : gadgetDecomposeRuns Backend.backend 32768 4 X Y')
    {Y : ExactMatrix Q1 N 4 6} {bound : Nat} (hY : compactCenteredRebaseRuns bound Y' Y)
    (x : ErrorPoly N) :
    (reducePoly Q1 N (((q2 : Nat) : Int) * x) • g) * Y =
      QqPoly • reduceMatrix Q1 N 1 6 ((((actionConst / q1 : Int) : ErrorPoly N) * x) • W) := by
  rw [gadget_mul_action_scaleUp_one (q' := M01) (p := q2) (by norm_num) (by norm_num) Backend.layout2
    layoutM01 rfl hg' hg hY' hY x, hX, ← reduceMatrix_smul]
  funext r c
  simp only [scaleUpMatrix, reduceMatrix, Matrix.smul_apply, smul_eq_mul]
  rw [scaleUp_reducePoly (by norm_num) (by norm_num) hN, QqPoly, ← reducePoly_intCast, ← map_mul]
  congr 1
  rw [← mul_assoc, ← mul_assoc, ← mul_assoc]
  congr 1
  have h := congrArg (fun z : Int ↦ (z : ErrorPoly N)) actionConst_mul
  simp only [Int.cast_mul] at h
  rw [show (((q2 : Nat) : Int) : ErrorPoly N) = ((q2 : Int) : ErrorPoly N) from rfl]
  linear_combination (x) * h

/-- A compact centered rebase within one modulus is the identity. -/
theorem compactRebase_self {q rows columns : Nat} (hq : 1 < q) {bound : Nat}
    {x y : ExactMatrix q N rows columns} (h : compactCenteredRebaseRuns bound x y) : y = x := by
  rw [centeredRebaseRuns_reduce h.2.2.1]
  exact reduceMatrix_centeredIntLift hq hN x

/-- The gadget of `Q₁` times the halves of the re-encoded digits of a two-row `z` are its rows. -/
theorem decompose_rows {z : ExactMatrix Q1 N (1 + 1) 6} {Dz' Dz : ExactMatrix Q1 N (6 + 6) 6}
    (hDz : gadgetDecomposeRuns Backend.backend 32768 6 z Dz')
    {bound : Nat} (hc : compactCenteredRebaseRuns bound Dz' Dz)
    {G : ExactMatrix Q1 N 1 6} (hG : gadgetMatrixRuns Backend.backend 32768 6 G) :
    G * topRows Dz = (fun _ c ↦ z 0 c) ∧ G * bottomRows Dz = (fun _ c ↦ z 1 c) := by
  rw [compactRebase_self (by norm_num) hc]
  have hz : concatRows (fun _ c ↦ z 0 c : ExactMatrix Q1 N 1 6) (fun _ c ↦ z 1 c) z := by
    intro row column
    fin_cases row <;> rfl
  obtain ⟨Dx, hDx⟩ := gadgetDecomposeRuns_exists (inner := 6) (backend := Backend.backend)
    (base := 32768) (digits := 6) Backend.layout4 layoutQ1 rfl (by norm_num) rfl rfl rfl
    (fun _ c ↦ z 0 c : ExactMatrix Q1 N 1 6)
  obtain ⟨Dy, hDy⟩ := gadgetDecomposeRuns_exists (inner := 6) (backend := Backend.backend)
    (base := 32768) (digits := 6) Backend.layout4 layoutQ1 rfl (by norm_num) rfl rfl rfl
    (fun _ c ↦ z 1 c : ExactMatrix Q1 N 1 6)
  obtain ⟨ht, hb⟩ := decompose_stack Backend.layout4 layoutQ1 rfl hz hDz hDx hDy
  rw [ht, hb]
  exact ⟨gadget_mul_decompose_one (by norm_num) Backend.layout4 layoutQ1 rfl hG hDx,
    gadget_mul_decompose_one (by norm_num) Backend.layout4 layoutQ1 rfl hG hDy⟩

/-- `T = (c' / q₁) x W` with `W` the centered lift of `w`. -/
noncomputable def actionValue (x : ErrorPoly N) (w : ExactMatrix q0 N 1 6) : ExactMatrix Q1 N 1 6 :=
  reduceMatrix Q1 N 1 6 ((((actionConst / q1 : Int) : ErrorPoly N) * x) •
    fun r c ↦ centeredIntLift (w r c))

set_option maxHeartbeats 16000000 in
/-- One output column of a decoder, from the runtime relations of the combination and the
semantic forms of its rows. -/
theorem column_value {g c₁ : ExactMatrix Q1 N 1 6} {gc : ExactMatrix Q1 N 1 (6 + 6)}
    (hgc : concatColumns g c₁ gc)
    {A' : ExactMatrix Q1 N (1 + 1) 6} {z : ExactMatrix Q1 N (1 + 1) 6}
    (hz : z = matrixMulScalarLeft (matrixPolynomial [1152919668508929361]) A')
    {Dz' D : ExactMatrix Q1 N (6 + 6) 6} (hDz : gadgetDecomposeRuns Backend.backend 32768 6 z Dz')
    (hD : compactCenteredRebaseRuns 16384 Dz' D)
    {G : ExactMatrix Q1 N 1 6} (hG : gadgetMatrixRuns Backend.backend 32768 6 G)
    {w350 : ExactMatrix q0 N 1 6} {w351 : ExactMatrix M01 N 1 6} (h351 : centeredRebaseRuns w350 w351)
    {X : ExactMatrix M01 N 1 6} (hX : X = matrixMulScalarRight w351 (matrixPolynomial [339699808396096934]))
    {Y' : ExactMatrix M01 N 4 6} (hY' : gadgetDecomposeRuns Backend.backend 32768 4 X Y')
    {Y : ExactMatrix Q1 N 4 6} (hY : compactCenteredRebaseRuns 16384 Y' Y)
    {g4' : ExactMatrix M01 N 1 4} (hg4' : gadgetMatrixRuns Backend.backend 32768 4 g4')
    {g4 : ExactMatrix Q1 N 1 4} (hg4 : centeredRebaseRuns g4' g4)
    {cx : ExactMatrix Q1 N 1 4} {fw : ExactMatrix Q1 N 1 6} {comb : ExactMatrix Q1 N 1 6}
    (hcomb : comb = matrixAdd (matrixSub (matrixMul gc D) (matrixMul cx Y)) fw)
    {a b' : Int} (ha : a.toNat < 6) {dk : ExactMatrix Q1 N 1 1} (hslice : sliceMatrix comb 0 1 a b' dk)
    {s₀ β : ExactPoly Q1 N} {x : ErrorPoly N}
    {Kg K₁ Kw : Fin 2 → ExactMatrix Q1 N 1 6} {Kx : Fin 2 → ExactMatrix Q1 N 1 4}
    {eg e₁ e' ew : ExactMatrix Q1 N 1 6} {ex : ExactMatrix Q1 N 1 4}
    (hg : g = maskPart s₀ Kg - β • G + eg) (hc₁ : c₁ = maskPart s₀ K₁ - G + e₁)
    (hcx : cx = maskPart s₀ Kx - reducePoly Q1 N (((q2 : Nat) : Int) * x) • g4 + ex)
    (hfw : fw = maskPart s₀ Kw - QqPoly • e' + ew)
    {b : ExactMatrix Q1 N 1 16} {B : Fin 2 → ExactMatrix Q1 N 1 16} {eb : ExactMatrix Q1 N 1 16}
    (hb : b = maskPart s₀ B + eb)
    {Kpre : ExactMatrix Q1 N 16 1} {Ksm : Fin 2 → ExactMatrix Q1 N 1 1}
    (hpre : ∀ i, B i * Kpre =
      (Kg i * topRows D + K₁ i * bottomRows D - Kx i * Y + Kw i) * unitCol (q := Q1) ⟨a.toNat, ha⟩ + Ksm i)
    {csm S esm H : ExactMatrix Q1 N 1 1} (hcsm : csm = maskPart s₀ Ksm - S + esm)
    {out : ExactMatrix Q1 N 1 1}
    (hout : out = matrixSub (matrixSub (matrixMul b Kpre) (matrixAdd dk csm)) H) :
    out = QqPoly • ((β • rowOf A' 0 + rowOf A' 1 - actionValue x w350 + e') * unitCol (q := Q1) ⟨a.toNat, ha⟩) + (S - H) +
      (eb * Kpre - ((eg * topRows D + e₁ * bottomRows D - ex * Y + ew) *
        unitCol (q := Q1) ⟨a.toNat, ha⟩) - esm) := by
  obtain ⟨hX₀, hX₁⟩ := decompose_rows hDz hD hG
  have hzr : ∀ r : Fin (1 + 1), (fun _ c ↦ z r c : ExactMatrix Q1 N 1 6) = QqPoly • rowOf A' r := by
    intro r
    funext i c
    rw [hz, matrixMulScalarLeft_eq]
    simp only [Matrix.smul_apply, smul_eq_mul, matrixPolynomial_single, QqPoly, rowOf]
  rw [hzr 0] at hX₀
  rw [hzr 1] at hX₁
  have hXw : X = reduceMatrix M01 N 1 6 (((actionConst : Int) : ErrorPoly N) •
      fun r c ↦ centeredIntLift (w350 r c)) := by
    rw [hX, matrixMulScalarRight_eq, centeredRebaseRuns_reduce h351, reduceMatrix_smul]
    funext r c
    simp only [Matrix.smul_apply, smul_eq_mul, matrixPolynomial_single, reduceMatrix,
      reducePoly_intCast]
  have hYx := action_mul hg4' hg4 hXw hY' hY x
  have hdk : dk = (gc * D - cx * Y + fw) * unitCol (q := Q1) ⟨a.toNat, ha⟩ := by
    rw [slice_col hslice ha, hcomb]
    rfl
  have h := refresh_column (H := H) hgc hX₀ hX₁ hYx hg hc₁ hcx hfw hb ⟨a.toNat, ha⟩ hpre hcsm hdk
  rw [hout]
  exact h

/-- Row `j` of a relation in mask-part form. -/
theorem Rel.row {q m : Nat} {sm : ErrorPoly N} {A : Fin 2 → Fin 2 → ExactMatrix q N 1 m}
    {t c : Fin 2 → ExactMatrix q N 1 m} {B : Nat} (h : Rel sm A t c B) (j : Fin 2) :
    ∃ e : ErrorMatrix N 1 m, CoeffBound e B ∧
      c j = maskPart (reducePoly q N sm) (fun k ↦ A k j) - t j + reduceMatrix q N 1 m e := by
  obtain ⟨e, he, hc⟩ := h
  exact ⟨e j, he j, by rw [hc j]; rfl⟩

/-- A row of target `t G` times the digits of `Q_q` has target `Q_q t`. -/
theorem row_mul_qq {sm : ErrorPoly N} {K : Fin 2 → ExactMatrix Q1 N 1 6} {t : ErrorPoly N}
    {gq : ExactMatrix Q1 N 1 6} (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq)
    {e : ErrorMatrix N 1 6} {B : Nat} (he : CoeffBound e B) {c : ExactMatrix Q1 N 1 6}
    (hc : c = maskPart (reducePoly Q1 N sm) K - reducePoly Q1 N t • gq + reduceMatrix Q1 N 1 6 e)
    {Dq' Dq : ExactMatrix Q1 N 6 1}
    (hD' : gadgetDecomposeRuns Backend.backend 32768 6
      (matrixPolynomial [1152919668508929361] : ExactMatrix Q1 N 1 1) Dq')
    (hD : compactCenteredRebaseRuns 16384 Dq' Dq) :
    ∃ e' : ErrorMatrix N 1 1, CoeffBound e' (6 * N * B * 16384) ∧
      c * Dq = maskPart (reducePoly Q1 N sm) (fun k ↦ K k * Dq) -
        reducePoly Q1 N t • (matrixPolynomial [1152919668508929361] : ExactMatrix Q1 N 1 1) +
          reduceMatrix Q1 N 1 1 e' := by
  obtain ⟨W, hW, hWb⟩ := hD.2.2.2
  have hg := gadget_mul_decompose_one (by norm_num) Backend.layout4 layoutQ1 rfl hgq hD'
  rw [← compactRebase_self (by norm_num) hD] at hg
  refine ⟨e * W, coeffBound_mul hN he hWb, ?_⟩
  rw [hc, Matrix.add_mul, Matrix.sub_mul, maskPart_mul, Matrix.smul_mul, hg, hW,
    ← reduceMatrix_mul]

end RefreshTopProof
