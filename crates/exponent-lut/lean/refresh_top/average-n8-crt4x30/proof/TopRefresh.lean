import TopLemmas
import ExponentRefresh
import ExponentGadget
import ExponentStacking

/-!
# The refresh combination

One output column of a decoder: the hint times the column's preimage, minus the column of the
combination `d = [g | c₁] D(Q_q A') - cₓ Y + f` and the column's smudge, minus the half constant.
`decoded_value` makes it `Q_q y + M + N`.
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh

/-- A one-row gadget times the digits of a one-row target is the target. -/
theorem gadget_mul_decompose_one {q' m m' : Nat} (hq' : 1 < q')
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q')
    (hlayout : backend.regularLayout q' N = some layout) (hexact : layout.droppedModuli = 0)
    {g : ExactMatrix q' N 1 m} (hg : gadgetMatrixRuns backend base digits g)
    {X : ExactMatrix q' N 1 m'} {D : ExactMatrix q' N m m'}
    (hD : gadgetDecomposeRuns backend base digits X D) : g * D = X := by
  obtain ⟨layout₁, hl₁, _, _, hw₁, hg⟩ := hg
  obtain ⟨layout₂, hl₂, _, _, hw₂, hD, _⟩ := hD
  rw [hlayout] at hl₁ hl₂
  cases hl₁
  cases hl₂
  subst hw₁
  simp only [castMatrixColumns, castMatrixRows] at hg hD
  rw [hg, hD]
  exact regularGadgetMatrix_reconstruct layout X hq' hN hexact

/-- The scaled one-row action: `(p x) g` (the next gadget re-encoded) times the re-encoded digits of
`X` is `scaleUp (x X)`. -/
theorem gadget_mul_action_scaleUp_one {q p q' m m' : Nat} (hq : q = p * q') (hq' : 1 < q')
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q')
    (hlayout : backend.regularLayout q' N = some layout) (hexact : layout.droppedModuli = 0)
    {g' : ExactMatrix q' N 1 m} (hg' : gadgetMatrixRuns backend base digits g')
    {g : ExactMatrix q N 1 m} (hg : centeredRebaseRuns g' g)
    {X : ExactMatrix q' N 1 m'} {D' : ExactMatrix q' N m m'}
    (hD' : gadgetDecomposeRuns backend base digits X D')
    {D : ExactMatrix q N m m'} {bound : Nat} (hD : compactCenteredRebaseRuns bound D' D)
    (x : ErrorPoly N) :
    (reducePoly q N (((p : Int) : ErrorPoly N) * x) • g) * D =
      scaleUpMatrix q p (reducePoly q' N x • X) := by
  let G : ErrorMatrix N 1 m := fun row column ↦ centeredIntLift (g' row column)
  let E : ErrorMatrix N m m' := fun row column ↦ centeredIntLift (D' row column)
  have hgq : g = reduceMatrix q N 1 m G := centeredRebaseRuns_reduce hg
  have hDq : D = reduceMatrix q N m m' E := centeredRebaseRuns_reduce hD.2.2.1
  have hgq' : g' = reduceMatrix q' N 1 m G := (reduceMatrix_centeredIntLift hq' hN g').symm
  have hDq' : D' = reduceMatrix q' N m m' E := (reduceMatrix_centeredIntLift hq' hN D').symm
  have hrec := gadget_mul_decompose_one hq' layout hlayout hexact hg' hD'
  rw [← hrec, hgq, hDq, hgq', hDq']
  simp only [← reduceMatrix_smul, ← reduceMatrix_mul]
  funext row column
  simp only [scaleUpMatrix, reduceMatrix]
  rw [scaleUp_reducePoly hq hq' hN]
  congr 1
  simp only [Matrix.smul_mul, Matrix.smul_apply, smul_eq_mul, mul_assoc]

/-- Column `k` as a matrix. -/
noncomputable def unitCol {q m : Nat} (k : Fin m) : ExactMatrix q N m 1 :=
  fun i _ ↦ if i = k then 1 else 0

theorem mul_unitCol {q rows m : Nat} (x : ExactMatrix q N rows m) (k : Fin m) :
    x * unitCol k = fun r _ ↦ x r k := by
  funext r c
  rw [Matrix.mul_apply, Finset.sum_eq_single k]
  · simp [unitCol]
  · intro i _ hi
    simp [unitCol, hi]
  · simp

/-- A one-column slice is the product with the unit column. -/
theorem slice_col {q m : Nat} {x : ExactMatrix q N 1 m} {out : ExactMatrix q N 1 1} {a b : Int}
    (h : sliceMatrix x 0 1 a b out) (ha : a.toNat < m) :
    out = x * unitCol ⟨a.toNat, ha⟩ := by
  rw [mul_unitCol]
  funext r c
  obtain rfl : r = 0 := Subsingleton.elim _ _
  obtain rfl : c = 0 := Subsingleton.elim _ _
  exact slice_entry h ha

set_option maxHeartbeats 4000000 in
/-- One output column of a decoder before the modulus switch. -/
theorem refresh_column {s₀ β Qq : ExactPoly Q1 N}
    {g c₁ G : ExactMatrix Q1 N 1 6} {gc : ExactMatrix Q1 N 1 (6 + 6)} (hgc : concatColumns g c₁ gc)
    {D : ExactMatrix Q1 N (6 + 6) 6} {A₀ A₁ : ExactMatrix Q1 N 1 6}
    (hX₀ : G * topRows D = Qq • A₀) (hX₁ : G * bottomRows D = Qq • A₁)
    {cx P : ExactMatrix Q1 N 1 4} {Y : ExactMatrix Q1 N 4 6} {Tx : ExactMatrix Q1 N 1 6}
    (hY : P * Y = Qq • Tx)
    {fw e' ew : ExactMatrix Q1 N 1 6}
    {Kg K₁ Kw : Fin 2 → ExactMatrix Q1 N 1 6} {Kx : Fin 2 → ExactMatrix Q1 N 1 4}
    {eg e₁ : ExactMatrix Q1 N 1 6} {ex : ExactMatrix Q1 N 1 4}
    (hg : g = maskPart s₀ Kg - β • G + eg) (hc₁ : c₁ = maskPart s₀ K₁ - G + e₁)
    (hcx : cx = maskPart s₀ Kx - P + ex) (hfw : fw = maskPart s₀ Kw - Qq • e' + ew)
    {b : ExactMatrix Q1 N 1 16} {B : Fin 2 → ExactMatrix Q1 N 1 16} {eb : ExactMatrix Q1 N 1 16}
    (hb : b = maskPart s₀ B + eb)
    (k : Fin 6) {Kpre : ExactMatrix Q1 N 16 1} {Ksm : Fin 2 → ExactMatrix Q1 N 1 1}
    (hpre : ∀ i, B i * Kpre =
      (Kg i * topRows D + K₁ i * bottomRows D - Kx i * Y + Kw i) * unitCol (q := Q1) k + Ksm i)
    {csm S esm H : ExactMatrix Q1 N 1 1} (hcsm : csm = maskPart s₀ Ksm - S + esm)
    {dk : ExactMatrix Q1 N 1 1}
    (hdk : dk = (gc * D - cx * Y + fw) * unitCol (q := Q1) k) :
    b * Kpre - (dk + csm) - H =
      Qq • ((β • A₀ + A₁ - Tx + e') * unitCol (q := Q1) k) + (S - H) +
        (eb * Kpre - ((eg * topRows D + e₁ * bottomRows D - ex * Y + ew) * unitCol (q := Q1) k) - esm) := by
  have h := decoded_value (R := ExactPoly Q1 N) (m := 6) (w := 4) (columns := 1) (hint := 16)
    s₀ β Qq G P Kg K₁ Kx (fun i ↦ Kw i * unitCol (q := Q1) k) eg e₁ ex (ew * unitCol (q := Q1) k)
    (topRows D * unitCol (q := Q1) k) (bottomRows D * unitCol (q := Q1) k) (Y * unitCol (q := Q1) k)
    (A₀ * unitCol (q := Q1) k) (A₁ * unitCol (q := Q1) k) (Tx * unitCol (q := Q1) k) (e' * unitCol (q := Q1) k)
    (by rw [← Matrix.mul_assoc, hX₀, Matrix.smul_mul]) (by rw [← Matrix.mul_assoc, hX₁, Matrix.smul_mul])
    (by rw [← Matrix.mul_assoc, hY, Matrix.smul_mul]) B eb Kpre Ksm (S - H) H esm
    (fun i ↦ by rw [hpre i]; simp only [Matrix.add_mul, Matrix.sub_mul, Matrix.mul_assoc])
  simp only at h
  have hd : dk = (maskPart s₀ Kg - β • G + eg) * (topRows D * unitCol (q := Q1) k) +
      (maskPart s₀ K₁ - G + e₁) * (bottomRows D * unitCol (q := Q1) k) -
      (maskPart s₀ Kx - P + ex) * (Y * unitCol (q := Q1) k) +
      (maskPart s₀ (fun i ↦ Kw i * unitCol (q := Q1) k) - Qq • (e' * unitCol (q := Q1) k) + ew * unitCol (q := Q1) k) := by
    rw [hdk, concatColumns_mul hgc, ← hg, ← hc₁, ← hcx, hfw]
    simp only [maskPart, Matrix.add_mul, Matrix.sub_mul, Matrix.mul_assoc, Matrix.smul_mul]
    try abel
  rw [hd, hb, hcsm]
  rw [show S - H + H = S by abel] at h
  convert h using 1
  · try abel
  · simp only [Matrix.add_mul, Matrix.sub_mul, Matrix.smul_mul, Matrix.mul_assoc, smul_add, smul_sub]
    try abel

end RefreshTopProof
