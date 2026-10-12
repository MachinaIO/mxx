import TopInst
import TopLinkC
import TopChain
import TopRow
import TopLemmas
import TopDecode

/-! The names the assembly of an instance states its result with. -/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh

/-- The ring `Q₁` of the top-level gadget `G`. (The ported proofs read chain `0`'s name `Q1` as
the instance's decoding ring `R i0`; the gadget's ring keeps this name.) -/
abbrev Q1top : Nat := Q1

/-- The hash tag of instance `i0`'s `A'`: the top refresh's prefix and the output component. -/
def topTagI (i0 : Nat) : Blob :=
  completeHashTag [114, 101, 102, 114, 101, 115, 104, 47, 118, 49, 47, 65, 116, 111, 112, 45, 114, 101,
    102, 114, 101, 115, 104, 45, 114, 111, 117, 110, 100, 45, 116, 114, 105, 112]
    [.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (compZ (i0 : Int))]

/-- The value `y_j + e'_j` whose residue modulo `q₀` the decode stage reads. -/
noncomputable def topValue (model : HashModel) (key : ByteArray) (tag : Blob) (s : Fin 8 → Int)
    (r : ExactMatrix Q0 N 1 1) (c e : Int) (G : ExactMatrix Q1 N 1 6) (E : ErrorMatrix N 1 6) :
    ExactMatrix Q0 N 1 6 :=
  expectedRow (model.sample Q0 N 2 6 key (completeHashTag tag [])) (sigmaPoly s) r c e G +
    reduceMatrix Q0 N 1 6 E

/-- The mask's sampled value `h` of instance `i0` halves one modulo its prime. -/
theorem hm_two (i0 : Nat) (hi : i0 < 8) : ((2 * hmZ (i0 : Int) : Int) : ZMod (qc i0)) = 1 := by
  interval_cases i0 <;> decide

/-- `Y` of the online evaluation and of the preprocessing agree (the gadget's ring `Rg` and `Y`'s ring
`RY` independent). -/
theorem y_uniqueG {Rg RY qc Mx : Nat} {g g' : ExactMatrix Rg N 1 6} (hg : gadgetMatrixRuns Backend.backend 32768 6 g)
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') {r r' : ExactMatrix Q0 N 1 6}
    (hr : centeredRebaseRuns g r) (hr' : centeredRebaseRuns g' r') {m m' : ExactMatrix qc N 1 6}
    (hm : modulusReduceRuns r m) (hm' : modulusReduceRuns r' m') {c3 c1 : ExactMatrix qc N 1 1}
    {s s' t t' : ExactMatrix qc N 1 6} (hs : s = matrixMulScalarRight m c3)
    (hs' : s' = matrixMulScalarRight m' c3) (ht : t = matrixMulScalarRight s c1)
    (ht' : t' = matrixMulScalarRight s' c1) {u u' : ExactMatrix Mx N 1 6}
    (hu : centeredRebaseRuns t u) (hu' : centeredRebaseRuns t' u') {c : ExactMatrix Mx N 1 1}
    {v v' : ExactMatrix Mx N 1 6} (hv : v = matrixMulScalarRight u c)
    (hv' : v' = matrixMulScalarRight u' c) {Yz Yz' : ExactMatrix Mx N 4 6}
    (hy : gadgetDecomposeRuns Backend.backend 32768 4 v Yz)
    (hy' : gadgetDecomposeRuns Backend.backend 32768 4 v' Yz') {Y Y' : ExactMatrix RY N 4 6}
    (hY : compactCenteredRebaseRuns 16384 Yz Y) (hY' : compactCenteredRebaseRuns 16384 Yz' Y') :
    Y = Y' := by
  obtain rfl := gadgetMatrixRuns_unique hg hg'
  obtain rfl := centeredRebaseRuns_unique hr hr'
  obtain rfl := modulusReduceRuns_unique hm hm'
  subst hs hs' ht ht'
  obtain rfl := centeredRebaseRuns_unique hu hu'
  subst hv hv'
  exact dec_rebase_unique hy hy' hY hY'

/-- A bound that holds at every instance holds at instance `i0` (the per-instance check is
`decide` over `Fin 8`). -/
theorem inst_bound {f : Nat → Nat} {c : Nat} (h : ∀ j : Fin 8, f j < c) (i0 : Nat) (hi : i0 < 8) :
    f i0 < c := h ⟨i0, hi⟩

/-- The payload of component `j`: the payload `r` for `j = 0` and the constant one for `j = 1`. -/
theorem row_payload (r : ErrorPoly N) (j : Fin 2) :
    (fun _ _ ↦ reducePoly Q0 N (![r, (1 : ErrorPoly N)] j) : ExactMatrix Q0 N 1 1) =
      (![fun _ _ ↦ reducePoly Q0 N r, 1] : Fin 2 → ExactMatrix Q0 N 1 1) j := by
  fin_cases j
  · rfl
  · exact one_poly

theorem completeHashTag_nil (p : Blob) : completeHashTag p [] = p := by
  simp [completeHashTag]

/-- The link case of instance `i0` with the gadget of the online stage in `Q₁`. -/
theorem link_caseQ (i0 : Nat) (hi : i0 < 8) {GR G : ExactMatrix Q1 N 1 6}
    (hGR : gadgetMatrixRuns Backend.backend 32768 6 GR) (hG : gadgetMatrixRuns Backend.backend 32768 6 G) :
    ((∀ c, centeredIntLift (GR 0 c) = centeredIntLift (G 0 c)) ∧
      ((((cpZ (i0 : Int)) / (kdiv i0 : Int)) * k3Z (i0 : Int) * k1Z (i0 : Int) : Int) : ZMod (qc i0)) =
        ((q3 : Int) : ZMod (qc i0))) ∨
    (((((cpZ (i0 : Int)) / (kdiv i0 : Int)) * k3Z (i0 : Int) * k1Z (i0 : Int) : Int) : ZMod (qc i0)) = 0 ∧
      ((q3 : Int) : ZMod (qc i0)) = 0) := by
  by_cases h6 : i0 < 6
  · left
    refine ⟨fun c ↦ by rw [gadgetMatrixRuns_unique hGR hG], ?_⟩
    rw [ZMod.intCast_eq_intCast_iff']
    interval_cases i0 <;> decide
  · right
    rw [ZMod.intCast_zmod_eq_zero_iff_dvd, ZMod.intCast_zmod_eq_zero_iff_dvd]
    interval_cases i0 <;> first | omega | decide

set_option maxHeartbeats 4000000 in
/-- (`link_chain` with the gadget in its own ring `Rg`.) The decode stage's residues of a chain, from the online stage's output modulo its prime. -/
theorem link_chainG {R qc : Nat} (hR : 1 < R) (hqc : 1 < qc) (hqcR : qc ∣ R) (hqcQ0 : qc ∣ Q0)
    {A : ExactMatrix Q0 N (1 + 1) 6} {A' : ExactMatrix R N (1 + 1) 6} (hA' : modulusReduceRuns A A')
    {Rg : Nat} {GR : ExactMatrix Rg N 1 6} {G : ExactMatrix Q1 N 1 6}
    (hLG : ∀ c, centeredIntLift (GR 0 c) = centeredIntLift (G 0 c))
    {w345 : ExactMatrix Q0 N 1 6} (h345 : centeredRebaseRuns GR w345) {w346 : ExactMatrix qc N 1 6}
    (h346 : modulusReduceRuns w345 w346) {k3 k1 k : Int} {w347 w349 : ExactMatrix qc N 1 1}
    (e347 : w347 = (matrixPolynomial [k3] : ExactMatrix qc N 1 1))
    (e349 : w349 = (matrixPolynomial [k1] : ExactMatrix qc N 1 1))
    {w348 w350 : ExactMatrix qc N 1 6} (e348 : w348 = matrixMulScalarRight w346 w347)
    (e350 : w350 = matrixMulScalarRight w348 w349)
    (hk : ((k * k3 * k1 : Int) : ZMod qc) = ((q3 : Int) : ZMod qc))
    {βI σI xI rI : ErrorPoly N} (hβ : reducePoly qc N βI = reducePoly qc N σI)
    {σ : ExactMatrix Q0 N 1 1} (hσ : σ = fun _ _ ↦ reducePoly Q0 N σI) {c e : Int}
    (hx : xI = message c e * rI) (E : ErrorMatrix N 1 6) {e' : ExactMatrix R N 1 6}
    (hE : e' = reduceMatrix R N 1 6 E) {res : ExactMatrix qc N 1 6}
    (hres : ∀ col i, (res 0 col).coeff i =
      ((((reducePoly R N βI • rowOf A' 0 + rowOf A' 1 - actionValueC k xI w350 + e' :
        ExactMatrix R N 1 6) 0 col).coeff i).val :
        ZMod qc)) :
    Residue res (expectedRow A σ (fun _ _ ↦ reducePoly Q0 N rI) c e G + reduceMatrix Q0 N 1 6 E) := by
  haveI : NeZero qc := ⟨by omega⟩
  subst hE
  set L : ErrorMatrix N (1 + 1) 6 := fun r k ↦ centeredIntLift (A r k) with hL
  have hA : A = reduceMatrix Q0 N (1 + 1) 6 L := (reduceMatrix_centeredIntLift (by norm_num) hN A).symm
  have hA'r := modulusReduceRuns_reduce (by norm_num) hR hN hA' L hA
  set Lg : ErrorMatrix N 1 6 := fun r k ↦ centeredIntLift (GR r k) with hLg
  have h345r := centeredRebaseRuns_reduce h345
  have h346r := modulusReduceRuns_reduce (by norm_num) hqc hN h346 Lg h345r
  intro j i
  rw [hres j i]
  set W : ErrorPoly N := centeredIntLift (w350 0 j) with hW
  have hWr : reducePoly qc N W = reducePoly qc N (Lg 0 j * (k3 : ErrorPoly N) * (k1 : ErrorPoly N)) := by
    rw [hW, reducePoly_centeredIntLift hqc hN, e350, e348, e347, e349, h346r]
    simp only [matrixMulScalarRight, matrixPolynomial_single, reduceMatrix, map_mul, map_intCast]
  have hy : (reducePoly R N βI • rowOf A' 0 + rowOf A' 1 - actionValueC k xI w350 +
      reduceMatrix R N 1 6 E : ExactMatrix R N 1 6) 0 j = reducePoly R N
        (βI * L 0 j + L 1 j - ((k : ErrorPoly N) * xI) * W + E 0 j) := by
    rw [hA'r]
    simp only [actionValueC, rowOf, reduceMatrix, Matrix.add_apply, Matrix.sub_apply,
      Matrix.smul_apply, Pi.smul_apply, smul_eq_mul, map_add, map_sub, map_mul, map_intCast, hW]
  have hv : (expectedRow A σ (fun _ _ ↦ reducePoly Q0 N rI) c e G + reduceMatrix Q0 N 1 6 E) 0 j =
      reducePoly Q0 N (σI * L 0 j + L 1 j - (((q3 : Int) : ErrorPoly N) * rI * message c e) *
        centeredIntLift (G 0 j) + E 0 j) := by
    rw [hσ, hA]
    simp only [expectedRow, matrixSub, matrixAdd, matrixMulScalarLeft, multiplyMonomial_liftInteger,
      matrixPolynomial_single, reduceMatrix, Matrix.add_apply, Matrix.sub_apply, map_add, map_sub,
      map_mul, map_intCast, centeredRebase_eq]
    rfl
  rw [hy, hv, val_reduce hR hqc hqcR, val_reduce (by norm_num) hqc hqcQ0]
  congr 1
  have hc : reducePoly qc N ((k : ErrorPoly N) * (k3 : ErrorPoly N) * (k1 : ErrorPoly N)) =
      reducePoly qc N ((q3 : Int) : ErrorPoly N) := by
    rw [← Int.cast_mul, ← Int.cast_mul, reducePoly_intCast, reducePoly_intCast, intCast_eq_algebraMap,
      intCast_eq_algebraMap, hk]
  have hl : Lg 0 j = centeredIntLift (G 0 j) := hLG j
  simp only [map_add, map_sub, map_mul, map_intCast, hWr, hx, ← hβ, hl] at hc ⊢
  linear_combination (-(reducePoly qc N (message c e) * reducePoly qc N rI *
    reducePoly qc N (centeredIntLift (G 0 j)))) * hc

set_option maxHeartbeats 4000000 in
/-- The decode stage's residues of a chain whose action vanishes modulo its prime (`q₃ ≡ 0`). -/
theorem link_chain0G {R qc : Nat} (hR : 1 < R) (hqc : 1 < qc) (hqcR : qc ∣ R) (hqcQ0 : qc ∣ Q0)
    {A : ExactMatrix Q0 N (1 + 1) 6} {A' : ExactMatrix R N (1 + 1) 6} (hA' : modulusReduceRuns A A')
    {Rg : Nat} {GR : ExactMatrix Rg N 1 6} {G : ExactMatrix Q1 N 1 6}
    {w345 : ExactMatrix Q0 N 1 6} (h345 : centeredRebaseRuns GR w345) {w346 : ExactMatrix qc N 1 6}
    (h346 : modulusReduceRuns w345 w346) {k3 k1 k : Int} {w347 w349 : ExactMatrix qc N 1 1}
    (e347 : w347 = (matrixPolynomial [k3] : ExactMatrix qc N 1 1))
    (e349 : w349 = (matrixPolynomial [k1] : ExactMatrix qc N 1 1))
    {w348 w350 : ExactMatrix qc N 1 6} (e348 : w348 = matrixMulScalarRight w346 w347)
    (e350 : w350 = matrixMulScalarRight w348 w349)
    (hk : ((k * k3 * k1 : Int) : ZMod qc) = 0) (hq3 : ((q3 : Int) : ZMod qc) = 0)
    {βI σI xI rI : ErrorPoly N} (hβ : reducePoly qc N βI = reducePoly qc N σI)
    {σ : ExactMatrix Q0 N 1 1} (hσ : σ = fun _ _ ↦ reducePoly Q0 N σI) {c e : Int}
    (hx : xI = message c e * rI) (E : ErrorMatrix N 1 6) {e' : ExactMatrix R N 1 6}
    (hE : e' = reduceMatrix R N 1 6 E) {res : ExactMatrix qc N 1 6}
    (hres : ∀ col i, (res 0 col).coeff i =
      ((((reducePoly R N βI • rowOf A' 0 + rowOf A' 1 - actionValueC k xI w350 + e' :
        ExactMatrix R N 1 6) 0 col).coeff i).val :
        ZMod qc)) :
    Residue res (expectedRow A σ (fun _ _ ↦ reducePoly Q0 N rI) c e G + reduceMatrix Q0 N 1 6 E) := by
  haveI : NeZero qc := ⟨by omega⟩
  subst hE
  set L : ErrorMatrix N (1 + 1) 6 := fun r k ↦ centeredIntLift (A r k) with hL
  have hA : A = reduceMatrix Q0 N (1 + 1) 6 L := (reduceMatrix_centeredIntLift (by norm_num) hN A).symm
  have hA'r := modulusReduceRuns_reduce (by norm_num) hR hN hA' L hA
  set Lg : ErrorMatrix N 1 6 := fun r k ↦ centeredIntLift (GR r k) with hLg
  have h345r := centeredRebaseRuns_reduce h345
  have h346r := modulusReduceRuns_reduce (by norm_num) hqc hN h346 Lg h345r
  intro j i
  rw [hres j i]
  set W : ErrorPoly N := centeredIntLift (w350 0 j) with hW
  have hWr : reducePoly qc N W = reducePoly qc N (Lg 0 j * (k3 : ErrorPoly N) * (k1 : ErrorPoly N)) := by
    rw [hW, reducePoly_centeredIntLift hqc hN, e350, e348, e347, e349, h346r]
    simp only [matrixMulScalarRight, matrixPolynomial_single, reduceMatrix, map_mul, map_intCast]
  have hy : (reducePoly R N βI • rowOf A' 0 + rowOf A' 1 - actionValueC k xI w350 +
      reduceMatrix R N 1 6 E : ExactMatrix R N 1 6) 0 j = reducePoly R N
        (βI * L 0 j + L 1 j - ((k : ErrorPoly N) * xI) * W + E 0 j) := by
    rw [hA'r]
    simp only [actionValueC, rowOf, reduceMatrix, Matrix.add_apply, Matrix.sub_apply,
      Matrix.smul_apply, Pi.smul_apply, smul_eq_mul, map_add, map_sub, map_mul, map_intCast, hW]
  have hv : (expectedRow A σ (fun _ _ ↦ reducePoly Q0 N rI) c e G + reduceMatrix Q0 N 1 6 E) 0 j =
      reducePoly Q0 N (σI * L 0 j + L 1 j - (((q3 : Int) : ErrorPoly N) * rI * message c e) *
        centeredIntLift (G 0 j) + E 0 j) := by
    rw [hσ, hA]
    simp only [expectedRow, matrixSub, matrixAdd, matrixMulScalarLeft, multiplyMonomial_liftInteger,
      matrixPolynomial_single, reduceMatrix, Matrix.add_apply, Matrix.sub_apply, map_add, map_sub,
      map_mul, map_intCast, centeredRebase_eq]
    rfl
  rw [hy, hv, val_reduce hR hqc hqcR, val_reduce (by norm_num) hqc hqcQ0]
  congr 1
  have hc : reducePoly qc N ((k : ErrorPoly N) * (k3 : ErrorPoly N) * (k1 : ErrorPoly N)) = 0 := by
    rw [← Int.cast_mul, ← Int.cast_mul, reducePoly_intCast, intCast_eq_algebraMap, hk, map_zero]
  have hc3 : reducePoly qc N ((q3 : Int) : ErrorPoly N) = 0 := by
    rw [reducePoly_intCast, intCast_eq_algebraMap, hq3, map_zero]
  simp only [map_add, map_sub, map_mul, map_intCast, hWr, hx, ← hβ] at hc hc3 ⊢
  linear_combination (-(reducePoly qc N (message c e) * reducePoly qc N rI * reducePoly qc N (Lg 0 j))) * hc +
    (reducePoly qc N rI * reducePoly qc N (message c e) * reducePoly qc N (centeredIntLift (G 0 j))) * hc3

/-- The decode stage's residues of any chain: its gadget lifts as `Q₁`'s and its action scales to
`q₃` (`link_chain`), or its action and `q₃` vanish modulo its prime (`link_chain0`). -/
theorem link_anyG {R qc : Nat} (hR : 1 < R) (hqc : 1 < qc) (hqcR : qc ∣ R) (hqcQ0 : qc ∣ Q0)
    {A : ExactMatrix Q0 N (1 + 1) 6} {A' : ExactMatrix R N (1 + 1) 6} (hA' : modulusReduceRuns A A')
    {Rg : Nat} {GR : ExactMatrix Rg N 1 6} {G : ExactMatrix Q1 N 1 6}
    {w345 : ExactMatrix Q0 N 1 6} (h345 : centeredRebaseRuns GR w345) {w346 : ExactMatrix qc N 1 6}
    (h346 : modulusReduceRuns w345 w346) {k3 k1 k : Int} {w347 w349 : ExactMatrix qc N 1 1}
    (e347 : w347 = (matrixPolynomial [k3] : ExactMatrix qc N 1 1))
    (e349 : w349 = (matrixPolynomial [k1] : ExactMatrix qc N 1 1))
    {w348 w350 : ExactMatrix qc N 1 6} (e348 : w348 = matrixMulScalarRight w346 w347)
    (e350 : w350 = matrixMulScalarRight w348 w349)
    (hcase : ((∀ c, centeredIntLift (GR 0 c) = centeredIntLift (G 0 c)) ∧
        ((k * k3 * k1 : Int) : ZMod qc) = ((q3 : Int) : ZMod qc)) ∨
      (((k * k3 * k1 : Int) : ZMod qc) = 0 ∧ ((q3 : Int) : ZMod qc) = 0))
    {βI σI xI rI : ErrorPoly N} (hβ : reducePoly qc N βI = reducePoly qc N σI)
    {σ : ExactMatrix Q0 N 1 1} (hσ : σ = fun _ _ ↦ reducePoly Q0 N σI) {c e : Int}
    (hx : xI = message c e * rI) (E : ErrorMatrix N 1 6) {e' : ExactMatrix R N 1 6}
    (hE : e' = reduceMatrix R N 1 6 E) {res : ExactMatrix qc N 1 6}
    (hres : ∀ col i, (res 0 col).coeff i =
      ((((reducePoly R N βI • rowOf A' 0 + rowOf A' 1 - actionValueC k xI w350 + e' :
        ExactMatrix R N 1 6) 0 col).coeff i).val :
        ZMod qc)) :
    Residue res (expectedRow A σ (fun _ _ ↦ reducePoly Q0 N rI) c e G + reduceMatrix Q0 N 1 6 E) := by
  rcases hcase with ⟨hLG, hk⟩ | ⟨hk, hq3⟩
  · exact link_chainG hR hqc hqcR hqcQ0 hA' hLG h345 h346 e347 e349 e348 e350 hk hβ hσ hx E hE hres
  · exact link_chain0G hR hqc hqcR hqcQ0 hA' h345 h346 e347 e349 e348 e350 hk hq3 hβ hσ hx E hE hres

end RefreshTopProof
