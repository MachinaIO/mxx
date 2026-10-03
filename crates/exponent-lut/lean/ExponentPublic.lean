import ExponentLut

/-!
# Public multiplication and public alignment

A public-payload encoding of `α` uses the mask as its payload: its target is
`t_j = (α (s₀, 1)_j) G` for the native gadget `G` of its modulus. Two operations act on it.

* `align_down`: the right action by the digits `H` of `p g`, for the gadget `g` of the next
  modulus re-encoded at `q = p q'`, followed by the runtime ModDown, gives the public encoding of
  `α` at `q'` with the native gadget of `q'`.
* `mul_public`: the stacked public encoding acting on the decomposition of a private key `A_s`,
  plus `α` times the private row, is an encoding of `α t_s` under the acted key: the gadget term
  `α (s₀ A_{s,0j} + A_{s,1j})` cancels against `α` times the private row.
-/

namespace MxxExponentLut

open Mxx.Primitives MxxRuntime
open scoped BigOperators

variable {n : Nat}

/-- The one-row gadget times the digits of a one-row target. -/
theorem gadget_mul_decompose {q m m' : Nat} (hq : 1 < q) (hn : 0 < n)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {X : ExactMatrix q n 1 m'} {D : ExactMatrix q n m m'}
    (hD : gadgetDecomposeRuns backend base digits X D) : G * D = X := by
  obtain ⟨layout₁, hl₁, _, _, hw₁, hG⟩ := hG
  obtain ⟨layout₂, hl₂, _, _, hw₂, hD, _⟩ := hD
  rw [hlayout] at hl₁ hl₂
  cases hl₁
  cases hl₂
  have hrec := regularGadgetMatrix_reconstruct (rows := 1) layout X hq hn hexact
  funext row column
  rw [← congrFun (congrFun hrec row) column]
  simp only [Matrix.mul_apply, hG, hD, castMatrixColumns_apply, castMatrixRows_apply]
  exact Fintype.sum_equiv (finCongr hw₁) _ _ (fun k ↦ rfl)

/-- A gadget decomposition is a function of its target. -/
theorem gadgetDecomposeRuns_unique {q rows columns inner : Nat} {backend : BackendContext}
    {base digits : Int} {X : ExactMatrix q n rows columns} {D D' : ExactMatrix q n inner columns}
    (hD : gadgetDecomposeRuns backend base digits X D)
    (hD' : gadgetDecomposeRuns backend base digits X D') : D = D' := by
  obtain ⟨layout, hl, _, _, hw, hD, _⟩ := hD
  obtain ⟨layout', hl', _, _, hw', hD', _⟩ := hD'
  rw [hl] at hl'
  cases hl'
  rw [hD, hD']

/-- The digits of a gadget decomposition are small. -/
theorem gadgetDecomposeRuns_small {q rows columns inner : Nat} (hq : 1 < q) (hn : 0 < n)
    {backend : BackendContext} {base digits : Int} {X : ExactMatrix q n rows columns}
    {D : ExactMatrix q n inner columns} (hD : gadgetDecomposeRuns backend base digits X D) :
    ∃ d : ErrorMatrix n inner columns, D = reduceMatrix q n inner columns d ∧
      CoeffBound d (base.toNat / 2) := by
  obtain ⟨layout, _, hbase, _, hw, hD, _⟩ := hD
  obtain ⟨d, hd, hb⟩ := regularDecomposeMatrix_bounded layout X hq hn
  refine ⟨fun row column ↦ d (Fin.cast hw row) column, ?_, fun row column ↦ ?_⟩
  · funext row column
    rw [hD, castMatrixRows_apply, hd]
    rfl
  · rw [hbase, Int.toNat_natCast]
    exact hb _ column

/-- `p x g` for the re-encoded gadget `g` of `q'` is `scaleUp` of `x g'`. -/
theorem scaleUpMatrix_rebasedGadget {q p q' m : Nat} (hq : q = p * q') (hq' : 1 < q')
    (hn : 0 < n) {g' : ExactMatrix q' n 1 m} {g : ExactMatrix q n 1 m}
    (hg : centeredRebaseRuns g' g) (x : ErrorPoly n) :
    scaleUpMatrix q p (reducePoly q' n x • g') = reducePoly q n (((p : Int) : ErrorPoly n) * x) • g := by
  let G : ErrorMatrix n 1 m := fun row column ↦ centeredIntLift (g' row column)
  have hgq : g = reduceMatrix q n 1 m G := centeredRebaseRuns_reduce hg
  have hgq' : g' = reduceMatrix q' n 1 m G := (reduceMatrix_centeredIntLift hq' hn g').symm
  rw [hgq, hgq', ← reduceMatrix_smul, ← reduceMatrix_smul]
  funext row column
  simp only [scaleUpMatrix, reduceMatrix, Matrix.smul_apply, smul_eq_mul]
  rw [scaleUp_reducePoly hq hq' hn, mul_assoc]

theorem matrixMulScalarRight_smul {q rows columns : Nat} (x : ExactMatrix q n rows columns)
    (a : ExactMatrix q n 1 1) : matrixMulScalarRight x a = a 0 0 • x := by
  funext i j
  simp [matrixMulScalarRight, mul_comm]

/-- A right action of one small matrix on every block. -/
theorem Rel.mulRight {q m m' : Nat} (hn : 0 < n) {sm : ErrorPoly n}
    {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {t c : Fin 2 → ExactMatrix q n 1 m} {B β : Nat}
    (h : Rel sm A t c B) (d : ErrorMatrix n m m') (hd : CoeffBound d β) :
    Rel sm (fun k j ↦ A k j * reduceMatrix q n m m' d) (fun j ↦ t j * reduceMatrix q n m m' d)
      (fun j ↦ c j * reduceMatrix q n m m' d) (m * n * B * β) := by
  obtain ⟨e, he, hc⟩ := h
  refine ⟨fun j ↦ e j * d, fun j ↦ coeffBound_mul hn (he j) hd, fun j ↦ ?_⟩
  beta_reduce
  rw [hc j, reduceMatrix_mul, Matrix.add_mul, Matrix.sub_mul, Matrix.add_mul, Matrix.smul_mul]

set_option maxHeartbeats 1000000 in
/-- Public alignment: the right action by the digits `H` of `p g` and the runtime ModDown move a
public encoding of `α` with the native gadget `G` of `q = p q'` to one with the native gadget
`g'` of `q'`. -/
theorem align_down {q q' p m m' : Nat} (hn : 0 < n) (hq : q = p * q') (hq1 : 1 < q)
    (hq' : 1 < q') (hp : 0 < p)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {g' : ExactMatrix q' n 1 m'} {g : ExactMatrix q n 1 m'} (hg : centeredRebaseRuns g' g)
    {sm α : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {c : Fin 2 → ExactMatrix q n 1 m} {B : Nat}
    (ha : Rel sm A (fun j ↦ reducePoly q n (α * unitPair sm j) • G) c B)
    {scaled : ExactMatrix q n 1 m'} (hscaled : scaled = reducePoly q n ((p : Int) : ErrorPoly n) • g)
    {H : ExactMatrix q n m m'} (hH : gadgetDecomposeRuns backend base digits scaled H)
    {Y : Fin 2 → ExactMatrix q n 1 m'} {K : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hY : ∀ j, Y j = matrixMul (c j) H) (hK : ∀ k j, K k j = matrixMul (A k j) H)
    {basis : List Nat} {c' : Fin 2 → ExactMatrix q' n 1 m'}
    {A' : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'}
    (hc' : ∀ j, blockModSwitchRuns basis 1 (Y j) (c' j))
    (hA' : ∀ k j, blockModSwitchRuns basis 1 (K k j) (A' k j)) :
    Rel sm A' (fun j ↦ reducePoly q' n (α * unitPair sm j) • g') c'
      ((m * n * B * (base.toNat / 2) + (n + 2) * (p / 2)) / p) := by
  obtain ⟨d, hd, hdb⟩ := gadgetDecomposeRuns_small hq1 hn hH
  have hact := ha.mulRight hn d hdb
  rw [← hd] at hact
  have hrec := gadget_mul_decompose hq1 hn layout hlayout hexact hG hH
  have ht : ∀ j, reducePoly q n (α * unitPair sm j) • G * H =
      scaleUpMatrix q p (reducePoly q' n (α * unitPair sm j) • g') := by
    intro j
    rw [Matrix.smul_mul, hrec, hscaled, smul_smul, ← map_mul,
      scaleUpMatrix_rebasedGadget hq hq' hn hg, mul_comm]
  have hpq : q / q' = p := by rw [hq, Nat.mul_div_cancel _ (by omega)]
  exact hact.modDown hn hq hq1 hq' hp hsm (A' := A') (c' := c') ht
    (fun k j ↦ by
      obtain ⟨r, hr, hrel⟩ := blockModSwitchRuns_scaleUp hn (hA' k j)
      rw [hpq] at hr hrel
      exact ⟨r, hr, by rw [hrel, hK k j]; rfl⟩)
    (fun j ↦ by
      obtain ⟨r, hr, hrel⟩ := blockModSwitchRuns_scaleUp hn (hc' j)
      rw [hpq] at hr hrel
      exact ⟨r, hr, by rw [hrel, hY j]; rfl⟩)

set_option maxHeartbeats 1000000 in
/-- Public multiplication: the stacked public encoding of `α` acting on the decomposition of the
private key `A_s`, plus `α` times the private row, encodes `α t_s` under the acted key. -/
theorem mul_public {q m m' bound : Nat} (hn : 0 < n) (hq1 : 1 < q)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {sm α : ErrorPoly n} (hα : polyNorm α ≤ 1)
    {Aa : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {ca : Fin 2 → ExactMatrix q n 1 m} {Ba : Nat}
    (ha : Rel sm Aa (fun j ↦ reducePoly q n (α * unitPair sm j) • G) ca Ba)
    {As : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {ts cs : Fin 2 → ExactMatrix q n 1 m'} {Bs : Nat}
    (hs : Rel sm As ts cs Bs)
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows (ca 0) (Aa 0 0) l01) (hl : concatRows l01 (Aa 1 0) l)
    (hr01 : concatRows (ca 1) (Aa 0 1) r01) (hr : concatRows r01 (Aa 1 1) r)
    (hboth : concatColumns l r both)
    {Xs : Fin 2 → ExactMatrix q n (1 + 1) m'} (hXs : ∀ j, concatRows (As 0 j) (As 1 j) (Xs j))
    {D' : Fin 2 → ExactMatrix q n (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xs j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {payload : ExactMatrix q n 1 1} (hpayload : payload = fun _ _ ↦ reducePoly q n α)
    {Y : Fin 2 → ExactMatrix q n 1 m'} {A' : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hY : ∀ j, sliceMatrix (matrixMul both (D j)) 0 1 0 m' (Y j))
    (hA0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (A' 0 j))
    (hA1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (A' 1 j))
    {c' : Fin 2 → ExactMatrix q n 1 m'}
    (hc' : ∀ j, c' j = Y j + matrixMulScalarRight (cs j) payload) :
    Rel sm A' (fun j ↦ reducePoly q n α • ts j) c' ((m + m) * n * Ba * bound + n * 1 * Bs) := by
  have hrows := fun j ↦ stacked_row hl01 hl hr01 hr hboth (D j)
  choose d hd using fun j ↦ (hD j).2.2.2
  have hDd : ∀ j, D j = reduceMatrix q n (m + m) m' (d j) := fun j ↦ (hd j).1
  have hDD' : ∀ j, D j = D' j := fun j ↦
    centeredRebaseRuns_unique (hD j).2.2.1 (centeredRebaseRuns_self hq1 hn (D' j))
  have hact := ha.action hn d (fun j ↦ (hd j).2)
  simp only [← hDd] at hact
  have hYe : ∀ j, Y j = pairMul ca (D j) := by
    intro j
    rw [sliceMatrix_row (rows := 1 + 1 + 1) 0 (hY j), ← pair_eta ca]
    exact (hrows j).1
  have hAe : ∀ k j, A' k j = pairMul (Aa k) (D j) := by
    intro k j
    match k with
    | 0 =>
      rw [sliceMatrix_row (rows := 1 + 1 + 1) 1 (hA0 j), ← pair_eta (Aa 0)]
      exact (hrows j).2.1
    | 1 =>
      rw [sliceMatrix_row (rows := 1 + 1 + 1) 2 (hA1 j), ← pair_eta (Aa 1)]
      exact (hrows j).2.2
  have ht : ∀ j, pairMul (fun i ↦ reducePoly q n (α * unitPair sm i) • G) (D j) =
      reducePoly q n α • (reducePoly q n sm • As 0 j + As 1 j) := by
    intro j
    rw [hDD' j, gadget_pairMul_decompose hq1 hn layout hlayout hexact hG (hD' j),
      (concatRows_two (hXs j)).1, (concatRows_two (hXs j)).2]
    simp only [unitPair, Matrix.cons_val_zero, Matrix.cons_val_one, map_mul, mul_one,
      smul_add, smul_smul, mul_comm]
  obtain ⟨ea, hea, hca⟩ := hact
  obtain ⟨es, hes, hcs⟩ := hs
  beta_reduce at hca
  refine ⟨fun j ↦ ea j + α • es j, fun j ↦ coeffBound_add (hea j) (coeffBound_smul hn hα (hes j)),
    fun j ↦ ?_⟩
  beta_reduce
  rw [hc' j, hYe j, hca j, ht j, matrixMulScalarRight_smul, hcs j, hpayload, hAe 0 j, hAe 1 j,
    reduceMatrix_add, reduceMatrix_smul]
  simp only [smul_add, smul_sub]
  abel

end MxxExponentLut
