import Dio2Diag
import GenInstLemmas
import GenFinal

/-!
# The fresh and smudge rows of a refresh part

A part slices its cells (columns of `m` entries) out of the fresh and smudge column totals, acts
on them by the digits of a block-diagonal matrix (`blockRow_mul_decompose`) and reads the result
as a row of the part (DIO2-SPEC §6.7).
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.GenInst Mxx.ExponentLut.Refresh

variable {n : Nat}

/-- A one-row slice of `C'` blocks at block `k₀` of a row of `C` blocks. -/
theorem slice_blocks {q C C' m : Nat} {x : ExactMatrix q n 1 (C * m)} {s e : Int}
    {y : ExactMatrix q n 1 (C' * m)} (h : sliceMatrix x 0 1 s e y) (k₀ : Nat)
    (hs : s = (m : Int) * (k₀ : Int)) (c : Fin C') (hc : k₀ + c.val < C) :
    blk y c = blk x ⟨k₀ + c.val, hc⟩ := by
  obtain ⟨-, -, -, -, -, -, -, -, hentry⟩ := h
  funext r l
  have hr : r = 0 := Subsingleton.elim _ _
  subst hr
  have hl := l.isLt
  have e1 : s.toNat = m * k₀ := by rw [hs]; exact_mod_cast rfl
  simp only [blk]
  rw [hentry 0 _ (by omega) (by rw [e1]; have := c.isLt; nlinarith)]
  congr 1
  apply Fin.ext
  show s.toNat + (c.val * m + l.val) = (k₀ + c.val) * m + l.val
  rw [e1]; ring

/-- A row of `C` blocks of width `m` is determined by its blocks. -/
theorem row_eq_of_blk {q C m : Nat} {x y : ExactMatrix q n 1 (C * m)} (h : ∀ c, blk x c = blk y c) :
    x = y := by
  funext r l
  have hm : 0 < m := Nat.pos_of_ne_zero (by rintro rfl; simpa using l.isLt)
  have e : l = ⟨(divFin l).val * m + (modFin l).val, by
      have := (divFin l).isLt; have := (modFin l).isLt; nlinarith⟩ := by
    apply Fin.ext
    simp only [divFin, modFin]
    exact (Nat.div_add_mod' l.val m).symm
  rw [e]
  exact congrFun (congrFun (h (divFin l)) r) (modFin l)

/-- **A row from its cells.** Cells `blk y c = s K_c - t_c G + e_c` assemble into
`y = s K - (t_c G)_c + e`. -/
theorem row_of_cells {q C m : Nat} {y : ExactMatrix q n 1 (C * m)} {s : ExactPoly q n}
    {K : Fin C → Fin 2 → ExactMatrix q n 1 m} {t : Fin C → ExactPoly q n}
    {G : ExactMatrix q n 1 m} {e : Fin C → ErrorMatrix n 1 m}
    (h : ∀ c, blk y c = maskPart s (K c) - t c • G + reduceMatrix q n 1 m (e c)) :
    y = maskPart s (fun k _ l ↦ K (divFin l) k 0 (modFin l)) - blockRow t G +
      reduceMatrix q n 1 (C * m) (fun _ l ↦ e (divFin l) 0 (modFin l)) := by
  apply row_eq_of_blk
  intro c
  rw [h c]
  funext r l
  have hr : r = 0 := Subsingleton.elim _ _
  subst hr
  have hm : 0 < m := Nat.pos_of_ne_zero (by rintro rfl; exact l.elim0)
  have hd : divFin (⟨c.val * m + l.val, by have := c.isLt; have := l.isLt; nlinarith⟩ : Fin (C * m)) = c := by
    apply Fin.ext
    simp only [divFin]
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hm, Nat.div_eq_of_lt l.isLt, zero_add]
  have hmd : modFin (⟨c.val * m + l.val, by have := c.isLt; have := l.isLt; nlinarith⟩ : Fin (C * m)) = l := by
    apply Fin.ext
    simp only [modFin]
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt l.isLt]
  simp only [blk, maskPart, blockRow, reduceMatrix, Matrix.add_apply, Matrix.sub_apply,
    Matrix.smul_apply, smul_eq_mul, hd, hmd]

set_option maxHeartbeats 4000000 in
/-- **A part row through a block-diagonal decomposition.** A row `y = s K - (t_c G)_c + E` of
gadget cells times the digits `D` of `diag(x, …, x)` is `s (K D) - (t_c x)_c + E D`. -/
theorem part_block_row {q m C b bound : Nat} (hq : 1 < q) (hn : 0 < n) {backend : BackendContext}
    {base digits : Int} (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0) {G : ExactMatrix q n 1 m}
    (hG : gadgetMatrixRuns backend base digits G)
    {y : ExactMatrix q n 1 (C * m)} {s : ExactPoly q n} {K : Fin 2 → ExactMatrix q n 1 (C * m)}
    {t : Fin C → ExactPoly q n} {E : ErrorMatrix n 1 (C * m)} {B : Nat}
    (hy : y = maskPart s K - blockRow t G + reduceMatrix q n 1 (C * m) E) (hE : CoeffBound E B)
    {x : ExactMatrix q n 1 b} {X : ExactMatrix q n C (C * b)} (hX : IsDiag x X)
    {D' D : ExactMatrix q n (C * m) (C * b)} (hD' : gadgetDecomposeRuns backend base digits X D')
    (hD : compactCenteredRebaseRuns bound D' D) :
    ∃ e : ErrorMatrix n 1 (C * b), CoeffBound e ((C * m) * n * B * (base.toNat / 2)) ∧
      y * D = maskPart s (fun k ↦ K k * D) - blockRow t x + reduceMatrix q n 1 (C * b) e := by
  have hDD : D = D' := RefreshTopProof.GenFinal.compactRebase_self_gen hn hq hD
  subst hDD
  obtain ⟨d, hd, hdb⟩ := gadgetDecomposeRuns_small hq hn hD'
  refine ⟨E * d, coeffBound_mul hn hE hdb, ?_⟩
  rw [hy, Matrix.add_mul, Matrix.sub_mul, maskPart_mul,
    blockRow_mul_decompose hq hn layout hlayout hexact hG t hX hD', hd, ← reduceMatrix_mul]

end Dio2Proof
