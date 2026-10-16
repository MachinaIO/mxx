import ExponentPublic

/-!
# The multi-row gadget

A refresh part decomposes a block-diagonal matrix of `r` rows (`concatDiagonal` of gadgets or
scalars), which needs the `r`-row gadget `G_r = I_r ⊗ g`: its entry `(i, c)` is `g_(c mod m)` when
`i = c / m` and zero otherwise, and `G_r G_r^-1(X) = X` without a modulus drop.
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {n : Nat}

theorem castExactMatrix_congr {a b rows columns rows' columns' : Nat} (h : a = b)
    {M : ExactMatrix a n rows columns} {M' : ExactMatrix a n rows' columns'} {i j i' j'}
    (e : M i j = M' i' j') : castExactMatrix h M i j = castExactMatrix h M' i' j' := by
  subst h
  exact e

theorem castExactMatrix_zero {a b rows columns : Nat} (h : a = b)
    {M : ExactMatrix a n rows columns} {i j} (e : M i j = 0) : castExactMatrix h M i j = 0 := by
  subst h
  exact e

theorem mul_add_cancel_of_lt {a b x y D : Nat} (hx : x < D) (hy : y < D)
    (h : a * D + x = b * D + y) : a = b := by
  have hD : 0 < D := by omega
  have e1 : (a * D + x) / D = a := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hD, Nat.div_eq_of_lt hx, zero_add]
  have e2 : (b * D + y) / D = b := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hD, Nat.div_eq_of_lt hy, zero_add]
  rw [← e1, ← e2, h]

/-- The entries of the `rows`-row gadget are those of the one-row gadget on the diagonal. -/
theorem regularGadgetMatrix_entry {q rows : Nat} (layout : RegularLayout q)
    (hexact : layout.droppedModuli = 0) (i : Fin rows) (c : Fin (rows * layout.digitCount))
    (c1 : Fin (1 * layout.digitCount)) (hc1 : c1.val = c.val % layout.digitCount) :
    (regularGadgetMatrix layout : ExactMatrix q n rows (rows * layout.digitCount)) i c =
      if i.val = c.val / layout.digitCount then
        (regularGadgetMatrix layout : ExactMatrix q n 1 (1 * layout.digitCount)) 0 c1
      else 0 := by
  have hcount : layout.digitCount = layout.crtModuli.length * layout.digitsPerTower := by
    simp [RegularLayout.digitCount, RegularLayout.retainedTowers, hexact]
  have hindex : ∀ {r : Nat} (index : Fin (r * layout.digitCount)),
      (regularColumnIndex layout index).val = index.val := by
    intro r index
    simp [regularColumnIndex, RegularLayout.digitCount, RegularLayout.retainedTowers, hexact,
      finProdFinEquiv, Nat.mod_add_div]
  simp only [regularGadgetMatrix, fullGadgetMatrix]
  set p := (regularIndexEquiv layout rows).symm (regularColumnIndex layout c) with hp
  have hpv := regularIndexEquiv_val layout p.1 p.2.1 p.2.2
  have hpe : regularIndexEquiv layout rows (p.1, ⟨p.2.1, p.2.2⟩) = regularColumnIndex layout c := by
    rw [hp]; simp
  rw [hpe, hindex] at hpv
  have hL := p.2.1.isLt
  have hD := p.2.2.isLt
  have hlt : p.2.1.val * layout.digitsPerTower + p.2.2.val < layout.digitCount := by
    rw [hcount]; nlinarith
  have hdiv : c.val / layout.digitCount = p.1.val := by
    rw [hpv, ← hcount, Nat.add_assoc, Nat.mul_comm, Nat.mul_add_div (by omega), Nat.div_eq_of_lt hlt]
    simp
  have hmod : c.val % layout.digitCount = p.2.1.val * layout.digitsPerTower + p.2.2.val := by
    rw [hpv, ← hcount, Nat.add_assoc, Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hlt]
  split_ifs with hi
  · apply castExactMatrix_congr
    set p1 := (regularIndexEquiv layout 1).symm (regularColumnIndex layout c1) with hp1
    have hp1v := regularIndexEquiv_val layout p1.1 p1.2.1 p1.2.2
    have hp1e : regularIndexEquiv layout 1 (p1.1, ⟨p1.2.1, p1.2.2⟩) =
        regularColumnIndex layout c1 := by
      rw [hp1]; simp
    rw [hp1e, hindex, hc1, hmod] at hp1v
    have h10 : p1.1 = 0 := Subsingleton.elim _ _
    rw [h10] at hp1v
    have hL1 := p1.2.1.isLt
    have hD1 := p1.2.2.isLt
    have ht : p1.2.1 = p.2.1 := by
      apply Fin.ext
      simp only [Fin.val_zero, zero_mul, zero_add] at hp1v
      exact mul_add_cancel_of_lt hD1 hD hp1v.symm
    have hd : p1.2.2.val = p.2.2.val := by
      simp only [Fin.val_zero, zero_mul, zero_add] at hp1v
      rw [ht] at hp1v
      omega
    simp only [regularGadgetUnflattened]
    rw [if_pos (Fin.ext (by rw [hi, hdiv])), if_pos (Subsingleton.elim _ _)]
    congr 2
    exact Sigma.ext ht.symm (heq_of_eq (Fin.ext (by
      show p.2.2.val = p1.2.2.val
      omega)))
  · apply castExactMatrix_zero
    simp only [regularGadgetUnflattened]
    rw [if_neg (fun h ↦ hi (by rw [hdiv, h]))]

set_option maxHeartbeats 4000000 in
/-- **The `r`-row gadget** of a one-row gadget `G` (`m` columns): block-diagonal with `G` on the
diagonal, and times the digits of any `r`-row target it is the target. -/
theorem gadget_rows {q m : Nat} (hq : 1 < q) (hn : 0 < n) {backend : BackendContext}
    {base digits : Int} (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0) {G : ExactMatrix q n 1 m}
    (hG : gadgetMatrixRuns backend base digits G) (r : Nat) :
    ∃ Gb : ExactMatrix q n r (r * m),
      (∀ (i : Fin r) (c : Fin (r * m)) (c1 : Fin m), c1.val = c.val % m →
        Gb i c = if i.val = c.val / m then G 0 c1 else 0) ∧
      ∀ {cols : Nat} {X : ExactMatrix q n r cols} {D : ExactMatrix q n (r * m) cols},
        gadgetDecomposeRuns backend base digits X D → Gb * D = X := by
  obtain ⟨layout₁, hl₁, hb, hd, hw₁, hG⟩ := hG
  rw [hlayout] at hl₁
  cases hl₁
  have hm : m = layout.digitCount := by simpa using hw₁
  subst hm
  refine ⟨regularGadgetMatrix layout, fun i c c1 hc1 ↦ ?_, fun {cols X D} hD ↦ ?_⟩
  · rw [regularGadgetMatrix_entry layout hexact i c (Fin.cast (by simp) c1) (by simpa using hc1), hG,
      castMatrixColumns_apply]
  · obtain ⟨layout₂, hl₂, _, _, hw₂, hD, _⟩ := hD
    rw [hlayout] at hl₂
    cases hl₂
    rw [hD]
    have hrec := regularGadgetMatrix_reconstruct (rows := r) layout X hq hn hexact
    convert hrec using 2

/-- `Y` is block-diagonal with the block `x` (`a × b`) on its diagonal. -/
def IsDiag {q R C a b : Nat} (x : ExactMatrix q n a b) (Y : ExactMatrix q n R C) : Prop :=
  ∀ (i : Fin R) (c : Fin C) (i1 : Fin a) (c1 : Fin b), i1.val = i.val % a → c1.val = c.val % b →
    Y i c = if i.val / a = c.val / b then x i1 c1 else 0

theorem isDiag_self {q a b : Nat} (x : ExactMatrix q n a b) : IsDiag x x := by
  intro i c i1 c1 hi hc
  rw [if_pos (by rw [Nat.div_eq_of_lt i.isLt, Nat.div_eq_of_lt c.isLt])]
  congr 1 <;> apply Fin.ext
  · rw [hi, Nat.mod_eq_of_lt i.isLt]
  · rw [hc, Nat.mod_eq_of_lt c.isLt]

/-- One more block on the diagonal. -/
theorem isDiag_step {q R C a b k : Nat} (ha : 0 < a) (hb : 0 < b) (hR : R = k * a) (hC : C = k * b)
    {x : ExactMatrix q n a b} {Y : ExactMatrix q n R C} (hY : IsDiag x Y)
    {Y' : ExactMatrix q n (R + a) (C + b)} (h : concatDiagonal Y x Y') : IsDiag x Y' := by
  intro i c i1 c1 hi hc
  have hil := i.isLt
  have hcl := c.isLt
  rw [h i c]
  by_cases hiR : i.val < R
  · have hia : i.val / a < k := by rw [Nat.div_lt_iff_lt_mul ha]; omega
    rw [dif_pos hiR]
    by_cases hcC : c.val < C
    · rw [dif_pos hcC]
      exact hY ⟨i.val, hiR⟩ ⟨c.val, hcC⟩ i1 c1 hi hc
    · rw [dif_neg hcC]
      have : k ≤ c.val / b := by rw [Nat.le_div_iff_mul_le hb]; omega
      rw [if_neg (by omega)]
  · have hia : i.val / a = k := by
      apply Nat.div_eq_of_lt_le <;> [nlinarith; nlinarith]
    rw [dif_neg hiR]
    by_cases hcC : c.val < C
    · rw [dif_pos hcC]
      have : c.val / b < k := by rw [Nat.div_lt_iff_lt_mul hb]; omega
      rw [if_neg (by omega)]
    · have hcb : c.val / b = k := by
        apply Nat.div_eq_of_lt_le <;> [nlinarith; nlinarith]
      rw [dif_neg hcC, if_pos (by omega)]
      have e1 : i.val % a = i.val - R := by
        have := Nat.mod_add_div i.val a
        rw [hia] at this
        subst hR; rw [Nat.mul_comm a k] at this; omega
      have e2 : c.val % b = c.val - C := by
        have := Nat.mod_add_div c.val b
        rw [hcb] at this
        subst hC; rw [Nat.mul_comm b k] at this; omega
      congr 1 <;> apply Fin.ext
      · simp only; omega
      · simp only; omega

/-- The block of column `l` of a row of `C` blocks of width `m`. -/
def divFin {C m : Nat} (l : Fin (C * m)) : Fin C :=
  ⟨l.val / m, by
    have hm : 0 < m := Nat.pos_of_ne_zero (by rintro rfl; simpa using l.isLt)
    rw [Nat.div_lt_iff_lt_mul hm]; exact l.isLt⟩

/-- The column of `l` in its block. -/
def modFin {C m : Nat} (l : Fin (C * m)) : Fin m :=
  ⟨l.val % m, Nat.mod_lt _ (Nat.pos_of_ne_zero (by rintro rfl; simpa using l.isLt))⟩

/-- The row of polynomials `t`. -/
def rowVec {q C : Nat} (t : Fin C → ExactPoly q n) : ExactMatrix q n 1 C := fun _ i ↦ t i

/-- The row whose block `c` is `t_c x`. -/
noncomputable def blockRow {q C m : Nat} (t : Fin C → ExactPoly q n) (x : ExactMatrix q n 1 m) :
    ExactMatrix q n 1 (C * m) :=
  fun _ l ↦ t (divFin l) * x 0 (modFin l)

/-- A row of polynomials times a block-diagonal matrix. -/
theorem row_mul_isDiag {q C b : Nat} (t : Fin C → ExactPoly q n) {x : ExactMatrix q n 1 b}
    {X : ExactMatrix q n C (C * b)} (hX : IsDiag x X) :
    rowVec t * X = blockRow t x := by
  funext r l
  rw [Matrix.mul_apply, Finset.sum_eq_single (divFin l)]
  · rw [hX _ l 0 (modFin l) (by rw [Nat.mod_one]; rfl) rfl, if_pos (by simp [divFin])]
    rfl
  · intro i _ hi
    rw [hX i l 0 (modFin l) (by rw [Nat.mod_one]; rfl) rfl,
      if_neg (fun h ↦ hi (Fin.ext (by rw [Nat.div_one] at h; simpa [divFin] using h))), mul_zero]
  · simp

set_option maxHeartbeats 4000000 in
/-- **A row of blocks times the digits of a block-diagonal matrix.** The row `(t_c G)_c` of
gadget blocks times the digits of `X = diag(x, …, x)` is `(t_c x)_c`. -/
theorem blockRow_mul_decompose {q m C b : Nat} (hq : 1 < q) (hn : 0 < n) {backend : BackendContext}
    {base digits : Int} (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0) {G : ExactMatrix q n 1 m}
    (hG : gadgetMatrixRuns backend base digits G) (t : Fin C → ExactPoly q n)
    {x : ExactMatrix q n 1 b} {X : ExactMatrix q n C (C * b)} (hX : IsDiag x X)
    {D : ExactMatrix q n (C * m) (C * b)} (hD : gadgetDecomposeRuns backend base digits X D) :
    blockRow t G * D = blockRow t x := by
  obtain ⟨Gb, hGb, hrec⟩ := gadget_rows hq hn layout hlayout hexact hG C
  have hGb' : IsDiag G Gb := fun i c i1 c1 hi hc ↦ by
    rw [hGb i c c1 hc, Nat.div_one]
    have : i1 = 0 := Subsingleton.elim _ _
    subst this
    rfl
  rw [← row_mul_isDiag t hGb', Matrix.mul_assoc, hrec hD, row_mul_isDiag t hX]

end Dio2Proof
