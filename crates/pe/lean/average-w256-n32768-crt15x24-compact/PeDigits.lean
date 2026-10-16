import PeDecodeRoot

/-!
# Digits of the switching and attribute ciphertexts

The online column digits of a ciphertext `(top, bot)` are, in their bottom rows, the digits of the
bottom entries alone (as encryption computes its companions) and, in their top rows, the digits of the
public top row (as preprocessing computes the public top terms).
-/

set_option maxRecDepth 200000

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The column digits exist for every column of the state modulus. -/
theorem column_runs (x y : ExactPoly Q N) :
    ∃ D' : ExactMatrix Q N (30 + 30) 1,
      gadgetDecomposeRuns Backend.backend 4096 30 (pairColumn x y) D' ∧
        compactCenteredRebaseRuns 2048 D' (columnDigits Q (30 + 30) Backend.backend 4096 30 2048 x y) :=
  columnDigits_runs hQ hQ hN g0_runs Backend.layout0 layoutQ rfl rfl
    (by have : Backend.layout0.base = 4096 := rfl; rw [this]; norm_num)
    (by have : Backend.layout0.base = 4096 := rfl; rw [this]; norm_num) x y

/-- A decomposition of one entry rebases to itself. -/
theorem entry_rebase (y : ExactPoly Q N) :
    compactCenteredRebaseRuns 2048 (entryDigits 30 Backend.backend 4096 30 (fun _ _ ↦ y) : ExactMatrix Q N 30 1)
      (entryDigits 30 Backend.backend 4096 30 (fun _ _ ↦ y)) := by
  have hb := (decompose_entry hQ hN Backend.layout0 layoutQ rfl g0_runs
    (entryDigits_runs hQ hN g0_runs Backend.layout0 layoutQ rfl (fun _ _ ↦ y))).2
  exact ⟨Or.inl dvd_rfl, hb, centeredRebaseRuns_self hQ hN _, hb⟩

/-- Bottom digit row `a` of the column `(x, y)` is digit `a` of `y`. -/
theorem columnDigits_bottom (x y : ExactPoly Q N) (a : Fin 30) :
    columnDigits Q (30 + 30) Backend.backend 4096 30 2048 x y (Fin.natAdd 30 a) 0 =
      reducePoly Q N (entryDigit y a) := by
  rw [(entryDigit_spec y a).1]
  symm
  have := columnDigits_entry (q := Q) (K := 30) (backend := Backend.backend) (base := 4096) (digits := 30)
    (bound := 2048) rfl (x := x) (y := y) (e := fun _ _ ↦ y) (Fin.natAdd 30 a)
    (by simp [pairColumn, Fin.natAdd]) (entryDigits_runs hQ hN g0_runs Backend.layout0 layoutQ rfl _)
    (entry_rebase y) (column_runs x y)
  have ha : (⟨(Fin.natAdd 30 a : Nat) % 30, by omega⟩ : Fin 30) = a := Fin.ext (by simp)
  rw [← this, ha]

/-- Bottom row `a` of the action digits of `(top, bot)` at component `j` and entry `l`. -/
theorem actionDigits_bottom (top bot : ExactMatrix Q N 1 60) (j : Fin 2) (a : Fin 30) (l : Fin 30) :
    actionDigits top bot j (Fin.natAdd 30 a) l =
      reducePoly Q N (entryDigit (bot 0 ⟨30 * (j : Nat) + l, by omega⟩) a) :=
  columnDigits_bottom _ _ a

/-- The digits of a row of the state modulus run and rebase to `rowDigits`. -/
theorem rowDigits_runs (R : ExactMatrix Q N 1 60) :
    ∃ D' : ExactMatrix Q N 30 60, gadgetDecomposeRuns Backend.backend 4096 30 R D' ∧
      compactCenteredRebaseRuns 2048 D' (rowDigits R) := by
  obtain ⟨layout', hl, hb, hd, hw, -⟩ := g0_runs
  rw [layoutQ] at hl
  cases hl
  have hw2 : 30 = 1 * Backend.layout0.digitCount := by simp at hw; omega
  let D' : ExactMatrix Q N 30 60 := castMatrixRows hw2.symm (regularDecomposeMatrix Backend.layout0 R)
  have hD' : gadgetDecomposeRuns Backend.backend 4096 30 R D' :=
    ⟨Backend.layout0, layoutQ, hb, hd, hw2, rfl,
      regularGadgetMatrix_residual_exact Backend.layout0 R hQ hN rfl⟩
  obtain ⟨wit, hwit, hwb⟩ := regularDecomposeMatrix_bounded Backend.layout0 R hQ hN
  have hbase : Backend.layout0.base / 2 = 2048 := rfl
  have hbD' : PreimageWithin D' 2048 := by
    refine ⟨fun row column ↦ wit (Fin.cast hw2 row) column, ?_, fun row column ↦ ?_⟩
    · funext row column
      simp only [D', castMatrixRows_apply, hwit, reduceMatrix]
    · rw [← hbase]
      exact hwb _ _
  have hreb : compactCenteredRebaseRuns 2048 D' D' :=
    ⟨Or.inl dvd_rfl, hbD', centeredRebaseRuns_self hQ hN _, hbD'⟩
  exact ⟨D', hD', by rw [← rowDigits_eq hD' hreb]; exact hreb⟩

/-- Top digit row `a` of the column `(x, y)` is row `a` of the digits of the row `R` at `jj`, when
`R_jj = x`. -/
theorem columnDigits_top (R : ExactMatrix Q N 1 60) (jj : Fin 60) (y : ExactPoly Q N) (a : Fin 30) :
    columnDigits Q (30 + 30) Backend.backend 4096 30 2048 (R 0 jj) y (Fin.castAdd 30 a) 0 = rowDigits R a jj := by
  obtain ⟨D', hD', hreb⟩ := column_runs (R 0 jj) y
  obtain ⟨E', hE', hE⟩ := rowDigits_runs R
  rw [hreb.2.2.1.2, hE.2.2.1.2]
  apply centeredRebase_congr
  exact gadgetDecomposeRuns_congr hD' hE' (r := 0) (c := 0) (r' := 0) (c' := jj)
    (by simp [pairColumn]) rfl (k := a) a.isLt (by simp) (by simp)

/-- Top row `a` of the action digits of `(top, bot)` at component `j` and entry `l`. -/
theorem actionDigits_top (top bot : ExactMatrix Q N 1 60) (j : Fin 2) (a : Fin 30) (l : Fin 30) :
    actionDigits top bot j (Fin.castAdd 30 a) l = rowDigits top a ⟨30 * (j : Nat) + l, by omega⟩ :=
  columnDigits_top top _ _ a

end PeProof
