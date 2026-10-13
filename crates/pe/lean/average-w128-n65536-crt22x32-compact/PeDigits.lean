import PeDecodeRoot

/-!
# Digits of the switching and attribute ciphertexts

The online column digits of a ciphertext `(top, bot)` are, in their bottom rows, the digits of the
bottom entries alone (as encryption computes its companions) and, in their top rows, the digits of the
public top row (as preprocessing computes the public top terms).
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The column digits exist for every column of the state modulus. -/
theorem column_runs (x y : ExactPoly Q N) :
    ∃ D' : ExactMatrix Q N (44 + 44) 1,
      gadgetDecomposeRuns Backend.backend 65536 44 (pairColumn x y) D' ∧
        compactCenteredRebaseRuns 32768 D' (columnDigits Q (44 + 44) Backend.backend 65536 44 32768 x y) :=
  columnDigits_runs hQ hQ hN g0_runs Backend.layout0 layoutQ rfl rfl
    (by have : Backend.layout0.base = 65536 := rfl; rw [this]; norm_num)
    (by have : Backend.layout0.base = 65536 := rfl; rw [this]; norm_num) x y

/-- A decomposition of one entry rebases to itself. -/
theorem entry_rebase (y : ExactPoly Q N) :
    compactCenteredRebaseRuns 32768 (entryDigits 44 Backend.backend 65536 44 (fun _ _ ↦ y) : ExactMatrix Q N 44 1)
      (entryDigits 44 Backend.backend 65536 44 (fun _ _ ↦ y)) := by
  have hb := (decompose_entry hQ hN Backend.layout0 layoutQ rfl g0_runs
    (entryDigits_runs hQ hN g0_runs Backend.layout0 layoutQ rfl (fun _ _ ↦ y))).2
  exact ⟨Or.inl dvd_rfl, hb, centeredRebaseRuns_self hQ hN _, hb⟩

/-- Bottom digit row `a` of the column `(x, y)` is digit `a` of `y`. -/
theorem columnDigits_bottom (x y : ExactPoly Q N) (a : Fin 44) :
    columnDigits Q (44 + 44) Backend.backend 65536 44 32768 x y (Fin.natAdd 44 a) 0 =
      reducePoly Q N (entryDigit y a) := by
  rw [(entryDigit_spec y a).1]
  symm
  have := columnDigits_entry (q := Q) (K := 44) (backend := Backend.backend) (base := 65536) (digits := 44)
    (bound := 32768) rfl (x := x) (y := y) (e := fun _ _ ↦ y) (Fin.natAdd 44 a)
    (by simp [pairColumn, Fin.natAdd]) (entryDigits_runs hQ hN g0_runs Backend.layout0 layoutQ rfl _)
    (entry_rebase y) (column_runs x y)
  have ha : (⟨(Fin.natAdd 44 a : Nat) % 44, by omega⟩ : Fin 44) = a := Fin.ext (by simp)
  rw [← this, ha]

/-- Bottom row `a` of the action digits of `(top, bot)` at component `j` and entry `l`. -/
theorem actionDigits_bottom (top bot : ExactMatrix Q N 1 88) (j : Fin 2) (a : Fin 44) (l : Fin 44) :
    actionDigits top bot j (Fin.natAdd 44 a) l =
      reducePoly Q N (entryDigit (bot 0 ⟨44 * (j : Nat) + l, by omega⟩) a) :=
  columnDigits_bottom _ _ a

/-- The digits of a row of the state modulus run and rebase to `rowDigits`. -/
theorem rowDigits_runs (R : ExactMatrix Q N 1 88) :
    ∃ D' : ExactMatrix Q N 44 88, gadgetDecomposeRuns Backend.backend 65536 44 R D' ∧
      compactCenteredRebaseRuns 32768 D' (rowDigits R) := by
  obtain ⟨layout', hl, hb, hd, hw, -⟩ := g0_runs
  rw [layoutQ] at hl
  cases hl
  have hw2 : 44 = 1 * Backend.layout0.digitCount := by simp at hw; omega
  let D' : ExactMatrix Q N 44 88 := castMatrixRows hw2.symm (regularDecomposeMatrix Backend.layout0 R)
  have hD' : gadgetDecomposeRuns Backend.backend 65536 44 R D' :=
    ⟨Backend.layout0, layoutQ, hb, hd, hw2, rfl,
      regularGadgetMatrix_residual_exact Backend.layout0 R hQ hN rfl⟩
  obtain ⟨wit, hwit, hwb⟩ := regularDecomposeMatrix_bounded Backend.layout0 R hQ hN
  have hbase : Backend.layout0.base / 2 = 32768 := rfl
  have hbD' : PreimageWithin D' 32768 := by
    refine ⟨fun row column ↦ wit (Fin.cast hw2 row) column, ?_, fun row column ↦ ?_⟩
    · funext row column
      simp only [D', castMatrixRows_apply, hwit, reduceMatrix]
    · rw [← hbase]
      exact hwb _ _
  have hreb : compactCenteredRebaseRuns 32768 D' D' :=
    ⟨Or.inl dvd_rfl, hbD', centeredRebaseRuns_self hQ hN _, hbD'⟩
  exact ⟨D', hD', by rw [← rowDigits_eq hD' hreb]; exact hreb⟩

/-- Top digit row `a` of the column `(x, y)` is row `a` of the digits of the row `R` at `jj`, when
`R_jj = x`. -/
theorem columnDigits_top (R : ExactMatrix Q N 1 88) (jj : Fin 88) (y : ExactPoly Q N) (a : Fin 44) :
    columnDigits Q (44 + 44) Backend.backend 65536 44 32768 (R 0 jj) y (Fin.castAdd 44 a) 0 = rowDigits R a jj := by
  obtain ⟨D', hD', hreb⟩ := column_runs (R 0 jj) y
  obtain ⟨E', hE', hE⟩ := rowDigits_runs R
  rw [hreb.2.2.1.2, hE.2.2.1.2]
  apply centeredRebase_congr
  exact gadgetDecomposeRuns_congr hD' hE' (r := 0) (c := 0) (r' := 0) (c' := jj)
    (by simp [pairColumn]) rfl (k := a) a.isLt (by simp) (by simp)

/-- Top row `a` of the action digits of `(top, bot)` at component `j` and entry `l`. -/
theorem actionDigits_top (top bot : ExactMatrix Q N 1 88) (j : Fin 2) (a : Fin 44) (l : Fin 44) :
    actionDigits top bot j (Fin.castAdd 44 a) l = rowDigits top a ⟨44 * (j : Nat) + l, by omega⟩ :=
  columnDigits_top top _ _ a

end PeProof
