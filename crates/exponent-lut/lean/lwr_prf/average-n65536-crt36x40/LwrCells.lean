import LwrSetup

/-!
# Cell families and their sums

The cells of a block form families of `113` matrices; the generated sum adds cells `0` and `1` in
the root and the others in a loop.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The six blocks `(A₀₀, A₀₁, A₁₀, A₁₁, c₀, c₁)` of a summed encoding. -/
abbrev SumState (q m : Nat) :=
  ExactMatrix q N 1 m × ExactMatrix q N 1 m × ExactMatrix q N 1 m × ExactMatrix q N 1 m ×
    ExactMatrix q N 1 m × ExactMatrix q N 1 m × Unit

/-- Member `k` of a family of 113 cells, zero past the family. -/
noncomputable def cellAt {q m : Nat} (f : Fin 113 → ExactMatrix q N 1 m) (k : Nat) :
    ExactMatrix q N 1 m :=
  if h : k < 113 then f ⟨k, h⟩ else 0

/-- Cells `2, ..., k + 1` added to an initial state. -/
noncomputable def addCells {q m : Nat} (init : SumState q m)
    (f0 f1 f2 f3 f4 f5 : Fin 113 → ExactMatrix q N 1 m) (k : Nat) : SumState q m :=
  (init.1 + ∑ l ∈ Finset.range k, cellAt f0 (2 + l),
    init.2.1 + ∑ l ∈ Finset.range k, cellAt f1 (2 + l),
    init.2.2.1 + ∑ l ∈ Finset.range k, cellAt f2 (2 + l),
    init.2.2.2.1 + ∑ l ∈ Finset.range k, cellAt f3 (2 + l),
    init.2.2.2.2.1 + ∑ l ∈ Finset.range k, cellAt f4 (2 + l),
    init.2.2.2.2.2.1 + ∑ l ∈ Finset.range k, cellAt f5 (2 + l), ())

/-- A cell read at the shifted index `2 + i`. -/
theorem familyGet_cell {q m : Nat} {f : Fin 113 → ExactMatrix q N 1 m} {i : Nat} {out : ExactMatrix q N 1 m}
    (h : familyGetDynamic f (2 + Int.ofNat i) out) : out = cellAt f (2 + i) := by
  obtain ⟨hk, hout⟩ := familyGet_eq (k := 2 + i) (by
    rw [show (Int.ofNat (2 + i)) = 2 + Int.ofNat i by simp only [Int.ofNat_eq_natCast]; push_cast; ring]
    exact h)
  rw [hout, cellAt, dif_pos hk]

/-- Cells `0` and `1` plus cells `2, ..., 112` are all 113 cells. -/
theorem sum_cells {q m : Nat} (f : Fin 113 → ExactMatrix q N 1 m) :
    f ⟨0, by decide⟩ + f ⟨1, by decide⟩ + ∑ l ∈ Finset.range 111, cellAt f (2 + l) =
      ∑ i : Fin 113, f i := by
  have hall : ∑ i : Fin 113, f i = ∑ k ∈ Finset.range 113, cellAt f k := by
    rw [← Fin.sum_univ_eq_sum_range]
    apply Finset.sum_congr rfl
    intro i _
    rw [cellAt, dif_pos i.isLt]
  have h := Finset.sum_range_add (fun k ↦ cellAt f k) 2 111
  rw [show (2 + 111 : Nat) = 113 from rfl] at h
  rw [hall, h]
  congr 1
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add]
  rfl

/-- A family read at a static index. -/
theorem familyGetStatic_eq {α : Type} {count : Nat} {f : Fin count → α} {k : Nat} {out : α}
    (h : familyGetStatic f (Int.ofNat k) out) : ∃ hk : k < count, out = f ⟨k, hk⟩ :=
  familyGet_eq h

end LwrProof
