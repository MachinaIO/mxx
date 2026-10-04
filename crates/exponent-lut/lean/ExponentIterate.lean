import IRIterRuns
import IRExpr
import RuntimeMatrixOps
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Fin

/-!
# Sums and conjunctions computed by generated loops

A generated `iterate` loop is an `IterRuns` relation. When every iteration adds a value to an
integer state, the final state is the initial state plus the sum of the values; when every
iteration conjoins a condition to a Boolean state, a true final state means a true initial state
and every condition.
-/

namespace MxxExponentLut

open MxxIR
open scoped BigOperators

/-- A loop whose iteration `i` adds `f i` computes the sum of the first `count` values. -/
theorem IterRuns.sum_of_step {body : Nat → Int → Int → Prop} {count : Nat} {init final : Int}
    (f : Nat → Int) (hstep : ∀ i current next, i < count → body i current next → next = current + f i)
    (h : IterRuns body count init final) : final = init + ∑ i ∈ Finset.range count, f i := by
  have := IterRuns.invariant
    (Invariant := fun k (s : Int) ↦ k ≤ count → s = init + ∑ i ∈ Finset.range k, f i)
    (by intro _; simp) (fun i current next hcur hbody hk ↦ by
      rw [hstep i current next (by omega) hbody, hcur (by omega), Finset.sum_range_succ, add_assoc])
    h
  exact this le_rfl

/-- A loop whose iteration `i` keeps a true state only when `P i` holds: a true final state means a
true initial state and every condition. -/
theorem IterRuns.all_of_step {body : Nat → Bool → Bool → Prop} {count : Nat} {init final : Bool}
    (P : Nat → Prop)
    (hstep : ∀ i current next, i < count → body i current next → next = true →
      current = true ∧ P i)
    (h : IterRuns body count init final) (hfinal : final = true) :
    init = true ∧ ∀ i < count, P i := by
  have := IterRuns.invariant
    (Invariant := fun k (s : Bool) ↦ k ≤ count → s = true → init = true ∧ ∀ i < k, P i)
    (fun _ hs ↦ ⟨hs, fun _ hi ↦ absurd hi (Nat.not_lt_zero _)⟩)
    (fun i current next hcur hbody hk hnext ↦ by
      obtain ⟨hc, hp⟩ := hstep i current next (by omega) hbody hnext
      obtain ⟨hinit, hall⟩ := hcur (by omega) hc
      exact ⟨hinit, fun j hj ↦ if hji : j < i then hall j hji else by
        obtain rfl : j = i := by omega
        exact hp⟩)
    h
  exact this le_rfl hfinal

/-- A sum over `Finset.range` of a family indexed by `Fin count` is the sum over `Fin count`. -/
theorem sum_range_fin {count : Nat} (f : Fin count → Int) :
    ∑ i ∈ Finset.range count, (if h : i < count then f ⟨i, h⟩ else 0) = ∑ i, f i := by
  rw [Finset.sum_range (fun i ↦ if h : i < count then f ⟨i, h⟩ else 0)]
  exact Finset.sum_congr rfl fun i _ ↦ by simp

/-! ## Constant tables -/

/-- Selecting entry `i` of a table of naturals cast to integers casts entry `i`. -/
theorem selectInt_map_ofNat (table : List Nat) (i : Nat) :
    MxxIR.selectInt (Int.ofNat i) (table.map Int.ofNat) = Int.ofNat (table.getD i 0) := by
  unfold MxxIR.selectInt
  rw [show (Int.ofNat i).toNat = i from rfl]
  by_cases hi : i < table.length
  · simp [List.getD, hi]
  · have h1 : (table.map Int.ofNat)[i]? = none := List.getElem?_eq_none (by simp; omega)
    have h2 : table[i]? = none := List.getElem?_eq_none (by omega)
    simp [List.getD, h1, h2]

/-- The runtime selection of entry `i` is `selectInt`. -/
theorem select_eq_selectInt {table : List Int} {i : Nat} {out : Int}
    (h : MxxRuntime.select (Int.ofNat i) table out) : out = MxxIR.selectInt (Int.ofNat i) table := by
  obtain ⟨p, hp, rfl⟩ := h
  have hpi : p.val = i := Int.ofNat.inj hp
  unfold MxxIR.selectInt
  rw [show (Int.ofNat i).toNat = i from rfl, ← hpi]
  simp

/-- Every entry of a table bounded by `M > 0`, and its default `0`, is below `M`. -/
theorem getD_lt_of_all {table : List Nat} {M : Nat} (hM : 0 < M)
    (h : table.all (· < M) = true) (i : Nat) : table.getD i 0 < M := by
  rw [List.getD_eq_getElem?_getD]
  cases hget : table[i]? with
  | none => exact hM
  | some x =>
    have hx : x ∈ table := List.mem_of_getElem? hget
    simpa using List.all_eq_true.mp h x hx

end MxxExponentLut
