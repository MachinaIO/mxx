import GenIface
import ExponentIterate

/-!
# Requirement loops, for any parameter set

The requirement module (`Requirement_0.lean`) checks its conditions with sequential Boolean loops:
every iteration conjoins conditions to the state the way the export writes a conjunction,
`next = decide ((cur : 1/0) + (b : 1/0) = 2)`, possibly through an inner loop (a check per member
of a row) or an inner integer sum (a one-hot row sums to one). The root chains the loops: each
loop starts from the previous loop's result, and the module's output is the last result.

This file turns these loops into `∀` facts without unrolling: one lemma per loop shape, each with a
`∀`-quantified step hypothesis (`GenIface.ReqAll.sem` / `IterRuns.invariant`).

* `andB_true`, `famGet_famAt`, `famGet_ofNat`, `select_ofNat`: reading a conjunction, a family
  read and a table selection as values.
* `reqStep_of_inner`, `reqAll_nested`: a step running an inner requirement loop.
* `reqStep_of_sum`: a step comparing an inner strided integer sum (`MemberSum1.Spec`) with a
  constant (the one-hot rows); `oneHot_of_sum`: a `0/1` row summing to one is one-hot.
* `reqAll_chain`, `reqAll_true`: the root's chain of loops.
-/

namespace GenReq

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.GenIface
open scoped BigOperators

/-! ## Values -/

/-- The export's conjunction is true exactly when both sides are. -/
theorem andB_true (a b : Bool) :
    decide (((if a then 1 else 0 : Int) + (if b then 1 else 0)) = 2) = true ↔ a = true ∧ b = true := by
  cases a <;> cases b <;> decide

/-- A family read at an index is the member at the index, in range. -/
theorem famGet_famAt {α : Type} [Zero α] {F : Nat} {f : Fin F → α} {idx : Int} {x : α}
    (h : familyGetDynamic f idx x) : 0 ≤ idx ∧ idx < F ∧ x = famAt f idx.toNat := by
  obtain ⟨p, hp, rfl⟩ := h
  refine ⟨by omega, by omega, ?_⟩
  have hlt : idx.toNat < F := by omega
  rw [famAt_of_lt f hlt]
  exact congrArg f (Fin.ext (by simp only; omega))

/-- A family read at a natural index. -/
theorem famGet_ofNat {α : Type} [Zero α] {F : Nat} {f : Fin F → α} {i : Nat} {x : α}
    (h : familyGetDynamic f (Int.ofNat i) x) : i < F ∧ x = famAt f i := by
  obtain ⟨_, h1, h2⟩ := famGet_famAt h
  rw [Int.ofNat_eq_natCast] at h1
  exact ⟨by omega, h2⟩

/-- A table selection at a natural index is the table's entry. -/
theorem select_ofNat {table : List Int} {i : Nat} {out : Int}
    (h : MxxRuntime.select (Int.ofNat i) table out) : out = MxxIR.selectInt (Int.ofNat i) table :=
  select_eq_selectInt h

/-! ## Steps -/

/-- A step whose conjunct is a condition `b`: the state stays true only if `b` holds. (The
export's form of `GenIface.ReqStep.of_and`, with the condition as a `Prop`.) -/
theorem reqStep_of_decide {P : Nat → Prop} {i : Nat} {cur next : Bool} {p : Prop} [Decidable p]
    (hp : p → P i)
    (hnext : next = decide (((if cur then 1 else 0 : Int) + (if decide p then 1 else 0)) = 2)) :
    ReqStep.Spec P i cur next :=
  ReqStep.of_and (fun hb ↦ hp (of_decide_eq_true hb)) hnext

/-- A step that runs an inner requirement loop of `count` iterations from the state: the state
stays true only if every inner condition holds. -/
theorem reqStep_of_inner {P : Nat → Prop} {Q : Nat → Prop} {i count : Nat} {cur next : Bool}
    {body : Nat → Bool → Bool → Prop}
    (hstep : ∀ k c c', body k c c' → ReqStep.Spec Q k c c')
    (h : MxxIR.IterRuns body count cur next) (hP : (∀ k < count, Q k) → P i) :
    ReqStep.Spec P i cur next := fun hn ↦
  have ⟨hc, hall⟩ := ReqAll.sem h hstep hn
  ⟨hc, hP hall⟩

/-- A requirement loop whose iteration `i` runs an inner requirement loop of `count'` iterations
checking `Q i k`: a true result means `Q i k` for all `i < count`, `k < count'`. -/
theorem reqAll_nested {Q : Nat → Nat → Prop} {count count' : Nat} {init final : Bool}
    {body : Nat → Bool → Bool → Prop}
    (hstep : ∀ i cur next, body i cur next → ∃ inner : Nat → Bool → Bool → Prop,
      MxxIR.IterRuns inner count' cur next ∧ ∀ k c c', inner k c c' → ReqStep.Spec (Q i) k c c')
    (h : MxxIR.IterRuns body count init final) :
    ReqAll.Spec (fun i ↦ ∀ k < count', Q i k) count init final :=
  ReqAll.sem h (fun i cur next hb ↦
    have ⟨_, hrun, hin⟩ := hstep i cur next hb
    reqStep_of_inner hin hrun id)

/-- A step comparing an inner strided integer sum (started at `0`) with `c`: the state stays
true only if the members `idx k`, `k < count`, of `f` sum to `c`. -/
theorem reqStep_of_sum {P : Nat → Prop} {i : Nat} {cur next : Bool} {F count : Nat}
    {f : Fin F → Int} {idx : Nat → Int} {s0 s c : Int} (hs0 : s0 = 0)
    (hsum : MemberSum1.Spec f idx count s0 s)
    (hnext : next = decide (((if cur then 1 else 0 : Int) + (if decide (s = c) then 1 else 0)) = 2))
    (hP : (∑ k ∈ Finset.range count, famAt f (idx k).toNat) = c → P i) :
    ReqStep.Spec P i cur next :=
  reqStep_of_decide (fun he ↦ hP (by rw [← he, hsum.2, hs0, zero_add])) hnext

/-- A row of `0/1` entries summing to one is one-hot. -/
theorem oneHot_of_sum {W : Nat} {x : Nat → Int} (hbit : ∀ k < W, x k = 0 ∨ x k = 1)
    (hsum : ∑ k ∈ Finset.range W, x k = 1) :
    ∃ k < W, x k = 1 ∧ ∀ k' < W, k' ≠ k → x k' = 0 := by
  classical
  have hnn : ∀ k ∈ Finset.range W, 0 ≤ x k := fun k hk ↦ by
    rcases hbit k (Finset.mem_range.1 hk) with h | h <;> simp [h]
  obtain ⟨k, hk, hxk⟩ : ∃ k < W, x k = 1 := by
    by_contra hno
    push_neg at hno
    have : ∑ k ∈ Finset.range W, x k = 0 := Finset.sum_eq_zero fun k hk ↦ by
      rcases hbit k (Finset.mem_range.1 hk) with h | h
      · exact h
      · exact absurd h (hno k (Finset.mem_range.1 hk))
    omega
  refine ⟨k, hk, hxk, fun k' hk' hne ↦ ?_⟩
  rcases hbit k' hk' with h | h
  · exact h
  · exfalso
    have hsub : ({k, k'} : Finset Nat) ⊆ Finset.range W := by
      intro y hy
      simp only [Finset.mem_insert, Finset.mem_singleton] at hy
      rcases hy with rfl | rfl <;> simpa
    have := Finset.sum_le_sum_of_subset_of_nonneg hsub fun y hy _ ↦ hnn y hy
    rw [Finset.sum_pair (Ne.symm hne), hxk, h] at this
    omega

/-! ## The root's chain -/

/-- Two chained requirement loops (the second starts from the first's result). -/
theorem reqAll_chain {P P' : Nat → Prop} {c c' : Nat} {a b d : Bool}
    (h1 : ReqAll.Spec P c a b) (h2 : ReqAll.Spec P' c' b d) :
    d = true → a = true ∧ (∀ i < c, P i) ∧ ∀ i < c', P' i := fun hd ↦
  have ⟨hb, h2'⟩ := h2 hd
  have ⟨ha, h1'⟩ := h1 hb
  ⟨ha, h1', h2'⟩

/-- A requirement loop's facts from a true result. -/
theorem reqAll_true {P : Nat → Prop} {count : Nat} {init final : Bool}
    (h : ReqAll.Spec P count init final) (hf : final = true) : ∀ i < count, P i :=
  (h hf).2

end GenReq
