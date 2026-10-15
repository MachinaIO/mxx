import GenReq
import W_Requirement_0

/-! The facts the requirement module checks (written by `generator/gen_requirement2.py` from
`W_Requirement_0.lean`): one theorem per loop scope (`sum_…`: the step adds a term; `step_…`: a true state
stays true only if the step's conditions hold) and `requirement_facts`: a true result gives every
condition of the root and every loop's conditions at every index (`Facts`). No loop is unrolled:
each loop fact comes from `GenIface.ReqAll.sem` / `GenIface.seq_sum` with the child's theorem as
the step hypothesis. -/

set_option maxRecDepth 65536
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
-- The root theorem of a large requirement exceeds the default budget (one-off cost).
set_option maxHeartbeats 0

namespace Dio2Requirement

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.GenIface
open scoped BigOperators

/-- The export's conjunction: a true `decide ((a:1/0) + (decide p:1/0) = 2)` gives `a` and `p`. -/
theorem link_and {a : Bool} {p : Prop} {inst : Decidable p}
    (h : decide (((if a = true then 1 else 0 : Int) + (if @decide p inst = true then 1 else 0)) = 2) = true) :
    a = true ∧ p := by
  cases a <;> by_cases hp : p <;> simp_all

/-- A requirement loop's true result gives a true initial state and the conditions. -/
theorem link_loop {P : Nat → Prop} {c : Nat} {a b : Bool} (h : ReqAll.Spec P c a b)
    (hb : b = true) : a = true ∧ ∀ i < c, P i := h hb

/-- A runtime table selection is `selectInt`. -/
theorem select_int {table : List Int} {idx out : Int} (h : MxxRuntime.select idx table out) :
    out = MxxIR.selectInt idx table := by
  obtain ⟨p, hp, rfl⟩ := h
  unfold MxxIR.selectInt
  have : idx.toNat = p.val := by omega
  rw [this]
  simp

/-- One step of `sequential_sequential_generatedRoot_298_5`: it adds a term to the state. -/
theorem sum_sequential_sequential_generatedRoot_298_5 {params : W_Requirement_0.Params} {j0 n0 : Nat} {cur next : Int} {x1 : Fin 72000 → Int} {x2 : Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_298_5 params j0 n0 (cur, (x1, (x2, ()))) next) :
    next = cur + (famAt x1 ((x2 * 2) + (Int.ofNat n0)).toNat) := by
  obtain ⟨w_7_0, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨-, hct2⟩ := hct1
  obtain ⟨a2, hct3⟩ := hct2
  have hout := hct3
  clear hct3
  have e_w_7_0 : w_7_0 = famAt x1 ((x2 * 2) + (Int.ofNat n0)).toNat := (GenReq.famGet_famAt a2).2.2
  clear a2
  subst e_w_7_0
  exact hout

/-- One step of `sequential_generatedRoot_298`: a true state stays true only if its conditions hold. -/
theorem step_sequential_generatedRoot_298 {params : W_Requirement_0.Params} {n1 : Nat} {cur next : Bool} {x1 : Fin 72000 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_298 params n1 (cur, (x1, ())) next) :
    ReqStep.Spec (fun n1 ↦ (0 + ∑ n0 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n1) * 2) + (Int.ofNat n0)).toNat)) = 1) n1 cur next := by
  obtain ⟨⟨w_5_0⟩, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨a1, hct2⟩ := hct1
  have hout := hct2
  clear hct2
  have e_w_5_0 : w_5_0 = 0 + ∑ n0 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n1) * 2) + (Int.ofNat n0)).toNat) := seq_sum (fun n0 ↦ famAt x1 (((Int.ofNat n1) * 2) + (Int.ofNat n0)).toNat) (fun _ _ _ hb ↦ sum_sequential_sequential_generatedRoot_298_5 hb) a1
  clear a1
  subst e_w_5_0
  intro hn
  rw [hout] at hn
  obtain ⟨hn1, f0⟩ := link_and hn
  exact ⟨hn1, f0⟩

/-- One step of `sequential_sequential_generatedRoot_300_5`: it adds a term to the state. -/
theorem sum_sequential_sequential_generatedRoot_300_5 {params : W_Requirement_0.Params} {j0 n2 : Nat} {cur next : Int} {x1 : Fin 72000 → Int} {x2 : Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_300_5 params j0 n2 (cur, (x1, (x2, ()))) next) :
    next = cur + (famAt x1 ((x2 * 2) + (Int.ofNat n2)).toNat) := by
  obtain ⟨w_7_0, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨-, hct2⟩ := hct1
  obtain ⟨a2, hct3⟩ := hct2
  have hout := hct3
  clear hct3
  have e_w_7_0 : w_7_0 = famAt x1 ((x2 * 2) + (Int.ofNat n2)).toNat := (GenReq.famGet_famAt a2).2.2
  clear a2
  subst e_w_7_0
  exact hout

/-- One step of `sequential_generatedRoot_300`: a true state stays true only if its conditions hold. -/
theorem step_sequential_generatedRoot_300 {params : W_Requirement_0.Params} {n3 : Nat} {cur next : Bool} {x1 : Fin 72000 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_300 params n3 (cur, (x1, ())) next) :
    ReqStep.Spec (fun n3 ↦ (0 + ∑ n2 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n3) * 2) + (Int.ofNat n2)).toNat)) = 1) n3 cur next := by
  obtain ⟨⟨w_5_0⟩, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨a1, hct2⟩ := hct1
  have hout := hct2
  clear hct2
  have e_w_5_0 : w_5_0 = 0 + ∑ n2 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n3) * 2) + (Int.ofNat n2)).toNat) := seq_sum (fun n2 ↦ famAt x1 (((Int.ofNat n3) * 2) + (Int.ofNat n2)).toNat) (fun _ _ _ hb ↦ sum_sequential_sequential_generatedRoot_300_5 hb) a1
  clear a1
  subst e_w_5_0
  intro hn
  rw [hout] at hn
  obtain ⟨hn1, f0⟩ := link_and hn
  exact ⟨hn1, f0⟩

/-- One step of `sequential_sequential_generatedRoot_302_5`: it adds a term to the state. -/
theorem sum_sequential_sequential_generatedRoot_302_5 {params : W_Requirement_0.Params} {j0 n4 : Nat} {cur next : Int} {x1 : Fin 16 → Int} {x2 : Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_302_5 params j0 n4 (cur, (x1, (x2, ()))) next) :
    next = cur + (famAt x1 ((x2 * 2) + (Int.ofNat n4)).toNat) := by
  obtain ⟨w_7_0, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨-, hct2⟩ := hct1
  obtain ⟨a2, hct3⟩ := hct2
  have hout := hct3
  clear hct3
  have e_w_7_0 : w_7_0 = famAt x1 ((x2 * 2) + (Int.ofNat n4)).toNat := (GenReq.famGet_famAt a2).2.2
  clear a2
  subst e_w_7_0
  exact hout

/-- One step of `sequential_generatedRoot_302`: a true state stays true only if its conditions hold. -/
theorem step_sequential_generatedRoot_302 {params : W_Requirement_0.Params} {n5 : Nat} {cur next : Bool} {x1 : Fin 16 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_302 params n5 (cur, (x1, ())) next) :
    ReqStep.Spec (fun n5 ↦ (0 + ∑ n4 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n5) * 2) + (Int.ofNat n4)).toNat)) = 1) n5 cur next := by
  obtain ⟨⟨w_5_0⟩, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨a1, hct2⟩ := hct1
  have hout := hct2
  clear hct2
  have e_w_5_0 : w_5_0 = 0 + ∑ n4 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n5) * 2) + (Int.ofNat n4)).toNat) := seq_sum (fun n4 ↦ famAt x1 (((Int.ofNat n5) * 2) + (Int.ofNat n4)).toNat) (fun _ _ _ hb ↦ sum_sequential_sequential_generatedRoot_302_5 hb) a1
  clear a1
  subst e_w_5_0
  intro hn
  rw [hout] at hn
  obtain ⟨hn1, f0⟩ := link_and hn
  exact ⟨hn1, f0⟩

/-- One step of `sequential_sequential_generatedRoot_304_5`: it adds a term to the state. -/
theorem sum_sequential_sequential_generatedRoot_304_5 {params : W_Requirement_0.Params} {j0 n6 : Nat} {cur next : Int} {x1 : Fin 16 → Int} {x2 : Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_304_5 params j0 n6 (cur, (x1, (x2, ()))) next) :
    next = cur + (famAt x1 ((x2 * 2) + (Int.ofNat n6)).toNat) := by
  obtain ⟨w_7_0, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨-, hct2⟩ := hct1
  obtain ⟨a2, hct3⟩ := hct2
  have hout := hct3
  clear hct3
  have e_w_7_0 : w_7_0 = famAt x1 ((x2 * 2) + (Int.ofNat n6)).toNat := (GenReq.famGet_famAt a2).2.2
  clear a2
  subst e_w_7_0
  exact hout

/-- One step of `sequential_generatedRoot_304`: a true state stays true only if its conditions hold. -/
theorem step_sequential_generatedRoot_304 {params : W_Requirement_0.Params} {n7 : Nat} {cur next : Bool} {x1 : Fin 16 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_304 params n7 (cur, (x1, ())) next) :
    ReqStep.Spec (fun n7 ↦ (0 + ∑ n6 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n7) * 2) + (Int.ofNat n6)).toNat)) = 1) n7 cur next := by
  obtain ⟨⟨w_5_0⟩, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨a1, hct2⟩ := hct1
  have hout := hct2
  clear hct2
  have e_w_5_0 : w_5_0 = 0 + ∑ n6 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n7) * 2) + (Int.ofNat n6)).toNat) := seq_sum (fun n6 ↦ famAt x1 (((Int.ofNat n7) * 2) + (Int.ofNat n6)).toNat) (fun _ _ _ hb ↦ sum_sequential_sequential_generatedRoot_304_5 hb) a1
  clear a1
  subst e_w_5_0
  intro hn
  rw [hout] at hn
  obtain ⟨hn1, f0⟩ := link_and hn
  exact ⟨hn1, f0⟩

/-- One step of `sequential_sequential_generatedRoot_306_5`: it adds a term to the state. -/
theorem sum_sequential_sequential_generatedRoot_306_5 {params : W_Requirement_0.Params} {j0 n8 : Nat} {cur next : Int} {x1 : Fin 14400 → Int} {x2 : Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_306_5 params j0 n8 (cur, (x1, (x2, ()))) next) :
    next = cur + (famAt x1 ((x2 * 2) + (Int.ofNat n8)).toNat) := by
  obtain ⟨w_7_0, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨-, hct2⟩ := hct1
  obtain ⟨a2, hct3⟩ := hct2
  have hout := hct3
  clear hct3
  have e_w_7_0 : w_7_0 = famAt x1 ((x2 * 2) + (Int.ofNat n8)).toNat := (GenReq.famGet_famAt a2).2.2
  clear a2
  subst e_w_7_0
  exact hout

/-- One step of `sequential_generatedRoot_306`: a true state stays true only if its conditions hold. -/
theorem step_sequential_generatedRoot_306 {params : W_Requirement_0.Params} {n9 : Nat} {cur next : Bool} {x1 : Fin 14400 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_306 params n9 (cur, (x1, ())) next) :
    ReqStep.Spec (fun n9 ↦ (0 + ∑ n8 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n9) * 2) + (Int.ofNat n8)).toNat)) = 1) n9 cur next := by
  obtain ⟨⟨w_5_0⟩, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨a1, hct2⟩ := hct1
  have hout := hct2
  clear hct2
  have e_w_5_0 : w_5_0 = 0 + ∑ n8 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n9) * 2) + (Int.ofNat n8)).toNat) := seq_sum (fun n8 ↦ famAt x1 (((Int.ofNat n9) * 2) + (Int.ofNat n8)).toNat) (fun _ _ _ hb ↦ sum_sequential_sequential_generatedRoot_306_5 hb) a1
  clear a1
  subst e_w_5_0
  intro hn
  rw [hout] at hn
  obtain ⟨hn1, f0⟩ := link_and hn
  exact ⟨hn1, f0⟩

/-- One step of `sequential_sequential_generatedRoot_308_5`: it adds a term to the state. -/
theorem sum_sequential_sequential_generatedRoot_308_5 {params : W_Requirement_0.Params} {j0 n10 : Nat} {cur next : Int} {x1 : Fin 14400 → Int} {x2 : Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_308_5 params j0 n10 (cur, (x1, (x2, ()))) next) :
    next = cur + (famAt x1 ((x2 * 2) + (Int.ofNat n10)).toNat) := by
  obtain ⟨w_7_0, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨-, hct2⟩ := hct1
  obtain ⟨a2, hct3⟩ := hct2
  have hout := hct3
  clear hct3
  have e_w_7_0 : w_7_0 = famAt x1 ((x2 * 2) + (Int.ofNat n10)).toNat := (GenReq.famGet_famAt a2).2.2
  clear a2
  subst e_w_7_0
  exact hout

/-- One step of `sequential_generatedRoot_308`: a true state stays true only if its conditions hold. -/
theorem step_sequential_generatedRoot_308 {params : W_Requirement_0.Params} {n11 : Nat} {cur next : Bool} {x1 : Fin 14400 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_308 params n11 (cur, (x1, ())) next) :
    ReqStep.Spec (fun n11 ↦ (0 + ∑ n10 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n11) * 2) + (Int.ofNat n10)).toNat)) = 1) n11 cur next := by
  obtain ⟨⟨w_5_0⟩, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨a1, hct2⟩ := hct1
  have hout := hct2
  clear hct2
  have e_w_5_0 : w_5_0 = 0 + ∑ n10 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n11) * 2) + (Int.ofNat n10)).toNat) := seq_sum (fun n10 ↦ famAt x1 (((Int.ofNat n11) * 2) + (Int.ofNat n10)).toNat) (fun _ _ _ hb ↦ sum_sequential_sequential_generatedRoot_308_5 hb) a1
  clear a1
  subst e_w_5_0
  intro hn
  rw [hout] at hn
  obtain ⟨hn1, f0⟩ := link_and hn
  exact ⟨hn1, f0⟩

/-- One step of `sequential_sequential_generatedRoot_310_5`: it adds a term to the state. -/
theorem sum_sequential_sequential_generatedRoot_310_5 {params : W_Requirement_0.Params} {j0 n12 : Nat} {cur next : Int} {x1 : Fin 16 → Int} {x2 : Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_310_5 params j0 n12 (cur, (x1, (x2, ()))) next) :
    next = cur + (famAt x1 ((x2 * 2) + (Int.ofNat n12)).toNat) := by
  obtain ⟨w_7_0, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨-, hct2⟩ := hct1
  obtain ⟨a2, hct3⟩ := hct2
  have hout := hct3
  clear hct3
  have e_w_7_0 : w_7_0 = famAt x1 ((x2 * 2) + (Int.ofNat n12)).toNat := (GenReq.famGet_famAt a2).2.2
  clear a2
  subst e_w_7_0
  exact hout

/-- One step of `sequential_generatedRoot_310`: a true state stays true only if its conditions hold. -/
theorem step_sequential_generatedRoot_310 {params : W_Requirement_0.Params} {n13 : Nat} {cur next : Bool} {x1 : Fin 16 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_310 params n13 (cur, (x1, ())) next) :
    ReqStep.Spec (fun n13 ↦ (0 + ∑ n12 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n13) * 2) + (Int.ofNat n12)).toNat)) = 1) n13 cur next := by
  obtain ⟨⟨w_5_0⟩, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨a1, hct2⟩ := hct1
  have hout := hct2
  clear hct2
  have e_w_5_0 : w_5_0 = 0 + ∑ n12 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n13) * 2) + (Int.ofNat n12)).toNat) := seq_sum (fun n12 ↦ famAt x1 (((Int.ofNat n13) * 2) + (Int.ofNat n12)).toNat) (fun _ _ _ hb ↦ sum_sequential_sequential_generatedRoot_310_5 hb) a1
  clear a1
  subst e_w_5_0
  intro hn
  rw [hout] at hn
  obtain ⟨hn1, f0⟩ := link_and hn
  exact ⟨hn1, f0⟩

/-- One step of `sequential_sequential_generatedRoot_312_5`: it adds a term to the state. -/
theorem sum_sequential_sequential_generatedRoot_312_5 {params : W_Requirement_0.Params} {j0 n14 : Nat} {cur next : Int} {x1 : Fin 16 → Int} {x2 : Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_312_5 params j0 n14 (cur, (x1, (x2, ()))) next) :
    next = cur + (famAt x1 ((x2 * 2) + (Int.ofNat n14)).toNat) := by
  obtain ⟨w_7_0, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨-, hct2⟩ := hct1
  obtain ⟨a2, hct3⟩ := hct2
  have hout := hct3
  clear hct3
  have e_w_7_0 : w_7_0 = famAt x1 ((x2 * 2) + (Int.ofNat n14)).toNat := (GenReq.famGet_famAt a2).2.2
  clear a2
  subst e_w_7_0
  exact hout

/-- One step of `sequential_generatedRoot_312`: a true state stays true only if its conditions hold. -/
theorem step_sequential_generatedRoot_312 {params : W_Requirement_0.Params} {n15 : Nat} {cur next : Bool} {x1 : Fin 16 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_312 params n15 (cur, (x1, ())) next) :
    ReqStep.Spec (fun n15 ↦ (0 + ∑ n14 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n15) * 2) + (Int.ofNat n14)).toNat)) = 1) n15 cur next := by
  obtain ⟨⟨w_5_0⟩, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨a1, hct2⟩ := hct1
  have hout := hct2
  clear hct2
  have e_w_5_0 : w_5_0 = 0 + ∑ n14 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n15) * 2) + (Int.ofNat n14)).toNat) := seq_sum (fun n14 ↦ famAt x1 (((Int.ofNat n15) * 2) + (Int.ofNat n14)).toNat) (fun _ _ _ hb ↦ sum_sequential_sequential_generatedRoot_312_5 hb) a1
  clear a1
  subst e_w_5_0
  intro hn
  rw [hout] at hn
  obtain ⟨hn1, f0⟩ := link_and hn
  exact ⟨hn1, f0⟩

/-- The facts a true requirement result gives. -/
structure Facts (x0 : Fin 2 → Int) (x1 : Fin 2 → Int) (x2 : Fin 5 → Int) (x3 : Fin 4 → Int) (x4 : Fin 4 → Int) (x5 : Fin 1 → Int) (x6 : Fin 1 → Int) (x7 : Fin 2 → Int) (x8 : Fin 2 → Int) (x9 : Fin 2 → Int) (x10 : Fin 1 → Int) (x11 : Fin 1 → Int) (x12 : Fin 72000 → Int) (x13 : Fin 72000 → Int) (x14 : Fin 16 → Int) (x15 : Fin 16 → Int) (x16 : Fin 14400 → Int) (x17 : Fin 14400 → Int) (x18 : Fin 16 → Int) (x19 : Fin 16 → Int) : Prop where
  /-- a condition of the root. -/
  cond_0 : ((if (decide (((famAt x0 0) * (famAt x0 0)) = 1)) then (1 : Int) else 0) + (if (decide ((((famAt x1 0) * 2) + ((1 - (famAt x0 0)) * 8)) = (((famAt x2 0) * 1) * 2))) then (1 : Int) else 0)) = 2
  /-- a condition of the root. -/
  cond_1 : ((if (decide (((famAt x0 1) * (famAt x0 1)) = 1)) then (1 : Int) else 0) + (if (decide ((((famAt x1 1) * 2) + ((1 - (famAt x0 1)) * 8)) = 0)) then (1 : Int) else 0)) = 2
  /-- a condition of the root. -/
  cond_2 : ((if (decide (((famAt x3 0) * (famAt x3 0)) = 1)) then (1 : Int) else 0) + (if (decide ((((famAt x4 0) * 2) + ((1 - (famAt x3 0)) * 8)) = (((famAt x2 1) * 1) * 2))) then (1 : Int) else 0)) = 2
  /-- a condition of the root. -/
  cond_3 : ((if (decide (((famAt x3 1) * (famAt x3 1)) = 1)) then (1 : Int) else 0) + (if (decide ((((famAt x4 1) * 2) + ((1 - (famAt x3 1)) * 8)) = (((famAt x2 2) * 1) * 2))) then (1 : Int) else 0)) = 2
  /-- a condition of the root. -/
  cond_4 : ((if (decide (((famAt x3 2) * (famAt x3 2)) = 1)) then (1 : Int) else 0) + (if (decide ((((famAt x4 2) * 2) + ((1 - (famAt x3 2)) * 8)) = (((famAt x2 3) * 1) * 2))) then (1 : Int) else 0)) = 2
  /-- a condition of the root. -/
  cond_5 : ((if (decide (((famAt x3 3) * (famAt x3 3)) = 1)) then (1 : Int) else 0) + (if (decide ((((famAt x4 3) * 2) + ((1 - (famAt x3 3)) * 8)) = (((famAt x2 4) * 1) * 2))) then (1 : Int) else 0)) = 2
  /-- a condition of the root. -/
  cond_6 : ((if (decide (((famAt x5 0) * (famAt x5 0)) = 1)) then (1 : Int) else 0) + (if (decide ((((famAt x6 0) * 2) + ((1 - (famAt x5 0)) * 8)) = (((famAt x7 0) * 2) * 2))) then (1 : Int) else 0)) = 2
  /-- a condition of the root. -/
  cond_7 : ((if (decide (((famAt x8 0) * (famAt x8 0)) = 1)) then (1 : Int) else 0) + (if (decide ((((famAt x9 0) * 2) + ((1 - (famAt x8 0)) * 8)) = (((((0 + (famAt x2 2)) + (famAt x2 4)) % 16) * 1) * 2))) then (1 : Int) else 0)) = 2
  /-- a condition of the root. -/
  cond_8 : ((if (decide (((famAt x8 1) * (famAt x8 1)) = 1)) then (1 : Int) else 0) + (if (decide ((((famAt x9 1) * 2) + ((1 - (famAt x8 1)) * 8)) = 0)) then (1 : Int) else 0)) = 2
  /-- a condition of the root. -/
  cond_9 : ((if (decide (((famAt x10 0) * (famAt x10 0)) = 1)) then (1 : Int) else 0) + (if (decide ((((famAt x11 0) * 2) + ((1 - (famAt x10 0)) * 8)) = (((famAt x7 1) * 2) * 2))) then (1 : Int) else 0)) = 2
  /-- every iteration of `sequential_generatedRoot_298`. -/
  loop_298 : ∀ n1 < 36000, (0 + ∑ n0 ∈ Finset.range 2, (famAt x12 (((Int.ofNat n1) * 2) + (Int.ofNat n0)).toNat)) = 1
  /-- every iteration of `sequential_generatedRoot_300`. -/
  loop_300 : ∀ n3 < 36000, (0 + ∑ n2 ∈ Finset.range 2, (famAt x13 (((Int.ofNat n3) * 2) + (Int.ofNat n2)).toNat)) = 1
  /-- every iteration of `sequential_generatedRoot_302`. -/
  loop_302 : ∀ n5 < 8, (0 + ∑ n4 ∈ Finset.range 2, (famAt x14 (((Int.ofNat n5) * 2) + (Int.ofNat n4)).toNat)) = 1
  /-- every iteration of `sequential_generatedRoot_304`. -/
  loop_304 : ∀ n7 < 8, (0 + ∑ n6 ∈ Finset.range 2, (famAt x15 (((Int.ofNat n7) * 2) + (Int.ofNat n6)).toNat)) = 1
  /-- every iteration of `sequential_generatedRoot_306`. -/
  loop_306 : ∀ n9 < 7200, (0 + ∑ n8 ∈ Finset.range 2, (famAt x16 (((Int.ofNat n9) * 2) + (Int.ofNat n8)).toNat)) = 1
  /-- every iteration of `sequential_generatedRoot_308`. -/
  loop_308 : ∀ n11 < 7200, (0 + ∑ n10 ∈ Finset.range 2, (famAt x17 (((Int.ofNat n11) * 2) + (Int.ofNat n10)).toNat)) = 1
  /-- every iteration of `sequential_generatedRoot_310`. -/
  loop_310 : ∀ n13 < 8, (0 + ∑ n12 ∈ Finset.range 2, (famAt x18 (((Int.ofNat n13) * 2) + (Int.ofNat n12)).toNat)) = 1
  /-- every iteration of `sequential_generatedRoot_312`. -/
  loop_312 : ∀ n15 < 8, (0 + ∑ n14 ∈ Finset.range 2, (famAt x19 (((Int.ofNat n15) * 2) + (Int.ofNat n14)).toNat)) = 1

/-- A true requirement result gives all its facts. -/
theorem requirement_facts {params : W_Requirement_0.Params} {x0 : Fin 2 → Int} {x1 : Fin 2 → Int} {x2 : Fin 5 → Int} {x3 : Fin 4 → Int} {x4 : Fin 4 → Int} {x5 : Fin 1 → Int} {x6 : Fin 1 → Int} {x7 : Fin 2 → Int} {x8 : Fin 2 → Int} {x9 : Fin 2 → Int} {x10 : Fin 1 → Int} {x11 : Fin 1 → Int} {x12 : Fin 72000 → Int} {x13 : Fin 72000 → Int} {x14 : Fin 16 → Int} {x15 : Fin 16 → Int} {x16 : Fin 14400 → Int} {x17 : Fin 14400 → Int} {x18 : Fin 16 → Int} {x19 : Fin 16 → Int} {out : Bool}
    (h : W_Requirement_0.generatedRoot params (x0, (x1, (x2, (x3, (x4, (x5, (x6, (x7, (x8, (x9, (x10, (x11, (x12, (x13, (x14, (x15, (x16, (x17, (x18, (x19, ())))))))))))))))))))) out) (ho : out = true) : Facts x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 := by
  obtain ⟨⟨w_3_0, w_9_0, w_18_0, w_33_0, w_38_0, w_62_0, w_68_0, w_76_0, w_91_0, w_96_0, w_104_0, w_119_0, w_124_0, w_132_0, w_147_0, w_152_0, w_160_0, w_176_0, w_182_0, w_191_0, w_207_0, w_213_0, w_240_0, w_245_0, w_269_0, w_275_0, w_283_0, w_298_0, w_300_0, w_302_0, w_304_0, w_306_0, w_308_0, w_310_0, w_312_0⟩, hc⟩ := h
  obtain ⟨hc_0, hct1⟩ := hc
  obtain ⟨a0, hc_0t1⟩ := hc_0
  obtain ⟨a1, hc_0t2⟩ := hc_0t1
  obtain ⟨a2, hc_0t3⟩ := hc_0t2
  obtain ⟨a3, hc_0t4⟩ := hc_0t3
  obtain ⟨a4, hc_0t5⟩ := hc_0t4
  obtain ⟨a5, hc_0t6⟩ := hc_0t5
  obtain ⟨a6, hc_0t7⟩ := hc_0t6
  obtain ⟨a7, hc_0t8⟩ := hc_0t7
  obtain ⟨a8, hc_0t9⟩ := hc_0t8
  obtain ⟨a9, hc_0t10⟩ := hc_0t9
  obtain ⟨a10, hc_0t11⟩ := hc_0t10
  obtain ⟨a11, hc_0t12⟩ := hc_0t11
  obtain ⟨a12, hc_0t13⟩ := hc_0t12
  obtain ⟨a13, hc_0t14⟩ := hc_0t13
  obtain ⟨a14, hc_0t15⟩ := hc_0t14
  obtain ⟨a15, hc_0t16⟩ := hc_0t15
  obtain ⟨a16, hc_0t17⟩ := hc_0t16
  obtain ⟨a17, hc_0t18⟩ := hc_0t17
  obtain ⟨a18, hc_0t19⟩ := hc_0t18
  obtain ⟨a19, hc_0t20⟩ := hc_0t19
  obtain ⟨a20, hc_0t21⟩ := hc_0t20
  obtain ⟨a21, hc_0t22⟩ := hc_0t21
  obtain ⟨-, hc_0t23⟩ := hc_0t22
  obtain ⟨a23, hc_0t24⟩ := hc_0t23
  obtain ⟨a24, hc_0t25⟩ := hc_0t24
  obtain ⟨a25, hc_0t26⟩ := hc_0t25
  obtain ⟨a26, hc_0t27⟩ := hc_0t26
  obtain ⟨a27, hc_0t28⟩ := hc_0t27
  obtain ⟨-, hc_0t29⟩ := hc_0t28
  obtain ⟨a29, hc_0t30⟩ := hc_0t29
  obtain ⟨-, hc_0t31⟩ := hc_0t30
  have a31 := hc_0t31
  clear hc_0t31
  have hc_1 := hct1
  clear hct1
  obtain ⟨-, hc_1t1⟩ := hc_1
  obtain ⟨a33, hc_1t2⟩ := hc_1t1
  obtain ⟨-, hc_1t3⟩ := hc_1t2
  obtain ⟨a35, hc_1t4⟩ := hc_1t3
  obtain ⟨-, hc_1t5⟩ := hc_1t4
  obtain ⟨a37, hc_1t6⟩ := hc_1t5
  obtain ⟨-, hc_1t7⟩ := hc_1t6
  obtain ⟨a39, hc_1t8⟩ := hc_1t7
  obtain ⟨-, hc_1t9⟩ := hc_1t8
  obtain ⟨a41, hc_1t10⟩ := hc_1t9
  obtain ⟨-, hc_1t11⟩ := hc_1t10
  obtain ⟨a43, hc_1t12⟩ := hc_1t11
  have hout := hc_1t12
  clear hc_1t12
  have e_w_3_0 : w_3_0 = famAt x0 0 := (GenReq.famGet_famAt a0).2.2
  have e_w_9_0 : w_9_0 = famAt x1 0 := (GenReq.famGet_famAt a1).2.2
  have e_w_18_0 : w_18_0 = famAt x2 0 := (GenReq.famGet_famAt a2).2.2
  have e_w_33_0 : w_33_0 = famAt x0 1 := (GenReq.famGet_famAt a3).2.2
  have e_w_38_0 : w_38_0 = famAt x1 1 := (GenReq.famGet_famAt a4).2.2
  have e_w_62_0 : w_62_0 = famAt x3 0 := (GenReq.famGet_famAt a5).2.2
  have e_w_68_0 : w_68_0 = famAt x4 0 := (GenReq.famGet_famAt a6).2.2
  have e_w_76_0 : w_76_0 = famAt x2 1 := (GenReq.famGet_famAt a7).2.2
  have e_w_91_0 : w_91_0 = famAt x3 1 := (GenReq.famGet_famAt a8).2.2
  have e_w_96_0 : w_96_0 = famAt x4 1 := (GenReq.famGet_famAt a9).2.2
  have e_w_104_0 : w_104_0 = famAt x2 2 := (GenReq.famGet_famAt a10).2.2
  have e_w_119_0 : w_119_0 = famAt x3 2 := (GenReq.famGet_famAt a11).2.2
  have e_w_124_0 : w_124_0 = famAt x4 2 := (GenReq.famGet_famAt a12).2.2
  have e_w_132_0 : w_132_0 = famAt x2 3 := (GenReq.famGet_famAt a13).2.2
  have e_w_147_0 : w_147_0 = famAt x3 3 := (GenReq.famGet_famAt a14).2.2
  have e_w_152_0 : w_152_0 = famAt x4 3 := (GenReq.famGet_famAt a15).2.2
  have e_w_160_0 : w_160_0 = famAt x2 4 := (GenReq.famGet_famAt a16).2.2
  have e_w_176_0 : w_176_0 = famAt x5 0 := (GenReq.famGet_famAt a17).2.2
  have e_w_182_0 : w_182_0 = famAt x6 0 := (GenReq.famGet_famAt a18).2.2
  have e_w_191_0 : w_191_0 = famAt x7 0 := (GenReq.famGet_famAt a19).2.2
  have e_w_207_0 : w_207_0 = famAt x8 0 := (GenReq.famGet_famAt a20).2.2
  have e_w_213_0 : w_213_0 = famAt x9 0 := (GenReq.famGet_famAt a21).2.2
  have e_w_240_0 : w_240_0 = famAt x8 1 := (GenReq.famGet_famAt a23).2.2
  have e_w_245_0 : w_245_0 = famAt x9 1 := (GenReq.famGet_famAt a24).2.2
  have e_w_269_0 : w_269_0 = famAt x10 0 := (GenReq.famGet_famAt a25).2.2
  have e_w_275_0 : w_275_0 = famAt x11 0 := (GenReq.famGet_famAt a26).2.2
  have e_w_283_0 : w_283_0 = famAt x7 1 := (GenReq.famGet_famAt a27).2.2
  have L_w_298_0 := ReqAll.sem a29 (fun _ _ _ hb ↦ step_sequential_generatedRoot_298 hb)
  have L_w_300_0 := ReqAll.sem a31 (fun _ _ _ hb ↦ step_sequential_generatedRoot_300 hb)
  have L_w_302_0 := ReqAll.sem a33 (fun _ _ _ hb ↦ step_sequential_generatedRoot_302 hb)
  have L_w_304_0 := ReqAll.sem a35 (fun _ _ _ hb ↦ step_sequential_generatedRoot_304 hb)
  have L_w_306_0 := ReqAll.sem a37 (fun _ _ _ hb ↦ step_sequential_generatedRoot_306 hb)
  have L_w_308_0 := ReqAll.sem a39 (fun _ _ _ hb ↦ step_sequential_generatedRoot_308 hb)
  have L_w_310_0 := ReqAll.sem a41 (fun _ _ _ hb ↦ step_sequential_generatedRoot_310 hb)
  have L_w_312_0 := ReqAll.sem a43 (fun _ _ _ hb ↦ step_sequential_generatedRoot_312 hb)
  have hn := ho
  rw [hout] at hn
  obtain ⟨hn1, f17⟩ := link_loop L_w_312_0 hn
  obtain ⟨hn2, f16⟩ := link_loop L_w_310_0 hn1
  obtain ⟨hn3, f15⟩ := link_loop L_w_308_0 hn2
  obtain ⟨hn4, f14⟩ := link_loop L_w_306_0 hn3
  obtain ⟨hn5, f13⟩ := link_loop L_w_304_0 hn4
  obtain ⟨hn6, f12⟩ := link_loop L_w_302_0 hn5
  obtain ⟨hn7, f11⟩ := link_loop L_w_300_0 hn6
  obtain ⟨hn8, f10⟩ := link_loop L_w_298_0 hn7
  obtain ⟨hn9, f9⟩ := link_and hn8
  obtain ⟨hn10, f8⟩ := link_and hn9
  obtain ⟨hn11, f7⟩ := link_and hn10
  obtain ⟨hn12, f6⟩ := link_and hn11
  obtain ⟨hn13, f5⟩ := link_and hn12
  obtain ⟨hn14, f4⟩ := link_and hn13
  obtain ⟨hn15, f3⟩ := link_and hn14
  obtain ⟨hn16, f2⟩ := link_and hn15
  obtain ⟨hn17, f1⟩ := link_and hn16
  obtain ⟨hn18, f0⟩ := link_and hn17
  try simp only [e_w_3_0, e_w_9_0, e_w_18_0, e_w_33_0, e_w_38_0, e_w_62_0, e_w_68_0, e_w_76_0, e_w_91_0, e_w_96_0, e_w_104_0, e_w_119_0, e_w_124_0, e_w_132_0, e_w_147_0, e_w_152_0, e_w_160_0, e_w_176_0, e_w_182_0, e_w_191_0, e_w_207_0, e_w_213_0, e_w_240_0, e_w_245_0, e_w_269_0, e_w_275_0, e_w_283_0] at f0
  try simp only [e_w_3_0, e_w_9_0, e_w_18_0, e_w_33_0, e_w_38_0, e_w_62_0, e_w_68_0, e_w_76_0, e_w_91_0, e_w_96_0, e_w_104_0, e_w_119_0, e_w_124_0, e_w_132_0, e_w_147_0, e_w_152_0, e_w_160_0, e_w_176_0, e_w_182_0, e_w_191_0, e_w_207_0, e_w_213_0, e_w_240_0, e_w_245_0, e_w_269_0, e_w_275_0, e_w_283_0] at f1
  try simp only [e_w_3_0, e_w_9_0, e_w_18_0, e_w_33_0, e_w_38_0, e_w_62_0, e_w_68_0, e_w_76_0, e_w_91_0, e_w_96_0, e_w_104_0, e_w_119_0, e_w_124_0, e_w_132_0, e_w_147_0, e_w_152_0, e_w_160_0, e_w_176_0, e_w_182_0, e_w_191_0, e_w_207_0, e_w_213_0, e_w_240_0, e_w_245_0, e_w_269_0, e_w_275_0, e_w_283_0] at f2
  try simp only [e_w_3_0, e_w_9_0, e_w_18_0, e_w_33_0, e_w_38_0, e_w_62_0, e_w_68_0, e_w_76_0, e_w_91_0, e_w_96_0, e_w_104_0, e_w_119_0, e_w_124_0, e_w_132_0, e_w_147_0, e_w_152_0, e_w_160_0, e_w_176_0, e_w_182_0, e_w_191_0, e_w_207_0, e_w_213_0, e_w_240_0, e_w_245_0, e_w_269_0, e_w_275_0, e_w_283_0] at f3
  try simp only [e_w_3_0, e_w_9_0, e_w_18_0, e_w_33_0, e_w_38_0, e_w_62_0, e_w_68_0, e_w_76_0, e_w_91_0, e_w_96_0, e_w_104_0, e_w_119_0, e_w_124_0, e_w_132_0, e_w_147_0, e_w_152_0, e_w_160_0, e_w_176_0, e_w_182_0, e_w_191_0, e_w_207_0, e_w_213_0, e_w_240_0, e_w_245_0, e_w_269_0, e_w_275_0, e_w_283_0] at f4
  try simp only [e_w_3_0, e_w_9_0, e_w_18_0, e_w_33_0, e_w_38_0, e_w_62_0, e_w_68_0, e_w_76_0, e_w_91_0, e_w_96_0, e_w_104_0, e_w_119_0, e_w_124_0, e_w_132_0, e_w_147_0, e_w_152_0, e_w_160_0, e_w_176_0, e_w_182_0, e_w_191_0, e_w_207_0, e_w_213_0, e_w_240_0, e_w_245_0, e_w_269_0, e_w_275_0, e_w_283_0] at f5
  try simp only [e_w_3_0, e_w_9_0, e_w_18_0, e_w_33_0, e_w_38_0, e_w_62_0, e_w_68_0, e_w_76_0, e_w_91_0, e_w_96_0, e_w_104_0, e_w_119_0, e_w_124_0, e_w_132_0, e_w_147_0, e_w_152_0, e_w_160_0, e_w_176_0, e_w_182_0, e_w_191_0, e_w_207_0, e_w_213_0, e_w_240_0, e_w_245_0, e_w_269_0, e_w_275_0, e_w_283_0] at f6
  try simp only [e_w_3_0, e_w_9_0, e_w_18_0, e_w_33_0, e_w_38_0, e_w_62_0, e_w_68_0, e_w_76_0, e_w_91_0, e_w_96_0, e_w_104_0, e_w_119_0, e_w_124_0, e_w_132_0, e_w_147_0, e_w_152_0, e_w_160_0, e_w_176_0, e_w_182_0, e_w_191_0, e_w_207_0, e_w_213_0, e_w_240_0, e_w_245_0, e_w_269_0, e_w_275_0, e_w_283_0] at f7
  try simp only [e_w_3_0, e_w_9_0, e_w_18_0, e_w_33_0, e_w_38_0, e_w_62_0, e_w_68_0, e_w_76_0, e_w_91_0, e_w_96_0, e_w_104_0, e_w_119_0, e_w_124_0, e_w_132_0, e_w_147_0, e_w_152_0, e_w_160_0, e_w_176_0, e_w_182_0, e_w_191_0, e_w_207_0, e_w_213_0, e_w_240_0, e_w_245_0, e_w_269_0, e_w_275_0, e_w_283_0] at f8
  try simp only [e_w_3_0, e_w_9_0, e_w_18_0, e_w_33_0, e_w_38_0, e_w_62_0, e_w_68_0, e_w_76_0, e_w_91_0, e_w_96_0, e_w_104_0, e_w_119_0, e_w_124_0, e_w_132_0, e_w_147_0, e_w_152_0, e_w_160_0, e_w_176_0, e_w_182_0, e_w_191_0, e_w_207_0, e_w_213_0, e_w_240_0, e_w_245_0, e_w_269_0, e_w_275_0, e_w_283_0] at f9
  try simp only [e_w_3_0, e_w_9_0, e_w_18_0, e_w_33_0, e_w_38_0, e_w_62_0, e_w_68_0, e_w_76_0, e_w_91_0, e_w_96_0, e_w_104_0, e_w_119_0, e_w_124_0, e_w_132_0, e_w_147_0, e_w_152_0, e_w_160_0, e_w_176_0, e_w_182_0, e_w_191_0, e_w_207_0, e_w_213_0, e_w_240_0, e_w_245_0, e_w_269_0, e_w_275_0, e_w_283_0] at f10
  try simp only [e_w_3_0, e_w_9_0, e_w_18_0, e_w_33_0, e_w_38_0, e_w_62_0, e_w_68_0, e_w_76_0, e_w_91_0, e_w_96_0, e_w_104_0, e_w_119_0, e_w_124_0, e_w_132_0, e_w_147_0, e_w_152_0, e_w_160_0, e_w_176_0, e_w_182_0, e_w_191_0, e_w_207_0, e_w_213_0, e_w_240_0, e_w_245_0, e_w_269_0, e_w_275_0, e_w_283_0] at f11
  try simp only [e_w_3_0, e_w_9_0, e_w_18_0, e_w_33_0, e_w_38_0, e_w_62_0, e_w_68_0, e_w_76_0, e_w_91_0, e_w_96_0, e_w_104_0, e_w_119_0, e_w_124_0, e_w_132_0, e_w_147_0, e_w_152_0, e_w_160_0, e_w_176_0, e_w_182_0, e_w_191_0, e_w_207_0, e_w_213_0, e_w_240_0, e_w_245_0, e_w_269_0, e_w_275_0, e_w_283_0] at f12
  try simp only [e_w_3_0, e_w_9_0, e_w_18_0, e_w_33_0, e_w_38_0, e_w_62_0, e_w_68_0, e_w_76_0, e_w_91_0, e_w_96_0, e_w_104_0, e_w_119_0, e_w_124_0, e_w_132_0, e_w_147_0, e_w_152_0, e_w_160_0, e_w_176_0, e_w_182_0, e_w_191_0, e_w_207_0, e_w_213_0, e_w_240_0, e_w_245_0, e_w_269_0, e_w_275_0, e_w_283_0] at f13
  try simp only [e_w_3_0, e_w_9_0, e_w_18_0, e_w_33_0, e_w_38_0, e_w_62_0, e_w_68_0, e_w_76_0, e_w_91_0, e_w_96_0, e_w_104_0, e_w_119_0, e_w_124_0, e_w_132_0, e_w_147_0, e_w_152_0, e_w_160_0, e_w_176_0, e_w_182_0, e_w_191_0, e_w_207_0, e_w_213_0, e_w_240_0, e_w_245_0, e_w_269_0, e_w_275_0, e_w_283_0] at f14
  try simp only [e_w_3_0, e_w_9_0, e_w_18_0, e_w_33_0, e_w_38_0, e_w_62_0, e_w_68_0, e_w_76_0, e_w_91_0, e_w_96_0, e_w_104_0, e_w_119_0, e_w_124_0, e_w_132_0, e_w_147_0, e_w_152_0, e_w_160_0, e_w_176_0, e_w_182_0, e_w_191_0, e_w_207_0, e_w_213_0, e_w_240_0, e_w_245_0, e_w_269_0, e_w_275_0, e_w_283_0] at f15
  try simp only [e_w_3_0, e_w_9_0, e_w_18_0, e_w_33_0, e_w_38_0, e_w_62_0, e_w_68_0, e_w_76_0, e_w_91_0, e_w_96_0, e_w_104_0, e_w_119_0, e_w_124_0, e_w_132_0, e_w_147_0, e_w_152_0, e_w_160_0, e_w_176_0, e_w_182_0, e_w_191_0, e_w_207_0, e_w_213_0, e_w_240_0, e_w_245_0, e_w_269_0, e_w_275_0, e_w_283_0] at f16
  try simp only [e_w_3_0, e_w_9_0, e_w_18_0, e_w_33_0, e_w_38_0, e_w_62_0, e_w_68_0, e_w_76_0, e_w_91_0, e_w_96_0, e_w_104_0, e_w_119_0, e_w_124_0, e_w_132_0, e_w_147_0, e_w_152_0, e_w_160_0, e_w_176_0, e_w_182_0, e_w_191_0, e_w_207_0, e_w_213_0, e_w_240_0, e_w_245_0, e_w_269_0, e_w_275_0, e_w_283_0] at f17
  exact ⟨f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17⟩

end Dio2Requirement
