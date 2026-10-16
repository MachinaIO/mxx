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

/-- One step of `sequential_generatedRoot_3`: a true state stays true only if its conditions hold. -/
theorem step_sequential_generatedRoot_3 {params : W_Requirement_0.Params} {n0 : Nat} {cur next : Bool} {x1 : Fin 5 → Int} {x2 : Fin 5 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_3 params n0 (cur, (x1, (x2, ()))) next) :
    ReqStep.Spec (fun n0 ↦ (famAt x1 n0) = (famAt x2 n0)) n0 cur next := by
  obtain ⟨w_4_0, w_6_0, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨-, hct2⟩ := hct1
  obtain ⟨a2, hct3⟩ := hct2
  obtain ⟨-, hct4⟩ := hct3
  obtain ⟨-, hct5⟩ := hct4
  obtain ⟨a5, hct6⟩ := hct5
  have hout := hct6
  clear hct6
  have e_w_4_0 : w_4_0 = famAt x1 n0 := (GenReq.famGet_ofNat a2).2
  have e_w_6_0 : w_6_0 = famAt x2 n0 := (GenReq.famGet_ofNat a5).2
  clear a2 a5
  subst e_w_4_0 e_w_6_0
  intro hn
  rw [hout] at hn
  obtain ⟨hn1, f0⟩ := link_and hn
  exact ⟨hn1, f0⟩

/-- One step of `sequential_generatedRoot_6`: a true state stays true only if its conditions hold. -/
theorem step_sequential_generatedRoot_6 {params : W_Requirement_0.Params} {n1 : Nat} {cur next : Bool} {x1 : Fin 2 → Int} {x2 : Fin 2 → Int} {x3 : Fin 5 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_6 params n1 (cur, (x1, (x2, (x3, ())))) next) :
    ReqStep.Spec (fun n1 ↦ ((if (decide (((famAt x1 n1) * (famAt x1 n1)) = 1)) then (1 : Int) else 0) + (if (decide ((((famAt x2 n1) * 2) + ((1 - (famAt x1 n1)) * 8)) = (((MxxIR.selectInt (if (decide ((Int.ofNat n1) = 1)) then (1 : Int) else 0) [(famAt x3 (((Int.ofNat n1) % 1) * 5).toNat), 0]) * 1) * 2))) then (1 : Int) else 0)) = 2) n1 cur next := by
  obtain ⟨w_4_0, w_10_0, w_26_0, w_30_0, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨-, hct2⟩ := hct1
  obtain ⟨a2, hct3⟩ := hct2
  obtain ⟨-, hct4⟩ := hct3
  obtain ⟨-, hct5⟩ := hct4
  obtain ⟨a5, hct6⟩ := hct5
  obtain ⟨-, hct7⟩ := hct6
  obtain ⟨-, hct8⟩ := hct7
  obtain ⟨-, hct9⟩ := hct8
  obtain ⟨a9, hct10⟩ := hct9
  obtain ⟨-, hct11⟩ := hct10
  obtain ⟨-, hct12⟩ := hct11
  obtain ⟨-, hct13⟩ := hct12
  obtain ⟨a13, hct14⟩ := hct13
  have hout := hct14
  clear hct14
  have e_w_4_0 : w_4_0 = famAt x1 n1 := (GenReq.famGet_ofNat a2).2
  have e_w_10_0 : w_10_0 = famAt x2 n1 := (GenReq.famGet_ofNat a5).2
  have e_w_26_0 : w_26_0 = famAt x3 (((Int.ofNat n1) % 1) * 5).toNat := (GenReq.famGet_famAt a9).2.2
  have e_w_30_0 : w_30_0 = MxxIR.selectInt (if (decide ((Int.ofNat n1) = 1)) then 1 else 0) [w_26_0, 0] := select_int a13
  clear a13 a2 a5 a9
  subst e_w_4_0 e_w_10_0 e_w_26_0 e_w_30_0
  intro hn
  rw [hout] at hn
  obtain ⟨hn1, f0⟩ := link_and hn
  exact ⟨hn1, f0⟩

/-- One step of `sequential_parallel_generatedRoot_41_11`: it adds a term to the state. -/
theorem sum_sequential_parallel_generatedRoot_41_11 {params : W_Requirement_0.Params} {j0 n2 : Nat} {cur next : Int} {x1 : Int} {x2 : Fin 5 → Int} {x3 : Int} {x4 : Int}
    (h : W_Requirement_0.sequential_parallel_generatedRoot_41_11 params j0 n2 (cur, (x1, (x2, (x3, (x4, ()))))) next) :
    next = cur + (MxxIR.selectInt (if (decide ((Int.ofNat n2) < x1)) then 1 else 0) [0, (famAt x2 (((x3 * 5) + ((0 + ((if (decide ((x4 / (2 ^ (Int.toNat ((2 * (Int.ofNat n2)))))) % 2 = 1)) then 1 else 0) * 1)) + ((if (decide ((x4 / (2 ^ (Int.toNat ((1 + (2 * (Int.ofNat n2))))))) % 2 = 1)) then 1 else 0) * 2))) + 1).toNat)]) := by
  obtain ⟨w_27_0, w_28_0, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨-, hct2⟩ := hct1
  obtain ⟨-, hct3⟩ := hct2
  obtain ⟨-, hct4⟩ := hct3
  obtain ⟨a4, hct5⟩ := hct4
  obtain ⟨-, hct6⟩ := hct5
  obtain ⟨-, hct7⟩ := hct6
  obtain ⟨-, hct8⟩ := hct7
  obtain ⟨a8, hct9⟩ := hct8
  have hout := hct9
  clear hct9
  have e_w_27_0 : w_27_0 = famAt x2 (((x3 * 5) + ((0 + ((if (decide ((x4 / (2 ^ (Int.toNat ((2 * (Int.ofNat n2)))))) % 2 = 1)) then 1 else 0) * 1)) + ((if (decide ((x4 / (2 ^ (Int.toNat ((1 + (2 * (Int.ofNat n2))))))) % 2 = 1)) then 1 else 0) * 2))) + 1).toNat := (GenReq.famGet_famAt a4).2.2
  have e_w_28_0 : w_28_0 = MxxIR.selectInt (if (decide ((Int.ofNat n2) < x1)) then 1 else 0) [0, w_27_0] := select_int a8
  clear a4 a8
  subst e_w_27_0 e_w_28_0
  exact hout

/-- The integer output of `parallel_generatedRoot_41`, without unrolling its sums. -/
theorem value_parallel_generatedRoot_41 {params : W_Requirement_0.Params} {n3 : Nat} {next : Int} {x0 : Fin 5 → Int} {x1 : Fin 5 → Int} {x2 : Fin 5 → Int}
    (h : W_Requirement_0.parallel_generatedRoot_41 params n3 (x0, (x1, (x2, ()))) next) : next = ((0 + ∑ n2 ∈ Finset.range 4, (MxxIR.selectInt (if (decide ((Int.ofNat n2) < (famAt x0 ((Int.ofNat n3) % 5).toNat))) then 1 else 0) [0, (famAt x1 (((((Int.ofNat n3) / 5) * 5) + ((0 + ((if (decide (((famAt x2 ((Int.ofNat n3) % 5).toNat) / (2 ^ (Int.toNat ((2 * (Int.ofNat n2)))))) % 2 = 1)) then 1 else 0) * 1)) + ((if (decide (((famAt x2 ((Int.ofNat n3) % 5).toNat) / (2 ^ (Int.toNat ((1 + (2 * (Int.ofNat n2))))))) % 2 = 1)) then 1 else 0) * 2))) + 1).toNat)])) % 16) := by
  obtain ⟨⟨w_5_0, w_10_0, w_11_0⟩, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨-, hct2⟩ := hct1
  obtain ⟨-, hct3⟩ := hct2
  obtain ⟨a3, hct4⟩ := hct3
  obtain ⟨-, hct5⟩ := hct4
  obtain ⟨-, hct6⟩ := hct5
  obtain ⟨-, hct7⟩ := hct6
  obtain ⟨a7, hct8⟩ := hct7
  obtain ⟨-, hct9⟩ := hct8
  obtain ⟨a9, hct10⟩ := hct9
  obtain ⟨-, hct11⟩ := hct10
  have hout := hct11
  clear hct11
  have e_w_5_0 : w_5_0 = famAt x0 ((Int.ofNat n3) % 5).toNat := (GenReq.famGet_famAt a3).2.2
  have e_w_10_0 : w_10_0 = famAt x2 ((Int.ofNat n3) % 5).toNat := (GenReq.famGet_famAt a7).2.2
  have e_w_11_0 : w_11_0 = 0 + ∑ n2 ∈ Finset.range 4, (MxxIR.selectInt (if (decide ((Int.ofNat n2) < w_5_0)) then 1 else 0) [0, (famAt x1 (((((Int.ofNat n3) / 5) * 5) + ((0 + ((if (decide ((w_10_0 / (2 ^ (Int.toNat ((2 * (Int.ofNat n2)))))) % 2 = 1)) then 1 else 0) * 1)) + ((if (decide ((w_10_0 / (2 ^ (Int.toNat ((1 + (2 * (Int.ofNat n2))))))) % 2 = 1)) then 1 else 0) * 2))) + 1).toNat)]) := seq_sum (fun n2 ↦ MxxIR.selectInt (if (decide ((Int.ofNat n2) < w_5_0)) then 1 else 0) [0, (famAt x1 (((((Int.ofNat n3) / 5) * 5) + ((0 + ((if (decide ((w_10_0 / (2 ^ (Int.toNat ((2 * (Int.ofNat n2)))))) % 2 = 1)) then 1 else 0) * 1)) + ((if (decide ((w_10_0 / (2 ^ (Int.toNat ((1 + (2 * (Int.ofNat n2))))))) % 2 = 1)) then 1 else 0) * 2))) + 1).toNat)]) (fun _ _ _ hb ↦ sum_sequential_parallel_generatedRoot_41_11 hb) a9
  clear a3 a7 a9
  subst e_w_5_0 e_w_10_0 e_w_11_0
  exact hout

/-- One step of `sequential_generatedRoot_42`: a true state stays true only if its conditions hold. -/
theorem step_sequential_generatedRoot_42 {params : W_Requirement_0.Params} {n4 : Nat} {cur next : Bool} {x1 : Fin 2 → Int} {x2 : Fin 2 → Int} {x3 : Fin 5 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_42 params n4 (cur, (x1, (x2, (x3, ())))) next) :
    ReqStep.Spec (fun n4 ↦ ((if (decide (((famAt x1 n4) * (famAt x1 n4)) = 1)) then (1 : Int) else 0) + (if (decide ((((famAt x2 n4) * 2) + ((1 - (famAt x1 n4)) * 8)) = (((MxxIR.selectInt (if (decide ((Int.ofNat n4) = 1)) then (1 : Int) else 0) [(famAt x3 (((Int.ofNat n4) % 1) * 5).toNat), 0]) * 1) * 2))) then (1 : Int) else 0)) = 2) n4 cur next := by
  obtain ⟨w_4_0, w_10_0, w_26_0, w_30_0, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨-, hct2⟩ := hct1
  obtain ⟨a2, hct3⟩ := hct2
  obtain ⟨-, hct4⟩ := hct3
  obtain ⟨-, hct5⟩ := hct4
  obtain ⟨a5, hct6⟩ := hct5
  obtain ⟨-, hct7⟩ := hct6
  obtain ⟨-, hct8⟩ := hct7
  obtain ⟨-, hct9⟩ := hct8
  obtain ⟨a9, hct10⟩ := hct9
  obtain ⟨-, hct11⟩ := hct10
  obtain ⟨-, hct12⟩ := hct11
  obtain ⟨-, hct13⟩ := hct12
  obtain ⟨a13, hct14⟩ := hct13
  have hout := hct14
  clear hct14
  have e_w_4_0 : w_4_0 = famAt x1 n4 := (GenReq.famGet_ofNat a2).2
  have e_w_10_0 : w_10_0 = famAt x2 n4 := (GenReq.famGet_ofNat a5).2
  have e_w_26_0 : w_26_0 = famAt x3 (((Int.ofNat n4) % 1) * 5).toNat := (GenReq.famGet_famAt a9).2.2
  have e_w_30_0 : w_30_0 = MxxIR.selectInt (if (decide ((Int.ofNat n4) = 1)) then 1 else 0) [w_26_0, 0] := select_int a13
  clear a13 a2 a5 a9
  subst e_w_4_0 e_w_10_0 e_w_26_0 e_w_30_0
  intro hn
  rw [hout] at hn
  obtain ⟨hn1, f0⟩ := link_and hn
  exact ⟨hn1, f0⟩

/-- One step of `sequential_sequential_generatedRoot_74_5`: it adds a term to the state. -/
theorem sum_sequential_sequential_generatedRoot_74_5 {params : W_Requirement_0.Params} {j0 n5 : Nat} {cur next : Int} {x1 : Fin 72000 → Int} {x2 : Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_74_5 params j0 n5 (cur, (x1, (x2, ()))) next) :
    next = cur + (famAt x1 ((x2 * 2) + (Int.ofNat n5)).toNat) := by
  obtain ⟨w_7_0, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨-, hct2⟩ := hct1
  obtain ⟨a2, hct3⟩ := hct2
  have hout := hct3
  clear hct3
  have e_w_7_0 : w_7_0 = famAt x1 ((x2 * 2) + (Int.ofNat n5)).toNat := (GenReq.famGet_famAt a2).2.2
  clear a2
  subst e_w_7_0
  exact hout

/-- One step of `sequential_generatedRoot_74`: a true state stays true only if its conditions hold. -/
theorem step_sequential_generatedRoot_74 {params : W_Requirement_0.Params} {n6 : Nat} {cur next : Bool} {x1 : Fin 72000 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_74 params n6 (cur, (x1, ())) next) :
    ReqStep.Spec (fun n6 ↦ (0 + ∑ n5 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n6) * 2) + (Int.ofNat n5)).toNat)) = 1) n6 cur next := by
  obtain ⟨⟨w_5_0⟩, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨a1, hct2⟩ := hct1
  have hout := hct2
  clear hct2
  have e_w_5_0 : w_5_0 = 0 + ∑ n5 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n6) * 2) + (Int.ofNat n5)).toNat) := seq_sum (fun n5 ↦ famAt x1 (((Int.ofNat n6) * 2) + (Int.ofNat n5)).toNat) (fun _ _ _ hb ↦ sum_sequential_sequential_generatedRoot_74_5 hb) a1
  clear a1
  subst e_w_5_0
  intro hn
  rw [hout] at hn
  obtain ⟨hn1, f0⟩ := link_and hn
  exact ⟨hn1, f0⟩

/-- One step of `sequential_sequential_generatedRoot_76_5`: it adds a term to the state. -/
theorem sum_sequential_sequential_generatedRoot_76_5 {params : W_Requirement_0.Params} {j0 n7 : Nat} {cur next : Int} {x1 : Fin 72000 → Int} {x2 : Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_76_5 params j0 n7 (cur, (x1, (x2, ()))) next) :
    next = cur + (famAt x1 ((x2 * 2) + (Int.ofNat n7)).toNat) := by
  obtain ⟨w_7_0, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨-, hct2⟩ := hct1
  obtain ⟨a2, hct3⟩ := hct2
  have hout := hct3
  clear hct3
  have e_w_7_0 : w_7_0 = famAt x1 ((x2 * 2) + (Int.ofNat n7)).toNat := (GenReq.famGet_famAt a2).2.2
  clear a2
  subst e_w_7_0
  exact hout

/-- One step of `sequential_generatedRoot_76`: a true state stays true only if its conditions hold. -/
theorem step_sequential_generatedRoot_76 {params : W_Requirement_0.Params} {n8 : Nat} {cur next : Bool} {x1 : Fin 72000 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_76 params n8 (cur, (x1, ())) next) :
    ReqStep.Spec (fun n8 ↦ (0 + ∑ n7 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n8) * 2) + (Int.ofNat n7)).toNat)) = 1) n8 cur next := by
  obtain ⟨⟨w_5_0⟩, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨a1, hct2⟩ := hct1
  have hout := hct2
  clear hct2
  have e_w_5_0 : w_5_0 = 0 + ∑ n7 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n8) * 2) + (Int.ofNat n7)).toNat) := seq_sum (fun n7 ↦ famAt x1 (((Int.ofNat n8) * 2) + (Int.ofNat n7)).toNat) (fun _ _ _ hb ↦ sum_sequential_sequential_generatedRoot_76_5 hb) a1
  clear a1
  subst e_w_5_0
  intro hn
  rw [hout] at hn
  obtain ⟨hn1, f0⟩ := link_and hn
  exact ⟨hn1, f0⟩

/-- One step of `sequential_sequential_generatedRoot_78_5`: it adds a term to the state. -/
theorem sum_sequential_sequential_generatedRoot_78_5 {params : W_Requirement_0.Params} {j0 n9 : Nat} {cur next : Int} {x1 : Fin 16 → Int} {x2 : Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_78_5 params j0 n9 (cur, (x1, (x2, ()))) next) :
    next = cur + (famAt x1 ((x2 * 2) + (Int.ofNat n9)).toNat) := by
  obtain ⟨w_7_0, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨-, hct2⟩ := hct1
  obtain ⟨a2, hct3⟩ := hct2
  have hout := hct3
  clear hct3
  have e_w_7_0 : w_7_0 = famAt x1 ((x2 * 2) + (Int.ofNat n9)).toNat := (GenReq.famGet_famAt a2).2.2
  clear a2
  subst e_w_7_0
  exact hout

/-- One step of `sequential_generatedRoot_78`: a true state stays true only if its conditions hold. -/
theorem step_sequential_generatedRoot_78 {params : W_Requirement_0.Params} {n10 : Nat} {cur next : Bool} {x1 : Fin 16 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_78 params n10 (cur, (x1, ())) next) :
    ReqStep.Spec (fun n10 ↦ (0 + ∑ n9 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n10) * 2) + (Int.ofNat n9)).toNat)) = 1) n10 cur next := by
  obtain ⟨⟨w_5_0⟩, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨a1, hct2⟩ := hct1
  have hout := hct2
  clear hct2
  have e_w_5_0 : w_5_0 = 0 + ∑ n9 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n10) * 2) + (Int.ofNat n9)).toNat) := seq_sum (fun n9 ↦ famAt x1 (((Int.ofNat n10) * 2) + (Int.ofNat n9)).toNat) (fun _ _ _ hb ↦ sum_sequential_sequential_generatedRoot_78_5 hb) a1
  clear a1
  subst e_w_5_0
  intro hn
  rw [hout] at hn
  obtain ⟨hn1, f0⟩ := link_and hn
  exact ⟨hn1, f0⟩

/-- One step of `sequential_sequential_generatedRoot_80_5`: it adds a term to the state. -/
theorem sum_sequential_sequential_generatedRoot_80_5 {params : W_Requirement_0.Params} {j0 n11 : Nat} {cur next : Int} {x1 : Fin 16 → Int} {x2 : Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_80_5 params j0 n11 (cur, (x1, (x2, ()))) next) :
    next = cur + (famAt x1 ((x2 * 2) + (Int.ofNat n11)).toNat) := by
  obtain ⟨w_7_0, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨-, hct2⟩ := hct1
  obtain ⟨a2, hct3⟩ := hct2
  have hout := hct3
  clear hct3
  have e_w_7_0 : w_7_0 = famAt x1 ((x2 * 2) + (Int.ofNat n11)).toNat := (GenReq.famGet_famAt a2).2.2
  clear a2
  subst e_w_7_0
  exact hout

/-- One step of `sequential_generatedRoot_80`: a true state stays true only if its conditions hold. -/
theorem step_sequential_generatedRoot_80 {params : W_Requirement_0.Params} {n12 : Nat} {cur next : Bool} {x1 : Fin 16 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_80 params n12 (cur, (x1, ())) next) :
    ReqStep.Spec (fun n12 ↦ (0 + ∑ n11 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n12) * 2) + (Int.ofNat n11)).toNat)) = 1) n12 cur next := by
  obtain ⟨⟨w_5_0⟩, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨a1, hct2⟩ := hct1
  have hout := hct2
  clear hct2
  have e_w_5_0 : w_5_0 = 0 + ∑ n11 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n12) * 2) + (Int.ofNat n11)).toNat) := seq_sum (fun n11 ↦ famAt x1 (((Int.ofNat n12) * 2) + (Int.ofNat n11)).toNat) (fun _ _ _ hb ↦ sum_sequential_sequential_generatedRoot_80_5 hb) a1
  clear a1
  subst e_w_5_0
  intro hn
  rw [hout] at hn
  obtain ⟨hn1, f0⟩ := link_and hn
  exact ⟨hn1, f0⟩

/-- One step of `sequential_sequential_generatedRoot_82_5`: it adds a term to the state. -/
theorem sum_sequential_sequential_generatedRoot_82_5 {params : W_Requirement_0.Params} {j0 n13 : Nat} {cur next : Int} {x1 : Fin 14400 → Int} {x2 : Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_82_5 params j0 n13 (cur, (x1, (x2, ()))) next) :
    next = cur + (famAt x1 ((x2 * 2) + (Int.ofNat n13)).toNat) := by
  obtain ⟨w_7_0, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨-, hct2⟩ := hct1
  obtain ⟨a2, hct3⟩ := hct2
  have hout := hct3
  clear hct3
  have e_w_7_0 : w_7_0 = famAt x1 ((x2 * 2) + (Int.ofNat n13)).toNat := (GenReq.famGet_famAt a2).2.2
  clear a2
  subst e_w_7_0
  exact hout

/-- One step of `sequential_generatedRoot_82`: a true state stays true only if its conditions hold. -/
theorem step_sequential_generatedRoot_82 {params : W_Requirement_0.Params} {n14 : Nat} {cur next : Bool} {x1 : Fin 14400 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_82 params n14 (cur, (x1, ())) next) :
    ReqStep.Spec (fun n14 ↦ (0 + ∑ n13 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n14) * 2) + (Int.ofNat n13)).toNat)) = 1) n14 cur next := by
  obtain ⟨⟨w_5_0⟩, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨a1, hct2⟩ := hct1
  have hout := hct2
  clear hct2
  have e_w_5_0 : w_5_0 = 0 + ∑ n13 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n14) * 2) + (Int.ofNat n13)).toNat) := seq_sum (fun n13 ↦ famAt x1 (((Int.ofNat n14) * 2) + (Int.ofNat n13)).toNat) (fun _ _ _ hb ↦ sum_sequential_sequential_generatedRoot_82_5 hb) a1
  clear a1
  subst e_w_5_0
  intro hn
  rw [hout] at hn
  obtain ⟨hn1, f0⟩ := link_and hn
  exact ⟨hn1, f0⟩

/-- One step of `sequential_sequential_generatedRoot_84_5`: it adds a term to the state. -/
theorem sum_sequential_sequential_generatedRoot_84_5 {params : W_Requirement_0.Params} {j0 n15 : Nat} {cur next : Int} {x1 : Fin 14400 → Int} {x2 : Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_84_5 params j0 n15 (cur, (x1, (x2, ()))) next) :
    next = cur + (famAt x1 ((x2 * 2) + (Int.ofNat n15)).toNat) := by
  obtain ⟨w_7_0, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨-, hct2⟩ := hct1
  obtain ⟨a2, hct3⟩ := hct2
  have hout := hct3
  clear hct3
  have e_w_7_0 : w_7_0 = famAt x1 ((x2 * 2) + (Int.ofNat n15)).toNat := (GenReq.famGet_famAt a2).2.2
  clear a2
  subst e_w_7_0
  exact hout

/-- One step of `sequential_generatedRoot_84`: a true state stays true only if its conditions hold. -/
theorem step_sequential_generatedRoot_84 {params : W_Requirement_0.Params} {n16 : Nat} {cur next : Bool} {x1 : Fin 14400 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_84 params n16 (cur, (x1, ())) next) :
    ReqStep.Spec (fun n16 ↦ (0 + ∑ n15 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n16) * 2) + (Int.ofNat n15)).toNat)) = 1) n16 cur next := by
  obtain ⟨⟨w_5_0⟩, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨a1, hct2⟩ := hct1
  have hout := hct2
  clear hct2
  have e_w_5_0 : w_5_0 = 0 + ∑ n15 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n16) * 2) + (Int.ofNat n15)).toNat) := seq_sum (fun n15 ↦ famAt x1 (((Int.ofNat n16) * 2) + (Int.ofNat n15)).toNat) (fun _ _ _ hb ↦ sum_sequential_sequential_generatedRoot_84_5 hb) a1
  clear a1
  subst e_w_5_0
  intro hn
  rw [hout] at hn
  obtain ⟨hn1, f0⟩ := link_and hn
  exact ⟨hn1, f0⟩

/-- One step of `sequential_sequential_generatedRoot_86_5`: it adds a term to the state. -/
theorem sum_sequential_sequential_generatedRoot_86_5 {params : W_Requirement_0.Params} {j0 n17 : Nat} {cur next : Int} {x1 : Fin 16 → Int} {x2 : Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_86_5 params j0 n17 (cur, (x1, (x2, ()))) next) :
    next = cur + (famAt x1 ((x2 * 2) + (Int.ofNat n17)).toNat) := by
  obtain ⟨w_7_0, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨-, hct2⟩ := hct1
  obtain ⟨a2, hct3⟩ := hct2
  have hout := hct3
  clear hct3
  have e_w_7_0 : w_7_0 = famAt x1 ((x2 * 2) + (Int.ofNat n17)).toNat := (GenReq.famGet_famAt a2).2.2
  clear a2
  subst e_w_7_0
  exact hout

/-- One step of `sequential_generatedRoot_86`: a true state stays true only if its conditions hold. -/
theorem step_sequential_generatedRoot_86 {params : W_Requirement_0.Params} {n18 : Nat} {cur next : Bool} {x1 : Fin 16 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_86 params n18 (cur, (x1, ())) next) :
    ReqStep.Spec (fun n18 ↦ (0 + ∑ n17 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n18) * 2) + (Int.ofNat n17)).toNat)) = 1) n18 cur next := by
  obtain ⟨⟨w_5_0⟩, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨a1, hct2⟩ := hct1
  have hout := hct2
  clear hct2
  have e_w_5_0 : w_5_0 = 0 + ∑ n17 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n18) * 2) + (Int.ofNat n17)).toNat) := seq_sum (fun n17 ↦ famAt x1 (((Int.ofNat n18) * 2) + (Int.ofNat n17)).toNat) (fun _ _ _ hb ↦ sum_sequential_sequential_generatedRoot_86_5 hb) a1
  clear a1
  subst e_w_5_0
  intro hn
  rw [hout] at hn
  obtain ⟨hn1, f0⟩ := link_and hn
  exact ⟨hn1, f0⟩

/-- One step of `sequential_sequential_generatedRoot_88_5`: it adds a term to the state. -/
theorem sum_sequential_sequential_generatedRoot_88_5 {params : W_Requirement_0.Params} {j0 n19 : Nat} {cur next : Int} {x1 : Fin 16 → Int} {x2 : Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_88_5 params j0 n19 (cur, (x1, (x2, ()))) next) :
    next = cur + (famAt x1 ((x2 * 2) + (Int.ofNat n19)).toNat) := by
  obtain ⟨w_7_0, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨-, hct2⟩ := hct1
  obtain ⟨a2, hct3⟩ := hct2
  have hout := hct3
  clear hct3
  have e_w_7_0 : w_7_0 = famAt x1 ((x2 * 2) + (Int.ofNat n19)).toNat := (GenReq.famGet_famAt a2).2.2
  clear a2
  subst e_w_7_0
  exact hout

/-- One step of `sequential_generatedRoot_88`: a true state stays true only if its conditions hold. -/
theorem step_sequential_generatedRoot_88 {params : W_Requirement_0.Params} {n20 : Nat} {cur next : Bool} {x1 : Fin 16 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_88 params n20 (cur, (x1, ())) next) :
    ReqStep.Spec (fun n20 ↦ (0 + ∑ n19 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n20) * 2) + (Int.ofNat n19)).toNat)) = 1) n20 cur next := by
  obtain ⟨⟨w_5_0⟩, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨a1, hct2⟩ := hct1
  have hout := hct2
  clear hct2
  have e_w_5_0 : w_5_0 = 0 + ∑ n19 ∈ Finset.range 2, (famAt x1 (((Int.ofNat n20) * 2) + (Int.ofNat n19)).toNat) := seq_sum (fun n19 ↦ famAt x1 (((Int.ofNat n20) * 2) + (Int.ofNat n19)).toNat) (fun _ _ _ hb ↦ sum_sequential_sequential_generatedRoot_88_5 hb) a1
  clear a1
  subst e_w_5_0
  intro hn
  rw [hout] at hn
  obtain ⟨hn1, f0⟩ := link_and hn
  exact ⟨hn1, f0⟩

/-- The facts a true requirement result gives. -/
structure Facts (x0 : Fin 5 → Int) (x1 : Fin 5 → Int) (x2 : Fin 2 → Int) (x3 : Fin 2 → Int) (x4 : Fin 1 → Int) (x5 : Fin 1 → Int) (x6 : Fin 2 → Int) (x7 : Fin 2 → Int) (x8 : Fin 2 → Int) (x9 : Fin 1 → Int) (x10 : Fin 1 → Int) (x11 : Fin 72000 → Int) (x12 : Fin 72000 → Int) (x13 : Fin 16 → Int) (x14 : Fin 16 → Int) (x15 : Fin 14400 → Int) (x16 : Fin 14400 → Int) (x17 : Fin 16 → Int) (x18 : Fin 16 → Int) : Prop where
  /-- every iteration of `sequential_generatedRoot_3`. -/
  loop_3 : ∀ n0 < 5, (famAt x0 n0) = (famAt x1 n0)
  /-- every iteration of `sequential_generatedRoot_6`. -/
  loop_6 : ∀ n1 < 2, ((if (decide (((famAt x2 n1) * (famAt x2 n1)) = 1)) then (1 : Int) else 0) + (if (decide ((((famAt x3 n1) * 2) + ((1 - (famAt x2 n1)) * 8)) = (((MxxIR.selectInt (if (decide ((Int.ofNat n1) = 1)) then (1 : Int) else 0) [(famAt x1 (((Int.ofNat n1) % 1) * 5).toNat), 0]) * 1) * 2))) then (1 : Int) else 0)) = 2
  /-- a condition of the root. -/
  cond_2 : ((if (decide (((famAt x4 0) * (famAt x4 0)) = 1)) then (1 : Int) else 0) + (if (decide ((((famAt x5 0) * 2) + ((1 - (famAt x4 0)) * 8)) = (((famAt x6 0) * 2) * 2))) then (1 : Int) else 0)) = 2
  /-- every iteration of `sequential_generatedRoot_42`. -/
  loop_42 : ∀ n4 < 2, ((if (decide (((famAt x7 n4) * (famAt x7 n4)) = 1)) then (1 : Int) else 0) + (if (decide ((((famAt x8 n4) * 2) + ((1 - (famAt x7 n4)) * 8)) = (((MxxIR.selectInt (if (decide ((Int.ofNat n4) = 1)) then (1 : Int) else 0) [(famAt (fun member : Fin 5 ↦ ((0 + ∑ n2 ∈ Finset.range 4, (MxxIR.selectInt (if (decide ((Int.ofNat n2) < (famAt (fun index : Fin 5 ↦ ((Sha256Columns.column [17, 229, 242, 240, 20, 113, 12, 254, 126, 237, 25, 125, 17, 124, 200, 38, 220, 135, 58, 4, 215, 86, 146, 138, 135, 103, 188, 97, 153, 28, 159, 14] 4 5 2 index.val).2 : Int)) ((Int.ofNat member.val) % 5).toNat))) then 1 else 0) [0, (famAt x1 (((((Int.ofNat member.val) / 5) * 5) + ((0 + ((if (decide (((famAt (fun index : Fin 5 ↦ ((Sha256Columns.column [17, 229, 242, 240, 20, 113, 12, 254, 126, 237, 25, 125, 17, 124, 200, 38, 220, 135, 58, 4, 215, 86, 146, 138, 135, 103, 188, 97, 153, 28, 159, 14] 4 5 2 index.val).1 : Int)) ((Int.ofNat member.val) % 5).toNat) / (2 ^ (Int.toNat ((2 * (Int.ofNat n2)))))) % 2 = 1)) then 1 else 0) * 1)) + ((if (decide (((famAt (fun index : Fin 5 ↦ ((Sha256Columns.column [17, 229, 242, 240, 20, 113, 12, 254, 126, 237, 25, 125, 17, 124, 200, 38, 220, 135, 58, 4, 215, 86, 146, 138, 135, 103, 188, 97, 153, 28, 159, 14] 4 5 2 index.val).1 : Int)) ((Int.ofNat member.val) % 5).toNat) / (2 ^ (Int.toNat ((1 + (2 * (Int.ofNat n2))))))) % 2 = 1)) then 1 else 0) * 2))) + 1).toNat)])) % 16)) (((Int.ofNat n4) % 1) * 5).toNat), 0]) * 1) * 2))) then (1 : Int) else 0)) = 2
  /-- a condition of the root. -/
  cond_4 : ((if (decide (((famAt x9 0) * (famAt x9 0)) = 1)) then (1 : Int) else 0) + (if (decide ((((famAt x10 0) * 2) + ((1 - (famAt x9 0)) * 8)) = (((famAt x6 1) * 2) * 2))) then (1 : Int) else 0)) = 2
  /-- every iteration of `sequential_generatedRoot_74`. -/
  loop_74 : ∀ n6 < 36000, (0 + ∑ n5 ∈ Finset.range 2, (famAt x11 (((Int.ofNat n6) * 2) + (Int.ofNat n5)).toNat)) = 1
  /-- every iteration of `sequential_generatedRoot_76`. -/
  loop_76 : ∀ n8 < 36000, (0 + ∑ n7 ∈ Finset.range 2, (famAt x12 (((Int.ofNat n8) * 2) + (Int.ofNat n7)).toNat)) = 1
  /-- every iteration of `sequential_generatedRoot_78`. -/
  loop_78 : ∀ n10 < 8, (0 + ∑ n9 ∈ Finset.range 2, (famAt x13 (((Int.ofNat n10) * 2) + (Int.ofNat n9)).toNat)) = 1
  /-- every iteration of `sequential_generatedRoot_80`. -/
  loop_80 : ∀ n12 < 8, (0 + ∑ n11 ∈ Finset.range 2, (famAt x14 (((Int.ofNat n12) * 2) + (Int.ofNat n11)).toNat)) = 1
  /-- every iteration of `sequential_generatedRoot_82`. -/
  loop_82 : ∀ n14 < 7200, (0 + ∑ n13 ∈ Finset.range 2, (famAt x15 (((Int.ofNat n14) * 2) + (Int.ofNat n13)).toNat)) = 1
  /-- every iteration of `sequential_generatedRoot_84`. -/
  loop_84 : ∀ n16 < 7200, (0 + ∑ n15 ∈ Finset.range 2, (famAt x16 (((Int.ofNat n16) * 2) + (Int.ofNat n15)).toNat)) = 1
  /-- every iteration of `sequential_generatedRoot_86`. -/
  loop_86 : ∀ n18 < 8, (0 + ∑ n17 ∈ Finset.range 2, (famAt x17 (((Int.ofNat n18) * 2) + (Int.ofNat n17)).toNat)) = 1
  /-- every iteration of `sequential_generatedRoot_88`. -/
  loop_88 : ∀ n20 < 8, (0 + ∑ n19 ∈ Finset.range 2, (famAt x18 (((Int.ofNat n20) * 2) + (Int.ofNat n19)).toNat)) = 1

/-- A true requirement result gives all its facts. -/
theorem requirement_facts {params : W_Requirement_0.Params} {x0 : Fin 5 → Int} {x1 : Fin 5 → Int} {x2 : Fin 2 → Int} {x3 : Fin 2 → Int} {x4 : Fin 1 → Int} {x5 : Fin 1 → Int} {x6 : Fin 2 → Int} {x7 : Fin 2 → Int} {x8 : Fin 2 → Int} {x9 : Fin 1 → Int} {x10 : Fin 1 → Int} {x11 : Fin 72000 → Int} {x12 : Fin 72000 → Int} {x13 : Fin 16 → Int} {x14 : Fin 16 → Int} {x15 : Fin 14400 → Int} {x16 : Fin 14400 → Int} {x17 : Fin 16 → Int} {x18 : Fin 16 → Int} {out : Bool}
    (h : W_Requirement_0.generatedRoot params (x0, (x1, (x2, (x3, (x4, (x5, (x6, (x7, (x8, (x9, (x10, (x11, (x12, (x13, (x14, (x15, (x16, (x17, (x18, ()))))))))))))))))))) out) (ho : out = true) : Facts x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 := by
  obtain ⟨⟨w_3_0, w_6_0, w_9_0, w_15_0, w_24_0, w_40_0, w_40_1, w_41_0, w_42_0, w_45_0, w_51_0, w_59_0, w_74_0, w_76_0, w_78_0, w_80_0, w_82_0, w_84_0, w_86_0, w_88_0⟩, hc⟩ := h
  obtain ⟨-, hct1⟩ := hc
  obtain ⟨a1, hct2⟩ := hct1
  obtain ⟨-, hct3⟩ := hct2
  obtain ⟨a3, hct4⟩ := hct3
  obtain ⟨a4, hct5⟩ := hct4
  obtain ⟨a5, hct6⟩ := hct5
  obtain ⟨a6, hct7⟩ := hct6
  obtain ⟨a7, hct8⟩ := hct7
  obtain ⟨-, hct9⟩ := hct8
  obtain ⟨a9, hct10⟩ := hct9
  obtain ⟨-, hct11⟩ := hct10
  obtain ⟨a11, hct12⟩ := hct11
  obtain ⟨a12, hct13⟩ := hct12
  obtain ⟨a13, hct14⟩ := hct13
  obtain ⟨a14, hct15⟩ := hct14
  obtain ⟨-, hct16⟩ := hct15
  obtain ⟨a16, hct17⟩ := hct16
  obtain ⟨-, hct18⟩ := hct17
  obtain ⟨a18, hct19⟩ := hct18
  obtain ⟨-, hct20⟩ := hct19
  obtain ⟨a20, hct21⟩ := hct20
  obtain ⟨-, hct22⟩ := hct21
  obtain ⟨a22, hct23⟩ := hct22
  obtain ⟨-, hct24⟩ := hct23
  obtain ⟨a24, hct25⟩ := hct24
  obtain ⟨-, hct26⟩ := hct25
  obtain ⟨a26, hct27⟩ := hct26
  obtain ⟨-, hct28⟩ := hct27
  obtain ⟨a28, hct29⟩ := hct28
  obtain ⟨-, hct30⟩ := hct29
  obtain ⟨a30, hct31⟩ := hct30
  have hout := hct31
  clear hct31
  have e_w_9_0 : w_9_0 = famAt x4 0 := (GenReq.famGet_famAt a4).2.2
  have e_w_15_0 : w_15_0 = famAt x5 0 := (GenReq.famGet_famAt a5).2.2
  have e_w_24_0 : w_24_0 = famAt x6 0 := (GenReq.famGet_famAt a6).2.2
  have e_w_40_0 : w_40_0 = (fun index : Fin 5 ↦ ((Sha256Columns.column [17, 229, 242, 240, 20, 113, 12, 254, 126, 237, 25, 125, 17, 124, 200, 38, 220, 135, 58, 4, 215, 86, 146, 138, 135, 103, 188, 97, 153, 28, 159, 14] 4 5 2 index.val).1 : Int)) := by funext index; exact (a7.2.2.2.2.2 index).1
  have e_w_40_1 : w_40_1 = (fun index : Fin 5 ↦ ((Sha256Columns.column [17, 229, 242, 240, 20, 113, 12, 254, 126, 237, 25, 125, 17, 124, 200, 38, 220, 135, 58, 4, 215, 86, 146, 138, 135, 103, 188, 97, 153, 28, 159, 14] 4 5 2 index.val).2 : Int)) := by funext index; exact (a7.2.2.2.2.2 index).2
  have e_w_41_0 : w_41_0 = (fun member : Fin 5 ↦ ((0 + ∑ n2 ∈ Finset.range 4, (MxxIR.selectInt (if (decide ((Int.ofNat n2) < (famAt w_40_1 ((Int.ofNat member.val) % 5).toNat))) then 1 else 0) [0, (famAt x1 (((((Int.ofNat member.val) / 5) * 5) + ((0 + ((if (decide (((famAt w_40_0 ((Int.ofNat member.val) % 5).toNat) / (2 ^ (Int.toNat ((2 * (Int.ofNat n2)))))) % 2 = 1)) then 1 else 0) * 1)) + ((if (decide (((famAt w_40_0 ((Int.ofNat member.val) % 5).toNat) / (2 ^ (Int.toNat ((1 + (2 * (Int.ofNat n2))))))) % 2 = 1)) then 1 else 0) * 2))) + 1).toNat)])) % 16)) := by funext member; exact value_parallel_generatedRoot_41 (a9 member)
  have e_w_45_0 : w_45_0 = famAt x9 0 := (GenReq.famGet_famAt a12).2.2
  have e_w_51_0 : w_51_0 = famAt x10 0 := (GenReq.famGet_famAt a13).2.2
  have e_w_59_0 : w_59_0 = famAt x6 1 := (GenReq.famGet_famAt a14).2.2
  have L_w_3_0 := ReqAll.sem a1 (fun _ _ _ hb ↦ step_sequential_generatedRoot_3 hb)
  have L_w_6_0 := ReqAll.sem a3 (fun _ _ _ hb ↦ step_sequential_generatedRoot_6 hb)
  have L_w_42_0 := ReqAll.sem a11 (fun _ _ _ hb ↦ step_sequential_generatedRoot_42 hb)
  have L_w_74_0 := ReqAll.sem a16 (fun _ _ _ hb ↦ step_sequential_generatedRoot_74 hb)
  have L_w_76_0 := ReqAll.sem a18 (fun _ _ _ hb ↦ step_sequential_generatedRoot_76 hb)
  have L_w_78_0 := ReqAll.sem a20 (fun _ _ _ hb ↦ step_sequential_generatedRoot_78 hb)
  have L_w_80_0 := ReqAll.sem a22 (fun _ _ _ hb ↦ step_sequential_generatedRoot_80 hb)
  have L_w_82_0 := ReqAll.sem a24 (fun _ _ _ hb ↦ step_sequential_generatedRoot_82 hb)
  have L_w_84_0 := ReqAll.sem a26 (fun _ _ _ hb ↦ step_sequential_generatedRoot_84 hb)
  have L_w_86_0 := ReqAll.sem a28 (fun _ _ _ hb ↦ step_sequential_generatedRoot_86 hb)
  have L_w_88_0 := ReqAll.sem a30 (fun _ _ _ hb ↦ step_sequential_generatedRoot_88 hb)
  have hn := ho
  rw [hout] at hn
  obtain ⟨hn1, f12⟩ := link_loop L_w_88_0 hn
  obtain ⟨hn2, f11⟩ := link_loop L_w_86_0 hn1
  obtain ⟨hn3, f10⟩ := link_loop L_w_84_0 hn2
  obtain ⟨hn4, f9⟩ := link_loop L_w_82_0 hn3
  obtain ⟨hn5, f8⟩ := link_loop L_w_80_0 hn4
  obtain ⟨hn6, f7⟩ := link_loop L_w_78_0 hn5
  obtain ⟨hn7, f6⟩ := link_loop L_w_76_0 hn6
  obtain ⟨hn8, f5⟩ := link_loop L_w_74_0 hn7
  obtain ⟨hn9, f4⟩ := link_and hn8
  obtain ⟨hn10, f3⟩ := link_loop L_w_42_0 hn9
  obtain ⟨hn11, f2⟩ := link_and hn10
  obtain ⟨hn12, f1⟩ := link_loop L_w_6_0 hn11
  obtain ⟨hn13, f0⟩ := link_loop L_w_3_0 hn12
  try simp only [e_w_9_0, e_w_15_0, e_w_24_0, e_w_40_0, e_w_40_1, e_w_41_0, e_w_45_0, e_w_51_0, e_w_59_0] at f0
  try simp only [e_w_9_0, e_w_15_0, e_w_24_0, e_w_40_0, e_w_40_1, e_w_41_0, e_w_45_0, e_w_51_0, e_w_59_0] at f1
  try simp only [e_w_9_0, e_w_15_0, e_w_24_0, e_w_40_0, e_w_40_1, e_w_41_0, e_w_45_0, e_w_51_0, e_w_59_0] at f2
  try simp only [e_w_9_0, e_w_15_0, e_w_24_0, e_w_40_0, e_w_40_1, e_w_41_0, e_w_45_0, e_w_51_0, e_w_59_0] at f3
  try simp only [e_w_9_0, e_w_15_0, e_w_24_0, e_w_40_0, e_w_40_1, e_w_41_0, e_w_45_0, e_w_51_0, e_w_59_0] at f4
  try simp only [e_w_9_0, e_w_15_0, e_w_24_0, e_w_40_0, e_w_40_1, e_w_41_0, e_w_45_0, e_w_51_0, e_w_59_0] at f5
  try simp only [e_w_9_0, e_w_15_0, e_w_24_0, e_w_40_0, e_w_40_1, e_w_41_0, e_w_45_0, e_w_51_0, e_w_59_0] at f6
  try simp only [e_w_9_0, e_w_15_0, e_w_24_0, e_w_40_0, e_w_40_1, e_w_41_0, e_w_45_0, e_w_51_0, e_w_59_0] at f7
  try simp only [e_w_9_0, e_w_15_0, e_w_24_0, e_w_40_0, e_w_40_1, e_w_41_0, e_w_45_0, e_w_51_0, e_w_59_0] at f8
  try simp only [e_w_9_0, e_w_15_0, e_w_24_0, e_w_40_0, e_w_40_1, e_w_41_0, e_w_45_0, e_w_51_0, e_w_59_0] at f9
  try simp only [e_w_9_0, e_w_15_0, e_w_24_0, e_w_40_0, e_w_40_1, e_w_41_0, e_w_45_0, e_w_51_0, e_w_59_0] at f10
  try simp only [e_w_9_0, e_w_15_0, e_w_24_0, e_w_40_0, e_w_40_1, e_w_41_0, e_w_45_0, e_w_51_0, e_w_59_0] at f11
  try simp only [e_w_9_0, e_w_15_0, e_w_24_0, e_w_40_0, e_w_40_1, e_w_41_0, e_w_45_0, e_w_51_0, e_w_59_0] at f12
  exact ⟨f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12⟩

end Dio2Requirement
