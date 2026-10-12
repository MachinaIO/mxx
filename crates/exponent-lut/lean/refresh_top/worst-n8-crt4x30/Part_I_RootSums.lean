import TopInst
import TopLemmas

/-! The member sums of the online root (written by `generator/gen_root_sums.py`). -/

namespace RefreshTopProof.Part_I

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

theorem rootStep332 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {i : Nat} {cur next : Blocks Q0 6}
    {g0 g1 g2 g3 g4 g5 : Fin 8 → ExactMatrix Q0 N 1 6}
    (h : Stage_online_eval.sequential_parallel_generatedRoot_290_332 Backend.backend model params i0 i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold Stage_online_eval.sequential_parallel_generatedRoot_290_332 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_17_0 w_21_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 at h
  simp only [Stage_online_eval.sequential_parallel_generatedRoot_290_332.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, h2_0_8, h2_0_9, h19_0, h2_0_10, h2_0_11, h23_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  have e4 := familyGet_cellAt h19_0
  have e5 := familyGet_cellAt h23_0
  subst e0 e1 e2 e3 e4 e5 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `332`: the sums of the families over their 8 members. -/
theorem rootSum332 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {g0 g1 g2 g3 g4 g5 : Fin 8 → ExactMatrix Q0 N 1 6}
    {a0 b0 a1 b1 a2 b2 a3 b3 a4 b4 a5 b5 : ExactMatrix Q0 N 1 6}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    (ha4 : familyGetStatic g4 0 a4) (hb4 : familyGetStatic g4 1 b4)
    (ha5 : familyGetStatic g5 0 a5) (hb5 : familyGetStatic g5 1 b5)
    {final : Blocks Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Blocks Q0 6) ↦
      Stage_online_eval.sequential_parallel_generatedRoot_290_332 Backend.backend model params i0 i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1, current.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5, ()) next) (Int.toNat 6)
      (matrixAdd a0 b0, matrixAdd a1 b1, matrixAdd a2 b2, matrixAdd a3 b3, matrixAdd a4 b4, matrixAdd a5 b5, ()) final) :
    final = (∑ k, g0 k, ∑ k, g1 k, ∑ k, g2 k, ∑ k, g3 k, ∑ k, g4 k, ∑ k, g5 k, ()) := by
  have hloop := iter_sum (fun i ↦ (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()))
    (fun i current next hbody ↦ rootStep332 hbody) hruns
  rw [show Int.toNat 6 = 6 from rfl] at hloop
  obtain ⟨_, ea0⟩ := familyGet_eq (k := 0) ha0
  obtain ⟨_, eb0⟩ := familyGet_eq (k := 1) hb0
  obtain ⟨_, ea1⟩ := familyGet_eq (k := 0) ha1
  obtain ⟨_, eb1⟩ := familyGet_eq (k := 1) hb1
  obtain ⟨_, ea2⟩ := familyGet_eq (k := 0) ha2
  obtain ⟨_, eb2⟩ := familyGet_eq (k := 1) hb2
  obtain ⟨_, ea3⟩ := familyGet_eq (k := 0) ha3
  obtain ⟨_, eb3⟩ := familyGet_eq (k := 1) hb3
  obtain ⟨_, ea4⟩ := familyGet_eq (k := 0) ha4
  obtain ⟨_, eb4⟩ := familyGet_eq (k := 1) hb4
  obtain ⟨_, ea5⟩ := familyGet_eq (k := 0) ha5
  obtain ⟨_, eb5⟩ := familyGet_eq (k := 1) hb5
  rw [hloop]
  simp only [ea0, eb0, ea1, eb1, ea2, eb2, ea3, eb3, ea4, eb4, ea5, eb5, matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 6) g0
  · exact sum_cells (w := 6) g1
  · exact sum_cells (w := 6) g2
  · exact sum_cells (w := 6) g3
  · exact sum_cells (w := 6) g4
  · exact sum_cells (w := 6) g5

theorem rootStep413 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {i : Nat} {cur next : Blocks Q0 6}
    {g0 g1 g2 g3 g4 g5 : Fin 8 → ExactMatrix Q0 N 1 6}
    (h : Stage_online_eval.sequential_parallel_generatedRoot_290_415 Backend.backend model params i0 i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold Stage_online_eval.sequential_parallel_generatedRoot_290_415 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_17_0 w_21_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 at h
  simp only [Stage_online_eval.sequential_parallel_generatedRoot_290_415.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, h2_0_8, h2_0_9, h19_0, h2_0_10, h2_0_11, h23_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  have e4 := familyGet_cellAt h19_0
  have e5 := familyGet_cellAt h23_0
  subst e0 e1 e2 e3 e4 e5 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `413`: the sums of the families over their 8 members. -/
theorem rootSum413 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {g0 g1 g2 g3 g4 g5 : Fin 8 → ExactMatrix Q0 N 1 6}
    {a0 b0 a1 b1 a2 b2 a3 b3 a4 b4 a5 b5 : ExactMatrix Q0 N 1 6}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    (ha4 : familyGetStatic g4 0 a4) (hb4 : familyGetStatic g4 1 b4)
    (ha5 : familyGetStatic g5 0 a5) (hb5 : familyGetStatic g5 1 b5)
    {final : Blocks Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Blocks Q0 6) ↦
      Stage_online_eval.sequential_parallel_generatedRoot_290_415 Backend.backend model params i0 i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1, current.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5, ()) next) (Int.toNat 6)
      (matrixAdd a0 b0, matrixAdd a1 b1, matrixAdd a2 b2, matrixAdd a3 b3, matrixAdd a4 b4, matrixAdd a5 b5, ()) final) :
    final = (∑ k, g0 k, ∑ k, g1 k, ∑ k, g2 k, ∑ k, g3 k, ∑ k, g4 k, ∑ k, g5 k, ()) := by
  have hloop := iter_sum (fun i ↦ (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()))
    (fun i current next hbody ↦ rootStep413 hbody) hruns
  rw [show Int.toNat 6 = 6 from rfl] at hloop
  obtain ⟨_, ea0⟩ := familyGet_eq (k := 0) ha0
  obtain ⟨_, eb0⟩ := familyGet_eq (k := 1) hb0
  obtain ⟨_, ea1⟩ := familyGet_eq (k := 0) ha1
  obtain ⟨_, eb1⟩ := familyGet_eq (k := 1) hb1
  obtain ⟨_, ea2⟩ := familyGet_eq (k := 0) ha2
  obtain ⟨_, eb2⟩ := familyGet_eq (k := 1) hb2
  obtain ⟨_, ea3⟩ := familyGet_eq (k := 0) ha3
  obtain ⟨_, eb3⟩ := familyGet_eq (k := 1) hb3
  obtain ⟨_, ea4⟩ := familyGet_eq (k := 0) ha4
  obtain ⟨_, eb4⟩ := familyGet_eq (k := 1) hb4
  obtain ⟨_, ea5⟩ := familyGet_eq (k := 0) ha5
  obtain ⟨_, eb5⟩ := familyGet_eq (k := 1) hb5
  rw [hloop]
  simp only [ea0, eb0, ea1, eb1, ea2, eb2, ea3, eb3, ea4, eb4, ea5, eb5, matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 6) g0
  · exact sum_cells (w := 6) g1
  · exact sum_cells (w := 6) g2
  · exact sum_cells (w := 6) g3
  · exact sum_cells (w := 6) g4
  · exact sum_cells (w := 6) g5

theorem rootStep438 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {i : Nat} {cur next : Blocks Q0 6}
    {g0 g1 g2 g3 g4 g5 : Fin 8 → ExactMatrix Q0 N 1 6}
    (h : Stage_online_eval.sequential_parallel_generatedRoot_290_440 Backend.backend model params i0 i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold Stage_online_eval.sequential_parallel_generatedRoot_290_440 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_17_0 w_21_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 at h
  simp only [Stage_online_eval.sequential_parallel_generatedRoot_290_440.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, h2_0_8, h2_0_9, h19_0, h2_0_10, h2_0_11, h23_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  have e4 := familyGet_cellAt h19_0
  have e5 := familyGet_cellAt h23_0
  subst e0 e1 e2 e3 e4 e5 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `438`: the sums of the families over their 8 members. -/
theorem rootSum438 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {g0 g1 g2 g3 g4 g5 : Fin 8 → ExactMatrix Q0 N 1 6}
    {a0 b0 a1 b1 a2 b2 a3 b3 a4 b4 a5 b5 : ExactMatrix Q0 N 1 6}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    (ha4 : familyGetStatic g4 0 a4) (hb4 : familyGetStatic g4 1 b4)
    (ha5 : familyGetStatic g5 0 a5) (hb5 : familyGetStatic g5 1 b5)
    {final : Blocks Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Blocks Q0 6) ↦
      Stage_online_eval.sequential_parallel_generatedRoot_290_440 Backend.backend model params i0 i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1, current.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5, ()) next) (Int.toNat 6)
      (matrixAdd a0 b0, matrixAdd a1 b1, matrixAdd a2 b2, matrixAdd a3 b3, matrixAdd a4 b4, matrixAdd a5 b5, ()) final) :
    final = (∑ k, g0 k, ∑ k, g1 k, ∑ k, g2 k, ∑ k, g3 k, ∑ k, g4 k, ∑ k, g5 k, ()) := by
  have hloop := iter_sum (fun i ↦ (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()))
    (fun i current next hbody ↦ rootStep438 hbody) hruns
  rw [show Int.toNat 6 = 6 from rfl] at hloop
  obtain ⟨_, ea0⟩ := familyGet_eq (k := 0) ha0
  obtain ⟨_, eb0⟩ := familyGet_eq (k := 1) hb0
  obtain ⟨_, ea1⟩ := familyGet_eq (k := 0) ha1
  obtain ⟨_, eb1⟩ := familyGet_eq (k := 1) hb1
  obtain ⟨_, ea2⟩ := familyGet_eq (k := 0) ha2
  obtain ⟨_, eb2⟩ := familyGet_eq (k := 1) hb2
  obtain ⟨_, ea3⟩ := familyGet_eq (k := 0) ha3
  obtain ⟨_, eb3⟩ := familyGet_eq (k := 1) hb3
  obtain ⟨_, ea4⟩ := familyGet_eq (k := 0) ha4
  obtain ⟨_, eb4⟩ := familyGet_eq (k := 1) hb4
  obtain ⟨_, ea5⟩ := familyGet_eq (k := 0) ha5
  obtain ⟨_, eb5⟩ := familyGet_eq (k := 1) hb5
  rw [hloop]
  simp only [ea0, eb0, ea1, eb1, ea2, eb2, ea3, eb3, ea4, eb4, ea5, eb5, matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 6) g0
  · exact sum_cells (w := 6) g1
  · exact sum_cells (w := 6) g2
  · exact sum_cells (w := 6) g3
  · exact sum_cells (w := 6) g4
  · exact sum_cells (w := 6) g5

theorem rootStep460 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {i : Nat} {cur next : Blocks Q0 6}
    {g0 g1 g2 g3 g4 g5 : Fin 8 → ExactMatrix Q0 N 1 6}
    (h : Stage_online_eval.sequential_parallel_generatedRoot_290_462 Backend.backend model params i0 i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold Stage_online_eval.sequential_parallel_generatedRoot_290_462 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_17_0 w_21_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 at h
  simp only [Stage_online_eval.sequential_parallel_generatedRoot_290_462.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, h2_0_8, h2_0_9, h19_0, h2_0_10, h2_0_11, h23_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  have e4 := familyGet_cellAt h19_0
  have e5 := familyGet_cellAt h23_0
  subst e0 e1 e2 e3 e4 e5 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `460`: the sums of the families over their 8 members. -/
theorem rootSum460 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {g0 g1 g2 g3 g4 g5 : Fin 8 → ExactMatrix Q0 N 1 6}
    {a0 b0 a1 b1 a2 b2 a3 b3 a4 b4 a5 b5 : ExactMatrix Q0 N 1 6}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    (ha4 : familyGetStatic g4 0 a4) (hb4 : familyGetStatic g4 1 b4)
    (ha5 : familyGetStatic g5 0 a5) (hb5 : familyGetStatic g5 1 b5)
    {final : Blocks Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Blocks Q0 6) ↦
      Stage_online_eval.sequential_parallel_generatedRoot_290_462 Backend.backend model params i0 i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1, current.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5, ()) next) (Int.toNat 6)
      (matrixAdd a0 b0, matrixAdd a1 b1, matrixAdd a2 b2, matrixAdd a3 b3, matrixAdd a4 b4, matrixAdd a5 b5, ()) final) :
    final = (∑ k, g0 k, ∑ k, g1 k, ∑ k, g2 k, ∑ k, g3 k, ∑ k, g4 k, ∑ k, g5 k, ()) := by
  have hloop := iter_sum (fun i ↦ (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()))
    (fun i current next hbody ↦ rootStep460 hbody) hruns
  rw [show Int.toNat 6 = 6 from rfl] at hloop
  obtain ⟨_, ea0⟩ := familyGet_eq (k := 0) ha0
  obtain ⟨_, eb0⟩ := familyGet_eq (k := 1) hb0
  obtain ⟨_, ea1⟩ := familyGet_eq (k := 0) ha1
  obtain ⟨_, eb1⟩ := familyGet_eq (k := 1) hb1
  obtain ⟨_, ea2⟩ := familyGet_eq (k := 0) ha2
  obtain ⟨_, eb2⟩ := familyGet_eq (k := 1) hb2
  obtain ⟨_, ea3⟩ := familyGet_eq (k := 0) ha3
  obtain ⟨_, eb3⟩ := familyGet_eq (k := 1) hb3
  obtain ⟨_, ea4⟩ := familyGet_eq (k := 0) ha4
  obtain ⟨_, eb4⟩ := familyGet_eq (k := 1) hb4
  obtain ⟨_, ea5⟩ := familyGet_eq (k := 0) ha5
  obtain ⟨_, eb5⟩ := familyGet_eq (k := 1) hb5
  rw [hloop]
  simp only [ea0, eb0, ea1, eb1, ea2, eb2, ea3, eb3, ea4, eb4, ea5, eb5, matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 6) g0
  · exact sum_cells (w := 6) g1
  · exact sum_cells (w := 6) g2
  · exact sum_cells (w := 6) g3
  · exact sum_cells (w := 6) g4
  · exact sum_cells (w := 6) g5

theorem rootStep482 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {i : Nat} {cur next : Blocks Q0 6}
    {g0 g1 g2 g3 g4 g5 : Fin 8 → ExactMatrix Q0 N 1 6}
    (h : Stage_online_eval.sequential_parallel_generatedRoot_290_484 Backend.backend model params i0 i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold Stage_online_eval.sequential_parallel_generatedRoot_290_484 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_17_0 w_21_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 at h
  simp only [Stage_online_eval.sequential_parallel_generatedRoot_290_484.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, h2_0_8, h2_0_9, h19_0, h2_0_10, h2_0_11, h23_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  have e4 := familyGet_cellAt h19_0
  have e5 := familyGet_cellAt h23_0
  subst e0 e1 e2 e3 e4 e5 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `482`: the sums of the families over their 8 members. -/
theorem rootSum482 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {g0 g1 g2 g3 g4 g5 : Fin 8 → ExactMatrix Q0 N 1 6}
    {a0 b0 a1 b1 a2 b2 a3 b3 a4 b4 a5 b5 : ExactMatrix Q0 N 1 6}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    (ha4 : familyGetStatic g4 0 a4) (hb4 : familyGetStatic g4 1 b4)
    (ha5 : familyGetStatic g5 0 a5) (hb5 : familyGetStatic g5 1 b5)
    {final : Blocks Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Blocks Q0 6) ↦
      Stage_online_eval.sequential_parallel_generatedRoot_290_484 Backend.backend model params i0 i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1, current.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5, ()) next) (Int.toNat 6)
      (matrixAdd a0 b0, matrixAdd a1 b1, matrixAdd a2 b2, matrixAdd a3 b3, matrixAdd a4 b4, matrixAdd a5 b5, ()) final) :
    final = (∑ k, g0 k, ∑ k, g1 k, ∑ k, g2 k, ∑ k, g3 k, ∑ k, g4 k, ∑ k, g5 k, ()) := by
  have hloop := iter_sum (fun i ↦ (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()))
    (fun i current next hbody ↦ rootStep482 hbody) hruns
  rw [show Int.toNat 6 = 6 from rfl] at hloop
  obtain ⟨_, ea0⟩ := familyGet_eq (k := 0) ha0
  obtain ⟨_, eb0⟩ := familyGet_eq (k := 1) hb0
  obtain ⟨_, ea1⟩ := familyGet_eq (k := 0) ha1
  obtain ⟨_, eb1⟩ := familyGet_eq (k := 1) hb1
  obtain ⟨_, ea2⟩ := familyGet_eq (k := 0) ha2
  obtain ⟨_, eb2⟩ := familyGet_eq (k := 1) hb2
  obtain ⟨_, ea3⟩ := familyGet_eq (k := 0) ha3
  obtain ⟨_, eb3⟩ := familyGet_eq (k := 1) hb3
  obtain ⟨_, ea4⟩ := familyGet_eq (k := 0) ha4
  obtain ⟨_, eb4⟩ := familyGet_eq (k := 1) hb4
  obtain ⟨_, ea5⟩ := familyGet_eq (k := 0) ha5
  obtain ⟨_, eb5⟩ := familyGet_eq (k := 1) hb5
  rw [hloop]
  simp only [ea0, eb0, ea1, eb1, ea2, eb2, ea3, eb3, ea4, eb4, ea5, eb5, matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 6) g0
  · exact sum_cells (w := 6) g1
  · exact sum_cells (w := 6) g2
  · exact sum_cells (w := 6) g3
  · exact sum_cells (w := 6) g4
  · exact sum_cells (w := 6) g5

theorem rootStep504 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {i : Nat} {cur next : Blocks Q0 6}
    {g0 g1 g2 g3 g4 g5 : Fin 8 → ExactMatrix Q0 N 1 6}
    (h : Stage_online_eval.sequential_parallel_generatedRoot_290_506 Backend.backend model params i0 i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold Stage_online_eval.sequential_parallel_generatedRoot_290_506 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_17_0 w_21_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 at h
  simp only [Stage_online_eval.sequential_parallel_generatedRoot_290_506.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, h2_0_8, h2_0_9, h19_0, h2_0_10, h2_0_11, h23_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  have e4 := familyGet_cellAt h19_0
  have e5 := familyGet_cellAt h23_0
  subst e0 e1 e2 e3 e4 e5 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `504`: the sums of the families over their 8 members. -/
theorem rootSum504 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {g0 g1 g2 g3 g4 g5 : Fin 8 → ExactMatrix Q0 N 1 6}
    {a0 b0 a1 b1 a2 b2 a3 b3 a4 b4 a5 b5 : ExactMatrix Q0 N 1 6}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    (ha4 : familyGetStatic g4 0 a4) (hb4 : familyGetStatic g4 1 b4)
    (ha5 : familyGetStatic g5 0 a5) (hb5 : familyGetStatic g5 1 b5)
    {final : Blocks Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Blocks Q0 6) ↦
      Stage_online_eval.sequential_parallel_generatedRoot_290_506 Backend.backend model params i0 i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1, current.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5, ()) next) (Int.toNat 6)
      (matrixAdd a0 b0, matrixAdd a1 b1, matrixAdd a2 b2, matrixAdd a3 b3, matrixAdd a4 b4, matrixAdd a5 b5, ()) final) :
    final = (∑ k, g0 k, ∑ k, g1 k, ∑ k, g2 k, ∑ k, g3 k, ∑ k, g4 k, ∑ k, g5 k, ()) := by
  have hloop := iter_sum (fun i ↦ (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()))
    (fun i current next hbody ↦ rootStep504 hbody) hruns
  rw [show Int.toNat 6 = 6 from rfl] at hloop
  obtain ⟨_, ea0⟩ := familyGet_eq (k := 0) ha0
  obtain ⟨_, eb0⟩ := familyGet_eq (k := 1) hb0
  obtain ⟨_, ea1⟩ := familyGet_eq (k := 0) ha1
  obtain ⟨_, eb1⟩ := familyGet_eq (k := 1) hb1
  obtain ⟨_, ea2⟩ := familyGet_eq (k := 0) ha2
  obtain ⟨_, eb2⟩ := familyGet_eq (k := 1) hb2
  obtain ⟨_, ea3⟩ := familyGet_eq (k := 0) ha3
  obtain ⟨_, eb3⟩ := familyGet_eq (k := 1) hb3
  obtain ⟨_, ea4⟩ := familyGet_eq (k := 0) ha4
  obtain ⟨_, eb4⟩ := familyGet_eq (k := 1) hb4
  obtain ⟨_, ea5⟩ := familyGet_eq (k := 0) ha5
  obtain ⟨_, eb5⟩ := familyGet_eq (k := 1) hb5
  rw [hloop]
  simp only [ea0, eb0, ea1, eb1, ea2, eb2, ea3, eb3, ea4, eb4, ea5, eb5, matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 6) g0
  · exact sum_cells (w := 6) g1
  · exact sum_cells (w := 6) g2
  · exact sum_cells (w := 6) g3
  · exact sum_cells (w := 6) g4
  · exact sum_cells (w := 6) g5

theorem rootStep526 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {i : Nat} {cur next : Blocks Q0 6}
    {g0 g1 g2 g3 g4 g5 : Fin 8 → ExactMatrix Q0 N 1 6}
    (h : Stage_online_eval.sequential_parallel_generatedRoot_290_528 Backend.backend model params i0 i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold Stage_online_eval.sequential_parallel_generatedRoot_290_528 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_17_0 w_21_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 at h
  simp only [Stage_online_eval.sequential_parallel_generatedRoot_290_528.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, h2_0_8, h2_0_9, h19_0, h2_0_10, h2_0_11, h23_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  have e4 := familyGet_cellAt h19_0
  have e5 := familyGet_cellAt h23_0
  subst e0 e1 e2 e3 e4 e5 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `526`: the sums of the families over their 8 members. -/
theorem rootSum526 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {g0 g1 g2 g3 g4 g5 : Fin 8 → ExactMatrix Q0 N 1 6}
    {a0 b0 a1 b1 a2 b2 a3 b3 a4 b4 a5 b5 : ExactMatrix Q0 N 1 6}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    (ha4 : familyGetStatic g4 0 a4) (hb4 : familyGetStatic g4 1 b4)
    (ha5 : familyGetStatic g5 0 a5) (hb5 : familyGetStatic g5 1 b5)
    {final : Blocks Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Blocks Q0 6) ↦
      Stage_online_eval.sequential_parallel_generatedRoot_290_528 Backend.backend model params i0 i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1, current.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5, ()) next) (Int.toNat 6)
      (matrixAdd a0 b0, matrixAdd a1 b1, matrixAdd a2 b2, matrixAdd a3 b3, matrixAdd a4 b4, matrixAdd a5 b5, ()) final) :
    final = (∑ k, g0 k, ∑ k, g1 k, ∑ k, g2 k, ∑ k, g3 k, ∑ k, g4 k, ∑ k, g5 k, ()) := by
  have hloop := iter_sum (fun i ↦ (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()))
    (fun i current next hbody ↦ rootStep526 hbody) hruns
  rw [show Int.toNat 6 = 6 from rfl] at hloop
  obtain ⟨_, ea0⟩ := familyGet_eq (k := 0) ha0
  obtain ⟨_, eb0⟩ := familyGet_eq (k := 1) hb0
  obtain ⟨_, ea1⟩ := familyGet_eq (k := 0) ha1
  obtain ⟨_, eb1⟩ := familyGet_eq (k := 1) hb1
  obtain ⟨_, ea2⟩ := familyGet_eq (k := 0) ha2
  obtain ⟨_, eb2⟩ := familyGet_eq (k := 1) hb2
  obtain ⟨_, ea3⟩ := familyGet_eq (k := 0) ha3
  obtain ⟨_, eb3⟩ := familyGet_eq (k := 1) hb3
  obtain ⟨_, ea4⟩ := familyGet_eq (k := 0) ha4
  obtain ⟨_, eb4⟩ := familyGet_eq (k := 1) hb4
  obtain ⟨_, ea5⟩ := familyGet_eq (k := 0) ha5
  obtain ⟨_, eb5⟩ := familyGet_eq (k := 1) hb5
  rw [hloop]
  simp only [ea0, eb0, ea1, eb1, ea2, eb2, ea3, eb3, ea4, eb4, ea5, eb5, matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 6) g0
  · exact sum_cells (w := 6) g1
  · exact sum_cells (w := 6) g2
  · exact sum_cells (w := 6) g3
  · exact sum_cells (w := 6) g4
  · exact sum_cells (w := 6) g5

theorem rootStep579 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {i : Nat} {cur next : Blocks (R i0) 1}
    {g0 g1 g2 g3 g4 g5 : Fin 160 → ExactMatrix (R i0) N 1 1}
    (h : Stage_online_eval.sequential_parallel_generatedRoot_290_581 Backend.backend model params i0 i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold Stage_online_eval.sequential_parallel_generatedRoot_290_581 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_17_0 w_21_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 at h
  simp only [Stage_online_eval.sequential_parallel_generatedRoot_290_581.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, h2_0_8, h2_0_9, h19_0, h2_0_10, h2_0_11, h23_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  have e4 := familyGet_cellAt h19_0
  have e5 := familyGet_cellAt h23_0
  subst e0 e1 e2 e3 e4 e5 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `579`: the sums of the families over their 160 members. -/
theorem rootSum579 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {g0 g1 g2 g3 g4 g5 : Fin 160 → ExactMatrix (R i0) N 1 1}
    {a0 b0 a1 b1 a2 b2 a3 b3 a4 b4 a5 b5 : ExactMatrix (R i0) N 1 1}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    (ha4 : familyGetStatic g4 0 a4) (hb4 : familyGetStatic g4 1 b4)
    (ha5 : familyGetStatic g5 0 a5) (hb5 : familyGetStatic g5 1 b5)
    {final : Blocks (R i0) 1}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Blocks (R i0) 1) ↦
      Stage_online_eval.sequential_parallel_generatedRoot_290_581 Backend.backend model params i0 i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1, current.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5, ()) next) (Int.toNat 158)
      (matrixAdd a0 b0, matrixAdd a1 b1, matrixAdd a2 b2, matrixAdd a3 b3, matrixAdd a4 b4, matrixAdd a5 b5, ()) final) :
    final = (∑ k, g0 k, ∑ k, g1 k, ∑ k, g2 k, ∑ k, g3 k, ∑ k, g4 k, ∑ k, g5 k, ()) := by
  have hloop := iter_sum (fun i ↦ (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()))
    (fun i current next hbody ↦ rootStep579 hbody) hruns
  rw [show Int.toNat 158 = 158 from rfl] at hloop
  obtain ⟨_, ea0⟩ := familyGet_eq (k := 0) ha0
  obtain ⟨_, eb0⟩ := familyGet_eq (k := 1) hb0
  obtain ⟨_, ea1⟩ := familyGet_eq (k := 0) ha1
  obtain ⟨_, eb1⟩ := familyGet_eq (k := 1) hb1
  obtain ⟨_, ea2⟩ := familyGet_eq (k := 0) ha2
  obtain ⟨_, eb2⟩ := familyGet_eq (k := 1) hb2
  obtain ⟨_, ea3⟩ := familyGet_eq (k := 0) ha3
  obtain ⟨_, eb3⟩ := familyGet_eq (k := 1) hb3
  obtain ⟨_, ea4⟩ := familyGet_eq (k := 0) ha4
  obtain ⟨_, eb4⟩ := familyGet_eq (k := 1) hb4
  obtain ⟨_, ea5⟩ := familyGet_eq (k := 0) ha5
  obtain ⟨_, eb5⟩ := familyGet_eq (k := 1) hb5
  rw [hloop]
  simp only [ea0, eb0, ea1, eb1, ea2, eb2, ea3, eb3, ea4, eb4, ea5, eb5, matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 158) g0
  · exact sum_cells (w := 158) g1
  · exact sum_cells (w := 158) g2
  · exact sum_cells (w := 158) g3
  · exact sum_cells (w := 158) g4
  · exact sum_cells (w := 158) g5

theorem rootStep634 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {i : Nat} {cur next : Blocks (R i0) 1}
    {g0 g1 g2 g3 g4 g5 : Fin 160 → ExactMatrix (R i0) N 1 1}
    (h : Stage_online_eval.sequential_parallel_generatedRoot_290_636 Backend.backend model params i0 i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold Stage_online_eval.sequential_parallel_generatedRoot_290_636 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_17_0 w_21_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 at h
  simp only [Stage_online_eval.sequential_parallel_generatedRoot_290_636.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, h2_0_8, h2_0_9, h19_0, h2_0_10, h2_0_11, h23_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  have e4 := familyGet_cellAt h19_0
  have e5 := familyGet_cellAt h23_0
  subst e0 e1 e2 e3 e4 e5 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `634`: the sums of the families over their 160 members. -/
theorem rootSum634 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {g0 g1 g2 g3 g4 g5 : Fin 160 → ExactMatrix (R i0) N 1 1}
    {a0 b0 a1 b1 a2 b2 a3 b3 a4 b4 a5 b5 : ExactMatrix (R i0) N 1 1}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    (ha4 : familyGetStatic g4 0 a4) (hb4 : familyGetStatic g4 1 b4)
    (ha5 : familyGetStatic g5 0 a5) (hb5 : familyGetStatic g5 1 b5)
    {final : Blocks (R i0) 1}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Blocks (R i0) 1) ↦
      Stage_online_eval.sequential_parallel_generatedRoot_290_636 Backend.backend model params i0 i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1, current.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5, ()) next) (Int.toNat 158)
      (matrixAdd a0 b0, matrixAdd a1 b1, matrixAdd a2 b2, matrixAdd a3 b3, matrixAdd a4 b4, matrixAdd a5 b5, ()) final) :
    final = (∑ k, g0 k, ∑ k, g1 k, ∑ k, g2 k, ∑ k, g3 k, ∑ k, g4 k, ∑ k, g5 k, ()) := by
  have hloop := iter_sum (fun i ↦ (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()))
    (fun i current next hbody ↦ rootStep634 hbody) hruns
  rw [show Int.toNat 158 = 158 from rfl] at hloop
  obtain ⟨_, ea0⟩ := familyGet_eq (k := 0) ha0
  obtain ⟨_, eb0⟩ := familyGet_eq (k := 1) hb0
  obtain ⟨_, ea1⟩ := familyGet_eq (k := 0) ha1
  obtain ⟨_, eb1⟩ := familyGet_eq (k := 1) hb1
  obtain ⟨_, ea2⟩ := familyGet_eq (k := 0) ha2
  obtain ⟨_, eb2⟩ := familyGet_eq (k := 1) hb2
  obtain ⟨_, ea3⟩ := familyGet_eq (k := 0) ha3
  obtain ⟨_, eb3⟩ := familyGet_eq (k := 1) hb3
  obtain ⟨_, ea4⟩ := familyGet_eq (k := 0) ha4
  obtain ⟨_, eb4⟩ := familyGet_eq (k := 1) hb4
  obtain ⟨_, ea5⟩ := familyGet_eq (k := 0) ha5
  obtain ⟨_, eb5⟩ := familyGet_eq (k := 1) hb5
  rw [hloop]
  simp only [ea0, eb0, ea1, eb1, ea2, eb2, ea3, eb3, ea4, eb4, ea5, eb5, matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 158) g0
  · exact sum_cells (w := 158) g1
  · exact sum_cells (w := 158) g2
  · exact sum_cells (w := 158) g3
  · exact sum_cells (w := 158) g4
  · exact sum_cells (w := 158) g5

theorem rootStep688 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {i : Nat} {cur next : Blocks (R i0) 1}
    {g0 g1 g2 g3 g4 g5 : Fin 160 → ExactMatrix (R i0) N 1 1}
    (h : Stage_online_eval.sequential_parallel_generatedRoot_290_690 Backend.backend model params i0 i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold Stage_online_eval.sequential_parallel_generatedRoot_290_690 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_17_0 w_21_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 at h
  simp only [Stage_online_eval.sequential_parallel_generatedRoot_290_690.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, h2_0_8, h2_0_9, h19_0, h2_0_10, h2_0_11, h23_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  have e4 := familyGet_cellAt h19_0
  have e5 := familyGet_cellAt h23_0
  subst e0 e1 e2 e3 e4 e5 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `688`: the sums of the families over their 160 members. -/
theorem rootSum688 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {g0 g1 g2 g3 g4 g5 : Fin 160 → ExactMatrix (R i0) N 1 1}
    {a0 b0 a1 b1 a2 b2 a3 b3 a4 b4 a5 b5 : ExactMatrix (R i0) N 1 1}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    (ha4 : familyGetStatic g4 0 a4) (hb4 : familyGetStatic g4 1 b4)
    (ha5 : familyGetStatic g5 0 a5) (hb5 : familyGetStatic g5 1 b5)
    {final : Blocks (R i0) 1}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Blocks (R i0) 1) ↦
      Stage_online_eval.sequential_parallel_generatedRoot_290_690 Backend.backend model params i0 i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1, current.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5, ()) next) (Int.toNat 158)
      (matrixAdd a0 b0, matrixAdd a1 b1, matrixAdd a2 b2, matrixAdd a3 b3, matrixAdd a4 b4, matrixAdd a5 b5, ()) final) :
    final = (∑ k, g0 k, ∑ k, g1 k, ∑ k, g2 k, ∑ k, g3 k, ∑ k, g4 k, ∑ k, g5 k, ()) := by
  have hloop := iter_sum (fun i ↦ (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()))
    (fun i current next hbody ↦ rootStep688 hbody) hruns
  rw [show Int.toNat 158 = 158 from rfl] at hloop
  obtain ⟨_, ea0⟩ := familyGet_eq (k := 0) ha0
  obtain ⟨_, eb0⟩ := familyGet_eq (k := 1) hb0
  obtain ⟨_, ea1⟩ := familyGet_eq (k := 0) ha1
  obtain ⟨_, eb1⟩ := familyGet_eq (k := 1) hb1
  obtain ⟨_, ea2⟩ := familyGet_eq (k := 0) ha2
  obtain ⟨_, eb2⟩ := familyGet_eq (k := 1) hb2
  obtain ⟨_, ea3⟩ := familyGet_eq (k := 0) ha3
  obtain ⟨_, eb3⟩ := familyGet_eq (k := 1) hb3
  obtain ⟨_, ea4⟩ := familyGet_eq (k := 0) ha4
  obtain ⟨_, eb4⟩ := familyGet_eq (k := 1) hb4
  obtain ⟨_, ea5⟩ := familyGet_eq (k := 0) ha5
  obtain ⟨_, eb5⟩ := familyGet_eq (k := 1) hb5
  rw [hloop]
  simp only [ea0, eb0, ea1, eb1, ea2, eb2, ea3, eb3, ea4, eb4, ea5, eb5, matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 158) g0
  · exact sum_cells (w := 158) g1
  · exact sum_cells (w := 158) g2
  · exact sum_cells (w := 158) g3
  · exact sum_cells (w := 158) g4
  · exact sum_cells (w := 158) g5

theorem rootStep742 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {i : Nat} {cur next : Blocks (R i0) 1}
    {g0 g1 g2 g3 g4 g5 : Fin 160 → ExactMatrix (R i0) N 1 1}
    (h : Stage_online_eval.sequential_parallel_generatedRoot_290_744 Backend.backend model params i0 i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold Stage_online_eval.sequential_parallel_generatedRoot_290_744 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_17_0 w_21_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 at h
  simp only [Stage_online_eval.sequential_parallel_generatedRoot_290_744.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, h2_0_8, h2_0_9, h19_0, h2_0_10, h2_0_11, h23_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  have e4 := familyGet_cellAt h19_0
  have e5 := familyGet_cellAt h23_0
  subst e0 e1 e2 e3 e4 e5 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `742`: the sums of the families over their 160 members. -/
theorem rootSum742 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {g0 g1 g2 g3 g4 g5 : Fin 160 → ExactMatrix (R i0) N 1 1}
    {a0 b0 a1 b1 a2 b2 a3 b3 a4 b4 a5 b5 : ExactMatrix (R i0) N 1 1}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    (ha4 : familyGetStatic g4 0 a4) (hb4 : familyGetStatic g4 1 b4)
    (ha5 : familyGetStatic g5 0 a5) (hb5 : familyGetStatic g5 1 b5)
    {final : Blocks (R i0) 1}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Blocks (R i0) 1) ↦
      Stage_online_eval.sequential_parallel_generatedRoot_290_744 Backend.backend model params i0 i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1, current.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5, ()) next) (Int.toNat 158)
      (matrixAdd a0 b0, matrixAdd a1 b1, matrixAdd a2 b2, matrixAdd a3 b3, matrixAdd a4 b4, matrixAdd a5 b5, ()) final) :
    final = (∑ k, g0 k, ∑ k, g1 k, ∑ k, g2 k, ∑ k, g3 k, ∑ k, g4 k, ∑ k, g5 k, ()) := by
  have hloop := iter_sum (fun i ↦ (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()))
    (fun i current next hbody ↦ rootStep742 hbody) hruns
  rw [show Int.toNat 158 = 158 from rfl] at hloop
  obtain ⟨_, ea0⟩ := familyGet_eq (k := 0) ha0
  obtain ⟨_, eb0⟩ := familyGet_eq (k := 1) hb0
  obtain ⟨_, ea1⟩ := familyGet_eq (k := 0) ha1
  obtain ⟨_, eb1⟩ := familyGet_eq (k := 1) hb1
  obtain ⟨_, ea2⟩ := familyGet_eq (k := 0) ha2
  obtain ⟨_, eb2⟩ := familyGet_eq (k := 1) hb2
  obtain ⟨_, ea3⟩ := familyGet_eq (k := 0) ha3
  obtain ⟨_, eb3⟩ := familyGet_eq (k := 1) hb3
  obtain ⟨_, ea4⟩ := familyGet_eq (k := 0) ha4
  obtain ⟨_, eb4⟩ := familyGet_eq (k := 1) hb4
  obtain ⟨_, ea5⟩ := familyGet_eq (k := 0) ha5
  obtain ⟨_, eb5⟩ := familyGet_eq (k := 1) hb5
  rw [hloop]
  simp only [ea0, eb0, ea1, eb1, ea2, eb2, ea3, eb3, ea4, eb4, ea5, eb5, matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 158) g0
  · exact sum_cells (w := 158) g1
  · exact sum_cells (w := 158) g2
  · exact sum_cells (w := 158) g3
  · exact sum_cells (w := 158) g4
  · exact sum_cells (w := 158) g5

theorem rootStep796 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {i : Nat} {cur next : Blocks (R i0) 1}
    {g0 g1 g2 g3 g4 g5 : Fin 160 → ExactMatrix (R i0) N 1 1}
    (h : Stage_online_eval.sequential_parallel_generatedRoot_290_798 Backend.backend model params i0 i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold Stage_online_eval.sequential_parallel_generatedRoot_290_798 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_17_0 w_21_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 at h
  simp only [Stage_online_eval.sequential_parallel_generatedRoot_290_798.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, h2_0_8, h2_0_9, h19_0, h2_0_10, h2_0_11, h23_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  have e4 := familyGet_cellAt h19_0
  have e5 := familyGet_cellAt h23_0
  subst e0 e1 e2 e3 e4 e5 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `796`: the sums of the families over their 160 members. -/
theorem rootSum796 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {g0 g1 g2 g3 g4 g5 : Fin 160 → ExactMatrix (R i0) N 1 1}
    {a0 b0 a1 b1 a2 b2 a3 b3 a4 b4 a5 b5 : ExactMatrix (R i0) N 1 1}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    (ha4 : familyGetStatic g4 0 a4) (hb4 : familyGetStatic g4 1 b4)
    (ha5 : familyGetStatic g5 0 a5) (hb5 : familyGetStatic g5 1 b5)
    {final : Blocks (R i0) 1}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Blocks (R i0) 1) ↦
      Stage_online_eval.sequential_parallel_generatedRoot_290_798 Backend.backend model params i0 i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1, current.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5, ()) next) (Int.toNat 158)
      (matrixAdd a0 b0, matrixAdd a1 b1, matrixAdd a2 b2, matrixAdd a3 b3, matrixAdd a4 b4, matrixAdd a5 b5, ()) final) :
    final = (∑ k, g0 k, ∑ k, g1 k, ∑ k, g2 k, ∑ k, g3 k, ∑ k, g4 k, ∑ k, g5 k, ()) := by
  have hloop := iter_sum (fun i ↦ (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()))
    (fun i current next hbody ↦ rootStep796 hbody) hruns
  rw [show Int.toNat 158 = 158 from rfl] at hloop
  obtain ⟨_, ea0⟩ := familyGet_eq (k := 0) ha0
  obtain ⟨_, eb0⟩ := familyGet_eq (k := 1) hb0
  obtain ⟨_, ea1⟩ := familyGet_eq (k := 0) ha1
  obtain ⟨_, eb1⟩ := familyGet_eq (k := 1) hb1
  obtain ⟨_, ea2⟩ := familyGet_eq (k := 0) ha2
  obtain ⟨_, eb2⟩ := familyGet_eq (k := 1) hb2
  obtain ⟨_, ea3⟩ := familyGet_eq (k := 0) ha3
  obtain ⟨_, eb3⟩ := familyGet_eq (k := 1) hb3
  obtain ⟨_, ea4⟩ := familyGet_eq (k := 0) ha4
  obtain ⟨_, eb4⟩ := familyGet_eq (k := 1) hb4
  obtain ⟨_, ea5⟩ := familyGet_eq (k := 0) ha5
  obtain ⟨_, eb5⟩ := familyGet_eq (k := 1) hb5
  rw [hloop]
  simp only [ea0, eb0, ea1, eb1, ea2, eb2, ea3, eb3, ea4, eb4, ea5, eb5, matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 158) g0
  · exact sum_cells (w := 158) g1
  · exact sum_cells (w := 158) g2
  · exact sum_cells (w := 158) g3
  · exact sum_cells (w := 158) g4
  · exact sum_cells (w := 158) g5

theorem rootStep850 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {i : Nat} {cur next : Blocks (R i0) 1}
    {g0 g1 g2 g3 g4 g5 : Fin 160 → ExactMatrix (R i0) N 1 1}
    (h : Stage_online_eval.sequential_parallel_generatedRoot_290_852 Backend.backend model params i0 i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold Stage_online_eval.sequential_parallel_generatedRoot_290_852 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_17_0 w_21_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 at h
  simp only [Stage_online_eval.sequential_parallel_generatedRoot_290_852.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, h2_0_8, h2_0_9, h19_0, h2_0_10, h2_0_11, h23_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  have e4 := familyGet_cellAt h19_0
  have e5 := familyGet_cellAt h23_0
  subst e0 e1 e2 e3 e4 e5 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `850`: the sums of the families over their 160 members. -/
theorem rootSum850 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {g0 g1 g2 g3 g4 g5 : Fin 160 → ExactMatrix (R i0) N 1 1}
    {a0 b0 a1 b1 a2 b2 a3 b3 a4 b4 a5 b5 : ExactMatrix (R i0) N 1 1}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    (ha4 : familyGetStatic g4 0 a4) (hb4 : familyGetStatic g4 1 b4)
    (ha5 : familyGetStatic g5 0 a5) (hb5 : familyGetStatic g5 1 b5)
    {final : Blocks (R i0) 1}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Blocks (R i0) 1) ↦
      Stage_online_eval.sequential_parallel_generatedRoot_290_852 Backend.backend model params i0 i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1, current.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5, ()) next) (Int.toNat 158)
      (matrixAdd a0 b0, matrixAdd a1 b1, matrixAdd a2 b2, matrixAdd a3 b3, matrixAdd a4 b4, matrixAdd a5 b5, ()) final) :
    final = (∑ k, g0 k, ∑ k, g1 k, ∑ k, g2 k, ∑ k, g3 k, ∑ k, g4 k, ∑ k, g5 k, ()) := by
  have hloop := iter_sum (fun i ↦ (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()))
    (fun i current next hbody ↦ rootStep850 hbody) hruns
  rw [show Int.toNat 158 = 158 from rfl] at hloop
  obtain ⟨_, ea0⟩ := familyGet_eq (k := 0) ha0
  obtain ⟨_, eb0⟩ := familyGet_eq (k := 1) hb0
  obtain ⟨_, ea1⟩ := familyGet_eq (k := 0) ha1
  obtain ⟨_, eb1⟩ := familyGet_eq (k := 1) hb1
  obtain ⟨_, ea2⟩ := familyGet_eq (k := 0) ha2
  obtain ⟨_, eb2⟩ := familyGet_eq (k := 1) hb2
  obtain ⟨_, ea3⟩ := familyGet_eq (k := 0) ha3
  obtain ⟨_, eb3⟩ := familyGet_eq (k := 1) hb3
  obtain ⟨_, ea4⟩ := familyGet_eq (k := 0) ha4
  obtain ⟨_, eb4⟩ := familyGet_eq (k := 1) hb4
  obtain ⟨_, ea5⟩ := familyGet_eq (k := 0) ha5
  obtain ⟨_, eb5⟩ := familyGet_eq (k := 1) hb5
  rw [hloop]
  simp only [ea0, eb0, ea1, eb1, ea2, eb2, ea3, eb3, ea4, eb4, ea5, eb5, matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 158) g0
  · exact sum_cells (w := 158) g1
  · exact sum_cells (w := 158) g2
  · exact sum_cells (w := 158) g3
  · exact sum_cells (w := 158) g4
  · exact sum_cells (w := 158) g5

end RefreshTopProof.Part_I
