import TopInst
import TopLemmas

/-! The member sums of the online root (written by `generator/gen_root_sums.py`). -/

namespace RefreshTopProof.Part_I

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

theorem prerootStep166 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {i : Nat} {cur next : Keys Q0 6}
    {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    (h : Stage_preprocess.sequential_parallel_generatedRoot_407_166 Backend.backend model params i0 i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold Stage_preprocess.sequential_parallel_generatedRoot_407_166 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [Stage_preprocess.sequential_parallel_generatedRoot_407_166.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `166`: the sums of the families over their 8 members. -/
theorem prerootSum166 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    {a0 b0 a1 b1 a2 b2 a3 b3 : ExactMatrix Q0 N 1 6}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    {final : Keys Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q0 6) ↦
      Stage_preprocess.sequential_parallel_generatedRoot_407_166 Backend.backend model params i0 i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, g0, g1, g2, g3, ()) next) (Int.toNat 6)
      (matrixAdd a0 b0, matrixAdd a1 b1, matrixAdd a2 b2, matrixAdd a3 b3, ()) final) :
    final = (∑ k, g0 k, ∑ k, g1 k, ∑ k, g2 k, ∑ k, g3 k, ()) := by
  have hloop := iter_sum (fun i ↦ (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()))
    (fun i current next hbody ↦ prerootStep166 hbody) hruns
  rw [show Int.toNat 6 = 6 from rfl] at hloop
  obtain ⟨_, ea0⟩ := familyGet_eq (k := 0) ha0
  obtain ⟨_, eb0⟩ := familyGet_eq (k := 1) hb0
  obtain ⟨_, ea1⟩ := familyGet_eq (k := 0) ha1
  obtain ⟨_, eb1⟩ := familyGet_eq (k := 1) hb1
  obtain ⟨_, ea2⟩ := familyGet_eq (k := 0) ha2
  obtain ⟨_, eb2⟩ := familyGet_eq (k := 1) hb2
  obtain ⟨_, ea3⟩ := familyGet_eq (k := 0) ha3
  obtain ⟨_, eb3⟩ := familyGet_eq (k := 1) hb3
  rw [hloop]
  simp only [ea0, eb0, ea1, eb1, ea2, eb2, ea3, eb3, matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 6) g0
  · exact sum_cells (w := 6) g1
  · exact sum_cells (w := 6) g2
  · exact sum_cells (w := 6) g3

theorem prerootStep233 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {i : Nat} {cur next : Keys Q0 6}
    {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    (h : Stage_preprocess.sequential_parallel_generatedRoot_407_235 Backend.backend model params i0 i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold Stage_preprocess.sequential_parallel_generatedRoot_407_235 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [Stage_preprocess.sequential_parallel_generatedRoot_407_235.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `233`: the sums of the families over their 8 members. -/
theorem prerootSum233 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    {a0 b0 a1 b1 a2 b2 a3 b3 : ExactMatrix Q0 N 1 6}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    {final : Keys Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q0 6) ↦
      Stage_preprocess.sequential_parallel_generatedRoot_407_235 Backend.backend model params i0 i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, g0, g1, g2, g3, ()) next) (Int.toNat 6)
      (matrixAdd a0 b0, matrixAdd a1 b1, matrixAdd a2 b2, matrixAdd a3 b3, ()) final) :
    final = (∑ k, g0 k, ∑ k, g1 k, ∑ k, g2 k, ∑ k, g3 k, ()) := by
  have hloop := iter_sum (fun i ↦ (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()))
    (fun i current next hbody ↦ prerootStep233 hbody) hruns
  rw [show Int.toNat 6 = 6 from rfl] at hloop
  obtain ⟨_, ea0⟩ := familyGet_eq (k := 0) ha0
  obtain ⟨_, eb0⟩ := familyGet_eq (k := 1) hb0
  obtain ⟨_, ea1⟩ := familyGet_eq (k := 0) ha1
  obtain ⟨_, eb1⟩ := familyGet_eq (k := 1) hb1
  obtain ⟨_, ea2⟩ := familyGet_eq (k := 0) ha2
  obtain ⟨_, eb2⟩ := familyGet_eq (k := 1) hb2
  obtain ⟨_, ea3⟩ := familyGet_eq (k := 0) ha3
  obtain ⟨_, eb3⟩ := familyGet_eq (k := 1) hb3
  rw [hloop]
  simp only [ea0, eb0, ea1, eb1, ea2, eb2, ea3, eb3, matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 6) g0
  · exact sum_cells (w := 6) g1
  · exact sum_cells (w := 6) g2
  · exact sum_cells (w := 6) g3

theorem prerootStep252 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {i : Nat} {cur next : Keys Q0 6}
    {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    (h : Stage_preprocess.sequential_parallel_generatedRoot_407_254 Backend.backend model params i0 i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold Stage_preprocess.sequential_parallel_generatedRoot_407_254 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [Stage_preprocess.sequential_parallel_generatedRoot_407_254.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `252`: the sums of the families over their 8 members. -/
theorem prerootSum252 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    {a0 b0 a1 b1 a2 b2 a3 b3 : ExactMatrix Q0 N 1 6}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    {final : Keys Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q0 6) ↦
      Stage_preprocess.sequential_parallel_generatedRoot_407_254 Backend.backend model params i0 i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, g0, g1, g2, g3, ()) next) (Int.toNat 6)
      (matrixAdd a0 b0, matrixAdd a1 b1, matrixAdd a2 b2, matrixAdd a3 b3, ()) final) :
    final = (∑ k, g0 k, ∑ k, g1 k, ∑ k, g2 k, ∑ k, g3 k, ()) := by
  have hloop := iter_sum (fun i ↦ (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()))
    (fun i current next hbody ↦ prerootStep252 hbody) hruns
  rw [show Int.toNat 6 = 6 from rfl] at hloop
  obtain ⟨_, ea0⟩ := familyGet_eq (k := 0) ha0
  obtain ⟨_, eb0⟩ := familyGet_eq (k := 1) hb0
  obtain ⟨_, ea1⟩ := familyGet_eq (k := 0) ha1
  obtain ⟨_, eb1⟩ := familyGet_eq (k := 1) hb1
  obtain ⟨_, ea2⟩ := familyGet_eq (k := 0) ha2
  obtain ⟨_, eb2⟩ := familyGet_eq (k := 1) hb2
  obtain ⟨_, ea3⟩ := familyGet_eq (k := 0) ha3
  obtain ⟨_, eb3⟩ := familyGet_eq (k := 1) hb3
  rw [hloop]
  simp only [ea0, eb0, ea1, eb1, ea2, eb2, ea3, eb3, matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 6) g0
  · exact sum_cells (w := 6) g1
  · exact sum_cells (w := 6) g2
  · exact sum_cells (w := 6) g3

theorem prerootStep268 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {i : Nat} {cur next : Keys Q0 6}
    {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    (h : Stage_preprocess.sequential_parallel_generatedRoot_407_270 Backend.backend model params i0 i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold Stage_preprocess.sequential_parallel_generatedRoot_407_270 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [Stage_preprocess.sequential_parallel_generatedRoot_407_270.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `268`: the sums of the families over their 8 members. -/
theorem prerootSum268 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    {a0 b0 a1 b1 a2 b2 a3 b3 : ExactMatrix Q0 N 1 6}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    {final : Keys Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q0 6) ↦
      Stage_preprocess.sequential_parallel_generatedRoot_407_270 Backend.backend model params i0 i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, g0, g1, g2, g3, ()) next) (Int.toNat 6)
      (matrixAdd a0 b0, matrixAdd a1 b1, matrixAdd a2 b2, matrixAdd a3 b3, ()) final) :
    final = (∑ k, g0 k, ∑ k, g1 k, ∑ k, g2 k, ∑ k, g3 k, ()) := by
  have hloop := iter_sum (fun i ↦ (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()))
    (fun i current next hbody ↦ prerootStep268 hbody) hruns
  rw [show Int.toNat 6 = 6 from rfl] at hloop
  obtain ⟨_, ea0⟩ := familyGet_eq (k := 0) ha0
  obtain ⟨_, eb0⟩ := familyGet_eq (k := 1) hb0
  obtain ⟨_, ea1⟩ := familyGet_eq (k := 0) ha1
  obtain ⟨_, eb1⟩ := familyGet_eq (k := 1) hb1
  obtain ⟨_, ea2⟩ := familyGet_eq (k := 0) ha2
  obtain ⟨_, eb2⟩ := familyGet_eq (k := 1) hb2
  obtain ⟨_, ea3⟩ := familyGet_eq (k := 0) ha3
  obtain ⟨_, eb3⟩ := familyGet_eq (k := 1) hb3
  rw [hloop]
  simp only [ea0, eb0, ea1, eb1, ea2, eb2, ea3, eb3, matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 6) g0
  · exact sum_cells (w := 6) g1
  · exact sum_cells (w := 6) g2
  · exact sum_cells (w := 6) g3

theorem prerootStep284 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {i : Nat} {cur next : Keys Q0 6}
    {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    (h : Stage_preprocess.sequential_parallel_generatedRoot_407_286 Backend.backend model params i0 i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold Stage_preprocess.sequential_parallel_generatedRoot_407_286 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [Stage_preprocess.sequential_parallel_generatedRoot_407_286.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `284`: the sums of the families over their 8 members. -/
theorem prerootSum284 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    {a0 b0 a1 b1 a2 b2 a3 b3 : ExactMatrix Q0 N 1 6}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    {final : Keys Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q0 6) ↦
      Stage_preprocess.sequential_parallel_generatedRoot_407_286 Backend.backend model params i0 i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, g0, g1, g2, g3, ()) next) (Int.toNat 6)
      (matrixAdd a0 b0, matrixAdd a1 b1, matrixAdd a2 b2, matrixAdd a3 b3, ()) final) :
    final = (∑ k, g0 k, ∑ k, g1 k, ∑ k, g2 k, ∑ k, g3 k, ()) := by
  have hloop := iter_sum (fun i ↦ (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()))
    (fun i current next hbody ↦ prerootStep284 hbody) hruns
  rw [show Int.toNat 6 = 6 from rfl] at hloop
  obtain ⟨_, ea0⟩ := familyGet_eq (k := 0) ha0
  obtain ⟨_, eb0⟩ := familyGet_eq (k := 1) hb0
  obtain ⟨_, ea1⟩ := familyGet_eq (k := 0) ha1
  obtain ⟨_, eb1⟩ := familyGet_eq (k := 1) hb1
  obtain ⟨_, ea2⟩ := familyGet_eq (k := 0) ha2
  obtain ⟨_, eb2⟩ := familyGet_eq (k := 1) hb2
  obtain ⟨_, ea3⟩ := familyGet_eq (k := 0) ha3
  obtain ⟨_, eb3⟩ := familyGet_eq (k := 1) hb3
  rw [hloop]
  simp only [ea0, eb0, ea1, eb1, ea2, eb2, ea3, eb3, matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 6) g0
  · exact sum_cells (w := 6) g1
  · exact sum_cells (w := 6) g2
  · exact sum_cells (w := 6) g3

theorem prerootStep300 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {i : Nat} {cur next : Keys Q0 6}
    {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    (h : Stage_preprocess.sequential_parallel_generatedRoot_407_302 Backend.backend model params i0 i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold Stage_preprocess.sequential_parallel_generatedRoot_407_302 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [Stage_preprocess.sequential_parallel_generatedRoot_407_302.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `300`: the sums of the families over their 8 members. -/
theorem prerootSum300 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    {a0 b0 a1 b1 a2 b2 a3 b3 : ExactMatrix Q0 N 1 6}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    {final : Keys Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q0 6) ↦
      Stage_preprocess.sequential_parallel_generatedRoot_407_302 Backend.backend model params i0 i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, g0, g1, g2, g3, ()) next) (Int.toNat 6)
      (matrixAdd a0 b0, matrixAdd a1 b1, matrixAdd a2 b2, matrixAdd a3 b3, ()) final) :
    final = (∑ k, g0 k, ∑ k, g1 k, ∑ k, g2 k, ∑ k, g3 k, ()) := by
  have hloop := iter_sum (fun i ↦ (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()))
    (fun i current next hbody ↦ prerootStep300 hbody) hruns
  rw [show Int.toNat 6 = 6 from rfl] at hloop
  obtain ⟨_, ea0⟩ := familyGet_eq (k := 0) ha0
  obtain ⟨_, eb0⟩ := familyGet_eq (k := 1) hb0
  obtain ⟨_, ea1⟩ := familyGet_eq (k := 0) ha1
  obtain ⟨_, eb1⟩ := familyGet_eq (k := 1) hb1
  obtain ⟨_, ea2⟩ := familyGet_eq (k := 0) ha2
  obtain ⟨_, eb2⟩ := familyGet_eq (k := 1) hb2
  obtain ⟨_, ea3⟩ := familyGet_eq (k := 0) ha3
  obtain ⟨_, eb3⟩ := familyGet_eq (k := 1) hb3
  rw [hloop]
  simp only [ea0, eb0, ea1, eb1, ea2, eb2, ea3, eb3, matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 6) g0
  · exact sum_cells (w := 6) g1
  · exact sum_cells (w := 6) g2
  · exact sum_cells (w := 6) g3

theorem prerootStep316 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {i : Nat} {cur next : Keys Q0 6}
    {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    (h : Stage_preprocess.sequential_parallel_generatedRoot_407_318 Backend.backend model params i0 i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold Stage_preprocess.sequential_parallel_generatedRoot_407_318 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [Stage_preprocess.sequential_parallel_generatedRoot_407_318.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `316`: the sums of the families over their 8 members. -/
theorem prerootSum316 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    {a0 b0 a1 b1 a2 b2 a3 b3 : ExactMatrix Q0 N 1 6}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    {final : Keys Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q0 6) ↦
      Stage_preprocess.sequential_parallel_generatedRoot_407_318 Backend.backend model params i0 i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, g0, g1, g2, g3, ()) next) (Int.toNat 6)
      (matrixAdd a0 b0, matrixAdd a1 b1, matrixAdd a2 b2, matrixAdd a3 b3, ()) final) :
    final = (∑ k, g0 k, ∑ k, g1 k, ∑ k, g2 k, ∑ k, g3 k, ()) := by
  have hloop := iter_sum (fun i ↦ (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()))
    (fun i current next hbody ↦ prerootStep316 hbody) hruns
  rw [show Int.toNat 6 = 6 from rfl] at hloop
  obtain ⟨_, ea0⟩ := familyGet_eq (k := 0) ha0
  obtain ⟨_, eb0⟩ := familyGet_eq (k := 1) hb0
  obtain ⟨_, ea1⟩ := familyGet_eq (k := 0) ha1
  obtain ⟨_, eb1⟩ := familyGet_eq (k := 1) hb1
  obtain ⟨_, ea2⟩ := familyGet_eq (k := 0) ha2
  obtain ⟨_, eb2⟩ := familyGet_eq (k := 1) hb2
  obtain ⟨_, ea3⟩ := familyGet_eq (k := 0) ha3
  obtain ⟨_, eb3⟩ := familyGet_eq (k := 1) hb3
  rw [hloop]
  simp only [ea0, eb0, ea1, eb1, ea2, eb2, ea3, eb3, matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 6) g0
  · exact sum_cells (w := 6) g1
  · exact sum_cells (w := 6) g2
  · exact sum_cells (w := 6) g3

theorem prerootStep336 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {i : Nat} {cur next : Keys (R i0) 1}
    {g0 g1 g2 g3 : Fin 160 → ExactMatrix (R i0) N 1 1}
    (h : Stage_preprocess.sequential_parallel_generatedRoot_407_338 Backend.backend model params i0 i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold Stage_preprocess.sequential_parallel_generatedRoot_407_338 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [Stage_preprocess.sequential_parallel_generatedRoot_407_338.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `336`: the sums of the families over their 160 members. -/
theorem prerootSum336 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {g0 g1 g2 g3 : Fin 160 → ExactMatrix (R i0) N 1 1}
    {a0 b0 a1 b1 a2 b2 a3 b3 : ExactMatrix (R i0) N 1 1}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    {final : Keys (R i0) 1}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys (R i0) 1) ↦
      Stage_preprocess.sequential_parallel_generatedRoot_407_338 Backend.backend model params i0 i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, g0, g1, g2, g3, ()) next) (Int.toNat 158)
      (matrixAdd a0 b0, matrixAdd a1 b1, matrixAdd a2 b2, matrixAdd a3 b3, ()) final) :
    final = (∑ k, g0 k, ∑ k, g1 k, ∑ k, g2 k, ∑ k, g3 k, ()) := by
  have hloop := iter_sum (fun i ↦ (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()))
    (fun i current next hbody ↦ prerootStep336 hbody) hruns
  rw [show Int.toNat 158 = 158 from rfl] at hloop
  obtain ⟨_, ea0⟩ := familyGet_eq (k := 0) ha0
  obtain ⟨_, eb0⟩ := familyGet_eq (k := 1) hb0
  obtain ⟨_, ea1⟩ := familyGet_eq (k := 0) ha1
  obtain ⟨_, eb1⟩ := familyGet_eq (k := 1) hb1
  obtain ⟨_, ea2⟩ := familyGet_eq (k := 0) ha2
  obtain ⟨_, eb2⟩ := familyGet_eq (k := 1) hb2
  obtain ⟨_, ea3⟩ := familyGet_eq (k := 0) ha3
  obtain ⟨_, eb3⟩ := familyGet_eq (k := 1) hb3
  rw [hloop]
  simp only [ea0, eb0, ea1, eb1, ea2, eb2, ea3, eb3, matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 158) g0
  · exact sum_cells (w := 158) g1
  · exact sum_cells (w := 158) g2
  · exact sum_cells (w := 158) g3

theorem prerootStep401 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {i : Nat} {cur next : Keys (R i0) 1}
    {g0 g1 g2 g3 : Fin 160 → ExactMatrix (R i0) N 1 1}
    (h : Stage_preprocess.sequential_parallel_generatedRoot_407_405 Backend.backend model params i0 i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold Stage_preprocess.sequential_parallel_generatedRoot_407_405 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [Stage_preprocess.sequential_parallel_generatedRoot_407_405.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `401`: the sums of the families over their 160 members. -/
theorem prerootSum401 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {g0 g1 g2 g3 : Fin 160 → ExactMatrix (R i0) N 1 1}
    {a0 b0 a1 b1 a2 b2 a3 b3 : ExactMatrix (R i0) N 1 1}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    {final : Keys (R i0) 1}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys (R i0) 1) ↦
      Stage_preprocess.sequential_parallel_generatedRoot_407_405 Backend.backend model params i0 i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, g0, g1, g2, g3, ()) next) (Int.toNat 158)
      (matrixAdd a0 b0, matrixAdd a1 b1, matrixAdd a2 b2, matrixAdd a3 b3, ()) final) :
    final = (∑ k, g0 k, ∑ k, g1 k, ∑ k, g2 k, ∑ k, g3 k, ()) := by
  have hloop := iter_sum (fun i ↦ (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()))
    (fun i current next hbody ↦ prerootStep401 hbody) hruns
  rw [show Int.toNat 158 = 158 from rfl] at hloop
  obtain ⟨_, ea0⟩ := familyGet_eq (k := 0) ha0
  obtain ⟨_, eb0⟩ := familyGet_eq (k := 1) hb0
  obtain ⟨_, ea1⟩ := familyGet_eq (k := 0) ha1
  obtain ⟨_, eb1⟩ := familyGet_eq (k := 1) hb1
  obtain ⟨_, ea2⟩ := familyGet_eq (k := 0) ha2
  obtain ⟨_, eb2⟩ := familyGet_eq (k := 1) hb2
  obtain ⟨_, ea3⟩ := familyGet_eq (k := 0) ha3
  obtain ⟨_, eb3⟩ := familyGet_eq (k := 1) hb3
  rw [hloop]
  simp only [ea0, eb0, ea1, eb1, ea2, eb2, ea3, eb3, matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 158) g0
  · exact sum_cells (w := 158) g1
  · exact sum_cells (w := 158) g2
  · exact sum_cells (w := 158) g3

theorem prerootStep445 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {i : Nat} {cur next : Keys (R i0) 1}
    {g0 g1 g2 g3 : Fin 160 → ExactMatrix (R i0) N 1 1}
    (h : Stage_preprocess.sequential_parallel_generatedRoot_407_449 Backend.backend model params i0 i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold Stage_preprocess.sequential_parallel_generatedRoot_407_449 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [Stage_preprocess.sequential_parallel_generatedRoot_407_449.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `445`: the sums of the families over their 160 members. -/
theorem prerootSum445 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {g0 g1 g2 g3 : Fin 160 → ExactMatrix (R i0) N 1 1}
    {a0 b0 a1 b1 a2 b2 a3 b3 : ExactMatrix (R i0) N 1 1}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    {final : Keys (R i0) 1}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys (R i0) 1) ↦
      Stage_preprocess.sequential_parallel_generatedRoot_407_449 Backend.backend model params i0 i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, g0, g1, g2, g3, ()) next) (Int.toNat 158)
      (matrixAdd a0 b0, matrixAdd a1 b1, matrixAdd a2 b2, matrixAdd a3 b3, ()) final) :
    final = (∑ k, g0 k, ∑ k, g1 k, ∑ k, g2 k, ∑ k, g3 k, ()) := by
  have hloop := iter_sum (fun i ↦ (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()))
    (fun i current next hbody ↦ prerootStep445 hbody) hruns
  rw [show Int.toNat 158 = 158 from rfl] at hloop
  obtain ⟨_, ea0⟩ := familyGet_eq (k := 0) ha0
  obtain ⟨_, eb0⟩ := familyGet_eq (k := 1) hb0
  obtain ⟨_, ea1⟩ := familyGet_eq (k := 0) ha1
  obtain ⟨_, eb1⟩ := familyGet_eq (k := 1) hb1
  obtain ⟨_, ea2⟩ := familyGet_eq (k := 0) ha2
  obtain ⟨_, eb2⟩ := familyGet_eq (k := 1) hb2
  obtain ⟨_, ea3⟩ := familyGet_eq (k := 0) ha3
  obtain ⟨_, eb3⟩ := familyGet_eq (k := 1) hb3
  rw [hloop]
  simp only [ea0, eb0, ea1, eb1, ea2, eb2, ea3, eb3, matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 158) g0
  · exact sum_cells (w := 158) g1
  · exact sum_cells (w := 158) g2
  · exact sum_cells (w := 158) g3

theorem prerootStep489 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {i : Nat} {cur next : Keys (R i0) 1}
    {g0 g1 g2 g3 : Fin 160 → ExactMatrix (R i0) N 1 1}
    (h : Stage_preprocess.sequential_parallel_generatedRoot_407_493 Backend.backend model params i0 i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold Stage_preprocess.sequential_parallel_generatedRoot_407_493 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [Stage_preprocess.sequential_parallel_generatedRoot_407_493.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `489`: the sums of the families over their 160 members. -/
theorem prerootSum489 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {g0 g1 g2 g3 : Fin 160 → ExactMatrix (R i0) N 1 1}
    {a0 b0 a1 b1 a2 b2 a3 b3 : ExactMatrix (R i0) N 1 1}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    {final : Keys (R i0) 1}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys (R i0) 1) ↦
      Stage_preprocess.sequential_parallel_generatedRoot_407_493 Backend.backend model params i0 i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, g0, g1, g2, g3, ()) next) (Int.toNat 158)
      (matrixAdd a0 b0, matrixAdd a1 b1, matrixAdd a2 b2, matrixAdd a3 b3, ()) final) :
    final = (∑ k, g0 k, ∑ k, g1 k, ∑ k, g2 k, ∑ k, g3 k, ()) := by
  have hloop := iter_sum (fun i ↦ (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()))
    (fun i current next hbody ↦ prerootStep489 hbody) hruns
  rw [show Int.toNat 158 = 158 from rfl] at hloop
  obtain ⟨_, ea0⟩ := familyGet_eq (k := 0) ha0
  obtain ⟨_, eb0⟩ := familyGet_eq (k := 1) hb0
  obtain ⟨_, ea1⟩ := familyGet_eq (k := 0) ha1
  obtain ⟨_, eb1⟩ := familyGet_eq (k := 1) hb1
  obtain ⟨_, ea2⟩ := familyGet_eq (k := 0) ha2
  obtain ⟨_, eb2⟩ := familyGet_eq (k := 1) hb2
  obtain ⟨_, ea3⟩ := familyGet_eq (k := 0) ha3
  obtain ⟨_, eb3⟩ := familyGet_eq (k := 1) hb3
  rw [hloop]
  simp only [ea0, eb0, ea1, eb1, ea2, eb2, ea3, eb3, matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 158) g0
  · exact sum_cells (w := 158) g1
  · exact sum_cells (w := 158) g2
  · exact sum_cells (w := 158) g3

theorem prerootStep533 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {i : Nat} {cur next : Keys (R i0) 1}
    {g0 g1 g2 g3 : Fin 160 → ExactMatrix (R i0) N 1 1}
    (h : Stage_preprocess.sequential_parallel_generatedRoot_407_537 Backend.backend model params i0 i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold Stage_preprocess.sequential_parallel_generatedRoot_407_537 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [Stage_preprocess.sequential_parallel_generatedRoot_407_537.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `533`: the sums of the families over their 160 members. -/
theorem prerootSum533 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {g0 g1 g2 g3 : Fin 160 → ExactMatrix (R i0) N 1 1}
    {a0 b0 a1 b1 a2 b2 a3 b3 : ExactMatrix (R i0) N 1 1}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    {final : Keys (R i0) 1}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys (R i0) 1) ↦
      Stage_preprocess.sequential_parallel_generatedRoot_407_537 Backend.backend model params i0 i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, g0, g1, g2, g3, ()) next) (Int.toNat 158)
      (matrixAdd a0 b0, matrixAdd a1 b1, matrixAdd a2 b2, matrixAdd a3 b3, ()) final) :
    final = (∑ k, g0 k, ∑ k, g1 k, ∑ k, g2 k, ∑ k, g3 k, ()) := by
  have hloop := iter_sum (fun i ↦ (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()))
    (fun i current next hbody ↦ prerootStep533 hbody) hruns
  rw [show Int.toNat 158 = 158 from rfl] at hloop
  obtain ⟨_, ea0⟩ := familyGet_eq (k := 0) ha0
  obtain ⟨_, eb0⟩ := familyGet_eq (k := 1) hb0
  obtain ⟨_, ea1⟩ := familyGet_eq (k := 0) ha1
  obtain ⟨_, eb1⟩ := familyGet_eq (k := 1) hb1
  obtain ⟨_, ea2⟩ := familyGet_eq (k := 0) ha2
  obtain ⟨_, eb2⟩ := familyGet_eq (k := 1) hb2
  obtain ⟨_, ea3⟩ := familyGet_eq (k := 0) ha3
  obtain ⟨_, eb3⟩ := familyGet_eq (k := 1) hb3
  rw [hloop]
  simp only [ea0, eb0, ea1, eb1, ea2, eb2, ea3, eb3, matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 158) g0
  · exact sum_cells (w := 158) g1
  · exact sum_cells (w := 158) g2
  · exact sum_cells (w := 158) g3

theorem prerootStep577 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {i : Nat} {cur next : Keys (R i0) 1}
    {g0 g1 g2 g3 : Fin 160 → ExactMatrix (R i0) N 1 1}
    (h : Stage_preprocess.sequential_parallel_generatedRoot_407_581 Backend.backend model params i0 i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold Stage_preprocess.sequential_parallel_generatedRoot_407_581 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [Stage_preprocess.sequential_parallel_generatedRoot_407_581.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `577`: the sums of the families over their 160 members. -/
theorem prerootSum577 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {g0 g1 g2 g3 : Fin 160 → ExactMatrix (R i0) N 1 1}
    {a0 b0 a1 b1 a2 b2 a3 b3 : ExactMatrix (R i0) N 1 1}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    {final : Keys (R i0) 1}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys (R i0) 1) ↦
      Stage_preprocess.sequential_parallel_generatedRoot_407_581 Backend.backend model params i0 i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, g0, g1, g2, g3, ()) next) (Int.toNat 158)
      (matrixAdd a0 b0, matrixAdd a1 b1, matrixAdd a2 b2, matrixAdd a3 b3, ()) final) :
    final = (∑ k, g0 k, ∑ k, g1 k, ∑ k, g2 k, ∑ k, g3 k, ()) := by
  have hloop := iter_sum (fun i ↦ (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()))
    (fun i current next hbody ↦ prerootStep577 hbody) hruns
  rw [show Int.toNat 158 = 158 from rfl] at hloop
  obtain ⟨_, ea0⟩ := familyGet_eq (k := 0) ha0
  obtain ⟨_, eb0⟩ := familyGet_eq (k := 1) hb0
  obtain ⟨_, ea1⟩ := familyGet_eq (k := 0) ha1
  obtain ⟨_, eb1⟩ := familyGet_eq (k := 1) hb1
  obtain ⟨_, ea2⟩ := familyGet_eq (k := 0) ha2
  obtain ⟨_, eb2⟩ := familyGet_eq (k := 1) hb2
  obtain ⟨_, ea3⟩ := familyGet_eq (k := 0) ha3
  obtain ⟨_, eb3⟩ := familyGet_eq (k := 1) hb3
  rw [hloop]
  simp only [ea0, eb0, ea1, eb1, ea2, eb2, ea3, eb3, matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 158) g0
  · exact sum_cells (w := 158) g1
  · exact sum_cells (w := 158) g2
  · exact sum_cells (w := 158) g3

end RefreshTopProof.Part_I
