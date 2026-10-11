import TopLemmas

/-! The member sums of the online root (written by `generator/gen_root_sums.py`). -/

namespace RefreshTopProof.Part_0_0

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

theorem prerootStep166 {model : HashModel} {params} {i : Nat} {cur next : Keys Q0 6}
    {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_166 Backend.backend model params i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold Stage_preprocess_0_0.sequential_generatedRoot_166 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [Stage_preprocess_0_0.sequential_generatedRoot_166.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `166`: the sums of the families over their 8 members. -/
theorem prerootSum166 {model : HashModel} {params} {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    {a0 b0 a1 b1 a2 b2 a3 b3 : ExactMatrix Q0 N 1 6}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    {final : Keys Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q0 6) ↦
      Stage_preprocess_0_0.sequential_generatedRoot_166 Backend.backend model params i
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

theorem prerootStep233 {model : HashModel} {params} {i : Nat} {cur next : Keys Q0 6}
    {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_233 Backend.backend model params i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold Stage_preprocess_0_0.sequential_generatedRoot_233 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [Stage_preprocess_0_0.sequential_generatedRoot_233.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `233`: the sums of the families over their 8 members. -/
theorem prerootSum233 {model : HashModel} {params} {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    {a0 b0 a1 b1 a2 b2 a3 b3 : ExactMatrix Q0 N 1 6}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    {final : Keys Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q0 6) ↦
      Stage_preprocess_0_0.sequential_generatedRoot_233 Backend.backend model params i
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

theorem prerootStep252 {model : HashModel} {params} {i : Nat} {cur next : Keys Q0 6}
    {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_252 Backend.backend model params i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold Stage_preprocess_0_0.sequential_generatedRoot_252 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [Stage_preprocess_0_0.sequential_generatedRoot_252.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `252`: the sums of the families over their 8 members. -/
theorem prerootSum252 {model : HashModel} {params} {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    {a0 b0 a1 b1 a2 b2 a3 b3 : ExactMatrix Q0 N 1 6}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    {final : Keys Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q0 6) ↦
      Stage_preprocess_0_0.sequential_generatedRoot_252 Backend.backend model params i
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

theorem prerootStep268 {model : HashModel} {params} {i : Nat} {cur next : Keys Q0 6}
    {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_268 Backend.backend model params i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold Stage_preprocess_0_0.sequential_generatedRoot_268 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [Stage_preprocess_0_0.sequential_generatedRoot_268.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `268`: the sums of the families over their 8 members. -/
theorem prerootSum268 {model : HashModel} {params} {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    {a0 b0 a1 b1 a2 b2 a3 b3 : ExactMatrix Q0 N 1 6}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    {final : Keys Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q0 6) ↦
      Stage_preprocess_0_0.sequential_generatedRoot_268 Backend.backend model params i
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

theorem prerootStep284 {model : HashModel} {params} {i : Nat} {cur next : Keys Q0 6}
    {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_284 Backend.backend model params i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold Stage_preprocess_0_0.sequential_generatedRoot_284 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [Stage_preprocess_0_0.sequential_generatedRoot_284.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `284`: the sums of the families over their 8 members. -/
theorem prerootSum284 {model : HashModel} {params} {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    {a0 b0 a1 b1 a2 b2 a3 b3 : ExactMatrix Q0 N 1 6}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    {final : Keys Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q0 6) ↦
      Stage_preprocess_0_0.sequential_generatedRoot_284 Backend.backend model params i
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

theorem prerootStep300 {model : HashModel} {params} {i : Nat} {cur next : Keys Q0 6}
    {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_300 Backend.backend model params i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold Stage_preprocess_0_0.sequential_generatedRoot_300 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [Stage_preprocess_0_0.sequential_generatedRoot_300.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `300`: the sums of the families over their 8 members. -/
theorem prerootSum300 {model : HashModel} {params} {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    {a0 b0 a1 b1 a2 b2 a3 b3 : ExactMatrix Q0 N 1 6}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    {final : Keys Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q0 6) ↦
      Stage_preprocess_0_0.sequential_generatedRoot_300 Backend.backend model params i
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

theorem prerootStep316 {model : HashModel} {params} {i : Nat} {cur next : Keys Q0 6}
    {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_316 Backend.backend model params i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold Stage_preprocess_0_0.sequential_generatedRoot_316 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [Stage_preprocess_0_0.sequential_generatedRoot_316.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `316`: the sums of the families over their 8 members. -/
theorem prerootSum316 {model : HashModel} {params} {g0 g1 g2 g3 : Fin 8 → ExactMatrix Q0 N 1 6}
    {a0 b0 a1 b1 a2 b2 a3 b3 : ExactMatrix Q0 N 1 6}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    {final : Keys Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q0 6) ↦
      Stage_preprocess_0_0.sequential_generatedRoot_316 Backend.backend model params i
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

theorem prerootStep336 {model : HashModel} {params} {i : Nat} {cur next : Keys Q1 1}
    {g0 g1 g2 g3 : Fin 160 → ExactMatrix Q1 N 1 1}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_336 Backend.backend model params i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold Stage_preprocess_0_0.sequential_generatedRoot_336 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [Stage_preprocess_0_0.sequential_generatedRoot_336.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `336`: the sums of the families over their 160 members. -/
theorem prerootSum336 {model : HashModel} {params} {g0 g1 g2 g3 : Fin 160 → ExactMatrix Q1 N 1 1}
    {a0 b0 a1 b1 a2 b2 a3 b3 : ExactMatrix Q1 N 1 1}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    {final : Keys Q1 1}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q1 1) ↦
      Stage_preprocess_0_0.sequential_generatedRoot_336 Backend.backend model params i
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

theorem prerootStep401 {model : HashModel} {params} {i : Nat} {cur next : Keys Q1 1}
    {g0 g1 g2 g3 : Fin 160 → ExactMatrix Q1 N 1 1}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_401 Backend.backend model params i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold Stage_preprocess_0_0.sequential_generatedRoot_401 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [Stage_preprocess_0_0.sequential_generatedRoot_401.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `401`: the sums of the families over their 160 members. -/
theorem prerootSum401 {model : HashModel} {params} {g0 g1 g2 g3 : Fin 160 → ExactMatrix Q1 N 1 1}
    {a0 b0 a1 b1 a2 b2 a3 b3 : ExactMatrix Q1 N 1 1}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    {final : Keys Q1 1}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q1 1) ↦
      Stage_preprocess_0_0.sequential_generatedRoot_401 Backend.backend model params i
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

theorem prerootStep445 {model : HashModel} {params} {i : Nat} {cur next : Keys Q1 1}
    {g0 g1 g2 g3 : Fin 160 → ExactMatrix Q1 N 1 1}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_445 Backend.backend model params i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold Stage_preprocess_0_0.sequential_generatedRoot_445 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [Stage_preprocess_0_0.sequential_generatedRoot_445.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `445`: the sums of the families over their 160 members. -/
theorem prerootSum445 {model : HashModel} {params} {g0 g1 g2 g3 : Fin 160 → ExactMatrix Q1 N 1 1}
    {a0 b0 a1 b1 a2 b2 a3 b3 : ExactMatrix Q1 N 1 1}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    {final : Keys Q1 1}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q1 1) ↦
      Stage_preprocess_0_0.sequential_generatedRoot_445 Backend.backend model params i
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

theorem prerootStep489 {model : HashModel} {params} {i : Nat} {cur next : Keys Q1 1}
    {g0 g1 g2 g3 : Fin 160 → ExactMatrix Q1 N 1 1}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_489 Backend.backend model params i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold Stage_preprocess_0_0.sequential_generatedRoot_489 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [Stage_preprocess_0_0.sequential_generatedRoot_489.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `489`: the sums of the families over their 160 members. -/
theorem prerootSum489 {model : HashModel} {params} {g0 g1 g2 g3 : Fin 160 → ExactMatrix Q1 N 1 1}
    {a0 b0 a1 b1 a2 b2 a3 b3 : ExactMatrix Q1 N 1 1}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    {final : Keys Q1 1}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q1 1) ↦
      Stage_preprocess_0_0.sequential_generatedRoot_489 Backend.backend model params i
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

theorem prerootStep533 {model : HashModel} {params} {i : Nat} {cur next : Keys Q1 1}
    {g0 g1 g2 g3 : Fin 160 → ExactMatrix Q1 N 1 1}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_533 Backend.backend model params i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold Stage_preprocess_0_0.sequential_generatedRoot_533 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [Stage_preprocess_0_0.sequential_generatedRoot_533.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `533`: the sums of the families over their 160 members. -/
theorem prerootSum533 {model : HashModel} {params} {g0 g1 g2 g3 : Fin 160 → ExactMatrix Q1 N 1 1}
    {a0 b0 a1 b1 a2 b2 a3 b3 : ExactMatrix Q1 N 1 1}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    {final : Keys Q1 1}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q1 1) ↦
      Stage_preprocess_0_0.sequential_generatedRoot_533 Backend.backend model params i
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

theorem prerootStep577 {model : HashModel} {params} {i : Nat} {cur next : Keys Q1 1}
    {g0 g1 g2 g3 : Fin 160 → ExactMatrix Q1 N 1 1}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_577 Backend.backend model params i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold Stage_preprocess_0_0.sequential_generatedRoot_577 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [Stage_preprocess_0_0.sequential_generatedRoot_577.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- The root sum `577`: the sums of the families over their 160 members. -/
theorem prerootSum577 {model : HashModel} {params} {g0 g1 g2 g3 : Fin 160 → ExactMatrix Q1 N 1 1}
    {a0 b0 a1 b1 a2 b2 a3 b3 : ExactMatrix Q1 N 1 1}
    (ha0 : familyGetStatic g0 0 a0) (hb0 : familyGetStatic g0 1 b0)
    (ha1 : familyGetStatic g1 0 a1) (hb1 : familyGetStatic g1 1 b1)
    (ha2 : familyGetStatic g2 0 a2) (hb2 : familyGetStatic g2 1 b2)
    (ha3 : familyGetStatic g3 0 a3) (hb3 : familyGetStatic g3 1 b3)
    {final : Keys Q1 1}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q1 1) ↦
      Stage_preprocess_0_0.sequential_generatedRoot_577 Backend.backend model params i
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

end RefreshTopProof.Part_0_0
