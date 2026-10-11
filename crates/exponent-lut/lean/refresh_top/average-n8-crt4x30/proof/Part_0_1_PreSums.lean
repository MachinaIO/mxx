import TopLemmas

/-! The member sums of the preprocessing's PRF keys (written by `generator/gen_pre_sums.py`). -/

namespace RefreshTopProof.Part_0_1

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

theorem pregather21 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_21_4 Backend.backend model params o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_21_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_21_4.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  subst e2_0 e4_0 e6_0 e8_0 hout
  rfl

theorem prestep21 {model : HashModel} {params} {o i : Nat} {cur next : Keys Q5 8}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q5 N 1 8}
    (h : W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_21_17 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_21_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_21_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `21`: the sum of its 5 cells. -/
theorem premembers21 {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess_0_1.parallel_generatedRoot_21 Backend.backend model params o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess_0_1.parallel_generatedRoot_21.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess_0_1.parallel_generatedRoot_21.constraints_0] at h
  obtain ⟨-, h4_3, h5_0, h6_0, h8_0, h9_0, h11_0, h12_0, h14_0, h15_0, -, h17_3, hout⟩ := h
  have hg : ∀ i : Fin 5, w_4_0 i = f0 (cellIndex o i) ∧
      w_4_1 i = f1 (cellIndex o i) ∧
      w_4_2 i = f2 (cellIndex o i) ∧
      w_4_3 i = f3 (cellIndex o i) := by
    intro i
    have := pregather21 (h4_3 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, -⟩ := this
    exact ⟨a0, a1, a2, a3⟩
  have hw0 : w_4_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_4_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_4_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_4_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2
  obtain ⟨_, e5_0⟩ := familyGet_eq (k := 0) h5_0
  obtain ⟨_, e6_0⟩ := familyGet_eq (k := 1) h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 0) h8_0
  obtain ⟨_, e9_0⟩ := familyGet_eq (k := 1) h9_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 0) h11_0
  obtain ⟨_, e12_0⟩ := familyGet_eq (k := 1) h12_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 0) h14_0
  obtain ⟨_, e15_0⟩ := familyGet_eq (k := 1) h15_0
  have hloop := iter_sum (fun i ↦ (cellAt w_4_0 (2 + i), cellAt w_4_1 (2 + i), cellAt w_4_2 (2 + i), cellAt w_4_3 (2 + i), ()))
    (fun i current next hbody ↦ prestep21 hbody) h17_3
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_7_0, w_10_0, w_13_0, w_16_0, e5_0, e6_0, e8_0, e9_0, e11_0, e12_0, e14_0, e15_0, hw0, hw1, hw2, hw3]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))

theorem pregather36 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 40 → ExactMatrix Q6 N 1 10} {outputs : Keys Q6 10}
    (h : W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_36_4 Backend.backend model params o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 8) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_36_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_36_4.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  subst e2_0 e4_0 e6_0 e8_0 hout
  rfl

theorem prestep36 {model : HashModel} {params} {o i : Nat} {cur next : Keys Q6 10}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q6 N 1 10}
    (h : W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_36_17 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_36_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_36_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `36`: the sum of its 5 cells. -/
theorem premembers36 {model : HashModel} {params} {o : Fin 8}
    {f0 f1 f2 f3 : Fin 40 → ExactMatrix Q6 N 1 10} {outputs : Keys Q6 10}
    (h : W_Stage_preprocess_0_1.parallel_generatedRoot_36 Backend.backend model params o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess_0_1.parallel_generatedRoot_36.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess_0_1.parallel_generatedRoot_36.constraints_0] at h
  obtain ⟨-, h4_3, h5_0, h6_0, h8_0, h9_0, h11_0, h12_0, h14_0, h15_0, -, h17_3, hout⟩ := h
  have hg : ∀ i : Fin 5, w_4_0 i = f0 (cellIndex o i) ∧
      w_4_1 i = f1 (cellIndex o i) ∧
      w_4_2 i = f2 (cellIndex o i) ∧
      w_4_3 i = f3 (cellIndex o i) := by
    intro i
    have := pregather36 (h4_3 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, -⟩ := this
    exact ⟨a0, a1, a2, a3⟩
  have hw0 : w_4_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_4_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_4_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_4_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2
  obtain ⟨_, e5_0⟩ := familyGet_eq (k := 0) h5_0
  obtain ⟨_, e6_0⟩ := familyGet_eq (k := 1) h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 0) h8_0
  obtain ⟨_, e9_0⟩ := familyGet_eq (k := 1) h9_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 0) h11_0
  obtain ⟨_, e12_0⟩ := familyGet_eq (k := 1) h12_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 0) h14_0
  obtain ⟨_, e15_0⟩ := familyGet_eq (k := 1) h15_0
  have hloop := iter_sum (fun i ↦ (cellAt w_4_0 (2 + i), cellAt w_4_1 (2 + i), cellAt w_4_2 (2 + i), cellAt w_4_3 (2 + i), ()))
    (fun i current next hbody ↦ prestep36 hbody) h17_3
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_7_0, w_10_0, w_13_0, w_16_0, e5_0, e6_0, e8_0, e9_0, e11_0, e12_0, e14_0, e15_0, hw0, hw1, hw2, hw3]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))

theorem pregather40 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 40 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_40_4 Backend.backend model params o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 8) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_40_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_40_4.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  subst e2_0 e4_0 e6_0 e8_0 hout
  rfl

theorem prestep40 {model : HashModel} {params} {o i : Nat} {cur next : Keys Q5 8}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q5 N 1 8}
    (h : W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_40_17 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_40_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_40_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `40`: the sum of its 5 cells. -/
theorem premembers40 {model : HashModel} {params} {o : Fin 8}
    {f0 f1 f2 f3 : Fin 40 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess_0_1.parallel_generatedRoot_40 Backend.backend model params o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess_0_1.parallel_generatedRoot_40.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess_0_1.parallel_generatedRoot_40.constraints_0] at h
  obtain ⟨-, h4_3, h5_0, h6_0, h8_0, h9_0, h11_0, h12_0, h14_0, h15_0, -, h17_3, hout⟩ := h
  have hg : ∀ i : Fin 5, w_4_0 i = f0 (cellIndex o i) ∧
      w_4_1 i = f1 (cellIndex o i) ∧
      w_4_2 i = f2 (cellIndex o i) ∧
      w_4_3 i = f3 (cellIndex o i) := by
    intro i
    have := pregather40 (h4_3 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, -⟩ := this
    exact ⟨a0, a1, a2, a3⟩
  have hw0 : w_4_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_4_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_4_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_4_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2
  obtain ⟨_, e5_0⟩ := familyGet_eq (k := 0) h5_0
  obtain ⟨_, e6_0⟩ := familyGet_eq (k := 1) h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 0) h8_0
  obtain ⟨_, e9_0⟩ := familyGet_eq (k := 1) h9_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 0) h11_0
  obtain ⟨_, e12_0⟩ := familyGet_eq (k := 1) h12_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 0) h14_0
  obtain ⟨_, e15_0⟩ := familyGet_eq (k := 1) h15_0
  have hloop := iter_sum (fun i ↦ (cellAt w_4_0 (2 + i), cellAt w_4_1 (2 + i), cellAt w_4_2 (2 + i), cellAt w_4_3 (2 + i), ()))
    (fun i current next hbody ↦ prestep40 hbody) h17_3
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_7_0, w_10_0, w_13_0, w_16_0, e5_0, e6_0, e8_0, e9_0, e11_0, e12_0, e14_0, e15_0, hw0, hw1, hw2, hw3]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))

theorem pregather211 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 240 → ExactMatrix Q6 N 1 10} {outputs : Keys Q6 10}
    (h : W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_211_4 Backend.backend model params o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 48) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_211_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_211_4.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  subst e2_0 e4_0 e6_0 e8_0 hout
  rfl

theorem prestep211 {model : HashModel} {params} {o i : Nat} {cur next : Keys Q6 10}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q6 N 1 10}
    (h : W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_211_17 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_211_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_211_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `211`: the sum of its 5 cells. -/
theorem premembers211 {model : HashModel} {params} {o : Fin 48}
    {f0 f1 f2 f3 : Fin 240 → ExactMatrix Q6 N 1 10} {outputs : Keys Q6 10}
    (h : W_Stage_preprocess_0_1.parallel_generatedRoot_211 Backend.backend model params o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess_0_1.parallel_generatedRoot_211.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess_0_1.parallel_generatedRoot_211.constraints_0] at h
  obtain ⟨-, h4_3, h5_0, h6_0, h8_0, h9_0, h11_0, h12_0, h14_0, h15_0, -, h17_3, hout⟩ := h
  have hg : ∀ i : Fin 5, w_4_0 i = f0 (cellIndex o i) ∧
      w_4_1 i = f1 (cellIndex o i) ∧
      w_4_2 i = f2 (cellIndex o i) ∧
      w_4_3 i = f3 (cellIndex o i) := by
    intro i
    have := pregather211 (h4_3 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, -⟩ := this
    exact ⟨a0, a1, a2, a3⟩
  have hw0 : w_4_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_4_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_4_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_4_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2
  obtain ⟨_, e5_0⟩ := familyGet_eq (k := 0) h5_0
  obtain ⟨_, e6_0⟩ := familyGet_eq (k := 1) h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 0) h8_0
  obtain ⟨_, e9_0⟩ := familyGet_eq (k := 1) h9_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 0) h11_0
  obtain ⟨_, e12_0⟩ := familyGet_eq (k := 1) h12_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 0) h14_0
  obtain ⟨_, e15_0⟩ := familyGet_eq (k := 1) h15_0
  have hloop := iter_sum (fun i ↦ (cellAt w_4_0 (2 + i), cellAt w_4_1 (2 + i), cellAt w_4_2 (2 + i), cellAt w_4_3 (2 + i), ()))
    (fun i current next hbody ↦ prestep211 hbody) h17_3
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_7_0, w_10_0, w_13_0, w_16_0, e5_0, e6_0, e8_0, e9_0, e11_0, e12_0, e14_0, e15_0, hw0, hw1, hw2, hw3]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))

theorem pregather215 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 240 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_215_4 Backend.backend model params o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 48) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_215_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_215_4.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  subst e2_0 e4_0 e6_0 e8_0 hout
  rfl

theorem prestep215 {model : HashModel} {params} {o i : Nat} {cur next : Keys Q5 8}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q5 N 1 8}
    (h : W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_215_17 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_215_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_215_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `215`: the sum of its 5 cells. -/
theorem premembers215 {model : HashModel} {params} {o : Fin 48}
    {f0 f1 f2 f3 : Fin 240 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess_0_1.parallel_generatedRoot_215 Backend.backend model params o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess_0_1.parallel_generatedRoot_215.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess_0_1.parallel_generatedRoot_215.constraints_0] at h
  obtain ⟨-, h4_3, h5_0, h6_0, h8_0, h9_0, h11_0, h12_0, h14_0, h15_0, -, h17_3, hout⟩ := h
  have hg : ∀ i : Fin 5, w_4_0 i = f0 (cellIndex o i) ∧
      w_4_1 i = f1 (cellIndex o i) ∧
      w_4_2 i = f2 (cellIndex o i) ∧
      w_4_3 i = f3 (cellIndex o i) := by
    intro i
    have := pregather215 (h4_3 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, -⟩ := this
    exact ⟨a0, a1, a2, a3⟩
  have hw0 : w_4_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_4_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_4_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_4_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2
  obtain ⟨_, e5_0⟩ := familyGet_eq (k := 0) h5_0
  obtain ⟨_, e6_0⟩ := familyGet_eq (k := 1) h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 0) h8_0
  obtain ⟨_, e9_0⟩ := familyGet_eq (k := 1) h9_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 0) h11_0
  obtain ⟨_, e12_0⟩ := familyGet_eq (k := 1) h12_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 0) h14_0
  obtain ⟨_, e15_0⟩ := familyGet_eq (k := 1) h15_0
  have hloop := iter_sum (fun i ↦ (cellAt w_4_0 (2 + i), cellAt w_4_1 (2 + i), cellAt w_4_2 (2 + i), cellAt w_4_3 (2 + i), ()))
    (fun i current next hbody ↦ prestep215 hbody) h17_3
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_7_0, w_10_0, w_13_0, w_16_0, e5_0, e6_0, e8_0, e9_0, e11_0, e12_0, e14_0, e15_0, hw0, hw1, hw2, hw3]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))

theorem pregather322 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Keys Q0 6}
    (h : W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_322_4 Backend.backend model params o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_322_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_322_4.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  subst e2_0 e4_0 e6_0 e8_0 hout
  rfl

theorem prestep322 {model : HashModel} {params} {o i : Nat} {cur next : Keys Q0 6}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q0 N 1 6}
    (h : W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_322_17 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_322_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_322_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `322`: the sum of its 5 cells. -/
theorem premembers322 {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Keys Q0 6}
    (h : W_Stage_preprocess_0_1.parallel_generatedRoot_322 Backend.backend model params o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess_0_1.parallel_generatedRoot_322.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess_0_1.parallel_generatedRoot_322.constraints_0] at h
  obtain ⟨-, h4_3, h5_0, h6_0, h8_0, h9_0, h11_0, h12_0, h14_0, h15_0, -, h17_3, hout⟩ := h
  have hg : ∀ i : Fin 5, w_4_0 i = f0 (cellIndex o i) ∧
      w_4_1 i = f1 (cellIndex o i) ∧
      w_4_2 i = f2 (cellIndex o i) ∧
      w_4_3 i = f3 (cellIndex o i) := by
    intro i
    have := pregather322 (h4_3 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, -⟩ := this
    exact ⟨a0, a1, a2, a3⟩
  have hw0 : w_4_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_4_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_4_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_4_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2
  obtain ⟨_, e5_0⟩ := familyGet_eq (k := 0) h5_0
  obtain ⟨_, e6_0⟩ := familyGet_eq (k := 1) h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 0) h8_0
  obtain ⟨_, e9_0⟩ := familyGet_eq (k := 1) h9_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 0) h11_0
  obtain ⟨_, e12_0⟩ := familyGet_eq (k := 1) h12_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 0) h14_0
  obtain ⟨_, e15_0⟩ := familyGet_eq (k := 1) h15_0
  have hloop := iter_sum (fun i ↦ (cellAt w_4_0 (2 + i), cellAt w_4_1 (2 + i), cellAt w_4_2 (2 + i), cellAt w_4_3 (2 + i), ()))
    (fun i current next hbody ↦ prestep322 hbody) h17_3
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_7_0, w_10_0, w_13_0, w_16_0, e5_0, e6_0, e8_0, e9_0, e11_0, e12_0, e14_0, e15_0, hw0, hw1, hw2, hw3]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))

theorem pregather382 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_382_4 Backend.backend model params o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_382_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_382_4.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  subst e2_0 e4_0 e6_0 e8_0 hout
  rfl

theorem prestep382 {model : HashModel} {params} {o i : Nat} {cur next : Keys Q5 8}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q5 N 1 8}
    (h : W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_382_17 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_382_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_382_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `382`: the sum of its 5 cells. -/
theorem premembers382 {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess_0_1.parallel_generatedRoot_382 Backend.backend model params o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess_0_1.parallel_generatedRoot_382.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess_0_1.parallel_generatedRoot_382.constraints_0] at h
  obtain ⟨-, h4_3, h5_0, h6_0, h8_0, h9_0, h11_0, h12_0, h14_0, h15_0, -, h17_3, hout⟩ := h
  have hg : ∀ i : Fin 5, w_4_0 i = f0 (cellIndex o i) ∧
      w_4_1 i = f1 (cellIndex o i) ∧
      w_4_2 i = f2 (cellIndex o i) ∧
      w_4_3 i = f3 (cellIndex o i) := by
    intro i
    have := pregather382 (h4_3 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, -⟩ := this
    exact ⟨a0, a1, a2, a3⟩
  have hw0 : w_4_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_4_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_4_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_4_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2
  obtain ⟨_, e5_0⟩ := familyGet_eq (k := 0) h5_0
  obtain ⟨_, e6_0⟩ := familyGet_eq (k := 1) h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 0) h8_0
  obtain ⟨_, e9_0⟩ := familyGet_eq (k := 1) h9_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 0) h11_0
  obtain ⟨_, e12_0⟩ := familyGet_eq (k := 1) h12_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 0) h14_0
  obtain ⟨_, e15_0⟩ := familyGet_eq (k := 1) h15_0
  have hloop := iter_sum (fun i ↦ (cellAt w_4_0 (2 + i), cellAt w_4_1 (2 + i), cellAt w_4_2 (2 + i), cellAt w_4_3 (2 + i), ()))
    (fun i current next hbody ↦ prestep382 hbody) h17_3
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_7_0, w_10_0, w_13_0, w_16_0, e5_0, e6_0, e8_0, e9_0, e11_0, e12_0, e14_0, e15_0, hw0, hw1, hw2, hw3]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))

theorem pregather387 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Keys Q0 6}
    (h : W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_387_4 Backend.backend model params o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_387_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_387_4.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  subst e2_0 e4_0 e6_0 e8_0 hout
  rfl

theorem prestep387 {model : HashModel} {params} {o i : Nat} {cur next : Keys Q0 6}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q0 N 1 6}
    (h : W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_387_17 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_387_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_387_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `387`: the sum of its 5 cells. -/
theorem premembers387 {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Keys Q0 6}
    (h : W_Stage_preprocess_0_1.parallel_generatedRoot_387 Backend.backend model params o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess_0_1.parallel_generatedRoot_387.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess_0_1.parallel_generatedRoot_387.constraints_0] at h
  obtain ⟨-, h4_3, h5_0, h6_0, h8_0, h9_0, h11_0, h12_0, h14_0, h15_0, -, h17_3, hout⟩ := h
  have hg : ∀ i : Fin 5, w_4_0 i = f0 (cellIndex o i) ∧
      w_4_1 i = f1 (cellIndex o i) ∧
      w_4_2 i = f2 (cellIndex o i) ∧
      w_4_3 i = f3 (cellIndex o i) := by
    intro i
    have := pregather387 (h4_3 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, -⟩ := this
    exact ⟨a0, a1, a2, a3⟩
  have hw0 : w_4_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_4_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_4_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_4_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2
  obtain ⟨_, e5_0⟩ := familyGet_eq (k := 0) h5_0
  obtain ⟨_, e6_0⟩ := familyGet_eq (k := 1) h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 0) h8_0
  obtain ⟨_, e9_0⟩ := familyGet_eq (k := 1) h9_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 0) h11_0
  obtain ⟨_, e12_0⟩ := familyGet_eq (k := 1) h12_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 0) h14_0
  obtain ⟨_, e15_0⟩ := familyGet_eq (k := 1) h15_0
  have hloop := iter_sum (fun i ↦ (cellAt w_4_0 (2 + i), cellAt w_4_1 (2 + i), cellAt w_4_2 (2 + i), cellAt w_4_3 (2 + i), ()))
    (fun i current next hbody ↦ prestep387 hbody) h17_3
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_7_0, w_10_0, w_13_0, w_16_0, e5_0, e6_0, e8_0, e9_0, e11_0, e12_0, e14_0, e15_0, hw0, hw1, hw2, hw3]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))

theorem pregather426 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_426_4 Backend.backend model params o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_426_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_426_4.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  subst e2_0 e4_0 e6_0 e8_0 hout
  rfl

theorem prestep426 {model : HashModel} {params} {o i : Nat} {cur next : Keys Q5 8}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q5 N 1 8}
    (h : W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_426_17 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_426_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_426_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `426`: the sum of its 5 cells. -/
theorem premembers426 {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess_0_1.parallel_generatedRoot_426 Backend.backend model params o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess_0_1.parallel_generatedRoot_426.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess_0_1.parallel_generatedRoot_426.constraints_0] at h
  obtain ⟨-, h4_3, h5_0, h6_0, h8_0, h9_0, h11_0, h12_0, h14_0, h15_0, -, h17_3, hout⟩ := h
  have hg : ∀ i : Fin 5, w_4_0 i = f0 (cellIndex o i) ∧
      w_4_1 i = f1 (cellIndex o i) ∧
      w_4_2 i = f2 (cellIndex o i) ∧
      w_4_3 i = f3 (cellIndex o i) := by
    intro i
    have := pregather426 (h4_3 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, -⟩ := this
    exact ⟨a0, a1, a2, a3⟩
  have hw0 : w_4_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_4_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_4_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_4_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2
  obtain ⟨_, e5_0⟩ := familyGet_eq (k := 0) h5_0
  obtain ⟨_, e6_0⟩ := familyGet_eq (k := 1) h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 0) h8_0
  obtain ⟨_, e9_0⟩ := familyGet_eq (k := 1) h9_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 0) h11_0
  obtain ⟨_, e12_0⟩ := familyGet_eq (k := 1) h12_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 0) h14_0
  obtain ⟨_, e15_0⟩ := familyGet_eq (k := 1) h15_0
  have hloop := iter_sum (fun i ↦ (cellAt w_4_0 (2 + i), cellAt w_4_1 (2 + i), cellAt w_4_2 (2 + i), cellAt w_4_3 (2 + i), ()))
    (fun i current next hbody ↦ prestep426 hbody) h17_3
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_7_0, w_10_0, w_13_0, w_16_0, e5_0, e6_0, e8_0, e9_0, e11_0, e12_0, e14_0, e15_0, hw0, hw1, hw2, hw3]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))

theorem pregather431 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Keys Q0 6}
    (h : W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_431_4 Backend.backend model params o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_431_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_431_4.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  subst e2_0 e4_0 e6_0 e8_0 hout
  rfl

theorem prestep431 {model : HashModel} {params} {o i : Nat} {cur next : Keys Q0 6}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q0 N 1 6}
    (h : W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_431_17 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_431_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_431_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `431`: the sum of its 5 cells. -/
theorem premembers431 {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Keys Q0 6}
    (h : W_Stage_preprocess_0_1.parallel_generatedRoot_431 Backend.backend model params o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess_0_1.parallel_generatedRoot_431.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess_0_1.parallel_generatedRoot_431.constraints_0] at h
  obtain ⟨-, h4_3, h5_0, h6_0, h8_0, h9_0, h11_0, h12_0, h14_0, h15_0, -, h17_3, hout⟩ := h
  have hg : ∀ i : Fin 5, w_4_0 i = f0 (cellIndex o i) ∧
      w_4_1 i = f1 (cellIndex o i) ∧
      w_4_2 i = f2 (cellIndex o i) ∧
      w_4_3 i = f3 (cellIndex o i) := by
    intro i
    have := pregather431 (h4_3 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, -⟩ := this
    exact ⟨a0, a1, a2, a3⟩
  have hw0 : w_4_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_4_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_4_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_4_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2
  obtain ⟨_, e5_0⟩ := familyGet_eq (k := 0) h5_0
  obtain ⟨_, e6_0⟩ := familyGet_eq (k := 1) h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 0) h8_0
  obtain ⟨_, e9_0⟩ := familyGet_eq (k := 1) h9_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 0) h11_0
  obtain ⟨_, e12_0⟩ := familyGet_eq (k := 1) h12_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 0) h14_0
  obtain ⟨_, e15_0⟩ := familyGet_eq (k := 1) h15_0
  have hloop := iter_sum (fun i ↦ (cellAt w_4_0 (2 + i), cellAt w_4_1 (2 + i), cellAt w_4_2 (2 + i), cellAt w_4_3 (2 + i), ()))
    (fun i current next hbody ↦ prestep431 hbody) h17_3
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_7_0, w_10_0, w_13_0, w_16_0, e5_0, e6_0, e8_0, e9_0, e11_0, e12_0, e14_0, e15_0, hw0, hw1, hw2, hw3]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))

theorem pregather470 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_470_4 Backend.backend model params o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_470_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_470_4.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  subst e2_0 e4_0 e6_0 e8_0 hout
  rfl

theorem prestep470 {model : HashModel} {params} {o i : Nat} {cur next : Keys Q5 8}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q5 N 1 8}
    (h : W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_470_17 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_470_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_470_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `470`: the sum of its 5 cells. -/
theorem premembers470 {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess_0_1.parallel_generatedRoot_470 Backend.backend model params o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess_0_1.parallel_generatedRoot_470.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess_0_1.parallel_generatedRoot_470.constraints_0] at h
  obtain ⟨-, h4_3, h5_0, h6_0, h8_0, h9_0, h11_0, h12_0, h14_0, h15_0, -, h17_3, hout⟩ := h
  have hg : ∀ i : Fin 5, w_4_0 i = f0 (cellIndex o i) ∧
      w_4_1 i = f1 (cellIndex o i) ∧
      w_4_2 i = f2 (cellIndex o i) ∧
      w_4_3 i = f3 (cellIndex o i) := by
    intro i
    have := pregather470 (h4_3 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, -⟩ := this
    exact ⟨a0, a1, a2, a3⟩
  have hw0 : w_4_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_4_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_4_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_4_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2
  obtain ⟨_, e5_0⟩ := familyGet_eq (k := 0) h5_0
  obtain ⟨_, e6_0⟩ := familyGet_eq (k := 1) h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 0) h8_0
  obtain ⟨_, e9_0⟩ := familyGet_eq (k := 1) h9_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 0) h11_0
  obtain ⟨_, e12_0⟩ := familyGet_eq (k := 1) h12_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 0) h14_0
  obtain ⟨_, e15_0⟩ := familyGet_eq (k := 1) h15_0
  have hloop := iter_sum (fun i ↦ (cellAt w_4_0 (2 + i), cellAt w_4_1 (2 + i), cellAt w_4_2 (2 + i), cellAt w_4_3 (2 + i), ()))
    (fun i current next hbody ↦ prestep470 hbody) h17_3
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_7_0, w_10_0, w_13_0, w_16_0, e5_0, e6_0, e8_0, e9_0, e11_0, e12_0, e14_0, e15_0, hw0, hw1, hw2, hw3]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))

theorem pregather475 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Keys Q0 6}
    (h : W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_475_4 Backend.backend model params o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_475_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_475_4.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  subst e2_0 e4_0 e6_0 e8_0 hout
  rfl

theorem prestep475 {model : HashModel} {params} {o i : Nat} {cur next : Keys Q0 6}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q0 N 1 6}
    (h : W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_475_17 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_475_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_475_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `475`: the sum of its 5 cells. -/
theorem premembers475 {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Keys Q0 6}
    (h : W_Stage_preprocess_0_1.parallel_generatedRoot_475 Backend.backend model params o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess_0_1.parallel_generatedRoot_475.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess_0_1.parallel_generatedRoot_475.constraints_0] at h
  obtain ⟨-, h4_3, h5_0, h6_0, h8_0, h9_0, h11_0, h12_0, h14_0, h15_0, -, h17_3, hout⟩ := h
  have hg : ∀ i : Fin 5, w_4_0 i = f0 (cellIndex o i) ∧
      w_4_1 i = f1 (cellIndex o i) ∧
      w_4_2 i = f2 (cellIndex o i) ∧
      w_4_3 i = f3 (cellIndex o i) := by
    intro i
    have := pregather475 (h4_3 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, -⟩ := this
    exact ⟨a0, a1, a2, a3⟩
  have hw0 : w_4_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_4_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_4_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_4_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2
  obtain ⟨_, e5_0⟩ := familyGet_eq (k := 0) h5_0
  obtain ⟨_, e6_0⟩ := familyGet_eq (k := 1) h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 0) h8_0
  obtain ⟨_, e9_0⟩ := familyGet_eq (k := 1) h9_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 0) h11_0
  obtain ⟨_, e12_0⟩ := familyGet_eq (k := 1) h12_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 0) h14_0
  obtain ⟨_, e15_0⟩ := familyGet_eq (k := 1) h15_0
  have hloop := iter_sum (fun i ↦ (cellAt w_4_0 (2 + i), cellAt w_4_1 (2 + i), cellAt w_4_2 (2 + i), cellAt w_4_3 (2 + i), ()))
    (fun i current next hbody ↦ prestep475 hbody) h17_3
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_7_0, w_10_0, w_13_0, w_16_0, e5_0, e6_0, e8_0, e9_0, e11_0, e12_0, e14_0, e15_0, hw0, hw1, hw2, hw3]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))

theorem pregather514 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_514_4 Backend.backend model params o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_514_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_514_4.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  subst e2_0 e4_0 e6_0 e8_0 hout
  rfl

theorem prestep514 {model : HashModel} {params} {o i : Nat} {cur next : Keys Q5 8}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q5 N 1 8}
    (h : W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_514_17 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_514_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_514_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `514`: the sum of its 5 cells. -/
theorem premembers514 {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess_0_1.parallel_generatedRoot_514 Backend.backend model params o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess_0_1.parallel_generatedRoot_514.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess_0_1.parallel_generatedRoot_514.constraints_0] at h
  obtain ⟨-, h4_3, h5_0, h6_0, h8_0, h9_0, h11_0, h12_0, h14_0, h15_0, -, h17_3, hout⟩ := h
  have hg : ∀ i : Fin 5, w_4_0 i = f0 (cellIndex o i) ∧
      w_4_1 i = f1 (cellIndex o i) ∧
      w_4_2 i = f2 (cellIndex o i) ∧
      w_4_3 i = f3 (cellIndex o i) := by
    intro i
    have := pregather514 (h4_3 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, -⟩ := this
    exact ⟨a0, a1, a2, a3⟩
  have hw0 : w_4_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_4_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_4_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_4_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2
  obtain ⟨_, e5_0⟩ := familyGet_eq (k := 0) h5_0
  obtain ⟨_, e6_0⟩ := familyGet_eq (k := 1) h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 0) h8_0
  obtain ⟨_, e9_0⟩ := familyGet_eq (k := 1) h9_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 0) h11_0
  obtain ⟨_, e12_0⟩ := familyGet_eq (k := 1) h12_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 0) h14_0
  obtain ⟨_, e15_0⟩ := familyGet_eq (k := 1) h15_0
  have hloop := iter_sum (fun i ↦ (cellAt w_4_0 (2 + i), cellAt w_4_1 (2 + i), cellAt w_4_2 (2 + i), cellAt w_4_3 (2 + i), ()))
    (fun i current next hbody ↦ prestep514 hbody) h17_3
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_7_0, w_10_0, w_13_0, w_16_0, e5_0, e6_0, e8_0, e9_0, e11_0, e12_0, e14_0, e15_0, hw0, hw1, hw2, hw3]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))

theorem pregather519 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Keys Q0 6}
    (h : W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_519_4 Backend.backend model params o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_519_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_519_4.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  subst e2_0 e4_0 e6_0 e8_0 hout
  rfl

theorem prestep519 {model : HashModel} {params} {o i : Nat} {cur next : Keys Q0 6}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q0 N 1 6}
    (h : W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_519_17 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_519_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_519_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `519`: the sum of its 5 cells. -/
theorem premembers519 {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Keys Q0 6}
    (h : W_Stage_preprocess_0_1.parallel_generatedRoot_519 Backend.backend model params o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess_0_1.parallel_generatedRoot_519.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess_0_1.parallel_generatedRoot_519.constraints_0] at h
  obtain ⟨-, h4_3, h5_0, h6_0, h8_0, h9_0, h11_0, h12_0, h14_0, h15_0, -, h17_3, hout⟩ := h
  have hg : ∀ i : Fin 5, w_4_0 i = f0 (cellIndex o i) ∧
      w_4_1 i = f1 (cellIndex o i) ∧
      w_4_2 i = f2 (cellIndex o i) ∧
      w_4_3 i = f3 (cellIndex o i) := by
    intro i
    have := pregather519 (h4_3 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, -⟩ := this
    exact ⟨a0, a1, a2, a3⟩
  have hw0 : w_4_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_4_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_4_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_4_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2
  obtain ⟨_, e5_0⟩ := familyGet_eq (k := 0) h5_0
  obtain ⟨_, e6_0⟩ := familyGet_eq (k := 1) h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 0) h8_0
  obtain ⟨_, e9_0⟩ := familyGet_eq (k := 1) h9_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 0) h11_0
  obtain ⟨_, e12_0⟩ := familyGet_eq (k := 1) h12_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 0) h14_0
  obtain ⟨_, e15_0⟩ := familyGet_eq (k := 1) h15_0
  have hloop := iter_sum (fun i ↦ (cellAt w_4_0 (2 + i), cellAt w_4_1 (2 + i), cellAt w_4_2 (2 + i), cellAt w_4_3 (2 + i), ()))
    (fun i current next hbody ↦ prestep519 hbody) h17_3
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_7_0, w_10_0, w_13_0, w_16_0, e5_0, e6_0, e8_0, e9_0, e11_0, e12_0, e14_0, e15_0, hw0, hw1, hw2, hw3]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))

theorem pregather558 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_558_4 Backend.backend model params o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_558_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_558_4.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  subst e2_0 e4_0 e6_0 e8_0 hout
  rfl

theorem prestep558 {model : HashModel} {params} {o i : Nat} {cur next : Keys Q5 8}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q5 N 1 8}
    (h : W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_558_17 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_558_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_558_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `558`: the sum of its 5 cells. -/
theorem premembers558 {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess_0_1.parallel_generatedRoot_558 Backend.backend model params o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess_0_1.parallel_generatedRoot_558.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess_0_1.parallel_generatedRoot_558.constraints_0] at h
  obtain ⟨-, h4_3, h5_0, h6_0, h8_0, h9_0, h11_0, h12_0, h14_0, h15_0, -, h17_3, hout⟩ := h
  have hg : ∀ i : Fin 5, w_4_0 i = f0 (cellIndex o i) ∧
      w_4_1 i = f1 (cellIndex o i) ∧
      w_4_2 i = f2 (cellIndex o i) ∧
      w_4_3 i = f3 (cellIndex o i) := by
    intro i
    have := pregather558 (h4_3 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, -⟩ := this
    exact ⟨a0, a1, a2, a3⟩
  have hw0 : w_4_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_4_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_4_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_4_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2
  obtain ⟨_, e5_0⟩ := familyGet_eq (k := 0) h5_0
  obtain ⟨_, e6_0⟩ := familyGet_eq (k := 1) h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 0) h8_0
  obtain ⟨_, e9_0⟩ := familyGet_eq (k := 1) h9_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 0) h11_0
  obtain ⟨_, e12_0⟩ := familyGet_eq (k := 1) h12_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 0) h14_0
  obtain ⟨_, e15_0⟩ := familyGet_eq (k := 1) h15_0
  have hloop := iter_sum (fun i ↦ (cellAt w_4_0 (2 + i), cellAt w_4_1 (2 + i), cellAt w_4_2 (2 + i), cellAt w_4_3 (2 + i), ()))
    (fun i current next hbody ↦ prestep558 hbody) h17_3
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_7_0, w_10_0, w_13_0, w_16_0, e5_0, e6_0, e8_0, e9_0, e11_0, e12_0, e14_0, e15_0, hw0, hw1, hw2, hw3]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))

theorem pregather563 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Keys Q0 6}
    (h : W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_563_4 Backend.backend model params o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_563_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess_0_1.parallel_parallel_generatedRoot_563_4.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  subst e2_0 e4_0 e6_0 e8_0 hout
  rfl

theorem prestep563 {model : HashModel} {params} {o i : Nat} {cur next : Keys Q0 6}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q0 N 1 6}
    (h : W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_563_17 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_563_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess_0_1.sequential_parallel_generatedRoot_563_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `563`: the sum of its 5 cells. -/
theorem premembers563 {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Keys Q0 6}
    (h : W_Stage_preprocess_0_1.parallel_generatedRoot_563 Backend.backend model params o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess_0_1.parallel_generatedRoot_563.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess_0_1.parallel_generatedRoot_563.constraints_0] at h
  obtain ⟨-, h4_3, h5_0, h6_0, h8_0, h9_0, h11_0, h12_0, h14_0, h15_0, -, h17_3, hout⟩ := h
  have hg : ∀ i : Fin 5, w_4_0 i = f0 (cellIndex o i) ∧
      w_4_1 i = f1 (cellIndex o i) ∧
      w_4_2 i = f2 (cellIndex o i) ∧
      w_4_3 i = f3 (cellIndex o i) := by
    intro i
    have := pregather563 (h4_3 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, -⟩ := this
    exact ⟨a0, a1, a2, a3⟩
  have hw0 : w_4_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_4_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_4_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_4_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2
  obtain ⟨_, e5_0⟩ := familyGet_eq (k := 0) h5_0
  obtain ⟨_, e6_0⟩ := familyGet_eq (k := 1) h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 0) h8_0
  obtain ⟨_, e9_0⟩ := familyGet_eq (k := 1) h9_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 0) h11_0
  obtain ⟨_, e12_0⟩ := familyGet_eq (k := 1) h12_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 0) h14_0
  obtain ⟨_, e15_0⟩ := familyGet_eq (k := 1) h15_0
  have hloop := iter_sum (fun i ↦ (cellAt w_4_0 (2 + i), cellAt w_4_1 (2 + i), cellAt w_4_2 (2 + i), cellAt w_4_3 (2 + i), ()))
    (fun i current next hbody ↦ prestep563 hbody) h17_3
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_7_0, w_10_0, w_13_0, w_16_0, e5_0, e6_0, e8_0, e9_0, e11_0, e12_0, e14_0, e15_0, hw0, hw1, hw2, hw3]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))

end RefreshTopProof.Part_0_1
