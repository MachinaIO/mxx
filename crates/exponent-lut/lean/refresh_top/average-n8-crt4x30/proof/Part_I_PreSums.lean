import TopInst
import TopLemmas

/-! The member sums of the preprocessing's PRF keys (written by `generator/gen_pre_sums.py`). -/

namespace RefreshTopProof.Part_I

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

theorem pregather21 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_21_4 Backend.backend model params i0 o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_21_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_21_4.constraints_0] at h
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

theorem prestep21 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat} {cur next : Keys Q5 8}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q5 N 1 8}
    (h : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_21_17 Backend.backend model params i0 o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_21_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_21_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `21`: the sum of its 5 cells. -/
theorem premembers21 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess.parallel_parallel_generatedRoot_407_21 Backend.backend model params i0 o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_21.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_21.constraints_0] at h
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

theorem pregather36 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 40 → ExactMatrix Q6 N 1 10} {outputs : Keys Q6 10}
    (h : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_36_4 Backend.backend model params i0 o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 8) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_36_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_36_4.constraints_0] at h
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

theorem prestep36 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat} {cur next : Keys Q6 10}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q6 N 1 10}
    (h : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_36_17 Backend.backend model params i0 o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_36_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_36_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `36`: the sum of its 5 cells. -/
theorem premembers36 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o : Fin 8}
    {f0 f1 f2 f3 : Fin 40 → ExactMatrix Q6 N 1 10} {outputs : Keys Q6 10}
    (h : W_Stage_preprocess.parallel_parallel_generatedRoot_407_36 Backend.backend model params i0 o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_36.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_36.constraints_0] at h
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

theorem pregather40 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 40 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_40_4 Backend.backend model params i0 o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 8) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_40_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_40_4.constraints_0] at h
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

theorem prestep40 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat} {cur next : Keys Q5 8}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q5 N 1 8}
    (h : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_40_17 Backend.backend model params i0 o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_40_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_40_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `40`: the sum of its 5 cells. -/
theorem premembers40 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o : Fin 8}
    {f0 f1 f2 f3 : Fin 40 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess.parallel_parallel_generatedRoot_407_40 Backend.backend model params i0 o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_40.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_40.constraints_0] at h
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

theorem pregather211 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 240 → ExactMatrix Q6 N 1 10} {outputs : Keys Q6 10}
    (h : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_213_4 Backend.backend model params i0 o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 48) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_213_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_213_4.constraints_0] at h
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

theorem prestep211 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat} {cur next : Keys Q6 10}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q6 N 1 10}
    (h : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_213_17 Backend.backend model params i0 o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_213_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_213_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `211`: the sum of its 5 cells. -/
theorem premembers211 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o : Fin 48}
    {f0 f1 f2 f3 : Fin 240 → ExactMatrix Q6 N 1 10} {outputs : Keys Q6 10}
    (h : W_Stage_preprocess.parallel_parallel_generatedRoot_407_213 Backend.backend model params i0 o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_213.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_213.constraints_0] at h
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

theorem pregather215 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 240 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_217_4 Backend.backend model params i0 o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 48) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_217_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_217_4.constraints_0] at h
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

theorem prestep215 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat} {cur next : Keys Q5 8}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q5 N 1 8}
    (h : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_217_17 Backend.backend model params i0 o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_217_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_217_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `215`: the sum of its 5 cells. -/
theorem premembers215 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o : Fin 48}
    {f0 f1 f2 f3 : Fin 240 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess.parallel_parallel_generatedRoot_407_217 Backend.backend model params i0 o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_217.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_217.constraints_0] at h
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

theorem pregather322 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Keys Q0 6}
    (h : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_324_4 Backend.backend model params i0 o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_324_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_324_4.constraints_0] at h
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

theorem prestep322 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat} {cur next : Keys Q0 6}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q0 N 1 6}
    (h : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_324_17 Backend.backend model params i0 o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_324_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_324_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `322`: the sum of its 5 cells. -/
theorem premembers322 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Keys Q0 6}
    (h : W_Stage_preprocess.parallel_parallel_generatedRoot_407_324 Backend.backend model params i0 o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_324.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_324.constraints_0] at h
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

theorem pregather382 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_386_4 Backend.backend model params i0 o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_386_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_386_4.constraints_0] at h
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

theorem prestep382 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat} {cur next : Keys Q5 8}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q5 N 1 8}
    (h : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_386_17 Backend.backend model params i0 o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_386_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_386_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `382`: the sum of its 5 cells. -/
theorem premembers382 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess.parallel_parallel_generatedRoot_407_386 Backend.backend model params i0 o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_386.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_386.constraints_0] at h
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

theorem pregather387 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Keys Q0 6}
    (h : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_391_4 Backend.backend model params i0 o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_391_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_391_4.constraints_0] at h
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

theorem prestep387 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat} {cur next : Keys Q0 6}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q0 N 1 6}
    (h : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_391_17 Backend.backend model params i0 o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_391_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_391_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `387`: the sum of its 5 cells. -/
theorem premembers387 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Keys Q0 6}
    (h : W_Stage_preprocess.parallel_parallel_generatedRoot_407_391 Backend.backend model params i0 o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_391.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_391.constraints_0] at h
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

theorem pregather426 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_430_4 Backend.backend model params i0 o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_430_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_430_4.constraints_0] at h
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

theorem prestep426 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat} {cur next : Keys Q5 8}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q5 N 1 8}
    (h : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_430_17 Backend.backend model params i0 o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_430_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_430_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `426`: the sum of its 5 cells. -/
theorem premembers426 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess.parallel_parallel_generatedRoot_407_430 Backend.backend model params i0 o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_430.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_430.constraints_0] at h
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

theorem pregather431 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Keys Q0 6}
    (h : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_435_4 Backend.backend model params i0 o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_435_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_435_4.constraints_0] at h
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

theorem prestep431 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat} {cur next : Keys Q0 6}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q0 N 1 6}
    (h : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_435_17 Backend.backend model params i0 o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_435_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_435_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `431`: the sum of its 5 cells. -/
theorem premembers431 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Keys Q0 6}
    (h : W_Stage_preprocess.parallel_parallel_generatedRoot_407_435 Backend.backend model params i0 o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_435.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_435.constraints_0] at h
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

theorem pregather470 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_474_4 Backend.backend model params i0 o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_474_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_474_4.constraints_0] at h
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

theorem prestep470 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat} {cur next : Keys Q5 8}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q5 N 1 8}
    (h : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_474_17 Backend.backend model params i0 o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_474_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_474_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `470`: the sum of its 5 cells. -/
theorem premembers470 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess.parallel_parallel_generatedRoot_407_474 Backend.backend model params i0 o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_474.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_474.constraints_0] at h
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

theorem pregather475 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Keys Q0 6}
    (h : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_479_4 Backend.backend model params i0 o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_479_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_479_4.constraints_0] at h
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

theorem prestep475 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat} {cur next : Keys Q0 6}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q0 N 1 6}
    (h : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_479_17 Backend.backend model params i0 o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_479_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_479_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `475`: the sum of its 5 cells. -/
theorem premembers475 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Keys Q0 6}
    (h : W_Stage_preprocess.parallel_parallel_generatedRoot_407_479 Backend.backend model params i0 o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_479.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_479.constraints_0] at h
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

theorem pregather514 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_518_4 Backend.backend model params i0 o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_518_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_518_4.constraints_0] at h
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

theorem prestep514 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat} {cur next : Keys Q5 8}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q5 N 1 8}
    (h : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_518_17 Backend.backend model params i0 o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_518_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_518_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `514`: the sum of its 5 cells. -/
theorem premembers514 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess.parallel_parallel_generatedRoot_407_518 Backend.backend model params i0 o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_518.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_518.constraints_0] at h
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

theorem pregather519 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Keys Q0 6}
    (h : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_523_4 Backend.backend model params i0 o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_523_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_523_4.constraints_0] at h
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

theorem prestep519 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat} {cur next : Keys Q0 6}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q0 N 1 6}
    (h : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_523_17 Backend.backend model params i0 o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_523_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_523_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `519`: the sum of its 5 cells. -/
theorem premembers519 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Keys Q0 6}
    (h : W_Stage_preprocess.parallel_parallel_generatedRoot_407_523 Backend.backend model params i0 o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_523.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_523.constraints_0] at h
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

theorem pregather558 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_562_4 Backend.backend model params i0 o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_562_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_562_4.constraints_0] at h
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

theorem prestep558 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat} {cur next : Keys Q5 8}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q5 N 1 8}
    (h : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_562_17 Backend.backend model params i0 o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_562_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_562_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `558`: the sum of its 5 cells. -/
theorem premembers558 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Keys Q5 8}
    (h : W_Stage_preprocess.parallel_parallel_generatedRoot_407_562 Backend.backend model params i0 o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_562.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_562.constraints_0] at h
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

theorem pregather563 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Keys Q0 6}
    (h : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_567_4 Backend.backend model params i0 o i
      (f0, f1, f2, f3, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_567_4 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  simp only [W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_407_567_4.constraints_0] at h
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

theorem prestep563 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o i : Nat} {cur next : Keys Q0 6}
    {g0 g1 g2 g3 : Fin 5 → ExactMatrix Q0 N 1 6}
    (h : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_567_17 Backend.backend model params i0 o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, g0, g1, g2, g3,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), ()) := by
  unfold W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_567_17 at h
  extract_lets -merge _params _backend w_0_0 w_5_0 w_9_0 w_13_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_7_0, h⟩ := h
  extract_lets -merge w_8_0 at h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 at h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  simp only [W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_407_567_17.constraints_0] at h
  obtain ⟨h2_0, h2_0_1, h3_0, h2_0_2, h2_0_3, h7_0, h2_0_4, h2_0_5, h11_0, h2_0_6, h2_0_7, h15_0, hout⟩ := h
  have e0 := familyGet_cellAt h3_0
  have e1 := familyGet_cellAt h7_0
  have e2 := familyGet_cellAt h11_0
  have e3 := familyGet_cellAt h15_0
  subst e0 e1 e2 e3 hout
  rfl

set_option maxRecDepth 65536 in
/-- Member `o` of the sum `563`: the sum of its 5 cells. -/
theorem premembers563 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Keys Q0 6}
    (h : W_Stage_preprocess.parallel_parallel_generatedRoot_407_567 Backend.backend model params i0 o
      (f0, f1, f2, f3, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_567.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 at h
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_567.constraints_0] at h
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

end RefreshTopProof.Part_I
