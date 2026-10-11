import TopLemmas

/-!
# The member sums of the online evaluation (written by `generator/gen_sums.py`)

Every PRF bucket sums the cells of each member: member `o`'s cells `w o, …, w o + w - 1` of the
bucket's cell family are gathered, the first two added, and the rest added by a loop.
-/

namespace RefreshTopProof.Part_0_1

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

theorem gather116 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 f4 f5 : Fin 40 → ExactMatrix Q6 N 1 10} {outputs : Blocks Q6 10}
    (h : W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_116_6 Backend.backend model params o i
      (f0, f1, f2, f3, f4, f5, ()) outputs) (ho : o < 8) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), f4 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), f5 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_116_6 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  simp only [W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_116_6.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, h1_0_8, h1_0_9, h10_0, h1_0_10, h1_0_11, h12_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0 h10_0 h12_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq h10_0
  obtain ⟨_, e12_0⟩ := familyGet_eq h12_0
  subst e2_0 e4_0 e6_0 e8_0 e10_0 e12_0 hout
  rfl

theorem step116 {model : HashModel} {params} {o i : Nat} {cur next : Blocks Q6 10}
    {g0 g1 g2 g3 g4 g5 : Fin 5 → ExactMatrix Q6 N 1 10}
    (h : W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_116_25 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_116_25 at h
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
  simp only [W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_116_25.constraints_0] at h
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
/-- Member `o` of the sum `116`: the sum of its 5 cells. -/
theorem members116 {model : HashModel} {params} {o : Fin 8}
    {f0 f1 f2 f3 f4 f5 : Fin 40 → ExactMatrix Q6 N 1 10} {outputs : Blocks Q6 10}
    (h : W_Stage_online_eval_0_1.parallel_generatedRoot_116 Backend.backend model params o
      (f0, f1, f2, f3, f4, f5, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ∑ k : Fin 5, f4 (cellIndex o k), ∑ k : Fin 5, f5 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval_0_1.parallel_generatedRoot_116.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 at h
  simp only [W_Stage_online_eval_0_1.parallel_generatedRoot_116.constraints_0] at h
  obtain ⟨-, h6_5, h7_0, h8_0, h10_0, h11_0, h13_0, h14_0, h16_0, h17_0, h19_0, h20_0, h22_0, h23_0, -, h25_5, hout⟩ := h
  have hg : ∀ i : Fin 5, w_6_0 i = f0 (cellIndex o i) ∧
      w_6_1 i = f1 (cellIndex o i) ∧
      w_6_2 i = f2 (cellIndex o i) ∧
      w_6_3 i = f3 (cellIndex o i) ∧
      w_6_4 i = f4 (cellIndex o i) ∧
      w_6_5 i = f5 (cellIndex o i) := by
    intro i
    have := gather116 (h6_5 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, a4, a5, -⟩ := this
    exact ⟨a0, a1, a2, a3, a4, a5⟩
  have hw0 : w_6_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_6_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_6_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_6_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.1
  have hw4 : w_6_4 = fun i ↦ f4 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.1
  have hw5 : w_6_5 = fun i ↦ f5 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.2
  obtain ⟨_, e7_0⟩ := familyGet_eq (k := 0) h7_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 1) h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq (k := 0) h10_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 1) h11_0
  obtain ⟨_, e13_0⟩ := familyGet_eq (k := 0) h13_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 1) h14_0
  obtain ⟨_, e16_0⟩ := familyGet_eq (k := 0) h16_0
  obtain ⟨_, e17_0⟩ := familyGet_eq (k := 1) h17_0
  obtain ⟨_, e19_0⟩ := familyGet_eq (k := 0) h19_0
  obtain ⟨_, e20_0⟩ := familyGet_eq (k := 1) h20_0
  obtain ⟨_, e22_0⟩ := familyGet_eq (k := 0) h22_0
  obtain ⟨_, e23_0⟩ := familyGet_eq (k := 1) h23_0
  have hloop := iter_sum (fun i ↦ (cellAt w_6_0 (2 + i), cellAt w_6_1 (2 + i), cellAt w_6_2 (2 + i), cellAt w_6_3 (2 + i), cellAt w_6_4 (2 + i), cellAt w_6_5 (2 + i), ()))
    (fun i current next hbody ↦ step116 hbody) h25_5
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_9_0, w_12_0, w_15_0, w_18_0, w_21_0, w_24_0, e7_0, e8_0, e10_0, e11_0, e13_0, e14_0, e16_0, e17_0, e19_0, e20_0, e22_0, e23_0, hw0, hw1, hw2, hw3, hw4, hw5]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f4 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f5 (cellIndex o k))

theorem gather120 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 f4 f5 : Fin 40 → ExactMatrix Q5 N 1 8} {outputs : Blocks Q5 8}
    (h : W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_120_6 Backend.backend model params o i
      (f0, f1, f2, f3, f4, f5, ()) outputs) (ho : o < 8) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), f4 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), f5 (cellIndex (⟨o, ho⟩ : Fin 8) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_120_6 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  simp only [W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_120_6.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, h1_0_8, h1_0_9, h10_0, h1_0_10, h1_0_11, h12_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0 h10_0 h12_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq h10_0
  obtain ⟨_, e12_0⟩ := familyGet_eq h12_0
  subst e2_0 e4_0 e6_0 e8_0 e10_0 e12_0 hout
  rfl

theorem step120 {model : HashModel} {params} {o i : Nat} {cur next : Blocks Q5 8}
    {g0 g1 g2 g3 g4 g5 : Fin 5 → ExactMatrix Q5 N 1 8}
    (h : W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_120_25 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_120_25 at h
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
  simp only [W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_120_25.constraints_0] at h
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
/-- Member `o` of the sum `120`: the sum of its 5 cells. -/
theorem members120 {model : HashModel} {params} {o : Fin 8}
    {f0 f1 f2 f3 f4 f5 : Fin 40 → ExactMatrix Q5 N 1 8} {outputs : Blocks Q5 8}
    (h : W_Stage_online_eval_0_1.parallel_generatedRoot_120 Backend.backend model params o
      (f0, f1, f2, f3, f4, f5, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ∑ k : Fin 5, f4 (cellIndex o k), ∑ k : Fin 5, f5 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval_0_1.parallel_generatedRoot_120.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 at h
  simp only [W_Stage_online_eval_0_1.parallel_generatedRoot_120.constraints_0] at h
  obtain ⟨-, h6_5, h7_0, h8_0, h10_0, h11_0, h13_0, h14_0, h16_0, h17_0, h19_0, h20_0, h22_0, h23_0, -, h25_5, hout⟩ := h
  have hg : ∀ i : Fin 5, w_6_0 i = f0 (cellIndex o i) ∧
      w_6_1 i = f1 (cellIndex o i) ∧
      w_6_2 i = f2 (cellIndex o i) ∧
      w_6_3 i = f3 (cellIndex o i) ∧
      w_6_4 i = f4 (cellIndex o i) ∧
      w_6_5 i = f5 (cellIndex o i) := by
    intro i
    have := gather120 (h6_5 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, a4, a5, -⟩ := this
    exact ⟨a0, a1, a2, a3, a4, a5⟩
  have hw0 : w_6_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_6_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_6_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_6_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.1
  have hw4 : w_6_4 = fun i ↦ f4 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.1
  have hw5 : w_6_5 = fun i ↦ f5 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.2
  obtain ⟨_, e7_0⟩ := familyGet_eq (k := 0) h7_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 1) h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq (k := 0) h10_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 1) h11_0
  obtain ⟨_, e13_0⟩ := familyGet_eq (k := 0) h13_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 1) h14_0
  obtain ⟨_, e16_0⟩ := familyGet_eq (k := 0) h16_0
  obtain ⟨_, e17_0⟩ := familyGet_eq (k := 1) h17_0
  obtain ⟨_, e19_0⟩ := familyGet_eq (k := 0) h19_0
  obtain ⟨_, e20_0⟩ := familyGet_eq (k := 1) h20_0
  obtain ⟨_, e22_0⟩ := familyGet_eq (k := 0) h22_0
  obtain ⟨_, e23_0⟩ := familyGet_eq (k := 1) h23_0
  have hloop := iter_sum (fun i ↦ (cellAt w_6_0 (2 + i), cellAt w_6_1 (2 + i), cellAt w_6_2 (2 + i), cellAt w_6_3 (2 + i), cellAt w_6_4 (2 + i), cellAt w_6_5 (2 + i), ()))
    (fun i current next hbody ↦ step120 hbody) h25_5
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_9_0, w_12_0, w_15_0, w_18_0, w_21_0, w_24_0, e7_0, e8_0, e10_0, e11_0, e13_0, e14_0, e16_0, e17_0, e19_0, e20_0, e22_0, e23_0, hw0, hw1, hw2, hw3, hw4, hw5]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f4 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f5 (cellIndex o k))

theorem gather387 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 f4 f5 : Fin 240 → ExactMatrix Q6 N 1 10} {outputs : Blocks Q6 10}
    (h : W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_387_6 Backend.backend model params o i
      (f0, f1, f2, f3, f4, f5, ()) outputs) (ho : o < 48) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), f4 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), f5 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_387_6 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  simp only [W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_387_6.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, h1_0_8, h1_0_9, h10_0, h1_0_10, h1_0_11, h12_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0 h10_0 h12_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq h10_0
  obtain ⟨_, e12_0⟩ := familyGet_eq h12_0
  subst e2_0 e4_0 e6_0 e8_0 e10_0 e12_0 hout
  rfl

theorem step387 {model : HashModel} {params} {o i : Nat} {cur next : Blocks Q6 10}
    {g0 g1 g2 g3 g4 g5 : Fin 5 → ExactMatrix Q6 N 1 10}
    (h : W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_387_25 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_387_25 at h
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
  simp only [W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_387_25.constraints_0] at h
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
/-- Member `o` of the sum `387`: the sum of its 5 cells. -/
theorem members387 {model : HashModel} {params} {o : Fin 48}
    {f0 f1 f2 f3 f4 f5 : Fin 240 → ExactMatrix Q6 N 1 10} {outputs : Blocks Q6 10}
    (h : W_Stage_online_eval_0_1.parallel_generatedRoot_387 Backend.backend model params o
      (f0, f1, f2, f3, f4, f5, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ∑ k : Fin 5, f4 (cellIndex o k), ∑ k : Fin 5, f5 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval_0_1.parallel_generatedRoot_387.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 at h
  simp only [W_Stage_online_eval_0_1.parallel_generatedRoot_387.constraints_0] at h
  obtain ⟨-, h6_5, h7_0, h8_0, h10_0, h11_0, h13_0, h14_0, h16_0, h17_0, h19_0, h20_0, h22_0, h23_0, -, h25_5, hout⟩ := h
  have hg : ∀ i : Fin 5, w_6_0 i = f0 (cellIndex o i) ∧
      w_6_1 i = f1 (cellIndex o i) ∧
      w_6_2 i = f2 (cellIndex o i) ∧
      w_6_3 i = f3 (cellIndex o i) ∧
      w_6_4 i = f4 (cellIndex o i) ∧
      w_6_5 i = f5 (cellIndex o i) := by
    intro i
    have := gather387 (h6_5 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, a4, a5, -⟩ := this
    exact ⟨a0, a1, a2, a3, a4, a5⟩
  have hw0 : w_6_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_6_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_6_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_6_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.1
  have hw4 : w_6_4 = fun i ↦ f4 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.1
  have hw5 : w_6_5 = fun i ↦ f5 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.2
  obtain ⟨_, e7_0⟩ := familyGet_eq (k := 0) h7_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 1) h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq (k := 0) h10_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 1) h11_0
  obtain ⟨_, e13_0⟩ := familyGet_eq (k := 0) h13_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 1) h14_0
  obtain ⟨_, e16_0⟩ := familyGet_eq (k := 0) h16_0
  obtain ⟨_, e17_0⟩ := familyGet_eq (k := 1) h17_0
  obtain ⟨_, e19_0⟩ := familyGet_eq (k := 0) h19_0
  obtain ⟨_, e20_0⟩ := familyGet_eq (k := 1) h20_0
  obtain ⟨_, e22_0⟩ := familyGet_eq (k := 0) h22_0
  obtain ⟨_, e23_0⟩ := familyGet_eq (k := 1) h23_0
  have hloop := iter_sum (fun i ↦ (cellAt w_6_0 (2 + i), cellAt w_6_1 (2 + i), cellAt w_6_2 (2 + i), cellAt w_6_3 (2 + i), cellAt w_6_4 (2 + i), cellAt w_6_5 (2 + i), ()))
    (fun i current next hbody ↦ step387 hbody) h25_5
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_9_0, w_12_0, w_15_0, w_18_0, w_21_0, w_24_0, e7_0, e8_0, e10_0, e11_0, e13_0, e14_0, e16_0, e17_0, e19_0, e20_0, e22_0, e23_0, hw0, hw1, hw2, hw3, hw4, hw5]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f4 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f5 (cellIndex o k))

theorem gather391 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 f4 f5 : Fin 240 → ExactMatrix Q5 N 1 8} {outputs : Blocks Q5 8}
    (h : W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_391_6 Backend.backend model params o i
      (f0, f1, f2, f3, f4, f5, ()) outputs) (ho : o < 48) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), f4 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), f5 (cellIndex (⟨o, ho⟩ : Fin 48) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_391_6 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  simp only [W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_391_6.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, h1_0_8, h1_0_9, h10_0, h1_0_10, h1_0_11, h12_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0 h10_0 h12_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq h10_0
  obtain ⟨_, e12_0⟩ := familyGet_eq h12_0
  subst e2_0 e4_0 e6_0 e8_0 e10_0 e12_0 hout
  rfl

theorem step391 {model : HashModel} {params} {o i : Nat} {cur next : Blocks Q5 8}
    {g0 g1 g2 g3 g4 g5 : Fin 5 → ExactMatrix Q5 N 1 8}
    (h : W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_391_25 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_391_25 at h
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
  simp only [W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_391_25.constraints_0] at h
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
/-- Member `o` of the sum `391`: the sum of its 5 cells. -/
theorem members391 {model : HashModel} {params} {o : Fin 48}
    {f0 f1 f2 f3 f4 f5 : Fin 240 → ExactMatrix Q5 N 1 8} {outputs : Blocks Q5 8}
    (h : W_Stage_online_eval_0_1.parallel_generatedRoot_391 Backend.backend model params o
      (f0, f1, f2, f3, f4, f5, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ∑ k : Fin 5, f4 (cellIndex o k), ∑ k : Fin 5, f5 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval_0_1.parallel_generatedRoot_391.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 at h
  simp only [W_Stage_online_eval_0_1.parallel_generatedRoot_391.constraints_0] at h
  obtain ⟨-, h6_5, h7_0, h8_0, h10_0, h11_0, h13_0, h14_0, h16_0, h17_0, h19_0, h20_0, h22_0, h23_0, -, h25_5, hout⟩ := h
  have hg : ∀ i : Fin 5, w_6_0 i = f0 (cellIndex o i) ∧
      w_6_1 i = f1 (cellIndex o i) ∧
      w_6_2 i = f2 (cellIndex o i) ∧
      w_6_3 i = f3 (cellIndex o i) ∧
      w_6_4 i = f4 (cellIndex o i) ∧
      w_6_5 i = f5 (cellIndex o i) := by
    intro i
    have := gather391 (h6_5 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, a4, a5, -⟩ := this
    exact ⟨a0, a1, a2, a3, a4, a5⟩
  have hw0 : w_6_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_6_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_6_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_6_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.1
  have hw4 : w_6_4 = fun i ↦ f4 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.1
  have hw5 : w_6_5 = fun i ↦ f5 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.2
  obtain ⟨_, e7_0⟩ := familyGet_eq (k := 0) h7_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 1) h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq (k := 0) h10_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 1) h11_0
  obtain ⟨_, e13_0⟩ := familyGet_eq (k := 0) h13_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 1) h14_0
  obtain ⟨_, e16_0⟩ := familyGet_eq (k := 0) h16_0
  obtain ⟨_, e17_0⟩ := familyGet_eq (k := 1) h17_0
  obtain ⟨_, e19_0⟩ := familyGet_eq (k := 0) h19_0
  obtain ⟨_, e20_0⟩ := familyGet_eq (k := 1) h20_0
  obtain ⟨_, e22_0⟩ := familyGet_eq (k := 0) h22_0
  obtain ⟨_, e23_0⟩ := familyGet_eq (k := 1) h23_0
  have hloop := iter_sum (fun i ↦ (cellAt w_6_0 (2 + i), cellAt w_6_1 (2 + i), cellAt w_6_2 (2 + i), cellAt w_6_3 (2 + i), cellAt w_6_4 (2 + i), cellAt w_6_5 (2 + i), ()))
    (fun i current next hbody ↦ step391 hbody) h25_5
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_9_0, w_12_0, w_15_0, w_18_0, w_21_0, w_24_0, e7_0, e8_0, e10_0, e11_0, e13_0, e14_0, e16_0, e17_0, e19_0, e20_0, e22_0, e23_0, hw0, hw1, hw2, hw3, hw4, hw5]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f4 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f5 (cellIndex o k))

theorem gather555 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Blocks Q5 8}
    (h : W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_555_6 Backend.backend model params o i
      (f0, f1, f2, f3, f4, f5, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f4 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f5 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_555_6 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  simp only [W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_555_6.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, h1_0_8, h1_0_9, h10_0, h1_0_10, h1_0_11, h12_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0 h10_0 h12_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq h10_0
  obtain ⟨_, e12_0⟩ := familyGet_eq h12_0
  subst e2_0 e4_0 e6_0 e8_0 e10_0 e12_0 hout
  rfl

theorem step555 {model : HashModel} {params} {o i : Nat} {cur next : Blocks Q5 8}
    {g0 g1 g2 g3 g4 g5 : Fin 5 → ExactMatrix Q5 N 1 8}
    (h : W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_555_25 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_555_25 at h
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
  simp only [W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_555_25.constraints_0] at h
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
/-- Member `o` of the sum `555`: the sum of its 5 cells. -/
theorem members555 {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Blocks Q5 8}
    (h : W_Stage_online_eval_0_1.parallel_generatedRoot_555 Backend.backend model params o
      (f0, f1, f2, f3, f4, f5, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ∑ k : Fin 5, f4 (cellIndex o k), ∑ k : Fin 5, f5 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval_0_1.parallel_generatedRoot_555.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 at h
  simp only [W_Stage_online_eval_0_1.parallel_generatedRoot_555.constraints_0] at h
  obtain ⟨-, h6_5, h7_0, h8_0, h10_0, h11_0, h13_0, h14_0, h16_0, h17_0, h19_0, h20_0, h22_0, h23_0, -, h25_5, hout⟩ := h
  have hg : ∀ i : Fin 5, w_6_0 i = f0 (cellIndex o i) ∧
      w_6_1 i = f1 (cellIndex o i) ∧
      w_6_2 i = f2 (cellIndex o i) ∧
      w_6_3 i = f3 (cellIndex o i) ∧
      w_6_4 i = f4 (cellIndex o i) ∧
      w_6_5 i = f5 (cellIndex o i) := by
    intro i
    have := gather555 (h6_5 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, a4, a5, -⟩ := this
    exact ⟨a0, a1, a2, a3, a4, a5⟩
  have hw0 : w_6_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_6_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_6_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_6_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.1
  have hw4 : w_6_4 = fun i ↦ f4 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.1
  have hw5 : w_6_5 = fun i ↦ f5 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.2
  obtain ⟨_, e7_0⟩ := familyGet_eq (k := 0) h7_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 1) h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq (k := 0) h10_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 1) h11_0
  obtain ⟨_, e13_0⟩ := familyGet_eq (k := 0) h13_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 1) h14_0
  obtain ⟨_, e16_0⟩ := familyGet_eq (k := 0) h16_0
  obtain ⟨_, e17_0⟩ := familyGet_eq (k := 1) h17_0
  obtain ⟨_, e19_0⟩ := familyGet_eq (k := 0) h19_0
  obtain ⟨_, e20_0⟩ := familyGet_eq (k := 1) h20_0
  obtain ⟨_, e22_0⟩ := familyGet_eq (k := 0) h22_0
  obtain ⟨_, e23_0⟩ := familyGet_eq (k := 1) h23_0
  have hloop := iter_sum (fun i ↦ (cellAt w_6_0 (2 + i), cellAt w_6_1 (2 + i), cellAt w_6_2 (2 + i), cellAt w_6_3 (2 + i), cellAt w_6_4 (2 + i), cellAt w_6_5 (2 + i), ()))
    (fun i current next hbody ↦ step555 hbody) h25_5
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_9_0, w_12_0, w_15_0, w_18_0, w_21_0, w_24_0, e7_0, e8_0, e10_0, e11_0, e13_0, e14_0, e16_0, e17_0, e19_0, e20_0, e22_0, e23_0, hw0, hw1, hw2, hw3, hw4, hw5]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f4 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f5 (cellIndex o k))

theorem gather559 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Blocks Q0 6}
    (h : W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_559_6 Backend.backend model params o i
      (f0, f1, f2, f3, f4, f5, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f4 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f5 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_559_6 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  simp only [W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_559_6.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, h1_0_8, h1_0_9, h10_0, h1_0_10, h1_0_11, h12_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0 h10_0 h12_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq h10_0
  obtain ⟨_, e12_0⟩ := familyGet_eq h12_0
  subst e2_0 e4_0 e6_0 e8_0 e10_0 e12_0 hout
  rfl

theorem step559 {model : HashModel} {params} {o i : Nat} {cur next : Blocks Q0 6}
    {g0 g1 g2 g3 g4 g5 : Fin 5 → ExactMatrix Q0 N 1 6}
    (h : W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_559_25 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_559_25 at h
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
  simp only [W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_559_25.constraints_0] at h
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
/-- Member `o` of the sum `559`: the sum of its 5 cells. -/
theorem members559 {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Blocks Q0 6}
    (h : W_Stage_online_eval_0_1.parallel_generatedRoot_559 Backend.backend model params o
      (f0, f1, f2, f3, f4, f5, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ∑ k : Fin 5, f4 (cellIndex o k), ∑ k : Fin 5, f5 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval_0_1.parallel_generatedRoot_559.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 at h
  simp only [W_Stage_online_eval_0_1.parallel_generatedRoot_559.constraints_0] at h
  obtain ⟨-, h6_5, h7_0, h8_0, h10_0, h11_0, h13_0, h14_0, h16_0, h17_0, h19_0, h20_0, h22_0, h23_0, -, h25_5, hout⟩ := h
  have hg : ∀ i : Fin 5, w_6_0 i = f0 (cellIndex o i) ∧
      w_6_1 i = f1 (cellIndex o i) ∧
      w_6_2 i = f2 (cellIndex o i) ∧
      w_6_3 i = f3 (cellIndex o i) ∧
      w_6_4 i = f4 (cellIndex o i) ∧
      w_6_5 i = f5 (cellIndex o i) := by
    intro i
    have := gather559 (h6_5 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, a4, a5, -⟩ := this
    exact ⟨a0, a1, a2, a3, a4, a5⟩
  have hw0 : w_6_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_6_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_6_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_6_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.1
  have hw4 : w_6_4 = fun i ↦ f4 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.1
  have hw5 : w_6_5 = fun i ↦ f5 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.2
  obtain ⟨_, e7_0⟩ := familyGet_eq (k := 0) h7_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 1) h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq (k := 0) h10_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 1) h11_0
  obtain ⟨_, e13_0⟩ := familyGet_eq (k := 0) h13_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 1) h14_0
  obtain ⟨_, e16_0⟩ := familyGet_eq (k := 0) h16_0
  obtain ⟨_, e17_0⟩ := familyGet_eq (k := 1) h17_0
  obtain ⟨_, e19_0⟩ := familyGet_eq (k := 0) h19_0
  obtain ⟨_, e20_0⟩ := familyGet_eq (k := 1) h20_0
  obtain ⟨_, e22_0⟩ := familyGet_eq (k := 0) h22_0
  obtain ⟨_, e23_0⟩ := familyGet_eq (k := 1) h23_0
  have hloop := iter_sum (fun i ↦ (cellAt w_6_0 (2 + i), cellAt w_6_1 (2 + i), cellAt w_6_2 (2 + i), cellAt w_6_3 (2 + i), cellAt w_6_4 (2 + i), cellAt w_6_5 (2 + i), ()))
    (fun i current next hbody ↦ step559 hbody) h25_5
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_9_0, w_12_0, w_15_0, w_18_0, w_21_0, w_24_0, e7_0, e8_0, e10_0, e11_0, e13_0, e14_0, e16_0, e17_0, e19_0, e20_0, e22_0, e23_0, hw0, hw1, hw2, hw3, hw4, hw5]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f4 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f5 (cellIndex o k))

theorem gather610 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Blocks Q5 8}
    (h : W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_610_6 Backend.backend model params o i
      (f0, f1, f2, f3, f4, f5, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f4 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f5 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_610_6 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  simp only [W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_610_6.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, h1_0_8, h1_0_9, h10_0, h1_0_10, h1_0_11, h12_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0 h10_0 h12_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq h10_0
  obtain ⟨_, e12_0⟩ := familyGet_eq h12_0
  subst e2_0 e4_0 e6_0 e8_0 e10_0 e12_0 hout
  rfl

theorem step610 {model : HashModel} {params} {o i : Nat} {cur next : Blocks Q5 8}
    {g0 g1 g2 g3 g4 g5 : Fin 5 → ExactMatrix Q5 N 1 8}
    (h : W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_610_25 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_610_25 at h
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
  simp only [W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_610_25.constraints_0] at h
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
/-- Member `o` of the sum `610`: the sum of its 5 cells. -/
theorem members610 {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Blocks Q5 8}
    (h : W_Stage_online_eval_0_1.parallel_generatedRoot_610 Backend.backend model params o
      (f0, f1, f2, f3, f4, f5, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ∑ k : Fin 5, f4 (cellIndex o k), ∑ k : Fin 5, f5 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval_0_1.parallel_generatedRoot_610.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 at h
  simp only [W_Stage_online_eval_0_1.parallel_generatedRoot_610.constraints_0] at h
  obtain ⟨-, h6_5, h7_0, h8_0, h10_0, h11_0, h13_0, h14_0, h16_0, h17_0, h19_0, h20_0, h22_0, h23_0, -, h25_5, hout⟩ := h
  have hg : ∀ i : Fin 5, w_6_0 i = f0 (cellIndex o i) ∧
      w_6_1 i = f1 (cellIndex o i) ∧
      w_6_2 i = f2 (cellIndex o i) ∧
      w_6_3 i = f3 (cellIndex o i) ∧
      w_6_4 i = f4 (cellIndex o i) ∧
      w_6_5 i = f5 (cellIndex o i) := by
    intro i
    have := gather610 (h6_5 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, a4, a5, -⟩ := this
    exact ⟨a0, a1, a2, a3, a4, a5⟩
  have hw0 : w_6_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_6_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_6_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_6_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.1
  have hw4 : w_6_4 = fun i ↦ f4 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.1
  have hw5 : w_6_5 = fun i ↦ f5 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.2
  obtain ⟨_, e7_0⟩ := familyGet_eq (k := 0) h7_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 1) h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq (k := 0) h10_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 1) h11_0
  obtain ⟨_, e13_0⟩ := familyGet_eq (k := 0) h13_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 1) h14_0
  obtain ⟨_, e16_0⟩ := familyGet_eq (k := 0) h16_0
  obtain ⟨_, e17_0⟩ := familyGet_eq (k := 1) h17_0
  obtain ⟨_, e19_0⟩ := familyGet_eq (k := 0) h19_0
  obtain ⟨_, e20_0⟩ := familyGet_eq (k := 1) h20_0
  obtain ⟨_, e22_0⟩ := familyGet_eq (k := 0) h22_0
  obtain ⟨_, e23_0⟩ := familyGet_eq (k := 1) h23_0
  have hloop := iter_sum (fun i ↦ (cellAt w_6_0 (2 + i), cellAt w_6_1 (2 + i), cellAt w_6_2 (2 + i), cellAt w_6_3 (2 + i), cellAt w_6_4 (2 + i), cellAt w_6_5 (2 + i), ()))
    (fun i current next hbody ↦ step610 hbody) h25_5
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_9_0, w_12_0, w_15_0, w_18_0, w_21_0, w_24_0, e7_0, e8_0, e10_0, e11_0, e13_0, e14_0, e16_0, e17_0, e19_0, e20_0, e22_0, e23_0, hw0, hw1, hw2, hw3, hw4, hw5]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f4 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f5 (cellIndex o k))

theorem gather614 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Blocks Q0 6}
    (h : W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_614_6 Backend.backend model params o i
      (f0, f1, f2, f3, f4, f5, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f4 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f5 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_614_6 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  simp only [W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_614_6.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, h1_0_8, h1_0_9, h10_0, h1_0_10, h1_0_11, h12_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0 h10_0 h12_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq h10_0
  obtain ⟨_, e12_0⟩ := familyGet_eq h12_0
  subst e2_0 e4_0 e6_0 e8_0 e10_0 e12_0 hout
  rfl

theorem step614 {model : HashModel} {params} {o i : Nat} {cur next : Blocks Q0 6}
    {g0 g1 g2 g3 g4 g5 : Fin 5 → ExactMatrix Q0 N 1 6}
    (h : W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_614_25 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_614_25 at h
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
  simp only [W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_614_25.constraints_0] at h
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
/-- Member `o` of the sum `614`: the sum of its 5 cells. -/
theorem members614 {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Blocks Q0 6}
    (h : W_Stage_online_eval_0_1.parallel_generatedRoot_614 Backend.backend model params o
      (f0, f1, f2, f3, f4, f5, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ∑ k : Fin 5, f4 (cellIndex o k), ∑ k : Fin 5, f5 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval_0_1.parallel_generatedRoot_614.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 at h
  simp only [W_Stage_online_eval_0_1.parallel_generatedRoot_614.constraints_0] at h
  obtain ⟨-, h6_5, h7_0, h8_0, h10_0, h11_0, h13_0, h14_0, h16_0, h17_0, h19_0, h20_0, h22_0, h23_0, -, h25_5, hout⟩ := h
  have hg : ∀ i : Fin 5, w_6_0 i = f0 (cellIndex o i) ∧
      w_6_1 i = f1 (cellIndex o i) ∧
      w_6_2 i = f2 (cellIndex o i) ∧
      w_6_3 i = f3 (cellIndex o i) ∧
      w_6_4 i = f4 (cellIndex o i) ∧
      w_6_5 i = f5 (cellIndex o i) := by
    intro i
    have := gather614 (h6_5 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, a4, a5, -⟩ := this
    exact ⟨a0, a1, a2, a3, a4, a5⟩
  have hw0 : w_6_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_6_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_6_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_6_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.1
  have hw4 : w_6_4 = fun i ↦ f4 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.1
  have hw5 : w_6_5 = fun i ↦ f5 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.2
  obtain ⟨_, e7_0⟩ := familyGet_eq (k := 0) h7_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 1) h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq (k := 0) h10_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 1) h11_0
  obtain ⟨_, e13_0⟩ := familyGet_eq (k := 0) h13_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 1) h14_0
  obtain ⟨_, e16_0⟩ := familyGet_eq (k := 0) h16_0
  obtain ⟨_, e17_0⟩ := familyGet_eq (k := 1) h17_0
  obtain ⟨_, e19_0⟩ := familyGet_eq (k := 0) h19_0
  obtain ⟨_, e20_0⟩ := familyGet_eq (k := 1) h20_0
  obtain ⟨_, e22_0⟩ := familyGet_eq (k := 0) h22_0
  obtain ⟨_, e23_0⟩ := familyGet_eq (k := 1) h23_0
  have hloop := iter_sum (fun i ↦ (cellAt w_6_0 (2 + i), cellAt w_6_1 (2 + i), cellAt w_6_2 (2 + i), cellAt w_6_3 (2 + i), cellAt w_6_4 (2 + i), cellAt w_6_5 (2 + i), ()))
    (fun i current next hbody ↦ step614 hbody) h25_5
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_9_0, w_12_0, w_15_0, w_18_0, w_21_0, w_24_0, e7_0, e8_0, e10_0, e11_0, e13_0, e14_0, e16_0, e17_0, e19_0, e20_0, e22_0, e23_0, hw0, hw1, hw2, hw3, hw4, hw5]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f4 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f5 (cellIndex o k))

theorem gather664 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Blocks Q5 8}
    (h : W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_664_6 Backend.backend model params o i
      (f0, f1, f2, f3, f4, f5, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f4 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f5 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_664_6 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  simp only [W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_664_6.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, h1_0_8, h1_0_9, h10_0, h1_0_10, h1_0_11, h12_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0 h10_0 h12_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq h10_0
  obtain ⟨_, e12_0⟩ := familyGet_eq h12_0
  subst e2_0 e4_0 e6_0 e8_0 e10_0 e12_0 hout
  rfl

theorem step664 {model : HashModel} {params} {o i : Nat} {cur next : Blocks Q5 8}
    {g0 g1 g2 g3 g4 g5 : Fin 5 → ExactMatrix Q5 N 1 8}
    (h : W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_664_25 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_664_25 at h
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
  simp only [W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_664_25.constraints_0] at h
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
/-- Member `o` of the sum `664`: the sum of its 5 cells. -/
theorem members664 {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Blocks Q5 8}
    (h : W_Stage_online_eval_0_1.parallel_generatedRoot_664 Backend.backend model params o
      (f0, f1, f2, f3, f4, f5, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ∑ k : Fin 5, f4 (cellIndex o k), ∑ k : Fin 5, f5 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval_0_1.parallel_generatedRoot_664.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 at h
  simp only [W_Stage_online_eval_0_1.parallel_generatedRoot_664.constraints_0] at h
  obtain ⟨-, h6_5, h7_0, h8_0, h10_0, h11_0, h13_0, h14_0, h16_0, h17_0, h19_0, h20_0, h22_0, h23_0, -, h25_5, hout⟩ := h
  have hg : ∀ i : Fin 5, w_6_0 i = f0 (cellIndex o i) ∧
      w_6_1 i = f1 (cellIndex o i) ∧
      w_6_2 i = f2 (cellIndex o i) ∧
      w_6_3 i = f3 (cellIndex o i) ∧
      w_6_4 i = f4 (cellIndex o i) ∧
      w_6_5 i = f5 (cellIndex o i) := by
    intro i
    have := gather664 (h6_5 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, a4, a5, -⟩ := this
    exact ⟨a0, a1, a2, a3, a4, a5⟩
  have hw0 : w_6_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_6_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_6_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_6_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.1
  have hw4 : w_6_4 = fun i ↦ f4 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.1
  have hw5 : w_6_5 = fun i ↦ f5 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.2
  obtain ⟨_, e7_0⟩ := familyGet_eq (k := 0) h7_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 1) h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq (k := 0) h10_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 1) h11_0
  obtain ⟨_, e13_0⟩ := familyGet_eq (k := 0) h13_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 1) h14_0
  obtain ⟨_, e16_0⟩ := familyGet_eq (k := 0) h16_0
  obtain ⟨_, e17_0⟩ := familyGet_eq (k := 1) h17_0
  obtain ⟨_, e19_0⟩ := familyGet_eq (k := 0) h19_0
  obtain ⟨_, e20_0⟩ := familyGet_eq (k := 1) h20_0
  obtain ⟨_, e22_0⟩ := familyGet_eq (k := 0) h22_0
  obtain ⟨_, e23_0⟩ := familyGet_eq (k := 1) h23_0
  have hloop := iter_sum (fun i ↦ (cellAt w_6_0 (2 + i), cellAt w_6_1 (2 + i), cellAt w_6_2 (2 + i), cellAt w_6_3 (2 + i), cellAt w_6_4 (2 + i), cellAt w_6_5 (2 + i), ()))
    (fun i current next hbody ↦ step664 hbody) h25_5
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_9_0, w_12_0, w_15_0, w_18_0, w_21_0, w_24_0, e7_0, e8_0, e10_0, e11_0, e13_0, e14_0, e16_0, e17_0, e19_0, e20_0, e22_0, e23_0, hw0, hw1, hw2, hw3, hw4, hw5]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f4 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f5 (cellIndex o k))

theorem gather668 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Blocks Q0 6}
    (h : W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_668_6 Backend.backend model params o i
      (f0, f1, f2, f3, f4, f5, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f4 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f5 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_668_6 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  simp only [W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_668_6.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, h1_0_8, h1_0_9, h10_0, h1_0_10, h1_0_11, h12_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0 h10_0 h12_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq h10_0
  obtain ⟨_, e12_0⟩ := familyGet_eq h12_0
  subst e2_0 e4_0 e6_0 e8_0 e10_0 e12_0 hout
  rfl

theorem step668 {model : HashModel} {params} {o i : Nat} {cur next : Blocks Q0 6}
    {g0 g1 g2 g3 g4 g5 : Fin 5 → ExactMatrix Q0 N 1 6}
    (h : W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_668_25 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_668_25 at h
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
  simp only [W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_668_25.constraints_0] at h
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
/-- Member `o` of the sum `668`: the sum of its 5 cells. -/
theorem members668 {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Blocks Q0 6}
    (h : W_Stage_online_eval_0_1.parallel_generatedRoot_668 Backend.backend model params o
      (f0, f1, f2, f3, f4, f5, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ∑ k : Fin 5, f4 (cellIndex o k), ∑ k : Fin 5, f5 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval_0_1.parallel_generatedRoot_668.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 at h
  simp only [W_Stage_online_eval_0_1.parallel_generatedRoot_668.constraints_0] at h
  obtain ⟨-, h6_5, h7_0, h8_0, h10_0, h11_0, h13_0, h14_0, h16_0, h17_0, h19_0, h20_0, h22_0, h23_0, -, h25_5, hout⟩ := h
  have hg : ∀ i : Fin 5, w_6_0 i = f0 (cellIndex o i) ∧
      w_6_1 i = f1 (cellIndex o i) ∧
      w_6_2 i = f2 (cellIndex o i) ∧
      w_6_3 i = f3 (cellIndex o i) ∧
      w_6_4 i = f4 (cellIndex o i) ∧
      w_6_5 i = f5 (cellIndex o i) := by
    intro i
    have := gather668 (h6_5 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, a4, a5, -⟩ := this
    exact ⟨a0, a1, a2, a3, a4, a5⟩
  have hw0 : w_6_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_6_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_6_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_6_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.1
  have hw4 : w_6_4 = fun i ↦ f4 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.1
  have hw5 : w_6_5 = fun i ↦ f5 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.2
  obtain ⟨_, e7_0⟩ := familyGet_eq (k := 0) h7_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 1) h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq (k := 0) h10_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 1) h11_0
  obtain ⟨_, e13_0⟩ := familyGet_eq (k := 0) h13_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 1) h14_0
  obtain ⟨_, e16_0⟩ := familyGet_eq (k := 0) h16_0
  obtain ⟨_, e17_0⟩ := familyGet_eq (k := 1) h17_0
  obtain ⟨_, e19_0⟩ := familyGet_eq (k := 0) h19_0
  obtain ⟨_, e20_0⟩ := familyGet_eq (k := 1) h20_0
  obtain ⟨_, e22_0⟩ := familyGet_eq (k := 0) h22_0
  obtain ⟨_, e23_0⟩ := familyGet_eq (k := 1) h23_0
  have hloop := iter_sum (fun i ↦ (cellAt w_6_0 (2 + i), cellAt w_6_1 (2 + i), cellAt w_6_2 (2 + i), cellAt w_6_3 (2 + i), cellAt w_6_4 (2 + i), cellAt w_6_5 (2 + i), ()))
    (fun i current next hbody ↦ step668 hbody) h25_5
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_9_0, w_12_0, w_15_0, w_18_0, w_21_0, w_24_0, e7_0, e8_0, e10_0, e11_0, e13_0, e14_0, e16_0, e17_0, e19_0, e20_0, e22_0, e23_0, hw0, hw1, hw2, hw3, hw4, hw5]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f4 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f5 (cellIndex o k))

theorem gather718 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Blocks Q5 8}
    (h : W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_718_6 Backend.backend model params o i
      (f0, f1, f2, f3, f4, f5, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f4 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f5 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_718_6 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  simp only [W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_718_6.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, h1_0_8, h1_0_9, h10_0, h1_0_10, h1_0_11, h12_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0 h10_0 h12_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq h10_0
  obtain ⟨_, e12_0⟩ := familyGet_eq h12_0
  subst e2_0 e4_0 e6_0 e8_0 e10_0 e12_0 hout
  rfl

theorem step718 {model : HashModel} {params} {o i : Nat} {cur next : Blocks Q5 8}
    {g0 g1 g2 g3 g4 g5 : Fin 5 → ExactMatrix Q5 N 1 8}
    (h : W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_718_25 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_718_25 at h
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
  simp only [W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_718_25.constraints_0] at h
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
/-- Member `o` of the sum `718`: the sum of its 5 cells. -/
theorem members718 {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Blocks Q5 8}
    (h : W_Stage_online_eval_0_1.parallel_generatedRoot_718 Backend.backend model params o
      (f0, f1, f2, f3, f4, f5, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ∑ k : Fin 5, f4 (cellIndex o k), ∑ k : Fin 5, f5 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval_0_1.parallel_generatedRoot_718.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 at h
  simp only [W_Stage_online_eval_0_1.parallel_generatedRoot_718.constraints_0] at h
  obtain ⟨-, h6_5, h7_0, h8_0, h10_0, h11_0, h13_0, h14_0, h16_0, h17_0, h19_0, h20_0, h22_0, h23_0, -, h25_5, hout⟩ := h
  have hg : ∀ i : Fin 5, w_6_0 i = f0 (cellIndex o i) ∧
      w_6_1 i = f1 (cellIndex o i) ∧
      w_6_2 i = f2 (cellIndex o i) ∧
      w_6_3 i = f3 (cellIndex o i) ∧
      w_6_4 i = f4 (cellIndex o i) ∧
      w_6_5 i = f5 (cellIndex o i) := by
    intro i
    have := gather718 (h6_5 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, a4, a5, -⟩ := this
    exact ⟨a0, a1, a2, a3, a4, a5⟩
  have hw0 : w_6_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_6_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_6_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_6_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.1
  have hw4 : w_6_4 = fun i ↦ f4 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.1
  have hw5 : w_6_5 = fun i ↦ f5 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.2
  obtain ⟨_, e7_0⟩ := familyGet_eq (k := 0) h7_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 1) h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq (k := 0) h10_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 1) h11_0
  obtain ⟨_, e13_0⟩ := familyGet_eq (k := 0) h13_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 1) h14_0
  obtain ⟨_, e16_0⟩ := familyGet_eq (k := 0) h16_0
  obtain ⟨_, e17_0⟩ := familyGet_eq (k := 1) h17_0
  obtain ⟨_, e19_0⟩ := familyGet_eq (k := 0) h19_0
  obtain ⟨_, e20_0⟩ := familyGet_eq (k := 1) h20_0
  obtain ⟨_, e22_0⟩ := familyGet_eq (k := 0) h22_0
  obtain ⟨_, e23_0⟩ := familyGet_eq (k := 1) h23_0
  have hloop := iter_sum (fun i ↦ (cellAt w_6_0 (2 + i), cellAt w_6_1 (2 + i), cellAt w_6_2 (2 + i), cellAt w_6_3 (2 + i), cellAt w_6_4 (2 + i), cellAt w_6_5 (2 + i), ()))
    (fun i current next hbody ↦ step718 hbody) h25_5
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_9_0, w_12_0, w_15_0, w_18_0, w_21_0, w_24_0, e7_0, e8_0, e10_0, e11_0, e13_0, e14_0, e16_0, e17_0, e19_0, e20_0, e22_0, e23_0, hw0, hw1, hw2, hw3, hw4, hw5]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f4 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f5 (cellIndex o k))

theorem gather722 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Blocks Q0 6}
    (h : W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_722_6 Backend.backend model params o i
      (f0, f1, f2, f3, f4, f5, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f4 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f5 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_722_6 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  simp only [W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_722_6.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, h1_0_8, h1_0_9, h10_0, h1_0_10, h1_0_11, h12_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0 h10_0 h12_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq h10_0
  obtain ⟨_, e12_0⟩ := familyGet_eq h12_0
  subst e2_0 e4_0 e6_0 e8_0 e10_0 e12_0 hout
  rfl

theorem step722 {model : HashModel} {params} {o i : Nat} {cur next : Blocks Q0 6}
    {g0 g1 g2 g3 g4 g5 : Fin 5 → ExactMatrix Q0 N 1 6}
    (h : W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_722_25 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_722_25 at h
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
  simp only [W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_722_25.constraints_0] at h
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
/-- Member `o` of the sum `722`: the sum of its 5 cells. -/
theorem members722 {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Blocks Q0 6}
    (h : W_Stage_online_eval_0_1.parallel_generatedRoot_722 Backend.backend model params o
      (f0, f1, f2, f3, f4, f5, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ∑ k : Fin 5, f4 (cellIndex o k), ∑ k : Fin 5, f5 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval_0_1.parallel_generatedRoot_722.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 at h
  simp only [W_Stage_online_eval_0_1.parallel_generatedRoot_722.constraints_0] at h
  obtain ⟨-, h6_5, h7_0, h8_0, h10_0, h11_0, h13_0, h14_0, h16_0, h17_0, h19_0, h20_0, h22_0, h23_0, -, h25_5, hout⟩ := h
  have hg : ∀ i : Fin 5, w_6_0 i = f0 (cellIndex o i) ∧
      w_6_1 i = f1 (cellIndex o i) ∧
      w_6_2 i = f2 (cellIndex o i) ∧
      w_6_3 i = f3 (cellIndex o i) ∧
      w_6_4 i = f4 (cellIndex o i) ∧
      w_6_5 i = f5 (cellIndex o i) := by
    intro i
    have := gather722 (h6_5 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, a4, a5, -⟩ := this
    exact ⟨a0, a1, a2, a3, a4, a5⟩
  have hw0 : w_6_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_6_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_6_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_6_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.1
  have hw4 : w_6_4 = fun i ↦ f4 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.1
  have hw5 : w_6_5 = fun i ↦ f5 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.2
  obtain ⟨_, e7_0⟩ := familyGet_eq (k := 0) h7_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 1) h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq (k := 0) h10_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 1) h11_0
  obtain ⟨_, e13_0⟩ := familyGet_eq (k := 0) h13_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 1) h14_0
  obtain ⟨_, e16_0⟩ := familyGet_eq (k := 0) h16_0
  obtain ⟨_, e17_0⟩ := familyGet_eq (k := 1) h17_0
  obtain ⟨_, e19_0⟩ := familyGet_eq (k := 0) h19_0
  obtain ⟨_, e20_0⟩ := familyGet_eq (k := 1) h20_0
  obtain ⟨_, e22_0⟩ := familyGet_eq (k := 0) h22_0
  obtain ⟨_, e23_0⟩ := familyGet_eq (k := 1) h23_0
  have hloop := iter_sum (fun i ↦ (cellAt w_6_0 (2 + i), cellAt w_6_1 (2 + i), cellAt w_6_2 (2 + i), cellAt w_6_3 (2 + i), cellAt w_6_4 (2 + i), cellAt w_6_5 (2 + i), ()))
    (fun i current next hbody ↦ step722 hbody) h25_5
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_9_0, w_12_0, w_15_0, w_18_0, w_21_0, w_24_0, e7_0, e8_0, e10_0, e11_0, e13_0, e14_0, e16_0, e17_0, e19_0, e20_0, e22_0, e23_0, hw0, hw1, hw2, hw3, hw4, hw5]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f4 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f5 (cellIndex o k))

theorem gather772 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Blocks Q5 8}
    (h : W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_772_6 Backend.backend model params o i
      (f0, f1, f2, f3, f4, f5, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f4 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f5 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_772_6 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  simp only [W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_772_6.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, h1_0_8, h1_0_9, h10_0, h1_0_10, h1_0_11, h12_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0 h10_0 h12_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq h10_0
  obtain ⟨_, e12_0⟩ := familyGet_eq h12_0
  subst e2_0 e4_0 e6_0 e8_0 e10_0 e12_0 hout
  rfl

theorem step772 {model : HashModel} {params} {o i : Nat} {cur next : Blocks Q5 8}
    {g0 g1 g2 g3 g4 g5 : Fin 5 → ExactMatrix Q5 N 1 8}
    (h : W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_772_25 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_772_25 at h
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
  simp only [W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_772_25.constraints_0] at h
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
/-- Member `o` of the sum `772`: the sum of its 5 cells. -/
theorem members772 {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Blocks Q5 8}
    (h : W_Stage_online_eval_0_1.parallel_generatedRoot_772 Backend.backend model params o
      (f0, f1, f2, f3, f4, f5, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ∑ k : Fin 5, f4 (cellIndex o k), ∑ k : Fin 5, f5 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval_0_1.parallel_generatedRoot_772.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 at h
  simp only [W_Stage_online_eval_0_1.parallel_generatedRoot_772.constraints_0] at h
  obtain ⟨-, h6_5, h7_0, h8_0, h10_0, h11_0, h13_0, h14_0, h16_0, h17_0, h19_0, h20_0, h22_0, h23_0, -, h25_5, hout⟩ := h
  have hg : ∀ i : Fin 5, w_6_0 i = f0 (cellIndex o i) ∧
      w_6_1 i = f1 (cellIndex o i) ∧
      w_6_2 i = f2 (cellIndex o i) ∧
      w_6_3 i = f3 (cellIndex o i) ∧
      w_6_4 i = f4 (cellIndex o i) ∧
      w_6_5 i = f5 (cellIndex o i) := by
    intro i
    have := gather772 (h6_5 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, a4, a5, -⟩ := this
    exact ⟨a0, a1, a2, a3, a4, a5⟩
  have hw0 : w_6_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_6_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_6_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_6_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.1
  have hw4 : w_6_4 = fun i ↦ f4 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.1
  have hw5 : w_6_5 = fun i ↦ f5 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.2
  obtain ⟨_, e7_0⟩ := familyGet_eq (k := 0) h7_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 1) h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq (k := 0) h10_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 1) h11_0
  obtain ⟨_, e13_0⟩ := familyGet_eq (k := 0) h13_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 1) h14_0
  obtain ⟨_, e16_0⟩ := familyGet_eq (k := 0) h16_0
  obtain ⟨_, e17_0⟩ := familyGet_eq (k := 1) h17_0
  obtain ⟨_, e19_0⟩ := familyGet_eq (k := 0) h19_0
  obtain ⟨_, e20_0⟩ := familyGet_eq (k := 1) h20_0
  obtain ⟨_, e22_0⟩ := familyGet_eq (k := 0) h22_0
  obtain ⟨_, e23_0⟩ := familyGet_eq (k := 1) h23_0
  have hloop := iter_sum (fun i ↦ (cellAt w_6_0 (2 + i), cellAt w_6_1 (2 + i), cellAt w_6_2 (2 + i), cellAt w_6_3 (2 + i), cellAt w_6_4 (2 + i), cellAt w_6_5 (2 + i), ()))
    (fun i current next hbody ↦ step772 hbody) h25_5
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_9_0, w_12_0, w_15_0, w_18_0, w_21_0, w_24_0, e7_0, e8_0, e10_0, e11_0, e13_0, e14_0, e16_0, e17_0, e19_0, e20_0, e22_0, e23_0, hw0, hw1, hw2, hw3, hw4, hw5]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f4 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f5 (cellIndex o k))

theorem gather776 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Blocks Q0 6}
    (h : W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_776_6 Backend.backend model params o i
      (f0, f1, f2, f3, f4, f5, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f4 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f5 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_776_6 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  simp only [W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_776_6.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, h1_0_8, h1_0_9, h10_0, h1_0_10, h1_0_11, h12_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0 h10_0 h12_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq h10_0
  obtain ⟨_, e12_0⟩ := familyGet_eq h12_0
  subst e2_0 e4_0 e6_0 e8_0 e10_0 e12_0 hout
  rfl

theorem step776 {model : HashModel} {params} {o i : Nat} {cur next : Blocks Q0 6}
    {g0 g1 g2 g3 g4 g5 : Fin 5 → ExactMatrix Q0 N 1 6}
    (h : W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_776_25 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_776_25 at h
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
  simp only [W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_776_25.constraints_0] at h
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
/-- Member `o` of the sum `776`: the sum of its 5 cells. -/
theorem members776 {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Blocks Q0 6}
    (h : W_Stage_online_eval_0_1.parallel_generatedRoot_776 Backend.backend model params o
      (f0, f1, f2, f3, f4, f5, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ∑ k : Fin 5, f4 (cellIndex o k), ∑ k : Fin 5, f5 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval_0_1.parallel_generatedRoot_776.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 at h
  simp only [W_Stage_online_eval_0_1.parallel_generatedRoot_776.constraints_0] at h
  obtain ⟨-, h6_5, h7_0, h8_0, h10_0, h11_0, h13_0, h14_0, h16_0, h17_0, h19_0, h20_0, h22_0, h23_0, -, h25_5, hout⟩ := h
  have hg : ∀ i : Fin 5, w_6_0 i = f0 (cellIndex o i) ∧
      w_6_1 i = f1 (cellIndex o i) ∧
      w_6_2 i = f2 (cellIndex o i) ∧
      w_6_3 i = f3 (cellIndex o i) ∧
      w_6_4 i = f4 (cellIndex o i) ∧
      w_6_5 i = f5 (cellIndex o i) := by
    intro i
    have := gather776 (h6_5 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, a4, a5, -⟩ := this
    exact ⟨a0, a1, a2, a3, a4, a5⟩
  have hw0 : w_6_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_6_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_6_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_6_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.1
  have hw4 : w_6_4 = fun i ↦ f4 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.1
  have hw5 : w_6_5 = fun i ↦ f5 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.2
  obtain ⟨_, e7_0⟩ := familyGet_eq (k := 0) h7_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 1) h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq (k := 0) h10_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 1) h11_0
  obtain ⟨_, e13_0⟩ := familyGet_eq (k := 0) h13_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 1) h14_0
  obtain ⟨_, e16_0⟩ := familyGet_eq (k := 0) h16_0
  obtain ⟨_, e17_0⟩ := familyGet_eq (k := 1) h17_0
  obtain ⟨_, e19_0⟩ := familyGet_eq (k := 0) h19_0
  obtain ⟨_, e20_0⟩ := familyGet_eq (k := 1) h20_0
  obtain ⟨_, e22_0⟩ := familyGet_eq (k := 0) h22_0
  obtain ⟨_, e23_0⟩ := familyGet_eq (k := 1) h23_0
  have hloop := iter_sum (fun i ↦ (cellAt w_6_0 (2 + i), cellAt w_6_1 (2 + i), cellAt w_6_2 (2 + i), cellAt w_6_3 (2 + i), cellAt w_6_4 (2 + i), cellAt w_6_5 (2 + i), ()))
    (fun i current next hbody ↦ step776 hbody) h25_5
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_9_0, w_12_0, w_15_0, w_18_0, w_21_0, w_24_0, e7_0, e8_0, e10_0, e11_0, e13_0, e14_0, e16_0, e17_0, e19_0, e20_0, e22_0, e23_0, hw0, hw1, hw2, hw3, hw4, hw5]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f4 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f5 (cellIndex o k))

theorem gather826 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Blocks Q5 8}
    (h : W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_826_6 Backend.backend model params o i
      (f0, f1, f2, f3, f4, f5, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f4 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f5 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_826_6 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  simp only [W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_826_6.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, h1_0_8, h1_0_9, h10_0, h1_0_10, h1_0_11, h12_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0 h10_0 h12_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq h10_0
  obtain ⟨_, e12_0⟩ := familyGet_eq h12_0
  subst e2_0 e4_0 e6_0 e8_0 e10_0 e12_0 hout
  rfl

theorem step826 {model : HashModel} {params} {o i : Nat} {cur next : Blocks Q5 8}
    {g0 g1 g2 g3 g4 g5 : Fin 5 → ExactMatrix Q5 N 1 8}
    (h : W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_826_25 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_826_25 at h
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
  simp only [W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_826_25.constraints_0] at h
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
/-- Member `o` of the sum `826`: the sum of its 5 cells. -/
theorem members826 {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q5 N 1 8} {outputs : Blocks Q5 8}
    (h : W_Stage_online_eval_0_1.parallel_generatedRoot_826 Backend.backend model params o
      (f0, f1, f2, f3, f4, f5, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ∑ k : Fin 5, f4 (cellIndex o k), ∑ k : Fin 5, f5 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval_0_1.parallel_generatedRoot_826.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 at h
  simp only [W_Stage_online_eval_0_1.parallel_generatedRoot_826.constraints_0] at h
  obtain ⟨-, h6_5, h7_0, h8_0, h10_0, h11_0, h13_0, h14_0, h16_0, h17_0, h19_0, h20_0, h22_0, h23_0, -, h25_5, hout⟩ := h
  have hg : ∀ i : Fin 5, w_6_0 i = f0 (cellIndex o i) ∧
      w_6_1 i = f1 (cellIndex o i) ∧
      w_6_2 i = f2 (cellIndex o i) ∧
      w_6_3 i = f3 (cellIndex o i) ∧
      w_6_4 i = f4 (cellIndex o i) ∧
      w_6_5 i = f5 (cellIndex o i) := by
    intro i
    have := gather826 (h6_5 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, a4, a5, -⟩ := this
    exact ⟨a0, a1, a2, a3, a4, a5⟩
  have hw0 : w_6_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_6_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_6_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_6_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.1
  have hw4 : w_6_4 = fun i ↦ f4 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.1
  have hw5 : w_6_5 = fun i ↦ f5 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.2
  obtain ⟨_, e7_0⟩ := familyGet_eq (k := 0) h7_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 1) h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq (k := 0) h10_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 1) h11_0
  obtain ⟨_, e13_0⟩ := familyGet_eq (k := 0) h13_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 1) h14_0
  obtain ⟨_, e16_0⟩ := familyGet_eq (k := 0) h16_0
  obtain ⟨_, e17_0⟩ := familyGet_eq (k := 1) h17_0
  obtain ⟨_, e19_0⟩ := familyGet_eq (k := 0) h19_0
  obtain ⟨_, e20_0⟩ := familyGet_eq (k := 1) h20_0
  obtain ⟨_, e22_0⟩ := familyGet_eq (k := 0) h22_0
  obtain ⟨_, e23_0⟩ := familyGet_eq (k := 1) h23_0
  have hloop := iter_sum (fun i ↦ (cellAt w_6_0 (2 + i), cellAt w_6_1 (2 + i), cellAt w_6_2 (2 + i), cellAt w_6_3 (2 + i), cellAt w_6_4 (2 + i), cellAt w_6_5 (2 + i), ()))
    (fun i current next hbody ↦ step826 hbody) h25_5
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_9_0, w_12_0, w_15_0, w_18_0, w_21_0, w_24_0, e7_0, e8_0, e10_0, e11_0, e13_0, e14_0, e16_0, e17_0, e19_0, e20_0, e22_0, e23_0, hw0, hw1, hw2, hw3, hw4, hw5]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f4 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f5 (cellIndex o k))

theorem gather830 {model : HashModel} {params} {o i : Nat}
    {f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Blocks Q0 6}
    (h : W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_830_6 Backend.backend model params o i
      (f0, f1, f2, f3, f4, f5, ()) outputs) (ho : o < 160) (hi : i < 5) :
    outputs = (f0 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f1 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f2 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f3 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f4 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), f5 (cellIndex (⟨o, ho⟩ : Fin 160) (⟨i, hi⟩ : Fin 5)), ()) := by
  unfold W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_830_6 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 at h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_6_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  simp only [W_Stage_online_eval_0_1.parallel_parallel_generatedRoot_830_6.constraints_0] at h
  obtain ⟨h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h1_0_6, h1_0_7, h8_0, h1_0_8, h1_0_9, h10_0, h1_0_10, h1_0_11, h12_0, hout⟩ := h
  have hidx : w_1_0 = Int.ofNat (5 * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at h2_0 h4_0 h6_0 h8_0 h10_0 h12_0
  obtain ⟨_, e2_0⟩ := familyGet_eq h2_0
  obtain ⟨_, e4_0⟩ := familyGet_eq h4_0
  obtain ⟨_, e6_0⟩ := familyGet_eq h6_0
  obtain ⟨_, e8_0⟩ := familyGet_eq h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq h10_0
  obtain ⟨_, e12_0⟩ := familyGet_eq h12_0
  subst e2_0 e4_0 e6_0 e8_0 e10_0 e12_0 hout
  rfl

theorem step830 {model : HashModel} {params} {o i : Nat} {cur next : Blocks Q0 6}
    {g0 g1 g2 g3 g4 g5 : Fin 5 → ExactMatrix Q0 N 1 6}
    (h : W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_830_25 Backend.backend model params o i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, g0, g1, g2, g3, g4, g5,
        ()) next) :
    next = cur + (cellAt g0 (2 + i), cellAt g1 (2 + i), cellAt g2 (2 + i), cellAt g3 (2 + i), cellAt g4 (2 + i), cellAt g5 (2 + i), ()) := by
  unfold W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_830_25 at h
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
  simp only [W_Stage_online_eval_0_1.sequential_parallel_generatedRoot_830_25.constraints_0] at h
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
/-- Member `o` of the sum `830`: the sum of its 5 cells. -/
theorem members830 {model : HashModel} {params} {o : Fin 160}
    {f0 f1 f2 f3 f4 f5 : Fin 800 → ExactMatrix Q0 N 1 6} {outputs : Blocks Q0 6}
    (h : W_Stage_online_eval_0_1.parallel_generatedRoot_830 Backend.backend model params o
      (f0, f1, f2, f3, f4, f5, ()) outputs) :
    outputs = (∑ k : Fin 5, f0 (cellIndex o k), ∑ k : Fin 5, f1 (cellIndex o k), ∑ k : Fin 5, f2 (cellIndex o k), ∑ k : Fin 5, f3 (cellIndex o k), ∑ k : Fin 5, f4 (cellIndex o k), ∑ k : Fin 5, f5 (cellIndex o k), ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval_0_1.parallel_generatedRoot_830.body at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 at h
  simp only [W_Stage_online_eval_0_1.parallel_generatedRoot_830.constraints_0] at h
  obtain ⟨-, h6_5, h7_0, h8_0, h10_0, h11_0, h13_0, h14_0, h16_0, h17_0, h19_0, h20_0, h22_0, h23_0, -, h25_5, hout⟩ := h
  have hg : ∀ i : Fin 5, w_6_0 i = f0 (cellIndex o i) ∧
      w_6_1 i = f1 (cellIndex o i) ∧
      w_6_2 i = f2 (cellIndex o i) ∧
      w_6_3 i = f3 (cellIndex o i) ∧
      w_6_4 i = f4 (cellIndex o i) ∧
      w_6_5 i = f5 (cellIndex o i) := by
    intro i
    have := gather830 (h6_5 i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨a0, a1, a2, a3, a4, a5, -⟩ := this
    exact ⟨a0, a1, a2, a3, a4, a5⟩
  have hw0 : w_6_0 = fun i ↦ f0 (cellIndex o i) := funext fun i ↦ (hg i).1
  have hw1 : w_6_1 = fun i ↦ f1 (cellIndex o i) := funext fun i ↦ (hg i).2.1
  have hw2 : w_6_2 = fun i ↦ f2 (cellIndex o i) := funext fun i ↦ (hg i).2.2.1
  have hw3 : w_6_3 = fun i ↦ f3 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.1
  have hw4 : w_6_4 = fun i ↦ f4 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.1
  have hw5 : w_6_5 = fun i ↦ f5 (cellIndex o i) := funext fun i ↦ (hg i).2.2.2.2.2
  obtain ⟨_, e7_0⟩ := familyGet_eq (k := 0) h7_0
  obtain ⟨_, e8_0⟩ := familyGet_eq (k := 1) h8_0
  obtain ⟨_, e10_0⟩ := familyGet_eq (k := 0) h10_0
  obtain ⟨_, e11_0⟩ := familyGet_eq (k := 1) h11_0
  obtain ⟨_, e13_0⟩ := familyGet_eq (k := 0) h13_0
  obtain ⟨_, e14_0⟩ := familyGet_eq (k := 1) h14_0
  obtain ⟨_, e16_0⟩ := familyGet_eq (k := 0) h16_0
  obtain ⟨_, e17_0⟩ := familyGet_eq (k := 1) h17_0
  obtain ⟨_, e19_0⟩ := familyGet_eq (k := 0) h19_0
  obtain ⟨_, e20_0⟩ := familyGet_eq (k := 1) h20_0
  obtain ⟨_, e22_0⟩ := familyGet_eq (k := 0) h22_0
  obtain ⟨_, e23_0⟩ := familyGet_eq (k := 1) h23_0
  have hloop := iter_sum (fun i ↦ (cellAt w_6_0 (2 + i), cellAt w_6_1 (2 + i), cellAt w_6_2 (2 + i), cellAt w_6_3 (2 + i), cellAt w_6_4 (2 + i), cellAt w_6_5 (2 + i), ()))
    (fun i current next hbody ↦ step830 hbody) h25_5
  rw [show Int.toNat 3 = 3 from rfl] at hloop
  rw [hout, hloop]
  simp only [w_9_0, w_12_0, w_15_0, w_18_0, w_21_0, w_24_0, e7_0, e8_0, e10_0, e11_0, e13_0, e14_0, e16_0, e17_0, e19_0, e20_0, e22_0, e23_0, hw0, hw1, hw2, hw3, hw4, hw5]
  simp only [matrixAdd]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (rfl)))))) <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
  · exact sum_cells (w := 3) (fun k ↦ f0 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f1 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f2 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f3 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f4 (cellIndex o k))
  · exact sum_cells (w := 3) (fun k ↦ f5 (cellIndex o k))

end RefreshTopProof.Part_0_1
