import Stage_sigma_online_eval
import W_Stage_sigma_online_eval
import ExponentSampling

/-! Every run of the average-case relations of `Stage_sigma_online_eval` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Stage_sigma_online_eval.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_58.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_58 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_58 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_58
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_59_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_59_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_59_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_59_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_59_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_59_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_59_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_59_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_59.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_59.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_59.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_59_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_59_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_59.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_59 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_59 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_59
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_59.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_62.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_62 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_62 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_62
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_63_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_63_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_63_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_63_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_63_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_63_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_63_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_63_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_63.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_63.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_63.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_63_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_63_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_63.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_63 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_63 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_63
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_63.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_66.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_66 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_66 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_66
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_67_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_67_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_67_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_67_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_67_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_67_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_67_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_67_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_67.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_67.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_67.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_67_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_67_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_67.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_67 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_67 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_67
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_67.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_70.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_70 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_70 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_70
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_71_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_71_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_71_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_71_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_71_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_71_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_71_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_71_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_71.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_71.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_71.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_71_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_71_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_71.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_71 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_71 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_71
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_71.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_74.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_74 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_74 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_74
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_75_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_75_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_75_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_75_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_75_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_75_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_75_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_75_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_75.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_75.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_75.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_75_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_75_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_75.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_75 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_75 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_75
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_75.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_78.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_78 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_78 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_78
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_79_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_79_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_79_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_79_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_79_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_79_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_79_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_79_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_79.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_79.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_79.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_79_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_79_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_79.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_79 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_79 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_79
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_79.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_82.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_82 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_82 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_82
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_83_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_83_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_83_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_83_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_83_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_83_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_83_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_83_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_83.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_83.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_83.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_83_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_83_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_83.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_83 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_83 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_83
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_83.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_86.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_86 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_86 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_86
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_87_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_87_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_87_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_87_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_87_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_87_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_87_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_87_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_87.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_87.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_87.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_87_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_87_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_87.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_87 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_87 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_87
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_87.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_90.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_90 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_90 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_90
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_91_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_91_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_91_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_91_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_91_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_91_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_91_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_91_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_91.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_91.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_91.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_91_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_91_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_91.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_91 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_91 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_91
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_91.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_94.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_94 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_94 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_94
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_95_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_95_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_95_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_95_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_95_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_95_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_95_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_95_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_95.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_95.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_95.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_95_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_95_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_95.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_95 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_95 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_95
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_95.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_98.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_98 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_98 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_98
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_99_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_99_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_99_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_99_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_99_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_99_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_99_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_99_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_99.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_99.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_99.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_99_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_99_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_99.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_99 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_99 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_99
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_99.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_102.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_102 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_102 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_102
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_103_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_103_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_103_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_103_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_103_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_103_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_103_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_103_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_103.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_103.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_103.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_103_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_103_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_103.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_103 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_103 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_103
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_103.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_106.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_106 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_106 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_106
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_107_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_107_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_107_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_107_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_107_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_107_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_107_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_107_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_107.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_107.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_107.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_107_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_107_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_107.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_107 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_107 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_107
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_107.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_110.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_110 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_110 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_110
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_111_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_111_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_111_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_111_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_111_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_111_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_111_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_111_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_111.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_111.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_111.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_111_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_111_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_111.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_111 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_111 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_111
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_111.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_114.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_114 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_114 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_114
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_115_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_115_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_115_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_115_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_115_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_115_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_115_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_115_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_115.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_115.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_115.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_115_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_115_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_115.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_115 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_115 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_115
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_115.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_118.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_118 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_118 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_118
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_119_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_119_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_119_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_119_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_119_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_119_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_119_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_119_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_119.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_119.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_119.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_119_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_119_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_119.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_119 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_119 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_119
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_119.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_122.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_122 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_122 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_122
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_123_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_123_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_123_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_123_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_123_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_123_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_123_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_123_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_123.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_123.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_123.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_123_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_123_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_123.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_123 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_123 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_123
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_123.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_126.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_126 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_126 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_126
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_127_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_127_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_127_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_127_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_127_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_127_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_127_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_127_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_127.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_127.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_127.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_127_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_127_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_127.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_127 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_127 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_127
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_127.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_130.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_130 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_130 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_130
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_131_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_131_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_131_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_131_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_131_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_131_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_131_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_131_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_131.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_131.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_131.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_131_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_131_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_131.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_131 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_131 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_131
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_131.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_134.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_134 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_134 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_134
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_135_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_135_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_135_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_135_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_135_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_135_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_135_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_135_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_135.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_135.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_135.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_135_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_135_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_135.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_135 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_135 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_135
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_135.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_138.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_138 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_138 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_138
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_139_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_139_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_139_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_139_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_139_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_139_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_139_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_139_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_139.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_139.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_139.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_139_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_139_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_139.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_139 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_139 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_139
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_139.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_142.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_142 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_142 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_142
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_143_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_143_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_143_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_143_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_143_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_143_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_143_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_143_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_143.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_143.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_143.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_143_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_143_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_143.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_143 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_143 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_143
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_143.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_146.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_146 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_146 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_146
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_147_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_147_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_147_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_147_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_147_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_147_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_147_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_147_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_147.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_147.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_147.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_147_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_147_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_147.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_147 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_147 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_147
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_147.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_150.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_150 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_150 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_150
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_151_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_151_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_151_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_151_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_151_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_151_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_151_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_151_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_151.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_151.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_151.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_151_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_151_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_151.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_151 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_151 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_151
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_151.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_154.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_154 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_154 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_154
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_155_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_155_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_155_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_155_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_155_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_155_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_155_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_155_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_155.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_155.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_155.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_155_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_155_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_155.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_155 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_155 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_155
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_155.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_158.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_158 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_158 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_158
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_159_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_159_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_159_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_159_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_159_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_159_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_159_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_159_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_159.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_159.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_159.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_159_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_159_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_159.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_159 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_159 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_159
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_159.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_162.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_162 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_162 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_162
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_163_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_163_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_163_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_163_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_163_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_163_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_163_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_163_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_163.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_163.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_163.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_163_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_163_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_163.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_163 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_163 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_163
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_163.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_166.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_166 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_166 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_166
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_167_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_167_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_167_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_167_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_167_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_167_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_167_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_167_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_167.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_167.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_167.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_167_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_167_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_167.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_167 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_167 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_167
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_167.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_170.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_170 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_170 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_170
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_171_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_171_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_171_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_171_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_171_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_171_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_171_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_171_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_171.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_171.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_171.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_171_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_171_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_171.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_171 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_171 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_171
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_171.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_174.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_174 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_174 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_174
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_175_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_175_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_175_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_175_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_175_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_175_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_175_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_175_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_175.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_175.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_175.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_175_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_175_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_175.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_175 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_175 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_175
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_175.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_178.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_178 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_178 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_178
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_179_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_179_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_179_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_179_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_179_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_179_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_179_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_179_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_179.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_179.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_179.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_179_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_179_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_179.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_179 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_179 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_179
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_179.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_182.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_182 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_182 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_182
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_183_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_183_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_183_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_183_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_183_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_183_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_183_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_183_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_183.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_183.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_183.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_183_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_183_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_183.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_183 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_183 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_183
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_183.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_186.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_186 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_186 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_186
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_187_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_187_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_187_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_187_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_187_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_187_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_187_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_187_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_187.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_187.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_187.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_187_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_187_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_187.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_187 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_187 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_187
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_187.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_190.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_190 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_190 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_190
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_191_6.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_191_6 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_191_6 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_191_6
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_191_25.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_191_25 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_191_25 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_191_25
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_191.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_191.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_191.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_parallel_generatedRoot_188_191_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_sigma_online_eval.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_191_25.bridge hs18))), h17⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_191.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_191 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_191 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_191
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_191.constraints_0.bridge h2)⟩)


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_192_18.bridge {backend hashModel tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_192_18 backend hashModel tape path params x5 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_192_18 backend hashModel params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_192_18
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_192.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_192.constraints_0 backend hashModel tape path params i_0 i_1 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_192.constraints_0 backend hashModel params' i_0 i_1 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, (Stage_sigma_online_eval.iterRuns_mono h6 (fun _ _ _ hs33 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_192_18.bridge hs33))), h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_192.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_192 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_192 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_192
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_2_0, wit1.w_4_0, wit1.w_6_0, wit1.w_8_0, wit1.w_18_0, wit1.w_18_1, wit1.w_18_2, wit1.w_18_3, wit1.w_18_4, wit1.w_18_5, wit1.w_18_6, wit1.w_18_7, wit1.w_18_8, wit1.w_18_9, wit1.w_19_0, wit1.w_21_0, wit1.w_22_0, wit1.w_23_0, wit1.w_23_concat_1, wit1.w_24_0, wit1.w_25_0, wit1.w_26_0, wit1.w_27_0, wit1.w_27_concat_1, wit1.w_28_0, wit1.w_29_0, wit1.w_30_0, wit1.w_31_0, wit1.w_32_0, wit1.w_32_concat_1, wit1.w_33_0, wit1.w_34_0, wit1.w_35_0, wit1.w_36_0, wit1.w_36_concat_1, wit1.w_37_0, wit1.w_38_0, wit1.w_39_0, wit1.w_40_0, wit1.w_42_0, wit1.w_43_0, wit1.w_44_0, wit1.w_45_0, wit1.w_46_0, wit1.w_47_0, wit1.w_48_0, wit1.w_50_0, wit1.w_52_0, wit1.w_53_0, wit1.w_54_0, wit1.w_55_0, wit1.w_57_0, wit1.w_58_0, wit1.w_59_0, wit1.w_60_0, wit1.w_61_0, wit1.w_62_0, wit1.w_63_0, wit1.w_65_0, wit1.w_67_0, wit1.w_68_0, wit1.w_70_0, wit1.w_71_0, wit1.w_73_0, wit1.w_74_0, wit1.w_75_0, wit1.w_78_0, wit1.w_79_0, wit1.w_80_0, wit1.w_83_0⟩, (by obtain ⟨h3, h4, h5, h6, h7⟩ := h2; exact ⟨(Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_192.constraints_0.bridge h3), h4, h5, h6, h7⟩)⟩)


theorem Stage_sigma_online_eval.sequential_parallel_generatedRoot_188_1121.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_generatedRoot_188_1121 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_generatedRoot_188_1121 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_generatedRoot_188_1121
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_generatedRoot_188_1187.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_generatedRoot_188_1187 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_generatedRoot_188_1187 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_generatedRoot_188_1187
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_1247_18.bridge {backend hashModel tape path params params' i_0 x6 i_2 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_1247_18 backend hashModel tape path params i_0 x6 i_2 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_1247_18 backend hashModel params' i_0 x6 i_2 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_1247_18
  exact h


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_1247.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_1247.constraints_0 backend hashModel tape path params i_0 i_1 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_1247.constraints_0 backend hashModel params' i_0 i_1 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, (Stage_sigma_online_eval.iterRuns_mono h6 (fun _ _ _ hs33 ↦ (Stage_sigma_online_eval.sequential_parallel_parallel_generatedRoot_188_1247_18.bridge hs33))), h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_1247.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_1247 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_1247 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_1247
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_2_0, wit1.w_4_0, wit1.w_6_0, wit1.w_8_0, wit1.w_18_0, wit1.w_18_1, wit1.w_18_2, wit1.w_18_3, wit1.w_18_4, wit1.w_18_5, wit1.w_18_6, wit1.w_18_7, wit1.w_18_8, wit1.w_18_9, wit1.w_19_0, wit1.w_21_0, wit1.w_22_0, wit1.w_23_0, wit1.w_23_concat_1, wit1.w_24_0, wit1.w_25_0, wit1.w_26_0, wit1.w_27_0, wit1.w_27_concat_1, wit1.w_28_0, wit1.w_29_0, wit1.w_30_0, wit1.w_31_0, wit1.w_32_0, wit1.w_32_concat_1, wit1.w_33_0, wit1.w_34_0, wit1.w_35_0, wit1.w_36_0, wit1.w_36_concat_1, wit1.w_37_0, wit1.w_38_0, wit1.w_39_0, wit1.w_40_0, wit1.w_42_0, wit1.w_43_0, wit1.w_44_0, wit1.w_45_0, wit1.w_46_0, wit1.w_47_0, wit1.w_48_0, wit1.w_50_0, wit1.w_54_0, wit1.w_55_0, wit1.w_56_0, wit1.w_58_0, wit1.w_59_0, wit1.w_60_0, wit1.w_61_0, wit1.w_62_0, wit1.w_63_0, wit1.w_64_0, wit1.w_66_0, wit1.w_70_0, wit1.w_74_0, wit1.w_78_0, wit1.w_79_0, wit1.w_84_0, wit1.w_85_0⟩, (by obtain ⟨h3, h4, h5, h6, h7⟩ := h2; exact ⟨(Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_1247.constraints_0.bridge h3), h4, h5, h6, h7⟩)⟩)


theorem Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_1248.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_1248 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_1248 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_1248
  exact h


theorem Stage_sigma_online_eval.sequential_parallel_generatedRoot_188_1267.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_online_eval.sequential_parallel_generatedRoot_188_1267 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_sigma_online_eval.sequential_parallel_generatedRoot_188_1267 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_online_eval.sequential_parallel_generatedRoot_188_1267
  exact h


theorem Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_87.bridge {backend hashModel tape path params params' i_0 w_1248_0 w_1248_1 w_1248_2 w_1248_3 w_1248_4 w_1248_5 w_1251_0 w_1252_0 w_1253_0 w_1254_0 w_1255_0 w_1256_0 w_1257_0 w_1258_0 w_1259_0 w_1260_0 w_1261_0 w_1262_0 w_1263_0 w_1264_0 w_1265_0 w_1266_0 w_1267_0 w_1267_1 w_1267_2 w_1267_3 w_1267_4 w_1267_5 w_1268_0 w_1269_0 w_1270_0 w_1271_0 w_1272_0 w_1273_0 outputs}
    (h : Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_87 backend hashModel tape path params i_0 w_1248_0 w_1248_1 w_1248_2 w_1248_3 w_1248_4 w_1248_5 w_1251_0 w_1252_0 w_1253_0 w_1254_0 w_1255_0 w_1256_0 w_1257_0 w_1258_0 w_1259_0 w_1260_0 w_1261_0 w_1262_0 w_1263_0 w_1264_0 w_1265_0 w_1266_0 w_1267_0 w_1267_1 w_1267_2 w_1267_3 w_1267_4 w_1267_5 w_1268_0 w_1269_0 w_1270_0 w_1271_0 w_1272_0 w_1273_0 outputs) : W_Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_87 backend hashModel params' i_0 w_1248_0 w_1248_1 w_1248_2 w_1248_3 w_1248_4 w_1248_5 w_1251_0 w_1252_0 w_1253_0 w_1254_0 w_1255_0 w_1256_0 w_1257_0 w_1258_0 w_1259_0 w_1260_0 w_1261_0 w_1262_0 w_1263_0 w_1264_0 w_1265_0 w_1266_0 w_1267_0 w_1267_1 w_1267_2 w_1267_3 w_1267_4 w_1267_5 w_1268_0 w_1269_0 w_1270_0 w_1271_0 w_1272_0 w_1273_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_sigma_online_eval.iterRuns_mono h12 (fun _ _ _ hs20 ↦ (Stage_sigma_online_eval.sequential_parallel_generatedRoot_188_1267.bridge hs20))), h13, h14, h15, h16, h17, h18, h19⟩)


theorem Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_86.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_192_0 w_192_1 w_192_2 w_192_3 w_192_4 w_192_5 w_1210_0 w_1218_0 w_1220_0 w_1224_0 w_1232_0 w_1234_0 w_1236_0 w_1238_0 w_1239_0 w_1240_0 w_1242_0 w_1243_0 w_1244_0 w_1246_0 w_1247_0 w_1247_1 w_1247_2 w_1247_3 w_1247_4 w_1247_5 w_1248_0 w_1248_1 w_1248_2 w_1248_3 w_1248_4 w_1248_5 w_1249_0 w_1250_0}
    (h : Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_86 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_192_0 w_192_1 w_192_2 w_192_3 w_192_4 w_192_5 w_1210_0 w_1218_0 w_1220_0 w_1224_0 w_1232_0 w_1234_0 w_1236_0 w_1238_0 w_1239_0 w_1240_0 w_1242_0 w_1243_0 w_1244_0 w_1246_0 w_1247_0 w_1247_1 w_1247_2 w_1247_3 w_1247_4 w_1247_5 w_1248_0 w_1248_1 w_1248_2 w_1248_3 w_1248_4 w_1248_5 w_1249_0 w_1250_0) : W_Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_86 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_192_0 w_192_1 w_192_2 w_192_3 w_192_4 w_192_5 w_1210_0 w_1218_0 w_1220_0 w_1224_0 w_1232_0 w_1234_0 w_1236_0 w_1238_0 w_1239_0 w_1240_0 w_1242_0 w_1243_0 w_1244_0 w_1246_0 w_1247_0 w_1247_1 w_1247_2 w_1247_3 w_1247_4 w_1247_5 w_1248_0 w_1248_1 w_1248_2 w_1248_3 w_1248_4 w_1248_5 w_1249_0 w_1250_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_1247.bridge (h28 i))), h29, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_1248.bridge (h30 i))), h31, h32⟩)


theorem Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_82.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_1157_0 w_1160_0 w_1163_0 w_1175_0 w_1180_0 w_1181_0 w_1182_0 w_1183_0 w_1184_0 w_1185_0 w_1186_0 w_1187_0 w_1187_1 w_1187_2 w_1187_3 w_1187_4 w_1187_5 w_1187_6 w_1187_7 w_1187_8 w_1187_9 w_1188_0 w_1189_0 w_1190_0 w_1191_0 w_1192_0 w_1192_concat_1 w_1193_0 w_1194_0 w_1195_0 w_1196_0 w_1196_concat_1 w_1197_0}
    (h : Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_82 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_1157_0 w_1160_0 w_1163_0 w_1175_0 w_1180_0 w_1181_0 w_1182_0 w_1183_0 w_1184_0 w_1185_0 w_1186_0 w_1187_0 w_1187_1 w_1187_2 w_1187_3 w_1187_4 w_1187_5 w_1187_6 w_1187_7 w_1187_8 w_1187_9 w_1188_0 w_1189_0 w_1190_0 w_1191_0 w_1192_0 w_1192_concat_1 w_1193_0 w_1194_0 w_1195_0 w_1196_0 w_1196_concat_1 w_1197_0) : W_Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_82 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_1157_0 w_1160_0 w_1163_0 w_1175_0 w_1180_0 w_1181_0 w_1182_0 w_1183_0 w_1184_0 w_1185_0 w_1186_0 w_1187_0 w_1187_1 w_1187_2 w_1187_3 w_1187_4 w_1187_5 w_1187_6 w_1187_7 w_1187_8 w_1187_9 w_1188_0 w_1189_0 w_1190_0 w_1191_0 w_1192_0 w_1192_concat_1 w_1193_0 w_1194_0 w_1195_0 w_1196_0 w_1196_concat_1 w_1197_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, (Stage_sigma_online_eval.iterRuns_mono h18 (fun _ _ _ hs33 ↦ (Stage_sigma_online_eval.sequential_parallel_generatedRoot_188_1187.bridge hs33))), h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_78.bridge {backend hashModel tape path params params' i_0 w_5_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_198_0 w_199_0 w_200_0 w_201_0 w_202_0 w_1104_0 w_1106_0 w_1108_0 w_1116_0 w_1118_0 w_1120_0 w_1121_0 w_1121_1 w_1121_2 w_1121_3 w_1121_4 w_1121_5 w_1121_6 w_1121_7 w_1121_8 w_1121_9 w_1122_0 w_1123_0 w_1124_0 w_1125_0 w_1126_0 w_1126_concat_1 w_1127_0 w_1128_0 w_1129_0 w_1130_0 w_1130_concat_1 w_1131_0 w_1132_0 w_1133_0 w_1134_0 w_1135_0 w_1135_concat_1 w_1136_0 w_1137_0}
    (h : Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_78 backend hashModel tape path params i_0 w_5_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_198_0 w_199_0 w_200_0 w_201_0 w_202_0 w_1104_0 w_1106_0 w_1108_0 w_1116_0 w_1118_0 w_1120_0 w_1121_0 w_1121_1 w_1121_2 w_1121_3 w_1121_4 w_1121_5 w_1121_6 w_1121_7 w_1121_8 w_1121_9 w_1122_0 w_1123_0 w_1124_0 w_1125_0 w_1126_0 w_1126_concat_1 w_1127_0 w_1128_0 w_1129_0 w_1130_0 w_1130_concat_1 w_1131_0 w_1132_0 w_1133_0 w_1134_0 w_1135_0 w_1135_concat_1 w_1136_0 w_1137_0) : W_Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_78 backend hashModel params' i_0 w_5_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_198_0 w_199_0 w_200_0 w_201_0 w_202_0 w_1104_0 w_1106_0 w_1108_0 w_1116_0 w_1118_0 w_1120_0 w_1121_0 w_1121_1 w_1121_2 w_1121_3 w_1121_4 w_1121_5 w_1121_6 w_1121_7 w_1121_8 w_1121_9 w_1122_0 w_1123_0 w_1124_0 w_1125_0 w_1126_0 w_1126_concat_1 w_1127_0 w_1128_0 w_1129_0 w_1130_0 w_1130_concat_1 w_1131_0 w_1132_0 w_1133_0 w_1134_0 w_1135_0 w_1135_concat_1 w_1136_0 w_1137_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (Stage_sigma_online_eval.iterRuns_mono h2 (fun _ _ _ hs33 ↦ (Stage_sigma_online_eval.sequential_parallel_generatedRoot_188_1121.bridge hs33))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_4.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_5_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_51_0 w_52_0 w_55_0 w_180_0 w_181_0 w_184_0 w_185_0 w_188_0 w_189_0 w_49_0 w_50_0 w_53_0 w_54_0 w_179_0 w_179_1 w_179_2 w_179_3 w_179_4 w_179_5 w_182_0 w_182_1 w_182_2 w_182_3 w_182_4 w_182_5 w_183_0 w_183_1 w_183_2 w_183_3 w_183_4 w_183_5 w_186_0 w_186_1 w_186_2 w_186_3 w_186_4 w_186_5 w_187_0 w_187_1 w_187_2 w_187_3 w_187_4 w_187_5 w_190_0 w_190_1 w_190_2 w_190_3 w_190_4 w_190_5 w_191_0 w_191_1 w_191_2 w_191_3 w_191_4 w_191_5 w_192_0 w_192_1 w_192_2 w_192_3 w_192_4 w_192_5 w_194_0 w_195_0 w_196_0 w_197_0 w_199_0 w_200_0 w_201_0 w_202_0 w_203_0 w_203_concat_1 w_204_0 w_204_concat_1 w_205_0 w_206_0}
    (h : Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_4 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_5_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_51_0 w_52_0 w_55_0 w_180_0 w_181_0 w_184_0 w_185_0 w_188_0 w_189_0 w_49_0 w_50_0 w_53_0 w_54_0 w_179_0 w_179_1 w_179_2 w_179_3 w_179_4 w_179_5 w_182_0 w_182_1 w_182_2 w_182_3 w_182_4 w_182_5 w_183_0 w_183_1 w_183_2 w_183_3 w_183_4 w_183_5 w_186_0 w_186_1 w_186_2 w_186_3 w_186_4 w_186_5 w_187_0 w_187_1 w_187_2 w_187_3 w_187_4 w_187_5 w_190_0 w_190_1 w_190_2 w_190_3 w_190_4 w_190_5 w_191_0 w_191_1 w_191_2 w_191_3 w_191_4 w_191_5 w_192_0 w_192_1 w_192_2 w_192_3 w_192_4 w_192_5 w_194_0 w_195_0 w_196_0 w_197_0 w_199_0 w_200_0 w_201_0 w_202_0 w_203_0 w_203_concat_1 w_204_0 w_204_concat_1 w_205_0 w_206_0) : W_Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_4 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_5_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_51_0 w_52_0 w_55_0 w_180_0 w_181_0 w_184_0 w_185_0 w_188_0 w_189_0 w_49_0 w_50_0 w_53_0 w_54_0 w_179_0 w_179_1 w_179_2 w_179_3 w_179_4 w_179_5 w_182_0 w_182_1 w_182_2 w_182_3 w_182_4 w_182_5 w_183_0 w_183_1 w_183_2 w_183_3 w_183_4 w_183_5 w_186_0 w_186_1 w_186_2 w_186_3 w_186_4 w_186_5 w_187_0 w_187_1 w_187_2 w_187_3 w_187_4 w_187_5 w_190_0 w_190_1 w_190_2 w_190_3 w_190_4 w_190_5 w_191_0 w_191_1 w_191_2 w_191_3 w_191_4 w_191_5 w_192_0 w_192_1 w_192_2 w_192_3 w_192_4 w_192_5 w_194_0 w_195_0 w_196_0 w_197_0 w_199_0 w_200_0 w_201_0 w_202_0 w_203_0 w_203_concat_1 w_204_0 w_204_concat_1 w_205_0 w_206_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h33, h34, h35, h36, h37, h38, h39, h40, h41, h42⟩ := (h2 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h33, h34, h35, h36, h37, h38, h39, h40, h41, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_182.bridge h42)⟩)), h3, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_183.bridge (h4 i))), h5, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h43, h44, h45, h46, h47, h48, h49, h50, h51, h52⟩ := (h6 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h43, h44, h45, h46, h47, h48, h49, h50, h51, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_186.bridge h52)⟩)), h7, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_187.bridge (h8 i))), h9, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h53, h54, h55, h56, h57, h58, h59, h60, h61, h62⟩ := (h10 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h53, h54, h55, h56, h57, h58, h59, h60, h61, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_190.bridge h62)⟩)), h11, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_191.bridge (h12 i))), h13, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_192.bridge (h14 i))), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_3.bridge {backend hashModel tape path params params' i_0 w_5_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_52_0 w_148_0 w_149_0 w_152_0 w_153_0 w_156_0 w_157_0 w_160_0 w_161_0 w_164_0 w_165_0 w_168_0 w_169_0 w_172_0 w_173_0 w_176_0 w_177_0 w_147_0 w_147_1 w_147_2 w_147_3 w_147_4 w_147_5 w_150_0 w_150_1 w_150_2 w_150_3 w_150_4 w_150_5 w_151_0 w_151_1 w_151_2 w_151_3 w_151_4 w_151_5 w_154_0 w_154_1 w_154_2 w_154_3 w_154_4 w_154_5 w_155_0 w_155_1 w_155_2 w_155_3 w_155_4 w_155_5 w_158_0 w_158_1 w_158_2 w_158_3 w_158_4 w_158_5 w_159_0 w_159_1 w_159_2 w_159_3 w_159_4 w_159_5 w_162_0 w_162_1 w_162_2 w_162_3 w_162_4 w_162_5 w_163_0 w_163_1 w_163_2 w_163_3 w_163_4 w_163_5 w_166_0 w_166_1 w_166_2 w_166_3 w_166_4 w_166_5 w_167_0 w_167_1 w_167_2 w_167_3 w_167_4 w_167_5 w_170_0 w_170_1 w_170_2 w_170_3 w_170_4 w_170_5 w_171_0 w_171_1 w_171_2 w_171_3 w_171_4 w_171_5 w_174_0 w_174_1 w_174_2 w_174_3 w_174_4 w_174_5 w_175_0 w_175_1 w_175_2 w_175_3 w_175_4 w_175_5 w_178_0 w_178_1 w_178_2 w_178_3 w_178_4 w_178_5 w_179_0 w_179_1 w_179_2 w_179_3 w_179_4 w_179_5}
    (h : Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_3 backend hashModel tape path params i_0 w_5_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_52_0 w_148_0 w_149_0 w_152_0 w_153_0 w_156_0 w_157_0 w_160_0 w_161_0 w_164_0 w_165_0 w_168_0 w_169_0 w_172_0 w_173_0 w_176_0 w_177_0 w_147_0 w_147_1 w_147_2 w_147_3 w_147_4 w_147_5 w_150_0 w_150_1 w_150_2 w_150_3 w_150_4 w_150_5 w_151_0 w_151_1 w_151_2 w_151_3 w_151_4 w_151_5 w_154_0 w_154_1 w_154_2 w_154_3 w_154_4 w_154_5 w_155_0 w_155_1 w_155_2 w_155_3 w_155_4 w_155_5 w_158_0 w_158_1 w_158_2 w_158_3 w_158_4 w_158_5 w_159_0 w_159_1 w_159_2 w_159_3 w_159_4 w_159_5 w_162_0 w_162_1 w_162_2 w_162_3 w_162_4 w_162_5 w_163_0 w_163_1 w_163_2 w_163_3 w_163_4 w_163_5 w_166_0 w_166_1 w_166_2 w_166_3 w_166_4 w_166_5 w_167_0 w_167_1 w_167_2 w_167_3 w_167_4 w_167_5 w_170_0 w_170_1 w_170_2 w_170_3 w_170_4 w_170_5 w_171_0 w_171_1 w_171_2 w_171_3 w_171_4 w_171_5 w_174_0 w_174_1 w_174_2 w_174_3 w_174_4 w_174_5 w_175_0 w_175_1 w_175_2 w_175_3 w_175_4 w_175_5 w_178_0 w_178_1 w_178_2 w_178_3 w_178_4 w_178_5 w_179_0 w_179_1 w_179_2 w_179_3 w_179_4 w_179_5) : W_Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_3 backend hashModel params' i_0 w_5_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_52_0 w_148_0 w_149_0 w_152_0 w_153_0 w_156_0 w_157_0 w_160_0 w_161_0 w_164_0 w_165_0 w_168_0 w_169_0 w_172_0 w_173_0 w_176_0 w_177_0 w_147_0 w_147_1 w_147_2 w_147_3 w_147_4 w_147_5 w_150_0 w_150_1 w_150_2 w_150_3 w_150_4 w_150_5 w_151_0 w_151_1 w_151_2 w_151_3 w_151_4 w_151_5 w_154_0 w_154_1 w_154_2 w_154_3 w_154_4 w_154_5 w_155_0 w_155_1 w_155_2 w_155_3 w_155_4 w_155_5 w_158_0 w_158_1 w_158_2 w_158_3 w_158_4 w_158_5 w_159_0 w_159_1 w_159_2 w_159_3 w_159_4 w_159_5 w_162_0 w_162_1 w_162_2 w_162_3 w_162_4 w_162_5 w_163_0 w_163_1 w_163_2 w_163_3 w_163_4 w_163_5 w_166_0 w_166_1 w_166_2 w_166_3 w_166_4 w_166_5 w_167_0 w_167_1 w_167_2 w_167_3 w_167_4 w_167_5 w_170_0 w_170_1 w_170_2 w_170_3 w_170_4 w_170_5 w_171_0 w_171_1 w_171_2 w_171_3 w_171_4 w_171_5 w_174_0 w_174_1 w_174_2 w_174_3 w_174_4 w_174_5 w_175_0 w_175_1 w_175_2 w_175_3 w_175_4 w_175_5 w_178_0 w_178_1 w_178_2 w_178_3 w_178_4 w_178_5 w_179_0 w_179_1 w_179_2 w_179_3 w_179_4 w_179_5 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h33, h34, h35, h36, h37, h38, h39, h40, h41, h42⟩ := (h2 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h33, h34, h35, h36, h37, h38, h39, h40, h41, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_150.bridge h42)⟩)), h3, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_151.bridge (h4 i))), h5, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h43, h44, h45, h46, h47, h48, h49, h50, h51, h52⟩ := (h6 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h43, h44, h45, h46, h47, h48, h49, h50, h51, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_154.bridge h52)⟩)), h7, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_155.bridge (h8 i))), h9, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h53, h54, h55, h56, h57, h58, h59, h60, h61, h62⟩ := (h10 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h53, h54, h55, h56, h57, h58, h59, h60, h61, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_158.bridge h62)⟩)), h11, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_159.bridge (h12 i))), h13, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h63, h64, h65, h66, h67, h68, h69, h70, h71, h72⟩ := (h14 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h63, h64, h65, h66, h67, h68, h69, h70, h71, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_162.bridge h72)⟩)), h15, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_163.bridge (h16 i))), h17, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h73, h74, h75, h76, h77, h78, h79, h80, h81, h82⟩ := (h18 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h73, h74, h75, h76, h77, h78, h79, h80, h81, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_166.bridge h82)⟩)), h19, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_167.bridge (h20 i))), h21, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h83, h84, h85, h86, h87, h88, h89, h90, h91, h92⟩ := (h22 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h83, h84, h85, h86, h87, h88, h89, h90, h91, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_170.bridge h92)⟩)), h23, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_171.bridge (h24 i))), h25, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h93, h94, h95, h96, h97, h98, h99, h100, h101, h102⟩ := (h26 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h93, h94, h95, h96, h97, h98, h99, h100, h101, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_174.bridge h102)⟩)), h27, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_175.bridge (h28 i))), h29, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h103, h104, h105, h106, h107, h108, h109, h110, h111, h112⟩ := (h30 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h103, h104, h105, h106, h107, h108, h109, h110, h111, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_178.bridge h112)⟩)), h31, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_179.bridge (h32 i)))⟩)


theorem Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_2.bridge {backend hashModel tape path params params' i_0 w_5_0 w_14_0 w_15_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_52_0 w_116_0 w_117_0 w_120_0 w_121_0 w_124_0 w_125_0 w_128_0 w_129_0 w_132_0 w_133_0 w_136_0 w_137_0 w_140_0 w_141_0 w_144_0 w_145_0 w_115_0 w_115_1 w_115_2 w_115_3 w_115_4 w_115_5 w_118_0 w_118_1 w_118_2 w_118_3 w_118_4 w_118_5 w_119_0 w_119_1 w_119_2 w_119_3 w_119_4 w_119_5 w_122_0 w_122_1 w_122_2 w_122_3 w_122_4 w_122_5 w_123_0 w_123_1 w_123_2 w_123_3 w_123_4 w_123_5 w_126_0 w_126_1 w_126_2 w_126_3 w_126_4 w_126_5 w_127_0 w_127_1 w_127_2 w_127_3 w_127_4 w_127_5 w_130_0 w_130_1 w_130_2 w_130_3 w_130_4 w_130_5 w_131_0 w_131_1 w_131_2 w_131_3 w_131_4 w_131_5 w_134_0 w_134_1 w_134_2 w_134_3 w_134_4 w_134_5 w_135_0 w_135_1 w_135_2 w_135_3 w_135_4 w_135_5 w_138_0 w_138_1 w_138_2 w_138_3 w_138_4 w_138_5 w_139_0 w_139_1 w_139_2 w_139_3 w_139_4 w_139_5 w_142_0 w_142_1 w_142_2 w_142_3 w_142_4 w_142_5 w_143_0 w_143_1 w_143_2 w_143_3 w_143_4 w_143_5 w_146_0 w_146_1 w_146_2 w_146_3 w_146_4 w_146_5 w_147_0 w_147_1 w_147_2 w_147_3 w_147_4 w_147_5}
    (h : Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_2 backend hashModel tape path params i_0 w_5_0 w_14_0 w_15_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_52_0 w_116_0 w_117_0 w_120_0 w_121_0 w_124_0 w_125_0 w_128_0 w_129_0 w_132_0 w_133_0 w_136_0 w_137_0 w_140_0 w_141_0 w_144_0 w_145_0 w_115_0 w_115_1 w_115_2 w_115_3 w_115_4 w_115_5 w_118_0 w_118_1 w_118_2 w_118_3 w_118_4 w_118_5 w_119_0 w_119_1 w_119_2 w_119_3 w_119_4 w_119_5 w_122_0 w_122_1 w_122_2 w_122_3 w_122_4 w_122_5 w_123_0 w_123_1 w_123_2 w_123_3 w_123_4 w_123_5 w_126_0 w_126_1 w_126_2 w_126_3 w_126_4 w_126_5 w_127_0 w_127_1 w_127_2 w_127_3 w_127_4 w_127_5 w_130_0 w_130_1 w_130_2 w_130_3 w_130_4 w_130_5 w_131_0 w_131_1 w_131_2 w_131_3 w_131_4 w_131_5 w_134_0 w_134_1 w_134_2 w_134_3 w_134_4 w_134_5 w_135_0 w_135_1 w_135_2 w_135_3 w_135_4 w_135_5 w_138_0 w_138_1 w_138_2 w_138_3 w_138_4 w_138_5 w_139_0 w_139_1 w_139_2 w_139_3 w_139_4 w_139_5 w_142_0 w_142_1 w_142_2 w_142_3 w_142_4 w_142_5 w_143_0 w_143_1 w_143_2 w_143_3 w_143_4 w_143_5 w_146_0 w_146_1 w_146_2 w_146_3 w_146_4 w_146_5 w_147_0 w_147_1 w_147_2 w_147_3 w_147_4 w_147_5) : W_Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_2 backend hashModel params' i_0 w_5_0 w_14_0 w_15_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_52_0 w_116_0 w_117_0 w_120_0 w_121_0 w_124_0 w_125_0 w_128_0 w_129_0 w_132_0 w_133_0 w_136_0 w_137_0 w_140_0 w_141_0 w_144_0 w_145_0 w_115_0 w_115_1 w_115_2 w_115_3 w_115_4 w_115_5 w_118_0 w_118_1 w_118_2 w_118_3 w_118_4 w_118_5 w_119_0 w_119_1 w_119_2 w_119_3 w_119_4 w_119_5 w_122_0 w_122_1 w_122_2 w_122_3 w_122_4 w_122_5 w_123_0 w_123_1 w_123_2 w_123_3 w_123_4 w_123_5 w_126_0 w_126_1 w_126_2 w_126_3 w_126_4 w_126_5 w_127_0 w_127_1 w_127_2 w_127_3 w_127_4 w_127_5 w_130_0 w_130_1 w_130_2 w_130_3 w_130_4 w_130_5 w_131_0 w_131_1 w_131_2 w_131_3 w_131_4 w_131_5 w_134_0 w_134_1 w_134_2 w_134_3 w_134_4 w_134_5 w_135_0 w_135_1 w_135_2 w_135_3 w_135_4 w_135_5 w_138_0 w_138_1 w_138_2 w_138_3 w_138_4 w_138_5 w_139_0 w_139_1 w_139_2 w_139_3 w_139_4 w_139_5 w_142_0 w_142_1 w_142_2 w_142_3 w_142_4 w_142_5 w_143_0 w_143_1 w_143_2 w_143_3 w_143_4 w_143_5 w_146_0 w_146_1 w_146_2 w_146_3 w_146_4 w_146_5 w_147_0 w_147_1 w_147_2 w_147_3 w_147_4 w_147_5 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h33, h34, h35, h36, h37, h38, h39, h40, h41, h42⟩ := (h2 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h33, h34, h35, h36, h37, h38, h39, h40, h41, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_118.bridge h42)⟩)), h3, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_119.bridge (h4 i))), h5, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h43, h44, h45, h46, h47, h48, h49, h50, h51, h52⟩ := (h6 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h43, h44, h45, h46, h47, h48, h49, h50, h51, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_122.bridge h52)⟩)), h7, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_123.bridge (h8 i))), h9, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h53, h54, h55, h56, h57, h58, h59, h60, h61, h62⟩ := (h10 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h53, h54, h55, h56, h57, h58, h59, h60, h61, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_126.bridge h62)⟩)), h11, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_127.bridge (h12 i))), h13, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h63, h64, h65, h66, h67, h68, h69, h70, h71, h72⟩ := (h14 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h63, h64, h65, h66, h67, h68, h69, h70, h71, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_130.bridge h72)⟩)), h15, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_131.bridge (h16 i))), h17, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h73, h74, h75, h76, h77, h78, h79, h80, h81, h82⟩ := (h18 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h73, h74, h75, h76, h77, h78, h79, h80, h81, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_134.bridge h82)⟩)), h19, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_135.bridge (h20 i))), h21, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h83, h84, h85, h86, h87, h88, h89, h90, h91, h92⟩ := (h22 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h83, h84, h85, h86, h87, h88, h89, h90, h91, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_138.bridge h92)⟩)), h23, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_139.bridge (h24 i))), h25, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h93, h94, h95, h96, h97, h98, h99, h100, h101, h102⟩ := (h26 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h93, h94, h95, h96, h97, h98, h99, h100, h101, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_142.bridge h102)⟩)), h27, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_143.bridge (h28 i))), h29, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h103, h104, h105, h106, h107, h108, h109, h110, h111, h112⟩ := (h30 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h103, h104, h105, h106, h107, h108, h109, h110, h111, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_146.bridge h112)⟩)), h31, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_147.bridge (h32 i)))⟩)


theorem Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_1.bridge {backend hashModel tape path params params' i_0 w_5_0 w_14_0 w_15_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_52_0 w_84_0 w_85_0 w_88_0 w_89_0 w_92_0 w_93_0 w_96_0 w_97_0 w_100_0 w_101_0 w_104_0 w_105_0 w_108_0 w_109_0 w_112_0 w_113_0 w_83_0 w_83_1 w_83_2 w_83_3 w_83_4 w_83_5 w_86_0 w_86_1 w_86_2 w_86_3 w_86_4 w_86_5 w_87_0 w_87_1 w_87_2 w_87_3 w_87_4 w_87_5 w_90_0 w_90_1 w_90_2 w_90_3 w_90_4 w_90_5 w_91_0 w_91_1 w_91_2 w_91_3 w_91_4 w_91_5 w_94_0 w_94_1 w_94_2 w_94_3 w_94_4 w_94_5 w_95_0 w_95_1 w_95_2 w_95_3 w_95_4 w_95_5 w_98_0 w_98_1 w_98_2 w_98_3 w_98_4 w_98_5 w_99_0 w_99_1 w_99_2 w_99_3 w_99_4 w_99_5 w_102_0 w_102_1 w_102_2 w_102_3 w_102_4 w_102_5 w_103_0 w_103_1 w_103_2 w_103_3 w_103_4 w_103_5 w_106_0 w_106_1 w_106_2 w_106_3 w_106_4 w_106_5 w_107_0 w_107_1 w_107_2 w_107_3 w_107_4 w_107_5 w_110_0 w_110_1 w_110_2 w_110_3 w_110_4 w_110_5 w_111_0 w_111_1 w_111_2 w_111_3 w_111_4 w_111_5 w_114_0 w_114_1 w_114_2 w_114_3 w_114_4 w_114_5 w_115_0 w_115_1 w_115_2 w_115_3 w_115_4 w_115_5}
    (h : Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_1 backend hashModel tape path params i_0 w_5_0 w_14_0 w_15_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_52_0 w_84_0 w_85_0 w_88_0 w_89_0 w_92_0 w_93_0 w_96_0 w_97_0 w_100_0 w_101_0 w_104_0 w_105_0 w_108_0 w_109_0 w_112_0 w_113_0 w_83_0 w_83_1 w_83_2 w_83_3 w_83_4 w_83_5 w_86_0 w_86_1 w_86_2 w_86_3 w_86_4 w_86_5 w_87_0 w_87_1 w_87_2 w_87_3 w_87_4 w_87_5 w_90_0 w_90_1 w_90_2 w_90_3 w_90_4 w_90_5 w_91_0 w_91_1 w_91_2 w_91_3 w_91_4 w_91_5 w_94_0 w_94_1 w_94_2 w_94_3 w_94_4 w_94_5 w_95_0 w_95_1 w_95_2 w_95_3 w_95_4 w_95_5 w_98_0 w_98_1 w_98_2 w_98_3 w_98_4 w_98_5 w_99_0 w_99_1 w_99_2 w_99_3 w_99_4 w_99_5 w_102_0 w_102_1 w_102_2 w_102_3 w_102_4 w_102_5 w_103_0 w_103_1 w_103_2 w_103_3 w_103_4 w_103_5 w_106_0 w_106_1 w_106_2 w_106_3 w_106_4 w_106_5 w_107_0 w_107_1 w_107_2 w_107_3 w_107_4 w_107_5 w_110_0 w_110_1 w_110_2 w_110_3 w_110_4 w_110_5 w_111_0 w_111_1 w_111_2 w_111_3 w_111_4 w_111_5 w_114_0 w_114_1 w_114_2 w_114_3 w_114_4 w_114_5 w_115_0 w_115_1 w_115_2 w_115_3 w_115_4 w_115_5) : W_Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_1 backend hashModel params' i_0 w_5_0 w_14_0 w_15_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_52_0 w_84_0 w_85_0 w_88_0 w_89_0 w_92_0 w_93_0 w_96_0 w_97_0 w_100_0 w_101_0 w_104_0 w_105_0 w_108_0 w_109_0 w_112_0 w_113_0 w_83_0 w_83_1 w_83_2 w_83_3 w_83_4 w_83_5 w_86_0 w_86_1 w_86_2 w_86_3 w_86_4 w_86_5 w_87_0 w_87_1 w_87_2 w_87_3 w_87_4 w_87_5 w_90_0 w_90_1 w_90_2 w_90_3 w_90_4 w_90_5 w_91_0 w_91_1 w_91_2 w_91_3 w_91_4 w_91_5 w_94_0 w_94_1 w_94_2 w_94_3 w_94_4 w_94_5 w_95_0 w_95_1 w_95_2 w_95_3 w_95_4 w_95_5 w_98_0 w_98_1 w_98_2 w_98_3 w_98_4 w_98_5 w_99_0 w_99_1 w_99_2 w_99_3 w_99_4 w_99_5 w_102_0 w_102_1 w_102_2 w_102_3 w_102_4 w_102_5 w_103_0 w_103_1 w_103_2 w_103_3 w_103_4 w_103_5 w_106_0 w_106_1 w_106_2 w_106_3 w_106_4 w_106_5 w_107_0 w_107_1 w_107_2 w_107_3 w_107_4 w_107_5 w_110_0 w_110_1 w_110_2 w_110_3 w_110_4 w_110_5 w_111_0 w_111_1 w_111_2 w_111_3 w_111_4 w_111_5 w_114_0 w_114_1 w_114_2 w_114_3 w_114_4 w_114_5 w_115_0 w_115_1 w_115_2 w_115_3 w_115_4 w_115_5 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h33, h34, h35, h36, h37, h38, h39, h40, h41, h42⟩ := (h2 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h33, h34, h35, h36, h37, h38, h39, h40, h41, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_86.bridge h42)⟩)), h3, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_87.bridge (h4 i))), h5, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h43, h44, h45, h46, h47, h48, h49, h50, h51, h52⟩ := (h6 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h43, h44, h45, h46, h47, h48, h49, h50, h51, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_90.bridge h52)⟩)), h7, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_91.bridge (h8 i))), h9, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h53, h54, h55, h56, h57, h58, h59, h60, h61, h62⟩ := (h10 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h53, h54, h55, h56, h57, h58, h59, h60, h61, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_94.bridge h62)⟩)), h11, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_95.bridge (h12 i))), h13, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h63, h64, h65, h66, h67, h68, h69, h70, h71, h72⟩ := (h14 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h63, h64, h65, h66, h67, h68, h69, h70, h71, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_98.bridge h72)⟩)), h15, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_99.bridge (h16 i))), h17, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h73, h74, h75, h76, h77, h78, h79, h80, h81, h82⟩ := (h18 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h73, h74, h75, h76, h77, h78, h79, h80, h81, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_102.bridge h82)⟩)), h19, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_103.bridge (h20 i))), h21, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h83, h84, h85, h86, h87, h88, h89, h90, h91, h92⟩ := (h22 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h83, h84, h85, h86, h87, h88, h89, h90, h91, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_106.bridge h92)⟩)), h23, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_107.bridge (h24 i))), h25, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h93, h94, h95, h96, h97, h98, h99, h100, h101, h102⟩ := (h26 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h93, h94, h95, h96, h97, h98, h99, h100, h101, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_110.bridge h102)⟩)), h27, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_111.bridge (h28 i))), h29, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h103, h104, h105, h106, h107, h108, h109, h110, h111, h112⟩ := (h30 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h103, h104, h105, h106, h107, h108, h109, h110, h111, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_114.bridge h112)⟩)), h31, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_115.bridge (h32 i)))⟩)


theorem Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_0.bridge {backend hashModel tape path params params' i_0 w_5_0 w_14_0 w_15_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_51_0 w_52_0 w_55_0 w_56_0 w_57_0 w_60_0 w_61_0 w_64_0 w_65_0 w_68_0 w_69_0 w_72_0 w_73_0 w_76_0 w_77_0 w_80_0 w_81_0 w_49_0 w_50_0 w_53_0 w_54_0 w_58_0 w_58_1 w_58_2 w_58_3 w_58_4 w_58_5 w_59_0 w_59_1 w_59_2 w_59_3 w_59_4 w_59_5 w_62_0 w_62_1 w_62_2 w_62_3 w_62_4 w_62_5 w_63_0 w_63_1 w_63_2 w_63_3 w_63_4 w_63_5 w_66_0 w_66_1 w_66_2 w_66_3 w_66_4 w_66_5 w_67_0 w_67_1 w_67_2 w_67_3 w_67_4 w_67_5 w_70_0 w_70_1 w_70_2 w_70_3 w_70_4 w_70_5 w_71_0 w_71_1 w_71_2 w_71_3 w_71_4 w_71_5 w_74_0 w_74_1 w_74_2 w_74_3 w_74_4 w_74_5 w_75_0 w_75_1 w_75_2 w_75_3 w_75_4 w_75_5 w_78_0 w_78_1 w_78_2 w_78_3 w_78_4 w_78_5 w_79_0 w_79_1 w_79_2 w_79_3 w_79_4 w_79_5 w_82_0 w_82_1 w_82_2 w_82_3 w_82_4 w_82_5 w_83_0 w_83_1 w_83_2 w_83_3 w_83_4 w_83_5}
    (h : Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_0 backend hashModel tape path params i_0 w_5_0 w_14_0 w_15_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_51_0 w_52_0 w_55_0 w_56_0 w_57_0 w_60_0 w_61_0 w_64_0 w_65_0 w_68_0 w_69_0 w_72_0 w_73_0 w_76_0 w_77_0 w_80_0 w_81_0 w_49_0 w_50_0 w_53_0 w_54_0 w_58_0 w_58_1 w_58_2 w_58_3 w_58_4 w_58_5 w_59_0 w_59_1 w_59_2 w_59_3 w_59_4 w_59_5 w_62_0 w_62_1 w_62_2 w_62_3 w_62_4 w_62_5 w_63_0 w_63_1 w_63_2 w_63_3 w_63_4 w_63_5 w_66_0 w_66_1 w_66_2 w_66_3 w_66_4 w_66_5 w_67_0 w_67_1 w_67_2 w_67_3 w_67_4 w_67_5 w_70_0 w_70_1 w_70_2 w_70_3 w_70_4 w_70_5 w_71_0 w_71_1 w_71_2 w_71_3 w_71_4 w_71_5 w_74_0 w_74_1 w_74_2 w_74_3 w_74_4 w_74_5 w_75_0 w_75_1 w_75_2 w_75_3 w_75_4 w_75_5 w_78_0 w_78_1 w_78_2 w_78_3 w_78_4 w_78_5 w_79_0 w_79_1 w_79_2 w_79_3 w_79_4 w_79_5 w_82_0 w_82_1 w_82_2 w_82_3 w_82_4 w_82_5 w_83_0 w_83_1 w_83_2 w_83_3 w_83_4 w_83_5) : W_Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_0 backend hashModel params' i_0 w_5_0 w_14_0 w_15_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_51_0 w_52_0 w_55_0 w_56_0 w_57_0 w_60_0 w_61_0 w_64_0 w_65_0 w_68_0 w_69_0 w_72_0 w_73_0 w_76_0 w_77_0 w_80_0 w_81_0 w_49_0 w_50_0 w_53_0 w_54_0 w_58_0 w_58_1 w_58_2 w_58_3 w_58_4 w_58_5 w_59_0 w_59_1 w_59_2 w_59_3 w_59_4 w_59_5 w_62_0 w_62_1 w_62_2 w_62_3 w_62_4 w_62_5 w_63_0 w_63_1 w_63_2 w_63_3 w_63_4 w_63_5 w_66_0 w_66_1 w_66_2 w_66_3 w_66_4 w_66_5 w_67_0 w_67_1 w_67_2 w_67_3 w_67_4 w_67_5 w_70_0 w_70_1 w_70_2 w_70_3 w_70_4 w_70_5 w_71_0 w_71_1 w_71_2 w_71_3 w_71_4 w_71_5 w_74_0 w_74_1 w_74_2 w_74_3 w_74_4 w_74_5 w_75_0 w_75_1 w_75_2 w_75_3 w_75_4 w_75_5 w_78_0 w_78_1 w_78_2 w_78_3 w_78_4 w_78_5 w_79_0 w_79_1 w_79_2 w_79_3 w_79_4 w_79_5 w_82_0 w_82_1 w_82_2 w_82_3 w_82_4 w_82_5 w_83_0 w_83_1 w_83_2 w_83_3 w_83_4 w_83_5 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h33, h34, h35, h36, h37, h38, h39, h40, h41, h42⟩ := (h6 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h33, h34, h35, h36, h37, h38, h39, h40, h41, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_58.bridge h42)⟩)), h7, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_59.bridge (h8 i))), h9, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h43, h44, h45, h46, h47, h48, h49, h50, h51, h52⟩ := (h10 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h43, h44, h45, h46, h47, h48, h49, h50, h51, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_62.bridge h52)⟩)), h11, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_63.bridge (h12 i))), h13, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h53, h54, h55, h56, h57, h58, h59, h60, h61, h62⟩ := (h14 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h53, h54, h55, h56, h57, h58, h59, h60, h61, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_66.bridge h62)⟩)), h15, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_67.bridge (h16 i))), h17, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h63, h64, h65, h66, h67, h68, h69, h70, h71, h72⟩ := (h18 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h63, h64, h65, h66, h67, h68, h69, h70, h71, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_70.bridge h72)⟩)), h19, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_71.bridge (h20 i))), h21, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h73, h74, h75, h76, h77, h78, h79, h80, h81, h82⟩ := (h22 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h73, h74, h75, h76, h77, h78, h79, h80, h81, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_74.bridge h82)⟩)), h23, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_75.bridge (h24 i))), h25, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h83, h84, h85, h86, h87, h88, h89, h90, h91, h92⟩ := (h26 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h83, h84, h85, h86, h87, h88, h89, h90, h91, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_78.bridge h92)⟩)), h27, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_79.bridge (h28 i))), h29, (fun i ↦ (by obtain ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h93, h94, h95, h96, h97, h98, h99, h100, h101, h102⟩ := (h30 i); exact ⟨w_14_0_gathered, w_15_0_gathered, w_52_0_gathered, h93, h94, h95, h96, h97, h98, h99, h100, h101, (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_82.bridge h102)⟩)), h31, (fun i ↦ (Stage_sigma_online_eval.parallel_parallel_generatedRoot_188_83.bridge (h32 i)))⟩)


theorem Stage_sigma_online_eval.parallel_generatedRoot_188.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_sigma_online_eval.parallel_generatedRoot_188 backend hashModel tape path params i_0 inputs outputs) : W_Stage_sigma_online_eval.parallel_generatedRoot_188 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_sigma_online_eval.parallel_generatedRoot_188
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_49_0, wit1.w_50_0, wit1.w_53_0, wit1.w_54_0, wit1.w_58_0, wit1.w_58_1, wit1.w_58_2, wit1.w_58_3, wit1.w_58_4, wit1.w_58_5, wit1.w_59_0, wit1.w_59_1, wit1.w_59_2, wit1.w_59_3, wit1.w_59_4, wit1.w_59_5, wit1.w_62_0, wit1.w_62_1, wit1.w_62_2, wit1.w_62_3, wit1.w_62_4, wit1.w_62_5, wit1.w_63_0, wit1.w_63_1, wit1.w_63_2, wit1.w_63_3, wit1.w_63_4, wit1.w_63_5, wit1.w_66_0, wit1.w_66_1, wit1.w_66_2, wit1.w_66_3, wit1.w_66_4, wit1.w_66_5, wit1.w_67_0, wit1.w_67_1, wit1.w_67_2, wit1.w_67_3, wit1.w_67_4, wit1.w_67_5, wit1.w_70_0, wit1.w_70_1, wit1.w_70_2, wit1.w_70_3, wit1.w_70_4, wit1.w_70_5, wit1.w_71_0, wit1.w_71_1, wit1.w_71_2, wit1.w_71_3, wit1.w_71_4, wit1.w_71_5, wit1.w_74_0, wit1.w_74_1, wit1.w_74_2, wit1.w_74_3, wit1.w_74_4, wit1.w_74_5, wit1.w_75_0, wit1.w_75_1, wit1.w_75_2, wit1.w_75_3, wit1.w_75_4, wit1.w_75_5, wit1.w_78_0, wit1.w_78_1, wit1.w_78_2, wit1.w_78_3, wit1.w_78_4, wit1.w_78_5, wit1.w_79_0, wit1.w_79_1, wit1.w_79_2, wit1.w_79_3, wit1.w_79_4, wit1.w_79_5, wit1.w_82_0, wit1.w_82_1, wit1.w_82_2, wit1.w_82_3, wit1.w_82_4, wit1.w_82_5, wit1.w_83_0, wit1.w_83_1, wit1.w_83_2, wit1.w_83_3, wit1.w_83_4, wit1.w_83_5, wit1.w_86_0, wit1.w_86_1, wit1.w_86_2, wit1.w_86_3, wit1.w_86_4, wit1.w_86_5, wit1.w_87_0, wit1.w_87_1, wit1.w_87_2, wit1.w_87_3, wit1.w_87_4, wit1.w_87_5, wit1.w_90_0, wit1.w_90_1, wit1.w_90_2, wit1.w_90_3, wit1.w_90_4, wit1.w_90_5, wit1.w_91_0, wit1.w_91_1, wit1.w_91_2, wit1.w_91_3, wit1.w_91_4, wit1.w_91_5, wit1.w_94_0, wit1.w_94_1, wit1.w_94_2, wit1.w_94_3, wit1.w_94_4, wit1.w_94_5, wit1.w_95_0, wit1.w_95_1, wit1.w_95_2, wit1.w_95_3, wit1.w_95_4, wit1.w_95_5, wit1.w_98_0, wit1.w_98_1, wit1.w_98_2, wit1.w_98_3, wit1.w_98_4, wit1.w_98_5, wit1.w_99_0, wit1.w_99_1, wit1.w_99_2, wit1.w_99_3, wit1.w_99_4, wit1.w_99_5, wit1.w_102_0, wit1.w_102_1, wit1.w_102_2, wit1.w_102_3, wit1.w_102_4, wit1.w_102_5, wit1.w_103_0, wit1.w_103_1, wit1.w_103_2, wit1.w_103_3, wit1.w_103_4, wit1.w_103_5, wit1.w_106_0, wit1.w_106_1, wit1.w_106_2, wit1.w_106_3, wit1.w_106_4, wit1.w_106_5, wit1.w_107_0, wit1.w_107_1, wit1.w_107_2, wit1.w_107_3, wit1.w_107_4, wit1.w_107_5, wit1.w_110_0, wit1.w_110_1, wit1.w_110_2, wit1.w_110_3, wit1.w_110_4, wit1.w_110_5, wit1.w_111_0, wit1.w_111_1, wit1.w_111_2, wit1.w_111_3, wit1.w_111_4, wit1.w_111_5, wit1.w_114_0, wit1.w_114_1, wit1.w_114_2, wit1.w_114_3, wit1.w_114_4, wit1.w_114_5, wit1.w_115_0, wit1.w_115_1, wit1.w_115_2, wit1.w_115_3, wit1.w_115_4, wit1.w_115_5, wit1.w_118_0, wit1.w_118_1, wit1.w_118_2, wit1.w_118_3, wit1.w_118_4, wit1.w_118_5, wit1.w_119_0, wit1.w_119_1, wit1.w_119_2, wit1.w_119_3, wit1.w_119_4, wit1.w_119_5, wit1.w_122_0, wit1.w_122_1, wit1.w_122_2, wit1.w_122_3, wit1.w_122_4, wit1.w_122_5, wit1.w_123_0, wit1.w_123_1, wit1.w_123_2, wit1.w_123_3, wit1.w_123_4, wit1.w_123_5, wit1.w_126_0, wit1.w_126_1, wit1.w_126_2, wit1.w_126_3, wit1.w_126_4, wit1.w_126_5, wit1.w_127_0, wit1.w_127_1, wit1.w_127_2, wit1.w_127_3, wit1.w_127_4, wit1.w_127_5, wit1.w_130_0, wit1.w_130_1, wit1.w_130_2, wit1.w_130_3, wit1.w_130_4, wit1.w_130_5, wit1.w_131_0, wit1.w_131_1, wit1.w_131_2, wit1.w_131_3, wit1.w_131_4, wit1.w_131_5, wit1.w_134_0, wit1.w_134_1, wit1.w_134_2, wit1.w_134_3, wit1.w_134_4, wit1.w_134_5, wit1.w_135_0, wit1.w_135_1, wit1.w_135_2, wit1.w_135_3, wit1.w_135_4, wit1.w_135_5, wit1.w_138_0, wit1.w_138_1, wit1.w_138_2, wit1.w_138_3, wit1.w_138_4, wit1.w_138_5, wit1.w_139_0, wit1.w_139_1, wit1.w_139_2, wit1.w_139_3, wit1.w_139_4, wit1.w_139_5, wit1.w_142_0, wit1.w_142_1, wit1.w_142_2, wit1.w_142_3, wit1.w_142_4, wit1.w_142_5, wit1.w_143_0, wit1.w_143_1, wit1.w_143_2, wit1.w_143_3, wit1.w_143_4, wit1.w_143_5, wit1.w_146_0, wit1.w_146_1, wit1.w_146_2, wit1.w_146_3, wit1.w_146_4, wit1.w_146_5, wit1.w_147_0, wit1.w_147_1, wit1.w_147_2, wit1.w_147_3, wit1.w_147_4, wit1.w_147_5, wit1.w_150_0, wit1.w_150_1, wit1.w_150_2, wit1.w_150_3, wit1.w_150_4, wit1.w_150_5, wit1.w_151_0, wit1.w_151_1, wit1.w_151_2, wit1.w_151_3, wit1.w_151_4, wit1.w_151_5, wit1.w_154_0, wit1.w_154_1, wit1.w_154_2, wit1.w_154_3, wit1.w_154_4, wit1.w_154_5, wit1.w_155_0, wit1.w_155_1, wit1.w_155_2, wit1.w_155_3, wit1.w_155_4, wit1.w_155_5, wit1.w_158_0, wit1.w_158_1, wit1.w_158_2, wit1.w_158_3, wit1.w_158_4, wit1.w_158_5, wit1.w_159_0, wit1.w_159_1, wit1.w_159_2, wit1.w_159_3, wit1.w_159_4, wit1.w_159_5, wit1.w_162_0, wit1.w_162_1, wit1.w_162_2, wit1.w_162_3, wit1.w_162_4, wit1.w_162_5, wit1.w_163_0, wit1.w_163_1, wit1.w_163_2, wit1.w_163_3, wit1.w_163_4, wit1.w_163_5, wit1.w_166_0, wit1.w_166_1, wit1.w_166_2, wit1.w_166_3, wit1.w_166_4, wit1.w_166_5, wit1.w_167_0, wit1.w_167_1, wit1.w_167_2, wit1.w_167_3, wit1.w_167_4, wit1.w_167_5, wit1.w_170_0, wit1.w_170_1, wit1.w_170_2, wit1.w_170_3, wit1.w_170_4, wit1.w_170_5, wit1.w_171_0, wit1.w_171_1, wit1.w_171_2, wit1.w_171_3, wit1.w_171_4, wit1.w_171_5, wit1.w_174_0, wit1.w_174_1, wit1.w_174_2, wit1.w_174_3, wit1.w_174_4, wit1.w_174_5, wit1.w_175_0, wit1.w_175_1, wit1.w_175_2, wit1.w_175_3, wit1.w_175_4, wit1.w_175_5, wit1.w_178_0, wit1.w_178_1, wit1.w_178_2, wit1.w_178_3, wit1.w_178_4, wit1.w_178_5, wit1.w_179_0, wit1.w_179_1, wit1.w_179_2, wit1.w_179_3, wit1.w_179_4, wit1.w_179_5, wit1.w_182_0, wit1.w_182_1, wit1.w_182_2, wit1.w_182_3, wit1.w_182_4, wit1.w_182_5, wit1.w_183_0, wit1.w_183_1, wit1.w_183_2, wit1.w_183_3, wit1.w_183_4, wit1.w_183_5, wit1.w_186_0, wit1.w_186_1, wit1.w_186_2, wit1.w_186_3, wit1.w_186_4, wit1.w_186_5, wit1.w_187_0, wit1.w_187_1, wit1.w_187_2, wit1.w_187_3, wit1.w_187_4, wit1.w_187_5, wit1.w_190_0, wit1.w_190_1, wit1.w_190_2, wit1.w_190_3, wit1.w_190_4, wit1.w_190_5, wit1.w_191_0, wit1.w_191_1, wit1.w_191_2, wit1.w_191_3, wit1.w_191_4, wit1.w_191_5, wit1.w_192_0, wit1.w_192_1, wit1.w_192_2, wit1.w_192_3, wit1.w_192_4, wit1.w_192_5, wit1.w_194_0, wit1.w_195_0, wit1.w_196_0, wit1.w_197_0, wit1.w_199_0, wit1.w_200_0, wit1.w_201_0, wit1.w_202_0, wit1.w_203_0, wit1.w_203_concat_1, wit1.w_204_0, wit1.w_204_concat_1, wit1.w_205_0, wit1.w_206_0, wit1.w_208_0, wit1.w_209_0, wit1.w_210_0, wit1.w_212_0, wit1.w_213_0, wit1.w_214_0, wit1.w_215_0, wit1.w_216_0, wit1.w_217_0, wit1.w_218_0, wit1.w_218_concat_1, wit1.w_219_0, wit1.w_221_0, wit1.w_222_0, wit1.w_223_0, wit1.w_225_0, wit1.w_226_0, wit1.w_227_0, wit1.w_228_0, wit1.w_229_0, wit1.w_230_0, wit1.w_231_0, wit1.w_231_concat_1, wit1.w_232_0, wit1.w_233_0, wit1.w_235_0, wit1.w_236_0, wit1.w_237_0, wit1.w_239_0, wit1.w_240_0, wit1.w_241_0, wit1.w_242_0, wit1.w_243_0, wit1.w_244_0, wit1.w_245_0, wit1.w_245_concat_1, wit1.w_246_0, wit1.w_248_0, wit1.w_249_0, wit1.w_250_0, wit1.w_252_0, wit1.w_253_0, wit1.w_254_0, wit1.w_255_0, wit1.w_256_0, wit1.w_257_0, wit1.w_258_0, wit1.w_258_concat_1, wit1.w_259_0, wit1.w_260_0, wit1.w_262_0, wit1.w_263_0, wit1.w_264_0, wit1.w_266_0, wit1.w_267_0, wit1.w_268_0, wit1.w_269_0, wit1.w_270_0, wit1.w_271_0, wit1.w_272_0, wit1.w_272_concat_1, wit1.w_273_0, wit1.w_275_0, wit1.w_276_0, wit1.w_277_0, wit1.w_279_0, wit1.w_280_0, wit1.w_281_0, wit1.w_282_0, wit1.w_283_0, wit1.w_284_0, wit1.w_285_0, wit1.w_285_concat_1, wit1.w_286_0, wit1.w_287_0, wit1.w_289_0, wit1.w_290_0, wit1.w_291_0, wit1.w_293_0, wit1.w_294_0, wit1.w_295_0, wit1.w_296_0, wit1.w_297_0, wit1.w_298_0, wit1.w_299_0, wit1.w_299_concat_1, wit1.w_300_0, wit1.w_302_0, wit1.w_303_0, wit1.w_304_0, wit1.w_306_0, wit1.w_307_0, wit1.w_308_0, wit1.w_309_0, wit1.w_310_0, wit1.w_311_0, wit1.w_312_0, wit1.w_312_concat_1, wit1.w_313_0, wit1.w_314_0, wit1.w_316_0, wit1.w_317_0, wit1.w_318_0, wit1.w_320_0, wit1.w_321_0, wit1.w_322_0, wit1.w_323_0, wit1.w_324_0, wit1.w_325_0, wit1.w_326_0, wit1.w_326_concat_1, wit1.w_327_0, wit1.w_329_0, wit1.w_330_0, wit1.w_331_0, wit1.w_333_0, wit1.w_334_0, wit1.w_335_0, wit1.w_336_0, wit1.w_337_0, wit1.w_338_0, wit1.w_339_0, wit1.w_339_concat_1, wit1.w_340_0, wit1.w_341_0, wit1.w_343_0, wit1.w_344_0, wit1.w_345_0, wit1.w_347_0, wit1.w_348_0, wit1.w_349_0, wit1.w_350_0, wit1.w_351_0, wit1.w_352_0, wit1.w_353_0, wit1.w_353_concat_1, wit1.w_354_0, wit1.w_356_0, wit1.w_357_0, wit1.w_358_0, wit1.w_360_0, wit1.w_361_0, wit1.w_362_0, wit1.w_363_0, wit1.w_364_0, wit1.w_365_0, wit1.w_366_0, wit1.w_366_concat_1, wit1.w_367_0, wit1.w_368_0, wit1.w_370_0, wit1.w_371_0, wit1.w_372_0, wit1.w_374_0, wit1.w_375_0, wit1.w_376_0, wit1.w_377_0, wit1.w_378_0, wit1.w_379_0, wit1.w_380_0, wit1.w_380_concat_1, wit1.w_381_0, wit1.w_383_0, wit1.w_384_0, wit1.w_385_0, wit1.w_387_0, wit1.w_388_0, wit1.w_389_0, wit1.w_390_0, wit1.w_391_0, wit1.w_392_0, wit1.w_393_0, wit1.w_393_concat_1, wit1.w_394_0, wit1.w_395_0, wit1.w_397_0, wit1.w_398_0, wit1.w_399_0, wit1.w_401_0, wit1.w_402_0, wit1.w_403_0, wit1.w_404_0, wit1.w_405_0, wit1.w_406_0, wit1.w_407_0, wit1.w_407_concat_1, wit1.w_408_0, wit1.w_410_0, wit1.w_411_0, wit1.w_412_0, wit1.w_414_0, wit1.w_415_0, wit1.w_416_0, wit1.w_417_0, wit1.w_418_0, wit1.w_419_0, wit1.w_420_0, wit1.w_420_concat_1, wit1.w_421_0, wit1.w_422_0, wit1.w_424_0, wit1.w_425_0, wit1.w_426_0, wit1.w_428_0, wit1.w_429_0, wit1.w_430_0, wit1.w_431_0, wit1.w_432_0, wit1.w_433_0, wit1.w_434_0, wit1.w_434_concat_1, wit1.w_435_0, wit1.w_437_0, wit1.w_438_0, wit1.w_439_0, wit1.w_441_0, wit1.w_442_0, wit1.w_443_0, wit1.w_444_0, wit1.w_445_0, wit1.w_446_0, wit1.w_447_0, wit1.w_447_concat_1, wit1.w_448_0, wit1.w_449_0, wit1.w_451_0, wit1.w_452_0, wit1.w_453_0, wit1.w_455_0, wit1.w_456_0, wit1.w_457_0, wit1.w_458_0, wit1.w_459_0, wit1.w_460_0, wit1.w_461_0, wit1.w_461_concat_1, wit1.w_462_0, wit1.w_464_0, wit1.w_465_0, wit1.w_466_0, wit1.w_468_0, wit1.w_469_0, wit1.w_470_0, wit1.w_471_0, wit1.w_472_0, wit1.w_473_0, wit1.w_474_0, wit1.w_474_concat_1, wit1.w_475_0, wit1.w_476_0, wit1.w_478_0, wit1.w_479_0, wit1.w_480_0, wit1.w_482_0, wit1.w_483_0, wit1.w_484_0, wit1.w_485_0, wit1.w_486_0, wit1.w_487_0, wit1.w_488_0, wit1.w_488_concat_1, wit1.w_489_0, wit1.w_491_0, wit1.w_492_0, wit1.w_493_0, wit1.w_495_0, wit1.w_496_0, wit1.w_497_0, wit1.w_498_0, wit1.w_499_0, wit1.w_500_0, wit1.w_501_0, wit1.w_501_concat_1, wit1.w_502_0, wit1.w_503_0, wit1.w_505_0, wit1.w_506_0, wit1.w_507_0, wit1.w_509_0, wit1.w_510_0, wit1.w_511_0, wit1.w_512_0, wit1.w_513_0, wit1.w_514_0, wit1.w_515_0, wit1.w_515_concat_1, wit1.w_516_0, wit1.w_518_0, wit1.w_519_0, wit1.w_520_0, wit1.w_522_0, wit1.w_523_0, wit1.w_524_0, wit1.w_525_0, wit1.w_526_0, wit1.w_527_0, wit1.w_528_0, wit1.w_528_concat_1, wit1.w_529_0, wit1.w_530_0, wit1.w_532_0, wit1.w_533_0, wit1.w_534_0, wit1.w_536_0, wit1.w_537_0, wit1.w_538_0, wit1.w_539_0, wit1.w_540_0, wit1.w_541_0, wit1.w_542_0, wit1.w_542_concat_1, wit1.w_543_0, wit1.w_545_0, wit1.w_546_0, wit1.w_547_0, wit1.w_549_0, wit1.w_550_0, wit1.w_551_0, wit1.w_552_0, wit1.w_553_0, wit1.w_554_0, wit1.w_555_0, wit1.w_555_concat_1, wit1.w_556_0, wit1.w_557_0, wit1.w_559_0, wit1.w_560_0, wit1.w_561_0, wit1.w_563_0, wit1.w_564_0, wit1.w_565_0, wit1.w_566_0, wit1.w_567_0, wit1.w_568_0, wit1.w_569_0, wit1.w_569_concat_1, wit1.w_570_0, wit1.w_572_0, wit1.w_573_0, wit1.w_574_0, wit1.w_576_0, wit1.w_577_0, wit1.w_578_0, wit1.w_579_0, wit1.w_580_0, wit1.w_581_0, wit1.w_582_0, wit1.w_582_concat_1, wit1.w_583_0, wit1.w_584_0, wit1.w_586_0, wit1.w_587_0, wit1.w_588_0, wit1.w_590_0, wit1.w_591_0, wit1.w_592_0, wit1.w_593_0, wit1.w_594_0, wit1.w_595_0, wit1.w_596_0, wit1.w_596_concat_1, wit1.w_597_0, wit1.w_599_0, wit1.w_600_0, wit1.w_601_0, wit1.w_603_0, wit1.w_604_0, wit1.w_605_0, wit1.w_606_0, wit1.w_607_0, wit1.w_608_0, wit1.w_609_0, wit1.w_609_concat_1, wit1.w_610_0, wit1.w_611_0, wit1.w_613_0, wit1.w_614_0, wit1.w_615_0, wit1.w_617_0, wit1.w_618_0, wit1.w_619_0, wit1.w_620_0, wit1.w_621_0, wit1.w_622_0, wit1.w_623_0, wit1.w_623_concat_1, wit1.w_624_0, wit1.w_626_0, wit1.w_627_0, wit1.w_628_0, wit1.w_630_0, wit1.w_631_0, wit1.w_632_0, wit1.w_633_0, wit1.w_634_0, wit1.w_635_0, wit1.w_636_0, wit1.w_636_concat_1, wit1.w_637_0, wit1.w_638_0, wit1.w_640_0, wit1.w_641_0, wit1.w_642_0, wit1.w_644_0, wit1.w_645_0, wit1.w_646_0, wit1.w_647_0, wit1.w_648_0, wit1.w_649_0, wit1.w_650_0, wit1.w_650_concat_1, wit1.w_651_0, wit1.w_653_0, wit1.w_654_0, wit1.w_655_0, wit1.w_657_0, wit1.w_658_0, wit1.w_659_0, wit1.w_660_0, wit1.w_661_0, wit1.w_662_0, wit1.w_663_0, wit1.w_663_concat_1, wit1.w_664_0, wit1.w_665_0, wit1.w_667_0, wit1.w_668_0, wit1.w_669_0, wit1.w_671_0, wit1.w_672_0, wit1.w_673_0, wit1.w_674_0, wit1.w_675_0, wit1.w_676_0, wit1.w_677_0, wit1.w_677_concat_1, wit1.w_678_0, wit1.w_680_0, wit1.w_681_0, wit1.w_682_0, wit1.w_684_0, wit1.w_685_0, wit1.w_686_0, wit1.w_687_0, wit1.w_688_0, wit1.w_689_0, wit1.w_690_0, wit1.w_690_concat_1, wit1.w_691_0, wit1.w_692_0, wit1.w_694_0, wit1.w_695_0, wit1.w_696_0, wit1.w_698_0, wit1.w_699_0, wit1.w_700_0, wit1.w_701_0, wit1.w_702_0, wit1.w_703_0, wit1.w_704_0, wit1.w_704_concat_1, wit1.w_705_0, wit1.w_707_0, wit1.w_708_0, wit1.w_709_0, wit1.w_711_0, wit1.w_712_0, wit1.w_713_0, wit1.w_714_0, wit1.w_715_0, wit1.w_716_0, wit1.w_717_0, wit1.w_717_concat_1, wit1.w_718_0, wit1.w_719_0, wit1.w_721_0, wit1.w_722_0, wit1.w_723_0, wit1.w_725_0, wit1.w_726_0, wit1.w_727_0, wit1.w_728_0, wit1.w_729_0, wit1.w_730_0, wit1.w_731_0, wit1.w_731_concat_1, wit1.w_732_0, wit1.w_734_0, wit1.w_735_0, wit1.w_736_0, wit1.w_738_0, wit1.w_739_0, wit1.w_740_0, wit1.w_741_0, wit1.w_742_0, wit1.w_743_0, wit1.w_744_0, wit1.w_744_concat_1, wit1.w_745_0, wit1.w_746_0, wit1.w_748_0, wit1.w_749_0, wit1.w_750_0, wit1.w_752_0, wit1.w_753_0, wit1.w_754_0, wit1.w_755_0, wit1.w_756_0, wit1.w_757_0, wit1.w_758_0, wit1.w_758_concat_1, wit1.w_759_0, wit1.w_761_0, wit1.w_762_0, wit1.w_763_0, wit1.w_765_0, wit1.w_766_0, wit1.w_767_0, wit1.w_768_0, wit1.w_769_0, wit1.w_770_0, wit1.w_771_0, wit1.w_771_concat_1, wit1.w_772_0, wit1.w_773_0, wit1.w_775_0, wit1.w_776_0, wit1.w_777_0, wit1.w_779_0, wit1.w_780_0, wit1.w_781_0, wit1.w_782_0, wit1.w_783_0, wit1.w_784_0, wit1.w_785_0, wit1.w_785_concat_1, wit1.w_786_0, wit1.w_788_0, wit1.w_789_0, wit1.w_790_0, wit1.w_792_0, wit1.w_793_0, wit1.w_794_0, wit1.w_795_0, wit1.w_796_0, wit1.w_797_0, wit1.w_798_0, wit1.w_798_concat_1, wit1.w_799_0, wit1.w_800_0, wit1.w_802_0, wit1.w_803_0, wit1.w_804_0, wit1.w_806_0, wit1.w_807_0, wit1.w_808_0, wit1.w_809_0, wit1.w_810_0, wit1.w_811_0, wit1.w_812_0, wit1.w_812_concat_1, wit1.w_813_0, wit1.w_815_0, wit1.w_816_0, wit1.w_817_0, wit1.w_819_0, wit1.w_820_0, wit1.w_821_0, wit1.w_822_0, wit1.w_823_0, wit1.w_824_0, wit1.w_825_0, wit1.w_825_concat_1, wit1.w_826_0, wit1.w_827_0, wit1.w_829_0, wit1.w_830_0, wit1.w_831_0, wit1.w_833_0, wit1.w_834_0, wit1.w_835_0, wit1.w_836_0, wit1.w_837_0, wit1.w_838_0, wit1.w_839_0, wit1.w_839_concat_1, wit1.w_840_0, wit1.w_842_0, wit1.w_843_0, wit1.w_844_0, wit1.w_846_0, wit1.w_847_0, wit1.w_848_0, wit1.w_849_0, wit1.w_850_0, wit1.w_851_0, wit1.w_852_0, wit1.w_852_concat_1, wit1.w_853_0, wit1.w_854_0, wit1.w_856_0, wit1.w_857_0, wit1.w_858_0, wit1.w_860_0, wit1.w_861_0, wit1.w_862_0, wit1.w_863_0, wit1.w_864_0, wit1.w_865_0, wit1.w_866_0, wit1.w_866_concat_1, wit1.w_867_0, wit1.w_869_0, wit1.w_870_0, wit1.w_871_0, wit1.w_873_0, wit1.w_874_0, wit1.w_875_0, wit1.w_876_0, wit1.w_877_0, wit1.w_878_0, wit1.w_879_0, wit1.w_879_concat_1, wit1.w_880_0, wit1.w_881_0, wit1.w_883_0, wit1.w_884_0, wit1.w_885_0, wit1.w_887_0, wit1.w_888_0, wit1.w_889_0, wit1.w_890_0, wit1.w_891_0, wit1.w_892_0, wit1.w_893_0, wit1.w_893_concat_1, wit1.w_894_0, wit1.w_896_0, wit1.w_897_0, wit1.w_898_0, wit1.w_900_0, wit1.w_901_0, wit1.w_902_0, wit1.w_903_0, wit1.w_904_0, wit1.w_905_0, wit1.w_906_0, wit1.w_906_concat_1, wit1.w_907_0, wit1.w_908_0, wit1.w_910_0, wit1.w_911_0, wit1.w_912_0, wit1.w_914_0, wit1.w_915_0, wit1.w_916_0, wit1.w_917_0, wit1.w_918_0, wit1.w_919_0, wit1.w_920_0, wit1.w_920_concat_1, wit1.w_921_0, wit1.w_923_0, wit1.w_924_0, wit1.w_925_0, wit1.w_927_0, wit1.w_928_0, wit1.w_929_0, wit1.w_930_0, wit1.w_931_0, wit1.w_932_0, wit1.w_933_0, wit1.w_933_concat_1, wit1.w_934_0, wit1.w_935_0, wit1.w_937_0, wit1.w_938_0, wit1.w_939_0, wit1.w_941_0, wit1.w_942_0, wit1.w_943_0, wit1.w_944_0, wit1.w_945_0, wit1.w_946_0, wit1.w_947_0, wit1.w_947_concat_1, wit1.w_948_0, wit1.w_950_0, wit1.w_951_0, wit1.w_952_0, wit1.w_954_0, wit1.w_955_0, wit1.w_956_0, wit1.w_957_0, wit1.w_958_0, wit1.w_959_0, wit1.w_960_0, wit1.w_960_concat_1, wit1.w_961_0, wit1.w_962_0, wit1.w_964_0, wit1.w_965_0, wit1.w_966_0, wit1.w_968_0, wit1.w_969_0, wit1.w_970_0, wit1.w_971_0, wit1.w_972_0, wit1.w_973_0, wit1.w_974_0, wit1.w_974_concat_1, wit1.w_975_0, wit1.w_977_0, wit1.w_978_0, wit1.w_979_0, wit1.w_981_0, wit1.w_982_0, wit1.w_983_0, wit1.w_984_0, wit1.w_985_0, wit1.w_986_0, wit1.w_987_0, wit1.w_987_concat_1, wit1.w_988_0, wit1.w_989_0, wit1.w_991_0, wit1.w_992_0, wit1.w_993_0, wit1.w_995_0, wit1.w_996_0, wit1.w_997_0, wit1.w_998_0, wit1.w_999_0, wit1.w_1000_0, wit1.w_1001_0, wit1.w_1001_concat_1, wit1.w_1002_0, wit1.w_1004_0, wit1.w_1005_0, wit1.w_1006_0, wit1.w_1008_0, wit1.w_1009_0, wit1.w_1010_0, wit1.w_1011_0, wit1.w_1012_0, wit1.w_1013_0, wit1.w_1014_0, wit1.w_1014_concat_1, wit1.w_1015_0, wit1.w_1016_0, wit1.w_1018_0, wit1.w_1019_0, wit1.w_1020_0, wit1.w_1022_0, wit1.w_1023_0, wit1.w_1024_0, wit1.w_1025_0, wit1.w_1026_0, wit1.w_1027_0, wit1.w_1028_0, wit1.w_1028_concat_1, wit1.w_1029_0, wit1.w_1031_0, wit1.w_1032_0, wit1.w_1033_0, wit1.w_1035_0, wit1.w_1036_0, wit1.w_1037_0, wit1.w_1038_0, wit1.w_1039_0, wit1.w_1040_0, wit1.w_1041_0, wit1.w_1041_concat_1, wit1.w_1042_0, wit1.w_1043_0, wit1.w_1045_0, wit1.w_1046_0, wit1.w_1047_0, wit1.w_1049_0, wit1.w_1050_0, wit1.w_1051_0, wit1.w_1052_0, wit1.w_1053_0, wit1.w_1054_0, wit1.w_1055_0, wit1.w_1055_concat_1, wit1.w_1056_0, wit1.w_1058_0, wit1.w_1059_0, wit1.w_1060_0, wit1.w_1062_0, wit1.w_1063_0, wit1.w_1064_0, wit1.w_1065_0, wit1.w_1066_0, wit1.w_1067_0, wit1.w_1068_0, wit1.w_1068_concat_1, wit1.w_1069_0, wit1.w_1070_0, wit1.w_1072_0, wit1.w_1073_0, wit1.w_1074_0, wit1.w_1076_0, wit1.w_1077_0, wit1.w_1078_0, wit1.w_1079_0, wit1.w_1080_0, wit1.w_1081_0, wit1.w_1082_0, wit1.w_1082_concat_1, wit1.w_1083_0, wit1.w_1085_0, wit1.w_1086_0, wit1.w_1087_0, wit1.w_1089_0, wit1.w_1090_0, wit1.w_1091_0, wit1.w_1092_0, wit1.w_1093_0, wit1.w_1094_0, wit1.w_1095_0, wit1.w_1095_concat_1, wit1.w_1096_0, wit1.w_1097_0, wit1.w_1099_0, wit1.w_1100_0, wit1.w_1101_0, wit1.w_1103_0, wit1.w_1104_0, wit1.w_1105_0, wit1.w_1106_0, wit1.w_1107_0, wit1.w_1108_0, wit1.w_1109_0, wit1.w_1111_0, wit1.w_1112_0, wit1.w_1113_0, wit1.w_1115_0, wit1.w_1116_0, wit1.w_1117_0, wit1.w_1118_0, wit1.w_1119_0, wit1.w_1120_0, wit1.w_1121_0, wit1.w_1121_1, wit1.w_1121_2, wit1.w_1121_3, wit1.w_1121_4, wit1.w_1121_5, wit1.w_1121_6, wit1.w_1121_7, wit1.w_1121_8, wit1.w_1121_9, wit1.w_1122_0, wit1.w_1124_0, wit1.w_1125_0, wit1.w_1126_0, wit1.w_1126_concat_1, wit1.w_1127_0, wit1.w_1128_0, wit1.w_1129_0, wit1.w_1130_0, wit1.w_1130_concat_1, wit1.w_1131_0, wit1.w_1132_0, wit1.w_1133_0, wit1.w_1134_0, wit1.w_1135_0, wit1.w_1135_concat_1, wit1.w_1136_0, wit1.w_1137_0, wit1.w_1138_0, wit1.w_1139_0, wit1.w_1139_concat_1, wit1.w_1140_0, wit1.w_1141_0, wit1.w_1142_0, wit1.w_1143_0, wit1.w_1145_0, wit1.w_1146_0, wit1.w_1147_0, wit1.w_1148_0, wit1.w_1149_0, wit1.w_1150_0, wit1.w_1151_0, wit1.w_1153_0, wit1.w_1154_0, wit1.w_1157_0, wit1.w_1158_0, wit1.w_1160_0, wit1.w_1161_0, wit1.w_1163_0, wit1.w_1164_0, wit1.w_1165_0, wit1.w_1166_0, wit1.w_1168_0, wit1.w_1169_0, wit1.w_1170_0, wit1.w_1171_0, wit1.w_1172_0, wit1.w_1173_0, wit1.w_1174_0, wit1.w_1176_0, wit1.w_1177_0, wit1.w_1180_0, wit1.w_1181_0, wit1.w_1183_0, wit1.w_1184_0, wit1.w_1186_0, wit1.w_1187_0, wit1.w_1187_1, wit1.w_1187_2, wit1.w_1187_3, wit1.w_1187_4, wit1.w_1187_5, wit1.w_1187_6, wit1.w_1187_7, wit1.w_1187_8, wit1.w_1187_9, wit1.w_1188_0, wit1.w_1190_0, wit1.w_1191_0, wit1.w_1192_0, wit1.w_1192_concat_1, wit1.w_1193_0, wit1.w_1194_0, wit1.w_1195_0, wit1.w_1196_0, wit1.w_1196_concat_1, wit1.w_1197_0, wit1.w_1198_0, wit1.w_1199_0, wit1.w_1200_0, wit1.w_1201_0, wit1.w_1201_concat_1, wit1.w_1202_0, wit1.w_1203_0, wit1.w_1204_0, wit1.w_1205_0, wit1.w_1205_concat_1, wit1.w_1206_0, wit1.w_1207_0, wit1.w_1208_0, wit1.w_1209_0, wit1.w_1211_0, wit1.w_1212_0, wit1.w_1213_0, wit1.w_1214_0, wit1.w_1215_0, wit1.w_1216_0, wit1.w_1217_0, wit1.w_1219_0, wit1.w_1221_0, wit1.w_1222_0, wit1.w_1223_0, wit1.w_1225_0, wit1.w_1226_0, wit1.w_1227_0, wit1.w_1228_0, wit1.w_1229_0, wit1.w_1230_0, wit1.w_1231_0, wit1.w_1233_0, wit1.w_1235_0, wit1.w_1237_0, wit1.w_1239_0, wit1.w_1240_0, wit1.w_1243_0, wit1.w_1244_0, wit1.w_1247_0, wit1.w_1247_1, wit1.w_1247_2, wit1.w_1247_3, wit1.w_1247_4, wit1.w_1247_5, wit1.w_1248_0, wit1.w_1248_1, wit1.w_1248_2, wit1.w_1248_3, wit1.w_1248_4, wit1.w_1248_5, wit1.w_1249_0, wit1.w_1250_0, wit1.w_1252_0, wit1.w_1253_0, wit1.w_1255_0, wit1.w_1256_0, wit1.w_1258_0, wit1.w_1259_0, wit1.w_1261_0, wit1.w_1262_0, wit1.w_1264_0, wit1.w_1265_0, wit1.w_1267_0, wit1.w_1267_1, wit1.w_1267_2, wit1.w_1267_3, wit1.w_1267_4, wit1.w_1267_5, wit1.w_1268_0, wit1.w_1269_0, wit1.w_1270_0, wit1.w_1271_0, wit1.w_1272_0, wit1.w_1273_0⟩, (by obtain ⟨h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32, h33, h34, h35, h36, h37, h38, h39, h40, h41, h42, h43, h44, h45, h46, h47, h48, h49, h50, h51, h52, h53, h54, h55, h56, h57, h58, h59, h60, h61, h62, h63, h64, h65, h66, h67, h68, h69, h70, h71, h72, h73, h74, h75, h76, h77, h78, h79, h80, h81, h82, h83, h84, h85, h86, h87, h88, h89, h90⟩ := h2; exact ⟨(Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_0.bridge h3), (Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_1.bridge h4), (Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_2.bridge h5), (Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_3.bridge h6), (Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_4.bridge h7), h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32, h33, h34, h35, h36, h37, h38, h39, h40, h41, h42, h43, h44, h45, h46, h47, h48, h49, h50, h51, h52, h53, h54, h55, h56, h57, h58, h59, h60, h61, h62, h63, h64, h65, h66, h67, h68, h69, h70, h71, h72, h73, h74, h75, h76, h77, h78, h79, h80, (Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_78.bridge h81), h82, h83, h84, (Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_82.bridge h85), h86, h87, h88, (Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_86.bridge h89), (Stage_sigma_online_eval.parallel_generatedRoot_188.constraints_87.bridge h90)⟩)⟩)


theorem Stage_sigma_online_eval.generatedRoot.constraints_0.bridge {backend hashModel tape path params params' w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_137_0 w_138_0 w_139_0 w_140_0 w_141_0 w_142_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_166_0 w_167_0 w_168_0 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_188_1 w_188_2 w_188_3 w_188_4 w_188_5 outputs}
    (h : Stage_sigma_online_eval.generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_137_0 w_138_0 w_139_0 w_140_0 w_141_0 w_142_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_166_0 w_167_0 w_168_0 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_188_1 w_188_2 w_188_3 w_188_4 w_188_5 outputs) : W_Stage_sigma_online_eval.generatedRoot.constraints_0 backend hashModel params' w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_137_0 w_138_0 w_139_0 w_140_0 w_141_0 w_142_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_166_0 w_167_0 w_168_0 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_188_1 w_188_2 w_188_3 w_188_4 w_188_5 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_online_eval.parallel_generatedRoot_188.bridge (h2 i))), h3⟩)


theorem Stage_sigma_online_eval.generatedRoot.bridge {backend hashModel tape path params params' inputs outputs}
    (h : Stage_sigma_online_eval.generatedRoot backend hashModel tape path params inputs outputs) : W_Stage_sigma_online_eval.generatedRoot backend hashModel params' inputs outputs := by
  delta W_Stage_sigma_online_eval.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_188_0, wit1.w_188_1, wit1.w_188_2, wit1.w_188_3, wit1.w_188_4, wit1.w_188_5⟩, (Stage_sigma_online_eval.generatedRoot.constraints_0.bridge h2)⟩)
