import Stage_online_0_Part_0
import W_Stage_online_0
import ExponentSampling

/-! Every run of the average-case relations of `Stage_online_0_Part_0` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Stage_online_0_Part_0.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Stage_online_0.parallel_generatedRoot_61.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_61 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_61 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_61
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_62_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_62_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_62_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_62_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_62_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_62_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_62_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_62_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_62.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_62.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_62.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_62_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_62_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_62.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_62 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_62 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_62
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_62.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_65.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_65 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_65 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_65
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_66_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_66_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_66_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_66_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_66_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_66_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_66_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_66_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_66.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_66.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_66.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_66_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_66_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_66.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_66 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_66 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_66
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_66.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_69.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_69 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_69 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_69
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_70_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_70_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_70_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_70_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_70_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_70_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_70_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_70_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_70.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_70.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_70.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_70_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_70_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_70.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_70 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_70 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_70
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_70.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_73.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_73 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_73 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_73
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_74_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_74_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_74_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_74_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_74_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_74_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_74_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_74_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_74.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_74.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_74.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_74_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_74_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_74.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_74 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_74 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_74
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_74.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_77.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_77 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_77 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_77
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_78_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_78_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_78_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_78_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_78_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_78_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_78_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_78_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_78.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_78.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_78.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_78_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_78_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_78.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_78 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_78 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_78
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_78.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_81.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_81 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_81 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_81
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_82_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_82_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_82_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_82_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_82_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_82_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_82_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_82_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_82.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_82.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_82.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_82_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_82_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_82.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_82 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_82 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_82
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_82.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_85.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_85 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_85 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_85
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_86_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_86_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_86_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_86_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_86_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_86_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_86_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_86_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_86.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_86.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_86.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_86_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_86_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_86.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_86 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_86 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_86
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_86.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_89.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_89 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_89 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_89
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_90_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_90_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_90_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_90_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_90_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_90_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_90_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_90_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_90.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_90.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_90.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_90_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_90_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_90.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_90 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_90 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_90
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_90.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_93.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_93 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_93 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_93
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_94_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_94_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_94_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_94_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_94_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_94_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_94_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_94_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_94.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_94.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_94.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_94_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_94_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_94.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_94 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_94 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_94
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_94.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_97.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_97 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_97 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_97
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_98_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_98_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_98_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_98_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_98_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_98_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_98_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_98_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_98.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_98.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_98.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_98_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_98_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_98.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_98 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_98 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_98
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_98.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_101.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_101 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_101 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_101
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_102_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_102_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_102_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_102_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_102_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_102_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_102_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_102_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_102.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_102.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_102.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_102_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_102_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_102.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_102 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_102 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_102
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_102.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_105.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_105 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_105 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_105
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_106_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_106_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_106_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_106_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_106_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_106_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_106_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_106_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_106.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_106.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_106.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_106_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_106_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_106.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_106 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_106 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_106
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_106.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_109.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_109 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_109 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_109
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_110_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_110_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_110_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_110_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_110_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_110_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_110_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_110_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_110.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_110.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_110.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_110_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_110_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_110.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_110 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_110 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_110
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_110.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_113.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_113 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_113 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_113
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_114_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_114_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_114_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_114_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_114_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_114_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_114_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_114_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_114.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_114.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_114.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_114_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_114_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_114.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_114 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_114 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_114
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_114.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_117.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_117 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_117 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_117
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_118_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_118_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_118_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_118_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_118_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_118_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_118_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_118_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_118.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_118.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_118.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_118_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_118_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_118.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_118 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_118 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_118
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_118.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_121.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_121 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_121 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_121
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_122_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_122_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_122_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_122_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_122_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_122_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_122_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_122_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_122.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_122.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_122.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_122_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_122_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_122.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_122 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_122 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_122
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_122.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_125.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_125 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_125 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_125
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_126_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_126_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_126_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_126_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_126_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_126_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_126_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_126_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_126.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_126.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_126.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_126_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_126_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_126.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_126 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_126 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_126
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_126.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_129.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_129 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_129 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_129
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_130_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_130_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_130_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_130_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_130_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_130_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_130_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_130_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_130.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_130.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_130.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_130_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_130_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_130.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_130 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_130 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_130
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_130.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_133.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_133 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_133 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_133
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_134_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_134_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_134_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_134_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_134_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_134_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_134_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_134_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_134.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_134.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_134.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_134_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_134_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_134.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_134 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_134 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_134
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_134.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_137.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_137 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_137 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_137
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_138_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_138_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_138_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_138_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_138_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_138_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_138_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_138_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_138.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_138.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_138.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_138_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_138_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_138.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_138 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_138 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_138
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_138.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_141.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_141 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_141 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_141
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_142_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_142_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_142_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_142_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_142_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_142_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_142_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_142_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_142.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_142.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_142.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_142_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_142_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_142.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_142 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_142 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_142
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_142.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_145.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_145 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_145 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_145
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_146_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_146_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_146_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_146_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_146_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_146_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_146_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_146_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_146.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_146.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_146.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_146_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_146_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_146.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_146 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_146 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_146
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_146.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_149.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_149 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_149 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_149
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_150_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_150_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_150_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_150_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_150_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_150_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_150_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_150_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_150.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_150.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_150.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_150_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_150_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_150.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_150 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_150 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_150
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_150.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_153.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_153 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_153 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_153
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_154_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_154_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_154_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_154_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_154_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_154_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_154_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_154_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_154.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_154.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_154.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_154_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_154_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_154.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_154 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_154 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_154
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_154.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_157.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_157 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_157 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_157
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_158_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_158_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_158_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_158_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_158_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_158_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_158_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_158_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_158.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_158.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_158.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_158_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_158_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_158.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_158 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_158 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_158
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_158.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_161.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_161 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_161 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_161
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_162_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_162_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_162_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_162_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_162_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_162_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_162_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_162_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_162.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_162.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_162.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_162_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_162_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_162.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_162 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_162 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_162
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_162.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_165.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_165 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_165 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_165
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_166_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_166_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_166_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_166_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_166_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_166_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_166_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_166_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_166.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_166.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_166.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_166_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_166_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_166.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_166 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_166 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_166
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_166.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_169.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_169 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_169 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_169
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_170_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_170_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_170_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_170_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_170_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_170_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_170_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_170_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_170.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_170.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_170.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_170_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_170_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_170.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_170 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_170 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_170
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_170.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_173.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_173 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_173 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_173
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_174_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_174_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_174_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_174_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_174_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_174_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_174_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_174_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_174.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_174.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_174.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_174_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_174_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_174.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_174 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_174 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_174
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_174.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_177.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_177 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_177 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_177
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_178_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_178_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_178_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_178_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_178_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_178_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_178_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_178_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_178.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_178.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_178.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_178_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_178_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_178.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_178 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_178 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_178
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_178.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_181.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_181 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_181 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_181
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_182_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_182_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_182_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_182_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_182_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_182_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_182_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_182_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_182.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_182.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_182.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_182_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_182_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_182.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_182 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_182 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_182
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_182.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_185.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_185 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_185 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_185
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_186_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_186_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_186_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_186_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_186_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_186_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_186_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_186_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_186.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_186.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_186.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_186_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_186_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_186.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_186 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_186 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_186
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_186.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_189.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_189 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_189 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_189
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_190_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_190_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_190_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_190_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_190_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_190_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_190_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_190_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_190.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_190.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_190.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_190_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_190_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_190.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_190 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_190 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_190
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_190.constraints_0.bridge h2)⟩)


theorem Stage_online_0.parallel_generatedRoot_193.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_193 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_193 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_193
  exact h


theorem Stage_online_0.parallel_parallel_generatedRoot_194_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_0.parallel_parallel_generatedRoot_194_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_0.parallel_parallel_generatedRoot_194_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_0.parallel_parallel_generatedRoot_194_6
  exact h


theorem Stage_online_0.sequential_parallel_generatedRoot_194_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_194_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_194_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_194_25
  exact h


theorem Stage_online_0.parallel_generatedRoot_194.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_0.parallel_generatedRoot_194.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_0.parallel_generatedRoot_194.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_0.parallel_parallel_generatedRoot_194_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_0_Part_0.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_0.sequential_parallel_generatedRoot_194_25.bridge hs18))), h17⟩)


theorem Stage_online_0.parallel_generatedRoot_194.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_194 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_194 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_194
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_0.parallel_generatedRoot_194.constraints_0.bridge h2)⟩)


theorem Stage_online_0.sequential_parallel_generatedRoot_195_18.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_195_18 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_195_18 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_195_18
  exact h


theorem Stage_online_0.parallel_generatedRoot_195.constraints_0.bridge {backend hashModel tape path params params' i_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1}
    (h : Stage_online_0.parallel_generatedRoot_195.constraints_0 backend hashModel tape path params i_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1) : W_Stage_online_0.parallel_generatedRoot_195.constraints_0 backend hashModel params' i_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, (Stage_online_0_Part_0.iterRuns_mono h6 (fun _ _ _ hs33 ↦ (Stage_online_0.sequential_parallel_generatedRoot_195_18.bridge hs33))), h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_online_0.parallel_generatedRoot_195.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_195 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_195 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_195
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_2_0, wit1.w_4_0, wit1.w_6_0, wit1.w_8_0, wit1.w_18_0, wit1.w_18_1, wit1.w_18_2, wit1.w_18_3, wit1.w_18_4, wit1.w_18_5, wit1.w_18_6, wit1.w_18_7, wit1.w_18_8, wit1.w_18_9, wit1.w_19_0, wit1.w_21_0, wit1.w_22_0, wit1.w_23_0, wit1.w_23_concat_1, wit1.w_24_0, wit1.w_25_0, wit1.w_26_0, wit1.w_27_0, wit1.w_27_concat_1, wit1.w_28_0, wit1.w_29_0, wit1.w_30_0, wit1.w_31_0, wit1.w_32_0, wit1.w_32_concat_1, wit1.w_33_0, wit1.w_34_0, wit1.w_35_0, wit1.w_36_0, wit1.w_36_concat_1, wit1.w_37_0, wit1.w_38_0, wit1.w_39_0, wit1.w_40_0, wit1.w_42_0, wit1.w_43_0, wit1.w_44_0, wit1.w_45_0, wit1.w_46_0, wit1.w_48_0, wit1.w_50_0, wit1.w_51_0, wit1.w_52_0, wit1.w_54_0, wit1.w_55_0, wit1.w_56_0, wit1.w_57_0, wit1.w_58_0, wit1.w_60_0, wit1.w_62_0, wit1.w_64_0, wit1.w_66_0, wit1.w_67_0, wit1.w_70_0, wit1.w_71_0⟩, (by obtain ⟨h3, h4, h5, h6, h7⟩ := h2; exact ⟨(Stage_online_0.parallel_generatedRoot_195.constraints_0.bridge h3), h4, h5, h6, h7⟩)⟩)


theorem Stage_online_0.sequential_parallel_generatedRoot_196_18.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_0.sequential_parallel_generatedRoot_196_18 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_online_0.sequential_parallel_generatedRoot_196_18 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_online_0.sequential_parallel_generatedRoot_196_18
  exact h


theorem Stage_online_0.parallel_generatedRoot_196.constraints_0.bridge {backend hashModel tape path params params' i_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1}
    (h : Stage_online_0.parallel_generatedRoot_196.constraints_0 backend hashModel tape path params i_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1) : W_Stage_online_0.parallel_generatedRoot_196.constraints_0 backend hashModel params' i_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, (Stage_online_0_Part_0.iterRuns_mono h6 (fun _ _ _ hs33 ↦ (Stage_online_0.sequential_parallel_generatedRoot_196_18.bridge hs33))), h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_online_0.parallel_generatedRoot_196.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_196 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_196 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_196
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_2_0, wit1.w_4_0, wit1.w_6_0, wit1.w_8_0, wit1.w_18_0, wit1.w_18_1, wit1.w_18_2, wit1.w_18_3, wit1.w_18_4, wit1.w_18_5, wit1.w_18_6, wit1.w_18_7, wit1.w_18_8, wit1.w_18_9, wit1.w_19_0, wit1.w_21_0, wit1.w_22_0, wit1.w_23_0, wit1.w_23_concat_1, wit1.w_24_0, wit1.w_25_0, wit1.w_26_0, wit1.w_27_0, wit1.w_27_concat_1, wit1.w_28_0, wit1.w_29_0, wit1.w_30_0, wit1.w_31_0, wit1.w_32_0, wit1.w_32_concat_1, wit1.w_33_0, wit1.w_34_0, wit1.w_35_0, wit1.w_36_0, wit1.w_36_concat_1, wit1.w_37_0, wit1.w_38_0, wit1.w_39_0, wit1.w_40_0, wit1.w_42_0, wit1.w_43_0, wit1.w_44_0, wit1.w_45_0, wit1.w_46_0, wit1.w_48_0, wit1.w_50_0, wit1.w_51_0, wit1.w_52_0, wit1.w_54_0, wit1.w_55_0, wit1.w_56_0, wit1.w_57_0, wit1.w_58_0, wit1.w_60_0, wit1.w_62_0, wit1.w_64_0, wit1.w_66_0, wit1.w_67_0, wit1.w_70_0, wit1.w_71_0⟩, (by obtain ⟨h3, h4, h5, h6, h7⟩ := h2; exact ⟨(Stage_online_0.parallel_generatedRoot_196.constraints_0.bridge h3), h4, h5, h6, h7⟩)⟩)


theorem Stage_online_0.parallel_generatedRoot_197.bridge {backend x1 tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.parallel_generatedRoot_197 backend x1 tape path params i_0 inputs outputs) : W_Stage_online_0.parallel_generatedRoot_197 backend x1 params' i_0 inputs outputs := by
  delta W_Stage_online_0.parallel_generatedRoot_197
  exact h


theorem Stage_online_0.sequential_generatedRoot_216.bridge {backend x1 tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_216 backend x1 tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_216 backend x1 params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_216
  exact h


theorem Stage_online_0.sequential_generatedRoot_320.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_320 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_320 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_320
  exact h


theorem Stage_online_0.sequential_generatedRoot_399.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_399 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_399 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_399
  exact h


theorem Stage_online_0.sequential_generatedRoot_457.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_457 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_457 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_457
  exact h


theorem Stage_online_0.sequential_generatedRoot_515.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_515 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_515 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_515
  exact h


theorem Stage_online_0.sequential_generatedRoot_573.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_573 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_573 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_573
  exact h


theorem Stage_online_0.sequential_generatedRoot_631.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_631 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_631 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_631
  exact h


theorem Stage_online_0.sequential_generatedRoot_689.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_689 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_689 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_689
  exact h


theorem Stage_online_0.sequential_generatedRoot_747.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_747 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_747 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_747
  exact h


theorem Stage_online_0.sequential_generatedRoot_805.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_805 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_805 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_805
  exact h


theorem Stage_online_0.sequential_generatedRoot_863.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_863 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_863 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_863
  exact h


theorem Stage_online_0.sequential_generatedRoot_921.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_921 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_921 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_921
  exact h


theorem Stage_online_0.sequential_generatedRoot_979.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_979 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_979 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_979
  exact h


theorem Stage_online_0.sequential_generatedRoot_1037.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_1037 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_1037 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_1037
  exact h


theorem Stage_online_0.sequential_generatedRoot_1095.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_1095 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_1095 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_1095
  exact h


theorem Stage_online_0.sequential_generatedRoot_1153.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_1153 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_1153 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_1153
  exact h


theorem Stage_online_0.sequential_generatedRoot_1211.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_1211 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_1211 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_1211
  exact h


theorem Stage_online_0.sequential_generatedRoot_1269.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_1269 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_1269 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_1269
  exact h


theorem Stage_online_0.sequential_generatedRoot_1327.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_1327 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_1327 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_1327
  exact h


theorem Stage_online_0.sequential_generatedRoot_1385.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_1385 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_1385 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_1385
  exact h


theorem Stage_online_0.sequential_generatedRoot_1443.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_1443 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_1443 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_1443
  exact h


theorem Stage_online_0.sequential_generatedRoot_1501.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_1501 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_1501 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_1501
  exact h


theorem Stage_online_0.sequential_generatedRoot_1559.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_1559 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_1559 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_1559
  exact h


theorem Stage_online_0.sequential_generatedRoot_1617.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_1617 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_1617 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_1617
  exact h


theorem Stage_online_0.sequential_generatedRoot_1675.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_1675 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_1675 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_1675
  exact h


theorem Stage_online_0.sequential_generatedRoot_1733.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_1733 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_1733 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_1733
  exact h


theorem Stage_online_0.sequential_generatedRoot_1791.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_1791 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_1791 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_1791
  exact h


theorem Stage_online_0.sequential_generatedRoot_1849.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_1849 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_1849 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_1849
  exact h


theorem Stage_online_0.sequential_generatedRoot_1907.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_1907 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_1907 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_1907
  exact h


theorem Stage_online_0.sequential_generatedRoot_1965.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_1965 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_1965 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_1965
  exact h


theorem Stage_online_0.sequential_generatedRoot_2023.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_2023 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_2023 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_2023
  exact h


theorem Stage_online_0.sequential_generatedRoot_2081.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_2081 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_2081 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_2081
  exact h


theorem Stage_online_0.sequential_generatedRoot_2139.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_2139 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_2139 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_2139
  exact h


theorem Stage_online_0.sequential_generatedRoot_2197.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_2197 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_2197 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_2197
  exact h


theorem Stage_online_0.sequential_generatedRoot_2255.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_2255 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_2255 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_2255
  exact h


theorem Stage_online_0.sequential_generatedRoot_2313.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_2313 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_2313 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_2313
  exact h


theorem Stage_online_0.sequential_generatedRoot_2371.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_2371 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_2371 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_2371
  exact h


theorem Stage_online_0.sequential_generatedRoot_2429.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_2429 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_2429 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_2429
  exact h


theorem Stage_online_0.sequential_generatedRoot_2487.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_2487 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_2487 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_2487
  exact h


theorem Stage_online_0.sequential_generatedRoot_2545.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_2545 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_2545 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_2545
  exact h


theorem Stage_online_0.sequential_generatedRoot_2603.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_2603 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_2603 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_2603
  exact h


theorem Stage_online_0.sequential_generatedRoot_2661.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_2661 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_2661 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_2661
  exact h


theorem Stage_online_0.sequential_generatedRoot_2719.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_2719 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_2719 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_2719
  exact h


theorem Stage_online_0.sequential_generatedRoot_2777.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_2777 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_2777 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_2777
  exact h


theorem Stage_online_0.sequential_generatedRoot_2835.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_2835 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_2835 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_2835
  exact h


theorem Stage_online_0.sequential_generatedRoot_2893.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_2893 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_2893 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_2893
  exact h


theorem Stage_online_0.sequential_generatedRoot_2951.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_2951 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_2951 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_2951
  exact h


theorem Stage_online_0.sequential_generatedRoot_3009.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_3009 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_3009 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_3009
  exact h


theorem Stage_online_0.sequential_generatedRoot_3067.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_3067 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_3067 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_3067
  exact h


theorem Stage_online_0.sequential_generatedRoot_3125.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_3125 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_3125 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_3125
  exact h


theorem Stage_online_0.sequential_generatedRoot_3183.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_3183 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_3183 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_3183
  exact h


theorem Stage_online_0.sequential_generatedRoot_3241.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_3241 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_3241 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_3241
  exact h


theorem Stage_online_0.sequential_generatedRoot_3299.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_3299 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_3299 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_3299
  exact h


theorem Stage_online_0.sequential_generatedRoot_3357.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_3357 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_3357 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_3357
  exact h


theorem Stage_online_0.sequential_generatedRoot_3415.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_3415 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_3415 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_3415
  exact h


theorem Stage_online_0.sequential_generatedRoot_3473.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_3473 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_3473 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_3473
  exact h


theorem Stage_online_0.sequential_generatedRoot_3531.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_3531 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_3531 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_3531
  exact h


theorem Stage_online_0.sequential_generatedRoot_3589.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_3589 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_3589 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_3589
  exact h


theorem Stage_online_0.sequential_generatedRoot_3647.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_3647 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_3647 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_3647
  exact h


theorem Stage_online_0.sequential_generatedRoot_3705.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_3705 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_3705 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_3705
  exact h


theorem Stage_online_0.sequential_generatedRoot_3763.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_3763 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_3763 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_3763
  exact h


theorem Stage_online_0.sequential_generatedRoot_3821.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_3821 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_3821 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_3821
  exact h


theorem Stage_online_0.sequential_generatedRoot_3879.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_3879 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_3879 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_3879
  exact h


theorem Stage_online_0.sequential_generatedRoot_3937.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_3937 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_3937 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_3937
  exact h


theorem Stage_online_0.sequential_generatedRoot_3995.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_3995 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_3995 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_3995
  exact h


theorem Stage_online_0.sequential_generatedRoot_4053.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_4053 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_4053 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_4053
  exact h


theorem Stage_online_0.sequential_generatedRoot_4111.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_4111 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_4111 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_4111
  exact h


theorem Stage_online_0.sequential_generatedRoot_4169.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_4169 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_4169 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_4169
  exact h


theorem Stage_online_0.sequential_generatedRoot_4227.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_4227 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_4227 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_4227
  exact h


theorem Stage_online_0.sequential_generatedRoot_4285.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_4285 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_4285 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_4285
  exact h


theorem Stage_online_0.sequential_generatedRoot_4343.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_4343 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_4343 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_4343
  exact h


theorem Stage_online_0.sequential_generatedRoot_4401.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_4401 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_4401 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_4401
  exact h


theorem Stage_online_0.sequential_generatedRoot_4459.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_4459 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_4459 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_4459
  exact h


theorem Stage_online_0.sequential_generatedRoot_4517.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_4517 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_4517 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_4517
  exact h


theorem Stage_online_0.sequential_generatedRoot_4575.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_4575 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_4575 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_4575
  exact h


theorem Stage_online_0.sequential_generatedRoot_4633.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_4633 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_4633 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_4633
  exact h


theorem Stage_online_0.sequential_generatedRoot_4691.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_4691 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_4691 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_4691
  exact h


theorem Stage_online_0.sequential_generatedRoot_4749.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_4749 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_4749 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_4749
  exact h


theorem Stage_online_0.sequential_generatedRoot_4807.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_4807 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_4807 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_4807
  exact h


theorem Stage_online_0.sequential_generatedRoot_4865.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_4865 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_4865 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_4865
  exact h


theorem Stage_online_0.sequential_generatedRoot_4923.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_4923 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_4923 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_4923
  exact h


theorem Stage_online_0.sequential_generatedRoot_4981.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_4981 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_4981 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_4981
  exact h


theorem Stage_online_0.sequential_generatedRoot_5039.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_5039 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_5039 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_5039
  exact h


theorem Stage_online_0.sequential_generatedRoot_5097.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_5097 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_5097 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_5097
  exact h


theorem Stage_online_0.sequential_generatedRoot_5155.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_5155 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_5155 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_5155
  exact h


theorem Stage_online_0.sequential_generatedRoot_5213.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_5213 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_5213 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_5213
  exact h


theorem Stage_online_0.sequential_generatedRoot_5271.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_5271 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_5271 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_5271
  exact h


theorem Stage_online_0.sequential_generatedRoot_5329.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_5329 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_5329 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_5329
  exact h


theorem Stage_online_0.sequential_generatedRoot_5387.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_5387 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_5387 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_5387
  exact h


theorem Stage_online_0.sequential_generatedRoot_5445.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_5445 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_5445 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_5445
  exact h


theorem Stage_online_0.sequential_generatedRoot_5503.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_5503 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_5503 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_5503
  exact h


theorem Stage_online_0.sequential_generatedRoot_5561.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_5561 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_5561 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_5561
  exact h


theorem Stage_online_0.sequential_generatedRoot_5619.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_5619 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_5619 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_5619
  exact h


theorem Stage_online_0.sequential_generatedRoot_5677.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_5677 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_5677 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_5677
  exact h


theorem Stage_online_0.sequential_generatedRoot_5735.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_5735 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_5735 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_5735
  exact h


theorem Stage_online_0.sequential_generatedRoot_5793.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_5793 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_5793 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_5793
  exact h


theorem Stage_online_0.sequential_generatedRoot_5851.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_5851 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_5851 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_5851
  exact h


theorem Stage_online_0.sequential_generatedRoot_5909.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_5909 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_5909 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_5909
  exact h


theorem Stage_online_0.sequential_generatedRoot_5967.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_5967 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_5967 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_5967
  exact h


theorem Stage_online_0.sequential_generatedRoot_6025.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_6025 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_6025 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_6025
  exact h


theorem Stage_online_0.sequential_generatedRoot_6083.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_6083 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_6083 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_6083
  exact h


theorem Stage_online_0.sequential_generatedRoot_6141.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_6141 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_6141 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_6141
  exact h


theorem Stage_online_0.sequential_generatedRoot_6199.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_6199 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_6199 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_6199
  exact h


theorem Stage_online_0.sequential_generatedRoot_6257.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_6257 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_6257 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_6257
  exact h


theorem Stage_online_0.sequential_generatedRoot_6315.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_6315 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_6315 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_6315
  exact h


theorem Stage_online_0.sequential_generatedRoot_6373.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_6373 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_6373 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_6373
  exact h


theorem Stage_online_0.sequential_generatedRoot_6431.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_6431 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_6431 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_6431
  exact h


theorem Stage_online_0.sequential_generatedRoot_6489.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_6489 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_6489 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_6489
  exact h


theorem Stage_online_0.sequential_generatedRoot_6547.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_6547 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_6547 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_6547
  exact h


theorem Stage_online_0.sequential_generatedRoot_6605.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_6605 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_6605 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_6605
  exact h


theorem Stage_online_0.sequential_generatedRoot_6663.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_6663 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_6663 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_6663
  exact h


theorem Stage_online_0.sequential_generatedRoot_6721.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_6721 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_6721 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_6721
  exact h


theorem Stage_online_0.sequential_generatedRoot_6779.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_6779 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_6779 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_6779
  exact h


theorem Stage_online_0.sequential_generatedRoot_6837.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_6837 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_6837 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_6837
  exact h


theorem Stage_online_0.sequential_generatedRoot_6895.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_6895 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_6895 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_6895
  exact h


theorem Stage_online_0.sequential_generatedRoot_6953.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_6953 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_6953 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_6953
  exact h


theorem Stage_online_0.sequential_generatedRoot_7011.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_7011 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_7011 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_7011
  exact h


theorem Stage_online_0.sequential_generatedRoot_7069.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_0.sequential_generatedRoot_7069 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_0.sequential_generatedRoot_7069 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_0.sequential_generatedRoot_7069
  exact h
