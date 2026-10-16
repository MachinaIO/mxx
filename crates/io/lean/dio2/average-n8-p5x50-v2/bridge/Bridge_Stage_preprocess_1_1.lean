import Stage_preprocess_1_1
import W_Stage_preprocess_1_1
import ExponentSampling

/-! Every run of the average-case relations of `Stage_preprocess_1_1` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Stage_preprocess_1_1.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_3_12.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_3_12 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_3_12 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_3_12
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_3_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_3_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_3_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_3_13
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_3.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_18_0 w_22_0 w_31_0 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_3.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_18_0 w_22_0 w_31_0 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_3.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_18_0 w_22_0 w_31_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_3_12.bridge (h11 i))), h12, (Stage_preprocess_1_1.iterRuns_mono h13 (fun _ _ _ hs17 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_3_13.bridge hs17))), h14, h15, h16⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_3.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_3 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_3 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_3
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_12_0, wit1.w_13_0, wit1.w_17_0, wit1.w_18_0⟩, (Stage_preprocess_1_1.parallel_generatedRoot_3.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_4_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_4_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_4_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_4_10
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_4_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_4_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_4_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_4_13
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_4.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_4.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_4.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_4_10.bridge (h11 i))), h12, (Stage_preprocess_1_1.iterRuns_mono h13 (fun _ _ _ hs16 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_4_13.bridge hs16))), h14, h15⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_4.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_4 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_4 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_4
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0⟩, (Stage_preprocess_1_1.parallel_generatedRoot_4.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_33.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_33 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_33 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_33
  exact h


theorem Stage_preprocess_1_1.sequential_generatedRoot_66.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_generatedRoot_66 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.sequential_generatedRoot_66 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_generatedRoot_66
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_75.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_75 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_75 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_75
  exact h


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_76_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_76_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_76_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_76_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_76_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_76_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_76_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_76_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_76.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_76.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_76.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_76_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_76_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_76.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_76 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_76 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_76
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_76.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_79.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_79 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_79 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_79
  exact h


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_80_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_80_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_80_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_80_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_80_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_80_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_80_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_80_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_80.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_80.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_80.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_80_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_80_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_80.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_80 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_80 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_80
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_80.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_81_10.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_81_10 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_81_10 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_81_10
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_81.constraints_0.bridge {backend hashModel tape path params params' i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_81.constraints_0 backend hashModel tape path params i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_81.constraints_0 backend hashModel params' i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (Stage_preprocess_1_1.iterRuns_mono h2 (fun _ _ _ hs4 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_81_10.bridge hs4))), h3⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_81.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_81 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_81 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_81
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_10_0, wit1.w_10_1, wit1.w_10_2, wit1.w_10_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_81.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_82.bridge {backend x1 tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_82 backend x1 tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_82 backend x1 params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_82
  exact h


theorem Stage_preprocess_1_1.sequential_generatedRoot_95.bridge {backend x1 tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_generatedRoot_95 backend x1 tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.sequential_generatedRoot_95 backend x1 params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_generatedRoot_95
  exact h


theorem Stage_preprocess_1_1.generatedRoot.constraints_3.bridge {backend hashModel tape path params params' w_0_0 w_1_0 w_6_0 w_67_0 w_68_0 w_73_0 w_74_0 w_77_0 w_78_0 w_2_0 w_3_0 w_3_1 w_3_2 w_3_3 w_4_0 w_4_1 w_4_2 w_4_3 w_33_2 w_33_3 w_66_0 w_66_1 w_66_2 w_66_3 w_69_0 w_70_0 w_71_0 w_72_0 w_75_0 w_75_1 w_75_2 w_75_3 w_75_4 w_75_5 w_75_6 w_75_7 w_76_0 w_76_1 w_76_2 w_76_3 w_79_0 w_79_1 w_79_2 w_79_3 w_79_4 w_79_5 w_79_6 w_79_7 w_80_0 w_80_1 w_80_2 w_80_3 w_81_0 w_81_1 w_81_2 w_81_3 w_82_0 w_82_1 w_82_2 w_82_3 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_95_1 w_95_2 w_95_3 outputs}
    (h : Stage_preprocess_1_1.generatedRoot.constraints_3 backend hashModel tape path params w_0_0 w_1_0 w_6_0 w_67_0 w_68_0 w_73_0 w_74_0 w_77_0 w_78_0 w_2_0 w_3_0 w_3_1 w_3_2 w_3_3 w_4_0 w_4_1 w_4_2 w_4_3 w_33_2 w_33_3 w_66_0 w_66_1 w_66_2 w_66_3 w_69_0 w_70_0 w_71_0 w_72_0 w_75_0 w_75_1 w_75_2 w_75_3 w_75_4 w_75_5 w_75_6 w_75_7 w_76_0 w_76_1 w_76_2 w_76_3 w_79_0 w_79_1 w_79_2 w_79_3 w_79_4 w_79_5 w_79_6 w_79_7 w_80_0 w_80_1 w_80_2 w_80_3 w_81_0 w_81_1 w_81_2 w_81_3 w_82_0 w_82_1 w_82_2 w_82_3 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_95_1 w_95_2 w_95_3 outputs) : W_Stage_preprocess_1_1.generatedRoot.constraints_3 backend hashModel params' w_0_0 w_1_0 w_6_0 w_67_0 w_68_0 w_73_0 w_74_0 w_77_0 w_78_0 w_2_0 w_3_0 w_3_1 w_3_2 w_3_3 w_4_0 w_4_1 w_4_2 w_4_3 w_33_2 w_33_3 w_66_0 w_66_1 w_66_2 w_66_3 w_69_0 w_70_0 w_71_0 w_72_0 w_75_0 w_75_1 w_75_2 w_75_3 w_75_4 w_75_5 w_75_6 w_75_7 w_76_0 w_76_1 w_76_2 w_76_3 w_79_0 w_79_1 w_79_2 w_79_3 w_79_4 w_79_5 w_79_6 w_79_7 w_80_0 w_80_1 w_80_2 w_80_3 w_81_0 w_81_1 w_81_2 w_81_3 w_82_0 w_82_1 w_82_2 w_82_3 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_95_1 w_95_2 w_95_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_75.bridge (h2 i))), h3, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_76.bridge (h4 i))), h5, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_79.bridge (h6 i))), h7, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_80.bridge (h8 i))), h9, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_81.bridge (h10 i))), h11, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_82.bridge (h12 i))), h13, h14, h15, h16, h17, h18, h19, h20, h21, (Stage_preprocess_1_1.iterRuns_mono h22 (fun _ _ _ hs24 ↦ (Stage_preprocess_1_1.sequential_generatedRoot_95.bridge hs24))), h23⟩)


theorem Stage_preprocess_1_1.generatedRoot.constraints_2.bridge {backend hashModel tape path params params' w_6_0 w_3_0 w_3_1 w_3_2 w_3_3 w_5_0 w_33_1 w_33_3 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_59_0 w_60_0 w_61_0 w_63_0 w_64_0 w_65_0 w_66_0 w_66_1 w_66_2 w_66_3 w_69_0 w_70_0 w_71_0 w_72_0}
    (h : Stage_preprocess_1_1.generatedRoot.constraints_2 backend hashModel tape path params w_6_0 w_3_0 w_3_1 w_3_2 w_3_3 w_5_0 w_33_1 w_33_3 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_59_0 w_60_0 w_61_0 w_63_0 w_64_0 w_65_0 w_66_0 w_66_1 w_66_2 w_66_3 w_69_0 w_70_0 w_71_0 w_72_0) : W_Stage_preprocess_1_1.generatedRoot.constraints_2 backend hashModel params' w_6_0 w_3_0 w_3_1 w_3_2 w_3_3 w_5_0 w_33_1 w_33_3 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_59_0 w_60_0 w_61_0 w_63_0 w_64_0 w_65_0 w_66_0 w_66_1 w_66_2 w_66_3 w_69_0 w_70_0 w_71_0 w_72_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, (Stage_preprocess_1_1.iterRuns_mono h28 (fun _ _ _ hs33 ↦ (Stage_preprocess_1_1.sequential_generatedRoot_66.bridge hs33))), h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.generatedRoot.constraints_0.bridge {backend hashModel tape path params params' w_0_0 w_1_0 w_6_0 w_31_0 w_32_0 w_2_0 w_3_0 w_3_1 w_3_2 w_3_3 w_4_0 w_4_1 w_4_2 w_4_3 w_7_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_17_0 w_19_0 w_20_0 w_22_0 w_23_0 w_24_0 w_25_0 w_27_0 w_28_0 w_29_0 w_30_0 w_33_0 w_33_1 w_33_2 w_33_3 w_34_0}
    (h : Stage_preprocess_1_1.generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_1_0 w_6_0 w_31_0 w_32_0 w_2_0 w_3_0 w_3_1 w_3_2 w_3_3 w_4_0 w_4_1 w_4_2 w_4_3 w_7_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_17_0 w_19_0 w_20_0 w_22_0 w_23_0 w_24_0 w_25_0 w_27_0 w_28_0 w_29_0 w_30_0 w_33_0 w_33_1 w_33_2 w_33_3 w_34_0) : W_Stage_preprocess_1_1.generatedRoot.constraints_0 backend hashModel params' w_0_0 w_1_0 w_6_0 w_31_0 w_32_0 w_2_0 w_3_0 w_3_1 w_3_2 w_3_3 w_4_0 w_4_1 w_4_2 w_4_3 w_7_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_17_0 w_19_0 w_20_0 w_22_0 w_23_0 w_24_0 w_25_0 w_27_0 w_28_0 w_29_0 w_30_0 w_33_0 w_33_1 w_33_2 w_33_3 w_34_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_3.bridge (h2 i))), h3, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_4.bridge (h4 i))), h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_33.bridge (h31 i))), h32⟩)


theorem Stage_preprocess_1_1.generatedRoot.bridge {backend hashModel tape path params params' inputs outputs}
    (h : Stage_preprocess_1_1.generatedRoot backend hashModel tape path params inputs outputs) : W_Stage_preprocess_1_1.generatedRoot backend hashModel params' inputs outputs := by
  delta W_Stage_preprocess_1_1.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_3_0, wit1.w_3_1, wit1.w_3_2, wit1.w_3_3, wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_7_0, wit1.w_10_0, wit1.w_13_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_20_0, wit1.w_23_0, wit1.w_24_0, wit1.w_25_0, wit1.w_28_0, wit1.w_29_0, wit1.w_30_0, wit1.w_33_0, wit1.w_33_1, wit1.w_33_2, wit1.w_33_3, wit1.w_34_0, wit1.w_35_0, wit1.w_36_0, wit1.w_38_0, wit1.w_39_0, wit1.w_41_0, wit1.w_42_0, wit1.w_45_0, wit1.w_46_0, wit1.w_47_0, wit1.w_48_0, wit1.w_49_0, wit1.w_50_0, wit1.w_53_0, wit1.w_54_0, wit1.w_55_0, wit1.w_56_0, wit1.w_57_0, wit1.w_59_0, wit1.w_60_0, wit1.w_61_0, wit1.w_64_0, wit1.w_65_0, wit1.w_66_0, wit1.w_66_1, wit1.w_66_2, wit1.w_66_3, wit1.w_69_0, wit1.w_70_0, wit1.w_71_0, wit1.w_72_0, wit1.w_75_0, wit1.w_75_1, wit1.w_75_2, wit1.w_75_3, wit1.w_75_4, wit1.w_75_5, wit1.w_75_6, wit1.w_75_7, wit1.w_76_0, wit1.w_76_1, wit1.w_76_2, wit1.w_76_3, wit1.w_79_0, wit1.w_79_1, wit1.w_79_2, wit1.w_79_3, wit1.w_79_4, wit1.w_79_5, wit1.w_79_6, wit1.w_79_7, wit1.w_80_0, wit1.w_80_1, wit1.w_80_2, wit1.w_80_3, wit1.w_81_0, wit1.w_81_1, wit1.w_81_2, wit1.w_81_3, wit1.w_82_0, wit1.w_82_1, wit1.w_82_2, wit1.w_82_3, wit1.w_83_0, wit1.w_84_0, wit1.w_86_0, wit1.w_87_0, wit1.w_89_0, wit1.w_90_0, wit1.w_92_0, wit1.w_93_0, wit1.w_95_0, wit1.w_95_1, wit1.w_95_2, wit1.w_95_3⟩, (by obtain ⟨h3, h4, h5, h6⟩ := h2; exact ⟨(Stage_preprocess_1_1.generatedRoot.constraints_0.bridge h3), h4, (Stage_preprocess_1_1.generatedRoot.constraints_2.bridge h5), (Stage_preprocess_1_1.generatedRoot.constraints_3.bridge h6)⟩)⟩)
